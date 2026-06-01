# 第 11 章 实现 batched decode

上一章我们实现了 batched prefill：多个请求的 prompt 合成一个 batch，一次 forward 得到首 token 和初始 KV Cache。本章继续实现 batched decode：多个正在生成的请求，每轮一起输入上一个 token，一起生成下一个 token。

Batched decode 是 continuous batching 的前置能力。先理解固定 batch 的 decode，才能理解后续“请求动态加入、动态退出”的调度系统。

一句话概括：

> Batched decode 的目标是让多个请求共享一次 decode forward，每个请求每轮生成一个新 token，并维护各自的状态和停止条件。

## 11.1 从单请求 decode 到 batched decode

单请求 decode 是：

```text
next_token + past_key_values -> model forward -> logits -> sample -> new_token
```

多个请求时，朴素做法是逐个 decode：

```text
request_1 decode
request_2 decode
request_3 decode
```

Batched decode 把它们合成：

```text
[token_1, token_2, token_3] + batched KV Cache -> model forward -> [new_1, new_2, new_3]
```

每轮 decode 中，每个还活跃的请求通常只贡献一个 token。因此输入形状一般是：

```text
input_ids: [batch_size, 1]
```

输出 logits 形状是：

```text
logits: [batch_size, 1, vocab_size]
```

## 11.2 固定 batch 的前提

本章先实现固定 batch decode。也就是说：

1. batch 中的请求一开始已经完成 prefill。
2. batch 顺序固定。
3. 每轮只对未完成请求生成。
4. 已完成请求可以用 mask 跳过或从 batch 中移除。
5. 暂时不加入新请求。

真实 continuous batching 会更复杂：每一轮都可能有新请求加入，也可能有旧请求退出。固定 batch 是理解它的第一步。

## 11.3 RequestState 需要哪些字段

为了维护 decode 状态，我们扩展 `RequestState`。

```python
class RequestState:
    def __init__(self, request_id, prompt, max_new_tokens=128):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.input_ids = None
        self.output_ids = []
        self.next_token_id = None
        self.finished = False
        self.finish_reason = None

    def append_token(self, token_id):
        self.output_ids.append(int(token_id))
        self.next_token_id = token_id

    def should_stop(self, eos_token_id):
        if self.finished:
            return True
        if self.output_ids and self.output_ids[-1] == eos_token_id:
            self.finished = True
            self.finish_reason = "stop"
            return True
        if len(self.output_ids) >= self.max_new_tokens:
            self.finished = True
            self.finish_reason = "length"
            return True
        return False
```

最重要的是：每个请求都有自己的 `output_ids`、`finished` 和 `finish_reason`。

Batched decode 的难点不是 forward，而是 batch 中每个请求状态不同。

## 11.4 batched prefill 后的状态

上一章 batched prefill 返回：

```python
{
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "next_token_ids": next_token_ids,
    "past_key_values": past_key_values,
}
```

我们要把首 token 写回每个 request：

```python
def apply_prefill_result(requests, prefill_result):
    input_ids = prefill_result["input_ids"]
    next_token_ids = prefill_result["next_token_ids"]

    for i, request in enumerate(requests):
        request.input_ids = input_ids[i]
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
```

注意：prefill 已经生成了第一个输出 token。decode loop 应该从这些首 token 继续往后生成。

## 11.5 最小 batched_decode_step

一轮 batched decode 可以写成：

```python
def batched_decode_step(model_wrapper, sampler, last_token_ids, past_key_values):
    logits, past_key_values = model_wrapper.forward(
        input_ids=last_token_ids,
        attention_mask=None,
        past_key_values=past_key_values,
    )
    next_token_ids = sampler.sample(logits)
    return next_token_ids, past_key_values
```

其中：

```text
last_token_ids shape: [batch_size, 1]
next_token_ids shape: [batch_size, 1]
```

这和单请求 decode 最大区别只是 batch 维度从 1 变成 N。

## 11.6 固定 batch decode loop

最小 decode loop：

```python
def run_batched_decode(requests, model_wrapper, sampler, past_key_values, eos_token_id):
    last_token_ids = torch.tensor(
        [[request.next_token_id] for request in requests],
        device=model_wrapper.device,
    )

    while not all(request.finished for request in requests):
        next_token_ids, past_key_values = batched_decode_step(
            model_wrapper=model_wrapper,
            sampler=sampler,
            last_token_ids=last_token_ids,
            past_key_values=past_key_values,
        )

        for i, request in enumerate(requests):
            if request.finished:
                continue
            token_id = next_token_ids[i].item()
            request.append_token(token_id)
            request.should_stop(eos_token_id)

        last_token_ids = next_token_ids

    return requests
```

这段代码能说明核心流程，但有一个问题：已经 finished 的请求仍然留在 batch 里参与 forward。

这会浪费计算，也可能让状态管理变复杂。

## 11.7 finished mask

可以维护一个 `active_mask`：

```text
request_1 active
request_2 finished
request_3 active
```

教学版最简单做法是：仍然保持固定 batch，但只更新 active 请求的输出。

```python
active_mask = torch.tensor(
    [not request.finished for request in requests],
    device=model_wrapper.device,
)
```

这种做法简单，但 finished 请求仍然消耗 decode 计算。

更高效做法是每轮把 finished 请求移出 batch，也就是 compact batch。

## 11.8 compact batch

compact batch 的思路是：每轮只把未完成请求放进 batch。

```python
active_requests = [request for request in requests if not request.finished]
```

然后构造：

```python
last_token_ids = torch.tensor(
    [[request.next_token_id] for request in active_requests],
    device=model_wrapper.device,
)
```

问题是：`past_key_values` 也必须按同样的 active indices 做选择。

教学版可以先讲概念，真实代码要实现 cache gather：

```python
def select_past_key_values(past_key_values, active_indices):
    selected = []
    for layer_past in past_key_values:
        key, value = layer_past
        selected.append((key[active_indices], value[active_indices]))
    return tuple(selected)
```

这说明 batch 顺序和 cache 顺序必须保持一致。

## 11.9 batch 顺序为什么重要

在 batched decode 中，第 i 行 token 必须对应 past_key_values 的第 i 行 cache。

如果顺序错了，会出现严重问题：

```text
request_1 的 token 用了 request_2 的 KV Cache
request_2 的 token 用了 request_1 的 KV Cache
```

输出会立刻变乱，而且很难 debug。

所以 serving engine 必须维护清晰映射：

```text
batch row -> request id -> KV cache slot
```

真实系统里，这个映射通常由 scheduler 和 KV Cache Manager 共同维护。

## 11.10 attention_mask 和 position_ids

在 decode 阶段，每轮只输入一个 token，但模型仍然需要知道当前位置。

有些模型可以根据 `past_key_values` 自动推断位置，有些模型需要显式 `position_ids`。

对 batch 中不同长度请求：

```text
request_1 当前长度 10
request_2 当前长度 100
request_3 当前长度 30
```

它们下一 token 的 position 不一样。

因此 batched decode 中 position_ids 往往类似：

```python
position_ids = torch.tensor([[10], [100], [30]], device=device)
```

如果 position 错了，模型会误解 token 在序列中的位置，输出可能异常。

教学版可以先依赖模型自动处理，但必须知道生产实现不能忽略这个问题。

## 11.11 每个请求不同 max_new_tokens

Batched decode 中，每个请求可能有不同长度限制。

```text
request_1 max_new_tokens=32
request_2 max_new_tokens=512
request_3 max_new_tokens=128
```

所以停止判断必须逐请求执行，而不是整个 batch 共用一个计数器。

```python
for request in requests:
    request.should_stop(eos_token_id)
```

这也是为什么 `RequestState` 必须保存自己的 `output_ids` 和 `max_new_tokens`。

## 11.12 不同 sampling params

真实 serving 中，不同请求可能有不同 sampling 参数：

```text
request_1 temperature=0
request_2 temperature=0.7, top_p=0.9
request_3 top_k=40
```

教学版可以先让 batch 共用一个 sampler。

生产版 sampler 需要支持 per-request params。实现方式通常是：

1. 对每一行 logits 应用对应参数。
2. 或按参数分组采样。
3. 或实现批量化 logits processor。

这会让 sampler 比第 8 章复杂很多。

## 11.13 固定 batch decode 的完整骨架

下面是一个更完整的教学版骨架：

```python
def run_batched_decode(requests, model_wrapper, sampler, past_key_values, eos_token_id):
    last_token_ids = torch.tensor(
        [[request.next_token_id] for request in requests],
        device=model_wrapper.device,
    )

    while True:
        active = [not request.finished for request in requests]
        if not any(active):
            break

        logits, past_key_values = model_wrapper.forward(
            input_ids=last_token_ids,
            attention_mask=None,
            past_key_values=past_key_values,
        )
        next_token_ids = sampler.sample(logits)

        for i, request in enumerate(requests):
            if request.finished:
                continue
            token_id = next_token_ids[i].item()
            request.append_token(token_id)
            request.should_stop(eos_token_id)

        last_token_ids = next_token_ids

    return requests, past_key_values
```

它能跑通概念，但不是高效实现，因为 finished 请求仍在 batch 中。

## 11.14 更接近 engine 的循环

更接近 engine 的写法是每轮重新构造 active batch：

```python
while running_requests:
    active_indices = [i for i, r in enumerate(requests) if not r.finished]
    active_requests = [requests[i] for i in active_indices]

    last_token_ids = build_last_token_ids(active_requests)
    active_past = select_past_key_values(past_key_values, active_indices)

    next_token_ids, active_past = batched_decode_step(
        model_wrapper, sampler, last_token_ids, active_past
    )

    update_requests(active_requests, next_token_ids)
    update_global_cache(past_key_values, active_indices, active_past)
```

这比固定 batch 复杂，因为要选择和回写 cache。

真实 vLLM 这类系统不会直接搬整个 `past_key_values`，而是通过 block table 和 cache block 管理请求到物理 cache 的映射。

## 11.15 batched decode 和 continuous batching 的区别

Batched decode：

```text
一组请求一起 decode，batch 基本固定
```

Continuous batching：

```text
每一轮 decode 都可以移除完成请求，也可以加入新请求
```

区别在于是否动态 admission。

batched decode 只解决“一起生成”的问题；continuous batching 还解决“每轮谁进入、谁退出、资源够不够”的问题。

下一章的请求队列和简单 scheduler，就是从 batched decode 走向 continuous batching 的关键一步。

## 11.16 常见 bug

Bug 一：batch 行和 request 对不上。

表现是输出混乱，不同请求内容串台。

Bug 二：finished 请求还在追加 token。

表现是已经 EOS 或达到长度限制后继续生成。

Bug 三：cache 选择顺序错。

表现是 decode 输出突然变坏，且难以从报错发现。

Bug 四：position_ids 错。

表现是长文本或不同长度 batch 下质量异常。

Bug 五：所有请求共用一个 max_new_tokens。

表现是短请求被迫生成太长，或长请求过早停止。

## 11.17 常见误区

误区一：batched decode 只是把 tokens 拼成 tensor。

还必须维护 request 状态、cache 顺序、停止条件和输出回写。

误区二：finished 请求留在 batch 里没有代价。

它们仍会消耗计算和 cache，影响吞吐和 TPOT。

误区三：batch 顺序无所谓。

batch row、request id 和 KV Cache 必须严格对应。

误区四：decode 阶段不需要关心 position。

不同请求上下文长度不同，position 处理错误会影响输出正确性。

误区五：实现 batched decode 就等于实现了 vLLM。

vLLM 还需要 continuous batching、PagedAttention、block manager、worker/executor 和复杂调度。

## 11.18 面试追问

1. Batched prefill 和 batched decode 的输入输出有什么区别？
2. Batched decode 中 `input_ids` 的形状通常是什么？
3. 为什么 batch row 和 KV Cache 顺序必须一致？
4. finished 请求应该如何处理？
5. 固定 batch decode 和 continuous batching 有什么区别？
6. batched decode 中 position_ids 为什么重要？
7. 不同请求有不同 sampling params 时 sampler 怎么办？
8. 如果输出串台，你会优先排查哪些地方？

参考回答思路：

1. 先说 prefill 输入完整 prompt，decode 每轮输入每个请求的最新 token。
2. 再说 decode batch 要维护 request 状态、finished mask、cache 顺序和停止条件。
3. 然后解释 finished 请求可以 mask 或移出，移出需要同步选择 cache。
4. 最后说明 continuous batching 在每轮动态加入新请求，是更完整的 serving 调度。

## 11.19 小练习

1. 用三个 prompt 做 batched prefill，然后接 batched decode 生成 20 个 token。
2. 给三个请求设置不同 `max_new_tokens`，观察 finished 状态变化。
3. 故意打乱 `next_token_ids` 和 request 的对应关系，观察输出问题。
4. 实现一个 `active_indices` 版本，只对未完成请求做 decode。
5. 思考如何把新请求在 decode 过程中加入 batch。

## 11.20 本章小结

本章实现了最小 batched decode。

Batched decode 让多个请求每轮一起生成一个 token。它的关键不只是模型 forward，还包括 request 状态、batch 顺序、KV Cache 对齐、finished mask、逐请求停止条件、position_ids 和输出回写。固定 batch decode 是 continuous batching 的基础，但还不能动态加入新请求。

下一章我们会引入请求队列和简单 scheduler，让系统开始具备真正的多请求调度能力。
