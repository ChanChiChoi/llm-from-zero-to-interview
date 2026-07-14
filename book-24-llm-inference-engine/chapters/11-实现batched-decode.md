# 第 11 章 实现 batched decode

上一章我们实现了 batched prefill：多个请求的 prompt 合成一个 batch，一次 forward 得到首 token 和初始 KV Cache。本章继续实现 batched decode：多个正在生成的请求，每轮一起输入上一个 token，一起生成下一个 token。

Batched decode 是 continuous batching 的前置能力。先理解固定 batch 的 decode，才能理解后续“请求动态加入、动态退出”的调度系统。

一句话概括：

> Batched decode 的目标是让多个请求共享一次 decode forward，每个请求每轮生成一个新 token，并维护各自的状态、cache 顺序和停止条件。

## 11.0 本讲资料边界与第二轮精修口径

本章第二轮精修前，先用公开资料校准口径：Transformers cache 文档强调自回归 decode 每步只处理新 token，但 attention mask 需要覆盖 past KV length 加 current tokens；cache 通常按 layer 保存 key/value，形状类似 `[batch_size, num_heads, seq_len, head_dim]`，并且 decode 时会把新 key/value 追加到已有 cache。vLLM 文档把更生产级的问题进一步放到 scheduler、KV cache 空间、decode 优先、chunked prefill 和 `max_num_batched_tokens` 等调度控制上。

因此，本章只实现教学版 batched decode：固定 batch、逐请求 finished mask、attention mask 长度增长、batch row 到 request/cache slot 的映射，以及 compact active batch 的直觉。它不实现真正 continuous batching、PagedAttention block table、动态 admission、per-request logits processor 或跨 worker KV 迁移。新增 0 依赖 demo 用 toy requests 验证固定 batch 的浪费、compact batch 的收益、逐请求停止条件和 cache alignment gate。

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

这里的 batch 维度不是普通离线批处理那么简单。第 `i` 行 token、attention mask、position、KV cache slot 和 request state 必须严格对应。

## 11.2 固定 batch 的前提

本章先实现固定 batch decode。也就是说：

1. batch 中的请求一开始已经完成 prefill。
2. batch 顺序固定。
3. 每轮只对未完成请求追加输出。
4. 已完成请求可以用 mask 跳过状态更新。
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
        self.prompt_len = 0
        self.input_ids = None
        self.output_ids = []
        self.next_token_id = None
        self.finished = False
        self.finish_reason = None

    def append_token(self, token_id):
        self.output_ids.append(int(token_id))
        self.next_token_id = int(token_id)

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

最重要的是：每个请求都有自己的 `output_ids`、`finished`、`finish_reason`、`prompt_len` 和 `max_new_tokens`。

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
def apply_prefill_result(requests, prefill_result, eos_token_id):
    input_ids = prefill_result["input_ids"]
    attention_mask = prefill_result["attention_mask"]
    next_token_ids = prefill_result["next_token_ids"]

    for i, request in enumerate(requests):
        request.input_ids = input_ids[i]
        request.prompt_len = int(attention_mask[i].sum().item())
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        request.should_stop(eos_token_id)
```

注意：prefill 已经生成了第一个输出 token。decode loop 应该从这些首 token 继续往后生成。

## 11.5 最小 batched_decode_step

一轮 batched decode 可以写成：

```python
def batched_decode_step(model_wrapper, sampler, last_token_ids, past_key_values, attention_mask=None, position_ids=None):
    logits, past_key_values = model_wrapper.forward(
        input_ids=last_token_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
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

这和单请求 decode 最大区别只是 batch 维度从 1 变成 N，但系统层复杂度高很多。

## 11.6 attention_mask 和 position_ids 如何增长

在 decode 阶段，每轮只输入一个 token，但模型仍然需要知道它能看见多长历史。

如果当前 cache 中已经有 `T_cache` 个 token，本轮输入 `T_new` 个 token，那么 attention mask 长度应该满足：

```math
T_{\mathrm{mask}}=T_{\mathrm{cache}}+T_{\mathrm{new}}
```

单 token decode 时，`T_new=1`。

对 batch 中不同长度请求：

```text
request_1 当前长度 10
request_2 当前长度 100
request_3 当前长度 30
```

它们下一 token 的 position 不一样。常见 position ids 直觉类似：

```python
position_ids = torch.tensor([[10], [100], [30]], device=device)
```

不同模型对 position 推断和 cache 接口的细节不同。教学版可以依赖模型自动处理，但生产实现不能忽略这个问题。

## 11.7 固定 batch decode loop

最小 decode loop：

```python
def run_batched_decode(requests, model_wrapper, sampler, past_key_values, eos_token_id):
    last_token_ids = torch.tensor(
        [[request.next_token_id] for request in requests],
        device=model_wrapper.device,
    )

    while not all(request.finished for request in requests):
        active_mask = torch.tensor(
            [not request.finished for request in requests],
            device=model_wrapper.device,
        )

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

## 11.8 finished mask

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

设第 `t` 轮 active 请求数为 `A_t`，原始 batch size 为 `B`。固定 batch 每轮处理 `B` 行：

```math
W_{\mathrm{fixed}}=BT
```

compact active batch 只处理活跃行：

```math
W_{\mathrm{compact}}=\sum_{t=1}^{T} A_t
```

浪费行数可以粗略写成：

```math
W_{\mathrm{waste}}=W_{\mathrm{fixed}}-W_{\mathrm{compact}}
```

真实 GPU kernel 的效率不只由行数决定，但这个公式能帮助理解 finished 请求留在 batch 里的直接代价。

## 11.9 compact batch

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

## 11.10 batch 顺序为什么重要

在 batched decode 中，第 `i` 行 token 必须对应 `past_key_values` 的第 `i` 行 cache。

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

## 11.13 batched decode 公式、状态和可运行 demo

下面这个 demo 不依赖 PyTorch。它模拟 3 个已经完成 prefill 的请求，每个请求有自己的目标输出、prompt length 和 max_new_tokens。demo 同时计算：

1. 固定 batch 每轮处理所有行的工作量。
2. compact batch 每轮只处理 active 行的工作量。
3. 每个请求的 position ids 和 attention mask length 如何增长。
4. batch row / request id / cache slot 是否保持一致。

```python
EOS = "<eos>"


class RequestState:
    def __init__(self, request_id, prompt_len, target_tokens, max_new_tokens):
        self.request_id = request_id
        self.prompt_len = prompt_len
        self.target_tokens = target_tokens
        self.max_new_tokens = max_new_tokens
        self.output_tokens = [target_tokens[0]]  # first token from prefill
        self.next_token = target_tokens[0]
        self.finished = False
        self.finish_reason = None
        self.cache_len = prompt_len + 1
        self.trace = []
        self.should_stop()

    def should_stop(self):
        if self.finished:
            return True
        if self.output_tokens[-1] == EOS:
            self.finished = True
            self.finish_reason = "stop"
            return True
        if len(self.output_tokens) >= self.max_new_tokens:
            self.finished = True
            self.finish_reason = "length"
            return True
        return False

    def decode_next(self):
        if self.finished:
            return self.next_token
        idx = len(self.output_tokens)
        token = self.target_tokens[idx] if idx < len(self.target_tokens) else EOS
        position_id = self.prompt_len + len(self.output_tokens) - 1
        mask_len = self.prompt_len + len(self.output_tokens)
        self.output_tokens.append(token)
        self.next_token = token
        self.cache_len += 1
        self.should_stop()
        self.trace.append({
            "position_id": position_id,
            "attention_mask_len": mask_len,
            "token": token,
            "finished": self.finished,
        })
        return token


requests = [
    RequestState("r1", prompt_len=2, target_tokens=["A", EOS], max_new_tokens=4),
    RequestState("r2", prompt_len=6, target_tokens=["B", "C", "D", EOS], max_new_tokens=4),
    RequestState("r3", prompt_len=3, target_tokens=["X", "Y", EOS], max_new_tokens=3),
]
cache_slots = {request.request_id: f"slot_{request.request_id}" for request in requests}

fixed_rows = 0
compact_rows = 0
step_records = []

while not all(request.finished for request in requests):
    active_requests = [request for request in requests if not request.finished]
    active_ids = [request.request_id for request in active_requests]
    row_to_cache_slot = [(request.request_id, cache_slots[request.request_id]) for request in active_requests]
    alignment_ok = all(request_id in slot for request_id, slot in row_to_cache_slot)

    fixed_rows += len(requests)
    compact_rows += len(active_requests)

    emitted = {}
    position_ids = {}
    mask_lens = {}
    for request in active_requests:
        token = request.decode_next()
        emitted[request.request_id] = token
        position_ids[request.request_id] = request.trace[-1]["position_id"]
        mask_lens[request.request_id] = request.trace[-1]["attention_mask_len"]

    step_records.append({
        "active_ids": active_ids,
        "emitted": emitted,
        "position_ids": position_ids,
        "attention_mask_lens": mask_lens,
        "row_to_cache_slot": row_to_cache_slot,
        "alignment_ok": alignment_ok,
    })

summary = {
    request.request_id: {
        "output": request.output_tokens,
        "finish_reason": request.finish_reason,
        "cache_len": request.cache_len,
    }
    for request in requests
}

print("step_records=", step_records)
print("summary=", summary)
print("fixed_rows=", fixed_rows)
print("compact_rows=", compact_rows)
print("saved_rows=", fixed_rows - compact_rows)
print("batched_decode_gate=", all(record["alignment_ok"] for record in step_records) and all(r.finished for r in requests))
```

一组稳定输出如下：

```text
step_records= [{'active_ids': ['r1', 'r2', 'r3'], 'emitted': {'r1': '<eos>', 'r2': 'C', 'r3': 'Y'}, 'position_ids': {'r1': 2, 'r2': 6, 'r3': 3}, 'attention_mask_lens': {'r1': 3, 'r2': 7, 'r3': 4}, 'row_to_cache_slot': [('r1', 'slot_r1'), ('r2', 'slot_r2'), ('r3', 'slot_r3')], 'alignment_ok': True}, {'active_ids': ['r2', 'r3'], 'emitted': {'r2': 'D', 'r3': '<eos>'}, 'position_ids': {'r2': 7, 'r3': 4}, 'attention_mask_lens': {'r2': 8, 'r3': 5}, 'row_to_cache_slot': [('r2', 'slot_r2'), ('r3', 'slot_r3')], 'alignment_ok': True}, {'active_ids': ['r2'], 'emitted': {'r2': '<eos>'}, 'position_ids': {'r2': 8}, 'attention_mask_lens': {'r2': 9}, 'row_to_cache_slot': [('r2', 'slot_r2')], 'alignment_ok': True}]
summary= {'r1': {'output': ['A', '<eos>'], 'finish_reason': 'stop', 'cache_len': 4}, 'r2': {'output': ['B', 'C', 'D', '<eos>'], 'finish_reason': 'stop', 'cache_len': 10}, 'r3': {'output': ['X', 'Y', '<eos>'], 'finish_reason': 'stop', 'cache_len': 6}}
fixed_rows= 9
compact_rows= 6
saved_rows= 3
batched_decode_gate= True
```

输出怎么读：

1. 第 1 轮三个请求都活跃，第 2 轮只剩 `r2` 和 `r3`，第 3 轮只剩 `r2`。
2. `position_ids` 和 `attention_mask_lens` 按请求自己的长度增长，不是全 batch 共用一个值。
3. `fixed_rows=9` 表示固定 batch 三轮都处理三行；`compact_rows=6` 表示只处理活跃行，节省 3 行 decode 工作。
4. `row_to_cache_slot` 和 `alignment_ok=True` 表示 batch row 和 cache slot 没有串台。

## 11.14 固定 batch decode 的完整骨架

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

## 11.15 更接近 engine 的循环

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

## 11.16 batched decode 和 continuous batching 的区别

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

## 11.17 常见 bug

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

## 11.18 常见误区

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

## 11.19 面试追问

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

## 11.20 小练习

1. 用三个 prompt 做 batched prefill，然后接 batched decode 生成 20 个 token。
2. 给三个请求设置不同 `max_new_tokens`，观察 finished 状态变化。
3. 故意打乱 `next_token_ids` 和 request 的对应关系，观察输出问题。
4. 实现一个 `active_indices` 版本，只对未完成请求做 decode。
5. 思考如何把新请求在 decode 过程中加入 batch。
6. 统计固定 batch 和 compact batch 的 decode row 数差异。

## 11.21 本章小结

本章实现了最小 batched decode。

Batched decode 让多个请求每轮一起生成一个 token。它的关键不只是模型 forward，还包括 request 状态、batch 顺序、KV Cache 对齐、finished mask、逐请求停止条件、position_ids、attention mask 增长和输出回写。固定 batch decode 是 continuous batching 的基础，但还不能动态加入新请求。

下一章我们会引入请求队列和简单 scheduler，让系统开始具备真正的多请求调度能力。
