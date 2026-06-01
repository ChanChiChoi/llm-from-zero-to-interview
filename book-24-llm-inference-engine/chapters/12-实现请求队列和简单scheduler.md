# 第 12 章 实现请求队列和简单 scheduler

前面几章已经实现了单请求生成、采样、KV Cache、batched prefill 和 batched decode。但这些还只是固定 batch。真正的 serving engine 面对的是不断到来的请求流：有的请求在等待，有的请求在 prefill，有的请求在 decode，有的请求已经结束。

本章实现最小请求队列和简单 scheduler，让系统从“手动组织 batch”升级为“自动选择请求执行”。

一句话概括：

> Scheduler 的核心职责是每一轮决定哪些请求进入 prefill，哪些请求继续 decode，并在吞吐、延迟、显存和公平性之间做取舍。

## 12.1 为什么需要 scheduler

没有 scheduler 时，我们的流程是人工的：

```text
准备一组 prompts -> batched prefill -> batched decode 到结束
```

真实服务里，请求是动态到达的：

```text
t=0: request_1 到达
t=1: request_2 到达
t=2: request_3 到达，同时 request_1 正在 decode
t=3: request_4 到达，request_2 prefill 很长
```

如果每次都等当前 batch 全部完成再处理下一批，新请求的 TTFT 会很差，GPU 也可能出现空洞。

Scheduler 解决的是：

1. 哪些 waiting 请求可以进入执行。
2. 哪些 running 请求继续 decode。
3. 一轮最多处理多少 token。
4. 一轮最多处理多少序列。
5. 资源不够时让谁等待。
6. 请求结束后如何释放 slot。

## 12.2 最小状态集合

我们先维护三个集合：

```text
waiting_queue: 等待 prefill 的请求
running_requests: 已完成 prefill、正在 decode 的请求
finished_requests: 已结束的请求
```

请求生命周期：

```text
WAITING -> PREFILLING -> DECODING -> FINISHED
```

对应到代码：

```python
from collections import deque


waiting_queue = deque()
running_requests = []
finished_requests = []
```

教学版先不处理失败、取消、超时，后续再补。

## 12.3 RequestState 扩展

为了调度，需要给 request 增加状态和时间字段。

```python
class RequestState:
    def __init__(self, request_id, prompt, max_new_tokens=128, arrival_time=0):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.arrival_time = arrival_time

        self.status = "WAITING"
        self.input_ids = None
        self.output_ids = []
        self.next_token_id = None
        self.finished = False
        self.finish_reason = None

        self.prefill_start_time = None
        self.first_token_time = None
        self.finished_time = None
```

这些时间字段后面可以计算：

1. queue waiting time。
2. TTFT。
3. total latency。

最小 scheduler 不一定马上使用它们，但设计时应该预留。

## 12.4 RequestQueue

先封装一个简单 FIFO 队列。

```python
class RequestQueue:
    def __init__(self):
        self.queue = deque()

    def push(self, request):
        request.status = "WAITING"
        self.queue.append(request)

    def pop_many(self, max_count):
        selected = []
        while self.queue and len(selected) < max_count:
            selected.append(self.queue.popleft())
        return selected

    def __len__(self):
        return len(self.queue)
```

FIFO 是最简单的公平策略。先来的请求先进入 prefill。

真实系统可能会加入优先级、租户配额、deadline、prompt 长度分桶等策略。

## 12.5 Scheduler 的输入和输出

最小 scheduler 每轮接收：

1. waiting queue。
2. running requests。
3. 最大 prefill 请求数。
4. 最大 running 请求数。

输出：

1. 本轮要 prefill 的请求。
2. 本轮要 decode 的请求。

```python
class SimpleScheduler:
    def __init__(self, max_prefill_requests=4, max_running_requests=16):
        self.max_prefill_requests = max_prefill_requests
        self.max_running_requests = max_running_requests

    def schedule(self, waiting_queue, running_requests):
        available_slots = self.max_running_requests - len(running_requests)
        prefill_count = min(self.max_prefill_requests, max(0, available_slots))
        prefill_requests = waiting_queue.pop_many(prefill_count)

        decode_requests = [r for r in running_requests if not r.finished]
        return prefill_requests, decode_requests
```

这个策略非常简单：只要 running 还有空位，就从 waiting 中取请求做 prefill；running 中未完成请求继续 decode。

## 12.6 为什么要限制 max_running_requests

`max_running_requests` 是最小版本里的资源保护。

如果无限制地把 waiting 请求都拉进来，会出现：

1. KV Cache 显存爆掉。
2. decode batch 太大。
3. 单轮延迟变长。
4. 已有请求输出抖动。

真实系统里限制不只是请求数，还会包括：

1. 最大 token budget。
2. 最大 KV Cache block 数。
3. 最大 prefill token 数。
4. 最大 decode 序列数。
5. 每租户配额。

教学版先用请求数近似资源预算。

## 12.7 Engine 主循环

现在可以写最小 engine loop。

```python
class MiniEngine:
    def __init__(self, tokenizer_wrapper, model_wrapper, sampler, scheduler):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.model_wrapper = model_wrapper
        self.sampler = sampler
        self.scheduler = scheduler
        self.waiting_queue = RequestQueue()
        self.running_requests = []
        self.finished_requests = []

    def add_request(self, request):
        self.waiting_queue.push(request)

    def step(self):
        prefill_requests, decode_requests = self.scheduler.schedule(
            self.waiting_queue,
            self.running_requests,
        )

        if prefill_requests:
            self.run_prefill(prefill_requests)

        if decode_requests:
            self.run_decode(decode_requests)

        self.cleanup_finished()
```

`step()` 表示 engine 的一次调度迭代。

真实 serving engine 基本也是类似结构，只是每个函数内部复杂得多。

## 12.8 run_prefill

`run_prefill` 做三件事：

1. 对 selected requests 做 batched prefill。
2. 把首 token 写回请求。
3. 把请求加入 running_requests。

```python
def run_prefill(self, requests):
    prompts = [request.prompt for request in requests]
    result = batched_prefill(
        self.tokenizer_wrapper,
        self.model_wrapper,
        self.sampler,
        prompts,
    )

    next_token_ids = result["next_token_ids"]

    for i, request in enumerate(requests):
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        request.status = "DECODING"
        request.first_token_time = now()
        self.running_requests.append(request)
```

这里隐藏了一个大问题：每批 prefill 都会返回自己的 `past_key_values`。如果要和已有 running requests 合并 decode，就需要统一管理 cache。

教学版可以先把每批 prefill 作为一个 decode group。更真实的版本需要 KV Cache Manager。

## 12.9 decode group 的概念

为了避免马上实现复杂 cache manager，可以引入 decode group。

一个 decode group 是一组一起 prefill、一起 decode 的请求，以及对应的 batched KV Cache。

```python
class DecodeGroup:
    def __init__(self, requests, past_key_values):
        self.requests = requests
        self.past_key_values = past_key_values
```

MiniEngine 维护：

```python
self.decode_groups = []
```

每次 prefill 新请求，就创建一个新的 decode group。

这样可以避免不同 prefill batch 的 cache 合并问题。

缺点是：不同 group 不能合并 decode，吞吐不如真正 continuous batching。

但作为教学版，它清楚展示了 scheduler、queue 和 batch 的关系。

## 12.10 run_prefill with decode group

```python
def run_prefill(self, requests):
    prompts = [request.prompt for request in requests]
    result = batched_prefill(
        self.tokenizer_wrapper,
        self.model_wrapper,
        self.sampler,
        prompts,
    )

    next_token_ids = result["next_token_ids"]
    past_key_values = result["past_key_values"]

    for i, request in enumerate(requests):
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        request.status = "DECODING"
        request.first_token_time = now()

    self.decode_groups.append(DecodeGroup(requests, past_key_values))
    self.running_requests.extend(requests)
```

后续 decode 时，对每个 group 分别做 batched decode step。

## 12.11 run_decode

```python
def run_decode(self):
    for group in self.decode_groups:
        active_requests = [r for r in group.requests if not r.finished]
        if not active_requests:
            continue

        last_token_ids = torch.tensor(
            [[r.next_token_id] for r in active_requests],
            device=self.model_wrapper.device,
        )

        next_token_ids, group.past_key_values = batched_decode_step(
            self.model_wrapper,
            self.sampler,
            last_token_ids,
            group.past_key_values,
        )

        for i, request in enumerate(active_requests):
            token_id = next_token_ids[i].item()
            request.append_token(token_id)
            request.should_stop(self.tokenizer_wrapper.eos_token_id)
```

这段代码表达了调度循环，但严格来说还缺少 active_requests 对 `past_key_values` 的选择和回写。如果 group 里有请求 finished 后被跳过，cache batch 维度就可能对不上。

教学版有两种处理方式：

1. 固定 group batch，不移除 finished 请求，只是不更新它们输出。
2. 实现 active index 选择和 cache compact。

为了简单，先用固定 group batch 更安全。

## 12.12 固定 group 的 decode

固定 group 意味着：group 中的所有请求一直参与 decode forward，但 finished 请求的输出不再更新。

```python
def run_decode_group(self, group):
    last_token_ids = torch.tensor(
        [[r.next_token_id] for r in group.requests],
        device=self.model_wrapper.device,
    )

    next_token_ids, group.past_key_values = batched_decode_step(
        self.model_wrapper,
        self.sampler,
        last_token_ids,
        group.past_key_values,
    )

    for i, request in enumerate(group.requests):
        if request.finished:
            continue
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        request.should_stop(self.tokenizer_wrapper.eos_token_id)
```

这种方式浪费计算，但避免 cache batch 顺序错乱。教学阶段可以接受。

真实系统不会长期这样做，因为 finished 请求继续占用计算和 cache 会拖慢吞吐。

## 12.13 cleanup_finished

每轮 step 后，要清理已完成请求。

```python
def cleanup_finished(self):
    still_running = []
    for request in self.running_requests:
        if request.finished:
            request.status = "FINISHED"
            request.finished_time = now()
            self.finished_requests.append(request)
        else:
            still_running.append(request)
    self.running_requests = still_running

    self.decode_groups = [
        group for group in self.decode_groups
        if any(not r.finished for r in group.requests)
    ]
```

这里仍然没有释放 KV Cache block，因为我们还没有实现真正 cache manager。

在真实系统中，cleanup 必须释放 cache，否则显存会泄漏。

## 12.14 最小调度循环示例

一个简化运行方式：

```python
engine = MiniEngine(tokenizer_wrapper, model_wrapper, sampler, scheduler)

engine.add_request(RequestState("r1", "Explain KV Cache", max_new_tokens=64))
engine.add_request(RequestState("r2", "Write a haiku", max_new_tokens=32))

while engine.waiting_queue or engine.running_requests:
    engine.step()

for request in engine.finished_requests:
    print(request.request_id, request.output_ids, request.finish_reason)
```

这个版本已经有了 serving engine 的基本形状：

1. 请求进入队列。
2. scheduler 选择请求。
3. prefill 建立 cache。
4. decode 逐步生成。
5. 请求结束后清理。

## 12.15 简单 scheduler 的局限

本章 scheduler 很简单，因此有明显局限：

1. 只按请求数限制资源，不按 token 数限制。
2. 不估算 KV Cache 显存。
3. 不支持优先级。
4. 不支持抢占。
5. 不支持 chunked prefill。
6. 不支持新请求加入已有 decode batch。
7. 不支持不同请求独立 sampling params。
8. 不支持取消和超时。

但它已经让系统从固定 batch 变成请求流调度，这是 serving engine 的关键一步。

## 12.16 从简单 scheduler 到 continuous batching

continuous batching 的核心升级是：每一轮都重新组织 batch。

```text
iteration 1: prefill r1, r2
iteration 2: decode r1, r2; prefill r3
iteration 3: decode r1, r2, r3; remove finished r1
iteration 4: decode r2, r3; prefill r4
```

要实现它，需要：

1. 统一 KV Cache Manager。
2. 能把不同请求映射到 cache block。
3. 每轮动态选择 prefill 和 decode。
4. 支持 finished 请求释放资源。
5. 支持 waiting 请求加入 running set。

本章的 waiting queue、running set 和 scheduler，就是 continuous batching 的骨架。

## 12.17 调度策略的 trade-off

调度策略会直接影响用户体验和成本。

优先 prefill：

1. 新请求 TTFT 更好。
2. 可能挤占 decode，导致已有请求输出不平滑。

优先 decode：

1. 已有请求 TPOT 更稳定。
2. 新请求等待更久，TTFT 变差。

限制 batch 小：

1. 单请求延迟可能更好。
2. GPU 利用率可能较低。

限制 batch 大：

1. 吞吐更高。
2. 排队和长尾延迟可能变差。

所以 scheduler 是 serving engine 的核心，而不是简单 FIFO 队列。

## 12.18 常见误区

误区一：scheduler 就是队列 pop。

队列只是输入，scheduler 还要考虑 running 请求、资源预算、prefill/decode 权衡和完成清理。

误区二：只要 batch 越大越好。

batch 大会增加显存和延迟，也可能让短请求被长请求拖慢。

误区三：finished 请求从列表删掉就够了。

真实系统还必须释放 KV Cache、关闭 stream、记录指标。

误区四：按请求数限制资源足够。

LLM serving 更应该按 token、KV Cache block 和显存预算限制。

误区五：简单 scheduler 和 vLLM scheduler 只差代码量。

真实 scheduler 要处理 continuous batching、PagedAttention、抢占、prefix cache、chunked prefill 和多 worker。

## 12.19 面试追问

1. 为什么 LLM serving 需要 scheduler？
2. waiting queue、running set、finished set 分别是什么？
3. scheduler 每轮要做哪些决策？
4. 为什么只按请求数限制资源不够？
5. 优先 prefill 和优先 decode 各有什么影响？
6. 简单 batched decode 和 continuous batching 的区别是什么？
7. finished 请求清理时要释放哪些资源？
8. 如果 TTFT 高但 TPOT 正常，scheduler 可能有什么问题？

参考回答思路：

1. 先说请求动态到达，不能固定 batch，因此需要 scheduler。
2. 再说 scheduler 管 waiting、running、finished，决定 prefill 和 decode。
3. 然后解释资源预算：请求数只是近似，真实要看 token 和 KV Cache。
4. 最后讲 trade-off：prefill 影响 TTFT，decode 影响 TPOT，scheduler 在两者之间平衡。

## 12.20 小练习

1. 实现一个 FIFO RequestQueue。
2. 给 SimpleScheduler 增加 `max_prefill_tokens` 限制。
3. 给请求增加 priority 字段，实现高优先级请求优先进入 prefill。
4. 记录每个请求的 queue waiting time 和 TTFT。
5. 思考如何把新请求加入已有 running batch，而不是创建新的 decode group。

## 12.21 本章小结

本章实现了最小请求队列和简单 scheduler。

我们引入了 waiting queue、running requests、finished requests、RequestQueue、SimpleScheduler 和 MiniEngine 主循环。这个版本还很简化，但已经体现了 serving engine 的核心：请求不是手动组成固定 batch，而是不断进入队列，由 scheduler 每轮决定谁 prefill、谁 decode、谁结束。

下一章我们会实现 token streaming，让生成结果不再等完整答案结束后才返回，而是每生成一个 token 就增量输出。
