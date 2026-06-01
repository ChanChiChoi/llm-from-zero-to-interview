# 第 54 章 把同步 engine loop 升级成异步 serving 架构

第 53 章我们给 mini engine 建立了 benchmark、指标和 trace。

有了这些工具之后，你很快会发现一个问题：

```text
GPU forward 不一定是唯一瓶颈。
```

在教学版 engine 里，我们经常写一个同步主循环：

```python
while engine_has_work():
    new_requests = api_server.pop_new_requests()
    scheduler.add_requests(new_requests)

    schedule = scheduler.schedule()
    batch = batch_builder.build(schedule)
    outputs = model_runner.forward(batch)
    tokens = sampler.sample(outputs)
    output_processor.process(tokens)
```

这个结构适合学习。

它把请求进入、调度、GPU 执行、采样、状态更新、输出返回都放在一条路径里。

但真实 serving 系统不能长期停留在这个结构。

当并发上来后，同步 loop 会暴露很多问题：

1. HTTP 接收请求会阻塞 engine step。
2. tokenization 会抢 CPU。
3. detokenization 会拖慢 streaming 输出。
4. 输出队列慢会反压 scheduler。
5. metrics 和日志写入会干扰主循环。
6. client 断开连接后 request 状态很难及时清理。
7. 多个 worker 或多张卡之间缺少清晰的异步边界。

本章就把 mini engine 从一个同步 loop 升级成异步 serving 架构。

## 54.1 本章目标

读完本章，你应该能讲清：

1. 同步 engine loop 为什么适合教学但不适合高并发 serving。
2. serving engine 里哪些工作应该异步化。
3. API server、tokenizer、engine、output worker 之间如何通过队列连接。
4. request 生命周期如何跨异步组件流转。
5. streaming 输出如何避免阻塞 GPU 调度。
6. backpressure、timeout、cancel 如何设计。
7. async 架构下如何保证状态一致性。
8. 哪些指标可以判断异步拆分是否真的有效。
9. 面试里如何描述一个生产级 LLM serving runtime。

本章重点不是某个 Python async 语法。

重点是 serving 系统的异步边界设计。

## 54.2 同步 loop 的隐藏问题

同步 loop 最大优点是简单。

每轮发生什么都在一段代码里。

但它的问题也来自这里：

```text
任何一个慢步骤都会拖慢整个 engine。
```

比如一次 engine step 里：

```text
pop request:        0.2 ms
tokenization:       5.0 ms
schedule:           0.5 ms
build batch:        0.7 ms
GPU forward:       12.0 ms
sample:             0.4 ms
detokenization:     3.0 ms
send response:     10.0 ms
metrics/logging:    1.0 ms
```

如果这些都串行执行，本轮总耗时就是：

```text
32.8 ms
```

但真正需要卡住 GPU 调度的，也许只有 schedule、build batch、GPU forward、sample 和核心状态更新。

tokenization、detokenization、网络发送、日志写入都可以放到旁路。

同步 loop 会把所有延迟加在一起。

异步架构的目标是：

```text
让 GPU 主循环只等待必须等待的事情。
```

## 54.3 哪些工作应该从主循环拆出去

一个 LLM serving 请求大致经历这些阶段：

1. HTTP/gRPC 接入。
2. 鉴权、参数校验、限流。
3. prompt tokenization。
4. admission control。
5. scheduler 排队。
6. prefill/decode GPU 执行。
7. sampling。
8. detokenization。
9. streaming response。
10. metrics、trace、日志。
11. finish、cancel、timeout 清理。

其中真正必须和 GPU engine loop 强绑定的是：

1. scheduler 排队。
2. batch 构造。
3. model forward。
4. sampling。
5. request 核心状态更新。
6. KV cache 分配和释放。

其他部分都应该尽量异步化。

可以拆成下面几个组件：

```text
Client
  |
  v
API Server
  |
  v
Tokenizer Worker
  |
  v
Engine Input Queue
  |
  v
Engine Core Loop
  |
  v
Output Queue
  |
  v
Detokenizer / Stream Worker
  |
  v
Client
```

这样做的核心收益是：

1. API server 不直接阻塞 GPU loop。
2. tokenization 可以并行。
3. detokenization 和网络发送可以并行。
4. engine loop 可以稳定按 iteration step 推进。
5. slow client 不会直接拖慢所有请求。

## 54.4 最小异步架构

先定义几个队列：

```python
from dataclasses import dataclass
from queue import Queue


engine_input_queue: Queue["TokenizedRequest"] = Queue(maxsize=1024)
engine_output_queue: Queue["EngineOutput"] = Queue(maxsize=4096)
cancel_queue: Queue[str] = Queue(maxsize=1024)
```

再定义请求对象：

```python
@dataclass
class RawRequest:
    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    arrival_time: float


@dataclass
class TokenizedRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    temperature: float
    arrival_time: float


@dataclass
class EngineOutput:
    request_id: str
    token_id: int | None
    finished: bool
    error: str | None = None
```

异步组件之间只传必要信息。

不要把复杂的 engine 内部对象直接暴露给 API server。

这能减少跨线程状态污染。

## 54.5 API server 的职责

API server 不应该直接做 GPU 调度。

它的职责是：

1. 接收请求。
2. 生成 request_id。
3. 校验参数。
4. 做基础限流。
5. 把 RawRequest 交给 tokenizer。
6. 维护 client connection 和输出通道。
7. 处理 client disconnect。

伪代码：

```python
async def handle_generate(http_request):
    raw = RawRequest(
        request_id=new_request_id(),
        prompt=http_request.prompt,
        max_tokens=http_request.max_tokens,
        temperature=http_request.temperature,
        arrival_time=time.time(),
    )

    output_stream = output_registry.register(raw.request_id)

    try:
        await tokenizer_queue.put(raw)
        async for text in output_stream:
            yield text
    finally:
        output_registry.unregister(raw.request_id)
        cancel_queue.put(raw.request_id)
```

这里有一个关键点：

```text
client 断开连接时，要把 cancel 信号传给 engine。
```

否则 engine 会继续为已经没人接收的请求生成 token。

这会浪费 GPU 和 KV cache。

## 54.6 Tokenizer worker

tokenization 通常在 CPU 上完成。

它可能比你想象中更容易成为瓶颈。

尤其是：

1. prompt 很长。
2. QPS 很高。
3. chat template 很复杂。
4. tokenizer 是 Python 实现或锁竞争严重。
5. API server 和 tokenizer 在同一个 event loop 里互相影响。

tokenizer worker 可以这样写：

```python
def tokenizer_worker(tokenizer, tokenizer_queue, engine_input_queue):
    while True:
        raw: RawRequest = tokenizer_queue.get()
        try:
            token_ids = tokenizer.encode(raw.prompt)
            tokenized = TokenizedRequest(
                request_id=raw.request_id,
                prompt_token_ids=token_ids,
                max_tokens=raw.max_tokens,
                temperature=raw.temperature,
                arrival_time=raw.arrival_time,
            )
            engine_input_queue.put(tokenized)
        except Exception as exc:
            engine_output_queue.put(EngineOutput(
                request_id=raw.request_id,
                token_id=None,
                finished=True,
                error=str(exc),
            ))
```

真实系统里 tokenizer worker 可以是线程池，也可以是独立进程。

选择取决于 tokenizer 实现是否释放 GIL、CPU 核数、内存开销和部署复杂度。

## 54.7 Engine core loop

engine core loop 仍然保持 iteration-level scheduling。

区别是它不再直接处理 HTTP 请求和字符串输出。

它只从 input queue 拉取已经 tokenized 的请求。

```python
def engine_loop(engine):
    while True:
        drain_cancel_queue(engine)
        drain_new_requests(engine)

        schedule = engine.scheduler.schedule()
        if schedule.is_empty():
            wait_for_work_briefly()
            continue

        batch = engine.batch_builder.build(schedule)
        outputs = engine.model_runner.forward(batch)
        sampled = engine.sampler.sample(outputs, batch)
        engine.output_processor.process(schedule, sampled)
```

`drain_new_requests` 不应该无限拉取。

否则可能导致本轮一直处理新请求，不进入 GPU forward。

应该设置上限：

```python
def drain_new_requests(engine, max_drain: int = 128) -> None:
    for _ in range(max_drain):
        try:
            req = engine_input_queue.get_nowait()
        except Empty:
            break
        engine.add_request(req)
```

这个小细节很重要。

异步系统里，任何 drain loop 都要避免饿死主循环。

## 54.8 OutputProcessor 不应该直接写网络

教学版里可能会在 output_processor 里直接：

```python
send_token_to_client(request_id, token)
```

这在高并发下很危险。

网络发送可能变慢。

client 可能断开。

HTTP stream 可能被 backpressure 卡住。

如果 OutputProcessor 直接写网络，就会把这些慢操作带进 engine loop。

更好的做法是：

```python
class OutputProcessor:
    def process(self, schedule, sampled_tokens):
        for req, token_id in sampled_tokens:
            req.output_token_ids.append(token_id)
            finished = self.check_finished(req)

            engine_output_queue.put(EngineOutput(
                request_id=req.request_id,
                token_id=token_id,
                finished=finished,
            ))

            if finished:
                self.finish_request(req)
```

OutputProcessor 只负责把 token id 放进 output queue。

真正的 detokenization 和网络发送交给 stream worker。

## 54.9 Detokenizer 和 stream worker

detokenizer worker 从 output queue 读取 token id，转成 text，再写给对应 client。

```python
def detokenizer_worker(tokenizer, engine_output_queue, output_registry):
    while True:
        out: EngineOutput = engine_output_queue.get()

        if out.error is not None:
            output_registry.send_error(out.request_id, out.error)
            output_registry.close(out.request_id)
            continue

        if out.token_id is not None:
            text = tokenizer.decode([out.token_id])
            output_registry.send(out.request_id, text)

        if out.finished:
            output_registry.close(out.request_id)
```

这里也有坑。

很多 tokenizer 的 decode 不是简单地逐 token decode 再拼接。

有些 BPE token 需要和上下文一起 decode 才能得到正确字符串边界。

更稳妥的方式是为每个请求维护增量 detokenization 状态：

```python
class DetokenizationState:
    def __init__(self):
        self.token_ids: list[int] = []
        self.emitted_text = ""

    def append(self, tokenizer, token_id: int) -> str:
        self.token_ids.append(token_id)
        full_text = tokenizer.decode(self.token_ids)
        delta = full_text[len(self.emitted_text):]
        self.emitted_text = full_text
        return delta
```

这不是最高效实现。

但它能解释核心问题：

```text
streaming detokenization 需要维护请求级状态。
```

真实系统会用更高效的增量 detokenizer。

## 54.10 backpressure

异步架构不是把所有东西丢进无限队列。

无限队列只会把延迟和内存问题藏起来。

每个队列都应该有容量上限。

当队列满了，要明确处理策略。

常见队列包括：

1. tokenizer_queue。
2. engine_input_queue。
3. engine_output_queue。
4. 每个 client 的 stream queue。

不同队列满了，策略不同。

### 54.10.1 tokenizer_queue 满

说明入口请求超过 CPU tokenization 能力。

可以：

1. 返回 429。
2. 排队但设置最大等待时间。
3. 扩容 tokenizer worker。
4. 降低上游 QPS。

### 54.10.2 engine_input_queue 满

说明 tokenizer 已经生成了很多请求，但 engine 接不动。

可以：

1. 暂停 tokenizer 输出。
2. 对 API server 反压。
3. 触发 admission control。
4. 返回 overload 错误。

### 54.10.3 engine_output_queue 满

这是非常危险的信号。

说明 engine 生成 token 的速度超过输出侧消费能力。

如果 engine loop 阻塞在 `put` 上，GPU 调度会被拖慢。

可以选择：

1. 给 output queue 足够容量。
2. 使用非阻塞 put，失败时 cancel 慢请求。
3. 按 request 维护小队列，慢 client 只影响自己。
4. 对 streaming 输出做合并，减少发送次数。

### 54.10.4 client stream queue 满

说明某个 client 消费太慢。

不要让它拖累全局 output queue。

常见策略是：

1. 超过阈值后取消该请求。
2. 合并多个 token 后再发送。
3. 设置最大 buffered bytes。
4. 关闭慢连接。

backpressure 的原则是：

```text
慢组件必须被隔离，不能拖垮 GPU 主循环。
```

## 54.11 timeout 和 cancel

异步系统必须认真处理 cancel。

取消可能来自：

1. client 断开连接。
2. 请求超过最大排队时间。
3. 请求超过最大执行时间。
4. 上游主动取消。
5. output queue 反压。
6. engine 内部错误。

cancel 不是简单地从某个 dict 里删掉 request。

如果请求已经进入 engine，它可能占有：

1. waiting queue 位置。
2. running queue 位置。
3. KV blocks。
4. prefix cache active refs。
5. output registry 状态。
6. trace 和 metrics 状态。

所以 cancel 应该统一由 engine core 处理。

API server 只发送 cancel signal。

```python
def drain_cancel_queue(engine, max_drain: int = 1024) -> None:
    for _ in range(max_drain):
        try:
            request_id = cancel_queue.get_nowait()
        except Empty:
            break
        engine.cancel_request(request_id)
```

`cancel_request` 要保证幂等：

```python
def cancel_request(self, request_id: str) -> None:
    req = self.requests.get(request_id)
    if req is None:
        return
    if req.status in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
        return

    self.scheduler.remove(req)
    self.block_manager.release_request_blocks(req)
    self.prefix_cache.release_active_refs(req)
    req.status = RequestStatus.CANCELLED
    self.metrics.inc("request_cancel_total")
    self.requests.pop(request_id, None)
```

幂等很重要。

同一个请求可能同时因为 client disconnect 和 timeout 被取消。

## 54.12 admission control

异步架构需要 admission control。

否则入口队列会把系统压垮。

admission control 要回答：

```text
这个请求现在能不能进入 engine？
```

它可以检查：

1. tokenizer_queue 长度。
2. engine_input_queue 长度。
3. waiting queue 长度。
4. running queue 长度。
5. KV free blocks。
6. 估算 prompt blocks。
7. max_model_len。
8. 当前 preemption rate。
9. 当前 p99 TTFT。

最小策略可以很简单：

```python
def should_admit(req, metrics, config) -> bool:
    if metrics.gauges["waiting_queue_size"] > config.max_waiting_requests:
        return False
    if len(req.prompt_token_ids) > config.max_model_len:
        return False
    if metrics.gauges["kv_free_blocks"] < config.min_free_blocks_to_admit:
        return False
    return True
```

这比把请求全部塞进 waiting queue 更好。

拒绝请求虽然不好，但比让所有请求一起超时更可控。

## 54.13 状态所有权

异步架构最容易出 bug 的地方是状态所有权不清晰。

一个重要原则是：

```text
engine 内部状态只允许 engine core 修改。
```

engine 内部状态包括：

1. RequestState。
2. waiting/running/swapped 队列。
3. KV block table。
4. prefix cache active refs。
5. scheduler metadata。
6. request status。

API server 不应该直接改 request status。

detokenizer 不应该释放 KV blocks。

tokenizer 不应该把请求插入 running queue。

它们只能通过消息告诉 engine：

1. 新请求来了。
2. 某个请求取消了。
3. 输出侧失败了。

engine core 决定如何更新内部状态。

这样可以避免大量锁和竞态。

## 54.14 线程、进程还是 asyncio

异步 serving 可以用不同实现方式。

### 54.14.1 asyncio

适合：

1. HTTP/gRPC 网络层。
2. streaming response。
3. 大量连接管理。
4. 非阻塞 IO。

不适合直接跑重 CPU tokenization。

CPU 重任务会阻塞 event loop。

### 54.14.2 线程

适合：

1. tokenizer 释放 GIL 的场景。
2. detokenization 和轻量 CPU 工作。
3. 简单共享内存队列。

缺点是要注意线程安全。

### 54.14.3 进程

适合：

1. CPU-heavy tokenization。
2. 隔离不同 worker。
3. 避免 GIL。
4. 多卡多 engine 部署。

缺点是序列化和跨进程通信成本更高。

### 54.14.4 实用组合

常见组合是：

```text
asyncio API server
+ tokenizer thread/process pool
+ dedicated engine thread/process
+ output async tasks
```

不要为了“全 async”把 GPU engine loop 写得过于复杂。

很多生产系统会让 engine core 独占一个线程或进程，保持调度状态简单。

## 54.15 异步架构下的指标

第 53 章的指标仍然需要保留。

但异步架构还要增加队列和组件指标。

### 54.15.1 队列指标

```text
tokenizer_queue_size
engine_input_queue_size
engine_output_queue_size
client_stream_queue_size
queue_put_blocked_total
queue_put_timeout_total
queue_drop_total
```

这些指标用来判断 backpressure 是否发生。

### 54.15.2 组件耗时

```text
api_request_validation_ms
tokenization_ms
engine_waiting_ms
scheduler_step_ms
gpu_forward_ms
detokenization_ms
stream_send_ms
```

这些指标用来定位瓶颈到底在 CPU、GPU 还是网络。

### 54.15.3 cancel 和 timeout

```text
request_cancel_total
client_disconnect_total
request_timeout_total
admission_reject_total
slow_client_cancel_total
```

这些指标用来判断系统是否在过载状态下健康降级。

如果 admission_reject_total 上升，但 p99 延迟稳定，说明限流可能在保护系统。

如果没有 reject，但 timeout 大量增加，说明系统在假装接收请求，实际无法完成。

## 54.16 异步拆分后的 benchmark 对比

异步化不是天然优化。

它也可能引入额外队列延迟和上下文切换。

所以需要 benchmark 验证。

对比同步版和异步版时，至少看：

1. TTFT p50/p90/p99。
2. TPOT p50/p90/p99。
3. output tokens/s。
4. GPU busy time。
5. tokenization queue wait。
6. output queue wait。
7. client disconnect 后释放 KV 的延迟。
8. engine step duration 抖动。

一个理想结果可能是：

```text
同步版：GPU forward 12ms，但 engine step 平均 31ms。
异步版：GPU forward 12ms，engine step 平均 15ms。
```

这说明 CPU 和网络工作被移出了主循环。

但如果结果是：

```text
异步版 TTFT p99 更高，engine_input_queue 长期积压。
```

说明异步队列或 tokenizer worker 成了新瓶颈。

## 54.17 常见 bug

bug 一：output queue 满导致 engine loop 阻塞。

```text
结果：GPU 空转，所有请求 TPOT 抖动。
```

bug 二：client 断开后没有 cancel engine request。

```text
结果：僵尸请求继续占用 KV cache 和 decode slots。
```

bug 三：cancel 直接由 API server 修改 engine 状态。

```text
结果：线程竞态，block table 或 queue 状态不一致。
```

bug 四：drain input queue 没有上限。

```text
结果：高 QPS 下 engine 一直接新请求，GPU forward 被饿死。
```

bug 五：tokenizer 在 asyncio event loop 里同步执行。

```text
结果：网络连接处理和 streaming 输出都被 CPU tokenization 阻塞。
```

bug 六：逐 token decode 不维护 detokenization 状态。

```text
结果：streaming 文本出现乱码、重复或空字符串。
```

bug 七：队列无限大。

```text
结果：系统不拒绝请求，但延迟无限增长，最终内存爆掉。
```

bug 八：timeout 后没有释放 prefix cache active refs。

```text
结果：cached blocks ref count 泄漏，后续 eviction 失效。
```

bug 九：metrics/logging 同步写磁盘。

```text
结果：磁盘抖动影响 engine step latency。
```

bug 十：没有请求状态机。

```text
结果：同一个请求可能同时 finished、cancelled、streaming，清理逻辑混乱。
```

## 54.18 面试高频问题

问题一：为什么 LLM serving engine 不能只写一个同步 loop？

回答要点：同步 loop 适合教学，但高并发下 tokenization、detokenization、网络发送、日志和 metrics 都可能阻塞 GPU 主循环。生产系统需要把 API server、tokenizer、engine core、output worker 拆开，让 GPU loop 只等待必须等待的调度、batch 构造、forward、sampling 和状态更新。

问题二：异步架构里 request 状态应该由谁修改？

回答要点：engine 内部状态应该只由 engine core 修改，包括 request status、waiting/running 队列、KV block table、prefix cache refs。API server、tokenizer、detokenizer 只能通过消息传递新请求、cancel 或输出失败，避免跨线程竞态。

问题三：如何处理 client disconnect？

回答要点：API server 检测到连接断开后发送 cancel signal，engine core 在自己的循环里消费 cancel queue，幂等地从 scheduler 移除请求，释放 KV blocks 和 prefix cache active refs，更新状态和指标。不能只关闭 HTTP 连接而不通知 engine。

问题四：backpressure 怎么做？

回答要点：所有关键队列都要有上限，包括 tokenizer queue、engine input queue、engine output queue 和 client stream queue。队列满时要选择拒绝、限流、cancel 慢请求或 admission control，不能用无限队列隐藏过载。慢 client 也要隔离，不能阻塞全局 output queue。

问题五：asyncio、线程和进程怎么选？

回答要点：asyncio 适合网络 IO 和 streaming 连接管理；线程适合轻量 CPU 工作或释放 GIL 的 tokenizer；进程适合 CPU-heavy tokenization、多卡隔离和避免 GIL。常见架构是 async API server 加 tokenizer worker pool，加独立 engine thread/process，加异步 output tasks。

## 54.19 标准回答模板

如果面试官问“生产级 LLM serving runtime 为什么要异步化”，可以这样回答：

```text
教学版 engine 可以写成同步 loop：接请求、tokenize、schedule、build batch、GPU forward、sample、detokenize、send response 全部串起来。但生产高并发下这样会把 CPU、网络和日志开销都加到 engine step 上，导致 GPU 主循环被 tokenization、detokenization、慢 client 或 metrics 阻塞。

我会把系统拆成几个异步组件：API server 负责接入、参数校验、限流和连接管理；tokenizer worker 负责 prompt tokenization；engine input queue 把 tokenized request 交给 engine core；engine core 独占调度状态，做 iteration-level scheduling、batch 构造、GPU forward、sampling、KV cache 和 request 状态更新；output queue 把 token id 交给 detokenizer 和 stream worker，后者负责增量 detokenization 和网络发送。

这里最重要的是状态所有权和 backpressure。engine 内部状态只能由 engine core 修改，其他组件通过消息传递新请求和 cancel 信号。所有队列都要有容量上限，队列满时要做 admission control、限流、超时或取消慢请求，不能用无限队列把问题藏起来。client disconnect 也必须传给 engine，幂等释放 KV blocks 和 prefix cache refs。

验证异步化是否有效不能只看吞吐，要看 TTFT、TPOT、E2E、engine step duration、GPU busy time、tokenizer queue wait、output queue wait、cancel cleanup latency 等指标。如果异步化后 GPU loop 更稳定、慢 client 不影响整体 TPOT、队列不长期积压，才说明架构拆分有效。
```

## 54.20 小练习

1. 给 mini engine 增加 `engine_input_queue` 和 `engine_output_queue`。
2. 把 API request 转成 `RawRequest`，不要直接进入 scheduler。
3. 实现一个 tokenizer worker，把 `RawRequest` 转成 `TokenizedRequest`。
4. 修改 engine loop，只从 input queue 拉取 tokenized requests。
5. 给 `drain_new_requests` 增加每轮最大拉取数量。
6. 修改 OutputProcessor，让它只写 `EngineOutput`，不直接写网络。
7. 实现 detokenizer worker，把 token id 转成 streaming text。
8. 给每个 request 增加 detokenization state。
9. 实现 client disconnect 到 cancel queue 的路径。
10. 实现幂等的 `cancel_request`。
11. 给所有队列设置 maxsize。
12. 增加 queue size、queue wait、cancel total 指标。
13. 构造慢 client 压测，验证不会阻塞 engine loop。
14. 构造 tokenizer CPU 瓶颈压测，观察 tokenizer queue。
15. 对比同步版和异步版的 TTFT、TPOT、engine step duration。

## 54.21 本章总结

同步 engine loop 适合教学，但不适合长期支撑高并发 serving。

真实系统里，GPU forward 不是唯一瓶颈。

tokenization、detokenization、网络 streaming、日志、metrics、慢 client、cancel 和 timeout 都会影响整体性能和稳定性。

异步 serving 架构的核心是把 API server、tokenizer、engine core、output worker 拆开，用有界队列连接。

engine core 应该独占 request 状态、scheduler 队列、KV block table 和 prefix cache refs。

其他组件通过消息和队列与 engine 通信，不直接修改 engine 内部状态。

backpressure 是异步架构的关键。

无限队列不是解决方案，只是把过载变成更大的延迟和内存风险。

client disconnect、timeout 和 cancel 必须最终回到 engine core，由 engine 幂等清理 request、KV blocks 和 refs。

异步化之后仍然要通过 benchmark 验证效果。

如果 engine step 更稳定、GPU idle 更少、慢 client 不拖累全局 TPOT、队列不长期积压，说明架构拆分是有效的。

下一章可以继续讨论：当单个 engine worker 不够时，如何扩展到多 worker、多 GPU 和 request router。
