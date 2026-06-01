# 第 43 章 如何把单机 Serving Engine 升级成 PD 分离架构

上一章讲了 PD 分离的优缺点、适用场景和反模式。

这一章进入更落地的问题：

```text
如果我已经有一个单机 serving engine，应该如何一步步把它改造成 PD 分离架构？
```

这里的重点不是一上来重写一个分布式系统，而是识别：

1. 哪些模块必须先抽象出来。
2. 哪些状态必须从单进程内存里剥离出来。
3. 哪些接口必须先稳定。
4. 哪些能力可以分阶段上线。
5. 哪些改造如果顺序错了，会让系统变得很难维护。

一句话概括：

> 从单机 serving engine 升级到 PD 分离，本质是把“一个进程内部的 generate 状态机”逐步拆成“router、prefill worker、decode worker、KV transfer 和分布式请求状态”共同维护的系统。

## 43.1 本章目标

读完本章，你应该能讲清：

1. 单机 serving engine 中哪些模块会阻碍 PD 分离。
2. 为什么不能直接把 `prefill()` 和 `decode()` 拆成两个服务。
3. 升级前需要先抽象哪些接口。
4. Request state、KV metadata、scheduler state 应该如何拆分。
5. Prefill worker 和 Decode worker 的职责如何重新定义。
6. KV transfer 如何接入现有 KV cache manager。
7. 如何分阶段从 unified engine 演进到 PD 分离。
8. 上线 PD 分离时如何灰度、回滚和观测。
9. 面试中如何回答“如何把单机推理框架升级成 PD 分离”。

## 43.2 先看单机 Engine 长什么样

一个简化的单机 serving engine 通常是这样：

```text
client
  -> HTTP API
  -> request queue
  -> scheduler
  -> model runner
       -> prefill
       -> decode loop
  -> streamer
```

代码里可能长这样：

```python
class Engine:
    def add_request(self, request):
        self.waiting_queue.append(request)

    def step(self):
        batch = self.scheduler.schedule(self.waiting_queue, self.running)

        if batch.has_prefill:
            self.model_runner.prefill(batch.prefill_requests)

        if batch.has_decode:
            tokens = self.model_runner.decode(batch.decode_requests)
            self.streamer.send(tokens)
```

这个结构适合教学项目和单机推理，但它有一个隐含假设：

```text
一个请求的所有状态都在同一个 engine 进程里。
```

包括：

1. 请求输入和采样参数。
2. 当前生成到第几个 token。
3. KV cache block 分配结果。
4. prefill 是否完成。
5. decode 是否正在运行。
6. streaming channel。
7. 取消、超时、失败状态。

PD 分离会打破这个假设。

## 43.3 为什么不能直接拆两个服务

很多人会把 PD 分离理解成：

```text
prefill_service.compute_prompt()
decode_service.generate_tokens()
```

这只是函数拆分，不是系统架构。

真正的问题是：

```text
Prefill 生成的 KV 在哪里？
Decode 怎么知道 KV 的 shape、block id、layer 数、dtype、layout？
请求失败时谁清理 KV？
Decode worker 崩溃后 router 怎么处理已经 prefill 完的请求？
客户端取消请求时 prefill、transfer、decode 三边如何同步取消？
```

如果这些问题没有答案，拆服务只会得到一个更慢、更脆弱的 unified engine。

所以升级顺序应该是：

```text
先抽象状态和接口，再拆 worker；
先做同进程逻辑分层，再做跨进程部署；
先做 same-node transfer，再做 cross-node transfer；
先能观测和回滚，再扩大流量。
```

## 43.4 升级前要先做的模块边界

单机 engine 要升级 PD 分离，至少要先把五个边界切清楚。

第一，API 接入层和 engine 核心分离。

```text
HTTP/gRPC server 不应该直接操作 KV cache、scheduler 内部队列和 model runner。
```

它应该只负责：

1. 接收请求。
2. 鉴权和限流。
3. 创建 request id。
4. 建立 streaming channel。
5. 把请求交给 engine/router。

第二，请求状态和执行状态分离。

请求状态是：

```text
request_id、prompt、sampling_params、tenant、deadline、priority、stream handle
```

执行状态是：

```text
current_stage、prefill_worker、decode_worker、kv_location、generated_tokens、finish_reason
```

第三，scheduler 和 model runner 分离。

Scheduler 负责决定谁运行。

Model runner 负责执行模型计算。

不要让 scheduler 直接依赖模型 forward 的内部 tensor。

第四，KV cache manager 和 model runner 分离。

Model runner 可以读写 KV，但 KV block 的生命周期、分配、释放、metadata 应该由 KV manager 管。

第五，streaming 输出和 decode loop 分离。

Decode worker 可以产生 token，但最好不要让它直接持有所有客户端连接状态。否则 decode worker 崩溃时很难恢复和回收。

## 43.5 单机 Engine 的状态拆分

升级前最重要的是状态盘点。

单机状态可以分成四类。

第一类，请求元数据。

```text
request_id
tenant_id
prompt_token_ids
sampling_params
max_new_tokens
priority
deadline
trace_id
```

这类状态应该能被 router、prefill worker、decode worker 共同理解。

第二类，执行进度。

```text
stage = WAITING_PREFILL / PREFILLING / WAITING_TRANSFER / DECODING / FINISHED / FAILED
prefill_started_at
prefill_finished_at
decode_started_at
generated_token_count
last_token_id
finish_reason
```

这类状态最好由 router 或外部 request state store 维护。

第三类，KV metadata。

```text
kv_id
model_id
layer_count
num_heads
head_dim
dtype
block_size
block_ids
sequence_length
device_location
transfer_status
```

这类状态不能只存在于某个 worker 的 Python 对象里，因为 decode worker 必须能根据它接管请求。

第四类，运行时临时状态。

```text
CUDA graph handle
temporary tensor
attention workspace
sampling buffer
local queue node
```

这类状态通常不能跨 worker 迁移，也不应该外置。

工程上最容易犯的错，是把第三类 KV metadata 和第四类 runtime temporary state 混在一起。

## 43.6 第一步：把 Prefill 和 Decode 变成显式阶段

很多教学 engine 里，prefill 和 decode 虽然概念上不同，但代码上混在 `step()` 里。

升级的第一步，是让请求阶段显式化。

例如：

```python
class RequestStage:
    WAITING_PREFILL = "waiting_prefill"
    PREFILLING = "prefilling"
    WAITING_DECODE = "waiting_decode"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"
```

请求对象里不要只有一个 `is_finished`，而应该有清晰阶段：

```python
class RequestState:
    def __init__(self, request_id, prompt_token_ids, sampling_params):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.stage = RequestStage.WAITING_PREFILL
        self.kv_metadata = None
        self.generated_token_ids = []
```

这样即使仍然是单进程，也能模拟 PD 生命周期：

```text
WAITING_PREFILL
  -> PREFILLING
  -> WAITING_DECODE
  -> DECODING
  -> FINISHED
```

这一步的价值是：

1. 后续 router 可以接管 stage transition。
2. prefill worker 和 decode worker 可以只处理自己关心的阶段。
3. 失败恢复可以按照阶段设计。
4. 指标可以按阶段统计。

## 43.7 第二步：抽象 Model Runner 接口

单机 model runner 通常暴露一个粗粒度接口：

```python
tokens = model.generate(input_ids, sampling_params)
```

这不适合 serving engine。

推理框架内部至少要拆成：

```python
class ModelRunner:
    def prefill(self, batch):
        """Run prompt tokens and write KV cache."""

    def decode(self, batch):
        """Run one or more decode steps using existing KV cache."""
```

进一步，为 PD 分离做准备，prefill 的返回值不能只是 logits。

它应该返回：

```python
class PrefillResult:
    request_id: str
    next_token_id: int
    kv_metadata: KVMetadata
    prompt_length: int
```

Decode 的输入也不能只是一串 token：

```python
class DecodeInput:
    request_id: str
    last_token_id: int
    kv_metadata: KVMetadata
    sampling_params: SamplingParams
```

也就是说，decode 应该通过 `kv_metadata` 找到已经生成的 KV，而不是依赖同一个 Python 对象引用。

## 43.8 第三步：抽象 KV Cache Manager

PD 分离里最关键的不是 worker 拆分，而是 KV 生命周期。

一个单机 KV cache manager 至少要支持：

```python
class KVCacheManager:
    def allocate(self, request_id, num_tokens):
        pass

    def get_metadata(self, request_id):
        pass

    def pin(self, kv_id):
        pass

    def unpin(self, kv_id):
        pass

    def free(self, kv_id):
        pass
```

为了升级 PD，还需要增加 transfer 相关抽象：

```python
class KVTransferManager:
    def export(self, kv_metadata):
        pass

    def import_(self, transfer_handle):
        pass

    def release(self, transfer_handle):
        pass
```

这里不要过早绑定某一种后端。

后端可能是：

1. 同 GPU 内部引用。
2. 同节点 GPU-to-GPU 拷贝。
3. CPU staging buffer。
4. RDMA transfer。
5. NIXL。
6. Mooncake。
7. 远端 KV cache。

接口要表达的是：

```text
这段 KV 可以被另一个执行单元接管。
```

而不是一开始就把实现写死成某个 `cudaMemcpyPeer`。

## 43.9 第四步：把 Scheduler 拆成两类

Unified engine 里通常只有一个 scheduler：

```text
waiting queue + running queue -> next batch
```

PD 分离后至少有两个 scheduler：

1. Prefill scheduler。
2. Decode scheduler。

Prefill scheduler 关注：

1. prompt token 数。
2. chunk size。
3. prefix cache 命中。
4. input tokens/s。
5. prefill queue waiting time。

Decode scheduler 关注：

1. active sequence 数。
2. 每轮 decode batch 大小。
3. TPOT。
4. max running requests。
5. KV capacity。
6. streaming deadline。

在单机阶段，可以先做逻辑拆分：

```text
one process
  prefill_scheduler
  decode_scheduler
  same model_runner
  same kv_manager
```

这时还没有真正 PD 分离，但代码已经具备演进条件。

## 43.10 第五步：引入 Router 状态机

Router 在 PD 分离里不是普通负载均衡。

它维护的是请求生命周期。

简化状态机如下：

```text
RECEIVED
  -> PREFILL_ASSIGNED
  -> PREFILL_RUNNING
  -> PREFILL_DONE
  -> KV_TRANSFERRING
  -> DECODE_ASSIGNED
  -> DECODING
  -> FINISHED
```

失败路径包括：

```text
PREFILL_FAILED
TRANSFER_FAILED
DECODE_FAILED
CANCELLED
TIMEOUT
```

Router 至少要维护：

```text
request_id -> request state
request_id -> prefill worker
request_id -> decode worker
request_id -> kv transfer handle
request_id -> stream handle
```

单机改造时，可以先把 router 做成进程内组件：

```text
API server -> local router -> local engine
```

等状态机稳定后，再把它升级为独立服务。

## 43.11 第六步：把 Worker 角色显式化

原来的 engine worker 是全能型：

```text
prefill + decode + kv allocation + streaming
```

PD 分离后，worker 要分角色。

Prefill worker 负责：

1. 接收 prompt token。
2. 运行 prefill。
3. 写入本地 KV。
4. 生成首 token 或 prefill logits。
5. 导出 KV metadata。
6. 触发或等待 KV transfer。

Decode worker 负责：

1. 接收 decode assignment。
2. 导入或挂载 KV。
3. 持续 decode。
4. 执行 sampling。
5. 输出 token stream。
6. 请求结束后释放 KV。

注意职责边界：

```text
Prefill worker 不应该继续持有请求直到生成结束。
Decode worker 不应该重新计算完整 prompt。
Router 不应该直接操作 CUDA tensor。
```

## 43.12 第七步：先做 Same-Node PD

不要一上来跨节点。

推荐演进路径是：

```text
阶段 1：同进程逻辑 PD
阶段 2：同节点多进程 PD
阶段 3：同节点多 GPU PD
阶段 4：跨节点 PD
```

Same-node PD 的好处是：

1. 网络变量少。
2. 故障域小。
3. debug 简单。
4. transfer 可以先用较简单的机制。
5. 可以验证状态机和调度策略。

如果 same-node PD 都无法带来收益，直接 cross-node PD 通常更危险。

## 43.13 第八步：接入 KV Transfer

KV transfer 是 PD 分离的核心数据路径。

一个 transfer 流程可以抽象成：

```text
prefill worker
  -> freeze/pin KV blocks
  -> export metadata
  -> transfer data
  -> notify router
  -> decode worker import
  -> decode starts
```

关键点有四个。

第一，KV layout 必须一致。

Decode worker 需要知道：

1. layer 数。
2. 每层 K/V 的 shape。
3. dtype。
4. block size。
5. head layout。
6. sequence length。
7. block table。

第二，transfer 状态必须可观测。

至少要有：

```text
transfer_pending
transfer_running
transfer_done
transfer_failed
transfer_cancelled
```

第三，失败要能清理。

例如：

```text
prefill done, transfer failed -> free prefill KV
transfer done, decode assignment failed -> free decode-side KV
client cancelled during transfer -> cancel transfer and free both sides
```

第四，decode 不能在 KV 未 ready 时盲目开始。

否则会出现难 debug 的错误：

1. 读到未初始化 KV。
2. block table 不完整。
3. sequence length 错位。
4. attention mask 错误。
5. token 输出异常但不报错。

## 43.14 第九步：重新设计 Streaming 路径

单机 engine 里，streaming 通常很简单：

```text
decode loop -> HTTP response stream
```

PD 分离后，decode worker 可能不在 API server 所在进程，甚至不在同一节点。

有三种常见做法。

第一，Decode worker 直接 stream 给 client。

优点是路径短。

缺点是连接管理复杂，worker 扩缩容和故障恢复困难。

第二，Decode worker stream 给 router/API server，再由 API server 转发给 client。

优点是客户端连接集中管理。

缺点是 router/API server 可能成为数据转发瓶颈。

第三，Decode worker 写入 message channel，API server 消费并返回。

优点是解耦强。

缺点是额外延迟和系统复杂度更高。

教学项目或早期系统可以先选择第二种：

```text
decode worker -> router/API -> client
```

因为它更容易做鉴权、取消、超时、trace 和回放。

## 43.15 第十步：引入 Backpressure

PD 分离后，系统不再是一个队列，而是多个队列：

```text
router queue
prefill queue
transfer queue
decode queue
streaming queue
```

如果没有 backpressure，常见事故是：

1. prefill 太快，decode 接不住，KV 堆积。
2. decode 太满，prefill 做了很多无效工作。
3. transfer backlog 增长，TTFT p99 爆炸。
4. router 继续接请求，最终所有队列一起雪崩。

Backpressure 信号至少包括：

```text
prefill_queue_depth
decode_queue_depth
transfer_queue_depth
decode_kv_free_blocks
gpu_memory_pressure
ttft_p95/p99
tpot_p95/p99
```

Router 应该根据这些信号决定：

1. 是否接新请求。
2. 是否延迟 prefill。
3. 是否选择另一个 decode worker。
4. 是否降级回 unified engine。
5. 是否拒绝低优先级租户。

## 43.16 第十一步：加完整观测

没有观测，不要上线 PD 分离。

至少要看五段延迟：

```text
request_queue_wait
prefill_time
kv_transfer_time
decode_wait_time
decode_time / tpot
```

还要看资源指标：

1. prefill GPU utilization。
2. decode GPU utilization。
3. prefill input tokens/s。
4. decode output tokens/s。
5. KV cache used/free blocks。
6. transfer bandwidth。
7. transfer p95/p99。
8. worker error rate。
9. cancel rate。
10. fallback rate。

Trace 里至少要包含：

```text
request_id
router_id
prefill_worker_id
decode_worker_id
kv_id
transfer_id
stage timestamps
failure reason
```

否则 PD 分离出问题时，很容易只看到：

```text
用户说慢了。
GPU 看起来没满。
日志里没有明显错误。
```

这是最难排查的状态。

## 43.17 第十二步：灰度和回滚

PD 分离必须能灰度。

常见灰度维度包括：

1. 按租户灰度。
2. 按模型灰度。
3. 按 prompt length 灰度。
4. 按请求比例灰度。
5. 按 region 或集群灰度。

推荐先灰度长 prompt 请求：

```text
if prompt_len > threshold:
    use_pd_path
else:
    use_unified_path
```

因为长 prompt 更可能从 PD 分离里获益。

同时要保留回滚路径：

```text
PD path unhealthy -> route new requests to unified engine
running PD requests -> allow finish or controlled cancel
failed transfer -> retry or fallback recompute
```

注意：已经进入 decode 的 streaming 请求，不一定能无损回滚。

所以回滚策略要区分：

1. 新请求回滚。
2. prefill 中请求回滚。
3. transfer 中请求回滚。
4. decode 中请求回滚。

## 43.18 一个推荐演进路线

如果从一个单机 engine 开始，推荐路线如下。

第一阶段：整理单机代码。

目标：让 engine 内部有清晰的 request state、prefill stage、decode stage、KV manager。

```text
API -> Engine -> Prefill/Decode logical stages
```

第二阶段：双 scheduler。

目标：让 prefill 和 decode 在逻辑上有不同队列和调度策略。

```text
prefill_queue
decode_queue
```

第三阶段：进程内 router。

目标：让 router 接管 request lifecycle。

```text
API -> local router -> local engine
```

第四阶段：worker role 拆分。

目标：让同一套 worker 可以以 prefill-only 或 decode-only 模式运行。

```text
worker --role=prefill
worker --role=decode
```

第五阶段：same-node PD。

目标：验证状态机、KV transfer、观测和故障清理。

```text
router -> prefill worker -> local transfer -> decode worker
```

第六阶段：cross-node PD。

目标：引入网络拓扑、RDMA/NIXL/Mooncake、跨节点故障恢复和容量调度。

```text
router -> prefill node -> network KV transfer -> decode node
```

第七阶段：高级优化。

目标：KV-aware routing、prefix-aware routing、多级 KV、租户隔离、自动扩缩容。

## 43.19 最小可行 PD 原型

一个教学级最小 PD 原型可以这样设计：

```text
PDRouter
  - request table
  - choose_prefill_worker()
  - choose_decode_worker()
  - update_stage()

PrefillWorker
  - prefill(batch)
  - export_kv()

DecodeWorker
  - import_kv()
  - decode_loop()

KVTransferManager
  - transfer(kv_metadata, src, dst)

Metrics
  - stage latency
  - queue depth
  - transfer latency
```

请求路径：

```text
1. client sends request
2. router creates request state
3. router assigns prefill worker
4. prefill worker runs prompt
5. prefill worker returns kv metadata and first token candidate
6. router assigns decode worker
7. transfer manager moves or exposes KV
8. decode worker imports KV
9. decode worker streams tokens
10. router marks request finished and releases resources
```

这个原型不需要一开始支持所有能力。

可以先不做：

1. 多租户。
2. prefix-aware routing。
3. remote KV cache。
4. 自动扩缩容。
5. 复杂重试。

但必须做：

1. request id。
2. stage state。
3. KV metadata。
4. transfer result。
5. cancel。
6. timeout。
7. resource cleanup。

## 43.20 关键接口设计示例

一个简化接口可以这样写。

```python
class PDRouter:
    def submit(self, request):
        state = self.request_store.create(request)
        prefill_worker = self.choose_prefill_worker(request)
        self.assign_prefill(state, prefill_worker)

    def on_prefill_done(self, request_id, prefill_result):
        state = self.request_store.get(request_id)
        state.kv_metadata = prefill_result.kv_metadata
        state.last_token_id = prefill_result.next_token_id

        decode_worker = self.choose_decode_worker(state)
        transfer = self.transfer_manager.start(
            state.kv_metadata,
            src=prefill_result.worker_id,
            dst=decode_worker.worker_id,
        )
        state.transfer_id = transfer.transfer_id

    def on_transfer_done(self, request_id):
        state = self.request_store.get(request_id)
        self.assign_decode(state)
```

Prefill worker：

```python
class PrefillWorker:
    def run_prefill(self, request):
        allocation = self.kv_manager.allocate(request.request_id, len(request.prompt_token_ids))
        logits = self.model_runner.prefill(request.prompt_token_ids, allocation)
        next_token = self.sampler.sample(logits, request.sampling_params)
        kv_metadata = self.kv_manager.get_metadata(request.request_id)
        return PrefillResult(request.request_id, next_token, kv_metadata)
```

Decode worker：

```python
class DecodeWorker:
    def start_decode(self, decode_input):
        self.kv_manager.import_(decode_input.kv_metadata)
        state = self.create_local_state(decode_input)

        while not state.finished:
            logits = self.model_runner.decode(state.last_token_id, state.kv_metadata)
            token = self.sampler.sample(logits, state.sampling_params)
            self.streamer.send(state.request_id, token)
            state.update(token)
```

这些代码只是示意，真实系统会更复杂。

但它体现了一个核心原则：

```text
跨组件传 metadata 和 handle，不传 Python 对象引用。
```

## 43.21 容量规划怎么变

Unified engine 里，容量规划通常看整体 QPS、GPU 利用率、显存和延迟。

PD 分离后，要分别看 P 和 D。

Prefill pool 容量主要由输入 token 决定：

```text
required_prefill_capacity ~= peak_input_tokens_per_second / prefill_tokens_per_second_per_gpu
```

Decode pool 容量主要由输出 token 和并发序列决定：

```text
required_decode_capacity ~= peak_output_tokens_per_second / decode_tokens_per_second_per_gpu
```

但这还不够，还要考虑 KV 驻留：

```text
decode_kv_capacity >= active_sequences * average_context_length * kv_bytes_per_token
```

以及 transfer：

```text
transfer_bandwidth >= prefill_completed_kv_bytes_per_second
```

如果 prefill 扩得太多而 decode 不扩，系统会变成：

```text
prefill 很快完成，KV 大量等待 decode，TTFT 仍然很差，显存还被占满。
```

所以 PD 分离不是简单地多加 prefill worker 或 decode worker，而是三个容量一起配：

1. Prefill compute。
2. Decode compute 和 KV capacity。
3. KV transfer bandwidth。

## 43.22 故障处理设计

PD 分离的故障处理要按阶段设计。

Prefill 前失败：

```text
请求还没有消耗大量资源，可以重试或换 prefill worker。
```

Prefill 中失败：

```text
释放已分配 KV，重新排队或返回错误。
```

Prefill 后 transfer 前失败：

```text
如果 prefill worker 还活着，可以重试 transfer；否则需要重新 prefill。
```

Transfer 中失败：

```text
取消 transfer，清理 src/dst 临时 KV，必要时重新 transfer 或重新 prefill。
```

Decode 中失败：

```text
如果已经 stream 出 token，通常无法透明重试；可以终止请求、返回错误，或在支持 deterministic replay 的条件下尝试恢复。
```

客户端取消：

```text
router 负责广播 cancel 到 prefill、transfer、decode，并最终确认 KV 释放。
```

这里最重要的是 cleanup。

很多线上问题不是请求失败本身，而是失败后 KV 没释放，最终把显存耗尽。

## 43.23 数据一致性问题

PD 分离里有几个容易忽略的一致性问题。

第一，token 位置一致。

Prefill 结束时，decode 的起始 position 必须和 prompt length 对齐。

第二，首 token 归属一致。

有些系统在 prefill 阶段顺便采样首 token，有些系统让 decode 阶段采样首 token。

两种都可以，但必须统一约定：

```text
prefill returns next_token -> decode starts from that token
prefill returns logits only -> decode samples first token
```

第三，sampling 状态一致。

如果使用随机采样，需要考虑 RNG seed、temperature、top-p、repetition penalty 等状态是否在 decode worker 上完整恢复。

第四，KV block table 一致。

Prefill 侧导出的 block table 和 decode 侧导入后的 block table 必须语义一致，即使物理 block id 不同。

第五，模型版本一致。

Prefill worker 和 decode worker 必须使用相同模型权重、tokenizer、rope scaling、quantization config。

否则会出现非常隐蔽的质量问题。

## 43.24 与 Prefix Cache 的关系

如果系统已经有 prefix cache，升级 PD 时要重新考虑 cache ownership。

有三种做法。

第一，prefix cache 只在 prefill pool 内部使用。

```text
router -> prefix-aware prefill worker -> generate KV -> transfer to decode
```

这种最简单。

第二，decode worker 也有本地 prefix cache。

这会让路由变复杂，因为 prefill 和 decode 都有 locality。

第三，引入远端 prefix/KV cache。

这适合大规模长上下文系统，但会增加网络和一致性成本。

早期升级建议采用第一种。

也就是：

```text
先让 prefill pool 负责 prefix cache 命中，再把最终 KV 交给 decode。
```

不要一开始同时做 PD 分离、prefix-aware decode routing 和 remote KV cache。

## 43.25 与 Chunked Prefill 的关系

Chunked prefill 是升级 PD 前常见的中间态。

如果系统已有 chunked prefill，可以这样演进：

```text
unified engine + chunked prefill
  -> logical P/D scheduler
  -> prefill chunks on prefill worker
  -> final KV transfer to decode worker
```

一个关键问题是：

```text
KV 是每个 chunk transfer，还是 prefill 完整结束后 transfer？
```

两种方式取舍不同。

完整结束后 transfer：

1. 实现简单。
2. 状态少。
3. decode 启动更晚。

Chunk 级 transfer：

1. 可以隐藏部分 transfer 时间。
2. pipeline 更充分。
3. 状态机更复杂。
4. 更容易出现 partial KV 清理问题。

早期建议先做完整 prefill 后 transfer。

等稳定后再考虑 chunk-level pipeline。

## 43.26 常见改造反模式

第一，只拆服务，不拆状态。

表现是 prefill service 和 decode service 都依赖同一个巨大 engine 对象，最终状态同步一团乱。

第二，只拆 API，不拆资源池。

两个服务还是抢同一批 GPU，P/D 干扰没有消失。

第三，KV metadata 不标准化。

每个 worker 用自己的隐式结构，transfer 后 decode 侧只能靠特殊逻辑适配。

第四，没有 cancel 和 cleanup。

压测时看起来能跑，真实流量下一堆取消请求导致 KV 泄漏。

第五，没有 fallback。

PD path 一出问题，新请求无路可走，只能全站故障。

第六，跨节点一步到位。

状态机、观测、transfer、网络问题同时出现，debug 成本极高。

第七，只看平均延迟。

PD 分离主要看尾延迟、队列等待、transfer p99 和 TPOT jitter。平均值改善不代表系统可靠。

## 43.27 面试官会怎么问

问题一：如何把一个单机 LLM serving engine 升级成 PD 分离？

回答要点：先不要直接拆服务，而是先显式化 request stage，抽象 model runner 的 prefill/decode 接口，抽象 KV cache manager 和 KV metadata，再引入 router 状态机、双 scheduler、prefill/decode worker role 和 KV transfer。落地上先做同进程逻辑 PD，再 same-node PD，最后 cross-node PD，并配套观测、backpressure、灰度和回滚。

问题二：升级过程中最关键的接口是什么？

回答要点：最关键的是 KV metadata 和 request state。Decode worker 不能依赖 prefill worker 的 Python 对象引用，而要通过标准化 KV metadata 接管 KV。同时 router 要维护请求阶段、worker assignment、transfer handle、失败状态和 stream handle。

问题三：为什么不能直接把 prefill 和 decode 拆成两个服务？

回答要点：因为真正困难的是 KV transfer、分布式状态、故障清理、取消、回滚和观测。只拆服务不拆状态和资源池，无法解决 P/D 干扰，反而会增加网络和状态同步成本。

问题四：升级 PD 分离时如何降低风险？

回答要点：先做逻辑拆分和 same-node PD，按长 prompt 或特定租户灰度；保留 unified fallback；对 prefill、transfer、decode 分阶段打点；按阶段设计失败清理；上线前压测 transfer p99、decode TPOT、KV 泄漏和 cancel 场景。

问题五：PD 分离后容量规划有什么变化？

回答要点：要分别规划 prefill compute、decode compute/KV capacity 和 KV transfer bandwidth。输入 token 压力决定 prefill pool，输出 token 和 active sequences 决定 decode pool，prefill 完成速率决定 transfer 带宽需求。

## 43.28 标准回答模板

如果面试官问“你如何把单机 serving engine 改造成 PD 分离架构”，可以这样回答：

```text
我不会一开始就把 prefill 和 decode 拆成两个远程服务，因为 PD 分离的难点不只是函数拆分，而是 request state、KV metadata、KV transfer、故障清理和调度状态的分布式化。

我会先在单机 engine 内部做逻辑拆分：把请求状态显式化成 waiting_prefill、prefilling、waiting_decode、decoding、finished、failed；把 model runner 拆成 prefill 和 decode 两个接口；把 KV cache manager 从 model runner 里抽出来，并让 prefill 返回标准化 KV metadata，而不是返回某个本地对象引用。

第二步，我会拆 scheduler。Prefill scheduler 关注 prompt tokens、chunk size、prefix cache 和 input tokens/s；Decode scheduler 关注 active sequence、KV capacity、TPOT 和 streaming deadline。这样即使还在同一个进程里，也能先验证 P/D 调度策略。

第三步，引入 router 状态机，由 router 维护 request_id 到 prefill worker、decode worker、kv transfer handle 和 stream handle 的映射。然后把 worker 显式分成 prefill role 和 decode role。

落地顺序上，我会先做 same-node PD，验证 KV transfer、取消、失败清理、metrics 和 fallback；稳定后再做 cross-node PD，引入 RDMA/NIXL/Mooncake 这类传输后端和 topology-aware routing。

上线时我会按长 prompt、租户或流量比例灰度，保留 unified engine fallback，并重点观察 prefill latency、transfer p95/p99、decode wait、TPOT p99、KV free blocks、cancel cleanup 和 fallback rate。如果 transfer p99 或 KV 泄漏不可控，就不会扩大流量。
```

## 43.29 小练习

1. 画出一个单机 serving engine 的请求生命周期，并标出哪些状态需要为 PD 分离外置。
2. 设计一个 `KVMetadata` 结构，包含 decode worker 接管 KV 所需的字段。
3. 写一个简化的 PD router 状态机。
4. 说明为什么 request state 和 runtime temporary state 不能混在一起。
5. 设计 same-node PD 的最小原型。
6. 设计 cross-node PD 的灰度策略。
7. 给出三种 KV transfer 失败场景和对应 cleanup 方案。
8. 比较“prefill 完整结束后 transfer”和“chunk 级 transfer”的优缺点。
9. 设计一组压测指标，验证 PD 分离是否真的改善长 prompt 请求。
10. 写一个面试回答：为什么不能直接把 prefill 和 decode 拆成两个服务？

## 43.30 本章总结

从单机 serving engine 升级到 PD 分离，不是简单拆函数，也不是普通微服务改造。

真正要做的是把单进程内的 generate 状态机拆成跨 router、prefill worker、decode worker、KV transfer backend 和 request state store 的分布式状态机。

推荐升级顺序是：先显式化 request stage，再抽象 model runner、KV cache manager 和 KV metadata；然后拆 prefill/decode scheduler，引入 router 状态机和 worker role；最后从 same-node PD 逐步走向 cross-node PD。

上线 PD 分离前，必须具备完整观测、backpressure、cancel、cleanup、灰度和 fallback。否则 PD 分离很容易从性能优化变成稳定性风险。

到这里，本书第五部分“PD 分离与高级 Serving 架构”就完成了闭环。下一章开始进入第六部分：如何通过 nano-vLLM、tiny-llm、mini-sglang 这类教学项目学习推理框架源码，并把前面讲过的 engine 模块落到代码阅读和实战改造中。
