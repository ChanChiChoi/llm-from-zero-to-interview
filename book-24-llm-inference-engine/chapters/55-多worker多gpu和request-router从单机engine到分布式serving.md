# 第 55 章 多 worker、多 GPU 和 request router：从单机 engine 到分布式 serving

第 54 章我们把同步 engine loop 拆成了异步 serving 架构。

现在系统大概长这样：

```text
API Server
  -> Tokenizer Worker
  -> Engine Input Queue
  -> Engine Core
  -> Output Queue
  -> Stream Worker
```

这个结构可以把 CPU、网络和 GPU 主循环解耦。

但它仍然是单 engine worker。

当流量继续上涨，单个 engine worker 会遇到新瓶颈：

1. 单张 GPU 显存不够。
2. 单张 GPU 算力不够。
3. 单个 engine 的 KV cache block pool 不够。
4. 单个 tokenizer/output worker 不够。
5. 单个进程故障会影响全部请求。
6. 多模型、多租户、多 LoRA adapter 需要隔离。
7. 不同 GPU 的负载不均会导致尾延迟恶化。

所以生产级 serving 不能只考虑一个 engine。

它需要多个 worker、多张 GPU，以及一个负责路由请求的 request router。

本章讨论从单机 engine 走向分布式 serving 的第一步。

## 55.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 serving engine 如何从单 worker 扩展到多个 worker、多张 GPU 和 request router。

公开资料校准主要参考四类口径：

1. vLLM parallelism / scaling 文档对单卡、多卡 tensor parallel、pipeline parallel、多节点 serving、Ray / multiprocessing runtime 和网络通信边界的公开说明。
2. vLLM data parallel deployment 文档对 data parallel rank、internal / hybrid / external load balancing、每个 DP rank 的队列、API server 瓶颈和 KV cache aware routing 方向的公开口径。
3. SGLang server / runtime 文档对 TP / DP、router、worker、online serving、abort 和 metrics 的公开工程抽象。
4. 本书第 24、33、41、52、53、54 章对并行方式、vLLM / SGLang 架构对比、跨节点网络、统一调度循环、benchmark framework 和异步 serving 边界的教学抽象。

本章不实现真实分布式 RPC、真实 GPU 通信、Ray actor、Kubernetes service、NCCL collective、跨 worker KV cache 迁移、真实 DP rank 同步、MoE expert routing、生产级一致性协议或完整外部负载均衡器。我们只验证一个最小闭环：

```text
Router snapshot -> capability filter -> global admission -> sticky candidate -> load-aware fallback -> request-to-worker map -> heartbeat failure -> retry / fail-stream decision
```

第二轮新增 demo 的验收重点是：router 只做 worker 级路由，不直接修改 worker 内部 scheduler / KV 状态；round-robin 只能作为 baseline；路由要同时看能力、健康、KV 空间、队列、延迟、preemption 和 prefix locality；sticky 过载时必须 fallback；worker failure 后没有输出的请求可以重试，已经 streaming 的请求必须显式失败。

## 55.1 本章目标

读完本章，你应该能讲清：

1. 多 worker serving 和单 worker serving 的核心差异。
2. request router 应该根据哪些信息分配请求。
3. 为什么不能只用 round-robin。
4. worker load metrics 应该包含哪些指标。
5. KV cache、prefix cache 和 sticky routing 的关系。
6. 多 GPU 场景下 tensor parallel 和 replica parallel 的区别。
7. request 失败、worker 崩溃和重试应该怎么处理。
8. router、worker、API server 的状态边界。
9. 分布式 serving 的 benchmark 应该额外关注什么。

本章不是讲大规模集群调度系统。

我们先建立一个最小但正确的多 worker serving 心智模型。

## 55.2 为什么需要多个 worker

一个 engine worker 通常绑定一个模型实例。

最简单的情况下，一个 worker 占用一张 GPU。

```text
Worker 0 -> GPU 0 -> Model Replica 0
```

当一张 GPU 不够时，可以增加 replica：

```text
Worker 0 -> GPU 0 -> Model Replica 0
Worker 1 -> GPU 1 -> Model Replica 1
Worker 2 -> GPU 2 -> Model Replica 2
Worker 3 -> GPU 3 -> Model Replica 3
```

每个 worker 都有自己的：

1. scheduler。
2. waiting/running queues。
3. KV block manager。
4. prefix cache。
5. engine loop。
6. metrics。
7. output stream 状态。

这叫 replica parallel。

每个 replica 都能独立处理完整请求。

router 的任务是决定：

```text
这个新请求应该发给哪个 worker？
```

## 55.3 最小多 worker 架构

一个最小多 worker serving 架构可以长这样：

```text
Client
  |
  v
API Server
  |
  v
Request Router
  |
  +--> Worker 0 -> GPU 0
  +--> Worker 1 -> GPU 1
  +--> Worker 2 -> GPU 2
  +--> Worker 3 -> GPU 3
```

如果沿用第 54 章的异步拆分，每个 worker 内部仍然有自己的 input/output queue：

```text
Request Router
  |
  +--> Worker 0 Input Queue -> Engine 0 -> Worker 0 Output Queue
  +--> Worker 1 Input Queue -> Engine 1 -> Worker 1 Output Queue
  +--> Worker 2 Input Queue -> Engine 2 -> Worker 2 Output Queue
  +--> Worker 3 Input Queue -> Engine 3 -> Worker 3 Output Queue
```

router 不应该直接修改 worker 内部 scheduler。

router 只负责选择 worker，并把 tokenized request 发送给该 worker。

worker 内部怎么排队、怎么分配 KV、怎么 preempt，仍然由 worker 自己决定。

## 55.4 WorkerHandle

router 需要维护 worker 的基本信息。

可以定义一个 `WorkerHandle`：

```python
from dataclasses import dataclass


@dataclass
class WorkerLoad:
    waiting_queue_size: int
    running_queue_size: int
    kv_free_blocks: int
    kv_used_blocks: int
    engine_input_queue_size: int
    recent_ttft_p90_ms: float
    recent_tpot_p90_ms: float
    preemption_rate: float
    healthy: bool


@dataclass
class WorkerHandle:
    worker_id: str
    model_name: str
    gpu_id: int
    input_queue: "Queue[TokenizedRequest]"
    load: WorkerLoad
```

router 不需要知道 worker 的所有内部细节。

但它至少要知道：

1. worker 是否健康。
2. worker 当前是否过载。
3. worker 是否有足够 KV 空间接收请求。
4. worker 最近的延迟是否恶化。
5. worker 是否匹配请求的模型或 adapter。

## 55.5 为什么 round-robin 不够

最简单的路由是 round-robin：

```python
class RoundRobinRouter:
    def __init__(self, workers):
        self.workers = workers
        self.next_idx = 0

    def route(self, req):
        worker = self.workers[self.next_idx]
        self.next_idx = (self.next_idx + 1) % len(self.workers)
        return worker
```

它的问题是：

```text
它假设每个请求成本一样，每个 worker 状态一样。
```

但 LLM serving 里这个假设通常不成立。

两个请求可能差别很大：

```text
请求 A: prompt 32 tokens, output 64 tokens
请求 B: prompt 16000 tokens, output 2048 tokens
```

如果 round-robin 把多个长请求连续打到同一个 worker，这个 worker 的 KV blocks 和 prefill 时间会被打爆。

另一个 worker 可能还很空。

round-robin 还忽略：

1. worker 当前 waiting queue。
2. worker 当前 running queue。
3. KV free blocks。
4. prefix cache 命中可能性。
5. 最近 preemption rate。
6. GPU 异构性能。
7. worker 是否刚经历 OOM 或重启。

所以 round-robin 只能作为 baseline。

生产系统至少需要 load-aware routing。

## 55.6 load-aware routing

load-aware routing 的目标是把请求发给更可能快速完成的 worker。

最小策略可以是打分：

```python
def estimate_request_blocks(req, block_size: int) -> int:
    total_tokens = len(req.prompt_token_ids) + req.max_tokens
    return (total_tokens + block_size - 1) // block_size


def score_worker(worker: WorkerHandle, req, block_size: int) -> float:
    load = worker.load
    if not load.healthy:
        return float("inf")

    needed_blocks = estimate_request_blocks(req, block_size)
    if load.kv_free_blocks < needed_blocks:
        return float("inf")

    score = 0.0
    score += load.waiting_queue_size * 10.0
    score += load.running_queue_size * 5.0
    score += load.engine_input_queue_size * 2.0
    score += load.preemption_rate * 100.0
    score += load.recent_ttft_p90_ms * 0.01
    score += load.recent_tpot_p90_ms * 0.05
    return score


def route(req, workers, block_size: int):
    candidates = [w for w in workers if w.model_name == req.model_name]
    scored = [(score_worker(w, req, block_size), w) for w in candidates]
    score, worker = min(scored, key=lambda x: x[0])
    if score == float("inf"):
        raise OverloadedError("no worker can admit request")
    return worker
```

这个策略很粗糙。

但它体现了几个关键点：

1. 先过滤健康状态和模型匹配。
2. 估算请求会消耗多少 KV blocks。
3. 避免把请求发给明显没有 KV 空间的 worker。
4. 使用 queue size、延迟和 preemption 作为负载信号。
5. 过载时明确拒绝，而不是盲目排队。

## 55.7 请求成本估算

LLM request 的成本不是一个数字。

它至少包括：

1. prefill 成本：主要由 prompt tokens 决定。
2. decode 成本：主要由 output tokens 和并发数决定。
3. KV cache 成本：由 prompt tokens + generated tokens 决定。
4. streaming 成本：由输出 token 数和客户端消费速度决定。

可以用简单结构记录估算：

```python
@dataclass
class RequestCost:
    prompt_tokens: int
    max_output_tokens: int
    estimated_total_tokens: int
    estimated_kv_blocks: int
```

构造方式：

```python
def estimate_cost(req, block_size: int) -> RequestCost:
    prompt_tokens = len(req.prompt_token_ids)
    max_output_tokens = req.max_tokens
    total = prompt_tokens + max_output_tokens
    blocks = (total + block_size - 1) // block_size
    return RequestCost(
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
        estimated_total_tokens=total,
        estimated_kv_blocks=blocks,
    )
```

这个估算偏保守，因为请求可能提前遇到 EOS。

但对于 admission control 和 routing，保守估算通常比乐观估算更安全。

## 55.8 least-loaded 也不总是最优

直觉上，把请求发给当前最空的 worker 很合理。

但在 LLM serving 里，least-loaded 也可能犯错。

原因是 prefix cache。

假设很多请求共享相同 system prompt：

```text
你是一个专业代码助手。请遵循以下规则：...
```

如果前面请求都打到 Worker 0，Worker 0 的 prefix cache 里已经有这段 prompt 的 KV blocks。

新请求继续发给 Worker 0，可能直接复用 prefix。

如果发给最空的 Worker 2，就要重新 prefill。

所以 router 有一个 trade-off：

```text
负载均衡 vs prefix cache locality
```

完全按负载路由，可能损失 prefix cache 命中。

完全按 prefix 路由，可能让某个 worker 过热。

## 55.9 sticky routing

sticky routing 指的是相似请求尽量打到同一个 worker。

常见 sticky key 包括：

1. tenant_id。
2. user_id。
3. conversation_id。
4. system_prompt_hash。
5. prefix_hash。
6. lora_adapter_id。

例如按 prefix hash 选择 worker：

```python
def choose_by_prefix_hash(req, workers):
    idx = stable_hash(req.prefix_hash) % len(workers)
    return workers[idx]
```

这样做有利于 prefix cache。

但它也可能造成热点。

比如某个热门 tenant 或热门 prompt 占大部分流量。

所以实际 router 通常会把 sticky 和 load-aware 结合起来。

## 55.10 sticky + load-aware

一种实用策略是：

1. 先根据 sticky key 得到首选 worker。
2. 如果首选 worker 健康且不过载，就发给它。
3. 如果首选 worker 过载，就在候选 worker 里选负载最低的。

伪代码：

```python
def route_with_sticky(req, workers, block_size: int):
    candidates = [w for w in workers if w.model_name == req.model_name]
    preferred = candidates[stable_hash(req.prefix_hash) % len(candidates)]

    preferred_score = score_worker(preferred, req, block_size)
    if preferred_score < 100.0:
        return preferred

    return route(req, candidates, block_size)
```

这里的 `100.0` 只是示意。

真实系统会基于 queue size、KV 空间、TTFT、preemption 和错误率做阈值。

sticky routing 的目标不是绝对粘住。

而是在不伤害尾延迟的前提下提高 cache locality。

## 55.11 多模型和多 adapter 路由

真实 serving 往往不止一个模型。

可能有：

1. base model A。
2. base model B。
3. chat model。
4. code model。
5. embedding model。
6. 多个 LoRA adapter。

router 必须先做能力匹配：

```python
def can_serve(worker, req) -> bool:
    if worker.model_name != req.model_name:
        return False
    if req.adapter_id is not None and req.adapter_id not in worker.loaded_adapters:
        return False
    return True
```

如果 adapter 可以动态加载，还要考虑加载成本。

这时 router 可能优先选择已经加载 adapter 的 worker。

```text
已加载 adapter 的 worker：低延迟，高 locality。
未加载 adapter 的 worker：需要加载，可能阻塞或排队。
```

这和 prefix cache 是类似问题。

都是 locality 与 load balance 的权衡。

## 55.12 Replica parallel 和 tensor parallel

多 GPU serving 有两类常见扩展方式。

### 55.12.1 Replica parallel

每张 GPU 或每组 GPU 放一份完整模型副本。

```text
Worker 0 -> GPU 0 -> Full Model
Worker 1 -> GPU 1 -> Full Model
Worker 2 -> GPU 2 -> Full Model
Worker 3 -> GPU 3 -> Full Model
```

优点：

1. 架构简单。
2. worker 之间请求独立。
3. 容错容易。
4. router 只需要做请求级负载均衡。

缺点：

1. 单个请求只能用一张 GPU 的显存和算力。
2. 模型太大时放不下一张 GPU。
3. 每张 GPU 都要加载完整权重，显存重复。

### 55.12.2 Tensor parallel

一个模型被切到多张 GPU 上，一个请求需要多张 GPU 协同完成。

```text
Worker 0 -> GPU 0 + GPU 1 + GPU 2 + GPU 3 -> Sharded Model
```

优点：

1. 可以服务单卡放不下的大模型。
2. 单请求可以使用多卡算力。
3. 对超大模型是必要手段。

缺点：

1. GPU 之间通信开销高。
2. 一个 worker 故障可能影响整组 GPU。
3. 调度和部署更复杂。
4. router 看到的是一个多 GPU worker，而不是多个独立 GPU。

从 router 视角看，tensor parallel worker 仍然是一个可路由目标：

```text
Request Router -> TP Worker 0(GPU 0,1,2,3)
Request Router -> TP Worker 1(GPU 4,5,6,7)
```

也就是说：

```text
router 做 worker 级路由，worker 内部处理模型并行。
```

## 55.13 Pipeline parallel 和 expert parallel

更复杂的模型并行还有 pipeline parallel 和 expert parallel。

### 55.13.1 Pipeline parallel

模型层被分到不同 GPU。

```text
GPU 0: layers 0-15
GPU 1: layers 16-31
GPU 2: layers 32-47
GPU 3: layers 48-63
```

它能降低单卡显存压力。

但在线 decode 场景下，pipeline bubble 和小 batch 会影响效率。

### 55.13.2 Expert parallel

MoE 模型里，不同 expert 分布在不同 GPU。

每个 token 会路由到部分 expert。

这会引入 token 级别的跨卡通信和负载均衡问题。

这些并行方式通常由模型执行框架内部处理。

本章重点仍然是 request router。

router 不需要知道每一层或每个 expert 怎么切。

它只需要知道 worker 的能力、健康状态和负载。

## 55.14 worker health check

多 worker 系统必须有健康检查。

worker 可能出现：

1. 进程崩溃。
2. GPU OOM。
3. CUDA error。
4. NCCL hang。
5. engine loop 卡死。
6. 延迟异常升高。
7. KV block 泄漏。
8. output queue 堵塞。

router 需要定期获取 worker heartbeat：

```python
@dataclass
class WorkerHeartbeat:
    worker_id: str
    timestamp: float
    healthy: bool
    error: str | None
    load: WorkerLoad
```

如果 heartbeat 超时，router 应该把 worker 标记为 unhealthy：

```python
def update_health(worker, now, heartbeat_timeout_s: float):
    if now - worker.last_heartbeat_time > heartbeat_timeout_s:
        worker.load.healthy = False
```

unhealthy worker 不应该接收新请求。

已经在 unhealthy worker 上执行的请求，要根据语义决定是否重试。

## 55.15 请求重试

请求重试不是总是安全的。

对于 LLM generation，请求可能已经向 client stream 了一部分 token。

如果这时 worker 崩溃，再换 worker 重试，会产生几个问题：

1. 新 worker 生成的文本不一定和前面一致。
2. sampling 有随机性。
3. client 已经收到部分输出，不能无缝替换。
4. prefix cache 和 KV 状态不能跨 worker 自动迁移。

所以重试策略要分阶段。

### 55.15.1 还没开始输出

如果请求还在 router、input queue 或 prefill 前，没有任何 token 发给 client，可以安全重试到其他 worker。

```text
safe to retry
```

### 55.15.2 已经输出 token

如果已经向 client 发送过 token，通常不能透明重试。

可以选择：

1. 返回错误并结束 stream。
2. 告诉上游请求失败。
3. 如果业务允许，重新开始一次新请求。

但不要假装是同一个连续 stream。

### 55.15.3 deterministic decode

如果 temperature=0，并且模型、版本、随机种子、kernel 行为都完全确定，理论上可以更接近可重试。

但生产上仍然要谨慎。

跨 worker 重放通常没有 KV 状态，只能重新 prefill。

## 55.16 worker crash 后如何清理

worker 崩溃时，router 和 API server 要处理三类状态：

1. 未发送到 worker 的请求。
2. 已发送但未输出的请求。
3. 已经 streaming 的请求。

可以维护 request location：

```python
@dataclass
class RoutedRequestState:
    request_id: str
    worker_id: str | None
    stream_started: bool
    status: str
```

worker crash 后：

```python
def handle_worker_failure(worker_id):
    for state in routed_requests_by_worker(worker_id):
        if not state.stream_started:
            retry_on_another_worker(state.request_id)
        else:
            fail_stream(state.request_id, "worker failed")
```

这里还是同一个原则：

```text
没有输出过的请求可以重试，已经输出过的请求应该显式失败。
```

## 55.17 分布式状态边界

多 worker 系统更需要明确状态所有权。

可以这样划分：

### 55.17.1 API server owns connection state

API server 负责：

1. client 连接。
2. HTTP/gRPC stream。
3. request_id。
4. auth 和 quota。
5. client disconnect。

### 55.17.2 Router owns routing state

router 负责：

1. worker 列表。
2. worker health。
3. worker load snapshot。
4. request 到 worker 的映射。
5. retry 决策。
6. admission reject。

### 55.17.3 Worker owns engine state

worker 负责：

1. scheduler queue。
2. RequestState。
3. KV block table。
4. prefix cache。
5. model runner。
6. sampling。
7. worker 内部 cancel 和 cleanup。

router 不应该直接释放 worker 的 KV blocks。

worker 不应该直接决定全局流量分配。

API server 不应该越过 router 随机选择 worker。

状态边界清楚，多 worker 系统才不会变成竞态地狱。

## 55.18 输出路径怎么走

多 worker 的输出路径有两种常见设计。

### 55.18.1 worker 直接回 API server

```text
API Server -> Router -> Worker
API Server <- Worker
```

优点：

1. 输出路径短。
2. router 不需要处理大量 token stream。
3. router 只做控制面。

缺点：

1. API server 要维护 request 到 worker 的连接。
2. worker 和 API server 之间需要 stream 通道。
3. worker failure 处理稍复杂。

### 55.18.2 worker 输出经过 router

```text
API Server -> Router -> Worker
API Server <- Router <- Worker
```

优点：

1. router 可以统一记录输出状态。
2. failure 和 retry 状态集中。
3. API server 只和 router 交互。

缺点：

1. router 可能成为 token stream 瓶颈。
2. router 数据面压力很大。
3. 大量输出 token 会增加网络和内存压力。

很多系统会让 router 偏控制面，避免所有 token 都经过 router。

但教学实现可以先让输出经过 router，方便理解 request 状态。

## 55.19 全局 admission control

第 54 章讨论的是单 worker admission control。

多 worker 系统需要全局 admission control。

它要回答：

```text
整个 serving 集群现在还能不能接这个请求？
```

可以检查：

1. 健康 worker 数量。
2. 可服务该模型的 worker 数量。
3. 总 waiting queue size。
4. 总 KV free blocks。
5. 最大 worker preemption rate。
6. 全局 p99 TTFT。
7. router queue length。
8. tokenizer pool queue length。

最小策略：

```python
def global_admit(req, workers, config) -> bool:
    candidates = [w for w in workers if can_serve(w, req) and w.load.healthy]
    if not candidates:
        return False

    if sum(w.load.waiting_queue_size for w in candidates) > config.max_total_waiting:
        return False

    if max(w.load.preemption_rate for w in candidates) > config.max_preemption_rate:
        return False

    return True
```

全局 admission control 和 routing 应该协同。

不要先全局接收请求，再发现没有 worker 能承载。

## 55.20 多 worker benchmark

多 worker benchmark 不能只看总 tokens/s。

还要看负载是否均衡。

新增指标包括：

```text
router_qps
router_route_latency_ms
router_reject_total
worker_selected_total{worker_id}
worker_inflight_requests{worker_id}
worker_waiting_queue_size{worker_id}
worker_running_queue_size{worker_id}
worker_kv_free_blocks{worker_id}
worker_ttft_p99_ms{worker_id}
worker_tpot_p99_ms{worker_id}
worker_preemption_total{worker_id}
worker_failure_total{worker_id}
request_retry_total
sticky_route_hit_total
sticky_route_fallback_total
```

重点观察：

1. 请求是否集中到少数 worker。
2. 某些 worker 是否长期 queue 高。
3. prefix cache 命中是否因为负载均衡下降。
4. sticky fallback 是否过多。
5. worker failure 后是否正确停止路由。
6. retry 是否造成重复输出。
7. router 本身是否成为瓶颈。

多 worker benchmark 的报告应该按 worker 展示指标。

只给一个全局平均值会掩盖热点。

## 55.21 常见错误

错误一：只用 round-robin。

```text
结果：长请求和短请求成本差异被忽略，某些 worker 容易被长 prompt 打爆。
```

错误二：router 不看 KV free blocks。

```text
结果：请求被发到没有 KV 空间的 worker，导致 preemption 或 reject 增加。
```

错误三：为了 prefix cache 完全 sticky。

```text
结果：热门 tenant 或热门 prompt 把单个 worker 打成热点。
```

错误四：worker 崩溃后仍然继续路由。

```text
结果：请求不断失败，排队延迟恶化。
```

错误五：已输出 token 的请求被透明重试。

```text
结果：client 收到不连续或重复的文本。
```

错误六：router 直接修改 worker 内部状态。

```text
结果：跨进程状态竞态，scheduler 和 KV block table 不一致。
```

错误七：只看全局平均吞吐。

```text
结果：少数 worker 的 p99 延迟和热点被掩盖。
```

错误八：router 同时承担所有 token 数据面。

```text
结果：router 成为 streaming 瓶颈，扩容 worker 也无效。
```

错误九：不区分 replica parallel 和 tensor parallel。

```text
结果：错误地把一个 tensor parallel worker 当成多个独立 worker 路由。
```

错误十：没有全局 admission control。

```text
结果：所有 worker 都已经过载，router 仍然继续接收请求，最终全局 timeout。
```

## 55.22 面试高频问题

问题一：多 GPU LLM serving 里，router 的职责是什么？

回答要点：router 负责根据模型能力、worker 健康、负载、KV 空间、延迟、preemption、prefix cache locality 等信息选择 worker。它不应该直接修改 worker 内部 scheduler 或 KV 状态。router 还负责全局 admission、worker failure 处理和必要的 retry 决策。

问题二：为什么 round-robin 不适合 LLM serving？

回答要点：LLM 请求成本差异很大，短 prompt 和长 prompt 对 prefill、decode 和 KV cache 的压力完全不同。worker 之间的 queue、KV free blocks、preemption rate、prefix cache 状态也不同。round-robin 忽略这些信号，容易造成热点和尾延迟恶化。

问题三：prefix cache 会如何影响路由？

回答要点：如果相似请求打到同一个 worker，就可能复用 prefix cache，降低 prefill 成本和 TTFT。但过度 sticky 会造成热点。所以路由需要在 cache locality 和 load balance 之间权衡，常见做法是 sticky 优先，过载时 fallback 到 load-aware routing。

问题四：replica parallel 和 tensor parallel 有什么区别？

回答要点：replica parallel 是每个 worker 有完整模型副本，请求级别分配到不同 worker，架构简单但单请求受单卡或单副本限制。tensor parallel 是一个模型切到多张 GPU，一个请求需要多卡协同，能服务更大模型但通信和部署复杂。从 router 视角看，一个 TP group 通常是一个 worker。

问题五：worker 崩溃后请求能不能重试？

回答要点：如果请求还没向 client 输出 token，通常可以重试到其他 worker。如果已经 streaming 了一部分 token，透明重试通常不安全，因为 sampling 和 KV 状态无法无缝恢复，可能产生重复或不连续文本。此时应该显式失败 stream，或者由业务层重新发起请求。

## 55.23 标准回答模板

如果面试官问“你会怎么设计多 GPU LLM serving 的 request router”，可以这样回答：

```text
我会把多 GPU serving 先抽象成多个 worker，每个 worker 可能是一张 GPU 上的完整模型副本，也可能是一个 tensor parallel group。router 的职责是做 worker 级路由，而不是直接干预 worker 内部 scheduler。每个 worker 上报 heartbeat 和 load，包括 waiting/running queue、engine input queue、KV free/used blocks、最近 TTFT/TPOT、preemption rate、错误状态等。

路由策略不会只用 round-robin，因为 LLM 请求成本差异很大。我会先做能力过滤，比如 model_name、adapter_id、worker health，然后估算请求成本，包括 prompt tokens、max output tokens 和预计 KV blocks。接着用 load-aware score 选择 worker，避免把请求发给 KV 空间不足、queue 很长或 preemption 很高的 worker。

同时我会考虑 prefix cache locality。对于共享 system prompt、tenant、conversation 或 adapter 的请求，可以用 sticky routing 提高 cache 命中，但 sticky 不能绝对化。如果首选 worker 过载，就要 fallback 到负载更低的 worker，避免热点。

故障处理上，worker heartbeat 超时或报错后，router 立即停止给它发新请求。对于已经路由过去的请求，如果还没有输出 token，可以重试到其他 worker；如果已经开始 streaming，通常不能透明重试，只能显式失败或让业务层重新发起。最后要有全局 admission control，在所有候选 worker 都过载时及时 reject，而不是让请求无限排队。
```

## 55.24 Multi-Worker Router 公式、故障门禁和可运行 demo

多 worker 路由先把请求成本估算成 KV block 数：

```math
C_i^{\mathrm{block}}=\left\lceil\frac{P_i+O_i^{\max}}{B}\right\rceil
```

其中 `P_i` 是 prompt tokens，`O_i^{\max}` 是最大输出 token 数，`B` 是 block size。

worker 的可路由条件可以写成：

```math
H_w M_{w,i} A_{w,i}\mathbf{1}[F_w\ge C_i^{\mathrm{block}}]=1
```

其中 `H_w` 表示健康，`M_{w,i}` 表示模型匹配，`A_{w,i}` 表示 adapter 可用，`F_w` 表示 free KV blocks。

一个教学版 load score 可以写成：

```math
S(w,i)=\alpha_q Q_w+\alpha_r R_w+\alpha_e E_w+\alpha_p P_w^{\mathrm{preempt}}+\alpha_f L_w^{\mathrm{first}}+\alpha_t L_w^{\mathrm{tok}}+\alpha_c C_i^{\mathrm{block}}
```

sticky routing 不是绝对粘住，而是满足阈值才使用首选 worker：

```math
w_i=\begin{cases}
w_i^{\mathrm{sticky}}, & S(w_i^{\mathrm{sticky}},i)\le \tau \\
\arg\min_{w\in \mathcal{C}_i} S(w,i), & S(w_i^{\mathrm{sticky}},i)>\tau
\end{cases}
```

全局 admission control 至少要保证候选 worker 总容量足够：

```math
\sum_{w\in \mathcal{C}_i}F_w\ge C_i^{\mathrm{block}}
```

worker failure 后的 retry 门禁可以写成：

```math
G_{\mathrm{retry},i}=\mathbf{1}[B_i^{\mathrm{stream}}=0]
```

也就是没有向 client 输出过 token 的请求才允许透明重试。

最终用一个组合门禁收束：

```math
G_{\mathrm{router}}=G_{\mathrm{cap}}G_{\mathrm{admit}}G_{\mathrm{load}}G_{\mathrm{sticky}}G_{\mathrm{health}}G_{\mathrm{retry}}G_{\mathrm{state}}G_{\mathrm{metrics}}
```

下面这个 0 依赖 demo 把这些约束串起来：

```python
from dataclasses import dataclass, field
from math import ceil, inf


def stable_hash(text: str) -> int:
    total = 0
    for index, char in enumerate(text):
        total += (index + 1) * ord(char)
    return total


@dataclass
class WorkerLoad:
    waiting_queue_size: int
    running_queue_size: int
    kv_free_blocks: int
    kv_used_blocks: int
    engine_input_queue_size: int
    recent_ttft_p90_ms: float
    recent_tpot_p90_ms: float
    preemption_rate: float
    healthy: bool
    last_heartbeat_s: float
    error: str | None = None


@dataclass
class WorkerHandle:
    worker_id: str
    model_name: str
    gpu_group: tuple[int, ...]
    loaded_adapters: set[str]
    hot_prefixes: set[str]
    load: WorkerLoad
    selected_total: int = 0


@dataclass
class TokenizedRequest:
    request_id: str
    model_name: str
    prompt_tokens: int
    max_output_tokens: int
    prefix_hash: str
    adapter_id: str | None = None


@dataclass
class RoutedRequestState:
    request_id: str
    worker_id: str | None
    stream_started: bool
    status: str
    events: list[str] = field(default_factory=list)


@dataclass
class RouterConfig:
    block_size: int
    sticky_score_threshold: float
    heartbeat_timeout_s: float
    max_total_waiting: int
    max_preemption_rate: float


class ToyMultiWorkerRouter:
    def __init__(self, workers: list[WorkerHandle], config: RouterConfig):
        self.workers = workers
        self.config = config
        self.rr_index = 0
        self.states: dict[str, RoutedRequestState] = {}
        self.metrics = {
            "router_reject_total": 0,
            "sticky_route_hit_total": 0,
            "sticky_route_fallback_total": 0,
            "request_retry_total": 0,
            "stream_fail_total": 0,
            "worker_failure_total": 0,
            "capability_filter_total": 0,
        }

    def estimate_blocks(self, req: TokenizedRequest) -> int:
        return ceil((req.prompt_tokens + req.max_output_tokens) / self.config.block_size)

    def can_serve(self, worker: WorkerHandle, req: TokenizedRequest) -> bool:
        if worker.model_name != req.model_name:
            return False
        if req.adapter_id is not None and req.adapter_id not in worker.loaded_adapters:
            return False
        return True

    def candidates(self, req: TokenizedRequest) -> list[WorkerHandle]:
        result = [worker for worker in self.workers if self.can_serve(worker, req)]
        self.metrics["capability_filter_total"] += 1
        return result

    def score_worker(self, worker: WorkerHandle, req: TokenizedRequest) -> float:
        load = worker.load
        needed_blocks = self.estimate_blocks(req)
        if not load.healthy or load.kv_free_blocks < needed_blocks:
            return inf

        locality_penalty = 0.0 if req.prefix_hash in worker.hot_prefixes else 10.0
        score = 0.0
        score += load.waiting_queue_size * 10.0
        score += load.running_queue_size * 6.0
        score += load.engine_input_queue_size * 3.0
        score += load.preemption_rate * 100.0
        score += load.recent_ttft_p90_ms * 0.01
        score += load.recent_tpot_p90_ms * 0.05
        score += needed_blocks * 0.5
        score += locality_penalty
        return round(score, 3)

    def global_admit(self, req: TokenizedRequest) -> bool:
        candidates = [w for w in self.candidates(req) if w.load.healthy]
        needed_blocks = self.estimate_blocks(req)
        if not candidates:
            return False
        if sum(w.load.waiting_queue_size for w in candidates) > self.config.max_total_waiting:
            return False
        if max(w.load.preemption_rate for w in candidates) > self.config.max_preemption_rate:
            return False
        if sum(w.load.kv_free_blocks for w in candidates) < needed_blocks:
            return False
        return True

    def round_robin_route(self, req: TokenizedRequest) -> WorkerHandle | None:
        candidates = self.candidates(req)
        if not candidates:
            return None
        worker = candidates[self.rr_index % len(candidates)]
        self.rr_index += 1
        return worker

    def sticky_preferred(self, req: TokenizedRequest, candidates: list[WorkerHandle]) -> WorkerHandle:
        for worker in candidates:
            if req.prefix_hash in worker.hot_prefixes:
                return worker
        return candidates[stable_hash(req.prefix_hash) % len(candidates)]

    def load_aware_route(
        self, req: TokenizedRequest, exclude_worker_id: str | None = None
    ) -> WorkerHandle | None:
        candidates = [w for w in self.candidates(req) if w.worker_id != exclude_worker_id]
        scored = [(self.score_worker(worker, req), worker) for worker in candidates]
        viable = [(score, worker) for score, worker in scored if score < inf]
        if not viable:
            self.metrics["router_reject_total"] += 1
            return None
        return min(viable, key=lambda item: (item[0], item[1].worker_id))[1]

    def route_with_sticky(self, req: TokenizedRequest) -> WorkerHandle | None:
        if not self.global_admit(req):
            self.metrics["router_reject_total"] += 1
            return None

        candidates = self.candidates(req)
        preferred = self.sticky_preferred(req, candidates)
        preferred_score = self.score_worker(preferred, req)
        if preferred_score <= self.config.sticky_score_threshold:
            self.metrics["sticky_route_hit_total"] += 1
            return preferred

        self.metrics["sticky_route_fallback_total"] += 1
        return self.load_aware_route(req)

    def commit_route(
        self, req: TokenizedRequest, worker: WorkerHandle, stream_started: bool
    ) -> RoutedRequestState:
        worker.selected_total += 1
        worker.load.waiting_queue_size += 1
        worker.load.kv_free_blocks -= self.estimate_blocks(req)
        state = RoutedRequestState(
            request_id=req.request_id,
            worker_id=worker.worker_id,
            stream_started=stream_started,
            status="routed",
            events=[f"route:{worker.worker_id}"],
        )
        self.states[req.request_id] = state
        return state

    def update_health(self, now_s: float) -> list[str]:
        unhealthy = []
        for worker in self.workers:
            stale = now_s - worker.load.last_heartbeat_s > self.config.heartbeat_timeout_s
            if stale and worker.load.healthy:
                worker.load.healthy = False
                worker.load.error = "heartbeat_timeout"
                unhealthy.append(worker.worker_id)
        return unhealthy

    def handle_worker_failure(self, worker_id: str, retry_requests: dict[str, TokenizedRequest]):
        self.metrics["worker_failure_total"] += 1
        for state in list(self.states.values()):
            if state.worker_id != worker_id or state.status != "routed":
                continue
            if state.stream_started:
                state.status = "failed_stream"
                state.events.append("fail_stream:worker_failed")
                self.metrics["stream_fail_total"] += 1
                continue

            retry_req = retry_requests[state.request_id]
            retry_worker = self.load_aware_route(retry_req, exclude_worker_id=worker_id)
            if retry_worker is None:
                state.status = "failed_before_stream"
                state.events.append("retry_failed:no_worker")
                continue
            state.worker_id = retry_worker.worker_id
            state.status = "retried"
            state.events.append(f"retry:{retry_worker.worker_id}")
            retry_worker.selected_total += 1
            self.metrics["request_retry_total"] += 1

    def worker_selection_counts(self) -> dict[str, int]:
        return {worker.worker_id: worker.selected_total for worker in self.workers}


workers = [
    WorkerHandle(
        worker_id="w0",
        model_name="chat",
        gpu_group=(0,),
        loaded_adapters={"lora-a"},
        hot_prefixes={"sys-a"},
        load=WorkerLoad(5, 4, 24, 40, 2, 900.0, 90.0, 0.20, True, 12.0),
    ),
    WorkerHandle(
        worker_id="w1",
        model_name="chat",
        gpu_group=(1,),
        loaded_adapters={"lora-a"},
        hot_prefixes=set(),
        load=WorkerLoad(1, 1, 32, 20, 1, 320.0, 35.0, 0.02, True, 0.0),
    ),
    WorkerHandle(
        worker_id="w2",
        model_name="chat",
        gpu_group=(2, 3),
        loaded_adapters={"lora-a"},
        hot_prefixes={"sys-c"},
        load=WorkerLoad(0, 0, 64, 0, 0, 0.0, 0.0, 0.00, False, 12.0, "starting"),
    ),
    WorkerHandle(
        worker_id="w3",
        model_name="code",
        gpu_group=(4,),
        loaded_adapters=set(),
        hot_prefixes=set(),
        load=WorkerLoad(0, 1, 40, 12, 0, 260.0, 28.0, 0.01, True, 12.0),
    ),
]

config = RouterConfig(
    block_size=128,
    sticky_score_threshold=80.0,
    heartbeat_timeout_s=10.0,
    max_total_waiting=12,
    max_preemption_rate=0.50,
)
router = ToyMultiWorkerRouter(workers, config)

req_a = TokenizedRequest("r1", "chat", 800, 256, "sys-a", "lora-a")
req_b = TokenizedRequest("r2", "chat", 64, 64, "sys-a", "lora-a")
req_big = TokenizedRequest("r3", "chat", 9000, 4096, "sys-big", "lora-a")
req_code = TokenizedRequest("r4", "code", 128, 128, "code-a", None)

round_robin_choice = router.round_robin_route(req_a).worker_id
load_aware_choice = router.load_aware_route(req_a).worker_id

worker_for_a = router.route_with_sticky(req_a)
state_a = router.commit_route(req_a, worker_for_a, stream_started=False)

worker_for_b = router.route_with_sticky(req_b)
state_b = router.commit_route(req_b, worker_for_b, stream_started=True)

big_admitted = router.global_admit(req_big)
if not big_admitted:
    router.metrics["router_reject_total"] += 1

code_worker = router.route_with_sticky(req_code)
if code_worker is not None:
    router.commit_route(req_code, code_worker, stream_started=False)

unhealthy_after_timeout = router.update_health(now_s=15.0)
router.handle_worker_failure(
    "w1", {"r1": req_a, "r2": req_b, "r3": req_big, "r4": req_code}
)

state_rows = {
    request_id: {
        "worker_id": state.worker_id,
        "stream_started": state.stream_started,
        "status": state.status,
        "events": state.events,
    }
    for request_id, state in sorted(router.states.items())
}
selection_counts = router.worker_selection_counts()
active_counts = [count for count in selection_counts.values() if count > 0]
selection_imbalance = max(active_counts) - min(active_counts) if active_counts else 0

summary = {
    "round_robin_choice_for_long": round_robin_choice,
    "load_aware_choice_for_long": load_aware_choice,
    "sticky_preferred_for_sys_a": "w0",
    "actual_worker_for_r1": worker_for_a.worker_id,
    "actual_worker_for_r2": worker_for_b.worker_id,
    "big_request_admitted": big_admitted,
    "code_worker": code_worker.worker_id if code_worker else None,
    "unhealthy_after_timeout": unhealthy_after_timeout,
    "state_rows": state_rows,
    "selection_counts": selection_counts,
    "selection_imbalance": selection_imbalance,
    "metrics": router.metrics,
}

gates = {
    "capability_filter_ready": code_worker.worker_id == "w3",
    "global_admission_ready": big_admitted is False,
    "load_aware_ready": round_robin_choice == "w0" and load_aware_choice == "w1",
    "sticky_fallback_ready": worker_for_a.worker_id == "w1"
    and router.metrics["sticky_route_fallback_total"] >= 1,
    "health_check_ready": unhealthy_after_timeout == ["w1"],
    "retry_gate_ready": state_rows["r1"]["status"] == "retried"
    and state_rows["r2"]["status"] == "failed_stream",
    "state_boundary_ready": all("route:" in row["events"][0] for row in state_rows.values()),
    "metrics_ready": router.metrics["request_retry_total"] == 1
    and router.metrics["stream_fail_total"] == 1,
}
gates["multi_worker_router_gate"] = all(gates.values())

print("multi_worker_router_summary=", summary)
print("multi_worker_router_gates=", gates)
```

一次运行的核心输出类似：

```text
multi_worker_router_summary= {'round_robin_choice_for_long': 'w0', 'load_aware_choice_for_long': 'w1', 'sticky_preferred_for_sys_a': 'w0', 'actual_worker_for_r1': 'w1', 'actual_worker_for_r2': 'w1', 'big_request_admitted': False, 'code_worker': 'w3', 'unhealthy_after_timeout': ['w1'], 'state_rows': {'r1': {'worker_id': 'w0', 'stream_started': False, 'status': 'retried', 'events': ['route:w1', 'retry:w0']}, 'r2': {'worker_id': 'w1', 'stream_started': True, 'status': 'failed_stream', 'events': ['route:w1', 'fail_stream:worker_failed']}, 'r4': {'worker_id': 'w3', 'stream_started': False, 'status': 'routed', 'events': ['route:w3']}}, 'selection_counts': {'w0': 1, 'w1': 2, 'w2': 0, 'w3': 1}, 'selection_imbalance': 1, 'metrics': {'router_reject_total': 1, 'sticky_route_hit_total': 1, 'sticky_route_fallback_total': 2, 'request_retry_total': 1, 'stream_fail_total': 1, 'worker_failure_total': 1, 'capability_filter_total': 12}}
multi_worker_router_gates= {'capability_filter_ready': True, 'global_admission_ready': True, 'load_aware_ready': True, 'sticky_fallback_ready': True, 'health_check_ready': True, 'retry_gate_ready': True, 'state_boundary_ready': True, 'metrics_ready': True, 'multi_worker_router_gate': True}
```

这个 demo 证明了几个关键点：

1. round-robin 会把长请求打到高负载 sticky worker，load-aware routing 会避开它。
2. router 先做模型和 adapter 能力过滤，code 请求不会被发给 chat worker。
3. sticky 只是首选，首选 worker 过载时必须 fallback。
4. 全局 admission control 会拒绝超出候选 worker 总 KV 容量的大请求。
5. heartbeat 超时后 worker 停止接收新请求。
6. worker failure 后，未输出请求可以重试，已经 streaming 的请求显式失败。
7. router 维护 request-to-worker map，但不直接修改 worker 内部 scheduler、KV block table 或 prefix cache。

## 55.25 小练习

1. 实现 `WorkerHandle` 和 `WorkerLoad`。
2. 给每个 worker 增加 heartbeat。
3. 实现最简单的 round-robin router。
4. 给 router 增加 model_name 过滤。
5. 实现 `estimate_request_blocks`。
6. 实现基于 queue size 和 KV free blocks 的 load-aware routing。
7. 增加 preemption_rate 和 recent_ttft_p90_ms 到 worker score。
8. 实现 prefix_hash sticky routing。
9. 实现 sticky 过载 fallback。
10. 增加 worker unhealthy 后停止路由。
11. 维护 request_id 到 worker_id 的映射。
12. 模拟 worker crash，测试未输出请求重试。
13. 模拟 worker crash，测试已输出请求失败 stream。
14. 给 router 增加 global admission control。
15. 写一个多 worker benchmark，输出每个 worker 的 TTFT、TPOT、KV free blocks 和 selected_total。

## 55.26 本章总结

从单 worker 到多 worker，核心变化不是多起几个进程这么简单。

系统多了一个 request router。

router 要根据 worker 能力、健康状态、负载、KV 空间、延迟、preemption 和 cache locality 做决策。

round-robin 可以作为 baseline，但不能作为生产级策略。

LLM 请求成本差异很大，长 prompt、长输出和短请求对 worker 的压力完全不同。

prefix cache 又让路由变得更复杂。

为了提高 cache 命中，相似请求应该尽量 sticky 到同一个 worker。

但过度 sticky 会造成热点。

所以实际策略通常是 sticky + load-aware。

多 GPU serving 还要区分 replica parallel 和 tensor parallel。

replica parallel 是多个完整模型副本，router 做请求级负载均衡。

tensor parallel 是一个模型跨多张 GPU，一个 TP group 从 router 视角看通常是一个 worker。

故障处理也必须谨慎。

没有输出过的请求可以重试，已经输出过 token 的请求通常不能透明重试。

最后，多 worker benchmark 必须按 worker 维度展示指标。

只看全局平均吞吐会掩盖热点、尾延迟和负载不均。

下一章可以继续讨论：多副本 serving 之外，如果模型太大或吞吐要求更高，如何理解 tensor parallel、pipeline parallel 和分布式 KV cache 的执行细节。
