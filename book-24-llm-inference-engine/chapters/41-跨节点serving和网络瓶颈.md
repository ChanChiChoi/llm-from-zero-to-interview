# 第 41 章 跨节点 Serving 和网络瓶颈

上一章讲了多级 KV Cache：当 KV 不能只放在单个 GPU 上时，需要 GPU、CPU、SSD、远端缓存和 recompute 共同组成一个系统级缓存层级。

一旦 KV、模型分片、请求调度或 prefix cache 走出单机，网络就会进入关键路径。

跨节点 serving 的难点不只是“多几台机器”。真正的问题是：

```text
模型计算、KV Cache、调度状态、请求流和 streaming 输出，哪些必须跨节点通信？
这些通信是否会进入 TTFT 或 TPOT 的关键路径？
```

一句话概括：

> 跨节点 serving 的核心不是把 GPU 堆起来，而是在模型并行、数据并行、PD 分离、KV transfer 和网络拓扑之间做取舍，避免网络把吞吐收益吃掉。

## 41.1 本章目标

读完本章，你应该能讲清：

1. 跨节点 serving 为什么比单机多卡复杂得多。
2. TP、PP、DP、PD 分离在跨节点时分别产生什么网络通信。
3. 为什么 TP 通常不适合跨低速网络。
4. KV transfer、remote KV cache 和 prefix sharing 如何受网络影响。
5. 带宽、延迟、拥塞、拓扑距离如何影响 TTFT、TPOT 和吞吐。
6. 如何判断网络瓶颈来自 all-reduce、pipeline activation、KV transfer 还是 router。
7. 跨节点 serving 的部署策略和调优方法。
8. 面试中如何设计一个跨节点推理服务。

## 41.2 跨节点 Serving 到底跨了什么

跨节点不一定只有一种形态。

常见有四类：

1. 多个独立 replica，外部负载均衡。
2. 一个模型 replica 内部跨节点做 TP/PP。
3. Prefill worker 和 Decode worker 跨节点，也就是跨节点 PD 分离。
4. KV Cache、prefix cache 或 session state 跨节点共享。

它们的通信模式完全不同。

```text
DP replica:
  request -> one replica
  replicas mostly independent

TP across nodes:
  every layer may communicate across nodes

PP across nodes:
  stage boundary transfers hidden states

PD across nodes:
  prefill side transfers KV to decode side

remote KV cache:
  worker fetches or pushes KV blocks over network
```

所以不能笼统说“跨节点慢”。要问清楚：跨节点的是请求、激活、KV、权重分片通信，还是控制面 metadata。

## 41.3 网络为什么会进入关键路径

Serving 的延迟指标主要有两个：

1. TTFT：从请求进入系统到首 token 返回。
2. TPOT：生成过程中每个 token 的平均或尾部间隔。

网络可能进入 TTFT：

```text
request arrives
  -> router selects remote worker
  -> remote prefix lookup
  -> fetch KV from remote cache
  -> prefill or decode
  -> first token returns
```

网络也可能进入 TPOT：

```text
decode step
  -> TP all-reduce across nodes
  -> next layer
  -> TP all-reduce across nodes
  -> ...
  -> output token
```

或者：

```text
decode step waits for missing KV from remote
```

只影响 TTFT 的网络开销，有时还能接受。影响每个 decode step 的网络开销，通常更危险。

因为 decode 是高频循环：

```text
one network hiccup per request -> occasional TTFT spike
one network hiccup per token   -> continuous TPOT jitter
```

## 41.4 带宽和延迟不是一回事

网络瓶颈通常分成两个维度：

1. Bandwidth：单位时间能传多少数据。
2. Latency：一次通信来回或单向需要多久。

大对象传输更看带宽。

例如 KV transfer：

```text
KV bytes = 2 GB
effective bandwidth = 25 GB/s
transfer time ~= 80 ms
```

小而频繁的通信更看延迟。

例如 TP all-reduce：

```text
每层一次或多次 collective
每个 decode token 走几十层
```

即使每次传的数据不大，如果通信频率极高，延迟也会累积成 TPOT。

面试中可以这样说：

```text
KV transfer 更偏大块带宽问题，跨节点 TP 更偏高频 latency + collective bandwidth 问题。
```

## 41.5 常见互联层级

从快到慢，常见互联大致是：

```text
GPU HBM
  -> NVLink / NVSwitch
  -> PCIe
  -> InfiniBand / RoCE / RDMA network
  -> ordinary Ethernet
```

不同层级适合的通信不同。

| 互联 | 典型范围 | 特点 | 适合通信 |
| --- | --- | --- | --- |
| NVLink/NVSwitch | 单机多 GPU | 低延迟高带宽 | TP、GPU-GPU KV copy |
| PCIe | 单机 CPU-GPU/GPU-GPU | 带宽和拓扑受限 | CPU offload、一般 GPU copy |
| InfiniBand/RDMA | 跨节点 | 高性能网络，但部署复杂 | PP、PD KV transfer、跨节点通信 |
| RoCE | 跨节点以太网 RDMA | 依赖网络配置 | RDMA transfer、分布式 serving |
| 普通以太网 | 跨节点 | 延迟高、抖动大 | 控制面、低频请求转发 |

跨节点 serving 通常要尽量把高频同步通信限制在高速互联范围内。

## 41.6 Data Parallel 跨节点

Serving 中的 Data Parallel 通常是多个独立 replica。

例如：

```text
router
  -> replica 0: model copy + KV cache
  -> replica 1: model copy + KV cache
  -> replica 2: model copy + KV cache
  -> replica 3: model copy + KV cache
```

DP 的跨节点通信最少。

每个请求只进入一个 replica，decode 过程中不需要和其他 replica 同步。

它的优点是：

1. 横向扩吞吐简单。
2. 故障隔离好。
3. TPOT 不依赖跨 replica 通信。
4. 调试相对容易。

它的代价是：

1. 每个 replica 都要一份权重。
2. 每个 replica 有独立 KV Cache。
3. prefix cache 容易重复。
4. 负载均衡要考虑 KV locality。
5. 热点 session 可能集中到某个 replica。

DP 适合模型单个 replica 能放下，主要目标是提升总 QPS 的场景。

## 41.7 Tensor Parallel 跨节点

Tensor Parallel 把每层矩阵或 attention heads 切到多个 rank。

它的问题是通信频率高。

decode 每生成一个 token，都要经过所有层。每层可能需要：

1. all-reduce。
2. all-gather。
3. reduce-scatter。
4. rank 间同步。

如果 TP group 跨节点，通信路径变成：

```text
GPU rank on node A
  -> network
  -> GPU rank on node B
```

这会直接影响 TPOT。

因此常见原则是：

```text
TP 尽量限制在单节点 NVLink/NVSwitch 内。
跨节点扩展优先考虑 PP 或 DP。
```

例如 2 台 8 卡机器：

```text
推荐：TP=8, PP=2
谨慎：TP=16, PP=1
```

前者把高频 TP 通信留在节点内，只让 pipeline stage 跨节点传 hidden states。后者每层都可能跨节点 collective，TPOT 更容易变差。

当然，如果网络极强、collective 优化很好、模型必须这样切，也不是绝对不能跨节点 TP。但它应该是经过压测验证的选择，而不是默认选择。

## 41.8 Pipeline Parallel 跨节点

Pipeline Parallel 按层切分模型。

跨节点时，stage 之间传递 hidden states。

例如：

```text
node 0: layers 0-19
node 1: layers 20-39
```

请求执行路径是：

```text
input hidden states
  -> node 0 forward
  -> transfer activations to node 1
  -> node 1 forward
  -> logits
```

PP 的跨节点通信频率通常低于 TP，因为它不是每层都 collective，而是在 stage 边界传 activation。

但 PP 有自己的问题：

1. 单请求延迟增加。
2. pipeline bubble。
3. stage 负载不均。
4. continuous batching 更复杂。
5. stage 之间的通信也会影响 TPOT。

Serving 中 decode step 很小，如果 batch 不够或调度不好，pipeline 很容易填不满。

所以 PP 适合：

1. 模型单节点放不下。
2. 跨节点 TP 通信太贵。
3. 层切分比较均衡。
4. 有足够并发填 pipeline。

不适合：

1. 极低延迟、小并发场景。
2. stage 间网络很差。
3. 模型层耗时严重不均且无法均衡切分。

## 41.9 TP + PP 组合

大模型 serving 常见组合是节点内 TP、节点间 PP。

例如 4 台机器，每台 8 张 GPU：

```text
TP = 8
PP = 4
total GPUs = 32
```

含义是：

```text
每个 pipeline stage 用一整台机器的 8 张 GPU 做 TP
4 台机器按层组成 PP
```

这样做的好处是：

1. 每层内部高频 TP 通信走节点内 NVLink/NVSwitch。
2. 跨节点只传 stage activation。
3. 可以支持单节点放不下的大模型。
4. 网络通信模式更可控。

代价是：

1. 单个 replica 占用大量 GPU。
2. pipeline bubble 需要并发填充。
3. 任一节点慢都会拖慢整个 replica。
4. KV Cache 也按 stage/rank 分布，管理更复杂。

面试中这个组合非常常见，应该熟练表达。

## 41.10 PD 分离跨节点

PD 分离把 prefill 和 decode 放到不同 worker 或资源池。

跨节点 PD 的通信核心是 KV transfer。

流程大致是：

```text
request -> prefill worker
prefill worker computes prompt KV
prefill worker transfers KV to decode worker
decode worker starts streaming decode
```

网络开销主要体现在：

1. prefill 完成后 KV 传输时间。
2. decode worker 等待 KV 的时间。
3. 多个 transfer 并发导致的网络拥塞。
4. 传输失败或超时后的恢复。

PD 分离是否值得，取决于：

```text
P/D 资源隔离收益
  > KV transfer 成本 + 调度复杂度 + 网络抖动成本
```

如果网络很差，跨节点 PD 可能把原来的 compute interference 变成 network interference。

这也是为什么 PD 分离常常需要高性能 transfer backend，例如 RDMA、GPUDirect RDMA、UCX、NIXL、Mooncake 这类能力或系统。

## 41.11 KV Transfer 的网络成本

KV transfer 的数据量与 prompt length、层数、KV heads、head dim、dtype 和 TP 切分有关。

可以粗略理解为：

```text
kv_bytes
  ~= num_tokens
    * num_layers
    * kv_heads
    * head_dim
    * 2  # K and V
    * bytes_per_element
```

如果模型使用 GQA/MQA，KV heads 更少，KV transfer 压力会降低。

如果 prompt 很长，KV transfer 会非常大。

例如：

```text
short prompt: 512 tokens
long prompt: 32000 tokens
```

后者 KV bytes 可能是前者的 62.5 倍。

因此跨节点 PD 对长 prompt 更敏感。Chunked prefill 可以把 transfer 流水化，但不能消除总 bytes。

## 41.12 Remote KV Cache 的网络成本

Remote KV Cache 和 PD KV transfer 很像，但访问模式可能不同。

PD transfer 通常是一次请求内从 prefill 到 decode 的交接。

Remote KV Cache 可能是多请求、多 worker 共享的缓存。

访问路径可能是：

```text
worker -> metadata service -> remote KV store -> worker
```

它的成本包括：

1. metadata lookup。
2. remote read/write。
3. 网络传输。
4. 反序列化或解压。
5. CPU staging。
6. GPU copy。
7. consistency check。

所以 remote cache hit 不等于快。

更准确的判断是：

```text
remote_hit_latency + promote_latency < recompute_latency
```

并且不能明显影响其他请求的 TPOT。

## 41.13 Router 也会制造网络瓶颈

很多人只关注 GPU 间通信，忽略 router。

跨节点 serving 中，router 负责：

1. 选择 replica。
2. 选择 prefill worker。
3. 选择 decode worker。
4. 根据 KV locality 路由。
5. 做 admission control。
6. 管理 streaming 连接。
7. 处理重试和故障转移。

如果 router 不理解网络拓扑，就可能做出坏决策。

例如：

```text
prefill worker on rack A
decode worker on rack B
remote KV cache on rack C
```

这个请求可能经历多次跨 rack 传输。

更好的路由要考虑：

1. worker 当前负载。
2. GPU KV capacity。
3. prefix cache locality。
4. P/D worker 网络距离。
5. pending KV transfers。
6. 当前网络拥塞。
7. tenant 或 session affinity。

这就是 topology-aware routing 和 KV-aware routing。

## 41.14 拓扑距离

网络拓扑不是平的。

常见层级包括：

```text
same GPU island
same node
same rack
same cluster fabric
cross rack
cross zone
```

同样是“远端”，距离可能完全不同。

例如：

```text
same node NVLink transfer: very fast
same rack RDMA: acceptable
cross rack under congestion: unstable
cross zone: usually not for TPOT path
```

跨节点 serving 的原则是：

```text
高频、低延迟通信放近；低频、大块、可异步通信可以放远。
```

对应到不同通信：

1. TP collective：尽量同机。
2. PP activation：尽量同 rack 或高性能 fabric。
3. PD KV transfer：最好 topology-aware 配对。
4. remote KV cache：尽量按 locality 分片或复制。
5. 控制面 metadata：可以更远，但要有缓存和超时。

## 41.15 拥塞和尾延迟

网络平均带宽够，不代表 serving 稳定。

LLM serving 很怕 tail latency。

例如：

```text
p50 transfer latency = 20 ms
p99 transfer latency = 300 ms
```

如果 p99 transfer 在 TTFT 关键路径上，用户会感到首 token 很慢。

如果 p99 communication 在 decode step 中，streaming 会卡顿。

造成 tail latency 的原因包括：

1. 多个 worker 同时传大 KV。
2. collective 与 KV transfer 争用网络。
3. remote cache 热点 shard。
4. rack 间 oversubscription。
5. RDMA 配置或网卡队列问题。
6. TCP retransmission 或 congestion control。
7. metadata service 抖动。

因此观测 p50 不够，必须看 p95/p99。

## 41.16 Control Plane 和 Data Plane

跨节点 serving 可以分成控制面和数据面。

控制面包括：

1. worker 注册。
2. heartbeat。
3. 负载信息上报。
4. KV metadata 查询。
5. 路由决策。
6. transfer plan。
7. 失败恢复。

数据面包括：

1. 请求输入。
2. streaming token 输出。
3. KV transfer。
4. activation transfer。
5. collective communication。
6. remote cache read/write。

控制面通常数据量小，但对一致性和超时敏感。

数据面数据量大，直接影响带宽和延迟。

设计上要避免控制面阻塞数据面，也要避免数据面拥塞导致控制面心跳误判。

## 41.17 Streaming 输出路径

Serving 不是只算完返回一个结果，很多场景需要 token streaming。

跨节点时，streaming 路径可能是：

```text
decode worker -> router/gateway -> client
```

也可能是：

```text
decode worker -> frontend connection owner -> client
```

需要注意：

1. token 输出很小但频繁。
2. 不应被大 KV transfer 阻塞。
3. gateway 不能成为单点瓶颈。
4. backpressure 要能传回 decode scheduler。
5. worker fail 后 streaming 如何结束要明确。

如果 streaming channel 和大对象 transfer 共用同一拥塞路径，也可能出现用户看到 token 断断续续的问题。

## 41.18 跨节点 Batch 和负载均衡

Continuous batching 在单机内已经复杂，跨节点后更复杂。

对于 DP replicas，router 要把请求分配给不同 replica。

简单 least-loaded 策略可能不够。

因为一个 worker 负载低，不代表它适合这个请求。

还要看：

1. 该请求 prefix 是否在这个 worker。
2. 这个 worker 的 KV blocks 是否足够。
3. 当前 decode batch 是否稳定。
4. 是否有 pending transfer。
5. 网络距离是否近。
6. 租户配额是否允许。

例如：

```text
worker A: load 70%, prefix hit on GPU
worker B: load 20%, no prefix hit, remote fetch required
```

worker A 可能反而更快。

所以跨节点路由目标不是最小当前负载，而是最小预计完成成本。

## 41.19 Admission Control

跨节点系统更需要 admission control。

如果没有限制，可能出现：

```text
大量长 prompt 同时进入
  -> prefill worker 忙
  -> KV transfer 排队
  -> decode worker 等 KV
  -> network saturated
  -> TTFT/TPOT 全部恶化
```

Admission control 可以限制：

1. 同时进行的 KV transfer 数。
2. 每个 worker 的 pending transfer bytes。
3. 每个租户的长 prompt 并发。
4. remote cache read/write QPS。
5. 跨 rack traffic。
6. prefill queue length。
7. decode queue length。

一个简单规则是：

```text
if pending_kv_transfer_bytes > threshold:
    stop admitting long-prefill requests
```

这不是降低吞吐，而是避免系统进入拥塞崩溃。

## 41.20 Backpressure

Backpressure 是下游压力向上游传递。

跨节点 PD 中尤其重要。

例如 decode worker 消费不过来：

```text
decode worker KV capacity low
  -> pending transfers increase
  -> prefill workers should slow down
  -> router should stop assigning more long prompts
```

如果没有 backpressure，prefill 侧会继续生成大量 KV，最后造成：

1. 网络排队。
2. CPU staging buffer 堆积。
3. GPU blocks 预留失败。
4. 请求超时。
5. 大量无用 prefill 计算。

Backpressure 信号可以包括：

1. decode queue length。
2. available KV blocks。
3. pending transfer bytes。
4. network utilization。
5. remote cache latency。
6. TPOT p95/p99。

## 41.21 故障模式

跨节点 serving 的故障模式比单机多。

常见包括：

1. worker crash。
2. network timeout。
3. partial KV transfer。
4. remote cache write succeeded but metadata update failed。
5. metadata says KV exists but data missing。
6. TP/PP group 中一个 rank 掉线。
7. router 重试导致重复请求。
8. streaming 中途断开。
9. prefill 成功但 decode worker 已不可用。
10. remote cache shard 热点或不可用。

系统要明确每种失败后的策略：

1. retry。
2. reroute。
3. recompute。
4. cancel request。
5. clean partial KV。
6. degrade to local only。

不要让半完成的 KV 长期占用 cache，也不要让 metadata 指向不存在的数据。

## 41.22 观测指标

跨节点 serving 的指标要按通信类型拆开。

总体指标：

1. TTFT p50/p95/p99。
2. TPOT p50/p95/p99。
3. end-to-end latency。
4. throughput tokens/s。
5. request timeout rate。

网络数据面指标：

1. KV transfer bytes/s。
2. KV transfer latency p50/p95/p99。
3. remote KV read latency。
4. remote KV write latency。
5. activation transfer latency。
6. collective communication time。
7. RDMA/TCP error count。
8. network retransmission。
9. NIC utilization。
10. cross-rack traffic bytes。

调度指标：

1. pending transfer count。
2. pending transfer bytes。
3. prefill queue length。
4. decode queue length。
5. router decision latency。
6. topology-aware routing hit rate。
7. prefix locality hit rate。
8. admission rejection count。
9. backpressure activation count。

GPU 侧指标：

1. GPU utilization。
2. GPU memory utilization。
3. KV block utilization。
4. preemption count。
5. decode step latency。
6. prefill latency。

关键是把延迟拆到阶段，否则只看总 latency 不知道瓶颈在哪。

## 41.23 如何定位网络瓶颈

可以按路径定位。

第一步，看是 TTFT 还是 TPOT 变差。

```text
TTFT 变差，TPOT 正常：
  优先看 prefill queue、remote prefix lookup、KV transfer、admission、router。

TPOT 变差：
  优先看 decode step、TP collective、PP activation、remote KV 是否进入 decode 路径、网络拥塞。
```

第二步，看是否和请求长度相关。

```text
长 prompt 更差：可能是 prefill 或 KV transfer。
长 output 更差：可能是 decode step 或 TP/PP 通信。
```

第三步，看是否和拓扑相关。

```text
same-node 正常，cross-node 差：网络或跨节点通信。
same-rack 正常，cross-rack 差：拓扑/oversubscription。
某些 worker 差：网卡、链路、配置或热点。
```

第四步，看 p99 是否远高于 p50。

```text
p50 正常，p99 很差：拥塞、排队、热点、重试或 metadata 抖动。
```

## 41.24 部署策略

一个务实的部署策略是：

1. 能用 DP 扩吞吐时，优先用 DP。
2. 单节点能放下模型时，尽量避免跨节点模型并行。
3. 需要 TP 时，优先把 TP 限制在单节点高速互联内。
4. 模型单节点放不下时，用节点内 TP + 节点间 PP。
5. PD 分离跨节点时，让 P/D worker topology-aware 配对。
6. remote KV cache 不要无脑进入关键路径。
7. 大 KV transfer 要做 admission control 和 backpressure。
8. prefix cache 要考虑 locality，而不是全局命中率至上。

可以用一个决策树：

```text
模型单节点能放下？
  yes -> 用单节点 TP + 多 replica DP
  no  -> 节点内 TP + 节点间 PP

长 prompt 严重干扰 decode？
  yes -> 考虑 PD 分离
       -> 网络能承受 KV transfer？
          yes -> 跨节点 PD
          no  -> 同机 PD 或 chunked prefill

prefix 复用率很高？
  yes -> local prefix cache first
       -> remote cache only if fetch < recompute
```

## 41.25 调优思路

调优跨节点 serving，不要一上来改复杂架构。

先做基线：

1. 单机单卡。
2. 单机多卡 TP。
3. 多 replica DP。
4. 节点内 TP + 节点间 PP。
5. PD 分离。
6. remote KV cache。

每一步都记录：

1. TTFT。
2. TPOT。
3. tokens/s。
4. GPU utilization。
5. network utilization。
6. p95/p99。

常见调参方向：

1. 减小跨节点 TP group。
2. 增加 DP replicas。
3. 调整 PP stage 切分。
4. 限制并发 KV transfer bytes。
5. 提高 prefix locality routing 权重。
6. 限制 remote cache 进入关键路径。
7. 为 streaming 与大对象 transfer 隔离队列或连接。
8. 增加 backpressure 阈值。

## 41.26 常见坑

坑一：把 TP 横跨普通网络。

每层高频 collective 进入网络，TPOT 可能严重恶化。

坑二：只看总带宽，不看 p99。

Serving 对尾延迟敏感，p99 transfer 或 p99 collective 会直接影响用户体验。

坑三：remote cache hit 率很高但延迟更差。

远端命中如果比 recompute 还慢，就是错误优化。

坑四：router 只看 worker load。

跨节点路由还要看 KV locality、网络距离、pending transfer 和 KV capacity。

坑五：KV transfer 没有 backpressure。

prefill 侧持续产出 KV，decode 侧消费不过来，会造成网络拥塞和大量请求超时。

坑六：控制面和数据面互相影响。

大对象传输拥塞导致 heartbeat 或 metadata 请求超时，系统可能误判 worker 故障。

坑七：streaming 输出和 KV transfer 争用路径。

用户看到 token 卡顿，但 GPU 可能并不慢，真正问题是网络队列被大对象堵住。

坑八：跨节点故障清理不完整。

partial KV、预留 blocks、metadata 和请求状态没有一致清理，会造成内存泄漏和错误命中。

## 41.27 面试官会怎么问

问题一：为什么 TP 不建议跨节点？

回答要点：TP 每层都可能有 all-reduce/all-gather/reduce-scatter，decode 每个 token 都要经过所有层。如果跨节点，通信频率高、延迟累积，会直接影响 TPOT。通常建议 TP 放在单节点 NVLink/NVSwitch 内，跨节点用 PP 或 DP。

问题二：跨节点 PD 分离的最大瓶颈是什么？

回答要点：Prefill 生成的 KV 必须传给 decode worker，KV transfer 的 bytes 与 prompt length、层数、KV heads、dtype 等相关。网络带宽、延迟、拥塞和 transfer backend 会决定 PD 分离收益是否大于传输成本。

问题三：Remote KV Cache 命中一定能降低延迟吗？

回答要点：不一定。要比较 remote lookup、read、network transfer、deserialize、promote 到 GPU 的总成本与 recompute 成本。如果 remote hit 在关键路径上更慢，或者影响 TPOT，就不应该用。

问题四：跨节点 router 应该考虑哪些信息？

回答要点：除了 worker load，还要考虑 KV capacity、prefix locality、pending transfers、P/D worker 网络距离、当前网络拥塞、decode TPOT、租户配额和 session affinity。

问题五：如何定位跨节点 serving 的网络瓶颈？

回答要点：先看 TTFT 还是 TPOT 变差；再按 prefill、decode、KV transfer、remote cache、collective、PP activation、router 分阶段打点；比较 same-node/same-rack/cross-rack；看 p50/p95/p99 和 pending transfer bytes。

## 41.28 标准回答模板

如果面试官问“如何设计跨节点 LLM serving”，可以这样回答：

```text
我会先区分跨节点通信类型。Serving 里跨节点可能是 DP replica 之间的请求分流，也可能是一个 replica 内部的 TP/PP 模型并行，也可能是 PD 分离里的 KV transfer，还可能是 remote KV cache。不同通信对网络的敏感性不同。

部署上，能用 DP 横向扩吞吐时优先用 DP，因为每个 replica 独立，decode 过程不需要跨节点同步。如果模型单节点放不下，我会优先采用节点内 TP、节点间 PP，让高频 TP collective 留在 NVLink/NVSwitch 内，跨节点只传 pipeline activation。除非网络和 collective 优化非常强，否则不默认让 TP 跨节点。

如果长 prompt prefill 干扰 decode，再考虑 PD 分离。但跨节点 PD 的关键是 KV transfer，必须比较 P/D 隔离收益和 KV 传输成本。Router 要做 topology-aware 和 KV-aware routing，考虑 decode worker 的 KV capacity、pending transfer、prefix locality 和 P/D 网络距离，而不是只看 worker load。

对 remote KV cache，我不会只看命中率，而会比较 remote fetch + promote 的成本和 recompute 成本，并避免 remote fetch 进入 TPOT 关键路径。系统还需要 admission control 和 backpressure，限制同时进行的大 KV transfer，防止网络拥塞把 TTFT 和 TPOT 一起拖垮。

观测上，我会按阶段拆指标：TTFT、TPOT、KV transfer latency、collective time、PP activation transfer、remote cache latency、pending transfer bytes、network p95/p99、router decision latency。这样才能判断瓶颈到底在 compute、KV、network 还是 routing。
```

## 41.29 小练习

1. 画出 DP、TP、PP、PD 分离四种跨节点 serving 的通信路径。
2. 解释为什么 TP 跨节点会影响 TPOT。
3. 对比 KV transfer 和 TP all-reduce 的网络瓶颈差异。
4. 给定 2 台 8 卡机器，说明为什么常见选择是 TP=8、PP=2。
5. 设计一个 topology-aware router 的打分函数。
6. 设计一个 admission control 策略限制大 KV transfer。
7. 分析 remote KV hit 什么时候不如 recompute。
8. 说明 streaming 输出为什么可能被大对象 transfer 间接影响。
9. 列出跨节点 serving 的 15 个关键观测指标。
10. 设计一个排查 TPOT p99 突然升高的流程。

## 41.30 本章总结

跨节点 serving 的核心问题是通信模式。DP 主要跨节点分流请求，通信最少；TP 是高频 collective，通常应限制在单节点高速互联内；PP 跨节点传 activation，适合模型单节点放不下的场景；PD 分离跨节点的关键瓶颈是 KV transfer；remote KV cache 则要比较 fetch 成本和 recompute 成本。

网络瓶颈不只是带宽问题，也包括延迟、拥塞、拓扑距离、tail latency、metadata 抖动和失败恢复。

设计跨节点 serving 时，要优先保护 TPOT 稳定性，避免高频通信进入慢网络；用 topology-aware 和 KV-aware routing 提高 locality；用 admission control 和 backpressure 限制大 KV transfer；用分阶段指标定位瓶颈。

下一章会回到架构取舍，系统讨论 PD 分离的优缺点、适用场景和反模式。不是所有系统都应该做 PD 分离，也不是所有长 prompt 问题都应该通过跨节点架构解决。
