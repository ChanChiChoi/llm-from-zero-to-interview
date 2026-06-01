# 第 56 章 Tensor Parallel、Pipeline Parallel 和分布式 KV Cache

第 55 章我们从 request router 的角度看了多 worker、多 GPU serving。

router 眼里的 worker 可能是一张 GPU 上的完整模型副本，也可能是一组 GPU 共同服务一个模型。

本章进入 worker 内部。

当模型大到单卡放不下，或者单卡吞吐不够时，就需要模型并行。

常见方式包括：

1. tensor parallel。
2. pipeline parallel。
3. expert parallel。
4. data/replica parallel。
5. 它们的组合。

对 LLM inference 来说，模型并行不仅影响权重怎么切，还会影响：

1. 每层 forward 的通信。
2. prefill 和 decode 的延迟。
3. KV cache 放在哪里。
4. scheduler 看到的 worker 边界。
5. failure domain。
6. benchmark 指标怎么解释。

本章重点不是推导并行训练算法。

重点是理解 serving 里模型并行和 KV cache 的工程含义。

## 56.1 本章目标

读完本章，你应该能讲清：

1. replica parallel、tensor parallel、pipeline parallel 的区别。
2. tensor parallel 为什么需要 all-reduce 或 all-gather。
3. pipeline parallel 为什么在线 decode 容易有 bubble。
4. 为什么模型并行会改变单请求延迟。
5. KV cache 在 tensor parallel 下如何切分。
6. KV cache 在 pipeline parallel 下如何按层分布。
7. 为什么跨 worker 迁移 KV cache 很难。
8. scheduler 和 parallel group 的边界是什么。
9. 面试中如何解释多 GPU inference 的 trade-off。

## 56.2 先区分三个概念

很多人会把多 GPU 的几个概念混在一起。

先把它们分开。

### 56.2.1 Replica parallel

每个 worker 都有完整模型副本。

```text
Worker 0: full model on GPU 0
Worker 1: full model on GPU 1
Worker 2: full model on GPU 2
Worker 3: full model on GPU 3
```

一个请求只进入其中一个 worker。

优点是简单、隔离好、router 容易做负载均衡。

缺点是单请求不能使用多卡显存。

如果模型单卡放不下，replica parallel 不解决问题。

### 56.2.2 Tensor parallel

模型的一些张量在多张 GPU 上切分。

```text
TP Worker 0:
  GPU 0: shard 0
  GPU 1: shard 1
  GPU 2: shard 2
  GPU 3: shard 3
```

一个请求会同时使用这个 TP group 里的所有 GPU。

优点是能服务单卡放不下的大模型。

缺点是每层 forward 都可能需要跨卡通信。

### 56.2.3 Pipeline parallel

模型层被切到不同 GPU。

```text
GPU 0: layers 0-15
GPU 1: layers 16-31
GPU 2: layers 32-47
GPU 3: layers 48-63
```

一个请求从 GPU 0 依次流到 GPU 3。

优点是按层降低显存压力。

缺点是流水线 bubble 和 stage imbalance 很难处理。

尤其在 decode batch 较小时，pipeline 利用率可能很差。

## 56.3 Router 看到的并行组

第 55 章说过：router 不应该把一个 tensor parallel group 里的 GPU 当成独立 worker。

如果一个模型需要 4 张 GPU 组成 TP group：

```text
TP Group 0: GPU 0,1,2,3
TP Group 1: GPU 4,5,6,7
```

router 的视角应该是：

```text
Request Router
  +--> Worker 0: TP Group 0
  +--> Worker 1: TP Group 1
```

不是：

```text
Request Router
  +--> GPU 0
  +--> GPU 1
  +--> GPU 2
  +--> GPU 3
```

因为一个请求不能只发给 GPU 0。

它需要整个 TP group 协同完成 forward。

所以 worker 的粒度应该是：

```text
能独立完成一个请求 forward 的最小执行单元。
```

这个执行单元可能是一张 GPU，也可能是一组 GPU。

## 56.4 Tensor parallel 的基本直觉

LLM 里最重的计算通常是矩阵乘法。

比如线性层：

```text
Y = XW
```

如果 `W` 很大，可以把它切到多张 GPU。

有两种常见切法：

1. column parallel。
2. row parallel。

### 56.4.1 Column parallel

把权重按输出维度切分：

```text
W = [W0, W1, W2, W3]
```

每张 GPU 计算一部分输出：

```text
Y0 = XW0
Y1 = XW1
Y2 = XW2
Y3 = XW3
```

最后可以得到分片输出：

```text
Y = [Y0, Y1, Y2, Y3]
```

如果下一层也能接受分片输入，就可以暂时不 all-gather。

如果后续操作需要完整 hidden states，就需要 all-gather。

### 56.4.2 Row parallel

把权重按输入维度切分：

```text
W = [W0; W1; W2; W3]
```

输入 `X` 也被切分：

```text
X = [X0, X1, X2, X3]
```

每张 GPU 计算部分结果：

```text
Z0 = X0W0
Z1 = X1W1
Z2 = X2W2
Z3 = X3W3
```

完整输出需要求和：

```text
Y = Z0 + Z1 + Z2 + Z3
```

这通常需要 all-reduce。

## 56.5 为什么 tensor parallel 有通信开销

tensor parallel 的核心 trade-off 是：

```text
用更多 GPU 分摊计算和显存，但引入跨 GPU 通信。
```

每层 Transformer 里可能出现多次通信：

1. attention projection 后需要同步。
2. MLP projection 后需要同步。
3. logits 计算可能需要 gather 或 reduce。
4. residual、layer norm 的布局也会影响通信。

通信开销取决于：

1. TP size。
2. hidden size。
3. batch tokens 数。
4. GPU 间互联，NVLink、PCIe、InfiniBand。
5. collective 实现，NCCL 等。
6. 是否能和计算重叠。

所以 TP size 不是越大越好。

TP size 变大时，每张 GPU 的计算减少，但通信次数和同步成本可能增加。

当 batch tokens 很小时，decode 阶段尤其容易被通信延迟主导。

## 56.6 Prefill 和 decode 对 TP 的压力不同

prefill 阶段一次处理 prompt 的多个 token。

```text
batch tokens = sum(prompt lengths)
```

矩阵乘法比较大，GPU 利用率更容易打满。

TP 的通信开销可以被较大的计算量摊薄。

decode 阶段每个 request 通常每轮只生成一个 token。

```text
batch tokens = running requests count
```

如果并发不高，每轮 decode 的矩阵很小。

这时通信 latency 占比会变高。

所以同一个 TP 配置下：

```text
prefill 可能很高效，decode 可能被通信和同步拖慢。
```

这也是 serving scheduler 要尽量把 decode batch 做大的原因之一。

## 56.7 Tensor parallel 下的 KV cache

KV cache 存的是每层 attention 的 key/value。

在 TP 下，attention heads 通常会被分到不同 GPU。

例如模型有 32 个 heads，TP size = 4：

```text
GPU 0: heads 0-7 的 K/V
GPU 1: heads 8-15 的 K/V
GPU 2: heads 16-23 的 K/V
GPU 3: heads 24-31 的 K/V
```

每张 GPU 只保存自己负责 heads 的 KV cache。

这意味着：

1. KV cache 本身也被 sharded。
2. 每张 GPU 的 KV block manager 管理本地 shard。
3. 一个 request 的 KV blocks 在 TP group 内是一组对应分片。
4. 分配和释放要在所有 TP ranks 上保持一致。

可以把一个逻辑 block 看成：

```text
Logical KV Block 17
  rank 0 shard: heads 0-7
  rank 1 shard: heads 8-15
  rank 2 shard: heads 16-23
  rank 3 shard: heads 24-31
```

从 scheduler 角度看，它分配的是逻辑 block。

从每个 GPU 角度看，它分配的是本地物理 block shard。

## 56.8 TP 下 block allocation 的一致性

TP group 里的每个 rank 都要对同一个 request 使用一致的 block table。

如果 rank 0 认为 request A 使用 blocks `[1, 2, 3]`，rank 1 却认为它使用 `[1, 2, 4]`，attention 会直接错乱。

所以 block allocation 通常由一个统一的调度逻辑决定，然后广播给各 rank。

伪代码：

```python
class TPBlockManager:
    def allocate(self, request_id: str, num_blocks: int) -> list[int]:
        block_ids = self.logical_allocator.allocate(num_blocks)
        self.broadcast_block_table(request_id, block_ids)
        return block_ids

    def free(self, request_id: str) -> None:
        block_ids = self.request_blocks.pop(request_id)
        self.broadcast_free(request_id, block_ids)
        self.logical_allocator.free(block_ids)
```

真实系统里不一定真的每次用 Python broadcast。

但原则是一样的：

```text
TP ranks 必须对 request 的 block table 达成一致。
```

## 56.9 Pipeline parallel 的基本直觉

pipeline parallel 按层切模型。

假设 4 个 stage：

```text
Stage 0: embedding + layers 0-15
Stage 1: layers 16-31
Stage 2: layers 32-47
Stage 3: layers 48-63 + lm_head
```

一次 forward 要依次经过所有 stage：

```text
hidden -> Stage 0 -> Stage 1 -> Stage 2 -> Stage 3 -> logits
```

为了提高利用率，可以把 batch 切成 micro-batches，让不同 stage 同时处理不同 micro-batch。

训练里这很常见。

但 serving decode 有特殊问题。

## 56.10 Pipeline bubble

pipeline 需要填充和排空。

假设 4 个 stage，只有一个 micro-batch：

```text
time 1: MB0 -> S0
time 2: MB0 -> S1
time 3: MB0 -> S2
time 4: MB0 -> S3
```

每个时刻只有一个 stage 在工作，其他 stage 空闲。

这就是 bubble。

如果有多个 micro-batch：

```text
time 1: MB0 -> S0
time 2: MB1 -> S0, MB0 -> S1
time 3: MB2 -> S0, MB1 -> S1, MB0 -> S2
time 4: MB3 -> S0, MB2 -> S1, MB1 -> S2, MB0 -> S3
```

利用率会提高。

但在线 LLM serving 的 decode 阶段，每轮 token 生成有强依赖：

```text
第 t+1 个 token 依赖第 t 个 token 的采样结果。
```

所以不能随便把同一个请求的未来 token 提前放进 pipeline。

只能依赖不同请求和不同 micro-batch 来填充 pipeline。

当并发不够时，pipeline bubble 会很明显。

## 56.11 Pipeline parallel 下的 KV cache

pipeline parallel 按层切分。

所以 KV cache 也自然按层分布。

```text
Stage 0: layers 0-15 的 KV cache
Stage 1: layers 16-31 的 KV cache
Stage 2: layers 32-47 的 KV cache
Stage 3: layers 48-63 的 KV cache
```

一个 request 的完整 KV cache 分布在多个 stage 上。

每个 stage 只保存自己负责层的 K/V。

这带来几个工程要求：

1. 每个 stage 都要知道 request 的位置和 block table。
2. request finish/cancel 时，所有 stage 都要释放对应 KV。
3. preemption 时，所有 stage 要一致处理。
4. 某个 stage OOM 会影响整个 pipeline worker。
5. stage 间要传递 hidden states，而不是完整 KV。

从 scheduler 视角看，仍然应该分配逻辑 request 资源。

但底层释放和清理必须覆盖所有 stage。

## 56.12 TP + PP 组合

大模型常常同时使用 tensor parallel 和 pipeline parallel。

例如 8 张 GPU：

```text
Pipeline Stage 0: GPU 0,1,2,3 组成 TP group
Pipeline Stage 1: GPU 4,5,6,7 组成 TP group
```

这叫 TP=4, PP=2。

模型层分成两个 pipeline stage。

每个 stage 内部再用 4 张 GPU 做 tensor parallel。

router 看到的仍然是一个 worker：

```text
Worker 0: TP=4, PP=2, GPUs 0-7
```

这个 worker 才能独立完成一个请求。

如果有 16 张 GPU，可以有两个这样的 worker：

```text
Worker 0: TP=4, PP=2, GPUs 0-7
Worker 1: TP=4, PP=2, GPUs 8-15
```

router 在 Worker 0 和 Worker 1 之间做请求级路由。

worker 内部做 TP/PP 通信和调度。

## 56.13 分布式 KV cache 为什么难迁移

第 55 章说过，worker crash 后，已经输出 token 的请求通常不能透明重试。

一个重要原因就是 KV cache 很难跨 worker 迁移。

KV cache 不是一个小对象。

对于长上下文和大模型，它可能非常大。

一个粗略估算：

```text
KV bytes = layers * tokens * kv_heads * head_dim * 2(K,V) * dtype_bytes
```

如果：

```text
layers = 80
tokens = 32000
kv_heads = 8
head_dim = 128
dtype_bytes = 2
```

那么：

```text
KV bytes = 80 * 32000 * 8 * 128 * 2 * 2
         ≈ 10 GB
```

这只是一个 request 的 KV cache 量级。

跨 worker 迁移意味着要把这些 GPU memory 从一个 worker 搬到另一个 worker，并且还要保持：

1. layer 顺序一致。
2. head shard 一致。
3. block table 一致。
4. dtype 和 layout 一致。
5. position encoding 状态一致。
6. sampling 状态一致。

这非常昂贵。

所以多数 serving 系统不会把正在 decode 的请求随意迁移到另一个 worker。

## 56.14 KV cache 的 locality

因为 KV cache 难迁移，所以 request 一旦进入某个 worker，通常会在这个 worker 上完成。

这叫 request locality。

它影响很多设计：

1. cancel 必须发回原 worker。
2. output stream 必须关联原 worker。
3. worker 过载时，已经 running 的请求不能简单搬走。
4. router 只能影响新请求，不能轻易重平衡正在运行的请求。
5. prefix cache 也只在 worker 本地有效。

这和传统无状态 HTTP 服务很不一样。

LLM serving worker 是有大量 GPU resident state 的。

## 56.15 分布式 prefix cache

prefix cache 在单 worker 内已经不简单。

到了多 worker，更复杂。

有三种思路。

### 56.15.1 worker-local prefix cache

每个 worker 只缓存自己处理过的 prefix。

优点：

1. 实现简单。
2. 不需要跨 worker KV 传输。
3. cache lookup 快。

缺点：

1. 相同 prefix 打到不同 worker 会重复 prefill。
2. cache 命中依赖 sticky routing。
3. 热点 prefix 可能造成 worker 热点。

这是最常见的起点。

### 56.15.2 centralized prefix directory

维护一个全局目录，记录哪个 worker 有某个 prefix。

```text
prefix_hash -> worker_ids
```

router 可以优先把请求发给已有 prefix 的 worker。

但目录只记录位置，不复制 KV。

真正的 KV 仍然在 worker 本地。

优点是能提高 locality。

缺点是目录有一致性和过期问题。

### 56.15.3 remote KV reuse

理论上，可以从远端 worker 拉取 KV cache。

但这很难：

1. KV 很大。
2. GPU 间跨节点传输慢。
3. layout 和 block table 要兼容。
4. 传输期间 prefix 可能被 eviction。
5. 拉取成本可能超过重新 prefill。

所以 remote KV reuse 需要非常谨慎。

只有当 prefix 很长、网络很快、复用很多时才可能划算。

## 56.16 Scheduler 和 parallel group 的关系

在模型并行 worker 里，scheduler 仍然做 request-level 和 token-level 调度。

但它调度的 batch 会被底层执行器分发到多个 GPU rank。

可以分成两层：

```text
Scheduler layer:
  decide which requests/tokens run this step

Distributed executor layer:
  run the batch across TP/PP ranks
```

scheduler 不应该关心每个 matmul 怎么切。

executor 不应该决定全局 waiting queue 谁先跑。

边界可以这样理解：

1. scheduler 输出 logical batch。
2. block manager 输出 logical block table。
3. distributed executor 把 batch 和 block table 转成 rank-local tensors。
4. 每个 rank 运行自己的 shard。
5. executor 汇总 logits 或 sampled token。
6. scheduler/output processor 更新 request 状态。

这个边界清晰，系统才可维护。

## 56.17 rank-local 数据

TP/PP 下，每个 rank 都有自己的本地数据。

常见 rank-local 数据包括：

1. 权重 shard。
2. KV cache shard。
3. CUDA graph 或 kernel workspace。
4. local block table。
5. local attention metadata。
6. local hidden states。
7. communication buffers。

不能假设 rank 0 拿到了全部数据。

很多 debug 难点都来自这里。

例如 logits 可能只有最后一个 pipeline stage 才有。

例如某些 TP 切分下，每个 rank 只有 vocab 的一部分 logits。

例如 KV block OOM 可能只发生在某个 rank。

所以 metrics 也要区分 global 和 rank-local。

## 56.18 分布式执行的指标

除了前面章节的指标，模型并行 worker 还需要：

```text
tp_size
pp_size
rank_forward_ms{rank}
rank_kv_used_blocks{rank}
rank_kv_free_blocks{rank}
rank_oom_total{rank}
collective_all_reduce_ms
collective_all_gather_ms
collective_send_recv_ms
pipeline_stage_idle_ms{stage}
pipeline_bubble_ratio
executor_sync_ms
```

这些指标能回答：

1. 是计算慢，还是通信慢。
2. 哪个 rank 或 stage 是 straggler。
3. pipeline bubble 是否严重。
4. KV cache 是否在某个 rank 上不均衡。
5. TP size 是否过大。
6. PP 切层是否不均衡。

没有这些指标，多 GPU 性能问题很难定位。

## 56.19 TP size 怎么选

选择 TP size 时要看四件事：

1. 模型权重是否能放下。
2. KV cache 是否能放下。
3. 单请求延迟要求。
4. GPU 互联带宽。

如果模型单卡放不下，TP 是必要的。

如果模型单卡能放下，但想提高吞吐，TP 不一定总是好。

因为 TP 会增加通信和同步。

粗略经验：

1. NVLink 机器上可以更积极使用 TP。
2. PCIe 机器上 TP 通信成本更明显。
3. decode batch 小时，TP size 过大容易拖慢 TPOT。
4. prefill-heavy workload 更容易从 TP 获益。
5. latency-sensitive 场景要谨慎扩大 TP size。

最终还是要用第 53 章的 benchmark 方法验证。

## 56.20 PP size 怎么选

选择 PP size 时要看：

1. 每个 stage 的权重显存。
2. 每个 stage 的 KV cache 显存。
3. stage 计算是否均衡。
4. stage 间通信带宽。
5. decode 并发是否足够填满 pipeline。

PP size 太小，单 stage 放不下。

PP size 太大，pipeline bubble 和通信开销变大。

在线 serving 里，PP 往往比 TP 更难调。

因为 decode 是逐 token 依赖的，无法像训练那样轻松堆大量 micro-batch。

如果必须使用 PP，要特别关注：

1. stage idle time。
2. pipeline bubble ratio。
3. per-stage latency。
4. micro-batch scheduling。
5. prefill/decode 混合时的 pipeline 调度。

## 56.21 常见错误

错误一：把 TP group 里的每张 GPU 当独立 worker 路由。

```text
结果：请求无法完整 forward，调度边界错误。
```

错误二：只看多卡总算力，不看通信。

```text
结果：TP size 增大后 decode 反而更慢。
```

错误三：认为 prefill 快就代表 decode 也快。

```text
结果：长 prompt benchmark 表现很好，但在线聊天 TPOT 很差。
```

错误四：PP 切层只按层数平均。

```text
结果：不同 stage 计算不均衡，最慢 stage 决定整体速度。
```

错误五：cancel 只释放某个 rank 的 KV。

```text
结果：其他 rank KV 泄漏，后续请求 OOM。
```

错误六：假设 KV cache 可以随便跨 worker 迁移。

```text
结果：迁移成本超过重新 prefill，或者 layout/block table 不一致。
```

错误七：prefix cache 做了全局目录，但没有处理 eviction 过期。

```text
结果：router 以为 worker 有 prefix，实际已经被淘汰。
```

错误八：只采集 rank 0 指标。

```text
结果：某个非 0 rank OOM、慢 kernel 或通信等待被掩盖。
```

错误九：忽略 pipeline bubble。

```text
结果：多卡显存够了，但 GPU 利用率很低。
```

错误十：把模型并行当成免费扩展。

```text
结果：显存问题解决了，但延迟、通信、容错和调度复杂度显著上升。
```

## 56.22 面试高频问题

问题一：tensor parallel 和 replica parallel 有什么区别？

回答要点：replica parallel 是多个完整模型副本，每个请求只进入一个 worker，router 做请求级负载均衡。tensor parallel 是一个模型的权重和计算切到多张 GPU，一个请求需要整个 TP group 协同完成。TP 可以解决单卡放不下的问题，但会引入跨卡通信和同步。

问题二：为什么 TP size 不是越大越好？

回答要点：TP size 增大能降低单卡权重和计算压力，但会增加 all-reduce、all-gather 等通信开销。decode 阶段 batch tokens 小，通信 latency 占比更高，TP size 过大可能让 TPOT 变差。选择 TP size 要结合显存、互联带宽、prefill/decode workload 和 benchmark。

问题三：pipeline parallel 为什么在线 inference 难？

回答要点：PP 按层切模型，需要多个 micro-batch 填满 pipeline。训练可以用大量 micro-batch，但在线 decode 每个请求的下一 token 依赖上一 token 采样结果，并发不足时 pipeline bubble 很大。还要处理 stage imbalance、stage 间通信和分布式 KV 释放。

问题四：TP/PP 下 KV cache 怎么放？

回答要点：TP 下 attention heads 通常被切到不同 rank，每个 rank 保存自己 head shard 的 KV，逻辑 block 在所有 TP ranks 上要保持一致。PP 下模型按层切分，每个 stage 保存自己负责层的 KV。request finish、cancel、preemption 时，所有 rank/stage 都要一致清理。

问题五：为什么正在 decode 的请求不容易跨 worker 迁移？

回答要点：因为请求的 KV cache 很大，且分布在 GPU memory、TP ranks 或 PP stages 上。迁移要保持 layer、head shard、block table、position、dtype、layout 和 sampling 状态一致，传输成本很高。多数 serving 系统会让请求在原 worker 上完成，而不是运行中迁移。

## 56.23 标准回答模板

如果面试官问“多 GPU LLM inference 里 tensor parallel、pipeline parallel 和 KV cache 有什么工程难点”，可以这样回答：

```text
我会先区分并行粒度。replica parallel 是多份完整模型副本，router 在 worker 之间做请求级负载均衡；tensor parallel 是一个模型的权重和计算切到多张 GPU，一个请求需要整个 TP group 协同；pipeline parallel 是按层切到不同 stage，一个请求依次经过各 stage。从 router 视角看，一个能独立完成 forward 的 TP/PP group 才是一个 worker，不能把 group 里的每张 GPU 当独立 worker。

TP 的核心 trade-off 是显存和计算分摊换通信。比如 column parallel、row parallel 会带来 all-gather 或 all-reduce。prefill 阶段 batch tokens 多，计算量大，通信相对容易摊薄；decode 阶段每轮 token 少，如果并发不够，通信 latency 会显著影响 TPOT。所以 TP size 不是越大越好，要结合显存、NVLink/PCIe、workload 和 benchmark 选。

PP 的难点是 pipeline bubble 和 stage imbalance。在线 decode 有逐 token 依赖，不能像训练一样随便堆未来 token 填 pipeline。并发不足时很多 stage 会 idle。PP 下 KV cache 也按层分布在不同 stage，cancel、finish、preemption 都要跨 stage 一致清理。

KV cache 是 serving 的关键状态。TP 下 KV 通常按 heads 分片，每个 rank 保存本地 shard，但逻辑 block table 必须一致；PP 下每个 stage 保存自己层的 KV。由于 KV cache 很大且和 block table、position、layout、rank shard 绑定，正在 decode 的请求很难跨 worker 迁移。实际系统通常依赖 request locality、sticky routing 和 worker-local prefix cache，而不是频繁迁移 KV。
```

## 56.24 小练习

1. 画出 replica parallel、TP=2、PP=2 的 worker 拓扑图。
2. 给一个 32 heads、TP=4 的模型写出每个 rank 负责哪些 heads。
3. 估算一个请求的 KV cache 大小。
4. 写一个函数，根据 `layers/tokens/kv_heads/head_dim/dtype_bytes` 计算 KV bytes。
5. 解释 column parallel 为什么可能需要 all-gather。
6. 解释 row parallel 为什么可能需要 all-reduce。
7. 给 TP worker 增加 `rank_kv_free_blocks` 指标。
8. 给 TP worker 增加 `collective_all_reduce_ms` 指标。
9. 给 PP worker 增加 `pipeline_stage_idle_ms` 指标。
10. 构造一个 decode batch 很小的 benchmark，观察 TPOT。
11. 构造一个长 prompt prefill benchmark，观察 TP 的收益。
12. 模拟 cancel 时只释放 rank 0 KV，说明会发生什么。
13. 设计一个 worker-local prefix cache 的 sticky routing 策略。
14. 设计一个 `prefix_hash -> worker_ids` 的 prefix directory。
15. 比较重新 prefill 和远程 KV 拉取在不同 prefix 长度下的成本。

## 56.25 本章总结

多 GPU inference 不是简单把 GPU 数量乘上去。

replica parallel、tensor parallel 和 pipeline parallel 的边界不同。

replica parallel 让多个完整模型副本独立处理请求，router 做请求级负载均衡。

tensor parallel 把一个模型的张量和计算切到多张 GPU，一个请求需要整个 TP group 协同完成。

pipeline parallel 按层切分模型，一个请求依次经过多个 stage。

从 router 视角看，能独立完成 forward 的并行组才是 worker。

TP 的关键成本是通信。

TP size 越大，单卡显存和计算压力越小，但 all-reduce、all-gather 和同步开销越明显。

prefill 和 decode 对 TP 的表现不同。

prefill 计算量大，更容易摊薄通信；decode batch 小时，TPOT 容易受通信影响。

PP 的关键问题是 pipeline bubble 和 stage imbalance。

在线 decode 有逐 token 依赖，并发不足时 pipeline 利用率会下降。

KV cache 在模型并行下也会分布式化。

TP 下 KV 通常按 head shard 分布在不同 ranks。

PP 下 KV 按层分布在不同 stages。

request finish、cancel、preemption 必须在所有 ranks/stages 上一致清理。

KV cache 很大，并且和 block table、position、layout、rank shard 强绑定。

所以正在 decode 的请求很难跨 worker 迁移。

实际系统通常依赖 request locality、worker-local prefix cache 和 sticky routing，而不是频繁搬迁 KV。

下一章可以继续讨论：如果要把这些能力产品化，需要怎样设计 OpenAI-compatible API、流式协议、错误码和限流鉴权。
