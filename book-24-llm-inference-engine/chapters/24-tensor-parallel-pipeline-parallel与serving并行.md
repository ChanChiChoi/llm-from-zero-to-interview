# 第 24 章 Tensor Parallel、Pipeline Parallel 与 serving 并行

上一章讲了 Prefix Caching 与 Prompt Cache：如何复用相同前缀的 KV blocks，降低重复 prefill 成本和 TTFT。

本章继续讲 serving engine 的另一条主线：并行。模型太大、吞吐不够、显存不足、请求太多时，单 GPU 已经不够，需要把模型或请求分布到多张 GPU、多台机器上。此时 Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel 会一起出现。

一句话概括：

> Serving 并行的核心不是“GPU 越多越快”，而是根据模型大小、显存压力、通信拓扑和业务目标，选择把模型切开、把层切开、把请求复制分流，还是把 MoE experts 分布出去。

## 24.1 本章目标

读完本章，你应该能讲清：

1. Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel 分别解决什么问题。
2. 为什么 vLLM 单机多卡常用 TP，跨节点常用 TP+PP。
3. TP 如何影响显存、KV Cache、通信和 TPOT。
4. PP 如何影响显存、延迟、pipeline bubble 和跨节点部署。
5. DP 如何提高吞吐，以及为什么每个 DP rank 有独立 KV Cache。
6. MoE 模型中 EP 和 DP/TP 的关系。
7. 面试中如何根据模型大小、GPU 拓扑和业务目标选择 serving 并行策略。

## 24.2 先把四种并行放在一张表里

| 并行方式 | 切分对象 | 主要目的 | 典型收益 | 典型代价 |
| --- | --- | --- | --- | --- |
| Tensor Parallel | 单层矩阵和 attention heads | 让单层参数/计算分布到多 GPU | 降低单卡权重显存，支持大模型 | 高频 collective 通信 |
| Pipeline Parallel | 模型层 | 把不同层放到不同 GPU/节点 | 降低单卡权重显存，支持跨节点 | pipeline bubble，单请求延迟变长 |
| Data Parallel | 请求 batch 或 engine replica | 复制模型处理不同请求 | 横向扩吞吐，隔离请求 | 每个 replica 都要一份权重和 KV cache |
| Expert Parallel | MoE experts | 把不同 expert 放到不同 GPU | 扩展 MoE expert 容量 | token routing 和负载不均 |

服务端最常见组合是：

```text
单模型副本内部：TP 或 TP + PP
多个模型副本之间：DP 或外部负载均衡
MoE expert 层：EP 或 TP/DP 组合
```

## 24.3 Serving 并行和训练并行有什么不同

训练并行要处理 forward、backward、gradient、optimizer states、checkpoint 和参数同步。

Serving 并行主要处理：

1. forward。
2. KV Cache。
3. continuous batching。
4. 请求调度。
5. streaming 输出。
6. 多副本负载均衡。
7. TPOT、TTFT 和吞吐。

因此 serving 下的并行目标也不同。

训练里你关心 step time、global batch size、显存能不能放下 optimizer。

Serving 里你关心：

1. 模型权重是否能放下。
2. 剩余显存能放多少 KV Cache。
3. 单 token decode 延迟是否稳定。
4. 请求能不能 continuous batching。
5. 通信是否拖慢 TPOT。
6. 多副本是否能提升总 QPS。

不要把训练里的并行结论直接搬到推理 serving。

## 24.4 Tensor Parallel 的直觉

Tensor Parallel，简称 TP，是把模型单层内部的大矩阵切到多张 GPU 上。

以 Linear 层为例：

```text
Y = X W
```

如果 `W` 太大，可以按列或按行切分：

```text
W = [W0, W1, W2, W3]
```

每张 GPU 持有一部分权重，计算一部分结果，最后通过通信合并。

Transformer 中 TP 常作用在：

1. QKV projection。
2. attention heads。
3. output projection。
4. MLP up/gate/down projection。
5. lm head。

直觉是：

```text
一层太宽，一张 GPU 放不下或算不过来，就把这一层拆给多张 GPU。
```

## 24.5 TP 在 serving 中解决什么问题

TP 首先解决权重显存问题。

例如一个模型权重需要 140GB，单张 80GB GPU 放不下。TP=2 后，每张 GPU 可能只放约一半权重，再加上 KV Cache 和 runtime buffer，模型就可能跑起来。

TP 还会间接增加 KV Cache 可用空间。

因为权重被分片后，每张 GPU 上权重占用减少，剩余 HBM 可以留给 KV Cache。

vLLM 优化文档中也提到，当频繁 preemption 是因为 KV cache 空间不足时，可以考虑增加 `tensor_parallel_size`，因为权重分片后每张卡可用于 KV Cache 的空间可能增加。

但 TP 不是免费提升。

TP 的代价是每层都可能需要通信。

常见通信包括：

1. all-reduce。
2. all-gather。
3. reduce-scatter。
4. tensor shard 间同步。

decode 阶段每生成一个 token 都要走所有层，所以 TP 通信会直接影响 TPOT。

## 24.6 TP 为什么偏好同机高速互联

TP 通信频率高，通常每层都要通信。

如果 TP group 跨节点，通信会走网络，延迟和带宽都比同机 NVLink/NVSwitch 差很多。

所以 vLLM parallelism 文档建议：

```text
跨节点部署时，常见做法是 tensor_parallel_size = 每个节点的 GPU 数，pipeline_parallel_size = 节点数。
```

例如 2 台机器，每台 8 张 GPU：

```text
TP = 8
PP = 2
```

这样 TP 通信主要留在节点内部，跨节点只发生 pipeline stage 间的激活传输。

如果你把 TP=16 横跨两台机器，每一层都可能跨节点通信，TPOT 可能明显变差。

面试中可以这样说：

```text
TP 是高频通信并行，最好放在 NVLink/NVSwitch 范围内；跨节点更适合用 PP 或 DP，除非网络非常强并且没有其他选择。
```

## 24.7 TP 和 KV Cache 的关系

TP 不只切权重，也会影响 KV Cache 布局。

在 attention 中，KV Cache 通常按层、KV heads、head dim、token 位置存储。

如果 attention heads 或 KV heads 被切到不同 TP ranks，每个 rank 只持有自己负责的 KV head 分片。

简化理解：

```text
TP rank 0: 部分 attention heads + 对应 KV cache
TP rank 1: 部分 attention heads + 对应 KV cache
TP rank 2: 部分 attention heads + 对应 KV cache
TP rank 3: 部分 attention heads + 对应 KV cache
```

这意味着：

1. 每个 rank 的 KV Cache 是模型分片的一部分。
2. block manager 和 attention metadata 要和 TP rank 对齐。
3. prefix cache 命中也要在所有 TP ranks 上保持一致。
4. 一个 TP group 中任一 rank 出错，整个 replica 通常不可用。

GQA/MQA 会进一步影响 KV Cache，因为 KV heads 少于 query heads，KV Cache 本身更小，但 TP 切分时仍要确保 head 分配合法。

## 24.8 Pipeline Parallel 的直觉

Pipeline Parallel，简称 PP，是把模型层切到不同 GPU 或节点上。

例如 40 层模型，用 PP=4：

```text
GPU 0: layers 0-9
GPU 1: layers 10-19
GPU 2: layers 20-29
GPU 3: layers 30-39
```

输入先经过 GPU 0，再传给 GPU 1，再传给 GPU 2，最后到 GPU 3。

直觉是：

```text
模型太深或节点装不下，就按层切开。
```

PP 的好处是每个 GPU 只保存一部分层的权重，显存压力下降。

PP 的代价是单个请求要按 stage 顺序走，天然增加串行路径。

## 24.9 PP 适合什么场景

PP 适合：

1. 模型单节点放不下，需要跨节点。
2. GPU 之间没有高速 NVLink，TP 通信太贵。
3. 模型层数多，按层切分比较自然。
4. GPU 显存不均或模型大小不能被 TP 均匀切分。
5. 已经把 TP 用到合理上限，还需要继续扩展模型容量。

vLLM 文档提到一个边界情况：如果模型能放在单节点内，但 GPU 数不能均匀切分模型，或者节点没有 NVLink，比如某些 L40S 机器，使用 pipeline parallel 可能比 tensor parallel 有更低通信开销和更好吞吐。

原因是 PP 不是每层都 all-reduce，而是 stage 之间传递 hidden states。

但 PP 也有明显代价：

1. 单请求延迟可能变长。
2. pipeline bubble 会浪费设备。
3. continuous batching 和 pipeline 调度结合更复杂。
4. stage 间负载不均会拖慢整体。

## 24.10 Pipeline bubble 是什么

Pipeline bubble 指某些 stage 没有活干的空泡。

例如 PP=4，只有一个 microbatch：

```text
time 0: stage 0 工作，stage 1/2/3 空闲
time 1: stage 1 工作，stage 0/2/3 空闲
time 2: stage 2 工作，stage 0/1/3 空闲
time 3: stage 3 工作，stage 0/1/2 空闲
```

如果没有足够 microbatches 填满 pipeline，GPU 利用率会很差。

训练中常用 microbatch 填 pipeline。

Serving 中 continuous batching 可以提供多个请求，但 decode 阶段每个 step 粒度小、请求长度不同、streaming 对延迟敏感，所以 pipeline 调度仍然复杂。

简单说：

```text
PP 能让更大模型放得下，但可能用延迟和调度复杂度换显存。
```

## 24.11 Data Parallel 的直觉

Data Parallel，简称 DP，在 serving 中通常不是训练里的梯度同步，而是复制模型副本处理不同请求。

例如 DP=4：

```text
DP rank 0: 一套 engine + worker group + KV cache
DP rank 1: 一套 engine + worker group + KV cache
DP rank 2: 一套 engine + worker group + KV cache
DP rank 3: 一套 engine + worker group + KV cache
```

请求由 router 或 API server 分发到不同 DP ranks。

DP 的目标是横向扩吞吐。

如果单个 replica 已经可以放下模型，但 QPS 不够，就增加 replicas。

DP 的代价是每个 replica 都要加载一份模型权重，并维护独立 KV Cache。

所以 DP 解决的是吞吐扩展，不解决单个模型副本放不下的问题。

## 24.12 DP rank 的 KV Cache 是独立的

vLLM data parallel 文档明确提到，每个 DP engine 有独立 KV Cache。

这有几个重要影响。

第一，prefix cache 不自动跨 DP rank 共享。

同一个 prompt 如果第一次打到 rank 0，第二次打到 rank 1，rank 1 不一定有 prefix cache 命中。

第二，路由会影响 cache 命中率。

如果同类 prompt 能稳定路由到同一个 DP rank，prefix cache 命中率可能更高。

第三，容量按 rank 分开计算。

`max_num_seqs`、KV block pool、waiting/running queue 都是每个 rank 自己一套。

第四，某个 rank 过载不会自动让另一个 rank 共享它的 KV state。

因此 DP 负载均衡最好考虑：

1. running queue。
2. waiting queue。
3. KV cache free blocks。
4. prefix cache locality。
5. tenant 或会话粘性。

简单 round-robin 不一定最优。

## 24.13 Internal、Hybrid 和 External DP load balancing

vLLM data parallel 文档中提到几种在线 DP 部署形态。

Internal load balancing：

```text
一个对外 API endpoint
API server 内部把请求分发到多个 DP ranks
```

优点是使用简单，对用户透明。

缺点是 API server 可能成为瓶颈，尤其 DP 很大时。

Hybrid load balancing：

```text
每个节点有自己的 API server
上游 LB 分发到节点
节点内 API server 分发到本地 DP ranks
```

优点是减少跨节点流量，避免单 head node 瓶颈。

External load balancing：

```text
每个 DP rank 或每组 ranks 是独立部署
外部 router 根据实时指标分发请求
```

优点是灵活，适合大规模生产平台。

缺点是平台侧要自己处理健康检查、负载指标、prefix cache locality、灰度和限流。

## 24.14 Expert Parallel 的直觉

Expert Parallel，简称 EP，主要用于 MoE 模型。

MoE 层里有多个 experts，每个 token 只路由到少数 experts。

如果 experts 很多，单张 GPU 放不下所有 expert 权重，就可以把 experts 分到多张 GPU。

简化图：

```text
GPU 0: expert 0, expert 1
GPU 1: expert 2, expert 3
GPU 2: expert 4, expert 5
GPU 3: expert 6, expert 7
```

执行时，每个 token 先经过 router，决定去哪些 experts，然后发生 token dispatch 和 combine。

EP 的收益是扩展 MoE expert 容量。

EP 的代价是：

1. token 路由通信复杂。
2. expert 负载可能不均。
3. 小 batch 时某些 experts 利用率低。
4. DP、TP、EP 组合时通信 group 更复杂。

vLLM 文档提到，对于 MoE 模型，可以使用 data parallel attention 配合 expert parallel 或 tensor parallel MoE layers。

## 24.15 如何选择并行策略

可以按这个决策树：

第一步，模型单卡能不能放下？

如果能放下，先单卡或多副本 DP，不要急着 TP。单卡通常延迟最低、实现最简单。

第二步，单节点多卡能不能放下？

如果单卡放不下，但单节点多卡能放下，优先 TP，尤其节点内有 NVLink/NVSwitch。

第三步，单节点也放不下？

如果需要多节点，常见选择是：

```text
TP = 每个节点 GPU 数
PP = 节点数
```

这样减少跨节点 TP 高频通信。

第四步，模型副本能跑但吞吐不够？

增加 DP 或外部 replicas。

第五步，是 MoE 模型吗？

考虑 EP，尤其 expert 权重很大或 expert 层成为瓶颈时。

第六步，看业务目标。

低延迟聊天优先减少通信和 pipeline bubble。

批量离线生成可以更激进地用多卡并行提高吞吐。

长上下文 RAG 要关注 KV Cache 容量和 prefix cache locality。

## 24.16 并行度对进程数的影响

结合第 22 章，worker 数量通常和 GPU 数一致。

一个 DP rank 内：

```text
worker_count = TP * PP
```

总 GPU 数大致是：

```text
total_gpus = DP * TP * PP
```

进程数还包括 API server、engine core 和可能的 DP coordinator。

例如 DP=4，TP=2，PP=1：

```text
总 GPU = 4 * 2 * 1 = 8
engine core = 4
GPU workers = 8
API servers = 通常至少 1，可能按 DP 或配置扩展
DP coordinator = MoE/DP 场景下可能需要
```

如果 DP=1，TP=8，PP=2：

```text
总 GPU = 1 * 8 * 2 = 16
engine core = 1
GPU workers = 16
```

前者是 4 个模型副本，每个副本 2 卡 TP。

后者是 1 个大模型副本，跨 2 个 pipeline stages，每个 stage 内 8 卡 TP。

它们解决的问题完全不同。

## 24.17 并行度对显存的影响

显存主要看：

1. 权重。
2. KV Cache。
3. activation/workspace。
4. 通信 buffer。
5. runtime overhead。

TP 对权重显存：降低单卡权重占用。

TP 对 KV Cache：通常每个 rank 保存自己分片相关的 KV，单卡 KV 压力也可能下降，但 metadata 和通信 buffer 增加。

PP 对权重显存：每张 GPU 只放部分层，显著降低单卡权重。

PP 对 KV Cache：每个 stage 只保存自己层的 KV Cache，按层分布。

DP 对权重显存：每个 DP rank 都复制一份模型副本，不降低单副本权重显存。

DP 对 KV Cache：每个 DP rank 独立维护 KV Cache，容量和命中也独立。

EP 对权重显存：expert 权重被分布到不同 GPU，降低单卡 expert 权重压力。

EP 对通信：增加 token routing 相关 buffer 和通信。

## 24.18 并行度对延迟和吞吐的影响

TP：

1. 可能降低单卡计算压力。
2. 但每层通信会增加 decode 延迟。
3. TP 太大时，通信占比上升，TPOT 可能变差。

PP：

1. 能让更大模型运行。
2. 但单请求要经过多个 stage，延迟变长。
3. 如果 batch 或 microbatch 不足，pipeline bubble 降低吞吐。

DP：

1. 通常最直接提升总吞吐。
2. 单请求延迟不一定下降。
3. 如果路由不均，部分 rank 排队严重，整体 p99 变差。

EP：

1. MoE 中能扩展 expert 容量。
2. 但 routing、dispatch、combine 和 expert 负载不均会影响延迟。

所以不能简单说“加 GPU 提升性能”。要问：

```text
瓶颈是权重放不下、KV cache 不够、单副本吞吐不够，还是 expert 容量不够？
```

不同瓶颈对应不同并行策略。

## 24.19 拓扑：PCIe、NVLink、NVSwitch、IB/RDMA

并行策略必须看硬件拓扑。

PCIe-only 单机：

TP 高频通信可能受限，PP 有时更合适。

NVLink/NVSwitch 单机：

适合 TP group 放在同机内，通信带宽和延迟更好。

多节点 InfiniBand/RDMA：

适合跨节点 PP、DP，或在网络足够强时跨节点 TP，但需要非常注意 NCCL、GDRDMA、拓扑和容器配置。

普通以太网：

不适合高频跨节点 TP。可以做外部 DP，但跨节点模型切分会非常吃力。

vLLM 文档中提到，可以通过 NCCL 日志确认是否走了高效网络路径，例如 GPUDirect RDMA，而不是普通 socket。

面试里要强调：

```text
并行策略不是只看 GPU 数量，而是看 GPU 之间怎么连。
```

## 24.20 常见配置例子

例子一：7B 模型，单张 80GB GPU 可以放下。

推荐先单卡，多副本 DP 扩吞吐。

```text
TP=1, PP=1, DP=N
```

例子二：70B 模型，单卡放不下，但 8 卡 NVSwitch 节点能放下。

推荐单节点 TP。

```text
TP=8, PP=1, DP=1
```

例子三：超大模型，单节点 8 卡也放不下，需要 2 个节点。

常见做法：

```text
TP=8, PP=2, DP=1
```

例子四：70B 模型可以用 4 卡 TP 跑起来，但 QPS 不够，有 16 张 GPU。

可以部署 4 个副本：

```text
TP=4, PP=1, DP=4
```

例子五：MoE 模型，attention 层不大，但 experts 很多。

可以考虑：

```text
DP attention + EP experts
```

具体还要看框架支持和 expert 负载。

## 24.21 常见工程坑

坑一：跨节点 TP。

表现是模型能启动，但 TPOT 很差。原因是每层 collective 都走跨节点网络。

坑二：盲目增大 TP。

TP 太大后，每张卡计算变少，通信占比变高，延迟可能变差。

坑三：DP 后 prefix cache 命中率下降。

因为每个 DP rank 的 KV cache 独立，请求被打散后相同前缀不一定落到同一 rank。

坑四：PP stage 不均衡。

某个 stage 层数多、MoE 多或 attention 更重，会拖慢整个 pipeline。

坑五：CPU 核数没跟上。

DP/TP 增加 worker 和 engine 数后，CPU 调度、tokenization、detokenization、IPC 都可能成为瓶颈。

坑六：NCCL 或网络配置错误。

表现为启动 hang、forward hang、吞吐异常低、跨节点不稳定。

坑七：把 DP 当成解决单副本显存不足的方法。

DP 是复制模型，不会让单个模型副本更容易放下。

## 24.22 面试官会怎么问

问题一：TP、PP、DP 的区别是什么？

回答要点：TP 切单层张量，PP 切模型层，DP 复制模型副本处理不同请求。TP/PP 解决单副本放不下或计算分布，DP 解决吞吐扩展。

问题二：为什么 TP 更适合同机？

回答要点：TP 每层都有高频 collective 通信，需要 NVLink/NVSwitch 这类高速互联；跨节点 TP 会严重拖慢 decode TPOT。

问题三：为什么跨节点常用 TP+PP？

回答要点：把 TP 限制在每个节点内部，PP 跨节点按层传 hidden states，减少每层跨节点 collective。

问题四：DP 为什么会影响 prefix cache？

回答要点：每个 DP rank 有独立 KV Cache 和 prefix cache，请求路由到不同 rank 会影响 cache locality，需要 cache-aware routing 或会话粘性。

问题五：什么时候用 PP 而不是 TP？

回答要点：模型需要跨节点、GPU 无高速互联、层切分更自然、TP 切分不均或 TP 通信代价过高时，可以考虑 PP。

问题六：MoE 为什么需要 EP？

回答要点：MoE expert 数量和权重很大，EP 把 experts 分布到不同 GPU，降低单卡 expert 显存，但引入 token routing 和负载均衡问题。

## 24.23 标准回答模板

如果面试官问“如何为 vLLM serving 选择 TP/PP/DP”，可以这样回答：

```text
我会先区分目标：是单个模型副本放不下，还是吞吐不够。

如果模型单卡能放下，通常先不用 TP/PP，直接单卡或多副本 DP，延迟和复杂度最低。如果单卡放不下但单节点多卡能放下，优先使用 tensor parallel，尤其在 NVLink/NVSwitch 机器上，因为 TP 可以把单层权重和计算分到多张 GPU，同时释放单卡显存给 KV cache。

如果单节点也放不下，需要跨节点，常见策略是 TP 等于每节点 GPU 数，PP 等于节点数。这样 TP 高频通信留在节点内部，跨节点只做 pipeline stage 之间的激活传输，避免每层 collective 都跨节点。

如果一个副本已经能跑，但 QPS 不够，就增加 data parallel 或外部 replicas。DP 每个 rank 有独立 engine、worker group 和 KV cache，所以要注意负载均衡和 prefix cache locality。MoE 模型还要考虑 expert parallel，把 expert 权重分布到不同 GPU，同时处理 token routing 和 expert 负载均衡。

最终选择要结合模型大小、GPU 显存、KV cache 需求、NVLink/IB 拓扑、TTFT/TPOT 目标和业务流量模式。
```

## 24.24 小练习

1. 画出 TP=4、PP=1、DP=1 的 worker 拓扑，并说明每个 worker 持有什么。
2. 画出 TP=8、PP=2、DP=1 的两节点拓扑，说明为什么 TP 放在节点内。
3. 解释为什么 DP=4 后 prefix cache 命中率可能下降。
4. 假设有 16 张 GPU，一个模型需要 4 张 GPU TP 才能放下，如何配置才能部署 4 个副本？
5. 对比 PCIe-only 8 卡和 NVSwitch 8 卡在 TP 场景下的差异。
6. 设计一个排查 TPOT 变差的 checklist，至少包含通信、调度、KV cache、CPU 四类指标。

## 24.25 本章总结

Serving 并行的核心是先判断瓶颈。

TP 把单层张量切到多 GPU，适合单机高速互联内的大模型推理；PP 把模型层切到不同 GPU 或节点，适合单节点放不下或 TP 跨节点太贵的场景；DP 复制模型副本处理不同请求，适合横向扩吞吐；EP 把 MoE experts 分布出去，适合 expert 权重和计算成为瓶颈的 MoE 模型。

并行策略会同时影响 worker 数量、进程数量、权重显存、KV Cache 容量、prefix cache locality、通信开销、TTFT、TPOT 和 p99 延迟。真正的工程选择必须结合模型大小、GPU 拓扑、业务延迟目标和流量分布，而不是简单把并行度开大。

下一章会继续讲 vLLM 性能瓶颈、参数调优和工程坑，把前面几章的 scheduler、KV cache、prefix cache、worker 架构和并行策略统一放到压测与线上调优视角下。
