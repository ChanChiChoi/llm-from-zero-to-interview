# 第 38 章 KV Cache 迁移、共享和路由

上一章讲了 PD 分离的系统架构：router、prefill worker、decode worker 和 KV transfer backend 共同组成一个跨组件状态机。

本章深入最关键的问题：KV Cache 如何从 Prefill 侧交给 Decode 侧。

PD 分离真正难的地方，不是把请求分给两个 worker，而是把 Prefill 阶段生成的大量 KV cache，以正确、快速、可恢复的方式交给 Decode 阶段使用。

一句话概括：

> KV Cache 迁移、共享和路由是 PD 分离能否落地的核心。它决定了 TTFT 是否真的下降，TPOT 是否稳定，以及系统复杂度是否可控。

## 38.0 本讲资料边界与第二轮精修口径

本讲第二轮精修前，先按 `WRITING_PLAN.md` 对公开资料做校准：参考 SGLang PD Disaggregation 文档对 KV transfer backend、Mooncake / NIXL、bootstrap port、heterogeneous TP staging buffer、PD multiplexing 和 routing policy 的公开口径；参考 vLLM disaggregated prefilling 文档对 KV connector、prefill / decode 示例和实验性边界的说明；参考 Mooncake 论文对 KV cache-centric disaggregated architecture、KV cache 管理和传输瓶颈的系统动机说明；并参考 PagedAttention / vLLM 资料对 paged KV blocks、block table 和 KV cache metadata 的稳定抽象。

本讲只讲 KV cache 迁移、共享和路由的通用系统问题：KV 大小估算、paged block metadata、layout / dtype / TP 兼容、push / pull / remote read、decode reservation、KV-aware routing、prefix-aware routing、失败清理和多租户隔离。不把某个框架版本的真实 connector 字段、RDMA/NIXL/Mooncake API、CUDA IPC 细节、网络拓扑参数或远端 KV cache 产品实现写成通用标准。

本讲新增 demo 是教学版 KV transfer router：用 0 依赖 Python 模拟 KV metadata 兼容性检查、decode capacity 预留、路由 cost score、prefix cache 命中、多租户缓存阻断、短 prompt recompute 路径和 transfer 失败清理计划，帮助把“KV-aware routing”从口号落到可运行证据。

## 38.1 本章目标

读完本章，你应该能讲清：

1. 为什么 KV Cache 是 PD 分离里的核心数据。
2. KV Cache 的大小、layout 和 metadata 为什么重要。
3. KV Cache 迁移有哪些基本模式。
4. push、pull、共享内存、远端 KV cache 的区别。
5. Decode worker 为什么通常要先预留 KV blocks。
6. KV-aware routing 和 prefix-aware routing 怎么工作。
7. KV transfer 失败时应该如何清理和重试。
8. 面试中如何回答“PD 分离的 KV Cache 怎么传”。

## 38.2 先回忆 KV Cache 是什么

Transformer 自回归生成时，每一层 attention 都会产生 Key 和 Value。

对于已经处理过的 token，后续 decode 不需要重新计算它们的 K/V，而是直接读取缓存。

这就是 KV Cache。

简化理解：

```text
prompt tokens
  -> prefill forward
  -> per-layer K/V tensors
  -> KV cache
  -> decode reads KV cache repeatedly
```

在 unified engine 中，KV cache 通常留在同一个 worker 的 GPU memory 里。

```text
same worker:
  prefill writes KV
  decode reads KV
```

PD 分离后，Prefill 和 Decode 不在同一个 worker 上。

```text
prefill worker:
  prefill writes KV

decode worker:
  decode needs KV
```

中间就出现了问题：

```text
How to move or share KV?
```

这就是本章主题。

## 38.3 为什么 KV Cache 不是普通数据

KV Cache 和普通 RPC payload 不一样。

普通 RPC 传的可能是：

1. prompt 文本。
2. token ids。
3. request metadata。
4. 采样参数。
5. JSON 或 protobuf。

KV Cache 的特点是：

1. 体积大。
2. 在 GPU memory 中。
3. 每一层都有。
4. 和模型结构强相关。
5. 和 dtype 强相关。
6. 和 tensor parallel 切分方式强相关。
7. 和 paged KV block layout 强相关。
8. 需要被 Decode 高频读取。

所以不能把它简单理解成“把一个数组传过去”。

它更像是一个带有复杂物理布局和所有权语义的分布式内存对象。

## 38.4 KV Cache 大小估算

理解 KV 迁移，必须先理解它为什么贵。

对一个 Transformer 模型，KV cache 大小大致和下面因素成正比：

1. 层数 `num_layers`。
2. KV head 数 `num_kv_heads`。
3. head dimension `head_dim`。
4. token 数 `seq_len`。
5. dtype 字节数。
6. batch 中请求数量。

单个 token 的 KV 大小可以粗略写成：

```text
per_token_kv_bytes
  = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
```

这里的 `2` 表示 K 和 V。

如果是 32 层、32 个 KV heads、head_dim 128、FP16：

```text
32 * 2 * 32 * 128 * 2 bytes
= 524288 bytes
≈ 512 KB / token
```

如果 prompt 有 4000 tokens：

```text
512 KB * 4000
≈ 2 GB
```

实际模型可能使用 GQA/MQA，`num_kv_heads` 会小一些，KV cache 也会小一些。

但结论不变：长 prompt 的 KV cache 可能非常大。

这也是为什么 PD 分离不能只看计算时间，还必须看 KV transfer 成本。

## 38.5 KV Cache 迁移的基本问题

Prefill worker 生成 KV 后，Decode worker 要使用它。

这件事至少包含 8 个问题：

1. 源 KV 在哪里。
2. 目标 worker 是谁。
3. 目标 worker 是否有足够 KV capacity。
4. 源和目标的 KV layout 是否一致。
5. 源和目标的 TP rank mapping 是否一致。
6. 传输走什么 backend。
7. 传输失败如何清理。
8. 传完后谁拥有 KV 的生命周期。

因此，KV transfer 不是一个单纯的数据复制动作。

它需要 router、prefill worker、decode worker 和 transfer backend 共同协作。

## 38.6 一个最小 KV Transfer 流程

最小流程可以画成：

```text
client
  -> router
  -> prefill worker
       run prefill
       produce KV
  -> router chooses decode worker
  -> decode worker reserves KV blocks
  -> transfer KV from prefill to decode
  -> decode worker starts decode
  -> stream tokens
```

注意一个关键点：

Decode worker 通常要先预留目标 KV blocks。

原因是 transfer 不能把 KV 传到一个“不知道放哪”的地方。

目标侧需要提前告诉 transfer backend：

1. request id。
2. 目标 block ids。
3. 每层目标地址。
4. 每个 TP rank 的目标地址。
5. 可接收的 dtype 和 layout。

然后源侧才能把 KV 写过去或让目标侧拉取。

## 38.7 Push 模式

Push 模式是指：Prefill worker 主动把 KV 推给 Decode worker。

流程：

```text
router -> choose prefill worker
router -> choose decode worker
decode worker -> reserve destination blocks
prefill worker -> run prefill
prefill worker -> push KV to decode worker
decode worker -> mark KV ready
decode worker -> start decode
```

优点：

1. 流程直观。
2. Prefill 结束后可以立即发起传输。
3. Router 容易理解 transfer 进度。
4. 适合源侧更主动的实现。

缺点：

1. Prefill worker 需要知道目标地址或目标 block mapping。
2. 源侧逻辑更复杂。
3. 如果 decode worker 已经变化，源侧需要处理目标失效。
4. 多租户和权限隔离更麻烦。

Push 模式适合系统控制面比较集中、P/D worker 通信关系明确的场景。

## 38.8 Pull 模式

Pull 模式是指：Decode worker 主动从 Prefill worker 拉取 KV。

流程：

```text
router -> choose prefill worker
prefill worker -> run prefill
prefill worker -> expose source KV metadata
router -> choose decode worker
decode worker -> reserve destination blocks
decode worker -> pull KV from prefill worker
decode worker -> mark KV ready
decode worker -> start decode
```

优点：

1. Decode worker 掌握目标内存分配。
2. 更符合“消费方拉取”的语义。
3. Decode worker 可以根据本地状态决定何时拉取。
4. 对目标侧资源控制更自然。

缺点：

1. Decode worker 需要访问源侧 KV metadata。
2. 源侧要在一段时间内保留 KV。
3. 如果 pull 延迟，prefill worker 的 KV memory 会被占住。
4. Router 要管理 source KV 的过期和清理。

Pull 模式适合 decode 侧资源控制更复杂的场景。

## 38.9 Push 和 Pull 的本质区别

Push 和 Pull 的区别不只是方向不同。

它们背后是所有权和状态机不同。

| 维度 | Push | Pull |
| --- | --- | --- |
| 谁发起数据移动 | Prefill worker | Decode worker |
| 谁更复杂 | Prefill 侧 | Decode 侧 |
| 目标 block 何时确定 | 传输前必须确定 | 拉取前必须确定 |
| 源 KV 保留时间 | 通常较短 | 可能更长 |
| 失败清理重点 | 目标写入失败 | 源 KV 过期和拉取失败 |
| 适合场景 | 控制面集中 | 消费方资源控制强 |

生产系统中也可能混合使用：控制面看起来是 push，底层传输实现却是由目标侧 RDMA read 完成。

所以面试时不要死记“push 一定是源写，pull 一定是目标读”。

更准确的说法是：

> Push/Pull 描述的是控制语义；底层数据面可能由 RDMA write、RDMA read、GPU IPC、NIXL 或其他 backend 完成。

## 38.10 共享模式

除了迁移，还有一种思路是共享。

共享模式下，Prefill 产生的 KV 不一定复制到 Decode worker 的本地 KV cache，而是让 Decode worker 以某种方式直接访问同一份 KV。

可能的共享方式包括：

1. 同机多进程 GPU IPC。
2. NVLink 互联下的 peer access。
3. RDMA 暴露远端 GPU memory。
4. 独立 KV cache server。
5. 分层 KV cache 系统。

共享的优点：

1. 可以减少复制。
2. 可以提升 prefix cache 复用。
3. 对长 prompt 可能更省带宽。
4. 可以把 KV cache 做成系统级资源。

共享的缺点：

1. 远端读取延迟可能影响 decode TPOT。
2. 一致性和生命周期更复杂。
3. 故障边界变大。
4. 调试和观测更困难。
5. 对网络和硬件拓扑更敏感。

因此，共享不是天然优于迁移。

如果 decode 每一步都要频繁读取远端 KV，而远端读取延迟不可控，TPOT 可能变差。

## 38.11 迁移模式和共享模式的区别

迁移模式：

```text
Prefill worker GPU memory
  -> copy KV
  -> Decode worker GPU memory
```

Decode 后续主要读本地 KV。

共享模式：

```text
Prefill side / remote cache keeps KV
  -> Decode worker reads shared KV
```

Decode 后续可能读远端 KV，或者部分读本地、部分读远端。

对比：

| 维度 | 迁移 | 共享 |
| --- | --- | --- |
| 数据复制 | 有 | 少或无 |
| Decode 读取 | 本地为主 | 可能远端 |
| TPOT 稳定性 | 通常更好 | 取决于远端访问 |
| TTFT | 受复制时间影响 | 受远端访问准备影响 |
| Prefix 复用 | 需要额外机制 | 更自然 |
| 生命周期 | 相对清晰 | 更复杂 |

一个简单判断：

如果请求生成很长，decode 会反复读 KV，那么把 KV 放到 decode 本地通常更稳。

如果大量请求共享长 prefix，共享或远端 KV cache 的收益可能更大。

## 38.12 Paged KV Cache 对迁移的影响

现代 serving engine 通常不会为每个请求分配连续大块 KV memory。

它们会使用 paged KV cache。

也就是把 KV cache 切成固定大小的 block。

```text
request A tokens:
  block 7 -> block 9 -> block 21

request B tokens:
  block 3 -> block 4
```

这样可以减少内存碎片，并支持更灵活的调度和复用。

但它也让 KV transfer 更复杂。

因为要传的不只是一个连续 tensor，而是一组 block mapping：

1. 第几个 token 在哪个 block。
2. block 内 offset 是多少。
3. 每层 K/V 的物理地址。
4. 每个 TP rank 的 block 对应关系。
5. source block 和 destination block 的映射。

所以 KV transfer metadata 至少要包含：

```text
request_id
model_id
model_version
num_layers
kv_dtype
block_size
seq_len
source_worker_id
source_rank_ids
source_block_ids
destination_worker_id
destination_rank_ids
destination_block_ids
layout_version
```

没有这些 metadata，decode worker 即使拿到了 bytes，也不知道怎么解释。

## 38.13 Layout 兼容性

KV layout 描述的是 K/V 在内存中的排列方式。

不同 engine 或不同 kernel 可能使用不同 layout。

例如：

```text
[layer][block][head][token][dim]
```

或者：

```text
[block][layer][kv][head][token][dim]
```

或者为了 kernel 访问效率做了更复杂的打包。

如果 prefill side 和 decode side layout 不一致，就会出现两个问题：

1. 传过去不能直接用。
2. 需要 layout conversion。

layout conversion 可能发生在：

1. Prefill worker 传输前。
2. Transfer backend 中。
3. Decode worker 接收后。

但无论在哪做，它都会引入额外成本。

所以生产系统里通常希望 P/D worker 使用一致的：

1. 模型版本。
2. dtype。
3. TP/DP 配置。
4. KV block size。
5. KV layout version。
6. attention kernel 期望的内存格式。

否则 KV transfer 会从“搬运”变成“搬运加转换”。

## 38.14 Tensor Parallel 下的 KV 迁移

Tensor Parallel 会把模型参数和计算切到多个 GPU rank 上。

KV cache 也会跟着切分。

例如一个 TP=4 的 worker：

```text
prefill worker P:
  rank 0 owns part of KV
  rank 1 owns part of KV
  rank 2 owns part of KV
  rank 3 owns part of KV
```

Decode worker 也是 TP=4：

```text
decode worker D:
  rank 0 needs corresponding KV
  rank 1 needs corresponding KV
  rank 2 needs corresponding KV
  rank 3 needs corresponding KV
```

如果 P 和 D 的 TP 配置一致，mapping 相对简单：

```text
P rank 0 -> D rank 0
P rank 1 -> D rank 1
P rank 2 -> D rank 2
P rank 3 -> D rank 3
```

如果 TP 配置不一致，问题会复杂很多。

例如 P 是 TP=8，D 是 TP=4。

这时 KV shard 可能需要重分片。

```text
P rank 0 + P rank 1 -> D rank 0
P rank 2 + P rank 3 -> D rank 1
...
```

重分片意味着：

1. 更多网络传输。
2. 更多内存拷贝。
3. 更复杂的 metadata。
4. 更高的失败概率。
5. 更难保证性能稳定。

所以实际系统中，P/D pool 通常会尽量保持兼容的并行配置。

异构不是不能做，但必须把 KV reshaping 的成本算进去。

## 38.15 Decode 为什么要先预留 KV Blocks

这是 PD 分离面试里很常见的问题。

Decode worker 在接收 KV 前通常要先预留 KV blocks，原因有四个。

第一，确认容量。

如果 decode worker 没有足够 KV capacity，传输到一半才发现放不下，会浪费 prefill 计算和网络带宽。

第二，确定目标地址。

KV transfer 需要明确写入位置。

不管是 RDMA write、GPU copy，还是目标侧 pull，都需要目标 buffer。

第三，建立 block mapping。

Decode 的 paged attention 需要知道每个 request 的 block table。

传输前就要知道：

```text
source block 10 -> destination block 81
source block 11 -> destination block 82
source block 20 -> destination block 83
```

第四，避免 race condition。

如果多个请求同时进入 decode worker，没有 reservation，就可能发生资源竞争。

所以正确流程通常是：

```text
decode worker checks capacity
decode worker reserves blocks
decode worker returns destination metadata
KV transfer starts
decode worker marks request ready
```

## 38.16 KV Transfer Backend 做什么

KV transfer backend 是数据面能力。

它不一定是一个独立服务，也可能是 worker 内部的一组库和通信线程。

它负责：

1. 注册 GPU memory。
2. 建立连接。
3. 暴露 source buffer。
4. 暴露 destination buffer。
5. 执行 copy、read 或 write。
6. 处理 completion event。
7. 上报 timeout 和 error。
8. 做必要的数据校验。
9. 控制并发传输。
10. 清理传输资源。

常见底层能力包括：

1. CUDA IPC。
2. NCCL。
3. RDMA。
4. NVLink / NVSwitch。
5. GPUDirect RDMA。
6. NIXL。
7. Mooncake 等 KV transfer / KV cache 系统。

面试中不需要把每种 backend 的 API 细节背下来。

但要讲清楚：

> Transfer backend 的价值是把 KV cache 从“worker 内部内存”变成“跨 worker 可移动或可访问的数据对象”。

## 38.17 Router 在 KV 迁移中的职责

Router 不直接搬 KV，但它必须理解 KV。

Router 至少要维护：

1. request id。
2. tenant id。
3. model id。
4. model version。
5. tokenizer version。
6. prompt length。
7. expected output length。
8. prefill worker id。
9. decode worker id。
10. source KV metadata。
11. destination KV metadata。
12. transfer state。
13. timeout deadline。
14. retry count。
15. cleanup owner。

Router 的关键职责不是“转发请求”，而是协调状态：

```text
WAITING
  -> PREFILL_ASSIGNED
  -> PREFILL_RUNNING
  -> PREFILL_DONE
  -> DECODE_RESERVED
  -> KV_TRANSFERRING
  -> KV_READY
  -> DECODING
  -> FINISHED
```

如果任何一步失败，router 要知道该释放哪些资源。

例如：

```text
PREFILL_DONE but KV_TRANSFERRING failed
```

需要清理：

1. Prefill side source KV。
2. Decode side reserved blocks。
3. Transfer backend connection / handle。
4. Request state。
5. Client response。

## 38.18 KV-aware Routing

普通路由可能只看 worker 负载。

例如：

```text
choose least loaded worker
```

KV-aware routing 要看更多信息。

对于 decode worker，router 需要关注：

1. 剩余 KV capacity。
2. running request 数。
3. decode batch size。
4. 当前 TPOT。
5. pending transfer 数。
6. 网络拓扑距离。
7. 是否已有相关 prefix KV。
8. 模型版本是否匹配。

一个简单的 score 可以写成：

```text
score(decode_worker)
  = a * available_kv_blocks
  - b * running_requests
  - c * pending_transfers
  - d * network_cost(prefill_worker, decode_worker)
  - e * current_tpot
```

真实系统不会这么简单，但思路类似。

Router 不能只问“哪个 worker 空”。

它还要问：

```text
哪个 worker 能低成本接住这个请求的 KV，并稳定完成 decode？
```

## 38.19 Prefix-aware Routing

很多请求会共享相同前缀。

例如：

1. 同一个 system prompt。
2. 同一个 RAG 模板。
3. 同一段长文档上下文。
4. 同一个 agent 工具说明。
5. 同一个多轮对话历史前缀。

如果某个 worker 已经缓存了这段 prefix 的 KV，新请求路由到它，可能可以复用 KV，减少 prefill。

这叫 prefix-aware routing 或 cache-aware routing。

示例：

```text
request A prompt:
  [system prompt][doc][question 1]

request B prompt:
  [system prompt][doc][question 2]
```

二者共享：

```text
[system prompt][doc]
```

如果共享 prefix 很长，复用收益很大。

Router 可以维护 prefix cache index：

```text
prefix_hash -> worker ids that hold KV
```

路由时优先选择已经有 prefix KV 的 worker。

但这也有代价：

1. cache index 需要更新。
2. prefix hash 要考虑 tokenizer 和 model version。
3. 热点 prefix 可能导致 worker 倾斜。
4. prefix KV 生命周期要管理。
5. 多租户之间不能随意共享。

所以 prefix-aware routing 不是简单地“命中缓存就去那个 worker”。

还要平衡 load、KV capacity 和 tail latency。

## 38.20 Prefix Cache 和 PD 分离的关系

Prefix cache 在 unified engine 中已经有价值。

PD 分离后，它更重要，也更复杂。

原因是：

1. Prefill 成本集中在 prefill pool。
2. Decode worker 需要 prompt KV 才能生成。
3. Prefix KV 可能在 P 侧、D 侧或远端 cache 中。
4. Router 需要知道 KV locality。

几种情况：

```text
case 1: prefix KV already on selected decode worker
  -> skip or reduce transfer

case 2: prefix KV on another decode worker
  -> decide migrate, remote read, or recompute

case 3: prefix KV on prefill worker
  -> maybe reuse prefill side KV

case 4: prefix KV in remote KV cache
  -> fetch or attach

case 5: no prefix KV
  -> run full prefill
```

这说明 PD router 需要理解 locality。

没有 locality 信息的 router，很难发挥 KV cache 复用的价值。

## 38.21 Recompute、Migrate、Remote Read 的取舍

当 KV 不在目标 decode worker 上时，系统有三种选择。

第一，重新计算。

```text
run prefill again on selected worker
```

第二，迁移 KV。

```text
copy KV to selected decode worker
```

第三，远端读取。

```text
decode reads KV from remote cache or remote worker
```

取舍可以这样看：

| 方案 | 优点 | 缺点 |
| --- | --- | --- |
| Recompute | 实现简单，不依赖远端 KV | 浪费计算，TTFT 高 |
| Migrate | Decode 本地读，TPOT 稳 | 需要传输，TTFT 受影响 |
| Remote Read | 减少复制，可复用 | Decode 受远端延迟影响 |

一个经验判断：

1. prompt 短，recompute 可能最简单。
2. prompt 长、输出也长，migrate 通常更稳。
3. prefix 很长且被大量请求共享，remote cache 可能值得。
4. 网络很差时，远端读取会显著拖累 decode。

## 38.22 KV Cache 的生命周期

KV cache 生命周期必须清楚，否则系统很容易泄漏 GPU memory。

一个请求的 KV 生命周期大致是：

```text
allocate source blocks
  -> write prompt KV
  -> expose source metadata
  -> transfer or share
  -> release source blocks
  -> decode appends new KV
  -> finish / abort
  -> release decode blocks
```

在 PD 分离中，source 和 destination 的生命周期可能不同。

Prefill side source KV：

1. Prefill 期间必须存在。
2. Transfer 完成前必须存在。
3. Transfer 完成后通常可以释放。
4. 如果做 prefix cache，可能继续保留。

Decode side KV：

1. Transfer 前预留。
2. Transfer 完成后进入 ready。
3. Decode 期间持续追加。
4. 请求结束后释放。
5. 如果做 cache reuse，部分 prefix blocks 可能保留。

生命周期管理的核心问题是：

```text
谁负责释放？什么时候释放？失败时释放哪些？
```

## 38.23 Reference Count

如果 KV cache 支持共享，就需要引用计数或类似机制。

例如多个请求共享同一个 prefix block：

```text
prefix block X
  used by request A
  used by request B
  used by request C
```

这时不能在 request A 完成后就释放 block X。

需要：

```text
ref_count(prefix block X) = 3
request A done -> ref_count = 2
request B done -> ref_count = 1
request C done -> ref_count = 0 -> can evict
```

生产系统中还要考虑：

1. request abort。
2. timeout。
3. worker crash。
4. router state 丢失。
5. prefix cache eviction。
6. 多租户隔离。

引用计数不一定显式叫 `ref_count`，但系统必须有等价的所有权管理。

## 38.24 KV Eviction 策略

KV cache 不是无限的。

当 GPU memory 不够时，系统要决定驱逐哪些 KV。

常见策略：

1. LRU：驱逐最近最少使用。
2. LFU：驱逐使用频率低的。
3. TTL：超过时间就驱逐。
4. Size-aware：优先驱逐大对象。
5. Cost-aware：保留重算成本高的 prefix。
6. Tenant-aware：按租户配额驱逐。

对 LLM serving 来说，cost-aware 很重要。

因为一个很长的 system prompt 或文档 prefix，虽然最近用得不多，但重算成本很高。

而一个短 prefix 即使命中率高，收益也可能不大。

因此更合理的 eviction score 可能同时考虑：

```text
eviction_score
  = recompute_cost
  * expected_reuse_probability
  / memory_size
```

实际系统会更复杂，但核心思想是：

> KV cache eviction 不只是内存问题，也是调度和成本问题。

## 38.25 多级 KV Cache

KV cache 可以分层。

例如：

```text
GPU KV cache
  -> CPU memory
  -> local SSD
  -> remote KV cache service
```

不同层级的特点：

| 层级 | 延迟 | 容量 | 适合存什么 |
| --- | --- | --- | --- |
| GPU | 最低 | 最小 | 活跃 decode KV |
| CPU | 中等 | 较大 | 暂时不活跃 prefix |
| SSD | 较高 | 更大 | 低频长 prefix |
| Remote cache | 取决于网络 | 可扩展 | 跨 worker 复用 prefix |

多级 KV cache 可以提升复用率，但会引入新的问题：

1. Promote / demote 策略。
2. Prefetch 策略。
3. Eviction 策略。
4. 一致性管理。
5. 远端访问对 TPOT 的影响。

第 40 章会专门讨论多级 KV Cache。

本章只需要记住：

> PD 分离让 KV 从单 worker 内部对象变成系统级资源，多级 KV cache 是这个方向的自然延伸。

## 38.26 KV Transfer 的性能指标

观测 KV transfer，不能只看总 QPS。

需要分阶段指标。

关键指标包括：

1. `kv_transfer_bytes_total`。
2. `kv_transfer_latency_ms`。
3. `kv_transfer_bandwidth_gbps`。
4. `kv_transfer_pending_count`。
5. `kv_transfer_timeout_count`。
6. `kv_transfer_failure_count`。
7. `source_kv_retention_ms`。
8. `decode_reserved_blocks`。
9. `decode_kv_utilization`。
10. `prefix_cache_hit_rate`。
11. `prefix_cache_bytes`。
12. `remote_kv_read_latency_ms`。
13. `time_to_first_token_ms`。
14. `time_per_output_token_ms`。

这些指标要结合看。

例如：

```text
TTFT increases
  and kv_transfer_latency increases
  and pending_transfer_count increases
```

说明瓶颈可能在 KV transfer。

再例如：

```text
TPOT increases
  and remote_kv_read_latency increases
```

说明远端 KV 读取可能拖慢 decode。

## 38.27 KV Transfer 的常见瓶颈

常见瓶颈有：

1. 网络带宽不足。
2. RDMA 连接数过多。
3. GPU memory copy 和 compute 争抢。
4. KV layout conversion 太慢。
5. 目标 decode worker KV capacity 不足。
6. Pending transfer 队列过长。
7. 源 KV 保留太久导致 prefill worker 内存压力。
8. 小块 transfer 太多导致调度开销高。
9. 跨机架或跨可用区传输。
10. Prefix cache 命中导致热点 worker 过载。

一个典型问题是：

```text
Prefill 很快，Decode 也不忙，但 TTFT 仍然高。
```

这时要检查中间是否卡在：

1. Decode block reservation。
2. KV metadata exchange。
3. KV transfer queue。
4. Actual data movement。
5. Transfer completion notification。

## 38.28 故障一：Prefill 成功但 Transfer 失败

这是最典型的故障。

状态：

```text
prefill done
source KV exists
decode blocks reserved
transfer failed
```

可能原因：

1. 网络 timeout。
2. Decode worker crash。
3. Source worker crash。
4. Destination buffer invalid。
5. Layout metadata mismatch。
6. RDMA connection reset。

处理策略：

1. 标记 request transfer failed。
2. 释放 decode reserved blocks。
3. 释放或保留 source KV。
4. 判断是否可以重试 transfer。
5. 如果不能重试，重新 prefill 或返回错误。

能否重试取决于 source KV 是否还存在。

如果 source KV 还在，可以选择新的 decode worker 再传一次。

如果 source KV 已经没了，就只能重新 prefill。

## 38.29 故障二：Decode Reserved 后 Worker 崩溃

状态：

```text
decode worker reserved blocks
before or during transfer
decode worker crashed
```

处理策略：

1. Router 标记 decode worker unhealthy。
2. 放弃该 worker 上的 reservation。
3. 如果 source KV 仍在，选择新的 decode worker。
4. 重新 reserve blocks。
5. 重新 transfer。
6. 如果 source KV 不在，重新 prefill。

关键点是：

```text
reserved blocks on crashed worker are not reliable resources
```

不能假设它们之后还能恢复。

## 38.30 故障三：Decode 中途失败

Decode 已经开始输出 token 后再失败，最麻烦。

状态：

```text
client already received partial tokens
decode worker crashed
```

这时很难做到完全无感恢复。

原因是：

1. 已输出 token 需要保持一致。
2. 采样可能有随机性。
3. Decode 期间新生成 token 的 KV 也在 worker 上。
4. 其他 worker 不一定有完整 KV。
5. Streaming 协议可能已经把内容发给客户端。

可选策略：

1. 返回错误，要求客户端重试。
2. 保存 generated tokens，重新 prefill prompt + generated tokens。
3. 如果有 checkpointed KV，迁移到新 worker。
4. 对确定性采样场景尝试恢复。

生产系统通常会把 decode 中途失败视为难以无感恢复的故障。

面试中可以明确说：

> Prefill 失败和 transfer 失败通常可以重试；decode 中途失败因为已经对外输出 token，恢复语义更复杂。

## 38.31 KV Metadata 不一致

KV metadata 不一致会导致严重问题。

例如：

1. 模型版本不一致。
2. dtype 不一致。
3. block size 不一致。
4. layout version 不一致。
5. TP rank mapping 不一致。
6. prompt tokenization 不一致。
7. RoPE scaling 配置不一致。

这些问题不一定会立即报错。

更糟糕的是，它们可能导致 silent correctness bug。

也就是模型还能输出，但输出语义是错的。

所以 transfer 前必须校验：

```text
model_id matches
model_version matches
tokenizer_version matches
kv_dtype matches
layout_version matches
parallel_config matches
rope_config matches
```

宁可拒绝请求，也不要把不兼容的 KV 交给 decode。

## 38.32 安全和多租户隔离

KV cache 不是普通缓存。

它包含用户输入的语义信息。

在多租户系统中，KV cache 共享必须非常谨慎。

风险包括：

1. 租户 A 复用租户 B 的 prefix KV。
2. Prefix hash 碰撞导致错误共享。
3. Debug dump 泄露 KV metadata。
4. Remote KV cache 权限控制不严。
5. Worker 复用后没有清理旧 KV。

基本原则：

1. Prefix cache key 必须包含 tenant boundary。
2. 不同租户默认不共享 KV。
3. KV metadata 不能暴露给无关租户。
4. Worker 下线或复用时要清理敏感 KV。
5. 远端 KV cache 要有访问控制。

面试中提到多租户隔离，会显得你对生产系统更敏感。

## 38.33 KV 路由策略示例

一个简化的 decode worker 选择策略：

```text
for each decode worker:
  if model_version mismatch:
    skip
  if available_kv_blocks < required_blocks:
    skip
  if pending_transfers too high:
    penalize
  if prefix_cache_hit:
    reward
  if same_node_as_prefill:
    reward
  if current_tpot high:
    penalize

choose best score
```

对应伪代码：

```python
def choose_decode_worker(request, prefill_worker, candidates):
    best_worker = None
    best_score = None

    for worker in candidates:
        if worker.model_version != request.model_version:
            continue
        if worker.available_kv_blocks < request.required_kv_blocks:
            continue

        score = 0
        score += worker.available_kv_blocks * 0.1
        score -= worker.running_requests * 1.0
        score -= worker.pending_transfers * 2.0
        score -= worker.current_tpot_ms * 0.5
        score -= network_cost(prefill_worker, worker)

        if worker.has_prefix(request.prefix_hash):
            score += request.prefix_reuse_score

        if best_score is None or score > best_score:
            best_score = score
            best_worker = worker

    return best_worker
```

这只是教学伪代码。

真实系统里 score 的权重要来自压测和线上观测。

## 38.34 什么时候不应该迁移 KV

不是所有请求都适合 KV 迁移。

不适合的情况包括：

1. Prompt 很短，重新计算比传输更便宜。
2. 网络拥塞严重，迁移会拖慢 TTFT。
3. Decode worker 本地 KV capacity 紧张。
4. P/D layout 不兼容，需要昂贵转换。
5. 请求预计只生成很短输出。
6. 源 KV 已经快过期或不可靠。
7. 跨机房传输成本过高。

这时可以选择：

1. 在同一个 worker 上完成请求。
2. 重新 prefill。
3. 路由到更近的 decode worker。
4. 降级为 unified scheduling。
5. 拒绝或排队。

优秀的 PD 系统应该能判断：

```text
transfer benefit > transfer cost
```

而不是无脑迁移所有 KV。

## 38.35 和 Chunked Prefill 的关系

Chunked Prefill 是把长 prompt prefill 切成多个 chunk 执行。

它和 KV 迁移有关，但不是同一个问题。

Chunked Prefill 解决的是：

1. 长 prompt prefill 太大。
2. Prefill 阶段阻塞 decode。
3. 大 batch 形态不稳定。
4. TTFT 和 TPOT 之间需要折中。

KV 迁移解决的是：

1. Prefill 生成的 KV 如何给 Decode。
2. P/D worker 间如何共享状态。
3. Decode worker 如何接管请求。

二者可以结合。

例如：

```text
long prompt
  -> chunked prefill on prefill pool
  -> progressively produce KV
  -> transfer KV chunks
  -> decode starts after enough KV ready
```

下一章会继续讲 Chunked Prefill 与 Disaggregated Prefill。

## 38.36 面试官会怎么问

问题一：PD 分离中 KV Cache 怎么从 Prefill 传到 Decode？

回答要点：Prefill worker 生成 prompt KV 和 metadata；router 选择 decode worker；decode worker 先检查 capacity 并预留目标 KV blocks；然后通过 transfer backend 以 push 或 pull 的方式把 KV 从 source blocks 迁移到 destination blocks；完成后 decode worker 标记 KV ready，进入 decode loop。

问题二：为什么 Decode worker 要先预留 KV blocks？

回答要点：为了确认容量、确定目标地址、建立 source 到 destination 的 block mapping，并避免并发请求抢占资源。否则传输到一半发现放不下，会浪费 prefill 计算和网络带宽。

问题三：Push 和 Pull 有什么区别？

回答要点：Push 是 Prefill 侧主动把 KV 交给 Decode 侧，Pull 是 Decode 侧根据 source metadata 主动拉取。区别主要是控制语义和状态所有权；底层实现可能仍然是 RDMA read/write 或其他 backend。

问题四：KV-aware routing 看哪些信息？

回答要点：看 decode worker 的 KV 剩余容量、running requests、pending transfers、TPOT、模型版本、网络拓扑、prefix cache 命中、P/D worker 距离等，不只是看 worker 是否空闲。

问题五：KV transfer 失败怎么处理？

回答要点：释放 decode reserved blocks，清理 transfer handle；如果 source KV 还在，可以换 decode worker 重试 transfer；如果 source KV 不在，可能需要重新 prefill；如果已经进入 decode 并对外输出 token，则无感恢复更困难。

问题六：为什么模型版本不一致不能复用 KV？

回答要点：KV cache 是模型内部状态，和权重、RoPE、tokenizer、dtype、layout、TP 配置都相关。版本不一致即使 shape 能对上，也可能产生 silent correctness bug。

## 38.37 标准回答模板

如果面试官问“PD 分离里的 KV Cache 迁移怎么设计”，可以这样回答：

```text
PD 分离中，Prefill worker 处理 prompt 后会生成每层 KV cache。因为 Decode worker 不在同一个进程或同一个 GPU 上，所以必须把这部分 KV 迁移或共享给 Decode worker。

一个典型流程是：router 先选择 prefill worker 执行 prefill；prefill 完成后生成 source KV metadata，包括 request id、seq len、block ids、layout、dtype、TP rank mapping 等。然后 router 选择 decode worker。Decode worker 先检查自己的 KV capacity，并预留目标 KV blocks，返回 destination metadata。随后 transfer backend 通过 push 或 pull 的方式把 source blocks 映射到 destination blocks。传输完成后 decode worker 把请求标记为 KV ready，加入 decode loop 开始持续生成。

这里最关键的是 metadata 和状态机。KV cache 不是普通 bytes，它和模型版本、KV layout、paged block table、dtype、tensor parallel rank mapping 都强相关。Router 需要做 KV-aware routing，考虑 decode worker 的剩余 KV blocks、pending transfers、TPOT、网络拓扑和 prefix cache locality。

故障处理也很重要。如果 transfer 失败，要释放 decode 侧预留 block，并根据 source KV 是否还存在决定重试 transfer 还是重新 prefill。如果 decode 已经开始输出 token，再失败就很难完全无感恢复，因为客户端可能已经收到部分 token。
```

## 38.38 KV Transfer 路由公式、隔离门禁和可运行 demo

KV transfer 和 routing 的核心不是“选择一个空闲 decode worker”，而是在兼容性、容量、拓扑、缓存命中、多租户隔离和 transfer 成本之间做取舍。

设一次 KV transfer 请求为：

```math
Q_i=(r_i,u_i,m_i,t_i,P_i,O_i,h_i,L_i,d_i,p_i,z_i)
```

其中 `r_i` 是 request id，`u_i` 是 tenant，`m_i` 是 model version，`t_i` 是 tokenizer version，`P_i` 是 prompt token 数，`O_i` 是预计输出 token 数，`h_i` 是 prefix hash，`L_i` 是 layout version，`d_i` 是 KV dtype，`p_i` 是 TP size，`z_i` 是 source zone。

目标 decode worker `w` 必须先过兼容门：

```math
G_{\mathrm{compat}}(i,w)=G_mG_tG_dG_LG_p
```

分别表示 model、tokenizer、dtype、layout 和 parallel config 兼容。任何一项不兼容，都不应该把 KV 交给这个 worker。

目标容量门可以写成：

```math
B_{\mathrm{need},i}=\left\lceil\frac{P_i+O_i}{S_{\mathrm{block}}}\right\rceil
```

```math
G_{\mathrm{cap}}(i,w)=\mathbf{1}[B_{\mathrm{free},w}\ge B_{\mathrm{need},i}]
```

KV transfer 成本粗估为：

```math
T_{\mathrm{transfer}}(i,w)=\frac{P_i m_{\mathrm{kv}}}{B_{\mathrm{net}}(z_i,z_w)}
```

一个教学版 decode worker score 可以写成：

```math
S(i,w)=aB_{\mathrm{free},w}-bR_w-cQ_w-dT_{\mathrm{tpot},w}-eT_{\mathrm{transfer}}(i,w)+fC_{\mathrm{prefix}}(i,w)
```

其中 `R_w` 是 running sequences，`Q_w` 是 pending transfers，`T_{\mathrm{tpot},w}` 是当前 TPOT，`C_{\mathrm{prefix}}` 表示同租户可复用 prefix cache 的奖励。注意“同租户”很重要，跨租户 prefix 命中必须阻断，不能因为 hash 命中就复用。

下面的 demo 覆盖三类路径：

1. `long_rag`：长 prompt 且同租户 prefix cache 命中，选择 cache 更有价值的 decode worker。
2. `short_chat`：短 prompt，重算比迁移更简单，走 recompute。
3. `cross_tenant`：看到了他租户 prefix cache，但不能复用，只能按无缓存路径路由。

```python
from dataclasses import dataclass, field


@dataclass
class KVRequest:
    request_id: str
    tenant: str
    model_version: str
    tokenizer_version: str
    prompt_tokens: int
    output_tokens: int
    dtype: str
    layout_version: str
    tp_size: int
    source_zone: str
    prefix_hash: str


@dataclass
class DecodeWorkerCandidate:
    name: str
    zone: str
    model_version: str
    tokenizer_version: str
    dtype: str
    layout_version: str
    tp_size: int
    available_blocks: int
    running_sequences: int
    pending_transfers: int
    current_tpot_ms: float
    prefix_cache: dict = field(default_factory=dict)


class ToyKVTransferRouter:
    def __init__(self, block_size=128, kv_mib_per_token=0.015625):
        self.block_size = block_size
        self.kv_mib_per_token = kv_mib_per_token
        self.skipped = []

    def _required_blocks(self, request):
        return (
            request.prompt_tokens + request.output_tokens + self.block_size - 1
        ) // self.block_size

    def _network_rate(self, source_zone, target_zone):
        return 1200.0 if source_zone == target_zone else 300.0

    def _compatible(self, request, worker):
        checks = {
            "model": request.model_version == worker.model_version,
            "tokenizer": request.tokenizer_version == worker.tokenizer_version,
            "dtype": request.dtype == worker.dtype,
            "layout": request.layout_version == worker.layout_version,
            "tp": request.tp_size == worker.tp_size,
        }
        return all(checks.values()), checks

    def route(self, request, workers):
        required_blocks = self._required_blocks(request)
        recompute_ms = round(request.prompt_tokens / 900.0, 3)
        kv_transfer_mib = request.prompt_tokens * self.kv_mib_per_token
        if request.prompt_tokens < 512:
            return {
                "request_id": request.request_id,
                "decision": "recompute",
                "required_blocks": required_blocks,
                "recompute_ms": recompute_ms,
                "kv_transfer_mib": round(kv_transfer_mib, 1),
                "reason": "short_prompt_recompute_cheaper",
            }

        scored = []
        tenant_cache_blocked = False
        for worker in workers:
            compatible, checks = self._compatible(request, worker)
            if not compatible:
                self.skipped.append({
                    "request_id": request.request_id,
                    "worker": worker.name,
                    "checks": checks,
                })
                continue
            if worker.available_blocks < required_blocks:
                self.skipped.append({
                    "request_id": request.request_id,
                    "worker": worker.name,
                    "checks": {"capacity": False},
                })
                continue
            rate = self._network_rate(request.source_zone, worker.zone)
            transfer_ms = kv_transfer_mib / rate
            cache_tenant = worker.prefix_cache.get(request.prefix_hash)
            prefix_reuse = cache_tenant == request.tenant
            if cache_tenant is not None and cache_tenant != request.tenant:
                tenant_cache_blocked = True
            prefix_bonus = 25.0 if prefix_reuse else 0.0
            score = (
                worker.available_blocks * 0.02
                - worker.running_sequences * 1.0
                - worker.pending_transfers * 3.0
                - worker.current_tpot_ms * 0.4
                - transfer_ms
                + prefix_bonus
            )
            scored.append({
                "worker": worker,
                "score": round(score, 3),
                "transfer_ms": round(transfer_ms, 3),
                "prefix_reuse": prefix_reuse,
                "tenant_cache_blocked": tenant_cache_blocked,
            })

        if not scored:
            return {
                "request_id": request.request_id,
                "decision": "route_failed",
                "required_blocks": required_blocks,
            }

        best = max(scored, key=lambda row: row["score"])
        worker = best["worker"]
        worker.available_blocks -= required_blocks
        worker.pending_transfers += 1
        return {
            "request_id": request.request_id,
            "decision": "migrate",
            "decode_worker": worker.name,
            "required_blocks": required_blocks,
            "reserved_blocks": required_blocks,
            "kv_transfer_mib": round(kv_transfer_mib, 1),
            "transfer_ms": best["transfer_ms"],
            "score": best["score"],
            "prefix_reuse": best["prefix_reuse"],
            "tenant_cache_blocked": best["tenant_cache_blocked"],
            "failure_plan": "release_destination_blocks_and_retry_if_source_kv_exists",
        }

    def audit(self, requests, workers):
        rows = [self.route(request, workers) for request in requests]
        summary = {
            "requests": len(rows),
            "migrate": sum(row["decision"] == "migrate" for row in rows),
            "recompute": sum(row["decision"] == "recompute" for row in rows),
            "route_failed": sum(row["decision"] == "route_failed" for row in rows),
            "total_transfer_mib": round(
                sum(row.get("kv_transfer_mib", 0) for row in rows if row["decision"] == "migrate"),
                1,
            ),
            "reserved_blocks": sum(row.get("reserved_blocks", 0) for row in rows),
            "skipped_incompatible": len([
                skipped for skipped in self.skipped
                if skipped.get("checks", {}).get("layout") is False
                or skipped.get("checks", {}).get("tp") is False
            ]),
            "tenant_cache_blocked": any(row.get("tenant_cache_blocked") for row in rows),
        }
        gates = {
            "metadata_compatibility_checked": summary["skipped_incompatible"] > 0,
            "decode_capacity_reserved": summary["reserved_blocks"] > 0,
            "routing_uses_cost_score": all(
                "score" in row for row in rows if row["decision"] == "migrate"
            ),
            "tenant_isolation_enforced": summary["tenant_cache_blocked"],
            "recompute_path_visible": summary["recompute"] == 1,
            "failure_plan_visible": all(
                "failure_plan" in row for row in rows if row["decision"] == "migrate"
            ),
            "metrics_ready": summary["total_transfer_mib"] > 0
            and summary["requests"] == 3,
        }
        gates["kv_transfer_routing_gate"] = all(gates.values())
        return rows, summary, gates


workers = [
    DecodeWorkerCandidate("decode_fast", "rack_a", "v1", "tok1", "fp16", "paged_v1", 4, 220, 18, 1, 12.0),
    DecodeWorkerCandidate("decode_cache", "rack_b", "v1", "tok1", "fp16", "paged_v1", 4, 180, 5, 0, 11.0, {"doc_a": "tenant_a"}),
    DecodeWorkerCandidate("decode_bad_layout", "rack_a", "v1", "tok1", "fp16", "paged_v2", 4, 500, 1, 0, 9.0, {"doc_a": "tenant_a"}),
]
requests = [
    KVRequest("long_rag", "tenant_a", "v1", "tok1", 8192, 256, "fp16", "paged_v1", 4, "rack_a", "doc_a"),
    KVRequest("short_chat", "tenant_a", "v1", "tok1", 256, 128, "fp16", "paged_v1", 4, "rack_a", "none"),
    KVRequest("cross_tenant", "tenant_b", "v1", "tok1", 4096, 256, "fp16", "paged_v1", 4, "rack_a", "doc_a"),
]

rows, summary, gates = ToyKVTransferRouter().audit(requests, workers)
print("kv_transfer_rows=", rows)
print("kv_transfer_summary=", summary)
print("kv_transfer_gates=", gates)
```

一次运行的核心输出类似：

```text
kv_transfer_summary= {'requests': 3, 'migrate': 2, 'recompute': 1, 'route_failed': 0, 'total_transfer_mib': 192.0, 'reserved_blocks': 100, 'skipped_incompatible': 2, 'tenant_cache_blocked': True}
kv_transfer_gates= {'metadata_compatibility_checked': True, 'decode_capacity_reserved': True, 'routing_uses_cost_score': True, 'tenant_isolation_enforced': True, 'recompute_path_visible': True, 'failure_plan_visible': True, 'metrics_ready': True, 'kv_transfer_routing_gate': True}
```

这个 demo 展示了几个关键点：

1. `metadata_compatibility_checked=True`：layout / TP 不兼容的 worker 被跳过，避免 silent wrong output。
2. `decode_capacity_reserved=True`：迁移路径先 reserve blocks，再开始 transfer，不把 KV 传到未知位置。
3. `routing_uses_cost_score=True`：路由不是只看空闲，而是综合 KV capacity、running sequences、pending transfers、TPOT、网络和 prefix reuse。
4. `tenant_isolation_enforced=True`：`cross_tenant` 看到了 `doc_a` 的 prefix cache，但因为属于其他 tenant，不能复用。
5. `recompute_path_visible=True`：短 prompt 直接 recompute，比迁移更简单。
6. `failure_plan_visible=True`：每个迁移请求都带有 transfer 失败后的清理 / 重试计划。

所以本章最终门禁是 `kv_transfer_routing_gate`：只有 metadata、capacity、route score、tenant isolation、recompute fallback、failure plan 和 metrics 都能闭环，KV transfer 才不是“复制 bytes”，而是可治理的系统能力。

## 38.39 小练习

1. 估算一个 32 层、GQA、FP16 模型在 8000 token prompt 下的 KV cache 大小。
2. 画出 push KV transfer 的状态机。
3. 画出 pull KV transfer 的状态机。
4. 解释为什么 paged KV cache 会让 transfer metadata 更复杂。
5. 写出 KV transfer metadata 至少应该包含的 12 个字段。
6. 设计一个 decode worker routing score，至少考虑 5 个因素。
7. 说明 prefix-aware routing 为什么可能导致热点 worker。
8. 讨论什么时候 recompute 比 migrate 更合理。
9. 解释 decode 中途失败为什么比 prefill 失败更难恢复。
10. 设计 8 个观测 KV transfer 的关键指标。

## 38.40 本章总结

PD 分离真正困难的地方，是让 Decode worker 正确、高效地使用 Prefill worker 生成的 KV Cache。

KV Cache 不是普通数据。它体积大、在 GPU memory 中、layout 复杂，并且和模型版本、dtype、paged block、TP rank mapping、attention kernel 强相关。

KV 迁移常见模式包括 push、pull 和共享。Push/Pull 描述的是控制语义，底层可能由 RDMA、GPU IPC、NIXL、Mooncake 或其他 backend 实现。

Decode worker 通常要先预留 KV blocks，确认容量、确定目标地址、建立 block mapping，然后才能开始 transfer。

Router 需要做 KV-aware routing 和 prefix-aware routing，不仅看 worker 负载，还要看 KV capacity、pending transfers、网络拓扑、prefix locality 和 TPOT。

故障处理的核心是资源清理和语义恢复。Prefill 或 transfer 失败通常可以重试；decode 中途失败因为已经对外输出 token，恢复更困难。

下一章会继续讨论 Chunked Prefill 与 Disaggregated Prefill，看看长 prompt prefill 如何进一步切分和调度。
