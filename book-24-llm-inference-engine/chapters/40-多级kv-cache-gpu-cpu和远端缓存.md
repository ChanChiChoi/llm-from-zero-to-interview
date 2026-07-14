# 第 40 章 多级 KV Cache：GPU、CPU 和远端缓存

前面几章已经把 KV Cache 从单个请求里的中间状态，逐步讲成了 serving 系统里的核心资源。

第 38 章讲了 KV Cache 迁移、共享和路由。第 39 章讲了长 prompt prefill 如何通过 chunked prefill 分块执行。本章继续往后走：如果 KV Cache 太大，或者希望跨 worker、跨节点复用，系统应该怎么管理它？

答案通常不是“全部放 GPU 显存”。

真实系统会把 KV Cache 做成多级缓存：

```text
GPU KV Cache
  -> CPU KV Cache
  -> local SSD / NVMe KV Cache
  -> remote KV Cache service
  -> recompute
```

一句话概括：

> 多级 KV Cache 是把 KV 从单 worker 的 GPU 显存对象，升级成跨内存层级、跨节点、可迁移、可淘汰、可预取的系统级缓存资源。

## 40.0 本讲资料边界与第二轮精修口径

本讲第二轮精修前，先按 `WRITING_PLAN.md` 对公开资料做校准：参考 LMCache 文档对 KV cache 复用、offload、CPU / local storage / remote storage connector 和 vLLM connector 的公开口径；参考 SGLang HiCache 文档对 L1 GPU KV cache、L2 host cache、L3 distributed storage、cache controller、eviction 和 writing policy 的说明；参考 Mooncake 论文对 KVCache-centric disaggregated architecture、KV cache 管理和远端 KV 传输瓶颈的系统动机说明；并参考 vLLM prefix caching / PagedAttention 资料对 full block、block hash、block table、reference count 和 KV block metadata 的稳定抽象。

本讲只讲多级 KV Cache 的通用系统问题：GPU / CPU / SSD / remote / recompute 的层级画像、residency-aware scheduling、promote / demote / prefetch / eviction、成本模型、正确性 key、多租户隔离、partial block 风险和观测指标。不把某个框架版本的真实 connector API、存储 backend 名称、cache controller 字段、默认阈值、网络拓扑参数或 benchmark 数字写成通用标准。

本讲新增 demo 是教学版多级 KV Cache 审计器：用 0 依赖 Python 模拟 GPU hot block 保护、CPU promote、remote fetch、SSD 不划算时 recompute、GPU 空间不足时 demote、跨租户命中阻断、residency metrics 和最终 gate，帮助把“多级缓存不是只看 hit rate”落到可运行证据。

## 40.1 本章目标

读完本章，你应该能讲清：

1. 为什么只靠 GPU KV Cache 不够。
2. GPU、CPU、SSD、远端 KV Cache 分别适合存什么。
3. KV offload、swap、remote KV store 和 recompute 的区别。
4. 多级 KV Cache 的 promote、demote、prefetch 和 eviction 策略。
5. 多级 KV Cache 如何影响 TTFT、TPOT、吞吐和显存利用率。
6. 多级 KV Cache 和 prefix cache、PD 分离、chunked prefill 的关系。
7. 实现多级 KV Cache 时需要哪些 metadata。
8. 面试中如何设计一个多级 KV Cache 系统。

## 40.2 为什么只靠 GPU KV Cache 不够

KV Cache 的特点是：

1. 体积大。
2. 增长快。
3. 与序列长度线性相关。
4. decode 每步都要访问历史 KV。
5. 活跃请求必须低延迟访问。
6. 非活跃 prefix 又可能有复用价值。

以一个大模型为例，单 token 的 KV 可能是几十 KB 到几百 KB。prompt 长度一旦达到几万 tokens，单个请求的 KV Cache 就可能达到 GB 级。

如果所有 KV 都常驻 GPU，会遇到几个问题：

1. GPU 显存被 KV 占满，batch size 上不去。
2. 长上下文请求挤压短请求。
3. prefix cache 命中后也未必有足够显存保留所有 prefix。
4. PD 分离中，prefill 生成的 KV 不一定马上被 decode 使用。
5. 多轮对话或 agent 任务中，请求可能短时间 idle，但之后又继续。
6. 跨 worker 复用 prefix 时，KV 不能只存在本地 GPU。

所以系统需要回答一个问题：

```text
哪些 KV 必须在 GPU？
哪些 KV 可以暂时放 CPU？
哪些 KV 值得放到远端供其他 worker 复用？
哪些 KV 干脆丢掉，后面需要时重算？
```

这就是多级 KV Cache 的核心。

## 40.3 KV Cache 的访问冷热

不是所有 KV 都一样重要。

对一个正在 decode 的请求来说，历史 KV 每一步都要参与 attention。

例如：

```text
step t:
  query = current token hidden state
  keys/values = all previous tokens' KV
```

这部分 KV 是热数据，必须尽量放在 GPU。

但还有很多 KV 并不总是热的：

1. 刚 prefill 完但还没被 decode 接管的 KV。
2. 多轮对话中暂时 idle 的 session KV。
3. prefix cache 中命中率高但当前没有请求使用的公共前缀。
4. RAG 模板、system prompt、工具描述这类重复 prefix。
5. 长请求中未来可能被恢复的 KV。
6. 被 preempt 的请求 KV。

这些 KV 不一定要一直占用 GPU。

多级缓存的基本思想是：

```text
hot KV    -> GPU
warm KV   -> CPU
cold KV   -> SSD / remote store
too cold  -> evict and recompute later
```

## 40.4 四个层级的基本画像

可以先用一张表建立直觉。

| 层级 | 延迟 | 容量 | 带宽 | 典型用途 |
| --- | --- | --- | --- | --- |
| GPU HBM | 最低 | 最小 | 最高 | 活跃 decode、当前 prefill |
| CPU DRAM | 中等 | 较大 | 受 PCIe/NVLink 限制 | idle session、暂存 prefix、swap |
| Local SSD/NVMe | 高 | 更大 | 受本地 IO 限制 | 低频长 prefix、冷 KV |
| Remote KV Store | 取决于网络 | 可扩展 | 受网络和服务端限制 | 跨 worker 共享、PD 分离、全局 prefix cache |

这张表背后的核心取舍是：

```text
越靠近 GPU，访问越快，但容量越贵。
越远离 GPU，容量越大，但访问越慢，系统复杂度越高。
```

多级 KV Cache 不是越多层越好。每加一层，都要付出 metadata、调度、故障恢复和性能抖动的代价。

## 40.5 GPU KV Cache

GPU KV Cache 是最核心的一层。

它存放：

1. 正在 decode 的请求 KV。
2. 本轮 prefill 正在写入的 KV。
3. 马上要被 decode 使用的 prefix KV。
4. 高复用、高命中、低延迟要求的 prefix blocks。

GPU KV 的特点是：

1. 访问延迟最低。
2. attention kernel 可以直接读。
3. 不需要 host-device 传输。
4. 容量最紧张。
5. 碎片和 block 管理最敏感。

因此 GPU KV Cache 的管理目标通常是：

```text
优先保证活跃 decode 的连续推进，避免因为显存不足导致 TPOT 抖动。
```

如果为了缓存更多历史 prefix，把活跃 decode 的 KV 挤出去，通常得不偿失。

## 40.6 CPU KV Cache

CPU KV Cache 是最常见的第二层。

它可以存放：

1. 被 preempt 请求的 KV。
2. 暂时 idle 的 session KV。
3. 短期内可能恢复的多轮对话 KV。
4. 不适合常驻 GPU 但重算成本较高的长 prefix。
5. PD 分离中等待 transfer 或等待 decode 接管的 KV。

CPU KV Cache 的优势是容量大，成本低。

它的问题是传输。

```text
GPU <-> CPU
  via PCIe or NVLink
```

如果 decode 每一步都要从 CPU 拉历史 KV，性能会非常差。因为 decode 是高频小步执行，TPOT 对延迟非常敏感。

所以 CPU KV 更适合做：

1. 请求暂停期间的保存层。
2. 恢复前的一次性 promote 来源。
3. 显存压力下的 swap 目标。
4. GPU prefix cache 的后备层。

它不适合作为每个 decode step 的在线 KV 读取路径。

## 40.7 Local SSD / NVMe KV Cache

SSD 或 NVMe 层更冷。

它适合：

1. 很长但低频复用的 prefix。
2. 大量 session 的冷状态。
3. 离线预热生成的公共 prompt KV。
4. 成本敏感场景下的长上下文缓存。

但 SSD 的问题也明显：

1. 延迟比 CPU 高得多。
2. IO 调度会带来 tail latency。
3. KV block 数量多时，小 IO 放大严重。
4. 序列化、压缩、校验和 metadata 管理复杂。
5. 随机读可能拖垮恢复路径。

因此 SSD KV Cache 更适合“恢复前批量加载”，而不是 decode 过程中频繁访问。

例如：

```text
request resumes
  -> load KV blocks from SSD to CPU
  -> promote needed blocks to GPU
  -> enter decode
```

而不是：

```text
each decode step
  -> read old KV from SSD
```

后者通常不可接受。

## 40.8 Remote KV Cache

Remote KV Cache 是跨节点的 KV 存储或缓存服务。

它可能以几种形态出现：

1. 独立 KV cache service。
2. 分布式内存缓存。
3. 对象存储或本地盘加索引服务。
4. prefill worker 暴露的 KV serving 接口。
5. decode worker 之间的 peer-to-peer KV sharing。

Remote KV Cache 的价值在于：

1. 跨 worker 复用 prefix。
2. 支持 PD 分离中的 KV 交接。
3. 避免每个 worker 重复 prefill 相同 system prompt。
4. 提升全局缓存容量。
5. 支持 session 迁移和故障恢复。

但它引入了网络瓶颈。

远端 KV 读取路径大致是：

```text
decode worker
  -> query metadata
  -> locate remote KV blocks
  -> fetch over network
  -> place into local GPU/CPU blocks
  -> update local block table
```

这比本地 GPU 命中复杂得多。

如果远端拉取发生在用户等待首 token 的关键路径上，TTFT 可能明显变差。如果发生在 decode 中间，TPOT 会抖动。

## 40.9 Recompute 也是一个层级

很多人讲多级缓存时只讲 GPU、CPU、SSD、remote store，但在推理系统里，recompute 也应该被看成最后一层。

如果一个 KV block 被淘汰了，系统并不一定彻底失败。

只要还保留原始 tokens、position、模型版本和必要上下文，就可以重新 prefill 得到 KV。

所以层级可以写成：

```text
GPU -> CPU -> SSD -> Remote -> Recompute
```

Recompute 的优势是：

1. 不占缓存容量。
2. 不需要存储旧 KV。
3. 避免跨节点传输大对象。

Recompute 的代价是：

1. 消耗 GPU compute。
2. 增加 TTFT 或恢复延迟。
3. 可能打断当前 decode batch。
4. 对长 prompt 很贵。

因此 eviction 策略本质上是在比较：

```text
保留 KV 的存储成本
vs
未来命中时传输 KV 的成本
vs
未来 miss 时重新计算的成本
```

## 40.10 Offload、Swap、Remote Cache 的区别

这几个词经常混用，但面试时最好区分清楚。

| 概念 | 核心含义 | 典型方向 | 主要目的 |
| --- | --- | --- | --- |
| Offload | 把 GPU 上的数据卸到较慢层级 | GPU -> CPU/SSD | 降低 GPU 显存压力 |
| Swap | 在显存不足时换出/换入运行状态 | GPU <-> CPU | 保护系统不 OOM |
| Remote Cache | 把 KV 放到远端服务或节点 | local <-> remote | 跨 worker 复用或迁移 |
| Recompute | 丢弃 KV，之后重算 | cache -> compute | 用算力换存储 |

简单说：

```text
offload 关注放到哪里；
swap 关注运行时换入换出；
remote cache 关注跨节点共享；
recompute 关注不存，后面重算。
```

## 40.11 多级 KV Cache 的基本状态机

一个 KV block 可以处于多个状态。

例如：

```text
GPU_RESIDENT
CPU_RESIDENT
SSD_RESIDENT
REMOTE_RESIDENT
IN_TRANSFER
EVICTED
RECOMPUTE_REQUIRED
```

更细一点，还需要区分：

1. 是否被活跃请求引用。
2. 是否属于完整 block。
3. 是否属于 partial block。
4. 是否已经可被 prefix cache 命中。
5. 是否正在被写入。
6. 是否正在被 transfer。
7. 是否已经校验完成。
8. 是否可以被淘汰。

一个典型状态流转是：

```text
prefill writes KV to GPU
  -> request finishes or becomes idle
  -> demote GPU KV to CPU
  -> memory pressure increases
  -> demote cold CPU KV to SSD or remote
  -> request resumes
  -> prefetch SSD/remote KV to CPU
  -> promote CPU KV to GPU
  -> decode continues
```

注意：状态机必须和 scheduler、block manager、prefix cache 一起工作。否则 scheduler 以为 KV 在 GPU，实际已经被换出，就会在 attention kernel 读错地址。

## 40.12 Promote 和 Demote

Promote 是把 KV 从慢层级搬到快层级。

例如：

```text
CPU -> GPU
SSD -> CPU
Remote -> CPU/GPU
```

Demote 是把 KV 从快层级搬到慢层级。

例如：

```text
GPU -> CPU
CPU -> SSD
GPU -> remote
```

Promote 通常发生在：

1. 请求即将进入 decode。
2. prefix cache 命中，需要将命中 blocks 放回 GPU。
3. scheduler 预测某个 session 很快恢复。
4. PD 分离中 decode worker 即将接管 prefill KV。

Demote 通常发生在：

1. 请求完成但 prefix 有复用价值。
2. session 进入 idle。
3. GPU block 不够。
4. 系统希望保护活跃 decode。
5. 某些长 prefix 短期内不会访问。

Promote 和 demote 的难点不是拷贝本身，而是何时拷贝、拷贝多少、是否阻塞用户请求。

## 40.13 Prefetch

Prefetch 是提前把 KV 搬到更快层级。

例如：

```text
request waiting in queue
  -> scheduler predicts it will run soon
  -> prefetch KV from CPU to GPU
  -> when scheduled, decode can start immediately
```

Prefetch 的收益是降低关键路径延迟。

Prefetch 的风险是抢占资源。

如果预测错了，就会出现：

1. GPU 显存被无用 KV 占用。
2. PCIe 或网络带宽被无用 transfer 占用。
3. 活跃请求受到干扰。
4. 刚 prefetch 的 KV 又被淘汰。

一个简单策略是只 prefetch 高置信度请求：

```text
if request is near head of queue
   and needed_kv_bytes <= available_gpu_budget
   and estimated_start_time < threshold:
       prefetch_to_gpu(request.kv_blocks)
```

更复杂的系统会把 prefetch 当成调度问题：

```text
prefetch_priority
  = resume_probability
  * recompute_cost
  / transfer_cost
```

## 40.14 Eviction

Eviction 是决定丢掉哪些 KV。

在普通缓存里，LRU 是常见策略。但 KV Cache 不能只看最近访问时间。

原因是：

1. 不同 KV block 大小和重算成本不同。
2. 长 prefix 的后续 block 依赖前面的 prefix 语义。
3. 有些 prefix 虽然最近没用，但未来复用概率很高。
4. 有些 KV 属于活跃 decode，不能淘汰。
5. 有些 partial block 不适合单独缓存。
6. 多租户场景要考虑配额和隔离。

一个更合理的打分可以是：

```text
eviction_score
  = reuse_probability
  * recompute_cost
  * latency_sensitivity
  / memory_bytes
```

分数高表示更值得保留，分数低表示更适合淘汰。

也可以反过来定义 victim score：

```text
victim_score
  = memory_bytes
  / (reuse_probability * recompute_cost + epsilon)
```

实际系统不会只靠一个公式，而是会加硬规则：

1. 活跃 decode KV 不淘汰。
2. 正在 transfer 的 KV 不淘汰。
3. 被多个请求引用的 prefix block 优先保留。
4. 超过租户配额的 KV 优先淘汰。
5. 校验失败或版本不匹配的 KV 直接丢弃。

## 40.15 多级 KV Cache 和 Prefix Cache

Prefix cache 关注的是“相同前缀能不能复用”。

多级 KV Cache 关注的是“可复用 KV 放在哪里”。

两者通常要结合。

例如一个 system prompt 被大量请求复用：

```text
system prompt tokens -> KV blocks
```

这些 blocks 可能同时有多个位置：

```text
remote store: full copy for global reuse
CPU cache: warm copy for local worker
GPU cache: hot copy for active batch
```

prefix cache 的索引可能告诉你：

```text
hash H hits blocks [b1, b2, b3]
```

但 block manager 还要知道：

```text
b1 is on GPU
b2 is on CPU
b3 is remote only
```

如果只知道命中，不知道位置，scheduler 仍然无法估算成本。

所以 prefix cache 命中应该返回的不只是 hit length，还应该包含：

1. block ids。
2. residency level。
3. transfer cost estimate。
4. GPU promote requirement。
5. 是否需要 recompute。

## 40.16 多级 KV Cache 和 PD 分离

PD 分离中，Prefill worker 生成 KV，Decode worker 使用 KV。

这天然需要跨组件管理 KV。

最简单的路径是：

```text
prefill worker GPU
  -> transfer KV
  -> decode worker GPU
```

但多级缓存会让路径变多：

```text
prefill GPU -> prefill CPU -> remote KV -> decode CPU -> decode GPU
```

或者：

```text
prefill GPU -> remote KV
decode worker cache hit remote KV
decode worker promotes to GPU
```

这样做的好处是：

1. prefill worker 不必等待 decode worker 立即接收。
2. decode worker 可以异步拉取 KV。
3. 多个 decode worker 可以复用同一份远端 prefix。
4. session 可以跨 worker 迁移。

代价是：

1. TTFT 中多了一段 remote fetch。
2. metadata 一致性更复杂。
3. 失败清理更复杂。
4. 网络带宽可能成为瓶颈。
5. KV 生命周期不再绑定单个 worker。

PD 分离下的关键原则是：

```text
Decode worker 在开始 decode 前，必须确认需要的 KV 已经在本地可访问层级，最好已经在 GPU。
```

否则用户会在 streaming 过程中看到明显卡顿。

## 40.17 多级 KV Cache 和 Chunked Prefill

Chunked Prefill 会分批生成 KV。

这让多级缓存有了更细粒度的操作空间。

例如：

```text
chunk 0 generated -> store to CPU or remote
chunk 1 generated -> store to CPU or remote
chunk 2 generated -> keep on GPU
```

好处是：

1. prefill 和 KV transfer 可以流水化。
2. 长 prompt 不必等全部完成后才开始写入远端。
3. 部分 prefix 可以提前进入缓存索引。
4. GPU 上只保留短期需要的 chunk。

但要小心：普通 causal LM 在生成首 token 前仍需要完整 prompt 的 KV。

所以 chunked prefill 与多级缓存结合时，常见目标不是“没 prefill 完就开始 decode”，而是：

1. 降低 prefill 单次调度粒度。
2. 分摊 KV transfer。
3. 降低 GPU 峰值占用。
4. 提前把冷 chunk demote 到 CPU 或 remote。

例如长 prompt：

```text
prompt = 32000 tokens
chunk size = 4096
```

系统可以：

```text
chunk 0 prefill -> KV to CPU
chunk 1 prefill -> KV to CPU
...
last chunk prefill -> all required KV promote to GPU for decode
```

这在实现上很复杂，因为 decode 开始前必须确保完整 KV 可被 attention kernel 正确访问。

## 40.18 调度器需要知道什么

多级 KV Cache 不能只由 cache manager 自己决定。scheduler 必须感知 KV 的位置和移动成本。

否则它会做出错误决策。

例如 scheduler 看到一个请求 prompt cache 命中 16000 tokens，以为很便宜：

```text
hit length = 16000
uncached length = 128
```

但如果命中的 KV 全在远端，实际关键路径可能是：

```text
fetch 16000 tokens KV from remote
  -> copy to GPU
  -> run suffix prefill
  -> decode
```

这未必比重新 prefill 便宜。

调度器至少需要这些信息：

1. 命中 KV 当前在哪一层。
2. promote 到 GPU 需要多少 bytes。
3. transfer 预计耗时。
4. GPU 是否有足够 block。
5. CPU/SSD/remote 当前队列是否拥塞。
6. 如果选择 recompute，需要多少 token compute。
7. 请求的 TTFT 和 TPOT 敏感度。
8. 当前 decode batch 是否能承受一次大 promote。

因此，多级 KV Cache 会把 scheduling 从“算 token budget”升级为：

```text
同时算 compute budget、memory budget、transfer budget 和 latency budget。
```

## 40.19 一个成本模型

可以用一个简单成本模型帮助决策。

对某段 KV，有几种选择：

1. GPU hit。
2. CPU promote。
3. Remote fetch。
4. Recompute。

分别估算：

```text
cost_gpu_hit = 0

cost_cpu_promote
  = kv_bytes / pcie_bandwidth + copy_overhead

cost_remote_fetch
  = network_rtt + kv_bytes / network_bandwidth + deserialize_overhead

cost_recompute
  = prefill_tokens / prefill_throughput + scheduling_delay
```

然后比较：

```text
choose min(cost_cpu_promote, cost_remote_fetch, cost_recompute)
```

但真实系统还要加限制：

1. GPU block 是否够。
2. 当前 PCIe 是否拥塞。
3. 当前网络是否拥塞。
4. prefill worker 是否忙。
5. 用户请求是否快超时。
6. 这个 KV 是否允许跨租户复用。

所以面试里不要把成本模型说成万能公式。它只是帮助解释系统取舍。

## 40.20 正确性：KV 不是普通字节缓存

KV Cache 看起来像一段 tensor，但它不是普通 byte cache。

复用一段 KV 必须满足严格条件：

1. 模型权重版本一致。
2. tokenizer 版本一致。
3. token ids 一致。
4. position ids 一致。
5. RoPE scaling、sliding window、attention mask 一致。
6. dtype 和量化格式一致。
7. tensor parallel / pipeline parallel 切分方式一致。
8. adapter、LoRA、prompt adapter 一致。
9. 多模态输入的 image/audio embedding 一致。
10. 租户权限允许复用。

如果这些条件不满足，KV 命中就是错误命中。

错误命中很危险，因为它不一定报错，而是生成质量异常、事实错乱或跨用户信息泄露。

所以多级 KV Cache 的 key 不能只是 prompt hash。

它至少应该包含：

```text
cache_key = hash(
  model_id,
  model_revision,
  tokenizer_revision,
  token_ids,
  position_range,
  attention_config,
  kv_dtype,
  parallel_config,
  adapter_id,
  tenant_scope,
  multimodal_hashes
)
```

## 40.21 Metadata 设计

一个 KV block 的 metadata 可能包括：

```text
KVBlockMeta:
  block_id
  cache_key
  token_start
  token_end
  num_tokens
  layer_range
  kv_shape
  dtype
  device_level
  gpu_block_id
  cpu_ptr
  ssd_offset
  remote_uri
  ref_count
  last_access_time
  reuse_count
  recompute_cost
  transfer_cost_estimate
  tenant_id
  model_revision
  checksum
  state
```

这些字段不是都必须一开始实现，但方向是清楚的：

1. 要知道这段 KV 是谁的。
2. 要知道它对应哪些 tokens 和 positions。
3. 要知道它现在在哪里。
4. 要知道谁正在引用它。
5. 要知道它能不能被复用、转移、淘汰或重算。

KV metadata 的一致性比 KV tensor 本身更容易出 bug。

## 40.22 引用计数和生命周期

Prefix KV 可能被多个请求共享。

例如：

```text
R1, R2, R3 all use same system prompt blocks
```

这些 blocks 不能因为 R1 结束就释放。

需要引用计数：

```text
ref_count = number of active users
```

但 ref_count 也不等于是否保留。

当 ref_count 变成 0 后，KV 可以进入 cache 状态：

```text
ACTIVE
  -> CACHED
  -> DEMOTED
  -> EVICTED
```

生命周期大致是：

```text
ALLOCATED
  -> WRITING
  -> READY
  -> REFERENCED
  -> CACHED
  -> DEMOTING / PROMOTING
  -> EVICTED
```

每个状态都要定义清楚能否：

1. 被 attention 读取。
2. 被 prefix cache 命中。
3. 被 transfer。
4. 被 eviction。
5. 被另一个请求引用。

## 40.23 Partial Block 问题

Paged KV Cache 通常按 block 管理。

例如 block size = 16 tokens。

完整 block 比较容易缓存：

```text
tokens [0, 16)
tokens [16, 32)
tokens [32, 48)
```

partial block 更麻烦：

```text
tokens [48, 53)
```

问题包括：

1. partial block 可能还会继续写。
2. hash 边界不稳定。
3. 多个请求共享 partial block 容易出错。
4. offload 后恢复要处理未填满部分。
5. chunked prefill 可能在 block 中间结束。

很多系统会选择：

```text
only full blocks are eligible for prefix cache sharing
partial blocks stay request-private
```

这样会损失一点复用率，但能显著降低正确性风险。

## 40.24 多租户和安全隔离

远端 KV Cache 最大的风险之一是跨租户泄露。

KV 虽然不是原文 token，但它包含 prompt 信息，不能随意共享。

多租户场景下必须考虑：

1. tenant scope。
2. user scope。
3. org scope。
4. model scope。
5. adapter scope。
6. data retention policy。
7. cache encryption。
8. access control。
9. audit log。

一个保守策略是：

```text
默认只在同 tenant、同 model revision、同 adapter、同权限域内复用 KV。
```

公共 system prompt 可以单独标记为 global cache，但必须由平台明确声明，而不是自动跨租户共享。

## 40.25 网络瓶颈

Remote KV Cache 的瓶颈通常不是“能不能传”，而是“传输是否值得”。

如果 KV 很大，网络会成为 TTFT 的一部分。

例如：

```text
KV size = 2 GB
effective network bandwidth = 25 GB/s
transfer time ~= 80 ms
```

这还没算：

1. RPC 开销。
2. 排队延迟。
3. 序列化开销。
4. GPU staging copy。
5. metadata 查询。
6. tail latency。

如果重新 prefill 只要 60 ms，那么 remote fetch 反而不划算。

所以 remote KV Cache 需要回答：

```text
fetch KV 是否比 recompute 更便宜？
```

这取决于模型大小、prompt 长度、prefill 吞吐、网络带宽、GPU 空闲程度和当前队列状态。

## 40.26 压缩和量化

为了降低存储和传输成本，可以压缩或量化 KV。

例如：

1. FP16/BF16 KV 存储。
2. FP8 KV cache。
3. INT8 KV cache。
4. 按层压缩。
5. 冷层级压缩，promote 时解压。

这样可以降低：

1. GPU 显存占用。
2. CPU 内存占用。
3. SSD 空间。
4. 网络传输 bytes。

但代价是：

1. 可能影响模型质量。
2. 需要额外 encode/decode kernel。
3. 解压可能进入关键路径。
4. 不同层级 dtype 不一致，metadata 更复杂。
5. prefix cache key 必须包含 KV dtype 和压缩格式。

如果面试官问“能不能把远端 KV 压缩存”，回答可以，但要强调正确性和延迟权衡。

## 40.27 和 Attention Kernel 的关系

Attention kernel 通常希望 KV 已经在 GPU 上，并且布局符合预期。

例如 PagedAttention 需要 block table 指向 GPU blocks。

如果 KV 在 CPU 或远端，kernel 不能直接读。

所以多级 KV Cache 不能让 attention kernel 变成：

```text
if block on GPU: read GPU
if block on CPU: copy then read
if block remote: RPC then read
```

这会把 kernel 路径搞得非常复杂，也会严重拖慢 decode。

更合理的做法是：

```text
scheduler/cache manager ensures required KV is GPU-resident
then attention kernel runs on normal GPU block table
```

也就是说，多级缓存主要发生在调度和准备阶段，不应该污染高频 attention kernel 的主路径。

## 40.28 一个简化实现

可以把系统拆成几层：

```text
Scheduler
  -> KVCacheManager
       -> GPUBlockManager
       -> CPUBlockStore
       -> RemoteKVClient
       -> EvictionPolicy
       -> Prefetcher
```

伪代码如下：

```python
def prepare_request(req):
    hits = prefix_cache.lookup(req.cache_key)
    plan = kv_manager.plan(req, hits)

    if not plan.can_run:
        return WAIT

    for block in plan.blocks_to_promote:
        kv_manager.promote_to_gpu(block)

    for block in plan.blocks_to_recompute:
        scheduler.enqueue_prefill(block.token_range)

    if kv_manager.gpu_free_blocks() < plan.required_blocks:
        victims = eviction_policy.pick_victims(plan.required_blocks)
        kv_manager.demote_or_evict(victims)

    req.block_table = kv_manager.build_gpu_block_table(req)
    return READY
```

这个伪代码省略了大量细节，但表达了重点：

1. 先查 prefix。
2. 再看 KV 在哪里。
3. 再决定 promote、recompute 或等待。
4. 最后构建 GPU block table。

## 40.29 常见策略组合

一个务实的起步版本可以这样做：

1. GPU 只保留活跃 decode 和高频 full prefix blocks。
2. CPU 保存 idle session KV 和刚被 preempt 的请求 KV。
3. 远端只保存明确可跨 worker 复用的公共 prefix。
4. partial block 不进入全局 cache。
5. remote fetch 只在预计比 recompute 便宜时使用。
6. prefetch 只对队首请求启用。
7. eviction 先保护活跃 decode，再保护高复用 prefix。

不要一开始就做全功能多级缓存。

更好的路线是：

```text
GPU block manager
  -> CPU offload for preemption
  -> local prefix cache
  -> remote prefix cache
  -> cost-based prefetch and eviction
```

## 40.30 常见坑

坑一：把 CPU KV 当成 decode 在线读取层。

decode 每步都跨 PCIe 读历史 KV，TPOT 会非常差。CPU 更适合恢复前 promote，而不是每步在线读取。

坑二：只看 cache hit，不看 residency。

远端命中不等于低成本命中。scheduler 必须知道 KV 在 GPU、CPU 还是 remote。

坑三：错误复用 KV。

模型版本、tokenizer、position、adapter、tenant scope 任一不一致，都可能导致错误结果或信息泄露。

坑四：过度 prefetch。

prefetch 错了会浪费 GPU blocks、PCIe 带宽和网络带宽，还可能挤掉真正活跃的 KV。

坑五：淘汰 partial block。

partial block 边界不稳定，跨请求复用容易出错。初版系统应尽量只缓存 full blocks。

坑六：remote KV 进入 TPOT 关键路径。

如果 streaming 过程中突然从远端拉 KV，用户会看到明显卡顿。

坑七：metadata 与 tensor 生命周期不一致。

metadata 说 block 在 GPU，但实际已经释放，这是非常危险的 bug。

坑八：忽略失败清理。

transfer 中断、worker crash、remote 写入失败，都要能清理半成品 KV。

## 40.31 观测指标

多级 KV Cache 必须可观测。

关键指标包括：

1. GPU KV utilization。
2. CPU KV utilization。
3. SSD KV bytes。
4. remote KV bytes。
5. GPU KV hit rate。
6. CPU KV hit rate。
7. remote KV hit rate。
8. recompute fallback count。
9. promote latency。
10. demote latency。
11. prefetch success rate。
12. prefetch waste rate。
13. eviction count by reason。
14. transfer bandwidth。
15. transfer queue length。
16. KV metadata lookup latency。
17. TTFT with cache hit vs miss。
18. TPOT impact during promote/demote。
19. remote fetch p50/p95/p99。
20. cache correctness failure count。

不要只看命中率。

一个高命中率但远端 fetch 很慢的系统，用户体验可能更差。

更重要的是：

```text
cache hit 是否真的降低了端到端延迟和 GPU compute？
```

## 40.32 调参思路

常见参数包括：

1. `gpu_kv_cache_budget_bytes`。
2. `cpu_kv_cache_budget_bytes`。
3. `remote_cache_enabled`。
4. `max_promote_bytes_per_step`。
5. `max_demote_bytes_per_step`。
6. `prefetch_queue_depth`。
7. `remote_fetch_timeout_ms`。
8. `min_recompute_saving_threshold`。
9. `full_block_cache_only`。
10. `tenant_cache_quota_bytes`。

调参目标不是让所有层级都尽量满，而是让关键路径稳定。

优先级通常是：

```text
保护 TPOT 稳定性
  -> 降低 TTFT
  -> 提高吞吐
  -> 提高缓存复用率
  -> 降低重算成本
```

如果为了提高 cache hit rate 导致 TPOT 抖动，通常是错误优化。

## 40.33 面试官会怎么问

问题一：为什么需要多级 KV Cache？

回答要点：GPU 显存容量有限，而 KV Cache 随序列长度和并发增长很快。活跃 decode KV 需要低延迟留在 GPU，但 idle session、可复用 prefix、PD 分离中等待消费的 KV 可以放到 CPU、SSD 或远端。多级 KV Cache 用容量换延迟，用传输或重算换显存。

问题二：CPU offload 适合什么场景？

回答要点：适合被 preempt 的请求、暂时 idle 的 session、短期可能恢复且重算成本较高的 KV。不适合 decode 每步在线从 CPU 读取，因为 PCIe/NVLink 传输会严重影响 TPOT。

问题三：Remote KV Cache 和 prefix cache 是什么关系？

回答要点：prefix cache 解决“是否可以复用前缀 KV”，remote KV cache 解决“可复用 KV 放在哪里以及如何跨 worker 获取”。prefix cache 命中后，还要看 KV 是 GPU-resident、CPU-resident 还是 remote-only。

问题四：Remote fetch 一定比 recompute 好吗？

回答要点：不一定。要比较 KV bytes、网络带宽、RPC 延迟、队列拥塞、prefill 吞吐和 GPU 空闲程度。如果 remote fetch 比重新 prefill 更慢，或者会影响 TPOT，就不应该 fetch。

问题五：多级 KV Cache 的正确性风险是什么？

回答要点：KV 复用必须保证模型版本、tokenizer、token ids、position、attention 配置、dtype、parallel config、adapter、多模态输入和租户权限一致。错误命中可能不会报错，但会导致生成异常或信息泄露。

问题六：实现多级 KV Cache 最容易踩什么坑？

回答要点：只看命中率不看位置和传输成本；让远端 fetch 进入 decode 关键路径；partial block 复用错误；metadata 和实际 tensor 生命周期不一致；prefetch 过度；缺少失败清理和租户隔离。

## 40.34 标准回答模板

如果面试官问“设计一个多级 KV Cache”，可以这样回答：

```text
我会先把 KV 分成热、温、冷几类。活跃 decode 的 KV 必须留在 GPU，因为 attention kernel 每步都要读，不能让 CPU 或远端访问进入 TPOT 关键路径。暂时 idle 的 session、被 preempt 的请求、短期可能恢复的长 prefix 可以 demote 到 CPU。更冷、复用频率低但重算成本高的 prefix 可以放到 SSD 或远端。远端 KV 主要用于跨 worker 复用和 PD 分离中的 KV 交接。

系统上我会让 scheduler、KV cache manager 和 prefix cache 协同工作。prefix cache 命中时，不只返回 hit length，还要返回 blocks 当前在哪一层、promote 成本、是否需要 recompute。scheduler 根据 compute cost、transfer cost、GPU block budget 和用户 latency SLO 决定是 promote、fetch remote、recompute 还是等待。

正确性上，KV cache key 必须包含 model revision、tokenizer、token ids、position、attention config、dtype、parallel config、adapter 和 tenant scope。初版我会只允许 full block 进入共享 prefix cache，partial block 保持 request-private。

策略上，先保证活跃 decode 的 TPOT 稳定，再优化 TTFT 和命中率。remote fetch 和 prefetch 都不能无脑做，必须比较 transfer cost 和 recompute cost，并通过指标观察 promote latency、remote hit latency、recompute fallback、TPOT 抖动和 cache hit 是否真的降低端到端延迟。
```

## 40.35 多级 KV Cache 成本、驻留门禁和可运行 demo

先把一个 KV block 抽象成：

```math
B_j=(b_j,u_j,m_j,L_j,N_j,M_j,R_j,F_j)
```

其中 `b_j` 是 block id，`u_j` 是 tenant，`m_j` 是 model revision，`L_j` 是 residency level，`N_j` 是 token 数，`M_j` 是 KV 大小，`R_j` 是 ref count，`F_j` 是 reuse / frequency 信号。

GPU 容量约束可以写成：

```math
\sum_{j:L_j=\mathrm{GPU}}M_j\le M_{\mathrm{gpu}}
```

CPU promote 成本：

```math
C_{\mathrm{cpu},j}=\frac{M_j}{B_{\mathrm{cpu}}}+\delta_{\mathrm{cpu}}
```

Remote fetch 成本：

```math
C_{\mathrm{remote},j}=R_{\mathrm{net}}+\frac{M_j}{B_{\mathrm{net}}}+\delta_{\mathrm{remote}}
```

重算成本：

```math
C_{\mathrm{recompute},j}=\frac{N_j}{X_{\mathrm{prefill}}}+\delta_{\mathrm{sched}}
```

一个教学版决策可以写成：

```math
A_j=\min(C_{\mathrm{cpu},j},C_{\mathrm{remote},j},C_{\mathrm{recompute},j})
```

但这个 `min` 只能在正确性门通过后使用。tenant、model revision、token ids、position、dtype、parallel config、adapter 和 multimodal hash 任一不兼容，都不能复用。

最终门禁可以写成：

```math
G_{\mathrm{mlkv}}=G_{\mathrm{hot}}G_{\mathrm{cpu}}G_{\mathrm{remote}}G_{\mathrm{recompute}}G_{\mathrm{tenant}}G_{\mathrm{demote}}G_{\mathrm{metric}}
```

下面的 demo 覆盖三类请求：

1. `resume_rag`：需要一个 CPU-resident block、一个 remote block 和一个 missing tail。
2. `cross_tenant_probe`：看到了 remote block，但属于其他 tenant，必须阻断。
3. `cold_doc`：SSD 上有冷 block，但 fetch 比 recompute 更慢，选择 recompute。

```python
from dataclasses import dataclass


@dataclass
class KVBlock:
    block_id: str
    tenant: str
    model_revision: str
    level: str
    tokens: int
    ref_count: int
    reuse_count: int


@dataclass
class KVRequest:
    request_id: str
    tenant: str
    model_revision: str
    needed_blocks: list


class ToyMultiLevelKVCacheAuditor:
    def __init__(self, gpu_capacity_blocks=4, block_mib=16.0):
        self.gpu_capacity_blocks = gpu_capacity_blocks
        self.block_mib = block_mib
        self.cpu_mib_per_ms = 8.0
        self.remote_mib_per_ms = 8.0
        self.ssd_mib_per_ms = 2.0
        self.prefill_tokens_per_ms = 256.0
        self.blocks = {
            "active_a": KVBlock("active_a", "tenant_a", "v1", "GPU", 128, 1, 20),
            "active_b": KVBlock("active_b", "tenant_a", "v1", "GPU", 128, 1, 18),
            "prefix_hot": KVBlock("prefix_hot", "tenant_a", "v1", "GPU", 128, 0, 8),
            "idle_cpu": KVBlock("idle_cpu", "tenant_a", "v1", "CPU", 128, 0, 5),
            "doc_remote": KVBlock("doc_remote", "tenant_a", "v1", "REMOTE", 128, 0, 12),
            "shared_doc_b": KVBlock("shared_doc_b", "tenant_b", "v1", "REMOTE", 128, 0, 12),
            "cold_ssd": KVBlock("cold_ssd", "tenant_a", "v1", "SSD", 512, 0, 1),
        }
        self.demotions = []
        self.evictions = []

    def _gpu_used(self):
        return sum(block.level == "GPU" for block in self.blocks.values())

    def _ensure_gpu_space(self):
        if self._gpu_used() < self.gpu_capacity_blocks:
            return []
        candidates = [
            block for block in self.blocks.values()
            if block.level == "GPU" and block.ref_count == 0
        ]
        if not candidates:
            return []
        victim = min(candidates, key=lambda block: block.reuse_count)
        victim.level = "CPU"
        self.demotions.append(victim.block_id)
        return [victim.block_id]

    def _compatible(self, request, block):
        return request.tenant == block.tenant and request.model_revision == block.model_revision

    def _cpu_cost(self):
        return round(self.block_mib / self.cpu_mib_per_ms + 0.2, 3)

    def _remote_cost(self):
        return round(1.0 + self.block_mib / self.remote_mib_per_ms + 0.3, 3)

    def _ssd_cost(self):
        return round(1.0 + self.block_mib / self.ssd_mib_per_ms + 0.5, 3)

    def _recompute_cost(self, tokens):
        return round(tokens / self.prefill_tokens_per_ms + 0.5, 3)

    def handle_block(self, request, block_id):
        if block_id not in self.blocks:
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": "recompute_missing",
                "cost_ms": self._recompute_cost(512),
                "transfer_mib": 0.0,
            }
        block = self.blocks[block_id]
        if not self._compatible(request, block):
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": "blocked_incompatible_or_tenant",
                "cost_ms": 0.0,
                "transfer_mib": 0.0,
            }
        if block.level == "GPU":
            block.ref_count += 1
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": "gpu_hit",
                "cost_ms": 0.0,
                "transfer_mib": 0.0,
            }
        if block.level == "CPU":
            demoted = self._ensure_gpu_space()
            block.level = "GPU"
            block.ref_count += 1
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": "promote_cpu_to_gpu",
                "cost_ms": self._cpu_cost(),
                "transfer_mib": self.block_mib,
                "demoted": demoted,
            }
        if block.level == "REMOTE":
            remote_cost = self._remote_cost()
            recompute_cost = self._recompute_cost(block.tokens * 16)
            if remote_cost <= recompute_cost:
                demoted = self._ensure_gpu_space()
                block.level = "GPU"
                block.ref_count += 1
                return {
                    "request_id": request.request_id,
                    "block_id": block_id,
                    "action": "fetch_remote_to_gpu",
                    "cost_ms": remote_cost,
                    "transfer_mib": self.block_mib,
                    "demoted": demoted,
                }
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": "recompute_instead_of_remote",
                "cost_ms": recompute_cost,
                "transfer_mib": 0.0,
            }
        if block.level == "SSD":
            ssd_cost = self._ssd_cost()
            recompute_cost = self._recompute_cost(block.tokens)
            action = "fetch_ssd_to_gpu" if ssd_cost <= recompute_cost else "recompute_instead_of_ssd"
            return {
                "request_id": request.request_id,
                "block_id": block_id,
                "action": action,
                "cost_ms": min(ssd_cost, recompute_cost),
                "transfer_mib": self.block_mib if action == "fetch_ssd_to_gpu" else 0.0,
            }
        raise ValueError(block.level)

    def audit(self, requests):
        rows = []
        for request in requests:
            for block_id in request.needed_blocks:
                rows.append(self.handle_block(request, block_id))
        summary = {
            "requests": len(requests),
            "gpu_hits": sum(row["action"] == "gpu_hit" for row in rows),
            "cpu_promotes": sum(row["action"] == "promote_cpu_to_gpu" for row in rows),
            "remote_fetches": sum(row["action"] == "fetch_remote_to_gpu" for row in rows),
            "ssd_fetches": sum(row["action"] == "fetch_ssd_to_gpu" for row in rows),
            "recomputes": sum(row["action"].startswith("recompute") for row in rows),
            "tenant_blocks": sum(row["action"] == "blocked_incompatible_or_tenant" for row in rows),
            "demotions": len(self.demotions),
            "evictions": len(self.evictions),
            "gpu_used_blocks": self._gpu_used(),
            "cpu_used_blocks": sum(block.level == "CPU" for block in self.blocks.values()),
            "remote_used_blocks": sum(block.level == "REMOTE" for block in self.blocks.values()),
            "total_transfer_mib": round(sum(row["transfer_mib"] for row in rows), 1),
            "estimated_latency_ms": round(sum(row["cost_ms"] for row in rows), 3),
        }
        gates = {
            "active_gpu_blocks_protected": all(
                self.blocks[block_id].level == "GPU" for block_id in ["active_a", "active_b"]
            ),
            "cpu_promote_visible": summary["cpu_promotes"] == 1,
            "remote_fetch_visible": summary["remote_fetches"] == 1,
            "recompute_fallback_visible": summary["recomputes"] == 2,
            "tenant_isolation_enforced": summary["tenant_blocks"] == 1,
            "demotion_makes_gpu_space": summary["demotions"] == 1,
            "residency_metrics_ready": (
                summary["total_transfer_mib"] == 32.0
                and summary["gpu_used_blocks"] == 4
            ),
        }
        gates["multi_level_kv_gate"] = all(gates.values())
        return rows, summary, gates


requests = [
    KVRequest("resume_rag", "tenant_a", "v1", ["idle_cpu", "doc_remote", "missing_tail"]),
    KVRequest("cross_tenant_probe", "tenant_a", "v1", ["shared_doc_b"]),
    KVRequest("cold_doc", "tenant_a", "v1", ["cold_ssd"]),
]

rows, summary, gates = ToyMultiLevelKVCacheAuditor().audit(requests)
print("multi_level_kv_rows=", rows)
print("multi_level_kv_summary=", summary)
print("multi_level_kv_gates=", gates)
```

一次运行的核心输出类似：

```text
multi_level_kv_summary= {'requests': 3, 'gpu_hits': 0, 'cpu_promotes': 1, 'remote_fetches': 1, 'ssd_fetches': 0, 'recomputes': 2, 'tenant_blocks': 1, 'demotions': 1, 'evictions': 0, 'gpu_used_blocks': 4, 'cpu_used_blocks': 1, 'remote_used_blocks': 1, 'total_transfer_mib': 32.0, 'estimated_latency_ms': 10.5}
multi_level_kv_gates= {'active_gpu_blocks_protected': True, 'cpu_promote_visible': True, 'remote_fetch_visible': True, 'recompute_fallback_visible': True, 'tenant_isolation_enforced': True, 'demotion_makes_gpu_space': True, 'residency_metrics_ready': True, 'multi_level_kv_gate': True}
```

这个 demo 展示了几个关键点：

1. `active_gpu_blocks_protected=True`：有 ref count 的 active decode KV 没有被 demote。
2. `cpu_promote_visible=True`：CPU-resident KV 可以 promote 回 GPU，但要计 transfer 成本。
3. `remote_fetch_visible=True`：远端 KV 在比重算更便宜时进入 GPU。
4. `recompute_fallback_visible=True`：missing block 和 SSD 冷 block 都走 recompute，说明 cache hit 不等于一定 fetch。
5. `tenant_isolation_enforced=True`：其他 tenant 的 remote block 被阻断。
6. `demotion_makes_gpu_space=True`：GPU 满时只 demote ref count 为 0 的 block，为 remote fetch 腾空间。
7. `residency_metrics_ready=True`：summary 同时给出 transfer MiB、GPU / CPU / remote residency 和 latency。

所以本章最终门禁是 `multi_level_kv_gate`：只有 hot KV 保护、CPU promote、remote fetch、recompute fallback、tenant isolation、demotion 和 residency metrics 都能闭环，多级 KV Cache 才不是“多加几层存储”，而是可治理的运行时缓存系统。

## 40.36 小练习

1. 画出 GPU、CPU、SSD、Remote、Recompute 五层 KV Cache 的访问路径。
2. 解释为什么 CPU KV 不适合作为 decode 每步在线读取层。
3. 设计一个判断 remote fetch 还是 recompute 的成本模型。
4. 给一个 32000-token prompt，分析哪些 KV 应该留 GPU，哪些可以 demote。
5. 说明 prefix cache 命中为什么还要返回 residency level。
6. 设计 `KVBlockMeta` 的字段，并解释每个字段用途。
7. 分析 partial block 为什么不适合跨请求共享。
8. 设计一个多租户 KV Cache 的隔离策略。
9. 列出 10 个观测多级 KV Cache 的指标。
10. 说明多级 KV Cache 如何服务 PD 分离。

## 40.37 本章总结

多级 KV Cache 的核心是把 KV 从 GPU 显存中的临时对象，升级成跨层级、跨节点、可迁移、可淘汰、可预取的系统级资源。

GPU 层负责活跃 decode 和低延迟访问；CPU 层适合 idle session、preemption 和短期恢复；SSD 和远端层适合更冷、更大、跨 worker 复用的 KV；recompute 则是最后的兜底层。

多级缓存不是单纯提高命中率。真正重要的是在显存、计算、传输、延迟和正确性之间做权衡。

实现时，scheduler 必须知道 KV 的 residency 和移动成本；prefix cache 必须和 block manager 结合；attention kernel 的主路径应尽量只面对 GPU-resident KV；cache key 必须包含模型、token、position、adapter、并行配置和租户权限等正确性信息。

下一章会继续讨论跨节点 serving 和网络瓶颈。多级 KV Cache 一旦进入远端访问，就不可避免地受到网络拓扑、带宽、延迟、拥塞和通信模式的限制。
