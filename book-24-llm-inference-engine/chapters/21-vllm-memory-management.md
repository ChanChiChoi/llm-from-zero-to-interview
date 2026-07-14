# 第 21 章 vLLM memory management

上一章从请求生命周期角度讲了 vLLM-like engine 的调度流程：请求进入 waiting queue，scheduler 每轮选择 prefill 和 decode，block manager 分配 KV blocks，worker 执行模型，output processor 判断是否结束并释放资源。

本章换一个视角：memory management。推理框架的吞吐、并发、TTFT、TPOT 和 OOM 风险，最终都会落到“显存和缓存如何被管理”上。

一句话概括：

> vLLM memory management 的核心是把 GPU memory 中最动态、最容易浪费的 KV Cache 管理成可分配、可复用、可淘汰、可观测的 block pool，并让 scheduler 基于 memory budget 做 admission control、preemption 和调度取舍。

## 21.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考五类公开资料：

1. vLLM PagedAttention 论文和项目文档对 block-based KV cache、非连续 physical blocks、block table 和高吞吐 serving 的公开口径。
2. vLLM prefix caching 设计文档对 `KVCacheBlock`、block hash、parent hash、extra hashes、free queue、cached-but-free block、LRU eviction 和引用计数的说明。
3. vLLM optimization / tuning 文档对 KV cache capacity、preemption、recompute、`gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens` 和 chunked prefill 的调优边界说明。
4. vLLM metrics 文档对 GPU / CPU KV cache usage、prefix cache hit rate、preemption 和 request queue / running / waiting 指标的观测口径。
5. vLLM Hybrid KV Cache Manager 文档对 full attention、sliding window attention、state space layer 等混合 cache 类型、KV cache group、page size 和统一管理接口的说明。

本章只讲 vLLM-like memory management 的教学抽象，不绑定某个 vLLM 版本的真实源码类名、CUDA kernel、真实内存分配器、swap 实现、KV connector、分布式 KV transfer、生产参数全集或具体 GPU 型号。本章 demo 用纯 Python list / dict 模拟 block pool、free queue、prefix cache、ref count、eviction、preemption 和 metrics，不等同于真实 vLLM runtime。

参考资料：

1. vLLM paper / PagedAttention：<https://arxiv.org/abs/2309.06180>
2. vLLM prefix caching design：<https://docs.vllm.ai/en/latest/design/prefix_caching/>
3. vLLM optimization and tuning：<https://docs.vllm.ai/en/latest/configuration/optimization/>
4. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics/>
5. vLLM Hybrid KV Cache Manager：<https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/>

## 21.1 本章目标

读完本章，你应该能讲清：

1. LLM serving 中 GPU memory 主要被哪些部分占用。
2. 为什么 KV Cache 是 memory management 的核心难点。
3. vLLM 如何用 block pool、free queue、block table、ref count 管理 KV Cache。
4. Prefix caching、LRU eviction、preemption、recompute 和 offload 分别解决什么问题。
5. `gpu_memory_utilization`、`max_num_seqs`、`max_num_batched_tokens` 这类参数如何影响显存和性能。
6. 混合 attention 模型为什么需要 Hybrid KV Cache Manager。
7. 面试中如何系统回答“vLLM 如何管理显存”。

## 21.2 LLM serving 的显存都花在哪里

先把 GPU memory 拆开。

一个 LLM serving worker 的显存通常包括：

1. 模型权重。
2. KV Cache。
3. 临时激活和中间 buffer。
4. attention backend 或 CUDA graph 需要的 workspace。
5. logits、sampling、通信 buffer。
6. LoRA adapter 或多模型附加权重。
7. 多模态 encoder 相关缓存或输入 tensor。

其中模型权重相对静态，服务启动时加载后基本不随请求数量变化。

KV Cache 则完全不同。它随请求数、prompt 长度、生成长度动态增长。

一个请求越长，占用 KV Cache 越多；并发越高，同时驻留的 KV Cache 越多；输出越长，decode 过程中还会继续追加 KV Cache。

所以线上 OOM 往往不是模型刚加载时立刻 OOM，而是在高并发、长 prompt、长输出或 prefix cache 不当时逐步逼近显存上限。

可以把单个 worker 的 GPU memory 粗略拆成：

$$
M_{\mathrm{gpu}}=M_{\mathrm{weight}}+M_{\mathrm{kv}}+M_{\mathrm{work}}+M_{\mathrm{graph}}+M_{\mathrm{sample}}+M_{\mathrm{adapter}}+M_{\mathrm{reserved}}
$$

其中 `M_{\mathrm{kv}}` 是最动态的一项。对普通 decoder-only full attention 模型，KV Cache 可以粗略估算为：

$$
M_{\mathrm{kv}}=2L H_{\mathrm{kv}}D_h b\sum_i T_i
$$

这里 `L` 是层数，`H_{\mathrm{kv}}` 是 KV heads，`D_h` 是 head dim，`b` 是单个元素字节数，`\sum_i T_i` 是所有 active 请求持有的 KV tokens 总数。

## 21.3 KV Cache 为什么是核心难点

KV Cache 有四个特点。

第一，大。

每个 token 在每层 attention 中都要保存 key 和 value。层数越多、hidden size 越大、上下文越长，KV Cache 越大。

第二，动态增长。

Prefill 时一次写入 prompt 的 KV，decode 时每生成一个 token 又追加一份 KV。

第三，生命周期不一致。

短请求很快释放，长请求长期占用；有的请求被 abort，有的请求触发 stop string，有的请求会生成到 max tokens。

第四，存在复用机会。

多个请求可能共享相同 system prompt、文档前缀、多轮对话历史或工具说明，这些 prefix 的 KV Cache 可以复用。

如果用一整块连续张量管理，每个请求很难动态增长、释放、共享和淘汰。PagedAttention 和 block manager 就是为了解决这个问题。

## 21.4 从连续 cache 到 block pool

传统思路可能是给每个请求预留一段连续 KV Cache。

问题是：

1. 请求长度不可提前准确知道。
2. 预留太多会浪费。
3. 预留太少需要搬迁。
4. 请求完成后会留下碎片。
5. prefix sharing 很难做。

vLLM 的核心思路是把 KV Cache 切成固定大小的 blocks。

简化结构：

```text
GPU KV memory
  block 0
  block 1
  block 2
  ...
  block N-1
```

每个请求维护自己的 block table：

```text
request A logical blocks: [0, 1, 2]
request A physical blocks: [17, 5, 88]
```

这样，请求逻辑上连续，但物理显存不要求连续。

这就是 memory management 的第一层抽象：

```text
连续 token 序列 -> logical blocks -> physical KV blocks
```

## 21.5 Block pool 的基本数据结构

一个最小 block pool 包括：

```python
class KVCacheBlock:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None
        self.prev_free_block = None
        self.next_free_block = None


class KVCacheManager:
    def __init__(self, num_blocks):
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_queue = FreeBlockQueue(self.blocks)
        self.request_blocks = {}
        self.cached_blocks = {}
```

这些字段分别解决：

1. `blocks`：所有 physical blocks 的对象池。
2. `free_queue`：当前可分配或可淘汰的 blocks。
3. `request_blocks`：每个请求当前持有哪些 blocks。
4. `cached_blocks`：prefix cache 中 hash 到 block 的映射。
5. `ref_cnt`：某个 block 当前被多少请求引用。
6. `block_hash`：一个 full block 是否能作为 prefix cache 被复用。

vLLM prefix caching 文档中特别强调，block 对象通常在初始化时一次性创建，避免运行时频繁创建 Python 对象；free queue 可以用 block 内部的双向链表指针维护，便于 O(1) 移动和淘汰。

## 21.6 Page size 和 block size

容易混淆两个概念。

Block size 通常指一个 block 容纳多少 token 的 KV。

Page size 指这个 block 在物理内存中占多少字节。

对普通 full attention 模型，一个简化 page size 可以理解为：

```text
page_size = block_size * num_layers * kv_hidden_size
```

这里 `kv_hidden_size` 表示一个 token 在单层中保存 KV 所需的字节数。

如果模型是 GQA 或 MQA，KV heads 少于 query heads，`kv_hidden_size` 会下降，KV Cache 显存也会下降。

如果模型层数更多、head dim 更大、KV dtype 更高精度，page size 就更大。

所以 block size 不是越大越好，也不是越小越好。

block size 大：

1. block table 更短。
2. 管理开销更小。
3. kernel 访问可能更规整。
4. 但最后一个 block 内部浪费更大。

block size 小：

1. 内部碎片更少。
2. 细粒度释放更灵活。
3. 但 block table 更长，管理和 kernel metadata 开销更大。

如果 block size 是 `S`，单个 token 的单层 KV 字节数是 `2H_{\mathrm{kv}}D_hb`，则普通 full attention 的单个 block page size 可以写成：

$$
P_{\mathrm{page}}=S\cdot L\cdot 2H_{\mathrm{kv}}D_hb
$$

请求 `i` 持有 `T_i` 个 KV tokens 时，需要的 block 数是：

$$
B_i=\left\lceil\frac{T_i}{S}\right\rceil
$$

最后一个 block 的内部浪费 tokens 是：

$$
R_i=SB_i-T_i
$$

## 21.7 Prefix cache 如何嵌入 memory management

Prefix caching 的目标是避免重复 prefill。

如果两个请求有相同前缀，后一个请求可以复用前一个请求已经算好的 KV blocks。

vLLM 的思路可以简化为：只缓存 full blocks，并为每个 full block 计算 hash。

一个 block hash 不只依赖本 block 的 tokens，还依赖前面 prefix 的 hash。

```text
block_hash_i = hash(parent_hash, block_tokens, extra_hashes)
```

其中 `extra_hashes` 可以包含：

1. LoRA id。
2. 多模态输入 hash。
3. cache salt。
4. 其他会影响模型计算结果的条件。

为什么要包含 parent hash？

因为同一个 block tokens 出现在不同上下文中，KV 结果可能不同。

例如：

```text
prefix A + block X
prefix B + block X
```

即使 `block X` 的 token 一样，attention 看到的历史上下文不同，KV cache 不能简单共享。

所以 prefix cache 本质是 memory management 和 correctness 的结合：既要复用显存中的 KV blocks，又不能错误复用不同上下文下的 KV。

可以把第 `j` 个 full block 的 hash 抽象成：

$$
h_j=H(h_{j-1},X_j,E_j)
$$

其中 `h_{j-1}` 是 parent hash，`X_j` 是本 block tokens，`E_j` 是 LoRA id、多模态 hash、cache salt、租户隔离信息等 extra hashes。prefix cache 命中率可以写成：

$$
R_{\mathrm{hit}}=\frac{N_{\mathrm{hit}}}{N_{\mathrm{lookup}}}
$$

## 21.8 Free queue 和 LRU eviction

当请求结束后，它占用的 blocks 不一定马上“彻底没用”。

如果某些 blocks 是 full blocks，并且有 hash，它们可以留在 prefix cache 中，供后续请求复用。

但这些 blocks 同时也可能进入 free queue，表示：

```text
当前没有 active request 引用，但如果内存紧张，可以被淘汰并重新分配。
```

这就形成了一个重要状态：

```text
cached but free
```

它不是正在被请求使用，但仍然保留着可复用 KV 内容。

当新请求需要分配 blocks 时，manager 会从 free queue 取 blocks。如果取到的是 cached block，就要先把它从 cache map 中删除，也就是 eviction。

LRU 的直觉是：越久没被使用的 cached block，越先被淘汰。

简化流程：

```python
def allocate_one_block():
    block = free_queue.pop_head()
    if block.block_hash is not None:
        cached_blocks.pop(block.block_hash, None)
        block.block_hash = None
    block.ref_cnt = 1
    return block
```

这里的关键点是：free 不等于清空内容，evict 才意味着这个 cached block 不再能被 prefix cache 命中。

用集合表示，active blocks、cached blocks 和 free blocks 不是互斥的。一个 block 可以同时属于 cached 与 free：

$$
N_{\mathrm{cachedfree}}=|\mathcal{C}\cap\mathcal{F}|
$$

真正的可分配容量由 free blocks 决定：

$$
F_t=|\mathcal{F}_t|
$$

但 prefix cache 的可复用机会由 cached blocks 和 hash 命中决定。

## 21.9 Ref count 和共享 blocks

Prefix cache 会让多个请求共享同一个 physical block。

例如：

```text
request A block table: [0, 1, 2, 7]
request B block table: [0, 1, 2, 9]
```

blocks 0、1、2 被两个请求共享，`ref_cnt=2`。

释放 request A 时，不能释放 blocks 0、1、2，只能把它们的 ref count 减 1。

只有当 `ref_cnt=0` 时，block 才能进入 free queue。

```python
def free_request(req):
    for block in req.blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            free_queue.push_tail(block)
```

ref count 是 prefix sharing 的底层安全机制。

没有 ref count，就可能出现 request A 结束释放了 block，但 request B 还在读这个 block，导致 attention 读到被覆盖的数据。

共享 block 数可以写成：

$$
N_{\mathrm{shared}}=\sum_{p}\mathbf{1}[c_p>1]
$$

其中 `c_p` 是 physical block `p` 的引用计数。

## 21.10 Preemption 和 recompute

即使有 block manager，系统仍可能出现 KV Cache 空间不足。

例如：

1. 突然来了大量长 prompt。
2. running 请求生成长度超预期。
3. `max_num_seqs` 设置太大。
4. `gpu_memory_utilization` 给 KV Cache 预留太少。
5. prefix cache 占用太多可复用但未被及时淘汰的 blocks。

当空间不足时，系统有几种选择：

1. 让新请求继续等待。
2. 拒绝请求。
3. 抢占某些 running 请求。
4. swap/offload 到 CPU。
5. 丢弃部分 KV，之后 recompute。

vLLM 优化文档中提到，V1 默认 preemption mode 更偏向 recompute，而不是 swap，因为在对应架构下 recompute 的开销可能更低。

Recompute 的意思是：释放某个请求的 KV Cache，让它暂时退出运行；等后续有空间时，再从 prompt 或部分状态重新计算 KV。

它的收益是系统更稳，不容易直接 OOM。

它的代价是端到端延迟变差，因为被抢占的请求要重算。

面试中可以这样说：

```text
Preemption 是 memory pressure 下的保护机制，不是性能优化本身。它保证系统能继续运行，但频繁 preemption 通常说明 KV cache budget 或并发参数设置不合理。
```

可以把抢占触发条件写成：

$$
G_{\mathrm{need\_preempt},\tau}=\mathbf{1}[\Delta B_{\tau}>F_{\tau}]
$$

如果本轮新增 block 需求 `\Delta B_{\tau}` 大于 free blocks `F_{\tau}`，scheduler 要么等待 / 拒绝，要么抢占、swap/offload 或 recompute。抢占率可以用：

$$
R_{\mathrm{preempt}}=\frac{N_{\mathrm{preempt}}}{N_{\mathrm{scheduled}}}
$$

## 21.11 Swap、offload 和多级 KV Cache

更广义的 memory management 不只看 GPU。

可以把 KV Cache 看成多级资源：

```text
GPU KV Cache -> CPU KV Cache -> remote KV store / distributed KV cache -> recompute
```

GPU KV Cache 最快，但容量最贵。

CPU KV Cache 容量更大，但通过 PCIe/NVLink 传输会增加延迟。

远端 KV Cache 可以跨节点共享或支持 PD 分离，但网络延迟和带宽会成为瓶颈。

Recompute 不占存储，但消耗算力和时间。

什么时候 offload 有意义？

1. 请求很长，但短期内不活跃。
2. GPU memory 紧张，但 CPU memory 充足。
3. 重算成本比传输成本更高。
4. 系统目标是避免 OOM，而不是极致低延迟。

什么时候 offload 可能是坏主意？

1. decode 每轮都需要访问被 offload 的 KV。
2. PCIe 带宽成为瓶颈。
3. 请求 TPOT 对用户非常敏感。
4. 实现复杂度和故障模式超过收益。

后面 PD 分离章节会更深入讨论跨节点 KV 迁移和共享。

## 21.12 Chunked prefill 和 memory budget

Chunked prefill 表面上是调度策略，本质也和 memory management 有关。

长 prompt prefill 有两个压力：

1. 一轮计算 token 数很大。
2. 一次性需要分配大量 KV blocks。

如果 prompt 长度是 16000，block size 是 16，就需要约 1000 个 blocks。

完整 prefill 会让本轮 token budget 和 block budget 同时吃紧。

Chunked prefill 把长 prompt 切成多个小块：

```text
16000 tokens -> 2048 + 2048 + ...
```

好处是：

1. 单轮 token budget 更可控。
2. 可以和 decode 请求混合调度。
3. 降低单轮延迟尖刺。
4. 避免长 prefill 长时间阻塞 streaming decode。

代价是：

1. 请求完成 prefill 需要多轮。
2. TTFT 可能受 chunk 策略影响。
3. scheduler 需要维护部分 prefill 状态。

vLLM 优化文档提到，V1 中 chunked prefill 在可用时默认启用，并且调度策略通常优先 decode，再用剩余 `max_num_batched_tokens` 预算调度 prefill；如果 prefill 放不下，会自动 chunk。

## 21.13 参数如何影响 memory management

几个参数经常一起出现。

`gpu_memory_utilization`：控制 vLLM 可以用多少 GPU memory 预分配 cache 或相关资源。

值更大：

1. KV Cache 空间更大。
2. 可承载更多并发或更长上下文。
3. preemption 可能减少。
4. 但留给其他进程、临时 buffer 或系统波动的余量更小。

`max_num_seqs`：控制同时调度的最大序列数。

值更大：

1. 并发上限更高。
2. decode batch 可能更大。
3. KV Cache 压力更大。
4. 单轮调度和采样开销可能增加。

`max_num_batched_tokens`：控制一轮最多处理多少 tokens。

值更大：

1. prefill 接纳能力更强。
2. TTFT 可能改善。
3. 吞吐可能提高。
4. 但 decode ITL/TPOT 可能抖动。

值更小：

1. 单轮延迟更可控。
2. streaming 更平滑。
3. 但长 prompt 更容易被切碎，TTFT 可能变差。

这些参数不能孤立调。正确做法是结合业务目标：低延迟聊天、批量离线生成、长文档 RAG、多租户 serving，它们的最优配置不同。

## 21.14 Hybrid KV Cache Manager：为什么混合模型更复杂

现代模型不一定所有层都是 full attention。

有些模型混合：

1. full attention。
2. sliding window attention。
3. local attention。
4. Mamba 或其他 state space 层。
5. KV sharing 层。

这些层对 cache 的需求不一样。

Full attention 需要保存所有历史 token 的 KV。

Sliding window attention 只需要最近窗口内的 KV。

Mamba 类层可能保存的不是标准 attention KV，而是 state。

如果仍然按“所有层、所有 token、同一种 block 规则”分配，会浪费大量显存，或者无法表达不同层的 cache 需求。

Hybrid KV Cache Manager 的核心思想是：把不同 attention/cache 类型分组，每组有自己的分配逻辑，但底层仍尽量使用统一 page size 的 memory pool。

简化理解：

```text
KV Cache Group 0: full attention layers
KV Cache Group 1: sliding window layers
KV Cache Group 2: another sliding window group or state group
```

Scheduler 看起来仍然向 KVCacheManager 要 slots，但内部可能由 coordinator 分发给多个 group manager。

这说明 memory management 会随着模型架构演进而变复杂。Serving engine 不能只为传统 decoder-only full attention 写死 cache 逻辑。

## 21.15 Memory metrics 应该看什么

排查 vLLM-like memory 问题时，不要只看 `nvidia-smi`。

应该看更细指标：

1. total KV blocks。
2. free KV blocks。
3. used KV blocks。
4. cached but free blocks。
5. prefix cache hit rate。
6. prefix cache eviction count。
7. allocation failure count。
8. preemption count。
9. recompute count。
10. blocks per request 分布。
11. prompt length 和 output length 分布。
12. GPU memory reserved 和 actual used。
13. CPU offload/swap in/out 次数。
14. TTFT、TPOT 与 free blocks 的相关性。

例如：

如果 GPU utilization 不高，但 TTFT 很差，可能是 waiting queue 被 KV block 不足卡住。

如果 tokens/s 正常但 TPOT 抖动，可能是长 prefill、preemption 或 CPU contention。

如果 free blocks 下降后不恢复，可能是 abort/failed 路径没有释放 blocks。

如果 prefix cache hit rate 高但吞吐没提升，可能瓶颈不在 prefill，而在 decode 或输出处理。

## 21.16 常见 OOM 和性能问题

问题一：长 prompt 高并发导致 OOM。

处理思路：降低 `max_num_seqs`，降低 `max_num_batched_tokens`，启用或调优 chunked prefill，增加 tensor parallel 或提高可用 KV cache 预算。

问题二：频繁 preemption。

处理思路：说明 KV cache 空间不足或并发过高。可以提高 `gpu_memory_utilization`、降低并发预算、减少上下文长度、调整 TP/PP 或优化 prefix cache 策略。

问题三：请求结束后显存不回落。

处理思路：区分 PyTorch reserved memory、KV block pool 预分配和真实泄漏。看 free blocks 是否恢复，而不是只看 `nvidia-smi`。

问题四：prefix cache 占用多但命中低。

处理思路：检查 prompt 是否真的共享、chat template 是否一致、cache salt 是否隔离、LoRA 或多模态 hash 是否导致无法复用。

问题五：offload 后 TPOT 明显变差。

处理思路：说明 decode 访问路径变慢，PCIe/网络传输可能成为瓶颈。需要减少 offload、调整抢占策略或换 recompute。

## 21.17 面试官会怎么问

问题一：vLLM 如何管理 KV Cache？

回答要点：把 KV Cache 切成固定大小 blocks，维护 block pool、free queue、request block table、ref count 和 prefix cache hash map。请求逻辑上连续，物理上可以不连续。

问题二：为什么 PagedAttention 能降低显存浪费？

回答要点：避免为每个请求预留大块连续 cache，按 block 动态分配，减少碎片和内部浪费，并支持请求结束后细粒度回收。

问题三：prefix caching 如何保证正确性？

回答要点：只缓存 full blocks，用 block tokens、parent hash 和 extra hashes 构造 block hash，确保相同 token 但不同上下文、LoRA、多模态输入或租户隔离不会错误共享。

问题四：preemption 是什么？

回答要点：当 KV cache 空间不足时，系统抢占部分请求释放 cache，后续通过 recompute 或 swap 恢复。它提升鲁棒性，但频繁发生会伤害端到端延迟。

问题五：如何调优 vLLM 显存相关参数？

回答要点：增加 `gpu_memory_utilization` 给 KV cache 更多空间，降低 `max_num_seqs` 或 `max_num_batched_tokens` 减少并发和单轮 token 压力，必要时增加 TP/PP 分摊权重显存，让更多显存留给 KV cache。

问题六：为什么混合 attention 模型需要特殊 KV cache manager？

回答要点：full attention、sliding window、Mamba 等层的 cache 需求不同，需要按类型分组和不同分配规则，同时保持统一 page size 或统一管理接口，避免浪费和错误复用。

## 21.18 标准回答模板

如果面试官问“vLLM 是怎么做 memory management 的”，可以这样回答：

```text
vLLM 的 memory management 核心在 KV Cache。模型权重通常是相对静态的，但 KV Cache 会随请求数、prompt 长度和生成长度动态增长，是高并发 serving 中最容易造成浪费和 OOM 的部分。

vLLM 用 PagedAttention 的思想把 KV Cache 切成固定大小的 physical blocks。每个请求维护 block table，把逻辑 token 位置映射到物理 blocks。KV cache manager 维护 block pool、free queue、request 到 blocks 的映射、ref count，以及 prefix cache 的 hash map。

这样做的好处是请求不需要连续显存，cache 可以按 block 动态分配和释放，finished 或 aborted 请求可以细粒度回收，多个请求还可以通过 prefix caching 共享相同 full blocks。Scheduler 在每轮调度时会结合 token budget 和 KV block budget 做 admission control，避免执行到一半才 OOM。

当 KV cache 空间不足时，系统可以让请求等待、触发 preemption、recompute 或 offload。调优上，gpu_memory_utilization、max_num_seqs、max_num_batched_tokens、TP/PP 都会影响留给 KV cache 的空间和并发能力。对于 sliding window、Mamba 等混合架构，还需要 Hybrid KV Cache Manager 按不同层的 cache 需求分组管理。
```

## 21.19 Memory management 公式、prefix cache 和可运行 demo

把本章的 memory management 验收合成一个教学版门禁：

$$
G_{\mathrm{mem}}=G_{\mathrm{pool}}G_{\mathrm{prefix}}G_{\mathrm{ref}}G_{\mathrm{free}}G_{\mathrm{evict}}G_{\mathrm{preempt}}G_{\mathrm{cleanup}}G_{\mathrm{metric}}
$$

其中：

1. `G_{\mathrm{pool}}`：block pool、free queue、request block table 初始化正确。
2. `G_{\mathrm{prefix}}`：相同 prefix 且 extra hashes 相同的 full blocks 能命中 prefix cache。
3. `G_{\mathrm{ref}}`：共享 blocks 有正确 ref count，不会被提前释放。
4. `G_{\mathrm{free}}`：finished 请求释放后，cached-but-free blocks 能被观测。
5. `G_{\mathrm{evict}}`：分配新 block 时能从 prefix cache 中淘汰 cached free block。
6. `G_{\mathrm{preempt}}`：KV 空间不足时能触发 preemption / recompute 证据。
7. `G_{\mathrm{cleanup}}`：所有请求结束后 active KV blocks 回到 0。
8. `G_{\mathrm{metric}}`：prefix hit、eviction、preemption、free blocks、max used blocks 都可复盘。

下面这个 0 依赖 demo 模拟 6 个 KV blocks、block size 为 4 的 toy memory manager。请求 A 写入两个 full prefix blocks 和一个 partial block；请求 B 复用 A 的两个 full prefix blocks；释放 A 后共享 block 不能被提前回收；请求 C 使用不同 extra hash，因此不能复用；请求 D 因 free blocks 不足触发 C 的 preemption / recompute，并在分配时淘汰一个 cached free block。

```python
from dataclasses import dataclass


@dataclass
class KVBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: tuple = None


class ToyKVMemoryManager:
    def __init__(self, total_blocks, block_size):
        self.block_size = block_size
        self.blocks = [KVBlock(i) for i in range(total_blocks)]
        self.free_queue = list(range(total_blocks))
        self.request_blocks = {}
        self.cached_blocks = {}
        self.trace = []
        self.prefix_lookups = 0
        self.prefix_hits = 0
        self.eviction_count = 0
        self.allocation_failures = 0
        self.preemption_count = 0
        self.recompute_count = 0
        self.max_used_blocks = 0

    def used_count(self):
        return len(self.blocks) - len(self.free_queue)

    def remember_peak(self):
        self.max_used_blocks = max(self.max_used_blocks, self.used_count())

    def block_hash(self, parent_hash, tokens, extra_hash):
        return (parent_hash, tuple(tokens), extra_hash)

    def chunks(self, tokens):
        step = self.block_size
        return [tokens[start:start + step] for start in range(0, len(tokens), step)]

    def _remove_from_free(self, block_id):
        if block_id in self.free_queue:
            self.free_queue.remove(block_id)

    def _take_free_block(self, reason):
        if not self.free_queue:
            return None
        block_id = self.free_queue.pop(0)
        block = self.blocks[block_id]
        if block.block_hash is not None and self.cached_blocks.get(block.block_hash) == block_id:
            self.cached_blocks.pop(block.block_hash)
            self.eviction_count += 1
            self.trace.append(f"evict_cached_block:{block_id}:{reason}")
        block.block_hash = None
        block.ref_cnt = 0
        return block

    def needed_new_blocks(self, tokens, extra_hash):
        needed = 0
        parent_hash = None
        for chunk in self.chunks(tokens):
            if len(chunk) == self.block_size:
                h = self.block_hash(parent_hash, chunk, extra_hash)
                if h not in self.cached_blocks:
                    needed += 1
                parent_hash = h
            else:
                needed += 1
        return needed

    def can_allocate_prompt(self, tokens, extra_hash):
        return self.needed_new_blocks(tokens, extra_hash) <= len(self.free_queue)

    def allocate_prompt(self, request_id, tokens, extra_hash):
        if not self.can_allocate_prompt(tokens, extra_hash):
            self.allocation_failures += 1
            self.trace.append(f"allocation_wait:{request_id}")
            return False

        table = []
        parent_hash = None
        for chunk in self.chunks(tokens):
            if len(chunk) == self.block_size:
                h = self.block_hash(parent_hash, chunk, extra_hash)
                self.prefix_lookups += 1
                if h in self.cached_blocks:
                    block_id = self.cached_blocks[h]
                    block = self.blocks[block_id]
                    self._remove_from_free(block_id)
                    block.ref_cnt += 1
                    self.prefix_hits += 1
                    table.append(block_id)
                    self.trace.append(f"reuse_prefix:{request_id}:block{block_id}")
                else:
                    block = self._take_free_block(f"allocate:{request_id}")
                    block.ref_cnt = 1
                    block.block_hash = h
                    self.cached_blocks[h] = block.block_id
                    table.append(block.block_id)
                    self.trace.append(f"allocate_full:{request_id}:block{block.block_id}")
                parent_hash = h
            else:
                block = self._take_free_block(f"allocate_partial:{request_id}")
                block.ref_cnt = 1
                table.append(block.block_id)
                self.trace.append(f"allocate_partial:{request_id}:block{block.block_id}")

        self.request_blocks[request_id] = table
        self.remember_peak()
        return True

    def release_request(self, request_id):
        released = []
        for block_id in self.request_blocks.pop(request_id, []):
            block = self.blocks[block_id]
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block_id not in self.free_queue:
                    self.free_queue.append(block_id)
                released.append(block_id)
        self.trace.append(f"release:{request_id}:{released}")
        return released

    def preempt_for_recompute(self, request_id):
        self.preemption_count += 1
        self.recompute_count += 1
        released = self.release_request(request_id)
        self.trace.append(f"preempt_recompute:{request_id}:{released}")
        return released

    def cached_free_blocks(self):
        return sorted(
            block.block_id
            for block in self.blocks
            if block.ref_cnt == 0 and block.block_hash is not None and block.block_id in self.free_queue
        )

    def ref_counts(self, block_ids):
        return {block_id: self.blocks[block_id].ref_cnt for block_id in block_ids}

    def prefix_hit_rate(self):
        if self.prefix_lookups == 0:
            return 0.0
        return round(self.prefix_hits / self.prefix_lookups, 3)

    def metrics(self):
        return {
            "total_blocks": len(self.blocks),
            "free_blocks": len(self.free_queue),
            "used_blocks": self.used_count(),
            "cached_free_blocks": len(self.cached_free_blocks()),
            "prefix_lookups": self.prefix_lookups,
            "prefix_hits": self.prefix_hits,
            "prefix_hit_rate": self.prefix_hit_rate(),
            "eviction_count": self.eviction_count,
            "allocation_failures": self.allocation_failures,
            "preemption_count": self.preemption_count,
            "recompute_count": self.recompute_count,
            "max_used_blocks": self.max_used_blocks,
        }


manager = ToyKVMemoryManager(total_blocks=6, block_size=4)
extra_prod = ("tenant", "prod", "lora:none")
extra_isolated = ("tenant", "isolated", "lora:none")

manager.allocate_prompt("A", [1, 2, 3, 4, 5, 6, 7, 8, 9], extra_prod)
a_table = list(manager.request_blocks["A"])

manager.allocate_prompt("B", [1, 2, 3, 4, 5, 6, 7, 8, 20, 21], extra_prod)
b_table = list(manager.request_blocks["B"])
shared_after_b = manager.ref_counts(b_table[:2])

manager.release_request("A")
after_release_a = manager.ref_counts(b_table[:2])

manager.allocate_prompt("C", [1, 2, 3, 4], extra_isolated)
c_table = list(manager.request_blocks["C"])

d_tokens = [30, 31, 32, 33, 34, 35, 36, 37, 38]
if not manager.can_allocate_prompt(d_tokens, extra_prod):
    manager.allocation_failures += 1
    manager.preempt_for_recompute("C")
manager.allocate_prompt("D", d_tokens, extra_prod)
d_table = list(manager.request_blocks["D"])

manager.release_request("B")
manager.release_request("D")

summary = {
    "a_table": a_table,
    "b_table_reused_prefix": b_table,
    "shared_ref_counts_after_b": shared_after_b,
    "ref_counts_after_release_a": after_release_a,
    "c_table_different_extra_hash": c_table,
    "d_table_after_preemption": d_table,
    "cached_free_blocks": manager.cached_free_blocks(),
    "metrics": manager.metrics(),
    "trace_tail": manager.trace[-8:],
}

gates = {
    "block_pool_initialized": manager.metrics()["total_blocks"] == 6,
    "prefix_cache_reused": b_table[:2] == a_table[:2] and manager.prefix_hits == 2,
    "ref_count_shared": all(count == 2 for count in shared_after_b.values()),
    "cached_free_visible": len(manager.cached_free_blocks()) >= 3,
    "lru_eviction_visible": manager.eviction_count == 1,
    "preemption_recompute_visible": manager.preemption_count == 1 and manager.recompute_count == 1,
    "cleanup_restores_free": manager.used_count() == 0 and manager.metrics()["free_blocks"] == 6,
    "metrics_ready": manager.prefix_hit_rate() > 0 and manager.max_used_blocks == 6,
}
gates["memory_management_gate"] = all(gates.values())

print("memory_management_summary=", summary)
print("memory_management_gates=", gates)
```

这份输出应当体现：

1. B 的前两个 blocks 复用了 A 的 prefix blocks，且 ref count 一度为 2。
2. A 释放后，共享 blocks 的 ref count 仍为 1，不会提前回到可覆盖状态。
3. C 使用不同 extra hash，即使 tokens 相同也不会复用 A/B 的 prefix block。
4. D 触发一次 preemption / recompute，并在分配过程中淘汰一个 cached free block。
5. 所有请求释放后 `used_blocks=0`，`free_blocks=6`，且仍能看到 cached-but-free blocks。

## 21.20 小练习

1. 假设 block size 为 16，一个 prompt 有 4096 tokens，需要多少 KV blocks？如果并发 32 个这样的请求，至少需要多少 blocks？
2. 画出 block pool、free queue、cached blocks、request block table 之间的关系。
3. 解释为什么 request finished 后，某些 blocks 可能进入 free queue 但仍保留在 prefix cache 中。
4. 设计一个 prefix cache hash，至少包含 parent hash、block tokens 和 LoRA id。
5. 观察一个线上 OOM case，列出你会检查的 8 个 memory metrics。
6. 比较 preemption recompute 和 CPU swap 的优缺点。

## 21.21 本章总结

vLLM memory management 的核心是 KV Cache block 化。

PagedAttention 让请求的逻辑 KV Cache 可以映射到非连续 physical blocks；KV Cache Manager 负责 block pool、free queue、ref count、request block table、prefix cache 和 eviction；scheduler 基于 token budget 和 block budget 做调度，避免显存不可控增长。

真正的线上 memory management 不只是“显存够不够”，还包括 prefix cache 是否命中、free blocks 是否恢复、preemption 是否频繁、chunked prefill 是否平衡 prefill/decode、CPU/远端 offload 是否值得，以及混合 attention 模型如何分组管理 cache。

下一章会继续讲 vLLM worker、executor 和 engine 架构，重点拆解 engine core 如何把 scheduler output 派发给 worker，worker 如何持有模型和 KV cache，并在单机多卡或多进程架构下执行 forward。
