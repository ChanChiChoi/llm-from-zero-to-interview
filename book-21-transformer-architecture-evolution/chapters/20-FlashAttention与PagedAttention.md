# 第二十章：FlashAttention、PagedAttention 与 Attention Kernel 优化

## 20.1 两个名字解决的是两层不同问题

FlashAttention 和 PagedAttention 经常被放在同一张“推理优化”清单里，但它们优化的对象不同。FlashAttention 是 attention 计算的 kernel/算法重排：不显式 materialize 巨大的 score/probability 矩阵，利用 GPU SRAM 做分块和在线 softmax，结果仍可与标准 attention 对齐。PagedAttention 是 KV cache 的内存管理和寻址方案：把每个请求的历史分成固定 block，通过 block table 管理动态增长、共享和回收。

前者主要减少 attention 中间结果的 HBM 读写，后者主要减少 KV cache 的碎片和预留浪费。一个 serving engine 可以同时使用两者，也可以只使用其中一个。

## 20.2 标准 attention 为什么会产生中间矩阵

标准计算为：

~~~math
S=\frac{QK^\top}{\sqrt{d}},\qquad
A=\operatorname{softmax}(S),\qquad
O=AV
~~~

若 query 和 key 长度都是 `N`，`S` 和 `A` 的元素规模为 `N^2`。训练时需要保存反向传播所需的中间量，推理时虽然可以避免保存全部梯度，但 naive kernel 仍可能在 GPU 全局内存中读写这些矩阵。问题不只是 FLOPs，而是 HBM 带宽和工作集大小。

FlashAttention 的目标是分块处理 `Q/K/V`，在片上存储一个小块的 score，做完 softmax 累积后丢弃，不把完整 `N×N` 矩阵写回 HBM。它没有把数学 attention 近似成线性 attention，因而应区分“memory-efficient exact attention”和“approximate attention”。

## 20.3 在线 softmax 的稳定公式

对一行 score，直接计算 `exp(s_j)` 可能溢出。分块 online softmax 维护最大值 `m`、归一化和 `l` 以及加权 value 累积 `o`。合并新块时可以写成：

~~~math
m_{new}=\max(m, m_b)
~~~

~~~math
l_{new}=e^{m-m_{new}}l+e^{m_b-m_{new}}l_b
~~~

~~~math
o_{new}=e^{m-m_{new}}o+e^{m_b-m_{new}}o_b
~~~

最后输出 `o/l`。这组式子说明为什么不需要保存完整概率矩阵，同时保持数值稳定。实际 FlashAttention 会进一步处理 causal mask、warp work partition、shared memory 和反向传播。

## 20.4 一个教学版分块 attention

~~~python
import math
import torch


def blocked_attention(q, k, v, block=2):
    # q/k/v: [seq, dim]，仅用于展示在线 softmax思想
    n, d = q.shape
    out = torch.zeros_like(q)
    for start in range(0, n, block):
        q_block = q[start:start + block]
        scores = q_block @ k.T / math.sqrt(d)
        future = torch.arange(n)[None, :] > torch.arange(start, min(start + block, n))[:, None]
        scores = scores.masked_fill(future, float("-inf"))
        out[start:start + q_block.size(0)] = scores.softmax(dim=-1) @ v
    return out


q = k = v = torch.randn(6, 4)
print(blocked_attention(q, k, v).shape)
~~~

这个版本仍然一次性生成每个 query block 的 score，不是生产级 FlashAttention；它只让读者看到“按 query block 处理”的接口。真正的 kernel 还要把 key/value 分块并在线合并，且要处理 `[batch, heads, seq, dim]`、变长 batch 和 dtype。

## 20.5 PagedAttention 解决 KV cache 的动态内存问题

自回归生成中，每个请求的 KV cache 随输出增长，不同请求长度不同、结束时间不同。如果为每个请求预留一段连续最大内存，会产生内部碎片；如果频繁搬移连续数组，会产生复制成本。PagedAttention 借鉴虚拟内存，把逻辑序列分成固定大小的 token block，物理 block 不要求连续。

逻辑位置 `p` 对应：

~~~math
b=\left\lfloor\frac{p}{B}\right\rfloor,\qquad o=p\bmod B
~~~

其中 `B` 是 block token 数。请求维护一个 block table：

~~~text
logical block 0 -> physical block 17
logical block 1 -> physical block 03
logical block 2 -> physical block 42
~~~

attention kernel 根据 table 找到对应物理页，模型逻辑上仍看到连续历史。

## 20.6 分页 cache 的最小模型

~~~python
class ToyPagedKV:
    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.tables = {}
        self.lengths = {}

    def append(self, request_id, token):
        length = self.lengths.get(request_id, 0)
        if length % self.block_size == 0:
            if not self.free:
                raise MemoryError("no free KV block")
            self.tables.setdefault(request_id, []).append(self.free.pop())
        self.lengths[request_id] = length + 1
        return self.tables[request_id][-1], length % self.block_size, token

    def release(self, request_id):
        for block in self.tables.pop(request_id, []):
            self.free.append(block)
        self.lengths.pop(request_id, None)


cache = ToyPagedKV(block_size=4, num_blocks=3)
for token in range(9):
    print(cache.append("A", token))
cache.release("A")
print("free", cache.free)
~~~

这个 demo 只管理 token 到 block 的映射，没有真正存 K/V，也没有并发锁、引用计数、prefix sharing 或 GPU kernel。它用来说明：请求的逻辑长度和物理存储位置是两个概念。

## 20.7 Prefix sharing 和引用计数

多个请求共享相同 system prompt 或对话前缀时，可以让它们的 block table 指向同一组物理 block。共享 block 不能被任一请求原地修改，因此追加新 token 时采用 copy-on-write 或只在新 block 写入。释放时要用引用计数：

~~~math
\operatorname{refcount}(b)=\#\{r\mid b\in\operatorname{table}(r)\}
~~~

只有计数归零才能回收到 free list。错误的回收会导致两个请求输出互相污染，错误的引用不释放则会造成隐性内存泄漏。

## 20.8 两种技术如何组合

FlashAttention 的输入可以是分页寻址后的 K/V block，PagedAttention 负责从哪些物理页取数据，Flash-like kernel 负责如何在 GPU 上高效计算 score/softmax/value。组合的关键是 block size、head layout、page table lookup、变长 batch 和 prefix sharing 的一致性。

如果 block 太小，页表和索引开销上升；太大则尾部碎片增加。若 prefix sharing 很多，物理页节省明显；若请求短且没有共享，收益可能小于管理开销。需要按 prompt 长度分布和并发测量，而不是只引用论文中的吞吐数字。

## 20.9 常见失败模式

FlashAttention 侧的失败包括 mask 错误、非 contiguous tensor、head_dim 或 dtype 不支持、因果/非因果模式混淆、数值误差造成 logits 差异和 fallback 到慢 kernel。PagedAttention 侧的失败包括 block table 越界、释放后仍被引用、batch 重排不更新 owner、prefix cache 跨 tokenizer/template 误复用、preemption 只保存逻辑长度而没有物理映射。

排查顺序是：在小序列上对比 dense reference 的 logits；打印每个 request 的逻辑 token、physical block、offset 和 refcount；验证 append、release、rollback、swap、resume；再看 profiler 中到底使用了哪一个 kernel。不要看到 API 名称就假定优化已经生效。

## 20.10 训练、推理和显存边界

FlashAttention 对训练的收益通常包括降低 activation memory 和 HBM traffic；PagedAttention 的主要价值出现在动态 serving，因为训练 batch 往往已固定、序列 padding/packing 策略不同。反之，Paged KV 管理不能消除 attention score 的计算，FlashAttention 也不能解决请求间 KV 碎片。

KV 的总容量仍可粗略写为：

~~~math
M_{KV}=2\cdot L\cdot B\cdot T\cdot H_{kv}\cdot d_h\cdot b
~~~

分页减少的是浪费和共享成本，并没有改变每个已存 token 的基础信息量。GQA、KV 量化、滑窗、latent cache 和递归 state 才会改变 payload 或历史形式。

## 20.11 机制与边界：不要用“理论复杂度”替代 roofline 分析

FlashAttention 通常仍保留 `O(N^2d)` 的算术计算量，只降低中间矩阵的内存占用和读写；它的优势来自 memory hierarchy、tile、occupancy 和 work partition。PagedAttention 的额外成本是页表查找、block gather、调度和可能的碎片；其收益来自提高可用 batch 和减少预留浪费。

对端到端系统应分别记录：prefill/decode tokens/s、TTFT、TPOT、峰值显存、KV 使用率、碎片率、page fault/eviction、prefix hit rate、kernel fallback 和 p99。一个优化可能提升吞吐却让单请求延迟变差，必须绑定 SLO。

## 20.12 面试追问、误区与练习

**问：FlashAttention 是不是把 attention 复杂度从二次降成线性？**

标准回答：通常不是。它在保持精确 attention 结果的前提下，用分块和 online softmax 减少中间矩阵的 HBM 读写，实际速度和显存更好，但数学 score 交互仍可能是二次。

**问：PagedAttention 是不是压缩了 KV？**

标准回答：主要是分页管理动态 KV 的物理存储，降低碎片、支持共享和灵活增长；它不等于低秩压缩或量化，单 token KV payload 仍由模型的 head 数、维度和 dtype 决定。

常见误区包括把两个技术当作同一算法、混淆 page/block 与 OS 页、只看显存不看 kernel、prefix sharing 没做引用计数，以及把框架集成名当作实际 kernel 证据。

练习：实现一个 dense reference、blocked attention 和 toy paged KV；为 100 个不同长度请求计算连续预留、分页和 prefix sharing 的理论浪费，列出仍未计入的 runtime 成本。

## 20.13 两种优化分别改变什么

FlashAttention 主要优化 attention 的 IO：在片上分块计算，避免把完整 score 和 probability 矩阵反复写回显存。PagedAttention 主要优化 KV cache 的物理管理：把历史按 block 分配，减少连续内存和请求长度变化带来的碎片。

因此二者可以组合，不能互相替代。FlashAttention 不会自动解决请求之间的 KV 碎片，PagedAttention 也不会自动让 score 计算变成线性。

## 20.14 从论文复杂度到服务指标

评估 attention kernel 时记录：

~~~text
flops
global_memory_bytes
kernel_time
occupancy
prefill_tokens_per_second
decode_tokens_per_second
kv_fragmentation
peak_memory
p99
~~~

相同 FLOPs 的 kernel 可能因为内存访问和形状不同而有完全不同的时间。服务端还要把 block 分配、释放、共享、抢占和重算纳入压测。

## 20.15 Kernel 优化要放回内存层次

FlashAttention 的核心不是改变 attention 的数学结果，而是通过 tiling、在线 softmax 和片上内存复用，减少中间矩阵写回 HBM。PagedAttention 主要解决的是自回归服务中 KV cache 的动态分配、碎片和共享。二者可以组合，但不应把“更快的 attention kernel”和“更好的 cache allocator”当成同一个优化。

分析一个 kernel 时要问：计算量在哪里，数据从哪一级内存读写，是否受 HBM 带宽限制，是否有足够并行度，变长 batch 是否导致 padding 浪费。理论 FLOPs 不变时，memory traffic、occupancy、kernel launch 和 page indirection 仍然会改变真实 tokens/s。

## 20.16 PagedAttention 的正确性不只是显存不爆

分页 cache 的逻辑序列通过 block table 映射到物理 page。每次 token 追加、batch 重排、preemption、prefix sharing 和释放都必须保持逻辑位置到物理页的一致性。page 被共享时需要引用计数或 copy-on-write；请求结束时不能提前释放仍被其他请求使用的前缀。

一个最小回归顺序是：单请求无共享、变长 batch、共享前缀、分叉后写入、抢占恢复、取消释放和长序列扩容。每一步比较 full recompute 与 cache decode 的 logits，记录 request id、logical position、page id、revision 和 checksum。这样才能区分“容量优化成功”和“输出悄悄错位”。

## 20.17 用端到端实验而不是论文数字验收

基准至少包含短输入/长输入、短输出/长输出、低并发/高并发、相同前缀/随机前缀、取消、抢占和 OOM 注入。报告 TTFT、TPOT、吞吐、p50/p95/p99、峰值显存、page waste、prefix hit、preemption 次数和单位成功成本。

FlashAttention 路径应额外检查数值误差、mask、causal 边界、不同 dtype 和长序列；PagedAttention 路径应额外检查 block table、引用计数和跨租户 cache key。若为了性能触发稠密 fallback，应把它作为可观察事件，不能在总平均值中隐藏。

本文所述 FlashAttention 可参考论文 https://arxiv.org/abs/2205.14135，PagedAttention 可参考论文 https://arxiv.org/abs/2309.06180；具体 API、kernel 支持范围和 vLLM/SGLang 行为随版本变化，应以对应实现和基准为准。

## 20.18 在线 softmax 和分页 cache 的共同正确性

FlashAttention 的在线 softmax 与 PagedAttention 的 block table 看起来属于不同层，但它们在 serving 中会共同影响输出。kernel 先按逻辑 token 顺序读取多个物理 block，再对 score 做稳定归一化；如果 page table 漏掉一个 block、重复一个 block 或把 padding 当作有效 token，online softmax 仍然会得到一个数值稳定但语义错误的结果。

因此正确性测试不能只检查 `NaN` 或输出 shape。对每个请求保存逻辑位置到物理页的映射、每个 query 的有效 key 数、mask 后的 score 范围和最终 logits checksum，并与 full recompute reference 比较。一个合理的回归矩阵包括：单 block、跨 block 边界、共享 prefix、分叉写入、释放后复用、抢占恢复和不同长度混合 batch。

```math
E_logit = max over t,v of abs(logit_cached(t,v) - logit_full(t,v))
```

如果 `E_logit` 只在 block 边界出现，优先查 offset 和 mask；如果所有位置都出现相同尺度误差，查 dtype、scale 或 kernel；如果释放另一个请求后才出错，查 reference count 和 owner 更新。这样可以把“FlashAttention 数值误差”和“PagedAttention 物理映射错误”分开。

## 20.19 从单请求节省到并发容量

假设设备可用 KV 容量为 `M_cap`，每个请求的有效 KV 为 `m_i`，连续预留带来的浪费为 `w_i`，分页 metadata 和共享管理成本为 `m_meta`，则可用并发不是简单的 `M_cap / mean(m_i)`，而应近似看成：

```math
N_admit = max N such that sum_i(m_i + w_i) + m_meta <= M_cap
```

分页降低 `w_i`，prefix sharing 降低重复前缀的 `m_i`，FlashAttention 主要影响 prefill workspace 和内存流量。三者对并发容量的贡献不同。短请求很多时，metadata 和调度成本可能占比上升；长请求很多时，KV payload 和 admission 变成主导；混合流量还要留出短请求的 deadline 余量。

一个有用的压测是固定总 token 数，分别改变请求切分方式：少量长请求、许多短请求、相同前缀请求和完全随机前缀请求。比较 continuous batching、paged cache 和 prefix sharing 的 TTFT、TPOT、峰值显存、碎片率、prefix hit、fallback 和单位成功成本。这样才能知道优化是否真正提升了产品容量，而不是只让单条 microbenchmark 更漂亮。

## 20.20 从 roofline 到真实 kernel 选择

FlashAttention 通过 tiling、在线 softmax 和减少中间矩阵读写改善 attention kernel 的内存行为，但收益取决于序列长度、head dimension、dtype、mask、GPU 架构和 batch。对短序列，启动和调度常数可能占主导；对长 prefill，HBM 访问和 workspace 可能占主导；对 decode，瓶颈又可能转成 KV 读取和小 batch。

因此选择 kernel 时应先标注阶段和瓶颈：

~~~math
T_{\mathrm{kernel}}
\gtrsim\max\left(
\frac{F}{P_{\mathrm{compute}}},
\frac{B_{\mathrm{HBM}}}{BW_{\mathrm{HBM}}},
\frac{B_{\mathrm{workspace}}}{BW_{\mathrm{workspace}}}
\right).
~~~

这个 roofline 只提供下界，不能替代 profiler。实际压测还要看 fused kernel 是否生效、非支持形状是否 fallback、mask 是否引入分支、量化/FP8 是否改变累加精度。报告“用了 FlashAttention”时，应同时给出 shape、dtype、版本和是否包含 fallback。

## 20.21 PagedAttention 的 allocator 不变量

分页 KV cache 的正确性依赖逻辑 token 顺序与物理 block table 的映射。对请求 `r`，可以抽象成：

~~~math
\mathrm{logical\_token}(r,j)
\longrightarrow
(\mathrm{block\_id},\mathrm{offset}),
~~~

并要求每个有效逻辑位置恰好映射一次，已释放 block 不再被请求引用，共享 prefix 只有在引用计数正确时才能复用。分叉写入时，父请求和子请求应共享只读 prefix，首个写入位置触发 copy-on-write；若直接覆盖共享页，两个请求的 logits 会同时变化。

回归应覆盖 block 边界、变长 batch、prefix sharing、分叉后写入、抢占、取消释放、page reuse 和 snapshot/restore。每次记录 request id、logical position、block table、owner 和 checksum。数值稳定但映射错误的输出最难排查，所以 allocator invariant 要在 kernel 之前单独测试。

## 20.22 把优化收益换算成并发容量

设设备可用 KV 容量为 `M_cap`，请求 `i` 的有效 cache 为 `m_i`，分页浪费为 `w_i`，元数据为 `m_meta`，则接入集合 `A` 需要满足：

~~~math
\sum_{i\in A}(m_i+w_i)+m_{\mathrm{meta}}
\le M_{\mathrm{cap}}.
~~~

FlashAttention 主要降低 attention 中间读写和 workspace，PagedAttention 主要降低 `w_i` 和动态分配碎片，prefix sharing 主要降低相同前缀的重复 `m_i`。三者不能把收益简单相加。短请求很多时，元数据和调度占比会上升；长请求很多时，KV payload 和 admission 是主导；共享前缀不同时，prefix cache 的收益又会消失。

容量实验应固定总输入/输出 token 数，改变请求切分方式：少量长请求、许多短请求、相同前缀和随机前缀。报告 TTFT、TPOT、p99、峰值显存、page waste、prefix hit、fallback、OOM 和单位成功成本。单请求显存下降而混合流量 p99 上升，不能算完整优化成功。

## 20.23 资料边界与小结

FlashAttention-2 论文见 https://arxiv.org/abs/2307.08691；PagedAttention/vLLM 论文见 https://arxiv.org/abs/2309.06180。具体 kernel 支持、block size、量化 cache、prefix cache 和调度策略必须以所用引擎版本的官方文档和 profiler 为准。

FlashAttention 优化“怎么算”，PagedAttention 优化“历史放在哪里、怎么增长和共享”。把这两个层次拆开，才能正确判断长上下文服务的瓶颈。
