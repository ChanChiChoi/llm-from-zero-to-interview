# 第十章：O(n^2) Attention 的计算与显存瓶颈

## 10.0 本讲资料边界与第二轮精修口径

截至 2026-06-10，本讲只使用公开论文、官方文档和主流框架资料来校准概念边界。这里讨论的是标准 scaled dot-product attention 的计算、显存和 IO 瓶颈，不把某个框架版本的 kernel 选择、某个模型的长上下文 recipe、或某个 toy demo 的阈值写成通用标准。

本讲重点区分四层问题：

1. 数学连接模式：full attention 允许每个 query token 和每个可见 key token 交互，因此 pair 数随序列长度平方增长。
2. Kernel / IO 优化：FlashAttention 等方法做 exact attention 的分块、在线 softmax 和 IO 优化，不把 full attention 变成线性 attention。
3. 模型结构优化：sliding window、sparse attention、linear attention、GQA / MQA / MLA 等会改变可见模式、近似形式或 KV 表示。
4. Serving 内存管理：PagedAttention 主要管理 KV cache 的 block 分配、复用和碎片，不是替代 scaled dot-product attention 的数学结构。

后面的公式默认使用：

1. `B` 表示 batch size。
2. `H` 表示 query head 数。
3. `T_q` 表示 query length，`T_k` 表示 key/value length。
4. `D_h` 表示 head dimension。
5. `s` 表示每个元素的字节数，例如 fp16/bf16 通常可粗略取 `s=2`。

## 10.1 本章定位

上一章讲了 Transformer 的优势：并行性、表达力和 scaling。现在开始讲它的核心瓶颈。

Transformer 最大的结构性瓶颈之一，就是 self-attention 对序列长度的二次复杂度。

如果序列长度是 `n`，每个 token 都和每个 token 计算一次相关性，那么 attention score 矩阵就是：

```text
n x n
```

当上下文从 2K 扩到 32K、128K、1M 时，问题会迅速放大。

本章要回答的问题是：

1. 为什么标准 self-attention 是 O(n^2)。
2. 这个 O(n^2) 具体来自哪些张量和操作。
3. 训练时 attention 的显存瓶颈在哪里。
4. 为什么长上下文不只是“显存多一点”的问题。
5. 为什么 FlashAttention 能省显存、提速度，但不改变 attention 数学复杂度。
6. 为什么很多 linear/sparse attention 方法没有完全替代标准 attention。
7. 训练、prefill、decode 阶段的瓶颈有什么区别。
8. 面试中如何准确区分 O(n^2) attention、KV cache、FlashAttention、PagedAttention。

本章的核心观点是：

```text
Attention 的强表达力来自 token-to-token 全连接路由，而这个全连接路由也带来了序列长度平方级的计算、显存和 IO 压力。
```

## 10.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 scaled dot-product attention，其 attention score 来自 QK^T。
2. Beltagy et al., 2020, *Longformer: The Long-Document Transformer*。指出标准 self-attention 随序列长度二次增长，并用 local window + global attention 处理长文档。
3. Wang et al., 2020, *Linformer: Self-Attention with Linear Complexity*。尝试用低秩近似把 self-attention 降到线性复杂度。
4. Choromanski et al., 2020, *Rethinking Attention with Performers*。用随机特征近似 softmax attention，实现线性时间和空间复杂度。
5. Dao et al., 2022, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*。提出 IO-aware exact attention，通过 tiling 减少 HBM 读写，不近似 attention。
6. Dao, 2023, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*。进一步优化 work partitioning 和并行效率，提高 attention kernel 性能。
7. PyTorch `scaled_dot_product_attention` 官方文档。用于校准当前框架中 math kernel、memory-efficient kernel、FlashAttention kernel、mask、dropout 和 grouped query attention 等接口边界。
8. Kwon et al., 2023, *PagedAttention/vLLM*。讨论 LLM serving 中 KV cache 动态增长和内存管理问题。它更偏第 11 章，但本章会用来区分 FlashAttention 和 PagedAttention。

需要说明的是，本章重点讲 attention 的训练和 prefill 阶段的 O(n^2) 瓶颈。KV cache 在自回归 decode 阶段的显存增长，会在下一章专门展开。

## 10.3 标准 Attention 的计算流程

标准 scaled dot-product attention：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

```math
S=\frac{QK^\top}{\sqrt{D_h}},\qquad A=\mathrm{softmax}(S+M),\qquad O=AV
```

这里 `M` 是可选 mask。causal mask、padding mask 或 packed sequence mask 最终都会影响 softmax 前的 score，可见位置加 0，不可见位置加一个足够小的负数。

常见 shape 是：

```math
X\in\mathbb{R}^{B\times T\times d},\qquad Q,K,V\in\mathbb{R}^{B\times H\times T\times D_h}
```

```math
S,A\in\mathbb{R}^{B\times H\times T_q\times T_k},\qquad O\in\mathbb{R}^{B\times H\times T_q\times D_h}
```

其中 `d=H D_h`。如果是普通 self-attention，通常 `T_q=T_k=T`；如果是 decode 单步，通常 `T_q=1`，`T_k` 是当前历史长度。

1. `X` shape 是 `[batch, seq_len, hidden]`。
2. `Q/K/V` shape 通常是 `[batch, heads, seq_len, head_dim]`。
3. `S` shape 是 `[batch, heads, seq_len, seq_len]`。
4. `A` shape 也是 `[batch, heads, seq_len, seq_len]`。
5. `O` shape 是 `[batch, heads, seq_len, head_dim]`。

瓶颈主要来自 `S` 和 `A`：

```math
N_{\mathrm{pair}}=B H T_q T_k
```

这就是 O(n^2) 的来源。

## 10.4 为什么是 O(n^2)

假设单个 batch、单个 head、序列长度 `n`、head dimension `d`。

QK^T 的计算是：

```math
Q\in\mathbb{R}^{n\times d},\qquad K\in\mathbb{R}^{n\times d},\qquad QK^\top\in\mathbb{R}^{n\times n}
```

每个 query token 都要和每个 key token 做一次 dot product。

dot product 的成本是 `d`，token pair 数量是 `n^2`，所以计算量约为：

```math
C_{QK}\approx 2n^2d
```

这里按一次 multiply-add 记 2 FLOPs。不同资料可能差一个常数因子，但不影响随 `n^2` 增长的结论。

再看 `A V`：

```math
A\in\mathbb{R}^{n\times n},\qquad V\in\mathbb{R}^{n\times d},\qquad AV\in\mathbb{R}^{n\times d}
```

这同样是：

```math
C_{AV}\approx 2n^2d
```

所以 attention 主要计算量是：

```math
C_{\mathrm{attn}}\approx 4n^2d
```

如果有 `h` 个 heads、batch size 是 `b`，就是：

```math
C_{\mathrm{attn}}\approx 4B H n^2 D_h
```

因为 `h * d = hidden_size`，也可以粗略看成：

```math
C_{\mathrm{attn}}=O(B n^2 d_{\mathrm{model}})
```

## 10.5 二次复杂度为什么可怕

序列长度翻倍，attention score 数量变成 4 倍。

例如单 head、单样本：

```math
N_{\mathrm{pair}}=n^2
```

```text
n=2K     -> 约 4M token pairs
n=4K     -> 约 16M token pairs
n=8K     -> 约 64M token pairs
n=32K    -> 约 1B token pairs
n=128K   -> 约 16B token pairs
```

这还只是单 head、单层、单样本。真实模型有几十层、几十个 heads、多个 batch。

所以长上下文训练不是简单地把 max_position 改大，也不是只要显存稍微多一点。

它会同时放大：

1. attention 计算量。
2. attention 中间激活显存。
3. HBM 读写压力。
4. backward 计算和保存张量。
5. 分布式通信和激活重计算成本。
6. 数据 packing 和 batch 构造复杂度。

## 10.6 Attention 显存主要花在哪里

训练时需要保存中间激活用于反向传播。

标准 attention 中，显存压力来自：

1. Q/K/V。
2. attention scores `S`。
3. softmax 后的 attention probabilities `A`。
4. dropout mask，如果 attention dropout 开启。
5. output `O`。
6. backward 所需的中间状态。

其中 `S` 和 `A` 是 `n x n`，最容易爆炸。

粗略估算，假设：

```text
B=1
H=32
T=32768
s=2 bytes
```

一个 attention score/probability 张量大小约为：

```math
M_{\mathrm{score}}=BHT^2s
```

```math
M_{\mathrm{score}}=1\times 32\times 32768^2\times 2\ \mathrm{bytes}\approx 68.7\ \mathrm{GB}\approx 64\ \mathrm{GiB}
```

这只是一个 `S` 或 `A` 张量，还没算 Q/K/V、其他层、梯度、optimizer state。如果朴素实现同时 materialize scores 和 probabilities，粗略就是两倍：

```math
M_{\mathrm{score+prob}}\approx 2BHT^2s
```

所以如果朴素 materialize 完整 attention matrix，32K 训练几乎不可接受。

## 10.7 训练显存不只是参数显存

很多初学者会先算模型权重：

```text
7B 参数 * 2 bytes ≈ 14GB
```

然后误以为一张 24GB 卡就可以轻松训练 7B。

真实训练显存包括：

1. 模型权重。
2. 梯度。
3. optimizer states，例如 Adam 的 m/v。
4. 前向激活。
5. attention scores/probabilities。
6. temporary buffers。
7. fragmentation 和框架开销。

在长序列训练中，activation 和 attention 中间张量可能比参数更难处理。

这也是为什么训练长上下文模型常用：

1. activation checkpointing。
2. FlashAttention。
3. sequence parallelism。
4. ZeRO/FSDP。
5. gradient accumulation。
6. mixed precision。
7. batch size / sequence length trade-off。

## 10.8 计算瓶颈和 IO 瓶颈

大模型训练不只是 FLOPs 问题，也受内存带宽影响。

GPU 有不同层级的内存：

```text
HBM：容量较大，但访问比片上 SRAM 慢。
SRAM/shared memory/register：容量小，但访问快。
```

朴素 attention 可能会把 `S` 和 `A` 写入 HBM，再读回来继续计算。对于 `n x n` 的大矩阵，这会产生巨大 IO。

FlashAttention 的核心洞察就是：

```text
attention 慢和占显存，不只是因为计算量大，也因为中间矩阵在 HBM 和片上内存之间反复读写。
```

所以高效 attention kernel 要关心 IO-awareness，而不只是数学复杂度。

## 10.9 FlashAttention 解决了什么

FlashAttention 是 exact attention，不改变数学结果。

它不是近似 attention，也不是 sparse attention。

它的核心做法是：

```text
把 Q/K/V 分块加载到片上 SRAM 中。
分块计算 attention。
在线维护 softmax 的归一化统计。
避免 materialize 完整 n x n attention matrix 到 HBM。
```

普通 attention 可能显式产生：

```text
S = QK^T
A = softmax(S)
O = AV
```

FlashAttention 逻辑上仍然计算同样的结果，但不把完整 `S` 和 `A` 常驻显存。

它带来的收益：

1. 显著减少 HBM 读写。
2. 训练时 attention activation 显存从二次级别显著降低。
3. wall-clock 速度提升。
4. 支持更长序列。
5. 保持 exact attention，不牺牲模型质量。

FlashAttention-2 进一步优化了 work partitioning 和 GPU 并行，使 attention kernel 更接近 GEMM 效率。

## 10.10 FlashAttention 没解决什么

FlashAttention 很重要，但不能误解。

它没有改变 full attention 的数学连接模式。

也就是说，每个 token 仍然要和每个 token 交互。理论计算复杂度仍然是：

```math
O(n^2)
```

它主要优化的是：

```text
IO、显存占用、kernel 执行效率
```

不是把 full attention 变成真正的 O(n)。

所以当序列长度极端变大，例如 1M tokens，即使用 FlashAttention，full attention 的计算量也可能不可接受。

此外，FlashAttention 对 mask pattern 和硬件 kernel 有要求。任意复杂 mask 不一定能高效支持。

## 10.11 为什么 softmax attention 难以简单线性化

标准 attention 是：

```math
O=\mathrm{softmax}(QK^\top)V
```

问题在于 softmax 作用在每个 query 对所有 keys 的 score 上。每个 query 的归一化分母不同：

```math
Z_i=\sum_{j=1}^{T_k}\exp(q_i^\top k_j)
```

这使得 attention 天然像一个全局 pairwise 操作。

如果想降到线性复杂度，常见思路包括：

1. 稀疏化：只算部分 token pair。
2. 低秩近似：认为 attention matrix 可压缩。
3. kernel approximation：用特征映射近似 softmax kernel。
4. 局部窗口：限制可见范围。
5. recurrence/state：用固定大小状态摘要历史。
6. memory/retrieval：只取相关片段进入上下文。

每条路线都在 trade-off：

```text
效率、精度、表达力、训练稳定性、硬件友好性、生态兼容性
```

## 10.12 Sparse Attention

Sparse attention 的思路是：不是所有 token pair 都计算。

例如：

```text
local window：每个 token 只看附近 w 个 token
global token：少数 token 可以看全局
strided pattern：按步长连接远处 token
block sparse：按块组织稀疏连接
```

如果每个 token 只看 `w` 个 token，复杂度可以从：

```math
O(n^2)
```

降到：

```math
O(nw)
```

当 `w` 远小于 `n` 时，接近线性。

Longformer 就使用 local window attention 加 task-motivated global attention，适合长文档任务。

Sparse attention 的问题是：

1. 远距离 token 不一定能直接交互。
2. pattern 设计影响任务效果。
3. 某些稀疏 pattern 硬件上不一定高效。
4. 对需要任意位置精确检索的任务可能不如 full attention。

## 10.13 Linear Attention 和 Kernel Approximation

Linear attention 试图避免显式构造 `n x n` attention matrix。

一种思路是把 softmax kernel 近似成特征映射内积：

```math
\exp(q^\top k)\approx \phi(q)^\top\phi(k)
```

于是：

```math
\mathrm{softmax}(QK^\top)V
```

可以改写为先聚合 K/V，再与 Q 交互，从而把复杂度降到线性级别。

Performer 使用 FAVOR+ 随机特征近似 softmax attention，目标是在理论上提供近似保证。

Linformer 则从低秩角度出发，认为 self-attention matrix 可以被低秩投影近似，从而降低时间和空间复杂度。

这些方法的价值是探索了 full attention 之外的可能性。

但它们没有完全取代标准 attention，原因包括：

1. 近似可能影响模型质量。
2. 对不同任务、规模、数据的稳定性不一。
3. 工程生态和 kernel 优化不如标准 attention 成熟。
4. 大规模 LLM 中 full attention + FlashAttention 的路线效果强且可靠。
5. 现代模型还可以通过 GQA、sliding window、RoPE scaling、检索等组合缓解瓶颈。

## 10.14 O(n^2) 在训练、Prefill、Decode 中不同

大模型使用有三个阶段要区分。

### 训练

训练时通常整段序列并行输入：

```math
T_q=T_k=n
```

attention 是完整的 `n x n`，即使 causal mask 屏蔽未来，朴素矩阵仍然是三角形级别，复杂度仍是 O(n^2)。

### Prefill

推理开始时，用户 prompt 一次性送入模型，构建第一批 KV cache。

如果 prompt 长度是 `n`：

```math
C_{\mathrm{prefill}}\propto n^2
```

长 prompt 的首 token 延迟，也就是 time-to-first-token，经常受 prefill 影响。

### Decode

自回归生成时，每次只生成一个新 token。

新 token 的 query_len 是 1，key_len 是历史长度 `t`：

```math
T_q=1,\qquad T_k=t,\qquad C_{\mathrm{decode-step}}\propto t
```

如果生成 `m` 个 token，总成本会累积。

```math
C_{\mathrm{decode-total}}\propto \sum_{r=1}^{m}(n+r-1)
```

Decode 阶段的核心瓶颈通常还包括 KV cache 显存和内存带宽。下一章会重点讲。

## 10.15 Causal Mask 是否把复杂度减半

Causal attention 只允许看过去，所以有效可见 token pair 是下三角：

```math
N_{\mathrm{causal}}=\frac{n(n+1)}{2}
```

相比 full bidirectional attention 的 `n^2`，大约少一半。

但复杂度阶数仍然是：

```math
O(n^2)
```

所以 causal mask 不能从根本上解决长序列瓶颈。

它只是把常数因子降低，并保证不看未来。

## 10.16 长上下文训练为什么难

把上下文从 4K 扩到 32K，不只是长度变成 8 倍。

attention token pair 变成：

```text
8^2 = 64 倍
```

这会影响：

1. 单 step 训练时间。
2. 激活显存。
3. batch size。
4. 梯度累计策略。
5. 数据吞吐。
6. 分布式通信。
7. checkpoint 保存和恢复时间。
8. 训练稳定性和 loss spike 排查。

很多长上下文训练必须牺牲 batch size 或使用更复杂并行策略。

这也解释了为什么“支持 128K 上下文”的模型不一定是在 128K 上从头 full attention 训练出来的。它可能结合了：

1. RoPE scaling / interpolation。
2. 长上下文继续训练。
3. sliding window 或 sparse attention。
4. synthetic long-context tasks。
5. retrieval-augmented training。
6. FlashAttention / sequence parallel。

## 10.17 Attention 瓶颈和 FFN 瓶颈的关系

Transformer block 中还有 FFN。

FFN 计算大约随序列长度线性增长：

```math
C_{\mathrm{ffn}}=O(n d_{\mathrm{model}} d_{\mathrm{ffn}})
```

Attention 则有：

```math
C_{\mathrm{attn}}=O(n^2 d_{\mathrm{model}})
```

当 `n` 较短时，FFN 可能占主要 FLOPs，尤其现代 LLM 的 FFN hidden size 很大。

当 `n` 很长时，attention 的 `n^2` 项会快速增长，成为主要瓶颈。

所以不同场景瓶颈不同：

```text
短上下文大模型：FFN/矩阵乘可能很重。
长上下文 full attention：attention score 和 AV 可能成为瓶颈。
高并发推理：KV cache 和内存带宽可能成为瓶颈。
```

面试中要避免一句“attention 一定是最大瓶颈”说死。要结合序列长度、模型结构、训练/推理阶段和硬件实现判断。

## 10.18 Memory-Efficient Attention 和 Activation Checkpointing

除了 FlashAttention，还有一些通用减显存方法。

Activation checkpointing 的思想是：

```text
前向时少存中间激活，反向时重新计算。
```

这可以降低显存，但会增加计算时间。

Memory-efficient attention 的思想是避免保存完整 attention probabilities，通过重计算或分块计算降低显存。

这些方法本质上都在做 trade-off：

```text
显存 ↓
计算/实现复杂度 ↑
```

FlashAttention 更进一步利用 GPU memory hierarchy，把分块、在线 softmax 和 IO 优化结合起来，因此成为现代训练和推理中非常常见的 attention kernel。

## 10.19 FlashAttention、PagedAttention、GQA 不是一回事

这三个词经常被混淆。

FlashAttention：

```text
优化 attention kernel。
不改变数学 attention。
主要减少 HBM 读写和中间矩阵显存。
```

PagedAttention：

```text
优化 serving 中 KV cache 的内存管理。
灵感来自操作系统分页。
减少碎片和重复，提升 batch size 和吞吐。
```

GQA/MQA：

```text
改变 attention 架构中的 K/V head 数。
从源头减少 KV cache 大小和读取带宽。
```

三者可以组合使用，但解决的问题层级不同：

```text
FlashAttention：kernel/IO 层。
PagedAttention：serving memory manager 层。
GQA/MQA：模型架构层。
```

## 10.20 面向专家：为什么 IO-aware 很关键

理论 FLOPs 不是全部。

一个算法如果需要频繁读写 HBM，即使 FLOPs 不高，也可能很慢。

朴素 attention 的问题之一是：

```text
QK^T 生成 S，写到 HBM。
softmax 读 S，写 A。
AV 读 A 和 V，写 O。
```

当 `S/A` 是巨大 `n x n` 矩阵时，HBM 流量非常大。

FlashAttention 用 tiling 让计算尽量在片上 SRAM 中完成，并用 online softmax 避免保存完整 attention matrix。

这就是 IO-aware 的意义：

```text
同样的数学结果，如果减少慢速内存读写，就能显著提升真实 wall-clock 性能。
```

这也是为什么一些理论上低复杂度的近似 attention，不一定在真实硬件上更快。稀疏、不规则、小矩阵操作可能无法充分利用 GPU。

## 10.21 面向专家：为什么近似 Attention 难替代 Full Attention

近似 attention 的目标很明确：降低复杂度。

但替代 full attention 很难，因为 full attention 有几个强优势：

1. 任意 token pair 可直接交互。
2. 数学形式简单稳定。
3. 大规模预训练效果反复验证。
4. kernel 生态成熟。
5. 支持 causal、bidirectional、cross-attention 等多种模式。
6. 与 RoPE、GQA、FlashAttention、tensor parallel 等生态高度兼容。

近似方法常见问题：

1. 任务依赖强。
2. 长距离精确检索可能受损。
3. 训练稳定性和超参更敏感。
4. 理论复杂度低但硬件利用率不一定高。
5. 大规模 LLM 复现和生态成本高。

所以实际路线经常是折中：

```text
核心层保留 full attention 或局部 full attention。
用 FlashAttention 优化 exact attention。
用 GQA/MQA 降 KV cache。
用 sliding window/sparse pattern 扩上下文。
用 retrieval/memory 减少必须放进上下文的 token。
```

## 10.22 Attention 成本审计指标与最小 demo

第二轮精修时，本章建议把 attention 瓶颈落到一组可审计指标，而不是只背 “O(n^2)”：

1. `attention_pair_count`：full、causal、sliding window 下分别有多少 query-key pair。
2. `score_tensor_memory`：如果 materialize score matrix，单个 `S` 张量要多少显存。
3. `score_plus_prob_memory`：如果同时 materialize `S` 和 `A`，中间矩阵要多少显存。
4. `prefill_tflops_per_layer`：长 prompt prefill 每层 attention 的粗略 FLOPs。
5. `decode_step_tflops_per_layer`：单步 decode 在同样历史长度下的 attention FLOPs。
6. `window_pair_ratio`：sliding window 相对 full attention 保留了多少 pair。
7. `attention_cost_gate`：把显存、FLOPs、上下文长度和部署预算组合成门禁。

下面是一个 0 依赖 demo。它不模拟真实 GPU kernel，只演示 “从 4K 到 32K 为什么是平方级放大”，以及 prefill 和 decode 的成本口径为什么不能混用。

```python
BYTES_PER_ELEM = 2


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def full_pairs(batch, heads, q_len, k_len):
    return batch * heads * q_len * k_len


def causal_visible_pairs(batch, heads, seq_len):
    return batch * heads * seq_len * (seq_len + 1) // 2


def attention_tflops(pair_count, head_dim):
    # QK^T and AV each cost roughly 2 * pairs * head_dim FLOPs.
    return 4 * pair_count * head_dim / 1e12


batch = 1
heads = 32
head_dim = 128
short_len = 4096
long_len = 32768
window = 4096

short_full = full_pairs(batch, heads, short_len, short_len)
long_full = full_pairs(batch, heads, long_len, long_len)
long_causal = causal_visible_pairs(batch, heads, long_len)
window_pairs = full_pairs(batch, heads, long_len, min(window, long_len))
decode_pairs = full_pairs(batch, heads, 1, long_len)

failed_gates = []
if gib(long_full * BYTES_PER_ELEM) > 16:
    failed_gates.append("full_score_tensor_over_budget")
if attention_tflops(long_full, head_dim) > 10:
    failed_gates.append("prefill_tflops_over_budget")
if window_pairs / long_full > 0.25:
    failed_gates.append("window_not_sparse_enough")

print("short_vs_long_pair_multiplier=", round(long_full / short_full, 2))
print("full_32k_score_tensor_gib=", round(gib(long_full * BYTES_PER_ELEM), 2))
print("full_32k_score_plus_prob_gib=", round(gib(2 * long_full * BYTES_PER_ELEM), 2))
print("causal_visible_32k_score_gib=", round(gib(long_causal * BYTES_PER_ELEM), 2))
print("prefill_32k_tflops_per_layer=", round(attention_tflops(long_full, head_dim), 4))
print("decode_one_step_32k_tflops_per_layer=", round(attention_tflops(decode_pairs, head_dim), 6))
print("sliding_window_4k_pair_ratio=", round(window_pairs / long_full, 3))
print("failed_gates=", failed_gates)
print("attention_cost_gate_pass=", len(failed_gates) == 0)
```

示例输出：

```text
short_vs_long_pair_multiplier= 64.0
full_32k_score_tensor_gib= 64.0
full_32k_score_plus_prob_gib= 128.0
causal_visible_32k_score_gib= 32.0
prefill_32k_tflops_per_layer= 17.5922
decode_one_step_32k_tflops_per_layer= 0.000537
sliding_window_4k_pair_ratio= 0.125
failed_gates= ['full_score_tensor_over_budget', 'prefill_tflops_over_budget']
attention_cost_gate_pass= False
```

这个 demo 的读法是：

1. 4K 到 32K 的长度是 8 倍，但 full attention pair 是 64 倍。
2. 32K 下单个 full score tensor 已经约 64 GiB；如果还 materialize probability tensor，就是约 128 GiB。
3. causal mask 的可见 pair 约少一半，但阶数仍然是平方级。
4. prefill 的每层 attention FLOPs 和 decode 单步不是一个量级，因此长 prompt 的 TTFT 和长输出 decode 的瓶颈要分开分析。
5. sliding window 能把 pair ratio 降到 0.125，但它改变了可见模式，需要再评估长距离证据召回和任务质量。

## 10.23 常见误区

### 误区 1：FlashAttention 把 O(n^2) 变成 O(n)

不对。FlashAttention 是 exact attention，主要优化 IO 和显存，不改变 full attention 的二次计算关系。

### 误区 2：O(n^2) 只是训练问题

不完全。训练和 prefill 有明显 O(n^2) attention 成本；decode 每步是 O(t)，但会受到 KV cache 显存和内存带宽限制。

### 误区 3：Causal mask 解决了二次复杂度

不对。Causal mask 只保留下三角，约少一半 token pair，但阶数仍是 O(n^2)。

### 误区 4：参数显存够就能训练长上下文

不对。长上下文训练中 attention activation、中间矩阵、optimizer state、梯度和临时 buffer 都会占显存。

### 误区 5：Linear attention 一定比 full attention 好

不一定。它可能降低理论复杂度，但可能带来近似误差、任务效果下降、训练不稳定或硬件利用率不足。

### 误区 6：PagedAttention 是一种模型 attention 结构

不准确。PagedAttention 主要是 serving 系统中的 KV cache 管理方法，不是像 MHA/GQA 那样的模型结构。

## 10.24 面试高频问题

### 题 1：为什么标准 self-attention 是 O(n^2)？

参考回答：

```text
因为每个 query token 都要和每个 key token 计算一次相关性。序列长度为 n 时，attention score matrix 的形状是 n x n。QK^T 和 attention probabilities 乘 V 都需要处理 n^2 个 token pair，所以复杂度是 O(n^2 * d)，通常简写为随序列长度 O(n^2)。
```

### 题 2：Attention 的显存瓶颈在哪里？

参考回答：

```text
训练时除了 Q/K/V，还可能需要保存 attention scores 和 softmax 后的 attention probabilities，它们的形状是 batch * heads * seq_len * seq_len。这个 n x n 中间矩阵会随序列长度平方增长。长上下文训练中，attention activation 和中间 buffer 可能成为显存瓶颈。
```

### 题 3：FlashAttention 解决了什么？

参考回答：

```text
FlashAttention 是 IO-aware 的 exact attention。它通过 tiling、在线 softmax 等方法，避免把完整 attention score/probability 矩阵写入 HBM，减少 HBM 读写和中间显存占用，从而提升速度并支持更长序列。它不近似 attention，也不改变数学结果。
```

### 题 4：FlashAttention 是否把复杂度从 O(n^2) 降到 O(n)？

参考回答：

```text
没有。FlashAttention 不改变 full attention 的 token pair 交互模式，每个 token 仍然需要和其他 token 计算注意力，所以理论计算复杂度仍是 O(n^2)。它主要减少 IO 和显存占用，提高 kernel 效率。
```

### 题 5：训练、prefill、decode 的 attention 成本有什么区别？

参考回答：

```text
训练时整段序列并行输入，query_len 和 key_len 都是 n，attention 是 O(n^2)。Prefill 阶段处理 prompt，也需要对 prompt 内 token 做 attention，因此长 prompt prefill 也是 O(n^2)。Decode 阶段每次生成一个 token，query_len 是 1，key_len 是历史长度 t，所以单步 attention 是 O(t)，但 KV cache 显存和内存带宽会成为瓶颈。
```

### 题 6：Sparse attention 和 linear attention 的区别是什么？

参考回答：

```text
Sparse attention 是减少 token pair，只计算局部窗口、全局 token 或块稀疏连接等部分 attention。Linear attention 通常试图通过低秩或 kernel approximation 避免显式 n x n attention matrix，把计算改写成线性形式。两者都想降低复杂度，但会在表达力、精确性、任务效果和硬件效率上产生 trade-off。
```

### 题 7：为什么理论复杂度低的方法不一定真实更快？

参考回答：

```text
GPU 性能不仅取决于 FLOPs，还取决于内存访问、kernel fusion、矩阵形状、并行度和硬件利用率。一些稀疏或近似 attention 理论复杂度低，但操作不规则、小矩阵多、访存模式差，可能无法充分利用 GPU，真实 wall-clock 不一定优于优化良好的 full attention + FlashAttention。
```

### 题 8：FlashAttention、PagedAttention、GQA 分别优化什么？

参考回答：

```text
FlashAttention 优化 attention kernel 和 GPU IO，不改变 attention 数学结果。PagedAttention 优化 LLM serving 中 KV cache 的内存管理，减少碎片和重复。GQA 改变模型结构，减少 K/V head 数，从源头降低 KV cache 显存和读取带宽。它们处在 kernel、系统、模型结构三个不同层面。
```

## 10.25 小练习

1. 推导单 head attention 中 QK^T 和 AV 的计算复杂度。
2. 计算 batch=1、heads=32、seq_len=8192、fp16 下一个 attention probability 矩阵大约占多少显存。
3. 解释为什么序列长度从 4K 到 32K，attention token pair 增长 64 倍。
4. 用自己的话解释 FlashAttention 为什么是 exact attention。
5. 比较 full attention、sliding window attention、linear attention 的优缺点。
6. 解释为什么 causal mask 不能从根本上解决 O(n^2)。
7. 画出训练、prefill、decode 三个阶段的 query_len/key_len 关系。
8. 用一张表区分 FlashAttention、PagedAttention、GQA/MQA。
9. 运行 10.22 的 attention 成本审计 demo，把 `long_len` 从 32768 改成 65536，解释 score tensor 显存和 prefill FLOPs 如何变化。
10. 把 `window` 从 4096 改成 8192，观察 `sliding_window_4k_pair_ratio` 的变化，并说明为什么更大的窗口不是免费提升。

## 10.26 本章总结

本章讲了 O(n^2) Attention 的计算与显存瓶颈。

核心结论：

1. 标准 self-attention 的 `QK^T` 会产生 `n x n` attention score matrix。
2. Attention 的计算复杂度随序列长度平方增长，长上下文下会迅速放大。
3. 训练时 attention scores/probabilities 等中间张量会造成巨大显存压力。
4. 长上下文训练不只是改 max length，还涉及显存、IO、batch size、分布式和稳定性。
5. FlashAttention 是 IO-aware exact attention，减少 HBM 读写和中间显存，但不改变 O(n^2) 数学复杂度。
6. Sparse/linear attention 试图降低复杂度，但会带来表达力、近似误差、硬件效率和生态兼容性 trade-off。
7. 训练、prefill、decode 的 attention 瓶颈不同，不能混为一谈。
8. FlashAttention、PagedAttention、GQA/MQA 分别优化 kernel、serving memory manager 和模型结构层面。
9. Attention 成本审计要同时看 pair count、score/prob memory、prefill FLOPs、decode step、window ratio 和质量风险。

下一章会进入 KV Cache、长上下文推理和显存增长问题，专门解释自回归 decode 阶段为什么 KV cache 会成为 serving 的核心瓶颈。
