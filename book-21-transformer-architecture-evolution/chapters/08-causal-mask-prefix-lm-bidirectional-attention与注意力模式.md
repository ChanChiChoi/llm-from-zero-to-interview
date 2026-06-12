# 第八章：Causal Mask、Prefix LM、Bidirectional Attention 与注意力模式

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修主要对齐 Transformer 原论文中的 encoder self-attention、masked decoder self-attention 和 encoder-decoder attention，BERT 的 deep bidirectional encoder，UniLM 用不同 self-attention mask 统一理解、生成和 seq2seq 目标的思路，UL2 对不同语言建模范式的混合，Sparse Transformer / Mistral 等长序列 attention pattern，以及 PyTorch `scaled_dot_product_attention` 对 `attn_mask`、`is_causal` 和高效 kernel 的公开语义。

写作边界如下：

1. 本讲重点是 mask 语义、可见性边界、信息泄漏和工程审计，不展开所有高效 attention kernel 的实现细节。
2. FlashAttention、block sparse、sliding window、varlen packing 等工程实现会因框架、GPU、版本和 kernel 选择而变化；正文只写稳定的概念边界，不把某个版本的支持矩阵泛化成永久结论。
3. 第二轮精修会把关键公式改成 GitHub Markdown 更稳的 fenced math block，并补一个 0 依赖 Python demo，专门审计 causal、prefix、packed、sliding window 和错误 mask 的可见性。

## 8.1 本章定位

上一章讲了位置编码：模型如何知道 token 的顺序和距离。本章继续讲另一个决定 Transformer 行为的核心机制：attention mask 和 attention pattern。

Self-Attention 的公式本身只说明“token 之间如何读信息”，但没有规定“哪些 token 可以被读”。这个可见性规则由 mask 决定。

同样是 Transformer，只要 mask 不同，就会变成完全不同的模型：

```text
Bidirectional attention：当前位置可以看左边和右边，适合理解。
Causal attention：当前位置只能看自己和过去，适合自回归生成。
Encoder-decoder attention：encoder 双向理解输入，decoder 因果生成输出，并通过 cross-attention 读取输入。
Prefix LM：prefix 部分双向可见，生成部分因果可见。
Sparse / sliding window attention：只看局部或稀疏位置，用效率换长上下文。
```

本章要回答的问题是：

1. attention mask 到底屏蔽什么。
2. causal mask 为什么能防止信息泄漏。
3. bidirectional attention 为什么适合理解任务。
4. decoder-only、encoder-only、encoder-decoder 的 mask 有什么区别。
5. Prefix LM 和普通 causal LM 有什么不同。
6. padding mask、loss mask、attention mask 为什么不能混淆。
7. packed sequence、KV cache、FlashAttention 中 mask 有哪些工程坑。
8. sparse attention、sliding window attention 等注意力模式想解决什么问题。

本章的核心观点是：

```text
Transformer 的架构形态，很大程度上是由 attention 可见性规则决定的。
```

## 8.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 Transformer encoder-decoder 架构，包含 encoder self-attention、masked decoder self-attention 和 encoder-decoder attention。
2. Devlin et al., 2018/2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*。代表 encoder-only 双向注意力路线。
3. Raffel et al., 2019/2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*。T5 统一 text-to-text 框架，使用 encoder-decoder 架构和 span corruption 预训练。
4. Lewis et al., 2019, *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension*。BART 可视为双向 encoder 加自回归 decoder 的去噪 seq2seq 模型。
5. Dong et al., 2019, *Unified Language Model Pre-training for Natural Language Understanding and Generation*。UniLM 用不同 self-attention masks 统一 bidirectional、unidirectional 和 seq2seq 目标。
6. Tay et al., 2022, *UL2: Unifying Language Learning Paradigms*。UL2 讨论多种语言建模范式和 mixture-of-denoisers。
7. Child et al., 2019, *Generating Long Sequences with Sparse Transformers*。提出稀疏 attention pattern，降低长序列计算成本。

需要说明的是，本章重点讲 mask 与注意力模式的架构含义，不展开所有高效 attention 算法。O(n^2) attention、KV cache、FlashAttention、长上下文推理会在后续章节继续深入。

## 8.3 Attention Mask 是什么

普通 scaled dot-product attention 是：

```math
S=\frac{QK^\top}{\sqrt{d_k}}
```

```math
A=\mathrm{softmax}(S)
```

```math
O=AV
```

mask 的作用是在 softmax 之前，把不允许看的位置对应的 score 变成一个极小值：

```math
\tilde{S}=S+M
```

常见做法是：

```math
M_{ij}=
\begin{cases}
0, & V_{ij}=1\\
-\infty, & V_{ij}=0
\end{cases}
```

其中 `V_ij=1` 表示 key 位置 `j` 对 query 位置 `i` 可见，`V_ij=0` 表示不可见。

这样 softmax 后，被禁止位置的概率接近 0。

例如：

```math
\mathrm{softmax}([2.0,1.0,-\infty])
\approx[0.731,0.269,0.000]
```

所以 attention mask 控制的是：

```text
每个 query 位置可以从哪些 key/value 位置读取信息。
```

它不是删除 token，也不是直接控制 loss，而是控制 attention 的信息流。

## 8.4 三种容易混淆的 mask

工程里经常同时出现三种 mask。

### Attention Mask

attention mask 控制 token 之间能不能互相看。

例如：

```text
当前位置不能看未来 token
padding token 不能被真实 token 关注
不同 packed sample 之间不能互相看
```

### Padding Mask

padding mask 用来标记哪些 token 是补齐出来的。

例如 batch 中两条样本长度不同：

```text
样本 A：长度 5
样本 B：长度 3，需要 pad 到 5
```

padding 位置不应该被 attention 读取，也通常不应该参与 loss。

### Loss Mask

loss mask 控制哪些 token 参与训练 loss。

例如 SFT 中通常只让 assistant 回复部分参与 loss：

```text
system/user token：labels = -100，不参与 loss
assistant token：参与 loss
```

这三者不能混淆：

```text
attention mask：控制能看谁。
padding mask：标记哪些是 pad。
loss mask：控制哪些位置算 loss。
```

一个 token 不参与 loss，不代表它不能被 attention 看见。SFT 中 user prompt 通常不参与 loss，但 assistant 生成时必须能看见 user prompt。

## 8.5 Causal Mask

Causal mask 也叫 autoregressive mask、look-ahead mask、future mask。

它的规则是：

```math
V_{ij}^{\mathrm{causal}}=\mathbf{1}[j\le i]
```

也就是第 `i` 个位置只能看第 `0` 到第 `i` 个位置，不能看 `i` 之后的未来 token。

用矩阵表示，`1` 表示可见，`0` 表示不可见：

```text
       key position
       0 1 2 3 4
q=0    1 0 0 0 0
q=1    1 1 0 0 0
q=2    1 1 1 0 0
q=3    1 1 1 1 0
q=4    1 1 1 1 1
```

这是一个下三角矩阵。

为什么需要 causal mask？因为语言模型训练时常把整段序列并行输入模型：

```text
输入：x_0, x_1, x_2, x_3
目标：x_1, x_2, x_3, x_4
```

如果没有 causal mask，位置 `x_1` 在预测 `x_2` 时可能已经通过 attention 看到了 `x_2`，这就是信息泄漏。训练 loss 会虚低，但模型推理时不能提前看到未来 token，性能会崩。

## 8.6 Causal LM 的训练和推理

Causal LM 的目标是 next-token prediction：

```math
P(x_1,\ldots,x_n)=
\prod_{t=1}^{n}P(x_t\mid x_{1:t-1})
```

训练时：

```text
整段 token 并行输入
用 causal mask 禁止每个位置看未来
每个位置预测下一个 token
```

推理时：

```text
已经生成的 token 作为上下文
模型预测下一个 token
采样或贪心选出下一个 token
把新 token 追加到上下文
重复
```

训练和推理的关键一致性是：

```text
训练中第 t 个位置看到的信息，不能超过推理中第 t 个位置可用的信息。
```

causal mask 正是保证这件事的机制。

## 8.7 手写 Causal Mask

最小 PyTorch 写法如下：

```python
import torch

def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask[None, None, :, :]

def apply_mask(scores, mask):
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

其中 scores shape 通常是：

```text
[batch, heads, query_len, key_len]
```

mask shape 可以是：

```text
[1, 1, query_len, key_len]
```

通过 broadcasting 作用到所有 batch 和 heads。

如果用 additive mask，也可以写成：

```python
def make_additive_causal_mask(seq_len, device, dtype):
    allowed = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]
```

工程里要注意：

1. mask 方向不能反。
2. dtype 要和 attention kernel 兼容。
3. fp16/bf16 中不要随意用过大的负数导致数值问题。
4. FlashAttention 等 kernel 可能不接受任意形状的显式 mask。
5. query_len 和 key_len 在 KV cache 场景下可能不相等。

## 8.8 Bidirectional Attention

Bidirectional attention 的规则是：

```text
每个非 padding token 可以看同一序列中的所有非 padding token。
```

可见性矩阵大致是全 1：

```text
       key position
       0 1 2 3 4
q=0    1 1 1 1 1
q=1    1 1 1 1 1
q=2    1 1 1 1 1
q=3    1 1 1 1 1
q=4    1 1 1 1 1
```

BERT 是典型的 bidirectional encoder。它通过 masked language modeling 训练：把部分 token 替换成 `[MASK]` 或其他形式，然后用左右上下文预测被 mask 的 token。

例如：

```text
输入：我 喜欢 [MASK] 学习
目标：机器
```

模型可以同时看左边“我 喜欢”和右边“学习”。这对理解任务非常有利。

Bidirectional attention 适合：

1. 文本分类。
2. 句子匹配。
3. 信息抽取。
4. 阅读理解。
5. reranking。
6. embedding 表征。

但它不适合直接做标准自回归生成，因为每个位置在训练时看到了未来上下文。

## 8.9 Encoder-Only、Decoder-Only、Encoder-Decoder

Transformer 的三大经典形态，可以从 attention 可见性来理解。

### Encoder-Only

代表模型：BERT、RoBERTa。

```text
self-attention：双向
训练目标：MLM、span prediction、对比学习等
擅长任务：理解、分类、抽取、embedding、rerank
```

Encoder-only 的每一层都允许 token 看左右上下文，输出是深层上下文表征。

### Decoder-Only

代表模型：GPT 系列、LLaMA、Qwen、Mistral。

```text
self-attention：causal
训练目标：next-token prediction
擅长任务：开放式生成、对话、代码、工具调用、in-context learning
```

Decoder-only 的优点是训练和推理形式高度统一：都是给定前文预测后文。

### Encoder-Decoder

代表模型：原始 Transformer、T5、BART。

```text
encoder self-attention：双向
decoder self-attention：causal
cross-attention：decoder 读取 encoder 输出
训练目标：seq2seq、denoising、translation、summarization
```

Encoder-decoder 适合输入输出结构差异明显的任务，例如翻译、摘要、文本改写、语音识别后处理等。

## 8.10 Encoder-Decoder 的三种 Attention

原始 Transformer decoder 中有三种 attention 相关模块。

第一，encoder self-attention：

```text
source token 之间双向可见
```

例如翻译中英文输入句子内部可以互相理解。

第二，decoder masked self-attention：

```text
target token 只能看已经生成的 target token
```

防止生成目标句子时看到未来目标词。

第三，encoder-decoder cross-attention：

```text
decoder 的每个 target position 可以读取 encoder 的所有 source positions
```

cross-attention 的 Q 来自 decoder hidden states，K/V 来自 encoder outputs：

```math
Q=H_{\mathrm{dec}}W_Q,\qquad
K=H_{\mathrm{enc}}W_K,\qquad
V=H_{\mathrm{enc}}W_V
```

这使得 decoder 在生成每个目标 token 时，都能对输入序列做对齐和读取。

## 8.11 Prefix LM

Prefix LM 介于 bidirectional LM 和 causal LM 之间。

它把序列分成两段：

```text
prefix：作为条件输入，可以双向可见
target：作为生成部分，只能因果可见
```

可见性规则是：

1. prefix token 之间可以互相看。
2. target token 可以看全部 prefix。
3. target token 之间只能看自己和过去 target。
4. prefix 通常不看 target，避免条件输入泄漏生成答案。

如果 prefix 长度为 `P`，总长度为 `N`，可以把 Prefix LM 的可见性写成：

```math
V_{ij}^{\mathrm{prefix}}=
\begin{cases}
1, & i<P,\ j<P\\
1, & i\ge P,\ j<P\\
1, & i\ge P,\ P\le j\le i\\
0, & \mathrm{otherwise}
\end{cases}
```

矩阵示例，假设前 3 个 token 是 prefix，后 3 个 token 是 target：

```text
       k0 k1 k2 | k3 k4 k5
q0     1  1  1  | 0  0  0
q1     1  1  1  | 0  0  0
q2     1  1  1  | 0  0  0
       ---------|---------
q3     1  1  1  | 1  0  0
q4     1  1  1  | 1  1  0
q5     1  1  1  | 1  1  1
```

Prefix LM 适合把“理解输入”和“生成输出”放在一个 decoder-like self-attention 结构里。

例如：

```text
prefix：请把下面英文翻译成中文：I love machine learning.
target：我喜欢机器学习。
```

target 可以完整读取 prefix，但不能看未来 target。

## 8.12 Prefix LM 和 Encoder-Decoder 的区别

Prefix LM 和 encoder-decoder 都能做条件生成，但结构不同。

Encoder-decoder：

```text
输入经过 encoder 双向编码。
输出经过 decoder 因果生成。
decoder 通过 cross-attention 读取 encoder。
```

Prefix LM：

```text
输入和输出放在同一个序列里。
通过 self-attention mask 区分 prefix 和 target 的可见性。
没有独立 encoder-decoder cross-attention。
```

工程 trade-off：

1. Prefix LM 结构统一，更接近 decoder-only。
2. Encoder-decoder 对输入输出分离更明确。
3. Encoder-decoder 可以预先编码 source，适合 source 很长、target 多次生成的场景。
4. Prefix LM 在现代 decoder-only 大模型中更容易和 prompt/instruction 格式结合。
5. Encoder-decoder 的 cross-attention 增加模块复杂度，但对 seq2seq 任务很自然。

## 8.13 UniLM：用 Mask 统一任务

UniLM 的思想很适合理解本章主题：

```text
同一套 Transformer 参数，通过不同 self-attention mask 支持不同语言建模目标。
```

它可以构造：

1. Bidirectional mask：用于理解任务。
2. Unidirectional causal mask：用于生成任务。
3. Seq2seq mask：source 双向可见，target 因果可见且可看 source。

这说明架构形态不一定完全由模块数量决定，也可以由 mask 决定。

当然，mask 统一并不意味着所有任务效果都会一样好。不同训练目标、数据分布、位置编码、推理方式仍然会影响最终能力。

## 8.14 UL2：不同语言建模范式的混合

UL2 进一步强调，不同任务可能需要不同的语言建模范式。

它把预训练目标拆成多种 denoising / language modeling 模式，例如：

1. 短 span corruption。
2. 长 span corruption。
3. causal language modeling。

这些目标背后对应不同的信息可见性和预测方式。

直觉上：

```text
理解任务需要双向上下文。
生成任务需要从左到右的因果建模。
填空/去噪任务需要根据可见片段恢复不可见片段。
```

UL2 的意义在于提醒我们：不要把“Transformer 架构”和“预训练目标”混为一谈。模型能看什么、预测什么、训练数据如何构造，都会塑造最终能力。

## 8.15 Sparse Attention 和 Sliding Window Attention

full attention 的可见性是：

```text
每个 token 可以看所有允许位置
```

对于长度 `n`，attention score 矩阵大小是 `n x n`，计算和显存都随 `n^2` 增长：

```math
|S|=n^2,\qquad C_{\mathrm{attn}}=O(n^2d_h)
```

长上下文下，这会非常贵。

Sparse attention 的想法是：

```text
不是所有 token pair 都需要直接相连，只保留一部分重要连接。
```

常见模式包括：

1. Local window：每个 token 只看附近窗口。
2. Strided attention：按固定步长看远处 token。
3. Global token：少数特殊 token 可以看全局。
4. Block sparse：按 block 组织可见性。
5. Sliding window：窗口随位置滑动，常用于长文本 decoder。

例如 sliding window causal attention：

```math
V_{ij}^{\mathrm{window}}=
\mathbf{1}[0\le i-j<w]
```

也就是第 `i` 个 token 只能看 `[i-w+1,\ldots,i]` 范围内的历史 token。

它的优点是显著降低长序列成本。缺点是远距离信息不能直接访问，必须通过多层传播或额外 global mechanism。

Mistral 等模型使用 sliding window attention，说明在小模型和长上下文场景里，局部注意力是非常实用的 trade-off。

## 8.16 Attention Pattern 的图视角

可以把 attention pattern 看成一张有向图：

```text
每个 token 是一个节点。
如果 token i 可以看 token j，就有一条从 i 到 j 的边。
```

Causal attention 是一张只指向过去的图。

Bidirectional attention 是近似完全图。

Sliding window attention 是局部带状图。

Global sparse attention 是少数中心节点连接很多节点。

这个图视角有助于理解信息传播。

如果第 10 个 token 不能直接看第 0 个 token，但第 10 层后可能间接获得第 0 个 token 信息吗？取决于图连通性和层数。

例如 window size 为 4 的 sliding window attention 中，每层最多把信息向前传播 4 个位置。多层叠加后，感受野会扩大：

```text
1 层：最多看到前 4 个 token
2 层：可能间接看到前 8 个 token
L 层：理论上可看到更远
```

但间接传播不等于 full attention。长距离信息会经历多层压缩和混合，效果可能不同。

## 8.17 Packed Sequence Training 中的 Mask

为了提高训练效率，很多大模型训练会把多条短样本 packing 到一个长 sequence 中。

例如：

```text
[样本 A tokens] [样本 B tokens] [样本 C tokens]
```

如果只用普通 causal mask，会出现问题：样本 B 的 token 可以看见样本 A 的 token，因为 A 在 B 的左边。

这通常是不希望的。正确做法是 block-diagonal causal mask：

```text
样本内部：因果可见
样本之间：完全不可见
```

可见性大致是：

```text
A 内部下三角 | A-B 不可见 | A-C 不可见
B-A 不可见  | B 内部下三角 | B-C 不可见
C-A 不可见  | C-B 不可见  | C 内部下三角
```

如果 packed mask 做错，会导致：

1. 样本之间信息泄漏。
2. loss 虚低。
3. 模型学习到错误的跨样本依赖。
4. 对话边界、文档边界混乱。
5. 线上推理表现和训练不一致。

## 8.18 KV Cache 场景下的 Mask

训练时通常 `query_len = key_len = seq_len`。

推理使用 KV cache 时，新 token 的 query_len 可能是 1，key_len 是历史长度加 1：

```text
Q: [batch, heads, 1, head_dim]
K: [batch, heads, past_len + 1, head_dim]
V: [batch, heads, past_len + 1, head_dim]
```

这时 causal mask 不再是简单的 `1 x 1` 下三角矩阵，而要允许新 query 看全部历史 K/V 和当前 K。

例如新 token 位于全局位置 `t`：

```text
允许看 key position <= t
禁止看 key position > t
```

在逐 token decoding 中，通常没有未来 key，所以可以不显式构造 causal mask。但在 batch decoding、prefill、chunked prefill、speculative decoding 中，mask 仍然很重要。

常见坑：

1. prefill 阶段和 decode 阶段 mask 逻辑不一致。
2. cache position 和 position id 不一致。
3. 不同样本 past length 不同，batch mask 没处理好。
4. prefix cache 复用时，后续 token 的可见范围错误。
5. sliding window cache 丢弃历史后，mask 和位置编码没有同步。

## 8.19 FlashAttention 和 Mask

FlashAttention 通过分块计算避免显式 materialize 完整 attention matrix，从而降低显存并提升速度。

但这也意味着：

```text
不是所有任意形状的 mask 都能高效支持。
```

常见高效支持的模式包括：

1. causal mask。
2. no mask。
3. padding mask 或 varlen sequence。
4. sliding window causal mask。

如果你传入一个非常复杂的任意 `n x n` mask，可能会：

1. 退化到普通 attention。
2. 触发不支持的 kernel。
3. 需要 block-sparse kernel。
4. 增加显存开销。

所以工程上设计 attention pattern 时，不只要考虑数学可行，还要考虑 kernel 是否支持。

这也是现代架构设计的现实约束：

```text
好的 attention pattern 必须同时满足建模需求和硬件/kernel 友好性。
```

## 8.20 Mask 的数值稳定性

mask 通常在 softmax 前加极小值。很多代码会写：

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

这在 fp32 中通常没问题，但在 fp16/bf16 或特定 kernel 中要更谨慎。

更稳妥的写法通常是：

```python
scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
```

但也要注意，如果一整行全被 mask，softmax 可能产生 NaN。

例如某个 query 没有任何可见 key：

```text
softmax([-inf, -inf, -inf]) -> NaN
```

因此需要保证每个有效 query 至少能看一个 key，或者在 kernel/逻辑里专门处理全 mask 行。

常见导致全 mask 的原因：

1. padding mask 和 causal mask 合并方向错误。
2. 左 padding 时 position 和 mask 没对齐。
3. packed sequence 边界处理错误。
4. query_len/key_len 不一致时 causal mask 偏移错误。
5. 空样本或截断后只剩特殊 token。

## 8.21 Mask 可见性审计指标与最小 demo

本章最容易落地出错的不是“知道 causal mask 是下三角”，而是多个 mask、样本边界、目标函数和高效 kernel 同时存在时，可见性是否仍然正确。可以定义一个简化的 mask 审计门禁：

```math
G_{\mathrm{mask}}=
\mathbf{1}[R_{\mathrm{future}}=0]\cdot
\mathbf{1}[R_{\mathrm{sample}}=0]\cdot
\mathbf{1}[C_{\mathrm{row}}=1]\cdot
\mathbf{1}[C_{\mathrm{kernel}}=1]
```

其中：

```math
R_{\mathrm{future}}=
\frac{
\sum_i\sum_j \mathbf{1}[j>i]\mathbf{1}[V_{ij}=1]
}{
\max(1,\sum_i\sum_j \mathbf{1}[V_{ij}=1])
}
```

表示 causal 或 decoder 场景下未来位置泄漏的比例。

```math
C_{\mathrm{row}}=
\prod_i \mathbf{1}\left[\sum_j V_{ij}\ge 1\right]
```

表示每个有效 query 至少有一个可见 key，避免全 mask 行造成 softmax NaN。

```math
C_{\mathrm{kernel}}=
\mathbf{1}[P_{\mathrm{mask}}\in\mathcal{P}_{\mathrm{fast}}]
```

表示当前 mask pattern 是否属于高效 kernel 支持的常见模式。这里不是说复杂 mask 不能用，而是提醒复杂任意 dense mask 可能导致 kernel 退化或需要专门的 block-sparse 实现。

下面 demo 用纯 Python 做一个最小可见性审计：它生成 causal、Prefix LM、packed causal、sliding window mask，并故意构造三个 bad case：普通 causal mask 用在 packed sequence 上造成跨样本泄漏，一整行全被 mask，以及任意 dense mask 不在常见 fast path 中。

```python
import math


def causal_mask(n):
    return [[j <= i for j in range(n)] for i in range(n)]


def prefix_lm_mask(prefix_len, target_len):
    n = prefix_len + target_len
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i < prefix_len:
                row.append(j < prefix_len)
            else:
                row.append(j < prefix_len or (prefix_len <= j <= i))
        rows.append(row)
    return rows


def packed_causal_mask(lengths):
    n = sum(lengths)
    sample_id = []
    for sid, length in enumerate(lengths):
        sample_id.extend([sid] * length)
    rows = []
    for i in range(n):
        rows.append([sample_id[i] == sample_id[j] and j <= i
                     for j in range(n)])
    return rows, sample_id


def sliding_window_causal_mask(n, window):
    return [[0 <= i - j < window for j in range(n)] for i in range(n)]


def visible_counts(mask):
    return [sum(row) for row in mask]


def leak_count(mask, sample_id=None):
    leaks = 0
    for i, row in enumerate(mask):
        for j, allowed in enumerate(row):
            if not allowed:
                continue
            if j > i:
                leaks += 1
            if sample_id is not None and sample_id[i] != sample_id[j]:
                leaks += 1
    return leaks


def softmax_visible(scores, mask_row):
    visible = [score for score, allowed in zip(scores, mask_row) if allowed]
    if not visible:
        return None
    max_score = max(visible)
    exps = [math.exp(score - max_score) if allowed else 0.0
            for score, allowed in zip(scores, mask_row)]
    total = sum(exps)
    return [round(value / total, 4) for value in exps]


causal = causal_mask(5)
prefix = prefix_lm_mask(prefix_len=3, target_len=3)
plain = causal_mask(5)
packed, sample_id = packed_causal_mask([2, 3])
sliding = sliding_window_causal_mask(6, window=3)
bad_all_mask = [[True, False], [False, False]]

scores = [0.2, 1.0, 2.0, 3.0, 4.0]
masked_probs = softmax_visible(scores, causal[2])
leaky_probs = softmax_visible(scores, [True] * 5)

issues = []
if leak_count(causal) != 0:
    issues.append("causal_future_leak")
if leak_count(plain, sample_id) != 0:
    issues.append("plain_causal_cross_sample_leak")
if any(count == 0 for count in visible_counts(bad_all_mask)):
    issues.append("all_mask_row")
if "arbitrary_dense_mask" not in {
    "causal", "none", "padding", "sliding_window"
}:
    issues.append("kernel_pattern_not_fast_path")

gate_pass = not issues

print("causal_counts=", visible_counts(causal))
print("prefix_counts=", visible_counts(prefix))
print("sliding_counts=", visible_counts(sliding))
print("masked_probs_q2=", masked_probs)
print("leaky_future_mass_q2=", round(sum(leaky_probs[3:]), 4))
print("plain_causal_packing_leak=", leak_count(plain, sample_id))
print("block_diag_packing_leak=", leak_count(packed, sample_id))
print("mask_issues=", issues)
print("mask_gate_pass=", gate_pass)
```

一组固定输出如下：

```text
causal_counts= [1, 2, 3, 4, 5]
prefix_counts= [3, 3, 3, 4, 5, 6]
sliding_counts= [1, 2, 3, 3, 3, 3]
masked_probs_q2= [0.1078, 0.2399, 0.6522, 0.0, 0.0]
leaky_future_mass_q2= 0.8683
plain_causal_packing_leak= 6
block_diag_packing_leak= 0
mask_issues= ['plain_causal_cross_sample_leak', 'all_mask_row', 'kernel_pattern_not_fast_path']
mask_gate_pass= False
```

这组结果可以这样读：

1. causal mask 中第 `i` 行可见数量依次为 `1,2,3,4,5`，符合只能看自己和过去。
2. Prefix LM 的前 3 行都能看 3 个 prefix token，target 行可见数量逐步增加，说明 target 能看全部 prefix 和过去 target。
3. sliding window 在窗口填满后每行只看 3 个 token，用局部可见性换取更低成本。
4. `masked_probs_q2` 的未来位置概率为 0；如果错误地让第 2 行看全序列，未来 token 概率质量会达到 `0.8683`。
5. 普通 causal mask 用在 packed sequence `[2,3]` 上会产生 6 个跨样本可见关系，block-diagonal causal mask 则把泄漏降为 0。
6. 门禁失败是刻意构造的：它暴露了 packed sequence 边界、全 mask 行和 kernel fast path 三类常见工程风险。

## 8.22 面向专家：Mask、目标函数和数据格式的耦合

mask 不是单独的工程细节，它和目标函数、数据格式强耦合。

以 SFT 为例：

```text
attention mask：assistant token 可以看 system/user/历史 assistant。
loss mask：通常只在 assistant 回复 token 上算 loss。
causal mask：assistant token 不能看未来 assistant token。
```

如果把 user token 的 attention 也 mask 掉，assistant 就看不到问题。

如果 user token 参与 loss，模型会学习复述 user 输入。

如果 causal mask 方向错，模型会看到未来答案。

所以排查训练异常时，不能只看 input_ids。至少要检查：

1. decoded text。
2. attention mask。
3. labels。
4. labels 中的 `-100` 分布。
5. position ids。
6. sample boundary。
7. logits/labels shift 是否正确。

很多“模型不听话”“loss 很低但生成很差”“SFT 后复读用户问题”的问题，本质都是 mask 或格式错。

## 8.23 面向专家：Bidirectional 不是不能生成，Causal 也不是不能理解

一个常见误解是：

```text
BERT 只能理解，GPT 只能生成。
```

更准确地说：

```text
BERT 的训练目标和双向 mask 更适合理解表征，不适合直接 left-to-right 生成。
GPT 的 causal mask 和 next-token objective 更适合自回归生成，但也能通过 prompt 完成理解任务。
```

Bidirectional 模型可以用于生成类任务，例如通过 mask filling、迭代填空、encoder-decoder 组合等方式。但它不是天然的逐 token 左到右生成器。

Causal LM 可以做分类、抽取、阅读理解，因为这些任务可以转化成文本生成或选择题格式。但它在一些纯表征任务上未必比专门训练的 encoder embedding 模型更高效。

架构倾向不是绝对能力边界，而是训练目标、mask 和推理方式共同形成的优势区域。

## 8.24 常见误区

### 误区 1：attention mask 和 loss mask 是一回事

不是。attention mask 控制能看谁，loss mask 控制哪里算 loss。

### 误区 2：Causal LM 训练是一个 token 一个 token 串行训练

不是。训练时整段序列并行输入，通过 causal mask 防止看未来。

### 误区 3：BERT 双向看上下文，所以一定比 GPT 强

不对。BERT 的双向 attention 适合理解表征，GPT 的 causal LM 适合自回归生成和统一任务接口。强弱取决于任务、规模、数据和训练方式。

### 误区 4：Prefix LM 等同于普通 causal LM

不等同。Prefix LM 的 prefix 内部可以双向可见，而普通 causal LM 从第一个 token 到最后一个 token 都只能看过去。

### 误区 5：只要 mask 数学上对，工程上就高效

不一定。复杂 mask 可能无法使用高效 attention kernel，导致速度和显存退化。

### 误区 6：packed sequence 只要拼起来就能训练

不行。必须处理样本边界，否则不同样本之间会信息泄漏。

## 8.25 面试高频问题

### 题 1：为什么 decoder-only LLM 需要 causal mask？

参考回答：

```text
Decoder-only LLM 使用 next-token prediction。训练时整段序列并行输入，如果没有 causal mask，当前位置可能通过 self-attention 看到未来 token，导致信息泄漏和 loss 虚低。Causal mask 保证第 i 个位置只能看自己和过去位置，使训练时可用信息和推理时一致。
```

### 题 2：attention mask、padding mask、loss mask 有什么区别？

参考回答：

```text
Attention mask 控制 token 之间能不能互相看，padding mask 标记哪些位置是补齐 token，loss mask 控制哪些 token 参与 loss。SFT 中 user token 通常不参与 loss，但 assistant 必须能通过 attention 看见 user token，所以不能把 loss mask 和 attention mask 混为一谈。
```

### 题 3：Bidirectional attention 为什么适合理解任务？

参考回答：

```text
Bidirectional attention 允许每个 token 同时利用左侧和右侧上下文，因此能形成更完整的上下文表征。分类、抽取、阅读理解、句子匹配等任务通常需要综合整段输入信息，所以 BERT 这类 encoder-only 双向模型很适合理解任务。
```

### 题 4：Encoder-only、decoder-only、encoder-decoder 的 attention mask 有什么区别？

参考回答：

```text
Encoder-only 通常使用双向 self-attention，所有非 padding token 互相可见。Decoder-only 使用 causal self-attention，只能看过去和当前位置。Encoder-decoder 中 encoder 是双向 self-attention，decoder 是 causal self-attention，同时 decoder 通过 cross-attention 读取 encoder 输出。
```

### 题 5：Prefix LM 的 mask 是什么样的？

参考回答：

```text
Prefix LM 把序列分为 prefix 和 target。Prefix 内部 token 可以双向可见；target token 可以看全部 prefix；target 内部只能 causal 可见；prefix 通常不能看 target。它适合条件生成，把输入和输出放在同一个 self-attention 序列中，通过 mask 控制可见性。
```

### 题 6：packed sequence training 中 mask 为什么重要？

参考回答：

```text
Packing 会把多条短样本拼成一个长序列。如果只用普通 causal mask，后面的样本可以看到前面样本的 token，造成跨样本信息泄漏。正确做法是在每个样本内部使用 causal mask，不同样本之间完全不可见，也就是 block-diagonal causal mask。
```

### 题 7：为什么复杂 attention mask 可能影响性能？

参考回答：

```text
高效 attention kernel 通常只针对常见模式优化，例如 causal、无 mask、padding varlen 或 sliding window。如果 mask 是任意 n x n 形状，可能无法使用 FlashAttention 等高效 kernel，导致退化到普通 attention 或需要 block-sparse kernel。因此 attention pattern 设计要考虑硬件和 kernel 支持。
```

### 题 8：KV cache 下 causal mask 有什么变化？

参考回答：

```text
训练时 query_len 和 key_len 通常相等，可以用标准下三角 causal mask。KV cache 推理时，新 token 的 query_len 可能是 1，key_len 是历史长度加当前 token。此时新 query 应该能看全部历史 K/V 和当前 K。prefill、decode、chunked prefill、不同样本 past length 不一致时，都要保证 mask、cache position 和 position id 对齐。
```

## 8.26 小练习

1. 画出长度为 6 的 causal mask 矩阵。
2. 画出 prefix 长度为 3、target 长度为 3 的 Prefix LM mask。
3. 用 PyTorch 实现一个 `make_causal_mask(seq_len)`。
4. 解释为什么 SFT 中 user token 不参与 loss，但 assistant token 仍要能看见 user token。
5. 设计一个 packed sequence 的 block-diagonal causal mask。
6. 比较 BERT、GPT、T5/BART 的 attention 可见性。
7. 思考 sliding window attention 如何通过多层扩大感受野。
8. 排查一个 loss 很低但生成很差的模型，列出你会检查的 mask 和格式项。

## 8.27 本章总结

本章讲了 Causal Mask、Prefix LM、Bidirectional Attention 和注意力模式。

核心结论：

1. Self-Attention 本身不规定可见性，attention mask 决定每个 token 能看谁。
2. Causal mask 防止未来信息泄漏，是 decoder-only LLM 自回归训练的关键。
3. Bidirectional attention 允许左右上下文联合建模，适合理解和表征任务。
4. Encoder-only、decoder-only、encoder-decoder 的核心差异可以从 attention 可见性理解。
5. Prefix LM 让 prefix 双向可见、target 因果可见，是条件生成的一种统一 mask 形式。
6. Attention mask、padding mask、loss mask 是三件不同的事，混淆会导致严重训练事故。
7. Packed sequence、KV cache、FlashAttention 都要求 mask 逻辑和工程实现严格对齐。
8. Sparse/sliding window attention 通过改变可见性降低长上下文成本，但会改变信息传播路径。

下一章会进入 Transformer 的并行性、表达力和 scaling 优势，解释为什么 Transformer 能在大规模数据、参数和算力下表现出强扩展性。
