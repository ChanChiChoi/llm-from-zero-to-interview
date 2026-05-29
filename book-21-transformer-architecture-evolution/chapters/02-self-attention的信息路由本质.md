# 第二章：Self-Attention 的信息路由本质

## 2.1 本章定位

上一章讲了 Transformer 为什么能成为大模型基础架构。本章进入 Transformer 的核心：Self-Attention。

很多人会背公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

但如果只会背公式，面试官继续追问就容易卡住：

1. Q、K、V 为什么要分开？
2. Attention score 到底表示什么？
3. 为什么要除以 `sqrt(d_k)`？
4. Softmax 后的权重能不能解释成“重要性”？
5. Self-Attention 为什么是一种信息路由机制？
6. 它和 RNN 的状态传递、CNN 的局部卷积有什么根本差别？
7. 为什么 attention 强，但长上下文仍然会 lost-in-the-middle？

本章目标是把 Self-Attention 从“公式模块”讲成“信息流系统”。学完后，你应该能从直觉、数学、代码、工程和面试五个角度解释它。

## 2.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。原论文定义 scaled dot-product attention 和 multi-head attention，指出 attention 可以让模型关注不同位置的信息，并替代 recurrence/convolution。
2. Devlin et al., 2018/2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*。BERT 展示了 encoder self-attention 如何联合条件化左右上下文，形成深层双向表示。
3. Michel et al., 2019, *Are Sixteen Heads Really Better than One?*。分析多头注意力中部分 head 可被裁剪，提醒我们不要把每个 attention head 都过度解释成稳定语义专家。
4. Tay et al., 2020/2022, *Efficient Transformers: A Survey*。系统梳理围绕 attention 计算和显存效率的改造路线。
5. Liu et al., 2023, *Lost in the Middle*。指出长上下文模型虽然能接收长输入，但对中间位置相关信息的利用不一定鲁棒。
6. Jiang et al., 2023, *Mistral 7B*。公开强调 GQA 和 Sliding Window Attention，用于提升推理效率和降低长序列成本。
7. Dao, 2023, *FlashAttention-2*。说明 attention 是长序列扩展的主要瓶颈之一，并从 GPU 并行和 work partitioning 角度优化 attention kernel。

本章会强调一个边界：attention weight 有解释价值，但不能被简单等同于“模型真正推理原因”。Attention 是信息混合权重，不是完整因果解释。

## 2.3 Self-Attention 解决的核心问题

序列模型的核心问题是：

```text
每个位置应该从哪些其他位置读取信息？读取多少？读到的信息如何汇总？
```

举一个句子：

```text
The animal didn't cross the street because it was too tired.
```

这里 `it` 指代谁？更可能是 `animal`，不是 `street`。

模型在处理 `it` 时，需要从前文中找到相关实体，并结合语义判断。

RNN 的方式是：

```text
把前文压缩到 hidden state，再一步步传过来。
```

CNN 的方式是：

```text
用局部窗口和多层堆叠扩大感受野。
```

Self-Attention 的方式是：

```text
当前位置直接向所有可见位置发起查询，根据匹配程度加权读取它们的信息。
```

这就是信息路由：

1. 谁需要信息。
2. 谁提供信息。
3. 谁和谁匹配。
4. 信息按什么权重流动。
5. 多个来源如何汇总成新的表示。

Self-Attention 不是简单“看所有 token”，而是为每个 token 动态构造一条从其他 token 汇聚信息的路由表。

## 2.4 Q、K、V 的角色直觉

Q、K、V 来自同一个输入表示 `X` 的三个线性投影：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

可以用信息检索类比：

1. Query：当前 token 发出的查询。
2. Key：每个 token 暴露出来的索引。
3. Value：每个 token 真正提供的内容。

为什么不直接用同一个向量？

因为“用什么来匹配”和“提供什么内容”不一定是同一件事。

举例：

```text
The animal didn't cross the street because it was too tired.
```

`it` 的 Query 可能在寻找：

```text
一个前文实体，能解释 tired 这个状态。
```

`animal` 的 Key 可能表示：

```text
我是一个可被代词指代的生物实体。
```

`animal` 的 Value 可能包含：

```text
animal 的语义、数、位置、上下文信息。
```

所以 Q/K/V 分开后，模型可以学到不同子空间：

1. 用 Q/K 做匹配。
2. 用 V 做信息传递。
3. 匹配逻辑和内容表示可以不同。

这是 attention 的一个核心设计。

## 2.5 Attention Score 是什么

Attention score 通常是点积：

```text
score(i, j) = q_i dot k_j
```

它表示位置 `i` 的 Query 和位置 `j` 的 Key 的匹配程度。

如果 `q_i` 和 `k_j` 方向相近，点积大，说明位置 `i` 更应该读取位置 `j` 的信息。

对一个长度为 `n` 的序列，所有 token 两两计算 score，得到一个 `n x n` 矩阵：

```text
S = Q K^T
```

第 `i` 行表示：

```text
第 i 个 token 对所有 token 的关注打分。
```

第 `j` 列表示：

```text
第 j 个 token 被其他 token 关注的程度。
```

这张矩阵就是一张动态信息路由图。

每个 forward pass，每一层，每个 head 都会根据当前输入重新生成不同的路由图。

这和固定卷积核很不一样。卷积的局部连接模式基本固定，而 attention 的连接权重由内容决定。

## 2.6 为什么要除以 sqrt(d_k)

Scaled dot-product attention 会计算：

```text
softmax(QK^T / sqrt(d_k))
```

为什么要除以 `sqrt(d_k)`？

直觉是：如果向量维度 `d_k` 很大，随机向量点积的方差会变大，score 数值可能很大。

Score 很大时，softmax 会变得非常尖锐：

```text
一个位置接近 1，其他位置接近 0
```

这会带来两个问题：

1. 梯度变小，训练不稳定。
2. 早期训练时 attention 过早变成硬选择。

除以 `sqrt(d_k)` 是为了控制 score 的尺度，让 softmax 落在更合适的梯度区间。

面试中可以这样说：

```text
缩放不是为了改变匹配关系，而是为了控制点积分布的方差，避免维度变大后 softmax 饱和。
```

## 2.7 Softmax 权重是不是“重要性”

Softmax 后得到 attention weights：

```text
a_ij = softmax(score(i, j))
```

然后输出：

```text
o_i = sum_j a_ij v_j
```

从计算角度，`a_ij` 确实表示位置 `j` 的 Value 对位置 `i` 输出的加权贡献。

但要小心：attention weight 不等于完整解释。

原因有几个：

1. Value 向量本身已经是复杂表示，不是原始 token。
2. 后面还有残差、FFN、LayerNorm、多层堆叠。
3. 多个 head 会并行混合信息。
4. 注意力权重可被参数重排影响。
5. 有些 head 可以裁剪而不明显影响性能。

Michel 等人的多头裁剪研究提醒我们：不是每个 head 都稳定承担可解释语义角色。某些 head 可能冗余，某些层更依赖多头。

所以更准确的表述是：

```text
Attention weight 是当前 head 当前层的信息混合权重，有局部解释价值，但不能直接等同于模型最终决策原因。
```

## 2.8 Self-Attention 的矩阵形状

假设：

```text
batch size = B
sequence length = N
hidden size = D
num heads = H
head dim = Dh = D / H
```

输入：

```text
X: [B, N, D]
```

投影后：

```text
Q: [B, H, N, Dh]
K: [B, H, N, Dh]
V: [B, H, N, Dh]
```

Attention score：

```text
QK^T: [B, H, N, N]
```

Attention output：

```text
Attn @ V: [B, H, N, Dh]
```

合并 heads 后：

```text
Output: [B, N, D]
```

这里最关键的是 `[N, N]`。

它说明每个 head 都构造了一张 token-to-token 的关系图。这也是 attention 强大的来源，也是 O(n^2) 瓶颈的来源。

## 2.9 一个最小 PyTorch 实现

下面是一个简化的单头 causal self-attention。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.out_proj = nn.Linear(head_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)  # [batch, seq_len, head_dim]
        k = self.k_proj(x)  # [batch, seq_len, head_dim]
        v = self.v_proj(x)  # [batch, seq_len, head_dim]

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.shape[-1])

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        return self.out_proj(out)
```

这段代码对应的核心链路是：

```text
x -> q/k/v -> score -> mask -> softmax -> weighted sum of v -> output projection
```

如果是 encoder self-attention，就不使用 causal mask，所有 token 可以双向互相看。BERT 就是典型的双向 Transformer encoder。

## 2.10 Causal Attention 和 Bidirectional Attention

Self-Attention 本身不规定能看哪些位置。能看哪些位置由 mask 决定。

Decoder-only LLM 使用 causal mask：

```text
当前位置只能看自己和过去 token，不能看未来 token。
```

这适合 next-token prediction：

```text
p(x_t | x_<t)
```

Encoder-only 模型如 BERT 使用 bidirectional attention：

```text
每个 token 可以同时看左边和右边。
```

这适合理解任务，比如分类、抽取、问答表示学习。

区别可以总结为：

| 类型 | 可见范围 | 典型模型 | 适合任务 |
|---|---|---|---|
| Causal self-attention | 只能看过去 | GPT、LLaMA、Qwen | 生成、对话、代码补全 |
| Bidirectional self-attention | 左右都能看 | BERT、RoBERTa | 理解、分类、抽取 |
| Encoder-decoder attention | decoder 看 encoder 输出 | T5、原始 Transformer 翻译 | seq2seq |

所以“Self-Attention”不等于“GPT 注意力”。GPT 的关键是 causal self-attention。

## 2.11 信息路由图：每层都在重写 token 表示

Self-Attention 的输出不是最终答案，而是新的 token 表示。

第 `l` 层输入：

```text
X_l = [x_1, x_2, ..., x_n]
```

经过 attention 后：

```text
x_i' = sum_j a_ij v_j
```

这意味着第 `i` 个 token 的表示被其他 token 的信息重写。

多层堆叠后，一个 token 的表示会逐步聚合越来越复杂的上下文信息。

例如在代码中：

```python
result = normalize_scores(values)
```

`result` 这个 token 的高层表示可能融合了：

1. 函数定义。
2. 参数类型。
3. 返回值含义。
4. 当前作用域。
5. 上下文中的变量命名习惯。

这就是为什么 Transformer 的每一层都不是简单“抽特征”，而是在重新路由和组合上下文信息。

## 2.12 Attention 和检索的相似与不同

Attention 很像检索：

```text
query -> match keys -> retrieve values
```

但它和传统向量检索不同。

相似点：

1. 都有 query。
2. 都根据相似度找相关内容。
3. 都返回某种 value 或文档表示。

不同点：

1. Attention 的 K/V 来自当前上下文内部，不是外部索引库。
2. Attention 是可微分的，会端到端训练。
3. Attention 每层每头动态生成路由。
4. Attention 输出是加权混合，而不是简单 top-k 文档。
5. Attention 受上下文长度限制。

RAG 可以看作把外部检索和内部 attention 结合：

```text
external retriever 找文档
-> 文档放入 prompt
-> Transformer attention 在上下文内读取文档
```

因此，Transformer 的 context window 是一种内部工作记忆，而 RAG 是外部记忆补充。

## 2.13 为什么长上下文仍然会失败

既然 attention 可以看所有 token，为什么长上下文模型还会漏掉中间信息？

原因是“可见”不等于“会用”。

Lost in the Middle 的研究发现，模型在多文档问答和 key-value retrieval 中，相关信息放在上下文开头或结尾时表现更好，放在中间时性能可能下降。

这说明：

1. Attention 矩阵允许访问中间 token。
2. 但模型训练分布、位置编码、注意力模式和生成策略可能让它更偏向首尾信息。
3. 长上下文中的干扰信息会竞争 attention 权重。
4. 深层表示不一定保留所有细节。

所以长上下文能力要分开看：

```text
能接收多长输入 != 能稳定利用所有位置的信息
```

这也是为什么后续会有长上下文评估、retrieval-aware training、context compression、memory、global attention、sliding window + global token 等路线。

## 2.14 Attention 的计算瓶颈

Self-Attention 强在 `N x N` 路由图，也贵在 `N x N` 路由图。

训练时，score 矩阵大小是：

```text
[batch, heads, seq_len, seq_len]
```

当 `seq_len` 翻倍，attention score 的元素数量约变成 4 倍。

FlashAttention-2 摘要里明确指出，attention layer 是扩展到更长序列时的主要瓶颈，运行时间和内存随序列长度平方增长。

高效 attention 路线通常做几类事情：

1. 不近似 attention，但优化 IO 和 kernel，例如 FlashAttention。
2. 改 attention pattern，例如 sliding window、block sparse、global tokens。
3. 近似 attention，例如 low-rank、kernel attention、linear attention。
4. 减少 KV cache，例如 MQA、GQA、MLA。
5. 用其他序列模型替代部分 attention，例如 SSM、Mamba、Hyena。

Mistral 7B 使用 GQA 和 Sliding Window Attention，就是在保持 Transformer 主干的同时降低推理成本和长序列成本。

## 2.15 Self-Attention 的常见误区

误区一：Attention 权重就是因果解释。

更准确：attention weight 是局部信息混合权重，不是完整因果解释。

误区二：Attention 能看全局，所以长上下文一定可靠。

更准确：可见范围只是上限，模型是否能稳定利用，还取决于训练、位置、干扰和评估。

误区三：多头越多越好。

更准确：多头提供多个子空间和路由模式，但部分 head 可能冗余，head 数需要和模型规模、任务、推理成本一起考虑。

误区四：Self-Attention 只是在做相似度检索。

更准确：它像可微分检索，但 Q/K/V、残差、FFN、多层堆叠让它远比一次相似度搜索复杂。

误区五：Attention 的 O(n^2) 只是训练问题。

更准确：训练有 score 矩阵成本，推理还有 KV cache 成本。长上下文 serving 时 KV cache 往往是核心瓶颈。

## 2.16 面试题

### 题 1：Self-Attention 的本质是什么？

参考回答：

```text
Self-Attention 的本质是内容相关的信息路由。每个 token 通过 Query 表示自己想找什么信息，其他 token 通过 Key 表示自己能被如何匹配，通过 Value 提供实际内容。模型计算 Q 和 K 的匹配分数，softmax 后得到从其他 token 读取信息的权重，再对 V 做加权求和。这样每个 token 都能根据当前上下文动态选择从哪些位置聚合信息。
```

### 题 2：为什么 Q、K、V 要分开？

参考回答：

```text
因为匹配逻辑和传递内容不一定相同。Q/K 用于判断当前位置应该关注哪些位置，V 表示被关注位置实际提供的信息。把它们分开后，模型可以在不同子空间里学习查询、索引和内容表达。如果只用同一个向量，匹配和内容会被绑死，表达能力更弱。
```

### 题 3：为什么 attention score 要除以 sqrt(d_k)？

参考回答：

```text
点积的方差会随维度增大而增大。如果不缩放，QK 的数值可能过大，softmax 会进入饱和区，变得过于尖锐，梯度也会变小。除以 sqrt(d_k) 是为了控制 score 尺度，让训练更稳定。
```

### 题 4：Attention weight 能解释模型行为吗？

参考回答：

```text
只能有限解释。Attention weight 表示某一层某个 head 中，当前 token 对其他 token 的 Value 加权混合程度，有局部解释价值。但模型还有多头、多层、残差、FFN 和后续非线性，最终输出不只由某个 attention map 决定。而且研究发现部分 attention head 可以裁剪而性能影响不大，所以不能简单把 attention weight 等同于因果解释。
```

### 题 5：为什么 attention 可以看全局，但长上下文仍会 lost-in-the-middle？

参考回答：

```text
因为可见不等于会稳定使用。标准 attention 让 token 理论上能访问上下文中所有位置，但模型训练分布、位置编码、注意力竞争、干扰信息和生成偏置会影响实际利用。Lost in the Middle 发现相关信息在上下文中间时模型性能可能下降，说明长上下文要评估的是有效利用能力，而不只是最大输入长度。
```

### 题 6：Self-Attention 和 RAG 中的检索有什么区别？

参考回答：

```text
Self-Attention 是上下文内部的可微分检索，Q/K/V 都来自当前输入，输出是对 Value 的加权混合。RAG 的检索通常是外部检索，从向量库或搜索系统里找文档，再把文档放进上下文。可以说 RAG 先扩展外部记忆，Transformer attention 再在扩展后的上下文里做内部信息路由。
```

## 2.17 小练习

1. 手写一个单头 self-attention 的形状推导，从 `[B, N, D]` 到 `[B, N, D]`。
2. 用自己的话解释 Q、K、V，不允许只说“查询、键、值”。
3. 构造一句包含代词指代的英文句子，说明 attention 如何帮助代词找到 antecedent。
4. 解释为什么 `QK^T` 是 `N x N`，以及这个矩阵为什么既强大又昂贵。
5. 阅读 BERT 摘要，解释 bidirectional attention 和 causal attention 的任务差异。
6. 阅读 Lost in the Middle 摘要，解释“能放进上下文”和“能用好上下文”的区别。
7. 用 PyTorch 写一个 multi-head causal self-attention，并打印每个中间张量形状。

## 2.18 本章总结

本章从信息路由角度解释了 Self-Attention。

核心结论：

1. Self-Attention 的本质是让每个 token 动态选择从哪些 token 读取信息。
2. Q 表示查询，K 表示匹配索引，V 表示被传递内容。
3. `QK^T` 形成 token-to-token 的动态路由图。
4. 除以 `sqrt(d_k)` 是为了控制点积分布尺度，避免 softmax 饱和。
5. Attention weight 有局部解释价值，但不是完整因果解释。
6. Causal attention 适合自回归生成，bidirectional attention 适合理解任务。
7. Self-Attention 像上下文内部的可微分检索，但不同于外部 RAG 检索。
8. Attention 的全局可见性不保证长上下文鲁棒利用，lost-in-the-middle 是重要现象。
9. Attention 的 `N x N` 路由图同时带来强表达力和 O(n^2) 计算/显存瓶颈。

下一章会进一步拆解 Q、K、V、Attention Head 与表示子空间，重点讲多头为什么有用、head 之间是否真的学到不同功能，以及 head 维度如何影响表达和效率。
