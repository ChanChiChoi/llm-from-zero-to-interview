# 第八章：Causal Mask、Prefix LM、Bidirectional Attention 与注意力模式

## 8.1 可见性规则改变信息流

第七章解释位置线索如何进入注意力计算；本章看另一个问题：每个 query 位置允许读取哪些 key/value 位置。内容相同，允许读取的范围不同，各位置形成的表示就可能不同。

矩阵 $S$ 用来表示 query-key 分数；可见性矩阵 $V$ 的元素取 0 或 1，用来标记允许和禁止的读取边。当 $V_{ij}=1$ 时，query 位置 $i$ 可以读取位置 $j$ 的 key；当 $V_{ij}=0$ 时，这条边被禁止。常见加性遮罩为：

```math
M_{ij}=
\begin{cases}
0, & V_{ij}=1,\\
-\infty, & V_{ij}=0,
\end{cases}
\qquad
A_{ij}=\mathrm{softmax}_j(S_{ij}+M_{ij}).
```

遮罩在 softmax 前把被禁止位置的分数移出竞争；softmax 后这些位置的权重为零。因而 attention mask 约束的是信息读取边，不会删除输入 token，也不决定哪些 token 参与训练损失。

例如，对于 query 位置 1，四个 key 的未遮罩分数向量 $[0,\ln 3,2,1]$ 按 key 顺序排列。因果规则只允许读取前两个位置；遮罩后的分数和归一化权重分别为：

```math
\begin{aligned}
S_{1,:}+M_{1,:}&=[0,\ln 3,-\infty,-\infty],\\
A_{1,:}&=[1/4,3/4,0,0].
\end{aligned}
```

被遮住的未来 key 不会影响该 query 的加权结果。

## 8.2 因果注意力与 next-token 对齐

自回归语言模型要预测下一个 token。训练时可以把一整段 token 一次送入模型，但位置 $i$ 的表示不能读取之后的位置。按输入 token 作为 query 的常见对齐方式，因果可见性是：

```math
V_{ij}^{\mathrm{causal}}=\mathbf{1}[j\le i].
```

这里允许读取当前位置本身。训练标签向右错开一位：模型在输入位置 $i$ 产生的 logit，以后续 token $x_{i+1}$ 为预测目标。因此模型能同时处理所有输入位置，又不会通过 attention 读取当前要预测的 token：

```math
\mathcal{L}_{\mathrm{next}}
=-\sum_{i=0}^{n-2}
\log p_\theta(x_{i+1}\mid x_0,\ldots,x_i).
```

以输入 $x_0,x_1,x_2,x_3$ 为例，位置 $1$ 的 query 可以读取前两个输入 token；该位置的输出以序列第三个 token $x_2$ 为预测目标。它不能读取该目标或之后的 token。Vaswani 等人的原始 Transformer decoder 同时使用右移目标输入与未来位置遮罩，保证每个预测只依赖已知前缀。[《Attention Is All You Need》，§§3.1、3.2.3](https://papers.neurips.cc/paper/7181-attention-isall-you-need.pdf)

有些图把允许范围画成严格下三角，有些图画成包含对角线的下三角。差异来自行表示的是输入位置，还是与目标词对齐的预测位置；读图时应连同 label shift 一起确认，不能只看三角形方向。

## 8.3 双向注意力与去噪目标

双向注意力让一个有效 token 读取同一序列中左右两侧的有效 token。它适合先结合整段输入形成表示，再预测被遮盖或经过扰动的内容。

BERT 使用双向 Transformer encoder，并通过 masked language modeling 预测部分被遮盖的 token。其目标只在被选中的位置计算预测损失，而未被遮盖的左右上下文都可以参与表示计算。[BERT，摘要与 §3.1](https://arxiv.org/html/1810.04805v2)

这说明“模型能读取哪些位置”和“哪些位置是预测目标”是两个问题。双向可见性本身既不会指定训练标签，也不会把模型变成标准的逐 token 自回归生成器。双向 encoder 可以服务于分类、抽取或去噪重建；具体任务效果取决于训练目标、数据和输出方式。

## 8.4 Prefix LM：前缀双向，目标因果

Prefix LM 将输入分成条件前缀和生成目标。前缀内部双向可见；目标位置可以读取完整前缀，以及目标中当前位置和之前的位置；前缀位置不读取目标。

用 $P$ 表示前缀长度。目标位置接在前缀之后，此时可见性为：

```math
V_{ij}^{\mathrm{prefix}}=
\begin{cases}
1, & i<P,\ j<P,\\
1, & i\ge P,\ j<P,\\
1, & i\ge P,\ P\le j\le i,\\
0, & \text{otherwise}.
\end{cases}
```

前缀长度与目标长度均为 $2$ 时，可见矩阵如下；行是 query，列是 key：

|  | $k_0$ | $k_1$ | $k_2$ | $k_3$ |
| --- | ---: | ---: | ---: | ---: |
| $q_0$ | 1 | 1 | 0 | 0 |
| $q_1$ | 1 | 1 | 0 | 0 |
| $q_2$ | 1 | 1 | 1 | 0 |
| $q_3$ | 1 | 1 | 1 | 1 |

Dong 等人的 UniLM 用不同 self-attention mask 表达单向、双向和 sequence-to-sequence 可见性。在其 Seq2Seq 模式中，source 内部双向可见，target 可读完整 source 和目标前缀；但 UniLM 的训练还包括 cloze token corruption。因此，Prefix/Seq2Seq 可见性矩阵本身不等于某一种固定的标签或 loss 方案。[UniLM，§2 与 Figure 1](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf)

## 8.5 四种常见结构形态与注意力模式

| 结构形态 | Self-attention 可见性 | 是否另有跨序列读取 | 代表性原始实例 |
| --- | --- | --- | --- |
| Encoder-only | 输入序列中的有效 token 双向读取 | 无 | [BERT](https://arxiv.org/html/1810.04805v2) |
| Decoder-only | 同一 token stream 按因果顺序读取 | 无独立 cross-attention | [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| Encoder-decoder | Encoder 对 source 双向读取；decoder 对 target 因果读取 | Decoder 可读取 encoder 输出 | [原始 Transformer](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)、[T5](https://arxiv.org/abs/1910.10683)、[BART](https://aclanthology.org/2020.acl-main.703.pdf) |
| Prefix-LM 单栈 | 前缀内部双向；目标读取完整前缀和目标前缀 | 无独立 cross-attention | [UniLM 的 Seq2Seq 可见性方案](https://proceedings.neurips.cc/paper/2019/file/c20bb2d9a50d5ac1f713f8b34d9aac5a-Paper.pdf) |

Encoder-decoder 把 source 表示和 target 生成分到两条栈；Prefix LM 则在同一栈中用 self-attention 规则区分条件前缀和生成目标。GPT-1 的原始报告描述了基于 Transformer decoder 的语言模型与前序上下文目标；这里引用它作为 decoder-only 实例，不把其具体目标推广为所有 GPT 版本的训练配方。[GPT-1，§3.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

BART 展示双向 encoder 与自回归 decoder 组合成去噪 seq2seq 模型的例子。[BART，摘要、§1 与 Figure 1](https://aclanthology.org/2020.acl-main.703.pdf)

这些模式描述信息流，不足以单独确定模型能力或性能。架构、attention 可见性、训练目标、数据和推理过程共同决定模型实际学到什么。UL2 将多种 denoising 目标组合到同一预训练框架，也说明预训练目标不应与模型架构或单一 mask 形式混为一谈。[UL2，摘要与 §§2–3](https://arxiv.org/html/2205.05131v1)

## 8.6 Attention、padding、loss 与 position 是不同约束

实际训练数据可能同时带有注意力可见性、padding 标记、监督标签和位置编号。它们作用于不同对象：

| 约束 | 它回答的问题 | 典型作用 |
| --- | --- | --- |
| Attention visibility | 某个 query/key 位置对之间是否允许读取？ | 因果、双向、prefix 或局部窗口关系 |
| Padding validity | 哪些位置只是为了凑齐 batch 而补出的？ | 避免真实 query 读取 padding key；padding 目标通常也不计 loss |
| Label/loss mask | 哪些目标位置参与训练目标？ | 例如某些指令微调设置只监督 assistant 回复 |
| Segment boundary | 哪些 token 属于同一独立训练样本？ | 独立样本打包时阻止跨样本 attention |
| Position ID | 每个 token 使用哪个位置坐标？ | 位置表示的编号；不代替 attention 或 label mask |

例如，assistant 回复可以读取 system/user prompt，但训练目标可以只覆盖回复 token；prompt 可见不等于 prompt 参与 loss。padding 位置需要按 batch 和 API 合同处理。位置编号则由第 7 章讨论的架构与数据约定决定，即使位置编号正确，也不会自动隔离不同样本。

“attention mask”在框架 API 中还可能指 key-validity 张量，而不一定是完整的 query-key 可见矩阵。实际调用时必须核对 API 定义和布尔值极性，不能只凭变量名称推断其含义。

## 8.7 Packed sequence：先确定样本边界语义

Packing 把多条短序列放进同一个 token buffer，可以减少 padding。是否需要在样本之间隔离，取决于训练目标：

- **独立样本语义：** 每条样本应像单独输入模型一样处理。此时每个样本内部使用所需的 attention pattern，样本之间不可见。
- **连续文本流语义：** 数据目标把文本拼成一条连续 stream，允许后续 token 条件于之前内容。此时跨文档读取可能是目标的一部分，不应机械套用独立样本的块对角 mask。

下面是两个独立样本的 block-diagonal causal visibility。样本 A 有两个 token，样本 B 有三个 token：

|  | $A_0$ | $A_1$ | $B_0$ | $B_1$ | $B_2$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| $q_{A_0}$ | 1 | 0 | 0 | 0 | 0 |
| $q_{A_1}$ | 1 | 1 | 0 | 0 | 0 |
| $q_{B_0}$ | 0 | 0 | 1 | 0 | 0 |
| $q_{B_1}$ | 0 | 0 | 1 | 1 | 0 |
| $q_{B_2}$ | 0 | 0 | 1 | 1 | 1 |

如果要求样本互相独立，除 attention 边界外，还要核对 next-token 标签如何跨边界对齐：不能让上一条样本末尾的输出意外成为下一条样本首个目标的上下文。若目标是连续 stream，这种转移则可能是有意的。EOS 等分隔 token 只提供序列内容，除非 attention 规则另作限制，否则它本身不会阻止跨边界读取。

PyTorch 2.15 的变长 attention 教程用累积序列边界标记 packed tensor 中各文档的逻辑范围；具体 kernel/API 约定会随实现变化。[PyTorch 2.15 变长 attention 教程](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)

## 8.8 Local 与 sparse attention 改变直接读取范围

全量注意力允许每个有效 query 读取所有允许的历史位置。局部因果窗口将每行限制在最近 $w$ 个位置（包括当前位置）：

```math
V_{ij}^{\mathrm{window}}=\mathbf{1}[0\le i-j<w].
```

当 $w=3$ 时，序列较早位置的可见数量逐步增加，窗口填满后每个 query 直接读取最近三个位置。多层堆叠能让信息逐层向更远位置传播，但这不等于每层都能直接读取全序列，也不保证长距离内容会被保留或正确使用。

Mistral 7B v0.1 报告的架构表列出 `window_size=4096` 与 `context_len=8192`，并说明信息可通过多层传播到单层窗口之外。这是该论文版本的配置和机制解释，不能外推成所有版本或任意任务的有效上下文结论。[Mistral 7B，§2](https://arxiv.org/html/2310.06825v1)

局部、稀疏和全局连接之间的复杂度与设计取舍见[第 10 章](10-o-n2-attention的计算与显存瓶颈.md)、[第 19 章](19-Sliding-Window与稀疏注意力.md)及[第 26 章](26-稀疏与块稀疏Attention.md)。

## 8.9 实现时核对 API、位置偏移和有效行

不可见位置常加上数值为 $-\infty$ 的遮罩。如果一个有效 query 的所有 key 都被屏蔽，softmax 输入就会全部取这个值，无法形成可归一化的分布；应由数据与遮罩规则保证有效行至少有可见 key，或使用实现明确支持的安全处理。

布尔遮罩的极性属于具体 API。以 PyTorch 2.15 为例，`torch.nn.Transformer` 的布尔 mask 中 `True` 表示禁止参与；`torch.nn.functional.scaled_dot_product_attention` 中 `True` 表示允许参与。两者相反，不能把其中一个接口的遮罩直接传给另一个。[PyTorch 2.15 Transformer API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Transformer.html)；[PyTorch 2.15 SDPA API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)

带 KV cache 的 decode 可能只有一个新 query，却对应许多历史 key；chunked prefill、滑动窗口或 prefix cache 还会改变绝对位置偏移。此时应依据 query/key 的实际位置判断可见关系，不要把等长训练时的三角矩阵直接套到任意矩形分数矩阵。缓存细节见[第 11 章](11-kv-cache长上下文推理和显存增长问题.md)，kernel 的能力与代价见[第 20 章](20-FlashAttention与PagedAttention.md)。

## 8.10 自测：从一条可见边追到训练目标

1. 在输入位置 $i$ 预测后一个 token $x_{i+1}$ 时，因果 attention 为什么仍可读取当前位置？右移标签解决了什么问题？
2. 令前缀长度和目标长度都取 2，写出 Prefix LM 的四行可见性，并说明 Prefix/Seq2Seq mask 本身没有规定什么？
3. 为什么一个 prompt token 可以被模型读取，却不计入 SFT loss？position ID 能否代替 attention mask？
4. 两条 packed 样本若按独立序列处理，哪些 attention 边和 shifted-label 转移需要检查？若训练目标是连续文本流，决定会怎样变化？
5. 局部窗口堆叠为什么能传播较远信息？为什么这不能推出它与全量注意力等价？

因果、双向、Prefix LM、encoder-decoder 和局部窗口定义了不同的信息读取范围；padding、loss 和样本边界则约束不同的数据或训练对象。分析一个具体模型时，应分别核对结构、可见性、监督目标和实现合同。
