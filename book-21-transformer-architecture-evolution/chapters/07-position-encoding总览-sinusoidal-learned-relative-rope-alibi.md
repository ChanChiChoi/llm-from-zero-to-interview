# 第七章：Position Encoding 总览：Sinusoidal、Learned、Relative、RoPE、ALiBi

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修主要对齐 Transformer 原论文、Shaw et al. relative position representation、T5 relative position bias、RoFormer / RoPE、ALiBi、Position Interpolation 和 YaRN 等公开论文。写作时只把这些资料中已经公开且可复核的机制写成确定结论；不同模型里的 RoPE base、scaling 配置、position id 细节、长上下文微调配方和评测结论不能互相泛化。

本讲的精修重点有三点：

1. 把位置编码的核心公式改成 GitHub Markdown 更稳定的 fenced math block，避免把代码变量名塞进 LaTeX 的 text 宏。
2. 补一个 0 依赖 Python demo，用同一段代码检查 sinusoidal、RoPE、ALiBi、Position Interpolation 和 position id 风险。
3. 把本讲新增的 Position Encoding Gate、Position ID Audit、RoPE Scaling 等概念同步到百科、题库、练习、术语表、项目清单和知识图谱。

## 7.1 本章定位

前面几章已经讲过 attention、FFN、residual 和 normalization。现在进入 Transformer 架构里另一个非常关键、也非常容易被面试追问的主题：位置编码。

Self-Attention 的核心操作是让每个 token 根据 Q/K/V 和其他 token 交互。这个机制本身并不知道 token 出现在第几个位置。如果不给模型额外的位置信息，下面两个序列在 attention 眼里很容易变成“同一批 token 的不同排列”：

```text
我 喜欢 机器 学习
机器 学习 喜欢 我
```

但自然语言、代码、数学公式、对话、多模态序列都强依赖顺序。位置编码要解决的问题就是：

```text
如何把 token 的顺序、距离和相对关系注入 Transformer，同时不破坏 attention 的并行性和可扩展性。
```

本章要回答的问题是：

1. 为什么 Self-Attention 需要位置编码。
2. Sinusoidal position encoding 为什么用 sin/cos。
3. learned absolute position embedding 有什么优缺点。
4. relative position representation 为什么比绝对位置更贴近 attention。
5. RoPE 如何把位置编码成 Q/K 的旋转。
6. ALiBi 为什么可以不用显式 position embedding。
7. 位置编码如何影响长上下文扩展。
8. 现代 LLM 为什么大量采用 RoPE，并围绕 RoPE 做 scaling、interpolation 和 YaRN 等扩展。

## 7.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 Transformer，并使用 sinusoidal positional encoding，也讨论 learned positional embedding 的相近效果。
2. Shaw et al., 2018, *Self-Attention with Relative Position Representations*。提出在 self-attention 中显式引入相对位置表示。
3. Huang et al., 2018, *Music Transformer*。改进相对位置 attention 的内存实现，使长序列音乐建模更可行。
4. Dai et al., 2019, *Transformer-XL*。使用相对位置编码和 segment-level recurrence，影响后续长上下文建模。
5. Raffel et al., 2020, *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*。T5 使用 relative position bias，简化相对位置信息注入方式。
6. Su et al., 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding*。提出 RoPE，把绝对位置编码成旋转矩阵，同时在 attention 中体现相对位置关系。
7. Press et al., 2021, *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*。提出 ALiBi，在 attention score 上加入与距离成比例的线性 bias。
8. Touvron et al., 2023, *LLaMA: Open and Efficient Foundation Language Models*。LLaMA 使用 RoPE，推动 RoPE 成为开源 LLM 常见配置。
9. Chen et al., 2023, *Extending Context Window of Large Language Models via Positional Interpolation*。提出通过位置插值扩展 RoPE 模型上下文窗口。
10. Peng et al., 2023, *YaRN: Efficient Context Window Extension of Large Language Models*。提出更高效的 RoPE 上下文扩展方法。

需要说明的是，位置编码是一个仍在演进的领域。不同模型、不同数据、不同上下文长度、不同 attention pattern 下，最佳选择可能不同。本章重点讲清主流方法的机制和工程 trade-off，不把某一种方法绝对化。

## 7.3 为什么 Self-Attention 本身不知道顺序

先看一个没有位置编码的 self-attention。

设输入 token embedding 组成矩阵：

```math
X=[x_1,x_2,\ldots,x_n]\in\mathbb{R}^{n\times d}
```

attention 计算可以写成：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

```math
\mathrm{Attention}(X)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

如果把输入 token 的顺序整体打乱，只要 Q/K/V 也跟着同样打乱，attention 输出也会按同样方式打乱。也就是说，self-attention 对序列顺序没有内置的先后概念。

更直观地说，attention 只会问：

```text
这个 token 和那个 token 的内容相似吗？应该从那个 token 读取多少信息？
```

它不会天然知道：

```text
那个 token 在我前面还是后面？
距离我 1 个位置还是 100 个位置？
这是句首、句中还是句尾？
```

所以 Transformer 必须通过某种方式注入位置。

常见注入位置的方式有三类：

1. 加到输入表示上，例如 sinusoidal 和 learned absolute position embedding。
2. 加到 attention score 或 attention value 上，例如 relative position representation、relative position bias、ALiBi。
3. 作用到 Q/K 几何结构上，例如 RoPE。

这三类方式的差异，决定了它们在外推、长上下文、KV cache、实现复杂度上的不同表现。

## 7.4 绝对位置编码：把“第几个位置”加到 token 上

最直接的方法是为每个位置准备一个向量，然后把它加到 token embedding 上：

```math
h_i=e_i+p_i
```

其中 `e_i` 是第 `i` 个 token 的内容 embedding，`p_i` 是第 `i` 个位置的位置向量，`h_i` 是送入 Transformer block 的表示。

这样第 `i` 个 token 的输入表示里同时包含两部分信息：

```text
这个 token 是什么
这个 token 在第几个位置
```

绝对位置编码的优点是简单。Transformer 原论文就是这样做的。

但它也有一个天然问题：attention 里真正重要的往往不只是绝对位置，而是相对关系。

例如语言模型预测下一个 token 时，经常需要知道：

```text
前一个词是什么？
前 5 个 token 里有没有主语？
这个右括号对应的是多远之前的左括号？
```

这些问题更像是“距离”和“相对位置”，而不是“第 137 个位置”。

## 7.5 Sinusoidal Position Encoding

Transformer 原论文使用固定的 sinusoidal position encoding。公式是：

```math
\mathrm{PE}(p,2i)=
\sin\left(\frac{p}{10000^{2i/d}}\right)
```

```math
\mathrm{PE}(p,2i+1)=
\cos\left(\frac{p}{10000^{2i/d}}\right)
```

其中：

1. `p` 是 token 位置。
2. `i` 是维度索引。
3. `d` 是 hidden size。

它的直觉是：

```text
用不同频率的 sin/cos 波表示位置。
低维度变化快，表示局部位置差异。
高维度变化慢，表示长距离位置趋势。
```

可以把它想象成给每个位置分配一组“多频率坐标”。位置 10、位置 11 在高频维度上会有明显差别；位置 1000、位置 2000 在低频维度上仍能体现大尺度差异。

Sinusoidal 的一个重要性质是：固定公式可以生成任意长度的位置向量。理论上，即使训练时只见过 512 长度，推理时也可以生成 1024、2048 位置的编码。

但这不等于模型一定能很好外推。原因是：

```text
位置向量能生成，不代表模型在训练中学会了如何使用更长位置上的 attention pattern。
```

这是面试中很容易混淆的点。

## 7.6 Sinusoidal 为什么用 sin 和 cos

sin/cos 不是随便选的。它有几个好处。

第一，值域稳定。

```text
sin 和 cos 的输出始终在 [-1, 1]
```

不会因为位置变大而数值爆炸。

第二，多频率可以覆盖不同尺度。

短距离和长距离都能有可区分的模式。

第三，sin/cos 的相位关系可以帮助模型推断相对位置。

对于同一频率，有：

```math
\sin(a+b)=\sin(a)\cos(b)+\cos(a)\sin(b)
```

```math
\cos(a+b)=\cos(a)\cos(b)-\sin(a)\sin(b)
```

这意味着位置 `pos + offset` 的编码，可以通过位置 `pos` 和 `offset` 的三角函数关系表达。也就是说，固定正弦位置编码天然包含某种相对位移结构。

不过要注意：Transformer 原论文里 sinusoidal 是加到输入 embedding 上，而不是直接加到 attention score 中。模型是否真正利用这些相对位移结构，要靠训练学出来。

## 7.7 Learned Absolute Position Embedding

另一种常见方法是 learned absolute position embedding。

做法很简单：

```python
position_embedding = nn.Embedding(max_position, d_model)
h_i = token_embedding_i + position_embedding_i
```

数学上仍然是：

```math
h_i=e_i+p_i,\qquad p_i\in\mathbb{R}^{d}
```

这和词表 embedding 类似，只不过词表 id 换成了位置 id。

优点：

1. 简单，容易实现。
2. 模型可以根据数据自动学习每个位置的表示。
3. 在固定最大长度任务上通常效果不错。

缺点：

1. 需要预先设定 `max_position`。
2. 超过训练长度的位置没有学过，不能自然外推。
3. 对长上下文扩展不友好。
4. 学到的位置向量可能过拟合训练长度分布。

GPT-2、BERT 等早期 Transformer 模型常使用 learned absolute position embedding。对于固定上下文长度的模型，这种方法足够直接。但随着 LLM 上下文窗口从几百、几千扩展到几万、几十万，learned absolute embedding 的局限变得明显。

## 7.8 绝对位置编码的核心限制

绝对位置编码最大的问题是：它把位置当作一个“绝对编号”。

但很多语言规律具有平移不变性。

例如：

```text
第 20 个 token 关注第 19 个 token
第 2000 个 token 关注第 1999 个 token
```

这两个模式本质上都是“关注前一个 token”。如果使用绝对位置，模型可能需要从不同位置编号中学出同一种相对规律。

相对位置方法则更直接：

```text
不问你在第几个位置，只问你和我相距多远。
```

这就是 relative position representation 的动机。

## 7.9 Relative Position Representation

Shaw et al. 提出的相对位置表示，把 token `i` 和 token `j` 之间的距离引入 attention。

普通 attention score 是：

```math
s_{ij}=q_i^\top k_j
```

加入相对位置后，可以变成类似：

```math
s_{ij}=q_i^\top(k_j+r_{i-j})
```

其中 `r_{i-j}` 是相对距离 `i-j` 对应的位置向量。

直觉是：

```text
token j 的内容重要不重要，不只取决于它的内容 k_j，也取决于它距离当前 token i 有多远。
```

比如在语言模型里，距离当前 token 很近的 token 通常更重要；在括号匹配、代码缩进、长距离依赖中，某些特定距离模式也可能重要。

相对位置的优点：

1. 更符合 attention 的 pairwise 结构。
2. 可以直接建模 token 两两之间的距离。
3. 对长度泛化通常比 learned absolute 更自然。

相对位置的缺点：

1. 实现比绝对位置复杂。
2. 朴素实现可能引入额外的 `n x n` 相对位置张量。
3. 长序列下内存和计算压力更大。

Music Transformer 的一个贡献就是改进相对 attention 的实现，降低中间相对信息的内存开销，使更长序列的音乐生成更可行。

## 7.10 Relative Position Bias

后续很多模型使用更简单的 relative position bias。

它不再为每个相对距离引入完整向量，而是给 attention score 加一个 bias：

```math
s_{ij}=
\frac{q_i^\top k_j}{\sqrt{d_h}}+
b_{\mathrm{bucket}(i-j)}
```

其中 `bucket(i-j)` 表示把相对距离分桶。

T5 就使用了类似思路。距离分桶的意义是：

```text
近距离区分得细一点，远距离区分得粗一点。
```

例如：

```text
距离 1、2、3、4 分别有不同 bias
距离 128 到 255 共享一个 bucket
距离 256 以上共享更粗的 bucket
```

这样做的好处是参数少、实现相对简单，并且贴合 attention score。

Relative bias 和 relative representation 的区别可以简单理解为：

```text
relative representation：给距离一个向量，影响 Q/K/V 的交互。
relative bias：给距离一个标量，直接偏置 attention score。
```

relative bias 更轻量，但表达能力也更受限。

## 7.11 RoPE：Rotary Position Embedding

RoPE 是现代 decoder-only LLM 中非常常见的位置编码方法。LLaMA、Qwen、ChatGLM、Baichuan 等大量模型都采用或改造了 RoPE。

RoPE 的核心思想是：

```text
不要把位置向量简单加到 token embedding 上，而是根据位置对 Q 和 K 做旋转。
```

假设 Q/K 的某两个维度组成一个二维平面：

```math
x=[x_1,x_2]^\top
```

RoPE 会根据位置 `pos` 和频率 `theta` 做旋转：

```math
\begin{bmatrix}
x'_1\\
x'_2
\end{bmatrix}
=
\begin{bmatrix}
\cos(p\theta) & -\sin(p\theta)\\
\sin(p\theta) & \cos(p\theta)
\end{bmatrix}
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
```

也就是把向量在二维平面里转一个角度。不同维度对使用不同频率。

用矩阵表示，就是：

```math
R_p=
\begin{bmatrix}
\cos(p\theta) & -\sin(p\theta)\\
\sin(p\theta) & \cos(p\theta)
\end{bmatrix}
```

然后：

```math
q'_i=R_iq_i,\qquad k'_j=R_jk_j
```

attention score 变成：

```math
s_{ij}=(R_iq_i)^\top(R_jk_j)
```

旋转矩阵有一个关键性质：

```math
R_i^\top R_j=R_{j-i}
```

所以：

```math
s_{ij}=q_i^\top R_{j-i}k_j
```

这说明 RoPE 虽然用的是绝对位置 `i` 和 `j` 分别旋转 Q/K，但最终 Q/K 点积里自然出现了相对位置 `j - i`。

这就是 RoPE 最重要的性质：

```text
用绝对位置实现，点积时体现相对位置。
```

## 7.12 RoPE 为什么适合 decoder-only LLM

RoPE 在现代 LLM 中流行，不只是因为效果好，还因为它非常适合 decoder-only attention 的工程形态。

第一，RoPE 只作用在 Q/K 上，不作用在 V 上。

这很自然，因为位置信息主要影响“该关注谁”，也就是 attention score。真正被读取的信息 V 可以保持内容表示。

第二，RoPE 兼容 causal attention。

decoder-only 模型每个位置只能看历史 token。RoPE 通过 Q/K 旋转注入位置，不改变 causal mask。

第三，RoPE 兼容 KV cache。

推理时历史 K/V 会缓存。RoPE 对 K 的旋转可以在生成 K 时完成并缓存下来。后续新 token 只需要对新的 Q/K 应用对应位置旋转。

第四，RoPE 有较好的相对位置归纳偏置。

语言模型常常关心“距离当前 token 多远”，RoPE 在 attention score 中自然产生相对距离项。

第五，RoPE 可以围绕频率和位置索引做上下文扩展。

很多长上下文方法本质上都在调整 RoPE 的位置尺度或频率分布。

## 7.13 RoPE 的实现直觉

在代码中，RoPE 通常不会真的构造旋转矩阵，而是用 `cos`、`sin` 和半维度交换来实现。

伪代码如下：

```python
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return stack([-x2, x1], dim=-1).flatten(-2)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin
```

对 Q/K 应用：

```python
q = apply_rope(q, cos[position_ids], sin[position_ids])
k = apply_rope(k, cos[position_ids], sin[position_ids])
```

需要注意几个工程细节：

1. `cos/sin` 通常可以预计算。
2. `position_ids` 必须和真实 token 位置一致。
3. 使用 KV cache 时，新 token 的 position id 要接在历史长度之后。
4. padding、packed sequence、sliding window、prefix cache 都会影响 position id 设计。
5. 如果只对一部分 head dimension 使用 RoPE，也要保证实现和模型配置一致。

## 7.14 ALiBi：Attention with Linear Biases

ALiBi 是另一条路线。它不把位置编码加到 embedding，也不旋转 Q/K，而是直接修改 attention score。

普通 attention score：

```math
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}
```

ALiBi 加上与距离成比例的线性惩罚：

```math
s_{ij}^{(h)}=
\frac{q_i^\top k_j}{\sqrt{d_h}}-m_h|i-j|
```

其中 `m_h` 是第 `h` 个 head 的斜率。不同 head 使用不同斜率。

对 causal LM 来说，当前位置 `i` 只能看过去位置 `j <= i`，距离可以写成：

```math
\Delta_{ij}=i-j,\qquad j\le i
```

因此 causal ALiBi 也常写成：

```math
s_{ij}^{(h)}=
\frac{q_i^\top k_j}{\sqrt{d_h}}-m_h(i-j),
\qquad j\le i
```

ALiBi 的直觉是：

```text
距离越远，attention score 越被惩罚。
不同 head 的惩罚强度不同，有些 head 更偏局部，有些 head 可以看得更远。
```

ALiBi 的优点：

1. 不需要 position embedding 参数。
2. 不需要预设最大位置 embedding 表。
3. 对长度外推友好。
4. 实现上只是 attention score bias。
5. 训练短上下文、测试更长上下文时有较好表现。

缺点：

1. 它强引入 recency bias，即越近越容易被关注。
2. 对需要远距离精确匹配的任务，线性惩罚可能过强。
3. 它不像 RoPE 那样在 Q/K 几何结构中编码相对相位。
4. 在现代 LLM 生态中，RoPE 的使用更广，相关优化和扩展方法更多。

ALiBi 的核心价值是说明：

```text
位置编码不一定要是 embedding，也可以是 attention score 上的结构化 bias。
```

## 7.15 Sinusoidal、Learned、Relative、RoPE、ALiBi 对比

可以从几个维度比较这些方法。

### 注入位置

```text
Sinusoidal：加到输入 embedding
Learned absolute：加到输入 embedding
Relative representation：加到 attention 的 K/V 或 score 交互
Relative bias：加到 attention score
RoPE：旋转 Q/K
ALiBi：加到 attention score
```

### 绝对还是相对

```text
Sinusoidal：绝对位置为主，但公式包含可推导的相对结构
Learned absolute：绝对位置
Relative representation：相对位置
Relative bias：相对位置
RoPE：用绝对位置实现，相对位置体现在 Q/K 点积中
ALiBi：相对距离 bias
```

### 长度外推

```text
Learned absolute：较弱，超过训练长度没有学过 embedding
Sinusoidal：公式可外推，但模型行为未必稳定
Relative position：通常比 learned absolute 更自然，但受实现和 bucket 影响
RoPE：有一定外推能力，但长距离相位可能失配，需要 scaling/interpolation
ALiBi：外推友好，设计目标就是 train short, test long
```

### 工程复杂度

```text
Learned absolute：最低
Sinusoidal：低
ALiBi：低到中，需 attention bias 支持
Relative bias：中
RoPE：中，需处理 Q/K reshape、cos/sin、position id、cache
Relative representation：中到高，朴素实现可能内存重
```

### 现代 LLM 使用情况

```text
早期 encoder/decoder：sinusoidal 或 learned absolute 常见
BERT/GPT-2：learned absolute 常见
T5：relative position bias
LLaMA-like decoder-only LLM：RoPE 常见
部分长上下文/实验模型：ALiBi 或 RoPE scaling 变体
```

## 7.16 位置编码和 KV Cache

位置编码与 KV cache 的关系非常重要，因为现代 LLM 推理几乎都依赖 KV cache。

对于 learned absolute 或 sinusoidal，如果位置编码只加在输入 embedding 上，那么每个 token 的 K/V 已经包含位置信息，缓存即可。

对于 RoPE，K 在进入 attention score 前会被旋转。推理时通常缓存旋转后的 K：

```text
历史 token 生成 K
根据历史位置对 K 应用 RoPE
缓存旋转后的 K
新 token 生成 Q/K
根据新位置对 Q/K 应用 RoPE
新 Q 与缓存 K 做 attention
```

如果 position id 出错，模型表现会明显异常。例如：

1. 继续生成时 position id 从 0 重新开始。
2. prefix cache 拼接后位置没有正确偏移。
3. packed batch 中不同样本的位置混在一起。
4. sliding window 中局部位置和全局位置处理不一致。

这些问题在训练 loss 上未必容易发现，但在推理长文本、续写、RAG、多轮对话中会造成质量下降。

## 7.17 位置编码和长上下文扩展

长上下文扩展不是简单把 `max_position` 改大。

对于 RoPE 模型，如果训练时最大长度是 2048，推理时直接用 8192，可能出现问题。原因是 RoPE 的旋转角度随位置增长，远超训练范围后，某些频率维度上的相位关系会进入模型没见过的区域。

直观地说：

```text
模型训练时只学过 0 到 2048 的位置相位模式。
推理时突然给它 8192 的相位，attention score 分布可能失真。
```

常见扩展思路包括：

1. 位置插值。
2. RoPE scaling。
3. NTK-aware scaling。
4. YaRN。
5. 长上下文继续预训练或微调。

## 7.18 Position Interpolation

Position Interpolation 的思路是：

```text
不要让位置索引直接外推到训练范围之外，而是把更长上下文的位置压缩回原训练范围。
```

假设原模型训练长度是 `L = 2048`，现在希望扩展到 `L' = 8192`。

直接外推：

```math
p=0,1,2,\ldots,8191
```

位置插值：

```math
p'=p\cdot\frac{L_{\mathrm{train}}}{L_{\mathrm{target}}}
```

于是最大位置从 8191 压回接近 2048。

这样做的好处是避免让 RoPE 进入过大的未训练相位区域。论文中也强调，相比直接 extrapolation，interpolation 的 attention score 上界更稳定。

但它也有代价：

```text
位置分辨率变低。
```

原来相邻 token 的位置差是 1，现在可能变成 0.25。模型需要通过少量长上下文微调适应这种新的位置尺度。

## 7.19 YaRN

YaRN 是 RoPE 长上下文扩展的一类高效方法。它的目标是用更少 token、更少训练步数，把已有 RoPE LLM 扩展到更长上下文。

从工程角度看，YaRN 的意义是：

```text
长上下文扩展不一定要从头预训练，也可以在已有 RoPE 模型上做位置编码改造和少量继续训练。
```

这类方法通常围绕以下问题设计：

1. 不同频率维度应该如何缩放。
2. 短上下文能力如何尽量保留。
3. 长上下文中 attention score 分布如何避免异常。
4. 需要多少长上下文数据继续训练。
5. passkey retrieval 这类任务能否通过，不代表真实长文理解一定强。

这一点非常重要。长上下文模型评估不能只看“能不能在 100K token 中找出一个 key”。真实任务还包括：

1. 多文档归纳。
2. 长代码仓库理解。
3. 跨段落推理。
4. 多轮对话一致性。
5. 长上下文中的抗干扰能力。

位置编码扩展只是长上下文能力的一部分，attention pattern、训练数据、任务分布、检索增强、推理策略都会影响最终效果。

## 7.20 位置编码审计指标与最小 demo

工程里位置编码最容易出错的地方不是公式本身，而是 position id、KV cache、packed sequence 和长上下文配置是否一致。可以把位置编码审计抽象成一个门禁：

```math
G_{\mathrm{pos}}=
\mathbf{1}[C_{\mathrm{cache}}=1]\cdot
\mathbf{1}[C_{\mathrm{pack}}=1]\cdot
\mathbf{1}[C_{\mathrm{long}}=1]
```

其中：

```math
C_{\mathrm{cache}}=
\mathbf{1}[p_{\mathrm{new},0}=L_{\mathrm{cache}}]
```

表示使用 KV cache 继续生成时，新 token 的第一个 position id 必须接在已有缓存长度之后。

```math
C_{\mathrm{long}}=
\mathbf{1}[L_{\mathrm{target}}\le L_{\mathrm{train}}]
+\mathbf{1}[L_{\mathrm{target}}>L_{\mathrm{train}}]\cdot
\mathbf{1}[S_{\mathrm{rope}}=1]
```

这里 `S_rope` 表示已经显式配置 RoPE scaling、Position Interpolation、YaRN 或继续训练策略。这个公式不是说某个策略一定有效，而是提醒：目标上下文超过训练上下文时，不能只改 `max_position`。

下面 demo 用纯 Python 同时检查五件事：

1. sinusoidal 位置向量的多频率结构。
2. RoPE 点积对相同相对距离给出相同分数。
3. ALiBi 对更远历史 token 给出更负的 bias。
4. Position Interpolation 如何把 8192 长度映射回 2048 训练尺度。
5. position id 审计如何发现 cache reset、packed sequence 泄漏和长上下文无 scaling。

```python
import math


def sinusoidal_position(pos, dim=8, base=10000.0):
    values = []
    for i in range(dim // 2):
        angle = pos / (base ** (2 * i / dim))
        values.extend([math.sin(angle), math.cos(angle)])
    return values


def rotate_pair(x, pos, theta):
    angle = pos * theta
    c = math.cos(angle)
    s = math.sin(angle)
    return [x[0] * c - x[1] * s, x[0] * s + x[1] * c]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rope_score(q, k, i, j, theta=0.125):
    return dot(rotate_pair(q, i, theta), rotate_pair(k, j, theta))


def alibi_bias(query_pos, key_pos, slope):
    if key_pos > query_pos:
        return None
    return -slope * (query_pos - key_pos)


def interpolate_position(pos, train_length, target_length):
    return pos * train_length / target_length


def audit_position_plan(cache_len, new_position_ids, packed_segments,
                        train_context, target_context, rope_scaling):
    issues = []
    if cache_len > 0 and new_position_ids and new_position_ids[0] != cache_len:
        issues.append("cache_position_reset")
    for name, ids in packed_segments[1:]:
        if ids and ids[0] != 0:
            issues.append("packed_sequence_position_leak")
            break
    if target_context > train_context and rope_scaling is None:
        issues.append("over_train_context_without_scaling")
    return issues, len(issues) == 0


q = [1.0, 0.2]
k = [0.4, -0.7]
rope_scores = {
    "score_2_5": round(rope_score(q, k, 2, 5), 6),
    "score_10_13": round(rope_score(q, k, 10, 13), 6),
    "score_2_6": round(rope_score(q, k, 2, 6), 6),
}

alibi = {
    "key_6": alibi_bias(7, 6, 0.125),
    "key_2": alibi_bias(7, 2, 0.125),
    "key_8": alibi_bias(7, 8, 0.125),
}

scaled_positions = [
    round(interpolate_position(p, 2048, 8192), 2)
    for p in [0, 1024, 4096, 8191]
]

issues, gate_pass = audit_position_plan(
    cache_len=4,
    new_position_ids=[0, 1],
    packed_segments=[("doc_a", [0, 1, 2]), ("doc_b", [3, 4])],
    train_context=2048,
    target_context=8192,
    rope_scaling=None,
)

print("sin_pos_3=", tuple(round(x, 4) for x in sinusoidal_position(3)))
print("rope_scores=", rope_scores)
print("alibi_bias=", alibi)
print("pi_scaled_positions=", scaled_positions)
print("audit_issues=", issues)
print("position_gate_pass=", gate_pass)
```

一组固定输出如下：

```text
sin_pos_3= (0.1411, -0.99, 0.2955, 0.9553, 0.03, 0.9996, 0.003, 1.0)
rope_scores= {'score_2_5': 0.527625, 'score_10_13': 0.527625, 'score_2_6': 0.602123}
alibi_bias= {'key_6': -0.125, 'key_2': -0.625, 'key_8': None}
pi_scaled_positions= [0.0, 256.0, 1024.0, 2047.75]
audit_issues= ['cache_position_reset', 'packed_sequence_position_leak', 'over_train_context_without_scaling']
position_gate_pass= False
```

这组结果可以这样读：

1. `sin_pos_3` 的不同维度变化速度不同，体现多频率位置坐标。
2. `score_2_5` 和 `score_10_13` 相等，因为两组位置差都是 3；`score_2_6` 不同，因为位置差变成 4。
3. ALiBi 中 `key_2` 比 `key_6` 离 query 更远，所以 bias 更负；未来位置 `key_8` 在 causal attention 中不可见。
4. Position Interpolation 把 8191 映射到 2047.75，避免直接落到训练长度外很远的位置。
5. 审计门禁失败不是因为某个位置编码算法错误，而是因为工程配置中同时出现了 cache position id 重置、packed sequence 位置泄漏和长上下文无 scaling。

## 7.21 面向专家：RoPE 的频率、相位和外推问题

RoPE 的每一对维度对应一个旋转频率。常见形式是：

```math
\theta_i=\theta_{\mathrm{base}}^{-2i/d}
```

其中 `base` 常见为 10000，后续模型可能使用不同 base 或 scaling 策略。

低维频率高，旋转快；高维频率低，旋转慢。位置变长时，高频维度会经历很多周期，低频维度负责更长尺度的位置变化。

外推问题来自：

```text
训练长度之外的相位组合没有被充分学习。
```

如果直接把位置扩展很多倍，Q/K 点积中的相对相位会发生模型未适应的变化。RoPE scaling 的各种方法，本质上都是试图重新分配或压缩这些相位变化。

可以把它理解为一个频谱问题：

```text
短上下文需要保留高分辨率局部位置信息。
长上下文需要避免远距离相位过快旋转导致 attention 失真。
```

这就是为什么简单线性缩放、NTK-aware scaling、YaRN 等方法会在不同频率维度上做更细致处理。

## 7.22 面向专家：位置编码和 attention pattern 的耦合

位置编码不是孤立模块。它和 attention pattern 强耦合。

对于 full attention，任意 token 可以看到任意历史 token，位置编码需要支持全局距离关系。

对于 sliding window attention，模型主要看到局部窗口，位置编码更强调窗口内局部距离。此时全局位置和局部位置如何选择，会影响模型是否能正确理解跨窗口信息。

对于 prefix LM 或 encoder-decoder attention，位置编码还要区分：

1. prefix 内部位置。
2. target 内部位置。
3. target 对 prefix 的 cross-attention 位置关系。

对于 packed sequence training，如果多个样本拼在同一个 sequence 里，position id 和 attention mask 必须配套设计。否则模型可能把不同样本之间的位置关系当成真实上下文。

对于多模态模型，位置编码更复杂。图像 patch、视频 frame、音频 token、文本 token 可能有不同位置结构。二维 RoPE、三维 RoPE、modality-specific position embedding 都是在解决这个问题。

## 7.23 常见误区

### 误区 1：有 position embedding 就一定能处理长上下文

不一定。位置编码只是让模型知道位置，长上下文能力还依赖训练长度、attention 计算、数据分布和评估任务。

### 误区 2：Sinusoidal 可以生成无限位置，所以天然能无限外推

公式能生成，不代表模型学会了使用训练范围外的位置。

### 误区 3：Learned position embedding 一定比固定位置编码强

在固定长度任务上 learned embedding 可能很好，但外推和扩展通常较弱。

### 误区 4：RoPE 是相对位置编码，所以没有外推问题

RoPE 在点积中体现相对位置，但仍有训练长度外的相位分布问题。

### 误区 5：ALiBi 总是优于 RoPE

ALiBi 外推友好、简单高效，但它引入强 recency bias。不同任务和模型规模下，效果不一定全面优于 RoPE。

### 误区 6：passkey retrieval 通过就说明长上下文理解强

passkey 更偏检索能力，不能完全代表长文推理、多证据整合和抗干扰能力。

## 7.24 面试高频问题

### 题 1：为什么 Transformer 需要位置编码？

参考回答：

```text
Self-Attention 本身主要根据 token 内容计算 Q/K/V 交互，对输入顺序没有内置建模能力。如果不加入位置信息，模型很难区分同一批 token 的不同排列。位置编码的作用是把绝对位置、相对距离或顺序关系注入 attention，使模型能够理解语序、局部邻近关系和长距离依赖。
```

### 题 2：Sinusoidal position encoding 的公式和直觉是什么？

参考回答：

```text
Transformer 原论文使用不同频率的 sin/cos 函数表示位置：偶数维用 sin，奇数维用 cos，频率随维度变化。直觉是用多频率信号为每个位置生成一个固定坐标，高频维度刻画局部差异，低频维度刻画长距离变化。它不需要学习参数，可以生成任意长度的位置编码，但这不等于模型一定能在训练长度外稳定泛化。
```

### 题 3：learned absolute position embedding 有什么问题？

参考回答：

```text
Learned absolute position embedding 简单有效，但需要预设最大长度，并且每个位置向量只在训练中被学习。超过训练长度的位置没有学过，天然外推能力弱。同时它把位置当作绝对编号，而语言中的很多规律更依赖相对距离，例如关注前一个 token 或匹配远处括号。
```

### 题 4：relative position representation 为什么更贴合 attention？

参考回答：

```text
Attention 本质上是 token pair 之间的交互，所以很多时候需要知道 query token 和 key token 的相对距离。Relative position representation 直接把 i 和 j 的距离加入 attention score 或 K/V 表示中，使模型能根据相对位置调节关注强度，比单纯绝对位置更符合许多序列任务的结构。
```

### 题 5：RoPE 的核心机制是什么？

参考回答：

```text
RoPE 根据位置对 Q 和 K 做二维旋转。第 i 个位置的 q 使用 R(i) 旋转，第 j 个位置的 k 使用 R(j) 旋转。由于旋转矩阵满足 R(i)^T R(j)=R(j-i)，所以 Q/K 点积中自然出现相对位置 j-i。RoPE 的特点是用绝对位置实现，但在 attention score 里体现相对位置关系。
```

### 题 6：为什么 RoPE 适合 decoder-only LLM？

参考回答：

```text
RoPE 只作用在 Q/K 上，直接影响 attention score，不改变 V，也不破坏 causal mask。它能自然表达相对距离，兼容 KV cache，推理时可以缓存旋转后的 K。现代 LLM 还可以围绕 RoPE 做 scaling、position interpolation、YaRN 等长上下文扩展，因此 RoPE 成为 LLaMA-like 模型的常见选择。
```

### 题 7：ALiBi 和 RoPE 有什么区别？

参考回答：

```text
ALiBi 不使用显式 position embedding，也不旋转 Q/K，而是在 attention score 上加入和距离成比例的线性 bias，距离越远惩罚越大。它简单、参数少、外推友好，但带有较强 recency bias。RoPE 则通过旋转 Q/K 在点积中体现相对位置，表达方式更几何化，也是现代 decoder-only LLM 更常见的选择。
```

### 题 8：为什么 RoPE 模型扩展上下文不能只改 max length？

参考回答：

```text
RoPE 的旋转角度随位置变化。训练时模型只见过某个长度范围内的相位模式，直接推理到更长位置会让 Q/K 点积出现未训练过的相位组合，attention 分布可能失真。Position Interpolation、RoPE scaling、YaRN 等方法通过缩放或重分配位置频率，让长上下文位置落在更稳定的相位范围内，并通常配合少量长上下文继续训练。
```

## 7.25 小练习

1. 用自己的话解释为什么没有位置编码的 self-attention 不知道 token 顺序。
2. 写出 sinusoidal position encoding 的公式，并解释高频和低频维度分别起什么作用。
3. 比较 learned absolute position embedding 和 relative position bias 的优缺点。
4. 推导 RoPE 中为什么 `(R(i)q)^T(R(j)k)` 会和 `j-i` 有关。
5. 用 PyTorch 写一个简化版 `apply_rope` 函数。
6. 解释 ALiBi 为什么有利于长度外推，以及它可能带来什么偏置。
7. 思考 packed sequence training 中 position id 和 attention mask 如果设计错误，会导致什么问题。
8. 阅读 Position Interpolation 摘要，解释为什么插值可能比直接外推更稳定。

## 7.26 本章总结

本章讲了 Transformer 位置编码的主要路线：Sinusoidal、learned absolute、relative position、RoPE 和 ALiBi。

核心结论：

1. Self-Attention 本身没有内置顺序感知能力，必须注入位置信息。
2. 绝对位置编码简单，但对相对距离和长度外推不一定友好。
3. Sinusoidal 用多频率 sin/cos 表示位置，可生成任意长度编码，但模型外推仍取决于训练。
4. Learned absolute position embedding 简单有效，但超过训练长度时天然受限。
5. Relative position representation 和 relative bias 更贴合 attention 的 pairwise 结构。
6. RoPE 通过旋转 Q/K 注入位置，在点积中自然体现相对位置，是现代 decoder-only LLM 的常见选择。
7. ALiBi 通过 attention score 的线性距离 bias 注入位置，简单且外推友好，但有 recency bias。
8. 长上下文扩展不仅是改最大长度，还涉及 RoPE 相位、位置尺度、继续训练和真实长文任务评估。

下一章会进入 Causal Mask、Prefix LM、Bidirectional Attention 与注意力模式，解释不同模型如何通过 mask 和 attention pattern 控制信息可见性。
