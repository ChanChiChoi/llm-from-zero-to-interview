# 第二章：Self-Attention 的信息路由本质

## 2.1 本章定位

上一章讲了 Transformer 为什么能成为大模型基础架构。本章进入 Transformer 的核心：Self-Attention。

很多人会背公式：

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

但如果只会背公式，面试官继续追问就容易卡住：

1. $Q,K,V$ 为什么要分开？
2. Attention score 到底表示什么？
3. 为什么要除以 $\sqrt{d_k}$？
4. Softmax 后的权重能不能解释成“重要性”？
5. Self-Attention 为什么是一种信息路由机制？
6. 它与固定卷积核的局部读取相比，有什么不同？
7. 为什么 attention 强，但长上下文仍然会 lost-in-the-middle？

本章把单个 Self-Attention 层当作信息混合机制来解释：一个位置如何用 Query 与可见位置的 Key 计算权重，再用这些权重混合 Value。读完后，你应能沿着一组小向量复走这几步，说明 softmax 的归一化范围，并区分注意力权重与模型最终预测。

## 2.2 本章与相邻章节的边界

本章先建立单头注意力的共同语言。下一章再深入多头投影、表示子空间与 head 的作用；第 8 章系统讲 attention mask 和不同可见模式；第 10、11、13 章分别展开注意力计算成本、KV cache 和长上下文的有效利用。这里会给出理解这些主题所需的最小连接，不重复完整推导。

## 2.3 Self-Attention 解决的核心问题

序列中的一个位置要更新自己的表示时，需要决定从哪些其他位置读取信息，以及怎样汇总读到的内容。早期序列到序列模型常把整句压进固定长度的上下文向量。Bahdanau 等人的 RNN encoder-decoder 注意力让 decoder 在生成每个目标词时，对输入编码器的各位置计算软对齐，再据此形成当前步的上下文向量；这样 decoder 就能按当前目标位置读取源序列的信息，减轻单个固定向量承载整句信息的瓶颈。这是一种跨序列读取。[Bahdanau 等人的论文](https://arxiv.org/html/1409.0473)

Transformer 原论文把这类加权读取用于同一序列内部：每个位置都可以根据自己的表示，从其他可见位置汇聚信息。[Transformer 原论文](https://arxiv.org/html/1706.03762v7) 因而理解 Self-Attention 时，可以沿着三个动作走：

```text
计算匹配分数 -> 把分数变成权重 -> 按权重混合内容
```

例如在句子

```text
The animal didn't cross the street because it was too tired.
```

中，读者可能会问：处理 `it` 的位置怎样从前文取得有用线索？Self-Attention 会先比较该位置的 Query 与各可见位置的 Key，再把得到的权重用于混合对应的 Value。这个例子说明要解决的问题；它不表示某个 attention 权重就能独立判定代词指向。

Self-Attention 也不只是“看所有 token”。哪些位置能够参与读取由 mask 等结构约束决定；权重是从当前表示计算出来的，输出是 Value 向量的加权混合。

## 2.4 Query、Key 与 Value 的角色直觉

令 $X$ 表示序列各位置的输入向量。Self-Attention 用三组可学习的线性投影产生 Query、Key 和 Value：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

可以把三种投影视为一次可微分的信息读取过程：

1. Query 是当前位置参与计算匹配分数的向量。
2. Key 是候选位置参与匹配的向量。
3. Value 是候选位置参与输出混合的向量。

为什么不直接用同一个向量？因为“如何匹配”和“传递什么表示”不一定相同。Query 与 Key 共同影响读取权重，Value 决定权重作用于哪些内容。

以下向量是人工构造的演算例子，标签不表示真实模型会把 `animal` 编码成某种可直接读出的语义向量。这里只跟踪一个 Query 从同一序列中的两个候选位置读取信息，借此隔离单次注意力运算，不代表完整层的全部计算。设 Query 为 $q=[1,0]$，两个 Key 分别为 $k_{\mathrm{animal}}=[1,0]$ 与 $k_{\mathrm{street}}=[0,1]$，对应 Value 为 $v_{\mathrm{animal}}=[2,0]$ 与 $v_{\mathrm{street}}=[0,4]$。这里 Key 的维度为 $d_k=2$。

位置分数先由点积得到： $q^\top k_{\mathrm{animal}}=1,\;q^\top k_{\mathrm{street}}=0$。除以 $\sqrt{2}$ 后，分数约为 $[0.707,0]$。第一项 softmax 权重为 $e^{0.707}/(e^{0.707}+e^0)\approx0.67$，另一项约为 $0.33$。因此输出约为：

```math
0.67v_{\mathrm{animal}}+0.33v_{\mathrm{street}}\approx[1.34,1.32]
```

这个计算把关键关系连起来：Query 和 Key 决定各 Value 获得的权重，输出则是这些 Value 的加权混合，不是把最高分位置原样复制。实际模型中的向量由训练得到的投影产生，不能按上述人为标签作语义解码。

## 2.5 Attention Score 是什么

标准 Transformer 使用缩放点积作为一种匹配分数。先看未缩放的点积：

```math
s_{ij}=q_i^\top k_j
```

它是位置 $i$ 的 Query 与位置 $j$ 的 Key 的未归一化分数。点积同时受向量夹角和长度影响；只有在长度等条件相近时，方向更一致才通常对应更大的点积。

对长度为 $N$ 的序列，全连接 self-attention 会形成一个 $N\times N$ 的分数矩阵：

```math
S=QK^\top,\qquad S\in\mathbb{R}^{N\times N}
```

第 $i$ 行收集位置 $i$ 对各候选 Key 的分数。第 $j$ 列收集各 Query 与位置 $j$ 的 Key 分别得到的分数；它不是该位置对最终输出的重要性排名。

这张矩阵记录了尚未归一化的候选连接分数。对每行沿 Key 位置应用 softmax，并按可见性 mask 去除非法连接后，才得到用于混合 Value 的权重。每层、每个 head 使用各自的表示和投影，因此权重也可能不同。

这和固定卷积核很不一样。卷积的局部连接模式基本固定，而 attention 的连接权重由内容决定。

## 2.6 为什么要除以 $\sqrt{d_k}$

Scaled dot-product attention 会把 Query-Key 点积除以 $\sqrt{d_k}$，再对每个 Query 的候选 Key 分数单独做 softmax：

```math
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
```

为什么缩放因子是 $\sqrt{d_k}$？

Transformer 原论文用一个简化假设解释这个选择：若 Query 和 Key 的各维分量相互独立、均值为 $0$、方差为 $1$，维度为 $d_k$ 的点积方差就是 $d_k$。因此维度增加时，点积的典型幅度也会增加。除以 $\sqrt{d_k}$ 后，在这一假设下分数方差回到约 $1$。

若输入 softmax 的分数幅度很大，分布可能变得很尖锐，进入梯度较小的区域：

```text
一个位置接近 1，其他位置接近 0
```

缩放的作用是控制点积随维度增长的尺度，缓和 softmax 过度饱和的风险。原论文将此作为 scaled dot-product 的设计动机；它不是对任意网络状态或训练配置都能保证训练稳定的定理。[原论文 §3.2.1](https://arxiv.org/html/1706.03762v7)

面试中可以这样说：缩放不是为了让分数本身变成概率，而是为了控制点积的尺度；softmax 再把每个 Query 的分数转换成沿候选 Key 位置归一化的权重。

## 2.7 Softmax 权重是不是“重要性”

对每个 Query 的分数沿 Key 位置应用 softmax 后，得到 attention weights：

```math
a_{ij}=\frac{\exp(s_{ij})}{\sum_{m=1}^{N}\exp(s_{im})}
```

然后输出：

```math
o_i=\sum_{j=1}^{N}a_{ij}v_j
```

在单个 head 的计算中， $a_{ij}$ 是位置 $j$ 的 Value 混入位置 $i$ 输出时使用的系数。这是对本层信息混合过程的直接描述，但不能单独回答“原输入中的哪个 token 导致模型最终给出这个预测”。Value 已经过投影；attention 输出还会与残差路径、前馈层和其他层的计算共同影响后续表示。

研究也提示解释结论要按模型和任务限定。Jain 与 Wallace 在若干使用 BiLSTM 的文本分类、问答和自然语言推理任务中发现，attention weights 与其他特征重要性指标并不总一致，也能构造关注位置很不同但预测近似的权重分布。[Jain 与 Wallace](https://arxiv.org/html/1902.10186v3) Wiegreffe 与 Pinter 则指出，是否构成解释取决于解释定义与测试设计，并在 RNN attention 设定中提出相应诊断方法。[Wiegreffe 与 Pinter](https://arxiv.org/html/1908.04626v2)

这两项工作主要研究 RNN attention，不能直接证明所有 Transformer 或大语言模型中的注意力权重都没有解释价值。稳妥说法是：attention weight 精确描述当前层当前 head 如何加权混合 Value；若要解释模型的最终决策，还需要针对相应模型、任务和解释定义提供额外证据。

## 2.8 Self-Attention 的矩阵形状

先看标准多头结构，暂不考虑 GQA、MQA 等变体。若 hidden size 为 $D$、head 数为 $H$、每个 head 的维度为 $d_h$，则：

```math
D=H d_h
```

批大小与序列长度分别用 $B$ 和 $N$ 表示。输入张量的形状是 $B\times N\times D$。标准多头投影后， $Q,K,V$ 的形状都可以排成 $B\times H\times N\times d_h$。

对每个 batch、每个 head，Query 与 Key 相乘得到一个 $N\times N$ 分数矩阵；softmax 后乘以 $V$，得到形状为 $B\times H\times N\times d_h$ 的各 head 输出。拼接这些输出并作输出投影后，形状回到 $B\times N\times D$。后续第 3 章会进一步解释不同 head 的投影，第 10 章再完整推导计算与显存成本。

这里的 $N\times N$ 表示每个 Query 位置与候选 Key 位置之间的成对分数数量，不表示所有实现都必须把完整矩阵写入显存。

## 2.9 一个最小 PyTorch 实现

下面是一个简化的单头 causal self-attention。输入 $x$ 的形状为 $B\times N\times D$，输出仍为 $B\times N\times D$。代码只展示投影、缩放点积、因果 mask 和 Value 加权，不包含多头、残差、归一化、前馈层或 padding mask。

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

示例状态：静态审阅，未运行。该模块依赖 PyTorch；把 `hidden_size` 设为输入宽度 $D$、`head_dim` 设为投影宽度，并传入形状为 $B\times N\times D$ 的张量，前向输出形状仍为 $B\times N\times D$。注意力分数的最后一维对应候选 Key 位置，因此 softmax 沿最后一维归一化；`causal_mask` 的上三角位置阻断未来 token。这里没有声明完整 Transformer block 或 production implementation。

这段代码依次完成输入投影、计算 Query-Key 分数、屏蔽未来位置、对候选 Key 作 softmax、加权混合 Value，最后把单头输出投回 hidden size。

因果 mask 只是任务允许读取范围的一种选择；下一节再看它与双向 encoder attention 的区别。

## 2.10 Causal Attention 和 Bidirectional Attention

Self-Attention 的 $Q,K,V$ 来自同一组序列表示，但它本身不决定哪些位置可以互相读取；可见关系由 attention mask 设定。Causal mask 屏蔽位置 $i$ 之后的 Key，使位置 $i$ 不能读到未来 token。在自回归语言建模中，这与按前缀预测下一个 token 的目标相配；代码示例把非法分数设为负无穷，再沿 Key 位置作 softmax。

双向 encoder attention 则允许一个位置同时读取左右上下文。BERT 使用 masked language modeling 训练这类双向表示；该设计服务于其训练目标，并不推出双向 attention 对所有任务都更好。[BERT 原论文](https://arxiv.org/html/1810.04805)

本章只用这两个例子说明“路由范围由 mask 控制”。Prefix LM、padding/loss mask、encoder-decoder 的注意力组合及其训练差异见[第 8 章](08-causal-mask-prefix-lm-bidirectional-attention与注意力模式.md)。

## 2.11 信息路由图：每层都在重写 token 表示

Attention 子层的输出不是最终答案，而是各位置的 Value 混合结果。对第 $\ell$ 层的位置 $i$，可写成：

```math
X^{(\ell)}=[x_1^{(\ell)},x_2^{(\ell)},\ldots,x_N^{(\ell)}]
```

```math
\tilde{x}_i^{(\ell)}=\sum_{j=1}^{N}a_{ij}^{(\ell)}v_j^{(\ell)}
```

所以位置 $i$ 的 attention 输出可以包含其他位置 Value 的信息。实际 Transformer block 通常还把该输出与残差路径结合，并继续经过归一化和前馈层；不能把上式当成完整 block 的全部计算。堆叠多层后，后层使用的是已经更新过的表示，跨位置的信息会继续混合。

## 2.12 Attention 和检索的相似与不同

Attention 与检索有相似的读取形式：Query 对候选 Key 计算权重，再混合对应的 Value。但 self-attention 的 K/V 是当前输入表示经过投影得到的，输出是可微分的加权混合；外部检索器则从索引中选择文档，再把所选内容交给模型。RAG 系统可以先做外部检索，再让 Transformer 在扩展后的输入中计算 self-attention。两者衔接，但不是同一个机制。

## 2.13 为什么长上下文仍然会失败

能够建立到某个位置的 attention 连接，不等于模型一定能在任务中可靠使用那里的信息。Lost in the Middle 在多文档问答和合成键值检索中移动相关信息的位置；论文所测模型在信息位于长上下文中部时，表现往往低于信息位于开头或结尾时。[原论文](https://arxiv.org/html/2307.03172v3)

这是特定模型和任务设置下的端到端表现，不能单独证明某个 attention head 没有读到该位置，也不能把失败归因于 attention 权重本身。长距离依赖、检索成功率和评测设计在[第 13 章](13-长距离依赖与lost-in-the-middle.md)单独讨论。

## 2.14 全连接注意力为什么成本随长度增长

全连接 attention 要为每个 Query 与每个候选 Key 建立分数关系。位置数从 $N$ 增至 $2N$ 时，成对关系数从 $N^2$ 增至 $4N^2$。这是密集连接模式的计数，不表示每种 kernel 都会把完整分数矩阵物化在显存里。计算量、实际显存与 kernel 优化分别见[第 10 章](10-o-n2-attention的计算与显存瓶颈.md)、[第 11 章](11-kv-cache长上下文推理和显存增长问题.md)和[第 20 章](20-FlashAttention与PagedAttention.md)。

## 2.15 Self-Attention 的常见误区

记住四个边界即可：

1. Attention weight 是当前层、当前 head 的 Value 混合系数，不是最终预测的完整因果解释。
2. 可见位置不等于任务能稳定利用的位置；评估结果必须注明模型和任务。
3. Self-Attention 与外部检索可以配合，但一个在输入表示间混合，另一个从索引中选文档。
4. 成对分数数量随 $N^2$ 增长，但具体计算、显存和推理缓存是不同成本问题，后续章节分别展开。

## 2.16 面试题

### 题 1：Self-Attention 的本质是什么？

参考回答：Self-Attention 让每个位置根据输入表示计算读取权重，再用这些权重混合候选位置的 Value。具体来说，Query 与各 Key 产生分数；softmax 将每个 Query 的分数沿候选 Key 位置归一化；归一化结果再作用于 Value。哪些候选位置可参与读取由 mask 控制。

### 题 2：为什么 Query、Key 和 Value 要分开？

参考回答：匹配逻辑和传递内容不一定相同。 $Q$ 与 $K$ 共同决定各位置的读取权重， $V$ 决定这些权重混合什么表示。使用不同投影，让模型可以分别学习匹配所需的表示和要传递的内容；多头如何进一步扩展这些投影由下一章说明。

### 题 3：为什么 attention score 要除以 $\sqrt{d_k}$？

参考回答：假设各分量相互独立、以零为均值、以一为方差，则 $QK^\top$ 中每个点积的方差随 $d_k$ 增加。除以 $\sqrt{d_k}$ 可按这一假设控制分数尺度，缓和 softmax 进入小梯度区域的风险。这是原论文给出的设计动机，不是对所有训练配置都成立的稳定性保证。

### 题 4：Attention weight 能解释模型行为吗？

参考回答：Attention weight 精确说明当前层、当前 head 如何加权混合各位置的 Value，但它不等于原输入对最终预测的因果贡献。把权重当解释时，必须说明模型、任务和解释标准。现有争论性实验主要针对 RNN attention，不能直接推广成所有 Transformer 注意力图都可解释或都不可解释。

### 题 5：为什么 attention 可以看全局，但长上下文仍会 lost-in-the-middle？

参考回答：可见范围说明计算图允许哪些位置互相传递信息，不保证模型会在任务中稳定利用所有位置。Lost in the Middle 在指定模型和多文档问答、合成键值检索设置中观察到位置敏感性；这支持评估有效使用能力，不足以把结果归因于某个 attention head。[原论文](https://arxiv.org/html/2307.03172v3)

### 题 6：Self-Attention 和 RAG 中的检索有什么区别？

参考回答：Self-Attention 在当前输入表示之间计算权重并混合 Value。RAG 的外部检索器先从索引中选出文档，再将文档作为输入上下文交给模型。两者可以接在同一系统里，但前者是神经网络内部的表示混合，后者是输入构造阶段的外部选择。

## 2.17 小练习

1. 沿单头计算写出张量形状从 $B\times N\times D$ 出发，经过分数、权重和 Value 混合后如何回到 $B\times N\times D$。
2. 用自己的话解释 $Q,K,V$ 的不同职责，不只复述“查询、键、值”。
3. 构造一句包含代词指代的英文句子，说明 self-attention 能提供哪种跨位置的信息路径；再说明为什么这条路径本身不能证明模型已经正确识别指代。
4. 解释为什么 $QK^\top$ 有 $N\times N$ 个成对分数，以及这与实际显存占用为何不是同一问题。
5. 说明 BERT 双向可见性与 causal attention 的差别，并指出它们各自受什么训练目标约束。
6. 阅读 [Lost in the Middle](https://arxiv.org/html/2307.03172v3)，概括其模型、任务和位置敏感性结论的边界。
7. 对照 §2.9 的代码，手工画出序列长度为 $4$ 时的 causal 可见关系，并解释 softmax 沿哪个维度归一化。
8. 在 §2.4 的演算例子中只改变 Value、保持 Query 与 Key 不变，说明权重为何不变而输出会改变。

## 2.18 本章总结

本章沿一条 Query 的读取过程解释了 Self-Attention： $Q$ 与各 $K$ 生成分数，按可见关系处理后经 softmax 得到权重，再用权重混合 $V$。缩放因子 $\sqrt{d_k}$ 控制点积分数随维度增长的尺度；attention weight 是局部混合系数，不是模型最终预测的完整解释。

Self-Attention 在当前输入表示之间传递信息，外部检索则从索引中选择内容。密集 attention 的成对关系数随 $N^2$ 增长，但可见性不等于长上下文中的可靠利用；这些性能与成本问题需要结合具体模型、任务和实现来评估。

下一章转向 $Q,K,V$ 投影和多头表示子空间，解释多头机制增加了什么表达方式，以及 head 重要性为何会随模型和任务变化。[第 3 章](03-qkv-attention-head与表示子空间.md)
