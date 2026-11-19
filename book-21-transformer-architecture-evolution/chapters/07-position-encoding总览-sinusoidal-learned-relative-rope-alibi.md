# 第七章：位置表示：绝对位置、相对位置、RoPE 与 ALiBi

## 7.1 同一批 token，为什么顺序会改变含义

比较“猫追狗”和“狗追猫”。两个序列包含相同的词，但谁追谁完全不同。模型如果只根据 token 内容做 self-attention，还需要一种线索把 token 身处的位置或两两之间的距离带入计算。

对没有位置输入、没有位置偏置、也没有 mask 的 self-attention，若用置换矩阵 $P$ 重排输入 $X$，输出会以相同方式重排：

```math
F(PX)=P F(X).
```

这叫置换等变：输入行换了顺序，输出行也跟着换。它不等于输出完全不变，但说明这层本身没有给“第几个 token”不同的标记。残差连接、逐位置前馈网络和多头注意力都不会自动补上这个缺口。

位置表示提供了顺序线索。它可以是加在输入上的向量，也可以是 query 和 key 之间的相对关系，或直接改变 attention score 的偏置。位置表示与 attention mask 是两个不同部件：前者告诉模型“位置或距离是什么”，后者限制“这个位置允许读取哪些位置”。第 8 章继续讲 mask 与可见性。

## 7.2 位置线索进入计算的三个位置

位置方法可以按它改变计算的位置来理解：

| 位置线索进入哪里 | 代表方法 | 它改变什么 |
| --- | --- | --- |
| token 输入表示 | Sinusoidal、Learned absolute | 把位置向量加到 token 表示上 |
| query-key 关系或 attention score | Shaw 相对表示、T5 relative bias、ALiBi | 让一对位置的距离影响打分，有些方法也让相对表示进入加权值 |
| query/key 向量变换 | RoPE | 按位置旋转 query 和 key，使点积出现相对位移项 |

这些方法不是一条从旧到新的替换链。它们把位置放进不同计算环节，也能与因果 mask 等结构一起使用。把位置向量、相对偏置或旋转参数并入矩阵计算，不要求模型按 token 逐个执行；训练时仍可并行处理多个位置，但具体实现会带来不同的计算和内存代价。

## 7.3 绝对位置：给 token 表示加上坐标

最直接的做法是给每个位置分配向量 $p_i$，再与位置 $i$ 的 token 表示 $x_i$ 相加：

```math
h_i=x_i+p_i.
```

这样输入 Transformer 的 $h_i$ 同时携带内容和位置。原始 Transformer 在 encoder 和 decoder 堆栈底部把 positional encoding 加到 token embedding 上；论文也比较了可学习的位置向量。该 WMT 翻译实验中，两种方案的结果接近，但这是单一任务与模型配置下的观察，不能推出它们在其他场景总是相同。[Vaswani 等，2017，§3.5](https://arxiv.org/html/1706.03762v7)

### Learned absolute position embedding

可学习绝对位置向量通常存成一张参数表：表的第 $i$ 行对应位置 $i$。模型通过训练更新这些行。

它的优点是形式简单，并能为训练中出现的位置学习各自向量。固定长度的位置表也带来明确边界：若请求的位置编号超出表范围，模型不能直接查到对应行；如果某些位置在训练数据中从未出现，它们也没有从这些样本中获得训练信号。扩表、插值或继续训练都属于额外处理，不是原表自然提供的能力。

### Sinusoidal：用多种频率生成位置向量

原始 Transformer 使用固定的正弦、余弦位置向量。对模型宽度 $d_{\text{model}}$，每两个维度共用一个频率：

```math
\begin{aligned}
PE_{p,2r} &=
\sin\left(\frac{p}{10000^{2r/d_{\text{model}}}}\right),\\
PE_{p,2r+1} &=
\cos\left(\frac{p}{10000^{2r/d_{\text{model}}}}\right).
\end{aligned}
```

这里 $p$ 是位置，$r$ 是维度对的编号。随着 $r$ 增大，频率降低、周期变长：有些维度对随相邻位置变化较快，另一些变化较慢。把这些不同周期的信号放在一起，模型就得到一个多尺度的位置坐标。位置向量仍是在输入处与 token 表示相加。

同一维度对的固定偏移关系可以直接从三角恒等式看出。令该维度对的频率为 $\omega_r$，位置向量片段为 $c_p=[\sin(p\omega_r),\cos(p\omega_r)]^\top$，那么：

```math
c_{p+\delta}=
\begin{bmatrix}
\cos(\delta\omega_r)&\sin(\delta\omega_r)\\
-\sin(\delta\omega_r)&\cos(\delta\omega_r)
\end{bmatrix}
c_p.
```

对固定偏移 $\delta$，右侧是对 $c_p$ 的线性变换。这说明正弦位置向量含有可利用的相对位移结构；原论文把它作为模型可能学习相对位置的设计动机。它并不表示 attention score 已被直接改成相对位置函数，也不证明模型一定学会了这种用法。

固定公式可以为任意整数位置计算向量。要区分“位置向量算得出来”和“模型能在训练长度外可靠工作”：后者需要模型在相应位置分布上学到可用行为，不能由公式本身保证。

## 7.4 相对位置：让一对 token 的距离参与计算

设第 $i$ 个 token 发出 query，第 $j$ 个 token 提供 key 和 value。定义相对偏移 $\delta=j-i$。$d_h$ 是单个 attention head 的 query/key 维度。相对位置方法不只问 token 位于全序列的哪个编号，也把这对 token 的相对距离带入 attention。

### Shaw：距离向量进入 key 和 value

Shaw 等人的方法为不同相对偏移学习表示，并在注意力打分和加权结果中使用它们。为了限制参数范围，论文把超出阈值 $k$ 的距离截到边界；$\mathrm{clip}(x,k)$ 把 $x$ 限制在区间 $[-k,k]$ 内：

```math
\begin{aligned}
e_{ij}
&=
\frac{q_i^\top(k_j+a^K_{ij})}{\sqrt{d_h}},\\
\alpha_{ij}
&=
\mathrm{softmax}_j(e_{ij}),\\
z_i
&=
\sum_j\alpha_{ij}(v_j+a^V_{ij}),\\
a^K_{ij},a^V_{ij}
&=
w^K_{\mathrm{clip}(j-i,k)},
w^V_{\mathrm{clip}(j-i,k)}.
\end{aligned}
```

$e_{ij}$ 是 query 对 key 的分数，$\alpha_{ij}$ 是归一化后的读取权重，$z_i$ 是输出。相对向量既能改变“关注谁”，也能改变“从该位置读出的表示”。这不是所有相对位置方法的共同实现。[Shaw 等，NAACL 2018，§3](https://aclanthology.org/N18-2074.pdf)

作者在 WMT 2014 英德、英法翻译配置中报告了相对于其绝对位置基线的 BLEU 改善；同时，加入绝对位置表示没有进一步改善这组翻译结果。该证据支持特定实现值得研究，不代表所有任务都从相对向量中获益。

### T5：把距离压成 attention score 的标量偏置

T5 使用另一种更直接的相对位置方案：把偏移映射到 bucket，再为该 bucket 查一个标量偏置并加到 attention logit 上：

```math
e_{ij}^{(h)}
=
\frac{q_i^\top k_j}{\sqrt{d_h}}
+
b^{(h)}_{\mathrm{bucket}(j-i)}.
```

近距离可以分到较细的 bucket，远距离使用更粗的分桶。T5 论文的具体设置使用 32 个相对位置 bucket，并将超过 128 的偏移合并到最远 bucket；这只是 T5 的参数化选择，不是相对偏置的固定标准。[Raffel 等，T5 论文，§2.1](https://arxiv.org/html/1910.10683v4)

Shaw 与 T5 都使用相对位置，但机制不同：前者用向量影响 key/value 交互，后者用标量直接调整 score。Transformer-XL 也把相对位置项写进 attention 打分，不过它与跨 segment 状态复用共同设计；论文的长依赖结果不能只归因于位置编码。[Dai 等，Transformer-XL，§3.3](https://arxiv.org/html/1901.02860)

## 7.5 RoPE：按绝对位置旋转 Q/K，让点积显出相对偏移

RoPE 把每个 attention head 的 query 和 key 维度分成二维对，因此 head 宽度通常取偶数。第 $m$ 对维度使用频率 $\omega_m$；位置 $p$ 对应旋转矩阵：

```math
R_m(p)=
\begin{bmatrix}
\cos(p\omega_m)&-\sin(p\omega_m)\\
\sin(p\omega_m)&\cos(p\omega_m)
\end{bmatrix}.
```

如果 query 位于 $i$，key 位于 $j$，分别旋转后计算内积。由于旋转矩阵是正交矩阵，两个旋转相乘时保留下来的位置项只与偏移有关：

```math
\begin{aligned}
(R_m(i)q_i)^\top(R_m(j)k_j)
&=q_i^\top R_m(i)^\top R_m(j)k_j\\
&=q_i^\top R_m(j-i)k_j.
\end{aligned}
```

一个简化例子可以看清这条关系：若某一维度对的内容 query 和 key 都是 $[1,0]^\top$，旋转后的点积为 $\cos((j-i)\omega_m)$。把两个位置同时平移相同距离，偏移不变，这一维度对的点积也不变。真实 attention 会对多个频率的维度对及其内容向量共同计算，不会只剩一个余弦值。

因此，RoPE 的绝对位置操作发生在 Q/K 上，而相对偏移出现在 QK 点积的代数形式里。它不把位置向量加到 token embedding，也不旋转 value。[Su 等，RoFormer，§§3.1–3.3](https://arxiv.org/html/2104.09864v4)

这个性质限定在位置旋转对单次 QK 点积的影响。完整网络的 hidden state、mask、训练分布和其他层仍会影响结果；“点积显式含相对偏移”不等于“整模型只依赖相对距离”，更不等于未见长度上的预测有保证。RoPE 的长上下文扩展方法及其评估边界见[第 18 章](18-RoPE与长上下文位置外推.md)。

## 7.6 ALiBi：把距离惩罚直接加到 score

ALiBi 不添加位置向量，也不旋转 Q/K。对于因果注意力中的 query 位置 $i$ 和可见的 key 位置 $j\le i$，第 $h$ 个 head 的 score 加上线性距离惩罚：

```math
e_{ij}^{(h)}
=
\frac{q_i^\top k_j}{\sqrt{d_h}}
-
m_h(i-j),
\qquad j\le i,
\quad m_h>0.
```

斜率 $m_h$ 控制距离惩罚的强度。若两个 key 的内容分数相同，较远的 key 会因为偏置更负而获得较低的 softmax 权重；但足够大的内容分数仍可能使远处 key 获得较高权重。因此 ALiBi 提供的是按距离递减的先验，不是“远处 token 不能被关注”的硬规则。[Press 等，ALiBi，§3](https://arxiv.org/html/2108.12409v2)

因为偏置可按任意距离计算，ALiBi 不需要为每个绝对位置分配一行 embedding。作者在特定语言模型实验中报告了短序列训练、长序列推理的结果，包括 1.3B 参数设置下以 1,024 token 训练并评估到 2,048 token。该论文结果受其模型、语料、训练和评测条件限定；距离偏置可以算到很远，不代表任意任务都能有效使用同样长的上下文。

## 7.7 位置公式、长度上限与有效能力是不同问题

比较位置方案时，先分清三个问题：

1. **位置编号能否表示？** 固定长度的 learned table 可能没有超出表范围的行；正弦函数、RoPE 公式和 ALiBi 距离偏置则可以按定义计算更远的位置。
2. **远距离如何区分？** T5 会把远距离合并到粗 bucket；Shaw 对距离截断；其他方法采用不同映射。公式定义范围，不等于对所有距离都保留同等精度。
3. **模型能否可靠完成任务？** 这取决于训练中见过的位置和任务、attention 路径、数据以及评估。位置公式在更长位置有定义，不足以回答这一问题。

Position Interpolation、YaRN 等方法通过改变 RoPE 的位置映射并在特定实验中继续适配模型，研究已有 checkpoint 的上下文扩展。[Position Interpolation](https://arxiv.org/html/2306.15595v2) 在 LLaMA 实验中报告由 2K 扩至 32K；[YaRN](https://arxiv.org/html/2309.00071v3) 在 LLaMA 2 的特定设置中报告扩至 128K。它们的算法、训练设置和评估不属于 RoPE 的基本代数性质；细节与证据边界见[第 18 章](18-RoPE与长上下文位置外推.md)。位置编码影响的是模型怎样处理位置线索，输入上限和长上下文实际能力还要结合注意力结构与任务测量。

## 7.8 因果 mask 能给出隐式顺序线索

“模型需要顺序信息”和“每个 Transformer 都必须使用显式位置编码”不是同一句话。因果 mask 让位置 $i$ 只能读取从开始到 $i$ 的前缀；不同位置可见的前缀大小不同，这本身可提供隐式顺序线索。Haviv 等人在特定 causal LM 实验中报告，即使没有显式位置编码，模型仍可学习位置信息；这不表示所有模型和任务都能去掉显式位置表示。[Haviv 等，Findings of EMNLP 2022](https://aclanthology.org/2022.findings-emnlp.99/)

所以比较位置方法前，应先确认 attention mask、状态更新和其他结构提供了什么顺序信息。第 8 章讲因果 mask 的可见性机制；第 64 章专门讨论 NoPE、隐式顺序以及它们不能支持哪些外推结论。

## 7.9 方法对照：同样处理位置，机制和边界不同

| 方法 | 位置进入哪里 | 相对距离怎样进入计算 | 结论边界 |
| --- | --- | --- | --- |
| Learned absolute | 输入 embedding | 由不同位置行间接表达 | 固定表外需另作处理；训练范围外表现不能由表内结果推出 |
| Sinusoidal | 输入 embedding | 固定偏移可由位置向量的线性变换表示 | 公式可计算更远位置，不保证模型能泛化 |
| Shaw relative | key/value 交互 | 距离对应的向量改变 score 和输出 | 论文翻译结果限于其模型和任务 |
| T5 relative bias | attention score | 距离 bucket 查标量偏置 | 远距离分桶会合并；bucket 数和边界是配置 |
| RoPE | query/key | 旋转后点积含位置差 | 代数性质不等于训练长度外的任务保证 |
| ALiBi | attention score | 按 head 加线性距离惩罚 | 偏置可扩到任意距离，不等于任意长度都有效 |

读论文或模型配置时，可以沿三个问题检查：位置线索加在哪里？对一对 token 的偏移如何计算？证据只支持公式性质、某组实验，还是具体版本的实际使用能力？这样能避免把“相对位置”“长长度可计算”和“长上下文效果好”混成一个结论。

## 7.10 自测与小结

1. 为什么没有位置输入、mask 或状态路径的 self-attention 是置换等变？“等变”与“位置不影响模型”有什么区别？
2. Learned absolute 与 Sinusoidal 都把位置加到输入表示上，它们的位置表/公式有什么不同？公式可外推能否推出能力外推？
3. Shaw 的相对向量、T5 的 bucket bias 与 RoPE 分别在哪一步注入位置？
4. 对 RoPE，写出旋转后的 QK 点积并指出其中依赖位置差的部分；这条代数性质没有证明什么？
5. ALiBi 的线性 bias 如何改变远近 key 的权重？为什么它不是远距离硬屏蔽？
6. 因果 mask 如何提供隐式顺序线索？为什么这仍不能推出任意长度上的可靠位置能力？

本章从输入位置向量、相对注意力关系、Q/K 旋转和 score 偏置四条路径解释了位置表示。Sinusoidal 的固定频率提供可计算的位置向量；learned absolute 直接学习位置表；Shaw 与 T5 用不同方式表达相对距离；RoPE 使位置差进入 QK 点积；ALiBi 以距离惩罚改变注意力分数。它们的机制可分别解释，但任何一种机制的公式都不能单独证明模型在训练范围外有可靠能力。
