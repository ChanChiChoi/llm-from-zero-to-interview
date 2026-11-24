# 第二十五章：Performer、Linear Transformer 与注意力近似

## 25.1 从精确重排到随机近似

[第二十四章](24-Linear-Attention基本思想与核技巧.md)已经说明：如果相似度 kernel 能写成有限特征内积，attention 的分子和分母都可以改变求和顺序。这个代数步骤对选定的 kernel 是精确的。Performer 要解决的是另一件事：怎样用有限计算近似标准 softmax attention 所使用的指数点积 kernel。

令 query 与 key 都是列向量，维度为 $d_k$。标准 attention 的未归一化相似度为：

```math
K_{\mathrm{SM}}(q,k)
=\exp\left(\frac{q^{\mathsf T}k}{\sqrt{d_k}}\right)
=\exp(x^{\mathsf T}y),
\qquad
x=\frac{q}{d_k^{1/4}},
\quad
y=\frac{k}{d_k^{1/4}}.
```

把缩放分到 query 和 key 两侧后，问题变成近似 $\exp(x^{\mathsf T}y)$。有限随机特征能把这个 kernel 估计成较短向量的内积，再使用第二十四章的结合律重排；随机估计本身仍有误差。这里的区别是：重排精确保留已经选择的 kernel，而 Performer 用随机特征去估计 softmax kernel。[Performer 原论文](https://arxiv.org/html/2009.14794)把这一路线称为 FAVOR+。

“随机特征”也不是一个可随意互换的公式。经典 Random Fourier Features 从指定 kernel 的谱分布构造随机映射，原始工作针对平移不变 kernel；要近似 softmax，必须选能对应指数内积的特征构造，不能把任意由随机投影矩阵 $W$ 和相位 $b$ 构成的余弦映射直接代入就称为 Performer。[Rahimi 与 Recht 的原始工作](https://papers.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)提供一般随机特征的背景，不是 FAVOR+ 的定义。

## 25.2 正值随机特征如何估计 softmax kernel

令随机方向 $\omega\sim\mathcal{N}(0,I_{d_k})$，定义一个正值特征：

```math
\psi_{\omega}(x)
=\exp\left(\omega^{\mathsf T}x-\frac{\lVert x\rVert^2}{2}\right).
```

为什么这个指数形式能恢复目标 kernel？高斯随机向量的矩母函数满足 $\mathbb{E}[\exp(\omega^{\mathsf T}u)]=\exp(\lVert u\rVert^2/2)$。把 $u=x+y$ 代入，两个特征相乘时范数项恰好抵消到内积：

```math
\begin{aligned}
\mathbb{E}_{\omega}\!\left[\psi_{\omega}(x)\psi_{\omega}(y)\right]
&=\exp\left(-\frac{\lVert x\rVert^2+\lVert y\rVert^2}{2}\right)
  \mathbb{E}_{\omega}\!\left[\exp\left(\omega^{\mathsf T}(x+y)\right)\right]\\
&=\exp\left(-\frac{\lVert x\rVert^2+\lVert y\rVert^2}{2}
            +\frac{\lVert x+y\rVert^2}{2}\right)\\
&=\exp(x^{\mathsf T}y).
\end{aligned}
```

这正是 Performer 论文 §2.3 的 softmax 正值随机特征构造；期望等式依赖上述高斯方向分布。[原论文 §2.3](https://arxiv.org/html/2009.14794)

取 $m$ 个随机方向 $\omega_1,\ldots,\omega_m$，把各个正值分量按 $1/\sqrt{m}$ 缩放并组成列向量。其内积是对 softmax kernel 的有限样本估计：

```math
\phi_m(x)
=\frac{1}{\sqrt{m}}
\begin{bmatrix}
\psi_{\omega_1}(x)\\
\vdots\\
\psi_{\omega_m}(x)
\end{bmatrix},
\qquad
\widehat K_m(x,y)
=\phi_m(x)^{\mathsf T}\phi_m(y)
=\frac{1}{m}\sum_{i=1}^{m}\psi_{\omega_i}(x)\psi_{\omega_i}(y).
```

对独立同分布的高斯方向取期望，$\widehat K_m$ 的期望等于 $K_{\mathrm{SM}}$；这不代表一次有限采样就等于真实 kernel。一个一维例子能看出两者的差别：令 $x=1,y=0$，目标值是 $1$，单个特征样本给出 $\exp(\omega-1/2)$。它对 $\omega$ 的期望为 $1$，但若这次恰好采到 $\omega=0$，样本值是 $\exp(-1/2)$。在 IID 估计中，增加 $m$ 是平均更多独立样本；FAVOR+ 的 ORF 方向并非独立，但在相应条件下仍保留无偏估计。两种采样都不会把有限特征估计变成逐次精确恒等式。

## 25.3 FAVOR+ 中正值与正交各自解决什么

FAVOR+ 的名称是 Fast Attention Via positive Orthogonal Random features。它把两项设计放在同一个随机特征估计器里，但二者作用不同。

**正值特征（Positive Random Features，PRF）**指每个分量都是正指数值，因此有限特征内积形成的未归一化分数为正，作为 attention 权重归一化时也不会因这些分数为负而抵消。它解决的是符号性质；指数仍可能因输入幅度和精度产生溢出或下溢，正值本身不等同于数值稳定保证。

**正交随机特征（Orthogonal Random Features，ORF）**指随机投影方向 $\omega_i$ 彼此正交，而不是特征向量彼此正交。在各向同性采样及论文给出的构造条件下，正交化可以保留方向的边际分布，同时降低样本间的冗余和估计误差。Performer 论文的 Theorem 2 对正值 ORF 与 IID 正值估计器给出特定的均方误差比较；其基础正交构造要求 $m\le d_k$。若需要更多方向，论文附录采用维度大小的正交块，而不是声称超过维度的全部方向仍两两正交。[论文的 ORF 定义和定理](https://arxiv.org/html/2009.14794)限定了这些条件。

正值和正交都不会消除抽样误差。前者保证分数符号，后者改善特定采样设计下的估计统计性质；二者都不意味着任意查询、任意模型和任意任务上的 attention 输出完全准确。

## 25.4 无偏 kernel 估计不等于无偏 attention 输出

softmax attention 先对同一 query 与历史各 key 的相似度求和，再用这些权重归一化 value。对因果位置 $t$，令 $x_t=q_t/d_k^{1/4}$、$y_j=k_j/d_k^{1/4}$，把特征 map 用于缩放后的 query 与 key，就得到近似 kernel 加权和。沿用[第二十四章](24-Linear-Attention基本思想与核技巧.md)的列向量约定，可写为：

```math
\widehat y_t
=\frac{\sum_{j\le t}\widehat K_m(x_t,y_j)v_j}
       {\sum_{j\le t}\widehat K_m(x_t,y_j)}
=\frac{S_t\phi_m(x_t)}
       {\phi_m(x_t)^{\mathsf T}z_t},
\qquad
S_t=\sum_{j\le t}v_j\phi_m(y_j)^{\mathsf T},
\quad
z_t=\sum_{j\le t}\phi_m(y_j).
```

这里 $q_t,k_j$ 是 query 与 key 列向量，$x_t=q_t/d_k^{1/4}$、$y_j=k_j/d_k^{1/4}$；$v_j\in\mathbb{R}^{d_v}$，$S_t\in\mathbb{R}^{d_v\times m}$，$z_t\in\mathbb{R}^{m}$。这个等式说明 FAVOR+ 怎样把有限特征估计接入前缀 attention；前缀状态的递推与训练、解码成本由[第二十四章](24-Linear-Attention基本思想与核技巧.md)说明，本节关注它估计的对象和误差。

设未归一化的准确分子、分母分别为向量 $N$ 和标量 $D$，估计值为 $\widehat N=N+\Delta N$、$\widehat D=D+\Delta D$。在 $D>0$ 且 $\widehat D>0$ 时，输出误差可改写为：

```math
\widehat y-y
=\frac{\widehat N}{\widehat D}-\frac{N}{D}
=\frac{\Delta N-y\,\Delta D}{\widehat D},
\qquad
y=\frac{N}{D}.
```

因此，即使每个未归一化 kernel 分数的估计无偏，分子和分母组成的比值也一般不会无偏；分母偏差还会与分子误差共同改变输出。若估计分母很小，输出误差会被放大。即使某层的 attention 近似误差不大，经过多个 head、MLP 和后续层后也可能继续传播。Performer 论文的多层实验专门观察了这种传播，因此不能用 kernel 的平均均方误差单独代表最终任务误差。[论文 §4.3](https://arxiv.org/html/2009.14794)报告的是其模型和训练设置下的现象。

## 25.5 理论保证和实验结果各自覆盖什么

“Performer 有误差界”不是一个不带条件的统一结论。论文的 Theorem 2 比较特定正值 ORF kernel estimator 与 IID 正值 estimator 的 MSE，并给出改进项；比较对象是 kernel 估计器，且基础 ORF 设定包含 $m\le d_k$ 条件。它不直接保证归一化后的 $\widehat y_t$ 与 dense softmax 输出有同样的误差界。

论文另一个常被混淆的结果是 Theorem 4：它对带 ORF 的三角函数特征估计器给出 attention matrix 的 uniform convergence 结论，要求 query/key 的 $L_2$ 范数有上界 $R$。定理中的随机方向数量随维度、范数半径和目标精度变化，而不随序列长度 $L$ 变化；但它讨论的是该特定估计器对 attention matrix 的近似，不能移作正值 FAVOR+、归一化输出或语言模型质量的普遍保证。[论文 §3](https://arxiv.org/html/2009.14794)把这些估计器与条件分别列出。

实验同样需要按对象和条件阅读。论文 Figure 4 在 $L=4096,d=16$ 下改变特征数，并用 $15$ 组适当归一化的随机矩阵输入比较估计误差；在该设置里，正交特征误差低于 IID 特征，正值特征误差低于三角函数特征。§4.3 进一步报告随机特征选择、周期性重采样以及 LM1B、PG-19 实验中的训练表现。前者是特定随机输入上的估计误差对照，后者绑定其模型、数据和训练做法，二者都不能推出所有任务增加 $m$ 后质量都会按同一曲线提升。[论文 §4.2—4.3](https://arxiv.org/html/2009.14794)

## 25.6 固定特征数下的计算成本与精确 attention 边界

特征数记为 $m$，query/key 宽度为 $d_k$，value 宽度为 $d_v$。按一般稠密随机投影计算特征映射，并计入特征与 value 的聚合，固定 $m$ 时单头的主要 attention 计算量关于序列长度 $T$ 为：

```math
C_{\mathrm{attn}}
=O(Tm d_k+Tm d_v),
\qquad
M_{\mathrm{causal\ state}}
=O(m d_v+m).
```

线性增长依赖 $m$ 固定；若为了目标精度随维度或输入范围提高特征数，常数和总成本也会改变。状态大小只描述因果累计状态，不含随机方向、整段训练的中间量、$Q/K/V$ 投影和 Transformer 其他层。渐近复杂度说明运算规模怎样随维度变化，不能单独推出某种 GPU 上的 wall-clock 延迟或整体模型吞吐。

Performer 与 FlashAttention 改变的是不同环节。Performer 用随机特征估计 softmax kernel，因而改变了 attention 的有限计算表达并引入估计误差；FlashAttention 保留 dense softmax attention 的数学目标，主要重排分块计算和显存读写，详见[第二十章](20-FlashAttention与PagedAttention.md)。比较时要分别说明“近似数学计算”与“精确 attention 的 IO 优化”，不能只把两者并列为一种“线性 attention”。本章讨论的是估计性质和成本口径，不据渐近符号给出特定设备或产品的性能结论。
