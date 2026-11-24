# 第二十四章：Linear Attention 的基本思想和核技巧

## 24.1 注意力权重依赖什么

标准因果 attention 在位置 $t$ 读取当前及以前的键值对。$q_t$ 是当前位置的 query，$k_j$ 和 $v_j$ 是位置 $j$ 的 key 与 value。点积先给出相似度，再经 softmax 转成权重，最后对 value 求加权和：

```math
y_t =
\frac{\sum_{j=1}^{t}\exp(q_t^{\mathsf T} k_j / \sqrt{d_k})\,v_j}
{\sum_{j=1}^{t}\exp(q_t^{\mathsf T} k_j / \sqrt{d_k})}.
```

分母让同一位置的权重归一化为 $1$。这个定义准确表达了标准 scaled dot-product attention 的因果版本，但每个 query 都要分别和前缀中的 keys 配对；处理长序列时，配对数量随序列长度平方增长。[第十章](10-o-n2-attention的计算与显存瓶颈.md)已解释这项成对计算及其实现显存口径。本章研究的问题更具体：在注意力权重能由有限维特征内积表达时，能否先汇总历史，再让 query 读取汇总结果？

## 24.2 核特征分解如何改变求和顺序

假设所选的相似度函数能够写成 query 特征和 key 特征的内积：

```math
K(q,k)=\phi(q)^{\mathsf T}\phi(k).
```

把它代入归一化 attention。利用标量乘法和矩阵乘法的结合律，分子与分母可以分别改写：

```math
\begin{aligned}
\sum_{j=1}^{t}K(q_t,k_j)v_j
&=\left(\sum_{j=1}^{t}v_j\phi(k_j)^{\mathsf T}\right)
  \phi(q_t),\\
\sum_{j=1}^{t}K(q_t,k_j)
&=\phi(q_t)^{\mathsf T}
  \left(\sum_{j=1}^{t}\phi(k_j)\right).
\end{aligned}
```

括号中的两个历史汇总都不依赖当前 query：第一项累计 key 特征与 value 的外积，第二项累计 key 特征本身。逐个 query 计算所有配对的工作，就变成先形成前缀汇总、再用当前 query 特征读出。这是代数重排的来源；它对已经选定的 $K$ 是精确恒等式。[Katharopoulos 等人](https://proceedings.mlr.press/v119/katharopoulos20a.html)从这一步导出因果线性 attention 的计算形式；[David Chiang 的课程讲义](https://academicweb.nd.edu/~dchiang/teaching/tonn/2024/chapter9.pdf)也沿相同关系连接核特征、前缀和与递归网络。

这里的“精确”有明确范围：重排本身不会引入误差，但它精确保留的是 $K$ 所定义的 attention。如果 $K$ 不等于标准 softmax 的指数点积核，重排后的结果就不是标准 softmax attention。近似 softmax 的特征构造属于下一章的 Performer 与随机特征路线。

## 24.3 因果 attention 如何成为前缀状态

令 $\phi(q_t),\phi(k_t)\in\mathbb{R}^{r}$，并令 $v_t\in\mathbb{R}^{d_v}$。把截至位置 $t$ 的两项前缀汇总记为 $S_t$ 与 $z_t$：

```math
S_t=\sum_{j=1}^{t}v_j\phi(k_j)^{\mathsf T}
\in\mathbb{R}^{d_v\times r},
\qquad
z_t=\sum_{j=1}^{t}\phi(k_j)
\in\mathbb{R}^{r}.
```

每读入一个新位置，只需把它的贡献加到上一时刻状态。输出再用当前 query 特征读取这两个状态：

```math
\begin{aligned}
S_t &= S_{t-1}+v_t\phi(k_t)^{\mathsf T},\\
z_t &= z_{t-1}+\phi(k_t),\\
y_t &= \frac{S_t\phi(q_t)}
             {\phi(q_t)^{\mathsf T}z_t},
\qquad S_0=0,\quad z_0=0.
\end{aligned}
```

分子状态和归一化状态必须一起更新。$S_t$ 的每一列累积一个特征方向对应的 value 内容，$z_t$ 记录相同方向的权重总量；query 与它们相乘后，分子得到一个 $d_v$ 维向量，分母得到一个标量。若只保存分子而不保存分母，就丢失了原定义中的归一化。

看一个明确标作合成的两位置算例。令 $\phi(k_1)=(1,0)^{\mathsf T}$、$v_1=(2,1)^{\mathsf T}$，令 $\phi(k_2)=(0,1)^{\mathsf T}$、$v_2=(0,3)^{\mathsf T}$。读入第一个位置后，$S_1$ 只含 $v_1$ 在第一特征方向的贡献，$z_1=(1,0)^{\mathsf T}$。第二个位置到来时，状态只增加它自己的外积和 key 特征：

```math
S_2=
\begin{bmatrix}
2 & 0\\
1 & 3
\end{bmatrix},
\qquad
z_2=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
```

若当前位置的 query 特征为 $\phi(q_2)=(1,1)^{\mathsf T}$，它对两个特征方向给出相同权重，于是分子为 $(2,4)$、分母为 $2$，输出为 $(1,2)^{\mathsf T}$，也就是两个 value 的平均。这个例子只展示给定特征值时的状态更新，不是 softmax 的数值近似，也不代表训练出来的特征。

## 24.4 线性长度从哪些维度得来

标准 attention 显式处理 query 与 key 的位置对；线性形式则先计算每个 token 的特征，再把贡献写入大小为 $r\times d_v$ 的状态。因此，对固定的 $r$、$d_k$ 和 $d_v$，随序列长度 $T$ 增长的核心工作为线性：

- 为每个 token 生成 $r$ 维 query/key 特征的代价取决于 $\phi$；若映射包含稠密的 $d_k\to r$ 线性投影，该部分约为 $O(Td_kr)$。
- 更新分子状态并读取输出的代价约为 $O(Trd_v)$。
- Q/K/V 投影、输出投影和其余 Transformer 层仍有各自成本，不包含在上述 attention 汇总项里。

逐 token 自回归解码时，每个 attention head 保存一个 $S_t$ 和一个 $z_t$；其状态大小为 $O(rd_v+r)$，不随已生成 token 数增加。实际总量还要乘以层数、head 数和 batch，并与模型权重、临时缓冲区及其他层状态一起计算。这里的定长结论只描述这类 attention 的历史汇总，不是整模型的固定内存承诺。

训练时整段序列已知，可以利用前缀和的结合性做并行 scan 或分块计算；自回归解码则按 token 递推。两种执行方式解决不同的并行问题。训练反向传播需要哪些中间量，取决于具体算法、检查点和重计算策略；直接从前向递推式不能推出所有训练实现都有常数显存。原论文为其方法推导了特定的累计和反向计算。固定宽度下序列长度的渐近复杂度，也不能单独保证实际延迟优于 dense attention：特征宽度、投影、内存访问、kernel 和硬件都会改变结果。[FlashAttention](20-FlashAttention与PagedAttention.md)则保留完整 softmax 配对语义并优化数据搬运，两条路线改写的对象不同。

## 24.5 有限特征与 softmax 的边界

有限维特征映射可以直接定义一个便于重排的正值相似度。例如，Katharopoulos 等人的一种实验特征映射为 $\phi(x)=\mathrm{ELU}(x)+1$。它产生正值特征，使内积 kernel 可用于归一化 attention；这并不意味着它等于指数点积 softmax kernel。指数点积核的精确特征表示是无限维的，有限宽度的映射若要接近它，就需要近似方法；另一种可能是直接采用不同的 kernel。两种做法改变的东西不同：前者追求逼近指定 softmax，后者定义另一种注意力相似度。

归一化还要求分母有定义。采用非负相似度、至少有一个有效 key 且相似度总量非零时，权重才可归一化；正值特征是保证这项条件的一种做法。全被 mask 的前缀或数值上接近零的分母，需要实现规定明确策略。加上 $\epsilon$ 可以避免除零，但会改变原来的函数，不能当作从代数推导自动得到的恒等步骤。

[Performer 论文](https://arxiv.org/abs/2009.14794v4)中的 FAVOR+ 使用正交随机特征近似 softmax kernel，并研究这种近似的估计误差与方差；这些保证和误差如何进入 attention 输出，取决于该方法的特征构造及其假设，见[下一章](25-Performer与Linear-Transformer.md)。本章只需要记住：重排的代数、kernel 的选择和 kernel 的近似误差是三个不同问题。

## 24.6 定长状态保留什么，又不能直接取回什么

状态 $S_t,z_t$ 把前缀中的 key/value 信息压到有限个特征方向上。对当前 query，它们可以给出由所选 kernel 加权的 value 汇总；但状态中没有逐 token 保留原始的每个 key/value 条目，因而不像逐项存储的 KV cache 那样能任意单独定位并返回某个历史条目。随着 token 增长，状态形状不变，保留信息的方式由特征映射、模型参数和任务共同决定。固定状态不等于无条件能记住每个细节，也不能只凭状态大小判断某个任务一定成功或失败。

本书[第十一章](11-kv-cache长上下文推理和显存增长问题.md)解释逐 token KV cache 的存储与增长；本章的线性状态则是对历史的特征化累计。两者都承接过去信息，但保存对象和读取方式不同。它也不等于所有 RNN 或 SSM：线性 attention 的状态更新有明确的 key 特征外积与 query 读出结构；[第 29 章](29-SSM数学直觉.md)和[第 30 章](30-SSM-RNN-CNN-Attention信息流对比.md)讨论的状态空间和其他递归模型采用各自的状态动力学。读到“线性复杂度”时，应继续确认 feature map 是否有限、归一化如何处理、固定状态保留了什么，以及复杂度是否把投影和执行条件计入。
