# mHC：用双随机矩阵约束残差流

> 本章讨论 mHC（Manifold-Constrained Hyper-Connections）怎样在多路残差流中限制层间混合。mHC 于 ICML 2026 发表；定义、参数化、实验设置和开销数字按论文公开全文核对（[PMLR 出版页](https://proceedings.mlr.press/v306/xie26aa.html)，[arXiv 全文](https://arxiv.org/html/2512.24880)）。前身 Hyper-Connections（HC）见[原论文](https://arxiv.org/abs/2409.19606)。

本章用一个 token 在某层的状态表示为矩阵 $\mathbf{X}_{l}\in\mathbb{R}^{n\times C}$ ：每一行是一条残差流， $n$ 是流数， $C$ 是模型隐藏维度。关键问题是：如何让这些流可以交互，同时限制残差路径在很多层连乘时不断放大或压小。

## 78.1 从标准残差到多路残差：mHC 改变了哪条路径

### 78.1.1 标准残差保留的是一条直接恒等路径

标准残差层可以写成 $x_{l+1}=x_l+F_l(x_l;\mathcal{W}_l)$ 。输入 $x_l$ 有一条不经过主变换的直接路径进入下一层；主变换 $F_l$ 再把本层计算结果加回来。这个结构说明残差支路提供了什么，并不表示整层输出或梯度在所有情况下都保持不变， $F_l$ 、归一化和训练设置仍然会影响它们。

Hyper-Connections 关注的正是这条连接路径：标准结构固定地把当前向量送到下一层，而 HC 让多条残差流可以根据当前状态学习如何相互混合，并把主变换的输出写回不同流。

### 78.1.2 Hyper-Connections 把一条流扩成多条可学习流

HC 将单条 $C$ 维残差流扩成 $n$ 条并行流，合在一起的状态形状为 $n\times C$ 。扩展发生在层间残差表示上；注意力或前馈网络等主变换仍接收宽度为 $C$ 的输入，并输出宽度为 $C$ 的结果。

因此 HC 需要两类连接：把多流状态整理成主变换可用的输入，以及把主变换结果分配回多条流；与此同时，它还学习残差支路本身怎样混合这些流。多流增加了层间信息路由的自由度，也把原来固定的恒等残差路径改成了可学习映射。

## 78.2 先看 HC 的整层数据流，再看深度风险

### 78.2.1 H_pre、H_post 与 H_res 分别承担读入、写回和流间混合

对第 $l$ 层，HC 的一层更新可写为：

~~~math
\mathbf{X}_{l+1}
=
\mathcal{H}^{\mathrm{res}}_{l}\mathbf{X}_{l}
+
\mathcal{H}^{\mathrm{post}\,\top}_{l}
F_l\!\left(
\mathcal{H}^{\mathrm{pre}}_{l}\mathbf{X}_{l};
\mathcal{W}_{l}
\right).
~~~

其中 $\mathcal{H}^{\mathrm{res}}_{l}\in\mathbb{R}^{n\times n}$ 混合残差流； $\mathcal{H}^{\mathrm{pre}}_{l}\in\mathbb{R}^{1\times n}$ 把 $n$ 条流聚合成 $F_l$ 所需的单路输入； $\mathcal{H}^{\mathrm{post}}_{l}\in\mathbb{R}^{1\times n}$ 把 $F_l$ 的单路输出分配回 $n$ 条流。前两者的结果仍是 $C$ 维，乘上 $\mathcal{H}^{\mathrm{post}\,\top}_{l}$ 后成为 $n\times C$ ，才能和残差项相加。

三组系数都可以随层和 token 改变。它们的职责不同： $\mathcal{H}^{\mathrm{res}}$ 控制原残差信息怎样继续传播， $\mathcal{H}^{\mathrm{pre}}$ 控制主变换读取哪些流， $\mathcal{H}^{\mathrm{post}}$ 控制主变换结果写回哪些流。mHC 的双随机约束只施加在 $\mathcal{H}^{\mathrm{res}}$ ，不能把三者统称为双随机映射。[HC 原论文](https://arxiv.org/abs/2409.19606)与 [mHC 论文全文](https://arxiv.org/html/2512.24880)给出了这一层级的定义和张量形状。

### 78.2.2 跨层复合映射解释了放大与衰减怎样累积

先只跟踪残差支路，不看每层新加入的 $F_l$ 输出。从第 $l$ 层传到第 $L$ 层的这部分状态，经过一串残差混合矩阵：

~~~math
\mathbf{X}_{L}^{\mathrm{skip}}
=
\mathcal{H}^{\mathrm{res}}_{L-1}
\mathcal{H}^{\mathrm{res}}_{L-2}
\cdots
\mathcal{H}^{\mathrm{res}}_{l}
\mathbf{X}_{l}.
~~~

当这些矩阵不受约束时，某些方向可能在连乘后放大，另一些方向可能缩小；这会影响沿深度传播的信号以及反向传播的梯度。mHC 论文在 HC 的大规模实验中观察到复合映射的增益异常，因此把残差映射作为稳定性分析的对象。这个问题是针对未经约束的映射提出的风险，不能理解成每个 HC 模型或每个训练过程必然发散。

### 78.2.3 表达路径更灵活，也让显存访问和流水线通信变重

扩展流数意味着每个 token 要保存和搬运更宽的残差状态。生成动态映射还需要读取多流表示；训练反向传播需要保留或重算相关中间量；流水线并行则要在 stage 之间传递多流激活。即使主变换 $F_l$ 的主体 FLOPs 不因流数按比例增加，额外的内存流量和通信仍可能降低训练吞吐。

这些成本来自残差表示和连接计算，不能仅用矩阵乘法 FLOPs 判断。第 78.8 节按论文的单层 I/O 账本展开具体数字。

## 78.3 mHC 约束的是残差混合矩阵的几何集合

### 78.3.1 非负和行和为一使每条输出流成为凸组合

mHC 把残差混合矩阵放入双随机矩阵集合，也称 Birkhoff 多面体：

~~~math
\mathcal{H}^{\mathrm{res}}_{l}\mathbf{1}_{n}=\mathbf{1}_{n},
\qquad
\mathbf{1}_{n}^{\top}\mathcal{H}^{\mathrm{res}}_{l}=\mathbf{1}_{n}^{\top},
\qquad
\mathcal{H}^{\mathrm{res}}_{l}\geq 0.
~~~

第一条条件表示每一行权重和为 $1$ 。由于元素非负，第 $i$ 条输出流是若干输入流的加权平均，而不是正负信号互相抵消后的任意线性组合。例如两流情况下，矩阵

~~~math
\begin{bmatrix}
0.7 & 0.3\\
0.3 & 0.7
\end{bmatrix}
~~~

会让第一条输出流取输入流的 $70\%$ 与第二流的 $30\%$ ，第二条输出流则反过来。这一解释只适用于残差混合项。

### 78.3.2 列和为一保存流轴总和与均值

第二条条件要求每一列之和也为 $1$ 。如果将每个隐藏维度单独看作一组跨流数值，那么残差项满足：

~~~math
\mathbf{1}_{n}^{\top}
\mathcal{H}^{\mathrm{res}}_{l}\mathbf{X}_{l}
=
\mathbf{1}_{n}^{\top}\mathbf{X}_{l}.
~~~

也就是说，经过 $\mathcal{H}^{\mathrm{res}}$ 后，每个隐藏维度沿流轴的总和和均值保持不变。两流示例中的输入与输出总和因此相等。这里保持的是流轴上的总量或均值，不是每条流的值，也不是完整 Transformer 层输出的均值。

### 78.3.3 Birkhoff 多面体把双随机矩阵写成置换矩阵的凸组合

Birkhoff–von Neumann 定理说明，任意双随机矩阵都可以表示成置换矩阵的凸组合：

~~~math
\mathcal{H}^{\mathrm{res}}_{l}
=
\sum_{k}\lambda_{k}\mathbf{P}_{k},
\qquad
\lambda_{k}\geq 0,
\qquad
\sum_{k}\lambda_{k}=1.
~~~

置换矩阵只会重新排列流；凸组合则在若干种排列之间混合。这个几何解释说明双随机约束允许流间交互，并不等价于把每一流锁死为恒等路径。当 $n=1$ 时，可行矩阵退化为标量 $1$ ，才恢复标准残差的恒等映射。

## 78.4 双随机约束有哪些严格性质，又不代表什么

### 78.4.1 谱范数至多为一，所以残差映射不会放大二范数

双随机矩阵可写成置换矩阵的凸组合，而置换矩阵是正交矩阵，二范数保持不变。因此双随机矩阵的谱范数至多为 $1$ ：

~~~math
\left\lVert\mathcal{H}^{\mathrm{res}}_{l}\right\rVert_{2}\leq 1.
~~~

它作用于多流状态时不会放大流轴上的二范数；相应的 Frobenius 范数也满足不增界。这个性质叫非扩张。它只给出增益的上界，不保证每个方向的范数都保持原值，也没有给出大于零的统一下界。

### 78.4.2 精确双随机矩阵的乘积仍双随机

两个双随机矩阵相乘后仍为双随机矩阵，因此如果每层的 $\mathcal{H}^{\mathrm{res}}$ 都严格满足约束，残差支路的跨层复合矩阵也满足同样约束。配合谱范数上界，这解释了 mHC 为什么能限制残差流混合在深度方向上的累积放大。

这一结论的前提是每层矩阵确实双随机。理论上的可行集合有严格性质，实际训练使用的矩阵则来自有限步数值算法；两者需要分开说明。

### 78.4.3 非扩张不等于恒范数，也不等于整网稳定

双随机矩阵可以收缩不同流之间的差异。例如 $n\times n$ 的均匀矩阵也是双随机的，它会把每条输出流变成所有输入流的平均，令流间差异消失。因此双随机约束既不保证每条流始终不同，也不防止流间过度混合。

而且，整层还包含 $\mathcal{H}^{\mathrm{post}\,\top}F_l(\mathcal{H}^{\mathrm{pre}}\mathbf{X})$ 这一主分支。 $\mathcal{H}^{\mathrm{pre}}$ 与 $\mathcal{H}^{\mathrm{post}}$ 的大小、主变换的 Jacobian、归一化、优化器、有限精度和训练数据都会影响实际梯度与激活。mHC 约束的是残差矩阵，不是整个 Transformer 的 Lipschitz 常数或训练稳定定理。[论文的理论性质](https://arxiv.org/html/2512.24880)也应按残差映射这一范围阅读。

## 78.5 从输入相关 logits 到 mHC 的受约束映射

### 78.5.1 归一化后的多流状态生成三组动态系数

mHC 沿用输入相关的映射方式：先把当前 token 的多流状态展平并做 RMSNorm，再由可学习投影和偏置生成三组 logits。其形状关系为 $\mathbf{X}_{l}\in\mathbb{R}^{n\times C}$ 、 $\mathrm{vec}(\mathbf{X}_{l})\in\mathbb{R}^{1\times nC}$ ； $\varphi^{\mathrm{pre}}_{l}$ 和 $\varphi^{\mathrm{post}}_{l}$ 的形状为 $nC\times n$ ， $\varphi^{\mathrm{res}}_{l}$ 为 $nC\times n^{2}$ 。后者的输出再重排为 $n\times n$ 。

用公式概括参数化过程：

~~~math
\begin{aligned}
\mathbf{x}'_{l}
&=
\mathrm{RMSNorm}\!\left(\mathrm{vec}(\mathbf{X}_{l})\right),\\
\widetilde{\mathcal{H}}^{\mathrm{pre}}_{l}
&=
\alpha^{\mathrm{pre}}_{l}
(\mathbf{x}'_{l}\varphi^{\mathrm{pre}}_{l})
+\mathbf{b}^{\mathrm{pre}}_{l},\\
\widetilde{\mathcal{H}}^{\mathrm{post}}_{l}
&=
\alpha^{\mathrm{post}}_{l}
(\mathbf{x}'_{l}\varphi^{\mathrm{post}}_{l})
+\mathbf{b}^{\mathrm{post}}_{l},\\
\widetilde{\mathcal{H}}^{\mathrm{res}}_{l}
&=
\alpha^{\mathrm{res}}_{l}
\mathrm{mat}(\mathbf{x}'_{l}\varphi^{\mathrm{res}}_{l})
+\mathbf{b}^{\mathrm{res}}_{l}.
\end{aligned}
~~~

所以这些系数不是对所有 token 固定的一张矩阵；输入状态会参与决定本层的读入、写回与残差混合。论文说明了投影矩阵、偏置和可学习缩放项的形状，具体实现再将它们融合进相应 kernel。[参数化定义见 mHC 论文第 4.2 节](https://arxiv.org/html/2512.24880)。

### 78.5.2 只把 H_res 投到双随机集合，H_pre/H_post 仅做非负映射

三组 logits 经不同变换得到最终映射：

~~~math
\begin{aligned}
\mathcal{H}^{\mathrm{pre}}_{l}
&=
\sigma\!\left(\widetilde{\mathcal{H}}^{\mathrm{pre}}_{l}\right),\\
\mathcal{H}^{\mathrm{post}}_{l}
&=
2\sigma\!\left(\widetilde{\mathcal{H}}^{\mathrm{post}}_{l}\right),\\
\mathcal{H}^{\mathrm{res}}_{l}
&=
\mathrm{SinkhornKnopp}\!\left(\widetilde{\mathcal{H}}^{\mathrm{res}}_{l}\right).
\end{aligned}
~~~

$\mathcal{H}^{\mathrm{pre}}$ 与 $\mathcal{H}^{\mathrm{post}}$ 因 sigmoid 而非负，但它们不是双随机矩阵，行和也没有被固定为 $1$ 。 $\mathcal{H}^{\mathrm{res}}$ 才经过 Sinkhorn–Knopp 处理以满足双随机约束。不能把主分支系数的非负性说成残差映射的双随机性质。

### 78.5.3 Sinkhorn 先列后行归一化，有限 20 轮仍是近似

Sinkhorn–Knopp 从正矩阵开始。mHC 先对残差 logits 取指数，再反复按列归一化、按行归一化。若 $\mathcal{T}_{c}$ 表示把每列缩放到列和为 $1$ ， $\mathcal{T}_{r}$ 表示把每行缩放到行和为 $1$ ，则论文写为：

~~~math
\mathbf{M}^{(0)}
=
\exp\!\left(\widetilde{\mathcal{H}}^{\mathrm{res}}_{l}\right),
\qquad
\mathbf{M}^{(t)}
=
\mathcal{T}_{r}\!\left(
\mathcal{T}_{c}(\mathbf{M}^{(t-1)})
\right).
~~~

每轮先做内层的列归一化，再做外层行归一化。为什么要交替？列归一化会改变行和，行归一化又会改变列和，所以通常要反复调整。正矩阵的迭代在理想收敛时得到双随机缩放；论文用 $t_{\max}=20$ 作为实践配置，这只是有限轮近似，不等于数值上严格满足两组和约束。[算法次序与轮数来自论文第 4.2 节](https://arxiv.org/html/2512.24880)。

用一个二乘二的正权重矩阵看一次迭代。令 logits 为 $\begin{bmatrix}\ln 4&0\\0&0\end{bmatrix}$ ，指数变换后得到正矩阵 $\mathbf{M}^{(0)}=\begin{bmatrix}4&1\\1&1\end{bmatrix}$ 。列归一化后为 $\begin{bmatrix}0.8&0.5\\0.2&0.5\end{bmatrix}$ ；再按行归一化得到：

~~~math
\mathbf{M}^{(1)}
=
\begin{bmatrix}
0.6154 & 0.3846\\
0.2857 & 0.7143
\end{bmatrix}.
~~~

此时两行之和已为 $1$ ，但两列之和约为 $0.9011$ 和 $1.0989$ ，说明一轮不能同时精确满足行列约束。继续交替归一化会把结果推向该例的双随机极限 $\begin{bmatrix}2/3&1/3\\1/3&2/3\end{bmatrix}$ 。这是教学计算，不是 mHC 论文中的训练矩阵或实际输出。直接实现时还要处理指数值范围、精度和反向传播；本章不把简化循环冒充生产实现。

## 78.6 mHC 的三组映射在一层中共同作用

### 78.6.1 残差混合把输入流映射到残差候选

仍用两条标量流说明数据怎样经过一层。假设某 token 当前状态为 $\mathbf{X}_{l}=\begin{bmatrix}10\\4\end{bmatrix}$ ，并假设当前输入生成的残差映射是 $\begin{bmatrix}0.7&0.3\\0.3&0.7\end{bmatrix}$ 。这个矩阵每行、每列都和为 $1$ ，因此残差分支得到：

~~~math
\mathcal{H}^{\mathrm{res}}_{l}\mathbf{X}_{l}
=
\begin{bmatrix}
0.7\times 10+0.3\times 4\\
0.3\times 10+0.7\times 4
\end{bmatrix}
=
\begin{bmatrix}
8.2\\
5.8
\end{bmatrix}.
~~~

两条流不再各自等于原输入，但总和仍为 $14$ 。这一步具体展示了 mHC 的约束如何限制残差流混合，同时保留非平凡的信息交换。

### 78.6.2 前映射与后映射将标准子层嵌入多流布局

接着假设 $\mathcal{H}^{\mathrm{pre}}_{l}=[0.3,0.6]$ ，主变换收到的单路输入是 $0.3\times 10+0.6\times4=5.4$ 。在这组教学设定中，令主变换输出 $F_l(5.4)=2$ ，并令 $\mathcal{H}^{\mathrm{post}}_{l}=[1.2,0.8]$ ，写回项就是 $\begin{bmatrix}2.4\\1.6\end{bmatrix}$ 。

实际模型会根据 token 和层状态生成这些系数， $F_l$ 也可能是注意力或 FFN。数字只是让读者看到同一个宽度为 $C$ 的子层输入和输出如何接入 $n$ 条流；不能把这组系数或输出当作论文参数、训练样本或性能证据。

### 78.6.3 流轴守恒只适用于残差项

把主变换写回后，下一层状态为 $\begin{bmatrix}8.2\\5.8\end{bmatrix}+\begin{bmatrix}2.4\\1.6\end{bmatrix}=\begin{bmatrix}10.6\\7.4\end{bmatrix}$ 。完整层的流总和从 $14$ 变成 $18$ ，因为子层写入增加了 $4$ ；列和守恒只约束 $\mathcal{H}^{\mathrm{res}}\mathbf{X}$ ，并没有约束 $\mathcal{H}^{\mathrm{post}\,\top}F(\cdot)$ 。

这个区别是理解 mHC 保证范围的关键：双随机性使跳连中的多流混合有明确的几何边界，不会把包含主分支的整层输出变成守恒系统。训练稳定性还取决于主分支和其他参数化细节。

## 78.7 论文实验支持什么样的训练收益

### 78.7.1 主实验使用 n=4 的 MoE 模型并覆盖多种训练规模

mHC 论文主要使用受 DeepSeek-V3 启发的 MoE 架构，HC 与 mHC 的扩展率设为 $n=4$ 。主实验为 $27\mathrm{B}$ 模型；作者另以 $3\mathrm{B}$ 和 $9\mathrm{B}$ 模型观察 compute scaling，并训练一个固定数据量为 $1$ 万亿 tokens 的 $3\mathrm{B}$ 模型分析 token scaling。论文还比较标准残差、HC 和 mHC，因而提供了方法间训练对照，而不只是架构定义。[实验设置见论文第 5.1 节](https://arxiv.org/html/2512.24880)。

### 78.7.2 27B 损失、下游表格与传播增益是不同证据

论文报告 $27\mathrm{B}$ 主实验中，mHC 相对标准残差基线的最终训练 loss 下降 $0.021$ 。下游表格按各任务自己的指标与 few-shot 数量评估：例如 BBH 3-shot EM 为 baseline $43.8$ 、HC $48.9$ 、mHC $51.0$ ；DROP 3-shot F1 为 $47.0$ 、 $51.6$ 、 $53.9$ 。但 MATH 4-shot EM 中 mHC 为 $26.0$ ，略低于 HC 的 $26.4$ 。所以应按任务读表，不能把一项平均成“每项都胜过 HC”。

稳定性图也是另一种证据：在论文所示的 $27\mathrm{B}$ 复合残差映射中，HC 的反向传播增益峰值接近 $3000$ ，mHC 约为 $1.6$ ；论文将偏离归因于有限轮 Sinkhorn 近似。该图说明特定模型和设置下的映射增益得到控制，不是任意模型都满足的训练保证。以上损失、benchmark 与稳定性数值均为作者报告，见[论文第 5.2、5.4 节及 Table 4](https://arxiv.org/html/2512.24880)。

### 78.7.3 论文结果仍受模型、数据、指标和实现条件限制

这些实验说明 mHC 在作者的模型规模、数据量、训练配置和评测协议中可改善训练动态，并取得有竞争力的下游结果。它们没有证明 mHC 对所有模型、任务、扩展率和硬件都更好；也没有把残差约束本身与所有可能的训练 recipe、优化器或 kernel 变化分离。

因此，把论文结果用于工程决策时要问清楚基线是什么、模型和 token 预算是否匹配、质量指标是否同一口径，以及运行开销是否来自相同实现。论文没有覆盖的设置，应当视为待验证，而不是默认继承结果。

## 78.8 mHC 的系统账本包括显存流量和通信

### 78.8.1 宽残差流让每层读写量随扩展率 n 增长

论文估算了一个 pre-norm Transformer 残差层、每个 token 的残差路径内存读写量，且不计 $F_l$ 主体内部的 I/O：

| 方法 | 读取元素数 | 写入元素数 |
| --- | ---: | ---: |
| 标准残差 | $2C$ | $C$ |
| HC 多流残差结构（mHC 沿用） | $(5n+1)C+n^2+2n$ | $(3n+1)C+n^2+2n$ |

当 $n=4$ 时，残差路径约读取 $21C+24$ 个元素、写入 $13C+24$ 个元素；而标准残差约为 $2C$ 读取、 $C$ 写入。这里的量是论文对 HC 多流残差结构的单层、逐 token I/O 账本，不是 mHC 完整训练路径、FLOPs 或带完整 attention/FFN、硬件和 batch 信息的端到端延迟。mHC 还要计算 Sinkhorn 映射，其成本需结合后续融合与重计算实现测量。[公式和边界见论文 Table 2](https://arxiv.org/html/2512.24880)。

### 78.8.2 融合、重计算和 DualPipe 负责收回系统开销

论文的优化把高维状态读取、系数生成和若干残差操作融合到专用 kernel 中；Sinkhorn 迭代也采用融合 kernel，反向传播通过在片上重算中间迭代减少显存保存。训练系统再用选择性激活重计算控制内存占用，并将 stage 边界与重计算范围对齐；在 DualPipe 中重叠部分 pipeline 通信与计算。

这些措施不是 mHC 的数学定义，而是让多流结构能在特定训练系统上运行的实现工作。一个朴素张量实现，即使数值上得到双随机矩阵，也不能据此推导出论文的吞吐或 wall-time。

### 78.8.3 6.7% 是 n=4 大规模实现的条件化时间结果

作者报告在 $n=4$ 的大规模训练实现中，mHC 增加约 $6.7\%$ 的训练时间开销。这个数字依赖其融合、重计算、pipeline overlap、模型与训练栈；论文报告的是优化后的训练时间 overhead，不是 Sinkhorn 单 kernel 的开销，也不是任何硬件和软件栈都能复现的常数。[工程章节与开销报告见论文第 4.3 节](https://arxiv.org/html/2512.24880)。

若要在新环境验证，应同时对齐模型规模、序列长度、batch、并行策略和硬件，然后测量质量、训练吞吐、峰值激活/显存、stage 间通信与残差路径带宽。只复现 6.7% 这个百分比而没有相同的负载与系统边界，没有可比意义。

## 78.9 放回相邻架构并设计可解释的对照

### 78.9.1 AttnRes、KDA 与 CSA/HCA 改的是不同信息轴

mHC 在同一网络深度上维护多条残差流，并约束它们沿流轴的混合；[Attention Residuals](75-attention-residuals深度方向注意力.md)沿网络深度选择并聚合历史表示；[KDA](61-kda递归注意力.md)沿 token 序列递归更新有限状态；[CSA/HCA](74-csa-hca压缩注意力.md)改变长序列 KV 的压缩和读取方式。它们都影响信息如何传播，但状态轴、更新机制和成本对象不同。

因此不能把 AttnRes 的深度读取、KDA 的序列状态或 CSA/HCA 的 cache 压缩当作 mHC 的子机制；对比时也不能只按“都改了信息流”混成一种架构。具体模型如何组合这些设计，回到其论文和配置逐项核对，例如[第 77 章](77-csa-hca从压缩kv到百万上下文.md)只把 V4 的 mHC 作为相邻机制引用。

### 78.9.2 mHC-lite 与 cmHC 说明双随机约束仍有后续问题

双随机集合本身不规定矩阵在集合中落在哪里：它可以接近置换顶点、几乎不混流，也可以接近均匀矩阵、使不同流趋同。2026 年 11 月 20 日提交的 cmHC 预印本据此研究欠混合与过混合，并提出把双随机矩阵向多面体内部收缩的方案；作者报告了从 GPT-2 到 MoE 的实验和 Ascend NPU 上最多 $44.8\%$ 的 hyper-connection 执行延迟下降。该工作仍是新预印本，其理论、训练结果和设备数字应视为作者主张，不能当作独立复现结论。[cmHC arXiv v1](https://arxiv.org/abs/2611.27193)。

另一篇 mHC-lite 预印本用 $n!$ 个置换矩阵的 softmax 凸组合直接构造严格双随机矩阵，目标是避开有限步 Sinkhorn 的近似和专用 kernel。作者报告了其效率与稳定性实验； $n!$ 项增长意味着扩展流数时还要评估参数和计算成本。[mHC-lite arXiv v1](https://arxiv.org/abs/2601.05732)。这两项工作是当前研究线索，不是 mHC 原方法的一部分，也不是本章的完整变体综述。

### 78.9.3 公平验证需要同时测质量、传播、资源和实现

面试或方案评审时，可以沿三个问题展开：

- 双随机约束保留了什么？答：残差矩阵的非负性、行列和、流轴总量，以及精确情况下的非扩张和连乘闭包。
- 它没有保留什么？答：每条流的原值、所有方向的范数下界、整层输出的流轴总和，以及对整个训练过程的绝对稳定保证。
- 怎么验证它值不值得采用？在相同模型、数据和训练预算下对比标准残差、HC 与 mHC，并同时报告训练 loss、匹配任务的下游质量、单层/复合传播增益、有限 Sinkhorn 行列误差、吞吐、激活内存和 pipeline 通信。若再比较 AttnRes、KDA 或 CSA/HCA，应根据它们各自改变的深度轴、序列轴或 cache 目标另设对照；跨论文数字不能替代同条件实验。

这样读者既能解释 mHC 对残差流做了什么，也能说明它依赖哪些实现条件，以及哪些结果需要在自己的模型和系统上重新测量。
