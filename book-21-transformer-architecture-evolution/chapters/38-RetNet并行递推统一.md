# 第三十八章：RetNet：Retention 机制与并行/递推统一

## 38.1 RetNet 要处理的训练与生成约束

Transformer 训练时能同时处理一段序列中的多个位置，但自回归生成仍要逐 token 读取不断变长的 KV cache。普通 RNN 每步只带一个状态，生成开销不随历史长度增长，却难以直接用整段并行计算来训练。RetNet 试图把这两项需求放进同一个序列混合机制：训练时使用 parallel retention，生成时使用 recurrent retention，长序列训练还可以使用 chunkwise recurrent retention。[原论文](https://arxiv.org/html/2307.08621)把这三种计算形态作为 RetNet 的核心设计。

本章讨论的是 RetNet 论文中的 retention 机制。它在 query-key 内容交互上加入因果衰减，并把历史贡献累积进一个矩阵状态。这里的“固定状态”表示状态尺寸不随已读 token 数增长；每步仍要执行与 head 宽度有关的投影、矩阵更新和读出。并行表示也保留因果依赖，只是用矩阵运算同时计算多个位置。

## 38.2 因果衰减交互如何写成递归状态

先看单个 head。设序列长度为 $T$，每个位置的 query 和 key 宽度为 $d_k$，value 宽度为 $d_v$。RetNet 为历史交互使用一个下三角衰减矩阵：当前位置 $t$ 只能接收位置 $j\leq t$ 的信息，距离越远，贡献乘上的衰减因子越小。

令 $D$ 表示因果衰减矩阵，简化归一化时，并行形式为：

~~~math
D_{t,j}=\begin{cases}
\gamma^{t-j}, & j\leq t,\\
0, & j>t,
\end{cases}
\qquad
Y=((QK^{\mathsf T})\odot D)V.
~~~

这与 softmax attention 的权重构造不同：$D$ 给定可见范围和距离衰减，$QK^{\mathsf T}$ 仍提供内容相关的交互分数，二者逐元素相乘后再作用于 $V$。Retention score 不构成和为一的概率分布。

位置本身还通过相位旋转进入 query 和 key。下式中的 $Q$、$K$ 已按位置应用相反方向的相位；它与 $D$ 的距离衰减承担不同作用：

~~~math
\Theta_t=e^{\mathrm{i}t\theta},\qquad
q_t=(x_tW_Q)\odot\Theta_t,\qquad
k_t=(x_tW_K)\odot\overline{\Theta_t},\qquad
v_t=x_tW_V.
~~~

其中 $\theta$ 给出各维度的位置频率，$\overline{\Theta_t}$ 表示复共轭。把旋转位置写进有效的 $Q$ 和 $K$ 后，仍可沿下式追踪衰减状态。

同一个并行计算可以改写为递归。令 $S_t\in\mathbb{R}^{d_k\times d_v}$，其中每个位置写入一个 key-value 外积；当前位置的 query 再从矩阵状态中读出结果：

~~~math
S_0=0,\qquad
S_t=\gamma S_{t-1}+k_t^{\mathsf T}v_t,\qquad
y_t=q_tS_t.
~~~

连续展开后，位置 $t$ 读到的是所有因果历史外积的衰减和：

~~~math
S_t=\sum_{j=1}^{t}\gamma^{t-j}k_j^{\mathsf T}v_j.
~~~

例如，第三个位置的状态包含 $\gamma^2 k_1^{\mathsf T}v_1$、$\gamma k_2^{\mathsf T}v_2$ 和 $k_3^{\mathsf T}v_3$。这个展开说明，parallel 矩阵式和 recurrent 状态式表达同一组位置贡献。状态形状为 $d_k\times d_v$，不会因为序列从 $T$ 增长到更长而增加历史条目；但各位置贡献被累加到有限矩阵中，并未为每个 token 保留独立地址。

## 38.3 Multi-Scale Retention block 的组成与数值尺度

单一衰减率只给出一种时间尺度。RetNet 将不同固定衰减率分配给不同 retention head，论文给出的默认序列为：

~~~math
\gamma_h=1-2^{-5-h},\qquad h=0,\ldots,H-1.
~~~

较大的 $\gamma_h$ 会让单项历史贡献下降得更慢。它描述的是加权系数的衰减速度，不直接等于模型能准确保存或检索多少条事实。多头结果先合并并按 head 分别做 GroupNorm，再经过 Swish 门控和输出投影：

~~~math
H_h=\mathrm{Retention}(X;\gamma_h),\qquad
Z=\mathrm{GroupNorm}_{\mathrm{head}}\bigl(\mathrm{Concat}(H_0,\ldots,H_{H-1})\bigr),
~~~

~~~math
\mathrm{MSR}(X)=\bigl(\mathrm{swish}(XW_G)\odot Z\bigr)W_O.
~~~

在论文的整体网络中，Multi-Scale Retention 与 FFN 都沿残差路径接入，并各自使用 LayerNorm 前置：

~~~math
Y^{\ell}=\mathrm{MSR}(\mathrm{LN}(X^{\ell}))+X^{\ell},\qquad
X^{\ell+1}=\mathrm{FFN}(\mathrm{LN}(Y^{\ell}))+Y^{\ell}.
~~~

RetNet 还对 retention score 做数值缩放。论文先按 $\sqrt{d_k}$ 缩放 $QK^{\mathsf T}$，再按因果衰减矩阵每行的平方根和缩放 $D$；随后把每行 score 除以其绝对行和与 $1$ 的较大值。GroupNorm 则在 retention 输出之后按 head 处理尺度。它们共同稳定计算，不是 softmax，也不能简化成给 key-value 另加一个通用分母。论文的 multi-scale 设计、门控和 score normalization 见[原论文 §2.2](https://arxiv.org/html/2307.08621)。

将上述缩放合并写出，可以看到它处理的是 score 行尺度：

~~~math
\widetilde D_{t,j}=\frac{D_{t,j}}{\sqrt{\sum_{r=1}^{t}D_{t,r}}},\qquad
R_{t,j}=\frac{q_t k_j^{\mathsf T}}{\sqrt{d_k}}\widetilde D_{t,j},
~~~

~~~math
\widetilde R_{t,j}=\frac{R_{t,j}}{\max\left(\left|\sum_{r=1}^{t}R_{t,r}\right|,1\right)},\qquad
Y=\widetilde R V.
~~~

这个逐行尺度修正不会把 score 变成和为 $1$ 的分布。它也不是额外递归维护一个 key 权重分母的归一化 retention；原论文后续对各 head 输出再做 GroupNorm。

## 38.4 Parallel、recurrent 与 chunkwise recurrent 三种执行形态

三种形式改变的是计算顺序和中间结果的组织方式。Parallel retention 同时构造多个位置之间的因果交互，适合训练时利用矩阵并行；若按全部 token 两两交互来算，score 规模随 $T^2$ 增长。Recurrent retention 每次输入一个新 token，只需读写每个 head 的矩阵状态；状态空间不随 $T$ 增长，但 token 之间仍按时间顺序递归，单步计算仍受 $d_k$ 和 $d_v$ 影响。

Chunkwise recurrent 把序列切成长度为 $B$ 的块。块内并行计算，块与块之间传递状态。设 $S_{\mathrm{in}}$ 是前一块结束后的状态，块内位置从 $0$ 开始编号；先忽略数值稳定化带来的尺度变换，则第 $r$ 个位置读到的状态和块结束时传出的状态分别为：

~~~math
S_{i,r}=\gamma^{r+1}S_{\mathrm{in}}
+\sum_{j=0}^{r}\gamma^{r-j}k_{i,j}^{\mathsf T}v_{i,j},\qquad
y_{i,r}=q_{i,r}S_{i,r}.
~~~

~~~math
S_{\mathrm{out}}=\gamma^B S_{\mathrm{in}}
+\sum_{j=0}^{B-1}\gamma^{B-1-j}k_{i,j}^{\mathsf T}v_{i,j}.
~~~

块内项由当前块的位置共同计算；来自此前块的状态按跨过的位置数继续衰减。若每块开始时都把 $S_{\mathrm{in}}$ 置零，结果就变成互不相连的窗口，丢掉了 RetNet 的跨块历史。固定 $B$ 时，块数随 $T$ 线性增加；实际内存和速度仍取决于块大小、head 宽度和实现。

在数学定义上，三种形式对应同一 retention 映射；对照时须使用相同的初始状态、位置相位、causal mask、衰减和 normalization。浮点实现不保证逐位相等。Microsoft 的 [TorchScale RetNet README](https://github.com/microsoft/unilm/blob/master/retnet/README.md) 记录过 chunkwise 一致性和 recurrent 数值精度修正；具体代码结论应绑定到所读 revision，不能只凭公式推断任意实现都等价。

## 38.5 Retention 状态的表达边界与相邻路线

Retention 与 softmax attention 都用 query、key 和 value 建立内容交互。Full attention 在每个位置对可见历史的 key/value 逐条计算分数；RetNet 则先把带距离衰减的 key-value 外积累加进矩阵状态，再由 query 读取。前者保留显式的历史条目，后者以固定大小的聚合状态换取对历史长度不敏感的存储。

Retention 和线性注意力都可以把 key-value 交互重排成矩阵状态，具体衰减、特征映射和归一化仍须按方法区分；第二十四、二十五章分别展开线性注意力与 Performer。RetNet 的 $\gamma$ 对每个 head 固定，不随当前 token 改变；RWKV 则使用自己的 WKV 递归、分子/分母状态和版本特有的更新方式，见[第三十七章](37-RWKV状态与训练.md)。这些形式不能因为都包含递归状态就互换公式。

由于历史外积会累加到有限状态中，RetNet 并没有把每个历史 token 变成可单独寻址的副本。多尺度衰减提供不同的历史权重尺度，却不保证任意远程实体、精确数字或冲突版本都能被无损读回。具体任务上的能力结论需要对应的模型和数据证据；整套替代架构的计算、表达与硬件比较由[第四十二章](42-替代架构横向对比.md)承担。

## 38.6 原论文的结果与效率主张边界

RetNet 原论文在 The Pile、C4 和 The Stack 的整理语料上训练了 $1.3\mathrm{B}$、$2.7\mathrm{B}$ 和 $6.7\mathrm{B}$ 模型，训练预算为 $100\mathrm{B}$ tokens。训练成本实验使用长度为 $8192$ 的序列、长度为 $512$ 的 chunk 和 $8$ 张 A100-80GB；RetNet 当时使用 vanilla PyTorch，实现中没有 kernel fusion。推理成本实验比较的是 $6.7\mathrm{B}$ 模型在 A100-80GB 上的结果。论文还分别报告了语言建模、训练成本与推理成本，这些是特定模型、语料、硬件、长度和实现下的实验观察，不等于所有现代 Transformer 或所有设备上的普遍性能排序。[原论文 §§3.1—3.4](https://arxiv.org/html/2307.08621) 给出了这些设置和结果。

因此，RetNet 的架构结论应分开表述：并行式展示它如何利用整段计算；递归式说明 decode 状态不必随历史长度持续存放；chunkwise 式在块内并行和块间状态传递之间折中。状态固定解决的是历史存储随 $T$ 增长的问题，不代表状态容量无限，也不单独证明吞吐、准确率或长程检索优势。RetNet 论文之后的官方实现记录还显示，三种计算路径曾需要修复数值与 chunkwise 一致性问题；这些维护事实提醒读者区分数学推导、论文实验与具体 revision 的实际行为。
