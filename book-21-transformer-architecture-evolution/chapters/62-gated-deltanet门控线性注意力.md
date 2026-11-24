# 第 62 章 Gated DeltaNet：带门控的 Delta 状态更新

本章研究一种把历史压进固定形状状态的序列层。状态面对当前 key 先给出 value 预测，delta rule 用预测误差改写相关联的方向；Gated DeltaNet 再让一个标量衰减控制旧状态整体保留多少。读完后，读者应能从状态读写解释公式，说明衰减与改写各自做什么，并判断 chunkwise 训练和有限状态分别带来什么能力与限制。

本章沿用第 61 章的状态化视角，但聚焦通用 Gated DeltaNet。第 61 章负责 KDA 的逐通道衰减，第 63 章讲 Gated MLA 的潜变量缓存，第 65 章比较局部、全局与递归路径的组合。下面不把这些相邻主题重新展开。

## 62.1 递归状态以固定大小保存 key-value 关联

把一个 head 的状态记为矩阵 $A_t\in\mathbb{R}^{d_k\times d_v}$。输入的 key $k_t\in\mathbb{R}^{d_k}$ 标识要访问的关联方向，value $v_t\in\mathbb{R}^{d_v}$ 是这条关联要保存的内容。状态因此可以看成一个从 key 特征到 value 表示的线性映射：

```math
\hat v_t=A_{t-1}^{\top}k_t.
```

$\hat v_t$ 表示当前状态按 key $k_t$ 读出的 value 预测。稍后到来的 query $q_t$ 也沿同一映射读取输出，按本章约定写为 $o_t=A_t^{\top}q_t$。

这张矩阵的大小由 $d_k$ 和 $d_v$ 决定，不随已经读过的 token 数增加。它可以持续累积历史关联，却没有为每个旧 token 保留独立地址。因而“状态大小固定”意味着历史被压进共享表示，不意味着模型能像 full attention 一样重新定位任意历史 token。线性状态是第 24 章介绍的线性注意力路线中一种递归表示；本章关心的是它如何读写和遗忘。

## 62.2 Delta rule：沿当前 key 修正预测误差

若只把新关联加到状态上，相同 key 多次出现时，旧 value 会和新 value 叠在一起。要改写旧关联，更新应先问：当前状态对这个 key 已经预测了什么？观察到的新 value 与预测差多少？

令误差为 $e_t=v_t-\hat v_t$。用它沿当前 key 方向做一个秩一修正：

```math
A_t=A_{t-1}+\beta_t k_t e_t^{\top}
=A_{t-1}+\beta_t k_t\left(v_t-A_{t-1}^{\top}k_t\right)^{\top}.
```

这里 $k_te_t^{\top}$ 的形状是 $d_k\times d_v$，与状态矩阵一致。$\beta_t$ 决定这次误差写入多少。若 $\lVert k_t\rVert_2=1$ 且 $\beta_t=1$，更新后沿同一 key 读出的结果正好变为新 value：

```math
A_t^{\top}k_t
=A_{t-1}^{\top}k_t+
\beta_t\lVert k_t\rVert_2^2
\left(v_t-A_{t-1}^{\top}k_t\right)
=v_t.
```

这说明 delta rule 会修正被当前 key 指向的关联；它不会因此保存完整历史。其他 key 若与 $k_t$ 不正交，读出也可能受到影响，所以多条相似关联仍会共享有限容量。

这个更新也可从一次在线回归梯度步理解。把状态视为快速权重，针对当前 key-value 样本定义平方误差：

```math
\ell_t(A)=\frac{1}{2}\left\lVert A^{\top}k_t-v_t\right\rVert_2^2,
\qquad
\nabla_A\ell_t=k_t\left(A^{\top}k_t-v_t\right)^{\top}.
```

执行一步大小为 $\beta_t$ 的梯度下降，得到 $A_t=A_{t-1}-\beta_t\nabla_A\ell_t$，正是上面的 delta 更新。这里更新的是随序列推进的状态 $A_t$，不是对模型参数做一次常规反向传播。原始 DeltaNet 与 fast-weight 教学资料都用这一联系说明“根据当前样本纠正已有映射”的来源；具体 Gated DeltaNet 公式以论文定义为准。[Yang 等人的 Gated Delta Networks 论文](https://arxiv.org/html/2412.06464v3)；[Behring 的 delta-rule 推导笔记](https://ssm-foundations.brandon-behring.dev/chapters/ch12-delta-rule-lineage/)。

## 62.3 Gated Delta Rule：全局遗忘与定向改写并存

delta rule 能改写被当前 key 命中的方向，但对未被再次访问的旧关联不会主动遗忘。Gated DeltaNet 在改写之前先缩放旧状态。令衰减后的状态为 $\bar A_t=\alpha_tA_{t-1}$，再从它计算预测误差：

```math
\bar A_t=\alpha_tA_{t-1},
\qquad
\hat v_t=\bar A_t^{\top}k_t,
\qquad
A_t=\bar A_t+\beta_t k_t(v_t-\hat v_t)^{\top}.
```

### 62.3.1 canonical 递归式与残差式的对应

合并中间量，可得本章采用的递归式：

```math
\boxed{
A_t=\alpha_t\left(I-\beta_t k_tk_t^{\top}\right)A_{t-1}
+\beta_t k_tv_t^{\top}
}
```

矩阵顺序来自 $A_t\in\mathbb{R}^{d_k\times d_v}$ 的约定：转移矩阵作用在状态左侧。2025 年原论文把状态写作 $S_t\in\mathbb{R}^{d_v\times d_k}$，对应方程是 $S_t=S_{t-1}\bigl(\alpha_t(I-\beta_t k_tk_t^{\top})\bigr)+\beta_t v_tk_t^{\top}$；转置后才得到上面的方向。[原论文 §3.1，Eq. 10](https://arxiv.org/html/2412.06464v3)。

### 62.3.2 $\alpha_t$ 与 $\beta_t$ 的不同职责

两个标量的工作不同：

- $\alpha_t$ 缩放整个旧状态。值越小，先前写入的各个方向保留越少；若 $\alpha_t=1$，不做额外的整体衰减。
- $\beta_t$ 控制当前 key-value 残差写入多少。值为 $0$ 时只执行衰减；key 为单位向量且 $\beta_t=1$ 时，当前 key 的预测被完全改到 $v_t$。

这两个动作按顺序配合：先将旧状态变为 $\bar A_t$，再用 $\bar A_t$ 对当前 key 的预测计算误差。如果先用未衰减状态算误差，或把 $\alpha_t$ 乘到新写入项，都会改变方法含义。

### 62.3.3 递归状态更新与在线回归

从在线回归看，Gated DeltaNet 先把旧状态移到 $\bar A_t$，再对这个衰减后的状态执行一小步 delta 梯度更新：

```math
A_t=\bar A_t-\beta_t\nabla_A\ell_t(\bar A_t),
\qquad
\ell_t(A)=\frac12\left\lVert A^\top k_t-v_t\right\rVert_2^2.
```

所以 $\alpha_t$ 改变了回归更新的起点，$\beta_t$ 决定从该起点朝当前样本修正多少。原论文也将二者分别解释为自适应 weight decay 和更新步长；这描述的是状态矩阵的逐 token 学习视角，不意味着每个 token 都对模型权重反向传播。

例如取两个正交 key $e_1=(1,0)^\top$、$e_2=(0,1)^\top$，令旧状态为 $A_{t-1}=(1,2)^\top$，它分别把这两个方向映射到标量 value $1$ 和 $2$。现在以 $k_t=e_1$ 写入 $v_t=3$，取 $\alpha_t=0.5$、$\beta_t=1$：

| 阶段 | 状态或中间量 | 含义 |
| --- | --- | --- |
| 衰减 | $\bar A_t=(0.5,1)^\top$ | 两条旧关联都先乘以 $0.5$ |
| 预测 | $\hat v_t=\bar A_t^{\top}e_1=0.5$ | 当前 key 在衰减状态中读到 $0.5$ |
| 残差 | $v_t-\hat v_t=2.5$ | 新目标比旧预测高 $2.5$ |
| 写入 | $A_t=\bar A_t+e_1(2.5)=(3,1)^\top$ | 第一方向改为 $3$，第二方向保留衰减后的 $1$ |

于是 $e_1$ 读出新值 $3$，而 $e_2$ 仍读出 $1$。这个两维例子展示了两种机制的分工：$\alpha_t$ 让所有旧内容共同衰减，delta correction 则沿当前 key 修正关联。真实模型中 $k_t$、$v_t$ 和状态是高维张量，数值例子不代表模型配置或效果测量。

## 62.4 Query 读出与 token mixer 中的其他门控

状态更新说明信息怎样进入记忆；query 说明当前 token 如何从记忆中读取。按本章的 key-by-value 方向，递归状态对 query 的读出是：

```math
o_t=A_t^{\top}q_t\in\mathbb{R}^{d_v}.
```

逐 token 的状态语义和训练时的块内计算要保持一致。chunkwise 表达会把一个块的输出拆成块入口状态贡献与块内因果修正，因此不能把块内所有 token 都当成只读取同一个块末状态。

具体 Gated DeltaNet token mixer 还包含投影和输出处理。原论文的模块图给出 query/key/value 投影、短因果卷积、SiLU、query/key 的 L2 normalization，以及单独的 output gate。output gate 调节递归读出进入后续网络的输出；它不替代状态方程中的 $\alpha_t$ 或 $\beta_t$。[原论文 §3.4](https://arxiv.org/html/2412.06464v3)。

## 62.5 Chunkwise WY：保留递归语义并并行处理块内计算

### 62.5.1 逐 token 状态依赖与块内并行目标

逐 token 更新时，$A_t$ 依赖 $A_{t-1}$；若训练时完全照此顺序执行，同一序列中的 token 状态就要逐步计算。GPU 更擅长批量矩阵运算，因此 Gated DeltaNet 论文把连续 token 分成 chunk，并推导 gated WY/UT 形式来组织块内运算。[原论文 §3.3](https://arxiv.org/html/2412.06464v3)。

### 62.5.2 WY/UT 表示怎样压缩一段转移

可以把一块的作用概括成一个仿射状态变换：输入是块入口状态，输出是块末状态；块内每个位置的输出则由入口状态贡献和块内的因果交互共同组成。delta 转移中的 key-key 秩一作用在块内累积成低秩辅助矩阵，衰减门的累计乘积同步进入块内计算。WY/UT 的目的，是在不逐 token 形成完整状态转移矩阵连乘的情况下，用小型三角求解和批量矩阵乘来算出这些作用。[Steiger 的 chunkwise DeltaNet 推导](https://www.ersteiger.com/posts/linear-attention-part-3/)展示了 rank-one 转移乘积如何改写成 WY 形式，并解释其如何避免显式构造 $d_k\times d_k$ 矩阵。

### 62.5.3 训练计算与递归结果的等价条件

chunkwise 训练仍须传递边界状态，因果顺序和 gate 语义也必须与递归式一致。分块改变计算排布，不会把模型变成能同时访问整条序列的 full attention；它也不保证所有 chunk 大小、kernel 和 GPU 都有相同速度。具体 throughput 是实现、硬件和序列负载共同决定的测量值。

## 62.6 固定状态的容量、代价与论文证据

### 62.6.1 状态大小与单步更新成本

在 $d_k,d_v$ 固定时，每个 head 的状态矩阵大小不随历史 token 数增长；但每个 token 仍要读状态、计算预测并进行秩一更新，序列计算成本依旧随 token 数增加。实际吞吐还受投影、状态布局、数据类型、kernel 融合和硬件带宽影响，不能从“状态固定”或线性复杂度直接推出端到端更快。

### 62.6.2 关联覆盖、干扰与精确回读边界

有限状态也有表达边界。delta correction 有助于改写当前 key 所指向的关联，$\alpha_t$ 可衰减没有被新 key 命中的旧关联；但所有历史关联仍共同占用一个矩阵。相似 key、多条相互干扰的关联、精确数字和需要返回原位置的任务，都应分别评估。它们的困难来自状态压缩和读写机制，不等于 Gated DeltaNet 在所有任务上都会失败。需要逐 token 精确回读时，保留显式注意力路径是架构层的另一种选择，具体混合安排见第 65 章。

### 62.6.3 GDN 与 GDN-2 实验的条件边界

原始 Gated Delta Networks 论文比较了 1.3B 参数模型，在 FineWeb-Edu 的 100B token 上训练，训练序列长度为 4K；其合成单针检索评测覆盖多个上下文长度，混合模型采用 2K sliding window。论文还报告单张 H100 的训练吞吐。这些结果说明该方法在相应训练与评测设置中的表现，不能推出任意规模、数据或任务下都优于 Transformer，也不是生产部署的复现结果。[原论文 §3.2、§4](https://arxiv.org/html/2412.06464v3)。

## 62.7 Gated DeltaNet、KDA 与 Gated DeltaNet-2 的门控边界

三种相邻方法可以按“旧状态如何衰减”与“当前关联如何修改”拆开看：

| 方法 | 旧状态衰减 | 当前 key/value 更新 |
| --- | --- | --- |
| Gated DeltaNet | 每个 token/head 一个标量 $\alpha_t$ | 标量 $\beta_t$ 同时控制 delta rule 的定向擦除与 value 残差写入 |
| KDA | 沿 key 通道的衰减向量 | 保留标量写入强度；第 61 章详解此参数化 |
| Gated DeltaNet-2 | 沿 key 通道的衰减向量 | 分别以 key-side erase gate $b_t$ 和 value-side write gate $w_t$ 控制改写 |

KDA 的主要变化是衰减由单一标量扩为逐 key 通道系数，而 delta 写入强度仍是标量。Gated DeltaNet-2 进一步拆开一次 delta 编辑中两个被绑在一起的动作。用本章的矩阵方向，其更新可写成：

```math
\bar A_t=D_tA_{t-1},
\qquad
e_t=b_t\odot k_t,
\qquad
z_t=w_t\odot v_t,
\qquad
A_t=\bar A_t+k_t\left(z_t-\bar A_t^{\top}e_t\right)^{\top}.
```

$D_t$ 负责逐 key 通道衰减，$e_t$ 决定读取并擦除旧关联时使用的 key 方向，$z_t$ 决定哪些 value 通道写入新状态。Gated DeltaNet-2 论文给出的参数约化关系是：让 $b_t$、$w_t$ 收缩为相同标量 gate 可得到 KDA；再让 $D_t$ 收缩为标量乘单位阵，可得到 Gated DeltaNet。[Hatamizadeh 等人的 Gated DeltaNet-2 v2](https://arxiv.org/html/2605.22791v2)。

这项工作是 2026 年论文提出的后续扩展。其匹配实验以 1.3B 模型和 100B FineWeb-Edu token 为主要设置，并包含 synthetic 与 real-world retrieval 及单 H100 吞吐比较；结论只覆盖作者报告的任务、规模、训练长度与实现条件。[Gated DeltaNet-2 v2 的实验与吞吐设置](https://arxiv.org/html/2605.22791v2)显示，独立擦除和写入值得研究；它不代表通用 Gated DeltaNet 的定义已经改变，也不足以证明这一变体已成为生产模型的普遍选择。

至此，核心判断可以从状态更新直接推出：$\alpha_t$ 管旧状态的整体衰减，$\beta_t$ 管当前 key 上的预测误差写入；chunkwise WY 让训练计算更适配并行硬件，同时保留块边界的递归状态。判断实际收益时，还要看有限状态能否承载目标任务所需的关联，以及论文或测量是否覆盖同样的数据、长度、模型和设备条件。
