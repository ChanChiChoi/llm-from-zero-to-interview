# 第 61 章 KDA：把历史表示成可门控的递归状态

标准因果注意力把每个历史 token 的 key 和 value 留在可寻址的序列中；KDA 则把 key-value 关联持续写入一个有限形状的矩阵状态。每来一个 token，状态先按 key 通道衰减，再用当前 key 对旧状态的预测误差做一次低秩修正。读者要抓住的不是“门控会记忆”这句概括，而是衰减作用在哪个轴、误差如何形成、写入怎样改变状态，以及这份压缩表示带来什么能力边界。

本章讲 Kimi Linear 报告中的 Kimi Delta Attention（KDA）机制。它与通用 Gated DeltaNet 有共同的 delta-rule 基础，但 KDA 采用逐 key 通道的对角衰减；具体层间组合和实现参数要绑定到对应模型版本。第 62 章负责一般 Gated DeltaNet 的 delta 更新与训练实现，第 63 章负责 Gated MLA 的潜变量 cache，第 65 章比较混合层之间的分工。

## 61.1 显式历史与递归状态

在标准因果注意力中，第 $t$ 个 query 可以对每个历史位置 $i\le t$ 计算权重，再读取相应的 value。位置仍然显式存在，因此模型有机会直接选择某个旧 token。推理时，这条路径通常要保留随 token 数增长的 key/value cache。

线性状态路径把多条 key-value 关联合并到矩阵 $S_t$ 中。为说明状态的含义，可以先看一个简化的外积累积：

```math
S_t=S_{t-1}+k_t v_t^{\mathsf T},
\qquad
o_t=S_t^{\mathsf T}q_t.
```

这里 $k_t,q_t\in\mathbb{R}^{d_k}$，$v_t,o_t\in\mathbb{R}^{d_v}$，所以 $S_t\in\mathbb{R}^{d_k\times d_v}$。每个 key 维度对应状态矩阵的一行，每个 value 维度对应一列；query 与状态相乘，读出一个 value 向量。这个简式是理解关联记忆的起点，不是 KDA 的完整定义，也不替代[第 24 章的线性注意力基础](24-Linear-Attention基本思想与核技巧.md)。

压缩改变了历史的寻址方式。$S_t$ 可以让后续 query 读出与 key 表示相关的内容，但它没有为每个旧 token 保留一个独立位置指针。多个历史模式可能共同影响同一状态；模型是否保住了任务所需的信息，取决于 key/value 表示、更新规则和训练任务。KDA 因此不是把完整历史无损搬进一个小张量，而是学习一种递归的信息摘要。

## 61.2 Delta Rule：让状态按当前 key 纠正预测

单纯累加外积会持续叠加旧关联。Delta rule 的改动是：先问当前状态会把 key $k_t$ 读成什么 value，再用预测误差修正这条关联。沿用上一节的矩阵方向，旧状态给出的预测是 $\widehat v_t=S_{t-1}^{\mathsf T}k_t$，误差是 $e_t=v_t-\widehat v_t$。把误差沿当前 key 方向写回状态，可以得到：

```math
S_t=S_{t-1}+\beta_t k_t e_t^{\mathsf T}.
```

$k_t e_t^{\mathsf T}$ 是一个秩一矩阵：它只沿当前 key 指定的方向调整状态，却同时修正所有 value 维度。若状态已经能从当前 key 读出接近 $v_t$ 的结果，误差较小，修正也较小；若当前关联与旧状态预测差异大，写入会更强。系数 $\beta_t$ 控制这次修正的幅度。

这是 delta-rule 记忆的核心直觉。它不等同于一般的“把新 token 加进摘要”，而是先针对当前 key 评估预测，再以残差方向更新关联。Simon Dong 的线性注意力教程用小型关联矩阵逐步解释了这个读出—误差—写入过程；精确的 KDA 定义仍以 [Kimi Linear 报告 v2](https://arxiv.org/abs/2510.26692v2) 为准。通用 DeltaNet 的目标、推导和其他参数化由[第 62 章](62-gated-deltanet门控线性注意力.md)展开。

## 61.3 KDA 的逐通道衰减和写入公式

KDA 在 delta 修正之前，先对旧状态施加对角衰减。令 $\alpha_t\in(0,1]^{d_k}$ 表示当前 token 对各个 key 通道的保留系数，并令 $D_t=\mathrm{Diag}(\alpha_t)$。先形成衰减后的状态，再从它读取当前 key 的预测：

```math
\bar S_t=D_tS_{t-1},
\qquad
\widehat v_t=\bar S_t^{\mathsf T}k_t,
\qquad
e_t=v_t-\widehat v_t.
```

然后以标量写入强度 $\beta_t$ 沿当前 key 方向注入误差：

```math
S_t=\bar S_t+\beta_t k_t e_t^{\mathsf T},
\qquad
o_t=S_t^{\mathsf T}q_t.
```

合起来，这就是 Kimi Linear 报告给出的 KDA 递归形式。展开后可写为：

```math
S_t=(I-\beta_t k_tk_t^{\mathsf T})D_tS_{t-1}
    +\beta_t k_tv_t^{\mathsf T}.
```

这个顺序很重要。$D_t$ 作用于状态矩阵的 key 轴，也就是按行缩放旧的 key-value 关联；它不是对 value 维逐项衰减。$\beta_t$ 则是每个 token、每个 head 的标量写入强度，控制本次误差修正的幅度。逐通道的 $\alpha_t$ 与标量 $\beta_t$ 分工不同，也都不是模型读出端的 output gate。

KDA 与 Gated DeltaNet 的共同点是递归状态和 delta correction；关键差别之一是此处的衰减在 key 通道维度上细化为对角矩阵，而原始 Gated DeltaNet 论文的基本门控是每个时间步/head 的标量衰减。名称相似不意味着它们的门向量、矩阵方向或完整模块相同；精确边界应分别按 [Kimi Linear v2](https://arxiv.org/abs/2510.26692v2) 与 [Gated Delta Networks v3](https://arxiv.org/abs/2412.06464v3) 的定义阅读。

## 61.4 一次完整读写的状态轨迹

用一个二维状态追踪衰减和纠错的先后关系。这里的数字只为演示矩阵更新，不是模型参数或实测数据。设初始状态为

```math
S_0=
\begin{bmatrix}
2&0\\
0&1
\end{bmatrix},
\qquad
\alpha_t=(\tfrac12,1),
\qquad
k_t=(1,0),
\qquad
v_t=(4,6),
\qquad
\beta_t=\tfrac12.
```

第一维 key 通道的旧关联减半，第二维不变：

```math
\bar S_t=
\mathrm{Diag}(\alpha_t)S_0
=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}.
```

当前 key 是第一维方向，所以衰减后的状态对它预测 $\widehat v_t=(1,0)$；新输入是 $(4,6)$，误差为 $e_t=(3,6)$。误差沿 $k_t$ 对应的第一行写入，写入比例为 $\tfrac12$：

```math
S_t=\bar S_t+\tfrac12
\begin{bmatrix}
1\\0
\end{bmatrix}
\begin{bmatrix}
3&6
\end{bmatrix}
=
\begin{bmatrix}
\tfrac52&3\\
0&1
\end{bmatrix}.
```

现在用 $q_t=(1,0)$ 读取，输出为 $o_t=S_t^{\mathsf T}q_t=(\tfrac52,3)$。这条轨迹同时显示了两个控制量的不同作用：$\alpha_t$ 先改变旧状态，$\beta_t$ 再控制当前误差写入的幅度。若只说“gate 决定记忆多少”，读者就看不到这两步分别作用在哪个对象上。

## 61.5 递归更新如何用于 chunkwise 并行训练

单步推理可以按时间执行同一条递归：衰减旧状态、预测当前 value、计算误差、写回状态，再用 query 读取。若训练也对每个 token 顺序启动一次小更新，硬件难以利用长序列中可批量组织的矩阵运算。

Kimi Linear 报告在第 3.1 节给出 chunkwise WY/UT 训练形式：把序列切成块，在块内改写并组织多 token 更新，在块间传递边界状态。块内计算因而可以使用更并行的矩阵操作，同时保持与逐 token 递归相容的状态语义；跨块的历史并没有被自动清零。这里不展开 WY/UT 的矩阵推导，重点是读者要区分“递归定义”和“训练时的计算排布”：前者说明信息如何变化，后者说明多个位置怎样被并行计算。

并行形式本身不保证某个模型在每种硬件和长度下更快。Moonshot 的 [FlashKDA 工程说明](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md)把 chunk 大小、门值的数值范围和 kernel 切分作为具体设计变量，并报告相对于其单 kernel 原型的条件性速度变化。这是特定实现的工程观察，不是 KDA 算法的普遍加速保证；全套内核和 serving 的成本测量属于第二十四册。

## 61.6 固定状态的资源收益和信息代价

对单个 KDA head，核心递归矩阵有 $d_kd_v$ 个元素。若 batch 大小为 $B$、KDA 层数为 $L_{\mathrm{KDA}}$、每层 head 数为 $H_{\mathrm{KDA}}$，状态元素字节数为 $b_s$，核心矩阵状态的简化账本为：

```math
M_{\mathrm{KDA}}\approx
B L_{\mathrm{KDA}}H_{\mathrm{KDA}}d_kd_vb_s.
```

对应的显式 attention K/V payload 会随历史 token 数 $T$ 增长，常见的简化形式为：

```math
M_{\mathrm{KV}}\approx
2BTL_{\mathrm{attn}}H_{\mathrm{kv}}d_hb_{\mathrm{kv}}.
```

这两式只比较核心状态 payload，未计入卷积状态、对齐、workspace、量化 scale 或其他层的 cache。它们说明 KDA 矩阵状态的主项不随 $T$ 线性增长；如果模型同时保留 MLA 或全局 attention 层，那些层仍有自己的 token 级缓存。KDA 不会让混合模型的所有历史状态自动变成常数空间。

有限状态的代价是历史不再逐 token 可寻址。若任务要求精确引用任意旧位置、比较多个相似实体的字段或恢复原文出处，仅凭状态读出可能不够；若任务主要是持续处理流并保留对后续预测有用的关联，递归状态可能符合其资源目标。是否能完成具体任务仍取决于模型训练和状态维度，不能仅凭状态大小推断能力。

Kimi Linear 报告用 Palindrome、Multi-Query Associative Recall（MQAR）和 Stack 等合成任务研究 KDA 等线性注意力机制；所述比较采用两层、两头、head dimension 为 $128$ 的小模型，序列长度为 $256$ 至 $2048$，最多训练 $20{,}000$ 步并搜索学习率。它们能支持这些任务和设置中的机制观察，不能单独证明 KDA 对真实长文档、任意远程引用或所有模型规模都占优。报告中的完整模型速度比较也绑定其模型、序列长度和硬件设置，不能外推为通用 decode 加速。

## 61.7 Kimi Linear 与 Kimi K3 的版本边界

KDA 的状态更新思想可以跨模型版本复用，但具体门的参数化、输出门和层间排布要逐版本说明。Kimi Linear 报告 v2 将 KDA 描述为以逐 key 通道的对角衰减扩展 Gated DeltaNet，并报告 Kimi Linear 使用三层 KDA 配一层全局 MLA。这个比例属于该模型的结构安排，不是 KDA 的定义。

Kimi K3 报告 v2 保留 KDA 状态递归，同时调整了衰减参数化：报告对比 Kimi Linear 的负 Softplus 形式，描述 K3 用带下界的 scaled sigmoid 限制每步 log-decay，并给出 $g_{\min}=-5$；作者把这一范围与累积衰减的倒数范围及块内矩阵计算联系起来。报告还将 Kimi Linear 的低秩 output gate 改为 headwise RMSNorm 后的输入相关 full-rank output gate。它们是 K3 版本的参数化与实现选择，不能改写成所有 KDA 必须采用的方程。

层间组合也有版本边界：K3 报告描述每三层 KDA 接一层 Gated MLA，并在末端另加 Gated MLA；Gated MLA 的潜变量 cache 和门控由[第 63 章](63-gated-mla门控潜变量注意力.md)主讲，K3 的完整层表、MoE 与其他组件见[第 88 章](88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)。读 KDA 时保留这条判断：方程说明一种状态转换，具体模型报告说明一次参数化和层组合，kernel 报告说明一条实现路径；三者不能互相代替。
