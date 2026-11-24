# 第 62 章 Gated DeltaNet：带门控的 Delta 状态更新

本章解释 Gated DeltaNet 如何在固定形状的 key-value 状态中组合两种不同动作：标量衰减控制旧状态整体保留多少，delta rule 沿当前 key 所指方向修正关联。章节从状态读写和预测误差出发，给出 Gated Delta Rule 的等价形式与变量职责，再说明 chunkwise WY 训练如何保留逐 token 递归语义并提高块内并行度。后半部分限定状态容量、检索能力和论文实验能够支持的结论，并把通用 Gated DeltaNet 与 KDA、Gated DeltaNet-2 的门控维度区分开。第 24 章负责线性注意力前置，第 61 章详讲 KDA 的逐 key 通道衰减，第 63 章负责 Gated MLA，第 65 章负责混合架构的路径分工，第 88 章负责 Kimi K3 案例；本章不展开通用 serving 生命周期或推理系统运维。

## 62.1 递归状态以固定大小保存 key-value 关联

本节把递归矩阵视为从 key 到 value 的快速映射，并交代 key 轴、value 轴及 query 读出关系。固定大小的状态减少了随历史 token 数增长的显式缓存，但多条关联会共享有限表示，不能保留每个 token 的独立地址；本节只唤回理解 Gated DeltaNet 所需的线性状态前置，不重复第 24 章的核映射推导。

## 62.2 Delta rule：沿当前 key 修正预测误差

本节围绕“状态已经如何响应当前 key、当前 value 与旧预测相差多少”解释 delta update。知识范围包括状态预测、value 残差、沿 key 方向的秩一修正，以及它相对简单累加写入对重复或冲突关联的处理差别；在线回归和一次梯度更新的对应关系说明该规则为何能改写关联，并限定 key 范数、步长与单步更新的作用范围。

## 62.3 Gated Delta Rule：全局遗忘与定向改写并存

本节给出 Gated DeltaNet 的 canonical 状态转换，采用 $A\in\mathbb{R}^{d_k\times d_v}$ 的方向约定，说明先对旧状态施加标量衰减 $\alpha_t$，再根据已衰减状态形成预测误差并由标量 $\beta_t$ 控制定向写入。知识重点是两种机制作用的范围不同、等价展开式如何体现这一关系，以及门控趋近边界时状态保留与覆盖发生的变化；正文数值推演应沿同一状态变量保持符号和矩阵形状一致。

### 62.3.1 canonical 递归式与残差式的对应

本节限定状态、key、value、query 与标量 gate 的形状，说明矩阵乘法方向和两种等价写法；明确状态衰减后再计算残差，避免把 $\alpha_t$ 错放到新写入项或误写成逐通道 gate。

### 62.3.2 $\alpha_t$ 与 $\beta_t$ 的不同职责

本节区分全局遗忘和定向更新：$\alpha_t$ 缩放既有状态，$\beta_t$ 控制当前 key-value 误差的写入强度。说明纯衰减、无衰减 delta 更新和单位范数 key 下完全修正当前关联等极限情形，并标明这些是由递归式推得的性质，而非 benchmark 结果。

### 62.3.3 递归状态更新与在线回归

本节把 key 到 value 的线性读出写成单样本平方误差，解释对状态矩阵做一步梯度下降如何得到 delta correction；Gated DeltaNet 则先将旧状态移向衰减后的目标，再对该状态执行修正。该视角帮助理解公式来源，不把测试时状态更新说成模型参数的常规反向传播。

## 62.4 Query 读出与 token mixer 中的其他门控

本节连接递归状态到输出：query 从更新后的 key-value 映射中读取 value 表示，并区分状态转移中的 $\alpha_t$、delta 步长 $\beta_t$ 与 token mixer 末端可能存在的 output gate。Gated DeltaNet 论文报告的 query/key/value 投影、短卷积、归一化和 output gate 属于其模块设计，不应与 gated delta rule 的状态方程混为一谈；具体投影和归一化位置按固定论文版本说明。

## 62.5 Chunkwise WY：保留递归语义并并行处理块内计算

本节解释逐 token 递归如何在训练时变为块内并行矩阵运算。知识范围包括状态依赖为何使朴素逐 token 扫描难以充分利用 GPU、块内 delta 转移的 WY/UT 表示如何组织累计修正、块边界如何传递递归状态，以及并行化不等于抹去跨块状态依赖。公式细节以 Gated Delta Networks 原论文的 gated chunkwise 算法为准，并借助相关教学推导澄清分块代数；不将特定 chunk 长度或 kernel 形状写成算法必需值。

### 62.5.1 逐 token 状态依赖与块内并行目标

本节区分递归更新的数学语义和训练 kernel 的计算排布，说明每个 token 的状态依赖关系，以及块内批量运算与块间边界状态各自承担的工作。

### 62.5.2 WY/UT 表示怎样压缩一段转移

本节聚焦 rank-one 转移矩阵的分块乘积如何用低秩辅助量和小型三角系统表达，避免逐项物化完整状态转移矩阵；对 gated 情况，还需追踪累计衰减如何进入块内 key/query 和边界状态。

### 62.5.3 训练计算与递归结果的等价条件

本节说明 chunkwise 算法必须保持原递归的 token 顺序、因果 mask、边界状态和 gate 口径。论文报告的并行训练效率只适用于其实现和硬件设置；chunk 并行形式本身不证明所有 kernel、序列长度或 serving 负载都更快。

## 62.6 固定状态的容量、代价与论文证据

本节把方法机制和可支持的工程判断相连：固定状态大小不随序列长度增长，但每个 token 仍需更新状态，且多个 key-value 关联会发生干扰，精确远距离回读能力需要按任务验证。原始 Gated Delta Networks 的合成检索、长上下文和模型训练实验，以及 Gated DeltaNet-2 的后续对照，需绑定论文版本、模型规模、token 预算、训练长度、任务和硬件条件；作者报告结果不能外推为普遍优于 full attention、生产吞吐收益或所有长上下文任务都可靠。

### 62.6.1 状态大小与单步更新成本

本节区分随上下文长度增长的 token cache payload 与固定维度递归状态，并交代固定状态仍有按 token 执行的矩阵读写开销。复杂度只描述所分析的状态更新，不替代 kernel、显存带宽或端到端延迟测量。

### 62.6.2 关联覆盖、干扰与精确回读边界

本节围绕有限状态为何不能天然支持对任意历史 token 的精确地址访问，讨论重复 key 的定向改写、未被新 key 命中的旧关联如何衰减，以及相近 key、多项干扰和内容精确引用带来的评估压力；混合 attention 的结构分工由第 65 章展开。

### 62.6.3 GDN 与 GDN-2 实验的条件边界

本节总结原始论文与后续 GDN-2 论文中可用于机制判断的实验类型，明确其匹配的模型、数据预算、训练长度、synthetic/real-world retrieval 指标和 H100 throughput 条件。结果用于说明研究假设在对应实验中的表现，不当作独立复现或部署经验。

## 62.7 Gated DeltaNet、KDA 与 Gated DeltaNet-2 的门控边界

本节按状态转移的轴与变量职责比较三种相邻路线：Gated DeltaNet 对旧状态使用每 head/token 标量衰减并以标量 $\beta_t$ 同时调节 key-side erase 与 value-side write；KDA 将衰减扩展为逐 key 通道系数但保留标量写入强度；Gated DeltaNet-2 在 channel-wise decay 上进一步分别引入 key-side erase gate 和 value-side write gate。说明这些方程在特定参数取值下如何退化为彼此，并将 KDA 的详细机制留给第 61 章；GDN-2 是 2026 年论文中提出的后续研究，不代表通用 Gated DeltaNet 已改变定义或形成生产共识。
