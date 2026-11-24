# 第五十二章大纲：Chinchilla、PaLM 与 Compute-Optimal Training

本章把前一章介绍的 scaling 规律落到预训练预算分配：在明确的训练计算预算与预训练损失目标下，模型参数量与训练 token 数如何共同变化。以 Chinchilla 的经验前沿和 PaLM 的模型/系统规模为两种不同证据，区分模型—数据分配的 compute-optimal 与多设备训练的系统效率。一般 scaling law、数据质量和下游指标由第九章主讲；精确 attention 计算、MoE 成本和助手后训练分别承接第十、二十一/二十二和第五十三章。

## 52.1 Compute-optimal：先固定目标和预算

本节说明 compute-optimal 不是无条件的“最强模型”，而是在固定预训练 FLOPs 预算下，按某个明确目标（如最终预训练 loss）选择参数量和训练 token 的分配。该定义限定了要优化的阶段、指标与资源，不把下游能力或部署体验自动当作同一目标。

## 52.2 模型参数和训练 token 的预算关系

本节说明参数量与训练 token exposure 如何共同决定 dense Transformer 的近似训练计算，并解释常用 $6ND$ 估算的来源、口径与省略项。区分总训练 token、唯一数据量和单样本上下文长度；真实 FLOPs 受 attention、输出层、重计算和实现影响，MoE 还涉及激活参数、总参数及通信。

## 52.3 Chinchilla：把同一算力从参数转向 token

本节围绕 Chinchilla 的实证路径展开：超过 400 个中小规模模型拟合预算前沿，再将该前沿用于大模型预算，并用 Chinchilla 的 $70\,\mathrm{B}$ 参数/$1.4\,\mathrm{T}$ token 与 Gopher 的约 $280\,\mathrm{B}$ 参数/$300\,\mathrm{B}$ token 作大规模检查。说明 $D/N\approx20$ 是特定估计的记忆点，不是固定自然常数；比较结果也受训练数据与协议范围约束。

## 52.4 PaLM：系统规模与预算配置是两类扩展

本节说明 PaLM 540B dense 模型、780B 训练 token 与 Pathways 多 TPU Pod 训练所呈现的模型/数据/硬件扩展。区分 hardware FLOPs utilization 与 model FLOPs utilization，也说明 PaLM 与 Chinchilla 的预算、数据和评估不是同预算单变量对照，因此 PaLM 不能单独验证 Chinchilla 的参数-token 比例。

## 52.5 结论何时迁移：实验边界与目标变化

本节概括 scaling 前沿对 FLOPs 计数、模型参数定义、数据与优化器/学习率设置的敏感性，并指出有限唯一数据与重复训练会构成另一类约束。区分预训练损失最优、下游任务表现、后训练需求和推理/部署目标；dense 估算不能直接搬到 MoE，具体后训练、系统服务及通用 scaling 指标由相关章节承担。
