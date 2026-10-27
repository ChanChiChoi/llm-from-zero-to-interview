# 第五册第六章：优化器与学习率策略研究笔记

研究日期：2026-10-26。范围是 `book-05-llm-training/chapters/06-优化器与学习率策略.md`；目标是保留旧章覆盖的训练优化知识，再把关键机制、研究证据、工程边界和比较方法讲透。来源链接用于进入论文/官方页面；结论只覆盖本笔记列出的阅读范围，不表示独立复现。

## 研究问题

1. 优化器之间到底改变了什么：逐元素统计、低秩状态、曲率估计，还是矩阵更新几何？各自针对的瓶颈是什么？
2. Sophia、Muon、Shampoo/SOAP、GaLore 的更新状态和额外计算怎样产生？论文报告的收益受哪些参数组、数据、规模、batch 和调参规则约束？
3. Cosine 与 WSD/WSD-S 的 checkpoint 路径和恢复状态有什么差别？loss spike 会怎样影响 decay 的评估？
4. 什么样的比较才能回答“某 optimizer 更快”：同 step、单步速度、最终 loss，还是达到目标质量所需的 tokens、FLOPs 和墙钟时间？
5. 梯度累积中的有效 batch 应按样本数、输入 token 数，还是进入 loss 的预测目标数计算？

## 旧版知识点覆盖图

| 旧章主题/读者问题 | 新章位置 | 处理层级与理由 |
| --- | --- | --- |
| 参数、梯度、训练步、状态账本 | §6.1 | 核心基础；区分 micro-step、optimizer step 和 token 预算，避免用 step 数代替训练量。 |
| SGD、Momentum、Adam、偏置修正、epsilon | §6.2—6.3 | 核心机制保留并补公式与数值边界；构成后续方法的参照基线。 |
| L2 正则和 weight decay、AdamW | §6.4、§6.15 | 核心；加入首步数值反例，说明自适应缩放下耦合 L2 与解耦衰减轨迹不同。 |
| 优化器状态显存、精度、分布方式 | §6.5 | 核心工程约束；区分 optimizer state、gradient、参数与 activation 的来源。 |
| Adafactor、Lion、Sophia | §6.6—6.8 | 均保留为不同约束下的代表方法；增加机制、代价、边界和原始实验。 |
| Muon、Shampoo/SOAP、GaLore | §6.9—6.10、§6.18.4 | 矩阵预条件、正交化和低秩状态路线分开讲；不把“省状态”“换几何”“少一个状态”混成一类。 |
| 参数分组 | §6.11、§6.9.2—6.9.3 | 核心工程条件；Muon 类方法须说明哪些矩阵参数进入、embedding/head/norm 如何处理。 |
| Warmup、cosine、linear、constant、恢复训练 | §6.12—6.13、§6.20 | 保留调度器与 scheduler state；重点说明步数边界、resume 和终点预算。 |
| WSD、分支 checkpoint、延长 stable 阶段 | §6.14 | 核心增量训练问题；补主干/decay 分支差异、MiniCPM 数据点及适用边界。 |
| gradient clipping 与混合精度执行顺序 | §6.16 | 保留并维持工程操作入口；本轮未把 clipping 说成优化器优劣证据。 |
| batch scaling、有效 batch、不同长度/mask | §6.17 | 核心；有效 batch 统一按 loss 中真实预测目标数统计，避免 rank 等权造成短序列过重。 |
| 优化器公平比较、PPL、数据效率 | §6.18 | 核心；将调参预算、最终训练预算、目标质量与完整成本放在同一比较协议下。 |
| toy optimizer 示例 | §6.19 | 保留并纠正命名边界：代码只展示矩阵 Frobenius 缩放，不声称实现 Muon 或 Sophia Hessian estimator。 |
| optimizer/scheduler checkpoint、恢复轨迹 | §6.20 | 保留；WSD/WSD-S 的恢复对象与学习率状态仍要分别存储。 |
| loss spike 归因、学习率、beta/epsilon、clipping | §6.21 | 保留为故障诊断入口；增加新优化器没收益时先排除调参、参数分组与计算瓶颈。 |
| 阅读清单、练习和总结 | §6.22—6.24 | 保留并增加一手来源、研究比较练习和“证据绑定条件”的总结。 |

本轮没有删去上述旧章主线知识点。优化器外的低层 kernel 实现和具体框架版本继续作为延伸阅读/工程核对入口，而不是被描述为算法本身的普适性质。

## 代表性方案各自解决什么

| 方法 | 要处理的问题 | 机制和成立条件 | 主要收益/成本与失效边界 |
| --- | --- | --- | --- |
| AdamW | 不同参数坐标梯度尺度差异；成熟的大模型基线 | EMA 一阶/二阶矩，weight decay 从自适应梯度中解耦 | 状态约为每参数两份；仍需按参数组调 LR、decay 和 scheduler。AdamW 论文解释了为何 adaptive method 中 L2 与 weight decay 不等价。 |
| Adafactor | Adam 二阶状态随参数量线性增加 | 对矩阵二阶统计作行/列因式分解；recipe 另含 relative step / update clipping | 降低矩阵参数状态；分解近似改变优化轨迹，向量参数和调度细节不能忽略。 |
| Lion | 减少 optimizer state 和逐元素自适应除法 | 保存一阶动量，以符号方向更新 | 少一份二阶状态；更新幅度和学习率配方变了，不能复用 AdamW 学习率直接比较。 |
| Sophia-H | Adam 不能适应异质曲率；完整 Hessian 太贵且非凸 Newton 不稳 | Hutchinson 随机向量估计 Hessian 对角，借 HVP；k 步刷新、EMA、分母下界和逐坐标裁剪 | 估计无偏但有随机噪声和 HVP 成本；裁剪限制坏估计影响。 |
| Sophia-G | 降低 Sophia-H 对 Hessian-vector product 的要求 | 利用交叉熵结构及重采样 label 的梯度构造 Gauss-Newton-Bartlett 对角估计 | 对 Gauss-Newton 项无偏、相对 Hessian 对角有偏，但非负；不应称作通用 Hessian 真值。 |
| GaLore | 全参数训练时 Adam 状态占显存 | 将矩阵梯度周期性投影到低秩子空间，在投影空间保存 optimizer state，映回更新原权重 | 减 optimizer-state 和梯度显存；rank/刷新周期影响误差、SVD 成本及 checkpoint 连续性；不是冻结基座的 LoRA。 |
| Muon | 逐元素缩放没有利用矩阵整体几何 | 对动量矩阵近似 polar/正交化，常用 Newton–Schulz；需要按 shape 校准 update RMS 与参数分组 | 可改变收敛/compute-efficiency；矩阵形状、优化器分组、正交化 kernel、all-gather/通信和缩放均影响复现。 |
| Shampoo | Adam 的逐坐标 preconditioner 忽略矩阵/张量坐标相关性 | 每个张量维度维护预条件矩阵，对梯度施加矩阵逆幂 | 能表达更丰富的相关性；状态、矩阵分解、blocking 和通信带来成本。 |
| SOAP | Shampoo 的矩阵分解/更新频率与稳定性成本 | 在 Shampoo 预条件器特征基中做 Adam 式二阶矩更新；在缓慢移动的基中持续跟踪统计 | 原论文在 360M/660M 语言模型的大 batch 条件下有收益；频率/基陈旧以及后续系统实现都是边界。 |
| Cosine | 总预算已知，需要随训练接近终点降学习率 | 将总步数纳入预先定义的衰减曲线 | 终点清楚、易比较；训练延长时原计划不再是自然 continuation。 |
| WSD | 数据/算力预算可能后续扩展，或需同一主干导出多个终点 | warmup 后稳定主干保持学习率；从 stable checkpoint 独立分支做 decay | 一条主线可支持多个终点；需保存主干并承担分支/存储成本。 |
| WSD-S | 希望连续单次 run 复用 cooldown 计算 | 从 decay 后 checkpoint 恢复到高 LR 继续主干 | 在论文预算范围优于 WSD、接近分别调优的 cosine；对 spike 后立即 decay 敏感，不能把作者 loss-landscape 解释当普适定理。 |

### 方案对照的证据范围

- **AdamW 与 L2 耦合**：[原论文](https://arxiv.org/abs/1711.05101)说明对 adaptive method，两者更新轨迹不同；章节的 `theta=8, grad=1` 首步例子只计算局部更新，不表示完整训练效果。
- **Adafactor**：[原论文](https://arxiv.org/abs/1804.04235)第 3—8 节说明 factored second moment、update clipping、relative step 等 recipe。矩阵行列因式分解只解释状态节省，不足以推出最终质量或吞吐更好。
- **Lion**：[原论文](https://arxiv.org/abs/2302.06675)第 3—6 节涉及算法发现、语言建模、超参调优和限制。CV/扩散结果不等于 LLM 预训练保证。
- **Sophia**：[原论文算法 3、§2.3、§3.1—3.5](https://arxiv.org/html/2305.14342)给出 H/GNB estimator、刷新方程、`max(gamma*h, epsilon)` 分母、[-1,1] clipping 与实验。GPT-2 125M—770M、GPT-NeoX 1.5B/6.6B、OpenWebText/Pile 条件下作者报告约 2× steps/compute/wall-clock，单步开销低于 5%；其超参在 30M tune，且刷新/估计样本数各异。该结果不能直接代表当前 frontier 模型。
- **GaLore**：[原论文](https://arxiv.org/html/2403.03507)第 3—6 节区分低秩梯度子空间、全参数更新与 LoRA；论文在其 C4/架构/精度配置中报告 optimizer-state 降幅，数字不能与只测 optimizer bytes 或总训练显存的其他表格混用。
- **Muon**：[Moonshot 报告](https://arxiv.org/html/2502.16982)方法和实验部分记录 Newton–Schulz、分布式 gather、shape-dependent update RMS、AdamW parameter split、399M—1.5B scaling runs 及 Moonlight 15.29B 总/2.24B 激活参数、5.7T tokens。其约 52% FLOPs 结论是作者在该 compute-optimal 比较中的结果。
- **Shampoo/SOAP**：[Shampoo 原论文](https://proceedings.mlr.press/v80/gupta18a.html)支撑结构化张量预条件；[SOAP 原论文](https://arxiv.org/abs/2409.11321)把 Shampoo 的预条件特征基与 Adam 更新结合，公开 LLM 评估为 360M/660M 模型及大 batch。原 paper 的 40% fewer iterations / 35% wall clock 等数据不应外推到不同实现或规模。
- **跨 optimizer 公平性**：[Deconstructing...](https://arxiv.org/abs/2407.07972)用于对照 Adam/Adafactor/Lion/SGD 的设置与变量；[Fantastic Pretraining Optimizers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ea4f0a6ad7088c9b93f22d74c0c9e8c2-Abstract-Conference.html)的 2025 预印本后来发表于 ICLR 2026。它覆盖 0.1B—1.2B、C4-EN/Llama-2 和 1—8× Chinchilla 数据量，发现每个 optimizer 需独立调参、最终预算评估重要，1.2B 设置下优势约降至 1.1×。其主要 speedup 由相同验证 loss 所需 token 数换算，并基于恰当实现可把矩阵 optimizer 额外逐步开销控制在 10% 以下；这不是实测端到端墙钟倍率。中途 checkpoint 排名可能随 decay 改变。
- **超参跨尺度迁移**：[NeurIPS 2025 论文](https://arxiv.org/abs/2512.05620)报告 μP + 按深度/宽度缩放 weight decay 后，Muon/SOAP/Shampoo 在 190M—1.4B FineWeb/Llama 实验中接近 1.4×；错误 scaling 下收益明显消失。它与 ICLR 2026 比较的 token/compute normalization、数据与 schedule 不同，两个数字并不组成排行榜。
- **大 batch 工程**：[NVIDIA 2026 报告](https://arxiv.org/html/2607.20548)包括 8B dense、30B-A3B 与 72B-A8B 等模型规模，报告在其测试配置下 Muon/SOAP 可到 100M global-token batch，并讨论 SOAP eigenbasis staleness、loss spikes、通信和 layer-wise distributed optimizer。它是报告作者的实现与数据，不是跨训练栈独立复测。
- **WSD**：[MiniCPM](https://arxiv.org/html/2404.06395)的 model-wind-tunnel/decay 实验使用 36M proxy 讨论 stable/decay 和数据规模；10% decay 的观察、2.5% 过短等信息只能绑定该 recipe。River Valley 研究与 WSD-S 的 0.1B—1.2B、Pile、4M token batch、50B/100B/200B 目标预算条件见[论文第 3—5 节](https://arxiv.org/html/2410.05192)；损失地形理论有明确假设，WSD-S 的 cooldown 比例和 loss-spike 结果也限于其实验。

## 公式与数字复核

1. AdamW 首步：decoupled decay 先对应因子 `1 - lr*wd = 0.99`，参数先从 8 缩到 7.92；无 bias-correction 误差时自适应步长为 0.1，结果约 7.82。将 L2 并入梯度会令梯度为 `1 + 0.1*8 = 1.8`；首步 Adam 的一阶/二阶矩归一化后方向仍约为 1，结果约 7.90。这个差值说明轨迹不同，不是一般维度上的固定差。
2. Sophia 原算法更新使用 `clip(m / max(gamma*h, epsilon), 1)`，clip 表示坐标截到 [-1,1]；Hutchinson 估 Hessian 对角无偏但可有负坐标，GNB 估计 Gauss-Newton 项并给非负 diagonal。负 `h` 时分母落到 epsilon，非零动量方向因此经裁剪退化为符号方向。论文的 k=10、H/GNB batch 子样本数、gamma 值以及单步开销各有实验上下文。
3. 有效目标数：内部 causal shift 且无 ignore mask 时，长度 L 的输入对应 `L-1` 个 next-token prediction targets；有 padding、packing 或其他 objective 时直接统计进入 loss 的目标数。DDP rank 等权平均 rank-local mean loss 并不自动等于全局 token mean。
4. Muon/SOAP 加速比较必须分别写清 numerator：同 step、同数据 token、同 FLOPs、same target loss 或 wall-clock。Moonshot 的 52% FLOPs、ICLR 2026 的约 1.1× token efficiency、NeurIPS 2025 scaling-transfer 的约 1.4×，采用的模型/数据/缩放方法不同，不应互换单位。
5. WSD 的 `stable` 与 `decay` 是训练阶段/恢复路径，不是 optimizer。原始 WSD 的 decay checkpoint 与主干 checkpoint 需区分；WSD-S 则明确从衰减后的 checkpoint 继续，恢复 scheduler/optimizer 状态后再升 LR。

## 章节改动和静态检查范围

- Sophia 章节增加 H/GNB 对照、曲率状态更新、数学上有下界的分母、逐坐标裁剪、手算数值和作者实验条件。
- Muon/矩阵路线增加 polar 方向、shape scaling、参数分组、Shampoo/SOAP 对照和 2025—2026 条件化证据；更正 Moonlight 激活/总参数表述。
- WSD 说明 stable 主干与 decay branch，增加 River Valley 的假设边界及 WSD-S continuous run；给 WSD-S 段增加直接论文链接。
- 有效 batch 改按 valid prediction targets 计数，覆盖 internal/external shift、mask、变长 rank 和 toy 计算。
- 公平比较表按瓶颈/机制/成本/第一验证组织，增加来源层面的争议和评估规则。
- `INTERVIEW_BANK.md` 增加 Sophia estimator、Muon/Shampoo/SOAP、加速倍率不可比、WSD/WSD-S checkpoint、Adafactor/GaLore/LoRA 状态边界以及公平 optimizer 比较题。
- `PAPERS.md` 与 `GLOSSARY_EN_ZH.md` 同步论文入口和优化器术语。
- Python 教学代码只作静态阅读，本轮没有执行 Python、训练脚本、论文复现实验、公式渲染或 PDF 构建。`Transformers` scheduler 文档为动态 `main` URL；实际框架行为仍应按项目依赖版本核对。

## 仍未验证的事项

- 未在目标数据、模型、硬件和训练框架上复现 Sophia/Muon/SOAP、WSD 或其他优化器论文；原文结果均应称作者报告。
- Muon 和 SOAP 的 distributed kernels、communication overlap、不同参数组的 update RMS、数值稳定性以及真实训练 wall-clock 不由本章 toy 代码覆盖。
- optimizer state 省显存不保证峰值总显存下降，激活、梯度、workspace 和通信 buffer 仍需逐项 profiling。
- 动态官方文档没有在固定框架版本下验证。本章的概念结论与具体 API 行为分开处理。
- 后续章节 toy 代码的运行验收依进度记录处理；本章未运行代码，不能标记为 runtime 通过。
