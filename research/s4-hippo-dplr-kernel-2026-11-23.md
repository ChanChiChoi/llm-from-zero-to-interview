# S4、HiPPO 与结构化卷积核计算

研究日期：2026-11-23。研究问题：核对 S4 如何从 HiPPO 的有限维在线投影记忆发展到 NPLR/DPLR 状态矩阵，并通过生成函数、Woodbury 恒等式、Cauchy 核和逆 FFT 计算有限长度卷积核；同时界定这些结论所支持的范围。

## 检索范围

- 读者和主题范围：已了解固定参数线性时不变 SSM 的连续方程、离散化及递归—卷积关系；本记录聚焦 S4/HiPPO 的结构与核生成，不研究 Mamba 选择性扫描、当前框架实现或服务性能。
- 检索日期与平台：2026-11-23，Web 搜索与原文页面。
- 搜索语言与关键词：英文。检索 `Efficiently Modeling Long Sequences Structured State Spaces NPLR DPLR Algorithm 1`、`HiPPO Recurrent Memory with Optimal Polynomial Projections`、`Annotated S4 HiPPO DPLR Cauchy kernel`、`S4 DPLR normal plus low rank generating function resolvent Woodbury Cauchy kernel roots of unity IFFT`、`Structured State Spaces for Sequence Modeling S4 Hazy Research`。
- 筛选范围：优先阅读原始论文对应章节与算法，再并行查作者团队的机制说明和逐步教学材料；未把摘要中的 benchmark 结论外推到现代系统。

## 主要发现

1. HiPPO 将截至时刻 $t$ 的历史函数按给定的时间测度投影到有限维函数子空间，并在线更新投影系数。测度决定哪些时间区域在近似误差中占更大权重；“最优”指指定测度和子空间下的函数近似，不代表无损保存每个离散 token，也不自动保证下游任务可准确恢复信息。
2. S4 原论文的技术问题分为两部分：朴素地对状态矩阵做稠密幂来生成长卷积核代价高；对 HiPPO 矩阵作朴素特征分解也可能因特征向量矩阵元素随状态维度指数增长而数值不稳定。
3. 原论文证明其讨论的 HiPPO 矩阵可表示为正规矩阵加低秩项（NPLR）。在正规部分的酉基下，整个 SSM 可等价地改写为对角加低秩形式（DPLR）。这里改变的是状态坐标，不是输入到输出的线性算子；DPLR 的低秩修正仍需显式处理。
4. S4 原论文采用双线性离散化。长度为 $L$ 的核元素为离散状态转移矩阵幂构成的有限序列。对截断生成函数，原论文在单位根处评价，使截断项可被吸收到读出向量；生成函数中的矩阵幂因此转为 resolvent 矩阵逆。Woodbury 恒等式处理低秩修正，对角 resolvent 归结为 Cauchy 核乘法，单位根取值通过逆 FFT 还原有限卷积核。
5. 原论文定理将一次结构化递归更新归为 $O(N)$，并将长度为 $L$ 的 S4 卷积核生成归约为 $4$ 次 Cauchy 乘法；在论文的低秩参数化与 Cauchy 算法假设下，报告约为 $\widetilde{O}(N+L)$ 的运算量与 $O(N+L)$ 的空间量。这些是论文中指定算法的渐近结论，不说明任意实现、完整模型、现代硬件或端到端训练/推理必然达到该速度。
6. 原论文报告的任务结果对应其公开实验设置。本文没有将 Long Range Arena、Sequential CIFAR、语音或语言建模结果写成 S4 相对于其他架构的普遍胜出结论，也没有把长序列建模结果等同于任意离散事实的精确检索能力。

## 来源及证据边界

| 来源 | 版本、阅读位置 | 支持内容 | 不支持的结论 / 使用限制 |
| --- | --- | --- | --- |
| Albert Gu、Karan Goel、Christopher Ré，[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)，arXiv:2111.00396，ICLR 2022 | §2.1—2.4、§3.1—3.4、Algorithm 1、Appendix B 的 HiPPO 对角化与数值稳定性、Appendix C 的生成函数/递归/卷积计算 | 连续与离散 SSM、HiPPO 状态来源、状态坐标共轭等价、NPLR/DPLR、双线性离散化、根单位求值、Woodbury、Cauchy 核、逆 FFT 及论文复杂度定理 | 复杂度限定于论文算法与其数学假设；实验指标限定于作者模型、数据、实现和硬件设置，不支持跨平台延迟、现代生产性能或普遍架构胜负。 |
| Albert Gu、Tri Dao、Stefano Ermon、Atri Rudra、Christopher Ré，[HiPPO: Recurrent Memory with Optimal Polynomial Projections](https://arxiv.org/abs/2008.07669)，arXiv:2008.07669，NeurIPS 2020 | §2.1—2.4，Appendix C 的一般投影与系数定义，Appendix D 中不同 measure/basis 的推导 | 时间变化的测度、有限维多项式子空间、在线系数、按测度定义的 $L_2$ 投影最优性以及相应连续/离散状态更新 | 各 measure 的历史权重与时间尺度不同；不能将一般 HiPPO 或某个指定变体描述为任意历史 token 的显式、可逆存储。本文未主张某个具体 measure 的细节。 |
| [Structured State Spaces for Sequence Modeling (S4)](https://hazyresearch.stanford.edu/blog/2022-01-14-s4-1)，Albert Gu、Karan Goel、Khaled Saab、Christopher Ré，作者团队博客，2022-01-14 | 文章开头、连续时间序列的挑战、S4 与系列论文定位 | 作者团队对问题动机、连续时间序列背景和系列结构的解释入口 | 属于发表于 2022 年的作者说明；不能单独支持跨任务、跨实现的效率或效果结论。 |
| Alexander Rush、Sidd Karamcheti，[The Annotated S4](https://iclr-blog-track.github.io/2022/03/25/annotated-s4/)，ICLR Blog Track，2022-03-25，页面标为 v2 | Part 1 的离散化、卷积与 HiPPO；Part 2 的生成函数、对角/DPLR、HiPPO 到 DPLR | 将原论文路径拆成可追踪的教学解释，并展示代码/公式如何对应 | 教学重写使用其当时 JAX/Flax 版本；不代表当前框架 API、生产实现、最佳数值方案或当前硬件表现。 |

## 未验证边界

- 未运行或复现 S4 代码、Cauchy kernel、FFT kernel、梯度或 benchmark；所有实现与速度结论均为 `source-backed`，不是本地实验。
- 未比较 S4 参数化与后续对角化 SSM 变体，也未评价新版实现、库或 GPU kernel；这不属于本次机制研究范围。
- HiPPO 的一般投影定义已核对；正文无需给出特定 LegS measure 的显式密度，因此本记录不据其推导更具体的时间权重解释。
