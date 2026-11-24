# Chinchilla、PaLM 与 Compute-Optimal Training 研究记录

## 研究问题与范围

核对“给定固定预训练计算预算时怎样分配模型参数与训练 token”的研究目标、Chinchilla 的估计与大模型验证、PaLM 的训练规模和系统证据，以及该结论对数据与训练协议的依赖。检索和阅读日期：2026-11-23。该记录用于第二十一册第五十二章，不将预训练损失最优扩展为所有任务、产品或推理阶段的最优。

检索词包括：`compute-optimal language model fixed FLOPs parameters tokens`、`Chinchilla 400 models data scale 70B 1.4T`、`PaLM 540B 780B TPU v4 Pathways`、`Kaplan Chinchilla discrepancy warmup FLOPs optimizer`、`data-constrained scaling repeated epochs`、`CS336 scaling laws lecture`。

## 主要发现

1. **Compute-optimal 有明确目标函数和资源边界。** Hoffmann 等人将它定义为：在固定训练 FLOPs 预算下，选择模型参数量与训练 token 数，以降低最终预训练损失。他们以超过 400 个、约 7,000 万至 160 亿参数的模型，配合约 50 亿至 5,000 亿 token 的训练实验拟合预算前沿。工作报告参数和训练 token 随预算大致等比例扩展；$D/N\approx20$ 是该研究结果的常见概括，具体比例仍应绑定实验、模型和数据口径。
2. **Chinchilla 的大模型试验是模型/数据分配的案例。** 作者在与 Gopher 相同的报告训练预算下，训练约 700 亿参数、1.4 万亿训练 token 的 Chinchilla，对照约 2,800 亿参数、3,000 亿 token 的 Gopher；论文报告 Chinchilla 在大多数被测下游任务上表现更好。研究者也说明，昂贵的大模型预算只有 Chinchilla 和 Gopher 两个可比较训练点，预测还依赖拟合前沿的形式，因此这些结果不是无限外推保证。
3. **$6ND$ 是 dense Transformer 的近似 FLOPs 账本，不是硬件实测。** 其中 $N$ 表示参数量、$D$ 表示训练 token exposure 数。Chinchilla 附录说明其详细 FLOPs 估算包含 embedding、attention、dense block 和 logits，并以反向传播约为前向传播两倍的假设比较常见 $6ND$ 近似；论文所测规模下，两者差异较小。长上下文 attention、重计算、MoE 路由、通信和设备利用率仍要求更精细或实现级测量。
4. **PaLM 提供规模化系统的对照案例。** PaLM 论文报告了约 5,400 亿参数的 dense 自回归 Transformer，使用 7,800 亿 token 的训练数据、Pathways 和 6,144 个 TPU v4 芯片；系统章节报告 model-FLOPs utilization（$46.2\%$）与 hardware-FLOPs utilization（$57.8\%$）。前者只计模型有效 FLOPs，后者还计入 rematerialization 的重计算 FLOPs；两者都以芯片理论峰值为分母。论文把 PaLM 作为模型、数据与多 TPU Pod 系统扩展案例，并没有把它与 Chinchilla 设为相同训练预算下的单变量实验。该论文称数据训练一个 epoch。比较不同论文的 $N$ 与 $D$ 不能绕过预算、tokenizer、数据和训练设置差异。
5. **Chinchilla 与 Kaplan 的分歧具有方法论因素。** NeurIPS 2024 的复现实验在 OpenWebText2 与 RefinedWeb 上分析了末层 FLOPs 计数、学习率 warmup 长度和按模型规模调参；作者报告修正这些因素后，compute-optimal 估计更接近 Chinchilla 的趋势。该结果说明 scaling 前沿依赖成本定义和训练 recipe，不是证明所有环境都遵循固定比例。
6. **有限唯一数据会改变优化问题。** 2025 年 JMLR 的 data-constrained 研究显式区分唯一 token 与重复训练 token，并用超过 400 个、约 1,000 万至 90 亿参数、最长 9000 亿训练 token 的实验研究重复数据。论文在其设置下观察到少量重复训练可接近新数据的损失效果，重复增多后收益下降；这不等于所有语料都具有统一重复阈值，而是说明在唯一数据预算受限时要增加数据约束条件。
7. **教学组织上应把“预算分配”与“系统扩展”分开。** Google DeepMind 的论文作者说明文章以固定计算预算问题引入参数/数据选择，Stanford CS336 当前课程安排将 scaling laws 列为独立讲次；这些资源适合学习概念组织。CS336 页面能确认讲次安排，本次浏览未能直接读取其 GitHub PDF 讲义，故不据此引用具体讲义结论。

## 来源与证据边界

| 来源 | 类型与阅读位置 | 支持的内容 | 不支持的外推 |
| --- | --- | --- | --- |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)（Hoffmann 等，2022） | 原始论文；摘要、§1、§4、§5、Appendix F–G | 固定训练 FLOPs 下的预训练 loss 目标、400 多模型的估计、Chinchilla/Gopher 案例、FLOPs 近似及大模型验证限制 | 任意模型、语料、目标或 serving 工作负载中的固定最优比例 |
| [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)（Chowdhery 等，2022） | 原始论文；摘要、§2–5、§13、Appendix B/F | 540B/780B 模型与数据、Pathways 多 Pod 训练、TPU v4 系统、FLOPs utilization 口径、任务结果与继续训练案例 | 将 PaLM 与 Chinchilla 视为相同预算、相同数据和训练协议的受控对照 |
| [An empirical analysis of compute-optimal large language model training](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)（Hoffmann、Borgeaud、Mensch、Sifre，Google DeepMind，2022） | 作者/机构说明文章；2022-04-12，正文固定预算问题、Figure 1–2 与 PaLM 比较段 | 参数量与数据量预算对照的教学路线；作者对 PaLM 预算的条件化预测 | 独立复现或普遍最优保证。文章写 PaLM 训练量为 768B token，而 PaLM 论文报告 780B；以论文值为正文事实口径 |
| [PaLM: Scaling Language Modeling with Pathways — Google Research](https://research.google/pubs/palm-scaling-language-modeling-with-pathways/) | 官方论文入口与摘要 | 论文作者、模型和芯片规模摘要 | 不能代替论文正文的指标口径和数据细节 |
| [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b6341525cd84f3be0ef203e4d7cd8556-Abstract-Conference.html)（Porian 等，NeurIPS 2024） | 同行评审论文摘要与 [arXiv 正文](https://arxiv.org/abs/2406.19146)；引言、§§3.1–3.5、Table 1 | 指定模型/语料实验中，FLOPs 计数、warmup 和按模型规模的超参数调优如何影响 Kaplan 与 Chinchilla scaling 估计 | 其他数据、硬件、模型族或训练 recipe 下必然得到同一前沿 |
| [Scaling Data-Constrained Language Models](https://www.jmlr.org/papers/volume26/24-1000/24-1000.pdf)（Muennighoff 等，JMLR 26，2025） | 同行评审论文；摘要、§§3、5–6、11 | 有限唯一数据和重复 epoch 下的具体数据约束建模及实验结果 | 通用重复次数、任意数据集的统一最优或对 Chinchilla 非重复数据设定的否定 |
| [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540)（Muennighoff 等，2024） | 预印本；摘要和 §6 limitations | 指定规模/数据分布下超训练与下游任务表现的联合建模 | Chinchilla 预训练 FLOPs 目标本身已优化部署成本，或所有模型都会按同一方式超训练 |
| [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) | 官方课程页面；Spring 2026 lecture schedule | scaling laws 在课程中的独立讲次及与并行、推理和数据的教学位置 | 本次未能读取 lecture PDF，不能将讲次目录当作其技术结论的证据 |

## 未解决边界

不同论文中的参数计数、FLOPs 定义、数据重复、训练长度、优化器与验证目标不完全相同。预训练 loss 的 compute-optimal 前沿不能自动决定下游任务、后训练或推理部署的最优方案；当唯一数据受限或改用 MoE、多模态与长上下文架构时，需要重新定义约束并做与目标一致的验证。
