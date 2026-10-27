# 第五册第一章 Scaling Laws 深度研究记录（2026-10-26）

## 研究范围与结论

本记录用于精修《训练总览》§1.7.2。问题不是重复“参数、数据、计算会一起增长”，而是回答：scaling law 实际拟合什么量；Kaplan 与 Chinchilla 的预算分配为何不同；独立数据受限、预计推理量很大或面向一组不同尺寸的发布模型时，原有预训练 FLOPs 最优是否仍是正确决策；哪些证据足以迁移到具体训练项目。

研究结论是：scaling law 是针对模型家族、tokenizer、数据、训练 recipe、损失和资源范围拟合的经验预测工具。Kaplan 与 Chinchilla 处理的是固定预训练计算下的 loss 优化；数据受限与 inference-aware 工作分别改变了数据可得性和生命周期成本目标；Llama 3 技术报告展示产品系列中的不同推理预算选择。它们不构成一个可通用的固定 token/parameter 配方。实际项目应把定义、口径、拟合区间与验证集结果写清，使用成本/部署侧数据重新求解。

## 代表研究对照

| 来源 | 研究问题与核心做法 | 实验支持的结论 | 证据边界及后续问题 |
| --- | --- | --- | --- |
| [Kaplan et al., 2020, Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 用不同模型规模、数据规模和训练 horizon 的 loss 曲线拟合幂律，并估计给定计算预算时的参数、batch、step 和数据分配。 | 论文报告近似最优参数规模随计算按 C^0.73 增长、数据按 C^0.27 增长；资源增加时主要扩大参数，batch 增长而串行 step 数变化较小。 | 指数和结论绑定论文模型、数据、优化与 batch 临界值模型；D 由 batch 与 step 联系，预算分配并非独立的数据规模 sweep。它是经验前沿，不是跨训练 recipe 的固定规律。 |
| [Hoffmann et al., 2022, Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | 用固定模型规模下改变 token 数、IsoFLOP 切片、参数化 loss 拟合三种方法，估计固定预训练 FLOPs 下最小 loss 的 N/D 分配。 | 研究超过 400 个 70M—16B 参数模型、约 5B—500B token。前两种方法给出 N/D 计算指数约 0.50/0.50 与 0.49/0.51，参数化拟合约 0.46/0.54。作者进一步训练 70B、1.4T token 的 Chinchilla，以与 280B、300B token 的 Gopher 按原论文 FLOPs 账本同预算比较，在论文测量的大多数任务上表现更好。 | 最终 70B 实验高于拟合实验中 16B 左右的主要范围；大型直接对照只有 Chinchilla 与 Gopher，作者明确提醒 power-law 前沿、高计算区间曲率和单 epoch 以外的数据重复仍需研究。优化目标主要是预训练 loss。摘要和表格里的圆整 N/D 数值不能通过 6ND 粗略公式精确还原该论文的 FLOPs 账本。 |
| [Muennighoff et al., 2023, Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | 把数据集的 unique-token 预算固定，改变参数和重复 epoch，估计重复数据的衰减收益。 | 400 多个 GPT-2 系列实验，最高约 8.7B 参数和 900B seen tokens；使用 C4、OSCAR 等英文语料。在 C4 例子中，8.7B 模型把 178B unique-token 单 epoch 与 44B unique-token 四 epoch 的相同 seen-token 预算相比，后者验证 loss 高约 0.5%。拟合显示独立数据受限时，把额外算力投给更多 epoch 可优于继续扩大模型；小量重放相对新 token 的损失差异可能很小。 | 收益随重复快速递减，不能把 seen token 当成 unique token；作者报告适当数据重复仍有收益，但过多 epoch 会恶化 loss。研究集中在 GPT 架构、C4/OSCAR、英语和固定超参数，主要重复整个语料；作者把分数重复、超参数敏感性、其他语料与架构列为后续问题。 |
| [Sardana et al., 2024/2025, Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) | 把模型预计服务的请求/token 量纳入训练和推理成本，比较 FLOPs 最优与美元成本最优；使用 47 个不同模型验证极高 token/parameter 下的 loss 趋势。 | 论文认为高推理需求通常支持更小、训练更久的模型。150M 参数模型实验可达到 10,000 token/parameter；更大模型实验范围最高约 1,000 token/parameter。其 2T 推理 token、约 7.02B 请求案例中，Chinchilla 70B 的 FLOPs 只比质量相当的 compute-optimal 模型多约 1.3%，但 dollar cost 比 cost-optimal 方案多约 36%；作者将差异与推理输出 token 的 MFU 约为训练 token 的 1/50 联系起来。 | 数字绑定论文硬件、吞吐与成本模型；低参数实验的结果外推到大型模型和真实服务，不能作为云成本报价。token/parameter 趋势在所测模型上的 loss 不封顶，不等于无限重放始终有收益；部署仍须重测目标延迟、batch、输出长度、量化和 utilization。 |
| [Meta, The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | 报告实际发布模型系列如何在训练预算与推理预算之间选择。 | 旗舰模型有 405B trainable parameters、15.6T 文本 token 和约 3.8×10^25 FLOPs；作者称其尺寸接近该训练预算下的 compute-optimal。较小模型则训练得比 compute-optimal 更久，报告称在相同推理预算下所得质量更好。 | 属模型团队的训练报告而非统一协议下的独立消融。它证明团队公开采用了这种取舍，不代表所有模型系列、所有下游任务或单个 8B 模型都应使用固定 token 比例。 |
| [Li et al., 2025, (Mis)Fitting: A Survey of Scaling Laws](https://arxiv.org/abs/2502.18969) | 跨论文检查 scaling law 函数形式、训练设置、数据采集、评估和拟合方式是否被充分报告。 | 调查 51 篇论文，其中 45 篇量化 scaling 关系；只有 40 篇描述训练设置、36 篇定义方程变量、29 篇说明评估、28 篇说明 curve fitting，表中 19 篇提供分析代码。作者指出这些选择能改变拟合指数和最优 token/parameter 结论，并给出复现报告清单。 | 调查计数仅描述被纳入的论文；它不裁决某个指数对所有模型都错误。其价值是证明比较 scaling law 时必须看训练 recipe、变量定义、FLOPs/参数计数、评估和拟合，而不能只摘一个结果。 |

## 关键机制与比较

Chinchilla 对 loss 拟合的一种形式为：

~~~math
\hat L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
~~~

其中 E、A、B、α、β 由实验拟合。若用 C≈6ND 近似固定预训练 FLOPs，则提高 N 会压缩可用于 D 的预算。参数化前沿的两个 exponent 来源于拟合出的 loss 曲线，而不是 6ND 单独推导出的固定 token/parameter 常数。论文中模型/数据参数的计数方式也需要与目标项目保持一致。

Kaplan 与 Chinchilla 的区别不能概括为“一个只看模型，另一个只看数据”。两者都在建模 loss，但对数据规模、训练时间、batch、不同训练 horizon 和固定 FLOPs 预算前沿的实验设计及参数化模型不同。Chinchilla 的改进包括三种独立估计路径、扩大固定预算下的 token 量实验，并用 70B 大规模模型检查其对 Gopher 预算的预测；其局限则包括主要 fit 规模较小、单 epoch 假设和大型直接比较数量少。

后续研究分别松开了 Chinchilla 问题中的约束：

1. **独立数据受限。** 当 unique corpus 到顶时，应报告 seen tokens、unique tokens、去重/采样方法和 epoch。重复数据会提供学习信号，但其边际收益随曝光递减；需要在目标数据与优化器上测量，不可直接使用“20 token/parameter”替代数据审计。
2. **推理需求显著。** 预训练 FLOPs 最优衡量的是训练期的算力分配；大量 serving 请求会使模型大小带来持续的推理计算和显存成本。Sardana 等把推理需求放进目标函数，并显示 FLOP 成本与货币成本可能给出不同排名。
3. **模型系列有不同推理档位。** Llama 3 报告采用旗舰近似 compute-optimal、较小模型训练更久的组合。该例说明要分别规划旗舰、低延迟、小显存或高并发模型，而不是为整个产品系列套单一比例。
4. **拟合本身需要审计。** scaling law 是由训练样本拟合得出；改变变量定义、tokenizer、数据混合、训练 schedule、参数计数、loss、评估点插值或拟合目标，都会影响 extrapolation。复现信息不全时，应把预测区间缩小，不能把作者曲线提升为行业保证。

作者维护的 [(Mis)Fitting 实验仓库](https://github.com/hadasah/scaling_laws)公开了论文图表/表格数据、训练配置和拟合脚本，README 称训练代码由 open_lm fork 修改，并注明部分模型 checkpoint 尚缺。它可用于追查拟合流程和做复现入口；由于仓库位于动态 master 且产物不全，本次只把它作为作者代码与配置来源，未声称已在本地复现。

## 数值例子与读者决策

用近似公式计算两个预算：

~~~math
C_A=6(40\times10^9)(2\times10^{12})=4.8\times10^{23}
~~~

~~~math
C_B=6(80\times10^9)(1\times10^{12})=4.8\times10^{23}
~~~

两组配置有相同的一阶训练 FLOPs，但 B 参数量翻倍；在同一推理技术栈下，B 通常需要更多权重显存并做更多每 token 参数计算。A 使用了两倍 seen tokens；若控制在相同 epoch 数内，就需要更多独立语料，若复用数据则需要评估重复收益。两个方案谁的验证 loss、下游能力、训练 wall-clock 或 dollar cost 更好，不能从乘法直接得出；需要同 recipe 的实验矩阵、目标任务切片、目标硬件吞吐和真实 serving 账本。

因此计划一个 scaling study 时至少记录：

- 要优化的目标：训练 loss、下游质量、固定延迟质量，还是预训练加推理总成本。
- 变量定义：total/trainable/non-embedding/active parameters；训练 seen tokens、unique tokens、上下文长度、数据重复次数；FLOPs 计数口径。
- 控制条件：tokenizer、数据版本和 mixture、模型架构、优化器、学习率和 schedule、batch、precision、硬件拓扑及训练稳定性。
- 数据采样点：不同 N、D 和 FLOPs 的矩阵；训练曲线的检查点与评估插值；独立模型规模/预算上的外推校验和拟合误差。
- 服务目标：预计请求量、输入/输出 tokens、batch、目标 p50/p95 latency、并发、GPU 利用率、硬件价格和模型权重驻留成本。

## 旧稿知识点覆盖图

本轮只扩写 1.7.2，修订资料阅读清单、章节练习和总结；没有删掉原有核心主题。逐节去向如下：

| 原章节 | 原有知识点 | 本轮处理 |
| --- | --- | --- |
| 1.0—1.6 | 训练证据链、目标定义、历史链路、训练阶段、自回归目标、交叉熵、label shift、mask、token/batch/epoch | 全部保留，正文没有重写这些段落。 |
| 1.7.1 | 6ND 估算和计算边界 | 保留；1.7.2 引用它说明固定 FLOPs 预算的近似约束。 |
| 1.7.2 | scaling law 的经验性、Chinchilla 联合规划、没有通用 token/parameter 比、数据/架构/部署约束 | 原结论全部保留并展开为公式、研究比较、数值例子和规划流程。 |
| 1.7.3 | Dense/MoE 总参数、active 参数、路由、token 与服务账本 | 全部保留。 |
| 1.8—1.10 | 训练时间、GPU 小时、显存、Tokenizer/数据/架构联动、optimizer update | 全部保留。 |
| 1.11—1.12 | Base/Instruct/Chat/Reasoning、SFT、偏好优化和验证器边界 | 全部保留。 |
| 1.13—1.16 | 监控、checkpoint、失败链、可复现交付 | 全部保留。 |
| 1.17 | Python 训练规划账本与校验条件 | 代码不变；本轮没有执行。 |
| 1.18 | 资料可支持与不可支持的结论 | 保留原证据类型矩阵，追加 scaling-law 综述、重复数据、推理成本和 Llama 3 来源。 |
| 1.19—1.20 | 原七项练习和训练总览总结 | 原练习保留，新增第八项固定预算选型练习；加强总结中的 scaling-law 条件与外推边界。 |

## 来源版本差异与未解决项

- Google DeepMind 2022 年 Chinchilla 博客把模型数据量写为 1.3T token；同年论文表格和摘要写为 1.4T。正文使用论文中的 1.4T，并记录这是来源版本/表达差异，不混用两个数。
- Muennighoff 的受控重复实验依赖英文 C4/OSCAR、GPT 变体和具体 schedule；其他语言、混合新旧数据、训练中途重复及现代模型架构需要重新测量。
- Sardana 论文的推理阶段 MFU、服务成本与请求 token 口径有指定假设；正文只用来解释目标函数变化，不迁移为目标公司的账单。
- Llama 3 的同推理预算质量比较是 Meta 报告的作者结论；本轮没有独立权重复现。
- 本轮未运行章节 Python 账本、缩放实验、训练代码、成本仿真或真实集群；数值例子为按 6ND 的手算，不代表实际吞吐/成本。

## 来源清单

1. [Kaplan et al., Scaling Laws for Neural Language Models, 2020](https://arxiv.org/html/2001.08361)：§4—6 及 Table 6，loss 随 N/D/compute 和固定预算分配。
2. [Hoffmann et al., Training Compute-Optimal Large Language Models, 2022](https://arxiv.org/html/2203.15556)：§3 各类估计与指数，§4 Chinchilla 70B 训练和评测，§5 limitations。
3. [Google DeepMind, An empirical analysis of compute-optimal large language model training, 2022](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)：官方发布解释和 Gopher 同 FLOPs 对比；页面与论文报告的 Chinchilla token 数表达略有差异。
4. [Muennighoff et al., Scaling Data-Constrained Language Models, 2023](https://arxiv.org/html/2305.16264)：§3—6、附录 Q，重复语料模型、C4/OSCAR 结果、过拟合和超参数限制。
5. [Sardana et al., Beyond Chinchilla-Optimal, arXiv v3, 2025-04-14](https://arxiv.org/html/2401.00448)：摘要、§2、§4、§6，FLOPs/dollar cost、请求量、MFU 和极端 token ratio 的范围。
6. [Meta, The Llama 3 Herd of Models, 2024](https://arxiv.org/html/2407.21783)：摘要和 §3，405B/15.6T 训练设置、训练预算最优估计及较小模型长训练报告。
7. [Li et al., (Mis)Fitting: A Survey of Scaling Laws, 2025](https://arxiv.org/html/2502.18969)：摘要、Table 1、§3—7，51 篇论文的报告情况及拟合方法敏感性。
8. [(Mis)Fitting 作者复现实验仓库](https://github.com/hadasah/scaling_laws)：README 和分析入口；用于追查代码/配置，不是已完成的本地复现证据。
