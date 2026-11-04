# 研究方法与复现：来源证据

记录日期：2026-11-03  
范围：研究主张、复现术语、AI 研究复现案例、官方投稿清单及章内方法案例。本文只记录联网核验到的知识事实、来源版本和适用边界。

## 1. AI 研究复现案例

- **来源与版本：** [The Unreasonable Effectiveness of Open Science in AI: A Replication Study](https://doi.org/10.1609/aaai.v39i25.34818)，AAAI 2025；同时核阅 [arXiv:2412.17859v1](https://arxiv.org/abs/2412.17859v1) 的摘要、结果表和限制部分。
- **来源支持的事实：** 作者最初纳入 30 篇高引用 AI 研究；8 篇因数据收集或专用硬件条件无法实际获得而排除，最终尝试复现 22 篇。结果为 6 篇完全成功、5 篇部分成功，合计 11/22（50%）至少部分得到复现。在同时开放代码和数据的 7 篇中，5 篇成功、1 篇部分成功（6/7，约 86%）；仅开放数据的 15 篇中，1 篇成功、4 篇部分成功（5/15，约 33%）。
- **证据等级：** 对精选 AI 研究进行的第三方复现研究；数字是作者对其样本和复现判定方式的报告。
- **边界：** 样本为小规模、高引用且主要来自复现清单普及前的研究；估计存在较大不确定性。每篇最多投入 40 个专注工作小时，只有 3/22 篇完成了文中所有实验。未得到结果可能源于论文描述不足，也可能源于复现者的知识或实现错误；不能把 50% 当作当代 AI 论文的普遍复现率，也不能把“部分成功”当成所有主张均获支持。

## 2. 机器学习研究中的“复现”术语

- **来源与版本：** [What Do Machine Learning Researchers Mean by “Reproducible”?](https://arxiv.org/abs/2412.03854v1)，arXiv v1，AAAI 2025 Senior Member Presentation Track；核阅引言、表 1 和研究范围说明。
- **来源支持的事实：** 作者回顾 2017 年后 101 篇论文，将文献中常被归入“reproducibility”的科学严谨性工作归纳成八个方面：repeatability、reproducibility、replicability、adaptability、model selection、label/data quality、meta & incentive、maintainability。前三项沿用 ACM 术语框架，其余来自作者对文献主题的归纳。
- **证据等级：** 对文献主题的人工综述与分类，不是统一术语标准或系统性元分析。
- **边界：** 作者指出社区对 reproducibility 等词存在不一致甚至不兼容的使用。该综述可支持“定义本书自己的操作目标并说明任务范围”，不能支持声称八类分类已被所有研究者接受。

## 3. NeurIPS 论文清单和代码/数据指南

- **来源与版本：** [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) 与 [Code and Data Submission Guidelines](https://neurips.cc/public/guides/CodeSubmissionPolicy)，于 2026-11-03 查阅的官方公开页面。
- **来源支持的事实：** Paper Checklist 要求作者按问题回答 yes、no 或 n/a，并允许简要说明；指南覆盖主张与范围、局限、理论假设、实验可复现路径、数据/代码、实验细节、统计显著性和计算资源等。页面说明 no 或 n/a 本身通常不构成拒稿理由，代码开放也不是所有论文的唯一复现路径。官方 Code and Data 指南鼓励在主要贡献依赖实验结果时提交相应代码；它也列出专用硬件、非开源或非免费依赖等例外情形。
- **证据等级：** 会议组织者发布的投稿政策和清单说明。
- **边界：** 页面内容会随会议年份更新。实际投稿应重新核对目标年份的官方指南；本条仅记录 2026-11-03 访问时的公开页面，不能替代投稿要求。

## 4. LoRA 与 DPO 的原论文实验范围

- **LoRA 来源与版本：** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685v2)，arXiv v2，2021-10-16。
- **来源支持的事实：** 方法冻结预训练权重，并在 Transformer 层注入可训练低秩分解矩阵。论文在 RoBERTa、DeBERTa、GPT-2 和 GPT-3 的指定任务及实现设置中报告，与全参数微调相比达到相当或更好的模型质量表现。
- **证据等级与边界：** 原始方法论文及其作者实验；不能从这些模型和任务外推到任意底座、rank、目标模块、数据规模、任务或训练配方。
- **DPO 来源与版本：** [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290v3)，arXiv v3，2024-07-29。
- **来源支持的事实：** 论文在其偏好/RLHF 建模假设下给出 DPO 的分类式目标，并报告情感控制、摘要和单轮对话等实验与 PPO-based RLHF 的比较。
- **证据等级与边界：** 原始方法论文及其作者实验；结论绑定论文的数据、模型、任务、偏好标注和训练配置，不能概括为 DPO 总体优于所有 RLHF 或偏好优化方法。

## 5. RULER 长上下文评测例

- **来源与版本：** [RULER: What’s the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654v3)，COLM 2024，arXiv v3，2024-08-06。
- **来源支持的事实：** RULER 在 Needle-in-a-Haystack 检索任务基础上提供可配置序列长度和任务复杂度，扩展到多 needle 变体、多跳追踪和聚合任务。论文据此指出单一 needle 检索无法覆盖长上下文理解的全部能力。
- **证据等级与边界：** 特定合成 benchmark、任务定义和论文所测模型的评估结果；不等于对任意长上下文任务或生产数据的完整测量。

## 6. 数值案例边界

- 章内 reranker 的 `142/200`、`156/200`、`420 ms` 和 `760 ms` 是明确构造的教学输入，不是论文或本地实验结果。算术直接由这些输入推出：支持率 `0.71` 与 `0.78`，差值 `0.07`（7 个百分点），相对基线增幅约 `9.9%`；延迟差 `340 ms`、相对增幅约 `81%`。
- 只有边际支持率而没有逐题的配对变化、评分协议和不确定性信息，不能据此判断配对显著性或因果归因。教学输入只用于练习分母、效应差、成本和证据边界。

## 7. 论文阅读的三遍法

- **来源与版本：** S. Keshav, “How to Read a Paper,” *ACM SIGCOMM Computer Communication Review*, 37(3), 83–84, July 2007；核阅 DOI 元数据及作者公开的两页论文副本，副本首页标注 Version of February 17, 2016。
- **来源支持的事实：** 作者将方法描述为最多三遍：第一遍快速建立全貌并判断是否继续，第二遍更仔细地理解内容，第三遍尝试按作者假设进行虚拟重建、检查隐含假设和遗漏。原文将第一遍时间估为约 5–10 分钟，并指出第三遍耗时随读者经验和论文而变。
- **证据等级：** 方法论文的一手文本与出版信息。
- **边界：** 这是作者提出的实用阅读方法，不是比较研究证明的最优流程。本书在第一遍前增加版本/阅读目标准备、把 claim 账本和证据边界嵌入第二遍，属于本书的操作化扩展；不应归功于 Keshav 原文。

## 8. 机器学习报告清单的适用范围

- **REFORMS 来源与版本：** Kapoor 等人，[REFORMS: Reporting Standards for Machine Learning Based Science](https://arxiv.org/abs/2308.07832v2)，arXiv v2，2023-09-19；核阅摘要、范围说明和清单结构。
- **来源支持的事实：** REFORMS 包含 32 个问题、8 个模块及配套指南，基于 19 位跨计算机科学、数据科学、数学、社会科学和生物医学研究者的共识过程。作者将核心对象定义为“用机器学习模型表现支持科学主张”的科学研究，并明确区分它与通常旨在改进通用机器学习方法的 ML methods research；论文指出部分 REFORMS 条目仍可能对方法研究有价值。
- **证据等级与边界：** 作者提出的报告标准及其适用说明；不证明每个条目适用于所有大模型论文，也不是质量分数或已普遍采纳的统一标准。
- **NeurIPS 官方指南与版本：** [Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) 与 [2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook)，于 2026-11-03 查阅。官方说明 2026 主会论文模板包含必填 checklist，目标是促使作者考虑可复现性、透明度、研究伦理和社会影响。通用指南要求按具体问题回答 yes/no/n/a，并允许说明；它定位为反思与评审信息，不会自动证明研究正确。会期规则可能变化，投稿要求须回查目标年份页面。

## 9. 当前论文评审维度与贡献类型

- **NeurIPS 主会来源与版本：** [NeurIPS 2026 Reviewer Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines)，于 2026-11-03 查阅贡献类型和通用标准部分。
- **来源支持的事实：** 2026 指南列出 General、Theory、Use-Inspired、Concept & Feasibility 和 Negative Results 等贡献类型；评审表保持一致，但各类贡献对标准的解释不同。通用标准为 Quality、Clarity、Significance、Originality；原创性不要求一律提出全新方法，也可以表现为对既有方法的新认识、效率或公平改进。Negative Results 类要求负结果有分析/实验支撑并能改变对重要问题的认识；指南称该类的重要性与原创性门槛较高，且不要求必须提供缓解方案。
- **NeurIPS Evaluations & Datasets 来源：** [2026 E&D Reviewer Guidelines](https://neurips.cc/Conferences/2026/EvaluationsDatasetsReviewerGuidelines)，于同日查阅。该赛道使用通用四维标准并按贡献类型补充解释，对数据、benchmark 和可复用执行工件强调负责任数据实践、透明度、文档和可复现性。
- **ICLR 来源与版本：** [ICLR 2026 Reviewer Guide](https://iclr.cc/Conferences/2026/ReviewerGuide)，于同日查阅 FAQ。页面说明没有达到 state-of-the-art 单独不足以构成拒稿理由，价值判断取决于是否可信地展示新颖、相关、有影响的知识；同一页面还包含针对该会期的近期工作引用规则。
- **证据等级与边界：** 会议官方评审说明，可支持“评审维度和贡献类型由具体会期/赛道解释”及“非 SOTA 结果仍可能有价值”；不能作为跨会议统一的论文质量标准，也不能单独判断某篇论文满足这些标准。指南会随会期和赛道变化，引用前应回查对应官方页面。

## 10. FlashAttention 原论文的训练时间比较

- **来源与阅读位置：** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，核阅摘要、算法说明和 BERT-large 表 1（原文第 4.1 节）。
- **来源支持的事实：** 主 FlashAttention 通过分块和在线 softmax 避免把完整 attention 矩阵反复写入/读取 GPU HBM，计算目标为 exact attention；论文另把 block-sparse FlashAttention 作为 approximate attention 扩展。BERT-large 训练比较使用相同的 MLPerf 初始权重、Wikipedia 和 8 张 A100，训练到 72.0% masked-language-modeling 目标准确率；各方法报告 10 次运行平均时间：MLPerf 1.1 的 NVIDIA 实现为 `20.0±1.5` 分钟，FlashAttention 为 `17.4±1.4` 分钟。
- **直接算术：** 墙钟时间减少 `(20.0−17.4)/20.0=13%`；按速度比计算为 `20.0/17.4≈1.149×`，约快 14.9%。两种表述分母不同。
- **证据等级与边界：** 原始算法与实验论文、作者报告结果。数字只支持论文指定的模型、数据、初始化、硬件、训练目标和实现对照；不能外推到任意 GPU、shape、dtype、推理服务、当前 runtime 或其他 attention kernel。报告中的 `±` 原样保留，不在未核实统计定义时将其改称标准差或置信区间。

## 11. MLRC 2026 复现研究 track

- **来源与日期：** NeurIPS Communication Chairs，[MLRC 2026: Reproducibility as an Official Track at NeurIPS](https://blog.neurips.cc/2026/05/04/mlrc-2026-reproducibility-as-an-official-track-at-neurips/)，2026-05-04。
- **来源支持的事实：** MLRC 2026 被宣布为 NeurIPS 官方 track；征稿范围包括针对具体论文主张的复现/重复研究、拓展至新设置或模型的泛化研究、跨论文的 meta-reproducibility、方法工具、AI-assisted reproducibility，以及以 AI 系统或 Agent 为研究对象的复现。公告明确说部分失败和负结果也有价值，并强调应报告具体主张、测试边界和新增认识。
- **证据等级：** 会议组织者关于一个会期及其 track 的公告。
- **边界：** 该公告描述特定 track 的征稿范围和组织安排，不是跨领域统一定义，也不表示任何单次失败都构成研究贡献。日期、资格和投稿安排只适用于 2026 会期。

## 12. PyTorch 2.15 随机性说明

- **来源与版本：** [PyTorch 2.15 Reproducibility documentation](https://docs.pytorch.org/docs/2.15/notes/randomness.html)，页面标注 2026-05-14 更新；2026-11-03 查阅。
- **来源支持的事实：** 文档说明，跨 PyTorch 发行版、独立提交版本和平台不保证完全可复现；CPU/GPU 使用相同 seed 也不保证相同结果。它区分设置随机数生成器 seed 与启用确定性算法，并指出确定性操作往往较慢。PyTorch seed 之外，Python、NumPy 和其他库的随机源也可能需要各自控制。
- **证据等级：** 特定 PyTorch 文档版本对公开运行时行为与配置选项的说明。
- **边界：** 该页面说明 PyTorch 2.15 文档覆盖的操作，不代表任意扩展、驱动或第三方框架的全部行为；跨版本复现应记录完整软件和硬件栈，并查阅实际运行版本的文档。

## 13. pass@k 的无偏估计

- **来源与版本：** Chen 等人，[Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)，ICML 2021 / arXiv 2021；核阅 §2.1 与附录 A。
- **来源支持的事实：** 对每个任务生成 n≥k 个候选、其中 c 个通过单元测试时，论文使用 1−C(n−c,k)/C(n,k) 估计从候选池无放回抽取 k 个至少含一个正确解的概率，并称该估计为无偏估计。论文指出，直接将经验 pass@1 代入 1−(1−p)^k 会产生向下偏差；评估还绑定 HumanEval 测试、每题样本数和温度等协议。
- **证据等级：** 指标原始方法论文及作者实验协议。
- **边界：** 该估计针对给定候选池、通过测试的定义和抽样口径；测试不完整、候选生成策略不同、候选间相关性或执行环境变化都会影响结果解释。pass@k 不等同于单次正确率，也不能独自代表一般代码能力。

## 14. 随机区组与分数因子设计

- **补充核验日期：** 2026-11-04。
- **来源与阅读位置：** NIST/SEMATECH [Randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)，阅读区组原理与 semiconductor furnace 示例；[Full factorial designs](https://itl.nist.gov/div898/handbook/pri/section3/pri333.htm)，阅读两水平全因子运行数；[How to model DOE data](https://www.itl.nist.gov/div898/handbook/pri/section4/pri43.htm)，阅读分数因子的建模和 alias structure 段落。
- **来源支持的事实：** NIST 将区组用于控制可识别的重要 nuisance factors，并建议区组内比较处理水平；其概括原则为“block what you can, randomize what you cannot”。两水平、k 因子的完整全因子设计包含 `2^k` 个组合；分数因子筛选时需知道 alias structure，模型必须只包含设计能够区分的效应，并依赖被混杂效应足够小等假设。
- **证据等级：** NIST/SEMATECH 统计设计手册中的 DOE 概念和教学案例。
- **边界：** 手册给出通用实验设计原则，不提供大模型、GPU、服务负载或 LLM judge 的专属测量证据。将硬件、时间段或评审顺序作为区组/随机化对象，是按原则作出的实验设计应用，具体随机单位仍需符合实际系统和抽样过程。

## 15. 机器学习评估中的推断性复现方法

- **来源与版本：** Hagmann、Meier 与 Riezler，[Towards Inferential Reproducibility of Machine Learning Research](https://arxiv.org/abs/2302.04054v7)，arXiv v7，2023-10-09，发表于 ICLR 2023；核阅 §1—5、线性混合效应模型、GLRT、方差分量分析和 worked example。
- **来源支持的事实：** 论文将训练运行间的评估分数差异视为包含算法、超参和数据属性等来源的 measurement noise，提出使用线性混合效应模型、广义似然比检验分析模型差异，并用方差分量分析估计不同噪声来源占比和可靠性。方法可保留重复测量的测试句子结构，并评估算法设置与输入特征的条件交互。
- **证据等级：** 机器学习评估统计方法论文，包含作者使用的数据和案例。
- **边界：** 论文提出的是面向特定评估设计的统计分析路线，不是所有大模型消融都必须采用的统一程序；混合模型需依据实际随机化、重复测量和分布假设设定。不能将其案例中的显著性或可靠性结果外推到本地模型、其他任务或服务系统。

## 16. 少量 seed 下的配对 bootstrap 预印本

- **来源与版本：** Wenzhang Du，[When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements](https://arxiv.org/abs/2511.19794)，arXiv v1，提交于 2025-11-24；核阅 §3—6、实验设置、结果和限制。
- **来源支持的事实：** 该单作者预印本提出按 seed 配对 baseline 与 variant，计算逐 seed 差值，再结合 BCa bootstrap 区间和符号翻转置换检验；其测试对象为 CIFAR-10、CIFAR-10N 和 AG News，并通过合成无改进/小增益/中等增益情景比较单次运行、非配对 Welch 检验和作者协议。作者报告在其三 seed 设置中，配对方案未将 0.5—2.0 个百分点的小增益判为显著。
- **证据等级：** 尚未证明为普遍方法的单篇 arXiv 预印本和其自设实验。
- **边界：** 数据集为图像分类和新闻分类，不含 LLM benchmark；收益情景为作者构造，严格双门槛偏保守且会降低检验功效。可用来说明 seed 很少时结论分辨率有限，不支持把该协议直接规定为所有 LLM 研究的默认检验，也不能把“未显著”解释为无效。

## 17. GSD-front 多指标文本评价候选

- **来源与版本：** Esteban Garces Arias 等，[Statistical Multicriteria Evaluation of LLM-Generated Text](https://arxiv.org/abs/2506.18082)，arXiv v2，2025-06-24；阅读摘要、方法描述与实验/限制部分。
- **来源支持的事实：** 论文把 Generalized Stochastic Dominance（GSD）用于多项 LLM 文本指标比较，提出 GSD-front 以部分序合并 cardinal 自动指标与 ordinal 人评，避免先指定任意权重；论文还处理其抽样设计可能偏离 i.i.d. 的情况。
- **证据等级：** 一篇文本生成评价方法预印本及作者实验。
- **边界：** 结果绑定开放式文本续写、有限解码策略和其人评协议；作者/有限评分者的偏好、任务类型及方法依赖均限制外推。它是多指标评价的专门候选方法，不意味着约束/Pareto 分析失效，也未作为本章的通用决策规则。
