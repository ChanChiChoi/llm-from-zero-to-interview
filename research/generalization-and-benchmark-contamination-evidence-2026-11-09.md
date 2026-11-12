# 泛化、数据重复与 benchmark contamination 来源证据

检索与阅读日期：2026-11-09。

## 过参数化与 double descent

- **来源与版本：** Belkin 等，[Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off](https://arxiv.org/abs/1812.11118)，arXiv v2（2019-09-10）；页面还链接至 PNAS DOI。
- **阅读范围：** arXiv 摘要中的插值、模型容量和 double-descent 主张及论文版本元数据。
- **来源支持：** 论文提出以 double-descent 性能曲线连接传统 U 形 bias–variance 图景，讨论超过插值阈值后模型容量继续增加时性能改善的情形，并报告跨多种模型/数据的证据。
- **证据边界：** 该曲线不是所有任务、容量轴或训练配方都会出现的定律；不支持“参数越多越能泛化”的无条件结论。本章只用它限制传统单调复杂度直觉。

## 训练重复与记忆

- **来源：** Lee 等，[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)，arXiv v2（2022-03-24）。
- **阅读范围：** arXiv HTML 摘要、数据集与重复识别方法、训练/验证重叠和模型影响实验、讨论。
- **来源支持：** 作者在所研究的语言建模语料发现 exact/near duplicates 和 train-test overlap；在其模型及训练配置中，去重减少了模型输出对训练文本的逐字复制，并可减少常用数据集验证集中的训练重叠。
- **证据边界：** 数字绑定论文语料、检测阈值、模型和训练过程；不意味着所有重复都无价值，或相同去重策略能在所有目标任务中提高质量。

- **来源：** Kandpal 等，[Deduplicating Training Data Mitigates Privacy Risks in Language Models](https://arxiv.org/abs/2202.06539)，arXiv v3（2022-12-20）。
- **阅读范围：** arXiv HTML 摘要、§2 实验设置、§3 重复频次与再生成、§5 去重后的成员推断和结论。
- **来源支持：** 在论文的模型、网页语料和抽取评估设置中，训练序列重复次数与再生成频率呈超线性关系；去重后论文测得的隐私攻击成功率下降。
- **证据边界：** 再生成率不是产品请求中的泄露概率；该相关关系和攻击结果不应外推到任意现代模型、数据或提示条件。

## 改写样本与语义近重复

- **来源：** Yang 等，[Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850)，arXiv v2（2023-11-11）。
- **阅读范围：** 摘要、§2 检测方法边界、§3 改写/翻译样本构造、§4 LLM-assisted decontaminator、§5 benchmark 实验和 §6 讨论。
- **来源支持：** 论文展示 n-gram 检查会漏掉语义相同的改写或翻译样本；embedding 相似度候选检索再经 LLM 判断可形成一条更强的筛查流程，但单独的 embedding 阈值会有误报。作者对特定 13B 模型及 MMLU、GSM8K、HumanEval 的实验报告，训练于改写 benchmark 样本可显著提高相应分数。
- **证据边界：** 论文结果绑定其构造的训练数据、模型、任务和训练量；该工作将多种语义相似视作潜在污染，实际项目仍须定义“同一任务/共享解法”与一般主题相近的边界，不能把相似度直接当作污染证明。

- **来源：** Spiesberger 等，[Soft Contamination Means Benchmarks Test Shallow Generalization](https://arxiv.org/abs/2602.12413)，arXiv v1（2026-02-12；本次按预印本处理）。
- **阅读范围：** arXiv HTML 摘要、§3 语料和 benchmark 选择、§3.2–3.4 方法、§4.1–4.4 结果、§5 限制及附录中的匹配判定说明。
- **来源支持：** 研究以公开训练语料的 Olmo3 为案例，对 1% 的 base training data 分层抽样并嵌入检索候选，再检查语义匹配。作者报告在每题 top-100 cosine-similar candidates 中，100% 的 MBPP 和 77.5% 的 CodeForces 题目至少找到一个语义 duplicate。另在特定 MuSR 设置中，以 10,000 条微调样本中的 5% 换为 500 个 semantic duplicates；Olmo3 的 seen 与 within-benchmark unseen 分数相对 clean 对照都提高，但同域不同 benchmark TrueDetective 保持稳定。
- **证据边界：** top-100 命中率的分母是 benchmark 题，不是语料 token 或所有训练记录；它绑定候选检索、标注方法和开放语料，不能称为所有模型的污染率。within-benchmark unseen 提升只说明同一 benchmark 分布内部的迁移，不能单独证明通用推理泛化；论文指出在更接近真实剂量的实验中，unseen 增益证据有混合结果，且不覆盖封闭语料。

## 推理后训练检测信号

- **来源：** Wang 等，[On The Fragility of Benchmark Contamination Detection in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html)，ICLR 2026；arXiv v2（2026-03-02）。详细核读范围与边界另见[评测设计与污染检测证据记录](evaluation-design-uncertainty-and-contamination-2026.md)。
- **可复用事实：** 作者在 Qwen2.5-7B-Instruct 的 benchmark 污染 SFT 设置中，再用 10K DeepMath-103K 问题训练最多 156 步 GRPO；所测多个方法和六个数学 benchmark 的 AUROC 随 GRPO 步数下降，部分 reference-free 检测约到 0.5。论文把机制解释与 PPO-style importance sampling/clipping 联系起来。
- **证据边界：** 结果绑定基座模型、污染方式、后训练和任务；不是所有 RL、所有推理模型或所有检测器必然失效的证明。“未检出”不构成无污染的充分证据。

## contamination 对分数与排名的不同影响

- **来源：** Xiao 与 Cheng，[Contamination Inflates Scores but Rarely Reorders Large Language Model Leaderboards](https://arxiv.org/abs/2609.02899)，arXiv v1（2026-07-05）。
- **阅读范围：** 论文 PDF 摘要、引言的“绝对分数 vs 排名”区别、方法中的 anchor-item/paraphrase 对照、公开模型与注入污染模型的主结果。
- **来源支持：** 在 47 个公开模型、74 个注入已知污染量的模型和 ARC、GSM8K、HellaSwag、MMLU 四个 benchmark 上，作者报告常规与 paraphrase-controlled 排名的相关系数为 0.997；188 个模型-benchmark 组合中仅 3 个的差异污染在两个参考源上获支持。研究提出共享污染可能同时抬高绝对分数而较少改变名次的解释。
- **证据边界：** 这是单篇 2026 预印本，结论绑定其模型集合、四个 benchmark 和 anchor-invariance 操作化；它不说明原分数无偏，也不证明其他榜单或存在差异污染时排名稳定。
