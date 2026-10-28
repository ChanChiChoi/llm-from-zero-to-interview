# 第九册第四章：去重与污染检测研究与精修记录

## 本轮范围

依据当前 `WRITING_SPEC.md` 从头复读第四章 23 个二级讲次及完整代码示例，围绕两个核心问题核对正文：如何控制训练集内部的重复暴露；怎样判断训练、后训练和评估迭代是否接触过 benchmark 信息，以及这类证据允许支持什么结论。

保留单篇结构，不新增或拆分章节。第四章继续主讲重复关系、候选召回、保留策略与评估污染证据；第三章主讲质量过滤与阈值代价，第五章主讲训练 mixture 和有效 exposure，第六章主讲代码/数学专项数据契约，第十一章主讲 manifest、lineage、删除和版本治理。本轮在第四章增加面向训练数据和模型评估两条分支的标准 Mermaid 图及其共享审计证据汇合点。

## 定向研究问题

1. 近重复是否应只以文档为单位、全量删至单份；片段粒度、重复频次、长度与保留预算有哪些证据？
2. 去重后频率如何改变训练分布；公开研究里的重复收益与性能下降分别绑定哪些模型、token 预算和 recipe？
3. 有访问语料与只能访问模型 API 时，各污染检测方法直接观测的对象和未观测对象是什么？
4. benchmark 改写、动态刷新和隐藏留出集是否能同时保持能力构念和评估独立？

## 来源与证据范围

### 去重、重复训练与文档频次

- [Datasets, Documents, and Repetitions: The Practicalities of Unequal Data Quality](https://papers.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html)，NeurIPS 2025。阅读全文和官方 proceedings 页面。研究使用 DCLM-baseline、RefinedWeb 与 C4，并按不同模型规模和 token/compute 预算比较重复。其 22-task centered-core 结果显示，在 recipe 适配（尤其重复数对应的 weight decay）后，DCLM-baseline 最多重复十轮可在作者测试的多档预算中优于十倍规模 RefinedWeb 单轮训练。DCLM-baseline 是按 FastText classifier 分数从 RefinedWeb 取前 10% 文档。论文也报告过度重复的递减收益。
- 同篇论文的文档级计数操纵实验使用 7B 模型和 138B token DCLM-baseline；22 任务 CORE 分数依次为 duplicate-aware subsampling 34.1、global deduplication 后抽样 41.7、uniform subsampling 44.4、保留重复数不少于 7 的文档再去重 45.8。作者认为在该过滤语料中重复频次与被过滤器选中的质量信号相关。这个结果只说明频数可携带有条件的信息，不等于应保留重复，也不能跨语料外推。
- [Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining](https://arxiv.org/abs/2608.03089)，2026-08 arXiv 预印本。直接阅读 arXiv 正文的摘要、方法、表格和实验设置。论文把检测和保留策略分离：自然边界切分、规范化精确 hash、分布式全局频次计数，再按重复组频率/片段长度分配保留预算，并将连续候选删除跨度合并，减少文档破碎。FineWeb-Edu 与代码网页实验以 Hy-MT2-30B-A3B 训练 36,000 步、约 294.9B token。FineWeb-Edu 八项基准均分中完整方案为 52.90，Suffix-Array 为 52.79，Keep-One 为 52.14；代码网页五项均分为 41.40，对照 No-Dedup 为 37.98。数字绑定作者的语料、模型、采样、预算和 benchmark；论文没有给出可直接移植的通用保留参数。
- MinHash/LSH 与 SimHash 的方法出处分别核对 A. Z. Broder 的 [On the Resemblance and Containment of Documents](https://doi.org/10.1109/SEQUEN.1997.666900)、Indyk 与 Motwani 的 [Approximate Nearest Neighbors](https://doi.org/10.1145/276698.276876)、Charikar 的 [Similarity Estimation Techniques from Rounding Algorithms](https://doi.org/10.1145/509907.509965)。本章中的小集合与 LSH 参数计算是教学推演，不是论文实验结果。

### benchmark 污染检测和缓解

- [Data Contamination Quiz](https://aclanthology.org/2025.tacl-1.37/)，TACL 2025。阅读全文及 ACL Anthology PDF。DCQ 把评估实例与词级扰动版本放进选择题，测试模型能否选回原始措辞；作者设计 Bias Compensator Quiz 处理选项位置偏差，并以模型/数据分区报告探针信号。它在不访问训练集或参数的场景下提供行为线索，不定位具体训练对象、来源、阶段或分数变化，也受扰动质量和模型选择偏好约束。
- [The Emperor’s New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination](https://proceedings.mlr.press/v267/sun25t.html)，ICML 2025。官方 PMLR 页面报告 10 个 LLM、5 个 benchmark、20 类缓解策略和两种污染场景，并提出 question-level fidelity 与 contamination resistance 指标。作者没有发现一种方法在所有 benchmark 上兼顾语义保真与抗污染；语义保留策略未在所有基准普遍提升抵抗力，语义改变策略则损害保真。正文据此要求动态/改写题先做构念、难度和评分等值性检查，而不是将新题默认视为可靠。
- [Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)，ICLR 2026。阅读全文。Self-Critique 以 critique prompting 后熵/路径信号检测受控 RL-MIA 成员；RL-MIA 使用 AIME24/AIME25、K&K、SAT 等任务的已知注入样本。Qwen2.5-7B-Instruct 与 DeepSeek-Math-7B-Instruct 的四任务平均 AUC 为 0.70/0.64，作者报告相对最佳非该方法基线各提升 19%，论文摘要报告实验最高提升 30%。属于特定模型、数学/逻辑任务和注入设置中的结果，不能外推为任意模型或自然污染的可靠检测。
- [On The Fragility of Benchmark Contamination Detection in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html)，ICLR 2026。阅读全文和 limitations。研究报告 SFT 污染在 GRPO 后的成员/非成员概率差下降；特定 Qwen2.5-7B-Instruct 实验把 10K DeepMath 用于最多 156 步的 GRPO，若干探针 AUROC 随步骤增加而下降，部分接近随机水平；CoT 污染推理模型的实验也使多数已测方法接近随机。结论仅限作者的模型、任务、污染流程和检测方法；论文未提出可替代的普遍检测算法。

## 内容去向与编辑决定

| 证据或发现 | 位置 | 处置 |
| --- | --- | --- |
| 重复既可能放大训练计算，也可能保留常见模式；高过滤数据可以在特定预算下重复训练 | §4.2、§4.9 | 加入 NeurIPS 2025 受控比较和分母/recipe，明确重复并非普遍收益 |
| 重复频率与样本质量在特定过滤集可能相关；Keep-One 与频次感知保留效果不同 | §4.9 | 补充 DCLM 22-task toy 数值、全局去重与文档重采样的区别，不外推为保留规则 |
| 子文档重复与全局频次可独立检测并配置 copy budget | §4.3、§4.9 | 引入 2026 arXiv 研究作为新方法证据，保留作者条件，不展开整套论文算法 |
| Jaccard 会稀释短片段被长文包含的程度 | §4.3.3 | 新增 containment 定义与 `3/7`、`3/3` 完整手算 |
| MinHash 有抽样方差；LSH 参数决定候选概率但不等于真重复概率 | §4.6 | 新增 `J=3/5`、四次估计 `3/4` 和 `b=4,r=2,s=0.6` 概率推演；补经典出处 |
| 行为检测依赖访问条件且会受后训练影响 | §4.12、§4.20.8 | 对比语料可读检索、DCQ、RL-MIA、自我批评探针与 GRPO 掩盖；坚持“未命中≠无污染” |
| 新鲜题目和原 benchmark 有保真度权衡 | §4.13、§4.20.9 | 加入 ICML 2025 controlled assessment；链接第七册承担的评估等值/统计细节 |
| 扫描候选、人工确认、训练隔离和分数影响必须分层 | §4.12、§4.17–§4.19 | 分开候选率、确认暴露率、弱信号率和任务正确率；样本重叠不单独证明分数因果 |

不新增章节。复杂代码/数学样本契约交第六章；数据 mixture 权重与 exposure 交第五章；dataset manifest、删除和 lineage 交第十一章；完整 benchmark 统计和新题等值不在本章替代第七册。

## 待验证与适用边界

- 新增研究链接均在本轮打开其官方 proceedings / Anthology / arXiv 原文；没有对所有既有参考链接做全章 HTTP 扫描。
- arXiv 子文档去重研究仍按预印本和作者实验理解；其指标不能证明本地数据管线或目标模型收益。
- Self-Critique 与 GRPO 脆弱性结果都来自受控污染构造及有限公开模型/任务，不构成自然污染总体率。
- GitHub 上 Mermaid 实际渲染、本册第一至第四章连续阅读和完整外链扫描尚未做；章节完成门禁保持开放。
