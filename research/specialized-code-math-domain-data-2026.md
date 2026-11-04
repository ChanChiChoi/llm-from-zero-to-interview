# Specialized Code, Math and Domain Data：来源证据

检索日期：2026-11-03。以下结论来自正式会议论文、ACL Anthology 或期刊全文；状态为 source-backed，未在本地复现实验。

## 代码数据

### rStar-Coder

- **来源与版本：** NeurIPS 2025 正式论文，[会议论文页](https://papers.nips.cc/paper_files/paper/2025/hash/54e847e1dffc87a8063844b149148557-Abstract-Conference.html)，[论文 PDF](https://papers.nips.cc/paper_files/paper/2025/file/54e847e1dffc87a8063844b149148557-Paper-Conference.pdf)。
- **阅读范围：** 摘要、引言和数据构造概述。
- **来源支持的事实：** 论文报告构造 418K 道竞赛级代码问题和 580K 条长推理解答；输入/输出测试生成分成三步，并通过互相验证确认输出标签；实验使用 Qwen 1.5B—14B 模型与指定代码 benchmark。
- **证据边界：** 这些数据规模、验证流程和模型结果绑定论文的来源、任务与评测设置。自动生成测试和互相验证仍有覆盖边界，不能证明任意仓库代码、语言或生产测试均可靠。

### The Common Pile v0.1

- **来源与版本：** NeurIPS 2025 Datasets and Benchmarks Track，[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/52acc050138d6f40dad6f12f91a4ce22-Paper-Datasets_and_Benchmarks_Track.pdf)。
- **阅读范围：** 摘要、引言、许可定义及训练模型概述。
- **来源支持的事实：** 论文介绍约 8 TB、30 个来源的公有领域或开放许可语料，覆盖包括代码在内的多个内容域；其“开放许可”按 Open Knowledge Foundation Open Definition 2.1 定义。作者报告训练两个 7B 模型 Comma v0.1-1T 与 Comma v0.1-2T，分别使用 1T 与 2T token，并发布语料构建代码、训练 mixture 和 checkpoint。
- **证据边界：** 这说明论文采用的许可准则可用于构建和研究开放语料，不是对其他项目、司法辖区、用途、再分发或具体文件授权的法律判断。公开可访问、仓库 license 标签与项目获准使用不是同一事实。

### ResearchCodeBench

- **来源与版本：** NeurIPS 2025 Datasets and Benchmarks Track，[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/cd0d0a873cc3e601c76f46dccc3d4c5f-Paper-Datasets_and_Benchmarks_Track.pdf)。
- **阅读范围：** 摘要、引言和 benchmark 构造概述。
- **来源支持的事实：** benchmark 从 20 篇 2024—2025 年机器学习论文构造 212 个实现挑战，并评估 32 个模型；109 题的 HARD 子集上，论文报告最佳模型通过率 33%，43 题没有被所测模型通过。
- **证据边界：** 结果对应指定论文、题目、测试和模型版本。它测量新研究实现任务，不直接证明训练语料质量或一般软件工程能力；不同 benchmark/harness 的 pass rate 不应直接横向拼表。

## 数学数据

### FLAMES

- **来源与版本：** Findings of EMNLP 2025，[ACL Anthology 正式论文 PDF](https://aclanthology.org/2025.findings-emnlp.1346.pdf)。
- **阅读范围：** 摘要、引言和实验设计概述。
- **来源支持的事实：** 论文系统比较十种数学合成策略及流水线因素；摘要报告，在其固定数据生成预算下，保留较广的问题覆盖优于只留下解答可靠的题目。
- **证据边界：** 结论绑定作者的模型、合成策略、数据预算、训练流程和数学评测。问题有效性与解答正确性是不同字段；该发现不意味着错误答案可以作为正向解题监督。

### More Data or Better Data?

- **来源与版本：** EMNLP 2025 Industry Track，[ACL Anthology 正式论文 PDF](https://aclanthology.org/2025.emnlp-industry.43.pdf)。
- **阅读范围：** 摘要、引言和统一评估流水线说明。
- **来源支持的事实：** 作者使用统一流程分析开源数学数据选择和合成，并报告更可解释的数据结构或较强模型蒸馏在其设置中常常比单纯扩大数据量更有用；研究同时讨论预训练数据筛选与 SFT 数据生成。
- **证据边界：** 这是论文摘要对其模型、数据、训练流程和数学任务的作者报告，不能外推为“高质量数据总胜过更多数据”的一般定律。

### MathDebugger

- **来源与版本：** EMNLP 2026 正式论文，[ACL Anthology 页面](https://aclanthology.org/2026.emnlp-main.608/)与[论文 PDF](https://aclanthology.org/2026.emnlp-main.608.pdf)，页 14001—14018。
- **阅读范围：** 正式摘要、引言、基准组成和错误类别概览。
- **来源支持的事实：** 基准含 2,000 道正确问题、2,000 道按四种细粒度错误类型均衡构造的问题，以及 2,000 条标注答案，其中 610 条答案被标为错误并覆盖三类错误；标注同时包含正确性与错误类型。论文评估 14 个模型（6 个闭源、8 个开源，规模 1.5B—72B）及 3 个 process reward model，并报告细粒度错误检测仍未饱和、求解与验证能力存在差距。
- **证据边界：** 这些数字描述基准构成和论文模型评估，不代表项目内部 verifier 的准确率。问题是否合理、答案是否正确、错误类型、解题过程完整度应分开标注和验收。

## 专业领域数据

### 医学模型记忆研究

- **来源与版本：** Anran Li et al., “Memorization in large language models in medicine: prevalence, characteristics, and implications,” *Nature Communications* 17, 7729 (2026-06-19), DOI [10.1038/s41467-026-73779-6](https://doi.org/10.1038/s41467-026-73779-6)，[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13434820/)。这是正式发表的期刊论文，不再是仅有 arXiv 预印本的状态。
- **阅读范围：** 摘要、三类适应设置、临床样本设计、Continued pretraining 与微调比较、记忆分类，以及 Meditron 持续预训练内容在后续微调后的重合分析。
- **来源支持的事实：** 研究考察医学语料继续预训练、公开医学 benchmark 微调和真实临床记录微调；真实临床案例包含耶鲁纽黑文医疗系统 13,000 份住院记录。作者把观察到的记忆分成有益回忆（如临床指南）、信息量低的模板复述，以及可能有害的敏感临床内容。摘要称在特定设置中，继续预训练记忆内容最高有 87% 在微调后仍保留；正文的重合分析针对 Meditron-7B、指定临床指南/MEDLINE 数据与 token 匹配口径，不能解释为普适记忆率。
- **证据边界：** 样本、模型、提示、数据集、训练目标和记忆度量均有限；13,000 条临床记录不是完整医学数据总体。该研究支持区分知识回忆、模板与敏感内容的审计设计，不判定某个项目的数据授权或安全性，也不提供临床建议。

## 跨来源限制

上述论文研究目标、模型规模、数据构造和评分协议各异。它们适合说明应保留问题/答案/测试/来源/权限等结构化证据，不能合并成一个通用训练配方或数据比例。数值只在各来源定义的分母、模型和 benchmark 条件下成立；这里没有复现代码、训练或评估实验。
