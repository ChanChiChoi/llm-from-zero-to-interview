# 预训练数据选择与多样性：来源证据

> 本记录只整理联网核验到的研究事实、版本和适用边界；不同论文采用的模型、数据池、预算与评测不构成统一排行榜。

## DataComp-LM：过滤、数据混合与去重

- **正式版本：** [DataComp-LM, NeurIPS 2024 Datasets and Benchmarks Track](https://papers.neurips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf)。重点阅读正文质量过滤比较与混合实验，以及 Appendix L 的 Table 18–19（PDF 第 57 页）。
- **分类器样本：** Appendix H 的 fastText 正例/负例消融固定为 400K 样本，200K 正例和 200K 负例；负例从较早、较小版本的 RefinedWeb reproduction 随机采样。正文 Table 4 的 fastText 过滤器比较在 1B-1x 规模进行；Table 5 的参考集选择消融在 7B-1x 规模进行，OH-2.5 与高分 ELI5 正例组合在该表 10% 阈值条件下取得最高 Core。两个实验的规模、问题与比较对象不同。
- **混合：** Table 6 比较从 Common Crawl 基础配方混入 Wikipedia、Books、StackExchange、arXiv 和 GitHub 等来源；弱一些的 C4、RefinedWeb 等网页基线可获益，而强过滤 DCLM-baseline 上按该配方补源并未带来同样结果。证据只支持论文的固定模型、比例、数据版本和评测设置。
- **近重复：** Table 19 的 7B-2x 行报告 BFF（min/max n-gram 13、10 shards）MMLU 44.3、Core 45.3、token yield 3T；MinHash+SuffixArray 行为 44.4、45.5、3.2T。指标接近不构成所有语料下的算法排名。
- **版本差异：** 正式 NeurIPS PDF 的 Table 19 将 BFF yield 写为 3T；当前可访问的 [arXiv HTML](https://arxiv.org/html/2406.11794) 对同一行显示 3.8T。两处来源不一致。报告正式发表结果时以 NeurIPS PDF 为准；如需要描述 arXiv 当前 HTML，必须明示它与会议版本不同，不能把两个数字混写。
- **评估基线：** [官方 DCLM 仓库](https://github.com/mlfoundations/dclm)说明保留 `Core_v1` / `Extended_v1` 与修订后的 `Core_v2` / `Extended_v2` 结果。论文表格中的聚合分数应标注论文版本，不能与仓库修订结果混算。

## FineWeb-Edu：教育内容筛选

- **来源与版本：** [FineWeb 论文 v2](https://arxiv.org/abs/2406.17557v2)，重点阅读 §4.2。
- 论文报告使用 Llama-3-70B-Instruct 对 FineWeb `CC-MAIN-2024-10` 快照中随机抽取的 460,000 个网页打教育价值分；410,000 条用于拟合，50,000 条作验证。作者在冻结的 Snowflake-arctic-embed-m 表征上训练线性回归器，并构造 1.3T token 的 FineWeb-Edu。
- 这是面向教育内容及知识/推理类评估的筛选目标。数字和效果是论文版本中的作者报告，不应与后来数据卡的更新快照、标注数或 token 统计拼接，也不证明其对代码、多语言或对话目标普遍更优。

## Dolma：PII 过滤的阈值例子

- **来源与版本：** [Dolma v2](https://arxiv.org/abs/2402.00159v2)，重点阅读正文 PII 处理描述与 Appendix I。
- 对 Common Crawl、C4、Reddit 和 GitHub 子集采用正则定位 email、phone、IP。每页检测到不超过 5 段时将命中替换成类型 token；达到 6 段及以上时移除整篇文档。作者报告在 25 个 Common Crawl 快照中，这两类处理分别影响约 0.02% 和 0.001% 文档。
- 这些阈值和比例是特定 corpus / filter 的实现与观测；作者在低占比消融中未见模型性能差异，不能推成其他语料、检测器或风险标准下删除与替换等价。

## 质量筛选目标依赖性

- **来源与版本：** [Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining v4](https://arxiv.org/abs/2510.00866v4)，重点阅读 §3–7。
- 论文用不同参考“高质量”集合为 RedPajama 数据打分并训练 350M 模型，评估包括 ARC、MMLU 和 RewardBench。以 ARC-Easy 作为参考集时，质量筛选能同时改善该任务成绩和参考集语言建模损失；对多数其他参考集，下游指标的改善与参考集损失不总是同步，后者可呈 U 形。
- 结果支持“筛选器分数受参考集和下游目标影响”，不支持一个跨任务通用的黄金质量标签；实验结论受论文模型、数据、训练预算和 benchmark 限制。

## 表示空间多样性选择

- **来源与版本：** [Towards Scalable Data Diversification for Language Model Pretraining via Leverage Score Sampling v1](https://arxiv.org/abs/2609.32484v1)，2026-09 预印本；重点阅读 §3–4 与附录中的数据选择和代码评测结果。
- 该方法根据嵌入表示计算 leverage score，以提升所选样本对表示空间的覆盖。它优化的是表示多样性代理，不直接测量事实正确性或授权状态。
- 表 2 的 web-only、40% 选择率实验中，DCLM-fastText（Qual）的 Base Mean 为 50.21（越高越好），Code Mean 为 266.77 `BPB × 10^-2`（越低越好）；Leverage Score Sampling 分别为 48.91 与 256.43。两个指标呈现通用基准均值与代码建模之间的取舍，不能混成一个无条件排名。
- 该预印本的实验规模、Common Crawl 构造、40% 筛选率、模型和任务均限定结论范围；尚无独立复现证据支持跨模型、语种和目标的稳定排序。

## 证据综合与未验证边界

不同数据选择研究以参考集相似度、目标任务分数、语言建模损失、去重后 token yield 或嵌入空间几何覆盖为优化目标。它们测量的是不同性质。合理的工程比较需要固定总训练预算与 tokenizer，按目标语言和领域分层报告保留率，并用未参与筛选的数据、下游任务切片和重复训练评估训练收益与回归。

上述论文报告不等于对目标项目数据的本地复现；分类器与嵌入式打分都不能替代许可、PII、secret 和 benchmark 污染的独立硬门禁。新预印本的排序结果仍需跨语料、模型规模、随机种子和独立实验检验。
