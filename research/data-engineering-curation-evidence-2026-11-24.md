# 数据工程与语料筛选主张核验

## 范围与检索

- 研究问题：核对第5册第2章中网页语料质量筛选、近重复、PII、质量分类器和多样性采样的数字、版本与实验边界。
- 日期：2026-11-24。
- 资料入口：arXiv 论文 HTML / 版本页、NeurIPS 正式论文 PDF、方法作者所在团队的公开工程文章和工具文档、DCLM 官方仓库。
- 检索词：`DataComp-LM Table 19 token yield BFF 3T 3.8T`、`FineWeb v2 MinHash 112 14 8`、`Dolma PII spans 0.02% 0.001%`、`Removing Noise not Finding Gold v4 quality filtering RedPajama`、`Leverage Score Sampling 40% Code Mean Base Mean`、`Dolma data curation pipeline practice`。
- 本记录关注正文引用的具体主张；论文报告的训练和评估结果均视为作者在所述设置下的结果，不视为本地复现或通用规律。

## 核验结果

### DataComp-LM：正式版与 arXiv HTML 的表格值不同

- [DataComp-LM: In search of the next generation of training sets for language models，NeurIPS 2024 正式论文，表19](https://papers.neurips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf)：7B-2x 表中 BFF、13-gram、10 shards 的 MMLU / CORE 为 44.3 / 45.3，Token Yield 为 3T；MinHash+SA 一行为 44.4 / 45.5 和 3.2T。
- [arXiv HTML 当前版本](https://arxiv.org/html/2406.11794)，表19：同一 BFF 行的 Token Yield 显示 3.8T；可见 HTML 与正式会议版存在版本差异。
- [DCLM 官方仓库 README](https://github.com/mlfoundations/dclm/blob/main/README.md)：说明 2025-09-05 修正 CORE / EXTENDED 基线计算，并保留 v1、v2 两组聚合分数。引用论文结果时应使用论文版指标并标明版本，不与当前仓库的分数混合。
- 边界：表格数字只描述该论文给定数据池、7B-2x 训练和评估配方；两种去重结果接近不构成跨语料算法优劣结论。

### FineWeb 与 Dolma 的数据处理参数

- [The FineWeb Datasets v2](https://arxiv.org/html/2406.17557v2) §3.4、附录 E：每文档 5-gram MinHash 使用 112 个哈希，分成 14 个 band、每 band 8 个哈希；候选命中概率在 Jaccard 0.70 / 0.75 / 0.80 / 0.85 时约为 56% / 77% / 92% / 98.8%。这说明 75% 是概率曲线附近的目标值，不是硬性的相似度切分。
- 同文 §4：从 CC-MAIN-2024-10 随机抽取 460,000 个网页，由 Llama-3-70B-Instruct 生成教育分数；410,000 个训练样本、50,000 个留出验证样本用于在冻结的 Snowflake-arctic-embed-m 表征上拟合线性回归器。
- [Dolma v2 论文](https://arxiv.org/html/2402.00159v2) §5.3、附录 I：25 个 Common Crawl 快照中，检测到 5 段或更少 PII 的文档替换为特殊标记，约影响 0.02% 文档；6 段或更多则删除整篇，约影响 0.001% 文档。作者未在其小比例影响数据的实验中观察到可测性能差异；该结果不能外推到所有检测器、阈值、语种或模型。

### 质量分类器的任务依赖与覆盖风险

- [Removing Noise, not Finding Gold v4](https://arxiv.org/html/2510.00866v4)，版本日期 2026-06-24：主要设置以 RedPajama-V2 为低质量候选集，用多个高质量参考集训练分类器；文中包含 350M 模型、ARC-Easy、MMLU、RewardBench 等评估。作者观察到参考集改变时，下游表现与参考集语言建模损失不一定同步；多数参考集下，筛选率较高会使参考集损失呈 U 形，ARC-Easy 参考集有不同趋势。
- 同文将结论限制为其数据、模型与任务设置，并分析分类器按似然比筛选，既受“接近参考集”影响，也受“相对候选集更少出现”影响。此证据支持对参考集偏好和覆盖损失保持审计，不支持“筛选分数等于通用质量”。

### Leverage Score Sampling：40% 保留量下的 web-only 对照

- [Towards Scalable Data Diversification for Language Model Pretraining via Leverage Score Sampling v1](https://arxiv.org/html/2609.32484v1)，表2(a)：仅网页数据、保留 40% 时，DCLM-fastText 的 Base Mean / Code Mean 为 50.21 / 266.77，Leverage Score Sampling 为 48.91 / 256.43。Base Mean 是七项常识任务的准确率均值；Code Mean 是七项代码任务的 BPB 均值，表内用 $10^{-2}$ BPB 标度。
- 相同表格另列 70% 网页 + 30% 专门代码数据的混合配方，分数另有变化；不能把 web-only 与加代码配方合并成一行或同一结论。该文为 2026-09 预印本，数字是作者设置中的实验结果，尚不代表独立复现。

### DoReMi：目标式与逐 token 权重更新的差别

- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining，NeurIPS 2023 正式论文](https://papers.neurips.cc/paper_files/paper/2023/file/dcba6be91359358c2355cd920da3fcbd-Paper-Conference.pdf) §2 将目标写成各域 proxy loss 减去 reference loss 的加权和，内层最大化选择 excess loss 最大的域；该目标源式不是 proxy 与 reference 两个负损失相加。
- 同论文 Algorithm 1 和算法细节段落指出，更新域权重时先按 token 求 proxy/reference 损失差，并因 Group DRO 优化器要求非负损失而将逐 token 差值截断为非负，再做域级汇总。目标式与实现中的逐 token clipping 应分别说明，不能将后者误写成论文目标式本身。
- 边界：第2章公式保留目标层面的简写；补充说明其算法更新细节，不声称复现了论文训练过程。

### 实践资料

- [FineWeb 作者团队公开的工程说明](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)：说明重复的定义可以落在文档、段落或行等不同粒度，MinHash 的哈希数与 band 数在边界召回和存储/计算成本之间权衡；其解释与论文附录参数一致。
- [Ai2 Dolma 数据集工程说明](https://allenai.org/blog/dolma-3-trillion-tokens-open-llm-corpus-9a0ff4b8da64)：区分来源专用处理和跨来源通用处理；Common Crawl 的 URL 去重与段落去重分阶段执行，PII 风险缓解结合分类器与规则。材料是团队对自家流程的回顾，支持该管线的实践选择，不证明它适用于所有来源。
- [Ai2 Dolma Toolkit 官方文档](https://docs.allenai.org/training_data/dolma/index.html)：现行 Toolkit 文档包含 tagging、deduplication、mixing 等数据流水线能力，并明确区分 Dolma 原始版本与 Dolma 3。正文讨论的是 Dolma 论文案例，不能把现行文档版本能力不加标记地回填为旧论文实现。

## 可用于正文的结论边界

- 表格数值必须带上来源版本、语料、筛选比例、任务、指标方向和单位。
- 质量筛选与覆盖/多样性选择在不同任务指标上可能呈现取舍；不要把单个平均分压成总体数据质量排名。
- 去重阈值和检测比例是具体语料、处理顺序及阈值的结果；不是天然常数。
- 本轮没有执行正文 toy pipeline、任何数据处理代码或模型训练实验。
