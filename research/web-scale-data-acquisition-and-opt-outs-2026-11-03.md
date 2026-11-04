# Web-scale 采集、语言覆盖与 robots opt-out：来源证据

记录日期：2026-11-03

本记录整理 Common Crawl 当前快照、网页归档格式、多语言语料研究和 robots opt-out 实验中的可核验事实与证据边界。它不提供法律意见；数据使用权仍须按真实来源、许可、合同、用途和适用法域单独判断。

## Common Crawl 快照与归档格式

- **官方快照列表：** [Common Crawl collection index](https://index.commoncrawl.org/collinfo.json)，2026-11-03 读取。列表最新项为 `CC-MAIN-2026-43`，名称为 October 2026 Index，记录的采集时间为 2026-10-05 10:56:23 至 2026-10-18 05:34:47。该 JSON 是滚动更新的当前列表，引用快照时应同时记录查询日期和具体 ID。
- **官方获取说明：** [Common Crawl Get Started](https://commoncrawl.org/get-started)，读取访问方式与格式说明。官方说明可通过 AWS S3 或 HTTP(S) 读取数据；WARC 保存抓取记录，WAT 提供从记录中提取的元数据，WET 提供提取文本。具体字段应按快照和记录文档核对，WET 不保留原始载荷全部布局信息。
- **证据边界：** collection index 只说明可用索引及其采集时间，不证明某个来源获准用于训练；Common Crawl 的公开访问也不消除许可、隐私、秘密、重复、污染和删除责任。

## FineWeb2 多语言数据构造

- **正式来源：** [FineWeb2: One Pipeline to Scale Them All — COLM 2025 正式论文 PDF](https://openreview.net/pdf?id=jnRBe6zatP)；早期版本为 [arXiv:2506.20920](https://arxiv.org/abs/2506.20920)。实际阅读正式论文的摘要、§§3、5、结论和 limitations。
- **来源支持：** 作者将其自适应多语言处理 pipeline 应用于 96 个 Common Crawl 快照（2013 年夏季至 2024 年 4 月），形成约 20 TB 文本的 FineWeb2，覆盖 1,868 个 language-script pairs；其中 1,226 个组合多于 100 份文档、474 个多于 1,000 份、203 个至少 10,000 份。用于 pipeline 设计与消融的 canary languages 为 9 种；与其他多语言语料比较覆盖 14 种语言，作者报告 FineWeb2 在其中 11 种语言的模型评估中优于所比较的既有多语言语料。
- **证据等级：** source-backed（COLM 2025 正式论文报告的语料构造、实验和作者结论）。
- **适用边界：** 1,868 是语言—文字系统组合数量，不代表每项都有充足样本或同等评估。论文只在语料的一小部分语言上运行 benchmark，消融训练较短；不少低资源组合的内容几乎或全部来自 Bible/Wikipedia。结果不保证所有语种、任务、下游模型或文化维度均有同等质量和能力收益，也不是对 Common Crawl 全量处理结果的独立复现。

## Robots opt-out 与模型表现

- **正式来源：** Fan et al., [“Can Performant LLMs Be Ethical? Quantifying the Impact of Web Crawling Opt-Outs” — COLM 2025 正式论文 PDF](https://openreview.net/pdf/c04ef11771a5e838ccf85b71b78636ed913f5ae0.pdf)；对应 [arXiv:2504.06219](https://arxiv.org/abs/2504.06219)。实际阅读正式论文摘要、§§3、5.1–5.3 及相应表格。
- **来源支持：** 论文把遵守网页 crawling opt-out 后的训练与未遵守的训练之间的下游差异定义为 data compliance gap。作者基于 FineWeb-Edu 和截至 2025 年 1 月的 robots 状态构造过滤数据，以 Llama 架构、1.5B 参数模型进行实验；从头训练 100B token 时，作者报告一般知识评测平均差异不显著，但 Pinocchio 中结构化事实和对抗性事实分类出现差异。论文另测试继续预训练：在累计 1.7T token 的长程设置下，向合规语料加入特定非合规医学数据后，PubMedQA 从 61.4 升至 63.0，Pinocchio 医学领域任务从 44.5 升至 46.9。
- **证据等级：** source-backed（COLM 2025 正式论文在指定语料、模型、过滤快照、训练预算和评测下的实验结果）。
- **适用边界：** 一般知识结果不显著不等于所有能力、模型和来源都无损；特定医学数据增益也不构成跨任务的普遍结论。robots filter 在该研究中是实验性数据筛选条件，论文测量的性能差异不能决定访问授权、版权状态、合同效力或组织合规义务。

## Robots Exclusion Protocol

- **规范来源：** IETF [RFC 9309](https://www.rfc-editor.org/rfc/rfc9309.html)，2022-09。实际阅读 §1、§§2.1–2.2。
- **来源支持：** RFC 标准化 crawlers 应遵守的 robots 规则、分组和路径匹配语法，并明确指出这些规则“不是访问授权”形式。
- **证据边界：** RFC 定义协议行为，不判断特定法域的版权、合同、隐私或训练使用权；robots 状态应与实际访问控制、内容许可、用途审查分别记录。
