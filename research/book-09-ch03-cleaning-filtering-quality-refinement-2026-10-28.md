# 第九册第三章：清洗、过滤与质量评分研究与精修记录

## scope

日期：2026-10-28。按当前 `WRITING_SPEC.md` 从头复读《清洗、过滤与质量评分》，围绕“过滤分数代表什么、阈值怎样权衡误删/漏删、怎样验证过滤后的训练价值”开展章级扫描、正文修订、手算、教学代码核验和配套同步。扫描重点是 classifier-based quality filtering 的适用边界、不同语言/领域的分数校准，以及下游验证能支持多强的结论。研究扫描用于发现与核对候选，不声称穷尽数据选择文献。

## chapter_contract / architecture_decisions

保留单篇教材，不新开讲次。主线收束为：样本对象与风险信号分离 → 便宜规则/统计特征/质量分类器/PPL 各自能提供的证据 → 分桶人工抽样 → 阈值代价与误差统计 → 小模型或目标训练验证 → 版本回放。

跨章分工：第二章负责来源、采集与解析对象；第三章负责有效性/质量/切片误差与风险动作接口；第四章主讲 exact/near dedup 和评估污染；第五章主讲训练 mixture 与采样预算；第六章主讲代码/数学/领域数据验证；第七章主讲 synthetic/distillation 生成；第八章主讲 preference/safety 数据标签；第十一章主讲 manifest、lineage、权限、删除和治理版本。目录已有“过滤规则、校准集、留存/拒绝账本”产物，本章无需同步改目录。

原章的大量“清洗对象—按样本类型—安全风险—污染/重复—机器生成—版本治理”内容彼此扩散。已删除重复讲解或改为短交接，避免第三章复讲第四、第五、第六、第七、第十一章。保留 PII/secret/safety 的本章接口，因为这些风险会形成与质量分数独立的删除、脱敏或隔离决定。章节核心论点明确为：质量分是带标签协议和目标域的排序信号，不能直接升级为“事实真值”或普遍准入许可。

## research_scan

检索窗口截至 2026-10-28，聚焦 2024–2026 的原始数据选择/质量过滤研究；打开论文全文或正式出版页面，核验方法、数据、模型、阈值和评估范围。

1. **分类器分数是参考集条件下的选择信号，不是通用质量尺度。** Nait Saada et al.，《Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining》，arXiv v4，2026-06-24 最新版本，页面注明 ICML 2026。作者以 32T-token RedPajama-V2 为背景集，研究不同高质量参考集、top-k 保留比例及下游任务；训练规模包含 125M、350M 和 1.3B，训练 horizon 最高到论文所列的 120B token 配置。作者报告 CQF 可提升特定下游任务，却不必改善整个“高质量”集上的语言建模；参考集会带来任务偏向，OpenOrca 例子还观察到长度混杂。其 2026 工作提出 optimization/data-conditioning 视角并以小模型 proxy 估计，但这些关系仍受该论文训练配方和任务边界约束。正文据此补充“分类器更像参考集”解释、混杂对照、目标切片复核，以及小模型只作筛选证据的限制。来源：[arXiv v4 摘要及历史](https://arxiv.org/abs/2510.00866v4)、[HTML 全文](https://arxiv.org/html/2510.00866v4)。

2. **多语言标签器既要看排序，也要看各自分数分布。** Ali et al.,《Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models》，EMNLP 2025。JQL 先构造 511 条人工标注文档并翻译到 35 种语言，用 LLM 生成标签后蒸馏到冻结多语言嵌入上的轻量回归头；正文报告约 14M 文档标签。端到端质量对照使用 2B 参数模型，27B/14B token 配置，主要训练对照为 13 种欧洲语言；35 语种的标签/注释覆盖不等于 35 语种都有相同规模的独立预训练实验。论文指出不同 judge 的绝对分布有偏移，但排名相关性可较高，因此采用按模型分布校准的分位阈值。论文在 0.6/0.7 分位设置报告对 FineWeb2 heuristic baseline 的相对 token 和 benchmark 变化；这些是作者的 13 语种 aggregate 与指定模型结果。正文只纳入“分位阈值可作为校准工具，必须在本地分布与下游预算中重验”，不把阈值或收益写成通用配方。来源：[ACL Anthology 正文](https://aclanthology.org/2025.emnlp-main.449/)、[固定 arXiv v2](https://arxiv.org/abs/2505.22232v2)。

3. **多维评分可以由目标实验挑组合，仍不产生普适质量标签。** Zhuang et al.,《Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models》，ACL 2025。工作提出 professionalism/readability/reasoning/cleanliness 四维标签，并在 627B-token SlimPajama 上组织 25 项信号，通过 proxy model 预测验证 loss、再搜索评分组合。作者报告主要 1.3B/30B-token 对照及 3.3B/100B-token 验证，并声称特定配置下收敛速度和下游表现改进。论文表格还显示 rating 计算 FLOPs 在 1.3B 主配置上显著，标注复用是成本论点之一。正文用一段说明“优化目标可以是下游验证损失，而不是发明普适质量词典”，不讲完整算法，以免与第五章的数据 mixture 和第十章估值实验重叠。来源：[ACL Anthology 正文](https://aclanthology.org/2025.acl-long.533/)、[arXiv 页面](https://arxiv.org/abs/2504.14194)。

**新增主题去向**：没有达到独立章节价值，合并进 §3.8 质量分类器、§3.13 阈值校准与 §3.14 下游验证。完整 dedup、contamination、数据配比和 teacher 生成分别回到第四、第五、第七章。没有把新论文的 reported score 或阈值当作可直接部署政策。

## changes

- 重写开篇契约和 §3.1–§3.4 主线，删除把 duplicate/contamination 作为综合质量分分量的表达，新增分流 Mermaid；质量分与隐私/秘密/安全门禁分离。
- §3.8 扩展“参考正例定义分数语义”“分类/排序/校准区别”，加入 2026 CQF 分析、EMNLP 2025 JQL 和 ACL 2025 Meta-rater 的证据边界。
- §3.10 把污染改成第四章交接；§3.11 将 synthetic/distillation 处理改成交接；§3.12 将重复的五类专题细节收为风险/证据/交接表。
- 修正 false positive/false negative 口径：以合格样本为正类、预测保留为正预测时，误删是 FN、漏删是 FP。新增显式代价函数和 8 例逐步手算，分别核对 precision、recall、false-delete rate、false-keep rate、document retention 和加权代价。
- 将原涵盖 PII、污染、去重、质量、语言和动作的综合大 demo 替换为单一的质量阈值审计示例，代码使用与手算一致的 8 条合成标签；不以文档留存代替 token 留存。
- 将低资源语言案例标注为合成教学场景，明确因果结论需要配对的小模型任务切片支持。重写 §3.17，仅保留质量过滤自身的版本与回放字段，并把完整 manifest/删除传播交给第十一章。更新失败模式、论文索引、练习第 7–8 题及面试题第 6、7、10 题。

## length_status

本文件按一个核心正文单元验收；§3.16 的代码与解释与前文同一阈值例题，不单列成独立辅助讲解单元。排除标题、代码/公式/图示围栏、表格单元格、引用与练习后，解释性中文正文机械计数为 **8,693 个汉字**，高于 1,800 字下限；计数包含 §3.21 结语的解释正文。统计只是下限核验，不能替代人工论证复读。

## formula_example_status

合格样本为正类，质量分高于阈值则预测保留。8 例中 `tau=0.60`：TP/FP/FN/TN = 3/2/1/2，precision `3/5=0.600`，recall `3/4=0.750`，误删率 `1/4=0.250`，漏删率 `2/4=0.500`，文档留存 `5/8=0.625`；在教学代价 `C_del=5, C_keep=2` 下，损失 `5×1+2×2=9`。`tau=0.75`：2/1/2/3，precision `2/3≈0.667`，recall `2/4=0.500`，误删率 `2/4=0.500`，漏删率 `1/4=0.250`，文档留存 `3/8=0.375`，损失 `5×2+2×1=12`。这说明更严格阈值提高 precision、降低 recall；哪个更合适取决于任务代价。样本、分数和代价都是合成教学输入，不是实际语料经验。

## code_status

§3.16 Python 代码块已在 Python 3.12.9 标准库环境运行。Markdown 中唯一 Python 围栏抽取到 `/tmp/book09-ch03-threshold-audit.py` 后执行；8 条固定标签输出与手算一致，空数据返回全部未定义比率/代价，无正类切片的 recall 返回未定义。代码只验算阈值计数，没有使用真实数据、训练模型、扫描敏感信息或评估许可。代码状态及边界在正文说明；正文代码区、状态行及紧邻说明均不写维护文件名或内部工作流。

## visual_map

§3.4 新增标准 Mermaid `flowchart TD` 表示样本分桶、质量排序、独立风险支路、人工审计、阈值/小模型验证和可回放清单。节点标签加引号；带文字虚线使用 `-.->|记录原因和版本|` 管道格式；源码静态语法和围栏配对检查通过。GitHub 页面未展示本地更改，目标渲染仍开放。

## link_status

新增三条原始论文链接（arXiv v4、ACL Anthology EMNLP 2025、ACL Anthology ACL 2025），均已打开核对。新增到第四、六、七、十一章的相对链接目标已逐一检查存在。Presidio 与既有研究链接仍未做全章 HTTP 扫描。

## sync_status

`PAPERS.md` 新增第三章三个研究入口及研究记录链接；`EXERCISES.md` 的数据工程第 7、8 题改为手算阈值、分层估计与空分母要求；`INTERVIEW_BANK.md` 第 6、7、10 题更新质量信号语义、FP/FN 口径与下游验证边界。章节编号和目录职责未改变，因此不需修改简介、目录或 glossary。

## validation / open_risks

固定 demo 在 Python 3.12.9 下执行并与手算一致。篇幅统计为 8,693 个汉字；整章重读、图/公式代码围栏配对、五处本地链接、无工作流残留检查和 `git diff --check` 均通过。Mermaid 目标页渲染和全部既有外链 HTTP 扫描未做。没有目标模型、人工标注集、生产语料或训练预算；论文结论只绑定作者的样本、模型、token 数和 benchmark，不能作为目标过滤器已校准的证据。本章门禁保持开放。

## status

第三章已完成实质改稿、阈值手算、合成教学代码执行、研究记录和配套索引同步；下一步进入第九册第四章。第九册全书连续复读及章节完整门禁仍开放。
