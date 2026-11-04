# 第十一章《Dataset Versioning 与 Governance》精修审计

审阅日期：2026-11-03

## 范围与章节判断

- 完整通读本章开篇及 §§0—23，检查对象版本、manifest、lineage、权限、删除、审计、文档、训练复现、schema、指标、Python 教学示例、公式和资料入口；依据当前规范完成章级定向扫描。章节职责、顺序和书册骨架不变。
- 本章负责把来源数据、处理过程、不可变快照、授权与删除证据、训练运行和发布文档连成可追溯链。它不替代第七册的评估统计方法、第八册的模型编辑/遗忘方法，也不提供司法辖区法律意见；§9 增加到第八册第九章的模型层交接链接。
- 五个连续核心教学单元按论证主线统计：开篇—§§0—7（对象身份和快照）**5,893** 字；§§8—10（访问、删除和审计）**1,972** 字；§§11—16（文档、复现、schema、职责和指标）**2,850** 字；§§17—19（治理控制面、实现例和发布判断）**3,205** 字；§§20—21（失败修复和回放）**1,882** 字。均超过 1,800 字下限。计数对象为解释性汉字；排除标题、表格、代码、公式、行内代码/公式、链接和空白；分组依据连续问题链，不按 H2 数量机械切分。

## 章级扫描与候选去向

- **日期、问题与时间窗：** 2026-11-03；检查 2024—2026 的数据集元数据、provenance、lineage 事件和使用条件规范，以及文档框架、工程工具和 GDPR Article 17 法律入口。问题包括：新格式是否能补足现有 schema/manifest 表达；格式、血缘工具和权限/删除执行是否被混为一谈；治理示例能否区分训练候选排除与完整删除；本章是否应扩展到模型层遗忘或法律解释。
- **资料类型与范围：** 阅读 Croissant 1.1 官方规范的 introduction、provenance、外部词汇和 data-use-conditions 部分及 MLCommons 发布说明；W3C PROV Overview；OpenLineage 1.55.0 文档的 About/Object Model；DVC 官方文档入口；Hugging Face Dataset Cards 官方说明；Datasheets for Datasets 与 Model Cards 论文摘要/版本信息；NIST AI RMF 官方页；EUR-Lex Article 17 文本索引。资料事实、版本和证据边界见 [`dataset-versioning-governance-evidence-2026-11-03.md`](../../research/dataset-versioning-governance-evidence-2026-11-03.md)。未作穷尽搜索。

| 候选方向 | 证据与评估 | 去向 |
| --- | --- | --- |
| Croissant 1.1 机器可读数据集描述 | 可表达数据集、文件、记录集和字段层 metadata；可借 PROV-O 表示来源、生成活动和责任主体，并表示使用条件。规范表示不验证真实授权或执行访问策略。 | 纳入 §5 和 §22，作为 schema/manifest 的可选互操作格式示例；明确“能表示”不等于“已证明/已强制”。不扩成 Croissant 实作教程。 |
| W3C PROV 与 OpenLineage | PROV 提供 entity/activity/agent 的 provenance 语义；OpenLineage 区分 Dataset、Job、Run，以及运行时和设计时事件。它们关注语义/事件，不等于不可变快照或内容校验。 | 保留现有概念与工具入口，§22 明确两者和快照/权限的边界；不把两套模型并列成互斥替代品。 |
| Datasheets、Model Cards 与平台 Dataset Cards | 两篇论文提供文档提问框架；Hugging Face 文档说明其 README/YAML 卡片呈现。均不能独立证明当前数据事实、授权或模型用途适配。 | 保留本章数据文档—模型文档主线，注明“文档完整”不等于事实已核验或法律合规。 |
| 数据删除与模型参数影响 | Article 17 有适用条件及例外；条文不规定适用于所有模型请求的单一遗忘技术。模型参数侧问题属于第八册第九章职责。 | §9 增加跨册入口；本章只记录请求定位、未来候选隔离、派生对象及历史运行影响，不作法律结论或扩写模型遗忘算法。 |
| 工具文档与风险框架 | DVC、OpenLineage、Hugging Face 和 NIST 各自提供工具/平台/风险管理入口，范围不同。 | 保留为实现与文档示例；明确这些入口不自动证明授权、内容正确、策略执行或合规。 |

## 正文修改、篇幅与例子

- 将 §5.1 的版本、manifest、lineage、覆盖率、删除时延、访问指标、文档完整度和治理状态向量公式统一为带大括号的数学下标；修正 `N_{\mathrm{token}}` 漏写反斜线及 §13 的 `V_{\mathrm{...}}`、`E_{\mathrm{...}}` 公式写法。补齐变量和分母定义，区分适用时延与未闭合状态。
- 修正访问阻断率分母：分母为策略判定应拒绝的未授权请求，而不是所有高风险访问；并列给出授权成功率和审计留痕率。加入一组明确是假设输入的逐步计算：lineage `960/1,000=0.96`，license 总体 `920/1,000=0.92`，高风险桶 `40/50=0.80`，未授权阻断 `7/8=0.875`，授权请求成功 `96/100=0.96`，决策留痕 `108/108=1.00`，文档完成 `5/6≈0.833`。同一例中删除请求在 20 分钟后隔离但未闭合，因此完成时延未定义。
- §7 的 token 汇总手算为 `1,130+820+1,040=2,990`；解释计数器与 manifest 不一致如何暴露漏读、重复读取或口径差异。
- 修正 §18.1 demo 的状态语义：把 `deleted` 改为 `excluded_from_training_candidates`，将“completed”改为 `excluded_from_training_candidates_only`；`deletion_targets_excluded_from_candidates` 不再冒充完整删除。当前候选排除后仍触发派生对象/历史运行核查；未命中请求保持 `unresolved_target`。
- demo 原先把 12 位字段称作 checksum，并用非空检查冒充校验。现将它标为 `toy_fingerprint`，说明该值只哈希少数 toy 字段且截断 SHA-256；`manifest_toy_fingerprints_present` 仅检查字段存在，不验证内容。示例输出用指纹占位符，避免把无法追溯运行的旧精确字符串当成已验证结果；manifest 明确是治理 ledger 清单，训练候选集合另行呈现。
- 为数据删除新增一幅 Mermaid 图，分开显示目标未解析、未来训练候选排除、派生 shard/index/cache 回放、历史训练运行/发布物影响和责任人状态复核。已有 §21 线性回放图保留，分别表达对象传播和生命周期顺序。
- 增加治理样例手算：demo lineage `4/6≈0.667`、datasheet `5/6≈0.833`；解释通过单项阈值不等于版本整体可以发布。§16 另补总体与高风险桶 license 覆盖率的分层演算。§20—21 增加请求命中 3 个样本、2 个 shard、1 个索引、1 个缓存及 1 个旧运行的合成回放，说明部分命中不能缩小原请求范围。
- `example_status`：算术推演均为正文手算或合成输入，不是生产测量；不将 Article 17、Croissant 或平台文档解释为合规或质量保证。

## 公式、代码、图示与链接

- **formula_status：** 11 个 math 围栏、47 个行内公式；静态扫描花括号成对，无 `\operatorname` 或未加大括号的数学下标。关键例子含逐步分子/分母、分层统计和时间状态解释。GitHub MathJax 页面未渲染验收。
- **code_status：** §18.1 一个 Python 标准库 demo。可执行语句本轮有变化，标为“静态审阅，未运行（2026-11-03；Python 版本未确认）”；未运行代码、测试、模型、benchmark 或 PDF。固定输入的候选 ID、lineage 分母、权限拒绝数、license 计数、文档完成度和删除状态经手算核对；toy fingerprint 未计算，输出用占位符。需以后在允许代码动态验证的门禁中复验。
- **visual_status：** 新增 1 幅 Mermaid `flowchart LR`，使用引号节点标签和标准管道边标签，代码围栏配对；另有既有 text 生命周期图。源码静态检查通过，GitHub 页面渲染待最终门禁。
- **link_status：** 正文含 11 个 Markdown 链接，其中 1 个相对本地链接指向第八册第九章，静态确认目标存在；其余为 10 个外链。新增 Croissant 1.1、Dataset Cards、OpenLineage、W3C PROV、NIST、Datasheets、Model Cards 与 EUR-Lex 等官方/原始来源已核对。DVC 文档入口当前重定向；历史外链未全量逐条巡检。

## 配套同步与未关闭门禁

- `PAPERS.md` 新增第九册第十一章入口，列出研究框架、Croissant 1.1、provenance 规范、工具文档、NIST 和 Article 17，并链接本审计和事实研究记录。
- 新增 `research/dataset-versioning-governance-evidence-2026-11-03.md`，仅记录外部来源事实、版本、阅读范围及证据边界；本审计和候选去向只记于 `progress/`。
- 更新第九册架构审计，标记第 1—11 章静态精修完成，下一章为第 12 章；本章职责和目录不变，因此不更新简介、目录、题库、术语、练习、项目或知识图谱。`plan.md` 未修改。
- 未关闭：代码未运行；GitHub Mermaid/MathJax 渲染未验收；历史外链未全量巡检；第九册连续复读及最终门禁未完成。PDF 不更新。
