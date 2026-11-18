# 第二十册第十八章精修审计

日期：2026-11-18  
章节：[第十八章《Context Folding》](../../book-20-agent-harness-runtime/chapters/18-context-folding上下文折叠.md)  
状态：章级研究、正文重写、目录与配套同步完成；第十九至二十四章及全套书精修 goal 继续 active。

## 章节契约与边界

本章承接第七章“每轮模型输入怎样选择”，深化第十七章“任务 checkpoint 和外部副作用怎样持久化”，讲长运行轨迹增长后怎样形成更短、仍能恢复的模型输入。读者进入时应能区分模型输入、会话 transcript、运行轨迹、任务 checkpoint、workspace 与外部事实；读完后应能分类待折叠材料、为不同类型选择保留/聚合/摘要/外置/丢弃方式、核验来源和约束，并在恢复时维持副作用的真实状态。

Context folding 在本章中是通用工程问题的教学名称，不是 Provider 中立的 API/schema。第七章继续负责上下文来源、每轮选择、预算、prompt injection 与长期 memory；第十七章完整负责 checkpoint、effect ledger、unknown、幂等和 fencing；第二十一章负责 workspace、Artifact 与外部系统的状态所有权。本章只解释它们怎样限制摘要和恢复，不重复主讲。

## 章级外部扫描与候选去向

检索与阅读日期：2026-11-18。问题范围：长运行 Agent 轨迹达到输入预算附近时，怎样压缩/裁剪上下文，何种任务信息必须保留，折叠怎样失败与恢复，以及怎样评估摘要影响。检索词包括 `agent context compaction evaluation long-horizon`, `context folding long-horizon coding agents`, `constraint erosion agent compaction`, `proactive compaction agent policy`, `context management Search Agent BrowseComp`, `Anthropic compaction API`, `OpenAI Agents SDK session input callback`，并查课程讲义的章节组织样本。完整来源、版本、事实和证据边界见[Context Folding 与长任务压缩来源记录](../../research/context-folding-and-compaction-strategy-2026-11-18.md)。

| 外部主题 | 证据支持与读者结果 | 正文去向 |
|---|---|---|
| 压缩对象：上下文输入、完整 transcript、checkpoint、Trace、workspace 和 provider state | Anthropic context-engineering 文章以及 OpenAI Agents SDK 当前 Sessions/Context 文档区分每轮模型输入选择与持久历史；第十七、二十一章提供任务/环境所有权边界。 | 18.1—18.2 定义折叠只改变模型输入视图，不把摘要当事实源；分别链接第七、十七、二十一章。 |
| 约束、事实、假设、决定、事件、Artifact 与未知 effect 的不同保留规则 | `Benchmarks Are Surprisingly Compaction-Friendly` 报告功能要求可从文件恢复时，评测可能漏看过程和范围约束流失；`The Compaction Cliff` 研究对不同知识类型采用分层保留与验证。 | 18.3—18.9 以 `task-42` 分类、转化、回查及逐项验证；保留 `unknown` 和 revision 绑定。 |
| 折叠流水线与预算预留 | Anthropic 工程建议从高召回调优；公开 API 文档显示供应商的 block、输入选择和 session 存储合同不同。课程样本以窗口变化图后接裁剪/压缩做讲解。 | 18.5—18.8 展示可检查步骤、明确的教学预算算例和发布校验，不制定通用阈值。 |
| 不同压缩和重启策略的取舍 | DeepSeek-V3.2 §4.4 对 Summary、Discard-75%、Discard-all 与并行轨迹做特定 Search Agent/BrowseComp 比较；分数、触发点、成本都绑定该实验。 | 18.10 解释策略差异；纠正 60.2 的纵轴口径，不把 80% 写成普适触发阈值。 |
| 折叠后是否继续遵守约束、能否恢复并避免重复副作用 | `Benchmarks Are Surprisingly Compaction-Friendly`、`The Compaction Cliff` 和 AutoCompact 的模型/任务/评测单位不同，但分别观察约束流失、类型化保留和折叠后续作行为。 | 18.12 独立报告任务、约束、证据、恢复与成本；不合并论文分数，不设综合 gate。 |
| Provider/SDK 的具体行为合同 | Anthropic Messages API 当前的 signed compaction block；OpenAI Agents SDK 的 `session_input_callback` 与 `SessionSettings` 的历史输入选择。 | 18.11 只作简短实现对照，详细 Anthropic 行为回链第七章；不把 SDK 裁剪当成已验证的通用摘要。 |
| 教学组织样本 | CUHK-Shenzhen 2026 Fall LLM Agents and Harness Engineering Lecture 07 从窗口压缩前后图引入；University of Mannheim IE685 将写入、选择、压缩、隔离放在 Context Engineering 总体架构下。 | 用作章节顺序参考，不主张它们证明了教学效果；正文仍按第七章输入构造→第十七章任务恢复→第十八章轨迹折叠的书内路线组织。 |

本轮扫描没有发现需要增设本册独立章节的新主线。新评测证据改变了本章的重点：必须把任务结果与约束保留分开看；不能以任务通过推断用户操作限制也被保留。上述发现只改变第十八章讲解与验收方法，不改变书册章节分工。

## 原稿覆盖与正文取舍

原稿 27 个小节含有可保留的关键知识，但反复描述分类 schema、预算水位、provenance、回放、失败回退和发布条件，且用多套自拟公式、任意 loss 权重和质量 gate 营造了并无来源支撑的统一度量。本轮合并为 14 节，围绕同一条 `task-42` 恢复路径讲解。

| 原稿内容 | 本轮处理 |
|---|---|
| 1—2：删最早消息的风险、约束/事实/假设/轨迹分类 | 保留并扩展至 18.1—18.4，增加 context / transcript / checkpoint / Trace / workspace 的定义边界。 |
| 3、11.1、11.4、13、17.1、17.3、17.4、17.6、17.17：状态 tuple、schema、冲突、来源与 provenance | 重组为 18.3—18.8：列类型、分型处理、可检查折叠步骤、task-42 交接摘要和摘要校验；移除会误读成固定 API 的重复类型 schema。 |
| 4—5、11.2、11.12、11.16：旧调试案例、任意 token-loss 优化、优先级分数和分层预算 | 将案例统一为第十七章 `task-42`；保留一次可复算的 128K/96K 教学预算；删除 `L_context`、`L_fold`、`U_i` 等自拟公式和优化权重。 |
| 6、11.3、11.5、11.14、11.20、11.24—25：水位、冷存、摘要版本、可逆性、提交/回退 | 合并为 18.5—18.9、18.13；保留安全时点、来源回查、版本/事件游标检查、`fold_pending`/提交语义与失败回退，删除任意通用水位/发布 gate。 |
| 7—8：RAG/memory 边界与来源引用图 | 在 18.1—18.2、18.5、18.7、18.11—12 中保留必要边界与来源引用；长期 memory 主讲交第七章，workspace/Artifact 交第二十一章。 |
| 9—10、11：回放评估和常见失败 | 扩展至 18.12—18.13；将评测单位、任务/约束/来源/副作用及预算拆开，不设总体成功分或固定 gate。 |
| 11.11.1—11.11.4：分类 schema、水位、provenance 和独立 worked example | 合并到 18.3—18.8，删掉重复案例，以全章同一任务演示。 |
| 12、14、19、21、23：自拟损失/事实率/约束损失/价值密度/统一发布条件 | 删除无来源的公式和综合门槛；分别说明任务结果、约束覆盖、来源支持、未知副作用和成本应各自报告。 |
| 13、15、16、18、20、22、26：伪代码 type schema、失败 trace 版本、流水线、token 压缩算例、CAS、人工 checkpoint、资料边界 | 保留读者需要的过程及失败恢复；使用可读的步骤表、教学预算算例与分型摘要；删除重复 pseudo schema。 |
| 27 DeepSeek V3.2 比较 | 保留策略定义和报告单位，链接官方技术报告及第二十一册 §19.35；将 60.2 约束为图中 BrowseComp 分数，明确 80% 是作者该实验触发点。 |
| 长 Python audit demo、硬阈值与 `release_condition` | 删除；章节不承诺可运行实现，且原指标没有任务通用定义。没有用另一套总分/gate 替换。 |

## 正文主线、教学材料与配套

- **reader_path：** 界定模型本轮实际可见的内容 → 区分 transcript/checkpoint/Trace/workspace → 回到 `task-42` 盘点历史 → 对材料分类并选择保留/聚合/摘要/检索/丢弃 → 算示例输入预算 → 生成带来源与 `unknown` 的工作摘要 → 独立核对约束、授权、版本、Artifact 与效果状态 → 恢复时重新查询易变事实 → 按任务挑选压缩策略并分别评估任务、约束、证据和成本。
- **example_status：** 全章沿用明确为教学构造的 `task-42`、`r41/r42`、`ci-884`、`review-481`；示例规定 CI 可按稳定 job ID 查询。作者知道远端审查对象存在，但恢复 Runtime 没有回执，折叠摘要因此保留 `unknown`。128K 窗口、96K 输入预算及 14K/77K/28K/68K 算术也是明确教学数字，不是产品配额或推荐水位。
- **evidence_status：** 官方产品文档用于说明具体 provider/SDK 合同；Anthropic 工程文章是供应商实践建议，不是独立实验。DeepSeek、AutoCompact、Compaction Cliff 和 Compaction-Friendly 结果均按原论文版本、模型、scaffold、任务与分母限定；长程评估/教学组织来源记录在 research 文件。
- **code_status / formula_status：** 不适用。本章目标是设计与审查 folding/recovery 路径；未承诺可运行代码。删除旧自拟损失、覆盖、约束损失和优先级分数公式。无需要图表才能理解而未呈现的视觉机会；预算和策略比较使用表格。
- **sync_status：** 更新目录第十八章摘要；新增练习 35EM 和面试题 40AB；更新册级 architecture map 与全局进度交接。简介仍准确，不需改写；未改 `plan.md`、PDF、术语表或知识图谱。

## 静态验收与遗留边界

- H2 标题 `18.1`—`18.14` 连续；无数学公式定界符、Python/可执行代码或 Mermaid；一个 `text` 围栏明确标为教学用摘要示意，围栏配对。
- 旧 `L_context`、`L_fold`、`R_fact`、`E_constraint`、`U_i`、固定 release gate 和旧案例标识无残留；DeepSeek 的 80%/60.2 均按来源范围表述。
- 静态验收：本章、目录、本章审计、来源记录、配套条目、册级架构审计和全局交接所含 80 个本地 Markdown 链接目标缺失 0；`git diff --check` 通过。目标读者试读、GitHub 页面锚点/渲染、所有外链的长期可达性未验证。
- 未运行书中代码、自动化测试、模型、Provider API、SDK、benchmark 或 PDF 构建；未做真实 Agent 折叠恢复 probe。
- 论文的摘要保留/行为结果均绑定其基准和实现。高风险生产任务还需验证实际权限执行、checkpoint、Artifact 保留与服务端查询合同，不能仅靠本章的文本摘要方案保证动作安全。

## 下一入口

[第二十册第十九章《Harness-Aware Evaluation》](../../book-20-agent-harness-runtime/chapters/19-harness-aware-evaluation.md)。在第十章评测 harness 基础上，核验模型、context-folding/harness、环境与任务交互如何共同影响结果；关注固定论文配置与成对比较条件，不把第十八章不同论文的结论合成排名。第二十册第 19—24 章和全套书精修 goal 继续 active。
