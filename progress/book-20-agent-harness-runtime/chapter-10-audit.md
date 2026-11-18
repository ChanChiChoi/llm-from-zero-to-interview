# 第二十册第十章精修审计

日期：2026-11-18  
章节：[第十章《Evaluation Harness》](../../book-20-agent-harness-runtime/chapters/10-evaluation-harness.md)  
状态：本轮正文精修完成；全书修书 goal 仍继续。

## 本章范围与读者路径

本章承接第九章的 Trace、Artifact 与运行证据，说明如何把多个 Agent 运行组织成可复现、可比较的评估。主要读者结果是：能够从 task fixture、环境预检、固定配置和 Runner 出发，选定相互独立的验收条件；能区分有效任务判定、Agent 失败、环境/运行故障、评分错误和无法判定；能成对比较 baseline 与 candidate，并在报告里给出各自口径、分母和证据边界。第十九章继续讨论 Model×Harness×Environment 的归因与交互；统计设计回第七册。

全章使用“首选显示名称优先、为空时回退账户名”的明确教学构造，不指称真实仓库或实际 run。五个 fixture A—E 中，E 在 Agent 启动前环境预检失败；在 A—D 上，baseline 通过 2/4、candidate 通过 3/4，D 展示功能测试通过但 diff 超出允许范围。另有独立的合成 .env 安全夹具，区分 Agent 请求、策略拒绝、实际派发和副作用。上述数字只服务手算，不是实验结果、经验基准或上线门槛。

## 章级定向扫描与证据

扫描日期：2026-11-18。问题：任务、环境、配置和验收器如何形成可重建的 Agent 评估；功能测试通过但过程/改动约束不满足时怎样判定；mock 测试能证明什么、外部集成还缺什么；环境/评分故障怎样与 Agent 失败分开；公平比较如何约束归因。

- [Agent Runtime 与 Harness 近期来源扫描](../../research/agent-runtime-and-harness-source-scan-2026-11-17.md)：保留确定性测试与 provider/network/sandbox 集成验证的边界，并限定到具体 SDK 文档。
- [Coding Agent 工作流与验收约束来源记录](../../research/coding-agent-workflows-and-acceptance-constraints-2026-11-17.md)：记录 SWE-Gate、ANCHOR 的任务范围、方法、数字和预印本边界。SWE-Gate 的 221/644 只陈述该数据集与实验设置中的 review-constraint 结果；不外推为通用违规率。
- [评估平台与裁判可靠性来源记录](../../research/evaluation-platform-evidence-2026-11-05.md)：Inspect scoring policy 区分模型有效 verdict、scorer error 和 unscored；OpenAI Evals 平台生命周期是单一供应商状态，不纳入通用章节主线。
- 本轮重新打开 OpenAI Agents SDK Testing、SWE-Gate arXiv v1、ANCHOR arXiv v2 和 Inspect AI Scoring Policy 核对正文使用的事实。动态文档的适用版本和范围均在正文或来源记录中标明。

扫描候选去向：
- 固定任务夹具、分桶与初态版本：本章正文；不新增章节。
- Environment preflight 与 Agent/scorer error 分类：本章正文，补足旧稿容易把不同故障混入一个成功率的风险。
- 功能测试与工程约束、轨迹评估：本章正文；第十九章深化配置交互。
- 内存态 mock 与真实 provider、网络、文件、进程、sandbox 行为：本章正文，使用特定 SDK 文档作为边界示例。
- OpenAI Evals 平台产品退役：留在 research 证据记录，不作为通用架构规律。
- 裁判校准、任务采样、方差和统计显著性：本章仅给必要引导；深入统计回第七册。

## 原有重要内容去向

| 原稿知识点 | 精修后位置/去向 |
| --- | --- |
| Evaluation Harness 目标与评估闭环 | 10.1—10.3：任务、环境、运行、证据、验收、比较、报告 |
| Dataset Loader、任务类型和样本 bucket | 10.3—10.5：固定夹具字段、去重/泄漏风险、预先声明任务桶 |
| Environment Manager、reset 与复现 | 10.3、10.6：依赖/镜像/revision、预检与启动前失败归类 |
| Runner、超时/预算/终止原因 | 10.7、10.15：配置身份和分层终止结果 |
| Tool/Sandbox Manager、mock tradeoff | 10.3、10.8：mock 覆盖 Harness 自有编排，真实集成验证外部系统边界 |
| 成功条件、功能/安全/范围判定 | 10.9、10.14：结果、过程约束、风险请求和副作用分开 |
| Metric Registry 与 15 项旧指标 | 10.5、10.11—10.16、10.19：保留 bucket、环境准备、verifier、hard success、diff、trace/版本、baseline 公平性、回归、安全、成本、重复结果和报告完整性意图；要求单位、适用条件和分母各自成立 |
| 旧 weighted/partial 分数 | 10.5、10.16：可按预先确定的业务理由呈现，但不作为默认 hard pass 或统一聚合目标 |
| Result Store、Trace 与 Artifact | 10.3、10.17—10.19：保存任务、配置、验收和可追溯的失败证据，并与第九章分工 |
| Regression Runner、baseline comparator | 10.12、10.18：成对条件、组件归因与回归维护 |
| Flaky evaluation、安全、真实坑 | 10.13、10.14、10.20：分别追查 Agent 随机性、环境/验收器波动和副作用 |
| 原始 10.19 聚合指标及 G_evalh | 删除统一门槛；保留相关观测目标，用独立分母和 N/A/unknown 表达 |
| 原始 10.19.1 约 450 行 Python 审计 demo | 删除；教学目标改为 A—E 手算、配对结果和安全事件追踪，不再声称运行代码 |
| 三道面试题、八道练习和总结 | 10.21—10.24 重写；[练习 35N](../../EXERCISES.md) 和[面试题 40L](../../INTERVIEW_BANK.md)同步改为手工设计与证据分析 |

## 正文、例子与形式状态

- **reader_path：** 从“Agent 自称完成”这个具体疑问进入；先界定 fixture/attempt/config/experiment，再按 pipeline 构造任务、预检环境、锁定 Runner、选择 mock/real coverage、设置 verifier 和 outcome 分类；最后用 A—E 手算成对比较，展开安全、成本、统计波动、报告与回归。
- **coverage_map：** 见上表；主线覆盖数据、环境、执行配置、验收与分母，支撑主题覆盖分桶、flakiness、trace/artifact、报告和回归；统计推断与第十九章的配置交互保留明确入口。
- **example_status：** 一个贯穿显示名修复任务；A—D 四个有效配对任务；E 在启动前环境失败；1 个独立合成 .env 请求被拦截的安全夹具。所有中间状态与计数已标记为教学构造，正文明确哪些数据不进入 agent denominator 但仍须报告。
- **evidence_status：** SWE-Gate/ANCHOR 仅支持功能判定与工程约束需分别评估的特定研究结论；OpenAI Agents SDK 示例限于当前 Testing utilities；Inspect scoring outcome 是该框架当前行为，不宣称通用标准。
- **formula_status：** 只保留任务成功率的基础比例公式，并定义 hard-pass 与有效判定分母，用 2/4 与 3/4 复算；没有为工程概念增造公式。GitHub MathJax 实际渲染未确认。
- **code_status：** 删除原长 Python audit demo；正文不再承诺可执行代码。本章只有一个 Mermaid 流程图，静态围栏配对检查通过；未在 GitHub 页面确认图形渲染。没有运行书中代码、自动化测试、模型、API、benchmark 或 PDF。
- **link_status：** 本章与本审计记录的本地 Markdown 链接均静态扫描，缺失目标为 0。
- **sync_status：** 更新共享练习 35N 与题库 40L；目录第十章摘要仍准确（固定任务/环境、verifier、可比条件），无需修改简介、目录、术语表或知识图谱。
- **删除与合并理由：** 原指标集合的有效观察意图已映射到各节；移除重复覆盖率、任意聚合权重和固定 pass gate，避免让清单本身替代可靠样本、验收质量和明确归因。原程序的输入字段审阅改为可以逐项核查的手工过程，正文没有保留未经验证的代码承诺。

## 静态验收

- H2 标题 10.1—10.24 连续；Mermaid 围栏标记 2 行、配对；行内数学分隔符 5 对，花括号配平。
- 本章与本审计的本地 Markdown 链接目标缺失 0；正文未残留 `G_evalh`、`evaluation_harness_gate` 或旧 audit demo 名称。
- `git diff --check` 通过；未运行测试或书中代码。

## 遗留事项

- Mermaid 和 MathJax 需有 GitHub 页面渲染时再做目标平台复核；本轮只做源码静态检查。
- 研究结论仍受任务样本、环境、模型、Harness、验收方式和论文版本限制。
- 无目标读者试读，编辑自检不冒充用户测试。
- 下一入口为[第十一章《Claude Code 架构分析》](../../book-20-agent-harness-runtime/chapters/11-claude-code架构分析.md)。第二十册第 11—24 章及全套书余项继续待精修。
