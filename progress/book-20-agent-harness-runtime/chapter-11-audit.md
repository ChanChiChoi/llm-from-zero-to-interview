# 第二十册第十一章精修审计

日期：2026-11-18  
章节：[第十一章《Claude Code 架构分析》](../../book-20-agent-harness-runtime/chapters/11-claude-code架构分析.md)  
状态：本轮章级扫描与正文精修完成；全套书精修 goal 继续 active。

## 本章范围与读者路径

本章承接第二十册第 1—10 章的 Harness、Runtime、工具、文件、终端、上下文、权限、Trace 和 Evaluation Harness，训练读者用公开证据分析一个内部细节未完全公开的 Coding Agent 产品。中心问题是：怎样从产品合同和可观察行为中解释工作循环与控制边界，同时把架构分层保持为有标记的分析推断？

读者进入时应已理解 agent loop、工具调用、权限与隔离、会话状态、运行 trace 和任务 verifier。读完后能沿一个代码修复任务区分官方公开描述、某次运行的观察、设计推断和未知内部；还能检查产品表面与执行环境、上下文来源、权限决定、实际执行、状态恢复、扩展和独立验收的关系。第 12 章转向可检查开源实现的 OpenCode，第 13 章对齐不同产品的比较维度。

## 章级定向扫描

扫描日期：2026-11-18。问题：Claude Code 当前公开资料支持哪些 agentic loop、产品表面、状态、执行边界和扩展事实；哪些架构描述仍属分析者推断；如何让本章案例沿证据而非闭源猜测组织？

- 技术资料：核对 Anthropic 官方 How Claude Code works、Platforms and integrations、Settings、Memory、Permissions、Sandboxing、Checkpointing、Hooks、MCP、Subagents、Skills、Plugins 与 Monitoring 文档。来源及逐项支持范围记录于[Claude Code 公开产品合同与架构边界](../../research/claude-code-public-product-contracts-2026-11-18.md)。
- 组织样本：对照《Architecture of Open Source Applications》官方编者序言和 FAQ；组织选择与证据边界记录于[软件架构案例章组织样本](../../research/software-architecture-case-study-organization-2026-11-18.md)。该样本可作案例章组织线索，不作为学习效果证据。
- 册级关系：对照第 10 章 Evaluation Harness、第 12 章 OpenCode 架构分析、第 13 章主流 Coding Agent 对比。任务评估主线留给第 10 章；开源实现细节由第 12 章承接；产品横向比较留给第 13 章。

候选主题去向：

| 候选主题 | 外部证据与边界 | 去向 |
| --- | --- | --- |
| gather context / take action / verify 与工具类别 | Anthropic 官方文档明确公开描述产品工作循环和工具类别，但不提供内部源码模块图。 | 本章 11.2、11.4—11.5；将任务顺序作为合成解释，不冒充实测。 |
| 多产品表面、执行环境与设置范围 | 官方文档列出 CLI、IDE、Desktop、Web、移动端和集成，并区分 local/cloud/Remote Control 及配置范围。 | 本章 11.3、11.6；将“终端型”缩为历史入口而非完整现状。 |
| Instructions、Auto memory、session transcript 与 checkpoint | 官方资料分别说明信息输入、会话记录和部分文件恢复能力。 | 本章 11.7—11.8；明确不同状态的证明力和撤销边界。 |
| 产品权限、OS shell sandbox 与扩展执行路径 | 当前官方文档明确 sandbox 只覆盖特定 shell 命令和子进程，列出在外部运行的 file/Web tools、MCP、hooks 等路径。 | 本章 11.9；核心控制边界，不合成一个 permission/sandbox 模块。 |
| hooks、MCP、subagents、skills 与 plugins | 各官方页面给出不同扩展功能和配置/执行语义。 | 本章 11.9；逐类审计来源、作用域、执行位置与权限。 |
| transcript、OpenTelemetry、trace 与评估 | 当前官方 Monitoring 文档列出 metrics、events/logs、可选 beta traces 及由配置控制的部分工具内容字段。 | 本章 11.10—11.11；完整任务集与 verifier 回第 10 章。 |
| 隐藏 planner、完整 system prompt、内部排序和路由 | 官方资料未公开这些机制；本轮没有产品源码或直接运行证据。 | 标为未知，不纳入产品内部事实，也不新增章节。 |

本次没有发现改变本册分工或须新增独立章节的主题。章节继续作为基于公开资料的产品案例，范围与相邻章节职责保持清楚。

## 原有内容去向

| 原稿主题 | 精修后去向 |
| --- | --- |
| Claude Code 作为 coding agent 案例及读者目标 | 11.1—11.4：明确产品范围、中心问题和证据等级。 |
| 终端 vs IDE 的单一产品划分 | 11.3：更新为多产品表面与执行环境、设置合同的比较。 |
| 用户输入、slash command 与 task routing | 11.2、11.6：用合成登录修复任务区分用户控制输入与模型工具调用；不猜测内部 router。 |
| Context Builder、Model Adapter、Action Parser、Tool Registry 等高层模块图 | 11.4—11.5：仅作为分析者可选的职责抽象，不声称是 Claude Code 的内部模块或服务。 |
| 上下文、文件、终端工具与验证循环 | 11.2、11.5、11.7、11.9：连接官方工具能力、执行回执和独立验收。 |
| 权限与人工确认 | 11.6、11.9：分开产品权限规则、用户决定、实际执行和 OS sandbox。 |
| 会话恢复、memory、checkpoint | 11.7—11.8：将 session/context、项目指令、Auto memory、settings 和文件 checkpoint 分账；外部副作用另查。 |
| trace / 日志 / 任务循环 | 11.5、11.8、11.10：区分本地 transcript、checkpoint、可选 OTel 和 verifier。 |
| 真实工程坑 | 11.12：保留证据越界、确认/隔离混淆、sandbox 范围、状态恢复和单次成功外推等风险。 |
| 10 个架构覆盖率及 G_claude 综合门槛 | 删除；将其审阅意图转为 11.11 的问题—证据—边界手工检查，不计算覆盖分数或通用 gate。 |
| 约 140 行 Python 架构审计 demo | 删除；练习改为合成任务人工证据审阅，不声称运行代码。 |
| 四道面试题及章末练习 | 改写为 11.13—11.14；同步 EXERCISES.md 35O 与 INTERVIEW_BANK.md 40M。 |

## 正文与形式状态

- **reader_path：** 从合成测试预期 401、实际 200 的修复任务切入；沿公开 gather/action/verify 描述走一遍，但每一步标清文件、工具、权限或输出只是合成示例；再据官方材料解释产品表面、证据等级、上下文/状态、权限/sandbox、扩展和观测；最后用人工审计问题将方法迁移到其他产品。
- **coverage_map：** 核心：产品公开能力与未知内部的证据边界；工具动作的提议/批准/执行/验收；权限规则与 OS sandbox 区别；上下文、会话、Auto memory 和 checkpoint；文件、MCP、hooks 等扩展边界。支撑：多表面配置、session transcript、OTel、产品控制命令、恢复与安全常见误区。通用评估统计交由第 10 章；OpenCode 源码实现交由第 12 章；横向排名不在本章处理。
- **example_status：** 使用一项合成登录测试任务：预期 HTTP 401、观察值 HTTP 200，追踪读取/提议/修改/权限/测试/diff/验收。正文明确没有把它写成 Claude Code 实测，也没有模拟工具执行结果。
- **evidence_status：** 产品事实限于 2026-11-18 核对的 Anthropic 官方在线文档；AOSA 只支持一种架构案例组织实例，不声称证明教学效果。未知内部实现保留为未知。没有调用产品或验证真实环境。
- **formula_status：** 不适用；纯产品/系统分析无必要数学公式。
- **code_status：** 删除旧 Python audit demo；正文无可执行代码。含一个 Mermaid 分析图；静态围栏配对后再复核，未确认 GitHub 页面渲染。
- **link_status：** 本章、目录、本章审计与研究记录的本地 Markdown 目标静态检查缺失 0。
- **sync_status：** 已更新目录第 11 章摘要、练习 35O 和面试题 40M；简介其余内容、术语表、知识图谱没有因本章职责变化而需调整。
- **open_risks：** 官方在线文档会变更，权限模式、执行表面和遥测字段须按访问日期复核；没有真实产品运行记录或目标读者试读；页面 Mermaid 渲染与所有外链的长期可达性尚未确认。

## 静态验收

- H2 标题 11.1—11.15 连续；Mermaid 围栏标记 2 行、成对；所有带文字箭头使用管道标签语法，图中职责分层明确标为分析模型。
- 本章、目录、本审计、册级架构审计和两份研究记录共扫描本地 Markdown 链接目标，缺失 0。
- 正文无 G_claude、旧覆盖率字段、11.18.1 或 Python audit demo 残留。
- git diff --check 通过。
- 未做 GitHub Mermaid 页面渲染、目标读者试读或全部外部链接长期可达性核验；未运行 Claude Code、测试、模型、API 或 benchmark。

## 连续复读补记（2026-11-19）

- 沿第十章的独立验收证据复读本章，确认公开工作循环、产品表面/执行环境、上下文来源、会话与文件 checkpoint、权限/OS 沙箱、扩展、遥测和未知内部之间的证据边界清楚；案例结构承接第 1—10 章并把开源实现和横向比较分别交给第 12、13 章，没有发现需要改变章节职责、目录或配套资料的新主题。
- 将合成任务的文件路径、HTTP 状态码、CLI 命令、配置作用域和权限规则值改用代码格式；将权限规则优先级明确写为从高到低 `deny`、`ask`、`allow`。对 OpenTelemetry 的定义由“可观测性标准”修正为“可观测性框架与工具集”，并补充其官方定义链接；对应来源证据已加到[Claude Code 公开产品合同与架构边界](../../research/claude-code-public-product-contracts-2026-11-18.md)。
- 复用 2026-11-18 的 Claude Code 官方文档扫描；该扫描范围、版本边界和本章主题未变化。仅针对术语定义于 2026-11-19 阅读 OpenTelemetry 官方说明，确认其生成、收集和导出遥测数据的框架/工具集定位，不把它误称为存储或可视化后端。
- 静态验收：`git diff --check` 通过；对正文及五份维护/研究记录中的 472 个本地 Markdown 链接目标进行检查，缺失 0；本章 Mermaid 围栏标记共 2 行、成对。此处未检查 GitHub 页面上的 Mermaid 实际渲染。
- 本轮未运行 Claude Code、书中代码、测试、模型、API 或 benchmark，未更新 PDF 或 `plan.md`。目标读者试读和全量外链可达性仍留作整册终验。
