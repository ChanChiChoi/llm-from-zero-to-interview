# 第二十册第十二章精修审计

日期：2026-11-18  
章节：[第十二章《OpenCode 架构分析》](../../book-20-agent-harness-runtime/chapters/12-opencode架构分析.md)  
状态：章级扫描与正文精修完成；全套书精修 goal 继续 active。

## 本章范围与读者路径

本章承接第二十册第 1—10 章的 Runtime、工具、文件/命令执行、上下文、权限、trace 和 Evaluation Harness，再与第十一章 Claude Code 公开产品案例形成对照。中心问题是：如何沿一项任务，用指定版本的公开合同和源码证据解释配置、Agent、权限、实际执行、Session 状态和最终验收之间的关系？

读者进入时应已理解模型工具调用、action/permission、执行环境和独立 verifier。读完后能沿一个 Coding Agent 任务区分模型提议、宿主决定、实际副作用、状态记录和验收；能审阅 OpenCode V2 的配置/指令、工具/Agent、权限、扩展、Server/API/SDK、Snapshot 和 compaction 边界；也能指出项目本身不提供 Agent sandbox。第十三章承接统一维度的产品比较。

## 章级定向扫描

扫描日期：2026-11-18。问题：OpenCode 当前 V2 对外暴露哪些控制与状态合同；哪些实现路径可由指定版本源码核验；开放源码、权限提示、服务接口和快照分别不能证明什么？

- **固定版本与时效来源：** 官方 changelog 与 GitHub release 确认 v2.3.2 为 2026-11-18 发布，release commit c0ad18b。根据官方迁移说明，V1/V2 配置、插件、网络服务 API 和权限命名不可混用。
- **产品合同：** 定向核验 OpenCode Config、Instructions、Agents、Tools、Permissions、MCP servers、Server plugins、API reference、SDK overview、Web、Snapshots、Compaction 文档。
- **源码：** 固定 revision c0ad18b，定向审阅 SECURITY.md、packages/core/src/permission.ts 和 packages/core/src/session.ts 的威胁模型、权限求值与 Session 操作接口；没有做全仓源码审计。
- **组织样本：** 复用[软件架构案例章组织样本](../../research/software-architecture-case-study-organization-2026-11-18.md)：用一项任务解释关键部件如何交互及设计原因，避免把章节写成模块目录。该样本是案例组织经验，不作为学习效果证明。
- **事实和来源范围：** 新发现与固定版本事实只记在[OpenCode V2 Runtime 合同与架构边界](../../research/opencode-v2-runtime-public-contracts-2026-11-18.md)。

候选主题去向：

| 外部主题 | 证据支持 | 正文去向 |
| --- | --- | --- |
| V2/V1 版本和配置合同差异 | 当前迁移文档与 v2.3.2 release | 12.2 固定版本，避免旧工具字段、插件和 API 说法混入 V2。 |
| 配置层级、Agent 规则、managed policy | 当前 Config/Agents/Permissions 文档 | 12.4 解释配置合并、agent permission 追加和组织策略边界。 |
| AGENTS.md、MCP/Skill instruction 与工具 | Instructions、Tools、MCP 文档 | 12.5、12.7、12.10 区分模型上下文与可执行能力。 |
| 子 Agent 权限与后台运行 | Agents 文档及 v2.3.0 changelog | 12.6 说明父级控制启动对象、子 Agent 使用自身权限，并且运行可在后台继续。 |
| action/resource/effect 与规则求值 | Permissions 文档及 c0ad18b permission.ts | 12.8 跟踪一次权限判断；明确规则、审批和执行隔离不是一回事。 |
| OpenCode 不提供 Agent sandbox | 固定版本 SECURITY.md | 12.9 作为本章高优先级执行边界。 |
| Code Mode、MCP 与 plugin | Tools、MCP servers、plugin 文档 | 12.7、12.10 分别说明嵌套工具调用和扩展路径；不等同于系统级隔离。 |
| Server API、Client 和嵌入式 SDK | V2 API、SDK、Web 和迁移文档 | 12.11 区分网络 API、本地 TUI/Web server 和进程内 SDK，并说明默认 loopback、浏览器配对、HTTP 客户端认证和密码变量兼容。 |
| Session、Event、Snapshot、Diff、Revert | API、Snapshot 文档及 c0ad18b session.ts | 12.12 解释运行记录与文件恢复范围；单个非忽略 untracked 文件超过 2 MiB 时不捕获，不把回滚写成完整事务。 |
| Summary 与 native compaction | Compaction 文档 | 12.13 解释模型可见历史的有损变化及可移植性边界。 |
| 任务级独立评估 | 第十章 Evaluation Harness | 12.3、12.12、12.14 只建立验收证据边界，不重复主讲评估统计。 |

本次没有发现需调整本册章节分工或新增独立章节的主题。第十二章继续作为固定版本开源 Runtime 案例，第十三章横向比较产品，第十四章讲协议集成。

## 原有内容去向

| 原稿主题 | 精修后去向 |
| --- | --- |
| 本章目标、产品定位、可信边界 | 12.1—12.2：以具体任务定义读者结果；固定 v2.3.2、commit 和证据等级。 |
| TUI/IDE/Web/SDK 产品表面与高层架构 | 12.3、12.11：沿任务画公开客户端和 Runtime 控制路径；图明确为分析抽象，不声称源码模块图。 |
| 配置来源、remote/managed、project/global config | 12.4：基于当前文档说明配置合并与 policy 作用域；不保留 V1 旧配置位置列表。 |
| AGENTS.md 与项目指令 | 12.5：保留上下文作用，纠正把指令当授权或安全边界的说法。 |
| primary/subagent、Build/Plan、隐藏 agent | 12.6：更新内建 Agent 名称和 V2 后台子 Agent 合同，纠正 Plan 普遍只读和子 Agent 自动继承父权限的说法。 |
| bash/edit/write/read/grep/glob/LSP/apply_patch/todowrite 等静态工具目录 | 12.7—12.8：不复述过时工具名；按能力类型和 action/resource/effect 解释目录、权限与派发。 |
| permission pattern、默认值和“权限是安全边界” | 12.8—12.9：删除旧 schema/default 与沙箱化表述；增加固定源码 last-match、saved approval 和 no-sandbox 边界。 |
| 文件、shell、search、LSP、MCP、自定义工具、插件 | 12.7、12.9—12.10：保留读者需要的扩展分类；说明 MCP/Skill/Plugin/Tool 各自职责，不承诺 extension 自带隔离。 |
| Compaction、Summary、Server/API、Session、Diff、Revert、SDK | 12.11—12.13：绑定 V2 文档与固定 Session 源码；区分网络客户端、进程内 SDK、事件记录、快照恢复与有损上下文压缩。 |
| Claude Code 比较 | 从本章主讲移出，交第十三章统一对齐条件，不在本章以产品类型做无证据概括。 |
| 真实工程坑 | 12.4—12.13 中按配置优先级、指令边界、子 Agent、无沙箱、remote service、snapshot 与 compaction 限制贴近原因解释。 |
| 旧 10 项覆盖率、G_opencode gate 与固定阈值 | 删除；用 12.14 的问题—证据—边界表保留人工审阅意图，不产出分数或发布门槛。 |
| 约 140 行 Python audit demo | 删除；不需要 toy capability schema 或执行脚本来展示架构判断。 |
| 面试和练习 | 改为证据分类、合成任务轨迹和恢复边界；同步练习 35P 和面试题 40N。 |

## 正文状态

- **reader_path：** 从合成认证任务的 200/401 结果和改动范围开始；固定 v2.3.2；区分文档合同、固定源码、一次运行观察和架构推断；依次追踪配置/指令、Agent/工具、权限、宿主执行、扩展和服务入口，再检查 Session/Snapshot/compaction 与独立验收。
- **coverage_map：** 核心：版本/证据边界、配置与 instruction、Agent 和子 Agent 权限、工具提议到实际派发、无 sandbox 和 host authority、MCP/Plugin/Skill、Server/API/SDK、Session 状态与独立 verifier。支撑：权限规则匹配、saved approval、Code Mode、事件、snapshot 范围、compaction。通用统计评估仍由第十章主讲；横向产品评价交第十三章；协议全景交第二十二册。
- **example_status：** 使用明确标为教学构造的认证测试修复任务，追踪 read、edit、局部测试、拒绝远端 push、diff/snapshot 和独立 verifier。没有把案例写成真实 OpenCode 运行。
- **evidence_status：** 官方文档访问日期为 2026-11-18；release/tag 固定至 v2.3.2 / c0ad18b。正文直链到 changelog、migration、Web/API/Snapshot 文档，以及固定 revision 的 permission.ts 与 session.ts；源码只支持各自审阅的具体路径。SECURITY.md 提供项目威胁模型声明。无完整源码审计或部署观察。
- **formula_status：** 不适用；本章是产品/系统架构案例，未为指标制造公式。
- **code_status：** 删除 Python audit demo；正文无可执行代码，含 1 个 Mermaid 任务路径图；静态检查带文字箭头使用管道标签。未确认 GitHub 页面实际渲染。
- **link_status：** 本章、目录、配套资料、审计和研究记录静态检查本地链接目标。
- **sync_status：** 更新目录第 12 章摘要、练习 35P、面试题 40N；简介书册职责未变，无需改简介。
- **open_risks：** 官方文档持续变化；下一轮须按所选版本重查。没有运行 OpenCode、MCP、plugin、API、model 或 test；没有目标读者试读、部署级权限/网络审计或 GitHub Mermaid 页面检查。API reference 中部分 V2 HttpApi 标为 experimental。

## 静态验收

- H2 标题 12.1—12.15 连续。
- Mermaid 围栏成对，流程图中的文字箭头采用管道标签；源码图的分析抽象边界在正文中声明。
- 第十二章、目录、第十二章审计、册级审计、练习、题库和 research 记录中的本地 Markdown 链接目标缺失 0。
- 正文无旧 G_opencode、coverage ratio、fixed threshold 或 audit demo 代码残留。
- git diff --check 通过。
- 未在 GitHub 确认 Mermaid 渲染；未检查所有外部链接的长期可达性，未做目标读者试读。
- 未运行书中代码、OpenCode、模型、API、MCP、插件、测试、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。

## 连续复读补记（2026-11-19）

- 沿目录复读第十二章，核对固定版本案例从配置/指令、Agent/工具、权限求值、宿主执行、扩展、Server/API/SDK、Session/Snapshot/compaction 到独立验收的解释链。与第十一章的闭源公开合同案例形成证据方法对照，并将统一维度比较交给第十三章；未发现需要改变章节分工、目录或配套资料的新主题。
- 规范 HTTP 状态码、路径、版本/revision、配置字段、权限 action/effect、API 类型和 package 名称的代码标记；把 Session 源码路径补全为 `packages/core/src/session.ts`。澄清固定源码“取最后匹配规则”的实现范围，并按官方文档说明 saved approval 不覆盖显式 `deny`、求值为 `deny` 时在 plugin `permission/evaluate` hook 前返回。
- 复用 2026-11-18 对 OpenCode `v2.3.2` / `c0ad18b` 的版本固定扫描。文档合同、定向源码路径及适用边界未变化；本轮未运行 OpenCode、插件、MCP、API 或测试。
- 静态验收：正文 Mermaid 围栏标记共 2 行、成对；`git diff --check` 通过；对第 11、12 章正文及相关审计、连续复读、架构和全套进度/研究记录共 8 份 Markdown 检查 479 个本地链接目标，缺失 0。未检查 GitHub 页面图示渲染、未做真实部署审计或目标读者试读。
