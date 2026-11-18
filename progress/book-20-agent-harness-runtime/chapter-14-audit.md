# 第二十册第十四章精修审计

日期：2026-11-18  
章节：[第十四章《MCP、A2A 与 Harness 集成》](../../book-20-agent-harness-runtime/chapters/14-mcp-a2a与harness集成.md)  
状态：章级扫描与正文精修完成；全套书精修 goal 继续 active。

## 章节契约与读者路径

本章承接第 4 章工具注册、第 8 章权限与沙箱、第 9 章 Trace 和第 10 章 Evaluation Harness。它不重复讲 MCP/A2A 的完整协议教程，而是回答：当一个任务需要外部工具和远端 Agent 时，Harness 如何把发现的能力、安全策略、执行事实、状态、Trace 和独立验收连起来？协议本身的概念、生态和完整交互仍由第二十二册主讲。

教学构造为“订单接口在客户端超时重试后偶尔创建重复订单；读取 issue/CI、准备本地修复和回归测试；禁止评论、push、部署和访问客户数据”。读者先沿 MCP 只读发现和请求路径，再看 A2A AgentCard、任务、补充输入和 Artifact，之后检查身份/授权/确认/隔离、状态取消、外部内容信任、Trace、Replay 与本地验收。离开本章时，读者应能为一次跨协议委派保留清楚的权限边界、状态所有权和验收证据，并能解释超时或取消时仍未知的部分。

## 章级扫描与候选去向

检索日期：2026-11-18。研究问题：MCP 与 A2A 的当前协议合同如何进入 Coding Agent Runtime；协议提供什么，而 Harness 仍须负责什么？已打开并核对 MCP `2026-07-28` 正式规范、发布说明、Tools、Authorization、安全最佳实践和 Tasks 扩展页面；打开并核对 A2A v1.0.1 规范和版本发布记录。教学组织参考既有[软件架构案例章组织样本](../../research/software-architecture-case-study-organization-2026-11-18.md)。各来源支持事实、版本状态和边界见[Agent 协议 Runtime 集成证据](../../research/agent-protocol-runtime-integration-evidence-2026-11-18.md)。

| 外部主题 | 读者结果 | 正文去向 | 证据及边界 |
|---|---|---|---|
| 协议对象与边界 | 区分 MCP server 的工具/上下文能力与独立 Agent 的 A2A task 协作 | 14.2、14.5 | MCP 固定版规范与 A2A v1.0.1 规范；完整概念转交第二十二册。 |
| MCP 生命周期与发现 | 区分 `2026-07-28` 无协议会话请求、`server/discover`、旧版握手和可选扩展 | 14.3—14.4、14.7 | 2026-07-28 正式规范与发布说明；Tasks 扩展页面当前标为 Draft，不推断所有 SDK/host 已实现。 |
| A2A 发现与任务结果 | 从 AgentCard 选择接口，保留 Task/Message/Artifact 边界并验证远端结果 | 14.3、14.5、14.11 | v1.0.1 固定规范；远端 AgentCard/Artifact 是对端声明，不是本地信任或验收证明。 |
| 认证、授权、确认与隔离 | 分开处理调用身份、远端资源授权、本地任务策略、人工确认和执行隔离 | 14.6 | MCP Authorization、安全指南与 A2A 认证/授权章节；协议不规定完整 Harness 业务策略或 OS sandbox。 |
| 任务状态、超时和取消 | 保留本地 ID、MCP request/可选 task handle、A2A task/context ID；避免未知状态下盲重试 | 14.7、14.12 | MCP Tasks Draft 与 A2A CancelTask/TaskState；取消只是请求，不能证明副作用已撤销。 |
| 外部内容、Trace 与评估 | 保留数据来源和敏感边界，将远端状态接入本地 trace、replay 和 verifier | 14.8—14.11 | MCP security best practices、A2A Artifact/Task 和既有 Trace 证据；不以远端完成状态替代本地任务验收。 |

章级扫描没有发现需要新增独立章节的新主题。协议语义、生态和完整教学各有第二十二册主讲位置；本章的统一读者结果是把远端能力纳入运行时闭环。简介、目录中的本章职责仍准确，章节数和职责不变。

## 原稿覆盖与本轮取舍

| 原稿内容 | 本轮处理 |
|---|---|
| MCP Host/Client/Server、Resources/Prompts/Tools 与 A2A AgentCard/Task/Artifact 基础 | 保留读者完成 Runtime 接入所需的最小模型；完整定义和实现例子链接第二十二册。 |
| “MCP 支持 stateful connections”的泛化和 latest specification 口径 | 改为固定 MCP `2026-07-28`：新版核心无协议级 session，较早版本仍有 handshake；另说应用可保存外部状态，Tasks 为 Draft opt-in extension。 |
| 将用户明确同意说成 MCP 对所有调用的硬要求 | 改为准确表达：Tools 规范不强制唯一 UI 模式，并 SHOULD 提供用户拒绝能力；Host 的具体策略和界面须由实现承担。 |
| A2A task 状态与 streaming/push 列表 | 固定 v1.0.1；采用 TaskState、Message、Artifact 和取消不保证成功等与 Runtime 边界直接相关的机制，不复制完整协议教程。 |
| 动态产品工具搜索、Claude Code/OpenCode 行为与产品细节 | 移出本章主线；保留通用发现、能力快照、schema/来源/版本审阅原则，不将特定 host 合同写成协议要求。 |
| 多组 capability coverage 公式、协议总 gate、长 Python audit demo 和练习脚本要求 | 删除；改用同一条合成运行轨迹、权限/状态/失败表、具体恢复问题和固定回执/mock/真实集成的证据边界。 |
| 题库、练习与章节结构 | 更新练习 35R 和面试题 40P；其他书册已有协议教程未重复粘贴。目录摘要仍匹配。 |

## 正文状态

- **reader_path：** 合成订单故障 → MCP/A2A 接入对象 → 版本化发现与能力注册 → MCP 只读工具请求 → A2A Agent 委派及 Artifact → 身份/策略/确认/隔离 → 状态、取消与幂等 → 不可信内容、Trace、Replay 和独立验收 → 复走完整账本与失败恢复。
- **coverage_map：** 核心为 MCP/A2A 运行时边界、版本和能力来源、远端身份及本地授权、跨协议 ID/状态映射、Task 完成与本地验收的区分、取消/超时/副作用和评估复现边界。支撑为 AgentCard、MCP tool/resource/prompt、Artifact、输出预算、Trace 关联和失败分类。协议完整对象模型与生态范围转交第二十二册；通用权限/Trace/Eval 方法回链本册第 8—10、17 章。
- **example_status：** 使用订单重试教学构造展示 issue/CI 的 MCP 只读访问、受限上下文的 A2A 测试设计委派、`INPUT_REQUIRED`、远端 Artifact、本地补丁/测试及独立验收。标明全程为教学构造，不暗示协议或产品真实运行。表格贯穿输入、动作、状态、原因、证据和下一状态拥有者。
- **evidence_status：** MCP 事实绑定 `2026-07-28` 版规范；明确较早握手版本仍存在，Tasks 扩展页面状态为 Draft。A2A 事实绑定 v1.0.1 规范/接口版本。MCP 工具用户交互按 SHOULD 表达；认证、远端授权和本地策略分别说明。新事实与未验证边界记录于对应 research 文件。
- **formula_status：** 不适用。移除原覆盖率公式与加权 gate，改用逐事件审阅和独立证据判断。
- **code_status：** 无可执行代码；1 个 Mermaid 运行图，源码有 1 对围栏。未运行协议 SDK、MCP server、A2A agent、模型、API 或测试。
- **link_status：** 本章、练习 35R、面试题 40P、第二十二册导航、章节审计、架构审计和研究记录的本地链接经静态检查，缺失 0；全站外链长期可达性未检查。
- **sync_status：** 已更新练习 35R、面试题 40P、第二十册架构审计；目录第十四章摘要准确，无需改简介或目录。
- **open_risks：** 未做 GitHub Mermaid 页面渲染、目标读者试读或 SDK/服务兼容测试。MCP Tasks 扩展为 Draft，具体客户端支持需按目标实现复核；远端 AgentCard、Task、Artifact 与业务系统副作用仍要由部署方核验。

## 静态验收

- H2 标题 14.1—14.15 连续；Markdown 围栏 2 个标记、成 1 对；无公式、Python demo、固定覆盖率或协议集成 gate 残留。
- 本章、练习、面试题、研究记录及进度记录涉及的本地 Markdown 链接静态检查缺失 0。
- `git diff --check` 通过。
- 未确认 GitHub Mermaid 页面渲染；未做目标读者试读；未运行代码、MCP/A2A 实现、产品、模型、测试、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

## 下一入口

[第二十册第十五章《Agent Harness 系统设计》](../../book-20-agent-harness-runtime/chapters/15-agent-harness系统设计.md)。将本章的 capability discovery、权限和委派边界接入全书架构设计，复核各模块的数据/状态所有权及端到端验收。第二十册第 15—24 章和全套书其余内容继续精修。
