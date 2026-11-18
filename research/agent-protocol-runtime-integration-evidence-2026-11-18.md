# Agent 协议 Runtime 集成：公开合同与证据边界

核验日期：2026-11-18。本文记录 MCP 与 A2A 如何描述外部能力、任务、身份和结果，以及协议合同没有替 Harness 完成的工作。版本化规范、动态安全指南和扩展草案分别标明；不记录章节审阅过程或正文取舍。

## 来源与阅读范围

| 来源 | 版本/日期 | 阅读范围与支持内容 |
|---|---|---|
| [MCP Specification](https://modelcontextprotocol.io/specification/2026-07-28) | 2026-07-28 固定版 | Overview、消息模型、host/client/server、stateless 请求、按请求能力协商、Tools/Resources/Prompts 与扩展。 |
| [MCP 2026-07-28 发布说明](https://blog.modelcontextprotocol.io/posts/2026-07-28/) | 2026-07-28 | 移除 initialize 握手和 `Mcp-Session-Id`、`server/discover`、`Mcp-Method`/`Mcp-Name` 路由头、Multi Round-Trip Requests、缓存提示、授权加固和扩展框架。 |
| [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) | 2026-07-28 固定版 | 工具列表、用户交互、input/output schema、`input_required`、工具结果和服务端管理的显式状态句柄。 |
| [MCP Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) | 2026-07-28 固定版 | 授权为可选能力；HTTP transport 的 OAuth 授权流程；stdio 凭据处理不使用该 HTTP 授权流程。 |
| [MCP Security Best Practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices) | 对应 2026-07-28 页面 | Confused deputy、token passthrough、SSRF、scope 最小化、显式状态句柄劫持、CIMD 信任策略等实现风险。 |
| [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks) | 2026-07-28；页面标为 Draft | 扩展标识、能力协商、服务端按请求决定是否返回异步 Task handle、`tasks/get`、`tasks/update`、`tasks/cancel` 和最终结果。 |
| [A2A Specification](https://a2a-protocol.org/v1.0.1/specification/) | v1.0.1 固定版 | AgentCard、supported interfaces、capabilities、security schemes、Message/Task/Artifact、状态、操作、认证、授权、取消、版本和安全边界。 |
| [A2A v1.0.1 release](https://github.com/a2aproject/A2A/releases/tag/v1.0.1) | tag v1.0.1；2026-05-26 | 固定所引用的 A2A 规范发布快照；release 列表显示其为当时最新 tag。 |
| [A2A 官方文档入口](https://a2a-protocol.org/v1.0.1/) | v1.0.1 文档集 | 官方入门导航与协议概念；用于确认快速入门和规范页面入口，不代替规范性定义。 |

教学组织样本见[软件架构案例章组织样本](software-architecture-case-study-organization-2026-11-18.md)。该样本记录 AOSA 对架构案例如何围绕模块互动与设计理由组织的编辑建议；它不证明某种组织方式对所有读者都更有效。

## MCP 公开合同

- MCP 以 JSON-RPC 消息连接 Host、Host 内的 Client connector 与提供上下文/能力的 Server。Server 可以提供 Resources、Prompts 和 Tools；可选扩展必须由双方显式支持。这个模型定义互操作对象，不指定所有 host 的 UI 或执行架构。
- `2026-07-28` 是无协议会话的版本：不再使用 `initialize`/`initialized` 与 `Mcp-Session-Id`；每个请求带版本、client 元数据与按请求能力，客户端可用 `server/discover` 预先了解服务能力。较早的 MCP 版本仍采用握手生命周期，不能把新版规则回推到旧客户端/服务器。
- Tool list 是服务端面向当前请求提供的能力集合。该集合可为空、随时间变化，也可根据随请求提供的授权改变；服务端应保持稳定顺序，以利缓存。一个 tool schema 约束接口形状，但不等同于 host 的权限策略、操作系统沙箱或任务验收。
- 规范不强制唯一的人机交互模式；MCP Tools 规范建议应用显示暴露给模型的工具、标示工具调用，并提供人拒绝调用的能力。协议中的建议仍需 host 实际落实。
- MCP HTTP 授权是可选实现能力；启用时应采用规范定义的 OAuth 流程。stdio transport 不应套用 HTTP 授权流程，而应从环境取得凭据。授权机制说明凭据怎样用于对 server 发请求，不替应用判定所有业务操作是否合法。
- 2026 规范核心无协议级会话，但需要跨请求状态的 Server 可生成显式状态 handle，并由调用者在后续请求传回。安全指南要求服务端不能把“持有 handle”当成认证；应验证请求身份、将 handle 绑定主体，并考虑随机性与到期。
- `io.modelcontextprotocol/tasks` 是独立 opt-in 扩展，扩展页面当前标为 Draft。双方声明支持后，Server 才可按具体请求自行选择返回异步 Task handle；客户端可以查询、更新或取消扩展任务。不能把该扩展写成 MCP 核心的必有能力，也不能假设所有 SDK 已实现。
- Multi Round-Trip Requests 允许工具在一次逻辑调用中返回 `input_required`，客户端补入 input response 后再续交原调用。这表达工具需要额外输入或确认，并不自动构成高风险动作的业务授权、执行前策略检查或执行后验证。

## A2A v1.0.1 公开合同

- AgentCard 是自描述清单，可声明 agent 身份、capabilities、skills、受支持接口/协议版本和安全要求。公开 card 是发现线索，不证明发布者身份可信、能力声明真实或调用者具有资源权限。经过认证后可请求 extended Agent Card，返回范围仍由服务端访问控制决定。
- A2A 的 `Task` 保存任务状态和可选交互历史；`Message` 用于发起、澄清、状态与继续交互；任务结果应使用与 Task 关联的 `Artifact` 表达。Artifact 是对端报告的结果载体，不等于请求方已经验证结果。
- Task 状态包括 `SUBMITTED`、`WORKING`、`COMPLETED`、`FAILED`、`CANCELED`、`REJECTED`、`INPUT_REQUIRED` 与 `AUTH_REQUIRED` 等。协议提供获取、列举、取消、订阅/推送和流式交互等操作；服务端的能力声明决定相关操作是否可用。
- A2A 要求服务端按其认证要求验证每个请求。认证之后怎样按身份、请求的 skill/action、数据策略或 OAuth scope 授权，由各 agent 的实现策略决定。协议对象、task ID 或 AgentCard 本身不授予业务权限。
- `TASK_STATE_AUTH_REQUIRED` 表示任务因需要认证/授权而中断。A2A 允许 agent 将处理授权请求的责任交回 client；协议要求通过带外渠道提供凭据，除非已另行协商扩展/带内机制。该状态不是“本地已批准某项具体副作用”的凭证；调用方仍需在执行点重新确认主体、目标、动作和 scope。
- CancelTask 表示请求服务端尝试取消；成功并不保证。取消请求不撤销已经发生的外部副作用，Harness 仍需从执行器、对端状态和业务系统读取最终结果。
- A2A 将自身定位为独立 agent 之间的协作/任务接口，将 MCP 定位为 agent 与工具、API、数据源及其他资源的连接接口。A2A agent 可以在内部使用 MCP，但一个协议不会替另一协议提供其状态、权限或验证语义。

## 可支持与不可外推的集成结论

- Harness 应将发现到的工具或 agent 能力视为带版本、身份与授权上下文的外部合同快照；执行前还要执行本地 policy、调用者身份和目标资源检查。此为跨协议系统设计推论，不是两个协议共同规定的统一 Registry schema。
- 工具/Agent 的描述与结果都可能来自外部来源。系统要分开保存“远端宣称的能力”“本地允许调用的能力”“本次真实执行的动作/副作用”和“独立 verifier 的任务结果”；协议连通、schema 成功、Task completed 都不能替代最后一项。
- 统一 Trace 可以用 parent/child 或 link 表示本地运行与远程调用的关联，但应保留不同协议的 task/call ID、版本和执行方边界。它不能保证远端内部 trace 可见，也不应把输入、token 或敏感 artifact 默认复制到日志。
- Replay 和 eval 需要固定所用协议版本、client/server 实现、MCP capability/tool snapshot 或 A2A AgentCard/interface、身份与 scope、环境和结果版本，并区分固定回执模拟、snapshot/mock 与真实外部调用。协议可重放消息不等于外部世界可复现，也不表示副作用可安全重放。

## 未验证项

- 未在本地连接或运行任何 MCP host/server、A2A client/agent 或 SDK；没有真实调用、兼容性、性能、安全或用户体验结果。
- 不同协议版本、SDK、provider 和 host 的可用能力及默认行为可能不同；正文凡涉及实现行为都应绑定公开固定版本或明确标成系统设计建议。
- OpenTelemetry/trace 的通用对象与敏感内容边界见[Agent 工具调用、安全确认与评测来源](agent-tool-calling-security-2026.md)；该记录中的动态 SDK 内容不能外推为任意 trace pipeline 的保证。
