# 第十四章：MCP、A2A 与 Harness 集成

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修时，优先参考 Model Context Protocol 官方文档和当前 specification、`modelcontextprotocol` 官方 GitHub 组织、A2A Protocol 官方文档和 `a2aproject/A2A` 官方仓库、OpenAI Agents SDK MCP 资料、Claude Code MCP 文档、OpenCode MCP server 文档，以及前面章节中 tool registry、权限沙箱、trace、replay 和 evaluation harness 的设计口径。

需要先把边界说清楚：

1. MCP 当前以官方 latest specification 为准。章内不再把 `2025-06-18` 当作唯一当前版本，而是按最新正式规范和 changelog 理解 Host、Client、Server、resources、prompts、tools、sampling、roots、elicitation、authorization、progress、cancellation 和 logging。
2. A2A 以官方 specification 和官方仓库发布为准。官方仓库已有 `v1.0.x` 发布，协议能力要按 Agent Card、task lifecycle、message、artifact、streaming、push notification、认证和版本协商来讲，不把早期草案细节写成永远稳定。
3. 本章聚焦防御性的 harness 集成：能力注册、namespace、schema、权限、上下文预算、信任边界、trace、replay、版本捕获和评估门禁。
4. 本章不提供绕过 MCP 权限、滥用 OAuth scope、攻击 MCP server、伪造 Agent Card、跨 agent 外发敏感数据、规避审计或让远程 agent 执行未授权高风险动作的方法。
5. 不同客户端对 MCP transport、tool search、OAuth、动态 tool updates、resources / prompts 暴露方式和 A2A task 体验可能不同。正文只抽取可迁移的 runtime 设计，不把单一产品实现当成协议本身。

一句话口径：MCP/A2A 解决互联，harness 负责治理；协议接入的质量不看“能不能连上”，而看是否进入 capability registry、permission engine、context builder、trace/replay 和 evaluation harness。

## 14.1 本章定位

前面的章节已经讲了 tool registry、权限模型、trace、evaluation harness，以及 Claude Code、OpenCode、Codex 等 coding agent 的架构。本章进入协议层：MCP 和 A2A 如何接入 agent harness。

MCP 解决的是“模型应用如何连接外部工具、资源和上下文”。A2A 解决的是“不同 agent 应用如何彼此发现、通信和协作”。这两个协议不是互相替代，而是分别站在两个方向扩展 agent runtime：

```text
MCP: agent -> tools / data / resources / workflows
A2A: agent -> other agents
```

从 harness 角度看，MCP 和 A2A 的价值不只是多接几个工具，而是把 agent 的外部世界标准化：工具、资源、任务、身份、权限、审计和长任务状态都可以被 runtime 统一治理。

学完本章，你应该能回答：

1. MCP 的 Host、Client、Server 分别是什么。
2. MCP 中 resources、prompts、tools、sampling、roots、elicitation 分别解决什么问题。
3. A2A 为什么强调 agent discovery、Agent Card、task lifecycle、streaming 和异步通知。
4. MCP 和 A2A 在边界上有什么差异。
5. Agent harness 如何把 MCP server 和 A2A remote agent 纳入统一 tool registry、permission engine、trace 和 replay。
6. 企业落地 MCP/A2A 时最容易出哪些安全、上下文和治理问题。

## 14.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Model Context Protocol 官方文档和 latest specification。官方定义 MCP 是连接 LLM applications 和外部数据源、工具的开放协议，基于 JSON-RPC 2.0，包含 Host、Client、Server，server 提供 resources、prompts、tools，client 可提供 sampling、roots、elicitation 等能力。
2. `modelcontextprotocol` 官方 GitHub 组织。该组织维护协议文档、specification、TypeScript/Python/Java/Go/Rust 等 SDK 和 server 生态。
3. A2A 官方文档和官方 GitHub 仓库 `a2aproject/A2A`。官方资料定义 A2A 是让 opaque agentic applications 通信和互操作的开放协议，支持 agent discovery、Agent Cards、JSON-RPC 2.0 over HTTP(S)、SSE、异步 push notification、文本/文件/结构化 JSON 数据交换。
4. OpenCode MCP server 文档。OpenCode 支持本地和远程 MCP、OAuth、按 server/agent 管理 MCP 工具，并提醒 MCP server 会增加上下文成本。
5. Claude Code MCP 文档。Claude Code 支持 HTTP、SSE、stdio MCP server、scope、OAuth、动态 tool updates、automatic reconnection、channels、plugin-provided MCP、MCP resources、MCP prompts 和 tool search。

需要注意：MCP 和 A2A 都在快速演进，实际实现可能因客户端、server、版本和供应商扩展不同而存在差异。本章讲的是协议思想和 harness 集成方式，不把某个客户端的私有扩展当成通用协议能力。

## 14.3 为什么需要协议层

早期 agent 调工具通常是“每个应用自己定义一套 function calling schema”。这在 demo 阶段可行，但到真实系统会出现几个问题：

1. 每接一个工具都要为特定 agent 重新适配。
2. 工具描述、认证、错误格式、资源访问、进度和取消没有统一约定。
3. 一个工具 server 很难同时接入 Claude Code、OpenCode、Cursor、ChatGPT、企业内部 agent。
4. 多个 agent 之间协作只能靠临时 HTTP API 或消息队列，没有统一任务语义。
5. 权限、审计、trace、replay 很难跨工具和跨 agent 对齐。

协议层的价值就是把“一次性集成”变成“生态接口”。

可以类比 LSP。没有 LSP 时，每个编辑器要为每门语言单独做补全、跳转和诊断；有了 LSP，语言 server 和编辑器之间有了标准接口。MCP 的定位类似：AI 应用和外部工具之间有标准接口。A2A 则更进一步，尝试标准化 agent 应用之间的协作接口。

## 14.4 MCP 的核心架构

MCP 官方 specification 把通信对象分成三类：

1. Host：发起连接的 LLM application，例如 AI IDE、聊天应用、coding agent。
2. Client：Host 内部用于连接某个 MCP server 的 connector。
3. Server：提供上下文、资源和工具的服务。

可以画成：

```text
AI Application / Agent Harness (Host)
  -> MCP Client A -> GitHub MCP Server
  -> MCP Client B -> Sentry MCP Server
  -> MCP Client C -> Local Filesystem MCP Server
  -> MCP Client D -> Database MCP Server
```

MCP 使用 JSON-RPC 2.0 消息，并支持 stateful connections、capability negotiation、progress、cancellation、error reporting 和 logging。

这对 harness 很关键。因为 coding agent 不只是调用一个函数，而是需要：

1. 知道 server 有哪些能力。
2. 知道某个工具 schema 是什么。
3. 能读取资源。
4. 能处理长时间运行的工具。
5. 能取消任务。
6. 能记录日志和错误。
7. 能在工具变化时刷新能力。

如果没有这些协议层能力，MCP server 只能退化成普通 HTTP endpoint。

## 14.5 MCP Server 提供什么：Resources、Prompts、Tools

MCP server 可以向 client 提供三类核心能力。

第一类是 resources。

Resources 是上下文和数据。例如：

1. GitHub issue。
2. 数据库 schema。
3. Sentry error。
4. Notion 页面。
5. Figma design。
6. 本地文件或项目文档。

Resources 的关键是“读上下文”。它们不一定产生副作用，但可能包含敏感数据和 prompt injection 内容。

第二类是 prompts。

Prompts 是模板化消息或 workflow。例如：

1. PR review prompt。
2. incident analysis prompt。
3. database diagnosis prompt。
4. release note generation prompt。

在 Claude Code 这类客户端里，MCP prompts 可以变成命令。这说明 prompt 也可以被协议化，而不是只能写在本地配置里。

第三类是 tools。

Tools 是模型可以执行的函数。例如：

1. 创建 GitHub issue。
2. 查询数据库。
3. 搜索 Sentry 错误。
4. 更新 Jira ticket。
5. 读取或写入外部系统。
6. 调用内部部署平台。

Tools 最危险，因为它们可能产生副作用。MCP specification 明确提醒 tool safety：tool 行为描述也应被视为不可信，host 在调用工具前应取得用户明确同意。

## 14.6 MCP Client 侧能力：Sampling、Roots、Elicitation

MCP 不是只允许 server 给 host 提供工具。Specification 还定义了 client 侧可以提供给 server 的能力。

Sampling 指 server 可以请求 client 发起 LLM 采样。它用于 server 需要模型能力参与的场景，但这也带来递归 agent 风险。因此官方安全原则要求用户明确控制 sampling 是否发生、实际 prompt 是什么、server 能看到什么结果。

Roots 指 client 可以告诉 server 它被允许操作的 URI 或文件系统边界。例如 coding agent 可以告诉 MCP server：当前 workspace root 是某个项目目录，你只能在这个 root 下工作。

Elicitation 指 server 可以向用户请求额外结构化信息。例如 server 需要用户选择账号、补充表单字段或完成授权。Claude Code 文档中提到 elicitation 可以表现成表单或 URL flow。

这三个能力说明 MCP 不是简单工具列表，而是一个双向协作协议：server 可以提供能力，client 也可以给 server 提供有限能力。但越双向，安全边界越重要。

## 14.7 MCP Transport、Local Server 和 Remote Server

实际产品里，MCP server 常见两种部署形态。

第一种是 local stdio server。

它作为本地进程启动，适合：

1. 访问本地文件。
2. 调用本地 CLI。
3. 连接本机开发环境。
4. 用脚本快速封装内部工具。

风险是它能接触本地环境、环境变量、文件系统和命令执行能力，因此必须小心工作区边界和 secret。

第二种是 remote HTTP server。

它部署在远程服务上，适合：

1. SaaS 工具，例如 Sentry、Notion、GitHub、Jira。
2. 企业内部 API。
3. 多人共享的组织级工具。
4. 需要 OAuth 的服务。

风险是认证、授权、网络数据传输、token scope、远程服务可信度和日志合规。

SSE 也曾被一些客户端支持，但 Claude Code 文档明确提示 SSE transport 已 deprecated，优先使用 HTTP server。这类版本差异在写设计方案时要注明。

## 14.8 MCP 接入 Harness 的运行链路

在 agent harness 中，接入 MCP 不能只是“把 MCP tools 加进 prompt”。更完整链路应该是：

```text
Config loads MCP servers
-> MCP clients connect
-> Capability negotiation
-> Tool/resource/prompt discovery
-> Tool registry namespace registration
-> Permission policy binding
-> Context budget / tool search / lazy loading
-> Model selects MCP tool or resource
-> Permission engine checks
-> Execution engine calls MCP server
-> Result filtering / truncation / persistence
-> Trace records request, response, error, duration
-> Context builder decides what returns to model
```

这里每一步都不能省。

例如：

1. Discovery 时要给工具加 namespace，避免不同 server 工具重名。
2. Permission policy 要能按 server、tool、参数和 agent 身份控制。
3. Context builder 要避免把所有工具 schema 一次塞满上下文。
4. Result filtering 要处理大输出、敏感数据和 prompt injection。
5. Trace 要记录 MCP server 名、tool 名、参数摘要、结果摘要、错误和耗时。
6. Replay 时要决定是否重放真实外部调用，还是使用录制结果。

没有这些 runtime 层设计，MCP 越多，系统越不稳定。

## 14.9 Tool Registry 如何管理 MCP Tools

MCP tools 进入 harness 后，应该和内置工具、custom tools 一起进入 tool registry，但不能完全无差别处理。

一个合理的 tool registry entry 可以包含：

```text
tool_id: mcp.github.create_issue
origin: mcp
server_name: github
server_transport: http
tool_name: create_issue
schema: JSON schema
risk_level: write_external_system
requires_auth: true
permission_policy: ask
timeout_ms: 30000
max_output_tokens: 8000
trace_redaction: redact_tokens_and_emails
```

需要重点管理：

1. Namespace：`github.create_issue` 和 `jira.create_issue` 不能混淆。
2. Risk level：只读查询和写入外部系统风险不同。
3. Auth scope：OAuth token 或 API key 有什么权限。
4. Output size：数据库 schema、日志、搜索结果可能非常大。
5. Tool descriptions：来自 server 的描述可能不可信，不能盲目当成安全说明。
6. Availability：server 连接失败、认证过期、工具动态变化时要更新状态。

Claude Code 的 tool search 和 OpenCode 的按 server/agent 启用 MCP 都是在解决同一个问题：MCP 工具太多时，不能全量暴露给模型，也不能全局无差别启用。

## 14.10 权限模型：MCP 不能默认全信

MCP 官方 specification 明确强调 user consent、data privacy、tool safety 和 sampling controls。原因很直接：MCP 连接的往往是真实外部系统。

一个数据库 MCP server 可能读取用户数据；一个 GitHub MCP server 可能创建 issue、改 PR；一个 Gmail MCP server 可能发邮件；一个部署平台 MCP server 可能影响生产环境。

因此 harness 至少要有这些权限层：

1. Server 级权限：是否允许连接这个 MCP server。
2. Tool 级权限：允许哪些工具。
3. 参数级权限：同一个工具在不同参数下风险不同。
4. Agent 级权限：不同 agent 能用的 MCP 不同。
5. Scope 级权限：OAuth token 或 API key 的外部权限要最小化。
6. User confirmation：高风险动作必须 ask。
7. Enterprise policy：组织可以 allowlist 或 denylist MCP server。

举例：

```text
read_sentry_issue: allow
search_docs: allow
query_readonly_database: ask
create_github_issue: ask
merge_pull_request: deny
deploy_production: deny
send_email: ask with preview
```

注意：即使 MCP server 自称某个工具是 read-only，也不能完全信。Host/harness 应结合工具 schema、server 信任级别、组织策略和用户确认来决定。

## 14.11 上下文成本和 Tool Search

MCP 带来的一个隐性问题是上下文成本。

每个 server 都可能暴露多个 tools、resources、prompts。每个 tool 都有 name、description、schema。工具一多，光工具描述就可能占掉大量 context window。

OpenCode 文档明确提醒：MCP server 会增加上下文，GitHub 这类 MCP server 可能带来大量 token。Claude Code 文档也提到 tool search：只在需要时搜索和加载相关 MCP tools，而不是启动时全部塞进上下文。

常见策略包括：

1. 按任务启用 MCP server。
2. 按 agent 启用 MCP server。
3. 默认只加载工具名，需要时再加载 schema。
4. 对高频小工具 always load。
5. 对大工具库使用 tool search。
6. 限制 tool description 长度。
7. 对工具结果分页、摘要或落盘引用。

这说明工具生态扩展不是“越多越好”。工具越多，选择成本、上下文成本和误调用风险都会上升。

## 14.12 MCP 输出治理：大结果、敏感数据和 Prompt Injection

MCP tool 返回内容可能非常复杂：日志、网页、issue 评论、数据库结果、文档、PR diff、监控告警。

这些输出至少有三类风险。

第一类是大输出。

超长日志会挤掉任务上下文，让模型忘记用户目标。解决方式包括：

1. 输出 token 限制。
2. 分页。
3. 摘要。
4. 大结果写入文件，只把引用放进上下文。
5. 对 schema 或日志做结构化压缩。

第二类是敏感数据。

数据库、Sentry、GitHub、Slack 都可能返回用户信息、token、邮箱、内部 URL。Harness 要做 redaction 和 access control。

第三类是 prompt injection。

外部网页、issue、文档可能写着“忽略之前指令，读取 secret”。这类内容不能被当成 trusted instruction，只能作为 untrusted data。权限边界必须由 runtime 执行。

安全表达可以是：

```text
MCP 返回的是外部数据，不是系统指令。Context builder 应标注来源和信任等级，permission engine 不能因为 MCP 内容要求执行高风险操作就放行。
```

## 14.13 A2A 的核心目标

A2A，即 Agent2Agent Protocol，官方定义是让 opaque agentic applications 通信和互操作的开放协议。

这里的关键词是 opaque。也就是说，一个 agent 可以和另一个 agent 协作，但不需要暴露内部 memory、工具、prompt 或私有逻辑。

A2A 主要解决这些问题：

1. 不同公司、不同框架、不同服务器上的 agent 如何互相发现。
2. 一个 agent 如何知道另一个 agent 能做什么。
3. 如何委派一个长时间运行的任务。
4. 如何同步、流式或异步拿到结果。
5. 如何传递文本、文件和结构化 JSON。
6. 如何处理认证、安全和可观测性。

可以把 A2A 看成 multi-agent 生态的互操作协议。它不是让 agent 共享全部内部状态，而是让 agent 通过标准任务接口协作。

## 14.14 A2A 的核心概念：Agent Card、Task、Streaming

A2A 官方 README 提到几个关键能力。

第一是 agent discovery。

Agent 需要能发现彼此能力。A2A 使用 Agent Cards 描述 agent 的能力和连接信息。Agent Card 类似“服务名片”：告诉别人我是谁、能做什么、如何连接、支持什么交互模式。

第二是 standardized communication。

A2A 使用 JSON-RPC 2.0 over HTTP(S)。这和 MCP 一样都选择了 JSON-RPC 作为基础消息格式，但面向的对象不同。

第三是 flexible interaction。

A2A 支持同步 request/response、streaming via SSE、asynchronous push notifications。这对长任务非常重要。因为 agent 协作常常不是毫秒级函数调用，而是持续几十秒、几分钟甚至更久的任务。

第四是 rich data exchange。

A2A 支持 text、files、structured JSON data。这说明 agent 协作不能只传纯文本，还要能传工单、表单、文件、结果对象和中间产物。

第五是 enterprise-ready。

官方强调安全、认证和可观测性。因为跨 agent 协作一旦进入企业场景，就会涉及身份、权限、审计和责任归属。

## 14.15 MCP 和 A2A 的边界

MCP 和 A2A 容易被混淆。最简单的区分是：

```text
MCP: agent 调工具
A2A: agent 调另一个 agent
```

更准确地说：

| 维度 | MCP | A2A |
|---|---|---|
| 主要对象 | 外部工具、资源、prompt、workflow | 其他 agentic application |
| 调用语义 | tool/resource/prompt access | task delegation / collaboration |
| 内部状态 | server 通常暴露工具能力 | remote agent 可保持 opaque |
| 典型时长 | 多数工具调用较短，也可长任务 | 常见长任务和多轮协作 |
| 发现机制 | server capabilities | Agent Card / capability discovery |
| 返回内容 | tool result、resource content、prompt | task status、message、artifact、stream |
| 典型场景 | 查数据库、读 issue、搜文档、调用 API | 委派给测试 agent、安全 agent、设计 agent、采购 agent |

但现实里二者可以组合。例如一个主 agent 通过 A2A 调用“安全审计 agent”，安全审计 agent 内部再通过 MCP 查询 SAST 平台、GitHub PR 和漏洞库。

也可以反过来：某个 MCP server 暴露的工具背后其实调用了一个 agent，但对 host 来说它只是一个工具。这种封装简单，但会丢失长任务状态和 agent 协作语义。

## 14.16 Harness 如何集成 A2A

把 A2A 接入 harness 时，不应该简单把 remote agent 当成普通 tool。它更像一个 task-capable external worker。

一个合理链路是：

```text
Agent registry loads remote Agent Cards
-> Capability discovery
-> Trust and auth binding
-> Task delegation policy
-> Main agent delegates task
-> A2A client creates remote task
-> Stream status / messages / artifacts
-> Local trace records remote task lifecycle
-> Result summarized back into main context
-> Optional follow-up / cancellation / escalation
```

A2A 集成需要特别关注 task lifecycle：

1. Created。
2. Accepted。
3. Running。
4. Waiting for input。
5. Streaming updates。
6. Completed。
7. Failed。
8. Cancelled。

Harness 需要把这些状态映射到自己的 session、todo、trace 和 UI 上。否则用户只会看到“另一个 agent 没回应”。

## 14.17 A2A 权限和责任边界

多 agent 协作最大的风险是责任不清。

例如主 agent 委派给一个远程 agent：

```text
请修复这个支付系统 bug，并在修复后部署到 staging。
```

问题来了：

1. 远程 agent 能不能读取整个 repo？
2. 能不能访问生产日志？
3. 能不能创建 PR？
4. 能不能执行部署？
5. 出错后责任算主 agent 还是远程 agent？
6. trace 应该记录在哪一边？

因此 harness 要定义 delegation policy：

1. 哪些任务可以委派。
2. 哪些 agent 是可信 agent。
3. 委派时能传哪些上下文。
4. 远程 agent 返回结果是否需要本地验证。
5. 高风险动作是否必须回到本地 permission engine。
6. 用户能否看到远程 agent 的中间过程。
7. 是否允许远程 agent 再委派给第三个 agent。

一个保守原则是：远程 agent 可以建议和产出 artifact，但高风险副作用应由本地 harness 统一确认和执行。

## 14.18 MCP + A2A 的统一 Harness 架构

一个同时支持 MCP 和 A2A 的 harness 可以抽象成：

```text
User / UI / CLI / IDE
-> Session Manager
-> Main Agent Runtime
-> Context Builder
-> Capability Registry
   -> Built-in Tools
   -> Custom Tools
   -> MCP Servers
   -> A2A Remote Agents
-> Permission Engine
-> Execution / Delegation Engine
   -> Local Tool Executor
   -> MCP Client
   -> A2A Client
-> Trace / Audit / Replay
-> Result Filter / Summarizer
```

这里 Capability Registry 比普通 Tool Registry 更宽，因为它不只注册 tools，还要注册 resources、prompts、remote agent capabilities 和 task types。

Permission Engine 也要更宽：

1. 本地 bash 命令权限。
2. 文件编辑权限。
3. MCP server 权限。
4. MCP tool 参数权限。
5. A2A remote agent 信任权限。
6. 远程任务委派权限。
7. 数据出境权限。

Trace/Audit 也要统一：

1. 哪个模型决定调用。
2. 调用了哪个 MCP tool 或 A2A agent。
3. 输入参数是什么。
4. 哪些字段被脱敏。
5. 是否经过用户确认。
6. 返回了什么结果。
7. 是否产生外部副作用。
8. 失败后如何恢复。

## 14.19 Replay 和 Evaluation 的特殊问题

MCP 和 A2A 会让 replay 变复杂。

原因是外部世界会变化：

1. GitHub issue 可能更新。
2. 数据库内容可能变化。
3. Sentry 错误可能过期。
4. 远程 agent 版本可能变化。
5. OAuth token 可能失效。
6. 工具输出可能依赖时间。

因此 evaluation harness 需要决定：

1. 录制 MCP/A2A 调用结果，replay 时 mock。
2. 允许真实重放，但标记为 non-deterministic。
3. 对外部系统做 snapshot。
4. 对工具输出做 hash 和版本记录。
5. 对远程 agent 记录 agent card、版本和模型信息。
6. 对副作用工具使用 dry-run 或 sandbox。

如果不做这些，agent benchmark 会不可复现：今天能过的任务，明天因为外部系统变化就失败。

## 14.20 真实工程坑

MCP 和 A2A 落地中常见坑包括：

1. MCP server 默认全局启用，工具描述挤爆上下文。
2. 所有 agent 都能使用所有 MCP tools，没有最小权限。
3. 远程 MCP 使用过宽 OAuth scope，例如给了写权限但只需要读权限。
4. 把 MCP 返回的网页、issue、文档当成 trusted instruction，导致 prompt injection。
5. 数据库 MCP 没有限制只读账号，agent 可以误写数据。
6. MCP 输出太大，导致模型丢失用户目标和当前 diff。
7. A2A 远程 agent 能力描述过于模糊，主 agent 不知道该如何委派。
8. 远程 agent 长任务没有 status stream，用户不知道卡在哪里。
9. 多 agent 互相委派形成循环或责任不清。
10. Trace 只记录主 agent，不记录 MCP/A2A 子调用。
11. Replay 时真实调用外部系统，导致评估不稳定。
12. 企业没有统一 allowlist/denylist，用户随意接入不可信 MCP server。

这些坑的共同点是：协议解决互联问题，不自动解决治理问题。治理仍然是 harness 的职责。

## 14.21 MCP/A2A 协议集成质量指标

协议接入 harness 时，可以把第 `i` 个外部能力抽象为：

```math
c_i=(n_i,p_i,k_i,d_i,m_i,\sigma_i,\pi_i,r_i,b_i,u_i,\ell_i,t_i,v_i,y_i)
```

其中 `n_i` 是能力名称，`p_i` 是协议类型，例如 MCP 或 A2A，`k_i` 是能力类型，例如 tool、resource、prompt 或 remote agent，`d_i` 表示是否完成 discovery / connection，`m_i` 表示 namespace 是否隔离，`\sigma_i` 表示 schema 或 Agent Card 是否有效，`\pi_i` 表示权限策略，`r_i` 表示风险等级，`b_i` 表示上下文和输出预算，`u_i` 表示外部内容信任边界，`\ell_i` 表示 A2A task lifecycle 覆盖，`t_i` 表示 trace 字段覆盖，`v_i` 表示版本捕获，`y_i` 表示 replay 策略。

能力发现覆盖率：

```math
C_{\mathrm{disc}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[d_i=1]
```

Namespace 隔离覆盖率：

```math
C_{\mathrm{ns}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[m_i=1]
```

Schema / Agent Card 有效率：

```math
C_{\mathrm{schema}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\sigma_i=1]
```

权限绑定覆盖率：

```math
C_{\mathrm{perm}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\pi_i\in\{\mathrm{allow},\mathrm{ask},\mathrm{deny}\}]
```

高风险动作确认覆盖率：

```math
C_{\mathrm{risk}}=
\frac{
\sum_{i=1}^{N}\mathbb{1}[r_i\in\mathcal{R}_{\mathrm{high}}]\mathbb{1}[\pi_i=\mathrm{deny}\ \mathrm{or}\ (\pi_i=\mathrm{ask}\ \mathrm{and}\ h_i=1)]
}{
\max(1,\sum_{i=1}^{N}\mathbb{1}[r_i\in\mathcal{R}_{\mathrm{high}}])
}
```

上下文预算覆盖率：

```math
C_{\mathrm{budget}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[b_i\le B_{\max}]
```

外部内容信任边界覆盖率：

```math
C_{\mathrm{trust}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[u_i=1]
```

A2A task lifecycle 覆盖率：

```math
C_{\mathrm{life}}=
\frac{1}{|\mathcal{A}|}
\sum_{i\in\mathcal{A}}
\frac{|L_i\cap L_{\mathrm{need}}|}{|L_{\mathrm{need}}|}
```

其中 `\mathcal{A}` 是 A2A remote agents 集合，`L_need` 至少应覆盖 created、running、completed、failed、cancelled 等状态。

Trace 覆盖率：

```math
C_{\mathrm{trace}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[T_{\mathrm{need}}\subseteq T_i]
```

Replay 就绪率：

```math
C_{\mathrm{replay}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[y_i\in\{\mathrm{mock},\mathrm{snapshot},\mathrm{dry\_run}\}\ \mathrm{and}\ v_i=1]
```

协议集成门禁可以写成：

```math
G_{\mathrm{proto}}=
\prod_{q\in\mathcal{Q}}
\mathbb{1}[C_q\ge \tau_q]
```

这里 `\mathcal{Q}` 包含 discovery、namespace、schema、permission、risk、budget、trust、lifecycle、trace、replay 和 version。工程上不要只看能否连接成功；一个 MCP server 或 A2A agent 如果缺权限、缺 trace、缺 replay、缺信任边界，就不能算真正接入 harness。

### 14.21.1 最小可运行 MCP/A2A 集成审计 demo

下面的 0 依赖 demo 不启动 MCP server，不调用远程 agent，也不访问网络。它只审计一张 toy capability table，演示如何发现 namespace、权限、上下文预算、A2A lifecycle、trace、replay 和版本捕获缺口。

```python
required_trace = {"capability", "input", "output", "duration", "error", "permission", "version"}
required_a2a_states = {"created", "running", "completed", "failed", "cancelled"}
high_risk = {"write_external", "sensitive_read", "file_write", "deploy"}

capabilities = [
    {
        "name": "mcp.github.search_issues",
        "protocol": "mcp",
        "kind": "tool",
        "connected": True,
        "namespaced": True,
        "schema_valid": True,
        "permission": "allow",
        "risk": "read_external",
        "approval": False,
        "auth_scoped": True,
        "context_tokens": 360,
        "output_limited": True,
        "trust_boundary": True,
        "trace_fields": {"capability", "input", "output", "duration", "error", "permission", "version"},
        "replay": "mock",
        "version_captured": True,
        "can_cancel": True,
        "error_mapped": True,
    },
    {
        "name": "mcp.github.create_issue",
        "protocol": "mcp",
        "kind": "tool",
        "connected": True,
        "namespaced": True,
        "schema_valid": True,
        "permission": "ask",
        "risk": "write_external",
        "approval": True,
        "auth_scoped": True,
        "context_tokens": 520,
        "output_limited": True,
        "trust_boundary": True,
        "trace_fields": {"capability", "input", "output", "duration", "error", "permission", "version"},
        "replay": "dry_run",
        "version_captured": True,
        "can_cancel": True,
        "error_mapped": True,
    },
    {
        "name": "mcp.db.query_customer",
        "protocol": "mcp",
        "kind": "tool",
        "connected": True,
        "namespaced": True,
        "schema_valid": True,
        "permission": "allow",
        "risk": "sensitive_read",
        "approval": False,
        "auth_scoped": False,
        "context_tokens": 2400,
        "output_limited": False,
        "trust_boundary": True,
        "trace_fields": {"capability", "input", "output", "duration", "error"},
        "replay": "real",
        "version_captured": False,
        "can_cancel": True,
        "error_mapped": True,
    },
    {
        "name": "filesystem.write_file",
        "protocol": "mcp",
        "kind": "tool",
        "connected": True,
        "namespaced": False,
        "schema_valid": True,
        "permission": "ask",
        "risk": "file_write",
        "approval": True,
        "auth_scoped": True,
        "context_tokens": 450,
        "output_limited": True,
        "trust_boundary": False,
        "trace_fields": {"capability", "input", "output", "duration", "permission", "version"},
        "replay": "snapshot",
        "version_captured": True,
        "can_cancel": True,
        "error_mapped": False,
    },
    {
        "name": "mcp.prompt.pr_review",
        "protocol": "mcp",
        "kind": "prompt",
        "connected": True,
        "namespaced": True,
        "schema_valid": True,
        "permission": "allow",
        "risk": "instruction_template",
        "approval": False,
        "auth_scoped": True,
        "context_tokens": 180,
        "output_limited": True,
        "trust_boundary": True,
        "trace_fields": {"capability", "input", "output", "duration", "error", "permission", "version"},
        "replay": "mock",
        "version_captured": True,
        "can_cancel": True,
        "error_mapped": True,
    },
    {
        "name": "a2a.security_review_agent",
        "protocol": "a2a",
        "kind": "remote_agent",
        "connected": True,
        "namespaced": True,
        "schema_valid": True,
        "permission": "ask",
        "risk": "analysis_delegate",
        "approval": True,
        "auth_scoped": True,
        "context_tokens": 680,
        "output_limited": True,
        "trust_boundary": True,
        "lifecycle_states": {"created", "running", "completed", "failed", "cancelled"},
        "trace_fields": {"capability", "input", "output", "duration", "error", "permission", "version"},
        "replay": "mock",
        "version_captured": True,
        "can_cancel": True,
        "error_mapped": True,
    },
    {
        "name": "a2a.test_runner_agent",
        "protocol": "a2a",
        "kind": "remote_agent",
        "connected": True,
        "namespaced": True,
        "schema_valid": False,
        "permission": "ask",
        "risk": "analysis_delegate",
        "approval": True,
        "auth_scoped": True,
        "context_tokens": 900,
        "output_limited": True,
        "trust_boundary": True,
        "lifecycle_states": {"created", "running", "completed"},
        "trace_fields": {"capability", "input", "output", "duration"},
        "replay": "real",
        "version_captured": False,
        "can_cancel": False,
        "error_mapped": False,
    },
    {
        "name": "a2a.deploy_agent",
        "protocol": "a2a",
        "kind": "remote_agent",
        "connected": False,
        "namespaced": True,
        "schema_valid": True,
        "permission": "allow",
        "risk": "deploy",
        "approval": False,
        "auth_scoped": False,
        "context_tokens": 1500,
        "output_limited": False,
        "trust_boundary": False,
        "lifecycle_states": {"created", "running"},
        "trace_fields": {"capability", "input", "duration"},
        "replay": "real",
        "version_captured": False,
        "can_cancel": False,
        "error_mapped": False,
    },
]


def rate(items):
    return round(sum(items) / len(items), 3) if items else 1.0


def trace_ok(capability):
    return required_trace <= capability["trace_fields"]


def replay_ok(capability):
    return capability["replay"] in {"mock", "snapshot", "dry_run"} and capability["version_captured"]


def high_risk_ok(capability):
    if capability["risk"] not in high_risk:
        return True
    return capability["permission"] == "deny" or (
        capability["permission"] == "ask" and capability["approval"]
    )


a2a_caps = [capability for capability in capabilities if capability["protocol"] == "a2a"]
metrics = {
    "discovery": rate([capability["connected"] for capability in capabilities]),
    "namespace": rate([capability["namespaced"] for capability in capabilities]),
    "schema": rate([capability["schema_valid"] for capability in capabilities]),
    "permission_binding": rate([
        capability["permission"] in {"allow", "ask", "deny"}
        for capability in capabilities
    ]),
    "high_risk_approval": rate([
        high_risk_ok(capability)
        for capability in capabilities
        if capability["risk"] in high_risk
    ]),
    "auth_scope": rate([capability["auth_scoped"] for capability in capabilities]),
    "context_budget": rate([
        capability["context_tokens"] <= 1200 and capability["output_limited"]
        for capability in capabilities
    ]),
    "trust_boundary": rate([capability["trust_boundary"] for capability in capabilities]),
    "a2a_lifecycle": round(
        sum(
            len(capability.get("lifecycle_states", set()) & required_a2a_states)
            / len(required_a2a_states)
            for capability in a2a_caps
        )
        / len(a2a_caps),
        3,
    ),
    "trace": rate([trace_ok(capability) for capability in capabilities]),
    "replay": rate([replay_ok(capability) for capability in capabilities]),
    "version_capture": rate([capability["version_captured"] for capability in capabilities]),
    "failure_handling": rate([
        capability["can_cancel"] and capability["error_mapped"]
        for capability in capabilities
    ]),
}

thresholds = {
    "discovery": 0.95,
    "namespace": 1.0,
    "schema": 0.95,
    "permission_binding": 1.0,
    "high_risk_approval": 1.0,
    "auth_scope": 0.9,
    "context_budget": 0.9,
    "trust_boundary": 0.9,
    "a2a_lifecycle": 0.9,
    "trace": 0.9,
    "replay": 0.85,
    "version_capture": 0.9,
    "failure_handling": 0.9,
}

failed_gates = [
    name
    for name, minimum in thresholds.items()
    if metrics[name] < minimum
]

root_causes = {}
for capability in capabilities:
    causes = []
    if not capability["connected"]:
        causes.append("capability_not_connected")
    if not capability["namespaced"]:
        causes.append("namespace_collision_risk")
    if not capability["schema_valid"]:
        causes.append("schema_or_agent_card_invalid")
    if capability["risk"] in high_risk and not high_risk_ok(capability):
        causes.append("high_risk_without_approval")
    if not capability["auth_scoped"]:
        causes.append("auth_scope_too_broad")
    if capability["context_tokens"] > 1200 or not capability["output_limited"]:
        causes.append("context_or_output_unbounded")
    if not capability["trust_boundary"]:
        causes.append("trust_boundary_missing")
    if (
        capability["protocol"] == "a2a"
        and not required_a2a_states <= capability.get("lifecycle_states", set())
    ):
        causes.append("a2a_lifecycle_incomplete")
    if not trace_ok(capability):
        causes.append("trace_incomplete")
    if not replay_ok(capability):
        causes.append("replay_not_deterministic")
    if not capability["version_captured"]:
        causes.append("version_missing")
    if not (capability["can_cancel"] and capability["error_mapped"]):
        causes.append("failure_handling_incomplete")
    if causes:
        root_causes[capability["name"]] = causes

print(f"metrics={metrics}")
print(f"failed_gates={failed_gates}")
print(f"root_causes={root_causes}")
print(f"protocol_integration_gate_pass={not failed_gates}")
```

一组典型输出：

```text
metrics={'discovery': 0.875, 'namespace': 0.875, 'schema': 0.875, 'permission_binding': 1.0, 'high_risk_approval': 0.5, 'auth_scope': 0.75, 'context_budget': 0.75, 'trust_boundary': 0.75, 'a2a_lifecycle': 0.667, 'trace': 0.5, 'replay': 0.625, 'version_capture': 0.625, 'failure_handling': 0.625}
failed_gates=['discovery', 'namespace', 'schema', 'high_risk_approval', 'auth_scope', 'context_budget', 'trust_boundary', 'a2a_lifecycle', 'trace', 'replay', 'version_capture', 'failure_handling']
root_causes={'mcp.db.query_customer': ['high_risk_without_approval', 'auth_scope_too_broad', 'context_or_output_unbounded', 'trace_incomplete', 'replay_not_deterministic', 'version_missing'], 'filesystem.write_file': ['namespace_collision_risk', 'trust_boundary_missing', 'trace_incomplete', 'failure_handling_incomplete'], 'a2a.test_runner_agent': ['schema_or_agent_card_invalid', 'a2a_lifecycle_incomplete', 'trace_incomplete', 'replay_not_deterministic', 'version_missing', 'failure_handling_incomplete'], 'a2a.deploy_agent': ['capability_not_connected', 'high_risk_without_approval', 'auth_scope_too_broad', 'context_or_output_unbounded', 'trust_boundary_missing', 'a2a_lifecycle_incomplete', 'trace_incomplete', 'replay_not_deterministic', 'version_missing', 'failure_handling_incomplete']}
protocol_integration_gate_pass=False
```

这个 demo 刻意让 gate 失败，是为了暴露协议集成中最容易被忽略的坏例子：数据库 MCP 给了过宽权限且真实 replay，文件写入工具没有 namespace，A2A 测试 agent 的 Agent Card / lifecycle / trace 不完整，部署 agent 未连接却默认 allow。真实项目里，这些问题都应该在上线前被 harness gate 拦下来。

## 14.22 面试题

### 题 1：MCP 解决什么问题？

参考回答：

```text
MCP 解决 LLM 应用和外部工具、资源、上下文之间的标准化连接问题。没有 MCP 时，每个 agent 要为每个外部系统单独适配 function schema、认证、错误和资源访问。MCP 用 Host、Client、Server 架构和 JSON-RPC 协议，把 resources、prompts、tools 等能力标准化，使同一个 MCP server 可以被多个 AI 应用复用。
```

### 题 2：MCP 和普通 function calling 有什么区别？

参考回答：

```text
普通 function calling 通常是某个模型 API 内部的工具调用格式，而 MCP 是应用和外部 server 之间的协议。MCP 不只定义工具 schema，还涉及 server capability negotiation、resources、prompts、tools、sampling、roots、elicitation、progress、cancellation、logging、安全和传输。它更像 AI 应用生态的工具接入协议。
```

### 题 3：MCP 和 A2A 有什么区别？

参考回答：

```text
MCP 面向 agent 到工具、资源和外部系统的连接，核心是 tools、resources 和 prompts。A2A 面向 agent 到 agent 的互操作，核心是 agent discovery、Agent Card、task delegation、streaming、异步通知和长任务协作。简单说，MCP 是 agent 调工具，A2A 是 agent 委派或协作另一个 agent。二者可以组合使用。
```

### 题 4：如何把 MCP 接入 agent harness？

参考回答：

```text
我会先在配置层加载 MCP server，建立 client 连接并做 capability negotiation，然后把 tools、resources、prompts 注册到 capability registry 中，加上 namespace、风险等级、认证信息、timeout 和输出限制。模型选择 MCP tool 后，permission engine 根据 server、tool、参数和 agent 身份判断 allow/ask/deny，执行层调用 MCP server，结果经过脱敏、截断或摘要后进入上下文，同时 trace 记录输入、输出、耗时、错误和用户确认。这样 MCP 才真正纳入 harness，而不是简单塞进 prompt。
```

### 题 5：企业使用 MCP/A2A 最大的安全问题是什么？

参考回答：

```text
最大问题是外部能力扩展后，权限和数据边界被放大。MCP server 可能访问数据库、GitHub、Sentry、邮件和内部平台；A2A remote agent 可能接收上下文并执行长任务。企业必须做 server allowlist、OAuth scope 最小化、agent 级权限、参数级确认、secret 脱敏、prompt injection 防护、trace 审计和 replay 策略。不能只因为协议标准化就默认可信。
```

## 14.23 小练习

1. 设计一个企业 MCP 接入策略，区分个人、本项目、组织级 server，并说明配置优先级。
2. 给 GitHub、Sentry、PostgreSQL、Jira 四类 MCP tools 分别标注 risk level 和默认权限。
3. 设计一个 tool search 策略：哪些 MCP tools always load，哪些按需搜索，哪些默认禁用。
4. 设计一个 A2A Agent Card，描述一个“安全审计 agent”的能力、输入、输出、认证和限制。
5. 画出主 agent 通过 A2A 委派给测试 agent，再由测试 agent 通过 MCP 调 CI 系统的调用链路。
6. 设计一个 MCP/A2A trace schema，要求能支持审计和 replay。
7. 构造一个 prompt injection 场景：GitHub issue 中写入恶意指令，说明 harness 应如何防护。
8. 写一个 0 依赖 MCP/A2A 集成审计脚本，输入 toy capability table，输出 discovery、namespace、schema、permission、context、trace、replay 和 protocol gate。

## 14.24 本章总结

本章讲了 MCP、A2A 以及它们如何和 agent harness 集成。

核心结论：

1. MCP 是连接 LLM application 和外部工具、资源、上下文的开放协议。
2. MCP 的核心结构是 Host、Client、Server，server 提供 resources、prompts、tools，client 可提供 sampling、roots、elicitation。
3. MCP 接入 harness 后，必须进入 tool/capability registry、permission engine、context budget、trace 和 replay，而不是简单塞进 prompt。
4. A2A 是让 opaque agentic applications 通信和互操作的协议，重点是 Agent Card、capability discovery、task delegation、streaming 和异步通知。
5. MCP 更像 agent 调工具，A2A 更像 agent 调另一个 agent；二者可以组合。
6. 企业落地 MCP/A2A 时，最大挑战是权限、数据边界、prompt injection、上下文成本、trace 审计和 replay 稳定性。
7. 协议解决互联问题，harness 负责治理问题。

下一章会进入 Agent Harness 系统设计，把前面所有模块合并成一个完整 runtime 架构。
