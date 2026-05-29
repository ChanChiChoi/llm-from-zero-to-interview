# 第十五章：Agent Harness 系统设计

## 15.1 本章定位

前面十四章分别讲了 harness 总览、agent runtime、coding agent 工作流、工具系统、文件编辑、终端执行、上下文管理、权限沙箱、trace/replay、evaluation harness、Claude Code、OpenCode、Codex 对比、MCP/A2A 集成。本章把这些内容合成一道系统设计题：

```text
设计一个 coding agent harness。
```

这道题不是让你设计一个“调用大模型的聊天机器人”。它考察的是你能不能把模型、工具、文件系统、终端、权限、安全、上下文、trace、replay、评估、MCP/A2A 和用户交互组织成一个可运行、可控、可观测、可扩展的软件系统。

学完本章，你应该能回答：

1. 面试中如何澄清 coding agent harness 的需求。
2. 一个生产级 harness 的核心模块有哪些。
3. 用户输入到 agent 完成任务的完整链路是什么。
4. 如何设计 session、message、tool call、permission、diff、trace 等数据模型。
5. 如何处理上下文窗口、工具选择、权限确认、沙箱执行和文件回滚。
6. 如何支持 evaluation、replay、MCP、A2A、server/SDK 和企业治理。
7. 这类系统最关键的 trade-off 是什么。

## 15.2 资料来源和可信边界

本章综合参考前面章节已检索过的公开资料：

1. Claude Code 官方文档：agentic coding tool、多入口、文件编辑、命令执行、MCP、skills、hooks、sub-agents、Agent SDK、安全和权限架构。
2. Claude Code security 文档：read-only 默认、显式权限、sandboxed bash、写权限限制、prompt injection 防护、trust verification、MCP 安全、云端 VM 隔离、audit logging。
3. OpenCode 官方 server 文档：TUI 与 server 解耦，HTTP server、OpenAPI 3.1、session/message/file/diff/permission/event/MCP/LSP/agent/log API。
4. OpenCode permissions 文档：allow/ask/deny、tool 和参数级规则、external directory、doom loop、agent 级权限覆盖。
5. SWE-agent 架构文档：CLI 初始化 environment、container/shell session、Agent、HistoryProcessor、parser、tool execution、trajectory 等。
6. Aider repo map 文档：用 concise repository map 和 token budget 优化大仓库上下文。
7. MCP specification：Host/Client/Server、JSON-RPC、resources/prompts/tools、security principles、progress/cancellation/logging。
8. OpenHands README：SDK、CLI、Local GUI、Cloud、Enterprise、REST API、RBAC、协作和集成。

这些资料代表的是公开能力和设计取向。具体实现可能随版本变化，本章重点提炼系统设计原则，不声称复刻某个产品的内部实现。

## 15.3 需求澄清

面试题是“设计一个 coding agent harness”，第一步不能直接画架构图，而要澄清边界。

可以先问：

1. 目标用户是谁：个人开发者、企业团队、研究者，还是平台用户？
2. 产品形态是什么：CLI、IDE、Web、Desktop、SDK，还是都要支持？
3. 任务范围是什么：问答解释、修 bug、写功能、重构、跑测试、开 PR、处理 issue？
4. 是否要自动执行命令和修改文件？是否需要用户确认？
5. 是否接入外部系统：GitHub、Jira、Sentry、数据库、Figma、Slack、MCP、A2A？
6. 是否需要沙箱：本地进程、Docker、远程 VM、云端 workspace？
7. 是否需要评估和回放：只做产品，还是要 benchmark 和 regression eval？
8. 安全要求是什么：只读默认、最小权限、审计、RBAC、组织策略、secret 防护？
9. 规模要求是什么：单用户本地，还是多租户云服务？

一个合理的面试假设可以是：

```text
我们设计一个面向开发者和企业团队的 coding agent harness，第一版支持 CLI 和 server API，后续可接 IDE/Web。它能读取和搜索代码库、编辑文件、运行测试、调用 MCP 工具、记录 trace、支持 diff/revert、权限确认和基础 evaluation。默认在本地 workspace 运行，也支持 Docker sandbox。高风险命令需要用户确认。
```

这个假设足够具体，可以继续设计。

## 15.4 功能需求

核心功能可以分成八类。

第一类是代码理解：

1. 读取文件。
2. 文件名搜索。
3. 正则搜索。
4. 符号搜索和 LSP。
5. repo map。
6. git diff、branch、status。

第二类是代码修改：

1. 精确 edit。
2. patch apply。
3. 新增文件。
4. 格式化。
5. diff 预览。
6. revert/undo。

第三类是命令执行：

1. 运行测试。
2. 运行 lint/typecheck。
3. 安装依赖。
4. 启动开发命令。
5. 限制危险命令。

第四类是 agent loop：

1. 规划。
2. 调用模型。
3. 选择工具。
4. 解析 tool call。
5. 执行工具。
6. 汇总结果。
7. 决定下一步。

第五类是上下文管理：

1. 项目规则。
2. 用户消息。
3. 历史压缩。
4. 文件片段选择。
5. 工具结果摘要。
6. token budget。

第六类是安全治理：

1. 权限 allow/ask/deny。
2. 只读模式和 build 模式。
3. sandbox。
4. 外部目录限制。
5. secret 防护。
6. prompt injection 防护。

第七类是可观测和可恢复：

1. trace。
2. event stream。
3. logs。
4. session diff。
5. snapshot。
6. replay。

第八类是扩展和评估：

1. MCP server。
2. A2A remote agents。
3. custom tools。
4. plugins/skills。
5. evaluation harness。
6. SDK/server API。

## 15.5 非功能需求

非功能需求决定系统是否能进生产。

1. 安全性：默认最小权限，高风险操作需要确认，secret 不应泄漏。
2. 可恢复性：每次文件修改都能 diff、undo、revert。
3. 可观测性：每次模型调用、工具调用、权限请求和错误都能追踪。
4. 可复现性：关键任务能 replay，评估能复跑。
5. 可扩展性：工具、模型、MCP、agent、UI 都能扩展。
6. 低延迟体验：常见读写和搜索不能太慢。
7. 成本可控：上下文构建和模型调用不能无节制膨胀。
8. 多租户隔离：如果是云端服务，需要 workspace、credential、日志隔离。
9. 兼容性：支持不同语言、不同包管理器、不同测试框架。
10. 审计合规：企业场景需要保存操作记录、权限决策和用户确认。

## 15.6 高层架构

一个完整 agent harness 可以抽象成下面的架构：

```text
CLI / IDE / Web / SDK
-> API Gateway / Server
-> Session Manager
-> Agent Orchestrator
-> Context Builder
-> Model Adapter
-> Capability Registry
   -> Built-in Tools
   -> Custom Tools
   -> MCP Clients
   -> A2A Clients
   -> Skills / Plugins
-> Permission Engine
-> Execution Engine
   -> File Engine
   -> Shell Runner
   -> Sandbox Manager
   -> Git/Diff Manager
   -> LSP/Search Engine
-> Trace Store / Event Bus / Log Store
-> Evaluation Runner / Replay Engine
-> Config / Policy / Auth / RBAC
```

各模块职责如下：

1. CLI/IDE/Web/SDK：用户入口和程序化入口。
2. API Gateway/Server：统一暴露 HTTP、OpenAPI、SSE、鉴权和 session API。
3. Session Manager：管理会话、消息、状态、todo、diff、revert。
4. Agent Orchestrator：运行 agent loop，决定模型调用和工具调用。
5. Context Builder：构建模型输入，控制 token budget。
6. Model Adapter：适配不同模型提供商和工具调用格式。
7. Capability Registry：统一管理工具、MCP、A2A、skills、plugins。
8. Permission Engine：判断 allow、ask、deny。
9. Execution Engine：执行文件、shell、MCP、A2A、git、LSP 等动作。
10. Trace Store：记录完整执行轨迹。
11. Evaluation Runner：跑 benchmark、回归任务和 replay。
12. Config/Policy/Auth/RBAC：配置、组织策略、认证和访问控制。

这个架构的关键是把“模型推理”和“动作执行”分开。模型可以建议动作，但真正执行动作的是 runtime。

## 15.7 核心数据模型

系统设计面试中，数据模型能体现你是否真的理解 runtime。

Session：

```text
Session {
  id
  user_id
  workspace_id
  parent_session_id
  title
  status
  current_agent
  created_at
  updated_at
}
```

Message：

```text
Message {
  id
  session_id
  role: user | assistant | tool | system
  parts
  model_id
  agent_id
  created_at
}
```

ToolCall：

```text
ToolCall {
  id
  session_id
  message_id
  tool_id
  origin: builtin | custom | mcp | a2a
  input_json
  status
  permission_id
  started_at
  finished_at
  error
}
```

PermissionRequest：

```text
PermissionRequest {
  id
  session_id
  tool_call_id
  action
  resource
  risk_level
  proposed_policy
  user_response: allow_once | allow_session | deny
  created_at
  resolved_at
}
```

FileChange：

```text
FileChange {
  id
  session_id
  message_id
  path
  change_type: create | edit | delete | rename
  before_hash
  after_hash
  patch
  reversible
}
```

TraceEvent：

```text
TraceEvent {
  id
  session_id
  type
  parent_event_id
  payload
  redaction_policy
  timestamp
}
```

EvalRun：

```text
EvalRun {
  id
  task_set_id
  agent_config_id
  model_config_id
  status
  metrics
  started_at
  finished_at
}
```

这些模型不是为了数据库设计而设计，而是为了支持四件事：恢复、审计、回放、评估。

## 15.8 一次请求的完整链路

以用户输入“修复登录失败的 bug，并运行测试”为例。

完整链路是：

```text
User submits prompt
-> Server creates Message
-> Session Manager loads session state
-> Context Builder builds context
-> Model Adapter calls model
-> Model returns plan or tool call
-> Permission Engine checks tool call
-> Execution Engine runs read/grep/bash/edit
-> Tool result enters trace and context
-> Agent loop continues
-> FileChange and diff generated
-> Tests run
-> Final response summarizes changes and test result
```

这个链路里有几个关键点：

1. 每一步都产生 trace event。
2. 每次文件修改都产生 FileChange 和 diff。
3. 每次高风险工具调用都经过 PermissionRequest。
4. 工具结果不是无限制塞回上下文，而是经过截断、摘要或文件引用。
5. Agent loop 有 step limit、token limit、time limit，防止无限循环。
6. 最终回答要把修改、验证、风险和未完成项讲清楚。

## 15.9 Context Builder 设计

Context Builder 是 coding agent 的核心模块之一。

它要决定模型看到什么。输入来源包括：

1. 系统 prompt。
2. agent prompt。
3. 项目规则，如 `AGENTS.md`、`CLAUDE.md`、rules。
4. 用户当前消息。
5. 会话历史。
6. 压缩摘要。
7. repo map。
8. 当前打开文件或相关文件。
9. 工具结果。
10. git diff。
11. todo 状态。
12. MCP resources。

Context Builder 的基本策略：

1. 高优先级指令先放。
2. 用户当前任务必须完整保留。
3. 项目规则只保留相关部分。
4. 历史对话可压缩成任务状态。
5. 大文件只放相关片段。
6. 大工具输出只放摘要和引用。
7. 外部内容标记为 untrusted data。
8. 保留当前 diff 和失败日志。

可以把 context 分成层：

```text
Policy layer: system / security / permissions
Project layer: rules / repo map / coding conventions
Task layer: user goal / plan / todo
Working layer: relevant files / diff / tool outputs
Memory layer: summaries / previous decisions
```

一个常见错误是把所有文件和日志都塞进去。更好的做法是 iterative retrieval：先用 repo map、grep、glob、LSP 找候选，再读取关键片段。

## 15.10 Agent Orchestrator 设计

Agent Orchestrator 负责运行循环。

基本伪代码：

```text
while not done:
  context = build_context(session)
  model_output = call_model(context, tools)
  action = parse(model_output)

  if action is final_answer:
    finish
  if action is tool_call:
    decision = permission_engine.check(action)
    if decision == ask:
      wait_user_approval
    if decision == deny:
      return denial to model
    result = execution_engine.run(action)
    record_trace(action, result)
    update_session(result)
```

Orchestrator 要处理：

1. step limit。
2. max runtime。
3. token budget。
4. repeated tool call detection。
5. failed tool retry。
6. user interruption。
7. model timeout。
8. parallel subagents。
9. plan/build mode。
10. final answer quality。

如果没有 Orchestrator，系统就只是模型和工具的松散拼接。

## 15.11 Model Adapter 设计

不同模型提供商对 tool calling、streaming、reasoning、structured output、token 统计、错误格式都不一样。

Model Adapter 的职责是统一：

1. message format。
2. tool schema format。
3. streaming events。
4. tool call parsing。
5. retry/backoff。
6. token usage。
7. model capability。
8. structured output。
9. rate limit handling。

一个 ModelConfig 可以包含：

```text
model_id
provider
context_window
supports_tool_calling
supports_streaming
supports_reasoning_tokens
supports_parallel_tools
cost_per_input_token
cost_per_output_token
default_temperature
```

面试中要强调：不能把业务逻辑写死在某个模型 API 格式上。Harness 应该能替换模型提供商，否则很难做评估、降本和容灾。

## 15.12 Capability Registry 设计

传统叫 Tool Registry，但在支持 MCP/A2A/skills 后，更准确地叫 Capability Registry。

它管理：

1. 内置工具：read、grep、glob、edit、bash、lsp、todo。
2. Custom tools：项目或组织自定义脚本。
3. MCP tools/resources/prompts。
4. A2A remote agents。
5. Skills/plugins。
6. System actions：summary、compaction、title。

每个 capability 需要记录：

```text
id
type: tool | resource | prompt | remote_agent | skill
origin
schema
description
risk_level
required_permission
timeout
output_limit
enabled_agents
enabled_scopes
version
```

为什么要注册，而不是直接给模型？

1. 方便按 agent 启用工具。
2. 方便权限检查。
3. 方便动态禁用。
4. 方便统计使用率。
5. 方便做 tool search。
6. 方便 trace 和 replay。

## 15.13 Permission Engine 设计

Permission Engine 是安全边界。

它的输入是：

1. 用户身份。
2. workspace。
3. agent 身份。
4. tool name。
5. tool input。
6. 文件路径或外部资源。
7. 当前模式：plan/build/read-only。
8. 组织策略。
9. 历史批准记录。

输出是：

```text
allow | ask | deny
```

权限规则要支持多级覆盖：

1. 系统默认。
2. 组织策略。
3. 用户全局配置。
4. 项目配置。
5. agent 配置。
6. session 临时批准。

高风险动作包括：

1. 删除文件。
2. 写外部目录。
3. 读取 `.env`。
4. 网络请求。
5. 安装依赖。
6. git commit/push。
7. 部署命令。
8. 数据库写入。
9. 发送邮件。
10. 调用远程 agent 执行副作用任务。

关键原则：prompt 不是权限边界。真正的 allow/ask/deny 必须在 runtime 执行。

## 15.14 Sandbox 和 Execution Engine

Execution Engine 负责执行工具。

核心子模块：

1. File Engine：read、edit、write、patch、snapshot。
2. Shell Runner：bash、timeout、cwd、env、output capture。
3. Search Engine：grep、glob、file search。
4. LSP Engine：symbols、definition、references。
5. Git/Diff Manager：status、diff、revert、branch。
6. MCP Client：外部工具调用。
7. A2A Client：远程 agent task。
8. Sandbox Manager：Docker、VM、dev container、local restricted mode。

Sandbox 要控制：

1. 文件系统边界。
2. 网络访问。
3. 环境变量。
4. 进程生命周期。
5. CPU/内存/时间限制。
6. 工作区挂载。
7. secret 注入。
8. 日志导出。

本地 CLI 可以先做轻量限制：工作目录写限制、外部目录 ask、危险命令 deny、`.env` deny。云端或企业场景则需要 VM/container 隔离、网络 allowlist、credential proxy 和审计日志。

## 15.15 文件编辑、Diff 和 Revert

文件编辑是 coding agent 的核心动作，也最容易出事故。

设计原则：

1. 尽量做小 patch，不要整文件重写。
2. 每次 edit 前记录 before hash。
3. 每次 edit 后生成 diff。
4. 支持按 message revert。
5. 支持用户审查。
6. 检测工作区已有未提交修改。
7. 不自动覆盖用户并发修改。
8. 格式化要可控。

FileChange 应该和 Message 绑定。用户说“撤销刚才那步”，系统知道要 revert 哪些 patch。

如果文件在 agent 读取后被用户修改，apply patch 可能失败。这时应该重新读取文件并让模型重新规划，而不是强行覆盖。

## 15.16 Trace、Replay 和 Observability

Trace 不是日志的同义词。Trace 要能回答：agent 为什么这么做，做了什么，结果是什么。

需要记录：

1. 用户输入。
2. context 摘要和版本。
3. model request metadata。
4. model output。
5. tool call 输入输出。
6. permission decision。
7. file diff。
8. shell output。
9. MCP/A2A 调用。
10. errors 和 retry。
11. final answer。

Observability 指标包括：

1. success rate。
2. tool call count。
3. permission ask rate。
4. edit failure rate。
5. test pass rate。
6. average steps。
7. latency。
8. token cost。
9. revert rate。
10. user interruption rate。

Replay 分两种：

1. Deterministic replay：mock 模型输出和工具结果，用于 debug。
2. Re-evaluation replay：重新调用模型和工具，用于比较新版本 agent。

对外部工具要记录版本、参数、输出 hash，并决定 replay 时 mock 还是真实调用。

## 15.17 Evaluation Harness 设计

Evaluation Harness 用来回答：这个 coding agent 真的变好了吗？

任务集可以包括：

1. 单文件 bug fix。
2. 多文件 feature。
3. 测试修复。
4. lint/type error 修复。
5. RAG/文档问题。
6. 安全 review。
7. SWE-bench 风格 issue。
8. 私有历史 issue。

每个任务需要定义：

```text
repo snapshot
initial prompt or issue
expected behavior
test command
scoring function
timeout
allowed tools
security constraints
```

指标包括：

1. pass rate。
2. test pass。
3. patch quality。
4. minimality。
5. regression。
6. cost。
7. latency。
8. human review score。
9. safety violation。
10. flakiness。

评估不能只看最终是否通过测试，还要看 patch 是否过度、是否破坏风格、是否引入安全问题。

## 15.18 Server、SDK 和多客户端

如果系统只支持 CLI，短期实现简单，但扩展受限。

更合理的架构是 runtime server 化：

```text
TUI / CLI
IDE Extension
Web UI
Desktop App
SDK / CI Bot
-> Harness Server API
```

Server API 应支持：

1. session create/list/get。
2. send message。
3. stream events。
4. file read/search。
5. diff/status。
6. permission response。
7. abort/fork/revert。
8. tool/MCP status。
9. eval run。
10. logs and traces。

OpenCode 的 server 架构就是很好的公开参考：TUI 是 client，server 暴露 OpenAPI，SDK 可以基于 spec 生成。

多客户端要解决一致性问题：同一个 session 可以从 CLI 发起，在 IDE 看 diff，在 Web 审批权限。这要求 session state 在 server 中统一管理，而不是散落在各 UI。

## 15.19 MCP/A2A 扩展设计

MCP 接入：

1. Config 加载 MCP server。
2. 建立 MCP client。
3. 发现 resources/prompts/tools。
4. 注册到 Capability Registry。
5. 绑定权限和输出限制。
6. 记录 trace。

A2A 接入：

1. Agent registry 读取 Agent Card。
2. 记录 remote agent capability。
3. 主 agent 按 policy 委派任务。
4. A2A client 处理 task lifecycle。
5. streaming update 进入 event bus。
6. 结果 artifact 进入 session。

关键治理点：

1. MCP server allowlist。
2. OAuth scope 最小化。
3. A2A remote agent trust level。
4. 数据出境审计。
5. 外部输出 prompt injection 标记。
6. replay 时 mock 外部系统。

## 15.20 安全威胁模型

一个 coding agent harness 至少要考虑这些威胁。

Prompt injection：

1. 来自 issue、网页、文档、MCP output。
2. 防护：外部内容标记 untrusted，权限层不听外部指令。

Command injection：

1. 用户输入或工具输出拼接进 shell。
2. 防护：参数化、命令解释、危险命令 blocklist、用户确认。

Secret leakage：

1. 读取 `.env`、日志、token 文件。
2. 防护：默认拒绝敏感文件、redaction、最小上下文。

Workspace escape：

1. 写父目录、读 home、访问网络路径。
2. 防护：external directory ask/deny、sandbox mount。

Destructive actions：

1. 删除文件、drop table、git push、deploy。
2. 防护：高风险 ask/deny、dry-run、二次确认。

Supply chain risk：

1. 自动安装依赖或执行远程脚本。
2. 防护：网络请求审批、依赖变更审查、sandbox。

MCP/A2A trust risk：

1. 不可信 server 或 remote agent。
2. 防护：allowlist、scope、auth、audit。

## 15.21 分阶段落地方案

第一阶段：本地 CLI MVP。

1. session/message。
2. read/grep/glob/edit/bash。
3. permission allow/ask/deny。
4. diff 和 revert。
5. 基础 trace。

第二阶段：上下文和代码理解增强。

1. repo map。
2. LSP。
3. history compaction。
4. todo。
5. test/lint 自动闭环。

第三阶段：server 和多客户端。

1. HTTP API。
2. event stream。
3. IDE/Web client。
4. SDK。
5. session fork/share。

第四阶段：扩展生态。

1. MCP。
2. custom tools。
3. skills/plugins。
4. A2A remote agents。

第五阶段：企业和评估。

1. RBAC。
2. org policy。
3. audit logs。
4. sandbox/VM。
5. evaluation harness。
6. replay 和 regression dashboard。

## 15.22 关键 Trade-off

自动化程度 vs 安全：

1. 自动化越高，用户越省心。
2. 但误删、误部署、泄密风险越高。
3. 解决方式是分风险等级、分模式、分 agent 权限。

上下文完整性 vs 成本：

1. 上下文越多，模型越可能理解任务。
2. 但 token 成本、延迟和干扰也更高。
3. 解决方式是 repo map、检索、摘要、tool search。

工具能力 vs 风险：

1. 工具越多，agent 能做的事越多。
2. 但误调用和权限面扩大。
3. 解决方式是 capability registry 和 permission engine。

通用性 vs 产品体验：

1. 通用 runtime 支持多场景。
2. 但可能不如 IDE 原生体验顺滑。
3. 解决方式是 server runtime + 多客户端 UI。

可观测性 vs 性能：

1. trace 记录越完整，debug 和审计越好。
2. 但存储、脱敏、性能成本更高。
3. 解决方式是分级 trace 和可配置 retention。

本地执行 vs 云端执行：

1. 本地更贴近用户环境和隐私。
2. 云端更适合异步、协作和隔离。
3. 解决方式是支持两种 workspace backend。

## 15.23 面试题

### 题 1：如何设计一个 coding agent harness？

参考回答：

```text
我会先澄清产品形态和安全边界。高层上分成客户端、server、session manager、agent orchestrator、context builder、model adapter、capability registry、permission engine、execution engine、trace store 和 eval runner。用户输入进入 session 后，context builder 构建上下文，model adapter 调模型，orchestrator 解析工具调用，permission engine 判断 allow/ask/deny，execution engine 执行文件、shell、MCP 或 A2A 调用，结果进入 trace 和 context，直到生成最终回答。所有文件修改生成 diff 和 revert 信息，所有高风险动作需要权限确认，评估和 replay 基于 trace 和 repo snapshot 实现。
```

### 题 2：为什么 session manager 很重要？

参考回答：

```text
因为 coding agent 不是单轮问答。Session 需要保存用户目标、消息、工具调用、文件 diff、todo、权限请求、summary、trace 和 revert 状态。没有 session manager，就无法支持长任务、撤销、fork、share、异步执行、event stream 和 evaluation replay。
```

### 题 3：权限系统应该怎么设计？

参考回答：

```text
权限系统要在 runtime 层执行，而不是只靠 prompt。它输入用户、workspace、agent、tool、参数、路径、组织策略和 session 临时授权，输出 allow、ask 或 deny。规则要支持系统默认、组织、用户、项目、agent 和 session 多层覆盖。高风险动作如读 .env、写外部目录、删除文件、git push、部署、数据库写入和发送邮件都应 ask 或 deny。
```

### 题 4：如何让 agent 既能理解大仓库，又不爆上下文？

参考回答：

```text
不能把整个仓库塞进上下文。我会结合 repo map、grep/glob、LSP、git diff 和历史摘要。先用 repo map 提供全局骨架，再根据用户任务检索相关文件，读取关键片段，工具结果做摘要或落盘引用。上下文分为 policy、project、task、working、memory 几层，并用 token budget 控制每层大小。
```

### 题 5：如何设计 trace 和 replay？

参考回答：

```text
Trace 要记录用户输入、context 版本、模型调用元信息、模型输出、工具调用输入输出、权限决策、文件 diff、shell 输出、MCP/A2A 调用、错误和最终答案。Replay 分 deterministic replay 和 re-evaluation replay。前者用录制的模型和工具结果复现 bug，后者重新调用模型比较新版本 agent。对外部系统要记录版本、输出 hash，并决定 replay 时 mock 还是真实调用。
```

## 15.24 小练习

1. 画出一个 coding agent harness 的模块图，要求包含 session、context、model、tool、permission、execution、trace、eval。
2. 设计 `Session`、`Message`、`ToolCall`、`PermissionRequest`、`FileChange` 五张表。
3. 给 `bash`、`edit`、`read`、`mcp.github.create_issue`、`a2a.security_agent` 设计权限规则。
4. 设计一个 context budget 分配方案：系统指令、项目规则、repo map、文件片段、工具结果、历史摘要各占多少。
5. 设计一个 replay 方案，说明如何处理 MCP 外部系统结果。
6. 设计一个 evaluation task schema，用于评估 agent 修复历史 bug 的能力。
7. 设计一个企业版 harness 的 RBAC 和 audit log 方案。

## 15.25 本章总结

本章把第二十册前面的内容整合成一个完整系统设计答案。

核心结论：

1. Coding agent harness 不是模型 API 包装，而是完整 runtime。
2. 高层架构应包含 client/server、session manager、agent orchestrator、context builder、model adapter、capability registry、permission engine、execution engine、trace store 和 eval runner。
3. Session、Message、ToolCall、PermissionRequest、FileChange、TraceEvent 是关键数据模型。
4. Context Builder 决定模型能否正确理解代码库，必须结合 repo map、检索、摘要和 token budget。
5. Permission Engine 和 Sandbox 是安全边界，不能只靠 prompt。
6. Trace、diff、revert、replay 和 evaluation 是生产级 coding agent 的核心能力。
7. MCP、A2A、skills、plugins 和 SDK 让 harness 具备扩展性，但也扩大了治理面。
8. 企业落地时要重点关注 RBAC、审计、secret、防 prompt injection、外部系统权限和评估复现。

下一章会整理 Agent Harness 真实实战坑和面试题，把本书的知识转成排查清单和高频问答。
