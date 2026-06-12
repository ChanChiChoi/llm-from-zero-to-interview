# 第十八章：MCP 基本概念：Client、Server、Tools、Resources、Prompts

## 18.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求，先对齐 MCP 官方介绍、MCP 2025-06-18 specification 中 lifecycle、tools、resources、prompts、roots、sampling、authorization、logging 和 transport 相关口径，以及 OpenAI Agents SDK 对 MCP server 接入的工程抽象。

本章只建立 MCP 的基本对象模型，不写完整 MCP Server 实现，不展开权限安全和本地沙箱，不讨论企业 MCP 平台系统设计，也不把某个 SDK、某个 server 模板、某家模型 provider 或某个 IDE 插件的字段写成通用标准。后续第 19 章会讲最小 server，第 23 章会讲权限安全，第 48 章会讲企业 MCP 工具平台。

第二轮精修重点放在三件事：

1. 用公式把 Host、Client、Server、Tools、Resources、Prompts、Transport、Roots、Sampling 和治理映射串成可检查的概念模型。
2. 区分 MCP 协议对象与模型 API、企业工具平台、插件系统、HTTP API 的职责边界。
3. 增加一个 0 依赖 Python demo，用 toy concept cases 审计一个 MCP 方案是否正确建模基本概念。

## 18.1 本章定位

上一章讲了 MCP 为什么出现。本章建立 MCP 的基本概念。

理解 MCP，不能只记“它能调用工具”。MCP 至少涉及：

1. Host。
2. Client。
3. Server。
4. Tools。
5. Resources。
6. Prompts。
7. Transport。
8. Capability discovery。
9. Sampling / Roots 等扩展概念。

本章的核心观点是：

```text
MCP 是一个围绕 Host、Client、Server 组织的上下文协议，Server 暴露 tools、resources 和 prompts，Client 负责发现和调用，Host 决定这些能力如何进入模型应用。
```

## 18.2 一张图理解 MCP

典型 MCP 架构可以画成：

```text
User
  ↓
Host Application
  ↓
LLM Provider

Host Application
  ↓
MCP Client
  ↓
MCP Server
  ↓
External System
```

例如 IDE Agent：

```text
IDE 是 Host Application
IDE 内部运行 MCP Client
文件系统/Git/数据库 MCP Server 暴露能力
LLM 根据 Host 提供的工具和上下文完成任务
```

关键点：MCP Server 通常不直接和 LLM 对话。Host / Client 负责把 MCP 能力接入模型。

## 18.3 Host 是什么

Host 是承载模型交互的应用。

例如：

1. 桌面聊天应用。
2. IDE。
3. 命令行 Agent。
4. 企业客服平台。
5. 数据分析平台。
6. 自动化办公助手。

Host 负责：

1. 管理用户会话。
2. 调用 LLM。
3. 管理 MCP Client。
4. 决定连接哪些 MCP Server。
5. 决定哪些 tools/resources/prompts 暴露给模型。
6. 做权限和用户确认。
7. 把 MCP tool result 回填给模型。

Host 是用户体验和安全策略的主要承载者。

如果 Host 不做权限和安全控制，MCP Server 暴露再规范也可能被滥用。

## 18.4 MCP Client 是什么

MCP Client 是 Host 中负责和 MCP Server 通信的组件。

它负责：

1. 建立连接。
2. 初始化协议。
3. 发现 server 能力。
4. 列出 tools。
5. 调用 tools。
6. 列出和读取 resources。
7. 获取 prompts。
8. 处理错误。
9. 把结果交给 Host。

一个 Host 可以连接多个 MCP Server。

例如：

```text
IDE Host
  ├── filesystem MCP Server
  ├── git MCP Server
  ├── database MCP Server
  └── issue-tracker MCP Server
```

Client 的职责不是替代工具治理，而是协议通信和能力发现。

## 18.5 MCP Server 是什么

MCP Server 是暴露上下文能力的一端。

它可以连接：

1. 本地文件系统。
2. Git 仓库。
3. 数据库。
4. 知识库。
5. 浏览器。
6. 第三方 API。
7. 企业内部系统。
8. 本地命令或脚本。

Server 对外暴露：

1. Tools。
2. Resources。
3. Prompts。

Server 的职责：

1. 声明自己支持什么能力。
2. 提供工具 schema。
3. 执行工具调用。
4. 返回资源内容。
5. 提供 prompt 模板。
6. 返回结构化错误。

Server 应该遵循最小权限原则，不要把整个系统能力无差别暴露出去。

## 18.6 Tools

Tools 是可调用动作。

例如：

1. `search_files`。
2. `read_file`。
3. `query_database`。
4. `create_issue`。
5. `run_tests`。
6. `search_docs`。

一个 MCP tool 通常包含：

1. name。
2. description。
3. input schema。
4. 调用结果。

它和 Function Calling 里的 tool 很像，但区别是：MCP tool 是 MCP Server 暴露给 MCP Client 的能力，Host 再决定如何投影给模型。

典型流程：

```text
Client list tools
  ↓
Host chooses which tools to expose to model
  ↓
Model requests tool call
  ↓
Host / Client calls MCP tool
  ↓
Server executes and returns result
```

## 18.7 Resources

Resources 是可读取的上下文资源。

例如：

1. 文件。
2. 文档。
3. 数据库记录。
4. 网页。
5. Git diff。
6. 项目配置。
7. 日志片段。
8. 当前 IDE 打开的文件。

Resource 与 Tool 的区别：

```text
Tool 强调动作：调用后做某件事。
Resource 强调上下文：读取某个可寻址的信息源。
```

Resource 通常有 URI 或类似标识。

例如：

```text
file:///project/src/main.py
database://customers/C123
doc://policy/expense-2026
```

模型应用可以通过 Client 读取 resource，把内容作为上下文提供给模型。

## 18.8 Prompts

Prompts 是可复用的提示模板或工作流入口。

例如：

1. 代码审查 prompt。
2. SQL 分析 prompt。
3. 故障排查 prompt。
4. 文档总结 prompt。
5. 客服回复 prompt。

Prompt 可以带参数。

例如：

```json
{
  "name": "review_code",
  "description": "对指定文件或 diff 做代码审查",
  "arguments": ["target", "focus"]
}
```

Prompts 的价值是让 server 提供领域最佳实践，而不是让每个 Host 自己写提示词。

但 Host 仍要决定是否信任、如何展示、是否允许用户选择这些 prompts。

## 18.9 Tools、Resources、Prompts 的边界

三者边界可以这样理解：

| 类型 | 关注点 | 例子 |
|---|---|---|
| Tool | 执行动作 | 查询数据库、创建工单、运行测试 |
| Resource | 暴露上下文 | 文件、文档、网页、数据库记录 |
| Prompt | 复用任务模板 | 代码审查模板、故障排查模板 |

一个场景可能同时用三者。

例如“帮我审查这次代码变更”：

1. Resource：Git diff。
2. Tool：运行测试。
3. Prompt：代码审查模板。

不要把所有东西都做成 tool。资源读取和 prompt 模板有自己的语义。

## 18.10 Transport

Transport 是 Client 和 Server 通信的传输方式。

常见可以包括：

1. 本地 stdio。
2. HTTP / SSE。
3. 其他流式或进程间通信方式。

本地 stdio 适合：

1. 本地工具。
2. IDE。
3. 桌面应用。
4. 文件系统和 Git server。

HTTP 类传输适合：

1. 远程服务。
2. 企业工具平台。
3. 多用户共享 server。

Transport 不改变 MCP 的抽象：Client 仍然发现 tools/resources/prompts，Server 仍然暴露能力。

但 transport 会影响安全、认证、部署和延迟。

## 18.11 初始化和能力发现

MCP 连接建立后，通常需要初始化和能力发现。

大致流程：

```text
Client connects to Server
  ↓
initialize / handshake
  ↓
Server declares capabilities
  ↓
Client lists tools/resources/prompts
  ↓
Host decides what to expose
```

能力发现让 Host 不必硬编码 server 有哪些工具。

但发现到能力不代表全部可用。Host 仍要按用户、权限、场景和风险过滤。

## 18.12 Roots

Roots 可以理解为 Host 暴露给 Server 的边界或工作区根。

例如 IDE Host 连接文件系统 server 时，可以告诉 server：

```text
当前允许访问的项目根目录是 /home/user/project
```

这样 server 不应读取根目录之外的文件。

Roots 的意义是限制上下文范围，尤其对文件系统、代码仓库、本地资源访问非常重要。

没有 roots 或类似边界，MCP Server 可能暴露过宽的本地资源。

## 18.13 Sampling

一些 MCP 设计中还会涉及 sampling，即 Server 请求 Host 帮忙调用模型生成内容。

可以理解为：

```text
Server 不直接持有模型，而是通过 Host 请求一次模型采样。
```

这类能力要非常谨慎，因为它可能让 server 间接影响模型调用。

Host 必须控制：

1. 哪些 server 可以请求 sampling。
2. sampling 使用什么模型。
3. 是否需要用户确认。
4. 输入输出是否脱敏。
5. 是否记录 trace。

对初学者来说，先理解 tools/resources/prompts 即可，sampling 属于更高级能力。

## 18.14 MCP 与 JSON-RPC 风格

MCP 的消息交互通常具有 RPC 风格：请求、响应、通知、错误。

你可以把它理解成：

```text
Client 调用 Server 的某个协议方法，Server 返回结构化结果或错误。
```

例如：

1. 列出工具。
2. 调用工具。
3. 列出资源。
4. 读取资源。
5. 获取 prompt。

这和模型对话消息不同。MCP 是 Host/Client 与 Server 的工程协议，模型 API 是 Host 与 LLM 的对话协议。

Host 负责把二者连接起来。

## 18.15 MCP Tool 调用流程

一次 MCP tool 调用可以拆成：

```text
1. Client 从 Server 获取 tools 列表。
2. Host 把部分 tools 投影给 LLM。
3. LLM 生成 tool call。
4. Host 判断允许执行。
5. MCP Client 向 MCP Server 发送 tool call 请求。
6. MCP Server 执行工具。
7. Server 返回结果或错误。
8. Host 包装 tool result，回填给 LLM。
9. LLM 生成最终回答。
```

这个流程说明：MCP tool 不是绕过 Host 直接执行。Host 仍然是策略和安全控制中心。

## 18.16 MCP Resource 读取流程

一次 resource 读取可以是：

```text
1. Client 列出可用 resources。
2. Host 或用户选择某个 resource。
3. Client 请求 Server 读取 resource。
4. Server 返回内容和 metadata。
5. Host 决定如何压缩、脱敏、放入模型上下文。
```

Resource 读取不一定由模型自动触发。也可能由用户选择、Host 自动加载、或工作流决定。

例如 IDE 中当前打开文件可以作为 resource 自动提供，但项目所有文件不应该全部塞进上下文。

## 18.17 MCP Prompt 使用流程

Prompt 使用流程：

```text
1. Client 列出 server 提供的 prompts。
2. 用户或 Host 选择 prompt。
3. Client 获取 prompt 模板。
4. Host 填充参数。
5. Host 把最终 prompt 放入模型对话。
```

Prompts 可以帮助标准化工作流。

例如数据库 MCP Server 提供：

```text
analyze_slow_query
```

它可以封装分析慢 SQL 所需的问题结构、上下文要求和输出格式。

但 prompt 也可能包含不合适指令，因此 Host 仍要做信任和审查。

## 18.18 MCP Server 的能力边界

MCP Server 应该清楚声明自己能做什么，不能做什么。

例如文件系统 server：

1. 能读取指定根目录下文件。
2. 能列目录。
3. 可能能写文件。
4. 不能访问根目录外路径。
5. 不能执行 shell。

能力边界应反映在：

1. tool description。
2. resource scope。
3. server 配置。
4. Host 权限界面。
5. runtime policy。

不要让一个 server 暴露过多不相关能力。能力越宽，攻击面越大。

## 18.19 MCP 和企业治理映射

企业平台可以这样映射 MCP 概念：

| MCP 概念 | 企业治理对象 |
|---|---|
| Server | 工具提供方 / connector |
| Tool | Tool Registry item |
| Resource | 受控上下文资源 |
| Prompt | 工作流模板 |
| Client | Agent runtime 连接器 |
| Host | 产品应用 / Agent 容器 |

这样 MCP 能接入上一部分讲的企业工具平台：

1. MCP tools 注册进 Registry。
2. MCP resources 加权限和范围。
3. MCP prompts 加审核和版本。
4. MCP calls 进入 Executor trace。
5. MCP Server 纳入安全治理。

## 18.20 MCP 基本概念指标与最小 demo

面试中讲 MCP 基本概念，不应只背 “Client、Server、Tools、Resources、Prompts” 这几个词。更好的回答是说明每个对象的职责、边界、数据流和治理证据。可以把一个 MCP 连接样本写成：

```math
u_i=(h_i,c_i,s_i,T_i,R_i,P_i,\ell_i,\rho_i,\sigma_i,g_i,z_i)
```

其中，`h_i` 是 Host，`c_i` 是 MCP Client，`s_i` 是 MCP Server，`T_i` 是 tools 集合，`R_i` 是 resources 集合，`P_i` 是 prompts 集合，`\ell_i` 是 lifecycle / capability discovery 信息，`\rho_i` 是 roots / resource scope，`\sigma_i` 是 sampling / model request 控制，`g_i` 是权限、trace、eval 和企业治理映射，`z_i` 是风险或缺陷标签。

对一组连接样本 `U`，可以用统一形式定义概念覆盖率：

```math
C_k=\frac{1}{|U|}\sum_{u_i\in U}\mathbf{1}[r_k(u_i)=1]
```

其中，`r_k(u_i)` 表示第 `k` 类 MCP 概念是否被正确建模。核心指标包括：

```math
C_{\mathrm{host}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{host\ owns\ UX\ policy}_i]
```

```math
C_{\mathrm{client}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{client\ server\ boundary}_i]
```

```math
C_{\mathrm{server}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{server\ capability\ declaration}_i]
```

```math
C_{\mathrm{tool}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{tool\ schema\ and\ result}_i]
```

```math
C_{\mathrm{resource}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{resource\ URI\ and\ metadata}_i]
```

```math
C_{\mathrm{prompt}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{prompt\ arguments\ and\ review}_i]
```

```math
C_{\mathrm{life}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{lifecycle\ negotiation}_i]
```

```math
C_{\mathrm{transport}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{transport\ policy}_i]
```

```math
C_{\mathrm{roots}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{roots\ boundary}_i]
```

```math
C_{\mathrm{sampling}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{sampling\ controlled}_i]
```

```math
C_{\mathrm{filter}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{host\ filtering}_i]
```

```math
C_{\mathrm{trace}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{trace\ eval\ mapping}_i]
```

如果要估算把 MCP 能力投影给模型后的上下文预算，可以写成：

```math
B_{\mathrm{ctx}}=B_{\mathrm{sys}}+B_{\mathrm{user}}+B_{\mathrm{tools}}+B_{\mathrm{resources}}+B_{\mathrm{prompts}}+B_{\mathrm{results}}
```

这里的重点是：连接 server 不等于把所有资源、工具描述和 prompt 全塞给模型。Host 必须按任务、权限、风险和上下文预算过滤。

综合门禁可以写成：

```math
G_{\mathrm{mcp\_concept}}=\mathbf{1}[C_{\mathrm{host}}\ge \tau_{\mathrm{host}}]\cdot
\mathbf{1}[C_{\mathrm{client}}\ge \tau_{\mathrm{client}}]\cdot
\mathbf{1}[C_{\mathrm{server}}\ge \tau_{\mathrm{server}}]\cdot
\mathbf{1}[C_{\mathrm{tool}}\ge \tau_{\mathrm{tool}}]\cdot
\mathbf{1}[C_{\mathrm{resource}}\ge \tau_{\mathrm{resource}}]\cdot
\mathbf{1}[C_{\mathrm{prompt}}\ge \tau_{\mathrm{prompt}}]\cdot
\mathbf{1}[C_{\mathrm{life}}\ge \tau_{\mathrm{life}}]\cdot
\mathbf{1}[C_{\mathrm{transport}}\ge \tau_{\mathrm{transport}}]\cdot
\mathbf{1}[C_{\mathrm{roots}}\ge \tau_{\mathrm{roots}}]\cdot
\mathbf{1}[C_{\mathrm{sampling}}\ge \tau_{\mathrm{sampling}}]\cdot
\mathbf{1}[C_{\mathrm{filter}}\ge \tau_{\mathrm{filter}}]\cdot
\mathbf{1}[C_{\mathrm{trace}}\ge \tau_{\mathrm{trace}}]
```

下面的 demo 用一组 toy cases 审计 MCP 基本概念。它故意包含错误样本：把 Client 当 LLM、让 Server 直接连模型、tool 缺 schema、resource 被设计成任意动作、prompt 自动执行、缺 lifecycle、roots 过宽、sampling 不受控、transport 没有认证、Host 不过滤能力等。

```python
checks = [
    "host_policy_ownership",
    "client_server_boundary",
    "server_capability_declaration",
    "tool_schema_and_result",
    "resource_uri_and_metadata",
    "prompt_argument_review",
    "lifecycle_negotiation",
    "transport_policy",
    "roots_boundary",
    "sampling_control",
    "host_capability_filtering",
    "trace_eval_mapping",
]

cases = [
    {
        "name": "host_client_server_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "tools_resources_prompts_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "lifecycle_discovery_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "roots_scope_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "enterprise_mapping_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "client_equals_llm_bad",
        "host_policy_ownership": True,
        "client_server_boundary": False,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "server_direct_to_model_bad",
        "host_policy_ownership": False,
        "client_server_boundary": False,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": False,
        "host_capability_filtering": False,
        "trace_eval_mapping": True,
    },
    {
        "name": "tool_without_schema_bad",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": False,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "resource_as_any_action_bad",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": False,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": False,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "prompt_auto_trusted_bad",
        "host_policy_ownership": False,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": False,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": False,
        "trace_eval_mapping": True,
    },
    {
        "name": "no_lifecycle_negotiation_bad",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": False,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": False,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "transport_without_auth_bad",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": False,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
    {
        "name": "roots_too_broad_bad",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": False,
        "sampling_control": True,
        "host_capability_filtering": False,
        "trace_eval_mapping": True,
    },
    {
        "name": "sampling_without_review_bad",
        "host_policy_ownership": False,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": False,
        "host_capability_filtering": True,
        "trace_eval_mapping": False,
    },
    {
        "name": "full_mcp_concept_ready_ok",
        "host_policy_ownership": True,
        "client_server_boundary": True,
        "server_capability_declaration": True,
        "tool_schema_and_result": True,
        "resource_uri_and_metadata": True,
        "prompt_argument_review": True,
        "lifecycle_negotiation": True,
        "transport_policy": True,
        "roots_boundary": True,
        "sampling_control": True,
        "host_capability_filtering": True,
        "trace_eval_mapping": True,
    },
]

context_budget = {
    "system": 600,
    "user": 260,
    "tools": 900,
    "resources": 1800,
    "prompts": 420,
    "results": 700,
}
context_budget["total"] = sum(context_budget.values())
context_budget["within_8k"] = context_budget["total"] <= 8000


def pass_rate(key):
    return round(sum(1 for case in cases if case[key]) / len(cases), 3)


metrics = {key: pass_rate(key) for key in checks}
failed_cases = [
    case["name"]
    for case in cases
    if any(case[key] is False for key in checks)
]
failed_gates = [key for key, value in metrics.items() if value < 1.0]
mcp_concept_gate_pass = context_budget["within_8k"] and not failed_gates

print("context_budget=", context_budget)
print("metrics=", metrics)
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("mcp_concept_gate_pass=", mcp_concept_gate_pass)
```

一组稳定输出如下：

```text
context_budget= {'system': 600, 'user': 260, 'tools': 900, 'resources': 1800, 'prompts': 420, 'results': 700, 'total': 4680, 'within_8k': True}
metrics= {'host_policy_ownership': 0.8, 'client_server_boundary': 0.867, 'server_capability_declaration': 0.933, 'tool_schema_and_result': 0.933, 'resource_uri_and_metadata': 0.933, 'prompt_argument_review': 0.933, 'lifecycle_negotiation': 0.933, 'transport_policy': 0.933, 'roots_boundary': 0.867, 'sampling_control': 0.867, 'host_capability_filtering': 0.8, 'trace_eval_mapping': 0.933}
failed_cases= ['client_equals_llm_bad', 'server_direct_to_model_bad', 'tool_without_schema_bad', 'resource_as_any_action_bad', 'prompt_auto_trusted_bad', 'no_lifecycle_negotiation_bad', 'transport_without_auth_bad', 'roots_too_broad_bad', 'sampling_without_review_bad']
failed_gates= ['host_policy_ownership', 'client_server_boundary', 'server_capability_declaration', 'tool_schema_and_result', 'resource_uri_and_metadata', 'prompt_argument_review', 'lifecycle_negotiation', 'transport_policy', 'roots_boundary', 'sampling_control', 'host_capability_filtering', 'trace_eval_mapping']
mcp_concept_gate_pass= False
```

这段 demo 的重点是：MCP 基本概念不是词汇表，而是一组边界。Host 管用户体验和策略，Client 管协议连接，Server 暴露能力，Tool / Resource / Prompt 各有语义，Transport / Roots / Sampling 会改变安全和治理要求。任何一个边界说错，后续实现 MCP Server、做权限治理或系统设计都会变形。

## 18.21 常见误区

### 18.21.1 Server 等于模型插件

不准确。MCP Server 是协议 server，暴露 tools/resources/prompts，可以被多个 Host 连接。

### 18.21.2 Client 等于 LLM

不准确。Client 是 Host 中和 MCP Server 通信的组件。LLM 通常通过 Host 间接使用 MCP 能力。

### 18.21.3 Resource 就是 Tool

不准确。Resource 是上下文资源，Tool 是可执行动作。读取文件可以设计成 resource，也可以设计成 tool，但语义不同。

### 18.21.4 Prompt 一定可信

不一定。Prompt 也来自 server，Host 应决定是否信任和如何使用。

### 18.21.5 连接 server 后所有能力都能给模型

不对。Host 要按用户、场景、权限、风险过滤。

## 18.22 面试题：解释 MCP 基本概念

面试官可能问：

```text
MCP 里 Client、Server、Tools、Resources、Prompts 分别是什么？
```

可以这样回答：

第一，Host 是模型应用，例如 IDE、桌面助手或企业 Agent 平台。

第二，MCP Client 是 Host 里负责连接 MCP Server 的组件。

第三，MCP Server 是暴露外部能力的一端，可以连接文件系统、数据库、Git、知识库或企业 API。

第四，Tools 是可调用动作，例如查询、搜索、创建、运行测试。

第五，Resources 是可读取上下文，例如文件、文档、网页、数据库记录。

第六，Prompts 是可复用提示模板或工作流入口。

第七，Host 负责把这些能力安全地提供给模型，并做权限、确认、trace 和结果回填。

一句话总结：

```text
MCP 把外部能力抽象成 Server 暴露的 tools、resources 和 prompts，由 Host 中的 Client 发现和调用，再由 Host 安全地接入模型对话。
```

## 18.23 小练习

### 练习 1：谁连接谁

MCP Server 是否直接连接大模型？

参考答案：通常不是。MCP Server 连接 MCP Client，Client 位于 Host Application 中，Host 再调用模型。

### 练习 2：Tool 还是 Resource

`file:///project/main.py` 更像 tool 还是 resource？

参考答案：更像 resource，因为它是可读取上下文。读取它的动作由 client/server 协议完成。

### 练习 3：运行测试

`run_tests` 更像 tool 还是 resource？

参考答案：更像 tool，因为它是执行动作，并可能产生新的结果。

### 练习 4：代码审查模板

`review_code_prompt` 更像什么？

参考答案：更像 prompt，因为它是可复用提示模板或工作流入口。

### 练习 5：Host 的职责

Host 连接了一个文件系统 MCP Server，是否应该把所有文件都自动放进模型上下文？

参考答案：不应该。Host 应按用户意图、权限、上下文预算和安全策略选择相关资源。

### 练习 6：概念审计

一个方案说“Client 就是 LLM，Server 直接把本地文件和 prompt 都交给模型”，这错在哪里？

参考答案：Client 是 Host 中的协议组件，不是 LLM；Server 不应直接绕过 Host 把能力交给模型；本地文件应通过 roots、权限、resource metadata 和 Host 过滤控制；prompt 也需要审查和参数治理。

## 18.24 本章小结

本章讲了 MCP 基本概念。

你需要掌握：

1. Host 是模型应用，负责用户体验和安全策略。
2. MCP Client 是 Host 中连接 MCP Server 的组件。
3. MCP Server 暴露外部 tools、resources 和 prompts。
4. Tools 是可调用动作。
5. Resources 是可读取上下文。
6. Prompts 是可复用提示模板或工作流入口。
7. Transport 决定 Client 和 Server 如何通信，但不改变 MCP 抽象。
8. 能力发现让 Client 获取 server 暴露的能力，但 Host 仍要过滤。
9. Roots 用于限制资源访问边界，尤其是本地文件和工作区。
10. MCP 是 Host/Client 与 Server 的协议，不是模型 API 本身。
11. 企业治理可以把 MCP Server、Tools、Resources、Prompts 映射到 Registry、权限、安全和 trace 系统。
12. 可以用 host policy、client/server boundary、server capability、tool schema、resource URI、prompt review、lifecycle、transport、roots、sampling、host filtering 和 trace/eval mapping 指标检查概念是否讲清楚。

如果只记一句话：

```text
MCP 的基本结构是 Host 内的 Client 连接外部 Server，Server 以 tools、resources、prompts 的形式暴露能力，Host 决定这些能力如何安全进入模型上下文。
```

下一章会进入 MCP Server 的最小实现，讲一个最小 server 应该如何声明工具、处理请求并返回结果。
