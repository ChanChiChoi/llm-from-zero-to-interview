# 第二十章：MCP Tool 与传统 Function Calling 的区别

## 20.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，参考 MCP 官方 2025-06-18 specification 中 tools、resources、prompts、lifecycle、transport 的协议口径，OpenAI function calling / tools / structured outputs 的模型 API 口径，以及 OpenAI Agents SDK 中 MCP server 接入、tool filtering、approval、tracing 和 hosted / streamable HTTP / stdio server 的工程抽象。正文只讨论稳定分层：模型 API 层的 function calling、Host / runtime 层的工具执行、MCP Client / Server 层的能力发现与连接，不把某一家 provider 的字段名、某个 SDK 的装饰器、某个 IDE 配置或某个 MCP server 模板写成通用标准。

本章新增公式和 demo 只用于面试与工程审计：它们帮助判断一个回答是否真正区分了 Function Calling 与 MCP Tool 的层次、发现方式、能力范围、执行边界、adapter、生命周期、安全和选型 trade-off。真实项目中，MCP Tool 常常会被 Host 投影成模型 API 的 function / tool schema；这说明二者可以组合，不说明二者是同一个协议层。

## 20.1 本章定位

前面讲了 MCP 的背景、基本概念和最小 server。本章专门回答一个高频问题：MCP Tool 和传统 Function Calling 到底有什么区别？

很多人第一次看 MCP，会觉得：

```text
这不就是又一种 function calling 吗？
```

这个理解不完全错，因为 MCP Tool 最终也可能被 Host 投影成模型可调用的 tool。但如果只把 MCP 当成“另一种 function calling”，就会错过它真正解决的问题。

本章的核心观点是：

```text
传统 Function Calling 是模型 API 层的工具调用表达；MCP Tool 是 MCP Server 向模型客户端暴露能力的协议对象，重点在跨客户端的发现、连接、上下文生态和治理边界。
```

## 20.2 先给结论

可以用一句话区分：

```text
Function Calling 关注“模型如何请求调用工具”；MCP 关注“工具和上下文能力如何被模型应用发现、连接和使用”。
```

更具体地说：

| 维度 | Function Calling | MCP Tool |
|---|---|---|
| 协议位置 | 模型 API / Host 与模型之间 | MCP Client 与 MCP Server 之间 |
| 主要对象 | tool schema、tool call、tool result | server、tools、resources、prompts |
| 谁暴露工具 | Host / 应用代码 | MCP Server |
| 谁调用工具 | Host runtime | MCP Client 调用 MCP Server |
| 发现机制 | 通常由应用静态传 tools | Client 可从 Server list tools |
| 生态目标 | 让模型结构化调用函数 | 让外部能力跨客户端复用 |
| 是否包含 resources/prompts | 通常不包含 | 原生包含 |
| 治理重点 | 参数、执行、回填 | server 连接、能力发现、上下文边界 |

## 20.3 协议位置不同

传统 Function Calling 的位置：

```text
Host Application ↔ LLM Provider
```

Host 把 tools 传给模型，模型返回 tool call，Host 执行工具。

MCP Tool 的位置：

```text
Host Application / MCP Client ↔ MCP Server
```

MCP Server 暴露 tools，Client 获取 tools，Host 再决定是否把这些 tools 提供给模型。

完整链路可能是：

```text
LLM
  ↑ function calling protocol
Host Application
  ↑ MCP protocol
MCP Server
  ↑ internal API / local system
External System
```

所以 MCP Tool 和 Function Calling 不在同一层。它们可以组合，而不是互相替代。

## 20.4 工具发现方式不同

传统 Function Calling 中，工具通常由应用代码静态定义：

```python
tools = [get_weather_tool, search_docs_tool]
model.generate(messages, tools=tools)
```

如果要新增工具，应用代码或配置要更新。

MCP 中，Client 可以连接 Server 后请求 tools list：

```text
Client → Server: list tools
Server → Client: get_weather, search_docs, read_file...
```

这意味着工具能力可以由 server 动态暴露，Host 不必为每个工具手写集成代码。

当然，Host 仍要过滤和治理工具。发现到工具不等于自动信任工具。

## 20.5 工具提供方不同

传统 Function Calling 中，工具通常由 Host 应用自己实现或包装。

例如：

```text
Chat App 内部实现 get_weather。
```

MCP 中，工具由 MCP Server 暴露。

例如：

```text
weather-mcp-server 暴露 get_weather。
file-system-mcp-server 暴露 read_file。
git-mcp-server 暴露 get_diff。
```

这样同一个 MCP Server 可以被多个 Host 使用。

例如 Git MCP Server 可以同时服务：

1. IDE Agent。
2. CLI Agent。
3. Code Review Agent。
4. 企业代码助手。

这就是 MCP 的生态复用价值。

## 20.6 执行边界不同

Function Calling 的工具执行通常在 Host runtime 内部完成：

```text
Host receives tool call → Host calls local function / API
```

MCP Tool 的执行边界是：

```text
Host receives model tool call → MCP Client calls MCP Server → Server executes
```

这带来两个变化。

第一，工具执行可以由独立 server 管理。

第二，Host 和 Server 之间需要明确权限、连接和传输边界。

如果 MCP Server 是本地文件系统 server，它的权限边界是本地 workspace。如果是远程企业 server，它需要认证和租户隔离。

## 20.7 Schema 的来源不同

Function Calling schema 通常由应用开发者写在 Host 侧。

MCP Tool schema 通常由 Server 提供。

这有好处：工具 owner 可以维护自己的 schema。

但也有风险：Host 不能无条件信任 Server 提供的 schema。

Host 仍要检查：

1. 工具名是否合规。
2. description 是否清晰。
3. schema 是否过宽。
4. 是否声明危险动作。
5. 是否进入企业 Registry。
6. 是否通过安全审查。

企业平台最好把 MCP Server 暴露的 tools 映射到内部 Tool Registry，再统一治理。

## 20.8 Function Calling 只关注 Tool，MCP 还关注 Resources 和 Prompts

传统 Function Calling 主要对象是 tool。

MCP 同时包含：

1. Tools。
2. Resources。
3. Prompts。

这很重要。

例如代码助手不仅需要运行测试工具，还需要读取：

1. 当前文件。
2. Git diff。
3. 项目配置。
4. 代码审查 prompt。

如果只有 function calling，这些资源和 prompt 需要另写机制。

MCP 把工具、资源、prompt 放到同一个 server 能力模型下。

## 20.9 生命周期不同

Function Calling 工具常常随 Host 应用发布。

MCP Server 可以独立部署、升级和重启。

这意味着：

1. Server 可以独立演进。
2. 多个 Host 可以复用同一 Server。
3. Server 版本需要治理。
4. Host 要处理 Server 不可用、工具变化、schema 变化。

MCP 让工具提供方和模型应用方更解耦，但也带来分布式系统复杂性。

## 20.10 连接方式不同

Function Calling 不规定工具从哪里来。工具可能是应用内函数、HTTP API、数据库查询。

MCP 明确规定 Client 和 Server 之间通过协议通信。

传输可以是：

1. stdio。
2. HTTP 类传输。
3. 其他支持的传输方式。

这让本地工具和远程工具可以用相似抽象接入。

例如本地 Git server 用 stdio，企业知识库 server 用远程 HTTP，Host 都通过 MCP Client 连接。

## 20.11 安全模型差异

Function Calling 的安全主要在 Host runtime：

1. tool choice。
2. 参数校验。
3. 权限。
4. executor sandbox。
5. tool result 包装。

MCP 增加了 Server 连接层安全：

1. 哪些 MCP Server 被允许连接。
2. Server 能访问哪些本地资源。
3. Server 暴露哪些 tools/resources/prompts。
4. Host 是否信任 Server 的 description 和 prompt。
5. Server 返回内容是否带 injection 风险。
6. 多 Server 之间数据流是否安全。

所以 MCP 安全不是更少，而是多了一层连接治理。

## 20.12 MCP Tool 可以被投影成 Function Calling Tool

实际系统中，MCP Tool 常常会被 Host 转成模型 API 的 tool。

流程：

```text
MCP Server exposes tool schema
  ↓
MCP Client gets tool list
  ↓
Host converts MCP tool to LLM provider tool format
  ↓
LLM emits function/tool call
  ↓
Host maps it back to MCP tool call
```

也就是说，MCP Tool 可以成为 Function Calling 的来源之一。

这也是为什么二者经常被混淆。

准确说：

```text
Function Calling 是模型调用表达层；MCP Tool 是外部能力发现和执行连接层。
```

## 20.13 Provider Adapter 与 MCP Adapter

如果系统同时支持多模型 provider 和 MCP，就会有两类 adapter。

Provider Adapter：

```text
内部 tool schema ↔ OpenAI / Anthropic / Gemini 等模型 API 格式
```

MCP Adapter：

```text
内部 ToolSpec / ToolCall / ToolResult ↔ MCP tools/list / tools/call 格式
```

企业平台可能流程如下：

```text
MCP Server tool
  ↓ MCP Adapter
Internal Tool Registry item
  ↓ Provider Adapter
LLM provider tool schema
```

这样可以把 MCP 工具纳入统一治理，而不是直接透传给模型。

## 20.14 错误处理差异

Function Calling 中，错误通常由 Host 包装成 tool result 回给模型。

MCP 中，Server 也可能返回协议层或工具层错误。

Host 要处理两类错误：

1. MCP 连接或协议错误。
2. 工具业务错误。

例如：

```text
MCP Server disconnected
```

和：

```text
get_weather(city) returned NOT_FOUND
```

这两者不同。前者是 server/transport 问题，后者是工具业务问题。

Host 应把错误规范化成模型可理解的 tool result，同时记录 trace。

## 20.15 性能和可用性差异

Function Calling 的工具如果是本地函数，调用延迟可能很低。

MCP Tool 需要经过 Client-Server 通信，可能增加：

1. 连接延迟。
2. 序列化开销。
3. 进程间通信或网络开销。
4. Server 冷启动。
5. 远程服务故障。

但 MCP 的优势是解耦和复用。

工程上要权衡：

1. 高频低延迟工具是否应该内置。
2. 复杂外部能力是否适合 MCP Server。
3. 是否需要连接池。
4. 是否缓存 tools list。
5. Server 不可用时如何降级。

## 20.16 什么时候用 Function Calling 就够了

如果系统很简单，Function Calling 就够了。

适用场景：

1. 工具数量少。
2. 工具只给单个应用使用。
3. 工具由应用团队维护。
4. 不需要跨客户端复用。
5. 没有 resources/prompts 生态需求。
6. 集成方式简单。

例如一个聊天机器人只有 `get_weather` 和 `search_faq` 两个工具，直接 function calling 很合理。

不要为了协议而协议。

## 20.17 什么时候 MCP 更合适

MCP 更适合：

1. 多个客户端复用同一工具能力。
2. 需要连接本地上下文。
3. 需要暴露 resources 和 prompts。
4. 工具由独立团队维护。
5. 需要插件生态。
6. 需要跨 IDE、桌面、CLI、企业平台复用。
7. 工具和 Host 需要解耦部署。

例如：

1. 文件系统 server。
2. Git server。
3. 数据库 server。
4. 企业知识库 server。
5. Issue tracker server。

这些都很适合做 MCP Server。

## 20.18 企业中如何同时使用二者

企业系统通常不是二选一，而是组合。

推荐架构：

```text
MCP Servers expose capabilities
  ↓
Enterprise Tool Platform imports them into Registry
  ↓
Router selects candidate tools
  ↓
Provider Adapter exposes tools via Function Calling
  ↓
Model emits tool calls
  ↓
Executor calls MCP Server or internal API
```

这样 MCP 负责连接生态，Function Calling 负责模型调用表达，企业平台负责治理。

三者分工清楚。

## 20.19 MCP Tool / Function Calling 对比指标与最小 demo

面试里常见的浅回答是：MCP Tool 也是一个 tool，Function Calling 也是一个 tool，所以差不多。这个回答的问题在于没有区分协议层次。可以把第 `i` 个对比样本记为：

```math
d_i=(l_i,f_i,u_i,e_i,p_i,a_i,v_i,g_i,s_i,q_i,z_i)
```

其中 `l_i` 表示协议层次是否清晰，`f_i` 表示工具来源和发现方式，`u_i` 表示 tools / resources / prompts 能力范围，`e_i` 表示执行边界，`p_i` 表示 MCP Tool 到 provider function calling tool 的投影映射，`a_i` 表示 Provider Adapter 和 MCP Adapter 是否分离，`v_i` 表示生命周期与版本意识，`g_i` 表示治理接入，`s_i` 表示安全边界，`q_i` 表示选型 trade-off，`z_i` 表示 trace、错误和可用性证据。

统一覆盖率可以写成：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[d_i\ \mathrm{passes}\ k]
```

这一章重点看这些门禁：

```math
C_{\mathrm{layer}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{model\ API\ layer\ and\ MCP\ layer\ are\ separated}]
```

```math
C_{\mathrm{discover}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{tool\ discovery\ boundary\ is\ clear}]
```

```math
C_{\mathrm{scope}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{tools\ resources\ prompts\ scope\ is\ covered}]
```

```math
C_{\mathrm{exec}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{execution\ boundary\ is\ explicit}]
```

```math
C_{\mathrm{proj}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{MCP\ tool\ projection\ is\ mapped}]
```

```math
C_{\mathrm{adapter}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{provider\ adapter\ and\ MCP\ adapter\ are\ separated}]
```

```math
C_{\mathrm{life}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{lifecycle\ and\ version\ change\ are\ handled}]
```

```math
C_{\mathrm{gov}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{MCP\ tools\ enter\ governance\ registry}]
```

```math
C_{\mathrm{safe}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{security\ boundary\ is\ enforced}]
```

```math
C_{\mathrm{choice}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{use\ case\ selection\ is\ justified}]
```

```math
C_{\mathrm{error}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{error\ surface\ is\ separated}]
```

```math
C_{\mathrm{avail}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{latency\ and\ availability\ tradeoff\ is\ considered}]
```

综合门禁：

```math
G_{\mathrm{mcp\_fc}}=\mathbf{1}\left[
\min_k C_k \ge \tau
\right]
```

这里 `\tau` 不是协议标准，而是教学审计阈值。回答 MCP 与 Function Calling 区别时，`C_{\mathrm{layer}}`、`C_{\mathrm{scope}}`、`C_{\mathrm{exec}}` 和 `C_{\mathrm{proj}}` 是最关键的四项：说不清这四项，基本就是把 MCP 当成另一种工具调用字段。

下面的 demo 不实现真实 MCP 或真实 provider API，只演示三件事：

1. MCP Tool 可以被 Host 投影成 provider tool。
2. 模型返回的 provider tool call 可以被 Host 映射回 MCP `tools/call`。
3. 对比二者时要同时检查协议层、发现层、能力层、执行层和治理层。

```python
from dataclasses import dataclass
from typing import Any


def project_mcp_tool_to_provider(server_name: str, tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f"{server_name}.{tool['name']}",
        "description": f"[from MCP server {server_name}] {tool['description']}",
        "parameters": tool["input_schema"],
        "source": {"protocol": "mcp", "server": server_name, "tool": tool["name"]},
    }


def map_provider_call_to_mcp(tool_call: dict[str, Any]) -> dict[str, Any]:
    server_name, tool_name = tool_call["name"].split(".", 1)
    return {
        "server": server_name,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": tool_call["arguments"]},
    }


def bridge_smoke_test() -> dict[str, Any]:
    mcp_server = {
        "name": "kb",
        "tools": [
            {
                "name": "search_docs",
                "description": "Search approved knowledge base documents.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ],
        "resources": ["kb://policy/refund"],
        "prompts": ["summarize_policy"],
    }
    provider_tool = project_mcp_tool_to_provider(mcp_server["name"], mcp_server["tools"][0])
    provider_call = {"name": "kb.search_docs", "arguments": {"query": "refund policy"}}
    mcp_call = map_provider_call_to_mcp(provider_call)
    return {
        "provider_tool_name": provider_tool["name"],
        "provider_tool_source": provider_tool["source"],
        "mcp_call": mcp_call,
        "has_resources": bool(mcp_server["resources"]),
        "has_prompts": bool(mcp_server["prompts"]),
        "adapter_layers": ["mcp_adapter", "tool_registry", "provider_adapter"],
    }


@dataclass
class CompareCase:
    name: str
    protocol_layer: bool = True
    discovery_boundary: bool = True
    capability_scope: bool = True
    execution_boundary: bool = True
    projection_mapping: bool = True
    adapter_separation: bool = True
    lifecycle_version: bool = True
    governance_registry: bool = True
    security_boundary: bool = True
    use_case_selection: bool = True
    error_surface: bool = True
    latency_availability: bool = True


CASES = [
    CompareCase("simple_function_calling_ok"),
    CompareCase("mcp_discovery_ok"),
    CompareCase("resources_prompts_ok"),
    CompareCase("projection_chain_ok"),
    CompareCase("adapter_registry_ok"),
    CompareCase("fc_equals_mcp_bad", protocol_layer=False, capability_scope=False),
    CompareCase("mcp_replaces_fc_bad", projection_mapping=False, protocol_layer=False),
    CompareCase("server_direct_to_model_bad", protocol_layer=False, execution_boundary=False),
    CompareCase("discovery_auto_trust_bad", discovery_boundary=False, governance_registry=False, security_boundary=False),
    CompareCase("resources_ignored_bad", capability_scope=False),
    CompareCase("schema_trust_bad", governance_registry=False, security_boundary=False),
    CompareCase("no_registry_import_bad", adapter_separation=False, governance_registry=False),
    CompareCase("adapter_mixed_bad", adapter_separation=False, projection_mapping=False),
    CompareCase("lifecycle_ignored_bad", lifecycle_version=False, error_surface=False, latency_availability=False),
    CompareCase("latency_tradeoff_ignored_bad", use_case_selection=False, latency_availability=False),
    CompareCase("full_boundary_ready_ok"),
]


METRIC_FIELDS = {
    "protocol_layer_clarity": "protocol_layer",
    "tool_discovery_boundary": "discovery_boundary",
    "capability_scope_coverage": "capability_scope",
    "execution_boundary_clarity": "execution_boundary",
    "projection_mapping_coverage": "projection_mapping",
    "adapter_separation_coverage": "adapter_separation",
    "lifecycle_version_awareness": "lifecycle_version",
    "governance_registry_import": "governance_registry",
    "security_boundary_enforcement": "security_boundary",
    "use_case_selection_fit": "use_case_selection",
    "error_surface_separation": "error_surface",
    "latency_availability_tradeoff": "latency_availability",
}


def ratio(values: list[bool]) -> float:
    return round(sum(values) / len(values), 3)


def audit_compare_cases(cases: list[CompareCase], threshold: float = 0.9) -> dict[str, Any]:
    metrics = {
        metric: ratio([getattr(case, field) for case in cases])
        for metric, field in METRIC_FIELDS.items()
    }
    failed_cases = [
        case.name
        for case in cases
        if not all(getattr(case, field) for field in METRIC_FIELDS.values())
    ]
    failed_gates = [metric for metric, value in metrics.items() if value < threshold]
    return {
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "mcp_function_calling_gate_pass": not failed_gates,
    }


print("bridge=", bridge_smoke_test())
report = audit_compare_cases(CASES)
print("metrics=", report["metrics"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("mcp_function_calling_gate_pass=", report["mcp_function_calling_gate_pass"])
```

一组输出示例：

```text
bridge= {'provider_tool_name': 'kb.search_docs', 'provider_tool_source': {'protocol': 'mcp', 'server': 'kb', 'tool': 'search_docs'}, 'mcp_call': {'server': 'kb', 'method': 'tools/call', 'params': {'name': 'search_docs', 'arguments': {'query': 'refund policy'}}}, 'has_resources': True, 'has_prompts': True, 'adapter_layers': ['mcp_adapter', 'tool_registry', 'provider_adapter']}
metrics= {'protocol_layer_clarity': 0.812, 'tool_discovery_boundary': 0.938, 'capability_scope_coverage': 0.875, 'execution_boundary_clarity': 0.938, 'projection_mapping_coverage': 0.875, 'adapter_separation_coverage': 0.875, 'lifecycle_version_awareness': 0.938, 'governance_registry_import': 0.812, 'security_boundary_enforcement': 0.875, 'use_case_selection_fit': 0.938, 'error_surface_separation': 0.938, 'latency_availability_tradeoff': 0.875}
failed_cases= ['fc_equals_mcp_bad', 'mcp_replaces_fc_bad', 'server_direct_to_model_bad', 'discovery_auto_trust_bad', 'resources_ignored_bad', 'schema_trust_bad', 'no_registry_import_bad', 'adapter_mixed_bad', 'lifecycle_ignored_bad', 'latency_tradeoff_ignored_bad']
failed_gates= ['protocol_layer_clarity', 'capability_scope_coverage', 'projection_mapping_coverage', 'adapter_separation_coverage', 'governance_registry_import', 'security_boundary_enforcement', 'latency_availability_tradeoff']
mcp_function_calling_gate_pass= False
```

这个 demo 的关键不是 `kb.search_docs` 这个命名，而是桥接责任：MCP Adapter 负责把 MCP Server 暴露的能力导入内部 Registry，Provider Adapter 负责把内部 ToolSpec 投影成模型 API 需要的 tool schema。模型输出的 tool call 仍要由 Host 映射回 MCP `tools/call`，并经过权限、安全、trace 和错误规范化。

## 20.20 常见误区

### 20.20.1 MCP 替代 Function Calling

不准确。MCP Tool 常常仍需要被 Host 投影成模型 API 的 function/tool call。

### 20.20.2 Function Calling 替代 MCP

也不准确。Function Calling 不解决跨客户端能力发现、resources、prompts 和本地 server 生态问题。

### 20.20.3 MCP Server 暴露工具后模型自动安全

错误。Host 和企业平台仍要做权限、安全、确认和审计。

### 20.20.4 MCP 一定更好

不一定。小系统直接 function calling 更简单，MCP 引入了 server、transport 和连接治理复杂度。

### 20.20.5 MCP Tool 不需要 Registry

企业场景不建议。MCP Tool 应映射到 Registry 统一治理。

## 20.21 面试题：MCP Tool 与 Function Calling 的区别

面试官可能问：

```text
MCP Tool 和 OpenAI/Anthropic 这类 Function Calling 有什么区别？
```

可以这样回答：

第一，层次不同：

1. Function Calling 是 Host 和模型之间的工具调用表达。
2. MCP 是 Host/Client 和外部 Server 之间的上下文连接协议。

第二，发现方式不同：

1. Function Calling 工具通常由 Host 静态传给模型。
2. MCP Client 可以从 MCP Server 动态发现 tools/resources/prompts。

第三，能力范围不同：

1. Function Calling 主要关注 tools。
2. MCP 同时关注 tools、resources、prompts。

第四，执行边界不同：

1. Function Calling 通常由 Host 执行函数或 API。
2. MCP Tool 由 Host 通过 MCP Client 调用 MCP Server 执行。

第五，生态目标不同：

1. Function Calling 让模型结构化表达调用意图。
2. MCP 让外部能力跨客户端、跨工具生态复用。

第六，二者可以组合：

1. MCP Server 暴露 tool。
2. Host 把它投影成模型 function calling tool。
3. 模型生成 tool call。
4. Host 再调用 MCP Server。

一句话总结：

```text
Function Calling 是模型调用工具的语言，MCP 是工具和上下文接入模型应用的连接协议。
```

## 20.22 小练习

### 练习 1：协议层次

模型返回 `tool_call` 是 MCP 协议还是 Function Calling 协议？

参考答案：通常是模型 provider 的 Function Calling 协议。Host 可能再把它映射成 MCP tool call。

### 练习 2：工具发现

MCP Client 如何知道 Server 暴露了哪些工具？

参考答案：通过 MCP 的 tools list / capability discovery，而不是 Host 必须提前硬编码所有工具。

### 练习 3：资源能力

Function Calling 是否原生解决 resources 和 prompts？

参考答案：通常不解决。MCP 原生把 tools、resources、prompts 放在同一协议生态中。

### 练习 4：小系统选型

一个机器人只有两个内部工具，只服务一个产品。是否一定要 MCP？

参考答案：不一定。直接 Function Calling 可能更简单。

### 练习 5：企业组合架构

企业平台接入 MCP Server 后，是否应该绕过 Tool Registry？

参考答案：不应该。应把 MCP tools 映射进 Registry，统一做权限、安全、trace、eval 和发布治理。

## 20.23 本章小结

本章讲了 MCP Tool 与传统 Function Calling 的区别。

你需要掌握：

1. Function Calling 是模型 API 层工具调用表达。
2. MCP Tool 是 MCP Server 暴露给 MCP Client 的能力对象。
3. Function Calling 位于 Host 与模型之间，MCP 位于 Host/Client 与 Server 之间。
4. MCP 支持能力发现，Function Calling 通常由 Host 传入 tools。
5. MCP 不只包含 tools，还包含 resources 和 prompts。
6. MCP Tool 可以被 Host 投影成 Function Calling tool。
7. MCP 增加了 Server 连接层的安全和可用性问题。
8. 小系统可直接用 Function Calling，大生态、多客户端、本地上下文更适合 MCP。
9. 企业场景中，MCP、Function Calling 和 Tool Platform 应组合使用。

如果只记一句话：

```text
MCP 不是 Function Calling 的同义词，而是 Function Calling 上游的工具和上下文连接层；二者组合起来，才能形成完整的模型工具生态。
```

下一章会讲 MCP Resources：文件、数据库、网页和上下文资源暴露，重点解释资源如何安全进入模型上下文。
