# 第十九章：MCP Server 的最小实现

## 19.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，参考 MCP 官方介绍、MCP 2025-06-18 specification 中 lifecycle、tools、resources、prompts、stdio / Streamable HTTP transport 的口径，以及 OpenAI Agents SDK 中 MCP server 接入、strict schema、tool filtering、approval 与 tracing 的工程抽象。正文只抽象最小 MCP Server 的稳定实现闭环，不绑定某个 SDK 的类名、装饰器、配置文件格式或某个 Host 产品的私有字段。

本章新增的公式和 demo 只用于面试与教学：公式用来把“最小实现是否完整”拆成可检查指标，代码用一个 0 依赖 toy server 演示 metadata、capabilities、tools/list、tools/call、schema validation、resources、prompts、transport policy、结构化错误和 trace 的最小闭环。真实项目中应优先使用官方 SDK 和当前协议版本，并把远程 server 的认证、授权、TLS、限流、审计和 token audience 校验放入生产治理。

## 19.1 本章定位

前两章讲了 MCP 的背景和基本概念。本章进入实现层：一个最小 MCP Server 到底需要做什么。

本章不会绑定某个具体 SDK 的全部细节，而是讲清楚最小 server 的共同结构：

1. 启动一个 server。
2. 声明 server 能力。
3. 暴露 tools。
4. 可选暴露 resources 和 prompts。
5. 接收 tool call。
6. 校验参数。
7. 执行真实逻辑。
8. 返回结构化结果或错误。
9. 被 Host / Client 连接和使用。

本章的核心观点是：

```text
MCP Server 的最小实现不是“写一个函数”，而是把外部能力包装成可发现、可描述、可调用、可返回错误的协议服务。
```

## 19.2 最小 MCP Server 的组成

一个最小 MCP Server 通常包含四部分。

第一，server metadata：

```text
server name、version、capabilities
```

第二，tools registry：

```text
有哪些 tools，每个 tool 的 name、description、input schema 是什么。
```

第三，tool handlers：

```text
每个 tool 被调用时，执行什么逻辑。
```

第四，transport：

```text
Client 如何连接 server，例如 stdio 或 HTTP。
```

如果只做一个最小 demo，通常只需要暴露一个 tool，例如 `get_weather` 或 `search_files`。

## 19.3 最小工具例子：天气查询

假设我们要做一个 MCP Server，暴露一个工具：

```text
get_weather(city)
```

工具定义包含：

1. name：`get_weather`。
2. description：查询指定城市天气。
3. input schema：city 是必填字符串。
4. handler：根据 city 返回天气。

概念性定义：

```json
{
  "name": "get_weather",
  "description": "查询指定城市的当前天气。",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名，例如北京、上海、深圳"
      }
    },
    "required": ["city"],
    "additionalProperties": false
  }
}
```

handler 的逻辑：

```python
def get_weather(city: str):
    return {
        "city": city,
        "condition": "晴",
        "temperature_c": 22,
    }
```

真实项目里 handler 可能调用天气 API，但最小实现可以先返回 mock 数据。

## 19.4 Server 初始化

MCP Server 启动时要告诉 Client：自己是谁，支持什么能力。

概念上包括：

```json
{
  "server_info": {
    "name": "weather-mcp-server",
    "version": "0.1.0"
  },
  "capabilities": {
    "tools": true,
    "resources": false,
    "prompts": false
  }
}
```

如果 server 只暴露 tools，就不必实现 resources 和 prompts。

最小实现的原则是：

```text
只声明自己真正支持的能力，不要虚报。
```

否则 Host 可能调用未实现的方法，导致体验和调试都变差。

## 19.5 Tools List

Client 连接后，通常会请求 tools 列表。

Server 返回：

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "查询指定城市的当前天气。",
      "input_schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"}
        },
        "required": ["city"],
        "additionalProperties": false
      }
    }
  ]
}
```

这一步很关键，因为 Host 会根据 tools list 决定是否把工具提供给模型。

如果 description 写得太差，模型可能不会正确调用。

## 19.6 Tool Call

当模型决定调用工具时，Host 会通过 MCP Client 调用 MCP Server。

概念请求：

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "北京"
  }
}
```

Server 收到后要做：

1. 检查工具名是否存在。
2. 校验 arguments。
3. 执行 handler。
4. 包装结果。
5. 返回给 Client。

不要因为 arguments 来自模型就直接执行。即使 MCP Server 很小，也应该做 schema validation。

## 19.7 Tool Result

工具成功时返回结构化内容。

概念结果：

```json
{
  "content": [
    {
      "type": "text",
      "text": "北京当前天气晴，22 摄氏度。"
    }
  ]
}
```

也可以返回结构化 JSON 的文本表示：

```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"city\":\"北京\",\"condition\":\"晴\",\"temperature_c\":22}"
    }
  ]
}
```

具体格式取决于 SDK 和协议版本，但核心是：结果要让 Host 能够回填给模型。

生产系统中，更推荐让 Host 再做一次安全包装、脱敏、压缩和 trace。

## 19.8 错误返回

工具失败时，不要只抛异常崩掉 server。

应该返回结构化错误。

常见错误：

1. unknown tool。
2. invalid arguments。
3. permission denied。
4. upstream timeout。
5. upstream error。
6. internal error。

示例：

```json
{
  "error": {
    "code": "INVALID_ARGUMENTS",
    "message": "city is required",
    "retryable": false
  }
}
```

错误信息要脱敏，不要暴露内部堆栈、密钥、数据库地址。

## 19.9 最小 Server 的伪代码

下面是一个与具体 SDK 无关的伪代码。

```python
class MCPServer:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, description, input_schema, handler):
        self.tools[name] = {
            "description": description,
            "input_schema": input_schema,
            "handler": handler,
        }

    def list_tools(self):
        return [
            {
                "name": name,
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
            for name, spec in self.tools.items()
        ]

    def call_tool(self, name, arguments):
        if name not in self.tools:
            return error("UNKNOWN_TOOL", "tool not found")

        spec = self.tools[name]
        validation = validate(arguments, spec["input_schema"])
        if not validation.ok:
            return error("INVALID_ARGUMENTS", validation.message)

        try:
            result = spec["handler"](**arguments)
            return success(result)
        except TimeoutError:
            return error("TOOL_TIMEOUT", "tool timed out", retryable=True)
        except Exception:
            return error("INTERNAL_ERROR", "tool failed")
```

真实 SDK 会帮你处理协议细节，但这个伪代码体现了最小 server 的结构。

## 19.10 最小实现不等于生产实现

最小实现可以只有一个 tool 和一个 handler。

生产实现还需要：

1. 参数校验。
2. 权限检查。
3. 超时。
4. 限流。
5. 错误规范化。
6. 日志和 trace。
7. 结果脱敏。
8. 安全边界。
9. 版本管理。
10. 测试和 eval。

面试时要明确区分：

```text
最小 demo 展示协议闭环，生产 server 需要治理能力。
```

## 19.11 暴露 Resources 的最小实现

如果 server 暴露 resources，最小需要两个能力：

1. list resources。
2. read resource。

例如文件资源：

```json
{
  "uri": "file:///project/README.md",
  "name": "README.md",
  "description": "项目说明文件"
}
```

读取资源：

```json
{
  "uri": "file:///project/README.md"
}
```

返回内容：

```json
{
  "contents": [
    {
      "uri": "file:///project/README.md",
      "mimeType": "text/markdown",
      "text": "# Project..."
    }
  ]
}
```

资源 server 必须注意访问范围。文件系统 server 不能随便读取整个磁盘。

## 19.12 暴露 Prompts 的最小实现

如果 server 暴露 prompts，最小需要：

1. list prompts。
2. get prompt。

Prompt 示例：

```json
{
  "name": "summarize_document",
  "description": "总结指定文档，并列出关键结论。",
  "arguments": [
    {
      "name": "document_uri",
      "description": "要总结的文档 URI",
      "required": true
    }
  ]
}
```

获取 prompt 后，server 可以返回消息模板：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "请总结文档 {{document_uri}}，并列出关键结论和风险。"
    }
  ]
}
```

Host 应决定如何把 prompt 交给模型，而不是 server 直接控制模型。

## 19.13 传输方式选择

最小本地 server 常用 stdio，因为简单：

1. Host 启动 server 进程。
2. Client 通过标准输入输出通信。
3. Server 不需要开放网络端口。

适合：

1. 本地文件系统。
2. Git。
3. IDE 工具。
4. 个人开发环境。

远程 server 可以使用 HTTP 类传输。

适合：

1. 企业共享服务。
2. 云端工具。
3. 多用户系统。

传输方式影响认证和安全。stdio 依赖本地进程边界，远程 HTTP 需要认证、授权、TLS、限流和审计。

## 19.14 Host 如何接入最小 Server

Host 接入 server 大致需要：

1. 配置 server 启动方式或 endpoint。
2. 建立 MCP Client 连接。
3. 初始化协议。
4. 列出 tools/resources/prompts。
5. 按策略过滤能力。
6. 把 tools 投影给模型。
7. 接收模型 tool call。
8. 调用 MCP Server。
9. 回填结果给模型。

配置示意：

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"]
    }
  }
}
```

不同 Host 的配置格式可能不同，但逻辑类似。

## 19.15 最小实现的测试

写完 server 后，至少要测试：

1. server 能启动。
2. client 能连接。
3. initialize 成功。
4. tools/list 返回正确 schema。
5. tools/call 正常返回结果。
6. 参数缺失返回错误。
7. 未知工具返回错误。
8. handler 异常不会崩掉 server。
9. 结果格式 Host 能识别。

如果有 resources，还要测试：

1. resources/list。
2. resources/read。
3. 越界 URI 被拒绝。

如果有 prompts，还要测试：

1. prompts/list。
2. prompts/get。
3. 参数缺失处理。

## 19.16 最小实现中的安全底线

即使是最小 server，也要有安全底线。

1. 不暴露不必要工具。
2. input schema 要严格。
3. 禁止额外字段。
4. 不执行任意命令。
5. 文件路径限制在根目录。
6. 错误脱敏。
7. 输出大小限制。
8. 不返回密钥和敏感环境变量。
9. 对写操作要求确认或不实现。

很多事故不是复杂系统才会发生，demo server 也可能读错文件、泄露环境变量或执行危险命令。

## 19.17 MCP Server 最小实现指标与最小 demo

面试中只说“注册一个函数”是不够的。一个最小 MCP Server 至少要证明它能被初始化、能声明能力、能列出工具、能校验参数、能执行 handler、能返回结构化结果或结构化错误，并且可选的 resources / prompts 不越过 Host 的治理边界。

可以把第 `i` 个最小 server 实现样本记为：

```math
r_i=(m_i,c_i,t_i,s_i,h_i,a_i,o_i,e_i,p_i,z_i)
```

其中 `m_i` 表示 metadata 和 version，`c_i` 表示 capabilities，`t_i` 表示 tools registry，`s_i` 表示 input schema，`h_i` 表示 handler 执行，`a_i` 表示 arguments validation，`o_i` 表示 output contract，`e_i` 表示 error contract，`p_i` 表示 prompts / resources / transport policy，`z_i` 表示安全、trace 和 Host 接入证据。

对任意检查项 `k`，覆盖率可以写成：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[r_i\ \mathrm{passes}\ k]
```

最小 server 可以进一步拆成这些门禁：

```math
C_{\mathrm{meta}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[m_i=1]
```

```math
C_{\mathrm{cap}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[c_i=1]
```

```math
C_{\mathrm{tool}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[t_i=1]
```

```math
C_{\mathrm{schema}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[s_i=1]
```

```math
C_{\mathrm{handler}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[h_i=1]
```

```math
C_{\mathrm{arg}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[a_i=1]
```

```math
C_{\mathrm{result}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[o_i=1]
```

```math
C_{\mathrm{error}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[e_i=1]
```

```math
C_{\mathrm{resource}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{resource\ scope\ is\ bounded}]
```

```math
C_{\mathrm{prompt}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{prompt\ template\ is\ reviewed}]
```

```math
C_{\mathrm{transport}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{transport\ policy\ is\ explicit}]
```

```math
C_{\mathrm{host}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{host\ connection\ is\ tested}]
```

```math
C_{\mathrm{safety}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{safety\ baseline\ is\ enforced}]
```

```math
C_{\mathrm{trace}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{trace\ fields\ are\ captured}]
```

综合门禁可以写成：

```math
G_{\mathrm{mcp\_server}}=\mathbf{1}\left[
\min_k C_k \ge \tau
\right]
```

这里的 `\tau` 是上线阈值。教学 demo 可以用 `0.95` 作为严格阈值；真实项目要按工具风险、传输方式、是否有写操作、是否接入企业权限系统来设置不同阈值。

下面的 demo 演示一个不依赖 SDK 的 toy MCP Server。它不是 MCP 协议实现，只用于帮助理解最小 server 要有哪些结构，以及为什么 bad case 不能因为“能跑通一个 handler”就算合格。

```python
from dataclasses import dataclass
from typing import Any, Callable


class ValidationError(Exception):
    pass


def validate_object_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValidationError("schema type must be object")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            raise ValidationError(f"{key} is required")

    if schema.get("additionalProperties") is False:
        extra = set(arguments) - set(properties)
        if extra:
            raise ValidationError(f"unexpected fields: {sorted(extra)}")

    for key, value in arguments.items():
        spec = properties.get(key)
        if spec is None:
            continue
        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(value, str):
            raise ValidationError(f"{key} must be string")
        if expected_type == "integer" and not isinstance(value, int):
            raise ValidationError(f"{key} must be integer")


class MiniMCPServer:
    def __init__(self, name: str, version: str, transport: str) -> None:
        self.server_info = {"name": name, "version": version}
        self.transport = transport
        self.capabilities = {"tools": True, "resources": False, "prompts": False}
        self.tools: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, str] = {}
        self.prompts: dict[str, dict[str, Any]] = {}
        self.trace: list[dict[str, Any]] = []

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., dict[str, Any]],
    ) -> None:
        self.tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler": handler,
        }

    def add_resource(self, uri: str, text: str) -> None:
        self.capabilities["resources"] = True
        self.resources[uri] = text

    def add_prompt(self, name: str, template: str, reviewed: bool) -> None:
        self.capabilities["prompts"] = True
        self.prompts[name] = {"name": name, "template": template, "reviewed": reviewed}

    def initialize(self) -> dict[str, Any]:
        return {"server_info": self.server_info, "capabilities": self.capabilities}

    def list_tools(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["input_schema"],
                }
                for tool in self.tools.values()
            ]
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.trace.append({"event": "tools/call", "name": name, "args": sorted(arguments)})
        if name not in self.tools:
            return self.error("UNKNOWN_TOOL", "tool not found", retryable=False)

        tool = self.tools[name]
        try:
            validate_object_schema(arguments, tool["input_schema"])
            result = tool["handler"](**arguments)
            return {"content": [{"type": "text", "text": str(result)}], "structured": result}
        except ValidationError as exc:
            return self.error("INVALID_ARGUMENTS", str(exc), retryable=False)
        except TimeoutError:
            return self.error("TOOL_TIMEOUT", "tool timed out", retryable=True)
        except Exception:
            return self.error("INTERNAL_ERROR", "tool failed", retryable=False)

    def list_resources(self) -> dict[str, Any]:
        return {"resources": [{"uri": uri, "name": uri.rsplit("/", 1)[-1]} for uri in self.resources]}

    def read_resource(self, uri: str) -> dict[str, Any]:
        if not uri.startswith("file:///project/"):
            return self.error("RESOURCE_OUT_OF_SCOPE", "resource is outside root", retryable=False)
        if uri not in self.resources:
            return self.error("RESOURCE_NOT_FOUND", "resource not found", retryable=False)
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": self.resources[uri]}]}

    def list_prompts(self) -> dict[str, Any]:
        return {"prompts": [{"name": name, "reviewed": spec["reviewed"]} for name, spec in self.prompts.items()]}

    def get_prompt(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        prompt = self.prompts.get(name)
        if prompt is None:
            return self.error("PROMPT_NOT_FOUND", "prompt not found", retryable=False)
        if not prompt["reviewed"]:
            return self.error("PROMPT_UNREVIEWED", "prompt is not reviewed", retryable=False)
        text = prompt["template"].format(**arguments)
        return {"messages": [{"role": "user", "content": text}]}

    @staticmethod
    def error(code: str, message: str, retryable: bool) -> dict[str, Any]:
        return {"error": {"code": code, "message": message, "retryable": retryable}}


def weather_handler(city: str) -> dict[str, Any]:
    if city == "timeout":
        raise TimeoutError()
    return {"city": city, "condition": "sunny", "temperature_c": 22}


def build_demo_server() -> MiniMCPServer:
    server = MiniMCPServer("weather-mcp-server", "0.1.0", transport="stdio")
    server.register_tool(
        "get_weather",
        "Return toy weather for one city. Use for demo only.",
        {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        weather_handler,
    )
    server.add_resource("file:///project/README.md", "demo project")
    server.add_prompt("summarize_resource", "Summarize {uri} in three bullets.", reviewed=True)
    return server


@dataclass
class ServerCase:
    name: str
    metadata: bool = True
    capabilities: bool = True
    tool_registry: bool = True
    strict_schema: bool = True
    handler: bool = True
    argument_validation: bool = True
    structured_result: bool = True
    structured_error: bool = True
    resource_scope: bool = True
    prompt_template: bool = True
    transport_policy: bool = True
    host_connection: bool = True
    safety_baseline: bool = True
    trace_ready: bool = True


CASES = [
    ServerCase("metadata_capabilities_ok"),
    ServerCase("list_tools_ok"),
    ServerCase("call_weather_ok"),
    ServerCase("invalid_args_block_ok"),
    ServerCase("unknown_tool_error_ok"),
    ServerCase("handler_exception_mapped_ok"),
    ServerCase("resource_scope_ok"),
    ServerCase("prompt_template_ok"),
    ServerCase("missing_schema_bad", strict_schema=False, argument_validation=False),
    ServerCase("additional_props_allowed_bad", strict_schema=False),
    ServerCase("no_structured_error_bad", structured_error=False),
    ServerCase("root_escape_resource_bad", resource_scope=False, safety_baseline=False),
    ServerCase("prompt_auto_trusted_bad", prompt_template=False, safety_baseline=False),
    ServerCase("remote_no_auth_bad", transport_policy=False, safety_baseline=False),
    ServerCase("no_trace_bad", trace_ready=False),
    ServerCase("full_server_ready_ok"),
]


METRIC_FIELDS = {
    "metadata_readiness": "metadata",
    "capability_declaration": "capabilities",
    "tool_registry_readiness": "tool_registry",
    "strict_schema_coverage": "strict_schema",
    "handler_execution_coverage": "handler",
    "argument_validation_coverage": "argument_validation",
    "structured_result_coverage": "structured_result",
    "structured_error_coverage": "structured_error",
    "resource_scope_coverage": "resource_scope",
    "prompt_template_coverage": "prompt_template",
    "transport_policy_coverage": "transport_policy",
    "host_connection_readiness": "host_connection",
    "safety_baseline_coverage": "safety_baseline",
    "trace_readiness": "trace_ready",
}


def ratio(values: list[bool]) -> float:
    return round(sum(values) / len(values), 3)


def audit_cases(cases: list[ServerCase], threshold: float = 0.95) -> dict[str, Any]:
    metrics = {
        name: ratio([getattr(case, field) for case in cases])
        for name, field in METRIC_FIELDS.items()
    }
    failed_cases = [
        case.name
        for case in cases
        if not all(getattr(case, field) for field in METRIC_FIELDS.values())
    ]
    failed_gates = [name for name, value in metrics.items() if value < threshold]
    return {
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "mcp_server_gate_pass": not failed_gates,
    }


def protocol_smoke_test() -> dict[str, Any]:
    server = build_demo_server()
    init = server.initialize()
    tools = server.list_tools()["tools"]
    success = server.call_tool("get_weather", {"city": "beijing"})
    invalid = server.call_tool("get_weather", {})
    unknown = server.call_tool("unknown", {"city": "beijing"})
    timeout = server.call_tool("get_weather", {"city": "timeout"})
    resource_ok = server.read_resource("file:///project/README.md")
    resource_escape = server.read_resource("file:///etc/passwd")
    prompt = server.get_prompt("summarize_resource", {"uri": "file:///project/README.md"})
    return {
        "server": init["server_info"],
        "capabilities": init["capabilities"],
        "tool_names": [tool["name"] for tool in tools],
        "weather_status": "structured" in success,
        "invalid_error": invalid["error"]["code"],
        "unknown_error": unknown["error"]["code"],
        "timeout_retryable": timeout["error"]["retryable"],
        "resource_ok": "contents" in resource_ok,
        "resource_escape_error": resource_escape["error"]["code"],
        "prompt_messages": len(prompt["messages"]),
        "trace_events": len(server.trace),
    }


print("protocol_smoke=", protocol_smoke_test())
report = audit_cases(CASES)
print("metrics=", report["metrics"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("mcp_server_gate_pass=", report["mcp_server_gate_pass"])
```

一组输出示例：

```text
protocol_smoke= {'server': {'name': 'weather-mcp-server', 'version': '0.1.0'}, 'capabilities': {'tools': True, 'resources': True, 'prompts': True}, 'tool_names': ['get_weather'], 'weather_status': True, 'invalid_error': 'INVALID_ARGUMENTS', 'unknown_error': 'UNKNOWN_TOOL', 'timeout_retryable': True, 'resource_ok': True, 'resource_escape_error': 'RESOURCE_OUT_OF_SCOPE', 'prompt_messages': 1, 'trace_events': 4}
metrics= {'metadata_readiness': 1.0, 'capability_declaration': 1.0, 'tool_registry_readiness': 1.0, 'strict_schema_coverage': 0.875, 'handler_execution_coverage': 1.0, 'argument_validation_coverage': 0.938, 'structured_result_coverage': 1.0, 'structured_error_coverage': 0.938, 'resource_scope_coverage': 0.938, 'prompt_template_coverage': 0.938, 'transport_policy_coverage': 0.938, 'host_connection_readiness': 1.0, 'safety_baseline_coverage': 0.812, 'trace_readiness': 0.938}
failed_cases= ['missing_schema_bad', 'additional_props_allowed_bad', 'no_structured_error_bad', 'root_escape_resource_bad', 'prompt_auto_trusted_bad', 'remote_no_auth_bad', 'no_trace_bad']
failed_gates= ['strict_schema_coverage', 'argument_validation_coverage', 'structured_error_coverage', 'resource_scope_coverage', 'prompt_template_coverage', 'transport_policy_coverage', 'safety_baseline_coverage', 'trace_readiness']
mcp_server_gate_pass= False
```

这个 demo 要传达的不是“自己手写 MCP 协议”，而是：哪怕只做一个天气查询 server，也要有可发现能力、严格 schema、结构化错误、resource 范围、prompt 审查、transport 策略、Host 接入和 trace。否则它只是一个本地函数，不是可治理的 MCP Server。

## 19.18 常见错误

### 19.18.1 只写 handler，不写 schema

问题：模型不知道如何生成参数，runtime 也无法校验。

修复：每个 tool 必须有 input schema。

### 19.18.2 description 过短

问题：Host 投影给模型后，模型无法正确选择工具。

修复：写清适用场景、不适用场景和参数含义。

### 19.18.3 handler 抛异常导致 server 崩溃

问题：一个工具失败影响整个 server。

修复：捕获异常并返回结构化错误。

### 19.18.4 文件 resource 无范围限制

问题：可能读取用户整个磁盘。

修复：使用 roots / workspace 限制。

### 19.18.5 把 MCP Server 当安全边界全部

问题：Host 仍需要权限、确认、trace 和安全策略。

修复：server 最小权限，Host 统一治理。

### 19.18.6 返回超长结果

问题：撑爆模型上下文。

修复：限制大小、分页、摘要或返回引用。

### 19.18.7 远程 server 无认证

问题：任何人都能调用工具。

修复：认证、授权、TLS、限流和审计。

## 19.19 面试题：如何实现一个最小 MCP Server

面试官可能问：

```text
如果让你实现一个最小 MCP Server，你会做哪些东西？
```

可以这样回答：

第一，定义 server metadata 和 capabilities。

第二，注册至少一个 tool，包括 name、description、input schema。

第三，实现 tool handler，接收 arguments，做参数校验，执行逻辑，返回结构化结果。

第四，实现错误处理，包括 unknown tool、invalid arguments、timeout、internal error。

第五，选择 transport。开发本地工具可用 stdio，远程共享工具可用 HTTP 类传输。

第六，接入 Host：Host 通过 MCP Client 连接 server，列出 tools，把工具投影给模型，模型发起调用后再通过 Client 调用 server。

第七，生产化需要补权限、限流、日志、trace、输出限制、版本和安全。

一句话总结：

```text
最小 MCP Server 要能被 Client 发现能力、列出工具、校验并执行工具调用、返回结果或结构化错误；生产 Server 还要补齐权限、安全和观测。
```

## 19.20 小练习

### 练习 1：最小 tool 定义

一个 `get_weather(city)` tool 至少需要哪些字段？

参考答案：name、description、input_schema，以及对应 handler。

### 练习 2：参数缺失

模型调用 `get_weather` 但没有传 `city`。Server 应该怎么做？

参考答案：返回结构化 `INVALID_ARGUMENTS` 错误，而不是执行默认城市或崩溃。

### 练习 3：资源范围

文件系统 MCP Server 能否默认读取整个磁盘？

参考答案：不能。应限制 roots / workspace，只暴露授权范围内资源。

### 练习 4：stdio 适合什么场景

stdio transport 适合本地还是远程共享服务？

参考答案：更适合本地 server，例如文件系统、Git、IDE 工具。远程共享服务更需要 HTTP 类传输和认证授权。

### 练习 5：生产化补什么

最小 MCP Server 上线生产前至少要补什么？

参考答案：权限、参数校验、错误规范化、超时、限流、日志 trace、输出限制、脱敏、安全审计和版本管理。

## 19.21 本章小结

本章讲了 MCP Server 的最小实现。

你需要掌握：

1. 最小 MCP Server 需要 server metadata、capabilities、tools list、tool handlers 和 transport。
2. Tool 必须有 name、description 和 input schema。
3. Server 收到 tool call 后要检查工具名、校验参数、执行 handler、返回结果或错误。
4. Resources 最小实现需要 list 和 read。
5. Prompts 最小实现需要 list 和 get。
6. stdio 适合本地工具，HTTP 类传输适合远程共享服务。
7. Host 通过 MCP Client 连接 Server，并决定哪些能力暴露给模型。
8. 最小实现不等于生产实现，生产还需要权限、安全、trace、限流、输出限制和版本管理。

如果只记一句话：

```text
MCP Server 的最小闭环是“声明能力 → 暴露工具 schema → 接收调用 → 校验执行 → 返回结果”，而不是只写一个裸函数。
```

下一章会讲 MCP Tool 与传统 Function Calling 的区别，重点解释 MCP tool 在协议位置、发现机制、执行边界和治理方式上的不同。
