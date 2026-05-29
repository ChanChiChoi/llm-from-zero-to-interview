# 第十九章：MCP Server 的最小实现

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

## 19.17 常见错误

### 19.17.1 只写 handler，不写 schema

问题：模型不知道如何生成参数，runtime 也无法校验。

修复：每个 tool 必须有 input schema。

### 19.17.2 description 过短

问题：Host 投影给模型后，模型无法正确选择工具。

修复：写清适用场景、不适用场景和参数含义。

### 19.17.3 handler 抛异常导致 server 崩溃

问题：一个工具失败影响整个 server。

修复：捕获异常并返回结构化错误。

### 19.17.4 文件 resource 无范围限制

问题：可能读取用户整个磁盘。

修复：使用 roots / workspace 限制。

### 19.17.5 把 MCP Server 当安全边界全部

问题：Host 仍需要权限、确认、trace 和安全策略。

修复：server 最小权限，Host 统一治理。

### 19.17.6 返回超长结果

问题：撑爆模型上下文。

修复：限制大小、分页、摘要或返回引用。

### 19.17.7 远程 server 无认证

问题：任何人都能调用工具。

修复：认证、授权、TLS、限流和审计。

## 19.18 面试题：如何实现一个最小 MCP Server

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

## 19.19 小练习

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

## 19.20 本章小结

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
