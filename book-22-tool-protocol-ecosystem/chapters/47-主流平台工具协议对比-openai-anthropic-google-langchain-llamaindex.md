# 第 47 章 主流平台工具协议对比：OpenAI、Anthropic、Google、LangChain、LlamaIndex

上一章我们从抽象层面对比了 Function Calling、MCP 和 A2A。本章进一步看主流平台和框架中的工具协议设计：OpenAI、Anthropic、Google、LangChain、LlamaIndex。

注意，本章不是逐字复述某个 API 文档。平台接口会不断变化，字段名也可能更新。我们更关心稳定的工程差异：它们如何描述工具、如何返回工具调用、如何处理工具结果、如何支持内置工具、如何组织 Agent 和 RAG，以及迁移时要注意什么。

你可以先记住一句话：

> 不同平台字段名会变，但工具协议的核心问题稳定不变：工具如何声明、模型如何选择、参数如何校验、结果如何回流、错误如何恢复、安全如何治理。

## 47.1 对比维度

比较工具协议时，不要只看字段名。应该看这些维度：

1. 工具声明方式。
2. Schema 表达能力。
3. 模型如何返回工具调用。
4. 工具结果如何回传。
5. 是否支持并行工具调用。
6. 是否支持强制工具调用。
7. 是否支持内置工具。
8. 是否支持流式工具调用。
9. 错误处理方式。
10. 与 Agent / RAG / Memory 的集成方式。
11. 安全和权限边界。
12. 可观测性和 eval 支持。

这套维度比“哪个平台更好”更有面试价值。

## 47.2 OpenAI 工具调用抽象

OpenAI 的工具调用通常围绕 tools、tool_choice、tool call、tool result 这些概念展开。

典型抽象：

1. 开发者提供工具列表。
2. 每个工具有 name、description、parameters schema。
3. 模型决定是否调用工具。
4. 模型输出 tool call 和 arguments。
5. 客户端执行工具。
6. 工具结果回传给模型。
7. 模型继续生成最终回答。

OpenAI 风格的重点是 provider 原生支持结构化 tool call，适合直接构建工具调用应用。

工程关注点：

1. tool schema 是否清晰。
2. tool_choice 如何控制。
3. parallel tool calls 是否开启。
4. arguments 是否需要二次校验。
5. tool result 是否需要裁剪和脱敏。
6. 是否使用 structured output 约束最终答案。

## 47.3 Anthropic 工具使用抽象

Anthropic 的工具使用通常强调 tool use block 和 tool result block。

典型流程：

1. 请求中提供 tools。
2. 模型输出 tool_use。
3. 客户端执行工具。
4. 用户侧消息中回传 tool_result。
5. 模型继续回答。

Anthropic 的文档和设计中很强调：工具是由客户端执行的，模型只是请求使用工具。这个边界非常重要，因为权限、执行和安全都在客户端/Host。

工程关注点：

1. tool_use 和 tool_result 的消息结构。
2. 工具结果是否作为用户消息回传。
3. 长工具结果如何裁剪。
4. 多轮工具循环如何控制。
5. 工具失败如何反馈给模型。
6. prompt injection 如何在 tool_result 中防御。

## 47.4 Google Gemini 工具调用抽象

Google Gemini 也支持函数调用和工具使用，通常围绕 function declarations、function call、function response 等概念。

工程上可以关注：

1. 函数声明如何表达参数。
2. 模型如何返回 function call。
3. 客户端如何回传 function response。
4. 是否支持自动函数调用模式。
5. 与 Google 生态内置工具的结合。
6. 多模态输入下工具调用如何工作。

Google 生态的一个特点是可能更容易和搜索、代码执行、多模态能力、云服务集成，但具体能力依赖产品版本。

面试时不要死记字段名，而要说清楚：Gemini 的 function calling 本质上仍然是模型输出结构化调用请求，客户端执行并回传结果。

## 47.5 LangChain 的工具抽象

LangChain 是框架，不是模型 provider。

它通常提供：

1. Tool 抽象。
2. Agent 抽象。
3. Chain / Runnable。
4. Retriever。
5. Memory。
6. Callback / tracing。
7. 多 provider 适配。

LangChain 的价值是把不同模型 provider、工具、检索器、Agent runtime 组合起来。

但也要注意：框架抽象越高，隐藏细节越多。生产系统仍然要理解底层 provider 的工具调用语义。

工程关注点：

1. Tool wrapper 是否保留 schema 细节。
2. Agent 是否会过度调用工具。
3. Callback trace 是否足够完整。
4. 错误恢复是否可控。
5. Memory 是否会污染上下文。
6. 多 provider 迁移时字段语义是否一致。

## 47.6 LlamaIndex 的工具和数据抽象

LlamaIndex 更偏数据和 RAG 生态，围绕 index、retriever、query engine、tool、agent 等抽象组织。

常见特点：

1. 强调数据连接器。
2. 强调索引和检索。
3. Query engine 可以作为工具给 Agent 使用。
4. 支持把结构化数据、文档、知识图谱等接入模型。
5. 适合 RAG-heavy 应用。

工程关注点：

1. Retriever 结果是否有引用。
2. Query engine 输出是否可追溯。
3. Agent 使用多个 query engine 时如何路由。
4. 文档权限和多租户隔离。
5. RAG prompt injection 防御。
6. 与外部 Tool runtime 的边界。

## 47.7 Provider 和 Framework 的区别

OpenAI、Anthropic、Google 是模型 provider。

LangChain、LlamaIndex 是应用框架。

二者不要混为一谈。

Provider 解决：

1. 模型 API。
2. 原生 tool call 格式。
3. 模型能力。
4. 内置工具能力。

Framework 解决：

1. 多模型适配。
2. 工具封装。
3. RAG pipeline。
4. Agent runtime。
5. tracing。
6. workflow 组合。

生产系统常见组合是：框架负责应用编排，provider 提供模型和原生工具调用能力。

## 47.8 Schema 差异

不同平台都支持类似 JSON Schema 的参数声明，但支持细节可能不同。

迁移时要注意：

1. 支持哪些 JSON Schema 字段。
2. enum 是否稳定。
3. nested object 支持程度。
4. array object 支持程度。
5. format 是否生效。
6. required 语义是否一致。
7. description 对模型行为的影响。
8. 是否支持 strict schema。

不要假设一个平台能接受的 schema，另一个平台一定完全等价。

## 47.9 Tool Choice 差异

不同平台对 tool_choice、auto、none、强制工具调用的支持方式不同。

工程上需要统一抽象：

1. auto：模型自行决定。
2. none：禁止工具。
3. required：必须调用某个工具或任意工具。
4. force_specific：强制调用指定工具。
5. allowlist：本轮只允许部分工具。

如果要做多 provider 适配，最好在自己的 Tool Runtime 里定义统一策略，再映射到各平台字段。

## 47.10 并行工具调用差异

有的平台支持模型一次返回多个 tool call，有的平台更偏顺序调用。

并行工具调用要注意：

1. 工具是否互相独立。
2. 是否都是只读。
3. 结果如何合并。
4. 错误如何处理。
5. 是否需要保持顺序。
6. 下游服务并发是否足够。

多 provider 迁移时，并行行为可能影响延迟和成本。

## 47.11 内置工具差异

主流平台可能提供内置工具，例如搜索、代码执行、文件检索、浏览器或计算工具等。

内置工具的优点：

1. 接入简单。
2. 与模型优化更紧密。
3. 平台可能内置安全限制。
4. 开发成本低。

缺点：

1. 可控性弱。
2. 可观测性可能有限。
3. 权限治理受平台限制。
4. 迁移困难。
5. 企业内网资源不一定可接入。

企业系统通常会混合使用内置工具和自建工具 runtime。

## 47.12 Streaming 差异

流式输出中，工具调用可能分片返回。

需要处理：

1. tool call arguments 增量拼接。
2. JSON 不完整状态。
3. 多 tool call 的流式事件。
4. 用户中途取消。
5. 工具执行和模型输出交错。
6. trace 事件顺序。

流式工具调用能降低用户感知延迟，但实现复杂度更高。

## 47.13 错误处理差异

Provider 原生 API 的错误格式不同，框架也可能封装错误。

生产系统最好定义内部统一错误模型：

```json
{
  "code": "TOOL_TIMEOUT",
  "category": "dependency",
  "retryable": true,
  "message": "Tool execution timed out.",
  "details": {}
}
```

再把不同平台错误映射进来。

否则 Agent 很难做统一恢复。

## 47.14 可观测性差异

不同平台和框架提供的 trace 能力不同。

你至少需要记录：

1. 模型请求。
2. tool schema。
3. tool choice。
4. tool call arguments。
5. validation result。
6. tool execution result。
7. final answer。
8. latency。
9. cost。
10. errors。

如果平台 trace 不够，企业系统需要自己补充。

## 47.15 迁移时的注意事项

从一个平台迁移到另一个平台，不能只改 API endpoint。

要检查：

1. Tool schema 兼容性。
2. Tool choice 行为。
3. 参数生成风格。
4. 并行工具调用差异。
5. 工具结果回传格式。
6. 流式事件格式。
7. 错误处理。
8. 安全策略。
9. 评估指标是否回归。
10. Prompt 是否需要调整。

迁移必须跑 Tool-use eval benchmark。

## 47.16 统一 Tool Runtime 的价值

如果企业要支持多 provider，可以建设统一 Tool Runtime。

统一 Runtime 负责：

1. 内部工具注册。
2. Schema 转换。
3. Tool choice 策略。
4. 参数校验。
5. 权限检查。
6. 工具执行。
7. 错误归一化。
8. Trace。
9. Provider adapter。
10. Eval。

这样应用层不用直接依赖某个 provider 的所有细节。

## 47.17 对比总结表

| 维度 | OpenAI | Anthropic | Google | LangChain | LlamaIndex |
| --- | --- | --- | --- | --- | --- |
| 类型 | Provider | Provider | Provider | Framework | Framework |
| 核心工具抽象 | tools / tool calls | tool_use / tool_result | function declarations / calls | Tool / Agent / Runnable | Query engine / Tool / Agent |
| 重点 | 原生结构化工具调用 | 清晰的工具使用消息块 | 多模态和生态集成 | 多 provider 编排 | 数据和 RAG 生态 |
| 适合 | 直接构建 tool-use 应用 | 可控工具循环 | Google 生态应用 | Agent 和链式编排 | RAG-heavy 应用 |
| 风险 | provider lock-in | 工具循环复杂度 | 生态依赖 | 抽象隐藏细节 | RAG 权限和引用治理 |

这张表只用于理解方向，不代表固定优劣。

## 47.18 常见误区

### 47.18.1 只比较字段名

字段名会变，工程语义更重要。

### 47.18.2 把框架当 provider

LangChain 和 LlamaIndex 是框架，它们底层仍然调用模型 provider。

### 47.18.3 以为迁移只改 API

工具调用行为、schema 支持、流式事件和错误恢复都可能不同。

### 47.18.4 忽略 eval

迁移或升级后必须跑 Tool-use eval，不然很容易出现工具选择和参数回归。

### 47.18.5 过度依赖内置工具

内置工具方便，但企业权限、审计和内网数据接入可能需要自建 runtime。

## 47.19 面试高频题

### 题 1：Provider 和 Framework 的工具抽象有什么区别？

参考回答：

Provider 如 OpenAI、Anthropic、Google 提供模型原生工具调用 API，定义模型如何输出 tool call、客户端如何回传结果。Framework 如 LangChain、LlamaIndex 提供更高层的工具封装、RAG、Agent、Workflow 和 tracing，底层仍要适配 provider。

### 题 2：迁移工具调用平台要注意什么？

参考回答：

要检查 schema 支持、tool choice 语义、参数生成风格、并行工具调用、工具结果回传、流式事件、错误格式、安全策略和 prompt 行为。迁移后必须跑 Tool-use eval 和 trace replay。

### 题 3：为什么需要统一 Tool Runtime？

参考回答：

统一 Tool Runtime 可以屏蔽不同 provider 的字段差异，集中做工具注册、schema 转换、权限、参数校验、错误归一化、trace、eval 和安全治理，降低 vendor lock-in。

### 题 4：内置工具和自建工具如何取舍？

参考回答：

内置工具接入简单，和模型集成好，但可控性、审计和企业权限可能有限。自建工具 runtime 更可控，适合企业内网、敏感数据和复杂权限，但开发成本更高。生产系统常混合使用。

### 题 5：LangChain 和 LlamaIndex 的侧重点有什么不同？

参考回答：

LangChain 更偏通用应用编排、Agent、Tool、Runnable 和多 provider 组合；LlamaIndex 更偏数据连接、索引、检索、Query Engine 和 RAG-heavy 应用。二者可以组合，但都不是模型 provider。

## 47.20 小练习

1. 设计一个 provider-agnostic Tool Runtime，需要支持哪些内部字段？
2. 写一个工具 schema，思考迁移到不同 provider 时哪些字段可能不兼容。
3. 设计一个迁移检查清单，从 OpenAI 风格工具调用迁移到另一个 provider。
4. 比较内置搜索工具和自建搜索 Tool 的优缺点。
5. 思考：为什么框架抽象越高，越需要 trace 和 eval？

## 47.21 本章小结

本章比较了 OpenAI、Anthropic、Google、LangChain 和 LlamaIndex 的工具协议与工具抽象。

Provider 提供模型原生工具调用能力，Framework 提供应用层编排和抽象。不同平台字段名、tool choice、schema 支持、并行调用、流式事件和错误处理都可能不同。企业系统如果要多 provider 支持，最好建设统一 Tool Runtime，并通过 eval 和 trace replay 保证迁移不回归。

你可以把本章重点记成一句话：

> 不要被平台字段名迷惑，真正要比较的是工具声明、调用循环、结果回传、安全治理、可观测性和迁移成本。

下一章我们会进入系统设计题：设计一个企业 MCP 工具平台。
