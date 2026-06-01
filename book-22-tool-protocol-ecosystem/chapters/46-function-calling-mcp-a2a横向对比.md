# 第 46 章 Function Calling / MCP / A2A 横向对比

到目前为止，我们已经分别讲过 Function Calling、MCP 和 A2A。很多同学学到这里会有一个非常自然的问题：这三者到底是什么关系？它们是不是互相替代？什么时候用 Function Calling，什么时候用 MCP，什么时候用 A2A？

这一章做横向对比。

我们不从某一家厂商的实现出发，而从工程抽象出发：Function Calling 是模型输出结构化工具调用意图的机制；MCP 是 Host/Client 与外部工具、资源、提示模板之间的连接协议；A2A 是 Agent 与 Agent 之间的协作协议。

你可以先记住一句话：

> Function Calling 解决“模型如何表达要调用工具”，MCP 解决“工具和资源如何标准化接入 Host”，A2A 解决“Agent 之间如何协作完成任务”。

## 46.1 三者的一句话定义

| 协议/机制 | 一句话定义 | 主要解决的问题 |
| --- | --- | --- |
| Function Calling | 模型输出结构化工具调用请求 | 模型如何选择工具并生成参数 |
| MCP | 标准化连接工具、资源和提示模板 | Host 如何发现和接入外部能力 |
| A2A | Agent 间任务委派和状态协作 | 多个 Agent 如何互相发现、委派和返回结果 |

这三个层次经常组合使用，但抽象层级不同。

## 46.2 从调用链看三者位置

一个典型调用链：

```text
用户
  -> Host / Agent Runtime
    -> 模型输出 Function Calling tool_call
    -> Host 将 tool_call 路由到 MCP Tool
    -> MCP Server 执行工具并返回结果
```

如果涉及多个 Agent：

```text
用户
  -> Orchestrator Agent
    -> A2A 委派给 Data Agent
      -> Data Agent 通过 Function Calling 决定调用工具
      -> Data Agent 通过 MCP 查询数据库
    -> A2A 委派给 Report Agent
      -> Report Agent 生成报告
```

可以看出：

1. Function Calling 在模型和 Host 之间。
2. MCP 在 Host/Client 和 Tool/Resource Server 之间。
3. A2A 在 Agent 和 Agent 之间。

## 46.3 抽象粒度对比

| 维度 | Function Calling | MCP | A2A |
| --- | --- | --- | --- |
| 粒度 | 单次工具调用意图 | 工具/资源/Prompt 服务连接 | Agent 任务协作 |
| 主要对象 | tool schema、tool_call、arguments | tools、resources、prompts、server | agent card、task、message、artifact |
| 生命周期 | 一次模型输出到工具结果 | Server 注册、连接、调用、资源读取 | 任务提交、状态同步、结果返回 |
| 状态 | 通常较轻 | 连接和能力状态 | 任务生命周期状态 |
| 典型场景 | 调 API、查数据、执行函数 | IDE、数据库、知识库、浏览器、终端接入 | 多专家 Agent 协作 |

Function Calling 最轻，A2A 最重，MCP 介于两者之间但偏工具接入。

## 46.4 Function Calling 适合什么

Function Calling 适合模型直接表达工具调用意图。

例如：

```json
{
  "tool": "get_weather",
  "arguments": {
    "city": "Beijing"
  }
}
```

适合场景：

1. 工具数量有限。
2. Host 已经知道工具列表。
3. 工具 schema 可以直接传给模型。
4. 调用链较短。
5. 主要关注参数生成和工具选择。

Function Calling 不负责：

1. 工具如何安装。
2. 工具服务如何发现。
3. 资源如何暴露。
4. 多 Agent 如何协作。
5. 企业级工具注册和权限治理。

## 46.5 MCP 适合什么

MCP 适合把外部工具、资源和提示模板标准化接入 Host。

适合场景：

1. IDE 接入代码文件、Git、终端。
2. 知识库接入文档和搜索工具。
3. 数据库接入 schema 和只读查询。
4. 浏览器接入网页资源和自动化工具。
5. 企业内部多个工具服务统一接入。

MCP 的价值是：

1. 标准化 Server 能力暴露。
2. 支持 tools/resources/prompts。
3. Host 可以统一管理权限和上下文。
4. 工具服务可以复用，不绑定某个模型 provider。

MCP 不等于模型调用格式。Host 可以把 MCP Tool 转换成某个模型供应商的 Function Calling schema。

## 46.6 A2A 适合什么

A2A 适合 Agent 与 Agent 协作。

适合场景：

1. 多个 Agent 由不同团队维护。
2. Agent 能力需要动态发现。
3. 任务需要异步执行。
4. 下游 Agent 可能追问。
5. 需要状态同步和 Artifact。
6. 需要跨 Agent 权限和审计。
7. 一个 Agent 会委派给另一个 Agent。

A2A 的核心不是调用一个函数，而是管理任务生命周期：

```text
submitted -> accepted -> running -> input_required -> completed / failed
```

A2A 不适合替代简单工具调用。如果只是读取文件或查询数据库，用 MCP/Tool 更合适。

## 46.7 安全边界对比

| 维度 | Function Calling | MCP | A2A |
| --- | --- | --- | --- |
| 主要风险 | 工具选错、参数错、危险工具调用 | 工具越权、资源泄露、本地沙箱 | 上下文扩散、转授权、Agent 信任 |
| 安全中心 | Host / Runtime | Host + MCP Server + Policy | A2A Runtime + Policy + Audit |
| 典型控制 | tool_choice、参数校验、确认 | roots、allowlist、sandbox、权限 | Agent allowlist、context_policy、OBO |
| 审计粒度 | tool_call | tool/resource access | task/message/artifact chain |

三者都需要安全，但安全关注点不同。

## 46.8 状态和生命周期对比

Function Calling 通常是短生命周期：

```text
model emits tool_call -> host executes -> tool_result -> model continues
```

MCP 有连接和能力生命周期：

```text
connect server -> discover tools/resources/prompts -> call/read -> update capabilities
```

A2A 有任务生命周期：

```text
discover agent -> delegate task -> receive status -> handle input_required -> collect artifact
```

如果系统需要长任务、状态同步和产物，A2A 更自然。如果只是一次工具调用，Function Calling/MCP 更自然。

## 46.9 上下文处理对比

Function Calling 中，工具结果通常作为 tool result 进入模型上下文。

MCP 中，Resources 可以作为可引用上下文，由 Host 决定何时读取和注入。

A2A 中，上下文在 Agent 之间传递，需要 context_policy、来源标记、权限和最小上下文原则。

| 维度 | Function Calling | MCP | A2A |
| --- | --- | --- | --- |
| 上下文来源 | tool result | resource / tool result / prompt | task context / message / artifact |
| 关键问题 | 结果如何回到模型 | 资源如何暴露和选择 | 上下文能不能转发给下游 Agent |
| 风险 | 工具结果污染上下文 | Resource prompt injection | 跨 Agent 泄露和漂移 |

## 46.10 组合使用的典型模式

最常见模式是：

```text
Agent 内部：Function Calling + MCP
Agent 之间：A2A
```

展开：

1. 模型通过 Function Calling 表达工具调用。
2. Host 把工具调用路由到 MCP Server。
3. MCP Server 执行工具或提供 Resource。
4. 如果任务需要其他 Agent，使用 A2A 委派。
5. 下游 Agent 内部也可以使用 Function Calling + MCP。

这是一套分层架构，而不是三选一。

## 46.11 决策树：该用哪个

可以用这个决策树：

1. 只是让模型选择工具并填参数？用 Function Calling。
2. 需要标准化接入外部工具、资源、提示模板？用 MCP。
3. 需要多个 Agent 互相发现、委派任务、同步状态？用 A2A。
4. 需要把 MCP Tool 暴露给模型？MCP + Function Calling。
5. 需要多 Agent 且每个 Agent 使用工具？A2A + MCP + Function Calling。

例子：

| 场景 | 推荐方案 |
| --- | --- |
| 单模型调用天气 API | Function Calling |
| Coding Agent 连接 IDE 和终端 | MCP + Function Calling |
| 企业知识库和数据库统一接入 | MCP |
| 总控 Agent 委派给数据 Agent 和报告 Agent | A2A |
| 多 Agent 企业助手，每个 Agent 都要查工具 | A2A + MCP + Function Calling |

## 46.12 常见混淆

### 46.12.1 把 MCP 当成 Function Calling

MCP 不是模型 provider 的 tool_call 格式。MCP 是工具/资源/Prompt 的连接协议。Host 可以把 MCP Tool 转成模型使用的 Function Calling schema。

### 46.12.2 把 A2A 当成工具调用

A2A 不是读文件、查数据库这种简单工具协议。它面向 Agent 任务生命周期。

### 46.12.3 用 A2A 包装所有工具

如果一个能力只是被动执行操作，就不应该强行包装成 Agent。

### 46.12.4 用 Function Calling 管全部生态

Function Calling 只解决模型输出结构化调用，不解决工具注册、安装、资源暴露、Agent 协作和企业治理。

## 46.13 面试中如何回答

如果面试官问：

```text
Function Calling、MCP、A2A 有什么区别？
```

可以这样回答：

```text
Function Calling 是模型和 Host 之间的结构化工具调用机制，解决模型如何选择工具和生成参数。MCP 是 Host/Client 和外部工具资源 Server 之间的标准连接协议，解决工具、资源和 Prompt 如何被发现和接入。A2A 是 Agent 与 Agent 之间的协作协议，解决能力发现、任务委派、状态同步、上下文边界和 Artifact 返回。它们不是互斥关系，生产系统常见模式是 Agent 内部通过 Function Calling 调 MCP Tool，Agent 之间通过 A2A 协作。
```

这段回答基本覆盖核心。

## 46.14 系统设计示例

设计一个企业研发助手：

1. 用户提交“修复测试失败并生成说明”。
2. Orchestrator Agent 使用 A2A 委派给 Coding Agent。
3. Coding Agent 通过 MCP 连接 IDE、Git、Terminal。
4. Coding Agent 内部模型通过 Function Calling 调用 read_file、search_code、run_tests、apply_patch。
5. 如果需要安全审查，Orchestrator 通过 A2A 委派给 Security Review Agent。
6. 所有工具结果带来源和 trace。
7. 代码修改需要 diff 预览和用户确认。

这个例子同时使用三者，但分工清楚。

## 46.15 对比表总结

| 维度 | Function Calling | MCP | A2A |
| --- | --- | --- | --- |
| 主要用途 | 模型调用工具 | Host 连接工具资源 | Agent 间协作 |
| 核心抽象 | tool schema / tool_call | tools / resources / prompts | agent card / task / message / artifact |
| 被调用方 | 工具函数 | MCP Server | Agent |
| 任务粒度 | 单次操作 | 工具或资源访问 | 多步任务 |
| 状态复杂度 | 低 | 中 | 高 |
| 是否适合长任务 | 不适合 | 部分适合 | 适合 |
| 是否处理 Agent 发现 | 不处理 | 不处理 Agent | 处理 |
| 是否处理资源暴露 | 不直接处理 | 处理 | 通过上下文和 Artifact 处理 |
| 安全重点 | 参数、工具选择、确认 | 工具权限、资源沙箱 | 身份、转授权、上下文扩散 |

## 46.16 常见误区

### 46.16.1 认为三者只能选一个

实际生产系统经常组合使用。

### 46.16.2 认为 MCP 比 Function Calling 更高级所以可以替代它

MCP 和 Function Calling处在不同层。MCP Tool 最终仍可能被 Host 投影成模型的 Function Calling tool。

### 46.16.3 认为 A2A 可以直接替代 Workflow

A2A 是 Agent 通信协议，Workflow 是任务编排方式。二者可以结合，但不是同一概念。

### 46.16.4 忽略 Host 的角色

Host 是三者组合中的核心安全和上下文控制点。

## 46.17 面试高频题

### 题 1：Function Calling 和 MCP 的区别是什么？

参考回答：

Function Calling 是模型输出结构化工具调用请求的机制，关注 tool schema、tool_call 和 arguments。MCP 是 Host 与外部工具、资源、Prompt Server 之间的连接协议，关注 tools、resources、prompts 的发现、调用和暴露。MCP Tool 可以被 Host 转成 Function Calling schema 给模型使用。

### 题 2：MCP 和 A2A 的区别是什么？

参考回答：

MCP 是 Agent-to-Tool/Resource，面向工具和资源接入；A2A 是 Agent-to-Agent，面向 Agent 之间的任务委派、状态同步、上下文边界和 Artifact 返回。MCP 被调用方通常是工具服务，A2A 被调用方是具备自主任务执行能力的 Agent。

### 题 3：三者如何组合？

参考回答：

常见模式是 Agent 内部使用 Function Calling 表达工具调用意图，Host 将调用路由到 MCP Server；多个 Agent 之间通过 A2A 委派任务和同步状态。Trace 需要串联 tool_call、MCP access 和 A2A task。

### 题 4：如果只接一个数据库工具，需要 A2A 吗？

参考回答：

不需要。只接数据库工具，MCP 或普通 Function Calling 就足够。A2A 适合多个 Agent 协作和任务生命周期管理，不应该为了简单工具调用引入复杂协议。

### 题 5：为什么 Host 很重要？

参考回答：

Host 负责把 MCP 能力投影给模型、执行 Function Calling、控制上下文、做权限检查、处理工具结果、阻止 prompt injection，并在多 Agent 系统中执行上下文转发和审计策略。没有 Host 控制，协议组合会失去安全边界。

## 46.18 小练习

1. 判断以下场景应使用 Function Calling、MCP、A2A 还是组合：天气查询、IDE Agent、合同审查多 Agent 系统、知识库问答。
2. 画出一个 A2A + MCP + Function Calling 的调用链。
3. 解释为什么不应该用 A2A 包装 read_file。
4. 解释为什么 MCP Tool 仍可能需要 Function Calling。
5. 思考：如果一个 MCP Server 内部开始自主规划和追问，它是否更像 Agent？应该如何重新设计？

## 46.19 本章小结

本章横向对比了 Function Calling、MCP 和 A2A。

Function Calling 解决模型如何表达工具调用意图，MCP 解决工具、资源和提示模板如何标准化接入 Host，A2A 解决 Agent 之间如何协作完成任务。三者不是互相替代，而是位于不同层次。生产系统常见组合是：Agent 内部使用 Function Calling 调 MCP Tool，Agent 之间使用 A2A 协作。

你可以把本章重点记成一句话：

> Function Calling 是模型调用语言，MCP 是工具资源连接层，A2A 是 Agent 协作层。

下一章我们会继续比较主流平台工具协议，包括 OpenAI、Anthropic、Google、LangChain 和 LlamaIndex 的工具抽象差异。
