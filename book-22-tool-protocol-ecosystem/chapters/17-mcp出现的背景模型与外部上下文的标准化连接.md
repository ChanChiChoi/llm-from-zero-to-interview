# 第十七章：MCP 出现的背景：模型与外部上下文的标准化连接

## 17.1 本章定位

前两部分讲了 Function Calling 和企业工具平台。本章开始进入 MCP 协议生态。

MCP，全称 Model Context Protocol，直译是模型上下文协议。它的核心目标不是发明一个新的“函数调用 demo”，而是把模型应用和外部工具、资源、上下文之间的连接方式标准化。

为什么需要标准化？

因为当模型应用越来越多时，大家会遇到一个重复问题：

```text
每个模型应用都要自己接数据库、文件系统、浏览器、Git、IDE、知识库、API、企业系统。
每个工具也要为不同模型应用重复适配。
```

这会产生大量重复集成和安全治理问题。

本章的核心观点是：

```text
MCP 出现的背景，是模型应用从单点工具调用走向开放上下文生态后，需要一个统一协议来连接工具、资源、提示模板和运行时能力。
```

## 17.2 从 Function Calling 到 Context Protocol

Function Calling 解决的是：

```text
模型如何表达“我想调用某个函数，并传入这些参数”。
```

但模型应用真正需要的不止函数。

它还需要：

1. 读取文件。
2. 查询数据库。
3. 浏览网页。
4. 搜索代码。
5. 获取 IDE 上下文。
6. 访问知识库。
7. 使用可复用 prompt。
8. 调用工具。
9. 订阅资源变化。
10. 在不同客户端之间复用能力。

Function Calling 更像模型 API 的一个工具调用能力。MCP 更像模型应用和外部上下文之间的连接协议。

可以粗略理解：

```text
Function Calling: 模型和应用之间如何表达工具调用。
MCP: 模型客户端和外部能力服务器之间如何发现、描述、读取和调用上下文能力。
```

## 17.3 集成爆炸问题

假设有 5 个模型客户端：

1. Chat App。
2. IDE Agent。
3. 数据分析 Agent。
4. 企业客服 Agent。
5. 自动化办公 Agent。

又有 8 类外部系统：

1. 文件系统。
2. Git 仓库。
3. 数据库。
4. 浏览器。
5. 知识库。
6. 工单系统。
7. 邮件系统。
8. 日历系统。

如果没有标准协议，每个客户端都要分别接每个系统：

```text
5 clients × 8 systems = 40 integrations
```

当客户端和系统数量继续增长，集成成本会爆炸。

MCP 想把这个关系变成：

```text
Clients implement MCP Client
Systems expose MCP Server
```

这样客户端和工具系统通过统一协议对接，减少重复适配。

## 17.4 为什么不只用 HTTP API

有人会问：外部系统本来就有 HTTP API，为什么还需要 MCP？

HTTP API 解决的是服务之间通信，不直接解决模型上下文连接问题。

普通 HTTP API 通常缺少：

1. 面向模型的工具描述。
2. 模型友好的参数 schema。
3. 资源枚举和读取协议。
4. 可复用 prompts。
5. 工具、资源、prompt 的统一发现机制。
6. 客户端与 server 的能力协商。
7. 面向 Agent 的上下文边界。
8. 本地工具和远程工具统一抽象。

当然，MCP Server 内部可以调用 HTTP API。但 MCP 面向的是模型客户端连接外部上下文的协议层，而不是替代所有 HTTP API。

## 17.5 为什么不只用插件系统

插件系统也能扩展模型能力。

但传统插件通常有几个问题：

1. 绑定某个平台。
2. manifest 格式不统一。
3. 权限模型不一致。
4. 本地资源访问能力弱。
5. 难以跨客户端复用。
6. 工具、资源、prompt 边界不清。
7. 开发者需要为每个平台重复适配。

MCP 更强调开放协议：一个 MCP Server 可以被多个支持 MCP 的客户端复用。

例如一个 Git MCP Server，可以被 IDE Agent、命令行 Agent、代码审查 Agent 使用。

## 17.6 MCP 连接的不是“模型本体”，而是模型客户端

一个常见误解是：MCP Server 直接连到大模型。

更准确地说，MCP 连接的是模型应用或模型客户端。

典型结构：

```text
LLM
  ↑
MCP Client / Host Application
  ↑
MCP Server
  ↑
External System
```

模型通常不知道 MCP wire protocol 的细节。Host Application 负责：

1. 连接 MCP Server。
2. 列出 tools / resources / prompts。
3. 把工具描述转成模型 API 可用格式。
4. 执行 MCP tool。
5. 把结果回填给模型。

所以 MCP 是模型应用架构的一部分，而不是模型权重内部能力。

## 17.7 Tools、Resources、Prompts 为什么都需要

Function Calling 主要关注 tools。

MCP 把能力分成多类，其中常见包括：

1. Tools：可调用动作，例如搜索、查询、创建。
2. Resources：可读取上下文资源，例如文件、数据库记录、网页、文档。
3. Prompts：可复用提示模板或工作流入口。

为什么 resources 很重要？

因为模型不只是需要“调用动作”，还需要“读取上下文”。

例如 IDE 场景：

1. 工具：运行测试。
2. 资源：当前文件、项目目录、Git diff。
3. Prompt：代码审查模板。

如果只有工具调用，资源读取和 prompt 复用就会各自发明协议。

MCP 把这些统一到同一连接模型下。

## 17.8 本地上下文问题

很多重要上下文在本地，而不是云端。

例如：

1. 用户本地文件。
2. IDE 当前项目。
3. 本地 Git 仓库。
4. 本地数据库。
5. 终端环境。
6. 浏览器 session。

云端模型 API 无法直接访问这些资源，也不应该默认直接访问。

MCP 的一个重要价值是让本地 host application 以受控方式连接本地 MCP Server，把本地上下文暴露给模型应用。

这也是为什么 MCP 在 IDE、coding agent、desktop agent 场景中很有吸引力。

## 17.9 安全边界为什么更重要

MCP 让模型应用更容易连接外部上下文，也让安全边界更重要。

风险包括：

1. MCP Server 暴露过多文件。
2. 工具执行危险命令。
3. 资源内容包含 prompt injection。
4. 客户端无权限控制。
5. 本地密钥泄露。
6. 多 server 之间数据流失控。
7. 用户不知道某个 server 能访问什么。

因此 MCP 不是“装上 server 就随便让模型用”。

需要：

1. 明确 server 权限。
2. 用户授权。
3. 沙箱。
4. 工具和资源白名单。
5. 结果脱敏。
6. 审计。
7. 客户端侧策略控制。

后面第 23 章会专门讲 MCP 权限、安全和本地沙箱。

## 17.10 MCP 与企业工具平台的关系

上一章讲企业工具平台。MCP 可以成为企业工具平台的一部分。

关系可以这样理解：

```text
MCP Server: 暴露一组工具、资源、prompt。
企业 Tool Registry: 管理这些工具的权限、版本、owner、风险和 eval。
Tool Executor: 通过 MCP Client 调用 MCP Server。
```

所以 MCP 不是替代企业治理，而是提供一种连接能力。

企业平台仍然要做：

1. 注册。
2. 权限。
3. 路由。
4. 审计。
5. 安全。
6. eval。
7. 灰度。

不要因为工具来自 MCP Server，就绕过安全治理。

## 17.11 MCP 解决的标准化对象

MCP 标准化的对象大致包括：

1. 客户端如何连接 server。
2. server 如何声明能力。
3. client 如何列出 tools。
4. client 如何调用 tools。
5. client 如何列出和读取 resources。
6. client 如何使用 prompts。
7. 消息如何编码。
8. 错误如何返回。
9. 能力如何协商。

这些对象看似基础，但生态很需要。

没有标准时，每个 Agent 框架都要自定义一套：工具 manifest、资源读取、prompt 模板、错误格式、连接方式。

## 17.12 MCP 不解决什么

理解 MCP 也要知道它不解决什么。

MCP 不等于：

1. 模型本身。
2. Agent 全部运行时。
3. 权限系统的完整实现。
4. 企业治理平台。
5. 安全沙箱的完整替代。
6. 工具质量评估系统。
7. 多 Agent 协作协议。

MCP 提供连接协议，但实际系统仍要实现：

1. 用户授权。
2. 工具选择。
3. 安全策略。
4. trace。
5. eval。
6. 发布治理。

面试中要避免把 MCP 说成万能平台。

## 17.13 MCP 与 A2A 的区别预告

后面会讲 A2A。这里先简单区分。

MCP 更偏：

```text
Agent / model client 连接工具、资源、上下文。
```

A2A 更偏：

```text
Agent 和 Agent 之间通信、委派任务、同步状态。
```

也就是：

```text
MCP: Agent-to-Tool / Agent-to-Context
A2A: Agent-to-Agent
```

二者可以结合。例如一个 Agent 通过 A2A 委派另一个 Agent，而另一个 Agent 通过 MCP 访问本地工具。

## 17.14 为什么 MCP 对开发者生态重要

MCP 的生态价值在于降低集成成本。

对工具开发者：

1. 写一个 MCP Server。
2. 多个客户端可复用。
3. 不必为每个 Agent 框架单独开发插件。

对客户端开发者：

1. 实现 MCP Client。
2. 可以连接多个 MCP Server。
3. 工具和资源接入方式统一。

对企业：

1. 更容易标准化内部能力暴露。
2. 更容易治理工具接入。
3. 更容易构建工具市场。
4. 更容易做审计和安全策略。

当然，这些价值能否实现，取决于客户端、server 和治理系统是否成熟。

## 17.15 面试中如何回答 MCP 背景

如果面试官问：

```text
MCP 为什么出现？它解决了什么问题？
```

可以这样回答：

第一，模型应用需要外部上下文：

1. tools。
2. files。
3. databases。
4. web。
5. IDE。
6. knowledge base。
7. prompts。

第二，单点 function calling 不够：

1. 每个客户端重复接工具。
2. 每个工具重复适配客户端。
3. resources 和 prompts 没有统一协议。
4. 本地上下文接入困难。

第三，MCP 提供标准连接层：

1. MCP Client。
2. MCP Server。
3. Tools。
4. Resources。
5. Prompts。
6. 能力发现和调用。

第四，它不是万能治理平台：

1. 仍需要权限。
2. 仍需要安全。
3. 仍需要 trace。
4. 仍需要 eval。
5. 仍需要企业发布治理。

一句话总结：

```text
MCP 的出现，是为了把模型应用连接外部工具和上下文的方式从各自为战的插件集成，推进到统一协议和可复用生态。
```

## 17.16 常见误区

### 17.16.1 MCP 等于 Function Calling

不对。Function Calling 主要是模型 API 表达工具调用。MCP 是 client 和 server 之间暴露 tools、resources、prompts 的协议。

### 17.16.2 MCP Server 直接连模型

通常不是。MCP Server 连接 MCP Client / Host Application，Host 再把能力提供给模型使用。

### 17.16.3 有 MCP 就不需要权限

不对。MCP 提供连接方式，不替代企业权限、安全和审计。

### 17.16.4 MCP 只适合远程 API

不对。MCP 很适合本地上下文，例如文件系统、IDE、Git 仓库、本地数据库。

### 17.16.5 MCP 是 Agent-to-Agent 协议

不准确。MCP 更偏 Agent-to-Tool / Agent-to-Context。Agent-to-Agent 通信更接近 A2A 讨论范围。

## 17.17 小练习

### 练习 1：Function Calling 与 MCP

Function Calling 和 MCP 最大区别是什么？

参考答案：Function Calling 关注模型如何输出工具调用意图；MCP 关注模型客户端如何标准化连接外部 tools、resources 和 prompts。

### 练习 2：集成爆炸

为什么没有统一协议会产生集成爆炸？

参考答案：多个模型客户端和多个外部系统需要两两适配，客户端数乘以系统数导致集成数量快速增长。

### 练习 3：MCP 是否替代企业工具平台

MCP 能否替代企业 Tool Registry、权限和审计？

参考答案：不能。MCP 是连接协议，企业平台仍需要 Registry、权限、安全、trace、eval 和发布治理。

### 练习 4：本地上下文

为什么 MCP 对 IDE Agent 有价值？

参考答案：IDE Agent 需要访问本地文件、Git diff、项目结构、测试工具等上下文，MCP 提供标准方式让本地 server 暴露这些能力。

### 练习 5：MCP 与 A2A

MCP 和 A2A 如何区分？

参考答案：MCP 偏 Agent-to-Tool / Agent-to-Context，A2A 偏 Agent-to-Agent 通信和任务委派。

## 17.18 本章小结

本章讲了 MCP 出现的背景。

你需要掌握：

1. MCP 解决的是模型客户端与外部工具、资源、prompt 的标准化连接问题。
2. Function Calling 解决工具调用表达，MCP 解决外部上下文生态连接。
3. 没有标准协议时，多客户端和多工具系统会产生集成爆炸。
4. HTTP API 和插件系统不能完全替代 MCP，因为它们不直接提供模型上下文层的统一抽象。
5. MCP 连接的是 MCP Client / Host Application 和 MCP Server，不是模型权重本身直接连接 server。
6. MCP 同时关注 tools、resources、prompts，而不只是 function call。
7. MCP 对本地上下文、IDE、文件系统、Git、数据库等场景很有价值。
8. MCP 不替代权限、安全、trace、eval 和企业治理。
9. MCP 与 A2A 的区别是 Agent-to-Context 与 Agent-to-Agent。

如果只记一句话：

```text
MCP 的核心意义，是把模型应用接入外部上下文的方式标准化，让工具、资源和提示模板可以跨客户端、跨系统复用，而不是每个 Agent 都重新造一套集成。
```

下一章会讲 MCP 基本概念：Client、Server、Tools、Resources、Prompts，建立后续实现 MCP Server 的基础。
