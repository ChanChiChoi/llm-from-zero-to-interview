# 第 48 章 设计一个企业 MCP 工具平台

前面我们已经讲过 MCP 的基本概念、MCP Server、Tools、Resources、Prompts、安全、IDE/知识库/数据库/浏览器/终端集成，也讲过工具产品化和协议层工程问题。

本章用系统设计面试题的形式，把这些内容串起来：如何设计一个企业 MCP 工具平台？

这个题非常高频，因为它把大模型应用工程里的几个核心难点都揉在一起：工具注册、能力发现、权限、安全、审计、沙箱、上下文管理、多租户、成本控制和开发者体验。

你可以先记住一句话：

> 企业 MCP 工具平台不是“把一堆 MCP Server 连起来”，而是建设一个可注册、可发现、可授权、可审计、可评估、可运营的工具和上下文接入层。

## 48.1 面试题描述

题目可以这样问：

```text
请设计一个企业 MCP 工具平台。公司内部有很多系统：知识库、数据库、代码仓库、工单系统、浏览器自动化、终端执行、业务 API。希望这些能力可以通过 MCP 标准接入到不同 Agent 和模型应用中，同时满足权限、安全、审计、灰度、评估和多租户要求。
```

这道题不要一上来就画 MCP Server。要先澄清需求。

## 48.2 需求澄清

可以问面试官：

1. 平台面向内部 Agent，还是也允许外部开发者？
2. 主要接入哪些工具类型？
3. 是否多租户？
4. 是否有敏感数据和合规要求？
5. 是否需要支持本地 IDE/终端？
6. 是否需要内置 Marketplace？
7. 是否需要支持多模型 provider？
8. 是否需要高风险动作人工确认？

如果要自己假设，可以这样说：

```text
我假设这是企业内部平台，支持多个业务团队接入 MCP Server。平台服务内部 Agent 和 AI 应用，重点支持知识库、数据库、代码仓库、终端和业务 API。系统需要多租户隔离、权限控制、审计、沙箱、安全审核和工具调用评估。
```

## 48.3 核心目标

企业 MCP 工具平台要实现：

1. MCP Server 注册和发现。
2. Tool / Resource / Prompt 能力管理。
3. 权限和身份治理。
4. Host 接入和工具路由。
5. 参数校验和执行控制。
6. 工具结果脱敏和上下文管理。
7. Prompt injection 防御。
8. Trace、Replay 和 Audit。
9. 成本、延迟和并发控制。
10. 开发者门户和审核流程。

这些目标可以分层设计。

## 48.4 总体架构

一个合理架构：

```text
AI Apps / Agents / IDE Hosts
  -> MCP Gateway
  -> Tool Router / Policy Engine / Context Manager
  -> MCP Registry
  -> MCP Server Runtime
      -> Knowledge Base Server
      -> Database Server
      -> Git / IDE Server
      -> Browser Server
      -> Terminal Server
      -> Business API Server
  -> Observability: Trace / Audit / Metrics / Replay
  -> Developer Portal / Review Console
```

模块说明：

1. AI Apps / Agents：调用工具的平台使用方。
2. MCP Gateway：统一入口，处理认证、路由、限流、日志。
3. Tool Router：选择目标 MCP Server 和 Tool。
4. Policy Engine：权限、数据分类、安全策略。
5. Context Manager：工具结果裁剪、脱敏、来源标记。
6. MCP Registry：注册 Tools、Resources、Prompts 和 Server metadata。
7. MCP Server Runtime：运行和托管 MCP Server。
8. Observability：trace、audit、metrics、replay。
9. Developer Portal：开发、提交、审核、发布和文档。

## 48.5 MCP Registry 设计

MCP Registry 是能力目录。

需要存：

1. server_id。
2. server_name。
3. owner。
4. version。
5. endpoint。
6. tools。
7. resources。
8. prompts。
9. permissions。
10. risk_level。
11. status。
12. tenant_scope。
13. health。
14. documentation。

Tool metadata 示例：

```json
{
  "server_id": "mcp.database.analytics",
  "tool_name": "query_order_metrics",
  "description": "Query aggregated order metrics by time range and region.",
  "input_schema": {},
  "output_schema": {},
  "required_scopes": ["metrics.read"],
  "risk_level": "medium",
  "data_classification": ["internal", "confidential"],
  "owner": "data-platform-team"
}
```

Registry 不只是展示列表，还要参与路由、权限和审核。

## 48.6 MCP Gateway

MCP Gateway 是统一入口。

它负责：

1. Host / Agent 认证。
2. 用户身份传递。
3. 租户识别。
4. 请求限流。
5. Tool 路由。
6. 参数校验。
7. Policy Engine 调用。
8. Trace 注入。
9. 错误归一化。
10. 响应脱敏。

为什么需要 Gateway？因为不能让每个 Agent 直接连所有 MCP Server。否则权限、审计、限流和安全策略会分散。

## 48.7 Tool Router

Tool Router 决定调用哪个 Server 和哪个 Tool。

路由依据：

1. tool_name。
2. capability。
3. tenant_scope。
4. user permission。
5. server health。
6. latency。
7. cost。
8. version。
9. risk_level。

如果同一个能力有多个 Server，可以根据健康状态、版本和租户策略选择。

## 48.8 Policy Engine

Policy Engine 是安全核心。

它判断：

1. 用户是否有权使用该工具。
2. Agent 是否有权调用该工具。
3. 租户是否启用该 MCP Server。
4. 数据分类是否允许进入上下文。
5. 是否需要人工确认。
6. 是否允许工具结果转发给其他 Agent。
7. 是否触发高风险策略。

输入示例：

```json
{
  "user_id": "user_123",
  "agent_id": "agent.report.v1",
  "tenant_id": "tenant_a",
  "tool": "query_order_metrics",
  "arguments": {
    "region": "east_china"
  },
  "task_context": {
    "purpose": "weekly_business_report"
  }
}
```

输出示例：

```json
{
  "decision": "allow",
  "constraints": {
    "data_level": "aggregate_only",
    "max_rows": 1000
  },
  "requires_confirmation": false
}
```

## 48.9 身份和授权

企业 MCP 平台要区分：

1. 用户身份。
2. Agent 身份。
3. Host 身份。
4. MCP Server 身份。
5. 服务实例身份。

推荐使用 on-behalf-of 模型：Agent 代表用户执行任务，工具访问不能突破用户权限。

例如数据查询应同时满足：

1. 用户有 metrics.read。
2. Agent 被允许使用数据工具。
3. 租户启用了该 MCP Server。
4. 任务目的符合数据策略。

## 48.10 工具执行和沙箱

不同 MCP Server 风险不同。

低风险：知识库搜索、只读文档。

中风险：数据库只读查询。

高风险：文件写入、浏览器自动化、发送消息。

极高风险：终端、代码执行、生产变更。

沙箱策略：

1. 文件系统 roots 限制。
2. 网络 allowlist。
3. 环境变量过滤。
4. 命令白名单。
5. CPU / 内存 / 时间限制。
6. 输出大小限制。
7. 高风险动作确认。

## 48.11 Context Manager

工具结果不能直接全量塞给模型。

Context Manager 负责：

1. 结果裁剪。
2. 摘要。
3. 脱敏。
4. 来源标记。
5. 可信级别标记。
6. 引用生成。
7. taint tracking。
8. Artifact 存储。

例如数据库查询返回 10000 行，Context Manager 可以只给模型聚合摘要和 Artifact 引用。

## 48.12 Prompt Injection 防御

企业 MCP 平台必须防间接注入。

防线：

1. Resource 和 Tool Result 标记来源。
2. 外部内容标记为 untrusted。
3. 指令与数据分离。
4. 不可信内容不能触发高风险工具。
5. 高风险动作需要确认。
6. taint 随 Artifact 和 A2A 消息传播。
7. 红队测试。

这部分是面试加分点。

## 48.13 Trace、Audit 和 Replay

平台必须记录：

1. 谁调用。
2. 代表哪个用户。
3. 调用了哪个工具。
4. 参数是什么。
5. 权限决策是什么。
6. 工具返回了什么。
7. 哪些内容进入上下文。
8. 最终回答引用了什么。
9. 是否人工确认。
10. 是否发生错误或重试。

Trace 用于排障，Audit 用于合规，Replay 用于复现和 eval。

## 48.14 成本、延迟和并发

平台要支持：

1. 每用户调用配额。
2. 每租户成本预算。
3. 每工具并发限制。
4. 超时和重试。
5. 缓存。
6. 批处理。
7. 大结果 Artifact 化。
8. 降级策略。

否则 MCP 工具平台很容易被 Agent 的循环调用拖垮。

## 48.15 开发者门户和审核

开发者门户提供：

1. MCP Server 模板。
2. Tool schema 规范。
3. 本地调试。
4. Manifest 校验。
5. 安全规范。
6. Eval 工具。
7. 提交审核。
8. 上架状态。
9. Trace 查看。

审核包括：

1. Manifest 完整性。
2. 权限合理性。
3. 安全风险。
4. 工具质量。
5. 文档质量。
6. 维护 owner。

## 48.16 多租户隔离

多租户要隔离：

1. Server 可见性。
2. Tool 权限。
3. Resource 访问。
4. Artifact。
5. Trace。
6. Audit。
7. 配额。
8. 配置。

服务发现时就要考虑 tenant_scope。不能让 A 租户看到 B 租户的私有工具。

## 48.17 评估体系

企业 MCP 平台需要评估：

1. Tool selection accuracy。
2. Argument validity。
3. Tool success rate。
4. Task success rate。
5. Safety violation rate。
6. Citation accuracy。
7. Latency。
8. Cost。
9. Prompt injection 防御率。
10. 权限拦截准确率。

需要支持 tool simulator 和 trace replay。

## 48.18 一个完整请求流程

用户请求：

```text
帮我分析上周华东区域退款率升高原因。
```

流程：

1. Agent 判断需要数据和知识库。
2. Host 通过 MCP Gateway 请求 query_order_metrics。
3. Gateway 校验用户、Agent、租户。
4. Policy Engine 限制 aggregate_only。
5. Tool Router 选择 analytics MCP Server。
6. MCP Server 执行聚合查询。
7. Context Manager 脱敏、生成引用和 Artifact。
8. Agent 调用知识库 MCP Server 查询指标定义。
9. Agent 生成结论，保留证据链。
10. Trace 和 Audit 记录全链路。

## 48.19 面试回答模板

可以这样组织：

```text
我会把企业 MCP 工具平台分成 MCP Gateway、Registry、Tool Router、Policy Engine、Context Manager、MCP Server Runtime、Observability 和 Developer Portal。

Registry 管理 Server、Tools、Resources、Prompts 和 metadata。Gateway 是统一入口，负责认证、租户、限流、路由、参数校验和 trace。Policy Engine 做用户、Agent、租户和数据策略校验。Context Manager 处理工具结果脱敏、裁剪、来源标记、引用和 Artifact。Server Runtime 托管不同 MCP Server，如知识库、数据库、IDE、浏览器、终端和业务 API。

安全上使用 OBO 授权、最小权限、roots、沙箱、工具风险分级、高风险确认、prompt injection 防御和审计。性能上做预算、超时、重试、缓存、并发限制和降级。开发者门户提供模板、schema 规范、调试、eval 和审核。最终通过 trace、audit、replay 和 benchmark 评估质量和安全。
```

## 48.20 常见扣分点

### 48.20.1 只说接很多 MCP Server

系统设计要讲 Registry、Gateway、Policy、Context、Trace、Audit、Eval。

### 48.20.2 忽略权限和多租户

企业平台必考点。

### 48.20.3 忽略本地沙箱

IDE、终端、浏览器接入时必须讲沙箱。

### 48.20.4 不讲工具结果如何进上下文

这是 MCP 平台和普通 API Gateway 的关键区别。

### 48.20.5 不讲 eval 和 replay

没有评估，平台无法持续迭代。

## 48.21 面试高频题

### 题 1：企业 MCP 工具平台的核心模块有哪些？

参考回答：

包括 MCP Gateway、MCP Registry、Tool Router、Policy Engine、Context Manager、MCP Server Runtime、Trace/Audit/Replay、Developer Portal 和 Review Console。

### 题 2：为什么需要 MCP Gateway？

参考回答：

Gateway 统一处理认证、租户、路由、限流、参数校验、权限检查、trace、错误归一化和结果脱敏。否则 Agent 直接连接各 MCP Server 会导致权限和审计分散。

### 题 3：如何做权限控制？

参考回答：

使用用户身份、Agent 身份、Host 身份、租户和任务上下文联合判断。推荐 OBO 授权，工具访问不能超过用户权限。Policy Engine 做最小权限、数据分类、高风险确认和上下文转发控制。

### 题 4：MCP 工具结果如何进入模型上下文？

参考回答：

工具结果先经过 Context Manager，做裁剪、摘要、脱敏、来源标记、可信级别、引用和 Artifact 存储。不要把原始大结果直接塞进上下文。

### 题 5：如何防 prompt injection？

参考回答：

Resource 和 Tool Result 标记来源和可信级别，不可信内容作为数据而非指令，taint 随上下文传播，高风险工具调用前由 Policy Engine 拦截或要求确认，浏览器和终端等高风险工具放入沙箱。

## 48.22 小练习

1. 画出企业 MCP 工具平台架构图，包含 Gateway、Registry、Policy、Context、Server Runtime。
2. 设计一个 MCP Tool metadata，包含 risk_level、required_scopes 和 owner。
3. 设计一个数据库 MCP Server 的权限策略。
4. 设计一个浏览器 MCP Server 的 prompt injection 防御策略。
5. 思考：为什么企业 MCP 工具平台不能只用普通 API Gateway 替代？

## 48.23 本章小结

本章用系统设计题的方式讲了企业 MCP 工具平台。

一个生产级企业 MCP 工具平台需要 Registry 管理能力目录，Gateway 统一入口，Tool Router 做路由，Policy Engine 做权限和安全，Context Manager 管理工具结果进入上下文，MCP Server Runtime 托管工具服务，Observability 支撑 trace、audit 和 replay，Developer Portal 支撑开发者生态。它不是普通 API Gateway，而是面向模型和 Agent 的工具资源接入层。

你可以把本章重点记成一句话：

> 企业 MCP 平台的核心价值，是把分散工具变成可发现、可授权、可审计、可安全进入模型上下文的标准能力。

下一章我们会继续系统设计题：设计一个跨 Agent 协作系统。
