# 第 26 章 Agent Card、能力声明和服务发现

上一章我们讲了 A2A 的背景：Agent 之间需要的不只是聊天，而是可发现、可委派、可追踪、可治理的协作关系。

本章进入 A2A 的第一个关键抽象：Agent Card。

如果你熟悉 Web API，可以把 Agent Card 粗略理解成“Agent 的服务说明书”；如果你熟悉 MCP，可以把它类比为 MCP Server 暴露 tools/resources/prompts 的能力列表。但 Agent Card 又不完全等同于 API 文档，因为 Agent 不是一个普通函数。Agent 往往能规划、追问、异步执行、调用自己的工具，并在任务过程中产生状态变化和产物。

本章要解决的问题是：

1. 一个 Agent 如何声明自己是谁？
2. 它如何声明自己能做什么、不能做什么？
3. 调用方如何发现合适的 Agent？
4. 多个 Agent 都能做同一件事时，如何选择？
5. Agent Card 里应该包含哪些字段？
6. 能力声明和权限、安全、版本治理有什么关系？

你可以先记住一句话：

> Agent Card 是 A2A 里的能力契约，它让 Agent 从“一个神秘聊天对象”变成“一个可发现、可匹配、可调用、可治理的服务化智能体”。

## 26.1 为什么需要 Agent Card

假设总控 Agent 收到用户任务：

```text
请分析过去一个月退款率升高的原因，并生成一份面向业务团队的报告。
```

总控 Agent 可能需要找这些下游 Agent：

1. 数据分析 Agent。
2. 报告写作 Agent。
3. 业务知识库 Agent。
4. 风控 Agent。
5. 图表生成 Agent。

问题是：总控 Agent 怎么知道谁能做什么？

如果靠硬编码，就会变成：

```text
如果是数据任务，调用 agent_data_v2。
如果是报告任务，调用 agent_report_cn。
如果是风控任务，调用 risk_agent_prod。
```

这在小 Demo 里能跑，但在企业系统里很快失控。Agent 会增加、下线、升级、灰度、换团队维护、变更权限、支持新任务类型。总控 Agent 不应该把这些信息都写死在 prompt 或代码里。

Agent Card 的作用就是把 Agent 的能力结构化暴露出来，让调用方可以发现、匹配和治理。

## 26.2 Agent Card 和传统 API 文档的区别

传统 API 文档通常描述：

1. Endpoint。
2. Method。
3. Request schema。
4. Response schema。
5. Error code。
6. Authentication。

Agent Card 当然也需要类似信息，但它还要描述 Agent 特有的语义：

1. 任务类型。
2. 能力边界。
3. 支持的交互模式。
4. 是否会追问。
5. 是否支持异步执行。
6. 是否支持流式状态更新。
7. 是否会调用外部工具。
8. 输入上下文要求。
9. 输出产物类型。
10. 风险和权限要求。

举个例子，一个普通翻译 API 可能是：输入文本，输出翻译文本。

但一个“本地化 Agent”可能会：

1. 判断目标地区。
2. 查询术语库。
3. 保留品牌词。
4. 追问语气风格。
5. 输出翻译版本、术语解释和不确定项。
6. 对敏感内容拒绝处理。

这就不能只用普通函数接口描述。

## 26.3 Agent Card 的核心字段

不同 A2A 实现可以有不同字段，但一个实用的 Agent Card 通常至少包含以下部分。

### 26.3.1 基本身份信息

基本身份信息回答“这个 Agent 是谁”。

常见字段包括：

1. id：稳定唯一标识。
2. name：人类可读名称。
3. description：简短描述。
4. version：版本号。
5. owner：维护团队或负责人。
6. provider：提供方。
7. endpoint：调用入口。
8. status：可用状态。

示例：

```json
{
  "id": "agent.data_analysis.v1",
  "name": "Data Analysis Agent",
  "description": "Analyze business metrics, detect anomalies, and produce structured insights.",
  "version": "1.4.2",
  "owner": "data-platform-team",
  "provider": "internal",
  "endpoint": "https://agents.example.internal/data-analysis",
  "status": "available"
}
```

这里的 id 和 version 非常关键。没有稳定 id，就难以审计；没有版本，就难以灰度、回滚和复现。

### 26.3.2 能力声明

能力声明回答“这个 Agent 能做什么”。

例如：

```json
{
  "capabilities": [
    {
      "name": "metric_anomaly_analysis",
      "description": "Analyze metric changes and identify possible root causes.",
      "input_modes": ["text", "structured_json"],
      "output_modes": ["summary", "report", "table"],
      "domains": ["orders", "payments", "growth"],
      "languages": ["zh-CN", "en-US"]
    }
  ]
}
```

好的能力声明应该避免两种极端。

第一种极端是过于抽象：

```text
我可以帮助你解决各种数据问题。
```

这对路由没有帮助。

第二种极端是过于碎片化：

```text
我可以计算 sum、avg、count、max、min。
```

这又退化成工具 API，而不是 Agent 能力。

能力声明应该介于两者之间，表达清楚“业务任务级能力”。例如：

1. 指标异常分析。
2. 合同风险审查。
3. 代码变更评审。
4. 测试计划生成。
5. 数据报表解释。
6. 用户反馈聚类。

### 26.3.3 输入要求

输入要求回答“调用我需要给什么”。

一个 Agent 可能不是任何输入都能处理。比如合同审查 Agent 至少需要合同文本或合同 URI；数据分析 Agent 需要指标名、时间范围和数据源权限；代码评审 Agent 需要 diff 或仓库引用。

示例：

```json
{
  "input_requirements": {
    "required": ["metric_name", "time_range"],
    "optional": ["segment", "baseline", "business_context"],
    "accepted_resources": ["kb://", "db://", "artifact://"],
    "max_context_tokens": 32000
  }
}
```

输入要求不是为了限制模型，而是为了减少无效委派。如果总控 Agent 在委派前就知道缺少 time_range，它可以先向用户追问，而不是把不完整任务丢给下游 Agent。

### 26.3.4 输出和产物

输出声明回答“调用我会得到什么”。

Agent 输出可能包括：

1. summary：摘要。
2. structured_result：结构化结果。
3. report：报告。
4. patch：代码补丁。
5. chart：图表。
6. table：表格。
7. artifact_uri：产物引用。
8. evidence：证据链。
9. confidence：置信度。
10. follow_up_questions：后续问题。

示例：

```json
{
  "output_contract": {
    "types": ["summary", "report", "table", "artifact"],
    "supports_citations": true,
    "supports_confidence": true,
    "artifact_retention_days": 30
  }
}
```

输出声明对上游编排非常重要。比如总控 Agent 如果最终要生成报告，就会优先选择能返回 report artifact 和 citation 的 Agent，而不是只返回一段聊天文本的 Agent。

### 26.3.5 交互模式

交互模式回答“调用过程怎么进行”。

常见交互模式包括：

1. sync：同步请求响应。
2. async：异步任务。
3. streaming：流式输出。
4. input_required：支持追问。
5. cancellable：支持取消。
6. resumable：支持恢复。
7. human_in_the_loop：需要人工确认。

示例：

```json
{
  "interaction": {
    "modes": ["async", "streaming", "input_required"],
    "supports_cancel": true,
    "supports_resume": false,
    "max_task_duration_seconds": 1800
  }
}
```

交互模式会影响编排策略。如果下游 Agent 是异步的，上游 Agent 就不能阻塞等待太久，而要订阅状态或安排回调。如果下游 Agent 可能追问，上游 Agent 就要准备把问题转给用户或从上下文中自动补齐。

### 26.3.6 权限和安全要求

权限字段回答“调用我需要什么权限，以及我会访问什么”。

示例：

```json
{
  "security": {
    "auth": "oauth2_on_behalf_of_user",
    "required_scopes": ["metrics.read", "reports.write"],
    "data_classification": ["internal", "confidential"],
    "external_data_sharing": false,
    "requires_user_confirmation": false
  }
}
```

这里有几个重点。

第一，Agent Card 应该说明它需要什么权限，而不是在运行时才突然报错。

第二，Agent Card 应该说明它可能访问什么类型的数据，方便上游判断是否允许传递上下文。

第三，权限不应该只靠 Agent 自己声明，还要由平台策略校验。Agent Card 是声明，不是最终安全边界。

### 26.3.7 质量、SLA 和成本

在企业系统里，选择 Agent 不只看能力，还要看质量、延迟和成本。

Agent Card 可以包含：

1. 平均延迟。
2. 最大任务时长。
3. 成本等级。
4. 成功率。
5. 最近健康状态。
6. 支持的并发。
7. 质量评分。
8. 适用环境。

示例：

```json
{
  "service_level": {
    "latency_class": "medium",
    "cost_class": "high",
    "availability": "99.5%",
    "recommended_for": ["deep_analysis", "business_report"],
    "not_recommended_for": ["low_latency_chat"]
  }
}
```

这对路由非常有用。同样是“数据分析”，一个 Agent 可能便宜快速但粗略，另一个 Agent 可能昂贵慢但准确。总控 Agent 或路由器应该根据任务要求选择。

## 26.4 一个较完整的 Agent Card 示例

下面是一个简化但比较完整的例子：

```json
{
  "id": "agent.business_metric_analyst.v1",
  "name": "Business Metric Analyst",
  "description": "Analyze business metric changes, identify anomalies, and generate evidence-based reports.",
  "version": "1.2.0",
  "owner": "data-platform-team",
  "endpoint": "https://agents.example.internal/business-metric-analyst",
  "status": "available",
  "capabilities": [
    {
      "name": "metric_anomaly_analysis",
      "description": "Identify possible root causes for metric increases or decreases.",
      "domains": ["orders", "payments", "growth"],
      "input_modes": ["text", "structured_json"],
      "output_modes": ["summary", "report", "table", "artifact"]
    },
    {
      "name": "business_report_generation",
      "description": "Generate business-facing reports with citations and caveats.",
      "domains": ["orders", "payments", "growth"],
      "input_modes": ["structured_json"],
      "output_modes": ["report", "artifact"]
    }
  ],
  "input_requirements": {
    "required": ["metric_name", "time_range"],
    "optional": ["segment", "baseline", "business_context"],
    "accepted_resources": ["kb://", "db://", "artifact://"],
    "max_context_tokens": 32000
  },
  "output_contract": {
    "types": ["summary", "report", "table", "artifact"],
    "supports_citations": true,
    "supports_confidence": true
  },
  "interaction": {
    "modes": ["async", "streaming", "input_required"],
    "supports_cancel": true,
    "max_task_duration_seconds": 1800
  },
  "security": {
    "auth": "oauth2_on_behalf_of_user",
    "required_scopes": ["metrics.read", "reports.write"],
    "data_classification": ["internal", "confidential"],
    "external_data_sharing": false
  },
  "service_level": {
    "latency_class": "medium",
    "cost_class": "medium",
    "recommended_for": ["root_cause_analysis", "weekly_business_review"]
  }
}
```

这个 Agent Card 不是为了让模型逐字阅读，而是为了让 Agent 平台、路由器、总控 Agent 和治理系统都能理解这个 Agent 的能力和边界。

## 26.5 服务发现：如何找到合适的 Agent

有了 Agent Card，还需要服务发现。

服务发现回答：

> 当我有一个任务时，应该找哪个 Agent？

常见服务发现方式有三类。

### 26.5.1 静态注册

最简单的方式是静态配置：

```json
{
  "agents": [
    "agent.data_analysis.v1",
    "agent.report_writer.v1",
    "agent.legal_review.v2"
  ]
}
```

优点是简单、可控。缺点是不够灵活，新增或下线 Agent 需要改配置。

适合早期系统、小规模团队或安全要求很高的场景。

### 26.5.2 注册中心

更常见的是注册中心。每个 Agent 发布自己的 Agent Card，平台统一管理。

注册中心可以支持：

1. 按能力搜索。
2. 按领域搜索。
3. 按版本过滤。
4. 按租户过滤。
5. 按权限过滤。
6. 按健康状态过滤。
7. 按成本和延迟排序。

例如，总控 Agent 可以查询：

```json
{
  "capability": "metric_anomaly_analysis",
  "domain": "payments",
  "language": "zh-CN",
  "required_scopes": ["metrics.read"],
  "max_latency_class": "medium"
}
```

注册中心返回候选 Agent 列表和匹配原因。

### 26.5.3 语义检索

有些能力很难完全靠标签匹配。例如用户说：

```text
帮我看看这份合同有没有对我们不利的付款条款。
```

这可能匹配“合同审查”“付款条款分析”“法务风险识别”等能力。此时可以对 Agent Card 的描述、能力说明和示例任务做向量检索或语义匹配。

但语义检索不能单独决定最终路由。因为它可能召回语义相近但权限不匹配、版本不稳定或不适合当前租户的 Agent。

更稳妥的方式是：

1. 语义召回候选 Agent。
2. 结构化条件过滤。
3. 权限和租户校验。
4. 健康状态和成本排序。
5. 必要时让总控 Agent 做最终选择。

## 26.6 Agent 路由：多个候选如何选择

服务发现找到候选 Agent 后，还要路由。

路由的输入通常包括：

1. 用户任务。
2. 任务类型。
3. 领域。
4. 语言。
5. 需要的输入输出。
6. 权限要求。
7. 延迟要求。
8. 成本限制。
9. 质量要求。
10. 历史表现。

### 26.6.1 基于规则的路由

规则路由简单可靠：

```text
如果任务类型是合同审查，优先选择 legal_review_agent。
如果任务涉及支付合规，必须同时调用 compliance_agent。
如果用户属于海外业务线，只允许调用 global_agents。
```

优点是可解释、容易审计。缺点是不够灵活，规则多了会难维护。

### 26.6.2 基于模型的路由

模型路由可以根据任务描述和 Agent Card 选择候选：

```text
用户任务：分析退款率升高原因。
候选 Agent：数据分析、客服反馈聚类、支付风控、报告写作。
模型判断：先调用数据分析和客服反馈聚类，再由报告写作汇总。
```

优点是灵活，适合开放任务。缺点是可能选择错误、不可解释或忽略权限。

所以模型路由必须受规则和权限系统约束。

### 26.6.3 混合路由

生产系统更常见的是混合路由：

1. 规则先过滤不允许的 Agent。
2. 语义检索召回可能相关的 Agent。
3. 模型根据 Agent Card 和任务选择组合。
4. 平台做权限、成本、健康状态校验。
5. 总控 Agent 执行任务委派。

这比单纯靠规则或单纯靠模型都更稳。

## 26.7 Agent Card 的版本治理

Agent 会升级。能力会变化。输入输出格式会变化。权限要求也会变化。

因此 Agent Card 必须支持版本治理。

### 26.7.1 为什么版本重要

假设一个 Agent 原来返回：

```json
{
  "summary": "...",
  "confidence": "high"
}
```

新版本改成：

```json
{
  "result": {
    "summary": "...",
    "confidence_score": 0.87
  }
}
```

如果上游 Agent 没有适配，就会解析失败。

所以 Agent Card 需要明确：

1. 当前版本。
2. 兼容版本范围。
3. 即将废弃字段。
4. breaking changes。
5. 灰度策略。
6. 回滚方式。

### 26.7.2 能力版本和 Agent 版本

一个 Agent 可以有整体版本，也可以有能力版本。

例如：

```json
{
  "version": "2.1.0",
  "capabilities": [
    {
      "name": "contract_review",
      "version": "1.3.0"
    },
    {
      "name": "payment_terms_analysis",
      "version": "2.0.0"
    }
  ]
}
```

这样可以避免因为一个能力升级而影响所有能力。

### 26.7.3 灰度和回滚

Agent Card 可以标记灰度状态：

```json
{
  "deployment": {
    "stage": "canary",
    "traffic_percent": 10,
    "fallback_agent": "agent.business_metric_analyst.v1"
  }
}
```

上游路由器可以根据租户、用户、任务类型和风险级别决定是否使用灰度 Agent。

## 26.8 Agent Card 与权限治理

Agent Card 不是权限系统，但它是权限治理的重要输入。

### 26.8.1 调用前权限检查

调用前应该检查：

1. 用户是否有权调用该 Agent。
2. 上游 Agent 是否有权委派给该 Agent。
3. 任务上下文是否允许传给该 Agent。
4. 该 Agent 是否允许访问所需资源。
5. 该 Agent 是否允许生成指定类型产物。

例如，一个外部供应商 Agent 即使有“合同审查”能力，也不一定能接收企业内部机密合同。

### 26.8.2 调用中权限约束

调用过程中，下游 Agent 可能请求更多上下文或工具权限。

此时不能因为它在 Agent Card 里声明了某个能力，就自动批准所有请求。平台需要根据当前任务、用户授权和数据分类做动态判断。

### 26.8.3 调用后结果治理

结果返回后，也要治理：

1. 结果是否包含敏感信息？
2. 是否能返回给原用户？
3. 是否能被其他 Agent 继续使用？
4. Artifact 保存多久？
5. Trace 是否需要脱敏？

这说明 Agent Card 只是一部分，完整治理还需要身份系统、权限系统、数据分类、审计和策略引擎。

## 26.9 Agent Card 的质量问题

Agent Card 本身也可能写得很差。

### 26.9.1 描述过度营销

例如：

```text
本 Agent 可以解决所有企业智能化问题。
```

这没有任何路由价值。好的描述应该具体、可验证、可限制。

### 26.9.2 能力边界不清

如果 Agent 只写“数据分析”，调用方不知道它能不能查实时数据、能不能处理隐私数据、能不能生成图表、能不能解释业务原因。

### 26.9.3 输入输出不稳定

如果 Agent Card 声明支持 report，但实际有时返回 markdown，有时返回 JSON，有时返回纯文本，上游编排会非常痛苦。

### 26.9.4 权限要求缺失

如果 Agent Card 不声明 required scopes，调用时才发现权限不足，会造成大量失败任务。

### 26.9.5 没有示例任务

示例任务对模型路由很有帮助。例如：

```json
{
  "examples": [
    {
      "input": "Analyze why refund rate increased last week.",
      "expected_capability": "metric_anomaly_analysis"
    }
  ]
}
```

示例可以帮助语义检索和模型路由更准确地理解能力边界。

## 26.10 一个服务发现流程示例

假设用户说：

```text
请检查这份供应商合同里有没有对付款周期不利的条款。
```

系统可以这样做：

1. 总控 Agent 抽取任务意图：合同审查、付款条款、风险识别。
2. 服务发现系统用语义检索召回 LegalReviewAgent、PaymentTermsAgent、ProcurementPolicyAgent。
3. 结构化过滤：要求支持中文合同、支持 artifact 输入、支持 citation 输出。
4. 权限过滤：合同密级为 confidential，只允许内部 Agent。
5. 健康状态过滤：排除不可用 Agent。
6. 路由器选择 LegalReviewAgent 和 PaymentTermsAgent 并行执行。
7. 总控 Agent 汇总两个结果，冲突部分请求人工复核。

这个流程里，Agent Card 提供了发现和匹配基础；权限系统保证不会把合同发给不该接收的 Agent；路由器负责在多个候选里做取舍。

## 26.11 面试高频题

### 题 1：Agent Card 是什么？为什么需要它？

参考回答：

Agent Card 是 Agent 的结构化能力声明，用来描述 Agent 的身份、能力、输入要求、输出契约、交互模式、权限要求、版本、SLA 和治理信息。它的价值是让 Agent 可以被发现、匹配、调用和审计，而不是靠硬编码或自然语言猜测。

### 题 2：Agent Card 里应该包含哪些字段？

参考回答：

至少包含基本身份信息、能力列表、输入要求、输出契约、交互模式、权限和安全要求、版本信息、服务状态、SLA、成本等级、维护方和示例任务。生产系统还会包含租户范围、数据分类、灰度状态、fallback Agent 和审计策略。

### 题 3：如何根据 Agent Card 做服务发现？

参考回答：

可以先通过语义检索或标签匹配召回候选 Agent，再按任务类型、领域、语言、输入输出要求、权限、租户、版本、健康状态、延迟和成本过滤，最后由规则或模型路由选择最合适的 Agent。最终调用前必须做权限校验。

### 题 4：Agent Card 和 MCP tool schema 有什么区别？

参考回答：

MCP tool schema 描述的是工具调用的输入输出，通常对应一个具体操作。Agent Card 描述的是一个具备自主规划和任务执行能力的 Agent，包括任务能力、交互模式、状态生命周期、产物类型、权限要求和版本治理。前者偏函数级，后者偏服务化智能体级。

### 题 5：Agent Card 为什么需要版本治理？

参考回答：

因为 Agent 能力、输入输出格式、权限要求和行为都会变化。如果没有版本治理，上游 Agent 可能因为字段变更、能力变更或行为变更而失败。版本治理支持兼容性声明、灰度、回滚、废弃字段通知和审计复现。

## 26.12 小练习

1. 为一个“代码评审 Agent”设计 Agent Card，至少包含身份、能力、输入、输出、权限和交互模式。
2. 为一个“客服反馈聚类 Agent”写 3 个示例任务，帮助路由器理解它的能力边界。
3. 设计一个服务发现流程：用户要做“合同付款条款风险检查”，系统如何找到合适 Agent？
4. 列出 Agent Card 中哪些字段会影响权限校验。
5. 思考：如果两个 Agent Card 都声明自己能做“数据分析”，你会用哪些信号判断该调用谁？

## 26.13 本章小结

本章我们讲了 A2A 中的第一个核心抽象：Agent Card。

Agent Card 的作用不是写一份好看的介绍，而是把 Agent 的身份、能力、输入要求、输出契约、交互模式、权限、安全、版本、SLA 和治理信息结构化。它让多 Agent 系统可以做服务发现、能力匹配、权限过滤、路由选择、灰度发布和审计复现。

你可以把本章重点记成一句话：

> 没有 Agent Card，多 Agent 协作只能靠猜；有了 Agent Card，Agent 才能成为可发现、可调用、可治理的系统组件。

下一章我们会继续讲 A2A 的核心运行流程：Agent 间任务委派、状态同步和结果返回。也就是当调用方选中一个 Agent 后，任务到底如何发过去、如何跟踪、如何追问、如何失败、如何返回产物。
