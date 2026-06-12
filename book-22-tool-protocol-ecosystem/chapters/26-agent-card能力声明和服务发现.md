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

## 26.0 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已按 `WRITING_PLAN.md` 核对 A2A 官方协议规范和 Agent Discovery 主题文档。正文采用这些公开资料中的稳定抽象：Agent Card 是远程 Agent 的自描述 manifest，服务发现可以通过 well-known URI、注册中心 / catalog、直接配置或定制发现机制完成；Agent Card 中与本章最相关的字段包括身份、描述、版本、supported interfaces、capabilities、default input / output modes、skills、security schemes、security requirements、provider、documentation、signatures 和扩展 Agent Card。

本讲不是逐字段翻译某个版本的协议定义，也不实现真实 A2A server、registry、OAuth 流程、签名校验或远程调用。不同实现可以对字段命名、扩展字段、缓存策略和发现方式做工程取舍；正文只保留面试和工程设计中稳定的结构：能力声明、技能粒度、接口声明、权限要求、服务发现、版本缓存、路由选择、trace 和 eval。

第二轮补充重点是：

1. 把“Agent Card 是能力契约”落到可审计字段，而不是泛泛说服务说明书。
2. 区分公开 Agent Card 和需要认证后获取的 extended Agent Card，避免把敏感能力、内部 URL 或租户策略无条件公开。
3. 增加稳定 MathJax 公式，用覆盖率指标表达 Agent Card 质量、发现质量、路由质量和上线门禁。
4. 补一个 0 依赖 Python demo，用 toy Agent Card 和委派请求审计字段完整度、skill 声明、supported interface、权限、版本缓存、发现匹配、路由决策、trace 和 eval。

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

不同 A2A 实现可以有不同字段。按官方 A2A 规范的稳定口径，Agent Card 更像一个自描述 manifest：它不仅有 `name`、`description`、`version`，还要声明 `supportedInterfaces`、`capabilities`、`skills`、默认输入输出模式和安全要求。工程落地时，不必强行让业务字段逐字等同于规范字段，但语义上至少要覆盖以下部分。

### 26.3.1 基本身份信息

基本身份信息回答“这个 Agent 是谁”。

常见字段包括：

1. id：稳定唯一标识。
2. name：人类可读名称。
3. description：简短描述。
4. version：版本号。
5. owner：维护团队或负责人。
6. provider：提供方。
7. supported interfaces：支持的调用接口、协议绑定和调用入口。
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
  "supported_interfaces": [
    {
      "url": "https://agents.example.internal/a2a/data-analysis",
      "protocol_binding": "HTTP+JSON",
      "protocol_version": "1.0"
    }
  ],
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

### 26.3.8 公开 Card 与扩展 Card

公开 Agent Card 通常用于“先让调用方知道这个 Agent 是否值得进一步连接”。它不应该无条件暴露所有内部信息，例如内部工具名、私有 endpoint、租户策略、敏感数据分类、供应商细节或高风险能力细节。

更稳妥的做法是分两层：

1. public Agent Card：暴露基本身份、粗粒度能力、公开接口、认证方式和文档入口。
2. extended Agent Card：调用方通过认证授权后，再获取更详细的技能、输入输出模式、权限 scope、内部限制、服务等级、租户可见性和治理信息。

这样做的核心原因是：服务发现要可用，但能力目录本身也可能是敏感信息。Agent Card 不是越详细越好，而是要在“可发现”和“最小披露”之间平衡。

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

常见服务发现方式有四类。

### 26.5.1 Well-known URI

一种标准化方式是让 Agent 在固定位置暴露公开 Agent Card，例如类似：

```text
https://agent.example.com/.well-known/agent-card.json
```

调用方先读取公开 Card，判断这个远程 Agent 的身份、接口、认证方式、粗粒度能力和文档入口。如果任务需要更详细的 skill、scope 或租户信息，再通过受控接口获取 extended Agent Card。

这种方式的优点是跨系统发现成本低，缺点是需要谨慎控制公开信息，避免把内部能力目录、敏感 endpoint、租户策略或高风险能力细节直接暴露出去。

### 26.5.2 静态注册

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

### 26.5.3 注册中心

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

### 26.5.4 语义检索

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

## 26.10 Agent Card 审计指标与最小 demo

如果把 Agent Card 当成“介绍页面”，它很容易写成营销文案；如果把它当成“协议对象”，就应该能被审计。

设第 `i` 个 Agent Card 或发现样本为：

```math
a_i=(f_i,s_i,u_i,p_i,r_i,e_i,v_i,t_i,q_i)
```

其中：

1. `f_i` 是基础字段集合，例如名称、描述、版本、owner、capabilities、skills 和默认输入输出模式。
2. `s_i` 是 skill 声明，包括 skill id、描述、tags、输入输出模式、示例和权限要求。
3. `u_i` 是 supported interface，包括 URL、协议绑定、协议版本和租户范围。
4. `p_i` 是权限和安全声明，包括认证方式、scope、数据分类和访问控制。
5. `r_i` 是服务发现结果，例如 well-known、registry、direct config 或语义检索。
6. `e_i` 是 extended Agent Card 控制信息。
7. `v_i` 是版本、缓存、签名和回滚信息。
8. `t_i` 是 trace 字段。
9. `q_i` 是 eval 标签和 golden case。

对任意审计维度 `k`，可以用覆盖率表达：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{pass}_k(a_i)]
```

本章建议至少看十个指标：

```math
\begin{aligned}
C_{\mathrm{field}} &= \mathrm{coverage}(\mathrm{required\ card\ fields})\\
C_{\mathrm{skill}} &= \mathrm{coverage}(\mathrm{skill\ declarations})\\
C_{\mathrm{interface}} &= \mathrm{coverage}(\mathrm{supported\ interfaces})\\
C_{\mathrm{security}} &= \mathrm{coverage}(\mathrm{security\ requirements})\\
C_{\mathrm{version}} &= \mathrm{coverage}(\mathrm{version\ and\ cache})\\
C_{\mathrm{discover}} &= \mathrm{match}(\mathrm{discovered\ agent},\mathrm{expected\ agent})\\
C_{\mathrm{route}} &= \mathrm{match}(\mathrm{routing\ decision},\mathrm{expected\ decision})\\
C_{\mathrm{extended}} &= \mathrm{coverage}(\mathrm{extended\ card\ control})\\
C_{\mathrm{trace}} &= \mathrm{coverage}(\mathrm{trace\ fields})\\
C_{\mathrm{eval}} &= \mathrm{coverage}(\mathrm{eval\ labels})
\end{aligned}
```

最后可以定义一个 Agent Card 门禁：

```math
G_{\mathrm{agent\_card}}=
\mathbf{1}\left[
\min(C_{\mathrm{field}},C_{\mathrm{skill}},C_{\mathrm{interface}},C_{\mathrm{security}},C_{\mathrm{version}},
C_{\mathrm{discover}},C_{\mathrm{route}},C_{\mathrm{extended}},C_{\mathrm{trace}},C_{\mathrm{eval}})\ge \tau
\right]
```

这里的 `\tau` 是上线阈值，例如 0.9。这个公式的意思不是说真实系统只需要十个指标，而是强调：Agent Card 质量、服务发现、权限、版本缓存、trace 和 eval 必须一起看。一个 Agent 即使能被发现，只要 Card 字段不完整、接口不安全、版本不可复现或 trace 缺失，就不应该进入高风险多 Agent 协作链路。

下面的 demo 只做静态审计。它模拟三个 Agent Card 和六个发现 / 路由样本，覆盖 happy path、权限不足、输出模式不支持、未知 skill、低质量 Card、trace 缺字段等情况。

```python
from pprint import pprint

REQUIRED_CARD_FIELDS = {
    "name", "description", "version", "supported_interfaces",
    "capabilities", "skills", "default_input_modes", "default_output_modes",
    "security_schemes", "security_requirements", "owner", "cache"
}
REQUIRED_SKILL_FIELDS = {
    "id", "name", "description", "tags", "input_modes", "output_modes", "examples"
}
REQUIRED_TRACE_FIELDS = {
    "case_id", "task", "selected_agent", "selected_skill", "decision", "reason", "card_version"
}
REQUIRED_EVAL_LABELS = {"expected_agent", "expected_skill", "expected_decision", "risk"}

AGENT_CARDS = [
    {
        "id": "agent.data.v2",
        "name": "Business Data Analyst",
        "description": "Analyze business metrics and produce cited anomaly reports.",
        "version": "2.3.0",
        "owner": "data-platform",
        "supported_interfaces": [
            {
                "url": "https://agents.example/a2a/data",
                "protocol_binding": "HTTP+JSON",
                "protocol_version": "1.0",
                "tenant": "enterprise",
            }
        ],
        "capabilities": {
            "streaming": True,
            "push_notifications": True,
            "extended_agent_card": True,
        },
        "default_input_modes": ["text/plain", "application/json"],
        "default_output_modes": ["application/json", "text/markdown"],
        "security_schemes": {"oauth2": {"type": "oauth2"}},
        "security_requirements": [{"oauth2": ["metrics.read", "reports.write"]}],
        "skills": [
            {
                "id": "metric_anomaly_analysis",
                "name": "Metric anomaly analysis",
                "description": "Find likely causes for metric movement with evidence.",
                "tags": ["metrics", "anomaly", "payments"],
                "input_modes": ["application/json"],
                "output_modes": ["application/json", "text/markdown"],
                "examples": ["Why did refund rate increase last week?"],
                "security_requirements": [{"oauth2": ["metrics.read"]}],
            }
        ],
        "discovery": {"methods": ["registry", "well_known"], "visibility": "private"},
        "cache": {"etag": "data-v2.3.0", "max_age_seconds": 600},
        "signatures": ["jws-placeholder"],
    },
    {
        "id": "agent.legal.v1",
        "name": "Legal Review Agent",
        "description": "Review contracts and payment terms for business risk.",
        "version": "1.5.1",
        "owner": "legal-ops",
        "supported_interfaces": [
            {
                "url": "https://agents.example/a2a/legal",
                "protocol_binding": "HTTP+JSON",
                "protocol_version": "1.0",
                "tenant": "enterprise",
            }
        ],
        "capabilities": {
            "streaming": False,
            "push_notifications": True,
            "extended_agent_card": False,
        },
        "default_input_modes": ["text/plain", "application/pdf"],
        "default_output_modes": ["text/markdown"],
        "security_schemes": {"oauth2": {"type": "oauth2"}},
        "security_requirements": [{"oauth2": ["contracts.read"]}],
        "skills": [
            {
                "id": "contract_payment_terms_review",
                "name": "Contract payment terms review",
                "description": "Identify unfavorable payment terms and cite contract clauses.",
                "tags": ["contract", "payment", "risk"],
                "input_modes": ["application/pdf", "text/plain"],
                "output_modes": ["text/markdown"],
                "examples": ["Check this vendor contract for payment-cycle risk."],
                "security_requirements": [{"oauth2": ["contracts.read"]}],
            }
        ],
        "discovery": {"methods": ["registry"], "visibility": "private"},
        "cache": {"etag": "legal-v1.5.1", "max_age_seconds": 900},
        "signatures": ["jws-placeholder"],
    },
    {
        "id": "agent.writer.v1",
        "name": "Report Writer",
        "description": "Generate polished reports from approved facts.",
        "owner": "growth-team",
        "supported_interfaces": [
            {"url": "http://agents.example/a2a/writer", "protocol_binding": "HTTP+JSON"}
        ],
        "capabilities": {"streaming": True},
        "default_input_modes": ["text/plain"],
        "default_output_modes": ["text/markdown"],
        "security_schemes": {},
        "security_requirements": [],
        "skills": [
            {
                "id": "report_generation",
                "name": "Report generation",
                "description": "Turn structured facts into a report.",
                "tags": ["report"],
                "input_modes": ["text/plain"],
                "output_modes": ["text/markdown"],
            }
        ],
        "discovery": {"methods": ["direct_config"], "visibility": "public"},
        "cache": {},
        "signatures": [],
    },
]

DISCOVERY_CASES = [
    {
        "id": "refund_analysis_ok",
        "task": "analyze refund rate increase in payments",
        "required_tags": {"payments", "anomaly"},
        "input_mode": "application/json",
        "output_mode": "application/json",
        "scopes": {"metrics.read", "reports.write"},
        "tenant": "enterprise",
        "requires_streaming": True,
        "needs_extended": True,
        "expected_agent": "agent.data.v2",
        "expected_skill": "metric_anomaly_analysis",
        "expected_decision": "allow",
        "risk": "medium",
        "trace": REQUIRED_TRACE_FIELDS,
    },
    {
        "id": "legal_review_ok",
        "task": "review vendor contract payment terms",
        "required_tags": {"contract", "payment"},
        "input_mode": "application/pdf",
        "output_mode": "text/markdown",
        "scopes": {"contracts.read"},
        "tenant": "enterprise",
        "requires_streaming": False,
        "needs_extended": False,
        "expected_agent": "agent.legal.v1",
        "expected_skill": "contract_payment_terms_review",
        "expected_decision": "allow",
        "risk": "high",
        "trace": REQUIRED_TRACE_FIELDS,
    },
    {
        "id": "missing_scope_bad",
        "task": "analyze payment anomalies without report permission",
        "required_tags": {"payments", "anomaly"},
        "input_mode": "application/json",
        "output_mode": "application/json",
        "scopes": {"metrics.read"},
        "tenant": "enterprise",
        "requires_streaming": True,
        "needs_extended": False,
        "expected_agent": "agent.data.v2",
        "expected_skill": "metric_anomaly_analysis",
        "expected_decision": "block",
        "risk": "medium",
        "trace": {"case_id", "task", "selected_agent", "decision", "reason"},
    },
    {
        "id": "unsupported_output_bad",
        "task": "get structured legal JSON output",
        "required_tags": {"contract", "payment"},
        "input_mode": "application/pdf",
        "output_mode": "application/json",
        "scopes": {"contracts.read"},
        "tenant": "enterprise",
        "requires_streaming": False,
        "needs_extended": False,
        "expected_agent": "agent.legal.v1",
        "expected_skill": "contract_payment_terms_review",
        "expected_decision": "block",
        "risk": "high",
        "trace": REQUIRED_TRACE_FIELDS,
    },
    {
        "id": "unknown_skill_bad",
        "task": "generate refund approval workflow",
        "required_tags": {"refund", "workflow"},
        "input_mode": "text/plain",
        "output_mode": "text/markdown",
        "scopes": set(),
        "tenant": "enterprise",
        "requires_streaming": False,
        "needs_extended": False,
        "expected_agent": None,
        "expected_skill": None,
        "expected_decision": "block",
        "risk": "medium",
        "trace": {"case_id", "task", "decision", "reason"},
    },
    {
        "id": "bad_card_quality",
        "task": "write a report from approved facts",
        "required_tags": {"report"},
        "input_mode": "text/plain",
        "output_mode": "text/markdown",
        "scopes": set(),
        "tenant": "enterprise",
        "requires_streaming": True,
        "needs_extended": False,
        "expected_agent": "agent.writer.v1",
        "expected_skill": "report_generation",
        "expected_decision": "block",
        "risk": "low",
        "trace": {"case_id", "task", "selected_agent", "selected_skill", "decision", "reason"},
    },
]


def ratio(ok, total):
    return round(ok / total, 3) if total else 1.0


def card_completeness(card):
    return ratio(sum(1 for f in REQUIRED_CARD_FIELDS if card.get(f)), len(REQUIRED_CARD_FIELDS))


def skill_completeness(card):
    skills = card.get("skills", [])
    if not skills:
        return 0.0
    scores = [
        ratio(sum(1 for f in REQUIRED_SKILL_FIELDS if skill.get(f)), len(REQUIRED_SKILL_FIELDS))
        for skill in skills
    ]
    return round(sum(scores) / len(scores), 3)


def interface_ready(card):
    interfaces = card.get("supported_interfaces", [])
    if not interfaces:
        return 0.0
    good = 0
    for item in interfaces:
        ok = bool(
            item.get("url", "").startswith("https://")
            and item.get("protocol_binding")
            and item.get("protocol_version")
        )
        good += int(ok)
    return ratio(good, len(interfaces))


def cache_ready(card):
    cache = card.get("cache", {})
    return 1.0 if card.get("version") and cache.get("etag") and cache.get("max_age_seconds") else 0.0


def security_ready(card):
    if card.get("security_requirements") and card.get("security_schemes"):
        return 1.0
    return 1.0 if card.get("discovery", {}).get("visibility") == "public" else 0.0


def find_skill(card, required_tags):
    for skill in card.get("skills", []):
        if required_tags.issubset(set(skill.get("tags", []))):
            return skill
    return None


def required_scopes(card, skill):
    scopes = set()
    for req in card.get("security_requirements", []):
        for values in req.values():
            scopes.update(values)
    for req in skill.get("security_requirements", []):
        for values in req.values():
            scopes.update(values)
    return scopes


def route(case):
    matches = []
    for card in AGENT_CARDS:
        skill = find_skill(card, case["required_tags"])
        if skill:
            matches.append((card, skill))
    if not matches:
        return None, None, "block", "NO_MATCHING_SKILL"

    scored = sorted(
        matches,
        key=lambda pair: (
            card_completeness(pair[0]),
            skill_completeness(pair[0]),
            interface_ready(pair[0]),
            security_ready(pair[0]),
        ),
        reverse=True,
    )
    card, skill = scored[0]

    if card_completeness(card) < 0.9 or skill_completeness(card) < 0.9:
        return card, skill, "block", "CARD_OR_SKILL_INCOMPLETE"
    if interface_ready(card) < 1.0:
        return card, skill, "block", "INTERFACE_NOT_PRODUCTION_READY"
    if case["input_mode"] not in skill.get("input_modes", card.get("default_input_modes", [])):
        return card, skill, "block", "UNSUPPORTED_INPUT_MODE"
    if case["output_mode"] not in skill.get("output_modes", card.get("default_output_modes", [])):
        return card, skill, "block", "UNSUPPORTED_OUTPUT_MODE"
    if case["requires_streaming"] and not card.get("capabilities", {}).get("streaming"):
        return card, skill, "block", "STREAMING_NOT_SUPPORTED"
    if case["needs_extended"] and not card.get("capabilities", {}).get("extended_agent_card"):
        return card, skill, "block", "EXTENDED_CARD_NOT_SUPPORTED"
    if not required_scopes(card, skill).issubset(case["scopes"]):
        return card, skill, "block", "MISSING_REQUIRED_SCOPE"
    if cache_ready(card) < 1.0:
        return card, skill, "block", "CACHE_VERSION_NOT_READY"
    return card, skill, "allow", "OK"


def audit():
    selected = []
    discovery_ok = routing_ok = extended_ok = trace_ok = eval_ok = 0
    failed_cases = []
    for case in DISCOVERY_CASES:
        card, skill, decision, reason = route(case)
        selected_agent = card["id"] if card else None
        selected_skill = skill["id"] if skill else None
        selected.append((case["id"], selected_agent, selected_skill, decision, reason))

        expected_agent_match = selected_agent == case["expected_agent"]
        expected_skill_match = selected_skill == case["expected_skill"]
        expected_decision_match = decision == case["expected_decision"]
        discovery_ok += int(expected_agent_match and expected_skill_match)
        routing_ok += int(expected_agent_match and expected_skill_match and expected_decision_match)
        extended_ok += int(
            (not case["needs_extended"])
            or bool(card and card.get("capabilities", {}).get("extended_agent_card"))
        )
        trace_ok += int(REQUIRED_TRACE_FIELDS.issubset(case["trace"]))
        eval_ok += int(REQUIRED_EVAL_LABELS.issubset(case.keys()))
        if not (expected_agent_match and expected_skill_match and expected_decision_match):
            failed_cases.append(case["id"])

    metrics = {
        "agent_card_field_completeness": round(
            sum(card_completeness(c) for c in AGENT_CARDS) / len(AGENT_CARDS), 3
        ),
        "agent_skill_declaration_quality": round(
            sum(skill_completeness(c) for c in AGENT_CARDS) / len(AGENT_CARDS), 3
        ),
        "supported_interface_readiness": round(
            sum(interface_ready(c) for c in AGENT_CARDS) / len(AGENT_CARDS), 3
        ),
        "agent_card_security_coverage": round(
            sum(security_ready(c) for c in AGENT_CARDS) / len(AGENT_CARDS), 3
        ),
        "agent_card_version_cache_readiness": round(
            sum(cache_ready(c) for c in AGENT_CARDS) / len(AGENT_CARDS), 3
        ),
        "agent_discovery_match_quality": ratio(discovery_ok, len(DISCOVERY_CASES)),
        "agent_routing_decision_quality": ratio(routing_ok, len(DISCOVERY_CASES)),
        "extended_agent_card_control": ratio(extended_ok, len(DISCOVERY_CASES)),
        "agent_card_trace_readiness": ratio(trace_ok, len(DISCOVERY_CASES)),
        "agent_card_eval_coverage": ratio(eval_ok, len(DISCOVERY_CASES)),
    }
    failed_gates = [name for name, value in metrics.items() if value < 0.9]
    return {
        "selected": selected,
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "agent_card_gate_pass": not failed_gates and not failed_cases,
    }


pprint(audit(), sort_dicts=False)
```

这段 demo 的输出会显示：`refund_analysis_ok` 和 `legal_review_ok` 能正确路由；`missing_scope_bad` 因权限不足被拦截；`unsupported_output_bad` 因输出模式不匹配被拦截；`unknown_skill_bad` 因没有匹配 skill 被拦截；`bad_card_quality` 因 Card / skill 信息不完整被拦截。

关键指标中，`agent_card_field_completeness`、`supported_interface_readiness`、`agent_card_version_cache_readiness` 和 `agent_card_trace_readiness` 没过门禁。这个结果说明：即使发现和路由决策看起来正确，低质量 Agent Card、非 HTTPS interface、缺版本缓存和 trace 字段不足仍然会让系统不适合上线。

## 26.11 一个服务发现流程示例

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

## 26.12 面试高频题

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

## 26.13 小练习

1. 为一个“代码评审 Agent”设计 Agent Card，至少包含身份、能力、输入、输出、权限和交互模式。
2. 为一个“客服反馈聚类 Agent”写 3 个示例任务，帮助路由器理解它的能力边界。
3. 设计一个服务发现流程：用户要做“合同付款条款风险检查”，系统如何找到合适 Agent？
4. 列出 Agent Card 中哪些字段会影响权限校验。
5. 思考：如果两个 Agent Card 都声明自己能做“数据分析”，你会用哪些信号判断该调用谁？

## 26.14 本章小结

本章我们讲了 A2A 中的第一个核心抽象：Agent Card。

Agent Card 的作用不是写一份好看的介绍，而是把 Agent 的身份、能力、输入要求、输出契约、交互模式、权限、安全、版本、SLA 和治理信息结构化。它让多 Agent 系统可以做服务发现、能力匹配、权限过滤、路由选择、灰度发布和审计复现。

你可以把本章重点记成一句话：

> 没有 Agent Card，多 Agent 协作只能靠猜；有了 Agent Card，Agent 才能成为可发现、可调用、可治理的系统组件。

下一章我们会继续讲 A2A 的核心运行流程：Agent 间任务委派、状态同步和结果返回。也就是当调用方选中一个 Agent 后，任务到底如何发过去、如何跟踪、如何追问、如何失败、如何返回产物。
