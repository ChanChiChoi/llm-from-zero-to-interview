# 第 27 章 Agent 间任务委派、状态同步和结果返回

上一章我们讲了 Agent Card。Agent Card 解决的是“怎么知道一个 Agent 是谁、能做什么、需要什么输入、能返回什么结果”。

但发现 Agent 只是第一步。真正运行时还要解决：任务怎么发过去？对方是否接受？执行到哪一步了？中途缺信息怎么办？失败了怎么表达？最后结果和产物怎么返回？

这就是本章要讲的内容：Agent 间任务委派、状态同步和结果返回。

如果把 A2A 看成一个协作协议，那么 Agent Card 是“名片”，Task 是“工单”，Status 是“工单状态”，Message 是“沟通过程”，Artifact 是“最终产物”。

你可以先记住本章主线：

> A2A 的运行核心不是一句“请你帮我做”，而是一个可追踪的任务生命周期。

## 27.1 为什么任务委派需要结构化

假设总控 Agent 对数据 Agent 说：

```text
帮我分析一下退款率为什么升高。
```

这句话对人类来说还算能理解，但对系统来说问题很多：

1. 退款率是哪条业务线？
2. 时间范围是什么？
3. 需要和哪个基线比较？
4. 输出是摘要、报告还是表格？
5. 是否可以查询数据库？
6. 是否可以访问用户级明细？
7. 任务多久必须完成？
8. 失败后能不能重试？
9. 结果要返回给谁？
10. 是否需要证据链？

自然语言可以表达任务意图，但不能可靠承载所有执行约束。因此 A2A 中的任务委派通常需要结构化字段。

一个任务委派请求至少应该回答：

1. 谁发起任务。
2. 委派给谁。
3. 要做什么。
4. 输入是什么。
5. 允许使用什么上下文。
6. 期望输出是什么。
7. 截止时间是什么。
8. 权限边界是什么。
9. 状态如何回传。
10. 如何处理失败和取消。

## 27.2 A2A Task 的基本结构

一个简化的 Task 可以长这样：

```json
{
  "task_id": "task_20260529_001",
  "parent_task_id": "task_root_001",
  "requester": {
    "agent_id": "agent.orchestrator.v1",
    "user_id": "user_123"
  },
  "assignee": {
    "agent_id": "agent.data_analysis.v1"
  },
  "goal": "Analyze why refund rate increased in the last 30 days.",
  "input": {
    "metric_name": "refund_rate",
    "time_range": {
      "start": "2026-04-29",
      "end": "2026-05-29"
    },
    "segments": ["region", "payment_method"]
  },
  "context": {
    "resources": [
      "kb://docs/refund-rate-definition",
      "dbschema://analytics/orders"
    ]
  },
  "output_requirements": {
    "format": "report",
    "include_citations": true,
    "include_confidence": true
  },
  "constraints": {
    "deadline_seconds": 900,
    "max_cost_class": "medium",
    "data_policy": "aggregate_only"
  },
  "callback": {
    "mode": "event_stream",
    "url": "a2a://agent.orchestrator.v1/tasks/task_20260529_001/events"
  }
}
```

这个例子里有几个关键点：

1. task_id 用于追踪和幂等。
2. parent_task_id 表示这是某个大任务的子任务。
3. requester 说明任务代表谁发起。
4. assignee 说明由谁执行。
5. goal 是自然语言目标。
6. input 是结构化输入。
7. context 是允许使用的上下文资源。
8. output_requirements 说明期望结果。
9. constraints 限制时间、成本和数据策略。
10. callback 指定状态回传方式。

这比一句 prompt 稳定得多。

## 27.3 任务生命周期状态机

A2A 的核心是任务生命周期。一个常见状态机如下：

```text
submitted
  -> accepted
    -> running
      -> input_required
      -> completed
      -> failed
      -> cancelled
      -> expired
  -> rejected
```

每个状态都应该有明确语义。

### 27.3.1 submitted

submitted 表示任务已经提交给下游 Agent，但还没有被确认接受。

这一状态常见于异步系统，因为任务可能先进入队列。

### 27.3.2 accepted

accepted 表示下游 Agent 已经接受任务，并承诺尝试执行。

accepted 不等于 completed。它只说明任务被接收，不说明任务一定会成功。

### 27.3.3 rejected

rejected 表示下游 Agent 拒绝任务。

常见原因包括：

1. 能力不匹配。
2. 输入缺失。
3. 权限不足。
4. 任务风险过高。
5. 当前负载过高。
6. 版本不兼容。

rejected 通常不应该被当成系统故障。它可能是合理拒绝。

### 27.3.4 running

running 表示任务正在执行。

此时下游 Agent 可能会调用自己的工具、读取资源、生成中间产物或继续拆分子任务。

### 27.3.5 input_required

input_required 表示下游 Agent 需要更多信息才能继续。

例如：

```json
{
  "status": "input_required",
  "question": "Should refund orders be included in gross revenue calculation?",
  "required_fields": ["revenue_definition"]
}
```

这不是失败，而是多轮协作的一部分。

### 27.3.6 completed

completed 表示任务完成，并且应该附带结果或产物。

### 27.3.7 failed

failed 表示任务失败。

失败需要带错误类型、错误信息、是否可重试、已经完成的部分和 trace 引用。

### 27.3.8 cancelled

cancelled 表示任务被取消。

取消可能由用户触发，也可能由上游 Agent、策略系统或超时控制触发。

### 27.3.9 expired

expired 表示任务超过 deadline，没有在约定时间内完成。

expired 和 failed 不完全一样。expired 通常说明时间约束失败，而不一定说明任务本身无法完成。

## 27.4 状态事件的结构

状态同步最好不要只发一段文字，而要有结构化事件。

例如：

```json
{
  "event_id": "evt_001",
  "task_id": "task_20260529_001",
  "timestamp": "2026-05-29T10:15:30Z",
  "source_agent": "agent.data_analysis.v1",
  "status": "running",
  "message": "Querying aggregate refund metrics by region.",
  "progress": {
    "percent": 35,
    "stage": "data_query"
  },
  "trace_id": "trace_abc"
}
```

状态事件通常包含：

1. event_id。
2. task_id。
3. timestamp。
4. source_agent。
5. status。
6. message。
7. progress。
8. artifact_refs。
9. error。
10. trace_id。

这些字段让上游 Agent、平台 UI、日志系统和审计系统都能理解任务发生了什么。

## 27.5 状态同步方式

状态同步有几种常见方式。

### 27.5.1 轮询

上游 Agent 定期查询：

```text
GET /tasks/task_123/status
```

优点是简单。缺点是延迟高、浪费请求，不适合大量长任务。

### 27.5.2 回调

下游 Agent 在状态变化时调用上游提供的 callback。

优点是实时。缺点是需要处理回调鉴权、重试、幂等和网络失败。

### 27.5.3 事件流

通过 SSE、WebSocket、消息队列或事件总线推送状态。

优点是适合流式任务和多订阅者。缺点是系统复杂度更高。

### 27.5.4 混合模式

生产系统通常会组合使用：

1. 短任务用同步响应。
2. 中等任务用轮询或回调。
3. 长任务用事件流。
4. UI 展示订阅事件流。
5. 失败恢复时用状态查询补偿。

## 27.6 input_required：A2A 中的追问机制

Agent 任务经常需要补充信息。

例如用户说：

```text
帮我分析最近转化率下降的原因。
```

数据 Agent 可能需要知道：

1. 最近是指几天？
2. 转化率定义是什么？
3. 是全部用户还是某个渠道？
4. 是否排除异常流量？

这时下游 Agent 不应该胡乱假设，而应该返回 input_required。

一个结构化追问可以是：

```json
{
  "task_id": "task_123",
  "status": "input_required",
  "questions": [
    {
      "id": "q1",
      "question": "What time range should be analyzed?",
      "type": "date_range",
      "required": true
    },
    {
      "id": "q2",
      "question": "Which conversion definition should be used?",
      "type": "single_choice",
      "options": ["visit_to_signup", "signup_to_purchase"],
      "required": true
    }
  ]
}
```

上游 Agent 可以选择：

1. 把问题转给用户。
2. 从已有上下文中自动回答。
3. 调用另一个 Agent 查询定义。
4. 取消任务。
5. 改写任务目标。

这就是 A2A 比普通工具调用复杂的地方：任务不是一次性函数调用，而是可以协商和补全的过程。

## 27.7 结果返回：不要只返回一段文本

任务完成后，下游 Agent 应该返回结构化结果。

一个完成事件可以长这样：

```json
{
  "task_id": "task_20260529_001",
  "status": "completed",
  "result": {
    "summary": "退款率升高主要来自华东区域的部分支付方式，和 5 月 18 日上线的退款聚合逻辑变更相关。",
    "confidence": "medium",
    "key_findings": [
      "华东区域退款率环比上升 3.2 个百分点。",
      "信用卡支付方式贡献了主要增量。",
      "代码变更改变了退款订单归因逻辑。"
    ],
    "limitations": [
      "未分析用户级明细，因为当前任务策略只允许聚合数据。"
    ]
  },
  "artifacts": [
    {
      "type": "report",
      "uri": "artifact://task_20260529_001/refund-analysis-report.md",
      "mime_type": "text/markdown"
    },
    {
      "type": "table",
      "uri": "artifact://task_20260529_001/refund-by-region.csv",
      "mime_type": "text/csv"
    }
  ],
  "evidence": [
    "kb://docs/refund-rate-definition",
    "trace://task_20260529_001/tool-call-7"
  ]
}
```

好的结果返回应该包含：

1. summary。
2. structured_result。
3. artifacts。
4. evidence。
5. confidence。
6. limitations。
7. follow_up_actions。
8. trace_id。

这样上游 Agent 才能继续汇总、验证、展示或委派后续任务。

## 27.8 Artifact：结果产物的引用

Artifact 是 Agent 任务产生的可引用产物。

常见 Artifact 包括：

1. 报告。
2. 图表。
3. 表格。
4. 代码补丁。
5. 测试结果。
6. 日志摘要。
7. 数据文件。
8. 审计证明。

Artifact 不应该只是一个裸链接。它应该带元数据：

```json
{
  "type": "patch",
  "uri": "artifact://task_456/fix-refund-calculation.patch",
  "mime_type": "text/x-diff",
  "created_by": "agent.code_fix.v1",
  "created_at": "2026-05-29T10:30:00Z",
  "size_bytes": 4821,
  "content_hash": "sha256:abc123",
  "classification": "internal",
  "expires_at": "2026-06-28T00:00:00Z"
}
```

这些元数据用于：

1. 校验产物完整性。
2. 判断是否可分享。
3. 控制保存时长。
4. 做审计追踪。
5. 支持后续 Agent 引用。

## 27.9 失败语义：failed 也要结构化

多 Agent 系统最怕模糊失败。

例如下游 Agent 返回：

```text
不好意思，我做不了。
```

这对上游没有帮助。到底是权限不足、输入缺失、工具失败、模型拒绝、超时，还是能力不匹配？

结构化失败应该包含：

```json
{
  "task_id": "task_123",
  "status": "failed",
  "error": {
    "code": "PERMISSION_DENIED",
    "message": "The agent is not allowed to access db://analytics/refunds.",
    "retryable": false,
    "category": "authorization",
    "details": {
      "required_scope": "refunds.read",
      "current_scopes": ["orders.read"]
    }
  },
  "partial_artifacts": [],
  "trace_id": "trace_xyz"
}
```

常见错误类别包括：

1. INVALID_INPUT。
2. MISSING_CONTEXT。
3. PERMISSION_DENIED。
4. CAPABILITY_MISMATCH。
5. TOOL_FAILURE。
6. TIMEOUT。
7. RATE_LIMITED。
8. POLICY_VIOLATION。
9. INTERNAL_ERROR。
10. CANCELLED_BY_USER。

错误语义清楚，上游 Agent 才能决定是重试、换 Agent、追问用户、降级处理还是终止任务。

## 27.10 重试、幂等和取消

### 27.10.1 重试

不是所有失败都能重试。

适合重试的错误：

1. 临时网络错误。
2. 短暂限流。
3. 下游服务短暂不可用。
4. 可恢复的工具超时。

不适合重试的错误：

1. 权限不足。
2. 输入缺失。
3. 能力不匹配。
4. 策略拒绝。
5. 用户取消。

因此 error.retryable 字段很重要。

### 27.10.2 幂等

任务委派需要幂等键。否则上游 Agent 因为网络问题重发请求，下游可能重复执行。

例如：

```json
{
  "task_id": "task_123",
  "idempotency_key": "orchestrator_abc_refund_analysis_20260529"
}
```

对于读任务，重复执行可能只是浪费资源。对于写任务，重复执行可能导致严重后果，例如重复发邮件、重复创建工单、重复修改代码。

### 27.10.3 取消

取消不是简单地断开连接。下游 Agent 需要知道任务被取消，并尽量停止正在进行的工作。

取消请求可以包含：

```json
{
  "task_id": "task_123",
  "reason": "user_cancelled",
  "requested_by": "agent.orchestrator.v1",
  "timestamp": "2026-05-29T10:40:00Z"
}
```

取消后，下游 Agent 应该返回 cancelled 状态，并说明是否产生了部分产物。

## 27.11 并行委派和结果汇总

多 Agent 系统经常并行委派。

例如总控 Agent 要分析退款率升高原因，可以同时委派：

1. 数据 Agent 分析指标变化。
2. 客服 Agent 聚类用户反馈。
3. 代码 Agent 检查最近相关代码变更。
4. 知识库 Agent 查询业务定义。

并行委派带来两个问题：

1. 状态如何汇总？
2. 结果冲突如何处理？

状态汇总可以是：

```text
root task
  data_analysis: completed
  feedback_clustering: running
  code_change_review: failed
  knowledge_lookup: completed
```

结果冲突则需要仲裁机制。例如数据 Agent 认为异常来自支付方式，代码 Agent 认为来自聚合逻辑变更，客服 Agent 认为来自促销政策。总控 Agent 不能简单平均，而应该看证据链、置信度、数据来源和时间线。

## 27.12 安全边界：委派不等于转授权

一个常见错误是：上游 Agent 有权限，所以它委派给下游 Agent 时，下游自动拥有同等权限。

这是危险的。

正确做法是：

1. 上游 Agent 只能传递任务所需的最小上下文。
2. 下游 Agent 的权限必须单独校验。
3. 用户授权是否允许转委派需要明确。
4. 下游 Agent 是否可以继续委派也要受控。
5. 所有转授权都要进入审计。

例如，总控 Agent 可以访问机密文档，不代表外部写作 Agent 也能看到整份文档。总控 Agent 可以只传递脱敏摘要，或者选择内部写作 Agent。

## 27.13 一个完整流程示例

用户说：

```text
请帮我生成一份退款率升高的根因分析报告。
```

系统流程：

1. 总控 Agent 创建 root task。
2. 根据 Agent Card 发现数据分析 Agent、客服反馈 Agent、报告写作 Agent。
3. 总控 Agent 向数据分析 Agent 委派 task_data。
4. 数据分析 Agent 返回 accepted。
5. 数据分析 Agent 进入 running，并发送进度事件。
6. 数据分析 Agent 发现缺少“退款率定义”，返回 input_required。
7. 总控 Agent 调用知识库 Agent 查询定义，并把结果补充给数据分析 Agent。
8. 数据分析 Agent 完成，返回 summary、table artifact 和 evidence。
9. 总控 Agent 向客服反馈 Agent 委派反馈聚类任务。
10. 客服反馈 Agent 完成，返回反馈主题和引用。
11. 总控 Agent 把两个结果委派给报告写作 Agent。
12. 报告写作 Agent 生成 report artifact。
13. 总控 Agent 检查证据链和敏感信息。
14. 最终向用户返回报告。

这个流程里，A2A 负责的是任务委派、状态同步、追问补全和结果产物传递；MCP 可能负责每个 Agent 内部访问数据库、知识库或文件系统。

## 27.14 常见误区

### 27.14.1 把任务状态写在自然语言里

例如“我差不多快好了”。这对编排系统没有稳定含义。状态必须结构化。

### 27.14.2 把 input_required 当失败

追问是 Agent 协作的重要能力。缺信息时应该进入 input_required，而不是胡乱猜测或直接失败。

### 27.14.3 结果只返回纯文本

纯文本不利于上游继续编排。应返回结构化结果、Artifact、证据链、置信度和限制说明。

### 27.14.4 没有幂等设计

重发请求可能导致重复执行。任务委派必须有 task_id 和 idempotency_key。

### 27.14.5 委派时全量传递上下文

这会扩大泄露面。默认应该传最小必要上下文，并标记数据分类和共享限制。

## 27.15 面试高频题

### 题 1：A2A Task 通常包含哪些字段？

参考回答：

通常包含 task_id、parent_task_id、requester、assignee、goal、structured input、context resources、output requirements、constraints、deadline、permission scope、callback 或 event channel、idempotency key 和 trace_id。生产系统还会包含租户、数据分类、成本预算和重试策略。

### 题 2：如何设计 A2A 任务状态机？

参考回答：

可以设计 submitted、accepted、rejected、running、input_required、completed、failed、cancelled、expired 等状态。每个状态要有明确语义。input_required 表示需要补充信息，不是失败；failed 要有错误类别和 retryable 字段；cancelled 要说明取消来源；completed 要带结果或 Artifact。

### 题 3：A2A 中 input_required 有什么价值？

参考回答：

它让下游 Agent 在缺少关键信息时可以结构化追问，而不是胡乱猜测或直接失败。上游 Agent 可以把问题转给用户、从上下文自动补齐、调用其他 Agent 查询，或取消任务。它是多轮任务协商的关键机制。

### 题 4：任务失败为什么要结构化？

参考回答：

因为上游 Agent 需要根据失败原因决定下一步。权限不足、输入缺失、工具失败、超时、限流、策略拒绝的处理方式不同。结构化错误应包含 code、message、category、retryable、details 和 trace_id。

### 题 5：委派任务时如何控制安全边界？

参考回答：

委派不等于转授权。上游 Agent 只能传递最小必要上下文，下游 Agent 权限必须单独校验。用户授权是否允许转委派要明确，下游是否能继续委派也要受控。所有上下文传递、权限检查和结果返回都要进入审计。

## 27.16 小练习

1. 设计一个 A2A Task JSON，用于“代码 Agent 修复单元测试失败”。
2. 为这个任务设计状态流：submitted、accepted、running、input_required、completed 或 failed。
3. 写一个 failed 事件，错误类型为 PERMISSION_DENIED，并说明为什么不可重试。
4. 设计一个 report Artifact 的元数据字段。
5. 思考：如果一个下游 Agent 长时间 running 但没有进度事件，上游 Agent 应该如何处理？

## 27.17 本章小结

本章我们讲了 A2A 的运行核心：任务委派、状态同步和结果返回。

Agent 间协作不能只靠一句自然语言请求。一个可靠的 A2A Task 应该包含任务 ID、发起方、接收方、目标、输入、上下文、输出要求、约束、回调、幂等键和权限边界。任务执行过程中，需要明确 submitted、accepted、running、input_required、completed、failed、cancelled、expired 等状态。任务完成后，不应该只返回纯文本，而应该返回结构化结果、Artifact、证据链、置信度、限制说明和 trace。

你可以把本章核心结论记成一句话：

> A2A 的本质是把 Agent 间协作变成可追踪的任务生命周期，而不是把多个聊天机器人简单串起来。

下一章我们会继续讲 Multi-Agent 协作中的消息格式和上下文边界，重点讨论 Agent 之间传什么、不能传什么、如何防止上下文污染和信息泄露。
