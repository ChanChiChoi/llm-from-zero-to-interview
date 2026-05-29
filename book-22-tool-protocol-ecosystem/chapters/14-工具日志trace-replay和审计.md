# 第十四章：工具日志、Trace、Replay 和审计

## 14.1 本章定位

前面讲了 Tool Registry、Router、Executor、权限和安全。本章讲可观测性和审计：工具调用发生后，我们如何知道系统到底做了什么。

没有日志和 trace 的工具系统，本质上是黑盒。出现问题时只能猜：

1. 模型到底有没有调用工具？
2. Router 为什么没给模型某个工具？
3. 参数是模型生成的还是 runtime 修复的？
4. 权限检查为什么放行或拒绝？
5. 工具到底执行了几次？
6. 有副作用动作是否重复执行？
7. 最终回答是否基于真实工具结果？
8. 用户投诉时如何复现？
9. 安全事故时如何追责？

本章的核心观点是：

```text
Trace 是工具调用系统的事实记录；没有 trace，就没有可靠评估、调试、审计、回放和安全追责。
```

## 14.2 日志、Trace、Replay、审计的区别

这几个概念容易混。

日志是事件记录。

```text
某个时间发生了什么。
```

Trace 是一次请求或一次 agent run 的完整链路。

```text
从用户输入到模型输出、工具调用、工具结果、最终回答的因果链。
```

Replay 是基于历史 trace 复现或模拟当时流程。

```text
用当时的输入、schema、模型输出和工具结果重新跑一遍。
```

审计是面向合规和追责的不可抵赖记录。

```text
谁在什么时间、以什么权限、对什么资源执行了什么动作，结果是什么。
```

它们互相关联，但目标不同。

## 14.3 一次工具调用的 Trace 应该包含什么

完整 trace 至少包含这些层次。

第一，用户和会话信息：

1. conversation_id。
2. run_id。
3. user_id。
4. tenant_id。
5. session_id。
6. product surface。
7. timestamp。

第二，模型输入输出：

1. system / developer 指令版本。
2. messages 摘要或脱敏内容。
3. provider。
4. model name。
5. model parameters。
6. assistant tool calls。
7. finish_reason。

第三，工具路由：

1. registry version。
2. 初始候选工具数量。
3. 过滤后的工具。
4. 被过滤工具和原因。
5. tool choice mode。
6. router version。
7. policy version。

第四，工具执行：

1. tool_call_id。
2. tool_name。
3. tool_version。
4. raw arguments。
5. normalized arguments。
6. validation result。
7. permission decision。
8. execution attempts。
9. latency。
10. status。
11. error code。
12. tool result。

第五，最终回答：

1. final answer。
2. citations。
3. safety filters。
4. user-visible error。
5. user feedback。

## 14.4 Trace ID 设计

Trace ID 是串联整条链路的关键。

常见 ID：

1. `conversation_id`：对话级。
2. `run_id`：一次 agent run。
3. `turn_id`：一轮用户交互。
4. `tool_call_id`：模型生成的工具调用。
5. `execution_id`：Executor 内部一次执行。
6. `job_id`：异步任务。
7. `audit_id`：审计事件。

关系大致是：

```text
conversation_id
  └── run_id
        └── turn_id
              └── tool_call_id
                    └── execution_id
                          └── job_id
```

这些 ID 要进入日志、指标、错误回填和用户问题排查系统。

没有稳定 ID，跨服务排查会非常困难。

## 14.5 Structured Logging

工具系统日志应使用结构化日志，而不是只写自然语言。

差的日志：

```text
调用订单工具失败了。
```

好的日志：

```json
{
  "event": "tool_execution_failed",
  "run_id": "run_123",
  "tool_call_id": "call_456",
  "tool_name": "get_order_status",
  "tool_version": "1.2.0",
  "error_code": "TOOL_TIMEOUT",
  "retryable": true,
  "latency_ms": 3000,
  "attempt": 2,
  "tenant_id": "T1"
}
```

结构化日志可以用于：

1. 查询。
2. 聚合指标。
3. 告警。
4. eval。
5. 审计。
6. 根因分析。

## 14.6 需要记录原始内容吗

Trace 是否记录完整 messages、arguments、tool results，是一个权衡。

记录完整内容的好处：

1. 可复现。
2. 易调试。
3. 可做 eval。
4. 可排查用户投诉。

风险：

1. 存储敏感数据。
2. 增加隐私合规压力。
3. 日志泄露风险。
4. 存储成本高。

常见策略：

1. 默认记录结构化元数据。
2. 对敏感字段脱敏或哈希。
3. 对高风险租户关闭 raw content 记录。
4. 对调试采样记录 raw，但加密和短 TTL。
5. 对审计保留最小必要信息。

原则：

```text
Trace 要足够复现关键决策，但不能变成新的敏感数据泄露源。
```

## 14.7 参数 Trace

参数是工具调用事故高发点，必须记录变化过程。

参数 trace 应包含：

1. raw_arguments。
2. parsed_arguments。
3. normalized_arguments。
4. final_execution_arguments。
5. validation errors。
6. repair attempts。
7. evidence map。
8. sensitive fields redaction。

例如：

```json
{
  "raw_arguments": "{\"status\":\"PAID\"}",
  "normalized_arguments": {"status":"paid"},
  "normalization": [
    {"field":"status","before":"PAID","after":"paid","reason":"enum_case_normalization"}
  ]
}
```

这样当工具执行错了，可以判断是模型填错、runtime 修错，还是下游执行错。

## 14.8 权限 Trace

权限 trace 应记录每次权限决策。

字段包括：

1. user_id。
2. tenant_id。
3. tool_name。
4. action。
5. resource_id。
6. permission decision。
7. reason code。
8. policy version。
9. field projection。
10. timestamp。

示例：

```json
{
  "event": "permission_decision",
  "tool_name": "get_customer_profile",
  "resource_id": "C123",
  "decision": "deny",
  "reason": "resource_not_assigned_to_user",
  "policy_version": "crm_acl_2026_05"
}
```

注意：给用户和模型的错误可以脱敏，但内部审计要记录足够原因。

## 14.9 Tool Result Trace

工具结果 trace 需要记录：

1. status。
2. output_schema validation。
3. result size。
4. source metadata。
5. sensitivity label。
6. redaction actions。
7. compression actions。
8. final content sent to model。

例如：

```json
{
  "raw_result_size_bytes": 120000,
  "projected_result_size_bytes": 3500,
  "redactions": ["phone", "payment_token"],
  "compression": "top_k_snippets",
  "sent_to_model": true
}
```

这能帮助排查：模型为什么没看到某个字段，或者为什么最终回答没有引用某条数据。

## 14.10 Replay 的用途

Replay 有很多用途：

1. 复现线上 bug。
2. 比较新旧模型行为。
3. 比较新旧 tool schema。
4. 回归测试。
5. 安全事故调查。
6. eval 数据生成。
7. 调试 router 和 executor。
8. 验证修复是否有效。

例如用户投诉：“AI 给我发错邮件了。”

Replay 可以回答：

1. 用户原话是什么。
2. 模型生成了哪个 tool call。
3. 收件人参数来自哪里。
4. 是否展示确认。
5. 用户是否确认。
6. Executor 执行了几次。
7. 下游邮件服务返回了什么。

没有 replay，只能猜。

## 14.11 Replay 的类型

Replay 可以分三种。

第一种：trace-only replay。

只重放历史记录，不重新调用模型或工具。用于查看当时发生了什么。

第二种：model replay。

使用相同输入、schema 和模型配置重新调用模型，比较新旧输出。

第三种：tool replay。

重新执行工具或调用 mock 工具，验证 runtime 行为。

对有副作用工具，不能默认 live replay。应使用 dry-run 或 sandbox。

## 14.12 Replay 的必要条件

要 replay，必须保存：

1. 当时的模型名和版本。
2. system/developer prompt 版本。
3. messages 或脱敏可复现摘要。
4. tools schema。
5. tool registry version。
6. router policy version。
7. provider adapter version。
8. model output。
9. tool results。
10. executor decisions。

如果 schema 没保存，只知道工具当前版本，就无法复现过去行为。

这也是为什么 Registry 和 trace 必须记录版本。

## 14.13 Replay 与安全

Replay 本身也有安全风险。

风险：

1. 重新执行副作用工具。
2. 重新访问敏感数据。
3. 扩散用户隐私。
4. 让调试人员看到无权数据。
5. 把历史攻击样例再次执行。

防御：

1. 默认 dry-run。
2. 有副作用工具禁止 live replay。
3. 使用 sandbox / mock。
4. replay 权限控制。
5. replay 日志审计。
6. raw content 脱敏。
7. 对高风险 replay 需要审批。

Replay 工具不能成为绕过权限的后门。

## 14.14 审计日志和普通日志的区别

审计日志比普通日志要求更高。

普通日志用于调试和监控，可以采样、脱敏、短期保存。

审计日志用于合规和追责，通常要求：

1. 完整性。
2. 不可篡改。
3. 时间准确。
4. 权限受控。
5. 保留周期明确。
6. 查询可追溯。
7. 关键字段不可缺失。

审计日志关注：

1. 谁。
2. 什么时候。
3. 以什么权限。
4. 对什么资源。
5. 做了什么动作。
6. 结果如何。
7. 是否经过确认。

例如发邮件、导出数据、修改权限、删除文件等工具必须有审计。

## 14.15 审计事件设计

审计事件可以这样设计：

```json
{
  "audit_id": "audit_123",
  "event_type": "tool_action_executed",
  "timestamp": "2026-05-29T10:00:00Z",
  "actor": {
    "user_id": "U123",
    "tenant_id": "T1",
    "agent_id": "agent_support"
  },
  "tool": {
    "name": "send_email_draft",
    "version": "2.1.0"
  },
  "action": "send_email",
  "resource": {
    "type": "email_draft",
    "id": "draft_456"
  },
  "decision": "allowed",
  "confirmation": {
    "required": true,
    "confirmed": true,
    "confirmation_id": "confirm_789"
  },
  "trace_id": "run_abc"
}
```

审计内容不一定保存邮件正文全文，但要能追踪动作和责任链。

## 14.16 日志脱敏

日志脱敏是必需的。

常见敏感字段：

1. token。
2. password。
3. secret。
4. phone。
5. email。
6. address。
7. id card。
8. payment card。
9. customer note。
10. private document content。

脱敏方式：

1. 删除字段。
2. 掩码。
3. 哈希。
4. 加密存储。
5. 权限控制查看。
6. 短 TTL。

注意：哈希也可能被字典攻击，特别是手机号、邮箱这类低熵数据。不要以为哈希一定安全。

## 14.17 Metrics 指标

Trace 可以生成指标。

工具层指标：

1. tool_call_count。
2. tool_success_rate。
3. tool_error_rate。
4. timeout_rate。
5. retry_rate。
6. fallback_rate。
7. permission_denied_rate。
8. validation_failure_rate。
9. p50 / p95 / p99 latency。
10. output_size。

模型和路由指标：

1. tool_call_precision。
2. tool_call_recall。
3. wrong_tool_rate。
4. argument_error_rate。
5. router_candidate_recall。
6. high_risk_tool_exposure_rate。

安全指标：

1. prompt_injection_detected。
2. sensitive_data_redacted。
3. unsafe_action_blocked。
4. duplicate_side_effect_prevented。
5. handoff_rate。

没有指标，就无法持续改进。

## 14.18 告警设计

告警不能只看系统 500。

工具系统应告警：

1. 某工具错误率突增。
2. p95 延迟突增。
3. 权限拒绝率异常。
4. 高风险工具调用量异常。
5. DLP 拦截突增。
6. prompt injection 命中突增。
7. 重试率或熔断次数突增。
8. 人工接管率突增。
9. tool result 过大。
10. trace 缺失率上升。

告警要有 owner。Tool Registry 中的 owner 可以用于路由告警。

## 14.19 Trace 与 Eval 的关系

第八章讲过 eval。工具 eval 离不开 trace。

基于 trace 可以自动判断：

1. 该调用时是否调用。
2. 调用了哪个工具。
3. 参数是否符合期望。
4. 权限是否正确。
5. 是否重复执行。
6. 是否使用了工具结果。
7. 最终任务是否完成。

Eval pipeline 可以读取线上 trace，抽样生成失败案例，也可以读取离线 golden trace 做回归测试。

没有 trace，eval 只能看最终回答，无法定位工具调用链路中的问题。

## 14.20 Trace 与隐私合规

Trace 保存用户请求、工具参数和结果，可能涉及隐私合规。

需要考虑：

1. 数据最小化。
2. 保留周期。
3. 删除权。
4. 访问控制。
5. 加密存储。
6. 跨境传输。
7. 数据用途限制。
8. 审计查询权限。

企业系统通常要支持：

1. 按租户配置 trace 保留策略。
2. 对敏感客户关闭 raw content。
3. 对调试访问做审批。
4. 用户数据删除后清理或匿名化 trace。

可观测性不能以牺牲隐私为代价。

## 14.21 常见错误

### 14.21.1 只记录最终回答

问题：无法知道工具是否正确调用。

修复：记录完整 tool trace。

### 14.21.2 不记录版本

问题：schema 和模型变化后无法复现。

修复：记录 model、prompt、tool schema、registry、policy、adapter 版本。

### 14.21.3 raw 日志无脱敏

问题：日志系统变成敏感数据泄露源。

修复：脱敏、加密、权限控制和 TTL。

### 14.21.4 有副作用工具可 live replay

问题：调试时重复发送、扣款、删除。

修复：默认 dry-run，副作用工具禁止生产 replay。

### 14.21.5 审计日志可被篡改

问题：安全事故无法追责。

修复：不可篡改存储、权限控制和完整性校验。

### 14.21.6 Trace ID 不统一

问题：跨模型、runtime、工具服务无法串联。

修复：统一 run_id、tool_call_id、execution_id 并透传。

### 14.21.7 不记录被过滤工具

问题：无法解释模型为什么没有某个工具。

修复：Router trace 记录过滤原因。

### 14.21.8 指标没有按工具拆分

问题：总体成功率掩盖单个工具故障。

修复：按 tool_name、version、tenant、model、scenario 聚合。

## 14.22 面试题：如何设计工具调用 Trace 和审计

面试官可能问：

```text
Agent 调用工具出了问题，你怎么排查？系统应该记录什么？
```

可以这样回答。

第一，记录完整链路：

1. 用户请求。
2. 模型输入输出。
3. Router 候选工具和过滤原因。
4. tool choice。
5. tool call。
6. 参数校验和修复。
7. 权限决策。
8. Executor 执行过程。
9. tool result。
10. 最终回答。

第二，记录版本：

1. model version。
2. prompt version。
3. tool schema version。
4. registry version。
5. router policy version。
6. adapter version。

第三，支持 replay：

1. trace-only replay 看当时发生什么。
2. model replay 比较新旧模型。
3. sandbox replay 测工具。
4. 有副作用工具默认 dry-run。

第四，审计高风险动作：

1. actor。
2. tenant。
3. tool。
4. resource。
5. permission decision。
6. confirmation。
7. result。
8. trace_id。

第五，隐私和安全：

1. 日志脱敏。
2. raw content 采样和短 TTL。
3. 审计日志不可篡改。
4. replay 权限控制。

一句话总结：

```text
我会把工具调用看成一条可追踪的因果链，记录从 router 到 executor 到 final answer 的结构化 trace，并对高风险动作做不可篡改审计和安全 replay。
```

## 14.23 小练习

### 练习 1：排查工具选错

模型调用了错误工具。trace 中最应该看什么？

参考答案：看 Router 候选工具、过滤原因、传给模型的 tool schema、模型输出 tool call，以及工具 description 版本。

### 练习 2：排查参数错误

最终执行参数和模型原始参数不一样。trace 中应该记录什么？

参考答案：raw_arguments、parsed_arguments、normalized_arguments、repair attempts、final_execution_arguments 和 normalization reason。

### 练习 3：审计发邮件动作

发邮件工具的审计日志至少需要哪些字段？

参考答案：user_id、tenant_id、tool_name、tool_version、draft_id 或 email_id、收件人摘要、confirmation_id、decision、timestamp、trace_id 和执行结果。

### 练习 4：Replay 风险

为什么不能对转账工具做 live replay？

参考答案：可能重复转账。应使用 dry-run、sandbox 或 mock，并保留幂等和审计记录。

### 练习 5：日志脱敏

日志中记录完整手机号和 token 有什么问题？

参考答案：日志系统会变成敏感数据泄露源。应脱敏、加密、限制访问，并设置保留周期。

## 14.24 本章小结

本章讲了工具日志、trace、replay 和审计。

你需要掌握：

1. 日志记录事件，trace 记录因果链，replay 用于复现，审计用于合规和追责。
2. 完整 trace 应覆盖用户、模型、router、tool call、参数、权限、executor、tool result 和最终回答。
3. Trace ID 要统一贯穿 conversation、run、turn、tool_call、execution 和 job。
4. 结构化日志比自然语言日志更适合查询、指标和告警。
5. 参数 trace、权限 trace、tool result trace 是工具系统排障关键。
6. Replay 分 trace-only、model replay 和 tool replay，有副作用工具默认 dry-run。
7. 审计日志要完整、不可篡改、可追责，特别是高风险工具。
8. 日志和 trace 必须脱敏、加密、权限控制，并符合隐私保留策略。
9. Trace 是 eval、回归测试、安全追责和线上监控的基础。
10. 没有 trace 的工具调用系统，无法称为生产级系统。

如果只记一句话：

```text
生产级工具调用必须让每一次模型意图、工具选择、参数变化、权限决策和真实执行都有迹可循、可复现、可审计。
```

下一章会讲工具版本管理、灰度发布和回滚，重点解释工具 schema、执行器和权限策略如何安全演进。
