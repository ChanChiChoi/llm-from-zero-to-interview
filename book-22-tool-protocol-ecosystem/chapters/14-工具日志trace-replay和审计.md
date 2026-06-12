# 第十四章：工具日志、Trace、Replay 和审计

## 14.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求，重点补齐工具日志、trace、replay 和审计的指标公式、变量解释和最小可运行 demo。资料边界对齐 OpenAI Agents SDK tracing 对 trace、span、processor 和 workflow 可观测性的抽象，OpenTelemetry 对 trace / span / context propagation 的通用语义，W3C Trace Context 对跨服务 traceparent / tracestate 传播的标准化要求，以及 MCP logging 对工具协议内日志消息、级别和进度通知的边界。

本章只抽象生产工具调用系统的稳定可观测层，不绑定某一家 provider 的 trace 字段、SDK 回调、日志平台、存储后端或审计产品。重点不是“多打日志”，而是证明每一次模型意图、工具选择、参数变化、权限判定、执行结果、脱敏动作、审计事件和 replay 证据都有结构化记录，并且能在隐私和权限边界内用于调试、评估、追责和回归。

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

## 14.21 Trace / Replay 审计指标与最小 demo

Trace、Replay 和审计要从“有日志”升级成“可验证的链路证据”。可以把一条工具执行 trace 样本抽象为：

```math
q_i=(m_i,r_i,a_i,p_i,e_i,o_i,v_i,\ell_i,b_i,z_i)
```

其中，`m_i` 是模型输入输出和 tool call，`r_i` 是 Router 决策，`a_i` 是参数解析、规范化和修复链路，`p_i` 是权限决策，`e_i` 是 Executor 执行记录，`o_i` 是 tool result 和最终回答，`v_i` 是模型、prompt、schema、registry、policy 和 adapter 版本，`\ell_i` 是结构化日志和指标导出，`b_i` 是 replay / audit / privacy 边界，`z_i` 是期望审计结论。

### 14.21.1 Trace schema 完整率

```math
C_{\mathrm{trace}}=
\frac{\sum_i I_i^{\mathrm{trace\_complete}}}{N}
```

其中，`I_i^{trace_complete}` 表示 trace 至少包含 run、turn、tool call、execution、tool name、status、latency 等核心字段。这个指标回答的是：一条线上问题能不能被查到基本链路。

### 14.21.2 ID 与 span tree 完整率

```math
C_{\mathrm{id}}=
\frac{\sum_i I_i^{\mathrm{id\_ok}} I_i^{\mathrm{tree\_ok}}}{N}
```

其中，`I_i^{id_ok}` 表示 run id、tool call id、execution id、job id 能跨服务对齐，`I_i^{tree_ok}` 表示 span parent / child 关系合法。缺少这层，trace UI 看起来有很多事件，但无法说明因果关系。

### 14.21.3 版本捕获覆盖率

```math
C_{\mathrm{version}}=
\frac{\sum_i I_i^{\mathrm{version\_complete}}}{N}
```

版本字段至少应覆盖 model、prompt、tool schema、registry、router policy 和 provider adapter。没有版本捕获，replay 只能“看历史”，不能比较新旧模型、新工具 schema 或新权限策略是否修复了问题。

### 14.21.4 参数 lineage 覆盖率

```math
C_{\mathrm{arg}}=
\frac{\sum_i I_i^{\mathrm{has\_args}}I_i^{\mathrm{arg\_lineage}}}
{\sum_i I_i^{\mathrm{has\_args}}}
```

其中，`I_i^{arg_lineage}` 表示 raw arguments、parsed arguments、normalized arguments、final arguments、validation result 和 evidence map 都被记录。它回答的是：执行参数到底来自模型、用户证据、runtime 规范化，还是修复逻辑。

### 14.21.5 权限与结果 trace 完整率

```math
C_{\mathrm{perm\_trace}}=
\frac{\sum_i I_i^{\mathrm{perm}}I_i^{\mathrm{perm\_complete}}}
{\sum_i I_i^{\mathrm{perm}}}
```

```math
C_{\mathrm{result\_trace}}=
\frac{\sum_i I_i^{\mathrm{result}}I_i^{\mathrm{result\_complete}}}
{\sum_i I_i^{\mathrm{result}}}
```

权限 trace 至少应记录 user、tenant、tool、action、resource、decision、reason 和 policy version；结果 trace 至少应记录 status、schema validation、raw size、projected size、sensitivity、redaction 和是否进入模型上下文。

### 14.21.6 隐私脱敏和审计事件完整率

```math
C_{\mathrm{privacy}}=
\frac{\sum_i I_i^{\mathrm{pii}}I_i^{\mathrm{masked}}}
{\sum_i I_i^{\mathrm{pii}}}
```

```math
C_{\mathrm{audit}}=
\frac{\sum_i I_i^{\mathrm{audit}}I_i^{\mathrm{audit\_complete}}}
{\sum_i I_i^{\mathrm{audit}}}
```

隐私脱敏指标关注 trace 是否把手机号、邮箱、密钥、付款信息、客户备注、私有文档等敏感信息最小化、脱敏、加密或短 TTL。审计事件完整率关注 audit id、actor、tenant、tool、action、resource、decision、timestamp、trace id 和 outcome 是否完整。

### 14.21.7 Replay readiness 与副作用安全率

```math
C_{\mathrm{replay}}=
\frac{\sum_i I_i^{\mathrm{replay}}I_i^{\mathrm{replay\_ready}}}
{\sum_i I_i^{\mathrm{replay}}}
```

```math
C_{\mathrm{side}}=
\frac{\sum_i I_i^{\mathrm{replay}}I_i^{\mathrm{side}}I_i^{\mathrm{live\_blocked}}}
{\sum_i I_i^{\mathrm{replay}}I_i^{\mathrm{side}}}
```

Replay readiness 依赖输入、输出、版本、工具 schema、工具结果、权限策略和 mock / sandbox 条件；副作用安全率要求发邮件、支付、删除、发布和权限修改等动作默认禁止 live replay，只能 dry-run、mock 或审批后执行。

### 14.21.8 指标、告警和 eval 链接覆盖率

```math
C_{\mathrm{metric}}=
\frac{\sum_i I_i^{\mathrm{metric\_complete}}}{N}
```

```math
C_{\mathrm{alert}}=
\frac{\sum_i I_i^{\mathrm{alert}}I_i^{\mathrm{owner}}}
{\sum_i I_i^{\mathrm{alert}}}
```

```math
C_{\mathrm{eval}}=
\frac{\sum_i I_i^{\mathrm{eval\_linked}}}{N}
```

指标导出要覆盖 tool success、latency、error rate、permission denied 和 trace missing 等核心监控；告警要有 owner；eval 链接表示 trace 能进入失败样本库、离线回归集或 golden trace 对比。

综合门禁可以写成：

```math
G_{\mathrm{trace}}=
I[
C_{\mathrm{trace}}\ge \tau_{\mathrm{trace}}
\land C_{\mathrm{id}}\ge \tau_{\mathrm{id}}
\land C_{\mathrm{version}}\ge \tau_{\mathrm{version}}
\land C_{\mathrm{arg}}\ge \tau_{\mathrm{arg}}
\land C_{\mathrm{perm\_trace}}\ge \tau_{\mathrm{perm}}
\land C_{\mathrm{result\_trace}}\ge \tau_{\mathrm{result}}
\land C_{\mathrm{privacy}}\ge \tau_{\mathrm{privacy}}
\land C_{\mathrm{audit}}\ge \tau_{\mathrm{audit}}
\land C_{\mathrm{replay}}\ge \tau_{\mathrm{replay}}
\land C_{\mathrm{side}}\ge \tau_{\mathrm{side}}
\land C_{\mathrm{metric}}\ge \tau_{\mathrm{metric}}
\land C_{\mathrm{alert}}\ge \tau_{\mathrm{alert}}
\land C_{\mathrm{eval}}\ge \tau_{\mathrm{eval}}
]
```

下面的 demo 用 0 依赖 Python 模拟一批工具 trace。输入是 list-of-dict，每个 dict 代表一条工具调用链路；输出是 trace / replay / audit 指标、失败样本、失败门禁和最终是否通过上线门禁。

```python
REQUIRED_TRACE = {"run_id", "turn_id", "tool_call_id", "execution_id", "tool_name", "status", "latency_ms"}
REQUIRED_VERSION = {"model", "prompt", "tool_schema", "registry", "router_policy", "adapter"}
REQUIRED_ARGS = {"raw_args", "parsed_args", "normalized_args", "final_args", "validation", "evidence"}
REQUIRED_PERMISSION = {"user_id", "tenant_id", "tool", "action", "resource", "decision", "reason", "policy_version"}
REQUIRED_RESULT = {"status", "schema_valid", "raw_size", "projected_size", "sensitivity", "redacted", "sent_to_model"}
REQUIRED_AUDIT = {"audit_id", "actor", "tenant_id", "tool", "action", "resource", "decision", "timestamp", "trace_id", "outcome"}
REQUIRED_METRICS = {"tool_success", "latency", "error_rate", "permission_denied", "trace_missing"}

cases = [
    {
        "id": "full_trace_ok",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": True,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT,
        "replay_needed": True,
        "replay_ready": True,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "missing_span_parent_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": False,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": False,
        "permission_fields": set(),
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": True,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": {"tool_success", "latency", "error_rate"},
        "alert_required": True,
        "alert_owner": False,
        "eval_linked": False,
    },
    {
        "id": "trace_id_mismatch_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": False,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": False,
        "arg_fields": set(),
        "permission_required": False,
        "permission_fields": set(),
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": True,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": True,
        "eval_linked": False,
    },
    {
        "id": "no_version_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": {"model", "prompt"},
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": False,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT,
        "replay_needed": True,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "arguments_no_lineage_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": {"raw_args", "final_args"},
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "permission_missing_reason_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": True,
        "permission_fields": {"user_id", "tenant_id", "tool", "action", "resource", "decision"},
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT - {"decision"},
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": {"tool_success", "latency", "permission_denied", "trace_missing"},
        "alert_required": True,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "tool_result_no_projection_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": False,
        "arg_fields": set(),
        "permission_required": False,
        "permission_fields": set(),
        "result_required": True,
        "result_fields": {"status", "raw_size", "sensitivity", "sent_to_model"},
        "pii_present": True,
        "pii_masked": False,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": True,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": True,
        "eval_linked": False,
    },
    {
        "id": "raw_pii_unmasked_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": False,
        "permission_fields": set(),
        "result_required": False,
        "result_fields": set(),
        "pii_present": True,
        "pii_masked": False,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "audit_missing_actor_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": False,
        "arg_fields": set(),
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT - {"actor", "resource"},
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": True,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": False,
        "eval_linked": True,
    },
    {
        "id": "replay_ready_ok",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": True,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT,
        "replay_needed": True,
        "replay_ready": True,
        "side_effect": True,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "live_replay_side_effect_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": True,
        "permission_fields": REQUIRED_PERMISSION,
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": False,
        "pii_masked": True,
        "audit_required": True,
        "audit_fields": REQUIRED_AUDIT,
        "replay_needed": True,
        "replay_ready": True,
        "side_effect": True,
        "live_replay_blocked": False,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": True,
        "eval_linked": False,
    },
    {
        "id": "metric_missing_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": False,
        "arg_fields": set(),
        "permission_required": False,
        "permission_fields": set(),
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": {"latency"},
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": True,
    },
    {
        "id": "alert_no_owner_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": False,
        "arg_fields": set(),
        "permission_required": False,
        "permission_fields": set(),
        "result_required": False,
        "result_fields": set(),
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": False,
        "replay_ready": False,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": True,
        "alert_owner": False,
        "eval_linked": True,
    },
    {
        "id": "eval_link_missing_bad",
        "trace_fields": REQUIRED_TRACE,
        "id_integrity": True,
        "span_tree_valid": True,
        "version_fields": REQUIRED_VERSION,
        "has_args": True,
        "arg_fields": REQUIRED_ARGS,
        "permission_required": False,
        "permission_fields": set(),
        "result_required": True,
        "result_fields": REQUIRED_RESULT,
        "pii_present": False,
        "pii_masked": True,
        "audit_required": False,
        "audit_fields": set(),
        "replay_needed": True,
        "replay_ready": True,
        "side_effect": False,
        "live_replay_blocked": True,
        "metric_exported": REQUIRED_METRICS,
        "alert_required": False,
        "alert_owner": True,
        "eval_linked": False,
    },
]


def rate(num, den):
    return round(num / den, 3) if den else 1.0


def complete(case, key, required):
    return required.issubset(case.get(key, set()))


arg_cases = [case for case in cases if case.get("has_args")]
permission_cases = [case for case in cases if case.get("permission_required")]
result_cases = [case for case in cases if case.get("result_required")]
pii_cases = [case for case in cases if case.get("pii_present")]
audit_cases = [case for case in cases if case.get("audit_required")]
replay_cases = [case for case in cases if case.get("replay_needed")]
side_effect_replay = [case for case in cases if case.get("replay_needed") and case.get("side_effect")]
alert_cases = [case for case in cases if case.get("alert_required")]

metrics = {
    "trace_schema_completeness": rate(sum(complete(c, "trace_fields", REQUIRED_TRACE) for c in cases), len(cases)),
    "id_tree_integrity": rate(sum(c.get("id_integrity") and c.get("span_tree_valid") for c in cases), len(cases)),
    "version_capture_coverage": rate(sum(complete(c, "version_fields", REQUIRED_VERSION) for c in cases), len(cases)),
    "argument_lineage_coverage": rate(sum(complete(c, "arg_fields", REQUIRED_ARGS) for c in arg_cases), len(arg_cases)),
    "permission_trace_completeness": rate(sum(complete(c, "permission_fields", REQUIRED_PERMISSION) for c in permission_cases), len(permission_cases)),
    "tool_result_trace_completeness": rate(sum(complete(c, "result_fields", REQUIRED_RESULT) for c in result_cases), len(result_cases)),
    "privacy_masking_coverage": rate(sum(c.get("pii_masked") for c in pii_cases), len(pii_cases)),
    "audit_event_completeness": rate(sum(complete(c, "audit_fields", REQUIRED_AUDIT) for c in audit_cases), len(audit_cases)),
    "replay_readiness_rate": rate(sum(c.get("replay_ready") for c in replay_cases), len(replay_cases)),
    "side_effect_replay_safety": rate(sum(c.get("live_replay_blocked") for c in side_effect_replay), len(side_effect_replay)),
    "metric_export_coverage": rate(sum(complete(c, "metric_exported", REQUIRED_METRICS) for c in cases), len(cases)),
    "alert_owner_coverage": rate(sum(c.get("alert_owner") for c in alert_cases), len(alert_cases)),
    "eval_linkage_coverage": rate(sum(c.get("eval_linked") for c in cases), len(cases)),
}

thresholds = {
    "trace_schema_completeness": 1.0,
    "id_tree_integrity": 1.0,
    "version_capture_coverage": 1.0,
    "argument_lineage_coverage": 1.0,
    "permission_trace_completeness": 1.0,
    "tool_result_trace_completeness": 1.0,
    "privacy_masking_coverage": 1.0,
    "audit_event_completeness": 1.0,
    "replay_readiness_rate": 0.95,
    "side_effect_replay_safety": 1.0,
    "metric_export_coverage": 1.0,
    "alert_owner_coverage": 1.0,
    "eval_linkage_coverage": 0.95,
}

failed_cases = []
for case in cases:
    failed = []
    if not complete(case, "trace_fields", REQUIRED_TRACE):
        failed.append("trace_fields")
    if not (case.get("id_integrity") and case.get("span_tree_valid")):
        failed.append("id_tree")
    if not complete(case, "version_fields", REQUIRED_VERSION):
        failed.append("version")
    if case.get("has_args") and not complete(case, "arg_fields", REQUIRED_ARGS):
        failed.append("argument_lineage")
    if case.get("permission_required") and not complete(case, "permission_fields", REQUIRED_PERMISSION):
        failed.append("permission_trace")
    if case.get("result_required") and not complete(case, "result_fields", REQUIRED_RESULT):
        failed.append("tool_result_trace")
    if case.get("pii_present") and not case.get("pii_masked"):
        failed.append("privacy_masking")
    if case.get("audit_required") and not complete(case, "audit_fields", REQUIRED_AUDIT):
        failed.append("audit_event")
    if case.get("replay_needed") and not case.get("replay_ready"):
        failed.append("replay_readiness")
    if case.get("replay_needed") and case.get("side_effect") and not case.get("live_replay_blocked"):
        failed.append("side_effect_replay")
    if not complete(case, "metric_exported", REQUIRED_METRICS):
        failed.append("metric_export")
    if case.get("alert_required") and not case.get("alert_owner"):
        failed.append("alert_owner")
    if not case.get("eval_linked"):
        failed.append("eval_linkage")
    if failed:
        failed_cases.append(case["id"])

failed_gates = [name for name, value in metrics.items() if value < thresholds[name]]

print("metrics=", metrics)
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("trace_replay_gate_pass=", not failed_gates)
```

输出示例：

```text
metrics= {'trace_schema_completeness': 1.0, 'id_tree_integrity': 0.857, 'version_capture_coverage': 0.929, 'argument_lineage_coverage': 0.889, 'permission_trace_completeness': 0.857, 'tool_result_trace_completeness': 0.857, 'privacy_masking_coverage': 0.5, 'audit_event_completeness': 0.667, 'replay_readiness_rate': 0.5, 'side_effect_replay_safety': 0.5, 'metric_export_coverage': 0.786, 'alert_owner_coverage': 0.625, 'eval_linkage_coverage': 0.643}
failed_cases= ['missing_span_parent_bad', 'trace_id_mismatch_bad', 'no_version_bad', 'arguments_no_lineage_bad', 'permission_missing_reason_bad', 'tool_result_no_projection_bad', 'raw_pii_unmasked_bad', 'audit_missing_actor_bad', 'live_replay_side_effect_bad', 'metric_missing_bad', 'alert_no_owner_bad', 'eval_link_missing_bad']
failed_gates= ['id_tree_integrity', 'version_capture_coverage', 'argument_lineage_coverage', 'permission_trace_completeness', 'tool_result_trace_completeness', 'privacy_masking_coverage', 'audit_event_completeness', 'replay_readiness_rate', 'side_effect_replay_safety', 'metric_export_coverage', 'alert_owner_coverage', 'eval_linkage_coverage']
trace_replay_gate_pass= False
```

这个结果故意不通过门禁，因为 toy 数据里存在 span tree 断链、trace id 不一致、版本缺失、参数 lineage 缺失、权限缺 reason、结果未投影、PII 未脱敏、审计事件缺 actor、有副作用工具 live replay、指标导出缺失、告警无 owner 和 eval 链接缺失。面试中要强调：Trace / Replay 不是排障时临时补日志，而是工具调用系统的事实层。

## 14.22 常见错误

### 14.22.1 只记录最终回答

问题：无法知道工具是否正确调用。

修复：记录完整 tool trace。

### 14.22.2 不记录版本

问题：schema 和模型变化后无法复现。

修复：记录 model、prompt、tool schema、registry、policy、adapter 版本。

### 14.22.3 raw 日志无脱敏

问题：日志系统变成敏感数据泄露源。

修复：脱敏、加密、权限控制和 TTL。

### 14.22.4 有副作用工具可 live replay

问题：调试时重复发送、扣款、删除。

修复：默认 dry-run，副作用工具禁止生产 replay。

### 14.22.5 审计日志可被篡改

问题：安全事故无法追责。

修复：不可篡改存储、权限控制和完整性校验。

### 14.22.6 Trace ID 不统一

问题：跨模型、runtime、工具服务无法串联。

修复：统一 run_id、tool_call_id、execution_id 并透传。

### 14.22.7 不记录被过滤工具

问题：无法解释模型为什么没有某个工具。

修复：Router trace 记录过滤原因。

### 14.22.8 指标没有按工具拆分

问题：总体成功率掩盖单个工具故障。

修复：按 tool_name、version、tenant、model、scenario 聚合。

## 14.23 面试题：如何设计工具调用 Trace 和审计

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

## 14.24 小练习

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

## 14.25 本章小结

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
