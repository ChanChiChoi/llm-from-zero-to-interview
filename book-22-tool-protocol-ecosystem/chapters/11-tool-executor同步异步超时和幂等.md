# 第十一章：Tool Executor：同步、异步、超时和幂等

## 11.1 本章定位

上一章讲 Tool Router，解决“当前请求应该给模型哪些工具”。本章讲 Tool Executor，解决“模型已经生成 tool call 后，runtime 如何真实执行工具”。

很多 demo 的执行逻辑很简单：

```python
if tool_call.name == "get_weather":
    result = get_weather(**tool_call.arguments)
```

生产系统不能这样简单。真实 Tool Executor 要处理：

1. 工具名解析。
2. 参数校验。
3. 权限检查。
4. 同步/异步执行。
5. 并发控制。
6. 超时和取消。
7. 重试和熔断。
8. 幂等和副作用。
9. 结果包装。
10. trace 和审计。
11. 沙箱和隔离。
12. 错误回填。

本章的核心观点是：

```text
Tool Executor 是模型意图进入真实世界的最后一道工程边界；它必须比模型更保守、更确定、更可审计。
```

## 11.2 Tool Executor 在整体架构中的位置

完整链路如下：

```text
User
  ↓
Model + Tools
  ↓
Assistant Tool Call
  ↓
Tool Router / Policy Check
  ↓
Tool Executor
  ↓
External System
  ↓
Tool Result
  ↓
Model Final Answer
```

Executor 不负责让模型选择工具，也不负责写最终回答。Executor 的职责是：

1. 接收结构化 tool call。
2. 判断是否允许执行。
3. 调用真实工具。
4. 把结果或错误结构化返回。
5. 记录执行全过程。

一句话：

```text
模型说“我想做什么”，Executor 决定“能不能做、怎么做、做完如何记录”。
```

## 11.3 Executor 的输入和输出

Executor 的输入通常是内部统一 ToolCall：

```json
{
  "tool_call_id": "call_123",
  "tool_name": "get_order_status",
  "arguments": {
    "order_id": "ORD_123"
  },
  "raw_arguments": "{\"order_id\":\"ORD_123\"}",
  "provider": "model_provider_a",
  "conversation_id": "conv_1",
  "run_id": "run_1"
}
```

Executor 的输出是内部统一 ToolResult：

```json
{
  "tool_call_id": "call_123",
  "tool_name": "get_order_status",
  "status": "success",
  "content": {
    "order_id": "ORD_123",
    "status": "shipped"
  },
  "metadata": {
    "latency_ms": 280,
    "attempts": 1
  }
}
```

错误输出：

```json
{
  "tool_call_id": "call_123",
  "tool_name": "get_order_status",
  "status": "error",
  "error": {
    "code": "PERMISSION_DENIED",
    "message": "current user cannot access this order",
    "retryable": false
  }
}
```

输出再由 provider adapter 转成模型需要的 tool result message。

## 11.4 Executor 状态机

Executor 可以看成状态机。

常见状态：

1. `RECEIVED`：收到 tool call。
2. `RESOLVING`：从 Registry 解析工具定义。
3. `VALIDATING`：校验参数。
4. `AUTHORIZING`：检查权限。
5. `CONFIRMING`：等待用户确认。
6. `QUEUED`：进入执行队列。
7. `RUNNING`：正在执行。
8. `RETRYING`：失败后重试。
9. `SUCCEEDED`：执行成功。
10. `FAILED`：执行失败。
11. `CANCELLED`：被取消。
12. `UNKNOWN`：有副作用动作状态不确定。

为什么要状态机？

因为工具执行不是单个函数调用。它跨越模型、runtime、外部系统、用户确认和审计系统。状态不清楚，就无法处理重试、取消、恢复和人工接管。

## 11.5 执行前校验顺序

Executor 执行前应按顺序检查：

```text
tool exists
  ↓
tool status active
  ↓
schema validation
  ↓
argument normalization
  ↓
business validation
  ↓
permission check
  ↓
risk and confirmation check
  ↓
rate limit / quota check
  ↓
idempotency check
  ↓
execute
```

不能跳过这些步骤。

例如模型生成了：

```json
{"tool_name":"delete_file","arguments":{"path":"/"}}
```

即使工具存在，参数是合法 JSON，也必须被业务校验和安全策略拦截。

## 11.6 同步执行

同步执行指 Executor 调用工具后等待结果，再把结果回填给模型。

适合：

1. 低延迟查询。
2. 只读工具。
3. 用户交互需要即时结果。
4. 结果较小。
5. 失败恢复简单。

例如：

1. 查询天气。
2. 查询订单状态。
3. 搜索知识库。
4. 读取当前页面上下文。

同步执行流程：

```python
result = executor.execute(tool_call, timeout_ms=3000)
messages.append(to_tool_result_message(result))
response = model.generate(messages)
```

同步执行的风险：

1. 工具慢会阻塞用户。
2. 多个工具串行会增加延迟。
3. 长任务容易超时。
4. 有副作用工具状态不确定时难处理。

因此同步工具必须有严格 timeout。

## 11.7 异步执行

异步执行指工具调用提交后，不一定立即返回最终结果。

适合：

1. 长时间任务。
2. 批处理。
3. 文件分析。
4. 代码执行。
5. 报表生成。
6. 外部审批。
7. 后台工作流。

异步执行通常返回 job id：

```json
{
  "status": "queued",
  "job_id": "job_123",
  "message": "report generation started"
}
```

后续可以：

1. 轮询 job 状态。
2. 等 webhook 回调。
3. 通知用户稍后查看。
4. 让模型继续处理其他步骤。

异步工具要设计：

1. job 状态查询接口。
2. 取消接口。
3. 进度信息。
4. 结果过期时间。
5. 权限校验。
6. 幂等提交。
7. 用户通知机制。

不要把长任务伪装成同步工具，否则会导致请求超时和用户体验差。

## 11.8 并发执行

Parallel Tool Calls 需要 Executor 并发执行多个 tool call。

并发执行要控制：

1. 最大并发数。
2. 每个工具并发数。
3. 每用户并发数。
4. 每租户并发数。
5. 全局队列长度。
6. 下游限流。

伪代码：

```python
async def execute_many(tool_calls):
    tasks = []
    for call in tool_calls:
        limiter = get_limiter(call.tool_name)
        tasks.append(execute_with_limiter(call, limiter))
    return await gather_with_timeout(tasks)
```

并发结果必须按 tool_call_id 对齐，不要按返回顺序隐式对齐。

并发执行还要处理部分失败：成功项正常回填，失败项结构化错误回填。

## 11.9 队列和调度

当工具执行耗时或并发高时，Executor 需要队列。

队列用于：

1. 削峰。
2. 限流。
3. 异步任务。
4. 重试调度。
5. 优先级控制。
6. 失败恢复。

队列任务应包含：

```json
{
  "job_id": "job_123",
  "tool_call_id": "call_123",
  "tool_name": "generate_report",
  "arguments": {...},
  "priority": "normal",
  "idempotency_key": "idem_abc",
  "created_at": "2026-05-29T10:00:00Z",
  "deadline": "2026-05-29T10:05:00Z"
}
```

队列不是万能的。对交互式请求，排队太久会让用户体验变差。要设置 deadline 和用户可见状态。

## 11.10 超时控制

Executor 必须设置超时。

超时分层：

1. 单次下游请求超时。
2. 单工具总超时。
3. 并发工具组总超时。
4. 整个 agent run 超时。
5. 异步 job deadline。

示例：

```json
{
  "request_timeout_ms": 1000,
  "tool_timeout_ms": 3000,
  "parallel_group_timeout_ms": 5000,
  "agent_run_timeout_ms": 20000
}
```

超时后要返回结构化错误：

```json
{
  "error": {
    "code": "TOOL_TIMEOUT",
    "message": "tool execution exceeded 3000ms",
    "retryable": true
  }
}
```

对有副作用工具，超时不一定代表未执行。必须区分提交前超时、提交后超时和状态未知。

## 11.11 取消语义

用户可能取消请求，runtime 也可能因为超时取消工具。

取消分几种：

1. 还没开始执行：直接取消。
2. 执行中但可中断：发送取消信号。
3. 已提交给下游：尝试调用下游取消接口。
4. 有副作用且状态未知：进入 UNKNOWN 或人工接管。

不同工具取消能力不同。

Registry 可以记录：

```json
{
  "runtime": {
    "cancellable": true,
    "cancel_handler": "report.cancel_job"
  }
}
```

注意：取消用户等待不等于取消下游执行。如果后台仍在跑，必须记录状态并避免重复提交。

## 11.12 幂等

幂等是 Executor 的核心能力。

幂等表示同一个请求执行多次，效果和执行一次一样。

只读工具天然接近幂等：查询天气、搜索文档、查询订单。

有副作用工具不一定幂等：发邮件、下单、转账、删除文件。

Executor 应为有副作用工具生成 idempotency key：

```text
idempotency_key = hash(user_id, conversation_id, tool_call_id, tool_name, normalized_arguments)
```

执行记录：

```json
{
  "idempotency_key": "idem_abc",
  "status": "succeeded",
  "result": {...},
  "created_at": "2026-05-29T10:00:00Z"
}
```

如果同一个 key 再次到来，Executor 返回已有结果，而不是再次执行副作用。

## 11.13 幂等键设计陷阱

幂等键不能随便设计。

错误做法：

```text
idempotency_key = random_uuid()
```

每次重试都生成新 key，就失去幂等意义。

也不能只用 tool_call_id。如果 provider 重试导致 tool_call_id 变化，可能无法识别重复。

更稳的做法：

1. 对同一次用户意图保持稳定 key。
2. 包含用户和会话上下文。
3. 包含规范化参数。
4. 对高风险工具使用业务幂等 key，例如订单号加操作类型。
5. 设置合理过期时间。

幂等键也不能跨用户复用，否则可能泄露或混淆结果。

## 11.14 有副作用工具执行流程

有副作用工具应该更严格。

典型流程：

```text
validate arguments
  ↓
check permission
  ↓
check risk policy
  ↓
prepare preview
  ↓
ask user confirmation
  ↓
generate confirmation token
  ↓
execute with idempotency key
  ↓
record audit log
  ↓
return result
```

例如发送邮件：

1. 先创建草稿。
2. 展示收件人、主题、正文。
3. 用户确认。
4. runtime 生成 confirmation token。
5. Executor 校验 token。
6. 执行发送。
7. 记录审计。

模型不能自己生成 confirmation token。Executor 必须验证 token 来源。

## 11.15 执行器类型

工具执行器可以有多种类型。

常见类型：

1. in-process function executor。
2. HTTP executor。
3. RPC executor。
4. database executor。
5. shell / code sandbox executor。
6. browser executor。
7. MCP client executor。
8. workflow executor。

每种执行器风险不同。

in-process function 简单但隔离差。HTTP/RPC 适合服务化工具。Database executor 需要严格查询模板和权限。Shell/code executor 风险最高，需要沙箱。MCP executor 需要处理远端工具能力和权限边界。

Registry 中应标记 executor_type，Executor 根据类型选择执行策略。

## 11.16 沙箱和隔离

对高风险执行器，需要沙箱。

例如：

1. 代码执行。
2. shell 命令。
3. 文件操作。
4. 浏览器自动化。
5. 数据库查询。

隔离手段：

1. 容器。
2. 只读文件系统。
3. 网络隔离。
4. CPU / memory 限制。
5. 文件路径白名单。
6. 命令白名单。
7. 超时。
8. 输出大小限制。
9. 权限最小化。

Executor 不能相信模型生成的命令、路径或 SQL。必须用沙箱和策略约束真实执行。

## 11.17 输出大小限制

工具可能返回大量输出。

Executor 应限制：

1. stdout 大小。
2. JSON 字段大小。
3. 数组长度。
4. 文件读取长度。
5. 日志行数。
6. 错误堆栈长度。

超出时可以：

1. 截断。
2. 摘要。
3. 分页。
4. 存储到对象存储并返回引用。
5. 要求模型或用户进一步缩小范围。

不要把几 MB 工具输出直接塞回模型上下文。

## 11.18 结果包装

Executor 执行后不应直接返回原始结果。

应做：

1. output_schema validation。
2. 字段投影。
3. 脱敏。
4. source metadata。
5. trust level。
6. error normalization。
7. trace metadata。

输出示例：

```json
{
  "tool_call_id": "call_123",
  "tool_name": "get_order_status",
  "status": "success",
  "content": {
    "order_id": "ORD_123",
    "status": "shipped"
  },
  "metadata": {
    "source": "order_service",
    "retrieved_at": "2026-05-29T10:00:00Z",
    "latency_ms": 280
  }
}
```

Executor 产出的结果应能直接进入第六章讲的 tool result 上下文包装流程。

## 11.19 错误规范化

不同工具可能返回不同错误。

Executor 要把它们规范化成统一错误码。

例如：

1. `VALIDATION_ERROR`。
2. `PERMISSION_DENIED`。
3. `NOT_FOUND`。
4. `RATE_LIMITED`。
5. `TOOL_TIMEOUT`。
6. `UPSTREAM_ERROR`。
7. `UNKNOWN_EXECUTION_STATE`。
8. `SAFETY_BLOCKED`。
9. `CONFIRMATION_REQUIRED`。
10. `CANCELLED`。

统一错误码有利于：

1. 模型理解。
2. runtime 决策。
3. 监控统计。
4. eval 分析。
5. 用户友好展示。

内部详细错误进入日志，给模型和用户的错误要脱敏。

## 11.20 Executor Trace

Executor 必须记录 trace。

字段包括：

1. run_id。
2. tool_call_id。
3. tool_name。
4. tool_version。
5. arguments_raw。
6. arguments_normalized。
7. validation result。
8. permission decision。
9. confirmation status。
10. idempotency_key。
11. attempts。
12. latency。
13. status。
14. error code。
15. output size。
16. executor_type。
17. sandbox id。

Trace 用于：

1. 调试。
2. 审计。
3. replay。
4. eval。
5. 告警。
6. 用户问题排查。

没有 Executor trace，工具执行就是黑盒。

## 11.21 Replay

Replay 是指用历史 trace 复现工具调用过程。

Replay 有两种：

1. dry-run replay：不真实执行副作用，只检查决策和参数。
2. live replay：在 sandbox 中真实调用或调用 mock。

对有副作用工具，默认只能 dry-run 或 sandbox replay，不能对生产系统重新执行。

Replay 需要保存：

1. 当时的 tool schema。
2. tool version。
3. model output。
4. arguments。
5. router decision。
6. permission decision。
7. executor result。

这也是上一章强调版本和 trace 的原因。

## 11.22 Executor Eval

Executor 也需要 eval。

指标包括：

1. validation correctness。
2. permission enforcement accuracy。
3. timeout rate。
4. retry success rate。
5. idempotency hit rate。
6. duplicate side effect rate。
7. cancellation success rate。
8. output schema pass rate。
9. error normalization accuracy。
10. trace completeness。

尤其要测试：

1. 参数非法。
2. 权限不足。
3. 下游超时。
4. 响应过大。
5. 重试。
6. 幂等重复。
7. 用户取消。
8. 有副作用未知状态。

Executor eval 是工程测试，不应完全依赖 LLM judge。

## 11.23 常见错误

### 11.23.1 模型一调用就执行

问题：绕过校验、权限和确认。

修复：Executor 前置 validation、authorization、confirmation。

### 11.23.2 没有超时

问题：请求挂死、资源耗尽。

修复：多层 timeout 和 cancellation。

### 11.23.3 有副作用工具无幂等

问题：重试导致重复发送、重复扣款。

修复：idempotency key 和执行记录。

### 11.23.4 取消后后台仍执行但无记录

问题：用户以为取消了，实际动作完成了。

修复：明确取消语义，记录 job 状态。

### 11.23.5 原始错误直接回填

问题：泄露堆栈、服务地址、敏感信息。

修复：错误规范化和脱敏。

### 11.23.6 输出无限制

问题：上下文爆炸，成本暴涨。

修复：输出大小限制、分页、摘要、引用。

### 11.23.7 并发结果错配

问题：结果按返回顺序回填，tool_call_id 对不上。

修复：永远按 tool_call_id 对齐。

### 11.23.8 没有 trace

问题：无法审计、回放、评估。

修复：Executor 全链路 trace。

## 11.24 面试题：如何设计 Tool Executor

面试官可能问：

```text
模型生成 tool call 后，你会怎么设计工具执行器？
```

可以这样回答。

第一，统一输入输出：

1. 内部 ToolCall。
2. 内部 ToolResult。
3. 由 provider adapter 做格式转换。

第二，执行前校验：

1. 工具是否存在且 active。
2. 参数 schema validation。
3. 业务校验。
4. 权限检查。
5. 风险和确认检查。
6. 限流和配额检查。

第三，执行控制：

1. 支持同步工具。
2. 支持异步 job。
3. 支持并发执行。
4. 设置 timeout。
5. 支持 cancellation。
6. 对只读工具做安全重试。

第四，副作用安全：

1. 高风险工具需要用户确认。
2. 使用 idempotency key。
3. 记录执行状态。
4. 状态未知时不盲目重试。
5. 必要时人工接管。

第五，结果和错误：

1. output schema validation。
2. 脱敏和字段投影。
3. 错误码规范化。
4. tool result 包装。

第六，观测和治理：

1. trace。
2. metrics。
3. replay。
4. audit log。
5. executor eval。

一句话总结：

```text
Tool Executor 是工具调用真正落地的安全执行层，必须把模型生成的意图转成经过校验、授权、限流、幂等、可观测的真实动作。
```

## 11.25 小练习

### 练习 1：能否直接执行

模型生成：

```json
{"tool_name":"send_email","arguments":{"to":"alice@example.com","body":"合同见附件"}}
```

能否直接发送？

参考答案：不能。发送邮件有外部副作用，需要参数校验、权限检查、展示草稿、用户确认、confirmation token 和幂等记录。

### 练习 2：同步还是异步

生成一个包含 20 万行日志分析的报告，应该同步执行还是异步执行？

参考答案：应异步执行，返回 job id 和进度，避免阻塞用户请求。

### 练习 3：超时含义

转账工具调用超时，能否告诉用户“转账失败，请重试”？

参考答案：不能。超时可能发生在提交后，状态未知。应查询状态，无法确认时转人工，避免重复转账。

### 练习 4：幂等键

为什么 `random_uuid()` 不适合作为重试幂等键？

参考答案：每次重试都会生成新 key，下游无法识别这是同一次意图，仍可能重复执行副作用。

### 练习 5：输出过大

工具返回 5MB 日志。Executor 应该怎么处理？

参考答案：限制输出大小，过滤相关片段，摘要或分页，必要时返回对象存储引用，不应直接塞进模型上下文。

## 11.26 本章小结

本章讲了 Tool Executor。

你需要掌握：

1. Executor 是模型 tool call 进入真实世界的最后一道工程边界。
2. Executor 输入应是统一 ToolCall，输出应是统一 ToolResult。
3. 执行前必须做工具解析、参数校验、业务校验、权限检查、风险确认、限流和幂等检查。
4. 同步执行适合低延迟只读任务，异步执行适合长任务和后台工作流。
5. 并发执行要限制并发并按 tool_call_id 对齐结果。
6. 工具必须有多层 timeout 和明确 cancellation 语义。
7. 有副作用工具必须有用户确认、幂等键和执行状态记录。
8. 执行器类型不同，安全隔离要求不同，代码、shell、文件、数据库类工具需要沙箱。
9. 工具输出要做大小限制、output schema validation、脱敏、错误规范化和结果包装。
10. Executor trace、replay 和 eval 是生产可观测性的基础。

如果只记一句话：

```text
模型可以提出动作，但 Executor 必须负责把动作变成安全、幂等、可取消、可追踪、可审计的真实执行。
```

下一章会讲工具权限模型：用户权限、租户权限和工具权限，重点解释企业工具平台如何防止越权访问和跨租户数据泄露。
