# 第九章：Trace、日志、回放与可观测性

## 9.1 本章目标

Agent 是多步系统。它会构造上下文、调用模型、解析动作、执行工具、修改文件、运行命令、处理权限、失败重试、最终总结。任何一步都可能出错。如果没有 trace，你只能看到“agent 失败了”，却不知道它为什么失败、看到了什么、调用了什么、改了什么、哪里判断错了。

本章讨论运行时可观测性：如何记录 trace、如何设计日志、如何支持 replay、如何做 debug、安全审计和质量监控。下一章会讲 evaluation harness，本章重点是一次真实 agent 执行过程如何被记录、复现和观察。

学完本章，你应该能回答：

1. 为什么 agent runtime 必须有 trace。
2. 一条完整 trace 应该记录哪些字段。
3. Trace、log、metric、event、artifact 的区别是什么。
4. Replay 如何帮助 debug、eval 和 regression test。
5. 如何在可观测性和隐私安全之间取舍。
6. 如何用 trace 分析 agent 失败原因。
7. 面试中如何设计 agent 的日志回放系统。

## 9.2 为什么需要 Trace

普通 LLM 调用通常是一问一答，记录 prompt 和 response 就能复盘大部分问题。

Agent 不一样。Agent 的结果来自一条执行链：

```text
用户任务
-> context builder
-> 模型输出
-> action parser
-> permission decision
-> tool execution
-> observation
-> state update
-> 下一轮
```

如果只记录最终回答，无法回答：

1. 模型是否看到了正确文件。
2. 工具是否选对。
3. 参数是否构造正确。
4. 权限是否被正确判断。
5. 命令是否真实执行。
6. 测试失败信息是否被压缩错。
7. 文件 diff 是否符合预期。
8. 哪一轮开始偏离目标。

Trace 的核心价值：让 agent 行为可解释、可复现、可审计、可评估。

面试表达：

```text
Agent 是多步系统，最终结果只是最后一层。Trace 要记录每一轮的上下文、模型输出、工具调用、权限决策、工具结果、文件 diff、命令输出、错误和恢复。这样失败时才能定位是上下文缺失、模型判断错、工具失败、权限拦截还是执行环境问题。
```

## 9.3 Trace、Log、Metric、Event、Artifact 的区别

这些概念容易混用。

Trace：

```text
一次任务从开始到结束的完整链路记录。
```

它强调因果关系和时序。

Log：

```text
系统在某个时间点输出的文本或结构化日志。
```

它强调事件详情。

Metric：

```text
可聚合的数值指标，例如成功率、错误率、耗时、token 成本。
```

它强调统计和监控。

Event：

```text
一个离散事件，例如 tool_called、permission_denied、command_timeout。
```

它强调状态变化。

Artifact：

```text
执行过程中产生的大对象，例如完整命令输出、文件 diff、截图、生成文件、评估报告。
```

它强调可引用的大内容。

设计上可以这样分：

1. Trace 串起完整过程。
2. Event 构成 trace 的节点。
3. Log 保存细节。
4. Metric 负责聚合统计。
5. Artifact 保存大对象。

## 9.4 一条完整 Trace 应记录什么

一条完整 agent trace 应记录：

任务级信息：

1. trace_id。
2. session_id。
3. task_id。
4. 用户输入。
5. 用户配置和权限策略。
6. workspace 信息。
7. 开始时间和结束时间。
8. 最终状态：success、failed、cancelled。

模型级信息：

1. 模型名称和版本。
2. 模型参数。
3. prompt/template 版本。
4. 每轮上下文摘要。
5. 模型输出。
6. token 用量。
7. 模型 API 错误和重试。

工具级信息：

1. 工具名和版本。
2. 工具输入参数。
3. 权限决策。
4. 用户确认结果。
5. 工具输出摘要。
6. 错误类型。
7. 耗时。

文件和命令信息：

1. 读取了哪些文件。
2. 修改了哪些文件。
3. 文件 diff。
4. 运行了哪些命令。
5. stdout/stderr 摘要。
6. exit code。
7. 是否截断和脱敏。

状态信息：

1. task state 变化。
2. plan 更新。
3. error handler 决策。
4. retry 次数。
5. rollback 或 recovery 行为。

安全信息：

1. 权限请求。
2. 权限拒绝。
3. prompt injection 迹象。
4. secret masking。
5. 网络访问。
6. 高风险命令拦截。

## 9.5 Trace Schema 示例

一个简化 trace schema：

```text
trace_id: tr_123
session_id: sess_456
task:
  user_request: fix empty password login 500 and add test
  status: success
  started_at: ...
  ended_at: ...

environment:
  workspace_root: /workspace/app
  git_commit: abc123
  branch: feature/auth-fix

steps:
  - step_id: 1
    type: model_call
    model: gpt-x
    prompt_version: pv12
    context_summary: user goal + auth files + plan
    output_summary: call search_code for login

  - step_id: 2
    type: tool_call
    tool: search_code
    tool_version: v1
    input: { pattern: "login" }
    output_summary: 3 matches
    status: success

  - step_id: 3
    type: file_patch
    files: [src/auth/login.ts]
    diff_ref: artifact://diff-789
    status: success

  - step_id: 4
    type: command
    command: npm test -- auth
    exit_code: 0
    duration_ms: 4321
    output_ref: artifact://cmd-456
```

实际系统可以用 JSON、protobuf、数据库表或事件流。关键是字段稳定、可查询、可回放。

## 9.6 Step 粒度如何设计

Trace 粒度太粗，无法 debug；太细，成本和隐私风险上升。

常见 step 粒度：

1. model_call。
2. tool_call。
3. permission_check。
4. file_read。
5. file_patch。
6. command_execution。
7. state_update。
8. user_confirmation。
9. error_recovery。
10. final_answer。

每个 step 应包含：

1. step_id。
2. parent_step_id 或 previous_step_id。
3. timestamp。
4. type。
5. input summary。
6. output summary。
7. status。
8. error_type。
9. artifact refs。

不要把所有内容都直接塞在 step 里。大文件、长日志、完整 prompt 可以放 artifact，再用 ref 引用。

## 9.7 Replay 是什么

Replay 是用 trace 复现一次 agent 执行过程。

Replay 可以有不同层级。

只读 replay：

1. 重放 trace 时间线。
2. 展示模型输入输出。
3. 展示工具调用和结果。
4. 不重新执行任何动作。

模拟 replay：

1. 使用 trace 中记录的工具结果。
2. 重新调用模型或不同模型。
3. 比较新旧输出。
4. 不触碰真实环境。

真实 replay：

1. 重新初始化环境。
2. 重新执行工具和命令。
3. 用于 benchmark 或 regression。
4. 风险更高，需要 sandbox。

Replay 的价值：

1. Debug。
2. Evaluation。
3. Regression test。
4. 安全审计。
5. 模型/工具版本对比。

## 9.8 Replay 的难点

Replay 听起来简单，实际很难。

难点包括：

1. 环境变化。
2. 文件版本变化。
3. 依赖版本变化。
4. 工具版本变化。
5. 模型输出非确定性。
6. 时间、网络和外部 API 不可复现。
7. Secret 和权限不可复用。
8. 随机数和并发影响结果。

提高可复现性需要记录：

1. git commit。
2. 工具版本。
3. prompt 版本。
4. 模型版本和参数。
5. 环境配置。
6. 依赖锁文件。
7. 权限策略版本。
8. sandbox 镜像版本。

Replay 不是只保存最终答案，而是保存足够恢复执行环境的信息。

## 9.9 Debug Workflow：如何用 Trace 排查失败

Agent 失败后，可以按 trace 分层排查。

第一，看任务理解。

1. 用户目标是否被正确保留。
2. 是否有误解或范围扩大。
3. 最新用户指令是否覆盖旧计划。

第二，看上下文。

1. 关键文件是否进入上下文。
2. 测试错误是否被正确摘要。
3. 是否有过期 summary。

第三，看模型输出。

1. 是否选择正确工具。
2. 是否构造正确参数。
3. 是否幻觉不存在文件。

第四，看工具执行。

1. 工具是否成功。
2. 错误是否结构化。
3. 权限是否拦截。

第五，看环境反馈。

1. 测试失败是否被正确理解。
2. 命令输出是否截断过度。
3. 文件 diff 是否符合预期。

第六，看恢复策略。

1. 是否盲目重试。
2. 是否请求用户。
3. 是否回滚错误修改。

Trace 能把“agent 不行”拆成具体失败点。

## 9.10 可观测性指标

Agent runtime 可以监控很多指标。

任务指标：

1. Task success rate。
2. Failure rate。
3. Cancellation rate。
4. Human intervention rate。
5. Average task duration。

模型指标：

1. Model call count。
2. Token cost。
3. Model error rate。
4. Invalid action rate。
5. Retry rate。

工具指标：

1. Tool call count。
2. Tool success rate。
3. Tool timeout rate。
4. Schema error rate。
5. Tool selection distribution。

安全指标：

1. Permission denied count。
2. Dangerous command blocked count。
3. Secret masking count。
4. Network access request count。
5. Prompt injection detection count。

质量指标：

1. Test pass rate。
2. Regression rate。
3. Unrelated file modification rate。
4. No-verification completion rate。
5. Rollback rate。

这些指标可以帮助团队发现 runtime、工具、模型和策略的问题。

## 9.11 日志太少和日志太多

日志太少的问题：

1. 无法复盘。
2. 无法定位失败点。
3. 无法比较版本。
4. 无法做安全审计。

日志太多的问题：

1. 成本高。
2. 检索困难。
3. 泄露隐私。
4. 保存敏感代码和 secret。
5. 噪声淹没关键事件。

策略：

1. Trace 保存结构化关键事件。
2. 大对象放 artifact。
3. 默认脱敏。
4. 设置保留周期。
5. 按权限控制访问。
6. 支持采样和压缩。

可观测性的目标不是记录一切，而是记录足够解释、复现和审计的信息。

## 9.12 隐私、脱敏和保留周期

Trace 可能包含敏感内容：

1. 私有代码。
2. 用户输入。
3. 环境变量。
4. 命令输出。
5. 文件 diff。
6. API key。
7. 业务数据。

必须考虑：

1. 哪些字段脱敏。
2. 哪些 artifact 加密。
3. 谁能访问 trace。
4. 保存多久。
5. 是否可用于训练。
6. 是否允许导出。
7. 用户是否能删除。

默认策略：

1. Secret pattern 自动 mask。
2. `.env`、SSH key、token 不进入模型上下文。
3. Trace 访问按角色控制。
4. 长期保存只保存摘要和指标。
5. 用于训练前必须做数据治理。

## 9.13 Trace 和 Evaluation Harness 的关系

Trace 是 evaluation harness 的原材料之一。

Evaluation harness 可以使用 trace：

1. 重放历史任务。
2. 构建 regression set。
3. 分析失败类型。
4. 比较模型和工具版本。
5. 统计成本和步数。
6. 审计安全事件。

但 trace 不等于 evaluation harness。

Trace 是单次或多次任务执行记录。Evaluation harness 是自动运行任务、计算指标、生成报告和做回归的框架。

下一章会重点展开 evaluation harness。

## 9.14 Trace UI 应该展示什么

如果 trace 只能存在数据库里，使用价值会下降。

Trace UI 可以展示：

1. 时间线。
2. 每轮模型调用。
3. 工具调用列表。
4. 权限确认。
5. 文件 diff。
6. 命令输出摘要。
7. 错误和 retry。
8. token 和耗时。
9. 安全事件。
10. 最终结果。

好的 Trace UI 应该支持：

1. 按 step 展开。
2. 搜索工具调用。
3. 查看 diff。
4. 查看截断输出的完整 artifact。
5. 标记 bad case。
6. 导出 replay case。

可观测性不只是后端日志，也是产品体验的一部分。

## 9.15 真实坑

常见真实坑：

1. 日志太少无法复盘。
2. 日志太多泄露隐私。
3. 没有记录 prompt 版本。
4. 工具结果不可复现。
5. 没有记录权限决策。
6. 命令输出被截断但未标记。
7. 文件 diff 没进入 trace。
8. Replay 时环境和原任务不一致。
9. Trace 中保存了 secret。
10. 指标只看最终成功率，不看过程风险。

经验法则：没有 trace 的 agent 失败不可 debug，没有 replay 的 agent 改进不可验证。

## 9.16 面试题：Agent Trace 应该记录什么

回答要点：

```text
Agent trace 应该记录一次任务的完整执行链路，包括用户请求、session/task id、模型版本、prompt 版本、每轮上下文摘要、模型输出、解析出的 action、工具调用、工具结果、文件 diff、命令输出、权限决策、错误和恢复、token 成本、耗时和最终状态。大对象可以放 artifact，通过引用关联。这样才能支持 debug、replay、evaluation 和安全审计。
```

## 9.17 面试题：Replay 有什么用

回答要点：

```text
Replay 可以复现或模拟一次 agent 执行过程。只读 replay 用于看时间线和 debug；模拟 replay 可以固定工具结果、换模型或 prompt 对比行为；真实 replay 可以在 sandbox 里重新执行任务，用于 regression test 和 benchmark。Replay 的关键是 trace 里要记录模型、prompt、工具、环境和权限策略版本，否则很难复现。
```

## 9.18 面试题：如何处理 Trace 中的隐私和 Secret

回答要点：

```text
Trace 很可能包含私有代码、用户输入、命令输出、diff 和 secret，所以必须默认脱敏和访问控制。Secret pattern 要自动 mask，敏感文件默认不进入模型上下文，trace artifact 可以加密并设置保留周期。用于训练或长期分析前，需要做数据治理和用户/企业策略授权。不能为了可观测性无限制保存所有内容。
```

## 9.19 小练习

1. 设计一个 agent trace schema，包含 task、model、tool、file、command、permission 和 error 字段。
2. 设计一个 replay 流程，用于复现一次失败的 code edit 任务。
3. 列出 10 个 agent runtime 可观测性指标。
4. 思考 trace 中哪些字段需要脱敏。
5. 设计一个 Trace UI 时间线视图。
6. 思考如何从 trace 中构建 regression test。
7. 思考命令输出被截断后如何支持后续 debug。
8. 用 3 分钟回答“为什么没有 trace 的 coding agent 很难生产化”。

## 9.20 本章总结

本章讨论了 Trace、日志、回放与可观测性。

核心结论：

1. Agent 是多步系统，只看最终回答无法复盘失败原因。
2. Trace 应记录用户输入、上下文、模型输出、工具调用、权限、文件 diff、命令输出、错误恢复和最终状态。
3. Log、metric、event、artifact 和 trace 各有不同职责。
4. Replay 可以用于 debug、evaluation、regression test 和安全审计。
5. 可观测性指标要覆盖任务、模型、工具、安全和质量。
6. 日志太少无法复盘，日志太多会带来成本、隐私和噪声问题。
7. Trace 必须考虑脱敏、访问控制和保留周期。
8. Trace 是下一章 evaluation harness 的基础输入之一。

下一章会进入 Evaluation Harness，系统讨论如何自动运行 agent 评估任务、构造数据集、复现环境、计算指标和生成回归报告。
