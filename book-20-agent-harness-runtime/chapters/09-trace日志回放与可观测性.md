# 第九章：Trace、日志、回放与可观测性

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修，联网核对了 OpenAI Agents SDK 中 tracing、spans、processors、guardrails 和 workflow tracing 的公开资料，OpenTelemetry 关于 traces、spans、events、attributes、status、logs / metrics 关系的公开文档，W3C Trace Context 关于 `traceparent`、`tracestate` 和跨服务上下文传播的标准口径，以及 LangSmith 关于 agent trace、调试、评估、监控和从生产 trace 构建数据集的公开说明。

本章聚焦防御性的 Agent trace、日志、回放和可观测性设计：如何让一次 agent run 可解释、可复盘、可审计、可导出为评估样本，同时控制隐私、secret、artifact、保留周期和访问权限风险。正文和 demo 不提供绕过日志、隐藏高风险行为、规避审计、泄露敏感信息、篡改生产 trace 或攻击可观测性系统的方法。

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

## 9.16 Trace / Replay 质量指标

Trace 不是“多打点日志”这么简单。真正能服务 debug、评估、安全审计和回放的 trace，必须同时满足结构完整、span 因果树正确、时间线一致、artifact 可追溯、版本可复现、隐私可控和最终状态可信。

可以把第 $i$ 条 agent trace 写成：

```math
\tau_i=(g_i,S_i,E_i,A_i,V_i,R_i,M_i,Q_i,y_i)
```

其中，$g_i$ 是 trace id、session id、task id 等任务级标识，$S_i$ 是 span 集合，$E_i$ 是事件序列，$A_i$ 是 artifact 集合，$V_i$ 是模型、prompt、工具、代码、沙箱和权限策略版本，$R_i$ 是 replay 所需环境信息，$M_i$ 是可聚合指标，$Q_i$ 是评估导出标签，$y_i$ 是最终状态。

第 $j$ 个 span 可以写成：

```math
s_{ij}=(p_{ij},k_{ij},t_{ij}^{0},t_{ij}^{1},u_{ij},v_{ij},a_{ij},e_{ij})
```

其中，$p_{ij}$ 是 parent span，$k_{ij}$ 是 span 类型，$t_{ij}^{0}$ 和 $t_{ij}^{1}$ 是开始 / 结束时间，$u_{ij}$ 和 $v_{ij}$ 是输入 / 输出摘要，$a_{ij}$ 是 artifact 引用，$e_{ij}$ 是错误类型、状态和 root cause。

Trace schema 完整率：

```math
C_{\mathrm{trace}}=
\frac{1}{N}\sum_{i=1}^{N}\frac{|F_i\cap F^{\ast}|}{|F^{\ast}|}
```

其中，$F^{\ast}$ 是 trace 必填字段集合，例如 trace id、session id、task id、workflow、状态、开始时间和结束时间。

Span schema 完整率：

```math
C_{\mathrm{span}}=
\frac{1}{\sum_i |S_i|}
\sum_{i=1}^{N}\sum_{s\in S_i}
\frac{|H_s\cap H^{\ast}|}{|H^{\ast}|}
```

其中，$H^{\ast}$ 是 span 必填字段集合，例如 span id、parent id、类型、时间、状态、输入摘要、输出摘要、artifact 引用和错误字段。

Span 树合法率：

```math
C_{\mathrm{tree}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\mathrm{root}(S_i)=1\wedge \mathrm{parent\_ok}(S_i)=1]
```

一条 trace 应该有一个清晰 root span，其他 span 的 parent 必须指向同一条 trace 中真实存在的 span。否则 replay UI 会无法还原因果链。

时间线合法率：

```math
C_{\mathrm{time}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\forall s\in S_i,\ t_i^{0}\le t_s^{0}\le t_s^{1}\le t_i^{1}]
```

时间线合法不是为了好看，而是为了排查 timeout、并发、重试、取消和外部工具延迟。

Artifact 引用覆盖率：

```math
C_{\mathrm{art}}=
\frac{\sum_i\sum_{s\in S_i}\mathbf{1}[r_s=1\wedge a_s\in A_i]}
{\sum_i\sum_{s\in S_i}\mathbf{1}[r_s=1]}
```

其中，$r_s=1$ 表示该 span 需要 artifact，例如文件 diff、截断命令输出、截图、报告或长工具结果。长内容不必直接塞入 span，但引用必须可追溯。

版本捕获率：

```math
C_{\mathrm{ver}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[V_i\supseteq V^{\ast}]
```

其中，$V^{\ast}$ 通常包括模型版本、prompt 版本、工具 registry 版本、git commit、sandbox 镜像和权限策略版本。没有版本信息，replay 通常只能“看历史”，不能可靠复现。

Replay 就绪率：

```math
R_{\mathrm{replay}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[C_{\mathrm{ver},i}=1\wedge d_i=1\wedge b_i=1\wedge q_i=1]
```

其中，$d_i$ 表示工具结果或外部依赖可确定，$b_i$ 表示 sandbox / 环境快照可恢复，$q_i$ 表示权限策略可复用或可模拟。真实 replay 必须比只读 replay 更严格。

敏感信息脱敏率：

```math
C_{\mathrm{mask}}=
\frac{\sum_i\sum_{z\in Z_i}\mathbf{1}[\mathrm{masked}(z)=1]}
{\sum_i |Z_i|}
```

其中，$Z_i$ 是 trace、span、log 和 artifact 中的敏感字段集合。可观测性不能以泄露 secret、私有代码或用户数据为代价。

错误归因覆盖率：

```math
C_{\mathrm{err}}=
\frac{\sum_i\mathbf{1}[y_i\ne \mathrm{success}\wedge c_i=1]}
{\sum_i\mathbf{1}[y_i\ne \mathrm{success}]}
```

其中，$c_i=1$ 表示失败 trace 至少能定位到 error type、失败 span 和 root cause。没有错误归因，trace 很容易变成长日志存档。

最终状态一致率：

```math
C_{\mathrm{final}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[\neg (y_i=\mathrm{success}\wedge z_i=0)]
```

其中，$z_i=1$ 表示测试、验证器、业务状态或用户验收通过。Agent 声称完成但 trace 里验证失败，是典型 false completion。

指标导出覆盖率：

```math
C_{\mathrm{metric}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[M_i\supseteq M^{\ast}]
```

其中，$M^{\ast}$ 可以包含 latency、token cost、tool count、step count、error count、permission denied count 和 validation result。没有可聚合指标，就很难发现版本回归。

评估导出覆盖率：

```math
C_{\mathrm{eval}}=
\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}[Q_i\supseteq Q^{\ast}]
```

其中，$Q^{\ast}$ 包含 replay case、failure taxonomy、bad case label、expected behavior 和可用于 regression 的输入输出摘要。

Trace / Replay 门禁可以写成：

```math
G_{\mathrm{trace}}=
\mathbf{1}[
C_{\mathrm{trace}}=1
\wedge C_{\mathrm{span}}=1
\wedge C_{\mathrm{tree}}=1
\wedge C_{\mathrm{time}}=1
\wedge C_{\mathrm{art}}=1
\wedge C_{\mathrm{ver}}=1
\wedge R_{\mathrm{replay}}\ge \alpha_{\mathrm{replay}}
\wedge C_{\mathrm{mask}}=1
\wedge C_{\mathrm{err}}=1
\wedge C_{\mathrm{final}}=1
\wedge C_{\mathrm{metric}}=1
\wedge C_{\mathrm{eval}}=1
]
```

这个门禁的目标不是无限制保存所有内容，而是在隐私和成本可控的前提下，让 agent 的关键行为可解释、可回放、可评估。

## 9.16.1 最小可运行 Trace / Replay 审计 demo

下面的 demo 不调用模型、不执行工具、不访问网络，只审计 toy traces 是否满足 trace / replay 门禁。它故意构造缺 artifact、span parent 错误、时间线越界、敏感 artifact 未脱敏、外部网络不可复现、最终状态虚假成功等 bad case。

```python
from dataclasses import dataclass
from typing import Optional


TRACE_FIELDS = (
    "trace_id",
    "session_id",
    "task_id",
    "workflow",
    "status",
    "started",
    "ended",
)
SPAN_FIELDS = ("span_id", "kind", "start", "end", "status")
VERSION_FIELDS = ("model", "prompt", "tools", "git", "sandbox", "permission")
METRIC_FIELDS = ("latency_ms", "token_cost", "tool_count", "step_count")


@dataclass(frozen=True)
class Span:
    span_id: str
    parent_id: Optional[str]
    kind: str
    start: float
    end: float
    status: str
    input_summary: str = ""
    output_summary: str = ""
    artifact_ref: str = ""
    truncated: bool = False
    error_type: str = ""
    root_cause: str = ""
    sensitive_masked: bool = True


@dataclass(frozen=True)
class Trace:
    trace_id: str
    session_id: str
    task_id: str
    workflow: str
    status: str
    started: float
    ended: float
    spans: tuple[Span, ...]
    artifacts: dict[str, dict]
    versions: dict[str, str]
    replay: dict[str, bool]
    metrics: dict[str, float]
    eval_labels: dict[str, str | bool]
    validation_passed: bool


def mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else 1.0


def present(value):
    return value is not None and value != ""


def trace_schema_score(trace):
    return mean(present(getattr(trace, name)) for name in TRACE_FIELDS)


def span_schema_score(span):
    return mean(present(getattr(span, name)) for name in SPAN_FIELDS)


def tree_ok(trace):
    span_ids = {span.span_id for span in trace.spans}
    roots = [span for span in trace.spans if span.parent_id is None]
    parents_ok = all(
        span.parent_id in span_ids or span.parent_id is None for span in trace.spans
    )
    return len(roots) == 1 and parents_ok


def time_ok(trace):
    inside = all(
        trace.started <= span.start <= span.end <= trace.ended
        for span in trace.spans
    )
    ordered = all(a.start <= b.start for a, b in zip(trace.spans, trace.spans[1:]))
    return trace.started <= trace.ended and inside and ordered


def artifact_ok(trace):
    required = [
        span
        for span in trace.spans
        if span.truncated or span.kind in {"file_patch", "command"}
    ]
    return mean(span.artifact_ref in trace.artifacts for span in required)


def version_ok(trace):
    return all(trace.versions.get(name) for name in VERSION_FIELDS)


def replay_ok(trace):
    replay_fields = ("env_snapshot", "deterministic_tools", "sandbox")
    return version_ok(trace) and all(trace.replay.get(name) for name in replay_fields)


def privacy_ok(trace):
    span_ok = all(span.sensitive_masked for span in trace.spans)
    artifact_ok_ = all(
        not artifact.get("raw_secret", False) for artifact in trace.artifacts.values()
    )
    return span_ok and artifact_ok_


def error_attributed(trace):
    if trace.status == "success":
        return True
    failed = [span for span in trace.spans if span.status == "error"]
    return bool(failed) and all(span.error_type and span.root_cause for span in failed)


def final_consistent(trace):
    return not (trace.status == "success" and not trace.validation_passed)


def metric_ok(trace):
    return all(name in trace.metrics for name in METRIC_FIELDS)


def eval_ok(trace):
    if "replay_case" not in trace.eval_labels:
        return False
    if trace.status != "success" and not trace.eval_labels.get("failure_taxonomy"):
        return False
    return True


traces = (
    Trace(
        "tr_good",
        "sess_a",
        "task_bugfix",
        "code_agent",
        "success",
        0.0,
        9.0,
        (
            Span("s1", None, "agent_run", 0.0, 9.0, "ok", "goal", "completed"),
            Span("s2", "s1", "model_call", 0.2, 1.4, "ok", "context", "search_code"),
            Span("s3", "s1", "tool_call", 1.4, 2.0, "ok", "search args", "3 matches"),
            Span(
                "s4",
                "s1",
                "file_patch",
                2.2,
                3.1,
                "ok",
                "patch",
                "diff saved",
                "diff_good",
            ),
            Span(
                "s5",
                "s1",
                "command",
                3.2,
                8.2,
                "ok",
                "pytest auth",
                "passed",
                "cmd_good",
            ),
            Span("s6", "s1", "final", 8.3, 9.0, "ok", "verified", "final answer"),
        ),
        {
            "diff_good": {"kind": "diff"},
            "cmd_good": {"kind": "stdout", "raw_secret": False},
        },
        {
            "model": "gpt-x",
            "prompt": "pv12",
            "tools": "tv4",
            "git": "abc123",
            "sandbox": "py311",
            "permission": "perm8",
        },
        {"env_snapshot": True, "deterministic_tools": True, "sandbox": True},
        {"latency_ms": 9000, "token_cost": 0.19, "tool_count": 3, "step_count": 6},
        {"replay_case": True},
        True,
    ),
    Trace(
        "tr_missing_artifact",
        "sess_a",
        "task_timeout",
        "code_agent",
        "failed",
        0.0,
        6.0,
        (
            Span(
                "s1",
                None,
                "agent_run",
                0.0,
                6.0,
                "error",
                "goal",
                "failed",
                error_type="command_timeout",
                root_cause="test_hung",
            ),
            Span("s2", "s1", "model_call", 0.2, 1.1, "ok", "context", "run tests"),
            Span(
                "s3",
                "s1",
                "command",
                1.2,
                6.0,
                "error",
                "pytest",
                "truncated",
                "missing_cmd",
                True,
                "timeout",
                "no_cancel",
            ),
        ),
        {},
        {
            "model": "gpt-x",
            "tools": "tv4",
            "git": "abc123",
            "sandbox": "py311",
            "permission": "perm8",
        },
        {"env_snapshot": True, "deterministic_tools": True, "sandbox": True},
        {"latency_ms": 6000, "token_cost": 0.12, "tool_count": 1, "step_count": 3},
        {"replay_case": True, "failure_taxonomy": "timeout"},
        False,
    ),
    Trace(
        "tr_orphan",
        "sess_b",
        "task_refactor",
        "code_agent",
        "failed",
        0.0,
        5.0,
        (
            Span("s1", None, "agent_run", 0.0, 5.0, "error", "goal", "failed"),
            Span("s2", "missing_parent", "tool_call", -0.1, 1.0, "ok", "read", "file"),
            Span("s3", "s1", "final", 1.2, 5.1, "error", "summary", "failed"),
        ),
        {},
        {
            "model": "gpt-y",
            "prompt": "pv13",
            "tools": "tv5",
            "git": "def456",
            "sandbox": "py311",
            "permission": "perm8",
        },
        {"env_snapshot": True, "deterministic_tools": False, "sandbox": True},
        {"latency_ms": 5000, "tool_count": 1, "step_count": 3},
        {"replay_case": False},
        False,
    ),
    Trace(
        "tr_secret",
        "sess_c",
        "task_config",
        "code_agent",
        "failed",
        0.0,
        4.0,
        (
            Span(
                "s1",
                None,
                "agent_run",
                0.0,
                4.0,
                "error",
                "goal",
                "blocked",
                error_type="secret_leak",
                root_cause="raw_artifact",
            ),
            Span(
                "s2",
                "s1",
                "tool_call",
                0.4,
                1.0,
                "error",
                "read config",
                "secret copied",
                "raw_config",
                False,
                "sensitive_data",
                "artifact_unmasked",
                False,
            ),
        ),
        {"raw_config": {"kind": "file", "raw_secret": True}},
        {
            "model": "gpt-x",
            "prompt": "pv12",
            "tools": "tv4",
            "git": "abc123",
            "sandbox": "py311",
            "permission": "perm8",
        },
        {"env_snapshot": True, "deterministic_tools": True, "sandbox": True},
        {"latency_ms": 4000, "token_cost": 0.07, "tool_count": 1, "step_count": 2},
        {"replay_case": True, "failure_taxonomy": "privacy"},
        False,
    ),
    Trace(
        "tr_network",
        "sess_d",
        "task_docs",
        "code_agent",
        "failed",
        0.0,
        7.0,
        (
            Span(
                "s1",
                None,
                "agent_run",
                0.0,
                7.0,
                "error",
                "goal",
                "failed",
                error_type="network_unrecorded",
                root_cause="external_docs_changed",
            ),
            Span("s2", "s1", "tool_call", 0.5, 2.0, "ok", "web fetch", "doc text"),
            Span("s3", "s1", "model_call", 2.1, 6.0, "ok", "doc text", "patch plan"),
        ),
        {},
        {
            "model": "gpt-x",
            "prompt": "pv12",
            "tools": "tv4",
            "git": "abc123",
            "permission": "perm8",
        },
        {"env_snapshot": False, "deterministic_tools": False, "sandbox": False},
        {"latency_ms": 7000, "token_cost": 0.21, "tool_count": 1, "step_count": 3},
        {"failure_taxonomy": "non_deterministic"},
        False,
    ),
    Trace(
        "tr_false_success",
        "sess_e",
        "task_test",
        "code_agent",
        "success",
        0.0,
        8.0,
        (
            Span("s1", None, "agent_run", 0.0, 8.0, "ok", "goal", "success"),
            Span(
                "s2",
                "s1",
                "command",
                2.0,
                7.0,
                "error",
                "pytest",
                "failed",
                "cmd_fail",
                False,
                "test_failed",
                "assertion",
            ),
            Span("s3", "s1", "final", 7.1, 8.0, "ok", "ignored failure", "claimed fixed"),
        ),
        {"cmd_fail": {"kind": "stdout", "raw_secret": False}},
        {
            "model": "gpt-z",
            "prompt": "pv14",
            "tools": "tv4",
            "git": "abc123",
            "sandbox": "py311",
            "permission": "perm8",
        },
        {"env_snapshot": True, "deterministic_tools": True, "sandbox": True},
        {"latency_ms": 8000, "token_cost": 0.16, "tool_count": 1, "step_count": 3},
        {"replay_case": True},
        False,
    ),
)

all_spans = [span for trace in traces for span in trace.spans]
metrics = {
    "trace_schema": mean(trace_schema_score(trace) for trace in traces),
    "span_schema": mean(span_schema_score(span) for span in all_spans),
    "summary_coverage": mean(
        bool(span.input_summary and span.output_summary) for span in all_spans
    ),
    "span_tree_validity": mean(tree_ok(trace) for trace in traces),
    "timeline_validity": mean(time_ok(trace) for trace in traces),
    "artifact_coverage": mean(artifact_ok(trace) for trace in traces),
    "version_capture": mean(version_ok(trace) for trace in traces),
    "replay_readiness": mean(replay_ok(trace) for trace in traces),
    "privacy_masking": mean(privacy_ok(trace) for trace in traces),
    "error_attribution": mean(error_attributed(trace) for trace in traces),
    "final_consistency": mean(final_consistent(trace) for trace in traces),
    "metric_export": mean(metric_ok(trace) for trace in traces),
    "eval_export": mean(eval_ok(trace) for trace in traces),
}

root_causes = {}
for trace in traces:
    causes = []
    if trace_schema_score(trace) < 1:
        causes.append("trace_schema_incomplete")
    if not tree_ok(trace):
        causes.append("span_tree_invalid")
    if not time_ok(trace):
        causes.append("timeline_invalid")
    if artifact_ok(trace) < 1:
        causes.append("artifact_reference_missing")
    if not version_ok(trace):
        causes.append("version_capture_missing")
    if not replay_ok(trace):
        causes.append("replay_not_ready")
    if not privacy_ok(trace):
        causes.append("sensitive_data_unmasked")
    if not error_attributed(trace):
        causes.append("error_not_attributed")
    if not final_consistent(trace):
        causes.append("final_status_inconsistent")
    if not metric_ok(trace):
        causes.append("metrics_incomplete")
    if not eval_ok(trace):
        causes.append("eval_export_missing")
    if causes:
        root_causes[trace.trace_id] = causes

thresholds = {
    "trace_schema": 1.0,
    "span_schema": 1.0,
    "summary_coverage": 0.95,
    "span_tree_validity": 1.0,
    "timeline_validity": 1.0,
    "artifact_coverage": 1.0,
    "version_capture": 1.0,
    "replay_readiness": 0.95,
    "privacy_masking": 1.0,
    "error_attribution": 1.0,
    "final_consistency": 1.0,
    "metric_export": 1.0,
    "eval_export": 1.0,
}
failed_gates = [
    name for name, threshold in thresholds.items() if metrics[name] < threshold
]

print(f"metrics={metrics}")
print(f"root_causes={root_causes}")
print(f"failed_gates={failed_gates}")
print(f"trace_replay_gate_pass={not failed_gates}")
```

一组故意包含坏 case 的输出示例：

```text
metrics={'trace_schema': 1.0, 'span_schema': 1.0, 'summary_coverage': 1.0, 'span_tree_validity': 0.833, 'timeline_validity': 0.833, 'artifact_coverage': 0.833, 'version_capture': 0.667, 'replay_readiness': 0.5, 'privacy_masking': 0.833, 'error_attribution': 0.833, 'final_consistency': 0.833, 'metric_export': 0.833, 'eval_export': 0.667}
root_causes={'tr_missing_artifact': ['artifact_reference_missing', 'version_capture_missing', 'replay_not_ready'], 'tr_orphan': ['span_tree_invalid', 'timeline_invalid', 'replay_not_ready', 'error_not_attributed', 'metrics_incomplete', 'eval_export_missing'], 'tr_secret': ['sensitive_data_unmasked'], 'tr_network': ['version_capture_missing', 'replay_not_ready', 'eval_export_missing'], 'tr_false_success': ['final_status_inconsistent']}
failed_gates=['span_tree_validity', 'timeline_validity', 'artifact_coverage', 'version_capture', 'replay_readiness', 'privacy_masking', 'error_attribution', 'final_consistency', 'metric_export', 'eval_export']
trace_replay_gate_pass=False
```

这个 demo 的重点是把“trace 不完整”拆成可修复的系统问题：补 artifact 引用、补 prompt / sandbox 版本、修正 span parent 和时间线、对 artifact 做脱敏、记录 failure taxonomy、把最终成功绑定到真实验证结果。

## 9.17 面试题：Agent Trace 应该记录什么

回答要点：

```text
Agent trace 应该记录一次任务的完整执行链路，包括用户请求、session/task id、模型版本、prompt 版本、每轮上下文摘要、模型输出、解析出的 action、工具调用、工具结果、文件 diff、命令输出、权限决策、错误和恢复、token 成本、耗时和最终状态。大对象可以放 artifact，通过引用关联。这样才能支持 debug、replay、evaluation 和安全审计。
```

## 9.18 面试题：Replay 有什么用

回答要点：

```text
Replay 可以复现或模拟一次 agent 执行过程。只读 replay 用于看时间线和 debug；模拟 replay 可以固定工具结果、换模型或 prompt 对比行为；真实 replay 可以在 sandbox 里重新执行任务，用于 regression test 和 benchmark。Replay 的关键是 trace 里要记录模型、prompt、工具、环境和权限策略版本，否则很难复现。
```

## 9.19 面试题：如何处理 Trace 中的隐私和 Secret

回答要点：

```text
Trace 很可能包含私有代码、用户输入、命令输出、diff 和 secret，所以必须默认脱敏和访问控制。Secret pattern 要自动 mask，敏感文件默认不进入模型上下文，trace artifact 可以加密并设置保留周期。用于训练或长期分析前，需要做数据治理和用户/企业策略授权。不能为了可观测性无限制保存所有内容。
```

## 9.20 小练习

1. 设计一个 agent trace schema，包含 task、model、tool、file、command、permission 和 error 字段。
2. 设计一个 replay 流程，用于复现一次失败的 code edit 任务。
3. 列出 10 个 agent runtime 可观测性指标。
4. 思考 trace 中哪些字段需要脱敏。
5. 设计一个 Trace UI 时间线视图。
6. 思考如何从 trace 中构建 regression test。
7. 思考命令输出被截断后如何支持后续 debug。
8. 用 3 分钟回答“为什么没有 trace 的 coding agent 很难生产化”。

## 9.21 本章总结

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
