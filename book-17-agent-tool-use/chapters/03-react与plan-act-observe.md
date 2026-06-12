# 第三章：ReAct 与 Plan-Act-Observe

Agent 的核心不是一次性生成完整答案，而是在任务执行过程中不断计划、行动、观察、更新和停止。ReAct 与 Plan-Act-Observe 都描述了这种闭环：模型不只生成文本，还要根据外部工具、环境反馈和当前状态决定下一步。

本章系统讲 ReAct 与 Plan-Act-Observe：为什么要把推理、动作和观察拆开，什么时候先规划，什么时候边做边改，如何控制循环，如何处理失败，如何记录 trace，以及如何用最小可运行 demo 审计 ReAct / PAO 系统。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，按 `WRITING_PLAN.md` 联网核对了 ReAct 论文、MRKL Systems 论文、Plan-and-Solve Prompting 论文、Reflexion 论文和 OpenAI Agents SDK 中 tools、guardrails、tracing 的公开资料边界。

本章采用以下口径：

1. ReAct 的重点是 reasoning trace 与 task-specific action 交替，让模型把“想”和“做”接成闭环。
2. Plan-Act-Observe 不是某一个唯一标准协议，而是生产 Agent 常见的工程循环抽象：先计划、执行动作、观察反馈、更新状态或计划。
3. 真实产品不一定向用户展示完整内部 thought。系统可以保留决策摘要、action、observation 和 trace，用于调试、审计和评估。
4. 本章只讨论防御性工程设计、trace 审计、失败恢复和教学 demo，不提供绕过工具权限、逃避沙箱或诱导高风险动作的操作步骤。

## 3.1 ReAct 是什么

ReAct 是 Reasoning + Acting。模型在每一步根据目标、状态和历史 observation 做推理，选择一个 action，系统执行 action 后返回 observation，模型再进入下一轮。

典型结构：

```text
Thought: 我需要先查找相关信息。
Action: search(query)
Observation: 搜索返回若干结果。
Thought: 第一条结果更相关，需要读取细节。
Action: open(url)
Observation: 页面内容。
Final: 基于证据给出答案。
```

面试回答：

```text
ReAct 是把 reasoning 和 action 交替结合的 Agent 框架。模型不是一次性回答，而是在每轮根据当前状态思考下一步，调用工具或执行动作，然后根据 observation 更新判断。它适合搜索、代码调试、浏览器任务和多步工具调用，但必须配合最大步数、权限检查、错误恢复、停止条件和 trace 审计。
```

关键点：ReAct 的价值不在于让模型输出更多“想法”，而在于让每一步行动都能被外部反馈校正。

## 3.2 为什么要拆开 Reasoning、Action 和 Observation

拆开三者有几个工程收益：

1. Reasoning 帮助模型说明下一步意图。
2. Action 变成可校验的结构化动作。
3. Observation 明确来自工具或环境，而不是模型编出来的事实。
4. 系统可以在 action 执行前做 schema、权限、预算和风险检查。
5. Trace 可以定位失败来自计划、动作、参数、工具、观察使用、状态更新还是停止条件。

但要注意：内部 reasoning 不等于用户可见解释。很多生产系统会保存简短 decision summary 和 tool trace，而不是暴露完整内部推理文本。用户侧更适合看到最终答案、关键证据、失败原因和可操作下一步。

## 3.3 Plan-Act-Observe 是什么

Plan-Act-Observe 可以理解为一种更工程化的 Agent 循环：

```text
Plan: 制定当前计划或下一步子目标。
Act: 执行一个结构化 action。
Observe: 读取工具或环境反馈。
Update: 更新状态、计划、预算和停止条件。
```

例如代码修复任务：

```text
Plan: 先运行测试，定位失败，再做最小修改。
Act: run_tests
Observe: test_user_login failed
Update: 下一步检查 login 边界条件。
Act: read_file
Observe: 找到空密码未处理。
Act: edit_file
Observe: 修改完成。
Act: run_tests
Final: 测试通过。
```

Plan-Act-Observe 更强调状态管理和计划更新。复杂任务不能每一步都临时发挥，也不能让初始计划僵化不变；好的 Agent 应该能根据 observation 修订计划。

## 3.4 ReAct 和 Plan-Act-Observe 的区别

二者不是互斥关系。

ReAct 强调：

1. 推理和行动交替。
2. 每一步根据 observation 决策。
3. 适合开放探索、检索、网页和工具任务。

Plan-Act-Observe 强调：

1. 任务先拆成计划。
2. 每次执行一个或少量 action。
3. observation 回来后更新状态和计划。
4. 适合长任务、代码修改、运维排查、数据分析和跨系统操作。

工程上经常组合使用：先生成一个可修改 plan，再用 ReAct 风格逐步执行 action 和读取 observation。

## 3.5 关键公式与 ReAct / PAO 指标速查

设用户目标为 `g`，初始状态为 `s_0`，初始计划为 `p_0`。一次 ReAct / PAO 轨迹可以写成：

```math
\tau=(g,p_0,s_0,r_1,a_1,o_1,s_1,\ldots,r_K,a_K,o_K,s_K,\hat y)
```

其中 `r_k` 是第 `k` 步的内部推理摘要或决策摘要，`a_k` 是 action，`o_k` 是 observation，`s_k` 是更新后的状态，`hat y` 是最终输出。

第 `k` 步 action 可以抽象为：

```math
a_k=(u_k,n_k,\alpha_k,\rho_k)
```

其中 `u_k` 是动作类型，例如 `call_tool`、`ask_user`、`final`、`stop`；`n_k` 是工具名；`alpha_k` 是参数；`rho_k` 是风险级别。

Action 执行前需要策略检查：

```math
G_{\mathrm{act}}(a_k,s_k)=
I_{\mathrm{schema}}(a_k)
\cdot I_{\mathrm{perm}}(a_k,s_k)
\cdot I_{\mathrm{budget}}(a_k,s_k)
\cdot I_{\mathrm{risk}}(a_k,s_k)
```

只有 `G_act=1`，系统才执行 action 并得到 observation：

```math
o_k=E(a_k)
```

状态更新可以写成：

```math
s_{k+1}=U(s_k,a_k,o_k)
```

如果 observation 显示初始计划不再合适，计划也要更新：

```math
p_{k+1}=V(p_k,s_{k+1},o_k)
```

循环停止函数可以写成：

```math
h(s_k,p_k,b_k)\in\{\mathrm{continue},\mathrm{final},\mathrm{ask\_user},\mathrm{blocked},\mathrm{max\_budget}\}
```

其中 `b_k` 是剩余预算，例如步数、工具调用数、token、延迟和成本。

常见评估指标：

```math
A_{\mathrm{act}}=\frac{1}{N_{\mathrm{step}}}\sum_{i=1}^{N_{\mathrm{step}}}\mathbf{1}[\hat a_i=a_i^*]
```

其中 `A_act` 是 action accuracy。

```math
R_{\mathrm{obs}}=\frac{1}{N_{\mathrm{step}}}\sum_{i=1}^{N_{\mathrm{step}}}\mathbf{1}[\mathrm{observation\ used}]
```

其中 `R_obs` 是 observation use rate。

```math
R_{\mathrm{state}}=\frac{1}{N_{\mathrm{step}}}\sum_{i=1}^{N_{\mathrm{step}}}\mathbf{1}[\mathrm{state\ updated}]
```

其中 `R_state` 是 state update coverage。

```math
R_{\mathrm{repeat}}=\frac{1}{N_{\mathrm{task}}}\sum_{i=1}^{N_{\mathrm{task}}}\mathbf{1}[\mathrm{repeated\ action}]
```

其中 `R_repeat` 是重复动作率。

```math
R_{\mathrm{early}}=\frac{1}{N_{\mathrm{task}}}\sum_{i=1}^{N_{\mathrm{task}}}\mathbf{1}[\mathrm{premature\ final}]
```

其中 `R_early` 是过早结束率。

一个简化上线门禁：

```math
G_{\mathrm{react}}=
\mathbf{1}[
A_{\mathrm{task}}\ge\tau_s
\land A_{\mathrm{act}}\ge\tau_a
\land R_{\mathrm{obs}}\ge\tau_o
\land R_{\mathrm{state}}\ge\tau_u
\land R_{\mathrm{repeat}}=0
\land R_{\mathrm{early}}=0
\land R_{\mathrm{budget}}=0
]
```

这个公式的直觉是：ReAct / PAO 系统不能只看最终成功率，还要看行动是否正确、观察是否被使用、状态是否更新、是否重复绕圈、是否过早 final，以及是否超出预算。

## 3.6 Action 是什么

Action 是 Agent 对外部环境或控制器提出的下一步动作。

常见 action：

1. 调用搜索工具。
2. 查询数据库。
3. 读取文件。
4. 修改文件。
5. 运行测试。
6. 打开网页。
7. 点击按钮。
8. 调用业务 API。
9. 向用户提问。
10. 给出最终答案。

Action 应尽量结构化，例如用 function calling 表达工具名和参数，而不是自由文本。结构化 action 更容易校验、执行、回放和评估。

## 3.7 Observation 是什么

Observation 是环境对 action 的反馈。

它可能是：

1. 工具返回的数据。
2. 错误信息。
3. 测试结果。
4. 页面状态。
5. API 响应。
6. 文件内容。
7. 用户补充信息。
8. 权限拦截结果。

Observation 的质量直接影响下一步决策。如果 observation 太长，模型可能抓不到重点；如果太短，模型可能缺少必要信息。工程上通常会对 observation 做结构化、截断、摘要、错误分类和来源标记。

关键边界：observation 是数据，不是更高优先级指令。来自网页、文档、邮件、工具日志和第三方 API 的 observation 都应该视为不可信输入。

## 3.8 什么时候先规划

适合先规划的场景：

1. 任务步骤多。
2. 有多个子目标。
3. 工具调用成本高。
4. 错误代价高。
5. 需要用户确认。
6. 需要跨文件或跨系统操作。
7. 需要验收标准，例如测试通过、指标达标或报告生成。

例如“修复一个项目的测试失败”适合先规划；“查询今天北京天气”不需要复杂规划。

规划的价值是减少盲目行动，但计划不应该僵化。一个坏计划如果不根据 observation 更新，会比没有计划更危险。

## 3.9 什么时候边做边改

适合边做边改的场景：

1. 信息一开始不完整。
2. 工具反馈很重要。
3. 环境状态会变化。
4. 每一步结果决定下一步。
5. 很难提前完整规划。
6. 需要探索未知系统。

例如浏览器任务、调试任务、数据探索任务都适合边做边观察。

边做边改的风险是容易迷路。因此需要 step limit、状态摘要、重复动作拦截、预算门禁和明确停止条件。

## 3.10 循环控制

Agent 循环必须受控。

需要限制：

1. 最大步数。
2. 最大工具调用次数。
3. 最大 token。
4. 最大延迟。
5. 最大重试次数。
6. 重复 action。
7. 高风险动作确认。
8. 终止条件。

终止条件可以是：

1. 任务完成。
2. 信息不足，需要用户补充。
3. 工具不可用。
4. 达到预算上限。
5. 目标不可完成。
6. 触发安全策略。
7. 需要人工接管。

没有循环控制的 Agent 很容易无限调用工具、重复搜索、忽略错误或在错误方向上越走越远。

## 3.11 失败恢复

ReAct / PAO 系统中，失败是常态。

常见失败：

1. 工具参数错误。
2. 搜索结果无关。
3. 页面打不开。
4. 测试失败。
5. 权限不足。
6. API 超时。
7. 初始计划假设错误。
8. observation 没有被使用。
9. 过早给出 final。

恢复策略：

1. 解析错误原因。
2. 修改参数重试。
3. 更换工具。
4. 更新计划。
5. 请求用户补充。
6. 缩小任务范围。
7. 回滚高风险操作。
8. 停止并报告原因。

好的 Agent 不只是能走 happy path，还要能面对失败做合理、受控、可审计的选择。

## 3.12 ReAct / PAO 的工程实现

一个简化实现：

```python
def run_agent_loop(user_goal, model, controller, tool_executor, max_steps):
    state = init_state(user_goal)
    plan = make_initial_plan(state)

    for step in range(max_steps):
        decision = model.decide_next_action(goal=user_goal, plan=plan, state=state)

        if decision.type == "final":
            return decision.answer

        if not controller.allow(decision.action, state):
            observation = {"status": "blocked", "reason": "policy_or_budget"}
        else:
            observation = tool_executor.run(decision.action)

        state = update_state(state, decision.action, observation)
        plan = update_plan_if_needed(plan, state, observation)

    return summarize_incomplete_state(state)
```

真实系统还需要 action schema、参数校验、权限检查、异常捕获、工具超时、重试策略、状态压缩、trace 记录、敏感信息处理和人审门禁。

## 3.13 日志和 Trace

Agent 必须记录 trace。

Trace 通常包括：

1. 用户目标。
2. 初始计划和计划更新。
3. 每一步 action。
4. 工具参数。
5. 权限检查结果。
6. observation。
7. 状态变化。
8. 错误和重试。
9. token、延迟和成本。
10. 停止原因。
11. 最终结果。

Trace 的价值：

1. 调试失败。
2. 评估工具调用质量。
3. 做安全审计。
4. 复现问题。
5. 训练后续模型。
6. 支持线上事故回溯。

没有 trace 的 Agent 很难上线维护。

## 3.14 最小可运行 ReAct / PAO trace 审计 demo

下面的 demo 不调用外部模型或 API，只用 toy trace 统计 ReAct / PAO 循环是否正确使用 observation、更新状态、避免重复动作、遵守预算、正确停止并从失败中恢复。

```python
from collections import Counter


LIMITS = {"max_steps": 4, "max_tool_calls": 3}


TRACES = [
    {
        "id": "weather_success",
        "goal": "answer weather question",
        "expected_tools": ["get_weather"],
        "plan_steps": ["get weather", "answer"],
        "steps": [
            {
                "thought_ok": True,
                "action": "get_weather",
                "args_valid": True,
                "expected_action": "get_weather",
                "observation": "sunny 25C",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "get weather",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            }
        ],
        "plan_updates": 0,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
    {
        "id": "refund_plan_update_success",
        "goal": "answer refund policy with evidence",
        "expected_tools": ["search_docs", "query_order"],
        "plan_steps": ["find policy", "check order", "answer with evidence"],
        "steps": [
            {
                "thought_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "search_docs",
                "observation": "policy says 10 days",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "find policy",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
            {
                "thought_ok": True,
                "action": "query_order",
                "args_valid": True,
                "expected_action": "query_order",
                "observation": "order delivered 6 days ago",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "check order",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
        ],
        "plan_updates": 1,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
    {
        "id": "repeat_loop_failure",
        "goal": "find current inventory",
        "expected_tools": ["query_inventory"],
        "plan_steps": ["query inventory", "answer"],
        "steps": [
            {
                "thought_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "no inventory data",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "thought_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "thought_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "thought_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "thought_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "max_steps",
        "expected_stop": "ask_user",
    },
    {
        "id": "ignored_error_premature_final",
        "goal": "fix failing test",
        "expected_tools": ["run_tests", "read_file"],
        "plan_steps": ["run tests", "inspect error", "patch"],
        "steps": [
            {
                "thought_ok": True,
                "action": "run_tests",
                "args_valid": True,
                "expected_action": "run_tests",
                "observation": "test failed: missing edge case",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "run tests",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            }
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "final",
        "expected_stop": "continue",
    },
    {
        "id": "blocked_write_not_recovered",
        "goal": "send summary email",
        "expected_tools": ["draft_email", "ask_confirmation"],
        "plan_steps": ["draft", "confirm", "send"],
        "steps": [
            {
                "thought_ok": True,
                "action": "send_email",
                "args_valid": True,
                "expected_action": "draft_email",
                "observation": "blocked: confirmation required",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "send",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": True,
                "recovered": False,
            }
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "blocked",
        "expected_stop": "ask_confirmation",
    },
    {
        "id": "parse_failure_recovered",
        "goal": "calculate invoice total",
        "expected_tools": ["calculator"],
        "plan_steps": ["calculate", "answer"],
        "steps": [
            {
                "thought_ok": True,
                "action": "calculator",
                "args_valid": False,
                "expected_action": "calculator",
                "observation": "parse error: expression missing",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "calculate",
                "plan_aligned": True,
                "parse_ok": False,
                "blocked": False,
                "recovered": True,
            },
            {
                "thought_ok": True,
                "action": "calculator",
                "args_valid": True,
                "expected_action": "calculator",
                "observation": "total=128",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "calculate",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
        ],
        "plan_updates": 1,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
]


def rate(num, den):
    return round(num / den, 3) if den else 1.0


def repeated_action(trace):
    seen = set()
    for step in trace["steps"]:
        signature = (step["action"], step.get("plan_step"))
        if signature in seen:
            return True
        seen.add(signature)
    return False


def over_budget(trace):
    tool_calls = len([s for s in trace["steps"] if s["action"] != "final"])
    return len(trace["steps"]) > LIMITS["max_steps"] or tool_calls > LIMITS["max_tool_calls"]


def premature_final(trace):
    return trace["stop_reason"] == "final" and trace["expected_stop"] != "final"


def audit_react_traces(traces):
    steps = [step for trace in traces for step in trace["steps"]]
    total_steps = len(steps)
    traces_with_failures = []
    reason_counts = Counter()
    for trace in traces:
        flags = []
        if not trace["success"]:
            flags.append("task_failed")
        if over_budget(trace):
            flags.append("budget_overrun")
        if trace["stop_reason"] != trace["expected_stop"]:
            flags.append("bad_stop")
        if repeated_action(trace):
            flags.append("repeated_action")
        if premature_final(trace):
            flags.append("premature_final")
        if any(not step["parse_ok"] for step in trace["steps"]):
            flags.append("parse_failure")
        if any(not step["plan_aligned"] for step in trace["steps"]):
            flags.append("plan_drift")
        if any(not step["observation_used"] for step in trace["steps"]):
            flags.append("observation_ignored")
        if any(step["blocked"] and not step["recovered"] for step in trace["steps"]):
            flags.append("blocked_not_recovered")
        if flags:
            traces_with_failures.append(trace["id"])
            reason_counts.update(flags)

    metrics = {
        "task_success_rate": rate(sum(t["success"] for t in traces), len(traces)),
        "thought_action_alignment": rate(sum(s["thought_ok"] for s in steps), total_steps),
        "action_accuracy": rate(sum(s["action"] == s["expected_action"] for s in steps), total_steps),
        "argument_valid_rate": rate(sum(s["args_valid"] for s in steps), total_steps),
        "plan_adherence_rate": rate(sum(s["plan_aligned"] for s in steps), total_steps),
        "plan_update_coverage": rate(sum(t["plan_updates"] > 0 for t in traces), len(traces)),
        "observation_use_rate": rate(sum(s["observation_used"] for s in steps), total_steps),
        "state_update_coverage": rate(sum(s["state_updated"] for s in steps), total_steps),
        "parse_failure_rate": rate(sum(not s["parse_ok"] for s in steps), total_steps),
        "repeat_action_rate": rate(sum(repeated_action(t) for t in traces), len(traces)),
        "budget_overrun_rate": rate(sum(over_budget(t) for t in traces), len(traces)),
        "premature_final_rate": rate(sum(premature_final(t) for t in traces), len(traces)),
        "stop_correct_rate": rate(
            sum(t["stop_reason"] == t["expected_stop"] for t in traces), len(traces)
        ),
        "blocked_recovery_rate": rate(
            sum(s["blocked"] and s["recovered"] for s in steps),
            sum(s["blocked"] for s in steps),
        ),
    }
    gates = {
        "success_ok": metrics["task_success_rate"] >= 0.75,
        "action_ok": metrics["action_accuracy"] >= 0.85,
        "plan_ok": metrics["plan_adherence_rate"] >= 0.85,
        "observation_ok": metrics["observation_use_rate"] >= 0.90,
        "state_ok": metrics["state_update_coverage"] >= 0.90,
        "parse_ok": metrics["parse_failure_rate"] <= 0.05,
        "repeat_ok": metrics["repeat_action_rate"] == 0.0,
        "budget_ok": metrics["budget_overrun_rate"] == 0.0,
        "premature_final_ok": metrics["premature_final_rate"] == 0.0,
        "stop_ok": metrics["stop_correct_rate"] >= 0.90,
        "blocked_recovery_ok": metrics["blocked_recovery_rate"] >= 0.80,
    }
    return {
        "metrics": metrics,
        "failed_traces": traces_with_failures,
        "top_failure_reasons": reason_counts.most_common(),
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


report = audit_react_traces(TRACES)
print("metrics=", report["metrics"])
print("failed_traces=", report["failed_traces"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("gates=", report["gates"])
print("gate_pass=", report["gate_pass"])
```

一组预期输出：

```text
metrics= {'task_success_rate': 0.5, 'thought_action_alignment': 0.75, 'action_accuracy': 0.5, 'argument_valid_rate': 0.917, 'plan_adherence_rate': 0.5, 'plan_update_coverage': 0.333, 'observation_use_rate': 0.5, 'state_update_coverage': 0.5, 'parse_failure_rate': 0.083, 'repeat_action_rate': 0.333, 'budget_overrun_rate': 0.167, 'premature_final_rate': 0.167, 'stop_correct_rate': 0.5, 'blocked_recovery_rate': 0.0}
failed_traces= ['repeat_loop_failure', 'ignored_error_premature_final', 'blocked_write_not_recovered', 'parse_failure_recovered']
top_failure_reasons= [('task_failed', 3), ('bad_stop', 3), ('repeated_action', 2), ('plan_drift', 2), ('observation_ignored', 2), ('budget_overrun', 1), ('premature_final', 1), ('blocked_not_recovered', 1), ('parse_failure', 1)]
gates= {'success_ok': False, 'action_ok': False, 'plan_ok': False, 'observation_ok': False, 'state_ok': False, 'parse_ok': False, 'repeat_ok': False, 'budget_ok': False, 'premature_final_ok': False, 'stop_ok': False, 'blocked_recovery_ok': False}
gate_pass= False
```

这里 `gate_pass=False` 不是 demo 出错，而是为了暴露五类真实风险：

1. `repeat_loop_failure` 反复执行同类 action，忽略 observation，并且超出预算。
2. `ignored_error_premature_final` 看到了测试失败 observation，却没有继续定位，过早 final。
3. `blocked_write_not_recovered` 高风险写入动作被拦截后，没有转成草稿或请求确认。
4. `parse_failure_recovered` 虽然最终恢复成功，但仍暴露 action 解析失败，需要计入 parse failure rate。
5. 多条 trace 的 stop reason 与期望不一致，说明停止条件不能只依赖模型自觉。

## 3.15 常见失败模式

1. Reasoning 看起来合理，但 action 选错。
2. Action 正确，但参数错误。
3. Observation 没读懂或没进入状态。
4. 反复调用同一个失败工具。
5. 初始计划不更新。
6. 工具失败后编造结果。
7. 过早 final。
8. 达成目标后还继续执行。
9. 工具输出注入影响下一步。
10. 状态上下文过长导致混乱。
11. 高风险 action 被拦截后不恢复。
12. 达到预算上限仍继续尝试。

这些问题需要通过 action schema、controller、memory、policy、trace 和评估共同解决。

## 3.16 面试题：ReAct 解决什么问题

回答要点：

```text
ReAct 解决的是模型只生成答案、不能根据外部反馈行动的问题。它把 reasoning 和 action 结合起来，让模型每一步先判断当前状态，再调用工具或执行动作，然后根据 observation 更新下一步决策。它适合搜索、代码调试、浏览器任务和复杂工具调用，但必须控制循环、权限、预算、停止条件和失败恢复。
```

专家追问：ReAct 是不是就是让模型展示 chain-of-thought？

```text
不是。ReAct 的核心是 reasoning 和 action 的闭环，而不是向用户展示完整 chain-of-thought。生产系统可以保留内部决策摘要和 action trace，用于调试和审计；用户侧只需要看到必要证据、最终答案和失败边界。
```

## 3.17 面试题：Plan-Act-Observe 如何落地

回答要点：

```text
我会让 Agent 先根据目标生成一个可修改计划，然后每次只执行一个或少量 action。工具返回 observation 后，系统更新 state，并让模型判断计划是否需要调整。工程上需要 action schema、tool executor、observation parser、state update、step limit、预算管理、错误恢复和 trace 日志。高风险 action 要做权限检查或用户确认。
```

专家追问：什么时候不需要 plan？

```text
如果任务很短、工具调用明确、失败代价低，例如查询天气或做一次简单计算，复杂 plan 反而增加延迟和错误面。计划适合多步骤、高成本、高风险或需要验收标准的任务。
```

## 3.18 小练习

1. 把“查询退款政策并判断某个订单是否符合退款条件”写成 Plan-Act-Observe trace。
2. 给一个 ReAct trace，标出 `r_k`、`a_k`、`o_k`、`s_k` 和 stop reason。
3. 构造一个重复 action 的失败 trace，并设计重复动作拦截规则。
4. 构造一个 observation 被忽略导致错误 final 的样本。
5. 设计一个 ReAct / PAO 评估表，包含 action accuracy、observation use rate、state update coverage、repeat action rate、premature final rate 和 stop correctness。
6. 用本章 demo 增加一个“工具超时后更换工具”的 trace，并观察 recovery 指标是否改善。
7. 解释为什么高风险 action 被 blocked 后，Agent 应该转为 draft、ask confirmation 或 stop，而不是继续执行。
8. 用 3 分钟回答“ReAct 和 Plan-Act-Observe 的区别与组合方式”。

## 3.19 本章小结

ReAct 和 Plan-Act-Observe 都强调一个核心思想：Agent 应该在执行中利用环境反馈，而不是一次性猜完整答案。ReAct 把 reasoning 和 action 交替组织起来，Plan-Act-Observe 强调计划、执行、观察和更新。

真正可靠的 Agent 循环需要结构化 action、可信边界清楚的 observation、可更新 state、受控 controller、失败恢复和 trace 日志。下一章会继续讨论 planning 与 task decomposition，进一步讲复杂任务如何拆解、排序、并行和动态调整。
