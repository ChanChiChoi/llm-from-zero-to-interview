# 第四章：Planning 与 Task Decomposition

Agent 要完成复杂任务，不能只靠一步一步临时反应。它需要把大目标拆成可执行的子目标，安排顺序，识别依赖，控制预算，执行后根据 observation 修正计划。Planning 与 task decomposition 是 Agent 从“会调用工具”走向“能稳定完成复杂任务”的核心能力。

本章系统讲 Agent 规划与任务分解：为什么要分解任务，如何定义子目标，如何表达依赖图，计划是一次性生成还是动态修正，如何处理并行、长期任务和失败恢复，如何评估规划质量，以及如何用 0 依赖 Python demo 审计规划质量。

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修时，按 `WRITING_PLAN.md` 联网核对了 Plan-and-Solve Prompting、LLM+P、Tree of Thoughts、Reflexion、ReAct 和 OpenAI Agents SDK 中 tools / guardrails / tracing 的公开资料边界。

本章采用以下口径：

1. Planning 不是让模型生成漂亮清单，而是把目标转成可执行、可验证、可回退的子任务结构。
2. LLM planner 灵活但不等于可靠规划器；涉及严格约束、资源调度或高风险动作时，应结合规则、搜索、验证器、传统 planner、权限门禁和人工确认。
3. Task decomposition 必须保留原始目标、约束和验收标准，不能只把任务机械拆小。
4. 动态重规划是可靠 Agent 的关键：计划要根据 observation、失败、预算、权限和用户变更修正。
5. 本章只讨论防御性工程设计、规划质量审计和教学 demo，不提供绕过权限、逃避审批或规避审计的操作方法。

## 4.1 为什么 Agent 需要 Planning

简单任务可以直接执行。例如“查询某个订单状态”，Agent 只需要一次工具调用。

复杂任务不同。例如：

```text
分析这个项目测试失败的原因，修复代码，补充必要测试，并总结修改。
```

这个任务至少包含：

1. 了解项目结构。
2. 运行测试。
3. 定位失败。
4. 阅读相关代码。
5. 制定修复方案。
6. 修改代码。
7. 再次运行测试。
8. 总结结果。

如果没有规划，Agent 可能乱查文件、反复运行同一命令、过早修改代码、忘记验证或在测试仍失败时声称完成。

面试回答：

```text
Planning 的作用是把复杂目标拆成可执行、可检查的子任务，并决定执行顺序、依赖关系、预算和停止条件。它能减少盲目工具调用，帮助 Agent 管理长期任务和复杂依赖。但计划不能僵化，执行过程中需要根据 observation 动态修正。
```

## 4.2 Task Decomposition 是什么

Task decomposition 是把大任务拆成多个子任务。

好的子任务应该满足：

1. 目标明确。
2. 可以执行。
3. 可以验证。
4. 输入输出清楚。
5. 粒度适中。
6. 有依赖关系说明。
7. 有失败处理方式。

坏的分解常见问题：

1. 子任务太大，仍然不可执行。
2. 子任务太小，步骤过多。
3. 顺序错误。
4. 忽略依赖。
5. 没有验收标准。
6. 分解后丢失原始目标。
7. 高风险步骤没有确认或回滚。

任务分解不是把目标随便拆成列表，而是把它变成可执行的工作流。

## 4.3 子目标和验收标准

每个子目标都应该有验收标准。

例如：

```text
子目标：定位测试失败原因
验收标准：能指出失败测试、触发条件、相关代码位置和根因假设
```

再例如：

```text
子目标：修复 bug
验收标准：相关测试通过，且没有引入明显回归
```

没有验收标准，Agent 容易“看起来做了很多”，但不知道是否真的完成。规划质量评估中，acceptance criteria coverage 应该是核心指标。

## 4.4 关键公式与 Planning 指标速查

设用户目标为 `g`，目标包含一组需求：

```math
\mathcal{R}_g=\{r_1,\ldots,r_m\}
```

任务分解产生子任务集合：

```math
\mathcal{V}=\{v_1,\ldots,v_n\}
```

每个子任务可以抽象为：

```math
v_i=(q_i,I_i,O_i,A_i,D_i,C_i,\rho_i)
```

其中 `q_i` 是子目标描述，`I_i` 是输入，`O_i` 是输出，`A_i` 是可用 action 或工具，`D_i` 是依赖集合，`C_i` 是验收标准，`rho_i` 是风险级别。

依赖图可以写成：

```math
G=(\mathcal{V},\mathcal{E})
```

如果 `(v_i,v_j)\in\mathcal{E}`，表示 `v_j` 必须在 `v_i` 之后执行。一个合法执行顺序 `pi` 应满足：

```math
(v_i,v_j)\in\mathcal{E}\Rightarrow \pi(i)<\pi(j)
```

目标覆盖率：

```math
C_{\mathrm{goal}}=
\frac{|\bigcup_i R(v_i)\cap\mathcal{R}_g|}{|\mathcal{R}_g|}
```

其中 `R(v_i)` 是子任务覆盖的目标需求。这个指标回答“计划是否覆盖原始目标”。

可执行步骤比例：

```math
R_{\mathrm{exec}}=
\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[v_i\ \mathrm{is\ executable}]
```

验收标准覆盖率：

```math
R_{\mathrm{acc}}=
\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[C_i\neq\emptyset]
```

依赖违规率：

```math
R_{\mathrm{dep}}=
\frac{1}{|\mathcal{E}|}\sum_{(v_i,v_j)\in\mathcal{E}}
\mathbf{1}[\pi(i)>\pi(j)]
```

动态重规划覆盖率：

```math
R_{\mathrm{replan}}=
\frac{\sum_t \mathbf{1}[\mathrm{failure}_t\land \mathrm{plan\ updated}_t]}
{\sum_t \mathbf{1}[\mathrm{failure}_t]}
```

关键路径长度可以写成：

```math
L_{\mathrm{crit}}=\max_{p\in\mathcal{P}(G)}\sum_{v_i\in p}c_i
```

其中 `c_i` 是子任务成本，`\mathcal{P}(G)` 是依赖图中的路径集合。关键路径决定了即使并行执行，任务理论上也不能短于多少成本。

一个简化 planning gate：

```math
G_{\mathrm{plan}}=
\mathbf{1}[
C_{\mathrm{goal}}\ge\tau_g
\land R_{\mathrm{exec}}\ge\tau_e
\land R_{\mathrm{acc}}\ge\tau_a
\land R_{\mathrm{dep}}=0
\land R_{\mathrm{replan}}\ge\tau_r
\land R_{\mathrm{risk}}=1
]
```

其中 `R_risk` 表示高风险步骤都有确认、权限或回滚方案。这个 gate 的直觉是：一个计划必须覆盖目标、可执行、可验证、依赖合法、能根据失败重规划，并且高风险步骤受控。

## 4.5 计划的粒度

计划粒度要适中。

太粗：

```text
1. 修复项目
2. 总结结果
```

这种计划对执行没帮助。

太细：

```text
1. 打开文件 A
2. 看第 1 行
3. 看第 2 行
...
```

这种计划成本高，也容易僵化。

合适粒度通常是“一个工具调用或一组紧密相关动作可以推进的子目标”。例如“运行相关测试并记录失败信息”“定位失败函数和输入边界”“做最小修复并补测试”就是比较合适的粒度。

## 4.6 一次性规划与动态重规划

一次性规划：先生成完整计划，然后执行。

优点：

1. 全局结构清晰。
2. 便于用户确认。
3. 适合流程稳定任务。
4. 成本更容易估算。

缺点：

1. 环境变化后容易失效。
2. 早期信息不足时计划质量差。
3. 容易僵化执行错误计划。

动态重规划：执行中不断更新计划。

优点：

1. 能适应 observation。
2. 适合调试、搜索、浏览器等开放任务。
3. 能处理失败和不确定性。

缺点：

1. 可能反复改计划。
2. 成本更高。
3. 需要更强状态管理。
4. 更依赖 trace 和停止条件。

实际系统通常结合二者：先生成粗计划，再每一步根据 observation 修正。

## 4.7 依赖关系与拓扑顺序

很多子任务有依赖关系。

例如代码修复：

```text
运行测试 -> 定位失败 -> 阅读代码 -> 修改代码 -> 再运行测试
```

不能在不知道失败原因时随意修改代码，也不能在修改前声称测试通过。

依赖关系可以分为：

1. 信息依赖：需要先获取信息。
2. 资源依赖：需要先拿到文件、权限或数据。
3. 顺序依赖：必须按顺序执行。
4. 验证依赖：必须先验证才能进入下一步。
5. 安全依赖：高风险动作前必须确认或审批。

Agent 如果忽略依赖，会产生无效行动甚至危险操作。工程上可以把子任务构造成依赖图，用拓扑顺序检查计划是否合法。

## 4.8 并行任务和关键路径

有些子任务可以并行。

例如写报告时：

1. 检索背景资料。
2. 收集竞品信息。
3. 查询内部指标。
4. 整理用户反馈。

这些可以并行执行，然后汇总。

但并行也有风险：

1. 成本上升。
2. 状态合并复杂。
3. 工具限流。
4. 结果冲突。
5. 安全审计更难。

工程上需要区分“可并行”和“必须串行”。如果两个子任务之间没有路径依赖，且工具、权限、预算允许，它们才适合并行。最终总耗时受关键路径限制，而不是受总步骤数直接限制。

## 4.9 长期任务

长期任务可能持续数小时、数天甚至更久。

例如：

1. 持续监控指标。
2. 自动处理工单。
3. 逐步整理知识库。
4. 跟踪实验结果。
5. 周期性生成报告。

长期任务需要：

1. 持久化状态。
2. 任务队列。
3. 断点恢复。
4. 定时触发。
5. 权限续期。
6. 人工确认节点。
7. 审计日志。
8. 失败告警。

长期 Agent 不能只依赖一次 prompt 上下文，必须有外部状态存储和恢复机制。

## 4.10 计划修正

需要修正计划的信号：

1. 工具返回和预期不一致。
2. 子目标无法完成。
3. 发现新约束。
4. 用户改变需求。
5. 预算不足。
6. 安全策略阻止动作。
7. 执行结果显示原假设错误。
8. 依赖任务失败。

修正计划时要保留原始目标，不要被局部 observation 带偏。最好记录“为什么修改计划”：是失败恢复、目标变更、证据冲突、预算收缩，还是权限限制。

## 4.11 失败恢复

Planning 中的失败恢复不只是重试。

常见恢复方式：

1. 重试同一步。
2. 更换工具。
3. 回到上一个状态。
4. 缩小子目标。
5. 请求用户补充信息。
6. 放弃不可完成分支。
7. 输出部分完成结果。
8. 触发人工接管。

例如工具超时，可以重试；权限不足，不能无限重试，应请求授权或降级；测试失败，应该分析原因，而不是重复运行相同测试。

## 4.12 规划中的 Memory

Memory 对 planning 很重要。

短期 memory 保存当前任务状态：

1. 当前计划。
2. 已完成步骤。
3. observation。
4. 失败尝试。
5. 剩余预算。

长期 memory 保存跨任务经验：

1. 用户偏好。
2. 常用工具。
3. 历史项目结构。
4. 常见错误模式。
5. 成功方案模板。

但 memory 也可能污染计划。过期记忆、错误记忆或不相关记忆会让 Agent 做错决策。因此 memory 需要检索、过滤和更新时间。

## 4.13 Planner 的实现方式

Planner 可以有多种实现：

1. Prompt-based planner：直接让 LLM 生成计划。
2. Rule-based planner：用规则和流程模板生成计划。
3. Hybrid planner：规则控制框架，LLM 填充细节。
4. Search-based planner：生成多个计划并评分。
5. Hierarchical planner：高层规划拆目标，低层执行动作。
6. External planner：把形式化目标交给传统规划器、调度器或求解器。

纯 LLM planner 灵活，但稳定性差；纯规则 planner 稳定，但适应性弱。工程上常用混合方案：关键流程、权限、审批和停止条件由系统控制，开放部分由 LLM 规划。

## 4.14 计划评估

评估一个计划可以看：

1. 是否覆盖目标。
2. 子任务是否可执行。
3. 依赖顺序是否正确。
4. 是否有验收标准。
5. 是否考虑风险和权限。
6. 是否成本可控。
7. 是否能根据反馈修正。
8. 是否有合理停止条件。
9. 是否能恢复失败分支。

最终评估还要看任务成功率，而不是计划文本是否漂亮。一个计划写得很完整但执行不了，没有价值。

## 4.15 最小可运行 planning audit demo

下面的 demo 不调用外部 API，只用 toy planning 记录审计目标覆盖、可执行性、验收标准、依赖顺序、动态重规划、失败恢复、高风险确认、过长计划、重复子目标和最终成功率。

```python
from collections import Counter


MAX_STEPS = 5


PLANS = [
    {
        "id": "code_fix_good",
        "requirements": {"inspect", "test", "patch", "verify", "summarize"},
        "replanned": False,
        "final_success": True,
        "steps": [
            {"id": "run_tests", "covers": {"test"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "inspect_failure", "covers": {"inspect"}, "deps": ["run_tests"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "patch_code", "covers": {"patch"}, "deps": ["inspect_failure"], "executable": True, "acceptance": True, "risk": "medium", "confirmed": True, "status": "done", "recovered": True},
            {"id": "rerun_tests", "covers": {"verify"}, "deps": ["patch_code"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "summarize", "covers": {"summarize"}, "deps": ["rerun_tests"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "missing_verify",
        "requirements": {"inspect", "patch", "verify", "summarize"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "inspect", "covers": {"inspect"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "patch", "covers": {"patch"}, "deps": ["inspect"], "executable": True, "acceptance": True, "risk": "medium", "confirmed": True, "status": "done", "recovered": True},
            {"id": "summarize", "covers": {"summarize"}, "deps": ["patch"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "deploy_dependency_violation",
        "requirements": {"collect", "evaluate", "deploy"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "deploy", "covers": {"deploy"}, "deps": ["evaluate"], "executable": True, "acceptance": True, "risk": "high", "confirmed": False, "status": "blocked", "recovered": False},
            {"id": "collect_metrics", "covers": {"collect"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "evaluate", "covers": {"evaluate"}, "deps": ["collect_metrics"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "replan_success",
        "requirements": {"find_policy", "answer"},
        "replanned": True,
        "final_success": True,
        "steps": [
            {"id": "search_docs", "covers": {"find_policy"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "failed", "recovered": True},
            {"id": "alternate_search", "covers": {"find_policy"}, "deps": [], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "answer", "covers": {"answer"}, "deps": ["alternate_search"], "executable": True, "acceptance": True, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
    {
        "id": "over_decomposed_loop",
        "requirements": {"diagnose", "answer"},
        "replanned": False,
        "final_success": False,
        "steps": [
            {"id": "think_1", "covers": set(), "deps": [], "executable": False, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "think_2", "covers": set(), "deps": ["think_1"], "executable": False, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "search_logs", "covers": {"diagnose"}, "deps": ["think_2"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "search_logs_again", "covers": {"diagnose"}, "deps": ["search_logs"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "answer_without_evidence", "covers": {"answer"}, "deps": ["search_logs_again"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
            {"id": "extra_check", "covers": set(), "deps": ["answer_without_evidence"], "executable": True, "acceptance": False, "risk": "low", "confirmed": True, "status": "done", "recovered": True},
        ],
    },
]


def rate(num, den):
    return round(num / den, 3) if den else 1.0


def dependency_violations(plan):
    done_or_seen = set()
    violations = []
    for step in plan["steps"]:
        for dep in step["deps"]:
            if dep not in done_or_seen:
                violations.append((step["id"], dep))
        done_or_seen.add(step["id"])
    return violations


def repeated_cover(plan):
    seen = set()
    repeats = 0
    for step in plan["steps"]:
        cover_key = tuple(sorted(step["covers"]))
        if cover_key and cover_key in seen:
            repeats += 1
        if cover_key:
            seen.add(cover_key)
    return repeats > 0


def critical_path_len(plan):
    by_id = {step["id"]: step for step in plan["steps"]}
    memo = {}

    def depth(step_id):
        if step_id in memo:
            return memo[step_id]
        step = by_id[step_id]
        if not step["deps"]:
            memo[step_id] = 1
        else:
            memo[step_id] = 1 + max(
                (depth(dep) for dep in step["deps"] if dep in by_id),
                default=0,
            )
        return memo[step_id]

    return max((depth(step["id"]) for step in plan["steps"]), default=0)


def audit_plans(plans):
    all_steps = [step for plan in plans for step in plan["steps"]]
    coverage_scores = []
    failed_or_blocked_plans = []
    dep_violation_plans = []
    overlong_plans = []
    repeated_plans = []
    missing_acceptance_plans = []
    for plan in plans:
        covered = set().union(*(step["covers"] for step in plan["steps"])) if plan["steps"] else set()
        coverage_scores.append(rate(len(covered & plan["requirements"]), len(plan["requirements"])))
        if any(step["status"] in {"failed", "blocked"} for step in plan["steps"]):
            failed_or_blocked_plans.append(plan)
        if dependency_violations(plan):
            dep_violation_plans.append(plan["id"])
        if len(plan["steps"]) > MAX_STEPS:
            overlong_plans.append(plan["id"])
        if repeated_cover(plan):
            repeated_plans.append(plan["id"])
        if any(not step["acceptance"] for step in plan["steps"]):
            missing_acceptance_plans.append(plan["id"])

    high_risk_steps = [s for s in all_steps if s["risk"] == "high"]
    failed_or_blocked_steps = [s for s in all_steps if s["status"] in {"failed", "blocked"}]
    metrics = {
        "goal_coverage": round(sum(coverage_scores) / len(coverage_scores), 3),
        "executable_step_rate": rate(sum(s["executable"] for s in all_steps), len(all_steps)),
        "acceptance_coverage": rate(sum(s["acceptance"] for s in all_steps), len(all_steps)),
        "dependency_violation_rate": rate(len(dep_violation_plans), len(plans)),
        "replan_coverage": rate(sum(p["replanned"] for p in failed_or_blocked_plans), len(failed_or_blocked_plans)),
        "failure_recovery_rate": rate(sum(s["recovered"] for s in failed_or_blocked_steps), len(failed_or_blocked_steps)),
        "risk_confirmation_coverage": rate(sum(s["confirmed"] for s in high_risk_steps), len(high_risk_steps)),
        "overlong_plan_rate": rate(len(overlong_plans), len(plans)),
        "repeat_subgoal_rate": rate(len(repeated_plans), len(plans)),
        "task_success_rate": rate(sum(p["final_success"] for p in plans), len(plans)),
        "max_critical_path_len": max(critical_path_len(p) for p in plans),
    }
    reasons = Counter()
    for plan in plans:
        if plan["id"] in dep_violation_plans:
            reasons["dependency_violation"] += 1
        if plan["id"] in overlong_plans:
            reasons["overlong_plan"] += 1
        if plan["id"] in repeated_plans:
            reasons["repeated_subgoal"] += 1
        if plan["id"] in missing_acceptance_plans:
            reasons["missing_acceptance"] += 1
        if not plan["final_success"]:
            reasons["task_failed"] += 1
        if any(s["risk"] == "high" and not s["confirmed"] for s in plan["steps"]):
            reasons["unconfirmed_high_risk"] += 1
        covered = set().union(*(step["covers"] for step in plan["steps"])) if plan["steps"] else set()
        if covered != plan["requirements"]:
            reasons["goal_not_fully_covered"] += 1
    gates = {
        "coverage_ok": metrics["goal_coverage"] >= 0.90,
        "executable_ok": metrics["executable_step_rate"] >= 0.90,
        "acceptance_ok": metrics["acceptance_coverage"] >= 0.90,
        "dependency_ok": metrics["dependency_violation_rate"] == 0.0,
        "replan_ok": metrics["replan_coverage"] >= 0.50,
        "recovery_ok": metrics["failure_recovery_rate"] >= 0.80,
        "risk_ok": metrics["risk_confirmation_coverage"] == 1.0,
        "length_ok": metrics["overlong_plan_rate"] == 0.0,
        "repeat_ok": metrics["repeat_subgoal_rate"] == 0.0,
        "success_ok": metrics["task_success_rate"] >= 0.75,
    }
    return {
        "metrics": metrics,
        "dependency_violations": {
            p["id"]: dependency_violations(p) for p in plans if dependency_violations(p)
        },
        "problem_plans": sorted(
            set(
                dep_violation_plans
                + overlong_plans
                + repeated_plans
                + missing_acceptance_plans
                + [p["id"] for p in plans if not p["final_success"]]
            )
        ),
        "top_failure_reasons": reasons.most_common(),
        "gates": gates,
        "gate_pass": all(gates.values()),
    }


report = audit_plans(PLANS)
print("metrics=", report["metrics"])
print("dependency_violations=", report["dependency_violations"])
print("problem_plans=", report["problem_plans"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("gates=", report["gates"])
print("gate_pass=", report["gate_pass"])
```

一组预期输出：

```text
metrics= {'goal_coverage': 0.95, 'executable_step_rate': 0.9, 'acceptance_coverage': 0.65, 'dependency_violation_rate': 0.2, 'replan_coverage': 0.5, 'failure_recovery_rate': 0.5, 'risk_confirmation_coverage': 0.0, 'overlong_plan_rate': 0.2, 'repeat_subgoal_rate': 0.4, 'task_success_rate': 0.4, 'max_critical_path_len': 6}
dependency_violations= {'deploy_dependency_violation': [('deploy', 'evaluate')]}
problem_plans= ['deploy_dependency_violation', 'missing_verify', 'over_decomposed_loop', 'replan_success']
top_failure_reasons= [('task_failed', 3), ('missing_acceptance', 2), ('repeated_subgoal', 2), ('goal_not_fully_covered', 1), ('dependency_violation', 1), ('unconfirmed_high_risk', 1), ('overlong_plan', 1)]
gates= {'coverage_ok': True, 'executable_ok': True, 'acceptance_ok': False, 'dependency_ok': False, 'replan_ok': True, 'recovery_ok': False, 'risk_ok': False, 'length_ok': False, 'repeat_ok': False, 'success_ok': False}
gate_pass= False
```

这里 `gate_pass=False` 不是 demo 出错，而是为了暴露几个真实 planning 风险：

1. `missing_verify` 没有覆盖验证要求，且总结步骤缺少验收标准。
2. `deploy_dependency_violation` 在 evaluate 之前尝试 deploy，且高风险步骤没有确认。
3. `over_decomposed_loop` 过度分解、重复覆盖同一子目标，关键路径过长。
4. `replan_success` 虽然最终成功，但因为有失败分支和重复覆盖，仍应在审计报告中显式记录。

## 4.16 常见失败模式

1. 计划脱离用户目标。
2. 分解过粗，无法执行。
3. 分解过细，成本过高。
4. 忽略依赖关系。
5. 不根据 observation 更新。
6. 失败后只会重复。
7. 长期任务状态丢失。
8. 没有明确停止条件。
9. 高风险步骤缺少确认。
10. 只追求完成步骤，不验证结果。
11. 并行任务结果冲突但没有合并策略。
12. 计划文本看起来专业，但没有对应 action 和验收。

Planning 的难点是让计划真正服务执行，而不是生成一段看起来专业的清单。

## 4.17 面试题：Agent 如何做任务分解

回答要点：

```text
我会先明确目标、约束和完成标准，然后把任务拆成可执行、可验证的子目标。每个子目标要有输入输出、验收标准和风险级别，再根据依赖关系安排顺序。执行过程中，根据工具 observation 更新计划。如果子任务失败，系统要能重试、换工具、请求用户补充、降级或人工接管。任务分解不是静态清单，而是和执行反馈结合的动态过程。
```

专家追问：如何判断子任务粒度是否合适？

```text
一个好子任务应该能通过一个工具调用或一组紧密相关动作推进，并且能验证是否完成。太粗会不可执行，太细会增加成本和僵化程度。实践中我会看 executable step rate、acceptance coverage、critical path length 和重复子目标率。
```

## 4.18 面试题：一次性规划和动态规划如何取舍

回答要点：

```text
一次性规划适合流程稳定、目标明确、需要用户确认的任务，优点是结构清晰、成本可控。动态规划适合信息不完整、环境反馈强、执行结果决定下一步的任务，例如调试、搜索和浏览器操作。实际系统通常先生成粗计划，再在每一步根据 observation 修正。关键是既要有全局方向，又不能僵化执行错误计划。
```

专家追问：动态重规划如何避免失控？

```text
需要设置 step limit、budget、重复动作拦截、plan update reason、stop condition 和人工接管。每次重规划都要说明是因为失败、目标变更、权限限制、预算变化还是新证据，而不是随意改方向。
```

## 4.19 小练习

1. 把“修复一个失败测试并补充回归测试”拆成子任务、依赖图和验收标准。
2. 给一个计划，标出哪些子任务不可执行、哪些缺少验收标准。
3. 构造一个依赖顺序错误的计划，并写出正确拓扑顺序。
4. 设计一个 planning audit 表，包含 goal coverage、acceptance coverage、dependency violation、replan coverage、risk confirmation 和 task success。
5. 用本章 demo 增加一个“并行收集资料并合并冲突结果”的计划。
6. 构造一个高风险部署任务，要求写出审批、回滚和停止条件。
7. 说明为什么“计划很详细”不等于“计划质量高”。
8. 用 3 分钟回答“Agent 如何做任务分解，以及如何评估规划质量”。

## 4.20 本章小结

Planning 与 task decomposition 让 Agent 能处理复杂任务。好的规划不是把任务拆成一堆步骤，而是明确子目标、依赖关系、执行顺序、验收标准、失败恢复、风险控制和停止条件。计划也不是一次生成后不可改变，而是要随着 observation 动态修正。

下一章会进入 memory 系统，讨论 Agent 如何保存短期任务状态、长期用户偏好、历史经验和外部知识，以及如何避免 memory 污染和隐私风险。
