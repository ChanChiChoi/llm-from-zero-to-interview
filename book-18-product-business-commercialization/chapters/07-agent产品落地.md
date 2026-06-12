# 第七章：Agent 产品落地

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修。联网资料主要核对五类口径：OpenAI Agents SDK 的 tools、handoffs、guardrails 和 tracing 资料提醒我们，Agent 产品要把工具、轨迹、护栏和交接显式工程化；OpenAI Evals 资料提醒我们，Agent 不能只看最终回答，要用可复现评估和 trace 评估任务成功、工具选择、参数、恢复和安全；OpenAI Model Spec 的指令层级口径提醒我们，工具输出、网页、RAG 文档和用户输入不能越过系统 / 开发者指令；OWASP LLM Top 10 中 excessive agency、prompt injection、sensitive information disclosure 和 unbounded consumption 风险提醒我们，能执行动作的 Agent 必须有权限、预算、确认和审计；Anthropic 关于 workflows 与 agents 的公开工程经验提醒我们，生产落地不应迷信“越自主越好”，稳定 workflow 和受控 Agent 往往需要组合。

本章不替代第十七册 Agent 原理、工具协议、安全评估和多 Agent 章节，也不展开具体框架 API。这里聚焦产品落地：怎么把 Agent 从“会调用工具的 demo”升级为可控、可验证、可审计、可预算、可人审、可进入企业工作流的任务执行产品。

Agent 产品和普通大模型问答产品最大的区别，是 Agent 会执行动作。它可能查询系统、修改文件、提交表单、调用 API、操作浏览器、运行代码或触发业务流程。因此 Agent 产品化的核心不是让模型“更自主”，而是让它在受控范围内可靠执行任务。

本章系统讲 Agent 产品落地：适合 Agent 的任务、工具权限、人工确认、失败恢复、审计日志、成本控制、用户体验、评估指标和常见失败模式。

## 7.1 Agent 产品的本质

Agent 产品不是聊天框加几个工具，而是任务执行系统。

它通常包含：

1. 用户目标。
2. 任务分解。
3. 工具选择。
4. 执行动作。
5. 观察反馈。
6. 状态更新。
7. 错误恢复。
8. 最终交付。

面试回答：

```text
Agent 产品的本质是让模型在受控环境中完成多步任务，而不是只生成回答。落地时要重点设计任务边界、工具权限、状态管理、失败恢复、人工确认、审计日志和评估指标。Agent 越能执行真实动作，越需要产品和安全设计，而不是只提升模型自主性。
```

## 7.2 哪些任务适合 Agent

适合 Agent 的任务通常满足：

1. 需要多步操作。
2. 需要调用工具。
3. 中间结果会影响下一步。
4. 有明确完成标准。
5. 可以验证结果。
6. 人工操作重复且耗时。
7. 失败风险可控。

例子：

1. 自动整理工单。
2. 代码修复和测试。
3. 数据报表生成。
4. 企业知识检索和汇总。
5. 浏览器后台操作。
6. 运维排查辅助。

不适合的任务：目标模糊、风险不可控、没有验收标准、强实时且低延迟、涉及不可逆高风险动作。

## 7.3 Agent 产品不要一开始全自动

很多 Agent 产品适合从辅助模式开始。

成熟路径：

1. 建议模式：Agent 给出建议，不执行。
2. 草稿模式：Agent 生成草稿，用户确认。
3. 半自动模式：Agent 执行低风险步骤，高风险步骤确认。
4. 自动模式：在边界清楚、验证充分后自动执行。

一开始就全自动，容易因为失败案例损害用户信任。先做人机协同，能更快收集数据和建立信任。

产品设计上可以把自动化层级写清楚：

1. `suggest`：只给建议，不执行动作。
2. `draft`：生成草稿或操作计划，用户确认后执行。
3. `semi_auto`：低风险步骤自动执行，高风险步骤确认。
4. `approval`：Agent 准备完整动作包，由负责人审批。
5. `auto`：仅在边界清楚、回归评估稳定、权限和回滚充分时使用。

面试中要强调：自动化层级是产品决策，不是模型能力越强就越自动。高风险任务如果没有验收器、人审、审计和回滚，就应该停留在 copilot 或 approval 形态。

## 7.4 工具权限设计

Agent 产品必须设计工具权限。

工具可以分层：

1. 只读工具：搜索、查询、读取文档。
2. 低风险写入：生成草稿、创建临时文件。
3. 中风险操作：修改内部记录、提交工单。
4. 高风险操作：删除、支付、发邮件、改权限。

权限原则：

1. 默认最小权限。
2. 按任务授权。
3. 按用户身份授权。
4. 高风险动作确认。
5. 工具层强校验。
6. 记录审计日志。

不要把权限控制写在 prompt 里就结束。系统层必须强制执行。

工具权限至少要有三层：

1. 用户权限：当前用户是否有权访问这个资源或执行这个动作。
2. 任务权限：当前任务是否真的需要这个工具和这个动作范围。
3. 风险权限：动作是否外发、删除、改权限、改生产数据或产生费用。

如果一个 Agent 拿到“全量工具 + 全量数据 + 自由执行”的权限，哪怕模型大多数时候表现很好，也不是生产级设计。生产系统应把工具执行器、权限检查器、参数校验器和审计系统放在模型之外。

## 7.5 人工确认

人工确认是 Agent 产品的安全阀。

需要确认的动作：

1. 对外发送内容。
2. 删除或覆盖数据。
3. 修改权限。
4. 提交审批。
5. 产生费用。
6. 影响客户或生产系统。

确认界面要展示：

1. Agent 准备做什么。
2. 影响对象是谁。
3. 参数是什么。
4. 为什么要做。
5. 可能风险是什么。
6. 如何取消。

如果确认信息不清楚，用户只是盲点“同意”，风险并没有降低。

## 7.6 失败恢复

Agent 产品一定会遇到失败。

失败类型：

1. 工具不可用。
2. 参数错误。
3. 权限不足。
4. 信息不完整。
5. 任务目标冲突。
6. 外部系统状态变化。
7. 执行结果不符合预期。

恢复策略：

1. 重试。
2. 更换工具。
3. 请求用户补充信息。
4. 降级为人工处理。
5. 回滚或撤销。
6. 输出部分完成结果。
7. 停止并说明原因。

成熟 Agent 产品不要求永远成功，但必须失败得可控、可解释、可恢复。

## 7.7 状态管理

Agent 执行任务需要维护状态。

状态包括：

1. 用户目标。
2. 当前计划。
3. 已完成步骤。
4. 已调用工具。
5. 工具结果。
6. 失败尝试。
7. 待确认动作。
8. 剩余预算。

没有状态管理，Agent 容易重复操作、忘记约束、误以为任务已完成。

## 7.8 审计日志

Agent 产品必须有 trace 和审计。

要记录：

1. 用户输入。
2. 计划。
3. 每一步 action。
4. 工具参数。
5. 工具返回摘要。
6. 权限检查。
7. 用户确认。
8. 错误和重试。
9. 最终结果。

审计日志用于排查问题、评估效果、安全合规和责任追踪。日志要脱敏，避免成为新的泄露源。

## 7.9 体验设计

Agent 产品体验不能只有“正在执行”。

用户需要看到：

1. Agent 计划做什么。
2. 当前执行到哪一步。
3. 哪些步骤已完成。
4. 哪些步骤失败。
5. 需要用户确认什么。
6. 最终结果和证据。

对长任务，进度反馈非常重要。用户不需要看完整内部推理，但需要知道系统在做什么，是否卡住，是否需要介入。

## 7.10 成本控制

Agent 通常成本高。

成本来自：

1. 多轮模型调用。
2. 工具调用。
3. 检索。
4. 浏览器操作。
5. 代码执行。
6. 失败重试。
7. 人工确认。
8. 日志存储。

控制方式：

1. 限制最大步骤。
2. 限制工具调用次数。
3. 简单任务不用 Agent。
4. 对低价值任务降级。
5. 缓存可复用结果。
6. 明确停止条件。

Agent 产品需要 budget manager，否则容易因为少数复杂任务消耗大量资源。

## 7.11 Agent 与 Workflow 结合

生产系统常把 Agent 嵌入 workflow。

例如工单处理：

```text
接收工单 -> 分类 -> Agent 补充信息 -> 人工确认 -> 提交处理 -> 记录结果
```

Workflow 负责稳定流程，Agent 负责复杂判断和生成。这样比让 Agent 完全自由行动更可控。

## 7.12 Agent 产品评估

评估指标：

1. 任务完成率。
2. 平均步骤数。
3. 工具调用成功率。
4. 错误恢复成功率。
5. 人工确认比例。
6. 人工接管比例。
7. 平均成本。
8. 平均延迟。
9. 安全违规率。
10. 用户满意度。

不要只看最终回答。Agent 的价值和风险都在执行轨迹里。

## 7.12.1 关键公式与 Agent 产品指标速查

可以把一次 Agent 产品任务样本写成：

```math
a_i=(g_i,u_i,\ell_i,T_i,s_i,A_i,O_i,H_i,C_i,L_i,B_i)
```

其中 `g_i` 是用户目标，`u_i` 是用户和角色，`\ell_i` 是自动化层级，`T_i` 是可用工具集合，`s_i` 是结构化状态，`A_i` 是动作序列，`O_i` 是观察结果，`H_i` 是人工确认记录，`C_i` 是成本，`L_i` 是延迟，`B_i` 是业务结果。

**1. 任务成功率**

Agent 产品首先要完成任务，而不是只生成看似合理的过程：

```math
R_{\mathrm{task}}=\frac{1}{N}\sum_{i=1}^{N}y_i
```

其中 `y_i=1` 表示第 `i` 个任务满足验收标准，例如工单被正确补全、代码测试通过、报表生成并被采纳。没有验收标准的 Agent，很难评估是否真的有价值。

**2. 工具执行成功率**

工具调用要看选择、参数、权限和执行结果：

```math
R_{\mathrm{tool}}=\frac{1}{M}\sum_{j=1}^{M}e_j
```

其中 `M` 是工具调用总数，`e_j=1` 表示第 `j` 次工具调用工具选择正确、参数合法、权限通过且执行成功。工具调用失败会带来重试、延迟、成本和用户不信任。

**3. 高风险确认覆盖率**

对外发送、删除、支付、改权限和生产变更等动作必须看确认覆盖：

```math
C_{\mathrm{conf}}=\frac{\sum_{j=1}^{M}z_j h_j}{\sum_{j=1}^{M}z_j+\epsilon}
```

其中 `z_j=1` 表示第 `j` 个动作是高风险动作，`h_j=1` 表示有清晰的用户确认、审批或二次验证。确认界面必须展示动作、对象、参数、原因和风险，否则用户只是盲点同意。

**4. 未授权动作率**

Agent 越能行动，越要控制越权：

```math
R_{\mathrm{unauth}}=\frac{N_{\mathrm{unauth}}}{M+\epsilon}
```

其中 `N_{\mathrm{unauth}}` 是未授权工具调用、越权数据访问、越权外发或越权修改事件数。严重越权事件不能被平均任务成功率掩盖。

**5. 失败恢复率**

成熟 Agent 产品不要求永远不失败，但要求失败可控：

```math
R_{\mathrm{rec}}=\frac{N_{\mathrm{rec\_ok}}}{N_{\mathrm{rec\_need}}+\epsilon}
```

其中 `N_{\mathrm{rec_need}}` 是需要恢复的失败次数，`N_{\mathrm{rec_ok}}` 是通过重试、换工具、澄清、降级、回滚或人工接管成功恢复的次数。

**6. 状态更新覆盖率与观察使用率**

Agent 的执行质量在 trace 里：

```math
C_{\mathrm{state}}=\frac{N_{\mathrm{state\_ok}}}{N_{\mathrm{state\_need}}+\epsilon},\qquad
R_{\mathrm{obs}}=\frac{N_{\mathrm{obs\_used}}}{N_{\mathrm{obs}}+\epsilon}
```

状态更新覆盖率低，说明 Agent 可能忘记约束、重复动作或误判完成；观察使用率低，说明它拿到了工具结果却没有真正纳入下一步决策。

**7. 预算超限率**

Agent 产品必须有预算管理：

```math
R_{\mathrm{budget}}=\frac{N_{\mathrm{over}}}{N+\epsilon}
```

其中 `N_{\mathrm{over}}` 是超过步数、工具调用、token、时间或费用预算的任务数。复杂 Agent 如果没有停止条件，少数长尾任务会吞掉大量资源。

**8. Agent 产品上线门禁**

一个简化上线门禁可以写成：

```math
G_{\mathrm{agentprod}}=\mathbf{1}[R_{\mathrm{task}}\geq r_0]\mathbf{1}[R_{\mathrm{tool}}\geq t_0]\mathbf{1}[C_{\mathrm{conf}}\geq c_0]\mathbf{1}[R_{\mathrm{unauth}}=0]\mathbf{1}[R_{\mathrm{budget}}\leq b_0]\mathbf{1}[L_{95}\leq L_{\mathrm{slo}}]
```

真实项目还要加入 trace 完整率、恢复率、人工接管率、成本、评估覆盖、反馈闭环和业务指标。Agent 产品不能只用“最终答案看起来对”做上线判断。

## 7.13 适合先落地的 Agent 场景

更适合作为早期落地：

1. 内部工具。
2. 低风险任务。
3. 可验证结果。
4. 有人工确认。
5. 数据权限清晰。
6. 用户愿意试用。

例如内部代码助手、工单摘要、知识检索、报表草稿，比直接让 Agent 自动处理客户资金或生产权限更适合早期试点。

## 7.14 常见失败模式

1. 任务边界不清。
2. 工具权限过大。
3. 没有人工确认。
4. 出错后无限重试。
5. 用户不知道 Agent 在做什么。
6. 没有审计日志。
7. 成本失控。
8. 最终结果不可验证。
9. 高风险动作自动执行。
10. 把 demo 当生产系统。

Agent 产品失败通常不是因为模型不会思考，而是因为系统没有控制行动边界。

## 7.15 面试题：Agent 产品落地最重要的是什么

回答要点：

```text
Agent 产品落地最重要的是控制任务边界和行动风险。Agent 能执行工具和真实动作，所以要明确适用场景、工具权限、状态管理、人工确认、失败恢复和审计日志。早期不建议直接全自动，应从低风险、可验证、人机协同的任务开始，用 trace 和评估指标持续改进。
```

## 7.16 面试题：如何设计一个企业 Agent 产品

回答要点：

```text
我会先选择高价值但风险可控的任务，定义完成标准。系统上设计 planner、tool registry、executor、state manager、permission checker、human confirmation、logger 和 evaluator。只给 Agent 当前任务需要的最小工具权限，高风险动作必须确认。上线后评估任务完成率、工具调用成功率、错误恢复、人工接管、安全违规、成本和用户满意度。
```

## 7.17 最小可运行 Agent 产品审计 demo

下面这个 demo 用 0 依赖 Python 模拟 Agent 产品上线审计。它把任务成功、工具执行、高风险确认、失败恢复、状态更新、观察使用、trace 覆盖、越权动作、预算超限、延迟、单位成本、评估集、反馈闭环和业务指标放进同一张表。

```python
def ratio(num, den):
    return num / den if den else 0.0


def audit_agent_product(agent):
    task_success = ratio(agent["tasks_completed"], agent["tasks_total"])
    tool_success = ratio(agent["tool_successes"], agent["tool_calls"])
    confirmation_coverage = (
        ratio(agent["high_risk_confirmed"], agent["high_risk_actions"])
        if agent["high_risk_actions"]
        else 1.0
    )
    recovery_rate = (
        ratio(agent["recoveries_succeeded"], agent["recoveries_needed"])
        if agent["recoveries_needed"]
        else 1.0
    )
    state_update_coverage = ratio(agent["state_updates"], agent["state_update_required"])
    observation_use_rate = ratio(agent["observations_used"], agent["observations_total"])
    trace_coverage = ratio(agent["trace_complete"], agent["tasks_total"])
    unauthorized_rate = ratio(agent["unauthorized_actions"], agent["tool_calls"])
    budget_overrun_rate = ratio(agent["budget_overruns"], agent["tasks_total"])
    gates = {
        "task_success": task_success >= 0.75,
        "tool_success": tool_success >= 0.85,
        "high_risk_confirmation": confirmation_coverage >= 0.90,
        "recovery": recovery_rate >= 0.60,
        "state_update": state_update_coverage >= 0.80,
        "observation_use": observation_use_rate >= 0.80,
        "trace_coverage": trace_coverage >= 0.90,
        "unauthorized_action": unauthorized_rate == 0.0,
        "budget_overrun": budget_overrun_rate <= 0.10,
        "p95_latency": agent["p95_latency_ms"] <= agent["latency_slo_ms"],
        "unit_cost": agent["cost_per_task"] <= agent["cost_slo"],
        "eval_ready": agent["eval_ready"] >= 0.80,
        "feedback_loop": agent["feedback_loop"] >= 0.75,
        "business_metric": agent["business_metric_defined"] >= 1.0,
    }
    agent_score = (
        0.18 * task_success
        + 0.12 * tool_success
        + 0.10 * confirmation_coverage
        + 0.10 * recovery_rate
        + 0.10 * state_update_coverage
        + 0.08 * observation_use_rate
        + 0.08 * trace_coverage
        + 0.08 * (1.0 - min(unauthorized_rate, 1.0))
        + 0.06 * (1.0 - min(budget_overrun_rate, 1.0))
        + 0.04 * (1.0 if gates["p95_latency"] else 0.0)
        + 0.03 * agent["eval_ready"]
        + 0.03 * agent["feedback_loop"]
    )
    return {
        "name": agent["name"],
        "mode": agent["mode"],
        "agent_score": round(agent_score, 3),
        "agent_gate": all(gates.values()),
        "metrics": {
            "task_success": round(task_success, 3),
            "tool_success": round(tool_success, 3),
            "confirmation_coverage": round(confirmation_coverage, 3),
            "recovery_rate": round(recovery_rate, 3),
            "unauthorized_rate": round(unauthorized_rate, 3),
            "budget_overrun_rate": round(budget_overrun_rate, 3),
        },
        "failed_gates": [name for name, ok in gates.items() if not ok],
    }


agents = [
    {
        "name": "support_ticket_agent",
        "mode": "semi_auto",
        "tasks_total": 120,
        "tasks_completed": 101,
        "tool_calls": 260,
        "tool_successes": 238,
        "high_risk_actions": 18,
        "high_risk_confirmed": 18,
        "recoveries_needed": 22,
        "recoveries_succeeded": 16,
        "state_update_required": 240,
        "state_updates": 218,
        "observations_total": 260,
        "observations_used": 236,
        "trace_complete": 116,
        "unauthorized_actions": 0,
        "budget_overruns": 8,
        "p95_latency_ms": 4200,
        "latency_slo_ms": 5000,
        "cost_per_task": 0.18,
        "cost_slo": 0.25,
        "eval_ready": 0.86,
        "feedback_loop": 0.80,
        "business_metric_defined": 1.0,
    },
    {
        "name": "code_fix_agent",
        "mode": "copilot",
        "tasks_total": 70,
        "tasks_completed": 48,
        "tool_calls": 190,
        "tool_successes": 168,
        "high_risk_actions": 4,
        "high_risk_confirmed": 4,
        "recoveries_needed": 28,
        "recoveries_succeeded": 15,
        "state_update_required": 150,
        "state_updates": 126,
        "observations_total": 190,
        "observations_used": 151,
        "trace_complete": 66,
        "unauthorized_actions": 0,
        "budget_overruns": 11,
        "p95_latency_ms": 7600,
        "latency_slo_ms": 8000,
        "cost_per_task": 0.42,
        "cost_slo": 0.45,
        "eval_ready": 0.78,
        "feedback_loop": 0.70,
        "business_metric_defined": 1.0,
    },
    {
        "name": "data_ops_agent",
        "mode": "approval",
        "tasks_total": 80,
        "tasks_completed": 55,
        "tool_calls": 210,
        "tool_successes": 171,
        "high_risk_actions": 24,
        "high_risk_confirmed": 18,
        "recoveries_needed": 30,
        "recoveries_succeeded": 17,
        "state_update_required": 190,
        "state_updates": 139,
        "observations_total": 210,
        "observations_used": 149,
        "trace_complete": 65,
        "unauthorized_actions": 2,
        "budget_overruns": 13,
        "p95_latency_ms": 9100,
        "latency_slo_ms": 6500,
        "cost_per_task": 0.31,
        "cost_slo": 0.30,
        "eval_ready": 0.75,
        "feedback_loop": 0.58,
        "business_metric_defined": 1.0,
    },
    {
        "name": "generic_browser_agent",
        "mode": "auto",
        "tasks_total": 60,
        "tasks_completed": 29,
        "tool_calls": 240,
        "tool_successes": 150,
        "high_risk_actions": 30,
        "high_risk_confirmed": 9,
        "recoveries_needed": 35,
        "recoveries_succeeded": 8,
        "state_update_required": 180,
        "state_updates": 80,
        "observations_total": 240,
        "observations_used": 96,
        "trace_complete": 36,
        "unauthorized_actions": 7,
        "budget_overruns": 21,
        "p95_latency_ms": 12500,
        "latency_slo_ms": 7000,
        "cost_per_task": 0.62,
        "cost_slo": 0.28,
        "eval_ready": 0.40,
        "feedback_loop": 0.20,
        "business_metric_defined": 0.0,
    },
]

results = [audit_agent_product(agent) for agent in agents]
ranked = sorted(
    [(r["name"], r["agent_score"], r["agent_gate"]) for r in results],
    key=lambda item: item[1],
    reverse=True,
)
agent_pass = [r["name"] for r in results if r["agent_gate"]]
needs_rework = {
    r["name"]: r["failed_gates"]
    for r in results
    if not r["agent_gate"]
}

print("ranked=", ranked)
print("agent_pass=", agent_pass)
print("sample_metrics=", results[0]["metrics"])
print("needs_rework=", needs_rework)
```

一组典型输出是：

```text
ranked= [('support_ticket_agent', 0.901, True), ('code_fix_agent', 0.821, False), ('data_ops_agent', 0.717, False), ('generic_browser_agent', 0.474, False)]
agent_pass= ['support_ticket_agent']
sample_metrics= {'task_success': 0.842, 'tool_success': 0.915, 'confirmation_coverage': 1.0, 'recovery_rate': 0.727, 'unauthorized_rate': 0.0, 'budget_overrun_rate': 0.067}
needs_rework= {'code_fix_agent': ['task_success', 'recovery', 'observation_use', 'budget_overrun', 'eval_ready', 'feedback_loop'], 'data_ops_agent': ['task_success', 'tool_success', 'high_risk_confirmation', 'recovery', 'state_update', 'observation_use', 'trace_coverage', 'unauthorized_action', 'budget_overrun', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop'], 'generic_browser_agent': ['task_success', 'tool_success', 'high_risk_confirmation', 'recovery', 'state_update', 'observation_use', 'trace_coverage', 'unauthorized_action', 'budget_overrun', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop', 'business_metric']}
```

这个 demo 的重点是：Agent 产品不是越自动越好。`support_ticket_agent` 通过门禁，是因为它是半自动、任务边界清楚、确认覆盖和 trace 稳定；`code_fix_agent` 适合继续做 copilot，因为任务成功、恢复、预算和反馈还没过线；`data_ops_agent` 虽然有审批形态，但高风险确认、越权、成本和延迟不过线；`generic_browser_agent` 说明通用全自动浏览器 Agent 如果没有强权限、预算、确认和评估，很难进入生产。

## 7.18 本章小结

Agent 产品化的核心是让模型在受控范围内执行任务。它比普通问答更有价值，也更有风险。落地时要优先选择边界清楚、可验证、低风险的任务，从辅助和半自动开始，逐步扩大自动化能力。

下一章会进入多模态产品落地，讨论图像、语音、视频和多模态理解生成能力如何转化为真实产品体验。
