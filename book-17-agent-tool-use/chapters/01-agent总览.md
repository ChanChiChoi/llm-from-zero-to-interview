# 第一章：Agent 总览

Agent 是把大模型从“回答问题”推进到“执行任务”的关键系统形态。普通 chat model 主要根据用户输入生成回复；Agent 还会围绕目标维护状态、选择工具、读取外部反馈、更新任务轨迹，并在预算和权限约束下多步行动。

本章先建立 Agent 的全局框架：Agent 和普通聊天模型、RAG、workflow 的区别是什么；Agent 的核心组成有哪些；目标、状态、动作、工具、观察、控制器分别负责什么；怎样用公式和 trace 描述 Agent；哪些任务适合 Agent，哪些任务不适合；上线前如何用最小审计脚本检查工具、预算、安全和停止条件。

## 0. 本讲资料边界与第二轮精修口径

本讲写作前按 `WRITING_PLAN.md` 的要求核对了 ReAct、Toolformer、MRKL Systems 等代表性论文，以及 OpenAI Agents SDK 中 tools、handoffs、guardrails、tracing 的公开文档口径。这里不把章节写成某个厂商 API 教程，也不讨论具体闭源模型内部实现。

本章采用三条边界：

1. 只讲 Agent 总览层的概念、公式、系统模块、评估指标和安全边界。
2. 工具调用、function calling、MCP、multi-agent、computer use、code agent 会在后续章节展开，本章只铺底层框架。
3. 所有例子只做教学级 trace 审计，不提供高风险真实操作流程；涉及写入、删除、发信、转账等动作时，只讨论权限、二次确认和审计。

第二轮精修重点：

1. 用稳定的 MathJax 公式描述 agent loop、状态更新、预算和评估指标。
2. 把 Agent、Chat Model、RAG、Workflow 的边界讲清楚，避免把所有 LLM 应用都叫 Agent。
3. 补一个 0 依赖 Python demo，帮助读者把“Agent 是否可靠”转成可检查的 trace 指标。

## 1.1 Agent 是什么

一个实用定义：

```text
Agent 是由 LLM 驱动、围绕目标进行多步决策，并通过工具或环境反馈执行任务的受控系统。
```

它通常包含：

1. Goal：要完成的目标和验收标准。
2. State：当前任务进展、约束、失败记录和中间产物。
3. Action：下一步要执行的动作，例如调用工具、追问用户、更新计划或结束任务。
4. Tool：可调用的外部能力，例如搜索、数据库、代码执行、文件读写或业务 API。
5. Observation：工具或环境返回的结果。
6. Policy：决定下一步动作的策略，通常由 LLM、规则、workflow 或它们的组合实现。
7. Memory：保存对话历史、任务日志、用户偏好或项目知识的机制。
8. Controller：控制循环、预算、权限、停止条件、失败恢复和 trace 的工程外壳。

面试回答可以这样说：

```text
Agent 不是单纯让模型回答问题，而是让模型在一个受控循环中围绕目标做多步决策。它会维护状态，选择工具，执行动作，读取观察结果，再决定下一步。相比普通 chat model，Agent 更强调任务执行、外部反馈、状态管理、预算控制、权限边界和失败恢复。
```

## 1.1.1 关键公式与 Agent 指标速查

可以把一次 Agent 执行写成一条轨迹：

```math
\tau=(g,s_0,a_1,o_1,s_1,\ldots,a_K,o_K,s_K,\hat y)
```

其中 `g` 是目标，`s_t` 是第 `t` 步后的状态，`a_t` 是动作，`o_t` 是观察，`K` 是执行步数，`hat y` 是最终输出。

第 `t` 步动作可以抽象为：

```math
a_t \sim \pi_\theta(a\mid g,s_{t-1},H_{t-1},\mathcal{T})
```

其中 `H_{t-1}` 是已有历史，`\mathcal{T}` 是可用工具集合，`\pi_\theta` 是由模型、规则和控制器共同形成的策略。

观察返回后，状态更新为：

```math
s_t=U(s_{t-1},a_t,o_t)
```

Agent 不是无限执行，必须有停止函数：

```math
\phi(s_t,H_t,B_t)\in\{0,1\}
```

当 `\phi=1` 时停止；`B_t` 是预算向量：

```math
B_t=(b_{\mathrm{step}},b_{\mathrm{tok}},b_{\mathrm{cost}},b_{\mathrm{time}},b_{\mathrm{tool}})
```

上下文长度可以粗略拆成：

```math
L_t=L_g+L_s+L_h+L_{\mathrm{tools}}+L_{\mathrm{obs}}
```

其中 `L_g` 是目标描述长度，`L_s` 是状态摘要长度，`L_h` 是历史长度，`L_tools` 是工具 schema 长度，`L_obs` 是观察结果长度。Agent 任务越长，越要管理状态摘要和 observation 截断，否则上下文会被无效历史占满。

常见评估指标：

```math
A_{\mathrm{task}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[z_i=1]
```

`A_task` 是任务成功率，`z_i` 表示第 `i` 个任务是否完成。

```math
A_{\mathrm{tool}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\hat t_j=t_j^*]
```

`A_tool` 是工具选择准确率，`\hat t_j` 是模型选择的工具，`t_j^*` 是期望工具。

```math
A_{\mathrm{arg}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\hat p_j\in\mathcal{P}_j]
```

`A_arg` 是参数合法率，`\hat p_j` 是生成参数，`\mathcal{P}_j` 是合法参数集合。

```math
U_{\mathrm{obs}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[o_j\Rightarrow s_j]
```

`U_obs` 是观察使用率，表示工具结果是否真的影响了后续状态或决策。

```math
R_{\mathrm{budget}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[B_i\ \mathrm{over}]
```

`R_budget` 是预算超限率。

```math
R_{\mathrm{unauth}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[a_j\notin\mathcal{A}_{\mathrm{allow}}]
```

`R_unauth` 是未授权动作率。

一个简化上线门禁可以写成：

```math
G_{\mathrm{agent}}=
\mathbf{1}[A_{\mathrm{task}}\geq q_1]\cdot
\mathbf{1}[A_{\mathrm{tool}}\geq q_2]\cdot
\mathbf{1}[A_{\mathrm{arg}}\geq q_3]\cdot
\mathbf{1}[U_{\mathrm{obs}}\geq q_4]\cdot
\mathbf{1}[R_{\mathrm{budget}}\leq q_5]\cdot
\mathbf{1}[R_{\mathrm{unauth}}=0]
```

面试中不要只说“Agent 能自动完成任务”。更强的表达是：Agent 的能力要落到 trajectory、tool trace、state update、budget、permission gate 和 task success 上衡量。

## 1.2 Agent 和 Chat Model 的区别

Chat model 的典型模式是：

```text
user input -> model response
```

Agent 的典型模式是：

```text
goal -> inspect state -> select action -> execute tool -> observe -> update state -> check stop -> final result
```

核心区别：

1. Chat model 主要生成文本，Agent 会执行动作或触发外部系统。
2. Chat model 通常是一轮或少量多轮，Agent 可能有明确的多步循环。
3. Chat model 的状态主要依赖上下文，Agent 倾向于显式维护结构化状态。
4. Chat model 不一定调用工具，Agent 通常依赖工具、环境和反馈。
5. Chat model 更像问答系统，Agent 更像任务执行系统。
6. Chat model 主要评估答案质量，Agent 还要评估工具轨迹、权限、成本、恢复能力和停止条件。

不要把“能聊天的模型”都叫 Agent。只有系统具备目标驱动、多步决策、外部动作、观察反馈和控制边界时，才更接近 Agent。

## 1.3 Agent 和 RAG 的区别

RAG 的核心是检索增强生成：

```text
query -> retrieve -> rerank -> construct context -> generate answer
```

Agent 可以包含 RAG，但不等于 RAG。Agentic RAG 会让模型主动决定是否检索、检索什么、是否需要二次检索、是否调用其他工具，以及如何根据观察结果继续行动。

对比：

1. RAG 主要解决知识获取和证据 grounding 问题。
2. Agent 主要解决多步任务执行问题。
3. RAG 的主流程通常较固定。
4. Agent 的流程更动态，下一步可能取决于工具返回。
5. RAG 可以是 Agent 的一个工具，也可以是 Agent 的一个子流程。

典型例子：

```text
用户问“公司差旅报销规则是什么”：更像 RAG。
用户说“查规则，检查我的报销单是否合规，不合规就列出修改建议”：更像 Agent。
```

第二个任务不仅要检索，还要读取报销单、比对规则、发现缺口、生成修改建议，必要时还要追问缺失字段。

## 1.4 Agent 和 Workflow 的区别

Workflow 是预定义流程。例如：

```text
分类 -> 检索 -> 总结 -> 格式化输出
```

Agent 则允许模型根据状态动态决定下一步：

```text
如果信息不足 -> 检索
如果需要计算 -> 调用计算工具
如果执行失败 -> 重试或换方案
如果权限不足 -> 请求授权或停止
如果目标完成 -> 停止
```

Workflow 的优点是稳定、可控、容易测试；Agent 的优点是灵活、能处理开放任务。工程上常用混合模式：

1. 外层 workflow 管住关键流程、权限和上线门禁。
2. 局部开放步骤交给 Agent 做动态决策。
3. 高风险动作必须回到规则、人工确认或只读审计。

面试时可以强调：Agent 不是 workflow 的替代品。生产系统经常用 workflow 限制 Agent 的行动空间，用 Agent 处理 workflow 覆盖不到的分支。

## 1.5 Agent 的执行循环

Agent 最经典的循环是：

```text
Think -> Act -> Observe -> Think -> Act -> Observe -> Final
```

工程上更推荐这样描述：

```text
read goal
inspect state
select action
validate action
execute tool
receive observation
update state
check budget and stop condition
repeat or finish
```

这个循环的关键不是让模型“想很多”，而是让模型把外部反馈纳入下一步决策。没有 observation，Agent 只是自说自话；没有 state，Agent 容易忘记任务进展；没有 budget，Agent 容易成本失控；没有 stop condition，Agent 可能无限循环。

一个更完整的控制循环通常包含：

1. Planner：给出下一步计划或动作候选。
2. Tool selector：选择工具。
3. Argument builder：生成参数。
4. Validator：检查 schema、权限、预算和风险。
5. Executor：执行工具。
6. Observation handler：解析工具结果。
7. State updater：更新任务状态。
8. Stop checker：判断是否完成、失败、需要追问或需要人工接管。

## 1.6 Goal：目标

Goal 是 Agent 要完成的任务。好的 goal 应该明确：

1. 要做什么。
2. 完成标准是什么。
3. 允许使用哪些工具。
4. 有什么约束。
5. 什么时候停止。
6. 失败时如何汇报。

不好的 goal 过于模糊：

```text
帮我弄好这个项目。
```

更好的 goal：

```text
运行测试，定位失败原因，修改最小代码使测试通过，并说明修改内容；不要改无关文件；如果需要删除文件或联网下载依赖，先请求确认。
```

目标不清楚会导致工具误用、循环冗长、验收困难和越权风险。Agent 任务越开放，越要把目标拆成可检查的验收条件。

## 1.7 State：状态

State 是 Agent 当前知道的任务进展。

状态可以包含：

1. 用户目标。
2. 已完成步骤。
3. 当前计划。
4. 工具调用结果。
5. 错误和失败尝试。
6. 中间产物。
7. 剩余预算。
8. 安全约束。
9. 待确认事项。

状态管理是 Agent 的核心难点。上下文太少，模型会忘记；上下文太多，模型会被噪声干扰。实际系统通常会结合：

1. 结构化 state：适合记录任务字段、预算、权限和状态机。
2. 短期上下文：适合保留最近对话和关键 observation。
3. 摘要：适合压缩长轨迹。
4. 外部 memory：适合保存长期偏好、项目知识和历史任务。
5. trace log：适合审计、复盘和 replay。

面向专家时要注意：memory 不等于 state。state 服务当前任务闭环，memory 服务跨任务或长周期信息复用。把长期记忆无差别塞回 prompt，容易造成隐私、过期信息和上下文污染。

## 1.8 Action 与 Tool：动作和工具

Action 是 Agent 决定做什么。Tool 是某些 action 的执行接口。

常见 action：

1. 调用工具。
2. 追问用户。
3. 更新计划。
4. 生成最终答案。
5. 放弃并解释原因。
6. 请求人工确认。

常见工具：

1. 搜索工具。
2. 数据库查询。
3. 代码执行。
4. 文件读写。
5. 浏览器操作。
6. API 调用。
7. 计算器。
8. RAG 检索。
9. 日历、邮件、工单系统。

工具让 Agent 能接触外部世界，也带来风险。工具必须有：

1. 清晰 schema。
2. 参数类型和必填字段。
3. 权限边界。
4. 超时和重试策略。
5. 只读 / 写入分级。
6. 高风险动作二次确认。
7. 错误返回格式。
8. 审计日志。

越高权限的工具，越不能只依赖模型“自己判断”。模型可以提出动作，系统必须负责验证动作。

## 1.9 Observation：观察

Observation 是工具或环境返回的信息。

例如：

```text
测试失败：test_login_invalid_password failed
搜索结果：找到三篇相关文档
API 返回：权限不足
浏览器观察：页面出现验证码
```

Agent 必须根据 observation 调整下一步。如果工具失败，它应该判断是参数错、权限错、环境错，还是目标不可完成。

成熟 Agent 的能力不在于从不失败，而在于能读懂失败并恢复。典型恢复策略包括：

1. 参数错误：修正参数后重试。
2. 权限不足：请求授权或停止。
3. 工具超时：降低请求规模或换工具。
4. 证据不足：补充检索或明确说明无法确定。
5. 测试失败：阅读错误日志并定位根因。
6. 目标不可完成：给出已完成部分、失败原因和下一步建议。

## 1.10 Controller：控制器

Controller 是 Agent 系统的工程外壳。它通常管理：

1. 最大步数。
2. 最大输入和输出长度。
3. 最大工具调用次数。
4. 最大成本。
5. 超时。
6. 失败重试。
7. 权限检查。
8. 终止条件。
9. 日志和 trace。
10. 人工接管。

没有 controller 的 Agent 很容易失控：无限循环、重复调用工具、乱改文件、成本不可控、把工具返回当成开发者指令、或者最终答案和执行结果不一致。

工程上，Agent 的可靠性很大程度来自 controller，而不只是模型本身。面试中如果只说“换更强模型”，回答会显得不够工程化；更完整的回答要包括 tool schema、permission gate、budget gate、trace、eval 和 fallback。

## 1.11 Agent 适合什么任务

适合 Agent 的任务通常有这些特点：

1. 需要多步执行。
2. 中途需要外部信息。
3. 需要根据反馈调整计划。
4. 有明确完成标准。
5. 可以通过工具验证结果。
6. 单步 workflow 难以覆盖所有情况。

例如：

1. 代码修复。
2. 数据分析。
3. 复杂检索和报告生成。
4. 自动化运维排查。
5. 浏览器任务。
6. 研究助理。
7. 工单处理。
8. 企业内部 API 助手。

一个经验判断：如果任务的关键价值来自“看到反馈后改变下一步”，Agent 比单次调用更合理。

## 1.12 Agent 不适合什么任务

不适合 Agent 的任务：

1. 简单问答。
2. 低延迟强约束任务。
3. 没有明确完成标准的开放聊天。
4. 高风险不可逆操作。
5. 工具权限无法安全控制的任务。
6. 失败代价很高且无法人工复核的任务。
7. 用固定 workflow 已经稳定解决的任务。

Agent 不是越复杂越好。如果一个任务用一次模型调用、一个分类器或固定 workflow 就能稳定完成，就没有必要上 Agent。

## 1.13 Agent 的常见失败模式

常见失败模式：

1. 目标理解错误。
2. 计划过大或过细。
3. 工具选择错误。
4. 工具参数错误。
5. 忽略 observation。
6. 无限循环。
7. 状态污染。
8. 旧记忆误用。
9. 成本失控。
10. 越权操作。
11. 工具结果注入。
12. 最终答案和执行结果不一致。

这些问题说明，Agent 的难点不是把工具列表塞给模型，而是让系统稳定地规划、执行、观察、修正和停止。

面试中可以把失败归因拆成四层：

1. 模型层：理解、规划、工具选择、参数生成能力不足。
2. 工具层：schema 不清、错误返回不结构化、工具本身不稳定。
3. 状态层：历史过长、状态未更新、记忆污染。
4. 控制层：权限、预算、重试、停止和审计不足。

## 1.14 Agent 评估应该看什么

评估 Agent 不能只看最终文本。至少要看：

1. Task success：任务是否完成。
2. Tool selection：工具是否选对。
3. Argument validity：参数是否完整、合法、语义正确。
4. Execution success：工具是否执行成功。
5. Observation use：是否真正使用工具返回。
6. State update：状态是否更新。
7. Stop correctness：是否该停时停、该继续时继续。
8. Budget：步数、长度、成本、耗时是否可控。
9. Permission：是否发生未授权动作。
10. Trace completeness：日志是否足以复盘。

真实评估通常还要按任务类型切片。例如 coding agent 要看测试通过率、隐藏测试泛化和 patch 最小性；research agent 要看引用准确率、证据覆盖和幻觉率；enterprise agent 要看权限、审计和人工接管。

## 1.15 最小可运行 Agent trace 审计 demo

下面的 demo 不调用真实工具，只审计 toy trace。它展示了 Agent 上线前应如何把“看起来能跑”拆成可量化指标。

输入：

1. 多个任务 trace。
2. 每步的实际工具、期望工具、参数是否合法、是否允许执行、是否使用 observation、是否更新 state。
3. 每个任务的成功、预算、停止条件和 trace 完整性。

输出：

1. 任务成功率。
2. 工具选择准确率。
3. 参数合法率。
4. observation 使用率。
5. state 更新覆盖率。
6. 预算超限率。
7. 未授权动作率。
8. 停止正确率。
9. Agent gate 是否通过。

```python
from collections import Counter


LIMITS = {"steps": 4, "tokens": 4000, "latency_ms": 10000}

TRACES = [
    {
        "id": "repo_fix",
        "goal": "run tests, make the smallest fix, and report the patch",
        "success": True,
        "tokens": 1800,
        "latency_ms": 8200,
        "stop_correct": True,
        "final": "fixed failing parser test",
        "steps": [
            {"tool": "read_file", "expected": "read_file", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "run_tests", "expected": "run_tests", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "edit_file", "expected": "edit_file", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "run_tests", "expected": "run_tests", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "rag_answer",
        "goal": "answer with citations from the policy documents",
        "success": True,
        "tokens": 1250,
        "latency_ms": 3100,
        "stop_correct": True,
        "final": "grounded answer with one citation",
        "steps": [
            {"tool": "search_docs", "expected": "search_docs", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "quote_answer", "expected": "quote_answer", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "calendar_overreach",
        "goal": "check availability, but do not send messages",
        "success": False,
        "tokens": 900,
        "latency_ms": 2100,
        "stop_correct": True,
        "final": "blocked unauthorized send action",
        "steps": [
            {"tool": "email_send", "expected": "calendar_lookup", "args_valid": True, "allowed": False,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "looping_data_task",
        "goal": "compute a small aggregate and stop",
        "success": False,
        "tokens": 5200,
        "latency_ms": 15100,
        "stop_correct": False,
        "final": "kept searching instead of computing",
        "steps": [
            {"tool": "search_docs", "expected": "calculator", "args_valid": False, "allowed": True,
             "observation_used": False, "state_updated": False},
        ],
    },
]


def rate(good, total):
    return round(good / total, 3) if total else 0.0


def trace_complete(trace):
    required_trace_fields = {"id", "goal", "steps", "success", "tokens", "latency_ms", "final"}
    required_step_fields = {"tool", "expected", "args_valid", "allowed", "observation_used", "state_updated"}
    if not required_trace_fields.issubset(trace):
        return False
    return all(required_step_fields.issubset(step) for step in trace["steps"])


def over_budget(trace):
    return (
        len(trace["steps"]) > LIMITS["steps"]
        or trace["tokens"] > LIMITS["tokens"]
        or trace["latency_ms"] > LIMITS["latency_ms"]
    )


def audit_agent_traces(traces):
    all_steps = [step for trace in traces for step in trace["steps"]]
    total_steps = len(all_steps)

    metrics = {
        "task_success_rate": rate(sum(t["success"] for t in traces), len(traces)),
        "tool_selection_accuracy": rate(sum(s["tool"] == s["expected"] for s in all_steps), total_steps),
        "arg_valid_rate": rate(sum(s["args_valid"] for s in all_steps), total_steps),
        "observation_use_rate": rate(sum(s["observation_used"] for s in all_steps), total_steps),
        "state_update_coverage": rate(sum(s["state_updated"] for s in all_steps), total_steps),
        "budget_overrun_rate": rate(sum(over_budget(t) for t in traces), len(traces)),
        "unauthorized_action_rate": rate(sum(not s["allowed"] for s in all_steps), total_steps),
        "stop_correct_rate": rate(sum(t["stop_correct"] for t in traces), len(traces)),
        "trace_completeness": rate(sum(trace_complete(t) for t in traces), len(traces)),
    }

    bad_traces = []
    reasons = {}
    for trace in traces:
        flags = []
        if not trace["success"]:
            flags.append("task_failed")
        if over_budget(trace):
            flags.append("budget_overrun")
        if not trace["stop_correct"]:
            flags.append("bad_stop")
        if any(step["tool"] != step["expected"] for step in trace["steps"]):
            flags.append("wrong_tool")
        if any(not step["allowed"] for step in trace["steps"]):
            flags.append("unauthorized_action")
        if any(not step["observation_used"] for step in trace["steps"]):
            flags.append("observation_ignored")
        if flags:
            bad_traces.append(trace["id"])
            reasons[trace["id"]] = flags

    thresholds = {
        "task_success_rate": 0.75,
        "tool_selection_accuracy": 0.80,
        "arg_valid_rate": 0.95,
        "observation_use_rate": 0.90,
        "state_update_coverage": 0.90,
        "budget_overrun_rate": 0.00,
        "unauthorized_action_rate": 0.00,
        "stop_correct_rate": 0.90,
        "trace_completeness": 1.00,
    }

    gate_checks = {
        name: (metrics[name] >= target if "rate" not in name or "overrun" not in name else metrics[name] <= target)
        for name, target in thresholds.items()
    }
    gate_checks["budget_overrun_rate"] = metrics["budget_overrun_rate"] <= thresholds["budget_overrun_rate"]
    gate_checks["unauthorized_action_rate"] = metrics["unauthorized_action_rate"] <= thresholds["unauthorized_action_rate"]

    summary = {
        "metrics": metrics,
        "bad_trace_count": len(bad_traces),
        "bad_traces": bad_traces,
        "top_failure_reasons": Counter(flag for flags in reasons.values() for flag in flags).most_common(),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
    }
    return summary


report = audit_agent_traces(TRACES)
print("metrics=", report["metrics"])
print("bad_traces=", report["bad_traces"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("gate_pass=", report["gate_pass"])
```

一组预期输出：

```text
metrics= {'task_success_rate': 0.5, 'tool_selection_accuracy': 0.75, 'arg_valid_rate': 0.875, 'observation_use_rate': 0.875, 'state_update_coverage': 0.875, 'budget_overrun_rate': 0.25, 'unauthorized_action_rate': 0.125, 'stop_correct_rate': 0.75, 'trace_completeness': 1.0}
bad_traces= ['calendar_overreach', 'looping_data_task']
top_failure_reasons= [('task_failed', 2), ('wrong_tool', 2), ('unauthorized_action', 1), ('budget_overrun', 1), ('bad_stop', 1), ('observation_ignored', 1)]
gate_pass= False
```

这里 `gate_pass=False` 不是 demo 出错，而是为了暴露三个真实风险：

1. `calendar_overreach` 试图执行目标不允许的发信动作。
2. `looping_data_task` 没有调用计算工具，忽略 observation，并且超出预算。
3. 即使 trace 字段完整，成功率、工具选择、参数、预算和权限仍然不达标。

这就是 Agent 评估和普通问答评估最大的区别：普通问答更关注最终答案，Agent 必须把中间行动链也纳入验收。

## 1.16 面试题：Agent 和普通 LLM 应用有什么区别

回答要点：

```text
普通 LLM 应用通常是输入到输出的生成系统，例如问答、摘要或分类。Agent 则是目标驱动的多步任务执行系统，它会维护状态、选择工具、执行动作、观察反馈并决定下一步。Agent 更适合需要外部信息、工具执行和动态决策的任务，但也更难控制成本、安全和稳定性。评估时不能只看最终回答，还要看 tool trace、state update、权限、预算和停止条件。
```

常见追问：是不是只要能 function calling 就是 Agent？

答：

```text
不是。Function calling 只是工具调用的一种接口机制。Agent 还需要目标、状态、控制循环、观察反馈、预算、停止条件和评估闭环。一个只按固定 schema 调一次 API 的系统，可以是 tool use，但不一定是 Agent。
```

## 1.17 面试题：Agent 系统有哪些核心模块

回答要点：

```text
我会把 Agent 系统拆成 goal、state、planner、tool registry、executor、observation handler、memory、controller 和 logger。Planner 决定下一步，tool registry 描述可用工具，executor 执行工具，observation handler 解析反馈，memory 维护历史，controller 控制预算、权限和停止条件，logger 用于调试、审计和评估。
```

专家追问：哪个模块最容易被忽视？

答：

```text
很多人会忽视 controller 和 trace。没有 controller，Agent 容易无限循环、越权调用或成本失控；没有 trace，就很难知道失败来自目标理解、工具选择、参数、工具执行、observation 使用还是停止条件。
```

## 1.18 小练习

1. 给一个“自动分析销售表并生成报告”的任务，写出 goal、state、tool、observation 和 stop condition。
2. 比较“固定 RAG 问答”和“Agentic RAG 调查报告”的流程差异。
3. 设计一个 Agent trace schema，至少包含 task id、goal、action、tool、arguments、observation、state diff、permission result、budget 和 final status。
4. 给 5 条 toy trace，手算 task success rate、tool selection accuracy、argument validity、budget overrun rate 和 unauthorized action rate。
5. 用本章 demo 增加一个“工具失败后正确追问用户”的 trace，并观察 gate 是否改善。

## 1.19 本章小结

Agent 的本质是目标驱动的多步任务执行系统。它和普通 chat model 的区别在于是否具备状态、工具、观察和执行循环；它和 RAG 的区别在于 RAG 主要增强知识获取，而 Agent 关注动态决策和任务完成；它和 workflow 的区别在于 Agent 更灵活，但也更难控制。

真正可靠的 Agent 不是“模型加工具”这么简单，而是需要清晰目标、结构化状态、受控工具、观察反馈、预算管理、失败恢复、安全边界、trace 和评估门禁。下一章会进入 tool use 与 function calling，具体讲 Agent 如何安全、稳定、可评估地调用工具。
