# 第九章：Multi-Agent

Multi-Agent，多智能体系统，是把多个 Agent 组织成协作网络来完成任务的架构。一个 Agent 负责规划，一个 Agent 负责检索，一个 Agent 负责执行，一个 Agent 负责评审，一个 Agent 负责汇总，听起来比单 Agent 更强，但它并不是“Agent 越多越智能”。

Multi-Agent 真正解决的问题是：复杂任务中，单个 Agent 的上下文、工具权限、专业视角和自我验证能力都有限。多 Agent 可以通过角色分工、并行执行、独立检查和冲突暴露提升可靠性。但它也会带来通信成本、责任不清、重复劳动、冲突处理、权限扩散和错误传播。

本章系统讲 Multi-Agent：为什么需要多 Agent，角色分工、通信协议、协调器、共享状态、辩论与评审、任务分配、冲突解决、共识与投票、成本控制、安全边界、评估指标和面试表达。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已按 `WRITING_PLAN.md` 联网核对 AutoGen、CAMEL、MetaGPT、ChatDev、AI safety via debate 和近年 LLM multi-agent survey 相关资料。这里不把任何框架写成唯一标准，也不展开特定框架 API，而是抽象出面试和工程设计中更稳定的共性问题：

1. 角色如何定义，什么时候值得拆成多个 Agent。
2. Agent 之间应该传什么，不应该传什么。
3. Coordinator、黑板、辩论、投票和工具验证各自解决什么问题。
4. 多 Agent 是否真的比单 Agent 更好，如何用指标证明。
5. 如何限制权限、成本、错误传播和不可审计的自由聊天。

本章不提供绕过权限、规避审批、利用 Agent 间通信传播恶意指令或自动执行高风险动作的方法。涉及安全问题时，只从防御性设计、权限隔离、审计和评估门禁角度讨论。

## 9.1 为什么需要 Multi-Agent

单 Agent 的问题是能力和状态都集中在一个执行循环里。复杂任务中，单 Agent 往往要同时承担规划、检索、写作、执行、验证、总结和风险判断，容易出现四类问题：

1. 角色冲突：既写答案又审答案，容易放过自己的错误。
2. 上下文拥挤：规划、证据、工具结果、历史状态都塞进同一个上下文。
3. 工具权限过大：为了完成所有步骤，单 Agent 往往被授予过多工具。
4. 无法并行：多个独立子任务只能串行完成。

Multi-Agent 的核心直觉是把复杂任务拆成多个受控角色，让它们专业化、互相校验或并行完成子任务。

面试回答：

```text
Multi-Agent 是用多个 Agent 通过角色分工、通信和协调来完成复杂任务。它可以提升并行性、专业化和互相检查能力，例如 planner、researcher、coder、reviewer 分别负责不同环节。但它也会增加通信成本、协调复杂度、冲突解决和权限管理问题。是否使用多 Agent，要看它是否真实提升任务成功率、验证质量或并行效率，而不是看系统里有几个 Agent。
```

## 9.2 Multi-Agent 的基本结构

常见结构包括：

1. Centralized：一个 coordinator 分配任务、收集结果、做最终决策。
2. Decentralized：多个 Agent 彼此通信，没有单一中心节点。
3. Hierarchical：高层 Agent 规划，低层 Agent 执行。
4. Debate：多个 Agent 给出不同观点，再由 judge 或 verifier 汇总。
5. Committee：多个 Agent 独立解题后投票、加权汇总或触发复核。
6. Blackboard：所有 Agent 通过共享状态表读写任务状态和证据。

工程系统最常用的是 centralized 或 hierarchical，因为它们更容易控制流程、权限、日志和预算。完全自由聊天式 multi-agent 在 demo 中很直观，但生产系统通常更需要结构化消息、状态机和审计 trace。

## 9.3 Agent 抽象与角色集合

可以把一个 multi-agent 系统中的 Agent 集合写成：

```math
\mathcal{A}=\{a_1,\ldots,a_n\}
```

其中每个 Agent 不只是一个 prompt，而是一个带角色、工具、权限、上下文和预算的执行单元：

```math
a_i=(r_i,T_i,P_i,C_i,B_i)
```

变量含义：

1. `r_i` 是第 `i` 个 Agent 的角色，例如 planner、researcher、coder、reviewer。
2. `T_i` 是它可调用的工具集合。
3. `P_i` 是它的权限边界。
4. `C_i` 是它能看到的上下文。
5. `B_i` 是它可消耗的预算，例如 token、时间、工具调用次数或成本。

这条公式的工程含义是：不要把 Agent 只理解为“一个不同的 system prompt”。可靠的 Multi-Agent 设计必须同时定义角色、工具、权限、上下文和预算。只换 prompt、不限权限、不管 trace，通常只是把单 Agent 的风险复制了多份。

## 9.4 关键公式与 Multi-Agent 指标速查

Multi-Agent 的核心对象包括 Agent、消息、任务分配、通信图、冲突和门禁。

### 9.4.1 消息协议

一条 Agent 间消息可以抽象为：

```math
m_t=(s_t,r_t,\iota_t,c_t,E_t,\gamma_t)
```

其中 `s_t` 是 sender，`r_t` 是 receiver，`\iota_t` 是 intent，`c_t` 是消息内容，`E_t` 是证据集合，`\gamma_t` 是置信度。

消息协议至少要回答三个问题：

1. 这条消息是谁发给谁的。
2. 它是请求、回答、证据、反驳、状态更新还是最终建议。
3. 它有没有可检查的证据和置信度。

消息有效率可以写成：

```math
R_{\mathrm{msg}}=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\mathrm{valid}(m_t)]
```

其中 `valid` 表示消息 schema 可解析、字段完整、接收方明确、意图合法。

### 9.4.2 任务分配

一个子任务分配可以写成：

```math
z_j=(u_j,a(z_j),d_j,\rho_j)
```

其中 `u_j` 是子任务，`a(z_j)` 是被分配的 Agent，`d_j` 是 deadline 或依赖约束，`\rho_j` 是风险等级。

角色匹配率衡量任务是否分给了合适角色：

```math
R_{\mathrm{role}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\mathrm{role}(a(z_j))=\mathrm{role}^{\star}(z_j)]
```

其中 `M` 是子任务数量，`\mathrm{role}^{\star}(z_j)` 是该子任务理想角色。

### 9.4.3 通信图

Agent 之间的通信关系可以写成有向图：

```math
G_{\mathrm{comm}}=(\mathcal{A},\mathcal{E}_{\mathrm{comm}})
```

如果所有 Agent 都能任意互发消息，通信图接近完全图，沟通成本会迅速增加。若有 coordinator，通信图通常更接近星型或层级结构，更容易审计。

通信成本可以粗略写成：

```math
C_{\mathrm{comm}}=\sum_{t=1}^{T}C(m_t)
```

其中 `C(m_t)` 可以按 token、延迟、模型调用成本或人工审阅成本估算。

### 9.4.4 证据支持与冲突解决

证据支持率衡量消息或结论是否有可追溯依据：

```math
R_{\mathrm{evid}}=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\mathrm{supported}(m_t)]
```

冲突解决率衡量系统是否识别并解决矛盾结论：

```math
R_{\mathrm{conf}}=\frac{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k \land \mathrm{resolved}_k]}{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k]}
```

注意：冲突“被解决”不等于“解决正确”。高风险任务还要看解决正确率：

```math
R_{\mathrm{conf\_ok}}=\frac{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k \land \mathrm{correct}_k]}{\sum_{k=1}^{K}\mathbf{1}[\mathrm{conflict}_k]}
```

### 9.4.5 重复劳动、收益和成本

重复劳动率衡量多个 Agent 是否在做同一件事：

```math
R_{\mathrm{dup}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\mathrm{duplicate}(z_j)]
```

Multi-Agent 相比单 Agent 的收益可以写成：

```math
L_{\mathrm{multi}}=S_{\mathrm{multi}}-S_{\mathrm{single}}
```

其中 `S_{\mathrm{multi}}` 是 multi-agent 系统任务分数，`S_{\mathrm{single}}` 是单 Agent baseline 分数。这个指标很重要，因为 Multi-Agent 的比较对象不是“空系统”，而是更简单、更便宜的单 Agent 或固定 workflow。

总成本可以写成：

```math
C_{\mathrm{multi}}=\sum_{i=1}^{n}C(a_i)+\sum_{t=1}^{T}C(m_t)+C_{\mathrm{coord}}+C_{\mathrm{verify}}
```

其中 `C_{\mathrm{coord}}` 是协调器成本，`C_{\mathrm{verify}}` 是验证和人工升级成本。

### 9.4.6 Multi-Agent 上线门禁

一个简化的上线门禁可以写成：

```math
G_{\mathrm{multi}}=\mathbf{1}[S_{\mathrm{multi}}\ge \tau_s \land L_{\mathrm{multi}}\ge \tau_l \land R_{\mathrm{conf}}\ge \tau_c \land R_{\mathrm{dup}}\le \tau_d \land R_{\mathrm{perm}}\le \tau_p \land C_{\mathrm{multi}}\le B]
```

其中 `R_{\mathrm{perm}}` 是权限违规率，`B` 是成本预算，`\tau_s,\tau_l,\tau_c,\tau_d,\tau_p` 是上线阈值。

直觉：Multi-Agent 必须同时证明“做得好”“比单 Agent 有增益”“冲突能处理”“没有大量重复劳动”“权限不乱”“成本可接受”。只展示一个看起来很热闹的协作过程，不足以上线。

## 9.5 角色分工

Multi-Agent 的关键是角色明确。

常见角色包括：

1. Planner：拆解任务和制定计划。
2. Researcher：检索资料、收集证据和标注来源。
3. Executor：调用工具或执行动作。
4. Coder：阅读代码、生成 patch、运行测试。
5. Reviewer：检查结果、diff、证据和测试。
6. Critic：专门找漏洞、反例和边界条件。
7. Judge：在候选答案或冲突结论之间做结构化裁决。
8. Summarizer：汇总输出，压缩 trace。
9. Coordinator：分配任务、控制流程、预算和权限。

角色不是越多越好。一个实用判断是：如果拆出某个 Agent 能降低单个上下文复杂度、提升验证独立性、限制权限范围或实现真实并行，就值得考虑；如果只是把同一份工作拆成多人聊天，通常会增加成本而不提升质量。

## 9.6 Coordinator 模式

Coordinator 是多 Agent 系统中的调度者。

它负责：

1. 理解用户目标和成功标准。
2. 拆解子任务和依赖关系。
3. 选择合适 Agent。
4. 控制每个 Agent 的上下文和工具权限。
5. 收集结果并维护全局状态。
6. 识别冲突和缺失证据。
7. 控制预算、超时和停止条件。
8. 生成最终回答或请求人工确认。

Coordinator 模式的优势是可控，适合生产系统。缺点是 coordinator 可能成为瓶颈。如果 coordinator 分错任务、漏看冲突或过早停止，整个系统会偏离方向。

工程上可以把 coordinator 设计成“状态机 + 路由器 + 审计器”，而不是一个自由生成长文本的 Agent。这样更容易做 replay、debug 和指标统计。

## 9.7 通信协议

Agent 之间不能随意聊天，最好有结构化通信协议。

一个实用消息 schema 可以包含：

1. `sender`：发送者。
2. `receiver`：接收者。
3. `task_id`：任务编号。
4. `intent`：消息目的，例如 assign、question、evidence、critique、decision。
5. `content`：正文内容。
6. `evidence`：证据、引用或工具结果。
7. `confidence`：置信度。
8. `status`：完成、失败、阻塞或需要澄清。
9. `risk_level`：是否涉及高风险动作。

结构化通信的价值：

1. 减少误解。
2. 降低上下文冗余。
3. 支持日志回放。
4. 支持自动评估。
5. 支持权限审计。
6. 支持冲突定位。

面试中可以强调：Multi-Agent 不是让多个模型互相发散聊天，而是要把通信变成可解析、可过滤、可度量的事件流。

## 9.8 共享状态与 Blackboard

多 Agent 需要共享部分状态，但共享越多不一定越好。

共享状态可以包括：

1. 全局目标。
2. 当前计划。
3. 子任务状态。
4. 已收集证据。
5. 已执行动作。
6. 冲突点。
7. 风险标记。
8. 最终决策。

工程上常用 blackboard 或 task state table：

```text
task_id | owner | status | evidence_ids | blockers | risk_level | decision
```

Blackboard 的优点是状态集中、便于审计。缺点是容易成为上下文膨胀点。如果每个 Agent 都读取全部 blackboard，系统仍然会退化成一个超长上下文单 Agent。更稳的做法是：按角色、任务和权限过滤可见状态。

## 9.9 辩论、Critique 与 Judge

多 Agent 常用于辩论或互评。

典型流程：

```text
Agent A 给出答案
Agent B 找反例
Agent C 检查证据
Judge 汇总并选择最终结论
```

优势：

1. 能发现单 Agent 忽略的问题。
2. 能从不同角度评估答案。
3. 适合复杂决策、开放问题和假设比较。
4. 可以把“生成”和“检查”分离。

局限：

1. 多个 Agent 可能共享同样偏差。
2. 辩论可能变成冗长文本互相说服。
3. Judge 也可能被流畅表达误导。
4. 成本明显上升。
5. 对可执行任务，工具验证通常比纯文本辩论更可靠。

面试中要避免把 debate 说成万能验证器。更稳的表达是：辩论适合暴露候选假设和不确定性，最终最好结合证据、工具验证、测试、规则门禁或人工复核。

## 9.10 并行执行

Multi-Agent 可以提升并行性。

例如研究报告任务：

1. Agent A 查背景。
2. Agent B 查竞品。
3. Agent C 查技术路线。
4. Agent D 查风险。
5. Coordinator 汇总。

并行执行能节省墙钟时间，但不是总成本更低。系统还需要处理结果合并、来源去重和冲突。如果多个 Agent 得到矛盾结论，系统必须记录来源并解决冲突，而不是让 summarizer 随机选择一个更像真的说法。

## 9.11 冲突解决

多 Agent 系统一定会出现冲突。

冲突类型：

1. 事实冲突：两个 Agent 给出不同事实。
2. 方案冲突：两个 Agent 推荐不同路径。
3. 优先级冲突：速度、成本、质量、安全目标冲突。
4. 工具结果冲突：检索、测试、执行结果不一致。
5. 权限冲突：某个 Agent 试图执行超出权限的动作。
6. 资源冲突：多个 Agent 争用预算、工具或上下文窗口。

解决方式：

1. 要求提供证据。
2. 调用外部工具验证。
3. 使用 judge Agent 做结构化裁决。
4. 由 coordinator 按规则决策。
5. 触发人工确认。
6. 保留不确定性，而不是强行统一。

高风险任务中，冲突不应只由模型闭环自动决定。更合理的设计是：低风险冲突可由规则或 verifier 自动处理，高风险冲突升级给人或确定性系统。

## 9.12 任务分配与权限隔离

任务分配要同时考虑能力、上下文、工具和权限。

例如：

1. Researcher 有检索工具，但没有写权限。
2. Coder 有文件编辑权限，但不能访问无关隐私数据。
3. Reviewer 只能读 diff、测试结果和审计 trace。
4. Judge 只能做裁决，不能执行外部动作。
5. Coordinator 可以分配任务，但不直接执行高风险动作。

权限分离可以降低事故范围。不要让所有 Agent 都拥有全部工具权限。否则 Multi-Agent 不但没有增加安全性，反而把一个过大权限 Agent 复制成多个过大权限 Agent。

## 9.13 Consensus、Voting 与加权汇总

Committee 型 Multi-Agent 常用投票或共识。

常见方式：

1. 多个 Agent 独立回答，简单多数投票。
2. 按历史准确率或任务匹配度加权投票。
3. 要求每个 Agent 提供证据，再按证据质量汇总。
4. 分歧过大时不输出确定结论，触发复核。

投票适合答案空间明确、独立错误概率较低的任务。它不适合所有开放任务，因为多个 Agent 可能共享同一模型、同一训练偏差、同一错误检索来源。投票前要尽量保证候选答案独立，投票后要看证据，而不是只看票数。

## 9.14 多 Agent 的成本

Multi-Agent 成本很高。

成本来源：

1. 多个模型调用。
2. Agent 之间通信。
3. 重复检索。
4. 重复验证。
5. 上下文汇总。
6. 冲突解决。
7. 人工升级。
8. 额外 trace 存储和回放。

因此要判断任务是否值得多 Agent。简单任务用单 Agent 或固定 workflow 更合适。一个常见反模式是：一个一句话任务被 planner、researcher、writer、reviewer、summarizer 转一圈，最后质量没有提升，只是成本上升。

## 9.15 Multi-Agent 和 Workflow

Multi-Agent 和 workflow 可以结合。

一种实用架构：

```text
固定 workflow 控制主流程
关键步骤交给专门 Agent 执行
Coordinator 汇总结果
Verifier 做最终检查
```

这样既保留 workflow 的稳定性，又利用 Agent 的灵活性。生产系统中更常见的是“workflow 主控 + Agent 局部自治”，而不是“多个 Agent 完全自由聊天”。

## 9.16 安全风险

Multi-Agent 增加了安全面。

风险包括：

1. Agent 之间传播错误信息。
2. 一个 Agent 接收不可信内容后影响其他 Agent。
3. 权限边界不清。
4. 日志难以审计。
5. 冲突处理不透明。
6. 成本失控。
7. 多 Agent 互相强化错误结论。
8. 高风险动作缺少统一审批。

安全设计：

1. 最小权限。
2. 明确角色边界。
3. 结构化消息。
4. 共享状态审计。
5. 高风险动作统一由 controller 审批。
6. 不信任其他 Agent 的未验证结论。
7. 对不可信内容做来源标记和上下文隔离。
8. 保留 trace，支持 replay 和责任定位。

## 9.17 评估指标

Multi-Agent 评估可以看：

1. 任务成功率。
2. 相比单 Agent 的提升。
3. 平均成本。
4. 平均延迟。
5. 通信轮数。
6. 角色匹配率。
7. 消息 schema 有效率。
8. 证据支持率。
9. 冲突解决成功率。
10. 重复工作比例。
11. 权限违规率。
12. 不必要 multi-agent 比例。
13. 人工接管率。
14. 最终答案证据支持率。

核心问题是：多 Agent 是否真的比单 Agent 更好，还是只是更贵、更复杂。面试中如果只说“多个 Agent 可以互相协作”，回答还停留在概念层；能说出 baseline、指标、成本、冲突和权限，才更像工程落地回答。

## 9.18 最小可运行 Multi-Agent audit demo

下面这个 demo 不调用外部模型，而是构造 4 条 toy multi-agent trace，审计角色匹配、任务成功、消息有效性、证据支持、冲突解决、重复劳动、权限违规和相对单 Agent 的收益。

它演示的问题是：Multi-Agent 不能只看最终是否成功，还要看协作过程是否可控、可解释、低重复、低权限风险，并且是否真的优于单 Agent baseline。

```python
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    tools: tuple
    permissions: tuple


@dataclass(frozen=True)
class Assignment:
    task_id: str
    required_role: str
    assigned_agent: str
    success: bool
    duplicate: bool = False


@dataclass(frozen=True)
class Message:
    message_id: str
    sender: str
    receiver: str
    intent: str
    valid_schema: bool
    supported_by_evidence: bool


@dataclass(frozen=True)
class Conflict:
    conflict_id: str
    detected: bool
    resolved: bool
    resolution_correct: bool


@dataclass(frozen=True)
class Run:
    run_id: str
    agents: dict
    assignments: tuple
    messages: tuple
    conflicts: tuple
    final_success: bool
    single_agent_success: bool
    total_cost: float
    permission_violations: int = 0
    unnecessary_multi_agent: bool = False


def role_match(run, assignment):
    return run.agents[assignment.assigned_agent].role == assignment.required_role


runs = [
    Run(
        run_id="research_report",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "plan": AgentSpec("plan", "planner", ("outline",), ("read",)),
            "search": AgentSpec("search", "researcher", ("search",), ("read",)),
            "check": AgentSpec("check", "verifier", ("compare",), ("read",)),
            "sum": AgentSpec("sum", "summarizer", ("write",), ("read",)),
        },
        assignments=(
            Assignment("scope", "planner", "plan", True),
            Assignment("collect", "researcher", "search", True),
            Assignment("verify", "verifier", "check", True),
            Assignment("summarize", "summarizer", "sum", True),
        ),
        messages=(
            Message("m1", "coord", "plan", "assign", True, True),
            Message("m2", "plan", "search", "request_evidence", True, True),
            Message("m3", "search", "check", "provide_evidence", True, True),
            Message("m4", "check", "sum", "verified_summary", True, True),
        ),
        conflicts=(Conflict("c1", True, True, True),),
        final_success=True,
        single_agent_success=False,
        total_cost=1.40,
    ),
    Run(
        run_id="code_review_swarm",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "dev": AgentSpec("dev", "coder", ("patch", "test"), ("read", "write")),
            "rev": AgentSpec("rev", "reviewer", ("diff",), ("read",)),
            "search": AgentSpec("search", "researcher", ("grep",), ("read",)),
        },
        assignments=(
            Assignment("inspect_diff", "reviewer", "search", False),
            Assignment("run_tests", "coder", "dev", True),
            Assignment("review_security", "reviewer", "rev", True),
            Assignment("inspect_diff_dup", "reviewer", "rev", True, duplicate=True),
        ),
        messages=(
            Message("m5", "search", "coord", "claim", True, False),
            Message("m6", "dev", "rev", "test_result", True, True),
            Message("m7", "rev", "coord", "review", True, True),
        ),
        conflicts=(),
        final_success=True,
        single_agent_success=False,
        total_cost=1.55,
        permission_violations=1,
    ),
    Run(
        run_id="debate_answer",
        agents={
            "a": AgentSpec("a", "debater", ("reason",), ("read",)),
            "b": AgentSpec("b", "debater", ("reason",), ("read",)),
            "j": AgentSpec("j", "judge", ("score",), ("read",)),
        },
        assignments=(
            Assignment("argue_yes", "debater", "a", True),
            Assignment("argue_no", "debater", "b", True),
            Assignment("judge", "judge", "j", False),
        ),
        messages=(
            Message("m8", "a", "j", "argument", True, False),
            Message("m9", "b", "j", "counterargument", True, False),
            Message("m10", "j", "a", "clarify", True, True),
            Message("m11", "a", "j", "unsupported_claim", False, False),
            Message("m12", "j", "coord", "final_judgment", True, True),
        ),
        conflicts=(Conflict("c2", True, False, False),),
        final_success=False,
        single_agent_success=False,
        total_cost=1.10,
    ),
    Run(
        run_id="over_coordinated_small_task",
        agents={
            "coord": AgentSpec("coord", "coordinator", ("route",), ("assign",)),
            "plan": AgentSpec("plan", "planner", ("outline",), ("read",)),
            "writer": AgentSpec("writer", "writer", ("write",), ("read",)),
            "rev": AgentSpec("rev", "reviewer", ("check",), ("read",)),
        },
        assignments=(
            Assignment("plan_one_sentence", "planner", "plan", True),
            Assignment("write_one_sentence", "writer", "writer", True),
            Assignment("review_one_sentence", "reviewer", "rev", True, duplicate=True),
        ),
        messages=(
            Message("m13", "plan", "writer", "outline", True, True),
            Message("m14", "rev", "coord", "approval", True, True),
        ),
        conflicts=(),
        final_success=True,
        single_agent_success=True,
        total_cost=1.25,
        unnecessary_multi_agent=True,
    ),
]

total_runs = len(runs)
all_assignments = [assignment for run in runs for assignment in run.assignments]
all_messages = [message for run in runs for message in run.messages]
all_conflicts = [conflict for run in runs for conflict in run.conflicts if conflict.detected]

metrics = {
    "task_success_rate": round(sum(run.final_success for run in runs) / total_runs, 3),
    "single_agent_lift_rate": round(
        sum(run.final_success and not run.single_agent_success for run in runs) / total_runs, 3
    ),
    "role_match_rate": round(
        sum(role_match(run, assignment) for run in runs for assignment in run.assignments) / len(all_assignments), 3
    ),
    "assignment_success_rate": round(sum(a.success for a in all_assignments) / len(all_assignments), 3),
    "message_valid_rate": round(sum(m.valid_schema for m in all_messages) / len(all_messages), 3),
    "evidence_support_rate": round(sum(m.supported_by_evidence for m in all_messages) / len(all_messages), 3),
    "conflict_resolution_rate": round(sum(c.resolved for c in all_conflicts) / len(all_conflicts), 3),
    "correct_resolution_rate": round(sum(c.resolution_correct for c in all_conflicts) / len(all_conflicts), 3),
    "duplicate_work_rate": round(sum(a.duplicate for a in all_assignments) / len(all_assignments), 3),
    "permission_violation_rate": round(
        sum(run.permission_violations for run in runs) / len(all_assignments), 3
    ),
    "unnecessary_multi_agent_rate": round(
        sum(run.unnecessary_multi_agent for run in runs) / total_runs, 3
    ),
    "avg_cost": round(sum(run.total_cost for run in runs) / total_runs, 3),
}

failure_reasons = Counter()
problem_runs = []
for run in runs:
    run_has_problem = False
    for assignment in run.assignments:
        if not role_match(run, assignment):
            failure_reasons["role_mismatch"] += 1
            run_has_problem = True
        if not assignment.success:
            failure_reasons["assignment_failed"] += 1
            run_has_problem = True
        if assignment.duplicate:
            failure_reasons["duplicate_work"] += 1
            run_has_problem = True
    for message in run.messages:
        if not message.valid_schema:
            failure_reasons["invalid_message_schema"] += 1
            run_has_problem = True
        if not message.supported_by_evidence:
            failure_reasons["unsupported_message"] += 1
            run_has_problem = True
    for conflict in run.conflicts:
        if conflict.detected and not conflict.resolved:
            failure_reasons["unresolved_conflict"] += 1
            run_has_problem = True
    if run.permission_violations:
        failure_reasons["permission_violation"] += run.permission_violations
        run_has_problem = True
    if run.unnecessary_multi_agent:
        failure_reasons["unnecessary_multi_agent"] += 1
        run_has_problem = True
    if not run.final_success:
        failure_reasons["task_failed"] += 1
        run_has_problem = True
    if run_has_problem:
        problem_runs.append(run.run_id)

gates = {
    "task_success": metrics["task_success_rate"] >= 0.80,
    "single_agent_lift": metrics["single_agent_lift_rate"] >= 0.30,
    "role_match": metrics["role_match_rate"] >= 0.90,
    "message_valid": metrics["message_valid_rate"] >= 0.95,
    "evidence_support": metrics["evidence_support_rate"] >= 0.80,
    "conflict_resolution": metrics["conflict_resolution_rate"] >= 0.80,
    "duplicate_work": metrics["duplicate_work_rate"] <= 0.10,
    "permission": metrics["permission_violation_rate"] == 0.0,
    "unnecessary_multi_agent": metrics["unnecessary_multi_agent_rate"] <= 0.10,
    "cost": metrics["avg_cost"] <= 1.50,
}

top_failure_reasons = sorted(failure_reasons.items(), key=lambda item: (-item[1], item[0]))

print(f"metrics={metrics}")
print(f"problem_runs={problem_runs}")
print(f"top_failure_reasons={top_failure_reasons}")
print(f"gates={gates}")
print(f"gate_pass={all(gates.values())}")
```

输出示例：

```text
metrics={'task_success_rate': 0.75, 'single_agent_lift_rate': 0.5, 'role_match_rate': 0.929, 'assignment_success_rate': 0.857, 'message_valid_rate': 0.929, 'evidence_support_rate': 0.714, 'conflict_resolution_rate': 0.5, 'correct_resolution_rate': 0.5, 'duplicate_work_rate': 0.143, 'permission_violation_rate': 0.071, 'unnecessary_multi_agent_rate': 0.25, 'avg_cost': 1.325}
problem_runs=['code_review_swarm', 'debate_answer', 'over_coordinated_small_task']
top_failure_reasons=[('unsupported_message', 4), ('assignment_failed', 2), ('duplicate_work', 2), ('invalid_message_schema', 1), ('permission_violation', 1), ('role_mismatch', 1), ('task_failed', 1), ('unnecessary_multi_agent', 1), ('unresolved_conflict', 1)]
gates={'task_success': False, 'single_agent_lift': True, 'role_match': True, 'message_valid': False, 'evidence_support': False, 'conflict_resolution': False, 'duplicate_work': False, 'permission': False, 'unnecessary_multi_agent': False, 'cost': True}
gate_pass=False
```

这个 demo 的 `gate_pass=False` 不是程序错误，而是刻意暴露 multi-agent 系统的常见问题：成功率不足、消息证据不够、冲突未解决、重复劳动、权限违规和小任务过度编排。真实系统上线前，必须把这些 trace 级问题纳入评估。

## 9.19 常见失败模式

1. 角色分工不清：每个 Agent 都在规划、执行和总结。
2. Coordinator 分配错误：任务给错角色，后面再努力也难补救。
3. 多个 Agent 重复做同一件事。
4. Agent 互相传递错误结论。
5. Judge 被流畅表达误导。
6. 冲突没有解决就输出答案。
7. 通信成本超过收益。
8. 权限边界混乱。
9. 共享 memory 或 blackboard 被污染。
10. 最终结果没人负责。
11. 简单任务过度使用多 Agent。
12. 缺少单 Agent baseline，无法证明增益。

Multi-Agent 的目标不是把系统变热闹，而是提高任务成功率、可验证性、并行效率和权限可控性。

## 9.20 面试题：什么时候需要 Multi-Agent

回答要点：

```text
当任务复杂、需要多个专业角色、可以并行处理、需要独立审查或需要不同视角暴露冲突时，可以考虑 Multi-Agent。例如研究报告、代码开发加 review、复杂规划、多来源验证和高不确定性决策。但简单任务不适合 Multi-Agent，因为通信和协调成本会超过收益。是否使用多 Agent 要看它是否真实提升任务成功率、验证质量、并行效率或权限隔离能力，并且要和单 Agent baseline 比较。
```

## 9.21 面试题：如何设计 Multi-Agent 系统

回答要点：

```text
我会先明确任务是否真的需要多 Agent，然后设计 coordinator、角色分工、工具权限和通信协议。每个 Agent 只拿到完成任务所需的上下文和工具。共享状态通过结构化 blackboard 管理，冲突通过证据、工具验证、judge 或人工确认解决。系统需要记录 trace，评估任务成功率、单 Agent 增益、角色匹配率、消息有效率、证据支持率、冲突解决率、重复工作率、权限违规率、成本和延迟。
```

## 9.22 面试题：Debate、Verifier 和投票有什么区别

回答要点：

```text
Debate 让多个 Agent 从不同角度提出观点和反驳，适合暴露假设和不确定性；Verifier 更强调用规则、测试、检索证据或工具结果检查结论，适合可执行或可验证任务；投票适合多个相对独立候选答案的聚合。工程上不能只依赖 debate 的自然语言说服力，高风险任务最好结合 verifier、证据引用、测试和人工复核。
```

## 9.23 小练习

1. 给一个“写一份竞品调研报告”的任务，设计 4 个 Agent 角色、每个角色的工具权限和消息 schema。
2. 给一个“修复代码 bug 并提交 patch”的任务，说明为什么 coder 和 reviewer 不应该拥有完全相同的权限。
3. 设计一个冲突处理规则：当 researcher 和 verifier 对事实结论不一致时，系统应该如何升级。
4. 修改本章 demo，让 `debate_answer` 的冲突被正确解决，观察 `conflict_resolution_rate` 和 `gate_pass` 如何变化。
5. 比较单 Agent、固定 workflow 和 Multi-Agent 在成本、可靠性、可解释性上的差异。

## 9.24 本章小结

Multi-Agent 通过角色分工、并行执行和互相检查，让 Agent 系统有机会处理更复杂任务。但它不是免费午餐，会引入通信成本、协调复杂度、冲突解决、权限管理、安全风险和评估难度。

可靠的 Multi-Agent 系统通常不是一群 Agent 自由聊天，而是有 coordinator、结构化消息、共享状态、明确权限、冲突解决和评估闭环的工程系统。下一章会进入 Agent 评估，系统讨论如何衡量 Agent 是否真的完成任务、是否安全、是否高效。
