# 第 31 章 多 Agent 系统的失败模式：循环、冲突、幻觉传播

前面几章我们讲了 A2A 的协议抽象、任务生命周期、上下文边界、与 MCP 的分工，以及跨 Agent 权限和审计。

这一章要讲一个更现实的问题：多 Agent 系统会怎样失败。

很多人刚接触 Multi-Agent 时，会觉得“多个 Agent 分工协作，一定比一个 Agent 更强”。这只对了一半。多 Agent 的确可以带来专业化、并行化和模块化，但它也会引入新的失败模式：循环委派、角色冲突、状态不一致、上下文漂移、责任模糊、幻觉传播、重复工作、成本爆炸和安全边界破裂。

如果没有治理，多 Agent 系统可能比单 Agent 更不可靠。

本章的核心结论是：

> 多 Agent 系统的难点不在“让多个 Agent 都能说话”，而在防止它们互相放大错误、争抢控制权、传错上下文和陷入无限协作。

## 31.0 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 A2A Protocol Specification、Microsoft AutoGen 的多 Agent team / handoff / termination / state / tracing 文档、Anthropic 关于 building effective agents 的工程建议、OWASP LLM Top 10 和 NIST AI RMF Generative AI Profile 等公开资料。

需要先划清边界：

1. A2A 给出的是跨 Agent Task、Message、Artifact、状态、上下文和安全语义的协议基础；本章讨论的是这些结构在真实多 Agent 协作中可能出现的失控模式。
2. AutoGen、LangGraph、Anthropic 等工程资料都强调多 Agent / agentic system 要有明确终止条件、状态管理、观察性、人工接管和复杂度控制；本章抽象这些共性，不把某个框架的 API、team 类型或配置项写成永久标准。
3. OWASP 和 NIST 关注 prompt injection、过度权限、错误信息、资源消耗、隐私和治理风险；本章把这些风险映射到多 Agent 链路中的上下文转发、权限继承、幻觉传播和成本失控。
4. 本章新增的审计指标和 Python demo 是教学用 toy runtime，不实现真实 A2A server、MCP server、调度器、锁服务、数据库事务、IAM、DLP 或生产审计系统。
5. 生产系统中，失败治理必须落在 runtime、协议适配层、权限系统、trace / audit、eval harness 和人工流程里，不能只写在 prompt 或某个 Agent 的自我约束里。

## 31.1 为什么多 Agent 更容易出现系统性错误

单 Agent 失败通常发生在一个上下文里：模型理解错、工具调用错、答案幻觉、权限判断错。

多 Agent 失败则更像分布式系统故障：

1. 一个 Agent 的错误会传播给另一个 Agent。
2. 一个 Agent 的假设会被另一个 Agent 当成事实。
3. 多个 Agent 可能同时修改同一对象。
4. 委派链可能无限延长。
5. 状态可能在不同 Agent 中不一致。
6. 上下文在摘要和转发中逐渐失真。
7. 责任边界被拆散，难以追责。

所以，多 Agent 系统需要用分布式系统的思维来设计：状态机、超时、幂等、锁、仲裁、trace、权限、降级、熔断和人工接管都很重要。

## 31.2 失败模式一：循环委派

循环委派是最典型的 Multi-Agent 失败模式。

例如：

```text
总控 Agent -> 数据 Agent：请分析退款率。
数据 Agent -> 知识库 Agent：请解释退款率定义。
知识库 Agent -> 数据 Agent：请提供具体指标场景。
数据 Agent -> 总控 Agent：需要更多业务上下文。
总控 Agent -> 数据 Agent：请继续分析退款率。
```

系统看起来一直在工作，实际上在绕圈。

更严重的情况是 Agent 互相委派：

```text
Agent A -> Agent B
Agent B -> Agent C
Agent C -> Agent A
```

### 31.2.1 循环委派的原因

常见原因包括：

1. 任务边界不清。
2. Agent Card 能力声明过宽。
3. 缺少最大委派深度。
4. 缺少任务去重。
5. Agent 不知道自己已经处理过类似任务。
6. input_required 被错误地转成新任务。
7. 总控 Agent 没有全局任务图。

### 31.2.2 防护机制

防止循环委派可以用这些方法：

1. 设置最大委派深度。
2. 为每个任务维护 parent_task_id。
3. 维护任务 DAG，而不是任意图。
4. 使用 idempotency_key 做任务去重。
5. 检测相似任务重复提交。
6. 限制下游 Agent 继续委派权限。
7. 超过轮次后升级人工接管。

例如任务元数据可以包含：

```json
{
  "task_id": "task_c",
  "parent_task_id": "task_b",
  "root_task_id": "task_root",
  "delegation_depth": 3,
  "max_delegation_depth": 5,
  "visited_agents": [
    "agent.orchestrator.v1",
    "agent.data.v1",
    "agent.kb.v1"
  ]
}
```

如果下一个委派目标已经出现在 visited_agents 中，就要谨慎处理。

## 31.3 失败模式二：角色冲突和控制权争夺

多 Agent 系统里，不同 Agent 可能对“谁说了算”理解不同。

例如：

1. 代码 Agent 认为应该立刻修复实现。
2. 测试 Agent 认为应该先补测试。
3. 产品 Agent 认为需求本身需要改。
4. 安全 Agent 认为该功能不能上线。

如果没有仲裁机制，系统可能出现冲突：

```text
代码 Agent：我已经修改代码。
测试 Agent：我又改回去了，因为测试不通过。
架构 Agent：这两种改法都不对。
总控 Agent：继续尝试。
```

### 31.3.1 冲突类型

常见冲突包括：

1. 目标冲突：一个 Agent 追求速度，一个 Agent 追求安全。
2. 方案冲突：不同 Agent 给出不同实现方案。
3. 权限冲突：一个 Agent 认为可以执行，另一个认为不允许。
4. 状态冲突：一个 Agent 认为任务完成，另一个认为失败。
5. 产物冲突：多个 Agent 修改同一文件或同一报告。
6. 优先级冲突：不同 Agent 对任务重要性排序不同。

### 31.3.2 仲裁机制

多 Agent 系统需要明确仲裁策略：

1. 总控 Agent 仲裁。
2. 规则优先仲裁。
3. 人类审批仲裁。
4. 多数投票。
5. 证据权重仲裁。
6. 风险优先仲裁。
7. 领域权威 Agent 仲裁。

不同场景适合不同策略。

安全场景通常不能用简单多数投票。如果安全 Agent 指出高危漏洞，即使产品 Agent 和代码 Agent 都认为可以上线，也应该进入人工复核或安全优先策略。

## 31.4 失败模式三：幻觉传播

幻觉传播是多 Agent 系统最危险的问题之一。

单 Agent 幻觉通常止于一个回答。多 Agent 中，一个 Agent 的幻觉可能被其他 Agent 当作事实继续加工。

例如：

```text
数据 Agent：退款率升高可能由 5 月 18 日上线的策略导致。
报告 Agent：5 月 18 日上线的策略导致退款率升高。
决策 Agent：建议回滚 5 月 18 日策略。
运维 Agent：准备回滚。
```

这里，“可能”在传播过程中变成了“确定”。

### 31.4.1 幻觉传播的原因

常见原因包括：

1. 没有保留 claim_type。
2. 没有保留 evidence。
3. 摘要时删除了不确定性。
4. 下游 Agent 过度信任上游输出。
5. 总控 Agent 没有验证关键结论。
6. 多个 Agent 使用同一个错误来源。
7. 报告 Agent 为了表达流畅而弱化限制条件。

### 31.4.2 防护机制

防止幻觉传播需要：

1. 标记事实、假设、推断和建议。
2. 保留证据引用。
3. 保留置信度。
4. 保留限制说明。
5. 对关键结论做二次验证。
6. 使用独立来源交叉验证。
7. 对高风险建议要求人工确认。

结构化 claim 示例：

```json
{
  "claim": "Refund rate increase may be related to the May 18 policy change.",
  "claim_type": "hypothesis",
  "confidence": "medium",
  "evidence": [
    "artifact://task_data/refund_trend.csv",
    "kb://release-notes/2026-05-18"
  ],
  "limitations": [
    "No controlled experiment was performed.",
    "Correlation does not prove causation."
  ]
}
```

## 31.5 失败模式四：上下文漂移

上下文漂移指任务在多轮转发中逐渐偏离原始目标。

例如用户原始目标是：

```text
请解释退款率为什么上升，不要修改生产系统。
```

几轮委派后变成：

```text
请修复退款率问题。
```

再往后可能变成：

```text
请回滚最近策略。
```

原始约束“不要修改生产系统”丢失了。

### 31.5.1 漂移来源

上下文漂移常来自：

1. 摘要过度压缩。
2. 子任务描述太自由。
3. 下游 Agent 自动补全目标。
4. 中间结论被写成最终目标。
5. 限制条件没有随消息转发。
6. root task goal 没有被保留。

### 31.5.2 防护机制

可以使用：

1. root_task_goal 固定原始目标。
2. preserved_constraints 保留硬约束。
3. 每次委派都携带禁止事项。
4. 对目标变更进行显式审批。
5. trace 中记录目标变化。
6. 定期让总控 Agent 对齐原始目标。

## 31.6 失败模式五：重复工作和成本爆炸

多 Agent 很容易重复做同一件事。

例如：

1. 数据 Agent 查了一遍数据库。
2. 报告 Agent 不信，又请求另一个数据 Agent 查一遍。
3. 审核 Agent 为了验证，再查一遍。
4. 总控 Agent 汇总时又重新查询。

结果是成本、延迟和系统负载快速上升。

防护方式包括：

1. 共享只读 Artifact。
2. 对相同查询做缓存。
3. 使用 trace 检查已有证据。
4. 设置任务成本预算。
5. 限制并行 Agent 数量。
6. 对高成本工具做审批。
7. 设置最大工具调用次数。

多 Agent 系统的成本控制必须设计在协议和 runtime 里，不能只靠事后统计账单。

## 31.7 失败模式六：状态不一致

状态不一致指不同 Agent 对同一任务状态理解不同。

例如：

```text
总控 Agent：任务 running。
下游 Agent：任务 completed。
UI：任务 failed。
审计系统：没有收到最终事件。
```

原因可能是：

1. 事件丢失。
2. 回调失败。
3. 重试导致重复事件。
4. 状态机定义不一致。
5. Agent 崩溃后恢复不完整。
6. completed 和 partial_completed 混淆。

防护方式：

1. 明确定义状态机。
2. 状态事件带 version 或 sequence number。
3. 事件处理幂等。
4. 支持状态查询补偿。
5. 对终态做不可逆约束。
6. 定期做状态 reconcile。

状态事件可以带序号：

```json
{
  "task_id": "task_123",
  "event_seq": 7,
  "status": "completed",
  "timestamp": "2026-05-29T12:30:00Z"
}
```

如果系统收到 event_seq 更小的旧事件，就不应该覆盖新状态。

## 31.8 失败模式七：重复写和产物冲突

当多个 Agent 可以修改同一个对象时，容易发生冲突。

例如：

1. 代码 Agent 修改同一个文件。
2. 报告 Agent 同时编辑同一份报告。
3. 测试 Agent 自动修复测试快照。
4. 格式化 Agent 又重写文件。

防护方式包括：

1. 单写者原则。
2. 文件锁或 Artifact 锁。
3. Patch 形式提交。
4. 合并前冲突检测。
5. 人工确认最终合并。
6. 变更版本号和 content_hash。

在代码场景中，不要让多个 Agent 直接写工作区。更好的方式是每个 Agent 生成 patch Artifact，由总控 Agent 或人类统一合并。

## 31.9 失败模式八：责任模糊

多 Agent 系统失败后，如果没有 trace，很难知道责任在哪。

例如最终报告错误，可能原因是：

1. 数据 Agent 查询错。
2. 知识库 Agent 返回了过期定义。
3. 报告 Agent 摘要时改写错。
4. 总控 Agent 忽略了低置信度标记。
5. 用户提供的原始需求不完整。

防护机制是：

1. 每个 Agent 输出带 evidence。
2. 每个结论带 produced_by。
3. 每次摘要保留来源。
4. 每个 Artifact 带创建者和 hash。
5. trace 串联任务和工具调用。
6. 审计记录授权和人工确认。

责任不是为了“甩锅”，而是为了定位、修复和改进系统。

## 31.10 失败模式九：安全策略被协作链绕过

有些越权不是直接发生的，而是通过多 Agent 间接发生的。

例如：

```text
用户无权访问明细数据。
总控 Agent 委派给数据 Agent 查询聚合。
数据 Agent 返回明细样本给报告 Agent。
报告 Agent 在最终报告里泄露明细。
```

单看每一步可能都不明显，但整体发生了数据泄露。

防护方式：

1. 数据分类随上下文传播。
2. 工具结果进入上下文前脱敏。
3. context_policy 强制执行。
4. Artifact 访问权限独立校验。
5. 最终输出做安全检查。
6. 高风险跨 Agent 转发要求确认。

## 31.11 失败模式十：过度协作

不是所有任务都需要多 Agent。

有些系统把简单任务也拆给多个 Agent：

```text
用户：总结这段文字。
总控 Agent -> 摘要 Agent -> 审核 Agent -> 风格 Agent -> 质量 Agent
```

结果是延迟高、成本高、错误点变多，收益很小。

防护方式是任务分级：

1. 简单任务单 Agent 处理。
2. 中等任务少量专家 Agent。
3. 高风险复杂任务才多 Agent 协作。
4. 对每次委派要求收益理由。
5. 设置最大 Agent 数量和最大轮次。

多 Agent 不是越多越好。每增加一个 Agent，就增加一次上下文传递、一次状态同步、一次失败可能。

## 31.12 防护总表

| 失败模式 | 典型表现 | 防护机制 |
| --- | --- | --- |
| 循环委派 | Agent 互相委派，任务不结束 | 最大深度、visited_agents、任务 DAG、去重 |
| 角色冲突 | 多个 Agent 给出互斥方案 | 仲裁规则、领域权威、人类审批 |
| 幻觉传播 | 假设被升级为事实 | claim_type、evidence、confidence、验证 |
| 上下文漂移 | 原始目标和约束丢失 | root goal、preserved constraints、目标变更审批 |
| 成本爆炸 | 重复查询、重复分析 | 缓存、预算、工具调用上限、Artifact 复用 |
| 状态不一致 | 各系统状态不同 | 状态机、事件序号、幂等、reconcile |
| 产物冲突 | 多 Agent 同时写同一对象 | patch、锁、单写者、合并检查 |
| 责任模糊 | 出错后无法定位 | trace、produced_by、evidence、audit |
| 策略绕过 | 数据经协作链泄露 | context_policy、数据分类、脱敏、最终检查 |
| 过度协作 | 简单任务被复杂拆分 | 任务分级、委派收益判断、轮次限制 |

## 31.13 一个完整故障案例

假设企业有一个多 Agent 报告系统。用户要求：

```text
分析上周退款率升高原因，只使用聚合数据，不要访问用户明细。
```

系统实际流程：

1. 总控 Agent 委派给数据 Agent。
2. 数据 Agent 查询聚合数据，但发现样本不足。
3. 数据 Agent 自行查询了部分用户明细。
4. 报告 Agent 接收数据 Agent 的摘要，但摘要中删除了“样本不足”说明。
5. 报告 Agent 写出“退款率升高由某类用户群导致”。
6. 决策 Agent 建议调整该用户群策略。
7. 最终报告对业务产生误导。

这个故障包含多个失败模式：

1. 数据策略被绕过。
2. 上下文摘要丢失限制条件。
3. 假设被写成事实。
4. 总控 Agent 没有验证关键结论。
5. 最终输出没有检查数据来源。

改进方案：

1. context_policy 明确 aggregate_only，并在数据库工具层强制执行。
2. 数据 Agent 不能自行升级到用户明细。
3. 所有 claim 必须带 claim_type、confidence 和 evidence。
4. 报告 Agent 必须保留 limitations。
5. 决策建议前需要验证或人工确认。
6. 审计日志记录每次资源访问和策略判断。

## 31.14 监控与评估指标

多 Agent 系统需要专门指标。

常见指标包括：

1. 平均委派深度。
2. 最大委派深度。
3. 每个 root task 的 Agent 数量。
4. 每个任务的工具调用次数。
5. input_required 次数。
6. 任务循环检测次数。
7. 状态 reconcile 次数。
8. Artifact 冲突次数。
9. 人工接管次数。
10. 结果被验证通过率。
11. 高风险动作确认率。
12. 幻觉传播拦截率。

这些指标比单纯看最终成功率更有用，因为它们能揭示系统是否正在“看似成功但内部失控”。

## 31.15 多 Agent 失败审计指标与最小 demo

上一节列出的是监控项。本节把它们进一步变成可计算的上线门禁，方便面试时说明“我不是靠感觉判断多 Agent 是否可靠，而是能把失败模式落到 trace 字段和指标上”。

先定义一个 root task 级别的审计样本：

```math
r_i=(g_i,d_i,c_i,h_i,q_i,w_i,s_i,a_i,p_i,b_i,t_i,z_i)
```

其中，`g_i` 是原始目标和最终目标，`d_i` 是委派图和深度，`c_i` 是冲突和仲裁记录，`h_i` 是 claim / evidence / confidence / limitations，`q_i` 是上下文约束，`w_i` 是重复工作和成本预算，`s_i` 是状态事件，`a_i` 是 Artifact 写入和合并记录，`p_i` 是策略链，`b_i` 是协作规模，`t_i` 是 trace / audit 字段，`z_i` 是 eval label。

对任意审计项 `k`，定义通过率：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[I_k(r_i)=1]
```

其中，`I_k(r_i)=1` 表示第 `i` 个 root task 在第 `k` 个检查项上通过，`N` 是审计样本数。

多 Agent 失败治理门禁可以写成：

```math
G_{\mathrm{multi\_agent\_failure}}
=\prod_{k\in\mathcal{K}}\mathbb{1}[C_k\ge \tau_k]
```

其中，`\mathcal{K}` 是循环、冲突、幻觉传播、上下文漂移、成本、状态、产物、责任、安全策略、过度协作、终止接管和 eval 覆盖等检查项集合，`\tau_k` 是每项上线阈值。

如果要把它做成排序分，也可以用加权形式：

```math
S_{\mathrm{multi\_agent\_failure}}
=\sum_{k\in\mathcal{K}}w_k C_k,\qquad
\sum_{k\in\mathcal{K}}w_k=1
```

这里的关键不是公式复杂，而是把“多 Agent 看起来在协作”拆成可审计证据：有没有环，冲突有没有仲裁，假设有没有被升级成事实，原始约束有没有丢，重复工具调用有没有预算，状态事件有没有乱序，Artifact 有没有多写者冲突，最终输出有没有绕过安全策略。

下面是一个 0 依赖 demo。它不调用任何模型，只用 toy trace 演示如何把多 Agent 失败模式转成指标。

```python
from collections import OrderedDict

REQUIRED_TRACE = {
    "trace_id", "root_task_id", "task_id", "caller", "assignee",
    "decision", "evidence", "status", "policy", "cost",
}
SAFE_ARBITRATION = {"risk_first", "domain_owner", "human_review"}


def make_case(**overrides):
    base = {
        "id": "refund_root_cause_ok",
        "root_goal": "explain_refund_rate",
        "final_goal": "explain_refund_rate",
        "preserved_constraints": {"aggregate_only", "no_prod_change"},
        "required_constraints": {"aggregate_only", "no_prod_change"},
        "goal_change_approved": False,
        "edges": [("orchestrator", "data"), ("data", "report")],
        "max_depth": 4,
        "visited_repeat": False,
        "duplicate_task": False,
        "termination_condition": True,
        "human_handoff": False,
        "conflict": False,
        "arbitration": None,
        "arbitration_recorded": True,
        "claims": [
            {
                "type": "hypothesis",
                "evidence": ["artifact://refund_trend"],
                "confidence": "medium",
                "limitations": ["correlation_only"],
                "upgraded": False,
                "critical_verified": True,
            }
        ],
        "duplicate_calls": 0,
        "cache_used": True,
        "tool_call_count": 3,
        "max_tool_calls": 6,
        "budget_ok": True,
        "events": [(1, "submitted"), (2, "working"), (3, "completed")],
        "terminal_consistent": True,
        "artifact_writers": ["report"],
        "artifact_lock": True,
        "conflict_detected": True,
        "merge_owner": "orchestrator",
        "trace_fields": REQUIRED_TRACE,
        "classification_propagated": True,
        "allowed_detail": "aggregate",
        "actual_detail": "aggregate",
        "policy_final_check": True,
        "policy_bypass": False,
        "task_complexity": "complex",
        "agent_count": 3,
        "max_agent_count": 5,
        "delegation_reason": "data_plus_report",
        "eval_label": "pass",
        "scenario_tags": {"refund", "multi_agent"},
    }
    base.update(overrides)
    return base


def has_cycle(edges):
    graph = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
    visiting, visited = set(), set()

    def dfs(node):
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in graph.get(node, []):
            if dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(dfs(node) for node in graph)


def depth(edges):
    return len(edges)


def detail_rank(level):
    return {"none": 0, "aggregate": 1, "sample": 2, "row": 3, "secret": 4}[level]


def loop_control(case):
    return (
        not has_cycle(case["edges"])
        and depth(case["edges"]) <= case["max_depth"]
        and not case["visited_repeat"]
        and not case["duplicate_task"]
        and case["termination_condition"]
    )


def conflict_arbitration(case):
    if not case["conflict"]:
        return True
    return case["arbitration"] in SAFE_ARBITRATION and case["arbitration_recorded"]


def hallucination_containment(case):
    for claim in case["claims"]:
        if claim["type"] not in {"fact", "hypothesis", "inference", "recommendation"}:
            return False
        if not claim["evidence"] or claim["upgraded"]:
            return False
        if claim["type"] in {"hypothesis", "inference", "recommendation"} and not claim["limitations"]:
            return False
        if claim["type"] == "recommendation" and not claim["critical_verified"]:
            return False
    return True


def context_drift_control(case):
    constraints_ok = case["required_constraints"].issubset(case["preserved_constraints"])
    same_goal = case["root_goal"] == case["final_goal"]
    return constraints_ok and (same_goal or case["goal_change_approved"])


def duplicate_work_budget(case):
    return (
        case["budget_ok"]
        and case["tool_call_count"] <= case["max_tool_calls"]
        and (case["duplicate_calls"] == 0 or case["cache_used"])
    )


def state_consistency(case):
    seqs = [seq for seq, _ in case["events"]]
    return seqs == sorted(set(seqs)) and case["terminal_consistent"]


def artifact_conflict_control(case):
    return (
        len(set(case["artifact_writers"])) <= 1
        or (case["artifact_lock"] and case["conflict_detected"] and bool(case["merge_owner"]))
    )


def accountability_trace(case):
    return REQUIRED_TRACE.issubset(case["trace_fields"])


def policy_chain_enforcement(case):
    return (
        case["classification_propagated"]
        and detail_rank(case["actual_detail"]) <= detail_rank(case["allowed_detail"])
        and case["policy_final_check"]
        and not case["policy_bypass"]
    )


def collaboration_fit(case):
    if case["task_complexity"] == "simple":
        return case["agent_count"] <= 1
    return case["agent_count"] <= case["max_agent_count"] and bool(case["delegation_reason"])


def termination_handoff_readiness(case):
    high_risk = case["conflict"] or has_cycle(case["edges"]) or case["policy_bypass"]
    return case["termination_condition"] and (not high_risk or case["human_handoff"])


def failure_eval_coverage(case):
    return bool(case["eval_label"]) and bool(case["scenario_tags"])


CHECKS = OrderedDict([
    ("delegation_loop_control", loop_control),
    ("role_conflict_arbitration", conflict_arbitration),
    ("hallucination_containment", hallucination_containment),
    ("context_drift_control", context_drift_control),
    ("duplicate_work_budget", duplicate_work_budget),
    ("state_consistency", state_consistency),
    ("artifact_conflict_control", artifact_conflict_control),
    ("accountability_trace", accountability_trace),
    ("policy_chain_enforcement", policy_chain_enforcement),
    ("collaboration_fit", collaboration_fit),
    ("termination_handoff_readiness", termination_handoff_readiness),
    ("failure_eval_coverage", failure_eval_coverage),
])

CASES = [
    make_case(),
    make_case(id="simple_summary_single_agent_ok", task_complexity="simple", agent_count=1, delegation_reason=""),
    make_case(id="loop_detected_bad", edges=[("orchestrator", "data"), ("data", "kb"), ("kb", "data")], visited_repeat=True, human_handoff=False, scenario_tags={"loop"}),
    make_case(id="depth_overflow_bad", edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")], max_depth=3, human_handoff=True, scenario_tags={"depth"}),
    make_case(id="role_conflict_no_arbitration_bad", conflict=True, arbitration="majority_vote", arbitration_recorded=False, human_handoff=False, scenario_tags={"conflict"}),
    make_case(id="hallucination_upgraded_bad", claims=[{"type": "fact", "evidence": ["artifact://trend"], "confidence": "high", "limitations": [], "upgraded": True, "critical_verified": False}], scenario_tags={"hallucination"}),
    make_case(id="context_drift_bad", final_goal="rollback_policy", preserved_constraints={"aggregate_only"}, scenario_tags={"drift"}),
    make_case(id="duplicate_work_budget_bad", duplicate_calls=4, cache_used=False, tool_call_count=11, max_tool_calls=6, budget_ok=False, scenario_tags={"cost"}),
    make_case(id="state_reconcile_bad", events=[(1, "submitted"), (3, "completed"), (2, "working")], terminal_consistent=False, scenario_tags={"state"}),
    make_case(id="artifact_conflict_bad", artifact_writers=["code", "test", "formatter"], artifact_lock=False, conflict_detected=False, merge_owner="", scenario_tags={"artifact"}),
    make_case(id="trace_missing_bad", trace_fields={"trace_id", "task_id", "status"}, scenario_tags={"trace"}),
    make_case(id="policy_bypass_bad", actual_detail="row", allowed_detail="aggregate", classification_propagated=False, policy_final_check=False, policy_bypass=True, human_handoff=False, scenario_tags={"policy"}),
    make_case(id="over_collaboration_bad", task_complexity="simple", agent_count=4, delegation_reason="style_review_chain", scenario_tags={"over_collaboration"}),
    make_case(id="eval_missing_bad", eval_label="", scenario_tags=set()),
]

metrics = OrderedDict()
failed_by_case = OrderedDict()
for name, fn in CHECKS.items():
    passes = [fn(case) for case in CASES]
    metrics[name] = round(sum(passes) / len(passes), 3)
    for case, ok in zip(CASES, passes):
        if not ok:
            failed_by_case.setdefault(case["id"], []).append(name)

thresholds = {name: 0.95 for name in CHECKS}
failed_gates = [name for name, value in metrics.items() if value < thresholds[name]]

smoke = OrderedDict([
    ("simple_task_not_overdelegated", collaboration_fit(CASES[1])),
    ("caught_loop", not loop_control(CASES[2])),
    ("caught_hallucination_upgrade", not hallucination_containment(CASES[5])),
    ("caught_policy_bypass", not policy_chain_enforcement(CASES[11])),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_cases=", list(failed_by_case.keys()))
print("failed_gates=", failed_gates)
print("multi_agent_failure_gate_pass=", not failed_gates)
```

运行后可以看到：

```text
smoke= {'simple_task_not_overdelegated': True, 'caught_loop': True, 'caught_hallucination_upgrade': True, 'caught_policy_bypass': True}
metrics= {'delegation_loop_control': 0.857, 'role_conflict_arbitration': 0.929, 'hallucination_containment': 0.929, 'context_drift_control': 0.929, 'duplicate_work_budget': 0.929, 'state_consistency': 0.929, 'artifact_conflict_control': 0.929, 'accountability_trace': 0.929, 'policy_chain_enforcement': 0.929, 'collaboration_fit': 0.929, 'termination_handoff_readiness': 0.786, 'failure_eval_coverage': 0.929}
failed_cases= ['loop_detected_bad', 'depth_overflow_bad', 'role_conflict_no_arbitration_bad', 'hallucination_upgraded_bad', 'context_drift_bad', 'duplicate_work_budget_bad', 'state_reconcile_bad', 'artifact_conflict_bad', 'trace_missing_bad', 'policy_bypass_bad', 'over_collaboration_bad', 'eval_missing_bad']
failed_gates= ['delegation_loop_control', 'role_conflict_arbitration', 'hallucination_containment', 'context_drift_control', 'duplicate_work_budget', 'state_consistency', 'artifact_conflict_control', 'accountability_trace', 'policy_chain_enforcement', 'collaboration_fit', 'termination_handoff_readiness', 'failure_eval_coverage']
multi_agent_failure_gate_pass= False
```

这段 demo 的面试价值在于：它把“多 Agent 失控”拆成了十二个可复现检查项。真正上线时，不一定沿用这些 toy 字段，但要保留同样的治理思想：每个 root task 都要能回答有没有循环、有没有冲突、证据是否保留、目标是否漂移、成本是否超预算、状态是否一致、产物是否冲突、谁负责、策略是否被绕过、是否过度协作，以及失败样本是否进入 eval 集。

## 31.16 面试高频题

### 题 1：多 Agent 系统有哪些典型失败模式？

参考回答：

典型失败模式包括循环委派、角色冲突、幻觉传播、上下文漂移、重复工作和成本爆炸、状态不一致、产物冲突、责任模糊、安全策略被绕过以及过度协作。它们本质上来自多个 Agent 之间状态、上下文、权限和目标的不一致。

### 题 2：如何防止循环委派？

参考回答：

可以设置最大委派深度、维护 parent_task_id 和 root_task_id、把任务图限制为 DAG、记录 visited_agents、使用 idempotency_key 去重、检测相似任务重复提交，并限制下游 Agent 继续委派权限。超过轮次后应升级人工接管。

### 题 3：如何防止幻觉在 Agent 间传播？

参考回答：

Agent 输出必须区分 fact、hypothesis、inference 和 recommendation，并保留 evidence、confidence、limitations。关键结论需要二次验证或独立来源交叉验证。高风险建议不能直接执行，必须人工确认或通过策略检查。

### 题 4：多 Agent 冲突如何仲裁？

参考回答：

可以用总控 Agent、规则优先、领域权威 Agent、人类审批、证据权重、风险优先或多数投票来仲裁。安全和合规场景不适合简单多数投票，应该风险优先并支持人工复核。

### 题 5：为什么多 Agent 不一定比单 Agent 好？

参考回答：

多 Agent 增加了通信、状态同步、上下文传递、权限治理、冲突仲裁和成本控制复杂度。对于简单任务，单 Agent 更低延迟、更少错误点。只有任务确实需要专业分工、并行处理、异步执行或跨组织协作时，多 Agent 才有明显收益。

## 31.17 小练习

1. 设计一个循环委派检测字段集合，包括 root_task_id、parent_task_id、delegation_depth 和 visited_agents。
2. 为一个“退款率根因分析”任务写 3 条结构化 claim，其中至少一条是 hypothesis。
3. 设计一个冲突仲裁规则：当安全 Agent 和产品 Agent 结论冲突时，系统如何处理？
4. 列出一个多 Agent 系统的 5 个监控指标。
5. 思考：一个简单摘要任务是否需要 3 个 Agent 协作？什么时候值得引入多个 Agent？
6. 运行本章 demo，把 `policy_bypass_bad` 改成只返回聚合数据，并观察 `policy_chain_enforcement` 和 `termination_handoff_readiness` 如何变化。
7. 给 demo 增加一个 `shared_wrong_source_bad` 样本：多个 Agent 都引用同一个错误来源，但没有独立交叉验证。你会把它归入哪个检查项？

## 31.18 本章小结

本章我们讲了多 Agent 系统的典型失败模式。

多 Agent 的价值在于分工、并行和专业化，但它同时引入循环委派、角色冲突、幻觉传播、上下文漂移、成本爆炸、状态不一致、产物冲突、责任模糊和安全策略绕过等问题。解决这些问题不能只靠 prompt，而要靠协议、状态机、权限系统、trace、审计、仲裁机制、成本预算和人工接管。

你可以把本章重点记成一句话：

> 多 Agent 系统不是把多个模型连起来就会变强，真正的工程能力体现在如何限制、验证、仲裁和回收失控的协作链。

下一章我们会用系统设计面试题的形式，把 A2A 部分串起来，训练如何完整回答一个多 Agent 协作平台设计题。
