# 第 49 章 设计一个跨 Agent 协作系统

前面我们已经讲过 A2A 的背景、Agent Card、任务委派、状态同步、消息格式、上下文边界、权限、信任模型和失败模式。

本章把这些内容组合成一道系统设计题：如何设计一个跨 Agent 协作系统？

这道题比“设计一个工具平台”更复杂，因为工具通常是被动执行者，而 Agent 是有目标、状态、推理过程、工具权限和失败模式的协作者。多个 Agent 协作时，系统不只是转发消息，还要管理任务、上下文、权限、冲突、审计和结果可信度。

先记住一句话：

> 跨 Agent 协作系统的核心不是“让很多 Agent 聊天”，而是让不同能力边界的 Agent 在可控上下文、可追踪任务和可治理权限下完成协同工作。

## 49.1 面试题描述

题目可以这样问：

```text
请设计一个跨 Agent 协作系统。系统中有多个专业 Agent，例如需求分析 Agent、代码 Agent、测试 Agent、数据分析 Agent、客服 Agent、知识库 Agent 和审批 Agent。用户提交复杂任务后，系统需要自动拆解任务，选择合适 Agent，完成任务委派、状态同步、结果聚合、权限控制、审计和失败恢复。
```

这道题不要直接回答“用一个 orchestrator 调其他 Agent”。这只是最粗的一层。

真正要讲的是：

1. Agent 如何声明能力？
2. 系统如何发现和选择 Agent？
3. 任务如何拆解、委派、跟踪和终止？
4. Agent 之间传什么上下文？
5. 谁能看什么数据、调用什么工具？
6. 多 Agent 冲突、循环、幻觉传播怎么防？
7. 如何审计、回放和评估？

## 49.2 需求澄清

面试里可以先问：

1. 这是企业内部系统，还是开放平台？
2. Agent 数量是几十个，还是成千上万个？
3. Agent 是同一个团队开发，还是第三方开发？
4. 是否需要跨租户隔离？
5. 是否有高风险动作，例如发邮件、改代码、下单、退款、执行命令？
6. 是否需要人工审批？
7. 是否需要实时协作，还是异步任务即可？
8. 是否需要接入 MCP 工具平台？

如果要给出假设，可以这样说：

```text
我假设这是企业内部跨 Agent 协作平台，服务研发、客服、数据分析和运营等场景。系统支持多个专业 Agent，通过 Agent Card 声明能力，通过 A2A Runtime 做任务委派和状态同步。平台需要权限控制、上下文隔离、任务追踪、人工确认、失败恢复、审计回放和评估监控。
```

## 49.3 核心目标

跨 Agent 协作系统要解决十个问题：

1. 能力发现：知道有哪些 Agent 能做什么。
2. 任务拆解：把复杂目标拆成可执行子任务。
3. Agent 选择：选择合适的执行 Agent。
4. 上下文传递：只传必要信息，不泄露全部上下文。
5. 状态同步：知道任务现在处于什么阶段。
6. 结果聚合：把多个 Agent 的产出合并成最终答案。
7. 权限治理：控制每个 Agent 能看、能做、能调用什么。
8. 失败恢复：处理超时、拒绝、失败、冲突和循环。
9. 审计回放：事后能解释谁做了什么。
10. 评估优化：持续评估协作质量和成本。

这十个问题对应系统的主要模块。

## 49.4 总体架构

一个合理架构如下：

```text
User / Business App
  -> Agent Orchestrator
  -> Planner / Task Decomposer
  -> Agent Registry / Agent Card Store
  -> A2A Runtime
      -> Agent A: Requirement Agent
      -> Agent B: Coding Agent
      -> Agent C: Test Agent
      -> Agent D: Data Agent
      -> Agent E: Approval Agent
  -> Context Manager
  -> Policy Engine
  -> Tool / MCP Gateway
  -> Trace / Audit / Replay / Metrics
  -> Human Review Console
```

模块解释：

1. User / Business App：用户入口，可以是聊天界面、IDE、工单系统或业务后台。
2. Agent Orchestrator：协作中枢，负责计划、调度、状态推进和结果汇总。
3. Planner / Task Decomposer：把复杂任务拆成子任务和依赖图。
4. Agent Registry：保存 Agent Card，支持能力发现。
5. A2A Runtime：提供 Agent 间任务、消息、状态、artifact 的标准交换机制。
6. Context Manager：控制上下文裁剪、脱敏、引用和隔离。
7. Policy Engine：身份、权限、风险、审批、租户隔离。
8. Tool / MCP Gateway：给 Agent 提供受控工具调用能力。
9. Trace / Audit / Replay：记录全链路过程。
10. Human Review Console：处理高风险动作、冲突和失败接管。

## 49.5 为什么不能只让 Agent 互相发消息

很多人会把跨 Agent 协作理解成“Agent A 调 Agent B，然后 Agent B 回一句话”。这在 demo 里可以，在企业系统里不够。

原因有五个：

1. 没有任务状态，无法知道子任务是 pending、running、blocked 还是 failed。
2. 没有上下文边界，一个 Agent 可能拿到不该看的信息。
3. 没有权限传递，无法判断 Agent 是否能代表用户执行动作。
4. 没有结果证据，最终答案无法追溯来源。
5. 没有失败控制，多个 Agent 容易互相甩锅、循环调用或传播错误结论。

所以跨 Agent 系统需要任务协议，而不是简单聊天协议。

## 49.6 Agent Card 设计

Agent Card 是 Agent 的能力声明。它像 Agent 的“服务名片”。

一个 Agent Card 至少包含：

1. agent_id。
2. name。
3. owner。
4. description。
5. capabilities。
6. input_types。
7. output_types。
8. supported_tasks。
9. tools_required。
10. permissions_required。
11. context_policy。
12. risk_level。
13. latency_slo。
14. cost_profile。
15. version。
16. health_status。

示例：

```json
{
  "agent_id": "agent.coding.backend",
  "name": "Backend Coding Agent",
  "description": "Implements backend changes and produces patches.",
  "capabilities": ["code_edit", "unit_test", "api_design"],
  "input_types": ["requirement_doc", "bug_report", "repo_context"],
  "output_types": ["patch", "test_report", "implementation_note"],
  "supported_tasks": ["fix_bug", "add_api", "refactor_module"],
  "tools_required": ["repo.read", "repo.write", "test.run"],
  "permissions_required": ["code:read", "code:write", "ci:run"],
  "context_policy": {
    "max_tokens": 24000,
    "allow_sensitive_data": false,
    "requires_source_citations": true
  },
  "risk_level": "high",
  "latency_slo_ms": 120000,
  "version": "1.4.2"
}
```

面试中要强调：Agent Card 不是宣传文案，而是调度、权限和评估的输入。

## 49.7 Agent Registry

Agent Registry 负责存储和查询 Agent Card。

它需要支持：

1. 按能力检索 Agent。
2. 按任务类型检索 Agent。
3. 按租户和权限过滤 Agent。
4. 按健康状态过滤 Agent。
5. 按成本、延迟、成功率排序。
6. 支持版本、灰度和下线。
7. 支持审核状态。

典型查询：

```text
Find agents where:
  capability contains "unit_test"
  task_type = "validate_patch"
  tenant = "team_a"
  required_permission subset of user_permission
  health = "healthy"
order by success_rate desc, latency p95 asc
```

如果 Registry 只按关键词搜索，就很容易选错 Agent。更好的方式是把能力结构化，同时保留语义检索作为补充。

## 49.8 Planner 与任务拆解

复杂任务通常不能一次交给一个 Agent。

例如用户说：

```text
帮我分析最近订单转化率下降的原因，给出修复建议，并如果需要修改推荐服务的配置，请生成变更方案。
```

Planner 可以拆成：

1. 数据 Agent 查询转化率指标。
2. 知识库 Agent 查最近业务变更记录。
3. 代码 Agent 检查推荐服务最近提交。
4. 分析 Agent 汇总可能原因。
5. 审批 Agent 判断是否允许生成配置变更。
6. Orchestrator 汇总最终报告。

任务拆解结果不应该只是一段自然语言，而应该是任务图：

```json
{
  "root_task_id": "task_001",
  "subtasks": [
    {
      "task_id": "task_001_a",
      "type": "query_metrics",
      "agent_capability": "data_analysis",
      "depends_on": [],
      "status": "pending"
    },
    {
      "task_id": "task_001_b",
      "type": "search_change_log",
      "agent_capability": "knowledge_search",
      "depends_on": [],
      "status": "pending"
    },
    {
      "task_id": "task_001_c",
      "type": "root_cause_analysis",
      "agent_capability": "business_analysis",
      "depends_on": ["task_001_a", "task_001_b"],
      "status": "pending"
    }
  ]
}
```

任务图有几个好处：

1. 能并发执行无依赖子任务。
2. 能明确阻塞关系。
3. 能单独重试失败子任务。
4. 能审计每个子任务的输入、输出和责任 Agent。

## 49.9 A2A Runtime 的核心对象

A2A Runtime 至少要抽象四类对象：

1. Task：任务，表示一个可追踪工作单元。
2. Message：消息，表示 Agent 间交流。
3. Status：状态，表示任务生命周期。
4. Artifact：产物，表示可引用结果。

Task 示例：

```json
{
  "task_id": "task_001_c",
  "parent_task_id": "task_001",
  "requester_agent": "agent.orchestrator",
  "assignee_agent": "agent.business.analysis",
  "task_type": "root_cause_analysis",
  "input_refs": ["artifact_metrics_001", "artifact_changelog_001"],
  "status": "running",
  "deadline_ms": 60000,
  "risk_level": "medium"
}
```

Message 示例：

```json
{
  "message_id": "msg_001",
  "task_id": "task_001_c",
  "from": "agent.orchestrator",
  "to": "agent.business.analysis",
  "role": "request",
  "content": "Analyze possible causes using the referenced metrics and changelog.",
  "context_refs": ["artifact_metrics_001", "artifact_changelog_001"]
}
```

Artifact 示例：

```json
{
  "artifact_id": "artifact_analysis_001",
  "task_id": "task_001_c",
  "type": "analysis_report",
  "content_ref": "object_store://reports/task_001_c/report.md",
  "source_refs": ["artifact_metrics_001", "artifact_changelog_001"],
  "confidence": 0.72,
  "created_by": "agent.business.analysis"
}
```

关键点：Agent 之间不应该只传大段文本，还要传结构化引用、状态和产物。

## 49.10 任务生命周期

一个任务可以有如下状态：

```text
created -> planned -> assigned -> running -> waiting_for_input -> waiting_for_approval -> completed
                                      -> failed
                                      -> cancelled
                                      -> timed_out
```

常见状态解释：

1. created：用户或上游 Agent 创建任务。
2. planned：任务已被拆解。
3. assigned：已分配给某个 Agent。
4. running：Agent 正在处理。
5. waiting_for_input：缺少必要信息。
6. waiting_for_approval：高风险动作等待人工确认。
7. completed：完成并产生 artifact。
8. failed：失败，可重试或转人工。
9. cancelled：被用户或系统取消。
10. timed_out：超过截止时间。

状态机很重要，因为它决定了系统能否可靠恢复。没有状态机，多 Agent 协作会变成不可控的长对话。

## 49.11 Agent 选择策略

Agent 选择不应该只看“名字像不像”。

可以综合以下因素：

1. capability match：能力是否匹配。
2. input/output match：输入输出格式是否兼容。
3. permission match：权限是否允许。
4. tenant match：租户是否可见。
5. health：是否健康。
6. success rate：历史成功率。
7. latency：延迟是否满足任务要求。
8. cost：成本是否可接受。
9. risk level：风险等级是否适配。
10. version：是否使用稳定版本。

伪代码：

```text
candidates = registry.find_by_capability(required_capability)
candidates = filter_by_tenant(candidates, tenant)
candidates = filter_by_permission(candidates, user, task)
candidates = filter_by_health(candidates)
candidates = rank(candidates, success_rate, latency, cost, risk)
selected = candidates[0]
```

如果任务风险高，可以选择两个 Agent 并行执行，再让审查 Agent 交叉验证。

## 49.12 上下文边界设计

跨 Agent 协作最容易出问题的是上下文泄露。

例如用户让 HR Agent 分析候选人简历，然后又让 Coding Agent 生成面试题。Coding Agent 不一定需要看到候选人的手机号、身份证号和薪资期望。

Context Manager 要做几件事：

1. 从全局任务上下文中抽取子任务所需信息。
2. 根据 Agent Card 的 context_policy 做裁剪。
3. 根据数据分类做脱敏。
4. 把原始数据改成可追踪引用。
5. 标注哪些内容是用户指令，哪些是工具结果，哪些是不可信外部数据。
6. 控制上下文 token 预算。

一个上下文包可以设计成：

```json
{
  "context_id": "ctx_001",
  "task_id": "task_001_c",
  "audience_agent": "agent.business.analysis",
  "instructions": [
    {
      "source": "orchestrator",
      "trust_level": "system",
      "content": "Analyze causes and cite evidence."
    }
  ],
  "data_refs": [
    {
      "artifact_id": "artifact_metrics_001",
      "trust_level": "tool_output",
      "classification": "internal",
      "allowed_usage": "analysis_only"
    }
  ],
  "redactions": ["personal_phone", "salary_expectation"]
}
```

面试中可以强调一句：

> 多 Agent 系统里，上下文不是共享内存，而是经过授权、裁剪和标注的任务输入。

## 49.13 权限与身份模型

跨 Agent 权限比单 Agent 更难，因为每个 Agent 都可能调用工具或委派任务。

需要区分三种身份：

1. user identity：原始用户身份。
2. agent identity：执行 Agent 身份。
3. service identity：平台服务身份。

常见授权模型是 OBO，也就是 on-behalf-of。Agent 代表用户执行动作，但不能获得超过用户本身或 Agent 自身策略允许的权限。

有效权限可以理解成：

```text
effective_permission = intersection(
  user_permission,
  agent_allowed_permission,
  task_policy,
  tenant_policy,
  risk_policy
)
```

例如用户有代码写权限，但某个数据分析 Agent 不允许写代码，那么它不能代表用户改代码。

高风险动作要额外要求：

1. 显式用户确认。
2. 审批 Agent 或人工审批。
3. 参数摘要和影响范围展示。
4. 幂等 key。
5. 审计记录。

## 49.14 Policy Engine

Policy Engine 负责判断“这个 Agent 在这个任务上下文里能不能做这件事”。

它可以检查：

1. Agent 是否可被当前租户使用。
2. Agent 是否允许接收该数据类型。
3. Agent 是否允许调用目标工具。
4. 当前用户是否有权限。
5. 动作风险是否需要审批。
6. 是否违反数据出境或数据驻留要求。
7. 是否超过成本和调用频率限制。
8. 是否存在循环委派风险。

策略决策结果最好是结构化的：

```json
{
  "decision": "allow_with_approval",
  "reason": "The action modifies production configuration.",
  "required_approval": "human",
  "redactions": ["user_email"],
  "max_tool_calls": 5
}
```

不要只返回 true / false。真实系统需要解释原因和后续动作。

## 49.15 与 MCP 工具平台的关系

A2A 管 Agent 之间的协作，MCP 管 Agent 与工具/资源的连接。

在跨 Agent 系统里，Agent 可能需要调用工具，例如：

1. 数据 Agent 通过 MCP 查询数据库。
2. 代码 Agent 通过 MCP 访问代码仓库。
3. 测试 Agent 通过 MCP 运行测试。
4. 知识库 Agent 通过 MCP 搜索文档。
5. 浏览器 Agent 通过 MCP 操作网页。

推荐架构是：

```text
Agent Orchestrator
  -> A2A Runtime
      -> Specialized Agent
          -> MCP Gateway
              -> MCP Server
                  -> Real Tool / Resource
```

这样分层后：

1. A2A 负责谁和谁协作。
2. MCP 负责 Agent 怎么使用工具和资源。
3. Policy Engine 横跨两层，统一做权限和审计。

不要把 A2A 和 MCP 混成一个协议，否则系统会很难治理。

## 49.16 结果聚合与冲突处理

多个 Agent 返回结果后，Orchestrator 不能简单拼接。

结果聚合需要处理：

1. 去重：多个 Agent 给出相同结论。
2. 合并：不同 Agent 给出互补信息。
3. 冲突：两个 Agent 结论相反。
4. 证据：每个结论是否有来源。
5. 置信度：结果是否足够可靠。
6. 时效性：数据是否过期。
7. 风险：是否涉及高风险建议。

冲突处理策略：

1. 要求 Agent 给出证据引用。
2. 启动第三个审查 Agent。
3. 回查原始工具输出。
4. 降低最终答案置信度。
5. 把冲突显式暴露给用户。
6. 对高风险动作转人工。

最终答案最好带来源：

```text
结论：转化率下降主要与推荐服务配置变更有关。
证据：
1. 数据 Agent 发现 5 月 10 日后首页点击率下降 12%。
2. 知识库 Agent 找到同日推荐召回策略变更记录。
3. 代码 Agent 确认配置 diff 改变了低活跃用户召回阈值。
限制：目前只分析了首页流量，未覆盖搜索流量。
```

## 49.17 失败模式与防护

跨 Agent 系统的失败模式很典型。

### 49.17.1 循环委派

Agent A 把任务委派给 Agent B，Agent B 又委派回 Agent A。

防护：

1. task_depth 限制。
2. delegation graph 检测环。
3. 同一任务类型最大委派次数。
4. 超过阈值转 Orchestrator 或人工。

### 49.17.2 冲突升级

两个 Agent 互相否定，系统不断要求重试。

防护：

1. 冲突次数限制。
2. 引入仲裁 Agent。
3. 要求引用原始证据。
4. 对无法解决的冲突显式返回。

### 49.17.3 幻觉传播

一个 Agent 编造了事实，另一个 Agent 把它当作事实继续推理。

防护：

1. 区分 claimed fact 和 verified fact。
2. 对关键事实要求工具证据。
3. Artifact 带 source_refs。
4. 下游 Agent 看到可信度和来源。

### 49.17.4 权限放大

低权限 Agent 通过高权限 Agent 间接执行动作。

防护：

1. OBO 权限求交集。
2. 委派链权限继承限制。
3. 高风险动作重新确认。
4. 审计完整 delegation_chain。

### 49.17.5 上下文污染

外部文档或工具输出里包含恶意指令，影响下游 Agent。

防护：

1. 标注不可信来源。
2. 工具输出不能升级成系统指令。
3. 下游 Agent 只把外部数据当数据。
4. 对敏感动作做二次策略检查。

## 49.18 Trace、Audit 与 Replay

跨 Agent 系统必须可观测。

Trace 应记录：

1. root_task_id。
2. subtask_id。
3. requester_agent。
4. assignee_agent。
5. input_context_refs。
6. messages。
7. tool_calls。
8. artifacts。
9. policy_decisions。
10. approvals。
11. status_transitions。
12. cost 和 latency。
13. errors。

Audit 要回答：

1. 谁发起了任务？
2. 哪些 Agent 参与了？
3. 每个 Agent 看到了什么上下文？
4. 每个 Agent 调用了什么工具？
5. 哪些策略允许或拒绝了动作？
6. 哪些结果被用于最终答案？
7. 是否有人审批？

Replay 用于调试和评估，但要注意：

1. 外部数据可能变化，需要记录快照或版本。
2. 模型输出有随机性，需要记录模型版本、参数和提示。
3. 工具调用可能有副作用，回放时要使用 dry-run 或 mock。

## 49.19 成本、延迟与并发控制

多 Agent 协作很容易失控，因为每个 Agent 都可能继续调用模型和工具。

需要控制：

1. 最大 Agent 数量。
2. 最大任务深度。
3. 最大子任务数量。
4. 最大工具调用次数。
5. token budget。
6. wall-clock deadline。
7. 每租户并发。
8. 每用户成本预算。
9. 失败重试次数。
10. 并行执行上限。

可以为每个 root task 设置 budget：

```json
{
  "max_agents": 6,
  "max_depth": 3,
  "max_subtasks": 20,
  "max_tool_calls": 50,
  "max_tokens": 200000,
  "deadline_ms": 300000,
  "max_cost_usd": 3.0
}
```

当预算快耗尽时，Orchestrator 应该降级：

1. 停止扩展新子任务。
2. 只保留高价值 Agent。
3. 返回部分结果。
4. 请求用户确认是否继续。

## 49.20 人工接管设计

跨 Agent 系统不能完全依赖自动化。

需要人工接管的场景：

1. 高风险动作。
2. 权限策略不确定。
3. Agent 之间结论冲突。
4. 任务连续失败。
5. 成本超过预算。
6. 用户要求解释。
7. 涉及合规或法律判断。

Human Review Console 应展示：

1. 原始用户请求。
2. 任务拆解图。
3. 参与 Agent。
4. 每个 Agent 的输入和输出摘要。
5. 关键证据引用。
6. 即将执行的动作和影响范围。
7. 策略决策原因。
8. 批准、拒绝、修改、转派选项。

人工不是系统失败的标志，而是高风险自动化的必要控制点。

## 49.21 评估指标

跨 Agent 协作的评估不能只看最终答案是否好。

需要分层评估：

1. 任务拆解是否合理。
2. Agent 选择是否正确。
3. 上下文是否足够且不过度暴露。
4. 权限判断是否正确。
5. 工具调用是否必要。
6. 中间结论是否有证据。
7. 冲突处理是否合理。
8. 最终答案是否正确。
9. 成本和延迟是否可接受。
10. 失败时是否安全降级。

线上指标包括：

1. task_success_rate。
2. subtask_failure_rate。
3. average_agent_count。
4. delegation_depth。
5. loop_detected_count。
6. human_escalation_rate。
7. policy_denial_rate。
8. conflict_rate。
9. cost_per_task。
10. p95_latency。

离线 benchmark 可以构造复杂任务集，标注期望任务图、期望 Agent、关键证据和最终答案。

## 49.22 一个完整请求流程

以“分析转化率下降并给出修复建议”为例：

1. 用户提交任务。
2. Orchestrator 创建 root_task。
3. Planner 拆解任务图。
4. Registry 根据能力和策略选择数据 Agent、知识库 Agent、代码 Agent。
5. Policy Engine 检查每个 Agent 的权限和上下文范围。
6. Context Manager 生成各自上下文包。
7. A2A Runtime 分发子任务。
8. 数据 Agent 通过 MCP 查询指标。
9. 知识库 Agent 通过 MCP 搜索变更记录。
10. 代码 Agent 通过 MCP 查询代码 diff。
11. 各 Agent 返回 artifact。
12. 分析 Agent 聚合证据并给出候选原因。
13. Orchestrator 检查冲突和证据完整性。
14. 若涉及生产配置修改，进入人工审批。
15. 最终返回带证据、限制和建议的报告。
16. Trace 系统记录全链路。

这个流程覆盖了系统设计题里的主干。

## 49.23 面试回答模板

可以按这个顺序回答：

```text
我会把跨 Agent 协作系统分成六层。

第一层是入口和 Orchestrator，负责接收用户任务、拆解任务图、调度子任务和汇总结果。

第二层是 Agent Registry，用 Agent Card 描述每个 Agent 的能力、输入输出、权限、风险、版本和健康状态。

第三层是 A2A Runtime，抽象 Task、Message、Status 和 Artifact，支持任务委派、状态同步、结果返回和失败恢复。

第四层是 Context Manager 和 Policy Engine，负责上下文裁剪、数据脱敏、OBO 权限、租户隔离、高风险审批和防止权限放大。

第五层是 Tool/MCP Gateway，让专业 Agent 通过受控方式访问数据库、知识库、代码仓库、终端和业务 API。

第六层是 Trace、Audit、Replay 和 Eval，用来追踪每个 Agent 看到了什么、做了什么、调用了什么工具、产出了什么证据，并持续评估协作质量。

重点风险包括循环委派、冲突升级、幻觉传播、上下文泄露和权限放大。我会用任务深度限制、委派图环检测、证据引用、策略引擎、人工审批和审计回放来控制这些风险。
```

## 49.24 常见扣分回答

扣分回答一：只说“用一个主 Agent 调多个子 Agent”。

问题是没有讲任务状态、权限、上下文和失败处理。

扣分回答二：把所有上下文广播给所有 Agent。

这会导致数据泄露、上下文污染和成本失控。

扣分回答三：没有 Agent Card。

没有结构化能力声明，系统无法稳定调度。

扣分回答四：没有权限链路。

跨 Agent 委派最怕权限放大。

扣分回答五：只看最终答案，不评估过程。

多 Agent 错误常发生在拆解、委派、上下文和中间结论阶段。

扣分回答六：没有循环和冲突处理。

多 Agent 系统在生产中很容易出现互相调用、互相否定和重复重试。

## 49.25 面试题

### 题 1：跨 Agent 系统为什么需要 Task，而不是只需要 Message？

答：Message 只表示一次通信，Task 表示可追踪工作单元。Task 有生命周期、负责人、输入、输出、状态、超时、重试和审计。没有 Task，系统无法可靠追踪子任务进度和失败恢复。

### 题 2：Agent Card 有什么作用？

答：Agent Card 用结构化方式声明 Agent 的能力、输入输出、权限、风险、版本、成本和健康状态。它是能力发现、调度、权限过滤、版本治理和评估的基础。

### 题 3：如何防止跨 Agent 权限放大？

答：使用 OBO 模型，将用户权限、Agent 权限、任务策略、租户策略和风险策略求交集。委派链中不能因为下游 Agent 权限更高而扩大原始用户权限。高风险动作需要重新确认和审计。

### 题 4：如何防止幻觉在多个 Agent 之间传播？

答：区分声明事实和已验证事实，要求关键结论带工具证据或来源引用。Artifact 要记录 source_refs 和 confidence。下游 Agent 不能把上游自然语言结论当作绝对事实，应优先引用原始证据。

### 题 5：A2A 和 MCP 在这个系统中如何分工？

答：A2A 处理 Agent 之间的任务委派、状态同步、消息交换和结果返回。MCP 处理 Agent 与工具/资源之间的连接，例如数据库、知识库、代码仓库和终端。二者处在不同层，Policy Engine 和 Trace 系统横跨两层。

## 49.26 小练习

练习一：设计一个客服跨 Agent 系统。

要求：至少包含客服 Agent、订单 Agent、退款 Agent、知识库 Agent 和审批 Agent。写出 Agent Card 的关键字段，并说明退款动作如何做权限控制和人工审批。

练习二：设计一个研发跨 Agent 系统。

要求：用户提交 bug，系统自动调用复现 Agent、代码 Agent、测试 Agent 和 Review Agent。写出任务状态机和失败恢复策略。

练习三：设计一个防循环机制。

要求：给出 delegation graph 的结构，说明如何检测 Agent A -> Agent B -> Agent A 这种循环。

练习四：设计一个多 Agent eval benchmark。

要求：构造 5 类任务，分别评估任务拆解、Agent 选择、上下文裁剪、结果聚合和失败恢复。

## 49.27 本章小结

本章用系统设计题串起了跨 Agent 协作系统。

你需要掌握：

1. 跨 Agent 协作不是简单聊天，而是可追踪任务协作。
2. Agent Card 是能力发现和调度的基础。
3. A2A Runtime 要抽象 Task、Message、Status 和 Artifact。
4. Orchestrator 负责任务拆解、调度、状态推进和结果聚合。
5. Context Manager 控制上下文边界，避免泄露和污染。
6. Policy Engine 负责 OBO 权限、租户隔离、审批和风险控制。
7. MCP 是 Agent 使用工具和资源的连接层，不要和 A2A 混淆。
8. 多 Agent 失败模式包括循环委派、冲突升级、幻觉传播、权限放大和上下文污染。
9. Trace、Audit、Replay 和 Eval 是生产化必需能力。

如果面试官问“设计一个跨 Agent 协作系统”，你不要只画一堆 Agent。你要画出任务图、Agent Registry、A2A Runtime、Context Manager、Policy Engine、MCP Gateway 和 Observability，并解释每一层如何保证系统可控、可追踪、可恢复。
