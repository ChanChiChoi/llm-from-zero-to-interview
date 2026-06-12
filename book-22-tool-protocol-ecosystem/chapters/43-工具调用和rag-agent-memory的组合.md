# 第 43 章 工具调用和 RAG、Agent、Memory 的组合

前面我们分别讲过工具调用、MCP、A2A、Skill、工作流、prompt injection 和工具输出可信度。现在我们把视角放到大模型应用的整体架构上：工具调用如何和 RAG、Agent、Memory 组合？

很多系统设计里，这几个概念会被混在一起：有人把 RAG 当工具，有人把 Memory 当 RAG，有人把 Agent 当工具路由器，有人把所有外部能力都叫 Tool。这些说法在口语上可以理解，但在工程设计中必须分清楚。

本章的目标是建立一套清晰边界：RAG 解决知识检索，Tool 解决外部操作，Agent 解决自主规划和任务执行，Memory 解决长期个性化和历史状态。它们可以组合，但不能互相替代。

你可以先记住一句话：

> RAG 提供知识，Tool 执行动作，Agent 组织过程，Memory 保留长期状态。

## 43.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，参考了 RAG 原论文对 retrieval augmented generation 的知识检索与生成边界，OpenAI Agents SDK 对 tools、handoffs、guardrails、sessions 和 tracing 的运行时抽象，LangGraph / LangChain 官方文档对 short-term memory、long-term memory、context engineering 和 graph state 的工程划分，以及前面章节已经建立的工具权限、工具输出可信度、prompt injection 防御和 trace / replay 口径。

为了避免把某个框架 API 写成通用标准，本章只抽象稳定的组合边界：

1. RAG 是知识检索和证据构造模式，不等于所有 search tool。
2. Tool 是外部可调用能力，不等于 Agent 的整体规划过程。
3. Agent 是围绕目标组织 RAG、Tool、Memory、状态、失败恢复和完成判定的 runtime。
4. Memory 是长期状态或个性化数据资产，不是无边界的上下文缓存。
5. 组合系统的质量要用上下文优先级、预算、证据链、状态更新、Memory 写入门禁、安全边界、trace 和分层 eval 共同证明。

## 43.1 四个概念的一句话区分

| 概念 | 一句话定义 | 典型问题 |
| --- | --- | --- |
| RAG | 从外部知识库检索相关上下文 | 应该参考哪些资料？ |
| Tool | 调用外部系统执行操作或查询 | 应该做哪个动作？ |
| Agent | 基于目标进行规划、调用工具和管理状态 | 应该如何完成任务？ |
| Memory | 保存用户、任务或系统的长期状态 | 过去发生了什么，用户偏好是什么？ |

这四者不是竞争关系，而是互补关系。

## 43.2 RAG 解决什么问题

RAG 的核心是检索增强生成。

它解决的是模型不知道或不应只靠参数记忆回答的问题：

1. 企业内部文档。
2. 最新政策。
3. 产品手册。
4. 历史工单。
5. 技术文档。
6. 法务条款。
7. 项目知识。

RAG 的典型流程：

```text
用户问题 -> query rewrite -> retrieve -> rerank -> context packing -> model answer with citations
```

RAG 主要输出是上下文和证据，不一定执行外部动作。

## 43.3 Tool 解决什么问题

Tool 解决外部操作或查询。

例如：

1. 查询数据库。
2. 发送邮件。
3. 创建工单。
4. 运行测试。
5. 读取文件。
6. 修改代码。
7. 调用业务 API。

Tool 的典型流程：

```text
模型决定调用工具 -> 生成参数 -> 校验权限 -> 执行工具 -> 返回结果 -> 进入上下文
```

Tool 关注输入输出 schema、权限、超时、错误、幂等和副作用。

## 43.4 Agent 解决什么问题

Agent 解决多步任务执行。

一个 Agent 不只是回答问题，而是会：

1. 理解目标。
2. 制定计划。
3. 选择工具。
4. 调用 RAG。
5. 使用 Memory。
6. 处理失败。
7. 判断是否完成。
8. 返回结果或 Artifact。

例如用户说：

```text
请定位这次退款率升高的原因，并生成报告。
```

Agent 可能会：

1. 查询指标定义。
2. 查询数据库。
3. 检索相关发布记录。
4. 分析异常。
5. 生成报告。
6. 请求人工确认。

Agent 是过程组织者，不是单个工具。

## 43.5 Memory 解决什么问题

Memory 解决长期状态和个性化。

Memory 可能保存：

1. 用户偏好。
2. 历史任务。
3. 项目上下文。
4. 已确认事实。
5. 常用配置。
6. 团队术语。
7. 用户反馈。

例如用户之前说过：

```text
我的周报默认用中文，面向产品团队，不要太技术化。
```

这个偏好可以进入 Memory，后续生成周报时自动使用。

Memory 的关键风险是过期、错误、隐私和污染。不是所有历史信息都应该长期保存。

## 43.6 RAG 和 Tool 的边界

RAG 和 Tool 最容易混淆。

一个检索工具 search_docs 看起来像 Tool，但它的业务作用是 RAG 的一部分。

可以这样区分：

1. Tool 是协议层可调用能力。
2. RAG 是知识检索和上下文构造模式。
3. RAG 可以通过 Tool 实现检索。
4. 不是所有 Tool 都是 RAG。

例如：

```text
search_docs(query) 是 Tool。
query rewrite + search_docs + rerank + context packing + citation 是 RAG pipeline。
```

RAG 是一条流程，Tool 是其中的一个操作。

## 43.7 Agent 和 Workflow 的边界

Agent 和 Workflow 也容易混淆。

Workflow 是预定义流程；Agent 是可以动态规划和决策的执行主体。

固定 Workflow：

```text
retrieve_docs -> summarize -> cite -> answer
```

Agent：

```text
根据目标决定是否检索、是否查数据库、是否运行工具、是否追问用户。
```

生产系统常用混合模式：

1. 高频高风险任务用固定 Workflow。
2. 开放任务用 Agent 动态规划。
3. Agent 的动作受 Workflow 模板、权限和预算约束。

## 43.8 Memory 和 RAG 的边界

Memory 和 RAG 都是在给模型补充上下文，但来源和语义不同。

RAG 通常来自外部知识库，面向事实资料。

Memory 通常来自用户历史、会话、偏好和任务状态。

例如：

```text
RAG：公司报销政策第 3 节规定差旅补贴标准。
Memory：用户偏好用表格形式展示报销信息。
```

Memory 不应该替代权威知识库。用户记忆中说“报销上限是 1000”，但政策文档更新为 800 时，应该以权威 RAG 来源为准。

## 43.9 一个典型组合架构

一个完整系统可以这样设计：

```text
User
  -> Agent Runtime
    -> Memory Store
    -> RAG Pipeline
      -> Retriever Tool
      -> Reranker
      -> Context Builder
    -> Tool Runtime
      -> Database Tool
      -> Email Tool
      -> Ticket Tool
    -> Policy Engine
    -> Trace / Audit
```

流程：

1. Agent 接收用户目标。
2. Memory 提供用户偏好和历史状态。
3. RAG 提供相关知识和引用。
4. Agent 决定是否调用 Tool。
5. Tool Runtime 执行权限检查和调用。
6. Policy Engine 控制高风险动作。
7. Trace 记录全链路。

## 43.10 例子：客服助手

用户问：

```text
帮我回复这个退款投诉，并判断是否需要升级。
```

系统组合：

1. RAG 检索退款政策、客服话术、升级规则。
2. Tool 读取当前工单和用户订单。
3. Memory 读取该用户历史投诉和偏好。
4. Agent 判断是否升级、生成回复草稿。
5. Tool 创建升级工单或发送回复。
6. 高风险发送动作需要人工确认。

如果只用 RAG，系统不能读取实时订单。只用 Tool，系统缺少政策上下文。只用 Memory，可能使用过期偏好。没有 Agent，系统难以组织整个过程。

## 43.11 例子：代码修复助手

用户说：

```text
这个测试失败了，请帮我修复。
```

系统组合：

1. Tool 读取测试日志。
2. Tool 搜索相关代码。
3. RAG 检索项目开发规范和历史类似问题。
4. Memory 读取该仓库常用测试命令。
5. Agent 制定修复计划。
6. Tool 应用 patch。
7. Tool 运行测试。
8. Agent 总结变更。

这里 RAG 提供规范和经验，Tool 执行文件和测试操作，Memory 保存项目偏好，Agent 负责过程。

## 43.12 上下文优先级

组合系统中，上下文来源很多，需要优先级。

一个常见优先级：

1. System policy。
2. 用户当前任务。
3. 权威工具结果。
4. 内部 verified RAG。
5. Memory 中的偏好。
6. 用户上传资料。
7. 外部网页。
8. 其他 Agent 未验证输出。

注意，Memory 不应覆盖 system policy，外部网页不应覆盖用户任务，未验证 Agent 输出不应覆盖权威数据。

## 43.13 上下文预算分配

RAG、Tool result、Memory 都会占用上下文窗口。

需要分配预算：

1. 用户当前问题必须保留。
2. 系统策略必须保留。
3. 关键工具结果优先。
4. RAG 片段按相关性和权威性选择。
5. Memory 只取和当前任务相关的。
6. 大工具结果摘要后进入上下文。
7. 原始大对象用引用保存。

否则上下文会被无关记忆或冗长工具输出挤爆。

## 43.14 组合系统中的安全问题

组合越复杂，安全越难。

常见风险：

1. RAG 文档注入诱导 Tool 调用。
2. Memory 中旧偏好覆盖新指令。
3. Tool 输出敏感信息被写入 Memory。
4. Agent 把未验证 RAG 内容当成事实。
5. 外部网页内容触发高风险 Tool。
6. 多 Agent 传递时丢失来源。

防御方式：

1. 指令与数据分离。
2. 来源和可信级别标记。
3. Memory 写入审核。
4. 高风险 Tool 确认。
5. Tool result 脱敏。
6. RAG 引用校验。
7. Trace 串联所有来源。

## 43.15 Memory 写入策略

Memory 最大的问题不是读取，而是写入。

不能把所有对话都写入长期记忆。

写入前要判断：

1. 是否长期有用。
2. 是否经过用户确认。
3. 是否包含敏感信息。
4. 是否可能过期。
5. 是否来自可信来源。
6. 是否可删除。
7. 是否有作用范围。

例如用户偏好可以写入：

```text
用户偏好：周报用中文，面向业务团队。
```

但工具返回的临时数据库结果不应该写入长期 Memory。

## 43.16 Memory 和权限

Memory 也需要权限。

例如：

1. 用户级 Memory 只能当前用户访问。
2. 团队级 Memory 只对团队可见。
3. 项目级 Memory 受项目权限控制。
4. 敏感 Memory 需要加密或禁止保存。
5. 用户应能查看、修改和删除自己的 Memory。

Memory 不是“免费的上下文缓存”，它也是数据资产。

## 43.17 常见组合模式

### 43.17.1 RAG-first

先检索知识，再回答或调用工具。

适合政策问答、文档问答、知识密集任务。

### 43.17.2 Tool-first

先查实时状态，再结合知识回答。

适合订单查询、库存查询、监控排障。

### 43.17.3 Memory-first

先读取用户偏好，再决定回答风格或默认参数。

适合个性化助手。

### 43.17.4 Agent-planned

Agent 根据任务动态决定先检索、先查工具还是先追问。

适合复杂开放任务。

生产系统可以根据任务类型选择组合模式。

## 43.18 常见失败模式

### 43.18.1 RAG 和 Memory 冲突

Memory 里保存旧政策，RAG 检索到新政策。应该以权威新政策为准，并更新或标记 Memory 过期。

### 43.18.2 Tool 结果缺少引用

Agent 给出结论，但无法追溯到工具调用。需要强制 evidence chain。

### 43.18.3 Memory 污染

把一次错误工具结果写入长期 Memory，后续不断复用。需要写入审核和过期机制。

### 43.18.4 Agent 过度调用工具

简单问答也频繁查库、检索、调用工具。需要成本预算和路由策略。

### 43.18.5 RAG 注入触发 Tool

文档内容诱导 Agent 调用外部工具。需要指令/数据分离和策略拦截。

## 43.19 评估组合系统

组合系统要分层评估：

1. RAG：召回率、引用准确率、上下文相关性。
2. Tool：调用准确率、参数正确率、成功率。
3. Agent：任务成功率、规划质量、失败恢复。
4. Memory：读取相关性、写入准确性、过期控制。
5. End-to-end：最终任务成功率、安全性、成本和用户满意度。

只评估最终答案不够，因为你不知道问题出在检索、工具、规划还是记忆。

## 43.20 RAG + Tool + Agent + Memory 组合审计指标与最小 demo

组合系统最容易出问题的地方，不是某一个模块完全不可用，而是边界混乱：RAG 证据被当成指令，Tool 临时结果被写进长期 Memory，Memory 旧偏好覆盖权威数据，Agent 状态没有记录，最终答案没有 evidence chain。

可以把一次组合系统审计样本写成：

```math
h_i=(q_i,r_i,t_i,a_i,m_i,b_i,e_i,s_i,p_i,z_i)
```

其中：

1. $q_i$ 是用户目标、任务类型和完成标准。
2. $r_i$ 是 RAG 检索、rerank、context packing、citation 和 source trust。
3. $t_i$ 是 Tool 调用、参数、权限、执行结果和 observation。
4. $a_i$ 是 Agent 的计划、动作序列、失败恢复、状态更新和终止条件。
5. $m_i$ 是 Memory 读取、写入候选、作用范围、TTL、删除和用户确认。
6. $b_i$ 是上下文优先级和预算，包括 system policy、用户当前任务、RAG、Tool result、Memory 和 artifact 引用。
7. $e_i$ 是 evidence chain，包括 source ids、tool call ids、artifact ids、claim map 和 limitations。
8. $s_i$ 是安全边界，包括 instruction / data separation、prompt injection 防御、敏感数据、权限和高风险确认。
9. $p_i$ 是冲突处理，包括 Memory vs RAG、Tool vs RAG、旧状态 vs 实时状态的优先级。
10. $z_i$ 是 trace、replay、eval 和线上监控证据。

对每个检查项 $j$，定义覆盖率：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(h_i)=1]
```

如果系统把不可信 RAG 内容变成工具指令，或把敏感 Tool result 写入长期 Memory，可以定义不安全组合放行率：

```math
R_{\mathrm{unsafe}}=\frac{\sum_{i=1}^{N}\mathbf{1}[\mathrm{unsafe\_integration}_i=1]}{N}
```

综合门禁可以写成：

```math
G_{\mathrm{integration}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{unsafe}}=0 \land P_0=0\right]
```

常见审计指标包括：

1. capability boundary clarity：RAG、Tool、Agent、Memory 的职责是否清楚。
2. orchestration mode fit：任务是否选择了合适的 RAG-first、Tool-first、Memory-first、Workflow 或 Agent-planned 模式。
3. context priority enforcement：system policy、用户任务、权威工具结果、verified RAG、Memory、外部网页的优先级是否稳定。
4. context budget allocation：上下文窗口是否为用户任务、证据、工具结果、Memory 和 artifact 引用分配预算。
5. evidence preservation：RAG citation、Tool observation、artifact、claim map 和 limitations 是否随流程保留。
6. tool observation use：Agent 是否正确使用工具结果，而不是忽略、误读或过度外推。
7. agent state update discipline：Agent 是否记录计划、状态、完成标准、失败恢复和停止原因。
8. memory read relevance：Memory 读取是否和当前任务相关，是否按用户、团队、项目或租户作用域过滤。
9. memory write gate：长期 Memory 写入是否经过长期价值、用户确认、敏感数据、可信来源、TTL / 删除策略检查。
10. memory scope permission：Memory 是否支持作用域隔离、查看、修改和删除。
11. conflict resolution policy：Memory、RAG、Tool result 和实时数据库冲突时是否按权威性、新鲜度和权限处理。
12. injection propagation control：RAG 文档、网页、Tool result 或其他 Agent 输出是否不能触发高风险工具。
13. sensitive data memory block：敏感 Tool result、PII、密钥和一次性查询结果是否不会写入长期 Memory。
14. trace linkage coverage：trace 是否串联用户目标、RAG query、retrieved docs、tool calls、memory reads / writes、Agent state 和最终 claim。
15. layered eval coverage：eval 是否分别覆盖 RAG、Tool、Agent、Memory 和 end-to-end。

下面是一个 0 依赖 toy demo。它不实现真实 Agent，而是模拟组合系统 trace 的审计表。

```python
from copy import deepcopy


THRESHOLD = 0.95


def recursive_update(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            recursive_update(base[key], value)
        else:
            base[key] = value
    return base


def make_case(name, **overrides):
    case = {
        "name": name,
        "boundary": {
            "rag_is_knowledge": True,
            "tool_is_action": True,
            "agent_orchestrates": True,
            "memory_is_long_term": True,
        },
        "mode": {
            "task_type": "diagnosis_report",
            "expected": "agent_planned",
            "actual": "agent_planned",
        },
        "priority": {
            "system_policy_first": True,
            "current_task_over_memory": True,
            "authoritative_tool_over_memory": True,
            "untrusted_below_user_task": True,
        },
        "budget": {
            "user_task": True,
            "rag_context": True,
            "tool_result": True,
            "memory": True,
            "artifact_refs": True,
        },
        "evidence": {
            "rag_citations": True,
            "tool_call_ids": True,
            "artifact_ids": True,
            "claim_map": True,
            "limitations": True,
        },
        "tool_observation": {
            "used": True,
            "trust_checked": True,
            "not_overgeneralized": True,
        },
        "agent_state": {
            "plan": True,
            "status": True,
            "completion_criteria": True,
            "failure_recovery": True,
            "stop_reason": True,
        },
        "memory_read": {
            "relevant": True,
            "scoped": True,
            "fresh_enough": True,
        },
        "memory_write": {
            "long_term_value": True,
            "user_confirmed": True,
            "not_sensitive": True,
            "trusted_source": True,
            "ttl_or_delete": True,
        },
        "memory_permission": {
            "user_scope": True,
            "tenant_scope": True,
            "view_edit_delete": True,
        },
        "conflict": {
            "has_conflict": False,
            "resolved": True,
            "authority_used": True,
            "freshness_used": True,
        },
        "security": {
            "instruction_data_separation": True,
            "injection_blocked": True,
            "sensitive_not_written": True,
            "high_risk_confirmed": True,
        },
        "trace": {
            "user_goal_id": True,
            "rag_query_ids": True,
            "retrieved_doc_ids": True,
            "tool_call_ids": True,
            "memory_event_ids": True,
            "agent_state_ids": True,
            "final_claim_ids": True,
        },
        "eval": {
            "rag": True,
            "tool": True,
            "agent": True,
            "memory": True,
            "end_to_end": True,
            "safety": True,
        },
        "unsafe_integration": False,
    }
    return recursive_update(case, deepcopy(overrides))


def capability_boundary_clarity(case):
    return all(case["boundary"].values())


def orchestration_mode_fit(case):
    mode = case["mode"]
    return mode["expected"] == mode["actual"]


def context_priority_enforcement(case):
    return all(case["priority"].values())


def context_budget_allocation(case):
    return all(case["budget"].values())


def evidence_preservation(case):
    return all(case["evidence"].values())


def tool_observation_use(case):
    return all(case["tool_observation"].values())


def agent_state_update_discipline(case):
    return all(case["agent_state"].values())


def memory_read_relevance(case):
    return all(case["memory_read"].values())


def memory_write_gate(case):
    return all(case["memory_write"].values())


def memory_scope_permission(case):
    return all(case["memory_permission"].values())


def conflict_resolution_policy(case):
    conflict = case["conflict"]
    if not conflict["has_conflict"]:
        return True
    return (
        conflict["resolved"]
        and conflict["authority_used"]
        and conflict["freshness_used"]
    )


def injection_propagation_control(case):
    security = case["security"]
    return security["instruction_data_separation"] and security["injection_blocked"]


def sensitive_data_memory_block(case):
    return case["security"]["sensitive_not_written"]


def trace_linkage_coverage(case):
    return all(case["trace"].values())


def layered_eval_coverage(case):
    return all(case["eval"].values())


CHECKS = {
    "capability_boundary_clarity": capability_boundary_clarity,
    "orchestration_mode_fit": orchestration_mode_fit,
    "context_priority_enforcement": context_priority_enforcement,
    "context_budget_allocation": context_budget_allocation,
    "evidence_preservation": evidence_preservation,
    "tool_observation_use": tool_observation_use,
    "agent_state_update_discipline": agent_state_update_discipline,
    "memory_read_relevance": memory_read_relevance,
    "memory_write_gate": memory_write_gate,
    "memory_scope_permission": memory_scope_permission,
    "conflict_resolution_policy": conflict_resolution_policy,
    "injection_propagation_control": injection_propagation_control,
    "sensitive_data_memory_block": sensitive_data_memory_block,
    "trace_linkage_coverage": trace_linkage_coverage,
    "layered_eval_coverage": layered_eval_coverage,
}


CASES = [
    make_case("complete_rag_tool_agent_memory_ok"),
    make_case("boundary_confused_bad", boundary={"rag_is_knowledge": False}),
    make_case("wrong_orchestration_mode_bad", mode={"actual": "rag_first"}),
    make_case(
        "context_priority_bad",
        priority={"authoritative_tool_over_memory": False},
    ),
    make_case(
        "context_budget_missing_bad",
        budget={"tool_result": False, "artifact_refs": False},
    ),
    make_case(
        "evidence_lost_bad",
        evidence={"tool_call_ids": False, "claim_map": False, "limitations": False},
    ),
    make_case(
        "tool_observation_ignored_bad",
        tool_observation={"used": False, "not_overgeneralized": False},
    ),
    make_case(
        "agent_state_missing_bad",
        agent_state={"completion_criteria": False, "stop_reason": False},
    ),
    make_case(
        "memory_read_irrelevant_bad",
        memory_read={"relevant": False, "scoped": False},
    ),
    make_case(
        "memory_write_unconfirmed_bad",
        memory_write={"long_term_value": True, "user_confirmed": False},
    ),
    make_case(
        "memory_permission_missing_bad",
        memory_permission={"tenant_scope": False, "view_edit_delete": False},
    ),
    make_case(
        "memory_rag_conflict_unresolved_bad",
        conflict={
            "has_conflict": True,
            "resolved": False,
            "authority_used": False,
            "freshness_used": False,
        },
    ),
    make_case(
        "rag_injection_triggers_tool_bad",
        security={"instruction_data_separation": False, "injection_blocked": False},
        unsafe_integration=True,
    ),
    make_case(
        "sensitive_tool_result_written_memory_bad",
        security={"sensitive_not_written": False},
        unsafe_integration=True,
    ),
    make_case(
        "trace_linkage_missing_bad",
        trace={"rag_query_ids": False, "memory_event_ids": False, "final_claim_ids": False},
    ),
    make_case(
        "layered_eval_missing_bad",
        eval={"rag": True, "tool": False, "agent": False, "memory": False, "safety": False},
    ),
]


def evaluate(cases):
    metrics = {}
    for name, check in CHECKS.items():
        passed = sum(1 for case in cases if check(case))
        metrics[name] = round(passed / len(cases), 3)

    unsafe_integration_rate = round(
        sum(1 for case in cases if case["unsafe_integration"]) / len(cases),
        3,
    )
    failed_cases = [
        case["name"]
        for case in cases
        if any(not check(case) for check in CHECKS.values())
        or case["unsafe_integration"]
    ]
    failed_gates = [
        name for name, value in metrics.items() if value < THRESHOLD
    ]
    if unsafe_integration_rate > 0:
        failed_gates.append("unsafe_integration_rate")
    return metrics, unsafe_integration_rate, failed_cases, failed_gates


case_by_name = {case["name"]: case for case in CASES}
metrics, unsafe_integration_rate, failed_cases, failed_gates = evaluate(CASES)

smoke = {
    "complete_case_passes": all(
        check(case_by_name["complete_rag_tool_agent_memory_ok"])
        for check in CHECKS.values()
    ),
    "caught_memory_conflict": not conflict_resolution_policy(
        case_by_name["memory_rag_conflict_unresolved_bad"]
    ),
    "caught_rag_injection": not injection_propagation_control(
        case_by_name["rag_injection_triggers_tool_bad"]
    ),
    "caught_sensitive_memory_write": not sensitive_data_memory_block(
        case_by_name["sensitive_tool_result_written_memory_bad"]
    ),
}

print("smoke=", smoke)
print("metrics=", metrics)
print("unsafe_integration_rate=", unsafe_integration_rate)
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("rag_tool_agent_memory_gate_pass=", len(failed_gates) == 0)
```

参考输出：

```text
smoke= {'complete_case_passes': True, 'caught_memory_conflict': True, 'caught_rag_injection': True, 'caught_sensitive_memory_write': True}
metrics= {'capability_boundary_clarity': 0.938, 'orchestration_mode_fit': 0.938, 'context_priority_enforcement': 0.938, 'context_budget_allocation': 0.938, 'evidence_preservation': 0.938, 'tool_observation_use': 0.938, 'agent_state_update_discipline': 0.938, 'memory_read_relevance': 0.938, 'memory_write_gate': 0.938, 'memory_scope_permission': 0.938, 'conflict_resolution_policy': 0.938, 'injection_propagation_control': 0.938, 'sensitive_data_memory_block': 0.938, 'trace_linkage_coverage': 0.938, 'layered_eval_coverage': 0.938}
unsafe_integration_rate= 0.125
failed_cases= ['boundary_confused_bad', 'wrong_orchestration_mode_bad', 'context_priority_bad', 'context_budget_missing_bad', 'evidence_lost_bad', 'tool_observation_ignored_bad', 'agent_state_missing_bad', 'memory_read_irrelevant_bad', 'memory_write_unconfirmed_bad', 'memory_permission_missing_bad', 'memory_rag_conflict_unresolved_bad', 'rag_injection_triggers_tool_bad', 'sensitive_tool_result_written_memory_bad', 'trace_linkage_missing_bad', 'layered_eval_missing_bad']
failed_gates= ['capability_boundary_clarity', 'orchestration_mode_fit', 'context_priority_enforcement', 'context_budget_allocation', 'evidence_preservation', 'tool_observation_use', 'agent_state_update_discipline', 'memory_read_relevance', 'memory_write_gate', 'memory_scope_permission', 'conflict_resolution_policy', 'injection_propagation_control', 'sensitive_data_memory_block', 'trace_linkage_coverage', 'layered_eval_coverage', 'unsafe_integration_rate']
rag_tool_agent_memory_gate_pass= False
```

这个 demo 的重点是：组合系统的合格标准不是“RAG、Tool、Agent、Memory 都出现了”，而是它们的边界、上下文、证据、状态、Memory 写入和安全策略都能被 trace 和 eval 证明。

## 43.21 面试高频题

### 题 1：RAG、Tool、Agent、Memory 的区别是什么？

参考回答：

RAG 负责从外部知识库检索相关上下文，Tool 负责调用外部系统执行操作或查询，Agent 负责基于目标进行规划和任务执行，Memory 负责保存长期偏好、历史状态和个性化信息。它们互补，不应混用。

### 题 2：RAG 和 Tool 的关系是什么？

参考回答：

RAG 是检索增强生成流程，Tool 是可调用能力。RAG 可以用 search_docs 这类 Tool 实现检索，但 RAG 还包括 query rewrite、rerank、context packing 和 citation。不是所有 Tool 都是 RAG。

### 题 3：Memory 和 RAG 如何区分？

参考回答：

RAG 通常检索权威外部知识，Memory 保存用户偏好、历史任务和长期状态。Memory 不能替代权威知识库，尤其当两者冲突时，应比较来源、时间和权威性。

### 题 4：组合系统如何做安全？

参考回答：

需要来源标记、可信级别、指令与数据分离、Memory 写入审核、Tool 权限控制、RAG 引用校验、高风险动作确认和全链路 trace。外部文档和网页不能触发高风险工具调用。

### 题 5：如何评估 RAG + Tool + Agent + Memory 系统？

参考回答：

要分层评估：RAG 看召回和引用，Tool 看调用和参数，Agent 看规划和任务成功，Memory 看读取和写入质量，最终再看端到端成功率、安全性、成本和用户满意度。

## 43.22 小练习

1. 设计一个客服助手，标出哪些部分用 RAG、哪些用 Tool、哪些用 Memory、哪些由 Agent 决策。
2. 为一个代码修复 Agent 设计 Memory 写入规则。
3. 设计一个上下文预算分配方案：用户问题、RAG、Tool result、Memory 各占多少。
4. 列出 RAG 文档注入触发 Tool 调用的防御策略。
5. 思考：如果 Memory 和数据库实时查询结果冲突，系统应该如何处理？

## 43.23 本章小结

本章我们讲了工具调用和 RAG、Agent、Memory 的组合。

RAG 提供知识，Tool 执行动作，Agent 组织过程，Memory 保留长期状态。它们组合后可以构成强大的大模型应用，但也会带来上下文冲突、注入传播、Memory 污染、工具过度调用和证据链丢失等问题。工程上需要清晰边界、来源标记、权限控制、上下文预算、Memory 写入策略和分层评估。

你可以把本章重点记成一句话：

> 复杂大模型应用不是 RAG、Tool、Agent、Memory 谁替代谁，而是它们各司其职、通过协议和运行时安全组合。

下一章我们会继续讲工具调用成本、延迟和并发控制，讨论工具生态从可用走向高性能时必须面对的工程问题。
