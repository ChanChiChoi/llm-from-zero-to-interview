# 第 54 章 设计一个企业级 LLMOps 平台

上一章设计了高并发大模型推理平台。本章继续系统设计题：设计一个企业级 LLMOps 平台。

LLMOps 平台关注的是大模型应用从开发、评估、发布、观测、治理到迭代的完整生命周期。它不是单纯训练平台，也不是单纯推理平台，而是把模型、prompt、RAG、Agent、工具、安全、评估、发布和成本整合起来的企业级平台。

先记住一句话：

> 企业级 LLMOps 平台的核心，是让大模型应用可开发、可评估、可发布、可观测、可治理、可审计、可持续优化。

## 54.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 OpenTelemetry 对生成式 AI trace / metric 语义、span 属性和系统观测信号的公开口径，MLflow 对 GenAI 评估、trace、run、metric、artifact 和 prompt / eval 配置管理的公开口径，OpenAI Evals 对可复用评估样例、评估逻辑和回归验证的工程边界，以及前文模型网关、推理平台、RAG / Agent 存储、评估平台、可观测性、安全治理、成本治理和审计变更治理章节已经校准过的口径。

这些资料共同指向一个稳定结论：企业级 LLMOps 平台不是“把模型 API 包一层控制台”，而是把大模型应用的模型路由、prompt、知识库、工具、Agent、评估、发布、trace、反馈、安全、成本和审计放进同一套应用生命周期控制面。

本章只抽象截至 2026-06 仍稳定的企业级 LLMOps 系统设计方法，不把某个 SaaS、SDK、模型供应商、评估框架、向量数据库或内部平台实现写成通用标准。正文公式用于面试表达、系统设计自查和 toy demo 审计；真实落地仍要结合企业身份体系、数据分级、模型供应商协议、业务流程、合规要求、可用性 SLO 和历史事故复盘校准。

## 54.1 题目理解

面试题可能这样问：

1. 请设计一个企业内部 LLMOps 平台。
2. 请设计一个支持 prompt、RAG、Agent、评估和发布的大模型应用平台。
3. 请设计一个多团队共享的大模型应用生命周期管理平台。
4. 请设计一个统一管理模型调用、知识库、工具和 trace 的平台。

这类题重点不是模型训练，而是大模型应用落地和治理。

## 54.2 LLMOps 和 MLOps 的区别

MLOps 更关注传统机器学习生命周期：

1. 数据。
2. 特征。
3. 训练。
4. 部署。
5. 监控。

LLMOps 额外关注：

1. Prompt 版本。
2. RAG 知识库。
3. Agent 工具调用。
4. 模型路由。
5. 输出安全。
6. 用户反馈。
7. LLM-as-Judge。
8. Trace 回放。
9. Token 成本。
10. 多模型能力治理。

LLMOps 更偏“大模型应用生命周期管理”。

## 54.3 需求澄清

设计前先问：

1. 平台服务哪些用户？算法、应用开发、业务团队、运维？
2. 是否接入多个基础模型和外部模型？
3. 是否支持 prompt 管理？
4. 是否支持 RAG 知识库？
5. 是否支持 Agent 和工具？
6. 是否支持评估和 A/B 测试？
7. 是否需要多租户隔离？
8. 是否有安全、合规和审计要求？
9. 是否需要成本预算和计量？
10. 是否对外提供 API/SDK？

企业级平台必须明确用户角色和治理要求。

## 54.4 核心目标

企业级 LLMOps 平台目标：

1. 统一模型接入和调用。
2. 管理 prompt 版本和发布。
3. 管理知识库和 RAG 链路。
4. 管理 Agent 和工具定义。
5. 支持评估、实验和对比。
6. 支持灰度、A/B 测试和回滚。
7. 支持 trace、日志和可观测性。
8. 支持权限、安全和审计。
9. 支持 token 成本和预算治理。
10. 提供 Web Console、API、SDK 和 CLI。

LLMOps 平台本质上是企业大模型应用的控制平面。

## 54.5 总体架构

可以设计如下架构：

```text
Users / Apps / Agents
  -> LLMOps API Gateway
  -> Auth / Tenant / Quota
  -> Model Gateway / Router
  -> Prompt Registry
  -> Knowledge Base / RAG Service
  -> Tool Registry / Agent Runtime
  -> Eval Platform
  -> Release / Experiment System
  -> Trace / Observability
  -> Cost / Audit / Governance
  -> Model Registry / Inference Platform
```

底层依赖推理平台、模型仓库、向量索引、评估平台和可观测性平台。

LLMOps 把这些能力包装成应用开发者能使用的产品。

## 54.6 Model Gateway

Model Gateway 是统一模型调用入口。

能力包括：

1. 多模型接入。
2. OpenAI-compatible API。
3. 模型路由。
4. 租户鉴权。
5. 限流配额。
6. 统一日志和 trace。
7. token 计量。
8. fallback 和降级。

它让应用不直接绑定某个模型供应商或某个内部 endpoint。

## 54.7 Prompt Registry

Prompt Registry 管理 prompt 模板。

它应支持：

1. prompt 创建。
2. prompt 版本。
3. 变量 schema。
4. prompt 预览。
5. 测试样例。
6. 评估结果。
7. 发布状态。
8. 灰度和回滚。
9. 权限控制。
10. 审计记录。

Prompt 是大模型应用的重要资产。

不要把 prompt 当成代码里的硬编码字符串。

## 54.8 Prompt 发布流程

Prompt 发布可以类似模型发布：

```text
draft -> reviewed -> evaluated -> staged -> canary -> production -> deprecated
```

发布前检查：

1. 变量是否完整。
2. 是否通过格式测试。
3. 是否通过安全测试。
4. 是否通过评估集。
5. 是否影响工具调用。
6. 是否有回滚版本。

Prompt 修改可能导致输出格式、拒答率和工具调用行为变化，因此必须可灰度和回滚。

## 54.9 Knowledge Base 管理

RAG 知识库管理包括：

1. 知识库创建。
2. 文档导入。
3. 文档解析。
4. chunking。
5. embedding。
6. 向量索引。
7. 权限同步。
8. 索引版本。
9. 检索配置。
10. citation。

企业知识库必须支持权限过滤。

不能让用户检索到自己无权访问的文档。

## 54.10 RAG Pipeline

RAG 请求链路：

```text
user query
  -> query rewrite
  -> retrieval
  -> metadata permission filter
  -> rerank
  -> prompt assembly
  -> model generation
  -> citation
  -> trace
```

平台要记录 retrieval trace 和 prompt assembly trace。

否则无法解释为什么模型引用了某些文档，或者为什么没有召回正确知识。

## 54.11 Tool Registry

Tool Registry 管理可供 Agent 调用的工具。

Tool Definition 包括：

1. tool name。
2. version。
3. description。
4. input schema。
5. output schema。
6. endpoint。
7. timeout。
8. retry policy。
9. permission requirement。
10. side effect level。

工具描述会直接影响模型调用行为，因此工具也需要版本化、评估和灰度。

## 54.12 Agent Runtime

Agent Runtime 负责任务执行。

它需要支持：

1. agent definition。
2. 多步执行。
3. 工具调用。
4. memory。
5. run state。
6. 超时。
7. step limit。
8. human confirmation。
9. execution trace。
10. error handling。

Agent 不是一次模型调用，而是一个多步状态机。

平台必须限制无限循环和高风险工具调用。

## 54.13 权限和安全

LLMOps 平台安全包括：

1. 模型访问权限。
2. prompt 修改权限。
3. 知识库访问权限。
4. 工具调用权限。
5. trace 查看权限。
6. secret 管理。
7. 输入输出安全检查。
8. 审计日志。

特别注意：应用开发者不一定能访问所有模型、所有知识库、所有工具。

权限要跟租户、项目、环境和数据等级绑定。

## 54.14 Eval Platform 集成

LLMOps 平台必须集成评估。

评估对象包括：

1. 模型版本。
2. prompt 版本。
3. RAG 配置。
4. 知识库版本。
5. 工具定义版本。
6. Agent 配置。

评估方式：

1. 离线评估。
2. LLM-as-Judge。
3. 人工评测。
4. 在线 A/B 测试。
5. 回归测试。

没有评估，LLMOps 只能靠感觉发布。

## 54.15 Release System

发布系统管理应用版本。

一个 LLM 应用版本可能包括：

1. model route policy。
2. prompt version。
3. RAG config。
4. knowledge base version。
5. tool versions。
6. safety policy。
7. generation config。

这些组合应该形成 release manifest。

发布支持：

1. 灰度。
2. A/B 测试。
3. 回滚。
4. 审批。
5. 变更记录。

## 54.16 Trace 和可观测性

LLMOps trace 应覆盖：

1. model call trace。
2. prompt assembly trace。
3. retrieval trace。
4. rerank trace。
5. tool call trace。
6. agent execution trace。
7. safety events。
8. cost trace。

核心指标：

1. latency。
2. TTFT。
3. TPOT。
4. token usage。
5. cost。
6. cache hit rate。
7. retrieval hit rate。
8. tool success rate。
9. user feedback。
10. safety violation rate。

大模型应用没有 trace 就很难 debug。

## 54.17 用户反馈闭环

企业 LLMOps 平台要支持反馈收集。

反馈类型：

1. 点赞点踩。
2. 用户评论。
3. 人工接管。
4. 任务成功失败。
5. 引用是否有用。
6. 工具调用是否正确。
7. 安全投诉。

反馈用于：

1. 评估数据构建。
2. 回归集更新。
3. prompt 优化。
4. RAG 调优。
5. 模型选择。
6. 安全规则改进。

反馈要和 trace 关联。

## 54.18 成本治理

LLMOps 成本治理关注：

1. token usage。
2. cost per app。
3. cost per tenant。
4. cost per user。
5. cost per model。
6. RAG 检索成本。
7. rerank 成本。
8. tool API 成本。
9. agent run 成本。

Agent 特别容易成本失控，因为它会多轮调用模型和工具。

平台要支持 budget、quota、step limit、max tokens 和 cost alert。

## 54.19 多环境管理

企业平台通常有多个环境：

1. dev。
2. staging。
3. production。

LLM 应用在不同环境中应有独立配置：

1. 模型版本。
2. prompt 版本。
3. 知识库版本。
4. 工具 endpoint。
5. 安全策略。
6. 成本配额。

从 dev 到 production 应通过发布流程，而不是手工复制配置。

## 54.20 Web Console 设计

Console 功能可以包括：

1. 应用列表。
2. 模型网关配置。
3. Prompt 管理。
4. 知识库管理。
5. 工具管理。
6. Agent 配置。
7. 评估任务。
8. 发布和灰度。
9. Trace 查看。
10. 成本和配额。
11. 权限和审计。

不同角色看到的页面不同。

应用开发者关注调试和发布，平台管理员关注安全、成本和稳定性。

## 54.21 API / SDK / CLI

平台应提供统一接口。

SDK 示例能力：

1. 调用模型。
2. 获取 prompt version。
3. 查询知识库。
4. 调用 agent。
5. 上传反馈。
6. 获取 trace ID。

CLI 示例：

```bash
llmops prompt publish customer-bot --version v12
llmops kb sync product-docs
llmops eval run customer-bot --suite regression
llmops release canary customer-bot --percent 5
llmops trace get <trace-id>
```

API、SDK、CLI 和 Console 应基于同一套资源模型。

## 54.22 核心数据模型

核心对象：

1. Application。
2. Environment。
3. PromptTemplate。
4. KnowledgeBase。
5. ToolDefinition。
6. AgentDefinition。
7. EvalSuite。
8. ReleaseManifest。
9. Trace。
10. Feedback。
11. Policy。
12. CostRecord。

这些对象构成 LLMOps 平台的控制平面。

## 54.23 LLMOps 平台系统设计指标和最小 demo

企业级 LLMOps 平台的系统设计，最好从“资源对象 + 生命周期门禁”讲起。一个应用样本可以抽象为：

```math
A_i=(m_i,p_i,k_i,t_i,g_i,e_i,r_i,o_i,f_i,c_i,s_i,u_i,z_i)
```

其中，`m_i` 是模型网关和路由，`p_i` 是 prompt 版本，`k_i` 是知识库和 RAG 配置，`t_i` 是工具与 Agent runtime，`g_i` 是治理策略，`e_i` 是评估，`r_i` 是 release manifest，`o_i` 是 trace 和观测，`f_i` 是用户反馈，`c_i` 是成本预算，`s_i` 是安全审计，`u_i` 是开发者接口和用户体验，`z_i` 是未关闭风险。

第 `j` 个设计维度覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(A_i)=1]
```

其中，`g_j` 是检查函数，例如是否有 prompt 版本、知识库权限、工具权限、评估回归、发布回滚、trace、成本归因和审计。这个公式的直觉是：LLMOps 不是一个功能列表，而是一组应用生命周期对象是否都可版本化、可验证、可回滚、可审计。

一次应用发布可以用 release manifest 表示：

```math
M_{\mathrm{rel}}=(v_{\mathrm{model}},v_{\mathrm{prompt}},v_{\mathrm{kb}},v_{\mathrm{tool}},v_{\mathrm{agent}},v_{\mathrm{safety}},\theta_{\mathrm{gen}})
```

其中，`v_{\mathrm{model}}` 是模型或路由策略版本，`v_{\mathrm{prompt}}` 是 prompt 版本，`v_{\mathrm{kb}}` 是知识库版本，`v_{\mathrm{tool}}` 是工具版本集合，`v_{\mathrm{agent}}` 是 Agent 配置版本，`v_{\mathrm{safety}}` 是安全策略版本，`\theta_{\mathrm{gen}}` 是生成参数。Manifest 不完整时，即使模型输出变差，也很难回放“到底变了什么”。

Manifest 完整率可以写成：

```math
C_{\mathrm{manifest}}=\frac{N_{\mathrm{filled}}}{N_{\mathrm{required}}}
```

Prompt 变量覆盖率可以写成：

```math
C_{\mathrm{prompt}}=\frac{|V_{\mathrm{provided}}\cap V_{\mathrm{required}}|}{|V_{\mathrm{required}}|}
```

其中，`V_{\mathrm{required}}` 是 prompt 模板声明的变量集合，`V_{\mathrm{provided}}` 是发布或调用时实际提供的变量集合。这个指标能阻断“模板改了但应用代码没改”的隐性事故。

RAG 权限阻断率可以写成：

```math
B_{\mathrm{rag}}=\frac{N_{\mathrm{blocked,unauth}}}{N_{\mathrm{unauth}}}
```

其中，`N_{\mathrm{unauth}}` 是无权访问但被检索候选召回的文档数，`N_{\mathrm{blocked,unauth}}` 是最终被权限过滤阻断的文档数。企业 RAG 的核心不是召回越多越好，而是正确内容和正确权限同时成立。

Agent 高风险工具确认覆盖率可以写成：

```math
C_{\mathrm{confirm}}=\frac{N_{\mathrm{confirmed,high}}}{N_{\mathrm{high}}}
```

其中，`N_{\mathrm{high}}` 是高风险工具调用次数，`N_{\mathrm{confirmed,high}}` 是经过人审、二次确认或审批策略放行的次数。这个指标提醒你：Agent 自动化能力越强，越要把副作用和权限边界显式化。

Trace 覆盖率可以写成：

```math
R_{\mathrm{trace}}=\frac{N_{\mathrm{trace,complete}}}{N_{\mathrm{runs}}}
```

完整 trace 至少要覆盖 model call、prompt assembly、retrieval、rerank、tool call、agent step、safety event、feedback 和 cost record。否则线上 bad case 很难复现，也无法沉淀成回归集。

反馈行动率可以写成：

```math
C_{\mathrm{feedback}}=\frac{N_{\mathrm{actioned}}}{N_{\mathrm{feedback}}}
```

其中，`N_{\mathrm{actioned}}` 是已经进入回归集、prompt 修复、知识库更新、工具策略修复或安全规则更新的反馈数。只收集点赞点踩但不进入行动闭环，不算真正的 LLMOps。

一次 Agent / RAG 应用运行成本可以拆成：

```math
K_{\mathrm{run}}=K_{\mathrm{model}}+K_{\mathrm{retrieval}}+K_{\mathrm{rerank}}+K_{\mathrm{tool}}+K_{\mathrm{review}}
```

企业平台最终要能按应用、租户、用户、模型、知识库和工具做成本归因，而不是只看统一账单。

评估发布门禁可以写成：

```math
G_{\mathrm{eval}}=\mathbf{1}\left[Q_{\mathrm{offline}}\ge\tau_q \land S_{\mathrm{safety}}\ge\tau_s \land R_{\mathrm{reg}}\ge\tau_r\right]
```

其中，`Q_{\mathrm{offline}}` 是离线质量指标，`S_{\mathrm{safety}}` 是安全评估通过率，`R_{\mathrm{reg}}` 是回归集通过率。Prompt、RAG 配置和工具 schema 的变更都应该走类似门禁。

最终 LLMOps 平台门禁可以写成：

```math
G_{\mathrm{llmops}}=\mathbf{1}\left[
\min_j C_j\ge\tau_j \land
C_{\mathrm{manifest}}=1 \land
B_{\mathrm{rag}}\ge\rho_{\mathrm{rag}} \land
C_{\mathrm{confirm}}\ge\rho_{\mathrm{tool}} \land
R_{\mathrm{trace}}\ge\rho_{\mathrm{trace}} \land
G_{\mathrm{eval}}=1 \land
P_0=0
\right]
```

其中，`P_0` 是未关闭 P0 风险。这个门禁强调：企业级 LLMOps 的答案不能只讲“有 prompt 管理和知识库管理”，还要证明版本、权限、评估、发布、trace、反馈、成本和安全能闭环。

下面这个 0 依赖 Python demo 演示一个“企业级 LLMOps 平台系统设计审计器”：输入 toy applications、release manifest、RAG 候选、工具调用、trace、反馈和成本记录，输出生命周期、成本和设计门禁结果。

```python
class MiniEnterpriseLLMOpsPlatformAudit:
    CHECKS = [
        "lifecycle_boundary_clarity",
        "application_resource_model",
        "model_gateway_policy",
        "prompt_registry_versioning",
        "rag_kb_permission_governance",
        "tool_agent_runtime_control",
        "eval_feedback_loop",
        "release_manifest_governance",
        "trace_observability_readiness",
        "cost_budget_governance",
        "security_audit_policy",
        "multi_environment_promotion",
        "developer_interface_readiness",
        "production_monitoring_guardrail",
        "tradeoff_boundary_reasoning",
        "llmops_platform_gate",
    ]

    REQUIRED_MANIFEST_FIELDS = [
        "model_route_version",
        "prompt_version",
        "knowledge_base_version",
        "tool_versions",
        "agent_version",
        "safety_policy_version",
        "generation_config",
        "rollback_target",
    ]

    def __init__(self):
        self.applications = [
            {"name": "customer_bot", "environment": "production", "eval_ready": True, "release_ready": True},
            {"name": "ops_copilot", "environment": "production", "eval_ready": True, "release_ready": True},
            {"name": "hr_agent", "environment": "staging", "eval_ready": False, "release_ready": False},
        ]
        self.release_manifest = {
            "model_route_version": "route-v7",
            "prompt_version": "prompt-v12",
            "knowledge_base_version": "kb-2026-06-01",
            "tool_versions": ["ticket-v3", "refund-v2"],
            "agent_version": "agent-v5",
            "safety_policy_version": "safety-v9",
            "generation_config": {"temperature": 0.2, "max_tokens": 800},
            "rollback_target": "release-v41",
        }
        self.prompt = {
            "required_variables": {"customer_tier", "issue_summary", "locale"},
            "provided_variables": {"customer_tier", "issue_summary", "locale"},
        }
        self.retrieval_candidates = [
            {"doc": "public_faq", "authorized": True, "blocked": False},
            {"doc": "enterprise_contract", "authorized": True, "blocked": False},
            {"doc": "hr_private_policy", "authorized": False, "blocked": True},
            {"doc": "finance_private_note", "authorized": False, "blocked": True},
        ]
        self.tool_calls = [
            {"tool": "search_docs", "high_risk": False, "confirmed": True},
            {"tool": "create_ticket", "high_risk": False, "confirmed": True},
            {"tool": "issue_refund", "high_risk": True, "confirmed": True},
            {"tool": "send_customer_email", "high_risk": True, "confirmed": True},
        ]
        self.trace_runs = [
            {"id": "r1", "complete": True},
            {"id": "r2", "complete": True},
            {"id": "r3", "complete": True},
            {"id": "r4", "complete": False},
            {"id": "r5", "complete": True},
            {"id": "r6", "complete": True},
            {"id": "r7", "complete": True},
            {"id": "r8", "complete": True},
            {"id": "r9", "complete": False},
        ]
        self.feedback = {"total": 64, "actioned": 48}
        self.cost_records = [
            {"app": "customer_bot", "tokens": 240000, "cost": 9.9, "success": 160, "budget": 12.0},
            {"app": "ops_copilot", "tokens": 170000, "cost": 3.1, "success": 80, "budget": 5.0},
            {"app": "hr_agent", "tokens": 50000, "cost": 2.0, "success": 20, "budget": 2.0},
        ]

    def lifecycle_summary(self):
        total = len(self.applications)
        production = sum(1 for app in self.applications if app["environment"] == "production")
        eval_ready = sum(1 for app in self.applications if app["eval_ready"])
        release_ready = sum(1 for app in self.applications if app["release_ready"])
        return {
            "apps": total,
            "production_apps": production,
            "eval_ready_rate": round(eval_ready / total, 3),
            "release_ready_rate": round(release_ready / total, 3),
        }

    def release_manifest_completeness(self):
        filled = sum(1 for field in self.REQUIRED_MANIFEST_FIELDS if self.release_manifest.get(field))
        return round(filled / len(self.REQUIRED_MANIFEST_FIELDS), 3)

    def prompt_variable_coverage(self):
        required = self.prompt["required_variables"]
        provided = self.prompt["provided_variables"]
        return round(len(required & provided) / len(required), 3)

    def rag_permission_block_rate(self):
        unauthorized = [item for item in self.retrieval_candidates if not item["authorized"]]
        blocked = [item for item in unauthorized if item["blocked"]]
        return round(len(blocked) / len(unauthorized), 3)

    def tool_confirmation_coverage(self):
        high_risk = [call for call in self.tool_calls if call["high_risk"]]
        confirmed = [call for call in high_risk if call["confirmed"]]
        return round(len(confirmed) / len(high_risk), 3)

    def trace_coverage(self):
        complete = sum(1 for run in self.trace_runs if run["complete"])
        return round(complete / len(self.trace_runs), 3)

    def feedback_action_rate(self):
        return round(self.feedback["actioned"] / self.feedback["total"], 3)

    def cost_summary(self):
        total_tokens = sum(item["tokens"] for item in self.cost_records)
        total_cost = round(sum(item["cost"] for item in self.cost_records), 2)
        total_success = sum(item["success"] for item in self.cost_records)
        budget_pass = all(item["cost"] <= item["budget"] for item in self.cost_records)
        return {
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "cost_per_success": round(total_cost / total_success, 3),
            "budget_pass": budget_pass,
        }

    def llmops_examples(self):
        return {
            "release_manifest_completeness": self.release_manifest_completeness(),
            "prompt_variable_coverage": self.prompt_variable_coverage(),
            "rag_permission_block_rate": self.rag_permission_block_rate(),
            "tool_confirmation_coverage": self.tool_confirmation_coverage(),
            "trace_coverage": self.trace_coverage(),
            "feedback_action_rate": self.feedback_action_rate(),
        }

    def build_design_cases(self):
        complete = {"name": "complete_enterprise_llmops_platform"}
        complete.update({check: True for check in self.CHECKS})
        cases = [complete]
        bad_names = [
            "lifecycle_boundary_missing_bad",
            "resource_model_missing_bad",
            "model_gateway_policy_missing_bad",
            "prompt_registry_missing_bad",
            "rag_permission_missing_bad",
            "agent_tool_control_missing_bad",
            "eval_feedback_loop_missing_bad",
            "release_manifest_missing_bad",
            "trace_observability_missing_bad",
            "cost_budget_missing_bad",
            "security_audit_missing_bad",
            "environment_promotion_manual_bad",
            "developer_interface_missing_bad",
            "production_guardrail_missing_bad",
            "tradeoff_boundary_missing_bad",
            "llmops_gate_missing_bad",
        ]
        for bad_name, failed_check in zip(bad_names, self.CHECKS):
            case = {"name": bad_name}
            case.update({check: True for check in self.CHECKS})
            case[failed_check] = False
            cases.append(case)
        return cases

    def audit_cases(self, cases):
        metrics = {
            check: round(sum(1 for case in cases if case.get(check)) / len(cases), 3)
            for check in self.CHECKS
        }
        failed_cases = [case["name"] for case in cases if not all(case.get(check) for check in self.CHECKS)]
        failed_gates = [check for check, value in metrics.items() if value < 1.0]
        hard_blockers = sum(1 for case in cases if case["name"].endswith("_bad"))
        gate_pass = min(metrics.values()) >= 0.95 and hard_blockers == 0
        return metrics, failed_cases, failed_gates, hard_blockers, gate_pass

    def run(self):
        metrics, failed_cases, failed_gates, hard_blockers, gate_pass = self.audit_cases(
            self.build_design_cases()
        )
        return {
            "lifecycle_summary": self.lifecycle_summary(),
            "llmops_examples": self.llmops_examples(),
            "cost_summary": self.cost_summary(),
            "metrics": metrics,
            "hard_blocker_count": hard_blockers,
            "failed_case_count": len(failed_cases),
            "failed_gate_count": len(failed_gates),
            "failed_case_sample": failed_cases[:4],
            "failed_gate_sample": failed_gates[:4],
            "llmops_platform_gate_pass": gate_pass,
        }


report = MiniEnterpriseLLMOpsPlatformAudit().run()
print("lifecycle_summary=", report["lifecycle_summary"])
print("llmops_examples=", report["llmops_examples"])
print("cost_summary=", report["cost_summary"])
print("metrics=", report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_case_count=", report["failed_case_count"])
print("failed_gate_count=", report["failed_gate_count"])
print("failed_case_sample=", report["failed_case_sample"])
print("failed_gate_sample=", report["failed_gate_sample"])
print("llmops_platform_gate_pass=", report["llmops_platform_gate_pass"])
```

输出示例：

```text
lifecycle_summary= {'apps': 3, 'production_apps': 2, 'eval_ready_rate': 0.667, 'release_ready_rate': 0.667}
llmops_examples= {'release_manifest_completeness': 1.0, 'prompt_variable_coverage': 1.0, 'rag_permission_block_rate': 1.0, 'tool_confirmation_coverage': 1.0, 'trace_coverage': 0.778, 'feedback_action_rate': 0.75}
cost_summary= {'total_tokens': 460000, 'total_cost': 15.0, 'cost_per_success': 0.058, 'budget_pass': True}
metrics= {'lifecycle_boundary_clarity': 0.941, 'application_resource_model': 0.941, 'model_gateway_policy': 0.941, 'prompt_registry_versioning': 0.941, 'rag_kb_permission_governance': 0.941, 'tool_agent_runtime_control': 0.941, 'eval_feedback_loop': 0.941, 'release_manifest_governance': 0.941, 'trace_observability_readiness': 0.941, 'cost_budget_governance': 0.941, 'security_audit_policy': 0.941, 'multi_environment_promotion': 0.941, 'developer_interface_readiness': 0.941, 'production_monitoring_guardrail': 0.941, 'tradeoff_boundary_reasoning': 0.941, 'llmops_platform_gate': 0.941}
hard_blocker_count= 16
failed_case_count= 16
failed_gate_count= 16
failed_case_sample= ['lifecycle_boundary_missing_bad', 'resource_model_missing_bad', 'model_gateway_policy_missing_bad', 'prompt_registry_missing_bad']
failed_gate_sample= ['lifecycle_boundary_clarity', 'application_resource_model', 'model_gateway_policy', 'prompt_registry_versioning']
llmops_platform_gate_pass= False
```

这个 demo 的重点不是生产系统容量结论，而是把面试答案变成可审计清单：一个企业级 LLMOps 平台如果不能说明应用资源模型、release manifest、prompt / KB / tool / agent 版本、评估回归、trace、反馈、权限、成本和发布回滚，就只是一个功能入口，不是可靠的平台控制面。

## 54.24 关键 trade-off

LLMOps 平台 trade-off：

1. 平台统一治理 vs 团队灵活性。
2. Trace 完整性 vs 隐私和成本。
3. Prompt 快速迭代 vs 发布风险。
4. RAG 召回更多内容 vs 权限和噪声风险。
5. Agent 自动化能力 vs 工具副作用风险。
6. 多模型路由优化成本 vs 输出一致性。
7. 强审批流程 vs 研发效率。

企业平台的难点往往不是技术组件，而是治理边界和协作流程。

## 54.25 面试回答模板

可以这样回答：

```text
我会把企业级 LLMOps 平台设计成大模型应用生命周期管理平台，覆盖模型调用、prompt、RAG、Agent、评估、发布、观测、安全和成本。

架构上包括 Model Gateway、Prompt Registry、Knowledge Base/RAG Service、Tool Registry、Agent Runtime、Eval Platform、Release System、Trace/Observability、Cost/Audit/Governance。底层对接模型仓库和推理平台。

应用开发者通过 Web Console、API、SDK 或 CLI 管理 prompt、知识库、工具和 agent 配置。每次发布生成 release manifest，包含模型路由、prompt 版本、RAG 配置、工具版本、安全策略和推理参数。发布前跑评估和回归，发布时支持灰度、A/B 测试和回滚。

运行中记录 model call、retrieval、prompt assembly、tool call 和 agent execution trace，并采集 token、latency、cost、feedback 和 safety 指标。平台通过租户权限、模型访问控制、知识库 ACL、工具权限、审计和预算配额保证企业治理。
```

这个回答体现了企业级 LLMOps 的完整边界。

## 54.26 常见扣分点

扣分点一：只说模型部署。

LLMOps 不只是部署模型，还包括 prompt、RAG、Agent、评估、发布和治理。

扣分点二：不提 prompt 版本。

Prompt 是应用行为的重要组成，必须版本化和灰度。

扣分点三：不提知识库权限。

企业 RAG 的核心风险是越权检索。

扣分点四：不提 trace。

没有 trace，RAG/Agent 应用无法调试和审计。

扣分点五：不提成本。

企业平台需要 token 计量、预算、配额和成本归因。

## 54.27 小练习

1. LLMOps 和 MLOps 的核心区别是什么？
2. Prompt Registry 应该支持哪些能力？
3. 一个 LLM 应用 release manifest 应包含哪些内容？
4. 企业 RAG 平台如何做权限控制？
5. Agent Runtime 为什么需要 step limit 和 human confirmation？
6. LLMOps trace 应覆盖哪些阶段？
7. 用户反馈如何进入模型和应用迭代闭环？
8. 如何设计 LLMOps 平台的成本治理？

## 54.28 本章小结

本章系统设计了一个企业级 LLMOps 平台。

你需要记住：

1. LLMOps 关注大模型应用生命周期，而不只是模型训练或部署。
2. 企业级 LLMOps 平台要管理模型调用、prompt、RAG、Agent、工具、评估、发布、trace、安全和成本。
3. Prompt、知识库、工具和 Agent 配置都要版本化、评估、灰度和回滚。
4. RAG/Agent 场景中，trace、权限和审计是核心能力。
5. 企业平台要在研发效率、治理、安全、成本和灵活性之间做平衡。

下一章我们会设计一个多租户 GPU 集群调度系统。
