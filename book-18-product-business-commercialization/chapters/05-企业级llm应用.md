# 第五章：企业级 LLM 应用

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修。联网资料主要核对五类口径：OpenAI 企业数据隐私和安全资料提醒我们，企业级应用必须关注数据所有权、默认训练使用边界、加密、数据保留、SSO、RBAC、审计日志和用量治理；NIST AI RMF / Generative AI Profile 强调生成式 AI 风险要进入 govern、map、measure、manage 的风险管理闭环；OWASP LLM Top 10 资料提醒我们，企业 LLM 应用要特别警惕提示注入、敏感信息泄露、向量与嵌入弱点、过度代理和无界资源消耗等应用层风险；Microsoft RBAC / Zero Trust 资料用于核对 IAM、最小权限、角色授权和身份控制口径；Google SRE 的 SLI / SLO / error budget 资料用于核对企业服务可靠性和延迟门禁。

本章不替代企业安全架构、法务合规审查、采购合同、真实 IAM 设计或行业监管要求；后续第九章会专门展开隐私合规与治理，第六章会专门展开 RAG 产品落地。本章的重点是让算法工程师在面试和项目复盘中能讲清：企业级应用为什么不能只做聊天入口，为什么权限、租户隔离、数据治理、审计、SLO、人审和业务指标要一起进入上线门禁。

企业级 LLM 应用和个人消费级应用有很大区别。个人用户更关注好不好用、有不有趣；企业更关注能否接入现有系统、是否符合权限和合规要求、是否能节省成本、是否能稳定服务多人协作和复杂流程。企业级应用不是简单加一个聊天框，而是把大模型嵌入真实业务流程。

本章系统讲企业级 LLM 应用：客服、知识库、代码助手、数据分析、办公自动化、行业助手等典型场景，以及企业落地中的权限、系统集成、数据治理、工作流、审计、运维、ROI 和评估。

## 5.1 企业级应用的特点

企业级 LLM 应用通常有这些特点：

1. 数据来自企业内部。
2. 权限和合规要求高。
3. 需要接入现有系统。
4. 用户角色复杂。
5. 任务结果影响业务流程。
6. 要有审计和监控。
7. 需要可衡量 ROI。
8. 不能只靠 demo 效果。

面试回答：

```text
企业级 LLM 应用的核心不是把模型接进来，而是把模型嵌入企业流程。它需要解决内部数据接入、权限控制、系统集成、稳定性、审计合规、用户培训和 ROI 评估。典型场景包括企业知识库、客服助手、代码助手、数据分析助手和办公自动化。
```

## 5.2 企业知识库问答

企业知识库是最常见的 LLM 应用。

目标：让员工快速找到内部制度、产品文档、技术文档、流程说明和历史经验。

关键能力：

1. 文档解析。
2. 权限过滤。
3. 语义检索。
4. 答案生成。
5. 引用来源。
6. 多轮追问。
7. 反馈纠错。

常见问题：

1. 文档过期。
2. 权限复杂。
3. 召回不准。
4. 引用错误。
5. 用户问法和文档表达不一致。
6. 缺少标准答案评估。

知识库问答的成败，很大程度取决于数据治理和权限设计，而不只是模型能力。

## 5.3 智能客服

智能客服是高频场景。

价值来源：

1. 降低人工客服压力。
2. 提高自助解决率。
3. 缩短响应时间。
4. 提升回复一致性。
5. 总结用户问题。
6. 辅助人工客服。

客服场景要关注：

1. 意图识别。
2. 知识库检索。
3. 多轮澄清。
4. 转人工策略。
5. 情绪识别。
6. 合规话术。
7. 用户满意度。

客服机器人不能为了减少转人工而强行回答。无法确认时及时转人工，反而更能保护体验。

## 5.4 客服 Copilot 和全自动客服

客服场景有两种形态。

客服 Copilot：

1. 给人工客服推荐答案。
2. 总结用户历史。
3. 提醒风险话术。
4. 自动生成工单摘要。

全自动客服：

1. 直接面对用户回答。
2. 自动处理简单问题。
3. 必要时转人工。

早期更推荐 Copilot，因为风险低、容易积累数据、人工能纠错。全自动适合边界清楚、知识稳定、风险低的问题。

## 5.5 代码助手

企业代码助手可以帮助：

1. 代码补全。
2. 单元测试生成。
3. 代码解释。
4. Bug 定位。
5. 代码审查。
6. 文档生成。
7. 迁移和重构辅助。

企业代码助手要关注：

1. 代码隐私。
2. 仓库权限。
3. 许可证风险。
4. 安全漏洞。
5. 与 IDE 和 CI 集成。
6. 是否符合团队规范。

代码助手的价值指标可以是开发效率、review 通过率、测试覆盖率、缺陷减少和新人上手速度。

## 5.6 数据分析助手

数据分析助手帮助业务人员用自然语言分析数据。

能力包括：

1. 自然语言转 SQL。
2. 指标解释。
3. 图表生成。
4. 异常分析。
5. 报告生成。
6. 数据口径解释。

难点：

1. 指标口径复杂。
2. 数据权限严格。
3. SQL 生成错误可能误导决策。
4. 需要防止查询敏感数据。
5. 需要可追溯计算过程。

数据分析助手不能只生成漂亮图表，还要保证口径、权限和可验证性。

## 5.7 办公自动化

办公自动化场景包括：

1. 会议纪要。
2. 邮件草稿。
3. 周报生成。
4. 文档润色。
5. PPT 大纲。
6. 表格处理。
7. 日程安排。
8. 工单摘要。

这类场景容错率较高，适合作为企业大模型试点。

但也要注意：

1. 不要泄露会议敏感信息。
2. 邮件发送前要人工确认。
3. 生成内容要符合企业风格。
4. 结果要可编辑。

## 5.8 行业助手

行业助手针对特定行业。

例如：

1. 金融投研助手。
2. 法律合同助手。
3. 医疗病历助手。
4. 教育备课助手。
5. 制造运维助手。
6. 保险理赔助手。

行业场景价值高，但要求也高：

1. 专业知识准确。
2. 数据合规。
3. 结果可追溯。
4. 风险可控。
5. 需要专家评估。
6. 不能替代最终责任人。

行业助手通常适合先做人机协同，而不是直接全自动决策。

## 5.9 企业系统集成

企业级 LLM 应用需要接入现有系统。

常见系统：

1. SSO 和身份系统。
2. 权限系统。
3. 文档系统。
4. CRM。
5. ERP。
6. 工单系统。
7. 数据仓库。
8. BI 系统。
9. 代码仓库。
10. 审计系统。

集成难点往往比模型难点更大。没有系统集成，AI 功能很难进入真实工作流。

企业集成至少要分三层看：

1. 身份层：SSO、MFA、SCIM、用户组、服务账号和离职回收。
2. 数据层：文档系统、数据仓库、代码仓库、知识库、向量库和日志系统。
3. 工作流层：工单、审批、CRM、ERP、BI、CI/CD 和人工复核入口。

面试中不要只说“接入企业系统”。更成熟的说法是：先接身份和权限，再接数据和知识库，最后把模型输出接入已有工作流；如果没有审计、回滚和人工接管，就不要直接让模型驱动高风险动作。

## 5.10 权限控制

企业应用必须做权限控制。

需要保证：

1. 用户只能访问有权限的数据。
2. RAG 检索不能越权召回。
3. Agent 工具不能越权调用。
4. 日志不能泄露敏感信息。
5. 不同租户数据隔离。
6. 管理员可审计访问记录。

权限控制不能只在前端做，必须贯穿检索、生成、工具调用和日志。

更具体地说，权限至少要穿过四个环节：

1. 检索权限：向量检索和关键词检索只能召回用户有权看的文档。
2. 生成权限：模型不能把无权信息通过总结、引用或多轮对话泄露出来。
3. 工具权限：Agent 调用工单、数据库、邮件、代码仓库等工具前要做角色和动作门禁。
4. 日志权限：prompt、检索片段、工具返回和模型输出写入日志前要脱敏并限制查看范围。

企业权限的关键不是“模型知道用户是谁”，而是后端系统在每次检索、工具调用、写日志和展示引用时都重新执行权限判断。只靠 prompt 告诉模型“不要泄露信息”，不是可靠权限控制。

## 5.11 数据治理

企业数据常见问题：

1. 文档过期。
2. 多版本冲突。
3. 命名不统一。
4. 权限不清。
5. 格式复杂。
6. 缺少元数据。
7. 没有标准答案。
8. 敏感信息混杂。

大模型不能自动解决数据治理问题。相反，数据治理差会放大模型幻觉和错误引用。

## 5.12 审计与合规

企业级应用需要审计。

审计内容：

1. 谁访问了什么数据。
2. 模型回答了什么。
3. 使用了哪些文档。
4. 调用了哪些工具。
5. 是否涉及敏感信息。
6. 是否有人审。
7. 是否发生异常。

合规要求因行业不同而不同。金融、医疗、法律和政企场景通常要求更严格。

## 5.13 上线策略

企业 LLM 应用不建议一次全量上线。

推荐路径：

1. 内部小范围试点。
2. 人机协同模式。
3. 收集反馈和失败样本。
4. 建立评估集。
5. 灰度扩大范围。
6. 再考虑自动化比例提升。

上线策略要和风险等级匹配。高风险场景必须更慢、更稳。

## 5.14 组织协作

企业级 LLM 项目需要多方协作。

参与方：

1. 业务团队。
2. 产品团队。
3. 算法团队。
4. 工程团队。
5. 数据团队。
6. 安全团队。
7. 法务合规。
8. 运维团队。

算法团队不能独立完成企业级落地。很多关键问题在数据、流程、权限和组织协作中。

## 5.15 企业应用评估指标

常见指标：

1. 使用率。
2. 任务完成率。
3. 用户满意度。
4. 人工节省时间。
5. 转人工率。
6. 答案正确率。
7. 引用准确率。
8. 响应延迟。
9. 单次任务成本。
10. 安全事件数。
11. 业务指标改善。

不同应用要选不同主指标。知识库看解决率和引用准确率；客服看自助解决率和满意度；代码助手看采纳率和测试通过率。

## 5.15.1 关键公式与企业级应用指标速查

可以把一个企业 LLM 应用样本写成：

```math
e_i=(u_i,d_i,p_i,\tau_i,w_i,a_i,r_i,l_i,m_i)
```

其中 `u_i` 是用户和角色集合，`d_i` 是接入的数据源，`p_i` 是权限策略，`\tau_i` 是租户或组织边界，`w_i` 是接入的业务工作流，`a_i` 是审计记录，`r_i` 是风险等级，`l_i` 是延迟观测，`m_i` 是业务指标。

**1. 权限通过率**

企业应用最核心的门禁之一，是检索、工具和日志都没有越权：

```math
R_{\mathrm{perm}}=\frac{1}{N}\sum_{i=1}^{N}h_i^{\mathrm{rag}}h_i^{\mathrm{tool}}h_i^{\mathrm{log}}
```

其中 `h_i` 是 0/1 指标。`h_i^{\mathrm{rag}}=1` 表示 RAG 检索没有越权召回；`h_i^{\mathrm{tool}}=1` 表示工具调用通过角色和动作校验；`h_i^{\mathrm{log}}=1` 表示日志没有泄露超出查看者权限的数据。直觉上，三层只要有一层失败，这次请求就不能算权限通过。

**2. 租户隔离违规率**

多租户企业产品要单独监控跨租户数据泄露：

```math
R_{\mathrm{viol}}=\frac{1}{N}\sum_{i=1}^{N}v_i
```

其中 `v_i=1` 表示第 `i` 次请求发生跨租户、跨部门或跨项目边界的违规访问。这个指标通常应接近 0；哪怕平均任务成功率很高，只要出现租户隔离事故，也不能直接上线。

**3. 引用支持率**

企业知识库和行业助手不能只要求“回答像真的”，还要要求关键结论能被授权证据支持：

```math
C_{\mathrm{cite}}=\frac{1}{N}\sum_{i=1}^{N}c_i
```

其中 `c_i=1` 表示第 `i` 个关键结论的引用存在、相关、足以支持结论、未过期，并且当前用户有权查看。引用支持率低时，产品应该更保守地展示“不确定”“需要人工复核”或触发知识库更新。

**4. 审计覆盖率**

企业应用需要能回答“谁在什么时候用什么身份访问了什么数据、调用了什么工具、得到什么结果”：

```math
C_{\mathrm{audit}}=\frac{1}{N}\sum_{i=1}^{N}a_i
```

其中 `a_i=1` 表示请求 trace 中包含用户身份、数据来源、引用文档、工具调用、权限判断、模型版本、输出摘要和人工复核状态。没有审计覆盖，事故发生后无法复盘，也无法证明系统按权限和流程运行。

**5. 数据新鲜度通过率**

企业知识库常见问题是文档过期。可以用文档年龄门禁粗略检查：

```math
C_{\mathrm{fresh}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_i \leq A_{\max}]
```

其中 `g_i` 是证据文档距当前时间的年龄，`A_{\max}` 是业务允许的最大文档年龄。不同业务阈值不同：报销制度可能按月更新，接口文档可能按版本更新，合规政策可能要求更严格的版本控制。

**6. SLO 通过率**

企业应用不仅要准，还要在业务流程可接受的时间内返回：

```math
R_{\mathrm{slo}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[l_i \leq L_{\mathrm{slo}}]
```

其中 `l_i` 是请求延迟，`L_{\mathrm{slo}}` 是场景定义的延迟目标。客服助手、代码补全、会议纪要、合同复核的 SLO 不一样，不能共用一个平均响应时间。

**7. 高风险人审覆盖率**

对金融、医疗、法律、合规、退款、权限变更等高风险任务，要看高风险样本是否有人审：

```math
C_{\mathrm{human}}=\frac{\sum_{i=1}^{N}z_i q_i}{\sum_{i=1}^{N}z_i+\epsilon}
```

其中 `z_i=1` 表示高风险任务，`q_i=1` 表示有人审、审批或二次确认，`\epsilon` 用于避免分母为 0。高风险任务不是不能用 LLM，而是不能跳过责任人、审批和审计。

**8. 企业就绪分与上线门禁**

一个简化企业就绪分可以写成：

```math
S_{\mathrm{ent}}=0.25R_{\mathrm{perm}}+0.15(1-R_{\mathrm{viol}})+0.15C_{\mathrm{cite}}+0.15C_{\mathrm{audit}}+0.10C_{\mathrm{fresh}}+0.10R_{\mathrm{slo}}+0.10C_{\mathrm{human}}
```

这个分数只用于面试和 toy demo 解释，真实项目要按行业和风险重新定权重。上线门禁可以写成：

```math
G_{\mathrm{ent}}=\mathbf{1}[R_{\mathrm{perm}}\geq 0.95]\mathbf{1}[R_{\mathrm{viol}}=0]\mathbf{1}[C_{\mathrm{audit}}\geq 0.90]\mathbf{1}[R_{\mathrm{slo}}\geq 0.95]\mathbf{1}[M_{\mathrm{biz}}=1]
```

其中 `M_{\mathrm{biz}}=1` 表示已定义业务主指标。直觉是：企业应用不是靠一个综合分上线，而是任何关键门禁不过线都要先修复。

## 5.16 常见失败模式

1. 只做聊天框，没有接入工作流。
2. 知识库数据质量差。
3. 权限控制不完整。
4. 缺少业务指标。
5. 用户不知道怎么用。
6. 模型回答无法追溯。
7. 成本没有监控。
8. 高风险场景没有人审。
9. 试点成功但无法规模化。
10. 组织协作不顺。

企业级 LLM 应用失败往往不是因为模型完全不行，而是因为没有解决企业环境中的系统问题。

## 5.17 面试题：企业级 LLM 应用和普通应用有什么区别

回答要点：

```text
企业级 LLM 应用更强调数据、权限、系统集成、稳定性、审计和 ROI。普通应用可能只关注用户体验和模型效果，但企业应用需要接入 SSO、知识库、工单、CRM、数据仓库等系统，保证用户只能访问有权限的数据，并记录审计日志。它还要能衡量业务指标，例如节省人工、提高解决率或降低成本。
```

## 5.18 面试题：如何落地企业知识库问答

回答要点：

```text
我会先做数据治理，整理文档来源、版本、权限和元数据，再构建 RAG 流程，包括文档解析、分块、embedding、检索、rerank、答案生成和引用。权限过滤必须贯穿检索和生成。上线前要建立评估集，评估召回率、答案正确率、引用准确率和无法回答率。上线后收集用户反馈，持续更新文档和优化检索。
```

## 5.19 最小可运行企业级应用审计 demo

下面这个 demo 用 0 依赖 Python 模拟企业级 LLM 应用上线审计。它不是生产治理系统，而是帮助你在面试中把“企业级”拆成可检查字段：身份、权限、租户隔离、RAG 权限过滤、工具权限、审计日志、PII 脱敏、数据新鲜度、引用支持、SSO、工作流、评估、反馈、SLO、业务指标和高风险人审。

```python
def mean(values):
    return sum(values) / len(values) if values else 0.0


def audit_app(app):
    high_risk = app["risk_level"] == "high"
    gates = {
        "permission_coverage": app["permission_coverage"] >= 0.90,
        "tenant_isolation": app["tenant_isolation"] >= 1.00,
        "rag_permission_filter": app["rag_permission_filter"] >= 0.95,
        "tool_permission_gate": app["tool_permission_gate"] >= 0.90,
        "audit_log_coverage": app["audit_log_coverage"] >= 0.90,
        "pii_redaction": app["pii_redaction"] >= 0.85,
        "data_freshness": app["data_freshness"] >= 0.75,
        "citation_support": app["citation_support"] >= 0.80,
        "sso_integration": app["sso_integration"] >= 1.00,
        "workflow_integration": app["workflow_integration"] >= 0.75,
        "eval_ready": app["eval_ready"] >= 0.80,
        "feedback_loop": app["feedback_loop"] >= 0.75,
        "sla_p95_latency_ok": app["sla_p95_latency_ok"] >= 1.00,
        "business_metric_defined": app["business_metric_defined"] >= 1.00,
        "human_review_coverage": (not high_risk) or app["human_review_coverage"] >= 0.80,
    }

    permission_score = mean([
        app["permission_coverage"],
        app["rag_permission_filter"],
        app["tool_permission_gate"],
    ])
    governance_score = mean([
        app["tenant_isolation"],
        app["audit_log_coverage"],
        app["pii_redaction"],
        app["data_freshness"],
    ])
    evidence_score = mean([
        app["citation_support"],
        app["eval_ready"],
        app["feedback_loop"],
        app["business_metric_defined"],
    ])
    integration_score = mean([
        app["sso_integration"],
        app["workflow_integration"],
    ])
    ops_score = mean([
        app["sla_p95_latency_ok"],
        app["human_review_coverage"] if high_risk else 1.0,
    ])

    enterprise_score = (
        0.30 * permission_score
        + 0.22 * governance_score
        + 0.18 * evidence_score
        + 0.15 * integration_score
        + 0.15 * ops_score
    )
    failed_gates = [name for name, ok in gates.items() if not ok]
    return {
        "name": app["name"],
        "scenario_type": app["scenario_type"],
        "enterprise_score": round(enterprise_score, 3),
        "enterprise_gate": not failed_gates,
        "failed_gates": failed_gates,
    }


apps = [
    {
        "name": "support_kb_rag",
        "scenario_type": "enterprise_knowledge_base",
        "users": 1800,
        "data_sources": 7,
        "permission_coverage": 0.96,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.98,
        "tool_permission_gate": 0.94,
        "audit_log_coverage": 0.93,
        "pii_redaction": 0.91,
        "data_freshness": 0.88,
        "citation_support": 0.90,
        "sso_integration": 1.00,
        "workflow_integration": 0.86,
        "eval_ready": 0.84,
        "feedback_loop": 0.78,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.82,
        "risk_level": "high",
    },
    {
        "name": "contract_copilot",
        "scenario_type": "legal_assistant",
        "users": 120,
        "data_sources": 5,
        "permission_coverage": 0.91,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.94,
        "tool_permission_gate": 0.88,
        "audit_log_coverage": 0.90,
        "pii_redaction": 0.86,
        "data_freshness": 0.70,
        "citation_support": 0.82,
        "sso_integration": 1.00,
        "workflow_integration": 0.72,
        "eval_ready": 0.80,
        "feedback_loop": 0.62,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.71,
        "risk_level": "high",
    },
    {
        "name": "data_analyst_nl2sql",
        "scenario_type": "data_analysis",
        "users": 260,
        "data_sources": 9,
        "permission_coverage": 0.86,
        "tenant_isolation": 1.00,
        "rag_permission_filter": 0.90,
        "tool_permission_gate": 0.73,
        "audit_log_coverage": 0.82,
        "pii_redaction": 0.79,
        "data_freshness": 0.93,
        "citation_support": 0.55,
        "sso_integration": 1.00,
        "workflow_integration": 0.84,
        "eval_ready": 0.72,
        "feedback_loop": 0.64,
        "sla_p95_latency_ok": 0.00,
        "business_metric_defined": 1.00,
        "human_review_coverage": 0.76,
        "risk_level": "high",
    },
    {
        "name": "office_summarizer",
        "scenario_type": "office_automation",
        "users": 2300,
        "data_sources": 4,
        "permission_coverage": 0.78,
        "tenant_isolation": 0.95,
        "rag_permission_filter": 0.80,
        "tool_permission_gate": 0.65,
        "audit_log_coverage": 0.70,
        "pii_redaction": 0.68,
        "data_freshness": 0.82,
        "citation_support": 0.45,
        "sso_integration": 0.70,
        "workflow_integration": 0.76,
        "eval_ready": 0.62,
        "feedback_loop": 0.50,
        "sla_p95_latency_ok": 1.00,
        "business_metric_defined": 0.60,
        "human_review_coverage": 0.70,
        "risk_level": "medium",
    },
    {
        "name": "generic_chat_portal",
        "scenario_type": "generic_chatbot",
        "users": 900,
        "data_sources": 2,
        "permission_coverage": 0.52,
        "tenant_isolation": 0.70,
        "rag_permission_filter": 0.40,
        "tool_permission_gate": 0.30,
        "audit_log_coverage": 0.35,
        "pii_redaction": 0.42,
        "data_freshness": 0.50,
        "citation_support": 0.20,
        "sso_integration": 0.00,
        "workflow_integration": 0.20,
        "eval_ready": 0.30,
        "feedback_loop": 0.25,
        "sla_p95_latency_ok": 0.00,
        "business_metric_defined": 0.00,
        "human_review_coverage": 0.20,
        "risk_level": "medium",
    },
]

results = [audit_app(app) for app in apps]
ranked = sorted(
    [(r["name"], r["enterprise_score"], r["enterprise_gate"]) for r in results],
    key=lambda item: item[1],
    reverse=True,
)
passed = [r["name"] for r in results if r["enterprise_gate"]]
needs_rework = {
    r["name"]: r["failed_gates"]
    for r in results
    if not r["enterprise_gate"]
}

print("ranked=", ranked)
print("enterprise_pass=", passed)
print("needs_rework=", needs_rework)
```

一组典型输出是：

```text
ranked= [('support_kb_rag', 0.927, True), ('contract_copilot', 0.866, False), ('data_analyst_nl2sql', 0.77, False), ('office_summarizer', 0.753, False), ('generic_chat_portal', 0.354, False)]
enterprise_pass= ['support_kb_rag']
needs_rework= {'contract_copilot': ['rag_permission_filter', 'tool_permission_gate', 'data_freshness', 'workflow_integration', 'feedback_loop', 'human_review_coverage'], 'data_analyst_nl2sql': ['permission_coverage', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'citation_support', 'eval_ready', 'feedback_loop', 'sla_p95_latency_ok', 'human_review_coverage'], 'office_summarizer': ['permission_coverage', 'tenant_isolation', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'citation_support', 'sso_integration', 'eval_ready', 'feedback_loop', 'business_metric_defined'], 'generic_chat_portal': ['permission_coverage', 'tenant_isolation', 'rag_permission_filter', 'tool_permission_gate', 'audit_log_coverage', 'pii_redaction', 'data_freshness', 'citation_support', 'sso_integration', 'workflow_integration', 'eval_ready', 'feedback_loop', 'sla_p95_latency_ok', 'business_metric_defined']}
```

这个 demo 的重点不是分数本身，而是审计口径：企业级应用必须把权限、租户隔离、审计、PII、引用、SLO、业务指标和人审放进同一张表。`contract_copilot` 分数不低，但因为法律场景高风险、人审和反馈闭环不足，不能直接全自动上线；`data_analyst_nl2sql` 工作流接入不错，但 SQL 工具权限、引用支持和延迟门禁不过线；`generic_chat_portal` 则说明“通用聊天入口”如果没有身份、权限、指标和工作流，只能算内部 demo。

## 5.20 本章小结

企业级 LLM 应用的难点在于真实环境：数据复杂、权限严格、系统多、流程长、用户角色多、合规要求高。客服、知识库、代码助手、数据分析、办公自动化和行业助手都是常见场景，但每个场景都需要结合业务流程和风险等级设计。

下一章会进入 RAG 产品落地，深入讨论企业知识库和检索增强生成系统如何从技术方案变成可用产品。
