# 第六章：RAG 产品落地

## 0. 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做公式和 demo 精修。联网资料主要核对四类口径：OpenAI retrieval / file search / evals 相关资料提醒我们，RAG 产品不是单一向量检索，而是查询改写、向量库、元数据过滤、重排、上下文构造、引用和评估组成的系统；RAGAS 等 RAG 评估论文提醒我们，RAG 要同时看 context precision、context recall、faithfulness / groundedness、answer relevance 和人工标注边界；OWASP LLM Top 10 中与向量和嵌入、提示注入、敏感信息泄露相关的风险提醒我们，企业 RAG 必须把不可信文档、权限过滤和敏感信息控制纳入产品门禁；Google SRE 的 SLI / SLO / error budget 口径提醒我们，RAG 的检索、重排和生成链路必须有延迟、错误率和成本目标。

本章不替代后续专门的 RAG 算法实现、向量数据库选型、安全合规审查或第十七册 Agentic RAG 深入章节。这里聚焦产品落地：怎么把企业知识库从“几个 PDF 的 demo”升级成可评估、可引用、可治理、可运营、可控成本的 RAG 产品。

RAG 是企业大模型落地中最常见的技术路线之一。它看起来简单：把文档切块、做向量检索、把检索结果喂给模型生成答案。但真正产品化时，难点往往不在“能不能检索”，而在知识是否最新、引用是否可信、权限是否正确、答案是否可评估、用户是否愿意使用，以及失败后如何持续改进。

本章系统讲 RAG 产品落地：企业文档治理、知识更新、文档解析、分块、检索、引用可信、权限控制、无法回答、反馈闭环、效果评估、上线运营和常见失败模式。

## 6.1 RAG 为什么适合企业落地

企业有大量内部知识：制度、流程、产品说明、技术文档、FAQ、合同模板、工单记录、会议纪要、历史案例。这些知识经常分散在不同系统中，员工查找成本高。

RAG 的价值是：

1. 让模型基于企业知识回答。
2. 降低幻觉风险。
3. 给出引用来源。
4. 支持知识更新。
5. 避免频繁微调模型。
6. 接入权限控制。

面试回答：

```text
RAG 适合企业落地，因为企业知识经常分散、更新频繁且需要权限控制。RAG 可以把检索到的内部文档作为上下文，让模型基于证据回答，并给出引用。相比直接微调，RAG 更容易更新知识和控制来源。但产品化时必须解决文档质量、权限过滤、引用准确、评估和反馈闭环。
```

## 6.2 RAG Demo 和 RAG 产品的区别

RAG demo：

1. 几个 PDF。
2. 简单切块。
3. 向量检索 top-k。
4. 生成答案。

RAG 产品：

1. 多来源文档接入。
2. 文档版本管理。
3. 权限过滤。
4. 增量更新。
5. 引用和可追溯。
6. 评估集。
7. 用户反馈。
8. 监控和运营。

demo 证明技术能跑通，产品要面对真实企业知识的混乱和变化。

## 6.3 企业文档治理

RAG 的上限很大程度由文档质量决定。

常见文档问题：

1. 过期文档未删除。
2. 多个版本互相冲突。
3. 标题不清楚。
4. 文档没有负责人。
5. 权限不明确。
6. 格式复杂。
7. 图片和表格难解析。
8. 内容重复。

文档治理要解决：

1. 来源可信。
2. 版本清晰。
3. 负责人明确。
4. 更新时间可见。
5. 权限可继承。
6. 元数据完整。

RAG 不是数据治理的替代品。脏知识库会让模型更自信地输出错误答案。

## 6.4 文档解析

文档解析决定进入 RAG 系统的原始文本质量。

要处理：

1. PDF。
2. Word。
3. PPT。
4. Excel。
5. HTML。
6. Markdown。
7. 图片和扫描件。
8. 表格和图表。

解析难点：

1. 页眉页脚噪声。
2. 表格结构丢失。
3. 多栏排版错乱。
4. OCR 错字。
5. 图片中的关键信息。
6. 章节层级丢失。

解析质量差，后面 embedding 和检索再强也难补救。

## 6.5 分块策略

分块影响召回和答案质量。

分块太小：上下文不完整。分块太大：召回不精确，token 成本高。

常见策略：

1. 按段落分块。
2. 按标题层级分块。
3. 按语义边界分块。
4. 滑动窗口重叠。
5. 表格单独处理。
6. 保留标题和路径元数据。

好的 chunk 不只是文本片段，还应该带有文档标题、章节、更新时间、权限、来源链接等元数据。

## 6.6 检索策略

企业 RAG 通常不只用向量检索。

常见组合：

1. 向量检索。
2. 关键词检索。
3. 混合检索。
4. 元数据过滤。
5. reranking。
6. 查询重写。
7. 多轮检索。

向量检索适合语义相似，关键词检索适合精确术语、产品型号、错误码和人名。企业系统通常需要混合检索。

## 6.7 权限控制

企业 RAG 必须做权限控制。

关键原则：

1. 用户只能检索有权限的文档。
2. 检索前或检索后必须过滤权限。
3. 引用不能暴露无权限来源。
4. 摘要不能泄露无权限内容。
5. 日志不能保存敏感内容。
6. 权限变更要及时生效。

一个严重错误是：模型不直接展示原文，但把无权限文档内容总结出来。这仍然是数据泄露。

产品上更稳的做法是把权限当成检索条件，而不是生成后的展示条件。换句话说，无权文档不应该进入候选集、rerank 列表、prompt 上下文、引用列表和原始日志。权限变更后，索引、缓存和向量库 metadata 也要同步失效，否则用户离职、转岗或项目权限收回后仍可能通过旧缓存看到答案。

## 6.8 知识更新

企业知识经常变化。

需要处理：

1. 新文档加入。
2. 老文档删除。
3. 文档内容修改。
4. 权限变化。
5. 文档过期。
6. 索引重建。
7. 增量 embedding。

产品上要显示答案依据的时间和版本。对于时效性强的问题，如果文档过期，系统应提示不确定，而不是强行回答。

## 6.9 引用可信

引用是 RAG 产品的信任基础。

好的引用应该：

1. 支持对应结论。
2. 指向具体段落。
3. 用户有权限查看。
4. 文档版本正确。
5. 来源可信。
6. 能打开原文。

坏引用会严重破坏信任。最常见问题是答案正确但引用错，或引用文档根本不支持结论。

引用可信要按 claim 检查，而不是按整段回答检查。一个回答可能有 5 个关键断言，其中 4 个有证据、1 个是模型补出来的；如果只看“回答整体还行”，这个 unsupported claim 就会漏掉。企业 RAG 最好把答案拆成 atomic claims，再检查每个 claim 是否被引用段落直接支持、是否使用最新版本、是否有权限展示。

## 6.10 无法回答

RAG 产品必须能说“不知道”。

无法回答的情况：

1. 检索不到相关文档。
2. 文档互相矛盾。
3. 用户无权限查看答案来源。
4. 问题超出知识库范围。
5. 证据不足。
6. 文档过期。

好的无法回答体验：

1. 说明原因。
2. 给出已检索范围。
3. 建议补充信息。
4. 引导用户联系负责人。
5. 允许反馈缺失文档。

强行回答会短期显得智能，长期损害信任。

## 6.11 用户反馈闭环

RAG 产品需要反馈闭环。

反馈可以包括：

1. 答案有用或无用。
2. 引用是否正确。
3. 是否解决问题。
4. 用户期望答案。
5. 缺失文档。
6. 错误文档。
7. 转人工原因。

反馈要进入改进流程：更新文档、调整分块、优化检索、补充评估集、改进 prompt 或权限规则。

## 6.12 评估指标

RAG 产品评估包括：

1. 检索召回率。
2. rerank 命中率。
3. 答案正确率。
4. 引用准确率。
5. 证据支持率。
6. 无法回答准确率。
7. 用户满意度。
8. 自助解决率。
9. 延迟。
10. 单次成本。

不同阶段关注不同指标。早期先看检索和答案正确，产品化后还要看使用率、满意度和业务收益。

## 6.12.1 关键公式与 RAG 产品指标速查

可以把一次 RAG 查询样本写成：

```math
q_i=(x_i,u_i,D_i,K_i,E_i,A_i,C_i,P_i,L_i,B_i)
```

其中 `x_i` 是用户问题，`u_i` 是用户和角色，`D_i` 是可访问文档集合，`K_i` 是检索候选，`E_i` 是最终进入上下文的证据，`A_i` 是模型答案，`C_i` 是引用集合，`P_i` 是权限判断，`L_i` 是延迟，`B_i` 是业务结果。

**1. 检索召回率**

检索层首先要问：应该找到的证据，有多少被召回到了候选集中？

```math
R_{\mathrm{ret}}=\frac{1}{N}\sum_{i=1}^{N}\frac{|K_i\cap G_i|}{|G_i|+\epsilon}
```

其中 `G_i` 是人工标注或专家确认的支持证据集合，`\epsilon` 避免分母为 0。召回率低时，后面的 rerank 和生成通常救不回来。

**2. 上下文精确率**

召回很多不等于最后上下文质量高。进入 prompt 的证据要尽量相关：

```math
P_{\mathrm{ctx}}=\frac{1}{N}\sum_{i=1}^{N}\frac{|E_i\cap G_i|}{|E_i|+\epsilon}
```

上下文精确率低，说明 prompt 里混入大量弱相关 chunk，会增加 token 成本、干扰生成，并提高引用错误概率。

**3. 证据支持率**

生成层要检查答案中的关键断言是否被证据支持：

```math
S_{\mathrm{ev}}=\frac{\sum_{i=1}^{N}n_i^{\mathrm{support}}}{\sum_{i=1}^{N}n_i^{\mathrm{claim}}+\epsilon}
```

其中 `n_i^{\mathrm{claim}}` 是第 `i` 个回答的关键断言数，`n_i^{\mathrm{support}}` 是被上下文证据直接支持的断言数。这个指标比“有引用”更严格。

**4. 引用准确率**

引用准确率检查引用是否真的支持对应 claim：

```math
A_{\mathrm{cite}}=\frac{\sum_{i=1}^{N}n_i^{\mathrm{cite\_ok}}}{\sum_{i=1}^{N}n_i^{\mathrm{cite}}+\epsilon}
```

其中 `n_i^{\mathrm{cite}}` 是引用数量，`n_i^{\mathrm{cite_ok}}` 是存在、相关、版本正确、权限正确且足以支持 claim 的引用数量。答案正确但引用错误，在企业产品里仍然是严重缺陷。

**5. 拒答准确率**

RAG 产品必须知道什么时候不能回答：

```math
A_{\mathrm{abs}}=\frac{1}{N_{\mathrm{abs}}}\sum_{i=1}^{N}\mathbf{1}[z_i=1]\mathbf{1}[\hat z_i=1]
```

其中 `z_i=1` 表示样本应该拒答或澄清，`\hat z_i=1` 表示系统实际拒答或澄清。拒答准确率低，说明系统在证据不足、权限不足、文档过期或冲突时仍然强行回答。

**6. 权限过滤通过率**

企业 RAG 要检查无权文档是否进入候选、上下文、引用或日志：

```math
R_{\mathrm{perm}}=1-\frac{N_{\mathrm{unauth}}}{N_{\mathrm{req}}+\epsilon}
```

其中 `N_{\mathrm{unauth}}` 是越权召回、越权引用、越权日志或越权摘要事件数。这个指标通常应接近 1，且严重事故不能被平均分掩盖。

**7. 过期证据率**

知识更新后，旧证据仍被引用会损害信任：

```math
R_{\mathrm{stale}}=\frac{N_{\mathrm{stale}}}{N_{\mathrm{evidence}}+\epsilon}
```

其中 `N_{\mathrm{stale}}` 是过期、被替换或版本不一致的证据数。时效性强的业务要把过期证据率纳入上线门禁。

**8. RAG 产品上线门禁**

一个简化门禁可以写成：

```math
G_{\mathrm{rag}}=\mathbf{1}[R_{\mathrm{ret}}\geq r_0]\mathbf{1}[P_{\mathrm{ctx}}\geq p_0]\mathbf{1}[S_{\mathrm{ev}}\geq s_0]\mathbf{1}[A_{\mathrm{cite}}\geq c_0]\mathbf{1}[R_{\mathrm{perm}}\geq h_0]\mathbf{1}[L_{95}\leq L_{\mathrm{slo}}]
```

其中 `r_0,p_0,s_0,c_0,h_0` 是场景阈值，`L_{95}` 是 P95 延迟。真实上线还要加入成本、反馈闭环、业务指标和人工抽检。RAG 产品不能只用一个问答准确率做上线判断。

## 6.13 评估集构建

RAG 评估需要问题集。

问题来源：

1. 历史搜索日志。
2. 客服工单。
3. 员工常见问题。
4. 业务专家整理。
5. 失败案例。
6. 新文档发布后的测试问题。

每个问题最好标注：

1. 标准答案。
2. 支持文档。
3. 关键段落。
4. 用户角色和权限。
5. 是否应拒答。

没有评估集，RAG 优化容易凭感觉。

## 6.14 上线运营

RAG 上线后要运营。

运营事项：

1. 监控热门问题。
2. 发现无答案问题。
3. 清理过期文档。
4. 维护文档负责人。
5. 分析低满意度回答。
6. 更新评估集。
7. 监控成本和延迟。
8. 处理权限异常。

RAG 产品不是一次建设，而是持续运营的知识系统。

## 6.15 常见失败模式

1. 文档质量差却怪模型。
2. 只用向量检索，忽略关键词和元数据。
3. 权限过滤不完整。
4. 引用不支持答案。
5. 文档更新后索引不同步。
6. 无法回答策略缺失。
7. 没有评估集。
8. 用户反馈没有进入改进流程。
9. 答案看起来流畅但没有证据。
10. 只做技术链路，不做知识运营。

RAG 落地的本质，是技术系统和知识管理系统一起建设。

## 6.16 面试题：RAG 产品落地最难的是什么

回答要点：

```text
RAG 落地最难的不只是检索算法，而是企业文档治理、权限控制、知识更新、引用可信和效果评估。文档过期、版本冲突、权限不清、引用不支持答案，都会让用户失去信任。一个可靠 RAG 产品需要文档解析、分块、混合检索、rerank、权限过滤、引用校验、评估集和用户反馈闭环。
```

## 6.17 面试题：如何评估企业 RAG

回答要点：

```text
我会分层评估。检索层看召回率和 rerank 命中率；生成层看答案正确率、引用准确率和证据支持率；产品层看自助解决率、用户满意度、无法回答准确率、延迟和成本。评估集要来自真实用户问题，并标注标准答案、支持文档、关键段落和权限信息。
```

## 6.18 最小可运行 RAG 产品审计 demo

下面这个 demo 用 0 依赖 Python 模拟 RAG 产品上线审计。它把检索召回、上下文精确、证据支持、引用准确、拒答、权限过滤、数据新鲜度、P95 延迟、单位成本、评估集、反馈闭环和业务指标放进同一张表。

```python
def ratio(num, den):
    return num / den if den else 0.0


def audit_product(product):
    retrieval_recall = ratio(product["retrieved_relevant"], product["relevant_total"])
    context_precision = ratio(product["context_relevant"], product["context_total"])
    evidence_support = ratio(product["supported_claims"], product["total_claims"])
    citation_accuracy = ratio(product["citation_supported"], product["citation_total"])
    abstention_accuracy = ratio(product["correct_abstentions"], product["expected_abstentions"])
    freshness = 1.0 - product["stale_evidence_rate"]
    permission_ok = (
        product["permission_filter_rate"] >= 0.98
        and product["unauthorized_hits"] == 0
    )
    gates = {
        "retrieval_recall": retrieval_recall >= 0.80,
        "context_precision": context_precision >= 0.65,
        "evidence_support": evidence_support >= 0.85,
        "citation_accuracy": citation_accuracy >= 0.85,
        "abstention_accuracy": abstention_accuracy >= 0.80,
        "permission_filter": permission_ok,
        "freshness": product["stale_evidence_rate"] <= 0.10,
        "p95_latency": product["p95_latency_ms"] <= product["latency_slo_ms"],
        "unit_cost": product["cost_per_answer"] <= product["cost_slo"],
        "eval_ready": product["eval_ready"] >= 0.80,
        "feedback_loop": product["feedback_loop"] >= 0.75,
        "business_metric": product["business_metric_defined"] >= 1.0,
    }
    rag_score = (
        0.18 * retrieval_recall
        + 0.14 * context_precision
        + 0.18 * evidence_support
        + 0.14 * citation_accuracy
        + 0.08 * abstention_accuracy
        + 0.10 * product["permission_filter_rate"]
        + 0.06 * freshness
        + 0.06 * (1.0 if gates["p95_latency"] else 0.0)
        + 0.03 * product["eval_ready"]
        + 0.03 * product["feedback_loop"]
    )
    return {
        "name": product["name"],
        "rag_score": round(rag_score, 3),
        "rag_gate": all(gates.values()),
        "metrics": {
            "retrieval_recall": round(retrieval_recall, 3),
            "context_precision": round(context_precision, 3),
            "evidence_support": round(evidence_support, 3),
            "citation_accuracy": round(citation_accuracy, 3),
            "abstention_accuracy": round(abstention_accuracy, 3),
        },
        "failed_gates": [name for name, ok in gates.items() if not ok],
    }


products = [
    {
        "name": "support_policy_rag",
        "retrieved_relevant": 42,
        "relevant_total": 48,
        "context_relevant": 31,
        "context_total": 42,
        "supported_claims": 66,
        "total_claims": 72,
        "citation_supported": 61,
        "citation_total": 68,
        "correct_abstentions": 9,
        "expected_abstentions": 10,
        "permission_filter_rate": 0.99,
        "unauthorized_hits": 0,
        "stale_evidence_rate": 0.06,
        "p95_latency_ms": 1800,
        "latency_slo_ms": 2200,
        "cost_per_answer": 0.043,
        "cost_slo": 0.08,
        "eval_ready": 0.86,
        "feedback_loop": 0.80,
        "business_metric_defined": 1.0,
    },
    {
        "name": "legal_contract_rag",
        "retrieved_relevant": 34,
        "relevant_total": 45,
        "context_relevant": 28,
        "context_total": 44,
        "supported_claims": 58,
        "total_claims": 70,
        "citation_supported": 49,
        "citation_total": 64,
        "correct_abstentions": 8,
        "expected_abstentions": 12,
        "permission_filter_rate": 0.98,
        "unauthorized_hits": 0,
        "stale_evidence_rate": 0.16,
        "p95_latency_ms": 2600,
        "latency_slo_ms": 2500,
        "cost_per_answer": 0.091,
        "cost_slo": 0.10,
        "eval_ready": 0.82,
        "feedback_loop": 0.62,
        "business_metric_defined": 1.0,
    },
    {
        "name": "codebase_rag",
        "retrieved_relevant": 28,
        "relevant_total": 36,
        "context_relevant": 18,
        "context_total": 35,
        "supported_claims": 42,
        "total_claims": 54,
        "citation_supported": 37,
        "citation_total": 50,
        "correct_abstentions": 6,
        "expected_abstentions": 8,
        "permission_filter_rate": 0.93,
        "unauthorized_hits": 1,
        "stale_evidence_rate": 0.08,
        "p95_latency_ms": 1900,
        "latency_slo_ms": 1800,
        "cost_per_answer": 0.052,
        "cost_slo": 0.08,
        "eval_ready": 0.70,
        "feedback_loop": 0.68,
        "business_metric_defined": 1.0,
    },
    {
        "name": "generic_doc_chat",
        "retrieved_relevant": 18,
        "relevant_total": 44,
        "context_relevant": 14,
        "context_total": 48,
        "supported_claims": 30,
        "total_claims": 68,
        "citation_supported": 19,
        "citation_total": 55,
        "correct_abstentions": 2,
        "expected_abstentions": 11,
        "permission_filter_rate": 0.72,
        "unauthorized_hits": 3,
        "stale_evidence_rate": 0.31,
        "p95_latency_ms": 3300,
        "latency_slo_ms": 2200,
        "cost_per_answer": 0.115,
        "cost_slo": 0.08,
        "eval_ready": 0.35,
        "feedback_loop": 0.20,
        "business_metric_defined": 0.0,
    },
]

results = [audit_product(product) for product in products]
ranked = sorted(
    [(r["name"], r["rag_score"], r["rag_gate"]) for r in results],
    key=lambda item: item[1],
    reverse=True,
)
rag_pass = [r["name"] for r in results if r["rag_gate"]]
needs_rework = {
    r["name"]: r["failed_gates"]
    for r in results
    if not r["rag_gate"]
}

print("ranked=", ranked)
print("rag_pass=", rag_pass)
print("sample_metrics=", results[0]["metrics"])
print("needs_rework=", needs_rework)
```

一组典型输出是：

```text
ranked= [('support_policy_rag', 0.889, True), ('legal_contract_rag', 0.726, False), ('codebase_rag', 0.705, False), ('generic_doc_chat', 0.387, False)]
rag_pass= ['support_policy_rag']
sample_metrics= {'retrieval_recall': 0.875, 'context_precision': 0.738, 'evidence_support': 0.917, 'citation_accuracy': 0.897, 'abstention_accuracy': 0.9}
needs_rework= {'legal_contract_rag': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_accuracy', 'freshness', 'p95_latency', 'feedback_loop'], 'codebase_rag': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_accuracy', 'permission_filter', 'p95_latency', 'eval_ready', 'feedback_loop'], 'generic_doc_chat': ['retrieval_recall', 'context_precision', 'evidence_support', 'citation_accuracy', 'abstention_accuracy', 'permission_filter', 'freshness', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop', 'business_metric']}
```

这个 demo 的重点是：RAG 产品不能只看“回答像不像对”。`legal_contract_rag` 有一定质量，但合同场景中证据过期、引用不足、拒答不足和反馈闭环缺失会影响责任边界；`codebase_rag` 权限和评估不稳，可能把无权仓库或旧代码片段带进回答；`generic_doc_chat` 则说明“文档聊天”如果没有评估、权限、引用、拒答和业务指标，很难算企业级 RAG 产品。

## 6.19 本章小结

RAG 产品落地的关键，是让模型基于正确、最新、有权限、可追溯的知识回答问题。技术链路包括文档解析、分块、embedding、检索、rerank、生成和引用；产品链路还包括文档治理、权限、更新、评估、反馈和运营。

下一章会进入 Agent 产品落地，讨论当产品不只是回答问题，而是要执行任务、调用工具、操作系统时，如何设计体验、权限、成本和安全边界。
