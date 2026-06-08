# 第九章：RAG 与 Agent 部署

## 本章目标

理解如何把 RAG 和 Agent 系统稳定部署到生产环境。

## 核心议题

1. 文档解析和切分。
2. embedding 服务。
3. vector database。
4. reranker 服务。
5. prompt assembly。
6. tool calling。
7. agent 状态管理。
8. 权限、安全和审计。

## 面试重点

RAG 和 Agent 部署不是只调一个模型 API，而是多个服务组件的可靠组合。

## 本章资料边界

本章第二轮精修时对照了 RAG 原论文、OpenAI tools / structured outputs 文档、LangChain Agent 部署与工具调用资料、LlamaIndex RAG 评估资料和前序 RAG / Agent 章节。这里聚焦生产部署中最常见的问题：

1. RAG 离线索引链路和在线查询链路如何拆分。
2. retrieval、rerank、prompt assembly、citation check 和 latency budget 如何公式化。
3. Agent tool calling 的 schema、权限、超时、重试和审计门禁。
4. RAG / Agent 出错时如何分层归因，而不是只替换大模型。

本章不展开成完整向量数据库、Agent 框架或工具协议实现手册。框架 API 会变化，但生产边界稳定：数据权限、证据质量、工具执行安全、trace 可观测性和失败回退必须由系统保证。

## 为什么 RAG 和 Agent 部署更复杂

普通 LLM 服务主要是：

```text
prompt -> model -> answer
```

RAG 和 Agent 系统通常是：

```text
用户请求
  -> 查询理解
  -> 检索 / 工具选择
  -> 外部系统调用
  -> 上下文组装
  -> LLM 生成
  -> 引用 / 校验 / 审计
  -> 返回答案
```

它涉及模型、检索、数据库、工具、权限、状态、日志和安全策略。

所以生产难点不是单个模型，而是多组件协作的可靠性。

面试表达：RAG 和 Agent 不是“模型加插件”，而是一个包含数据、检索、工具、状态、安全和观测的系统工程。

## 1. RAG 的生产链路

一个生产级 RAG 通常包含两条链路。

### 1.1 离线构建链路

```text
文档采集
  -> 解析
  -> 清洗
  -> 切分
  -> embedding
  -> 建索引
  -> 版本管理
```

### 1.2 在线查询链路

```text
用户问题
  -> query rewrite
  -> retrieval
  -> rerank
  -> prompt assembly
  -> LLM generation
  -> citation / grounding check
  -> response
```

面试中要把这两条链路分开讲。

离线链路决定知识库质量，在线链路决定实时回答效果和延迟。

在线 RAG 可以抽象为：

```math
Q'=U(q)
```

```math
C_K=\mathrm{TopK}(\mathrm{Retrieve}(Q',D),K)
```

```math
E_k=\mathrm{TopK}(\mathrm{Rerank}(q,C_K),k)
```

```math
y=M(\mathrm{Assemble}(q,E_k,r))
```

其中 `q` 是用户问题，`U` 是 query rewrite，`D` 是知识库，`C_K` 是召回候选，`E_k` 是最终证据集合，`r` 是输出格式和引用约束，`M` 是生成模型。

## 2. 文档解析和切分

文档解析要把 PDF、Word、Markdown、HTML、表格、图片 OCR 等转成可检索文本。

### 2.1 解析难点

1. PDF 排版混乱。
2. 表格结构丢失。
3. OCR 识别错误。
4. 标题层级丢失。
5. 公式和代码块格式损坏。

### 2.2 Chunking

切分决定检索粒度。

chunk 太小会丢上下文，chunk 太大检索不精准。

常见策略：

1. 固定长度切分。
2. 按段落切分。
3. 按标题层级切分。
4. 按语义边界切分。
5. 带 overlap 的滑动窗口。

面试表达：RAG 的效果经常不是模型问题，而是文档解析和 chunking 出了问题。

## 3. Embedding 服务

Embedding 服务负责把 query 和 chunk 编码成向量。

生产中要关注：

1. embedding 模型版本。
2. 向量维度。
3. batch 推理吞吐。
4. 多语言能力。
5. 领域匹配。
6. 向量归一化。

### 3.1 版本问题

如果 embedding 模型升级，旧文档向量可能需要重建。

否则 query embedding 和 doc embedding 来自不同分布，检索质量会下降。

面试表达：embedding 模型是 RAG 的检索协议，升级 embedding 往往意味着索引也要重新构建或做兼容。

## 4. Vector database

向量数据库负责存储和检索向量。

核心能力：

1. 向量写入。
2. ANN 检索。
3. metadata filter。
4. 增量更新。
5. 删除和权限控制。
6. 分片和扩容。
7. 多租户隔离。

常见问题：

1. 索引构建慢。
2. 更新延迟。
3. 召回率和速度 trade-off。
4. 权限过滤和向量召回结合复杂。
5. embedding 版本不一致。

面试表达：向量库解决的是大规模向量检索和管理，不直接保证 RAG 答案正确。

## 5. Hybrid retrieval 和 reranker

只用向量检索不够。

企业场景经常需要精确匹配：

1. 产品型号。
2. 错误码。
3. 合同条款编号。
4. 函数名。
5. 金额和日期。

因此常用 hybrid retrieval：

```text
dense retrieval + sparse retrieval + reranker
```

Reranker 用更精细的模型重新排序候选文档。

面试表达：retriever 决定候选池，reranker 决定最终证据顺序。RAG 质量差时要区分是召回问题还是排序问题。

混合检索可写成：

```math
s(d,q)=\alpha s_{\mathrm{dense}}(d,q)+(1-\alpha)s_{\mathrm{sparse}}(d,q)+\beta s_{\mathrm{meta}}(d,q)
```

其中 `s_dense` 是 embedding 相似度，`s_sparse` 是 BM25 / keyword 类稀疏分数，`s_meta` 是权限、版本、时间、领域等 metadata 加权项。最终 `alpha`、`beta` 不能拍脑袋，要用检索集和下游答案质量调参。

## 6. Prompt assembly

Prompt assembly 是把用户问题、检索证据、系统指令和输出要求拼成最终 prompt。

需要控制：

1. 证据数量。
2. 证据顺序。
3. 引用格式。
4. token budget。
5. 冲突证据。
6. 系统指令和安全规则。

### 常见问题

1. 塞入太多无关证据。
2. 关键证据被放在中间导致 lost in the middle。
3. 证据相互矛盾但没有说明。
4. 模型没有按证据回答。

面试表达：RAG 的 prompt 组装不是简单 top-k 拼接，而是证据选择、排序、压缩和引用约束。

上下文组装需要满足 token budget：

```math
T_{\mathrm{sys}}+T_q+\sum_{e_i\in E}T(e_i)+T_{\mathrm{out}}\le T_{\max}
```

如果证据超过预算，应该做 evidence selection、压缩或分步回答，而不是无脑截断。

## 7. RAG 评估和监控

RAG 要分层评估。

### 7.1 检索层

1. Recall@k。
2. MRR。
3. nDCG。
4. 命中文档比例。

### 7.2 生成层

1. 答案正确性。
2. 忠实于证据。
3. 引用是否匹配。
4. 幻觉率。

### 7.3 系统层

1. 检索延迟。
2. rerank 延迟。
3. LLM 延迟。
4. 端到端 P95/P99。
5. 失败率。

面试表达：RAG 评估要拆成检索质量、生成质量和系统性能三层。

检索层常用 Recall@K：

```math
\mathrm{Recall@K}=\frac{|\mathrm{Gold}(q)\cap \mathrm{TopK}(q)|}{|\mathrm{Gold}(q)|}
```

引用校验可以抽象为 claim-level 支持率：

```math
S_{\mathrm{cite}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{Evidence}(c_i)\Rightarrow c_i]
```

端到端延迟可以拆成：

```math
T_{\mathrm{e2e}}=T_{\mathrm{rewrite}}+T_{\mathrm{retrieve}}+T_{\mathrm{rerank}}+T_{\mathrm{llm}}+T_{\mathrm{tool}}+T_{\mathrm{check}}
```

RAG 上线门禁至少要同时看：

```math
G_{\mathrm{rag}}=G_{\mathrm{retrieval}}\land G_{\mathrm{grounding}}\land G_{\mathrm{permission}}\land G_{\mathrm{latency}}
```

## 8. Agent 和普通 Chat 的区别

普通 chat model 主要是生成回答。

Agent 还要：

1. 规划。
2. 调用工具。
3. 观察结果。
4. 更新状态。
5. 多步执行。
6. 判断停止。

典型循环：

```text
observe -> think -> act -> observe -> ... -> answer
```

可以把 Agent 执行抽象为状态转移：

```math
s_{t+1}=F(s_t,o_t,a_t)
```

其中 `s_t` 是当前状态，`o_t` 是观察，`a_t` 是模型选择的动作或工具调用。部署时必须限制最大步数、最大成本和最大 wall time：

```math
t\le t_{\max},\qquad C_{\mathrm{agent}}\le C_{\max}
```

面试表达：Agent 的核心不是一次回答，而是带工具和状态的多步决策执行循环。

## 9. Tool calling 部署

Tool calling 要解决的不只是模型输出函数名。

生产中还要处理：

1. tool registry。
2. schema validation。
3. 参数校验。
4. 权限检查。
5. 超时和重试。
6. 工具结果截断和摘要。
7. 审计日志。

如果工具参数错了，不能直接执行危险操作。

面试表达：工具调用部署的关键是 schema、权限、超时、审计和错误恢复。

工具执行门禁可写成：

```math
G_{\mathrm{tool}}=G_{\mathrm{schema}}\land G_{\mathrm{permission}}\land G_{\mathrm{risk}}\land G_{\mathrm{timeout}}
```

只有 `G_tool` 通过，系统才应该执行真实工具。高风险写操作还应增加用户确认或人工审批。

## 10. Agent 状态管理

Agent 需要管理状态。

状态包括：

1. 对话历史。
2. 工具调用记录。
3. 中间观察。
4. 文件或任务状态。
5. 用户权限。
6. 当前计划。

这些状态不能无限塞进 context window。

常见做法：

1. 外部状态存储。
2. 关键历史摘要。
3. trace 日志。
4. memory 压缩。
5. 按需检索历史。

## 11. 权限、安全和审计

Agent 会调用工具，风险比普通聊天更高。

需要控制：

1. 用户能调用哪些工具。
2. 工具能访问哪些数据。
3. 是否需要用户确认。
4. 是否允许写操作。
5. 是否允许执行命令。
6. 日志如何审计。

对于高风险操作，应该有人类确认或策略拦截。

面试表达：Agent 安全的关键是权限边界和执行控制，而不是只靠 prompt 约束。

## 12. Agent 可观测性

Agent 系统必须记录 trace。

Trace 应包括：

1. 用户请求。
2. 模型中间输出。
3. 工具调用参数。
4. 工具返回。
5. 状态变化。
6. 最终回答。
7. 错误和重试。

没有 trace，很难 debug Agent 为什么做错。

## 13. RAG 与 Agent 的部署风险

常见风险：

1. 检索错误。
2. 引用幻觉。
3. 权限绕过。
4. Prompt injection。
5. 工具误调用。
6. 循环调用停不下来。
7. 成本不可控。
8. 外部服务依赖失败。

需要对应的防护：

1. 权限过滤。
2. 引用校验。
3. tool sandbox。
4. max steps。
5. timeout。
6. budget limit。
7. fallback。

## 14. 最小 Python demo：RAG 与工具调用部署审计

下面的 0 依赖 demo 用 toy 文档和 toy 工具模拟生产部署中的几个关键门禁：租户权限过滤、检索与重排、context budget、引用校验、工具 schema / 权限检查和延迟拆分。

```python
import re


DOCS = [
    {
        "id": "return_policy",
        "tenant": "acme",
        "text": "ACME laptop returns are allowed within 30 days. Manager approval is required for devices over $1000.",
    },
    {
        "id": "warranty",
        "tenant": "acme",
        "text": "ACME laptops include a 2 year warranty for manufacturing defects.",
    },
    {
        "id": "salary_private",
        "tenant": "hr",
        "text": "Salary adjustment records are confidential and visible only to HR admins.",
    },
    {
        "id": "setup_guide",
        "tenant": "acme",
        "text": "New laptops should be encrypted before first use.",
    },
]

QUERY = "Can an ACME employee return a laptop after 20 days, and is manager approval needed?"
USER = {"tenant": "acme", "roles": {"employee"}, "can_write": False}


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_score(query, doc):
    q, d = tokens(query), tokens(doc["text"])
    return len(q & d) / max(len(q), 1)


def retrieve(query, docs, user, top_k=3):
    visible = [d for d in docs if d["tenant"] == user["tenant"]]
    scored = sorted(((lexical_score(query, d), d) for d in visible), reverse=True, key=lambda x: x[0])
    return [{"id": d["id"], "score": round(score, 3), "text": d["text"]} for score, d in scored[:top_k]]


def rerank(query, rows):
    q = tokens(query)
    boosted = []
    for row in rows:
        score = row["score"]
        if "return" in q and "returns" in row["text"].lower():
            score += 0.35
        if "approval" in q and "approval" in row["text"].lower():
            score += 0.20
        boosted.append({**row, "rerank_score": round(score, 3)})
    return sorted(boosted, key=lambda x: x["rerank_score"], reverse=True)


def assemble_context(rows, token_budget=28):
    selected, used = [], 0
    for row in rows:
        n_tokens = len(tokens(row["text"]))
        if used + n_tokens <= token_budget:
            selected.append(row)
            used += n_tokens
    return selected, used


def citation_ok(answer, context):
    cited = re.findall(r"\[(.*?)\]", answer)
    context_ids = {row["id"] for row in context}
    return bool(cited) and all(cid in context_ids for cid in cited)


retrieved = retrieve(QUERY, DOCS, USER)
reranked = rerank(QUERY, retrieved)
context, used_tokens = assemble_context(reranked)
answer = (
    "Yes. A 20 day laptop return is within the 30 day window, "
    "and manager approval is required for devices over $1000. [return_policy]"
)

TOOLS = {
    "order_lookup": {"required": {"order_id"}, "roles": {"employee", "support"}, "write": False},
    "delete_user": {"required": {"user_id"}, "roles": {"admin"}, "write": True},
}

CALLS = [
    {"name": "order_lookup", "args": {"order_id": "A-100"}},
    {"name": "delete_user", "args": {"user_id": "u-7"}},
]


def check_tool(call, user):
    spec = TOOLS.get(call["name"])
    if spec is None:
        return False, ["unknown_tool"]
    issues = []
    missing = spec["required"] - set(call["args"])
    if missing:
        issues.append("missing_args:" + ",".join(sorted(missing)))
    if not (user["roles"] & spec["roles"]):
        issues.append("permission_denied")
    if spec["write"] and not user["can_write"]:
        issues.append("write_not_allowed")
    return len(issues) == 0, issues


tool_report = {call["name"]: check_tool(call, USER) for call in CALLS}
latency_ms = {"rewrite": 20, "retrieve": 35, "rerank": 45, "llm": 420, "tool": 80}

print("retrieved_ids=", [row["id"] for row in retrieved])
print("reranked_ids=", [row["id"] for row in reranked])
print("context_ids=", [row["id"] for row in context], "used_tokens=", used_tokens)
print("citation_ok=", citation_ok(answer, context))
print("tool_report=", tool_report)
print("latency_total_ms=", sum(latency_ms.values()), "latency_breakdown=", latency_ms)
print("gate_pass=", citation_ok(answer, context) and tool_report["order_lookup"][0] and not tool_report["delete_user"][0])
```

一组可能输出：

```text
retrieved_ids= ['return_policy', 'warranty', 'setup_guide']
reranked_ids= ['return_policy', 'warranty', 'setup_guide']
context_ids= ['return_policy', 'warranty'] used_tokens= 26
citation_ok= True
tool_report= {'order_lookup': (True, []), 'delete_user': (False, ['permission_denied', 'write_not_allowed'])}
latency_total_ms= 600 latency_breakdown= {'rewrite': 20, 'retrieve': 35, 'rerank': 45, 'llm': 420, 'tool': 80}
gate_pass= True
```

这段 demo 的关键点是：

1. `salary_private` 因租户权限不匹配，不会进入可检索集合。
2. context assembly 受 token budget 限制，只保留可放入上下文的证据。
3. 高风险写工具 `delete_user` 被权限和写操作门禁拦截，不能只靠模型自行判断。

## 15. 面试官会怎么问

### 问法 1：生产级 RAG 包含哪些组件？

可以这样答：

```text
生产级 RAG 包含离线文档管道和在线查询管道。离线包括文档解析、清洗、切分、embedding、索引和版本管理；在线包括 query rewrite、召回、rerank、prompt assembly、LLM 生成、引用校验和监控。还要处理权限、安全、增量更新和评估。
```

### 问法 2：RAG 效果差怎么排查？

可以这样答：

```text
我会分层排查。先看检索是否召回正确文档，再看 reranker 是否把正确证据排前面，再看 prompt assembly 是否塞入太多噪声或丢失关键信息，最后看 LLM 是否忠实于证据回答。不能一上来就换大模型。
```

### 问法 3：Agent 部署比普通 Chat 难在哪里？

可以这样答：

```text
Agent 不只是生成回答，还要规划、调用工具、观察结果、维护状态和多步执行。因此要处理 tool registry、schema 校验、权限、安全、状态存储、trace 日志、超时重试和成本控制。风险比普通 chat 更高。
```

### 问法 4：如何防止 Agent 工具误调用？

可以这样答：

```text
需要工具 schema 校验、参数校验、权限检查、高风险操作人工确认、沙箱执行、超时和重试控制，并记录完整 trace。对于写操作、支付、删除、执行命令这类高风险工具，不能只靠模型自己判断。
```

### 问法 5：RAG 和长上下文怎么选？

可以这样答：

```text
长上下文适合一次性阅读较完整材料，但成本和延迟高，也可能 lost in the middle。RAG 适合从大规模知识库中筛选相关证据，成本更可控且知识可更新。真实系统常结合两者：先检索筛选，再把高质量证据放入上下文。
```

## 16. 本章小结

本章核心结论：

1. RAG 和 Agent 部署是多组件系统，不是单模型 API。
2. RAG 要区分离线文档构建链路和在线查询链路。
3. 文档解析、chunking、embedding、向量库和 reranker 都会影响效果。
4. Prompt assembly 决定证据如何进入模型上下文。
5. RAG 评估要拆成检索、生成和系统性能三层。
6. Agent 是带工具和状态的多步执行系统。
7. Tool calling 要有 schema、权限、超时、审计和错误恢复。
8. Agent 状态不能无限塞进上下文，需要外部存储和 trace。
9. RAG 与 Agent 的生产风险包括权限、安全、注入、成本和外部依赖失败。
