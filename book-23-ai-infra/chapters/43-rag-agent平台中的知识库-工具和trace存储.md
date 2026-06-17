# 第 43 章 RAG/Agent 平台中的知识库、工具和 trace 存储

上一章讲了 Feature Store、Embedding Store 与向量索引基础设施。本章作为第五部分收尾，讲 RAG/Agent 平台中的知识库、工具和 trace 存储。

RAG 和 Agent 是大模型应用平台中的常见形态。它们不只依赖模型本身，还依赖知识库、工具、权限、记忆、调用链路和可回放 trace。如果这些资产没有统一管理，应用很快会变成难以调试和治理的黑盒。

先记住一句话：

> RAG/Agent 平台的核心数据资产，不只是 prompt 和向量库，还包括知识库、工具定义、权限、检索记录、工具调用记录和完整执行 trace。

## 43.0 本讲资料边界与第二轮精修口径

本章按通用 RAG / Agent 平台基础设施抽象来写，不绑定某个向量数据库、Agent 框架、工具协议、可观测性产品、trace SaaS 或内部平台实现。资料校准时，主要参考 OpenTelemetry 对 trace / span / event / attribute 的通用可观测性模型、W3C Trace Context 对跨服务 trace 上下文传播的标准化口径、LangSmith / LlamaIndex 对 RAG / Agent 应用中的 run、document、node、metadata、retrieval 和 eval 记录方式、OpenAI Agents SDK 对 agent workflow tracing 的工程边界，以及 MCP 对 tool 定义、schema 和 server 能力声明的协议化表达。

第二轮精修只做三件事：

1. 把知识库、文档、chunk、citation、tool、memory、run state 和 trace 统一成可审计对象。
2. 补齐权限、版本、回放、隐私、采样、成本和删除传播的公式化门禁。
3. 增加一个 0 依赖 Python demo，用 toy cases 证明 RAG / Agent 平台存储不能只靠向量库或日志表。

## 43.1 为什么需要 RAG/Agent 平台存储

一个简单 RAG demo 可能只需要：

```text
PDF -> chunk -> embedding -> vector search -> prompt -> LLM
```

一个简单 Agent demo 可能只需要：

```text
LLM -> tool call -> tool result -> LLM
```

但生产平台要处理：

1. 多知识库。
2. 多租户权限。
3. 文档版本。
4. 工具版本。
5. 工具权限。
6. 检索 trace。
7. 工具调用 trace。
8. 多轮状态。
9. 失败回放。
10. 安全审计。

这些都需要专门的数据和存储设计。

可以把一次 RAG / Agent 平台运行抽象成：

$$
P_i=(k_i,d_i,c_i,e_i,v_i,q_i,p_i,a_i,u_i,m_i,r_i,s_i,z_i)
$$

其中 `k_i` 是知识库版本，`d_i` 是文档版本集合，`c_i` 是 chunk 集合，`e_i` 是 embedding 版本，`v_i` 是向量索引版本，`q_i` 是检索请求，`p_i` 是 prompt assembly，`a_i` 是 agent definition，`u_i` 是 tool 调用集合，`m_i` 是 memory / run state，`r_i` 是执行 trace，`s_i` 是安全与权限决策，`z_i` 是成本和审计记录。

平台存储覆盖率可以写成：

$$
C_{\mathrm{storage}}=\frac{1}{N}\sum_{i=1}^{N} I(k_i,d_i,c_i,e_i,v_i,q_i,p_i,a_i,u_i,m_i,r_i,s_i,z_i\ \mathrm{complete})
$$

直觉上，只有 prompt 或向量库记录完整不够；RAG / Agent 的一次回答必须能追溯到知识、工具、权限、trace、成本和审计证据。

## 43.2 RAG 平台中的核心资产

RAG 平台常见资产包括：

1. Knowledge Base。
2. Document。
3. Chunk。
4. Embedding。
5. Vector Index。
6. Retrieval Config。
7. Rerank Config。
8. Prompt Template。
9. Citation。
10. Retrieval Trace。

其中知识库是逻辑集合，文档是原始内容，chunk 是检索单元，embedding 和 index 负责召回，trace 负责解释和排障。

RAG 资产完整率可以写成：

$$
C_{\mathrm{rag}}=\frac{|\{a\in A_{\mathrm{rag}}:a\ \mathrm{versioned}\land a\ \mathrm{traceable}\}|}{|A_{\mathrm{rag}}|}
$$

`A_{\mathrm{rag}}` 是知识库、文档、chunk、embedding、index、retrieval config、rerank config、prompt template、citation 和 retrieval trace 的集合。这个指标强调每类资产都要有版本和血缘，而不是只要求向量可查。

## 43.3 Knowledge Base 设计

Knowledge Base 是知识库对象。

它通常记录：

1. 知识库名称。
2. 所属租户。
3. 描述。
4. 数据来源。
5. 权限策略。
6. embedding 模型版本。
7. chunking 规则版本。
8. 索引版本。
9. 同步状态。
10. 创建人和维护人。

知识库不是一个向量索引名，而是一组文档、版本、权限、索引和策略的集合。

一个 Knowledge Base 契约可以写成：

$$
K_j=(\mathrm{kb\_id},\mathrm{tenant},\mathrm{owner},S_j,\pi_j,e_j,g_j,v_j,\rho_j,\tau_j)
$$

其中 `S_j` 是来源连接器集合，`\pi_j` 是权限策略，`e_j` 是 embedding 版本，`g_j` 是 chunking 规则版本，`v_j` 是索引版本，`\rho_j` 是同步状态，`\tau_j` 是保留和删除策略。

知识库契约完整率：

$$
C_{\mathrm{kb}}=\frac{\sum_j I(K_j\ \mathrm{has\ required\ fields})}{|\{K_j\}|}
$$

## 43.4 Document 设计

Document 是知识库中的原始文档。

它可以来自：

1. PDF。
2. Markdown。
3. HTML。
4. Word 文档。
5. Wiki 页面。
6. 工单。
7. 数据库记录。
8. 代码文件。
9. API 文档。

Document 元数据包括：

1. document ID。
2. knowledge base ID。
3. source URI。
4. title。
5. author。
6. created time。
7. updated time。
8. content hash。
9. permission labels。
10. document version。
11. parsing status。

文档必须版本化，否则检索结果无法复现。

文档版本对象可以写成：

$$
D_j=(\mathrm{doc\_id},\mathrm{kb\_id},u_j,t_j,h_j,\ell_j,v_j,\sigma_j)
$$

其中 `u_j` 是 source URI，`t_j` 是更新时间，`h_j` 是内容 hash，`\ell_j` 是权限标签，`v_j` 是 document version，`\sigma_j` 是解析状态。文档 hash 和版本一起决定“当时被检索的到底是哪份内容”。

## 43.5 Chunk 设计

Chunk 是向量检索的基本单元。

Chunk 元数据包括：

1. chunk ID。
2. document ID。
3. chunk index。
4. text。
5. token count。
6. offset。
7. heading path。
8. chunking rule version。
9. embedding ID。
10. permission labels。

Chunk 要能回溯到原文位置。

这样模型回答引用内容时，才能给出 citation，也方便用户验证来源。

Chunk 记录可以写成：

$$
C_j=(\mathrm{chunk\_id},\mathrm{doc\_id},v_j,o_j,n_j,g_j,e_j,h_j,\ell_j)
$$

其中 `o_j` 是原文 offset，`n_j` 是 token 数，`g_j` 是 chunking rule version，`e_j` 是 embedding ID，`h_j` 是 chunk checksum，`\ell_j` 是权限标签。

chunk 到原文的可引用覆盖率：

$$
C_{\mathrm{cite\_lineage}}=\frac{\sum_j I(\mathrm{doc\_id}_j,v_j,o_j,h_j\ \mathrm{present})}{|\{C_j\}|}
$$

如果 chunk 没有 offset、document version 或 checksum，citation 就只能展示一个看似正确的链接，不能证明生成时引用的原文片段。

## 43.6 文档解析和清洗存储

RAG 平台需要保存解析状态。

文档解析可能包括：

1. OCR。
2. PDF 结构解析。
3. 表格抽取。
4. 标题层级识别。
5. 代码块识别。
6. 图片说明生成。
7. 噪声去除。

解析结果应作为 artifact 或 document version 的一部分保存。

否则每次重建索引都可能得到不同 chunk。

## 43.7 知识库同步

企业知识库经常来自外部系统，例如 Wiki、网盘、工单系统和代码仓库。

同步系统要记录：

1. source connector。
2. sync job ID。
3. sync time。
4. 新增文档。
5. 更新文档。
6. 删除文档。
7. 失败文档。
8. 权限变化。
9. 索引更新状态。

知识库同步不是只导入文本，还要同步权限和删除状态。

如果源文档删除了，RAG 索引也必须处理删除或失效。

删除传播率可以写成：

$$
R_{\mathrm{delete}}=\frac{|\{d:d\in D_{\mathrm{deleted}}\land d\in D_{\mathrm{tombstone}}\}|}{|D_{\mathrm{deleted}}|}
$$

`D_{\mathrm{deleted}}` 是源系统删除或撤权的文档集合，`D_{\mathrm{tombstone}}` 是索引、chunk store 和 citation store 中已标记失效的集合。`R_{\mathrm{delete}}<1` 时，旧文档可能继续被召回。

## 43.8 权限和 ACL

企业 RAG 最容易出事故的是权限。

ACL 是 access control list，访问控制列表。

文档、chunk 和检索结果都要带权限信息。

检索时必须根据用户身份过滤：

1. 用户所属租户。
2. 用户所在部门。
3. 用户角色。
4. 文档权限标签。
5. 文档安全等级。
6. 数据地域约束。

不能先检索再让模型“不要回答无权限内容”。

权限过滤必须在检索阶段强制执行。

检索权限门禁可以写成：

$$
G_{\mathrm{acl}}(u,c)=I(\mathrm{tenant}(u)=\mathrm{tenant}(c))\cdot I(\ell_c\subseteq L_u)\cdot I(r_c\le r_u)
$$

其中 `u` 是用户或 agent 运行身份，`c` 是候选 chunk，`\ell_c` 是 chunk 权限标签，`L_u` 是用户可访问标签集合，`r_c` 和 `r_u` 分别是数据等级和用户授权等级。只有 `G_{\mathrm{acl}}=1` 的 chunk 才能进入候选、rerank、prompt 和日志。

## 43.9 Retrieval Trace

Retrieval trace 记录一次检索发生了什么。

至少包括：

1. query。
2. query rewrite 结果。
3. embedding model version。
4. index version。
5. filters。
6. top-k 参数。
7. candidate chunks。
8. similarity scores。
9. rerank scores。
10. 最终进入 prompt 的 chunks。
11. 被过滤掉的原因。
12. latency。

当用户问“为什么回答引用了这个文档”，retrieval trace 是排查入口。

Retrieval trace 完整率：

$$
C_{\mathrm{retr\_trace}}=\frac{\sum_i I(q_i,e_i,v_i,f_i,k_i,s_i,r_i,l_i\ \mathrm{present})}{N}
$$

其中 `q_i` 是 query / rewrite，`e_i` 是 query embedding 版本，`v_i` 是 index version，`f_i` 是 filter，`k_i` 是 top-k，`s_i` 是 candidate scores，`r_i` 是 rerank 结果，`l_i` 是 latency。缺任一关键字段，线上 bad case 就很难复盘。

## 43.10 Prompt Assembly Trace

RAG 不是检索完就结束，还要组装 prompt。

Prompt assembly trace 记录：

1. system prompt version。
2. user query。
3. retrieved chunks。
4. chunk 排序。
5. token budget。
6. 被截断的内容。
7. 引用格式。
8. 最终 prompt hash。

很多 RAG 问题不是检索错了，而是 prompt 组装时把关键 chunk 截掉了。

Prompt assembly 可以用稳定 hash 绑定：

$$
h_{\mathrm{prompt}}=H(t_{\mathrm{sys}},t_{\mathrm{user}},O_{\mathrm{chunk}},B,\Gamma)
$$

其中 `t_sys` 是 system prompt version，`t_user` 是用户输入，`O_chunk` 是 chunk 排序，`B` 是 token budget，`\Gamma` 是截断和引用格式策略。只记录最终 prompt 文本容易泄露隐私，记录版本、排序、预算和 hash 更适合审计。

## 43.11 Citation 存储

Citation 是回答引用的来源。

Citation 应记录：

1. answer span。
2. chunk ID。
3. document ID。
4. source URI。
5. offset。
6. score。
7. index version。
8. document version。

好的 citation 不只是展示链接，还要能追溯到当时的文档版本。

否则文档更新后，用户看到的引用可能和生成时不一致。

Citation 版本绑定率：

$$
C_{\mathrm{citation}}=\frac{\sum_j I(\mathrm{span}_j,\mathrm{chunk\_id}_j,\mathrm{doc\_id}_j,\mathrm{doc\_version}_j,o_j,v_{\mathrm{index},j}\ \mathrm{present})}{|\{\mathrm{citation}_j\}|}
$$

这个指标衡量引用是否能从答案 span 回到当时的 chunk、文档版本、offset 和索引版本。

## 43.12 Agent 平台中的核心资产

Agent 平台常见资产包括：

1. Agent Definition。
2. Prompt Template。
3. Tool Definition。
4. Tool Permission。
5. Memory。
6. Plan。
7. Tool Call Trace。
8. Execution Trace。
9. Run State。
10. Evaluation Result。

Agent 平台比 RAG 更复杂，因为它有多步决策、工具调用、状态变化和副作用。

## 43.13 Agent Definition

Agent Definition 描述一个 agent 的配置。

包括：

1. agent 名称。
2. 版本。
3. system prompt。
4. 可用工具列表。
5. 模型配置。
6. 记忆策略。
7. 最大步骤数。
8. 超时设置。
9. 权限策略。
10. 安全策略。

Agent Definition 必须版本化。

否则同一个 agent 今天和明天行为不同，trace 无法复现。

Agent definition 可以写成：

$$
A_j=(\mathrm{agent\_id},v_j,p_j,T_j,m_j,b_j,\tau_j,\pi_j,\gamma_j)
$$

其中 `v_j` 是 agent version，`p_j` 是 system prompt version，`T_j` 是可用工具及版本集合，`m_j` 是模型配置，`b_j` 是预算和最大步数，`\tau_j` 是超时策略，`\pi_j` 是权限策略，`\gamma_j` 是安全策略。

## 43.14 Tool Definition

Tool Definition 描述工具能力。

包括：

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

工具描述会影响模型是否调用工具以及如何填参数。

所以工具定义变化也要版本化、灰度和回滚。

工具定义契约完整率：

$$
C_{\mathrm{tool}}=\frac{\sum_j I(n_j,v_j,d_j,x_j,y_j,\pi_j,\eta_j,\rho_j\ \mathrm{present})}{|\{T_j\}|}
$$

其中 `n_j` 是工具名，`v_j` 是版本，`d_j` 是描述，`x_j` 是输入 schema，`y_j` 是输出 schema，`\pi_j` 是权限要求，`\eta_j` 是副作用等级，`\rho_j` 是 retry / timeout 策略。

## 43.15 Tool Permission

工具权限比普通 API 权限更复杂。

因为调用者可能是 agent，不是人直接点击按钮。

平台需要判断：

1. 用户是否有权限使用该工具。
2. agent 是否被授权调用该工具。
3. 工具是否会产生副作用。
4. 是否需要用户确认。
5. 是否需要审批。
6. 是否允许在当前租户和地域调用。

例如查询工具和删除工具的风险完全不同。

高风险工具调用必须记录审计，必要时需要 human-in-the-loop。

工具权限门禁：

$$
G_{\mathrm{tool}}=I(p_{\mathrm{user}}\supseteq p_{\mathrm{tool}})\cdot I(p_{\mathrm{agent}}\supseteq p_{\mathrm{tool}})\cdot I(\mathrm{tenant\ ok})\cdot I(\mathrm{confirm\ ok}\lor \eta\le \eta_{\mathrm{safe}})
$$

`p_user` 是用户权限，`p_agent` 是 agent 被授权权限，`p_tool` 是工具要求权限，`\eta` 是工具副作用等级。查询类工具和删除、转账、发信这类高风险工具必须有不同门禁。

## 43.16 Tool Call Trace

Tool call trace 记录每一次工具调用。

包括：

1. run ID。
2. step ID。
3. tool name。
4. tool version。
5. input arguments。
6. output result。
7. start time。
8. end time。
9. latency。
10. status。
11. error message。
12. permission decision。
13. retry count。

工具输入和输出可能包含敏感信息，要做脱敏和权限控制。

Tool call trace 完整率：

$$
C_{\mathrm{tool\_trace}}=\frac{\sum_i I(r_i,s_i,n_i,v_i,x_i,y_i,t_i,\sigma_i,\pi_i\ \mathrm{present})}{N_{\mathrm{tool}}}
$$

其中 `r_i` 是 run ID，`s_i` 是 step ID，`n_i` 是 tool name，`v_i` 是 tool version，`x_i` 是输入参数，`y_i` 是输出引用或结果摘要，`t_i` 是耗时，`\sigma_i` 是状态，`\pi_i` 是权限决策。

## 43.17 Execution Trace

Execution trace 是 Agent 一次完整运行的记录。

它可以包含：

1. user input。
2. agent version。
3. model version。
4. prompt version。
5. planning steps。
6. tool calls。
7. observations。
8. intermediate messages。
9. final answer。
10. token usage。
11. cost。
12. errors。
13. safety events。

Execution trace 是 Agent 调试、评估、审计和回放的基础。

Execution trace 可以看成 span 图：

$$
G_{\mathrm{trace}}=(V_{\mathrm{span}},E_{\mathrm{parent}},R_{\mathrm{artifact}})
$$

`V_span` 是检索、LLM、工具、memory、rerank、安全检查等 span，`E_parent` 是父子关系，`R_artifact` 是 prompt、tool output、citation、eval result 等 artifact 引用。trace 不是一串日志，而是一棵可回放的证据树。

## 43.18 Trace 回放

Trace 回放用于复现一次 RAG/Agent 行为。

回放需要固定：

1. 模型版本。
2. prompt 版本。
3. agent definition。
4. tool definition。
5. tool outputs。
6. knowledge base version。
7. index version。
8. retrieval results。
9. sampling 参数。

如果外部工具结果已经变化，回放需要使用当时记录的 tool output，或者标记为不可完全复现。

回放就绪门禁：

$$
G_{\mathrm{replay}}=I(m,p,a,T,o,k,v,\theta\ \mathrm{fixed})
$$

其中 `m` 是 model version，`p` 是 prompt version，`a` 是 agent definition，`T` 是 tool definition，`o` 是 tool output 或外部观察，`k` 是 knowledge base version，`v` 是 index version，`\theta` 是 sampling 参数。任一关键项不可固定，都只能做近似回放。

## 43.19 Memory 存储

Agent 常有 memory。

Memory 可以分为：

1. 会话记忆。
2. 用户长期记忆。
3. 任务状态。
4. 工具结果缓存。
5. 偏好信息。

Memory 存储要考虑：

1. 生命周期。
2. 用户可见性。
3. 删除机制。
4. 权限。
5. 隐私。
6. 是否允许进入 prompt。
7. 是否允许用于训练。

记忆不是随便保存聊天记录。它是敏感数据资产。

Memory 写入门禁：

$$
G_{\mathrm{mem}}=I(\mathrm{purpose\ ok})\cdot I(\mathrm{scope\ ok})\cdot I(\mathrm{sensitive}=0\lor \mathrm{explicit\ approval})\cdot I(\mathrm{ttl\ set})\cdot I(\mathrm{delete\ supported})
$$

记忆命中率不能单独作为成功指标。更重要的是写入是否有目的、作用域、权限、TTL、删除机制和用户可见性。

## 43.20 Run State 存储

长任务 Agent 需要保存 run state。

例如：

1. 当前步骤。
2. 已完成任务。
3. 待执行工具。
4. 中间结果。
5. 用户确认状态。
6. 超时状态。
7. cancellation 状态。

Run state 支持：

1. 任务恢复。
2. 暂停和继续。
3. 人工接管。
4. 失败诊断。
5. 审计。

长任务 Agent 不能只依赖内存变量。

## 43.21 安全与审计

RAG/Agent 平台要记录安全事件：

1. 无权限文档过滤。
2. 高风险工具调用。
3. 用户确认。
4. 安全策略拦截。
5. 敏感信息脱敏。
6. 越权访问尝试。
7. 工具调用失败。
8. 异常高成本运行。

审计日志要能回答：谁在什么时间，通过哪个 agent，用了哪个模型，检索了哪些知识，调用了哪些工具，产生了什么结果。

## 43.22 成本追踪

RAG/Agent 的成本来源包括：

1. embedding 生成。
2. 向量检索。
3. rerank。
4. LLM tokens。
5. tool API 调用。
6. 长任务多轮循环。
7. trace 存储。

成本追踪要按：

1. tenant。
2. application。
3. agent。
4. user。
5. run。
6. model。
7. tool。

进行归因。

Agent 特别容易成本失控，因为它可能循环调用模型和工具。

一次 run 的成本可以拆成：

$$
K_{\mathrm{run}}=K_{\mathrm{llm}}+K_{\mathrm{retr}}+K_{\mathrm{rerank}}+K_{\mathrm{tool}}+K_{\mathrm{embed}}+K_{\mathrm{trace}}
$$

成本归因完整率：

$$
C_{\mathrm{cost}}=\frac{\sum_i I(\mathrm{tenant}_i,\mathrm{app}_i,\mathrm{agent}_i,\mathrm{run}_i,\mathrm{component}_i,K_i\ \mathrm{present})}{N_{\mathrm{cost}}}
$$

如果成本只按总账记录，Agent 循环、工具 API、trace 存储和重排成本都无法定位。

## 43.23 Trace 存储的隐私问题

Trace 里可能包含大量敏感信息：

1. 用户输入。
2. 检索文档片段。
3. 工具参数。
4. 工具返回结果。
5. 模型中间推理内容。
6. 最终输出。

因此 trace 存储要支持：

1. 字段级脱敏。
2. 访问控制。
3. TTL。
4. 加密。
5. 审计。
6. 用户删除请求。
7. 采样存储。

不是所有 trace 都应该永久保存完整内容。

Trace 脱敏覆盖率：

$$
C_{\mathrm{redact}}=\frac{N_{\mathrm{redacted\ fields}}}{N_{\mathrm{sensitive\ fields}}}
$$

Trace 保留门禁：

$$
G_{\mathrm{retention}}=I(C_{\mathrm{redact}}\ge \alpha)\cdot I(\mathrm{ttl}\le \tau_{\max})\cdot I(\mathrm{access\ scoped})\cdot I(\mathrm{delete\ supported})
$$

其中 `alpha` 是脱敏阈值，`\tau_max` 是最大保留时长。trace 里如果保存工具参数和检索片段，就必须按敏感数据资产治理。

## 43.24 Trace 采样

完整保存所有 trace 成本高，也有隐私风险。

可以按策略采样：

1. 错误请求全量保存。
2. 高价值业务全量保存。
3. 普通成功请求采样保存。
4. 敏感请求只保存摘要。
5. 高成本 run 保存完整 trace。
6. 用户反馈差的请求保存完整 trace。

采样策略要兼顾排障、评估、成本和隐私。

分层采样率可以写成：

$$
p_{\mathrm{store}}(r)=
\begin{cases}
1, & r\in R_{\mathrm{error}}\cup R_{\mathrm{high\ value}}\cup R_{\mathrm{feedback\ bad}} \\
p_s, & r\in R_{\mathrm{success}} \\
p_{\mathrm{summary}}, & r\in R_{\mathrm{sensitive}}
\end{cases}
$$

采样不是随便丢日志，而是用错误、价值、反馈、敏感等级和成本共同决定保存粒度。

## 43.25 RAG/Agent 平台存储架构

一个可能的架构：

```text
Knowledge Base Store
  -> Document Store
  -> Chunk Store
  -> Embedding Store
  -> Vector Index

Agent Registry
  -> Prompt Registry
  -> Tool Registry
  -> Permission Store
  -> Memory Store

Trace Platform
  -> Retrieval Trace Store
  -> Tool Call Trace Store
  -> Execution Trace Store
  -> Audit Log
  -> Cost Store
```

不同存储承担不同职责，不要把所有东西塞进一个向量数据库。

## 43.26 RAG/Agent 平台存储审计指标和最小 demo

把本章落到平台审计时，可以用 16 个门禁：

1. Knowledge Base Contract：知识库是否有租户、owner、来源、权限、embedding、chunking、索引、同步和保留策略。
2. Document Chunk Version Contract：document、chunk、offset、checksum、document version 和 embedding ID 是否完整。
3. Sync Delete Propagation：源系统删除、撤权和权限变化是否传播到 chunk、index 和 citation。
4. ACL Permission Enforcement：检索、rerank、prompt 和 trace 是否都强制 tenant / ACL filter。
5. Retrieval Trace Completeness：query、rewrite、embedding version、index version、filter、candidate、rerank、final chunk 和 latency 是否完整。
6. Prompt Assembly Trace：system prompt、template、chunk 顺序、token budget、截断和 prompt hash 是否可复现。
7. Citation Version Binding：answer span 是否绑定 chunk、document version、offset、source URI 和 index version。
8. Agent Definition Versioning：agent definition、prompt、model、工具集合、预算、权限和安全策略是否版本化。
9. Tool Definition Contract：tool name、version、description、input/output schema、endpoint、timeout、权限和副作用等级是否完整。
10. Tool Permission Gate：用户权限、agent 权限、工具权限、副作用等级和确认策略是否一起判断。
11. Tool Call Trace Completeness：每次工具调用是否记录 run、step、tool version、参数、结果引用、耗时、状态、错误、权限决策和重试。
12. Execution Trace Replay Readiness：span 树、artifact 引用、模型 / prompt / tool / KB / index / sampling 参数是否足够回放。
13. Memory Privacy Lifecycle：memory 写入是否有 purpose、scope、TTL、可见性、删除机制和敏感信息阻断。
14. Trace Privacy Retention：trace 是否支持字段级脱敏、TTL、加密、访问控制、删除请求和采样。
15. Cost Attribution Governance：LLM、retrieval、rerank、tool、embedding 和 trace 成本是否能按 tenant / app / agent / run / component 归因。
16. RAG Agent Storage Gate：以上证据是否有 owner、门禁、回滚和 P0 风险阻断。

综合门禁可以写成：

$$
G_{\mathrm{rag\_agent\_storage}}=\prod_{j=1}^{16}G_j
$$

其中每个 `G_j` 是上面一个子门禁。面试里不要只说“我们把 trace 存起来”，而要说明 trace 是否能解释、回放、评估、审计、脱敏、删除和归因。

下面是一个 0 依赖 demo。它不是生产实现，而是展示如何把 RAG/Agent 平台存储设计变成可检查的结构化审计。

```python
from copy import deepcopy


class MiniRAGAgentStorageAudit:
    GATES = [
        "knowledge_base_contract",
        "document_chunk_version_contract",
        "sync_delete_propagation",
        "acl_permission_enforcement",
        "retrieval_trace_completeness",
        "prompt_assembly_trace",
        "citation_version_binding",
        "agent_definition_versioning",
        "tool_definition_contract",
        "tool_permission_gate",
        "tool_call_trace_completeness",
        "execution_trace_replay_readiness",
        "memory_privacy_lifecycle",
        "trace_privacy_retention",
        "cost_attribution_governance",
        "rag_agent_storage_gate",
    ]

    KB_FIELDS = [
        "kb_id",
        "tenant",
        "owner",
        "source_connectors",
        "permission_policy",
        "embedding_version",
        "chunking_version",
        "index_version",
        "sync_state",
        "retention_policy",
    ]
    DOC_FIELDS = [
        "doc_id",
        "kb_id",
        "version",
        "source_uri",
        "content_hash",
        "permission_labels",
        "parsing_status",
    ]
    CHUNK_FIELDS = [
        "chunk_id",
        "doc_id",
        "doc_version",
        "offset_start",
        "offset_end",
        "embedding_id",
        "permission_labels",
        "checksum",
        "tenant",
    ]
    RETRIEVAL_FIELDS = [
        "query_id",
        "query",
        "query_embedding_version",
        "index_version",
        "filters",
        "top_k",
        "candidate_chunks",
        "rerank_scores",
        "final_chunks",
        "latency_ms",
    ]
    PROMPT_FIELDS = [
        "system_prompt_version",
        "template_version",
        "chunk_order",
        "token_budget",
        "prompt_hash",
        "context_tokens",
    ]
    CITATION_FIELDS = [
        "answer_span",
        "chunk_id",
        "doc_id",
        "doc_version",
        "source_uri",
        "offset_start",
        "offset_end",
        "index_version",
    ]
    AGENT_FIELDS = [
        "agent_id",
        "version",
        "system_prompt_version",
        "tool_versions",
        "model_version",
        "max_steps",
        "timeout_s",
        "permission_policy",
        "memory_policy",
        "safety_policy",
    ]
    TOOL_FIELDS = [
        "name",
        "version",
        "description",
        "input_schema",
        "output_schema",
        "endpoint",
        "timeout_s",
        "retry_policy",
        "permission_requirement",
        "side_effect_level",
    ]
    TOOL_TRACE_FIELDS = [
        "run_id",
        "step_id",
        "tool_name",
        "tool_version",
        "input_args",
        "output_ref",
        "status",
        "latency_ms",
        "permission_decision",
        "retry_count",
        "redacted",
    ]
    EXEC_TRACE_FIELDS = [
        "run_id",
        "agent_version",
        "model_version",
        "prompt_version",
        "span_ids",
        "parent_map",
        "retrieval_trace_ids",
        "tool_trace_ids",
        "final_answer_hash",
        "token_usage",
        "artifact_refs",
    ]
    COST_FIELDS = ["tenant", "application", "agent", "run_id", "component", "usd"]

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    def chunk_map(self, case):
        return {chunk["chunk_id"]: chunk for chunk in case.get("chunks", [])}

    def doc_map(self, case):
        return {doc["doc_id"]: doc for doc in case.get("documents", [])}

    def tool_map(self, case):
        return {tool["name"]: tool for tool in case.get("tools", [])}

    def chunk_allowed(self, case, chunk_id):
        chunks = self.chunk_map(case)
        chunk = chunks.get(chunk_id)
        if not chunk:
            return False
        user = case.get("user", {})
        user_labels = set(user.get("labels", []))
        chunk_labels = set(chunk.get("permission_labels", []))
        return (
            chunk.get("tenant") == user.get("tenant")
            and chunk_labels.issubset(user_labels)
        )

    def knowledge_base_contract(self, case):
        kb = case.get("knowledge_base", {})
        return self.coverage(kb, self.KB_FIELDS) == 1.0

    def document_chunk_version_contract(self, case):
        documents = case.get("documents", [])
        chunks = case.get("chunks", [])
        docs_ok = documents and all(
            self.coverage(document, self.DOC_FIELDS) == 1.0
            for document in documents
        )
        chunks_ok = chunks and all(
            self.coverage(chunk, self.CHUNK_FIELDS) == 1.0
            and chunk["offset_end"] > chunk["offset_start"]
            for chunk in chunks
        )
        return bool(docs_ok and chunks_ok)

    def sync_delete_propagation(self, case):
        sync = case.get("sync_job", {})
        required = ["sync_id", "source_connector", "deleted", "tombstone_applied", "permission_changes"]
        if self.coverage(sync, required) < 1.0:
            return False
        deleted = set(sync.get("deleted", []))
        tombstone = set(sync.get("tombstone_applied", []))
        return deleted.issubset(tombstone)

    def acl_permission_enforcement(self, case):
        trace = case.get("retrieval_trace", {})
        for candidate in trace.get("candidate_chunks", []):
            allowed = self.chunk_allowed(case, candidate.get("chunk_id"))
            if not allowed and not candidate.get("filtered", False):
                return False
        return all(self.chunk_allowed(case, chunk_id) for chunk_id in trace.get("final_chunks", []))

    def retrieval_trace_completeness(self, case):
        trace = case.get("retrieval_trace", {})
        if self.coverage(trace, self.RETRIEVAL_FIELDS) < 1.0:
            return False
        candidate_ids = {candidate.get("chunk_id") for candidate in trace["candidate_chunks"]}
        final_ids = set(trace["final_chunks"])
        candidate_scores_ok = all("score" in candidate for candidate in trace["candidate_chunks"])
        return final_ids.issubset(candidate_ids) and candidate_scores_ok

    def prompt_assembly_trace(self, case):
        prompt = case.get("prompt_trace", {})
        trace = case.get("retrieval_trace", {})
        if self.coverage(prompt, self.PROMPT_FIELDS) < 1.0:
            return False
        return set(trace.get("final_chunks", [])).issubset(set(prompt.get("chunk_order", [])))

    def citation_version_binding(self, case):
        citations = case.get("citations", [])
        chunks = self.chunk_map(case)
        if not citations:
            return False
        for citation in citations:
            if self.coverage(citation, self.CITATION_FIELDS) < 1.0:
                return False
            chunk = chunks.get(citation["chunk_id"])
            if not chunk:
                return False
            if chunk["doc_id"] != citation["doc_id"]:
                return False
            if chunk["doc_version"] != citation["doc_version"]:
                return False
        return True

    def agent_definition_versioning(self, case):
        agent = case.get("agent_definition", {})
        if self.coverage(agent, self.AGENT_FIELDS) < 1.0:
            return False
        return bool(agent.get("tool_versions"))

    def tool_definition_contract(self, case):
        tools = case.get("tools", [])
        if not tools:
            return False
        for tool in tools:
            if self.coverage(tool, self.TOOL_FIELDS) < 1.0:
                return False
            if not tool.get("input_schema") or not tool.get("output_schema"):
                return False
        return True

    def tool_permission_gate(self, case):
        tools = self.tool_map(case)
        policy = case.get("tool_policy", {})
        user_permissions = set(policy.get("user_permissions", []))
        agent_permissions = set(policy.get("agent_permissions", []))
        confirmations = set(policy.get("confirmed_actions", []))
        high_risk = {"write", "destructive", "external"}
        for trace in case.get("tool_traces", []):
            tool = tools.get(trace.get("tool_name"))
            if not tool:
                return False
            required = tool["permission_requirement"]
            has_permission = required in user_permissions and required in agent_permissions
            needs_confirmation = tool["side_effect_level"] in high_risk
            confirmed = f"{trace.get('tool_name')}:{trace.get('step_id')}" in confirmations
            if trace.get("status") == "ok" and not has_permission:
                return False
            if trace.get("status") == "ok" and needs_confirmation and not confirmed:
                return False
        return True

    def tool_call_trace_completeness(self, case):
        tools = self.tool_map(case)
        traces = case.get("tool_traces", [])
        if not traces:
            return False
        for trace in traces:
            if self.coverage(trace, self.TOOL_TRACE_FIELDS) < 1.0:
                return False
            tool = tools.get(trace["tool_name"])
            if not tool or tool["version"] != trace["tool_version"]:
                return False
            if trace.get("redacted") is not True:
                return False
        return True

    def execution_trace_replay_readiness(self, case):
        trace = case.get("execution_trace", {})
        replay = case.get("replay", {})
        if self.coverage(trace, self.EXEC_TRACE_FIELDS) < 1.0:
            return False
        spans = set(trace.get("span_ids", []))
        parent_map = trace.get("parent_map", {})
        non_root_spans = spans - {"root"}
        parents_ok = all(span in parent_map and parent_map[span] in spans for span in non_root_spans)
        replay_ok = replay and all(replay.values())
        return parents_ok and replay_ok

    def memory_privacy_lifecycle(self, case):
        memory = case.get("memory", {})
        if not memory.get("deletion_supported"):
            return False
        for write in memory.get("writes", []):
            if not write.get("ttl_days"):
                return False
            if not write.get("visible_to_user"):
                return False
            sensitive = write.get("sensitive", False)
            blocked = write.get("blocked", False)
            approved = write.get("explicit_approval", False)
            if sensitive and not blocked and not approved:
                return False
        return True

    def trace_privacy_retention(self, case):
        privacy = case.get("trace_privacy", {})
        total = privacy.get("fields_total", 0)
        redacted = privacy.get("redacted_fields", 0)
        if total <= 0:
            return False
        redaction_coverage = redacted / total
        return (
            redaction_coverage >= 0.95
            and privacy.get("ttl_days", 9999) <= 90
            and bool(privacy.get("encryption"))
            and bool(privacy.get("access_policy"))
            and privacy.get("raw_cot_stored") is False
        )

    def cost_attribution_governance(self, case):
        costs = case.get("costs", [])
        if not costs:
            return False
        total = 0.0
        for cost in costs:
            if self.coverage(cost, self.COST_FIELDS) < 1.0:
                return False
            total += float(cost["usd"])
        budget = case.get("platform_gate", {}).get("cost_budget_usd", 9999.0)
        return total <= budget

    def rag_agent_storage_gate(self, case):
        gate = case.get("platform_gate", {})
        return (
            gate.get("enabled") is True
            and bool(gate.get("owner"))
            and gate.get("p0_open") is False
            and gate.get("rollback_ready") is True
        )

    def audit_case(self, case):
        return {gate: getattr(self, gate)(case) for gate in self.GATES}

    def run_all(self, cases):
        results = {case["case_id"]: self.audit_case(case) for case in cases}
        metrics = {}
        for gate in self.GATES:
            passed = sum(1 for result in results.values() if result[gate])
            metrics[gate] = round(passed / len(cases), 3)
        failed_cases = [
            case_id
            for case_id, result in results.items()
            if not all(result.values())
        ]
        failed_gates = [
            gate
            for gate in self.GATES
            if any(not result[gate] for result in results.values())
        ]
        return {
            "metrics": metrics,
            "hard_blocker_count": len(failed_cases),
            "failed_cases": failed_cases,
            "failed_gates": failed_gates,
            "rag_agent_storage_gate_pass": metrics["rag_agent_storage_gate"] == 1.0,
        }

    def example_outputs(self, case):
        trace = case["retrieval_trace"]
        prompt = case["prompt_trace"]
        costs = case["costs"]
        privacy = case["trace_privacy"]
        return {
            "kb_contract_coverage": round(self.coverage(case["knowledge_base"], self.KB_FIELDS), 3),
            "document_version_coverage": round(
                sum(self.coverage(doc, self.DOC_FIELDS) for doc in case["documents"])
                / len(case["documents"]),
                3,
            ),
            "chunk_citation_coverage": round(
                sum(self.coverage(chunk, self.CHUNK_FIELDS) for chunk in case["chunks"])
                / len(case["chunks"]),
                3,
            ),
            "permission_filtered_chunks": [
                candidate["chunk_id"]
                for candidate in trace["candidate_chunks"]
                if candidate.get("filtered")
            ],
            "retrieval_trace_complete": self.retrieval_trace_completeness(case),
            "prompt_hash_ready": bool(prompt.get("prompt_hash")),
            "citation_version_bound": self.citation_version_binding(case),
            "tool_schema_coverage": round(
                sum(self.coverage(tool, self.TOOL_FIELDS) for tool in case["tools"])
                / len(case["tools"]),
                3,
            ),
            "tool_permission_allowed": case["tool_traces"][0]["status"] == "ok",
            "dangerous_tool_blocked": case["tool_traces"][1]["status"].startswith("blocked"),
            "tool_trace_complete": self.tool_call_trace_completeness(case),
            "trace_replay_ready": self.execution_trace_replay_readiness(case),
            "memory_private_blocked": case["memory"]["writes"][1]["blocked"],
            "trace_redaction_coverage": round(
                privacy["redacted_fields"] / privacy["fields_total"],
                3,
            ),
            "estimated_run_cost_usd": round(sum(cost["usd"] for cost in costs), 3),
            "audit_event_count": len(case["audit_logs"]),
        }


def build_good_case():
    return {
        "case_id": "production_ready",
        "knowledge_base": {
            "kb_id": "kb_support",
            "tenant": "acme",
            "owner": "ai-infra",
            "source_connectors": ["wiki", "tickets"],
            "permission_policy": "tenant_acl_labels",
            "embedding_version": "emb_v4",
            "chunking_version": "chunk_v3",
            "index_version": "idx_7",
            "sync_state": "green",
            "retention_policy": "90d_trace_immutable_doc_versions",
        },
        "documents": [
            {
                "doc_id": "doc_policy",
                "kb_id": "kb_support",
                "version": "v3",
                "source_uri": "wiki://policy",
                "content_hash": "sha256:policy",
                "permission_labels": ["public"],
                "parsing_status": "parsed",
            },
            {
                "doc_id": "doc_salary",
                "kb_id": "kb_support",
                "version": "v5",
                "source_uri": "wiki://salary",
                "content_hash": "sha256:salary",
                "permission_labels": ["finance_private"],
                "parsing_status": "parsed",
            },
        ],
        "chunks": [
            {
                "chunk_id": "chunk_policy_public",
                "doc_id": "doc_policy",
                "doc_version": "v3",
                "offset_start": 0,
                "offset_end": 120,
                "embedding_id": "emb_001",
                "permission_labels": ["public"],
                "checksum": "sha256:chunk-policy",
                "tenant": "acme",
            },
            {
                "chunk_id": "chunk_salary_private",
                "doc_id": "doc_salary",
                "doc_version": "v5",
                "offset_start": 20,
                "offset_end": 180,
                "embedding_id": "emb_002",
                "permission_labels": ["finance_private"],
                "checksum": "sha256:chunk-salary",
                "tenant": "acme",
            },
        ],
        "sync_job": {
            "sync_id": "sync_2026_06_15",
            "source_connector": "wiki",
            "added": ["doc_policy"],
            "updated": [],
            "deleted": ["doc_old"],
            "tombstone_applied": ["doc_old"],
            "permission_changes": ["doc_salary"],
            "failed": [],
        },
        "user": {
            "tenant": "acme",
            "roles": ["employee"],
            "labels": ["public"],
        },
        "retrieval_trace": {
            "query_id": "query_1",
            "query": "What is the refund policy?",
            "query_embedding_version": "emb_v4",
            "index_version": "idx_7",
            "filters": {"tenant": "acme", "labels": ["public"]},
            "top_k": 3,
            "candidate_chunks": [
                {"chunk_id": "chunk_policy_public", "score": 0.91, "filtered": False},
                {
                    "chunk_id": "chunk_salary_private",
                    "score": 0.88,
                    "filtered": True,
                    "reason": "acl",
                },
            ],
            "rerank_scores": {"chunk_policy_public": 0.97},
            "final_chunks": ["chunk_policy_public"],
            "latency_ms": 42,
        },
        "prompt_trace": {
            "system_prompt_version": "sys_v5",
            "template_version": "rag_template_v2",
            "chunk_order": ["chunk_policy_public"],
            "token_budget": 4000,
            "truncated_chunks": [],
            "prompt_hash": "sha256:prompt",
            "context_tokens": 512,
        },
        "citations": [
            {
                "answer_span": "refunds are handled within 7 days",
                "chunk_id": "chunk_policy_public",
                "doc_id": "doc_policy",
                "doc_version": "v3",
                "source_uri": "wiki://policy",
                "offset_start": 0,
                "offset_end": 120,
                "index_version": "idx_7",
            }
        ],
        "agent_definition": {
            "agent_id": "support_agent",
            "version": "agent_2026_06_01",
            "system_prompt_version": "sys_v5",
            "tool_versions": {"kb_search": "1.2", "ticket_delete": "1.0"},
            "model_version": "chat_prod_7",
            "max_steps": 6,
            "timeout_s": 60,
            "permission_policy": "least_privilege",
            "memory_policy": "scoped_user_memory",
            "safety_policy": "confirm_high_risk_tools",
        },
        "tools": [
            {
                "name": "kb_search",
                "version": "1.2",
                "description": "Search approved knowledge base chunks.",
                "input_schema": {"query": "string", "kb_id": "string"},
                "output_schema": {"chunks": "list"},
                "endpoint": "internal://kb_search",
                "timeout_s": 5,
                "retry_policy": "safe_read_retry",
                "permission_requirement": "kb.read",
                "side_effect_level": "read",
            },
            {
                "name": "ticket_delete",
                "version": "1.0",
                "description": "Delete a ticket after explicit approval.",
                "input_schema": {"ticket_id": "string"},
                "output_schema": {"deleted": "bool"},
                "endpoint": "internal://ticket_delete",
                "timeout_s": 10,
                "retry_policy": "no_auto_retry",
                "permission_requirement": "ticket.delete",
                "side_effect_level": "destructive",
            },
        ],
        "tool_policy": {
            "user_permissions": ["kb.read", "ticket.read"],
            "agent_permissions": ["kb.read", "ticket.read"],
            "confirmed_actions": [],
            "tenant": "acme",
            "region": "us",
        },
        "tool_traces": [
            {
                "run_id": "run_1",
                "step_id": "s1",
                "tool_name": "kb_search",
                "tool_version": "1.2",
                "input_args": {"query": "refund policy", "kb_id": "kb_support"},
                "output_ref": "artifact://tool/run_1/s1",
                "status": "ok",
                "latency_ms": 32,
                "permission_decision": "allowed",
                "retry_count": 0,
                "redacted": True,
            },
            {
                "run_id": "run_1",
                "step_id": "s2",
                "tool_name": "ticket_delete",
                "tool_version": "1.0",
                "input_args": {"ticket_id": "T-7"},
                "output_ref": "blocked:no_output",
                "status": "blocked_requires_permission",
                "latency_ms": 0,
                "permission_decision": "blocked_requires_permission",
                "retry_count": 0,
                "redacted": True,
            },
        ],
        "execution_trace": {
            "run_id": "run_1",
            "agent_version": "agent_2026_06_01",
            "model_version": "chat_prod_7",
            "prompt_version": "sys_v5",
            "span_ids": ["root", "retrieval", "tool_s1", "final"],
            "parent_map": {"retrieval": "root", "tool_s1": "root", "final": "root"},
            "retrieval_trace_ids": ["query_1"],
            "tool_trace_ids": ["s1", "s2"],
            "final_answer_hash": "sha256:answer",
            "token_usage": {"input": 900, "output": 140},
            "errors": [],
            "safety_events": ["ticket_delete_blocked"],
            "artifact_refs": ["artifact://prompt/run_1", "artifact://tool/run_1/s1"],
        },
        "replay": {
            "model_version_fixed": True,
            "prompt_version_fixed": True,
            "agent_definition_fixed": True,
            "tool_definitions_fixed": True,
            "tool_outputs_recorded": True,
            "kb_version_fixed": True,
            "index_version_fixed": True,
            "sampling_params_fixed": True,
        },
        "memory": {
            "writes": [
                {
                    "key": "preferred_language",
                    "scope": "user",
                    "value": "zh",
                    "sensitive": False,
                    "blocked": False,
                    "ttl_days": 180,
                    "visible_to_user": True,
                    "training_allowed": False,
                },
                {
                    "key": "private_card_number",
                    "scope": "user",
                    "value": "redacted",
                    "sensitive": True,
                    "blocked": True,
                    "ttl_days": 1,
                    "visible_to_user": True,
                    "training_allowed": False,
                },
            ],
            "reads": [{"key": "preferred_language", "authorized": True, "used_in_prompt": True}],
            "deletion_supported": True,
        },
        "trace_privacy": {
            "fields_total": 8,
            "redacted_fields": 8,
            "ttl_days": 30,
            "encryption": "kms",
            "access_policy": "trace.read.scoped",
            "sample_policy": "error_full_success_sampled",
            "deletion_request_ids": ["del_1"],
            "raw_cot_stored": False,
        },
        "costs": [
            {
                "tenant": "acme",
                "application": "support",
                "agent": "support_agent",
                "run_id": "run_1",
                "component": "llm",
                "usd": 0.15,
            },
            {
                "tenant": "acme",
                "application": "support",
                "agent": "support_agent",
                "run_id": "run_1",
                "component": "retrieval",
                "usd": 0.02,
            },
            {
                "tenant": "acme",
                "application": "support",
                "agent": "support_agent",
                "run_id": "run_1",
                "component": "tool",
                "usd": 0.047,
            },
        ],
        "audit_logs": [
            {"event": "retrieval_allowed", "run_id": "run_1"},
            {"event": "acl_filtered_chunk", "chunk_id": "chunk_salary_private"},
            {"event": "tool_allowed", "tool": "kb_search"},
            {"event": "tool_blocked", "tool": "ticket_delete"},
            {"event": "trace_redacted", "run_id": "run_1"},
        ],
        "platform_gate": {
            "enabled": True,
            "owner": "ai-infra",
            "p0_open": False,
            "rollback_ready": True,
            "cost_budget_usd": 1.0,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "knowledge_base_contract_missing_bad"
    case["knowledge_base"].pop("owner")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "document_chunk_version_missing_bad"
    case["chunks"][0].pop("checksum")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "sync_delete_not_propagated_bad"
    case["sync_job"]["tombstone_applied"] = []
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "acl_filter_missing_bad"
    case["retrieval_trace"]["candidate_chunks"][1]["filtered"] = False
    case["retrieval_trace"]["final_chunks"] = ["chunk_policy_public", "chunk_salary_private"]
    case["prompt_trace"]["chunk_order"] = ["chunk_policy_public", "chunk_salary_private"]
    case["citations"].append(
        {
            "answer_span": "private salary rule",
            "chunk_id": "chunk_salary_private",
            "doc_id": "doc_salary",
            "doc_version": "v5",
            "source_uri": "wiki://salary",
            "offset_start": 20,
            "offset_end": 180,
            "index_version": "idx_7",
        }
    )
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "retrieval_trace_incomplete_bad"
    case["retrieval_trace"].pop("query_embedding_version")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "prompt_trace_hash_missing_bad"
    case["prompt_trace"].pop("prompt_hash")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "citation_version_missing_bad"
    case["citations"][0].pop("doc_version")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "agent_definition_unversioned_bad"
    case["agent_definition"].pop("version")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tool_definition_schema_missing_bad"
    case["tools"][0]["input_schema"] = {}
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tool_permission_bypass_bad"
    case["tool_traces"][1]["status"] = "ok"
    case["tool_traces"][1]["permission_decision"] = "allowed"
    case["tool_traces"][1]["output_ref"] = "artifact://tool/run_1/s2"
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "tool_call_trace_incomplete_bad"
    case["tool_traces"][0].pop("permission_decision")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "execution_replay_not_ready_bad"
    case["replay"]["tool_outputs_recorded"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "memory_sensitive_write_bad"
    case["memory"]["writes"][1]["blocked"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "trace_privacy_retention_bad"
    case["trace_privacy"]["redacted_fields"] = 5
    case["trace_privacy"]["ttl_days"] = 365
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cost_attribution_missing_bad"
    case["costs"][0].pop("tenant")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "rag_agent_storage_gate_missing_bad"
    case["platform_gate"]["enabled"] = False
    cases.append(case)

    return cases


audit = MiniRAGAgentStorageAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("rag_agent_storage_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("rag_agent_storage_gate_pass=" + repr(summary["rag_agent_storage_gate_pass"]))
```

参考输出应类似：

```text
rag_agent_storage_examples={'kb_contract_coverage': 1.0, 'document_version_coverage': 1.0, 'chunk_citation_coverage': 1.0, 'permission_filtered_chunks': ['chunk_salary_private'], 'retrieval_trace_complete': True, 'prompt_hash_ready': True, 'citation_version_bound': True, 'tool_schema_coverage': 1.0, 'tool_permission_allowed': True, 'dangerous_tool_blocked': True, 'tool_trace_complete': True, 'trace_replay_ready': True, 'memory_private_blocked': True, 'trace_redaction_coverage': 1.0, 'estimated_run_cost_usd': 0.217, 'audit_event_count': 5}
metrics={'knowledge_base_contract': 0.941, 'document_chunk_version_contract': 0.941, 'sync_delete_propagation': 0.941, 'acl_permission_enforcement': 0.941, 'retrieval_trace_completeness': 0.941, 'prompt_assembly_trace': 0.941, 'citation_version_binding': 0.941, 'agent_definition_versioning': 0.941, 'tool_definition_contract': 0.941, 'tool_permission_gate': 0.941, 'tool_call_trace_completeness': 0.941, 'execution_trace_replay_readiness': 0.941, 'memory_privacy_lifecycle': 0.941, 'trace_privacy_retention': 0.941, 'cost_attribution_governance': 0.941, 'rag_agent_storage_gate': 0.941}
hard_blocker_count=16
failed_cases=['knowledge_base_contract_missing_bad', 'document_chunk_version_missing_bad', 'sync_delete_not_propagated_bad', 'acl_filter_missing_bad', 'retrieval_trace_incomplete_bad', 'prompt_trace_hash_missing_bad', 'citation_version_missing_bad', 'agent_definition_unversioned_bad', 'tool_definition_schema_missing_bad', 'tool_permission_bypass_bad', 'tool_call_trace_incomplete_bad', 'execution_replay_not_ready_bad', 'memory_sensitive_write_bad', 'trace_privacy_retention_bad', 'cost_attribution_missing_bad', 'rag_agent_storage_gate_missing_bad']
failed_gates=['knowledge_base_contract', 'document_chunk_version_contract', 'sync_delete_propagation', 'acl_permission_enforcement', 'retrieval_trace_completeness', 'prompt_assembly_trace', 'citation_version_binding', 'agent_definition_versioning', 'tool_definition_contract', 'tool_permission_gate', 'tool_call_trace_completeness', 'execution_trace_replay_readiness', 'memory_privacy_lifecycle', 'trace_privacy_retention', 'cost_attribution_governance', 'rag_agent_storage_gate']
rag_agent_storage_gate_pass=False
```

这个 demo 的核心不是把所有存储组件写成一个类，而是展示平台验收方式：每个 bad case 都应该触发一个明确门禁，最后能回答“为什么这次回答、工具调用或 Agent run 能被解释、复现、审计和删除”。

## 43.27 常见误区

误区一：RAG 平台就是向量库。

向量库只是检索组件，RAG 还需要知识库、文档版本、chunk、权限、citation、prompt assembly 和 trace。

误区二：Agent trace 只用于 debug。

Trace 还用于评估、审计、成本归因、回放、质量改进和安全分析。

误区三：工具定义不需要版本化。

工具描述、schema、权限和 endpoint 变化都会影响 agent 行为。

误区四：Memory 越多越好。

Memory 会带来隐私、污染、成本和错误引用风险，必须有生命周期和权限治理。

误区五：权限可以交给模型判断。

权限必须由平台在检索和工具调用阶段强制执行，不能依赖模型自觉。

## 43.28 面试常见追问

问题一：企业 RAG 平台要存哪些东西？

可以回答：要存 knowledge base、document、chunk、embedding、index version、权限元数据、retrieval config、rerank config、prompt template、citation 和 retrieval trace。

问题二：Agent 平台为什么需要 trace？

可以回答：Agent 是多步执行系统，trace 记录模型决策、工具调用、中间结果、错误、安全事件和成本，用于调试、回放、审计和评估。

问题三：如何避免 RAG 越权检索？

可以回答：文档和 chunk 存权限标签，检索时根据用户身份做强制 metadata filter 或物理索引隔离，不能把无权限内容交给模型再要求它不回答。

问题四：工具调用如何做权限控制？

可以回答：同时检查用户权限、agent 权限、工具风险等级、租户策略和是否需要用户确认，高风险工具调用要审计并支持 human-in-the-loop。

问题五：如何设计 RAG / Agent 平台存储审计？

可以回答：先把对象拆成 knowledge base、document、chunk、citation、agent definition、tool definition、memory、run state、retrieval trace、tool call trace、execution trace、audit log 和 cost record；再按版本、权限、回放、隐私、删除、采样、成本和 owner 做门禁；最后用 trace span tree 串起检索、prompt、工具、memory、安全事件和最终回答，证明线上 bad case 可解释、可复现、可回滚。

## 43.29 小练习

1. Knowledge Base、Document 和 Chunk 有什么区别？
2. Retrieval trace 应该记录哪些字段？
3. 为什么 citation 要绑定 document version 和 chunk ID？
4. Agent Definition 应该包含哪些内容？
5. Tool Definition 为什么要版本化？
6. Agent memory 存储有哪些隐私风险？
7. Trace 采样策略如何设计？
8. 如何设计一个支持回放的 Agent trace 系统？
9. 用纯 Python 实现一个最小 RAG / Agent 平台存储审计 demo，覆盖知识库契约、ACL、retrieval trace、tool permission、execution replay、memory 隐私和成本归因。
10. 构造 5 个线上事故样本：越权 chunk 进入 prompt、tool schema 变更未版本化、trace 缺 tool output、memory 写入敏感字段、源文档删除未传播，并说明各自应触发的门禁。

## 43.30 本章小结

本章讲了 RAG/Agent 平台中的知识库、工具和 trace 存储。

你需要记住：

1. RAG 平台不只是向量库，还包括知识库、文档、chunk、embedding、索引、权限、citation 和 retrieval trace。
2. Agent 平台要管理 agent definition、tool definition、permission、memory、run state 和 execution trace。
3. 权限必须在检索和工具调用阶段由平台强制执行，不能依赖模型判断。
4. Trace 是调试、回放、评估、审计和成本治理的核心数据。
5. RAG/Agent 存储系统必须同时考虑版本、权限、隐私、成本、生命周期和可观测性。
6. 第二轮精修后，本章新增的核心抓手是 RAG / Agent Storage Gate：让每次回答、检索、工具调用、memory 读写和成本记录都有版本、权限、trace、回放、隐私和审计证据。

下一章我们会进入第六部分：可观测性、可靠性与成本治理，先讲 AI Infra 可观测性总览。
