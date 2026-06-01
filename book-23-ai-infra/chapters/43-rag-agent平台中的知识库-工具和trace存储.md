# 第 43 章 RAG/Agent 平台中的知识库、工具和 trace 存储

上一章讲了 Feature Store、Embedding Store 与向量索引基础设施。本章作为第五部分收尾，讲 RAG/Agent 平台中的知识库、工具和 trace 存储。

RAG 和 Agent 是大模型应用平台中的常见形态。它们不只依赖模型本身，还依赖知识库、工具、权限、记忆、调用链路和可回放 trace。如果这些资产没有统一管理，应用很快会变成难以调试和治理的黑盒。

先记住一句话：

> RAG/Agent 平台的核心数据资产，不只是 prompt 和向量库，还包括知识库、工具定义、权限、检索记录、工具调用记录和完整执行 trace。

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

## 43.26 常见误区

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

## 43.27 面试常见追问

问题一：企业 RAG 平台要存哪些东西？

可以回答：要存 knowledge base、document、chunk、embedding、index version、权限元数据、retrieval config、rerank config、prompt template、citation 和 retrieval trace。

问题二：Agent 平台为什么需要 trace？

可以回答：Agent 是多步执行系统，trace 记录模型决策、工具调用、中间结果、错误、安全事件和成本，用于调试、回放、审计和评估。

问题三：如何避免 RAG 越权检索？

可以回答：文档和 chunk 存权限标签，检索时根据用户身份做强制 metadata filter 或物理索引隔离，不能把无权限内容交给模型再要求它不回答。

问题四：工具调用如何做权限控制？

可以回答：同时检查用户权限、agent 权限、工具风险等级、租户策略和是否需要用户确认，高风险工具调用要审计并支持 human-in-the-loop。

## 43.28 小练习

1. Knowledge Base、Document 和 Chunk 有什么区别？
2. Retrieval trace 应该记录哪些字段？
3. 为什么 citation 要绑定 document version 和 chunk ID？
4. Agent Definition 应该包含哪些内容？
5. Tool Definition 为什么要版本化？
6. Agent memory 存储有哪些隐私风险？
7. Trace 采样策略如何设计？
8. 如何设计一个支持回放的 Agent trace 系统？

## 43.29 本章小结

本章讲了 RAG/Agent 平台中的知识库、工具和 trace 存储。

你需要记住：

1. RAG 平台不只是向量库，还包括知识库、文档、chunk、embedding、索引、权限、citation 和 retrieval trace。
2. Agent 平台要管理 agent definition、tool definition、permission、memory、run state 和 execution trace。
3. 权限必须在检索和工具调用阶段由平台强制执行，不能依赖模型判断。
4. Trace 是调试、回放、评估、审计和成本治理的核心数据。
5. RAG/Agent 存储系统必须同时考虑版本、权限、隐私、成本、生命周期和可观测性。

下一章我们会进入第六部分：可观测性、可靠性与成本治理，先讲 AI Infra 可观测性总览。
