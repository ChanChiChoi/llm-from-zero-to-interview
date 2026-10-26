# Agent Memory 深度样稿研究记录

- 研究日期：2026-10-26
- 对应正文：book-17-agent-tool-use/chapters/05-memory系统.md
- 研究目的：把“Memory 是一种能力、分若干类型、检索后注入 prompt”的宏观介绍拆成可说明、可追溯、可评估的机制；重点核实分类、写入/检索、时间更新、上下文管理、反思、删除和长期记忆评测。
- 阅读方式：以论文摘要和正文相关章节、官方框架文档和工程团队文章为依据；搜索结果只用于发现材料。论文里的数字按原论文设置转述，博客中的公司结论标记为作者/公司报告。技术文章不能证明未公开的产品行为。

## 资料与核验结果

### 综述：统一分类框架

**Memory in the Age of AI Agents: A Survey — Yuyang Hu et al.**
[arXiv 页面](https://arxiv.org/abs/2512.13564)，初版 2025-12-15，本文使用 v2（2026-01-13）。

- 综述指出只按长/短期分类不足以覆盖当前 Agent Memory 研究，提出 Forms、Functions、Dynamics 三个分析视角。
- Forms 区分 token-level、parametric、latent；Functions 区分 factual、experiential、working 等用途；Dynamics 讨论记忆如何形成、演化和检索。
- 对正文的影响：拆开“表示载体”“记忆功能”“生命周期”，并指出向量是索引方式，不是与情景记忆并列的功能类型。
- 证据边界：这是研究领域的综述分类，不是行业标准或必须照搬的数据模型；本章将其作为教学组织框架，并把讨论范围限定为可检查的外部显式记忆及任务状态。

### 代表性原始研究

**Generative Agents: Interactive Simulacra of Human Behavior — Park et al.**
[arXiv 页面](https://arxiv.org/abs/2304.03442)，正文 HTML 版本 [v2](https://arxiv.org/html/2304.03442v2)。

- 系统维护观察/事件流，按相关性、新近度和重要性取回记忆；还会从观察中生成高层反思，再影响计划与行动。
- 正文将它作为“存储、检索、归纳共同影响行为”的机制示例，而不是生产用户记忆系统的安全证明。
- 证据边界：评估场景是 Smallville 社会模拟；行为连贯不等于真实用户事实准确、权限隔离或删除已完成。

**MemGPT: Towards LLMs as Operating Systems — Packer et al.**
[arXiv 页面](https://arxiv.org/abs/2310.08560)，正文 HTML 版本 [v2](https://arxiv.org/html/2310.08560v2)。

- 论文把有限上下文类比为受管理的主存。队列管理器可将部分消息移出上下文，利用已有递归摘要和移出的消息生成新摘要；被移出的消息保留在 recall storage 并可通过函数调用读取。
- 对正文的影响：说明 context paging 要解决“当前窗口装不下”的问题。
- 证据边界：recall storage 的可读性和上下文换页不代表长期个人记忆治理、授权或删除协议；正文不把两者混为一谈。

**Reflexion: Language Agents with Verbal Reinforcement Learning — Shinn et al.**
[arXiv 页面](https://arxiv.org/abs/2303.11366)，正文 HTML 版本 [v4](https://arxiv.org/html/2303.11366v4)。

- 系统根据环境反馈生成语言反思，并放入 episodic memory，供之后的试次参考；反馈可以来自环境或模型评估。
- ALFWorld 部分报告，结合 Reflexion 的 Agent 在作者设置中完成 134 项任务里的 130 项，使用简单启发式检测低效/失败并在连续试次间保留反思。
- 对正文的影响：用来说明“从失败中提炼可执行经验”与“原样保存轨迹”不同。
- 证据边界：130/134 是特定任务、模型、反馈和试次流程的结果；不代表任意自我总结都有效，也不证明开放环境下记忆一定可靠。

**LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory — Wu et al.**
[arXiv 页面](https://arxiv.org/abs/2410.10813)，v2 更新于 2025-03-04，论文列为 ICLR 2025；正文 HTML [v2](https://arxiv.org/html/2410.10813v2)。

- 基准包含 500 个问题，覆盖信息抽取、多 session 推理、知识更新、时间推理和未知信息拒答。论文把长期记忆系统拆成 indexing、retrieval、reading 三阶段。
- 论文摘要报告商业助手和长上下文模型在其持续交互设置下出现约 30% 的记忆准确率下降；同时研究 session 细分、fact-augmented key expansion、time-aware query expansion 等优化。
- 对正文的影响：评测必须覆盖时间变化和未知信息；检索也要关注值的颗粒度、索引线索和查询时间范围。
- 证据边界：30% 是该论文所测系统和数据设置的报告值，不能作为任意产品的固定衰减率；其优化结果不保证跨数据集和框架复现。

**Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory — Chhikara et al.**
[arXiv 页面](https://arxiv.org/abs/2504.19413)，v1 于 2025-04-28 提交，预印本。

- 方法部分描述从新消息对、近期对话和对话摘要中抽取候选事实，再检索语义相近的旧记录并决定新增/更新等操作。图版本用关系表示，并可把过时关系标成无效以保留时间变化。
- 论文在 LOCOMO 上报告与其他 memory/RAG/full-context 方案的比较。
- 对正文的影响：写入并非直接 append；候选提取之后还需要与已有记录匹配和解决更新。
- 证据边界：性能和成本数字由论文作者按其数据集、基线、LLM-as-a-Judge 等配置报告；本章不把它写成独立复现或普遍生产结论。

### 官方文档与工程团队材料

**LangGraph Memory 文档**
[官方文档](https://docs.langchain.com/oss/python/langgraph/add-memory)，读取日期 2026-10-26。

- 文档区分 thread-level persistence 的 short-term memory 与跨 session 的 user-specific/application-level long-term memory；示例展示 checkpointer、thread_id、namespace 和长期 store 操作。
- 正文用它说明框架如何划分任务状态与跨会话存储。
- 边界：文档示例的接口不证明调用方 ACL、来源可信度、删除传播或某个部署的实际策略已正确实现。

**OpenAI Agents SDK Sessions 文档**
[官方文档](https://openai.github.io/openai-agents-python/sessions/)，读取日期 2026-10-26。

- 文档覆盖中断后恢复同一 session、历史合并、限制检索历史、基本 memory 操作和内置 session 实现。
- 正文只用来说明“会话历史持久化/恢复”是框架提供的能力，不将它等同于用户级事实记忆。
- 边界：SDK 文档不替代应用层用户/组织隔离、项目事实版本、保留策略和产品删除语义；文档是动态页面，后续应按版本复核。

**LangChain 团队：Context Engineering**
[文章](https://www.langchain.com/blog/context-engineering-for-agents)，发布于 2025-07-02。

- 将上下文工程归纳成 write、select、compress、isolate，并讨论上下文窗口、scratchpad、长期记忆、观察输入和评估。
- 正文把它当作工程审阅框架，帮助读者检查每一步究竟写什么、选什么、压缩什么、隔离什么。
- 边界：这是工程团队对相关工作的综合文章，不是标准或实证定理。

**Databricks：Memory scaling for AI agents**
[文章](https://www.databricks.com/blog/memory-scaling-ai-agents)，发布于 2026-04-10。

- 讨论外部记忆和模型参数学习、长上下文的关系，以及 episodic/semantic 与个人/组织记忆；还强调权限、来源、保留和清除等部署问题。
- 正文用它作为规模化工程讨论的来源入口。
- 边界：文章包含公司自己的实验和主张；若采用其具体效果数字，必须连同任务和实验设置限定。本章没有把这些数字泛化成普遍结论。

## 研究到正文的转换

| 发现 | 正文如何体现 |
|---|---|
| 记忆类型分类混合了时间、功能和存储/索引形式 | 用载体、功能、生命周期三个轴重新组织 |
| “项目用了哪个版本”缺少环境、来源和有效时间 | 用本地、声明支持范围、生产和 CI 的 Python 场景贯穿章节 |
| 只说“检索不要只看 embedding”不够具体 | 展开任务解析、授权过滤、混合召回、时效/来源检查、冲突处理和上下文组装 |
| 长期交互会更新事实且会问未知问题 | 按 LongMemEval 五类能力构造评测表，并界定论文数据范围 |
| 总结/反思不自动变成可信事实 | 对 Generative Agents、Reflexion 的用途和边界分别说明 |
| 一条记录会派生到索引、缓存、摘要和备份 | 用 tombstone、在线阻断、幂等传播、派生摘要失效和验收路径解释删除；明确这是教学架构建议，产品承诺需逐项定义 |
| 任务 checkpoint 和跨会话事实记忆常被并称“短期/长期” | 对照 LangGraph 和 OpenAI Sessions 的文档接口阐明状态恢复与事实记忆边界 |
| 原章节的大型 toy demo 以很多组件展示伪系统，但没有足够证据说明真实语义 | 改成可审阅的端到端证据轨迹，不再把玩具实现包装成 production memory |

## 尚未解决的边界

- 记忆 schema、用户确认策略、敏感信息保留和删除承诺因产品而异；本文给的是教学建模和可验证的设计问题，不代表任何厂商的实现。
- 文中 tombstone 和副本清理是工程设计推理，未声称来自某个 memory benchmark 的结论，也没有宣称适用于所有法规或备份策略。
- 论文与框架页面可能继续更新；后续引用具体模型/API 行为时应按版本重核。
- 本样稿尚未得到维护者评阅；不据此宣布 book-17 或全书精修完成。
