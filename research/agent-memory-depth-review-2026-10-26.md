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
- 截至第一版样稿记录时，尚未得到维护者评阅；后续反馈及本轮复审见下节。本研究记录不代表 book-17 或全书精修完成。

## 按新版 WRITING_SPEC.md 复审与覆盖映射（2026-10-26）

本轮复审读取了旧版章节（`d9d60e2`）和当前样稿，按 `WRITING_SPEC.md` 的 `coverage_map` 要求逐项检查：核心话题补足机制；支持性或低价值内容压缩时仍保留概念入口；不以保留原代码/标题代替知识覆盖。

### 新增研究材料与对正文的影响

**Agent Memory 综述：Memory in the Age of AI Agents**
[论文](https://arxiv.org/abs/2512.13564)，并阅读对应的 arXiv HTML 正文（2026-10-26）。

- 综述明确指出仅按短期/长期划分不足以覆盖 Agent Memory，并从 Forms、Functions、Dynamics 讨论载体、用途和记忆变化过程。
- 对工作记忆，综述强调它是对当前任务上下文的主动选择、维护和改写；单纯的 context window 更接近被动缓冲区。
- 对情景/语义分类，情景保留具体经历的主体、时间和情境，语义保留不依赖单次发生场景的事实；这是一种分析分类，不要求对应不同数据库。
- 正文因此扩写了短期/会话/长期的生命周期边界、工作状态与持久化的区别，以及工作/情景/语义/程序性/偏好记忆的功能差异，并用同一个 Python 项目例子说明事件到事实的区别。
- 证据边界：综述整合了快速发展的研究与多种术语，不是行业数据模型标准。正文把它作为分类框架，不宣称所有 Agent 都实现相同记忆类型。

**LangGraph 当前 Memory 文档**
[官方文档](https://docs.langchain.com/oss/python/langgraph/add-memory)，读取日期 2026-10-26。

- 当前文档将 short-term memory 定义为 thread-level persistence，作为 Agent state 的一部分；long-term memory 用于跨会话的 user-specific 或 application-level data。
- 对正文的影响：明确短期不等于进程内易失、长期也不等于永久；用框架定义支撑 checkpoint 与跨 session store 的例子，同时标注这是框架术语口径。
- 证据边界：官方 API 示例不证明任意应用已经有正确 ACL、版本冲突、来源追踪或删除传播。

**MemoryAgentBench**
[arXiv 论文](https://arxiv.org/abs/2507.05257)，并核对 [ICLR 2026 会议论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fd1eff9dd295df50a41f2521942fa31d-Abstract-Conference.html)。

- 论文将记忆能力拆成 accurate retrieval、test-time learning、long-range understanding 和 selective forgetting；其 selective forgetting 指遇到新信息时处理过时或矛盾事实，避免旧值继续主导回答。
- 对正文的影响：把冲突更新从普通 retrieval 中单独说明，并明确 benchmark 的 selective forgetting 不等同于用户发起删除后跨副本清除。
- 证据边界：论文的能力定义和结果受其数据集、构造与评估配置约束；不将术语写成普遍产品承诺。

**EverMemBench / Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues**
[arXiv PDF](https://arxiv.org/pdf/2602.01313)，本轮读取正文方法、统计和结果部分。

- 论文描述 5 个协作项目、51,023 轮对话、4,225,555 tokens、2,400 组 QA；任务覆盖细粒度回忆、memory awareness 和 profile understanding，并包含多群组归因、跨话题、随时间变化的信息。
- 详细结果区分 full-context 与 oracle-evidence 设置：Gemini-3-Flash 的 multi-hop 得分从 26.51% 提升到 88.37%；GPT-4.1-mini 从 2.41% 提升到 97.99%。这些是论文具体 benchmark/model/setting 的值，正文将其用作证据归属和跨群组整合的失败案例，不泛化到 Agent 总体能力。
- 来源核对注意：论文摘要将“26%”描述为 oracle 下结果，但正文 Table 5 和实验分析把 26.51% 对应到 Gemini-3-Flash full-context，并报告其 oracle 为 88.37%。正文采用详细表格和实验分析的分模型口径，并在段落中注明摘要与正文口径差异。
- 证据边界：这是公开预印本中的合成企业协作对话 benchmark。它证明此类评测题可暴露归属、时间和检索问题，不证明其场景覆盖所有真实企业会话。

### `coverage_map`：旧稿知识点到现稿去向

| 旧稿知识点/章节 | 新版重要性与深度 | 当前去向与处理 |
|---|---|---|
| 5.1 为什么 Agent 需要 Memory | 核心背景 | 5.1 故障案例、5.17 小结；保留连续性价值及记错风险 |
| 5.2 Context、State、Memory、RAG | 核心边界 | 5.3；区分调用输入、任务恢复状态、跨任务信息和外部知识检索 |
| 5.3 Memory 分层结构 | 核心分类 | 5.2 增加时间跨度/生命周期视角；5.5 展开 task checkpoint 与跨任务记忆 |
| 5.4 记忆表示与检索公式/指标速查 | 结构字段与过滤为核心；无校准的统一打分公式低价值 | schema 在 5.4；相关性、来源、时效和授权在 5.6；分母明确的指标在 5.12。旧 `exp(-λΔt)` 统一衰减和合成 `Q_mem` 不作为通用公式保留：时间新旧不等于事实有效性，指标也不应被压成单一分数；recency/有效时间含义保留在 5.4、5.6、5.8 |
| 5.5 短期记忆 | 核心 | 5.2 解释短期/会话工作范围；5.5 增加工具副作用前后 checkpoint、崩溃窗口、幂等键和恢复权限复核 |
| 5.6 长期记忆 | 核心 | 5.2 区分跨 session 复用与“永久保存”；5.5 说明筛选写入与生命周期 |
| 5.7 向量记忆 | 重要支持知识 | 5.2 说明向量是索引表示而非功能类型；5.6 展开候选召回、结构过滤、排序和证据校验 |
| 5.8 情景记忆 | 核心 | 5.2 用主体、时间、环境、动作、结果和来源定义事件记录，解释其与当前事实的区别 |
| 5.9 语义记忆 | 核心 | 5.2/5.4/5.8 说明事实、来源、scope、valid time 和版本，不把推断当永久知识 |
| 5.10 用户偏好记忆 | 核心常见用例 | 5.2 列为 profile/preference 功能；5.5 区分明确且稳定偏好与一次性请求，按范围写入 |
| 5.11 写入策略 | 核心 | 5.5；保留候选来源、筛选、原子化、去重、冲突比对、增改并存/待确认/拒绝及模型推断边界 |
| 5.12 更新、冲突和遗忘 | 核心难点 | 5.8 增加主张键与冲突决策表；拆分事实更新、到期/复核、压缩和用户删除；5.9 深化跨副本删除轨迹 |
| 5.13 检索策略 | 核心 | 5.6；权限先行、混合候选、时效/来源/冲突检查及上下文组装 |
| 5.14 压缩和反思 | 核心机制/支持性案例 | 5.7 解释摘要血缘和删改失效；5.10 解释反思只产生候选、需证据和评估 |
| 5.15 Memory 与 RAG | 核心邻接边界 | 5.3；允许共享基础设施，但区分证据来源、写入主体、更新/删除责任 |
| 5.16 隐私安全、5.17 污染 | 核心治理主题 | 5.4 scope/来源/敏感等级，5.5 不可信输入写入策略，5.6 权限硬过滤，5.13 链路安全与注入污染 |
| 5.18 Memory 评估 | 核心 | 5.11/5.12；LongMemEval、MemoryAgentBench、EverMemBench 互补，新增覆盖/来源归属/过期/更新/冲突处置/不安全候选写入/选择性遗忘等分母明确指标 |
| 5.19 最小可运行 audit demo | 教学代码形式不保留；保留其知识验证目的 | 旧 demo 的虚构阈值和综合分数容易被读成通用生产契约；其检索/越权/过期/冲突/不安全写入/删除后残留概念，分别由 5.6、5.8、5.9、5.12、5.13 的轨迹、测试题和指标承接。正文用可核对的 YAML、状态流程、证据案例和分母指标说明机制，不声称运行生产 memory。可在后续需要时针对单一机制新增小型可运行 demo。 |
| 5.20 常见失败模式 | 根因分散且跨生命周期 | 5.1、5.5–5.13 分别落到错误写入、过期/冲突、越权、相似误召回、摘要损失、删除残留和评测盲区；不重复保留一张缺少诊断路径的长清单 |
| 5.21 命名空间 | 核心安全机制 | 5.4 将 `scope` 明确解释为地址，区分用户/项目/任务/组织空间；5.6 在候选生成阶段硬过滤 |
| 5.22 删除传播 | 核心生命周期机制 | 5.9；tombstone、不可读优先、幂等传播、摘要/缓存/备份恢复和验收路径 |
| 5.23 长期任务恢复 | 核心 Agent 能力 | 5.5；区分 pending action、工具已完成但结果未落盘、结果已保存三类恢复窗口；说明 exactly-once、幂等/对账和授权重验边界 |
| 5.24 Python 版本 worked example | 核心教学例子 | 5.1、5.4、5.8、5.15；贯穿 scope/时间/来源/检索/冲突和回答生成 |
| 5.25 练习 | 支持性教学 | 5.16；保留 scope、授权、时间、来源、删除和未知问题检查，移除依赖原 demo 内部函数的代码改造题 |
| 5.26 小结、5.27 延伸资料 | 必要收束与入口 | 5.17/5.18；补入 ICLR 2026 MemoryAgentBench 和 2026 EverMemBench 预印本，并标明证据边界 |

### 审查结论与剩余边界

- 当前章节维持一个入口文件；本轮核心概念和生命周期轨迹仍能在一章内按顺序阅读，书册目录入口有效，暂不拆分文件。
- 短期/长期、情景/语义、偏好、冲突、namespace、任务恢复及过期/删除之间的读者入口都已保留；不再以“旧版每个标题都需独立长篇”作为覆盖标准。
- 旧 toy demo 不再以代码形式存在，其所服务的审计问题逐项映射到正文与评估维度；它的固定分数阈值不当作行业标准沿用。
- EverMemBench 仍是预印本；其 benchmark 数字只用于说明这套评测上的现象，尚未在本项目中独立复现。
- 删除传播部分仍是明确标注的教学架构设计建议，不是所有产品的已验证通用实现。
