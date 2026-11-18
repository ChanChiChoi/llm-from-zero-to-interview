# Context Folding 与长任务压缩：来源证据

检索与复核日期：2026-11-18。本记录仅保存联网发现的技术事实、组织样本、来源和证据边界；章节审阅决定、正文取舍和修改历史记入 `progress/`。

## 上下文压缩的工程建议与 API 合同

### Anthropic：上下文工程建议

- **来源与版本：** Anthropic Engineering, [“Effective context engineering for AI agents”](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)，2025-09-29 发布；2026-11-18 阅读。
- **阅读范围：** context engineering 定义、有限上下文、compaction、structured note-taking 小节。
- **来源支持的内容：** Anthropic 将 context engineering 描述为每次推理时对系统指令、工具、外部数据和消息历史等输入材料进行选择与维护；其实践文章建议对长轨迹做 compaction，保留架构决策、未解决问题和实现细节，去掉重复工具输出；先提高召回、再调整精确度，并用复杂 Agent 轨迹调优摘要提示。文章也将外部持久笔记与当前模型上下文区分。
- **证据边界：** 这是供应商工程实践文章，提供组织建议和案例，不是跨模型对照实验，也不证明某一种压缩策略普遍最优。

### Anthropic Messages API：compaction block

- **来源与版本：** Anthropic 官方 [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) 与 [Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)，2026-11-18 访问。
- **阅读范围：** block 创建与回放、beta header、平台和模型能力、阈值配置、异步压缩注意事项。
- **来源支持的内容：** 按需 compaction 是独立请求，返回含文本摘要与签名的 `compaction` block；后续请求按文档合同原样回传该 block，并用它代替被摘要消息。阈值压缩在普通请求中生成 block。两种流程均有 beta、模型与托管平台适用范围；当前文档列有不同的 API header/配置形态，不能按通用文本摘要协议处理。被压缩的图像、文档、上传内容和 URL 抓取内容不会因此自动变成可恢复原件。
- **证据边界：** 动态产品文档，结论绑定访问日期、API、beta、模型与平台；实际集成须再核对部署合同。该 block 的语义不能推广为其他供应商的格式。

### OpenAI Agents SDK：session 输入选择不等于历史压缩

- **来源与版本：** OpenAI Agents SDK 官方 [Sessions](https://openai.github.io/openai-agents-python/sessions/) 和 [Context management](https://openai.github.io/openai-agents-python/context/)，2026-11-18 访问。
- **阅读范围：** history 与 new input 的合并、`session_input_callback`、`SessionSettings`、session 持久化说明。
- **来源支持的内容：** 默认情况下，SDK 将取回的 session history 与当前新输入合并后送给模型。`session_input_callback` 可在调用模型前筛选或重排本轮历史；文档说明 SDK 仍只持久化新 turn 的项目，不把筛选后的旧历史重新作为新 turn 写入。`SessionSettings(limit=N)` 取最近 N 项，是按项目数限制取回历史的方式，不是自动生成语义摘要。
- **证据边界：** 以上是当前 Agents SDK 文档的实现合同；不能外推到不同 session backend 或托管 Agents API。使用时应固定 SDK 版本并检查所用 session 实现。

## 长任务压缩的评测研究

### Benchmarks Are Surprisingly Compaction-Friendly

- **来源与版本：** Salle et al., [arXiv:2610.31778v1](https://arxiv.org/html/2610.31778)，2026-10-22 提交。
- **阅读范围：** 摘要、实验设置 §§3.1—3.4、结果 §§4.1—4.2、limitations、附录中约束抽取与判断说明。
- **来源支持的内容：** 研究固定 OpenCode scaffold，在 PaperBench、ProgramBench、PostTrainBench 上比较四个模型、五种压缩策略；每个任务/配置运行两次。论文报告，在这些设置下没有压缩策略显著优于默认 self-summarization，且结果受模型和基准影响。作者另在 ProgramBench 观察到反作弊规则会随连续压缩流失；在 44 个含 compaction 的 SWE-chat 会话中抽取并标注 269 条用户约束，整体约 81% 被保留，过程指令与范围排除丢失率明显高于功能要求（分别为 35.9%、36.8% 和 6.2%）。作者据此指出，以固定目标、可从文件恢复进度的 benchmark 可能低估流程约束丢失。
- **证据边界：** 单篇 v1 预印本；实验固定一个 scaffold、每格只有两次运行，基准任务有限；SWE-chat 的约束抽取/分类由模型辅助，样本为 44 个会话。结果提示要分别测任务通过与过程/范围约束，不构成所有生产 Agent 的通用错误率或压缩策略排名。作者还明确指出，不同 context 长度、模型与 scaffold 会改变压缩次数和观察结果。

### The Compaction Cliff in Long-Running AI Agent Memory

- **来源与版本：** Zerhoudi et al., [arXiv:2608.22752v1](https://arxiv.org/html/2608.22752)，作者标注为 CIKM ’26 论文版本，会议日期 2026-11-07—11；2026-11-18 阅读。
- **阅读范围：** 摘要、类型化知识模型、compaction 实验设置与结果 §§3.1、4.2—4.3、讨论与评估限制。
- **来源支持的内容：** 论文从 AgentArtifactCorpus 的公开配置材料中抽样，比较把不同信息一视同仁处理的结构/模型压缩器与类型化保留方案。对 20 个配置、50% 压缩、作者的约束标注和测量方法，最佳 type-blind 策略单轮约束召回为 0.53；五轮 `/compact` 后从 0.53 降到 0.10。作者的 TypeCompact 将分类出的约束与程序性规则分开保留，并由 verifier 检查；它的可行预算内保证依赖分类器召回，预算装不下被固定保留项时应停止或转入其他处理，不是无条件保证。正文还报告分类标注跨领域迁移、人工分歧和预算不足时的限制。
- **证据边界：** 召回值依赖作者构造的 AgentArtifactCorpus、配置抽样、规则类型定义、分类器、约束匹配口径和压缩预算。论文把公开仓库配置当作 Agent 知识样本，不能据此推断闭源企业语料或所有真实会话；分类器漏标的规则不受其保留保证覆盖。

### AutoCompact

- **来源与版本：** Zhang et al., [“AutoCompact: Learning When to Compact Context in Long-Horizon Coding Agents,” arXiv:2610.02163v1](https://arxiv.org/html/2610.02163)，2026-10-01 提交。
- **阅读范围：** 摘要、方法、实验设置、主要结果与限制。
- **来源支持的内容：** 该方法让 coding agent 学习何时压缩、摘要应保存什么、以及压缩后怎样继续；作者用 judge 修正后的轨迹做 SFT，再以最终任务结果做 RL。实验固定 Qwen3-Coder-30B-A3B-Instruct 和一个终端式 REPL scaffold，在 SWE-bench Verified、SWE-PolyBench Verified 报告分别 39.6% 和 24.5% 的 pass rate；相对同一研究中的 base 设置高 9.2 和 5.0 个百分点。定量结果平均三次运行；评估还比较 256K 与 16K 等设置。
- **证据边界：** 单篇预印本，模型、训练数据、judge、REPL scaffold、任务集、成本估计和评估设置固定；作者明确将其称为 model-harness co-design，并指出研究仅使用单一 scaffold。结果支持“压缩时机、摘要内容和后续行为可作为联动的 Agent 决策研究”这一问题，不证明生产 Agent 需要采用该训练方法或可迁移到任意模型。

### DeepSeek-V3.2 Search Agent 上下文管理

- **来源与版本：** DeepSeek-AI, [“DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models,” arXiv:2512.02556v1](https://arxiv.org/html/2512.02556)，提交于 2025-12-02。
- **阅读范围：** §4.4、与 Search Agent 评估相关的 §4.1/§4.3 和 Table 2。
- **来源支持的内容：** 在使用商业搜索 API 的 Search Agent / BrowseComp 设置中，报告比较 `Summary`（总结溢出的轨迹后重新 rollout）、`Discard-75%`、`Discard-all` 与 `Parallel-fewest-step`；论文在工具轨迹超过上下文窗口 80% 时触发管理策略。Table 2 报告无管理 Pass@1 为 51.4、带管理为 67.6*；§4.4 报告 Summary 平均扩展到 364 steps，并称 performance improvement “up to 60.2”。作者指出不同策略的效率不同，需要计入实际计算成本。
- **证据边界：** 所有数值绑定该 Search Agent、128K context、商业搜索 API、BrowseComp、作者的计算预算和 harness；80% 仅是此实验的触发配置。图 6 纵轴是 BrowseComp 分数，`60.2` 不应改写成 `+60.2%`。其摘要、丢弃与并行策略不能据此排出其他任务或产品的普适优劣。相关章节还记录了图轴复核与具体判断。

## 组织样本

- **CUHK-Shenzhen NLP 课程：** [“LLM Agents and Harness Engineering”, Lecture 07](https://nlp-course-cuhksz.github.io/materials/2026_fall_slides/Lecture07_LLM_Agents_and_Harness_Engineering.pdf)，2026 Fall 课程讲义，2026-11-18 阅读。第 72 页用“窗口接近满载—折叠后新窗口—清理可重读工具结果”的前后图引入压缩，随后衔接 Anthropic 的 context engineering 与 API 示例。这里只借鉴从可观察上下文变化切入并逐步连接实现合同的章节组织，不把讲义当成学习效果证据。
- **University of Mannheim IE685：** [“LLMs and Agents: Context Engineering”](https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Teaching/Large_Language_Models_and_Agents/FSS2026/IE685_LA_06_ContextEngineering.pdf)，版本日期 2026-03-16。讲义先讲动机和窗口成本，再将架构问题拆成 write/select/compress/isolate，之后展开各类技术。它展示了一种将“信息进出上下文”放进整体 Agent 架构的组织方式；不证明该顺序对所有目标读者最优。

## 尚未验证的边界

- 未在真实 Agent、Provider API 或 SDK 上运行压缩/恢复 probe；记录的是报告与官方文档，而非本地复现。
- 多篇 2026 年论文仍有预印本或单一 harness/任务集边界；未做跨团队复现，也不将不同论文的分母、指标或策略合并。
- 论文判定“保留”不等于真实动作安全；不同任务的独立 verifier、权限强制、artifact/trace 保留和外部副作用核对仍需单独设计。
