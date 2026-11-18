# 长上下文 Harness 与 Agent Memory：来源证据

检索与复核日期：长上下文 Harness 与 RAG 来源于 2026-10-29 初检；Agent 长期记忆相关论文和 OpenAI Agents SDK Sessions 文档于 2026-11-16 复核。本记录仅存来源支持的技术事实、版本、阅读范围与证据边界；章节审阅决定、去向和修改状态见各书册对应的 `progress/` 记录。

## 长上下文 Harness 评测

- **来源与版本**：[Pham et al., “LongHarness Bench: Stress-Testing Language Model Harnesses for Long-Context Reasoning,” arXiv:2609.38137v1](https://arxiv.org/abs/2609.38137)，提交于 2026-09-29。
- **阅读范围**：arXiv HTML 摘要、任务构造 §3.1、主要结果 §§4.2–4.5。
- **来源支持的事实**：论文提出四项 long-context harness 评测任务，覆盖多条件检索、跨文档记忆识别、程序执行轨迹及等价程序对搜索；其中 Constraint Solving Search 每个样本约含 120K tokens、160 人的材料和 3–5 个条件。论文比较多个模型与四种 harness，报告同一模型在不同 harness/任务下准确率和费用差异很大、没有一个 harness 在四项任务上都领先；GPT-5.6-sol + mini-swe-agent 的四任务 macro accuracy 为 68%。在 Outlier Memo Detection 上，作者报告 RLM 与 mini-swe-agent 得到相近准确率，但 RLM 每实例估算费用高 12.4 倍。
- **证据等级与边界**：source-backed（预印本的自建 benchmark、作者模型/harness 配置和成本估算）。上述分数、成本和任务难度只适用于论文数据、模型版本、API 价格及 harness 实现；不能用来给生产 Agent 或其他 harness 排名。该工作支持把“同一模型搭配不同长上下文处理器的质量—成本变化”作为独立评测对象。

## RAG 路线：自适应、纠错和分层检索

- **Self-RAG**：[Asai et al., ICLR 2024 paper / arXiv:2310.11511](https://arxiv.org/abs/2310.11511)。阅读 arXiv 与 ICLR 官方摘要。论文用 special reflection tokens 训练模型按需决定检索，并对检索段落和生成结果作反思式判定；结果受论文训练数据、模型和任务限制。反思 token 是模型训练和推理方案，不等于外部独立验证器或通用的事实正确保证。
- **Corrective RAG (CRAG)**：[Yan et al., ICLR 2024 paper / arXiv:2401.15884](https://arxiv.org/abs/2401.15884)。阅读 arXiv 摘要。论文以 retrieval evaluator 估计检索文档质量，并按置信度触发不同检索动作；方案还包括对检索文本做分解和重组，并把大规模 web search 用作静态语料的补充。证据范围限于论文设计和四个数据集实验；真实系统还须校准 evaluator、保证搜索结果的权限/来源质量并测量额外调用成本。
- **RAPTOR**：[Sarthi et al., ICLR 2024 / arXiv:2401.18059](https://arxiv.org/abs/2401.18059)。阅读 arXiv 与 ICLR 官方摘要。论文递归地 embedding、聚类并总结 chunk，形成从原始叶节点到高层摘要的树，推理时可按不同抽象层次检索；作者报告指定多步问答任务上的结果。方法把成本部分移到索引和摘要构建，并可能在上层摘要中压缩掉精确条件；效果需按语料更新频率、精确 span 需求和摘要保真度评估。
- **GraphRAG**：[Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” arXiv:2404.16130](https://arxiv.org/abs/2404.16130)。阅读 arXiv 摘要。方案从文档抽取实体关系图、构建 community summaries；global query 时先逐社区生成 partial answer，再汇总为最终回答。作者在约百万 token 的数据集级 global sensemaking 任务中报告相对 naive RAG 的收益。结论限于论文语料、问题和实现；图谱/社区摘要有索引构建、刷新和来源回溯成本，不能替代局部精确检索。

上述方法分别改变检索决策、低质量召回的恢复、文档层级表示和全局语料摘要路线；机制不可互换，论文实验也不能合并成跨系统的性能排名。

## Agent 长期记忆的写入、使用和安全（来源复核：2026-11-16）

- **AdaMem**：[Chen et al., EMNLP 2026 Industry Track](https://aclanthology.org/2026.emnlp-industry.189/)。阅读 ACL Anthology 论文记录和摘要。AdaMem 将记忆写入建模为 preference-conditioned write control：每种交互情境有自然语言 memory policy，并通过周期反馈更新；AdaMem-Bench 使用 6 个并行 persona 和 5 段 10 周故事。摘要报告，在两种抽取模型与两种反馈模式下，其平均 QA accuracy 从 Mem0 的 80.0% 到 84.35%，persistent memory 减少 9.27%；作者同时报告模型仍不能稳定执行选择性写入政策。数值绑定论文基准、baseline 和配置，不代表生产记忆策略的普遍收益。
- **Memora**：[Uddin et al., Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1337/)。阅读 ACL Anthology 论文记录和摘要。基准覆盖数周至数月的对话，评 remembering、reasoning、recommending，并提出 Forgetting-Aware Memory Accuracy 以惩罚对已过期或失效记忆的依赖；作者测试 4 个 LLM 与 6 个 memory agents，报告常见过期记忆复用和新旧信息协调问题。该摘要不提供跨产品的普遍失效率。
- **Mem2ActBench**：[Shen et al., ACL 2026](https://aclanthology.org/2026.acl-long.370/)。阅读 ACL Anthology 论文记录和摘要。论文评估长期间记忆能否用于具体工具动作和参数补全；其数据构造包含 2,029 条对话 session、平均 12 个 user/assistant/tool turns，并从 memory chains 生成 400 个 tool-use tasks，人工评估中 91.3% 被标为强 memory-dependence。结果绑定该合成/整理语料、任务和系统配置。
- **OP-Bench**：[Hu et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1168/)。阅读 ACL Anthology 论文记录和摘要。论文把 over-personalization 分成 Irrelevance、Repetition、Sycophancy，构造 1,700 个 verified instances，并报告所测模型和 memory-augmentation 方法中常出现无关或过度使用用户记忆。该 benchmark 评估对话个性化适用性，不直接测试所有产品中的隐私授权、删除或动作安全。
- **PS-Bench**：[Guo et al., ACL 2026](https://aclanthology.org/2026.acl-long.1260/)，arXiv:2601.17887。阅读 ACL Anthology 论文记录和摘要。论文提出 intent legitimation：无害个人记忆可能使模型对有害请求推断出良性意图；在多个 memory-augmented frameworks 与 base models 上，摘要报告相对 stateless baselines 的 attack success rate 增幅为 15.8%–243.7%。这是论文测试配置的相对变化范围，不是百分点、通用风险率或所有记忆系统的因果效应。
- **MemoryCD**：[Zhang et al., Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1804/)。阅读 ACL Anthology 论文记录和摘要。其 benchmark 从 Amazon Review 相关行为数据构造跨年、多领域 personalization 任务，摘要列出 14 个 base models、6 种 memory baselines、4 类任务和 12 个领域。该摘要支持其数据与任务设计，不足以证明 benchmark 可代表真实用户日常对话或其他隐私/授权制度。

这些来源分别强调写入策略、过期记忆、行动参数使用、过度个性化、个性化安全以及跨领域历史。数据集和结果不可合并为一个“记忆能力”指标；读摘要所得结论限于上述摘要明确披露的信息。

## 长期记忆基准：召回、遗忘与行动（来源复核：2026-11-16）

- **LongMemEval**：[Wu et al., arXiv:2410.10813v2，ICLR 2025](https://arxiv.org/abs/2410.10813)。阅读 arXiv HTML 摘要、统一设计 §4、时间查询实验 §5.4。论文用 500 道问题评估信息抽取、多 session 推理、时间推理、知识更新和拒答；摘要报告，在持续交互记忆设置下，商业助手和长上下文 LLM 出现约 30% 的准确率下降。作者将系统分成 indexing、retrieval、reading 三阶段，并报告会话拆分、事实扩展索引和时间感知查询扩展等改法。§5.4 的时间扩展收益取决于负责解析时间范围的 query LLM；结果限于 LongMemEval 的问题、模型和检索配置，不是所有产品的固定下降或增益。
- **MemoryAgentBench**：[Hu et al., arXiv:2507.05257v4，ICLR 2026](https://arxiv.org/abs/2507.05257)。阅读当前 v4 摘要、Selective Forgetting 定义 §B.4 和任务提示说明。论文把记忆能力拆为 accurate retrieval、test-time learning、long-range understanding 和 selective forgetting；后者指在新证据与旧知识矛盾时检测并修订、覆盖或移除旧信息。其 FactConsolidation 任务显式要求按编号优先考虑较新的事实。这个操作定义用于基准中的矛盾更新任务，不等同产品收到用户请求后清理索引、摘要、缓存和备份的删除义务；具体 benchmark 结果也受任务提示和样本构造影响。
- **EverMemBench**：[Hu et al., arXiv:2602.01313v3](https://arxiv.org/html/2602.01313v3)，版本修订于 2026-03-11。阅读数据统计 §3.4、主要结果表 4—5 及误差分析 §§4.2—4.3。该预印本构造 5 个项目/15 个子项目、170 名员工、51,023 轮对话、4,225,555 tokens 和 2,400 组问答。表 4 的 Gemini-3-Flash full-context multi-hop 准确率为 26.51%，提供 gold evidence 的 oracle 设置为 88.37%；GPT-4.1-mini 对应为 2.41% 和 97.99%。摘要把“即使给 oracle evidence 也仅约 26%”作为概括，与详细分模型表及 §4.2 对 Gemini 的 oracle 结果不一致；正文引用采用分模型表值，保留此来源内部差异。所有分数均限于其合成协作对话、模型和评测设置，不能外推为通用产品水平。

## OpenAI Agents SDK Sessions 的会话历史信任边界

- **来源与版本**：[OpenAI Agents SDK Sessions：SQLite storage trust boundary](https://openai.github.io/openai-agents-python/sessions/#sqlite-storage-trust-boundary)，动态文档，核阅日期 2026-11-16。
- **阅读范围**：Sessions 概览、SQLite storage trust boundary、EncryptedSession 说明。
- **来源支持的事实**：Agents SDK 的 session 用于跨运行保存和恢复对话历史。SQLiteSession 文档明确说明，session ID 只是选择会话历史的键，不认证用户，也不授权访问该历史；应用负责授权每个 session，并保护数据库文件、底层存储和备份。文档还指出 EncryptedSession 提供加密，但不验证历史条目是否完整、顺序是否正确；它接受未经加密的条目，也不能替代存储完整性控制。
- **证据等级与边界**：官方文档快照，source-backed；授权和完整性描述针对文档中的 SQLite trust boundary，不能外推到每种 session backend。文档与 SDK 会更新；涉及具体版本行为时应再核对固定 package revision 及所用后端实现。

## Agentic RAG：检索决策、工具接口与评测（来源复核：2026-11-16）

- **FLARE**：[Jiang et al., “Active Retrieval Augmented Generation,” arXiv:2305.06983](https://arxiv.org/abs/2305.06983)。阅读 arXiv 摘要。Forward-Looking Active Retrieval Augmented Generation (FLARE) 预测即将生成的句子，把预测内容用作检索查询，并在出现低置信 token 时结合新检索的材料重新生成句子；作者在四个长篇知识密集生成任务/数据集上比较 FLARE 与基线。机制和结果限于论文方法、数据、模型及实验设置，不代表任一产品会默认执行置信触发或保证生成事实正确。
- **IRCoT**：[Trivedi et al., “Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions,” arXiv:2212.10509](https://arxiv.org/abs/2212.10509)。阅读 arXiv 摘要。IRCoT 在逐步推理过程中交错检索：先前已推导或检索到的信息会影响下一步该查什么，检索结果再进入后续推理；作者报告了 HotpotQA、2WikiMultiHopQA、MuSiQue 和 IIRC 上的实验。证据适用于论文的多跳问答设置，不等价于对任意开放域 Agent 的生产效果。
- **A-RAG**：[Du et al., “A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces,” arXiv:2602.03442v1](https://arxiv.org/abs/2602.03442)。该版本提交于 2026-02-03；阅读摘要、§3.2 的检索接口与实验摘要。论文向模型提供 `keyword_search`、`semantic_search` 和 `chunk_read` 三类接口，以不同粒度访问语料；作者在多个开放域 QA benchmark 上报告其配置的答案表现和检索 token 使用。结果绑定论文基准、模型与实现；它支持“检索工具粒度可成为模型决策接口”的实例，不证明分层工具在所有语料或权限制度下都优于其他设计。
- **RAGCap-Bench**：[Lin et al., AACL-IJCNLP 2026, ACL Anthology 2026.aacl-long.245](https://aclanthology.org/2026.aacl-long.245/)。阅读 ACL Anthology 元数据、论文摘要及 §§2—3.2。论文以开源 deep-research 问题和 Agentic RAG 轨迹构建 255 道筛选后的选择题，覆盖 Planning、Evidence Extraction、Grounded Reasoning、Noise Robustness 四类中间能力，并由专家参与答案标注。它为组件级诊断提供一个 benchmark；这些题目和实验不能代替部署系统的检索端到端质量、权限、成本或安全评估。
- **Q2D-Web**：[Schall et al., “Q2D-Web: A Large-Scale Benchmark for Retrieval in Agentic RAG Systems,” arXiv:2609.08887v2](https://arxiv.org/abs/2609.08887v2)。v2 日期为 2026-09-28；阅读摘要、§3 的查询/语料/相关性标注构造及 §4 的第一阶段检索实验。论文描述约 1.9 亿网页文档、约 7 万条十种语言的 Agent 改写查询，查询采样自九个月的去 PII 生产流量，并设置 Agent citations、production rankings、Combined+LLM-judged 三种相关性判断；比较 13 个词法、dense 和 late-interaction retrievers。各模型仅读取每篇文档前 512 个 token（平均文档约 3.3K token），大规模 benchmark 的结论因此不覆盖全文长文档读取；该工作针对第一阶段检索，不测完整 RAG 的引用推理或应用权限。语料、查询和 judgments 不公开，复现实验受限。

这些来源分别观察检索触发、多跳步骤依赖、工具粒度、Agent 中间能力和第一阶段召回；各自的任务对象和指标不同，数字不能合并成统一的 Agentic RAG 排名。

## 2026-11-18 上下文选择、会话输入与 compaction API

- **Anthropic Messages API 按需压缩**：[官方文档](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)，2026-11-18 访问；阅读请求/回放流程、beta header、能力查询和平台可用性部分。按需请求与普通会话轮次分开，返回一个带摘要文本和签名的 `compaction` block，`stop_reason` 为 `compaction`；后续请求需原样回传签名 block，并用它替换被摘要的消息。文档要求为摘要请求及回传 block 的请求携带 `compact-2026-09-04` beta header，并建议通过 model capability 查询具体模型支持。文档列出 Claude API、Amazon Bedrock、Claude Platform on AWS、Google Cloud Anthropic-managed 和 Microsoft Foundry；逐平台模型列表并不相同，不能只按 feature 名称推断某模型/平台合同。被摘要的图片、文档、上传容器和抓取 URL 内容不会由摘要 block 保留。
- **Anthropic Messages API 阈值压缩**：[官方文档](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)，2026-11-18 访问；阅读 `context_management.edits`、trigger 参数、block 回放和 usage 部分。Messages API 使用 `compact_20260112` edit；当前文档给出的默认 trigger 为 150,000 input tokens，最小配置值为 50,000，`pause_after_compaction` 默认 false。API 在普通请求中创建 compaction block，后续请求可保留旧消息列表并由 API 忽略 block 前内容，或由调用方自行裁掉前缀。该功能为 beta，具体 trigger、支持模型、云平台和计费应按当前 API 文档/部署核对。
- **Anthropic 后台压缩和 thinking block 回放**：[后台压缩文档](https://platform.claude.com/docs/en/build-with-claude/compaction-background)与[保留 thinking 文档](https://platform.claude.com/docs/en/build-with-claude/compaction-thinking-blocks)，2026-11-18 访问。后台摘要请求可与完整历史上的后续轮次并行；回包后需只替换发送摘要的历史前缀，保留期间追加的新轮次，不能在等待期间改写旧消息或同时启动第二次压缩。保留在压缩点之后的 thinking block 是否有效，取决于模型能力、消息紧邻与原样回放、`system`/`tools` 前缀等条件；各条件和平台绑定相应 Messages API beta 语义。
- **OpenAI Agents SDK Session 输入选择**：[官方 Sessions 文档](https://openai.github.io/openai-agents-python/sessions/)，2026-11-18 访问；阅读 session history 与 new input 合并、`session_input_callback`、`SessionSettings` 和 SQLite storage trust boundary。默认输入由 session history 与新 turn input 合并；callback 可定制本轮送入模型的列表，而 SDK 仍只把新 turn 内容作为新输入持久化。`SessionSettings(limit=N)` 可限制每次运行读取的最近 session item 数。SQLiteSession 的 session ID 只选择历史记录，不认证用户或授权访问；应用负责逐 session 授权及保护数据库文件、存储和备份。结论限于该 SDK/后端文档，不代表所有 session 实现。
- **OpenAI 托管 Agents API 状态所有权**：[官方 Agents API 概览](https://developers.openai.com/api/docs/guides/agents-api/overview)，2026-11-18 访问；阅读管理型 harness 责任说明。当前公开合同将 sessions、orchestration、context compaction 和 recovery 归于 OpenAI 管理的 Agents API；这是托管产品边界，不等于 Agents SDK、Responses API 或其他 provider 的默认状态所有权。
- **外部内容与间接提示注入**：[OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)，2026-11-18 访问；阅读 indirect injection、mitigation 和 threat scenario 部分。文件、网页等外部内容可携带影响模型行为的指令；指南建议分离并清楚标记不可信内容，同时采用最小权限、高风险人工审批和攻击测试。该指南明确不提供可保证消除 prompt injection 的单一方法；分隔标签属于风险缓解，不是执行器授权或隔离边界。
