# 大模型资料更新计划 v2

更新时间：2026-09-16

## 目标

以 2025 年以来新发布或快速进入主流评测的新模型为锚点，更新本项目的模型谱系、技术知识、论文资料和面试训练内容。新文件是当前执行计划的唯一入口；旧版 `plan.md` 保留作历史参考。自 2026-09-14 起，新模型候选的发现入口严格收敛为 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜。

## 当前重点厂商范围

主动关注范围仅包括：OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。其他厂商及其新模型暂不主动检索、核验或扩写；已有历史记录保留，但不进入下一轮更新队列，除非用户明确通知需要关注。

## 锚点研究原则

后续以两个排行榜中近期出现、排名靠前或在 Agent/推理/长上下文/多模态方向具有代表性的模型作为热点锚点，不追求把八家厂商的所有历史版本和长尾模型列全。对每个锚点，先归并 `effort`、fallback、provider 和 Agent harness 配置，再沿模型的官方博客、技术文档、模型卡、技术报告、论文、代码仓库和 API 文档追踪新技术、方法和知识点；这些权威资料用于扩展锚点，不作为新的模型发现入口。

“闭环”按内容阶段判断：排行榜发现、权威来源、研究笔记、正式章节和配套同步均具备，标为“内容专题闭环”；已有权威来源、研究笔记和配套同步但尚无独立正式专题，标为“资料级闭环”；只有部分官方入口或被系列章节覆盖，标为“部分覆盖”；只有排行榜条目，标为“仅候选”。任何状态都不代表参数、训练配方、内部架构或独立 benchmark 已全部公开确认。

## 工作流

1. **模型发现（唯一入口）**：定期查看 [Artificial Analysis](https://artificialanalysis.ai/zh) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 两个排行榜；只有在这两个榜单中出现的模型才进入新的候选盘点。记录榜单名称、具体页面、配置/effort、榜单日期、采集日期和快照哈希。
2. **官方核验**：对已由上述两个榜单发现的候选，优先收集其官方博客、技术报告、论文、模型卡、代码仓库和 API 文档；这些资料只用于核验候选事实和扩展周边技术，不作为新的模型发现入口。社区博客只用于补充工程经验。
3. **技术拆解**：每个模型记录发布时间、组织、开放/闭源状态、参数或上下文信息、训练/推理特色、已确认来源和待核验点。
4. **知识映射**：把新技术映射到现有书册：架构、训练、推理、Reasoning、Agent、工具协议、多模态、AI Infra、评估与安全。
5. **内容更新**：先更新第四册百科和相关专题章节，再同步 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。
6. **质量检查**：检查时间语境、模型名称、许可证、版本差异、公式和代码；所有事实保留来源链接与核验日期。

## 首批模型观察清单

以下是历史上从两个排行榜发现、随后进入核验流程的候选，不代表项目已确认其全部事实：GPT、Claude、Gemini、DeepSeek、Qwen、Kimi、GLM、MiniMax、Mistral、Step、Grok 及其他推理、代码、多模态和 Agent 模型。历史清单中的其他厂商记录继续保留，但当前不主动追踪其新模型。以后不得仅因官方新闻、Hugging Face Trending、LMSYS、云厂商目录、论文列表或搜索结果出现名称，就把它新增为模型候选；必须先回溯到上述两个排行榜中的条目，并且属于当前重点厂商范围，或已有用户明确指示。

## 每个模型的记录模板

- 模型与版本：
- 首次公开日期：
- 发布组织与链接：
- 开放性、许可证与可用渠道：
- 规模、上下文和模态：
- 核心新技术：
- 训练/后训练/推理特点：
- 主要基准与局限：
- 官方来源：
- 论文/技术报告：
- 社区资料：
- 核验日期与可信度：
- 对本书系的更新位置：

## 交付顺序

先提交模型盘点表和来源索引，再更新百科与专题章节，最后同步题库、练习、术语、项目、论文和进度文件。每轮以可审阅的小批次提交，避免再次形成难以维护的大型总表。

## 正式章节的深度要求

正式正文以讲清讲透为验收条件，不设置篇幅上限。研究清单和来源摘记只用于组织证据，不能直接充当章节。每个重要知识点独立展开，不把多个未解释的术语堆在同一段。

先用具体情境或生动例子说明问题，再逐步引入定义、历史动机、直觉、机制和公式。公式解释符号及适用假设；代码说明依赖、输入输出、关键步骤和运行现象。适合代码辅助理解的主题提供可运行的小实验，并说明教学实现与真实系统之间的差距。

面向初学者解释前置概念、常见误解与中间推导；面向有经验读者展开复杂度、工程取舍、失败条件、实验对照和面试追问。各项按主题需要组织，不机械填模板；必要时拆成多节或独立章节，不能通过省略解释缩短篇幅。

## 当前研究入口

- [模型候选盘点](research/model-update-2026-09/model-inventory.md)：需区分独立模型和推理配置，并核验榜单日期。
- [GPT-5.6 官方资料摘记](research/model-update-2026-09/gpt-5.6-source-notes.md)：已核验 Sol/Terra/Luna 模型页、Reasoning、Agents、Tools、Prompt Caching、Compaction 和开发者博客；当前为资料级闭环。
- [GPT-5.5 官方资料摘记](research/model-update-2026-09/gpt-5.5-source-notes.md)：已核验 GPT-5.5/Pro 模型页、专属指南、Reasoning、Tools、Tool search、Prompt Caching、Compaction、Images/Vision、Conversation state 和 Background；当前为资料级闭环。
- [GPT-5.4 官方资料摘记](research/model-update-2026-09/gpt-5.4-source-notes.md)：已复验两个排行榜、GPT-5.4 模型页与专属指南，并提取 1M context、deferred `tool_search`、computer use、native compaction、custom tools/CFG、`allowed_tools`、`phase` 和 Responses 状态等面试主线；当前为资料级闭环。
- [Kimi K3 官方资料摘记](research/model-update-2026-09/kimi-k3-source-notes.md)：已发现架构、优化、量化、缓存和 Agent 兼容性扩写方向。
- [Kimi K2.7 Code 官方资料摘记](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)：已核验两个排行榜、Moonshot/Kimi 官方资源/API 文档、固定 revision 模型卡、配置和部署指南；当前为资料级闭环。
- [Claude Opus 5 官方资料摘记](research/model-update-2026-09/claude-opus-5-source-notes.md)：已核验两个排行榜锚点、模型目录/专属页、adaptive thinking、工具与 effort 中途变更、fallback、缓存边界、发布方评测与 system card 入口；当前为资料级闭环。
- [Claude Fable 5 官方资料摘记](research/model-update-2026-09/claude-fable-5-source-notes.md)：已核验两个排行榜锚点、模型页、发布/重新部署公告、thinking/effort、拒答/fallback、memory、程序化工具调用、compaction、context editing 和 task budgets；当前为资料级闭环。
- [Claude Opus 4.8 官方资料摘记](research/model-update-2026-09/claude-opus-4.8-source-notes.md)：已核验两个排行榜锚点、Anthropic 发布公告、system card、Dynamic Workflows 博客、effort 和 Messages API 的 system entry；当前为资料级闭环。
- [Claude Fable 5.1 官方资料摘记](research/model-update-2026-09/claude-fable-5.1-source-notes.md)：已复验 Artificial Analysis 榜单、确认 DataCurve 暂无 Fable 5.1 行，并补齐 Anthropic 发布页、System Card 入口、Fable/Mythos safeguards、cache read 定价、发布方 benchmark、安全摘要和 arXiv 外部使用检索；当前为资料级闭环。
- [Claude Sonnet 5 官方资料摘记](research/model-update-2026-09/claude-sonnet-5-source-notes.md)：已完成两个排行榜、Anthropic 发布页、模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向复核；已核验 Adaptive thinking、effort/预算、thinking block、tokenizer、context awareness、compaction、工具调用和配置级评测边界，当前为资料级闭环。
- [Claude Sonnet 4.6 官方资料摘记](research/model-update-2026-09/claude-sonnet-4.6-source-notes.md)：已完成两个排行榜、Anthropic 发布页、真实 Models Overview、System Card 入口和 Agent 运行时资料复核；已核验 1M context beta、adaptive/extended thinking、context compaction、tool search、computer use 与 prompt-injection 边界，当前为资料级闭环。
- [Claude Haiku 4.5 官方资料摘记](research/model-update-2026-09/claude-haiku-4.5-source-notes.md)：已核验模型目录快照、200K context、64K 输出、extended thinking、fastest 延迟、平台与成本边界。
- [DeepSWE v1.1 排行榜快照](research/model-update-2026-09/deepswe-snapshot-notes.md)：已核验页面更新时间、任务/仓库/语言规模、统一 `mini-swe-agent` harness 和配置级 Pass@1/成本字段。
- [Artificial Analysis 快照字段解释](research/model-update-2026-09/inventory-interpretation.md)：已补记 2026-09-09 配置级指数示例及 `releaseDate`、参数和开放性字段的第三方证据边界。
- [DeepSeek V4.1-Flash 官方资料摘记](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)：已核验发布页、固定 Hugging Face revision、模型卡、配置、encoding/evaluation README 和 API alias 边界；技术报告 PDF 已逐页提取，补充层排布、缓存/部署、训练基础设施、后训练和评测边界。完整 kernel source、线上接受率和独立 profiling 仍待核验。
- [DeepSeek V4 Flash Vision 官方资料摘记](research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md)：锚点为 Artificial Analysis 的 `deepseek-v4-flash-vision`；已核验实验发布、Vision/Files/Responses/价格文档、图像预算、`file_id` 生命周期、工具图像回灌和旧 alias → V4.1-Flash 路由；DataCurve 当前无同名行，视觉专属架构/训练报告仍待核验，当前为资料级闭环。
- [Artificial Analysis 2026-09-14 实时快照](research/model-update-2026-09/artificial-analysis-2026-09-14-snapshot.md)：独立保存 1.77 MB 页面快照、哈希、约 702 个配置条目和 13 个新增结构 slug/别名；DeepSeek V4.1-Flash、DeepSeek V4 Flash Vision 与 K2 36B/A4B 已进入官方核验专题，其余新增项仍停留在发现层。
- [K2-Horizon-MoVA-36B-A4B 官方资料摘记](research/model-update-2026-09/k2-horizon-source-notes.md)：锚点来自 Artificial Analysis 榜单；官方 Hugging Face 模型卡、固定 revision、配置、实现和部署资料只用于核验该候选及扩展 MoVA/MoE/长上下文/Serving 技术。
- 上一锚点为 `IFM/K2-Horizon-MoVA-36B-A4B`：已完成 MoVA、稀疏 MoE、长上下文和 serving 的专题同步；后续新模型候选仍只能从两个排行榜产生，再按证据等级推进核验。
- [Qwen3.8 官方资料摘记](research/model-update-2026-09/qwen3.8-source-notes.md)：已核验 27B、2.4T-A95B、Flash-Next 的官方模型卡、Flash-Next GitHub/技术报告及 Qwen Cloud 的 hosted/open 边界；核心技术为 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol。
- 当前 Qwen3.8 路线进入“内容专题闭环”：正式落点为第二十一册第 83 章。QSA 的完整 kernel、GR/N-gram host-memory 端到端收益、Muon 分布式实现、线上 acceptance rate、目标硬件 profiling 和独立 benchmark 继续保持待核验；Flash-Next 报告结果只写成发布方自报。
- [Gemini 3.7 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.7-flash-source-notes.md)：已重新复验 Artificial Analysis、DataCurve DeepSWE、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Interactions API、工具组合、Thinking、Context caching、Computer Use 和 agentic video 文档；当前为资料级闭环。
- Gemini 3.7 Flash 的面试主线是 thinking budget、interaction/step/state/background、tool context circulation、加密 signature 的 stateful/stateless 回放、built-in/custom tool 责任边界，以及 agentic video 的主动时间轴浏览和 `processing_call`/`processing_result`；Model Card 没有公开独立架构、参数或训练 recipe，暂不新增专属正式章节。
- [Gemini 3.6 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)：已复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking、Interactions、工具组合、视频、缓存和 Computer Use 文档，并完成 arXiv 精确/全文检索；当前为资料级闭环。
- Gemini 3.6 Flash 的面试主线是默认 medium 与 `minimal/low/medium/high` thinking、Interactions/state replay、Gemini 3 tool signatures 与 tool context circulation、agentic video 的 `processing_call`/`processing_result`、4,096-token implicit caching 和 Computer Use 的宿主执行边界；Model Card 将架构/训练/硬件/软件资料指向 Gemini 3.5 Flash，暂不新增专属正式章节。
- [Gemini 3.5 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.5-flash-source-notes.md)：已重新复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；当前为资料级闭环。
- Gemini 3.5 Flash 的面试主线是默认 thinking 从 high 改为 medium、`thinking_level`/`thinking_budget` 互斥、thought preservation、Interactions/state replay、tool context circulation、4,096-token implicit caching、Computer Use prompt-injection detection，以及静态视频与 agentic video 支持列表的证据边界；官方把架构/训练/软硬件资料指向 Gemini 3 Flash，暂不新增专属正式章节。
- [Gemini 3.1 Pro Preview 官方资料摘记](research/model-update-2026-09/gemini-3.1-pro-preview-source-notes.md)：已重新复验 Artificial Analysis、DataCurve、Google AI Developers 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、thought signatures、工具组合、long context、caching 和 arXiv 定向检索；当前为资料级闭环。
- Gemini 3.1 Pro Preview 的面试主线是 `thinking_level` 与共同 output budget、`customtools` endpoint 的工具选择偏置、thought/tool signature 与 `id` 回放、tool context circulation、1M context/caching、评测设置隔离和 Frontier Safety 的 CCL/成本边界。Model Card 将架构/训练/硬件/软件资料指向 Gemini 3 Pro，暂无独立 3.1 技术报告，因此不新增专属正式章节。
- GLM-5 本轮只作为 Artificial Analysis 单榜候选推进：DataCurve 当前快照没有 `mini_swe_agent_glm_5_*` 精确行。官方技术报告已确认，研究主线转向 DSA indexer/top-k 召回、MoE 总容量与 active compute、`slime` 的 rollout/trainer 解耦、长轨迹 Agent RL 和评测分层；完整 DSA kernel、异步调度细节、训练 recipe、硬件 profiling 和独立复现待补证，不新增独立正式章节。
- [Gemini 3.8 Flash 官方资料摘记](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)：已核验 Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF，以及 Thinking、Thought signatures、Interactions API、工具组合、Computer Use、File Search、URL Context、Code Execution、Structured outputs 和 Long context 文档。
- Gemini 3.8 Flash 已完成榜单发现、官方资料入库和周边技术提取，状态为“资料级闭环”。由于没有独立公开的 3.8 架构/训练报告，暂不新增专属正式章节，先映射到 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving 与评测章节。
- [Grok 4.5 官方资料摘记](research/model-update-2026-09/grok-4.5-source-notes.md)：已核验两个排行榜的 `high` 配置、xAI 发布公告、Grok 4.5 模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档；当前为资料级闭环。
- [Grok 4.6 官方资料摘记](research/model-update-2026-09/grok-4.6-source-notes.md)：已核验两个排行榜的四档 effort、xAI 发布公告、官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；当前为资料级闭环。
- [GLM-5.3-Flash 官方资料摘记](research/model-update-2026-09/glm-5.3-flash-source-notes.md)：已复验两个排行榜、Z.ai 官方模型文档/博客、固定 revision 模型卡与配置；已核验 hybrid linear+sparse attention、IndexPool、稀疏 MoE、mHC、原生视觉 coding loop、EPD serving、API thinking/tool streaming 与评测边界；当前已进入“内容专题闭环”收口。
- [GLM-5 官方资料摘记](research/model-update-2026-09/glm-5-source-notes.md)：Artificial Analysis 有精确条目，但 DataCurve 当前没有精确 GLM-5 行；已核验 Z.ai 模型卡、专属技术报告、API/部署资料、DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering。当前为 AA 单榜资料闭环，不迁移 GLM-5.2/5.3 的 DeepSWE 结果。
- [GLM-5.2 官方资料摘记](research/model-update-2026-09/glm-5.2-source-notes.md)：已重新复验 Artificial Analysis 的 max/non-reasoning 与 DataCurve 的 high/max，补齐 Z.ai 官方 1M/128K、长周期 Coding Agent、MCP、缓存和工作流资料；当前为双榜资料级闭环，SAO/compaction 原始定义、参数/架构和完整训练 recipe 待核验。

GPT-5.6 已从“仅候选”升级为“资料级闭环”：Sol/Terra/Luna 的官方模型页、reasoning mode/effort、persisted reasoning、Responses 工具循环、prompt caching、compaction 和开发者博客工程证据已入库。由于没有独立架构或训练报告，暂不新增 GPT-5.6 专属正式章节；正式面试主线映射到第六、七、十六、十七、二十和二十四册。

GPT-5.5 与 GPT-5.5 Pro 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、官方模型页、专属指南和 API 周边资料已入库。正式面试主线是高效 reasoning 的可测性、outcome-first prompting、tool search、图像 detail、`text.verbosity`、Responses `phase` 回放、compaction、Pro background mode，以及与 GPT-5.6 的 prompt-cache 协议差异；由于没有独立架构或训练报告，暂不新增 GPT-5.5 专属正式章节，映射到第六、七、十六、十七、二十和二十四册。

GPT-5.4 已从“仅候选”升级为“资料级闭环”：两个排行榜的可追溯配置、OpenAI 官方模型页与专属指南已入库，面试主线覆盖 1M context、reasoning/visible output 分账、deferred `tool_search`、computer use、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、opaque compaction 和模型—协议—harness—执行器分层。GPT-5.4 Pro 只作为关联服务档位记录，mini/nano 尚未独立核验；由于没有 GPT-5.4 专属参数、架构、完整训练报告或独立复现，不新增专属正式章节。

Gemini 3.7 Flash 已从“仅候选”升级为“资料级闭环”：Artificial Analysis 与 DataCurve 的三档配置、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Interactions/工具/视频文档、arXiv 精确标题负检索和外部使用论文均已入库。由于 Model Card 将架构/训练/软硬件资料指向 Gemini 3.6，暂无独立 Gemini 3.7 技术报告，因此不新增专属架构章节；当前锚点切换为 `gemini-3.7-flash`，下一轮仍从两个排行榜的剩余重点厂商条目中选择。

Claude Fable 5 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、Anthropic 模型页、发布/重新部署公告、thinking/effort、拒答与 fallback、memory、程序化工具调用、compaction、context editing、task budgets 及 system card 入口已入库。面试主线是 opaque reasoning state、effort 与硬输出预算的区分、业务态拒答、跨模型路由、长任务 harness 和工具结果隔离；由于没有独立架构或训练报告，暂不新增 Fable 5 专属架构章节，映射到第六、七、十六、十七、二十和二十四册。

Claude Opus 4.8 已从“仅候选”升级为“资料级闭环”：两个排行榜的配置锚点、Anthropic 发布公告、system card、Dynamic Workflows 官方博客和研究笔记已入库。面试主线是动态编排图、并行 subagents 与独立验证、effort sweep、任务中途 system entries、prompt cache 保持和“完成声明”与 artifact 验证的分离；由于没有独立架构或训练报告，暂不新增 Opus 4.8 专属架构章节，映射到第六、七、十六、十七、二十和二十四册。当前锚点为 `claude-opus-4.8`，下一步转向 `kimi-k2.7-code`。

Kimi K2.7 Code 已从“仅候选”升级为“资料级闭环”：两个排行榜的条目、官方资源/API 文档、固定 revision 模型卡、MoE/MLA/MoonViT/Native INT4 配置、thinking/preserve-thinking/tool-call 协议、部署指南和研究笔记已入库。面试主线是长周期 coding agent、1T/32B active MoE、MLA 长上下文账本、INT4 部署、可回放 reasoning state 和 harness-aware evaluation；由于没有独立专属训练报告或外部复现，暂不新增 K2.7 Code 专属正式章节，映射到架构、推理、Agent、工具协议、Serving 和评测章节。

Grok 4.5 已从“仅候选”升级为“资料级闭环”：Artificial Analysis 的 `high`、DataCurve 的 `mini_swe_agent_grok_4_5_high`、xAI 发布方对训练/评测/异步 rollout 的公开描述、Grok 4.5 官方模型/API 文档和研究笔记均已入库。面试主线是 reasoning effort 与 opaque state、Responses 30 天状态、context compaction、server-side/client-side tool responsibility、MCP 最小权限、Web/X Search 和 harness-aware evaluation。由于没有独立架构或训练报告，不新增 Grok 4.5 专属正式章节；专属模型页与通用 Reasoning 文档的 `xhigh` 字段冲突保留为待复验项。

Grok 4.6 已从“仅候选”升级为“资料级闭环”：Artificial Analysis/DataCurve 的四档配置、xAI 2026-08-12 发布公告、模型生成数据与 SFT 轨迹筛选、agentic RL、长任务 self-testing/verification、Grok 4.6 API/工具/compaction 文档和论文负检索均已入库。由于没有独立架构或完整训练报告，不新增 Grok 4.6 专属正式章节；该锚点已完成，后续转入 Anthropic 系列。

新模型发现边界：Artificial Analysis 与 DataCurve DeepSWE 是唯一的候选入口。选定锚点后，可以沿官方资料追踪其同系列模型、adapter、论文、训练方法、推理内核和部署工具；这些是“锚点周边技术”，不能绕过两个排行榜变成新的独立候选发现。对排行榜未出现的周边模型（例如 Uno adapter）必须明确标为“官方资料中的关联技术”，不得写成榜单候选或独立模型排名。

Claude Opus 5 已从“仅榜单发现”升级为“资料级闭环”：两个排行榜的 `max`/`mini-swe-agent` 配置、Anthropic 官方模型/API 文档、发布公告、system card 入口、thinking/effort 中途变更、fallback、缓存边界、发布方评测和 arXiv/Research 负检索均已入库。没有公开参数、架构或完整训练报告，因此不新增 Opus 5 专属架构章节；内容映射到 Reasoning、长上下文/缓存、Agent 工具协议、安全 fallback 和公平评测章节。

Claude Sonnet 5 已从“仅榜单发现”升级为“资料级闭环”：两个排行榜的 `max/xhigh/high/medium/low/Non-reasoning` 配置、Anthropic 发布页/模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向检索和研究笔记均已入库。面试主线是 adaptive thinking 与 effort/`max_tokens` 的分层、thinking block/signature 回放、新 tokenizer 对预算的影响、context awareness/compaction、工具宿主责任以及 Artificial Analysis、DeepSWE 和发布方评测的证据隔离；由于没有独立架构或训练报告，不新增 Sonnet 5 专属正式章节。

Qwen3.8 已从“仅榜单发现”升级为“内容专题闭环”：27B、A95B 和 Flash-Next 的参数/层布局/模态/思考协议来自官方模型卡，QSA/GR/N-gram/Muon 来自 Flash-Next 官方技术报告与仓库；Qwen3.8-Max 只作为基于 A95B 的 hosted version 记录。不要把 Max、Flash hosted 或 `reasoning_effort` 行写成新的 open checkpoint。

GLM-5.3-Flash 已从“部分覆盖”升级为“内容专题闭环”：Artificial Analysis 与 DataCurve 只提供锚点和配置级 Agent 结果；Z.ai 文档/博客与固定 revision 模型卡支持 320B/18B、45 层、34 linear/11 sparse、288 routed/top-8/1 shared、IndexPool、mHC、视觉 self-verification、EPD serving 与 thinking/tool streaming 的专题写作。官方披露的 3.01×/4.44× attention/KV、约 3× serving、benchmark 和视觉工作流均保留为发布方自报；生产 kernel、真实 cache/state bytes、硬件 profiling、线上 acceptance rate、完整训练 recipe 和独立复现仍待核验。正式落点为第二十一册第 84 章。

2026-09-15 Claude Opus 5 资料级闭环：`10.237.126.170:1234` 和 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Opus 5 详情页、DataCurve 和 Anthropic 官方页面均返回 HTTP 200；`10.24.27.134:7890` 对 DataCurve 成功，但 Artificial Analysis 大页面传输超时。Opus 5 的 Artificial Analysis 详情快照为 3,532,163 bytes，SHA-256 `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f`；DataCurve 快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

Anthropic 官方资料确认 `claude-opus-5`、2026-07-24、1M context、128K/300K 输出、adaptive thinking、默认 `high`、五档 effort、512-token 缓存门槛、tool/effort 中途变更和 server-side fallback。DeepSWE max 的 327/444、约 73.65% Pass@1、约 `$11.84`/task 与 Artificial Analysis 约 50.70 指数均保留配置、harness、任务集、工具和 verifier 归因；不作为裸模型排名。发布方 Frontier-Bench/CursorBench/ARC-AGI 3/OSWorld 等结论也标为 Anthropic 自报，不能替代独立复现。

arXiv 精确标题检索得到两篇将 Opus 5 作为被测模型的文章，没有 Opus 5 专属技术报告；Anthropic Research 页也没有专属报告条目。System card 已登记哈希但当前环境无法稳定抽取正文，故不扩写未明确的安全数字。当前锚点切换为 `claude-opus-5`，下一锚点为 `claude-fable-5.1`；下一轮沿 Anthropic 官方资料补 Fable 5.1 的榜单配置、运行时协议和长任务证据。

2026-09-14 质量收口：第二十册第 13、17、18、21 章残留的 `max(1, …)` 指标分母已改为严格非空定义域，并补充 `not_applicable` 语义；第 13 章 demo 对空需求/空风险集合返回 `None`，聚合分数和门禁显式拒绝无样本。官方页面复访仍受 DNS 解析失败影响，未将网络失败写成资料不存在。

2026-09-14 榜单代理复访：Artificial Analysis 中文首页与 DataCurve DeepSWE 均可通过已提供的三个代理返回 HTTP 200。Artificial Analysis `/zh` 本次快照为 1,771,203 bytes、SHA-256 `510bb1916a029700fbd78084a4b26cbe1e2b2690ce9e25179fde2d4fe650a7d4`；与此前实时快照规范化比较，已跟踪的模型 release 记录无新增或删除。当前仍有 GPT-6 Astra（榜单 `releaseDate` 2026-09-03）、Claude Fable 5.1（2026-09-01）、GLM-5.3（2026-08-18）和 GLM-5.3-Flash（2026-08-26）。DataCurve 页面快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，与既有 `/tmp/deepswe.html` 字节一致，页面更新时间仍为 2026-09-03；其中有 GPT-6 Astra、GLM-5.3 和 Claude Fable 5，但没有 Claude Fable 5.1。此次没有产生新的候选，后续仍按两榜单唯一入口推进。

## 2026-09-14 Qwen3.8 配套同步收口

Qwen3.8 的专题内容已同步到 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并保留研究笔记、模型盘点、来源索引及第二十一册第 83 章的交叉入口。同步内容只围绕已发现锚点的 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol 展开。

第 83 章的 QSA 教学 demo 已修正为先为 causal tail 预留 token budget，再选择完整 micro-block；预算无法容纳 tail 时显式报错，并用断言锁定 tail 不被截断。该 demo 仍是标准库教学实现，不代表 Flash-Next 的生产 kernel；报告的 benchmark、loss、速度和稳定性数字继续按发布方自报处理。

下一步仍是补证队列：完整生产 kernel、目标硬件 profiling、线上 acceptance rate、N-gram host-memory 端到端收益、Muon 分布式实现、全系列训练/后训练配方和独立 benchmark。除非两个排行榜出现新条目，否则不新增模型候选。

## 2026-09-14 GPT-5.6 资料级闭环与后续顺序

GPT-5.6 的研究笔记已完成，闭环范围包括榜单发现、官方模型页、API 周边文档、开发者博客和证据边界。当前最适合面试的技术主线是：`standard/pro` 与 effort 解耦、跨轮 opaque reasoning state、function call 后的 reasoning item 回传、GPT-5.6 prompt-cache 显式断点/1,024 token 门槛/30m TTL、compaction 对缓存复用的影响，以及模型—协议—harness—执行器的责任分层。

本轮不把 GPT-5.6 或 GPT-5.5 的 DeepSWE 数字写成裸模型能力，也不把 API 字段推断为参数、MoE、训练或 RL 机制。下一锚点转向 Claude Fable 5；GPT-5.5/5.6 后续只补官方版本变更、真实 API 行为和独立评测复现。

2026-09-15 网络复访：`10.24.27.134:8098` 和 `10.237.126.170:1234` 均可访问 Artificial Analysis `/zh` 与 DataCurve 并返回 HTTP 200；`10.24.27.134:7890` 本次访问 DataCurve 返回 200，但访问 Artificial Analysis 出现 TLS EOF。Fable 5 页面快照已通过可用代理保存，页面大小和 SHA-256 见研究笔记。

## 2026-09-15 Claude Opus 4.8 资料级闭环与网络复验

Opus 4.8 已完成从两个排行榜发现到官方资料、研究笔记和配套同步的资料级闭环。Artificial Analysis 详情页仍保留 `claude-opus-4-8` 历史条目，并标记为 deprecated、指向 Opus 5；DataCurve DeepSWE v1.1 同时保留 `xhigh` 和 `max` 配置。新鲜网络复验中，`10.237.126.170:1234` 对 Artificial Analysis `/zh`、Opus 4.8 详情页和 DataCurve 均返回 HTTP 200；`10.24.27.134:8098` 对两个排行榜也均返回 HTTP 200。复验快照哈希分别为 Artificial Analysis `/zh` `264357ee84d21d14f383cc7b179c34843ab39ef45c5bafad397946c7fadc23fd`、DataCurve `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` 和 Opus 4.8 详情页 `369ef837ec32f3de7b59c8f6fdc7657b8608a22ccff0717c75c6cf3c0167c522`。

Opus 4.8 的公开面试主线是 Dynamic Workflows 的外部编排、并行 subagents、独立验证和断点恢复；`high`/`xhigh`/`max` effort 的成本—成功率—延迟对照；任务中途 system entries 对权限、预算、环境状态和缓存的影响；以及把模型的诚实性声明转成测试、lint、类型检查和领域 verifier 的 artifact 门禁。Anthropic 没有公开参数规模、网络结构、训练配方或独立可复现技术报告，因此这些空白保持为待核验，不能由产品行为反推。

当前锚点为 `kimi-k2.7-code`，已完成资料级闭环；下一锚点切换为 `grok-4.5`。后续仍只从两个排行榜核对 Grok 4.5，再沿 xAI 官方资料追踪论文、技术文档、模型页、API/Agent 协议和评测边界。

## 2026-09-15 Claude Fable 5.1 断点恢复与闭环

昨晚可能受工作时段代理中断影响，本轮重新发起联网搜索。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Fable 5.1、Artificial Analysis 中文首页和 DataCurve 均返回 HTTP 200；7890 继续作为备用代理，但对 Artificial Analysis 大页面有 TLS EOF。新鲜抓取结果已经写入 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)。

本轮补证完成的重点是：Artificial Analysis 的 `max + Default Fallback` 配置与 53.3738 指数、65.98 tokens/s、212.01s TTFT 和 `$7.63`/task；DataCurve 没有 Fable 5.1 行；Anthropic 发布页对同一 underlying model/不同 safeguards、High/Medium 默认入口、`$0.25/M` cache read、EFS、anti-distillation 和发布方 benchmark 的说明；以及标题精确为 0、全文为 4 篇的 arXiv 定向检索。System Card 已重新下载并登记哈希，但 PDF 正文仍未稳定提取。

Fable 5.1 现在标为“资料级闭环”，不新增独立架构章节。官方没有公开参数规模、稠密/MoE 架构、完整训练/后训练配方、独立技术报告或独立 benchmark 复现；发布页 benchmark 和科学工作流只能作为发布方证据。下一锚点切换为 `claude-sonnet-5`，继续按“榜单配置 → 官方发布/模型页 → API/Agent 文档 → System Card/论文 → 书系映射”的顺序推进。

## 2026-09-15 Kimi K2.7 Code 资料级闭环与联网复验

本轮在用户提供的代理恢复后重新请求两个排行榜和 Kimi 官方资料。`10.237.126.170:1234` 对 Artificial Analysis K2.7 详情页、DataCurve 和 Kimi 资源/API 页面返回 HTTP 200；Hugging Face 固定 revision 页面通过 `10.24.27.134:7890` 返回 HTTP 200。三条代理对两个排行榜的轻量连通性均已复验成功，但 7890 对大页面曾出现读取超时，8098/1234 更适合完整抓取。

Artificial Analysis 详情页快照为 3,605,694 bytes，SHA-256 `e37ce8e8f224233a95ca93771fe9f2130e544ef071703bae475295bd1ac527db`；DataCurve 快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。DeepSWE 行为配置是 `mini-swe-agent`、`reasoning_effort:null`、138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155` 和平均 149.1 steps；这些是系统组合结果。

官方 Hugging Face revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的模型卡和 `config.json` 确认 1T/32B active MoE、61 层、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 路线。Kimi API 快速开始确认 always-on thinking、`preserve_thinking`、固定采样参数、`tool_choice` 限制、reasoning_content 回传和多步工具调用协议；部署指南覆盖 vLLM、SGLang 和 KTransformers。

本轮没有找到 Kimi K2.7 Code 专属 arXiv 论文或独立技术报告：官方资源页和固定 revision README 没有提供此类链接，定向 arXiv 检索未返回专属结果，GitHub 官方组织 API 对同名仓库查询为零结果。该负面证据只表示本轮公开入口未检出，不证明未来不会发布报告。Kimi Linear、Attention Residuals、Mooncake 和 K2 Thinking 继续作为关联路线单独记录，不能改写成 K2.7 Code 专属训练事实。

研究笔记：[`kimi-k2.7-code-source-notes.md`](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)。当前不新增正式章节；后续锚点为 `grok-4.5`。

## 2026-09-15 Grok 4.6 资料级闭环与联网复验

Grok 4.6 已完成从两个排行榜发现到 xAI 官方资料、研究笔记和配套同步的资料级闭环。Artificial Analysis 详情页确认 `Grok 4.6 (high)` 及 low/medium/xhigh 配置；DataCurve DeepSWE v1.1 确认四档 `mini_swe_agent_grok_4_6_*` 配置。四档 DataCurve Pass@1 为 41.648%/67.478%/65.188%/66.741%，成本为 `$1.0424/$3.4490/$4.3849/$5.4977`，均绑定统一 `mini-swe-agent`、工具、任务集、环境和 verifier。

xAI 发布公告确认 2026-08-12 发布，并公开披露更长 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks、面向知识工作/编码/领域环境的 agentic RL 和长轨迹 self-testing/verification。官方模型页确认 500K context、2026-02-01 knowledge cutoff、文本/图像输入、`low/medium/high/xhigh`、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution、`prompt_cache_key`、`x-grok-conv-id` 和 compaction 建议。上述内容不支持参数、架构或完整训练 recipe 推断。

联网复验已重新发起并成功获取所需网页：`10.237.126.170:1234` 和 `10.24.27.134:8098` 对 Artificial Analysis Grok 4.6、DataCurve 均返回 HTTP 200；`10.24.27.134:7890` 对 xAI 官方模型页返回 HTTP 200。轻量复验中 `1234` 访问 xAI 曾返回代理 503，7890 访问 Artificial Analysis 偶发 TLS EOF，因此继续把代理失败与“页面不存在”区分开。官方模型页 HTML 快照为 408,905 bytes，SHA-256 `9669e2c5e96dafb74296f6e11af7c3b0fc74e8dec58a3d674edf18154c1caf27`。

arXiv 精确标题检索本轮未检出 Grok 4.6 专属论文或独立技术报告；这是当前公开入口的负面证据，不是对未来发布的绝对否定。研究笔记见 [`grok-4.6-source-notes.md`](research/model-update-2026-09/grok-4.6-source-notes.md)；下一步切换到 `claude-opus-5`，不新增 Grok 4.6 专属架构章节。

## 2026-09-15 Claude Sonnet 5 资料级闭环与后续顺序

考虑到用户工作时段可能造成夜间代理中断，本轮重新使用可用代理复核 Sonnet 5。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis、DataCurve 和 Anthropic 发布页均返回 HTTP 200；`10.24.27.134:7890` 仅作为备用线路，访问 Artificial Analysis 大页面仍出现 TLS/读取不稳定。当前不能把单个代理的失败解释为网页不存在。

本轮已完成 Sonnet 5 的资料级闭环：Artificial Analysis 主配置 `max` 的第三方指数、速度、TTFT、成本和 1M context；DataCurve v1.1 五档 effort 的 Pass@1、Pass@4、成本、输出 token 和 Agent steps；Anthropic 模型目录/发布页的模型与产品字段；完整开发者文档中的 adaptive thinking、effort、thinking block/signature、新 tokenizer、context awareness、server-side compaction、computer toolset、web/programmatic tool calling 和参数限制；System Card/发布方评测；以及 arXiv 精确标题和 Anthropic Research 负检索。

本轮明确保留三类边界：第一，Sonnet 5 的 effort/Non-reasoning 行是运行配置，不是不同基础模型；第二，Artificial Analysis、DeepSWE 和 Anthropic 自报 benchmark 不互相拼接为裸模型能力；第三，公开资料没有参数规模、稠密/MoE 结构、完整训练/后训练 recipe、内部 adaptive-thinking 机制或独立技术报告。故 Sonnet 5 暂不新增独立架构章节，内容映射到第四、六、七、十七、二十和二十四册。

下一锚点继续从两个排行榜中剩余的重点厂商候选选择；不因 Anthropic 官方目录中的 Haiku 4.5 或其他只在官方页面出现的名称新增候选。已闭环模型后续只补版本变更、真实 API 行为、完整 kernel/硬件 profiling、线上接受率和独立评测等证据缺口。

## 2026-09-15 DeepSeek V4 Flash Vision 资料级闭环与联网恢复复验

`deepseek-v4-flash-vision` 已完成从 Artificial Analysis 锚点到 DeepSeek 官方实验发布、当前 API 文档、研究笔记和第二十一册第 85 章的资料级闭环。历史实验 alias、当前服务路由和榜单配置被分开记录，避免把 API 产品字段误写成视觉模型内部架构。

本次针对夜间代理中断再次完整抓取：`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Vision 详情页均返回 HTTP 200、3,609,278 bytes，SHA-256 均为 `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68`；三个代理对 DataCurve 均返回 HTTP 200、268,313 bytes，SHA-256 均为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；三个代理对 DeepSeek Quick Start 均返回 HTTP 200、46,116 bytes，SHA-256 均为 `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c`。`10.24.27.134:7890` 访问 Artificial Analysis 大页面仍发生 TLS EOF，不能据此判定页面不存在或内容不可用。

Artificial Analysis 当前仍确认 `deepseek-v4-flash-vision` 的 `max` 配置、`releaseDate` `2026-08-21`、1M context、284/13 目录参数、配置级 Intelligence Index `35.0122378035969`、约 `215.1792 tokens/s` 输出速度和 `1.2986s` median TTFT。DataCurve 当前没有 Vision 同名行，不能迁移 V4 Flash/Pro 的 DeepSWE 结果。

官方资料已收口为四条面试主线：一是 `requested_model`、排行榜 `catalog_anchor` 与响应 `served_model` 的三身份账本；二是图像 detail/resize、视觉 token、文本/工具/历史共同预算；三是 Files API `file_id` 的资源生命周期与 Responses `function_call_output(input_image)` 的观察回灌；四是 OpenAI/Anthropic 兼容接口的 capability probe、静默忽略字段、端到端成本和外部 verifier 门禁。实验公告的 384 image tokens 与当前 guide 的约 1024 tokens/image 保留各自日期、alias 和文档语境。

第 85 章 demo 已完成 AST 检查、正常执行和 9 组边界测试；教学估算不冒充生产 tokenizer、视觉 encoder 或计费公式。已同步模型盘点、来源索引、榜单解释、第四册百科、第二十一册目录/时间线和书系配套文件。

当前状态为“资料级闭环”，但视觉专属参数/encoder、完整训练或后训练 recipe、独立技术报告、DataCurve Vision 结果、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验。下一步切换到两个排行榜中剩余的重点候选；不从 DeepSeek 官方目录或这次研究笔记额外发现新模型，已闭环锚点只补证据缺口。

## 2026-09-15 GPT-5.4 资料级闭环与夜间代理恢复复验

本轮针对用户所说的夜间 20:00—次日 09:00 代理中断重新发起联网搜索。`10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 对 Artificial Analysis 中文首页与 DataCurve DeepSWE 均返回 HTTP 200；三条线路取得的排行榜页面字节完全一致。AA `/zh` 快照为 `1,773,715` bytes、SHA-256 `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0`；DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。

GPT-5.4 详情页通过 `7890` 返回 HTTP 200，快照为 `3,488,457` bytes、SHA-256 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`；页面仍确认 `GPT-5.4 (xhigh)`、`releaseDate=2026-03-05`、Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT、1.05M context，并标记 deprecated、指向 GPT-5.5。DataCurve 仍为 `mini_swe_agent_gpt_5_4_xhigh`：234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` steps；这些数字绑定 `mini-swe-agent`、工具、环境、任务集和 verifier，不是裸模型分数。

OpenAI 官方模型页与专属指南本轮均实际抓取成功：模型页 `gpt-5.4-2026-03-05` 快照 SHA-256 为 `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5`，指南快照 SHA-256 为 `61a8e21bea3bc4592dd4eb19383a577a292ee770ae10f2e982aa788c349c04be`。已核验 1,050,000 context、128,000 max output、`none`—`xhigh` effort、text/image → text、deferred `tool_search`、built-in computer use、native compaction、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、reasoning 状态回放、prompt caching 和 server-side compaction。

GPT-5.4 当前为“资料级闭环”：GPT-5.4/Pro 的关联关系已记录，但 Pro 未单独建立完整资料档案，mini/nano 仍为关联候选；没有公开参数规模、激活参数、层/专家结构、训练与后训练 recipe、system card、独立技术报告或完整 benchmark 复现，因此不新增 GPT-5.4 专属正式章节。研究笔记见 [`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)，当前锚点为 `gpt-5.4`；下一步仍从两个排行榜的剩余重点厂商候选中选择，不从 OpenAI 官方目录另发现模型。

## 2026-09-15 Gemini 3.7 Flash 资料级闭环与后续顺序

考虑到用户工作时段可能造成夜间代理中断，本轮重新复验 Gemini 3.7 Flash 的网页资料。沙箱内三条代理均无法直连代理端口；受控联网重试后，8098 成功获取 Artificial Analysis、DataCurve、DeepMind Model Card 和 Research，7890 成功获取 Google AI Developers 与 arXiv，1234 成功获取 arXiv/DeepMind Research。7890 对部分大页面仍有超时，单个代理失败不解释成资料不存在。

Gemini 3.7 Flash 已完成资料级闭环：两个排行榜的 high/medium/low 配置、Google AI Developers 模型页、DeepMind Model Card、官方评测/安全入口、Thinking/Interactions/工具/视频文档和 arXiv 精确/全文检索均已入库。面试主线是 1M 输入与 65K 输出预算、thinking budget、interaction/step/state/background、tool context circulation、加密 signature 的 stateful/stateless 回放、built-in/custom tool 责任边界、agentic video 主动时间轴浏览以及 `processing_call`/`processing_result`。

证据边界保持严格：Model Card 只披露核心 reasoning foundation 的算法改进、agentic video 和可调 thinking，架构/训练/软硬件资料指向 Gemini 3.6；没有 Gemini 3.7 独立参数、完整训练 recipe 或官方专属技术报告。arXiv 精确标题返回 0 个结果，全文检索的 3 篇论文只是外部使用/评测，不是 Google 模型报告。因此不新增 Gemini 3.7 专属架构章节，技术映射到 Reasoning、Agent/工具协议、视频多模态、长上下文/Serving、评测与安全章节。

当前锚点切换为 `gemini-3.7-flash`；下一锚点继续从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜剩余的重点厂商条目中选择，不从 Gemini 3.6 前代资料、DeepMind 官方目录或外部论文另发现模型。

## 2026-09-15 Gemini 3.6 Flash 资料级闭环与联网恢复复验

昨晚可能受 20:00—次日 09:00 工作时段代理中断影响，本轮重新请求并确认三个用户提供的代理均可访问榜单入口；随后使用可用线路重新取得 Gemini 3.6 的 Artificial Analysis 详情/provider 页、DataCurve、Google AI Developers、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 检索页。单一代理偶发连接失败仍不作为页面不存在的证据。

Gemini 3.6 Flash 已完成从两个排行榜锚点到官方资料的“资料级闭环”。Artificial Analysis high 详情页记录 `releaseDate` `2026-07-21`、Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s` 和 1M context；DataCurve 的 `mini-swe-agent` high 行为 211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095`、平均 116.73 Agent steps。这些仍分别绑定第三方测量和 DataCurve harness/工具/环境/verifier。

Google 官方模型页确认 `gemini-3.6-flash`、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、默认 medium 与 `minimal/low/medium/high` thinking、caching/code execution/File Search/function calling/grounding/structured outputs/URL context、Computer Use Preview 以及 Batch/Flex/Priority inference。官方文档还明确把 Gemini 3.6 列入 Interactions、agentic video 和 4,096-token implicit caching 支持表；Gemini 3 工具组合的 thought/tool signatures 和 tool context circulation 已作为协议层技术记录。

DeepMind Model Card 明确 3.6 based on Gemini 3.5 Flash，架构/训练数据/数据处理/硬件/软件资料指向 3.5 Model Card；已收录 Model Card benchmark、安全评估和 Frontier Safety 边界，但不把前代资料迁移为 3.6 独有架构。arXiv 精确标题检索返回 0 篇，全文检索 9 篇均为外部使用/评测论文，不是 Gemini 3.6 技术报告。

当前不新增 Gemini 3.6 专属正式章节；完整研究笔记、9 篇论文标题与摘要要点、快照哈希和待核验项见 [`gemini-3.6-flash-source-notes.md`](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)。后续仍只能从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜选择下一个重点锚点，不从 Gemini 3.5 Model Card、官方目录或外部论文另发现模型。

## 2026-09-15/16 Gemini 3.5 Flash 断点恢复与资料级闭环

昨晚 20:00—今早 09:00 的联网中断风险按“重新收集”处理：2026-09-15 工作时段通过 `10.237.126.170:1234` 与 `10.24.27.134:8098` 重新获取 Artificial Analysis Gemini 3.5 详情页和 DataCurve，两个代理返回的榜单快照逐字节一致；7890 获取 Google 官方页面。2026-09-16 轻量复探中，三个代理对 AA/DataCurve 均返回 HTTP 200，7890 对 Google API 模型页返回 HTTP 200，8098/1234 对该大页面超时；后续仍按站点选择代理，单代理失败不写成页面不存在。

Gemini 3.5 Flash 已完成资料级闭环。Artificial Analysis 的 `high` 详情页字段为第三方 `releaseDate` `2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、TTFT `18.3631s`、1M context 和约 `$1.5625`/task；DataCurve `mini_swe_agent_gemini_3_5_flash_high` 为 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、约 105.30 steps。两者均绑定配置、provider/harness、工具、任务环境和 verifier，不能写成裸模型能力；AA 当前 deprecated/指向 3.6 只保留为第三方目录字段。

Google API/What's New/Model Card 与周边文档确认 1M/65K、默认 medium 和四档 thinking、默认 thought preservation、Interactions、tool context circulation、signature/id 回放、4,096-token implicit caching、Computer Use Preview 及 screenshot prompt-injection detection。视频文档的 agentic processing 支持列表列出 3.5 Flash-Lite、3.6、3.7、3.8，但没有明确列出 3.5 Flash，因此不迁移 3.6 的 agentic video 结论。Model Card 把 architecture、training dataset、data processing、hardware 和 software 指向 Gemini 3 Flash Model Card；arXiv 标题精确检索为 0，全文检索为 30 篇外部使用/评测论文，没有发现独立 3.5 技术报告。

当前不新增 Gemini 3.5 专属正式章节，内容映射到 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving、评测与安全章节。下一锚点优先为 `gemini-3.1-pro-preview`，但仍必须先在 Artificial Analysis 或 DataCurve 中确认其榜单锚点；不得仅因官方目录或 3.5/3.6 前代资料自动新增候选。

## 2026-09-16 Gemini 3.1 Pro Preview 断点恢复与资料级闭环

昨晚 20:00—今早 09:00 的联网中断风险按“重新收集”处理。2026-09-16 重新使用三条用户提供的代理抓取 Artificial Analysis 中文首页、Gemini 3.5/Gemini 3.1 Pro 详情页和 DataCurve；六份榜单请求均返回 HTTP 200，同一页面三条线路字节一致。随后通过 7890 重新取得 Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、工具组合、Long context、Caching 和 arXiv 检索页。

Gemini 3.1 Pro Preview 已完成从两个唯一排行榜到权威资料的资料级闭环：Artificial Analysis 详情页记录第三方 Intelligence Index `30.3597`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT、1M context；DataCurve `mini_swe_agent_gemini_3_1_pro_preview_high` 为 53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均 75.56 steps。所有数字保留配置、provider/harness、工具、任务集、环境和 verifier 边界，不能拼成裸模型排名。

官方资料确认 2026-02-19 发布、1,048,576 输入/65,536 输出、多模态输入、thinking、function calling、grounding、structured outputs 和 context caching；`gemini-3.1-pro-preview-customtools` 是面向 bash/custom tools 的 endpoint variant，不新增模型。Model Card 明确 3.1 基于 Gemini 3 Pro，并把架构、训练数据、硬件和软件资料指向 Gemini 3 Pro；没有独立参数、层/专家结构、完整训练 recipe 或专属技术报告。

本轮提取的面试主线为 `thinking_level` 与共同 output budget、thought/tool `signature` 与 `id` 回放、tool context circulation、built-in/custom tool 宿主责任、1M context 的缓存/召回/成本工程，以及 Model Card/Frontier Safety 的评测设置与 CCL 证据边界。arXiv 标题精确检索为 0，全文检索 198 篇均为外部使用/评测结果。研究笔记见 [`gemini-3.1-pro-preview-source-notes.md`](research/model-update-2026-09/gemini-3.1-pro-preview-source-notes.md)。当前不新增独立正式架构章节；下一锚点继续从两个排行榜的剩余重点厂商候选中选择。

## 2026-09-16 Claude Sonnet 4.6 断点恢复与后续顺序

昨晚 20:00—今早 09:00 的联网中断风险已按计划重新复验。三条代理对 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Sonnet 4.6 发布页均返回 HTTP 200；两个排行榜快照逐字节一致。`8098/1234` 的 Anthropic 模型目录被区域页重定向，`7890` 成功取得真实 `platform.claude.com` Models Overview，因此按线路差异记录，不把单代理失败解释为页面不存在。

Claude Sonnet 4.6 已完成“排行榜锚点 → 官方发布页/模型目录 → System Card 入口 → Agent 文档 → 研究笔记”的资料级闭环。后续面试重点是 1M context 的有效容量、adaptive/extended thinking 的成本—延迟—成功率分层、compaction 状态协议、computer-use 执行器安全，以及 tool search 与上下文预算；参数、架构、训练 recipe、compaction 内部格式和独立 benchmark 仍待核验，不新增独立架构章节。

当前锚点切换为 `claude-sonnet-4-6`；下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜选择重点厂商候选，继续按“榜单配置 → 官方资料 → 技术拆解 → 面试映射 → 证据边界”的顺序推进。
