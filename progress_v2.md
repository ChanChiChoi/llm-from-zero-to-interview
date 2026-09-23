# 大模型资料更新进度 v2

更新时间：2026-09-23

## 当前状态

- 阶段：模型候选核验、专题写作与纵向同步（持续进行）
- 主线：新模型发现 → 官方资料核验 → 技术知识映射 → 书系同步 → 代码/链接/证据审计
- 规划文件：[`plan_v2.md`](plan_v2.md)
- 旧版进度：`PROGRESS.md`，仅作历史记录
- 当前活动锚点：`Kimi K3`；Gemini 3.8 Flash 已完成本轮 Thinking/Interactions state/signature、tool context 和 local replay toy 阶段，DeepSeek V4.1-Flash 已完成官方 API contract 和 local protocol toy 阶段。K3 本轮沿两个排行榜已有的 `kimi-k3` canonical 条目继续，不从 Kimi 官方仓库、HF、vLLM 或 SGLang 另发现模型；重点是当前榜单复验、官方 artifact revision、DataCurve harness 账本和发布限制的面试边界。`low/max` 仍归并为一个基础模型，DataCurve 只引用精确 `mini_swe_agent_kimi_k3_max` 行；K3 完整权重、目标硬件、双状态恢复、tool/verifier acceptance 和生产 SLO 仍为 `unverified`。

## 已完成

- 上一活动锚点收口（2026-09-22）：已由 Artificial Analysis 发现的 `Qwen3-VL-235B-A22B` 完成专题写作和同步。三条代理取得 AA instruct/reasoning 详情页逐字节一致快照，DataCurve 没有精确 Agent 行；本轮完成 HF 固定 revision/config、Qwen3-VL Technical Report、三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 curriculum、SAPO、Thinking with Images 和 tool-call reward 证据核验，并新增第二十一册第 92 章；K2 Horizon 3.7B、Qwen3-Omni、Grok 4.7 和 Claude Sonnet 5 作为前序历史记录保留。当前活动锚点见文件顶部和最后一条记录。

- 明确以新发布模型和公开排行榜作为资料更新入口。
- 建立模型记录模板、来源优先级和书系同步范围。
- 建立首批待核验模型候选清单，包含 GPT、Claude、Gemini、DeepSeek、Qwen、Kimi、GLM、Llama、Mistral、Grok 等系列。
- 已完成 GPT-6 Astra 与 GLM-5.3 两条可追溯专题闭环：官方文档摘记、正式章节、目录、百科、术语、题库、练习、论文、项目和知识图谱均已同步。
- 已核验新增章节的 Python 示例、AST、围栏配对、相对链接和 `git diff --check`；未公开架构、参数量、发布日期和 SAO 机制均保留为待核验。
- 已完成 K2 Horizon MoVA 36B/A4B 的榜单发现、官方模型卡/固定 revision 核验、第二十一册第 82 章和全局资料同步；K2-Horizon-7B-Uno 作为官方关联 adapter/论文技术记录，不作为排行榜新增模型。
- 已完成 Qwen3.8 的榜单归并、官方 27B/A95B/Flash-Next 模型卡、Flash-Next GitHub/技术报告和 Qwen Cloud 关系核验；新增研究笔记、第二十一册第 83 章及书系配套同步。Qwen3.8 当前状态为“内容专题闭环”。
- 已完成 Gemini 3.8 Flash 的两榜单归并、Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF 及 Gemini API 周边文档核验；新增研究笔记并完成索引/清单/计划同步。Gemini 3.8 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.8 Flash 2026-09-23 官方文档复验：通过 7890 获取模型页、Thinking、Interactions、Tool combination 和 Thought signatures；补齐 stable `1,048,576/65,536`、默认 medium、`max_output_tokens` 合计预算/`incomplete`、store/retention/delete、previous interaction 参数作用域、stateful/stateless replay、modality compatibility、tool call/result `id`、server/client-side tool 和 validated mode。Thinking 与 Tool combination 对标准 function call signature 的位置描述不完全一致，已保留为文档冲突并要求 capability probe。
- 已新增 [`gemini_interactions_replay_demo.py`](research/model-update-2026-09/code/gemini_interactions_replay_demo.py)，标准库合成验证 stateful/stateless、opaque signature、call/result id、SSE 事件顺序、store/delete/retention 和 modality gate；运行输出 `ok=true`、15 个有序事件、`network_called=false`。证据等级为 local protocol toy，不代表真实 Gemini API、模型质量或生产 SLA。
- 已完成 Gemini 3.7 Flash 的夜间断点联网复验：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card、DeepMind Research/可用 publications 入口和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点和计划同步。Gemini 3.7 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.6 Flash 的夜间断点恢复与资料级闭环：重新获取 Artificial Analysis 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.6 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.5 Flash 的断点恢复与资料级闭环：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 精确/全文检索；新增研究笔记并完成来源索引、模型盘点、榜单解释和计划同步。Gemini 3.5 Flash 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Gemini 3.5 Flash-Lite 的 AA 单榜资料级闭环：重新获取 Artificial Analysis、DataCurve、Google AI Developers 模型页、DeepMind Model Card/PDF、Thinking、Video understanding 和发布方评测；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划同步及书系配套。已确认默认 `minimal` thinking、agentic video 时间轴按需读取和 `processing_call/result` 边界；Model Card 明确基于 Gemini 3.1 Flash-Lite，不新增独立架构章节。
- 已完成 GPT-5.6 Sol/Terra/Luna 的两榜单归并、三个 OpenAI 官方模型页、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档和开发者博客核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 GPT-5.5/GPT-5.5 Pro 的两榜单锚点归并、OpenAI 两个官方模型页、GPT-5.5 专属指南及 Reasoning/Tools/Tool search/Prompt Caching/Compaction/Images/Conversation state/Background 文档核验；新增研究笔记并完成索引/清单/计划同步。GPT-5.5 当前状态为“资料级闭环”，暂无独立正式章节；GPT-5.5 Instant May/June 已补做 AA 详情复核和 OpenAI 精确路径负检索，仍只作为榜单级关联配置。
- 已完成 Claude Fable 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型页、发布/重新部署公告、API/Agent 文档和 system card 入口核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Fable 5 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Claude Opus 4.8 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 发布公告、System Card 入口和 Dynamic Workflows 官方博客核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Claude Opus 4.8 当前状态为“资料级闭环”，暂无独立架构章节。
- 已完成 Kimi K2.7 Code 的 Artificial Analysis/DataCurve 锚点归并、Moonshot/Kimi 官方资源/API 文档、固定 revision 模型卡、配置、许可证和部署指南核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Kimi K2.7 Code 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.5 的 Artificial Analysis/DataCurve 锚点归并、xAI 发布公告、Grok 4.5 官方模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档核验；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.6 的 Artificial Analysis/DataCurve 四档锚点归并、xAI 发布公告、Grok 4.6 官方模型页、Reasoning/Compaction/Tools 文档和论文精确标题检索；新增研究笔记并完成来源索引、模型盘点、榜单解释、计划和进度同步。Grok 4.6 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.20 0309 v2 的 Artificial Analysis 精确条目核验、DataCurve 精确行缺失核验、xAI 模型页/模型注册表、Reasoning/Multi Agent/Compaction/Tools/Release Notes 和 arXiv/新闻入口负检索；新增研究笔记并同步来源索引、模型盘点、榜单解释、计划和本进度表。Grok 4.20 当前状态为“AA 单榜资料级闭环”，暂无独立正式章节。
- 已完成 Claude Opus 5 的 Artificial Analysis/DataCurve 锚点归并、Anthropic 官方模型目录/专属页、完整开发者文档、发布公告、System Card 下载和 Anthropic Research/arXiv 定向检索；研究笔记、来源索引、模型盘点、候选解释、计划和进度已同步。已确认 adaptive thinking、五档 effort、工具/effort 中途变更、512-token 缓存门槛、fallback、1M context 和配置级评测边界；Claude Opus 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Fable 5.1 的断点恢复核验：重新抓取 Artificial Analysis、DataCurve、Anthropic 发布页和 System Card，补齐 Fable/Mythos safeguards 分层、cache read 定价、产品入口 effort、发布方 benchmark/安全摘要及 arXiv 外部使用检索；确认 DataCurve 当前没有 Fable 5.1 行。Claude Fable 5.1 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Claude Sonnet 5 的断点恢复核验：重新获取 Artificial Analysis、DataCurve、Anthropic 发布页、System Card、完整开发者文档、Research 页面和 arXiv 精确标题检索；补齐 adaptive thinking、effort/`max_tokens`、thinking block/signature、new tokenizer、context awareness、server-side compaction、computer toolset、programmatic tool calling、System Card/发布方评测及负面证据。Claude Sonnet 5 当前状态为“资料级闭环”，暂无独立正式章节。
- 已完成 Grok 4.7 的基础锚点核验：三条代理取得一致的 Artificial Analysis 中文首页、Grok 4.7 详情页和 DataCurve 快照；已核验 xAI 模型页、发布页、Reasoning、Context Compaction、Function Calling、Structured Outputs、Remote MCP，并记录更大 base、更长 RL、困难长任务混合、自验证与 safeguard stack 的发布方披露。已同步 encrypted reasoning state、opaque compaction、工具最小权限和发布方 benchmark 的证据边界；DataCurve 没有精确行，不迁移 Grok 4.6 结果。Grok 4.7 当前状态为“AA 单榜内容专题闭环”，暂无独立架构章节。
- 已完成 Qwen3-Omni 30B A3B 的当前锚点核验：三条代理取得一致的 Artificial Analysis instruct 详情页，并取得 reasoning 页面快照；已核验 Qwen3-Omni 官方 GitHub、HF Instruct/Thinking 配置、Qwen 博客、阿里云文档、arXiv `2509.17765` 论文与源码。已同步 Thinker-Talker、AuT、TM-RoPE、视觉 encoder、多码本 Talker/MTP、Code2Wav、异步 chunked prefill、三阶段训练和 Thinker/Talker 后训练的证据边界；DataCurve 没有精确行，不迁移其他 Qwen 结果。新增第二十一册第 91 章，Qwen3-Omni 当前状态为“AA 单榜内容专题闭环”。
- 已完成 DeepSeek V3.2 候选身份归并：`deepseek-v3-2-reasoning-0925` 和 `deepseek-v3-2-0925` 是 deprecated 的 V3.2-Exp revision，`deepseek-v3-2-reasoning`/`deepseek-v3-2` 是 deprecated 的 V3.2 配置，`deepseek-v3-2-speciale` 是同家族专项 checkpoint；DataCurve 没有精确 V3.2 行，因此不新增模型或迁移 V4/V3.1 Agent 分数。
- 已完成 DeepSeek V4 Flash Vision 的断点恢复核验与资料级闭环：重新抓取 Artificial Analysis、DataCurve 和 DeepSeek Quick Start；确认两个可用代理对 AA 返回相同完整页面，三个代理对 DataCurve/Quick Start 返回相同完整页面，7890 对 AA 大页面仍 TLS EOF。已核验历史实验多模态 API、当前旧 alias → V4.1-Flash 路由、图像预算、Files `file_id`、Responses 图像工具回灌和证据边界；第 85 章 demo AST、正常执行及 9 组边界测试通过。
- 已完成 GPT-5.4 的断点恢复核验与资料级闭环：通过三个代理重新抓取两个排行榜，并通过 7890 实际抓取 GPT-5.4 详情页、官方模型页和专属指南；确认三条线路对排行榜返回相同完整页面，AA 详情与既有哈希一致，DataCurve 仍无 GPT-5.4 Pro 或 mini/nano 精确模型 ID 行（`mini_swe_agent_gpt_5_4_xhigh` 是 base + harness 行）。已提取 1M context、deferred `tool_search`、computer use、native compaction、custom tools/CFG、`allowed_tools`、`phase` 和 Responses 状态等面试主线；暂无独立正式章节。
- 已完成 GPT-5.4 mini/nano 的独立资料级闭环：Artificial Analysis 精确条目、OpenAI 两个 sibling 模型页与 GPT-5.4 运行时指南已核验；mini/nano 各自的 snapshot、400K context、272K maximum input、128K maximum output、effort、定位、价格和工具清单已分开记录。DataCurve 只有 `mini_swe_agent_gpt_5_4_xhigh` 的 GPT-5.4 base 行，没有 mini/nano 精确模型 ID，未迁移 base 的 Agent 成绩；详见 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)。
- 已完成 GLM-5.2 官方博客补证和正式专题落地：正确路由为 `https://z.ai/blog/glm-5.2`，正文资源中的 IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 和 coding-agent anti-hack 已进入研究笔记与第二十一册第 86 章；SAO/compaction 原始定义、参数架构、完整训练 recipe、生产 kernel、硬件 profiling 和独立复现仍待核验。
- 已完成 GPT-6 Astra 运行时资料补证：通过 OpenAI 官方模型页、Reasoning、Prompt caching、Conversation state、Compaction 和 Tool search 文档，补齐动态 `configuration_update`、reasoning/`phase` item replay、deferred tool search、缓存前缀、canonical compaction context 和长任务状态账本；研究笔记与第六册第 18 章已同步。内部架构、参数、训练 recipe、system card 和专属技术报告仍待核验。
- 已完成 GPT-6 Astra 官方模型指南与 Agent 协议补证：通过 OpenAI 官方模型指南、Async tool calling、Mid-turn steering、Misalignment monitoring、Agents、Fast mode 文档和 `Rethinking skills and prompts for GPT-6 Astra` 博客，补齐应用侧异步 function/custom tool、`call_id`/job registry、WebSocket `response.steer`、steered continuation、异步安全告警/阻断、skills/`AGENTS.md` 渐进披露和完成定义等面试主线；已同步研究笔记、第六册第 18 章、第四册百科和全局配套。`phase` 当前专节以 GPT-5.5/GPT-5.4 为示例，已标明为通用 replay 规则，不写成 GPT-6 专属能力。
- 已完成 GPT-5.3 Codex 资料级闭环：Artificial Analysis 精确锚点、DataCurve 无精确行的负证据、OpenAI 官方模型页、Codex Prompting Guide、Compaction/Conversation state/Tools/Agents/Prompt caching 文档已核验；新增研究笔记、来源索引、模型盘点、榜单解释、计划/进度及书系配套。已确认 Responses-only、400K/272K/128K、reasoning effort、Codex harness、phase、完整 replay、compaction、工具契约和缓存前缀边界；参数、架构、训练 recipe、system card、专属报告、kernel 和线上 acceptance rate 仍待核验。
- 已完成 OpenAI gpt-oss 资料级闭环：从 Artificial Analysis 发现 `gpt-oss-120b` 与 `gpt-oss-20b`，确认 DataCurve 没有精确 `mini_swe_agent` 行；已核验 OpenAI Model Card/arXiv、官方 gpt-oss 仓库、Harmony、Hugging Face 模型卡/配置和 Cookbook。已新增第二十一册第 87 章并同步来源索引、模型盘点、榜单解释、计划、面试题、练习、术语、项目、论文、知识图谱及 Agent/Serving 相关章节；参数、完整训练 recipe、router 负载均衡、MXFP4 kernel、目标硬件 profiling、线上 acceptance rate 和独立 Agent 评测仍待核验。
- 已完成 Claude Opus 4.6 资料级闭环：从 Artificial Analysis 的 `adaptive` 与基础配置发现，确认 DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行；已核验 Anthropic 发布页、System Card、模型页、adaptive thinking、effort、thinking signature、server-side compaction、tool search、computer use、Advanced tool use 和 context engineering。已同步研究笔记、来源索引、模型盘点、计划及书系面试配套；参数、架构、完整训练/后训练 recipe、compaction 内部编码、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现仍待核验。
- 已完成 Claude Opus 4.7 内容专题闭环：从 Artificial Analysis 的 `claude-opus-4-7` 与 `claude-opus-4-7-non-reasoning` 配置发现，确认 DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行；已核验 Anthropic 发布页、System Card、模型页、`xhigh` effort、task budgets、更新 tokenizer、高分辨率视觉、compaction、tool search 和 cyber safeguards 控制面。已同步研究笔记、来源索引、模型盘点、计划、第二册 `7.28`、第二十四册 `32.36` 及书系配套；参数、架构、训练 recipe、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现仍待核验。
- 已完成 DeepSeek V4 Pro 0813 的双榜锚点核验：Artificial Analysis 精确条目为 `deepseek-v4-pro` 的 `Reasoning, Max Effort`，DataCurve 精确行为配置为 `mini_swe_agent_deepseek_v4_pro_max`；已补齐官方 V4 Pro GA、Quick Start、Responses/Thinking/Tool Calls、模型卡、配置与 `arXiv:2606.19348` 的证据链。已确认 `low/high/max`、Responses stateless、1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8、SFT+GRPO+on-policy distillation 和配置字段边界；复用既有第二十一册章节，不新增重复 Transformer 专题。
- 已完成 DeepSeek V4 Pro 官方实现证据补强：固定 HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`，记录 64 个 safetensors 分片的 metadata 边界、`config.json`、DSML encoding、inference config/model/kernel 的文件级哈希；补齐 gated compressor/overlap state、causal top-k indexer、128-token local window、前三层 hash routing、FP4/FP8 kernel、MTP、Hyper-Connections/Sinkhorn 和 serving manifest。完整权重未下载，未运行 CUDA/TileLang inference，`MP=8` 仍只是官方转换示例。

## 进行中

- 只从 DataCurve DeepSWE 与 Artificial Analysis（含 `https://artificialanalysis.ai/zh`）两个排行榜记录新进入者；其他网站不再作为新的模型发现入口。
- 对已由两个排行榜发现的候选模型寻找官方发布、论文/技术报告、模型卡、代码仓库和 API 文档，用于事实核验与周边技术扩展。
- 区分模型页、API 文档、开发者博客和榜单快照的证据责任，避免把产品/运行时字段写成内部训练事实。
- 上一轮活动锚点依次为 `DeepSeek V3.2`、`DeepSeek V4 Pro 0813`、`DeepSeek V4.1-Flash`、`Qwen3.8 Max (0902)`、`Claude Opus 4.7`、`Claude Fable 5.1`、`Claude Opus 5`、`GPT-6 Astra`、`Kimi K3`、`Gemini 3.8 Flash`、`Grok 4.6`、`Grok 4.7`、`Qwen3-Omni 30B A3B`、`K2 Horizon 3.7B` 和 `Qwen3-VL-235B-A22B`；对应的 reference implementation、`deepseek-recipe` 协议、QwenCloud runtime、Anthropic/OpenAI runtime、K3 stable source、Gemini Model Card、Grok runtime、Thinker-Talker、dense-vs-MoVA 和视觉多模态资料已分别收口。最近完成 GPT-5.6 Luna、Qwen3.7 Plus、DeepSeek V4.1-Flash stable release、GLM-5.3 标准 DSA、Claude Opus 5.5 和 GPT-6 Sol 的阶段性复验，当前活动锚点以文件顶部为准。
- 当前 `GPT-6 Sol` 为 AA 单榜资料级闭环；本轮确认 AA 详情、OpenAI 模型页及 Reasoning/Agents/Tools/Compaction 文档，DataCurve 没有精确 Agent 行。下一步只补 GPT-6 Sol 的完整训练/架构来源、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 证据，不迁移其他 GPT 的 Agent 结果。
- 对 K2 3.7B 继续补齐完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance、独立 benchmark 和精确 DataCurve Agent 行；对 K2 36B/A4B 的训练报告、Uno 接受率和目标硬件 profiling 的补证仍属于既有锚点后续，不产生新的模型候选入口。

## 待完成

- 对仍处于“仅候选”“部分覆盖”或“资料级闭环”的重点锚点继续补齐官方来源、研究笔记和专题内容。
- 对已闭环锚点继续补证完整 kernel、目标硬件 profiling、线上接受率、完整训练/后训练配方和独立 benchmark，不把资料缺失误写成已确认事实。
- 完成全仓库链接、版本、许可证、时间语境和 Markdown/Python 代码质量检查。
- 继续按可用代理复访官方页面，重新确认价格、版本快照、模型卡、许可证和技术报告；单个代理的 TLS/DNS 失败不能被写成资料不存在。

## 记录规则

事实必须有来源和核验日期；未确认内容标为“待核验”，不进入正式结论。每完成一个模型或一个专题，立即在本文件追加条目。

## 2026-09-20：DeepSeek V4 Pro 0813 内容专题闭环（上一锚点）

- Artificial Analysis 精确页为 [`deepseek-v4-pro`](https://artificialanalysis.ai/models/deepseek-v4-pro)，标题 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`；详情快照 3,934,926 bytes，SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`。AA 目录字段为 release date `2026-08-13`、Intelligence Index `35.9967791278402`、1M context、约 1.6T/49B total/active。
- DataCurve 精确行是 `mini_swe_agent_deepseek_v4_pro_max`：Pass@1 `62.831858%`、Pass@4 `88.495575%`、`n_runs=4`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` Agent steps。该结果绑定 max effort、mini-swe-agent、工具、任务集、环境和 verifier，不是裸模型分数。
- 官方闭环：V4 Pro GA 公告确认 2026-08-13、`low/high/max`、Responses API 和 Codex 优化；Responses 文档确认 stateless、function tools、`apply_patch`、并行调用和不支持参数可能静默忽略；模型卡/配置/技术报告支持 1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8、32T+、SFT+GRPO+on-policy distillation 及 61 层/384 routed/6 selected/1 shared 等实现字段。
- 当前闭环状态：**内容专题闭环（双榜锚点）**。待核验完整独立 benchmark、生产 kernel、目标硬件 profiling、线上 tool acceptance、完整 recipe 和 API 快照差异；后续把这些作为补证，不再另建重复章节或迁移其他 DeepSeek 配置的评测。

## GPT-5.5 Instant June 本轮收口

- AA 的 June/May 条目仍保留为 revision 关联配置；官方 OpenAI 模型目录确认 `gpt-5.5` 与 snapshot `gpt-5.5-2026-04-23`，精确 `gpt-5.5-instant` 页面返回 HTTP 404。
- DataCurve 没有 Instant 精确行，因此不迁移 GPT-5.5 base 的 context、tool catalog 或 DeepSWE 结果。Instant June 当前状态为**榜单级关联配置 + 官方身份负证据**，不新增专属 Transformer 章节。

## 2026-09-09：Artificial Analysis 首次采集

- 来源：https://artificialanalysis.ai/
- 已成功获取首页 HTML；原始快照暂存 `/tmp/artificialanalysis.html`（临时文件，不作为长期归档）。
- 页面文本出现 GPT-6、GPT-5.6、Claude Opus 5、Claude Sonnet 5、Claude Fable 5.1、DeepSeek V4 Pro 0813 等候选名称。当前只确认名称出现在页面中，尚未确认对应榜单记录、发布日期、可用性或官方技术说明。
- 同时出现 Claude Code 与不同模型的组合名称，后续必须区分基础模型、推理配置及 Agent 系统，不能混成同一模型榜单。
- 下一步：解析具体条目及评测方法，再逐项交叉核验官方发布、模型卡、论文和技术文档；不从排名反推训练方法。

## 2026-09-09：结构化候选与 Kimi K3 官方核验

- 已从 Artificial Analysis 页面结构化数据提取 2026-09-09 历史表中的 273 个候选配置条目，见 `research/model-update-2026-09/model-inventory.md`；不是 273 个独立基础模型，另有 2026-09-14 实时增量单独记录。
- Kimi K3、GLM-5.3、GLM-5.3-Flash 均在 Artificial Analysis 与 DeepSWE 快照中发现。
- 已读取 Kimi K3 官方发布正文与博客索引，记录架构、训练、部署、评测限制及八条知识扩写入口，见 `research/model-update-2026-09/kimi-k3-source-notes.md`。
- Z.ai 博客索引本次返回内容不足，未将其记为官方核验完成。
- 尚未更新正式章节；下一步继续定位技术报告及模型卡，并核对其他模型官方发布。

## 2026-09-09：写作要求与 GLM 来源状态

- 已将用户强调的篇幅、逐知识点展开、生动例子、定义、解释、代码和分层深度要求写入 `plan_v2.md`。
- Z.ai `/blog` 快照实际为 404 页面；本次访问 `/blog/glm-5.3` 未获得可读正文。两者都不能作为已核验官方发布的依据，后续转查官方模型文档、模型仓库及排行榜外链。
- 整体目标仍进行中，正式书籍章节扩写和配套同步尚未完成。

## 2026-09-09：GLM-5.3 官方文档已读取

- 成功读取官方模型文档，确认其披露沿用 GLM-5.2 基础模型、以环境扩展等后训练方法提升能力。
- 整理五个独立扩写方向及推理参数迁移差异，见 `research/model-update-2026-09/glm-5.3-source-notes.md`。
- SAO 算法定义、权重许可证、报告与评测复现仍待核验；未编造缺失机制。
- Kimi K3 发布文章的普通 HTML 锚点中未提取到论文、GitHub 或权重链接；仍需从官方研究与模型组织寻找。

## 2026-09-09：GPT-6 Astra 官方模型页已核验

- 已读取官方 Markdown 模型页，准确标识为 `gpt-6-astra`，整理输入输出模态、窗口预算、推理档位、工具支持及计费规则。
- 资料见 `research/model-update-2026-09/gpt-6-astra-source-notes.md`。
- 参数与训练机制未由该页披露，发布日期和技术报告待查；不将知识截止日期当成发布日期。

## 2026-09-09：Attention Residuals 原始资料

- 已读取 MoonshotAI 官方仓库 README 与 arXiv 摘要，核验论文首发日期 2026-03-16。
- 已记录深度注意力、可学习伪查询、块级聚合、张量维度和实验边界，见 `research/model-update-2026-09/attention-residuals-source-notes.md`。
- 已形成六步章节讲解设计；论文全文、教学实现与正式正文仍待完成。

## 2026-09-09：Attention Residuals 全文获取状态

- 已获取 arXiv HTML，但该版本生成失败，只有错误页和元信息；已确认 arXiv:2603.15031v1、提交日期 2026-03-16。
- 已下载论文 PDF（约 1.1 MB），当前环境没有 `pdftotext`，尚未完成正文提取；因此仍只依据官方仓库 README 和论文摘要记录机制，不扩写未核验的附录细节。
- 下一步可用 Python PDF 解析库或阅读器提取正文，再核对初始化、两阶段计算和实验设置。

- 已检查本地 PDF 解析能力：环境未提供 `pdftotext`、`pypdf`、`PyPDF2` 或 `fitz`；仅核验 PDF 元数据，未虚构正文提取结果。

## 2026-09-09：Attention Residuals LaTeX 源码核验完成

- 成功下载 arXiv e-print 源码并读取 `3-attnres.tex`、`4-infra.tex`，补齐 Full/Block AttnRes 定义、复杂度、跨 stage caching、两阶段推理和 online softmax 合并依据。
- 研究笔记已增加论文源码核验段落；现可据此设计教学公式和实现，但正式章节仍需独立写作、代码验证及与 Kimi K3 配置的边界说明。

## 2026-09-09：候选条目分层解释

- 已核对重点名称在候选表中的实际条目：GPT-6 Astra 多档 effort、Kimi K3 low/max、GLM-5.3 与 Flash、DeepSeek V4 Flash/Pro/Vision、多档 Claude Opus 5、Qwen3.8 多种规格及 Grok 4.6 多档 effort。
- 新增 `research/model-update-2026-09/inventory-interpretation.md`，正式采用基础模型、推理配置、Agent 系统三层记录规则，避免把排行榜行数当成独立模型数。

## 2026-09-09：Kimi Delta Attention 原始论文核验

- 从 arXiv 读取 `Kimi Linear` 论文摘要与 LaTeX 源码，确认 KDA 的递推、逐通道衰减、rank-1 delta 更新、DPLR 简化和 3:1 KDA/MLA 混合实验设计。
- 新增 `research/model-update-2026-09/kda-source-notes.md`；明确 Kimi Linear 的实验配置不能直接当作 Kimi K3 的完整架构事实。
- 后续将据论文机制设计独立教学章节和最小 recurrent demo，再决定同步到第二十一册、第二十四册与第四册百科。

## 2026-09-09：KDA 教学 demo 验证

- 新增零依赖教学代码 `research/model-update-2026-09/code/kda_recurrent_demo.py`，实现论文递推式的单头状态更新，不冒充 FlashKDA/chunkwise kernel。
- 当前环境未安装 PyTorch，因此先用纯 Python 验证输出与状态形状；运行通过。正式章节将同时提供 PyTorch 版本并解释教学实现与生产 kernel 的差距。

## 2026-09-09：GLM-5.2 前置资料核验

- 已读取 Z.ai 官方 GLM-5.2 文档，确认其长任务、1M 上下文、Agent 场景和产品工作流披露。
- 文档正文未给出 SAO 或 compaction 的算法定义；已新增 `research/model-update-2026-09/glm-5.2-source-notes.md`，明确不从使用示例反推内部训练机制。

## 2026-09-09：SAO/compaction 索引核验

- 已读取 Z.ai 官方 `llms.txt` 和 GLM-5.2 Markdown 文档版本；文档索引仅列出 GLM-5.2 页面，正文没有 SAO 或 compaction 算法定义。
- 当前证据只能支持“GLM-5.3 文档声称继承 GLM-5.2 的 SAO with compaction”，不能确认 SAO 全称、损失函数或压缩算法。已将该项保留为待核验，避免从缩写推断机制。

## 2026-09-09：DeepSeek V4 官方公告核验

- 已读取 DeepSeek 官方 V4 Pro GA、V4 Flash Vision Experimental 和 API Quick Start 页面。
- 已确认 V4 Pro 2026-08-13 公告、Flash Vision Experimental 2026-08-21 公告、API 别名与后端快照更新规则，以及 reasoning effort、Responses API、Codex 优化和多模态 Files API 等公开能力。
- 新增 `research/model-update-2026-09/deepseek-v4-source-notes.md`；参数规模、架构和训练方法仍待技术报告或模型卡核验。

## 2026-09-09：DeepSeek V4 模型卡与技术报告入口

- 读取官方 Hugging Face 模型卡，补齐 V4-Pro/Flash 参数量、激活参数、1M context、MoE、FP4/FP8、CSA/HCA、mHC、Muon、32T+预训练、SFT+GRPO+on-policy distillation 等披露。
- 已将这些信息与 V4 Preview 官方公告交叉记录；新增内容见 `deepseek-v4-source-notes.md`。
- 后续需通读 `arXiv:2606.19348`，核对公式、消融、压缩误差、mHC 与训练协议，暂不直接写入正式章节。

## 2026-09-09：DeepSeek V4 技术报告源码核验

- 成功下载并读取 `arXiv:2606.19348` LaTeX 源码，补齐 CSA/HCA 压缩与稀疏流程、mHC 双随机约束、GRPO+on-policy distillation 后训练、Muon/AdamW 分工、异构 KV cache 和工程优化依据。
- 已将指标基线、精度和压缩倍率的解释边界写入 `deepseek-v4-source-notes.md`；正式章节仍需单独写作、公式检查和教学代码验证。

## 2026-09-09：DeepSeek V4 书系映射

- 新增 `research/model-update-2026-09/deepseek-v4-book-mapping.md`，将 CSA、HCA、mHC、Muon、FP4/FP8 QAT、GRPO、on-policy distillation、异构 KV cache 与 1M context 评估分别映射到专题书和配套文件。
- 清单明确正式章节必须逐主题展开、提供例子/公式/代码/工程取舍和来源边界；当前尚未直接改动正式书稿，避免研究底稿未经完整核验进入正文。

## 2026-09-09：来源索引收口

- 新增 `research/model-update-2026-09/source-index.md`，统一记录 GPT-6 Astra、Kimi K3、GLM-5.3 和 DeepSeek V4 的排行榜发现来源、官方文档、模型卡、论文与待核验项。
- 当前四个重点系列均已有独立研究笔记；下一阶段应从“来源盘点”转入至少一个专题的正式章节扩写，并同步百科、题库、练习和术语。

## 2026-09-09：首篇正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/75-attention-residuals深度方向注意力.md`，覆盖问题例子、标准残差、Full/Block AttnRes、公式、零依赖 demo、工程边界、面试追问和练习。
- 已将第 75 章加入第二十一册 `目录.md`。
- 已完成本次文件广告注入检查；后续需运行 Markdown 数学渲染检查、代码测试，并同步第四册百科、面试题库、练习、术语和知识图谱。

## 2026-09-09：KDA 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/76-kda从delta规则到线性注意力.md`，覆盖问题场景、状态记忆直觉、KDA 递推、逐通道衰减、Delta/DPLR 关系、KDA/MLA 混合、零依赖代码、训练/推理差异、Kimi K3 证据边界、面试追问和练习。
- 已将第 76 章加入第二十一册 `目录.md`。
- 提取并运行第 76 章中的一个 Python 代码块，通过；此前尝试用 `runpy` 执行 Markdown 的检查命令已识别为错误，不影响正文代码。

## 2026-09-09：CSA/HCA 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md`，覆盖 KV 账本、CSA、HCA、滑动窗口、异构 cache、成本基线、教学代码、局限、面试题和练习。
- 已将第 77 章加入第二十一册目录；章节 Python 示例提取运行通过。

## 2026-09-09：mHC 正式专题落地

- 新增第二十一册正式章节 `book-21-transformer-architecture-evolution/chapters/78-mhc双随机残差连接.md`，覆盖 Hyper-Connections 背景、双随机矩阵定义、封闭性、Sinkhorn 投影、零依赖代码、与 AttnRes/KDA 区别、工程代价、面试追问和练习。
- 已将第 78 章加入目录；章节 Python 示例提取运行通过。

## 2026-09-09：纵向资料同步与代码复核

- 已将 AttnRes、KDA、CSA/HCA、mHC 的术语追加到 `GLOSSARY_EN_ZH.md`。
- 已将 7 道新面试题追加到 `INTERVIEW_BANK.md`，将 5 个新练习追加到 `EXERCISES.md`。
- 修复并复核第 75 章 RMSNorm 教学实现；修正 Block AttnRes 的退化表述；明确 mHC Sinkhorn 示例要求正矩阵输入。
- 提取并运行第二十一册第 75-78 章所有 Python fenced blocks，全部通过；检查未发现 `\\operatorname` 公式宏或广告注入文本。
- 仍待同步第四册百科、`KNOWLEDGE_GRAPH.md`、`PROJECTS.md`、`PAPERS.md` 和 `PROGRESS.md` 的正式索引，并完成 Markdown 数学渲染检查。

## 2026-09-09：论文、知识图谱与项目路线同步

- 已将 Kimi Linear、Attention Residuals、DeepSeek V4 技术报告加入 `PAPERS.md`。
- 已将 KDA、CSA/HCA、mHC、AttnRes、后训练合并和 Serving Cache 依赖关系加入 `KNOWLEDGE_GRAPH.md`。
- 已将 Million-Context Attention Budget Lab 与 Residual Flow Stability Lab 加入 `PROJECTS.md`。
- 本轮正式章节与纵向文件同步完成；仍需同步第四册百科条目、README/BOOK_SERIES/ROADMAP 的进度摘要，并执行全局链接和公式检查。

## 2026-09-09：同步检查结果

- PAPERS、KNOWLEDGE_GRAPH、PROJECTS、GLOSSARY、INTERVIEW_BANK、EXERCISES 的本轮新增内容均无 `\\operatorname`；注入防护关键词检查未发现异常文本。
- 第二十一册目录已包含第 75-78 章，来源索引和研究底稿均可追溯到官方页面/论文。
- PAPERS 中 DeepSeek V4 已有一条既存入口和一条专题入口，属于有意保留的不同上下文描述，后续可在总体验收时合并重复索引。
- 未同步 `book-04-llm-encyclopedia/`、README/BOOK_SERIES/ROADMAP/PROGRESS 等总控文件；这些是下一轮需要完成的收口项，不把当前轮标记为全目标完成。

## 2026-09-09：百科与总控文件收口

- 新增第四册百科条目 `book-04-llm-encyclopedia/chapters/19-frontier-architecture-updates.md`。
- 已同步 README、BOOK_SERIES、ROADMAP、PROGRESS 的新模型专题摘要和阅读路径。
- 本轮研究、正式章节、配套训练文件和总控入口已形成闭环；后续仍需检查链接可达性、目录索引和 PDF 构建结果。

## 2026-09-09：第四册目录索引修复

- 检查发现百科正文文件虽已创建，但未加入第四册 `目录.md`；现已补入第 19 项索引。
- 根目录摘要已包含新模型专题；总体验收继续检查链接、目录、公式与 PDF 构建。

## 2026-09-09：本轮链接检查

- 修正路径类型后重新检查新增章节、百科条目、plan_v2 和 progress_v2；本地相对链接缺失数为 0。
- 外部链接已保留来源 URL；网络可达性与 PDF 构建将在环境具备相应工具时继续验证。

## 2026-09-09：Qwen3.8 候选核验

- Artificial Analysis 候选表中发现 `Qwen3.8 2.4T A95B`、`Qwen3.8 27B`、`Qwen3.8 Max` 和 `Qwen3.8-Flash-Next` 等配置。
- Qwen 官方旧博客首页已重定向到 `qwen.ai/research`；Qwen Studio 动态页面只暴露 `/blog?id=qwen3.8` 路由，未返回可读的发布正文、模型卡或技术报告。
- 已将 Qwen3.8 保留为“排行榜候选发现”，未把名称、日期或架构写成正式事实；状态与来源已加入 `inventory-interpretation.md` 和 `source-index.md`。

## 2026-09-09：DeepSeek V4 后训练章节落地

- 新增第五册正式章节 `book-05-llm-training/chapters/18-deepseek-v4领域专家与统一蒸馏.md`，逐项展开领域专家 SFT、GRPO 组内优势、verifier 与权限边界、on-policy distillation、reverse KL、教师路由、成本和失败模式。
- 已将第 18 章加入第五册目录；章节标注了 DeepSeek V4 报告披露范围，未虚构奖励权重、教师数量或完整 loss。
- 本章无 Python fenced block；已检查公式宏和注入防护关键词，未发现异常文本。
- 已补充零依赖 GRPO 组内优势代码示例，并提取运行通过；章节状态从“无代码示例”修正为包含一个教学实现。

## 2026-09-09：GPT-6 Astra 章节验证与配套同步

- 提取并运行第六册第 18 章唯一 Python fenced block，输出 `5.7`，与示例中的 300K 输入、100K 缓存命中、20K 输出和阈值倍率一致。
- 检查该章未发现 `\\operatorname`、异常广告/注入关键词或缺失本地相对链接；官方页本轮因 DNS 不可用无法复访，事实以已保存的 2026-09-09 官方模型页摘记为准。
- 已把 GPT-6 Astra 的已确认字段同步到第四册百科、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和 `PROJECTS.md`。
- 同步内容严格限制在 `gpt-6-astra` 模型 ID、文本/图像输入、文本输出、窗口/输入/输出预算、reasoning effort、工具/端点和价格阈值；参数量、架构、训练方法、发布日期和技术报告继续标为待核验。

## 2026-09-09：GLM-5.3 长任务章节与配套同步

- 新增第十六册第 20 章 `book-16-reasoning-models/chapters/20-glm-5.3长任务环境与验证器.md`，逐项展开可执行环境、任务契约、oracle/no-op/unsolved-state、奖励捷径、上下文压缩、协议迁移和证据边界；零依赖 verifier demo 已提取运行通过，输出与正文一致。
- 已将第 20 章加入第十六册目录，并同步第四册百科、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和 `PROJECTS.md`。
- 正文只使用 Z.ai 官方 GLM-5.3/5.2 文档已确认的版本关系、接口字段和高层训练流程；`SAO with compaction` 的全称、损失和压缩机制继续标记为待核验。
- 第 20 章检查通过：Python fenced block 运行输出为 `incomplete: throughput target not met | success | invalid: protected verifier changed`；未发现 `\\operatorname`、异常广告/注入关键词或缺失本地相对链接，`git diff --check` 通过。

## 2026-09-09：Kimi K3 发布证据与 Harness 章节

- 新增第十七册第 15 章 `book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md`，展开发布文章与论文的证据分层、`R=F(M,H,E,B,D)`、思考状态清单、跨模型迁移、harness-aware evaluation 和 benchmark 读法。
- 已将第 15 章加入目录，并同步第四册百科、术语、题库、练习、论文、项目和知识图谱。
- 正文不把 Kimi K3 发布文章中的参数、权重承诺、Stable LatentMoE、量化或评测数字当作独立复现事实；KDA 与 AttnRes 仅用对应论文解释一般机制，K3 的完整配置和权重状态继续待核验。
- 已同步 README、`BOOK_SERIES.md`、`ROADMAP.md` 和 `PROGRESS.md` 的新模型阅读入口与完成状态；全目标仍保持 active，候选模型的后续官方核验未结束。
- Kimi K3 章节新增零依赖评测可比性 demo，输出 `0.1 0.5 False`，验证同 harness/硬件/effort/任务集才计算绝对与相对提升；Python AST、实际运行、链接检查和 `git diff --check` 均通过。

## 2026-09-09：Mistral Small 4 与 Step 3.5 Flash 专题落地

- 新增第二十一册第 79 章 `Mistral Small 4：混合推理与 EAGLE/NVFP4 部署`，覆盖模型卡公开的 119B/约 6.5B MoE、128 experts/4 active、256K、多模态输入、`reasoning_effort`、EAGLE、NVFP4、许可证和部署边界。
- 新增第二十一册第 80 章 `Step 3.5 Flash：MTP-3、滑动窗口和 11B 激活参数`，覆盖约 196.81B/约 11B active、45 层、288 routed + 1 shared、top-8、MTP-3、3:1 SWA/full attention、Context Manager 和评测协议边界。
- 已将两章加入目录，并同步第四册百科、术语、题库、练习、项目、论文、知识图谱、README、`BOOK_SERIES.md` 和 `ROADMAP.md`；来源索引新增对应模型卡与技术报告入口，重复的 Qwen3.8 状态已清理。
- 两章示例均已通过 Python AST 和实际运行检查；本轮对 26 个改动/新增 Markdown 文件完成本地相对链接检查，新增章节的数学/代码围栏配对检查和 `git diff --check` 通过。模型卡 benchmark、吞吐、后端 MTP 支持、完整训练配方与发布日期含义仍需绑定 revision、硬件和后端继续核验；当前环境未提供 `pandoc`、`xelatex`、`wkhtmltopdf` 或 `typst`，PDF 构建留待环境具备工具后复查。

## 2026-09-10：Claude Opus 5 官方模型目录核验

- Anthropic 官方模型目录缓存页已提供 `claude-opus-5` 的一手接口字段：1M context、128K 最大输出、300K batch 最大输出、adaptive thinking、默认 high effort、平台、价格和 2026-05 知识/训练截止字段；页面还给出 `2026-07-24` 发布日期字段。
- 已新增 `research/model-update-2026-09/claude-opus-5-source-notes.md`，并完成来源索引、候选解释、第四册百科、术语、题库、练习、论文、知识图谱、项目和总控文件同步。
- 仍不把参数量、MoE/稠密结构、训练数据、后训练算法、完整推理机制或独立 benchmark 复现写成事实；`adaptive`、effort 与榜单档位均按运行时配置处理。
- 网络复访当前受 DNS 影响；system card、announcement 和真实 API 行为留作下一轮逐页核验，整体新模型发现与官方核验目标保持 active。

## 2026-09-10：Claude Fable 5.1 官方模型页核验

- Anthropic Fable 5.1 专属模型页缓存已提供 `claude-fable-5-1` 的一手字段：2026-09-01 发布、1M context、128K 最大输出、adaptive always-on、默认 high effort、平台、价格和 2026-06 知识/训练截止字段。
- 页面自述的长任务、研究、文档处理优势，以及 preserved thinking、跨轮模型切换、per-message effort、turn-scoped system messages、工具间进度更新等 beta 能力已记录，但没有升级为独立 benchmark 或内部算法事实。
- 已新增 `research/model-update-2026-09/claude-fable-5.1-source-notes.md`，完成来源索引、候选解释、第四册百科、术语、题库、练习、论文、知识图谱、项目和总控文件同步。
- Fable 参数量、架构、训练/后训练细节和独立复现仍待核验；网络复访与关联 announcement/system card 逐页复核留待下一轮，整体目标保持 active。

## 2026-09-10：Claude Sonnet 5 官方模型目录核验

- 从 Anthropic 官方模型目录缓存页核验 `claude-sonnet-5`、2026-06-30 发布字段、1M context、128K 普通最大输出、300K batch 最大输出、Adaptive thinking、默认 high effort、Fast latency 字段、平台、价格和 2026-01 知识/训练截止字段。
- 已新增 `research/model-update-2026-09/claude-sonnet-5-source-notes.md`，并将模型字段同步到来源索引、候选解释、第四册百科、术语、面试题、练习、项目、论文、知识图谱以及 README、`BOOK_SERIES.md`、`ROADMAP.md`、`PROGRESS.md`、`progress_v2.md` 和 `plan_v2.md`。
- 证据边界保持严格：模型目录只支持接口、平台和运行时字段，不支持参数量、稠密/MoE 架构、训练配方、后训练算法、完整推理机制或独立 benchmark 复现；价格和延迟字段仍需按目标平台实时复核。
- 网络复访仍受 DNS 影响；system card、announcement、跨平台差异和真实 API 行为留作后续逐页核验，整体新模型发现与官方核验目标保持进行中。

## 2026-09-10：官方页面复访（DNS 重试）

- 再次尝试访问 Anthropic Opus 5 announcement/system card、Opus/Fable 模型页，以及 Qwen3.8 官方博客/研究入口；`www.anthropic.com`、`platform.claude.com`、`qwen.ai` 和 `qwenlm.github.io` 均因 DNS 解析失败返回 curl 退出码 6。
- 本次没有新增一手事实；Opus/Fable 的关联 announcement/system card 与 Qwen3.8 的模型卡/技术报告继续保持“待核验”，不把网络失败当作资料不存在。

## 2026-09-10：新增专题代码离线验收

- 重新提取并运行 Mistral Small 4、Step 3.5 Flash、Attention Residuals、KDA、CSA/HCA、mHC、DeepSeek V4、GPT-6 Astra、GLM-5.3 和 Kimi K3 章节中的 10 个 Python fenced blocks；全部通过 AST 解析和独立运行，无失败样例。
- 新增章节与总控文件的 `git diff --check`、实际本地相对链接检查和 Markdown 围栏配对检查均通过；外部页面因 DNS 不可用未将访问失败写成资料不存在。

## 2026-09-10：DeepSWE v1.1 快照结构化

- 读取本地缓存 `/tmp/deepswe.html`，核验页面标注的 2026-09-03 更新时间、v1.1、113 个任务、91 个仓库、5 种语言、统一 `mini-swe-agent` harness，以及主表 21 个可见配置的 Pass@1、区间、平均成本、输出 token 和 Agent steps。
- 新增 `research/model-update-2026-09/deepswe-snapshot-notes.md`，并同步来源索引、候选解释和计划入口；结果明确归因于模型配置 + Agent harness + 工具/verifier，不把 DeepSWE 分数当作基础模型能力或实时榜单。
- 完整 model revision、供应商端点、系统提示、工具 schema、重试策略、硬件和独立复现仍待核验；网络复访失败时继续使用带日期的快照边界。

## 2026-09-10：排行榜与官方页面刷新尝试

- 尝试刷新 Artificial Analysis、DeepSWE、Anthropic Fable 页面和 Fable 发布入口，当前环境四个域名均因 DNS 解析失败返回 curl 退出码 6；因此候选盘点继续使用 2026-09-09 已保存的榜单快照，并明确其采集日期，不把它描述为 2026-09-10 实时排行榜。

## 2026-09-10：第六册第 4 章解码策略收口

- 第六册第 4 章 `book-06-llm-deployment/chapters/04-解码策略与生成控制.md` 已完成中断处的第二轮收口，当前 828 行、47080 字节；补充 speculative decoding 残差分母为零、PassRate 无有效 run 的 `not_applicable` 语义。
- 零依赖 logits demo 拒绝空/非有限 logits、非法 temperature、top-k/top-p/min-p、负 penalty、未知历史 token、越界采样索引和非法 step；内置 6 个非法配置回归，额外通过 10 个边界输入测试，预期输出同步 `invalid_cases`。
- 本轮对第 4 章及此前 10 个专题章节的 11 个 Python 围栏完成 AST 与独立运行；GenerationConfig、vLLM sampling/structured outputs 与 speculative decoding 论文在线复访因 DNS 解析失败，继续按待核验处理。

## 2026-09-11：第二十册第 5 章文件编辑审计边界收口

- 继续会话 `01a0843d-2910-7993-840a-8587cea054bd`，在 `book-20-agent-harness-runtime/chapters/05-文件系统与代码编辑.md` 增加空集合/空分母回归：`safe_div(1, 0)`、`safe_mean([])`、`rounded(None)` 为 `None`，比较门禁对 `None` 一律返回失败，避免无样本伪造 100% 通过。
- 正文新增 `not_applicable` 语义说明，明确不使用 epsilon、历史分数或默认 `1.0` 掩盖零分母；需补样本或由上层策略显式跳过整组评估。
- 章节 1106 行、35957 字节；Python fenced block 已通过 AST、独立执行和边界断言，正常 demo 输出与既定失败门禁保持一致。下一步继续第二十册第 6 章“终端执行与命令安全”。

## 2026-09-11：第二十册第 6-8 章审计指标边界收口

- 第 6 章终端执行 demo 去除 `safe_div` 默认 `1.0`，补充 `safe_mean`、空分母回归和 `not_applicable_metrics`；第 6 章当前 1111 行、37667 字节，AST、独立执行和边界断言通过，危险命令放行/网络未控/超时未取消/trace 缺失坏例保持使门禁失败。
- 第 7 章上下文 demo 将 `stale_summary_rate` 等空分母指标改为 `None` 语义，补充空集合回归与显式门禁比较；当前 1094 行、35067 字节，AST、独立执行和边界断言通过，缺失关键上下文、过期摘要、信号丢失和注入边界坏例保持可见。
- 第 8 章权限沙箱 demo 将空集合默认 1.0 改为 `None`，新增 `not_applicable_metrics` 和阈值安全比较；当前 1116 行、33545 字节，AST、独立执行和边界断言通过，矩阵缺行、网络外发、沙箱缺失及 dry run 缺失坏例保持可见。
- 第 5-8 章入口已在第二十册目录确认；下一步继续第 9 章 trace、日志回放与可观测性，并在阶段收口重跑全库围栏、相对链接和 `git diff --check`。

## 2026-09-11：第二十册第 9 章 Trace/Replay 指标边界收口

- 第 9 章 trace/replay demo 增加 `mean_defined` 与空集合回归：无 artifact 需求或无有效事件时返回 `None/not_applicable`，不再用 `mean([])=1.0` 伪造完整覆盖率；门禁对 `None` 显式失败。
- `artifact_reference_missing` 根因判断改为仅针对有定义且低于 1 的指标；唯一 Python 围栏通过 AST、独立运行和边界断言，既有 span 树、时间线、版本、隐私、回放和最终状态坏例保持可见。
- 章节当前 1246 行、37444 字节；下一步继续第二十册第 10 章 Evaluation Harness，并在阶段收口重跑全库检查。

## 2026-09-11：第二十册第 10 章 Evaluation Harness 指标边界收口

- 第 10 章 demo 将 `mean([])` 改为 `None`，`weighted_mean` 对零总权重返回 `None`，新增空集合回归、`at_least`/`at_most` 门禁和 `not_applicable_metrics` 输出；无有效 run 不再伪造成功率或安全零率。
- 唯一 Python 围栏通过 AST、独立运行和边界断言；既有环境/验收器/公平比较/flaky/安全执行/报告缺失坏例保持可见，`evaluation_harness_gate_pass=False`。
- 章节当前 1211 行、33865 字节；下一步扫描第二十册剩余章节并继续修复同类指标定义域问题。
- 2026-09-11：第二十册第 11、12、14、15、16 章统一收口空分母指标：`ratio`、`rate`、`avg` 对空样本返回 `None`，第 14 章 `a2a_lifecycle` 空集合不再除零；五章加入 `at_least`、`not_applicable_metrics` 与空集合回归。五个 Python fenced blocks 均通过 AST、独立运行和边界断言；第 17-22 章未发现同类评估代码或定义域问题。

## 2026-09-11：第二十册指标公式定义域复核

- 复读第 11、12、14、15、16 章指标公式，补充统一非空定义域：空集合或零分母均为 `not_applicable`（代码 `None`），不得用 epsilon、历史值、默认值或人为下限分母伪造指标；门禁显式拒绝 `None`。
- 第 14 章高风险覆盖率改为真实风险样本数分母，空风险集合不适用；A2A lifecycle 明确要求非空 agent 和状态需求集合。其余四章补齐 `N`、分组集合和必需 trace 集合的定义域说明。
- 验证结果：第二十册 16 个 Python 围栏 AST/独立运行全通过；全库 659 个 Markdown 文件的 9622 对围栏配对通过，1290 个 Python 围栏 AST 全通过；`git diff --check` 通过。链接扫描保留既有代码样式误报，不修改无关内容。

## 2026-09-14：第二十册剩余指标定义域收口

- 第 13 章横向 Coding Agent 审计把 `C_risk` 改为真实 `|R_i|` 分母（要求 `|R_i|>0`），所有覆盖率空需求集合返回 `None/not_applicable`；总分在加权项无定义时返回 `None`，权限/评估/风险门禁显式拒绝 `None`，并保留空集合回归断言。
- 第 17 章恢复成功率、重复副作用率，第 18 章上下文折叠约束损失，第 21 章 workspace 可靠性均改为严格正分母公式，正文明确无样本时报告 `not_applicable`，禁止用 `max(1, …)` 伪造 0/1 分数。
- 第二十册 16 个 Python 围栏 AST 与独立运行通过；全库 659 个 Markdown、9622 对围栏配对、1290 个 Python 围栏 AST 通过；`git diff --check` 通过。全库逐块独立运行仍包含既有“片段依赖前文变量”的教学代码，已单独验证本轮受影响的第 13 章完整 demo。
- 2026-09-14 官方 OpenAI 模型页复访因 DNS 解析失败（curl 退出码 6）；未新增一手事实，GPT-6、Kimi K3、GLM-5.3 等待核验边界保持不变。
- 同日复读本地保存的 Z.ai GLM-5.3 官方文档快照，补记其发布方自报的 Terminal-Bench 3.0（4.6→28.3）、DeepSWE v1.1（46.2→66.9）和 Agents' Last Exam（23.8→28.5）前后数值，以及私有 Z.ai Code Bench 的双指标描述；已明确标注为页面快照/发布方自报，未升级为独立复现或基础模型单独增益。
- 复读本地 Kimi K3 官方发布页缓存，补记 Availability 与评测脚注：Kimi Work 3.1.0+、Kimi Code `/model`、API `kimi-k3`、$0.30/$3/$15 每百万 token 价格、Mooncake 分离式推理与编码工作负载缓存命中率“超过 90%”的发布方自报，以及 `max`、temperature=1.0、top-p=1.0 和按基准切换 Kimi Code/Claude Code/Codex harness 的条件。均绑定发布文章快照，未升级为长期费率、普遍性能保证或独立复现。
- 复读本地 DeepSWE v1.1 HTML 快照的数据对象，补记精确生成时间 `2026-09-03T22:24:37.984682+00:00`、统一 `mini-swe-agent` 仓库链接、GPT-6 Astra 最近作业时间及 Kimi K3 的 309/451、Pass@4≈89.4%、4 次重复运行和约 ±4.5% 区间字段；强调表格展示值经过四舍五入，引用时应保留分子/分母、重复次数和区间。

## 2026-09-14：第七册专项 Frontier 评测指标定义域复核

- `book-07-evaluation-experiments/chapters/15-specialized-frontier-eval-cluster.md` 的隐藏测试通过率从 `h/max(1,n)` 改为 `h/n, n>0`，代码新增统一 `ratio` 函数；空隐藏测试集合返回 `None/not_applicable`，trace 覆盖率和平均成本在空任务集上也不伪造数值。
- 补充 `ratio(1, 0)`、`ratio(0, 0)` 回归断言，并使空任务集进入 `remeasure_before_comparison` 门禁；该章 Python demo 已通过 AST、独立运行和 `git diff --check`。
- 额外按反引号与波浪号两种 Markdown 围栏重新扫描全库：1,698 个 Python 围栏全部通过 AST，未发现未闭合围栏；此前 1,290 个反引号 Python 围栏统计仍保留作历史可比口径。

## 2026-09-14：Claude Haiku 4.5 官方目录快照补充

- 从本地 Anthropic Models Overview 快照结构化字段核验 Claude Haiku 4.5：`claude-haiku-4-5-20251001`、别名 `claude-haiku-4-5`、2025-10-15 发布字段、200K context、64K 最大输出、extended thinking、`fastest` 延迟字段、平台 ID、价格及 2025-02/2025-07 cutoff。
- 新增 `research/model-update-2026-09/claude-haiku-4.5-source-notes.md`，并同步来源索引、候选解释、模型百科、术语、面试题、练习、论文、知识图谱和项目审计器。未读取的 announcement/system card、参数规模、训练架构、完整推理机制和独立 benchmark 继续标为待核验。
- Haiku 4.5 未出现在 2026-09-09 Artificial Analysis 采集切片中，因此不补写榜单日期；明确区分“官方目录发现”和“排行榜发现”。
- 已将 Haiku 4.5 入口同步到 `plan_v2.md`、`README.md`、`BOOK_SERIES.md` 和 `ROADMAP.md`，并完成受影响 Markdown 链接检查。

## 2026-09-14：Artificial Analysis 快照字段边界补强

- 从本地 `/tmp/artificialanalysis.html` 提取并复核 2026-09-09 快照中的配置级 Artificial Analysis Intelligence Index 示例：Fable 5.1 max with fallback 约 53.37、GPT-6 Astra max 约 52.81、Opus 5 max 约 50.70、GLM-5.3 max 约 44.86、Grok 4.6 high 约 44.41、Kimi K3 max 约 43.78、Gemini 3.8 Flash high 约 41.19。
- 已在 `inventory-interpretation.md` 和 `source-index.md` 标注：这些是第三方榜单快照配置，`releaseDate`/`parameters`/开放性字段不能替代官方发布日期、参数或许可证；指数不能与 DeepSWE 或其他 harness 分数直接合并。
- 已将该快照解释入口加入 `plan_v2.md`，后续引用排行榜时统一保留采集日期、effort/fallback、模型 revision 和 harness 条件。
- 快照还显示 GLM-5.3/Kimi K3 的第三方 `parameters` 字段分别为 753/2800；已明确标注为目录估算或归档字段，不升级为官方参数规模或许可证事实。
- 复读本地 Z.AI `llms.txt` 官方文档索引，确认 GLM-5.3-Flash 专属文档、迁移指南和 coding-agent 接入页面入口；未将索引摘要升级为 Flash 的参数、模态、发布日期或 benchmark 事实。

## 2026-09-14：模型候选盘点表结构审计

- 对 `research/model-update-2026-09/model-inventory.md` 的 2026-09-09 历史 Markdown 表执行结构检查：273 条候选记录、273 个唯一名称、273 个唯一链接，日期范围为 2026-01-04 至 2026-09-07；2026-09-14 实时增量另列 8 个 canonical 条目，未覆盖历史表。
- 该统计只证明候选发现表的结构完整，不改变“榜单日期/条目需要回到官方资料核验”的证据边界。

## 2026-09-14：DeepSeek-R1-0528 官方发布页补充

- 从本地 DeepSeek API Docs 快照 `/tmp/deepseek-news.html` 核验 DeepSeek-R1-0528 发布页：页面标注 2025/05/28，明确写出 JSON output、function calling、API 使用方式不变，并提供开源权重入口。
- 新增 `research/model-update-2026-09/deepseek-r1-0528-source-notes.md`，同步来源索引、候选解释和模型盘点；benchmark 图片、参数、架构、训练配方、许可证和独立复现继续标为待核验。
- 该模型未出现在 2026 Artificial Analysis 候选表切片中，因此不伪造榜单日期或第三方分数。
- 已为 Artificial Analysis 快照记录文件大小、mtime 和 SHA-256，并为 Anthropic 模型目录快照记录文件大小与 SHA-256，便于网络恢复后的内容漂移比对。

## 2026-09-14：DeepSeek V4.1-Flash 官方核验与 Artificial Analysis 实时刷新

- 已使用可用网页代理 `10.24.27.134:7890` 复访 Artificial Analysis，独立保存 `/tmp/proxy-live-artificialanalysis.html`：1,771,961 bytes，SHA-256 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。结构化数据约 702 个配置条目、673 个唯一名称，最新日期为 2026-09-11。
- 相对 2026-09-09 快照新增 13 个结构 slug/别名：DeepSeek V4.1 Flash、Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon MoVA/7B/3.7B/0.9B 的成对配置、`mbzuai` 目录项和 DeepSeek V4 Flash Non-reasoning。新增独立文件 `artificial-analysis-2026-09-14-snapshot.md`，没有覆盖旧候选表或伪造榜单日期。
- DeepSeek 官方发布页 `news260910` 返回 HTTP 200，页面标注 2026/09/10；已核验 API 名 `deepseek-flash`、旧 alias 临时路由以及 2026-09-14 04:00 UTC 起 `deepseek-v4-pro` 路由说明。发布页快照 SHA-256 为 `420cbb7b5e8e97632fa45cb49cd2b5f22b57c8f9e125d1c34a22bd67bbc33705`。
- Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash` 模型卡固定 revision 为 `dba1be0a40aa45a94ad051997016db3960a90277`；已核验 552B backbone、1M context、20 层 causal encoder + 20 层 decoder 的 CED、8B/16B prefill/decode active、SWA Bounded Replay、CSA2 `Full/Reindex/Reuse`、Hierarchical Sparse Indexer、FP4 main KV/890 bytes global KV token、384 routed + 1 shared/6 routed per token、196B Engram、DSpark、DeepSeek-ViT、45T token、64K 到 1M 扩展、`SFT -> RL -> OPD` 和数值 reasoning effort。模型卡 README SHA-256 为 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251`。
- 已读取固定 `config.json`、encoding README、evaluation README；补记 hidden size 5120、40 层、64 attention heads、1 KV head、head dim 512、YaRN factor 16、`sliding_window=128`、indexer/candidate/Engram/DSpark 字段，以及 DSML 前导空格、`low/high/max=50/75/100`、中途 system message、`mini-swe-agent` 复现入口。配置、encoding、技术报告哈希已写入来源笔记。
- 已下载技术报告 PDF，SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`；环境无 `pdftotext`/`pypdf` 等工具，本轮只做下载与哈希核验，不声称已逐页读取报告正文。
- 新增第二十一册第 81 章，更新目录和第四册第 19 章；同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`plan_v2.md`、来源索引、候选解释和历史模型盘点说明。
- 模型卡自报 base/Agent 数字均保留 benchmark、effort、temperature/top-p、harness、工具、环境和 verifier 条件；未把 DeepSWE 74.2 等组合结果归因给基础模型。章节 demo 和全局格式、链接、AST 检查仍需在本轮末执行，完整 kernel、API 价格图片、线上路由实测和独立 benchmark 继续待核验。

## 2026-09-14：DeepSeek V4.1-Flash 技术报告逐页核验

- 通过 `10.24.27.134:7890` 重新获取固定 revision 的技术报告，51 页逐页文本提取成功；哈希仍为 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`，并记录 PDF 书签与页码范围。
- 报告补充并已写入研究笔记/第 81 章：CED 的 `O(NL/2+n_win*L/2)` 复杂度、CSA2 的 `m=2/m=1` 层分组、`2048*8=16,384` HSI 候选池、Single-Pass mHC traffic、Engram 两模块设置、DSpark 五位置草稿器、RoPE 后 FP4 QAT、EPD 解耦、SWA 的 10% host-DRAM 短 TTL 池、45T/100.6M-token 训练设置和异步 RL/OPD 机制。
- 报告的 Table 1/3、effort 曲线、Dsec 容器密度和 cache 生命周期均保留为发布方内部设置与自报结果；完整 kernel source、所有参数分片、线上接受率、目标硬件 profiling、API 价格/限流和独立 benchmark 仍待核验。
- 清理第 81 章中由区间读取造成的重复段落，更新 `source-index.md`、`PAPERS.md`、`BOOK_SERIES.md`、`KNOWLEDGE_GRAPH.md` 和 `plan_v2.md` 的报告状态；全库结构检查随后执行。

## 2026-09-14：K2 Horizon MoVA 36B/A4B 与 Uno 周边技术

- K2 Horizon MoVA 36B/A4B 由 Artificial Analysis 榜单发现；DataCurve DeepSWE 本地 v1.1 快照未检出 K2，因此只把 Artificial Analysis 记为本轮 K2 的发现证据。
- 已读取 IFM 官方模型卡、固定 revision `de2d2efb32ed7639b7140bccbefe131a0063a982` 的配置与实现，以及 SGLang 部署参考；核验 36B/约 4B active proxy、48 层、前 3 层 dense、后 45 层 MoVA + MoE、32Q/8KV GQA、64 value experts/top-4、100 routed FFN experts/top-8 + 1 shared expert、524,288 context 和路由语义。
- 新增第二十一册第 82 章，并同步第四册第 19 章、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md` 和本计划/进度文件。
- `K2-Horizon-7B-Uno` 仅作为官方关联 adapter/论文技术记录：冻结 K2 7B AR base、LoRA diffusion draft 和 `Psi-Spec` AR rejection verification；它不是排行榜新增模型，也不是 36B 架构变体。0.9B 卡片的 MOPD 只保留为同系列训练流程边界，不能反推 36B recipe。
- 完整训练报告、MoVA/FFN 生产 kernel、Uno 接受率、目标硬件 profiling 和独立 benchmark 仍待核验；本轮新增资料不下载几十 GB 权重，正式评测需固定 revision、后端、硬件和 harness。

## 2026-09-14：两个排行榜代理复访与热门模型状态核对

- 通过用户提供的 `10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 三个代理访问 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)，六次请求均返回 HTTP 200；这只证明当前入口可读取，不改变榜单字段的第三方证据等级。
- 本次保存 Artificial Analysis `/zh` 响应到 `/tmp/artificialanalysis-live-2026-09-14.html`：1,771,203 bytes，SHA-256 `510bb1916a029700fbd78084a4b26cbe1e2b2690ce9e25179fde2d4fe650a7d4`。与此前 `/tmp/proxy-live-artificialanalysis.html` 的规范化 release 记录比较，已跟踪模型没有新增或删除；页面仍包含 GPT-6 Astra（2026-09-03）、Claude Fable 5.1（2026-09-01）、GLM-5.3（2026-08-18）和 GLM-5.3-Flash（2026-08-26）。
- 本次保存 DataCurve 响应到 `/tmp/deepswe-live-2026-09-14.html`：268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，与既有 `/tmp/deepswe.html` 字节一致；页面更新时间仍为 2026-09-03，数据对象生成时间仍为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库规模仍为 113/91。当前可见模型行包括 GPT-6 Astra、GLM-5.3 和 Claude Fable 5，不包含 Claude Fable 5.1。
- 因此 GPT-6 Astra、Claude Fable 5.1 和 GLM-5.3 的“榜单发现 + 资料入库 + 书系配套同步”已完成；但三者的参数/内部架构/完整训练配方/独立复现并未全部公开核验。Fable 5.1 当前仍是官方接口字段与产品定位级核验，GPT-6 的训练架构和发布日期、GLM-5.3 的 SAO with compaction 原始定义与模型卡仍列为待补证；本轮没有把“已更新”写成“全部技术细节已确认”。
- 同步修正 `source-index.md`：Fable 5.1 保留为 Artificial Analysis 候选，但不再误记为由 DataCurve DeepSWE 发现；候选表和已完成的书系内容不受影响。

## 2026-09-14：主动关注厂商范围收敛

- 根据用户最新要求，后续主动关注的大模型厂商限定为 OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。
- 其他厂商及其新模型暂不主动检索、核验或扩写；现有历史记录不删除、不回退，仅保留作为已有资料，直到用户明确通知重新纳入关注范围。该范围约束已写入 `plan_v2.md`。

## 2026-09-14：热点锚点首轮提取与逐项闭环检查

### 提取口径

- 本轮只从 [Artificial Analysis 2026-09-14 实时快照](research/model-update-2026-09/artificial-analysis-2026-09-14-snapshot.md) 和 [DataCurve DeepSWE v1.1 快照](research/model-update-2026-09/deepswe-snapshot-notes.md) 提取候选；厂商范围限定为 OpenAI、Anthropic、Z.ai（GLM）、Qwen、Kimi/Moonshot、DeepSeek、Google（Gemini）和 xAI（Grok）。
- `AA` 表示出现在 Artificial Analysis 快照，`DS` 表示出现在 DataCurve DeepSWE v1.1 的模型筛选或主表配置中。榜单中的 `effort`、fallback、provider 和 Agent harness 不作为独立模型重复计数。
- 这是一份面向面试技术追踪的首批热点锚点表，不是全量模型目录。GPT-4.x、Gemini 1/2.x、Qwen3 各小规格、DeepSeek V2/V3 历史版本、Claude 4.x 长尾版本等历史条目继续保留在旧盘点中，但不进入本轮主动研究队列。

### 锚点名称与状态

| 厂商 | 热点锚点（按基础模型/release 归并） | 榜单 | 当前闭环状态 | 已有证据与主要缺口 |
|---|---|---|---|---|
| OpenAI | GPT-6 Astra | AA/DS | 内容专题闭环 | 官方模型页、研究笔记、第六册第 18 章及配套同步已完成；参数、训练架构、发布日期和技术报告待补证。 |
| OpenAI | GPT-5.6 Sol、GPT-5.6 Luna、GPT-5.6 Terra | AA；Sol/Luna 亦见 DS | 资料级闭环 | 三个官方模型页、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档、开发者博客和研究笔记已完成；参数、架构、训练配方、system card、技术报告和独立复现待核验。 |
| OpenAI | GPT-5.5、GPT-5.5 Pro、GPT-5.5 Instant | AA；基础 GPT-5.5 亦见 DS | GPT-5.5/Pro 资料级闭环；Instant 为 AA-only 关联配置 | GPT-5.5/Pro 的模型页、专属指南、推理/工具/缓存/状态/后台文档已完成；Instant May/June 的 AA 快照已复核，但官方 `gpt-5.5-instant` 路径 404、DataCurve 无精确行，不能并入 `gpt-5.5` 或迁移其 Agent 结果。 |
| OpenAI | GPT-5.4、GPT-5.4 Pro、GPT-5.4 mini/nano | AA；基础 GPT-5.4 亦见 DS | GPT-5.4/mini/nano 资料级闭环；Pro 为关联档位 | GPT-5.4 base 的两个排行榜配置、mini/nano 的精确 AA 条目、OpenAI sibling 模型页和专属指南已完成；mini/nano 的 400K/272K/128K、effort、任务定位、价格和精确工具矩阵已分开核验；未公开架构、训练 recipe、独立技术报告和 mini/nano 精确 Agent 复现。 |
| OpenAI | gpt-oss-120b、gpt-oss-20b | AA 单榜 | 内容专题闭环 | DataCurve 当前没有精确 `mini_swe_agent_gpt_oss_*` 行；OpenAI Model Card/arXiv、官方仓库、Harmony、Hugging Face 和 Cookbook 已核验；MoE、MXFP4、Harmony、variable-effort reasoning 已进入第二十一册第 87 章，完整 recipe、kernel、profiling 和独立 Agent 评测待补证。 |
| Anthropic | Claude Fable 5.1 | AA | 资料级闭环 | 官方专属模型页、研究笔记和全套配套同步已完成；没有独立正式专题，参数、架构、训练细节和外部复现待补证。 |
| Anthropic | Claude Fable 5 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max + Opus 4.8 Fallback`、DeepSWE 的 `xhigh + mini-swe-agent`、Anthropic 模型页、发布/重新部署公告、API/Agent 文档、system card 入口和研究笔记已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 5 | AA/DS | 资料级闭环 | 两榜单配置、官方模型目录/专属页、开发者文档、发布公告、System Card 入口、研究笔记和配套同步已完成；参数、架构、训练细节和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 4.8 | AA/DS | 资料级闭环 | Artificial Analysis 的 `max`、DeepSWE 的 `xhigh`/`max` + `mini-swe-agent`、Anthropic 发布公告、System Card 入口、Dynamic Workflows 博客、研究笔记和配套同步已完成；参数、架构、训练配方和独立 benchmark 待核验。 |
| Anthropic | Claude Sonnet 5 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页/模型目录、完整 API/Agent 文档、System Card、Research/arXiv 定向检索、研究笔记和配套同步已完成；参数、架构、完整训练/后训练 recipe 和独立 benchmark 仍待核验。 |
| Anthropic | Claude Sonnet 4.6 | AA/DS | 资料级闭环 | 两榜单配置、Anthropic 发布页、真实模型目录、System Card 入口和 Agent 运行时资料已完成；1M context beta、adaptive/extended thinking、compaction、tool search、computer use 已核验；参数、架构、训练 recipe 和独立 benchmark 待核验。 |
| Anthropic | Claude Opus 4.7 | AA；DataCurve 无精确行 | 内容专题闭环（复用章节） | AA 的 adaptive/max 与 non-reasoning/high 配置、Anthropic 发布页/System Card/模型页、三层预算、更新 tokenizer、高分辨率视觉、compaction、tool search 和 cyber safeguards 控制面已核验；参数、架构、训练 recipe、生产 kernel 和独立 benchmark 待核验。 |
| Z.ai（GLM） | GLM-5.3 | AA/DS | 内容专题闭环 | 官方 GLM-5.3/5.2 文档、研究笔记、第十六册第 20 章及配套同步已完成；SAO with compaction 原始定义、模型卡、权重和许可证待补证。 |
| Z.ai（GLM） | GLM-5.3-Flash | AA/DS | 内容专题闭环 | 已核验 Z.ai 文档/博客、固定 revision 模型卡与配置；已新增第二十一册第 84 章并同步书系配套。完整训练 recipe、生产 kernel、硬件 profiling、线上接受率和独立评测仍待补证。 |
| Z.ai（GLM） | GLM-5 | AA 单榜 | 内容专题闭环（AA 单榜） | Artificial Analysis 有精确 `glm-5` 条目；DataCurve 当前没有精确 `mini_swe_agent_glm_5_*` 行。已核验 Z.ai 模型卡、GLM-5 专属技术报告、DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering；已新增第二十一册第 89 章及配套同步。完整 DSA kernel、调度细节、训练 recipe、硬件 profiling 和独立复现待补证。 |
| Z.ai（GLM） | GLM-5.2 | AA/DS | 资料级闭环 | 已完成 AA/DataCurve high/max 复验、Z.ai 官方文档、正确博客正文、研究笔记和第二十一册第 86 章；IndexShare、MTP、长上下文 serving、critic-based PPO、anti-hack、1M/128K、MCP、缓存和工作流已核验；参数、架构、SAO/compaction 原始定义、训练 recipe 和独立复现待补证。 |
| Qwen | Qwen3.8 Max、Qwen3.8 Max (0902)、Qwen3.8 27B、Qwen3.8 2.4T A95B、Qwen3.8-Flash-Next | AA；Max/0902 亦见 DS 泛化条目 | 系列内容专题闭环；0902 revision 资料级闭环 | 三份官方模型卡、Flash-Next GitHub/技术报告、Qwen Cloud Max/0902 页面与运行时文档、研究笔记、第 83 章及配套同步已完成；0902 无精确 DataCurve 行；完整生产 kernel、线上接受率、目标硬件 profiling、全系列训练配方和独立 benchmark 待核验。 |
| Kimi/Moonshot | Kimi K3 | AA/DS | 内容专题闭环 + stable release/source implementation evidence | 官方发布资料、K3 仓库/技术报告/许可证、HF 固定 revision/config、FlashKDA README/commit、vLLM `0.29.0` stable release/source、recipe、研究笔记、第十七册第 15 章和第二十一册第 88 章已同步；完整权重未下载，未完成本地 wheel 安装、目标硬件加载、完整 recipe、profiling、独立复现、hybrid cache recovery 和线上 acceptance。 |
| Kimi/Moonshot | Kimi K2.7 Code | AA/DS | 资料级闭环 | 官方资源/API 文档、固定 revision 模型卡/配置/许可证、部署指南和研究笔记已完成；1T/32B active MoE、MLA、MoonViT、native INT4、thinking/preserve-thinking/tool-call 协议已核验；专属训练报告、完整后训练配方、线上接受率和独立 benchmark 待核验。 |
| DeepSeek | DeepSeek V4.1 Flash | AA | 内容专题闭环 | 官方发布页、模型卡、技术报告、研究笔记和第二十一册第 81 章已完成；完整 kernel、线上接受率、目标硬件 profiling 和独立 benchmark 待补证。 |
| DeepSeek | DeepSeek V4 Pro、V4 Flash | AA/DS | V4 Pro 双榜内容专题闭环；V4 Flash 为系列资料闭环 | V4 Pro 已有 AA 精确 `deepseek-v4-pro`、DataCurve `mini_swe_agent_deepseek_v4_pro_max`、官方公告/API/模型卡/配置/技术报告和第 77/78 章映射；DataCurve 结果绑定 max + `mini-swe-agent`，生产 kernel、硬件 profiling、线上 acceptance 和独立复现仍待核验。V4 Flash/Flash Vision 的 alias 与视觉边界另行记录。 |
| DeepSeek | DeepSeek V4 Flash Vision | AA | 资料级闭环 | 已完成实验发布、Vision/Files/Responses/价格文档、研究笔记和第二十一册第 85 章；历史 alias、当前路由、图像预算和工具回灌已核验；视觉专属架构/训练报告、DataCurve 同名结果和独立评测仍待核验。 |
| DeepSeek | DeepSeek V3.2 | AA 单榜 | AA 单榜资料级闭环 | Artificial Analysis 有精确 `deepseek-v3-2` Non-reasoning 条目；DataCurve 当前无精确 `mini_swe_agent_deepseek_v3_2_*` 行。官方模型卡、技术报告和 V3.2-Exp 入口已核验；DSA 两阶段训练、2,048 KV top-k、scalable RL、GRPO 稳定化、Agent 合成/验证和 `thinking with tools` 已入库；公式/图表视觉复核、完整 kernel、线上 acceptance rate、硬件 profiling 和独立 benchmark 待补证。 |

## 2026-09-18：DeepSeek V3.2 AA 单榜资料级闭环

- 本轮重新抓取两个唯一排行榜：Artificial Analysis 当前详情页确认 `deepseek-v3-2` 的 `DeepSeek V3.2 (Non-reasoning)`，页面字段为 2025 年 12 月、128K context、Intelligence Index `16.043537719683`，第三方参数字段约 648B total/37B active；DataCurve 当前快照没有精确 `mini_swe_agent_deepseek_v3_2_*` 行，不迁移相邻版本结果。
- Artificial Analysis 详情快照为 3,391,821 bytes、SHA-256 `488c2c63fdb6d1746525f317222642d12cd57c7daf0f0abeba31653a6924ab7a`；DataCurve 快照为 268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 官方模型卡核验三条技术主线：DeepSeek Sparse Attention（DSA）、scalable RL framework、large-scale agentic task synthesis pipeline；并核验 `thinking with tools`、新的工具调用模板、`developer` role 仅限 search-agent 且 API 不接受、V3.2-Speciale 不支持 tool calling、MIT license 和 V3.2-Exp 结构入口。
- 官方技术报告 PDF 已经通过 7890 下载，907,086 bytes、SHA-256 `f6fda5753db7b106baa5eeb1286877f17e6a111354762f8aa53c7e6556498df7`；已用标准库完成正文抽取，抽取文本 252,018 bytes、SHA-256 `f082910a550666ad32c914db94056daa59a13b75a7ffd2bbd548695aa5cd043c`。报告正文补充确认 DSA 的 dense warm-up/sparse training、2,048 KV top-k、约 2.1B/943.7B tokens、主 attention 复杂度边界、specialist distillation、GRPO 的 unbiased KL/off-policy masking/Keep Routing/Keep Sampling Mask，以及 code/search/general/interpreter Agent 数据规模与 verifier 闭环。公式/图表因 PDF 字体抽取失真仍待视觉复核；完整 kernel、层排布、训练超参、线上 acceptance rate、硬件 profiling 和独立 benchmark 继续待核验。
- 当前状态为“AA 单榜资料级闭环”，不新增重复正式章节；研究笔记见 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)。后续继续从两个排行榜剩余重点条目选择下一锚点。
- 配套闭环已完成：已同步 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`，并在第二十一册稀疏注意力、第四册后训练与第二十四册工具 serving 中补入 DSA、Agent 任务合成和 `thinking with tools` 的面试映射；未把 GLM/DeepSeek V4 的实现字段迁移为 V3.2 事实。
| Google（Gemini） | Gemini 3.8 Flash | AA/DS | 资料级闭环 | Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF、研究笔记及工具/推理文档已入库；GA/stable、1M 输入、64K 输出、thinking、Interactions API 和工具闭环已核验；3.8 独立架构/训练报告和正式专题待补证。 |
| Google（Gemini） | Gemini 3.7 Flash | AA/DS | 资料级闭环 | 已完成 Google AI Developers 模型页、DeepMind Model Card、评测/安全入口、Interactions/工具/视频文档和 arXiv 定向检索；1M 输入、thinking、agentic video、step/state/signature 回放已核验；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.6 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 9 篇外部使用论文核验；默认 medium 与 `minimal/low/medium/high` thinking、agentic video、Model Card benchmark/safety 已记录，独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.5 Flash | AA/DS | 资料级闭环 | 已完成 Artificial Analysis/DataCurve 原始配置、Google AI Developers 模型页、What's New、DeepMind Model Card、Gemini 3 Flash 依赖 Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 定向检索；默认 medium、thought preservation、工具上下文循环、Computer Use prompt-injection detection 和证据边界已记录；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.5 Flash-Lite | AA 单榜 | 资料级闭环（AA 单榜） | Artificial Analysis 有精确 `gemini-3-5-flash-lite` 条目；DataCurve 当前无精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行。已核验 Google API 模型页、DeepMind Model Card/PDF、Thinking、Video understanding、发布方 benchmark 和安全边界；1M/65K、默认 minimal、agentic video、`processing_call/result` 已记录；独立架构/训练报告待补证。 |
| Google（Gemini） | Gemini 3.1 Pro Preview | AA/DS | 资料级闭环 | 已完成三代理榜单复验、Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking/工具组合/Long context/Caching 文档和 arXiv 检索；1M/65K、多模态、`thinking_level`、`customtools` endpoint、signature/id 回放和 tool context circulation 已核验；独立架构/训练报告待补证。 |
| xAI（Grok） | Grok 4.6 | AA/DS | 资料级闭环 | xAI 官方发布页、模型/API 文档、四档 DataCurve 配置、研究笔记和配套同步已完成；官方发布日期、500K context、模型生成数据/SFT 轨迹筛选、agentic RL、self-testing/verification 和工具协议已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |
| xAI（Grok） | Grok 4.5 | AA/DS | 资料级闭环 | 两个排行榜的 `high` 配置、xAI 发布公告、官方模型/API 文档、研究笔记和同步记录已完成；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验。 |

### 后续逐项顺序

（历史顺序记录）Grok 4.5、Grok 4.6、Claude Opus 5、Claude Fable 5.1、Claude Sonnet 5、Claude Sonnet 4.6、Gemini 3.7 Flash、Gemini 3.6 Flash、Gemini 3.5 Flash 和 Gemini 3.1 Pro Preview 均已完成“仅候选”到“资料级闭环”的升级，不新增专属架构章节。GLM-5 已完成 AA 单榜资料闭环，GLM-5.2 已完成双榜资料级闭环并新增第二十一册第 86 章；GPT-5.4、GPT-5.5、GPT-5.6、Gemini 3.8 Flash、Claude Fable 5、Claude Opus 4.8 和 Kimi K2.7 Code 继续只补独立架构/训练报告、真实 API 行为和外部评测；已达到“内容专题闭环”的 GPT-6 Astra、GLM-5.3、GLM-5.3-Flash、Kimi K3、DeepSeek V4/V4.1、K2 Horizon 36B/A4B 和 Qwen3.8 只补待核验项，不重复建立全量模型档案。该段之后已完成 GPT-6 Astra、Qwen3.8 Max (0902)、GPT-5.3 Codex、gpt-oss 与 Claude Opus 4.6 收口；本段为历史顺序，当前活动锚点以文件顶部状态和最后一条记录为准。

## 2026-09-14：Qwen3.8 内容专题闭环

- Artificial Analysis 快照确认四类 Qwen3.8 条目：Max（2026-08-03）、2.4T-A95B（2026-08-12）、27B（2026-08-14）和 Flash-Next（2026-08-26）；DataCurve DeepSWE 当前快照只确认 Max。不同 `reasoning_effort`/Non-reasoning 行按配置归并，不按行数计基础模型。
- 官方核验完成：27B 和 A95B 模型卡、Flash-Next 模型卡与 GitHub、28 页技术报告、Qwen Cloud Max/27B/Flash 页面。研究笔记见 [`qwen3.8-source-notes.md`](research/model-update-2026-09/qwen3.8-source-notes.md)。
- 已整理面试主线：GDN + Gated Attention 的混合记忆、QSA 的 micro-block indexer/两阶段训练、四分支 Gated Residual、Layer 2 N-gram Embedding 的 host-memory prefetch、Muon/AdamW 分工与 batch-size warmup 取舍，以及 `enable_thinking`/`reasoning_effort`/`preserve_thinking` 协议。
- 新增第二十一册第 83 章，并同步第四册百科、目录、来源索引、模型盘点、候选解释、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、README、`BOOK_SERIES.md`、`ROADMAP.md`、`PROGRESS.md` 和本计划/进度文件。
- 报告自报边界：QSA 平均分 75.9 -> 76.8、512K--1M RULER 90.08 -> 93.00、1M MRCR 20.71 -> 26.44、kernel prefill/decode 约 7.6x/4.9x；这些结果绑定报告设置和 FlashInfer baseline，不是独立复现。
- 待核验保留：完整生产 kernel、线上 MTP/QSA acceptance rate、目标硬件 profiling、host-memory 预取端到端收益、全系列训练/后训练配方和独立 benchmark。Qwen3.8-Max 是基于 A95B 的 hosted version，不写成独立 open checkpoint。

## 2026-09-14：Qwen3.8 纵向同步与 demo 收口

- 已将 Qwen3.8 研究笔记和第二十一册第 83 章同步到论文路线、面试题库、练习、术语、项目、知识图谱、README、书系目录和路线图；核心主题是 GDN/Gated Attention、QSA、Gated Residual、N-gram Embedding、Muon/AdamW 与 thinking protocol。
- 已修复第 83 章 QSA demo 的预算截断问题：先预留当前 causal tail，再选择完整 block；`token_budget` 不足以容纳 tail 时显式拒绝，正常示例增加 tail 保留断言。
- 已检查 hosted/open 边界：Qwen3.8-Max 只记录为官方说明基于 A95B 的 hosted version，Flash hosted 不转写成独立 checkpoint，effort/Non-reasoning 行不计为新基础模型。
- 本轮不新增模型候选；Flash-Next 报告中的速度、loss、benchmark 和稳定性仍是发布方自报，完整生产 kernel、目标硬件 profiling、线上 acceptance rate、host-memory 端到端收益和独立 benchmark 保持待核验。

## 2026-09-14：Gemini 3.8 Flash 资料级闭环

- Artificial Analysis 快照确认 `Gemini 3.8 Flash` 的 high/medium/low 三个配置，榜单日期均为 2026-09-02；DataCurve DeepSWE v1.1 快照确认 high 配置。不同 effort 行按一个基础模型归并。
- DeepSWE 的 high 配置记录为 Pass@1 74%（±1%）、平均成本约 $2.36、输出 token 约 143K、Agent steps 166；这些数字属于模型配置 + `mini-swe-agent` + 工具/环境/verifier 的系统结果。
- Google 官方模型页和最新模型迁移页将 `gemini-3.8-flash` 标为 GA/stable，确认 1,048,576 输入 token、65,536 输出 token、文本/图像/视频/音频/PDF 输入、文本输出、low/medium/high thinking，以及 caching、Search/Maps grounding、function calling、structured outputs、URL Context、File Search、Code Execution 和 Computer Use（Preview）。
- 已从官方周边文档提取面试技术点：Interactions API 的 thought/tool/model-output steps 与 `previous_interaction_id`，thought summary/signature 的状态连续性，搜索/URL 深读/文件检索/代码沙箱/电脑操作的工具闭环，1M 长上下文与 implicit caching 的成本边界。
- 官方最新模型页称复杂任务会采用更小的 reasoning steps、迭代调用工具并验证结果；该内容按公开系统行为记录，不推断具体 RL、verifier、搜索树或 Transformer 架构。Model Card 将 3.8 架构、训练数据及软硬件信息指向 3.7 Model Card，当前没有独立 3.8 参数/架构/训练报告或论文证据。
- 新增研究笔记 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)，同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md` 和本进度表；暂无 Gemini 专属正式章节，先映射已有 Reasoning、Agent、工具协议、Computer Use、长上下文/Serving 和评测章节。
- 该项闭环属于“资料级闭环”，不等于内部技术已全部公开；后续补证重点为独立架构/训练资料、真实 API 行为、生产工具安全边界和外部 benchmark 复现。

## 2026-09-14：GPT-5.6 资料级闭环

- Artificial Analysis 2026-07-09 快照记录 Sol/Terra/Luna 的多档 effort；DataCurve DeepSWE v1.1 快照记录 `gpt-5.6-sol` max（73% ±3%、约 $6.46、60K 输出 token、61 steps）和 `gpt-5.6-luna` max（67% ±4%、约 $0.61、73K 输出 token、102 steps）。不同 effort 行归并为同一 GPT-5.6 家族；DeepSWE 数字归因于模型配置 + `mini-swe-agent` + 工具/环境/verifier。
- OpenAI 三个模型页确认 Sol/Terra/Luna 的层级、`gpt-5.6` 到 Sol 的 alias、1,050,000 context、922,000 maximum input、128,000 maximum output、text/image input、text output、reasoning tokens 以及 `none`--`max` effort；Chat Completions、Responses、Batch 和 Responses 工具支持也已记录。
- Reasoning 文档补充 GPT-5.6 的 Responses `standard/pro` mode、effort 解耦、`reasoning.context=all_turns` 默认语义、同家族 reasoning 复用、function call 后保留 reasoning items 和 output budget/incomplete 边界。
- Prompt Caching 文档补充 GPT-5.6 的 1,024 可见 token 最小缓存前缀、implicit/explicit 断点、最多四次写入、0.1x 读取、1.25x 写入、30m TTL、自动路由和 compaction 可能降低 cache hit 的边界；Compaction/Agents/Tools 文档补充长期任务、工具搜索、MCP、Skills、Shell 和 harness 分层。
- 官方开发者博客 `Codex as a platform` 给出 retained reasoning + context compaction 使特定 GPT-5.6 Sol 工程流程分数从 13.3% 到 38.3%、输出 token 减少六倍的系统级例子；该数字不写成裸模型 benchmark。`Shell + Skills + Compaction` 与 `One year of Responses` 用于解释长期 Agent 的 runtime 组合。
- 本次没有找到 GPT-5.6 专属参数、架构、训练/后训练 recipe、system card、独立技术报告或可复现 benchmark。`latest-model.md?model=gpt-5.6` 当前实际解析为 GPT-6 Astra，已排除出 GPT-5.6 证据链。研究笔记见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- GPT-5.6 当前为“资料级闭环”，不新增专属正式章节；面试映射到第六、七、五、八、十六、十七、二十和二十四册。下一锚点转向 GPT-5.5。

## 2026-09-15：GPT-5.5 资料级闭环

- Artificial Analysis 历史快照确认 GPT-5.5 的 `xhigh/high/medium/low/Non-reasoning` 配置与 GPT-5.5 Pro `xhigh`，榜单日期为 2026-04-23；DataCurve DeepSWE v1.1 快照确认 `gpt-5.5` `xhigh`（67% ±6%、约 $7.23、46K 输出 token、82 steps）。这些数字归因于模型配置 + `mini-swe-agent` + 工具/环境/verifier，不写成裸模型能力。
- Artificial Analysis 另有 GPT-5.5 Instant 的 2026-05-05/2026-06-25 条目；当前没有把它们与 OpenAI API `gpt-5.5` 合并，仍标为仅榜单级关联发现。
- OpenAI 官方模型页确认 `gpt-5.5` 的默认 snapshot `gpt-5.5-2026-04-23`、1,050,000 context、128,000 max output、text/image input、text output、reasoning tokens 和 `none`--`xhigh` effort；Pro 页确认 `gpt-5.5-pro-2026-04-23`、Responses/Batch、`medium`--`xhigh`、默认 high、较高计算量和无 cached input discount。
- GPT-5.5 专属指南和 API 文档提取了面试主线：同 effort 下更少 reasoning tokens 的官方行为描述、outcome-first prompting、工具描述与 tool search、`text.verbosity`、图像 `detail` 的 token/精度权衡、Responses `phase` 的 assistant-item 回放、compaction、Pro background mode，以及与 GPT-5.6 不同的 prompt-cache 断点/key/TTL/统计/写入规则。
- 本轮未找到 GPT-5.5 专属参数规模、架构、训练/后训练 recipe、system card、独立技术报告或可复现 benchmark；不能从 `reasoning.effort`、`phase`、tool search、compaction、image_detail 或 1.05M context 反推内部机制。研究笔记见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md)。
- GPT-5.5/Pro 当前为“资料级闭环”，不新增 GPT-5.5 专属正式章节；内容映射到第六、七、十六、十七、二十和二十四册。下一锚点切换为 Claude Fable 5。

## 2026-09-15：Claude Fable 5 资料级闭环与联网复访

- Artificial Analysis 详情页确认 `claude-fable-5` 的 `max + Opus 4.8 Fallback` 配置，页面 `releaseDate` 字段为 2026-06-09；当前快照为 3,531,469 bytes，SHA-256 为 `a69a5ea342984d9b4054bec82756fab3e67863589e7aaf4b31ca38ce3f9ba4d0`。
- DataCurve DeepSWE v1.1 当前快照仍显示 113 个任务、91 个仓库、5 种语言和 `mini-swe-agent`；Fable 5 `xhigh` 为 316/452，Pass@1 约 70% ±3%、Pass@4 约 88.5%、平均成本约 `$13.41`。这些数字绑定模型配置、effort、harness、工具、任务集和 verifier，不是裸模型分数。
- Anthropic 官方资料确认 `claude-fable-5` 为 Active (legacy)，1M context、128K max output、adaptive always-on thinking、默认 `high`、拒答/fallback 协议；官方文档还覆盖 Fallback credit、memory、programmatic tool calling、compaction、context editing 和 task budgets。重新部署公告记录了 cyber safety classifier 更新及向 Opus 4.8 的安全 fallback。
- 面试技术主线已整理为：opaque thinking/signature 与原始思维链的区分、effort 与 `max_tokens` 的区别、HTTP 200 refusal 的业务处理、跨模型 fallback 的工具/权限/缓存边界、just-in-time memory、代码执行中的批量工具调用和三层长任务账本。
- 已下载并登记官方 Fable 5/Mythos 5 system card，文件约 26.96 MB，SHA-256 为 `f95d413845ad8624f384ba026963f2bad2158f10f2626575bb45e823e3c2e0ca`；当前环境未能稳定提取 PDF 正文，因此不从目录或文件字符串扩写安全数值。
- 本次三代理轻量复验结果：`10.24.27.134:8098` 与 `10.237.126.170:1234` 访问两个排行榜均返回 HTTP 200；`10.24.27.134:7890` 访问 DataCurve 返回 200，访问 Artificial Analysis 出现 TLS EOF。Artificial Analysis 快照为 1,772,838 bytes、SHA-256 `1aa28f64778508223642522caffb86da7b9c34395da1030437a5ca1827f369c3`；DataCurve 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Claude Fable 5 当前标记为“资料级闭环”，不新增独立架构章节；参数规模、内部架构、训练/后训练配方、完整技术报告和独立 benchmark 仍待核验。下一锚点切换为 `claude-opus-4.8`。

## 2026-09-15：Claude Opus 4.8 资料级闭环与联网复验

- Artificial Analysis 详情页确认 `claude-opus-4-8` 的 `max` 配置，页面 `releaseDate` 字段为 2026-05-28；当前页面标记 deprecated，并指向 `claude-opus-5`。DataCurve DeepSWE v1.1 当前快照仍显示 Opus 4.8 的 `xhigh` 和 `max` 配置，统一使用 `mini-swe-agent`。
- Opus 4.8 的 Artificial Analysis 条目约为 41.99 Intelligence Index、约 56.1 output tokens/s、1M context 和约 `$4.08`/task；DeepSWE 的 `xhigh` 记录为 243/447、Pass@1 约 54.36%、Pass@4 约 80.53%、平均成本约 `$8.01`，`max` 记录为 253/429、Pass@1 约 58.97%、平均成本约 `$13.22`。这些数字分别绑定榜单配置、effort、任务集、provider、harness、工具和 verifier，不是裸模型分数。
- Anthropic 官方公告确认 2026-05-28 发布、API ID `claude-opus-4-8`、默认 `high` effort、普通/fast mode 价格边界和更高 effort 使用建议；Messages API 支持任务中途插入 system entries，并强调不破坏 prompt cache。
- Dynamic Workflows 官方博客公开了动态生成 orchestration scripts、数十至数百并行 subagents、独立检查、反驳式复核和长任务断点恢复。面试主线整理为：外部编排图、effort sweep、权限/预算/环境更新、缓存保持、取消/超时治理，以及把“完成声明”转化为测试、lint、类型检查和领域 verifier 的 artifact 门禁。
- 已下载并登记 Opus 4.8 System Card；当前环境未能稳定提取 PDF 正文，因此不从文件目录或字符串扩写安全数值。Anthropic 公开资料没有参数规模、内部架构、训练/后训练 recipe、可独立复现技术报告或完整外部 benchmark。
- 新鲜联网复验：`10.237.126.170:1234` 对 Artificial Analysis `/zh`、Opus 4.8 详情页和 DataCurve 均返回 HTTP 200；`10.24.27.134:8098` 对两个排行榜均返回 HTTP 200。复验文件 SHA-256 为 AA `/zh` `264357ee84d21d14f383cc7b179c34843ab39ef45c5bafad397946c7fadc23fd`、DataCurve `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`、Opus 详情页 `369ef837ec32f3de7b59c8f6fdc7657b8608a22ccff0717c75c6cf3c0167c522`。
- Claude Opus 4.8 当前标记为“资料级闭环”，不新增独立架构章节；内容映射到第六、七、十六、十七、二十和二十四册。当前锚点为 `claude-opus-4.8`，下一锚点切换为 `kimi-k2.7-code`。

## 2026-09-15：Kimi K2.7 Code 资料级闭环与联网复验

- 重新使用用户提供的代理访问 [Artificial Analysis Kimi K2.7 Code](https://artificialanalysis.ai/models/kimi-k2-7-code)、[DataCurve DeepSWE](https://deepswe.datacurve.ai/)、[Kimi 官方资源页](https://www.kimi.ai/resources/kimi-k2-7-code) 和 [API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)，均获得 HTTP 200。`10.24.27.134:7890` 对 Hugging Face 固定 revision 页面获得 HTTP 200；其对超大页面曾读取超时，未把超时写成资料不存在。
- Artificial Analysis 重抓快照为 3,605,694 bytes，SHA-256 `e37ce8e8f224233a95ca93771fe9f2130e544ef071703bae475295bd1ac527db`；DataCurve 重抓快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。两榜单条目均确认 canonical slug `kimi-k2-7-code`；DataCurve 配置为 `mini-swe-agent`、`reasoning_effort:null`，记录 138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155`、平均 149.1 steps。
- 固定 revision `74797c9c62378b951a1f6fcf5c4631024e9b8bef` 的官方模型卡/config 确认 1T 总参数、32B active、61 层、1 dense layer、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 配置；许可证为 Modified MIT，并含大规模商业产品显示名称的附加条款。
- Kimi API 快速开始确认只支持 thinking、强制 `preserve_thinking`、固定 temperature/top-p/n/penalty、`tool_choice` 仅 `auto`/`none`，以及多步 tool call 中回传完整 `reasoning_content` 的协议建议。官方部署指南覆盖 vLLM、SGLang 和 KTransformers；Kimi Code CLI 是推荐的 coding-agent harness。
- 模型卡自报 benchmark 已记录，但所有横向分数均绑定 CLI/Codex/Claude Code、effort、工具预算、上下文长度和重复次数；DeepSWE 结果还绑定 `mini-swe-agent`，不能拼成裸模型排名。动态工具、Context Caching、Partial Mode 和自动断线重连作为 Kimi API 周边技术记录，其中动态工具文档明确当前仅 K3 支持，不写成 K2.7 独有能力。
- 定向检索没有找到 Kimi K2.7 Code 专属 arXiv 论文或独立训练技术报告：官方资源页/模型卡没有提供专属报告链接，`site:arxiv.org "Kimi K2.7 Code"` 查询未返回专属结果，MoonshotAI GitHub repository API 同名查询为零结果。该项是本轮公开入口的负面证据，不是对未来发布的否定。
- Kimi K2.7 Code 当前为“资料级闭环”，暂无独立正式章节；研究笔记见 [`kimi-k2.7-code-source-notes.md`](research/model-update-2026-09/kimi-k2.7-code-source-notes.md)。当前锚点切换为 `kimi-k2.7-code`，下一锚点为 `grok-4.5`。

## 2026-09-15：Grok 4.5 资料级闭环与联网复验

- 本轮重新使用用户提供的代理复验网络状态：`10.237.126.170:1234` 访问 Artificial Analysis Grok 4.5、DataCurve 和 xAI 官方资料均返回 HTTP 200；`10.24.27.134:8098` 访问 Artificial Analysis 与 DataCurve 均返回 HTTP 200。`10.24.27.134:7890` 对 DataCurve 成功，但 Artificial Analysis 大页面出现 TLS EOF，因此不把单个代理失败写成资料不存在。
- Artificial Analysis Grok 4.5 页面配置为 `Grok 4.5 (high)`，第三方 `releaseDate` 为 2026-07-08，Intelligence Index 约 39.08、页面排序约 #38/200、输出速度约 60.98 tokens/s、context 500K；快照 `/tmp/aa-1234.html` 为 3,517,851 bytes，SHA-256 `839c29a20b9f5977fde997e6868f96c15d6c4d4f41c8fb53e5bac15bf7ce078`。
- DataCurve DeepSWE v1.1 页面更新时间为 2026-09-03，`generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库/语言规模为 113/91/5，统一 harness 为 `mini-swe-agent`。Grok 4.5 行配置为 `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high`，记录 243/452、Pass@1 53.761%、Pass@4 77.876%、4 runs、平均成本约 `$2.4157`、约 61.33 Agent steps、约 35,525 输出 token 和约 484.7s；快照 `/tmp/deepswe-1234.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。这些结果绑定 harness、工具、任务集、环境和 verifier，不是裸模型分数。
- xAI [Grok 4.5 发布公告](https://x.ai/news/grok-4-5) 标注 2026-07-16，公开定位为 coding、agentic tasks 和 knowledge work；发布方描述了 coding/science/engineering/math 数据、deduplication、quality scoring、domain-focused selection、数以万计 GB300 GPU、数十万任务 RL、automated/model-based grading 和 highly asynchronous agentic rollouts。DeepSWE 1.0/1.1、SWE Marathon、Terminal-Bench 2.1、SWE-Bench Pro 和 80 TPS 数字均按 xAI 自报及其各自配置记录，不反推完整训练配方。
- xAI 官方模型页确认 `grok-4.5`、`grok-4.5-latest`/`grok-build-latest` aliases、500K prompt、text/image input、text output、function calling、structured outputs、reasoning、Batch API 不支持、150 RPS、50M TPM、区域和 200K prompt 价格阈值。专属模型页列出 `low/medium/high/xhigh`，但通用 Reasoning 文档和 Release Notes 称 Grok 4.5 的可用档位为 low/medium/high、`xhigh` 按 high 处理；该官方文档不一致保留为待复验项。
- 已从 xAI 官方周边文档提取面试技术点：reasoning_tokens 与 opaque/encrypted reasoning state、Responses 30 天状态化会话、`/v1/responses/compact` 的单项 opaque compaction、server-side built-in tools 与 client-side function calling 的责任分层、默认 parallel function calling、Web/X Search、Remote MCP `allowed_tools` 最小权限。`grok-4.20-multi-agent` 的 4/16 agents 和 leader agent 属于独立模型，不能套到 Grok 4.5。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.5%22&searchtype=title) 本轮返回 0 个结果；全文检索命中的论文只是将 Grok 4.5 作为被测或背景模型，当前公开入口未检出 xAI 发布的 Grok 4.5 专属论文或独立技术报告。该项是负面检索证据，不是对未来发布的绝对否定。
- 新增研究笔记 [`grok-4.5-source-notes.md`](research/model-update-2026-09/grok-4.5-source-notes.md)，同步来源索引、模型盘点、榜单解释、`plan_v2.md` 和本进度表。Grok 4.5 当前为“资料级闭环”，没有独立架构/训练报告，因此不新增专属正式章节；当前锚点切换为 `grok-4.5`，下一锚点为 `grok-4.6`。

## 2026-09-15：Grok 4.6 资料级闭环与联网复验

- 本轮重新使用用户提供的代理获取并复验 Artificial Analysis Grok 4.6、DataCurve DeepSWE 和 xAI 官方页面：`10.237.126.170:1234` 与 `10.24.27.134:8098` 对两个排行榜均返回 HTTP 200，`10.24.27.134:7890` 对 xAI 官方模型页返回 HTTP 200。轻量复验中 `1234` 访问 xAI 曾返回代理 503，7890 访问 Artificial Analysis 偶发 TLS EOF；单代理失败未被解释为页面不存在。
- Artificial Analysis 详情页确认 `Grok 4.6 (high)`，并列出 low/medium/xhigh 配置；第三方 `releaseDate` 为 2026-08-12，Intelligence Index `44.4050073012592`，输出速度 `58.5035 tokens/s`，TTFT `40.8596s`，context 500K，价格 `$2/$6`，proprietary 且 parameters 为 null。两个代理抓到的快照均为 3,522,180 bytes，SHA-256 `8d6c96aa27f6db87537f3f1802ca07122d58385ee4b6d5e1a5ab84897f360383`。
- DataCurve DeepSWE v1.1 页面仍为 2026-09-03 更新，`generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务/仓库/语言规模为 113/91/5，统一 harness 为 `mini-swe-agent`。Grok 4.6 四档 Pass@1 为 low 41.648%、medium 67.478%、high 65.188%、xhigh 66.741%，平均成本约 `$1.0424/$3.4490/$4.3849/$5.4977`；这些数字绑定 effort、harness、工具、任务集、环境和 verifier，不是裸模型分数。快照为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- xAI [Grok 4.6 发布公告](https://x.ai/news/grok-4-6) 的 JSON-LD 日期为 2026-08-12，公开披露更长 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks、面向知识工作/编码/领域环境的 agentic RL，以及长轨迹 self-testing/verification。发布方未公开参数规模、网络结构、具体 optimizer、RL objective 或完整 recipe。
- xAI 官方模型页确认 `grok-4.6`、500K context、2026-02-01 knowledge cutoff、text/image input、text output、无文本输出上限、`low/medium/high/xhigh`（默认 high）、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution、`prompt_cache_key`、`x-grok-conv-id` 和长循环 compaction 建议。实际 HTML 快照为 408,905 bytes，SHA-256 `9669e2c5e96dafb74296f6e11af7c3b0fc74e8dec58a3d674edf18154c1caf27`。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.6%22&searchtype=title) 本轮返回 0 个结果；当前公开入口未检出 Grok 4.6 专属论文、独立技术报告或公开权重代码。该项是负面检索证据，不是对未来发布的绝对否定。
- Grok 4.6 当前为“资料级闭环”，不新增独立架构章节；研究笔记见 [`grok-4.6-source-notes.md`](research/model-update-2026-09/grok-4.6-source-notes.md)。当前锚点切换为 `grok-4.6`，下一锚点为 `claude-opus-5`。

## 2026-09-15：Claude Opus 5 资料级闭环与网络复验

- 重新发起联网搜索后，`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Artificial Analysis Opus 5 详情页、DataCurve DeepSWE 和 Anthropic 官方发布/研究页面均返回 HTTP 200；`10.24.27.134:7890` 对 DataCurve 返回 200，但 Artificial Analysis 大页面在超时前只收到部分内容。代理传输失败没有被解释成页面不存在。
- Artificial Analysis 详情页确认 `Claude Opus 5 (Adaptive Reasoning, Max Effort)`、canonical slug `claude-opus-5`、榜单 `releaseDate` `2026-07-24`，约 `50.7002` Intelligence Index、`50.0725` output tokens/s、`46.5045s` median TTFT、1M context 和约 `$5.8584`/task。详情页把 parameters 留为 null、开放性标为 proprietary；这些是第三方页面字段，不是 Anthropic 参数披露。
- DataCurve DeepSWE v1.1 仍标注 2026-09-03、113 个任务、91 个仓库、5 种语言和统一 `mini-swe-agent` harness。Opus 5 max 配置 `mini_swe_agent_claude_opus_5_max` 为 327/444、Pass@1 `73.6486%`（页面约 `74% ±4%`）、Pass@4 `88.4956%`、平均成本 `$11.8376`、平均输出 117,566 tokens、约 99.04 Agent steps；这是模型配置 + effort + harness + 工具/环境/verifier 的组合结果。
- 本轮快照：Artificial Analysis 中文首页 `/tmp/recheck-aa-1234.html` 为 1,773,814 bytes、SHA-256 `5f70a4b28d24ce6c560f83561ed0fe0f7150675b3e8a5a33ea649a238695e4ea3`；Opus 5 详情 `/tmp/recheck-aa-opus5-1234.html` 为 3,532,163 bytes、SHA-256 `c9626e404c0ff538a28bc58fff9f05bd276db64a1791b72bfa22ce7f4202d47f`；DataCurve `/tmp/recheck-deepswe-1234.html` 为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Anthropic 官方模型总览、Opus 5 专属页和完整 Markdown 文档确认：`claude-opus-5`、2026-07-24、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、adaptive thinking、默认 `high`、`low/medium/high/xhigh/max` 五档 effort、$5/$25 输入输出价格、512-token 最小缓存 prompt、Claude API/Bedrock/Vertex/Foundry 平台以及 2026-05 截止字段。文档还确认 thinking block 按 `content[].type` 解析、工具循环原样回传 thinking/signature、工具和 effort 中途变更、`fallbacks: "default"`、refusal 业务态和 web fetch 不支持。
- Anthropic 发布方声称 Frontier-Bench、CursorBench、ARC-AGI 3、Zapier AutomationBench、OSWorld 2.0 及内部生命科学任务有提升；这些保留为发布方数据。Frontier-Bench 脚注绑定内部运行、`mini-SWE-agent`、GKE、每任务 5 次尝试和安全拒答时向 Opus 4.8 fallback，不能与两个排行榜结果拼成裸模型能力。
- 官方 System Card 已下载登记：16,281,258 bytes，SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。当前环境无法稳定抽取 PDF 正文，因此不从目录或二进制字符串扩写安全数字。
- arXiv 精确标题检索返回 2 篇把 Opus 5 当被测模型的文章（arXiv:2608.14992、2608.07776），没有 Opus 5 专属技术报告；Anthropic Research 页面也没有 Opus 5 专属报告条目。该负面结果有日期和入口范围，不是否定未来发布。
- Claude Opus 5 当前为“资料级闭环”，不新增独立架构章节；研究笔记见 [`claude-opus-5-source-notes.md`](research/model-update-2026-09/claude-opus-5-source-notes.md)。当前锚点为 `claude-opus-5`，下一锚点为 `claude-fable-5.1`。

## 2026-09-15：Claude Fable 5.1 断点恢复、联网复验与闭环

- 针对“夜间 20:00—次日 09:00 可能发生代理中断”的风险，本轮重新发起联网搜索。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis 中文首页、Fable 5.1 详情页和 DataCurve 均返回 HTTP 200；7890 仍可访问部分页面，但对 Artificial Analysis 大页面有 TLS EOF/读取不稳定。没有把单代理失败解释成页面不存在。
- Artificial Analysis 新鲜详情页确认 `claude-fable-5-1` 的 `max + Default Fallback` 主配置、`releaseDate: 2026-09-01`、Intelligence Index `53.3737509623252`、输出速度 `65.9769929669683 tokens/s`、TTFT `212.0122070875s`、1M context、`$10/$50` 每百万 token和约 `$7.6297`/task；页面同时有 xhigh/high/medium/low fallback 变体。快照 `/tmp/recheck-aa-fable51-20260915.html` 为 3,615,956 bytes，SHA-256 `aa253dd4d4e8ad3285f60d0910f698239af5bfcefd268bb0127e41a74c93ba4b`；首页 `/tmp/recheck-aa-home-20260915.html` 为 1,773,775 bytes，SHA-256 `8589909c7199e311a750a2b3c0435e139508da4c6343ae063c935d0c3dbd1a98`。
- DataCurve DeepSWE v1.1 新鲜快照仍为 113 tasks、91 repositories、5 languages、`mini-swe-agent`，只有 `claude-fable-5` 的 effort 行，没有 `claude-fable-5-1`。因此不为 Fable 5.1 编造 DeepSWE 分数，也不把 Fable 5 的 316/452 结果迁移给它。快照 `/tmp/recheck-deepswe-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- Anthropic 发布页本次返回 HTTP 200，确认 Fable 5.1 与 Mythos 5.1 是同一 underlying model、不同 safeguards；Claude Code 默认 High，Claude Cowork/Claude.ai 默认 Medium；cache read 为 `$0.25/M`；典型工作负载成本约低 25%、高度 Agent 化工作负载最高约低 45%（均为发布方分析）；并公开 EFS 客户云数据边界、anti-distillation 历史编辑限制、自定义 GPU kernel/中间缓存科学案例和生产 safeguards 影响 benchmark 的说明。
- 发布方 benchmark 摘要已登记为自报结果：Terminal-Bench-Science 0.1 52.6%、Terminal-Bench 4.0 55.8%（Fable）/60.9%（Mythos）、GDPval-AA v2 1853、OSWorld 2.0 77.9% partial/41.7% strict、Humanity’s Last Exam 60.9% 无工具/65.0% 有工具、AutomationBench 31.4%、CursorBench 3.2.0 73.4%。这些数字绑定 safeguards、effort、工具和测试版本，不替代独立复现。
- System Card 重新下载成功：`/tmp/recheck-anthropic-fable51-system-card-page-20260915.bin`，16,397,488 bytes，SHA-256 `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。当前环境仍无稳定 PDF 正文提取器，未从二进制猜测安全数值。
- arXiv 标题精确查询 `"Claude Fable 5.1"` 返回 0 篇；全文查询返回 4 篇外部使用/评测论文（arXiv:2609.15597、2609.15494、2609.10420、2609.08847），没有 Fable 5.1 专属技术报告。Anthropic Research 页面可访问但未检出专属报告条目；详细标题和证据边界已写入研究笔记。
- 已同步 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本文件。Fable 5.1 当前为“资料级闭环”，不新增独立架构章节；当前锚点切换为 `claude-fable-5.1`，下一锚点为 `claude-sonnet-5`。

## 2026-09-15：Claude Sonnet 5 资料级闭环与联网复核

- 考虑到用户工作时段可能造成夜间代理中断，本轮重新发起联网搜索并成功恢复 Sonnet 5 资料获取。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis、DataCurve 和 Anthropic 发布页均返回 HTTP 200；`10.24.27.134:7890` 可作为备用线路，但访问 Artificial Analysis 大页面仍有 TLS/读取超时。单个代理失败不被解释为页面不存在。
- Artificial Analysis 快照 `/tmp/recheck2-aa-sonnet5-20260915.html` 为 3,614,205 bytes，SHA-256 `3e8257d0efec85f2cc30bf78b31ed3e5fa10c0be4da6ec55413de7cae9b2d9bf`；DataCurve 快照 `/tmp/recheck2-deepswe-sonnet5-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；Anthropic 发布页快照 `/tmp/recheck2-anthropic-sonnet5-release-20260915.html` 为 264,609 bytes，SHA-256 `5cc40094a584f1821874847040cff912056845dac56bc770826d4dbcd11cb725`。System Card PDF、完整开发者文档、Research 和 arXiv 快照哈希已登记在研究笔记。
- Artificial Analysis 详情页确认 `claude-sonnet-5` 的 `max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；主配置 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、1M context、输入/输出 `$2/$10/M` 和约 `$5.0912`/task。DataCurve DeepSWE v1.1 仍为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`；五档 Pass@1（max→low）为 53.846%/49.667%/48.230%/39.778%/30.512%，平均成本为 `$26.40/$11.89/$7.43/$4.08/$2.19`。这些数字绑定 effort、provider、harness、工具、环境、任务集和 verifier，不是裸模型排名。
- Anthropic 官方目录和发布页确认 `claude-sonnet-5`、2026-06-30、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、Adaptive、默认 high、Fast、平台、价格和 2026-01 cutoff 字段。完整文档补齐的面试技术点包括：手动 `thinking.type: "enabled"` + `budget_tokens` 在 Sonnet 5 上返回 400；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/文本共享的硬上限；thinking blocks/signatures 要按异构 `content` 回放；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 的长任务协议。
- System Card/发布页的 SWE-bench Verified 85.2%、SWE-bench Pro 63.2%、Multilingual SWE-bench 78.3%、Terminal-Bench 2.1 80.4%、BrowseComp 84.7%、OSWorld-Verified 81.2% 和 GDPval-AA v2 Elo 1618 单独作为 Anthropic 发布方结果记录。arXiv 精确标题 `"Claude Sonnet 5"` 截至本日返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告；参数、架构、完整训练/后训练 recipe、内部 adaptive-thinking 机制和独立 benchmark 复现仍待核验。
- Sonnet 5 当前已完成资料级闭环，不新增独立架构章节；研究笔记见 [`claude-sonnet-5-source-notes.md`](research/model-update-2026-09/claude-sonnet-5-source-notes.md)。下一模型候选必须继续从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的剩余重点厂商条目中选择。

## 2026-09-15：GLM-5.3-Flash 重新联网复验与内容专题闭环

- 针对用户提出的夜间工作时段可能造成网页访问中断，本轮重新发起联网复验。`10.237.126.170:1234` 与 `10.24.27.134:8098` 可获取 Artificial Analysis、DataCurve 和 Z.ai 官方页面；`10.24.27.134:7890` 作为备用线路，对 Artificial Analysis 大页面偶发 TLS EOF/读取超时。没有把单代理失败解释为页面不存在。
- Artificial Analysis 当前快照 `/tmp/glm53flash-aa-current-8098-20260915.html` 为 3,684,322 bytes，SHA-256 `7800ff202ced5e5cc170d7f1858d8070cf8d41c47b3ab6bace60b75c596b6319`；确认 `glm-5-3-flash`、`max`、页面 `releaseDate` `2026-08-26`、Intelligence Index `41.907366113455`、median output speed `114.22108687545 tokens/s`、median TTFT `2.45458272199994s` 和 1M context。输入/输出/cache hit 价格字段约为 `$0.15/$0.50/$0.026` 每百万 token，均是第三方配置字段。
- DataCurve 快照 `/tmp/glm53flash-deepswe-current-8098-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；`mini_swe_agent_glm_5_3_flash_max` 为 `max`、`n_runs=4`、284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本 `$0.2409818562`、平均输出 `72829.77` token、平均 Agent steps `122.89`。这是 `mini-swe-agent` + 工具 + 任务集 + 环境 + verifier 的系统结果，不是裸模型分数。
- Z.ai 官方文档快照 SHA-256 `a127bf7eff2780aacebfc4ffdcadfac5820b75caeaafdb932da0c8942eee879f`，博客正文通过 JS 资源读取，SHA-256 `225196c63b5944629606d26982c0a43c4a8fbd6edb8a7a2e9bf8abfacd35fcbb`；固定 Hugging Face revision 为 `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`，README/config 哈希已写入研究笔记。
- 官方资料确认 320B total/18B activated、`Glm5NextForConditionalGeneration`、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts/top-8/1 shared、1M position、IndexPool（4-key weighted pooling）、mHC（`hc_mult=4`、20 次 Sinkhorn）和原生 video/image/text/file 输入。视觉 coding loop 以 observe—render/use—verify—refine 为主线；serving 线索包括 SGLang、ReplaySSM、W8A8、INT8/FP8/BF16 hybrid cache quantization、Layer Split 和 EPD。
- 已新增研究笔记 [`glm-5.3-flash-source-notes.md`](research/model-update-2026-09/glm-5.3-flash-source-notes.md) 和第二十一册第 84 章 [`glm-5.3-flash混合注意力与视觉闭环.md`](book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)，并同步 `source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、第四册百科、第二十一册目录/时间线以及论文、题库、练习、术语、项目和知识图谱。
- 当前状态由“部分覆盖”升级为“内容专题闭环”。3.01×/4.44× attention/KV、约 3× serving、官方 benchmark 和视觉 workflow 数字均保留发布方自报边界；完整训练/后训练 recipe、生产 kernel、真实 state/KV/indexer/EPD bytes、硬件 profiling、线上 acceptance rate、API 错误/限流回归和独立 benchmark 仍待核验。下一锚点优先转向榜单已出现的 `deepseek-v4-flash-vision`，不从官方周边资料新增模型候选。

## 2026-09-15：DeepSeek V4 Flash Vision 断点恢复、专题验证与资料级闭环

- 针对夜间 20:00—次日 09:00 可能发生的代理中断，本轮重新发起当前时点复验。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Artificial Analysis Vision 详情页均返回 HTTP 200、3,609,278 bytes，SHA-256 均为 `a3ab00557356d09aca3729c2fead3900459ab73b1c63d31b03e7c3267200ea68`；三个代理对 DataCurve 均返回 HTTP 200、268,313 bytes，SHA-256 均为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；三个代理对 DeepSeek Quick Start 均返回 HTTP 200、46,116 bytes，SHA-256 均为 `6e2eb037db92ebef6a8f6408d87c12318c973388d6e27321606bb0e67dd67a6c`。`10.24.27.134:7890` 访问 Artificial Analysis 大页面仍 TLS EOF，但对另外两个入口成功；没有把代理失败解释成网页不存在。
- Artificial Analysis 当前锚点仍是 `deepseek-v4-flash-vision` 的 `max` 配置，`releaseDate` `2026-08-21`、1M context、284/13 目录参数、Intelligence Index `35.0122378035969`、约 `215.1792 tokens/s` 输出速度和 `1.2986s` median TTFT；DataCurve 当前没有 Vision 同名行，不迁移 V4 Flash/Pro 的 DeepSWE 分数。
- DeepSeek 官方实验公告、Vision/Files/Responses/价格文档已收口：历史 `deepseek-v4-flash-vision-exp` 的多模态 API、base64/URL/Files 输入和工具观察回灌，与当前 `deepseek-flash`/V4.1-Flash 路由分开记录；图像 detail/resize/token、`file_id` 生命周期、`function_call_output(input_image)`、兼容接口 capability probe 和端到端成本形成面试主线。历史公告的 384 image tokens 与当前 guide 的约 1024 tokens/image 保留日期、alias 和文档语境。
- 新增研究笔记 [`deepseek-v4-flash-vision-source-notes.md`](research/model-update-2026-09/deepseek-v4-flash-vision-source-notes.md) 与第二十一册第 85 章 [`deepseek-v4-flash-vision多模态api与路由账本.md`](book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)，并完成模型盘点、来源索引、榜单解释、第四册百科、第二十一册目录/时间线和书系配套同步。
- 第 85 章唯一 Python demo 已通过 AST、正常执行和 9 组边界测试；全库 679 个 Markdown 文件的 17,161 对围栏、1,704 个 Python 围栏 AST、932 个相对链接以及 `git diff --check` 均通过。全库链接审计使用平衡括号解析，避免把中文标点和代码内容误报为坏链接。
- DeepSeek V4 Flash Vision 当前由“部分覆盖”升级为“资料级闭环”。视觉专属参数/encoder、完整训练或后训练 recipe、独立技术报告、DataCurve Vision 结果、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验；下一锚点从两个排行榜剩余重点条目中选择，不从 DeepSeek 官方目录额外发现模型。

## 2026-09-15：GPT-5.4 资料级闭环与夜间代理恢复复验

- 针对用户所说的夜间 20:00—次日 09:00 代理中断，本轮重新发起联网搜索。`10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 对 Artificial Analysis 中文首页与 DataCurve DeepSWE 均返回 HTTP 200；三条线路取得的排行榜页面字节完全一致。AA `/zh` 快照为 `1,773,715` bytes、SHA-256 `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0`；DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- GPT-5.4 详情页通过 `7890` 返回 HTTP 200，快照为 `3,488,457` bytes、SHA-256 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`；页面仍确认 `GPT-5.4 (xhigh)`、`releaseDate=2026-03-05`、Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT、1.05M context，并标记 deprecated、指向 GPT-5.5。DataCurve 仍为 `mini_swe_agent_gpt_5_4_xhigh`：234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` steps；这些数字绑定 `mini-swe-agent`、工具、环境、任务集和 verifier，不是裸模型分数。
- OpenAI 官方模型页与专属指南本轮均实际抓取成功：模型页 `gpt-5.4-2026-03-05` 快照 SHA-256 为 `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5`，指南快照 SHA-256 为 `61a8e21bea3bc4592dd4eb19383a577a292ee770ae10f2e982aa788c349c04be`。已核验 1,050,000 context、128,000 max output、`none`—`xhigh` effort、text/image → text、deferred `tool_search`、built-in computer use、native compaction、custom tools/CFG、`allowed_tools`、tool preambles、Responses `phase`、reasoning 状态回放、prompt caching 和 server-side compaction。
- GPT-5.4 当日记录为“资料级闭环”：GPT-5.4/Pro 的关联关系已记录，mini/nano 在 2026-09-15 当时仍是待独立核验的关联变体；该历史状态已由 2026-09-20 的 mini/nano 独立资料级闭环更新。没有公开参数规模、激活参数、层/专家结构、训练与后训练 recipe、system card、独立技术报告或完整 benchmark 复现，因此不新增 GPT-5.4 专属正式章节。研究笔记见 [`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)，当前后续仍从两个排行榜的剩余重点厂商候选中选择，不从 OpenAI 官方目录另发现模型。

## 2026-09-15：Gemini 3.7 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能造成代理中断，本轮重新发起联网搜索。沙箱内三条代理均无法连接；经受控联网重试后，`10.24.27.134:8098` 成功获取 Artificial Analysis 中文首页、DataCurve、DeepMind Model Card 和 DeepMind Research，`10.24.27.134:7890` 成功获取 Google AI Developers 模型页和 arXiv，`10.237.126.170:1234` 也成功获取 arXiv/DeepMind Research。7890 对 Artificial Analysis 大页面和部分 Google 页面存在超时，不能把单代理失败解释为网页不存在。
- Artificial Analysis high 详情页本轮复验为 3,605,373 bytes、SHA-256 `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c`；页面确认 `Gemini 3.7 Flash (high)`、`releaseDate: 2026-08-13`、Intelligence Index `39.4295316404896`、约 `292.3389 tokens/s`、约 `10.2161s` input/TTFT 字段、1M context、proprietary 且参数字段为空。输入/输出/cache hit 价格字段约为 `$0.75/$3.75/$0.075` 每百万 token；这些都是 Artificial Analysis 的第三方配置字段。
- DataCurve fresh 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，页面仍为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`，包含 Gemini 3.7 Flash low/medium/high 四次运行配置。Pass@1 分别为 `53.7611%/65.4867%/65.2655%`，平均成本约 `$1.8323/$2.0251/$2.1763`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型能力。
- Google AI Developers 模型页 fresh 快照 SHA-256 `53e8a497f27951f2d533efa4609a9a45e816e5bda697a4d7470489c88ed9243a`，确认模型 ID `gemini-3.7-flash`、输入 text/image/video/audio/PDF、text 输出、1,048,576 输入 token、65,536 输出 token、low/medium/high thinking、`minimal` 报错、caching/code execution/file search/function calling/Maps/Search grounding/structured outputs/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference。
- DeepMind Model Card fresh 快照为 159,258 bytes、SHA-256 `b8051a19b7578abd1ab60e222f0afa829db38b415b66fbe91a24d6d89e3cdbc5`。官方只披露核心 reasoning foundation 的算法改进、agentic video understanding 和可调 thinking；架构、训练数据、硬件和软件信息指向 Gemini 3.6 Flash Model Card。没有把前代信息升级成 Gemini 3.7 独有事实。
- 已核验的面试技术线索包括：Interactions API 的 interaction/step、`previous_interaction_id`、store 保留/后台执行/implicit caching；tool context circulation、thought/tool signature 的 stateful/stateless 回放和 built-in/custom tool 执行边界；agentic video 的主动时间轴、transcript/帧/帧率/分辨率选择、`processing_call`/`processing_result` 以及 stateless processing step 回放。
- Google 官方评测 PDF 的发布方数字已单独记录：DeepSWE v1.1 65.3%、Terminal-Bench 2.1 85.8%、LVBench 85.4%、GDM-MRCR v2 128K 97.0%、OSWorld-2.0 47.9%。安全边界记录为 cybersecurity 达到 alert threshold 但未达到 CCL，CBRN 未达到 TCL/CCL；这些是 Google 自报安全/评测资料，不与 DataCurve 或 Artificial Analysis 拼接。
- arXiv 精确标题查询 [`title:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=title) 返回 0 个结果，快照 SHA-256 `a92b072d2026186568d09e3e42129649ee852886d6572d278cbb529f57b217a2`；全文查询找到 3 篇外部使用/评测论文：arXiv:2609.15983、2609.05232、2608.20563。它们分别研究多 Agent 研究 harness、执行资源契约和长时程安全 Agent 失败诊断，不是 Gemini 3.7 专属技术报告。
- 新增 [`gemini-3.7-flash-source-notes.md`](research/model-update-2026-09/gemini-3.7-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.7 Flash 当前为“资料级闭环”，暂无独立正式章节；当前锚点切换为 `gemini-3.7-flash`，下一锚点继续从两个排行榜的剩余重点厂商条目中选择。

## 2026-09-15：Gemini 3.6 Flash 断点恢复、联网复验与资料级闭环

- 针对用户所说的夜间 20:00—次日 09:00 可能导致代理中断，本轮重新发起联网搜索。三条代理对 Artificial Analysis 中文首页均返回 HTTP 200；随后通过可用线路重新获取 Gemini 3.6 详情/provider 页、DataCurve、Google AI Developers 模型页、DeepMind Model Card、Thinking、Interactions、工具组合、视频理解、Context caching、Computer Use 和 arXiv 检索页。个别代理的瞬时连接失败没有被解释为页面不存在。
- Artificial Analysis 详情页快照 `/tmp/gemini36-recheck-aa-8098-20260915.html` 为 3,600,704 bytes，SHA-256 `362fd94a45955783974462edb3cdc99ed932625a56d8c0afe7b9bca4a7f2e721`；页面确认 `Gemini 3.6 Flash (high)`、`releaseDate: 2026-07-21`、Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s`、1M context、约 `$0.15/$0.75/$3.75` cache-hit/input/output 每百万 token。provider 页当前只展示 `Google AI Studio` 一个 benchmark provider，不能与详情页 FAQ 的 provider 可用性数量混读。
- DataCurve 快照 `/tmp/gemini36-recheck-ds-8098-20260915.html` 为 268,313 bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；原始行是 `mini_swe_agent_gemini_3_6_flash_high`、`high`、4 runs、211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095066`、平均输出 `95,844.86` token、平均 Agent steps `116.73`、median peak context `151,398.5`。这是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型分数。
- Google AI Developers 模型页快照 `/tmp/gemini36-recheck-ai-7890-20260915.html` 为 104,839 bytes，SHA-256 `e0d55ee75ca1c4e80c128284a907de9e2dd602d9a6a39e660599abea1bf43404`；确认 `gemini-3.6-flash`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/thinking/URL context/Computer Use Preview，以及 Batch/Flex/Priority inference；稳定 alias 为 `gemini-3.6-flash`，latest update 为 July 2026。
- Thinking 文档明确 3.6 的默认档位为 `medium`，支持 `minimal/low/medium/high`；Interactions 文档模型表列出 3.6；视频文档将 3.6 列入 agentic video 模型，Context caching 文档给出 implicit caching 最低输入 `4,096` tokens。Gemini 3 工具组合文档记录 thought/tool signatures 与 tool context circulation；这些是 API/Agent 协议事实，不是内部 Transformer 机制。
- DeepMind Model Card 快照 `/tmp/gemini36-recheck-card-8098-20260915.html` 为 154,640 bytes，SHA-256 `c2e25fec9cc337856c4f612b729f8d1b3985a0627a7234e66091b53dddd9e1c3`。官方明确 3.6 based on Gemini 3.5 Flash，架构、训练数据、数据处理、硬件和软件资料指向 3.5 Model Card；公开 benchmark 包括 SWE-Bench Pro `58.7%`、DeepSWE v1.1 `49%`、Terminal-Bench 2.1 `78.0%`、GDPVal-AA v2 `1421`、OSWorld-Verified `83.0%`、GDM-MRCR v2 128K `91.8%`/1M `54.0%`。安全表和 Frontier Safety 结论按发布方/自动评测边界记录，未迁移成 3.6 独有训练事实。
- arXiv 精确标题检索快照 `/tmp/gemini36-recheck-arxiv-title-7890-20260915.html` 为 16,452 bytes、SHA-256 `03b6d5dabe226f9c165a6f725f670f2f846659fcc7308c037b9c0a4fe8db9127`，返回 0 篇；全文检索快照 `/tmp/gemini36-recheck-arxiv-all-7890-20260915.html` 为 63,191 bytes、SHA-256 `2e5c93fde5c302f8a1cd655425d9df2e885c31f4a904318c90f00db6ef372ffa`，返回 9 篇外部使用/评测论文，不是 Gemini 3.6 专属技术报告。9 篇的标题、摘要要点和证据边界已写入研究笔记。
- 新增 [`gemini-3.6-flash-source-notes.md`](research/model-update-2026-09/gemini-3.6-flash-source-notes.md)，并同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.6 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；当前锚点为 `gemini-3.6-flash`，下一锚点仍从两个排行榜剩余重点条目选择。

## 2026-09-15/16：Gemini 3.5 Flash 断点恢复、联网复验与资料级闭环

- 针对用户提出的 20:00—次日 09:00 代理中断规则，本轮重新发起联网收集。2026-09-15 工作时段重新抓取时，`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Gemini 3.5 Flash 的 Artificial Analysis 与 DataCurve 均返回 HTTP 200，两个代理取得的 AA 详情和 DataCurve 快照逐字节一致；7890 成功获取 Google AI Developers、DeepMind Model Card 和 Google API 周边文档。2026-09-16 轻量复探中，三个代理对 AA/DataCurve 均返回 HTTP 200，7890 对 Google 模型页返回 HTTP 200，8098/1234 对该大页面超时；代理差异仍不被解释成页面不存在。
- Artificial Analysis 快照为 3,607,081 bytes、SHA-256 `abdbed0cf8069ae81c272aea386724302a0659235fa5b950980a9a9a0d162ec7`；页面确认 `Gemini 3.5 Flash (high)`、第三方 `releaseDate: 2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、约 `18.3631s` TTFT、1M context 和约 `$1.5625`/task。当前页面的 `deprecated: true`/`deprecatedTo: gemini-3-6-flash` 只作为 Artificial Analysis 目录状态记录，不能写成 Google 官方退役公告。
- DataCurve 快照为 268,313 bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，仍为 113 tasks、91 repositories、5 languages、4 runs、统一 `mini-swe-agent`。`mini_swe_agent_gemini_3_5_flash_high` 原始行是 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均输出 `75,730.19` token、平均 Agent steps `105.30`；这些是配置 + harness + 工具 + 环境 + verifier 的系统结果。
- Google 官方 API 页面确认 1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/URL context、Computer Use Preview、Batch/Flex/Priority inference；What's New 页面补充 GA、默认 effort 从 high 改为 medium、`minimal/low/medium/high`、legacy `thinking_budget` 互斥、thought preservation 和 2025 年 1 月 knowledge cutoff。
- Interactions API 模型表明确支持 `gemini-3.5-flash`；工具组合文档的 `id`/`signature`、tool context circulation、stateful/stateless 回放和宿主执行责任已入库。Context caching 文档明确 implicit caching 最低输入 `4,096` tokens。Computer Use 文档明确 3.5 Flash 支持 opt-in screenshot prompt-injection detection，默认关闭。
- 视频证据保持保守：API 支持视频输入，但当前视频文档的 agentic processing 列表明确列出 3.8/3.7/3.6 和 3.5 Flash-Lite，没有明确列出 3.5 Flash；因此不把 3.6 的 `processing_call`/`processing_result` agentic video 结论迁移给 3.5 Flash。
- DeepMind Model Card 的发布方 benchmark、安全表和 Frontier Safety 结论已记录；Model Card 明确 architecture、training dataset、data processing、hardware 和 software 均指向 Gemini 3 Flash Model Card。arXiv 精确标题检索返回 0 篇，全文检索返回 30 篇外部使用/评测论文，没有检出 Gemini 3.5 Flash 专属技术报告。
- 新增 [`gemini-3.5-flash-source-notes.md`](research/model-update-2026-09/gemini-3.5-flash-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。Gemini 3.5 Flash 当前由“仅候选”升级为“资料级闭环”，暂无独立正式章节；下一锚点优先选择 `gemini-3.1-pro-preview`，仍必须先由两个排行榜确认。

## 2026-09-16：Gemini 3.1 Pro Preview 断点恢复与资料级闭环

- 按用户设定的夜间 20:00—次日 09:00 中断规则，本轮重新收集 Artificial Analysis 中文首页、Gemini 3.1 Pro 详情页和 DataCurve DeepSWE。三条代理均返回 HTTP 200；AA 首页、详情页和 DataCurve 的同一页面快照逐字节一致。随后通过可用线路重新获取 Google API 模型页、Release Notes、DeepMind Model Card、评测方法、Thinking、thought signatures、工具组合、Long context、Caching 和 arXiv 检索页。
- Artificial Analysis 详情快照 `/tmp/overnight-aa-g31-1234-20260916.html` 为 `3,607,783` bytes，SHA-256 `ba842f82908f6237acd84c1ed0d1ab03d4652af503c8de6d00b56bb21b7731eb`；确认 `gemini-3-1-pro-preview`、第三方 `releaseDate` `2026-02-19`、Intelligence Index `30.3596656132261`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT 和 1M context。
- DataCurve 快照 `/tmp/overnight-deepswe-1234-20260916.html` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；确认 `mini_swe_agent_gemini_3_1_pro_preview_high`、`reasoning_effort: high`，53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均输出 `28,368.88` token、平均 Agent steps `75.56`。这些结果绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不是裸模型能力。
- Google 官方资料确认 2026-02-19 发布、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、thinking、function calling、grounding、structured outputs 和 context caching；`gemini-3.1-pro-preview-customtools` 是 bash/custom tools 工具优先级优化的 endpoint variant，不是新基础模型。
- Model Card 明确 `Gemini 3.1 Pro is based on Gemini 3 Pro`，并将架构、训练数据、数据处理、硬件和软件资料指向 Gemini 3 Pro；因此本轮不把前代资料升级为 3.1 独有架构或训练事实。Model Card 的发布方 benchmark、安全和 Frontier Safety 结果单独保留，不与两个排行榜拼接。
- 面试主线已收口为：`thinking_level` 与共同 output budget；thought/tool `signature` 和 `id` 的 stateful/stateless 回放；tool context circulation；built-in/custom tool 的宿主责任；1M context 的缓存、召回和成本账本；评测设置与 CCL 证据边界。arXiv 标题精确检索为 0，全文 198 篇均为外部使用/评测结果，没有发现 Google 发布的 Gemini 3.1 Pro 专属技术报告。
- 已修正研究笔记中 Long context 快照哈希为 `cbed824426278d5f59e46f59b477eb268d5b8056970b29fe246406188b02cd78`，并同步研究笔记、来源索引、模型盘点和计划。Gemini 3.1 Pro Preview 当前为“资料级闭环”，暂无独立正式架构章节；下一主锚点仍只能从两个排行榜剩余重点条目选择。

## 2026-09-16：GLM-5 AA 单榜资料闭环

- 本轮重新使用三条用户提供的代理抓取 [Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)。三条线路对两个页面均返回 HTTP 200；AA 快照为 `3,577,227` bytes、SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`，DataCurve 快照为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- AA 有精确 `GLM-5 (Reasoning)`/`glm-5` 条目，并列 `glm-5-non-reasoning`；页面第三方字段约为 744B total、40B active、200K context、72.4 tokens/s 和 1.34s TTFT。DataCurve 当前只检出 GLM-5.2、GLM-5.3 和 GLM-5.3 Flash 配置，没有精确 `GLM-5` 行，因此不记录 GLM-5 的 DataCurve 分数，也不迁移相邻版本结果。
- Z.ai 官方模型卡、[GLM-5 专属技术报告](https://arxiv.org/abs/2602.15763)、API 文档和 GitHub 已核验：744B total/40B active、28.5T 预训练 tokens、DSA、`GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048`、202752 position，以及面向长周期 Agent 的 `slime` 异步 RL 基础设施。
- 面试主线为：DSA indexer/top-k 召回与端到端成本；MoE 总参数、active compute、通信和显存的区分；rollout/trainer 解耦带来的 policy lag、样本新鲜度和 off-policy 风险；长轨迹 Agent RL 的 verifier 与 credit assignment；Agentic Engineering 中规划—工具—执行—测试—修复闭环；模型、API、harness、工具、环境和 verifier 的评测分层。
- 模型卡 README 的 HLE、SWE-bench、Terminal-Bench、BrowseComp、MCP-Atlas 等数字均是发布方在明确 prompt、最大生成长度、温度、harness、judge、超时和重复次数下的结果，不与 AA 或 DataCurve 拼成裸模型能力。
- 新增 [`glm-5-source-notes.md`](research/model-update-2026-09/glm-5-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。GLM-5 当前为“AA 单榜资料闭环”，不新增独立正式章节；完整 DSA indexer 训练目标、生产 kernel、`slime` 调度、完整训练/后训练 recipe、硬件 profiling、线上接受率和独立复现仍待核验。

## 2026-09-16：GLM-5.2 双榜资料级闭环

- 按夜间 20:00—次日 09:00 可能中断的规则重新抓取 [Artificial Analysis GLM-5.2](https://artificialanalysis.ai/models/glm-5-2) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)，两个页面均返回 HTTP 200。AA 快照 `/tmp/glm52-aa-20260916.out` 为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve 快照 `/tmp/glm52-ds-20260916.out` 为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 两个排行榜均有精确 GLM-5.2 配置：AA 为 `glm-5-2` 的 max/non-reasoning，DataCurve 为 `mini_swe_agent_glm_5_2_high` 和 `mini_swe_agent_glm_5_2_max`。DataCurve high/max 的 Pass@1 分别为 `36.2832%`/`43.7778%`，平均成本约 `$2.8355`/`$3.9199`，平均 Agent steps `121.88`/`129.13`；结果绑定 `mini-swe-agent`、4 runs、113 tasks、工具、环境和 verifier，不是裸模型能力。
- Z.ai 官方 [GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2) 快照为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。官方确认 1M context、128K 最大输出、thinking、function calling、context caching、structured output、MCP，以及项目级代码库、跨文件重构和分阶段验证的长周期工程工作流。
- “lossless context”、数月 Coding Agent 专项训练、开发者案例和长任务 benchmark 只按 Z.ai 发布方描述记录；不能解释成数学意义上的绝对无损，也不能把提示词中的 `/goal`、CLAUDE.md/Agent.md、ADB/logcat 或工作流案例写成内部算法。GLM-5.3 引用的 SAO with compaction 仍没有 GLM-5.2 原始定义。
- 已更新 [`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表。GLM-5.2 当前为“资料级闭环”，正式专题见第二十一册第 86 章；下一主锚点继续从两个排行榜的剩余重点候选中选择。

## 2026-09-17：联网恢复复核结果

- 为补查昨晚 20:00—今早 09:00 可能中断的联网任务，本轮重新测试三条用户提供的代理：`10.237.126.170:1234`、`10.24.27.134:8098`、`10.24.27.134:7890`，当前时点均无法连接代理服务器；Artificial Analysis 与 DataCurve 通过代理均返回 HTTP `000`，curl 退出码 `7`。
- 直连 `https://www.baidu.com` 返回 HTTP `200`、约 `2443` bytes，说明当前基础网络并非完全中断；Artificial Analysis、DataCurve、Z.ai 和 Google 直连均因 DNS 解析失败，curl 退出码 `6`。因此本轮将状态记录为“代理入口不可用/目标站点直连 DNS 失败”，不解释为目标网页不存在。
- 本轮没有把 IFM/K2-Horizon-7B-Uno 等非重点厂商条目提升为新锚点；网络恢复后仍只从 Artificial Analysis 与 DataCurve DeepSWE 重新确认 OpenAI、Anthropic、GLM、Qwen、Kimi、DeepSeek、Gemini、Grok 的新条目，再继续官方资料核验。

## 2026-09-17/18：排行榜恢复复核与 GPT-6 Astra 运行时资料补证

- 受控联网恢复后，三条用户提供的代理均可访问百度并返回 HTTP `200`；随后三条线路重新抓取 Artificial Analysis 中文首页和 DataCurve DeepSWE，均成功且逐字节一致。Artificial Analysis 快照为 `1,782,207` bytes、SHA-256 `d547f7bde6adc164aaea60026d06e9a59178fc86d89c8ce5333fe8d28ffb755a`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 与已有快照比较，八家重点厂商没有新增基础模型候选；DataCurve 的模型集合也没有新增或删除。页面中出现的版本日期/目录条目变化不提升为新锚点，仍按基础模型、推理配置和 Agent harness 三层记录。
- GPT-6 Astra 仍同时出现在 Artificial Analysis 与 DataCurve：DataCurve 保留 `gpt-6-astra` 的 xhigh 配置；Artificial Analysis 主榜仍列 `GPT-6 Astra (max)`。本轮不把 AA 指数或 DeepSWE Pass@1 当作裸模型能力。
- 通过 `10.24.27.134:7890` 实际读取 OpenAI 官方 GPT-6 Astra 模型页、Reasoning、Prompt caching、Conversation state、Compaction 和 Tool search 文档。补齐的面试主线包括：动态 `configuration_update`、reasoning/`phase` 完整 item replay、`previous_response_id` 的状态与计费边界、deferred tool search 的缓存前缀/权限审计，以及 server-side/standalone compaction 的 canonical context。
- 已同步 [`gpt-6-astra-source-notes.md`](research/model-update-2026-09/gpt-6-astra-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md)、第六册第 18 章和面试配套文件。GPT-6 Astra 当前维持“内容专题闭环”，但没有公开参数、架构、训练 recipe、system card 或专属技术报告；下一锚点继续从两个排行榜中剩余的八家重点厂商条目选择。

## 2026-09-18：Qwen3.8 Max (0902) revision 资料级闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮在该目录继续，保留已有用户修改。
- 重新核对两个唯一排行榜：Artificial Analysis canonical `qwen3-8-max` 当前页面标题为 `Qwen3.8 Max (0902)`，release slug 为 `qwen3-8-max-0902`，详情快照 `3,607,154` bytes、SHA-256 `e9152a5d81063cbeb45fb21ba621d7eb7da05073235b611f27a344f38c6288ae`，第三方 Intelligence Index `45.4354834980521`、context 约 `984K`；本轮 `/zh` 首页快照为 `1,769,534` bytes、SHA-256 `ce8fb0d2ca827cead0642152b02716022936060bfde1d74addbfa6657bc48f46`；DataCurve 当前只有泛化 `mini_swe_agent_qwen3_8_max_xhigh`。
- DataCurve 泛化行记录为 258/449、Pass@1 `57.4610%`、Pass@4 `83.1858%`、平均成本约 `$3.7291`、平均输出 `95,075` tokens、平均 Agent steps `111.34`、4 runs。由于没有 `qwen3_8_max_0902` 精确行，已明确禁止把该结果绑定成 0902 revision 的独立成绩。
- 通过 [Qwen3.8-Max-0902](https://www.qwencloud.com/models/qwen3.8-max-0902)、[Thinking](https://docs.qwencloud.com/developer-guides/text-generation/thinking)、[Function Calling](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling) 和 [Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache) 完成官方核验。产品页 alias 为 `qwen3.8-max-2026-09-02`，称其为 `qwen3.8-max` 的 upgraded snapshot，`last-modified` 为 `2026-09-18 10:45:04`。
- 已记录官方服务边界：1M context、991K 普通最大输入、983K thinking 最大输入、131K 最大输出；输入 `$2/M`、输出 `$6/M`、implicit cache `$0.25/M` 的价格字段按页面快照记录。官方产品定位包括 coding、工程规模项目、长周期 autonomous development、多工具 Agent 和视觉理解，但没有把这些描述升级成新架构事实。
- 已记录 `reasoning_effort` 的 `low/medium/xhigh`（默认 `xhigh`）与 `thinking_budget` 互斥；thinking 模式下 `tool_choice` 只能为 `auto`/`none`，强制工具选择需要关闭 thinking；多模态调用使用 `MultiModalConversation`。
- 已记录 explicit、implicit、session cache 的不同生命周期/命中/计费语义和最小 `1,024` tokens；没有把 provider cache 直接等同为永久 GPU KV cache。
- 新增 [`qwen3.8-max-0902-source-notes.md`](research/model-update-2026-09/qwen3.8-max-0902-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 及书系配套文件，并扩展第二十一册第 83 章和第二十四册工具 serving 章节；不新增重复架构章节。
- 当前状态：Qwen3.8 Max (0902) 为“资料级闭环”。参数、层排布、专属技术报告、完整训练/后训练 recipe、生产 kernel、线上 acceptance rate、目标硬件 profiling 以及 hosted 0902 与 A95B 的精确服务差异仍待核验；下一轮仍从两个排行榜中的八家重点厂商条目选择锚点。

## 2026-09-18：GPT-5.3 Codex 锚点资料级闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮继续在该目录工作并保留既有修改。
- 重新核对两个唯一排行榜：Artificial Analysis `/zh` 快照为 `1,769,512` bytes、SHA-256 `d50456b4597b637829b46332b3991a8f6a004507341c3fc4d74bc390f42f3cff`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。GPT-5.3 Codex 出现在 Artificial Analysis 详情页；DataCurve 没有精确 `mini_swe_agent_gpt_5_3_codex_*` 行。
- Artificial Analysis 详情快照 `/tmp/gpt53codex-aa-detail-20260918.out` 为 `3,528,646` bytes、SHA-256 `565d91572b1a0bd9fb8f7f89f16c8beefbaadfdea79de5b229a9bd5a997b27ef`；页面标题 `GPT-5.3 Codex (xhigh)`、release date 字段 `2026-02-05`、Intelligence Index `32.5028174368983`（estimated）、context `400,000`、knowledge cutoff `2025-08-31`。这些是第三方配置字段。
- 通过 `10.24.27.134:7890` 获取 OpenAI 官方 [GPT-5.3-Codex 模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md) 和 [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)，并复核 Compaction、Conversation state、Tools、Agents、Prompt caching 文档。确认 `gpt-5.3-codex`、`low/medium/high/xhigh`、文本/图像输入、文本输出、400K context、272K maximum input、128K maximum output、Responses-only、function calling、web search、hosted shell、skills 和当前价格字段。
- 新增 Codex 运行时面试主线：较少 reasoning tokens、交互式任务优先 medium、困难任务使用 high/xhigh、长时自治、first-class compaction、工具 schema/并行调用/`apply_patch`/固定工作目录、assistant `phase`、完整 output item replay、opaque reasoning、canonical compaction context、KV prefix cache，以及模型输出/宿主授权/工具执行/artifact 的四段式责任边界。
- 明确 `tool_search` 不能从 GPT-5.4+ 文档迁移到 GPT-5.3 Codex；同时不把 GPT-5.5/5.6/5.4 或其他 Codex 配置的 DataCurve 结果迁移到本锚点。GPT-5.3 Codex 当前为“资料级闭环”，没有公开参数、架构、训练 recipe、system card、专属技术报告、生产 kernel、硬件 profiling 或线上 acceptance rate。

## 2026-09-16：Claude Sonnet 4.6 断点恢复、联网复验与资料级闭环

- 针对昨晚 20:00—今早 09:00 可能发生的代理中断，本轮重新使用三条代理抓取 Artificial Analysis 中文首页、DataCurve DeepSWE 和 Anthropic Sonnet 4.6 发布页，均返回 HTTP 200。三个代理取得的 Artificial Analysis 首页均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`；DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`，未发现截断或代理间差异。
- Sonnet 4.6 发布页三条线路均返回 HTTP 200、`281,283` bytes；动态响应哈希不同，但核心发布信息一致。`8098/1234` 访问 Anthropic 模型目录被重定向到区域不可用页，不能作为目录证据；`7890` 成功取得真实 `platform.claude.com` Models Overview，`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`，确认 `Claude Sonnet 4.6`、`claude-sonnet-4-6`、1M context 和 128K output。
- Artificial Analysis 的 Sonnet 4.6 adaptive 快照为 `3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`；DataCurve 精确行 `mini_swe_agent_claude_sonnet_4_6_high` 为 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、平均成本约 `$5.5224`、平均输出 `76,160.31` token、平均 Agent steps `133.66`。这些属于榜单配置和 `mini-swe-agent` 系统结果，不能写成裸模型能力。
- Anthropic 官方资料确认 2026-02-17 发布、coding/computer use/long-context reasoning/agent planning/knowledge work/design 定位、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory 和 programmatic tool calling。Computer-use prompt injection 风险虽称相较 Sonnet 4.5 改善，但不写成已解决。
- Claude Sonnet 4.6 当前为“资料级闭环”。没有公开参数规模、内部架构、完整训练/后训练 recipe 或独立技术报告；adaptive 预算、compaction 摘要格式、tool search 质量、线上 computer-use 接受率和独立复现仍待核验，不新增独立正式架构章节。研究笔记见 [`claude-sonnet-4.6-source-notes.md`](research/model-update-2026-09/claude-sonnet-4.6-source-notes.md)。

## 2026-09-18：OpenAI gpt-oss 内容专题闭环

- 确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。
- Artificial Analysis 当前存在 `gpt-oss-120b` 与 `gpt-oss-20b` 精确条目，详情快照分别为 `3,701,133` 和 `3,701,427` bytes；DataCurve 当前没有精确的 `mini_swe_agent_gpt_oss_120b_*`/`20b_*` 行，因此没有迁移其他 GPT/Codex 的 Agent 结果。
- OpenAI Model Card/arXiv、官方仓库、Harmony、Hugging Face 模型卡/配置与 Cookbook 已核验：120B/20B total/active 参数、128/32 experts、top-4、交替 sliding/full attention、GQA、RoPE/YaRN、MXFP4、`o200k_harmony`、CoT RL、Harmony channels/recipients 和 `low/medium/high` variable-effort reasoning。
- 已创建第二十一册第 87 章并加入目录；同步研究来源、模型盘点、榜单解释和 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`，并将工具权限/协议和 serving 账本映射到第六、五、十七、二十四册相关章节。
- 当前状态为“内容专题闭环”。完整训练/后训练 recipe、router 负载均衡、MXFP4 kernel/量化误差、目标硬件 profiling、线上接受率和独立 gpt-oss Agent 评测仍待核验；下一轮继续只从两个排行榜选择锚点。

## 2026-09-18：Claude Opus 4.6 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留此前 dirty worktree 修改。Artificial Analysis 详情页确认 `claude-opus-4-6-adaptive` 与 `claude-opus-4-6`，DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行。
- AA adaptive/基础详情快照分别为 `3,497,470`/`3,485,093` bytes，SHA-256 分别为 `0bcf25542c95489db50a83925caf5e32bab672e4f7134126009fb94937b6b32d` 与 `544de10658d16de8ac25da44e6c6792ab51b64093a5a410f3eb3be1173e6d219`。第三方字段包括 `releaseDate=2026-02-05`、Intelligence Index 约 32（estimated）、1M context、约 37.7 output tokens/s、约 19.75s TTFT 和 `$5/$25` 每百万 token；不把这些字段写成 Anthropic 内部架构事实。
- Anthropic 官方资料已核验：发布页、System Card、模型页、Adaptive thinking、Effort、Compaction、Tool search、Computer use、Advanced tool use 和 context engineering。确认 1M context、128K 普通/300K Batch 最大输出、默认 high 与 `max/high/medium/low` effort、thinking signature 原样回传、`compact-2026-01-12` compaction、约 150K 默认/50K 最低 trigger、regex/BM25 tool search、最多 5 个 `tool_reference`，以及 `computer_20251124` 的宿主执行器边界。
- 发布方 1M retrieval、BigLaw Bench、盲测和 subagent 数字与 AA 指数分开保存；arXiv 精确检索未发现 Anthropic 的 Opus 4.6 专属技术报告，外部论文只作为使用/评测证据。完整研究笔记见 [`claude-opus-4.6-source-notes.md`](research/model-update-2026-09/claude-opus-4.6-source-notes.md)。
- 已同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`plan_v2.md`](plan_v2.md) 和书系配套面试资料。当前状态为“资料级闭环”，不新增独立 Transformer 章节；参数、架构、完整训练/后训练 recipe、adaptive 内部机制、compaction 编码、生产 kernel、硬件 profiling、线上 acceptance rate 和独立 benchmark 复现仍待核验。下一轮继续从两个唯一排行榜的八家重点厂商条目选择锚点。

## 2026-09-18：Claude Opus 4.7 正式章节收口

- 已在工作区恢复后继续上一轮遗留任务，确认 `/home/zzc/llm-from-zero-to-interview` 是 `/data/zzc/llm-from-zero-to-interview` 的同一项目目录。
- 已新增第二册 Agent 章节 `7.28 Claude Opus 4.7：把长任务预算和视觉输入纳入 Code Agent`，补充 `effort`、task budget、`max_tokens` 三层预算、tokenizer 迁移、高分辨率视觉、compaction 和 cyber safeguards 的面试/工程边界。
- 已新增第二十四册 serving 章节 `32.36 Claude Opus 4.7：三层预算、高分辨率视觉与安全控制面`，补充 serving manifest、预算账本、视觉输入账本、缓存/compaction 关系、安全策略和生产验收表。
- Opus 4.7 现在具备“排行榜发现 → 官方资料 → 研究笔记 → 正式章节 → 题库/练习/术语/项目/论文/知识图谱同步”的内容专题闭环；仍不把它升级为内部架构或独立 benchmark 已公开的结论。
- 下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择锚点；已入库模型优先补齐版本变化、独立复现和 serving profiling，不从官方目录或论文搜索另发现模型。
## 2026-09-18：Kimi K3 官方技术报告与仓库核验

- 本轮在 `/data/zzc/llm-from-zero-to-interview` 继续，确认其与用户指定的 `/home/zzc/llm-from-zero-to-interview` 为同一项目；保留既有 dirty worktree 修改。
- 重新抓取两个唯一排行榜：Artificial Analysis K3、Artificial Analysis `/zh` 首页和 DataCurve DeepSWE。当前可用代理为 `10.237.126.170:1234`；`10.24.27.134:8098` 当前超时，`10.24.27.134:7890` 当前拒绝连接。代理失败只记录为线路状态，不解释为网页不存在。
- 新鲜快照：AA K3 `/tmp/aa-k3-1234b-20260918.out`，3,696,739 bytes，SHA-256 `258cafa80dba24b5528f0933a8c45c0e1c0b41d1064ababebef77680ecd8a310`；AA `/zh` `/tmp/aa-home-1234b-20260918.out`，1,769,161 bytes，SHA-256 `4c0304f1955031c61e41a078a3028df13bdff36fa1aacd384c09e19a513e9aa0`；DataCurve `/tmp/ds-1234-20260918.out`，268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 已核验 MoonshotAI 官方仓库 `MoonshotAI/Kimi-K3`、README、`k3_tech_report.pdf` 和 Kimi K3 License。技术报告 PDF `/tmp/k3-tech-report-20260918.pdf` SHA-256 `86fb82a63ced501f0c3f4f404c0c6fa88a7a6cfac17aae81fd1a8f455998067c`；报告正文提取到 `/tmp/k3-tech-report-extracted-20260918.txt`。
- K3 核验配置：2.8T total、104B activated、93 layers、69 KDA + 24 Gated MLA、每 3 个 KDA 后接 1 个 Gated MLA、末尾 Gated MLA、8 个 Block AttnRes/每 block 12 层、896 routed experts/16 selected/2 shared、1,048,576 context、MoonViT-V2、MXFP4 weights + MXFP8 activations QAT、SiTU-GLU 和 Quantile Balancing。
- 报告新增面试主线：三条扩展轴；lower-bounded decay (`g_min=5`)；latent-width routed experts + full-width shared experts；冻结 bias + fixed Top-k；Per-Head Muon；multi-teacher on-policy distillation；KDA context parallelism、MoonEP、长轨迹 RL、外部 KV retention、可恢复 microVM sandbox、budget-aware effort scheduling；XTM 的 `[open]/[sep]/[close]/[end_of_msg]`、`think/response/tool` channel、动态 `tool-declare` 和 tool/index 配对。
- 许可证已核验，覆盖 weights、parameters、configuration、code 和 documentation，并含商业条件。README 声明完整权重已发布并给出 HF/ModelScope 入口，但具体权重仓库、文件和 revision 本轮未稳定确认，不能写成“已下载权重”。
- 已同步研究底稿、来源索引、榜单解释、计划和相关 Agent 章节；第二十一册新增第 88 章，KDA/AttnRes 旧章补充 K3 report 与前作论文的边界。
- 当前状态：Kimi K3 从“资料级、报告待核验”升级为“内容专题闭环”。仍待核验具体权重下载/revision、FlashKDA 代码细节、vLLM prefix-cache 合并状态、完整训练 recipe、硬件 profiling、独立 benchmark 复现和线上 acceptance rate。下一锚点仍只能从两个排行榜的八家重点厂商条目选择。

## 2026-09-18：Gemini 3 Deep Think 配置级锚点闭环

- 确认工作目录 `/data/zzc/llm-from-zero-to-interview` 与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留此前 dirty worktree 修改。
- Artificial Analysis 已确认精确 `Gemini 3 Deep Think` 条目；DataCurve 当前没有精确 `mini_swe_agent_gemini_3_deep_think_*` 行，因此不迁移 Gemini 3.1 Pro、Gemini 3.7 Flash 或其他 Gemini 配置的 Pass@1、成本和 Agent steps。
- Google 官方同名 API 模型页、DeepMind Model Card 和 Blog 路径本轮没有提供独立 Deep Think model ID、权重或技术报告；官方可核验对象是 Gemini 3.1 Pro / Gemini 3 系列的 Deep Think 运行配置与评测设置。研究笔记见 [`gemini-3-deep-think-source-notes.md`](research/model-update-2026-09/gemini-3-deep-think-source-notes.md)。
- 已同步 `plan_v2.md`、`source-index.md`、`model-inventory.md` 和 `inventory-interpretation.md`。面试主线是 `thinking_level` 与共同 output budget、thought/tool signature 和 `id` 回放、1M context/caching，以及 test-time compute、能力、延迟、成本和安全评测的联合分析；不新增独立架构章节。
- 当前状态：资料级闭环（配置级锚点）。Artificial Analysis 的指数、约 130K context、速度、价格和 provider 字段仍仅作为第三方目录信息；独立 checkpoint、参数、架构、训练 recipe、独立 benchmark 和精确 DataCurve Agent 评测仍待核验。

## 2026-09-18：Qwen3.5-397B-A17B 内容专题闭环

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留已有 dirty worktree 修改。
- Artificial Analysis 精确确认 `Qwen3.5 397B A17B`，页面的 Reasoning/Non-reasoning 按同一基础模型归并；详情快照 `/tmp/qwen35-397-aa.html` 为 3,700,616 bytes，SHA-256 `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719`。DataCurve 当前没有精确 `mini_swe_agent_qwen3_5_397b_a17b_*` 行，因此不迁移其他 Qwen 模型的 Agent 分数。
- 通过 Qwen 官方模型卡、`config.json`、QwenLM/Qwen3.8 官方仓库和 Qwen3.5 发布博客核验 397B total/17B active、60 层、hidden 4096、15 组 `3 x Gated DeltaNet + 1 x Gated Attention`、512 experts、10 routed + 1 shared、原生 vision encoder、262,144 native context、约 1,010,000 YaRN context 和 MTP serving 示例。
- 官方模型卡/博客还声明 early-fusion 多模态训练、trillions of multimodal tokens、million-agent RL environments、asynchronous RL framework 和 201 languages/dialects；这些内容按 Qwen 官方自报记录，不能升级为独立复现或完整训练 recipe。
- 新增 [`qwen3.5-397b-a17b-source-notes.md`](research/model-update-2026-09/qwen3.5-397b-a17b-source-notes.md)，同步来源索引、模型盘点、榜单解释、`plan_v2.md`；第二十一册第 83 章新增 Qwen3.5 前置/对比小节并更新目录。
- 明确证据边界：Qwen3.8 的 QSA、Gated Residual、N-gram Embedding 和 Muon 不能反向迁移为 Qwen3.5 技术；GDN/GA 完整 kernel、state layout、MTP acceptance length、视觉独立复现、硬件 profiling 和 Qwen3.5-Plus hosted/open 精确差异仍待核验。
- 当前状态：Qwen3.5-397B-A17B 为“内容专题闭环”；下一轮仍只能从 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：GLM-5 DSA、slime 与 Agentic Engineering 正式专题闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。重新抓取两个唯一排行榜：Artificial Analysis 当前页面标题为 `GLM-5 (Reasoning)`，并提示已有更新模型 `GLM-5.1`；DataCurve 仍没有精确 `mini_swe_agent_glm_5_*` 行。
- 新鲜快照：AA `/tmp/glm-aa-20260920.out` 为 `3,811,809` bytes，SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`；DataCurve `/tmp/aa-check-8098-ds-20260920.out` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。不迁移 GLM-5.2/5.3/5.3 Flash 的 DeepSWE Pass@1、成本或 steps。
- Z.ai 博客资源 `/tmp/glm-blog-js-20260920.out` 为 `145,189` bytes，SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`，可复验 `GLM-5`、`DSA`、`slime`、`744B`、`40B`、`28.5T` 和 `Agentic` 等入口正文字符串。`https://docs.z.ai/guides/llm/glm-5.md` 及无扩展名文档页本轮由代理返回 503；这只记录为线路失败，不解释为官方页面不存在。
- 新增第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)，并更新该册目录和架构创新时间线。章节将 DSA indexer/top-k 召回、744B/40B total/active 分账、MLA 低秩字段、`slime` rollout/trainer 解耦、policy lag/freshness、长轨迹 verifier 和最终 artifact gate 组织成面试主线。
- 已同步 [`glm-5-source-notes.md`](research/model-update-2026-09/glm-5-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 以及本进度文件；随后补齐论文、题库、练习、术语、项目和知识图谱。
- 当前状态：GLM-5 升级为“内容专题闭环（AA 单榜）”。DataCurve 精确评测、完整 DSA indexer loss/recall、生产 kernel、`slime` 调度与 freshness 控制、完整训练/后训练 recipe、硬件 profiling、线上 tool acceptance 和独立复现仍待核验；下一轮仍只从八家重点厂商在两个排行榜中的剩余条目选择锚点。

## 2026-09-20：GLM-5 之后选择 DeepSeek V3.2 下一锚点

- 重新抓取 [Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2)：`10.24.27.134:8098` 与 `10.237.126.170:1234` 均返回 HTTP 200；两个快照均为 `3,625,720` bytes，SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`，逐字节一致。页面仍为 `DeepSeek V3.2 (Non-reasoning)`，128K context、Intelligence Index `16.043537719683`、第三方参数约 648B total/37B active。
- DataCurve `/tmp/deepswe-20260920-8098.out` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，仍没有精确 `mini_swe_agent_deepseek_v3_2_*` 行；不迁移 V3.1/V4 的 Pass@1、成本或 Agent steps。
- DeepSeek V3.2 已有研究笔记 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)，其中官方模型卡/技术报告/V3.2-Exp 已核验 DSA 两阶段训练、2,048 KV top-k、scalable RL、GRPO 稳定化、Agent 任务合成/验证和 `thinking with tools`；当前仍为“AA 单榜资料级闭环”，不新增重复正式架构章节。
- 当前顺序：GLM-5 已完成内容专题收口；`deepseek-v3-2` 作为下一锚点，后续只补 PDF 公式/图表视觉复核、完整 DSA kernel/层排布、训练细节、线上 tool acceptance、硬件 profiling 和独立 benchmark。

## 2026-09-20：DeepSeek V3.2 官方实现与 serving 证据补充

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮只在该目录推进并保留既有 dirty worktree 修改。
- 研究笔记新增官方实现复核：V3.2-Exp inference demo 的 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal mask/top-k、prefill MHA、decode MQA、latent KV/position cache 和 FP8 KV cache；本地 `model.py`/`kernel.py` 快照哈希已登记。
- 明确配置边界：固定官方 V3.2 `config.json` 与 V3.2-Exp inference config 都核验为 `q_lora_rank=1536`；实验 config 同时记录 `dim=7168`、61 layers、3 dense layers、256 routed/8 active experts、`index_topk=2048` 和 FP8/`ue8m0`。两份 artifact 仍按 revision、实现路径和证据等级分账，不合并成完整生产配置。
- 核验 [TileLang DeepSeek V3.2 examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)、[DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200)、[FlashMLA PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98) 和 [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)：补齐 radix/histogram top-k、sparse MLA gather、pipelined double buffering、MoE/MQA logits、FP8/BF16 KV cache、`DP=8, EP=8, TP=1` 与 TP fallback 的面试边界。
- vLLM recipe 的 GSM8K 5-shot `0.9591`/20-shot `0.9538` 已标为 V3.2-Exp + vLLM + lm-eval 的 recipe/harness 结果，未写入 DataCurve 或模型能力字段；DataCurve 仍没有精确 V3.2 行。
- 已同步 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 以及论文、题库、练习、术语、项目和知识图谱。
- 当前状态仍为“AA 单榜资料级闭环”：公开实验实现路径已补齐，但最终 production kernel 全覆盖、召回/误差曲线、各 GPU profiling、真实并发 p99、线上 tool-call acceptance rate 和独立 benchmark 仍待核验；不新增 V3.2 重复架构章节。

## 2026-09-20：Gemini 3.5 Flash-Lite 资料级闭环

- 重新确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；本轮继续保留既有 dirty worktree 修改。
- Artificial Analysis 精确条目为 [`gemini-3-5-flash-lite`](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)，快照 `3,845,070` bytes、SHA-256 `7de22f529d3b4e2dd440ef8a8e9447480e18dde08437fc508ece06767571e8ef`；第三方字段为 `releaseDate: 2026-07-21`、`isReasoning=true`、Intelligence Index `22.1685424839812`。DataCurve 快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，无精确 Lite Agent 行，不迁移相邻 Gemini 的 Pass@1、成本或 steps。
- Google API 模型页确认 `gemini-3.5-flash-lite`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/Computer Use Preview/File Search/function calling/Maps/Search grounding/structured output/thinking/URL context；Thinking 文档把默认设为 `On (minimal)`，支持 `minimal/low/medium/high`。
- Video understanding 文档把 Lite 列入 agentic video：static 路径约 1 FPS 固定取帧，agentic 路径按 prompt 动态读取时间轴、transcript、帧和音频，并用 `processing_call`/`processing_result` 暴露处理状态；88% token efficiency 与 7% quality 是文档发布方的总体表述，不是每个请求的保证。
- DeepMind Model Card/PDF 明确 Lite based on Gemini 3.1 Flash-Lite，并将 architecture、training dataset、data processing、hardware 和 software 指向 3.1 Model Card。SWE-Bench Pro `54.2%`、Terminal-Bench 2.1 `54.0%`、OSWorld-Verified `74.0%`、GDM-MRCR v2 128K/1M `72.2%`/`21.3%` 和 `$0.30/$2.50` 价格均按 Google 发布方评测/产品设置记录。
- 正确的 `whats-new-gemini-3.5` 页面是 Gemini 3.5 Flash 页面，不能把 Flash 的默认 `medium`、GA 或 thought preservation 回写到 Lite。当前状态为“资料级闭环（AA 单榜）”，不新增独立架构章节；完整证据和待核验项见 [`gemini-3.5-flash-lite-source-notes.md`](research/model-update-2026-09/gemini-3.5-flash-lite-source-notes.md)。

## 2026-09-20：Kimi K2.6 AA 单榜资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留已有 dirty worktree 修改。
- Artificial Analysis 精确条目为 [`kimi-k2-6`](https://artificialanalysis.ai/models/kimi-k2-6)，三个代理的详情快照逐字节一致：`3,938,140` bytes，SHA-256 `f24ad7d9cd3ddbb253470750dcf9f47aea3fc0d675c9f8f7eb8bdbcd50dcfe18`。页面的 deprecated/K3 迁移提示只作为第三方目录状态。
- DataCurve 快照为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；当前没有精确 `mini_swe_agent_kimi_k2_6_*` 行，因此不迁移 K2.7 Code/K3 的 Pass@1、成本、输出 token 或 Agent steps。
- 已新增 [`kimi-k2.6-source-notes.md`](research/model-update-2026-09/kimi-k2.6-source-notes.md) 和第二十一册第 90 章 [`Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收`](book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)，同步目录、架构时间线、来源索引、模型盘点和榜单解释。
- 已核验官方 1T/32B MoE、MLA、MoonViT、256K、native INT4、K2.5 架构路线复用、vLLM/SGLang/KTransformers 部署、thinking/preserve_thinking/tool protocol、Agent Swarm 和 Kimi Vendor Verifier。300 sub-agents 属于 harness，不是内部 expert；KVV 属于部署/实现验收，不是模型分数。
- 当前状态：**AA 单榜资料级闭环**。完整训练/后训练 recipe、层排布、生产 kernel、INT4 误差与硬件 profiling、Agent Swarm coordinator、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验；下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：GPT-5.4 mini/nano 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作；Artificial Analysis 精确条目为 [`gpt-5-4-mini`](https://artificialanalysis.ai/models/gpt-5-4-mini) 与 [`gpt-5-4-nano`](https://artificialanalysis.ai/models/gpt-5-4-nano)。mini 详情快照为 `3,857,350` bytes、SHA-256 `a01a6da4a38077bc6309333d509bf303f8b1beb36729fd94f79fe7d3a8ffe0ee`；nano 为 `3,859,704` bytes、SHA-256 `2e6ddfefc9a3483437f0fe2648ce23ff42fe4f8614cec0c365dfc69777a166b4`。两页 release date 均为 `2026-03-17`，第三方 Intelligence Index 分别为 `24.0682169231964` 与 `20.7197352504557`，当前目录状态均提示迁移到 GPT-5.6 对应配置；这些仍是 AA 第三方字段。
- DataCurve 当前仅有 `mini_swe_agent_gpt_5_4_xhigh`，它绑定 GPT-5.4 base + `mini-swe-agent` + 工具/环境/verifier；没有 GPT-5.4 mini 或 nano 的精确模型 ID 行。因此不把 GPT-5.4 base 的 Pass@1、成本、输出 token 或 Agent steps 迁移给两个 sibling。
- OpenAI 官方模型页已确认 snapshot `gpt-5.4-mini-2026-03-17` 与 `gpt-5.4-nano-2026-03-17`，两者均为 text/image → text、400K context、272K maximum input、128K maximum output，并支持 `none/low/medium/high/xhigh` reasoning effort。mini 的官方定位偏高吞吐 coding、computer use 和 Agent workflow；nano 偏 classification、extraction、ranking 和窄任务 sub-agent。
- 能力矩阵按精确 model page 记录：mini 页面列出 `tool_search` 与 `computer_use`，nano 当前页面未列出这两项，不能从“GPT-5.4 家族”自动继承；两者的 function calling、web search、file search、code interpreter、hosted shell、apply patch、skills 和 MCP 仍需按 endpoint/宿主权限做 capability probe。
- 本轮新增 [`gpt-5.4-mini-nano-source-notes.md`](research/model-update-2026-09/gpt-5.4-mini-nano-source-notes.md)，并同步模型清单、榜单解释、来源索引、[`gpt-5.4-source-notes.md`](research/model-update-2026-09/gpt-5.4-source-notes.md)、[`plan_v2.md`](plan_v2.md) 和现有书系配套。mini/nano 复用 GPT-5.4 的 Agent serving、reasoning、routing 和评测章节，不新增重复 Transformer 章节。
- 当前状态：**资料级闭环（AA 单榜关联档位）**。参数、层/专家/注意力结构、训练与后训练 recipe、system card、公开权重、独立技术报告、mini/nano 精确 DataCurve Agent 评测、生产 kernel、目标硬件 profiling 和线上 acceptance rate 仍待核验；下一轮继续只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：GPT-5.5 Instant (June 2026) 下一锚点与官方身份复核

- 本轮从 Artificial Analysis 的精确条目选择 `GPT-5.5 Instant (June 2026)`，同时复核 May 历史 revision；没有从 OpenAI 官方目录另发现模型。June 详情快照为 `3,847,333` bytes、SHA-256 `9c6fa94b9c42a89eb7c8f59bad08a30c168bf0860971fbb666c4be47e6a05599`，release date `2026-06-25`，AA Intelligence Index `26.0135173401368`，约 `130.9925 output tokens/s`，400K context，页面未标记 deprecated。May 详情为 `3,548,855` bytes、SHA-256 `ed410b6cecbd8eb1dcaf65715547771bcb7435e444f6c5ba7ce815e16614e51e`，release date `2026-05-05`，AA Index `22.6863693000789`，已 deprecated。
- 两个 AA 页面都显示 text/image → text、reasoning、知识截止 2025-08-31 和 `$5/M` input、`$30/M` output；这些是 Artificial Analysis 的第三方目录/测量字段或其 API 价格引用，不是 OpenAI 内部架构、训练机制或正式 model ID 证明。
- 通过 `7890` 获取的 OpenAI 官方页面确认的是 `gpt-5.5` 与 snapshot `gpt-5.5-2026-04-23`；访问精确 `https://developers.openai.com/api/docs/models/gpt-5.5-instant.md` 返回 HTTP 404（9 bytes）。因此不能把 Instant June 自动改写成 `gpt-5.5`，也不能把 GPT-5.5 base 的 1.05M context、tool catalog 或 DeepSWE 67% 结果迁移给 Instant。
- DataCurve 当前快照没有 `gpt_5_5_instant` 精确 `mini_swe_agent` 行；GPT-5.5 base 的 xhigh 结果仍绑定 base model、harness、工具、环境和 verifier。当前不新增 Instant 专属 Transformer 章节，研究底稿见 [`gpt-5.5-source-notes.md`](research/model-update-2026-09/gpt-5.5-source-notes.md) 第 9 节。
- 当前状态：**榜单级关联配置，官方身份待确认**。下一步优先查找 OpenAI 是否出现与 June revision 对应的官方发布、API alias、model card、system card 或技术说明；若仍无专属资料，则保留负面证据并切换到下一个重点厂商锚点。

## 2026-09-20：GLM-5.1 资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。按照两榜唯一发现规则，选择 Artificial Analysis 的 [`GLM-5.1`](https://artificialanalysis.ai/models/glm-5-1)，不从 Z.ai 官方目录或论文搜索另发现模型。
- 三条代理重新抓取 Artificial Analysis GLM-5.1 详情页，均返回 HTTP 200 且逐字节一致：`3,935,095` bytes，SHA-256 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`。页面标题为 `GLM-5.1 (Reasoning)`，release 字段为 2026 年 4 月；第三方字段约为 744B total/40B active、200K context、Intelligence Index `26.0585912980095`、约 39.9 tokens/s。上述是 AA 配置/provider 字段，不是官方参数或裸模型能力。
- DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，当前没有精确 `mini_swe_agent_glm_5_1_*` 行；不迁移 GLM-5/5.2/5.3/5.3-Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 7890 成功抓取 Z.ai GLM-5.1 Markdown 文档（`19,347` bytes，SHA-256 `69958d7d95452d3853903524612a0c6a83f368c79417a6889b9c531f45ab3d01`）、Hugging Face README（`10,751` bytes，SHA-256 `2cb148b732f596716f8c9366edf072e35746de6d06fe69c92301ddb492557077`）和 config（`1,379` bytes，SHA-256 `726e45e28d1be1636a5834048a4101a2cfe384819ad2ebe08d41f6fe04656526`）。核验 200K/128K、`glm-5.1` API ID、MIT、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048` 和 202752 positions。
- 补齐官方 [Deep Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md)、[Pricing](https://docs.z.ai/guides/overview/pricing.md) 和 [Release Notes](https://docs.z.ai/release-notes/new-released.md)。重点技术线是 long-horizon 约 8 小时 Agent、实验—分析—优化闭环、multi-turn SFT/RL/process-quality evaluation framework、thinking/tool/cache 协议；Z.ai 当前价格表为 GLM-5.1 input `$1.40`、cached input `$0.26`、output `$4.40`/每百万 token。
- Z.ai 自报 SWE-Bench Pro `58.4`、Linux desktop 655 次迭代/6.9× 向量数据库吞吐、KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 均保留为发布方结果，不能与 AA/DataCurve 拼接。`thinking.type=enabled/disabled` 是请求级模式；当前文档将 `reasoning_effort` 支持列为 GLM-5.2 及以上，不能迁移给 GLM-5.1。
- 证据边界已修正：2026-09-20 的一次访问曾使 `https://z.ai/blog/glm-5.1` 返回 HTTP 404；2026-09-21 已通过三条代理取得 HTTP 200 的官方博客壳和正文 JS 资源，因此旧 404 只代表当时的线路/页面状态，不能继续写成当前页面不可得。模型卡链接 GLM-5 技术报告；arXiv `all:"GLM-5.1"` 精确检索没有 GLM-5.1 专属技术报告。1234 对 HF 的 CONNECT 超时只记为线路失败，不解释为页面不存在。
- 新增 [`glm-5.1-source-notes.md`](research/model-update-2026-09/glm-5.1-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md) 和 [`plan_v2.md`](plan_v2.md)。不新增重复 Transformer 正式章节；内容复用 GLM-5 DSA/MoE、Agentic Engineering、reasoning、工具协议和 serving 主线。
- 当前状态：**AA 单榜资料级闭环**。完整参数独立披露、DSA indexer loss/recall、完整训练/后训练 recipe、过程质量 verifier、生产 kernel、硬件 profiling、8 小时 harness、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验；下一锚点仍只从两个排行榜的八家重点厂商条目选择。

## 2026-09-20：Grok 4.20 0309 v2 AA 单榜资料级闭环

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。按照“两榜唯一发现”规则，从 Artificial Analysis 的精确 [`grok-4-20`](https://artificialanalysis.ai/models/grok-4-20) 条目选择 `Grok 4.20 0309 v2 (Reasoning)`，没有从 xAI 官方目录另发现模型。
- 通过 `10.24.27.134:7890`、`10.24.27.134:8098` 和 `10.237.126.170:1234` 重新抓取 Artificial Analysis，均 HTTP 200、逐字节一致：513,564 bytes，SHA-256 `2fc88248faf152f46f659310f300f2fe3e5b84e419babb9c8c7dc3752452cdd8`。AA 字段为 `releaseDate: 2026-04-07`、estimated Intelligence Index `25.6550155187053`、2M context、约 `97.0079` output tokens/s、TTFT `22.7098s`、input/output `$1.25/$2.50`；当前目录还标记 deprecated -> `grok-4-3`，只作为第三方状态记录。
- 三条代理重新获取 DataCurve，均 HTTP 200、逐字节一致；当前响应 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面有 Grok 4.5 high 与 Grok 4.6 low/medium/high/xhigh，但没有精确 `mini_swe_agent_grok_4_20_*` 行；因此不迁移相邻版本的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 7890 获取 xAI [Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20)及 Markdown、模型注册表、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)、[Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)、Tools、Prompt Caching 和 [Release Notes](https://docs.x.ai/developers/release-notes.md)。官方主 ID 为 `grok-4.20-0309-reasoning`，并区分 non-reasoning 与 `grok-4.20-multi-agent-0309`；官方模型页写 1M maximum prompt/context、text/image -> text、function calling、structured output、reasoning、Batch、200K long-context price threshold、37 RPS/10M TPM。
- 官方新技术主线是 Realtime Multi-agent Research beta：多个专门 Agent 并行搜索、分析、交叉核验，由 leader agent 汇总；`agent_count=4/16` 或 Responses `reasoning.effort=low/medium` vs `high/xhigh` 控制协作规模，而不是普通模型的思考深度或 MoE expert 数量。内置 web/X/code/collections 工具由 xAI server 执行，子 Agent 中间状态可通过 `use_encrypted_content` 以 opaque encrypted state 保留。
- 已提取面试主线：Responses compaction 返回必须原样回放的单个 opaque item；自动 prompt caching 与 `x-grok-conv-id`/`prompt_cache_key` 是 provider 层复用；function calling 默认 parallel、streaming 时完整 call 在单 chunk 返回；Web/X Search 的 allowlist/filter/citation；Code Execution 的 Python sandbox；Remote MCP 的 Streaming HTTP/SSE、`allowed_tools` 最小权限和 tool schema/context 成本；混合 server/client tools 中 client call 会暂停请求并重置后续 request 的 `max_turns`。
- 证据差异已单独登记：AA 写 2M context，xAI 官方服务字段写 1M；普通 4.20 的完整 reasoning effort 档位未由专属页明确列出，不迁移 Grok 4.6 的 effort 表。xAI Release Notes 的 March 2026 条目确认 Grok 4.20/Multi-agent live，但 `x.ai/news/grok-4-20`、`grok-4.20`、`grok-4-20-multi-agent` 本轮均 HTTP 404。
- arXiv 标题精确检索 [`"Grok 4.20"`](https://arxiv.org/search/?query=%22Grok+4.20%22&searchtype=title)无结果；全文命中的 7 篇论文只是外部评测/使用。没有找到 Grok 4.20 专属技术报告、参数架构、公开权重、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 或独立 DataCurve Agent benchmark。
- 新增 [`grok-4.20-source-notes.md`](research/model-update-2026-09/grok-4.20-source-notes.md)，同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和书系配套。当前状态：**AA 单榜资料级闭环**，不新增重复 Transformer 正式章节；下一锚点仍从两个排行榜的八家重点厂商条目选择。

## 2026-09-20：Grok 4.20 全局资料配套同步完成

- 在已有 Grok 4.20 研究笔记、来源索引、模型清单、榜单解释和书系章节基础上，补齐 [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)、[`EXERCISES.md`](EXERCISES.md)、[`PAPERS.md`](PAPERS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md)、[`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`PROJECTS.md`](PROJECTS.md)。
- 题库、练习和项目覆盖 4/16 agent-count 消融、leader/sub-agent trace、opaque encrypted state、compaction replay、prompt cache、server/client tools、跨请求 `max_turns`、MCP `allowed_tools`、2M/1M context discrepancy 和 DataCurve 精确行缺失；论文索引明确没有 Grok 4.20 专属 arXiv 技术报告。
- 全局资料继续保留证据边界：multi-agent 是 xAI runtime/编排能力，不是已公开的 MoE expert 数量或普通 reasoning 深度；AA 与官方 context 字段按 source/endpoint 分账；不迁移 Grok 4.5/4.6 的 DeepSWE 结果。
- 当前状态仍为 **AA 单榜资料级闭环**；暂无独立 Grok 4.20 架构/训练报告、公开权重、完整 recipe、生产 kernel、硬件 profiling、线上 acceptance 或精确 DataCurve Agent 行。下一轮只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：Gemini 3.8 Flash 当前活动锚点与复验完成

- 本轮确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现规则仍只允许 Artificial Analysis 与 DataCurve DeepSWE 两个排行榜，重点厂商仍为 OpenAI、Anthropic、GLM/Z.ai、Qwen、Kimi/Moonshot、DeepSeek、Gemini 和 Grok。
- 三条代理重新抓取 AA `/zh` 首页和 DataCurve，均 HTTP 200、逐字节一致：AA 首页 `1,777,695` bytes / SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- AA Gemini 3.8 high/medium/low 详情本轮分别为 `517,803` / `511,687` / `509,053` bytes，哈希为 `0618989e412ae947b3a9f499c1cdae16e4ea3f5ee65a25def360249ae33985ef` / `d96907dd254166f038084e7586e30ab869fc8a3225501c37c958650859aec6a8` / `ef35f6f567518bd1b9e9e1bc17e4826ea479b7d1e7868bc35f8daf7c464ee5b3`；当前字段仍为 2026-09-02 release、约 1M context、FAQ 指数约 41。三个 effort 行归并为一个基础模型。
- DataCurve 精确对象为 `gemini-3-8-flash` + `mini-swe-agent` + `high`：`n_runs=4`、`n_attempted=447`、Pass@1 `0.738255033557047`、Pass@4 `0.8584070796460177`、平均成本 `$2.362349413758389`、平均输出 `143242.6644295302` token、平均 `166.3131991051454` steps、median `161`。这些数字绑定 harness、工具、任务环境和 verifier，不外推到 low/medium 或其他 Gemini 版本。
- 通过 `7890` 重新核验 Google 模型页、Thinking、Interactions，响应分别为 `23,603` / `37,414` / `23,899` bytes，哈希已写入研究笔记与来源索引。官方字段确认 `low/medium/high`、`minimal` 不支持、thought summary/signature、Interactions steps、工具组合、Computer Use、1M 输入和 caching；本轮不把这些 API/runtime 能力写成内部架构或训练算法。
- 已补齐 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md) 的 2026-09-20 快照复验，并同步 [`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)、[`EXERCISES.md`](EXERCISES.md)、[`PAPERS.md`](PAPERS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md)、[`GLOSSARY_EN_ZH.md`](GLOSSARY_EN_ZH.md) 和 [`PROJECTS.md`](PROJECTS.md)。
- 书系已将 Gemini 3.8 的案例接入第十六册 `14.28`（thinking/signature）、第十七册 `7.30`（Interactions/Computer Use）、第二十册 `19.31`（harness-aware evaluation）和第二十四册 `32.41`（state/cache/tool serving）；由于没有独立 3.8 架构/训练报告，不新建重复的 Gemini 专属 Transformer 章节。
- 当前状态：**资料级闭环**。尚无独立 Gemini 3.8 架构/参数/训练报告、生产 kernel、目标硬件 profiling、线上 acceptance 或独立复现；下一步做真实 API capability probe、thinking 消融、Interactions replay、工具组合和长上下文/cache 实验，不新增重复正式章节。目标仍在持续推进，不标记为 complete。

## 2026-09-20：GLM-5.3 当前活动锚点与资料级闭环

- 本轮确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商范围不变。
- Artificial Analysis 精确条目为 [`glm-5-3`](https://artificialanalysis.ai/models/glm-5-3) 的 max 配置；三代理详情快照逐字节一致，`3,928,329` bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`。第三方/provider 字段为 release `2026-08-18`、Intelligence Index `44.777392385614`、约 `72.1152 tokens/s`、TTFT 约 `2.99s`、1M context 和 `$1.40/$4.40`。
- DataCurve 当前快照三代理逐字节一致，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。精确 `mini_swe_agent_glm_5_3_max` 行为 451 次尝试、311 次通过、Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均成本约 `$3.9934`、平均输出约 `80.4K` token、平均 124.47 steps；结果绑定 harness、工具、环境、预算和 verifier。
- 已复验 Z.ai GLM-5.3 文档、迁移指南、Thinking/Function Calling/Tool Streaming/Context Caching/Structured Output、release notes、官方博客资源、Hugging Face README/config、`slime` 和 IndexCache/SAO 关联论文。新增面试主线包括：同 GLM-5.2 基座的后训练归因、可执行长周期环境、judge 与无 reference verifier、oracle/no-op/unsolved-state、reward shortcut audit、train/rollout/data-buffer 单 dataflow、logprob 一致性、`thinking.type=enabled`、`low/high/max` effort 和 `tool_stream` delta 重组。
- 证据边界：SAO 直接论文对象为 GLM-5.2，不能写成 GLM-5.3 独有算法；IndexCache 是 DSA serving 关联证据；Z.ai benchmark、`1e-7` logprob 和 `>2.3x` throughput 是发布方结果/描述；HF config 字段不等于完整参数、训练 recipe 或生产 kernel。
- 已完成研究笔记、来源索引、模型清单、榜单解释、第十六册第 20 章和全局题库/练习/术语/项目/论文/知识图谱同步。当前状态：**双榜资料级闭环**；完整 recipe、5.3 专属 SAO/compaction 定义、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance 仍待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：DeepSeek V3.2 当前活动锚点与实现证据补强

- 本轮选择 Artificial Analysis 精确 [`deepseek-v3-2`](https://artificialanalysis.ai/models/deepseek-v3-2) 条目；DataCurve 当前没有精确 `mini_swe_agent_deepseek_v3_2_*` 行，不迁移 V3.1/V4 的 Pass@1、成本、输出 token 或 Agent steps。AA 详情快照为 `3,625,720` bytes、SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`；`Non-reasoning`、128K、约 648B/37B 和指数 `16.043537719683` 是第三方目录字段。
- 已复核并入库 V3.2 模型卡、官方技术报告、V3.2-Exp inference、TileLang 示例、DeepGEMM/FlashMLA PR 和 vLLM recipe。正式章节补充 DSA 的 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal/top-k、sparse MLA gather、prefill MHA/decode MQA、latent/positional cache、FP8 KV cache 与 dense fallback。
- 本轮明确最终 V3.2 与实验 artifact 分层：固定官方 `config.json` 与 V3.2-Exp inference 都是 `q_lora_rank=1536`；前版把最终字段读成不同数值的说法已纠正。实验代码、kernel 和 vLLM recipe 证明公开实现路径，不等于最终生产性能或裸模型能力。`DP=8, EP=8, TP=1`、TP fallback、GSM8K recipe 和 kernel 局部结果必须绑定版本、硬件和 harness。
- `thinking with tools` 的 reasoning/tool call/tool result/final 边界已与 V3.2-Speciale 的“不支持 tool calling”分开记录；宿主仍需 schema、权限、确认、超时、重试、回灌和 artifact verifier。研究笔记、全局配套和第 21 册第 19 章已同步。
- 当前状态：**AA 单榜资料级闭环**。完整层排布、最终 production kernel、召回曲线、完整训练/RL recipe、线上 tool acceptance、硬件 profiling、PDF 图表/公式视觉复核和独立 benchmark 仍待核验；下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：Kimi K2.6 Kimi Code harness 补证与 K2.8 边界

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作；这是既有 Kimi K2.6 补证，不新增模型锚点，当前活动锚点仍是 `DeepSeek V3.2`。只把官方 Kimi Code 文档作为已发现 K2.6 的周边技术资料，不把官方产品目录当作新模型发现入口。
- [Kimi Code 模型配置](https://www.kimi.com/code/docs/kimi-code/models.html) 当前列出 K3、K2.8 Preview、K2.7 Code HighSpeed 三类、4 个 model ID：`k3`、`k3-256k`、`kimi-for-coding`、`kimi-for-coding-highspeed`。K2.8 Preview 虽在官方产品文档出现，但 2026-09-20 AA/DataCurve 快照没有精确 K2.8 条目，因此不进入候选清单、不建立 K2.8 研究笔记。
- [Kimi Code 内置工具/AgentSwarm](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html) 与 [Agent/subagent](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html) 补齐当前 harness 证据：最多 128 个 subagent、默认 2 小时超时、`resume_agent_ids`、模型池、聚合报告、AgentSwarm 必须是该响应中的唯一工具调用、默认从 5 个并发开始且每 700ms 增加 1 个、`KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY` 限制并发；工具和 subagent 列表在实际派发前再次校验，权限规则独立于可见工具列表。
- [Kimi Code 会话与上下文](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html) 补齐 `state.json`、`wire.jsonl` 事件流、`--continue`、`--session`、`/compact`、`/fork`、导出和工具 schema/MCP 清单回放。它支持“模型 + harness + 可恢复状态 + verifier”的面试拆解，但没有公开 K2.6 专属 coordinator 内部实现。
- 证据分账：K2.6 官方博客的 300 sub-agents/4,000 coordinated steps 是发布方案例；Kimi Code 当前 AgentSwarm 的 128 是 CLI 工具约束。二者不合并、不取最大值写成 K2.6 固有上限，也不把 subagent 数量写成 MoE expert 数量。完整细节与快照哈希已写入 [`kimi-k2.6-source-notes.md`](research/model-update-2026-09/kimi-k2.6-source-notes.md) 和第二十一册第 90 章。
- 当前状态仍为：`Kimi K2.6` **AA 单榜资料级闭环**；K2.6 专属训练/后训练 recipe、production kernel、INT4 profiling、Agent Swarm 内部 coordinator、线上 acceptance、精确 DataCurve 行和独立复现待核验。下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：DeepSeek V3.2 官方 config 纠错与 DSML encoding 补证

- 本轮定位并修正 V3.2 专属证据错误：官方固定 revision `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` 的 `config.json`（1,552 bytes，SHA-256 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`）确认最终 `q_lora_rank=1536`，与 V3.2-Exp inference config 同值；此前把最终字段读成不同数值的记录已同步纠正。
- 同步写入官方配置的 61 layers、前三层 dense、256 routed/top-8/1 shared、64 index heads、`index_head_dim=128`、`index_topk=2048`、163840 positions、YaRN factor 40、BF16 和 FP8 `e4m3`/`ue8m0`、128×128 block 字段。以上是固定 config 证据，不等于完整训练 recipe 或 production kernel。
- 固定 `encoding/encoding_dsv32.py` 快照为 14,317 bytes，SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`；新增 DSML function-call/result、role、`<think>`/`reasoning_content` 和 JSON 参数分支，并在第 19 章新增 19.33，明确 parser、schema、权限、执行器和 verifier 的责任边界。
- 已更新 [`deepseek-v3.2-source-notes.md`](research/model-update-2026-09/deepseek-v3.2-source-notes.md)、第 21 册第 19 章、[`plan_v2.md`](plan_v2.md)、模型清单、榜单解释、来源索引、[`PAPERS.md`](PAPERS.md)、[`PROJECTS.md`](PROJECTS.md)、[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md) 和 [`INTERVIEW_BANK.md`](INTERVIEW_BANK.md)。当前目标仍 active；PDF 视觉复核、完整 production kernel、召回曲线、硬件 profiling、线上 tool acceptance 和完整训练/RL recipe 仍待核验。

## 2026-09-20：两榜当前时点联网复验

- 三条代理重新抓取 Artificial Analysis `/zh`：均 HTTP 200，1,777,695 bytes，SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`，逐字节一致。
- 三条代理重新抓取 DataCurve DeepSWE：均 HTTP 200，268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致。
- 当前 AA 首页前列的 Fable 5.1、GPT-6 Astra、Claude Opus 5、GLM-5.3、Grok 4.6、Kimi K3、Gemini 3.8 Flash、DeepSeek V4.1 Flash 等重点条目，以及 DataCurve 的重点精确行，均已在现有模型清单/研究笔记中有记录；本轮没有新增模型，也没有从排行榜之外引入候选。
- （历史复验记录）当时活动锚点仍为 DeepSeek V3.2；V3.2 的官方 config 纠错和 DSML encoding 补证已完成。此处只记录当时的研究入口；后续已依次切换到 DeepSeek V4 Pro 0813 和 DeepSeek V4.1-Flash，当前活动锚点以文件顶部和最新记录为准。

## 2026-09-20：Kimi K3 权重 revision、FlashKDA 与 vLLM serving 补证

- 本轮沿两个排行榜中已存在的 Kimi K3 条目推进，没有从 HF、GitHub 或 vLLM 页面另发现模型。工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。
- 官方 HF API 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`，metadata 响应 SHA-256 为 `24b2606b8f2604828ce078a93f76209b1ce97e366e4cfa0ea20176975d9ac07e`；固定 `config.json` 为 7,006 bytes，SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。配置确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 full-attention/Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 experts/top-16/2 shared、1,048,576 context；完整 safetensors 未下载。
- FlashKDA README 为 4,431 bytes、SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；master 最新可见 commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934` 将 fp16 Neumann inverse 换为 `8x8 fp32 forward substitution + 16x16 bf16 merge`。README 要求 SM90+/CUDA 12.9+/PyTorch 2.4+，支持 `chunk_kda`、recurrent state 和变长 batch；H20/GB200 固定长度 benchmark 相对 baseline 为 `1.85x`/`2.31x`，属于仓库发布结果，不是本地或端到端复现。
- vLLM K3 recipe 页面 SHA-256 `98e380fbbc8b85d236b60b9a4570ace01359745451cb436532f077714899a0e6`，YAML 为 19,414 bytes、SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`，更新时间 2026-09-10，要求 vLLM 0.29.0/K3-enabled nightly、CUDA 13/cu130 与 r580+ driver。hybrid KV manager 同时管理 MLA attention cache 与 KDA recurrent state；Blackwell 路径包含 FP8 KV、TOKENSPEED MLA、prefix caching 和 `--prefix-match-unit 128`，并按 DCP/TP/TEP/DEP/PP、RDMA/NVLink 记录部署拓扑。
- recipe 明确提示 K3 偶发生成 parser 不期望的 tool-call 格式；面试和教学实现已补充 schema validation、retry/idempotency、权限检查和 verifier 的分层。vLLM `0.29.0` stable release/source implementation entry 已由 PyPI metadata 与 v0.29.0 tag 固定；完整训练/optimizer recipe、目标硬件 profiling、独立 benchmark、完整权重加载和线上 tool-call acceptance 仍待核验。
- 已完成同步：`kimi-k3-source-notes.md`、`inventory-interpretation.md`、`source-index.md`、`model-inventory.md`、`plan_v2.md`、本文件、`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md`。下一步是在验证本轮链接/Markdown/代码块后，再从两张排行榜选择下一个未闭环重点锚点；目标保持 active。

## 2026-09-20：DeepSeek V4 Pro reference implementation 与 serving manifest 补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径与 `/home/zzc/llm-from-zero-to-interview` 等同；没有新建模型候选，仍沿两个排行榜已经发现的 `deepseek-v4-pro` 推进。
- 固定官方 HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`，确认 64 个 safetensors 分片和 `1,598,839,674,782` bytes metadata storage；完整权重未下载。`README.md`、`config.json`、encoding、inference config/model/kernel 和 generation config 的文件级哈希已写入研究笔记。
- 从官方 `inference/model.py` 补齐 gated KV compressor、ratio-4 overlap state、learned causal/top-k indexer、MLA + compressed sparse attention + 128-token window、前三层 hash routing、后续 `sqrtsoftplus` routing、top-6 + shared expert、MTP 和 Hyper-Connections/Sinkhorn 证据；从 `kernel.py` 补齐 block FP8/FP4 quantization、GEMM、online softmax 和 HC Sinkhorn。
- 从官方 `encoding/encoding_dsv4.py` 补齐 DSML tool-call、tool role -> `<tool_result>`、string/JSON parameter 分支、`<think>`/reasoning 保留规则和严格 malformed-output 失败边界。已做 AST 与无 CUDA encode/parse round trip；未运行 TileLang/CUDA 或完整权重推理。
- 已更新 [`deepseek-v4-source-notes.md`](research/model-update-2026-09/deepseek-v4-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md)、第二十一册第 77 章、第二十四册第 32.42 节及全局论文/项目/知识图谱/题库/练习/术语。
- 该段记录当时的 `DeepSeek V4 Pro 0813` 阶段，状态为**双榜内容专题闭环 + reference implementation 证据补强**。当时待核验项包括完整权重加载、固定 kernel/convert commit、目标 GPU 的 FP4/FP8 profiling、compression/index/evidence recall、线上 tool acceptance、完整训练 recipe 和 API snapshot 差异；当前活动锚点已在后续记录切换为 `DeepSeek V4.1-Flash`，目标保持 active。

## 2026-09-20：DeepSeek V4.1-Flash 固定 revision 实现与 serving 边界补证

- 已确认当前工作目录 `/data/zzc/llm-from-zero-to-interview` 与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。本轮仍遵守“模型只从 Artificial Analysis 与 DataCurve DeepSWE 发现”的规则，选择 AA 已有精确 slug `deepseek-v4-1-flash`，没有从 HF 仓库另发现模型。
- 两榜当前时点复验快照：Artificial Analysis `/zh` 三代理均 HTTP 200、`1,777,695` bytes、SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve 三代理均 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，结果逐字节一致。DataCurve 页面没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 V4 Pro/V4 Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 通过 `10.24.27.134:7890` 读取固定 HF revision `dba1be0a40aa45a94ad051997016db3960a90277` 的 API 与源码；API response 为 `6,687` bytes，SHA-256 `fb3aefa7794da9101d0253ccc4e6e36fbace841778f6857b1370cb2880235d63`，列出 48 个 safetensors 分片。完整权重没有下载；HF inventory 的 dtype bucket 统计不替代模型卡 `552B backbone` 或本地参数加载。
- 新增实现证据：固定 revision 的 `inference/README.md`、`inference/config.json`、`model.py`、TileLang `kernel.py`、`convert.py`、Engram/vision、`encoding.py`、`evaluation/README.md` 和 `dsh-minimal.patch` 的文件大小/SHA-256 已写入研究笔记、来源索引和模型盘点。`model.py` 能看到 SWA ring、compressed KV overlap state、两级 candidate/index top-k、sparse attention、MoE、Engram、mHC/Sinkhorn 与 DSpark block；`kernel.py` 能看到 FP4/FP8 quantization/GEMM、sparse online softmax 和 Sinkhorn 路径。
- 关键边界已经写入第二十一册第 81 章与第二十四册第 32.43 节：官方 README 把 inference tree 定义为 readable reference implementation；`model.py` 虽有 `forward_spec`，但 `generate.py` 仍调用普通 `model.forward`，没有完整 draft/verify/rollback scheduler。因此当前不宣称 DSpark 已完成 speculative serving 或吞吐复现；EPD、host-DRAM SWA pool、global KV、alias/served-model 和 `not_applicable` DataCurve manifest 分开记录。
- 本地校验：下载的 Python 源码 `py_compile` 通过；encoding smoke test 覆盖 DSML 前导空格、`reasoning_effort=max`、中途 system message 和 thinking parse，结果通过。没有执行 CUDA、TileLang、完整权重、目标硬件 profiling、线上 API 或独立 benchmark。
- 已同步：[`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md)、第二十一册第 81 章、第二十四册第 32.43 节、`PAPERS.md`、`PROJECTS.md`、`KNOWLEDGE_GRAPH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md`。
- 当前状态：**内容专题闭环 + reference implementation 证据补强（AA 单榜）**。仍待核验完整权重加载、生产 kernel 覆盖、候选/Top-K recall、FP4 误差、DSpark 接受长度与 rollback、EPD 真实调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark；goal 保持 active，下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 2026-09-20：两榜重点厂商前列审计

- 复核 2026-09-20 的 Artificial Analysis 首页快照（`1,777,695` bytes，SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`）和 DataCurve DeepSWE 快照（`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`）。按当前重点厂商过滤，AA 前列依次覆盖 `Claude Fable 5.1`、`GPT-6 Astra`、`Claude Opus 5`、`GLM-5.3`、`Grok 4.6`、`Kimi K3`、`Gemini 3.8 Flash` 和 `DeepSeek V4.1 Flash`；Muse Spark 等非重点厂商不进入本项目更新队列。
- 上述八个重点条目均已有研究笔记、官方来源和书系/配套同步；本轮审计没有发现需要新建研究笔记的重点模型，也没有把 effort、fallback、provider 或关联 artifact 误计为新的基础模型。
- 因此当前活动锚点仍为 `DeepSeek V4.1-Flash`。下一步继续补齐实现级待核验项：完整权重加载、生产 kernel 覆盖、candidate/index Top-K recall、FP4 误差、DSpark draft/verify/rollback 调度、EPD 实际调度、目标硬件 profiling、线上 tool acceptance 和独立 benchmark。goal 保持 active。

## 2026-09-20：DeepSeek V4.1-Flash 召回与 FP4 教学实验

- 新增标准库脚本 [`deepseek_v41_cache_demo.py`](research/model-update-2026-09/code/deepseek_v41_cache_demo.py)，不下载权重、不使用 CUDA/TileLang，只用合成 indexer 分数和 toy KV 向量演示两个待核验指标：candidate-pool recall 与候选池内 conditional Top-K recall，以及分组 E2M1-like FP4 的 MSE/max error。
- 脚本大小 `6,317` bytes，SHA-256 `6c65d44f2f092369ac8b0cdf83a58fe4c2dad38e46698e5d7f1badc209c9517e`；运行和 `py_compile` 均通过，固定输出为 candidate recall `0.600`、conditional Top-K recall `0.667`、端到端 recall `0.400`、toy FP4-like MSE `0.025862`、最大绝对误差 `0.300000`。
- 已将脚本链接和证据边界同步到 DeepSeek V4.1 研究笔记、来源索引、模型盘点、榜单解释和第二十一册第 81 章。该实验只验证指标分解和教学代码，不升级为真实模型召回、FP4 质量、DSpark acceptance 或硬件性能证据；goal 保持 active。

## 2026-09-21：DeepSeek V4.1-Flash `deepseek-recipe` 官方协议实现补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），没有新增模型候选；仍以 Artificial Analysis 已发现的 `deepseek-v4-1-flash` 为锚点。DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移其他 DeepSeek 版本的 Agent 结果。
- 通过现有代理获取 DeepSeek 官方 [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) 的官方 Atom、pinned raw README 和 commit archive；固定 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`，归档 3,996,119 bytes，SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`。GitHub API 因匿名 rate limit 返回 403，但不影响 pinned archive/Atom/raw 文件的版本证据；没有把 403 误判为仓库不存在。
- README 快照为 5,261 bytes、SHA-256 `0cccc69baa118d7689fc3ff2c2e47ab652e05410b777744c43a424f4db5fc0af`；streaming/tokenizer 文档和 Rust 源码进一步核验：`ConversationRequest` 协议规范化、V4.1 `reasoning_effort`/DSML/mid-system 渲染、跨 chunk 的 reasoning/DSML/JSON/stop 状态机、显式 tokenizer bridge、图像 URL/data URL/bytes quota、并发/重试/预处理和 mock server 接线。
- 新增面试知识：协议 adapter、prompt/tokenizer、stream parser、图像安全、inference backend、HTTP transport、tool executor、policy 和 verifier 必须分层；parser 成功不等于工具已经执行、JSON 业务语义有效或 verifier 通过。固定 commit README 明确不负责 inference/transport/tool execution/权限/verifier，并列出 `logprobs`、server-side web search、JSON Schema/strict、`n>1`、Responses storage 和 encrypted thinking 等协议库未支持项。
- 已同步 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md)、第二十一册第 81 章 `81.10.4`/来源列表、第二十四册第 32.44 节，以及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。研究笔记记录了默认图像限制（600/32 MiB/64 MiB/8 concurrent）、5 redirects、10/60 秒超时和默认无 SSRF/private-address filtering 的边界。
- 本轮只完成固定源码归档、哈希和静态阅读；没有下载 V4.1 权重，没有运行真实 inference、CUDA/TileLang、目标硬件 profiling 或线上 API/tool acceptance。当前状态：**内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**；goal 保持 active。下一步仍是完整权重、production kernel、真实 recall/FP4、DSpark draft/verify/rollback、EPD 调度、硬件 profiling、线上 acceptance 和独立 benchmark。

## 2026-09-21：两榜当前时点复验，无新增重点模型，切换 GLM-5.1

- 三条代理重新获取 Artificial Analysis `/zh`，快照为 `1,777,588` bytes，SHA-256 `10630c5152df60351ceff5819c90dfd104f1e9e502959eb36e470abae9ab2c9`。与 9 月 20 日快照相比页面内容/哈希有变化，但按 canonical `models/<slug>` 归一化后，八家重点厂商没有新增或删除模型条目。
- 三条代理重新获取 DataCurve DeepSWE，均为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致；重点 `mini_swe_agent_*` 配置集合与 9 月 20 日一致。`mini_swe_agent_kimi_k2_7_code_default` 是既有 Kimi K2.7 Code 条目，不是新增锚点。
- 本轮没有从排行榜之外引入模型，也没有把 effort/fallback/provider/harness/verifier 误计为基础模型。下一活动锚点切换为已在 Artificial Analysis 候选队列中的 `GLM-5.1`。
- GLM-5.1 当前仍是 **AA 单榜资料级闭环**：研究笔记已覆盖官方模型文档、HF 模型卡/config、Thinking/Function Calling/Cache、release note 和论文负检索；本轮继续补齐 Z.ai 官方博客/发布说明、长周期 Agent 评测 harness、过程质量 verifier、生产 kernel/硬件 profiling 等证据，不迁移相邻 GLM 版本的 DataCurve 结果。

## 2026-09-21：GLM-5.1 官方博客恢复与长周期评测补证

- 之前一次联网记录把 `https://z.ai/blog/glm-5.1` 记为 404；本轮三条代理均返回官方博客壳页面 200，并固定正文 JS 资源：221,954 bytes，SHA-256 `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`。博客壳为 598 bytes，SHA-256 `6fa12ef1d6f8bd1e834b4d8074da0035e36112641ba609d7855d31f92e2727db`；旧负面结论已按时间和线路状态修正。
- 官方博客补出三类长周期反馈实验：VectorDBBench 的 Recall ≥ 95%/QPS 外循环（600+ iterations、6,000+ tool calls、21.5k QPS 发布方结果）；KernelBench Level 3 的 50 problems、独立 Docker/H100、1,200-turn cap、数值正确性和 Claude Opus 4.6/GPT-5.4 双审计器；以及没有单一标量目标的 8 小时 Linux desktop self-review harness。
- 已将新证据同步到研究笔记、来源索引、模型盘点、榜单解释、`PAPERS.md`、`PROJECTS.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`。这些仍是 Z.ai 发布方实验设置，不等于 GLM-5.1 专属技术报告、训练 recipe、独立硬件 profiling 或线上 acceptance；DataCurve 无精确 GLM-5.1 行，继续不迁移相邻 GLM 版本结果。

## 2026-09-21：Qwen3.8 Max (0902) runtime recheck

- 本轮确认工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree 修改。模型发现规则仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，未从 QwenCloud 文档另发现模型。
- 三条代理重新获取两榜：Artificial Analysis `/zh` 均 HTTP 200，`1,777,588` bytes，SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`；DataCurve 均 HTTP 200，`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致。按重点厂商 canonical 条目归一化后没有新增模型。
- Artificial Analysis Qwen3.8 Max 详情当前为 `3,835,865` bytes、SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`；标题为 `Qwen3.8 Max (0902)`，第三方字段约 Intelligence Index `45.4152084980521`、`37.275289705095 tokens/s`、`984K` context 和 `$5.408509428374016`/task。QwenCloud 产品页 `last-modified` 为 `2026-09-21 11:01:05`，alias `qwen3.8-max-2026-09-02`，仍明确是 `qwen3.8-max` 的 upgraded snapshot；动态字段导致三条代理 hash 不同，但页面均为 `98,992` bytes。
- 本轮补齐 QwenCloud 服务契约：Thinking/Function Calling/Context Cache 示例已迁移到 `maas.qwencloudapi.com`；这属于 endpoint/provider adapter 变更，不是模型能力或架构升级。当前价格字段为 input/output/implicit cache `$2/$6/$0.25`，explicit cache creation/read `$2.50/$0.17` 每百万 token。
- 新增 [Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits) 证据：文档 `408,930` bytes、SHA-256 `ba010f2386fbb32c0b69bcbdde64efb520435cefc16475cb030b55253909949e`；限流按 account+model 聚合，workspace 可 override，TPM 按自然月调整，达到保证值后仍可能因余量继续服务；`qwen3.8-max-0902` 保证 TPM 为 `1,500,000 / 1,500,000 / 1,500,000`。这是 hosted quota，不是模型吞吐、GPU capacity 或榜单结果。
- DataCurve 仍没有 `qwen3_8_max_0902` 精确行，继续不迁移泛化 `mini_swe_agent_qwen3_8_max_xhigh` 的 `57.4610%` Pass@1、成本、输出 token 或 Agent steps。当前状态：**资料级闭环（runtime recheck）**；待核验 0902 专属技术报告/权重、完整训练 recipe、生产 kernel、硬件 profiling、线上 tool acceptance、真实 429/retry 行为和独立 benchmark。已同步研究笔记、来源索引、模型清单、榜单解释、`plan_v2.md`、第二十四册 serving 章节和全局配套文件；goal 保持 active。

## 2026-09-21：Claude Opus 4.7 runtime recheck 与活动锚点切换

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同），保留既有 dirty worktree 修改；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE。
- Artificial Analysis 仍确认 `claude-opus-4-7`（adaptive/max）与 `claude-opus-4-7-non-reasoning`（non-reasoning/high）属于同一基础模型的配置条目。`/zh` 三条代理均 HTTP 200，快照为 `1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。当前 max Intelligence Index `40.6897928205908`（estimated）、约 `51.1354` output tokens/s、1M context、`$5/$25` input/output；这些是第三方字段。
- DataCurve 三条代理均 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致；当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，不迁移其他 Claude 版本的 Agent 结果。
- Anthropic 官方模型页快照为 `11,882` bytes、SHA-256 `d7ca2e12d25c77f32dda9e59d2e6f1a201a71cc48fc93316c6d2c105188b759b`，确认 API ID `claude-opus-4-7`、1M context、128K synchronous output、300K Batch beta、adaptive thinking、默认 `high`、更新 tokenizer；官方生命周期为 `Active (legacy)`，建议迁移 Opus 5。AA 详情页的 deprecated 只代表第三方目录状态，不能写成官方 API 已下线。
- 已复核的面试主线为 `effort`/task budget/`max_tokens` 三层预算、server-side countdown 与 compaction continuity、`1.0-1.35x` tokenizer 迁移成本、`2576 px/4784 visual tokens` 高分辨率视觉以及 cyber safeguards 的策略/授权/沙箱边界。相关研究笔记、来源索引、模型清单、榜单解释、第二册 `7.28`、第二十四册 `32.36` 和全局配套已同步。
- 当前状态：**内容专题闭环（runtime recheck）**；没有公开参数、架构、完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance 或精确 DataCurve Agent 评测。goal 保持 active，下一轮仍只从两张排行榜的八家重点厂商条目选择锚点或补充实现级证据。

## 2026-09-21：Claude Fable 5.1 runtime recheck 与活动锚点切换

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同），保留既有 dirty worktree 修改；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，没有从官方目录、论文或博客另发现模型。
- Artificial Analysis Fable 5.1 详情快照为 `3,854,152` bytes、SHA-256 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`；canonical slug `claude-fable-5-1`，主配置标题为 `Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)`，`releaseDate=2026-09-01`、`deprecated=false`，Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。这些是第三方配置/provider 字段。
- Anthropic 官方资源已重新固定：模型页 `14,914` bytes / `f773571dce563d7cb8a501b9ad2eb6531d8164938c5868f1aee8dbda8b462f`，发布页 `449,779` bytes / `70d5aaccdd890496070d810944693b9738e8b8c68a97c4e545ddd0e83f9fcd18`，System Card PDF `16,397,488` bytes / `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39`。DataCurve 仍无精确 `mini_swe_agent_claude_fable_5_1_*` 行，不迁移 Fable 5 的 `316/452` 结果。
- 新增面试知识：forced tool use 不兼容时返回错误；较早模型不能读取 Fable 5.1 thinking blocks；编辑较早历史 turn 会使 thinking blocks 失效。新增/明确能力为 per-message effort（beta）、turn-scoped system messages（beta）、工具调用间 `display: "updates"` 进度事件、cache read 降价和 content provenance。已将这些理解为带版本、状态、成本和溯源账本的运行时协议，不把 thinking block 当普通文本、不把进度事件当工具成功、不把 provenance 当事实正确性证明。
- 已同步 Fable 5.1 研究笔记、来源索引、模型清单、榜单解释、`plan_v2.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md` 和第四册百科既有章节；不新增 Fable 5.1 专属 Transformer 章节，复用 Agent、reasoning、tool protocol、serving 和安全章节。
- 当前状态：**资料级闭环**。参数规模、稠密/MoE 架构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确 DataCurve Agent 结果仍待核验；goal 保持 active，下一轮回到两榜单的八家重点厂商候选队列选择新锚点。

## 2026-09-21：Claude Opus 5 当前时点复验与活动锚点切换

- 本轮没有从排行榜之外引入模型；活动锚点切换到已在 Artificial Analysis 与 DataCurve DeepSWE 出现的 `Claude Opus 5`。当前时点重新尝试 `10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234`，三条代理均无法连接，未产生新的网页快照或指标。
- 研究笔记中的 Artificial Analysis/DataCurve/Anthropic 资料仍是 2026-09-15 的缓存证据，不能冒充本轮联网成功。Opus 5 继续保持资料级闭环，面试补证主线为 `display: "omitted"`、thinking block/signature 原样回放、`thinking disabled` 与 `xhigh/max` capability gate、refusal/fallback、fallback credit、工具/effort 中途变更、512-token cache、web fetch 宿主边界和 subagent/自验证预算。
- 不迁移其他 Claude 版本的 DataCurve 成绩；参数、架构、完整训练/后训练 recipe、独立技术报告、生产 kernel、硬件 profiling、线上 acceptance 和精确独立 Agent 评测仍待核验。goal 保持 active。

## 2026-09-21：Claude Opus 5 联网恢复后的新鲜快照

- 三条代理当前均可用：Artificial Analysis 中文首页 HTTP 200，7890 快照 `1,778,568` bytes / SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；Opus 5 详情三条代理逐字节一致，`3,868,875` bytes / SHA-256 `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`。DataCurve 三条代理逐字节一致，`268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 当前 AA Opus 5 max Intelligence Index 为 `50.7771115797629`，median output speed `60.707552496689 tokens/s`，median TTFC `46.6837206345s`，1M context，约 `$5.8584`/task；这些是第三方字段。DataCurve max 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`、平均成本 `$11.8376`，未迁移其他 Claude 版本结果。
- 两榜按 canonical slug 归一化后没有新增八家重点厂商的模型候选；AA 页面新增/移除的其他 slug 不进入本项目队列。Anthropic 模型页 1234/8098 内容一致，选定快照 `461,560` bytes / SHA-256 `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5`；发布页 8098 快照 `352,773` bytes / SHA-256 `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f`。
- 当前状态仍为**资料级闭环**：新鲜快照只更新榜单/页面字段，不增加参数、架构、训练 recipe 或独立 benchmark 结论；goal 保持 active。

## 2026-09-21：GPT-6 Astra 官方模型指南与 Agent 协议补证

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE；两榜没有新增八家重点厂商 canonical 模型，因此没有新增模型笔记或迁移其他模型的 Agent 结果。
- `10.24.27.134:7890` 成功返回 OpenAI 官方 GPT-6 Astra 模型页、专属模型指南、Agents、Async tool calling、Mid-turn steering、Misalignment monitoring、Fast mode 和开发者博客；模型页 HTML 为 `430,363` bytes、SHA-256 `d2ce52cb3c514d70d124867ff18c1ab14105ae3fc372fead76d4f7532b621864`。`10.24.27.134:8098` 与 `10.237.126.170:1234` 对模型页返回 Vercel `403 Forbidden`，只记录为线路/站点防护，不能解释为资料不存在。
- 专属模型指南新增 `async: true` 的 function/custom tool 调用：模型可在应用运行慢工具时处理独立部分，应用仍负责 job registry、原始 `call_id`、超时、重试、幂等、权限和结果回灌；新增 WebSocket `response.steer`：输入先排队，再生成 continuation，原响应可能以 `incomplete_details.reason=steered` 结束，已发送文本、已启动工具和外部副作用不会自动回滚。
- `misalignment monitoring` 被记录为平台安全控制面：它异步检查高后果请求的推理与动作，可能告警或阻断；阻断时匹配 `misalignment_policy_violation`、停止自动重试、保留 request/response/tool trace 并检查已有副作用。监控可能误报或漏报，不能替代权限、沙箱、审批和 verifier。
- 官方博客把 skills、`AGENTS.md` 和任务提示视为上下文路由层，补充短描述、渐进披露、按需读取、减少冲突 recipe、明确完成条件和持续范围等面试知识。已同步研究笔记、第六册第 18 章、第四册百科、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md` 和 `KNOWLEDGE_GRAPH.md`。
- `phase` 证据边界已修正：Reasoning 文档现行专节以 GPT-5.5/GPT-5.4 为示例，只作为通用 assistant item replay 规则，不标成 GPT-6 专属新能力。GPT-6 Astra 当前仍没有公开参数规模、内部架构、完整训练/后训练 recipe、system card、专属技术报告、生产 kernel、目标硬件 profiling 或独立 benchmark。
- 当前状态：**内容专题闭环（官方运行时与 Agent 协议补证）**。新增 [`gpt6_agent_protocol_demo.py`](research/model-update-2026-09/code/gpt6_agent_protocol_demo.py)，6,621 bytes、SHA-256 `4156ef886bbcfc3a3ac8358d4735c80000da79fc4bfb875d0b65c229952b3afa`，已通过 `py_compile` 和主流程断言，覆盖 async tool 的 `call_id`/重复回传、steering lineage/不可回滚副作用和 skill progressive disclosure；下一步进行异常路径扩展、`git diff --check`、Markdown fence/链接检查和新鲜官方哈希复验；goal 保持 active。

## 2026-09-21：Kimi K3 固定 manifest、混合 cache 与源码边界审计

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同用户指定的 `/home/zzc/llm-from-zero-to-interview`），没有从排行榜之外引入模型。Artificial Analysis 当前快照为 `1,778,568` bytes、SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；DataCurve DeepSWE 当前快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。按 canonical slug 归一化后，八家重点厂商没有新增模型候选；不把 effort、fallback、provider、harness 或 verifier 当成模型，也不迁移其他版本的 Agent 结果。
- 当前活动锚点切换为已在两榜发现的 `Kimi K3`。官方 Hugging Face 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`；配置快照为 `7,006` bytes、SHA-256 `9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213`。本轮重新获取 `model.safetensors.index.json`，文件为 `59,764,096` bytes、SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`。
- 新增 [`kimi_k3_manifest_audit.py`](research/model-update-2026-09/code/kimi_k3_manifest_audit.py)，只读取 JSON，不加载权重；`py_compile` 和主流程断言均通过。审计确认 `497,220` 个 tensor、连续的 `model-00001-of-000096.safetensors` 至 `model-00096-of-000096.safetensors`、93 层、69 个 KDA 层、24 个 full-attention 层、92 个 expert-bearing layer、每层 896 experts，以及 `247,296` 对 `weight_packed`/`weight_scale` 配对；量化字段为 MXFP4、4 bit。
- 证据口径已分开：index 的 `metadata.total_size=1,560,860,324,864` 是 MXFP4 packed 分片文件口径；HF API 的 `safetensors.total=2,779,931,837,184` 是 U8/BF16/F32 参数 dtype 统计，不能相加或互换为一个“模型大小”。完整 safetensors 权重没有下载，manifest 一致性也不等于本地模型加载成功。
- 固定的 `modeling_kimi_linear.py`（SHA-256 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`）确认 `KimiDynamicCache` 分开保存 full-attention 的 `key_cache/value_cache` 与 KDA 的 `conv_states/recurrent_states`；单 token decode 使用 `fused_recurrent_kda`，prefill/chunk 使用 `chunk_kda` 和 `cu_seqlens`。因此 serving manifest 不能只记录一个 KV-cache 长度，cache 恢复必须同时验证 MLA attention state、KDA recurrent state 和短卷积 state。
- 已同步 `kimi-k3-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、第二十一册第 88 章、十七册/二十四册对应章节以及题库、练习、术语、项目、论文和知识图谱。当前状态为 **内容专题闭环 + fixed manifest/runtime source + vLLM 0.29.0 stable release/source evidence**；仍待核验本地 wheel 安装、完整 serving recipe、完整权重加载、目标硬件 profiling、hybrid cache recovery、线上 tool-call acceptance、完整训练/optimizer recipe 和独立 benchmark。完成这些 K3 补证后，再从两个排行榜的剩余重点 canonical 模型选择下一锚点；goal 保持 active。

## 2026-09-21：Kimi K3 vLLM upstream/runtime recheck

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同用户指定的 `/home/zzc/llm-from-zero-to-interview`），只沿 Artificial Analysis 与 DataCurve DeepSWE 中已经发现的 `Kimi K3` 推进，没有从 vLLM registry、API 文档或 FlashKDA 仓库另发现模型；goal 保持 active。
- 三条代理在本轮均可访问外网并已用于联网核验：`10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234`。本节记录的是 2026-09-21 的 vLLM/FlashKDA 快照，不把前一时段失败线路当成网页不存在。
- vLLM stable supported-models 页面 `https://docs.vllm.ai/en/stable/models/supported_models/` 为 `738,169` bytes、SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`，页面更新时间 2026-08-29；列出 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3`。stable K3 API `https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/` 为 `799,430` bytes、SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`，页面更新时间 2026-09-09，暴露 `KimiK3ForConditionalGeneration` 与 `KimiK3MTP`。latest supported-models 另为 `792,208` bytes、SHA-256 `4604b83e2dffa3d6cf154003bf4aa1a29d3b7d7503ed278b556c90f7c29d098f`，不与 stable 混写。
- vLLM main registry `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py` 为 `64,391` bytes、SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，包含 `KimiLinearForCausalLM`、`KimiK3ForConditionalGeneration`、`K3DSparkModel`、`KimiK3MTPModel`；K3 package `__init__.py` 为 `1,444` bytes、SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`，按 `current_platform` 分流 NVIDIA/ROCm，TPU 不主动加载 GPU 实现。本轮只把它们作为 main 源码/硬件隔离入口证据。
- vLLM K3 recipe raw YAML `https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml` 为 `19,414` bytes、SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`，`date_updated=2026-09-10`、`min_vllm_version=0.29.0`，仍标记 `Pre-release`，并要求 K3-enabled nightly/image、CUDA 13/cu130、NVIDIA r580+ driver。recipe 的 hybrid KV manager、TP/TEP/DEP/PP、prefix-match unit 128、DCP 和 tool-call parser 警告已写入研究笔记，但不升级为 stable wheel/生产 SLO。
- FlashKDA README 当前为 `4,431` bytes、SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；Atom 为 `8,454` bytes、SHA-256 `2155ff08883c6240b3fb53920a8ecdabfd79d2045f41db35d15e4b9d699f07c5`；最新可见 commit 仍为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`（2026-09-01）。没有新的 kernel 或 upstream merge 证据。
- 证据边界正式收口：stable docs/API 与 vLLM main registry/package 已有 K3 实现入口；recipe 仍是 pre-release/nightly 部署路径；完整权重加载、stable wheel 在目标硬件运行、NVIDIA/ROCm profiling、hybrid cache recovery、线上 tool-call acceptance 和独立 benchmark 仍未证明。不能把“文档能查到类名”说成“生产 serving 已完成”。
- 已同步 `kimi-k3-source-notes.md`、`source-index.md`、`model-inventory.md`、`plan_v2.md`、第二十一册第 88 章、第二十四册 `32.46`，以及题库、练习、术语、项目、论文和知识图谱。manifest audit、`py_compile`、Markdown fence/链接检查和 `git diff --check` 均已通过；下一步只剩最终 diff 审阅和后续 stable wheel/目标硬件/线上 acceptance 验收。

## 2026-09-21：Kimi K3 vLLM 0.29.0 stable artifact correction

- 本轮沿两个排行榜已发现的 `Kimi K3` 继续核验，没有从 PyPI 或 vLLM 另发现模型。此前把 K3 写成“stable upstream merge 未证明/只有 nightly 入口”过于保守，现根据 PyPI release metadata 和 v0.29.0 tag source 修正证据边界。
- 通过 `10.24.27.134:7890` 成功取得 [PyPI vLLM metadata](https://pypi.org/pypi/vllm/json)：完整 JSON `253,652` bytes、SHA-256 `a232f3e31b111ebfb0d99cbe69b71da682a680ccb4cbf1c57f8cdab3801e5b94`；`info.version=0.29.0`。x86_64 wheel `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，aarch64 wheel `310,033,787` bytes、SHA-256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b`，均于 `2026-09-09` 上传。本轮没有下载或安装 wheel。
- v0.29.0 tag source：`registry.py` `63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`；K3 package init `1,444` bytes / SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`；NVIDIA K3 `model.py` `87,313` bytes / SHA-256 `e74026a83c28cce7ee62889e9ed434d1fff6076dbd263d388aa379f4cf987a4d`。registry 含 K3/MTP/DSpark，package 含 NVIDIA/ROCm/TPU 分支，NVIDIA model 含 K3 KDA/MLA/MoE 实现入口。
- K3 当前正确状态：**vLLM 0.29.0 stable release/source implementation evidence 已证明；目标硬件加载、完整权重、FlashKDA/backend 性能、MLA/KDA hybrid cache recovery、线上 tool-call acceptance 和独立 benchmark 仍未证明**。K3 recipe 的 `Pre-release`/K3-enabled nightly、CUDA 13/cu130 和 r580+ driver 是特定优化部署路径，不是“stable release 不存在”的证据。
- 7890 成功获取 PyPI；直连 DNS 失败，8098/1234 对 PyPI metadata 分别超时并留下截断文件，均只记录为线路状态。一次 source-tar 下载也只得到空/截断文件，未引用其内容或哈希作为证据。
- 已同步研究笔记、来源索引、模型清单、计划和进度；下一步同步教材中“stable artifact vs optimized recipe”的措辞，并继续做目标 runtime/权重加载/双状态恢复门禁。goal 保持 active。

## 2026-09-21：Kimi K3 stable artifact 本机门禁复核

- 复核工作目录仍为 `/data/zzc/llm-from-zero-to-interview`（等同 `/home/zzc/llm-from-zero-to-interview`）。K3 manifest audit 重新运行通过：`497,220` tensors、96 shards、93 layers、69 KDA/24 full-attention、92 expert-bearing layers、896 experts/layer、`weight_packed`/`weight_scale` 各 `247,296` 且成对；`py_compile` 也通过。
- 本机当前没有 `vllm`、`torch` 或 `transformers` Python 包，`nvidia-smi -L` 报告无法与 NVIDIA driver 通信；因此不能在本环境声称完成 stable wheel 导入、K3 权重加载、CUDA kernel、双状态恢复或 profiling。
- 根据 PyPI metadata 解析到的 x86_64 wheel URL 通过 `10.24.27.134:7890` 尝试下载约 55 秒仍无字节返回，已中止且未留下 wheel 文件；不把该失败写成 artifact 不存在。现有 PyPI metadata 与 v0.29.0 tag source 仍足以证明 stable release/source implementation entry，但本机 installation gate 保持未通过。
- 证据口径已统一：K3 recipe 的 `Pre-release`/K3-enabled nightly 是优化部署路径；vLLM `0.29.0` stable release/source 有 K3 实现入口；两者都不能替代目标硬件和生产 serving 验收。下一步若获得可用下载线路、GPU 节点或预装运行环境，再执行 wheel import、最小 config/model construction、完整权重加载、MLA/KDA cache recovery 和 tool-call acceptance；否则继续从两榜剩余重点 canonical 模型选择下一锚点。

## 2026-09-21：Kimi K3 stable artifact 本机门禁复核

- 复核工作目录仍为 `/data/zzc/llm-from-zero-to-interview`（等同 `/home/zzc/llm-from-zero-to-interview`）。K3 manifest audit 重新运行通过：`497,220` tensors、96 shards、93 layers、69 KDA/24 full-attention、92 expert-bearing layers、896 experts/layer、`weight_packed`/`weight_scale` 各 `247,296` 且成对；`py_compile` 也通过。
- 本机当前没有 `vllm`、`torch` 或 `transformers` Python 包，`nvidia-smi -L` 报告无法与 NVIDIA driver 通信；因此不能在本环境声称完成 stable wheel 导入、K3 权重加载、CUDA kernel、双状态恢复或 profiling。
- 根据 PyPI metadata 解析到的 x86_64 wheel URL 通过 `10.24.27.134:7890` 尝试下载约 55 秒仍无字节返回，已中止且未留下 wheel 文件；不把该失败写成 artifact 不存在。现有 PyPI metadata 与 v0.29.0 tag source 仍足以证明 stable release/source implementation entry，但本机 installation gate 保持未通过。
- 证据口径已统一：K3 recipe 的 `Pre-release`/K3-enabled nightly 是优化部署路径；vLLM `0.29.0` stable release/source 有 K3 实现入口；两者都不能替代目标硬件和生产 serving 验收。下一步若获得可用下载线路、GPU 节点或预装运行环境，再执行 wheel import、最小 config/model construction、完整权重加载、MLA/KDA cache recovery 和 tool-call acceptance；否则继续从两榜剩余重点 canonical 模型选择下一锚点。

## 2026-09-21：Gemini 3.8 Flash Model Card 与长周期 Agent 复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作（等同 `/home/zzc/llm-from-zero-to-interview`），模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE；没有从 Gemini 3.8 Flash Cyber 或 Google 官方目录另发现模型。AA `/zh` 快照为 `1,776,713` bytes / SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`，Gemini 3.8 high 详情为 `3,862,728` bytes / SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`；DataCurve 为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。按 canonical slug 归一化后八家重点厂商没有新增模型。
- DeepMind Model Card 本轮为 `156,591` bytes / SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`，明确 3.8 基于 3.7，架构、训练数据、数据处理、软硬件和评测方法均指向 3.7 Model Card。官方博客本轮为 `407,603` bytes / SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`，公开描述 shared foundational intelligence、long-running agentic loops、递归评估/改进、额外 reasoning steps 和迭代工具调用；这些仍是发布方系统描述，不是已确认的 RL、verifier 或网络结构。
- DataCurve high 的 Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36` 和平均 `166.31` steps 绑定 `mini-swe-agent`、工具、环境和 verifier，不能迁移到 low/medium 或相邻 Gemini。Google AI Developers 本轮经 1234 返回 503，7890/8098 超时，旧快照不冒充新鲜响应。
- 当前状态保持**资料级闭环**：研究笔记、来源索引、模型清单、榜单解释、面试题库、术语、论文、练习、项目和知识图谱均已同步；不新增 3.8 专属 Transformer 章节。`git diff --check`、研究代码 `py_compile`、K3 manifest audit、GPT-6/DeepSeek/KDA 教学脚本运行和 Markdown 围栏检查均通过；下一步只补真实 API capability probe、Interactions replay、thinking 消融及工具/长上下文成本实验，goal 保持 active。

## 2026-09-21：Grok 4.6 当前时点排行榜复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE。本轮没有从 xAI 官方目录、论文或其他网站新增模型，活动锚点切换为已有两榜条目 `Grok 4.6`。
- Artificial Analysis 当前详情页 `/tmp/grok46-aa-20260921.out` HTTP 200，文件为 `3,859,075` bytes，SHA-256 `a23b19aceae3fee2b1a92421d21358eb5739043c86b6d24a81348d849f887e94`。`Grok 4.6 (high)` 当前第三方字段为 Intelligence Index `44.3113073012592`、输出速度 `66.6843264403358 tokens/s`、TTFT `46.00s`、500K context、`$2/$6`，`releaseDate` 为 `2026-08-12`。9 月 15 日缓存快照的 `44.4050073012592`、`58.5035284934629 tokens/s`、`40.8595908855s` 保留为历史测量，不解释为模型 revision 或训练变化。
- DataCurve 当前 `/tmp/ds-current-1234.html` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；页面仍为 2026-09-03 更新、`generated_at=2026-09-03T22:24:37.984682+00:00`、113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`。四档 `mini_swe_agent_grok_4_6_*` 的 Pass@1、Pass@4、平均成本、输出 token 和 Agent steps 与既有记录一致，并继续绑定 4 runs、工具、环境和 verifier。
- 本轮重新尝试 xAI 发布公告、模型页、Markdown、Reasoning、Compaction、Tools 和 Remote MCP：`10.237.126.170:1234` 为 EOF/超时，`10.24.27.134:7890` 与 `10.24.27.134:8098` 连接超时。已有 9 月 15 日官方快照不冒充当前响应；没有新增官方技术结论，不新增 Grok 4.6 专属正式章节。
- 当前状态：Grok 4.6 仍为**资料级闭环**；已同步研究笔记、来源索引、模型清单、榜单解释、计划和进度。本轮 `git diff --check`、研究代码 `py_compile`、K3 manifest audit 和 704 个 Markdown 文件的围栏配对检查均通过；本轮没有新增相对链接，既有全库相对链接检查保持通过。goal 保持 active，下一步仍从两榜八家重点厂商条目选择或复验锚点。

## 2026-09-21：GPT-5.6 Luna 当前时点复验与 OpenAI 访问边界

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；只沿两个排行榜已发现的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。AA 详情快照为 `3,861,087` bytes / SHA-256 `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`，max 的 Intelligence Index `37.3244239690841`、164.5096 tokens/s、1M context、约 `$0.20/$1.20`；DataCurve 快照为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，精确 max 行 301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均成本约 `$0.6056`、平均输出约 `73.4K` token、101.68 steps。
- 结果归因保持分账：AA 是第三方 max 配置测量；DataCurve 是 `mini-swe-agent`、工具、环境和 verifier 的组合系统，不能拼成裸模型能力。
- OpenAI 官方模型索引、Luna、Reasoning 和 Prompt Caching 页面本轮经 1234 返回 HTTP `403`，7890/8098 超时，直连 DNS 失败。既有官方页面快照继续作为历史证据，不冒充当前新鲜响应，也不因访问失败断言模型不存在或 API 已变化。
- 已同步 `gpt-5.6-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、本文件、第四册 Agent harness 正文及论文/题库/练习/术语/项目/知识图谱配套。GPT-5.6 当前为**双榜资料级闭环**，不新增架构章节；参数、内部结构、完整训练/后训练 recipe、system card、独立报告、目标硬件 profiling 和线上 acceptance 仍待核验。goal 保持 active。

## 2026-09-21：GLM-5.3-Flash serving/runtime 复核完成

- 当前活动锚点回到已有双榜模型 `GLM-5.3-Flash`；本轮没有从官方页面新增模型。`GLM-5.3-FlashX` 仅作为 Z.ai 关联服务入口记录，不在两个排行榜中新增锚点。
- Artificial Analysis 当前详情快照为 `3,942,546` bytes、SHA-256 `42f880600d3637489a0c48ff27357fe7986e53510ad5ee016c7d6170bd201fae`；release `2026-08-26`、context `1,048,576`、Intelligence Index `41.807466113455`、median output speed `95.0131572798129 tokens/s`。与旧快照相比的指数/速度变化按第三方测量漂移记录，不解释成模型版本或训练变化。
- DataCurve 精确行仍为 `mini_swe_agent_glm_5_3_flash_max`：284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本 `$0.2409818562`、平均输出 `72829.77` token、约 122.89 steps；结果仍绑定 `mini-swe-agent`、工具、任务集、环境和 verifier。
- Z.ai 官方文档补证了 1M context/128K max output、thinking enabled、`reasoning_effort`、`clear_thinking`、`tool_stream`、FlashX endpoint 与 Coding Plan 配额边界；API 字段没有被写成内部训练事实。
- SGLang cookbook 已核对：45 个文本层（MLA/DSA/KDA）、24 层视觉 encoder、288 routed/top-8、原生 MTP；paged KV pool 与 KDA state pool 双账本；低延迟 MTP `5/1/6`、高吞吐可关 speculative；Blackwell/Hopper 的 FP8/BF16 KV 与 TRT-LLM/TileLang DSA 必须配对；EPD/PD 的视频、encoder disaggregation、dummy-weight 和数值 correctness 门禁已写入研究笔记与第 84 章。
- vLLM recipe 已记录 v0.29.0+、native FP8/MTP、hybrid KDA+sparse MLA、Ascend/MI355X 等路线和 FlashInfer 页面内部版本门槛差异；Transformers GLM5-Next 明确不含 MTP layer。三者分别属于 serving recipe、部署实现和基础接入证据，不能互相升级。
- 已更新研究笔记、来源索引、模型盘点、榜单解释、`plan_v2.md`、第 84 章、第 32 章及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`；`git diff --check`、研究代码 `compileall`、全库 Markdown 围栏配对和 47 个改动 Markdown 文件的本地相对链接检查均通过。

## 2026-09-21：DeepSeek V3.2 当前 AA 快照与 Exp README 复核

- 工作目录仍为 /data/zzc/llm-from-zero-to-interview，等同 /home/zzc/llm-from-zero-to-interview；本轮只沿两个排行榜已经发现的 DeepSeek V3.2 推进，没有从官方仓库、kernel 或 serving 文档新增模型。
- Artificial Analysis 详情快照 /tmp/v32-aa-20260921.out 为 3,638,730 bytes、SHA-256 4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3。当前结构化字段为 DeepSeek V3.2 (Non-reasoning)、release 2025-12-01、685B total/37B active、128K context、Intelligence Index 16.043537719683、输入/输出 0.28/0.42 美元；当前对象没有 output-speed/TTFT 字段。
- 9 月 20 日详情快照的 648B 与本轮 685B 属于第三方目录/provider 页面漂移，不构成模型 revision、训练或架构变化证据；旧快照仍保留为历史测量。当前 AA /zh 为 1,776,748 bytes / e73b156ffdc11dd391b48ac9c9f114f31bfab711ba9171a55f66c647cf22c73a；DataCurve 为 268,571 bytes / 67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870，仍没有精确 mini_swe_agent_deepseek_v3_2_* 行。
- V3.2-Exp README 为 6,899 bytes / dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74；新增收口点为 V3.1-Terminus 对齐对照、indexer non-interleaved RoPE 与 MLA layout 修复、TileLang/DeepGEMM/FlashMLA 三层实现、SGLang dsv32 镜像和 tp=8, dp=8, enable-dp-attention 启动命令。README benchmark 仍是发布方对照，不是 DataCurve。
- vLLM recipe 当前 URL 通过 1234 返回 HTTP 404，已记录为 URL/线路负证据，不解释为 vLLM 没有实现；HF/PDF 503 也只代表当前访问失败，不覆盖固定 revision 历史证据。
- 已同步 V3.2 研究笔记、来源索引、模型清单、榜单解释、plan_v2.md、第二十一册第 19 章及 INTERVIEW_BANK.md、EXERCISES.md、PROJECTS.md、GLOSSARY_EN_ZH.md、PAPERS.md、KNOWLEDGE_GRAPH.md。当前状态仍为 **AA 单榜资料级闭环**；完整 production kernel、召回/误差曲线、硬件 profiling、线上 tool-call acceptance、独立 benchmark 和完整 RL recipe 待核验。

## 2026-09-21：Claude Sonnet 5 System Card 深读与当前快照复核

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`。本轮没有从官方目录新增模型，继续沿两个排行榜已经发现的 `Claude Sonnet 5` 锚点推进；goal 保持 active。
- 当前 Artificial Analysis Sonnet 5 详情快照为 `/tmp/aa-sonnet5-20260921.html`，3,872,014 bytes，SHA-256 `2bd51075cf1ad426140a25dddb097e62af0ef4d07308c00fbc8c022b9158318e`。当前 `max` 为 Intelligence Index `38.1638712882576`、median output speed `86.1790204452512 tokens/s`、median TTFC `137.660255319s`；9 月 15 日的 `38.357696...`、约 80 tokens/s、约 202.63s 只保留为历史 provider/测量值，不解释为模型 revision 或训练变化。
- DataCurve 当前快照为 268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；Sonnet 5 的 max/xhigh/high/medium/low 仍分别为 Pass@1 `53.846%/49.667%/48.230%/39.778%/30.512%`，绑定 `mini-swe-agent`、工具、任务集、仓库环境和 verifier，不能与 AA 或 Anthropic 结果拼成裸模型分数。
- System Card 快照为 `/tmp/sonnet5-system-card-20260921.html`，10,786,341 bytes，SHA-256 `33573adb9f1871b903f79b77d7755a44cc20bb913ef48ac9000ddb090e41f7ed`；标准库文本提取 SHA-256 为 `3daed308eaedefa8ea61fe3bcfcc99a1206d03dcb02148dc63ba8ffdea9c47f6`。深读确认：训练数据只公开为公开互联网、公开/私有和合成数据的专有混合数据，并提到去重、分类、ClaudeBot 爬取和 post-training/fine-tuning；没有完整训练 recipe、参数规模或内部 adaptive-thinking 机制。
- 安全和 Agentic 证据已补录：Sonnet 5 不跨 automated AI R&D threshold，Autonomy threat model 1 适用且 stealth rate 接近零，CB-2 未跨越；没有网络安全专项训练，覆盖 ExploitBench、OSS-Fuzz、CyberGym、Firefox 147，默认 mitigations 下部分结果为 0。Claude Code 恶意请求拒答率 `92.37%`，computer-use 恶意任务拒答率 `84.68%`；Gray Swan IPI 覆盖 28 个场景、去重后 1,130 个攻击。它们均是带 safeguards/harness 的发布方行为证据。
- 发布方 benchmark 另记录 SWE-bench Verified `85.2%`、SWE-bench Pro `63.2%`、Multilingual SWE-bench `78.3%`、Terminal-Bench 2.1 `80.4%`、BrowseComp `84.7%`、OSWorld-Verified `81.2%`、GDPval-AA v2 Elo `1618`、Toolathlon Pass@1 `54.3%`、AA-Briefcase Elo `1393`；配置通常为 adaptive + max、约 5 trials，BrowseComp 使用 10M token limit 并在约 200K 触发 compaction。所有数字保留发布方/system-card harness 标签。
- 已同步 `claude-sonnet-5-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、本文件、第四册/第十六册/第十七册/第二十册/第二十四册相关章节，以及 `PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md`、`KNOWLEDGE_GRAPH.md`。当前状态为**资料级闭环**；参数、内部架构、完整训练/后训练 recipe、独立 benchmark 复现、目标硬件 profiling 和线上 serving acceptance 仍待核验。下一步从两个排行榜剩余重点 canonical 模型中选择或复验锚点。

## 2026-09-22：Grok 4.7 活动锚点闭环与 DeepSeek V3.2 归并

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改，goal 保持 active。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 三条代理取得一致的 Artificial Analysis 中文首页：`1,785,528` bytes，SHA-256 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`；DataCurve：`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。新 canonical 条目为 `grok-4-7`，详情 `Grok 4.7 (xhigh)`，AA 指数 `46.4465506302286`、约 `38.7732 tokens/s`、500K、Intelligence task cost 约 `$3.7383`。
- DataCurve 没有精确 `mini_swe_agent_grok_4_7_*` 行，因此不迁移 Grok 4.6 结果。xAI 模型页快照 `376,912` bytes / `c3ddd6b44b97fb4b527096ca69e4d9eacdca99e0e4e44427c9da5b81a615181e`，发布页 `288,007` bytes / `af8eda968823b546480818ba27f6edf7f9787b5126c51a9f5ac4568e23da406f`；已补录 500K、effort、不可关闭 reasoning、Responses `encrypted_content`、compaction opaque item、function/structured/MCP 工具协议。
- xAI 发布方披露更大 base、更长 RL run、困难长任务混合、自验证、长上下文管理和 safeguard stack，并给出 CursorBench 4.0 `46.3%`、DeepSWE v1.1 `71.0%`、EEBench `64.0%`、AA Briefcase `1657`、Terminal-Bench `38.0%`、Harvey `19.6%`、HealthBench Professional `56.7%`、GDPval `1695` 等数字；均保留 benchmark/harness/effort 口径，不与 AA/DataCurve 合并。
- arXiv 精确标题检索无 Grok 4.7 结果，快照 SHA-256 `8f8c4bb0f6c8a0346e28a56864a4aa4d630bdde9c536ee1320be29f84e6e9da6`；不补写参数、架构、完整训练 recipe、生产 kernel 或独立 benchmark。Grok 4.7 当前状态为**AA 单榜内容专题闭环**。
- DeepSeek V3.2 的 `0925`、reasoning/non-reasoning、Speciale 对象已核验为 deprecated/redirect 的历史 revision、配置或同家族专项 checkpoint；DataCurve 没有精确 V3.2 行，不新增 DeepSeek 模型，不迁移 V3.1/V4 的 Agent 分数。
- 已同步 Grok 4.7 研究笔记、模型清单、来源索引、榜单解释、`plan_v2.md`、本文件和书系/全局配套；下一步完成全仓库 Markdown、相对链接、Python 编译和证据审计，然后继续从两个排行榜的八家重点厂商 canonical 条目选择锚点。目标保持 active，不标记 complete。

## 2026-09-22：Qwen3-Omni 30B A3B 专题落地

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮沿 Artificial Analysis 已发现的 `Qwen3 Omni 30B A3B` 推进；instruct 详情三代理均 HTTP 200、`3,833,501` bytes、SHA-256 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`，reasoning 快照为 `3,845,331` bytes、SHA-256 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。DataCurve 没有精确 `mini_swe_agent_qwen3_omni_*` 行，不迁移其他 Qwen 的 Agent 结果。
- 已核验 Qwen3-Omni 官方 GitHub、HF Instruct/Thinking 配置、Qwen 博客、阿里云 Qwen-Omni 文档、arXiv `2509.17765` 技术报告及源码；补齐 Thinker-Talker、AuT（约 2,000 万小时监督音频、8 倍 Conv2D、12.5 Hz）、Qwen3-VL/SigLIP2-So400m 视觉 encoder、TM-RoPE `24/20/20`、多码本 Talker/MTP、Code2Wav、三阶段预训练、Thinker/Talker 后训练和异步 chunked prefill。
- 已新增研究笔记 [`qwen3-omni-source-notes.md`](research/model-update-2026-09/qwen3-omni-source-notes.md) 和第二十一册第 91 章 [`Qwen3-Omni：Thinker-Talker、AuT 与流式多模态`](book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)，并同步模型清单、来源索引、榜单解释、计划文件。Instruct、Thinking、Captioner 作为同家族 artifact 归并，不新增三个基础模型。
- 报告的 audio/video first packet `234/547 ms` 继续标注为论文/发布方口径，不是本机实测；官方 README 当前说明 vLLM 主要支持 Thinker，Instruct 音频输出仍在实现推进中。完整参数账本、生产 kernel、目标硬件流式 profiling、音频 acceptance、完整 recipe、线上工具验收和独立 benchmark 仍待核验。
- 当前状态：**AA 单榜内容专题闭环**；下一步同步第四册百科、reasoning/Agent/serving 交叉段落、全局论文/题库/练习/术语/项目/知识图谱、BOOK_SERIES/ROADMAP 和第二十一册目录/时间线，然后执行 Markdown、链接和章节代码验证。goal 保持 active。

## 2026-09-22：K2 Horizon 3.7B dense 对照专题收口

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮继续处理已由 Artificial Analysis 发现的 `K2 Horizon 3.7B`，没有从 IFM 官方目录、HF 关联模型或 Uno 论文另发现模型。AA 三条代理详情页逐字节一致：`3,756,408` bytes、SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`；字段为 `releaseDate=2026-09-03`、reasoning、open weights、3.7B、524,288 context、Apache 2.0、AA Index `15.6103951330976`。
- DataCurve 当前快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` 没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 或 K2 Horizon 行；没有迁移 K2 36B、K2.7 Code、K3 或其他模型的 Agent 分数。
- 已固定 IFM main revision `6360f705b2e57d542959e6a2e67ebeb95dae0373`（`lastModified=2026-09-21T01:40:06Z`），当前 artifact 为 `K2HorizonForCausalLM`、36 层、2560 hidden、32Q/8KV GQA、head dim 128、vocab 250,624、524,288 position、default RoPE、`num_experts=0`、`mova_num_experts=0`、BF16；权重 index 为 36 shards/327 tensors/`10,116,510,720` bytes。3.7B 是 dense 对照，不能继承 36B MoVA value top-4、FFN top-8 或 shared expert 的 forward-time 路由。
- 训练证据已记录 22.9T/8K pretraining、32K/128K/512K 分阶段 midtraining、512K SFT Phase 1/2、Math/Code/STEM-Code RL 分支和 self-attention ISO merge/other weights RAM；中间 checkpoint 可用于阶段能力比较。`/tmp/k2-migration-fixed-20260922.out` 固定为 `k2_aurora -> k2_horizon`、copy、`weights_reencoded=false`、BF16、36 shards/327 tensors，归类为 artifact 迁移证据，不是重新训练证据。
- 已记录 vLLM recipe（5.06B dense、512K、H200、`k2_horizon` reasoning/tool parser）和 SGLang PR #37654（TP1/BF16/FlashAttention-3/H200、发布方 TTFT/TPOT/GSM8K）；SGLang 文档引用的 revision `c177771836a4c460743c00002c22483f6f18d1eb` 当前 HF raw/API 404，只记为旧部署 revision 不可解析。发布方 benchmark 不写成本机实测或统一裸模型分数。
- 当前文档冲突已分账：current config/migration/index 支持 `K2HorizonForCausalLM` + BF16；旧 `APPENDIX.md` 的 `XllmForCausalLM`/FP32 和 `3.78B core / 5.06B including embeddings` 标为旧 revision/残留字段。已新增 [`k2-horizon-3.7b-source-notes.md`](research/model-update-2026-09/k2-horizon-3.7b-source-notes.md)，扩展第二十一册第 82 章，并同步 model inventory、source index、inventory interpretation、`plan_v2.md`。
- 当前状态：**AA 单榜资料级闭环**。仍待核验完整训练 recipe、生产 kernel、目标硬件 profiling、线上 tool acceptance、独立 benchmark 和精确 DataCurve Agent 行；下一步完成本轮书系配套文件同步和全仓库质量检查，再从两榜八家重点厂商的既有 canonical 条目选择下一锚点。该段是 K2 历史锚点记录，当前活动锚点以文件顶部和最新记录为准；goal 保持 active。

## 2026-09-22：Qwen3-VL-235B-A22B 专题落地

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 本轮沿 Artificial Analysis 已发现的 `Qwen3-VL-235B-A22B` 推进；instruct/reasoning 两个详情页三代理逐字节一致，AA 页面字段约为 `235B total / 22B active`、`262,144` context。DataCurve 当前没有精确 `mini_swe_agent_qwen3_vl_*` 行，不迁移其他 Qwen 的 Agent 分数。两个配置归并为同一基础模型，不新增两个模型。
- 已核验 Qwen3-VL 官方 GitHub、HF Instruct revision `710c13861be6c466e66de3f484069440b8f31389`、Thinking revision `6664affde68449468deb7527186455c7450c13c0`、config、README、Qwen3-VL Technical Report `arXiv:2511.21631` 和 Transformers raw main。已固定 `Qwen3VLMoeForConditionalGeneration`、94 层、64Q/4KV、128 experts/top-8、262K position、RoPE theta `5,000,000`、视觉 depth 27、patch 16/temporal patch 2、DeepStack `[8,16,24]`。
- 研究和章节已写入三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3（67B/1T/1T/100B、8K/8K/32K/262K）训练阶段、square-root normalized loss、SAPO/General RL、Thinking with Images、GUI/tool reward 和多模态 serving 证据边界；新增研究笔记 [`qwen3-vl-source-notes.md`](research/model-update-2026-09/qwen3-vl-source-notes.md) 与第二十一册第 92 章 [`Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳`](book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)。
- 当前状态：**AA 单榜内容专题闭环**。完整权重加载、生产视觉/稀疏 kernel、跨视频 chunk replay、目标硬件 profiling、端到端 multimodal streaming、GUI/tool acceptance、完整训练/后训练 recipe 和独立 benchmark 仍待核验；不将 goal 标记为 complete。

## 2026-09-22：Opus 5 与 Gemini 3.5 Flash-Lite 断点复验

- 已确认工作目录为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；既有 dirty worktree 修改全部保留。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、GLM/Z.ai、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。
- Claude Opus 5：当前 AA 详情快照 `3,869,351` bytes / SHA-256 `c18260ab4ff331d5bd3305691db2d4b6051dc2ebe642aa1458c5b8fa2c367643`；当前第三方字段为 Intelligence Index `50.7771115797629`、`56.4471785104486 tokens/s`、`49.2490756305s` TTFC、1M context 和约 `$5.8584`/task。9 月 21 日的速度/TTFC 保留为历史值；Opus 5 的 DataCurve max 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`，不迁移相邻 Claude 版本结果。状态：**资料级闭环**。
- Gemini 3.5 Flash-Lite：AA 详情快照 `3,858,652` bytes / SHA-256 `bbe13cb52c85772080a105b1ec98c42bd71e7194dd67abbc26cb52f9ba1115d`；当前第三方字段为 release `2026-07-21`、Intelligence Index `22.1685424839812`、`386.373295824977 tokens/s`、`11.2053542005s` TTFC、1M context 和约 `$0.1235`/task。DataCurve 快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` 没有精确 Lite Agent 行；不迁移相邻 `gemini_3_5_flash_high` 结果。状态：**AA 单榜资料级闭环**。
- 已将两份研究笔记、`model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan_v2.md` 和本进度表同步；两者均不新增重复 Transformer 正式章节。下一步执行全仓库 Markdown 围栏、相对链接、研究 Python `py_compile`/`compileall`、`git diff --check` 和证据边界审计；goal 保持 active。

## 2026-09-22：GLM-5.1 当前时点复验与活动锚点切换

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。
- 三条代理取得一致的 Artificial Analysis GLM-5.1 详情页；当前快照 `3,971,543` bytes / SHA-256 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`，中文首页 `1,798,627` bytes / SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`。页面仍为 `GLM-5.1 (Reasoning)`、release `April 2026`、200K context；当前第三方测量为 Intelligence Index `26.0585912980095`、`37.1922485381327 tokens/s` 和 cost per Intelligence Index task `0.9217822401466147`。与 9 月 20/21 日的速度和价格差异按 provider/采集时点漂移记录。
- DataCurve 当前快照 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` 没有精确 `mini_swe_agent_glm_5_1_*` 行；不记录或迁移 GLM-5/5.2/5.3/5.3-Flash 的 Pass@1、成本、输出 token 或 Agent steps。
- 当前 Z.ai 文档与博客证据支持 long-horizon Agent、multi-turn SFT/RL/process-quality evaluation、VectorDBBench 外循环、KernelBench Level 3 verifier/harness、Linux desktop 自评、DSA/MoE 配置和 thinking/tool/cache 协议。博客、AA 和 provider 字段继续分账；模型卡链接的是 GLM-5 报告，不能把 GLM-5 的训练/架构结论改写为 GLM-5.1 专属事实。
- 已更新 `glm-5.1-source-notes.md`、`model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan_v2.md` 和本进度表。当前状态仍为 **AA 单榜资料级闭环**，不新增重复 Transformer 正式章节；完整 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、线上 tool acceptance、独立技术报告和复现仍待核验。goal 保持 active。

## 2026-09-22：GPT-5.6 Luna 当前活动锚点与官方运行时补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商范围仍为八家。
- 本轮沿两个排行榜已经确认的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。Artificial Analysis 详情 HTTP 200，当前快照 `3,861,318` bytes / SHA-256 `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；第三方字段为 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `158.728370482714 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context、页面 input/output `$0.20/$1.20`。
- DataCurve 当前快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；精确 `mini_swe_agent_gpt_5_6_luna_max` 行为 attempted `448`、passed `301`、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 Agent steps `101.68080357142857`、median peak context `201647`。这些结果继续绑定 `mini-swe-agent + tools + task environment + verifier`，不写成裸模型分数。
- 7890 代理取得官方 Markdown 页面并记录哈希：Luna `3,744` bytes / `1f425d8f...fed0`，Reasoning `70,253` / `91604df9...6719`，Agents `5,432` / `df4f61b6...43a`，Tools `33,282` / `4722fa10...a341`，Tool Search `39,288` / `9d6c3855...1fd8`，Prompt Caching `47,097` / `c70d858e...b2d1`，Compaction `14,272` / `73fd2fd1...0dd`；完整 URL、大小和 SHA-256 见 [`gpt-5.6-source-notes.md`](research/model-update-2026-09/gpt-5.6-source-notes.md)。
- 新增技术知识已完成归纳：Agents API 托管 Codex harness 和 session/turn/item 状态，Agents SDK 由应用控制 loop、工具、handoff、部署、存储和审批，Responses API 由应用直接管理 response item 与工具循环；tool search 分 hosted/client 两种，分别由服务端加载或应用回传 `tool_search_call`/`tool_search_output`；工具加载追加到上下文末端以尽量保留 cache prefix，但不改变权限、沙箱、幂等和 verifier 责任。
- 已将 prompt cache 与 compaction 的交互写清：model、tools、parallel tool calls、structured output、reasoning effort、verbosity 和 `context_management` 可能影响前缀；compaction 会替换旧上下文并造成首次 cache miss，tool search 会减少初始 schema token 但增加工具发现/版本状态。当前状态为 **双榜资料级闭环**，暂无 GPT-5.6 专属架构章节；正式书章和题库同步后，继续做全库质量检查，goal 保持 active。

## 2026-09-22：Qwen3.7 Plus 当前活动锚点与托管 Agent 合同

- 工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口继续严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商继续为 OpenAI、Anthropic、Z.ai、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮从 Artificial Analysis 当前详情选择 `Qwen3.7 Plus` 作为下一活动锚点：`/tmp/q37-aa-20260922.out` 为 `3,859,009` bytes / SHA-256 `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040`；release June 2026、Intelligence Index `25.1622215821984`、`68.5428061089526 tokens/s`、cost per Intelligence Index task `0.32527343198119174`、约 1M context 和约 `$0.40/$1.60` 均是第三方/provider 字段。
- DataCurve 当前快照 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，生成时间 `2026-09-22T06:27:15.860279+00:00`；没有精确 `mini_swe_agent_qwen3_7_plus_*` 行，不迁移其他 Qwen 的 Agent 分数。
- 重新取得 Alibaba Cloud 官方资料：Qwen3.7 Plus 文档 `44,386` bytes / `662ebedd3a538e22490e00c4eb29def16c0c5e9ff2de1d3f8fe7d7d334b43656`，推荐模型页 `27,344` / `9df37ab1723f032187895b9961865f2831cb575147985a281310d0616f8f4fe8`，OpenAI-compatible API 文档 `60,448` / `ec3f4a6c3db9ea89fc4141e78cfcfaf389aafa67ff11efec664b3600aa40714`；完整 URL 和快照表见 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)。
- 已提取的面试主线：多模态交互式混合 Agent、读屏/GUI/移动端导航和视觉参考生成代码是官方产品定位；模型只提出 action proposal，region/scope、schema、权限、GUI/mobile executor、观察回灌、幂等、重试和 verifier 组成真实闭环。Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching 按 region/scope 分层，不能写成无条件的模型本体能力。
- 已记录官方上下文合同：1M context、991,808 max input、131,072 max output；thinking max input `983,616`，max chain-of-thought length `262,144`。北京/Global scope capability 与 Virginia US scope 不同，价格按区域、scope、输入长度、缓存和 batch 分档；不能用 AA 的 `$0.40/$1.60` 代替所有 provider 价格。
- 已新建 [`qwen3.7-plus-source-notes.md`](research/model-update-2026-09/qwen3.7-plus-source-notes.md)，并更新 `model-inventory.md`、`source-index.md`、`inventory-interpretation.md`、`plan_v2.md`。接下来完成第十五、十七、二十、二十四册已有章节和八个全局面试/索引文件的同步；当前状态为 **AA 单榜资料级闭环**，goal 保持 active。

## 2026-09-22：DeepSeek V4.1-Flash 当前活动锚点与联网恢复复验

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree 修改。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。本轮没有从 DeepSeek 官方目录、HF 关联文件或 recipe 仓库另发现模型。
- 三条代理对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 HTTP 200：AA 首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。没有新的重点厂商 canonical 模型。
- 沿既有锚点复验 DeepSeek V4.1 发布页和 AA `deepseek-v4-1-flash` 详情页，三条代理均 HTTP 200。发布页 `27,838` bytes / `bea79d60a0712f1971554724c94e8ff2145a048d3c0e80326efa4bacd2bf8e11`；AA 详情 `3,948,908` bytes / `114cc90d1cb8125174d9464141cbfd0faeac76f52b132f78630c0375c9fcf8fe`。AA 当前 Index `39.456167472527`、约 1M context、约 `$0.30/$1.20` 是第三方/provider 字段，不是本项目实测。
- DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移 V4 Pro/V4 Flash 或其他 DeepSeek 版本的 Agent 成绩。HF API 经 7890 成功取得 `6,714` bytes / `df3cb8b368d3a77a4eb3b96c8a4f85abfb3a199245bf9006a68d374a4b425ca3`，revision 仍为 `dba1be0a40aa45a94ad051997016db3960a90277`、`lastModified=2026-09-10T08:18:10Z`、48 个 safetensors 分片；8098 的 503/连接失败与 1234 失败只记录为线路边界。`deepseek-recipe` 经 1234/7890 可获取，8098 对 GitHub raw 为 TLS EOF，内容未见变化。
- 已同步 `plan_v2.md`、DeepSeek 研究笔记、`source-index.md` 和 `inventory-interpretation.md`。当前状态保持 **内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**；没有新模型或新权重 revision。后续只补完整权重加载、production kernel、candidate/index Top-K recall、真实 FP4 误差、DSpark draft/verify/rollback、EPD 调度、目标硬件 profiling、tool acceptance 和独立 benchmark，不新增重复 Transformer 章节。

## 2026-09-22：DeepSeek V4.1-Flash vLLM upstream runtime 补证

- 本轮沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点继续推进，没有从 vLLM 仓库另发现模型。通过 1234 和 7890 代理取得的 vLLM `main` registry 逐字节一致；8098 对同一组 raw 文件超时。`main` registry 快照为 `64,391` bytes / SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，包含 `DeepseekV41ForCausalLM -> vllm.models.deepseek_v41` 与 `DSparkV41DraftModel -> vllm.models.deepseek_v41`；正确实现路径是 `vllm/models/deepseek_v41/` 包。
- 已固定 `__init__.py`、`quant_config.py`、NVIDIA/ROCm `vl_model.py` 与 `dspark.py` 快照及 SHA-256。代码证据确认：`expert_dtype=fp4/fp8` 分支分别对应 MXFP4 + `ue8m0` scale、block-FP8 + float32 scale；vision wrapper 通过 `inputs_embeds` 注入 ViT/aligner embedding、保留 raw `input_ids` 供 `bias_vl` 路由、支持 encoder CUDA graph/ViT data parallel，并显式跳过 `mtp.*`，当前 vision variant 不支持 MTP/DSpark draft heads；DSpark 使用 3 个 draft 层、目标层 ids、`[max_num_batched_tokens, index_topk]` Top-K buffer、target checkpoint 的 `mtp.{0,1,2}.*` 权重、Markov/confidence head、逐位置 sigmoid confidence、共享 embedding/lm head、context KV/SWA cache 插入和 FP4/FP8 scale 分支。
- 对照 vLLM `0.29.0` stable registry：`63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`，本快照有通用 V4/DSpark 入口，但没有两个 V4.1 专用类名。因此当前状态更新为 **内容专题 + HF reference implementation + vLLM upstream main runtime evidence + recipe protocol evidence（AA 单榜）**。这不等于 stable wheel 已支持、完整权重已加载、GPU/ROCm 已验收、speculative acceptance/吞吐已复现或生产 SLO 已证明。
- 已同步 `plan_v2.md`、DeepSeek 研究笔记、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md` 和第二十一册第 81 章；第 81 章新增 vLLM runtime 小节、2 个面试追问和来源。后续仍补完整权重加载、stable release/目标硬件、candidate/index Top-K recall、真实 FP4 误差、DSpark verify/rollback/acceptance、EPD 调度、GPU profiling、tool acceptance 和独立 benchmark，不新增重复 Transformer 正式章节，goal 保持 active。

## 2026-09-22：DeepSeek V4.1-Flash SGLang main/stable runtime 对照补证

- 本轮沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点推进，没有从 SGLang 仓库另发现模型。通过 `10.237.126.170:1234` 的 GitHub Contents API 固定 SGLang `main` 的 `deepseek_v4.py`、`deepseek_v4_dspark.py`、`deepseek_v41_vit.py` 和 `deepseek_v4_nextn.py`，分别为 `241,421`、`47,640`、`5,126`、`9,459` bytes；完整 SHA-256、Git blob 和 URL 已写入 DeepSeek 研究笔记与来源索引。raw 端点超时只代表线路边界，不能解释成源码不存在。
- main 代码确认 V4.1 vision 的 TP/EP/DP 路径和 CP/PP/MoE A2A 限制、2D-RoPE ViT/Aligner、attention data parallel、V4/V4.1 FP8 block-size 差异、MXFP8/FP8 prefill autotune、FlashInfer、unified KV、DSV4 sparse indexer/cache，以及 DSpark 的 Markov/confidence head、`mtp.*` 映射和 draft stage `vision_n_layers=0`。这属于 **SGLang upstream main runtime implementation evidence**。
- 对照 SGLang `v0.5.20`（tag `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布 `2026-09-18T22:41:33Z`）：stable 只有通用 V4/DSpark 文件，`deepseek_v41_vit.py` 404，相关源码中 `deepseek_v41`/`dsv41` 计数为 0。不能把 main 实现升级为 stable V4.1 serving 支持。
- 已同步 `deepseek-v4.1-flash-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md` 和第二十一册第 81 章，新增 SGLang 小节与面试追问。当前状态为 **官方内容 + HF reference + vLLM main + SGLang main + recipe protocol evidence（AA 单榜）**；完整权重、目标硬件、视觉 draft/target verify、FP4/FP8 质量、DSpark acceptance/rollback、EPD 调度、profiling、tool acceptance 和生产 SLO 仍待核验，goal 保持 active。

## 2026-09-22：Kimi K3 SGLang stable/main runtime 对照

### 后续联网补证：SGLang main commit history

- 前一条代理失败记录属于当时的首次尝试；随后网络恢复，百度经三条代理均 HTTP 200，GitHub commits API 经 10.237.126.170:1234 成功，7890 返回 403，8098 出现 TLS/代理协议错误。
- K3 commit history 响应为 51,251 bytes、SHA-256 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，最新记录时间 2026-09-22T06:35:12Z。当前 main 文本源码仍为 171,101 bytes、blob 383a6f47812bccd1cb91b76814cd0730ff945dd7、SHA-256 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e。
- 已补记 8ac19cc deferred KDA gate、c4d3770 CUDA graph stream、c2c3629 O(1) expert lookup、72d5c5b FP32 routing finalize、f4c2563 PP/DCP/DSpark、2d0e94e shared-expert process group、8ac39c6 Ascend A5/NPU 和 cb32dbc ROCm KDA projection。完整解释已同步研究笔记、来源索引、模型清单、正式章节、推理引擎章节和题库。
- 这些提交仍是 mutable main source evidence，不等于 stable wheel、完整权重、双状态恢复、目标硬件 profile、视觉正确性、DSpark acceptance 或线上 tool/verifier 通过；goal 保持 active。

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，等同 `/home/zzc/llm-from-zero-to-interview`；不回滚既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为八家范围。本轮没有从 SGLang 源码目录另发现模型。
- 已固定 SGLang `v0.5.20` K3 text 源码：168,114 bytes、blob `b0ede48c88264d518351a66abf623f1bcf8a730e`、SHA-256 `7a3ef867394a2fd52b3a71a979c053b51e9f2310c35cd4c12ef3aa8e7be172e5`；tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`、发布时间 `2026-09-18T22:41:33Z`。已固定 SGLang `main` K3 text：171,101 bytes、blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`、SHA-256 `54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e`。
- 两版本 `kimi_k3_vl.py` 逐字节一致：32,790 bytes、blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`、SHA-256 `2924c38f652a6ebff2ef79c49f2f336ba18723ea4b854d3ac14d95292988acdb`。因此本轮新增差异主要是文本 runtime：LatentMoE 的 latent down/up、MegaMoE/DeepEP/Mooncake/Ascend-FuseEP/MoRI A2A、DP/SP token shard、shared-expert TP/reduce-scatter、SBO/NPU dual-stream、ModelSlim fused QKVG/packed loader、KDA fused decode capability gate 和 fallback。
- 已同步 K3 研究笔记、来源索引、模型清单、榜单解释、`plan_v2.md`、第二十一册第 88 章和第二十四册第 32.52 节；面试题下一步补入 stable/main/source/target acceptance 对照。当前状态为 **内容专题闭环 + HF/vLLM + SGLang stable/main source evidence**，不等于完整权重、SGLang 依赖安装、双状态恢复、视觉数值、目标硬件 profile、线上 tool acceptance 或生产 SLO 已通过。
- 本轮重新请求 GitHub commits API 时，`7890`、`1234`、`8098` 三条代理均连接失败；因此没有固定 mutable `main` 的 commit history，也没有把失败解释成源码不存在。网络恢复后优先补 history/commit message，再决定是否需要更新源码哈希和证据等级；goal 保持 active。

## 2026-09-22：两榜复验与 gpt-oss provider 兼容性/raw CoT 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；不回滚既有 dirty worktree。三条代理重新获取两个唯一排行榜并逐字节一致：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve DeepSWE `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。
- 按八家重点厂商 canonical slug 归一化后没有新的重点模型。当前 AA 页面中可见的 K2 Horizon 0.9B、7B、375B 等其他厂商条目按用户范围排除，不建立新的锚点，也不把 K2-Horizon-7B-Uno 作为排行榜发现。
- 选择已有 Artificial Analysis 锚点 `gpt-oss-120b`/`gpt-oss-20b` 继续补证。gpt-oss README 当前 `24,453` bytes / SHA-256 `578ad0f82c1d823229f9bf2b52b3f1c55ce82772ac57f634f0ca4eb46a6370aa`，与 9 月 18 日一致；GitHub commit history `44,611` bytes / `420dcfab2dad69aa386b5207ac3aa35f627151e292c63caced716d46c59e288c` 的最新提交 `7b583341fe16`（2026-07-24）是 You.com backend API key 修复，不是模型或核心推理更新。
- OpenAI Cookbook 当前专题新增/列出两个直接相关的官方入口：[实现兼容性验证](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)，`336,633` bytes / SHA-256 `4f293d2bd51666966a8d3657893eb1d1093b4b3230a405c9178cbc7302d664b8`；[raw CoT 处理](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot)，`338,262` bytes / SHA-256 `7330273ff59d6e1a344abe3f435ae421a20ebd775bda9122d22ff606f0782072`。
- 新知识已写入：Responses `reasoning.content[].reasoning_text`、`response.reasoning_text.delta/done`、item/index/turn lineage replay；Chat Completions provider 的 `reasoning`/delta 兼容约定；raw CoT 不直接面向终端用户；Harmony/API shape 与 tool-call smoke test、AIME 16 次/题、GPQA 8 次/题、HealthBench 1 次/题的质量 eval 分层。
- 官方 compatibility-test 的 0 invalid requests 且 `pass@k`/`pass^k` 均超过 90% 只是强信号，不等于 MXFP4/MoE kernel、硬件 profiling、独立复现或生产 acceptance。当前 gpt-oss 状态为**内容专题闭环 + 官方 provider 兼容性/raw CoT protocol evidence**；完整 recipe、kernel/误差、目标硬件、精确 DataCurve Agent 行和线上接受率仍待核验。
- 已同步 `gpt-oss-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、第二十一册第 87 章、第二十四册 serving 章节、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`；goal 保持 active。

## 2026-09-22 GLM-5.3-Flash upstream/runtime 对照补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree，不回滚其他模型专题。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，当前活动锚点为 `GLM-5.3-Flash`。
- 本轮通过 `10.237.126.170:1234` 取得 SGLang/vLLM GitHub Contents/tree 与提交历史；`10.24.27.134:7890` 对部分 GitHub API 返回 403/rate limit，`10.24.27.134:8098` 出现 TLS wrong-version/读取失败；绕过代理时 DNS 不可用。代理失败只作为访问路径边界记录，不解释为源码或页面不存在。
- SGLang `v0.5.20` tag（commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`）已固定 `glm5_next.py`：`61,466` bytes、SHA-256 `12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2`；对应 config `12,026` bytes、SHA-256 `3b3c7aa3ae60e1cf59edf91e1c11a6aa49be7532340c8e2482f7f75ed859f3e0`。这证明固定 stable source entry，不代表本机权重加载或目标硬件 profile。
- SGLang `main` tree commit `9d58189c12e4e14a7eea20f24f9aa7b17e221778` 的 GLM5Next 模型文件为 `68,097` bytes、SHA-256 `1cd324533aa0827e7542e27fc39c5901c2330823c4b3a32b79bcb26521dbcfec`；补证 projection fusion、KDA projection/prefill metadata、mHC boundary fusion、AMD FP8/Quark MXFP4 和 B200/H200 测试门禁。相关提交 `c8eb54c41da1`、`2fa6b94e3440`、`b44e2486824e` 作为 mutable upstream history 记录。
- vLLM `main` tree commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa` 已有 `vllm/models/glm5next/`：`attention.py`、`kda.py`、`model.py`、`mtp.py`、`common/sparse_indexer.py`。源码事实包括 pool 粒度 `Glm5NextIndexerCache`、保存未完成 pool raw BF16 K/gate score 的 `Glm5NextTailCache`、独立 KDA/Mamba/GDN state、`num_spec` conv 宽度和 MTP layer/weight mapping、top-k reuse、slot compact/local argmax。
- vLLM `v0.29.0` tree 快照为 `1,979,822` bytes、SHA-256 `7131879ae9592d90738776d24ae213f789577317b2c5427f350761a87ca05070`，路径搜索未发现 `vllm/models/glm5next/` 专属路径。因此 recipe 的 `v0.29.0+` 只能记录为部署门槛/路线声明；当前正确状态是 SGLang stable source + SGLang main + vLLM main implementation evidence，不能写成 vLLM `v0.29.0` stable tag 已含 GLM5Next。
- 已同步研究笔记、来源索引、模型清单、榜单解释、`plan_v2.md`、第二十一册第 84 章、第二十四册第 32.47.6 节、`INTERVIEW_BANK.md`、`EXERCISES.md`、`PROJECTS.md`、`PAPERS.md`、`GLOSSARY_EN_ZH.md` 和 `KNOWLEDGE_GRAPH.md`。完整权重、stable wheel 运行、目标硬件数值/性能、PD/EPD recovery、MTP acceptance、tool/verifier 和生产 SLO 仍待核验；goal 保持 active。

## 2026-09-22 GLM-5.3 标准 DSA runtime 闭环推进

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同。保留既有 dirty worktree；模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，当前活动锚点从 `GLM-5.3-Flash` 切换为标准 `GLM-5.3`。
- 三代理新鲜排行榜复验一致：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商没有新的 canonical 模型，因此没有新增模型名称，也没有把 Flash/其他模型的 Agent 结果迁移到标准版。
- 标准版 HF revision 固定为 `aca966e4e02791568aa6a4ced368624b3d897f42`；config 为 `29,464` bytes / SHA-256 `3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`。它确认 `glm_moe_dsa`、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_share_for_mtp_iteration=true` 和 1,048,576 positions；实现账本为 21 个 Full、57 个 Shared indexer 层。
- Transformers main ref 为 `0bc252863a4e5c0e709893664cc57369ebcd7353`，固定了 interleaved indexer RoPE、Full 层 top-k、Shared 层 `prev_topk_indices` 和 sparse mask。vLLM main ref `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa` 与 v0.29.0 registry 都把模型映射到 `deepseek_v32`；SGLang main ref `861b11f087af2822cb545ea1895059a721831014` 与 v0.5.20 都包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。完整 bytes/blob/SHA-256 已写入 `source-index.md` 与研究笔记。
- 已同步 `research/model-update-2026-09/source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、本文件、第二十一册第 89 章、第二十四册第 32.53 节，以及 `INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md`、`PAPERS.md`、`KNOWLEDGE_GRAPH.md`。标准版与 Flash 的边界已明确：不迁移 Flash 的 `RadixLinearAttention`、KDA、视觉、双 state pool、EPD 或 `glm5_next.py`。
- 当前状态：**双榜资料级闭环 + stable/main runtime source evidence**。source entry 不等于完整权重加载；index/evidence recall、MLA/indexer cache recovery、MTP acceptance、目标硬件 profiling、tool/verifier acceptance 和生产 SLO 仍待实测。下一步按 gate 做本地导入/最小构造与可用硬件 profiling；若环境不具备权重或目标硬件，保持 `unverified`，goal 继续 active。
- 本轮本地环境门禁已检查：`torch`、`transformers`、`vllm`、`sglang`、`safetensors` 均未安装；`nvidia-smi` 无法连接 NVIDIA driver；仓库及 `/tmp` 未发现标准 GLM-5.3 权重分片。故没有伪造 full-weight load、数值正确性、index/evidence recall、MTP acceptance 或 GPU profile 结果；这些项继续标记为 `unverified`，待具备依赖、权重和目标硬件后再执行。

## 2026-09-22 Grok 4.7 当前时点 reasoning/MCP 增量补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree，goal 保持 active。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 三条代理刷新排行榜：Artificial Analysis 中文首页 `1,762,420` bytes / SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。没有新的重点厂商 canonical 模型；Grok 4.7 仍没有精确 `mini_swe_agent_grok_4_7_*` 行。
- `10.24.27.134:7890` 的 xAI 刷新快照已固定：Reasoning `522,822` bytes / `ce3cd1997308094472d2cce8017eee510cf3441c7be2711029870c9d18d1cc87`，Compaction `521,486` / `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`，Remote MCP `475,038` / `459b83b7522eaa9139677f9da9760593e56072d52fd63a1883ba6a370a13c705`；模型页 `376,913` / `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5`，发布页 `288,007` / `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`。
- 新增技术点：Grok 4.7 Responses 每次返回 `reasoning.encrypted_content`，服务端工具的加密输出也需原样保留；Reasoning summary 通过 `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` 流出，但不是完整 CoT；服务端 thinking trace rehydration 与 `store` 决定的 `previous_response_id` 存储行为分开；`presencePenalty`/`frequencyPenalty`/`stop` 是 reasoning 请求兼容性禁用字段；`xhigh` 从 Grok 4.6 起可用。
- Remote MCP 新增协议边界：只支持 Streaming HTTP/SSE；Responses 的 `allowed_tools`/`headers` 与 xAI SDK 的 `allowed_tool_names`/`extra_headers` 不同；`server_description` 可选；OpenAI Responses API 当前不支持 `require_approval`/`connector_id`。这些字段仍不替代宿主授权、凭据隔离、网络策略、审计、幂等和 verifier。
- 已同步 Grok 4.7 研究笔记、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`plan_v2.md`、本文件和必要书系/题库/全局面试配套；当前仍为 **AA 单榜内容专题闭环**，不新增重复 Transformer 章节。参数、内部架构、完整训练/后训练 recipe、完整权重加载、生产 kernel、硬件 profiling、线上 acceptance 和独立复现继续标记为待核验。

## 2026-09-23：DeepSeek V4.1-Flash vLLM `v0.30.0` stable release

- 本轮继续沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 锚点推进，没有从 vLLM/PyPI/SGLang 的仓库目录另发现模型。DataCurve 当前仍没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移 V4 Pro、V4 Flash 或其他 DeepSeek 版本的 Agent 结果。
- vLLM `v0.30.0` release 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定 [release API 快照](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.30.0) 为 `65,334` bytes / SHA-256 `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；[v0.30.0 registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py) 为 `64,420` bytes / SHA-256 `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`，已包含 `DeepseekV41ForCausalLM` 与 `DSparkV41DraftModel`。
- v0.30.0 stable package 的固定源码证据为：`__init__.py` `625` bytes / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe`；`quant_config.py` `9,006` bytes / `bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9`；NVIDIA `vl_model.py` `13,728` bytes / `a5a3f477225990946093092d4781db181b59b52102aff2e0e345e631973817f1`；NVIDIA `dspark.py` `23,059` bytes / `4d9c2bfa4c123aa5b95b637f24dc3d04748227455857bdc1376b27cda8af5954`；ROCm `vl_model.py` `13,736` bytes / `6f3fcc8a5896432ef51f809348097e92c7782ab226adb5ecb32cdc599ce05af4`；ROCm `dspark.py` `22,731` bytes / `a109e581711a74a7c5597b3f5a07d81ed05aac0ed2619dfa3f859050efc0b9d9`。
- v0.30.0 release notes 的面试级新增点包括：DeepSeek V4.1-Flash 模型接入；SM100 FlashMLA V4.1 record 的 MXFP8 whole-KV；DeepGEMM Mega-mHC；mHC post block folded into delayed pre projection；Triton-fused input metadata preparation；CPU-offloaded Engram async prefetch 与 Engram DP sharding；DSpark draft states 在 sequence-parallel all-gather 前折叠；DSpark 未继承未初始化 EPLB state；V4.1 strict tool parameters 的 XGrammar 约束；Responses text parts；Vision-Exp 专用 image sentinel padding。上述是 release notes/runtime implementation evidence，不是 DeepSeek 独立 benchmark 或本机运行结果。
- [PyPI `vllm/0.30.0` metadata](https://pypi.org/pypi/vllm/0.30.0/json) 快照为 `13,218` bytes / SHA-256 `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`；x86_64 wheel `314,883,777` bytes / `ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`，aarch64 wheel `309,984,160` bytes / `eb3e11bab695d085098579a6eda2d602419adec3826ebfbcce9a3ffa543eb62e`，sdist `42,432,229` bytes / `5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b`。
- 当前状态更新为：**内容专题 + HF reference implementation + vLLM `v0.30.0` stable release/source evidence + SGLang main + recipe protocol evidence（AA 单榜）**。完整权重、GPU/ROCm/NPU 运行、真实 FP4 质量、candidate/index Top-K recall、DSpark acceptance/rollback、EPD、tool acceptance、目标硬件 profiling、独立 benchmark 和生产 SLO 继续保持 `unverified`；goal 保持 active。
- 已同步研究笔记、来源索引、模型清单、榜单解释、`plan_v2.md`、第二十一册第 81 章，以及 `PAPERS.md`、`BOOK_SERIES.md`、`ROADMAP.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`；没有新增排行榜之外的模型候选。

## 2026-09-23：排行榜复访失败与 vLLM release 离线审计

- 本轮重新访问两个唯一排行榜时，三条代理均无法建立连接：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 curl connection failure。没有把该结果解释为榜单不存在，也没有据此新增、删除或切换模型候选；恢复后仍需重新获取两个页面并记录快照哈希。
- 利用已保存的 vLLM `v0.30.0` release API 离线复核：`tag_name=v0.30.0`、`target_commitish=main`、`published_at=2026-09-22T05:20:54Z`，API 快照 `65,334` bytes / `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；release API 列出 9 个平台 artifact，包含 CUDA 12.9 x86_64 wheel `545,459,905` bytes / `e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc` 与 aarch64 wheel `519,981,036` bytes / `fdb57ab5fa1c3ac4c94a6eff52579df32cc9aab5da9863880d03e7167e088e77`。
- 离线源码审计确认 registry 中 `DeepseekV41ForCausalLM` 位于第 379--381 行，`DSparkV41DraftModel` 位于第 647--648 行；NVIDIA/ROCm vision wrapper 都声明 V4.1 target 类，DSpark 文件都包含 `mtp.*` 映射。该结果强化 stable source evidence，但仍不产生完整权重、数值正确性、acceptance 或生产 SLO 证据。
- release body 的周边面试点已同步到研究笔记和来源索引：Fast Start per-GPU weight-cache/CUDA IPC、HiSparse host-resident sparse-MLA tier、Model Runner V2 overlap，以及 online adaptive verification。它们归因于 vLLM runtime release，不作为 V4.1 独有模型技术或新排行榜模型。

## 2026-09-23 Claude Opus 5.5 新锚点与联网恢复

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮先复验三条代理：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098` 访问百度均 HTTP 200。随后三条代理对 Artificial Analysis 中文首页和 DataCurve DeepSWE 均返回 HTTP 200 且逐字节一致。AA 首页为 `1,783,001` bytes / SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`；DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。此前短时连接失败只记录为代理路径故障，不再作为当前状态。
- AA 当前前列新增/展示了重点厂商 canonical 条目 `Claude Opus 5.5`，本轮选择 `claude-opus-5-5` 作为下一活动锚点；`Grok 4.7`、`GPT-6 Sol` 等也在页面中出现，但本轮不并行展开。没有把 effort/provider 变体重复计作基础模型。
- Artificial Analysis Opus 5.5 详情三代理逐字节一致：`3,824,824` bytes / SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`。字段为 `releaseDate=2026-09-22`、1M context、Intelligence Index `57.6223698102963`、`parameters=null`、proprietary、input/output `$4/$20`、cache hit `$0.20`；这些是 AA 第三方/provider 字段。
- DataCurve 当前没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，因此不迁移 Opus 5、Fable 5.1 或其他 Claude 版本的 Pass@1、成本、输出 token、steps 和 Agent 结果。
- Anthropic 官方发布页 canonical URL 为 [`https://www.anthropic.com/claude-opus-5-5`](https://www.anthropic.com/claude-opus-5-5)，快照 `619,938` bytes / SHA-256 `1b4b50a9df4c7f24d786b811c88f372ef066e4cc71ed7d43a5a1b3509136f293`。官方正文确认：Opus 5.5 是 Claude 5.5 家族首个模型；发布方称典型成本比 Opus 5 低 40%，cache read 为 `$0.20/M`，强调更少 token/steps、agentic coding、computer use、knowledge work 和更快输出。
- 新知识已归纳为五条面试主线：`quality / successful_task_cost` 需要同时看 token、工具轮次、fallback 与 verifier；effort/max/medium 与 benchmark 成本必须分账；长任务 coding 的关键是 context gather、完整 patch、回归测试和 artifact verifier；Cyber/Life Sciences/Distillation 使用 capability routing 与 fallback；WANDR/OSWorld/Terminal/Automation 等是模型+工具+harness 结果，不是裸模型分数。
- 官方发布页还称 Opus 5.5 使用接近 Fable 5.1 的网络安全、生物和蒸馏 safeguards，部分 cyber 任务 fallback 到 Opus 4.8，biology/frontier LLM development 任务可能由 Opus 5 完成；preserved thinking 被称为 anti-distillation safeguard。必须记录实际执行模型、触发类别、权限、缓存、重试和最终 artifact，不能把 fallback 成功归因给 Opus 5.5。
- System Card PDF 已取得：`17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`；PDF 标题确认是 Claude Opus 5.5 System Card，目录含训练过程、能力、生物/化学、网络安全、自治、Agent 安全、对齐、模型福利和白盒分析章节。当前环境没有 PDF 文本抽取器，不把目录或二进制字符串升级为具体安全数字。
- Platform model page 在 1234/8098 仍重定向到区域不可用页，旧快照为 `439,341` bytes / SHA-256 `e89f00b6cdf35d049751c086b15a98846e40ba4e6bce18a12777020d3395d577`；7890 成功取得真实 Markdown `3,184` bytes / SHA-256 `aa9389d5f328023dea11650d26898b43b4e04969a4facd327881df145a307927`。这是线路差异，不代表模型不存在。
- 7890 同时取得 What's new `5,972` bytes / `90fb6efa547a193cbf1eb4b836ef5310234da054f2e83abbe15ce41b0d4c5a6a`、Migration guide `4,740` bytes / `0c8f717ab25184446431b1c579905a2458e4917258cfb71d66b65737eb3feab5` 和 Fast mode `6,962` bytes / `c23e9562b6c76c530dd95ca51578a360bbf0304166839fef2f8ec38db746e913`。新核验点为 always-on thinking、forced tool choice 400、thinking block model/conversation binding、`computer_toolset_20260801`、progress-update blocks、on-demand compaction、inline tools、fast mode `usage.speed`/独立限流和 refusal/fallback 契约；这些是 API observable behavior，不升级为内部架构结论。
- 已更新 [`claude-opus-5.5-source-notes.md`](research/model-update-2026-09/claude-opus-5.5-source-notes.md)，并同步 `plan_v2.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md` 及 reasoning/Agent/serving 正式章节。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练/后训练 recipe、精确 DataCurve Agent 行、生产 kernel、目标硬件 profiling、独立 benchmark 和线上 acceptance 待核验；goal 保持 active。

## 2026-09-23：GPT-6 Sol 活动锚点与 OpenAI runtime 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，与用户指定的 `/home/zzc/llm-from-zero-to-interview` 等同；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 三条代理取得 Artificial Analysis 中文首页和 DataCurve DeepSWE 的当前快照；首页为 `1,783,001` bytes / SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`，DataCurve 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 前列展示 `GPT-6 Sol`，本轮从 `Claude Opus 5.5` 切换到 `gpt-6-sol`，没有把 effort/provider 变体重复计作基础模型。
- Artificial Analysis [`gpt-6-sol`](https://artificialanalysis.ai/models/gpt-6-sol) 三代理逐字节一致：`547,729` bytes / SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`。详情标题为 `GPT-6 Sol (max)`，max Intelligence Index `47.5276426437724`、median output speed `115.205383643174 tokens/s`、cost per Intelligence Index task `1.0564240894076389`；这些是 AA 第三方/provider 字段。
- DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，因此不迁移 GPT-6 Astra、GPT-5.6 或其他 GPT 版本的 Pass@1、成本、输出 token、steps 和 Agent 结果。
- OpenAI 官方模型页 [`gpt-6-sol`](https://developers.openai.com/api/docs/models/gpt-6-sol.md) 快照为 `1,664` bytes / SHA-256 `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`。已确认 `gpt-6-sol`、complex coding/agentic workflows、text/image input、text output、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-04-20 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch 端点和 Responses 工具目录。
- OpenAI [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) 快照为 `70,315` bytes / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；[Agents](https://developers.openai.com/api/docs/guides/agents.md) 为 `5,432` bytes / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；[Tools](https://developers.openai.com/api/docs/guides/tools.md) 为 `33,282` bytes / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；[Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 为 `14,272` bytes / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`。
- 新知识已分为四条面试主线：`reasoning.mode` 与 `reasoning.effort` 是独立控制面；GPT-6 family 的 `configuration_update` 可以在会话中调整后续 effort；Responses/Agents API/SDK 的状态和执行所有权不同；server-side/standalone compaction 返回 opaque/encrypted 状态，不能当可读摘要或永久记忆。模型页工具列表只证明能力入口，不证明内部架构或训练算法。
- 已新增 [`gpt-6-sol-source-notes.md`](research/model-update-2026-09/gpt-6-sol-source-notes.md)，并同步 `plan_v2.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、Reasoning/Coding Agent/Responses runtime/serving 正式章节、`BOOK_SERIES.md`、`ROADMAP.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练 recipe、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 待核验，goal 保持 active。

## 2026-09-23：GPT-6 Luna 当前活动锚点与 OpenAI sibling/runtime 核验

- 三条代理重新获取 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)，逐字节一致为 `1,782,868` bytes / SHA-256 `24182a9f96da6b6b96567bc2d069eddb553109a6bb323d07d79d900bb107770c`；首页前列出现 `gpt-6-luna`。三条代理对 [Artificial Analysis GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna) 逐字节一致，为 `3,992,084` bytes / SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`。详情记录 release date `2026-09-22`、max Intelligence Index `37.2559686869738`、median output speed `153.87508473888 tokens/s`；这些是 AA 第三方/provider 字段。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照仍为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，没有精确 `mini_swe_agent_gpt_6_luna_*` 行。因此不迁移 GPT-6 Sol、GPT-6 Astra、GPT-5.6 或其他 GPT 的 Pass@1、成本、输出 token 和 Agent steps。
- 7890 代理取得 [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)，`4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`；确认 `gpt-6-luna`、focused/high-volume 定位、text/image input、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-05-18 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 `$0.10/$0.50` token 价格。1234/8098 对该官方页返回 403，作为线路访问边界记录。
- Reasoning、Agents、Tools、Compaction 官方快照与上一轮 GPT-6 Sol 完全一致：分别为 `70,315`/`5,432`/`33,282`/`14,272` bytes，哈希不变。由此复用 GPT-6 family 的 mode/effort 分离、`configuration_update`、runtime ownership、tool capability surface 和 opaque compaction replay；不把通用文档写成 Luna 专属训练或架构披露。
- 已新增 [`gpt-6-luna-source-notes.md`](research/model-update-2026-09/gpt-6-luna-source-notes.md)，并同步 `plan_v2.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、第六/十六/十七/二十/二十四册相关章节、`BOOK_SERIES.md`、`ROADMAP.md`、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。当前状态：**AA 单榜资料级闭环**；参数、架构、完整训练 recipe、独立 benchmark、完整权重、目标硬件 profiling、tool acceptance 和生产 SLO 待核验，goal 保持 active。

- 本轮质量门禁已通过：`git diff --check` 通过；全库 Markdown fenced block 平衡；GPT-6 Luna 新增相对链接均存在；三个 AA 详情快照、OpenAI model page 和 DataCurve 快照哈希与研究笔记一致。下一轮仍先重新抓取 Artificial Analysis 与 DataCurve，再从八家重点厂商中选择未闭环或新出现的 canonical 条目；不会从 OpenAI 官方目录另发现模型，也不会迁移缺失的 Agent 分数。

## 2026-09-23：Claude Opus 5.5 System Card 正文解析与资料级闭环升级

- 工作目录仍为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi、DeepSeek、Gemini、Grok。
- 本轮复核三条代理的最新续抓快照：Artificial Analysis 中文首页 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`；此前同日 `1,783,001` bytes / `2fd4bd...` 作为历史快照保留。DataCurve DeepSWE `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，三代理逐字节一致。
- `Claude Opus 5.5` AA 详情仍为 `3,824,824` bytes / SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`，release `2026-09-22`、1M context、Intelligence Index `57.6223698102963`、约 `$4/$20` input/output；DataCurve 没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，不迁移 Opus 5、Fable 5.1 或其他 Claude 版本的 Agent 结果。
- System Card PDF 固定为 `17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`。此前“没有 PDF 文本抽取器、只保留目录”的记录已由本轮 Node.js `zlib` + ToUnicode/CMap 只读解析更正；原始 PDF 哈希不变。
- 正文新增训练数据边界：公开互联网、公开/私有数据、获许可用户数据和合成数据的组合；去重/分类与 ClaudeBot；不访问密码页、登录页或 CAPTCHA 页面；knowledge cutoff 为 2026 年 6 月。参数规模、层数、dense/MoE、优化器、完整训练/后训练 recipe 和 adaptive thinking 内部机制仍未公开。
- 正文新增能力证据：Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials；约 2.5% requests fallback，影响约 10% trials）、ProgramBench `91.2%`（166 golden tasks）、OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、最多 500 actions、5 runs、完整截图、>100K tokens 后 server-side compaction）。五 Agent team 在 ProgramBench 同分比较约 `2.7x` derived-latency improvement；DRACO 在 `0.5x` latency budget 下五 Agent team 约 `2.8x` speedup；100-agent task 运行 24 小时并出现 flat/hierarchical sub-lead 组织。
- 正文新增 RSP/安全证据：CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达；CoBench 2.1 Opus 5.5 `55.8%`、Mythos 5.1 `53.4%`、Opus 5 `53.2%`；AECI `169.36`，global CI `[165.23, 177.05]`、local CI `[167.99, 171.24]`，新 fit 为 374 benchmarks、7,985 observations、732 models。
- Cyber 结果明确关闭 safeguards：ExploitBench `13.99` flags、`91%` Cap%、完整 ACE `301/410 = 73.4%`；CyScenarioBench `67.6%`；ExploitGym 2h/6h `289/869`、`300/869`。Agent safety 为 Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 `0.1%/0.7%/1.0%`，约 18% fallback 到 Opus 4.8 且 1,310 个 fallback rollout 无成功攻击；coding Shade 无 safeguards/probe 为 `54.61%/11.13%`，computer-use probe `0.04%`，browser auto `0/110`。
- fallback 规则按类别分账：biology/CB -> Opus 5，cyber -> Opus 4.8，frontier LLM kernel 等窄范围 AI R&D -> Opus 5；conventional weapons/high-yield explosives 与 distillation/hidden reasoning extraction 无 fallback，其他 provider 可能不同。alignment 评测还包含 sandbox escape `1.5%`（均 low severity）、早期/最终 snapshot 的粘贴恶意指令 `52%` -> `2%`、不可见字符 `26%` -> `7.4%`，产品 mitigations 后为零；这些数字必须绑定 snapshot、safeguard、模拟环境和 verifier。
- 已同步 `claude-opus-5.5-source-notes.md`、`source-index.md`、`model-inventory.md`、`inventory-interpretation.md`、`PAPERS.md`、第七/八/十六/十七/二十/二十四册相关章节和 `INTERVIEW_BANK.md`。当前状态升级为 **AA 单榜 + System Card 正文证据**；仍不新增 Transformer 架构章节，DataCurve 精确 Agent 行、参数/架构、完整 recipe、独立复现、目标硬件 profiling、生产 kernel 和线上 acceptance 继续保持待核验，goal 保持 active。

## 2026-09-23：下一活动锚点切换为 DeepSeek V4.1-Flash

- 最新 Artificial Analysis 中文首页续抓为 `1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`，DataCurve DeepSWE 为 `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；三条代理内容一致。AA 前列已有 canonical `DeepSeek V4.1 Flash (max)`，第三方 Intelligence Index 为 `39.456167472527`；本轮没有新的八家重点厂商模型名称。
- Opus 5.5 已完成本轮 System Card 正文同步，因此活动锚点切换到已有的 `deepseek-v4-1-flash`，不是从 vLLM、SGLang、HF 或官方目录另发现模型。当前研究笔记已经覆盖 HF revision、技术资料、reference inference、vLLM `main`/`v0.30.0` stable release surface、SGLang source、DSpark/视觉边界和 recipe 协议。
- 下一轮的闭环门禁为：完整权重加载、FP4/FP8 量化误差、candidate/index Top-K recall、DSpark draft/target acceptance 与 rollback、EPD/TP/EP/DP 调度、视觉 token/cache 一致性、目标硬件 profile、tool/verifier acceptance 和生产 SLO。release registry、wheel、README 或榜单分数不能越级替代这些结果。
- DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 DeepSeek V4 Pro、V4 Flash、V3.2 或其他 DeepSeek 版本的 Agent 成绩。goal 保持 active，下一轮仍不新增重复 Transformer 架构章节。

## 2026-09-23：DeepSeek V4.1-Flash 官方 API contract 补证

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；本轮没有从 API 文档、HF、vLLM、SGLang 或 recipe 另发现模型，活动锚点仍是两个排行榜已确认的 `deepseek-v4-1-flash`。
- 通过 `10.237.126.170:1234` 取得并固定官方 Pricing、Rate Limit、Error Codes、Vision、Files、Responses API 和 Tool Calls 页面：大小/SHA-256 依次为 `23,149/2fecee48...a4506a`、`35,133/1190b37c...26413a`、`20,387/0df2698a...5094a`、`78,174/a805d8a...7dac7`、`61,828/1020efca...40345`、`56,250/1719ac1b...1e3ca2`、`70,636/5ee72ac...b4027`；完整哈希已写入研究笔记和来源索引。
- 当前可用于面试的 API 知识点已收口：`deepseek-flash` 对应 V4.1-Flash；1M context、384K max output、最高 2500 concurrency；Vision 的 URL 8192 字符/60 秒/32 MiB、`file_id` 64 MiB、请求体 48 MiB、600 图和约 1024 image tokens/图；Files 的 `purpose=user_data`、64 MiB、保存期、25 GiB/10,000 文件；Responses stateless semantic SSE、递增 `sequence_number`、completed/incomplete/failed 终态和不发送 `[DONE]`；图像 tool output 回灌；`/beta strict=true` schema 约束。
- 证据边界已明确：以上是 endpoint/provider contract，不是 V4.1 内部架构、完整权重、生产 kernel 或质量 benchmark；`function_call_output` 代表可回灌，不代表模型拥有权限；strict schema 代表结构校验，不代表业务 verifier 通过；`requested_model`、alias 和响应 `served_model` 必须分账。
- 已同步 [`deepseek-v4.1-flash-source-notes.md`](research/model-update-2026-09/deepseek-v4.1-flash-source-notes.md)、第二十一册第 81 章、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md) 和 [`plan_v2.md`](plan_v2.md)。下一步先做不收费的 parser/error/retry/idempotency toy，再按完整权重、目标硬件和 verifier 门禁推进；完整权重加载、FP4/FP8 误差、Top-K recall、DSpark acceptance/rollback、EPD、视觉 cache、tool acceptance、独立 benchmark 和生产 SLO仍为 `unverified`，goal 保持 active。

## 2026-09-23：DeepSeek V4.1-Flash API contract toy 完成

- 新增 [`deepseek_v41_api_contract_audit.py`](research/model-update-2026-09/code/deepseek_v41_api_contract_audit.py)。脚本只使用 Python 标准库和合成输入，不联网、不调用付费 API、不加载模型权重。
- 已验证 semantic SSE：把事件按 5 字符切块，仍得到 3 个有序事件、文本 `contract audit` 和 `response.completed`；序号乱序、终态后事件和非 JSON data 拒绝。
- 已验证工具四级账本：`schema_valid -> authorized -> executed -> verified`。一次预执行 transient failure 后第 2 次尝试成功；相同 idempotency key 的重复调用返回 duplicate receipt，side effect 计数仍为 1；额外字段触发 `SchemaError`，越权路径触发 `PermissionDenied`。
- 文本和 `input_image/file_id` 两种 `function_call_output` 结构均能生成 typed observation；`network_called=false`。这只升级为 **local protocol toy evidence**，不升级真实 endpoint、strict enforcement、模型质量、工具权限、完整权重、目标硬件或生产 SLO。
- 已同步研究笔记、第二十一册第 81 章 `81.13.9`、来源索引、模型清单、榜单解释和 `plan_v2.md`。下一步检查本机 full-weight/runtime/hardware 条件；不可运行时维持 `unverified`，并按两榜既有 canonical 条目切换下一周边锚点，goal 保持 active。

## 2026-09-23：Gemini 3.8 Flash Interactions state/signature 与 thinking budget 复验

- 本轮继续在 `/data/zzc/llm-from-zero-to-interview` 工作，该路径等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Google 官方目录另发现模型，活动锚点切换到已有排行榜 canonical `gemini-3-8-flash`。
- 通过 `10.24.27.134:7890` 获取 Google 官方五页快照：model page `103,745` bytes / `e609ede21f99462207a0552cde54c8a7f7462e23a28a7515fb774c84f8f16f2b`；Thinking `226,287` / `a443bebc66c224284ea8f18ab8f6576fc4aebeb9d3419113be722da4bf7feef8`；Interactions `105,650` / `ee250076b4cb66072a5e98951cc2c7d2003b0ae6fce3b5cce398821443b0a553`；Tool combination `124,824` / `038adc9426510e7f15075d966186d7689817a06fd3c3be1c52100a3f397991ff`；Thought signatures `87,723` / `6b269d05196fef00a5837fb9aa4e42d31034acd072f6bb6064fd91b6a6c89e8c`。1234/8098 对 Google 文档超时，作为线路边界记录；不把失败解释为页面不存在。
- Thinking 表精确确认 `gemini-3.8-flash` 默认 `On (medium)`，支持 `low/medium/high`，不支持 `minimal`；`max_output_tokens` 是 thought + visible output 的硬上限，思考阶段触顶返回 `incomplete` 或截断/空输出，已产生的 thought tokens 仍计费；`total_thought_tokens` 应与 visible output 分账。
- Interactions 页面精确确认：Interaction 是按时间排列的 execution steps；默认 `store=true`；付费层保留 55 天、免费层 1 天；付费项目可配置 7/14/28/55 天日志删除，也可以按 interaction ID delete；`store=false` 不能配 background execution 或后续 `previous_interaction_id`。`previous_interaction_id` 只保留 history，tools、system instruction、generation config 是 interaction-scoped；stateful/stateless 均支持 implicit caching；跨模型 continuation 必须检查输出模态兼容性。
- Thinking 与 Tool combination 两页对 signature 位置存在可见差异：前者把 Interactions signature 限定在 thought/built-in tool steps，后者将 Gemini 3+ tool call/result signatures 描述为 tool context circulation 的普遍字段。已在研究笔记中保留冲突，工程实现要求原样保存实际返回的 opaque fields，并以具体 endpoint/schema/SDK capability probe 为准；function call/result 的 `id` 对齐始终是硬门禁。
- Tool combination 还确认 built-in/custom tool 可组合，Search/Maps/URL/File Search 属于 server-side，Code Execution 有独立 server-side steps，Computer Use/custom function 属于 client-side；工具组合要求 `validated`，不支持 `auto`。宿主仍负责权限、执行、超时、幂等、沙箱、回灌和业务 verifier；工具字段不是模型内部架构证据。
- 新增 [`gemini_interactions_replay_demo.py`](research/model-update-2026-09/code/gemini_interactions_replay_demo.py)，无依赖、无网络、无真实 API/权重。运行通过：`ok=true`、15 个有序 toy events、stateful parent history、stateless signature preservation、tool id alignment、paid/free `55/1` retention、删除后拒绝 continuation、`network_called=false`。这是 **local protocol toy evidence**，不升级真实 endpoint、signature 验证、模型质量、billing、硬件或生产 SLO。
- 已同步 [`gemini-3.8-flash-source-notes.md`](research/model-update-2026-09/gemini-3.8-flash-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md) 和本进度表；书系/题库/练习/术语/项目/知识图谱的 Gemini 3.8 已补入 retention、parameter scope、signature conflict 和 toy replay 门禁。
- 当前状态仍为**资料级闭环**：没有独立 3.8 参数/架构/训练报告、真实 API capability probe、low/medium/high 独立消融、长上下文/cache billing、完整权重、目标硬件 profiling、独立复现或线上 acceptance；不新增重复 Gemini Transformer 正式章节。goal 保持 active。

## 2026-09-23：Kimi K3 当前时点复验与面试边界补证

- 工作目录继续为 `/data/zzc/llm-from-zero-to-interview`，等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Kimi 官方仓库、HF、vLLM 或 SGLang 另发现模型，活动锚点切换到已有排行榜 canonical `kimi-k3`。
- 三条代理对 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)、[Kimi K3 详情](https://artificialanalysis.ai/models/kimi-k3) 和 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 均成功且逐字节一致：首页 `1,783,605` bytes / SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`；K3 详情 `4,080,091` bytes / `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`；DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。八家重点厂商按 canonical slug 去重后没有新增模型。
- AA K3 `max` 当前记录 Intelligence Index `43.5938229518782`、median output speed `36.9982439081499 tokens/s`、cost per Intelligence Index task `2.0001323004425493`、1M context。DataCurve 精确行 `mini_swe_agent_kimi_k3_max` 为 `309/451`、Pass@1 `0.6851441241685144`、Pass@4 `0.8938053097345132`、平均成本 `$4.654682129933482`、平均输出 `81,499.84` tokens、平均 Agent steps `97.5876`、4 runs/113 tasks；已明确绑定 `mini-swe-agent + tools + environment + verifier`，不迁移为裸模型分数。
- K3 README 复验为 `45,004` bytes / SHA-256 `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2`，HF metadata 为 `9,436` bytes / `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`，revision 仍为 `f831ab66814297da540d832a5235f8e904f29d06`。固定 config、vLLM recipe、FlashKDA 证据未漂移；SGLang main 当前为 `171,130` bytes / `9af22b45f8d8f8a5931c3bd60310316090973fd5cf9626aabd2a618a9e4baa6d`，相较前一快照只有 import/type annotation 变化，没有实质 runtime 技术变化。
- 面试限制已补入研究笔记：约 `2.5x scaling efficiency` 是发布方声明而非独立复现；preserved thinking history 需要结构化原样回传；跨模型切换可能不稳定；excessive proactiveness 是官方限制；发布评测混用 Kimi Code、Claude Code、Codex、H20/H100 与 compaction 条件，不能拼接为统一排名。
- 已同步 [`kimi-k3-source-notes.md`](research/model-update-2026-09/kimi-k3-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、`plan_v2.md`、第 15/88 章、题库和练习；K3 manifest audit、研究代码编译、协议 toy、Markdown 围栏、相对链接和 `git diff --check` 已通过。当前状态为**内容专题闭环 + 当前时点榜单/官方 revision 复验**；完整权重、目标硬件 profile、hybrid cache recovery、tool/verifier acceptance 和生产 SLO 仍为 `unverified`，goal 保持 active。

## 2026-09-23 Claude Fable 5.1 System Card 正文解析与证据闭环升级

- 工作目录确认仍为 `/data/zzc/llm-from-zero-to-interview`，等同 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。模型发现入口仍严格限制为 Artificial Analysis 与 DataCurve DeepSWE，重点厂商仍为 OpenAI、Anthropic、Z.ai/GLM、Qwen、Kimi/Moonshot、DeepSeek、Google/Gemini 和 xAI/Grok。本轮没有从 Anthropic System Card、Research 页面或外部论文另发现模型。
- Fable 5.1 最新 Artificial Analysis 详情为 `4,006,915` bytes / SHA-256 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`；canonical `claude-fable-5-1`、`releaseDate=2026-09-01`、`deprecated=false`，Intelligence Index `53.3549259623252`、median output speed `65.4865856934115 tokens/s`、median TTFT `298.446428812s`、1M context、cost per Intelligence Index task `7.629706364004841`。这些是第三方 provider/configuration 观察值，不是裸模型能力或统一 API 延迟。
- 本轮重新取得 AA 中文首页 `1,783,605` bytes / SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`、Anthropic 发布页 `534,503` bytes / SHA-256 `610f2e5cf15100fdc85edf0c4bee750878cdbf2db3520606aed088f757bd33f2`、System Card PDF `16,397,488` bytes / SHA-256 `b0d59edc7a60eef32a879c13d713cce60c3fefd7e6b5183afdc8b835af3c8c39` 和 DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。DataCurve 仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，不迁移 Fable 5 的 `316/452` 或其他 Claude 的 Agent 结果。
- 新增 [`pdf_text_extract.js`](research/model-update-2026-09/code/pdf_text_extract.js)，使用 Node.js `zlib` 与 PDF ToUnicode/CMap 做只读文本抽取；正文 `/tmp/fable51-system-card-extracted-20260923.txt` 为 `376,181` bytes、SHA-256 `d61d0a99b770ae104150ceb2ee04e0ee02fc4a728d7521ee0319b3742ed091e5`。解析器检查通过，原始 PDF 未改写。
- System Card 正文确认 Fable 5.1/Mythos 5.1 共享相同模型权重，差异主要在 safeguards/访问计划；训练数据包括公开互联网、公共/私有数据和合成数据，使用 deduplication、classification 和 ClaudeBot，knowledge cutoff 为 2026-06。RSP 针对 Mythos 判为 CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达。参数规模、层数、dense/MoE、优化器和完整训练/后训练 recipe 仍未公开。
- 正文代表性能力数字已入研究笔记：Terminal-Bench 4.0 `55.8%`、Terminal-Bench-Science 0.1 `52.6%`、CursorBench `73.4%`、OSWorld partial/strict `77.9%/41.7%`、GDPval-AA v2 `1853`、AutomationBench `31.4%`、ProgramBench `87.6%`。这些是 Anthropic 发布方条件结果，必须绑定 snapshot、effort、tools、safeguards、fallback、任务环境和 verifier；不升级为本地复现或 Fable 裸模型综合分数。
- 安全证据已补入：Gray Swan IPI `k=1/10/15` 为 `0.1%/0.7%/1.0%`，较强 Shade coding attacker 无 probes/启用 probes 为 `56.87%/12.80%`，browser auto 为 `0/110`，无 safeguards browser-use 对照为 `2.64%`。System Card 还披露 rare permission-hook/classifier bypass、sandbox vulnerability、白盒内部分析、SHADE/LinuxArena covert behavior 与 CoT controllability；这些结果需保留 `actual_model`、`fallback_reason`、safeguard state 和 verifier，不能泛化成生产攻击率。
- 已同步 [`claude-fable-5.1-source-notes.md`](research/model-update-2026-09/claude-fable-5.1-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、[`plan_v2.md`](plan_v2.md)，并补充第四册百科、评测证据、安全 fallback、`INTERVIEW_BANK.md` 和 `EXERCISES.md` 的 System Card 条件。Fable 5.1 当前状态升级为 **AA + System Card 正文证据闭环**，不新增独立 Transformer 架构章节，goal 保持 active。
- 下一步门禁：把 System Card、AA、DataCurve 三类分数分账；设计 thinking block/forced tool/history edit/updates/provenance 的本地协议回放；继续保持真实 API capability probe、完整权重、目标硬件 profile、独立复现、tool/verifier acceptance 和生产 SLO 为 `unverified`。下一轮先复抓两榜，再从八家重点厂商已有 canonical 条目选择锚点。

## 2026-09-23 GLM-5.3：SAO 关联技术与长轨迹 RL 补证

- 工作目录确认：`/data/zzc/llm-from-zero-to-interview` 等同用户指定的 `/home/zzc/llm-from-zero-to-interview`；保留既有 dirty worktree。本轮没有从论文、官方目录、HF、vLLM 或 SGLang 另发现模型，活动锚点仍是两个排行榜已有的标准 `GLM-5.3`。
- 两榜当前快照已重新固定：Artificial Analysis 中文首页 `1,783,597` bytes / SHA-256 `ce87eece01c108877686e76d76f7e26d463416d0a6cd895e554e284549a286ad`；GLM-5.3 详情 `4,074,151` bytes / SHA-256 `805918eef4c32b3cdeaf482f316b3311016b0523aef4a24886700e02e906ca73`；DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。AA 当前记录 Intelligence Index `44.777392385614`、median output speed `57.0553482500823 tokens/s`、cost per Intelligence Index task `2.0056375150449584`、1M context；这些是第三方/provider 配置字段。
- DataCurve 精确行是 `model=glm-5-3`、`harness=mini-swe-agent`、`reasoning_effort=max`、`config=mini_swe_agent_glm_5_3_max`：`311/451`、Pass@1 `0.6895787139689579`、Pass@4 `0.8761061946902655`、平均成本 `$3.9933584893126386`、平均输出 `80435.60975609756` token、平均 `124.47228381374723` Agent steps。继续绑定工具、任务集、环境和 verifier，不写成裸模型能力。
- 新增官方 Z.ai Markdown：GLM-5.3 `23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`；GLM-5.2 `25,293` bytes / SHA-256 `eeca971119068b0619137ce9c9d3c9362131baf16fdab0c067eefe5802919023`。确认 GLM-5.3 沿用 GLM-5.2 基础模型，改进来自 post-training，并继承 `SAO with compaction`；没有公开 5.3 专属 compaction 数学/序列化实现。
- 新增 SAO 论文证据：[arXiv:2607.07508](https://arxiv.org/abs/2607.07508)，HTML `190,454` bytes / SHA-256 `953b8968fa30d5f579a9650cc17d95521515ccac6cc7408b2a45c8126651822d`，PDF `664,828` bytes / SHA-256 `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48`。已核验 single-rollout immediate update、rollout logprob + DIS、双侧 token clipping/masking、critic `K=2`、frozen-attention value model 和 Skip-Observation GAE。
- 归因边界：论文主干实验使用 Qwen3-30B-A3B，摘要称 SAO 部署到 GLM-5.2（750B-A40B）Agent RL pipeline；不能把 Qwen3 实验、GLM-5.2 结果、论文算法或 DataCurve/AA 结果迁移为 GLM-5.3 独有能力。当前状态升级为 **GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据**。
- 已同步：[`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md)、[`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、第二十一册第 86 章、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、[`inventory-interpretation.md`](research/model-update-2026-09/inventory-interpretation.md)、`PAPERS.md`、`INTERVIEW_BANK.md`、`EXERCISES.md`、`GLOSSARY_EN_ZH.md`、`PROJECTS.md` 和 `KNOWLEDGE_GRAPH.md`。
- 待核验门禁：GLM-5.3 专属 compaction 状态/序列化、完整 post-training recipe、完整权重加载、FP8/量化误差、index/evidence recall、MTP acceptance、目标硬件 profile、独立 benchmark、tool/verifier acceptance 和生产 SLO。goal 保持 `active`。

## 2026-09-23 SAO local protocol toy 完成

- 新增 [`sao_async_rl_toy.py`](research/model-update-2026-09/code/sao_async_rl_toy.py)，仅使用 Python 标准库和合成数据；不联网、不调用付费 API、不加载 GLM-5.2/5.3 权重，也不声称复现 SAO 论文 benchmark。
- 运行通过：group barrier 的合成等待总量为 `11`，single-rollout wait 为 `0`；ratio `0.1`/`5.0` 被 mask，`1.0`/`1.1` 保留；observation token 从 `1` 增至 `3` 时 Skip-Observation GAE 的 `a0` 保持 `1.0603`，普通 token GAE 首段从约 `1.04276` 变为 `0.99734`；critic proxy 的 `K=2` 最终平方误差 `0.0625`，`K=1` 为 `0.25`。
- 这只升级为 **local protocol toy evidence**：验证公开机制的合成状态机内部一致性，不升级真实 rollout engine、policy lag calibration、GLM-5.2/5.3 训练效果、产品 compaction、完整权重、目标硬件、verifier 或生产 SLO。
- 已同步 [`glm-5.2-source-notes.md`](research/model-update-2026-09/glm-5.2-source-notes.md)、[`glm-5.3-source-notes.md`](research/model-update-2026-09/glm-5.3-source-notes.md)、[`source-index.md`](research/model-update-2026-09/source-index.md)、[`model-inventory.md`](research/model-update-2026-09/model-inventory.md)、`PROJECTS.md`、`EXERCISES.md`、`plan_v2.md` 和本进度表。下一门禁仍是 5.3 专属 compaction/recipe 与 full-weight/runtime 条件，goal 保持 `active`。

## 2026-09-23 GLM-5.3 compaction 复核与本机门禁

- 本轮重新尝试三条代理：`10.237.126.170:1234`、`10.24.27.134:7890`、`10.24.27.134:8098`。它们访问 `https://docs.z.ai/guides/llm/glm-5.3`、`https://z.ai/blog/glm-5.3` 和 `https://www.baidu.com` 均在连接阶段失败，curl 返回 HTTP `000`。因此当前结果记录为代理不可达，不解释为网页不存在；网络恢复后仍需重新复抓两个排行榜和官方资料。
- 使用本机已保存的官方 Markdown `23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`、博客正文资源 `30,414` bytes / SHA-256 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3` 做离线复核：资料仍只说 GLM-5.3 沿用 GLM-5.2 基座、改进来自 post-training，并继承 `SAO with compaction`；没有状态 schema、序列化、切分阈值、压缩质量门禁或完整 5.3 recipe。
- 本机不存在 GLM 权重文件；`torch`、`transformers`、`vllm`、`sglang`、`safetensors` 不可导入，`nvidia-smi` 无法连接驱动。因此未进行完整权重加载、kernel、数值、硬件 profiling 或线上 acceptance，相关项目继续标记 `unverified`。
- 新增并运行 [`glm53_compaction_contract_audit.py`](research/model-update-2026-09/code/glm53_compaction_contract_audit.py)：完整候选的 schema round-trip、goal/plan、工具回执链、幂等键、待执行副作用、artifact digest、verifier 状态、预算和 cut marker 检查全部通过；故意丢字段的候选被拒绝。输出证据等级为 `local_protocol_toy`，不代表 Z.ai 内部实现、真实 GLM-5.3 或 SAO 论文 benchmark。
- 当前状态保持：**GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据 + stable/main runtime source evidence**；5.3 专属 compaction、完整 recipe、完整权重、目标硬件和 tool/verifier acceptance 仍待核验，goal 保持 `active`。
