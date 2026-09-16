# 榜单条目解释规则

核验日期：2026-09-15。

Artificial Analysis 与 DeepSWE 的页面同时列出模型、reasoning effort、非推理版本、Agent harness、供应商和不同快照。排行榜中的每一行不等于一个独立基础模型。本项目后续采用三层记录：

1. **基础模型/发布版本**：例如 `GPT-6 Astra`、`Kimi K3`、`GLM-5.3`。这一层讨论组织、公开日期、模态、上下文、开放性、模型卡与技术资料。
2. **推理配置**：例如 `max`、`high`、`xhigh`、`low` 或 `Non-reasoning`。这一层讨论推理预算、延迟、输出 token 和评测条件，不能写成不同模型架构。
3. **Agent 系统组合**：例如 Claude Code、Kimi Code CLI、Codex 或 mini-SWE-agent 与模型的组合。这一层讨论 harness、工具、沙箱、上下文管理、验证器和任务预算，不能把系统得分归因给模型本身。

## 当前重点候选的结构化解释

| 基础模型 | 页面上发现的配置 | 当前一手核验状态 |
|---|---|---|
| GPT-6 Astra | low/medium/high/xhigh/max/Non-reasoning | 已读取 OpenAI 官方模型页；训练机制与发布日期待核验 |
| GPT-5.6 Sol/Terra/Luna | 各自的 low/medium/high/xhigh/max/Non-reasoning；DeepSWE 的 Sol/Luna 为 max | 已读取三个 OpenAI 官方模型页及 Reasoning/Tools/Prompt Caching/Compaction 文档；运行时字段已核验，参数、架构、训练配方和独立技术报告待核验 |
| GPT-5.5 / GPT-5.5 Pro | GPT-5.5 的 low/medium/high/xhigh/Non-reasoning，Pro 的 xhigh；DeepSWE 的 GPT-5.5 为 xhigh | 已读取 OpenAI 两个官方模型页、GPT-5.5 专属指南和 Reasoning/Tools/Tool search/Prompt Caching/Compaction/Images/Conversation state/Background 文档；高效 reasoning、`phase`、图像 detail、缓存差异和 Pro 后台边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| GPT-5.4 / GPT-5.4 Pro | GPT-5.4 的 xhigh/low/Non-reasoning，Pro 的 xhigh；DeepSWE 的 GPT-5.4 为 xhigh | 已读取 GPT-5.4 官方模型页、专属指南和 Reasoning/Tools/Tool search/Compaction/Conversation state/Prompt caching/Agents 文档；1M context、deferred tool loading、computer use、custom tools/CFG、`allowed_tools`、`phase`、compaction 和状态回放已核验，参数、架构、训练配方和独立技术报告待核验 |
| Claude Fable 5 | Artificial Analysis 的 `max + Opus 4.8 Fallback`；DeepSWE 的 `xhigh + mini-swe-agent` | 已读取 Anthropic 官方模型页、发布/重新部署公告、Thinking/Effort、拒答/fallback、Fallback credit、Memory、Programmatic tool calling、Compaction、Context editing 和 Task budgets 文档；运行时与安全边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| Claude Opus 4.8 | Artificial Analysis 的 `max`；DeepSWE 的 `xhigh`/`max` + `mini-swe-agent` | 已读取 Anthropic 发布公告、System Card 入口和 Dynamic Workflows 博客；发布日期、effort、Messages API system entries、动态编排、并行 subagents、独立验证和断点恢复边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| Kimi K3 | low/max；Kimi Code 等 Agent 组合 | 已读取 Kimi 官方发布文章（含接入/价格与评测条件快照）；技术报告、权重交付状态待核验 |
| Kimi K2.7 Code | Artificial Analysis 的 `kimi-k2-7-code`；DeepSWE 的 `mini-swe-agent`、`reasoning_effort:null` | 已读取 Kimi 官方资源/API 文档、固定 revision 模型卡、配置、许可证和部署指南；1T/32B active MoE、MLA、MoonViT、native INT4、always-on thinking、preserve_thinking 和 tool-call 协议已核验，专属技术报告与独立 benchmark 待核验 |
| GLM-5.3 | max | 已读取 Z.ai 官方文档；SAO with compaction 原始定义、技术报告、权重细节待核验 |
| GLM-5.3-Flash | max；`mini_swe_agent_glm_5_3_flash_max` | 已读取 Z.ai 官方文档/博客、固定 revision 模型卡与配置；hybrid attention、IndexPool、mHC、视觉 coding、EPD 和 API 协议已核验，完整训练 recipe、生产 kernel、硬件 profiling 和独立 benchmark 待核验 |
| GLM-5 | Artificial Analysis `glm-5`/`glm-5-non-reasoning`；DataCurve 当前无 `mini_swe_agent_glm_5_*` | AA 单榜资料闭环；已读取 Z.ai 模型卡、GLM-5 专属技术报告、API/部署资料；DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering 已核验，完整 DSA/训练 recipe、生产 kernel、硬件 profiling 和独立复现待核验 |
| DeepSeek V4 系列 | Flash/Pro、Vision、不同推理档位和快照 | 已读取 DeepSeek 官方公告、模型卡与技术报告；不同快照行为和独立 benchmark 复现仍待核验 |
| DeepSeek V4 Flash Vision | Artificial Analysis `deepseek-v4-flash-vision` max；DataCurve 当前无该行 | 已读取 DeepSeek 实验发布页、Vision/Files/Responses/价格文档；历史实验 alias、当前 `deepseek-flash` -> V4.1-Flash 路由、图像预算和工具回灌已核验，视觉架构、专属训练报告和 DataCurve 结果待核验 |
| K2 Horizon MoVA 36B/A4B | MoVA 36B/A4B、K2 Horizon 7B/3.7B/0.9B | 36B/A4B 已由 IFM 官方模型卡、固定配置和实现核验；7B/0.9B 的官方卡片仅用于 Uno/MOPD 关联边界，3.7B 完整资料待核验 |
| DeepSeek-R1-0528 | 官方发布页；JSON/function calling；开放权重入口 | 已读取 DeepSeek API 发布页快照；模型卡、架构和独立 benchmark 仍待核验 |
| Claude Opus 5 系列 | adaptive reasoning 与多档 effort | 已读取 Anthropic 官方模型目录；内部架构与训练细节待核验 |
| Claude Fable 5.1 | Artificial Analysis `max/xhigh/high/medium/low + fallback`；长任务/多步研究定位 | 已复验 Artificial Analysis、Anthropic 发布页、模型页缓存和 System Card 入口；Fable/Mythos safeguards、cache read 定价、发布方 benchmark 与外部论文检索已补记；内部架构与独立复现待核验 |
| Claude Sonnet 5 | adaptive；快速/高吞吐定位 | 已读取 Anthropic 官方模型目录；内部架构与独立复现待核验 |
| Claude Sonnet 4.6 | adaptive/extended thinking；1M context beta；coding、computer use、long-context reasoning 和 Agent planning | 已读取 Anthropic 发布页、真实 Models Overview、System Card 入口和 Agent 文档；内部架构、adaptive 预算、compaction 格式和独立复现待核验 |
| Claude Haiku 4.5 | extended thinking；fastest；低成本高吞吐定位 | 已读取 Anthropic 官方模型目录快照；参数、训练架构与独立复现待核验 |
| Qwen3.8 系列 | 27B、2.4T A95B、Flash、Max及多档 effort | 已读取三份官方模型卡、Flash-Next 技术报告和 Qwen Cloud 页面；模型结构与接口字段已核验，独立 benchmark、完整训练配方、生产 kernel 和 hosted/open checkpoint 差异仍待核验 |
| Gemini 3.6 Flash | Artificial Analysis `high`；DataCurve `mini_swe_agent_gemini_3_6_flash_high`、`reasoning_effort: high` | 已读取 Google AI Developers 模型页、DeepMind Model Card、Thinking/Interactions/工具组合/视频/缓存/Computer Use 文档和 arXiv 9 篇外部使用论文；默认 medium、`minimal/low/medium/high`、agentic video、1M/64K、Model Card benchmark 与安全边界已核验，独立架构、参数、训练配方和专属技术报告待核验 |
| Gemini 3.8 Flash | low/medium/high；DeepSWE high | 已读取 Google 官方模型页、DeepMind Model Card、发布博客、评测 PDF 和 Gemini API 工具/推理文档；接口与 Agent 协议已核验，3.8 独立架构、参数、训练配方和独立复现待核验 |
| Grok 4.6 | Artificial Analysis `low/medium/high/xhigh`；DataCurve `mini_swe_agent_grok_4_6_low/medium/high/xhigh` | 已读取 xAI 发布公告、官方模型页、Reasoning/Compaction/Tools 文档和 arXiv 精确标题检索；2026-08-12 官方发布日期、500K context、四档 effort、模型生成数据/SFT 轨迹筛选、agentic RL、长任务 self-testing/verification 和 API 协议已核验；参数、架构、完整训练配方和独立报告待核验 |
| Grok 4.5 | Artificial Analysis `high`；DeepSWE `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high` | 已读取 xAI 发布公告、Grok 4.5 专属模型页、Reasoning/Responses/Compaction/Tools/Web Search/X Search/Remote MCP 文档；训练与评测只记录发布方描述，参数、架构、完整训练配方和独立技术报告待核验 |
| Mistral Small 4 | 119B/6.5B MoE；reasoning `none/high` | 已读取 Mistral 官方 Hugging Face 模型卡；训练配方与发布日期含义待核验 |
| Step 3.5 Flash | 196.81B/约 11B active；MTP-3；3:1 SWA/full attention | 已读取 StepFun 官方 Hugging Face 模型卡与技术报告入口；后端 MTP3 支持和完整训练细节待核验 |

排行榜发现的名称可能反映未来日期、实验条目、地区可用性或页面快照；正式结论必须回到发布组织的官方来源。对于没有官方来源的条目，只记录“候选发现”，不扩写模型架构、训练方法或发布日期。

## Artificial Analysis 快照字段边界

本地 `/tmp/artificialanalysis.html` 快照（2026-09-09 采集）还包含部分配置的 `releaseDate`、`isOpenWeights`、`openSourceCategorization`、`parameters` 和 Artificial Analysis Intelligence Index。它们是第三方页面字段：`releaseDate` 不是发布组织公告日期，`parameters` 可能是估算或目录归档值，指数也不是统一基础模型能力证明。本项目只把这些字段用于候选发现和复现实验索引，不将其升级为官方参数、许可证或训练事实。

快照中可见的示例（均为榜单配置，不是独立基础模型）包括：Claude Fable 5.1 max with fallback，指数约 53.37；GPT-6 Astra max，约 52.81；Claude Opus 5 max，约 50.70；GLM-5.3 max，约 44.86；Grok 4.6 high，约 44.41；Kimi K3 max，约 43.78；Gemini 3.8 Flash high，约 41.19。引用时必须同时写明快照日期、effort/fallback、任务协议和第三方来源，不能把点估计排序当成跨 harness 的结论。

同一快照把 GLM-5.3 max 的 `parameters` 字段写为 753、Kimi K3 max 写为 2800，并分别标为 `commercial-license`；这些值和分类只代表 Artificial Analysis 的第三方目录字段，不是模型卡、许可证正文或参数规模的独立核验，正式章节仍按“待核验”处理。

## 2026-09-14 实时快照的增量

独立快照见 [`artificial-analysis-2026-09-14-snapshot.md`](artificial-analysis-2026-09-14-snapshot.md)。本轮页面大小为 1,771,961 bytes，SHA-256 为 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`，结构化数据约有 702 个配置条目、673 个唯一名称，最新日期字段为 `2026-09-11`。

与 2026-09-09 快照比较，页面结构新增 13 个 slug/别名：DeepSeek V4.1 Flash、Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon MoVA 36B A4B、K2 Horizon 7B、K2 Horizon 3.7B、K2 Horizon 0.9B 的成对配置，以及 `mbzuai` 目录项和 DeepSeek V4 Flash 的 Non-reasoning 配置。成对 slug 不应被计为两个基础模型，`mbzuai` 也不是模型名称。

本轮 DeepSeek V4.1-Flash 与 K2 Horizon MoVA 36B/A4B 已获得相应官方资料支持并进入正式专题；Agnes、Ling 和 K2 Horizon 3.7B 仍停留在发现层，K2 7B/0.9B 只在关联资料边界内引用官方卡片。榜单的 `releaseDate`、参数、开放性和指数继续不能替代官方证据。

K2 Horizon 的 36B/A4B 条目已从“仅榜单发现”升级为“榜单发现 + 官方模型卡/配置/实现核验”。Artificial Analysis 的 `2026-09-03` 仍只是第三方榜单日期；正式架构结论绑定 IFM 模型卡和固定 revision。K2 7B-Uno 通过官方 7B/adapter 资料与论文进入“锚点周边技术”记录，不作为独立排行榜候选；0.9B 卡片披露的 MOPD 只说明同系列另一条训练流程，不能转写为 36B 事实。

## GLM-5 当前核验结果

Artificial Analysis 的 `glm-5` 是 GLM-5 的精确榜单条目；DataCurve 当前快照只有 GLM-5.2、GLM-5.3 和 GLM-5.3-Flash 配置，没有精确的 GLM-5 行。故 GLM-5 的状态是 AA 单榜资料闭环：可以沿 Z.ai 官方模型卡和专属技术报告提取 DSA、MoE、`slime` 和 Agentic Engineering，但不能记录或迁移相邻版本的 DeepSWE Pass@1、成本或 Agent steps。

GLM-5 模型卡/配置公开 744B total、40B active、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048` 和约 202K position；技术报告与模型卡还描述 28.5T 预训练 tokens、DeepSeek Sparse Attention 和异步 RL 基础设施。`total/active`、indexer top-k、rollout/trainer 解耦和 Agent harness 都属于不同证据层，不能合并为单一“模型分数”或完整内部训练机制。具体快照和待核验项见 [`glm-5-source-notes.md`](glm-5-source-notes.md)。

## GLM-5.3-Flash 当前核验结果

Artificial Analysis 的 `glm-5-3-flash` 主配置为 `max`，页面 `releaseDate` 字段为 `2026-08-26`；本次快照记录 Intelligence Index `41.907366113455`、约 `114.22108687545 tokens/s`、约 `2.45458272199994s` TTFT 和 1M context。DataCurve v1.1 记录 `mini_swe_agent_glm_5_3_flash_max`，284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、约 `$0.2409818562`/task、约 `72829.77` output tokens、约 `122.89` Agent steps。前者是第三方配置字段，后者是 `mini-swe-agent` + 工具 + 任务集 + 环境 + verifier 的系统结果；两者都不能直接解释为裸模型能力。

Z.ai 官方文档、博客和固定 revision 模型卡公开了 320B total/18B activated、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts/top-8/1 shared、1M position、IndexPool（4 个 indexer key vectors 加权池化）和 mHC（`hc_mult=4`、20 次 Sinkhorn 字段）。视觉路径支持 video/image/text/file 输入，并将 self-visual judgment、test-time improvement、GUI/渲染反馈纳入 coding loop；serving 资料公开 SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split 和 EPD。

当前状态为“内容专题闭环”，正式落点为第二十一册第 84 章。不能把 `index_topk=2048` 直接解释为最终可见 token 数，不能把 18B active 当作 18B dense 的显存/延迟，也不能把官方 3.01×/4.44× attention/KV、约 3× serving 和 benchmark 数字写成独立复现。完整训练/后训练 recipe、生产 kernel、真实 state/KV/indexer bytes、目标硬件 profiling、线上 acceptance rate、API 错误/限流行为和独立 benchmark 仍待核验；详见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)。

## Qwen3.8 当前核验结果

Artificial Analysis 页面确认了 Qwen3.8 Max（2026-08-03）、2.4T-A95B（2026-08-12）、27B（2026-08-14）和 Flash-Next（2026-08-26）四类条目；DataCurve DeepSWE 当前快照只检出 Qwen3.8 Max。后续已读取 Qwen 官方模型卡、Flash-Next GitHub/技术报告和 Qwen Cloud 页面，因而将该系列从“仅榜单发现”升级为“内容专题闭环”。

已确认的公开字段包括：27B dense vision-language、64 层、3 GDN + 1 Gated Attention；A95B 的 2.4T total/95B active、92 层、512 experts 与强制 thinking；Flash-Next 的 125B/6B active、约 51B N-gram、4B MTP、GDN + QSA、四分支 GR 和 Muon/AdamW。Max 是官方说明基于 A95B 的 hosted version，不能把产品能力写成新的 open checkpoint。

Flash-Next 技术报告中的 QSA、GR、N-gram 和 Muon 机制、训练设置及 benchmark/速度数字已整理到 [`qwen3.8-source-notes.md`](qwen3.8-source-notes.md) 和第二十一册第 83 章。完整生产 kernel、目标硬件 profiling、线上接受率、全系列训练/后训练配方和独立 benchmark 仍待核验；报告数字均保留为发布方自报。

## Gemini 3.6 Flash 当前核验结果

Artificial Analysis 的 `Gemini 3.6 Flash (high)` 与 DataCurve 的 `mini_swe_agent_gemini_3_6_flash_high` 归并为一个基础模型；`high` 是运行配置，不是独立架构。Artificial Analysis 详情页第三方字段为 Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s`、1M context；DataCurve 为 211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095`、平均 116.73 steps。两套结果不能拼接为裸模型能力。

Google API 模型页确认 `gemini-3.6-flash` 的 text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、默认 medium 和 `minimal/low/medium/high` thinking；Interactions、tool context circulation、thought/tool signatures、agentic video、implicit caching 和 Computer Use Preview 由对应官方文档支持。Model Card 明确 3.6 基于 3.5 Flash，架构/训练数据/硬件/软件资料指向前代卡片；Model Card benchmark 与安全结果仅作为发布方证据。

arXiv 精确标题检索无结果，全文检索的 9 篇命中都是外部使用/评测论文。完整证据见 [`gemini-3.6-flash-source-notes.md`](gemini-3.6-flash-source-notes.md)。当前为“资料级闭环”，不新增独立 Gemini 3.6 架构章节。

## Gemini 3.5 Flash 当前核验结果

Artificial Analysis 的 `Gemini 3.5 Flash (high)`、`medium`、`minimal` 与 DataCurve 的 `mini_swe_agent_gemini_3_5_flash_high` 归并为一个基础模型；`high` 是运行配置，不是独立架构。Artificial Analysis 详情页第三方字段为 `releaseDate` `2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、TTFT `18.3631s`、1M context 和约 `$1.5625`/task；DataCurve 为 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均 105.30 Agent steps。两套结果不能拼接为裸模型能力；AA 当前 deprecated 标记只代表第三方目录状态。

Google API 模型页和 What's New 页面确认 `gemini-3.5-flash`、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、默认 `medium` 与 `minimal/low/medium/high` thinking、Batch/Flex/Priority、thought preservation、Interactions、implicit caching、combined tool use 和 Computer Use Preview。Computer Use 文档还明确 3.5 Flash 支持可选的 screenshot prompt-injection detection，默认关闭。当前视频文档的 agentic processing 支持列表列出 3.5 Flash-Lite、3.6、3.7、3.8，没有明确列出 3.5 Flash，因此不把 3.6 的 agentic video 结论迁移给它。

Model Card 将 architecture、training dataset、data processing、hardware 和 software 全部指向 Gemini 3 Flash Model Card；公开评测、安全和 Frontier Safety 结果只能作为 Google 发布方证据。arXiv 精确标题检索为 0 篇，全文检索返回 30 篇外部使用/评测论文，没有检出 Gemini 3.5 Flash 专属技术报告。完整来源和待核验项见 [`gemini-3.5-flash-source-notes.md`](gemini-3.5-flash-source-notes.md)。当前为“资料级闭环”，不新增独立 Gemini 3.5 架构章节。

## Gemini 3.8 Flash 当前核验结果

Artificial Analysis 的 2026-09-02 快照记录了 Gemini 3.8 Flash 的 low/medium/high 配置，DataCurve DeepSWE v1.1 的 2026-09-03 快照还记录了 high 配置。不同 effort 行归并为一个基础模型；DeepSWE 的 Pass@1、成本、输出 token 和 Agent steps 归因于模型配置与 `mini-swe-agent`/工具/verifier 的组合。

Google Gemini API 模型页将 `gemini-3.8-flash` 标为 GA/stable，公开了 1,048,576 输入 token、65,536 输出 token、文本/图像/视频/音频/PDF 输入、文本输出，以及 low/medium/high thinking 和多种内置工具。官方最新模型页还描述了长任务中的小步 reasoning、迭代工具调用和过程验证；这些是公开的系统行为描述，不是内部算法证明。

Gemini 3.8 Flash 的周边面试主线是 Interactions API 的 step/event 状态协议、thought summary/signature 的连续性、Search/URL Context/File Search/Code Execution/Computer Use 的工具闭环、Structured Outputs 的结构约束、1M 长上下文与 implicit caching 的成本账本。完整证据和待核验项见 [`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。目前标为“资料级闭环”，不新增 Gemini 专属正式章节。

## GPT-5.6 当前核验结果

Artificial Analysis 的 2026-07-09 快照记录了 Sol、Terra、Luna 的多档 effort；DataCurve DeepSWE v1.1 还记录了 Sol max（73% ±3%，约 $6.46、60K 输出 token、61 steps）和 Luna max（67% ±4%，约 $0.61、73K 输出 token、102 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分。

OpenAI 三个模型页确认了不同服务档位、`gpt-5.6` 到 Sol 的别名、1,050,000 context、922,000 maximum input、128,000 maximum output、文本/图像输入、文本输出和 `none`--`max` effort。Reasoning 文档进一步确认 Responses API 的 `standard/pro` mode、`reasoning.context=all_turns`、同家族 reasoning 复用及 function calling 时回传 reasoning items；Prompt caching 文档确认 1,024 token 最小前缀、显式断点、30m TTL、自动路由和 GPT-5.6 的读写费率差异。

当前将 GPT-5.6 标为“资料级闭环”：官方模型/API 文档、周边 Agent 文档、开发者博客和研究笔记齐备，但没有 GPT-5.6 专属参数、架构、训练 recipe、system card 或独立技术报告。`latest-model.md?model=gpt-5.6` 当前返回的 front matter/正文实际是 GPT-6 Astra，因此不作为 GPT-5.6 证据。详见 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## GPT-5.5 当前核验结果

Artificial Analysis 的 2026-04-23 历史表记录 GPT-5.5 的多档 effort 和 GPT-5.5 Pro `xhigh`；DataCurve DeepSWE v1.1 的 2026-09-03 快照记录 `gpt-5.5` `xhigh`（67% ±6%、约 $7.23、46K 输出 token、82 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分；GPT-5.5 Instant 的榜单条目暂不与 API `gpt-5.5` 合并。

OpenAI 模型页确认 `gpt-5.5` 的默认 snapshot 为 `gpt-5.5-2026-04-23`、1,050,000 context、128,000 max output、text/image input、text output、reasoning tokens、`none`--`xhigh` effort 和 Chat Completions/Responses/Batch；Pro 使用 `gpt-5.5-pro-2026-04-23`、只支持 Responses/Batch、默认 high、可用 `medium`--`xhigh`，可能需要 background mode。

GPT-5.5 官方指南的主要面试主线是：同 effort 下更少 reasoning tokens 的官方行为描述、outcome-first prompt、工具描述与 tool search、`text.verbosity` 与 reasoning 分离、`image_detail` 的 token/精度取舍、Responses assistant `phase` 的手工回放、compaction 的长期任务状态，以及与 GPT-5.6 不同的 prompt cache 断点/key/TTL/统计/写入计费规则。它们描述模型行为和运行时协议，不构成内部架构或训练机制证据。

当前将 GPT-5.5 基础模型与 Pro 标为“资料级闭环”：两个排行榜的锚点、官方模型页、专属指南、周边 API 文档和研究笔记齐备；但没有 GPT-5.5 专属参数、架构、训练 recipe、system card 或独立技术报告，因此暂不新增专属正式章节。详见 [`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。

## GPT-5.4 当前核验结果

Artificial Analysis 的 2026-03-05 历史条目记录 GPT-5.4 的 `xhigh`、`low` 和 `Non-reasoning` 配置，以及 GPT-5.4 Pro 的 `xhigh`；2026-03-17 还记录 mini/nano 变体。2026-09-15 详情页的主配置已标记 deprecated 并指向 GPT-5.5。DataCurve DeepSWE v1.1 的 2026-09-03 快照记录 `gpt-5.4` `xhigh`（51.7699% Pass@1、77.8761% Pass@4、约 `$5.6525`、约 71.4K 输出 token、70.47 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分。

OpenAI 官方模型页确认 `gpt-5.4-2026-03-05`、text/image 输入、text 输出、1,050,000 context、128,000 max output、2025-08-31 knowledge cutoff、`none`--`xhigh` effort、Chat Completions/Responses/Batch 和模型工具列表。专属指南提取的面试主线包括：deferred `tool_search`、built-in computer use、native compaction、custom tool freeform input、CFG、`allowed_tools`、tool preambles、assistant `phase`、Responses reasoning/state replay、1M/272K 长上下文账本、reasoning 与 verbosity 分离以及研究 citation contract。

当前将 GPT-5.4 基础模型标为“资料级闭环”，GPT-5.4 Pro 作为同家族关联服务档位记录，mini/nano 仍为未独立核验的关联变体。公开资料没有 GPT-5.4 专属参数规模、架构、训练 recipe、system card、公开权重或独立技术报告，因此不新增专属正式章节。详见 [`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md)。

## Claude Fable 5 当前核验结果

Artificial Analysis 的 `claude-fable-5` 条目是 `max + Opus 4.8 Fallback` 配置，页面 `releaseDate` 字段为 2026-06-09；DataCurve DeepSWE v1.1 另记录 `xhigh + mini-swe-agent`，可见结果为 316/452、Pass@1 约 70% ±3%。两个结果的 effort、fallback、harness、工具和 verifier 不同，不能合并为一个模型分数。

Anthropic 官方资料确认 `claude-fable-5` 为 Active (legacy)，1M context、128K max output、adaptive always-on thinking、默认 `high`、拒答和 fallback 机制，并公开了 memory、programmatic tool calling、compaction、context editing 与 task budgets 等长任务运行时协议。重新部署公告还记录了 cyber safety classifier 更新及向 Opus 4.8 的安全 fallback；这不是 Fable 5 的内部架构披露。

当前将 Claude Fable 5 标为“资料级闭环”：已有榜单锚点、官方来源和研究笔记，但没有参数规模、稠密/MoE 结构、训练配方、完整技术报告或独立 benchmark 复现。详见 [`claude-fable-5-source-notes.md`](claude-fable-5-source-notes.md)。

## Claude Opus 4.8 当前核验结果

Artificial Analysis 的 `claude-opus-4-8` 条目是 `max` 配置，页面 `releaseDate` 字段为 2026-05-28，当前详情页标记 deprecated 并指向 Opus 5；DataCurve DeepSWE v1.1 另记录 `xhigh` 与 `max` 配置。不同 effort、任务集、provider、harness、工具和 verifier 的结果不能合并为裸模型能力。

Anthropic 官方公告确认 Opus 4.8 于 2026-05-28 发布，默认 `high` effort，支持更高 effort 以及普通/fast mode；Dynamic Workflows 博客公开了动态生成编排脚本、数十至数百并行 subagents、独立检查、反驳式复核和长任务断点恢复的系统模式。发布公告还记录 Messages API 可在任务中插入 system entries，并强调不会破坏 prompt cache。这些是产品/运行时协议与公开工程描述，不等于模型内部架构或训练算法。

当前将 Claude Opus 4.8 标为“资料级闭环”：两个排行榜锚点、Anthropic 官方来源、研究笔记和配套同步均已具备；没有参数规模、稠密/MoE 结构、训练配方、完整技术报告或独立 benchmark 复现，因此不新增 Opus 4.8 专属架构章节。详见 [`claude-opus-4.8-source-notes.md`](claude-opus-4.8-source-notes.md)。

## Grok 4.6 当前核验结果

Artificial Analysis 的 `grok-4-6` 条目包含 `low/medium/high/xhigh` 四档配置，DataCurve DeepSWE v1.1 也包含四档 `mini_swe_agent_grok_4_6_*` 配置；不同 effort 行归并为一个 Grok 4.6 基础模型。DataCurve 的 Pass@1、成本、输出 token 和 Agent steps 绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不能作为裸模型能力。

xAI 发布公告的 JSON-LD 日期为 2026-08-12，公开描述更长的 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks、agentic RL 和长轨迹 self-testing/verification。官方模型页还列出 500K context、2026-02-01 knowledge cutoff、文本/图像输入、`low/medium/high/xhigh`、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution、`prompt_cache_key`、`x-grok-conv-id` 和 compaction 建议。

当前将 Grok 4.6 标为“资料级闭环”：两个排行榜锚点、xAI 发布页、官方模型/API 文档、研究笔记和论文精确标题负检索均已具备；没有参数规模、MoE/稠密结构、完整训练 recipe、专属技术报告、公开权重或独立 benchmark 复现，因此不新增 Grok 4.6 专属架构章节。详见 [`grok-4.6-source-notes.md`](grok-4.6-source-notes.md)。

## Grok 4.5 当前核验结果

Artificial Analysis 的 `grok-4-5` 条目是 `high` 配置，DataCurve DeepSWE v1.1 的对应配置为 `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high`。DataCurve 的 243/452、Pass@1/Pass@4、成本、输出 token 和 Agent steps 绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不能作为裸模型能力。

xAI 发布页将 Grok 4.5 定位为 coding、agentic tasks 和 knowledge work，并公开描述 deduplication、quality scoring、domain-focused selection、GB300 训练、数十万任务 RL、automated/model-based grading 和 highly asynchronous agentic rollouts。这些是发布方描述，不等于完整训练 recipe。官方模型页与通用 Reasoning 文档对 `xhigh` 支持存在差异：专属页列出 `xhigh`，通用文档称 Grok 4.5 按 `high` 处理，当前不把它写成独立推理档位。

官方 API 文档还提供 reasoning encrypted state、30 天状态化 Responses、Context Compaction、server-side built-in tools、parallel function calling、Web/X Search 和 Remote MCP `allowed_tools`。论文精确标题检索本轮未检出 Grok 4.5 专属论文或独立技术报告；完整证据和待核验项见 [`grok-4.5-source-notes.md`](grok-4.5-source-notes.md)。当前状态为“资料级闭环”，不新增独立架构章节。

## Kimi K2.7 Code 当前核验结果

Artificial Analysis 的 `kimi-k2-7-code` 与 DataCurve 的 `mini-swe-agent` 行已确认是同一基础模型的不同评测配置。DataCurve 的 138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155` 和约 149.1 steps 绑定 harness、工具和 verifier，不能当作裸模型分数。

官方模型卡和固定配置公开了 1T 总参数、32B active、61 层、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 路线。官方 API 文档还规定 always-on thinking、`preserve_thinking`、固定采样参数、`auto`/`none` tool choice 和多步调用中的 `reasoning_content` 回传。动态工具、Context Caching、Partial Mode 和断线重连保留为 API 周边，其中动态工具文档当前只支持 K3，不能直接归因给 K2.7。

当前状态为“资料级闭环”：已有两榜单锚点、官方资料、研究笔记和同步记录，但没有 K2.7 专属公开训练报告或独立 benchmark 复现。详见 [`kimi-k2.7-code-source-notes.md`](kimi-k2.7-code-source-notes.md)。

## DeepSWE v1.1 当前核验结果

DeepSWE 页面快照（2026-09-03 更新）显示 113 个长周期软件工程任务、91 个仓库、5 种语言，并说明所有模型使用 `mini-swe-agent`。页面主表的 Pass@1、平均成本、输出 token 和 Agent steps 是 Agent 系统组合结果；`gpt-6-astra`、`claude-opus-5`、`glm-5.3`、`kimi-k3`、`claude-sonnet-5` 等行不能直接当作基础模型能力排名。完整配置、revision、供应商、工具 schema、重试、硬件和独立复现仍待核验，详见 `deepswe-snapshot-notes.md`。

## Claude Opus 5 当前核验结果

Artificial Analysis 的 `claude-opus-5` 是 `max` 配置（并列出其他 effort），DataCurve DeepSWE v1.1 的 `claude-opus-5` 行使用 `mini-swe-agent`，max 为 327/444、Pass@1 约 73.65% ±4%、Pass@4 约 88.50%、平均成本约 `$11.84`、约 99 steps。它们是配置级 Agent 系统结果，不是基础模型分数。

Anthropic 官方模型目录和专属页列出 `claude-opus-5`、2026-07-24 发布字段、1M context、128K 普通/300K Batch 最大输出、adaptive thinking、默认 `high` effort、五档 effort、512-token 最小可缓存 prompt、平台和价格字段。官方完整 Markdown 文档进一步确认 thinking block 回放、tool/effort 中途变更、refusal/fallback、fallback credit、web fetch 不支持以及相关安全/缓存边界。

当前证据支持接口、运行时协议和发布方评测边界，不支持参数量、MoE/稠密结构、训练配方、后训练算法、adaptive thinking 内部机制或独立 benchmark 复现。arXiv 精确标题检索返回的两篇文章只是将 Opus 5 作为被测模型，没有 Opus 5 专属技术报告；详见 [`claude-opus-5-source-notes.md`](claude-opus-5-source-notes.md)。

## Claude Fable 5.1 当前核验结果

2026-09-15 重新联网后，Artificial Analysis 详情页仍确认 `claude-fable-5-1` 的 `max + Default Fallback`，Intelligence Index `53.3738`、输出速度约 `65.98 tokens/s`、TTFT 约 `212.01s`、成本约 `$7.63`/task；这些是第三方配置级字段。DataCurve DeepSWE v1.1 页面仍只有 `claude-fable-5` 的各 effort 行，没有 Fable 5.1，因此不移植 Fable 5 的 316/452 结果。

Anthropic 官方模型页缓存列出 `claude-fable-5-1`、2026-09-01 发布字段、1M context、128K 最大输出、adaptive always-on thinking、默认 high effort、平台、价格和 2026-06 知识/训练截止字段。新鲜发布页进一步确认 Fable 5.1 与 Mythos 5.1 共享 underlying model 但 safeguards 不同，Claude Code 默认 High、Claude Cowork/Claude.ai 默认 Medium，cache read 为 `$0.25/M`，并公开 EFS、anti-distillation、科学工作流和发布方 benchmark 摘要。当前模型页 URL 在本环境跳转至区域不可用页，所以缓存字段和当前发布页正文分开记录。

arXiv 标题精确检索返回 0 篇，全文检索返回 4 篇外部使用/评测论文；Anthropic Research 页面未检出 Fable 5.1 专属报告。当前仍不支持参数量、MoE/稠密结构、训练配方、独立技术报告或独立 benchmark 复现结论，详见 [`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。

## Claude Sonnet 5 当前核验结果

2026-09-15 重新联网后，Artificial Analysis 详情页确认 `claude-sonnet-5` 的 `max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；它们属于同一基础模型的配置，不是六个独立模型。主配置 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、1M context、输入/输出 `$2/$10/M` 和约 `$5.0912`/task。DataCurve DeepSWE v1.1 仍是 113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`；Sonnet 5 五档 Pass@1 为 max 53.846%、xhigh 49.667%、high 48.230%、medium 39.778%、low 30.512%，平均成本约 `$26.40/$11.89/$7.43/$4.08/$2.19`。这些是配置 + harness + 工具/环境 + verifier 的结果，不能写成裸模型分数。

Anthropic 官方目录/发布页确认 `claude-sonnet-5`、2026-06-30、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、Adaptive、默认 high、Fast、平台、价格和 2026-01 cutoff 字段。官方文档补充了面试关键行为：手动 `thinking.type: "enabled"` + `budget_tokens` 在 Sonnet 5 上返回 400；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/可见文本共享的硬上限；thinking blocks/signatures 要按异构 `content` 回传；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 构成长任务运行时。发布页/System Card 的 benchmark 数字另作为 Anthropic 自报记录，不与两个排行榜拼接。

截至 2026-09-15，arXiv 精确标题检索 `"Claude Sonnet 5"` 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告。当前状态为“资料级闭环”，但仍不支持参数量、MoE/稠密结构、完整训练/后训练配方、内部 adaptive-thinking 机制或独立 benchmark 复现；详见 [`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

## Claude Sonnet 4.6 当前核验结果

2026-09-16 重新复验确认，Claude Sonnet 4.6 已同时出现在 Artificial Analysis 与 DataCurve DeepSWE；AA 的 adaptive/max 配置和 DataCurve 的 `mini_swe_agent_claude_sonnet_4_6_high` 归并为同一基础模型，不按 effort 或 harness 重复计数。DataCurve 的 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、成本和 Agent steps 是 `mini-swe-agent` 系统结果，不能作为裸模型能力。

Anthropic 官方发布页和真实 Models Overview 确认 `claude-sonnet-4-6`、2026-02-17、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、computer use、web/code/memory/programmatic tool calling。当前没有 Sonnet 4.6 专属参数、内部架构、完整训练 recipe 或独立技术报告，因此状态为“资料级闭环”，不新增独立架构章节。

本次夜间中断复验中，三个代理均完整取得 Artificial Analysis 首页和 DataCurve，快照分别为 `1,773,553` 与 `268,313` bytes 且逐字节一致；`8098/1234` 的 Anthropic 模型目录被区域页拦截，`7890` 取得真实目录。该线路差异已记录为访问证据，不解释成模型不存在。

## 面试与教学上的关键区别

比较两个模型时，必须同时固定模型版本、推理档位、工具集、harness、任务集、硬件、超时和上下文压缩策略。否则“模型 A 分数更高”无法说明基础模型能力更高。正式章节应给出一个相同模型在 low 与 max 档位下的预算、延迟和成功率对照示例，再讲排行榜为何需要分层读取。
