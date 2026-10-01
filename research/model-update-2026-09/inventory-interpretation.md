# 榜单条目解释规则

## 2026-09-29 Claude Sonnet 5.5

Artificial Analysis 新增 `claude-sonnet-5-5` 与 effort 配置，归并为一个 canonical 模型锚点；DataCurve 无精确 Sonnet 5.5 Agent 行。Anthropic 官方材料补齐 `between_tools`、thinking block 三种绑定、工具迁移、三阶段 cyber safeguards 与 refusal-specific fallback。System Card 数字须绑定 harness、effort、fallback/actual model、资源和 verifier；不推导参数/架构/训练 recipe，也不迁移相邻 Claude 的 Agent 结果。状态：**AA 单榜专题闭环 + 官方 API/System Card 证据**，详见 [`claude-sonnet-5.5-source-notes.md`](claude-sonnet-5.5-source-notes.md)。

核验日期：2026-09-23。

Artificial Analysis 与 DeepSWE 的页面同时列出模型、reasoning effort、非推理版本、Agent harness、供应商和不同快照。排行榜中的每一行不等于一个独立基础模型。本项目后续采用三层记录：

1. **基础模型/发布版本**：例如 `GPT-6 Astra`、`Kimi K3`、`GLM-5.3`。这一层讨论组织、公开日期、模态、上下文、开放性、模型卡与技术资料。
2. **推理配置**：例如 `max`、`high`、`xhigh`、`low` 或 `Non-reasoning`。这一层讨论推理预算、延迟、输出 token 和评测条件，不能写成不同模型架构。
3. **Agent 系统组合**：例如 Claude Code、Kimi Code CLI、Codex 或 mini-SWE-agent 与模型的组合。这一层讨论 harness、工具、沙箱、上下文管理、验证器和任务预算，不能把系统得分归因给模型本身。

## 当前重点候选的结构化解释

| 基础模型 | 页面上发现的配置 | 当前一手核验状态 |
|---|---|---|
| GPT-6 Astra | low/medium/high/xhigh/max/Non-reasoning | 两榜当前时点已复验；已读取 OpenAI 官方模型页、专属模型指南、Reasoning/状态/缓存/压缩/工具搜索/异步工具/steering/misalignment 文档；运行时协议与 local toy 已核验，训练机制、官方发布日期和内部架构待核验 |
| GPT-5.6 Sol/Terra/Luna | 各自的 low/medium/high/xhigh/max/Non-reasoning；DeepSWE 的 Sol/Luna 为 max | 已读取三个 OpenAI 官方模型页及 Reasoning/Tools/Prompt Caching/Compaction 文档；运行时字段已核验，参数、架构、训练配方和独立技术报告待核验 |
| GPT-5.5 / GPT-5.5 Pro | GPT-5.5 的 low/medium/high/xhigh/Non-reasoning，Pro 的 xhigh；DeepSWE 的 GPT-5.5 为 xhigh | 已读取 OpenAI 两个官方模型页、GPT-5.5 专属指南和 Reasoning/Tools/Tool search/Prompt Caching/Compaction/Images/Conversation state/Background 文档；高效 reasoning、`phase`、图像 detail、缓存差异和 Pro 后台边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| GPT-5.4 / GPT-5.4 Pro | GPT-5.4 的 xhigh/low/Non-reasoning，Pro 的 xhigh；DeepSWE 的 GPT-5.4 为 xhigh | 已读取 GPT-5.4 官方模型页、专属指南和 Reasoning/Tools/Tool search/Compaction/Conversation state/Prompt caching/Agents 文档；1M context、deferred tool loading、computer use、custom tools/CFG、`allowed_tools`、`phase`、compaction 和状态回放已核验，参数、架构、训练配方和独立技术报告待核验 |
| GPT-5.4 mini / nano | Artificial Analysis 的 `gpt-5-4-mini`、`gpt-5-4-nano` 多档配置；DataCurve 只有 GPT-5.4 base 的 `mini_swe_agent_gpt_5_4_xhigh` | 已读取两个精确 sibling 模型页和 GPT-5.4 指南；400K/272K/128K、effort、任务定位、价格和逐模型工具清单已核验；mini/nano 没有精确 DataCurve Agent 行，不迁移 base 结果 |
| Claude Fable 5 | Artificial Analysis 的 `max + Opus 4.8 Fallback`；DeepSWE 的 `xhigh + mini-swe-agent` | 已读取 Anthropic 官方模型页、发布/重新部署公告、Thinking/Effort、拒答/fallback、Fallback credit、Memory、Programmatic tool calling、Compaction、Context editing 和 Task budgets 文档；运行时与安全边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| Claude Opus 4.8 | Artificial Analysis 的 `max`；DeepSWE 的 `xhigh`/`max` + `mini-swe-agent` | 已读取 Anthropic 发布公告、System Card 入口和 Dynamic Workflows 博客；发布日期、effort、Messages API system entries、动态编排、并行 subagents、独立验证和断点恢复边界已核验，参数、架构、训练配方和独立技术报告待核验 |
| Kimi K3 | low/max；Kimi Code 等 Agent 组合 | 内容专题闭环 + vLLM `v0.29.0`/`v0.30.0` stable source entry 和 main/recipe 增量；adaptive DSpark 与 cache-rebind 修复仍是 stable tag 之后的 main 证据，MI355X ROCm recipe 是 nightly/pre-release。完整权重、目标硬件 profiling、cache recovery、独立复现和线上 acceptance 未验收 |
| Kimi K2.7 Code | Artificial Analysis 的 `kimi-k2-7-code`；DeepSWE 的 `mini-swe-agent`、`reasoning_effort:null` | 双榜内容专题闭环，正式章节为第二十一册第 94 章；已读取 Kimi 官方资源/API 文档、固定 revision 模型卡、配置、许可证和部署指南。KTransformers 有 RAWINT4 发布方命令，但同提交通用支持矩阵未列 K2.7，兼容性未实测；K2.5 `0.7.0.post4` SFT recipe 的 RAWINT4 权重后端与 `bf16:true` 训练精度字段不外推为 K2.7 配方。Highspeed 是官方称为同一模型的服务路由变体，不另建候选；专属技术报告、训练 recipe、独立 benchmark 与生产验收待核验 |
| Kimi K2.6 | Artificial Analysis 的 `kimi-k2-6`/`kimi-k2-6-non-reasoning`；DataCurve 当前无精确 `mini_swe_agent_kimi_k2_6_*` | AA 单榜资料级闭环；已读取 Kimi K2.6 模型卡、固定配置、部署指南、Agent Swarm 博客、Thinking API 和 Vendor Verifier；K2.5 架构路线复用、1T/32B MoE、MLA、MoonViT、native INT4 和验收分层已核验，完整 recipe、生产 kernel、硬件 profiling 和独立复现待核验 |
| GLM-5.3 | max | 内容专题闭环 + stable/main runtime source evidence；已读取 Z.ai 官方文档/博客、SAO 原始论文、固定 config 与标准 DSA runtime 入口；5.3 专属 compaction、完整 recipe、权重加载、硬件和生产验收待核验 |
| GLM-5.3-Flash | max；`mini_swe_agent_glm_5_3_flash_max` | 已读取 Z.ai 官方文档/博客、固定 revision 模型卡与配置；hybrid attention、IndexPool、mHC、视觉 coding、EPD 和 API 协议已核验。vLLM `v0.30.0` stable 的 cache/KDA/MTP/multimodal 路径已与固定 main 逐文件对照，并记录 kernel/quant/indexer 差异。完整训练 recipe、目标硬件 profiling 和独立 benchmark 待核验 |
| GLM-5 | Artificial Analysis `glm-5`/`glm-5-non-reasoning`；DataCurve 当前无 `mini_swe_agent_glm_5_*` | AA 单榜资料闭环；已读取 Z.ai 模型卡、GLM-5 专属技术报告、API/部署资料；DSA、744B/40B MoE、`slime` 异步 RL 和 Agentic Engineering 已核验，完整 DSA/训练 recipe、生产 kernel、硬件 profiling 和独立复现待核验 |
| DeepSeek V4 系列 | Flash/Pro、Vision、不同推理档位和快照 | 已读取 DeepSeek 官方公告、模型卡与技术报告；补充 SGLang v0.5.20 的 V4 Flash SWA branch-point cache、CSA/HCA kernel 与 SM120 indexer/MoE source evidence；runtime PR benchmark 仍按硬件、baseline 与 workload 归因，不替代模型独立复现 |
| DeepSeek V4 Flash Vision | Artificial Analysis `deepseek-v4-flash-vision` max；DataCurve 当前无该行 | 已读取 DeepSeek 实验发布页、Vision/Files/Responses/价格文档；历史实验 alias、当前 `deepseek-flash` -> V4.1-Flash 路由、图像预算和工具回灌已核验，视觉架构、专属训练报告和 DataCurve 结果待核验 |
| K2 Horizon MoVA 36B/A4B | MoVA 36B/A4B、K2 Horizon 7B/3.7B/0.9B | 36B/A4B 已由 IFM 官方模型卡、固定配置和实现核验；7B/0.9B 的官方卡片仅用于 Uno/MOPD 关联边界，3.7B 完整资料待核验 |
| DeepSeek-R1-0528 | 官方发布页；JSON/function calling；开放权重入口 | 已读取 DeepSeek API 发布页快照；模型卡、架构和独立 benchmark 仍待核验 |
| Claude Opus 5 系列 | adaptive reasoning 与多档 effort | 已读取 Anthropic 官方模型目录；内部架构与训练细节待核验 |
| Claude Fable 5.1 | Artificial Analysis `max/xhigh/high/medium/low + fallback`；长任务/多步研究定位 | 已复验 Artificial Analysis、Anthropic 发布页、模型页缓存并解析 System Card 正文；Fable/Mythos 共享权重但 safeguards/访问计划不同，训练数据边界、CB-1/CB-2、autonomy、发布方 benchmark、安全与 fallback 条件已补记；内部架构、完整 recipe、独立复现和目标硬件验收待核验 |
| Claude Sonnet 5 | adaptive；快速/高吞吐定位 | 已读取 Anthropic 官方模型目录；内部架构与独立复现待核验 |
| Claude Sonnet 4.6 | adaptive/extended thinking；1M context beta；coding、computer use、long-context reasoning 和 Agent planning | 已读取 Anthropic 发布页、真实 Models Overview、System Card 入口和 Agent 文档；内部架构、adaptive 预算、compaction 格式和独立复现待核验 |
| Claude Haiku 4.5 | extended thinking；fastest；低成本高吞吐定位 | 已读取 Anthropic 官方模型目录快照；参数、训练架构与独立复现待核验 |
| Qwen3.8 系列 | 27B、2.4T A95B、Flash、Max/0902及多档 effort | 已读取三份官方模型卡、Flash-Next 技术报告、Qwen Cloud Max/0902 页面和运行时文档；模型结构与接口字段已核验，0902 作为 hosted revision 记录，独立 benchmark、完整训练配方、生产 kernel 和 hosted/open 差异仍待核验 |
| Qwen3-VL-235B-A22B | Artificial Analysis instruct/reasoning；DataCurve 当前无精确 `mini_swe_agent_qwen3_vl_*` | AA 单榜内容专题闭环；已读取 Qwen3-VL Technical Report、GitHub README、HF Instruct/Thinking 固定 revision/config 和 Transformers raw main；三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3、SAPO、Thinking with Images 和 tool-call reward 已核验，完整权重、生产 kernel、端到端 serving、GUI/tool acceptance 和独立 benchmark 待核验 |
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

## GLM-5.2 当前核验结果

GLM-5.2 同时出现在 Artificial Analysis 与 DataCurve DeepSWE，前者提供 `max/non-reasoning` 配置，后者提供 `high/max` 的 `mini-swe-agent` 配置。Z.ai 正确博客路由为 `https://z.ai/blog/glm-5.2`；正文资源补充了 IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 和 coding-agent anti-hack。它们分别落在注意力候选、解码加速、系统优化、长轨迹 RL 和 verifier 设计上，不能合并成“模型已公开完整训练配方”。正式专题为第二十一册第 86 章；SAO/compaction 原始定义、生产 kernel、硬件 profiling 和独立复现仍待核验。

## GLM-5 当前核验结果

Artificial Analysis 的 `glm-5` 是 GLM-5 的精确榜单条目；DataCurve 当前快照只有 GLM-5.2、GLM-5.3 和 GLM-5.3-Flash 配置，没有精确的 GLM-5 行。故 GLM-5 的状态是 AA 单榜资料闭环：可以沿 Z.ai 官方模型卡和专属技术报告提取 DSA、MoE、`slime` 和 Agentic Engineering，但不能记录或迁移相邻版本的 DeepSWE Pass@1、成本或 Agent steps。

GLM-5 模型卡/配置公开 744B total、40B active、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048` 和约 202K position；技术报告与模型卡还描述 28.5T 预训练 tokens、DeepSeek Sparse Attention 和异步 RL 基础设施。`total/active`、indexer top-k、rollout/trainer 解耦和 Agent harness 都属于不同证据层，不能合并为单一“模型分数”或完整内部训练机制。2026-09-20 AA 页面已提示更新模型 GLM-5.1，故 GLM-5 保留为历史榜单锚点；DataCurve 仍没有精确 GLM-5 行。正式专题已落到第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](../../book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)。具体快照和待核验项见 [`glm-5-source-notes.md`](glm-5-source-notes.md)。

## GLM-5.3-Flash 当前核验结果

Artificial Analysis 的 `glm-5-3-flash` 主配置为 `max`，页面 `releaseDate` 字段为 `2026-08-26`；本次快照记录 Intelligence Index `41.907366113455`、约 `114.22108687545 tokens/s`、约 `2.45458272199994s` TTFT 和 1M context。DataCurve v1.1 记录 `mini_swe_agent_glm_5_3_flash_max`，284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、约 `$0.2409818562`/task、约 `72829.77` output tokens、约 `122.89` Agent steps。前者是第三方配置字段，后者是 `mini-swe-agent` + 工具 + 任务集 + 环境 + verifier 的系统结果；两者都不能直接解释为裸模型能力。

Z.ai 官方文档、博客和固定 revision 模型卡公开了 320B total/18B activated、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts/top-8/1 shared、1M position、IndexPool（4 个 indexer key vectors 加权池化）和 mHC（`hc_mult=4`、20 次 Sinkhorn 字段）。视觉路径支持 video/image/text/file 输入，并将 self-visual judgment、test-time improvement、GUI/渲染反馈纳入 coding loop；serving 资料公开 SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split 和 EPD。

当前状态为“内容专题闭环 + SGLang stable/main、vLLM `v0.30.0` stable source 与固定 main 对照证据”，正式落点为第二十一册第 84 章。不能把 `index_topk=2048` 直接解释为最终可见 token 数，不能把 18B active 当作 18B dense 的显存/延迟，也不能把官方 3.01×/4.44× attention/KV、约 3× serving 和 benchmark 数字写成独立复现。vLLM stable 已确认 IndexerCache/TailCache、KDA、MTP 和 multimodal 路径；KDA/multimodal 同 blob，MTP 主体相同，attention/indexer 路径及 KPool stride handling 与固定 main 有具体差异，main 的 Quark 权重映射增量不属于 stable。完整训练/后训练 recipe、真实 state/KV/indexer bytes、目标硬件 profiling、线上 acceptance rate、API 错误/限流行为和独立 benchmark 仍待核验；详见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)。

### GLM-5.3-Flash 的 upstream/runtime 证据分层

证据链应写成：vLLM `v0.30.0` stable source -> 已核验的 cache/KDA/MTP/multimodal 文件 -> 与固定 main 的逐项差异 -> stable wheel 安装与完整权重 -> cache/state recovery、数值/recall、目标硬件 profile -> EPD、tool/verifier 与生产 SLO。目录、源码和 release artifact 不能跨级替代后续验证。

固定 stable tag、mutable main、recipe 和本地运行结果必须分栏：SGLang `v0.5.20` 的 `glm5_next.py` 证明 stable source entry；SGLang `main` 的新增 fusion/量化与测试门禁证明 upstream 正在演进；vLLM `main` 的 `Glm5NextIndexerCache`、`Glm5NextTailCache`、独立 KDA state 和 MTP top-k mapping 证明 runtime 实现细节；vLLM `v0.29.0` 的专属路径缺失是负证据，不能被 recipe 的版本门槛覆盖。任何一层都不能替代固定权重加载、目标硬件数值/性能、PD/EPD recovery、tool/verifier 和生产 SLO 验收。

## Qwen3.8 当前核验结果

Artificial Analysis 页面确认了 Qwen3.8 Max（2026-08-03）、2.4T-A95B（2026-08-12）、27B（2026-08-14）和 Flash-Next（2026-08-26）四类条目；DataCurve DeepSWE 当前快照只检出 Qwen3.8 Max。后续已读取 Qwen 官方模型卡、Flash-Next GitHub/技术报告和 Qwen Cloud 页面，因而将该系列从“仅榜单发现”升级为“内容专题闭环”。

已确认的公开字段包括：27B dense vision-language、64 层、3 GDN + 1 Gated Attention；A95B 的 2.4T total/95B active、92 层、512 experts 与强制 thinking；Flash-Next 的 125B/6B active、约 51B N-gram、4B MTP、GDN + QSA、四分支 GR 和 Muon/AdamW。Max 是官方说明基于 A95B 的 hosted version，不能把产品能力写成新的 open checkpoint。

Flash-Next 技术报告中的 QSA、GR、N-gram 和 Muon 机制、训练设置及 benchmark/速度数字已整理到 [`qwen3.8-source-notes.md`](qwen3.8-source-notes.md) 和第二十一册第 83 章。完整生产 kernel、目标硬件 profiling、线上接受率、全系列训练/后训练配方和独立 benchmark 仍待核验；报告数字均保留为发布方自报。

## Qwen3.8 Max (0902) 的解释

2026-09-21 的 Artificial Analysis 页面继续将 canonical `qwen3-8-max` 展示为 `Qwen3.8 Max (0902)`，release slug 为 `qwen3-8-max-0902`；旧的 0803 条目已标记 deprecated。当前详情快照为 `3,835,865` bytes、SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`，第三方 Intelligence Index `45.4152084980521`、约 `37.275289705095 tokens/s`、约 `$5.408509428374016`/task、约 `984K` context；这些不替代一方模型说明。Qwen Cloud 官方页给出 alias `qwen3.8-max-2026-09-02`，并将它描述为 `qwen3.8-max` 的 upgraded snapshot。因此，本项目将 0902 记作 hosted service revision，而不是新增基础模型或公开权重。

官方页/文档确认 1M context、991K 普通最大输入、983K thinking 最大输入、131K 最大输出、`reasoning_effort=low/medium/xhigh`（默认 `xhigh`）、与 `thinking_budget` 互斥、thinking 模式下 `tool_choice` 只能为 `auto`/`none`，以及 `MultiModalConversation` 和 explicit/implicit/session cache。2026-09-23 经 `7890` 复核的 Context Cache 正文进一步确认 `ephemeral` marker、最多 4 个且只取最后 4 个、20 content-block lookback、explicit/session 5 分钟 TTL/reset、implicit 无固定 TTL且不保证命中、account/model 隔离、`x-dashscope-session-cache: enable` + `previous_response_id` 和 `usage.input_tokens_details.cached_tokens`。2026-09-21 文档示例已从 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`；这属于 provider endpoint/adapter 变化，不是模型能力变化。产品页当前还显示 input/output/implicit cache 为 `$2/$6/$0.25`，explicit cache 创建/读取为 `$2.50/$0.17` 每百万 token；Context Cache 文档的典型相对口径是 explicit/session 创建 `125%`、命中 `10%`，implicit 创建 `100%`、命中 `20%`，二者按不同层级记录。coding、工程规模项目、长周期 autonomous development、多工具 Agent 和视觉理解属于产品升级描述；没有专属 0902 架构、训练或 kernel 证据时，不把这些描述扩写成内部机制。

Dynamic Rate Limiting 是 hosted provider 的服务配额机制：限流按 `account + model` 聚合，workspace 可以覆盖单模型 quota，TPM tier 按自然月消费调整，达到保证 TPM 后仍可能因平台余量继续服务。`qwen3.8-max-0902` 的保证 TPM tier 为 `1,500,000 / 1,500,000 / 1,500,000`。应在 serving manifest 中分开记录 guaranteed/observed TPM、429、`Retry-After`、queue latency、account、workspace 和 revision；不能把保证 TPM 当作模型吞吐、GPU capacity 或 Artificial Analysis/DataCurve 结果。

DataCurve 当前只有 `mini_swe_agent_qwen3_8_max_xhigh` 泛化行（258/449，Pass@1 `57.4610%`，Pass@4 `83.1858%`，约 `$3.7291`/task，约 `95,075` output tokens，约 `111.34` steps，4 runs），没有 `qwen3_8_max_0902` 精确行。该结果只能作为 Qwen3.8 Max 系列的 Agent 配置参考，不能迁移成 0902 revision 的独立成绩。完整证据见 [`qwen3.8-max-0902-source-notes.md`](qwen3.8-max-0902-source-notes.md)。

新增 [`qwen_max0902_cache_contract_audit.py`](code/qwen_max0902_cache_contract_audit.py) 只验证标准库 toy 中的请求 capability gate、cache marker/lookback/TTL、account/model 隔离、session lineage 和 `cached_tokens` 读取；其证据等级为 `local_protocol_toy`，不能替代真实 QwenCloud 命中率、结算、模型质量或生产 SLO。

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

Gemini 3.8 Flash 的周边面试主线是 Interactions API 的 step/event 状态协议、thought summary/signature 的连续性、Search/URL Context/File Search/Code Execution/Computer Use 的工具闭环、Structured Outputs 的结构约束、1M 长上下文与 implicit caching 的成本账本。该锚点现为“内容专题闭环”，正式专题落在[第二十册第 23 章](../../book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)；2026-09-24 固定 SDK schema 补证明确 `function_call.id` → `function_result.call_id`、custom function signature 未在 typed fields 声明、额外字段/未知 step 有 forward-compatible 保留路径，但 endpoint 实际字段行为仍待 probe。3.8 独立架构/训练资料仍待核验。完整证据和边界见 [`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。

2026-09-23 文档复验进一步明确了三层边界：

- `thinking_level`、`max_output_tokens`、`total_thought_tokens` 是可观察预算/usage 控制面；它们能支持 thinking 消融和成本账本，不能反推内部搜索、RL 或 Transformer 结构。
- `store=true`/`previous_interaction_id` 是服务端 history 管理；`store=false`、55/1 天 retention、delete 和 stateless full-history replay 是数据生命周期/状态协议，不能写成永久记忆或 GPU KV cache。tools、system instruction、generation config 是 interaction-scoped，不能假设跨 turn 继承。
- Thinking 页面与 Tool combination 页面对标准 function-call signature 的出现位置存在表述差异。固定 Python SDK typed schema 未声明 custom function signature，但 `extra="allow"` 允许额外字段；Interactions 的明确关联为 `function_call.id` → `function_result.call_id`。工程应原样保存实际响应的 opaque fields，真实 endpoint 行为仍待授权 probe。

2026-09-28 当前正式 API reference 与 Python SDK main commit 的 schema 都使用 `function_call.id` → `function_result.call_id`、不在 custom-function typed fields 声明 signature；但 Tool combination prose 仍使用 `function_response.id` 并称自定义 tool steps 有 signature，Thinking prose 的 thought signature “必有”也与 schema optional 标记冲突。应报告为“正式 schema 更支持窄口径，服务端实际响应未 probe”，不能写成 endpoint 已禁止/必返回字段。具体快照见研究笔记 §18。

本地 [`gemini_interactions_replay_demo.py`](code/gemini_interactions_replay_demo.py) 只证明合成协议状态机可审计：stateful/stateless、signature、id、SSE 终态、删除/保留期和 modality gate；不升级真实 Gemini endpoint、模型质量或生产 SLA。

## GPT-5.6 当前核验结果

Artificial Analysis 的 2026-07-09 快照记录了 Sol、Terra、Luna 的多档 effort；DataCurve DeepSWE v1.1 还记录了 Sol max（73% ±3%，约 $6.46、60K 输出 token、61 steps）和 Luna max（67% ±4%，约 $0.61、73K 输出 token、102 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分。

OpenAI 三个模型页确认了不同服务档位、`gpt-5.6` 到 Sol 的别名、1,050,000 context、922,000 maximum input、128,000 maximum output、文本/图像输入、文本输出和 `none`--`max` effort。Reasoning 文档进一步确认 Responses API 的 `standard/pro` mode、`reasoning.context=all_turns`、同家族 reasoning 复用及 function calling 时回传 reasoning items；Prompt caching 文档确认 1,024 token 最小前缀、显式断点、30m TTL、自动路由和 GPT-5.6 的读写费率差异。

当前将 GPT-5.6 标为“资料级闭环”：官方模型/API 文档、周边 Agent 文档、开发者博客和研究笔记齐备，但没有 GPT-5.6 专属参数、架构、训练 recipe、system card 或独立技术报告。`latest-model.md?model=gpt-5.6` 当前返回的 front matter/正文实际是 GPT-6 Astra，因此不作为 GPT-5.6 证据。详见 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## GPT-5.5 当前核验结果

Artificial Analysis 的 2026-04-23 历史表记录 GPT-5.5 的多档 effort 和 GPT-5.5 Pro `xhigh`；DataCurve DeepSWE v1.1 的 2026-09-03 快照记录 `gpt-5.5` `xhigh`（67% ±6%、约 $7.23、46K 输出 token、82 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分；GPT-5.5 Instant 的 May/June 条目已经重新核验为 AA 榜单配置，但仍不与 API `gpt-5.5` 合并。

OpenAI 模型页确认 `gpt-5.5` 的默认 snapshot 为 `gpt-5.5-2026-04-23`、1,050,000 context、128,000 max output、text/image input、text output、reasoning tokens、`none`--`xhigh` effort 和 Chat Completions/Responses/Batch；Pro 使用 `gpt-5.5-pro-2026-04-23`、只支持 Responses/Batch、默认 high、可用 `medium`--`xhigh`，可能需要 background mode。

GPT-5.5 官方指南的主要面试主线是：同 effort 下更少 reasoning tokens 的官方行为描述、outcome-first prompt、工具描述与 tool search、`text.verbosity` 与 reasoning 分离、`image_detail` 的 token/精度取舍、Responses assistant `phase` 的手工回放、compaction 的长期任务状态，以及与 GPT-5.6 不同的 prompt cache 断点/key/TTL/统计/写入计费规则。它们描述模型行为和运行时协议，不构成内部架构或训练机制证据。

当前将 GPT-5.5 基础模型与 Pro 标为“资料级闭环”：两个排行榜的锚点、官方模型页、专属指南、周边 API 文档和研究笔记齐备；但没有 GPT-5.5 专属参数、架构、训练 recipe、system card 或独立技术报告，因此暂不新增专属正式章节。详见 [`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。

### GPT-5.5 Instant 的身份边界

AA 的 `gpt-5-5-instant-05-26` 与 `gpt-5-5-instant-06-26` 是可追踪的榜单条目，June 版本的第三方 Index 为 `26.0135173401368`、当前约 123.8052 output tokens/s、400K context；约 130.9925 tokens/s 是历史采集值。May 版本的第三方 Index 为 `22.6863693000789`，且页面已标记 deprecated。两个页面都不能替代 OpenAI 的 model ID、snapshot 或技术报告。

本轮通过 `7890` 再次检查精确页面，仍返回 HTTP 404；当前 9-byte body 的 SHA-256 为 `e3ebaa16dd9d9b9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31`。官方可获取的页面是 `gpt-5.5`，其 `gpt-5.5-2026-04-23`、1.05M context、128K output 和 effort 字段只作为家族对照。DataCurve 当前没有 Instant 精确行，不能迁移 GPT-5.5 base 的 Agent 结果。GPT-5.5 Instant June 当前时点收口为“榜单级关联配置 + 官方身份负证据”，不再把它当作当前活动锚点，也不把缺失的官方 ID 补写成 `gpt-5.5`。

## GPT-5.4 当前核验结果

Artificial Analysis 的 2026-03-05 历史条目记录 GPT-5.4 的 `xhigh`、`low` 和 `Non-reasoning` 配置，以及 GPT-5.4 Pro 的 `xhigh`；2026-03-17 还记录 mini/nano 变体。2026-09-15 详情页的主配置已标记 deprecated 并指向 GPT-5.5。DataCurve DeepSWE v1.1 的 2026-09-03 快照记录 `gpt-5.4` `xhigh`（51.7699% Pass@1、77.8761% Pass@4、约 `$5.6525`、约 71.4K 输出 token、70.47 steps）。这些是配置 + `mini-swe-agent` + 工具/verifier 的组合结果，不能作为基础模型裸分。

OpenAI 官方模型页确认 `gpt-5.4-2026-03-05`、text/image 输入、text 输出、1,050,000 context、128,000 max output、2025-08-31 knowledge cutoff、`none`--`xhigh` effort、Chat Completions/Responses/Batch 和模型工具列表。专属指南提取的面试主线包括：deferred `tool_search`、built-in computer use、native compaction、custom tool freeform input、CFG、`allowed_tools`、tool preambles、assistant `phase`、Responses reasoning/state replay、1M/272K 长上下文账本、reasoning 与 verbosity 分离以及研究 citation contract。

当前将 GPT-5.4 基础模型与 mini/nano 标为“资料级闭环”，GPT-5.4 Pro 仍作为同家族关联服务档位记录。mini/nano 的 400K 窗口、精确 snapshot、定位、价格和工具矩阵不能套用 base 的 1.05M 窗口或工具列表；公开资料没有家族专属参数规模、架构、训练 recipe、system card、公开权重或独立技术报告，因此不新增专属正式章节。详见 [`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md) 与 [`gpt-5.4-mini-nano-source-notes.md`](gpt-5.4-mini-nano-source-notes.md)。

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

## Grok 4.6：2026-09-21 当前时点解释

当前 AA 详情页的 Intelligence Index、输出速度和 TTFT 与 9 月 15 日缓存快照不同，但这些是第三方榜单在不同采集时点的测量字段，不是模型 revision、训练配方或架构变化的证据。引用时必须同时保留采集日期、页面哈希、配置 `high`、provider/测量口径和 `releaseDate` 的第三方属性；不能用当前值覆盖历史值，也不能把两次测量拼成趋势。

DataCurve 快照哈希和四条 Grok 4.6 行本轮没有变化。Pass@1、Pass@4、成本、输出 token 和 Agent steps 仍然是 `mini-swe-agent` 在 113 个任务、4 runs、指定工具/环境/verifier 下的配置级结果；medium 的 Pass@1 高于 high 不构成基础模型或 effort 的因果结论。

本轮 xAI 官方页面三条代理均失败，因此旧官方网页快照只能作为历史证据。负面网络结果不能升级为页面不存在，也不产生新的技术结论；Grok 4.6 仍保持资料级闭环，不新增专属架构章节。

## Grok 4.5 当前核验结果

Artificial Analysis 的 `grok-4-5` 条目是 `high` 配置，DataCurve DeepSWE v1.1 的对应配置为 `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high`。DataCurve 的 243/452、Pass@1/Pass@4、成本、输出 token 和 Agent steps 绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不能作为裸模型能力。

xAI 发布页将 Grok 4.5 定位为 coding、agentic tasks 和 knowledge work，并公开描述 deduplication、quality scoring、domain-focused selection、GB300 训练、数十万任务 RL、automated/model-based grading 和 highly asynchronous agentic rollouts。这些是发布方描述，不等于完整训练 recipe。官方模型页与通用 Reasoning 文档对 `xhigh` 支持存在差异：专属页列出 `xhigh`，通用文档称 Grok 4.5 按 `high` 处理，当前不把它写成独立推理档位。

官方 API 文档还提供 reasoning encrypted state、30 天状态化 Responses、Context Compaction、server-side built-in tools、parallel function calling、Web/X Search 和 Remote MCP `allowed_tools`。论文精确标题检索本轮未检出 Grok 4.5 专属论文或独立技术报告；完整证据和待核验项见 [`grok-4.5-source-notes.md`](grok-4.5-source-notes.md)。当前状态为“资料级闭环”，不新增独立架构章节。

## Kimi K2.7 Code 当前核验结果

Artificial Analysis 的 `kimi-k2-7-code` 与 DataCurve 的 `mini-swe-agent` 行已确认是同一基础模型的不同评测配置。DataCurve 的 138/452、Pass@1 约 30.53%、Pass@4 约 61.06%、平均成本约 `$2.8155` 和约 149.1 steps 绑定 harness、工具和 verifier，不能当作裸模型分数。

官方模型卡和固定配置公开了 1T 总参数、32B active、61 层、384 routed experts、top-8、1 shared expert、MLA、256K/262,144 context、MoonViT 400M 和 native INT4 路线。官方 API 文档还规定 always-on thinking、`preserve_thinking`、固定采样参数、`auto`/`none` tool choice 和多步调用中的 `reasoning_content` 回传。动态工具、Context Caching、Partial Mode 和断线重连保留为 API 周边，其中动态工具文档当前只支持 K3，不能直接归因给 K2.7。

当前状态为“双榜内容专题闭环”：已有两榜单锚点、官方资料、研究笔记和同步记录，但没有 K2.7 专属公开训练报告或独立 benchmark 复现。KTransformers 的 RAWINT4 部署命令、通用支持矩阵、dual prefill 阈值与 K2.5 旧/新 SFT 文档必须按版本和型号分别解释，不代表 K2.7 已完成运行时兼容或训练验证。详见 [`kimi-k2.7-code-source-notes.md`](kimi-k2.7-code-source-notes.md) §9。

## DeepSWE v1.1 当前核验结果

DeepSWE 页面快照（2026-09-03 更新）显示 113 个长周期软件工程任务、91 个仓库、5 种语言，并说明所有模型使用 `mini-swe-agent`。页面主表的 Pass@1、平均成本、输出 token 和 Agent steps 是 Agent 系统组合结果；`gpt-6-astra`、`claude-opus-5`、`glm-5.3`、`kimi-k3`、`claude-sonnet-5` 等行不能直接当作基础模型能力排名。完整配置、revision、供应商、工具 schema、重试、硬件和独立复现仍待核验，详见 `deepswe-snapshot-notes.md`。

## 2026-09-23 GPT-6 Astra 当前时点解释

7890 复验的 AA 详情仍是同一 `gpt-6-astra` canonical release 的 `max` 配置，`parameters=null`、`proprietary`、1M context 和第三方 Intelligence Index/速度/成本字段没有被解释为参数或架构披露。DataCurve 当前同时列出 low/medium/high/xhigh/max 五个 `mini_swe_agent` 配置；它们是同一基础模型在不同 effort 下的 Agent 系统行，不是五个 checkpoint。

当前资料状态为**内容专题闭环 + 双榜当前时点复验 + local protocol toy**。9 月 23 日 latest-model guide 从 Astra 专属页面漂移为 GPT-6 family 迁移页面；其 effort、endpoint、function-calling 和 EU data residency 矩阵属于 family-level API contract，不能当作 Astra 内部机制。toy 的重复句柄、错误 `call_id`、pending 结果、重复 steering、断线 recovery、misalignment 阻断和副作用账本属于本地协议验证；它们不提升为真实 API capability probe、服务端实现、质量、延迟或生产安全证据。官方未公开的参数规模、dense/MoE、训练/后训练 recipe、system card、专属技术报告、生产 kernel、目标硬件 profile 和独立 benchmark 继续标为 `unverified`。

## Claude Opus 5 当前核验结果

Artificial Analysis 的 `claude-opus-5` 是 `max` 配置（并列出其他 effort），DataCurve DeepSWE v1.1 的 `claude-opus-5` 行使用 `mini-swe-agent`，max 为 327/444、Pass@1 约 73.65% ±4%、Pass@4 约 88.50%、平均成本约 `$11.84`、约 99 steps。它们是配置级 Agent 系统结果，不是基础模型分数。

Anthropic 官方模型目录和专属页列出 `claude-opus-5`、2026-07-24 发布字段、1M context、128K 普通/300K Batch 最大输出、adaptive thinking、默认 `high` effort、五档 effort、512-token 最小可缓存 prompt、平台和价格字段。官方完整 Markdown 文档进一步确认 thinking block 回放、tool/effort 中途变更、refusal/fallback、fallback credit、web fetch 不支持以及相关安全/缓存边界。

当前证据支持接口、运行时协议和发布方评测边界，不支持参数量、MoE/稠密结构、训练配方、后训练算法、adaptive thinking 内部机制或独立 benchmark 复现。arXiv 精确标题检索返回的两篇文章只是将 Opus 5 作为被测模型，没有 Opus 5 专属技术报告；详见 [`claude-opus-5-source-notes.md`](claude-opus-5-source-notes.md)。

2026-09-24 补读 System Card 正文后，Opus 5 的专题增加 Agentic Safety 与评测方法证据：card changelog 因 Cowork harness mismatch 而重跑基线；IPI/bug bounty/Shade/Cowork 分别有不同攻击集、预算、产品 safeguard 和分母，不能合成统一安全率。产品级双层防护将 probe 放在不可信 tool result 的输入侧、classifier 放在危险 tool call 的动作侧。系统卡将 Opus 5 评为 CB-1、未达 CB-2，并报告内部安全监测/行为审计；这些仍是 Anthropic 发布方结果，不是独立安全审计或生产普遍概率。当前内容专题覆盖已闭环，但模型参数、架构、完整训练 recipe 和独立复现仍未证实。

## Claude Fable 5.1 当前核验结果

2026-09-21 重新联网后，Artificial Analysis 详情页确认 `claude-fable-5-1` 的 `max + Default Fallback`，主配置 Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`；这些是第三方配置/provider 字段。DataCurve DeepSWE v1.1 页面仍只有 `claude-fable-5` 的各 effort 行，没有 Fable 5.1，因此不移植 Fable 5 的 `316/452` 结果。

Anthropic 模型页和发布页确认 `claude-fable-5-1`、2026-09-01、1M context、128K 最大输出、adaptive always-on thinking、默认 high effort、平台、价格和 2026-06 知识/训练截止字段。Fable 5.1 与 Mythos 5.1 共享 underlying model 但 safeguards 不同，Claude Code 默认 High、Claude Cowork/Claude.ai 默认 Medium，cache read 为 `$0.25/M`；这些是模型/产品/治理层证据，不是内部架构披露。

迁移时必须单独记录三个 breaking changes：forced tool use 可能返回错误；较早模型不能读取 Fable 5.1 thinking blocks；编辑历史 turn 会使 thinking blocks 失效。新增能力包括 per-message effort、turn-scoped system messages、`display: "updates"` 工具间进度事件、较低 cache read 价格和 content provenance。它们共同说明模型接口是带版本、状态和审计语义的协议，不能把 thinking block 当普通文本、把进度事件当工具成功或把 provenance 当事实正确性证明。

arXiv 标题精确检索返回 0 篇，全文检索返回 4 篇外部使用/评测论文；Anthropic Research 页面未检出 Fable 5.1 专属报告。当前仍不支持参数量、MoE/稠密结构、训练配方、独立技术报告或独立 benchmark 复现结论，详见 [`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。

## 2026-09-23 Claude Fable 5.1 状态协议解释

7890 复验没有产生新模型：AA Fable 详情仍是 `claude-fable-5-1`，DataCurve 没有精确 Fable 5.1 行。最新官方 Markdown 把模型迁移门禁具体化为：assistant prefill 400、adaptive thinking always-on、forced tool 400、方向性的 thinking block 兼容、prefix mismatch 的 reject/drop 分支，以及 30-day retention/ZDR 和 Priority Tier 部署边界。这里的 `drop_block` 是显式状态转换并需要 `input_transformations` 记录，不能当作静默成功。

本轮 [`claude_fable51_state_protocol_audit.py`](code/claude_fable51_state_protocol_audit.py) 的 toy 结果只证明宿主账本的内部一致性：进度事件不计作工具完成，结构化 tool result 必须带回 call/idempotency lineage，provenance 需要独立 artifact verifier，重复回放只产生一个副作用。它不能证明 Anthropic server-side schema、hidden reasoning、Fable 5.1 质量或生产 SLO。随后新增[第二十册第 21 章 21.29](../../book-20-agent-harness-runtime/chapters/21-persistent-workspace与状态边界.md)，将当前闭环升级为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**，而非架构或权重闭环。

## Claude Sonnet 5 当前核验结果

2026-09-15 重新联网后，Artificial Analysis 详情页确认 `claude-sonnet-5` 的 `max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；它们属于同一基础模型的配置，不是六个独立模型。主配置 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、1M context、输入/输出 `$2/$10/M` 和约 `$5.0912`/task。DataCurve DeepSWE v1.1 仍是 113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`；Sonnet 5 五档 Pass@1 为 max 53.846%、xhigh 49.667%、high 48.230%、medium 39.778%、low 30.512%，平均成本约 `$26.40/$11.89/$7.43/$4.08/$2.19`。这些是配置 + harness + 工具/环境 + verifier 的结果，不能写成裸模型分数。

Anthropic 官方目录/发布页确认 `claude-sonnet-5`、2026-06-30、1M context、128K 普通/300K Batch 最大输出、文本/图像输入、Adaptive、默认 high、Fast、平台、价格和 2026-01 cutoff 字段。官方文档补充了面试关键行为：手动 `thinking.type: "enabled"` + `budget_tokens` 在 Sonnet 5 上返回 400；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/可见文本共享的硬上限；thinking blocks/signatures 要按异构 `content` 回传；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 构成长任务运行时。发布页/System Card 的 benchmark 数字另作为 Anthropic 自报记录，不与两个排行榜拼接。

截至 2026-09-21，arXiv 精确标题检索 `"Claude Sonnet 5"` 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告。System Card 深读补齐了训练公开性、RSP、网络安全和 agentic safety 的发布方证据，但没有公开参数量、MoE/稠密结构、完整训练/后训练配方、内部 adaptive-thinking 机制或独立 benchmark 复现；详见 [`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

## Claude Sonnet 4.6 当前核验结果

2026-09-16 重新复验确认，Claude Sonnet 4.6 已同时出现在 Artificial Analysis 与 DataCurve DeepSWE；AA 的 adaptive/max 配置和 DataCurve 的 `mini_swe_agent_claude_sonnet_4_6_high` 归并为同一基础模型，不按 effort 或 harness 重复计数。DataCurve 的 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、成本和 Agent steps 是 `mini-swe-agent` 系统结果，不能作为裸模型能力。

Anthropic 官方发布页和真实 Models Overview 确认 `claude-sonnet-4-6`、2026-02-17、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、computer use、web/code/memory/programmatic tool calling。当前没有 Sonnet 4.6 专属参数、内部架构、完整训练 recipe 或独立技术报告，因此状态为“资料级闭环”，不新增独立架构章节。

本次夜间中断复验中，三个代理均完整取得 Artificial Analysis 首页和 DataCurve，快照分别为 `1,773,553` 与 `268,313` bytes 且逐字节一致；`8098/1234` 的 Anthropic 模型目录被区域页拦截，`7890` 取得真实目录。该线路差异已记录为访问证据，不解释成模型不存在。

## 面试与教学上的关键区别

比较两个模型时，必须同时固定模型版本、推理档位、工具集、harness、任务集、硬件、超时和上下文压缩策略。否则“模型 A 分数更高”无法说明基础模型能力更高。正式章节应给出一个相同模型在 low 与 max 档位下的预算、延迟和成功率对照示例，再讲排行榜为何需要分层读取。

## DeepSeek V3.2 当前核验结果

Artificial Analysis 当前有精确 `deepseek-v3-2` 条目，详情页呈现的是 `Non-reasoning` 配置；DataCurve 当前快照没有 `mini_swe_agent_deepseek_v3_2_*` 行。因此 V3.2 是 AA 单榜候选，不能迁移 DeepSeek V3.1/V4 的 DeepSWE 结果。

DeepSeek 官方模型卡把 V3.2 的新技术归为 DSA、可扩展 RL 和大规模 Agent 任务合成，并补充 `thinking with tools` 的 chat-template/encoding 改动。模型卡还明确 V3.2-Speciale 只面向深度推理、不支持 tool calling。`developer` role 只为 search-agent 场景保留，官方 API 不接受它；这些是协议约束，不是宿主权限控制。2026-09-20 重新取得 AA 详情页且两个代理逐字节一致；DataCurve 仍无精确行，因此 V3.2 保持 AA 单榜资料级闭环，并作为 GLM-5 之后的下一锚点。

## DeepSeek V4 Pro 当前核验结果

Artificial Analysis 的精确条目是 `deepseek-v4-pro`，标题为 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`；AA 详情记录 `2026-08-13`、1M context、约 1.6T/49B total/active 和 Intelligence Index `35.9967791278402`。DataCurve 有精确的 `mini_swe_agent_deepseek_v4_pro_max` 行：Pass@1 `62.831858%`、Pass@4 `88.495575%`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` steps。后一组数字只能解释为 max effort + mini-swe-agent + 工具/环境/verifier 的系统结果。

官方公告、Quick Start、Responses/Thinking/Tool Calls 文档、模型卡、配置和 `arXiv:2606.19348` 支持的面试主线是：

1. `low/high/max` 是同一 V4 Pro API 模型的请求级 reasoning effort；不能将它们当作三个 checkpoint，也不能跨模型直接比较档位名。
2. Responses API 是 stateless：没有 `previous_response_id`、`conversation`、`background` 或 `store`；function tools、`apply_patch` 和并行工具调用的边界属于协议/执行器系统，不能据此反推隐藏推理结构。
3. V4-Pro 的 1.6T/49B MoE、1M context、CSA/HCA、mHC、Muon、FP4/FP8 和 SFT+GRPO+on-policy distillation 形成一条“模型结构—训练—serving—后训练”的技术链，但每个数字仍要带官方模型卡、技术报告、配置或发布方证据标签。
4. 配置字段（61 层、384 routed/6 selected/1 shared、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024`、YaRN factor 16）描述公开 artifact 的实现接口，不等于完整训练 recipe、生产 kernel 或目标硬件性能。

因此 V4 Pro 当前标为**内容专题闭环（双榜锚点）**，但不新增重复 Transformer 章节；复用 CSA/HCA、mHC、MoE、低精度、后训练和 Agent serving 既有章节。独立 benchmark、生产 kernel、硬件 profiling、线上 tool acceptance、不同 API 快照行为和完整超参仍待核验。

本轮通过 7890 代理下载官方技术报告 PDF（907,086 bytes，SHA-256 `f6fda5753db7b106baa5eeb1286877f17e6a111354762f8aa53c7e6556498df7`），并用标准库完成正文抽取（252,018 bytes，SHA-256 `f082910a550666ad32c914db94056daa59a13b75a7ffd2bbd548695aa5cd043c`）。正文确认 DSA 的两阶段训练、2,048 KV top-k、约 2.1B/943.7B tokens、主 attention 复杂度、specialist distillation、GRPO 稳定化策略，以及 Agent 任务合成与 verifier 闭环；公式/图表仍因抽取失真保留待视觉复核。详细证据见 [`deepseek-v3.2-source-notes.md`](deepseek-v3.2-source-notes.md)。

## GPT-5.3 Codex 当前核验结果

GPT-5.3 Codex 当前只能在 Artificial Analysis 中作为精确候选锚点确认；DataCurve 快照没有精确的 `mini_swe_agent_gpt_5_3_codex_*` 行。因此：

1. `xhigh` 是 Artificial Analysis 的配置字段，不能计作独立基础模型或独立架构。
2. 不能把 GPT-5.5、GPT-5.6、GPT-5.4 或其他 Codex 配置的 DeepSWE Pass@1、成本、输出 token 和 Agent steps 迁移到 GPT-5.3 Codex。
3. AA 的 Intelligence Index `32.5028174368983`、400K context 和 `2026-02-05` release date 字段均是第三方页面字段；不能替代 OpenAI 官方模型页或推断参数/训练方法。
4. OpenAI 官方资料确认的对象是模型/API/harness contract：`gpt-5.3-codex`、`low/medium/high/xhigh`、400K/272K/128K、Responses-only、function calling、web search、hosted shell、skills、phase、完整 output replay 和 compaction。它们不证明公开了内部 attention、MoE、训练 recipe 或 kernel。
5. `tool_search` 的当前文档支持边界从 GPT-5.4 及之后模型开始变化，不能因为 GPT-5.3 Codex 支持 skills/hosted shell/function calling，就把 deferred tool search 自动记入该模型能力。

当前状态为“资料级闭环”：有榜单发现、官方资料、研究笔记和书系配套，但没有独立架构/训练报告；下一轮仍只从 Artificial Analysis 与 DataCurve DeepSWE 的八家重点厂商条目选择锚点。

## gpt-oss 当前核验结果

Artificial Analysis 当前有精确的 `gpt-oss-120b` 与 `gpt-oss-20b` 条目，页面以 `high` 配置展示；DataCurve 当前没有精确的 `mini_swe_agent_gpt_oss_120b_*` 或 `mini_swe_agent_gpt_oss_20b_*` 行。因此这两个对象是 OpenAI 的 AA 单榜锚点，不能迁移 GPT-5.x、Codex 或其他 OpenAI 模型的 DeepSWE Pass@1、成本、输出 token 和 Agent steps。

OpenAI Model Card、官方 gpt-oss 仓库、Harmony 仓库、Hugging Face 固定文件和 Cookbook 共同支持以下公开技术：120B/20B 两个尺寸的 total/active 参数差异，128/32 experts 与 top-4 路由，交替 sliding-window/full attention，GQA，RoPE/YaRN，post-training MXFP4，`o200k_harmony` tokenizer，CoT RL，Harmony 的角色层级/channel/recipient/tool 结构，以及 `low/medium/high` variable-effort reasoning。`active parameters` 不能替代总权重显存、KV cache、通信 buffer 或并发 workspace。

“可在单个 80GB GPU 运行”“约 16GB memory 级别”是官方指定实现、量化和硬件条件下的部署目标；不能推广成所有后端的 SLO。模型卡给出 benchmark、工具和安全结果时，必须同时保留 effort、工具、harness、verifier 和量化条件。完整训练/后训练 recipe、router 负载均衡、MXFP4 kernel 细节与误差消融、目标硬件 profiling、线上接受率和独立 Agent 评测仍待核验。

OpenAI Cookbook 的实现验证与 raw CoT 指南进一步把服务验收拆成：Harmony/render 和 API shape、tool-call compatibility smoke test、raw CoT/state replay、AIME/GPQA/HealthBench quality eval，以及 kernel/precision/hardware/production acceptance。Responses 使用 `reasoning.content[].reasoning_text` 与 `response.reasoning_text.delta/done`；Chat Completions provider 可采用 `reasoning` 字段约定。raw CoT 默认不能展示给终端用户，且 final、tool call、analysis 的保留范围必须按 turn lineage 回放。

当前状态为“内容专题闭环 + 官方 provider 兼容性/raw CoT protocol evidence”：正式章节为第二十一册第 87 章，研究笔记为 [`gpt-oss-source-notes.md`](gpt-oss-source-notes.md)。这两个尺寸属于一个模型族，不重复创建两篇架构章节。兼容性 smoke test 的通过不能替代 MXFP4 kernel、MoE dispatch、质量 eval、目标硬件 profiling 或线上 acceptance。

2026-09-23 的当前时点复验没有改变上述归并：7890 代理下百度、AA 中文首页和 DataCurve 均 HTTP 200；AA 详情为 120B `4,078,295` bytes / `43e5f13a3e58976b077acac1810bce82ac60ce33a7d81476f0dc2b17175532de`、20B `4,083,068` bytes / `64d67047a5964c0109f103108f9b77c64edbaf1ee17f3fec3c82ffccd7cea2cc`。当前 AA 字段为 120B `11.6028431512592` / `196.389235173389 tokens/s` / `0.10742452290394947` cost/task，20B `8.9675171856126` / `185.656167974395 tokens/s` / `0.012460640242179213` cost/task；它们只说明第三方 provider 测量时点，不能写成模型升级。

本轮 DataCurve 快照仍无两个 gpt-oss 的精确 Agent 行，故保持“不适用/负证据”而不是把相邻 OpenAI 模型的 harness 结果借过来。官方 Cookbook 页面当前哈希变化也只记为动态文档快照：兼容性验证 `337,423` bytes / `e6500d57bc6041133a8f63acc03101260182ea58d45d2f7248806ae609c85474`，raw CoT `339,052` bytes / `1c08e3fb06a129256596c1af9ba907b7b3c435be3b61cfb00a72da83b1071fe6`；没有观察到新的模型 ID、权重 revision 或架构声明。

## Claude Opus 4.6 的证据分层

Artificial Analysis 的 `claude-opus-4-6-adaptive` 与 `claude-opus-4-6` 是同一基础模型的配置级条目；`adaptive`、effort 和 provider 不应被算成独立 checkpoint。DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_6_*` 行，不能把相邻 Claude 版本的 Agent 结果迁移过来。

Anthropic API 文档能够证明可观察的 1M/128K/300K 边界、adaptive thinking、effort、thinking signature、compaction、tool search 和 computer-use 协议；它们不能证明参数量、MoE/稠密结构、训练 recipe 或内部 verifier。发布页 benchmark 属于发布方自报，AA Intelligence Index 属于第三方配置字段，DeepSWE Pass@1 属于 `mini-swe-agent` 组合结果，三者必须分栏保存。

对于面试和服务设计，Opus 4.6 最有价值的可迁移知识是状态协议：thinking signature、tool call/result、compaction block、权限决定、执行回执和 artifact verifier 必须成为可回放 trace。tool search 只降低 schema 上下文，computer use 只产生动作提案；权限、沙箱、allowlist、人工确认和副作用验证仍属于宿主系统。

## Claude Opus 4.7 的证据分层

Artificial Analysis 的 `claude-opus-4-7` 与 `claude-opus-4-7-non-reasoning` 是同一基础模型的配置级条目；`xhigh`、`high`、`adaptive` 和 provider 不应被算成独立 checkpoint。DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，不能迁移其他 Claude 版本的 Agent 结果。

2026-09-21 runtime recheck 的 Artificial Analysis `/zh` 首页为 `1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`，max 详情为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`；详情页显示该条目 deprecated，当前 max Intelligence Index `40.6897928205908`（estimated）、约 `51.1354` output tokens/s、1M context。Anthropic 官方模型页同时仍给出 `Active (legacy)`，并建议迁移 Opus 5；这是第三方目录状态与一方 API 生命周期字段的差异，不能混写成“模型已下线”。DataCurve 仍无精确行。

Anthropic 官方资料能够证明可观察的 1M/128K 边界、effort、task budget、compaction、tool search 和高分辨率视觉输入契约；它们不能证明参数量、MoE/稠密结构、训练 recipe、内部 tokenizer 实现或视觉编码器结构。task budget、effort 和 `max_tokens` 分别属于 loop、step 和 request 层预算，不能相加推导并发容量。

Opus 4.7 发布页中的 cyber safeguards 与 Cyber Verification Program 是部署和安全控制面证据。面试或服务设计应把策略拦截、授权、沙箱、审计、人工升级和模型能力分层，不把自动拒答当作内部安全算法或已经获得执行权限的证明。当前已通过第二册和第二十四册既有章节完成内容专题闭环。

## Gemini 3 Deep Think 的配置级归并规则

Artificial Analysis 的 `gemini-3-deep-think` 是精确榜单条目，但 DataCurve 当前没有精确的 `mini_swe_agent_gemini_3_deep_think_*` 行。Google 官方没有公开独立 Deep Think API model ID、Model Card、权重或专属技术报告，因此本项目将它记录为“配置级锚点”，而不是新的基础模型或 checkpoint。

官方可核验对象是 Gemini 3.1 Pro Preview / Gemini 3 系列的 Deep Think 推理设置及其运行时协议。`thinking_level`、共同 output budget、thought/tool signature、stateful/stateless `id` 回放、1M context 和 caching 可以作为面试知识；但不能改写成 Deep Think 独有架构、参数、训练 recipe 或独立 benchmark。Artificial Analysis 的指数、约 130K context、速度、价格和 provider 字段只能作为第三方目录与复现索引。

## Qwen3.5-397B-A17B 的证据分层

Artificial Analysis 的 `qwen3-5-397b-a17b` 详情页存在 Reasoning/Non-reasoning 配置，DataCurve 当前没有精确 `mini_swe_agent_qwen3_5_397b_a17b_*` 行。前者是第三方目录配置字段，后者只表示当前没有该配置的 `mini-swe-agent` 组合结果；不能迁移其他 Qwen 版本的 Pass@1、成本、输出 token 或 Agent steps。

Qwen 官方模型卡和固定配置支持 397B total/17B active、60 层、hidden 4096、15 组 `3 x Gated DeltaNet + 1 x Gated Attention`、512 experts、10 routed + 1 shared、vision encoder、262,144 native context、约 1,010,000 YaRN context 和 MTP serving 示例。`active` 不等于总权重显存、KV/state cache 或并发容量；MTP serving 示例也不等于线上 acceptance rate。

模型卡/博客中的 early-fusion、多模态 token、million-agent RL、asynchronous RL 和 201 languages/dialects 是 Qwen 官方自报。Qwen3.8 README 表示其建立在 Qwen3.5 架构基础上，但 QSA、Gated Residual、N-gram Embedding、Muon 等 Qwen3.8/Flash-Next 技术不能反向写入 Qwen3.5，除非存在 Qwen3.5 专属来源。

当前状态为“内容专题闭环”：研究笔记、第二十一册第 83 章前置/对比小节及配套同步均已具备。完整 GDN/GA kernel、state layout、MTP acceptance length、视觉独立 benchmark、Qwen3.5-Plus hosted/open 精确差异、训练/后训练 recipe 和目标硬件 profiling 继续保持待核验。

## DeepSeek V3.2 实现证据的分层规则

V3.2 的 Artificial Analysis 条目仍是 `Non-reasoning` 配置，DataCurve 没有精确 `mini_swe_agent_deepseek_v3_2_*` 行；因此任何 vLLM/`lm-eval` 结果都不能填补排行榜缺口。实现层证据需要再分四层：V3.2 最终模型卡字段、V3.2-Exp 参考 inference、DeepGEMM/FlashMLA/TileLang kernel、vLLM recipe/harness。

固定官方 V3.2 `config.json` 与 V3.2-Exp inference config 都核验为 `q_lora_rank=1536`，但绑定不同 revision/artifact。后者同时公开 `index_topk=2048`、FP8 indexer、non-interleaved indexer RoPE、prefill MHA/decode MQA 和 FP8 KV cache 路径；这些是实验实现事实，不能单独推出完整生产配置或最终线上性能。

DeepGEMM PR #200、FlashMLA PR #98 和 TileLang `examples/deepseek_v32` 说明 kernel 分工和 sparse MLA 执行链；vLLM recipe 的 `DP=8, EP=8, TP=1`、TP fallback、FP8/BF16 KV cache 和 `max-num-seqs` 建议只绑定特定硬件/版本/recipe。recipe GSM8K 结果必须和模型卡 benchmark、Artificial Analysis 指数、DataCurve Agent 结果分栏保存。

## Gemini 3.5 Flash-Lite 当前核验结果

Artificial Analysis 的 `gemini-3-5-flash-lite` 是本轮的明确 AA 候选；第三方页面字段为 2026-07-21、`isReasoning=true` 和 Intelligence Index `22.1685424839812`。DataCurve 当前页面没有精确的 `mini_swe_agent_gemini_3_5_flash_lite_*` 行，只有相邻 Gemini 3.x Flash 条目，所以不迁移任何 Pass@1、成本或 Agent steps。

Google API 模型页确认 `gemini-3.5-flash-lite`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching、code execution、Computer Use Preview、File Search、function calling、Maps/Search grounding、structured output、thinking 和 URL context。Thinking 文档把它列为默认 `On (minimal)`，支持 `minimal/low/medium/high`；这是请求级 reasoning 配置，不是四个 checkpoint。

Google 视频文档把 3.5 Flash-Lite 列入 agentic video processing。static 路径按固定 1 FPS 取帧；agentic 路径动态浏览时间轴，按需加载 transcript/帧/音频，并通过 `processing_call`/`processing_result` 事件提供可审计处理状态。文档声称长内容最多约 88% token efficiency 和约 7% quality 提升，但这属于发布方通用描述，不能当作每个请求的保证。

DeepMind Model Card 明确说明 3.5 Flash-Lite based on Gemini 3.1 Flash-Lite，并将 architecture、training dataset、data processing、hardware、software 都指向 3.1 Model Card。它自报 SWE-Bench Pro `54.2%`、Terminal-Bench 2.1 `54.0%`、MLE-Bench `39.2%`、OSWorld-Verified `74.0%`、GDM-MRCR v2 128K/1M `72.2%`/`21.3%`，同时标出 `$0.30/$2.50` input/output price；这些不能与 AA 或其他 harness 结果拼接。

正确的 `whats-new-gemini-3.5` 页面正文属于 Gemini 3.5 Flash，不属于 Lite，因而不能把 Flash 的默认 `medium`、GA 或 thought preservation 写到 Lite。当前状态为“资料级闭环（AA 单榜）”，暂无独立架构章节；完整快照哈希、来源和待核验项见 [`gemini-3.5-flash-lite-source-notes.md`](gemini-3.5-flash-lite-source-notes.md)。
## 2026-09-18 Kimi K3 证据层升级

Kimi K3 已同时出现在 Artificial Analysis 与 DataCurve，因此保留为重点锚点；`low`/`max` 是运行配置，DataCurve 的 `mini-swe-agent` 行是 Agent harness 结果，不拆成新的基础模型。与旧记录相比，K3 已不再是“报告和配置待核验”：MoonshotAI 官方仓库、`k3_tech_report.pdf`、README 和许可证均已核验。

技术报告支持的 K3 专属配置包括 2.8T total、104B activated、93 layers、69 KDA + 24 Gated MLA、3:1 KDA/Gated MLA、末尾 Gated MLA、8 个 Block AttnRes（每 block 12 层）、896 routed experts/16 selected/2 shared、1,048,576 context，以及 MXFP4/MXFP8 QAT、SiTU-GLU 和 Quantile Balancing。Kimi Linear/Attention Residuals 的论文机制仍必须与这些 K3 report 字段分栏引用。

README 的“完整权重已发布”是官方声明；本轮已固定 Hugging Face API revision `f831ab66814297da540d832a5235f8e904f29d06` 与 `config.json`，但没有下载完整权重，因此仍不能写成本地权重已获取。许可证已核验，但商业条件应随许可证版本记录。FlashKDA master commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`、H20/GB200 仓库 benchmark 和 vLLM recipe 已进入官方实现证据；stable upstream merge、完整训练数据/optimizer recipe、独立复现、目标硬件 profiling 和线上 acceptance rate 继续待核验。

## 2026-09-20 Kimi K3 实现与 serving 证据升级

本轮没有从官方仓库另发现模型；HF、FlashKDA 和 vLLM 资料只用于核验已由两个排行榜发现的 K3。官方 HF API 固定 revision 为 `f831ab66814297da540d832a5235f8e904f29d06`，`config.json` 确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 full-attention/Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 experts/top-16/2 shared 和 1,048,576 context。完整 safetensors 仅作 metadata 证据，本地未下载权重。

FlashKDA README/Atom feed 固定 master commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`；`8x8 fp32 forward substitution + 16x16 bf16 merge` 替代 fp16 Neumann inverse，要求 SM90+、CUDA 12.9+、PyTorch 2.4+，并提供 `chunk_kda`/recurrent state/变长 batch 接口。H20 与 GB200 的固定长度 benchmark 分别报告 `1.85x` 与 `2.31x` 相对仓库 baseline，但不是本地复现或端到端 K3 吞吐。

vLLM K3 recipe（页面更新 2026-09-10）要求 vLLM 0.29.0 及 K3-enabled nightly/image；hybrid KV manager 同时管理 MLA attention cache 与 KDA recurrent state，Blackwell 路径涉及 FP8 KV、TOKENSPEED MLA、prefix caching 和 `--prefix-match-unit 128`，并按 TP/TEP/DEP/PP、RDMA/NVLink 拆分部署账本。recipe 还警告 tool-call parser 偶发不兼容，因此宿主仍需 schema validation、retry、幂等和 verifier。

当前结论从“报告已核验、实现待核验”升级为“内容专题闭环 + 实现资料补证”；不能把 recipe 写成 stable upstream merge，也不能把仓库 benchmark 写成独立复现或线上 acceptance。

## Kimi K2.6 当前核验结果

Artificial Analysis 的 `kimi-k2-6`/`kimi-k2-6-non-reasoning` 是 Kimi K2.6 的配置级条目；页面的 deprecated 和指向 K3 只是第三方目录状态。DataCurve 当前没有精确 `mini_swe_agent_kimi_k2_6_*` 行，因此不迁移 K2.7 Code 或 K3 的 Pass@1、成本、输出 token 和 Agent steps。

K2.6 官方模型卡与配置公开 1T total/32B active、61 层、384 routed experts、top-8、1 shared expert、MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、YaRN 256K、MoonViT 约 400M 和 native INT4 的实现字段。顶层 `KimiK25ForConditionalGeneration`/`kimi_k25` 与文本子配置 `DeepseekV3ForCausalLM`/`kimi_k2` 是实现兼容标识，不足以推导 K2.6 的完整训练 recipe，也不能改写为 DeepSeek 发布模型。

官方博客的 300 sub-agents、4,000 coordinated steps、长周期 coding 案例属于 Agent harness/发布方案例，不属于模型内部专家数或推理深度。`preserve_thinking`、`reasoning_content`、interleaved thinking 和 multi-step tool call 属于可回放协议；模型输出 tool call 仍不是宿主执行权限。Kimi Vendor Verifier 的 pre-flight、视觉、长输出、ToolCall 和 SWE-Bench 检查属于部署/验收层，不能写成 K2.6 模型能力分数。

当前状态为“AA 单榜资料级闭环”，正式专题为第二十一册第 90 章 [`Kimi K2.6：Native Multimodal、Agent Swarm 与推理验收`](../../book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)。当前 Kimi Code 文档中的 K3/K2.8/K2.7 model ID、128-agent AgentSwarm、resume、模型池、会话 wire replay 和权限复核属于通用 harness 补证；K2.8 未在两个排行榜快照出现，不升级为候选或独立模型。完整层排布、训练/后训练 recipe、production kernel、INT4 误差与硬件 profiling、K2.6 Agent Swarm coordinator、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验。

## GPT-5.4 mini/nano 当前核验结果

Artificial Analysis 的 `gpt-5-4-mini` 与 `gpt-5-4-nano` 是本轮的精确 AA 候选；DataCurve 当前只有 `mini_swe_agent_gpt_5_4_xhigh`，其中 `mini_swe_agent` 是 harness 名称，不代表 GPT-5.4 mini。没有 `gpt_5_4_mini` 或 `gpt_5_4_nano` 精确 Agent 行，因此不能把 base GPT-5.4 的 Pass@1、成本、输出 token 和 Agent steps 迁移到 sibling。

OpenAI 官方模型页分别确认 `gpt-5.4-mini-2026-03-17` 与 `gpt-5.4-nano-2026-03-17`，两者都是 text/image → text、400K context、272K maximum input、128K maximum output，并支持 `none/low/medium/high/xhigh` reasoning effort。mini 的官方定位是高吞吐 coding、computer use 和 Agent workflow；nano 的官方定位是 classification、extraction、ranking 和窄任务 sub-agent。官方价格按精确 model ID 分开记录，不能使用 AA 目录的价格替代。

能力清单必须按精确页面读取：mini 当前列出 `tool_search` 与 `computer_use`，nano 当前页面没有列出这两项。两者即使同属 GPT-5.4，也不能共享一份 capability manifest；实际 router 还要加入 endpoint、provider、宿主授权、沙箱、审批、超时和 verifier。

面试归因应采用：`task shape -> capability probe -> mini/nano/base route -> explicit prompt contract -> verifier -> escalation`。nano 不适合默认承担开放式多步规划；应先把输入边界、输出 schema、工具顺序、失败恢复、停止条件和 abstain 行为写清。`reasoning_effort` 是请求级投入旋钮，不是可直接拿来做并发容量规划的硬 token 上限。

当前状态为“AA 单榜资料级闭环（关联档位）”。完整证据见 [`gpt-5.4-mini-nano-source-notes.md`](gpt-5.4-mini-nano-source-notes.md)；参数、层/专家/注意力结构、训练 recipe、system card、独立技术报告、精确 DataCurve 结果、生产 kernel、硬件 profiling 和线上 acceptance rate 仍待核验。

## GLM-5.1 当前核验结果

Artificial Analysis 有精确的 `glm-5-1` 与 `glm-5-1-non-reasoning` 条目；DataCurve 当前没有精确 `mini_swe_agent_glm_5_1_*` 行。因此 GLM-5.1 是 **AA 单榜资料级闭环**：可以沿 Z.ai 官方文档、模型卡/配置和 release notes 提取长周期 Agent、过程质量评估、DSA/MoE 配置和 API 协议，但不能记录或迁移 GLM-5、GLM-5.2、GLM-5.3 或 GLM-5.3-Flash 的 DeepSWE 分数。

官方配置公开 `GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed experts、top-8、1 shared expert、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048` 和 `202752` position。Artificial Analysis 的约 744B/40B、200K、指数和速度仍是第三方目录/provider 字段；配置没有独立公开总参数账本。不能因为 GLM-5 的技术报告使用相近 DSA/MoE 字段，就把 GLM-5 的 28.5T、`slime` 或具体训练叙述改写为 GLM-5.1 专属事实。

Z.ai 文档、release notes 和官方博客把 GLM-5.1 的新重点放在最长约 8 小时的 long-horizon agentic engineering、实验—分析—优化循环、数百轮/数千工具调用和过程质量；博客还区分有数值指标的 VectorDBBench 外循环、带正确性/反作弊审计的 KernelBench Level 3，以及无单一指标的 Linux desktop 自评闭环。release notes 没有公开具体 RL 算法、奖励模型或 verifier 实现。SWE-Bench Pro `58.4`、600+ 迭代/6,000+ 工具调用/21.5k QPS、655 次迭代/6.9× 吞吐和 KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 是发布方自报，不能与 AA 或其他 harness 拼成裸模型能力。

`thinking.type=enabled/disabled` 是 GLM-5.1 的请求级模式；当前 Z.ai 文档将 `reasoning_effort` 支持列为 GLM-5.2 及以上，不能从 GLM-5.2 迁移到 GLM-5.1。Function calling、MCP、structured output 和 context caching 是 API/Agent 协议能力，工具授权、执行、回灌、缓存命中和最终 verifier 仍属于宿主系统。2026-09-21 已取得博客正文资源，但模型卡仍链接 GLM-5 报告，arXiv 精确检索未发现 GLM-5.1 专属技术报告。

完整证据和快照见 [`glm-5.1-source-notes.md`](glm-5.1-source-notes.md)。当前不新增重复 Transformer 章节，复用 GLM-5 DSA/MoE、Agentic Engineering、reasoning、工具协议和 serving 内容；待核验项包括完整参数/训练 recipe、DSA indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、8 小时 harness、线上 tool acceptance、精确 DataCurve 行和独立复现。

## Grok 4.20 当前核验结果

Artificial Analysis 有精确的 `grok-4-20` 页面，canonical 标题为 `Grok 4.20 0309 v2 (Reasoning)`；DataCurve 当前只有 Grok 4.5/4.6 配置，没有精确 `mini_swe_agent_grok_4_20_*` 行。因此 Grok 4.20 是 **AA 单榜资料级闭环**，不能迁移相邻 Grok 版本的 Agent 成绩。

AA 当前字段为第三方 estimated Intelligence Index `25.6550155187053`、2M context、约 97.01 tokens/s、TTFT 22.71s、`$1.25/$2.50` input/output，并标记页面 deprecated -> `grok-4-3`。xAI 官方模型页/注册表则确认主 ID `grok-4.20-0309-reasoning`、non-reasoning 与 multi-agent sibling，写明 1M maximum prompt、text/image -> text、function calling、structured output、reasoning、Batch、200K long-context price threshold 和区域限流。2M vs 1M 是第三方目录与官方服务字段差异，不能压成一个结论。

2026-09-24 7890 快照为 3,857,337 bytes / SHA-256 `d56d4557d8a661ee9c3015c6a3661f5baae6c00f9b16c5b2acf5eabcd3e61300`；当前页面给出的速度为 `106.190642707844 tokens/s`、TTFT `21.53034693s`、端到端 `26.23885972594964s`。页面同时声明只继续更新默认 10K input token workload 的性能基准、其他 workload 为不再更新的历史结果。快照中的指标数值变化只说明 AA/provider 展示值或采集时点变化，不证明模型 revision；未绑定具体 workload 的汇总字段应标 freshness 未确认。`deprecatedTo: grok-4-3` 是 AA 目录关系，不是 xAI API 下线证据。

xAI 的新技术公开在 Agent runtime：`grok-4.20-multi-agent` 为 beta，由多个专门 Agent 搜索、分析、交叉核验，leader agent 汇总；4/16 agents 由 `agent_count` 或 `reasoning.effort` 映射。子 Agent 中间状态默认不直接返回，`use_encrypted_content` 可携带 opaque encrypted state。Context Compaction、自动 prompt caching、server-side Web/X Search/Code Execution、client-side function calling、parallel calls、Remote MCP `allowed_tools` 和 mixed-tool `max_turns` 是可迁移的面试主线。

当前没有 Grok 4.20 专属架构报告、参数披露、公开权重、完整训练/后训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 或独立 DataCurve Agent 评测。Release Notes 的 March 2026 条目只确认 Grok 4.20/Multi-agent live；三个 xAI 新闻候选路径返回 404，arXiv 标题精确检索没有专属报告。完整证据见 [`grok-4.20-source-notes.md`](grok-4.20-source-notes.md)。

## 2026-09-21 GLM-5.3 官方博客脚注的解释边界

本轮两榜当前快照在重点 canonical slug 层面没有新增模型，因此 GLM-5.3 仍是既有双榜锚点。官方博客正文资源哈希只能证明这次读取到的页面内容版本，不能证明文章中的 benchmark 已被外部复现。

Z.ai Code Bench 的 50% 提升和 Max/High effort 的 completion/token efficiency 说明发布方把完成率和输出成本同时作为 Agent 指标；它们仍受私有任务集、checklist、harness、effort 和统计口径约束。网络安全部分形成能力阶梯：CyberGym 的白盒漏洞发现/故障验证、ExploitBench 的更深利用推理、ExploitGym 的时间归一化任务完成；2,436/269/1,097 是合作代码库经专家复核、筛选和去重后的发布方统计，不能等同于模型单独自动发现的全量漏洞。

因此 GLM-5.3 的官方评测结果应表示为：

    result = F(model_revision, effort, harness, tools, environment,
               timeout, isolation, domain_policy, verifier, aggregation)

缺少其中任一关键字段，都不能把 CyberGym、ExploitBench、ExploitGym、Terminal-Bench、ALE、DeepSWE 和 Z.ai Code Bench 的数字放在同一排序中。特别是 ExploitGym 使用 Artificial Analysis TPS 做时间归一化，DataCurve 则是另一套 mini-swe-agent 运行系统；二者不能相互校准。

当前面试最值得保留的结论是：后训练规模化的瓶颈继续从模型本体转移到环境生成、reference-free verifier、reward-shortcut audit、长轨迹状态、train/rollout numerical alignment 和 benchmark anti-hack。SAO 论文现在提供了公开算法细节，但仍只有“GLM-5.3 继承自 GLM-5.2”的版本归因；不能升级成 5.3 独有算法。

## 2026-09-20 Gemini 3.8 Flash 复验解释

本轮三条代理对 AA 首页、Gemini 3.8 三个详情页和 DataCurve 均返回一致内容；Google 官方模型页、Thinking、Interactions 通过 `7890` 成功获取。AA high 的约 1M context、FAQ Intelligence Index 约 41、Google API TTFT 16.44s 和 `$0.75/$3.75` 是第三方目录/测量或文档价格字段；Google 的 `minimal` 不支持是官方模型能力字段。它们必须按 source、endpoint、snapshot 和核验日期分账。

DataCurve high 的 Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36`、平均 `166.31` Agent steps 绑定 `mini-swe-agent`、工具、任务环境、运行配置和 verifier，不能与 AA 指数拼成统一能力分数，也不能外推 low/medium 或 Gemini 3.7/3.5。`thinking level`、thought summary/signature、Interactions step、tool call 和 Computer Use 描述的是 API/runtime 协议；当前仍没有 3.8 独立参数、架构、训练 recipe 或 verifier 论文证据。因此 Gemini 3.8 维持“资料级闭环”，活动锚点转入全局配套与可复现实验设计。

## 2026-09-20 GLM-5.3 证据分层解释

本轮新鲜 AA 与 DataCurve 响应三代理一致。AA 的 753B/40B、1M、指数、速度、TTFT 和价格是第三方目录/provider 测量字段；HF config 的 78 层、256 routed/top-8/1 shared、DSA indexer、MTP index sharing 和 1M position 是公开实现配置；Z.ai 文档的“同 GLM-5.2 基座、收益来自后训练”、环境生成、verifier、SAO with compaction、slime 和 >2.3x throughput 是发布方描述；IndexCache/SAO 是关联论文，其中 SAO 的直接 GLM 部署证据落在 GLM-5.2。四层不能合并成一个“GLM-5.3 内部已完全公开”的结论。

DataCurve GLM-5.3 max 的 Pass@1 `68.9579%`、Pass@4 `87.6106%`、成本约 `$3.99` 和平均 124.47 steps 绑定 `mini-swe-agent`、400K context/工具/超时/verifier 等系统条件；Z.ai 模型卡的 DeepSWE `66.9` 则绑定官方 Claude Code harness、temperature/top-p、timeout 和其他脚注。两者不是同一 benchmark，不能互相校准或写成裸模型分数。

当前最值得保留的面试结论是：后训练 scaling 的瓶颈从“模型 loss”转移到环境可执行性、验证器抗 reward shortcut、长轨迹状态和 train/rollout consistency；SAO 论文补充了公开算法，但 `SAO with compaction` 在 GLM-5.3 页面仍只是继承描述，尚无 5.3 专属算法报告。因此 GLM-5.3 维持“双榜资料级闭环”，不新建基础架构章节。
## 2026-09-20 DeepSeek V4 Pro：reference implementation evidence 的解释边界

V4 Pro 的公开证据现在可以分为四层：AA 的 `deepseek-v4-pro` 配置、DataCurve 的 `mini_swe_agent_deepseek_v4_pro_max` harness 行、官方模型卡/报告/API，以及固定 HF revision 下的 encoding/inference artifact。第四层能证明某些类、配置字段和 kernel 路径存在，但不能自动证明线上 API 使用同一 revision、完整权重已下载、生产 GPU 达到 README 的吞吐或模型内部 reasoning 机制已经公开。

面试时应把以下链条说完整：`Compressor` 的压缩误差、`Indexer` 的候选漏检、核心 attention 的 evidence recall、局部窗口的近期信息补偿和 `overlap state` 的块边界连续性是不同指标；`n_hash_layers=3`、`sqrtsoftplus`、`hc_mult=4`、`expert_dtype=fp4` 是 reference inference 字段，不应改写成完整训练 recipe；`MP=8` 是官方转换示例，不是硬件无关的生产并行结论。

encoding 也要单独归层：DSML、`<think>`、tool role 合并、严格 parser 和 `reasoning_effort` 接受范围描述 prompt/protocol 行为，不等于模型真实隐藏思维链。最后，DataCurve 的 Pass@1/4 仍绑定 max effort、mini-swe-agent、工具、任务集、环境和 verifier；reference code、榜单指数和 Agent 分数不可拼成一个裸模型能力值。

## 2026-09-20 DeepSeek V4.1-Flash：reference implementation 的证据边界

Artificial Analysis 的精确 slug 为 `deepseek-v4-1-flash`；DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行。故 V4.1-Flash 的 Agent 数字不能从 V4 Pro 或 V4 Flash 借用，当前状态仍是 AA 单榜的内容专题闭环。

本轮固定 Hugging Face revision `dba1be0a40aa45a94ad051997016db3960a90277`，确认仓库公开 `inference/model.py`、TileLang `inference/kernel.py`、`convert.py`、Engram/vision 路径和 `dsh-minimal.patch`。这些源码足以把 CED 外的执行账本具体化为 SWA ring、压缩 KV、两级 candidate/index top-k、sparse attention、MoE、Engram 和 mHC/Sinkhorn；但 `inference/README.md` 把它定义为 readable reference implementation，`generate.py` 的入口仍是普通自回归生成。

因此应分开记录三层事实：模型卡/技术报告描述的 DSpark 设计；reference code 中存在的 `forward_spec` 和 DSpark block；以及尚未完成的 draft/verify/rollback scheduler、接受长度、GPU profiling 和生产 serving。HF API 的 48 个权重分片与 dtype metadata 也不能替代本地完整权重加载。源码静态编译和 encoding smoke test 通过，只证明源码/协议快照可解析，不证明 CUDA、TileLang 或线上服务验收。

新增的 [`deepseek_v41_cache_demo.py`](code/deepseek_v41_cache_demo.py) 只做 CPU 标准库教学实验：它把 candidate-pool recall、候选池内 conditional Top-K recall、端到端 recall 和分组 E2M1-like 误差分开输出。该实验用于说明待核验指标如何设计，不是 V4.1 权重、生产 kernel、acceptance length 或硬件性能的替代证据。

## 2026-09-21 DeepSeek V4.1-Flash：协议实现不能升级模型证据等级

本轮将官方 `deepseek-recipe` 固定到 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`，并核对 3,996,119 bytes 源码归档（SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`）。它是 V4.1 已有榜单锚点的协议/serving 周边资料，不是从 GitHub 目录发现的新模型。

面试时应按下面的账本分层：

1. `ProtocolRequest -> ConversationRequest` 是 API schema 的规范化；它保存 conversation、推理参数、解析选项、model 和 stream，不代表后端已经执行请求。
2. `DeepseekV41Encoding` 是 prompt/template/tokenizer 层；它固定 DSML 标签、mid-conversation system 和 effort 渲染，不公开隐藏 reasoning 算法。
3. `StateMachine -> StreamProcessor` 是跨 chunk 的增量解析层；它能识别 reasoning、DSML tool call、JSON 和 stop marker，但不执行工具、不验证 JSON 业务语义、不授予权限。
4. `ImageResolver` 是输入资源层；它有 URL/data URL/bytes、并发和总字节 quota，但默认 fetcher 明确不做 SSRF/private-address filtering。
5. inference backend、HTTP transport、tool executor、policy、verifier、timeout 和线上 SLO 仍属于宿主应用。

README 还列出该 commit 尚未支持的 `logprobs`、server-side web search、JSON Schema/strict enforcement、`n>1`、Responses context storage 和 encrypted thinking；这些是适配器的已知边界，不能当作 V4.1 模型能力的负面结论。默认 image limits（600/32 MiB/64 MiB/8 concurrent）和 mock server 也只能作为 runtime implementation facts。当前状态因此升级为**协议实现证据补强**，不升级为完整权重加载、生产 kernel、speculative acceptance、硬件 profiling 或线上 Agent benchmark 闭环。

## 2026-09-21 当前时点网络证据边界

本轮对三个代理 `10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234` 的连接均失败。该结果只说明当前访问路径不可用，不改变 Artificial Analysis、DataCurve 或官方页面的存在性，也不应把旧快照的日期改成当前采集日期。后续榜单比较必须重新记录采集时间、URL、配置、页面哈希和代理状态；网络失败期间不得生成新的模型候选或迁移相邻模型的分数。

同一轮随后网络恢复：三条代理对 Artificial Analysis、DataCurve 和 Opus 5 官方页面均返回 HTTP 200。当前 AA Opus 5 详情为 `3,868,875` bytes / `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`，DataCurve 为 `268,571` bytes / `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；新鲜 AA canonical slug 集合没有增加八家重点厂商模型。故应将前一失败记录标为短时传输故障，并以新鲜快照作为当前页面证据，同时继续区分榜单配置分数与模型本体结论。

## 2026-09-21 Kimi K3 manifest 与 hybrid cache 证据边界

Kimi K3 的固定 HF revision 现在有三层可复核证据：`config.json` 描述模型结构与量化契约，`model.safetensors.index.json` 描述 tensor 到 96 个分片的映射，`modeling_kimi_linear.py` 描述 Transformers reference path 的 cache/state 分工。三者都不等于完整权重已下载、CUDA backend 已运行或线上 API 使用同一 commit。

本轮审计通过 `497,220` 个 tensor、连续 96 个分片、93 个 layer id、69 个 KDA/24 个 full-attention 配置层、92 个 expert-bearing layer，以及 `247,296` 对 packed/scale tensor。`metadata.total_size=1,560,860,324,864` 是 packed index 的文件权重口径；HF API 的 `U8/BF16/F32` 参数统计总量 `2,779,931,837,184` 是另一种 dtype 统计口径。两者不能加总、互换或写成同一个“模型大小”。

源码层的面试结论是：full-attention 层按序列追加 `key_cache/value_cache`，KDA 层保存短卷积状态与递归 state；prefill/chunk 与单 token decode 选择不同 kernel 路径。于是恢复、prefix hit、beam reorder、KV dtype、cache eviction 和跨 revision 回放都要在 manifest 中区分 MLA cache 与 KDA state。新增 [`kimi_k3_manifest_audit.py`](code/kimi_k3_manifest_audit.py) 只做本地 JSON/index 审计，不做权重加载或性能宣称。

## 2026-09-22 Kimi K3 SGLang 证据解释

SGLang 对照把 K3 的 serving 证据再分成 stable tag 与 mutable main 两层。`v0.5.20` 的 K3 text/vision 文件证明 stable release source entry 存在；`main` 的较新文本文件证明当前 upstream 正在处理 token shard、EP/A2A、shared-expert TP、SBO、ModelSlim 权重映射和 KDA fused decode 的实现问题。它们不能互相替代，也不能推导完整权重已经被加载。

最重要的 runtime 语义是三重分离：

1. **状态分离**：MLA `key_cache/value_cache` 与 KDA `conv_states/recurrent_states` 不能合并为一个 KV length；prefill/chunk/decode/verify 也要记录不同 kernel path。
2. **通信分离**：routed expert 的 latent A2A、shared expert 的 TP gather/reduce-scatter、attention/SP-MoE 的 reduce-scatter/all-gather 有不同 token layout；把它们记成一个 `tp=8` 会隐藏重复 dispatch、错误 reduce 或死锁风险。
3. **证据分离**：源码/registry entry、stable artifact、完整权重 load、数值正确性、目标 profile、cache recovery 和 tool/verifier acceptance 必须逐级验收。

视觉文件在两个 SGLang 快照中逐字节一致，说明本轮新增差异集中在文本 serving/runtime，而不是视觉代码已经发生版本升级。仍然要单独验收视觉 grid/patch 对齐、变长 segment、attention backend、CUDA graph 和多模态状态注入。

## 2026-09-22 Kimi K3 SGLang commit history 的解释边界

本轮网络恢复后固定的 SGLang commit history 不改变模型发现规则：commit 只是已经确认的 Kimi K3 的 serving artifact。8ac19cc 和 c4d3770 分别把 deferred KDA gate 和 CUDA graph stream 的责任边界讲清楚；c2c3629 与 72d5c5b 说明 expert loader、ModelSlim mapping 和 routing dtype 是加载正确性的一部分；f4c2563、2d0e94e、8ac39c6、cb32dbc 则把 PP/DCP/DSpark、process group、Ascend A5 和 ROCm 分支纳入 runtime 证据。

因此证据顺序更新为：榜单 canonical identity -> HF revision/config/index -> stable runtime entry -> mutable main commit history -> full-weight load -> dual-state recovery -> target hardware profile -> tool/schema/idempotency/verifier acceptance。commit message、改动文件和测试入口不能替代后三层运行证据。当前 K3 状态仍是内容专题闭环 + SGLang stable/main source evidence，不把 main 写成 release，也不把硬件分支写成本机性能结果。

## 2026-09-21 GPT-5.6 Luna 当前快照的解释边界

GPT-5.6 Luna 本轮仍满足“双榜锚点”：AA 精确详情页与 DataCurve 精确 `mini_swe_agent_gpt_5_6_luna_max` 行都存在。AA 的 `37.3244239690841`、164.5096 tokens/s、1M 和 `$0.20/$1.20` 是第三方 max 配置/provider 测量；DataCurve 的 301/448、Pass@1/4、成本、输出 token 和 Agent steps 是 `mini-swe-agent` 加工具、环境和 verifier 的组合结果。两者可以证明当前配置证据，但不能拼成模型本体的统一分数。

OpenAI 官方页面本轮返回 403/超时/DNS 失败，只能说明当前访问路径不可用；已有 9 月 14 日官方模型/API 快照仍支持 persisted reasoning、`reasoning.context`、prompt-cache breakpoint、30m TTL、tool search 和 compaction 的既有记录。不能因为本轮未重新打开页面就否定这些历史证据，也不能把历史快照标成 9 月 21 日新鲜响应。参数规模、内部结构、训练配方和独立 GPT-5.6 报告仍属于未公开/待核验。

## 2026-09-21 GLM-5.3-Flash runtime 证据分层

GLM-5.3-Flash 的新鲜榜单快照仍指向同一个 canonical 模型：AA 的指数/速度发生测量漂移，DataCurve 的 `mini_swe_agent_glm_5_3_flash_max` 行保持 `284/448`。因此不能因为第三方页面的数值变化就创建新模型或新 checkpoint。

本轮新增的 SGLang、vLLM 和 Transformers 资料回答的是不同问题：

1. SGLang cookbook 证明某个页面提供了 Flash 的 serving recipe，并公开 paged KV pool、KDA state pool、MTP、KV/DSA backend pairing、EPD/PD 验证门禁；它不证明本机目标硬件已经加载完整权重或通过 accuracy/SLO。
2. vLLM recipe 证明存在 v0.29.0+、native FP8/MTP、hybrid KDA+sparse MLA 的部署路径，并列出硬件/依赖门槛；`FlashInfer 0.6.17+` 与 troubleshooting `0.6.18+` 的页面差异本身就是迁移风险，不能选一个数字就声称环境已验收。
3. Transformers GLM5-Next 文档明确不包含 MTP layer。vLLM/SGLang 在 serving 层提供 MTP 路径，不能反向推出 checkpoint 或 Transformers 基础实现含 MTP。
4. `GLM-5.3-FlashX` 是 Z.ai 的关联服务入口，不在两个排行榜的新增候选中；其约 200 tokens/s、配额和 endpoint 字段不产生新的模型锚点。

面试中应把证据链说成：`榜单 canonical identity -> 官方模型卡/config -> runtime recipe -> pinned dependency/hardware -> full-weight load -> cache/state recovery -> target profiling -> tool/verifier acceptance`。当前 Flash 已达到内容专题闭环并完成 runtime 资料补证，仍未达到本机生产 serving 闭环。

## 2026-09-21 DeepSeek V3.2 当前页面漂移与 README 证据边界

本轮的 3,638,730 bytes AA 详情页仍指向同一个 `deepseek-v3-2` canonical identity。页面结构化字段为 `Non-reasoning`、685B/37B、128K、Intelligence Index `16.043537719683` 和 `$0.28/$0.42`；旧快照的 648B 不应与新字段拼成趋势，也不能解释为模型更新。当前页面没有 output-speed/TTFT 数据，就保持缺失，不用其他模型或旧 provider 的速度补位。

V3.2-Exp README 的新鲜证据应按三层理解：README/模型卡是官方发布描述；TileLang 是研究可读实现；DeepGEMM/FlashMLA/SGLang 是特定 kernel/serving 入口。README 中的 V3.1-Terminus 对照表是发布方 benchmark，不是 DataCurve；SGLang 的镜像和 tp=8, dp=8, enable-dp-attention 命令也不等于本机 full-weight load、数值正确性或 p99 已验收。

vLLM recipe 当前 URL 404 的正确解释是“当前 URL/线路没有取得页面”，不是“vLLM 没有 V3.2 实现”。因此证据状态应写成：历史 recipe 记录保留，当前链接可用性为 negative access evidence，最终部署 capability 仍需固定 revision、依赖、硬件和 harness 后验证。

## 2026-09-22 DeepSeek V3.2 的 deprecated 与 redirect 解释

Artificial Analysis 当前把 `deepseek-v3-2-reasoning-0925`、`deepseek-v3-2-0925`、`deepseek-v3-2-reasoning`、`deepseek-v3-2` 和 `deepseek-v3-2-speciale` 标为 V3.2 家族的 deprecated/redirect 对象。正确的 inventory 处理是把它们归并到 V3.2 的 canonical 家族，并保留 Exp、reasoning/non-reasoning 和 Speciale 的 revision/configuration 语义；不能按页面数量新增五个模型，也不能把 redirect 目标解释成训练或架构升级的证明。

DataCurve 只有 V4 Flash/Pro 的精确行，没有 `mini_swe_agent_deepseek_v3_2_*`。因此 V3.2 仍只有 AA 单榜资料级证据，不迁移相邻版本的 Pass@1、成本或 Agent steps。deprecated 是第三方目录当前状态，不能替代官方 API 的下线公告；不同页面的 648B/685B 参数字段也只是目录漂移，不能解释为 checkpoint 变化。

## 2026-09-22 Grok 4.7 的证据分层

Grok 4.7 当前只有 Artificial Analysis 的精确 canonical 条目，没有 DataCurve 精确行。AA 的 `46.4465` 指数、约 `38.77 tokens/s` 和约 `$3.7383` Intelligence task cost 是第三方配置/provider 测量；不能与 xAI 发布页的 DeepSWE `71.0%`、CursorBench `46.3%` 或本地 Agent 实验混成一个能力分数。

xAI 发布页提供“更大 base、更长 RL、困难长任务、自验证和 safeguard stack”等高层训练/系统披露；它没有公开参数、MoE/稠密结构、完整 optimizer/reward/rollout 配方。模型页中的 500K、effort 档位、价格、限流和 `algorithm` 配置字段属于服务合同；Responses 的 encrypted reasoning 和 compaction item 属于 opaque 协议状态；function calling、structured output 与 MCP 属于工具控制面。每一层都不能反向证明下一层。

发布方 benchmark 必须带 benchmark、prompt、effort、harness、任务集、verifier 和统计时间；`reasoning_effort` 不是四个模型，`allowed_tools` 也不是完整安全边界。arXiv 精确标题检索无 Grok 4.7 结果，因此当前结论是 **AA 单榜内容专题闭环**，而不是架构或训练配方已公开。

本轮 reasoning 页面刷新把运行时证据再拆成三层：`reasoning.encrypted_content` 及服务端工具加密输出是必须原样回放的 opaque state；`response.reasoning_text.delta`/`response.reasoning_summary_text.delta` 是面向开发者的可见 summary/stream 事件；`store` 与 `previous_response_id` 决定服务端 response 状态引用，不能替代客户端自己的事件账本。Remote MCP 还存在 API/SDK 字段差异：Responses 使用 `allowed_tools`/`headers`，xAI SDK 使用 `allowed_tool_names`/`extra_headers`，当前 transport 只到 Streaming HTTP/SSE，`require_approval`/`connector_id` 不能作为已支持的权限控制面。它们都属于 API 合同与 harness 设计证据，不是 Grok 4.7 的内部推理或安全算法披露。

## 2026-09-23 Grok 4.7 当前复验

`7890` 沙箱外请求恢复后，两榜首页均返回 HTTP 200；当前快照没有八家重点厂商的新 canonical 条目。Grok 4.7 AA 详情的页面 hash/大小发生动态变化，但 release、Intelligence Index、500K context、proprietary 和无参数字段保持一致。xAI 模型页、Reasoning、Compaction 的正文也没有新的模型合同差异，因此这次只算当前时点复验，不算新版本。

新增的 [`grok47_state_replay_audit.py`](code/grok47_state_replay_audit.py) 把公开协议转成最小 host-side 门禁：encrypted content 必须按 opaque item 原样保存；可见 summary 不能替代它；tool call/output 必须以 call id 和 idempotency key 对齐；compaction 只能有一个且不能重复执行已提交副作用；`store=false` 不能依赖 `previous_response_id`。这是 `local_protocol_toy`，不提供 xAI 真实实现、加密语义或模型能力证据。

## 2026-09-22 K2 Horizon 3.7B 的证据分层

Artificial Analysis 有精确的 `k2-horizon-3-7b` 条目；DataCurve 当前没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 行。因此 K2 Horizon 3.7B 是 **AA 单榜资料级闭环**：可以沿 IFM 模型卡、固定配置、迁移清单和 serving recipe 提取 dense 长上下文技术，但不能记录或迁移 K2 Horizon 36B/A4B、K2.7 Code、K3 或其他模型的 DeepSWE 分数。

当前 revision `6360f705b2e57d542959e6a2e67ebeb95dae0373` 的配置是 `K2HorizonForCausalLM`、36 层、2560 hidden、32Q/8KV GQA、524,288 position、default RoPE、BF16，且 `num_experts=0`、`mova_num_experts=0`。这使它成为 36B MoVA 的 dense 对照，而不是 36B 的缩小版：36B 后 45 层还有 MoVA value top-4、FFN top-8 和 shared expert，3.7B 没有这些 forward-time 路由。

参数口径必须分层：AA 的 3.7B 是目录名称；权重 index 的 10,116,510,720 BF16 bytes 与 vLLM 的 5.06B dense/including-embeddings 口径相符；旧 `APPENDIX.md` 的 3.78B core/5.06B including embeddings 可以作为 core/embedding 解释，但其 `XllmForCausalLM`/FP32 字段与当前 K2Horizon/BF16 revision 冲突。不能把旧文档直接覆盖当前 artifact，也不能用一个数字替代 core、embedding、dtype 和 revision 账本。

训练阶段、RL expert merge 和中间 checkpoint 属于官方模型卡高层披露；migration manifest 属于 artifact 复制/登记证据；vLLM/SGLang 的 TTFT、TPOT、吞吐和 GSM8K 属于发布方 recipe 结果。三者不能升级成完整训练 recipe、所有 backend 等价或本机实测。SGLang 文档引用的旧 revision 当前 404 只说明该部署引用不可解析，不说明模型不存在。完整证据见 [`k2-horizon-3.7b-source-notes.md`](k2-horizon-3.7b-source-notes.md)。

## 2026-09-22 Qwen3-VL-235B-A22B 的身份归并与内容闭环

Artificial Analysis 有精确的 Qwen3-VL-235B-A22B instruct/reasoning 页面；两个页面是同一基础模型的运行配置或 artifact，不应计为两个模型。DataCurve 当前没有精确 `mini_swe_agent_qwen3_vl_*` 行，因此不迁移其他 Qwen 模型的 Pass@1、成本或 Agent steps。AA 的 `235B total / 22B active`、速度、价格和 `262,144` context 是第三方目录/provider 字段，不能与论文 benchmark、HF config 或本地 profiling 混成一个无来源数字。

当前状态升级为 **AA 单榜内容专题闭环**：研究笔记、第二十一册第 92 章、模型清单、来源索引及全局配套已具备。Qwen3-VL 的公开实现主线是 SigLIP2 vision encoder、两层 MLP merger、Qwen3 MoE decoder、Interleaved-MRoPE、DeepStack `[8,16,24]`、Video Timestamp、S0-S3 长上下文 curriculum、SAPO/General RL 和 Thinking with Images 的 tool-call reward。

证据边界需要保留：Interleaved-MRoPE 是三轴频率分配，不是 1M context 证明；DeepStack 不增加额外视觉 token 长度，但会增加投影/激活计算；Video Timestamp 是显式时间证据，不代替媒体采样和回放审计；论文中的 visual-agent interaction、reward 和 benchmark 不是本地 Agent 成功率；Transformers raw main 代码快照没有固定 upstream commit，不等于生产 kernel。完整权重加载、跨视频 chunk replay、视觉/稀疏 kernel、目标硬件 profiling、端到端多模态 serving、GUI/tool acceptance、完整训练 recipe 和独立 benchmark 仍待核验。详见 [`qwen3-vl-source-notes.md`](qwen3-vl-source-notes.md)。

## 2026-09-22 Qwen3-Omni 30B A3B 的证据分层

Artificial Analysis 当前有精确的 `qwen3-omni-30b-a3b-instruct` 与 reasoning 页面；DataCurve 当前没有精确 `mini_swe_agent_qwen3_omni_*` 行。因此 Qwen3-Omni 是 **AA 单榜内容专题闭环**：可以沿 Qwen 官方 GitHub、模型卡、博客、技术报告和阿里云文档提取多模态架构与服务周边，但不能记录或迁移 Qwen3.8、Qwen3.5 或其他 Qwen 条目的 DeepSWE 分数。

第三方 instruct 页面字段约为 `35.3B` total、`3B` active、`66K` context、Intelligence Index `6.0061` 和 `94.59 tokens/s`；这些绑定页面/provider 和采集日期。官方报告/模型卡支持 Thinker-Talker、AuT、TM-RoPE、Talker 多码本/MTP 和 Code2Wav，但完整参数统计、专家布局、生产 kernel 和流式状态接口仍未完全公开。

面试时应把 Qwen3-Omni 的技术链拆成四层：

1. **多模态编码层**：AuT 以 12.5 Hz/约 80 ms 音频时间粒度编码，视觉 encoder 来自 Qwen3-VL/SigLIP2-So400m，TM-RoPE 对齐 temporal/height/width。
2. **理解与生成层**：Thinker 负责多模态理解/推理，Talker 使用多模态特征生成首码本和 residual codebooks；不能把 Talker 简化成只接 Thinker 文本的 TTS。
3. **系统介入层**：Thinker 输出后可以插入 RAG、function calling、policy/safety 和 verifier，再把允许的上下文交给 Talker；模型 tool proposal 不等于工具执行。
4. **serving 层**：异步 chunked prefill、Code2Wav、音频 packet、取消/恢复和高并发调度需要独立 manifest；报告的 `234/547 ms` 首包数字不是本机实测或线上 p99。

Instruct、Thinking、Captioner 记录为同一 Qwen3-Omni 家族内的不同 artifact。官方 README 当前还说明 vLLM 主要支持 Thinker，Instruct 音频输出处于实现推进阶段；因此“仓库可运行”“Thinker 可服务”和“完整音频输出已生产验收”不能合并成一个结论。完整证据见 [`qwen3-omni-source-notes.md`](qwen3-omni-source-notes.md)。

## 2026-09-22 当前时点复验的解释

Claude Opus 5 与 Gemini 3.5 Flash-Lite 本轮都属于已存在的排行榜锚点，官方资料只用于补充它们的协议、评测和技术周边，不产生新的模型候选。Opus 5 的 9 月 21/22 日 AA 速度和 TTFC 差异应按第三方 provider/采集时点测量漂移处理；没有 release、canonical slug 或官方 revision 变化证据，不能写成模型升级。

Gemini 3.5 Flash-Lite 只有 Artificial Analysis 的精确候选；DataCurve 当前只有相邻 `gemini_3_5_flash_high`，因此不能迁移 Agent 分数。Google Model Card 明确把 3.5 Flash-Lite 的 architecture、training dataset、data processing、hardware 和 software 指向 Gemini 3.1 Flash-Lite；当前可以写的是版本依赖、API 输入输出、thinking level、agentic video processing events 和发布方评测边界，不能写成 Lite 独有的 MoE、层数、参数量、optimizer 或训练算法。

本轮两者均为资料级闭环，不新增重复 Transformer 正式章节。后续比较必须继续同时固定模型 canonical identity、effort、provider、工具集、harness、任务集、verifier、采集日期和页面哈希。

## 2026-09-22 GLM-5.1 当前时点复验的解释

Artificial Analysis 当前有精确的 `glm-5-1` 与 `glm-5-1-non-reasoning` 条目；三条代理取得一致详情页，当前快照为 `3,971,543` bytes / SHA-256 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`。当前第三方字段为 Intelligence Index `26.0585912980095`、`37.1922485381327 tokens/s`、cost per Intelligence Index task `0.9217822401466147` 和 200K context。9 月 20/21 日的速度、价格和页面大小按 provider/采集时点漂移保存，不解释成 checkpoint、训练或架构变化。

DataCurve 2026-09-22 快照为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_glm_5_1_*` 行。因此 GLM-5.1 当前是 **AA 单榜资料级闭环**：可以沿 Z.ai 模型文档、博客、模型卡/config、release note 和 API 文档提取 long-horizon Agent、process-quality evaluation、DSA/MoE 配置和 thinking/tool/cache 协议，但不能记录或迁移 GLM-5、GLM-5.2、GLM-5.3 或 GLM-5.3-Flash 的 DeepSWE 分数。

GLM-5.1 的官方博客 benchmark 仍必须绑定发布方 harness：VectorDBBench 的 600+ iterations/6,000+ tool calls/21.5k QPS，KernelBench Level 3 的 H100、最多 1,200 turns、`atol=rtol=1e-4`、双审计器和 3.6x 对比 `torch.compile` max-autotune 1.49x，以及 Linux desktop 的 8 小时自评循环，均不是裸模型分数。模型卡链接的是 GLM-5 技术报告；不能把 GLM-5 的 28.5T、slime 或训练/架构结论自动迁移为 GLM-5.1 独有事实。当前不新增重复 Transformer 正式章节，后续只补完整 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、线上 acceptance 和独立复现。

## 2026-09-22 GPT-5.6 Luna 当前时点的证据分层

GPT-5.6 Luna 当前同时满足两个排行榜的精确锚点。AA 的 `37.3244239690841`、`158.728370482714 tokens/s`、1M context 和 `$0.20/$1.20` 是 `max` 配置的第三方/provider 测量；DataCurve 的 301/448、Pass@1/Pass@4、成本、输出 token、Agent steps 和 median peak context 是 `mini-swe-agent + tools + task environment + verifier` 的联合结果。两者不能拼成一个“裸模型总分”，9 月 21 日快照也不能与 9 月 22 日测量拼趋势。

本轮官方页面恢复后，证据可以进一步分为三层：

1. **模型合同**：Luna 页面确认 cost-sensitive/high-volume 定位、1,050,000 context、922,000 maximum input、128,000 output、`none`--`max` effort、端点和工具兼容性。这些是可观察接口字段，不是内部结构。
2. **模型—协议状态**：Reasoning 文档确认 GPT-5.6 的 `standard/pro` mode、effort 独立和跨轮 reasoning item 语义；Prompt Caching 确认 1,024 visible-token 门槛、显式断点、最多四次写入和 `30m` TTL；Compaction 确认 `context_management`、opaque compaction item 和 stateless/`previous_response_id` 两种回放路径。
3. **Agent runtime 所有权**：Agents API 托管 Codex harness 和进度状态；Agents SDK 把 loop、部署、存储、审批和 runtime 交给应用；Responses API 让应用直接管理 response item 和工具循环。Hosted tool search 由服务端搜索并返回加载集合，client-executed tool search 由应用以原 `call_id` 回传加载集合；搜索到工具不等于获得执行权限。

缓存与 compaction 的联合解释是新的面试重点：tool search 把新增 schema 追加到上下文尾部，尽量保持旧 cache prefix；compaction 则替换较早上下文，可能从第一个变化位置起让旧 prefix 失配。运行时评估要同时记录 cache hit/write、compaction、tool loading、权限决定、工具回执、artifact 和成功成本，不能只比较输入 token 数。当前状态为 **双榜资料级闭环**，仍没有 GPT-5.6 参数、MoE/稠密结构、完整训练/后训练 recipe 或独立技术报告证据。

## 2026-09-22 Qwen3.7 Plus 的证据分层

Artificial Analysis 的精确 `qwen3-7-plus` 页面把 Qwen3.7 Plus 标为 2026 年 6 月发布的 text/image/video-in、text-out 服务，并给出约 1M context、Intelligence Index `25.1622215821984`、`68.5428061089526 tokens/s` 和约 `$0.40/$1.60` 的第三方/provider 字段。DataCurve 当前快照没有精确 `mini_swe_agent_qwen3_7_plus_*` 行，因此不能记录或迁移其他 Qwen 的 Pass@1、成本、输出 token 和 Agent steps。

Alibaba Cloud 官方页面把它描述为多模态交互式混合 Agent，并在 Model Studio capability table 中提供 GUI/视觉定位、Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching、1M context 和 region/scope 差异。这些属于托管产品合同和可观察 API 行为；“能读屏/导航”不等于公开了视觉 encoder、GUI policy、手机执行器或端到端成功率。

本锚点最重要的面试边界是 `model proposal -> schema -> host permission -> GUI/mobile executor -> observation -> verifier`。Structured Outputs 只约束形状，Function Calling 只约束调用协议，Web Search 只表示某个 region/scope 的能力开关；权限、沙箱、幂等、超时、重试、截图 revision 和真实副作用仍由宿主负责。Prefix Completion/Context Caching 也应和 GPU KV cache、应用 memory、compaction 分开记账。

当前状态为 **AA 单榜资料级闭环**：研究笔记、官方模型/API 文档、索引、书系配套和面试训练均已具备；参数规模、dense/MoE、层数、视觉架构、训练/后训练 recipe、公开权重、专属技术报告、生产 kernel、移动端 acceptance 和独立 benchmark 仍待核验。不能把 Qwen3.5、Qwen3.8、Qwen3-VL 或 Qwen3-Omni 的技术反向写成 Qwen3.7 Plus 内部实现。

## 2026-09-22 DeepSeek V4.1-Flash 当前时点复验的解释

本轮三条代理均能取得 Artificial Analysis 中文首页、DataCurve DeepSWE、DeepSeek V4.1 发布页和 AA 详情页，页面哈希与大小已写入来源索引；因此可以确认当前榜单/发布页可达和 canonical identity 未变化。HF API 只有 7890 成功，8098 的 503/连接失败和 1234 的失败必须单独记为代理线路证据，不能作为模型仓库不存在、revision 被删除或权重不公开的结论。

V4.1-Flash 的 AA `39.456167472527`、约 1M context 和 `$0.30/$1.20` 是当前 provider 目录字段；DataCurve 没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，所以不能把 V4 Pro/V4 Flash 的 Agent 结果迁移过来。HF revision `dba1be0a40aa45a94ad051997016db3960a90277` 与 48 个 safetensors 分片保持不变，说明本轮没有观察到新权重 revision，但不等于完整权重已下载或生产 serving 已验收。

面试和后续实验仍应把 CED/CSA2/SWA/FP4/Engram/DSpark/EPD 的“机制证据”“reference 实现”“目标硬件 profiling”“线上 acceptance”分开记账。当前可继续补的不是另一个模型名称，而是 candidate/index Top-K 的真实召回曲线、FP4 对 logits/质量的影响、DSpark 接受长度与 rollback、EPD 调度以及完整权重和生产 kernel 的可复现证据。

### vLLM main 与 stable release 的额外分层

本轮补到的 vLLM `main` registry 在 `vllm.models.deepseek_v41` 下登记了 `DeepseekV41ForCausalLM` 和 `DSparkV41DraftModel`，并已有 NVIDIA/ROCm 的视觉和 DSpark 实现文件。这一事实可以升级 V4.1 的 **upstream runtime implementation evidence**，但不能升级为 stable release、可安装 wheel 或生产服务支持。vLLM `0.29.0` stable registry 的固定快照仍只有通用 V4/DSpark 入口，没有这两个 V4.1 专用类名。

因此，V4.1 的证据等级按以下顺序记录：官方模型卡/技术报告描述机制；HF 固定 revision 的 inference 目录提供 reference code；vLLM `main` 提供 upstream serving code；stable registry/wheel 才能证明 release surface；完整权重加载、GPU/ROCm profiling、speculative acceptance、FP4 质量、EPD 调度和线上 SLO 还需要独立运行证据。特别是 vLLM vision wrapper 显式跳过 `mtp.*`，所以不能把视觉路径和 DSpark draft 路径因为同属 V4.1 就自动合并。

## 2026-09-22 DeepSeek V4.1-Flash：SGLang main 与 stable 的证据分层

SGLang `main` 的文件实现把 V4.1 runtime 证据推进了一层，但没有改变模型发现规则：`deepseek_v4.py`、`deepseek_v4_dspark.py`、`deepseek_v41_vit.py` 和 `deepseek_v4_nextn.py` 都只是已确认 V4.1 锚点的 serving artifact。main 代码中的 V4.1 vision 分支在 `model_type=deepseek_v41` 且 `vision_n_layers > 0` 时创建 ViT/Aligner，支持 TP/EP/DP，明确不支持 CP/PP/MoE A2A；ViT 使用 patch embedding、full bidirectional attention、2D RoPE 和 attention data parallel。DSpark 路径则使用 Markov/confidence head、`mtp.*` 映射、共享 target embedding/lm head，并把含 vision 配置的 draft stage 的 `vision_n_layers` 置零，因此不能把 target vision 与 draft vision 当成已经合并的能力。

`_dequant_fp8` 对 V4 `128x128` 与 V4.1 `32x32` block size 的分支、MXFP8/FP8 prefill autotune、FlashInfer、unified KV 和 DSV4 sparse indexer 都是源码实现事实；它们不等于完整权重加载、kernel 正确性、真实召回或线上吞吐。SGLang `v0.5.20` stable tag（`94602c9c...`）只有通用 V4/DSpark 文件，缺少 `deepseek_v41_vit.py`，源码中 `deepseek_v41`/`dsv41` 计数为 0。因此不能用 main 文件替代 stable release，也不能把 runtime integration 的存在写成 production support。

本轮后，V4.1 的 serving 证据顺序为：HF reference implementation -> vLLM main -> SGLang main -> stable release/wheel -> 完整权重与目标硬件验收。仍待核验视觉 token 的 draft/target verify、FP4/FP8 质量、DSpark acceptance/rollback、EPD 调度、GPU/ROCm/NPU profiling、tool acceptance 和生产 SLO。

## 2026-09-22 GLM-5.3 标准 DSA 与 Flash 版的证据分层

本轮把活动锚点从 `GLM-5.3-Flash` 切回标准 `GLM-5.3`。两者名字相近，但不能共用实现结论：标准版的固定 config 是 `model_type=glm_moe_dsa` / `GlmMoeDsaForCausalLM`，Flash 版的入口是 `Glm5NextForConditionalGeneration`，后者才涉及 `RadixLinearAttention`、KDA、视觉模块和双 state pool。

标准版当前最可靠的结构证据来自 HF revision `aca966e4e02791568aa6a4ced368624b3d897f42` 与 Transformers main。配置字段给出 78 层、前三层 dense、256 routed experts/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048` 和 1,048,576 positions；源码把 indexer 分成 21 个 `full` 层与 57 个 `shared` 层。Full 层计算当前候选，Shared 层复用 `prev_topk_indices`，并在主 attention 中施加 sparse mask。这里可以确认实现路径和状态依赖，不能从字段推导完整训练损失、召回率、实际 FLOPs 或生产吞吐。

vLLM `main` 与 `v0.29.0` 的 registry 都把 `GlmMoeDsaForCausalLM` 路由到 `deepseek_v32`；SGLang `main` 与 `v0.5.20` 也包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。因此“标准 GLM-5.3 复用 DeepSeek-V3.2 DSA runtime”是源码路由层面的事实，不等于两个 checkpoint 的权重、训练 recipe、kernel 或 benchmark 等价。vLLM main 的 PCP/DCP、`SparseCacheRole.INDEXER`、HiSparse 和 logical top-k 也只能标为 main 分支实现证据。

面试时应按以下证据阶梯回答：

```text
榜单 canonical identity
  -> 固定 HF revision/config
  -> Transformers forward semantics
  -> stable/main registry and model source
  -> full-weight load and numerical check
  -> index/evidence recall and cache recovery
  -> MTP acceptance and target hardware profile
  -> tool/verifier/SLO acceptance
```

源码文件存在只通过第四层；recipe 或 registry 不会自动通过完整权重、目标硬件、召回、MTP、工具和生产 SLO。尤其不能把 Flash 的 linear/KDA/视觉/EPD 结论倒灌到标准 DSA 版，也不能把 IndexCache 或 SAO 关联论文改写成 GLM-5.3 独有的已证实训练算法。

## 2026-09-23 DeepSeek V4.1-Flash：vLLM `v0.30.0` release surface 的解释

本轮把 vLLM 的证据从 mutable `main` 推进到正式 `v0.30.0` tag。tag commit、release API、stable registry、NVIDIA/ROCm package 和 PyPI artifact 彼此一致，说明 V4.1 专用入口已经进入可分发的 release surface；这修正了此前只能写“main 有入口、v0.29.0 未证明”的阶段性结论。

但 release surface 仍然不是 runtime acceptance。`registry.py` 只证明类可以被模型类型映射发现，package 文件只证明对应代码随 tag 发布，wheel/SDist 元数据只证明 artifact 可分发。它们没有证明完整 48 分片权重已经下载、权重转换与 dtype layout 一致、NVIDIA/ROCm kernel 在目标设备上数值正确、DSpark 的 draft/target verify 和 rollback 可用，或真实任务的 acceptance length、FP4 质量、EPD 调度、tool acceptance 和生产 SLO 达标。

因此本轮状态升级为：**内容专题 + HF reference implementation + vLLM `v0.30.0` stable release/source evidence + SGLang main + recipe protocol evidence（AA 单榜）**。后续仍按 `release surface -> full-weight load -> numerical check -> recall/acceptance -> target profile -> tool/verifier/SLO` 推进；SGLang `v0.5.20` 的 V4.1 stable 专用 vision/runtime 仍未由本轮证据证明。

## Claude Opus 5.5 的榜单解释

2026-09-23 的 Artificial Analysis 首页新增/展示 `Claude Opus 5.5 (max with fallback)`，详情页 canonical slug 为 `claude-opus-5-5`，并同时展示 low/medium/high/xhigh effort 变体。它们应归并为一个基础模型条目；`max with fallback` 是 effort 与安全路由配置，不是新的 checkpoint。当前 DataCurve 页面没有精确 `mini_swe_agent_claude_opus_5_5_*` 行，不能把 Opus 5、Fable 5.1 或其他 Claude 的 Agent 成绩迁移过来。

Anthropic 发布页称 Opus 5.5 通过更少的 token、工具调用和重试降低长任务成本，并对网络安全、生物和蒸馏使用更严格的 safeguards/fallback。这里至少有三种不同证据：

1. AA 的 Intelligence Index、价格、context 和 provider 速度是第三方配置测量；
2. Anthropic 的成本、benchmark 和客户案例是发布方/合作方结果，绑定 effort、工具、任务、环境、safeguard 和可能的 fallback；
3. DataCurve 若出现精确行，仍会绑定 `mini-swe-agent`、任务仓库、工具、verifier、重试和成本统计。

因此“Opus 5.5 更便宜/更高效”只能在固定模型、effort、fallback、工具、harness、verifier 和任务集后做质量-成本对照；不能从 40% 成本下降推导参数更少、架构更简单或训练方法改变。System Card PDF 已使用 Node.js `zlib` 与 ToUnicode/CMap 只读解析正文，具体安全数字现在可以记录，但仍必须绑定 snapshot、safeguards、工具、环境和 verifier。

## Claude Opus 5.5 的 API contract

7890 代理取得 Anthropic 官方 Markdown 模型页和迁移文档后，可以把 Opus 5.5 的已知事实再分成“服务契约”和“模型内部未知量”两层。服务契约包括 `claude-opus-5-5`、1M context、128K synchronous max output、always-on adaptive thinking、default `medium` effort、各平台 model ID、cache/batch pricing，以及 thinking/tool/computer-use 的兼容性变化；这些不能反推参数量、网络结构或训练 recipe。

面试和线上迁移应特别检查四类边界：`thinking` 不能 disabled 或手工 budget，`tool_choice` 不能用 `any`/指定工具强制调用，thinking block 与模型/会话和消息前缀绑定，Claude API/Google Cloud 要从 `computer_20251124` 迁到 `computer_toolset_20260801`。此外，tool-call 之间的进度可能以默认隐藏文本的 `thinking` block 返回，on-demand compaction 会产生带签名的摘要 block，fast mode 是同权重的更快 inference configuration。这里记录的是 API observable behavior，不是“模型发明了某种新 attention”。

## Claude Opus 5.5 的 System Card 正文证据

System Card 的训练资料边界是公开互联网、公开/私有数据、获许可用户数据和合成数据的组合；使用去重/分类与 ClaudeBot，不访问密码页、登录页或 CAPTCHA 页面，knowledge cutoff 为 2026 年 6 月。评测默认使用最终 snapshot，但部分章节使用早期/替代 snapshot 或关闭生产 safeguards。这个条件本身就是证据的一部分，不能把所有数字放入同一个“模型分数”字段。

能力侧，Terminal-Bench 4.0 为 `66.36%`（xhigh、Claude Code `--bare`、5 trials），ProgramBench 为 `91.2%`（166 golden tasks），OSWorld 2.0 partial/strict 为 `81.8%/48.7%`（108 tasks、1080p、最多 500 actions、5 runs、完整 screenshot、超过 100K tokens 后 server-side compaction）。五 Agent team 在 ProgramBench 同分比较中约 `2.7x` derived-latency improvement，DRACO 在 `0.5x` latency budget 下约 `2.8x` speedup；100-agent task 运行 24 小时并观察到 flat 与 hierarchical sub-lead 组织。derived latency 由 context token、tool time 和 handoff clock 计算，不等于 wall-clock。

安全侧，RSP 为 CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达。CoBench 2.1 为 Opus 5.5 `55.8%`，Mythos 5.1 `53.4%`，Opus 5 `53.2%`；AECI `169.36`，global CI `[165.23, 177.05]`，local CI `[167.99, 171.24]`。新 fit 使用 374 benchmarks、7,985 observations、732 models，不能与旧 fit 直接比较。Cyber 结果关闭 safeguards，ExploitBench 完整 ACE 为 `301/410 = 73.4%`，CyScenarioBench `67.6%`，ExploitGym 2h/6h 为 `289/869`、`300/869`，不能当作生产 cyber 成绩。

Agent safety 与 prompt injection 也要拆分：Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 为 `0.1%/0.7%/1.0%`，18% rollout fallback 到 Opus 4.8，1,310 个 fallback rollout 无成功攻击；coding Shade 从无 safeguards 的 `54.61%` 降到 probe 开启的 `11.13%`，computer-use probe `0.04%`，browser auto `0/110`。fallback 类型还区分 biology/CB -> Opus 5、cyber -> Opus 4.8、窄范围 frontier LLM kernel/AI R&D -> Opus 5，以及没有 fallback 的武器和 distillation/hidden reasoning extraction。

因此当前解释应升级为：**AA 单榜 + System Card 正文证据**。它支持面试中的安全评测、评测归因和 Agent harness 讨论，但仍不支持参数、架构、完整 recipe、独立复现或生产 acceptance；长轨迹、多 Agent、语言差异和模拟环境真实性仍是公开评测盲点。

## 2026-09-23 GPT-6 Sol：模型、配置与 runtime 的三层分账

Artificial Analysis 的 `gpt-6-sol` canonical 条目是 `GPT-6 Sol (max)`，页面同时包含 low、medium、high、xhigh、max 和 non-reasoning 等配置字段。它们应归并为一个基础模型；`max` 是榜单推理配置，不是新的 checkpoint。DataCurve 当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不能把 GPT-6 Astra、GPT-5.6 或其他 GPT 的 Agent 成绩迁移到 Sol。

本锚点至少有三层必须分开：

1. AA 的 Intelligence Index、速度、成本和 context 是第三方/provider 配置测量；
2. OpenAI model page 的 model ID、端点、effort、窗口、价格、工具和 region/processing 字段是服务合同；
3. Reasoning、Agents、Tools、Compaction 文档描述通用 API/runtime 协议，证明状态、预算、工具和恢复的可观察行为，但不证明 GPT-6 Sol 的参数、MoE/dense 结构、attention 变体或训练配方。

GPT-6 family 的 `reasoning.mode` 与 `reasoning.effort` 是不同控制面，`configuration_update` 可以改变后续 effort；这不等于会话内更换权重。Agents API、Agents SDK、Responses API 的 loop/state/executor 所有权不同；server-side 和 standalone compaction 都返回 opaque/encrypted state，不能把它们简化成可读摘要或永久记忆。评测必须固定模型、snapshot、effort、mode、工具、harness、任务环境、verifier、compaction 策略和 whole-request pricing threshold。

本轮重新抓取后的状态为 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**：榜单身份、官方模型页、运行时协议、研究笔记和配套同步已具备；DataCurve 精确 Agent 行、参数规模、内部架构、完整训练 recipe、公开权重、生产 kernel、目标硬件 profile 和独立 benchmark 仍待核验。`gpt6_sol_contract_audit.py` 只验证公开协议的状态机，不升级为真实模型能力或生产验收。

## GPT-6 Luna：同 family，不同模型合同

Artificial Analysis 的 `gpt-6-luna` 是独立 canonical release，不能因为它和 GPT-6 Sol 共享 family 名称就复用 Sol 的指标、价格或 Agent 评测。AA 的 `max`、`xhigh`、`high`、`medium`、`low` 和 `non-reasoning` 是配置变体；当前 DataCurve 没有精确 `mini_swe_agent_gpt_6_luna_*` 行，故不迁移任何相邻 GPT 结果。

OpenAI model page 给出的“focused、high-volume”是产品定位，不是参数规模或架构披露。Luna 的 1.05M/922K/128K、2026-05-18 knowledge cutoff、六档 effort、Responses 工具目录和 `$0.10/$0.50` 价格合同必须与 Sol 的独立合同分账；超过 272K 的 whole-request threshold、Batch/Flex、Fast mode 和 regional premium 进入 serving cost ledger。

GPT-6 family 的 `reasoning.mode`/`reasoning.effort`、`configuration_update`、Agents API/SDK/Responses 所有权以及 opaque compaction replay 可以复用为 runtime 证据，但不等于 Luna 具有某种已公开的 reasoning architecture 或训练算法。当前状态为 **AA 单榜资料级闭环**，正式内容复用既有长上下文、reasoning、Agent 和 serving 章节，不新增重复 Transformer 架构章。

## DeepSeek V4.1-Flash：API contract 不等于架构闭环

当前 `deepseek-v4-1-flash` 的 canonical identity 仍只由 Artificial Analysis 锚定；`deepseek-flash` 是官方 API 模型名。官方最新 API 文档补齐了 1M/384K/2500 concurrency、Vision 媒体预算、Files 生命周期、Responses semantic SSE、typed tool output 和 `/beta strict=true` schema enforcement。它们把服务接入合同补齐到可审计状态，但没有提升参数、训练、kernel 或完整权重的证据等级。

解释榜单时要保持三条边界：第一，AA 的 provider/速度/价格字段和官方 API 价格不必相同；第二，Responses/Files/Tools 的能力是 endpoint/runtime surface，不代表模型拥有权限或已经完成业务动作；第三，API 接受图片和工具回灌不代表视觉 token 的跨轮 cache、DSpark draft/target verify、FP4 质量或线上 SLO 已通过。当前状态为 **内容专题 + 官方 API contract + HF/reference + vLLM `v0.30.0` stable/source + SGLang main evidence（AA 单榜）**，待核验清单不变。

本地 API toy 只把上述边界变成可运行门禁：SSE parser 负责协议顺序和终态，schema gate 负责结构，permission gate 负责宿主授权，executor 负责副作用，verifier 负责业务结果。一次脚本通过不能证明真实服务通过；在证据矩阵中单独标为 `local_protocol_toy`，不能替换 endpoint probe、完整权重、硬件 profile 或生产 SLO。

## 2026-09-23 Kimi K3 当前时点复验的解释边界

本轮三条代理的 Artificial Analysis 中文首页和 DataCurve 快照逐字节一致；K3 详情页也逐字节一致。因而当前能确认的是榜单观察值和页面快照稳定，不能由页面稳定推导出模型权重或架构升级。

- AA 的 `43.5938229518782` Intelligence Index、速度和成本属于配置/provider 字段；DataCurve 的 `309/451`、Pass@1/Pass@4、token、steps 和成本属于 `mini-swe-agent + tools + environment + verifier` 系统账本。两者都必须带 `max`、harness、工具、环境、任务集和采集快照解释。
- README 与 HF metadata 哈希复验且 revision 仍为 `f831ab...`，说明当前没有观测到官方 artifact revision 变化；SGLang `main` 只增加 import/type annotation 级变化，不能写成 K3 runtime 功能升级。
- 发布方约 `2.5x scaling efficiency` 仍是声明；preserved thinking history、跨模型切换不稳定和 excessive proactiveness 是官方行为/协议边界，应分别进入 harness、状态回放、权限和 verifier 设计。
- K3 评测混用 Kimi Code、Claude Code、Codex、H20/H100 和 compaction 条件；因此不能从不同表格拼接出统一排名，也不能把发布评测或 DataCurve 精确行当成本地复现。

当前证据状态是**内容专题闭环 + 当前时点复验**。完整权重加载、目标硬件 profile、MLA/KDA 双状态 recovery、tool/schema/idempotency/verifier acceptance 和生产 SLO 仍属于 `unverified`。

## 2026-09-23 GLM-5.3：从 SAO 论文到模型继承关系

GLM-5.3 的官方文档给出的是版本继承声明：同一 GLM-5.2 基座、post-training 改进，并继承 `SAO with compaction`。SAO 论文提供的是公开算法证据：异步 RL 用 single-rollout 减少 group barrier，DIS 用 rollout log-probability 和双侧 token clipping 控制 policy lag，value model 通过更频繁更新、冻结 attention 和 MoE projection 微调来稳定 critic，Skip-Observation GAE 则跳过环境 observation 做 action 段之间的 credit assignment。

证据阶梯必须保持如下顺序：

```text
GLM-5.3 canonical榜单身份
  -> GLM-5.2继承声明
  -> SAO论文公开算法
  -> 5.3产品compaction实现（未公开）
  -> 5.3完整post-training recipe（未公开）
  -> full-weight / hardware / verifier / SLO验收
```

因此，SAO 论文可以成为 GLM-5.3 面试锚点的技术解释材料，但不能填充 5.3 的独立训练、参数、权重、硬件或 Agent 质量字段。论文的 Qwen3-30B-A3B 实验、GLM-5.2 结果和 DataCurve 的 `mini_swe_agent_glm_5_3_max` 行必须各自保留来源和配置。

## 2026-09-23 DeepSeek V4.1-Flash：Harness preview 的解释边界

本轮没有新增模型。`deepseek-v4-1-flash` 仍是 Artificial Analysis 已确认的 canonical 条目，官方 API 使用 `deepseek-flash`；DeepSeek Harness 文档来自同一厂商的 Agent runtime preview，只能扩展周边系统知识，不能反向证明 V4.1 的内部架构。

证据阶梯应保持为：

```text
AA canonical identity
  -> official API/provider contract
  -> Harness preview: provider/session/plugin/MCP/webhook
  -> local protocol toy
  -> full-weight / hardware / tool acceptance / SLO (unverified)
```

Harness 文档可支持的面试结论是：provider ID 和 session 中的实际模型必须稳定可追溯；credential 只可脱敏；插件依赖和外部 effect 要随生命周期清理；MCP 断线后必须重新发现并在预算耗尽后注销旧工具；GitHub review 的签名 webhook 和 HTTP `202` 只表示异步 admission，重复 delivery、出站权限、artifact 和 verifier 需要独立记录。

[`deepseek_harness_protocol_audit.py`](code/deepseek_harness_protocol_audit.py) 只验证上述责任边界的合成拒绝路径，证据等级为 `local_protocol_toy`。因此当前状态是 **AA 单榜内容专题 + API/runtime + Harness preview + local protocol toy**；DataCurve 无精确 V4.1 行，不能迁移其他 DeepSeek 的 Agent 结果。Harness 资料、API 资料、vLLM/SGLang source、完整权重和生产验收必须继续分账。

## 2026-09-23 GPT-5.6 Luna：当前时点复验的解释边界

本轮没有新增模型。`GPT-5.6 Luna` 仍是 Artificial Analysis 与 DataCurve 已确认的 canonical 条目；7890 恢复访问后，AA 的速度字段发生 provider/采集时点漂移，官方五份 Markdown 没有产生足以证明模型升级的新架构或训练信息。

证据阶梯保持为：

```text
AA canonical identity + current provider measurement
  -> DataCurve mini-swe-agent system measurement
  -> OpenAI model/API runtime contract
  -> local state-replay toy
  -> full-weight / hardware / endpoint capability / verifier / SLO (unverified)
```

本轮最有价值的面试知识不是“1M context 能装多少”，而是状态所有权：`current_turn/all_turns` 控制 opaque reasoning 的可用范围；完整 output item、function call/output 和 compaction item 组成可回放协议；prompt cache 只复用稳定前缀，compaction 改写前缀后可能 miss；hosted/client tool search 只改变 schema 发现责任，不授予 executor 权限。[`gpt56_luna_state_replay_audit.py`](code/gpt56_luna_state_replay_audit.py) 将这些边界转为可运行的拒绝路径和幂等回执，证据等级固定为 `local_protocol_toy`。

因此 GPT-5.6 Luna 当前状态是 **双榜当前时点复验 + 官方 runtime contract + local protocol toy**。不能从 API 字段、榜单速度、DataCurve Agent 行或 toy 结果反推出参数、内部架构、训练 recipe、完整权重、目标硬件性能或生产 SLO。

## 2026-09-24 两榜刷新：无新增重点 canonical 模型

当前 AA 中文首页与 DataCurve 页面均通过 7890 返回 HTTP 200。相较 2026-09-23 快照，AA 的八家重点厂商 canonical path 集合与 DataCurve 的 `mini_swe_agent_*` 配置集合都没有新增；AA 当前发布列表中的 Opus 5.5、GPT-6 Luna、GPT-6 Sol 均已有记录。因此本轮继续沿 Opus 5.5 官方 API 状态协议推进，不从 provider/effort 变体扩张模型清单。

Opus 5.5 与 Luna 的 AA 快照哈希/字节数变化，但 Opus 的 Intelligence Index 和 task cost、Luna 的 Index 和 task cost 与既有观测相同；Luna 输出速度作为当前 provider 时点值保存。页面快照变化只说明获取内容不同，不能证明模型权重或 revision 变化。DataCurve 配置集合相同且没有 Opus 5.5 精确行，所以它仍是 AA 单榜锚点，其他 Claude 的 Agent 结果不可迁移。

Opus 5.5 model page 的当前合同将同步 128K 与 Batch beta 300K 输出上限分开，并列出 always-on adaptive thinking、默认 medium、512-token cache minimum 与 cache read/write 单价。面试中应将上下文/输出限制、effort、缓存与 endpoint 账单拆开；这些公开字段不能用于反推 adaptive thinking 内部算法。第八册第 11 章新增 System Card 治理审计案例，统一绑定 snapshot、safeguards、fallback、成功定义/分母与计时口径；专题状态更新为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**。

## 2026-09-24 GLM-5.3：复验后的证据边界

本轮 7890 代理成功访问两榜和 Z.ai 资料，但没有产生新的 canonical 模型。Artificial Analysis `/zh` 当前快照为 `1,784,793` bytes / `a374adfb…`，标准 GLM-5.3 详情为 `4,057,703` bytes / `4f60e893…`；DataCurve 仍为 `268,036` bytes / `14436c31…`。这些页面的动态字节和 provider 测量只能说明当前获取观察，不证明权重 revision 或训练升级。

标准 GLM-5.3 的证据阶梯现在应写成：

```text
两榜 canonical identity
  -> Z.ai 官方 API/博客与 GLM-5.2 基座继承声明
  -> SAO 公开论文算法 + GLM-5.3 环境/verifier/slime 资料
  -> fixed config + Transformers/vLLM/SGLang DSA source entry
  -> local compaction contract toy
  -> full-weight / hardware / independent benchmark / production acceptance（未核验）
```

因此状态可以从“资料级闭环”升级为**内容专题闭环 + stable/main runtime source evidence**：面试所需的技术主线、权威来源、实现入口和本地教学验收已串起来。但不能把 `glm_moe_dsa` source entry 写成完整权重已加载，不能把 `index_topk=2048` 写成最终可见 token 数，不能把 vLLM/SGLang main/stable 入口写成目标硬件性能或线上 SLO，也不能把 Flash 版 KDA/视觉/EPD 迁移给标准版。

剩余门禁继续单独记账：完整权重与 stable wheel、index/evidence recall、FP8/量化误差、MTP、目标硬件 profiling、5.3 专属 compaction schema/阈值/质量门禁、完整 post-training recipe、独立 benchmark、tool/verifier acceptance 和生产 SLO。

## 2026-09-24 GPT-6 Luna：榜单复验与证据不升级

7890 实时抓取显示 AA `/zh` 首页 `1,783,893` bytes / `3b895865…`，Luna 同日后续详情 `3,974,386` bytes / `9c6376c8…`，DataCurve `268,036` bytes / `14436c31…`。Luna 当前 AA 字段为 Intelligence Index `37.2559686869738`、cost/task `$0.06809498628701058`、median output speed `132.242651126596 tokens/s`。较早同日详情为 `3,974,113` bytes / `8a460332…`、速度 `131.449006457181 tokens/s`；Index/成本稳定，动态字节与速度测量变化均不构成模型 revision 证据。

OpenAI model page `4,019` bytes / `561a86af…` 与既有官方快照相同。DataCurve 仍没有精确 Luna Agent 行，因此不迁移其他 GPT 的工具环境系统结果。Luna 的服务合同、siblings 对照、预算门槛和 harness responsibility 已有正式书册小节与题库/练习；这一轮无新的 Luna 专属技术披露，状态维持 **AA 单榜资料级闭环**，不为状态升级而重复开章。

同日另复核 OpenAI [data residency guide](https://developers.openai.com/api/docs/guides/your-data.md)：EU residency 对 GPT-6 Luna 仅明确适用于 Standard processing 的 Responses 与 Chat Completions。regional storage 不等同 regional processing；system data、Remote MCP 第三方服务数据不自动包含在驻留保证内，非美国地区还需满足适用的 abuse-monitoring/retention 条件。此为服务合同边界，不是 Luna 专属模型技术。

## 2026-09-24 GPT-6 Sol：EU residency 合同与榜单时点

当前 AA `/zh` 首页 `1,783,966` bytes / `0baa28c8…`，GPT-6 Sol 详情 `3,976,802` bytes / `ff0aeaed…`；Index `47.5276426437724`、cost/task `$1.0564240894076389` 未变，median output speed 为 `109.551294011457 tokens/s`，较早记录 `126.038858615917` 只作 provider/采集时点变化。DataCurve 当前仍为 `268,036` bytes / `14436c31…`，无精确 `mini_swe_agent_gpt_6_sol_*` 行。

OpenAI data-residency guide 明确 Sol/Luna EU residency 仅适用于 Standard processing 的 Responses/Chat Completions；regional storage 与 inference regional processing 区分，system data 不在 customer-content guarantee 内，Remote MCP 遵循第三方 policy。`Standard processing` 不等于模型 `reasoning.mode=standard`。这是一条 API/部署 capability contract，不是架构或模型能力证据；Sol 保持 AA 单榜资料级闭环，不迁移其他 GPT 的 Agent 分数。

## 2026-09-24 Qwen3.5-Omni Plus / Flash 的证据分层

Artificial Analysis 已有 `qwen3-5-omni-plus` 与 `qwen3-5-omni-flash` 两条配置；当日刷新后两页哈希分别为 `b2127d33…`、`17bd65b3…`。DataCurve 刷新为 `268,036` bytes / `14436c31…`，未检出精确 `mini_swe_agent_qwen3_5_omni_*` 行，故不迁移其他 Qwen 的 Agent 结果。榜单中的 Plus 价格、Index、速度和上下文只作为第三方/provider 当时字段。

内容专题依据 Qwen Team 技术报告 v2：AuT 为 `6.25 Hz`/约 `160 ms`，Thinker 与 Talker 采用 Hybrid MoE，显式秒级 timestamp 补充 TM-RoPE，ARIA 以样本级整体 speech:text token ratio 限制任一生成前缀；训练报告披露 S1 encoder alignment、约 4T S2 和 262,144-token S3。阿里云当前 API 文档则说明 `qwen3.5-omni-plus`、流式音频输出和 Plus/Flash custom voice 的接口边界。

口径上必须与 Qwen3-Omni 分代：本代 6.25 Hz 不沿用前代 12.5 Hz；报告中的 1 亿小时整体音视频、AuT 4,000 万小时与 Talker 2,000 万小时以上分属不同说明口径，不简单相加。作者 benchmark/延迟不等于本地复现；完整参数、数据 recipe、权重运行、真实 API response、DataCurve 行与线上 SLO 仍待核验。研究笔记和正式章见 [`qwen3.5-omni-source-notes.md`](qwen3.5-omni-source-notes.md) 与第二十一册第 93 章。

## 2026-09-29 Kimi K3：serving 证据分层

本次仍由 AA `kimi-k3` 与 DataCurve `mini_swe_agent_kimi_k3_max` 确认模型身份；vLLM、SGLang、InferenceX 只用于解释该锚点周边的 serving，不作为模型发现入口。adaptive DSpark、cache-pointer invalidation 与 ROCm capability gate 补足了调度、内存生命周期和后端约束知识，但 main commit、合入 PR、stable tag 与 nightly recipe 是不同证据等级。

面试或部署结论至少拆成四层：stable release 中存在实现入口；mutable main/PR 有后续实现或修复；固定 recipe 描述特定硬件/依赖组合；目标机器上的完整权重、数值正确性、恢复与性能验收。本轮只推进前三层的公开资料证据，没有完成最后一层。`synthetic acceptance` 注入外部给定的接受长度并跳过 target verification，只能回答假设接受率下的吞吐，不能和真实 block rejection 的正确性/任务结果混用。完整状态见 [`kimi-k3-source-notes.md`](kimi-k3-source-notes.md)。
