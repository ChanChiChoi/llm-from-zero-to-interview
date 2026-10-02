# 2026 年模型候选盘点

## 2026-10-01 Gemini 4 Argon 新锚点

| canonical anchor | 榜单观察 | 官方/实现增量与边界 | 状态 |
|---|---|---|---|
| `gemini-4-argon` | Artificial Analysis 当前 `/zh` 包含该路由；DataCurve 没有精确 Agent 配置 | Google The Keyword 官方发布页已核验：1M 输出上限、Fairwind 分阶段开放、coding/enterprise/cyber 定位、发布方评测与安全边界；暂无公开模型卡、技术报告、权重、完整 API 或独立复现 | **AA 单榜内容专题闭环（Google 发布方证据）；工程/独立评测待核验** |

不迁移其他 Gemini 的 Agent 结果，不新增正式章节。详见 [`gemini-4-argon-source-notes.md`](gemini-4-argon-source-notes.md)。

## 2026-10-01 GPT-6.1 Sol 新锚点

| canonical anchor | 榜单观察 | 官方/实现增量与边界 | 状态 |
|---|---|---|---|
| `gpt-6-1-sol` | Artificial Analysis 实时 `/zh` 快照包含该 OpenAI 路由及其 effort 变体；DataCurve 当前没有精确 `mini_swe_agent_gpt_6_1_sol_*` | 初次发现阶段尚未取得官方资料；该前置状态已由下方 2026-10-01 官方资料收口条目更新 | **历史发现状态，见下方收口条目** |

## 2026-10-01 GPT-6.1 Sol 官方资料收口

| canonical anchor | 榜单观察 | 官方/实现增量与边界 | 状态 |
|---|---|---|---|
| `gpt-6-1-sol` | AA 实时快照包含 canonical 和 effort 变体；DataCurve 无精确 Agent 行 | OpenAI model page、Reasoning、Agents、Compaction 已核验：Responses tool calling、1.05M/922K/128K、effort/mode 分离、opaque all-turns reasoning replay、encrypted compaction、EU residency | **AA 单榜内容专题闭环 + 官方 API/runtime contract**；参数、架构、训练 recipe、完整权重、硬件、真实 endpoint 和生产验收待核验 |

## 2026-09-29 当前锚点：DeepSeek V4.1-Flash serving 补证

| canonical anchor | 榜单观察 | 官方/实现增量与边界 | 状态 |
|---|---|---|---|
| `deepseek-v4-1-flash` | AA 详情当前快照 3,969,971 bytes / `630f1c9df1abd6ce900d6a7016daf1038d5482c349fd5f4f2a419973ef790a91`；DataCurve 有 `mini_swe_agent_deepseek_v4_flash_max`，不是精确 V4.1 行 | SGLang `v0.5.20` V4-family AMD/HIP：DSpark graph replay metadata、unified-KV/SWA request ring、FP4 indexer schedule fusion；SGLang `main` PR #39313 新增受门禁的 FP8 shared + MXFP4 routed MegaMoE fusion 与 graph-capture side-stream fork | **既有内容专题闭环 + stable/main runtime evidence**；#39313 的 4×B300 accuracy/throughput 只属于 DeepSeek-V4-Flash-0731 发布方测试，不迁移为 V4.1 结果；不能升级为权重加载、目标硬件或生产验收 |

PR 自报的 `+83.6%` full-attention KV token capacity 与 concurrency 4 下 `+15.3%` output throughput 均绑定 AMD/HIP 指定负载；不是本机复现或跨硬件保证。DSA exact top-k PR #37591 注释/测试指向 V3.2/GLM-5.2，未迁移为 V4.1 证据。详见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-29 Claude Sonnet 5.5

| canonical anchor | 榜单观察 | 官方增量与边界 | 状态 |
|---|---|---|---|
| `claude-sonnet-5-5` | AA releaseDate `2026-09-28`，max + Default Fallback，另有 low/medium/high/xhigh；DataCurve 当前无精确 `mini_swe_agent_claude_sonnet_5_5_*` 行 | 官方模型页、What's New、Migration、Prompting 与 148 页 System Card：`between_tools`、thinking block model/prefix/account binding、强制 tool/computer-use 迁移、refusal category/fallback、三阶段 cyber safeguards、发布方 Agent/safety benchmark 条件 | **AA 单榜内容专题闭环 + 官方 API/System Card 证据**；参数、内部架构、完整训练 recipe、独立复现、真实 endpoint 和生产验收仍 `unverified` |

AA 的 Index `55.9779549012591`、约 `138.67 tokens/s`、`327.86s` TTFT 与 `$7.60` task cost 是 max + fallback 的第三方观测，不能和 System Card/DataCurve 混成裸模型排名。详情、页面哈希与边界见 [`claude-sonnet-5.5-source-notes.md`](claude-sonnet-5.5-source-notes.md)。arXiv 精确标题查询无结果；PDF 用仓库 parser 抽取正文但未视觉核验。

补充：2026-09-28 通过 `10.24.27.134:7890` 复验 Artificial Analysis `/zh`、AA release 页面与 DataCurve DeepSWE；八家重点厂商没有新增 canonical 锚点。Qwen3.6-27B 官方博客正文经 Qwen 文章 API 成功恢复，技术边界与快照哈希见下方专题及 [`source-index.md`](source-index.md)。

补充：2026-09-28 沿既有 DeepSeek V4.1-Flash 锚点用 7890 重验 serving release：vLLM `releases/latest` 仍指向 stable `v0.30.0`，Atom feed 已有 `v0.30.1rc0` 预发布项（可见标题为 ROCm/MI355 CI kernel mirror）；SGLang 当日留存的 release/PyPI 元数据仍为 `v0.5.20`。这只更新发布通道状态，不代表 wheel、权重或硬件验收，详见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md#2026-09-28-7890-复验与-vllm-stable预发布边界)。

补充：2026-09-28 本轮两榜快照仍可见既有 `GPT-6 Astra` canonical，DataCurve 含 `mini_swe_agent_gpt_6_astra_{low,medium,high,xhigh,max}` 精确配置；这不是新模型发现。Astra 官方模型页与 family guide 当日复验无漂移；经 7890 成功复取官方定价页并补齐 Standard/Batch/Flex/Fast mode 的 short/long context 费率。专题博客正文复核未发现超出现有章节的新架构或协议，详见 [`gpt-6-astra-source-notes.md`](gpt-6-astra-source-notes.md)。

> 2026-09-09 的候选表保持原样，作为历史发现快照。2026-09-14 的实时首页刷新与新增 13 个结构条目见 [`artificial-analysis-2026-09-14-snapshot.md`](artificial-analysis-2026-09-14-snapshot.md)；2026-09-15/16 对重点锚点的网络复验只更新对应核验段。2026-09-17/18 的排行榜恢复复核见下方补充，不用新快照日期覆盖下表日期。

补充：2026-09-17 重新抓取 Artificial Analysis 中文首页与 DataCurve DeepSWE，三条代理取得逐字节一致快照；八家重点厂商没有新增基础模型候选。该次复核的快照哈希和 GPT-6 Astra 状态见 [`source-index.md`](source-index.md) 与 [`inventory-interpretation.md`](inventory-interpretation.md)。

补充：2026-09-21 继续核验 GPT-6 Astra 的官方模型指南和 Agent 运行时文档；没有新增榜单模型。`async tool calling`、WebSocket steering、misalignment monitoring 和技能/`AGENTS.md` 上下文治理均作为既有 `gpt-6-astra` 的周边协议资料记录，不新增模型条目或迁移其他模型的 Agent 结果。

补充：2026-09-23 使用 `10.24.27.134:7890` 复验两榜和 Astra 详情，AA/DataCurve 页面均 HTTP `200`，没有新的八家重点厂商 canonical 模型。AA Astra 详情为 `3,997,013` bytes / `c4c040e6708555c1965efaf4eecd3d61b4199ffe337d07ce53970150b79103ce`，当前 `max` 指标仍为 Intelligence Index `52.673669395513`、median output speed `58.6808088606541 tokens/s`、1M context；DataCurve 保留 `low/medium/high/xhigh/max` 五个精确配置。model page、Async/Steering/Misalignment 直接文档与上一轮一致；latest-model guide 从 Astra 专属页漂移为 GPT-6 family 迁移页，新增 family capability matrix 和迁移边界。本轮新增的是文档版本证据与失败路径 toy，不是模型版本升级。

采集日期：2026-09-09。来源：[Artificial Analysis](https://artificialanalysis.ai/)。以下日期为榜单字段，尚未逐项经官方核验；不同推理档位保留独立行。此表为候选发现记录，不代表官方发布确认或技术结论。

补充说明：Anthropic 官方模型目录快照还核验到 Claude Haiku 4.5（2025-10-15 发布字段），但它未出现在本次 Artificial Analysis 候选表的采集切片中，因此不伪造榜单日期；该模型作为“官方目录发现”单独记录于 [`claude-haiku-4.5-source-notes.md`](claude-haiku-4.5-source-notes.md) 和来源索引。

补充说明：DeepSeek 官方 API 发布页快照还核验到 DeepSeek-R1-0528（2025/05/28），但它未出现在本次 2026 Artificial Analysis 候选表切片中；该模型作为“官方发布发现”单独记录于 [`deepseek-r1-0528-source-notes.md`](deepseek-r1-0528-source-notes.md)，不补写榜单日期或第三方分数。

## 2026-09-14 实时增量（Artificial Analysis）

以下是实时首页相对 2026-09-09 历史表的模型级增量。成对 slug 合并为一个 canonical 条目；本节不改写下方 273 条历史发现记录。除 DeepSeek V4.1-Flash、K2 Horizon MoVA 36B/A4B、Qwen3.8、Gemini 3.1 Pro Preview、Gemini 3.5 Flash、Gemini 3.7 Flash、Gemini 3.8 Flash、GPT-5.6、GPT-5.5、GPT-5.4、Claude Fable 5、Claude Opus 5、Claude Opus 4.8、Kimi K2.7 Code、Grok 4.5、Grok 4.6、GLM-5 和 GLM-5.2 外，新增项仍停留在候选发现层。

## 2026-09-16 GLM-5.2 官方核验增补

本节升级已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 GLM-5.2；Z.ai 官方文档用于核验和扩展周边技术，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5.2 | Artificial Analysis：`glm-5-2` 的 max/non-reasoning；DataCurve：`mini_swe_agent_glm_5_2_high`、`mini_swe_agent_glm_5_2_max` | [Z.ai GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2)、[文档索引](https://docs.z.ai/llms.txt) | 资料级闭环；1M/128K、长周期 Coding Agent、MCP、缓存和工作流证据已核验；参数、架构、SAO/compaction 原始定义、训练 recipe 和独立复现待核验 |

Artificial Analysis 2026-09-16 复验快照为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；页面第三方 FAQ 记录约 1M context、Intelligence Index `34`、约 `72 tokens/s` 和 `$1.40/$4.40` 每百万 input/output token。DataCurve 快照为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；high/max 分别为 Pass@1 `36.2832%`/`43.7778%`、平均成本约 `$2.8355`/`$3.9199`、平均 Agent steps `121.88`/`129.13`。这些是配置 + `mini-swe-agent` + 工具/环境/verifier 的系统结果，不能写成裸模型能力。

Z.ai 文档快照为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。官方资料确认 1M context、128K max output、thinking、function calling、context caching、structured output、MCP，以及项目级代码库接管、跨文件重构和分阶段验证的长周期工程工作流；“lossless context”仍只按发布方描述记录。

正式专题见第二十一册第 86 章 [`GLM-5.2：IndexShare、MTP 与长轨迹 RL`](../../book-21-transformer-architecture-evolution/chapters/86-glm-5.2-indexshare-mtp与长轨迹rl.md)。完整快照、证据边界和待核验项见 [`glm-5.2-source-notes.md`](glm-5.2-source-notes.md)。

## 2026-09-16 GLM-5 官方核验增补

本节只升级 Artificial Analysis 已精确发现的 GLM-5；DataCurve 当前快照没有精确 `GLM-5` 行，不能把 GLM-5.2/5.3 的 DeepSWE 结果迁移给它。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5 | Artificial Analysis：`glm-5`，`GLM-5 (Reasoning)`，并列 `glm-5-non-reasoning`；DataCurve：无精确同名行 | [Z.ai 博客](https://z.ai/blog/glm-5)、[API 文档](https://docs.z.ai/guides/llm/glm-5)、[模型卡](https://huggingface.co/zai-org/GLM-5)、[技术报告](https://arxiv.org/abs/2602.15763)、[官方 GitHub](https://github.com/zai-org/GLM-5) | 内容专题闭环（AA 单榜）；744B/40B、28.5T、DSA、`slime`、78 层/256 experts/top-8、202K position 和 Agentic Engineering 已核验；DataCurve 精确结果、完整 DSA/训练 recipe、生产 kernel 和独立复现待核验；正式专题为第二十一册第 89 章 |

Artificial Analysis 2026-09-16 复验快照为 `3,577,227` bytes，SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`。页面第三方字段显示约 `744B` total、`40B` active、`200K` context、约 `72.4 tokens/s`、约 `1.34s` TTFT；这些字段与官方模型卡交叉一致的规格才进入事实记录，指数/价格/速度/TTFT仍是第三方配置/provider 测量。

DataCurve 2026-09-16 复验快照为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；页面有 `mini_swe_agent_glm_5_2_*`、`mini_swe_agent_glm_5_3_*` 和 `mini_swe_agent_glm_5_3_flash_max`，没有 `mini_swe_agent_glm_5_*`。不记录 GLM-5 的 DataCurve Pass@1、成本或 Agent steps。

Hugging Face 模型卡/配置与 GLM-5 专属技术报告支持 DSA、MoE、异步 RL 基础设施 `slime`、长周期 Agent RL 和 Agentic Engineering 作为研究主线。配置公开 `GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、前三层 dense、`index_topk=2048`、`q_lora_rank=2048`、`kv_lora_rank=512` 和 `max_position_embeddings=202752`。这些是实现字段，不等于完整训练 recipe。

完整证据、评测设置、负面检索和待核验项见 [`glm-5-source-notes.md`](glm-5-source-notes.md)。2026-09-20 AA 快照为 `3,811,809` bytes、SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，仍无精确 `mini_swe_agent_glm_5_*` 行。Z.ai 博客资源本轮为 `145,189` bytes、SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`；文档 Markdown 端点返回代理 503，只记为线路失败。正式专题见第二十一册第 89 章 [`GLM-5：DSA、MoE、slime 与 Agentic Engineering`](../../book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)。

## 2026-09-15 GLM-5.3-Flash 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 GLM-5.3-Flash；Z.ai 官方资料用于核验和扩展锚点周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5.3-Flash | Artificial Analysis：`glm-5-3-flash`，`max`，页面 `releaseDate` `2026-08-26`；DataCurve：`mini_swe_agent_glm_5_3_flash_max`，`max` | [Z.ai 模型文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)、[Z.ai 官方博客](https://z.ai/blog/glm-5.3-flash)、[固定 revision 模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)、[配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) | 内容专题闭环 + SGLang `v0.5.20` stable/main 与 vLLM `v0.30.0` stable/fixed-main runtime source evidence；320B/18B、45 层、34 linear/11 sparse、288 routed/top-8/1 shared、IndexPool、mHC、原生视觉 coding loop、EPD serving、thinking/tool streaming 已核验；完整 kernel、训练 recipe、硬件 profiling 和独立复现待核验 |

Artificial Analysis 当前主配置为 Intelligence Index `41.907366113455`、约 `114.22108687545 tokens/s`、约 `2.45458272199994s` TTFT、1M context、约 `$0.15/$0.50/$0.026` 每百万 token；DataCurve 为 284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、约 `$0.2409818562`/task、约 `72829.77` output tokens、约 `122.89` Agent steps。两组数字均绑定不同的 provider/configuration 或 `mini-swe-agent`/工具/环境/verifier，不能拼成裸模型排名。

官方配置支持 `Glm5NextForConditionalGeneration`、320B total/18B activated、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts、top-8、1 shared、1,048,576 position、`index_kpool=4`、`index_topk=2048`、`mhc=true`、`hc_mult=4` 和 20 次 Sinkhorn；视觉配置给出 24 层、448 image、patch 14、temporal patch 2、输出投影 4096。官方博客/文档还描述 30T multimodal corpus、visual self-judgment/test-time improvement、SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split、EPD 和约 3× serving 自报。

完整快照哈希、API 字段、评测边界和待核验项见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)；正式专题见第二十一册第 84 章 [`glm-5.3-flash混合注意力与视觉闭环.md`](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)。

2026-09-22 upstream 对照补证（截至当日快照）：SGLang `v0.5.20` 的固定 tag 已有 `glm5_next.py`/配置入口，SGLang `main` 继续演进 projection/KDA/mHC/量化路径；vLLM `main` 已拆出 `glm5next` 的 indexer tail cache、KDA state 和 MTP runtime。vLLM `v0.29.0` tag 的 tree 快照未发现该专属路径。2026-09-24 又确认 v0.30.0 stable source 已有多项实现并与固定 main 做了逐文件对照，详见下方 dated entry。任一源码入口都不等于完整权重、目标硬件 profiling、状态恢复或生产 acceptance。

## 2026-09-15 DeepSeek V4 Flash Vision 官方核验增补（2026-09-24 复验）

本节只升级已出现在 Artificial Analysis 的 `deepseek-v4-flash-vision`；DataCurve DeepSWE 当前快照没有该行，不能把 `deepseek-v4-flash` 或 `deepseek-v4-pro` 的配置结果迁移给它。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| DeepSeek V4 Flash Vision | Artificial Analysis：`deepseek-v4-flash-vision`，`max`，页面 `releaseDate` `2026-08-21`；DataCurve：无同名行 | [V4 Flash Vision Exp 公告](https://api-docs.deepseek.com/news/news260821)、[Vision guide](https://api-docs.deepseek.com/guides/vision)、[Files API](https://api-docs.deepseek.com/guides/files_api)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[当前价格页](https://api-docs.deepseek.com/quick_start/pricing) | 资料级闭环；历史实验多模态 API、当前 alias 路由、图像 detail/token、Files `file_id`、Responses 图像工具回灌和成本/评测边界已核验；专属视觉架构、训练 recipe、独立技术报告和 DataCurve 结果待核验 |

2026-09-15 Artificial Analysis 初始快照字段为 Intelligence Index `35.0122378035969`、约 `215.179167697513 tokens/s`、约 `1.29855545700002s` TTFT、1M context、284/13 目录参数和约 `$0.44/$1.32/$0.014` 每百万 token；均绑定第三方页面配置、provider 和测量时间。同期官方 Quick Start 已说明旧的 `deepseek-v4-flash-vision-exp` alias 由 `DeepSeek-V4.1-Flash` 服务；当前服务价格与历史 AA 字段不能直接拼接。

2026-09-24 经 `10.24.27.134:7890` 复验：AA 详情 HTTP 200，`3,967,107` bytes / SHA-256 `a5e5259ee84eac5aa88915dd6436ba155e265ede940ab663b52c4a681ae43ed7`；Index `34.8390628035969`、median output speed `217.762710468086 tokens/s`、TTFC `0.970116780000126s`、cost/task `$0.314372044499279`。9/15 的 `35.0122378035969`、`215.179167697513 tokens/s` 和 `1.29855545700002s` 保留为历史 provider/测量值，不解释为模型 revision。DataCurve 当前快照 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，仍无精确 `mini_swe_agent_deepseek_v4_flash_vision_*` 行。

9/24 官方 Quick Start、Pricing 和 9/10 V4.1-Flash 发布页确认：`deepseek-v4-flash-vision-exp` 旧名仍接受但对应模型已 retired，请求暂时由 V4.1-Flash 服务并按 Flash 当前峰谷价格计费；新服务目标发布页称其来自新架构系列且具备 native visual understanding，这不反向披露旧 Vision-Exp 架构。价格页列 1M context、384K max output。当前 Vision guide 仍写单图 resize 后上限 1024 tokens，并补充 600 张/请求、单图与合计大小、8192/4096 px 尺寸限制。没有真实 API probe，不声称观测到实际 `response.model`。完整页面快照和解释见研究笔记；正式专题见第二十一册第 85 章 [`deepseek-v4-flash-vision多模态api与路由账本.md`](../../book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)。

## 2026-09-14 Qwen3.8 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 Qwen3.8 条目；官方资料用于核验和扩展技术，不是新的候选发现入口。

| 基础模型/服务 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Qwen3.8-27B | Artificial Analysis，2026-08-14 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense native vision-language；64 层；3 GDN + 1 Gated Attention；thinking 可按请求关闭 |
| Qwen3.8-2.4T-A95B | Artificial Analysis，2026-08-12 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2.4T total/95B active；92 层；512 experts，10 routed + 1 shared；text-only；thinking 强制开启 |
| Qwen3.8-Flash-Next | Artificial Analysis，2026-08-26 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | 125B/6B active；51B N-gram；GDN + QSA；四分支 GR；Muon/AdamW；实验性开源架构预览 |
| Qwen3.8 Max | Artificial Analysis，2026-08-03；DataCurve DeepSWE 当前快照亦有 `qwen3.8-max` | [Qwen Cloud](https://www.qwencloud.com/models/qwen3.8-max)、A95B 模型卡 | 托管版本；官方说明基于 A95B，并增加视觉、非 thinking、默认 1M 和内置工具等产品能力，不当作独立 open checkpoint |

具体页面、快照哈希、来源优先级和待核验项见 [`qwen3.8-source-notes.md`](qwen3.8-source-notes.md)。

## 2026-09-21 Qwen3.8 Max (0902) runtime recheck

本节只升级已经在 Artificial Analysis 与 DataCurve DeepSWE 中出现的 Qwen3.8 Max；Qwen Cloud 和开发者文档用于核验服务 revision 与 API 周边技术，不作为新的候选发现入口。

| 基础模型/服务 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Qwen3.8 Max (0902) | Artificial Analysis canonical `qwen3-8-max` 当前标题为 `Qwen3.8 Max (0902)`，release slug `qwen3-8-max-0902`；DataCurve 只有泛化 `mini_swe_agent_qwen3_8_max_xhigh` | [Qwen3.8-Max-0902](https://www.qwencloud.com/models/qwen3.8-max-0902)、[Thinking](https://docs.qwencloud.com/developer-guides/text-generation/thinking)、[Function Calling](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)、[Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)、[Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits) | 资料级闭环；官方确认 upgraded snapshot、alias、1M/991K/983K/131K 预算、reasoning effort、thinking/tool-choice、cache、endpoint migration 和 hosted quota 边界；不写成新的 open checkpoint，DataCurve 泛化结果不迁移到 0902 |

Qwen Cloud 页面给出 `qwen3.8-max-2026-09-02` alias，并将 0902 描述为 `qwen3.8-max` 的 upgraded snapshot；2026-09-21 `last-modified` 为 `2026-09-21 11:01:05`。当前 Artificial Analysis 详情快照为 `3,835,865` bytes、SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`，第三方字段为 Intelligence Index `45.4152084980521`、约 `37.275289705095 tokens/s`、约 `$5.408509428374016`/Intelligence Index task、约 `984K` context；AA `/zh` 首页为 `1,777,588` bytes、SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`。这些仍是第三方目录/测量字段。

官方字段还包括 `low/medium/xhigh`（默认 `xhigh`）和与 `thinking_budget` 的互斥关系、thinking 下 `tool_choice` 只能使用 `auto`/`none`、`MultiModalConversation`，以及 explicit/implicit/session cache（最小 1,024 tokens）。2026-09-23 经 `7890` 重新核验 Context Cache 正文：explicit marker 只能是 `ephemeral`，单请求最多 4 个且只取最后 4 个，超过 20 个 content block 的 backward lookback 可能 miss；explicit/session TTL 为 5 分钟且命中刷新，implicit 无固定 TTL且不保证命中；session 使用 `x-dashscope-session-cache: enable`、Responses API 和 `previous_response_id`，命中数从 `usage.input_tokens_details.cached_tokens` 读取。文档示例已迁移到 `maas.qwencloudapi.com`，应把 endpoint 作为 provider adapter 版本字段记录；这不是模型能力升级。产品页关于 coding、长周期 autonomous development、多工具 Agent 和视觉理解的描述按服务定位记录，不升级为内部架构事实。

Dynamic Rate Limiting 文档确认 account+model 聚合、workspace override、按自然月调整的 TPM tier 和 soft limit；`qwen3.8-max-0902` 保证 TPM 为 `1,500,000 / 1,500,000 / 1,500,000`。生产账本应另记 guaranteed TPM、observed TPM、429、`Retry-After`、queue latency、account、workspace 和 revision；保证 TPM 不是模型吞吐、GPU capacity 或 benchmark 结果。当前产品价格字段还包括 input `$2`、output `$6`、implicit cache `$0.25`、explicit cache creation `$2.50` 和 explicit cache read `$0.17`/百万 token。

2026-09-18 的 Artificial Analysis 历史详情快照为 `3,607,154` bytes、SHA-256 `e9152a5d81063cbeb45fb21ba621d7eb7da05073235b611f27a344f38c6288ae`，第三方 Intelligence Index `45.4354834980521`、context 约 `984K`。DataCurve 泛化行是 258/449、Pass@1 `57.4610%`、Pass@4 `83.1858%`、约 `$3.7291`/task、约 `95,075` output tokens、约 `111.34` Agent steps、4 runs；没有 0902 精确行，不能把该系统结果写成 0902 revision 的单项能力。

Context Cache 文档的典型相对计费为 explicit/session 创建 `125%`、命中 `10%`，implicit 创建 `100%`、命中 `20%`；产品页 `$2/$6/$0.25/$2.50/$0.17` 价格字段和 usage 需另行核对。完整证据和待核验项见 [`qwen3.8-max-0902-source-notes.md`](qwen3.8-max-0902-source-notes.md)。零依赖审计器 [`qwen_max0902_cache_contract_audit.py`](code/qwen_max0902_cache_contract_audit.py) 只提供 `local_protocol_toy` 证据。本轮不新增重复架构章节，内容扩展进入第二十一册第 83 章、第二十四册工具 serving 章节及书系配套文件。

## 2026-09-24 Qwen3.8-Flash-Next 固定版本 serving 源码补证

本次继续沿 Artificial Analysis 已发现的 `Qwen3.8-Flash-Next` 锚点；Qwen README、vLLM/SGLang 源码和 recipe 只用于技术核验，不新增模型候选。

| 固定来源 | 补充确认 | 状态边界 |
|---|---|---|
| [vLLM `v0.30.0` Qwen4Exp](https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/models/qwen4_exp) 与 [registry](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/registry.py) | 独立 `Qwen4ExpForConditionalGeneration`/MTP 注册；QSA、PLE/N-gram、Gated Residual、MTP 与测试位于专属路径。当前 NVIDIA QSA 激活/QKV 与主 KV 路径要求 BF16；压缩 indexer-key cache 可用 BF16 或 FP8 E4M3；不支持 KV quantization/context parallelism。 | 固定 release 源码与测试文件已核对；wheel、完整权重、GPU test 和线上实例未验收。 |
| [SGLang `v0.5.20` source](https://github.com/sgl-project/sglang/tree/v0.5.20/python/sglang/srt/models) | 有 Qwen4Exp/QSA/MTP/PLE 实现；符合特定 graph-capture 与 token 阈值时，QSA indexer 可与当前 stream 的 Q/K/V 准备并行，之后再启动 attention。PLE 有 pinned-host 与 file-backed sparse mmap 路径；源码注释约 47.7 GiB FP8 PLE table。 | indexer 与最终 attention kernel 并不重叠；PLE 组合对 two-batch overlap、N-gram speculation 和 speculative top-k 有限制，file backend 依赖特定统一内存能力。 |
| [vLLM Qwen recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next) | recipe 记录 H100/H200、GB200/GB300、MI355X 配置，以及指定 4×H200 TP4/16,384 max sequence 的 FP8 验证条件；提醒 H100 80GB/GPU 对 51B PLE/N-gram table 余量不足。 | 发布方 recipe 条件，不是本地复现或通用吞吐/容量保证。 |

研究笔记和正式教学落点分别见 [`qwen3.8-source-notes.md`](qwen3.8-source-notes.md) 与第二十一册第 83 章。内容专题及固定源码路径已形成资料级闭环；硬件 profiling、端到端延迟/吞吐、完整权重、跨后端行为和生产验收仍待核验。

## 2026-09-15 Gemini 3.7 Flash 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Gemini 3.7 Flash；Google 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.7 Flash | Artificial Analysis：2026-08-13 的 high/medium/low；DataCurve DeepSWE v1.1：low/medium/high | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)、[Model Card](https://deepmind.google/models/model-cards/gemini-3-7-flash/)、[评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf) | 资料级闭环；1M 输入、64K 输出、low/medium/high thinking、agentic video、Interactions API、工具组合和 stateless/stateful signature 回放已核验；独立架构/训练报告待核验 |

Artificial Analysis high 详情页本轮复验为 3,605,373 bytes、SHA-256 `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c`，第三方 Intelligence Index `39.4295316404896`、约 `292.3389 tokens/s`、约 `10.2161s` input/TTFT 字段和 1M context；DataCurve 三档结果为 low/medium/high Pass@1 `53.7611%/65.4867%/65.2655%`，平均成本约 `$1.8323/$2.0251/$2.1763`。两者均绑定配置、provider、harness、任务集、工具、环境和 verifier，不能拼成裸模型排名。

Model Card 只披露核心 reasoning foundation 的算法改进、agentic video understanding 和可调 thinking，并将架构、训练数据、软硬件资料指向 Gemini 3.6 Flash Model Card。Interactions API、tool context circulation、加密 signature、built-in/custom tool 责任分层，以及 agentic video 的 `processing_call`/`processing_result` 是主要面试线索。

arXiv 精确标题检索返回 0 个结果；全文检索的 3 篇命中只是外部使用/评测论文，不是 Gemini 3.7 专属技术报告。详细来源、快照哈希、外部论文标题和证据边界见 [`gemini-3.7-flash-source-notes.md`](gemini-3.7-flash-source-notes.md)。当前不新增独立正式章节。

## 2026-09-15 Gemini 3.6 Flash 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Gemini 3.6 Flash；Google 官方资料用于核验和扩展锚点周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.6 Flash | Artificial Analysis：`high`，页面 `releaseDate` `2026-07-21`；DataCurve：`mini_swe_agent_gemini_3_6_flash_high`、`reasoning_effort: high` | [API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-6-flash/)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)、[视频理解](https://ai.google.dev/gemini-api/docs/video-understanding) | 资料级闭环；1M 输入、64K 输出、默认 medium 与 minimal/low/medium/high thinking、agentic video、Interactions、tool signature/circulation、Computer Use Preview 和 Model Card benchmark/safety 已核验；独立架构/训练报告待核验 |

Artificial Analysis 当前详情页字段为 Intelligence Index `34.3395678920714`、约 `192.7337 tokens/s`、TTFT `19.2239s`、1M context、约 `$0.15/$0.75/$3.75` cache-hit/input/output 每百万 token；provider benchmark 页当前只有 `Google AI Studio`。DataCurve 原始行是 211/452、Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095`、平均输出 `95,844.86` token、平均 Agent steps `116.73`。两组数字均绑定各自 provider/configuration 或 `mini-swe-agent`/工具/环境/verifier，不能拼成裸模型排名。

Model Card 明确写出 Gemini 3.6 Flash based on Gemini 3.5 Flash，并把架构、训练数据、数据处理、硬件和软件资料指向 Gemini 3.5 Flash Model Card；公开 benchmark 包括 SWE-Bench Pro `58.7%`、DeepSWE v1.1 `49%`、Terminal-Bench 2.1 `78.0%`、GDPVal-AA v2 `1421`、OSWorld-Verified `83.0%`、GDM-MRCR v2 128K `91.8%`/1M `54.0%`。这些是 Google 发布方设置，不替代 DataCurve 或独立复现。

arXiv 精确标题检索返回 0 篇；全文检索返回 9 篇外部使用/评测论文，不是 Gemini 3.6 专属技术报告。完整论文标题、Model Card 安全表、快照哈希和待核验项见 [`gemini-3.6-flash-source-notes.md`](gemini-3.6-flash-source-notes.md)。当前不新增独立正式章节。

## 2026-09-15 Gemini 3.5 Flash 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Gemini 3.5 Flash；Google 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.5 Flash | Artificial Analysis：`high`，页面 `releaseDate` `2026-05-19`，另有 `medium`/`minimal`；DataCurve：`mini_swe_agent_gemini_3_5_flash_high`、`reasoning_effort: high` | [API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)、[What's New](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash/)、[Gemini 3 Flash Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf) | 资料级闭环；默认 medium、`minimal/low/medium/high` thinking、thought preservation、Interactions、tool context circulation、4,096-token implicit caching、Computer Use 及 prompt-injection detection 已核验；架构/训练/软硬件被官方指向 Gemini 3 Flash，独立参数和训练报告待核验 |

Artificial Analysis 详情页第三方字段为 Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、约 `18.3631s` TTFT、1M context 和约 `$1.5625`/task；当前目录把条目标为 deprecated 并指向 Gemini 3.6，只表示 Artificial Analysis 的历史 benchmark 状态。DataCurve 原始行是 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均输出 `75,730.19` token、平均 Agent steps `105.30`。两组数字均绑定各自 provider/configuration 或 `mini-swe-agent`/工具/环境/verifier，不能拼成裸模型排名。

What's New 页面确认 3.5 将 Gemini 3 Flash Preview 的默认 effort 从 high 调整为 medium，并开启跨轮 thought preservation；Interactions、tool context circulation、signature/id 回放、implicit caching 和 Computer Use prompt-injection detection 是主要面试线索。当前视频文档的 agentic processing 列表明确列出 3.5 Flash-Lite 而非 3.5 Flash，因此不把 3.6 的 agentic video 结论迁移给 3.5 Flash。

Model Card 的 benchmark、安全和 Frontier Safety 结果只作为 Google 发布方证据；其 architecture、training dataset、data processing、hardware 和 software 均指向 Gemini 3 Flash Model Card。arXiv 精确标题返回 0 篇，全文检索返回 30 篇外部使用/评测论文，没有找到独立 3.5 技术报告。详细来源、快照哈希、评测表和待核验项见 [`gemini-3.5-flash-source-notes.md`](gemini-3.5-flash-source-notes.md)。

## 2026-09-16 Gemini 3.1 Pro Preview 官方核验增补

本节只升级已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `Gemini 3.1 Pro Preview`；Google 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.1 Pro Preview | Artificial Analysis：`gemini-3-1-pro-preview`，页面 `releaseDate` `2026-02-19`；DataCurve：`mini_swe_agent_gemini_3_1_pro_preview_high`、`reasoning_effort: high` | [API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)、[Release Notes](https://ai.google.dev/gemini-api/docs/changelog)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)、[评测方法](https://deepmind.google/models/evals-methodology/gemini-3-1-pro) | 资料级闭环；1M/65K、多模态、thinking、`customtools` endpoint、thought/tool signature 回放、tool context circulation、评测/安全边界已核验；独立架构/训练报告待核验 |

Artificial Analysis 2026-09-16 复验字段为 Intelligence Index `30.3596656132261`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT、1M context、约 `$2/$12/$0.20` input/output/cache-hit 每百万 token；DataCurve 原始行是 53/452、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本 `$2.1434`、平均输出 `28,368.88` token、平均 Agent steps `75.56`。两组数字均绑定不同配置/provider 或 `mini-swe-agent`/工具/环境/verifier，不能拼成裸模型排名。

Google Model Card 确认 2026-02-19 发布、原生多模态 reasoning、1M 输入和 64K 输出，并明确 `Gemini 3.1 Pro is based on Gemini 3 Pro`；架构、训练数据、数据处理、硬件和软件资料均指向 Gemini 3 Pro Model Card。API 页还公开 `gemini-3.1-pro-preview-customtools`，面向 bash 与自定义工具混用时的工具优先级优化；它是 endpoint variant，不新增独立基础模型。

Thinking、thought signature、tool context circulation、stateful/stateless 回放、1M long context、context caching 和评测/Frontier Safety 证据已整理在 [`gemini-3.1-pro-preview-source-notes.md`](gemini-3.1-pro-preview-source-notes.md)。arXiv 标题精确检索为 0，全文检索为 198 篇外部使用/评测论文，没有找到 Gemini 3.1 Pro 专属技术报告；当前不新增独立正式架构章节。

## 2026-09-14 Gemini 3.8 Flash 官方核验增补

本节只升级已经出现在两个排行榜的 Gemini 3.8 Flash；Google 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.8 Flash | Artificial Analysis：2026-09-02 的 high/medium/low；DataCurve DeepSWE v1.1：high | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)、[发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/)、[评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf) | **内容专题闭环**，正式落点为第二十册第 23 章；GA/stable、1M 输入、64K 输出、多模态输入、thinking、工具组合与长任务 Agent 文档已核验；固定 Python SDK schema 不要求自定义 function signature，但因 `extra="allow"` 且文档口径不一，真实 endpoint 字段行为仍未裁决；3.8 独立架构/训练报告仍待核验 |

具体来源、快照哈希、Interactions API、thought signature、Computer Use、File Search、URL Context、Code Execution 和评测边界见 [`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。

2026-09-23 官方文档复验补充：`gemini-3.8-flash` stable alias 的 `1,048,576/65,536` 输入/输出上限、默认 `medium` 与合法 `low/medium/high` 已按当前 Thinking 表固定；`max_output_tokens` 包含 thought 与 visible output，触顶可返回 `incomplete`。Interactions 默认 `store=true`，付费/免费保留 `55/1` 天，`store=false` 不能继续 `previous_interaction_id`，工具、system instruction 和 generation config 不由 history 自动继承；tool combination 的 `id`/signature、server/client-side 工具责任和 `validated` mode 已记录。Thinking 与 Tool combination 页面对标准 function call 是否带 signature 的范围描述不完全一致，保留为待 capability probe 的文档冲突，不静默合并。新增标准库 toy [`gemini_interactions_replay_demo.py`](code/gemini_interactions_replay_demo.py)，证据等级为 local protocol toy。该段是 2026-09-23 时点记录；此后已新增第二十册第 23 章并于 2026-09-24 补入固定 SDK schema，当前状态为**内容专题闭环**，真实 endpoint 字段行为和独立架构/训练报告仍待核验。

2026-09-28 7890 复核：当前 overview 指向 [`Interactions API reference`](https://ai.google.dev/api/interactions-api)，此前记录的 `/api/interactions` 本轮返回 404。新 reference 和 2026-09-26 Google Gen AI Python SDK commit 都使用 `function_call.id` → `function_result.call_id`，自定义 function schema 未声明 `signature`；reference/SDK 的 `ThoughtStep.signature` 为 optional，与 Thinking prose“必有”相冲突。Tool combination prose 仍使用 `function_response.id` 并覆盖称 tool call/result 有 signature。schema 更支持 Thinking 的窄口径，但 extra fields 可保留且没有真实 API probe，因此 endpoint 行为仍未裁决。具体快照及类文件 SHA 见研究笔记 §18。

2026-09-28 补充 SDK namespace 证据：签名字段结论特指 Google Gen AI 的 Interactions client `google/genai/_gaos/interactions.py` 及 `_gaos/types/interactions/*`。其 `_gaos/types/basemodel.py` 是 `extra="allow"`；另一套 `google/genai/_common.py` BaseModel 为 `extra="forbid"`，不能泛化成“整个 Python SDK 都允许/拒绝 extras”。`ThoughtStep.signature` 在生成 schema 中 optional，而 replay toy 的默认要求是应用侧保守策略，不是 wire schema 必填；完整 pinned-path/hash 见 Gemini 3.8 来源笔记 §19。没有真实 endpoint probe。

## 2026-09-14 GPT-5.6 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.6 条目；OpenAI 官方资料用于核验和扩展 reasoning、缓存、工具与 Agent 运行时，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.6 Sol | Artificial Analysis，2026-07-09 的多档 effort；DataCurve DeepSWE v1.1：`max` | [Sol 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-sol.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) | 旗舰档；1.05M context、922K maximum input、128K maximum output、`standard/pro`、`all_turns` 和工具支持已核验 |
| GPT-5.6 Terra | Artificial Analysis，2026-07-09 的多档 effort；DataCurve DeepSWE 当前快照含 `mini_swe_agent_gpt_5_6_terra_{low,medium,high,xhigh,max}` 五行 | [Terra 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md)；精确配置结果与证据边界见 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md#15-2026-09-28-gpt-5-6-terra-的-datacurve-配置行补核) | 智能/成本平衡档；共同 GPT-5.6 接口字段与 DataCurve 配置级 Agent 结果已核验 |
| GPT-5.6 Luna | Artificial Analysis，2026-07-09 的多档 effort；DataCurve DeepSWE v1.1：`max` | [Luna 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) | 高并发/成本敏感档；共同 GPT-5.6 接口字段已核验 |

推理配置仍按一个 GPT-5.6 家族归并；`Non-reasoning` 是榜单标签，不自动等于 API 的 `none`。完整的 persisted reasoning、prompt caching、compaction、tool search、harness 和负面证据见 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## 2026-09-15 GPT-5.5 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.5 条目；OpenAI 官方资料用于核验和扩展推理、视觉、工具、状态和缓存技术，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.5 | Artificial Analysis，2026-04-23 的多档 effort；DataCurve DeepSWE v1.1：`xhigh` | [模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md)、[GPT-5.5 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.5.md) | 旗舰档；1.05M context、128K max output、`none`--`xhigh`、Responses/工具/`phase`/图像 detail/缓存差异已核验 |
| GPT-5.5 Pro | Artificial Analysis，2026-04-23 的 `xhigh`；DataCurve 当前快照无主表行 | [模型页](https://developers.openai.com/api/docs/models/gpt-5.5-pro.md) | 使用更多计算；Responses/Batch、`medium`--`xhigh`、background mode 和无 cached input discount 已核验 |

GPT-5.5/Pro 按同一产品家族归并，但 `gpt-5.5-pro` 的端点、effort、价格和缓存计费不能直接套用基础 `gpt-5.5`。Artificial Analysis 的 `GPT-5.5 Instant` 仍只保留为榜单级关联配置：已重新抓取 May/June 详情；2026-09-28 官方目录进一步确认 `chat-latest` 只是 ChatGPT 当前 Instant 的动态 API alias、未披露底层 snapshot，精确 `gpt-5.5-instant` 路径仍为 404。不能将它们并入 `gpt-5.5`，也不能假定它们对应 `chat-latest`。完整的 `phase` 回放、outcome-first prompting、tool search、compaction、5.5/5.6 缓存对比和负面证据见 [`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。

## 2026-09-20 GPT-5.5 Instant 榜单配置复核

本节只记录两个排行榜中已经出现的 GPT-5.5 Instant 条目；OpenAI 官方 `gpt-5.5` 资料只用于身份对照，不作为 Instant 的专属能力证明。

| 榜单配置 | 榜单发现 | 官方身份核验 | 当前状态 |
|---|---|---|---|
| GPT-5.5 Instant (May 2026) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-5-instant-05-26)，2026-05-05；AA Index `22.6863693000789`（页面约 23），400K context，页面 deprecated；详情 `3,548,855` bytes / `ed410b6cecbd8eb1dcaf65715547771bcb7435e444f6c5ba7ce815e16614e51e` | [GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md)确认的是 `gpt-5.5`，不是 Instant ID | 榜单级关联；不与 June 或 `gpt-5.5` 合并 |
| GPT-5.5 Instant (June 2026) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-5-instant-06-26)，2026-06-25；AA Index `26.0135173401368`（页面约 26），约 130.9925 output tokens/s，400K context，未 deprecated；详情 `3,847,333` bytes / `9c6fa94b9c42a89eb7c8f59bad08a30c168bf0860971fbb666c4be47e6a05599` | 精确 [`gpt-5.5-instant` 页面](https://developers.openai.com/api/docs/models/gpt-5.5-instant.md)仍 HTTP 404；官方 [Chat Latest](https://developers.openai.com/api/docs/models/chat-latest.md) 只声明动态 Instant alias，不披露当前 snapshot | 榜单级关联配置 + 官方身份未解析；不与 `gpt-5.5` 或 `chat-latest` 合并 |

两个 AA 页面都显示 text/image → text、reasoning、知识截止 2025-08-31 和 `$5/M` input、`$30/M` output；这些是第三方目录/测量或其 API 价格引用。DataCurve 当前没有 `gpt_5_5_instant` 精确 `mini_swe_agent` 行，不能迁移 `gpt-5.5` xhigh 的 67% Pass@1、成本、输出 token 或 Agent steps。完整负面证据和后续顺序见 [`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。

### 2026-09-23 当前时点复核

7890 线路重新取得 June 详情 `3,993,729` bytes / SHA-256 `960e416df7115bfadab84a915b169662e2100d47aef0ffd1a2f421f8e3894139`。`releaseDate=2026-06-25`、`deprecated=false`、400K context 和 Index `26.0135173401368` 保持不变；当前 provider 速度 `123.805190515587 tokens/s`、TTFC `1.06285215s` 与 cost/task `$0.6914853678439488` 仅作为采集时点观察值。精确 `gpt-5.5-instant` 官方页面仍返回 HTTP 404，DataCurve 仍无精确 Instant 行。本条目继续保持“榜单级关联配置 + 官方身份负证据”，不把 Instant 当作 `gpt-5.5` 的新 checkpoint，也不阻塞其他锚点。

### 2026-09-28 官方 Instant alias 复核

通过 `10.24.27.134:7890` 取得 [OpenAI Models](https://developers.openai.com/api/docs/models.md)（12,055 bytes / `ad6adde919a6b4f2e92b03acb891366d2ded0e078229f406a519722c71a406c3`）、[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5.md)（4,132 / `fdb0fc8fe9ea7f276716c2a5b49b902e7e676ff262bb8adfdf2d5b8911bad3ba`）、[Chat Latest](https://developers.openai.com/api/docs/models/chat-latest.md)（3,294 / `b6b9dc5e8a6c720641cfc8ada231e311d66ebbf2ba5ac14d7d480535679079cc`）和 [GPT-5.3 Chat](https://developers.openai.com/api/docs/models/gpt-5.3-chat-latest.md)（3,206 / `df599cfb2ef17e5de61e912684cefad0db9101f6dccd91ff15d72e5528cda6d2`）。GPT-5.5 API ID/snapshot 是 `gpt-5.5` / `gpt-5.5-2026-04-23`；`chat-latest` 是 ChatGPT 当前 Instant 的动态 alias，官方未注明当前底层模型或 snapshot；GPT-5.3 Chat 曾明确指向 GPT-5.3 Instant，但已 deprecated。没有任何一项证据证明 AA 的 GPT-5.5 Instant 等同于 `chat-latest`。精确 [`gpt-5.5-instant.md`](https://developers.openai.com/api/docs/models/gpt-5.5-instant.md) 再次 HTTP 404（9 bytes / `e3ebaa16dd9d9b9fc107c42183fb6cf9d22927e1af03dbbdfa0ccc38e4e4ac31`）。没有真实 API probe；研究状态仍是历史榜单关联配置、官方身份未解析。

## 2026-09-15 GPT-5.4 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.4 条目；OpenAI 官方资料用于核验和扩展推理、工具、长上下文、状态和 compaction 技术，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.4 | Artificial Analysis：2026-03-05 的 `xhigh`、`low`、`Non-reasoning`；DataCurve DeepSWE v1.1：`xhigh` | [模型页](https://developers.openai.com/api/docs/models/gpt-5.4.md)、[GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 资料级闭环；1.05M context、128K max output、`none`--`xhigh`、tool search、computer use、custom tools/CFG、`phase`、compaction 和 Responses 状态已核验 |
| GPT-5.4 Pro | Artificial Analysis：2026-03-05 的 `xhigh`；DataCurve 当前快照无主表行 | [GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 与 GPT-5.4 同家族的关联服务档位；本轮未单独建立 Pro 资料闭环 |
| GPT-5.4 mini/nano | Artificial Analysis：2026-03-17 的精确 mini/nano 配置；DataCurve 当前只有 `mini_swe_agent_gpt_5_4_xhigh` 的 base 行，无 mini/nano 精确模型 ID | [GPT-5.4 mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md)、[GPT-5.4 nano 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-nano.md)、[GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 资料级闭环（AA 单榜关联档位）；各自 snapshot、400K/272K/128K、effort、定位、价格和工具清单已单独核验，不迁移 base Agent 成绩 |

Artificial Analysis 主配置 `GPT-5.4 (xhigh)` 当前页面标记 deprecated 并指向 GPT-5.5，第三方字段为 Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT 和 1M context；DataCurve GPT-5.4 xhigh 为 234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本约 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` Agent steps。两榜单数字绑定不同配置、provider、harness、任务集、工具、环境和 verifier，不能互相拼接为裸模型排名。

完整证据、快照哈希、面试主线和未公开内容见 [`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md)。

## 2026-09-20 GPT-5.4 mini/nano 独立核验

本节只升级已经出现在 Artificial Analysis 的 GPT-5.4 mini/nano 条目；OpenAI 官方资料用于核验精确 sibling 的 API 契约和周边技术，不作为新的模型发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.4 mini | Artificial Analysis：[`gpt-5-4-mini`](https://artificialanalysis.ai/models/gpt-5-4-mini)，2026-03-17，xhigh；DataCurve：无 `mini_swe_agent_gpt_5_4_mini_*` 精确行 | [GPT-5.4 mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md)、[GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | AA 单榜资料级闭环；高吞吐 coding、computer use、Agent workflow、400K/272K/128K、`none`--`xhigh`、价格和工具清单已核验 |
| GPT-5.4 nano | Artificial Analysis：[`gpt-5-4-nano`](https://artificialanalysis.ai/models/gpt-5-4-nano)，2026-03-17，xhigh；DataCurve：无 `mini_swe_agent_gpt_5_4_nano_*` 精确行 | [GPT-5.4 nano 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-nano.md)、[GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | AA 单榜资料级闭环；classification、extraction、ranking、窄任务 sub-agent、400K/272K/128K、`none`--`xhigh`、价格和工具清单已核验 |

mini 详情快照为 `3,857,350` bytes、SHA-256 `a01a6da4a38077bc6309333d509bf303f8b1beb36729fd94f79fe7d3a8ffe0ee`；nano 为 `3,859,704` bytes、SHA-256 `2e6ddfefc9a3483437f0fe2648ce23ff42fe4f8614cec0c365dfc69777a166b4`。AA 的 Intelligence Index、速度、价格、deprecated/迁移提示和 context 目录字段仍是第三方配置信息；官方价格单独记录为 mini `$0.75/$4.50`、nano `$0.20/$1.25` input/output 每百万 token。

OpenAI 官方页确认 snapshot 分别为 `gpt-5.4-mini-2026-03-17` 和 `gpt-5.4-nano-2026-03-17`，两者都是 text/image → text，400K context、272K maximum input、128K maximum output，支持 `none/low/medium/high/xhigh` effort。mini 页面列出 `tool_search` 与 `computer_use`；nano 当前页面未列出这两项，不能按家族名自动继承。两者均需按精确 model ID、snapshot、endpoint、tool catalog 和宿主权限做 capability probe。

完整证据、负面证据和面试主线见 [`gpt-5.4-mini-nano-source-notes.md`](gpt-5.4-mini-nano-source-notes.md)。不新增重复 Transformer 章节，技术映射进入既有 reasoning、router、Agent serving、工具协议和公平评测章节。

## 2026-09-15 Claude Fable 5 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Fable 5；Anthropic 官方资料用于核验模型身份、运行时协议和安全边界，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Fable 5 | Artificial Analysis：2026-06-09 的 `max + Opus 4.8 Fallback`；DataCurve DeepSWE v1.1：`xhigh` + `mini-swe-agent` | [模型页](https://platform.claude.com/docs/en/models/fable-5/overview)、[发布说明](https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5)、[重新部署公告](https://www.anthropic.com/news/redeploying-fable-5)、[System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card) | Active (legacy)；1M context、128K max output、adaptive always-on、默认 `high`、拒答/fallback、memory、程序化工具调用、compaction/context editing 和 task budgets 已核验 |

Artificial Analysis 详情页约为 49.70 Intelligence Index、60.6 output tokens/s、88.23s TTFT 和 `$8.75`/task；DeepSWE 页面记录 316/452、Pass@1 约 70% ±3%、Pass@4 约 88.5%、平均成本约 `$13.41`。这些分别绑定榜单配置、任务集、provider、harness、工具和 verifier，不能拼接成同一排名或裸模型分数。

当前将 Claude Fable 5 标为“资料级闭环”：官方模型页、发布/重新部署公告、API/Agent 文档、system card 入口和研究笔记均已具备；参数规模、内部架构、完整训练/后训练配方、可独立复现的技术报告和外部 benchmark 仍待核验，不新增 Fable 5 专属架构章节。详见 [`claude-fable-5-source-notes.md`](claude-fable-5-source-notes.md)。

## 2026-09-15 Claude Opus 5 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Opus 5；Anthropic 官方资料用于核验模型身份、adaptive thinking、长任务运行时和安全边界，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Opus 5 | Artificial Analysis：2026-07-24 的 `max`，并有 low/medium/high/xhigh 配置；DataCurve DeepSWE v1.1：`max` 及其他 effort + `mini-swe-agent` | [模型目录](https://platform.claude.com/docs/en/models/overview)、[专属模型页](https://platform.claude.com/docs/en/models/opus-5/overview)、[发布页](https://www.anthropic.com/research/claude-opus-5)、[System Card](https://www.anthropic.com/claude-opus-5-system-card)、[Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback) | 资料级闭环；1M context、128K/300K 输出、adaptive thinking、默认 `high`、五档 effort、512-token 缓存门槛、工具/effort 中途变更、fallback 和发布方评测边界已核验 |

Artificial Analysis 详情页约为 50.7002 Intelligence Index、50.07 output tokens/s、46.50s TTFT 和 `$5.8584`/task；DeepSWE max 记录 327/444、Pass@1 约 73.65% ±4%、Pass@4 约 88.50%、平均成本约 `$11.84`、约 99 steps。两者分别绑定第三方任务、provider、effort、工具、任务集、harness 和 verifier，不能拼接成裸模型分数。

Anthropic 公告称 Frontier-Bench、CursorBench、ARC-AGI 3、Zapier AutomationBench、OSWorld 2.0 及内部生命科学结果有显著提升；这些是发布方数据，Frontier-Bench 还绑定内部运行、`mini-SWE-agent`、GKE、每任务 5 次尝试和安全拒答时向 Opus 4.8 fallback。arXiv 精确标题检索只得到两篇把 Opus 5 当被测模型的文章，未找到 Opus 5 专属论文或完整技术报告。官方没有公开参数规模、架构、训练/后训练配方或独立 benchmark 复现，因此不新增 Opus 5 专属架构章节；面试主线映射到 Reasoning、长上下文/缓存、Agent 工具协议、fallback 和公平评测章节。详见 [`claude-opus-5-source-notes.md`](claude-opus-5-source-notes.md)。

## 2026-09-15 Claude Fable 5.1 网络复验

本节只更新已经出现在 Artificial Analysis 的 Claude Fable 5.1；DataCurve DeepSWE 页面本次仍没有该模型条目，因此不把 Fable 5 的 DeepSWE 结果迁移给 Fable 5.1。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Fable 5.1 | Artificial Analysis：`max + Default Fallback`，并有 `xhigh/high/medium/low + Default Fallback`；内嵌 `releaseDate` 为 2026-09-01 | [Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5-1)、[Anthropic 发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)、[模型页](https://platform.claude.com/docs/en/models/fable-5-1/overview)、[System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) | **内容专题闭环（AA 单榜）**；除 1M context、128K output、adaptive always-on、默认 high、Fable/Mythos safeguards 分层、cache read `$0.25/M` 和发布方 benchmark/安全摘要外，已新增第二十册第 21 章 21.29，记录 Opus 5.5 → Fable 5.1 单向 thinking-block 兼容及 API/model/prefix-binding 边界；真实 API、参数/架构/训练和独立复现仍待核验 |

2026-09-15 Artificial Analysis 主配置字段为 Intelligence Index `53.3738`、输出速度约 `65.98 tokens/s`、TTFT 约 `212.01s`、约 `$7.63`/Intelligence Index task；所有数值均绑定 `max + fallback`、第三方任务、provider 和测量时间。DataCurve 快照仍为 113 tasks/91 repositories/5 languages/`mini-swe-agent`，只有 `claude-fable-5` 的各 effort 行。

Anthropic 发布页新增的面试线索包括：Fable 5.1 与 Mythos 5.1 共享 underlying model 但 safeguards 不同；Fable 5.1 在 Claude Code 默认 High、Claude Cowork/Claude.ai 默认 Medium；cache reads 为 `$0.25/M`；EFS 以客户云基础设施支持零数据保留语义；anti-distillation 通过限制新账户编辑历史同时保留 prior thinking transcript；以及模型在科学工作流中的自定义 GPU kernel 与中间结果缓存案例。上述能力和数字均标明为发布方自述，不升级为内部架构或训练配方。

arXiv 标题精确检索返回 0 篇，全文检索返回 4 篇外部使用/评测论文，详见 [`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。当前没有 Fable 5.1 专属参数/架构披露、完整训练报告、可独立复现技术报告或 DataCurve 结果；不新增独立架构章节。该段记录当时的后续顺序；当前活动锚点和下一步以文件最新补充及 `plan.md`/`progress_v3.md` 顶部为准。

## 2026-09-21 Claude Fable 5.1 runtime recheck

本节只更新已经由 Artificial Analysis 发现的 Claude Fable 5.1；官方模型页、发布页和 System Card 用于核验与扩展运行时技术，不作为新的模型发现入口。DataCurve 当前仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，因此不迁移 Fable 5 或其他 Claude 版本的 Agent 结果。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Fable 5.1 | Artificial Analysis：`max + Default Fallback`，并有 `xhigh/high/medium/low + Default Fallback`；`releaseDate` `2026-09-01`，`deprecated=false` | [Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5-1)、[Anthropic 模型页](https://platform.claude.com/docs/en/models/fable-5-1/overview)、[Anthropic 发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)、[System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) | **内容专题闭环（AA 单榜）**；运行时 breaking changes（forced tool、thinking block 兼容/历史编辑失效）、per-message effort、turn-scoped system、`display: "updates"`、content provenance 和第二十册第 21 章 21.29 已同步；参数、架构、完整训练 recipe、独立技术报告、真实 endpoint probe 和独立复现待核验 |

Artificial Analysis 当前详情快照为 `3,854,152` bytes、SHA-256 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`；主配置 Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。这些是第三方配置/provider 字段。官方模型页、发布页和 System Card 快照分别为 `14,914`/`449,779`/`16,397,488` bytes，哈希详见研究笔记。

## 2026-09-23 Claude Fable 5.1 System Card 正文解析

本节只补强已经由 Artificial Analysis 发现的 `claude-fable-5-1`，不从 System Card、Anthropic Research 或论文另发现模型。Artificial Analysis 最新详情为 `4,006,915` bytes / SHA-256 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`；Intelligence Index `53.3549259623252`、median output speed `65.4865856934115 tokens/s`、median TTFT `298.446428812s`、1M context、约 `$7.6297`/task。DataCurve 仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行。

System Card 正文已解析并确认：Fable/Mythos 共享相同模型权重，区别主要在 safeguards 与访问计划；训练数据边界包括互联网公开信息、公共/私有数据和合成数据，使用去重/分类/ClaudeBot，knowledge cutoff 为 2026-06；RSP 只支持 Mythos 在 CB-1、未达 CB-2，autonomy threat model 1 为 low、threat model 2 未达。代表性能力结果和 Gray Swan/Shade/browser 安全结果均绑定具体 snapshot、effort、tools、fallback、safeguard state 与 verifier，不能迁移给 Fable 裸模型或 DataCurve。

当前状态升级为 **AA + System Card 正文证据**；仍不支持参数量、层数、dense/MoE、完整训练/后训练 recipe、Fable 5.1 独立技术报告、独立 benchmark 复现、目标硬件 profile 或生产 SLO 结论，因此不新增独立 Transformer 架构章节。研究笔记保留原始 PDF 哈希、正文抽取哈希和解析器入口。

Fable 5.1 与 Mythos 5.1 共享 underlying model、规格和价格，但 safeguards 与访问计划不同；Claude Code 默认 High、Claude.ai/Cowork 默认 Medium。上述是产品/治理和运行时证据，不能反推 Fable 5.1 内部架构或把 Mythos 当作独立公开模型。当前不新增 Fable 5.1 专属 Transformer 章节，复用 Agent、reasoning、tool protocol、serving 和安全章节。

## 2026-09-15 Claude Sonnet 5 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Sonnet 5；Anthropic 官方资料用于核验模型身份、API/Agent 运行时和发布方评测，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Sonnet 5 | Artificial Analysis：`max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；DataCurve DeepSWE v1.1：五档 effort + `mini-swe-agent` | [Artificial Analysis](https://artificialanalysis.ai/models/claude-sonnet-5)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Sonnet 5 发布公告](https://www.anthropic.com/news/claude-sonnet-5)、[System Card](https://www.anthropic.com/claude-sonnet-5-system-card)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) | 资料级闭环；`claude-sonnet-5`、2026-06-30、1M context、128K/300K 输出、Adaptive、默认 high、配置级榜单数据、thinking/tool block 协议、tokenizer、context awareness、compaction 和 programmatic tool calling 已核验；参数、架构、完整训练/后训练 recipe 和独立复现待核验 |

Artificial Analysis 主配置快照为 Intelligence Index `38.3576962882576`、约 80.0022 output tokens/s、约 202.6275s TTFT、输入/输出 `$2/$10/M`、约 `$5.0912`/task；DataCurve 五档 Pass@1 从 `max` 到 `low` 为 53.846%/49.667%/48.230%/39.778%/30.512%，平均成本为 `$26.40/$11.89/$7.43/$4.08/$2.19`。两榜单数字分别绑定第三方配置、provider、任务集、harness、工具、环境和 verifier，不能拼接为裸模型排名。

Anthropic 官方资料还确认 adaptive thinking 不接受手动 `thinking.type: "enabled"` + `budget_tokens`，effort 是行为信号而不是严格 token 预算，`max_tokens` 是共享硬上限；thinking blocks/signatures 需要按异构 `content` 回放；新 tokenizer 对相同文本约增加 30% token；Sonnet 5 支持 context awareness、两种不同 beta 的 server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling。System Card/发布方评测中的 SWE-bench Verified 85.2%、Terminal-Bench 2.1 80.4% 等数字单独保留为发布方证据。

2026-09-21 当前 AA 快照将主配置测量更新为 Intelligence Index `38.1638712882576`、`86.1790204452512 tokens/s` 和 `137.660255319s` median TTFC；9 月 15 日数值保留为历史测量，不推断模型 revision。System Card 深读显示训练数据仍只公开为专有混合数据，未给出完整训练 recipe；RSP、cyber 和 agentic safety 结果绑定 safeguards、任务和 harness。发布方还报告 Toolathlon Pass@1 `54.3%` 和 AA-Briefcase Elo `1393`，不能与 AA/DataCurve 拼成统一排名。

arXiv 精确标题检索截至 2026-09-21 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告。当前不支持参数规模、稠密/MoE 架构、完整训练/后训练配方、内部 adaptive-thinking 机制或独立 benchmark 复现结论；因此不新增 Sonnet 5 专属架构章节。具体快照哈希、完整配置表和面试主线见 [`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

2026-09-28 通过 `10.24.27.134:7890` 复取 What's New、Migration、Prompting、Models API/overview 与 compaction/preserved-thinking 官方 Markdown。重要补充：on-demand `compact-2026-09-04` 返回含 summary/signature 的单个 block，后续请求需将唯一最新 block 原样置于 history 开头并带回 beta header；threshold `compact-2026-01-12` 由 `context_management.edits` 触发并由服务端裁剪旧前缀。不能把这两个 beta 合成一个流程，且不支持在同一请求组合。保留 thinking 的通用回放条件不应误写成 Sonnet 5 的 prefix-binding enforcement：官方 Preserved thinking 文档称该检查从 Fable 5.1 开始，Sonnet 5 不运行该检查。详细快照、错误码、平台覆盖与证据边界见 Sonnet 5 研究笔记；真实 Messages API 未 probe。

## 2026-09-16 Claude Sonnet 4.6 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Sonnet 4.6；Anthropic 官方资料用于核验模型身份和扩展运行时技术，不是新的模型发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Sonnet 4.6 | Artificial Analysis：`claude-sonnet-4-6-adaptive` 等配置；DataCurve DeepSWE v1.1：`mini_swe_agent_claude_sonnet_4_6_high` | [Artificial Analysis](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive)、[DataCurve DeepSWE](https://deepswe.datacurve.ai/)、[Sonnet 4.6 发布公告](https://www.anthropic.com/news/claude-sonnet-4-6)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[System Card](https://www.anthropic.com/claude-sonnet-4-6-system-card) | 资料级闭环；`claude-sonnet-4-6`、2026-02-17、1M context（发布时 beta）、adaptive/extended thinking、compaction、tool search、computer use 和 Agent 工具链已核验；参数、架构、完整训练/后训练 recipe 和独立复现待核验 |

Artificial Analysis adaptive 快照为 `3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`；DataCurve 结果为 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、平均成本约 `$5.5224`、平均输出 `76,160.31` token、平均 Agent steps `133.66`。这些是第三方配置与 `mini-swe-agent` 系统结果，不能拼成裸模型分数。

Anthropic 发布页确认 coding、computer use、long-context reasoning、agent planning、knowledge work 和 design 定位，以及 1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory 和 programmatic tool calling。Computer use 的网页 prompt injection 风险和官方“较 Sonnet 4.5 改善”的表述按发布方证据记录，不写成风险已解决。

2026-09-16 三代理复验中，Artificial Analysis 首页快照均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`；DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。 `8098/1234` 的模型目录请求进入区域不可用页，`7890` 取得真实目录并确认 Sonnet 4.6 条目；这属于代理线路差异，不是模型不存在。

完整来源、面试主线、负面论文检索和待核验项见 [`claude-sonnet-4.6-source-notes.md`](claude-sonnet-4.6-source-notes.md)。暂无 Sonnet 4.6 独立正式架构章节。

## 2026-09-15 / 2026-09-28 Claude Opus 4.8 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Opus 4.8；Anthropic 官方资料用于核验模型身份和扩展动态工作流、effort 与长任务可靠性，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Opus 4.8 | Artificial Analysis：2026-05-28 的 `max`；DataCurve DeepSWE v1.1：`xhigh` 与 `max` + `mini-swe-agent` | [发布公告](https://www.anthropic.com/news/claude-opus-4-8)、[System Card](https://www.anthropic.com/claude-opus-4-8-system-card)、[Dynamic Workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) | 2026-05-28 发布；默认 `high`、支持更高 effort、Messages API system entries、动态编排/复核/反驳/checkpoint-resume 与使用门禁已核验；System Card 多 Agent/agentic-honesty 评测已抽取；参数、架构、完整训练配方和独立技术报告待核验 |

2026-09-28 经 7890 复验，Artificial Analysis 详情仍标记 deprecated 并指向 `claude-opus-5`；DataCurve `xhigh/max` 行的任务数、Pass@1/4、成本和 Agent steps 与研究笔记所载一致。System Card 报告的 multi-agent derived latency 与 Dynamic Workflows 产品不可混用；所有分数、速度、成本和 Agent steps 继续作为带配置条件的历史评测记录。完整来源、快照哈希、勘误和证据边界见 [`claude-opus-4.8-source-notes.md`](claude-opus-4.8-source-notes.md)。

## 2026-09-15 Kimi K2.7 Code 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Kimi K2.7 Code；Moonshot/Kimi 官方资料用于核验模型身份、架构摘要、推理协议和部署边界，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Kimi K2.7 Code | Artificial Analysis：2026-06-12 的 `kimi-k2-7-code`；DataCurve DeepSWE v1.1：`mini-swe-agent`、`reasoning_effort:null` | [Kimi 资源页](https://www.kimi.ai/resources/kimi-k2-7-code)、[API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)、[固定 revision 模型卡](https://huggingface.co/moonshotai/Kimi-K2.7-Code/tree/74797c9c62378b951a1f6fcf5c4631024e9b8bef)、[配置](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json) | 双榜内容专题闭环，正式章节为第二十一册第 94 章；1T/32B active MoE、MLA、256K/262,144 context、MoonViT 400M、native INT4、thinking/tool 协议及 2026-09-24 的 32K 默认输出预算、视觉 token estimate 和 Highspeed 服务变体边界已核验；完整训练/后训练配方、目标硬件、线上接受率、独立 benchmark 待核验 |

具体来源、快照哈希、benchmark 及负面论文检索证据见 [`kimi-k2.7-code-source-notes.md`](kimi-k2.7-code-source-notes.md)。

## 2026-09-15 Grok 4.6 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Grok 4.6；xAI 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Grok 4.6 | Artificial Analysis：2026-08-12 的 `high/medium/low/xhigh`；DataCurve DeepSWE v1.1：四档 `mini_swe_agent_grok_4_6_*` | [xAI 发布公告](https://x.ai/news/grok-4-6)、[官方模型页](https://docs.x.ai/developers/grok-4-6)、[Markdown 版本](https://docs.x.ai/developers/grok-4-6.md)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)、[工具文档](https://docs.x.ai/developers/tools/overview) | 资料级闭环；2026-08-12 官方发布日期、500K context、reasoning/opaque state、compaction、function calling、Web/X Search、Code Execution、Remote MCP 和发布方训练/评测描述已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验 |

Artificial Analysis 详情页约为 44.405 Intelligence Index、58.50 tokens/s 和 40.86s TTFT；DataCurve 四档 Pass@1 为 low 41.648%、medium 67.478%、high 65.188%、xhigh 66.741%，平均成本约 `$1.0424/$3.4490/$4.3849/$5.4977`。这些数字分别绑定榜单配置、任务集、provider、harness、工具和 verifier，不能拼接为裸模型分数。

xAI 发布页公开了比 Grok 4.5 更长的 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks，以及面向知识工作、编码和领域环境的 agentic RL；长任务中还强调 self-testing/verification。这些是发布方披露，不等于参数、网络结构、具体 optimizer、RL objective 或完整训练 recipe。

具体来源、快照哈希、四档 DataCurve 原始配置、面试主线、论文检索负面证据和未公开内容见 [`grok-4.6-source-notes.md`](grok-4.6-source-notes.md)。

## 2026-09-21 Grok 4.6 当前时点排行榜复验

本轮只复验已经存在于两个排行榜的 Grok 4.6，没有从 xAI 官方目录、论文或其他网站新增模型。Artificial Analysis 详情页当前快照为 `3,859,075` bytes、SHA-256 `a23b19aceae3fee2b1a92421d21358eb5739043c86b6d24a81348d849f887e94`；`Grok 4.6 (high)` 的第三方字段为 Intelligence Index `44.3113073012592`、median output speed `66.6843264403358 tokens/s`、TTFT `46.00s`、500K context、`$2/$6`，`releaseDate` 仍为 `2026-08-12`。9 月 15 日旧详情页的 `44.4050073012592`、`58.5035284934629 tokens/s` 和 `40.8595908855s` 保留为历史测量，不能解释成模型或训练版本变化。

DataCurve 当前快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，仍为 2026-09-03 更新的 113-task/91-repository/5-language `mini-swe-agent` 页面。四档 `mini_swe_agent_grok_4_6_*` 的 Pass@1 仍为 low `41.648%`、medium `67.478%`、high `65.188%`、xhigh `66.741%`，平均成本仍约为 `$1.0424/$3.4490/$4.3849/$5.4977`；结果绑定 effort、4 runs、工具、任务、环境和 verifier。

本轮 xAI 官方页面三条代理均未取得新响应：`1234` 为 EOF/超时，`7890` 与 `8098` 连接超时。已有官方发布页、模型页和 API 文档仍有效，但本轮不新增官方技术结论；Grok 4.6 状态继续为资料级闭环，不新增专属正式章节。

## 2026-09-15 Grok 4.5 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Grok 4.5；xAI 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Grok 4.5 | Artificial Analysis：2026-07-08 的 `high`；DataCurve DeepSWE v1.1：`mini_swe_agent_grok_4_5_high`、`reasoning_effort: high` | [xAI 发布公告](https://x.ai/news/grok-4-5)、[官方模型页](https://docs.x.ai/developers/models/grok-4.5)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)、[工具文档](https://docs.x.ai/developers/tools/overview) | 资料级闭环；500K context、reasoning/opaque state、compaction、function calling、Web/X Search、Remote MCP 和发布方训练/评测描述已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验 |

Artificial Analysis 约为 39.08 Intelligence Index、约 60.98 tokens/s；DataCurve 为 243/452、Pass@1 约 53.761%、Pass@4 约 77.876%、平均成本约 `$2.4157`。这些数字分别绑定榜单配置、任务集、provider、harness、工具和 verifier，不能拼接为裸模型分数。专属模型页列出 `xhigh`，通用 Reasoning 文档却称 Grok 4.5 的 `xhigh` 按 `high` 处理，该官方文档差异保持为待复验项。

具体来源、快照哈希、面试主线、论文检索负面证据和未公开内容见 [`grok-4.5-source-notes.md`](grok-4.5-source-notes.md)。

| 页面名称 | 榜单记录日期 | 条目 |
|---|---|---|
| DeepSeek V4.1 Flash (Reasoning, Max Effort) | 2026-09-10 | [deepseek-v4-1-flash](https://artificialanalysis.ai/models/deepseek-v4-1-flash) |
| Agnes 3.0 Flash | 2026-09-11 | [agnes-3-0-flash](https://artificialanalysis.ai/models/agnes-3-0-flash) |
| Ling-3.0-flash-VL | 2026-09-10 | [ling-3-0-flash-vl](https://artificialanalysis.ai/models/ling-3-0-flash-vl) |
| K2 Horizon MoVA 36B A4B | 2026-09-03 | [k2-mova-36b-mid5](https://artificialanalysis.ai/models/k2-mova-36b-mid5)、[k2-horizon-mova-36b-a4b](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b) |
| K2 Horizon 7B | 2026-09-03 | [k2-7b-ph2](https://artificialanalysis.ai/models/k2-7b-ph2)、[k2-horizon-7b](https://artificialanalysis.ai/models/k2-horizon-7b) |
| K2 Horizon 3.7B | 2026-09-03 | [k2-4b-ph1](https://artificialanalysis.ai/models/k2-4b-ph1)、[k2-horizon-3-7b](https://artificialanalysis.ai/models/k2-horizon-3-7b) |
| K2 Horizon 0.9B | 2026-09-03 | [k2-1b-final](https://artificialanalysis.ai/models/k2-1b-final)、[k2-horizon-0-9b](https://artificialanalysis.ai/models/k2-horizon-0-9b) |
| DeepSeek V4 Flash (Non-reasoning) | 2026-04-24 | [deepseek-v4-flash-0420-non-reasoning](https://artificialanalysis.ai/models/deepseek-v4-flash-0420-non-reasoning) |

| 页面名称 | 榜单记录日期 | 条目 |
|---|---|---|
| MiniCPM5-2B | 2026-09-07 | [minicpm5-2b](https://artificialanalysis.ai/models/minicpm5-2b) |
| MiniMax | 2026-09-03 | [minimax](https://artificialanalysis.ai/models/minimax) |
| GPT-6 Astra (max) | 2026-09-03 | [gpt-6-astra](https://artificialanalysis.ai/models/gpt-6-astra) |
| GPT-6 Astra (xhigh) | 2026-09-03 | [gpt-6-astra-xhigh](https://artificialanalysis.ai/models/gpt-6-astra-xhigh) |
| GPT-6 Astra (high) | 2026-09-03 | [gpt-6-astra-high](https://artificialanalysis.ai/models/gpt-6-astra-high) |
| GPT-6 Astra (medium) | 2026-09-03 | [gpt-6-astra-medium](https://artificialanalysis.ai/models/gpt-6-astra-medium) |
| GPT-6 Astra (low) | 2026-09-03 | [gpt-6-astra-low](https://artificialanalysis.ai/models/gpt-6-astra-low) |
| GPT-6 Astra (Non-reasoning) | 2026-09-03 | [gpt-6-astra-non-reasoning](https://artificialanalysis.ai/models/gpt-6-astra-non-reasoning) |
| K2 Horizon 375B A23B | 2026-09-03 | [k2-horizon-375b-a23b](https://artificialanalysis.ai/models/k2-horizon-375b-a23b) |
| DeepSeek | 2026-09-02 | [deepseek](https://artificialanalysis.ai/models/deepseek) |
| Muse Spark 1.3 (max) | 2026-09-02 | [muse-spark-1-3](https://artificialanalysis.ai/models/muse-spark-1-3) |
| Muse Spark 1.3 (xhigh) | 2026-09-02 | [muse-spark-1-3-xhigh](https://artificialanalysis.ai/models/muse-spark-1-3-xhigh) |
| Gemini 3.8 Flash (high) | 2026-09-02 | [gemini-3-8-flash](https://artificialanalysis.ai/models/gemini-3-8-flash) |
| Gemini 3.8 Flash (medium) | 2026-09-02 | [gemini-3-8-flash-medium](https://artificialanalysis.ai/models/gemini-3-8-flash-medium) |
| Gemini 3.8 Flash (low) | 2026-09-02 | [gemini-3-8-flash-low](https://artificialanalysis.ai/models/gemini-3-8-flash-low) |
| Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback) | 2026-09-01 | [claude-fable-5-1](https://artificialanalysis.ai/models/claude-fable-5-1) |
| Claude Fable 5.1 (Adaptive Reasoning, Xhigh Effort, Default Fallback) | 2026-09-01 | [claude-fable-5-1-xhigh](https://artificialanalysis.ai/models/claude-fable-5-1-xhigh) |
| Claude Fable 5.1 (Adaptive Reasoning, High Effort, Default Fallback) | 2026-09-01 | [claude-fable-5-1-high](https://artificialanalysis.ai/models/claude-fable-5-1-high) |
| Claude Fable 5.1 (Adaptive Reasoning, Medium Effort, Default Fallback) | 2026-09-01 | [claude-fable-5-1-medium](https://artificialanalysis.ai/models/claude-fable-5-1-medium) |
| Claude Fable 5.1 (Adaptive Reasoning, Low Effort, Default Fallback) | 2026-09-01 | [claude-fable-5-1-low](https://artificialanalysis.ai/models/claude-fable-5-1-low) |

## 2026-09-23 Claude Fable 5.1 状态协议复验

本节继续只处理两个排行榜已经发现的 `claude-fable-5-1`，不从 Anthropic 官方文档、System Card 或论文另发现 `Claude Mythos 5.1`。7890 当前 AA 首页、Fable 详情和 DataCurve 均返回 HTTP 200；当前详情为 `4,008,017` bytes / SHA-256 `4ea24782d05bcaae7d31f3cf348e5a573851678998706a9df82fa34679292f96`，`max with fallback` 的 Intelligence Index 为 `53.3549259623252`，median output speed 为 `64.7060238772019 tokens/s`，cost per Intelligence Index task 为 `7.629706364004841`。DataCurve 仍没有精确 `mini_swe_agent_claude_fable_5_1_*` 行。

Anthropic overview/What's new/migration Markdown 复验哈希分别为 `13e8aeb6bbd207311ac032916f2ab37e7a454c9d752c027cf76967a5c958a076`、`59d2a26f6e123d009a9e86aac9283705f7bfd15549b185dc07aa1858241baf19` 和 `c1d7bd16475ce9ad03ad556cc363635d93e695ecafc293e6acb85023e9f869b6`。本轮把 prefill 400、30-day retention/ZDR、Priority Tier、prefix mismatch 的 error/drop 和 `input_transformations` 记入 Fable 运行时证据。

新增 [`claude_fable51_state_protocol_audit.py`](code/claude_fable51_state_protocol_audit.py)，`py_compile`/主流程均通过，输出 `local_protocol_toy`，覆盖 adaptive/forced-tool/prefill、单向 thinking block、前缀编辑、进度与 tool result、provenance verifier、fallback 和幂等回放。当前状态为 **AA + System Card/API contract + local protocol toy**；参数、架构、完整 recipe、独立技术报告、精确 Agent 结果、完整权重、目标硬件和生产 SLO 仍待核验。
| Apodex 1.1 | 2026-08-30 | [apodex-1-1](https://artificialanalysis.ai/models/apodex-1-1) |
| Thinking Machines | 2026-08-26 | [thinking-machines](https://artificialanalysis.ai/models/thinking-machines) |
| GLM-5.3-Flash | 2026-08-26 | [glm-5-3-flash](https://artificialanalysis.ai/models/glm-5-3-flash) |
| Qwen3.8-Flash-Next | 2026-08-26 | [qwen3-8-flash-next](https://artificialanalysis.ai/models/qwen3-8-flash-next) |
| Agnes 2.5 Pro Beta | 2026-08-26 | [agnes-2-5-pro-beta](https://artificialanalysis.ai/models/agnes-2-5-pro-beta) |
| Granite 4.2 30B | 2026-08-25 | [granite-4-2-30b](https://artificialanalysis.ai/models/granite-4-2-30b) |
| Granite 4.2 8B | 2026-08-25 | [granite-4-2-8b](https://artificialanalysis.ai/models/granite-4-2-8b) |
| Granite 4.2 3B | 2026-08-25 | [granite-4-2-3b](https://artificialanalysis.ai/models/granite-4-2-3b) |
| DeepSeek V4 Flash Vision (Reasoning, Max Effort) | 2026-08-21 | [deepseek-v4-flash-vision](https://artificialanalysis.ai/models/deepseek-v4-flash-vision) |
| G9v3-39A5B | 2026-08-20 | [g9v3-39a5b](https://artificialanalysis.ai/models/g9v3-39a5b) |
| Anthropic | 2026-08-18 | [anthropic](https://artificialanalysis.ai/models/anthropic) |
| GLM-5.3 (max) | 2026-08-18 | [glm-5-3](https://artificialanalysis.ai/models/glm-5-3) |
| Meta | 2026-08-14 | [meta](https://artificialanalysis.ai/models/meta) |
| Qwen3.8 27B (xhigh) | 2026-08-14 | [qwen3-8-27b](https://artificialanalysis.ai/models/qwen3-8-27b) |
| Qwen3.8 27B (medium) | 2026-08-14 | [qwen3-8-27b-medium](https://artificialanalysis.ai/models/qwen3-8-27b-medium) |
| Qwen3.8 27B (low) | 2026-08-14 | [qwen3-8-27b-low](https://artificialanalysis.ai/models/qwen3-8-27b-low) |
| Qwen3.8 27B (Non-reasoning) | 2026-08-14 | [qwen3-8-27b-non-reasoning](https://artificialanalysis.ai/models/qwen3-8-27b-non-reasoning) |
| NVIDIA | 2026-08-13 | [nvidia](https://artificialanalysis.ai/models/nvidia) |
| Gemini 3.7 Flash (medium) | 2026-08-13 | [gemini-3-7-flash-medium](https://artificialanalysis.ai/models/gemini-3-7-flash-medium) |
| Gemini 3.7 Flash (high) | 2026-08-13 | [gemini-3-7-flash](https://artificialanalysis.ai/models/gemini-3-7-flash) |
| Gemini 3.7 Flash (low) | 2026-08-13 | [gemini-3-7-flash-low](https://artificialanalysis.ai/models/gemini-3-7-flash-low) |
| DeepSeek V4 Pro 0813 (Reasoning, Max Effort) | 2026-08-13 | [deepseek-v4-pro](https://artificialanalysis.ai/models/deepseek-v4-pro) |
| Google | 2026-08-12 | [google](https://artificialanalysis.ai/models/google) |
| Grok 4.6 (high) | 2026-08-12 | [grok-4-6](https://artificialanalysis.ai/models/grok-4-6) |
| Grok 4.6 (xhigh) | 2026-08-12 | [grok-4-6-xhigh](https://artificialanalysis.ai/models/grok-4-6-xhigh) |
| Grok 4.6 (medium) | 2026-08-12 | [grok-4-6-medium](https://artificialanalysis.ai/models/grok-4-6-medium) |
| Grok 4.6 (low) | 2026-08-12 | [grok-4-6-low](https://artificialanalysis.ai/models/grok-4-6-low) |
| Qwen3.8 2.4T A95B | 2026-08-12 | [qwen3-8-2-4t-a95b](https://artificialanalysis.ai/models/qwen3-8-2-4t-a95b) |
| Motif 3 | 2026-08-12 | [motif-3](https://artificialanalysis.ai/models/motif-3) |
| Solar Open2 250B | 2026-08-12 | [solar-open2-250b](https://artificialanalysis.ai/models/solar-open2-250b) |
| A.X-K2 | 2026-08-12 | [a-x-k2](https://artificialanalysis.ai/models/a-x-k2) |
| K-EXAONE 2.0 | 2026-08-12 | [k-exaone-2-0-0803](https://artificialanalysis.ai/models/k-exaone-2-0-0803) |
| Nemotron 3.5 Lightning | 2026-08-11 | [nemotron-3-5-lightning](https://artificialanalysis.ai/models/nemotron-3-5-lightning) |
| Muse Glimmer (high) | 2026-08-10 | [muse-glimmer](https://artificialanalysis.ai/models/muse-glimmer) |
| Quasar 438B (max, based on GLM-5.2) | 2026-08-10 | [quasar-438b](https://artificialanalysis.ai/models/quasar-438b) |
| Solar Pro 4 | 2026-08-06 | [solar-pro4](https://artificialanalysis.ai/models/solar-pro4) |
| Ling 3.0 Tiny | 2026-08-06 | [ling-3-0-tiny](https://artificialanalysis.ai/models/ling-3-0-tiny) |
| Muse Spark 1.2 (xhigh) | 2026-08-05 | [muse-spark-1-2](https://artificialanalysis.ai/models/muse-spark-1-2) |
| Ling 3.0 Flash | 2026-08-04 | [ling-3-0-flash](https://artificialanalysis.ai/models/ling-3-0-flash) |
| LFM2.5-2.6B | 2026-08-04 | [lfm2-5-2-6b](https://artificialanalysis.ai/models/lfm2-5-2-6b) |
| Qwen3.8 Max | 2026-08-03 | [qwen3-8-max](https://artificialanalysis.ai/models/qwen3-8-max) |
| DeepSeek V4 Flash 0731 (Reasoning, Max Effort) | 2026-07-31 | [deepseek-v4-flash](https://artificialanalysis.ai/models/deepseek-v4-flash) |
| Inkling Small | 2026-07-30 | [inkling-small](https://artificialanalysis.ai/models/inkling-small) |
| Alibaba | 2026-07-24 | [alibaba](https://artificialanalysis.ai/models/alibaba) |
| Claude Opus 5 (Adaptive Reasoning, Max Effort) | 2026-07-24 | [claude-opus-5](https://artificialanalysis.ai/models/claude-opus-5) |
| Claude Opus 5 (Adaptive Reasoning, Xhigh Effort) | 2026-07-24 | [claude-opus-5-xhigh](https://artificialanalysis.ai/models/claude-opus-5-xhigh) |
| Claude Opus 5 (Adaptive Reasoning, High Effort) | 2026-07-24 | [claude-opus-5-high](https://artificialanalysis.ai/models/claude-opus-5-high) |
| Claude Opus 5 (Adaptive Reasoning, Medium Effort) | 2026-07-24 | [claude-opus-5-medium](https://artificialanalysis.ai/models/claude-opus-5-medium) |
| Claude Opus 5 (Adaptive Reasoning, Low Effort) | 2026-07-24 | [claude-opus-5-low](https://artificialanalysis.ai/models/claude-opus-5-low) |
| Agnes 2.5 Pro Alpha | 2026-07-24 | [agnes-2-5-pro-alpha](https://artificialanalysis.ai/models/agnes-2-5-pro-alpha) |
| Celeris-1 | 2026-07-24 | [celeris-1](https://artificialanalysis.ai/models/celeris-1) |
| G9v3-3B | 2026-07-23 | [g9v3-3b](https://artificialanalysis.ai/models/g9v3-3b) |
| Gemini 3.5 Flash-Lite | 2026-07-21 | [gemini-3-5-flash-lite](https://artificialanalysis.ai/models/gemini-3-5-flash-lite) |
| Gemini 3.6 Flash (high) | 2026-07-21 | [gemini-3-6-flash](https://artificialanalysis.ai/models/gemini-3-6-flash) |
| OpenAI | 2026-07-16 | [openai](https://artificialanalysis.ai/models/openai) |
| Kimi K3 (max) | 2026-07-16 | [kimi-k3](https://artificialanalysis.ai/models/kimi-k3) |
| Kimi K3 (low) | 2026-07-16 | [kimi-k3-low](https://artificialanalysis.ai/models/kimi-k3-low) |
| Inkling (xhigh) | 2026-07-15 | [inkling](https://artificialanalysis.ai/models/inkling) |
| Motif 3 (Beta) | 2026-07-14 | [motif-0714](https://artificialanalysis.ai/models/motif-0714) |
| Z AI | 2026-07-09 | [zai](https://artificialanalysis.ai/models/zai) |
| GPT-5.6 Sol (max) | 2026-07-09 | [gpt-5-6-sol](https://artificialanalysis.ai/models/gpt-5-6-sol) |
| GPT-5.6 Sol (xhigh) | 2026-07-09 | [gpt-5-6-sol-xhigh](https://artificialanalysis.ai/models/gpt-5-6-sol-xhigh) |
| GPT-5.6 Sol (high) | 2026-07-09 | [gpt-5-6-sol-high](https://artificialanalysis.ai/models/gpt-5-6-sol-high) |
| GPT-5.6 Terra (max) | 2026-07-09 | [gpt-5-6-terra](https://artificialanalysis.ai/models/gpt-5-6-terra) |
| GPT-5.6 Sol (medium) | 2026-07-09 | [gpt-5-6-sol-medium](https://artificialanalysis.ai/models/gpt-5-6-sol-medium) |
| GPT-5.6 Terra (xhigh) | 2026-07-09 | [gpt-5-6-terra-xhigh](https://artificialanalysis.ai/models/gpt-5-6-terra-xhigh) |
| GPT-5.6 Luna (max) | 2026-07-09 | [gpt-5-6-luna](https://artificialanalysis.ai/models/gpt-5-6-luna) |
| GPT-5.6 Luna (xhigh) | 2026-07-09 | [gpt-5-6-luna-xhigh](https://artificialanalysis.ai/models/gpt-5-6-luna-xhigh) |
| GPT-5.6 Terra (high) | 2026-07-09 | [gpt-5-6-terra-high](https://artificialanalysis.ai/models/gpt-5-6-terra-high) |
| GPT-5.6 Sol (low) | 2026-07-09 | [gpt-5-6-sol-low](https://artificialanalysis.ai/models/gpt-5-6-sol-low) |
| GPT-5.6 Luna (high) | 2026-07-09 | [gpt-5-6-luna-high](https://artificialanalysis.ai/models/gpt-5-6-luna-high) |
| GPT-5.6 Terra (medium) | 2026-07-09 | [gpt-5-6-terra-medium](https://artificialanalysis.ai/models/gpt-5-6-terra-medium) |
| GPT-5.6 Sol (Non-reasoning) | 2026-07-09 | [gpt-5-6-sol-non-reasoning](https://artificialanalysis.ai/models/gpt-5-6-sol-non-reasoning) |
| GPT-5.6 Terra (low) | 2026-07-09 | [gpt-5-6-terra-low](https://artificialanalysis.ai/models/gpt-5-6-terra-low) |
| GPT-5.6 Luna (medium) | 2026-07-09 | [gpt-5-6-luna-medium](https://artificialanalysis.ai/models/gpt-5-6-luna-medium) |
| GPT-5.6 Terra (Non-reasoning) | 2026-07-09 | [gpt-5-6-terra-non-reasoning](https://artificialanalysis.ai/models/gpt-5-6-terra-non-reasoning) |
| GPT-5.6 Luna (low) | 2026-07-09 | [gpt-5-6-luna-low](https://artificialanalysis.ai/models/gpt-5-6-luna-low) |
| GPT-5.6 Luna (Non-reasoning) | 2026-07-09 | [gpt-5-6-luna-non-reasoning](https://artificialanalysis.ai/models/gpt-5-6-luna-non-reasoning) |
| Muse Spark 1.1 (xhigh) | 2026-07-09 | [muse-spark-1-1](https://artificialanalysis.ai/models/muse-spark-1-1) |
| JT-4.1 Flash 236B A21B | 2026-07-09 | [jt-4-1-flash-236b-a21b](https://artificialanalysis.ai/models/jt-4-1-flash-236b-a21b) |
| Grok 4.5 (high) | 2026-07-08 | [grok-4-5](https://artificialanalysis.ai/models/grok-4-5) |
| Hy3 | 2026-07-06 | [hy3](https://artificialanalysis.ai/models/hy3) |
| Claude Sonnet 5 (Adaptive Reasoning, Max Effort) | 2026-06-30 | [claude-sonnet-5](https://artificialanalysis.ai/models/claude-sonnet-5) |
| Claude Sonnet 5 (Non-reasoning, High Effort) | 2026-06-30 | [claude-sonnet-5-non-reasoning](https://artificialanalysis.ai/models/claude-sonnet-5-non-reasoning) |
| Claude Sonnet 5 (Adaptive Reasoning, Medium Effort) | 2026-06-30 | [claude-sonnet-5-medium](https://artificialanalysis.ai/models/claude-sonnet-5-medium) |
| Claude Sonnet 5 (Adaptive Reasoning, Low Effort) | 2026-06-30 | [claude-sonnet-5-low](https://artificialanalysis.ai/models/claude-sonnet-5-low) |
| Claude Sonnet 5 (Adaptive Reasoning, High Effort) | 2026-06-30 | [claude-sonnet-5-high](https://artificialanalysis.ai/models/claude-sonnet-5-high) |
| Claude Sonnet 5 (Adaptive Reasoning, Xhigh Effort) | 2026-06-30 | [claude-sonnet-5-xhigh](https://artificialanalysis.ai/models/claude-sonnet-5-xhigh) |
| LongCat 2.0 | 2026-06-29 | [longcat-2-0](https://artificialanalysis.ai/models/longcat-2-0) |
| GPT-5.5 Instant (June 2026) | 2026-06-25 | [gpt-5-5-instant-06-26](https://artificialanalysis.ai/models/gpt-5-5-instant-06-26) |
| GLM-5.2 (max) | 2026-06-16 | [glm-5-2](https://artificialanalysis.ai/models/glm-5-2) |
| GLM-5.2 (Non-reasoning) | 2026-06-16 | [glm-5-2-non-reasoning](https://artificialanalysis.ai/models/glm-5-2-non-reasoning) |
| Grok Build 0.1 0616 | 2026-06-16 | [grok-build-0-1-06-16](https://artificialanalysis.ai/models/grok-build-0-1-06-16) |
| Kimi K2.7 Code | 2026-06-12 | [kimi-k2-7-code](https://artificialanalysis.ai/models/kimi-k2-7-code) |
| DiffusionGemma 26B A4B | 2026-06-10 | [diffusiongemma-26b-a4b](https://artificialanalysis.ai/models/diffusiongemma-26b-a4b) |
| SpaceXAI | 2026-06-09 | [xai](https://artificialanalysis.ai/models/xai) |
| Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback) | 2026-06-09 | [claude-fable-5](https://artificialanalysis.ai/models/claude-fable-5) |
| North Mini Code | 2026-06-09 | [north-mini-code](https://artificialanalysis.ai/models/north-mini-code) |
| Nemotron 3 Ultra 550B A55B (Reasoning) | 2026-06-04 | [nvidia-nemotron-3-ultra-550b-a55b](https://artificialanalysis.ai/models/nvidia-nemotron-3-ultra-550b-a55b) |
| Gemma 4 12B (Reasoning) | 2026-06-03 | [gemma-4-12b](https://artificialanalysis.ai/models/gemma-4-12b) |
| Gemma 4 12B (Non-reasoning) | 2026-06-03 | [gemma-4-12b-non-reasoning](https://artificialanalysis.ai/models/gemma-4-12b-non-reasoning) |
| Nex-N2-Pro | 2026-06-02 | [nex-n2-pro](https://artificialanalysis.ai/models/nex-n2-pro) |
| Qwen3.7 Plus | 2026-06-01 | [qwen3-7-plus](https://artificialanalysis.ai/models/qwen3-7-plus) |
| MiniMax-M3 | 2026-06-01 | [minimax-m3](https://artificialanalysis.ai/models/minimax-m3) |
| Step 3.7 Flash | 2026-05-29 | [step-3-7-flash](https://artificialanalysis.ai/models/step-3-7-flash) |
| Claude Opus 4.8 (Adaptive Reasoning, Max Effort) | 2026-05-28 | [claude-opus-4-8](https://artificialanalysis.ai/models/claude-opus-4-8) |
| LFM2.5-8B-A1B | 2026-05-28 | [lfm2-5-8b-a1b](https://artificialanalysis.ai/models/lfm2-5-8b-a1b) |
| HyperNova 60B 2605 (high, based on gpt-oss-120b) | 2026-05-26 | [hypernova-60b](https://artificialanalysis.ai/models/hypernova-60b) |
| MiniCPM5-1B (Reasoning) | 2026-05-25 | [minicpm5-1b](https://artificialanalysis.ai/models/minicpm5-1b) |
| MiniCPM5-1B (Non-reasoning) | 2026-05-25 | [minicpm5-1b-non-reasoning](https://artificialanalysis.ai/models/minicpm5-1b-non-reasoning) |
| Command A+ | 2026-05-20 | [command-a-plus](https://artificialanalysis.ai/models/command-a-plus) |
| Qwen3.7 Max | 2026-05-19 | [qwen3-7-max](https://artificialanalysis.ai/models/qwen3-7-max) |
| Gemini 3.5 Flash (medium) | 2026-05-19 | [gemini-3-5-flash-medium](https://artificialanalysis.ai/models/gemini-3-5-flash-medium) |
| Gemini 3.5 Flash (high) | 2026-05-19 | [gemini-3-5-flash](https://artificialanalysis.ai/models/gemini-3-5-flash) |
| Gemini 3.5 Flash (minimal) | 2026-05-19 | [gemini-3-5-flash-minimal](https://artificialanalysis.ai/models/gemini-3-5-flash-minimal) |
| JT-35B-Flash | 2026-05-14 | [jt-35b-flash](https://artificialanalysis.ai/models/jt-35b-flash) |
| MiniCPM-V 4.6 1.3B | 2026-05-11 | [minicpm-v4-6-1-3b](https://artificialanalysis.ai/models/minicpm-v4-6-1-3b) |
| Ring-2.6-1T | 2026-05-08 | [ring-2-6-1t](https://artificialanalysis.ai/models/ring-2-6-1t) |
| GPT-5.5 Instant (May 2026) | 2026-05-05 | [gpt-5-5-instant-05-26](https://artificialanalysis.ai/models/gpt-5-5-instant-05-26) |
| Grok 4.3 (high) | 2026-04-30 | [grok-4-3](https://artificialanalysis.ai/models/grok-4-3) |
| Grok 4.3 (medium) | 2026-04-30 | [grok-4-3-medium](https://artificialanalysis.ai/models/grok-4-3-medium) |
| Grok 4.3 (low) | 2026-04-30 | [grok-4-3-low](https://artificialanalysis.ai/models/grok-4-3-low) |
| Grok 4.3 (Non-reasoning) | 2026-04-30 | [grok-4-3-non-reasoning](https://artificialanalysis.ai/models/grok-4-3-non-reasoning) |
| Nemotron 3 Nano Omni 30B A3B Reasoning | 2026-04-29 | [nemotron-3-nano-omni-30b-a3b](https://artificialanalysis.ai/models/nemotron-3-nano-omni-30b-a3b) |
| Mistral Medium 3.5 | 2026-04-29 | [mistral-medium-3-5](https://artificialanalysis.ai/models/mistral-medium-3-5) |
| Granite 4.1 30B | 2026-04-29 | [granite-4-1-30b](https://artificialanalysis.ai/models/granite-4-1-30b) |
| Granite 4.1 8B | 2026-04-29 | [granite-4-1-8b](https://artificialanalysis.ai/models/granite-4-1-8b) |
| Granite 4.1 3B | 2026-04-29 | [granite-4-1-3b](https://artificialanalysis.ai/models/granite-4-1-3b) |
| DeepSeek V4 Pro (Reasoning, Max Effort) | 2026-04-24 | [deepseek-v4-pro-0424](https://artificialanalysis.ai/models/deepseek-v4-pro-0424) |
| DeepSeek V4 Pro (Reasoning, High Effort) | 2026-04-24 | [deepseek-v4-pro-0424-high](https://artificialanalysis.ai/models/deepseek-v4-pro-0424-high) |
| DeepSeek V4 Flash (Reasoning, High Effort) | 2026-04-24 | [deepseek-v4-flash-0420-high](https://artificialanalysis.ai/models/deepseek-v4-flash-0420-high) |
| DeepSeek V4 Flash (Reasoning, Max Effort) | 2026-04-24 | [deepseek-v4-flash-0420](https://artificialanalysis.ai/models/deepseek-v4-flash-0420) |
| DeepSeek V4 Pro (Non-reasoning) | 2026-04-24 | [deepseek-v4-pro-0424-non-reasoning](https://artificialanalysis.ai/models/deepseek-v4-pro-0424-non-reasoning) |
| DeepSeek V4 Flash (Non-reasoning) | 2026-04-24 | [deepseek-v4-flash-non-reasoning](https://artificialanalysis.ai/models/deepseek-v4-flash-non-reasoning) |
| Mistral | 2026-04-23 | [mistral](https://artificialanalysis.ai/models/mistral) |
| GPT-5.5 (xhigh) | 2026-04-23 | [gpt-5-5](https://artificialanalysis.ai/models/gpt-5-5) |
| GPT-5.5 (high) | 2026-04-23 | [gpt-5-5-high](https://artificialanalysis.ai/models/gpt-5-5-high) |
| GPT-5.5 (medium) | 2026-04-23 | [gpt-5-5-medium](https://artificialanalysis.ai/models/gpt-5-5-medium) |
| GPT-5.5 (low) | 2026-04-23 | [gpt-5-5-low](https://artificialanalysis.ai/models/gpt-5-5-low) |
| GPT-5.5 (Non-reasoning) | 2026-04-23 | [gpt-5-5-non-reasoning](https://artificialanalysis.ai/models/gpt-5-5-non-reasoning) |
| GPT-5.5 Pro (xhigh) | 2026-04-23 | [gpt-5-5-pro](https://artificialanalysis.ai/models/gpt-5-5-pro) |
| Hy3-preview (Reasoning) | 2026-04-23 | [hy3-preview](https://artificialanalysis.ai/models/hy3-preview) |
| Hy3-preview (Non-reasoning) | 2026-04-23 | [hy3-non-reasoning](https://artificialanalysis.ai/models/hy3-non-reasoning) |
| Ling-2.6-1T | 2026-04-23 | [ling-2-6-1t](https://artificialanalysis.ai/models/ling-2-6-1t) |
| Qwen3.6 27B (Reasoning) | 2026-04-22 | [qwen3-6-27b](https://artificialanalysis.ai/models/qwen3-6-27b) |
| Qwen3.6 27B (Non-reasoning) | 2026-04-22 | [qwen3-6-27b-non-reasoning](https://artificialanalysis.ai/models/qwen3-6-27b-non-reasoning) |
| MiMo-V2.5-Pro | 2026-04-22 | [mimo-v2-5-pro](https://artificialanalysis.ai/models/mimo-v2-5-pro) |
| MiMo-V2.5 | 2026-04-22 | [mimo-v2-5-0424](https://artificialanalysis.ai/models/mimo-v2-5-0424) |
| MiMo-V2.5-Pro (Non-reasoning) | 2026-04-22 | [mimo-v2-5-pro-non-reasoning](https://artificialanalysis.ai/models/mimo-v2-5-pro-non-reasoning) |
| Ling 2.6 Flash | 2026-04-21 | [ling-2-6-flash](https://artificialanalysis.ai/models/ling-2-6-flash) |
| Kimi K2.6 | 2026-04-20 | [kimi-k2-6](https://artificialanalysis.ai/models/kimi-k2-6) |
| Kimi K2.6 (Non-reasoning) | 2026-04-20 | [kimi-k2-6-non-reasoning](https://artificialanalysis.ai/models/kimi-k2-6-non-reasoning) |
| Qwen3.6 Max Preview | 2026-04-20 | [qwen3-6-max](https://artificialanalysis.ai/models/qwen3-6-max) |
| Claude Opus 4.7 (Adaptive Reasoning, Max Effort) | 2026-04-16 | [claude-opus-4-7](https://artificialanalysis.ai/models/claude-opus-4-7) |
| Claude Opus 4.7 (Non-reasoning, High Effort) | 2026-04-16 | [claude-opus-4-7-non-reasoning](https://artificialanalysis.ai/models/claude-opus-4-7-non-reasoning) |
| Qwen3.6 35B A3B (Reasoning) | 2026-04-16 | [qwen3-6-35b-a3b](https://artificialanalysis.ai/models/qwen3-6-35b-a3b) |
| Qwen3.6 35B A3B (Non-reasoning) | 2026-04-16 | [qwen3-6-35b-a3b-non-reasoning](https://artificialanalysis.ai/models/qwen3-6-35b-a3b-non-reasoning) |
| JT-MINI | 2026-04-15 | [jt-mini](https://artificialanalysis.ai/models/jt-mini) |
| EXAONE 4.5 33B | 2026-04-09 | [exaone-4-5-33b](https://artificialanalysis.ai/models/exaone-4-5-33b) |
| EXAONE 4.5 33B (Non-reasoning) | 2026-04-09 | [exaone-4-5-33b-non-reasoning](https://artificialanalysis.ai/models/exaone-4-5-33b-non-reasoning) |
| Muse Spark | 2026-04-08 | [muse-spark](https://artificialanalysis.ai/models/muse-spark) |
| GLM-5.1 (Reasoning) | 2026-04-07 | [glm-5-1](https://artificialanalysis.ai/models/glm-5-1) |
| GLM-5.1 (Non-reasoning) | 2026-04-07 | [glm-5-1-non-reasoning](https://artificialanalysis.ai/models/glm-5-1-non-reasoning) |
| Grok 4.20 0309 v2 (Reasoning) | 2026-04-07 | [grok-4-20](https://artificialanalysis.ai/models/grok-4-20) |
| Grok 4.20 0309 v2 (Non-reasoning) | 2026-04-07 | [grok-4-20-non-reasoning](https://artificialanalysis.ai/models/grok-4-20-non-reasoning) |
| Solar Pro 3 | 2026-04-06 | [solar-pro-3](https://artificialanalysis.ai/models/solar-pro-3) |
| Gemma 4 E4B (Reasoning) | 2026-04-03 | [gemma-4-e4b](https://artificialanalysis.ai/models/gemma-4-e4b) |
| Gemma 4 E4B (Non-reasoning) | 2026-04-03 | [gemma-4-e4b-non-reasoning](https://artificialanalysis.ai/models/gemma-4-e4b-non-reasoning) |
| Qwen3.6 Plus | 2026-04-02 | [qwen3-6-plus](https://artificialanalysis.ai/models/qwen3-6-plus) |
| Gemma 4 26B A4B (Reasoning) | 2026-04-02 | [gemma-4-26b-a4b](https://artificialanalysis.ai/models/gemma-4-26b-a4b) |
| Gemma 4 31B (Reasoning) | 2026-04-02 | [gemma-4-31b](https://artificialanalysis.ai/models/gemma-4-31b) |
| Gemma 4 31B (Non-reasoning) | 2026-04-02 | [gemma-4-31b-non-reasoning](https://artificialanalysis.ai/models/gemma-4-31b-non-reasoning) |
| Gemma 4 26B A4B (Non-reasoning) | 2026-04-02 | [gemma-4-26b-a4b-non-reasoning](https://artificialanalysis.ai/models/gemma-4-26b-a4b-non-reasoning) |
| Gemma 4 E2B (Reasoning) | 2026-04-02 | [gemma-4-e2b](https://artificialanalysis.ai/models/gemma-4-e2b) |
| Gemma 4 E2B (Non-reasoning) | 2026-04-02 | [gemma-4-e2b-non-reasoning](https://artificialanalysis.ai/models/gemma-4-e2b-non-reasoning) |
| Step 3.5 Flash 2603 | 2026-04-02 | [step-3-5-flash](https://artificialanalysis.ai/models/step-3-5-flash) |
| GLM 5V Turbo (Reasoning) | 2026-04-01 | [glm-5v-turbo](https://artificialanalysis.ai/models/glm-5v-turbo) |
| Trinity Large Thinking | 2026-04-01 | [trinity-large-thinking](https://artificialanalysis.ai/models/trinity-large-thinking) |
| Qwen3.5 Omni Plus | 2026-03-30 | [qwen3-5-omni-plus](https://artificialanalysis.ai/models/qwen3-5-omni-plus) |
| Qwen3.5 Omni Flash | 2026-03-30 | [qwen3-5-omni-flash](https://artificialanalysis.ai/models/qwen3-5-omni-flash) |
| MiMo-V2-Omni-0327 | 2026-03-27 | [mimo-v2-omni-0327](https://artificialanalysis.ai/models/mimo-v2-omni-0327) |
| KAT Coder Pro V2 | 2026-03-27 | [kat-coder-pro-v2](https://artificialanalysis.ai/models/kat-coder-pro-v2) |
| MiMo-V2-Omni | 2026-03-19 | [mimo-v2-omni](https://artificialanalysis.ai/models/mimo-v2-omni) |
| Nemotron Cascade 2 30B A3B | 2026-03-19 | [nemotron-cascade-2-30b-a3b](https://artificialanalysis.ai/models/nemotron-cascade-2-30b-a3b) |
| MiniMax-M2.7 | 2026-03-18 | [minimax-m2-7](https://artificialanalysis.ai/models/minimax-m2-7) |
| MiMo-V2-Pro | 2026-03-18 | [mimo-v2-pro](https://artificialanalysis.ai/models/mimo-v2-pro) |
| GPT-5.4 mini (xhigh) | 2026-03-17 | [gpt-5-4-mini](https://artificialanalysis.ai/models/gpt-5-4-mini) |
| GPT-5.4 nano (xhigh) | 2026-03-17 | [gpt-5-4-nano](https://artificialanalysis.ai/models/gpt-5-4-nano) |
| GPT-5.4 nano (medium) | 2026-03-17 | [gpt-5-4-nano-medium](https://artificialanalysis.ai/models/gpt-5-4-nano-medium) |
| GPT-5.4 mini (medium) | 2026-03-17 | [gpt-5-4-mini-medium](https://artificialanalysis.ai/models/gpt-5-4-mini-medium) |
| GPT-5.4 nano (Non-Reasoning) | 2026-03-17 | [gpt-5-4-nano-non-reasoning](https://artificialanalysis.ai/models/gpt-5-4-nano-non-reasoning) |
| GPT-5.4 mini (Non-Reasoning) | 2026-03-17 | [gpt-5-4-mini-non-reasoning](https://artificialanalysis.ai/models/gpt-5-4-mini-non-reasoning) |
| NVIDIA Nemotron 3 Nano 4B | 2026-03-16 | [nvidia-nemotron-3-nano-4b](https://artificialanalysis.ai/models/nvidia-nemotron-3-nano-4b) |
| Mistral Small 4 (Reasoning) | 2026-03-16 | [mistral-small-4](https://artificialanalysis.ai/models/mistral-small-4) |
| Mistral Small 4 (Non-reasoning) | 2026-03-16 | [mistral-small-4-non-reasoning](https://artificialanalysis.ai/models/mistral-small-4-non-reasoning) |
| GLM-5-Turbo | 2026-03-15 | [glm-5-turbo](https://artificialanalysis.ai/models/glm-5-turbo) |
| Nemotron 3 Super 120B A12B (Reasoning) | 2026-03-11 | [nvidia-nemotron-3-super-120b-a12b](https://artificialanalysis.ai/models/nvidia-nemotron-3-super-120b-a12b) |
| Grok 4.20 0309 (Reasoning) | 2026-03-10 | [grok-4-20-0309](https://artificialanalysis.ai/models/grok-4-20-0309) |
| Grok 4.20 0309 (Non-reasoning) | 2026-03-10 | [grok-4-20-0309-non-reasoning](https://artificialanalysis.ai/models/grok-4-20-0309-non-reasoning) |
| Sarvam 105B (high) | 2026-03-06 | [sarvam-105b](https://artificialanalysis.ai/models/sarvam-105b) |
| Sarvam 30B (high) | 2026-03-06 | [sarvam-30b](https://artificialanalysis.ai/models/sarvam-30b) |
| GPT-5.4 (xhigh) | 2026-03-05 | [gpt-5-4](https://artificialanalysis.ai/models/gpt-5-4) |
| GPT-5.4 (low) | 2026-03-05 | [gpt-5-4-low](https://artificialanalysis.ai/models/gpt-5-4-low) |
| GPT-5.4 (Non-reasoning) | 2026-03-05 | [gpt-5-4-non-reasoning](https://artificialanalysis.ai/models/gpt-5-4-non-reasoning) |
| GPT-5.4 Pro (xhigh) | 2026-03-05 | [gpt-5-4-pro](https://artificialanalysis.ai/models/gpt-5-4-pro) |
| Gemini 3.1 Flash-Lite | 2026-03-03 | [gemini-3-1-flash-lite-preview](https://artificialanalysis.ai/models/gemini-3-1-flash-lite-preview) |
| Qwen3.5 9B (Reasoning) | 2026-03-02 | [qwen3-5-9b](https://artificialanalysis.ai/models/qwen3-5-9b) |
| Qwen3.5 9B (Non-reasoning) | 2026-03-02 | [qwen3-5-9b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-9b-non-reasoning) |
| Qwen3.5 4B (Reasoning) | 2026-03-02 | [qwen3-5-4b](https://artificialanalysis.ai/models/qwen3-5-4b) |
| Qwen3.5 4B (Non-reasoning) | 2026-03-02 | [qwen3-5-4b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-4b-non-reasoning) |
| Qwen3.5 2B (Reasoning) | 2026-03-02 | [qwen3-5-2b](https://artificialanalysis.ai/models/qwen3-5-2b) |
| Qwen3.5 2B (Non-reasoning) | 2026-03-02 | [qwen3-5-2b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-2b-non-reasoning) |
| Qwen3.5 0.8B (Reasoning) | 2026-03-02 | [qwen3-5-0-8b](https://artificialanalysis.ai/models/qwen3-5-0-8b) |
| Qwen3.5 0.8B (Non-reasoning) | 2026-03-02 | [qwen3-5-0-8b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-0-8b-non-reasoning) |
| LFM2 24B A2B | 2026-02-25 | [lfm2-24b-a2b](https://artificialanalysis.ai/models/lfm2-24b-a2b) |
| Qwen3.5 27B (Reasoning) | 2026-02-24 | [qwen3-5-27b](https://artificialanalysis.ai/models/qwen3-5-27b) |
| Qwen3.5 27B (Non-reasoning) | 2026-02-24 | [qwen3-5-27b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-27b-non-reasoning) |
| Qwen3.5 35B A3B (Reasoning) | 2026-02-24 | [qwen3-5-35b-a3b](https://artificialanalysis.ai/models/qwen3-5-35b-a3b) |
| Qwen3.5 122B A10B (Non-reasoning) | 2026-02-24 | [qwen3-5-122b-a10b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-122b-a10b-non-reasoning) |
| Qwen3.5 122B A10B (Reasoning) | 2026-02-24 | [qwen3-5-122b-a10b](https://artificialanalysis.ai/models/qwen3-5-122b-a10b) |
| Qwen3.5 35B A3B (Non-reasoning) | 2026-02-24 | [qwen3-5-35b-a3b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-35b-a3b-non-reasoning) |
| Mercury 2 | 2026-02-20 | [mercury-2](https://artificialanalysis.ai/models/mercury-2) |
| Gemini 3.1 Pro Preview | 2026-02-19 | [gemini-3-1-pro-preview](https://artificialanalysis.ai/models/gemini-3-1-pro-preview) |
| Claude Sonnet 4.6 (Adaptive Reasoning, Max Effort) | 2026-02-17 | [claude-sonnet-4-6-adaptive](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive) |
| Claude Sonnet 4.6 (Non-reasoning, High Effort) | 2026-02-17 | [claude-sonnet-4-6](https://artificialanalysis.ai/models/claude-sonnet-4-6) |
| Claude Sonnet 4.6 (Non-reasoning, Low Effort) | 2026-02-17 | [claude-sonnet-4-6-non-reasoning-low-effort](https://artificialanalysis.ai/models/claude-sonnet-4-6-non-reasoning-low-effort) |
| Tiny Aya Global | 2026-02-17 | [tiny-aya-global](https://artificialanalysis.ai/models/tiny-aya-global) |
| Qwen3.5 397B A17B (Non-reasoning) | 2026-02-16 | [qwen3-5-397b-a17b-non-reasoning](https://artificialanalysis.ai/models/qwen3-5-397b-a17b-non-reasoning) |
| Qwen3.5 397B A17B (Reasoning) | 2026-02-16 | [qwen3-5-397b-a17b](https://artificialanalysis.ai/models/qwen3-5-397b-a17b) |
| MiniMax-M2.5 | 2026-02-12 | [minimax-m2-5](https://artificialanalysis.ai/models/minimax-m2-5) |
| GLM-5 (Reasoning) | 2026-02-11 | [glm-5](https://artificialanalysis.ai/models/glm-5) |
| GLM-5 (Non-reasoning) | 2026-02-11 | [glm-5-non-reasoning](https://artificialanalysis.ai/models/glm-5-non-reasoning) |
| Nanbeige4.1-3B | 2026-02-11 | [nanbeige4-1-3b](https://artificialanalysis.ai/models/nanbeige4-1-3b) |
| Tri-21B-think Preview | 2026-02-10 | [tri-21b-think-preview](https://artificialanalysis.ai/models/tri-21b-think-preview) |
| Tri-21B-Think | 2026-02-10 | [tri-21b-think-v0-5](https://artificialanalysis.ai/models/tri-21b-think-v0-5) |
| Claude Opus 4.6 (Adaptive Reasoning, Max Effort) | 2026-02-05 | [claude-opus-4-6-adaptive](https://artificialanalysis.ai/models/claude-opus-4-6-adaptive) |
| Claude Opus 4.6 (Non-reasoning, High Effort) | 2026-02-05 | [claude-opus-4-6](https://artificialanalysis.ai/models/claude-opus-4-6) |
| GPT-5.3 Codex (xhigh) | 2026-02-05 | [gpt-5-3-codex](https://artificialanalysis.ai/models/gpt-5-3-codex) |
| Gemini 3 Deep Think | 2026-02-05 | [gemini-3-deep-think](https://artificialanalysis.ai/models/gemini-3-deep-think) |
| Qwen3 Coder Next | 2026-02-03 | [qwen3-coder-next](https://artificialanalysis.ai/models/qwen3-coder-next) |
| Step 3.5 Flash | 2026-02-02 | [step-3-5-flash-0202](https://artificialanalysis.ai/models/step-3-5-flash-0202) |
| LongCat Flash Lite | 2026-01-28 | [longcat-flash-lite](https://artificialanalysis.ai/models/longcat-flash-lite) |
| Kimi K2.5 (Reasoning) | 2026-01-27 | [kimi-k2-5](https://artificialanalysis.ai/models/kimi-k2-5) |
| Kimi K2.5 (Non-reasoning) | 2026-01-27 | [kimi-k2-5-non-reasoning](https://artificialanalysis.ai/models/kimi-k2-5-non-reasoning) |
| Qwen3 Max Thinking | 2026-01-26 | [qwen3-max-thinking](https://artificialanalysis.ai/models/qwen3-max-thinking) |
| Step3 VL 10B | 2026-01-20 | [step-3-vl-10b](https://artificialanalysis.ai/models/step-3-vl-10b) |
| LFM2.5-1.2B-Thinking | 2026-01-20 | [lfm2-5-1-2b-thinking](https://artificialanalysis.ai/models/lfm2-5-1-2b-thinking) |
| GLM-4.7-Flash (Reasoning) | 2026-01-19 | [glm-4-7-flash](https://artificialanalysis.ai/models/glm-4-7-flash) |
| GLM-4.7-Flash (Non-reasoning) | 2026-01-19 | [glm-4-7-flash-non-reasoning](https://artificialanalysis.ai/models/glm-4-7-flash-non-reasoning) |
| Olmo 3.1 32B Instruct | 2026-01-13 | [olmo-3-1-32b-instruct](https://artificialanalysis.ai/models/olmo-3-1-32b-instruct) |
| LFM2.5-1.2B-Instruct | 2026-01-05 | [lfm2-5-1-2b-instruct](https://artificialanalysis.ai/models/lfm2-5-1-2b-instruct) |
| LFM2.5-VL-1.6B | 2026-01-05 | [lfm2-5-vl-1-6b](https://artificialanalysis.ai/models/lfm2-5-vl-1-6b) |
| Falcon-H1R-7B | 2026-01-04 | [falcon-h1r-7b](https://artificialanalysis.ai/models/falcon-h1r-7b) |

## 2026-09-18 DeepSeek V3.2 官方核验增补

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| DeepSeek V3.2 | Artificial Analysis：[`deepseek-v3-2`](https://artificialanalysis.ai/models/deepseek-v3-2)，当前为 `Non-reasoning`，2025-12，128K；DataCurve：当前无精确行 | [V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)、[V3.2-Exp 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | AA 单榜资料级闭环；2026-09-20 AA 快照 `3,625,720` bytes / `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`，报告正文已核验 DSA 两阶段训练、2,048 KV top-k、约 2.1B/943.7B tokens、复杂度边界、specialist distillation、GRPO 稳定化和 Agent 合成/验证规模；实现证据已补入第二十一册第 19 章 19.29--19.32；公式视觉复核、完整 kernel/层排布/recipe、线上 acceptance rate、硬件 profiling 和独立 benchmark 待补证 |

2026-09-20 实现复核补充：V3.2-Exp inference demo 已核验 FP8 indexer、non-interleaved indexer RoPE、`fp8_index`、causal/top-k、prefill MHA、decode MQA、latent KV/position cache 和 FP8 KV cache；TileLang、DeepGEMM PR #200、FlashMLA PR #98 与 vLLM recipe 已纳入来源索引。固定官方 V3.2 `config.json` 与实验 inference config 都是 `q_lora_rank=1536`，但分属不同 revision/artifact，不能合并为完整生产配置；vLLM 的 GSM8K 数字按 recipe/harness 结果记录，不升级为模型能力。

## 2026-09-18 GPT-5.3 Codex 锚点补充

本节只记录已经在 Artificial Analysis 中出现的 GPT-5.3 Codex；DataCurve 当前没有精确同名配置，不能迁移其他 GPT/Codex 行的 Agent 结果。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| GPT-5.3 Codex (xhigh) | Artificial Analysis：`gpt-5-3-codex`；AA 页面标题 `GPT-5.3 Codex (xhigh)`，release date 字段 `2026-02-05`；DataCurve：无精确 `mini_swe_agent_gpt_5_3_codex_*` 行 | [OpenAI GPT-5.3-Codex 模型页](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md)、[Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)、Compaction/Conversation state/Tools/Agents/Prompt caching | 资料级闭环；400K/272K/128K、Responses-only、Codex harness、phase、replay、compaction、工具契约和缓存语义已核验；参数、架构、训练 recipe、专属报告和独立评测待核验 |

Artificial Analysis 详情快照为 `3,528,646` bytes、SHA-256 `565d91572b1a0bd9fb8f7f89f16c8beefbaadfdea79de5b229a9bd5a997b27ef`，第三方 Intelligence Index `32.5028174368983`（estimated）、context `400,000`、knowledge cutoff `2025-08-31`。这些字段不等于官方模型规格。DataCurve 没有精确行，因此本节不记录 GPT-5.3 Codex 的 Pass@1、成本、输出 token 或 Agent steps。

官方模型页确认 `gpt-5.3-codex` 支持 `low/medium/high/xhigh` effort、文本/图像输入、文本输出、400K context、272K maximum input、128K maximum output、Responses API、function calling、web search、hosted shell 和 skills。官方 Codex 指南进一步把较少 reasoning tokens、长时自治、first-class compaction、工具 schema、并行调用、`apply_patch`、固定工作目录和 phase/replay 作为 harness 设计重点。

当前不新增 GPT-5.3 Codex 专属 Transformer 章节。它作为 Agent runtime 锚点映射到第六册、七册、十六册、十七册、二十册和二十四册；完整研究、快照哈希和待核验内容见 [`gpt-5.3-codex-source-notes.md`](gpt-5.3-codex-source-notes.md)。

## 2026-09-18 OpenAI gpt-oss 锚点补充

本节只升级已经在 Artificial Analysis 中出现的 `gpt-oss-120b` 与 `gpt-oss-20b`；官方模型卡、代码仓库和 Cookbook 用于核验和扩展技术，不作为新的模型发现入口。DataCurve 当前没有两个模型的精确 `mini_swe_agent` 行。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| gpt-oss-120b (high) | Artificial Analysis：[`gpt-oss-120b`](https://artificialanalysis.ai/models/gpt-oss-120b)；DataCurve：无精确行 | [Model Card/arXiv](https://arxiv.org/abs/2508.10925)、[官方仓库](https://github.com/openai/gpt-oss)、[120B 模型卡](https://huggingface.co/openai/gpt-oss-120b) | 内容专题闭环；116.8B total、5.13B active、128 experts/top-4、MXFP4、Harmony、可变 reasoning 已核验；完整 recipe、kernel、profiling 和精确 Agent 评测待核验 |
| gpt-oss-20b (high) | Artificial Analysis：[`gpt-oss-20b`](https://artificialanalysis.ai/models/gpt-oss-20b)；DataCurve：无精确行 | [Model Card/arXiv](https://arxiv.org/abs/2508.10925)、[20B 模型卡](https://huggingface.co/openai/gpt-oss-20b)、[Harmony](https://github.com/openai/harmony) | 同一模型族的 sibling 尺寸，不另建重复架构章节；20.9B total、3.61B active、32 experts/top-4、约 16GB memory 级部署目标按官方资料记录 |

Artificial Analysis 详情快照：120B `3,701,133` bytes，SHA-256 `197c0a6caa8791acd9803daf2d656596b77c84aae448ad9d788e01099d8e4940`；20B `3,701,427` bytes，SHA-256 `7990041b97d0ea2f04122ca6817f9f64c6e3b756b5d30ba433e92c3fd3d630fa`。AA 的 Intelligence Index、速度、价格、context 和 `high` 配置保持第三方字段，不与官方 Model Card benchmark 拼成裸模型排名。

### 2026-09-23 当前时点复验

通过 `10.24.27.134:7890` 重新取得两榜：AA 中文首页 `1,783,626` bytes / SHA-256 `0580fad58c167fb96e87289addbabccd40cbd60302c527437805c8eb6ee7530b`，DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；百度、两榜均 HTTP `200`。八家重点厂商的 canonical 集合没有因本次复验产生新模型。

| 基础模型/配置 | AA 当前详情快照 | AA 当前第三方字段 | DataCurve |
|---|---|---|---|
| `gpt-oss-120b (high)` | `4,078,295` bytes / `43e5f13a3e58976b077acac1810bce82ac60ce33a7d81476f0dc2b17175532de` | 117B total、5.1B active、131,072 context、Index `11.6028431512592`、约 `196.389235173389 tokens/s`、cost/task `0.10742452290394947` | 无精确 `mini_swe_agent_gpt_oss_120b_*` |
| `gpt-oss-20b (high)` | `4,083,068` bytes / `64d67047a5964c0109f103108f9b77c64edbaf1ee17f3fec3c82ffccd7cea2cc` | 21B total、3.6B active、131,072 context、Index `8.9675171856126`、约 `185.656167974395 tokens/s`、cost/task `0.012460640242179213` | 无精确 `mini_swe_agent_gpt_oss_20b_*` |

本节字段是 AA/provider 在当前采集时点的观察值；详情页字节变化不等于模型 revision 变化。DataCurve 的缺失仅作为当前快照负证据，不迁移其他 OpenAI/Codex 模型的 Agent 结果。

官方资料确认两个 text-only autoregressive MoE Transformer 交替使用 sliding-window/full attention，采用 GQA、RoPE/YaRN、post-training MXFP4、`o200k_harmony` tokenizer、CoT RL、Harmony response format 和 `low/medium/high` variable-effort reasoning。9 月 22 日 Cookbook 补证了 provider 兼容性 smoke test、Responses `reasoning_text` raw CoT、Chat Completions `reasoning` 约定、AIME/GPQA/HealthBench 分层 eval 以及 raw CoT 的终端展示安全边界。正式专题见第二十一册第 87 章 [`GPT-OSS：开放权重 MoE、Harmony 协议与可变推理`](../../book-21-transformer-architecture-evolution/chapters/87-gpt-oss开放权重moe-harmony与可变推理.md)；完整证据和待核验项见 [`gpt-oss-source-notes.md`](gpt-oss-source-notes.md)。

## 2026-09-18 Claude Opus 4.6 官方核验增补

本节只升级已经出现在 Artificial Analysis 的 Claude Opus 4.6；DataCurve 当前没有精确同名 `mini_swe_agent` 行，因此不迁移相邻 Claude 版本的 Agent 结果。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Claude Opus 4.6 | Artificial Analysis：`claude-opus-4-6-adaptive`、`claude-opus-4-6`；DataCurve：无精确行 | [发布页](https://www.anthropic.com/news/claude-opus-4-6)、[System Card](https://www.anthropic.com/claude-opus-4-6-system-card)、[模型页](https://platform.claude.com/docs/en/models/opus-4-6/overview.md)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) | 资料级闭环；1M/128K、adaptive thinking、effort、thinking signature、server-side compaction、tool search 和 computer-use 执行器边界已核验；参数、架构、训练 recipe 和独立复现待核验 |

AA 两个详情页的 release date 字段均为 `2026-02-05`；adaptive 页面字段约为 Intelligence Index `32`（estimated）、37.7 output tokens/s、19.75s TTFT，基础页面与其属于同一模型的配置级观察。价格、速度、指数和 context 是 Artificial Analysis 字段，不与 Anthropic 发布方 benchmark 混写。

官方资料确认 `claude-opus-4-6` 为 legacy snapshot，默认 `high`、可选 `max/high/medium/low`，支持 adaptive thinking；thinking block 的 encrypted signature 在工具循环中必须原样回传。`compact-2026-01-12` beta 下可返回 compaction block，默认约 150K input tokens、最低 trigger 约 50K；tool search 使用 `defer_loading`，regex/BM25 结果默认最多 5 个 `tool_reference`；computer use 仍是 `computer_20251124`，宿主负责执行器、沙箱、allowlist、人工确认和 prompt-injection 防护。

发布页的 retrieval、BigLaw Bench、盲测和 subagent 数字按 Anthropic 自报记录；arXiv 精确标题检索没有发现 Opus 4.6 官方技术报告，外部论文不升级为官方证据。完整研究笔记见 [`claude-opus-4.6-source-notes.md`](claude-opus-4.6-source-notes.md)。

## 2026-09-18 Claude Opus 4.7 锚点补充

本节只记录已经出现在 Artificial Analysis 的 Claude Opus 4.7；DataCurve 当前没有精确 `mini_swe_agent_claude_opus_4_7_*` 行，因此不迁移 Opus 4.6、4.8、5 或其他 Claude 版本的 Agent 结果。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Claude Opus 4.7 adaptive/max 与 non-reasoning/high | Artificial Analysis：[`claude-opus-4-7`](https://artificialanalysis.ai/models/claude-opus-4-7)、[`claude-opus-4-7-non-reasoning`](https://artificialanalysis.ai/models/claude-opus-4-7-non-reasoning)；DataCurve：无精确行 | [发布页](https://www.anthropic.com/news/claude-opus-4-7)、[System Card](https://www.anthropic.com/claude-opus-4-7-system-card)、[模型页](https://platform.claude.com/docs/en/models/opus-4-7/overview.md)、[Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)、[Vision](https://platform.claude.com/docs/en/build-with-claude/vision)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) | 内容专题闭环（复用 Agent/tool 与 inference serving 章节）；`xhigh`、三层预算、1M/128K、更新 tokenizer、高分辨率视觉和 cyber safeguards 控制面已核验；参数、架构、训练 recipe 和独立复现待核验 |

Artificial Analysis 详情页给出 release date `2026-04-16`、1M context；max 配置 Intelligence Index 约 `40.6898`（estimated），non-reasoning/high 约 `30.9317`（estimated）。本轮 Artificial Analysis `/zh` 快照为 `1,769,141` bytes、SHA-256 `5999975b2bddb1f1280677a3a98646f6f8bf59622865d3aa54ba4fe97008c68c`，DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；这些字段仅用于榜单证据追溯。

官方模型页确认 `claude-opus-4-7`、128K synchronous max output、300K Batch output、adaptive thinking 和默认 `high` effort；官方文档确认 `xhigh` 位于 `high` 与 `max` 之间，task budget 覆盖 thinking/tool calls/tool results/output，server-side compaction 不重置当前 turn 的已用预算。Vision 文档确认 Opus 4.7 及之后的 high-resolution tier 为最长边 `2576 px`、最多 `4784` visual tokens，普通档位为 `1568 px`/`1568` visual tokens。

更新 tokenizer 可能令相同输入变为旧版本的约 `1.0-1.35x` token；发布页/System Card 的 cyber safeguards 与 Cyber Verification Program 按部署和安全控制面记录，不能升级为内部架构或训练算法。完整证据见 [`claude-opus-4.7-source-notes.md`](claude-opus-4.7-source-notes.md)。

## 2026-09-21 Claude Opus 4.7 runtime recheck

本轮只复核已在 Artificial Analysis 出现的 Opus 4.7，不从 Anthropic 官方目录或文档另发现模型。Artificial Analysis `/zh` 三条代理均为 HTTP 200、`1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。当前 max 字段为 Intelligence Index `40.6897928205908`（estimated）、约 `51.1354` output tokens/s、1M context、`$5/$25` input/output；这些仍是第三方目录/测量字段。DataCurve 三条代理仍为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 Opus 4.7 行。

AA 详情页的 deprecated 是第三方目录状态；官方模型页在同一轮仍把 `claude-opus-4-7` 标为 `Active (legacy)`，并建议迁移 Opus 5。该生命周期差异不改变榜单锚点身份，也不授权迁移 Opus 4.6/4.8/5 的 DataCurve 结果。完整运行时技术与官方资源哈希见 [`claude-opus-4.7-source-notes.md`](claude-opus-4.7-source-notes.md)。

## 2026-09-18 Gemini 3 Deep Think 配置级核验增补

本节只升级已经出现在 Artificial Analysis 的 `Gemini 3 Deep Think`；Google 官方资料用于核验和扩展该榜单配置，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Gemini 3 Deep Think | Artificial Analysis：[`gemini-3-deep-think`](https://artificialanalysis.ai/models/gemini-3-deep-think)，`releaseDate=2026-02-05`；DataCurve：无精确行 | [Gemini 3.1 Pro API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)、[Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)；官方 3.1 Pro/3 系列资料未提供独立 Deep Think model ID、权重或专属技术报告 | 资料级闭环（配置级锚点）；`thinking_level`、共同 output budget、signature/id 回放、1M context/caching 和 test-time compute 评测边界已核验，不新增独立架构章节 |

Artificial Analysis 的参数规模、约 130K context、指数、速度、价格和 provider 字段均保留为第三方配置级目录信息；不能替代 Google 官方规格，也不能把 Gemini 3.1 Pro 或其他 Gemini 配置的 DeepSWE 结果迁移给该条目。详细证据见 [`gemini-3-deep-think-source-notes.md`](gemini-3-deep-think-source-notes.md)。

## 2026-09-18 Qwen3.5-397B-A17B 锚点补充

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Qwen3.5-397B-A17B | Artificial Analysis：[`qwen3-5-397b-a17b`](https://artificialanalysis.ai/models/qwen3-5-397b-a17b)，Reasoning/Non-reasoning 按同一基础模型归并；DataCurve：无精确 `mini_swe_agent_qwen3_5_397b_a17b_*` 行 | [Qwen3.5-397B-A17B 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)、[config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json)、[Qwen3.8 官方仓库](https://github.com/QwenLM/Qwen3.8)、[Qwen3.5 博客](https://qwen.ai/blog?id=qwen3.5) | 内容专题闭环；397B/17B、Gated DeltaNet/Gated Attention、512 experts、原生视觉、MTP、262K/约 1.01M context 和 serving 边界已核验；完整 recipe、kernel、MTP acceptance、视觉独立复现和硬件 profiling 待核验 |

Artificial Analysis 详情快照 `/tmp/qwen35-397-aa.html` 为 3,700,616 bytes，SHA-256 `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719`；第三方 context、指数、速度和价格字段不与官方 benchmark 拼成裸模型能力。Qwen 官方关于 early-fusion、多模态 token、million-agent RL、异步 RL 和 201 languages/dialects 的内容保持发布方自报边界。

Qwen3.5-Plus 是模型卡声明的 hosted counterpart，不新增为独立 open checkpoint。Qwen3.8 的 QSA、Gated Residual、N-gram Embedding 和 Muon 不反向迁移给 Qwen3.5；研究笔记见 [`qwen3.5-397b-a17b-source-notes.md`](qwen3.5-397b-a17b-source-notes.md)，正式落点为第二十一册第 83 章新增前置/对比小节。

## 2026-09-20 Gemini 3.5 Flash-Lite 官方核验增补

本节只升级已经出现在 Artificial Analysis 的 `Gemini 3.5 Flash-Lite`；DataCurve 当前没有精确 Lite Agent 行。Google 官方资料用于核验和扩展周边技术，不是新的模型发现入口。

| 模型 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Gemini 3.5 Flash-Lite | Artificial Analysis：`gemini-3-5-flash-lite`，第三方 `releaseDate` `2026-07-21`、Intelligence Index `22.1685424839812`、`isReasoning=true`；DataCurve 无 `mini_swe_agent_gemini_3_5_flash_lite_*` | [API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) | 资料级闭环（AA 单榜）；1M/65K、低延迟高吞吐定位、默认 minimal thinking、agentic video、发布方评测和安全边界已核验；模型卡明确基于 Gemini 3.1 Flash-Lite，独立架构/训练报告待核验 |

Google 模型页给出 text/image/video/audio/PDF 输入、text 输出、caching、code execution、Computer Use Preview、File Search、function calling、Maps/Search grounding、structured output、thinking 和 URL context；视频文档把 Lite 列入 agentic processing，支持模型按需读取时间轴。不能把 Gemini 3.5 Flash 的 `medium` 默认 thinking、GA 或 thought preservation 回写到 Lite。

## 2026-09-20 Kimi K2.6

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Kimi K2.6 / Kimi K2.6 (Non-reasoning) | Artificial Analysis：[`kimi-k2-6`](https://artificialanalysis.ai/models/kimi-k2-6)；DataCurve：无精确 `mini_swe_agent_kimi_k2_6_*` 行 | [模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)、[固定配置](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)、[部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)、[官方博客](https://www.kimi.com/blog/kimi-k2-6)、[Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html) | AA 单榜资料级闭环；1T/32B、MLA、MoonViT、native INT4、Agent Swarm、thinking/tool 协议和 KVV 分层已核验；完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 和精确 DataCurve 行待核验 |

Artificial Analysis 2026-09-20 详情快照为 `3,938,140` bytes，SHA-256 `f24ad7d9cd3ddbb253470750dcf9f47aea3fc0d675c9f8f7eb8bdbcd50dcfe18`；页面的 deprecated/K3 迁移提示是第三方目录状态。不能迁移 K2.7 Code 或 K3 的 DeepSWE 结果，也不能把 300 sub-agents 写成 MoE experts。

Kimi Code 当前官方模型配置表另列 K3、K2.8 Preview 和 K2.7 Code HighSpeed 共 4 个 model ID；其中 K2.8 在 2026-09-20 的 AA/DataCurve 快照中没有精确条目，故只作为官方周边文档中的关联版本和负面证据，不进入本项目候选盘点。当前 CLI `AgentSwarm` 的 128 subagent、2 小时超时、resume/model pool、聚合报告、并发和权限复核属于 harness，不与 K2.6 博客的 300 sub-agents/4,000 steps 案例合并。

研究笔记：[`kimi-k2.6-source-notes.md`](kimi-k2.6-source-notes.md)；正式章节：[`第二十一册第 90 章`](../../book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)。

## 2026-09-20 DeepSeek V4 Pro 0813 当前活动锚点

本节只升级已经出现在两个排行榜中的 `DeepSeek V4 Pro`；官方 API、模型卡、配置和技术报告用于核验及扩展周边技术，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| DeepSeek V4 Pro 0813 (Reasoning, Max Effort) | Artificial Analysis：[`deepseek-v4-pro`](https://artificialanalysis.ai/models/deepseek-v4-pro)，release date `2026-08-13`；DataCurve：`mini_swe_agent_deepseek_v4_pro_max`，`max` | [V4 Pro GA](https://api-docs.deepseek.com/news/news260813)、[Quick Start](https://api-docs.deepseek.com/quick_start)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)、[技术报告](https://arxiv.org/abs/2606.19348) | 内容专题闭环（双榜锚点）；1.6T/49B、1M、CSA/HCA、mHC、Muon、FP4/FP8、SFT+GRPO+on-policy distillation、Responses stateless 边界已核验；生产 kernel、硬件 profiling、线上 acceptance、完整独立复现和完整 recipe 待核验 |

AA 详情快照为 3,934,926 bytes，SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`；第三方字段为 Intelligence Index `35.9967791278402`、1,000,000 context、约 1.6T/49B total/active、开放权重和 MIT。DataCurve 精确行的 Pass@1/Pass@4 为 `62.831858%`/`88.495575%`，`n_runs=4`，平均成本 `$1.6660232187`，平均输出 `105998.9` token，平均 `154.71` Agent steps。两组字段分别绑定 AA provider 测量与 `mini-swe-agent` harness，不能拼接为裸模型结论。

官方配置公开 61 层、384 routed experts、每 token 6 个 routed experts、1 个 shared expert、hidden size 7168、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024`、YaRN factor 16 和 FP8 quantization。Responses 文档的 stateless、function tools、`apply_patch` 和静默忽略不支持参数属于协议层行为；不能由此推断内部 reasoning 实现。研究笔记见 [`deepseek-v4-source-notes.md`](deepseek-v4-source-notes.md)，正式内容复用第二十一册第 77/78 章及既有 DeepSeek V4/V4.1 章节。

### 2026-09-20 官方实现证据补强

固定 HF revision 为 `b5968e9190ef611bbf34a7229255be88a0e937c1`，API metadata 列出 64 个 safetensors 分片，总 storage `1,598,839,674,782` bytes；完整权重未下载。固定 `encoding/` 与 `inference/` 文件的大小和 SHA-256 见研究笔记，避免把可变 `main` 分支当作同一 artifact。

实现账本新增：`inference/config.json` 的 `n_hash_layers=3`、`sqrtsoftplus`、`window_size=128`、64 个 index heads、128 head dim、`hc_mult=4`、20 次 Sinkhorn、FP4 expert dtype 和 128/4 compression ratios；`model.py` 的 gated compressor/overlap state、causal top-k indexer、MLA + compressed sparse attention、前三层 hash routing、top-6 + shared expert、MTP 与 Hyper-Connections；`kernel.py` 的 block FP8/FP4 quantization、GEMM、sparse online softmax 和 HC Sinkhorn。该层把 V4 Pro 升级为“内容专题 + reference implementation evidence”，仍不等于完整权重本地加载、生产 kernel profiling 或线上 acceptance。

## 2026-09-20 GLM-5.1 官方核验增补

本节只升级已经出现在 Artificial Analysis 的 `GLM-5.1`；DataCurve 当前没有精确 `mini_swe_agent_glm_5_1_*` 行，因此不把 GLM-5、GLM-5.2、GLM-5.3 或 GLM-5.3-Flash 的 DeepSWE 结果迁移给它。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| GLM-5.1 (Reasoning / Non-reasoning) | Artificial Analysis：[`glm-5-1`](https://artificialanalysis.ai/models/glm-5-1) 与 [`glm-5-1-non-reasoning`](https://artificialanalysis.ai/models/glm-5-1-non-reasoning)，2026-04-07；DataCurve：无精确行 | [Z.ai 模型文档](https://docs.z.ai/guides/llm/glm-5.1)、[Hugging Face README](https://huggingface.co/zai-org/GLM-5.1/raw/main/README.md)、[配置](https://huggingface.co/zai-org/GLM-5.1/raw/main/config.json)、[Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md) | AA 单榜资料级闭环；长周期 Agent、过程质量评估、DSA/MoE 配置字段、thinking/tool/cache 协议已核验；独立 GLM-5.1 报告、完整 recipe、生产 kernel、硬件 profiling、精确 DataCurve 行和独立复现待核验 |

Artificial Analysis 详情快照为 `3,935,095` bytes，SHA-256 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`；页面第三方字段约为 `744B/40B`、`200K` context、Intelligence Index `26.0585912980095` 和约 `39.9 tokens/s`。这些属于配置/provider 测量。官方配置公开 `GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、前三层 dense、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048` 和 `202752` positions；配置不提供完整参数账本或生产 kernel 证明。

Z.ai 文档、release notes 和 2026-09-21 重新取得的官方博客将 GLM-5.1 定位为最长约 8 小时的 long-horizon agentic engineering 模型，并提到 multi-turn SFT、RL 与 process-quality evaluation framework。博客进一步给出 VectorDBBench 的 600+ iterations/6,000+ tool calls/21.5k QPS、KernelBench Level 3 的 H100/1,200-turn/双审计器设置，以及无标量目标的 8 小时 Linux desktop 自评 harness。SWE-Bench Pro `58.4`、上述长任务数字和 KernelBench Level 3 `3.6×` 对比 `torch.compile` max-autotune `1.49×` 均按发布方自报记录，不能与榜单或 DeepSWE 组合成裸模型能力。模型卡链接的是 GLM-5 技术报告；arXiv 精确检索没有 GLM-5.1 专属报告。

研究笔记：[`glm-5.1-source-notes.md`](glm-5.1-source-notes.md)。不新增重复 Transformer 正式章节；内容映射到第二十一册 GLM-5 DSA/MoE、Agentic Engineering、十六册 reasoning、十七册工具协议和第二十四册 serving 主线。下一锚点仍只从两个排行榜的八家重点厂商条目选择。

## 2026-09-20 Grok 4.20 官方核验增补

本节只升级已经出现在 Artificial Analysis 的 `Grok 4.20 0309 v2 (Reasoning)`；DataCurve 当前没有精确的 `mini_swe_agent_grok_4_20_*` 行，因此不迁移 Grok 4.5/4.6 的 DeepSWE 结果。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Grok 4.20 0309 v2 (Reasoning) | [Artificial Analysis](https://artificialanalysis.ai/models/grok-4-20)，2026-04-07，estimated Intelligence Index `25.6550155187053`、2M context；DataCurve 无精确行 | [xAI 模型页](https://docs.x.ai/developers/models/grok-4.20)、[Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)、[Tools](https://docs.x.ai/developers/tools/overview.md)、[Release Notes](https://docs.x.ai/developers/release-notes.md) | AA 单榜内容专题闭环（runtime/API 与 harness）；官方主 ID、1M prompt、普通/non-reasoning/multi-agent 服务区分、4/16 agent-count、beta/API/tool 限制、加密 continuation、全体 Agent/tool 计费、compaction 和证据边界已核验；架构、训练 recipe、独立报告和 DataCurve 精确行待核验 |

AA 三代理详情历史快照均为 513,564 bytes、SHA-256 `2fc88248faf152f46f659310f300f2fe3e5b84e419babb9c8c7dc3752452cdd8`。xAI 模型页 HTML 为 396,958 bytes、SHA-256 `7997425c354b938ae78881b6c0b5d6a4efc9c10dfbf687c70828575c80fd3f34`；Markdown 为 1,564 bytes、SHA-256 `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`。DataCurve 2026-09-20 历史响应为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。

- 2026-09-24 复访：AA Grok 4.20 详情 HTTP 200，`3,857,337` bytes / SHA-256 `d56d4557d8a661ee9c3015c6a3661f5baae6c00f9b16c5b2acf5eabcd3e61300`；默认 10K input token workload 的性能基准仍更新，其他 workload 明确属于历史数据且不再更新。当前页面显示速度 `106.190642707844 tokens/s`、TTFT `21.53034693s`、端到端 `26.23885972594964s`；指标抓取时点不等于其 benchmark workload 的最后更新时间。
- DataCurve 2026-09-24 HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；精确 `mini_swe_agent_grok_4_20_*` 行仍缺失。xAI 模型 Markdown 仍为 HTTP 200、1,564 bytes / SHA-256 `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`。

xAI 官方文档把 `grok-4.20-0309-reasoning`、`grok-4.20-0309-non-reasoning` 和 `grok-4.20-multi-agent-0309` 作为不同服务对象；普通模型页写 1M context，Artificial Analysis 写 2M，必须按来源和 endpoint 分账。Multi-agent 文档把 `reasoning.effort=low/medium` 映射为 4 agents、`high/xhigh` 映射为 16 agents，并由 leader 汇总；这不是普通模型的思考深度，也不是 MoE expert 数量。Context Compaction 返回不可解析的 opaque item，Remote MCP 的 `allowed_tools` 同时影响上下文开销和最小权限。

Release Notes 的 March 条目确认 Grok 4.20/Multi-agent live；三个可能的 xAI 新闻路径本轮 HTTP 404，arXiv 标题精确检索无 Grok 4.20 专属报告。完整证据和待核验项见 [`grok-4.20-source-notes.md`](grok-4.20-source-notes.md)。

2026-09-29 更新：xAI Multi Agent 专页 HTTP 200，19,603 bytes / SHA-256 `0463d9d2022fd9453ece718e3fc898103c9cab20dc8cd4a2601d7c92b0fab6d4`。页面声明 beta；支持 xAI SDK/Responses、内置工具、Remote MCP 和 `previous_response_id`，不支持 Chat Completions、client-side/custom function tools 或 `max_tokens`。leader 与子 Agent 的全部 tokens 和 server-side tool calls 计费；细节及成本字段见研究笔记 2026-09-29 补证段。

## 2026-09-20 Gemini 3.8 Flash 当前快照复验

Gemini 3.8 Flash 仍只作为两个排行榜中已出现的 canonical 锚点。Artificial Analysis high/medium/low 三个 effort 页面本轮与 8098 代理逐字节一致，DataCurve 仍只有 high 的精确 `mini_swe_agent_gemini_3_8_flash_high` 行；不把三个 effort 当作三个 checkpoint，也不把相邻 Gemini 版本的 Agent 数字迁移过来。

| 证据层 | 本轮结果 |
|---|---|
| AA 首页 | 1,777,695 bytes，SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912` |
| AA 详情 | high/medium/low 分别 517,803/511,687/509,053 bytes；SHA-256 `0618989e412ae947b3a9f499c1cdae16e4ea3f5ee65a25def360249ae33985ef` / `d96907dd254166f038084e7586e30ab869fc8a3225501c37c958650859aec6a8` / `ef35f6f567518bd1b9e9e1bc17e4826ea479b7d1e7868bc35f8daf7c464ee5b3` |
| DataCurve | 268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；high Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本 `$2.3623`、平均 166.31 steps |
| Google 官方 | 模型页/Thinking/Interactions 通过 7890 获取，大小 23,603/37,414/23,899 bytes；对应哈希已登记在来源索引和研究笔记 |

截至本节 2026-09-20 快照，状态记录为**资料级闭环**；随后 Interactions 正式章节及题库同步完成，当前状态已升级为**内容专题闭环**。2026-09-28 又补入官方 Video Understanding 中 Gemini 3.8 Flash 的 agentic video processing：静态 1 FPS 与按需读取、`processing_call`/`processing_result`、thought/tool-use token 账本和短视频 TTFT 权衡。该补充仍是 API/runtime 能力，不是视觉架构证据；3.8 独立架构、参数、训练配方、生产 kernel、硬件 profiling 和独立复现仍待核验。

## 2026-09-20 GLM-5.3 当前快照与后训练资料补证

| 证据层 | 本轮结果 |
|---|---|
| Artificial Analysis | `glm-5-3`，max；3,928,329 bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`；release `2026-08-18`、指数 `44.777392385614`、约 72.1152 t/s、1M、`$1.40/$4.40` |
| DataCurve | `mini_swe_agent_glm_5_3_max`；268,571 bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；Pass@1 `68.9579%`、Pass@4 `87.6106%`、451 attempts、平均 124.47 steps |
| Z.ai / HF | 文档、迁移指南、thinking/function/tool streaming/cache/structured output、官方博客资源、HF README/config 和 slime 已复验 |
| 论文关联 | IndexCache（跨层 index 复用）与 SAO（单 rollout 异步 RL）；SAO 直接对象是 GLM-5.2，不改写为 GLM-5.3 独有算法 |

GLM-5.3 的新增面试知识集中在“后训练系统工程”：可执行长周期环境、无 reference verifier、oracle/no-op/unsolved-state 门禁、SAO with compaction 的继承关系、slime 的 train/rollout/data-buffer 单 dataflow、training-rollout 数值一致性与工具流式协议。当前状态为**双榜资料级闭环**；不把 Z.ai 自报 benchmark、AA 参数/指数或 config 字段拼成完整内部训练事实。

## 2026-09-20 Kimi K3 实现证据补充

| 基础模型/配置 | 榜单发现 | 新增官方实现证据 | 当前状态 |
|---|---|---|---|
| Kimi K3 (low/max) | Artificial Analysis：[`kimi-k3`](https://artificialanalysis.ai/models/kimi-k3)；DataCurve：K3 的 `mini-swe-agent` 配置 | HF 官方 revision `f831ab66814297da540d832a5235f8e904f29d06`、固定 `config.json`；[FlashKDA](https://github.com/MoonshotAI/FlashKDA) master commit `7afb9f454f160a6c4bbc0999beca0a8c40a38934`；[vLLM stable supported-models](https://docs.vllm.ai/en/stable/models/supported_models/)、[K3 API](https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/)、[v0.29.0 registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/models/registry.py)、[vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3) | 内容专题闭环 + 实现资料补证；vLLM `0.29.0` stable release/source 已有 K3 implementation entry，但 recipe 仍为 `Pre-release`/K3-enabled nightly；完整权重未下载，未做本地 wheel 安装、目标硬件加载、完整 recipe、profiling、hybrid cache recovery、独立复现和线上 acceptance |

固定 config 再确认 `KimiK3ForConditionalGeneration`、93 layers、69 KDA + 24 full-attention/Gated MLA、`q_lora_rank=1536`、`kv_lora_rank=512`、896 routed experts/top-16/2 shared 和 1,048,576 context。FlashKDA README 记录 SM90+/CUDA 12.9+/PyTorch 2.4+、`chunk_kda` backend、recurrent state 与变长 batch；H20/GB200 固定长度 benchmark 相对仓库 baseline 分别为 `1.85x`/`2.31x`，属于发布方实现 benchmark，不是本地复现。vLLM recipe 要求 0.29.0/K3-enabled nightly，hybrid KV manager 同时管理 MLA cache 与 KDA state，并提示 tool-call parser 需要宿主 schema validation、retry 和 verifier。SGLang stable/main 对照见下方 2026-09-22 记录。

## 2026-09-21 Kimi K3 固定 manifest 与源码审计

| 基础模型/配置 | 新增证据 | 审计结果 | 当前状态 |
|---|---|---|---|
| Kimi K3 (low/max) | 固定 revision 的 `model.safetensors.index.json`、`modeling_kimi_linear.py` 与零依赖审计脚本 | `497,220` tensors、96 个连续分片、93 层、69 KDA/24 full-attention、92 个 expert-bearing layer、每层 896 experts；`weight_packed/weight_scale` 各 `247,296` 且成对 | 内容专题闭环 + 固定 manifest/config/runtime source evidence；完整权重未下载，未完成 CUDA/线上 serving 验收 |

index `metadata.total_size=1,560,860,324,864` 是 MXFP4 packed 分片口径；HF API 的 `safetensors.total=2,779,931,837,184` 是 U8/BF16/F32 参数统计，不能合并为一个模型大小。源码显示 full-attention cache 与 KDA `conv_states/recurrent_states` 分开维护，单 token decode 使用 fused recurrent path，chunk/prefill 使用 chunk path；恢复和 prefix serving 必须同时记录两类状态。vLLM stable supported-models 页面为 `738,169` bytes / SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`，stable K3 API 页面为 `799,430` bytes / SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`；main registry 为 `64,391` bytes / SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，package init 为 `1,444` bytes / SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`。这些是实现入口证据，不等于本地 wheel 安装、完整权重加载或生产 serving 验收。完整证据见 [`kimi-k3-source-notes.md`](kimi-k3-source-notes.md)。

补充修正：本轮通过 PyPI metadata 固定 vLLM `0.29.0` stable release；x86_64 wheel 为 `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，2026-09-09 上传。公开 v0.29.0 tag 的 registry 为 `63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`，NVIDIA K3 model 为 `87,313` bytes / SHA-256 `e74026a83c28cce7ee62889e9ed434d1fff6076dbd263d388aa379f4cf987a4d`。因此 K3 已有 stable release implementation evidence；recipe 的 nightly/pre-release 是优化部署路径，不是 stable release 不存在。目标硬件加载、完整权重、双状态恢复、profiling、线上 acceptance 和独立复现仍待核验。

## 2026-09-22 Kimi K3 SGLang runtime 对照

| 基础模型/配置 | 新增 source evidence | 当前解释 |
|---|---|---|
| Kimi K3 (low/max) | SGLang `main` `kimi_k3.py` 171,101 bytes / blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`；SGLang `v0.5.20` `kimi_k3.py` 168,114 bytes / blob `b0ede48c88264d518351a66abf623f1bcf8a730e`；两版本 `kimi_k3_vl.py` 相同，blob `c423027994645e4a839ec47bd26bcc1b2a7cb012` | stable tag 已有 K3 text/vision implementation entry；main 继续演进 DP/SP token shard、A2A、shared-expert TP/reduce-scatter、SBO/NPU overlap、ModelSlim mapping 和 KDA fused decode gate；不等于 stable wheel、完整权重或生产 serving 已验收 |

`main` 的 `KimiK3MoE` 将 routed experts 放在 latent width，并识别 MegaMoE、DeepEP、Mooncake、Ascend-FuseEP 和 MoRI A2A；shared expert 可按 group 做 gather/TP-sharded MLP/reduce-scatter。KDA 路径有 shape/dtype capability gate 和普通 fallback；视觉实现两版本逐字节一致，包含 MoonViT3d、2D RoPE、变长 metadata 和形状选择的视觉 attention backend。完整证据见 [`kimi-k3-source-notes.md`](kimi-k3-source-notes.md) 与第二十一册第 88 章；当前仍待完整权重加载、双状态恢复、目标硬件 profiling、视觉正确性和 tool/verifier acceptance。

## 2026-09-22 Kimi K3 SGLang main commit history 补证

本轮通过 10.237.126.170:1234 取得 SGLang kimi_k3.py 的最近 10 个 commit。GitHub API 响应为 51,251 bytes，SHA-256 为 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，最新提交截至 2026-09-22T06:35:12Z。当前 main 文本源码复抓后仍为 171,101 bytes、blob 383a6f47812bccd1cb91b76814cd0730ff945dd7、SHA-256 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e；history 是对已有 source snapshot 的可追溯补证，不是新的模型候选。

关键提交包括：8ac19cc 修正 deferred KDA gate projection；c4d3770 收敛 CUDA graph side-stream fork/join；c2c3629 为 expert weight loader 建立 O(1) lookup，并同步 ModelSlim fused QKVG/NPU packed mapping；72d5c5b 接受 fused MoE finalize 的 FP32 routing weights；f4c2563 组合 PP prefill、DCP decode、PD disaggregation 和 DSpark；2d0e94e 修正 shared-expert process group identity；8ac39c6 增加 Ascend A5/NPU 分支；cb32dbc 为 ROCm 小 token KDA input projection 增加单 GEMM 和布局测试。

这些 commit 只能升级 K3 的 mutable upstream source evidence，不能升级为 stable wheel、完整权重加载、目标 GPU/ROCm/NPU profiling、双状态恢复、DSpark acceptance、视觉正确性或线上 tool/verifier acceptance。当前模型状态保持：内容专题闭环 + HF/vLLM + SGLang stable/main source evidence。

## 2026-09-20 DeepSeek V4.1-Flash reference implementation 补证

本轮继续使用 Artificial Analysis 的精确 `deepseek-v4-1-flash` 条目作为模型发现锚点；没有从 Hugging Face 仓库另发现模型。2026-09-20 两榜复验快照为：Artificial Analysis 首页 `1,777,695` bytes、SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`；DataCurve `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此不迁移 V4 Pro/V4 Flash 的 Pass@1、成本、输出 token 或 Agent steps。

官方核验继续绑定 HF revision `dba1be0a40aa45a94ad051997016db3960a90277`。固定仓库公开 `inference/model.py`、TileLang `inference/kernel.py`、FP4/FP8 权重转换器、Engram/vision 路径和 `dsh-minimal` 评测补丁；`inference/README.md` 明确其是 readable reference implementation，`generate.py` 是普通自回归入口。`model.py` 中的 DSpark `forward_spec`、候选池/Top-K、压缩 KV、SWA ring 和 mHC/Sinkhorn 代码可作为实现证据，但没有在本轮升级为生产 engine、speculative acceptance 或 GPU 性能结论。

当前状态：**内容专题闭环 + reference implementation 证据补强（AA 单榜）**。已完成固定源码哈希、runtime config、编码 smoke test、Python 静态编译和标准库教学实验；教学实验只拆分 synthetic candidate-pool/Top-K recall 与 E2M1-like 误差，不代表模型实测。完整权重、CUDA/TileLang、生产 kernel 覆盖、DSpark draft/verify/rollback、目标硬件 profiling、线上 acceptance 和独立 benchmark 仍待核验。下一活动锚点继续从两个排行榜的八家重点厂商条目选择，不把官方仓库中的关联 artifact 当作新模型。

## 2026-09-21 DeepSeek V4.1-Flash：deepseek-recipe 协议工具补证

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| DeepSeek V4.1-Flash | Artificial Analysis：[`deepseek-v4-1-flash`](https://artificialanalysis.ai/models/deepseek-v4-1-flash)；DataCurve：无精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行 | [DeepSeek `deepseek-recipe` pinned commit](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea)、[README](https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/8cadfede7063c896b944e7bae05daa3549ae97ea/README.md)、[streaming guide](https://github.com/deepseek-ai/deepseek-recipe/blob/8cadfede7063c896b944e7bae05daa3549ae97ea/docs/streaming.md)、[tokenizer guide](https://github.com/deepseek-ai/deepseek-recipe/blob/8cadfede7063c896b944e7bae05daa3549ae97ea/docs/tokenizer.md) | V4.1 内容专题 + 固定 recipe commit 协议证据补强（AA 单榜）；不把协议库写成推理引擎或新模型 |

固定 commit 为 `8cadfede7063c896b944e7bae05daa3549ae97ea`，源码归档 3,996,119 bytes、SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`；README 为 5,261 bytes、SHA-256 `0cccc69baa118d7689fc3ff2c2e47ab652e05410b777744c43a424f4db5fc0af`。仓库公开 Rust/Python 协议转换、V4/V4.1 prompt/tokenizer、跨 chunk 流式状态机、图像 quota/预处理和 mock server 接线；模型 inference、HTTP transport、工具执行、权限和 verifier 仍由宿主提供。默认图像限制和 SSRF 警告见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-21 Claude Opus 5 当前时点复验

本节只更新已经出现在两个排行榜的 Claude Opus 5；本轮没有从 Anthropic 官方目录、论文或博客另发现模型。三条代理 `10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234` 当前均无法连接 Artificial Analysis、DataCurve 或 Anthropic 页面，因此不新增快照和榜单指标。

Opus 5 继续标为**资料级闭环**。已有的 2026-09-15 Artificial Analysis/DataCurve/Anthropic 快照仍作为历史缓存证据；官方运行时补充项为 `thinking.display: "omitted"`、thinking block/signature 原样回放、`thinking disabled` 与 `xhigh/max` 的能力门槛、refusal/fallback、fallback credit、工具/effort 中途变更、512-token prompt cache、web fetch 宿主边界及 subagent/自验证预算。DataCurve 没有可迁移的其他 Claude 版本结果；参数、架构、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 和精确独立评测仍待核验。

## 2026-09-21 Claude Opus 5 联网恢复后的新鲜快照

三条代理在本轮先后经历短时连接失败，随后均返回 HTTP 200。Artificial Analysis Opus 5 详情三条代理逐字节一致，`3,868,875` bytes、SHA-256 `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`；DataCurve 三条代理逐字节一致，`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；AA 中文首页 7890 快照为 `1,778,568` bytes、SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`。

当前 AA max 的 Intelligence Index 为 `50.7771115797629`、median output speed `60.707552496689 tokens/s`、median TTFC `46.6837206345s`；DataCurve max 仍为 `327/444`、Pass@1 `73.6486%`、Pass@4 `88.4956%`、平均成本 `$11.8376`。Anthropic 模型页 1234 快照为 `461,560` bytes / `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5`，发布页 8098 快照为 `352,773` bytes / `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f`。新鲜快照没有增加八家重点厂商 canonical 模型；Opus 5 仍为资料级闭环，不升级内部架构或独立评测证据。

## 2026-09-21 GLM-5.3 官方博客正文与评测脚注补证

本节只补充已经出现在两个排行榜中的 GLM-5.3，不从博客脚本或官方页面另发现模型。Artificial Analysis 当前榜单快照为 1,776,713 bytes，DataCurve 当前快照为 268,571 bytes；八家重点厂商的 canonical 集合没有变化。官方博客壳、正文 JS、source bundle 分别为 598、30,414、245,425 bytes，哈希已登记在来源索引与研究笔记。

博客新增 Z.ai Code Bench 相对 GLM-5.2 的 50% 发布方提升声明、Max/High effort 的完成率与 token efficiency 对照、可执行长周期环境、reference-free verifier、solver trajectory、reward-shortcut audit，以及 CyberGym/ExploitBench/ExploitGym 的阶段化安全能力。合作代码库统计为 269 个项目、2,436 个漏洞和 1,097 个中高危发现。评测脚注还固定了 DeepSWE、Terminal-Bench 3.0、ALE、CyberGym、ExploitGym 和 ExploitBench 的 harness、temperature/top-p、上下文、turn/timeout、隔离、域名白名单和 verifier 条件。

当前状态保持双榜资料级闭环；这些是 Z.ai 发布方与 benchmark 脚注证据，不能与 DataCurve 或 Artificial Analysis 拼成裸模型分数，也不新增重复 Transformer 章节。完整证据见 glm-5.3-source-notes.md，仍待核验独立技术报告、完整后训练 recipe、5.3 专属 SAO/compaction 实现、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance。

## 2026-09-21 Gemini 3.8 Flash Model Card 与长周期 Agent 复验

本轮仍沿两个排行榜已经发现的 `Gemini 3.8 Flash` 推进，没有把 `Gemini 3.8 Flash Cyber` 作为新的排行榜模型。Artificial Analysis `/zh` 为 `1,776,713` bytes / SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`；high 详情为 `3,862,728` bytes / SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`，当前 Intelligence Index `40.9262321765904`、速度 `328.820180257944` tokens/s、1M context、`$0.75/$3.75` input/output 与 `$0.075` cache-hit。DataCurve 仍为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，精确 high 行 Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36`、平均 `166.31` steps。

DeepMind Model Card 快照为 `156,591` bytes / SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`，明确 3.8 基于 3.7，并把架构、训练数据、软硬件和评测方法指向 3.7 Model Card；HLE-Verified `54.9%` 与 multilingual safety `+5.4pp` 按发布方评测记录。Google 发布博客快照为 `407,603` bytes / SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`，新增 shared foundational intelligence、long-running agentic loops、额外 reasoning steps 与迭代工具调用的公开系统描述。当前状态仍为**资料级闭环**；3.8 独立架构、参数、训练/后训练 recipe、生产 kernel、硬件 profiling、独立复现和线上 acceptance 待核验。

## 2026-09-21 GPT-5.6 Luna 当前时点复验与 OpenAI 访问边界

本节只更新已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `GPT-5.6 Luna`，不从 OpenAI 官方文档路径另发现模型。

| 基础模型/配置 | 榜单当前快照 | 当前状态 |
|---|---|---|
| GPT-5.6 Luna (max) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-6-luna)：`3,861,087` bytes / `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`，Intelligence Index `37.3244239690841`、164.5096 tokens/s、1M、约 `$0.20/$1.20`；DataCurve：`mini_swe_agent_gpt_5_6_luna_max`，301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均 `$0.6056`、约 73.4K output tokens、101.68 steps | 双榜资料级闭环；AA 与 DataCurve 结果分别绑定各自配置/测量和 `mini-swe-agent` harness |

DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。本轮 OpenAI 官方模型索引、Luna、Reasoning 和 Prompt Caching 页面通过 1234 均返回 `403`，7890/8098 超时，直连 DNS 失败；旧官方页面快照仍作为既有资料，不冒充当前新鲜读取。当前不新增 GPT-5.6 专属架构章节，待官方页面恢复后再补版本/行为差异。

## 2026-09-21 GLM-5.3-Flash serving/runtime 复核

本节只更新已经出现在两个排行榜中的 `GLM-5.3-Flash`，不把 Z.ai 文档中的 `GLM-5.3-FlashX` 关联服务入口另列为新模型。

| 基础模型/配置 | 榜单当前快照 | 新增官方实现证据 | 当前状态 |
|---|---|---|---|
| GLM-5.3-Flash (max) | [Artificial Analysis](https://artificialanalysis.ai/models/glm-5-3-flash)：`3,942,546` bytes / `42f880600d3637489a0c48ff27357fe7986e53510ad5ee016c7d6170bd201fae`，release `2026-08-26`、1,048,576 context、Intelligence Index `41.807466113455`、median output speed `95.0131572798129 tokens/s`；[DataCurve](https://deepswe.datacurve.ai/) 精确行 `mini_swe_agent_glm_5_3_flash_max`：284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本 `$0.2409818562`、平均输出 `72829.77` token、约 122.89 steps | [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md)：两类 state pool、MTP、KV/DSA backend pairing、EPD/PD 门禁；[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)：v0.29.0+、native FP8/MTP、hybrid KDA+sparse MLA；[Transformers GLM5-Next](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md)：当前实现不含 MTP layer | 内容专题闭环 + serving/runtime 资料补证；未升级为本机 runtime 或目标硬件验收 |

SGLang 页面当前更新时间为 `2026-09-21T12:21:10.188Z`，快照 `247,030` bytes、SHA-256 `fef983feab9a25311de8cf0b62c539b8450ef9acef9015a288ab6639ecd89019`。页面列出 45 个文本层（MLA/DSA/KDA）、24 层视觉 encoder、288 routed/top-8 和原生 MTP；低延迟使用 MTP `5/1/6`，高吞吐可以关闭 speculative。服务必须同时考虑 paged KV pool 和 KDA state pool，后者可能先限制并发。

页面的硬件组合是约束而非建议清单：Blackwell 为 FP8 KV + TRT-LLM DSA，H100/H200 为 BF16 KV + TileLang DSA，FP8 KV + TileLang DSA 标为无效组合。GB300 throughput、KV capacity、视频 2 FPS/约 240K visual tokens、4x GB300 encoder disaggregation 的 decode gap、PD dummy weights 和 speculative/TP 数值门禁均属于 recipe/发布方结果，不是本机复现。vLLM 页面还存在 FlashInfer `0.6.17+` 与 troubleshooting `0.6.18+` 的内部门槛差异，必须按实际依赖路径锁定版本。

Transformers 不包含 MTP layer，不与 vLLM/SGLang serving 层的 MTP 能力混写。`GLM-5.3-FlashX` 约 200 tokens/s、Flash 1M/128K、thinking enabled 和 Coding Plan 配额是 endpoint/service 字段，不产生排行榜新条目。完整 kernel、MTP acceptance、PD load/accuracy、目标硬件 profiling、线上 tool acceptance 和独立 benchmark 仍待核验。

## 2026-09-21 DeepSeek V3.2 当前 AA 与 Exp README 复核

本节只更新已经出现在 Artificial Analysis 的 `DeepSeek V3.2`，不从 V3.2-Exp README、TileLang、DeepGEMM、FlashMLA、SGLang 或 vLLM 页面另发现模型。

| 模型 | 当前排行榜/页面字段 | 当前状态 |
|---|---|---|
| DeepSeek V3.2 | Artificial Analysis：3,638,730 bytes / SHA-256 `4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3`；Non-reasoning、2025-12-01、685B total/37B active、128K context、Intelligence Index `16.043537719683`、0.28/0.42 per 1M input/output tokens。DataCurve 当前无精确 V3.2 行 | AA 单榜资料级闭环；第三方页面当前没有可用 output-speed/TTFT 字段，不补写该指标 |

9 月 20 日快照的 648B 与本轮 685B 是 AA 目录/provider 数据在不同采集时点的差异，不能当作模型 revision、参数训练变化或架构变化。当前 /zh 1,776,748 bytes / `e73b156ffdc11dd391b48ac9c9f114f31bfab711ba9171a55f66c647cf22c73a` 与 DataCurve 268,571 bytes / `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` 未带来八家重点厂商的新 canonical 模型，也未产生精确 `mini_swe_agent_deepseek_v3_2_*` 行。

V3.2-Exp README 快照为 6,899 bytes / `dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74`。本轮把它新增的 RoPE 修复、TileLang/DeepGEMM/FlashMLA 三层实现、SGLang dsv32 镜像和 tp=8, dp=8, enable-dp-attention 作为实现/serving 资料补证；vLLM recipe URL 当前 404 只记为访问边界，不升级为“无实现”。

研究笔记见 deepseek-v3.2-source-notes.md；正式内容继续复用第二十一册第 19 章 19.28--19.35，不新增重复 Transformer 章节。完整最终 production kernel、召回曲线、硬件 profiling、线上 acceptance、独立 benchmark 和完整 RL recipe 仍待核验。

## 2026-09-22 DeepSeek V3.2 候选身份归并

本轮继续从 Artificial Analysis 详情页核对 V3.2 家族，没有从官方仓库、论文或部署文档新增模型。以下条目都应视为同一 V3.2 家族的历史 revision、配置或专项 checkpoint，而不是新的 canonical 锚点：

| 条目 | Artificial Analysis 当前身份 | 归并结论 |
|---|---|---|
| `deepseek-v3-2-reasoning-0925` | `V3.2 Exp (Reasoning)`，2025-09-29，deprecated，指向 `deepseek-v3-2-reasoning` | V3.2-Exp 历史 reasoning revision |
| `deepseek-v3-2-0925` | `V3.2 Exp (Non-reasoning)`，deprecated，指向 `deepseek-v3-2` | V3.2-Exp 历史 non-reasoning revision |
| `deepseek-v3-2-reasoning` | 2025-12-01 V3.2 配置，deprecated，指向 V4 Pro 0424 | V3.2 reasoning canonical 的历史目录对象 |
| `deepseek-v3-2` | 2025-12-01 V3.2 配置，deprecated，指向 V4 Pro 0424 | V3.2 non-reasoning canonical 的历史目录对象 |
| `deepseek-v3-2-speciale` | V3.2 专项推理 checkpoint，deprecated，指向 V4 Pro 0424 | 同家族专项 checkpoint，不能当作新基础模型 |

DataCurve 当前只有 `deepseek_v4_flash` 和 `deepseek_v4_pro`，没有精确的 `mini_swe_agent_deepseek_v3_2_*` 行。因此不新增 DeepSeek 模型，不迁移 V3.1、V3.2、V4 或 Speciale 的 Agent 分数；V3.2 的状态仍为 AA 单榜资料级闭环。deprecated/redirect 字段是 Artificial Analysis 当前目录状态，不等于官方 API 关闭时间或权重训练变化。

## 2026-09-22 Grok 4.7 当前活动锚点

本节只升级已经出现在 Artificial Analysis 的 `Grok 4.7 (xhigh)`；DataCurve 当前没有精确 `mini_swe_agent_grok_4_7_*` 行。xAI 官方资料用于核验和扩展周边技术，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Grok 4.7 (`xhigh`) | [Artificial Analysis](https://artificialanalysis.ai/models/grok-4-7)，release date `2026-09-21`、Intelligence Index `46.4465506302286`、500K context；DataCurve 无精确行 | [模型页](https://docs.x.ai/developers/models/grok-4.7)、[发布页](https://x.ai/news/grok-4-7)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)、[Function Calling](https://docs.x.ai/developers/tools/function-calling)、[Structured Outputs](https://docs.x.ai/developers/features/structured-outputs)、[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp) | AA 单榜内容专题闭环；更大 base、更长 RL、长任务自验证、每次 Responses encrypted reasoning、服务端工具加密输出、reasoning summary 流、opaque compaction、function/structured/MCP 协议已核验；参数、架构、完整训练 recipe、生产 kernel、线上 acceptance 和独立复现待核验 |

AA 2026-09-22 详情快照为 `3,882,692` bytes、SHA-256 `e62290c7cc0d937afb8b8e4f08328c9935e6b40a4db9af3029579052b57baa5c`；中文首页为 `1,785,528` bytes、SHA-256 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`；DataCurve 为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。三条代理的榜单页面均逐字节一致。

xAI 官方模型页旧快照为 `376,912` bytes / `c3ddd6b44b97fb4b527096ca69e4d9eacdca99e0e4e44427c9da5b81a615181e`，本轮刷新为 `376,913` bytes / `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5`；发布页本轮为 `288,007` bytes / `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`；Context Compaction 本轮为 `521,486` bytes / `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`。去除 HTML 包装后正文没有模型合同变化。官方公开 `grok-4.7`、500K、`low/medium/high/xhigh`、每次 Responses reasoning encrypted content、服务端工具加密输出、reasoning summary 流、原样回放的 compaction item、function calling、structured outputs 和 Remote MCP `allowed_tools`；这些是 API/运行时证据，不是内部网络结构证明。

Reasoning 页还明确：`store` 决定 response 是否可由 `previous_response_id` 复用，不能把它与 encrypted field 的默认返回或服务端 thinking trace rehydration 混为一谈；MCP 当前只支持 Streaming HTTP/SSE，xAI SDK 的字段名与 OpenAI Responses API 不完全相同，且 `require_approval`/`connector_id` 当前不可用。

xAI 发布页给出 CursorBench 4.0 `46.3%`、DeepSWE v1.1 `71.0%`、EEBench `64.0%`、AA Briefcase v1.1 `1657`、Terminal-Bench 4.0 `38.0%`、Harvey `19.6%`、HealthBench Professional `56.7%` 和 GDPval `1695`，以及 LatchBio `62.4%`、HackerBench v0.3 危险 prompt 放行率 `3.3%`。这些数字必须带发布方 benchmark、effort、harness、任务集和 verifier 标签，不能与 AA/DataCurve 合并成裸模型分数。

arXiv 精确标题检索未找到 Grok 4.7 论文（快照 SHA-256 `8f8c4bb0f6c8a0346e28a56864a4aa4d630bdde9c536ee1320be29f84e6e9da6`）。因此不补写参数规模、MoE/稠密架构、完整训练 recipe 或生产 kernel。完整证据见 [`grok-4.7-source-notes.md`](grok-4.7-source-notes.md)。

## 2026-09-23 Grok 4.7 当前时点复验

本轮使用 `10.24.27.134:7890` 刷新两个唯一排行榜。Artificial Analysis 中文首页为 `1,783,572` bytes / `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2`，DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；八家重点厂商没有新的 canonical 模型。

Grok 4.7 详情为 `3,992,680` bytes / `9b24aa029ee4023aa2918495910a282a96f8bd9bba42848921f9496684e3e6bb`，关键字段仍是 release `2026-09-21`、Intelligence Index `46.4465506302286`、500K context、proprietary 和 parameters null。DataCurve 没有精确 `mini_swe_agent_grok_4_7_*` 行，不迁移 Grok 4.6 结果。xAI 三页当前快照和本地回放 toy 的证据边界见 [`grok-4.7-source-notes.md`](grok-4.7-source-notes.md)。状态为 **AA 单榜内容专题闭环 + 当前复验 + local protocol toy**，不升级为架构/训练/硬件闭环。

## 2026-09-22 Qwen3-Omni 30B A3B 官方核验增补

本节只升级已经由 Artificial Analysis 发现的 `Qwen3 Omni 30B A3B`；Qwen GitHub、Hugging Face、Qwen 博客、技术报告和阿里云文档用于核验和扩展周边技术，不作为新的模型发现入口。

| 基础模型/变体 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Qwen3-Omni-30B-A3B | Artificial Analysis：[`qwen3-omni-30b-a3b-instruct`](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-instruct) 与 reasoning 页面；DataCurve 当前没有精确 `mini_swe_agent_qwen3_omni_*` 行 | [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)、[Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)、[Thinking 配置](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/raw/main/config.json)、[技术报告](https://arxiv.org/abs/2509.17765)、[Qwen 博客](https://qwen.ai/blog?id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4&from=research.latest-advancements-list)、[阿里云 Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni) | AA 单榜内容专题闭环；Thinker-Talker、AuT、TM-RoPE、Talker 多码本/MTP、Code2Wav、异步 chunked prefill、训练/后训练主线和 Instruct/Thinking/Captioner 边界已核验；完整参数账本、生产 kernel、真实流式 profiling、端到端音频 acceptance、完整 recipe 和独立评测仍待核验 |

Artificial Analysis instruct 详情三代理均 HTTP 200、`3,833,501` bytes、SHA-256 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`；reasoning 详情快照为 `3,845,331` bytes、SHA-256 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。instruct 页面第三方字段约为 Intelligence Index `6.0061`、`94.59 tokens/s`、`66K` context、`35.3B` total、`3B` active、输入/输出约 `$0.25/$0.97` 每百万 token；这些字段绑定页面/provider 和 2026-09-22 采集时间，不写成官方参数或本地实测。

官方技术报告把 Omni 的主线拆成 Thinker（多模态理解/文本推理）与 Talker（流式语音生成）。AuT 约 2,000 万小时监督音频、8 倍 Conv2D、12.5 Hz、动态 1-8 秒 attention window、约 0.6B 参数；视觉 encoder 来自 Qwen3-VL、初始化于 SigLIP2-So400m、约 543M 参数；TM-RoPE 使用 temporal/height/width 三个坐标和 `24/20/20` 的 rotary angle 分配，音频/视频时间对齐约 80 ms。Talker 使用首码本自回归 + MTP 预测 residual codebooks，多码本 code rate 为 12.5 Hz，Code2Wav 使用轻量 causal ConvNet。报告给出的 audio/video first packet `234/547 ms` 是报告口径的理论/实验数字，不是本机或线上 p99。

Instruct、Thinking 和 Captioner 记录为同一 Qwen3-Omni 家族内的不同 artifact，不按三个独立基础模型计数。官方 README 当前说明 `transformers>=5.2.0`、`qwen-omni-utils`、FlashAttention 2，`disable_talker()` 可节省约 10GB 显存，vLLM 主要支持 Thinker，Instruct 音频输出仍在实现推进中。完整证据、临时抓取路径和待核验项见 [`qwen3-omni-source-notes.md`](qwen3-omni-source-notes.md)；正式专题见第二十一册第 91 章 [`Qwen3-Omni：Thinker-Talker、AuT 与流式多模态`](../../book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)。

## 2026-09-22 K2 Horizon 3.7B 官方核验增补

本节只升级已经由 Artificial Analysis 发现的 `K2 Horizon 3.7B`；DataCurve 当前没有精确 K2 Horizon 3.7B 行，不能迁移 K2 Horizon 36B、K2.7 Code、K3 或其他模型的 Agent 评测。

| 基础模型 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| K2 Horizon 3.7B | Artificial Analysis：[`k2-horizon-3-7b`](https://artificialanalysis.ai/models/k2-horizon-3-7b)，第三方 `releaseDate=2026-09-03`、reasoning、open weights、3.7B、524,288 context、Apache 2.0、AA Index `15.6103951330976`；DataCurve 无精确行 | [IFM 模型卡](https://huggingface.co/IFM/K2-Horizon-3.7B)、固定 main revision `6360f705b2e57d542959e6a2e67ebeb95dae0373`、[vLLM recipe](https://recipes.vllm.ai/IFM/K2-Horizon-3.7B)、[SGLang PR #37654](https://github.com/sgl-project/sglang/pull/37654) | **AA 单榜资料级闭环**；dense 架构、BF16 artifact、分阶段长上下文训练、RL expert merge、migration 和 serving 证据已核验；精确 Agent 结果、完整 recipe、生产 kernel、硬件 profiling 和独立复现待核验 |

当前配置为 `K2HorizonForCausalLM`，36 层、2560 hidden、32Q/8KV GQA、head dim 128、vocab 250,624、524,288 position、default RoPE (`theta=10,000,000`)，`num_experts=0`、`mova_num_experts=0`，所有层 dense。权重 index 为 36 shards/327 tensors/`10,116,510,720` bytes，BF16 存储约对应 5.06B 参数。旧 `APPENDIX.md` 仍写 `XllmForCausalLM`、FP32 和 `3.78B core / 5.06B including embeddings`，应按旧 revision/文档残留记录，不能覆盖当前 config 和 migration manifest。

三条代理的 AA 详情页逐字节一致：`3,756,408` bytes，SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`；DataCurve 快照为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 行。完整证据见 [`k2-horizon-3.7b-source-notes.md`](k2-horizon-3.7b-source-notes.md)；3.7B 与 36B MoVA 的对照、训练 checkpoint、migration 和 serving revision 已并入第二十一册第 82 章。

## 2026-09-22 Qwen3-VL-235B-A22B 官方核验增补

本节只升级已经由 Artificial Analysis 发现的 `Qwen3-VL-235B-A22B`；Qwen 官方 GitHub、HF 模型卡/config、技术报告和 Transformers 上游代码用于核验和扩展该锚点，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Qwen3-VL-235B-A22B | Artificial Analysis：[`instruct`](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct) 与 [`reasoning`](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-reasoning)，页面约 `235B total / 22B active`、`262,144` context；DataCurve 无精确 `mini_swe_agent_qwen3_vl_*` 行 | [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)、[Technical Report](https://arxiv.org/abs/2511.21631)、[Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)、[Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)、[Transformers implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_vl) | **AA 单榜内容专题闭环**；三模块结构、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 curriculum、SAPO、Thinking with Images 和 tool-call reward 已核验；完整权重、生产 kernel、目标硬件 profiling、端到端多模态 serving、GUI/tool acceptance 和独立 benchmark 待核验；instruct/reasoning/Thinking 归并为同一基础模型的配置或 artifact |

AA 中文首页三代理逐字节一致：`1,798,627` bytes / SHA-256 `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`；instruct 详情 `3,832,139` bytes / `a58d3922d8e872c76684314e9de21eee78e431ee0d527d10c65cadc9f7755efa`；reasoning 详情 `3,849,165` bytes / `28f97633dfbeb145e696b9f87a01ba891755d80bfaaf7e7a1db4d8d7ecaa59b3`。AA 的速度、价格、指数和 active 参数字段继续按第三方目录/provider 口径记录。

当前 HF revision 为 Instruct `710c13861be6c466e66de3f484069440b8f31389`、Thinking `6664affde68449468deb7527186455c7450c13c0`。config 固定 `Qwen3VLMoeForConditionalGeneration`、94 层、64Q/4KV、128 experts/top-8、262K position、RoPE theta `5,000,000`、视觉 depth 27、patch 16、temporal patch 2 和 DeepStack `[8,16,24]`。论文公开 SigLIP2 vision encoder、两层 MLP merger、Interleaved-MRoPE、Video Timestamp、67B/1T/1T/100B 的 S0-S3 curriculum、square-root normalized loss、SAPO/General RL，以及 visual-agent/tool-integrated RL；这些事实不能扩写成完整训练 recipe 或独立 benchmark。

完整证据见 [`qwen3-vl-source-notes.md`](qwen3-vl-source-notes.md)；正式专题见第二十一册第 92 章 [`Qwen3-VL：Interleaved-MRoPE、DeepStack 与视频时间戳`](../../book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)。

## 2026-09-22 Claude Opus 5 当前时点复验

本节只复验已经由两个排行榜确认的 `Claude Opus 5`，不把当前页面测量漂移或 Anthropic 官方资料中的 fallback 目标另列为新模型。

| 基础模型/配置 | 榜单当前快照 | 官方核验 | 当前状态 |
|---|---|---|---|
| Claude Opus 5 (`max`) | [Artificial Analysis](https://artificialanalysis.ai/models/claude-opus-5)：`3,869,351` bytes / `c18260ab4ff331d5bd3305691db2d4b6051dc2ebe642aa1458c5b8fa2c367643`，release `2026-07-24`、Intelligence Index `50.7771115797629`、`56.4471785104486 tokens/s`、TTFC `49.2490756305s`、1M context、约 `$5.8584`/task；[DataCurve](https://deepswe.datacurve.ai/) 精确 `mini_swe_agent_claude_opus_5_max` 为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`、平均成本约 `$11.8376`、约 117,566 output tokens、约 99.04 steps | [Anthropic Opus 5 模型页](https://platform.claude.com/docs/en/models/opus-5/overview)、[发布页](https://www.anthropic.com/research/claude-opus-5)、[System Card](https://www.anthropic.com/claude-opus-5-system-card)：1M/128K/300K、adaptive/high、effort、thinking block 回放、工具/effort 中途变更、fallback 和缓存边界 | 资料级闭环；9 月 21 日速度/TTFC 保留为历史 provider 测量，不解释为模型 revision；参数、架构、完整训练 recipe、独立报告和生产 acceptance 仍待核验 |

DataCurve 当前复验页面为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。Anthropic 发布页当前快照为 `352,846` bytes / `7bb18f8e14e20fe2651e4a8308947f53a9541b0dcce6b7549e2b2c244202dce5`，System Card 为 `16,281,258` bytes / `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。这些页面证据补强 API/评测边界，不升级为内部网络结构或训练机制证据。

## 2026-09-22 Gemini 3.5 Flash-Lite 当前时点复验

本节只复验已经由 Artificial Analysis 确认的 `Gemini 3.5 Flash-Lite`；Google 官方页面用于扩展该锚点，不作为新的模型发现入口。

| 基础模型/配置 | 榜单当前快照 | 官方核验 | 当前状态 |
|---|---|---|---|
| Gemini 3.5 Flash-Lite (`high`) | [Artificial Analysis](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)：`3,858,652` bytes / `bbe13cb52c85772080a105b1ec98c42bd71e7194dd67abbc26cb52f9ba1115d`，release `2026-07-21`、Intelligence Index `22.1685424839812`、`386.373295824977 tokens/s`、TTFC `11.2053542005s`、1M context、约 `$0.1235`/task；[DataCurve](https://deepswe.datacurve.ai/) 当前快照 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，没有精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行 | [Gemini 3.5 Flash-Lite API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)：1,048,576/65,536、`minimal/low/medium/high`、文本/图像/视频/音频/PDF 输入、agentic video 和 `processing_call/result` | AA 单榜资料级闭环；Model Card 明确基于 Gemini 3.1 Flash-Lite，不能把前代架构/训练/软硬件资料写成 3.5 独有发明；不迁移相邻 Flash 变体 Agent 分数，不新增独立 Transformer 章节 |

Google 官方新鲜快照为 API 模型页 `106,715` bytes / `052846d394f282870f2334d4c079947b45cb3f83faa39de63eb338f78ca5a4a9`、Thinking `226,283` bytes / `ab9d354068333aedd8199c5afafd23d36e6b0c6677df531de4d8804b48833951`、Video understanding `262,508` bytes / `7420a3b0b4baf5a6667371167c86fe8f178a339c9eb6bd1e5108daa789b4125e`。`minimal` 是请求级思考配置，不等于新 checkpoint；1M 输入上限也不等于均匀有效记忆。

## 2026-09-22 GLM-5.1 当前时点复验

本节只复验已经由 Artificial Analysis 确认的 `GLM-5.1`；Z.ai 官方资料用于扩展该锚点，不作为新的模型发现入口。Reasoning 与 non-reasoning 页面归并为同一基础模型的运行配置，不新增两个 checkpoint。

| 模型/配置 | 榜单证据 | 官方资料与技术主线 | 当前状态 |
|---|---|---|---|
| GLM-5.1 (`reasoning` / `non-reasoning`) | [Artificial Analysis](https://artificialanalysis.ai/models/glm-5-1)：2026-09-22 当前详情 `3,971,543` bytes / `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`，release `April 2026`、200K context、Intelligence Index `26.0585912980095`、`37.1922485381327 tokens/s`、cost per Intelligence Index task `0.9217822401466147`；[DataCurve](https://deepswe.datacurve.ai/) 当前 `268,571` bytes / `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，无精确 `mini_swe_agent_glm_5_1_*` 行 | [Z.ai 模型文档](https://docs.z.ai/guides/llm/glm-5.1)、[官方博客](https://z.ai/blog/glm-5.1)、[HF README/config](https://huggingface.co/zai-org/GLM-5.1)、[Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md)；long-horizon Agent、multi-turn SFT/RL/process-quality evaluation、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、202752 position、thinking/tool/cache 协议 | AA 单榜资料级闭环；博客 benchmark/harness 和 AA provider 字段按来源分账，不迁移相邻 GLM 版本 Agent 成绩；完整 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、线上 acceptance、独立报告和复现待核验 |

2026-09-22 Z.ai 完整 GLM-5.1 页面快照为 `509,732` bytes / `6eda05875cc6f63769e0365c1cc7ddad4b499ed7292610e25d24a1daaf7e6ab8`；稳定 Markdown 提取为 `19,347` bytes / `69958d7d95452d3853903524612a0c6a83f368c79417a6889b9c531f45ab3d01`。当前页面没有新增独立 GLM-5.1 技术报告；模型卡链接 GLM-5 报告，不能将 GLM-5 的 28.5T、slime 或具体训练叙述自动迁移给 GLM-5.1。9 月 20/21 日 AA 的历史速度/价格与当前测量分开保存，不解释为模型升级。

## 2026-09-22 GPT-5.6 Luna 当前时点复验与官方运行时补证

本节只更新已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `GPT-5.6 Luna`，不从 OpenAI 官方文档路径另发现模型；`Sol/Terra/Luna` 仍按同一 GPT-5.6 家族的服务档位归并。

| 基础模型/配置 | 榜单当前快照 | 官方页面与新增知识 | 当前状态 |
|---|---|---|---|
| GPT-5.6 Luna (`max`) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-5-6-luna)：`3,861,318` bytes / `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；release `2026-07-09`、Intelligence Index `37.3244239690841`、`158.728370482714 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context、约 `$0.20/$1.20`。DataCurve：`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，精确 `mini_swe_agent_gpt_5_6_luna_max` 为 301/448、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、约 `$0.6056`、约 73.4K output tokens、约 101.68 steps | [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)；三层 runtime ownership、persisted reasoning、hosted/client tool search 和 cache/compaction 交互 | **双榜资料级闭环**；DataCurve 结果绑定 `mini-swe-agent + tools + task environment + verifier`，不迁移 Sol/其他 GPT 版本成绩 |

本轮官方 Markdown 页面均由 `10.24.27.134:7890` 返回 HTTP 200：Luna `3,744` bytes / `1f425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`，Reasoning `70,253` / `91604df954335250d16e33f3b07ebbfa3e6e2b7f821f0722ad7a69b9d586719a`，Agents `5,432` / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`，Tools `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`，Tool Search `39,288` / `9d6c3855a4cb722a98fb364618a852fb436e9822a36fe9b54650f75bbc8d1fd8`，Prompt Caching `47,097` / `c70d858eecd09681cdc671a1a037e7d51916a793eb85c9dc08be240d1cb9b2d1`，Compaction `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`。

Agents API、Agents SDK 与 Responses API 是不同的运行时资源/所有权边界；hosted tool search 由服务端返回加载工具，client-executed tool search 要求应用以同一 `call_id` 回传工具集合，二者都把工具追加到上下文尾部以尽量保留缓存前缀。`context_management` compaction 会替换旧上下文，可能从变化位置起打断 cache prefix；这些 API/runtime 字段不能反推出 GPT-5.6 的参数、MoE/稠密结构或训练 recipe。

## 2026-09-22 Qwen3.7 Plus 官方托管合同与交互式混合 Agent

本节只升级已经由 Artificial Analysis 确认的 `Qwen3.7 Plus`；Alibaba Cloud Model Studio 资料用于核验该榜单锚点，不作为新的模型发现入口。

| 基础模型/配置 | 榜单当前快照 | 官方核验 | 当前状态 |
|---|---|---|---|
| Qwen3.7 Plus | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-7-plus)：`3,859,009` bytes / `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040`，release June 2026、Intelligence Index `25.1622215821984`、`68.5428061089526 tokens/s`、cost per Intelligence Index task `0.32527343198119174`、约 1M context、约 `$0.40/$1.60` input/output；[DataCurve](https://deepswe.datacurve.ai/) 当前 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_qwen3_7_plus_*` 行 | [Alibaba Cloud Qwen3.7 Plus 文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)、[推荐模型页](https://www.alibabacloud.com/help/en/model-studio/models)、[OpenAI-compatible API](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)：qwen3.7-plus 等价于 `qwen3.7-plus-2026-05-26` snapshot；text/image/video input、text output；Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching 按 region/scope 分层；Fine-tuning unsupported | **AA 单榜资料级闭环**；无公开参数、权重、专属架构/训练报告，不迁移其他 Qwen 的 Agent 结果 |

官方资料把 Qwen3.7 Plus 定位为多模态交互式混合 Agent，覆盖读屏、GUI 交互、视觉参考生成代码和移动端端到端导航。上下文合同为 1,000,000 window、991,808 max input、131,072 max output；thinking mode max input `983,616`，max chain-of-thought length `262,144`。这些是 provider API 字段，不等于内部架构或可读 CoT。区域 capability matrix、价格分档、GUI executor 责任链和证据边界见 [`qwen3.7-plus-source-notes.md`](qwen3.7-plus-source-notes.md)。

## 2026-09-28 Qwen3.7 Max：跨 harness RL 与长程 Agent 实验

本节只扩展 Artificial Analysis 已列出的 Qwen3.7 Max；Qwen 官方博客和 Alibaba Cloud 文档是锚点核验/扩展来源，不是新的模型发现入口。

| 基础模型/配置 | 榜单当前快照 | 官方核验 | 当前状态 |
|---|---|---|---|
| Qwen3.7 Max | [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-7-max)：`3,834,482` bytes / `829a322408dc044f12eefaa769eab0001904ceecb33d891abc801061fced7522`，release 2026-05-19；Index `29`、约 207 tokens/s、约 1M context、约 `$1.15` 平均任务成本、约 `$2.50/$7.50` input/output 均为 AA/provider 字段；[DataCurve](https://deepswe.datacurve.ai/) `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_qwen3_7_max_*` 行 | [Qwen 官方博客](https://qwen.ai/blog?id=qwen3.7) 披露环境扩展、Task/Harness/Verifier rollout 解耦、跨配置 RL、M890 kernel agent 与 reward-hacking monitor；[Model Studio 文档](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-max) 列出版本化 alias 与 API limits | **AA 单榜内容专题闭环**；发布方报告，无独立复现/精确 DataCurve 行；未披露参数/架构/完整训练 recipe |

Model Studio 的 `qwen3.7-max` alias 对应 `qwen3.7-max-2026-05-20`，为纯文本接口；`2026-06-08` 才加 image/video 输入，不能回写 alias。官方博客报告 `Task × Harness × Verifier` 正交组合及跨 harness/verifier RL；M890 上 35h kernel 任务的 `10.0x` 是相对 SGLang Triton 的发布方几何平均结果，与 H100 KernelBench L3 `1.98x/96%` 使用不同设备、baseline 和统计量。奖励作弊监控报告的 13 条新规则、1,618 个案例缺少 precision/recall 与标注分母，不推断 detector 准确率。完整证据和面试边界见 [`qwen3.7-max-source-notes.md`](qwen3.7-max-source-notes.md)。

## 2026-09-22 DeepSeek V4.1-Flash 当前时点复验

本节只复验已经由 Artificial Analysis 确认的 `DeepSeek V4.1-Flash`，不从 HF 文件、GitHub recipe 或官方目录另发现模型。三条代理取得一致的 AA 中文首页、DataCurve DeepSWE、DeepSeek V4.1 发布页和 AA 详情页；本轮没有新的重点厂商 canonical 模型。

| 基础模型/配置 | 榜单当前快照 | 官方核验 | 当前状态 |
|---|---|---|---|
| DeepSeek V4.1-Flash | [Artificial Analysis](https://artificialanalysis.ai/models/deepseek-v4-1-flash)：`3,948,908` bytes / `114cc90d1cb8125174d9464141cbfd0faeac76f52b132f78630c0375c9fcf8fe`，canonical slug `deepseek-v4-1-flash`、Intelligence Index `39.456167472527`、约 1M context、约 `$0.30/$1.20`；[DataCurve](https://deepswe.datacurve.ai/) 当前 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行 | [DeepSeek V4.1 发布页](https://api-docs.deepseek.com/news/news260910)、[固定 Hugging Face revision](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277)、[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)、[`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea)：CED、CSA2/Hierarchical Sparse Indexer、SWA Bounded Replay、FP4 KV、Engram、Single-Pass mHC、DSpark、EPD 和协议状态机 | **内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**；不迁移 V4 Pro/V4 Flash 的 Agent 结果，不把 reference 代码写成生产验收 |

HF 当前 revision `dba1be0a40aa45a94ad051997016db3960a90277`、`lastModified=2026-09-10T08:18:10Z` 和 48 个 safetensors 分片未变化；本轮没有观察到新模型或新权重 revision。完整权重加载、production kernel、candidate/index Top-K recall、真实 FP4 误差、DSpark draft/verify/rollback、EPD 调度、目标硬件 profiling、tool acceptance 和独立 benchmark 仍待核验。完整证据见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-22 DeepSeek V4.1-Flash vLLM upstream runtime 补证

本节继续沿 Artificial Analysis 的 `deepseek-v4-1-flash` 锚点补充 serving 实现证据，不把 vLLM 仓库中的类名当作新的排行榜模型。vLLM `main` registry 快照为 `64,391` bytes / SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，明确登记 `DeepseekV41ForCausalLM` 与 `DSparkV41DraftModel`；专用包位于 `vllm/models/deepseek_v41/`，不是 `model_executor/models/deepseek_v41.py`。

固定的 `quant_config.py`、NVIDIA/ROCm `vl_model.py` 和 `dspark.py` 证实：V4.1 runtime 同时保留 FP4/FP8 expert dtype 分支；视觉 wrapper 通过 `inputs_embeds` 注入 ViT/aligner embedding、保留 raw `input_ids` 供 `bias_vl` 路由、支持 encoder CUDA graph/ViT data parallel，并显式跳过 `mtp.*`，所以当前 vision wrapper 不支持 MTP/DSpark draft heads；DSpark 则使用 3 个预测层、目标层 ids、`[max_num_batched_tokens, index_topk]` Top-K buffer、目标 checkpoint 的 `mtp.{0,1,2}.*` 权重、Markov/confidence head、逐位置 sigmoid confidence、共享 embedding/lm head 和 SWA/FP4/FP8 cache 插入路径。

对照 vLLM `v0.29.0` stable registry（`63,102` bytes / SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`）：该快照有通用 V4/DSpark 入口，但没有本轮两个 V4.1 专用类名。因此当前状态升级为 **内容专题 + HF reference implementation + vLLM upstream main runtime evidence + recipe protocol evidence（AA 单榜）**，仍不等于 stable wheel 已支持、完整权重已加载、GPU/ROCm 已验收或 speculative acceptance/生产 SLO 已证明。完整文件哈希和待核验项见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-22 DeepSeek V4.1-Flash：SGLang main/stable runtime 对照

本节只沿 Artificial Analysis 已确认的 `deepseek-v4-1-flash` 补 serving 实现证据，不从 SGLang 仓库另发现模型。SGLang `main` 的 `deepseek_v4.py`、`deepseek_v4_dspark.py`、`deepseek_v41_vit.py` 和 `deepseek_v4_nextn.py` 已通过 GitHub Contents API 固定，分别为 `241,421` / `4164c354...71ded`、`47,640` / `61dc79f...d7b61`、`5,126` / `29f4d988...c94eb`、`9,459` / `a3ca101d...5d90b`；Git blob 为 `a3b8b610...`、`baebc2de...`、`d5777a4f...`、`6694abb6...`。main 代码支持 V4.1 vision 的 TP/EP/DP（不支持 CP/PP/MoE A2A）、2D-RoPE ViT/Aligner、DSV4 sparse indexer/unified KV/FP8 路径，以及 DSpark 的 Markov/confidence head、`mtp.*` 映射和把 draft stage 的 `vision_n_layers` 设为 `0`。

SGLang `v0.5.20` 于 `2026-09-18T22:41:33Z` 发布，tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。stable 的通用 V4 文件为 `170,193` / `252f6176...0caa4`，DSpark 文件为 `40,396` / `59ac079e...fd04`；`deepseek_v41_vit.py` 不存在，相关源码中 `deepseek_v41` 和 `dsv41` 出现次数均为 0。因此当前状态可升级为 **SGLang upstream main runtime implementation evidence**，但不能升级为 v0.5.20 stable V4.1 serving 支持；完整权重、目标硬件、视觉 draft/target verify、FP4/FP8 质量、DSpark acceptance/rollback、EPD 调度和生产 SLO 仍待核验。完整哈希、提交和来源边界见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-23 DeepSeek V4.1-Flash vLLM `v0.30.0` stable release

本节只更新已经由 Artificial Analysis 确认的 `DeepSeek V4.1-Flash`，不从 vLLM release、PyPI 或 SGLang 文件另发现模型。

| 基础模型/配置 | stable release 证据 | 当前状态 |
|---|---|---|
| DeepSeek V4.1-Flash | [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)，`2026-09-22T05:20:54Z`，tag `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；release API `65,334` bytes / `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`；stable registry `64,420` bytes / `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`，包含 `DeepseekV41ForCausalLM`/`DSparkV41DraftModel` | **内容专题 + HF reference implementation + vLLM v0.30.0 stable release/source evidence + SGLang main + recipe protocol evidence（AA 单榜）** |

v0.30.0 package 的 V4.1 NVIDIA/ROCm `vl_model.py`、`dspark.py`、`quant_config.py` 和 `__init__.py` 已固定，release notes 还记录 FlashMLA V4.1 MXFP8 whole-KV、DeepGEMM Mega-mHC、mHC folding、Triton metadata fusion、Engram async prefetch/DP sharding、DSpark state folding/EPLB isolation、strict tool parameters 的 XGrammar 和 Vision-Exp image sentinel 等 runtime 变化。PyPI metadata 为 `13,218` bytes / `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`；x86_64/aarch64 wheel 与 sdist 的大小、SHA-256 已在 [`source-index.md`](source-index.md) 和 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md) 固定。

stable release/source entry 只证明可发布 artifact 和 registry surface，不证明完整权重、安装依赖、GPU/ROCm/NPU 运行、真实 FP4 质量、candidate/index Top-K recall、DSpark acceptance/rollback、EPD、tool acceptance、目标硬件 profiling、独立 benchmark 或生产 SLO。vLLM `v0.29.0` 仍作为历史负证据；SGLang stable 仍为 `v0.5.20`，本轮未证明其 V4.1 专用 vision/runtime 已进入 stable。

## 2026-09-22 GLM-5.3 标准版：DSA runtime source evidence

本节只升级已经由 Artificial Analysis 与 DataCurve 确认的标准 `GLM-5.3`；`GLM-5.3-Flash` 的 linear/KDA/视觉路径保持独立，不作为标准版的实现证据。

| 基础模型/配置 | 榜单发现 | 官方/上游核验 | 当前状态 |
|---|---|---|---|
| GLM-5.3 (standard DSA, max) | Artificial Analysis：[`glm-5-3`](https://artificialanalysis.ai/models/glm-5-3)，release `2026-08-18`；DataCurve：`mini_swe_agent_glm_5_3_max`，451 attempts/311 passed，Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均约 124.47 steps | [HF fixed revision](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42)、[Transformers main](https://github.com/huggingface/transformers/tree/0bc252863a4e5c0e709893664cc57369ebcd7353)、[vLLM main](https://github.com/vllm-project/vllm/tree/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa)、[vLLM v0.29.0](https://github.com/vllm-project/vllm/tree/v0.29.0)、[SGLang main](https://github.com/sgl-project/sglang/tree/861b11f087af2822cb545ea1895059a721831014)、[SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/v0.5.20) | **双榜资料级闭环 + stable/main runtime source evidence**；标准 `glm_moe_dsa`、Full/Shared indexer、MLA/indexer cache 和 DeepSeek-V3.2 共用 runtime 路径已固定；完整权重、召回、硬件和生产验收待核验 |

固定 HF config 为 `GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed experts、top-8、1 shared expert、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_share_for_mtp_iteration=true` 和 1,048,576 positions。实现源码进一步表明，21 个 indexer `full` 层计算候选，57 个 `shared` 层复用相邻 full 层的 top-k；`indexer_types` 是实现配置，不是完整参数或训练账本。

Transformers main 明确实现 interleaved indexer RoPE、Full/Shared top-k 状态和 sparse attention mask。vLLM main 与 v0.29.0 registry 都把该类路由到 `deepseek_v32`；SGLang stable/main 都包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。main 相比 stable 的 PCP/DCP、HiSparse、logical top-k 或其他优化，只能标为 upstream source evidence，不能写成发布版普遍可用能力。

因此标准 GLM-5.3 的证据链应写成：`榜单 canonical identity -> HF config/revision -> Transformers forward -> stable/main runtime entry -> full-weight load -> index/evidence recall -> cache/MTP recovery -> target profile -> tool/verifier/SLO`。任何一层都不能越级证明后一层；尤其不能把 Flash 的 `RadixLinearAttention`、KDA state、视觉模块、EPD 或 `glm5_next.py` 迁移到标准版。

## 2026-09-23 Claude Opus 5.5：token efficiency 与安全路由

本节只升级已经出现在 Artificial Analysis 的 `Claude Opus 5.5`，不从 Anthropic 发布页、System Card 或开发者目录另发现 Sonnet 5.5/Haiku 5.5 等模型。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| Claude Opus 5.5 (`max with fallback`) | [Artificial Analysis](https://artificialanalysis.ai/models/claude-opus-5-5)：本轮 `3,824,658` bytes / `727da6095c069bf0750a36cea442c7e582bbfeefdf4fc4a330a656bd8bd45c31`，release `2026-09-22`、Intelligence Index `57.6223698102963`、1M context、约 `$4/$20` input/output；[DataCurve](https://deepswe.datacurve.ai/) `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_claude_opus_5_5_*` 行 | [Anthropic 发布页](https://www.anthropic.com/claude-opus-5-5)、[Model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md)、[What's new](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)、[Migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md)、[System Card](https://www.anthropic.com/claude-opus-5-5-system-card)：2026-09-22 发布；确认 `claude-opus-5-5`、1M/128K、always-on adaptive thinking、default `medium` effort、API compatibility 变更、fast mode、token/steps、Cyber/Life Sciences Verification、fallback、preserved thinking anti-distillation，以及正文中的 RSP、Cyber、Agent safety、prompt injection、OSWorld/ProgramBench/Terminal-Bench 和多 Agent 评测条件 | **AA 单榜 + System Card/API contract + local protocol toy**；参数、架构、完整训练/后训练 recipe、DataCurve Agent 行、System Card 数字的独立复现、生产 kernel、目标硬件 profiling 和线上 acceptance 待核验 |

Anthropic 的 benchmark 表格必须与 harness 分账：多数 Opus 5.5 结果为 adaptive + max，Terminal-Bench 为 xhigh，成本对照常使用 default/medium；Cyber/biology 触发的 fallback 还可能由 Opus 4.8 或其他策略完成任务。发布方结果不迁移为 AA/DataCurve 的裸模型分数。新知识主线是 token efficiency、质量-成本曲线、always-on thinking 与 effort、thinking block binding、tool contract migration、compaction/inline tools、Safeguard capability routing、网络工具宿主和 anti-distillation 状态协议；不新增重复 Transformer 架构章节。

2026-09-28 补入 compaction 双协议：`compact-2026-09-04` on-demand 与 `compact-2026-01-12` threshold 不能混用请求字段、beta header 或平台兼容表。Opus 5.5 的 on-demand keep-tail 仅在 preserved-thinking 条件成立时保留 thinking；threshold compaction 不保留压缩点前的 thinking，暂停后重插 tail 时须移除相关 reasoning block，或按文档选择 `drop_block`。另把 model binding 与 prefix binding 分开，并记录 2026-08-31 00:00 UTC 的新账号默认 prefix-check 门槛及旧账号 opt-in 例外。详见 [`claude-opus-5.5-source-notes.md` §9](claude-opus-5.5-source-notes.md) 和第二十册第 7.25 节；没有真实 Messages API probe。

System Card 正文补齐了安全和长任务面试证据：Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials），ProgramBench `91.2%`（166 golden tasks），OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、500 actions 上限、5 runs、超过 100K tokens 后 compaction）；五 Agent team 在同分 ProgramBench 下约 `2.7x` derived-latency improvement，DRACO 在 `0.5x` latency budget 下约 `2.8x` speedup。CoBench 2.1 `55.8%`、AECI `169.36` 和 Cyber/Gray Swan/Alignment 数字都必须绑定对应 snapshot、safeguards、fallback、环境和 verifier。

训练数据只公开为公开互联网、公开/私有、获许可用户和合成数据的组合，使用去重/分类与 ClaudeBot，不访问密码页、登录页或 CAPTCHA 页面，knowledge cutoff 为 2026 年 6 月。没有因此得到参数、架构或完整 recipe 证据；System Card 还明确列出长轨迹、多 Agent、语言差异和模拟环境真实性等评测盲点。

## 2026-09-23 GPT-6 Sol：长上下文预算与 Agent runtime 协议

本节只升级已经出现在 Artificial Analysis 的 `GPT-6 Sol`，不从 OpenAI 官方模型目录另发现 GPT-6 family 的其他模型。

| 基础模型/配置 | 榜单发现 | 官方核验 | 当前状态 |
|---|---|---|---|
| GPT-6 Sol (`max`) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-6-sol)：当前详情 `3,994,562` bytes / `3d08f91c71e2770a4ef84b593c2a197d711193bbcbd8c3a1f330393b65f0ea86`，release 2026-09、Intelligence Index `47.5276426437724`、median output speed `126.038858615917 tokens/s`、cost per Intelligence Index task `1.0564240894076389`；[DataCurve](https://deepswe.datacurve.ai/) `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_gpt_6_sol_*` 行 | [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)：`gpt-6-sol`、complex coding/agentic workflows、1.05M/922K/128K、`none`--`max` effort、standard/pro mode、动态 `configuration_update`、Responses/Chat Completions function-calling 边界、工具目录、Agent runtime 所有权、opaque compaction state 与 prefix cache 规则 | **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**；参数、架构、完整训练/后训练 recipe、DataCurve Agent 行、完整权重、生产 kernel、目标硬件 profiling、独立 benchmark 和线上 acceptance 待核验 |

GPT-6 Sol 的面试重点是服务合同和运行时状态边界，而不是从工具目录反推模型架构。模型页给出 `reasoning.effort` 的六档值、1,050,000 context、922,000 maximum input、128,000 maximum output、text/image input、text output、Responses/Chat Completions/Batch 端点，以及 web/file search、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search 工具目录。超过 `272K` input tokens 时，整次请求 input/cache 按 2x、output 按 1.5x；Batch/Flex 为 50%，Fast mode 为 2x。这些是可观察 API/provider 字段，不是参数或 FLOPs 证据。

Reasoning 文档把 `mode` 和 `effort` 分成两个控制面：GPT-6 family 的 `standard/pro` 选择执行模式，`reasoning.effort` 控制模式内投入；`configuration_update` 可以在单 Agent 会话中调整后续 effort，但更新项不能相邻，也不能和自动 compaction/truncation 组合。Reasoning tokens 计入 output/context，`usage.output_tokens_details` 可观察统计，预算耗尽可能在可见输出前返回 incomplete。

Agents 文档把 Agents API、Agents SDK 和 Responses API 的 state/loop/executor 所有权分开。Compaction 文档则要求 stateless chain 保留 output items 和 encrypted compaction item，`previous_response_id` chain 不要手工裁剪；standalone `/responses/compact` 的整个返回窗口是 canonical next context。以上协议可以训练长任务 Agent 的状态账本、成本账本和 verifier 追问，但不能证明 GPT-6 Sol 的内部 reasoning architecture、训练 recipe 或永久记忆。

本轮新增 [`gpt6_sol_contract_audit.py`](code/gpt6_sol_contract_audit.py)，用零依赖状态机验证 mode/effort 正交、合法与非法 `configuration_update`、预算 `incomplete`、272K whole-request 计价、permission/executor/verifier、幂等 replay、opaque compaction 和 compaction 后 cache prefix miss。脚本结果为 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`；它是面试协议练习，不是生产 API 验收或模型质量证明。

## 2026-09-23 GPT-6 Luna：高吞吐 sibling 与 runtime 分账

本节只升级已经出现在 Artificial Analysis 的 `GPT-6 Luna`，不从 OpenAI 官方模型目录另发现模型，也不把 GPT-6 Sol 的榜单或 Agent 结果迁移给 Luna。

| 候选 | 排行榜证据 | 官方核验 | 当前状态 |
|---|---|---|---|
| GPT-6 Luna (`max`) | [Artificial Analysis](https://artificialanalysis.ai/models/gpt-6-luna)：`3,992,084` bytes / `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`，release `2026-09-22`、Intelligence Index `37.2559686869738`、速度 `153.87508473888 tokens/s`；[DataCurve](https://deepswe.datacurve.ai/) `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，无精确 `mini_swe_agent_gpt_6_luna_*` 行 | [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：focused/high-volume 定位、1.05M/922K/128K、2026-05-18 cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录、GPT-6 family mode/effort 和 Agent runtime/compaction 协议 | **AA 单榜资料级闭环**；参数、架构、完整训练/后训练 recipe、DataCurve Agent 行、完整权重、生产 kernel、目标硬件 profiling、独立 benchmark 和线上 acceptance 待核验 |

Luna 的面试重点是 task-shape routing 与服务成本账本：官方“高效率”定位不能反推参数规模或 dense/MoE；它与 Sol 共享 GPT-6 family 的 mode/effort、`configuration_update`、Responses typed items、tool search 和 opaque compaction 运行时规则，但模型 ID、knowledge cutoff、价格和榜单配置必须独立记录。模型页与通用 runtime 文档不证明 Luna 的内部 reasoning architecture、训练 recipe 或永久记忆。

## 2026-09-23 DeepSeek V4.1-Flash：API contract 盘点

| canonical 模型 | 官方身份/合同 | 当前证据等级 | 尚未证明 |
|---|---|---|---|
| `deepseek-v4-1-flash` / API `deepseek-flash` | 1M context、384K max output、最高 2500 concurrency；Vision、Files、Responses、Tool Calls 文档已固定 | **AA 单榜 + 官方 API contract + HF/reference + vLLM `v0.30.0` stable/source + SGLang main** | 完整权重加载、FP4/FP8 数值、candidate/index recall、DSpark verify/rollback、EPD、目标硬件、tool/verifier acceptance、生产 SLO |

本条只登记官方可观察服务能力，不把 alias、媒体限制、SSE 事件或 strict schema 当作模型内部架构。Vision 当前约 1024 image tokens/图、URL/file/request 大小和 600 图上限属于 guide contract；历史 384 image-token 公告属于旧 alias/旧时间语境。Responses stateless、不支持项静默忽略、工具回灌和文件生命周期均要在部署 manifest 中独立记录。

### API contract toy 状态

已新增 [`deepseek_v41_api_contract_audit.py`](code/deepseek_v41_api_contract_audit.py)，以本地合成事件和工具调用验证 `schema_valid -> authorized -> executed -> verified`、SSE 序号/终态、执行前有限重试和幂等副作用。该结果是 local protocol toy，不改变模型的 API contract、reference/runtime、完整权重或生产验收证据等级。

## 2026-09-23 Kimi K3 当前时点复验

| 基础模型/配置 | 当前榜单观察 | 当前证据结论 |
|---|---|---|
| Kimi K3 `max` | Artificial Analysis 详情快照 `4,080,091` bytes / `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`；Intelligence Index `43.5938229518782`、median output speed `36.9982439081499 tokens/s`、cost per Intelligence Index task `2.0001323004425493`、1M context | 第三方/provider 配置字段；不是官方参数、架构或独立 benchmark |
| Kimi K3 `max` + `mini-swe-agent` | `mini_swe_agent_kimi_k3_max`：`309/451`、Pass@1 `0.6851441241685144`、Pass@4 `0.8938053097345132`、平均成本 `$4.654682129933482`、平均输出 `81,499.84` tokens、平均 Agent steps `97.5876`、4 runs/113 tasks | 绑定 `mini-swe-agent + tools + environment + verifier` 的系统结果；不迁移到 low、其他 harness 或裸模型 |
| Kimi K3 官方 artifact | README `45,004` bytes / `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2`；HF metadata `9,436` bytes / `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`；revision `f831ab66814297da540d832a5235f8e904f29d06` | 当前时点未发现 revision/架构漂移；完整权重未下载 |

当前 K3 记录继续标为**内容专题闭环**，新增的是榜单和官方 revision 的复验，不升级为完整权重、目标硬件、双状态恢复、tool/verifier 或生产 SLO 闭环。

## 2026-09-23 GLM-5.3：SAO 关联算法证据

| 基础模型/配置 | 榜单锚点 | 新增权威资料 | 当前状态 |
|---|---|---|---|
| GLM-5.3 standard / `max` | Artificial Analysis `glm-5-3`；DataCurve `mini_swe_agent_glm_5_3_max`，`311/451`，Pass@1 `68.9579%`、Pass@4 `87.6106%` | Z.ai GLM-5.3/5.2 官方 Markdown；[SAO arXiv:2607.07508](https://arxiv.org/abs/2607.07508)；论文正文哈希见 [`source-index.md`](source-index.md) | **双榜资料级闭环 + SAO 关联论文算法证据**；5.3 专属 compaction、完整 recipe、完整权重、目标硬件和线上验收待核验 |

SAO 的直接实验主干使用 Qwen3-30B-A3B，摘要称路线部署到 GLM-5.2（750B-A40B）Agent RL pipeline。它可以解释 GLM-5.3 官方所说“继承的 SAO”包含哪些公开优化思想，但不产生 GLM-5.3 独有 benchmark 或裸模型能力。论文的 single-rollout、DIS、双侧 token clipping、K=2 critic update、frozen-attention value model 和 Skip-Observation GAE 进入知识点账本；compaction 序列化、5.3 post-training recipe、目标硬件 profile、tool/verifier acceptance 仍是 `unverified`。

本地 [`sao_async_rl_toy.py`](code/sao_async_rl_toy.py) 已验证上述机制的合成状态机：它输出 group barrier 与 single-rollout 的等待差异、ratio mask、observation-length invariance 和 critic update proxy。该结果单独标为 `local_protocol_toy`，不改变本条模型的资料级闭环状态。

### GLM-5.3 compaction 状态

官方资料仍只披露“继承 `SAO with compaction`”，没有公开状态 schema、序列化、压缩触发或质量门禁。当前本机没有权重、目标 runtime 或可用 NVIDIA 驱动；[`glm53_compaction_contract_audit.py`](code/glm53_compaction_contract_audit.py) 的状态恢复检查仅为 `local_protocol_toy`。因此模型条目继续保持 **双榜资料级闭环 + SAO 关联算法证据 + runtime source evidence**，不得升级为真实 compaction 实现或生产验收。

2026-09-28 补充：Z.ai 文档/博客当前快照不变；固定 `THUDM/slime@8ee9c1e` 为压缩后 Agent RL 样本分叉、token provenance 与 loss-mask 的通用实现证据，并发现该 revision 的 reward 文档与代码/单测存在不一致。它不证明 GLM-5.3 使用该实现；模型专属 compaction 仍 `unverified`。当前 AA/DataCurve 与官方快照、来源边界见 [`glm-5.3-source-notes.md`](glm-5.3-source-notes.md#2026-09-28-当前榜单复验与-slime-的-compaction-aware-trajectory)。

## 2026-09-23 DeepSeek V4.1-Flash：Harness preview 与 runtime 责任边界

本节继续沿两个排行榜已有的 `deepseek-v4-1-flash` canonical 条目推进；DeepSeek Harness 文档只是该模型周边的官方 runtime 资料，不是新的模型发现入口。DataCurve 当前仍没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，不迁移 V4 Pro、V4 Flash 或其他 DeepSeek 版本的 Agent 分数。

官方 Harness preview 的可观察合同可以归纳为：

| 组件 | 证据 | 面试结论 |
|---|---|---|
| provider | 稳定 provider ID、credential 脱敏、协议/reasoning/image 配置 | provider identity 必须可追溯；手工 capability claim 仍需 endpoint probe |
| session | session log 固化实际模型和 runtime 事件 | 不能中途静默换模型；摘要不能替代原始状态/副作用 receipt |
| plugin | 依赖检查、listener/resource cleanup、profile bundle | 源码存在不等于已加载；外部 effect 必须显式清理 |
| MCP | `mcp__server__tool` namespace、环境变量过滤、重连/重发现/注销 | 断线后旧 registry 不再自动可信；预算耗尽应清除工具 |
| GitHub review | signed webhook、异步 `202`、重复 delivery、workspace/session | `202` 只表示 admission；入站认证与出站 GitHub 权限分离 |

这些字段补齐了 `model -> provider -> Harness -> tool/MCP -> executor -> verifier` 的系统边界，但不提升 V4.1 参数、训练、完整权重、kernel、DSpark acceptance、硬件或生产 SLO 的证据等级。[`deepseek_harness_protocol_audit.py`](code/deepseek_harness_protocol_audit.py) 的运行结果为 `ok=true`、`duplicate_webhook_admissions=2`、MCP 工具在预算耗尽后为空、`network_called=false`，证据等级固定为 `local_protocol_toy`。

当前模型状态：**AA 单榜内容专题 + 官方 API/runtime + Harness preview + local protocol toy**。后续仍按 pinned revision -> runtime import -> 完整权重 -> 目标硬件 -> 数值/工具/verifier/SLO 验收推进，不把 Harness preview 或 toy 结果写成 V4.1 内部实现。

## 2026-09-23 GPT-5.6 Luna 当前时点复验与状态回放 toy

本节继续只更新两个排行榜已经确认的 `GPT-5.6 Luna`，不从 OpenAI 官方文档另发现模型。

| 基础模型/配置 | 当前榜单证据 | 官方/runtime 证据 | 当前状态 |
|---|---|---|---|
| GPT-5.6 Luna (`max`) | Artificial Analysis：`3,996,959` bytes / `4246b96416b4aaf4f8bbcacf72f8c45787944e24f02c59c6e3db1fed4de93c41`；Intelligence Index `37.3244239690841`、`142.271568203031 tokens/s`、cost/task `0.17829726152289094`、1M context。DataCurve：`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，精确行仍为 301/448 | Luna/Reasoning/Prompt caching/Compaction/Tools 当前官方快照；`reasoning.context`、opaque replay、cache breakpoint/TTL、canonical compaction、hosted/client tool search | **双榜当前时点复验 + 官方 runtime contract + local protocol toy** |

AA 速度相对 9 月 22 日的变化只作为 provider/采集时点漂移，DataCurve 仍绑定 `mini-swe-agent + tools + task environment + verifier`，不与 AA 字段合成裸模型能力。新增长期状态审计 [`gpt56_luna_state_replay_audit.py`](code/gpt56_luna_state_replay_audit.py) 已通过主流程；它验证 `current_turn/all_turns`、function call/output 血缘、compaction/cache 交互、tool search 所有权和幂等 verifier，不代表真实 endpoint、隐藏 reasoning、模型质量或生产 SLO。

## 2026-09-24 两榜刷新与 Claude Opus 5.5 服务合同

2026-09-24 09:30 UTC 后续重抓：Artificial Analysis /zh 为 HTTP 200、1,784,793 bytes、SHA-256 a374adfb81fea4fc68e7071ef191612d7f7d92c1c0e05412f3c338e7fe9edbc2；与同日稍早页面相差 12 bytes，八家重点厂商 canonical slug 未见新增，动态页面字节变化不解释为模型 revision。DataCurve DeepSWE 规范主机为 HTTP 200、268,036 bytes、SHA-256 14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1，重点厂商配置集合未新增。本次均通过 10.24.27.134:7890 获取。

本节只复验两个排行榜已有的重点厂商 canonical 条目，不将 effort/provider 行另建为模型。

| 来源/模型 | 当前快照 | 结果与边界 |
|---|---|---|
| [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) | HTTP 200；`1,784,805` bytes；SHA-256 `5284847a4499b219872725221324a3461de68464956a4b8f44b5cdd96d34c2c8` | 与 2026-09-23 首页比较，八家重点厂商 canonical 集合无新增；当前最新重点发布列表含 Claude Opus 5.5、GPT-6 Luna、GPT-6 Sol，均已建档 |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | HTTP 200；`268,036` bytes；SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` | `mini_swe_agent_*` 配置集合与上一快照一致；没有重点厂商新配置，Claude Opus 5.5 精确行仍缺失 |
| [Claude Opus 5.5 AA detail](https://artificialanalysis.ai/models/claude-opus-5-5) | `3,810,653` bytes；`cc5a94faec8125d88e46dc6545a564f91f32855b5d05da4df7cd3bca08851e95` | Index `57.6223698102963`、cost/task `$5.982012019521066` 与既有观测一致；为 AA 配置/provider 评测口径 |
| [GPT-6 Luna AA detail](https://artificialanalysis.ai/models/gpt-6-luna) | 同日后续快照 `3,974,386` bytes；`9c6376c8ca63fe1ed56fcc6a85cb5042900ba3ec96df92512760d85002cf594f` | Index `37.2559686869738`、cost/task `$0.06809498628701058` 稳定；速度 `132.242651126596 tokens/s` 按 provider/测量时点记录，较早快照保留作历史，不据此推断 revision |

Anthropic 当前 [Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md) 快照为 `14,244` bytes / `3325e10a852cc3f40cfaf01737d97a5cdf3a37b4cd7c0b68005e204ea32b1dd4`。官方合同为 1M context、同步输出上限 128K、adaptive thinking always-on、默认 `medium` effort、输入/输出 `$4/$20` 每 MTok、cache read `$0.20`、最小 cacheable prompt 512 tokens、5m/1h cache write `$5/$8`；Message Batches API 通过 `output-300k-2026-03-24` beta header 支持最多 300K 输出。它们是 endpoint/服务合同，不能据此推断模型内部算法。

当前 Opus 5.5 状态为 **AA 单榜内容专题闭环 + System Card/API contract + local protocol toy**：研究笔记、正式章节和面试配套均已收口；参数、架构、完整训练 recipe、精确 DataCurve Agent 行、独立 benchmark、目标硬件/生产 kernel 和线上 acceptance 仍待核验。

## 2026-09-24 GLM-5.3：当前时点复验与内容专题闭环

本节只更新已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的标准 `GLM-5.3`，不从 Z.ai 官方资料、Hugging Face、Transformers、vLLM 或 SGLang 另发现模型，也不把 Flash 版结论迁移到标准版。

| 基础模型/配置 | 当前排行榜证据 | 官方/运行时证据 | 当前状态 |
|---|---|---|---|
| GLM-5.3 (`max`) | [Artificial Analysis 详情](https://artificialanalysis.ai/models/glm-5-3)：`4,057,703` bytes / `4f60e893bcfee4586479c77c9d43f1af9f9dee3145e92a23661666d7a072a3fd`；当前页面约 1M context、约 45 Index、约 `$2.01/task`、约 61 tokens/s、`$1.40/$4.40` input/output。DataCurve 首页：`268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，精确 `mini_swe_agent_glm_5_3_max` 行未变 | Z.ai GLM-5.3/5.2 文档、博客正文、SAO 论文；固定 config 的 `glm_moe_dsa`、Transformers `GlmMoeDsaForCausalLM`、vLLM/SGLang DSA source entry；完整 compaction schema 未公开 | **双榜内容专题闭环 + stable/main runtime source evidence** |

本轮 7890 代理复验的 AA 中文首页为 HTTP 200、`1,784,793` bytes / `a374adfb81fea4fc68e7071ef191612d7f7d92c1c0e05412f3c338e7fe9edbc2`，博客壳为 `598` bytes / `240cedb6d23b13b8bdd177e51410dbe1c7783fbd0cfca98be1e0af26688878c0`，正文 JS 为 `30,414` bytes / `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`，官方 Markdown 与既有 `9545c3d6…` 快照一致。页面动态字段变化只记为 provider/采集时点，不解释为模型 revision。

内容专题闭环意味着 DSA/Full-Shared indexer、MoE、长任务环境与 verifier、SAO 关联算法、slime 训练—rollout 对齐、thinking/tool/cache 和 compaction 验收方法已经能支撑面试回答；它不等于完整权重、目标硬件、独立 benchmark 或生产服务通过。完整权重、stable wheel、recall/量化/MTP/profile、5.3 专属 compaction、完整 recipe、独立 benchmark、tool/verifier acceptance 和生产 SLO 仍为 `unverified`。

## 2026-09-24 GPT-6 Luna：当前时点复验

本条只复核 AA 已发现的 `GPT-6 Luna (max)`，不从 OpenAI 官方目录新增模型。7890 对 Artificial Analysis 中文首页、Luna 详情、DataCurve 和 OpenAI 模型页均可访问。

| 来源/对象 | 当前快照与字段 | 边界 |
|---|---|---|
| [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) | `1,783,893` bytes / `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36` | 动态页面快照；重点模型 canonical 集合无新增 |
| [GPT-6 Luna (max)](https://artificialanalysis.ai/models/gpt-6-luna) | 同日后续快照 `3,974,386` bytes / `9c6376c8ca63fe1ed56fcc6a85cb5042900ba3ec96df92512760d85002cf594f`；Index `37.2559686869738`、cost/task `$0.06809498628701058`、median output speed `132.242651126596 tokens/s`、约 1M context。此前同日快照为 `3,974,113` bytes / `8a4603328627740cab856ad4d9b0b37415697b66cc33c59220b0f16615327c81`，速度 `131.449006457181 tokens/s` | 第三方/provider 字段；指数与任务成本稳定，速度/页面字节变化属于采集时点，不代表模型 revision |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` | 无精确 Luna Agent 行，不迁移 Sol/Astra/GPT-5.6 结果 |
| [OpenAI GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md) | `4,019` bytes / `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945` | 与既有文档快照一致；合同无可见变化 |
| [OpenAI API data residency guide](https://developers.openai.com/api/docs/guides/your-data.md) | HTTP 200；`79,672` bytes / `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648` | Sol/Luna 的 EU residency 仅明确适用于 Standard processing 的 Responses/Chat Completions；regional storage 不等于 regional processing，system data 与 Remote MCP 第三方数据不自动落入保证范围 |

当前状态仍为 **AA 单榜资料级闭环**。已同步到第六/十六/十七/二十/二十四册相关正式小节、面试题和练习的内容聚焦在 sibling routing、whole-request cost ledger、工具责任边界和精确 Agent attribution；本轮复验未发现值得新建章节的 Luna 独有技术披露。内部架构、完整 recipe、DataCurve 精确 Agent 行、完整权重、独立 benchmark、目标硬件 profile 和生产 acceptance 继续未核验。

## 2026-09-24 GPT-6 Sol：榜单当前快照与 EU residency 合同

本条继续沿 AA 已发现的 GPT-6 Sol 锚点推进，不从 OpenAI 官方目录另发现模型。实时榜单检查使用 1234 代理获取 Artificial Analysis，7890 代理获取 DataCurve；八家重点厂商的 AA canonical slug 与 DeepSWE 配置集合未发现新增条目。

| 来源/对象 | 当前快照与字段 | 边界 |
|---|---|---|
| [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) | `1,783,966` bytes / `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8` | 1234 代理抓取；只作为两榜发现入口快照 |
| [Artificial Analysis GPT-6 Sol](https://artificialanalysis.ai/models/gpt-6-sol) | `3,976,802` bytes / `ff0aeaedb21ad7a672653b3d019c0574c6e468a74808db1f3973731d311d5ba5`；Index `47.5276426437724`、cost/task `$1.0564240894076389`、median output speed `109.551294011457 tokens/s` | 第三方/provider 字段；Index 与任务成本保持既有值，速度从较早记录 `126.038858615917 tokens/s` 变化只作时点观察，不推断 revision |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` | 与前次快照一致；没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不迁移其他模型/effort 的 Agent 分数 |
| [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md) | `3,991` bytes / `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0` | EU data residency 只标 Standard processing；regional processing 溢价 10% |
| [OpenAI API data residency guide](https://developers.openai.com/api/docs/guides/your-data.md) | HTTP 200；`79,672` bytes / `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648` | Sol/Luna EU residency 只对 Standard Responses/Chat Completions 明确可用；storage 与 processing 区分，system data 与第三方 MCP 在保证范围外 |

面试新增点：把 `Standard processing` 与 reasoning `mode=standard` 视为不同字段；Residency eligibility 应按 model × endpoint × region × processing mode 查询，不能把 Batch/Flex/Fast 价格选项当作资格证明。当前状态仍是 **AA 单榜资料级闭环 + 当前时点复验 + local protocol toy**；没有新的 Sol 专属架构、训练 recipe、DataCurve Agent 行或生产验收证据。

## 2026-09-24 GPT-6 Sol：动态 effort telemetry 官方复核

两榜当前快照仍为 AA `/zh` `1,783,966` bytes / `0baa28c8ab69a104a078645983ca843c95c5db879b6c52789104aa3d1758efb8` 与规范 DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；重点八家 canonical 集合没有新增模型。项目环境直连 7890 TLS EOF；按审批在限制外用同代理取得 HTTP 200。

OpenAI 官方 Reasoning 当前正文 HTTP 200（经官方 `platform.openai.com` clean HTML 路径跳转），`1,070,709` bytes / SHA-256 `3bcb1b7771731d3269783464180b72c2b45283688118dda185ebd2d65e84616c`。GPT-6 family 的 `configuration_update` 可在 standard single-agent Responses/WebSocket `response.create` 历史中调节 effective effort，直到下一次覆盖；响应里的 `reasoning.effort` 仍表示 request-level setting。保留固定请求配置并追加 update 可保留 prompt prefix、利于 cache reuse，但不保证 cache hit。与 automatic compaction/truncation、独立 `/responses/compact` 的 incompatibility 及显式 compaction 后 fresh update 要求仍有效。

已将该区别同步至 GPT-6 Sol 研究笔记、第十六/二十册、`INTERVIEW_BANK.md`、`EXERCISES.md`、`KNOWLEDGE_GRAPH.md` 与 `plan.md`。新增 toy 检验 session override 和 request-level telemetry 分离、稳定 prefix digest 不变；仍明确不等同真实 endpoint 或 cache hit。Sol 继续是 AA 单榜资料级闭环，内部结构/recipe/完整权重/独立评测/DataCurve 精确 Agent 行仍未确认。

## 2026-09-24 Kimi K3：vLLM v0.30.0 release/source 更新

| 榜单模型锚点 | release/source 变化 | 状态与边界 |
|---|---|---|
| Kimi K3 (`kimi-k3`; DataCurve `mini_swe_agent_kimi_k3_max`) | vLLM `v0.30.0` tag `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；K3 stable registry/model/DSpark entry 在 `v0.29.0` 已存在，v0.30 比较显示实现演进，不是首次支持 | **内容专题闭环 + v0.30.0 stable release/source 更新**；IPC weight cache、stream finalize、PP auxiliary hidden states、DSpark context-KV gate 和 KDA state dtype 已记录。wheel 未下载/安装，完整权重、目标硬件数值/profile、双状态恢复和生产 acceptance 未验证 |

本轮发现入口仍严格限 Artificial Analysis 与 DataCurve；vLLM release 不作为新增模型来源。v0.30 wheel metadata 是 artifact 可用性证据，不等于本地安装或 K3 serving 已验收。

## 2026-09-24 Claude Opus 5：System Card 安全评测与方法学补证

- 本轮继续沿两榜既有 `claude-opus-5` canonical 锚点；未从 Anthropic System Card 或公开研究目录发现新模型。Artificial Analysis `/zh` 为 `1,781,428` bytes / SHA-256 `15e9763f0d516791e7b7c2c5679e87e5878eaffed4bef31e485c9043b758ba59`；Opus 5 详情为 `3,977,410` bytes / `18acc956d77c5b49c391eb85b46ef2c7fdbd7edd282ddf2275940bd418d25774`，Index `50.7771115797629`。DataCurve DeepSWE 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，精确 `mini_swe_agent_claude_opus_5_max` 仍为 `327/444`、Pass@1 `73.6486%`、Pass@4 `88.4956%`；都是对应配置与 harness 的系统结果。
- Anthropic [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card) 当前 PDF `16,281,258` bytes / SHA-256 `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`，与旧副本相同；用仓库 parser 抽取 `4,641` 行、`334,056` bytes，文本 SHA-256 `4ae20472ab82c967d90f386239ee6987ddf74b1124db6b44a1dea69a576e1f4c`。卡片 changelog 最近记载 2026-08-19 的 prompt-injection bounty 数据和 Cowork harness 重跑，不是 9 月 24 日新发布。
- 新的面试知识是评测归因：已饱和 ART 改为 IPI/adaptive red-team；单独记录 model-only 与 probe/Auto mode 产品层；attempt-level 与 scenario-level ASR 同时报；Cowork harness 版本不一致时重跑旧基线。Opus 5 的 RSP 为发布方将其评为 CB-1、未达 CB-2 并保持 ASL-3 防护；内部行为审计和监测同为发布方条件证据。
- 已同步第八册第 11 章 6.5、第二十册第 19 章 19.38、研究笔记、来源索引、榜单解释、题库、练习、论文索引、项目和知识图谱。当前为 **双榜锚点 + System Card Agentic Safety/评测方法内容专题闭环**；参数/架构、完整训练 recipe、独立复现、真实生产风险概率和 SLO 仍 `unverified`。

## 2026-09-24 GLM-5.3-Flash：vLLM stable/fixed-main 文件对照

该条目沿用既有双榜 canonical：Artificial Analysis `glm-5-3-flash` 与 DataCurve `mini_swe_agent_glm_5_3_flash_max`，不是从 vLLM 发现模型。vLLM `v0.30.0`（tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，2026-09-22 发布）的官方 `vllm/models?ref=v0.30.0` Contents API 返回 `vllm/models/glm5next/` 目录，tree SHA `118165530271cb3aa749d77bf927499957db2bd4`；此前固定的 `v0.29.0` tree 未检出该目录。

递归 tree 已固定并与 vLLM main commit `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa` 逐文件比对。v0.30.0 stable 已有 IndexerCache/TailCache、KDA、MTP 和 multimodal 源码；KDA 与 multimodal 对应 blob 完全相同，MTP 主体相同但 import path 不同。attention 文件虽不同 blob，两个版本均实现 pool metadata 与未完成 pool tail；main 后续改为模型专属 sparse-indexer 路径、pool-length workspace sizing 和新的 RoPE 类型映射。main 的 Quark `.weight_scale`/dense gate-up mapping 不属于 stable。KPool tail-cache kernel 的 main 版使用真实 tensor strides，而 stable 版按 dense offset 计算；这一差异列为需带目标 cache layout 实测的正确性风险点，不直接判为 bug。

证据状态为 **v0.30.0 stable source feature paths confirmed + fixed-main deltas documented**。这不证明 wheel 已安装、完整权重已加载、cache stride/layout 数值正确或目标硬件/生产 acceptance 通过；见 [`GLM-5.3-Flash source notes`](glm-5.3-flash-source-notes.md)。
## 2026-09-24 Claude Fable 5.1：排行榜复验与 thinking-state 专题升级

| canonical anchor | AA / DataCurve 当前观察 | 权威资料增量 | 状态 |
| --- | --- | --- | --- |
| `claude-fable-5-1` | AA `max with fallback` Index `53.3549259623252`、cost/task `7.629706364004841`；DataCurve 无精确 Fable 5.1 行 | Anthropic Thinking 文档明确 Claude API 上 Opus 5.5 → Fable 5.1 可读，反向不可读；并将 model-binding 与 prefix-binding 的错误/丢弃、计费和 telemetry 分开 | **AA 单榜内容专题闭环 + API/System Card + local protocol toy** |

本轮没有产生新 canonical 模型。Artificial Analysis 首页/详情与 DataCurve 快照及官方文档的日期、大小、哈希见 [`source-index.md`](source-index.md) 和 [`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。正式教学落点为第二十册第 21 章 21.29。文档可读边不等于默认 fallback target：Fable 5.1 文档列出的默认 fallback 仍是 Opus 4.8/Opus 5。真实 API、参数/架构、完整训练 recipe、独立 benchmark、目标硬件和生产 SLO 仍未验证。

## 2026-09-24 DeepSeek V4 Flash：SGLang v0.5.20 serving update

本节沿用两个排行榜已确认的 DeepSeek V4 Flash canonical 锚点（AA `deepseek-v4-flash`、DataCurve `mini_swe_agent_deepseek_v4_flash_max`）；SGLang 仅作为该锚点的 serving/runtime 来源，不作为模型发现入口。官方 `/releases/latest` 当前仍指向 `v0.5.20`。新补的 interview 主线有三条：

1. Unified radix tree 在共享分支点保留 SWA state，使 Full KV prefix hit 不会掩盖缺失的滑窗状态；SGLang PR #34565 的 DeepSeek-V4-Flash-0731 指定 shared-prefix workload 报告 token hit rate `43.81% → 60.75%`、mean TTFT `1,569.93 → 1,069.58 ms`，属于 PR 自报、条件绑定的 serving 结果。
2. B200/SM100/103 的 TRT-LLM kernel 覆盖 CSA/HCA；PR #30805 报告 FP8/TP1 unit-kernel prefill `~1.2x`、decode `~1.45x`，不是端到端 benchmark。
3. RTX PRO 6000/SM120 通过 DeepGEMM paged-MQA indexer、FlashInfer sparse prefill 和 DeepGEMM FP4 MoE 优化；PR #29927 的 `3.4x` 单并发 TPOT 以慢速 torch fallback 为 baseline，且另一个 HC prenorm 改动贡献约 `3.2%`，不可归因成模型能力提升。

细节、来源快照、硬件/负载和证据边界见 [`deepseek-v4-source-notes.md`](deepseek-v4-source-notes.md)；稳定包中的 V4.1 FlashMLA pin 与完整 V4.1 runtime/vision 证据分开记录于 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。本轮未改动排行榜分数或新增模型身份。

## 2026-09-24 Kimi K2.7 Code：榜单复验与 API 资料专题

| 锚点 | 两榜当前观察 | 官方增量与边界 |
|---|---|---|
| Kimi K2.7 Code | AA 当前 Index 25.8121062401836、256K context；DataCurve 精确 mini_swe_agent_kimi_k2_7_code_default：113 tasks、4 runs、Pass@1 30.53%、Pass@4 61.06%、452 attempts/138 passed | 官方快速开始补充默认 max_tokens=32,768、强制 thinking、固定 sampling 参数、视觉 token 估算和视频文件/抽帧限制；另列同一模型的 Highspeed 服务变体。固定部署指南给出 KTransformers RAWINT4 与 LoRA SFT 场景数据，但与同提交通用支持矩阵未对齐，K2.7 兼容性未实测。K2.5 `0.7.0.post4` SFT recipe 明确区分 RAWINT4 权重后端与 `bf16:true` 训练字段，不外推为 K2.7 配方；发布方吞吐不是本项目硬件实测。细节见研究笔记 §9 与第二十一册第 94.14–94.15 节 |

Kimi API 文档称 kimi-k2.7-code-highspeed 与 kimi-k2.7-code 是同一个模型，速度值为发布方声明且注明供给有限；它不是新 canonical 模型候选，不另建 inventory 行。2026-09-29 固定 revision README/config 与部署指南已通过 7890 复取；固定版本哈希、部署场景和证据边界见 [Kimi K2.7 Code source notes](kimi-k2.7-code-source-notes.md) 与第二十一册第 94 章。

## 2026-09-24 Qwen3.5-Omni Plus / Flash 技术专题

| canonical anchor | 榜单观察 | 权威来源与新增知识 | 状态 |
|---|---|---|---|
| `qwen3-5-omni-plus` / `qwen3-5-omni-flash` | Artificial Analysis 有 Plus、Flash 详情；Plus 的 256K、Index、速度与价格是第三方/provider 快照字段。DataCurve `268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` 未检出精确 `mini_swe_agent_qwen3_5_omni_*` 行 | Qwen3.5-Omni Technical Report v2 与 Model Studio 官方 API 文档；Hybrid MoE Thinker-Talker、6.25 Hz AuT、显式秒级 timestamp + TM-RoPE、ARIA 单流 token-rate 约束、分阶段多模态训练/后训练 | **AA 单榜内容专题闭环**；DataCurve Agent 行、完整参数/recipe、权重加载、独立 benchmark 和生产验收仍未确认 |

Plus 与 Flash 在此按同一家族的两个榜单配置记录，不扩张为两个基础模型。详见 [`qwen3.5-omni-source-notes.md`](qwen3.5-omni-source-notes.md) 与第二十一册第 93 章；不把 Qwen3-Omni 的 12.5 Hz、80 ms 或首包数字迁移到本代。

## 2026-09-28 Qwen3.6-27B 专题

| canonical anchor | 榜单观察 | 权威来源与新增知识 | 状态 |
|---|---|---|---|
| qwen3-6-27b / qwen3-6-27b-non-reasoning | Artificial Analysis 两条配置标记 2026-04-22，归并为一个基础锚点；DataCurve 当前快照无精确 Qwen3.6-27B Agent 行 | Qwen 官方 ModelScope 卡/config：27B dense、64 层、3×Gated DeltaNet + 1×Gated Attention 周期；Qwen 官方博客经文章 API 恢复正文并披露发布方 benchmark 与客户端/API 示例；另有一篇外部 arXiv 预印本 GDN Tree-Scan，在该 checkpoint 上讨论 recurrent-hybrid tree verification | **AA 单榜身份核验 + 官方模型卡/config/template/博客 + 第三方 serving 预印本专题**；博客不是专属架构/训练报告；预印本为作者自报的 B=1 decode 结果，未独立复现；完整权重和线上/硬件验收仍未确认 |

详见 [qwen3.6-27b-source-notes.md](qwen3.6-27b-source-notes.md)、第二十一册第 83 章 83.16.4–83.16.5 与第二十四册第 61 章 61.31–61.32。GDN Tree-Scan 是外部 serving 研究，不应记成 Qwen 官方架构或性能承诺。

## 2026-09-28 Qwen3.6-35B-A3B 专题

| canonical anchor | 榜单观察 | 权威来源与新增知识 | 状态 |
|---|---|---|---|
| `qwen3-6-35b-a3b` / `qwen3-6-35b-a3b-non-reasoning` | Artificial Analysis 两条配置均标记 2026-04-16，归并为同一模型；DataCurve 当前快照无精确 Qwen3.6 Agent 行 | Qwen 官方模型卡/config/chat template + 发布博客 + arXiv v1 Qwen Technical Report VHD-Play：35B/3B、周期性 Gated DeltaNet/Gated Attention + MoE、历史 thinking block 保留协议、benchmark evaluator 依赖，以及机制先求解再生成可验证 Agent 环境的作者报告 | **AA 单榜身份核验 + 官方资料/技术报告方法专题闭环**；VHD-Play 训练与 benchmark 数字是论文作者报告、未独立复现；完整预训练 recipe、真实 API、硬件和多 seed 复现未确认 |

详细分数、SWE-bench Pro revision、Terminal-Bench/SkillsBench harness 及 TAU3/VITA/MCP evaluator 条件见 [`qwen3.6-35b-a3b-source-notes.md`](qwen3.6-35b-a3b-source-notes.md) 和第二十一册第 83 章 83.15.3；不将 GPT/Claude/Gemini evaluator 作为本项目新模型锚点。

详见 [`qwen3.6-35b-a3b-source-notes.md`](qwen3.6-35b-a3b-source-notes.md)。`preserve_thinking` 只控制当前 prompt 对历史 reasoning block 的序列化，不是独立长期记忆。

后续发现的 [arXiv:2609.27321v1《Verifiable Hidden Dynamics Play》](https://arxiv.org/abs/2609.27321v1) 评论标为 “Qwen Technical Report”：它以 Qwen3.6-35B-A3B 为训练模型，报告机制先求解、再生成可验证 stateful environment 的 Agent RL 方法；Qwen3.7-Max 仅作辅助 setter/benchmark 对照。该论文结果是作者报告，且不证明 VHD-Play 属于 Max 的内部训练 recipe。完整证据和实验边界见 [`qwen3.6-35b-a3b-source-notes.md`](qwen3.6-35b-a3b-source-notes.md#9-vhd-play先求解机制再生成可交互环境) 与第二十册第 19.41 节。

## 2026-09-28 Qwen3.8 Max (0902)：缓存与 Responses 状态契约复验

本轮仍沿用两榜中既有的 Qwen3.8 Max canonical/service revision，不从 QwenCloud 官方资料另发现模型。7890 当前可由 agent 工作区直连两榜与 QwenCloud 文档：

| 来源 | 当前响应 | 观察 |
|---|---|---|
| Artificial Analysis `/zh` | HTTP 200，1,674,048 bytes / `56359658c7d187e4a214ae65857f0a7668d42ebcc1da0e2d4fab477642e2640c` | 较同日早先快照相差 139 bytes，视为动态页面漂移；未据此宣称模型更新 |
| Artificial Analysis release / Qwen3.8 Max detail | release：938,807 / `913306564eb762ad0587c8845bece903f5004a0198f64709af7318fbcb960ff2`；详情：3,881,440 / `f9c920db61e73a4d71fcd9a10d057d7bf22ce01d979974dd32ec4ed1f911cccb` | 0902 仍沿用 canonical `qwen3-8-max` 下的 release/service revision |
| DataCurve DeepSWE | HTTP 200，268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` | 仍只有泛化 `mini_swe_agent_qwen3_8_max_xhigh`，无精确 0902 行；不迁移成绩 |
| QwenCloud 产品页 | 98,992 bytes / `ce97b7c9098ed54c195aa2c6b1a5c0dfc6ad43e7290aea10bcce8484f42b9a6b`；`last-modified=2026-09-28 09:55:09` | 仍称 0902 为 `qwen3.8-max` upgraded snapshot；更新的多工具/端到端 Agent 与视觉文案是产品定位，不是架构报告或定量 benchmark |

官方缓存合同的面试增量为：1,024-token 门槛是资格而非命中保证；同页带 `cache_control` marker 的 explicit-cache 示例另写 “input must exceed 1024 tokens”，边界措辞需 endpoint probe；marker 前向回看最多 20 个 content block，与缓存块后 `other messages≤20` 才复用的 message 计数规则不同。缓存字段也需按 endpoint 区分：Chat Completions 为 `usage.prompt_tokens_details.cached_tokens`，Responses 为 `usage.input_tokens_details.cached_tokens`。Responses 的 `previous_response_id` 会延续前轮 input/output，但不会自动继承前轮 `instructions`；不能与 `conversation` 同用，`store=false` response 不可续接，response ID 文档有效期为 7 天。当前缓存和限流文档核心 0902 字段仍为 5 分钟 TTL、四 marker、`1,500,000 / 1,500,000 / 1,500,000` guaranteed TPM。

官方页面快照、字段范围及 source boundary 见 [qwen3.8-max-0902-source-notes.md](qwen3.8-max-0902-source-notes.md)。已扩展第二十一册第 83 章、第二十四册第 32 章、面试题、练习和本地 cache contract toy；没有调用真实 API、下载权重或将 hosted 服务说明写成模型内部结构。

## 2026-09-28 GPT-6 Luna：7890 当前快照与 API 状态边界

本轮只复验两个排行榜中已有的 `gpt-6-luna`，不从 OpenAI 文档另发现模型。经 `10.24.27.134:7890` 获取：

| 来源 | 当前快照 | 结论 |
|---|---|---|
| Artificial Analysis `/zh` | 1,660,578 bytes / `d95b93814772c81fd71e3b33bf2851b7b87edf7d62ce5a7ef1336ae0169bae2f` | 动态首页；八家重点厂商未发现新 canonical |
| [GPT-6 Luna 详情](https://artificialanalysis.ai/models/gpt-6-luna) | HTTP 200，3,840,137 bytes / `0c62e5133280f5b671d1d9c0b2fd550672d7ed2f3a96feab9149448969224f1f` | 仍列 `gpt-6-luna`；页面大小/hash 漂移不单独证明模型 revision |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` | 无精确 `mini_swe_agent_gpt_6_luna_*` 行；不迁移其他 GPT 结果 |
| OpenAI model / Reasoning / Agents / Tools / Compaction | `4,019 / 70,315 / 5,432 / 33,282 / 14,272` bytes；哈希见 [研究笔记](gpt-6-luna-source-notes.md) | 均与已存官方页面快照一致 |

官方 GPT-6-family Reasoning guide 明确：`configuration_update` 不能与 automatic compaction/truncation 组合，含 update 的 history 也不能传给 standalone `/responses/compact`；要在 `/responses` 显式压缩，可用 `compaction_trigger`，并在压缩后、下一条 user message 前重新放置 update。该状态协议已在第二十册第 20.27.2 作通用说明，本轮仅把权威来源锚定到 Luna；不是 Luna 专属内部架构。状态继续为 **AA 单榜资料级闭环**，没有真实 API probe、权重下载或生产验收。

## 2026-09-28 DeepSeek V3.2 Search Agent context-management 补证

本轮只沿已有 `deepseek-v3-2` 锚点复验，没有从论文或实现仓库发现新模型。Artificial Analysis 详情 HTTP 200，3,647,642 bytes / SHA-256 `239b8fef11d18d7b06e5e7c177ed76cbbfd29e07d795d83c5dc3e6ac8acef0b8`；当前仍是 Non-reasoning、2025-12-01、685B total/37B active、128K、Index `16.043537719683`。9 月 20 日约 648B 只作历史页面字段。DataCurve 当前快照 268,036 bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，无精确 `mini_swe_agent_deepseek_v3_2_*` 行，不迁移任何 DeepSeek 邻近版本指标。

后续完成 Figure 6 视觉复核：arXiv v1 SVG 经 7890 获取并与 1234 抓取字节一致（75,689 bytes，SHA-256 `e2fb1029cf88458b2881d49546c1b9db4dd426fb4d2b3337dd6897b9c6fdb7a3`）；确认四种 context-management/并行策略及 Real Steps/Browsecomp 坐标轴。未将无标签散点目测值当作精确测量；Summary 的 364 steps / 60.2 仍以论文正文报告为准。没有新增模型或改变锚点状态。

固定 [arXiv v1 HTML](https://arxiv.org/html/2512.02556v1) 的 §4.4 披露 Search Agent 在上下文用量超过 80% 时采用 Summary、Discard-75%、Discard-all；另以 N 条独立轨迹选最少 steps 的 Parallel-fewest-step 作对照。作者称约 20% 以上案例超过 128K；BrowseComp Pass@1 为无 context management 51.4、带管理 67.6*。Summary 平均 364 steps，“up to 60.2”原文未标百分号。内容已映射到第二十一册第 19 章 19.35、第二十册第 18 章 18.27、INTERVIEW_BANK 与既有 BrowseComp Context Manager exercise；结果绑定 Search Agent / commercial-search harness，不是 DataCurve 或裸模型分数。

- Figure 1 benchmark SVG 已视觉核验（128,338 bytes / SHA-256 `14354740f4d54692b6af6323cc12d3a5f0e0f937bc2b7dd65021307bd7820973`）：不同任务使用不同量纲轴；HMMT 2025 February、HLE text-only、Thinking 的 HLE 模板差异、tool-use 内部环境等均须绑定报告口径。图表结果不是 AA/DataCurve 或独立复现；对应第二十一册 19.41。Figures 1–7 均已视觉核验，其余公式仍待检查。

- DSA Eq. (1)–(4) 的 arXiv HTML TeX source 已文本核对：Eq. (1) 是加权 ReLU rank score，Eq. (2) 才由 Top-k latent KV 调主 attention；Eq. (3)/(4) 分别对应 dense target 与 sparse selected-set KL。Eq. (4) 未明确展示 `p_{t,S_t}` 的 post-selection normalization，需查训练代码，不脑补。PDF 已获取但本环境无法渲染，未声称视觉核验。对应第二十一册 19.42。

- V3.2-Exp 官方 GitHub 网页/raw README 可取，但公开根目录只显示 inference demo、报告与相关 kernel 链接；当前未见 trainer/loss 代码路径。GitHub API tree 返回 403，故不把它写成“DeepSeek 从未公开训练实现”的全局结论；Eq. (4) 归一化细节仍未解。

## 2026-09-28 Gemini 3.8 Flash：7890 官方页面 HTTPS 重试

沿已有榜单锚点重试先前记录的两个 Gemini 3.8 HTTPS 页面：Artificial Analysis high 详情经 `10.24.27.134:7890` + HTTP/1.1 返回 HTTP 200，`3,842,574` bytes / SHA-256 `f88d6155277116adc1580030fa298d821edf244375c76526e9d68572762b12f5`；Google Video Understanding 返回 HTTP 200，`303,547` bytes / SHA-256 `e65216368a92d03403d4e37485148f7048b1f7e75be6de37891fd0ab2cb01414`。AA 仍为 2026-09-02 / Index `40.9262321765904` / 1M / `$0.75/$3.75`；此页 `medianOutputSpeed=311.225851164667`，其他同日测量快照保留各自字段，不解释成模型更新。Google 视频页关键 media-step 与 thought/tool-use usage 字段未变。没有新增模型或进行真实 API probe；原 TLS EOF 已由后续 HTTP 200 重试修复。

## 2026-09-29 两榜刷新与 DeepSeek V3.2 图示核验

- 同一既有 V3.2 锚点的 RL 公式补证：arXiv v1 Eq. (5)–(9) HTML TeX annotations 已文本核对，确认 Eq. (5) 先做 group response mean 再做每条 response token mean；Eq. (6) advantage 展示组内减均值而非显式 z-score；Eq. (7) 用 current/old importance ratio 估计 current/reference KL；Eq. (8) 的 mask 不覆盖 KL 项；Eq. (9) 依据负 advantage 与 response 平均 divergence 门控。对应第二十一册 19.43。未在 PDF 页面视觉核验，不视为完整 RL recipe。

- Artificial Analysis `/zh` 当前快照：1,694,614 bytes / SHA-256 `04f62f3f39c1f65c3c1d8dd564782f9d7d20fb506fa671ee75fcfaf4e9482f9f`。相较留存首页唯一 `/models/<slug>` 路由数由 55 增至 62，无移除；新增七条中 GLM-5.3-Flash、Kimi K3、Qwen3.8 2.4T A95B、Qwen3.8 27B 已在本项目盘点中，另外三条属于暂不跟踪厂商。八家重点厂商没有新增 canonical 锚点。
- DataCurve DeepSWE：268,036 bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`；70 个唯一 `mini_swe_agent_*` 配置 ID 与留存快照相同，无新增模型配置。
- DeepSeek V3.2 仍为既有 AA 锚点，DataCurve 无精确 Agent 行。arXiv v1 Figure 2 已渲染核验，indexer/top-k 仅负责候选检索，主注意力计算输出；研究笔记与第二十一册第 19 章 19.36 已同步。完整记录见 [`deepseek-v3.2-source-notes.md`](deepseek-v3.2-source-notes.md)。
- Figure 3 的 prefill/decode cost panels 也已视觉核验：H800 实际部署服务、`$2/GPU-hour` 租赁估价下，V3.2 曲线在长 token position 更平缓，极短位置则有交叉；不把图上成本当成 API 价格或硬件无关结论。对应第二十一册 19.37；其余论文图表/公式尚未全部核验。
- Figure 4 也已视觉核验：同一轮内追加 tool call/result 保留旧 thinking；新 user message 到来后，旧 thinking 被移除，而 tool calls/results 与前轮 answer 留存。该机制依赖消息类型与 history replay，不应写成永久记忆。对应第二十一册 19.38。
- Figure 5 合成 Agent RL 曲线（arXiv v1 `synthesis-rl-plot.png`，222,137 bytes / SHA-256 `2107344cc2002f51bc1922df0bdcabd7afa74bb0f6cb5d3aa996f33e3f0d7cc3`）已视觉核验：发布方训练消融总体显示合成 general-agent tasks 上 RL 曲线随 steps 提升，但分环境有波动。它不是最终 V3.2 的独立 benchmark，不能据此证明真实环境泛化因果、数据无污染或完整 recipe 已公开；对应第二十一册 19.39。Figures 2–7 已视觉核验，其余论文图表/公式仍待检查。
- Figure 7 的 MLA-MHA/MQA 两个 SVG panel 已渲染视觉核验，文件大小与 SHA-256 见 source-index。图示说明 MHA per-head K/V projection 与 MQA shared latent KV 的差别；论文图注的 training/prefill MHA、decode MQA 明确指 V3.1-Terminus，V3.2-Exp inference demo 是另一份实现证据。不能把图示提升为所有 serving backend 的合同或实际性能保证。对应第二十一册 19.40；Figures 2–7 已视觉核验，其余图表/公式仍待检查。

## 2026-09-29 Kimi K3 serving 补证

Kimi K3 仍是两榜既有锚点，没有由 vLLM/SGLang runtime 仓库新增候选。当前 AA 首页快照包含 K3 路由（1,746,726 bytes；60 个路由），DataCurve 快照仍含精确 `mini_swe_agent_kimi_k3_max`（268,036 bytes；70 个配置 ID）。

新增上游证据聚焦 serving：vLLM main 的 adaptive DSpark 以 confidence head 产生逐位置接受概率，以 `k+1` CUDA graph capture 配合 offsets/masks 容纳批内变长 verify；cache rebind 修复要求 context pointer cache 跟随 buffer 地址变化失效。MI355X recipe 与 SiTUv2 a4w4/AITER 版本绑定，仍属 nightly/pre-release；AgentX synthetic acceptance throughput sweep 跳过 target verification。以上只支持源码、配置和发布方基准语义，不支持权重加载、目标卡 correctness/performance 或线上 acceptance 结论。固定提交、patch 哈希与版本界限见 [`source-index.md`](source-index.md) 和 [`kimi-k3-source-notes.md`](kimi-k3-source-notes.md)。

## 2026-09-30 Kimi K3 DFLASH

Kimi K3 仍是两榜既有锚点；9 月 30 日 AA 首页为 1,752,755 bytes / `35e3197c2ad3b4259369bf55f72d912e03c0ecd61ad7d181db1fdda86f0dd527`，DataCurve 为 268,036 bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`。新增内容是 K3 的 serving 周边，不是新模型候选。

公开 `modal-labs/Kimi-K3-DFlash` revision `c192d15a...` 是 draft-only BF16 配套模型，metadata 统计约 2.6B 参数；config 为 6 层 draft、93 层 target、六个 target output taps 和 block size 16，README 推荐部署默认 block size 8。DFlash 论文给出 target hidden feature 逐层 K/V 注入与 block diffusion 并行 draft；SGLang #40794 为 K3 增加不带 `+1` 的 layer-output capture hook。PR 集成结果使用未公开 production draft 和多 PR build，不能升级为公开 checkpoint 的可复现结果。状态更新为 **K3 内容专题闭环 + DFlash paper/HF/main source evidence**；权重、目标硬件、stable release、真实 acceptance 和生产 SLO 仍未验证。

## 2026-09-29 GLM-5.3 关联框架：Straw checkpoint 与 archive

GLM-5.3 仍只由两榜作为模型锚点；Z.ai 博客引用 slime，当前 slime README 也列出 GLM-5.3。本轮核对的是 2026-09-28/29 的通用上游能力：distributed fully async rollout、Straw 持久队列、模型与 queue/builder snapshot 的联合 commit marker、历史 step branch rollback，以及带独立 retention ownership 的 indexed rollout archive。它们不能倒推成 GLM-5.3 发布时使用的确切实现或其 compaction recipe。四卡 Qwen2.5-0.5B fork test 只作为上游测试源码，尚未运行；全权重、真实 GPU 恢复、生产 profiling 和 GLM-5.3 专属状态继续未验证。完整文件哈希与边界见 [`source-index.md`](source-index.md) 和 [`glm-5.3-source-notes.md`](glm-5.3-source-notes.md)。
