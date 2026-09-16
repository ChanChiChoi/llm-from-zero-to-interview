# 2026 年模型候选盘点

> 2026-09-09 的候选表保持原样，作为历史发现快照。2026-09-14 的实时首页刷新与新增 13 个结构条目见 [`artificial-analysis-2026-09-14-snapshot.md`](artificial-analysis-2026-09-14-snapshot.md)；2026-09-15/16 对重点锚点的网络复验只更新对应核验段，不用新快照日期覆盖下表日期。

采集日期：2026-09-09。来源：[Artificial Analysis](https://artificialanalysis.ai/)。以下日期为榜单字段，尚未逐项经官方核验；不同推理档位保留独立行。此表为候选发现记录，不代表官方发布确认或技术结论。

补充说明：Anthropic 官方模型目录快照还核验到 Claude Haiku 4.5（2025-10-15 发布字段），但它未出现在本次 Artificial Analysis 候选表的采集切片中，因此不伪造榜单日期；该模型作为“官方目录发现”单独记录于 [`claude-haiku-4.5-source-notes.md`](claude-haiku-4.5-source-notes.md) 和来源索引。

补充说明：DeepSeek 官方 API 发布页快照还核验到 DeepSeek-R1-0528（2025/05/28），但它未出现在本次 2026 Artificial Analysis 候选表切片中；该模型作为“官方发布发现”单独记录于 [`deepseek-r1-0528-source-notes.md`](deepseek-r1-0528-source-notes.md)，不补写榜单日期或第三方分数。

## 2026-09-14 实时增量（Artificial Analysis）

以下是实时首页相对 2026-09-09 历史表的模型级增量。成对 slug 合并为一个 canonical 条目；本节不改写下方 273 条历史发现记录。除 DeepSeek V4.1-Flash、K2 Horizon MoVA 36B/A4B、Qwen3.8、Gemini 3.1 Pro Preview、Gemini 3.5 Flash、Gemini 3.7 Flash、Gemini 3.8 Flash、GPT-5.6、GPT-5.5、GPT-5.4、Claude Fable 5、Claude Opus 5、Claude Opus 4.8、Kimi K2.7 Code、Grok 4.5、Grok 4.6 和 GLM-5 外，新增项仍停留在候选发现层。

## 2026-09-16 GLM-5.2 官方核验增补

本节升级已经同时出现在 Artificial Analysis 与 DataCurve DeepSWE 的 GLM-5.2；Z.ai 官方文档用于核验和扩展周边技术，不作为新的模型发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5.2 | Artificial Analysis：`glm-5-2` 的 max/non-reasoning；DataCurve：`mini_swe_agent_glm_5_2_high`、`mini_swe_agent_glm_5_2_max` | [Z.ai GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2)、[文档索引](https://docs.z.ai/llms.txt) | 资料级闭环；1M/128K、长周期 Coding Agent、MCP、缓存和工作流证据已核验；参数、架构、SAO/compaction 原始定义、训练 recipe 和独立复现待核验 |

Artificial Analysis 2026-09-16 复验快照为 `3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；页面第三方 FAQ 记录约 1M context、Intelligence Index `34`、约 `72 tokens/s` 和 `$1.40/$4.40` 每百万 input/output token。DataCurve 快照为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；high/max 分别为 Pass@1 `36.2832%`/`43.7778%`、平均成本约 `$2.8355`/`$3.9199`、平均 Agent steps `121.88`/`129.13`。这些是配置 + `mini-swe-agent` + 工具/环境/verifier 的系统结果，不能写成裸模型能力。

Z.ai 文档快照为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。官方资料确认 1M context、128K max output、thinking、function calling、context caching、structured output、MCP，以及项目级代码库接管、跨文件重构和分阶段验证的长周期工程工作流；“lossless context”仍只按发布方描述记录。

当前不新增独立正式章节。完整快照、证据边界和待核验项见 [`glm-5.2-source-notes.md`](glm-5.2-source-notes.md)。

## 2026-09-16 GLM-5 官方核验增补

本节只升级 Artificial Analysis 已精确发现的 GLM-5；DataCurve 当前快照没有精确 `GLM-5` 行，不能把 GLM-5.2/5.3 的 DeepSWE 结果迁移给它。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5 | Artificial Analysis：`glm-5`，`GLM-5 (Reasoning)`，并列 `glm-5-non-reasoning`；DataCurve：无精确同名行 | [Z.ai 博客](https://z.ai/blog/glm-5)、[API 文档](https://docs.z.ai/guides/llm/glm-5)、[模型卡](https://huggingface.co/zai-org/GLM-5)、[技术报告](https://arxiv.org/abs/2602.15763)、[官方 GitHub](https://github.com/zai-org/GLM-5) | AA 单榜资料闭环；744B/40B、28.5T、DSA、`slime`、78 层/256 experts/top-8、202K position 和 Agentic Engineering 已核验；DataCurve 精确结果、完整 DSA/训练 recipe、生产 kernel 和独立复现待核验 |

Artificial Analysis 2026-09-16 复验快照为 `3,577,227` bytes，SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`。页面第三方字段显示约 `744B` total、`40B` active、`200K` context、约 `72.4 tokens/s`、约 `1.34s` TTFT；这些字段与官方模型卡交叉一致的规格才进入事实记录，指数/价格/速度/TTFT仍是第三方配置/provider 测量。

DataCurve 2026-09-16 复验快照为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；页面有 `mini_swe_agent_glm_5_2_*`、`mini_swe_agent_glm_5_3_*` 和 `mini_swe_agent_glm_5_3_flash_max`，没有 `mini_swe_agent_glm_5_*`。不记录 GLM-5 的 DataCurve Pass@1、成本或 Agent steps。

Hugging Face 模型卡/配置与 GLM-5 专属技术报告支持 DSA、MoE、异步 RL 基础设施 `slime`、长周期 Agent RL 和 Agentic Engineering 作为研究主线。配置公开 `GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、前三层 dense、`index_topk=2048`、`q_lora_rank=2048`、`kv_lora_rank=512` 和 `max_position_embeddings=202752`。这些是实现字段，不等于完整训练 recipe。

完整证据、评测设置、负面检索和待核验项见 [`glm-5-source-notes.md`](glm-5-source-notes.md)。当前不新增独立正式章节，先映射到已有 MoE、稀疏注意力、RL、Agent harness 和评测分层章节。

## 2026-09-15 GLM-5.3-Flash 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 GLM-5.3-Flash；Z.ai 官方资料用于核验和扩展锚点周边技术，不是新的候选发现入口。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GLM-5.3-Flash | Artificial Analysis：`glm-5-3-flash`，`max`，页面 `releaseDate` `2026-08-26`；DataCurve：`mini_swe_agent_glm_5_3_flash_max`，`max` | [Z.ai 模型文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)、[Z.ai 官方博客](https://z.ai/blog/glm-5.3-flash)、[固定 revision 模型卡](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)、[配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json) | 内容专题闭环；320B/18B、45 层、34 linear/11 sparse、288 routed/top-8/1 shared、IndexPool、mHC、原生视觉 coding loop、EPD serving、thinking/tool streaming 已核验；完整 kernel、训练 recipe、硬件 profiling 和独立复现待核验 |

Artificial Analysis 当前主配置为 Intelligence Index `41.907366113455`、约 `114.22108687545 tokens/s`、约 `2.45458272199994s` TTFT、1M context、约 `$0.15/$0.50/$0.026` 每百万 token；DataCurve 为 284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、约 `$0.2409818562`/task、约 `72829.77` output tokens、约 `122.89` Agent steps。两组数字均绑定不同的 provider/configuration 或 `mini-swe-agent`/工具/环境/verifier，不能拼成裸模型排名。

官方配置支持 `Glm5NextForConditionalGeneration`、320B total/18B activated、45 层、前 3 层 dense MLP、34 `linear_attention`/11 `deepseek_sparse_attention`、288 routed experts、top-8、1 shared、1,048,576 position、`index_kpool=4`、`index_topk=2048`、`mhc=true`、`hc_mult=4` 和 20 次 Sinkhorn；视觉配置给出 24 层、448 image、patch 14、temporal patch 2、输出投影 4096。官方博客/文档还描述 30T multimodal corpus、visual self-judgment/test-time improvement、SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split、EPD 和约 3× serving 自报。

完整快照哈希、API 字段、评测边界和待核验项见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)；正式专题见第二十一册第 84 章 [`glm-5.3-flash混合注意力与视觉闭环.md`](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)。

## 2026-09-15 DeepSeek V4 Flash Vision 官方核验增补

本节只升级已出现在 Artificial Analysis 的 `deepseek-v4-flash-vision`；DataCurve DeepSWE 当前快照没有该行，不能把 `deepseek-v4-flash` 或 `deepseek-v4-pro` 的配置结果迁移给它。

| 基础模型/配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| DeepSeek V4 Flash Vision | Artificial Analysis：`deepseek-v4-flash-vision`，`max`，页面 `releaseDate` `2026-08-21`；DataCurve：无同名行 | [V4 Flash Vision Exp 公告](https://api-docs.deepseek.com/news/news260821)、[Vision guide](https://api-docs.deepseek.com/guides/vision)、[Files API](https://api-docs.deepseek.com/guides/files_api)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[当前价格页](https://api-docs.deepseek.com/quick_start/pricing) | 资料级闭环；历史实验多模态 API、当前 alias 路由、图像 detail/token、Files `file_id`、Responses 图像工具回灌和成本/评测边界已核验；专属视觉架构、训练 recipe、独立技术报告和 DataCurve 结果待核验 |

Artificial Analysis 当前字段为 Intelligence Index `35.0122378035969`、约 `215.179167697513 tokens/s`、约 `1.29855545700002s` TTFT、1M context、284/13 目录参数和约 `$0.44/$1.32/$0.014` 每百万 token；均绑定第三方页面配置、provider 和测量时间。2026-09-15 官方 Quick Start 说明旧的 `deepseek-v4-flash-vision-exp` alias 由 `DeepSeek-V4.1-Flash` 服务；当前服务价格与历史 AA 字段不能直接拼接。

完整快照哈希和证据分层见 [`deepseek-v4-flash-vision-source-notes.md`](deepseek-v4-flash-vision-source-notes.md)；正式专题见第二十一册第 85 章 [`deepseek-v4-flash-vision多模态api与路由账本.md`](../../book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)。

## 2026-09-14 Qwen3.8 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 Qwen3.8 条目；官方资料用于核验和扩展技术，不是新的候选发现入口。

| 基础模型/服务 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Qwen3.8-27B | Artificial Analysis，2026-08-14 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B dense native vision-language；64 层；3 GDN + 1 Gated Attention；thinking 可按请求关闭 |
| Qwen3.8-2.4T-A95B | Artificial Analysis，2026-08-12 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 2.4T total/95B active；92 层；512 experts，10 routed + 1 shared；text-only；thinking 强制开启 |
| Qwen3.8-Flash-Next | Artificial Analysis，2026-08-26 | [模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf) | 125B/6B active；51B N-gram；GDN + QSA；四分支 GR；Muon/AdamW；实验性开源架构预览 |
| Qwen3.8 Max | Artificial Analysis，2026-08-03；DataCurve DeepSWE 当前快照亦有 `qwen3.8-max` | [Qwen Cloud](https://www.qwencloud.com/models/qwen3.8-max)、A95B 模型卡 | 托管版本；官方说明基于 A95B，并增加视觉、非 thinking、默认 1M 和内置工具等产品能力，不当作独立 open checkpoint |

具体页面、快照哈希、来源优先级和待核验项见 [`qwen3.8-source-notes.md`](qwen3.8-source-notes.md)。

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
| Gemini 3.8 Flash | Artificial Analysis：2026-09-02 的 high/medium/low；DataCurve DeepSWE v1.1：high | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)、[发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/)、[评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf) | GA/stable；1M 输入、64K 输出、多模态输入、thinking、工具组合与长任务 Agent 文档已核验；3.8 独立架构/训练报告待核验 |

具体来源、快照哈希、Interactions API、thought signature、Computer Use、File Search、URL Context、Code Execution 和评测边界见 [`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。

## 2026-09-14 GPT-5.6 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.6 条目；OpenAI 官方资料用于核验和扩展 reasoning、缓存、工具与 Agent 运行时，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.6 Sol | Artificial Analysis，2026-07-09 的多档 effort；DataCurve DeepSWE v1.1：`max` | [Sol 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-sol.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) | 旗舰档；1.05M context、922K maximum input、128K maximum output、`standard/pro`、`all_turns` 和工具支持已核验 |
| GPT-5.6 Terra | Artificial Analysis，2026-07-09 的多档 effort；DataCurve 当前快照无主表行 | [Terra 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md) | 智能/成本平衡档；共同 GPT-5.6 接口字段已核验 |
| GPT-5.6 Luna | Artificial Analysis，2026-07-09 的多档 effort；DataCurve DeepSWE v1.1：`max` | [Luna 模型页](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) | 高并发/成本敏感档；共同 GPT-5.6 接口字段已核验 |

推理配置仍按一个 GPT-5.6 家族归并；`Non-reasoning` 是榜单标签，不自动等于 API 的 `none`。完整的 persisted reasoning、prompt caching、compaction、tool search、harness 和负面证据见 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## 2026-09-15 GPT-5.5 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.5 条目；OpenAI 官方资料用于核验和扩展推理、视觉、工具、状态和缓存技术，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.5 | Artificial Analysis，2026-04-23 的多档 effort；DataCurve DeepSWE v1.1：`xhigh` | [模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md)、[GPT-5.5 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.5.md) | 旗舰档；1.05M context、128K max output、`none`--`xhigh`、Responses/工具/`phase`/图像 detail/缓存差异已核验 |
| GPT-5.5 Pro | Artificial Analysis，2026-04-23 的 `xhigh`；DataCurve 当前快照无主表行 | [模型页](https://developers.openai.com/api/docs/models/gpt-5.5-pro.md) | 使用更多计算；Responses/Batch、`medium`--`xhigh`、background mode 和无 cached input discount 已核验 |

GPT-5.5/Pro 按同一产品家族归并，但 `gpt-5.5-pro` 的端点、effort、价格和缓存计费不能直接套用基础 `gpt-5.5`。Artificial Analysis 的 `GPT-5.5 Instant` 只保留为榜单级关联条目；它没有进入本次官方 API 核验结论。完整的 `phase` 回放、outcome-first prompting、tool search、compaction、5.5/5.6 缓存对比和负面证据见 [`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。

## 2026-09-15 GPT-5.4 官方核验增补

本节只升级已经出现在 Artificial Analysis/DeepSWE 的 GPT-5.4 条目；OpenAI 官方资料用于核验和扩展推理、工具、长上下文、状态和 compaction 技术，不是新的候选发现入口。

| 基础模型/服务档位 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| GPT-5.4 | Artificial Analysis：2026-03-05 的 `xhigh`、`low`、`Non-reasoning`；DataCurve DeepSWE v1.1：`xhigh` | [模型页](https://developers.openai.com/api/docs/models/gpt-5.4.md)、[GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 资料级闭环；1.05M context、128K max output、`none`--`xhigh`、tool search、computer use、custom tools/CFG、`phase`、compaction 和 Responses 状态已核验 |
| GPT-5.4 Pro | Artificial Analysis：2026-03-05 的 `xhigh`；DataCurve 当前快照无主表行 | [GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 与 GPT-5.4 同家族的关联服务档位；本轮未单独建立 Pro 资料闭环 |
| GPT-5.4 mini/nano | Artificial Analysis：2026-03-17 的多档配置；DataCurve 当前快照无对应主表行 | [GPT-5.4 指南](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) | 关联变体/候选；未单独核验模型页、snapshot 和独立 benchmark |

Artificial Analysis 主配置 `GPT-5.4 (xhigh)` 当前页面标记 deprecated 并指向 GPT-5.5，第三方字段为 Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT 和 1M context；DataCurve GPT-5.4 xhigh 为 234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本约 `$5.6525`、平均输出约 `71,408.87` token、约 `70.47` Agent steps。两榜单数字绑定不同配置、provider、harness、任务集、工具、环境和 verifier，不能互相拼接为裸模型排名。

完整证据、快照哈希、面试主线和未公开内容见 [`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md)。

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
| Claude Fable 5.1 | Artificial Analysis：`max + Default Fallback`，并有 `xhigh/high/medium/low + Default Fallback`；内嵌 `releaseDate` 为 2026-09-01 | [Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5-1)、[Anthropic 发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)、[模型页](https://platform.claude.com/docs/en/models/fable-5-1/overview)、[System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) | 资料级闭环；1M context、128K output、adaptive always-on、默认 high、Fable/Mythos safeguards 分层、cache read `$0.25/M` 和发布方 benchmark/安全摘要已核验 |

2026-09-15 Artificial Analysis 主配置字段为 Intelligence Index `53.3738`、输出速度约 `65.98 tokens/s`、TTFT 约 `212.01s`、约 `$7.63`/Intelligence Index task；所有数值均绑定 `max + fallback`、第三方任务、provider 和测量时间。DataCurve 快照仍为 113 tasks/91 repositories/5 languages/`mini-swe-agent`，只有 `claude-fable-5` 的各 effort 行。

Anthropic 发布页新增的面试线索包括：Fable 5.1 与 Mythos 5.1 共享 underlying model 但 safeguards 不同；Fable 5.1 在 Claude Code 默认 High、Claude Cowork/Claude.ai 默认 Medium；cache reads 为 `$0.25/M`；EFS 以客户云基础设施支持零数据保留语义；anti-distillation 通过限制新账户编辑历史同时保留 prior thinking transcript；以及模型在科学工作流中的自定义 GPU kernel 与中间结果缓存案例。上述能力和数字均标明为发布方自述，不升级为内部架构或训练配方。

arXiv 标题精确检索返回 0 篇，全文检索返回 4 篇外部使用/评测论文，详见 [`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。当前没有 Fable 5.1 专属参数/架构披露、完整训练报告、可独立复现技术报告或 DataCurve 结果；不新增独立架构章节。下一锚点切换为 `claude-sonnet-5`。

## 2026-09-15 Claude Sonnet 5 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Sonnet 5；Anthropic 官方资料用于核验模型身份、API/Agent 运行时和发布方评测，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Sonnet 5 | Artificial Analysis：`max` 主配置及 `xhigh/high/medium/low/Non-reasoning` 变体；DataCurve DeepSWE v1.1：五档 effort + `mini-swe-agent` | [Artificial Analysis](https://artificialanalysis.ai/models/claude-sonnet-5)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Sonnet 5 发布公告](https://www.anthropic.com/news/claude-sonnet-5)、[System Card](https://www.anthropic.com/claude-sonnet-5-system-card)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) | 资料级闭环；`claude-sonnet-5`、2026-06-30、1M context、128K/300K 输出、Adaptive、默认 high、配置级榜单数据、thinking/tool block 协议、tokenizer、context awareness、compaction 和 programmatic tool calling 已核验；参数、架构、完整训练/后训练 recipe 和独立复现待核验 |

Artificial Analysis 主配置快照为 Intelligence Index `38.3576962882576`、约 80.0022 output tokens/s、约 202.6275s TTFT、输入/输出 `$2/$10/M`、约 `$5.0912`/task；DataCurve 五档 Pass@1 从 `max` 到 `low` 为 53.846%/49.667%/48.230%/39.778%/30.512%，平均成本为 `$26.40/$11.89/$7.43/$4.08/$2.19`。两榜单数字分别绑定第三方配置、provider、任务集、harness、工具、环境和 verifier，不能拼接为裸模型排名。

Anthropic 官方资料还确认 adaptive thinking 不接受手动 `thinking.type: "enabled"` + `budget_tokens`，effort 是行为信号而不是严格 token 预算，`max_tokens` 是共享硬上限；thinking blocks/signatures 需要按异构 `content` 回放；新 tokenizer 对相同文本约增加 30% token；Sonnet 5 支持 context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling。System Card/发布方评测中的 SWE-bench Verified 85.2%、Terminal-Bench 2.1 80.4% 等数字单独保留为发布方证据。

arXiv 精确标题检索截至 2026-09-15 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告。当前不支持参数规模、稠密/MoE 架构、完整训练/后训练配方、内部 adaptive-thinking 机制或独立 benchmark 复现结论；因此不新增 Sonnet 5 专属架构章节。具体快照哈希、完整配置表和面试主线见 [`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

## 2026-09-16 Claude Sonnet 4.6 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Sonnet 4.6；Anthropic 官方资料用于核验模型身份和扩展运行时技术，不是新的模型发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Sonnet 4.6 | Artificial Analysis：`claude-sonnet-4-6-adaptive` 等配置；DataCurve DeepSWE v1.1：`mini_swe_agent_claude_sonnet_4_6_high` | [Artificial Analysis](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive)、[DataCurve DeepSWE](https://deepswe.datacurve.ai/)、[Sonnet 4.6 发布公告](https://www.anthropic.com/news/claude-sonnet-4-6)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[System Card](https://www.anthropic.com/claude-sonnet-4-6-system-card) | 资料级闭环；`claude-sonnet-4-6`、2026-02-17、1M context（发布时 beta）、adaptive/extended thinking、compaction、tool search、computer use 和 Agent 工具链已核验；参数、架构、完整训练/后训练 recipe 和独立复现待核验 |

Artificial Analysis adaptive 快照为 `3,626,831` bytes，SHA-256 `5feda773b5ccefaaad7265b5f2d2401551d4d96e32cfa524a7fedd550db5b0de`；DataCurve 结果为 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%`、平均成本约 `$5.5224`、平均输出 `76,160.31` token、平均 Agent steps `133.66`。这些是第三方配置与 `mini-swe-agent` 系统结果，不能拼成裸模型分数。

Anthropic 发布页确认 coding、computer use、long-context reasoning、agent planning、knowledge work 和 design 定位，以及 1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory 和 programmatic tool calling。Computer use 的网页 prompt injection 风险和官方“较 Sonnet 4.5 改善”的表述按发布方证据记录，不写成风险已解决。

2026-09-16 三代理复验中，Artificial Analysis 首页快照均为 `1,773,553` bytes、SHA-256 `60e491dc5eccaf2d032205a7380f30a24b2726583b5a9cfaccfe14c17d736476`；DataCurve 均为 `268,313` bytes、SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。 `8098/1234` 的模型目录请求进入区域不可用页，`7890` 取得真实目录并确认 Sonnet 4.6 条目；这属于代理线路差异，不是模型不存在。

完整来源、面试主线、负面论文检索和待核验项见 [`claude-sonnet-4.6-source-notes.md`](claude-sonnet-4.6-source-notes.md)。暂无 Sonnet 4.6 独立正式架构章节。

## 2026-09-15 Claude Opus 4.8 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Claude Opus 4.8；Anthropic 官方资料用于核验模型身份和扩展动态工作流、effort 与长任务可靠性，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Claude Opus 4.8 | Artificial Analysis：2026-05-28 的 `max`；DataCurve DeepSWE v1.1：`xhigh` 与 `max` + `mini-swe-agent` | [发布公告](https://www.anthropic.com/news/claude-opus-4-8)、[System Card](https://www.anthropic.com/claude-opus-4-8-system-card)、[Dynamic Workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) | 2026-05-28 发布；默认 `high`、支持更高 effort、Messages API system entries、动态编排/并行 subagents/独立验证已核验；参数、架构、训练配方和独立技术报告待核验 |

Artificial Analysis 当前详情页将 Opus 4.8 标为 deprecated，并指向 `claude-opus-5`；详情页与 DeepSWE 的分数、速度、成本和 Agent steps 仍只作为带配置条件的历史评测记录。完整来源、快照哈希、面试技术主线和未公开内容见 [`claude-opus-4.8-source-notes.md`](claude-opus-4.8-source-notes.md)。

## 2026-09-15 Kimi K2.7 Code 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Kimi K2.7 Code；Moonshot/Kimi 官方资料用于核验模型身份、架构摘要、推理协议和部署边界，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Kimi K2.7 Code | Artificial Analysis：2026-06-12 的 `kimi-k2-7-code`；DataCurve DeepSWE v1.1：`mini-swe-agent`、`reasoning_effort:null` | [Kimi 资源页](https://www.kimi.ai/resources/kimi-k2-7-code)、[API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)、[固定 revision 模型卡](https://huggingface.co/moonshotai/Kimi-K2.7-Code/tree/74797c9c62378b951a1f6fcf5c4631024e9b8bef)、[配置](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json) | 资料级闭环；1T/32B active MoE、MLA、256K/262,144 context、MoonViT 400M、native INT4、always-on thinking、preserve_thinking 和工具协议已核验；专属技术报告、完整训练/后训练配方、线上接受率和独立 benchmark 待核验 |

具体来源、快照哈希、benchmark 及负面论文检索证据见 [`kimi-k2.7-code-source-notes.md`](kimi-k2.7-code-source-notes.md)。

## 2026-09-15 Grok 4.6 官方核验增补

本节只升级已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Grok 4.6；xAI 官方资料用于核验和扩展周边技术，不是新的候选发现入口。

| 基础模型/服务配置 | 榜单发现 | 官方证据 | 当前状态 |
|---|---|---|---|
| Grok 4.6 | Artificial Analysis：2026-08-12 的 `high/medium/low/xhigh`；DataCurve DeepSWE v1.1：四档 `mini_swe_agent_grok_4_6_*` | [xAI 发布公告](https://x.ai/news/grok-4-6)、[官方模型页](https://docs.x.ai/developers/grok-4-6)、[Markdown 版本](https://docs.x.ai/developers/grok-4-6.md)、[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)、[工具文档](https://docs.x.ai/developers/tools/overview) | 资料级闭环；2026-08-12 官方发布日期、500K context、reasoning/opaque state、compaction、function calling、Web/X Search、Code Execution、Remote MCP 和发布方训练/评测描述已核验；参数、架构、完整训练配方、专属技术报告和独立 benchmark 待核验 |

Artificial Analysis 详情页约为 44.405 Intelligence Index、58.50 tokens/s 和 40.86s TTFT；DataCurve 四档 Pass@1 为 low 41.648%、medium 67.478%、high 65.188%、xhigh 66.741%，平均成本约 `$1.0424/$3.4490/$4.3849/$5.4977`。这些数字分别绑定榜单配置、任务集、provider、harness、工具和 verifier，不能拼接为裸模型分数。

xAI 发布页公开了比 Grok 4.5 更长的 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成的多 effort/多 harness SFT trajectories、model-based checks，以及面向知识工作、编码和领域环境的 agentic RL；长任务中还强调 self-testing/verification。这些是发布方披露，不等于参数、网络结构、具体 optimizer、RL objective 或完整训练 recipe。

具体来源、快照哈希、四档 DataCurve 原始配置、面试主线、论文检索负面证据和未公开内容见 [`grok-4.6-source-notes.md`](grok-4.6-source-notes.md)。

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
