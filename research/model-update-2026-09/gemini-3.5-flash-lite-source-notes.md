# Gemini 3.5 Flash-Lite：排行榜锚点与官方资料摘记

原始核验日期：2026-09-20；当前时点联网复验：2026-09-22。本文只记录已经出现在 Artificial Analysis 的 `Gemini 3.5 Flash-Lite`，再沿 Google AI Developers 与 Google DeepMind 的官方资料扩展面试相关技术。DataCurve 当前没有该 Lite 变体的精确 Agent 行，因此不把 Gemini 3.5 Flash、3.6 Flash 或其他 Gemini 版本的 DeepSWE 结果迁移过来。

## 1. 锚点和证据边界

- Artificial Analysis 详情页：[Gemini 3.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)。2026-09-20 快照 `/tmp/gemini35lite-aa-20260920.html` 为 `3,845,070` bytes，SHA-256 为 `7de22f529d3b4e2dd440ef8a8e9447480e18dde08437fc508ece06767571e8ef`。
- AA 页面中的第三方字段包括 `releaseDate: 2026-07-21`、`isReasoning: true` 和 Intelligence Index `22.1685424839812`。这些字段用于确认榜单身份和配置索引，不替代 Google 官方发布日期、模型结构或训练报告。
- DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 快照 `/tmp/deepswe-gemini35lite-20260920.html` 为 `268,571` bytes，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面当前检索到 `mini_swe_agent_gemini_3_5_flash_high` 等相邻条目，但没有精确的 `mini_swe_agent_gemini_3_5_flash_lite_*`；因此不迁移相邻模型的 Pass@1、成本或 Agent steps。
- 这满足“重点厂商 + 两个排行榜之一”的 AA 单榜候选条件，当前状态为“资料级闭环（AA 单榜）”，不是双榜 Agent 评测闭环。

### 1.1 2026-09-22 当前时点新鲜快照

本轮重新获取同一个 `gemini-3.5-flash-lite` 锚点；没有从 Google 官方目录或其他页面另发现模型。Artificial Analysis 详情页 `/tmp/aa-gemini35lite-fresh-20260922.out` 为 `3,858,652` bytes，SHA-256 为 `bbe13cb52c85772080a105b1ec98c42bd71e7194dd67abbc26cb52f9ba1115d`。当前第三方/provider 字段为 `releaseDate=2026-07-21`、`isReasoning=true`、effort `high`、Intelligence Index `22.1685424839812`、median output speed `386.373295824977 tokens/s`、median time to first chunk `11.2053542005s`、1M context、cost per Intelligence Index task `$0.12352689651593446`，输入/输出价格 `$0.30/$2.50`，cache-hit `$0.03`。这些字段用于记录榜单配置和采集时点，不能当作 Google 的参数规模或训练披露。

DataCurve 当前时点页面快照为 `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，页面生成时间为 `2026-09-22T06:27:15.860279+00:00`；仍没有精确的 `mini_swe_agent_gemini_3_5_flash_lite_*` 行。页面中的 `gemini_3_5_flash_high` 是相邻 Flash 变体，不能迁移其 Pass@1、成本、输出 token 或 Agent steps。

本轮官方页面新鲜快照包括：Google API 模型页 `/tmp/google-model-fresh-20260922.out` 为 `106,715` bytes、SHA-256 `052846d394f282870f2334d4c079947b45cb3f83faa39de63eb338f78ca5a4a9`；Thinking 文档 `/tmp/google-thinking-fresh-20260922.out` 为 `226,283` bytes、SHA-256 `ab9d354068333aedd8199c5afafd23d36e6b0c6677df531de4d8804b48833951`；Video understanding 文档 `/tmp/google-video-fresh-20260922.out` 为 `262,508` bytes、SHA-256 `7420a3a0b4baf5a6667371167c86fe8f178a339c9eb6bd1e5108daa789b4125e`。它们继续支持 1,048,576 输入 token、65,536 输出 token、`minimal/low/medium/high`、agentic video 和 `processing_call/result` 的既有记录。

本轮只更新榜单测量和官方页面的新鲜证据，不把 Google 3.1 Flash-Lite Model Card 的架构/训练/软硬件指向改写成 3.5 独有发明；当前状态仍是**资料级闭环（AA 单榜）**，不新增独立 Transformer 正式章节。

## 2. Google API 模型页

主要来源是 [Gemini 3.5 Flash-Lite API model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)。2026-09-20 快照 `/tmp/gemini35lite-google-model-20260920.html` 为 `106,715` bytes，SHA-256 为 `7735a99586dd72f483b0a37a68727cdc386ef577c95c6a8b18acb94eb3a5e800`。

页面直接给出以下接口契约：

| 字段 | 已确认内容 |
|---|---|
| Model code | `gemini-3.5-flash-lite` |
| 产品定位 | 低延迟、低成本、高吞吐；面向 subagent、文档解析、高量级 agentic workflow、简单抽取和分类 |
| 输入 | 文本、图像、视频、音频、PDF |
| 输出 | 文本 |
| 输入上限 | `1,048,576` tokens |
| 输出上限 | `65,536` tokens |
| 能力 | caching、code execution、Computer Use Preview、File Search、function calling、Maps/Search grounding、structured outputs、thinking、URL context |
| 不支持项 | audio generation、image generation、Live API |
| 版本 | stable：`gemini-3.5-flash-lite` |
| 页面更新时间 | July 2026 |

这里的 1M 是 API 输入容量，不是“每个位置都能被同等利用”的保证。面试时应把输入 token、thinking token、工具结果、视频处理 token、缓存命中、TTFT、TPOT 和有效检索率分栏记录。

## 3. Model Card：Lite 自身没有独立架构披露

主要来源是 [Gemini 3.5 Flash-Lite Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/)。HTML 快照 `/tmp/gemini35lite-deepmind-card-20260920.html` 为 `154,601` bytes，SHA-256 为 `18efbacc7e078f438db2d86053282062f735fe6d92d313601ad187e42df896ac`；官方 PDF [Gemini 3.5 Flash-Lite Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Lite-Model-Card.pdf) 为 `286,833` bytes，SHA-256 为 `d4d8e79d69377a5d1add81b7833197f3e84dd6ebe983b5d5a39f1530e2052571`。

Model Card 的关键边界是：

- 描述：Gemini 3 系列中的原生多模态 reasoning model，强调 translation、classification、高吞吐和 latency-sensitive agentic workflow。
- 依赖关系：`Gemini 3.5 Flash-Lite is based on Gemini 3.1 Flash-Lite`。
- architecture、training dataset、training data processing、hardware 和 software 都指向 [Gemini 3.1 Flash-Lite Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Flash-Lite-Model-Card.pdf)，该 PDF 本轮下载为 `331,126` bytes，SHA-256 为 `72730fbcf1b1864afa8245f53644173da025a6f192a77cd6f9fcb33353982d56`。
- 因此不能把 Gemini 3.1 Flash-Lite 的架构或训练资料改写成 3.5 Flash-Lite 的新发明；也不能从 `Flash-Lite` 这个产品名推断稠密/MoE、层数、参数量、注意力变体或 optimizer。
- 输入是文本、图像、音频、视频，context window up to 1M；输出为文本，64K token。API 页面给出的 `1,048,576`/`65,536` 是更精确的接口上限。

### 3.1 发布方评测与价格

Model Card 的结果表标注为 July 2026 的发布方评测，价格和 benchmark 必须绑定 Google 的评测设置，不能与 AA 或 DataCurve 拼接成一个总分：

| 项目 | Gemini 3.5 Flash-Lite |
|---|---:|
| Input price（无缓存，$/1M tokens） | `$0.30` |
| Output price（$/1M tokens） | `$2.50` |
| SWE-Bench Pro (Public) | `54.2%` |
| Terminal-Bench 2.1（Terminus-2 harness） | `54.0%` |
| MLE-Bench | `39.2%` |
| GDPVal-AA v2（Elo） | `1140` |
| OSWorld-Verified | `74.0%` |
| CharXiv Reasoning，无工具/有工具 | `74.5%` / `76.5%` |
| GDM-MRCR v2，128K/1M | `72.2%` / `21.3%` |

结果表与价格表支持“低成本、高吞吐模型也能进入 coding/Agent/长上下文工作流”的面试讨论，但不支持推导内部训练 recipe。`1M` 的 GDM-MRCR 结果还提醒：接口可装入长上下文，不等于在百万 token 上保持 128K 区间的召回质量。

### 3.2 安全与限制

Model Card 记录 hallucination、偶发 slow/timeout、jailbreak resistance 持续改进和 Frontier Safety mitigation。知识截止为 2026 年 3 月，但部分领域仍可能只有 2025 年 1 月的知识覆盖。安全表是自动评测与发布方人工 red teaming 的结果；Frontier Safety 结论借用了 Gemini 3.1 Pro 的评估，不应写成 3.5 Flash-Lite 的独立安全训练算法。

## 4. Thinking：默认 minimal 是运行时配置，不是新权重

Google [Thinking](https://ai.google.dev/gemini-api/docs/thinking) 文档快照 `/tmp/gemini-thinking-7890-20260920.html` 为 `226,287` bytes，SHA-256 为 `8b2ffcf69f00ea2a088200d71bfa1329e1e449062a2944b599f3bd67939af8ee`。模型表列出：

- `gemini-3.5-flash-lite` 默认 `On (minimal)`；支持 `minimal`、`low`、`medium`、`high`。
- 文档称 Gemini models 默认进行 dynamic thinking，会按请求复杂度调整 reasoning effort；`thinking_level` 是请求级控制字段。
- `minimal` 与“绝对关闭 reasoning”不是同一个概念；它表示尽量少的思考，复杂任务仍可能产生少量 reasoning。
- `low/medium/high` 改变的是单请求的质量、延迟、token 和成本曲线，不是三个 checkpoint。比较时要固定 prompt、工具、输出上限、缓存和 verifier。

## 5. Agentic video：按需读取时间轴

Google [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) 快照 `/tmp/gemini-video-7890-20260920.html` 为 `262,504` bytes，SHA-256 为 `7f58e353ad4b3753b6db639430f1aad76372c4fc37ebe5c26f4ac286bf40ede6`。文档明确把 Gemini 3.5 Flash-Lite 列入 agentic video understanding 支持列表：

1. Static 模式默认以约 1 FPS 抽取帧，一次性把媒体放进上下文；适合短视频或要求全片固定采样的任务。
2. Agentic 模式由模型动态探索时间轴，按 transcript、帧或音频按需加载，并在需要时调整采样率和分辨率；它不是简单地把所有视频帧塞进 1M context。
3. 文档发布方表述为长视频场景最多约 88% 更高 token efficiency、约 7% quality 提升；这是文档的总体产品结果，不是每个视频请求的保证。
4. Agentic processing 可通过交互步骤观察：`processing_call` 表示模型请求加载片段或 transcript，`processing_result` 通过 `call_id` 关联加载结果。模型选择内容不等于宿主已授权文件、网络或执行器权限。

这个主线很适合作为面试题：当视频证据稀疏且查询只指向局部时间段时，动态读取减少 token 和无关噪声，但增加了规划步骤、额外延迟、失败重试和可审计状态；应同时评估证据召回、处理调用次数、总 token、TTFT、端到端成功率和单位成功成本。

## 6. 3.5 Flash 的 What's New 不能回写到 Lite

本轮也重新获取了正确的 [What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5) 页面：`201,303` bytes，SHA-256 `749446da96b9102bf31c669e55aae05efc7d9579431167cf3f2950c200dbfc02`。页面标题和正文都是 `Gemini 3.5 Flash`，主要讲 GA、默认 thinking 从 `high` 改为 `medium`、thought preservation 和 agentic/coding performance；它不是 Flash-Lite 的发布页。

因此本笔记只把该页面作为 Gemini 3.5 系列背景证据，不把 Flash 的默认 `medium`、GA 语句或 `thought preservation` 自动归因给 Lite。Lite 的默认 thinking 以 Thinking 文档的 `On (minimal)` 表为准，Lite 的能力与限制以自身 API Model Card 为准。

## 7. 论文、技术报告与可写入书系的知识点

截至本轮，未找到由 Google 针对 Gemini 3.5 Flash-Lite 单独公开的架构论文或完整训练报告。应保留以下可迁移的面试知识，而不是编造模型内部创新：

- **模型身份分层**：AA 的 `releaseDate`、`isReasoning` 和 Intelligence Index 是第三方目录字段；`gemini-3.5-flash-lite` 是官方 API model code；Model Card 的 `based on Gemini 3.1 Flash-Lite` 是公开依赖关系。
- **推理预算分层**：`thinking_level` 是单请求 effort 控制；它影响输出质量、reasoning token、延迟和成本，但不等于独立权重或严格 token 上限。
- **视频证据分层**：static 是固定采样，agentic 是模型驱动的时间轴检索；`processing_call/result` 是可审计的处理事件，不是模型拥有执行权限的证明。
- **长上下文评测**：context window、缓存、视频 token、needle recall 和端到端任务成功率必须分开测；1M 输入上限不等于 1M token 的均匀有效记忆。
- **模型卡阅读**：当新版本把架构、训练数据和硬件指向前代 Model Card 时，只能记录版本依赖，不能把前代内容包装成新版本发明。

## 8. 当前状态与待核验项

Gemini 3.5 Flash-Lite 当前为**资料级闭环（AA 单榜）**：已具备 AA 候选、官方 API Model Page、DeepMind Model Card、Model Card PDF、Thinking/Video 官方文档、发布方评测和安全边界。暂不新增独立 Transformer 正式章节，映射到已有 Reasoning、Agent/工具协议、多模态视频、长上下文/Serving、评测与安全章节。

仍待核验：3.5 Flash-Lite 独有的参数规模、层/专家/注意力结构、完整训练与后训练 recipe、独立技术报告、生产 video/attention kernel、目标硬件 profiling、线上 agent/tool acceptance rate、DataCurve 精确 Agent 行和独立 benchmark 复现。3.1 Flash-Lite Model Card 只作为官方依赖入口，不填补这些 3.5 独有空白。
