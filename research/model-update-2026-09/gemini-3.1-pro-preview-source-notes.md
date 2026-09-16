# Gemini 3.1 Pro Preview：推理、工具协议与长上下文摘记

核验日期：2026-09-16。本文只把已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `Gemini 3.1 Pro Preview` 作为模型锚点；Google 官方模型页、Model Card、API 文档和评测方法用于核验该锚点及扩展面试知识，不作为新的模型发现入口。

## 1. 两个排行榜中的锚点

### Artificial Analysis

2026-09-16 通过用户提供的三条代理重新抓取 [Gemini 3.1 Pro Preview 详情页](https://artificialanalysis.ai/models/gemini-3-1-pro-preview)。三份完整 HTML 字节一致，页面结构化字段确认：

| 字段 | 页面值 | 证据边界 |
|---|---:|---|
| canonical slug | `gemini-3-1-pro-preview` | Artificial Analysis 目录条目 |
| releaseDate | `2026-02-19` | 第三方目录字段，不替代 Google 发布日期 |
| reasoning | `true` | 页面配置标签，不等于内部推理实现披露 |
| effort | 未拆成多条基础模型记录；详情页为 Preview 配置 | 运行配置与模型身份要分开 |
| open weights | `false` | 第三方开放性字段 |
| parameters | `null` | Google 未公开参数规模；不能由 context 或指数反推 |
| context window | `1,000,000` tokens | 配置上下文上限，不等于所有任务都能无损召回 |
| Intelligence Index | `30.3596656132261` | Artificial Analysis 自有评测协议的配置级结果 |
| median output speed | `108.405187277136 tokens/s` | 页面称基于 Google API 的测量，受时间/provider/输入影响 |
| median TTFT | `24.6087743645001s` | 第三方测量字段，不是模型理论延迟 |
| API input/output price | `$2` / `$12` per 1M tokens | 页面引用的 provider/API 价格，可能随时间变化 |
| cache hit price | `$0.20` per 1M tokens | 页面 pricing dataset 字段 |

Artificial Analysis 的详情页还称该模型可以通过 3 个 API provider 获取。该页的 FAQ/JSON-LD 把上面的速度、成本、输入模态、1M context 和 Intelligence Index 作为页面结论；它们只能作为榜单快照引用，不能与 DataCurve 的 Agent 结果拼成一个“裸模型分数”。

### DataCurve DeepSWE

2026-09-16 通过三个代理重新抓取 [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/)。三个 HTML 快照均为 `268,313` bytes，SHA-256 相同；当前页面仍为 113 tasks、91 repositories、5 languages、4 runs，统一 harness 为 `mini-swe-agent`。

原始配置行为：

| 字段 | 值 |
|---|---:|
| model | `gemini-3-1-pro-preview` |
| harness | `mini-swe-agent` |
| reasoning_effort | `high` |
| config | `mini_swe_agent_gemini_3_1_pro_preview_high` |
| passed / attempted | `53 / 452` |
| Pass@1 | `11.72566371681416%` |
| Pass@4 | `28.31858490566046%` |
| mean cost | `$2.1434045568888886` |
| mean output tokens | `28,368.8844` |
| mean agent steps | `75.5644` |
| mean duration | `867.7979s` |

这些数字绑定 `mini-swe-agent`、工具、任务仓库、执行环境、重复运行和 verifier。尤其不能把 `11.73%` 写成 Gemini 3.1 Pro 在所有 SWE-Bench 或真实工程场景的裸能力，也不能用它解释 Artificial Analysis 的 30.36 Intelligence Index。

### 本轮断点恢复快照

| 页面 | 临时快照 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `/tmp/overnight-aa-home-1234-20260916.html` | `1,773,522` | `934084fafb5dd020e1d015db5b141b78837cdd29449b3dff4478628de1df2321` |
| Artificial Analysis Gemini 3.5 详情 | `/tmp/overnight-aa-g35-1234-20260916.html` | `3,615,449` | `c2c4f7a6286877cd844aaa7b67ebc413b45a2f96c7bb1e6aaf5239873e8b6173` |
| Artificial Analysis Gemini 3.1 详情 | `/tmp/overnight-aa-g31-1234-20260916.html` | `3,607,783` | `ba842f82908f6237acd84c1ed0d1ab03d4652af503c8de6d00b56bb21b7731eb` |
| DataCurve DeepSWE | `/tmp/overnight-deepswe-1234-20260916.html` | `268,313` | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |

三条代理对同一页面得到相同哈希；这证明本轮传输完整一致，不证明排行榜数据永久不变。

## 2. Google 官方模型信息

[Gemini 3.1 Pro API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)确认：

- 模型 ID 为 `gemini-3.1-pro-preview`，输入为 text、image、video、audio 和 PDF，输出为 text；
- 输入 token limit 为 `1,048,576`，输出 token limit 为 `65,536`；
- 支持 caching、code execution、function calling、Google Maps/Search grounding 和 structured outputs；File Search 在 AI Studio 可用；不支持 audio generation、image generation 和 Live API；
- 页面额外列出 `gemini-3.1-pro-preview-customtools` endpoint，针对 bash 与自定义工具混用场景优化工具优先级。官方同时提醒该 endpoint 在不需要这类工具的场景中可能出现质量波动；它是同一模型家族的产品入口，不能单独当作新基础模型。

Google API [Release Notes](https://ai.google.dev/gemini-api/docs/changelog) 的 2026-02-19 条目确认发布 Gemini 3.1 Pro Preview，并同时发布 `customtools` endpoint。Google DeepMind [Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/) 的发布时间为 2026-02-19，描述它是 Gemini 3 系列的原生多模态 reasoning 模型，能够处理 text、audio、image、video 和整个 code repository。

Model Card 的关键依赖关系是：

> Gemini 3.1 Pro is based on Gemini 3 Pro.

其 Architecture、Training Dataset、Training Data Processing、Hardware 和 Software 小节都把进一步资料指向 [Gemini 3 Pro Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf)。因此当前公开证据支持“3.1 是基于 Gemini 3 Pro 的迭代模型”，不支持独立的参数规模、层数、稠密/MoE 结构、注意力变体、优化器或完整训练/后训练 recipe。

## 3. 面试相关的新技术

### 3.1 可调 thinking，而不是把输出长度当作推理预算

[Thinking 文档](https://ai.google.dev/gemini-api/docs/thinking)把 reasoning effort 暴露为 `thinking_level`。对 Gemini 3 系列，常见档位为 `low`、`medium`、`high`；文档强调 thinking tokens 与输出 tokens 共同受到 output token limit 约束。需要更长最终答案时，不能只盯着 reasoning，而应同时留下足够的 output budget。

面试时可以这样区分：

1. `thinking_level` 是模型运行时对思考投入的控制信号；
2. `max output tokens` 是包含思考与答案的硬资源上限；
3. 低 thinking 可能降低复杂问题质量，高 thinking 又会增加 token、延迟和成本；
4. 官方建议对事实检索、分类等简单任务使用较低 thinking，对复杂规划和 agentic workflow 使用更高 thinking。

不能从这个 API 字段推断 Gemini 内部一定采用某个显式 CoT loss、树搜索或特定 RL 算法。公开资料只证明了可观察的推理控制接口。

### 3.2 Thought signature 与工具上下文循环

Gemini 3 工具协议的核心不是把所有中间文本重新拼进 prompt，而是保留完整响应中的 thought/tool 结构和加密 `signature`。Google [工具组合文档](https://ai.google.dev/gemini-api/docs/tool-combination)说明：

- Gemini 3 models use `tool context circulation`，让内置 server-side tools 的上下文能够暴露给同一次交互中的 custom function tools；
- `signature` 出现在 thought steps，以及 Gemini 3+ 的 tool call/result steps 中，用于跨 interaction 维护加密上下文；
- Stateful mode 使用 `previous_interaction_id`，服务端自动管理 `id` 与 `signature`；
- Stateless mode 由客户端手动回放历史，必须保留正确的 `id` 与 `signature`；SDK 在传入完整 response object 时可以代管这部分字段；
- 宿主仍负责执行 custom function、校验参数、授权、返回 function response 和审计结果，签名不能替代权限系统。

这是很好的 Agent 面试切入点：模型决定“调用什么”，宿主决定“是否允许及如何执行”，协议状态负责把上一轮 reasoning/tool context 可靠地带入下一轮。若只保存最终文本而丢掉 thought/tool signature，就可能造成上下文断裂、重复规划或工具链无法继续。

[Thought signatures 页面](https://ai.google.dev/gemini-api/docs/thought-signatures)本轮返回迁移提示，已将正文移动到 Thinking guide 的 signatures 小节；不能把迁移页误认为新的独立协议。

### 3.3 `customtools` 是工具选择偏置，不是新权重

官方模型页专门给出 `gemini-3.1-pro-preview-customtools`。其定位是：当 coding agent 同时使用 bash 和 `view_file`、`search_code` 等自定义工具时，帮助模型优先考虑这些工具。面试中应把它理解为产品/服务端针对 tool selection 的配置入口：

```text
model identity: gemini-3.1-pro-preview
endpoint variant: gemini-3.1-pro-preview-customtools
host responsibility: tool schema, permission, execution, result validation
model responsibility: plan, select, and issue tool calls
```

该 endpoint 的质量波动提示了工具分布偏移问题：一个针对工具密集任务优化的提示/路由策略，未必在无工具或普通问答上更好。不能仅因为 endpoint 名称不同，就新增一个模型条目或推断训练了另一套参数。

### 3.4 一百万 token 的上下文仍需要工程策略

[Long context 文档](https://ai.google.dev/gemini-api/docs/long-context)把 1M context 解释为可一次性放入大量文档、代码仓库、视频或音频的短期记忆。对面试尤其重要的是，context window 是容量，不是自动的长期记忆或保证每个位置同等可检索：

- 输入越长，每次请求的输入成本、处理延迟和上下文管理压力越高；
- 重复使用的大段固定资料适合结合 [context caching](https://ai.google.dev/gemini-api/docs/caching) 降低成本和延迟；
- 应把任务指令放在完整资料之后，并用结构化标题、索引和局部摘要帮助模型定位；
- 仍应测量长上下文 recall，而不是因为窗口为 1M 就取消 RAG、分块或 verifier；
- 多轮 Agent 需要同时管理 interaction state、tool state、外部工作区和最终 artifact，不能把 1M 当成无限持久化状态。

### 3.5 多模态与 Agent 的边界

Model Card 与 API 页确认 text/image/audio/video/PDF 输入以及代码仓库理解，适合构造“多模态观察 -> reasoning -> tool call -> 外部执行 -> 新观察”的面试系统图。公开资料没有把 `customtools`、内置搜索、Code Execution 或多模态输入等同于 Google 已公开某种新的视觉 encoder、projector、联合训练 loss 或 agent policy。

## 4. 官方评测与安全证据

Model Card 的评测方法覆盖 reasoning、multimodal capabilities、agentic tool use、multilingual performance 和 long-context；结果页面以 2026 年 2 月数据比较 Gemini 3.1 Pro、Gemini 3 Pro、Sonnet 4.6、Opus 4.6、GPT-5.2 和 GPT-5.3-Codex 等配置。代表性结果包括：

| Benchmark | 设置 | Gemini 3.1 Pro |
|---|---|---:|
| Humanity's Last Exam | no tools | `44.4%` |
| Humanity's Last Exam | Search (blocklist) + Code | `51.4%` |
| ARC-AGI-2 | ARC Prize Verified | `77.1%` |
| GPQA Diamond | no tools | `94.3%` |
| SWE-Bench Pro | single attempt | `54.2%` |
| LiveCodeBench Pro | Elo | `2887` |
| SciCode | scientific research coding | `59%` |
| APEX-Agents | long-horizon professional tasks | `33.5%` |
| GDPval-AA | Elo | `1317` |
| tau2-bench | Retail / Telecom | `90.8%` / `99.3%` |
| MCP Atlas | multi-step MCP workflows | `69.2%` |
| BrowseComp | Search + Python + Browse | `85.9%` |
| MMMU-Pro | no tools | `80.5%` |
| MMMLU | multilingual Q&A | `92.6%` |
| MRCR v2 | 128K / 1M | `84.9%` / `26.3%` |

这些是 Google Model Card 的发布方结果，包含不同工具、模式、对照模型和评测设置；DataCurve 的 452 次 Agent 任务和 Artificial Analysis 的第三方指数不能与这些数字直接合并。

安全部分报告了与 Gemini 3 Pro 的自动评测差异：Text-to-Text `+0.10%`、Multilingual `+0.11%`、Image-to-Text `-0.33%`、Tone `+0.02%`、Unjustified-refusals `-0.08%`。Model Card 明确这些是自动评测相对变化，仍需人工检查和 red teaming。

Frontier Safety Framework 部分把 CBRN、Cyber、Harmful Manipulation、Machine Learning R&D 和 Misalignment 分开评估。页面称 Gemini 3.1 Pro 没有达到这些 domain 的 CCL；Cyber 达到 alert threshold 但没有达到 CCL，并特别说明在计入 inference cost 后，Deep Think mode 的结果显著差于不使用 Deep Think 的设置。这个例子适合面试追问“能力评测是否要把 test-time compute 的成本纳入比较”，但不能据此推导内部安全训练算法。

## 5. 论文与代码检索

截至 2026-09-16：

- [arXiv 标题精确检索](https://arxiv.org/search/?query=%22Gemini+3.1+Pro%22&searchtype=title) 返回 0 个结果；
- [arXiv 全文检索](https://arxiv.org/search/?query=%22Gemini+3.1+Pro%22&searchtype=all) 返回 198 个结果，命中内容主要是外部论文把 Gemini 3.1 Pro 当作被测模型、对照模型或工具链组件；
- 当前没有检出 Google 发布的 Gemini 3.1 Pro 专属 arXiv 技术报告、公开权重或独立代码仓库；
- [Gemini API Cookbook](https://github.com/google-gemini/cookbook) 可用于复现 Google API 的调用、工具和多模态工程模式，但 cookbook 示例不是 Gemini 3.1 Pro 的训练代码。

因此，论文负检索是公开资料状态的记录，不是对未来发布的绝对否定；外部论文可以用于学习评测隔离和 Agent 工程，但不能反向证明 Gemini 3.1 Pro 的内部架构或训练算法。

## 6. 面试问答主线与证据边界

建议围绕以下链路准备：

```text
榜单 configuration
  -> thinking_level / output budget
  -> thought + tool call + signature replay
  -> built-in/custom tool boundary
  -> 1M context + caching
  -> external execution + verifier
```

高频追问及回答边界：

- 为什么不能只看模型名比较 DeepSWE？因为 effort、harness、工具、仓库、环境和 verifier 都在结果生成链路中；
- thought signature 是不是明文 CoT？公开文档将其描述为加密上下文签名/回放字段，不等于可读的完整思维链，也不能当成训练 recipe；
- customtools 是不是新模型？不是。官方把它作为针对 bash/custom tools 工具选择的 endpoint variant；
- 1M context 是否取代 RAG？不取代。成本、延迟、位置召回、更新频率和访问控制仍要求缓存、检索、分块和验证；
- 1M/65K 能否推出模型参数规模？不能。它们是 API token limits；Model Card 没有公开参数和独立架构；
- Deep Think 的能力是否只看分数？不应。安全页面给出了计入 inference cost 后的比较，说明 test-time compute、质量、成本和风险必须一起评估。

## 7. 闭环结论与待核验项

`Gemini 3.1 Pro Preview` 当前升级为**资料级闭环**：两个唯一排行榜的锚点和配置、Google API 模型页、官方 release notes、DeepMind Model Card、评测/安全页面、Thinking、thought signatures、工具组合、long context、caching 及 arXiv 定向检索均已具备。由于 Google 将 3.1 的架构、训练数据、软硬件资料指向 Gemini 3 Pro，且没有独立公开的 3.1 技术报告，本轮不新增 Gemini 3.1 Pro 专属正式架构章节；知识映射进入 Reasoning、Agent/工具协议、长上下文/Serving、多模态、评测与安全章节。

仍待核验：独立参数/层/专家结构、完整训练与后训练 recipe、生产 kernel、目标硬件 profiling、线上工具接受率、`customtools` 与普通 endpoint 的严格 A/B、1M context 的独立 recall/成本复现、Model Card PDF 的版本变化以及 Google 专属技术报告或论文未来是否公开。

## 8. 本轮官方与榜单快照

| 本地快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/overnight-g31-api-7890-20260916.html` | Google AI Developers 模型页 | `106,942` | `15905d54f62691201d5d5def1f93fccd21c8ca7d79e2068b0418545d409a1c5d` |
| `/tmp/overnight-g31-card-7890-20260916.html` | Google DeepMind Model Card 页面 | `160,720` | `22a1568481334165aa2bdde9c500a938ea7ffe5297ee49f86f1b0f9648b5063c4` |
| `/tmp/overnight-g31-evals-20260916.html` | Google DeepMind 评测方法页面 | `378,103` | `5a0a36e6d25d587f5325fe70d185593a9270a46b340d9d97f8cf569360ab8b70` |
| `/tmp/overnight-g31-thinking-20260916.html` | Gemini Thinking 文档 | `35,059` | `9307f175f94749d59327a9e9b4f4c1d3712e48b25b6215d0c46f796abd216df0` |
| `/tmp/overnight-g31-combine-20260916.html` | 工具组合文档 | `25,113` | `373db4c6044658638315380b41089cae50eb4e09a7a773f1214a21108a3146cc` |
| `/tmp/overnight-g31-long-20260916.html` | Long context 文档 | `24,022` | `cbed824426278d5f59e46f59b477eb268d5b8056970b29fe246406188b02cd78` |
| `/tmp/overnight-g31-cache-20260916.html` | Context caching 文档 | `19,727` | `611be0705baf1407cd2c38e686addccb0ee8fdee97f3120cd55fcda310a931a5` |
| `/tmp/overnight-g31-signatures-20260916.html` | Thought signatures 迁移页 | `18,876` | `f767c62adc5c9d3853f509160839c553a25f1968474f0f098748d91795612760` |
| `/tmp/overnight-g31-pdf-20260916.pdf` | Gemini 3.1 Pro Model Card PDF | `875,842` | `70bcda79a248255d2c63d7df6b0548f77565db4764debe0e0d7e1b5f4608fd59` |
| `/tmp/overnight-g3-pdf-20260916.pdf` | Gemini 3 Pro Model Card 依赖入口 | `834,704` | `9a1734bc3aa310691772d1756990b01383dc7732d4f9cb0a2888e2e06a7f84d8` |
| `/tmp/overnight-g31-arxiv-title-20260916.html` | arXiv 标题精确检索 | `16,446` | `ae0471a2894cb658b32775ea7a3956ef0b288defc6c393a1e812836f0757b60d` |
| `/tmp/overnight-g31-arxiv-all-20260916.html` | arXiv 全文检索 | `265,179` | `1cd9ac2ddce64c825fbe140c756b2ed5023a831bdc557503e96726444c56d0c2` |
