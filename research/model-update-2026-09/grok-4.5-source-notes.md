# Grok 4.5：官方发布、API 与 Agent 运行时资料摘记

核验日期：2026-09-15。本笔记只研究已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 Grok 4.5；xAI 官方资料用于核验模型和扩展周边技术，不作为新的模型发现入口。榜单页面的日期、指数和 Agent 结果均保留其配置语境，不替代官方发布日期或裸模型能力。

## 榜单锚点

| 来源 | 发现记录 | 可复核字段 |
|---|---|---|
| [Artificial Analysis Grok 4.5](https://artificialanalysis.ai/models/grok-4-5) | 页面配置为 `Grok 4.5 (high)`；榜单 `releaseDate` 字段为 2026-07-08 | Intelligence Index 约 39.08，页面排序约 #38/200，输出速度约 60.98 tokens/s，context 500K，输入/输出价格 `$2/$6` 每百万 token |
| [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) | `mini_swe_agent_grok_4_5_high`；`reasoning_effort: high` | 243/452；Pass@1 53.761%；Pass@4 77.876%；4 runs；平均成本约 `$2.4157`；平均 Agent steps 约 61.33；平均输出约 35,525 tokens；平均耗时约 484.7s |

Artificial Analysis 页面快照保存为 `/tmp/aa-1234.html`，大小 3,517,851 bytes，SHA-256 为 `839c29a20b9f5977fde997e6868f96c15d6c4d4f41c8fb53e5bac15bf7ce078`。DataCurve 快照保存为 `/tmp/deepswe-1234.html`，大小 268,313 bytes，SHA-256 为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；页面更新时间为 2026-09-03，数据对象的 `generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，任务集为 113 个任务、91 个仓库和 5 种语言。

两榜单的数字不能合并为一个分数：Artificial Analysis 的指数、速度和价格绑定它自己的评测协议，DeepSWE 的 Pass@1/4、成本、输出 token 和 steps 绑定 `mini-swe-agent`、工具、任务集、环境、verifier 和 4 次运行。Artificial Analysis 的 `releaseDate` 也只是第三方页面字段，不能单独证明 Grok 4.5 的首发日期。

## xAI 官方发布资料

- [Introducing Grok 4.5](https://x.ai/news/grok-4-5) 标注发布日期为 2026-07-16，并将 Grok 4.5 定位为 coding、agentic tasks 和 knowledge work 模型。
- xAI 描述训练数据覆盖 coding、science、engineering 和 math，并提到 deduplication、quality scoring 与面向领域的数据选择。页面还称训练使用数以万计的 NVIDIA GB300 GPU。
- xAI 描述后训练/RL 使用数十万任务，重点包括多步软件工程和技术任务，并使用 automated grading 与 model-based grading。发布页还提到 highly asynchronous agentic rollouts：rollout 可以持续数小时，同时训练过程继续进行。
- 发布页给出的 DeepSWE 1.0 62.0%、DeepSWE 1.1 53%、SWE Marathon 29.0%、Terminal-Bench 2.1 83.3% 和 SWE-Bench Pro 64.7% 都是 xAI 发布方结果；它们各自的版本、effort、harness、工具、任务和 verifier 不完全相同，不能改写为统一的裸模型 benchmark。
- xAI 发布页还称 80 TPS，并给出 SWE-Bench Pro 平均输出约 15,954 tokens 的比较。该比较仍属于发布方口径，不能与 Artificial Analysis 的 60.98 tokens/s 或 DataCurve 的平均输出直接拼接。

发布页快照来自 `/tmp/xai-grok45-news-2.html`，大小 292,938 bytes，SHA-256 为 `6192e05a76a16bedf3082572bbc2b6cbf34ea49b410c1bbf860a4920ece8d490`。相关的 [Cursor 与 SpaceXAI 合作文章](https://cursor.com/blog/spacex-model-training) 只描述 Composer 1.5/2 的训练合作和 Colossus 计划，没有点名 Grok 4.5，因此不把其中的 continued pretraining 或 RL 扩大 20 倍等内容归因给 Grok 4.5。

## 官方模型与 API 合同

主要来源为 [Grok 4.5 官方模型页](https://docs.x.ai/developers/models/grok-4.5) 和其 [Markdown 版本](https://docs.x.ai/developers/models/grok-4.5.md)。Markdown 快照保存为 `/tmp/xai-grok45-readme.html`，SHA-256 为 `81ddddf9109893f5a84c92cf524984afe446009043dada7aeb4b85cbbb781b2b`。

已确认的公开字段如下：

- 模型 ID 为 `grok-4.5`，别名为 `grok-4.5-latest` 和 `grok-build-latest`。
- 输入为 text/image，输出为 text；最大 prompt/context 为 500,000 tokens。
- 支持 function calling、structured outputs 和 reasoning；Batch API 不支持。
- 专属模型页列出 `low`、`medium`、`high`、`xhigh` 四个 reasoning effort，默认值为 `high`。
- 低于 200K prompt tokens 时，输入、cached input、输出价格分别为 `$2/$0.30/$6` 每百万 token；prompt 达到 200K 后，整次请求按 `$4/$0.60/$12` 计费，而不是只对超出部分加价。
- 限流字段为 150 requests/s 和 50M tokens/minute；可用区域为 `us-east-1`、`us-west-2`。

### 官方文档的不一致

通用 [Reasoning 文档](https://docs.x.ai/developers/model-capabilities/text/reasoning) 的表格写明 `xhigh` 从 Grok 4.6 起支持，Grok 4.5 的 `xhigh` 请求会按 `high` 处理；[Release Notes](https://docs.x.ai/developers/release-notes) 的 Grok 4.5 条目也只列出 low/medium/high。专属模型页却列出了 `xhigh`。这是官方页面之间的版本/规格不一致，当前 DataCurve 的 `high` 配置不受影响；在 API 实测或文档统一前，不能把 Grok 4.5 的 `xhigh` 写成额外推理能力。

## 可迁移的面试技术点

### Reasoning state 与会话延续

通用 reasoning 文档确认 Grok 4.5 的 reasoning 不能关闭，默认 effort 为 `high`，并且 presence penalty、frequency penalty 和 stop 不能用于 reasoning models。Responses API 可以暴露 `reasoning_tokens` 使用统计；通过 `include: ["reasoning.encrypted_content"]` 可以返回加密的 reasoning state，后续请求需要原样回传。这个 opaque blob 是协议状态，不是可供客户端解析的原始思维链。

[Generate Text / Responses API 文档](https://docs.x.ai/developers/model-capabilities/text/generate-text) 说明状态化交互默认开启，response 会在服务端保存 30 天，期间可以用 response ID 继续会话；超过 30 天需要应用自行保存历史和 encrypted thinking content。面试中应把“模型继续推理”拆成服务端状态、模型 reasoning state 和客户端持久化三件事，不要混为一个 memory 特性。

### Context Compaction

[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction) 提供 `/v1/responses/compact`。接口把已有对话压成单个 opaque compaction item，保留系统提示、附件、先前 reasoning 和压缩后的轮次状态，同时去除冗长工具输出和来回消息；下一次请求必须把 `encrypted_content` 或整个 output item 原样传回。

这个机制带来三个面试边界：

1. compaction 只能压缩当前仍能放进 context window 的请求，不能挽救已经超限的输入。
2. 客户端不能解析、裁剪、重排或手工合并 opaque 内容；它应作为可持久化的协议对象处理。
3. compaction 本身也消耗 token，实际系统应比较压缩成本、后续输入成本、TTFT、缓存命中和信息损失，而不是默认压缩越频繁越好。

### 工具循环与责任分层

[Tools Overview](https://docs.x.ai/developers/tools/overview) 把工具分成两类：Web Search、X Search、Code Interpreter 等 built-in tools 由 xAI 服务端执行；custom function calling 只由模型请求，真正的数据库、API、文件和副作用操作由宿主执行器完成。公开的工具循环是“分析请求 → 决定调用 → 执行或返回调用请求 → 处理结果 → 继续或结束”，工具结果和引用应作为独立的运行时事件管理。

[Function Calling](https://docs.x.ai/developers/tools/function-calling) 还确认：`tool_choice` 支持 `auto`、`required`、`none` 和指定函数；默认开启 parallel function calling，可以在一次响应中提出多个调用；每次请求最多 350 个工具定义。工程上必须为工具调用设置权限、超时、幂等、重试、审计和结果校验，不能把 function calling 当作模型已经执行了动作。

### 实时检索与远程 MCP

- [Web Search](https://docs.x.ai/developers/tools/web-search) 支持实时搜索和页面浏览、`allowed_domains`/`excluded_domains`（各最多 5 个域名）、网页图片理解和显式图片搜索。检索结果带 citations；实时搜索是外部数据接入，不等于训练知识或模型天然联网。
- [X Search](https://docs.x.ai/developers/tools/x-search) 支持关键词、语义、用户和 thread fetch，并可按 handle、日期范围、图片和视频理解进行过滤。其费用和返回条目数量需要单独计入 Agent 成本。
- [Remote MCP Tools](https://docs.x.ai/developers/tools/remote-mcp) 支持 Streaming HTTP/SSE 传输。`allowed_tools` 可以把远程 MCP server 暴露的工具限制为最小集合；如果省略，服务器上的所有工具定义可能被注入模型上下文。安全设计应同时检查认证、网络出口、工具 allowlist 和工具定义带来的 context 成本。

### 不要混淆 Multi-agent 模型

[Multi Agent 文档](https://docs.x.ai/developers/model-capabilities/text/multi-agent) 的 supported model 只有 `grok-4.20-multi-agent`；4-agent/16-agent 配置、leader agent 和子 agent 状态属于该独立模型。它们不能套用到 Grok 4.5，也不能因为 Grok 4.5 的发布页提到 asynchronous rollouts，就声称 Grok 4.5 API 内部使用了 4/16 个并行 agent。

## 论文与技术报告检索边界

- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.5%22&searchtype=title) 本轮返回 0 个结果。
- arXiv 全文检索可以找到把 Grok 4.5 作为被测模型或背景模型的论文，但没有检出 xAI 发布的 Grok 4.5 专属论文或独立技术报告。
- 因此当前结论是“本轮公开入口未检出专属论文/报告”，不是绝对断言未来不会发布。不能用其他论文中对 Grok 4.5 的黑盒测评反推 xAI 的训练配方或内部架构。

## 面试映射与待核验项

建议把 Grok 4.5 作为以下面试主线的现实案例：

1. `reasoning_effort` 与基础模型的关系，以及如何设计固定模型、固定 harness 的 low/high 实验。
2. encrypted reasoning state、response ID、context compaction 如何组成长期 Agent 的状态账本。
3. server-side built-in tools、client-side function calling、MCP remote tools 的执行责任、权限边界和成本账本。
4. asynchronous agentic rollout、automated/model-based grading 与长周期 software-engineering 评测之间的关系；哪些是发布方公开描述，哪些仍不能推断。
5. 为什么 DeepSWE 的 243/452 不能直接写成 Grok 4.5 的裸模型 Pass@1，以及如何复现实验协议。

当前仍待核验：参数规模、稠密/MoE 结构、层数和注意力变体；训练数据规模、优化器和完整后训练配方；release page 中 rollout 的具体调度实现；完整 benchmark 任务、verifier、provider 和硬件；线上 acceptance rate、kernel 与独立 profiling；许可证的完整适用范围。没有这些证据，不新增 Grok 4.5 专属架构章节。

### 关联官方文档快照

本轮还保存了 xAI 官方 Markdown 文档快照：`/tmp/xai-reasoning-md.html`、`/tmp/xai-generate-text-md.html`、`/tmp/xai-context-compaction-md.html`、`/tmp/xai-tools-overview-md.html`、`/tmp/xai-function-calling-md.html`、`/tmp/xai-web-search-md.html`、`/tmp/xai-x-search-md.html`、`/tmp/xai-remote-mcp-md.html` 和 `/tmp/xai-release-notes-md.html`。它们用于本笔记的页面复核；正式引用仍以文档链接和核验日期为准。
