# GPT-6 Sol：官方接口、推理控制与 Agent runtime 证据

核验日期：2026-09-23。研究底稿；本轮活动锚点。模型发现仍严格来自 Artificial Analysis 与 DataCurve DeepSWE，官方资料只用于核验已发现的 `gpt-6-sol` 及其周边技术。

## 锚点身份与排行榜边界

Artificial Analysis 的 canonical slug 为 [`gpt-6-sol`](https://artificialanalysis.ai/models/gpt-6-sol)，详情页标题为 `GPT-6 Sol (max)`。三条代理返回逐字节一致的详情页：`547,729` bytes，SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`。页面显示 release 为 2026 年 9 月，max 配置 Intelligence Index 为 `47.5276426437724`、median output speed 为 `115.205383643174 tokens/s`、cost per Intelligence Index task 为 `1.0564240894076389`；这些是 Artificial Analysis 的第三方/provider 测量字段，不是 OpenAI 对模型内部能力的证明。

2026-09-23 的 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 三条代理一致，为 `1,783,001` bytes，SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`。首页同时展示 GPT-6 Sol、Claude Opus 5.5 和 Grok 4.7 等重点厂商条目；本轮只选择 GPT-6 Sol 作为活动锚点，不按 low/medium/high/xhigh/max 或 provider 测量重复建模。

[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照为 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。页面没有精确的 `mini_swe_agent_gpt_6_sol_*` 行，因此不迁移 GPT-6 Astra、GPT-5.6 或其他 GPT 模型的 Pass@1、成本、输出 token 和 Agent steps。排行榜结果若未来出现，也必须按 effort、provider、harness、任务集和 verifier 单独记录。

## OpenAI 官方模型页确认的字段

[GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md) 给出的模型 ID 和定位是：

- Model ID/default snapshot：`gpt-6-sol`；用于复杂 coding 与 agentic workflows。
- 输入为 text/image，输出为 text；支持 reasoning tokens。
- context window `1,050,000`，maximum input `922,000`，maximum output `128,000`。
- knowledge cutoff 为 2026-04-20；这不是发布日期，也不等价于训练数据完整边界。
- `reasoning.effort` 支持 `none`、`low`、`medium`、`high`、`xhigh`、`max`，默认 `medium`。
- Chat Completions、Responses、Batch 支持；Realtime、Fine-tuning、Embeddings、图像生成、音频、视频和 legacy Completions 等端点不支持。
- Responses API 下的支持工具目录包括 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search。
- 支持 streaming、structured outputs、function calling、file search、image input、web search 和 prompt caching。

模型页的价格快照为 input `$2/M`、cached input `$0.20/M`、cache writes `$2.50/M`、output `$10/M`。超过 `272K` input tokens 后，整次请求的 input/cache 按 2x、output 按 1.5x 计价；Batch/Flex 为标准价格的 50%，Fast mode 为适用价格的 2x，regional processing 在可用区域增加 10%。这是一份可观察的服务合同，不能由价格反推参数规模或推理 FLOPs。

## Reasoning：两个独立控制面

[Reasoning models guide](https://developers.openai.com/api/docs/guides/reasoning.md) 的 GPT-6 family 规则给出一个适合面试的控制面拆分：

1. `reasoning.mode` 选择 `standard` 或 `pro` 执行模式；`standard` 是默认模式，`pro` 适合能容忍更多延迟和 token 使用的困难任务。
2. `reasoning.effort` 控制所选模式内部投入多少推理；mode 与 effort 独立。指南明确 GPT-6 Sol 和 Luna 默认 `medium` effort。
3. `pro` 会聚合更多模型工作，并按所选模型标准 token 价格计费；它是执行模式，不是新的 checkpoint。

Reasoning tokens 对 API 客户端不可见，但会占用 context window 并计入 output token。`usage.output_tokens_details.reasoning_tokens` 可用于账本；如果 context 或 `max_output_tokens` 耗尽，响应可能在产生可见文本前以 `incomplete` 结束。因此评测应同时保存 reasoning token、visible output、工具调用、恢复和失败成本，不能只统计最终文本长度。官方建议开始实验时为 reasoning 与 output 预留至少 25,000 tokens。

GPT-6 family 还支持在单 Agent 标准模式中使用 `configuration_update`，在同一会话中把后续 effort 从 low 调高到 high 或反向调整。它改变的是后续请求的推理预算，不是模型权重；更新项需要保留在 `previous_response_id` 或手工 replay 的原位置，两个相邻更新会被拒绝。官方还规定 configuration update 不能和自动 compaction/automatic truncation 组合；显式 compact 后需要在下一条用户消息前重新放置 update。

Reasoning summary 是可选的 summary array，不是 raw reasoning。无状态 replay 时，应用应保留 Responses output items 的 opaque/encrypted 状态；这些状态是协议数据，不应被当作可读思维链或永久记忆。

## 工具与 Agent runtime：模型能力不等于宿主执行

模型页只证明 GPT-6 Sol 的 Responses 工具目录和 feature flags；[Using tools](https://developers.openai.com/api/docs/guides/tools.md) 描述的是平台通用工具接口。工具搜索可以把大工具库拆成 namespace、defer loading 和按需加载的 schema，但搜索结果仍要经过应用的权限、版本和 schema 校验。function calling 只产生调用意图，真实副作用仍由应用或托管执行环境承担。

[Agents](https://developers.openai.com/api/docs/guides/agents.md) 将三种 runtime 的责任分开：

| Runtime | 状态/执行所有者 | 面试时要记录 |
|---|---|---|
| Agents API | OpenAI 托管的 Codex harness | session、turn、item、托管工具、sandbox 和平台进度 |
| Agents SDK | 应用及其 runner | loop、handoff、storage、approval、部署和 sandbox |
| Responses API | 应用自建 harness | response/input item、tool executor、permission、artifact、verifier 和 replay |

一个 Agents API session、SDK session、Responses conversation 和 sandbox 不是同一个资源。模型 ID、effort、工具 schema、权限、executor、verifier 和恢复状态必须放入同一份 trace manifest，才能解释“模型更强”与“宿主更完整”的差异。

## Compaction：长任务的 opaque 状态迁移

[Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 提供两条长任务路径：

- server-side compaction：在 Responses 请求的 `context_management` 中设置 `compact_threshold`；达到阈值后，服务端在同一 stream 中返回 encrypted compaction item，并继续推理。stateless chaining 要把 output items（包括 compaction item）追加到下一次 input；`previous_response_id` chaining 只传新用户消息，不能手工 prune。
- standalone `/responses/compact`：应用提交完整 context window，得到 canonical next context。返回窗口可能包含保留的旧 items，不能只取可读摘要或再次随意裁剪；下一次 `/responses` 应原样使用该窗口。

compaction item 是携带 prior state/reasoning 的 opaque 协议对象，不是给人阅读的摘要。评测和生产 trace 需要记录 compaction 触发点、输入/输出 item、cache prefix、工具 registry、权限和 artifact verifier；只看压缩后 token 数会漏掉 cache miss、重复工具调用和恢复错误。

## 适合面试的技术主线

GPT-6 Sol 本轮公开的新知识主要是“模型合同与 Agent runtime”而非内部架构：

1. 用 `context window`、maximum input、maximum output 和 reasoning/output reservation 建立长上下文预算账本。
2. 用 mode/effort/configuration update 区分执行模式、推理投入和会话内动态调整。
3. 用 Responses typed items、tool call、compaction item、permission、executor 和 verifier 分层解释 coding agent 的闭环。
4. 用 `272K` whole-request pricing threshold、cache read/write、Batch/Flex、Fast mode 和 tool cost 计算质量-成本曲线。
5. 用 `standard/pro`、不同 effort、工具集、harness、任务环境和 verifier 固定评测条件；不把 AA 的 max 配置或未来 DeepSWE 行直接当裸模型能力。

## 证据边界与待核验

截至 2026-09-23，没有从 OpenAI 官方资料确认 GPT-6 Sol 的参数规模、dense/MoE 结构、attention 变体、训练数据、优化器、完整 SFT/RL/蒸馏 recipe、system card、专属技术报告、公开权重、生产 kernel、目标硬件 profiling 或独立 benchmark 复现。Reasoning、tools、Agents 和 compaction 文档证明 API/runtime 行为，不证明模型内部“发明”了某种架构或训练算法。

当前闭环状态为 **AA 单榜资料级闭环**：排行榜 canonical identity、官方模型页、OpenAI runtime 文档、研究笔记和书系配套具备；DataCurve 精确 Agent 行、内部架构、完整训练 recipe、完整权重和生产验收仍缺失。

## 官方快照

| 来源 | 快照 | SHA-256 |
|---|---:|---|
| [Artificial Analysis GPT-6 Sol](https://artificialanalysis.ai/models/gpt-6-sol) | `547,729` bytes | `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116` |
| [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md) | `1,664` bytes | `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0` |
| [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) | `70,315` bytes | `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251` |
| [Agents](https://developers.openai.com/api/docs/guides/agents.md) | `5,432` bytes | `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | `14,272` bytes | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |
| [Using tools](https://developers.openai.com/api/docs/guides/tools.md) | `33,282` bytes | `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` |

