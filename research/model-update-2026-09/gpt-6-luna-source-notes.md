# GPT-6 Luna：高吞吐 sibling 的模型合同与 Agent runtime 证据

核验日期：2026-09-23。研究底稿；本轮活动锚点。模型发现仍严格来自 Artificial Analysis 与 DataCurve DeepSWE，官方资料只用于核验已发现的 `gpt-6-luna` 及其周边技术。

## 锚点身份与排行榜边界

Artificial Analysis 的 canonical slug 为 [`gpt-6-luna`](https://artificialanalysis.ai/models/gpt-6-luna)，详情页标题为 `GPT-6 Luna (max)`。三个代理返回逐字节一致：`3,992,084` bytes，SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`。AA 模型对象记录 release date `2026-09-22`、`isReasoning=true`、max effort、`contextWindowTokens=1,000,000`、`parameters=null`、`isOpenWeights=false`；max Intelligence Index 为 `37.2559686869738`，median output speed 为 `153.87508473888 tokens/s`，cost per Intelligence Index task 为 `0.06809498628701058`。这些是 Artificial Analysis 的第三方/provider 测量字段，不能写成 OpenAI 对模型内部能力、参数或 FLOPs 的证明。

2026-09-23 的 [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) 三个代理也逐字节一致：`1,782,868` bytes，SHA-256 `24182a9f96da6b6b96567bc2d069eddb553109a6bb323d07d79d900bb107770c`。当前重点厂商前列同时出现 Claude Opus 5.5、GPT-6 Astra、Grok 4.7、GLM-5.3、Gemini 3.8 Flash、DeepSeek V4.1 Flash、GPT-6 Luna 和 Kimi K3；本轮只选择 `gpt-6-luna`，不把 max、xhigh、high、medium、low、non-reasoning 或 provider 变体重复建模。

[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照为 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。页面没有精确的 `mini_swe_agent_gpt_6_luna_*` 行，因此不迁移 GPT-6 Sol、GPT-6 Astra、GPT-5.6 或其他 GPT 模型的 Pass@1、成本、输出 token 和 Agent steps。DataCurve 的模型集合里有相邻 GPT 配置，但“同厂商”或“同 family”不构成精确评测身份；未来若出现 Luna 行，仍需绑定 effort、provider、harness、任务集和 verifier。

## OpenAI 官方模型页确认的字段

[GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md) 通过 `10.24.27.134:7890` 返回 HTTP 200，快照为 `4,019` bytes，SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`；另两条代理本轮返回 HTTP 403，这是线路访问结果，不是模型页不存在的证据。

- Model ID/default snapshot：`gpt-6-luna`；官方定位是面向 focused、high-volume tasks 的高效率模型。
- 输入为 text/image，输出为 text；支持 reasoning tokens。
- context window `1,050,000`，maximum input `922,000`，maximum output `128,000`。
- knowledge cutoff 为 `2026-05-18`；它是知识截止字段，不等价于发布日期或完整训练数据边界。
- `reasoning.effort` 支持 `none`、`low`、`medium`、`high`、`xhigh`、`max`，默认 `medium`。
- Chat Completions、Responses、Batch 支持；Realtime、Fine-tuning、Embeddings、图像生成、音频、视频和 legacy Completions 等端点不支持。
- Responses API 的支持工具目录包括 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search。
- Chat Completions 只有在 `reasoning_effort=none` 时支持 function calling；这是一条 endpoint/configuration 约束，不是模型没有工具相关能力。

Luna 的价格合同为 input `$0.10/M`、cached input `$0.01/M`、cache writes `$0.125/M`、output `$0.50/M`。超过 `272K` input tokens 后，整次请求的 input/cache 按 2x、output 按 1.5x 计价；Batch/Flex 为标准价格的 50%，Fast mode 为适用价格的 2x，regional processing 在可用区域增加 10%。这与 GPT-6 Sol 的价格档位不同，不能用兄弟模型的价格或 provider 指标替代 Luna 合同。

## 与 GPT-6 family runtime 的关系

以下四份官方通用文档与 GPT-6 Sol 本轮快照逐字节一致，作为 GPT-6 family 的 runtime 证据复用，而不是 Luna 专属内部机制：

| 来源 | 快照 | SHA-256 |
|---|---:|---|
| [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md) | `70,315` bytes | `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251` |
| [Agents](https://developers.openai.com/api/docs/guides/agents.md) | `5,432` bytes | `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a` |
| [Using tools](https://developers.openai.com/api/docs/guides/tools.md) | `33,282` bytes | `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | `14,272` bytes | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |

这些文档给出本轮适合面试的 runtime 主线：

1. `reasoning.mode` 的 `standard/pro` 与 `reasoning.effort` 是两个独立控制面；`configuration_update` 调整的是同一会话后续预算，不是切换 checkpoint。
2. reasoning tokens 占用 context 并计入 output；预算耗尽可能在可见输出前返回 `incomplete`，因此 Luna 的长任务账本必须分开记录 reasoning、visible output、tool call、恢复和失败成本。
3. Agents API、Agents SDK 和 Responses API 的状态/执行所有权不同；模型页的工具目录只证明 capability surface，权限、executor、sandbox、副作用和 verifier 仍由宿主 runtime 负责。
4. server-side 与 standalone compaction 都返回 opaque/encrypted 协议状态；stateless replay 要保留 output items 和 compaction item，不能把它当作可读摘要或永久记忆。

## 面试知识与证据边界

GPT-6 Luna 最有价值的差异不是已公开的新 Transformer 结构，而是 sibling 路由和 serving 成本：用官方“focused、high-volume”定位、精确 model ID、effort、endpoint、context、272K whole-request threshold 和价格合同建立 task-shape/cost ledger；不能把 “efficient” 猜成更小参数、dense/MoE 或特定 kernel。和 GPT-6 Sol 对照时，至少固定 model ID、snapshot、effort、mode、provider、工具、harness、任务环境、verifier、compaction 策略和价格档位。

截至 2026-09-23，没有从 OpenAI 官方资料确认 GPT-6 Luna 的参数规模、dense/MoE 结构、attention 变体、训练数据、优化器、完整 SFT/RL/蒸馏 recipe、system card、专属技术报告、公开权重、生产 kernel、目标硬件 profiling、独立 benchmark 复现或线上 tool acceptance。Reasoning、tools、Agents 和 compaction 文档证明 API/runtime 行为，不证明 Luna 内部“发明”了某种架构或训练算法。

当前闭环状态为 **AA 单榜资料级闭环**：排行榜 canonical identity、OpenAI 官方模型页、GPT-6 family runtime 文档、研究笔记和书系配套具备；DataCurve 精确 Agent 行、内部架构、完整训练 recipe、完整权重和生产验收仍缺失。不新增重复 Transformer 正式章节，正式面试落点复用第六、十六、十七、二十和二十四册。
