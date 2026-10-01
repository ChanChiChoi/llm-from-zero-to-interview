# GPT-6 Luna：高吞吐 sibling 的模型合同与 Agent runtime 证据

首次核验日期：2026-09-23；最近复核：2026-09-28。研究底稿；本轮活动锚点。模型发现仍严格来自 Artificial Analysis 与 DataCurve DeepSWE，官方资料只用于核验已发现的 `gpt-6-luna` 及其周边技术。

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

### EU data residency 是单独的处理资格，不等于所有数据都留在欧盟

2026-09-24 通过 7890 复核 [OpenAI API data controls / data residency 文档](https://developers.openai.com/api/docs/guides/your-data.md)：HTTP 200，79,672 bytes，SHA-256 `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648`。该文档与 Luna model page 的明确说明合读：GPT-6 Luna 的 EU data residency 仅支持 **Standard processing 的 Responses 与 Chat Completions**。Luna 另有 Batch/Flex 折扣和 Fast mode 加价；若项目要求 EU residency，不能仅根据这些价格选项就假设它们也满足资格，应在 capability manifest 中明确校验 endpoint × region × processing mode。

Data residency 是项目级控制：对需要持久化的服务，eligible customer content 可按合同存储在选择的区域；只有支持 regional processing 的 endpoint/model 才能保证推理也在该区域进行。官方文档把 customer content 与 system data 分开；账号/使用元数据、计费与支持信息、structured-output schema 等 system data 不在 residency 保证内。非美国地区还需要 abuse-monitoring controls 审批及 Modified Retention amendment；欧洲资格表要求适用的 ZDR、Modified Abuse Monitoring、Private Retention with PSP 或 Safety Retention 控制。Remote MCP 是第三方服务，其接收数据受对方自己的 residency policy 约束；客户/终端侧基础设施的位置也可能使传输或存储发生在区域外。因此面试和生产设计不能把“EU endpoint”概括为“所有数据和工具调用都不会离开 EU”。

Data residency 文档说明：2026-03-05 之后发布且符合条件的模型，其 residency endpoint 有 10% uplift；Luna model page 也列出 regional processing 的 10% premium。成本账应将地域资格、项目控制、处理模式、第三方工具和额外费用与 token/Fast/Batch/Flex 单独核算。以上是平台服务/合规合同，不是 Luna 专属模型技术或内部架构证据。

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

## 2026-09-24：7890 当前时点复验

本轮只沿两榜已发现的 `gpt-6-luna` canonical 条目复核，没有从 OpenAI 官方目录另发现模型。`10.24.27.134:7890` 对百度、Artificial Analysis 和 DataCurve 均可连接；AA 页面仍列出 Luna，DataCurve 仍无精确 Agent 配置。

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：HTTP 200，`1,783,893` bytes / SHA-256 `3b895865547acf9b8d9d67d0c2a9a0a0369648f925014b2367817e1f5ac42f36`。与同日旧快照的字节差异只视为动态页面变化。
- [Artificial Analysis GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna)：同日后续快照 HTTP 200，`3,974,386` bytes / SHA-256 `9c6376c8ca63fe1ed56fcc6a85cb5042900ba3ec96df92512760d85002cf594f`。当前 max Intelligence Index `37.2559686869738`、cost per Intelligence Index task `$0.06809498628701058`、median output speed `132.242651126596 tokens/s`；较早同日快照为 `3,974,113` bytes / `8a4603328627740cab856ad4d9b0b37415697b66cc33c59220b0f16615327c81`、速度 `131.449006457181 tokens/s`。Index/成本不变，速度与页面字节只按 provider/采集时点记录，不据此推断模型 revision。约 1M context、`$0.10/$0.50` input/output、September 2026 release 均是 AA 页面/provider 字段。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：HTTP 200，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；未检出 `gpt-6-luna` / `mini_swe_agent_gpt_6_luna_*` 精确配置，不迁移 Sol、Astra 或 GPT-5.6 的 Agent 结果。
- [OpenAI GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)：仍为 `4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`，与既有快照一致。模型 ID、focused/high-volume 产品定位、1,050,000/922,000/128,000 token limits、六档 effort、endpoint/tool support、272K whole-request pricing threshold 和价格合同没有观察到文档变化。
- [OpenAI API data residency guide](https://developers.openai.com/api/docs/guides/your-data.md)：HTTP 200，`79,672` bytes / SHA-256 `4ea5386847a06cbaae7eef04f691d395fed3abcd925b133c4a808e999b700648`。新增核验：Luna 在 EU 仅 Standard processing 的 Responses/Chat Completions 支持 data residency；项目级 customer-content storage 与 regional processing 分开，system data/第三方 MCP 不自动包含在保证范围内；地域与 retention eligibility 进入部署账本。

本轮没有新增 Luna 专属架构、训练 recipe、技术报告或 API 行为差异，故状态继续是 **AA 单榜资料级闭环**，不升级为新技术专题，也不把 GPT-6 family 通用 runtime 文档描述为 Luna 独有发明。面试落点仍是 sibling routing、whole-request cost ledger、能力表与 harness/verifier 归因；内部架构、DataCurve 精确行、权重、独立 benchmark、目标硬件和生产验收维持 `unverified`。

## 2026-09-28：7890 复验与 reasoning-update / compaction 约束

本轮工作区通过 `10.24.27.134:7890` 获取两榜与 OpenAI 官方页面；用户 shell 提供的百度结果与 agent 工作区实测相互印证。Artificial Analysis `/zh` 为 `1,660,578` bytes / SHA-256 `d95b93814772c81fd71e3b33bf2851b7b87edf7d62ce5a7ef1336ae0169bae2f`；Luna 详情 HTTP 200，`3,840,137` bytes / `0c62e5133280f5b671d1d9c0b2fd550672d7ed2f3a96feab9149448969224f1f`。页面仍列 canonical `gpt-6-luna`。首页和详情是动态榜单快照，字节/hash 漂移不能单独证明模型 revision 或能力变化。

DataCurve DeepSWE 当前为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，未检出精确 `mini_swe_agent_gpt_6_luna_*` 行；不迁移 Sol、Astra 或其他 GPT 的 Agent 结果。

OpenAI [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md) 仍为 `4,019` bytes / `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`。GPT-6 family 的 [Using GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model.md?model=gpt-6-luna) 为 `17,456` bytes / `8982485767fcefe9b4c588777de67683b4f56abe9c9af1121c9431eb9c305557`；[Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning.md) 为 `70,315` bytes / `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；[Agents](https://developers.openai.com/api/docs/guides/agents.md) `5,432` / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`、[Tools](https://developers.openai.com/api/docs/guides/tools.md) `33,282` / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` 与 [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) `14,272` / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` 均与已记录快照一致。本轮工作区通过 `curl --proxy http://10.24.27.134:7890` 重新取得 model page、family guide、Reasoning 与 Compaction Markdown，大小和哈希均复现。

本次把跨页状态边界明确归入 Luna 的 GPT-6-family API 合同，而非 Luna 独有机制：`configuration_update` 只改变后续 reasoning effort；它不能与自动 compaction/truncation 同用，含该 item 的历史也不能提交给独立 `/responses/compact`。若要保留显式预算切换，可在 `/responses` 中使用 `compaction_trigger`；压缩完成后、下一条 user message 之前重新加入所需 update。相邻 update 仍会被拒绝。此协议适合做 history/replay 与拒绝路径验收，但本轮未调用真实 API；官方文档说明不构成模型内部架构证据。

另一个容易误读的观测边界是：官方 [Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning.md) 明确说明，响应中的 `reasoning.effort` 仍报告 request-level setting，不报告 `configuration_update` 选择的当前生效 effort。因此不能只读单个 response 字段来重建实际预算；客户端须保留并按序回放 `configuration_update` 历史，并遵守显式 compaction 后、下一条 user message 前重新插入 update 的规则。本轮只复核文档，没有真实 API probe；这是 API 可观测性/状态账本要求，不是模型内部机制。

状态保持 **AA 单榜资料级闭环**：Luna canonical 仍在 AA，DataCurve 精确行、内部架构/训练 recipe、独立 benchmark 与生产验收仍未确认。本轮对照现有模型清单未发现新的重点厂商 canonical；历史清单中的 GLM-5/GLM-5V Turbo 等候选不属于本次 Luna 核验范围。
