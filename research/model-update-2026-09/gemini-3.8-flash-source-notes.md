# Gemini 3.8 Flash：官方资料与周边技术摘记

核验日期：2026-09-14。本文只记录已经出现在 Artificial Analysis 或 DataCurve DeepSWE 的 `Gemini 3.8 Flash`，再沿 Google 官方资料扩展其周边技术。产品页对能力的描述不等于对模型内部架构或训练算法的披露。

## 1. 榜单锚点

- Artificial Analysis 记录了 `Gemini 3.8 Flash (high)`、`(medium)` 和 `(low)` 三个配置，榜单日期均为 2026-09-02。对应页面分别是 [high](https://artificialanalysis.ai/models/gemini-3-8-flash)、[medium](https://artificialanalysis.ai/models/gemini-3-8-flash-medium) 和 [low](https://artificialanalysis.ai/models/gemini-3-8-flash-low)。
- DataCurve DeepSWE v1.1 的 2026-09-03 快照包含 `gemini-3.8-flash` 的 high 配置：Pass@1 74%（区间 ±1%）、平均成本约 $2.36、输出 token 约 143K、Agent steps 166。该结果是模型配置、`mini-swe-agent`、工具、任务环境和 verifier 的组合结果，不是基础模型单独的能力分数。
- Artificial Analysis 的历史快照曾记录 `Gemini 3.8 Flash high` 的 Intelligence Index 约 41.19。该数字属于第三方配置级指数，只能作为榜单复现实验索引，不能与 DeepSWE 的 Pass@1 直接合并或解释成统一能力排序。

这里的三个 effort 行应归并为一个基础模型。`low/medium/high` 是请求时的思考预算或运行配置，不是三个模型架构；DeepSWE 的 Agent steps 和成本也不能反推模型内部用了多少层或多少参数。

## 2. 官方身份与公开字段

主要依据是 [Gemini 3.8 Flash 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) 和 [What's new in Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/latest-model)。

| 字段 | 已确认内容 | 证据边界 |
|---|---|---|
| 模型 ID | `gemini-3.8-flash` | API 文档字段 |
| 生命周期 | GA，稳定版本别名为 `gemini-3.8-flash` | 官方模型页/迁移页；不等于所有地区或账户同时可用 |
| 输入 | 文本、图像、视频、音频、PDF | 接口支持字段 |
| 输出 | 文本 | 不代表支持原生图像或音频生成 |
| 输入上限 | 1,048,576 tokens | 接口预算，不等于有效检索长度 |
| 输出上限 | 65,536 tokens | 还要考虑 thinking tokens 与输出预算的关系 |
| thinking | `low`、`medium`、`high`；默认 `medium` | `minimal` 不支持并会返回错误 |
| 工具 | caching、code execution、computer use（Preview）、file search、function calling、Google Search/Maps grounding、structured outputs、URL context | “支持”只说明协议入口存在，执行权限由宿主与平台控制 |
| 其他 | Batch、Flex、Priority inference 可用；Live API、图像生成和音频生成不在该模型页的支持项中 | 需按账户、平台和实时文档复核 |

Google 文档把该模型定位为面向长周期软件工程、自治 Agent 和复杂企业工作流的 Flash workhorse。页面还给出 introductory price：2026-12-31 前输入/输出分别为每百万 token $0.75/$3.75，2027-01-01 起的 standard price 为 $1.50/$7.50。价格是文档快照字段，不能替代实际账户、地区和平台的计费核对。

## 3. 模型卡与技术报告的证据边界

[Google DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/) 是重要的一手安全与模型资料入口，但它明确把 Gemini 3.8 Flash 的架构、训练数据以及软硬件相关信息指向 Gemini 3.7 Flash Model Card。当前没有看到一份独立公开的 3.8 参数规模、层数、稠密/MoE 结构、优化器、完整训练数据或后训练损失报告。

因此，下列说法不能从当前资料推出：

- 不能把 `Flash` 写成某一种已确认的稀疏架构，也不能猜参数量、专家数或注意力变体。
- 不能把页面中的 `reasoning`、`thinking_level` 或 DeepSWE 的 Agent steps 当成公开的 RL、search、MCTS 或 self-consistency 算法定义。
- 不能把 `1M context` 换算成固定 KV cache 显存、并发数或长文档检索准确率。
- 不能因为官方页面使用“most intelligent Flash”或“smaller reasoning steps”就宣称内部一定采用某种 decoder、奖励模型或 verifier 结构。

Google 的 [官方发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) 发布了两个变体：Gemini 3.8 Flash 和面向受信任防御者的 3.8 Flash Cyber。博客称两者由同一 foundational intelligence 驱动，并通过 long-running agentic loops 递归评估、改进模型；同时强调网络安全领域的严格训练。以上属于发布方的产品/研究叙述，可以作为技术方向线索，但不是具体 RL 算法或网络结构的证明。3.8 Flash Cyber 在本文中只作为同源周边变体，不新增为排行榜锚点。

对 `Gemini 3.8 Flash` 的精确标题进行 arXiv 检索没有得到独立论文结果。截至核验日，论文缺失只能说明尚未找到独立公开论文，不能证明 Google 内部没有未公开报告。

## 4. Adaptive thinking：把推理预算变成运行时旋钮

官方 [Thinking 文档](https://ai.google.dev/gemini-api/docs/thinking) 和迁移页面把 thinking level 作为请求协议的一部分：

- `low` 适合低延迟事件响应、实时聊天、草稿和快速分析。
- `medium` 是 Gemini 3.8 Flash 的默认值，官方建议将其用于复杂代码和 Agent 任务。
- `high` 用于数学、深度推理和困难的多步规划。
- `minimal` 在 Gemini 3.8 Flash 上不支持，发送它会报错；不能把它当成“比 low 更低”的合法档位。

面试时应区分三件事：thinking level 是预算/策略接口；实际内部推理 token 不等于 API 输出的 thought summary；总的 token、延迟、工具轮次和任务成功率才是线上成本账本。官方文档还提醒，`max_output_tokens` 涉及 thinking 与输出的合计预算，过低可能截断最终答案。

这产生一个实用的任务路由原则：简单分类和事实抽取使用低档位，复杂代码修改使用 medium，关键的多步数学或修复任务再使用 high。比较不同档位时，必须记录总 token、TTFT、TPOT、工具失败、重试、最终 artifact 和单位成功成本，而不是只比较回答长度。

## 5. Thought summary 与 thought signature

Google 的 [Thought signatures 文档](https://ai.google.dev/gemini-api/docs/thought-signatures) 和 Thinking 文档把模型的内部推理状态与可见文本分开：

- thought summary 是可选的、面向开发者的摘要，不是完整 chain-of-thought。
- thought signature 是模型内部 reasoning 的加密表示，用于跨多轮交互维持推理连续性。
- 一个 thought block 可能只有 signature、没有 summary；客户端不能假设 summary 一定存在。
- 流式响应中，summary 可以以 `thought_summary` 增量到达，signature 在该 step 的尾部事件中出现。

在 [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) 的 stateful 模式下，设置 `store: true`，下一轮传 `previous_interaction_id`，服务端会保留对话中的 thought block 和 signature，调用方不需要手工拼接它们。stateless 模式则需要在每次请求中原样携带之前返回的模型 steps；内置工具的 call/result 也可能有独立 signature。

这不是 KV cache，也不是永久记忆。它是一个受 API 协议约束的跨轮状态传输机制；切换模型、工具或平台时要检查 signature 兼容性、历史回放和数据保留策略。

## 6. Interactions API：把长任务显式建模为 steps

Interactions API 把一次长任务拆成可观察的 steps，例如 `thought`、`tool_call`、`tool_result` 和 `model_output`。与只返回一段最终文本的接口相比，它更适合长任务 Agent：调用方可以追踪当前处于推理、调用工具还是输出阶段，并用 `previous_interaction_id` 继续下一轮。

典型闭环可以抽象为：

```text
用户目标
  -> 模型 thought / plan
  -> 内置工具或自定义 function call
  -> 工具执行结果
  -> 模型重新评估
  -> 最终 model output 或下一次工具调用
```

官方文档支持 SSE 流式事件，例如 `interaction.created`、`step.start`、`step.delta`、`step.stop` 和 `interaction.completed`。因此 Agent 运行时应把 step/event 当作可审计的 trace，而不是只保存最终字符串。

Interactions API 的价值主要在状态和协议层，不证明 Google 公开了新的 Transformer 结构。它与宿主 harness 仍然是两层：API 产生工具调用意图，宿主负责权限、网络、沙箱、超时、幂等、重试和回滚。

## 7. 工具组合与上下文循环

官方 [工具组合文档](https://ai.google.dev/gemini-api/docs/tool-combination) 允许在同一个工作流里组合 Google 内置工具与自定义 function calling。面试回答应强调“工具结果重新进入上下文”，而不是把工具当成模型权重的一部分。

### 7.1 Google Search grounding

在 [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search) 中，模型可以根据任务自动生成一个或多个搜索查询，获得搜索结果，并返回 grounding metadata/inline citations。这里有三个独立组件：查询生成、外部检索、答案中的引用映射。只打开搜索开关并不保证每个事实都正确，仍要检查引用覆盖、时间、新鲜度和冲突来源。

### 7.2 URL Context

[URL Context](https://ai.google.dev/gemini-api/docs/url-context) 适合对指定网页做深读。文档描述了先检查内部索引/缓存，未命中时再进行 live fetch 的路径。它和 Google Search 的组合可以形成“先广搜候选，再深读指定 URL”的两阶段工作流，但抓取失败、登录墙、动态网页和引用粒度仍需由应用处理。

### 7.3 File Search

[File Search](https://ai.google.dev/gemini-api/docs/file-search) 把文档预处理成可持久化的 store：自动 chunk、生成 embedding、执行 semantic search，再把命中的片段和页面/媒体 citation 放回交互上下文。索引时会产生 embedding 成本，存储免费；查询命中的文档 token 按普通 context token 计费。它解决的是应用文档检索，不等于模型训练或永久记忆。

### 7.4 Code Execution

[Code Execution](https://ai.google.dev/gemini-api/docs/code-execution) 提供 Python 沙箱，并允许模型读取执行结果后继续生成。官方文档给出约 30 秒的执行限制；错误场景最多重新生成 5 次。生产系统仍需限制依赖、网络、文件、资源配额和副作用，不能把“有 Python 沙箱”当成任意代码执行权限。

### 7.5 Computer Use

[Computer Use](https://ai.google.dev/gemini-api/docs/computer-use) 是 Preview 能力，模型通过截图观察浏览器、移动端或桌面环境，返回 action intent 和动作参数；客户端执行动作后再把新的截图/结果送回模型。文档涉及归一化到 `1000x1000` 的坐标表示，以及 `regular`、`require_confirmation`、`blocked` 等安全决策。

Computer Use 的关键面试点是执行闭环：模型只提出动作，客户端仍负责真实点击、权限、确认、域名限制、敏感操作拦截、超时和审计。截图坐标归一化解决跨分辨率接口问题，但不能解决 UI 漂移、遮挡、状态竞争和误点击风险。

### 7.6 Structured outputs

[Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output) 支持 JSON Schema，以及 Python 的 Pydantic、JavaScript/TypeScript 的 Zod 等类型入口，也支持流式的部分 JSON。它保证的是输出结构约束，不保证字段语义、数值正确性、引用完整性或业务规则；应用仍应做 schema 校验、业务校验和重试。

## 8. Long context、caching 与 Agent 成本

Gemini 3.8 Flash 的 1M 输入窗口适合长代码库、视频/音频资料、多轮 Agent 状态和 many-shot in-context learning。它带来的是“可以提交更大的输入”的接口能力，不代表模型对每个远端 token 都同样有效。长上下文文档仍提醒，多 needle 检索、位置偏差、延迟和费用会随任务形态变化；需要用固定位置、文档数量、查询数量和上下文长度做实验。

模型页标记支持 caching。相关文档对 Gemini 3 系列和 Interactions API 的重点是 implicit caching；Gemini 3.8 Flash 的最低隐式缓存输入为 4,096 tokens。缓存命中可以降低重复前缀的成本和延迟，但它不替代 KV cache：前者是服务侧前缀复用策略，后者是一次生成过程中的注意力状态。评测长任务时还要区分缓存命中率、实际计费 token、TTFT 和后续轮 TPOT。

## 9. 官方自述的“新技术”应如何解释

Google 最新模型页称，Gemini 3.8 Flash 在困难、多步任务上会使用更小的 reasoning steps，迭代调用工具，并在过程中验证结果；它也明确提醒日常任务可以降低 thinking effort 以减少 token 消耗。这个描述最适合映射为“模型能力 + Agent runtime policy + 工具/验证闭环”的联合系统：

1. **模型层**：生成计划、工具调用参数和候选答案。
2. **协议层**：通过 thought、signature、tool call、tool result 和 model output 保存步骤。
3. **执行层**：宿主运行搜索、代码、文件或电脑动作。
4. **验证层**：将工具结果、测试结果或引用重新送回模型，决定继续、修正还是结束。

目前没有资料证明上述“验证”具体使用哪一种 verifier、RL 目标、搜索树或奖励模型。面试时应把它回答成公开的系统行为，而不是臆测的内部训练算法。

## 10. 评测与复现建议

以 DeepSWE 记录为例，Gemini 3.8 Flash high 的 74% Pass@1 与 166 Agent steps 同时说明了成功率和执行路径成本；只看 74% 会漏掉长轨迹和高 token 消耗，只看 166 steps 又不能说明质量。复现实验至少固定：

- 模型 ID 和服务版本；
- thinking level、输出上限和是否启用 caching；
- 工具清单、工具 schema、网络/文件权限和 Computer Use 审批策略；
- Agent harness、最大 steps、重试与 verifier；
- 任务版本、仓库 revision、硬件、超时和并发；
- Pass@1、成功 artifact、总 token、TTFT/TPOT、工具失败、重试次数和单位成功成本。

官方 [Gemini 3.8 Flash evaluation PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf) 可用于核对 Google 自己的评测方法和条件，但 PDF 中的发布方数字仍应和外部复现分栏记录。

## 11. 闭环状态与书系映射

当前状态：**内容专题闭环**。已具备两个排行榜的锚点记录、Google 官方资料、研究笔记、第二十册第 23 章和配套同步；独立 3.8 架构/训练报告、真实 API capability probe、目标硬件、独立复现及线上 acceptance 仍未确认。

- Reasoning：thinking level、thought summary/signature、预算与截断；正式专题见第二十册第 23 章。
- Agent 与工具协议：Interactions API、step/event trace、function calling、工具组合和验证闭环；正式专题见[第二十册第 23 章](../../book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)。
- Computer Use：截图、坐标、action intent、审批和宿主执行边界。
- Serving 与长上下文：1M 输入、隐式缓存、TTFT/TPOT、token 成本和长文档检索。
- 评测：DeepSWE 的模型配置 + harness 归因、固定版本与长轨迹复现。

2026-09-24 已将高价值 API/runtime 面试主线落为[第二十册第 23 章](../../book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)，因此 Gemini 3.8 Flash 从“资料级闭环”升级为“内容专题闭环”。这只表示排行榜、权威资料、研究笔记、正式章节与配套同步齐备；独立架构/训练来源、真实 endpoint capability probe、独立 benchmark、目标硬件和线上 acceptance 仍待核验。相关状态同步到 [`source-index.md`](source-index.md)、[`model-inventory.md`](model-inventory.md)、[`inventory-interpretation.md`](inventory-interpretation.md)、[`progress_v3.md`](../../progress_v3.md) 和 [`plan.md`](../../plan.md)。

## 12. 本地快照审计标识

本轮读取的官方页面保存于临时目录，仅作为核验审计辅助，不作为仓库资料依赖：

| 本地快照 | 内容 | SHA-256 |
|---|---|---|
| `/tmp/gemini-3.8-flash.html` | Google AI Developers 模型页 | `304cdd27782d83c124b0eef814f93de3caaa5deced1cf9f6bcc6466bb526b154` |
| `/tmp/gemini-model-card.html` | Google DeepMind Model Card | `1c79de893104f601f8dd2dde77a1138f03d6f5fd673db541c775b6a2c4c5e6d1` |
| `/tmp/blog-3-8.html` | Google 官方发布博客 | `c6792d7f75038f86944a8fd6209ee24a33a1c960743fbbd79fe6b91e4ae1f6aa` |
| `/tmp/latest-model.html` | Gemini API 最新模型/迁移页 | `f0e4abf67c0a47efd2d2d4901bd1c11eea2339b7cefd3f49ccb37a442750c6d5` |
| `/tmp/gemini-3-8-evals.html` | 官方评测 PDF（本地扩展名沿用抓取文件名） | `6c9a1793455330efe88254fd0342d8d0680c560f506f67f982bbabc3eed07a` |

## 13. 2026-09-20 当前快照复验

本轮重新从三个可用代理抓取排行榜，并从 `7890` 重新抓取 Google 官方资料。以下哈希是本轮在线响应的审计标识，与第 12 节的早期本地快照分开保存，不能混用为同一时刻的内容。

| 资料 | 本轮响应大小 | SHA-256 | 复验结果 |
|---|---:|---|---|
| Artificial Analysis `/zh` 首页 | 1,777,695 bytes | `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912` | 三条代理 HTTP 200，逐字节一致 |
| Artificial Analysis Gemini 3.8 high | 517,803 bytes | `0618989e412ae947b3a9f499c1cdae16e4ea3f5ee65a25def360249ae33985ef` | 与 8098 结果一致 |
| Artificial Analysis Gemini 3.8 medium | 511,687 bytes | `d96907dd254166f038084e7586e30ab869fc8a3225501c37c958650859aec6a8` | 与 8098 结果一致 |
| Artificial Analysis Gemini 3.8 low | 509,053 bytes | `ef35f6f567518bd1b9e9e1bc17e4826ea479b7d1e7868bc35f8daf7c464ee5b3` | 与 8098 结果一致 |
| DataCurve DeepSWE | 268,571 bytes | `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` | 三条代理 HTTP 200，逐字节一致 |
| Google Gemini 3.8 模型页 | 23,603 bytes | `1ef402096c255cc1a960f6981e31cc0a575c791e0afed3920678f523ea6ba4dd` | `7890` HTTP 200 |
| Google Thinking 文档 | 37,414 bytes | `47bfe9ed296f388f2aa8021c7431be7bc7de09ffd686598ec442813ef5bb70ae` | `7890` HTTP 200 |
| Google Interactions 文档 | 23,899 bytes | `9c9eac247c1c05d0382d5acb3b53fa96ee713d23bfc15d3eed8765505d69e4ea` | `7890` HTTP 200 |

本轮 AA high 页面仍显示 2026-09-02 release、约 1M context、页面 FAQ Intelligence Index 约 41、Google API TTFT 16.44s 和 `$0.75/$3.75` input/output；`minimal` 不支持来自 Google 官方字段，不能写成 AA 推断。DataCurve 精确对象仍为 `gemini-3-8-flash` + `mini-swe-agent` + `high`：`n_runs=4`、`n_attempted=447`、Pass@1 `0.738255033557047`、Pass@4 `0.8584070796460177`、平均成本 `$2.362349413758389`、平均输出 `143242.6644295302` token、平均 Agent steps `166.3131991051454`、median `161`。

复验没有改变结论：Gemini 3.8 Flash 仍是**资料级闭环**，但没有独立公开的 3.8 架构/参数/训练报告。下一阶段应补真实 API capability probe、Interactions SSE/state replay、thinking budget 消融和工具组合评测；不能由 1M context、thinking 或 DeepSWE steps 反推内部 Transformer、RL 或 verifier 实现。

## 14. 2026-09-21 新鲜榜单与官方 Model Card/发布博客复验

本轮继续沿两个排行榜已经发现的 `Gemini 3.8 Flash` 推进，没有从 Gemini 3.8 Flash Cyber 或官方页面另发现模型。当前可用线路为 `10.237.126.170:1234`；`10.24.27.134:7890`、`10.24.27.134:8098` 对榜单请求超时，Google AI Developers 请求分别出现 503/超时，因此本节不把旧的 AI Developers 文件哈希冒充为本轮新鲜响应。

### 榜单快照

- Artificial Analysis `/zh` 本轮响应为 `1,776,713` bytes，SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`；按 `models/<slug>` 归一化后，八家重点厂商没有新增或删除 canonical 模型。
- Artificial Analysis `gemini-3-8-flash` high 详情为 `3,862,728` bytes，SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`。当前页面字段为 release `2026-09-02`、Intelligence Index `40.9262321765904`、median output speed `328.820180257944` tokens/s、1,000,000 context、输入 `$0.75`、输出 `$3.75`、cache hit `$0.075` 每百万 token。它们都是 Artificial Analysis 目录/测量/provider 字段。
- DataCurve 本轮仍为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确 `mini_swe_agent_gemini_3_8_flash_high` 行为 `n_runs=4`、`n_attempted=447`、Pass@1 `0.738255033557047`、Pass@4 `0.8584070796460177`、平均成本 `$2.362349413758389`、平均输出 `143242.6644295302` token、平均 Agent steps `166.3131991051454`、median `161`。这些数字仍绑定 high、`mini-swe-agent`、工具、环境和 verifier。

### 官方 Model Card 与博客新增证据

- [Gemini 3.8 Flash Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/) 本轮为 `156,591` bytes，SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`。它明确写出：3.8 基于 Gemini 3.7 Flash；输入包括文本、图像、音频、视频，context up to 1M；文本输出 up to 64K；架构、训练数据、数据处理、硬件、软件和评测方法均指向 3.7 Flash Model Card。这是版本继承证据，不是 3.8 独立架构披露。
- Model Card 的评测表把 3.8 与 3.7、Claude Opus 5、Claude Sonnet 5、GPT-5.6 Sol/Terra 并列；其中 HLE-Verified 为 `54.9%`。安全段落称整体安全/语气表现与 3.7 相近，但 multilingual safety 相对 3.7 有 `+5.4pp` 的自动评测回归（数值越低越好）。这些是发布方评测/安全结果，不能与 DataCurve 或 AA 指数拼成统一能力分数。
- [Google 官方发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) 本轮为 `407,603` bytes，SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`。博客把 3.8 Flash 与面向 trusted defenders 的 3.8 Flash Cyber 描述为共享同一 foundational intelligence，并称 long-running agentic loops 会递归评估和改进底层模型；同时把收益归因于网络安全领域的严格训练。这是官方研究/产品叙述，不能还原具体 RL、verifier、搜索树或网络结构。
- 博客还明确描述复杂任务上的行为：模型会执行额外 reasoning steps、迭代调用工具，较高 effort 可能使用更多 token；低 effort 适合计算效率优先的场景。面试应把它回答成“模型策略 + 工具协议 + 宿主验证”的公开系统行为，而不是声称有某种已公开的内部推理算法。

### 当前闭环结论

本轮把 Gemini 3.8 的证据从“API/runtime 资料”补充为“Model Card 继承关系 + 发布方长周期 Agent 叙述 + 新鲜目录字段”。当前仍为**资料级闭环**：没有独立 3.8 参数、层/专家结构、完整训练/后训练 recipe、生产 kernel、目标硬件 profiling、独立复现或线上 acceptance。3.8 Flash Cyber 继续只作为官方周边变体，不进入本项目候选盘点；下一轮仍只从两个排行榜的八家重点厂商条目选择锚点。

## 15. 2026-09-23 Gemini API 文档复验：Interactions 状态、预算和工具上下文

本轮通过 `10.24.27.134:7890` 重新获取 Google AI Developers 的五个官方页面。`10.237.126.170:1234` 与 `10.24.27.134:8098` 对 Google 文档超时；代理差异不被解释成页面不存在。临时快照只用于审计，仓库结论仍以官方 URL 为准：

| 页面 | 快照大小 | SHA-256 |
| --- | ---: | --- |
| [Gemini 3.8 Flash model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) | 103,745 bytes | `e609ede21f99462207a0552cde54c8a7f7462e23a28a7515fb774c84f8f16f2b` |
| [Thinking](https://ai.google.dev/gemini-api/docs/thinking) | 226,287 bytes | `a443bebc66c224284ea8f18ab8f6576fc4aebeb9d3419113be722da4bf7feef8` |
| [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) | 105,650 bytes | `ee250076b4cb66072a5e98951cc2c7d2003b0ae6fce3b5cce398821443b0a553` |
| [Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination) | 124,824 bytes | `038adc9426510e7f15075d966186d7689817a06fd3c3be1c52100a3f397991ff` |
| [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures) | 87,723 bytes | `6b269d05196fef00a5837fb9aa4e42d31034acd072f6bb6064fd91b6a6c89e8c` |

### 15.1 模型页与 thinking budget

- Google 模型页把 `gemini-3.8-flash` 标为 stable alias，输入上限为 `1,048,576` tokens，输出上限为 `65,536` tokens；Thinking 支持 `low`、`medium`、`high`，发送 `minimal` 会报错。模型能力表是 API capability contract，不是参数或架构披露。
- Thinking 文档把动态 thinking 作为默认行为，当前表中 3.8 Flash 默认 `On (medium)`。`thinking_level` 是请求级运行控制，不是三个 checkpoint，也不能从档位名称推出搜索树、RL 目标或 verifier。
- Interactions 的 `thought` 是独立 step。`signature` 是必有的加密/opaque reasoning state，即使模型只进行 minimal reasoning 也可能存在；这不表示 `gemini-3.8-flash` 接受 `minimal` 请求档位。`summary` 是可选数组，可能因 `thinking_summaries` 设置、思考量或内容类型而为空。默认只返回最终输出，开启 `thinking_summaries` 才读取摘要。
- `max_output_tokens` 同时覆盖 thought tokens 和 visible output，是基础设施执行的硬截止线，不会改变 `thinking_level` 如何分配预算。若思考阶段触顶，状态为 `incomplete`，最终输出可能为空或被截断，同时已经产生的 thought tokens 仍计费。想降成本/延迟应先降低 thinking level，而不是设置过小的硬上限。
- Thinking 开启时，价格/usage 应拆成 `total_thought_tokens` 与 `total_output_tokens`。面试中的“输出短”不能直接等同于“没有推理”，评测也不能只保存可见文本。

### 15.2 Stateful/stateless 与 Interaction 生命周期

- Interactions API 的核心资源代表一次完整 turn，按时间顺序保存 `user_input`、`thought`、server/client tool call、tool result 和 `model_output` 等 execution steps；存储资源可通过 `interactions.get` 查看 user input，而 `interactions.create` 响应只返回模型生成的 steps。
- 默认 `store=true`。付费层保留 Interaction `55` 天，免费层 `1` 天；付费项目可以在 AI Studio 配置 `7/14/28/55` 天的日志保留窗口，也可以按 ID 调用 `delete`。保留期结束后自动删除。
- `store=false` 会关闭存储，不能与 `background=true` 或后续 `previous_interaction_id` 一起使用。它和状态管理是两个控制面：不存储不等于客户端不能自己维护完整 history，但客户端必须承担回放完整性。
- Stateful continuation 使用已完成 interaction 的 `id` 作为 `previous_interaction_id`。服务端只替调用方保留 conversation history；`tools`、`system_instruction` 和 `generation_config`（包括 thinking level、temperature 等）是 interaction-scoped，下一轮需要重新指定。模型/工具参数不会因为 history 链接而隐式继承。
- Implicit caching 同时支持 stateful 和 stateless。使用 `previous_interaction_id` 更容易复用 conversation history 的缓存前缀，但它仍是服务侧缓存策略，不是应用永久记忆或 GPU KV cache。
- 同一 conversation 可以混用模型和专用 Agent，但后续模型必须能接受前一个模型的输出模态；例如产生图像后不能接入不接受图像输入的 text-only 模型。这是历史回放的 modality compatibility gate。
- 当前 Interactions 文档还列出尚未可用的边界：Batch API、Python automatic function calling、explicit caching 和自定义 safety settings；Gemini 3 remote MCP 仍不支持。这些是当前 API surface，不是模型训练限制。

### 15.3 Signature、tool id 与文档间的冲突记录

- Thinking 页面给出的 Interactions 规则是：`thought` step 的 `signature` 必有；signature 只出现在 `thought` 或 built-in tool steps（如 `google_search_call/result`），不会出现在 user input、model output 或标准 function call 上。它同时要求 stateless 客户端原样回传所有 thought blocks，并保留 built-in tool result signatures；切换模型时也不要删除前一模型的 thought blocks，后端负责兼容性。
- Tool combination 页面则把 Gemini 3+ 的 tool context circulation 描述得更宽：`function_call` 与 `function_response` 通过 `id` 对齐，`thought` 以及所有 tool call/result steps 都可能带 signature；stateful 模式由服务端维护 `id`/`signature`，stateless 模式由客户端原样回传。
- 这两个官方页面的字段范围表述并不完全一致。本项目不把差异抹平为单一事实：**必须保留实际响应中出现的 opaque fields，不能自行生成、删除或改写；标准 function call 是否带 signature 要以具体 endpoint/schema/SDK 响应为准，并在 capability probe 中单独记录。** 无论采用哪一种字段变体，function call/result 的 `id` 对齐和 thought/tool state 的完整回放都是硬门禁。

### 15.4 Tool context circulation 与执行责任

- Gemini 3 的 tool context circulation 允许 built-in tools 与 custom function declarations 在同一 interaction 中组合；API 返回 built-in tool steps 和 custom `function_call` steps，宿主执行自定义函数并回灌结果。
- `google_search`、Maps、URL Context、File Search 是 server-side tools；Code Execution 也是 server-side，但有自己的 `code_execution`/`code_execution_result` context path。Computer Use 与 custom function calling 是 client-side，仍由应用执行真实动作。
- Tool combination 页面以 `function_call`/`function_response` 描述调用与回执及 signature circulation；固定 Python SDK Interactions schema 使用 `function_call.id` → `function_result.call_id`（详见第 17 节）。signature 是 opaque protocol field；工具意图不等于权限，宿主仍负责 allowlist、审批、超时、幂等、网络/文件沙箱和业务 verifier。
- tool combination 需要 `validated` mode；文档明确不支持 `auto` mode。Built-in tool call parts 在请求中计入 prompt tokens；Google Search 按 query 计费，是 token 计费的特殊例外。

### 15.5 本地协议 toy 与证据等级

新增 [`gemini_interactions_replay_demo.py`](code/gemini_interactions_replay_demo.py)，只使用 Python 标准库和合成 steps，不联网、不调用付费 API、不加载权重。它验证：

- stateful `previous_interaction_id` 只继承 history，下一轮 generation config/tools 仍独立；
- stateless history 必须保留 thought signature 和任何实际收到的 opaque tool fields；toy 分别测试 SDK schema 下 custom function signature 可缺省、文档宽口径下要求存在；
- `function_call.id` → `function_result.call_id` 关联、SSE step/event 顺序和 `interaction.completed` 终态；
- `store=false` 不能接后续 `previous_interaction_id`，删除后不能继续已删除 state；
- paid/free retention 的 `55/1` 天策略，以及前一轮图像输出到 text-only 模型的 modality gate。

运行结果为 `ok: true`、15 个有序 SSE toy events、`paid_retention_days=55`、`free_retention_days=1`、`sdk_schema_allows_missing_function_signatures=true`、`strict_doc_profile_rejects_missing_function_signature=true`、`network_called=false`。这是 **local protocol toy evidence**：两种 signature policy 是对公开 schema/文档边界的合成回归，不是 Gemini endpoint 实际返回、signature 验证、模型质量、API SLA、完整权重、生产 kernel 或线上 acceptance 证据。

当前 Gemini 3.8 Flash 更新为**内容专题闭环**：排行榜锚点、Google 官方来源、研究笔记、第二十册第 23 章和配套题库/练习/项目均已具备。本地 toy 仍只是合成协议证据；真实 API capability probe、low/medium/high thinking 消融、工具组合、长上下文/cache 和真实 billing 仍待在有授权/预算的环境中完成，不把 toy 结果升级为生产结论。

## 16. 2026-09-24 Thinking 与 Tool-combination signature 字段范围复验

本轮用 `10.24.27.134:7890` 重新抓取 Google AI Developers 官方 Markdown。排行榜锚点沿用本轮既有 AA/DataCurve 快照，没有从 Google 文档另建模型。快照记录如下：

| 官方页面 | HTTP / bytes | SHA-256 | 本轮用途 |
| --- | ---: | --- | --- |
| [Thinking](https://ai.google.dev/gemini-api/docs/thinking.md) | 200 / 286,147 | `fb492e15a1ffeed4d2244e90db56352194f87d2adea5d775d2f2ac8fd993ed16` | 明确区分 GenerateContent 与 Interactions 的 signature 范围 |
| [Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination.md) | 200 / 136,858 | `f2b104688cf487094ec37fc944843261b417d0c68eee97203e34e0fa30a83de0` | 给出 Gemini 3+ Interaction tool-call/result 的较宽字段声明 |
| [Function calling](https://ai.google.dev/gemini-api/docs/function-calling.md) | 200 / 983,336 | `f0e00d0dc8f9876ba15d9a1f4fb6997332a199de794a5d8a0a933e2746934ee0` | 说明 Gemini 3 SDK 会自动处理 thought signatures，但未消除 Interactions 字段范围冲突 |
| [Thought signatures（旧入口）](https://ai.google.dev/gemini-api/docs/thought-signatures.md) | 200 / 88,620 | `799b3ed22f5afcb653f5ab589903ecf04e170750961653720d078924dbaace45` | 页面已迁移并重定向到 Thinking 的 signatures 小节 |

Thinking 当前正文明确区分两套 API：GenerateContent 没有专门的 thought step，signature 元数据可附在任意 part（例如 `functionCall` 或响应末 part）；Interactions 把 `thought` 作为一等 step，正文称 signature 只出现在 `thought` 或 built-in tool steps，不出现在标准 function call。相对地，Tool combination 当前正文以 Interaction steps 为上下文，称 Gemini 3+ 的所有 tool-call/result steps（举例包括 `function_call`/`function_response`）都可能有 signature。两者对 **Interactions 自定义 function-call/result** 的范围仍直接不一致；Function calling 页关于 SDK 自动处理的说明不是字段位置的独立裁决。

因此保留为**当前官方文档冲突**，而不是把旧内容简单归因于一个已更新页面。工程不变量仍是保存实际响应中的所有 opaque fields，不自行合成、剥除或改写；`function_call` 与 `function_response` 用 `id` 配对；只有在获准并具备 endpoint/API key 时，才用真实 Gemini 3.8 Interactions 请求记录 schema、SDK 版本、stateful/stateless、原始 steps 与回放结果。此轮未调用 Gemini API；Interactions Markdown URL 本轮超时，所以不宣称完成真实 capability probe。已有 local protocol toy 仅验证保留策略，不验证 Google 服务端究竟在哪些 step 返回 signature。

## 17. 2026-09-24 Interactions API reference 与固定 SDK schema 补证

用户提供的 `10.24.27.134:7890` 百度请求成功；本轮此前已通过同一代理取得 Google 官方 Interactions API reference，HTTP 200、`787,186` bytes，SHA-256 `a06a779e779608c470b178dce0361fb3548ae5121b0c7a17766f8dcce91ceb2e`。同时审计 Google 官方 `googleapis/python-genai` 固定 revision [`4742c9a5c213a587add126a500a271824e2f0add`](https://github.com/googleapis/python-genai/tree/4742c9a5c213a587add126a500a271824e2f0add)，不以 mutable `main` 代表版本化证据。

固定 SDK 的 Interactions typed models 显示：

- `ThoughtStep.signature` 是可选字段。
- `FunctionCallStep` 要求 `id`，但其声明字段中没有 `signature`；`FunctionResultStep` 使用 `call_id`，声明字段中也没有 `signature`。
- `GoogleSearchCallStep` 与 `GoogleSearchResultStep` 显式声明可选 `signature`。这与 Thinking 页面区分 thought/built-in tool 和自定义 function 的收窄口径相容；但 Tool combination 的宽泛说明仍未被真实服务响应裁决。
- SDK 基础 `BaseModel` 配置 `extra="allow"`，序列化时把 extra fields 合并回结果；step union 使用 lenient open-union parser，未知 step 由保留 raw payload 的 `UnknownStep` 表示。因此，typed model 未声明某字段不等于 SDK 必然拒绝或服务端绝不会返回该字段。

Interactions 的 SDK schema 关系是 `function_call.id` → `function_result.call_id`；不要把 GenerateContent 的 `functionCall`/`functionResponse` 命名直接套到 Interactions step。由此将结论细化为：**固定 SDK 声明不要求自定义 function call/result 带 signature，但 endpoint 的实际可选返回行为仍未验证，官方文档范围差异继续保留。**适配层应无损保存已收到的 extra/未知字段，用 call ID 关联结果；不能人为伪造 signature，也不能因为当前 typed model 未声明它就剥掉响应中的字段。

本地摘录的审计标识：Interactions API reference 为 `/tmp/gemini-interactions-api-20260924.html`（上述 SHA）；固定 revision 的 `step`、`thoughtstep`、`functioncallstep`、`functionresultstep`、`googlesearchcallstep`、`googlesearchresultstep` 与 `basemodel` 摘录 SHA-256 分别为 `78addffca98707d944b4477799d3a660ccd13609d4cfd894edac8d7a071b3e11`、`76d4fa71748ce99759d35cdcd722f5b3a51608a51eaff9bdb724d401a91e6edd`、`d47af67c48748cee0aaedc6c836a456fc6871568bd9c5fc23f7fdd6933db25cc`、`56349c5c37d9a3db98d2011ee2dce3b5eeab1fbce4a396b0b48db43da61feb15`、`1c0ef51257825e7b32e180d8e1b855b0b019672262ef43e307f1344afc056e06`、`d7cf27682598cd9e3eb25b1a45354b49520696b0469f5c9b693623032a61f3cb` 和 `1755356700983f1f8e4e80bfb716e0062eb2228b8c0f1670f30ecc9361757f50`。

该次工作区可用 7890 代理访问外网；无 Gemini API key/授权 endpoint，因此没有真实 API capability probe。SDK schema 与官方文档补证提高了协议层证据精度，不升级为真实 endpoint、模型内部架构/训练、硬件或生产验收结论。

## 18. 2026-09-28 7890 当前复验：Interactions 正式 schema 与文档范围差异

用户贴出的百度 HTML 证明其 shell 可经 `10.24.27.134:7890` 出网；当前工作区也用同一代理复现百度 HTTP 200（2,381 bytes），并成功重新读取 Google 官方页面。Google 的 Interactions overview 当前链接到 [`/api/interactions-api`](https://ai.google.dev/api/interactions-api)；旧路径 [`/api/interactions`](https://ai.google.dev/api/interactions) 本轮返回 HTTP 404（83,861-byte 错误页），因此以下使用新路径的 HTTP 200 reference，而不沿用旧 URL。

| 当前官方快照 | bytes | SHA-256 |
|---|---:|---|
| [Thinking](https://ai.google.dev/gemini-api/docs/thinking) | 286,228 | `34b61701fc6594aee9b4bb3bb74abc84e4ef3617ab8d438c1b46dcbaf1f88fe1` |
| [Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination) | 136,835 | `8a3d14cd0b9c21a6ab77c6623ef3c61e35f949a26e4ef93970db7b8654127aaa` |
| [Function calling](https://ai.google.dev/gemini-api/docs/function-calling) | 983,313 | `e23be8459397c437cf1dde590f7ffda0f9a13707aeff909e83f384a9694e057a` |
| [Thought signatures redirect](https://ai.google.dev/gemini-api/docs/thought-signatures) | 88,601 | `342565bfab491081e750e31fb42648fc6eb15d8215c8798e3b1c51060dcd2ace` |
| [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview) | 154,404 | `ba11fde3ed67dab99fca39692485e3839fca678f3d6b011fc49cd38ff030a3d2` |
| [Interactions API reference](https://ai.google.dev/api/interactions-api) | 739,123 | `2514ccc9d5d587f3bd0db50565d77fbece1687075e1f04b7ea004164bcdcc2eb` |

### 18.1 Formal reference/SDK schema 与 prose 的交叉结果

- Thinking 页面仍称 Interactions 的每个 `thought` step 都有必需 `signature`；签名只出现在 thought 或 built-in tool steps，标准 custom function calls 不带签名。
- Tool combination 页面仍用 `function_call`/`function_response`，称两者用 `id` 配对，并称 `signature` 出现在 Gemini 3+ 的所有 tool call/result steps（举例含自定义 function）上。这和 Thinking 的 custom-function 范围直接冲突。
- 最新 Interactions overview 与 API reference 使用 `function_call`/`function_result` 命名。Reference 的 `FunctionCallStep` 字段是 `arguments`、`id`、`name`、`type`；`FunctionResultStep` 是 `call_id`、`result`、可选 `is_error`/`name`、`type`。二者 schema 均未声明 `signature`；正式关联是 `function_call.id` → `function_result.call_id`。Reference 的 `ThoughtStep.signature` 也标为 optional，与 Thinking prose 的“必有”不一致。Search 等 built-in call/result 则显式列出 optional `signature`。
- Google Gen AI Python SDK 当前 `main` Atom feed 最新提交（截至本次读取）为 2026-09-26 的 [`6d012889752f65c1a51d0ad6e5970fc97d19c4ca`](https://github.com/googleapis/python-genai/tree/6d012889752f65c1a51d0ad6e5970fc97d19c4ca)。该 revision 的生成 schema 与 9 月 24 日固定 revision `4742c9a5c213a587add126a500a271824e2f0add` 对关键文件逐字节一致：`FunctionCallStep` SHA `d47af67c48748cee0aaedc6c836a456fc6871568bd9c5fc23f7fdd6933db25cc`、`FunctionResultStep` SHA `56349c5c37d9a3db98d2011ee2dce3b5eeab1fbce4a396b0b48db43da61feb15`、`ThoughtStep` SHA `76d4fa71748ce99759d35cdcd722f5b3a51608a51eaff9bdb724d401a91e6edd`，以及 Step union SHA `78addffca98707d944b4477799d3a660ccd13609d4cfd894edac8d7a071b3e11`。SDK schema 与正式 reference 一致：自定义 function step 未声明签名，thought signature 可缺省。
- Interactions SDK 生成包中的 `google/genai/_gaos/types/basemodel.py` 配置 `extra="allow"` 并在 dump 时合并额外字段；lenient open-union 的 `UnknownStep.raw` 保存未知 step payload。因此字段未出现在 typed schema 里不等于服务端拒绝它，适配器也不应丢弃真实响应的 extra/opaque fields。这里特指 `_gaos` Interactions 类型，不泛指同仓库其他 SDK model base。

结论收窄为：**最新正式 schema 与最新固定 SDK 对自定义 function steps 的字段形状一致，且更支持 Thinking 的窄口径；但 Tool combination prose 仍与二者冲突，Thinking prose 对 `ThoughtStep.signature` 的必需性也与 reference/SDK 的 optional 标记冲突。**当前能确认的是 schema 声明，不是 endpoint 的实际响应、未知字段拒绝策略或模型服务行为。无 Gemini API key/授权 endpoint，本轮仍未发出真实 Interactions capability probe；不要伪造/删除签名，不要把 schema 缺省写成服务端禁止。

## 19. 2026-09-28：Interactions SDK 包边界与 replay-policy/schema 分离

本轮沿 Artificial Analysis 已有的 [`gemini-3-8-flash`](https://artificialanalysis.ai/models/gemini-3-8-flash) canonical 复核；DataCurve DeepSWE 同时有精确 high 配置，未从 Google SDK 或 API 文档另发现模型。复抓的 AA `/zh` 页面为 `1,660,509` bytes / SHA-256 `f17b0194a49a67d83e6079b2ce9a8365964b38edc76bdb827d1fcda4996b0f3a`，与同日先前快照字节数相同但动态内容 hash 不同；对比两份页面抽取出的 `slug` 集合没有差异。AA 模型详情为 `3,842,322` bytes / `9ec6be9274e53726246879eb843cf2912dc0981838b47090f51a6b3b7488053a`。DataCurve 为 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，精确行为 `gemini-3-8-flash + mini-swe-agent + high`：`330/447` runs passed、113 tasks attempted、Pass@1 `73.8255%`、Pass@4 `85.8407%`、4 whole-benchmark runs；页面标示的 95% run-to-run interval `[72.4082%,75.2428%]`、平均成本 `$2.3623` 和平均 166.31 steps 都是该 Agent 配置结果，不是裸模型分数。

7890 当前时点再次读取 Google 正式文档：

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [Interactions API reference](https://ai.google.dev/api/interactions-api) | `739,127` | `1af35ba568a7b9b6e90289f8fb56aa800e0d29a4d4235e31a23ecc3ce79beaa5` |
| [Thinking](https://ai.google.dev/gemini-api/docs/thinking) | `286,228` | `8ef1b32a42126c887bd59266cb59a4922df7be488d37ee466bba6c467cd9dccd` |
| [Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination) | `136,839` | `22f9e3500a732c24f493ceb16fd4adb8f0b71bd3ff722a3e82af48071e8a6949` |
| [Function calling](https://ai.google.dev/gemini-api/docs/function-calling) | `983,313` | `82648a45f98c639f15c269f31678a9dafcdf0afce3f279e78eabb9db588c2b78` |

同日较早的正式 reference 仅相差 4 bytes，关键 schema 片段逐行相同；重新读到的 Thinking/Tool combination 正文也仍保留 §18 所述冲突，没有证据显示接口字段已经统一。Google Gen AI Python SDK `main` 的 Atom feed 快照为 `17,692` bytes / SHA-256 `a6cae16b59565fbacfd0e7c001077d0fb31ce1c6b52246b49d420e071ea9b203`；最新提交仍为 `6d012889752f65c1a51d0ad6e5970fc97d19c4ca`（2026-09-26）。固定该 commit 的 source archive 为 `10,400,023` bytes / `94d2a082ef03cc75bcd0485612ad0ab06c2be5b5bbe38a9be21f2897f4b08d1e`。

源码审计进一步限定了前述“SDK schema”结论的适用范围：Interactions client 在 `google/genai/_gaos/interactions.py` 使用 `_gaos` 生成 types；不是查看同仓库 `google/genai/types.py` 或无关的 `google/genai/_common.py` 就能代表该 endpoint。固定 revision 的 Interactions 专属模型位于：

- `google/genai/_gaos/types/interactions/functioncallstep.py`：字段为 `arguments`、`id`、`name`、`type`，未声明 `signature`。
- `google/genai/_gaos/types/interactions/functionresultstep.py`：`call_id` 与 `result` 必需，`is_error`/`name` 可选，未声明 `signature`。
- `google/genai/_gaos/types/interactions/thoughtstep.py`：`signature` 与 `summary` 均为 optional。
- `google/genai/_gaos/types/basemodel.py`：此 Interactions 专属 base 配置 `extra="allow"`，并在 `model_dump` 中回灌 extra fields。`FunctionCallStep`、`FunctionResultStep`、`ThoughtStep`、Step union 和这个 base 文件的 SHA-256 分别为 `d47af67c48748cee0aaedc6c836a456fc6871568bd9c5fc23f7fdd6933db25cc`、`56349c5c37d9a3db98d2011ee2dce3b5eeab1fbce4a396b0b48db43da61feb15`、`76d4fa71748ce99759d35cdcd722f5b3a51608a51eaff9bdb724d401a91e6edd`、`78addffca98707d944b4477799d3a660ccd13609d4cfd894edac8d7a071b3e11`、`1755356700983f1f8e4e80bfb716e0062eb2228b8c0f1670f30ecc9361757f50`。
- Step union 调用 `parse_open_union(..., lenient=True)`，未知 variant 进 `UnknownStep.raw`。注意同仓库另一个 `google/genai/_common.py` 的普通 `BaseModel` 配置 `extra="forbid"`；这是不同模块边界，不与 `_gaos` Interactions base 矛盾，也不能把某一处配置泛化成全 SDK 行为。

因此“schema 没声明 signature”不能改写成“endpoint 禁止 signature”：在该 Interactions 生成模型里，custom-function signature 属于未声明但可随 extra round-trip 的字段；thought signature 是显式 optional。当前 replay toy 的默认 thought-signature gate 则是更严格的宿主 stateless-replay 安全策略，不是 SDK schema 必填规则。本轮已将两者分开测试。官方 Thinking prose 仍要求精确保留 thought signature，真实 endpoint 接收/返回哪种合法组合依旧需要授权 capability probe；没有 API key，本轮未调用 Interactions API。

## 20. 2026-09-28 Gemini 3.8 Flash：Agentic Video 的按需证据读取

本节继续沿排行榜已确认的 Gemini 3.8 Flash 锚点。Google 模型页列出视频输入；同日成功取得的 [Video understanding 官方文档](https://ai.google.dev/gemini-api/docs/video-understanding) 在请求示例中明确使用 `gemini-3.8-flash`，因此这里讨论的是该模型可调用的视频 API/runtime 路径，不是从文档中新发现模型，也不是内部视觉架构披露。

### 20.1 两种处理策略

- **Static** 是文档描述的默认处理方式：按固定 `1 FPS` 抽帧，一次性把帧送入上下文。它适合短片或低延迟问题，但稀疏采样可能漏掉瞬时动作。
- **Agentic** 根据问题动态浏览时间轴，按需读取 transcript、视频帧或音频片段。Interactions steps 中的 `processing_call` 表示模型请求加载某段媒体（带 `id`）；对应 `processing_result` 用 `call_id` 关联加载结果。它是 Google 媒体处理协议里的专用 step，不应与标准自定义 `function_call/function_result` 的 schema 或 signature 讨论混为一谈。
- 自定义帧率与 clip interval 是 static 模式的定制项；不能把这些选项假定为 agentic 模式下同样有效。

### 20.2 成本、延迟和可观测性

Google 文档称，长视频场景下 agentic 处理通常可比 static 少用最多约 `88%` 的总 token，并报告约 `7%` 的质量提升；同一文档同时提醒，短于 5 分钟的视频可能因导航推理和处理往返而增加 TTFT。这些是发布方文档中的条件性描述，不能外推成所有数据集、问题或模型配置的保证。

Usage 中应区分视频导航推理的 `total_thought_tokens` 与按需载入的帧/音频/transcript 所计的 `total_tool_use_tokens`。因此“少传了几帧”不等于总成本必然下降：还要计入规划、处理步数、往返延迟、失败重试和最终证据是否充分。自建对比至少固定视频 revision、时长/帧率、问题时间位置和模型配置，同时报告关键事件召回、时间定位误差、处理调用数、thought/tool-use tokens、TTFT、端到端成功率和单位成功成本。

### 20.3 Agent/runtime 的责任边界

`processing_call` 是请求读取媒体证据，不是模型获得任意文件系统或外部网络权限。宿主仍须验证媒体对象及租户授权、允许的时间范围和媒体类型，并处理超时、取消、重试、审计与证据引用。Stateful Interactions 通过 `previous_interaction_id` 延续服务端历史；stateless harness 则要按接口规则回放完整的历史 steps。无论哪种模式，都应保留处理调用、返回片段、时间轴位置和最终答案引用，且不把摘要当作可复现的原始证据。

### 20.4 本轮联网与证据边界

较早的本轮请求经 `10.24.27.134:7890` 成功获取 Video Understanding 页面：`303,547` bytes，SHA-256 `b2ec14972a66fd969108167f338e19e00cc3c80d3fc0ee604cfd962545d825c5`；对应 Gemini 3.8 模型页快照为 `106,650` bytes，SHA-256 `6ac136b1e6161438f10cdf1b509272e89825173e0b646a770e9c0cfe958a479b`。本地另有 2026-09-20 视频文档快照 `/tmp/gemini-video-7890-20260920.html`，`262,504` bytes / SHA-256 `7f58e353ad4b3753b6db639430f1aad76372c4fc37ebe5c26f4ac286bf40ede6`，其中也能核对 Gemini 3.8 的示例和上述 step 名称。

此前一次收尾重试经 7890 对上述两个 HTTPS URL 返回 curl `35`（TLS unexpected EOF）；随后在 2026-09-28 使用同一代理并指定 HTTP/1.1 重试，均取得 HTTP 200。AA high 详情新快照为 `3,842,574` bytes / SHA-256 `f88d6155277116adc1580030fa298d821edf244375c76526e9d68572762b12f5`；仍显示 release `2026-09-02`、Intelligence Index `40.9262321765904`、1M context 及 `$0.75/$3.75` 输入/输出价。此响应的 `medianOutputSpeed` 字段为 `311.225851164667` tokens/s；另一份同日 3,862,728-byte 快照曾记 `328.820180257944`，按第三方 provider/测量快照差异分开保存，不推断模型 revision。Video Understanding 新响应为 `303,547` bytes / SHA-256 `e65216368a92d03403d4e37485148f7048b1f7e75be6de37891fd0ab2cb01414`，字节数与先前页面相同但 hash 不同；复核仍含 `processing_call`、`processing_result`、`total_thought_tokens` 和 `total_tool_use_tokens` 等关键字段。故旧 TLS EOF 只代表当时具体请求失败，不代表代理或两个 HTTPS 主机当前不可达；不同页面响应仍按时点、URL 和 artifact 分开记录。

这项增补扩展了已有的 [第十七册 Code Agent 视频案例](../../book-17-agent-tool-use/chapters/07-code-agent.md) 与 [第二十册第 23 章](../../book-20-agent-harness-runtime/chapters/23-gemini-interactions与thought-state回放.md)：前者侧重任务设计与评估，后者侧重 Interaction 状态、媒体 step 回放和宿主权限。仍未调用 Gemini API、下载权重或验证服务端实际 step payload；上述 token/质量数字不是本项目独立复现，也不提供参数量、视觉编码器、训练 recipe 或生产性能证据。

## 21. 2026-09-29：7890 重试先前超时的 Interactions 文档

在用户终端确认 `10.24.27.134:7890` 可访问百度后，本工作区经同一代理重试两个排行榜和先前记录超时的 Google Interactions Markdown 页面。原先尝试的 `https://datacurve.ai/benchmarks/deepswe` 返回 404；项目既有规范入口 `https://deepswe.datacurve.ai/` 则 HTTP 200，不能把错误主机/路径的 404 解释成 DataCurve 不可达。

| 来源 | HTTP / bytes | SHA-256 |
| --- | ---: | --- |
| Artificial Analysis `/zh` | 200 / 1,746,541 | `e0b467739db28960b4736aaee3e0e27951eda5d4794843d7d53f1284b32ca2e9` |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 200 / 268,036 | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1` |
| [Interactions overview Markdown](https://ai.google.dev/gemini-api/docs/interactions-overview.md) | 200 / 154,404 | `73047c07e2547f6ccd1817098c59ded2c2dfc4d1e968ae24e490bc127a9960ea` |
| [Interactions API reference Markdown](https://ai.google.dev/api/interactions-api.md) | 200 / 739,127 | `ba1c19a6868a16d29c66008a34b27f66fb5babcbdbeeba00f3b33ebb48a857b8` |
| [Tool combination Markdown](https://ai.google.dev/gemini-api/docs/tool-combination.md) | 200 / 136,839 | `f4f2645b78da6388ce20f2814107f5475c4dcf11373c7f33f98225afa0b8426e` |
| [Thinking Markdown](https://ai.google.dev/gemini-api/docs/thinking.md) | 200 / 286,232 | `68cde323e37c84ab63aa66d02c22cd3f5d2bb028d8c4185beef3102263b19a22` |

AA 与 2026-09-29 已存快照比较有 60 个唯一 `/models/<slug>` 路由，未见新增或删除；这是路由总数，包含厂商/导航入口，不当作 60 个模型。DataCurve 快照哈希与此前一致，内嵌 70 个唯一 `mini_swe_agent_*` 配置 ID；没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行，因此本轮也没有切换 DeepSeek V4.1 锚点或迁移相邻版本分数。

Google 文档在本次仍返回 HTTP 200，相关声明没有消除既有冲突：Thinking prose 说标准自定义 function steps 不带 signature、thought signature 必有；Tool combination prose 仍称 Gemini 3+ 的所有 tool call/result steps（包括自定义 function）都有 signature。当前正式 API reference 的 `FunctionCallStep` 字段为 `arguments/id/name/type`，`FunctionResultStep` 为 `call_id/result` 加可选 `is_error/name`，都未声明 `signature`；`ThoughtStep.signature` 在 schema 中为 optional。由此只确认文档与 schema 声明范围不一致，不能推断 endpoint 一定返回或拒绝该字段。无 API key/授权 endpoint，未调用 Gemini 服务；stateless 回放仍应无损保留实际收到的 opaque/extra fields，不自行合成或剥除 signature。

本次是对先前网络失败记录的成功重试，未产生新的模型候选或改变 Gemini 3.8 Flash 的**内容专题闭环**状态；这只是既有锚点资料复验，不改变排行榜队列的锚点选择规则。

同日再次通过 7890 对完整排行榜集合及 Google 来源做独立复验。AA `/zh` 新快照为 1,747,602 bytes / SHA-256 `970aaacc0dedaf2a624b5ede1e07fa7f6659da78f8b8d2a0dd451b67da3b3120`，与 13:32 UTC 左右保存的 1,747,864-byte 快照相比，按 `/models/<slug>` 提取并排序后的 60 项路由集合无变化；DataCurve 为 268,036 bytes / `14436c31be1e50a0b62171e4ee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与对照逐字节一致，70 个配置 ID 无增删。

本次 Google 快照为：Interactions overview 154,408 bytes / `d96ea69f069954e120811027a5b43117cd748192dc7ea40f30c74fee603bc71f`；Thinking 286,232 / `079e06dc8cf69c42ee3ab6746b9e819f7a6999e713c72e692812f0aed2f6e554`；Tool combination 136,835 / `84e1ae87bfa171972f30cbdcbd73f526f4af14b3e4fa085503cc14c3512c2d30`；Interactions API reference 739,127 / `6f0aad30591ec004c4de404f0746762a6bf2047b91f1b5c22bbcdadfd73347a6`。Thinking/Tool combination prose 差异及 reference schema 的 optional/未声明字段仍与 §18–§20 相同。

为检查是否有更新的 SDK schema，读取了 Python SDK main Atom feed（HTTP 200，18,637 bytes）；最新项是 2026-09-29T06:17:59Z commit [`b6535e822a6a984bec43056551b2788cd61509f0`](https://github.com/googleapis/python-genai/commit/b6535e822a6a984bec43056551b2788cd61509f0)，内容为 tuning job 的 GCS metrics URI。该 revision 下 `FunctionCallStep`、`FunctionResultStep`、`ThoughtStep`、Google Search call step、Step union 和 Interactions `_gaos` BaseModel 均经 raw URL 复取；六份源码哈希与 §18 固定版本相同。故此次 feed 更新没有改变 Interactions schema，签名范围冲突仍未裁决。没有 Gemini API key/授权 endpoint，未调用真实服务；兼容适配继续原样保留实际返回的 opaque/extra fields，不自行构造或剥除 signature。
