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

当前状态：**资料级闭环**。理由是已具备两个排行榜的锚点记录、Google 官方模型页、DeepMind Model Card、官方发布博客、官方评测 PDF、迁移页和工具/推理文档；但没有独立 3.8 架构或训练报告，也尚无独立正式专题章节。

- Reasoning：thinking level、thought summary/signature、预算与截断。
- Agent 与工具协议：Interactions API、step/event trace、function calling、工具组合和验证闭环。
- Computer Use：截图、坐标、action intent、审批和宿主执行边界。
- Serving 与长上下文：1M 输入、隐式缓存、TTFT/TPOT、token 成本和长文档检索。
- 评测：DeepSWE 的模型配置 + harness 归因、固定版本与长轨迹复现。

本轮不新增 Gemini 专属正式章节；待获得独立架构/训练来源，或现有专题需要专门扩写时，再决定是否落入正式书稿。相关来源和状态同步到 [`source-index.md`](source-index.md)、[`model-inventory.md`](model-inventory.md)、[`inventory-interpretation.md`](inventory-interpretation.md)、[`progress_v2.md`](../../progress_v2.md) 和 [`plan_v2.md`](../../plan_v2.md)。

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
- `function_call`/`function_response` 的 `id` 是调用和回执的关联键；signature 是用于上下文循环的 opaque authenticity/context field。工具意图不等于权限，宿主仍负责 allowlist、审批、超时、幂等、网络/文件沙箱和业务 verifier。
- tool combination 需要 `validated` mode；文档明确不支持 `auto` mode。Built-in tool call parts 在请求中计入 prompt tokens；Google Search 按 query 计费，是 token 计费的特殊例外。

### 15.5 本地协议 toy 与证据等级

新增 [`gemini_interactions_replay_demo.py`](code/gemini_interactions_replay_demo.py)，只使用 Python 标准库和合成 steps，不联网、不调用付费 API、不加载权重。它验证：

- stateful `previous_interaction_id` 只继承 history，下一轮 generation config/tools 仍独立；
- stateless history 必须保留 thought/tool 的 opaque signature，修改或缺失时拒绝；
- function call/result 的 `id` 对齐、SSE step/event 顺序和 `interaction.completed` 终态；
- `store=false` 不能接后续 `previous_interaction_id`，删除后不能继续已删除 state；
- paid/free retention 的 `55/1` 天策略，以及前一轮图像输出到 text-only 模型的 modality gate。

运行结果为 `ok: true`、15 个有序 SSE toy events、`paid_retention_days=55`、`free_retention_days=1`、`network_called=false`。这是 **local protocol toy evidence**，不是 Gemini endpoint、真实 signature 验证、模型质量、API SLA、完整权重、生产 kernel 或线上 acceptance 证据。

当前 Gemini 3.8 Flash 仍为**资料级闭环**。本轮把待办从“补 Interactions 基础资料”推进为“可复验的 state/signature/tool protocol”；真实 API capability probe、low/medium/high thinking 消融、工具组合、长上下文/cache 和真实 billing 仍待在有授权/预算的环境中完成，不把 toy 结果升级为生产结论。
