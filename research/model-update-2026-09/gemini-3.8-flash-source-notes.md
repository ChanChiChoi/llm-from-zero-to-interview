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
