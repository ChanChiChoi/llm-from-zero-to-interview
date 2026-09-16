# Gemini 3.7 Flash：官方资料与周边技术摘记

核验日期：2026-09-15。本文只记录已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `Gemini 3.7 Flash`，再沿 Google 官方资料追踪面试相关技术。榜单和 API 页面说明的是可观察的模型配置与运行时协议，不等于 Google 已公开了内部参数、架构或训练配方。

## 1. 榜单锚点

- Artificial Analysis 记录了 `Gemini 3.7 Flash (high)`、`(medium)` 和 `(low)` 三个配置，页面内嵌 `releaseDate` 为 2026-08-13。详情页分别为 [high](https://artificialanalysis.ai/models/gemini-3-7-flash)、[medium](https://artificialanalysis.ai/models/gemini-3-7-flash-medium) 和 [low](https://artificialanalysis.ai/models/gemini-3-7-flash-low)。
- high 详情页的第三方字段包括 Intelligence Index `39.4295316404896`、1,000,000 context window、约 `292.3389 tokens/s` median output speed、约 `10.2161s` input/TTFT 字段，以及输入 `$0.75/M`、输出 `$3.75/M`、cache hit `$0.075/M` 的价格字段。页面把该条目标为 proprietary，参数字段为空；这些都是 Artificial Analysis 的 provider 测量或聚合字段。
- DataCurve DeepSWE v1.1 页面快照（页面日期 2026-09-03）包含 `gemini-3.7-flash` 的 low/medium/high 配置，统一 harness 为 `mini-swe-agent`，113 tasks、4 runs。结果如下：

| effort | passed | Pass@1 | Pass@4 | 平均成本 | 输出 token | Agent steps |
|---|---:|---:|---:|---:|---:|---:|
| low | 243/452 | 53.7611% | 77.8761% | `$1.8323` | 73,364.96 | 130.38 |
| medium | 296/452 | 65.4867% | 83.1858% | `$2.0251` | 93,990.85 | 117.37 |
| high | 295/452 | 65.2655% | 82.3009% | `$2.1763` | 107,248.30 | 124.52 |

这些不是裸模型能力：它们同时受 effort、mini-SWE-agent、工具集合、任务环境、任务版本、重试、超时和 verifier 影响。不能把 Artificial Analysis 的指数、速度或价格与 DataCurve 的 Pass@1 拼成一个统一排名，也不能用 Agent steps 反推 Transformer 层数或内部推理算法。

## 2. 官方身份与公开 API 边界

主要依据是 [Google AI Developers 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash) 和 [Google DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-7-flash/)。

| 字段 | 已确认内容 | 证据边界 |
|---|---|---|
| 模型 ID | `gemini-3.7-flash`；stable alias 也是该名称 | API 文档字段，不是权重 revision |
| 发布时间/更新 | Model Card 发布时间为 2026-08-13；API 页 latest update 为 August 2026 | 产品页面时间不等于训练完成时间 |
| 输入/输出 | 输入 text、image、video、audio、PDF；输出 text | 模态接口支持，不代表每种输入都具有相同质量 |
| token limits | 输入 1,048,576；输出 65,536 | 接口预算不等于有效长上下文检索长度 |
| thinking | `low`、`medium`、`high`；`minimal` 不支持并返回错误 | thinking level 是请求级旋钮，不是公开的内部搜索算法 |
| 工具/能力 | caching、code execution、computer use（Preview）、file search、function calling、Maps grounding、Google Search grounding、structured outputs、URL context | 支持协议入口；真实执行仍由 Google 服务与宿主权限控制 |
| consumption | Batch、Flex、Priority inference 均支持 | 账户、区域和实时价格仍需复核 |
| 不支持项 | audio generation、image generation、Live API | API 模型页字段；不推导通用产品能力 |

Model Card 将 Gemini 3.7 描述为 Gemini 3 系列的下一代模型，并公开两条能力方向：对核心 reasoning foundation 的算法改进，以及 agentic video understanding；同时支持可调 thinking 配置，以控制质量、成本和延迟之间的组合。它没有公开 Gemini 3.7 的参数规模、层数、稠密/MoE 结构、训练数据配方、优化器或完整后训练损失。

更具体地，Model Card 把架构、训练数据、硬件和软件信息指向 [Gemini 3.6 Flash Model Card](https://deepmind.google/models/model-cards/gemini-3-6-flash/)。因此 Gemini 3.6 的资料只能作为官方引用的前代技术边界，不能直接写成 Gemini 3.7 独有事实。

## 3. Thinking：质量、成本和延迟的运行时旋钮

Google 模型页只确认 `low/medium/high` 三档 thinking。面试时应把它解释成请求级预算/策略接口：

- low 适合低延迟分类、抽取、草稿或简单工具调用；
- medium 是折中档，适合多数复杂代码和 Agent 任务；
- high 适合多步规划、数学和困难修复；
- minimal 对 Gemini 3.7 Flash 不合法，发送后会报错，不能把它当成 low 以下的隐藏档位。

真实成本不仅取决于最终可见文本，还包括 thinking token、工具轮次、失败重试、缓存命中、TTFT 和宿主执行时间。比较 effort 时应固定模型版本、任务、工具和 verifier，并同时记录成功率、总 token、延迟、工具失败和单位成功成本。API 页没有证明 Gemini 3.7 使用了某种公开的 RL、MCTS、self-consistency 或 verifier 算法。

## 4. Interactions API：把长任务表示为可恢复的 steps

在 [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) 中，一次 interaction 是包含完整执行过程的资源，steps 可以包含 thought、工具调用/结果以及 model output。客户端可以用 `previous_interaction_id` 继续服务端状态，而不是每轮手工重建整个历史。

关键协议边界：

- `store=true` 默认开启；付费层的 interaction 保留 55 天，免费层保留 1 天；
- `store=false` 与 `previous_interaction_id` 和 background execution 不兼容；
- 每次新 interaction 都需要重新指定 tools 和 generation config；上一轮的服务端状态不等于永久记忆，也不等于 KV cache；
- 支持 background execution 和 implicit caching；多模型串联时必须检查输入/输出模态兼容性。

这使 Agent harness 可以把模型调用、工具执行、结果回灌和最终输出作为 trace 审计，而不是只存一段最终字符串。API 仍只负责协议和状态，宿主仍负责工具权限、沙箱、网络、幂等、超时、重试、回滚和敏感动作审批。

## 5. 工具组合、签名与责任分层

[Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination) 文档说明 Gemini 3 系列可以把 built-in tools 与 custom function calling 组合。核心概念是 tool context circulation：工具结果回到模型上下文，模型据此继续规划、修正或结束。

- thought、tool call 和 tool result 可能携带加密 `signature`；它是协议状态的连续性/完整性材料，不是可读的 chain-of-thought；
- stateful 模式由服务端维护 interaction 的 `id` 与 signature；
- stateless 模式必须完整回放先前的 `id`、signature 和相关 steps；只复制可见文本可能破坏后续工具调用或思考连续性；
- built-in tool 由服务端执行；custom function 和 computer use 动作由客户端/宿主执行。

面试中应明确“模型提出动作”和“系统执行动作”是两件事。function calling 的 schema、重试、权限和幂等由应用负责；Computer Use 还需要截图状态、坐标校验、敏感操作确认、域名/窗口限制和审计日志。

## 6. Agentic video understanding：动态浏览，而不是固定抽帧

[Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) 文档把 Gemini 3.7 Flash 的视频能力分为 static 与 agentic 路径：

- static 模式按固定规则处理视频，默认约 1 FPS；
- agentic 模式可以动态浏览时间轴，选择 transcript、帧、帧率和分辨率；
- 同一请求可以混合 agentic/static 视频处理；
- agentic 流程新增 `processing_call` 与 `processing_result` step 类型；
- stateful 模式由服务端保留视频上下文；stateless 模式必须回放 processing steps，否则后续质量会明显下降；
- 长视频建议使用 File API、streaming 或 background execution。

Google 文档自报长视频最多约 88% 的 token efficiency 提升和约 7% 的 quality 提升。它们是产品文档中的发布方测量，不能直接解释成所有视频、所有采样策略都能得到的固定比例。面试重点是“查询—选择证据—处理—再查询”的主动感知闭环，以及 processing step 如何成为可恢复状态。

## 7. 官方评测与安全边界

Google DeepMind 的 [Gemini 3.7 Flash evaluation PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf) 给出发布方评测表。示例数字包括 DeepSWE v1.1 `65.3%`、Terminal-Bench 2.1 `85.8%`、LVBench `85.4%`、GDM-MRCR v2 128K `97.0%` 和 OSWorld-2.0 `47.9%`。这些数字必须与本笔记前面的 DataCurve 结果分栏：前者是 Google 的评测设置，后者是 DataCurve 的统一 harness/环境/verifier 设置。

安全材料显示：cybersecurity 能力达到 alert threshold 但未达到 CCL；CBRN 未达到 TCL/CCL；Google 同时发布了更新后的 CBRN/cyber safeguards。Model Card 还强调，模型可以完成单个 coding task，但尚不能在无人介入下串起端到端研究工作流。安全等级和“能做一个任务”都不能被扩写成内部训练算法或完全自主 Agent 证明。

## 8. 论文、技术报告与官方发布页检索

截至 2026-09-15：

- arXiv 精确标题检索 [`title:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=title) 返回 0 个结果；这只说明没有检出标题精确匹配的独立论文。
- arXiv 全文/摘要检索 [`all:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=all) 返回 3 篇外部使用/评测论文：
  - [Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science](https://arxiv.org/abs/2609.15983)：把 Gemini 3.7 Flash 作为研究 harness 的被测配置之一；论文主题是多 Agent 研究编排，不是 Gemini 3.7 技术报告。
  - [Substrate-Aware AI Agents: Execution Context as a First-Class Input](https://arxiv.org/abs/2609.05232)：将 Gemini 3.7 Flash 与 Claude Opus 5、GPT-5.6-Sol 做执行约束实验；论文主题是把 RAM/时间契约作为 Agent 输入，不披露 Gemini 内部机制。
  - [Beyond End-to-End Success: Diagnosing Failures in Long-Horizon Security LLM Agents](https://arxiv.org/abs/2608.20563)：把 Gemini 3.7 Flash 用于长时程安全 Agent 诊断；论文主题是 checkpoint/failure diagnosis，不是模型发布报告。
- Google DeepMind 的 [Research](https://deepmind.google/research/) 与 [Publications](https://deepmind.google/research/publications/) 入口本次均可访问；未在定向入口中找到 Gemini 3.7 专属技术报告。Google 官方发布博客的候选专属路径此前返回 404，因此不把博客缺失写成模型不存在，也不把前代 Gemini 3.6 的内容迁移为 3.7 独有事实。

结论：当前可支持“Gemini 3.7 Flash 有官方 Model Card、API/Agent 文档、评测与安全资料，以及若干外部使用论文”，不能支持“已有公开的 Gemini 3.7 独立架构论文或完整训练报告”。

## 9. 面试知识映射

- Reasoning：thinking level 与质量/成本/延迟账本；不要把 effort 当成公开的内部搜索算法。
- Agent runtime：Interactions 的 interaction/step/state/background/cache；把服务端状态与宿主 harness 状态分层。
- Tool protocol：built-in/custom 工具组合、tool context circulation、加密 signature、stateful/stateless 回放。
- 多模态 Agent：agentic video 的主动时间轴浏览、transcript/帧选择、processing step 和上下文恢复。
- 长上下文与 Serving：1M 输入、65K 输出、隐式缓存、TTFT、视频 File API/streaming/background 的成本取舍。
- Evaluation：Google 自报 benchmark、Artificial Analysis、DataCurve 三套设置分别记录，固定 provider、harness、工具、任务和 verifier 后再比较。
- Safety：cyber/CBRN threshold、人工介入边界、Computer Use 的执行权限和审计责任。

当前不新增 Gemini 3.7 专属正式架构章节。它已达到**资料级闭环**：两个排行榜锚点、官方模型页、Model Card、官方评测/安全材料、API/工具/视频文档和论文负面/外部使用检索均具备；但独立架构、参数、训练 recipe、专属技术报告和独立 benchmark 复现仍待核验。

## 10. 本地快照审计标识

下列文件位于临时目录，只用于本轮核验审计，不作为仓库资料依赖：

| 本地快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/gemini37-recheck-escalated-aa-detail-20260915.html` | Artificial Analysis high 详情页 | 3,605,373 | `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c` |
| `/tmp/gemini37-aa-medium-detail-20260915.html` | Artificial Analysis medium 详情页 | 3,481,420 | `500fc34c876fa5c2637d5fced42db6c31ec233741dce49320ff50b32f0a99055` |
| `/tmp/gemini37-aa-low-detail-20260915.html` | Artificial Analysis low 详情页 | 3,480,855 | `5b445a95f7720273388fc60cbcf425b482995e867b00d3dee2bbf43243610e4b` |
| `/tmp/gemini37-recheck-escalated-ds-20260915.html` | DataCurve DeepSWE | 268,313 | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| `/tmp/gemini37-recheck-escalated-7890-ai-20260915.html` | Google AI Developers 模型页 | 104,739 | `53e8a497f27951f2d533efa4609a9a45e816e5bda697a4d7470489c88ed9243a` |
| `/tmp/gemini37-recheck-escalated-card-20260915.html` | Google DeepMind Model Card | 159,258 | `b8051a19b7578abd1ab60e222f0afa829db38b415b66fbe91a24d6d89e3cdbc5` |
| `/tmp/gemini37-eval-20260915` | Google 官方评测 PDF | 344,118 | `7971771c34a03898090f4525883e76c8b5bef0fb0eb5afdfd511df8a8faf9f19` |
| `/tmp/gemini37-recheck-escalated-arxiv-title-20260915.html` | arXiv 精确标题检索 | 16,452 | `a92b072d2026186568d09e3e42129649ee852886d6572d278cbb529f57b217a2` |

