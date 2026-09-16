# Gemini 3.6 Flash：排行榜、Model Card 与 Agent 运行时摘记

核验日期：2026-09-15。本文只记录已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `Gemini 3.6 Flash`，再沿 Google 官方模型页、DeepMind Model Card、API 文档和论文检索结果追踪面试相关技术。榜单、API 和 Model Card 的公开字段不等于 Google 已公开了完整参数、架构或训练配方。

## 1. 榜单锚点与证据边界

### Artificial Analysis

- 发现页：[Gemini 3.6 Flash (high)](https://artificialanalysis.ai/models/gemini-3-6-flash)。页面内嵌的第三方 `releaseDate` 为 `2026-07-21`；本轮通过 `10.24.27.134:8098` 重新获取。
- 详情页字段：Intelligence Index `34.3395678920714`；median output speed `192.733708656795 tokens/s`；time to first answer token/TTFT `19.2238583065s`；context window `1,000,000` tokens；约 `90M` Intelligence Index output tokens；cost per Intelligence Index task `$0.9288448964661649`；页面正文将完整评测成本显示为 `$1036.81`。
- 页面列出的配置是 `high` reasoning 版本，模型标为 proprietary，参数字段为空。上述指数、速度、TTFT、成本和 `releaseDate` 都是 Artificial Analysis 的第三方页面/测量字段，不是 Google 的训练事实。
- 详情页 pricing dataset 记录 cache hit/input/output 约为 `$0.15/$0.75/$3.75` 每百万 token。provider benchmark 页只展示 `Google AI Studio` 一个实际 provider，输出速度 `192.7 t/s`、TTFT `19.22s`、blended price `$0.63/M`；详情页 FAQ 同时写“可通过 4 个 API provider 使用”。这两个数字描述的口径不同：不能把 provider 可用性 FAQ 误读成当前 provider benchmark 已有四个 provider。
- 详情页与 provider 页快照的价格/速度可能随 provider、时间和价格档位变化；本笔记优先保留页面原始字段和抓取日期，不把 `$0.75/$3.75` 当成所有渠道的永久价格。

### DataCurve DeepSWE v1.1

DataCurve 当前快照为 113 tasks、91 repositories、5 languages、统一 `mini-swe-agent`，页面生成时间为 `2026-09-03T22:24:37.984682+00:00`。原始数据行是：

```text
model=gemini-3-6-flash
harness=mini-swe-agent
reasoning_effort=high
config=mini_swe_agent_gemini_3_6_flash_high
n_passed=211, n_attempted=452, n_runs=4
pass_at_1=0.4668141592920354, pass_at_4=0.7522123893805309
mean_cost_usd=2.209506638666667, median_cost_usd=1.7223476625000005
mean_output_tokens=95844.86222222222, median_output_tokens=86727
mean_input_tokens=12595957.344444444, median_input_tokens=8949018.5
mean_cache_tokens=11254636.448888889
mean_duration_seconds=1477.8607510707964, median_duration_seconds=1214.8670550000002
mean_agent_steps=116.73333333333333, median_agent_steps=108
median_peak_context_tokens=151398.5
median_output_tokens_to_pass=83414
```

这是一条“模型配置 + harness + 工具 + 仓库环境 + 任务集 + timeout/预算 + verifier”的系统结果。不能把 `46.6814%` 归因给裸 Gemini 3.6，也不能把它与 Artificial Analysis Intelligence Index 拼成统一排名。尤其不能用 Agent steps 反推 Transformer 层数、推理 token 或内部搜索算法。

## 2. Google 官方身份与模型卡

主要来源是 [Gemini 3.6 Flash API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash) 和 [Gemini 3.6 Flash Model Card](https://deepmind.google/models/model-cards/gemini-3-6-flash/)。

| 字段 | 官方已确认内容 | 证据边界 |
|---|---|---|
| 模型 ID | `gemini-3.6-flash`；稳定别名也是该 ID | API 产品字段，不是权重 revision |
| Model Card 日期 | 2026-07-21 | Model Card 发布时间，不等于训练完成时间 |
| 依赖关系 | Gemini 3.6 Flash is based on Gemini 3.5 Flash | 只能说明 Model Card 的公开依赖，不能把 3.5 的全部细节写成 3.6 独有创新 |
| 输入/输出 | text、image、video、audio、PDF 输入；text 输出 | 接口能力，不代表每种模态内部共享同一处理路径 |
| token limits | 输入 `1,048,576`；输出 `65,536` | API 预算，不等于有效长上下文检索率 |
| thinking | 默认 `medium`；支持 `minimal/low/medium/high` | 请求级策略/预算旋钮，不是公开的内部搜索算法 |
| 能力 | caching、code execution、Computer Use（Preview）、File Search、function calling、Google Maps/Search grounding、structured outputs、thinking、URL context | “Supported”只证明协议入口，宿主仍负责权限、执行和验证 |
| 不支持项 | audio generation、image generation、Live API | 当前模型页的能力矩阵 |
| 消费档位 | Batch API、Flex inference、Priority inference 均列为支持 | 实际可用性还受账户、区域和实时计费策略影响 |
| 知识截止 | 2026 年 3 月 | Model Card 字段；不能当成实时联网能力 |

Model Card 将 3.6 Flash 定位为 Gemini 3 系列的原生多模态 reasoning workhorse，适用方向包括 agentic workflows、复杂视频推理、coding 和 enterprise workflows。它明确把架构、训练数据、数据处理、硬件和软件资料指向 Gemini 3.5 Flash Model Card。当前没有公开确认的 3.6 参数规模、层数、稠密/MoE 结构、注意力变体、优化器或完整后训练 loss。

## 3. Model Card benchmark：发布方结果与安全边界

Model Card 的比较表覆盖 reasoning、coding、agentic、multimodal 和 long-context，表内同时列出 Gemini 3.5 Flash、Gemini 3.1 Pro、GPT-5.6 Luna、Grok 4.5 和 Claude Sonnet 5。Gemini 3.6 Flash 一列如下：

| 项目 | Gemini 3.6 Flash | 备注 |
|---|---:|---|
| Input price $/1M（无 caching） | `$1.50` | Model Card 表格口径，与 Artificial Analysis provider 字段不同 |
| Output price $/1M | `$7.50` | 同上 |
| SWE-Bench Pro (Public) | `58.7%` | diverse agentic coding tasks |
| DeepSWE v1.1 | `49%` | long-horizon software engineering |
| Terminal-Bench 2.1 | `78.0%` | Terminus-2 harness |
| MLE-Bench | `63.9%` | machine learning engineering |
| GDPVal-AA v2 | `1421` | Elo |
| OSWorld-Verified | `83.0%` | agentic computer use |
| CharXiv（无工具） | `85.2%` | complex charts |
| CharXiv（有工具） | `89.4%` | complex charts |
| GDM-MRCR v2（128K average） | `91.8%` | 8-needle long context |
| GDM-MRCR v2（1M pointwise） | `54.0%` | 同一 Model Card 表格 |

这些是 Google Model Card 的发布方评测，不能与 DataCurve 的 `mini-swe-agent`/113 tasks/4 runs 或 Artificial Analysis 的 provider 测量相互替代。尤其是 Model Card 的 DeepSWE `49%` 与 DataCurve 的 `46.6814%` 并非同一 harness、任务快照或统计口径。

Model Card 还说明内部自动安全评测相对 Gemini 3.5 Flash 的变化（文档注明不是人工评估或 red teaming）：Text-to-Text Safety `-1.35%`（lower is better）、Multilingual Safety `-5.45%`（lower is better）、Image-to-Text Safety `0%`（lower is better）、Tone `-3.31`（higher is better）、Unjustified-refusals `+0.25%`（lower is better）。Tone 的脚注说明正向变化才代表改善，因此这里按表格原值记录，不把“整体更安全”扩写为所有维度均改善。

Frontier Safety Assessment 的公开结论是：相对于 Gemini 3.1 Pro，3.6 Flash 在框架覆盖领域没有 meaningful new capabilities 或 material performance increases，预计不会达到 CCL；针对 cyber 额外测试仍低于 cyber CCL。Model Card 同时说加强了 CBRN 与 cyber misuse safeguards，并努力减少有益请求的误拒。已知限制包括 hallucination、偶发 slow/timeout，以及知识截止的跨领域不一致。上述是安全评估边界，不是对内部安全训练算法的披露。

## 4. Thinking 与长任务协议

### 4.1 Gemini 3.6 的 thinking 档位

Google [Thinking 文档](https://ai.google.dev/gemini-api/docs/thinking) 的模型表明确列出：Gemini 3.6 Flash 默认 thinking 为 `On (medium)`，支持 `minimal`、`low`、`medium`、`high`。这一点与 Gemini 3.7 Flash 的支持表不同，不能把 3.7 的“minimal 不支持”迁移给 3.6。

面试中应把 thinking level 当作请求级质量—成本—延迟控制面：简单抽取可用 minimal/low，多步 coding 或 Agent 任务用 medium，困难数学/规划用 high。它不等于公开的 RL、MCTS、self-consistency 或 verifier 定义；比较档位时应同时记录总 token、TTFT、工具轮次、重试、成功 artifact 和单位成功成本。

### 4.2 Interactions API 与 state replay

Google [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) 的支持表列出 `gemini-3.6-flash`。interaction 由多个 steps 组成，可表达 thought、内置工具调用/结果、function call/response 和 model output；`previous_interaction_id` 允许服务端继续前一 interaction，stateless 模式则由调用方回放完整历史。

这里的状态是 API 协议状态，不是 KV cache 或永久记忆。应用仍需负责工具权限、沙箱、网络、超时、幂等、重试、回滚和敏感操作审批。stateful/stateless 的选择还会影响 signature、隐式缓存、审计和数据保留策略。

### 4.3 Tool context circulation 与签名

Google [工具组合文档](https://ai.google.dev/gemini-api/docs/tool-combination)说明 Gemini 3 支持 built-in tools 与 custom function calling 的组合，并通过 tool context circulation 把内置工具上下文暴露给后续自定义工具。文档明确指出，Gemini 3+ 的 thought、function call 和 function response steps 可能带有加密 `signature`；stateful 模式由服务端维护，stateless 模式必须完整回放 `id` 与 `signature`。签名是连续性/完整性材料，不是可读的 chain-of-thought。

工程上要分开四层：模型提出动作、API 编排 steps、客户端执行函数、verifier 检查结果。模型返回 function call 不等于动作已经发生；宿主必须做 schema 校验、最小权限、幂等、敏感动作确认和审计。

### 4.4 Agentic video understanding

Google [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) 文档明确把 Gemini 3.6 Flash 列入 agentic video 支持模型。Static 模式以固定采样率（文档示例为 1 FPS）一次性放入上下文；agentic 模式由模型动态探索时间轴、按需加载片段或 transcript。文档给出的产品级描述是长内容最多约 88% token efficiency 提升和约 7% quality 提升，不能当成所有视频任务的固定保证。

agentic 处理可在 interaction.steps 中观察 `processing_call` 与 `processing_result`：前者请求片段或 transcript，后者回灌结果并以 `call_id` 关联。视频上下文可跨轮保持；stateless 回放时不能只保留最终文字，必须保留相关 processing/tool/thought steps 和 signatures。这个机制适合面试中解释“主动感知—证据选择—再推理”的闭环，但不证明内部采用了特定 video encoder 或搜索算法。

### 4.5 Caching 与 Computer Use

[Context caching](https://ai.google.dev/gemini-api/docs/caching) 页面列出 Gemini 3.6 Flash 的 implicit caching 最小输入为 `4,096` tokens，并说明 stateful/stateless 模式均可使用。隐式缓存是服务侧前缀复用/计费策略，不等于一次 decode 中的 KV cache，也不等于永久记忆。

模型页把 Computer Use 标为 Preview；[Computer Use 文档](https://ai.google.dev/gemini-api/docs/computer-use)按 Gemini 3.x 说明截图观察、动作意图和客户端执行循环。真实点击、坐标校验、域名/窗口限制、敏感动作确认、超时和审计仍属于宿主，不应因为模型“支持 Computer Use”就把权限归给模型。

## 5. arXiv 全文检索：9 篇外部使用/评测论文

[精确标题检索](https://arxiv.org/search/?query=%22Gemini+3.6+Flash%22&searchtype=title) 返回 0 篇；[全文检索](https://arxiv.org/search/?query=%22Gemini+3.6+Flash%22&searchtype=all) 返回 9 篇。它们都不是 Gemini 3.6 专属技术报告，只能作为外部使用和评测证据：

| arXiv | 标题 | 论文中 Gemini 3.6 的角色与面试价值 |
|---|---|---|
| [2609.14973](https://arxiv.org/abs/2609.14973) | *PhysBrain 1.5: From Vision-Language Models to Physical Foundation Models* | 8B embodied model 在 28 个 benchmark 上与 Gemini 3.6 Flash 做能力对照；论文主题是 observation—interaction—environment loop，不披露 Gemini 内部机制。 |
| [2609.08402](https://arxiv.org/abs/2609.08402) | *Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method* | AGOS-Agent 以 search—handoff—verify 协作协议增强多 VLM；Gemini 3.6 hard split 的 SR 从 8.6% 到 55.7%、SPL 从 7.6% 到 44.0%，属于外部 agent 方法结果。 |
| [2608.24921](https://arxiv.org/abs/2608.24921) | *post-graph-rag: A PostgreSQL-Native Bi-Temporal Graph RAG Engine with Temporal Grounding at Synthesis* | 在 LongMemEval 中把 Gemini 3.6 作为推理模型，报告 bi-temporal graph RAG 94.0；技术贡献是时间有效性/信念时间建模，不是 Gemini 架构。 |
| [2608.23061](https://arxiv.org/abs/2608.23061) | *Improving O-RADS Risk Stratification from Ultrasound Reports: A Comparative Evaluation of Hybrid versus End-to-End LLM Reasoning Strategies* | feature extraction 与 deterministic rule classification 解耦；Gemini 3.6 的 hybrid 方案报告 99.2% accuracy、weighted kappa 1.00，说明“模型抽取 + 规则执行”比端到端更可审计，但属于医疗系统实验。 |
| [2608.16663](https://arxiv.org/abs/2608.16663) | *Bounded Semantic Planning and Deterministic Compilation for Reliable Enterprise Text-to-SQL* | SPC 将语义规划、图遍历、grain lowering、SQL 构造和检查放进确定性编译路径；Gemini 3.6 只出现在额外 robustness runs，作者明确不作 compilation-only 因果结论。 |
| [2608.06361](https://arxiv.org/abs/2608.06361) | *The Low Frequency Trap: Video Language Models Fail at Simple Event Bookkeeping* | 用可执行 event trace 测视频计数；Gemini 3.6 对持久状态转移在 0.5/1.0 Hz、最多 12 个事件达到 80% reliability，但对 transient blinking 没有可靠正区间，展示“多抽帧不必然等于忠实证据恢复”。 |
| [2607.23976](https://arxiv.org/abs/2607.23976) | *Tag Questions and the Generational Reversal of Sycophancy Across 45 Language Models* | 将 Gemini 3.6 与 Claude Opus 5 作为 out-of-sample release 测试；研究的是 tag-question 表面结构与迎合行为，不是 Google 的训练报告。 |
| [2607.23893](https://arxiv.org/abs/2607.23893) | *Who Gets Named: Citation Type Predicts Individual Naming by Grounded Language Models, and a Roster Instrument Captures 0.5% of It* | 2,400 次 grounded API 调用中包含 Gemini 3.6；论文研究引用/语言/类别对人物命名的影响，报告 Gemini 命名率 9.3%，不能归因成模型通用知识能力。 |
| [2601.16755](https://arxiv.org/abs/2601.16755) | *An Empirical Study of Foundation Models for Variability-Induced Compilation Errors in Configurable C Code* | 在受控样本中 Gemini 3.6 修复 182/190 个 faulty snippets（95.8%）；结果绑定任务构造、编译器和评测样本，不是通用代码修复 benchmark。 |

这些论文最适合扩展面试问题：如何设计 hybrid deterministic boundary、如何测长视频事件恢复、如何分离 RAG 时间有效性、如何做 grounded model 行为审计，以及为什么外部论文中的模型分数不能反推厂商训练方法。

## 6. 资料级闭环结论与待核验项

当前已具备：

1. Artificial Analysis 与 DataCurve 两个排行榜的锚点、配置、日期、快照和原始结果；
2. Google AI Developers 模型页、DeepMind Model Card、官方 API/Agent/视频/缓存/Computer Use 文档；
3. Model Card benchmark 与安全边界；
4. arXiv 精确负检索和 9 篇外部使用论文的标题、摘要要点与证据边界。

因此 `Gemini 3.6 Flash` 从“仅候选”升级为**资料级闭环**，暂不新增独立正式架构章节。面试主线映射到 Reasoning（thinking level）、Agent runtime（Interactions/steps/state）、工具协议（signature/tool context circulation）、视频多模态（agentic processing）、长上下文/Serving（1M、implicit caching）、Computer Use 权限和评测/安全证据分层。

仍待核验：Gemini 3.6 专属参数规模、层/专家/注意力结构、完整训练和后训练配方、生产 video/attention/kernel、目标硬件 profiling、线上 tool/agent acceptance rate、独立复现和可下载的专属技术报告。Gemini 3.5 Model Card 的内容仅作为官方引用的前代边界，不升级成 3.6 独有事实。

## 7. 本地快照审计标识

临时快照只用于本轮核验，不作为仓库资料依赖：

| 本地快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/gemini36-recheck-aa-8098-20260915.html` | Artificial Analysis 详情页 | 3,600,704 | `362fd94a45955783974462edb3cdc99ed932625a56d8c0afe7b9bca4a7f2e721` |
| `/tmp/gemini36-recheck-aa-providers-7890-20260915.html` | Artificial Analysis provider 页 | 735,919 | `08a9a7b46db5d352016fa894b9bf051a503ee1b8a1d44c71569affc840d082bf` |
| `/tmp/gemini36-recheck-ai-7890-20260915.html` | Google AI Developers 模型页 | 104,839 | `e0d55ee75ca1c4e80c128284a907de9e2dd602d9a6a39e660599abea1bf43404` |
| `/tmp/gemini36-recheck-card-8098-20260915.html` | DeepMind Model Card 页面 | 154,640 | `c2e25fec9cc337856c4f612b729f8d1b3985a0627a7234e66091b53dddd9e1c3` |
| `/tmp/gemini36-recheck-ds-8098-20260915.html` | DataCurve DeepSWE 页面 | 268,313 | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| `/tmp/gemini36-recheck-arxiv-title-7890-20260915.html` | arXiv 精确标题检索 | 16,452 | `03b6d5dabe226f9c165a6f725f670f2f846659fcc7308c037b9c0a4fe8db9127` |
| `/tmp/gemini36-recheck-arxiv-all-7890-20260915.html` | arXiv 全文检索 | 63,191 | `2e5c93fde5c302f8a1cd655425d9df2e885c31f4a904318c90f00db6ef372ffa` |
| `/tmp/gemini36-thinking-7890-20260915.html` | Thinking 文档 | — | 本轮抓取；内容用于支持档位表 |
| `/tmp/gemini36-interactions-7890-20260915.html` | Interactions API 文档 | — | 本轮抓取；内容用于支持 3.6 模型表 |
| `/tmp/gemini36-tool-combination-7890-20260915.html` | 工具组合文档 | — | 本轮抓取 |
| `/tmp/gemini36-video-7890-20260915.html` | 视频理解文档 | — | 本轮抓取；内容用于支持 agentic video |
| `/tmp/gemini36-context-caching-7890-20260915.html` | Context caching 文档 | — | 本轮抓取；内容用于支持 4,096 token 下限 |
| `/tmp/gemini36-computer-use-7890-20260915.html` | Computer Use 文档 | — | 本轮抓取 |

