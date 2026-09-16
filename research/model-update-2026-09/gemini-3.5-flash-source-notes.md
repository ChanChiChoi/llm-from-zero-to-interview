# Gemini 3.5 Flash：官方资料与周边技术摘记

核验日期：2026-09-15（2026-09-16 轻量联网复探）。本文只记录已经出现在 Artificial Analysis 与 DataCurve DeepSWE 的 `Gemini 3.5 Flash`，再沿 Google 官方模型页、Model Card、API 文档和论文检索结果追踪面试相关技术。排行榜字段、API 能力和产品行为不等于 Google 已公开了完整参数、架构或训练配方。

## 1. 榜单锚点与证据边界

### Artificial Analysis

- 详情页：[Gemini 3.5 Flash (high)](https://artificialanalysis.ai/models/gemini-3-5-flash)。页面内嵌的第三方 `releaseDate` 为 `2026-05-19`；`medium` 和 `minimal` 是同一 release 的其他 effort 配置，不当作独立模型。
- 当前主配置是 `Gemini 3.5 Flash (high)`。页面字段包括 Intelligence Index `32.9816033695905`、median output speed `221.808004827641 tokens/s`、median TTFT `18.3631341649999s`、context window `1,000,000` tokens、cost per Intelligence Index task `1.5625413241435933`。
- 页面 pricing dataset 记录 cache hit/input/output 约为 `$0.15/$1.50/$9.00` 每百万 token；页面正文的完整评测成本字段约为 `$2172.43`。这些价格、速度、TTFT、指数和成本都是 Artificial Analysis 的第三方配置/provider 字段，不是 Google 的训练事实。
- 当前 Artificial Analysis serialized model object 将该条目标为 `deprecated: true`，并指向 `gemini-3-6-flash`；页面同时提示只继续更新默认 10K 输入工作负载，其他 workload 结果为历史值。这里的 deprecated 是榜单目录状态，不等同于 Google 官方 API 页面已经宣布退役。
- Artificial Analysis 页面把参数字段留空、开放性标为 proprietary；不能从页面指数或目录字段推断参数量、稠密/MoE 结构或训练数据。

### DataCurve DeepSWE v1.1

DataCurve 当前页面快照为 113 tasks、91 repositories、5 languages、4 runs，统一 harness 为 `mini-swe-agent`。原始行如下：

```text
model=gemini-3-5-flash
harness=mini-swe-agent
reasoning_effort=high
config=mini_swe_agent_gemini_3_5_flash_high
n_passed=163, n_attempted=452, n_runs=4
pass_at_1=0.3606194690265487, pass_at_4=0.6371681415929203
mean_cost_usd=3.4467114588691796, median_cost_usd=2.8624087500000015
mean_output_tokens=75730.19290465632, median_output_tokens=68910
mean_input_tokens=7629071.858093127, median_input_tokens=5694082
mean_cache_tokens=6428494.862527716
mean_duration_seconds=1125.1601607057523, median_duration_seconds=973.6770345
mean_agent_steps=105.30376940133037, median_agent_steps=97
median_peak_context_tokens=113979
median_output_tokens_to_pass=64913
```

这是一条“模型配置 + `high` effort + harness + 工具 + 仓库环境 + 任务集 + timeout/预算 + verifier”的系统结果。不能把 `36.0619%` 归因给裸 Gemini 3.5 Flash，也不能把它与 Artificial Analysis Intelligence Index 拼成统一排名；Agent steps 更不能反推 Transformer 层数、参数规模或内部搜索算法。

## 2. Google 官方身份与公开 API 边界

主要来源是 [Gemini 3.5 Flash API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)、[What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5) 和 [Gemini 3.5 Flash Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash/)。

| 字段 | 官方已确认内容 | 证据边界 |
|---|---|---|
| 模型 ID | `gemini-3.5-flash`；稳定版本别名也是该 ID | API 产品字段，不是公开的权重 revision |
| 生命周期 | Model Card 发布于 2026-05-19；API 页标为 stable，latest update 为 May 2026 | 产品更新时间不等于训练完成时间；Artificial Analysis 的 deprecated 另作第三方目录状态记录 |
| 输入/输出 | API 页：text、image、video、audio、PDF 输入；text 输出 | 模态接口支持，不代表每种模态共享同一内部处理路径 |
| token limits | API：输入 `1,048,576`；输出 `65,536`；Model Card 将输出写作 64K | 接口预算不等于有效长上下文检索能力 |
| thinking | 默认 `medium`；支持 `minimal/low/medium/high` | 请求级质量—成本—延迟旋钮，不是公开的内部搜索算法 |
| 工具/能力 | caching、code execution、Computer Use（Preview）、File Search、function calling、Google Maps/Search grounding、structured outputs、URL context | “Supported”只证明协议入口；权限、执行、失败处理和审计仍由平台/宿主负责 |
| consumption | Batch API、Flex inference、Priority inference 均列为支持 | 实际可用性仍受账户、区域和实时计费策略影响 |
| 不支持项 | audio generation、image generation、Live API | 当前模型页能力矩阵；不能泛化到 Gemini 产品家族 |
| 知识截止 | 官方迁移/What's New 页面写为 2025 年 1 月 | 这是模型资料字段；需要新信息时使用 Search Grounding，不能当作实时联网能力 |

Model Card 将 Gemini 3.5 Flash 描述为 Gemini 3 系列原生多模态 reasoning 模型，并称其基于 Gemini 3 Flash reasoning foundation，通过 thinking levels 调节质量、成本和延迟。更重要的是，Model Card 将以下资料全部指向 Gemini 3 Flash Model Card：

- architecture；
- training dataset；
- training data processing；
- hardware；
- software；
- known limitations、acceptable usage 和主要安全政策。

因此当前公开资料不支持 Gemini 3.5 Flash 的独立参数规模、层数、稠密/MoE 结构、注意力变体、优化器、完整训练数据配方或后训练 loss 结论。3.5 的独立价值主要体现为产品级 reasoning/Agent API 变化、模型卡评测和运行时协议，而不是一份公开的新架构报告。

## 3. 3.5 的主要变化：thinking effort 与跨轮 thought preservation

### 3.1 从 high 默认值切换到 medium

Google 的 What's New 页面明确说明：从 Gemini 3 Flash Preview 迁移到 Gemini 3.5 Flash 时，默认 thinking effort 从 `high` 改为 `medium`。官方给出的使用建议是：

- `minimal`：追求响应速度，适合聊天、快速事实回答和简单工具调用；
- `low`：适合低延迟代码/Agent 任务，以及需要少量思考的分析和写作；
- `medium`：默认折中档，适合大多数复杂代码和 Agent 工作流；
- `high`：最大化思考和工具使用，适合困难数学、深度推理和最难的代码/Agent 任务。

官方 Thinking 文档的模型表也确认 3.5 Flash 为 `On (medium)`，支持 `minimal/low/medium/high`。`minimal` 不保证完全关闭思考；复杂任务仍可能发生极少量 reasoning。

面试中应把这个变化回答成运行时成本控制面，而不是“模型换了一套公开推理算法”：默认档位改变会同时影响思考 token、工具轮次、TTFT、最终质量和单位成功成本。对线上路由，应该根据任务难度和失败代价选择 effort，再用成功 artifact、总 token、重试和 verifier 结果做闭环。

### 3.2 新旧参数不能混用

What's New 页面建议从旧的 `thinking_budget` 迁移到 `thinking_level`。`thinking_budget` 仍为向后兼容入口，但同一请求不能同时传 `thinking_level` 与 `thinking_budget`，否则返回 400。这个边界适合面试中的 API 兼容性问题：参数迁移不是简单地把两个字段都传上去，而是需要在客户端做版本化配置和 capability probe。

### 3.3 Thought preservation

Google 明确描述了 3.5 的跨轮 reasoning context：

- Interactions API 自动保留 thoughts，不需要调用方额外处理；
- GenerateContent API 从 Gemini 3.5 Flash 开始，在历史中存在 thought signatures 时使用此前轮次的 reasoning context；
- 调用方需要传递完整且未修改的历史，包括 thought signatures，官方 SDK 会自动处理；
- 这会改善迭代调试、代码重构等多步任务，但可能增加 token 使用量。

这里的 “preservation” 是 API 协议中的加密 reasoning context，不是可读的 chain-of-thought、永久记忆或一次 decode 中的 KV cache。stateless 回放时丢掉 signature 或擅自重写历史，可能导致工具/思考连续性损失。

## 4. Interactions API 与长任务状态

[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) 的支持表明确列出 `gemini-3.5-flash`。一次 interaction 是包含完整任务历史的资源，steps 可以表达 thought、server/client-side tool call、tool result 和 model output；后续请求可以通过 `previous_interaction_id` 继续服务端状态，也可以选择 stateless 模式回放完整历史。

面试时应把状态分成四层：

1. 模型提出的 thought、tool call 和最终 output；
2. API 保存的 interaction/step/signature 协议状态；
3. 宿主真正执行工具、GUI 动作、网络请求和沙箱命令的 runtime 状态；
4. verifier、测试、权限审计和业务数据库保存的外部事实。

`previous_interaction_id` 只解决 API 侧状态连续性，不自动赋予模型文件权限、网络权限、幂等性或回滚能力。stateful 模式便于缓存和审计；stateless 模式便于自托管回放，但必须完整保存相关 steps、`id` 和 `signature`。

## 5. 工具组合、签名与责任分层

Gemini 3.5 Flash 官方迁移页写明它继承 Gemini 3 family capabilities，包括：

- encrypted reasoning context；
- structured outputs with built-in tools；
- multimodal function responses；
- code execution with images；
- built-in tools 与 custom function calling 的 combined tool use；
- image/video/PDF 的 media resolution 控制；
- Gemini 3 thought signatures。

[工具组合文档](https://ai.google.dev/gemini-api/docs/tool-combination)把其中的关键协议称为 tool context circulation：内置工具的上下文可以暴露给同一 interaction 中的自定义工具，工具结果再进入模型上下文。关键字段是：

- `id`：把 `function_call` 与 `function_response` 配对；
- `signature`：出现在 thought、tool call 和 tool result steps 上，作为加密的连续性材料；
- stateful 模式由服务端管理 `id`/`signature`；
- stateless 模式必须在后续请求中完整回传它们，官方 SDK 在传入完整 response object 时可自动处理。

文档还明确：Google Search、Maps、URL Context、File Search、Code Execution、Computer Use 和 custom functions 均有工具上下文支持；组合模式默认使用 validated，而不是 `auto`。这里仍要区分模型意图和系统动作：客户端负责 schema 校验、最小权限、幂等、重试、超时、敏感动作确认和 verifier。

## 6. Computer Use：宿主执行与截图注入防护

API 模型页将 Computer Use 列为 Preview，Computer Use 文档把 `gemini-3.5-flash` 列为 previous stable model supporting computer use。循环是：发送 prompt + screenshot → 模型返回建议动作 → 客户端执行动作 → 获取新 screenshot/结果 → 继续请求。

对面试特别有价值的是，Google 文档明确写出 Computer Use for Gemini 3.5 Flash or later 支持可选的 prompt-injection detection：它检查截图中是否存在隐藏的对抗指令（例如 “Ignore previous commands”），发现后可以阻止执行；该机制是 opt-in，默认值为 `false`。

这不能替代宿主安全边界。生产 harness 仍需维护网站/窗口 allowlist、动作权限、条款/支付/删除等敏感操作确认、截图和动作审计、状态一致性、超时与人工接管。模型返回 click/type 等动作不代表动作已经发生。

## 7. Caching 与多模态上下文预算

[Context caching](https://ai.google.dev/gemini-api/docs/caching) 页面列出 Gemini 3.5 Flash 的 implicit caching 最小输入为 `4,096` tokens。隐式缓存默认开启，stateful（`previous_interaction_id`）与 stateless 模式都支持；Interactions API 只支持 implicit caching，若要显式创建/管理 cache object，需要使用 GenerateContent API。

这应拆成两本账：

- API/计费账：输入 token、cache hit token、thinking token、工具中间步骤和输出 token；
- 推理运行账：单次生成中的 KV cache、显存、并发、TTFT 和 TPOT。

前者可以由服务端隐式缓存策略降低成本，不能直接推出后者的显存占用或 attention kernel。1M 输入窗口也只是接口上限，不等于所有位置的检索准确率相同。

## 8. 视频与多模态能力的证据边界

Gemini 3.5 Flash API 页面确认视频、音频、图像和 PDF 输入，What's New 页面确认媒体分辨率和多模态 function response 等 Gemini 3 family 能力。对于视频处理，需要保守区分：

- 所有 Gemini 模型可以处理视频的 static 路径；
- 当前视频文档的 agentic processing 支持列表明确列出 Gemini 3.8 Flash、3.7 Flash、3.6 Flash 和 Gemini 3.5 Flash-Lite；
- 该列表没有明确列出 Gemini 3.5 Flash，因此本轮不把 `processing_call`/`processing_result` 的 agentic video 结论迁移给 3.5 Flash；
- 3.5 Flash 的视频证据目前足以支持“多模态/静态视频输入”，不足以支持“3.5 Flash 已公开支持 agentic video timeline navigation”。

这条负面边界很重要：相邻版本的产品文档相似，不意味着每个模型都支持同一条内部处理路径。若实际项目需要 agentic video，应以当前 API capability probe 和模型页支持列表为准。

## 9. Model Card 的发布方评测与安全边界

Gemini 3.5 Flash Model Card 给出截至 2026 年 5 月的发布方结果。代表性数字如下：

| 类别 | Benchmark | Gemini 3.5 Flash |
|---|---|---:|
| Coding | Terminal-Bench 2.1（Terminus-2） | 76.2% |
| Coding | SWE-Bench Pro (Public) | 55.1% |
| Agentic | MCP Atlas | 83.6% |
| Agentic | Toolathlon | 56.5% |
| UI control | OSWorld-Verified | 78.4% |
| Expert tasks | Finance Agent v2 | 57.9% |
| Expert tasks | GDPval-AA Elo | 1656 |
| Multimodal | CharXiv Reasoning（无工具） | 84.2% |
| Multimodal | MMMU-Pro（无工具） | 83.6% |
| Multimodal | Blueprint-Bench 2 | 33.6% |
| Long context | MRCR v2（128K average） | 77.3% |
| Long context | MRCR v2（1M pointwise） | 26.6% |
| Reasoning | Humanity's Last Exam | 40.2% |
| Reasoning | ARC-AGI-2 | 72.1% |

这些数字是 Google Model Card 的评测设置，不能替代 DataCurve 的 `mini-swe-agent` 结果，也不能与 Artificial Analysis 指数拼成裸模型排名。尤其要记录 harness、任务版本、工具、是否使用搜索/代码执行、尝试次数和 verifier。

安全表是相对于 Gemini 3 Flash 的自动评测变化（文档明确不是人工评估或 red teaming）：Text-to-Text Safety `-3.9%`、Multilingual Safety `-2.6%`、Image-to-Text Safety `0%`、Tone `+8.9%`、Unjustified-refusals `+0.8% (non-egregious)`。Google 还说明人工 red teaming 下儿童安全达到上线阈值，整体内容安全与 Gemini 3 Flash 相当或更好；Frontier Safety 结论基于 Gemini 3.1 Pro 的评估，认为 3.5 Flash 不会达到已定义 CCL，额外 cyber 测试仍低于 cyber CCL。

这些是发布方安全评估结论，不是安全训练 recipe，也不是对所有真实部署风险的证明。

## 10. 论文、技术报告与官方入口检索

截至 2026-09-15：

- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Gemini+3.5+Flash%22&searchtype=title) 返回 0 篇；
- [arXiv 全文检索](https://arxiv.org/search/?query=%22Gemini+3.5+Flash%22&searchtype=all) 返回 30 篇，均是将 Gemini 3.5 Flash 作为被测模型、工具链组件、外部对照或背景模型的论文，没有检出 Google 发布的 Gemini 3.5 Flash 专属技术报告；
- 代表性外部论文包括：[Harness or Model?](https://arxiv.org/abs/2609.11987)（隔离 Agent coding harness 与模型效应）、[SIR: Self-improving Red-teaming for Compute Use Agents](https://arxiv.org/abs/2608.30207)（Computer Use 间接提示注入红队）、[ARQ](https://arxiv.org/abs/2608.20637)（执行证据驱动的 CodeQL query refinement）、[TempJail](https://arxiv.org/abs/2608.19737)（视频字幕时序 jailbreak）、[Which Source Wins?](https://arxiv.org/abs/2608.17205)（冲突图文中的来源依赖）、[Specification Grounding Drives Test Effectiveness for LLM Code](https://arxiv.org/abs/2607.06636)（规格驱动测试）、[Refused in Chat, Written in Code](https://arxiv.org/abs/2607.03968)（IDE Agent 工作流级 jailbreak）、[LLMs as Teaching Assistants for Mathematics Exam Grading](https://arxiv.org/abs/2607.01247)、[IPO Finance Agent](https://arxiv.org/abs/2606.23032) 和 [Verifiable Benchmarking of Long-Horizon Spatial Biology](https://arxiv.org/abs/2605.28065)。

这些论文可以扩展面试讨论中的评测隔离、Computer Use 安全、规格—测试—verifier 闭环和长任务 Agent 诊断，但不能反向证明 Gemini 3.5 Flash 的内部架构或训练算法。

## 11. 面试知识映射与闭环结论

适合面试的主线：

- **Reasoning runtime**：`thinking_level`、默认 medium、legacy `thinking_budget`、thought preservation 和 thinking/output 共同预算；
- **Agent protocol**：Interactions 的 interaction/step/state、`previous_interaction_id`、stateful/stateless 回放和缓存；
- **Tool protocol**：tool context circulation、`id`/`signature`、内置工具与 custom function calling 的组合，以及模型意图与宿主执行的分层；
- **Computer Use safety**：截图—动作—新截图闭环、prompt-injection detection、最小权限、确认、审计和人工接管；
- **Multimodal boundary**：1M 输入、PDF/audio/video/image 接口支持，静态视频与 agentic video 明确支持列表的差异；
- **Evaluation**：Artificial Analysis、DataCurve 和 Google Model Card 三套数字各自绑定不同配置与评测协议，不能互相拼接。

`Gemini 3.5 Flash` 当前升级为**资料级闭环**：两个排行榜的锚点、Google 官方模型页、What's New、Model Card、Thinking/Interactions/工具组合/缓存/Computer Use 文档、Model Card 评测安全表和 arXiv 定向检索均已具备。由于官方把架构、训练、软硬件资料指向 Gemini 3 Flash，且没有独立公开的参数/训练技术报告，本轮不新增 Gemini 3.5 专属正式架构章节；内容映射到 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving、评测与安全章节。

仍待核验：Gemini 3.5 Flash 专属参数/层/专家结构、完整训练和后训练 recipe、生产 kernel、目标硬件 profiling、线上 tool/agent acceptance rate、agentic video 是否在当前服务版本开放，以及独立 benchmark 复现。

## 12. 本地快照审计标识

下列文件位于临时目录，只用于本轮核验审计，不作为仓库资料依赖：

| 本地快照 | 内容 | 大小 | SHA-256 |
|---|---|---:|---|
| `/tmp/gemini35-aa-recheck-1234-20260915.html` | Artificial Analysis 详情页；8098 抓取结果字节一致 | 3,607,081 | `abdbed0cf8069ae81c272aea386724302a0659235fa5b950980a9a9a0d162ec7` |
| `/tmp/gemini35-ds-recheck-1234-20260915.html` | DataCurve DeepSWE；8098 抓取结果字节一致 | 268,313 | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |
| `/tmp/gemini35-api-recheck-7890-20260915.html` | Google AI Developers 模型页 | 104,969 | `a12584dfd737267ee96a9081908de9a2a80b9096acc2fe572cec19eb5fe264e5` |
| `/tmp/gemini35-card-recheck-7890-20260915.html` | DeepMind Model Card 页面 | 153,914 | `0f629b675520d44d2220108a463a2c079e1249e76f5ad42fde659c7d091b3c4a` |
| `/tmp/gemini35-whatsnew-7890-20260915.html` | Gemini 3.5 migration/What's New | 180,088 | `4ef4f204b26c8d52120893d3b53e84d788b4935f3ce17f8a1c1858817a9b13dd` |
| `/tmp/gemini3-card-7890-20260915.pdf` | 官方 Gemini 3 Flash Model Card（3.5 的依赖入口） | 380,481 | `b2800104a47322d76d80a72bfb128ba7f0bacffc419ae9bb630eae289c761ba0` |
| `/tmp/gemini35-arxiv-title-7890-20260915.html` | arXiv 精确标题检索 | 16,452 | `da2bfa57ea2ffdd7c101b47df4db34876a1ded9e91bf76b6d504e8873e48e608` |
| `/tmp/gemini35-arxiv-all-7890-20260915.html` | arXiv 全文检索 | 158,490 | `4ebb0a76e96bffe92f3bbe776b2de5350f4281e5957dc3465ac7dd2108489f86` |
| `/tmp/gemini35-thinking-7890-20260915.html` | Thinking 文档 | 200,726 | `919ea9da44594e0bf39befc6fc1ab02bc76d10264f0388dc3a622e29914c6d8d` |
| `/tmp/gemini35-interactions-7890-20260915.html` | Interactions API 文档 | 104,613 | `5fb917530fbc98d5adeb82d77df287b7948708aafd32cc29fce5dd1b1e0a392e` |
| `/tmp/gemini35-tool-combination-7890-20260915.html` | 工具组合文档 | 123,795 | `e278ce6c2740f8e227977637b40cafcecc75c2d3e1085b475972afbb3b7e7237` |
| `/tmp/gemini35-video-7890-20260915.html` | 视频理解文档 | 270,983 | `4f9468688570043870a91582bd68520a4ec88d63082f21d225bd424ae4923542` |
| `/tmp/gemini35-context-caching-7890-20260915.html` | Context caching 文档 | 88,880 | `7a5570b08594565e6258020bba1e49a2299efb3b18e56cde5df6169d115d9402` |
| `/tmp/gemini35-computer-use-7890-20260915.html` | Computer Use 文档 | 414,846 | `f13eca02c158b996d559c2875800c4ce952e57a34fe36c31d5614534a991e694` |

2026-09-16 轻量复探：三个代理对 Artificial Analysis 与 DataCurve 均返回 HTTP 200；7890 对 Google API 模型页返回 HTTP 200，8098/1234 对该大页面在超时前未完成。不同代理对不同站点的传输差异不被解释成页面不存在。
