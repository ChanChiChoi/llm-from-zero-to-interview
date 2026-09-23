# Claude Opus 4.7 官方资料摘记

核验日期：2026-09-21（2026-09-18 首轮专题收口，2026-09-21 runtime recheck）

## 1. 锚点身份与榜单证据

- 重点厂商：Anthropic。
- Artificial Analysis 发现了两个同一基础模型的配置：[Claude Opus 4.7 adaptive/max](https://artificialanalysis.ai/models/claude-opus-4-7) 与 [Claude Opus 4.7 non-reasoning/high](https://artificialanalysis.ai/models/claude-opus-4-7-non-reasoning)。它们按模型 release 归并，不按 effort 或 reasoning 配置重复计数。
- AA 详情页给出的 release date 为 `2026-04-16`，context 为 `1,000,000` tokens，参数规模未公开；max 配置的 Intelligence Index 为 `40.6897928205908`（estimated），median output speed 为约 `51.1354278542757` tokens/s，价格为 `$5/$25` 每百万输入/输出 token。non-reasoning/high 配置约为 `30.9317`（estimated）。这些是 Artificial Analysis 的配置级字段。
- 当前 DataCurve DeepSWE v1.1 快照没有精确的 `mini_swe_agent_claude_opus_4_7_*` 行，因此不迁移 Opus 4.6、Opus 4.8、Opus 5 或其他 Claude 版本的 Pass@1、成本、输出 token 和 Agent steps。
- 2026-09-21 runtime recheck：Artificial Analysis `/zh` 三条代理均返回 HTTP 200、`1,777,588` bytes、SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情页为 `3,798,671` bytes、SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。详情页当前标记该第三方目录条目为 deprecated；这与 Anthropic 官方模型页的 `Active (legacy)`、建议迁移到 Opus 5 并不矛盾，属于两个来源的生命周期语义，不能把 AA 状态改写成 API 立即不可用。
- 同一时点 DataCurve 三条代理均返回 HTTP 200、`268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致；仍没有精确的 `mini_swe_agent_claude_opus_4_7_*` 行。

## 2. 官方资料

- [Anthropic 发布页：Introducing Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7)
- [Claude Opus 4.7 System Card](https://www.anthropic.com/claude-opus-4-7-system-card)
- [Claude Opus 4.7 模型页](https://platform.claude.com/docs/en/models/opus-4-7/overview.md)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)
- [Vision](https://platform.claude.com/docs/en/build-with-claude/vision)
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)

2026-09-21 固定的 Anthropic 官方 Markdown 资源快照：模型页 `11,882` bytes / SHA-256 `d7ca2e12d25c77f32dda9e59d2e6f1a201a71cc48fc93316c6d2c105188b759b`；Task budgets `681,298` bytes / `24415c74c9748a3a717937e2e1ea072347c89aab3af7d6c99c18d393204548bc`；Effort `655,024` bytes / `11e3a4544cee5fa386b43b42bee4f4afa13370dcb3b5e18fe2a3309050748d00`；Vision `879,844` bytes / `c5215fe3c6010c7d07ac63df6941e0b3ae8ff819a4572fdd7373c61713e35d31`；Compaction `2,129,206` bytes / `7d0f6506f25f18eec8ef43edda742188d3fa73ffe5557380ed05c43778ffdd47`。这些哈希用于本轮证据追溯，不把文档页面大小当作模型指标。

发布页与 System Card 是模型和安全的一手资料；API 文档说明可观察的运行时协议。Anthropic 官方模型页当前把 Opus 4.7 标为 legacy，并推荐迁移到 Opus 5；这改变的是生命周期状态，不抹去其作为榜单锚点的历史身份。

## 3. 已确认的模型与运行时边界

### 3.1 版本、窗口与 effort

官方模型页给出 API model ID `claude-opus-4-7`、1M context、128K synchronous max output、Batch beta 300K output、`$5/$25` 每百万输入/输出 token，以及 adaptive thinking 和默认 `high` effort。Opus 4.7 引入 `xhigh`，位于 `high` 与 `max` 之间；`xhigh` 是长时 coding/Agent 工作的行为档位，不是新的权重或独立模型。

`effort` 是控制行为和推理投入的信号，不是严格 token 上限。`max_tokens` 仍然是单次响应的硬上限；task budget 则是跨一个完整 Agent loop 的 advisory 总预算。容量规划应分别记 input、thinking、可见输出、tool call/result、重试、compaction 和缓存命中，不能把 1M context、128K output 和 effort 相加成并发容量。

### 3.2 Task budgets：把 Agent loop 作为预算对象

Anthropic 文档把 task budget 定义为一个完整 agentic loop 的 advisory token budget，覆盖 thinking、tool calls、tool results 和 output。请求通过 `output_config.task_budget` 与 `task-budgets-2026-03-13` beta header 启用，预算对象包含 `type: "tokens"`、`total`，以及可选的跨请求 `remaining`。

模型能看到服务端注入的倒计时，并据此在预算接近耗尽时收束任务、总结发现或结束循环。API response 的 usage 不返回 remaining-budget 字段，客户端若要记账必须累加各请求的新 token，不能把每次重发的完整历史重复扣除。server-side compaction 不会重置当前 turn 已消耗的预算；但不同 user turn 会建立新的预算语义。

这提供一个重要的 Agent 设计分层：`effort` 决定每一步愿意投入多少，task budget 约束整条工作链能做多少，`max_tokens` 约束单次响应最多生成多少。三者分别对应 step policy、loop policy 和 request safety cap。

### 3.3 更新 tokenizer 与成本迁移

Opus 4.7 发布页说明它使用更新的 tokenizer。同样的输入迁移到 4.7 后可能映射为更多 token，发布方给出的经验范围约为旧版本的 `1.0-1.35x`，具体取决于内容类型；这不是模型参数量变化的证据。Agent 迁移评估不能只复用旧模型的 token budget，应同时测 tokenizer 后的输入 token、缓存命中、thinking/output token、端到端成本和成功率。

### 3.4 高分辨率视觉输入

官方 vision 文档说明 Claude 4.7 及之后模型自动使用 high-resolution tier：最长边上限 `2576 px`、最多 `4784` visual tokens；其他模型的 standard tier 为 `1568 px`、`1568` visual tokens。图片以 `28x28` patch 计为 visual token，超过上限会按规则缩放；computer/browser tool result 的截图超限时应由应用在回灌前自行缩放。

这不是“视觉编码器结构已公开”的结论，而是 API 输入契约。高分辨率会提升密集截图、图表、坐标和文档的可读性，也可能把同一请求的输入 token 和延迟推高约三倍，因此应把像素尺寸、缩放策略、视觉 token、坐标映射和成本放进 serving trace。

### 3.5 网络安全 safeguards 的责任边界

发布页和 System Card 将 Opus 4.7 描述为在 Project Glasswing 背景下用于验证新 cyber safeguards 的较弱模型，并称服务会自动检测、阻断被判定为禁止或高风险的网络安全请求；合法安全研究者可通过 Cyber Verification Program 申请验证路径。这里公开的是部署与安全控制面，不是模型内部“学会了一个安全算法”的证明。

生产系统仍须把模型能力、实时策略分类、用户/组织授权、沙箱、网络隔离、审计和人工升级分层。自动拦截的误报/漏报、提示注入、工具副作用和安全研究例外都需要单独评测，不能只引用发布方安全摘要。

## 4. 发布方评测与证据边界

Anthropic 发布页给出 coding、long-context、vision、BigLaw、Cursor、Terminal-Bench、research-agent 等发布方或合作方评测，并强调 Opus 4.7 相比 Opus 4.6 在长任务、一致性、工具错误和视觉分辨率上改善。BigLaw、合作方 benchmark、Anthropic 内部 agentic coding 曲线和 System Card 安全结果必须保留原始 harness、effort、资源分配和评测主体，不能与 AA Intelligence Index 或 DataCurve Pass@1 拼成一条排名。

本轮没有找到 Anthropic 为 Opus 4.7 单独公开的完整参数规模、稠密/MoE 结构、注意力变体、训练数据、pre-training/post-training recipe、生产 kernel、目标硬件 profiling 或线上 tool-call acceptance rate。System Card 提供安全评估和部署边界，但不能替代这些工程细节。

## 5. 面试主线与待核验项

1. 为什么 `effort`、task budget 和 `max_tokens` 要建三本账？应回答它们分别控制 step、loop 和 request，且工具结果与 thinking 也消耗整条 Agent loop 的预算。
2. 为什么迁移 Opus 4.6 到 4.7 要重新测 token 成本？应回答 tokenizer 变化可能让相同输入变成 `1.0-1.35x` token，且更高 effort 的后续 Agent turn 可能产生更多输出。
3. 高分辨率视觉支持的工程代价是什么？应回答 `2576 px/4784 visual tokens` 提升密集视觉证据，但会改变输入预算、坐标映射、延迟和缓存命中。
4. task budget 为什么不能由客户端简单地用每次 payload token 相减实现？应回答服务端倒计时只计算模型在当前 agentic turn 新处理的内容，重发历史不会重复扣除；compaction 也不重置已经消耗的预算。
5. 自动 cyber safeguard 是否等于模型已经获得安全权限？应回答它是部署策略与授权门禁，仍需要沙箱、网络隔离、审计、人工升级和独立误报/漏报评测。

仍待核验：参数规模、内部架构、完整训练 recipe、tokenizer 词表与压缩细节、task budget 对不同 provider 的实际实现差异、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现。当前状态为“内容专题闭环（复用 Agent/tool 与 inference serving 章节）”，不新增独立 Transformer 架构章节。
