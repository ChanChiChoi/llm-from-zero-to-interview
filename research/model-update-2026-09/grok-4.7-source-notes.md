# Grok 4.7：长轨迹 RL、加密推理状态与上下文压缩

核验日期：2026-09-22。本笔记只研究已经出现在 Artificial Analysis 的 `grok-4-7`。DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_grok_4_7_*` 行，因此不迁移 Grok 4.6 或其他 Grok 版本的 Agent 结果。xAI 官方模型页、发布页和 API 文档用于核验这个榜单锚点并扩展面试知识，不作为新的模型发现入口。

## 1. 两个排行榜的锚点

### Artificial Analysis

[Grok 4.7](https://artificialanalysis.ai/models/grok-4-7) 当前 canonical 展示为 `Grok 4.7 (xhigh)`，页面的第三方 `releaseDate` 为 `2026-09-21`。详情页字段包括：

- Artificial Analysis Intelligence Index：`46.4465506302286`；
- 输出速度中位数约 `38.7732 tokens/s`；
- Intelligence task 的第三方成本约 `$3.7383`；
- context window：`500,000` tokens；
- `proprietary`、非开放权重，参数字段为空，`deprecated=false`。

这些字段是 Artificial Analysis 的目录或 provider 测量，不替代 xAI 官方发布日期、内部参数或训练结论。`xhigh` 是推理配置，不是独立 checkpoint。

9 月 22 日三条代理抓取的详情页逐字节一致，文件大小均为 `3,882,692` bytes，SHA-256 为 `e62290c7cc0d937afb8b8e4f08328c9935e6b40a4db9af3029579052b57baa5c`。同日中文首页快照为 `1,785,528` bytes，SHA-256 为 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`。

### DataCurve DeepSWE

[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 同日三条代理取得相同的 `268,571` bytes 快照，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。当前页面没有精确的 `mini_swe_agent_grok_4_7_*` 行；因此本轮不记录 Grok 4.7 的 Pass@1、Pass@4、成本、输出 token 或 Agent steps，也不把 Grok 4.6 的四档结果迁移过来。

两榜的证据责任不同：AA 的指数、速度和成本属于配置/provider 测量；DataCurve 的分数属于 `mini-swe-agent`、工具、任务集、环境、verifier 和运行次数共同形成的系统结果。没有精确 DataCurve 行时，缺失就是事实边界，不用相邻版本补齐。

## 2. xAI 官方模型与服务字段

主要来源是 [Grok 4.7 模型页](https://docs.x.ai/developers/models/grok-4.7) 和 [Introducing Grok 4.7](https://x.ai/news/grok-4-7)。模型页快照为 `376,912` bytes，SHA-256 为 `c3ddd6b44b97fb4b527096ca69e4d9eacdca99e0e4e44427c9da5b81a615181e`；发布页快照为 `288,007` bytes，SHA-256 为 `af8eda968823b546480818ba27f6edf7f9787b5126c51a9f5ac4568e23da406f`。

官方页面确认的外部合同是：

- model ID 为 `grok-4.7`；text/image input，text output；最大 prompt/context 为 `500K`；
- input `$2/M`、cached input `$0.50/M`、output `$6/M`；超过 `200K` 输入的请求按更高长上下文价格处理；
- 支持 reasoning、function calling 和 structured outputs；Batch API 不支持；
- `reasoning_effort` 为 `low`、`medium`、`high`、`xhigh`，默认 `high`；reasoning 不能关闭；
- 官方限流字段为 `150 RPS`、`50M TPM` 和 `500K maximum prompt`，具体账户/区域限制仍以服务端为准。

本轮经 `10.24.27.134:7890` 刷新的 raw HTML 快照：模型页 `376,913` bytes、SHA-256 `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5`；发布页 `288,007` bytes、SHA-256 `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`。去除 Next.js 包装后，正文与前一份快照一致；页面 hash 变化不作为模型 revision 变化。

官方注册配置对象还出现 `algorithm: grokSlopSafetySystemTurn`。这是服务配置字段，官方没有公开其内部定义；不能从名称推断注意力结构、路由、奖励模型或安全算法。

## 3. 发布方披露的训练与安全方向

xAI 发布页把 Grok 4.7 相对 Grok 4.6 的变化概括为更大的 base model、更长的 reinforcement learning run，以及更困难、偏向需要数小时完成的任务混合。发布页同时强调：

1. 长上下文管理与自验证能力被作为长任务工作流的一部分；
2. 模型原生理解 Grok Bot harness，目标是连续完成代码、研究和知识工作；
3. 发布了新的 safeguard stack，并对部分网络安全合作方开放 invite-only red-team access；
4. LatchBio biosafety benchmark 为 `62.4%`，危险双用途 HackerBench v0.3 prompt 的放行率为 `3.3%`。

这些是 xAI 的发布方描述和发布方评测结果，不等于完整训练 recipe、独立复现或通用安全概率。发布页给出的其他数字也必须绑定其 benchmark、prompt、harness、effort 和统计口径：

| 发布方 benchmark | xAI 展示值 | 必须保留的边界 |
| --- | ---: | --- |
| CursorBench 4.0 | `46.3%` | 发布方测试设置与 coding harness |
| DeepSWE v1.1 | `71.0%` | high effort、任务集、工具与 verifier |
| EEBench | `64.0%` | 发布方评测协议 |
| AA Briefcase v1.1 | `1657` | 指标定义和版本 |
| Terminal-Bench 4.0 | `38.0%` | 终端环境与成功判定 |
| Harvey Legal Agent Benchmark | `19.6%` | 法律任务集和 Agent 配置 |
| HealthBench Professional | `56.7%` | 专业医疗评测设置 |
| GDPval | `1695` | 任务、聚合方法与发布方脚注 |

不能把 xAI 的 DeepSWE `71.0%` 与 DataCurve 其他模型行、AA Intelligence Index 或本地实验拼成裸模型排名。也不能由“更大的 base model”和“更长的 RL run”推断参数规模、MoE/稠密结构、优化器、奖励模型或 rollout 配方。

## 4. 加密 reasoning state：协议状态不是可见 CoT

xAI 的 [Reasoning 文档](https://docs.x.ai/developers/model-capabilities/text/reasoning)当前明确，Responses API 对 `grok-4.7` 每次都会返回 `reasoning.encrypted_content`，即使请求没有显式要求 `include`；服务端工具产生的加密输出也属于需要保留的 reasoning/runtime 状态。后续请求应把相关 item 原样回传。

这带来几个面试边界：

- encrypted content 是 opaque runtime state，不是开发者可读的完整 chain-of-thought；
- 这个默认返回行为不等于客户端必须自行保存明文思维；xAI 仍可能在服务端保存 thinking trace 以便 rehydration。是否把 response 存下来供 `previous_response_id` 使用，由 `store` 决定，而不是由 `include` 或 encrypted field 的默认返回决定；
- 客户端不能编辑、裁剪、重排或把它当作普通摘要；
- 它不是 prompt cache，也不是应用永久记忆，更不是可以直接解释的 GPU KV cache；
- Chat Completions 不提供同样的 ciphertext 字段，Responses 与 Chat Completions 不能用同一状态回放假设；
- harness 必须保存模型 ID、revision/快照、item 顺序、call id、工具回执和错误重试，否则跨轮恢复可能丢失 reasoning 协议状态。

“reasoning 不能关闭”只说明 API 的控制面没有关闭开关，不证明每次请求使用相同的隐藏预算，也不公开内部搜索、停止或验证算法。评测时应分别记录 effort、thinking/reasoning usage、工具轮数、总延迟、失败恢复和单位成功成本。

同一份文档还公开了单独的 summarized reasoning content。Responses 流可以观察 `response.reasoning_text.delta` 与 `response.reasoning_summary_text.delta`，但它们是面向开发者的可见 reasoning/summary 事件，不等于完整隐藏 CoT，也不能替代需要原样回放的 `encrypted_content`。推理模型还不能使用 `presencePenalty`、`frequencyPenalty` 或 `stop`；这些字段属于请求兼容性门禁，不是训练机制证据。`xhigh` 从 Grok 4.6 起可用，不能据此把 effort 档位计为不同 checkpoint。

本轮 `7890` reasoning 页面 raw HTML 快照为 `522,822` bytes，SHA-256 `ce3cd1997308094472d2cce8017eee510cf3441c7be2711029870c9d18d1cc87`。

## 5. Context Compaction 的状态迁移

xAI 的 [Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)使用 `POST /v1/responses/compact`，返回一个单独的 opaque `type=compaction` item。后续请求应将它作为新的上下文起点原样回放。

协议上有四条硬边界：

1. compaction item 不能修改、裁剪或重排；
2. compaction 不能挽救已经超过当前 context limit 的请求，会话必须先处在可处理范围内；
3. 每次请求最多执行一次 compaction；
4. reasoning model 应尽量保留 encrypted reasoning content，以免压缩后丢失此前推理状态。

因此 compaction 不是“删掉历史再继续”，也不是把历史 token 计费清零。Agent harness 需要同时保存 compaction 前的事件账本、压缩触发原因、工具副作用、权限决定、workspace artifact 和压缩后的 opaque item；恢复测试要验证不会重复执行已经提交的副作用。

本轮 `7890` Context Compaction raw HTML 快照为 `521,486` bytes，SHA-256 `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`；正文与前一份快照等价。Compaction item 中的 `encrypted_content` 同样只能存储并原样回传，不能把它改写成应用摘要。

## 6. 工具、结构化输出与 MCP

Function calling 的模型输出只是调用提案，宿主负责授权、schema/业务校验、实际执行、超时、幂等、结果回执和 verifier。Structured Outputs 的 JSON Schema 可以降低解析错误，但不保证工具结果正确，也不替代业务规则、权限和最终 artifact 验收。

Remote MCP 支持 `server_url`、`server_label`、可选的 `server_description`、`allowed_tools`、authorization 和 headers；当前只支持 Streaming HTTP 与 SSE transport。未设置 `allowed_tools` 时，服务器暴露的工具定义默认全部注入上下文；空集合也表示允许全部。xAI 原生 SDK 对应参数名是 `allowed_tool_names` 和 `extra_headers`，而 OpenAI Responses API 当前不支持 `require_approval` 与 `connector_id`。`allowed_tools` 同时有两种作用：减少暴露给模型的工具 schema 上下文成本，以及缩小可调用权限范围。它不是完整的安全边界，宿主仍需做服务器信任、凭据隔离、网络策略、超时、审计和结果验证。

本轮 `7890` Remote MCP raw HTML 快照为 `475,038` bytes，SHA-256 `459b83b7522eaa9139677f9da9760593e56072d52fd63a1883ba6a370a13c705`。

## 7. 论文检索、未知项与闭环状态

[arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.7%22&searchtype=title)返回 `produced no results`；快照为 `/tmp/grok47-arxiv-title-20260922.out`，SHA-256 为 `8f8c4bb0f6c8a0346e28a56864a4aa4d630bdde9c536ee1320be29f84e6e9da6`。因此当前不能声称存在 Grok 4.7 专属论文、公开参数规模、MoE/稠密架构、完整训练 recipe 或生产 kernel。

本锚点当前状态为 **AA 单榜内容专题闭环**：Artificial Analysis 身份、xAI 官方发布/模型/API 资料、每次 Responses encrypted reasoning、服务端工具加密输出、可见 reasoning summary 流、`store`/`previous_response_id` 状态边界、compaction、工具协议、发布方 benchmark 边界和论文负检索均已入库；DataCurve 没有精确行，所以不伪造 Agent 分数。参数、内部架构、完整训练/后训练 recipe、线上 acceptance、硬件 profiling、独立 benchmark 复现和完整安全评测仍待核验。

## 8. 书系映射与面试问题

- 第四册：把“模型能力、effort 配置、provider 测量和 API 协议”分层。
- 第五册：更大 base、更长 RL、困难长任务混合和自验证披露的证据边界。
- 第十六册：reasoning effort、不可关闭的 reasoning、encrypted state 与 compaction。
- 第十七册、第二十册：function calling、MCP、最小权限、幂等、verifier 和长任务状态。
- 第二十四册：opaque state replay、context compaction、缓存/成本/限流与 serving gate。

建议面试追问：

1. Grok 4.7 的 `reasoning_effort` 是四个模型吗？为什么不是？
2. encrypted reasoning state 与可见 CoT、prompt cache、应用记忆和 compaction 有什么区别？
3. 更长 RL run 如何通过固定长度、工具轨迹、验证器和成本账本证明收益不是单纯输出变长？
4. 为什么模型自验证不能替代独立 verifier？
5. `allowed_tools` 如何同时影响最小权限和上下文成本？
6. 为什么不能把 xAI 发布方的 DeepSWE `71.0%` 与 DataCurve 或 AA 分数合并？
7. Grok 4.7 的 `encrypted_content`、可见 reasoning summary 和 `store`/`previous_response_id` 分别解决什么问题？
8. 为什么 MCP 的 `allowed_tools`、`allowed_tool_names`、`extra_headers` 和 `require_approval` 不能直接当作统一安全模型？
