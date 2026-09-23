# Qwen3.8 Max (0902) 官方资料摘记

核验日期：2026-09-21（runtime recheck）。本笔记只处理已经出现在 Artificial Analysis 与 DataCurve DeepSWE 中的 `Qwen3.8 Max (0902)`。官方资料将它描述为 `qwen3.8-max` 的 upgraded snapshot/service revision；本轮没有证据表明它是新的公开权重、独立基础架构或新的开源 checkpoint。官方产品描述、API 文档和缓存文档用于核验服务行为，不能反向创造新的模型候选。

## 1. 榜单锚点与证据分层

| 来源 | 当前条目/结果 | 证据边界 |
|---|---|---|
| [Artificial Analysis Qwen3.8 Max](https://artificialanalysis.ai/models/qwen3-8-max) | 页面标题为 `Qwen3.8 Max (0902)`；canonical 页面仍为 `qwen3-8-max`，当前 release slug 为 `qwen3-8-max-0902`；release date 字段为 2026-09；当前 Intelligence Index `45.4152084980521`、median output speed 约 `37.275289705095 tokens/s`、cost per Intelligence Index task 约 `$5.408509428374016`、第三方 context 约 `984K` | 这是第三方目录和配置/测量字段；指数、速度、成本、context 显示和 release 字段不能替代 Qwen 官方发布说明，也不能直接推出架构或参数 |
| Artificial Analysis 当前详情快照 | `3,835,865` bytes；SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541` | 固定 2026-09-21 页面身份；后续页面可能继续变化 |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | `mini_swe_agent_qwen3_8_max_xhigh`：258/449，Pass@1 `57.4610%`，Pass@4 `83.1858%`，平均成本约 `$3.7291`，平均输出 `95,075` tokens，平均 Agent steps `111.34`，4 runs | 页面没有 `qwen3_8_max_0902` 精确行；这些数字属于 Qwen3.8 Max 系列的泛化 Agent 配置，不能绑定为 0902 revision 的独立结果 |
| DataCurve 当前快照 | `268,571` bytes；SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` | 只证明当前页面模型集合和配置行；不把泛化 alias 当作 revision 级测量 |

本轮恢复联网后，Artificial Analysis `/zh` 首页三条代理均 HTTP 200，快照为 `1,777,588` bytes、SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`；DataCurve 首页三条代理均 HTTP 200，快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，均逐字节一致。两张首页哈希用于确认榜单入口当前可达，不替代 Qwen3.8 Max 详情页的模型字段。2026-09-18 的旧 AA 详情快照为 `3,607,154` bytes、SHA-256 `e9152a5d81063cbeb45fb21ba621d7eb7da05073235b611f27a344f38c6288ae`，保留为历史页面身份。

旧的 `qwen3-8-max-0803` 已在 Artificial Analysis 页面标记 deprecated，并指向新的 0902 release。这里应使用“同一产品/模型入口的服务 revision 更新”来描述，而不是把 0803 和 0902 计作两个独立基础模型。

## 2. 官方资料与页面身份

- [Qwen3.8-Max-0902 产品页](https://www.qwencloud.com/models/qwen3.8-max-0902)：页面名称为 `Qwen3.8-Max-0902`，alias 为 `qwen3.8-max-2026-09-02`，并明确称其为 `qwen3.8-max` 的 upgraded snapshot。页面 `last-modified` 已更新为 `2026-09-21 11:01:05`；三条代理均取得 `98,992` bytes，但页面含动态 trace/CSS/asset 字段，哈希分别为 7890=`aef93a9892e52504043a81a941a2150a24a5210dd3daf769a85e72dbe5949dad`、8098=`8aa221b29ac0236eee10dc745c374e2c77362077e9d852789a6a9f3fd3c70270`、1234=`39cabb0a6291c417acf6c9904c2c769be89c4739016d41c3ceecfd6f10160e5e`；因此以页面字段和 last-modified 作为正文核验依据，不把动态哈希差异误判为模型差异。
- [Thinking 文档](https://docs.qwencloud.com/developer-guides/text-generation/thinking)：说明 `reasoning_effort`、`thinking_budget` 和 reasoning mode 的调用边界。
- [Function Calling 文档](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)：说明 thinking 模式的 `tool_choice` 约束及 `MultiModalConversation` 接口。
- [Context Cache 文档](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)：说明 explicit、implicit、session cache 的语义、命中和计费边界。
- [Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits)：说明 account+model 聚合、workspace override、月度 TPM tier 和 soft limit；文档快照为 `408,930` bytes，SHA-256 `ba010f2386fbb32c0b69bcbdde64efb520435cefc16475cb030b55253909949e`。

产品页把 0902 的升级方向描述为更强的 coding、工程规模项目、长周期 autonomous development、多工具 Agent 协作和视觉理解，同时保留 1M context、thinking mode 和完整工具生态。这些是官方服务定位和能力说明；在没有专属技术报告、配置文件或公开权重的情况下，不能改写成新的 attention、MoE、训练数据或 kernel 事实。

## 2.1 Provider endpoint 迁移

2026-09-21 重新核验的 Thinking、Function Calling、Context Cache 文档已经把示例 endpoint 从旧的 `dashscope-intl.aliyuncs.com` 迁移到 QwenCloud 的 `maas.qwencloudapi.com`：

| 接入面 | 当前文档示例 |
|---|---|
| OpenAI-compatible mode | `https://maas.qwencloudapi.com/compatible-mode/v1` |
| DashScope API | `https://maas.qwencloudapi.com/api/v1` |
| Realtime WebSocket | 使用 `maas.qwencloudapi.com` 的 realtime endpoint |

这属于 provider adapter/域名和路由配置变更，不是模型能力、权重或架构升级。迁移时 manifest 至少应保存 `provider`、endpoint、region、model alias、resolved revision、API mode、文档核验日期和 tokenizer/template 版本；历史快照中的旧 endpoint 可以保留作时间证据，但不能继续标作当前默认地址。endpoint 变更还应单独回归认证、SSE/WebSocket、重试、限流、缓存命中和工具协议，避免把 transport 故障误判为模型质量回退。

## 2.2 QwenCloud Dynamic Rate Limiting

QwenCloud 的 [Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits) 文档说明，TPM tier 依据账户月度消费动态调整；达到保证 TPM 后，如果平台仍有余量，服务可能继续接受请求，因此保证值是 soft limit 而不是硬性即时拒绝阈值。限流粒度是 `account + model`，同一账户的 workspace/API key 调用量聚合；workspace 可以为某个模型设置独立 quota，未设置时继承 account-level quota。RPM 相对较大，正常使用通常不需要另设 RPM。调整按自然月进行：每月 10 日通知受影响用户，15 日新 tier 生效，tier 只升档或保持不变。

本轮文档表中与当前锚点直接相关的保证 TPM 为：

| 模型服务 | 保证 TPM tier（文档顺序） |
|---|---:|
| `qwen3.8-max-0902` | `1,500,000 / 1,500,000 / 1,500,000` |

这几个数字是 QwenCloud hosted provider 的服务配额，不是 Qwen3.8 Max 的模型吞吐、GPU capacity、上下文窗口、训练能力或 Artificial Analysis/DataCurve 的评测结果。生产 observability 应分别记录 account、workspace、model/revision、guaranteed TPM、observed TPM、429、`Retry-After`、queue latency、并发和单位成功成本；不能用保证 TPM 替代目标硬件 profiling，也不能从该配额表另发现其他模型候选。

## 3. Revision、hosted service 与 open checkpoint 的边界

| 名称 | 本轮可确认的身份 | 不应推出的结论 |
|---|---|---|
| `qwen3.8-max-0902` | Qwen Cloud 的 0902 upgraded snapshot，API alias 为 `qwen3.8-max-2026-09-02` | 不能由 alias 推出新的参数规模、层数、权重格式或独立训练配方 |
| `qwen3.8-max` | Qwen3.8 Max 的 canonical hosted product/model entry | 不能把榜单行、服务能力或产品页字段当作公开本地 checkpoint |
| Qwen3.8-2.4T-A95B | 既有 Qwen3.8 系列公开 MoE checkpoint；Max 产品页说明其 hosted service 基于 A95B | “基于”不等于 0902 权重与开源 A95B 完全相同，也不等于服务端没有额外路由、视觉、工具或模板层 |
| `reasoning_effort` 行 | 同一服务的推理预算/请求配置 | 不能计作新的模型架构或新的 checkpoint |

因此，本轮的正式表述应是“Qwen3.8 Max 0902 服务 revision 更新”，而不是“Qwen 发布了一个新的 Qwen3.8 open model”。Artificial Analysis 和 DataCurve 是锚点发现入口；Qwen Cloud 页面和开发者文档只负责核验已经发现的条目。

## 4. 1M context 与请求预算几何

Qwen Cloud 页面给出的服务边界如下。金额和上限都是本轮页面快照，部署时应再次读取实时文档：

| 模式/字段 | 官方页面字段 |
|---|---:|
| Context | `1M` |
| 普通模式最大输入 | `991K` |
| Thinking 模式最大输入 | `983K` |
| 最大输出 | `131K` |
| 普通输入价格 | `$2 / 1M tokens` |
| 输出价格 | `$6 / 1M tokens` |
| Implicit cache 价格字段 | `$0.25 / 1M tokens` |
| Explicit cache 创建价格 | `$2.50 / 1M tokens` |
| Explicit cache 读取价格 | `$0.17 / 1M tokens` |

这里至少有三个容易混淆的数字：产品 context window、某种请求模式的 maximum input，以及 maximum output。它们不能简单相加后当作可用并发；实际容量还受 tokenizer、reasoning tokens、缓存命中、KV/workspace、工具结果、并发和 provider 限流影响。AA 页面约 `984K` 的第三方 context 字段也不能替换官方的 991K/983K API 输入上限。

在面试或压测中，建议把一次请求分成：用户/系统输入、工具 schema 与结果、reasoning 预算、可见输出、缓存命中前缀、KV 和 Agent workspace。至少固定 0902 alias、mode、effort、输入长度、输出上限、工具数量、缓存状态和并发，分别记录 TTFT、TPOT、p95/p99、显存/缓存字节、失败重试和单位成功成本。

## 5. `reasoning_effort` 与 `thinking_budget`

Thinking 文档明确 `qwen3.8-max-0902` 支持 `reasoning_effort`，可选 `low`、`medium`、`xhigh`，默认值为 `xhigh`。`reasoning_effort` 与 `thinking_budget` 不能同时设置。

这条协议带来三个面试要点：

1. `low/medium/xhigh` 是请求级推理控制，不是三个新模型；排行榜按 effort 展开的多行应归并到同一个服务 revision。
2. `reasoning_effort` 是离散档位，`thinking_budget` 是另一种预算表达；客户端不能同时发送两套控制字段后期待 provider 自行仲裁。
3. 默认 `xhigh` 只说明 API 默认策略，不公开内部 token 分配、停止条件、训练方法或质量保证。实测时要记录实际 reasoning/output token 和任务成功率，不能由档位名称推导算法。

## 6. Thinking 模式与工具选择的协议限制

Function Calling 文档明确：在 thinking 模式下，`tool_choice` 只能使用 `auto` 或 `none`；如果业务需要强制指定一个工具，需要关闭 thinking。Qwen3.8 Max 的多模态调用还使用 `MultiModalConversation` 接口。

这不是“模型不会调用工具”，而是 provider 对 reasoning 与强制工具选择组合施加了协议约束。生产 harness 需要把以下状态分开记录：

- 模型是否开启 thinking，以及发送的是 `reasoning_effort` 还是 `thinking_budget`；
- `tool_choice` 是 `auto`、`none` 还是强制工具，以及 provider 是否接受该组合；
- schema 是否被加载、模型是否产生 call、宿主是否通过权限检查、工具是否真正执行；
- 多模态消息、工具结果、取消、超时、重试和最终 artifact 是否能在同一 trace 中回放。

模型产生一个 tool call 仍不等于工具已执行；schema 校验、权限、确认、幂等、超时和结果可信边界继续由宿主负责。

## 7. Explicit、implicit 与 session cache

Qwen Cloud 的 Context Cache 文档把 Qwen Max 模型列入三类缓存能力：

| 类型 | 适合回答的问题 | 评测/服务时要记录 |
|---|---|---|
| Explicit cache | 应用是否主动创建并复用一个可管理的上下文前缀？ | cache identity、创建/更新/失效、命中、租户和权限 |
| Implicit cache | provider 是否自动识别可复用前缀？ | 命中与未命中、实际计费、前缀稳定性和请求路由 |
| Session cache | 同一个会话的历史状态如何继续复用？ | session identity、过期/切换、并发访问、恢复与隔离 |

文档给出最小缓存长度 `1,024` tokens，并为不同模式规定不同的命中、计费和有效期语义。不要把三者都简化成“KV cache 永久放在 GPU”；它们可能对应不同的 provider 状态、生命周期和账单。也不要只看 cache hit rate：还要记录 cache bytes、命中节省的 prefill、provider 路由、失效后的重算和跨 revision/tokenizer/template 的兼容性。

## 8. 面试中的技术归纳

0902 的高价值知识点不是一项已公开的新 Transformer 算法，而是“模型 revision 如何改变服务契约”：

1. snapshot/revision 与基础模型身份要分开；旧 0803 deprecated、新 0902 alias、canonical entry 和 hosted/open 状态要同时记录。
2. 1M context 不是一条可直接换算吞吐的数字；991K/983K input cap、131K output、reasoning、工具结果和 cache 都要进入预算账本。
3. reasoning effort 与 thinking budget 是互斥的 API 表达；thinking 模式又限制强制 tool choice，客户端迁移必须做 schema 回归。
4. explicit/implicit/session cache 的生命周期、命中和计费不同；cache key、租户、revision、模板和权限必须进入可观测性。
5. coding、长周期 autonomous development、多工具 Agent 和视觉理解是产品定位，应通过固定 harness、工具、环境和 verifier 验收，不能直接当作架构或 benchmark 事实。

## 9. 待核验与书系映射

截至本轮，以下内容没有被 0902 官方页面公开确认：参数规模、层排布、稠密/MoE 结构、注意力变体、训练数据、完整 pre-training/post-training 配方、专属技术报告、生产 kernel、线上 acceptance rate、目标硬件 profiling，以及与 A95B 的精确权重/服务差异。

DataCurve 的泛化 `qwen3_8_max_xhigh` 行可以作为 Qwen3.8 Max Agent 配置的参考，但不能写成 `qwen3.8-max-0902` 的独立 Pass@1。Artificial Analysis Intelligence Index、DataCurve Pass@1、Qwen 官方产品字段也不合并成一个裸模型能力分数。

本轮不新增重复的 Qwen3.8 架构章节：扩展第二十一册第 83 章的模型身份/协议段，并把 cache、tool-choice 和 revision 迁移映射到第二十四册工具 serving 章节、题库、练习、术语、项目和知识图谱。后续若获得 0902 专属技术报告或固定实现，再单独评估是否有必要增加专题。
