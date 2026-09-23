# GPT-5.5：高效推理、Phase 回放与工具/缓存运行时资料摘记

核验日期：2026-09-15。本笔记只把 GPT-5.5 在 Artificial Analysis 与 DataCurve DeepSWE 中的条目作为模型锚点，再沿 OpenAI 官方模型页、模型专属指南和 API 文档追踪面试相关技术。官方资料没有公开 GPT-5.5 的参数规模、网络结构或完整训练报告；运行时字段和产品自述不能反推出这些内部事实。

## 1. 榜单锚点与归并口径

- Artificial Analysis 的 2026-09-09 历史快照记录了 `GPT-5.5` 的 `xhigh`、`high`、`medium`、`low` 和 `Non-reasoning` 配置，以及 `GPT-5.5 Pro (xhigh)`；这些页面日期均为 2026-04-23。对应条目保存在 [`model-inventory.md`](model-inventory.md) 中。
- 同一 Artificial Analysis 历史表还出现了 `GPT-5.5 Instant` 的 2026-05-05 和 2026-06-25 条目。本轮重新抓取两个详情页并检查 OpenAI 精确 `gpt-5.5-instant` 路径；它们仍不能自动映射成 OpenAI API 的 `gpt-5.5`，产品名称也不当作同一基础模型的正式别名。
- DataCurve DeepSWE v1.1 的 2026-09-03 快照包含 `gpt-5.5` 的 `xhigh` 配置：Pass@1 67%（页面区间 ±6%）、平均成本约 `$7.23`、输出 token 约 46K、82 个 Agent steps。快照没有 `gpt-5.5-pro` 主表行。

DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, effort, mini_swe_agent, tools,
           task_set, verifier, timeout, retry, context_policy, provider)
```

因此上面的 67% 不是 GPT-5.5 的裸模型分数。它绑定了 `xhigh`、统一声明的 `mini-swe-agent`、任务集、工具、执行环境和 verifier；成本、输出 token 与 Agent steps 也不能单独解释质量。Artificial Analysis 的不同 effort 行同样应归并到一个基础模型，再保留配置维度。

## 2. 官方模型身份与接口字段

主要依据是 [GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md)、[GPT-5.5 Pro 模型页](https://developers.openai.com/api/docs/models/gpt-5.5-pro.md) 和 [OpenAI Models](https://developers.openai.com/api/docs/models.md)。

| 模型 ID | 官方定位 | 默认 snapshot | 输入/输出 | context / max output | effort |
|---|---|---|---|---:|---|
| `gpt-5.5` | 复杂专业工作的旗舰模型 | `gpt-5.5-2026-04-23` | text/image -> text | 1,050,000 / 128,000 | `none`、`low`、`medium`、`high`、`xhigh`，默认 `medium` |
| `gpt-5.5-pro` | 使用更多计算、追求更稳定精确答案的 GPT-5.5 版本 | `gpt-5.5-pro-2026-04-23` | text/image -> text | 1,050,000 / 128,000 | `medium`、`high`、`xhigh`，默认 `high` |

GPT-5.5 官方模型页还确认：

- `gpt-5.5` 支持 Chat Completions、Responses 和 Batch；官方专属指南建议涉及 reasoning、工具调用或多轮状态的应用优先使用 Responses API。
- `gpt-5.5-pro` 只支持 Responses 和 Batch，不支持 Chat Completions。它的请求可能需要数分钟，官方建议对长请求评估 background mode。
- `gpt-5.5` 文本 token 价格快照为输入 `$5`、缓存输入 `$0.5`、输出 `$30`/每百万 token；超过 272K input tokens 时，standard、batch 和 flex 的整次 session 按输入 2 倍、输出 1.5 倍计价。
- `gpt-5.5-pro` 文本 token 价格快照为输入 `$30`、输出 `$180`/每百万 token，模型页明确写出不提供 cached input discount。价格、区域加价和可用性属于文档快照，不应写成永久费率。
- 两个模型页都公开 reasoning token 支持、文本/图像输入和文本输出；这不等于公开了原始 chain-of-thought，也不说明图像生成能力。
- GPT-5.5 的知识截止字段为 2025-12-01。知识截止日期是模型字段，不能当成模型发布日期；榜单日期和官方 snapshot 也承担不同证据责任。

## 3. GPT-5.5 官方指南明确提出的能力变化

[Using GPT-5.5](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.5.md) 给出的内容主要是模型行为与应用调优指导，而不是架构论文。适合面试的知识点如下。

### 3.1 推理 token 效率是可测的产品属性

官方指南声称，GPT-5.5 在相同 reasoning effort 下可以用比前代更少的 reasoning tokens 达到较强结果。这里能得到的工程结论是：模型迁移不能只比较最终准确率，还要记录 reasoning tokens、可见输出 tokens、端到端延迟和单位成功成本；但文档没有给出统一比例，也没有说明该效率来自某个公开的蒸馏、稀疏激活或搜索算法。

`reasoning.effort` 是预算/行为控制，不是内部参数量。GPT-5.5 默认 `medium`，低延迟任务可以先测 `low`，只有评测显示质量收益时才升到 `high` 或 `xhigh`；官方还提醒，更高 effort 在停止条件弱、工具权限开放或指令冲突时可能带来过度思考、无效搜索甚至质量回退。

### 3.2 Outcome-first prompting

GPT-5.5 专属指南建议把 prompt 从“规定每一步怎么做”改成“规定完成什么以及如何判断完成”：

```text
目标：完成一次有证据的资料核验。

成功标准：
- 每个结论都绑定允许的权威来源；
- 缺少证据时返回待核验项，而不是猜测；
- 最终结果包含结论、来源、已执行动作和阻塞项；
- 达到停止条件后结束工具循环。
```

这不是让模型无约束地自由发挥。安全边界、审批、输出字段和禁止副作用仍应作为硬约束；可以省略的是那些模型能够自行选择、且产品并不要求固定的过程步骤。面试时应说明：prompt 的职责是定义目标、约束、证据和停止条件，harness 的职责是执行权限、循环、重试和验证。

官方指南还建议尽量使用 Structured Outputs 表达输出 schema，而不是在自然语言 prompt 中重复描述 JSON 结构。Structured Outputs 负责 schema adherence；它不等于事实正确，也不替代 verifier。

### 3.3 工具选择与工具面治理

官方把 GPT-5.5 定位为适合大型工具面、长任务和多步服务流程，并建议把工具何时使用、输入、权限副作用、重试安全性和常见错误写入工具描述。工具搜索（tool search）允许把不常用函数的参数 schema 延迟到模型决定需要它时再加载：

1. 初始请求只暴露 namespace/server 的高层描述或可搜索工具元数据。
2. 模型发起 hosted 或 client-executed `tool_search`。
3. 被发现的工具定义被注入上下文，随后才进行 function call。
4. 工具执行、权限校验、结果回传和错误处理仍由应用或宿主负责。

官方 [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) 还说明，把工具放在上下文末端有助于保留前面的缓存前缀；延迟 schema 能减少初始上下文和成本，但引入了工具发现步骤、schema 版本管理和权限审计状态。它解决的是工具面上下文膨胀，不是让模型获得新的主机权限。

### 3.4 图像 detail 是精度、token 与坐标稳定性的旋钮

GPT-5.5 专属指南称，`image_detail` 未设置或为 `auto` 时采用更接近 `original` 的处理：最多保留约 10,240,000 像素或 6,000 像素维度；显式 `high` 的上限约为 2,500,000 像素或 2,048 像素维度；`low` 更强调上下文效率，会对超过 512 像素维度的图像更激进地缩放。实际图像 token 还受模型的 patch 规则和输入尺寸影响，不能把这些像素上限直接当作 token 数。

面试上可以从 OCR、表格小字、computer use 坐标三个场景解释取舍：`low` 省上下文但可能损失细节，`original` 保留空间信息但提高输入 token、延迟和成本；坐标任务还要处理缩放后的坐标映射，不能因为设置了 `original` 就假定图像永远不被缩放。

### 3.5 可见输出长度与推理质量分离

GPT-5.5 的 `text.verbosity` 控制最终可见回答的长度和风格，专属指南建议需要简洁输出时尝试 `low`。`verbosity` 与 reasoning effort 是不同旋钮：降低最终回答长度不等于禁止模型推理，增加 reasoning effort 也不保证最终答案更长。评测应分别记录 reasoning tokens、visible output tokens 和完成率。

## 4. Responses 状态、`phase` 与 reasoning 边界

### 4.1 `previous_response_id` 与手工回放

GPT-5.5 官方指南建议多轮应用使用 `previous_response_id` 管理状态；如果应用为了无状态或 Zero Data Retention 手工传回历史，则应保留 Responses 返回的相关 output items。Conversation state 文档特别提醒，reasoning-enabled 请求默认会返回可回放的加密 reasoning item，手工 replay 时不能只挑最终文本，还要保留 output items 以及 assistant item 的 `phase`。

这几个概念要分开：

- 对话历史是应用可见的消息和工具结果。
- reasoning item 是可携带但 opaque 的运行时状态，不是公开的原始思维链。
- `previous_response_id` 或 Conversation 是服务端/协议层的状态引用。
- prompt cache 是输入前缀复用，不能等同于 reasoning state 或 KV cache。

当前官方 Reasoning 文档明确把 `reasoning.context=all_turns` 和默认跨轮 reasoning 复用写在 GPT-5.6 家族上；GPT-5.5 专属资料只要求正确使用 `previous_response_id`、output replay 和 `phase`，本轮不把 GPT-5.6 的 `all_turns` 语义迁移成 GPT-5.5 的专属能力。

### 4.2 `phase` 是 Agent 协议字段

对于 GPT-5.5 的长任务或工具密集型 Responses 流程，官方建议在 assistant message 上保留 `phase`：

| `phase` | 作用 |
|---|---|
| `commentary` | 工具调用前的 preamble 或中间进展更新 |
| `final_answer` | 已完成工作的最终答复 |

如果应用手工 replay assistant items，应原样传回 `phase`；丢失它可能让 preamble 被模型或宿主误判为最终答案。`phase` 改善协议状态机和 harness 行为，不是模型网络结构，也不是新的 reasoning loss。

### 4.3 Compaction 解决窗口增长，不保证无损记忆

GPT-5.5 支持 compaction。长期 Agent 可以在上下文接近阈值时让服务端或 standalone endpoint 返回更短的 opaque compaction item，再把它接入下一轮。应用应保留已完成动作、活动假设、ID、工具结果、未解决阻塞和下一目标；压缩后仍需评估工具错误、重试、成功率、延迟、token 和缓存命中。

压缩后的上下文不是完整 transcript 的可读替代物，也不是 prompt cache。它减少窗口压力，可能改变后续请求的前缀，因此应把 compaction 前后的质量和 cache hit 分开测量。

## 5. GPT-5.5 与 GPT-5.6 的 Prompt Caching 差异

这是 GPT-5.5 最适合面试的工程对比之一。官方 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 的模型差异表把 GPT-5.5 与 GPT-5.6 分开：

| 维度 | GPT-5.5 / 5.5 Pro | GPT-5.6 及以后 |
|---|---|---|
| 隐式断点 | 通常按 2,048 token 间隔 | 最新 eligible message 末尾 |
| 显式断点 | 不支持 | 支持 `prompt_cache_breakpoint` |
| cache key | 稳定 `prompt_cache_key` 用于优化路由 | 自动路由；key 主要用于分组计量 |
| 最小可缓存前缀 | 随请求设置变化 | 1,024 个可见输入 token |
| cached token 统计 | 排除隐藏 token，并向下取整到 128 的倍数 | 报告精确 eligible boundary |
| TTL 参数 | `prompt_cache_retention`，支持 `24h` | `prompt_cache_options.ttl`，支持 `30m` |
| 写入计费 | 没有额外 cache-write charge | 写入为未缓存输入费率的 1.25 倍 |

因此从 GPT-5.6 迁移回 GPT-5.5 时有几个容易答错的点：

- 不能把 GPT-5.6 的 explicit breakpoint 配置直接带回 GPT-5.5；5.5 不支持该断点字段。
- 不能用 GPT-5.6 的 `prompt_cache_options.ttl` 代替 5.5 的 `prompt_cache_retention`。
- 5.5 的长共享前缀仍应放在前面、动态内容放在后面，并用稳定 `prompt_cache_key` 帮助路由，但 key 不保证命中。
- cache hit 统计不能直接拿两个模型的数字比较；5.5 还受 128-token rounding、工具/图像/effort/verbosity 等请求设置影响。
- GPT-5.5 Pro 的模型页不提供 cached input discount，即使协议涉及缓存，也不能按普通 GPT-5.5 的 `$0.5`/百万 token 计算。

超过 272K 的 whole-session 价格规则与缓存命中是两个账本：命中减少可计费输入的成本，但不能绕过长上下文的整次请求价格档位。一个用于实验的最小记录表应包含模型 snapshot、effort、verbosity、工具 schema、缓存 retention/key、输入长度、cached tokens、reasoning tokens、延迟和最终成功率。

## 6. GPT-5.5 Pro 的后台执行

GPT-5.5 Pro 使用更多计算，官方模型页明确提醒部分请求可能需要几分钟；[Background mode](https://developers.openai.com/api/docs/guides/background.md) 提供异步响应、轮询 queued/in-progress 状态、取消和可选流式恢复的协议。面试时要把它拆成三层：

1. 模型层：生成更高 effort 的 reasoning 和最终输出。
2. API 层：`background: true` 返回可轮询的 response 状态。
3. 应用层：保存 response ID、超时/取消策略、用户通知、重试幂等性和最终 artifact。

后台模式解决连接保持和请求超时，不等于模型本身异步，也不自动解决工具副作用的幂等、授权和审批问题。

## 7. 评测设计与面试回答抓手

GPT-5.5 官方指南要求迁移时同时比较 accuracy、token consumption 和 end-to-end latency；结合 DeepSWE 快照，建议至少固定以下字段：

```text
model + snapshot + effort + verbosity + image_detail
+ tool catalog + tool search mode + phase replay
+ prompt/cache policy + compaction policy + harness
+ task set + verifier + timeout + retry + provider
```

可直接用于面试的问答：

1. **GPT-5.5 的“更高效推理”是否说明用了稀疏 MoE？** 不能。官方只给出同 effort 下 reasoning token 更少的行为描述，没有公开参数、专家或训练机制。
2. **为什么 GPT-5.5 推荐 outcome-first prompt？** 因为模型可以自行选择达到目标的路径；应用仍要明确成功标准、证据要求、权限边界和停止条件。
3. **`phase` 和 reasoning effort 的关系是什么？** effort 控制推理预算，`phase` 标记 assistant 输出处于中间 commentary 还是 final answer；前者是模型配置，后者是状态协议。
4. **为什么 GPT-5.5 的缓存配置不能照抄 GPT-5.6？** 两者的隐式断点、显式断点、key、TTL、最小长度、统计和写入费率不同。
5. **Tool search 的收益和代价是什么？** 它延迟加载不常用工具 schema，减少初始上下文；代价是多一个发现状态、schema 版本与权限审计面，且发现失败会影响后续工具调用。
6. **为什么 Pro 要用 background mode？** Pro 可能需要数分钟；background 让应用轮询 response，但应用仍负责连接恢复、取消、重试和副作用幂等。
7. **DeepSWE 的 GPT-5.5 67% 能否与 GPT-5.6 的 73% 直接比较？** 只能作为该快照中固定 harness 配置的观测，不能忽略 effort、model revision、工具、任务集、verifier 和运行环境后下裸模型结论。

## 8. 未公开内容与闭环状态

截至本次核验，在 GPT-5.5/GPT-5.5 Pro 官方模型页、GPT-5.5 专属指南、OpenAI Models 索引以及 Reasoning、Prompt caching、Compaction、Tools、Tool search、Images/Vision、Structured Outputs、Conversation state 和 Background mode 文档中，没有找到 GPT-5.5 专属的：

- 参数总量、激活参数、层数、专家数、稠密/MoE 结构或注意力变体；
- 预训练数据规模、数据配比、优化器、学习率策略、硬件集群或训练成本；
- SFT、RLHF/RLVR、DPO、verifier、蒸馏或安全后训练的完整配方；
- 可独立复现的 GPT-5.5 技术报告、system card 或完整 benchmark 报告。

因此 GPT-5.5 的“高效 reasoning”“更准确工具选择”“更强 instruction following”只能作为官方模型指导和待复现实验假设。不能从 `reasoning.effort`、`phase`、tool search、compaction、image_detail 或 1.05M context 反推内部算法。

当前状态：**资料级闭环**。理由是 GPT-5.5 已有两个排行榜的可追溯锚点、官方模型页、Pro 页、专属指南、API 周边文档和研究笔记；仍缺少独立架构/训练报告，因此不新增 GPT-5.5 专属正式章节。

书系映射：

- 第十六册：reasoning tokens、effort、verbosity 与“更多推理不必然更好”的评测。
- 第十七册：Responses、tool search、Structured Outputs、工具 schema 和 `phase` 回放。
- 第二十册：模型、协议、harness、后台执行、compaction、重试和副作用边界。
- 第六册与第二十四册：1.05M context、272K whole-session 价格门槛、图像 detail、prompt cache 和单位成功成本。
- 第七册：固定 snapshot/effort/tools/harness/task/verifier 的公平评测。

## 9. GPT-5.5 Instant：榜单配置与官方身份负证据

本节只处理已经出现在 Artificial Analysis 的 `GPT-5.5 Instant (May 2026)` 与 `GPT-5.5 Instant (June 2026)`；它们不能因为名称相近就并入官方 `gpt-5.5` API model ID。

| 条目 | Artificial Analysis 快照/字段 | OpenAI 官方身份 | 当前状态 |
|---|---|---|---|
| GPT-5.5 Instant (May 2026) | [`gpt-5-5-instant-05-26`](https://artificialanalysis.ai/models/gpt-5-5-instant-05-26)，2026-05-05；3,548,855 bytes，SHA-256 `ed410b6cecbd8eb1dcaf65715547771bcb7435e444f6c5ba7ce815e16614e51e`；AA Intelligence Index `22.6863693000789`（页面四舍五入为 23）；400K context；页面标记 deprecated | [GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md) 只确认 `gpt-5.5`，不存在可直接对应的 Instant model ID | 榜单级关联；不与 `gpt-5.5` 或 June revision 合并 |
| GPT-5.5 Instant (June 2026) | [`gpt-5-5-instant-06-26`](https://artificialanalysis.ai/models/gpt-5-5-instant-06-26)，2026-06-25；3,847,333 bytes，SHA-256 `9c6fa94b9c42a89eb7c8f59bad08a30c168bf0860971fbb666c4be47e6a05599`；AA Intelligence Index `26.0135173401368`（页面四舍五入为 26）；约 130.9925 output tokens/s；400K context；页面标记未 deprecated | 直接访问 [OpenAI `gpt-5.5-instant` 模型页](https://developers.openai.com/api/docs/models/gpt-5.5-instant.md) 返回 HTTP 404（9 bytes）；可获取的官方页面仍是 `gpt-5.5` | 下一锚点；尚未形成模型专属资料级闭环 |

两个 AA 页面都把对象描述为 text/image → text、reasoning 配置，知识截止为 2025-08-31，价格字段为 input `$5/M`、output `$30/M`；这些是第三方目录/测量与 API 价格引用，不是 OpenAI 对 Instant model ID、内部结构或训练机制的确认。DataCurve 当前快照只有 `gpt-5.5` 的 `xhigh` base 行，没有 `gpt_5_5_instant` 精确 `mini_swe_agent` 行，因此不能迁移 67% Pass@1、成本、输出 token 或 Agent steps。

官方 `gpt-5.5` 模型页确认的是 `gpt-5.5-2026-04-23`、1,050,000 context、128,000 max output、`none`--`xhigh` effort、text/image → text 和 Responses/Chat Completions/Batch；这些家族级资料只作为对照，不能改写成 Instant June 的独有能力。当前没有找到 `GPT-5.5 Instant` 专属官方博客、模型卡、技术报告、system card、参数、训练 recipe、公开权重或独立 benchmark。正确结论是：**AA 已发现一个可追踪的 June 配置，但官方身份与 API 端点未建立映射；不新增架构章节，不把 GPT-5.5 base 的技术或评测迁移给 Instant。**

## 10. 来源清单

### 榜单发现

- [Artificial Analysis GPT-5.5](https://artificialanalysis.ai/models/gpt-5-5)、[GPT-5.5 high](https://artificialanalysis.ai/models/gpt-5-5-high)、[medium](https://artificialanalysis.ai/models/gpt-5-5-medium)、[low](https://artificialanalysis.ai/models/gpt-5-5-low)、[Non-reasoning](https://artificialanalysis.ai/models/gpt-5-5-non-reasoning) 和 [GPT-5.5 Pro](https://artificialanalysis.ai/models/gpt-5-5-pro)：2026-04-23 榜单配置日期。
- [Artificial Analysis GPT-5.5 Instant (May)](https://artificialanalysis.ai/models/gpt-5-5-instant-05-26) 与 [June](https://artificialanalysis.ai/models/gpt-5-5-instant-06-26)：2026-09-20 详情快照和哈希见第 9 节；June 是当前下一锚点，但仍是榜单级关联配置。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：2026-09-03 v1.1 快照中的 `gpt-5.5` xhigh 行；完整快照解释见 [`deepswe-snapshot-notes.md`](deepswe-snapshot-notes.md)。

### OpenAI 官方资料

- [GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md) 与 [GPT-5.5 Pro 模型页](https://developers.openai.com/api/docs/models/gpt-5.5-pro.md)
- [Using GPT-5.5](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.5.md)
- [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)
- [Images and vision](https://developers.openai.com/api/docs/guides/images-vision.md)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Background mode](https://developers.openai.com/api/docs/guides/background.md)
- [GPT-5.5 Instant 精确路径](https://developers.openai.com/api/docs/models/gpt-5.5-instant.md)：2026-09-20 通过 `7890` 检查返回 HTTP 404；这是官方身份负证据，不是模型能力或训练资料。

本轮还实际检查了 `https://developers.openai.com/api/docs/guides/latest-model?model=gpt-5.5` 的查询路由；当前返回的通用页面 front matter/正文实际指向 GPT-6 Astra，因此没有把它当作 GPT-5.5 专属指南。证据以可直接获取的 `latest-model/gpt-5.5.md`、模型页和明确的 API 指南为准。
