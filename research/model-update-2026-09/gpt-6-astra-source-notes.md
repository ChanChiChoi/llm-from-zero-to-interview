# GPT-6 Astra：官方模型文档核验

初次核验：2026-09-18；最近复验：2026-09-23。研究底稿；正式落点为第六册第 18 章。

已读取 [官方模型文档](https://developers.openai.com/api/docs/models/gpt-6-astra.md)。裸路径 `/models/gpt-6` 返回 Page not found，不能把该路径的导航文字当作模型正文。准确模型标识为 `gpt-6-astra`。2026-09-18 通过 `10.24.27.134:7890` 获取官方 Markdown 页面，快照为 3,812 bytes，SHA-256 `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625`；同一代理获取的相关指南快照记录在本笔记末尾。此前 `10.237.126.170:1234` 与 `10.24.27.134:8098` 对该页分别返回代理侧 `403 Forbidden`，不作为资料缺失证据。

## 已公开的接口事实

官方定位为用于复杂推理、编程、computer use、研究与文档创建的模型。输入模态为文本和图像，输出模态为文本。支持 `low`、`medium`、`high`、`xhigh`、`max` 推理档位。

文档分别列出 1,050,000 上下文窗口、922,000 最大输入和 128,000 最大输出。三者必须分别解释：窗口总容量不能直接当成最大可输入长度；输出预算也不能在总窗口之外任意追加。知识截止日期为 2026-04-30，不能把它当作发布日期。

文档列出 Chat Completions、Responses、Batch 支持，fine-tuning 和 Realtime 不支持。Responses 下可使用 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search。模型自身文本输出与调用图像生成工具是不同层次的能力，不能据工具列表称模型原生输出图像。

## 成本与缓存教学入口

本次页面标示每百万 token 输入 10 美元、缓存输入 1 美元、缓存写入 12.5 美元、输出 50 美元。输入超过 272K token 时，整次请求输入及缓存费率乘 2，输出费率乘 1.5。价格仅代表本次文档快照，正式示例应注明时间，不能作为长期固定配置。

这一规则适合独立讲解长上下文成本：超过门槛后不能只给超出部分加价。还应分别讲解缓存读取、缓存写入、输出与工具费用。实际计费与账户支持仍需核对价格页及接口说明，当前未调用付费 API。

## 尚未公开或本轮未核验

本页没有给出参数规模、稠密或 MoE 架构、预训练数据、优化器、强化学习算法和完整技术报告；不得将其他模型的机制移植为 GPT-6 的事实。首次发布日期、系统卡和模型指南仍需继续核验。页面默认快照与别名均为 `gpt-6-astra`，没有在已读部分提供带日期的固定快照，不能虚构版本锁定标识。

## 扩写方向

1. 第六册与第二十四册：总窗口、最大输入、输出预算与长上下文分段计费的区别，配合可运行预算计算示例。
2. 第十七册、第二十册与第二十二册：模型能力、宿主工具、协议与运行时的分工，用图像工具和 shell 举例逐层讲解。
3. 第七册与第十六册：推理档位如何影响预算与评测可比性；档位名称不能跨厂商直接对等。
4. 第十册与第四册：公开接口事实与内部机制证据的边界，训练研究不从产品功能反推算法。

第六册第 18 章已据上述已核验字段落地，并完成零依赖成本示例运行。官方模型页尚未提供参数规模、训练架构、发布日期或完整技术报告；后续如网络可用，再单独核对 model guide、价格页和版本快照，不把缺失字段写成推测。

## 2026-09-18：官方 Reasoning 与状态协议补证

本轮通过 OpenAI 官方文档补充了模型页之外的运行时知识。以下内容是 API/Agent 协议事实，不是 GPT-6 Astra 的内部网络结构或训练配方：

- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) 明确建议 reasoning 模型优先使用 Responses API；GPT-6 Astra 不支持 `none` effort，支持 `low`、`medium`、`high`、`xhigh` 和 `max`。`reasoning.effort` 是请求级预算旋钮，不能当成跨厂商统一质量等级。
- 同一指南说明，GPT-6 及后续模型可在对话中追加 `configuration_update` input item，动态切换后续响应的 effort；更新项只控制后续生成，不能把这种接口能力解释成模型权重或训练阶段发生变化。
- Reasoning token 与可见输出共同受 context window 和 `max_output_tokens` 约束；预算不足时响应可能是 `incomplete`，且可能在产生可见文本前耗尽。工程评测应记录 reasoning、visible output、工具和恢复成本，而不是只记录最终文本长度。
- [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md) 要求无状态 replay 保留 Responses 的全部 output items，包括加密 reasoning item 和 assistant `phase`；`previous_response_id` 或 Conversations API 是状态引用，不是可读思维链。即使使用 `previous_response_id`，历史输入 token 仍会计费。

## 2026-09-18：缓存、工具搜索与 Compaction 补证

- [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 将缓存解释为可复用前缀的 KV states，而不是保存 token 文本；模型、工具名称/描述/schema/顺序、相关指令或历史前缀变化都可能影响命中。工具加载到上下文末端有助于保留前缀，但“有 session”不等于必然命中。
- [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) 支持 hosted 与 client-executed 两种发现路径。`defer_loading` 延迟函数参数 schema 或 namespace/MCP 细节；清晰的 namespace 描述和较小的工具集合有利于搜索。动态加载工具后，工具集合成为会话状态，移除已加载工具会破坏后续缓存，所有 schema 仍需可信来源校验。
- [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 的 server-side 方式通过 `context_management` 和 `compact_threshold` 触发，流中会返回加密 compaction item；独立 `/responses/compact` 返回的整个窗口是 canonical next context，不能再随意手工裁剪。使用 `previous_response_id` 时不应重复手工 prune。
- 这些机制共同形成长周期 Agent 的状态账本：模型输出、加密 reasoning/compaction item、工具集合、缓存前缀、权限和 artifact 必须分层记录。它们不能被简化成“GPT-6 有永久记忆”或“工具搜索改变了模型架构”。

## 官方快照与证据边界

2026-09-18 通过 `10.24.27.134:7890` 获取的官方 Markdown 快照：

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra.md) | 3,812 | `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625` |
| [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) | 70,253 | `91604df954335250d16e33f3b07ebbfa3e6e2b7f821f0722ad7a69b9d586719a` |
| [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) | 45,605 | `ecced3016c57dcef3115dbda6f73ba6f153f99856f1dadac46d165c12611ab44` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | 14,272 | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |
| [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md) | 28,787 | `c38bcd32ee9ff9904746a5a092c020e30d647d1598418bea93e949cce5021e14` |
| [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) | 39,288 | `9d6c3855a4cb722a98fb364618a852fb436e9822a36fe9b54650f75bbc8d1fd8` |

官方资料仍未公开 GPT-6 Astra 的参数规模、稠密/MoE 结构、注意力变体、训练数据、优化器、完整后训练配方、system card 或专属技术报告。上面的状态协议、缓存和工具字段不能反推这些内容。

## 2026-09-21：模型指南与异步 Agent 协议复验

本轮通过 `10.24.27.134:7890` 重新抓取 OpenAI 官方专属模型指南与周边页面。GPT-6 Astra 模型 Markdown、Reasoning、Conversation state、Prompt caching、Compaction 和 Tool search 的内容哈希与 9 月 18 日快照一致；因此本轮的新增重点是模型指南明确列出的协议能力和提示词/Agent 工程建议，而不是把网页复验误写成模型架构升级。

- [Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md) 明确列出异步工具调用、WebSocket 中途 steering、动态 `configuration_update`、persisted reasoning、compaction、multi-agent orchestration、programmatic tool calling 和 prompt caching。异步工具调用只适用于应用执行的 function/custom tool：模型可以在工具任务运行时继续处理独立部分，应用仍负责执行，并在后续请求中用原始 `call_id` 回传结果；它不改变 hosted built-in tool 的执行责任，也不应与 multi-agent parallel tool calls 混用。
- [Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md) 建议把慢任务登记为应用侧 job，并把 `call_id`、任务句柄、完成状态、超时、重试和幂等性写入 trace。`wait` 工具只是应用自定义的同步函数，不是 Responses 内建工具；工具完成不等于模型已经消费结果，必须再发 continuation 并核对响应 lineage。
- [Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md) 当前只对 GPT-6 Astra 的 Responses WebSocket 开放。客户端发送 `response.steer` 后，服务端先返回 queued/accepted，再在合适的 output item 边界创建 continuation；原响应可能以 `incomplete_details.reason=steered` 结束。steering 不会撤销已经发送的输出、回滚已经发生的副作用或取消已启动的工具，因此宿主仍需保留执行状态和补偿/回滚策略。
- [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md) 是平台安全控制面，不是 GPT-6 Astra 的内部对齐算法。官方文档称它异步检查高后果上下文中的推理与动作；使用 persisted reasoning、WebSocket 或 OpenAI compaction 的 Responses 请求可被识别为连续会话并自动阻断，普通 Responses 可配置 webhook 告警，Chat Completions 不在该监控覆盖内。被阻断时应匹配 `misalignment_policy_violation`，停止自动重试并审计已有工具动作；监控可能漏报或误报，且不会撤销已经完成的动作。
- 模型指南与 [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) 将技能描述、`AGENTS.md` 和任务提示视为上下文路由层：描述应短而精确，复杂流程采用 progressive disclosure，文档按任务需要读取，避免互相冲突的 instructions 和过度脚本化的 recipe；长任务要明确定义完成条件、允许的自主推进范围和何时停止。这些是官方 prompting/harness 建议，不是训练数据或模型内部记忆结构。
- 模型指南的迁移清单还指出：GPT-6 Astra 的 function calling 应使用 Responses API；迁移时不要把 `temperature`、`top_p`、`top_logprobs` 和不支持的 `logprobs` 静默保留；EU data residency 下不能使用 GPT-6 Astra Fast mode，且 Fast mode 对该模型没有 latency SLA。它们属于接口/服务契约，不能混入模型质量结论。

`phase` 的证据边界需要单独保留：Reasoning 文档的现行专节以 GPT-5.5/GPT-5.4 的长任务流为示例，只能证明 Responses assistant `phase` 的通用回放规则，不能据此声称 `phase` 是 GPT-6 Astra 专属新能力。GPT-6 Astra 的专属新能力在本轮有直接页面支持的是 `configuration_update`、async tool calling 和 WebSocket steering。

## 2026-09-21：本轮官方快照

以下均为本轮通过 `10.24.27.134:7890` 读取的官方内容。`10.24.27.134:8098` 与 `10.237.126.170:1234` 对模型页返回 Vercel `403 Forbidden`；这只记录线路/站点防护结果，不作为官方资料不存在的证据。

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [GPT-6 Astra model page HTML](https://developers.openai.com/api/docs/models/gpt-6-astra) | 430,363 | `d2ce52cb3c514d70d124867ff18c1ab14105ae3fc372fead76d4f7532b621864` |
| [Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md) | 16,592 | `aac7e7e1b0bf90b346f5fc4ffd523902279841c60c3adf572452a529bd5b1e5e` |
| [Agents](https://developers.openai.com/api/docs/guides/agents.md) | 5,432 | `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a` |
| [Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md) | 21,237 | `082072ad7a35e2fc744e02ed069b7a3797248bb5a1ac8663d406dd9d1ba001` |
| [Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md) | 12,988 | `d90a3a8e0c8bcb2851ce3374a2fc70c2344aa42aa38d5b631352e6b0e3155d5a` |
| [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md) | 7,634 | `c46dccdc8ad14fc9c36335721187cadf12cc0227b6b46805a5fef36caef19a52` |
| [Fast mode](https://developers.openai.com/api/docs/guides/fast-mode.md) | 8,592 | `a7446692ebe0fce715269e5c9a14af67cce5efdb0033312adbd4ae670b941c23` |
| [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) | 315,959 | `bc0820527b8aaf2eb5caef185fb4316052d34e2c4f299ab2b6e317c67769376b` |

本轮没有获得参数、架构、完整训练/后训练 recipe、system card、专属技术报告或独立 benchmark 的新一手证据；GPT-6 Astra 仍是**内容专题闭环（官方运行时与 Agent 协议补证）**。

## 本地 toy 验收

新增 [`gpt6_agent_protocol_demo.py`](code/gpt6_agent_protocol_demo.py)，不调用 API、不执行真实工具，也不模拟隐藏 reasoning，只用标准库验证三组状态契约：

1. async tool 的任务句柄、原始 `call_id`、重复结果幂等、工具完成与模型消费结果的分离；
2. steering 的 `response.steer.accepted`、`incomplete:steered`、continuation lineage，以及已启动副作用不自动回滚；
3. skill 路由先加载短描述，选中后才读取细节，体现 progressive disclosure 的上下文边界。

2026-09-21 已运行 `py_compile` 和脚本主流程，固定输出验证通过；脚本为 6,621 bytes，SHA-256 `4156ef886bbcfc3a3ac8358d4735c80000da79fc4bfb875d0b65c229952b3afa`。该实验只验证协议账本和失败门禁设计，不能升级为 GPT-6 Astra 的真实 API 行为、延迟或模型质量证据。

## 2026-09-23：7890 当前时点复验与 DataCurve effort 账本

本轮使用用户提供的 `10.24.27.134:7890` 代理重新抓取两个排行榜和 Astra 详情。全部请求返回 HTTP `200`；没有把线路状态、页面动态包装或快照大小变化解释成模型 revision：

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh) | 1,783,572 | `999ead1b8d025a8abccc5d6e879e548544770335863dec1be06b21d99378f8e2` |
| [Artificial Analysis GPT-6 Astra](https://artificialanalysis.ai/models/gpt-6-astra) | 3,997,013 | `c4c040e6708555c1965efaf4eecd3d61b4199ffe337d07ce53970150b79103ce` |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 268,036 | `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1` |

AA 详情的当前配置对象仍是 `gpt-6-astra` + `max`：`releaseDate=2026-09-03`、`knowledgeCutoffDate=2026-04-30`、context `1,000,000`、Intelligence Index `52.673669395513`、median output speed `58.6808088606541 tokens/s`、Intelligence Index cost per task `3.2575003134834164`、输入/输出 `$10/$50`、cache hit `$1`、cache write `$12.5`，`parameters=null`、`proprietary`。这些都是 Artificial Analysis 的第三方/provider 字段；`releaseDate` 不是 OpenAI 发布公告，不能替代专属 snapshot 或训练资料。

DataCurve 页面生成时间为 `2026-09-22T06:27:15.860279+00:00`，任务集为 113 个任务、4 个完整 runs、`mini-swe-agent` harness，当前 Astra 的精确行如下。Pass@1、成本、输出 token 和 Agent steps 都绑定 provider、工具、环境与 verifier，不能迁移成裸模型能力：

| effort | config | passed / attempts | Pass@1 | Pass@4 | mean cost USD | mean output tokens | mean Agent steps |
|---|---|---:|---:|---:|---:|---:|---:|
| `xhigh` | `mini_swe_agent_gpt_6_astra_xhigh` | 335 / 452 | 0.7411504424778761 | 0.8053097345132744 | 4.429117203539823 | 29557.327433628318 | 28.754424778761063 |
| `high` | `mini_swe_agent_gpt_6_astra_high` | 331 / 452 | 0.7323008849557522 | 0.8230088495575221 | 3.9237237909292038 | 26505.853982300883 | 27.424778761061948 |
| `max` | `mini_swe_agent_gpt_6_astra_max` | 331 / 452 | 0.7323008849557522 | 0.7964601769911505 | 7.497827996681416 | 61148.50663716814 | 28.45353982300885 |
| `medium` | `mini_swe_agent_gpt_6_astra_medium` | 329 / 452 | 0.7278761061946902 | 0.8230088495575221 | 3.0754742975663714 | 20361.809734513274 | 26.030973451327434 |
| `low` | `mini_swe_agent_gpt_6_astra_low` | 303 / 452 | 0.6703539823008849 | 0.7964601769911505 | 1.595223439159292 | 10579.53982300885 | 19.537610619469028 |

本轮重新取得的 Astra model page、latest-model guide、Async tool calling、Steering 和 Misalignment monitoring Markdown 分别为 `3,812`、`17,456`、`24,200`、`12,983` 和 `7,830` bytes。Model page、Async tool calling、Steering 和 Misalignment monitoring 与 9 月 21 日对应快照一致；但 `/latest-model/gpt-6-astra.md` 的内容发生了文档路由漂移：9 月 21 日是 Astra 专属指南（`16,592` bytes），9 月 23 日变为 GPT-6 family 指南（`17,456` bytes）。这不是模型 revision 证据，但必须按文档版本记录：

| 页面 | SHA-256 |
|---|---|
| [Model page](https://developers.openai.com/api/docs/models/gpt-6-astra.md) | `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625` |
| [Using GPT-6 Astra，2026-09-21 Astra 专属版本](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md) | `aac7e7e1b0bf90b346f5fc4ffd523902279841c60c3adf572452a529bd5b1e5e` |
| [Using GPT-6，2026-09-23 当前 family 版本](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md) | `8982485767fcefe9b4c588777de67683b4f56abe9c9af1121c9431eb9c305557` |
| [Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md) | `0615053c6f766ef82641639178b560949ad1ea98130d4542395442cb77265e14` |
| [Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md) | `7cc82246149877a788243be0fde39d995b0519819feb58a10386304e0d9cfba2` |
| [Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md) | `f2b674bbf98b65161bf235c2c8a96d3d93114f5eab73e55fa8f53afcdb14af46` |

当前 family guide 的新增/改写点是：Astra、Sol、Luna 被放入同一迁移矩阵；Astra 不支持 `none` effort，而 Sol/Luna 支持；Astra 的 reasoning tool calling 需要 Responses，Sol/Luna 只有在 `reasoning_effort="none"` 时支持 Chat Completions function calling；三者在 EU data residency 下只使用 Standard processing。文档还把 `temperature`、`top_p`、`top_logprobs` 的移除条件写成“非 `none` effort”，并把 GPT-6 family 的 prompt/cache 与迁移建议集中到同一页面。它们是 family-level API contract，不能回写为 Astra 的参数、训练方法或内部架构。

### 失败路径 toy 增量

[`gpt6_agent_protocol_demo.py`](code/gpt6_agent_protocol_demo.py) 已扩展为标准库失败门禁，并通过 `py_compile`、主流程和 `git diff --check`：

1. async registry 拒绝重复 `task_handle`/`call_id`、错误 `call_id`、pending job 的提前消费和已完成结果的内容漂移；相同结果重传仍保持幂等。
2. steering 拒绝同一 response 的重复 steering；断开 WebSocket 后，新连接不能复用旧 steering，恢复必须创建显式 continuation，且不自动重放 steering。
3. `misalignment_policy_violation` 将响应置为 blocked，禁止盲目 retry，同时保留已经启动的外部副作用供人工审计或补偿。

当前脚本为 `11,820` bytes，SHA-256 `f3ef1aea13de9dc432fb427f15e02a6f9b96e2505311a8cd219f54681f62f245`。这仍然只是 `local_protocol_toy`：它验证状态账本和失败策略，不调用真实 API、不执行工具、不解密 reasoning，也不能证明 Astra 的生产事件顺序、服务端实现、延迟或质量。

当前 Astra 状态保持**内容专题闭环（官方运行时与 Agent 协议补证 + 当前时点双榜复验 + local protocol toy）**。参数规模、稠密/MoE 结构、注意力变体、训练/后训练 recipe、system card、专属技术报告、完整权重、生产 kernel、目标硬件 profiling、独立 benchmark 和线上 acceptance 仍为 `unverified`。

## 2026-09-28：7890 官方文档时点复验

- 经 `10.24.27.134:7890` 访问 OpenAI 官方开发者站点成功。当前 GPT-6 Astra 模型 Markdown 返回 HTTP 200、3,812 bytes，SHA-256 `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625`；与已记录快照一致。
- 当前 GPT-6 family guide 返回 HTTP 200、17,456 bytes，SHA-256 `8982485767fcefe9b4c588777de67683b4f56abe9c9af1121c9431eb9c305557`；仍是 9 月 23 日记录的 family 迁移指南，未见内容漂移。[官方 Markdown 页面](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md)
- 开发者博客 Markdown 正文返回 HTTP 200、6,517 bytes，SHA-256 `a1deee4c0b3ca16a20385c27f0692e22d5150fdec81710c04742dd5da29f0468`；文章为 2026-09-11 的 [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)。正文建议精简 skill 描述、采用 progressive disclosure、让 `AGENTS.md` 按任务路由、减少过度脚本化指令，并明确长任务完成条件和安全自主范围。这些知识已在本笔记与第六册第 18 章记录，本次未发现新增方法。
- 本轮只做公开文档读取，没有调用模型 API、下载权重或新增模型。Astra 继续保持内容专题闭环；参数、内部架构、训练配方和专属技术报告仍未公开核实。

## 2026-09-28：官方模型页与定价页补验

用户终端的百度请求证明其网络环境能通过 `10.24.27.134:7890` 出网；当前工作区沙箱内直连该代理先返回 `curl: (7)`，经获批的沙箱外重试后，OpenAI 官方页面均成功返回 HTTP 200。此结果按执行环境和目标 URL 记录，不把沙箱内失败外推为代理全局不可用。

| 官方页面 | HTTP | bytes | SHA-256 |
|---|---:|---:|---|
| [GPT-6 Astra model page（Markdown）](https://developers.openai.com/api/docs/models/gpt-6-astra.md) | 200 | 3,812 | `f45ae813c3f69708e2576328056de14cdf71fec174c875f4b81d236105545625` |
| [GPT-6 Astra model page（HTML）](https://developers.openai.com/api/docs/models/gpt-6-astra) | 200 | 431,650 | `63be1ce7336601a62056eb8436f68d5cee0780aa6f095267c8f811dffe5c6ab5` |
| [OpenAI API pricing（Markdown）](https://developers.openai.com/api/docs/pricing.md) | 200 | 23,475 | `0de899a93b1d9a8c7cf7b3a01ed3553337636c7ff374153821f218a203a203c8` |
| [OpenAI API pricing（HTML）](https://developers.openai.com/api/docs/pricing) | 200 | 552,704 | `ec4a7fd545d1fb71e03de5278d417bb8eae77859df046b0c6d6d36590fe50f1a` |

模型 Markdown 与 9 月 18 日及 23 日记录逐字节一致。当前列出的 default snapshot 仍只有 `gpt-6-astra`，没有日期化、可固定的独立 snapshot ID；不要把排行榜 release date 当成 API snapshot。价格页按每百万 token 给出以下 Astra 费率（次序为 input / cached input / cache writes / output）：

| 计费模式 | Short context | Long context |
|---|---|---|
| Standard | `$10 / $1 / $12.50 / $50` | `$20 / $2 / $25 / $75` |
| Batch | `$5 / $0.50 / $6.25 / $25` | `$10 / $1 / $12.50 / $37.50` |
| Flex | `$5 / $0.50 / $6.25 / $25` | `$10 / $1 / $12.50 / $37.50` |
| Fast mode | `$20 / $2 / $25 / $100` | `$40 / $4 / $50 / $150` |

模型页说明，输入超过 272K token 后整次请求的 input/cache 按 2 倍、output 按 1.5 倍计；pricing 表中的 Astra long-context Standard 行与此规则相符。Batch/Flex 为 Standard 的一半，Fast mode 为适用费率的两倍。价格页是当前公开费率快照，不代表任一账户的实际可用 tier、实时折扣或请求的最终成本；模型指南另明确 Astra Fast mode 没有 latency SLA，且 EU data residency 不支持该模式，不能把更高价格表述成确定的速度保证。

本次没有发现新的模型机制或架构披露；新增的面试要点是区分上下文门槛、计费模式和服务等级，并按 cache read/write、长上下文、批处理/Flex/Fast 分项估算单位成功成本。正式书稿已同步，见第六册第 18 章。
