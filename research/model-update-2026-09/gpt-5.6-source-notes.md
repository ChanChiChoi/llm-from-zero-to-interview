# GPT-5.6：推理状态、缓存与 Agent 运行时资料摘记

核验日期：2026-09-23；当前时点榜单复验：2026-09-23（保留 2026-09-21/22 的历史测量）。本笔记只把 GPT-5.6 在 Artificial Analysis 与 DataCurve DeepSWE 中的条目作为锚点，再沿 OpenAI 官方模型页、API 文档和开发者博客追踪周边技术。官方当前没有公开 GPT-5.6 的参数规模、网络结构或完整训练报告；运行时字段不能反推出这些内部事实。

## 1. 榜单锚点与归并口径

- Artificial Analysis 的 2026-09-09 历史快照记录了 `GPT-5.6 Sol`、`GPT-5.6 Terra` 和 `GPT-5.6 Luna` 的 `max`、`xhigh`、`high`、`medium`、`low` 及 `Non-reasoning` 配置，榜单日期均为 2026-07-09。对应条目保存在 [`model-inventory.md`](model-inventory.md) 中。
- DataCurve DeepSWE v1.1 的 2026-09-03 快照包含 `gpt-5.6-sol` 的 `max` 配置：Pass@1 73%（页面区间 ±3%）、平均成本约 $6.46、输出 token 约 60K、61 个 Agent steps。
- 同一快照包含 `gpt-5.6-luna` 的 `max` 配置：Pass@1 67%（页面区间 ±4%）、平均成本约 $0.61、输出 token 约 73K、102 个 Agent steps；快照没有 `gpt-5.6-terra` 主表行。
- DeepSWE 的观测对象是 `F(base_model, model_revision, effort, mini_swe_agent, tools, task_set, verifier, timeout, retry, context_policy, provider)`。因此上面的数字是模型配置、统一声明的 `mini-swe-agent`、工具、任务环境和 verifier 的联合结果，不能直接当成基础模型排行榜。

三个名字按同一 GPT-5.6 家族的不同服务档位归并；`Sol/Terra/Luna` 是官方模型层级，`max/high/...` 是请求配置，DeepSWE 的 Agent steps 是运行轨迹字段。Artificial Analysis 的 `2026-07-09` 是第三方榜单日期，不单独证明 OpenAI 的发布日期。

## 2. 官方模型身份与接口字段

主要依据是 [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol.md)、[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md)、[GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) 和 [OpenAI Models](https://developers.openai.com/api/docs/models.md)。

| 模型 ID | 官方定位 | 价格快照（每百万 token） | GPT-5.6 家族关系 |
|---|---|---:|---|
| `gpt-5.6-sol` | 复杂专业工作的旗舰档 | 输入 $4；缓存输入 $0.4；输出 $20 | 无后缀 `gpt-5.6` 别名指向 Sol |
| `gpt-5.6-terra` | 智能与成本平衡 | 输入 $2；缓存输入 $0.2；输出 $12 | 约对应较早 GPT-5 系列的 mini 层 |
| `gpt-5.6-luna` | 成本敏感、高并发工作负载 | 输入 $0.2；缓存输入 $0.02；输出 $1.2 | 约对应较早 GPT-5 系列的 nano 层 |

三个官方模型页共同确认：

- 输入模态为文本和图像，输出模态为文本；“支持 image input”不等于原生图像输出。
- context window 为 1,050,000 tokens，maximum input 为 922,000，maximum output 为 128,000。三者是不同约束，不能简单相加后当成任意请求都能使用的容量。
- 页面把知识截止字段写为 2026-02-16；它是知识字段，不是发布日期。
- 三者支持 reasoning tokens。官方 `reasoning.effort` 值为 `none`、`low`、`medium`、`high`、`xhigh` 和 `max`，默认 `medium`。Artificial Analysis 的 `Non-reasoning` 标签是第三方配置名称，不能自动等同于 API 的 `none`。
- Chat Completions、Responses 和 Batch 端点受支持；Realtime、Assistants、fine-tuning、embedding、图像生成、音频和视频等端点不在三个模型页的支持列表中。
- 页面列出 streaming、structured outputs、function calling、file search、image input、web search 和 prompt caching；Responses 工具列表包括 web search、file search、image generation、code interpreter、hosted shell、apply patch、skills、computer use、MCP 和 tool search。

工具列表说明的是模型与 API 的兼容入口。工具执行、文件和网络权限、审批、沙箱、超时、重试与回滚仍属于宿主运行时，不能因为模型页列出 `hosted_shell` 或 `computer_use` 就推断模型拥有任意主机权限。

## 3. 长上下文计费与服务档位

三个模型页都说明：输入超过 272K tokens 时，整次请求的输入/缓存输入价格乘 2，输出价格乘 1.5；不是只给超出部分加价。缓存写入按未缓存输入费率的 1.25 倍计费。Sol 页还声明当前价格的促销期至少持续到 2026-11-21；价格随账户、平台和文档版本变化，不能写成永久费率。

一个可用于面试的成本分解是：

```text
C_request = C_uncached_input + C_cached_input + C_cache_write
            + C_output + C_tool_calls
```

它只是账本分层，不是完整平台计费公式。实际核算还要固定模型档位、输入是否超过 272K、缓存命中/写入、工具类型、Batch 与服务层级。长上下文的“能放下”与“单位成功成本更低”是两个实验问题。

## 4. Reasoning mode 与 effort 是两个旋钮

官方 [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) 明确区分：

1. `reasoning.mode` 选择 `standard` 或 `pro`。GPT-5.6 在 Responses API 中支持两者，`standard` 是默认值。
2. `reasoning.effort` 选择模型在所选 mode 内投入多少推理，支持 `none`、`low`、`medium`、`high`、`xhigh` 和 `max`，默认 `medium`。
3. `mode` 与 `effort` 相互独立。`pro` 会聚合更多模型工作，按所选模型的标准 token 费率计费，但通常消耗更多 token、延迟更高。

因此 `max`、`pro` 和 `Sol/Terra/Luna` 不能混成一个“模型强弱”字段。公平实验至少要把 `model_id`、snapshot、mode、effort、工具、上下文策略和任务集一起固定。

Reasoning tokens 不在 API 中返回原始文本，但会占用上下文并作为 output tokens 计费。`max_output_tokens` 覆盖 reasoning tokens、可见输出和非可见格式化 token；预算太低时，响应可能在生成可见答案前以 `incomplete` 结束。官方建议刚开始试验时为 reasoning 与最终输出预留至少 25,000 tokens，并通过 usage 中的 `output_tokens_details.reasoning_tokens` 观察实际开销。

## 5. GPT-5.6 的 persisted reasoning

GPT-5.6 的一项关键运行时差异是：在多步会话中，模型默认会把可用的、兼容的早先 reasoning 带入后续 sample。`reasoning.context` 可以写成：

| 值 | 语义 |
|---|---|
| `auto` | 使用所选模型的默认行为；省略时等价于 `auto` |
| `current_turn` | 只让当前 turn 的 reasoning 可用，不把早先 turn 的 reasoning 带入下一个 sample |
| `all_turns` | 将可用且兼容的早先 reasoning item 带入后续 sample；GPT-5.6 支持该值并默认使用 |

这里的“带入”不是公开 chain-of-thought，也不是永久记忆。reasoning item 保持 opaque，只有同一模型家族之间可复用；`gpt-5.6-sol`、`gpt-5.6-terra` 和 `gpt-5.6-luna` 可以互相复用，但不能把 GPT-5.6 的 reasoning 直接带到 GPT-5.5 家族。

`all_turns` 只有在请求能访问旧 response items 时才生效，例如使用 `previous_response_id`、Conversation 或手工回放完整历史。第一次请求没有早先 reasoning，因此 `current_turn` 与 `all_turns` 没有行为差异。

在 Responses API 的 function calling 中，官方建议把最近一次 function call 返回的 reasoning items 与函数结果一起回传；如果连续调用多个函数，则从最近一个 user message 之后的 reasoning、function call 和 function call output 都要保留。原因是模型的下一步不只依赖函数结果，也依赖仍在进行的推理状态。手工裁剪历史时，最近一个 user message 到 function output 之间的这些 item 应原样保留。

面试时要区分四种状态：可见对话历史、opaque reasoning state、prompt cache prefix 和 compaction item。它们分别服务于语义回放、跨轮推理连续性、输入前缀复用和上下文压缩，不能用“上下文记忆”一个词代替。

## 6. Prompt caching 的 GPT-5.6 变化

官方 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 将 GPT-5.6 及以后模型单独列出：

- 可缓存前缀的最小长度为 1,024 个可见输入 token；隐藏的 OpenAI system content 不计入这个最小值。
- 支持 implicit 与 explicit 两种缓存。implicit 会把最新 eligible message 的结尾作为断点；explicit 使用 `prompt_cache_breakpoint` 标记稳定内容的结尾。
- `prompt_cache_options.mode: "explicit"` 只使用开发者选的断点。没有显式断点时不会写入缓存；断点之后的动态内容按未缓存输入计费，不产生缓存写入费。单个请求最多四次缓存写入。
- 缓存读费率为标准未缓存输入费率的 0.1 倍，写入费率为 1.25 倍。稳定前缀能被重用时，写一次再读一次的等价成本为 1.35 倍，低于重复处理两次的 2 倍。
- `prompt_cache_options.ttl` 的唯一支持值和默认值为 `30m`；最近一次写入或复用后至少 30 分钟具备复用资格，平台可能保留更久。
- GPT-5.6 的缓存路由由 OpenAI 自动处理，`prompt_cache_key` 不再是提高命中率所必需的；它仍可用于按用户、租户或 workspace 分离缓存计量。
- compaction 会替换较早的对话前缀，因此压缩后的第一次请求可能降低 cache hit。更短的输入仍可能省钱，但应同时比较 `cached_tokens`、`cache_write_tokens`、延迟和总成本。

这给出一个容易被面试追问的迁移陷阱：从旧模型迁移到 GPT-5.6 时，不能只保留长 prefix 就假设会命中缓存。若默认断点落在变化内容之后，应在稳定 developer content 后放显式断点；旧的 `prompt_cache_retention` 迁移为 `prompt_cache_options.ttl`。缓存复用也不能与 KV cache 混为一谈：prompt cache 是服务侧前缀复用，KV cache 是一次生成过程中的注意力状态。

## 7. Responses、工具和 Agent harness

官方 [Using tools](https://developers.openai.com/api/docs/guides/tools.md) 以 Responses API 作为工具编排入口，列出 function calling、web search、file search、remote MCP、skills、shell、computer use、tool search 和 Programmatic Tool Calling。GPT-5.6 模型页列出的工具支持说明它们可以接入该模型，但不改变工具的执行责任边界。

### 7.1 Tool search 与 Programmatic Tool Calling

- tool search 允许把不常用的函数定义延迟加载；开发者可以把函数放入 namespace，并给不常用的工具设置 `defer_loading: true`，模型需要时再搜索并加载定义。官方工具页说明 GPT-5.4 及以后模型支持 tool search，因此 GPT-5.6 属于支持范围。
- 官方把 tool search 分成两种协议路径：[hosted tool search](https://developers.openai.com/api/docs/guides/tools-tool-search) 由 OpenAI 在同一个 response 中搜索已声明的函数、namespace 或 MCP server；[client-executed tool search](https://developers.openai.com/api/docs/guides/tools-tool-search) 则由模型发出 `tool_search_call`，应用按项目/租户状态完成搜索，再以相同 `call_id` 回传 `tool_search_output`。前者的 `execution` 为 `server` 且 `call_id` 为 `null`，后者的 `execution` 为 `client`，返回的工具集合才进入后续可调用集合。
- 两种路径都会把已加载工具追加到上下文末端，以尽量保留已有缓存前缀；这降低 schema 的初始 token，但把工具发现、schema 校验、版本、权限和可信来源变成了新的会话状态。`tool_search` 或 `defer_loading` 只改变发现/加载，不授予文件、网络、shell 或业务写权限。
- Programmatic Tool Calling 让模型生成 JavaScript 来编排一组工具调用，减少“模型调用一次、应用回传一次”的往返；安全执行、工具权限、失败处理和最终结果回传仍由宿主负责。
- `parallel_tool_calls`、工具 schema、MCP approval 和 function output 都会改变 Agent 的观测轨迹。评测时应将它们记录在配置中，而不是只记录模型名称。

### 7.2 Agents API、Agents SDK 与 Responses API

官方 [Agents](https://developers.openai.com/api/docs/guides/agents.md) 将三者分工为：Agents API 由 OpenAI 管理 agent loop、Codex harness、session configuration、turns 和 items；Agents SDK 把 agent loop、工具、handoff、部署、存储、审批和 runtime 集成控制交给应用；Responses API 直接暴露模型 response、历史和工具循环，应用自行组合 Agent。Agents API 文档还列出 automatic context compaction、multi-agent orchestration、programmatic tool calling 和 MCP 支持。

这三个入口不是三个模型，也不是同一 session 的不同 SDK 名称：Agents API session、Agents SDK session、Responses conversation 和 sandbox 是不同资源，状态保存、清理、工具执行环境和审计责任不同。面试中应先选运行时所有者，再讨论模型能力：OpenAI 托管 harness、应用自持 loop，或者直接管理 response item。

这组 API 是运行时层的选择，不是三个不同模型。面试回答时应把模型能力、Responses 协议、Agent harness、执行环境和业务应用拆成层次：模型提出计划/工具意图，协议携带 output items，harness 维护状态和循环，执行器决定真实权限，业务系统负责审批与落库。

### 7.3 Compaction 与长期任务

官方 [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 提供两条路径：

- server-side compaction：在 Responses 请求中设置 `context_management` 的 `compact_threshold`，达到阈值后服务端在流中压缩并返回 encrypted compaction item；
- standalone `/responses/compact`：应用主动提交完整上下文，得到下一轮应原样传入的 compacted context。

压缩 item 会携带继续任务所需的状态和 reasoning，但本身是 opaque、不可供人阅读的摘要。stateless input-array chaining 要把 output items（包括 compaction item）接回下一轮；使用 `previous_response_id` 时传新的 user message 即可让服务端携带它。独立 compact endpoint 返回的窗口不能再手工裁剪，否则可能破坏 canonical context。

Compaction 解决的是上下文窗口和长期运行问题，不等于把全部历史无损保存，也不等于 prompt cache。应用仍需测量压缩前后的成功率、工具错误、重试、token、延迟和缓存命中。

Prompt Caching 文档进一步明确：`context_management` 触发 compaction 后，渲染上下文从第一个变化位置开始可能不再匹配旧 cache prefix；工具搜索虽把新工具追加到末尾以保留前缀，但 compaction 会替换较早对话。于是长期 Agent 的成本账本至少要同时记录 `cached_tokens`、`cache_write_tokens`、compaction 次数、压缩后输入长度和任务成功率，不能只看“窗口变短了”。

## 8. 官方开发者博客给出的系统层证据

### 8.1 Harness 可能改变模型结果

[Codex as a platform: build on the open agent harness](https://developers.openai.com/blog/codex-as-a-platform.md) 明确指出，一个可用 Agent 还需要上下文维护、工具调用、进度、失败处理、审批和跨轮执行，这一层被称为 harness。文章给出一个 GPT-5.6 Sol 的工程例子：加入 retained reasoning 和 context compaction 后，ARC-AGI-3 的分数从 13.3% 提升到 38.3%，输出 token 减少六倍。

该数字应写成“特定评测、模型、harness、状态策略和 compaction 组合下的发布方工程结果”。它说明系统封装可能显著影响结果，不能写成 GPT-5.6 Sol 裸模型能力提升，也不能从它推断内部训练算法。

文章还把 Codex harness 的职责列为 conversation state、streaming execution、tools、sandbox/approval policy 和跨轮工作，并强调应用界面、业务上下文、MCP 工具及最终审批仍可由产品拥有。这是“模型 API 与 Agent 产品之间的边界”面试题的直接材料。

### 8.2 Skills、Shell 与 Compaction 的组合

[Shell + Skills + Compaction](https://developers.openai.com/blog/skills-shell-tips.md) 将长期 Agent 拆成三个可组合原语：

- Skills 是可版本化、按需加载的流程和模板；元数据承担路由，完整 `SKILL.md` 在触发后进入上下文。
- Shell 是执行层，可在托管容器或本地 runtime 中运行命令、安装依赖并生成 artifact；网络应受组织级与请求级 allowlist 限制。
- Compaction 是长期运行的上下文管理层，支持自动流内压缩和显式 compact endpoint。

文章建议复用同一 container、传 `previous_response_id` 并把 compaction 当成长任务默认原语。这些是 Agent runtime 的工程模式，不是 GPT-5.6 的内部架构；适合映射到本项目的 skills、sandbox、长任务 checkpoint、缓存和评测章节。

### 8.3 Responses API 的协议视角

[From prompts to products: One year of Responses](https://developers.openai.com/blog/one-year-of-responses.md) 将 Responses 描述为“reasoning and acting”的结构化循环：输入证据后模型可以调查、调用工具，再返回结果；response 可以包含 tool calls、structured outputs 和中间 item，而不只是最终字符串。文章还强调 Responses 在跨调用时保留 reasoning state。

面试时可以把一次调用画成：

```text
input -> reasoning item -> tool call -> tool result -> reasoning continuation -> final output
```

这里的中间 item 是可审计的协议记录；它仍不是对外暴露的原始 chain-of-thought。应用要保留 tool call、tool result、错误、重试和最终 artifact，才能解释长任务为什么成功或失败。

## 9. 未公开内容与负面结论

截至本次核验，在上述官方模型页、模型目录、Reasoning/Agents/Tools/Prompt Caching/Compaction 文档和开发者博客中没有找到 GPT-5.6 专属的：

- 参数总量、激活参数、层数、专家数、稠密/MoE 结构或注意力变体；
- 预训练数据规模、数据配比、优化器、学习率策略、硬件集群或完整训练成本；
- SFT、RLHF/RLVR、DPO、verifier、蒸馏或安全后训练的完整配方；
- 可独立复现的 GPT-5.6 技术报告、system card 或训练 benchmark 报告。

因此不能因为 API 有 `reasoning.effort`、`pro`、tool search、compaction 或 1M context，就声称 GPT-5.6 使用了某个特定搜索算法、RL 目标、MoE 结构或 verifier。可以确认的是 OpenAI 公开了这些请求协议和运行时机制，以及它们在特定系统评测中的工程影响。

本次也实际检查了 `latest-model.md?model=gpt-5.6`：返回文档的 front matter 将 `latestModelInfo.model` 解析为 `gpt-6-astra`，正文也是 GPT-6 Astra 指南。因此它不能作为 GPT-5.6 专属模型指南使用；GPT-5.6 事实以三个模型页和 Reasoning/Tools/Prompt Caching 等明确写出 GPT-5.6 的段落为准。

## 10. 面试提问与回答抓手

1. **为什么 `pro` 和 `max` 不能直接比较？** `pro` 是执行 mode，`max` 是 effort；前者改变模型工作量聚合方式，后者控制推理预算，必须联合实验。
2. **GPT-5.6 的 reasoning state 是什么？** 是 API 可携带但 opaque 的 reasoning item；通过 `previous_response_id` 或历史回放跨轮使用，不等于可见 CoT、永久记忆或 KV cache。
3. **function call 后为什么要回传 reasoning item？** 下一次生成需要延续最近 user message 后仍有效的推理状态；只回传函数结果可能打断多工具规划。
4. **GPT-5.6 prompt caching 的新点是什么？** 1,024 token 最小可缓存前缀、显式断点、30 分钟 TTL、自动路由，以及与旧模型不同的写入/读取费率和断点语义。
5. **Compaction 会不会让缓存永远命中？** 不会。压缩会改变前缀，可能降低第一次命中；它减少输入规模，但要重新测命中、成本和质量。
6. **为什么 DeepSWE 分数不能当模型裸分？** 评测结果还包含 revision、effort、mini-SWE-agent、工具、任务、verifier、超时、重试、上下文策略和 provider。
7. **tool search 解决什么瓶颈？** 工具 schema 本身会占上下文；延迟加载减少初始 token 和工具选择噪声，但引入了工具发现、schema 版本和权限审计的新状态。

## 11. 书系映射与当前状态

当前状态：**资料级闭环**。理由是已经具备两个排行榜的锚点记录、三个官方模型页、明确的 GPT-5.6 Reasoning/Prompt Caching 资料、Compaction/Tools/Agents 文档和官方开发者博客工程证据；但没有独立 GPT-5.6 架构/训练报告，因此不新增 GPT-5.6 专属正式章节。

- 第十六册：reasoning tokens、standard/pro、effort 与 persisted reasoning。
- 第十七册：Responses output items、function calling、tool search、Programmatic Tool Calling、MCP 和 approval。
- 第二十册：model/protocol/harness/executor/application 分层、compaction、sandbox、checkpoint、trace 与 verifier。
- 第六册与第二十四册：1.05M context、922K input、128K output、272K whole-request price threshold、prompt cache 和单位成功成本。
- 第七册：固定 model/revision/mode/effort/tools/harness/task/verifier 的公平评测。
- 第五册与第八册：把 `reasoning`、`pro`、tool use 和 compaction 记录为运行时控制或系统行为，不能反推后训练 loss、RL 配方或安全训练细节。

完整来源与候选状态已同步到 [`source-index.md`](source-index.md)、[`model-inventory.md`](model-inventory.md)、[`inventory-interpretation.md`](inventory-interpretation.md)、[`plan_v2.md`](../../plan_v2.md) 和 [`progress_v2.md`](../../progress_v2.md)。

## 14. 2026-09-23 `7890` 当前时点复验与状态回放审计

本轮仍只沿两个排行榜中已经确认的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。使用 `10.24.27.134:7890` 获取 Artificial Analysis 详情和 OpenAI 官方 Markdown；页面可达性恢复不改变模型身份，也不把页面测量漂移解释成模型 revision。

### 14.1 两榜快照与测量漂移

- Artificial Analysis [GPT-5.6 Luna (max)](https://artificialanalysis.ai/models/gpt-5-6-luna) HTTP 200；当前快照为 `3,996,959` bytes、SHA-256 `4246b96416b4aaf4f8bbcacf72f8c45787944e24f02c59c6e3db1fed4de93c41`。页面仍记录 release `2026-07-09`、Intelligence Index `37.3244239690841`、1M context、约 `$0.20/$1.20` input/output；当前 median output speed 为 `142.271568203031 tokens/s`，cost per Intelligence Index task 为 `0.17829726152289094`。这些都是 AA/provider 当前采集时点字段，速度相对 9 月 22 日的 `158.728370482714` 只记为测量漂移。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 当前快照仍为 `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。精确 `mini_swe_agent_gpt_5_6_luna_max` 行仍是 attempted `448`、passed `301`、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 Agent steps `101.68080357142857`、median peak context `201647`。它仍是 `mini-swe-agent + tools + task environment + verifier` 的系统结果。

### 14.2 官方文档当前哈希

本轮通过 `7890` 取得的实际 Markdown 页面如下。模型页、Tools 和 Compaction 与既有快照一致；Reasoning 和 Prompt caching 的当前内容/字节数以本轮实际响应为准，解释仍沿用既有协议结论。

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) | 3,744 | `1f425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0` |
| [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) | 70,315 | `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251` |
| [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) | 47,098 | `69680fbec38e31a7abc8b0e33a6582b0aae29687e8de98404789337a55b55e2e` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | 14,272 | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |
| [Using tools](https://developers.openai.com/api/docs/guides/tools.md) | 33,282 | `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` |

没有由这些页面推出新的参数规模、内部架构、训练/后训练 recipe、system card 或生产性能结论。当前仍应把以下字段当作可观察运行时合同：`reasoning.context` 的 `current_turn/all_turns`、同家族兼容的 opaque reasoning item、完整 output item replay、显式缓存最多四个 breakpoint、1,024 visible-token 最小前缀、`30m` TTL、server-side/standalone compaction 的 canonical window，以及 hosted/client tool search 的所有权差异。

### 14.3 本地状态协议审计

新增 [`gpt56_luna_state_replay_audit.py`](code/gpt56_luna_state_replay_audit.py)，只使用 Python 标准库和合成状态。主流程通过，输出 `ok=true`、`evidence_level=local_protocol_toy`、`network_called=false`，并覆盖：

1. `current_turn`、`all_turns` 和 `auto` 的 reasoning 可用范围，以及 GPT-5.6 家族切换时丢弃不兼容 reasoning；
2. encrypted reasoning、assistant phase、function call/output 的原样回放和 `call_id` 血缘；
3. idempotency key、重复回执、结果漂移拒绝和独立 artifact verifier；
4. compaction item 的 opaque/canonical replay，及压缩后 prefix 改变导致的 cache miss；
5. 1,024 token 门槛、四个 explicit breakpoint、30 分钟 toy TTL，以及 hosted/server 与 client tool search 的 `call_id` 所有权。

该 toy 不证明真实 endpoint 的 schema enforcement、服务端事件顺序、隐藏 reasoning、模型质量、完整权重、目标硬件 profiling 或生产 SLO。GPT-5.6 Luna 当前状态更新为：**双榜当前时点复验 + 官方 runtime contract + local protocol toy**；无独立架构/训练报告，仍不新增 GPT-5.6 专属 Transformer 正式章节。

## 12. 2026-09-21 当前时点榜单复验与 OpenAI 访问边界

本轮仍只复验已经出现在两个排行榜的 `GPT-5.6 Luna`，没有从 OpenAI 官方目录另发现模型。Artificial Analysis 精确详情页 [GPT-5.6 Luna (max)](https://artificialanalysis.ai/models/gpt-5-6-luna) HTTP 200，快照为 `3,861,087` bytes、SHA-256 `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`；页面第三方字段为 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `164.509614645781 tokens/s`、1M context、约 `$0.20/$1.20` input/output。它是 `max` 配置的第三方测量，不是 OpenAI 内部能力或版本变更证明。

DataCurve 当前快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。精确行 `mini_swe_agent_gpt_5_6_luna_max` 为 `n_attempted=448`、`n_passed=301`、Pass@1 `0.671875`、Pass@4 `0.9026548672566371`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` token、平均 `101.68080357142857` Agent steps；该结果绑定 `mini-swe-agent`、effort、工具、任务环境、超时、重试和 verifier，不能与 AA 指数拼成裸模型能力。当前 `xhigh/high/medium/low` 行也存在，但不把它们合并成一个模型分数。

本轮经 `10.237.126.170:1234` 访问 `developers.openai.com` 的模型索引、Luna 模型页、Reasoning 和 Prompt Caching 页面均返回 HTTP `403`；`7890`/`8098` 对同一官方路径超时；直连因 DNS 解析失败。故本轮没有把旧的官方页面快照冒充新鲜响应，也没有改变既有官方资料结论。当前仍是**双榜资料级闭环**：已有官方运行时资料、研究笔记和全局配套，但没有独立 GPT-5.6 架构/训练报告；下一步只在官方页面恢复可核验时补新鲜哈希和行为差异，不新增重复架构章节。

## 13. 2026-09-22 当前时点复验与官方页面恢复

本节只记录已存在的 GPT-5.6 Luna 锚点在 2026-09-22 的页面复验，不从 OpenAI 官方目录另发现模型，也不把 provider 测量漂移解释成模型升级。

### 13.1 两个排行榜的当前快照

- Artificial Analysis [GPT-5.6 Luna (max)](https://artificialanalysis.ai/models/gpt-5-6-luna) HTTP 200；快照 `3,861,318` bytes，SHA-256 `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`。第三方字段为 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `158.728370482714 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context、页面价格 input/output `$0.20/$1.20`。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) HTTP 200；当前快照 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。精确行 `mini_swe_agent_gpt_5_6_luna_max` 为 attempted `448`、passed `301`、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 Agent steps `101.68080357142857`、median peak context `201647` tokens。
- DataCurve 这一行仍是 `mini-swe-agent + tools + task environment + verifier` 的系统结果；AA 指数、速度和 task cost 是另一种第三方配置/provider 测量。9 月 21 日的 AA `164.509614645781 tokens/s`、`3,861,087` bytes 及 DataCurve `268,571` bytes 快照保留为历史证据，不与本轮拼成趋势。

### 13.2 2026-09-22 官方页面哈希

本轮通过 `10.24.27.134:7890` 取得 HTTP 200 的 Markdown 页面；以下哈希对应实际抓取内容：

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md) | 3,744 | `1f425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0` |
| [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md) | 70,253 | `91604df954335250d16e33f3b07ebbfa3e6e2b7f821f0722ad7a69b9d586719a` |
| [Agents](https://developers.openai.com/api/docs/guides/agents.md) | 5,432 | `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a` |
| [Using tools](https://developers.openai.com/api/docs/guides/tools.md) | 33,282 | `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` |
| [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) | 39,288 | `9d6c3855a4cb722a98fb364618a852fb436e9822a36fe9b54650f75bbc8d1fd8` |
| [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) | 47,097 | `c70d858eecd09681cdc671a1a037e7d51916a793eb85c9dc08be240d1cb9b2d1` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | 14,272 | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |

### 13.3 本轮新增的面试主线

1. **运行时所有权**：Agents API 托管 Codex harness 和会话进度；Agents SDK 把 loop、部署、存储、审批和 runtime 交给应用；Responses API 让应用直接管理 response item 与工具循环。
2. **工具发现协议**：hosted tool search 在同一响应内返回 `tool_search_call`/`tool_search_output`；client-executed tool search 要求应用搜索并以原 `call_id` 回传，返回工具才可用。二者都追加到上下文尾部来保护前缀缓存。
3. **状态与成本交互**：reasoning item、tool search output、compaction item 和 prompt cache prefix 是四种不同状态。compaction 可能从变更点起打断旧 cache prefix；tool search 可以减少初始 schema token，但不能替代授权、沙箱、幂等和 verifier。

当前状态仍为**双榜资料级闭环**：运行时协议、官方页面和书系配套已具备；参数、架构、完整训练/后训练 recipe、system card、生产 kernel、目标硬件 profiling 和独立 GPT-5.6 报告仍待核验，因此不新增 GPT-5.6 专属 Transformer 正式章节。

## 15. 2026-09-28：GPT-5.6 Terra 的 DataCurve 配置行补核

沿 AA 已有 canonical `gpt-5-6-terra` 复核两榜，不新增模型。AA `/zh` 快照为 `1,660,542` bytes / SHA-256 `bc147b7c89c75bfb81257de530184f7f2d25b93543ee0ac82c93b2304a6cc38f`；Terra 详情为 `3,841,417` bytes / `ecb893eec11fc3528e2df327abb152a773bd5cc1eb2a2cb13b7973d75333c41d`，页面列出 `max/xhigh/high/medium/low/non-reasoning`，AA 的 releaseDate 为 `2026-07-09`。DataCurve DeepSWE 当前页 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7c6e095ad59f1f1`，与当天先前保存快照逐字节一致；原页面内嵌配置对象已包含以下五个精确条目。因此这是补正此前“Terra 无 DataCurve 主表行”的抽取遗漏，不是新发布或分数更新。

| DataCurve config / effort | passed / attempted | Pass@1 | Pass@4 | mean cost (USD) | mean output tokens | mean Agent steps |
|---|---:|---:|---:|---:|---:|---:|
| `mini_swe_agent_gpt_5_6_terra_low` / low | 108 / 449 | 24.05% | 44.25% | 0.342 | 8,572 | 21.46 |
| `mini_swe_agent_gpt_5_6_terra_medium` / medium | 158 / 450 | 35.11% | 60.18% | 0.467 | 11,747 | 25.15 |
| `mini_swe_agent_gpt_5_6_terra_high` / high | 243 / 452 | 53.76% | 80.53% | 0.908 | 21,517 | 33.51 |
| `mini_swe_agent_gpt_5_6_terra_xhigh` / xhigh | 272 / 452 | 60.18% | 80.53% | 1.702 | 39,617 | 43.07 |
| `mini_swe_agent_gpt_5_6_terra_max` / max | 314 / 451 | 69.62% | 88.50% | 3.957 | 71,939 | 75.93 |

五行均为 113 个已尝试任务、4 runs，且受 `mini-swe-agent`、工具、环境、超时/重试及 verifier 影响；不同 effort 的尝试数略有差异。页面的 95% CI 方法按 whole-benchmark repeated runs 估计。可以讨论这个 harness 下 pass rate、token/cost 与 Agent steps 的观测权衡；不能由这些配置行单独推出 effort 的因果效果或裸模型质量。该补核不改变 GPT-5.6 家族已完成的 API/runtime 专题状态。
