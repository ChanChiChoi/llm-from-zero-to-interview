# 第 20 章 Responses API：从消息接口到可编排的响应对象

## 20.1 为什么传统 Chat Completions 不够表达 Agent

最早的应用接口常把请求表示成一组 `messages`，把返回表示成一段 assistant 文本。这对问答足够，但 Agent 任务会同时产生文本、工具调用、工具结果、推理摘要、文件、图像和多个流式事件。若所有东西都塞进一段字符串，客户端必须自己猜哪些字符是工具参数、哪些是用户可见回答。

Responses API 这一类接口的核心变化，是把一次响应看成有类型的 item 或 event 集合。它让模型回答、工具动作、工具结果和最终状态成为可组合的协议对象，为长任务和多模态交互提供更清晰的边界。

这里需要区分“一个具体 provider 的 Responses API”与一般的响应对象思想。本章讨论接口设计和迁移方法；具体字段、事件名称和模型支持范围必须以对应版本的官方文档为准。

## 20.2 从 message 到 item

传统接口可以抽象为：

```text
messages -> completion text
```

可编排响应更像：

```text
input items -> response object
                     ├─ output text
                     ├─ tool call
                     ├─ reasoning item
                     ├─ file/image item
                     └─ status/usage/error
```

每个 item 有类型、ID、状态和生命周期。客户端不必从文本中解析 JSON 来判断模型是否提出工具调用，而是根据 item 类型选择 executor。工具结果再以对应 call id 回传，模型才知道是哪一次动作产生了观察。

## 20.3 工具调用的完整生命周期

一次工具调用通常经过：

```text
response.created
  -> tool_call.created
  -> tool_call.arguments.delta
  -> tool_call.completed
  -> authorization
  -> executor
  -> tool_result
  -> next response
```

流式事件可能在参数仍未完成时到达，客户端不能在 JSON 尚未闭合时执行工具。正确的做法是按 item id 缓冲增量，完成后做 schema 校验和权限检查，再交给 executor。

工具调用的协议状态不等于执行状态。模型输出了 `tool_call` 不意味着工具已经执行；executor 返回成功后才有观察；如果工具有外部副作用，还要区分 started 与 committed。

## 20.4 为什么响应对象适合多模态

多模态输入可能同时包含文字、图片、音频和文件。输出也可能包含文字、引用、图像生成结果或工具 artifact。统一 item 模型可以保留每种内容的类型和元数据，避免应用把图片 URL、文件 ID 和文本描述混在一个字符串中。

例如一个文档分析请求可以返回“结论文本 + 引用文件 + 页面坐标 + 需要用户确认的动作”。客户端根据 item 类型渲染或继续执行，而不是重新解析自然语言。

## 20.5 Responses 与会话状态

一个响应可能通过 `previous_response_id`、conversation id 或应用自己的 checkpoint 继续。无论采用哪种机制，都要明确状态由谁保存、是否跨请求可见、多久过期、是否绑定模型版本，以及输入和工具结果是否会被自动保留。

服务端托管状态很方便，但会带来数据保留、租户隔离、删除和费用问题。客户端自管状态更可控，却要自己保存 item、call id、artifact 和幂等信息。不能因为接口支持“继续响应”就假设它是永久 memory。

对 Grok 4.7，Responses 每次返回的 encrypted reasoning item 以及服务端工具的加密输出都属于协议状态；客户端若自行管理历史，应原样回传。这个默认返回行为不等于 response 一定可通过 `previous_response_id` 找回，后者受 `store` 控制。可见的 reasoning summary 流事件也应与 opaque encrypted state 分账，不能用摘要替换回放状态。

## 20.6 Streaming 的正确拼接

流式响应不只是把文本逐字吐出。客户端要处理 item 开始、文本增量、工具参数增量、完成、错误、取消和 usage。若用户已经看到一段文本，而后续验证发现响应失败，应用需要定义这段文本是草稿还是已提交答案。

对工具调用尤其要避免重复。只有在参数 item 完成、JSON Schema 通过且权限批准后才执行；如果连接在完成事件前断开，应根据 response id 查询状态，不能直接重发可能已经提交的工具。

## 20.7 与传统 Chat 接口迁移

迁移不能只把 URL 和字段名换掉。要建立映射：

| 旧概念 | 响应对象中的对应物 | 迁移风险 |
| --- | --- | --- |
| assistant message | output text item | 文本边界和 role 改变 |
| function call | tool call item | call id、状态和参数增量 |
| tool message | tool result item | 回传关联和错误语义 |
| hidden reasoning | provider-specific item | 可见性和保留策略 |
| usage | response usage | reasoning、缓存和工具计费差异 |

旧客户端如果只读取 `choices[0].message.content`，可能丢掉工具、文件或结构化输出。适配层应把 native、adapted 和 unsupported 明确区分。

## 20.8 一个代码 Agent 的请求

用户要求“检查这个仓库的 failing test，并生成 patch”。响应对象先产生一个读取文件的 tool call，executor 返回 artifact；模型随后产生测试调用，观察到失败；最后返回 patch、测试结果和需要人工确认的提交动作。

客户端应该把 patch 作为 artifact 保存，把测试作为 verifier result，把提交动作标记为 pending effect。不能把整个 response 转成一段文本后自动执行其中的 shell 命令。

## 20.9 错误与取消

错误至少分为输入校验、模型拒绝、工具错误、权限拒绝、超时、限流、内部错误和取消。不同错误的重试策略不同：限流可以退避，权限拒绝不能盲重试，工具可能已经执行则必须查询状态。

取消也有阶段差异。取消排队请求很容易；取消已经启动的外部动作可能只能阻止后续提交，不能撤销已发生的副作用。响应对象应让客户端知道取消是否被接受、当前 item 是否完成和是否需要人工确认。

## 20.10 评估协议而不只是文本

建立 golden response suite，覆盖非流式文本、多轮、工具成功、工具失败、并行工具、结构化输出、文件、图片、reasoning item、超长输入和取消。记录 item 序列、ID、usage、错误、最终 artifact 和重放结果。

协议回归可以先做结构比较，再做模型质量、吞吐和成本压测。这样能区分“字段丢失”与“模型回答变化”。

## 20.11 常见误区

把 Responses API 当成更换 URL；把 item 当成普通文本；在流式参数未完成时执行工具；把 response id 当永久 memory；忽略 usage 和 reasoning 的计费；只测试 HTTP 200，不测试事件顺序、取消和重复副作用。

## 20.12 面试回答与练习

回答“Responses API 与传统聊天接口差异”时，应说前者把响应建模为有类型、可流式和可继续的 item/event 集合，能表达工具、文件、多模态和状态；迁移时要处理 item、call id、usage、错误、取消、状态保留和 fallback，不能只改字段名。

练习一：设计一个 tool call 从 delta 到 commit 的状态机。

练习二：列出旧消息接口迁移到 item 接口的五个 golden cases。

练习三：解释为什么 response id 不等于永久 memory。

### 20.12.1 item/event 的状态机

一个 tool item 的生命周期可以是：

```text
created -> streaming -> proposed -> policy_checked
        -> running -> result_received -> verified
        -> committed | rejected | unknown
```

文本 delta 到达不代表工具已经执行，工具结果到达也不代表副作用已经提交。客户端 UI、服务端 trace 和恢复逻辑都应使用结构化状态，而不是通过字符串猜测 finish reason。

### 20.12.2 事件顺序和重放

流式连接可能断开、重复发送或乱序。事件要有单调 sequence、event ID、item ID 和 response ID；客户端应能去重并从最后确认的 sequence 继续。服务端要区分“已发送给客户端”和“已在工具侧提交”，不能用网络 ACK 代替业务确认。

取消请求也有竞态：用户点击取消时，工具可能已经执行。取消响应应返回状态查询入口，不能简单说“已取消”。

### 20.12.3 迁移一个旧聊天调用

旧接口把工具调用编码在 assistant message 中，迁移到 item/event 接口时，需要把消息内容拆成 text item、tool-call item、tool-result item 和 metadata。历史数据的 message ID、tool call ID、usage、错误和权限决策要有可追溯映射；不能只把整段 JSON 放进一个文本字段。

迁移 golden cases 至少包括纯文本、并行工具、工具拒绝、流式中断、工具超时、部分输出、用户取消和模型 fallback。每个 case 检查最终文本之外的 item 序列和状态。

### 20.12.4 状态保留的边界

response ID 可以帮助服务端关联请求或继续某个上下文，但不必然意味着永久 memory、跨租户可见或跨模型可复用。客户端要显式声明 retention、权限和删除语义；恢复时仍需检查模型、工具、模板和策略版本。

### 20.12.5 响应协议的完整性

一个响应只有在结构完整、身份可追踪、事件顺序可重放、策略状态已确认，并且副作用边界明确时，才适合交给下游消费。可以把最小协议验收条件写成：

```math
G_{\mathrm{response}}
=G_{\mathrm{item}}G_{\mathrm{id}}G_{\mathrm{order}}G_{\mathrm{policy}}G_{\mathrm{replay}}
```

其中，`G_item` 检查每个 item 的类型和终态，`G_id` 检查 response、item 和 tool call 的 ID 关联，`G_order` 检查事件序列没有缺失或无法解释的乱序，`G_policy` 检查执行时使用的权限仍然有效，`G_replay` 检查断线重连或重复投递不会再次提交副作用。任一验收条件为零，响应仍可以作为草稿或错误记录保存，但不能被当作已经完成的业务结果。

这组条件也解释了为什么“收到 HTTP 200”不是成功标准。文本可能已经到达浏览器，工具 item 却仍处于 `running`；或者流已经结束，业务提交仍然是 `unknown`。只有协议状态、执行状态和业务提交状态都能对齐，客户端才有足够信息决定展示、重试或请求人工处理。

## 20.13 请求、响应与执行状态的三层封套

一个可编排响应最好把三种状态分开：协议是否完整、模型 item 是否完成、业务副作用是否提交。它们经常被错误地压成一个 `status=completed`。

```text
protocol envelope
  response_id, sequence, event_id, usage, error
model item
  text | tool_call | file | reasoning, item_status
execution record
  authorization, started, observed, committed, effect_id
```

例如，最后一个文本 item 已经完成，但工具 item 仍是 `running`，整个 response 不能交给业务层当作“任务完成”。反过来，工具返回了结构化结果，也不意味着外部数据库提交成功。客户端展示可以消费草稿文本，业务提交则必须等待执行记录通过验收。

这三层分离还方便错误处理。协议层的 sequence 缺失需要重连或重新拉取；模型层的 JSON 参数不合法需要拒绝该 item；执行层的 `unknown` 需要查询 effect id。若三种错误都只返回一段 `error` 文本，恢复程序无法采取正确动作。

## 20.14 流式断线、重复事件与重新连接

流式协议至少需要 `response_id`、`item_id`、`event_id` 和单调的 `sequence`。客户端要持久化最后一个已确认的序号；重连时请求从该序号之后的事件，收到重复事件则按 `event_id` 去重。仅依靠 TCP 连接是否断开，不能判断服务端是否已经执行工具。

一个安全的重连过程是：

```text
断线
  -> 保存 last_acked_sequence
  -> 查询 response / effect 状态
  -> 拉取缺失事件并去重
  -> 校验 item 和参数完整性
  -> 继续渲染或进入人工确认
```

如果工具参数增量在断线前没有收到完成事件，客户端不能猜测 JSON 的剩余部分；如果工具已经收到完整参数但响应事件未到达，应通过 call id 或 effect id 查询。对于不支持查询的副作用工具，协议应把状态标为 `unknown`，而不是自动重发。

事件顺序也有语义。`tool_result` 出现在对应 `tool_call.completed` 之前，可能是网关重排或服务端 bug；`response.completed` 早于所有 item 终态，则是一个违反契约的响应。conformance test 应把这些序列当成协议错误，而不是用最后一段文本勉强兜底。

## 20.15 Usage、缓存与成本对账

兼容接口中的 `usage` 往往只覆盖部分资源。输入 token、输出 token、reasoning token、缓存命中、图片 patch、音频帧、工具时间和重试可能由不同字段或不同服务计量。客户端显示的 token 不能直接当成账单真相。

一次响应的成本可以建模为：

```math
C_{mathrm{response}}
=p_i n_i+p_o n_o+p_r n_r+p_m n_m
 +C_{mathrm{tool}}+C_{mathrm{retry}}+C_{\mathrm{storage}}
```

其中，`n_i`、`n_o`、`n_r` 和 `n_m` 分别是输入、输出、推理和多模态资源，价格 `p` 必须绑定价格版本。缓存命中可能减少实际计算，却不一定与输入 token 的计费规则相同；因此 trace 要同时保存 provider usage、网关计数、缓存 key/version 和最终账单对账结果。

迁移后至少做三项对账：同一 golden request 的服务端 usage 与客户端计数差异；流式与非流式请求的计数是否一致；工具重试和取消后是否仍有隐藏资源消耗。差异持续超过阈值时，应先标记成本数据不可信，再讨论模型质量。

## 20.16 能力协商、权限与数据保留

响应对象提供了更丰富的能力，也扩大了协议攻击面。客户端在请求前应完成 capability negotiation：后端是否支持某种 item、工具并行、文件引用、取消、状态保留和结构化输出。unsupported 能力必须显式失败或走已验证的 adapter，不能静默把工具调用转成普通文本。

工具 item 进入执行器前，至少要检查调用者、租户、工具 schema hash、参数、资源范围和用户确认状态。模型产生了文件 ID，不代表它有权读取文件内容；response id 能继续上下文，也不代表当前用户拥有旧响应的访问权。

状态保留要定义生命周期：response、conversation、tool result、文件、trace 和缓存的过期、删除、备份和跨租户边界分别是什么。用户删除会话后，派生 artifact 和审计日志可能仍受不同政策约束，接口应提供可查询的删除结果，而不是只返回一个成功文本。

## 20.17 协议验收的最小案例集

一个真正可上线的响应协议测试集，至少要覆盖：纯文本完成、部分文本后错误、工具参数增量、并行工具、工具拒绝、工具超时、未知副作用、断线重连、重复事件、用户取消、状态过期、结构化输出失败、文件权限拒绝和 usage 对账。每个案例保存完整 event sequence 与最终执行状态。

验收可以分层：

```math
G_{\mathrm{api}}
=G_{\mathrm{parse}}
 G_{\mathrm{sequence}}
 G_{\mathrm{state}}
 G_{\mathrm{policy}}
 G_{\mathrm{replay}}
```

`G_parse` 通过只说明 JSON 能解析；`G_sequence` 还要保证事件关系正确；`G_state` 要保证 item 和执行终态一致；`G_policy` 检查权限与保留边界；`G_replay` 检查重连、重复投递和取消不会产生重复副作用。只有全部通过，响应才可以作为业务事实；否则最多作为草稿、诊断记录或待人工处理项。

## 20.18 item 协议的状态机

把响应看成一串 item 后，客户端必须处理 item 的创建、增量、完成、失败和取消，而不能只拼接字符串。一个工具 item 可能先有调用参数，随后等待外部结果，再产生 observation，最后才让模型继续；多模态 item 还要处理媒体下载、解析失败和 token 预算。

客户端应保存 `response_id`、`item_id`、顺序号和最后确认的事件。若断线后收到重复事件，按事件 id 去重；若序号跳跃，进入重新拉取或失败状态。把网络重试当成“再次发送最后一条消息”，容易在工具调用和外部动作上造成重复。

## 20.19 从旧接口迁移的行为兼容

传统 message 接口的迁移不只是字段改名。要逐项比较 system/developer role、特殊 token、tool call id、并行工具、流式 delta、usage、finish reason、取消、图片/文件和 reasoning item 的语义。旧客户端若只取 `text` 字段，可能静默丢掉工具状态或把未完成事件当成最终答案。

一个稳妥的 adapter 应先把 provider 事件转成内部规范，再由内部状态机决定可见文本和可执行动作。适配器升级时用 golden trace 比较事件序列和最终业务状态，而不是只比较最终字符串。

## 20.20 item 事件和副作用的分离

Responses 风格的 item 流可以同时承载文本、推理、工具调用、工具结果、图片或文件。客户端展示层可以消费一部分事件，executor 却必须等待授权、参数校验和提交确认。不能因为某个 item 已经出现在流中，就把它理解成外部动作已经成功。

## 20.21 状态续接和重复请求

续接请求时需要明确是继续同一个 response、重新生成候选，还是从最后一个已确认 item 开始重放。保存 `response_id` 和事件序号只能帮助定位，不能替代幂等查询。工具已经执行但 response 未完成时，应先查询动作状态，再决定是否继续生成用户可见结果。

## 20.22 协议迁移的真实验收

把旧 Chat 接口和 Responses adapter 放在同一组任务上，比较结构化输出、工具调用、流式断开、取消、usage、错误、图片/文件和最终业务状态。文本相同但事件顺序不同，仍可能破坏客户端；事件相同但 template 不同，仍可能改变模型行为。

## 20.23 多模态 item 的预算和生命周期

图片、文件、音频和视频 item 会触发 processor、存储、异步下载和 token 预算，不能和普通文本一样即时拼接。客户端应知道媒体正在处理、处理失败还是已经进入模型；服务端应记录 media hash、权限、留存和取消状态。

## 20.24 API 兼容的灰度策略

先在 shadow 中比较新旧事件，不执行真实工具；再对只读工具 canary，最后才扩大到可写动作。每一步检查流式顺序、usage、错误、取消、工具副作用和 p99。响应协议升级不能只依赖客户端“看起来还能显示文字”。

## 20.25 Responses API 的阶段性判断与资料边界

Responses API 的工程价值是把复杂 Agent 交互从字符串解析提升为有类型的响应对象和事件协议。真正困难的部分仍然是状态、工具、权限、流式一致性、错误恢复、成本对账和版本迁移。协议完整不等于业务完成，HTTP 200 也不等于副作用已经提交。

本章关于具体字段和事件不作跨 provider 泛化；字段名称、模型支持、状态保留和计费以对应版本的官方 API 文档为准。

## 20.26 GPT-5.6 Luna：Responses item 与 Agent runtime 所有权

GPT-5.6 Luna 当前最适合用来讲清一个经常被混淆的边界：模型、响应协议和 Agent runtime 不是同一层。Luna 模型页公开的是模型合同，例如 `1,050,000` context、`922,000` maximum input、`128,000` maximum output、`none`--`max` effort 和支持的端点/工具；这些字段不能反推出参数规模、MoE/稠密结构或训练配方。

OpenAI 官方 Agents 文档把运行时分成三种入口：

| 入口 | 谁管理 agent loop | 状态/执行边界 | 适合的问题 |
|---|---|---|---|
| Agents API | OpenAI 管理 Codex harness 和底层 Agent 基础设施 | 保存 session configuration、turns、items，并可使用托管 sandbox/工具 | 长任务，希望由平台托管进度 |
| Agents SDK | 应用管理部署、存储、审批和 runtime；SDK runner 管理 loop/handoff | 工具、sandbox、session 和业务状态由应用集成 | 需要自定义工作流和多 Agent 编排 |
| Responses API | 应用直接管理 response、历史和工具循环 | 应用决定 item 回放、executor、权限和状态存储 | 直接调用模型或从零构造 harness |

三种入口的 session、conversation 和 sandbox 不是同一个资源。迁移时要把下面的状态分开保存：

```text
model contract
-> response items / reasoning state
-> tool discovery and tool call
-> permission + executor
-> artifact + verifier
-> session / trace / cost ledger
```

GPT-5.6 的 tool search 又把“工具存在”和“工具已加载”分开。Hosted tool search 由服务端搜索已声明的 namespace/MCP/function，并在同一个 response 中产生 `tool_search_call` 与 `tool_search_output`；client-executed tool search 由模型产生 `tool_search_call`，应用搜索自己的项目/租户状态，再用相同 `call_id` 回传 `tool_search_output`。两种路径都会把已加载工具追加到上下文末端，以尽量复用前缀缓存；返回工具仍需通过 schema、权限、沙箱和 verifier。

长期任务要同时维护两本账：一是状态账，记录 reasoning item、tool search output、tool call/result、compaction item 和 artifact；二是成本账，记录 model、effort、工具 schema、`cached_tokens`、`cache_write_tokens`、compaction 次数和重试。Compaction 会替换较早上下文，可能让旧 cache prefix 从变化位置开始失配；它不是 prompt cache，也不是可读的人工摘要。面试中如果只回答“把历史压缩后继续请求”，就遗漏了 canonical context、opaque state、权限和副作用恢复。

## 20.27 GPT-6 Sol：三种 Agent runtime 的所有权矩阵

GPT-6 Sol 的官方模型页只规定 `gpt-6-sol` 的模型合同；Agents 文档把平台入口分为 Agents API、Agents SDK 和 Responses API。三者都能驱动 Agent，但状态所有者不同：

| 入口 | 主要状态所有者 | 恢复/审计时的最小记录 |
|---|---|---|
| Agents API | OpenAI 托管 Codex harness | session configuration、turn、item、托管工具、sandbox、平台进度 |
| Agents SDK | 应用及 SDK runner | deployment、storage、approval、handoff、工具、session、sandbox |
| Responses API | 应用自建 harness | input/output item、previous response 或完整 replay、executor、permission、artifact、verifier |

这三个入口不能因为都叫 Agent API 就共享同一份状态。尤其是 session、conversation 和 sandbox 是不同资源；迁移时必须明确谁保存历史、谁执行工具、谁批准副作用、谁决定 compaction 后是否继续。

### 20.27.1 Opaque state 的回放规则

Reasoning 与 compaction item 不是可读思维链。stateless input-array chaining 需要保留 output items，包括 encrypted reasoning/compaction item；`previous_response_id` chaining 只传新用户消息。standalone `/responses/compact` 返回的是 canonical next context，不能只取一段摘要再自行拼装。

一份可恢复的 manifest 可以写成：

```text
model, snapshot, mode, effort, request_prefix_hash
response_id, previous_response_id, input/output_items
tool_registry_hash, call_id, permission, executor_receipt
compaction_id, cache_state, artifact_digest, verifier_result
```

GPT-6 family 的 `configuration_update` 还需要记录在原始历史位置；相邻 update 会被拒绝，且不能与自动 compaction/truncation 或独立 `/responses/compact` 同用。显式 compact 后，下一条用户消息前要重新插入 update。effective effort 会持续到下一个覆盖项；响应中的 `reasoning.effort` 仍是 request-level 值，因此不能拿它当实际生效档位。更新项适用于 Responses 与 WebSocket `response.create`，并以保留原 prompt prefix、利于缓存复用为设计动机；缓存命中仍须看实际 usage，而非从协议设计推定。

### 20.27.2 面试中的一句话

回答 GPT-6 Sol Agent 系统设计时，可以用一句话收束：模型提出 reasoning/tool intent，Responses 携带 typed items，harness 管理 loop 和状态，permission engine 决定 allow/ask/deny，executor 产生副作用回执，verifier 判定任务是否完成。1.05M context 和工具目录不等于模型拥有永久记忆、网络权限或公开了内部架构。

资料依据：[GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)。

### 20.27.3 GPT-6 Sol 合同审计 toy

本节的最小可运行练习是 [`gpt6_sol_contract_audit.py`](../../research/model-update-2026-09/code/gpt6_sol_contract_audit.py)。它把 mode/effort、update placement、budget `incomplete`、whole-request pricing、permission decision、executor receipt、artifact digest、function-call lineage、opaque compaction 和 cache prefix 分成独立状态；重复相同 idempotency key 只返回 duplicate receipt，不重复外部执行。该结果只能说明本地状态机通过，不能证明 OpenAI 服务端实现、隐藏 reasoning 或生产 SLO。

### 20.28 GPT-6 Luna：同一 runtime，不同 sibling contract

Luna 可以复用 GPT-6 family 的 Agents API、Agents SDK、Responses API ownership、tool search 和 opaque compaction replay 规则，但 trace 中必须保留精确 model ID、snapshot、effort、mode、价格和 provider。模型页的工具列表不自动授予 shell、MCP、computer use 或网络权限；permission engine、executor、sandbox 和 verifier 仍归宿主 harness。

Luna 当前是 AA 单榜资料级闭环，DataCurve 没有精确 Agent 行。因而测试应在 sibling 之间做合同和成本对照，而不是迁移 Sol 的 Agent score；缺失精确行时输出 `not_applicable`。effort 更新与 compaction 的拒绝/恢复边界见 20.27.2；该 family-level API 合同不是 Luna 的内部架构特性。证据见 [`gpt-6-luna-source-notes.md`](../../research/model-update-2026-09/gpt-6-luna-source-notes.md) 与 [OpenAI Reasoning guide](https://developers.openai.com/api/docs/guides/reasoning.md)。

## 20.29 榜单标签、API alias 与 resolved snapshot 必须分账

评测和线上 trace 至少要区分排行榜展示名、排行榜 canonical/configuration、请求 API ID、动态服务 alias、响应中的实际 `model` 和 resolved snapshot。名字相近、能力描述相似或都带 “Instant” 都不构成身份映射。

OpenAI 官方目录把 `chat-latest` 描述为 ChatGPT 当前 latest Instant model，且说明底层 snapshot 会定期更新，但没有公开当前 snapshot；GPT-5.5 模型页则给出独立 API ID `gpt-5.5` 和 snapshot `gpt-5.5-2026-04-23`。AA 的 `GPT-5.5 Instant (June 2026)` 因而不能仅凭名称等同于 `chat-latest` 或 `gpt-5.5`。如果评测 harness 不能观察 resolved snapshot，报告应明确记为 `unresolved`，而非填入猜测值。

| 字段 | 示例 | 证据用途 |
|---|---|---|
| leaderboard label/config | AA `GPT-5.5 Instant (June 2026)` | 识别第三方评测对象；不自动等同于 API ID |
| requested API model | `chat-latest` 或 `gpt-5.5` | 记录客户端实际请求字符串；两者不能互换 |
| served model / resolved snapshot | `response.model`、provider trace 或明确版本记录 | 证明请求最终落到哪个模型版本；当前 `chat-latest` 文档未给出预映射 |

面试回答的核心是：先证明模型身份与请求血缘，再比较分数、价格、延迟或能力；没有显式 mapping 或可审计的 endpoint trace，就不迁移 benchmark 结果。资料依据：[OpenAI Models](https://developers.openai.com/api/docs/models.md)、[Chat Latest](https://developers.openai.com/api/docs/models/chat-latest.md)、[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5.md) 和 [`gpt-5.5-source-notes.md`](../../research/model-update-2026-09/gpt-5.5-source-notes.md)。
