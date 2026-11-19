# 第 23 章 Gemini Interactions：跨轮状态与恢复边界

## 23.1 恢复一轮对话，不等于恢复整个任务

设想一个 Agent 正在处理工单 742：它先读取工单，再准备添加一条评论并总结结果。运行到一半时，客户端断线或进程崩溃。恢复时至少要回答三个不同的问题：模型能否接着已有对话生成？刚才的事件流从哪里继续读取？评论到底有没有写入工单？

Gemini Interactions API 为第一类问题提供 Interaction 历史和续接方式，也为流式响应与长解码提供各自的恢复信息。但 API 里的对话历史不是任务数据库：它不能证明外部评论是否已写入，也不保存宿主的权限决定、工作区版本或业务验收结果。本章要沿着工单示例分清这些状态分别由谁持有，再据此选择恢复路径。

Interactions 的文档在 2026 年 6 月宣布 API 正式可用，并建议新项目采用；实际接入仍应固定自己使用的 API 版本、SDK 版本和模型标识。[Interactions overview](https://aistudio.google.com/docs/interactions-overview)

## 23.2 五种标识指向五种不同的恢复对象

一条恢复记录里可能出现几个看起来相似的 ID 或 token。它们不能互换：有的指向一轮服务端交互，有的配对工具往返，有的只恢复事件流，还有的继续尚未完成的长解码。

| 标识 | 它指向什么 | 它能帮助恢复什么 |
|---|---|---|
| Interaction `id` | 服务端的一条交互资源 | 获取、轮询或删除这条 Interaction；后台运行会先返回它 |
| `previous_interaction_id` | 前一条已存储 Interaction | 创建后续 Interaction 时，让服务端取回前一轮对话历史 |
| function call 的 `id` 与结果的 `call_id` | 一次自定义工具请求及其结果 | 把宿主回传的结果关联到模型提出的那次调用 |
| SSE `event_id` | 流中的一个事件位置 | 断线后从流中的相应位置恢复接收事件 |
| `continuation_token` | 一次尚未结束的长解码 | 在状态为 `requires_continuation` 时，按接口要求继续该次生成 |

其中 function 的关联方式以当前 API v1 reference 为准：`FunctionCallStep.id` 对应 `FunctionResultStep.call_id`。它与 `previous_interaction_id` 属于不同层次；工具结果不会因为对话历史可取回就自动写进外部系统。[Interactions API v1 reference](https://aistudio.google.com/api-reference/interactions-api?api-version=v1)

## 23.3 Stateful 与 stateless：选择谁来保存对话历史

**Stateful continuation** 让服务端根据 `previous_interaction_id` 取回此前存储的对话历史。调用方不必在每个请求里重新发送整段历史，但必须明确传入这一轮仍需要的配置。官方文档说明，`tools`、`system_instruction` 和 `generation_config` 都作用于当前 Interaction；如果下一轮仍要使用它们，就要在下一次请求中重新指定。

**Stateless replay** 由客户端保存并提交后续所需的历史。这样，宿主可以在自己的持久存储里管理重放材料，但也必须保留协议需要的步骤和字段，不能把摘要或最终文本当作完整历史。两种方式解决的是“模型对话从哪里接上”，都不会自动恢复工单评论、文件改动或其他外部副作用。

下面是概念示意，不是可直接发送的 API 请求体；它只展示历史来源和逐轮配置的区别。

```text
Stateful:
  create(turn_1, tools=A, config=X) -> Interaction I1
  create(turn_2, previous_interaction_id=I1, tools=B, config=Y)
  # I1 提供对话历史；本轮需要的 B 和 Y 仍显式传入

Stateless:
  client 保存并重放所需历史 H1
  create(history=H1, current_input=turn_2, tools=B, config=Y)
```

因此，宿主还要为工单 742 保存独立的 task 状态，例如当前工单版本、已授权动作、调用回执和验收状态。Interaction ID 是服务端交互资源的引用，不是完整 runtime snapshot。[Interactions overview](https://aistudio.google.com/docs/interactions-overview)

## 23.4 Thought summary 可缺省；signature 是不透明的协议状态

Thinking 文档把 thought summary 描述为可选的开发者可见摘要；它可能为空或不存在，也不是完整的隐藏推理。signature 则是供协议继续交互使用的不透明字段。调用方不应尝试解码、重建或改写它，也不应把它当成可展示的推理文本。[Thinking guide](https://aistudio.google.com/docs/thinking)

stateless replay 时，应按文档要求原样保留实际收到的 thought blocks 和相关签名。这里有一处需要如实表达的文档边界：Thinking guide 称每个 thought 都有 signature；API v1 reference 列出了 `ThoughtStep.signature`，但没有明确标注该字段必需还是可选。因而，字段表没有标注 required 不能证明服务端接受缺少 signature 的回放。若应用依赖缺省字段的行为，应针对目标 API 版本确认；在此之前，不要静默删除实际收到的签名，也不要自行补造一个。

## 23.5 工具提议、外部执行和结果回传是三个动作

回到工单 742。模型可以提出 `add_comment` 的 function call；这个 step 表示模型请求宿主考虑执行，不表示工单服务已经收到评论。宿主先检查当前用户、工单和参数是否获准，再调用工单服务，记录执行结果，最后按 API 的工具结果格式把结果关联回原调用。

危险的窗口出现在“外部服务已执行、宿主还没持久化回执”之间。假如评论已写入而宿主随即崩溃，重连后再次看到相同工具请求，并不能说明评论尚未写入。事件流恢复不会替外部调用提供 exactly-once 保证。宿主应先用外部服务提供的幂等键或查询接口核实状态；若无法确认结果，应把该动作保留为未知并进入明确的恢复流程，不要盲目重试。

Google 管理的工具步骤与自定义 function 也要分开看。API reference 还定义了媒体处理用的 `processing_call` / `processing_result` steps；它们表示 provider 管理的处理过程，不是宿主自定义函数的另一种写法。无论是哪种工具，应用自己的资源授权和业务验收边界仍由宿主负责。

本册第 4 章讲工具注册，第 8 章讲权限，第 17 章讲副作用与幂等，第 21 章讲持久工作区和事实来源；本章只把这些通用机制接到 Gemini 的 Interaction step 与 ID 合同上。

## 23.6 状态、事件流和长解码需要不同的恢复动作

Interactions API 区分以下状态：

| 状态 | 调用方要判断什么 |
|---|---|
| `in_progress` | 交互仍在运行 |
| `requires_action` | API 正等待所需的调用方动作 |
| `completed` | 这条 Interaction 已完成 |
| `failed`、`cancelled` | 交互以失败或取消结束 |
| `incomplete` | 交互未完整产出预期结果；不能当作成功完成 |
| `requires_continuation` | 长解码可以继续，需按要求回传 `continuation_token` |

SSE 的 `event_id` 用于恢复事件流；`continuation_token` 用于继续尚未完成的生成；`previous_interaction_id` 用于让服务端取回已存储的对话历史。恢复事件流不会重新执行宿主工具，续接生成也不等于恢复外部副作用。后台执行同样以 Interaction ID 为入口，调用方可轮询状态、订阅事件流或重新连接。[Streaming](https://aistudio.google.com/docs/streaming) · [Background execution](https://aistudio.google.com/docs/background-execution)

`max_output_tokens` 是硬上限，并包含 thought tokens。若生成在思考阶段触及上限，Interaction 可能以 `incomplete` 结束，可见输出可能截断或为空；已经产生的 thought tokens 仍可能计入用量。因此，判断工单任务是否完成时，不能只看是否收到某个 SSE 事件或模型有没有给出一句总结：还要核对 API 状态、外部执行回执和任务自己的验收结果。[Thinking guide](https://aistudio.google.com/docs/thinking)

## 23.7 存储期限决定服务端历史能保留多久

截至 2026-11-19，Google 文档说明 Interactions 默认使用 `store=true`；当时列出的默认保留期为付费项目 55 天、免费项目 1 天，付费项目还可配置 7、14、28 或 55 天的保留窗口，并可按 ID 删除 Interaction。这些值属于查阅日期对应的产品合同，实际部署要重新核对目标项目和当前文档。

`store=false` 会关闭该 Interaction 的服务端存储，因此它不能用于之后通过 `previous_interaction_id` 续接，也不兼容后台执行。若宿主需要跨请求恢复，就必须在自己的系统里保管所需历史和任务状态，并另行设定访问、保留及删除规则。删除服务端 Interaction 不会自动删除应用日志、工作区文件、工单评论或其他外部系统记录；这些数据有各自的所有者和生命周期。[Interactions overview](https://aistudio.google.com/docs/interactions-overview)

## 23.8 恢复时按状态来源逐项对账

面对一次中断，先问“我现在要恢复的是什么”，再选择相应的 API 状态或宿主记录：

1. 要接着对话历史生成，确认服务端存储和 `previous_interaction_id` 是否仍可用；stateless 路径则提交完整、未被改写的所需历史。
2. 要接着收事件，使用事件流恢复信息；不要把恢复事件误当成重新执行工具。
3. 要接着尚未完成的生成，检查 `requires_continuation` 并使用对应的 `continuation_token`。
4. 要判断外部动作是否生效，查询宿主执行账本或外部系统；状态未知时先消除不确定性，再决定是否重试。
5. 只有 API 状态、外部回执和业务验收都满足任务要求，才把应用任务标记为完成。

接入记录应固定 API version、SDK version 和 model ID。Google Gen AI Python SDK 的 release history 记载，v3.0.0 将 Gemini Developer API 的默认版本从 `v1beta` 改为 `v1`，而部分模型别名仍只在 `v1beta` 可用；默认值可能随 SDK 改变，不能代替应用显式选择接口版本。[SDK release history](https://github.com/googleapis/python-genai/releases)

本章的范围是公开 Interactions API 与宿主恢复责任。官方文档说明的是其公开合同，不据此推断模型参数、内部架构、训练方法、signature 编码或服务端恰好执行一次工具的语义。更完整的来源范围、版本和未验证项见[Gemini Interactions 状态证据记录](../../research/gemini-interactions-state-evidence-2026-11-19.md)。
