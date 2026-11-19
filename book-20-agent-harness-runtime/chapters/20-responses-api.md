# 第 20 章 Responses API：从消息接口到可编排的响应对象

本章介绍 OpenAI Responses API 的公开接口合同。读者需要知道模型通过 API 接收输入并返回结果；工具调用的宿主执行经验可先回看[第四章](04-工具系统与tool-registry.md)。读完本章，读者应能辨认 Response 外层状态与 output item 的区别，沿一次自定义工具调用走完 API 往返，并判断续接、保留、流式和后台处理分别由谁负责。

这里讲的是 OpenAI 当前公开的 Responses API。字段、模型能力和保留策略都要以访问时的官方文档为准；通用 Runtime、任务 checkpoint 和 workspace 状态分别见[第十七章](17-long-running-agent与checkpoint.md)、[第十八章](18-context-folding上下文折叠.md)和[第二十一章](21-persistent-workspace与状态边界.md)。

## 20.1 为什么从一条回答字符串转向 Response 对象

对只生成一段文本的请求，应用把用户输入交给模型，再读取回答即可。Agent 应用还可能收到工具调用、拒绝、结构化输出、多模态内容和流式进度。此时若客户端只取“回答字符串”，就会丢掉回答由哪些部分组成、当前接口请求是否结束以及下一步是否要回传工具结果。

Chat Completions 也支持工具调用等功能，所以迁移不是“旧接口不能做 Agent”。Responses API 提供统一的 Response 对象、typed output items 和语义化流式事件；需要新能力或更完整的对象状态时，客户端可按官方迁移指南逐步适配。真正的变化是应用需要按对象类型处理输出，不再假定每次调用都只有一段 assistant 文本。[迁移指南](https://developers.openai.com/api/docs/guides/migrate-to-responses)按输入、输出、工具、结构化输出和多轮状态分别列出差异。

## 20.2 先读 Response 外层状态

一次请求的结果由 Response 封套包住。它有自己的 ID、请求模型、生成状态、output 数组、usage 和可能的 error/incomplete details。当前 API reference 列出的生成状态包括 <code>queued</code>、<code>in_progress</code>、<code>completed</code>、<code>failed</code>、<code>cancelled</code> 与 <code>incomplete</code>；具体请求是否经过后台模式也会影响可用状态与后续操作。[Create Response reference](https://developers.openai.com/api/reference/resources/responses/methods/create)

读结果时先回答两个问题：

1. **这次 Response 生成到哪一步？** 检查外层 <code>status</code>、<code>error</code> 和 <code>incomplete_details</code>。
2. **模型输出了哪些 item？** 遍历 <code>output</code>，按 item 的 <code>type</code> 处理。

不要把 <code>output[0]</code> 当成必然的最终文本。数组可能先有工具调用，随后才有 assistant message；一次 Response 也可能包含多个 item。SDK 的 <code>output_text</code> helper 适合明确只需要文本的场景，但不能替代对工具、拒绝或多模态 item 的读取。

<code>status=completed</code> 只表示这次 API 响应已完成生成。它不表示应用已经执行模型提出的工具、不表示文件或数据库已提交，也不表示任务 verifier 已通过。对于 coding agent，这些事实仍由宿主执行器和外部验收负责。

## 20.3 Output item 说明“这部分内容是什么”

Response 的 <code>output</code> 是一组有类型的 item。常见结构包括：

| Item | 含义 | 客户端需要注意 |
|---|---|---|
| <code>message</code> | assistant 面向用户或后续处理的消息 | <code>content</code> 里还可能有文本、拒绝、注释或其他内容部分 |
| <code>function_call</code> | 模型提出的一次自定义函数调用 | 包含名称、参数和关联用的 <code>call_id</code>；它是调用提议，不是执行结果 |
| Hosted tool item | 使用 OpenAI 托管工具的调用及结果 | 依具体工具文档解释状态和结果；不能套用自定义函数流程 |
| <code>reasoning</code> | 与推理模型请求相关的结构化状态 | 不要把它当作可读的完整思维链；若采用无状态回放，按文档保留需要续接的返回 item |
| 多模态 item | 输入或输出中的图像、音频、文件等内容 | 按内容类型和模型支持读取，不能当普通文本拼接 |

Assistant message 的 <code>content</code> 也可能由多个内容部分组成，例如 <code>output_text</code> 或 <code>refusal</code>。因此，客户端可以为用户界面定义显示策略，但不能假定“一个 response 就有一个文本字段”。

如果应用要让模型输出符合一个 JSON Schema，可在 Responses 请求中通过 <code>text.format</code> 配置 Structured Outputs。它约束受支持条件下的输出结构；应用仍须处理拒绝、不完整响应、API 错误和 schema 无法表达的业务规则。JSON 形状正确并不代表数据真实、权限已通过或操作安全。[Structured Outputs 指南](https://developers.openai.com/api/docs/guides/structured-outputs)

## 20.4 Function call 是一次宿主与模型的往返

看一个代码 Agent 检查 <code>src/parser.py</code> 的任务。模型返回 <code>function_call</code> item，表示它请求调用 <code>read_file</code>。应用用 <code>call_id</code> 找到对应调用，检查路径和权限，再由自己的工具执行器读取文件。结果通过 <code>function_call_output</code> 回传，模型之后才有机会根据文件内容继续回答。

| 步骤 | 参与者与对象 | 发生的事 |
|---|---|---|
| 1 | 宿主 → Responses API | 发送用户输入和可用工具定义 |
| 2 | API → 宿主 | 返回包含 <code>function_call</code> 的 Response；参数可能是 JSON 字符串 |
| 3 | 宿主 → 工具执行器 | 按 <code>call_id</code> 关联调用，校验参数、权限和业务条件后执行 |
| 4 | 执行器 → 宿主 | 返回工具结果或错误；宿主保存必要回执 |
| 5 | 宿主 → Responses API | 以 <code>function_call_output</code> 回传结果，并保留原 <code>call_id</code> |
| 6 | API → 宿主 | 返回后续 Response；应用继续解析 item 或呈现最终回答 |

如果模型一次提出多个函数调用，应用要逐项处理，并根据自己的执行策略决定是否并行。函数参数可以满足 API schema，却仍需要宿主执行前的权限和业务检查。自定义函数由应用执行；API 只传递调用和结果，不替应用访问它的数据库或文件系统。完整的返回格式和多工具示例见[Function Calling 指南](https://developers.openai.com/api/docs/guides/function-calling)。

## 20.5 流式事件让客户端看到生成过程

设置 <code>stream=true</code> 时，Responses API 通过 Server-Sent Events（SSE）逐步发送语义化事件。常见事件包括 <code>response.created</code>、<code>response.output_item.added</code>、<code>response.output_text.delta</code>、<code>response.function_call_arguments.delta</code>、<code>response.output_item.done</code> 和终态事件。完整名称和字段以[Streaming events reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)为准。

| 事件 | 读者可以据此做什么 | 不能据此断言什么 |
|---|---|---|
| <code>response.created</code> / <code>response.in_progress</code> | 建立当前 response 的处理状态 | 用户请求最终成功 |
| <code>response.output_item.added</code> | 为新 item 建立显示或缓冲位置 | item 内容已完整 |
| <code>response.output_text.delta</code> | 增量显示文本或暂存为草稿 | 这段文字一定是最终答案 |
| <code>response.function_call_arguments.delta</code> | 缓冲函数参数增量 | JSON 参数已闭合，可以立即执行 |
| <code>response.output_item.done</code> | 读取该 item 的完整结果 | 自定义工具已经执行 |
| <code>response.completed</code>、<code>response.failed</code>、<code>response.incomplete</code> | 处理 Response 的终态 | Agent 任务的外部副作用已提交 |

例如，参数增量还只是 <code>{"path":"src/</code> 时，客户端必须继续缓冲；只有对应调用 item 完整后，才能解析并交给应用的校验流程。SSE 的 <code>sequence_number</code> 用来标记事件顺序，不是业务数据库的提交序号，也不是工具执行回执。流式传输和事件类型的细节见[Streaming API responses 指南](https://developers.openai.com/api/docs/guides/streaming-responses)。

## 20.6 三种方式管理多轮对话状态

Responses API 提供几种不同的上下文续接方式。选择哪一种，取决于应用想由谁管理对话历史：

| 方式 | 下一轮如何带上上下文 | 适合的场景与边界 |
|---|---|---|
| 应用手工回放 | 把需要保留的前轮输入和完整 output items 放入下一次 <code>input</code> | 应用要自己筛选、裁剪和持久化历史；无状态推理请求应按文档保留续接所需 item |
| <code>previous_response_id</code> | 在下一次请求引用前一 Response | 由 API 关联前序响应；前轮输入 token 仍计入后续调用的输入用量 |
| Conversations API | 使用 conversation ID 让多个 response 关联到持久 conversation | 需要跨请求或跨会话保存 conversation items；它不是任务 checkpoint 或 workspace |

这三条路径的历史由不同一方持有。官方的 <code>store=false</code> 手工续接示例由应用把前轮输出 item 加入自己的 history，再将累计历史放进下一次 <code>input</code>；<code>previous_response_id</code> 则让 API 按前序响应 ID 关联上下文。若要关闭 Response 的常规存储并继续多轮对话，应由应用保存和回放所需的输入与输出 item，不能把响应 ID 当作应用自己的历史副本。Conversations API 会持久化 conversation items，生命周期另见 20.7。[Conversation state 指南](https://developers.openai.com/api/docs/guides/conversation-state)

有一个容易漏掉的细节：如果把长期规则放在请求顶层 <code>instructions</code>，它不会自动沿 <code>previous_response_id</code> 传到下一次请求；而首轮 <code>input</code> 中的 developer message 会保留在链式历史中。迁移或续接时要选择合适的规则位置，或在需要时重新发送 instructions。[Conversation state 指南](https://developers.openai.com/api/docs/guides/conversation-state)

这些能力管理的是 OpenAI API 中的对话上下文。它们不替应用保存分支、工作区 revision、测试回执、用户授权或外部服务结果。关于模型输入的折叠见[第十八章](18-context-folding上下文折叠.md)；跨会话文件和外部状态见[第二十一章](21-persistent-workspace与状态边界.md)。

## 20.7 <code>store</code>、Conversation 与数据保留不是同一设置

Response 对象默认存储并可按 ID 检索；官方文档将一般 Response application-state 保留说明为 30 天。创建 Response 时使用 <code>store=false</code> 可以关闭该类 Response 存储。Conversation 对象及其 items 有不同的生命周期，官方 Conversation State 文档说明它们不适用 Response 的 30 天 TTL。

这不表示一个布尔值就能概括整条数据路径。Zero Data Retention、Modified Abuse Monitoring、背景处理、音频/图像/文件、prompt cache、托管容器和第三方 MCP 服务都有自己的条件。尤其是 background mode 为了处理请求和提供结果会临时存储数据，即使请求设置了 <code>store=false</code>。官方 Data controls 文档把默认 <code>soonest</code> 完成窗口的后台响应磁盘暂存说明为约 10 分钟，用于运行请求并让响应可供检索；启用 Modified Abuse Monitoring 时，后台响应只有显式设置 <code>store=true</code> 才遵循标准保留期。因此，涉及敏感数据时要按组织资格、处理模式、工具和输入类型检查[Data controls 文档](https://developers.openai.com/api/docs/guides/your-data)，不能从 <code>store=false</code> 推断“数据没有被保存”或“所有下游都已删除”。

“API 数据不用于训练，除非显式同意”与“服务会为提供功能暂存或保留数据”是不同问题。接口字段不能替代组织的数据处理约定或第三方工具的保留政策。

## 20.8 Background mode 提供异步查询和取消合同

普通请求可以等待一次前台响应；耗时较长的 Response 可以启用 <code>background=true</code>，收到 response ID 后通过 retrieve endpoint 查询状态，也可以配置 webhook 获得完成或失败通知。客户端要处理 <code>queued</code>、<code>in_progress</code> 和终态，不应把“已收到 ID”显示成“任务已完成”。[Background mode 指南](https://developers.openai.com/api/docs/guides/background)

取消也有接口边界：Responses 的 cancel endpoint 只适用于以 background mode 创建的响应。对同步响应，客户端关闭连接；对后台响应，API 可接受取消，但这不会撤销应用已经执行的自定义工具或其他外部动作。取消状态应和先前已经收到、已经执行的工具调用分别记录。[Cancel a response reference](https://developers.openai.com/api/reference/resources/responses/methods/cancel)

当前官方指南只在默认 <code>soonest</code> 完成窗口下支持创建时设置 <code>stream=true</code> 的后台流式；选择 <code>2h</code> 或 <code>8h</code> 窗口时，应省略 <code>stream</code> 或设为 <code>false</code>。若连接中断，可用最后收到的 <code>sequence_number</code> 作为 <code>starting_after</code> 续读；创建响应时没有启用 streaming，就不能之后再开启流式读取。这个重连合同针对后台 Response 的流；不要据此宣称所有普通同步流都能从任意断点恢复。完成窗口、模型资格、计费与 ZDR 暂存限制可能变化，部署时应以[当前 Background 文档](https://developers.openai.com/api/docs/guides/background)为准。本章不抄固定模型名单或折扣价。

## 20.9 用量、错误与“响应完成”

Response 的 <code>usage</code> 通常提供输入和输出 token 总量及细分信息，例如缓存输入或 reasoning token。它是接口返回的计量字段，不自动等于整次 Agent 任务的总成本：应用自有工具、重试、人工处理、基础设施，以及某些托管工具的独立计费，都要按相应合同核算。

Conversation 链接也不会让既往输入免费。官方文档说明，即使使用 <code>previous_response_id</code>，链中以前的输入 token 仍按输入计量。准确价格随模型、服务档位、缓存与后台选项变化，核算时查看[官方 Pricing 页面](https://developers.openai.com/api/docs/pricing)，并记录实际请求的模型和 Response usage；不应从模型别名或旧榜单价格推断账单。

错误处理时同时看 HTTP/API 错误和 Response 的 <code>status</code>、<code>error</code>、<code>incomplete_details</code>。<code>incomplete</code> 可能有具体原因，例如输出预算或内容过滤；<code>failed</code> 也不能用已经生成的一段文本掩盖。反过来，<code>completed</code> 只是 API 调用完成，仍需应用判断是否有待执行工具、工具结果是否有效以及任务验收是否通过。Responses API 只为调用提供对象和状态，应用要按这些字段决定显示、继续或失败恢复。

## 20.10 从 Chat Completions 迁移时检查语义

迁移时不要只改 endpoint 和字段名。官方指南提供了常见映射：

| Chat Completions 中的用法 | Responses 中的对应处理 | 需要重新核对的语义 |
|---|---|---|
| <code>messages[]</code> | <code>input</code> 字符串或 input items | system/developer 指令是放顶层 <code>instructions</code> 还是作为历史 item |
| 读取 <code>choices[0].message.content</code> | 遍历 <code>response.output</code>；只需文本时可用 SDK <code>output_text</code> | 输出可能有工具、拒绝、多模态等非文本 item |
| function call / tool result | <code>function_call</code> 与带关联 ID 的 <code>function_call_output</code> | 参数、调用 ID、并行工具和往返状态 |
| <code>response_format</code> | <code>text.format</code> | schema 支持范围、拒绝和不完整输出 |
| 应用自行拼接历史 | <code>previous_response_id</code>、Conversations API 或手工回放 | instructions 是否延续、存储期限、前序输入用量 |
| Chat Completions 流式 delta | Responses typed events | 事件名称、item 生命周期与终态处理 |

发布前可用同一组任务比较旧、新适配器：纯文本、拒绝、结构化输出、单个和并行工具、工具错误、流式中断、背景查询、取消、状态续接和多模态输入。比较完整 item 序列与最终任务状态，而不只比较两段回答字符串。官方 migration guide 也建议在逐步迁移时检查行为、延迟、token 使用和错误。

## 20.11 面试回答与本章练习

回答“Responses API 和 Chat Completions 有什么区别”时，可以按一条真实调用过程作答：Responses 用 Response 封套承载有类型的 output items；自定义工具调用通过 <code>call_id</code> 与应用执行和结果回传关联；SSE 把生成过程拆成有类型事件；多轮上下文可手工回放、引用 previous response 或放入 Conversation。随后指出 API 的 <code>completed</code>、conversation state 和 <code>store</code> 都有各自边界，不等于任务完成、workspace 持久化或外部副作用提交。

在[练习 35EO](../../EXERCISES.md)中，沿一次 <code>read_file</code> 工具请求检查 output item、call ID、参数 delta、工具结果回传、final message 和 response 终态，并根据约束选择多轮状态方式。若只保留一个要点：先辨认 API 对象和状态，再判断由哪个宿主组件执行或持久化下一步。
