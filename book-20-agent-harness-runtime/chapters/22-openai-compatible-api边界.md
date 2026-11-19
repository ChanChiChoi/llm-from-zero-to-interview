# 第 22 章 OpenAI-Compatible API：字段相似，语义要逐项核对

把一个已有 Agent 接到另一家模型服务时，改写客户端的 `base_url`、API key 和 model 名称，常常就能让请求发出去。但请求能发出，只说明客户端与某个接口完成了通信；它没有说明目标服务是否按调用方期待的方式处理每个字段。

本章面向已经会用 SDK 或 HTTP 发送模型请求的读者，讨论如何判断一次迁移中哪些调用可以复用、哪些行为需要适配、哪些能力没有证据支持。这里从调用方看服务合同；OpenAI Responses API 的原生对象与事件见[第二十章](20-responses-api.md)，Gemini Interactions 与 thinking state 见[第二十三章](23-gemini-interactions与thought-state回放.md)，Claude 的 thinking-state 合同见[第二十四章](24-claude-sonnet-5.5-thinking-state与安全路由.md)。如何把推理引擎实现成兼容服务端，则属于[第二十四册第五十七章](../../book-24-llm-inference-engine/chapters/57-openai-compatible-api流式协议错误码和限流鉴权.md)。

## 22.1 “兼容”要对应到具体接口

“OpenAI-compatible”不是一个足以描述所有能力的完整清单。不同服务会把它用于不同接口范围：有的提供 Chat Completions，有的还提供 Responses 或 embeddings；有的只是允许复用 OpenAI 客户端语法，有的另有 provider 路由和模型专属字段。

先把三件事分开：

| 要核对的对象 | 要回答的问题 | 不能由什么直接推出 |
|---|---|---|
| 客户端与传输 | 这个 SDK 能否按目标服务的地址、认证方式和请求路径发出请求？ | 不能推出目标实现了所有 OpenAI API 端点 |
| 请求与响应形状 | 当前端点接受哪些字段、返回哪些对象？ | 不能推出同名字段会按同一种语义处理 |
| 服务行为与运行合同 | 角色、工具、流式事件、状态、用量和错误会怎样变化？ | 不能推出模型质量、工具效果或计费相同 |

这是读文档时的一组检查角度，不是某个协议标准规定的分层。OpenAI 自己也将 Chat Completions、Responses 和其他 API surface 分别列出；Responses 用 typed items 和事件流表达内容，与 Chat Completions 的 message/chunk 结构并不相同。[OpenAI API Overview](https://developers.openai.com/api/reference/overview) [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) 因此，目标写着“OpenAI-compatible”时，先确认它兼容的是哪个 endpoint、哪种请求和哪类模型。

官方文档给出的范围也不一样。Google 说明其 OpenAI library support 仍处于 beta，并建议新集成直接使用 Gemini API；Hugging Face 分别列出 OpenAI SDK 兼容的 Chat Completion 接口和 Responses API（beta），后者文档称 Inference Providers 的 chat-completion 模型都应兼容。这个范围声明仍受 beta 状态限制，也没有证明每个字段和模型行为都与 OpenAI 等价。Anthropic 则把 OpenAI SDK compatibility 描述为测试和评估 Claude API 的路径。它们都能降低某些调用改造成本，但不构成一份共同的全功能承诺。[Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) [Hugging Face Chat Completion](https://huggingface.co/docs/inference-providers/tasks/chat-completion) [Hugging Face Responses API（beta）](https://huggingface.co/docs/inference-providers/guides/responses-api) [Anthropic OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)

## 22.2 同名字段可能被支持、改写、忽略或拒绝

判断字段能否迁移，不能只看它是否出现在 JSON 里。文档可能说明一个字段被处理、被映射成不同表示、只在特定模型上有效、被忽略，或触发错误；有些字段则没有公开说明。记录时可以使用下表的简短状态，但它们只是团队沟通用语，不是标准定义：

| 记录状态 | 含义 |
|---|---|
| 文档明确支持 | 文档说明此端点和适用范围处理该能力 |
| 有条件或有转换 | 字段可用，但语义、默认值、位置或模型条件不同 |
| 明确不支持 | 文档说明会被忽略、报错或不可调用 |
| 未说明 | 当前证据不足；不能据此写成支持或不支持 |

Anthropic 的兼容层展示了几种容易混淆的结果。`system` 和 `developer` 消息会被移到对话前部并合并进顶层 system 字段；即使原请求把它们放在后续轮次，它们也不会保留原位置。工具定义中的 `strict` 被忽略，所以“字段能发送”不等于工具参数得到严格 schema 保证。对于不支持的字段，文档还分别列出静默忽略和返回 400 的情况。[Anthropic 的兼容限制与字段表](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)

Responses 状态也不能仅凭同名字段推断。DeepSeek 文档说明其 Responses API 是无状态的，不支持 `previous_response_id` 和 `conversation`；`store` 固定为 false；`parallel_tool_calls` 被忽略，并始终启用并行工具调用。若应用依赖服务端会话续接或关闭并行，这些都是迁移决策的一部分，而不是小型格式差异。[DeepSeek Responses API 兼容说明](https://api-docs.deepseek.com/guides/responses_api/)

模型和运行时仍会影响接口行为。vLLM 文档限定 Chat Completions 适用于带 chat template 的文本生成模型，指出 `user` 参数会被忽略；并行工具调用的实际结果还取决于模型。这个例子说明：同一个 HTTP server 即使提供相同路由，换了模型或模板也可能改变请求解释。[vLLM v0.30.0 OpenAI-Compatible Server](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/)

## 22.3 工具、结构化输出和 reasoning 要按行为核对

工具调用至少涉及三件事：调用方怎样描述工具，服务怎样返回一次调用提议，宿主怎样把结果回传。调用参数可以通过 JSON 解析，却仍不代表 schema 被严格遵守，也不代表工具已经执行。Responses API 中的 function call 与宿主执行责任见[第二十章](20-responses-api.md)；通用工具授权与结果校验见[第四章](04-工具系统与tool-registry.md)。

Anthropic 的兼容文档明确指出 `strict` 会被忽略，并说明强 schema 输出应看原生 Claude API 的 structured outputs；vLLM 则说明设定 `parallel_tool_calls` 只能约束至多返回一个工具调用，开启它也不保证模型会给出多个调用。迁移工具工作流时，逐项对齐工具选择、参数 schema、调用 ID、并行规则、工具结果格式与拒绝行为；若其中一项不支持，就要决定是否调整应用流程或改用该服务的原生接口。

推理参数也不是跨 provider 的通用旋钮。Anthropic 的兼容层忽略 OpenAI 的 `reasoning_effort`，而 Gemini 文档对 `reasoning_effort` 给出了模型相关的行为和限制；Alibaba Model Studio 另外列出 `enable_thinking`、`search_options` 等非 OpenAI 标准字段，且部分模型的参数不同。字段名相似只说明调用者发出了一个值；它实际控制什么，需要回到目标服务、端点和模型的文档确认。[Anthropic compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai) [Alibaba Model Studio OpenAI-compatible Chat](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)

同样的检查适用于 JSON 输出约束。要分别确认约束写在哪个请求字段、接受哪种 schema 子集、失败时是报错还是返回普通文本，以及拒绝或不完整输出如何表示。API 返回了可解析 JSON，只说明结果满足某种语法检查；业务值是否正确、工具能否安全执行仍由应用负责。

## 22.4 流式、会话状态、错误和用量是独立合同

流式接口必须按具体 endpoint 读事件。OpenAI Chat Completions 使用包含 delta 的 chunk；Responses 流使用带事件类型的 item 生命周期事件。即使两个服务都接受 `stream=true`，调用方也应确认事件名称、字段、结束条件、用量何时出现，以及断线后的处理方式。Responses 事件的完整语义见[第二十章](20-responses-api.md)和[官方事件参考](https://developers.openai.com/api/reference/resources/responses/streaming-events)。

会话状态也要核对由谁持有。OpenAI Responses 的手工回放、`previous_response_id` 和 Conversations API 有不同的历史管理方式；DeepSeek 的 Responses 兼容实现则明确无状态；Gemini Interactions 的服务端续接合同见[第二十三章](23-gemini-interactions与thought-state回放.md)。不能因为请求都带有一组 messages 或 input，就假定服务端会自动恢复上一轮。

错误和用量字段还可能暴露“看起来成功”的差异。Anthropic 文档对同一兼容层同时列有被忽略的字段和返回 400 的字段；DeepSeek 的 usage 字段使用自己的 input/output 口径；Alibaba 可通过额外字段返回搜索工具使用统计。比较成本时要核对具体端点的用量定义、模型价格和工具计费说明，不能只把不同响应里的同名 token 字段相加。[DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/) [Alibaba Model Studio Chat 参数](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)

2026 年的一篇 API 转换预印本进一步展示了 schema 检查的边界：作者在五个 translator codebase 的合成请求套件中，报告部分已完成转换通过了目标 schema 检查，却仍改变了作者所定义的 prompt、role 或 tool 信息；另一个独立流式试验发现部分翻译结果会让受测 OpenAI SDK 抛错或静默拼装错误。这是特定代码库、输入套件和 SDK 版本上的结果，不是生产流量发生率；它支持的是一个窄结论：解析成功本身不能证明对话信息在转换前后保持不变。[Liu et al., “Testing Conversation Preservation Across LLM APIs,” arXiv:2611.15543v1](https://arxiv.org/abs/2611.15543)

## 22.5 复走一次迁移：OpenAI SDK 调用 Anthropic 兼容层

假设一个 Agent 使用 OpenAI SDK 发起 Chat Completions 请求，依赖 developer 指令、严格工具参数和文件输入。把客户端改为使用 Claude API 地址、Anthropic key 和 Claude model 后，SDK 可以调用该兼容层；但迁移结论要看应用真正依赖的行为：

| 应用依赖 | 兼容文档说明 | 对迁移的影响 |
|---|---|---|
| 复用 OpenAI SDK 发请求 | 文档提供兼容用法，并要求更换 base URL、key 和 model | 客户端语法可以复用 |
| developer 消息在中途改变指令 | system/developer 内容会合并并移到对话开头 | 依赖原位置和轮次的逻辑不能原样沿用 |
| 工具参数严格符合 schema | `strict` 被忽略 | 不能把此调用当作严格结构化输出保证 |
| 用户消息包含音频或文件 | 兼容层不支持并返回 400 | 需改用受支持输入路径、原生 API 或其他后端 |
| 请求要求 reasoning effort | `reasoning_effort` 被忽略 | 不能把这个值记录成已生效的预算 |

因此，这不是简单的“可用/不可用”判断。若应用只需要 SDK 构造与发送基础文本请求，可以复用客户端；若依赖 mid-conversation 指令、严格 schema 或文件内容，就要改变调用方式、使用原生能力或选择别的后端。依据来自 Anthropic 文档对该兼容层的描述；没有对真实账户和当前 endpoint 进行 probe，所以不把文档外行为推断为已验证事实。

## 22.6 用一张有范围的兼容记录卡作决定

审查一次迁移时，先固定被讨论的对象：SDK 及其版本、base URL、API endpoint、provider、model、地区/工作空间和查阅日期。然后只记录应用实际依赖的能力，不必罗列整个接口的所有字段：

| 应用依赖的行为 | 目标文档或探测证据 | 记录状态 | 适配或限制 |
|---|---|---|---|
| messages 中角色与顺序 |  |  |  |
| tool choice、调用 ID 与参数约束 |  |  |  |
| reasoning / thinking 参数 |  |  |  |
| stream 事件、终止与错误 |  |  |  |
| 会话续接与历史保留 |  |  |  |
| usage、价格和限额 |  |  |  |
| 图片、音频或文件输入 |  |  |  |

状态可记“文档明确支持”“有条件或有转换”“明确不支持”或“未说明”，并为每项写明它适用的 endpoint、model 和文档版本。若某项对正确性或安全性关键，就设计针对该行为的集成用例，保留原始请求、响应或事件和最终应用结果；不要把一次成功调用当作整套接口完全兼容。集成验证只覆盖测试过的场景，也不能替代服务方文档或数据处理合同。

供应商选择和自建推理 API 的关注点不同。比如 Hugging Face 的 OpenAI-compatible endpoint 可自动选择推理 provider，Alibaba 的 base URL 按区域和工作空间区分；这些信息会影响路由与部署边界，但不属于 OpenAI 请求 JSON 本身。[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) [Alibaba Model Studio base URLs](https://help.aliyun.com/en/model-studio/base-url) 自建服务端的请求校验、SSE、错误码、鉴权和限流实现见[第二十四册第五十七章](../../book-24-llm-inference-engine/chapters/57-openai-compatible-api流式协议错误码和限流鉴权.md)。

## 22.7 回到迁移问题

OpenAI-compatible 可以降低客户端接入成本，但兼容范围必须落实到具体 endpoint、字段语义和服务行为。先确定 SDK 实际调用哪个 API，再查目标服务对角色、工具、reasoning、stream、状态、错误和用量的说明；对关键行为，记录支持条件、必要适配和未知项。超出目标服务明示范围的部分，要么换用原生接口，要么在应用中适配，要么不承诺该功能。

如果面试被问“OpenAI-compatible API 是否能无改动替换”，回答应先说明复用哪一层：客户端构造、某个 endpoint 的 schema，还是应用依赖的完整行为。随后举一项具体差异，例如被忽略的 strict 字段、移动位置的 developer 消息或无状态的 Responses 实现，再说明会如何核对版本与证据。练习见[35EQ](../../EXERCISES.md)。
