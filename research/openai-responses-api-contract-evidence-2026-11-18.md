# OpenAI Responses API 当前合同与章节教学范围

检索日期：2026-11-18。记录对象是 OpenAI Responses API 的输入/输出对象、工具回合、流式事件、会话状态、保留、后台响应、用量和迁移合同。以下按本次官方文档访问状态记录；文档、模型支持和价格会更新。本文件只记录知识及证据，不记录书稿审阅进度。

## 检索范围与方法

- 读者问题：怎样从一次 Responses API 调用中辨认 Response、output item、工具交接、流式进度、对话续接与响应终态；哪些状态由 API 保存，哪些仍属于宿主应用。
- 平台：Web 搜索后打开 OpenAI 官方 API 文档和 API reference 原文；为确认教学组织方式，另读大学课件与 CMU 教学目标指南。未把搜索摘要或本地旧稿当作事实来源。
- 检索式：site:developers.openai.com/api/docs/guides/responses Responses API response object output items streaming；site:developers.openai.com/api/docs/guides/conversation-state Responses API previous_response_id conversations store；site:developers.openai.com/api/docs/guides/streaming-responses Responses API event types；site:developers.openai.com/api/reference/resources/responses/methods/create fields；site:developers.openai.com/api/docs/guides/function-calling Responses function_call function_call_output call_id；site:developers.openai.com/api/reference/resources/responses/methods/cancel background response；site:developers.openai.com/api/docs/guides/your-data Responses API retention store；site:developers.openai.com/api/docs/guides/background response streaming starting_after；site:developers.openai.com/api/docs/guides/structured-outputs Responses text.format json_schema；site:developers.openai.com/api/docs/pricing token usage Responses API；university course syllabus LLM API function calling tool use structured outputs。
- 原文范围：Create Response、Function Calling、Streaming API responses、Conversation state、Background mode、Data controls、Structured outputs、Migrate to Responses、Responses API reference 与官方 Pricing 页面，均于 2026-11-18 访问。
- 本次没有向 API 发请求，没有执行 SDK、示例代码或 benchmark。

## 主题、证据与边界

| 主题 | 官方资料支持的事实 | 边界 |
|---|---|---|
| Response 封套与 output | POST /v1/responses 返回带 ID、状态、模型、usage 和 output 数组的 Response；数组可含 message、function call、内置工具调用、reasoning 等不同 item，长度和顺序不是固定的。SDK 的 output_text 适合只取最终文本，不代表其他 item 可忽略。状态可包括 queued、in_progress、completed、failed、cancelled、incomplete。来源：[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)、[Responses reference](https://developers.openai.com/api/reference/typescript/resources/responses)。 | API response 完成不代表宿主任务成功、外部副作用提交或业务验收通过；具体 item 类型取决于模型、工具和请求。 |
| Function calling | 工具调用是模型与应用之间的多步 API 往返。模型可返回一个或多个 function_call item，包含 call_id、名称和参数；应用执行自己拥有的函数，并将对应 function_call_output 与原 call_id 返回模型。流式参数以 delta 到达，聚合完成前只是部分参数。来源：[Function calling](https://developers.openai.com/api/docs/guides/function-calling)。 | 自定义函数由应用执行；内置 hosted tools 使用各自端点合同。Schema 或 Structured Outputs 不等于权限授权、业务规则或事实验收。 |
| Streaming | stream=true 使用 SSE 发送语义化、带类型的事件。事件可表示 response 创建/终态、output item 生命周期、文本 delta、函数参数 delta 等；事件带 sequence_number。来源：[Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)、[Streaming events reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)。 | sequence_number 描述事件顺序，不是工具副作用回执。后台流可在特定条件下用 starting_after 续读；不能把这一点推广成所有同步流都可断点续传。 |
| 对话状态 | 可手工把前一响应 output items 回放进下一次输入；官方无状态续接示例设置 <code>store=false</code>，并由应用把前轮输出 item 加入自己的 history 后再次提交；也可用 previous_response_id 续接，或使用带持久 ID 的 Conversations API。首轮 input 中的 developer message 会随链保留；顶层 instructions 不会自动带到下一响应。使用 previous_response_id 时，前轮输入 token 仍按输入 token 计费。来源：[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)。 | response chain 和 conversation item 是 provider conversation state，不等同于宿主任务 checkpoint、workspace 或外部系统事实。官方文档展示了无状态手工回放与 previous_response_id 两条路径，但没有据此声明两者普遍不兼容；本章只说明各自的状态持有方式，不推断未明确写出的组合限制。链路可用性与保留受配置和期限约束。 |
| 存储与保留 | Response 默认存储，常规保留说明为 30 天；store:false 禁用相应 Response application-state 存储。Conversation 对象及其 items 不适用该 30 天 TTL。Zero Data Retention、Modified Abuse Monitoring、图片/文件、缓存、hosted container、第三方 MCP 等有单独条件或例外。背景响应即使 store:false 也会有处理请求所需的临时存储。来源：[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Data controls](https://developers.openai.com/api/docs/guides/your-data)、[Background mode](https://developers.openai.com/api/docs/guides/background)。 | “不用于训练”不等于“没有存储”。项目资格、ZDR、处理模式、工具和数据类型都可能改变保留边界；生产合规判断须查目标组织当时的 data controls 和合同。 |
| Background 与取消 | background:true 返回可轮询或可由 webhook 通知的 Response ID；可查询 response 状态。只有后台响应可调用 cancel endpoint；重复 cancel 幂等。同步流可关闭连接。创建为流式的后台响应在连接中断后可凭最后的 sequence_number 用 starting_after 续读。来源：[Background mode](https://developers.openai.com/api/docs/guides/background)、[Cancel a response](https://developers.openai.com/api/reference/resources/responses/methods/cancel)。 | API 取消只适用于该 Response 生成流程，不回滚应用之前执行的自定义工具或其他系统副作用。不同 completion window、model eligibility 和 temporary retention 会变。 |
| Structured Outputs | Responses 使用 text.format 配置 JSON Schema。文档说明它约束完成且非拒绝的模型输出遵循受支持 schema；应用仍需处理拒绝、不完整响应、API 错误及 schema 未表达的业务规则。来源：[Structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[Create a model response](https://developers.openai.com/api/reference/resources/responses/methods/create)。 | 结构符合 schema 不证明内容真实、权限有效或业务操作正确；可支持的 schema 子集与模型范围应以当前文档为准。 |
| 迁移与计量 | 官方 migration guide 将 messages[] 映射到 input，将输出读取改为遍历 typed output item；比较 Responses 与 Chat Completions 时需适配 function-call 格式、Structured Outputs 字段、state 管理和错误/事件处理。API usage 有 input/output token 和细分信息；价格按当前准确模型与计费选项查询。来源：[Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)、[Pricing](https://developers.openai.com/api/docs/pricing)、[Responses reference](https://developers.openai.com/api/reference/typescript/resources/responses)。 | Usage 对象不一定等于完整账单；模型价、缓存、后台窗口、托管工具及其他收费项会变化。此处不记录固定价格或用统一公式估算账单。 |

## 教学组织资料

- CMU 10601 课程的 LLM tool-use 课件沿“模型给出结构化工具调用 → 接收工具结果 → 再继续响应”解释循环：[Lecture 26: Code / tool calling](https://www.cs.cmu.edu/~mgormley/courses/10601/slides/lecture26-code.pdf)，访问日期 2026-11-18。它是课程讲义，不是 Responses API 版本合同或学习效果研究。
- University of Mannheim IE685《LLMs and Agents》课件版本 2026-02-27 的 Tool Use 部分按工具定义、调用、客户端执行和返回结果展开，并在后续另讲 Agent、Harness 和评估：[LLM Agents and Tool Use](https://www.uni-mannheim.de/media/Einrichtungen/dws/Files_Teaching/Large_Language_Models_and_Agents/FSS2026/IE685_LA_04_LLMAgentsAndToolUse.pdf)。它是课程组织案例；示例是旧 function-calling 形状，不能支持当前 Responses 字段或普遍教学效果。
- CMU Eberly Center 指南建议以学生可执行、可观察的学习目标表达课程结果：[Learning Objectives](https://www.cmu.edu/teaching/designteach/syllabus/newcourse/learningobjectives.html)，访问日期 2026-11-18。它支持把本章结果写成辨认 item、复走 tool loop、区分状态和保留，不证明某个固定章节顺序有效。

## 证据归纳

1. Responses 的核心读者变化是从单一文本读取习惯转为解析带状态的 Response 封套和有类型 output items；只需最终文本时可用 SDK helper，涉及工具或多模态时必须按类型处理。
2. 自定义工具的模型输出是应用可消费的调用提议。调用、执行、返回结果和后续响应是分开的 API 往返；call_id 把结果关联到工具调用。应用仍拥有自定义工具实际执行与业务权限。
3. SSE 把响应分为可增量消费的事件，但 delta 不是完成 item，事件序号不是业务提交回执。Background 模式提供特定的查询、取消和流恢复合同。
4. previous_response_id、Conversations API 和手工 replay 是三种不同的上下文管理路线。API conversation state 不能替代宿主的任务状态、workspace 或外部事实。
5. store:false 是响应 application-state 存储选项，不是所有 API 数据处理路径一概不保留的承诺；背景处理、Conversation、媒体、托管工具和第三方服务须分别核对。
6. Response status=completed 只说明这一 API response 已结束；Structured Output 符合 schema 只说明输出形状约束满足。两者都不能独自证明 Agent 任务完成或外部动作成功。
7. API 计量和价格必须绑定具体模型、计费项和请求配置。以 Responses API usage 近似完整成本，会漏掉 API 页面未包含的服务、工具或应用端资源。

## 未解决边界

- 本记录没有按账户、地区、端点、模型、tool type、ZDR/MAM 配置逐项验证数据保留；需由实际部署按当前官方 data controls 核验。
- 本记录没有运行 API 请求、验证 SDK 类型映射或测量计费。
- 官方文档与模型目录持续更新；API 的支持 item、字段、模型、工具、状态、后台时窗和价格不能脱离访问日期作为永久合同。
- 教学组织资料是课程实例或教学建议，不代表对所有读者均有效的教学效果研究。

## 2026-11-19 官方文档复核补充

以下是本日重新访问官方文档后确认的当前合同细节，适用边界仍限于页面列出的模式和条件：

- 后台流式只用于默认完成窗口 <code>soonest</code>，创建时须设置 <code>stream=true</code>；<code>2h</code> 和 <code>8h</code> 请求应省略 <code>stream</code> 或设为 <code>false</code>。连接中断后，可以将最后事件的 <code>sequence_number</code> 作为 <code>starting_after</code> 恢复；若创建时未启用 streaming，不能事后再流式读取。来源：[Background mode](https://developers.openai.com/api/docs/guides/background)，2026-11-19 访问，章节 “Streaming a background response”。
- Data controls 将默认完成窗口的后台 Response 数据磁盘暂存说明为约 10 分钟，用于运行请求并使响应可供检索；Modified Abuse Monitoring 下，默认完成窗口的后台响应只有显式设置 <code>store=true</code> 才遵循标准保留期。该说明不推广到其他完成窗口、组织配置或工具的数据保留。来源：[Data controls](https://developers.openai.com/api/docs/guides/your-data)，2026-11-19 访问，<code>/v1/responses</code> 条目。
