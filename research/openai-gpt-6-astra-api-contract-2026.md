# GPT-6 Astra API 合同与预算：来源证据

检索与阅读日期：2026-11-02。本记录只保存官方公开页面支持的产品/API 行为、版本时点和边界；书籍章节审阅、纳入决定与修改状态写入 `progress/`。

## 模型页公开规格

- **来源与版本**：[GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)，OpenAI API 当前动态页面，访问日期 2026-11-02；另见 [Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model)，访问日期 2026-11-02。
- **实际阅读位置**：模型名/slug、context window、max output tokens、reasoning effort、pricing、modalities、endpoints、tool support；指南中的工具调用要求和 “What’s new” 部分。
- **来源支持的事实**：模型页列出的 API 模型 ID 为 `gpt-6-astra`，context window 为 1,050,000 tokens，最大输出为 128,000 tokens；`reasoning.effort` 接受 `low`、`medium`、`high`、`xhigh`、`max`。页面列出文本输入/输出与图像输入，并标注音频、视频不支持。
- **公开价格（每百万 token）**：standard input 为 $10、cached input 为 $1、cache writes 为 $12.50、output 为 $50。超过 272K input tokens 的请求，文档规定整次请求按 2 倍 input/cache 单价和 1.5 倍 output 单价计费；Batch 与 Flex 为 standard rates 的 50%，Fast 为适用费率的 2 倍。模型页还说明 search、computer use 等 tool-specific models 可能按 tool call 另收费。以上均为页面访问时的公开价目，不构成未来价格承诺。
- **证据等级与边界**：`source-backed`（供应商官方当前文档）。模型定位、相对能力和“state of the art”等为发布方表述；本记录不把这些表述当成独立 benchmark 结论。context window 和 max output 是公开规格，不说明任意请求的可用用户文本额度或实际延迟。

## Responses、工具与中途控制

- **来源与版本**：[Using GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model)，OpenAI API 当前动态文档，访问日期 2026-11-02；模型能力交叉核对 [GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)。
- **实际阅读位置**：工具调用建议、工具路由与功能更新说明；异步工具、mid-turn steering、`configuration_update` 和兼容性提示。
- **来源支持的事实**：指南明确要求 GPT-6 的工具调用使用 Responses API；尽管若干 GPT-6 模型也列出 Chat Completions endpoint，工具调用仍要求 Responses。对 function/custom tool 设 `async: true` 后，模型可继续处理其他工作；应用仍负责执行该工具、维护待完成工作，并以原 `call_id` 返回结果。mid-turn steering 通过 WebSocket Responses 接收追加指令并继续已有工作。`configuration_update` 可在对话中改变 reasoning effort，文档称该变化可避免重写原 prompt prefix 并保留 cache；其生效状态持续到后续更新。
- **证据等级与边界**：`source-backed`（供应商公开接口指南）。这些是 GPT-6 API 的产品合同，不是所有兼容 API 或本地 runtime 的通用状态机。异步 function tool 的应用侧执行/恢复责任不能被表述成模型托管工具执行；mid-turn steering、异步工具与后台 Responses 是不同功能，部署行为须按实际接口和版本重新核对。

## 工程预算时的适用边界

- 文档规格可以支持对公开 context/output 上限、effort 选项、API 价格和指定工具功能的描述；成本计算必须固定访问日期，并逐项区分 input、cached input、cache writes、output、长输入价格门槛、服务档位及工具调用费。
- **工程推论**：设计请求预算时，不能把 context window 直接当作用户正文额度；system/developer 指令、历史状态、工具定义、当前输入和预留输出均要进入实际请求账本。该预算拆分是由公开接口上限推导出的设计要求，不是文档保证的精确 tokenizer 公式。
- 页面没有公开模型权重/架构细节、可自托管能力、生产 GPU kernel、任意工作负载的延迟/吞吐或内部 serving 拓扑；不能从 API 模型页推导这些事实，也不能将 API 工具支持等同于任意应用的权限授权。

## 生命周期

上述页面是可变的 current 文档；当用于正文时须在修改当日复查模型 ID、窗口/输出上限、effort、价格门槛、工具功能和 API 兼容性，并记录访问日期。若官方页面更改，旧规格应标记为历史快照，而不是静默沿用。

## 2026-11-02 价格档位、缓存与地域费用

- **来源与版本**：[GPT-6 Astra 模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)、[API pricing](https://developers.openai.com/api/docs/pricing)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Fast mode](https://developers.openai.com/api/docs/guides/fast-mode)、[Ultrafast mode](https://developers.openai.com/api/docs/guides/ultrafast-mode)，均为 OpenAI 当前动态页面，访问日期 2026-11-02。
- **实际阅读位置**：模型页 Pricing、Prompt caching 的 Cache write TTL 与 pricing tables、Pricing 的旗舰模型档位和地域/FedRAMP 注记、Fast mode latency SLA、Ultrafast pricing/availability/residency sections。
- **来源支持的事实**：Standard 的 Astra 价格为每百万输入 $10、缓存读取 $1、30 分钟写入 $12.50、12 小时写入 $25、输出 $50。超过 272K 输入 token 的请求整次按 2 倍输入/缓存单价和 1.5 倍输出单价计；不是按超出部分分档。Batch/Flex 为 Standard 的 50%，Fast 为适用费率 2 倍，Ultrafast 为 6 倍。Astra Fast 没有 latency SLA；Ultrafast 速率限制单列，residency 当前支持 US 与 global processing，不支持 EU residency。
- **区域费用边界**：定价页说明 2026-03-05 及之后发布且符合 residency 条件的模型使用 regional-processing endpoint 时有 10% uplift；自 2026-11-01 起 FedRAMP endpoints 另列 10% uplift。模型/端点资格及两项费用是否组合应按当前合同和账单核对，不能自行假设费率叠加。
- **证据等级与边界**：`source-backed`（供应商动态价表和服务档位说明）。价格不是长期承诺；地区、账户合同、tool-specific call fee、缓存是否命中和具体 workload 质量/延迟不由本价表决定。教学算例的 token 分类和费率乘法是据文档进行的算术，不是实收账单。

## Reasoning、工具发现与会话状态

- **来源与版本**：[Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)，访问日期 2026-11-02。
- **实际阅读位置**：Using GPT-6 的 model selection/API migration、tool calling、configuration update 和 limitations；Reasoning 的 output/token reserve 与 incomplete handling；Conversation state 的 stateless replay 与 persisted reasoning；Tool search 的 deferred schema、hosted/client search、namespace 与 cache handling。
- **来源支持的事实**：GPT-6 family guide 要求工具调用使用 Responses；Astra、GPT-6.1 Sol 与 GPT-6.1 Terra 虽列有 Chat Completions endpoint，工具调用仍需 Responses。GPT-6 Sol/Luna 的 Chat Completions function calling 只在 `reasoning_effort="none"` 条件下支持；Astra 与 GPT-6.1 Sol 不支持 `none`。guide 将 pro mode 列为 GPT-6 可用 API capability，并说明 mode 与 effort 是独立维度。Reasoning guide 建议初次实验为 reasoning 与 output 至少预留 25K tokens；这是起步缓冲建议而非必需消耗量。
- **配置更新边界**：`configuration_update` 可改变后续 reasoning effort，直到被后续 update 覆盖。当前指南建议标准模式单 Agent 请求保持 request-level effort 不变，通过 input item 更新以保留稳定 prefix；兼容文档说明它不能与自动 compaction/truncation 或 standalone compact 混用。该机制改变对话运行配置，不代表模型权重或 checkpoint 变化。
- **工具发现边界**：当前 Tool search guide 描述 GPT-5.4 及之后模型的 Responses API 支持。Hosted search 从请求中声明的 deferred tools 查找候选，client-executed search 由应用检索并回传匹配输出；工具加载在 context 末端，设计目标包含保留已有缓存前缀。defer namespace/MCP 时模型先看到目录级描述；单个 deferred function 的名称与描述仍可见。搜索或加载工具不授予执行权限，应用必须校验返回 schema 和授权。
- **证据等级与边界**：`source-backed`（官方公开 API 指南）。model page、family guide 和能力矩阵是可变产品合同；不证明产品能力在所有 provider、API compatibility layer 或本地 runtime 中同样存在，也不支持对模型内部机制的推断。

## Async tool calling、steering、background mode 与 compaction

- **来源与版本**：[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering)、[Background mode](https://developers.openai.com/api/docs/guides/background)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)，访问日期 2026-11-02。
- **Async tool calling**：应用运行 `async: true` function/custom tool。guide 要求应用把唯一 `task_handle` 绑定到原始 `call_id` 和实际 job；handle 在整段 conversation 中保持唯一，包括已完成任务。若工具工作完成前出现其他对话轮，应从最新 `response_id` 续接，同时仍用原 `call_id` 回传其输出。工具完成与模型消费回执是两件事。
- **Mid-turn steering**：通过同一 WebSocket 对已创建的 response 发送 `response.steer`；`response.steer.accepted` 表示输入排队，不表示已采取行动。除非应用仍需提供工具结果或审批，API 会创建 continuation；原响应可能以 steered incomplete 结束，也可能先正常完成。token/tool-call cap 按 response 单独计算，且 steering 不会撤回已输出文本或既有副作用。
- **Background mode**：当前指南概述提到 GPT-5.2/GPT-5.2 Pro，但其请求示例也明确使用 `model: "gpt-6-astra"` 与 `background: true`，并展示对 `queued`/`in_progress` 状态轮询直至终态；因此 Astra 是该公开指南当前明确展示的后台 Responses 模型。该证据支持“官方指南记录了 Astra 后台请求用法”，不保证任意账户或 endpoint 均已开通，集成时仍须核对项目可用性。
- **Background 数据留存**：指南说明 ZDR 项目以 `store=false` 运行，响应数据为异步执行和轮询暂存磁盘约 10 分钟。MAM（含增强型 MAM）项目的后台响应只有在显式设置 `store=true` 时才会在轮询结束后继续保留；省略 `store` 或设为 `false` 时约 10 分钟后删除。该留存行为按项目的数据控制配置理解，不应将后台轮询本身当作长期结果存储。
- **Compaction**：server-side compaction 通过 `context_management` 和 `compact_threshold` 触发，并返回 opaque compaction item。stateless input-array chaining 按协议继续传递输出状态；standalone `/responses/compact` 返回的整窗是下一次请求的 canonical context，不要手工 prune。用 `previous_response_id` 链接时按 guide 传递新输入，不在客户端手动裁剪服务端对话。
- **证据等级与边界**：`source-backed`（官方状态/调用指南）。文档未披露 compaction 的完整内部算法，也不保证它是可读摘要；server event/network order、应用 job durability 和外部副作用回滚由相应系统负责，教学状态轨迹不能替代 API 集成验证。

## 安全监控、技能指令与模型生命周期

- **来源与版本**：[Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring)、[Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)、[GPT-5.3 Codex model page](https://developers.openai.com/api/docs/models/gpt-5.3-codex)、[GPT-6 Sol](https://developers.openai.com/api/docs/models/gpt-6-sol)、[GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)、[GPT-6.1 Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol)，访问日期 2026-11-02。
- **实际阅读位置**：监控页的 request coverage、stopped request error 和 webhook alerts；skills/prompts article 的 progressive disclosure；模型页的 deprecated/current label、context/output、reasoning 和 endpoint sections。
- **监控范围**：Responses requests with persisted reasoning, WebSocket, or OpenAI compaction 可自动监测并阻断后续执行；其他 Responses 请求可通过配置的 webhook 收到告警但系统不会自动停止 conversation；Chat Completions 不在该监控系统覆盖范围内。阻断错误可返回 HTTP 403，code 为 `misalignment_policy_violation`；流式客户端即使已经收到部分 output 仍要处理后续错误。此系统不撤销先前副作用，也不代替 permission/sandbox/approval/verifier。
- **提示词/技能建议**：开发者博客建议短路由文档与 progressive disclosure，指出读取技能会消耗上下文、把无关指引强制带入可能增加 compaction 并影响路由。GPT-6 family guide 另提醒 Astra 对 skills/AGENTS.md 一类 instruction sources 更敏感，并建议审核可访问指令。该建议是官方工程观察，不是对所有工作负载的量化收益承诺。
- **模型生命周期与比较**：GPT-5.3 Codex 当前模型页标为 Deprecated，并列 400,000 context、128,000 max output、Chat Completions 与 Responses endpoint。GPT-6 Sol 页面指向较新的 GPT-6.1 Sol；GPT-6 Sol、Luna、6.1 Sol 页面列出各自的能力与参数差异。相同窗口数字或产品定位不代表质量、延迟、费用、架构或训练方式相同。
- **证据等级与边界**：`source-backed`（OpenAI 官方动态页面与开发者博客）。公开页面未给出模型参数量、生产 kernel、训练 recipe、unpublished serving topology 或任意 workload 延迟；产品定位文字不等于独立评估。GPT-5.3 Codex 仅为已弃用的历史合同例子。

## 2026-11-13 reasoning 配置更新复核

- **来源与阅读范围：** OpenAI 当前[Reasoning 指南](https://developers.openai.com/api/docs/guides/reasoning)及[GPT-6 Sol 模型页](https://developers.openai.com/api/docs/models/gpt-6-sol)，访问日期 2026-11-13；复核 `configuration_update` 的支持模式、历史保留、compaction 限制和响应中的 `reasoning.effort` 含义。
- **来源支持的事实：** GPT-6 标准单 Agent Responses 会话可将 `configuration_update` 放入历史，以更新后续响应的 effort；更新持续生效直至后续 item 覆盖。该 item 必须随后续 history replay 保留。官方指南规定它不能与自动 compaction/truncation 或独立 `/responses/compact` 混用；显式压缩后若仍需所设 effort，应在下一条用户消息前重新加入配置更新。响应的 `reasoning.effort` 字段反映请求级设置，不足以单独还原更新后的有效设置。
- **证据等级与边界：** `source-backed`（OpenAI 官方动态 API 指南）。该合同只描述文档列出的 GPT-6 接口模式，不证明不同 API surface 或 provider 兼容层具有相同状态语义，也不揭示内部推理量或质量。
