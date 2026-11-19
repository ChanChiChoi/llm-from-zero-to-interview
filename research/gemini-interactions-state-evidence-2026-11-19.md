# Gemini Interactions 状态、恢复与工具步骤：来源证据

研究日期：2026-11-19。对象是 Gemini Developer API 的 Interactions surface；以 Google AI Studio 当前文档和 API v1 reference 为主，另核对 Google Gen AI Python SDK 官方 release notes。动态产品值仅表示查阅日文档公开的合同，不代表真实账户 probe、生产可用性保证或模型内部实现。

## 范围与检索

- 问题：调用方怎样区分 Interaction history、think/tool replay state、SSE event recovery、long-decode continuation 和宿主任务状态；哪些配置要逐轮重发；服务端存储与删除怎样限制可恢复范围。
- 时间范围：截至 2026-11-19 的官方资料；API 行为固定到明确的 `v1` reference 或 SDK release。
- 检索词包括：`Gemini Interactions API v1 previous_interaction_id store retention`、`ThoughtStep signature stateless replay`、`Interactions event_id continuation_token`、`background execution`、`google-genai v3.3.0 release`、`processing_call processing_result`。
- 直接查看 AI Studio Interactions overview、API v1 reference、Thinking、Streaming、Background execution 页面和 `googleapis/python-genai` release history；没有运行 SDK 或调用 Gemini endpoint。

## 来源卡与支持范围

### Interactions overview

[Interactions API overview](https://aistudio.google.com/docs/interactions-overview) 将 Interactions 描述为 2026 年 6 月 GA 的新接口，并推荐新项目使用；它提供 `previous_interaction_id` 续接、步骤可观察性和后台执行。

- 服务端续接会按 `previous_interaction_id` 取回 conversation history。该 ID 不代表宿主工作区、工具副作用或业务任务的快照。
- 文档说明 `tools`、`system_instruction` 与 `generation_config`（包括 thinking level、temperature）是当前 Interaction 的配置；若下一轮仍需要它们，调用方须重新传入。
- 默认 `store=true`。查阅日文档写明付费层保留 55 天、免费层保留 1 天；付费项目可配置 7、14、28 或 55 天窗口。文档允许按 ID 删除 Interaction。
- `store=false` 可关闭该 Interaction 的存储，但与 background execution 不兼容，也不能用于之后的 `previous_interaction_id` 续接。数值和账户选项会变，部署时应重新查目标项目、区域和当期合同。
- 文档称服务端管理状态包含 thought blocks/signatures；无状态调用则由客户端提交完整历史。

### Interactions API v1 reference

[API v1 reference](https://aistudio.google.com/api-reference/interactions-api?api-version=v1) 列出 `POST /v1/interactions`、按 ID 获取和删除 Interaction 的端点及其 Step、status、usage 与 SSE event schema。

- `FunctionCallStep.id` 与 `FunctionResultStep.call_id` 是模型工具请求和调用方结果之间的关联键。API 记录调用提议和回传结果，不替调用方做业务授权或保证外部副作用恰好执行一次。
- `ThoughtStep` 有 `signature` 字段；该参考页没有把它标为 required，也没有把它标为 optional。自定义 `FunctionCallStep` 不列 `signature`；provider-managed tool steps 可列该字段。
- Interaction status 包含 `in_progress`、`requires_action`、`completed`、`failed`、`cancelled`、`incomplete` 和 `requires_continuation`。后者附带不透明 `continuation_token`，文档要求下次创建请求原样传回以继续长解码。
- SSE 事件带有 `event_id`，API reference 说明可用它恢复 Interaction stream。该事件游标不等于 `continuation_token`，也不证明模型发出的工具已被宿主执行。
- Reference 定义 provider-managed `processing_call` / `processing_result` media steps；这类 step 不应仅按普通客户端 function calling 解释。

### Thinking 文档

[Gemini thinking guide](https://aistudio.google.com/docs/thinking) 说明 Interactions 用独立 `thought` steps 表示思考状态；signature 是供后续协议延续使用的加密状态，summary 是可选、可能为空的可见摘要，不是完整可读推理。

- 该 guide 称每个 thought signature 必有；API v1 reference 则只列出字段而没有显式标注 required。两页的必需性表述不一致，不能据此声称目标 endpoint 一定接受缺少 signature 的回放请求。
- 对 stateless replay，guide 要求原样重发收到的全部 thought blocks，不得删改；若使用带有 signature 的 built-in tool steps，也须保留其签名。它另称标准 function call 不带这些 signatures。
- `max_output_tokens` 包含 thought tokens，是硬截止线；思考阶段达到上限可返回 `incomplete`、截断或空的可见输出，同时已经产生的 thought tokens 仍可能计费。文档建议需要控制成本/延迟时降低 thinking level，而非把硬上限压到会截断任务的程度。

### Streaming、后台与 SDK 版本

- [Streaming interactions](https://aistudio.google.com/docs/streaming) 用 SSE 暴露有序 event、step 与 delta；API reference 为这些事件定义 `event_id`，可用于从中断处恢复事件流。重放事件与重做自定义工具副作用是两个独立状态机。
- [Background execution](https://aistudio.google.com/docs/background-execution) 说明创建后台 Interaction 后先取得 ID，之后可轮询状态、订阅流或重新连接；它不将 Interaction ID 定义为宿主任务的持久 checkpoint。
- Google Gen AI Python SDK 的[官方 release history](https://github.com/googleapis/python-genai/releases) 显示 v3.3.0 于 2026-11-12 发布；v3.0.0 于 2026-11-05 将 Gemini Developer API 默认版本从 `v1beta` 改为 `v1`，部分 model alias 仍只在 `v1beta` 提供。v2.34.0 于 2026-10-29 加入 `requires_continuation` / `continuation_token` 支持和 SDK 自动 continuation 选项。SDK 默认版本及可用模型会随版本改变，接入记录应同时写 SDK 版本、显式 API version 和 model ID。

## 证据限制与不外推范围

- 官方 API 文档说明公开字段与操作，不证明某个模型的内部思维内容、signature 编码、训练方法或隐藏服务实现。
- 没有真实账户、model、region、endpoint 的 probe，也未验证 disconnect 后 SDK/server 的完整运行行为；应用关键路径需要按当前 SDK 和目标服务做有边界的集成检查。
- paid/free retention 数值和可配选项绑定 2026-11-19 文档快照；不能推广到不同账户、地区、企业产品或后续合同。
- `event_id`、`continuation_token`、`previous_interaction_id` 与 function `call_id` 分属事件流、长解码、对话历史及工具往返；外部工具的幂等、授权、执行回执和 artifact 验收仍由宿主负责。
- 本记录不使用 Artificial Analysis 或 DataCurve 分数证明 API 行为；旧章节的具体 benchmark、视频 token/质量百分比与模型上限不用于本次 API 契约判断。
