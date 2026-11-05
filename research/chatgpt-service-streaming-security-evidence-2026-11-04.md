# ChatGPT 服务流式协议与安全资料核验

核验日期：2026-11-04 至 2026-11-05

本记录只保存联网发现的协议、产品文档和安全资料事实及其证据边界，不记录书籍审阅进度。

## SSE 与产品事件

### WHATWG HTML Living Standard

- 来源：[Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)，开发者版显示最近更新于 2026-11-04。
- 标准说明浏览器可通过 HTTP text/event-stream 接收事件，事件可包含 event、data 和 id 字段；设置事件 ID 后，EventSource 在重连时会通过 Last-Event-ID 请求头传递最后收到的 ID。
- 证据等级：协议标准。它定义事件格式和客户端重连游标，不规定应用必须保留多久事件，也不提供服务端事件存储、授权、业务状态恢复或重放实现。

### OpenAI Streaming API 文档

- 来源：[Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses) 与 [Responses streaming events reference](https://developers.openai.com/api/reference/resources/responses/streaming-events)，访问核验日期为 2026-11-04。页面未标明稳定版本号，文档内容可能随 API 演进。
- 当前指南描述 Responses API 可通过 HTTP SSE 流式输出，并展示语义化事件；事件参考列出事件类型和字段，包括 sequence_number。
- 证据等级：厂商当前公开接口文档。它只支持相应 OpenAI API 接口在访问时的公开行为，不证明其他 API 或系统必须沿用相同事件名、字段、保存语义和重连承诺。书中教学事件模型应明确标成应用自行设计的示例。

## LLM 应用安全风险清单

### OWASP GenAI LLM Top 10 2026

- 来源：[OWASP GenAI LLM Top 10 2026 资源页](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/) 与 [2026 年 9 月 1 日发布公告](https://genai.owasp.org/2026/09/01/owasp-genai-security-project-unveils-2026-top-10-for-llm-applications-new-agent-control-standard-and-sponsors-as-community-tops-30000-members/)。
- 资源页标注 2026 版日期为 2026-08-03；项目公告于 2026-09-01 宣布发布，并称该版扩展风险覆盖、更新排序及框架映射。
- 证据等级：社区维护的安全风险和缓解指南。可用于建立威胁词汇和审查候选风险；不能单独证明某个系统存在或不存在风险，也不能代替威胁建模、实现审计或安全评测。

## 可观测性概念

### OpenTelemetry Observability Primer

- 来源：[Observability primer](https://opentelemetry.io/docs/concepts/observability-primer/)，页面标注最近修改于 2026-04-23。
- 该指南将 telemetry signals 说明为 traces、metrics 和 logs，并定义 metric 为一段时间内的数值聚合、SLI 为从用户视角衡量服务行为的指标、span 为一次工作或操作、trace 为请求穿过多个服务的路径。
- 证据等级：OpenTelemetry 概念文档。术语有助于组织埋点，但不规定特定业务必须采用的指标分母、隐私字段或服务级别目标。

## AI 风险管理框架

### NIST AI RMF

- 来源：[NIST AI Risk Management Framework 页面](https://www.nist.gov/itl/ai-risk-management-framework) 与 [AI RMF 1.0 正文](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)，核验日期为 2026-11-05。
- 页面说明 AI RMF 1.0 于 2023-01-26 发布、用于自愿采用，四个高层功能为 GOVERN、MAP、MEASURE 和 MANAGE，并显示 1.0 正在修订。NIST 另于 2024-07-26 发布 Generative AI Profile，作为 AI RMF 1.0 的生成式 AI 配套资源。
- 证据等级：NIST 自愿风险管理框架。四个功能用于组织风险管理活动，不是特定服务的合规清单；正文使用这些术语不能替代行业法规分析或系统安全评测。
