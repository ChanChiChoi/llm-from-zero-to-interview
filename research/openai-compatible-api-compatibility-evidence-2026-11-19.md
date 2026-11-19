# OpenAI 兼容 API：接口与语义边界证据

## 检索范围与方法

- 检索日期：2026-11-19。
- 问题范围：不同服务以 OpenAI SDK、HTTP 路径或请求/响应格式提供兼容接口时，哪些层面可复用，哪些能力或行为仍由服务、端点、模型、版本和区域决定。
- 范围限制：抽查官方 API 参考、官方兼容接口文档、自托管推理服务文档、路由服务文档和一篇新近研究报告；不试图穷尽所有供应商，也未进行在线 API 调用或本地复现。
- 检索平台：Web 搜索；随后打开官方原文和论文原文，按字段、端点、流式事件、错误和能力限制定位证据。
- 本轮实际检索词：
  - `site:developers.openai.com/api/reference Chat Completions Responses API streaming events API reference`
  - `site:platform.claude.com/docs/en openai sdk compatibility layer ignored fields system developer messages`
  - `site:ai.google.dev/gemini-api/docs/openai Gemini OpenAI compatibility unsupported features file upload`
  - `site:api-docs.deepseek.com Responses API compatibility previous_response_id unsupported`
  - `site:docs.ollama.com OpenAI compatibility API OpenAI SDK official`
  - `site:openrouter.ai/docs OpenAI API compatibility endpoints response streaming official`
  - `site:huggingface.co/docs/inference-providers OpenAI compatible chat completion endpoint official`
  - `OpenAI-compatible API provider compatibility matrix error streaming official documentation`
  - `Diátaxis official documentation tutorial how-to explanation reference framework user needs`
  - `CMU Eberly Center writing learning objectives observable actions align activities assessment official`
- 来源筛选：优先选择供应商、服务维护者、标准/课程组织或论文作者的一手资料；搜索摘要只用于发现来源，关键事实均回到原文核对。没有把论坛、榜单或聚合页面作为事实依据。

## 外部主题和证据

### 1. OpenAI API 提供多个接口表面，参考对象需先限定到端点

- [OpenAI API Overview](https://developers.openai.com/api/reference/overview)，官方在线文档，访问于 2026-11-19。概览按应用任务区分 Responses、Realtime 和 Administration，并将端点、请求/响应 schema、流式事件、认证、错误、限速和请求 ID 分开作为参考对象。
- [Chat Completions 创建接口](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)，官方在线参考，访问于 2026-11-19。该端点接收消息列表，返回 completion 对象或流式 chunk；流式以 server-sent events 传送。参考同时说明参数支持会因模型而异。
- [Responses 流式事件参考](https://developers.openai.com/api/reference/resources/responses/streaming-events)，官方在线参考，访问于 2026-11-19。Responses 流包含带类型和序号的事件，例如 `response.created`、`response.in_progress`、`response.completed` 与 `error`；这是特定端点的事件结构，不能从 Chat Completions 的 chunk 形状直接推出。
- 证据边界：OpenAI 参考描述 OpenAI 自有接口，不自动证明其他服务实现了这些端点或字段；在线参考会更新，正文引用时需记录访问日期或固定版本。

### 2. 官方兼容接口把“兼容”限定在有明示范围内

- [Anthropic OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)，Anthropic 官方文档，访问于 2026-11-19，页面内容为动态文档。说明该兼容层用于通过 OpenAI SDK 测试和评估 Claude API。文档列出多处行为差异：`system` / `developer` 消息会从会话中的位置移出并合并到顶层 system；一些参数被忽略；`strict` 工具参数被忽略；音频和文件输入会报 400；错误格式相近但详细错误信息不等同。温度、`top_p`、强制工具选择等条件还会随具体模型而变化。
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)，Google 官方文档当前重定向到 [AI Studio OpenAI compatibility](https://aistudio.google.com/docs/openai)，访问于 2026-11-19。文档称可通过 OpenAI Python/TypeScript SDK 或 REST 调用部分 Gemini 能力，同时建议新集成直接使用 Gemini API；页面明确称 OpenAI 库支持仍处于 beta。文档给出 Chat Completions、流式、函数调用、图像输入、embeddings 等示例，也逐项列出未覆盖路径，例如特定模型的视频生成不支持兼容 `/v1/videos`，Files API 的上传和下载不支持兼容接口。`reasoning_effort` 的行为依模型而变。
- [DeepSeek Responses API](https://api-docs.deepseek.com/guides/responses_api/)，DeepSeek 官方文档，访问于 2026-11-19，动态文档。文档称其 Responses 格式部分兼容 OpenAI，并明示服务为无状态：不支持 `previous_response_id` 和 `conversation`；`store` 固定为 false；`parallel_tool_calls` 被忽略且并行工具调用始终启用；工具种类和 reasoning 字段为部分支持，能力随工具/模型条件而异。
- 证据边界：上述是各文档对各自服务、文档版本和模型条件的说明；不能据此概括所有兼容服务。页面中“支持”的字段也不证明与 OpenAI 的语义、默认值、错误细节或质量表现相同。

### 3. 自托管服务可复用客户端形式，但还受模型与部署实现制约

- [vLLM v0.30.0 OpenAI-Compatible Server](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/)，官方固定版本文档。文档将接口按 `/v1/completions`、`/v1/chat/completions`、`/v1/responses`、embeddings、audio 等分别列出，并限定适用模型类型。Chat Completions 需要文本生成模型和 chat template；该页写明 `user` 被忽略、Completions 的 `suffix` 不支持。额外采样参数可经客户端 `extra_body` 传递；模板配置和模型仓库中的 `generation_config.json` 也会影响实际行为。
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index)，官方在线文档，访问于 2026-11-19。其 OpenAI 兼容端点面向 chat completion，路由器可自动选择 provider；其他任务（如 embeddings、图像或语音）需使用相应的 Inference Client。文档还允许选择或约束 provider，说明兼容路由可能同时包含模型服务选择这一层。
- 证据边界：vLLM 的结论固定在 v0.30.0 文档和相应服务实现；Hugging Face 的任务与路由说明是访问时的在线状态。不能推断其他自托管框架、服务版本或模型均相同。

### 4. 供应商扩展、地区和具体模型可能改变可移植性

- [Alibaba Cloud Model Studio OpenAI-compatible Chat](https://help.aliyun.com/en/model-studio/qwen-api-via-openai-chat-completions)，阿里云官方动态文档，访问于 2026-11-19。页面描述按地区区分的 base URL，并列出非 OpenAI 标准字段，例如 `enable_thinking`、`search_options` 和 `include_tool_usage`，通过 OpenAI SDK 时部分字段放入 `extra_body`。文档同时指出可用模型、参数默认值和能力因模型而异，并明确 Qwen-Audio 不支持 OpenAI-compatible protocol。
- [Alibaba Cloud Model Studio base URLs](https://help.aliyun.com/en/model-studio/base-url)，阿里云官方动态文档，访问于 2026-11-19。端点地址按区域、工作空间和产品方案区分；不能把一个示例 base URL 当作所有账户或地区的通用值。
- 证据边界：字段和区域信息是访问日期时的产品文档，不代表稳定协议；使用时须核对具体模型、地区、工作空间、产品方案和接口版本。

### 5. 新近研究把“请求可被接受”与“对话信息被保留”区分开

- Xin Liu 等，[*Testing Conversation Preservation Across LLM APIs: A Rule-Based Oracle for Detecting Silent Information Loss*](https://arxiv.org/abs/2611.15543)，arXiv:2611.15543v1，2026-11-11，cs.SE / cs.AI 预印本，访问于 2026-11-19。
- 作者提出将源、目标请求与回复映射到共享表示，再按规则检查信息是否保留。主请求研究涵盖五个 translator codebase 的六种配置、14 条源到目标路径和合成请求形状；其报告中，685 个已完成翻译有 127 个（18.5%）通过静态目标 schema 检查却丢失至少一项作者定义为可避免的 prompt、role 或 tool 信息。另一项独立流式实验中，558 个翻译后的回复流有 203 个（36.4%）在至少一个被测官方 OpenAI SDK 中抛出异常或造成静默损坏。
- 作者明确将主请求集称为合成套件，并说明该比例不是现实流量中的信息丢失率；覆盖范围、字段表示、捕获点、translator 版本和目标能力都会限制解释。论文还说明未测所有商业托管网关，也未外部探测 Anthropic 自有兼容层或 OpenAI 自身层。该预印本提出“对话信息保留”这一可测试属性，但不能证明所有兼容层都存在相同问题或给出全行业发生率。

## 组织与教学资料

- [Diátaxis](https://www.diataxis.fr/)，框架作者的官方文档，访问于 2026-11-19。区分教程、how-to、参考和解释四种文档需求，并明确读者不必按固定线性顺序进入这些材料。它适合帮助判断接口章节中的操作教程、机制解释和规格查询分别服务哪类任务；它是技术文档框架，不是教材目录模板，也不是章节次序提高学习效果的因果证据。
- [Carnegie Mellon Eberly Center: Learning Objectives](https://www.cmu.edu/teaching/designteach/syllabus/newcourse/learningobjectives.html)，大学教学设计指南，访问于 2026-11-19。建议学习目标以学生为中心、使用可观察的动作并可测量，并使目标、评估和教学活动相互对齐。它支持把“理解兼容性”转化为可检查的比较/判断能力；不直接验证某种 API 章节编排适用于所有读者。
- [OpenAI API Overview](https://developers.openai.com/api/reference/overview) 本身也按接口表面、端点参考、schema、流式事件、认证、错误、限速和请求 ID 提供查询入口。它是官方 API 文档的组织实例，不证明初学者课程应照搬其结构。

## 尚未核实的事实边界

- 没有对任何供应商端点发起请求；本记录不包含本地复现、端到端兼容测试或当前服务可用性结论。
- 兼容能力会随 endpoint、API 版本、SDK 版本、provider、model、地区、workspace、服务配置和时间变化；文档示例不构成所有组合的保证。
- “字段被接受”可能意味着处理、部分处理、忽略、固定输出、报错或映射到不同语义；仅从 JSON schema、SDK 调用成功或响应字段同名，不能推断语义等价。
- 本轮抽样不穷尽所有供应商、模型家族、API 端点、代理/网关或流式实现。没有证据的具体组合保持未知。
