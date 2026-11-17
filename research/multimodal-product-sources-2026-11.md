# 多模态产品化来源记录

检索与阅读日期：2026-11-16。范围：当前公开的图像/语音产品接口、实时语音的状态与打断、评估平台时效、多模态提示注入、生成式 AI 风险管理及内容来源凭证。本记录只保存联网发现的事实和证据，不记录书稿修改或章节审阅过程。

## OpenAI 图像与视觉、图像生成文档

- 来源：[Images and vision](https://developers.openai.com/api/docs/guides/images-vision)，OpenAI API 官方开发者文档，2026-11-16 检索。
- 阅读位置：Overview、图像输入和生成/编辑路线说明。
- 支持的事实：文档把图像分析与图像生成/编辑区分，并列出 Responses、Images 和 Chat Completions 等当前入口及其公开使用范围。
- 证据等级：供应商官方 API 文档，支持其当前公开产品接口说明。
- 边界：页面持续更新，没有固定 API revision；不能据此推断不同 provider、不同模型或下游业务任务的准确率、成本、延迟与授权结论。

## OpenAI 音频、Realtime 与语音交互文档

- 来源：[Audio and voice](https://developers.openai.com/api/docs/guides/audio)、[Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)，OpenAI API 官方文档，2026-11-16 检索。
- 阅读位置：音频工作流选择表、连接方式说明、Realtime session/conversation 状态、Interruption and Truncation。
- 支持的事实：当前文档区分 speech-to-speech agent、既有 text agent 的语音界面、文件转写、实时字幕、翻译和语音合成等路径；浏览器与服务端可选不同传输方式。Realtime 文档说明 session/conversation/response 的状态概念，并解释用户打断时如何取消未完成响应、处理已播放与尚未播放的音频。
- 证据等级：供应商官方文档，支持其特定 API 的公开合同和事件行为。
- 边界：这是单一 provider 的持续更新接口示例，不能据此断言所有实时语音系统都采用同一协议、状态处理或延迟水平；文档能力不代表目标设备、网络、语言和噪声条件下的产品效果。

## OpenAI Evals 平台生命周期

- 来源：[Working with evals](https://developers.openai.com/api/docs/guides/evals)，OpenAI API 官方文档，2026-11-16 检索。
- 阅读位置：文档顶部 Evals deprecation 提示。
- 支持的事实：官方说明 Evals 于 2026-10-31 转为现有用户只读，计划于 2026-11-30 关闭，并建议新建迭代评估的用户关注 Datasets 等替代入口。
- 证据等级：供应商官方产品生命周期文档。
- 边界：此为带日期的平台状态快照，可能变更；通用的任务契约、样本集、分母定义、切片分析、回归和人工复核流程不依赖该产品。该指南的通用评估说明也不能单独证明当前多模态样本 schema 的具体支持范围。

## OWASP 多模态提示注入

- 来源：[LLM01:2025 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/)，OWASP GenAI Security Project，当前词条检索于 2026-11-16。
- 阅读位置：Direct/Indirect Prompt Injections、multimodal AI、prevention and mitigation。
- 支持的事实：OWASP 说明外部文件等输入可携带间接提示注入；多模态模型可能遇到跨模态攻击，例如图像中的隐蔽指令与正常文字共同输入。其缓解建议包括限制模型职责、验证输出格式、分隔可信控制和外部数据、缩小模型可调用的权限；页面也指出不能保证存在万无一失的提示注入防护。
- 证据等级：OWASP 社区安全指南。
- 边界：风险词条不是对某个模型、媒体审核器或目标产品的实测，不是安全认证；攻击后果取决于应用可触达的数据、工具权限和业务动作。指南建议需通过目标系统威胁建模和测试落实。

## NIST Generative AI Profile

- 来源：[Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)，NIST AI 600-1，2024-07-26 发布；官方页面标注 2026-04-08 更新；2026-11-16 检索。
- 阅读位置：官方出版信息、框架适用范围；报告中的治理、测量、隐私和内容来源相关建议。
- 支持的事实：该文件是面向生成式 AI 的跨行业风险管理 profile，作为 AI RMF 的补充资源，供组织自愿采用；行动建议需按相关 AI actor 与其特定应用判断适用性。
- 证据等级：政府研究机构发布的风险管理框架资料。
- 边界：NIST profile 不是法规、认证或某产品的合规结论；其风险类别和建议需结合具体组织、地域、数据和用途分析。

## C2PA Content Credentials 规范

- 来源：[C2PA Content Credentials Technical Specification 2.5](https://spec.c2pa.org/specifications/specifications/2.5/specs/ContentCredentials.html)，C2PA 官方规范，当前页面检索于 2026-11-16。
- 阅读位置：claim signing、hard binding、manifest validation 和 trust model 相关章节。
- 支持的事实：规范定义带签名的 claim/manifest、数字内容绑定和验证流程，可用于检查声明与资产的关联、签名有效性及某些篡改情形，并表示创作或编辑来源信息。
- 证据等级：内容溯源互操作技术规范。
- 边界：规范不替用户判断内容的语义真伪、素材使用授权或版权状态；签名有效只表示在相应信任配置下的声明和资产绑定通过验证。此处关于“验证成功不等于事实/法律结论”的说明是根据规范验证对象范围作出的边界推论。

## 证据范围限制

未运行任何模型、API 请求、媒体处理、业务样本评估或性能 benchmark；未评估特定 provider 的准确率、实时体验、生成质量、安全拦截率、价格或法律适用性。正文中为讲解分母而构造的票据数量和结果均为明确标注的合成教学数据，不是生产经验或行业基准。


## OpenAI 图像 detail 与录音转写接口快照

- **来源与检索日期：** [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)、[File transcription](https://developers.openai.com/api/docs/guides/speech-to-text)，OpenAI 官方开发者文档，2026-11-17。
- **实际阅读位置：** 图像页 detail 参数及坐标/缩放说明；语音页文件转写模型示例、completed recording streaming 与 timestamp 说明。
- **来源支持的事实：** 图像 detail 选项依模型而定，公开值为 low、high、original、auto，省略时默认 auto；original 仍可能按模型像素或 patch 预算缩放，坐标任务要在变换后映回原图。语音文档使用 `/v1/audio/transcriptions` 与 `gpt-transcribe` 展示文件转写；对完成录制的音频设置 `stream=true` 会分段返回 transcript events，输入仍在到达时使用 Realtime transcription。页面在 2026-11-17 还将 `whisper-1`、`gpt-4o-transcribe`、`gpt-4o-mini-transcribe`、`gpt-4o-transcribe-diarize` 和 `/v1/audio/translations` 标为弃用，并称计划于 2027-02-26 关闭。
- **证据等级：** source-backed，适用于页面检索时点的 OpenAI API 公开接口说明。
- **边界：** 文档页面持续更新；detail 参数不是跨模型统一的像素保证，文件转写流式事件不等于麦克风实时会话，弃用/关闭日期也只代表访问时的公开计划；接口说明不支持对准确率、业务字段可靠性、延迟或成本作保证。
