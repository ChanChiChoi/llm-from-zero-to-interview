# Qwen3.7 Plus 托管 API 来源记录

记录日期：2026-11-12

## Alibaba Cloud Model Studio 模型页

- **来源：** [Qwen3.7 Plus](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)，动态官方文档；访问日期 2026-11-12。
- **阅读范围：** Model Capabilities 与 Context Limits 表、服务区域/范围条目；未把价格表作为本记录的稳定事实。
- **来源支持：** 页面将 `qwen3.7-plus` 说明为与 `qwen3.7-plus-2026-05-26` snapshot 功能等价；输入模态列为 text/image/video，输出模态为 text。产品说明将读屏、GUI 交互、视觉参考生成代码和移动端导航列为模型定位。Context Limits 当前列出 1,000,000 Context Window、991,808 Max Input Length、131,072 Max Output Length；Thinking Mode 另列 983,616 Max Input Length、131,072 Max Output Length 和 262,144 Max Chain-of-Thought Length。
- **能力范围：** 页面按服务区域及 scope 展示 Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching、Batch Inference 等支持项；具体支持状态会随部署条目变化。调用与复现实验应记录 snapshot、region、scope、endpoint 和 mode。
- **证据等级：** source-backed（provider 对指定托管服务的公开产品和 API 合同说明）。
- **边界：** 页面为动态文档，记录仅表示访问日可见的字段；Context Window、Max Input Length、Max Output Length 和 thinking 字段是不同合同项，不可自行相加。产品定位不公开模型参数、训练配方、GUI policy、内部视觉路径、生产 kernel 或真实设备上的任务成功率。Structured Outputs 支持不证明 action 语义正确、坐标有效或动作已授权。

## Alibaba Cloud Context Cache

- **来源：** [Context Cache](https://www.alibabacloud.com/help/en/model-studio/context-cache)，动态官方文档；访问日期 2026-11-12。
- **阅读范围：** prefix matching、支持模型与部署范围、最小前缀条件和 cache hit 说明。
- **来源支持：** 文档描述 provider-managed context cache 对请求消息中的共同前缀进行匹配；达到文档列出的最小 token 条件只是满足写入/命中的技术前提，不保证实际命中。支持的模型列表按 deployment scope 变化。
- **证据等级：** source-backed（访问日的 provider 服务说明）。
- **边界：** 文档不披露内部缓存 key、驱逐实现或命中保证；该功能不等同于 GPU KV cache 或应用级会话 memory。具体模型、区域、请求和缓存命中应按当前 API 响应与合同核验。

## 2026-11-17 地域与 scope 能力矩阵复核

- **来源：** [Alibaba Cloud Model Studio Qwen3.7 Plus](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)，页面标注 Last Updated: 2026-10-11；2026-11-17 阅读 Model Capabilities 表。
- **来源支持：** 文档在 US (Virginia) 分别列出 Global 与 US scope。Global scope 表列 `Structured Outputs` 和 `Web Search` 为 supported，US scope 表列两项为 unsupported。两张表的输入模态行只是列项顺序不同（Global 写作 Image/Text/Video，US 写作 Text/Image/Video）；它们列出的模态集合相同，不据此推断输入模态能力差异。China (Beijing)、Singapore (International) 与其他地域另有各自能力表。
- **证据等级与边界：** source-backed（供应商当前动态模型/API 页面）。差异说明调用合同按 region/scope 细分；页面没有说这代表模型权重或推理质量不同。发起请求时仍需记录 endpoint 和部署范围，并以当日文档/请求结果为准。
