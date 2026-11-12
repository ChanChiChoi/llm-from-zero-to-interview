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
