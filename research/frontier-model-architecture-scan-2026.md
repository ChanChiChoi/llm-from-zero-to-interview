# 前沿架构与模型快照证据（2026）

检索日期：2026-10-30。以下记录只收录本轮册级扫描中确认的新证据；产品发布和模型卡不能单独证明独立性能或未公开实现。

## Gemini 4 Argon

- 来源：[Gemini 4 Argon Model Card](https://deepmind.google/models/model-cards/gemini-4-argon/)，Google DeepMind，发布于 2026-10。
- 阅读位置：Model Information / Architecture；Implementation and Sustainability / Hardware、Software。
- 支持事实：卡片称其为 large-scale sparse MoE Transformer，原生支持文本、视觉和音频输入；训练使用 TPU、JAX 与 ML Pathways。卡片没有公开参数量或完整层配置。
- 证据等级：官方模型卡。
- 边界：模型卡中的评测和训练描述是发布方披露；不构成独立复现，也不能推出具体稀疏路由、推理 kernel 或线上吞吐。

## Mistral Large 4

- 来源：[Mistral Large 4 发布公告](https://mistral.ai/news/mistral-large-4/)（2026-10-06）；[模型文档](https://docs.mistral.ai/models/mistral-large-4)；[权重发布公告](https://mistral.ai/it/news/mistral-large-4-oss/)（页面于本次检索时已发布）。
- 阅读位置：发布公告的模型规模、训练与待发布说明；模型文档顶部规格；权重发布公告的模型版本说明。
- 支持事实：模型文档快照列出 1.05T 总参数、52B active 参数、1.6B 视觉编码器、1M context 和 public preview；Mistral 后续公告称已发布经过额外 RL 的权重版本。NVIDIA 官方模型页将 49B 说明为每 token 使用的专家参数，并将约 52B 计入 embeddings 与输出层，因此 49B/52B 应先按统计口径解读。来源：[Mistral/NVIDIA 公告](https://mistral.ai/news/mistral-nvidia-ai-for-industrial-engineering/)，[NVIDIA 模型页](https://docs.api.nvidia.com/nim/reference/mistralai-mistral-large-4-1t-a52b)。
- 证据等级：发布方公告、官方模型文档和官方权重入口。
- 边界：Mistral 文档与 NVIDIA 页面采用不同字段口径，预览服务与后续开放权重也可能不是同一 artifact；正式写入时须固定 revision 并读取对应 config/model card，不能把不同交付物的数据拼在一起。发布方 benchmark 不代表独立复现。

## GPT-6.1 Terra

- 来源：[Introducing GPT-6.1 Terra](https://openai.com/index/introducing-gpt-6-1-terra/)，OpenAI，2026-10-26。
- 阅读位置：From visual understanding to action；A significant step up on factuality and coding；Safety and alignment；Pricing and availability。
- 支持事实：官方介绍聚焦视觉理解、电脑操作、写作、延迟/成本与发布方评测；该公告未披露模型内部架构。
- 证据等级：官方产品发布说明。
- 边界：发布方评测需保留 benchmark、effort、工具和价格条件；不能据产品用途推断 Transformer 结构、参数或训练配方。

## Claude Haiku 5.5

- 来源：[Anthropic Transparency Hub](https://www.anthropic.com/transparency) 中 Claude Haiku 5.5 系统卡摘要；[Claude Haiku 5.5 发布页](https://www.anthropic.com/claude-haiku-5-5)，Anthropic，2026-10。
- 阅读位置：Transparency Hub 的 Haiku 5.5 系统卡摘要及发布页产品介绍。系统卡 PDF 本轮因文件过大无法读取，未把全文视为已核验。
- 支持事实：官方摘要列出文本和图像输入，并概述预训练与后训练；摘要没有披露确切层结构、参数规模或完整训练配方。
- 证据等级：官方系统卡摘要与官方发布页；PDF 全文未读。
- 边界：发布定位和摘要能支持的输入/训练概述不能代替完整系统卡，也不支持架构或独立性能结论。

## GPT-6.1 Terra

- 来源：[Introducing GPT-6.1 Terra](https://openai.com/index/introducing-gpt-6-1-terra/) 与[官方 API 模型页](https://developers.openai.com/api/docs/models/gpt-6.1-terra)，OpenAI，2026-10-26。
- 阅读位置：发布页的 visual reasoning/computer-use 说明；API 模型页的 model ID、上下文与最大输出字段。
- 支持事实：发布页强调视觉推理和电脑操作；API 文档列出 1,050,000-token context window 与 128K maximum output tokens。
- 证据等级：官方发布说明与 API 文档。
- 边界：两页均未披露内部架构、参数量或完整训练配方；接口能力和上下文字段不构成架构证据。
