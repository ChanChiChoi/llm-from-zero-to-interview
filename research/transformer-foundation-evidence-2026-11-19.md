# Transformer 基础架构路线的原始证据与边界

检索日期：2026-11-19

本记录只保存原始论文支持的事实及其适用边界，不记录章节审阅和修改过程。

## Transformer 原始论文

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani et al., arXiv v7。
- 阅读位置：摘要、§4 的层复杂度/最长路径比较、机器翻译实验概述。
- 来源支持：论文将 Transformer 用于机器翻译，并与当时的 recurrent/convolutional sequence transduction 模型比较；摘要报告其在两个机器翻译任务上质量更好、可并行化程度更高且训练时间更短。§4 分析了论文所定义的 self-attention、recurrent 和 convolutional 层在复杂度、顺序运算数与路径长度上的差别。
- 证据等级：原始论文的方法分析和作者实验结果。
- 适用边界：实验结果直接支持论文任务、数据、架构和基线；§4 表格限定于所比较的标准层形式，不描述所有现代 recurrent/SSM 变体，也不是任意硬件上的运行时间保证。

## GPT-3 与上下文学习

- 来源：[Language Models are Few-Shot Learners](https://arxiv.org/html/2005.14165)，Brown et al., arXiv v4。
- 阅读位置：摘要、§§1–3、§5 Limitations。
- 来源支持：论文在其 zero/one/few-shot 评估中不对 GPT-3 做梯度更新或 fine-tuning，而是通过文本指定任务并提供示例；作者报告多个任务有较强结果，同时也明确指出有些任务的 few-shot 表现较弱或提升有限。
- 证据等级：原始论文的评测设计、任务结果和作者限制说明。
- 适用边界：只支持论文所测模型、提示与 benchmark 设置中的上下文学习行为；结果本身不识别其神经机制，不能单独证明 attention 是上下文学习的充分原因，也不保证所有任务都能可靠适配。

## LLaMA 的数据与模型披露

- 来源：[LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/html/2302.13971)，Touvron et al., arXiv v1。
- 阅读位置：摘要、§1。本文只记录摘要和引言直接支持的事实，不据此补写未核对的语料组成、过滤比例或 benchmark 数字。
- 来源支持：作者报告训练 7B–65B 参数模型，使用 trillions of tokens，并称其训练使用 publicly available datasets；论文另称向研究社区发布这些模型，并在所列 benchmark 上报告模型结果。
- 证据等级：原始论文作者报告。
- 适用边界：“使用公开可得数据集”说明数据来源类型，不等同于论文发布完整训练语料、筛选后样本或全部数据处理产物；论文基准结果也不能单独归因于 Transformer 架构，或推广为所有模型、数据和训练配置的比较结论。
