# Gemini、GPT-4V 与 Claude 3 的公开披露证据

## 研究问题与范围

核对 2023—2024 年 Gemini 1.0/1.5、GPT-4V 与 Claude 3 的公开材料：哪些资料直接说明架构，哪些只说明输入/输出接口或发布状态，长上下文和视觉结果各绑定什么实验。范围是历史版本的公开披露，不追踪 2026 年当前 API 行为，也不从外部行为反推未披露结构。

- 检索日期：2026-11-24。
- 检索方式：网页搜索；打开论文、官方模型卡、system card、模型发布页及大学课程资料原文。
- 关键词：`Gemini 1.0 technical report architecture jointly multimodal`; `Gemini 1.5 sparse MoE context 2403.05530 v1`; `GPT-4V system card image training architecture disclosure`; `Claude 3 model card vision 200k 1M needle`; `vision-language model lecture notes architecture`; `long-context needle test multi-document QA limitations`。
- 时间与版本：Gemini 1.5 arXiv v1 于 2024-03-08 提交；arXiv 页面说明最新版本为 v5（2024-12-16）。正文引用 v1 固定历史语境，避免把后续修订和 2024 年 2 月产品 preview 混为一条发布记录。

## 来源与支持边界

| 来源 | 类型/版本 | 支持的主张 | 不支持的主张 |
| --- | --- | --- | --- |
| [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | Google 技术报告，2023；§2、附录模型卡 | Gemini 1.0 decoder-only Transformer、高层注意力/上下文描述、跨图像/音频/视频/文本联合训练、视频帧和视觉输入描述 | 未列出的内部层表、所有 Gemini 产品/API 版本一致的实现 |
| [Gemini 1.5 Technical Report v1](https://arxiv.org/abs/2403.05530v1) | Google 技术报告，2024-03-08 初版；§1、§4.2.1.2、Figure 1 | 1.5 Pro 的 MoE 模型描述；报告中合成 NIAH 及长上下文任务的特定结果；报告区分检索与更复杂任务 | 10M 是常规产品窗口；一个 NIAH 结果代表长文综合或普遍能力；未明示的专家数/路由细节 |
| [Introducing Gemini 1.5](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/) | Google 官方发布说明，2024-02-15 | Gemini 1.5 Pro 标准 128k 窗口；最多 1M 仅在当时 private preview 中向受限客户开放；公开发布时的 MoE 解释 | 实验报告中的更长上下文就是当时所有客户可用的接口；内部具体 MoE 配置 |
| [GPT-4V System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf) | OpenAI system card，2023-09-25；§1 | GPT-4V 图像输入、高层训练说明（文本/图像预训练数据、随后 RLHF）及视觉部署安全评估范围 | 完整视觉 encoder、token 化或融合位置 |
| [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | OpenAI 技术报告，2023-03；§2 | GPT-4 的 Transformer-style 高层描述，以及作者明确未披露架构细节（包括模型规模）、硬件、训练计算、数据构造和训练方法 | GPT-4V 的内部视觉结构，或对 GPT-4V 图片处理路径的具体实现结论 |
| [Claude 3 Model Card](https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf) | Anthropic 模型卡，2024-03；§5.3、§5.8 | Claude 3 视觉输入、文本输出和视觉任务评测；卡中生产上下文状态及 Opus 的长上下文结果 | 模型内部视觉连接结构，或将评测直接外推到其他版本 |
| [Introducing the next generation of Claude](https://www.anthropic.com/news/claude-3-family) | Anthropic 发布页，2024-03-04 | 200k 初始发布窗口；超过 1M 的能力与选择性开放措辞；NIAH 的 30 组随机 needle/question 配对之一及众包语料描述 | 所有用户在发布时都可用 1M；与 Gemini 协议一致的可比 NIAH 分数 |
| [Stanford CS231N Lecture 16: Multimodal Foundation Models](https://cs231n.stanford.edu/slides/2025/lecture_16.pdf) | 大学课程讲义，2025-05-27 | 作为视觉语言模型高层讲解路线参考，查看编码器、连接器和语言模型数据流怎样用图示呈现 | Gemini、GPT-4V 或 Claude 3 的具体内部架构 |
| [Lost in the Middle](https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.tacl2023.pdf) | TACL 研究论文，2023；摘要、§2、结论 | 指定旧版本模型上的多文档问答与 key-value 检索表明，相关证据位置会影响结果；说明上下文长度与使用效果不同 | 对 Gemini 1.5 或 Claude 3 的直接性能结论 |
| [Lost in the Middle, or just lost?](https://www.ycmak.net/how-lost-in-the-middle/) | 实践者技术博客，2025-01 完成；§“Existing research” | 以个人长上下文实验和研究综述辅助构思“广告长度/窗口”与任务级检索效果的讲解桥接 | 普遍生产性能或未逐篇回查的第三方模型排名 |

## 核心核验结果

1. **Gemini 1.0 的报告披露高层结构。** §2 将其描述为 decoder-only Transformer，报告最高 32k 上下文、multi-query attention、跨图像/音频/视频/文本联合训练，并描述视频帧序列和视觉编码的研究来源。应把“报告如此描述”与完整实现细节分开。
2. **Gemini 1.5 的模型类别和产品开放状态是两类证据。** v1 报告称 Gemini 1.5 Pro 为 MoE 模型，并报告合成 retrieval 测试；Google 2024-02-15 发布页则将 128k 列为标准窗口、1M 列为 private preview。不能把 v1 的 10M 测试结果改写成常规产品接口。
3. **GPT-4V 可确认图像输入与高层训练流程，不能补出视觉子结构。** GPT-4V system card 报告图像输入及预训练/后训练概述；底层 GPT-4 报告明确列出其未公开的架构与训练细节。两份资料应合读，但不能把安全评估材料当作架构蓝图。
4. **Claude 3 资料主要支持接口和发布方评测。** 官方卡/发布页确认图片输入、文本输出、200k 发布期窗口和选择性长输入主张。NIAH 结果由 Anthropic 指定的随机配对和众包语料产生，不与 Gemini 的测试分数直接排名。
5. **NIAH 不等于一般长上下文理解。** Gemini v1 将合成针测与长文/视频等较真实任务分开报告；Claude 页也给出其自己的协议。正文将指标限定为该测试的召回观察，并回链本册第十三章与第七册相应评估章节。

## 教学组织参考与未验证边界

Stanford CS231N 讲义用于观察怎样以视觉特征、连接层和语言模型串起一般 VLM 数据流；本章不复述这套通用架构，而以 Gemini 的报告披露与 GPT-4V/Claude 3 的未知项作对照。TACL 的位置控制实验和实践者长上下文文章帮助把“可接收长度”与“在任务中使用证据”分开讲；前者的模型并非 Gemini 1.5/Claude 3，后者的结果不作为供应商事实。

尚未核验或不能由上述来源确定的内容：Gemini 1.5 的精确专家数与路由拓扑；GPT-4V 与 Claude 3 的视觉编码器、融合层位置和图像 token 细节；不同 NIAH 结果在相同数据、模型 revision、输入预处理和测量口径下的直接对比。正文均保留为未知。
