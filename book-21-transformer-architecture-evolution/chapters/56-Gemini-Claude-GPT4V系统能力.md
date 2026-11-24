# 第 56 章 Gemini、GPT-4V 与 Claude 3：多模态与长上下文的公开证据

本章固定阅读 2023—2024 年公开资料中的几个具体对象：Gemini 1.0 与 1.5、GPT-4V，以及 Claude 3。重点是辨认这些资料分别公开了模型结构、输入接口、产品可用范围和实验结果中的哪些部分。它们不是同一个时间点、同一种披露深度的三个 checkpoint；当前产品页面和 API 已发生变化，本章的容量与能力陈述均按资料发布日期理解。

## 56.1 先固定模型版本与证据对象

读闭源模型资料时，先问“这句话描述的对象是什么”。技术报告中的模型家族和实验版本、system card 中的系统准备工作、发布页中的产品接口，往往回答不同问题。它们可以互相补充，但不能把发布页的能力声明直接变成内部架构图，也不能把实验版本的结果当作产品默认设置。

| 资料类型 | 可以直接确认 | 不能单独推出 |
| --- | --- | --- |
| 技术报告、模型卡 | 发布方写明的模型类型、训练或输入输出描述，以及报告中的实验条件 | 没有披露的模块连接、精确参数、路由实现或所有产品版本的共同结构 |
| system card | 指定系统的输入能力、安全准备与报告过的测试 | 完整的模型架构或训练配方 |
| 发布说明、产品页 | 发布时的模型名、接口和开放范围 | 所有用户都能使用的能力、内部模块组成或跨版本不变的限制 |
| 论文评测 | 指定模型、任务、数据与协议下的观察 | 对其他任务、版本或供应商的通用排名，也不能单独证明某个架构组件造成结果 |

以下按发布时间分别阅读来源：Gemini 1.0 技术报告发表于 2023 年；Gemini 1.5 的初版技术报告标为 2024 年 3 月 8 日；GPT-4V system card 发布于 2023 年 9 月；Claude 3 在 2024 年 3 月发布。锁定这些版本，才不会把后来产品的能力倒填进早期模型。

## 56.2 Gemini：从联合多模态训练到稀疏 MoE 长上下文

Gemini 1.0 技术报告给出了比“支持图片输入”更具体的高层结构描述：模型基于 decoder-only Transformer，报告写明使用最高 32k 上下文和 multi-query attention；模型从训练之初就联合使用图像、音频、视频与文本数据。报告还说明视频按帧序列进入上下文、输入分辨率可变，视觉编码借鉴 Flamingo、CoCa 和 PaLI 等工作。这里能确认的是论文公开的模型级描述；“借鉴”并不等于复制这些系统的完整结构，也不足以推出所有模态共享完全相同的编码路径。[Gemini 1.0 技术报告](https://arxiv.org/abs/2312.11805)

Gemini 1.5 技术报告把公开结构推进到“稀疏 MoE Transformer”这一层，并将长上下文能力与架构、训练和 serving 方面的变化联系起来。它足以支持“报告将该版本描述为稀疏 MoE Transformer”，但不应仅凭 MoE 这个名称补出专家数量、路由拓扑或每层配置。MoE 的路由与计算路径见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)，本章只保留它作为模型版本披露的一项结构事实。[Gemini 1.5 技术报告 v1](https://arxiv.org/abs/2403.05530v1)

两个版本之间的变化说明，家族名不能替代版本记录：Gemini 1.0 报告强调多模态训练和 decoder-only 主干；Gemini 1.5 报告进一步披露稀疏 MoE，并报告面向长上下文的实验。发布说明、技术报告和后续产品界面要分别标注，不能把其中一类材料的信息移到另一类对象上。

## 56.3 GPT-4V 与 Claude 3：图像能力公开到哪里

OpenAI 的 GPT-4V system card（2023 年 9 月）明确说明用户可以要求 GPT-4 分析图像输入。它也披露了训练流程的高层信息：预训练使用互联网及授权来源的文本和图像数据，随后使用 RLHF 微调。system card 的主题是视觉能力部署前的准备、评估与缓解；它没有给出图像编码器、视觉 token 如何接入主干等足以复原内部数据流的细节。[GPT-4V system card](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

底层 GPT-4 技术报告说明模型属于 Transformer 风格，但明确表示不进一步披露架构细节（包括模型规模）、硬件、训练计算量、数据构造和训练方法。因此，GPT-4V 的公开资料可以支持“GPT-4 接收图像，并且其预训练数据包含文本和图像”这样的有限结论；它不支持对视觉编码器或融合位置作确定判断。[GPT-4 技术报告](https://arxiv.org/abs/2303.08774)

Claude 3 的 2024 年 3 月模型卡与发布说明确认了另一类信息：Opus、Sonnet 和 Haiku 都有视觉输入能力，用户可把图片与文字提示一起提交，模型以文本作答；模型卡也报告了若干视觉任务结果。材料说明了公开接口和评测对象，却没有提供一张足以还原图像如何编码、在哪里与语言表示融合的架构图。因而“具备视觉输入”是可确认的接口事实，“内部采用某种特定视觉 encoder 或连接器”仍属未公开信息。[Claude 3 模型卡](https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf)；[Claude 3 发布说明](https://www.anthropic.com/news/claude-3-family)

这两个案例也说明，系统卡和模型卡不一定以披露内部结构为目标。GPT-4V 的公开资料偏向图像输入和部署评估；Claude 3 的资料偏向模态接口与能力评测。披露侧重点不同，不等于其中一个模型必然采用某种“更原生”的多模态架构。一般视觉语言模型的编码与融合机制由[第十五册第四章](../../book-15-multimodal-generative-models/chapters/04-vlm架构.md)和[第十五册第十三章](../../book-15-multimodal-generative-models/chapters/13-encoder-free-unified-multimodal.md)展开。

## 56.4 长上下文：发布上限与检索实验分开读

发布时的上下文窗口描述接口或当时的开放范围。Google 在 2024 年 2 月介绍 Gemini 1.5 Pro 时，给出的标准窗口是 128k；最多 1M token 的访问仅向一小部分开发者和企业客户提供 private preview。技术报告中的更长序列实验回答的是研究测试能否处理多长输入，不等于普通用户当时可以提交同样长度的请求。[Gemini 1.5 发布说明](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/)；[Gemini 1.5 技术报告 v1](https://arxiv.org/abs/2403.05530v1)

Gemini 1.5 v1 报告的 needle-in-a-haystack 测试把一条特定信息插入长输入，再按不同位置询问该信息。报告在 1M token 时展示了文本、视频和音频的近乎完全召回，并将文本测试延伸到 10M token；音频和视频结果分别延伸到约 9.7M 和 9.9M token。这是特定合成检索任务上的作者报告结果，不应改写成“所有模态都在 10M token 上达到同一成绩”，也不能据此断言产品接口支持 10M token 或一般长文推理已经解决。

Anthropic 在 2024 年 3 月发布 Claude 3 时，将 200k 描述为初始提供的上下文窗口；模型卡称当时生产环境提供 200k，超过 1M 的输入能力则可能面向特定客户开放。Anthropic 还报告 Claude 3 Opus 在其 NIAH 测试中超过 99% 的召回率，并说明每个 prompt 使用 30 组随机 needle/question 配对之一及众包文档语料。[Claude 3 模型卡](https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf)；[Claude 3 发布说明](https://www.anthropic.com/news/claude-3-family)

两个发布页中的 token 数和两个 NIAH 结果不可直接拼成排行榜：产品开放状态、模型版本、needle 构造和语料都不同；单条信息检索也没有覆盖多文档综合、位置变化下的稳定性或长序列任务成本。长距离依赖与位置效应见[第十三章](13-长距离依赖与lost-in-the-middle.md)，评测协议见[第七册第七章](../../book-07-evaluation-experiments/chapters/07-多模态与长上下文评估.md)和[第十三章](../../book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md)。

## 56.5 让公开证据支撑有限的架构判断

这几组公开材料呈现了不同披露层次：Gemini 1.0 报告公开了 decoder-only 主干、部分注意力与多模态训练描述；Gemini 1.5 报告把模型类型明确为稀疏 MoE Transformer；GPT-4V 与 Claude 3 的资料则足以确认视觉输入和发布方报告的行为，但不足以还原完整视觉数据流。读者可以比较各来源实际披露了什么，不能把产品标签补成缺失的层结构。

阅读新的闭源模型发布时，先把结论归入三类：公开的结构事实、绑定版本与接口的产品事实、绑定任务和协议的实验观察。第四类——未公开的结构、训练细节或路由机制——应保留为未知。关于 Agent 动作和执行器的边界见[第四十九章](49-Agentic-Model-Architecture.md)；跨家族架构比较由[第五十九章](59-主流模型架构创新横向对比.md)承接。
