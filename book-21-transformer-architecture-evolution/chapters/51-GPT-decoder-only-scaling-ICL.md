# 第 51 章 GPT 系列：Decoder-only、Scaling 与 In-Context Learning

## 51.1 GPT-1 到 GPT-3：任务适配方式如何变化

GPT 早期发展的主线，不只是参数量逐代增加，而是同一类生成式语言模型如何逐步承担更多任务。任务适配从“先在未标注文本上预训练，再用任务标签微调”，推进到直接测试模型能否从任务格式或上下文示例中作答。

| 阶段 | 任务信息怎样进入模型 | 论文结果支持的历史判断 |
| --- | --- | --- |
| GPT（后来常称 GPT-1，2018） | 在未标注文本上进行生成式预训练，再以任务数据和监督目标微调 | 预训练语言模型可以迁移到多类 NLP 任务，但仍使用任务级监督和适配。[原始论文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| GPT-2（2019） | 在 WebText 上训练较大的语言模型，再用文本格式直接评估零样本任务迁移 | 研究检验了不做任务微调时的迁移能力；作者报告不同任务表现不一，部分结果仍较弱。[技术报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| GPT-3（2020） | 在 zero-shot、one-shot 和 few-shot 条件下，把任务说明及示例作为文本输入 | 论文将固定参数下的上下文任务表现系统化为 ICL 评测；结果随任务而异。[原始论文](https://arxiv.org/abs/2005.14165) |

三项工作共享以语言建模为中心的生成路线，但训练数据、模型规模、下游任务格式和评测方式也在变化。仅凭 GPT-1、GPT-2、GPT-3 的先后顺序，不能把观察到的能力差异单独归因于参数增长。

## 51.2 共同的 Decoder-only 生成接口

这条谱系使用的共同接口是：给定已有 token 前缀，模型预测后续 token。因果可见性限制当前位置读取未来 token，因此一段文本可以同时包含任务说明、输入内容和待生成的答案。分类标签、翻译文本或问答结果都可以被组织为某种续写任务。

这说明了 GPT 路线如何复用语言建模接口承载不同文本任务，不表示所有任务都应该使用 decoder-only，也不意味着任务格式、训练目标和评估标准因此相同。Decoder-only 与 encoder-only、encoder-decoder 的信息流比较见[第十六章](16-Decoder-only与通用LLM.md)；GPT-3 等指定版本的 block 配置见[第十七章](17-LLaMA-GPT-Mistral架构对比.md)。

## 51.3 Scaling laws：先看损失曲线，再看任务指标

早期 scaling 研究测量的不只是“模型变大后看起来更聪明”。Kaplan 等人考察了 held-out cross-entropy 随模型参数、训练数据和计算量变化的经验关系，在指定模型和训练范围内观察到幂律趋势。[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

语言建模损失是一个明确、可连续测量的指标；下游任务准确率、生成质量或产品成功率则是不同的观测对象。GPT-2 的论文报告若干零样本任务表现随模型规模提高，也明确指出部分任务仍未达到实用水平。GPT-3 的 few-shot 结果同样不是每个任务都强。因此，“损失随规模呈规律变化”不能直接改写成“所有能力都按相同曲线增长”。

参数量、训练 token 和预训练计算预算如何匹配，是后续 compute-optimal 研究的主题。Kaplan 等人的分配估计与 Chinchilla 后续实验之间的差异，以及其适用条件，放在[第五十二章](52-Chinchilla-PaLM-compute-optimal.md)讨论。

## 51.4 GPT-3 与 ICL：示例进入当前输入

GPT-3 论文对 zero-shot、one-shot 和 few-shot 进行比较：模型可以只收到任务描述，也可以在待答输入前看到一个或多个任务示例。论文在这些评测中不对模型执行梯度更新或任务微调；变化发生在本次输入上下文中，而不是新生成一份 checkpoint。[论文定义与实验设置](https://arxiv.org/abs/2005.14165)

这里的 in-context learning（上下文学习，ICL）描述的是固定参数模型根据当前上下文生成任务相关输出的行为。它与通过训练样本更新参数的 fine-tuning 不同，但“没有梯度更新”本身并没有说明模型内部具体使用了什么算法。Stanford CS224N 的课程讲义用短示例对照了微调和上下文学习；Stanford AI Lab 的研究者文章则将 Bayesian inference 等解释作为研究框架，而不是已解决的通用机制。[CS224N 课程讲义](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture09-pretraining.pdf)；[Stanford AI Lab：ICL 解释框架](https://ai.stanford.edu/blog/understanding-incontext/)

上下文示例具体提供了哪些信号、模型怎样读取可见 token、示例顺序和位置怎样影响表现，以及不同机制假说各自的证据范围，见[第十二章](12-in-context-learning与显式token检索能力.md)。

## 51.5 GPT 早期结果的证据边界与后续路线

GPT-3 论文既报告了不少任务中的 few-shot 提升，也列出表现较弱的任务，并专门分析了预训练数据与 benchmark 的潜在重叠。作者的分析发现，许多任务的清洁子集分数变化不大，同时也标出需要留意的具体任务和分析限制。合理结论是：基准污染需要审查，不能据此把所有结果说成无效，也不能假设所有成绩都已排除数据重叠。[污染分析与限制](https://arxiv.org/abs/2005.14165)

面试或阅读 GPT 论文时，可以把四个问题分开：模型采用什么架构版本，训练时怎样预测 token，任务信息通过微调还是上下文输入，论文又用什么指标和数据评估。这样既能概括 GPT-1 到 GPT-3 的历史变化，也不会把参数规模、decoder-only、ICL 和下游表现混成同一个原因。后训练如何进一步把基础模型变成助手，见[第五十三章](53-InstructGPT-ChatGPT-RLHF.md)。
