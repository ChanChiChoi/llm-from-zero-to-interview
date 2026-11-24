# GPT 谱系、Scaling 与 ICL 研究记录

## 研究问题与范围

研究 GPT（后称 GPT-1）、GPT-2 和 GPT-3 在 2018—2020 年间如何把生成式预训练连接到下游任务；确认早期 scaling 研究测量了什么；核对 GPT-3 论文中 ICL 的评测定义和限制。资料检索与阅读日期：2026-11-23。研究用来支持第二十一册第五十一章的历史脉络、章节边界和主张限定，不试图综述现代 GPT 产品或所有 ICL 机制。

检索词包括：Improving Language Understanding by Generative Pre-Training、GPT-2 unsupervised multitask zero-shot、GPT-3 few-shot no gradient updates、Scaling Laws cross-entropy model size data compute、GPT-3 in-context learning lecture notes、in-context learning mechanism Stanford。

## 主要发现

1. **GPT（后称 GPT-1）：生成式预训练后进行任务适配。** 2018 年论文使用带 masked self-attention 的 decoder-only Transformer，以语言建模目标进行生成式预训练，再通过任务输入变换和监督目标适配分类、蕴含、相似度等任务。论文标题没有使用“GPT-1”编号；正文宜说明这是后来的谱系称呼。证据支持“预训练表示可迁移并经监督微调用于若干 NLP 任务”，不支持把它说成已经采用 GPT-3 式 ICL。
2. **GPT-2：把零样本迁移作为重点评估对象。** 2019 年报告描述 1.5B 参数模型在 WebText 上训练，并以任务文本格式评估多项下游行为，而不对每个任务再做监督训练或改动模型。作者同时报告不同任务结果差异，并明确指出部分任务的量化表现仍很弱；不能把零样本迁移表述为普遍可用的能力。
3. **GPT-3：系统比较 zero-shot、one-shot 与 few-shot。** 2020 年论文评估 175B 参数自回归模型，将任务说明和若干示例作为文本输入，在评测时不进行梯度更新或 fine-tuning。论文既报告多项任务的提升，也记录表现弱项和 benchmark contamination 分析。它支持“在论文任务与设置中观察到上下文条件化行为”，不证明所有新任务都能可靠学习，也不确定单一内部机制。
4. **Scaling laws 的直接观测对象是语言建模损失。** Kaplan 等人研究 held-out cross-entropy 随模型参数、数据规模和训练计算变化的经验关系，并在其模型、数据和训练设定中拟合幂律趋势。趋势可支持对相应损失的条件化预测，不能直接等同于所有下游任务、用户价值或产品能力都服从同一曲线。Hoffmann 等人的 Chinchilla 工作重新研究固定计算预算下参数量和 token 数的配置；具体计算最优分析留给第五十二章。
5. **ICL 的教学表述应区分行为与机制。** GPT-3 论文和 Stanford CS224N 讲义都将 ICL 描述为在前向推理中根据 prompt 内示例产生任务输出、没有该次参数梯度更新。Stanford AI Lab 的研究者文章介绍了一种隐式 Bayesian inference 框架，但把它作为解释框架提出，不能当作适用于全部模型和任务的已证实机制。

## 来源与证据边界

| 来源 | 类型与阅读位置 | 支持的内容 | 不支持的外推 |
| --- | --- | --- | --- |
| [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)（Radford 等，2018） | 原始论文；模型定义、§3.1—3.3、模型规格 | decoder-only masked self-attention、生成式预训练、监督任务微调与任务输入变换 | 后续 GPT 版本结构或 GPT-3 式 ICL |
| [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（Radford 等，2019） | 原始技术报告；摘要、§1—3、§6—7 | GPT-2/WebText、零样本测试和作者报告的任务差异 | “所有任务都可零样本解决”或现代模型的行为 |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)（Brown 等，2020） | 原始论文；摘要、§2—5、contamination analysis | GPT-3 的 zero/one/few-shot 设计、无需梯度更新的评测、优势/弱项及污染分析 | 任意任务迁移、无污染 benchmark、单一 ICL 内部算法 |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)（Kaplan 等，2020） | 原始论文；摘要、§1.1—1.2、规模与计算分配分析 | 指定实验范围中 cross-entropy 与模型、数据、计算量的经验关系 | 跨任务通用能力定律或永久有效的参数/token 比 |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)（Hoffmann 等，2022） | 原始论文；摘要、§3—4 | Chinchilla 对固定计算预算下参数与训练 token 配比的再估计 | 不同目标、数据质量、推理需求下的普遍最优方案 |
| [GPT-3, In-context learning, and very large models](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture09-pretraining.pdf)（Stanford CS224N，2025） | 课程讲义；第 56 页附近 | 将预训练/微调与上下文示例学习对照，并用短输入输出示例呈现 ICL | 独立证明其技术主张或适用于所有读者的教学效果 |
| [How does in-context learning work?](https://ai.stanford.edu/blog/understanding-incontext/)（Xie、Min，Stanford AI Lab，2022） | 研究者博客；定义与 Bayesian framework 部分 | ICL 作为固定参数的上下文条件化行为；一种可能的解释框架 | 已解决的通用机制或跨任务定律 |

## 章节设计边界

- 第十六章承担 decoder-only 作为通用生成接口的结构理由；第十七章承担 GPT-3 等指定版本的 block 配置。
- 第十二章已详细解释 prompt 示例、ICL 与 fine-tuning、示例信息、顺序/位置和机制假说；第五十一章只保留其在 GPT 历史中的里程碑位置。
- 第五十二章承担 Chinchilla 与 compute-optimal 的参数/token 预算研究；第五十三章承担 SFT、奖励建模和 RLHF。第五十一章不把现代 ChatGPT 产品表现反推为基础 GPT 结构的结果。

## 未解决边界

GPT 论文对模型和评测的公开范围有限，跨版本能力变化无法仅凭谱系名称拆分为架构、规模、数据或训练配方的单一因果贡献。GPT-3 的 contamination 处理也不能证明所有结果已排除数据重叠；引用具体分数时必须回到论文的任务与清洁子集分析。
