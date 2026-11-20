# In-Context Learning 与上下文内检索：来源证据

检索与阅读日期：2026-11-20。研究范围：固定参数下的 zero/few-shot 定义、示例携带的信息、ICL 的机制假说、示例顺序/位置敏感性、上下文内 token 使用与外部 RAG 的边界，以及适合初学者的讲解材料。下列研究结果均绑定各自模型、任务和输入设置；本文不记录章节编辑决定。

## 检索范围

- **平台：** arXiv 与 ACL Anthology 原始论文、PMLR 正式会议论文、Transformer Circuits 作者研究文章、Stanford CS224U/CME295 课程材料；并用网页搜索发现后续争议与教学入口。
- **检索词：** “Language Models are Few-Shot Learners”; “Rethinking the Role of Demonstrations”; “In-context Learning and Induction Heads”; “What learning algorithm is in-context learning linear models”; “Do pretrained Transformers Learn In-Context by Gradient Descent”; “Fantastically Ordered Prompts”; “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”; “Stanford in-context learning lecture notes”; “in-context learning prompt order sensitivity”; “Attention is not Explanation”; “Attention is not not Explanation”.
- **时间范围：** 重点追溯 2020—2024 年定义、机制和受控实验；教学材料按课程学年标注。长上下文位置结果复用 `long-context-evaluation-and-effective-capability-2026-11-03.md` 中对 Lost in the Middle 的证据，不在本记录重复。

## 固定参数与示例的作用

- **Brown et al., “Language Models are Few-Shot Learners,” arXiv:2005.14165v4，2020-07-22。** [论文 HTML](https://arxiv.org/html/2005.14165)。阅读摘要、引言、方法与结果概述、§5 Limitations 和 benchmark contamination 分析。作者在 GPT-3 的 zero/one/few-shot 评估中不做梯度更新或 fine-tuning，而以文本指定任务并提供示例；论文同时报告任务间强弱不一，并讨论网页训练数据与 benchmark 重叠问题。证据支持“固定参数下通过输入条件执行任务”及该论文的任务结果，不识别 ICL 的内部机制，也不证明任意任务都能适配或任务对预训练真正全新。
- **Min et al., “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?”, arXiv:2202.12837v2，2022-10-20；EMNLP 2022。** [论文 HTML](https://arxiv.org/html/2202.12837)。阅读摘要、§§3—6 与任务设置/结果。作者比较 6 个 decoder-only LM 的 direct/channel 两种推理方式，共 12 个模型/推理设置；数据含 26 个分类和多选数据集，主要设置使用 16 个 demonstration。论文报告，在所测的一组分类和多选任务中随机替换 demonstration 标签对表现影响较小，并指出输入文本分布、标签空间和序列格式仍能传递信号。该结果不表示正确的输入—输出映射在所有任务中都不重要；实验范围不覆盖任意新映射或所有生成任务。
- **Potts, Stanford CS224U In-Context Learning handout, 2023。** [课程讲义 PDF](https://web.stanford.edu/class/cs224u/slides/cs224u-incontextlearning-2023-handout.pdf)。阅读第 5—8 页以及相关 techniques 页面。讲义将 ICL 讲为冻结语言模型仅以 prompt 条件执行任务，并区分指令、示例与 zero/few-shot；其严格 few-shot 定义还要求目标行为未在训练中出现，同时明确指出外部读者通常无法验证这一点。讲义支持定义和教学顺序，不支持普遍性能或学习效果结论。
- **Stanford CME295 Fall 2025 Lecture 3。** [课程 slides PDF](https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf)，阅读 PDF 文本第 467—505 行对应 slides。课程用零样本/少样本 prompt 对照，说明加入示例增加计算量、成本和延迟。该材料用于讲解路线，不作为跨模型效果证据。

## 机制解释及争议

### 注意力权重与解释性边界

- **Jain & Wallace, “Attention is not Explanation,” NAACL-HLT 2019。** [ACL Anthology 正式论文页](https://aclanthology.org/N19-1357/)。核对正式出版信息与摘要。作者报告在其覆盖的多项 NLP 任务中，attention 权重与基于梯度的特征重要性经常不相关，并可找到产生等价预测的显著不同权重分布。摘要足以支持“权重分布不应未经检验就当作输入重要性的充分解释”；具体模型/任务范围有限，不支持对所有 Transformer attention 电路作通用结论。
- **Wiegreffe & Pinter, “Attention is not not Explanation,” EMNLP-IJCNLP 2019。** [ACL Anthology 正式论文页](https://aclanthology.org/D19-1002/)。核对正式出版信息与摘要。作者指出 Jain 与 Wallace 的判断依赖 explanation 的定义及评估是否考虑模型其他部分，并提出多种替代测试；论文摘要将讨论对象主要放在 RNN NLP 模型。该工作支持保留方法论争议、按解释目标和测试条件判断，不证明 attention 图在所有设置中都可靠。

两篇工作共同支持有限结论：attention 权重和可视化本身不能独立证明模型输出忠实依赖目标证据；若要主张解释性，需要针对定义与任务采用相称测试。它们不是 Transformer 中任意 attention head 的因果机制研究，也不证明 attention 权重毫无解释价值。

### ICL 的几类机制证据

- **Olsson et al., “In-context Learning and Induction Heads,” Transformer Circuits, 2022-03-08。** [作者研究文章](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)。阅读 Key Concepts、Arguments Summary、Discussion 与相关模型分析。论文用 `[A][B] … [A] → [B]` 描述 induction-head pattern copying；作者对两层 attention-only 模型给出较具体的 circuit 分析，并将大型模型中 induction head 对一般 ICL 的作用明确写为初步、间接证据支持的假说。不能把 induction head 说成所有 ICL 的唯一或已证实充分机制。
- **Xie et al., “An Explanation of In-context Learning as Implicit Bayesian Inference,” arXiv:2111.02080v6，2022-07-21。** [论文 HTML](https://arxiv.org/html/2111.02080)。阅读摘要、§§2—4、结论及实验设置。论文在预训练分布为 HMM 混合的受控理论设定中证明，模型可从 prompt 推断共享的潜在概念；作者用合成 GINC 数据展示 Transformer 与 LSTM 的 ICL 行为。它提供 latent-task/Bayesian 解释路线，不是对自然语料预训练前沿 LLM 内部算法的普遍证明。
- **Akyürek et al., “What Learning Algorithm Is In-Context Learning? Investigations with Linear Models,” ICLR 2023，arXiv:2211.15661。** [ICLR 论文 PDF](https://webdocs.cs.ualberta.ca/~dale/papers/iclr23d.pdf)，[arXiv 论文](https://arxiv.org/abs/2211.15661)。阅读摘要、线性回归设定、§§3—5 与结论。作者构造了实现 gradient descent/ridge regression 的 Transformer，并在合成线性回归任务上训练 ICL 模型，报告其预测在特定设置下接近 OLS、ridge 或 gradient-descent estimators。对象是受控线性模型任务；不证明自然数据预训练的大模型在推理时真的执行标准反向传播。
- **Shen, Mishra & Khashabi, “Position: Do Pretrained Transformers Learn In-Context by Gradient Descent?”, ICML 2024 / PMLR 235，arXiv:2310.08540。** [PMLR 正式论文页](https://proceedings.mlr.press/v235/shen24d.html)，[论文 HTML](https://arxiv.org/html/2310.08540)。阅读摘要、§§3—5、讨论和 order-sensitivity 实验。作者指出既有等价论证依赖与自然预训练不同的 ICL 训练目标或手工构造权重；在其预训练模型/任务/度量比较中，ICL 与 GD 系列呈现不同的示例顺序敏感性及输出变化。作者结论是等价假说仍开放，需要后续研究；其反证同样不构成所有 ICL 任务永远不可能模拟某种优化算法的证明。

## 顺序、位置与外部检索边界

- **Lu et al., “Fantastically Ordered Prompts and Where to Find Them,” ACL 2022，arXiv:2104.08786v2，2022-03-03。** [论文 HTML](https://arxiv.org/html/2104.08786)，[ACL Anthology](https://aclanthology.org/2022.acl-long.556/)。阅读摘要、§2 的 order sensitivity 分析、§4—5 实验及结论。作者在所测 prompt permutation、分类任务和 GPT 系列设置中观察到示例顺序对结果有显著影响，且较好排列不能稳定迁移到别的模型/任务。该结论支持“顺序是需控制的变量”，不推出总把最新/最相关示例放在末尾就更好。
- **Liu et al., “Lost in the Middle: How Language Models Use Long Contexts,” arXiv:2307.03172v3 / TACL 2024。** [论文](https://arxiv.org/abs/2307.03172)。本章仅复用 `research/long-context-evaluation-and-effective-capability-2026-11-03.md` 中已记录的阅读范围：多文档问答和 synthetic key-value retrieval 中，对所测模型与设置控制证据长度和位置；中段信息在这些任务上可能比开头/末尾更难被利用。该现象不是所有模型或任务的固定曲线，也不能仅凭结果归因于 attention 权重。
- **Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” NeurIPS 2020，arXiv:2005.11401v4，2021-04-12。** [论文 HTML](https://arxiv.org/html/2005.11401)。阅读摘要、§§2—4 和讨论。原始 RAG 由带参数的 seq2seq 生成器与非参数化 dense vector index/神经检索器组合，检索文档再作为生成条件；论文还研究端到端训练的具体形式。它支持“外部文档检索系统包含独立组件，模型再条件化于检索结果”的区分，不意味着所有现代 RAG 产品都复现该论文训练架构，也不意味着 RAG 就等于 ICL。
- **Self-attention 的信息路由公式：** Vaswani et al., “Attention Is All You Need,” [论文](https://arxiv.org/abs/1706.03762)。标准 full attention 在 mask 允许的 query/key 位置间计算相似度并聚合 value；这只说明信息路径，不保证模型会正确定位、绑定或使用目标证据。视觉到答案之间还经过 value 聚合、残差和后续层处理。

## 综合判断与未解决边界

1. ICL 可用操作性定义描述为固定参数下的 prompt 条件化任务行为；“任务对训练真正全新”不是这个定义自动保证的属性。
2. Demonstrations 可同时暴露输入分布、标签集合和序列格式；真实映射是否重要受任务设定影响。Min et al. 的随机标签结果必须留在其分类/多选实验范围内。
3. Induction heads、潜在任务推断/Bayesian inference 和类梯度下降算法分别解释一些可观测行为，但证据来自不同模型和任务，彼此不构成排他总解释。
4. Attention 提供可读上下文位置间的信息通路；是否正确取证、绑定约束并据此作答仍需行为评估。注意力图或某个自拟相似度分数本身不是模型检索正确性的证据。
5. 对架构比较，显式上下文检索可作为一类 workload；单项复制/查找结果不等于完整 ICL、长文档推理或架构总体排名。位置效应和长上下文诊断细节交第 13 章，归纳偏置与状态记忆交第 14 章及替代架构比较章节。
