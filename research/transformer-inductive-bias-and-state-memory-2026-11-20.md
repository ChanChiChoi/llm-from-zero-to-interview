# Transformer 归纳偏置、顺序与状态记忆：来源证据

检索日期：2026-11-20

## 研究问题与范围

- **问题范围：** 为第二十一册第十四章核验序列架构如何编码顺序和局部性、显式 token 历史与递归状态的差别，以及 recall 对状态表示的任务条件。重点区分架构约束、可学习行为和具体实现，不推导通用架构排名。
- **读者与前置：** 已了解基本 Transformer、self-attention 和 causal mask；本章唤回必需对象，详细位置编码、mask、KV cache 和 SSM 数学分别回链本册第 7、8、11、29—33 章。
- **检索范围：** 2017 年 Transformer 原始论文、D2L 的位置/卷积/RNN 教材章节、S4/Mamba/Zoology/BASED 原始论文，以及作者和研究者的解释性博客。检索词包括 “Transformer permutation equivariance positional information”, “convolution locality translation equivariance”, “RNN hidden state sequence memory”, “S4 recurrent convolutional representation”, “Mamba input-dependent selective scan”, “Zoology multi-query associative recall”, “BASED MQAR state size lower bound” 和对应中文词组。
- **筛选边界：** 课程、教材和博客用于学习解释路径；技术定义及实验数字回到原始论文。未开展系统综述，也不以单篇教材的编排证明对所有读者都有效。

## 序列架构所编码的结构先验

- Vaswani et al., *Attention Is All You Need*, arXiv v7，[原论文](https://arxiv.org/abs/1706.03762)。阅读摘要及模型机制部分。论文说明 Transformer 以 attention 为基础，移除 recurrence 与 convolution，并加入 positional encodings，使序列位置信息进入表示。该论文实验和性能结论限于所报告的翻译任务、数据与系统设置；本章只借其架构动机，不外推当代架构的统一速度或质量结论。
- Zhang et al., *Dive into Deep Learning*, “Positional Information”，在线教材章节，[原文](https://d2l.smola.org/chapter_attention/positional-information.html)，2026-11-20 阅读。材料给出在 dropout 关闭、self-attention 不含位置输入/位置偏置或 mask 时的置换等变性质，并说明位置机制用于提供顺序/位置信息。此性质是限定结构条件下的数学结论；加入 causal mask 后该对称性已被打破，不能据此说 decoder-only Transformer 没有顺序信息。
- Zhang et al., *Dive into Deep Learning*, “From Fully Connected Layers to Convolutions”，在线教材章节，[原文](https://d2l.smola.org/chapter_convolutional-neural-networks/why-conv.html)，2026-11-20 阅读。材料从局部感受野和各位置共享同一过滤器解释 convolution 的 locality 与 shift equivariance。其边界条件包括规则网格和边界处理；“等变”表示输入平移时特征图相应平移，不等于分类输出保持不变。
- Zhang et al. et al., *Dive into Deep Learning*, “Recurrent Neural Networks”，在线教材章节，[原文](https://d2l.smola.org/chapter_recurrent-neural-networks/rnn.html)，2026-11-20 阅读。材料将 RNN hidden state 写成由当前输入和前一状态递归更新的固定形状表示，并解释有限精度状态对历史信息的实际瓶颈。该教学说明不表示所有状态模型采用同一种更新/读出，也不意味着每项历史任务都无法由递归状态完成。

## SSM 的状态与训练/推理表示

- Gu, Goel, and Ré, *Efficiently Modeling Long Sequences with Structured State Spaces*, arXiv v3，[论文](https://arxiv.org/abs/2111.00396)。其方法建立在状态空间模型的结构化参数与长序列建模上；本章仅把它作为“状态更新可以有不同计算表示”的入口，不复述 S4 推导。
- Rush and Karamcheti, *The Annotated S4*, v3，[讲解](https://srush.github.io/annotated-s4/)。讲解按离散递归表示、卷积训练表示和实践层组织 S4，说明其特定 S4 实现可在训练时使用卷积视角、推理时使用递归视角。它是作者明确标注的教学解释与实现示例，不代表所有 SSM 或任意硬件上的统一路径。
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv v1，[论文版本入口](https://arxiv.org/abs/2312.00752v1)。§§2—3 说明时间不变 SSM 可在递归和卷积形式间转换；Mamba 令部分 SSM 参数依赖输入以支持选择性信息传递，因而不再等价于固定卷积，并用并行 associative scan 及硬件感知算法计算。该机制、长序列实验和性能只支持论文实现与其所测任务，详细数学和硬件设计由本册第 29、31、33 章主讲。

## Recall 与状态规模的任务证据

- Arora et al., *Zoology: Measuring and Improving Recall in Efficient Language Models*, arXiv v1，[论文](https://arxiv.org/abs/2312.04927v1)。§§3—5 及实验附录：作者训练 17 个语言模型，覆盖 5 种架构和 70M—1.4B 参数规模；在 Pile 的 repeated n-gram associative-recall 切片中，按作者启发式划分的 AR 命中约占测试 token 的 6.4%，并解释 gated-convolution 与 attention 的平均 perplexity gap 的 82%。作者指出单次 query 的合成任务与自然语言中多次、不同位置 recall 不同，提出 MQAR 分析。数字仅限作者的模型、训练和测试设置；该切片归因不是自然语言整体能力的普遍因果分解。
- Arora et al., *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff*, arXiv v1，[论文](https://arxiv.org/abs/2402.18668v1)。§3、Appendix F、实验设置：作者在 MQAR 中改变 6 种 sequence mixer 的状态相关超参数，训练长度 256，测试长度 1,024，并报告状态字节数与 recall accuracy 的关系；理论部分对形式化 exact associative recall 给出下界。Theorem 3.1 的结论依赖输入为任意 MQAR/AR key-value 序列、模型因果递归和状态元素有界等假设；它不是关于所有语言建模任务、近似答案或可访问外部存储的普遍定理。各模型的“state size”按各自实现具体计算，不能跨研究直接当成同一类内存占用。
- Eyuboglu, Arora, and Zhang, “Zoology (Blogpost 1): Measuring and Improving Recall in Efficient Language Models,” Hazy Research，[作者团队讲解](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)，2023-12-11。用于理解 MQAR 的查询/键值/距离差异及教学解释路线；其中方法事实和数字仍以同名论文及其评测协议为准。

## 教学组织观察

- 上述 D2L 内容把位置等变性、卷积局部性、递归状态分别组织为可独立复述的机制，再通过模型比较连接它们；《The Annotated S4》沿递归视角到卷积视角讲同一状态空间方法。可借此把本章按“序列结构先验 → 顺序如何进入计算 → 历史怎样写入和读取”组织。该观察只说明公开教材/讲解采用了这些组织路径，不证明其普遍学习效果；更具体的架构数学移交相邻章节。
