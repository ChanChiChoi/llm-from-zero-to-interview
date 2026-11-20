# Transformer 并行性、表达力与 Scaling：来源证据

检索日期：2026-11-19

## 研究范围与检索记录

- **问题范围：** 核验 Transformer 的序列位置并行性、计算复杂度与路径长度；理论表达力定理和形式语言限制；语言模型 loss 的 scaling law、固定训练预算分配、数据质量变化和下游能力“涌现”的证据边界。
- **时间范围：** 2017—2026 年的基础论文和后续研究；侧重原始论文、同行评审论文及会议正式摘要。
- **平台：** arXiv、OpenReview、ACL Anthology、NeurIPS Proceedings、ICLR Proceedings、Stanford NLP 课程页、NVIDIA 官方性能指南、Google Cloud TPU 文档。
- **检索词：** “Transformer sequential operations maximum path length Table 1”; “Yun Transformers universal approximator sequence-to-sequence positional encoding”; “Hahn self-attention PARITY 2DYCK soft attention fixed depth”; “Chiang Cholak overcome theoretical limitation self-attention PARITY FIRST”; “Kaplan scaling laws neural language models cross entropy”; “Hoffmann compute optimal Chinchilla 70B 1.4T”; “emergent abilities metrics BIG-Bench Schaeffer”; “PaLM BIG-Bench discontinuous improvements”; “text quality interventions reshape scaling laws ICLR 2026”; “NVIDIA GPU GEMM Tensor Cores matrix multiplication guide”; “Google TPU Matrix Multiplication Unit architecture”; “Stanford CS224N 2021 lecture 9 Transformer”.
- **筛选范围：** 搜索结果仅作发现入口；技术主张优先核对原始论文/会议记录。课程材料用于记录一种公开讲授组织，不用于声称某种编排已被证明提高学习效果。未做系统综述。

## 并行性、复杂度与信息路径

- Vaswani et al., “Attention Is All You Need,” NeurIPS 2017，[arXiv 原文](https://arxiv.org/abs/1706.03762)。
  - **版本与阅读范围：** 2017 原论文，重点阅读 §4.1、Table 1 与相邻的复杂度/路径解释。
  - **来源支持：** 在其标准层比较中，self-attention 的复杂度为 $O(n^2 d)$、序列位置串行操作数为 $O(1)$、最长位置路径为 $O(1)$；recurrent layer 分别为 $O(n d^2)$、$O(n)$、$O(n)$；convolutional layer 分别为 $O(k n d^2)$、$O(1)$、$O(\log_k n)$。作者说明当 $n<d$ 时其 self-attention layer 比 recurrent layer 更快，并讨论卷积路径长度受结构设置影响。
  - **证据等级：** source-backed（原论文对标准层结构的复杂度/路径分析及其指定机器翻译实验）。
  - **适用边界：** 操作数与依赖路径不是现代 kernel 的墙钟性能测量；结果不概括所有卷积、循环、稀疏或状态空间架构，也不意味着因果生成能够并行地产生后续 token。

- Stanford CS224N, 2021 Lecture 9, *Self-Attention and Transformers*，[课程讲义 PDF](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/slides/cs224n-2021-lecture09-transformers.pdf)，[课程首页](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1214/)。
  - **阅读范围：** 讲次材料中从 recurrence 问题、attention、Transformer 组件到实验结果和注意力成本/缺点的顺序。
  - **来源支持：** 该课程按问题、机制、模型结构、实验结果和局限组织这一主题，可作公开课程编排样本。
  - **适用边界：** 单个课程讲义不能证明这种顺序对所有读者最佳，也不能证明它提升学习结果。

- NVIDIA, *Matrix Multiplication Background User's Guide*，[官方性能指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。
  - **阅读范围：** GPU GEMM 实现与 Tensor Core 要求部分。
  - **来源支持：** GPU 通过分 tile 的 GEMM 执行矩阵乘；Tensor Core 是否可用及效率受库版本、数据类型和矩阵维度对齐等条件影响。
  - **适用边界：** 指定厂商库与硬件说明不构成任意 Transformer workload 的吞吐保证。

- Google Cloud, *TPU architecture*，[官方系统架构文档](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)。
  - **阅读范围：** TPU TensorCore 与 Matrix Multiplication Unit (MXU) 说明。
  - **来源支持：** TPU 使用 MXU 执行矩阵乘加；不同代际的单元形状与能力不同。
  - **适用边界：** 架构文档说明硬件单元，不证明任何特定模型、shape、精度和编译器配置上的模型训练性能。

## 理论表达力与形式语言边界

- Yun et al., “Are Transformers Universal Approximators of Sequence-to-Sequence Functions?”, ICLR 2020，[OpenReview 论文](https://openreview.net/forum?id=ByxRM0Ntvr)。
  - **版本与阅读范围：** ICLR 2020 正式论文，摘要、Theorems 2–3、§4.4 和讨论。
  - **来源支持：** 无位置编码时，论文证明 Transformer 可逼近连续、置换等变、紧支撑的序列到序列函数；增加可训练的位置编码后，可在紧致域上逼近任意连续序列到序列函数。证明区分 self-attention 的 contextual mapping 与 FFN 的 value mapping；论文讨论最坏情况构造需要随问题规模大幅增长的参数/层数。
  - **证据等级：** source-backed（数学存在性定理）。
  - **适用边界：** 不保证优化可找到构造、所需样本和计算可行、某一 checkpoint 已学得目标函数，或模型可外推到未覆盖的长度/分布。定理中的函数域、位置编码和序列定义不可省略。

- Hahn, “Theoretical Limitations of Self-Attention in Neural Sequence Models,” TACL 2020，[ACL Anthology 正式论文页](https://aclanthology.org/2020.tacl-1.11/)，[原论文 PDF](https://nlp.stanford.edu/pubs/hahn2020theoretical.pdf)。
  - **版本与阅读范围：** TACL 第 8 卷正式论文，摘要、§§3–4、Theorem 1、Lemma 5、Theorem 6 和讨论。
  - **来源支持：** 对 hard attention，论文证明其分析的 Transformer 不能识别 PARITY 与 2DYCK。对 soft attention，论文不给相同强度的准确率界，而证明在固定深度/参数条件下，PARITY 下一符号交叉熵趋向 unigram chance level，2DYCK 的交叉熵与最优值仍有正差距。作者说明结果渐近依赖输入长度，且实践系统可通过增加层数、heads 或参数规模等避开分析范围。
  - **证据等级：** source-backed（指定形式语言、模型形式和渐近设定下的定理）。
  - **适用边界：** 不能把形式语言和无限输入长度结果直接当作自然语言能力结论；hard-attention 的识别结果与 soft-attention 的交叉熵结果也不能混写成同一个主张。

- Chiang and Cholak, “Overcoming a Theoretical Limitation of Self-Attention,” ACL 2022，[ACL Anthology 正式论文页](https://aclanthology.org/2022.acl-long.527/)。
  - **版本与阅读范围：** ACL 2022 正式论文，摘要及关于 PARITY、FIRST、LayerNorm 和长度泛化的实验/构造说明。
  - **来源支持：** 作者给出能正确识别 PARITY 和 FIRST 的 Transformer 构造；指出构造的交叉熵可随长度趋近随机基线，并研究用 LayerNorm 降低交叉熵；训练模型对 FIRST 从较短长度到较长长度的泛化仍会失败，论文给出针对性改动。
  - **证据等级：** source-backed（作者的形式构造和论文实验）。
  - **适用边界：** 识别规则、预测置信度/交叉熵、训练可学性和长度泛化是不同结果；针对这些任务和构造的反例不构成通用架构保证。

- **跨来源综合（推断）：** Yun 的定理描述紧致域上的函数逼近存在性；Hahn 分析固定模型面对无限增长形式序列；Chiang 与 Cholak 的结果进一步区分识别、置信度和训练泛化。它们关注的输入域、模型约束和指标不同，因此不应简化成“Transformer 可表示任何东西”或“Transformer 理论上不能处理递归结构”两句口号。

## 经验 Scaling 与 compute-optimal 训练

- Kaplan et al., “Scaling Laws for Neural Language Models,” arXiv:2001.08361，[原论文](https://arxiv.org/abs/2001.08361)。
  - **版本与阅读范围：** 2020 论文，摘要、§1 规律总结、§1.3 参数定义、§2.1 参数/FLOPs 估算和 §3 实验范围。
  - **来源支持：** 以 held-out cross-entropy 拟合非 embedding 参数量、训练 token 数和计算量的幂律关系。论文主要实验覆盖约 768—1.5B 非 embedding 参数、约 2,200 万—230 亿 token，并讨论规律受模型、数据、训练等其他变量限制。
  - **证据等级：** source-backed（指定 GPT 类模型、数据、训练流程和规模内的经验拟合）。
  - **适用边界：** 幂指数和拟合系数不构成跨架构、数据分布或下游能力的自然常数；loss scaling 不直接预测特定 benchmark、ICL、安全或长上下文能力。

- Hoffmann et al., “Training Compute-Optimal Large Language Models,” NeurIPS 2022，[arXiv 原文](https://arxiv.org/abs/2203.15556)，[NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf)。
  - **版本与阅读范围：** NeurIPS 2022 正式论文，摘要及 compute-optimal allocation、Chinchilla 70B 与 Gopher 对照部分。
  - **来源支持：** 作者训练 400 多个模型（约 70M 至 16B 参数、约 5B 至 500B token），拟合固定训练计算预算下的参数/token 分配。论文建议在其设定中将参数和训练 token 一起增长，并以 70B 参数、1.4T token 的 Chinchilla 与相同训练计算预算下的 280B Gopher 对照验证。
  - **证据等级：** source-backed（该模型族、数据与预训练计算目标下的实验和外推）。
  - **适用边界：** 参数和数据增长比例不是跨数据质量、语料重复、模型族、训练目标和 serving 负载的通用常数；“最优”指论文设定的训练目标，不自动等于生命周期/部署成本最优。

## 数据质量对 scaling 拟合的影响

- Ardalani et al., “How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study,” ICLR 2026，[ICLR 正式论文摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)。
  - **阅读范围：** ICLR 正式论文摘要；未在此条目复核全文实验协议。
  - **来源支持：** 摘要报告使用 23 组 QualityPajama 数据，训练 100M—8B 参数、100M—200B token 的 2,000 多个模型；比较去重、启发式过滤和 LLM 引导改写，并报告干预会影响 scaling 拟合系数和指数，数据排序及 token/parameter 的计算最优比例依资源/干预改变。
  - **证据等级：** source-backed（正式会议摘要中的作者报告）。
  - **适用边界：** 结果限于 QualityPajama、所用干预、模型、预算和评估；不定义跨语料通用的数据质量标量或固定比例。

## 下游能力与 Emergent Abilities

- Wei et al., “Emergent Abilities of Large Language Models,” arXiv:2206.07682，[原论文](https://arxiv.org/abs/2206.07682)。
  - **版本与阅读范围：** 2022 论文摘要与定义。
  - **来源支持：** 作者将 emergent ability 描述为在小规模模型中不存在、在较大模型中出现，且不能由小规模表现的简单外推预测的能力。
  - **适用边界：** 该定义依赖任务、指标和“出现/不存在”的操作化；它不是所有规模增长都必然产生跃迁的机制证明。

- PaLM authors, “PaLM: Scaling Language Modeling with Pathways,” arXiv:2204.02311，[原论文](https://arxiv.org/abs/2204.02311)。
  - **版本与阅读范围：** 2022 技术报告 v5，模型规模与 BIG-Bench 结果讨论。
  - **来源支持：** 作者报告其 8B、62B、540B 模型系列在约四分之一 BIG-Bench 任务上出现不连续的规模相关提升。
  - **适用边界：** 这是三个规模点、特定 benchmark 和评分口径中的作者报告；不能归因给架构单一因素，也不能推广成一般能力曲线。

- Schaeffer et al., “Are Emergent Abilities of Large Language Models a Mirage?”, NeurIPS 2023，[正式论文](https://papers.neurips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Paper-Conference.pdf)。
  - **版本与阅读范围：** NeurIPS 2023 正式论文，摘要、指标选择解释、GPT-3/InstructGPT 与 BIG-Bench 分析。
  - **来源支持：** 作者展示非线性/离散评估指标和有限统计分辨率可使渐进能力看起来像阈值式跃迁；他们在指定模型、任务和若干视觉任务中比较指标选择。
  - **证据等级：** source-backed（论文报告的方法分析和评估结果）。
  - **适用边界：** 不证明所有任务能力都连续，也不否定所有 emergence 定义；结论依所研究任务、模型系列、指标和样本规模。

- **跨来源综合（推断）：** “涌现”争论应同时报告所选能力定义、模型系列和规模点、任务样本量、prompt/评测器、连续或离散指标及其不确定性。单一模型报告支持其所测结果；跨模型或任务的普遍跃迁结论需要更多同口径证据。
