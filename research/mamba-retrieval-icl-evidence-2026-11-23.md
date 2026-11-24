# Mamba 精确检索与上下文学习证据研究记录

检索日期：2026-11-23。

## 研究问题与范围

面向已学习 Mamba 选择性状态与 Selective Scan 的读者，核实三个彼此相关但不可混为一谈的问题：历史信息压入递归状态后，哪些精确检索任务会变难；Mamba 能否从上下文示例学习规则；现有结果对模型规模、训练预算、任务构造和具体版本有哪些依赖。研究对象主要是 Mamba/Mamba-2 语言模型及可比较的状态空间或混合模型，不把一般 RNN 理论自动视作 Mamba 的实证结论。

## 检索方式

平台包括 arXiv、PMLR、ACL Anthology、ICLR Proceedings、OpenReview、作者/实验室页面及《Dive into Deep Learning》。检索词覆盖 Mamba exact retrieval、associative recall、MQAR、copying、Phonebook、in-context learning、Markov chains、state capacity、retrieval gating、Mamba-3，以及对应中文关键词。先以论文/会议论文核对定义、任务、实验和结果，同时检索研究者与课程讲解材料，检查选择性状态、状态读取和任务拆分的讲解路径；未以榜单、新闻或搜索摘要作为结论证据。

## 主要来源与证据边界

1. Gu 与 Dao，[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2)，arXiv v2（2024-05-31）。论文将选择性机制用于离散序列中的内容依赖保留与遗忘，并以 selective copying、induction heads 等合成任务说明设计动机。它支持输入依赖更新改善既有 SSM 内容选择能力；这些合成任务不证明任意历史 token 都能被精确寻址。
2. Jelassi 等，[Repeat After Me: Transformers are Better than State Space Models at Copying](https://proceedings.mlr.press/v235/jelassi24a.html)，ICML/PMLR 2024。论文在广义状态空间模型（GSSM）的固定大小 latent state 假设下分析字符串复制，并报告合成复制及预训练模型的复制/上下文检索比较。理论区分依赖模型类别、状态表示和精度假设；不能简化成所有实际 Mamba checkpoint 在任意任务中必败的无条件结论。作者也说明表达能力结果不等于模型一定能学到该表示，足够大的 state 可能容纳输入。
3. Grazzi 等，[Is Mamba Capable of In-Context Learning?](https://proceedings.mlr.press/v256/grazzi24a.html)，AutoML/PMLR 2024。作者在函数近似与自然语言处理任务中报告 Mamba 和 Transformer 的 ICL 表现接近。结果只支持其模型、任务与 prompt 设置，不覆盖所有关联召回、多证据推理或任意长度。
4. Park 等，[Can Mamba Learn How to Learn? A Comparative Study on In-Context Learning Tasks](https://arxiv.org/abs/2402.04248)，arXiv 预印本（2024-02-06）。从任务分布采样示例并训练模型，研究报告标准回归 ICL 中两类架构表现相近，Mamba 在 sparse parity 上成功而 Transformer 未能解决，Mamba 在 decision tree 和 vector-valued MQAR 等非标准检索任务上失败，混合变体可以补足部分任务。不同任务使用特定训练分布；这是任务切片证据，且本文引用的版本为预印本。
5. You 等，[Revealing and Mitigating the Local Pattern Shortcuts of Mamba](https://aclanthology.org/2025.findings-acl.629/)，Findings of ACL 2025。作者通过合成 MQAR 控制 key-value 数量、位置/分散程度和测试长度，报告单条局部 key-value 在其测试中可以延伸到 60K context；多条分散或局部密集信息的抽取更易下降，并观察到位置与 n-gram pattern shortcut。该结论属于论文训练数据、tokenizer、合成模板和模型设置，不能直接等同于自然文档问答表现。
6. Waleffe 等，[An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)，arXiv 研究论文（2024）。作者比较约 8B 的 Mamba、Mamba-2、Transformer 和 Mamba-2-Hybrid；数据训练预算包括 1.1T 与 3.5T tokens。论文说明 3.5T 对照包含 Mamba-2 与 Transformer，并未训练 3.5T 的纯 Mamba；纯 SSM 在多项零样本任务可匹配或超过 Transformer，但在 5-shot MMLU、Phonebook copying/lookup 和部分长上下文推理任务有差距。Hybrid 的结果不能记到纯 SSM 名下；长上下文 prompt、continued pretraining 和 LongBench 格式也会影响比较。
7. Bondaschi 等，[From Markov to Laplace: How Mamba In-Context Learns Markov Chains](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8fef8a350b54e18758ea546bc984ba28-Abstract-Conference.html)，ICLR 2026。论文在随机 Markov 链任务中分析 Mamba 的上下文预测，给出单层 Mamba 可学习 Bayes/minimax 最优 Laplacian smoothing estimator 的理论与实验结果。它证明特定序列预测任务的可学习性，不代表任意 key-value 记忆或通用长文档推理。
8. Wang 等，[Understanding In-Context Learning Beyond Transformers: An Investigation of State Space and Hybrid Architectures](https://aclanthology.org/2026.findings-acl.39/)，Findings of ACL 2026。论文针对两类 knowledge-based ICL 任务结合行为评估和干预分析；作者报告架构间行为表现可以相似而内部机制不同，并区分 parametric knowledge retrieval 与 contextual knowledge understanding。对本章有用的边界是：行为总分和内部机制不是同一个主张；结论仍绑定论文所测模型与任务。
9. Arda、Cayci 与 Eryilmaz，[A Comparative Analysis of Attention versus State-Space Models for In-Context Learning](https://arxiv.org/abs/2609.32341)，arXiv 预印本（2026-09-26）。论文从广义线性回归 ICL 形式区分 evidence assembly、belief maintenance 和 content addressing；在其正式化的 content-routing 类中推导部分 softmax attention 与 selective SSM 的宽度差异，并用 LLaMA-type Transformer/Mamba-2 验证部分结论。该模型类和任务假设是理论边界，不能读成对所有真实语言模型的普遍排行榜。
10. Yarden、Ben-Artzy 与 Schwartz，[Beyond State Capacity: Why Retrieval Breaks in Mamba](https://arxiv.org/abs/2610.19709)，arXiv 预印本（2026-10-14）。作者在 query-first Phonebook 类任务中比较仅 query 与增加局部 target 标记的输入；局部相关性标记显著提高所测 NVIDIA-Mamba/Codestral-Mamba 的准确率，门控 patching 也能大幅改变结果。论文据此将一部分失败关联到 relevance signal 形成和进入 selection gate 的时机，而不是只归因于 state 容量。该工作仍为预印本，任务是受控合成设置，不能据此给所有 Mamba 变体下因果定论。
11. Lahoti 等，[Mamba-3: Improved Sequence Modeling using State Space Principles](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)，ICLR 2026。作者报告新递归、复值状态和 MIMO 设计改善其 retrieval、state-tracking 与下游评测；1.5B 比较绑定所测模型和基线。Princeton/作者团队的[研究说明](https://pli.princeton.edu/blog/2026/mamba-3-improved-sequence-modeling-using-state-space-principles)进一步指出纯 Transformer 在该组 retrieval 测试仍有优势，且 NIAH、半结构化与真实 retrieval 可能对细节改动有不同反应。它说明模型版本和任务定义会改变结果，不应被解读成对此前结果的反证或对整个架构家族的胜出证明。
12. 教学材料：《Dive into Deep Learning》[Selective State Space Models](https://d2l.smola.org/chapter_recurrent-modern/mamba.html) 用 selective-copying 任务追踪输入依赖的 $Δ_t$、写入/读出状态及局部卷积；适合作为“选择性写入不等于任意 query 直接重读历史”的组织参照。它不是新的跨架构性能证据。

## 综合判断

ICL 包含规则/函数学习、上下文预测、格式适配与关联召回等不同工作负载。Mamba 在某些函数学习和 Markov 序列任务上能表现出 ICL；在复制、多条目精确检索、Phonebook 或部分多证据任务上的差距则取决于任务、训练和模型版本。固定 state 是历史访问路径的结构差异，不能单独证明任务必败；gate 接收到的相关性线索、信息分布、训练长度、答案格式和模型深度都会影响观测结果。

规模比较还要把纯 SSM 与 hybrid 分开，核对训练 token、数据、上下文 curriculum、prompt、输出格式、实现和版本。论文结果不替代本地复现；本研究未运行模型、构建 benchmark 或验证代码。

## 待核边界

不同来源采用的 MQAR、Phonebook、copying、Markov prediction 与长文档问答不是可互换指标。未来采用新论文或模型时，应重新记录具体 checkpoint/revision、任务协议、训练预算与对照实现。2026-09 和 2026-10 arXiv 预印本的结论仍按预印本证据等级处理。
