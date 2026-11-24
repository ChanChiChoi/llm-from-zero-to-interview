# 块稀疏 Attention 布局与动态路由研究记录

研究日期：2026-11-20。范围为第二十一册第二十六章的职责、结构与正文事实核验。读者已理解 causal mask、Q/K/V attention 与 GPU tile 的基本作用。本章要回答：token 级可见关系怎样变成硬件能够跳过的计算块；块大小、动态选择器和稀疏 kernel 各自怎样改变计算与误差；理论稀疏能否转成目标执行路径上的实际收益。

## 检索范围与组织材料

- 使用英文检索词 `block sparse attention layout`, `block mask every element sparse`, `dynamic block selection attention routing overhead`, `sparse attention selected set softmax`, `adaptive block size attention`, `hierarchical top-k block selection`；查阅 arXiv、ACL Anthology、PyTorch、NVIDIA 文档及课程讲义。
- 对照第十九章现行大纲、正文和审计：固定窗口、dilation、global/random pattern 与 query-conditioned selection 的信息图由第十九章讲；第二十六章只解释其中的块粒度表示、执行路径与动态选块成本。
- 教学组织材料：[Stanford MS&E 319, Lecture 5: KV Caching, Sparse Attention, and Compression](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf)，2026 年秋季讲义，阅读 §2.2—2.5。其讲解把固定窗口、indexer 打分/Top-K 和主 attention 分开，并单列选择成本与漏掉概率质量的误差；讲义的 DSA 公式和 decode 情境不代表所有块选择方法。
- pattern 图示材料：[CUHK CSC6203 Lecture 3, Transformer and Beyond](https://nlp-course-cuhksz.github.io/materials/2026_fall_slides/Lecture04_Transformer_and_Beyond.pdf)，2026 年秋季讲义，第 111—112 页。其 full、sliding、dilated、global、local/random/global 图示用于检查模式图的读法；这些经典模式的解释已由第十九、二十七章承担，不在本章复讲。

## 研究发现

### Token mask 与执行块

- PyTorch `flex_attention` 当前文档的 `BlockMask` 小节（2026-11-20 查阅）将 block-sparse 表示定义为：一个 query/key tile 只有在其中所有 token 元素都被 mask 时才作为稀疏块跳过；块布局贴合连续加载和计算，优先服务实现简化与 kernel 效率，不保证压缩最小的索引存储。该描述支持“逻辑边稀疏不等于执行块稀疏”，但具体格式、反向元数据与性能数字只适用于所查 PyTorch 文档版本。
- NVIDIA [cuDNN Block Sparse Attention](https://docs.nvidia.com/cudnn/operations/sparse_attention/bsa) 文档（Latest 页面，2026-11-20 查阅）以每个 query block 的 KV block 列表定义操作，并说明 softmax 在选中块内的有效 token 上统一归一化。该 API 当时标注为实验性且面向非因果 scaled dot-product attention；只作为块列表与归一化的实现例子，不能据此声称所有稀疏 kernel 都支持相同 mask、布局或后端。

### 路由阶段与稀疏 attention 阶段

- Jiang 等，[MInference 1.0](https://arxiv.org/abs/2407.02490)，NeurIPS 2024；查阅 arXiv HTML 的 pattern 分析、§3.1—3.2、实验条件及 Appendix C.4。方法先离线给 head 选择 pattern，再针对输入在线生成稀疏索引，最后调用对应稀疏 kernel；block-sparse heads 使用块级 Q/K 聚合估计重要区域。论文报告的速度和质量只绑定所测模型、A100、dtype、上下文、任务和基线；“最高 10×”不是通用加速保证。
- Yuan 等，[Native Sparse Attention](https://aclanthology.org/2025.acl-long.1126/)，ACL 2025；查阅作者 arXiv v2 §3.2—3.4 与 ACL 元数据。NSA 将层级 token 压缩、选择和局部路径纳入可训练稀疏 attention，并设计与硬件执行配合的实现，说明“加载一个现成模型后做推理时选块”与“训练时学习稀疏路径”是不同方法契约。速度和任务结论限于论文模型及设备配置。
- Tang 等，[Block Sparse Attention with Log-Linear Complexity](https://arxiv.org/abs/2609.31093v1)，arXiv v1，2026-09-25；查阅摘要和 §1。论文指出，若每个 query 对全部 key blocks 打分，所选注意力虽只读取 Top-K blocks，选择阶段仍可能随序列长度二次增长（其推导含 block size 因子）；作者提出 coarse-to-fine pyramid Top-K 方案并报告 $O(N\log N)$ 选择复杂度。该项是 2026 年预印本的方法与作者主张，不能扩写成所有选择器的复杂度或已确立的普遍性能结论。

### 块粒度与异质性

- Liu 等，[AB-Sparse](https://arxiv.org/abs/2605.12110v1)，arXiv v1，2026-05-12；查阅摘要与引言。作者报告不同 attention head 对块粒度的敏感度并不相同，提出按 head 分配块大小并配合自定义 kernel。此结果支持把 block size 看成算法、工作负载和 kernel 的共同参数；“自适应一定更准/更快”并非跨模型和硬件的结论。
- 块粒度带来两种同时存在的代价：块变大可减少索引数量并形成更规整的矩阵乘，但会把块内原本不需要的 pair 一起执行；块变小可让所选 token 更贴近逻辑 mask，却增加索引量、调度和不规则访问。尾部不完整块、causal 对角块和块内 token mask 会影响实际执行，不能只看 mask 的零元素比例。

## 证据与结论边界

- 在选定候选集合上计算标准 score、softmax 与 value 加权，只能称为“对该集合内的 attention 精确”；若候选集合排除了 dense attention 中有非零权重的 key，输出及分母都已改变，不能称为与 dense attention 等价。
- 动态稀疏的总开销包括候选估计、索引/排序、选中块读取和主 attention。减少主 attention FLOPs 不足以证明端到端收益；每种选择器的扫描范围和复杂度必须分别核对。
- 训练/服务一致性涉及 causal 与 padding 规则、变长尾块、query/key block 映射、head/batch 索引及 router 版本。模型名、Top-K 字段或块稀疏 API 名称不能替代对实际 mask 和执行路径的确认。
- 渐近复杂度、attention 权重图、选中块比例、kernel latency 与端到端系统结果是不同证据；每项数字都要保留模型、硬件、精度、长度、负载和基线。

