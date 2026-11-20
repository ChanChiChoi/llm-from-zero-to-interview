# Attention 复杂度、工作内存与在线 softmax

核验日期：2026-11-20。研究范围是标准 scaled dot-product attention 的序列长度成本、显式 score/probability 矩阵的存储条件、FlashAttention 的在线分块计算，以及稀疏/近似路线改变的对象。本文分别记录技术证据和教学解释来源；没有运行 kernel、模型或 benchmark。

## 检索记录

- 检索平台：网页搜索；arXiv 原始论文；Stanford、Washington、Berkeley 公开课程材料；Dao-AILab 官方实现仓库。
- 英文检索词：FlashAttention lecture notes course notes tiling online softmax；FlashAttention-4 algorithm kernel pipelining；Self-attention Does Not Need O(n^2) Memory；Longformer sliding window global attention；Linformer linear complexity；Performer FAVOR+.
- 中文检索词：注意力复杂度、attention 显存、在线 softmax、分块 attention、稀疏注意力、线性注意力。
- 筛选范围：标准 attention 定义与算法论文用于事实和机制；课程讲义用于比较解释顺序、图示和必要推理台阶；仓库主页用于确认当前版本演进入口，不据此推断特定工作负载性能。

## 技术证据

### 标准 scaled dot-product attention

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，2017，arXiv v7 日期 2023-08-02。阅读 §3.2.1、§3.2.3。
- 支持：scaled dot-product attention 先计算 Q 与 K 的点积并除以 $\sqrt{d_k}$，按 key 位置执行 softmax，再用权重汇总 V；因果约束通过 softmax 前的 mask 排除未来位置。论文说明 scaling 的方差动机依赖其写出的独立、零均值、单位方差假设。
- 本文可直接由矩阵形状推出：给定 batch $B$、query heads $H_q$、query 长度 $T_q$、key 长度 $T_k$ 和 head dimension $d_h$，dense score/probability 张量形状为 $[B,H_q,T_q,T_k]$；两个主要矩阵乘的浮点运算量合计约为 $4BH_qT_qT_kd_h$，按 multiply-add 计 2 FLOPs，未计 projection、softmax 和其他逐元素操作。
- 边界：复杂度式是对指定 dense attention core 的 shape 账本，不是整层、训练 step 或 wall-clock 预测。矩阵 shape 推导本身不表示某算法必须把 score 或 probability 矩阵保存到显存。

### 二次计算与非二次额外内存可以并存

- 来源：[Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682)，Rabe 与 Staats，2021。阅读摘要与算法/内存分析。
- 支持：attention 的时间复杂度仍为序列长度平方；论文给出额外内存随序列长度增长低于二次的算法，并报告其加速器实现。它反驳了“标准 attention 数学本身必然需要二次额外显存”的概括。
- 边界：论文中的算法和具体内存数字属于其实现、序列长度和实验设置；不应把作者的渐近空间界直接写成所有框架的实际峰值显存。

### FlashAttention 的 dense exact 路径

- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2)，Dao 等，2022-06-23。阅读 §2.2、§3.1–3.3、§4、§5、附录 B.1–B.4 和附录 C。
- 支持：dense FlashAttention 用 tiling 将 Q/K/V 分块载入 GPU 片上 SRAM；对每行维护在线归一化统计并累积输出，避免将完整 $N\times N$ score/probability 中间矩阵反复写入 HBM。反向传播可重算局部 score/probability，保存逐行统计量而非持久保存完整矩阵。论文的额外 attention memory 随 $N$ 线性增长，主要算术仍为 $O(N^2d)$。
- “Exact”边界：dense 主算法计算与 full softmax attention 相同的数学算子；不同浮点运算顺序仍可能产生舍入差异。论文另列的 block-sparse 扩展会改变连接图，不能与 dense exact 路径混作同一算法。
- 性能边界：速度、显存和 HBM 访问数字绑定论文 GPU、baseline、mask、shape、dtype 和执行设置。理论 pair count 或旧硬件上的论文数字不能推出目标设备的端到端吞吐。

### FlashAttention-2

- 来源：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)，Dao，2023。阅读摘要、§3.1–3.3、§4 和 §5。
- 支持：通过减少非矩阵乘 FLOPs、增加单 head 的 block 并行、改进 warp 工作分配来提升 GPU 利用率；不改变 dense attention 的 pair 数。
- 边界：论文的相对速度、TFLOPs/s 和模型 FLOPs utilization 绑定 A100、具体 shape、dtype、baseline 与训练设置。

### 稀疏、低秩与 kernel 特征路线

- 来源：[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)，Beltagy 等，2020。阅读摘要、§2–3。
- 支持：Longformer 将局部窗口 attention 与少量任务相关 global attention 组合。固定窗口和固定数量 global tokens 时，其连接数量对序列长度近似线性。
- 边界：这是指定连接图及任务设计；局部/全局 pattern 会改变 full attention 的可见关系，不能当作同一个 dense 算子仅换 kernel。

- 来源：[Linformer: Self-Attention with Linear Complexity](https://arxiv.org/abs/2006.04768)，Wang 等，2020。阅读摘要和主要方法分析。
- 支持：作者提出以低秩近似压缩序列维，形成线性时间/空间复杂度路线。
- 边界：低秩近似、秩设定、模型规模和论文任务决定其结论；不表示任意 attention matrix 都能以同一秩准确表示。

- 来源：[Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)，Choromanski 等，2021。阅读摘要和 FAVOR+ 方法说明。
- 支持：FAVOR+ 使用正交随机特征近似 softmax attention kernel，使固定随机特征维度下的序列计算和内存呈线性形式，并给出论文假设下的估计误差分析。
- 边界：这是对 kernel 的随机特征估计，不是 dense softmax attention 的逐项精确等价；论文保证和性能需连同特征数、模型、任务及实验条件阅读。

### 在线 softmax 的算法讲解核对

- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2)，附录 B；算法细节见论文的 forward pass 与 online normalization。
- 支持：对于每个 query 行，可维护当前最大值 $m$、稳定归一化质量 $\ell$ 与加权和值向量 $u$；加入新 key/value tile 时，先按新最大值重标定旧统计，再累入新 tile。最终输出为 $u/\ell$。按此状态追踪即可说明 softmax 跨块归一化为何不会被拆成互不相关的局部 softmax。
- 边界：上述是解释 dense forward 核心的数学支架；实际 GPU kernel 还要选择 tile、调度 warp、处理 mask/dropout、反向传播及硬件布局。

## 教学解释来源

- 来源：[From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)，Zihao Ye，UW CSE 599M Spring 2023，2023-05-11。阅读 §1–3。材料先指出整行 softmax 与分块处理的冲突，再用 online softmax 推出 tile 间的重标定和累积。
- 用途与边界：适合作为“为什么只做 tile matmul 不够，必须同时维护全局归一化统计”的教学路径参考；技术公式仍以 Dao 等人的原始算法核对。

- 来源：[Scalable AI: Bridging Theory, Understanding, and Practice — Lecture 6: Attention and MoE Kernels](https://scalable-ai.eecs.berkeley.edu/F2026/assets/lecture_slides/lecture6.pdf)，UC Berkeley Fall 2026，阅读第 2–7 页。
- 用途与边界：讲义依次从 score 矩阵的平方存储、HBM 读写、两个 block 的加权合并、稳定在线统计、tile 调度推进到 backward 重算；其中两块加权例子能揭示“不能直接平均各块局部输出”。它提供本课程采用的讲解选择，不证明这套顺序对所有读者具有普遍学习效果；技术结论回到 FlashAttention 原文。

- 来源：[CS336 Spring 2025, Lecture 5 — GPUs](https://cs336.stanford.edu/spring2025-lectures/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf)，Stanford，阅读 Part 3 “Using what we know to understand Flash Attention”。
- 用途与边界：将先前 GPU memory hierarchy 与 recomputation 知识带入 FlashAttention，借图对比朴素 attention 多 kernel 与融合 kernel。这里只参考衔接设计和图示方式，不使用其课程图表作为实测性能证据。

## 时效与待核边界

- FlashAttention-2/3/4 的并行调度、异步流水、dtype、GPU 和版本支持随硬件与实现更新；版本细节及当前官方仓库入口见[FlashAttention IO、在线 softmax 与硬件演进记录](flashattention-io-softmax-and-hardware-evolution-2026.md)。本章关于 dense attention 的结论不依赖某一发布版本的 kernel 性能。
- PyTorch SDPA、GPU kernel 的覆盖、causal fast path、mask 语义和 fallback 都是版本/shape 相关的实现事实。没有核对指定 runtime 的源码或设备 profile 时，只说明“连接数/矩阵 shape/算法复杂度”，不宣称实际 kernel 已跳过被 mask 的算术或达到具体速度。
- 本地没有运行教学代码、FlashAttention kernel、模型或性能基准；实验结果只按来源报告，不写成本地复现。
