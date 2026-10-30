# FlashAttention 的 IO、在线 softmax 与硬件演进

核验日期：2026-10-29。本文记录联网发现并核对的技术事实、来源版本、实际阅读位置和证据边界。

## FlashAttention（2022）

- 来源：[Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv:2205.14135v2](https://arxiv.org/html/2205.14135v2)，v2 日期 2022-06-23。阅读摘要、§2.2、§3.1–3.3、§4、§5、附录 B.1–B.4 和附录 C 的内存分析。
- 标准实现将 score 矩阵 `S=QKᵀ` 与 softmax 概率矩阵 `P` 写入 HBM，产生随序列长度平方增长的矩阵存储和 HBM 访问。FlashAttention 的 dense 算法以 tiling 把 Q/K/V block 载入片上 SRAM，并通过在线归一化累积 attention 输出；训练反向传播保存必要的逐行统计量，在 backward 重新计算 attention matrix，而非持久保存完整矩阵。论文的额外内存为 O(N)，不包括输入、输出、输出梯度和输入梯度；主要计算量仍为 O(N²d)。
- 论文的“exact”指 dense 核心计算同一 full softmax attention 算子，不近似/丢弃完整 attention 连接；浮点累加顺序不同仍会有舍入差异。原论文还提出 block-sparse 扩展，该扩展只计算给定稀疏模式中的 block，属于相对全连接 attention 的近似路径；不能将它与 dense exact 核心混写。
- 原论文中的速度、内存和任务质量结果属于其 GPU、基线、序列长度和实验设置。该论文支持 IO-aware tiling 的方法与实验结论，不支持对现代 GPU 或所有 serving shape 的普遍速度承诺。

## FlashAttention-2（2023）

- 来源：[Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv:2307.08691v1](https://arxiv.org/html/2307.08691)，v1 日期 2023-07-17。阅读摘要、§3.1–3.3、§4 和§5。
- FA2 调整非矩阵乘 FLOPs、thread block 与 warp 的工作分配，并沿序列维并行单个 head 的工作，以提高 occupancy、减少共享内存通信。论文摘要报告其在 A100 上相对 FA1 约 2 倍以及特定 GPT 训练配置下最高 225 TFLOPs/s、72% MFU；这些数值绑定论文的 GPU、形状、dtype、基线和训练设置。

## FlashAttention-3（2024）

- 来源：[Shah et al., *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, arXiv:2407.08608v2](https://arxiv.org/html/2407.08608)，v2 日期 2024-07-12。阅读摘要、§2.2、§3.1–3.3、§4、§5。
- 该工作针对 Hopper GPU，使用异步 Tensor Core 与 Tensor Memory Accelerator（TMA）搬运、warp specialization/ping-pong scheduling 等技术重叠数据移动和计算；另提出 FP8 路径。论文的主实验使用 H100，报告特定设置下相对 FA2、PyTorch 和 cuDNN 的性能。FP8 章节单独评估数值误差，表明低精度收益应与参考精度误差一起报告；论文也列出短序列/因果场景的实现限制。
- 论文结果不代表所有 H100/Hopper workload，也不能外推到所有 decode kernel、dtype 或 API 集成。

## FlashAttention-4（2026）

- 来源：[Zadouri et al., *FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling*, arXiv:2603.05451v1](https://arxiv.org/html/2603.05451)，v1 日期 2026-03-05。阅读摘要、§2、§3.1–3.3、§4、§5 和附录 A.1。
- 论文针对 Blackwell 的计算吞吐与 shared-memory/指数单元扩展不对称，提出 fully asynchronous MMA pipeline、较大 tile、软件 exp 与 conditional softmax rescaling、tensor memory 和 2-CTA backward 路径，并使用 CuTe-DSL 实现。
- 论文速度实验在 B200 上以 BF16 运行，序列长度为 1k–32k、总 token 数固定为 32k，并测试 head dimensions 64、128 及 `(192,128)` 等设置。论文报告相对 cuDNN 9.13 最高约 1.3 倍、相对 Triton 最高约 2.7 倍；同一论文明确说更新的 cuDNN 纳入了若干技术后性能已相近。FA3 未在 B200 上运行，因此该论文不构成 FA3/FA4 在相同硬件上的直接对照。
- 这些数字不是跨 GPU 的性能保证。可比结论必须保留 GPU、cuDNN/Triton 版本、dtype、attention mask、head dimension、序列长度和 token batching 条件。

## 官方实现仓库（访问时状态）

- 来源：[Dao-AILab/flash-attention README](https://github.com/Dao-AILab/flash-attention)，访问日期 2026-10-29。当前 `main` 页面将 FA4 标为 CuTeDSL 实现，并称其面向 Hopper 与 Blackwell（例如 H100、B200）。
- GitHub `main` 是可变页面；上述是访问时 README 状态，不是固定 release 的支持合同。具体 CUDA、PyTorch、GPU、dtype、shape 与接口兼容性应按固定 commit/release 核对。

## 独立的 FP4 预印本与名称区分

- 来源：[Hu, *Hardware-Aware FP4 FlashAttention-4*, arXiv:2609.04105v1](https://arxiv.org/abs/2609.04105)，提交日期 2026-09-03。该条目作者为 Robert Hu，与 2026-03 的 FA4 多作者论文是不同 arXiv 作品，也未出现在上述 Dao-AILab 官方 README 的 FA4 版本说明中。
- 摘要提出 GB200 上的 Direct-P FP4 概率路径与因果训练路径，报告特定非因果前向的最高 2.13 倍 BF16 throughput、单 GPU 8B 更新的最高 1.14 倍；作者还报告匹配的分布式训练中，测试的 MXFP4 probability/value 轨迹出现 divergence。以上仅是该单篇预印本摘要所述的窄范围实验，尚不能当作主线 FA4 版本或通用 FP4 注意力行为。

## 结论边界

- dense FlashAttention 的主要矩阵乘计算仍随 `T²` 增长；算法层面的额外行统计量可随 `T` 线性增长，但这不代表完整训练或模型显存线性增长。
- IO 优化是否提高端到端速度取决于工作形状、精度、kernel 覆盖、GPU、其他算子和服务调度。论文峰值和单 kernel 结果不能替代目标负载评估。
- “exact”不等于 bitwise identical，也不自动覆盖 FP8/FP4 低精度路径、block-sparse 变体或所有框架实现。
- 本记录未运行本地 FlashAttention kernel、GPU benchmark、模型训练或 serving 测量；数值结论均为来源所报告的实验结果或论文算法分析。
