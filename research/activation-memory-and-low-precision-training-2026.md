# 长上下文训练显存与低精度训练证据

检索与核验日期：2026-11-02。以下记录论文作者报告、固定版本官方文档及其适用边界；它们不构成本地运行或目标硬件复现结果。

## OOMB：百万 token 训练的系统级显存方案

- **来源与阅读范围：** [Out of the Memory Barrier: A Highly Memory Efficient Training System for LLMs with Million-Token Contexts](https://arxiv.org/abs/2602.02108v4)，arXiv:2602.02108v4（2026-03-01）；阅读摘要、系统方法（§3–4）、实验结果（§5）和结论/限制（§6）。
- **来源支持的事实：** OOMB 将 chunk-recurrent 执行、在线激活重算、分页 KV 管理、异步 CPU offload、自定义 kernel 和 page-level sparse attention 组合成训练系统。论文表 1 另列 OOMB + Dense Attention 配置；作者报告其 Qwen2.5-7B 设置可在单张 H200 上使用最高 4M token context。对该模型配置，作者报告 context 每增加 10K token，端到端训练显存开销约增加 10MB。论文也比较 sparse-attention 配置；O(1) activation memory 是这套系统的作者分析，不能直接当作任意模型的复杂度保证。
- **证据边界：** chunk-wise 串行处理会增加相对全并行方案的延迟；page-level sparse attention 是对 dense attention 的近似，其对需要稠密全局上下文任务的影响仍需评测；异步 offload 的效果依赖 CPU–GPU 互连带宽。单 H200 与 4M context 数字绑定论文实现、模型、数据和调度方案，不代表一般 checkpoint API 或普通 GPU 训练可以达到相同增长率。
- **证据状态：** source-backed；未在本地复现。

## 全量 FP4 训练的两种研究路线

### FP4 All the Way

- **来源与阅读范围：** [FP4 All the Way: Fully Quantized Training of LLMs](https://arxiv.org/abs/2505.19115v2)，arXiv:2505.19115v2（2025-08-10）；阅读摘要、方法 §3、理论分析 §4.2、训练实验及 limitations。
- **来源支持的事实：** 作者使用 E2M1 FP4 值、每 16 个值共享 E4M3 scale 的 NVFP4-like 格式；forward 使用 round-to-nearest，backward/update 路径使用 stochastic rounding。论文报告 Llama2-7B 在 RedPajama 上训练 1T token、使用 256 张 Intel Gaudi2；作者称短暂 QAF 阶段可弥合其与 BF16 的最终训练 loss 差距，并报告下游任务表现接近 BF16。
- **证据边界：** Gaudi2 没有专用 FP4 支持，论文明确说明 FP4 实验以模拟方式运行并承担额外精度转换开销；论文未实测原生 FP4 的时间或能耗。作者给出的相对 FP8/BF16 time-to-train 加速是估计，不能写成 Gaudi2 原生测量结果。
- **证据状态：** source-backed；未在本地复现。

### Quartet

- **来源与阅读范围：** [Quartet: Native FP4 Training Can Be Optimal for Large Language Models](https://arxiv.org/abs/2505.14669v4)，arXiv:2505.14669v4（2026-01-15）；阅读格式和误差分析、Quartet 方法与训练结果（§4–5）及结论/限制（§6）。
- **来源支持的事实：** Quartet 面向 Blackwell 支持的 MXFP4；该格式使用 E2M1 值、每 32 个值共享 E8M0 scale。作者方案结合 forward 的 QuEST 量化与 backward 的 round-to-nearest 路径，以 FP4 执行线性层训练所需的矩阵乘。论文报告 C4 上 Llama 系列训练与 scaling 实验，并单独在 RTX 5090 上测量单层和 Transformer block kernel 效率；7B 预训练使用 8 张 H100，作者报告该项约 6,000 GPU-hours。
- **证据边界：** RTX 5090 的局部 kernel 测量不等于相同幅度的端到端训练提速；论文强调算法针对特定数据类型和计算架构设计，并把更大规模分布式模型执行列为后续工作。训练质量、kernel speed 和推理 prefill speed 必须分开读。
- **证据状态：** source-backed；未在本地复现。

## PyTorch 2.15 的显存、AMP 与重算接口

- **CUDA allocator：** [CUDA semantics — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/notes/cuda.html) 说明 PYTORCH_ALLOC_CONF 控制缓存 allocator；PYTORCH_CUDA_ALLOC_CONF 是向后兼容别名。expandable_segments 是默认关闭的实验性选项，目标是更好处理经常变化的分配大小，并不减少仍活着的张量占用。empty_cache 释放未占用的缓存段供其它 GPU 应用使用，不释放活张量。
- **显存快照：** [Understanding CUDA Memory Usage — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/torch_cuda_memory.html) 说明快照默认观察 PyTorch allocator 管理的 CUDA 设备内存；直接由其他库申请的 CUDA 内存不保证可见。该版本提供通过内部接口 _record_memory_history(record_pinned_host_memory=True) 记录 pinned host memory 的选项，数据出现在 host_segments/host_traces，但官方 visualizer 尚不显示 host memory。
- **AMP：** [Automatic Mixed Precision — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/amp.html) 将 torch.cuda.amp.autocast 与 torch.cuda.amp.GradScaler 标为 deprecated，推荐 torch.amp.autocast("cuda", ...) 与 torch.amp.GradScaler("cuda")。该页还说明 torch.compile 下 backward_pass_autocast 默认 same_as_forward；若采用 forward 在 autocast 内、backward 在外的推荐训练模式，编译区域应根据该版本指引将其设为 off。该配置路径位于带下划线的内部命名空间，依赖版本。
- **Accumulation：** [Automatic Mixed Precision examples — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html) 要求同一个 effective batch 累积期间梯度保持 scaled 且 scale 不变；一个 optimizer 在 step 前只 unscale 一次，且须等该 optimizer 的所有 micro-batch 梯度累积完成。
- **Activation checkpoint：** [torch.utils.checkpoint — PyTorch 2.15](https://docs.pytorch.org/docs/2.15/checkpoint.html) 要求显式设置 use_reentrant，并推荐 use_reentrant=False；默认 preserve_rng_state=True。非 reentrant 路径支持 early-stop 重算等额外行为。前向与重算若不等价，仍可能造成错误梯度。
