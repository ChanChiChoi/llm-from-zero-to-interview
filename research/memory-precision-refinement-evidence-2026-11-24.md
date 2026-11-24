# 显存优化与数值精度章节精修证据

检索日期：2026-11-24。记录第五册第8章修订时新核对的原始论文与指定版本官方文档。此文件只保存新核实的知识、证据来源和未验证边界，不记录章节审阅进度。

## 证据等级

- **A：原始论文或正式技术报告。** 支持论文所述定义、算法和作者在明确实验设置下报告的结果；不自动支持跨模型、数据、硬件和实现的普遍收益。
- **A-：指定版本的官方框架文档或源码。** 支持版本内接口语义、配置和实现路径；不能外推到其他版本、后端或框架。
- **B：官方工程文档。** 支持核对日期所列的工程机制；具体运行效果仍需绑定软件版本和目标配置。

## 显存观测、optimizer 与 AMP

| 主题 | 证据与可支持内容 | 未验证边界 |
| --- | --- | --- |
| PyTorch CUDA allocator 与 snapshot | **A-**：PyTorch CUDA semantics 2.15，[文档](https://docs.pytorch.org/docs/2.15/notes/cuda.html)，说明 `PYTORCH_ALLOC_CONF`、实验性 `expandable_segments` 和 allocator 管理方式。PyTorch 2.15 [memory snapshot 文档](https://docs.pytorch.org/docs/2.15/torch_cuda_memory.html)说明 snapshot 默认只记录经 PyTorch allocator 管理的 CUDA 分配；`record_pinned_host_memory=True` 可记录 pinned-host allocator 的 segments/traces，但官方 visualizer 尚不展示该部分。 | CUDA API、NCCL 或第三方库绕过 PyTorch allocator 的分配不在默认 snapshot 中；pinned-host snapshot 不代表进程全部 RSS、页缓存或 NVMe 占用。`expandable_segments` 改变分配段行为，不减少仍活着的张量字节数；须在实际 allocator backend 和输入形状分布下验证。 |
| Adam 首次状态与 foreach 峰值 | **A-**：PyTorch v2.15.0 [Adam 源码](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/optim/adam.py)按首次有梯度更新的参数惰性初始化 `exp_avg`、`exp_avg_sq` 等状态。PyTorch 2.15 [Adam 文档](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html)指出 foreach 路径的 tensor-list 中间量带来约 `sizeof(params)` 的额外峰值内存。 | 参数状态首次创建和 foreach 临时量是不同的显存来源；该文档数字不能直接套用到 fused、for-loop 或其他 optimizer 实现。 |
| AMP 与累积 | **A-**：PyTorch AMP 2.15 [文档](https://docs.pytorch.org/docs/2.15/amp.html)推荐 autocast 只包 forward/loss；[AMP examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)规定 GradScaler 在一个 effective batch 的梯度累积期间保持 scale 不变，累积结束后再 unscale/step/update，且每个 optimizer 每次 step 前仅 unscale 一次。`torch.compile` [反向语义文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_backward.html)说明默认 `backward_pass_autocast="same_as_forward"`；若 backward 在 autocast 外运行，应为编译区域设为 `"off"`。PyTorch 2.15 [DDP 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)规定 `no_sync()` 的 forward 也必须位于 context 内。 | 这些是指定版本接口与推荐调用边界，不证明任意模型的梯度数值稳定，也不替代实际 compile/DDP 配置验证。 |
| Activation checkpointing | **A-**：PyTorch checkpoint 2.15 [文档](https://docs.pytorch.org/docs/2.15/checkpoint.html)推荐显式使用 `use_reentrant=False`；默认保留 CPU 和一个设备类型的 RNG 状态，重算可有开销；非重入实现默认可在所需张量已重建后提前停止。若重算函数依赖变化的全局状态、执行不同分支或在函数内转移到新设备，重算可能与 forward 不等价。 | API 语义不提供固定显存节省倍率；实际 saved tensors、重算峰值、kernel workspace 和通信重叠必须测量。 |

## Attention、精度格式和量化状态

| 主题 | 证据与可支持内容 | 未验证边界 |
| --- | --- | --- |
| FlashAttention | **A**：Dao et al., *FlashAttention*, arXiv:2205.14135 v2，[论文](https://arxiv.org/abs/2205.14135)。分块 online-softmax 的 exact attention 减少 HBM 与片上 SRAM 间读写，避免把完整 score/probability 矩阵物化到 HBM。 | exact full attention 仍处理全部可见 query-key 对；论文的 IO 和速度分析不能推出当前任意 kernel 的吞吐，也不能把算术量改写为线性。 |
| Transformer Engine FP8/FP4 | **A-**：NVIDIA Transformer Engine 2.21.0 [FP8 Primer](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)描述 delayed-scaling、MXFP8 每 32 个连续值的 E8M0 scale、NVFP4 每 16 个值的 E4M3 scale 与 tensor scale；训练线性层权重路径另有 16×16 scale 用于处理权重及转置布局。 | 数值格式定义与可用训练 recipe 分开看；需绑定 TE、GPU、kernel 和权重布局版本。FP4/NVFP4 数值计算格式与 QLoRA NF4 冻结权重存储格式不是同一方法。 |
| 8-bit optimizer | **A**：Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861 v2，[论文](https://arxiv.org/abs/2110.02861)。Adam 的 32-bit m/v 为每参数 8 bytes，论文方案将状态压到 8-bit；通过每 2,048 个元素分块、dynamic-tree quantization 和 stable embedding 处理误差及 embedding 梯度分布。209M RoBERTa-corpus 小模型消融中，启用 dynamic、blockwise 和 stable embedding 的 8-bit Adam 报告 0% unstable runs、median perplexity 16.4；同表 32-bit Adam + stable embedding 为 0%、16.3。 | 结果绑定论文训练语料、模型、optimizer、超参和“unstable”定义。论文指出 stable embedding 及高精度 embedding state 是方案的一部分；不能把结果解释为所有 8-bit optimizer 无条件稳定。 |
| QLoRA | **A**：Dettmers et al., *QLoRA*, arXiv:2305.14314 v1，[论文](https://arxiv.org/abs/2305.14314)。NF4 利用近似零中心正态权重分布构造 quantile code；double quantization 再量化 block scale，论文报告平均节省约 0.373 bit/parameter；paged optimizer 使用 CUDA Unified Memory 应对瞬时内存压力。 | 量化底座主要减少冻结权重存储，不等于整个计算均在 4-bit，也不消除激活、反量化工作区或 adapter state。论文 65B/48GB 是其方法、数据与实验设置下的结果。 |

## FP4 全量训练和超长上下文系统

| 主题 | 证据与可支持内容 | 未验证边界 |
| --- | --- | --- |
| FP4 All the Way v2 | **A**：Chmiel et al., arXiv:2505.19115 v2，[摘要](https://arxiv.org/abs/2505.19115v2)与[实验全文 §5–6](https://arxiv.org/html/2505.19115v2)。§5 报告 Llama2-7B、RedPajama、1T token、256 Gaudi2；NVFP4 为 E2M1、16 元素一组、E4M3 scale。权重及 forward activations 用 round-to-nearest；反向/更新 GEMM 中的 neural gradients 和更新 GEMM 的 activations 用 stochastic rounding。QAF 再训练约 40B tokens，forward 留在 FP4、backward/update 改用 BF16，作者称 loss 差距闭合。 | 同一 v2 的摘要写数据集规模“up to 200 billion tokens”，与 §5 的 1T 记录不一致。Gaudi2 不支持专用原生 FP4 执行，作者在 Gaudi2 上模拟 FP4；所述 time-to-train 提速约 35–40%（相对 FP8）、约 85%（相对 BF16）是估算而非该实验的原生 FP4 实测。 |
| Quartet v4 | **A**：Castro et al., arXiv:2505.14669 v4，[论文](https://arxiv.org/abs/2505.14669v4)与[全文](https://arxiv.org/html/2505.14669v4)。方案面向 Blackwell MXFP4（E2M1、每 32 个值一份 E8M0 scale）；forward 用 Hadamard + QuEST，backward 用随机 Hadamard + round-to-nearest。质量实验使用 C4 的 Llama 模型，并测试到约 7B；资源说明写明预训练实验在 8×H100。RTX 5090 的速度实验测单层及 Transformer block，batch=64、sequence=512；论文报告该测量中最高约 1.6× FP8、2.9× BF16 的训练速度。 | RTX 5090 block 测量不等于 7B 端到端或分布式训练速度；论文把更大规模分布式模型执行列为后续工作。结果绑定 Blackwell/MXFP4 实现。 |
| OOMB v4 | **A**：Li et al., arXiv:2602.02108 v4，[论文](https://arxiv.org/abs/2602.02108v4)。OOMB 将 chunk-recurrent 调度、在线 activation recomputation、分页 KV/gradient 管理、异步 CPU offload 与 dense/sparse attention 组合。摘要和结论报告 Qwen2.5-7B 单 H200、4M context，并给出每多 10K tokens 约 10MB 的作者显存估计；论文另报告单 H200 的 dense-attention 对照到 256K、4M–8M 扩展图使用 4×H200 TP。 | sparse attention 近似 dense attention；单卡与四卡实验不能混写成同一条硬件配置。常数级 activation memory 不代表整卡所有状态与 KV 都为常数，也不是普通 checkpoint API 的保证。 |

## 验证边界

上述原始论文支持特定格式、算法和作者实验设置；PyTorch 与 Transformer Engine 文档支持其版本的接口/recipe 语义。第8章中的账本和数值示例用于说明关系，不等于目标机器峰值预测；真实训练要记录 GPU/版本、输入 shape、dtype、并行配置、allocator backend、optimizer 实现、有效 token 数、peak memory、吞吐与任务质量。
