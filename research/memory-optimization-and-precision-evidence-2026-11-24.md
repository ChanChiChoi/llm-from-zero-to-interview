# 训练显存优化与数值精度证据（2026-11-24）

本文记录可用于第五册显存与精度正文的技术事实、来源和适用边界。

## PyTorch 2.15 接口与实现

- [AMP 2.15](https://docs.pytorch.org/docs/2.15/amp.html) 将 `torch.cuda.amp.autocast` 与 `torch.cuda.amp.GradScaler` 标记为弃用，并指向 `torch.amp.autocast("cuda", ...)` 与 `torch.amp.GradScaler("cuda", ...)`。与 `torch.compile` 联用时，文档要求在推荐的 autocast forward、autocast 外 backward 形态下将编译区域 `backward_pass_autocast` 设为 `"off"`。
- [AMP examples 2.15](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)：同一 effective batch 的梯度累积期间保持 scale 不变；所有 micro-batch 完成后再 unscale、裁剪、step 和 update；每个 optimizer 在一次 step 前最多调用一次 `unscale_`。
- [Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/checkpoint.html)：应显式传 `use_reentrant`，推荐 `False`；non-reentrant 路径支持 early-stop recompute。RNG 保存/恢复默认开启但有开销；依赖变动的全局状态或在重算中迁移至新设备可能破坏前后向等价性。
- [CUDA semantics 2.15](https://docs.pytorch.org/docs/2.15/notes/cuda.html)：`expandable_segments` 是实验性 allocator 配置，默认关闭；它帮助处理变化大小的分配请求，不会释放仍存活的张量。
- [CUDA memory guide 2.15](https://docs.pytorch.org/docs/2.15/torch_cuda_memory.html)：默认 snapshot 只记录 PyTorch CUDA allocator 管理的设备分配，NCCL 和直接 CUDA 分配可能不可见；私有 `_record_memory_history(record_pinned_host_memory=True)` 可纳入 pinned host allocator 的 segments/traces，官方可视化器尚不展示该 host 数据。
- [PyTorch Adam v2.15.0 源码](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/optim/adam.py)：对有梯度的参数，Adam 在 `_init_group` 中懒创建 `step`、`exp_avg` 和 `exp_avg_sq`；因此第一次 optimizer step 可能首次分配持久状态。AMSGrad 还会创建 `max_exp_avg_sq`。

## Transformer Engine 2.21.0 低精度格式

来源：[Using FP8 and FP4 with Transformer Engine 2.21.0](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html)

- 文档将 H100 的 E4M3/E5M2 FP8 用法与 Blackwell 对 MXFP8、NVFP4 的硬件支持区分；delayed scaling 基于历史 amax 选择 scale，current scaling 需先获得当前输出的 amax。
- MXFP8 对每 32 个连续值用 E8M0 power-of-two scale；训练矩阵乘的转置布局可能要求另存从高精度源分别生成的正向和转置量化副本。
- 通用 NVFP4 表示采用 E2M1 数值、每 16 值的 E4M3 scale，并附 per-tensor FP32 scale。Transformer Engine 的线性权重训练 recipe 使用 16×16 的二维权重 scale，以处理权重及其转置之间的表示一致性；不应把该 recipe 与 NVFP4 的通用 1D 格式定义混为一谈。
- TE NVFP4 recipe 还涉及随机舍入、激活/梯度的 Hadamard 变换以及最后若干层的高精度执行建议；这些是该实现的训练配方，不等于所有 FP4 实现都采用相同路径。

## 低比特训练论文

- [FP4 All the Way v2](https://arxiv.org/abs/2505.19115v2) 主文报告 Llama2-7B 在 Red Pajama 上训练 1T token，使用 256 张 Gaudi2、全局 batch 4M token、训练约 30 天；Gaudi2 当时没有专用 FP4 执行支持，FP4 实验使用模拟，因此不能据此声称 native FP4 实测速度/能耗。摘要说其研究数据集最高 200B tokens；主文的 1T 是模型训练 token 数，二者口径应分开。
- [Quartet v4](https://arxiv.org/abs/2505.14669v4) 针对 Blackwell MXFP4：前向用 QuEST 方案，反向用 RTN，线性层矩阵乘在 MXFP4 执行。论文在 C4 训练/验证任务上报告不同规模模型与精度结果；RTX 5090 的效率测量包含特定线性层及 Transformer block 聚合，不能直接解释为任意端到端分布式训练加速。
- [QLoRA](https://arxiv.org/abs/2305.14314) 报告 NF4、Double Quantization 与 paged optimizer。在 7B LLaMA、FLAN v2、batch=1 的内存案例中，LoRA input gradients 为 567 MB、LoRA 参数 26 MB、4-bit base 5,048 MB；启用 gradient checkpointing 后 input-gradient 平均为 18 MB/序列。均是论文实现与设置的统计，不是普适设备预算。
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) 使用 2048 元素 block、dynamic quantization 和 stable embedding 等组件。其 209M RoBERTa-corpus 对照中，完整 8-bit recipe 表列 0% unstable runs、perplexity 16.4，32-bit Adam 为 0%、16.7；该表只支持作者所测配置与稳定性定义。

## Attention 与重计算

- [FlashAttention v2](https://arxiv.org/abs/2205.14135) 是 IO-aware 的精确 attention：分块降低 HBM 读写和完整 score/probability 中间量物化，仍计算全部可见 query-key 对，标准 full attention 的算术量仍随序列长度平方增长。
- [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198) 的 sequence parallel 与 selective recomputation 结果绑定 Megatron 配置和 A100 集群。其 530B、2240 A100 案例报告 MFU 54.2% 对 42.1%，约 29% 快于完整重算对照；摘要报告 activation memory 约降 5×，不构成其他实现的固定比例。

## OOMB 长上下文系统

来源：[OOMB v4](https://arxiv.org/abs/2602.02108v4)

- 作者将 on-the-fly activation recomputation 的 activation memory 复杂度报告为对 context length 的 O(1)，同时说明主要增长项转为 KV cache；端到端显存不能因此说成常数。
- 作者报告 Qwen2.5-7B 每增加 10K 上下文，端到端训练内存开销约增 10MB，并声称单 H200 可训练 4M context。H200 单卡表的 dense attention 测量覆盖至 256K；4M 以上的部分扩展图采用 4×H200 tensor parallel。引用时要保留不同实验的设备/attention 口径。
- OOMB 将 dense attention 与 page-level sparse attention 区分；稀疏路径是近似 attention，作者还指出分块串行调度可能增加相对全并行方法的延迟，CPU offload 效果依赖 CPU–GPU 互连。

