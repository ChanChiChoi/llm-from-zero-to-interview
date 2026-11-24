# 分布式训练系统与接口证据（2026-11-24）

本文只记录可用于分布式训练正文的技术事实、来源和适用边界。

## PyTorch 2.15 DDP

来源：[DistributedDataParallel 2.15 API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)

- DDP 将梯度按 data-parallel process group 归约；文档说明，对局部 summed loss 求导后再按 rank 平均，与全局求和目标的梯度相差参与 rank 数的比例因子，因此 token loss 的全局分母应按实际 reduction 语义推导。
- `bucket_cap_mb` 的文档默认值为 25 MiB；bucket 形成和发送时序受梯度就绪次序影响，默认值不是特定模型、拓扑上的性能保证。
- `join()` 可让提前耗尽输入的 rank 影子执行 DDP 通信，使剩余 rank 继续；作业还包含 DDP 不知晓的 collective（如 SyncBatchNorm）时，需使用 `throw_on_early_termination=True`，以免其余 rank 永久等待。
- `divide_by_initial_world_size=True` 按初始 DP group 大小缩放尾部梯度；关闭时按仍活动的 rank 数缩放，二者对尾部样本的权重不同。

## PyTorch 2.15 FSDP2 与 Distributed Checkpoint

来源：[FSDP2 `fully_shard` 2.15](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)、[Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)

- FSDP2 按模块划分参数组，前向计算前 all-gather 参数，反向后 reduce-scatter 梯度。逐模块应用 `fully_shard` 会形成通信组；模块顺序与粒度影响显存和通信重叠。
- reshard-after-forward 会在前向后释放完整参数，并在反向需要时重新聚齐；实际峰值需结合模块布局、参数获取时机与临时 buffer 测量。
- DCP 支持多 rank 分布式保存/加载、根据目标布局 load-time resharding，以及异步保存。`async_save` 的 `staging_completion` 表示本地 staging 已完成，`upload_completion` 表示上传已完成。
- DCP 不替应用承诺 checkpoint 目录的原子提交；多 shard 一致提交、manifest 版本与训练状态步号的对应应由应用/存储协议保证。DCP 文档也未承诺跨任意 PyTorch 版本的向后兼容。

## Megatron Core 并行语义

来源：[并行策略指南（latest）](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)、[Context Parallelism（latest）](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)

- Megatron Core 的 context parallelism 将序列输入/激活沿 context 维分片；其 attention 路径可在设备间交换 K/V 块，并在反向中归约相应梯度。具体通信实现受配置影响，不能把某一种 ring 路径等同于 CP 的唯一实现。
- Megatron 文档中的 sequence parallelism 与 context parallelism 切分目标不同：SP 主要配合 TP，缩小特定操作（例如 normalization、dropout）上的序列激活；CP 将长序列 token 区间分配到不同 rank，并在 attention 中交换远端所需的信息。
- `latest` 页面持续更新；记录的接口和配置语义应附核对日期，不应被视作固定版本 API 或性能承诺。

## PyTorch 2.15 分布式运行时更新

来源：[PyTorch 2.15 发布说明（2026-10-28）](https://pytorch.org/blog/pytorch-2-15-release-blog/)

- 发布说明称，重写后的 NCCL backend 成为 NCCL process group 的默认实现，并增加 collective hang 诊断记录/flight recorder 能力。
- FSDP2 与 pipeline parallel 的相关更新描述的是 PyTorch 2.15 的接口与运行时行为；不能据此推断旧版、非 NCCL backend 或其他厂商 runtime 具有相同能力。

## 论文结果的边界

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) 在特定 GPT-3/MT-NLG、TP=8、序列长 2048 和 A100 配置下报告了选择性激活重算的显存/FLOPs 变化；SP 与选择性重算组合结果不能解释为所有 Transformer 的固定收益。
- [PaLM](https://arxiv.org/abs/2204.02311) 的 MFU/HFU 报告和 rematerialization 开销属于论文所述 540B 模型及 TPUv4 设置。
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) 描述其 16 路 PP、64 路 EP、ZeRO-1 DP 与 DualPipe 等具体系统布局；这些配置和通信重叠结果不直接构成其他模型的默认设计。

