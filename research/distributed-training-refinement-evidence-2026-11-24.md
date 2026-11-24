# 分布式训练章节精修证据

检索日期：2026-11-24。记录第五册第7章精修所核对的论文与指定版本官方文档。此文件只保存本轮新核实的知识、证据来源和未验证边界，不记录逐段审阅进度。

## 证据等级

- **A：原始论文或正式技术报告。** 支持论文中的定义、作者算法和该论文具体实验；不自动支持跨硬件、模型或 runtime 的普遍收益。
- **A-：指定版本的官方框架文档或发布说明。** 支持版本内接口语义与发布能力；不能外推到其他版本、后端或框架。
- **B：持续更新的官方工程指南。** 支持文档核对日列出的配置路线；文档可能变更，具体执行行为仍需绑定软件版本和代码路径。

## 并行调度与注意力切分

| 主题 | 证据与可支持内容 | 未验证边界 |
| --- | --- | --- |
| 并行布局与 pipeline 调度 | **A**：Megatron-LM, arXiv:2104.04473, §2.2、§3.1、§5.3，[论文](https://arxiv.org/abs/2104.04473)。其配置满足 $PP\times TP\times DP=GPU\ count$；模型切分后每个 TP/PP 坐标持有相应局部层/张量状态，再由 DP 维复制或分片。该论文的 flush pipeline bubble/ideal-time 口径为 $(p-1)/m$；交错 schedule 可降低气泡，但会增加每个样本的 pipeline 通信。作者观察到收益会随 batch 与 scatter/gather 优化变化。 | 局部状态大小还取决于实际权重切法、层划分不均、padding 和框架缓冲区；论文分析不能代替目标配置 profiler。GPipe, arXiv:1811.06965, §2.2，[论文](https://arxiv.org/abs/1811.06965)，把 idle bubble 量级写为 $O((K-1)/(M+K-1))$；不能把该口径和 Megatron 的 bubble/ideal-time 比值当成同一分母。 |
| Zero Bubble | **A**：Qi et al., arXiv:2401.10241，[论文](https://arxiv.org/abs/2401.10241)。方法把 backward 拆成跨 stage 依赖的输入梯度部分与可灵活调度的参数梯度部分；调度受时间、显存、通信和同步假设约束。作者报告的吞吐收益绑定具体实现和实验设置。 | “zero bubble”是调度目标/特定调度结果，不代表任意模型和配置都没有 idle time，也不保证增加吞吐。 |
| DualPipe | **A**：DeepSeek-V3 Technical Report, arXiv:2412.19437 v1, §3.2.1–3.2.2，[报告](https://arxiv.org/abs/2412.19437)。报告描述前后双向 micro-batch 调度、前反向 chunk 与 EP all-to-all/PP 通信重叠；反向拆输入梯度和权重梯度。Table 2 比较口径为 1F1B bubble $(PP-1)(F+B)$、ZB1P bubble $(PP-1)(F+B-2W)$、DualPipe bubble $(PP/2-1)(F\&B+B-3W)$；DualPipe 用 $2\times$ 参数副本和 $PP+1$ 激活记账。报告要求 PP stages 与 micro-batches 可被 2 整除，并不要求 micro-batches 必须是 PP stages 的倍数。 | 这些表格量依赖报告定义的 chunk、重叠和内存记账；不得外推为其他 MoE、通信拓扑、模型切块或实现的固定速度/显存结果。DeepSeek-V3 的训练布局为 16-way PP、64-way EP 跨 8 节点、ZeRO-1 DP，作者称未用 TP。 |
| SP 与 CP | **A-**：Megatron Core Context Parallel 官方指南（持续更新，2026-11-24 核对），[指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)。SP 主要切分 TP 下 LayerNorm/dropout 一类序列激活；CP 将输入和激活沿序列区间分给 rank，attention 需取得跨 rank K/V，且可和 DP/TP/PP 组合。 | CP 的具体环形 P2P、all-gather、a2a 路径依赖框架版本、模型路径和配置；“启用 CP”等同于 Ring Attention 不成立。 |
| Ring Attention | **A**：Liu et al., arXiv:2310.01889，[论文](https://arxiv.org/abs/2310.01889)。将 blockwise attention 与 K/V block 的 ring 传输安排为可重叠算法。 | 论文设计不能代表任意框架 CP 实现；具体效果受 kernel、因果 mask、每轮通信与硬件链路限制。 |

## 框架通信、分片与恢复

| 主题 | 证据与可支持内容 | 未验证边界 |
| --- | --- | --- |
| DDP 不等长输入与 bucket | **A-**：PyTorch 2.15 DDP 文档，[DDP 2.15](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)。`join()` 可 shadow 未耗尽 rank 的 DDP collective；`divide_by_initial_world_size` 改变尾部样本的归一化口径；`bucket_cap_mb` 默认 25 MiB 且可用于与 backward 重叠。 | `join()` 不会自动补齐 DDP 不知道的自定义 collective；bucket 最佳值依赖梯度 ready 顺序、网络和模型。 |
| FSDP2 参数生命周期 | **A-**：PyTorch 2.15 `fully_shard` 文档，[FSDP2 2.15](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)。`fully_shard` 以 DTensor 表示分片参数，模块前向/反向前聚齐参数；`reshard_after_forward` 决定前向后保留或重分片的参数状态。 | checkpoint 重算是否额外触发 all-gather 取决于重算区域、包装边界、reshard 策略与运行时，不能由 API 名称单独推定次数。 |
| Distributed Checkpoint | **A-**：PyTorch 2.15 DCP 文档，[DCP 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)。支持多 rank 分片读写和 load-time resharding；`async_save` 返回的 `staging_completion` 表示本地状态副本准备完成，`upload_completion` 表示该次 checkpoint 保存完成；文件系统 writer 会写全局 `.metadata`，并假设文件创建原子。 | DCP 不替应用证明模型、optimizer、外部数据游标与配置是同一训练步；该版本文档明确不保证跨 PyTorch 版本的 state_dict 向后兼容。其他 storage writer 的耐久性语义需单独核对。 |
| PyTorch 2.15 NCCL 诊断与流水线 | **A-**：PyTorch 2.15 Release Blog（2026-10-28），[发布说明](https://pytorch.org/blog/pytorch-2-15-release-blog/)。重写 NCCL backend 成为 `nccl` process group 默认实现，带 hang 后 flight recorder；FSDP2 和 pipeline 的部分内存/通信控制有所更新。 | 发布说明中的 API 和运行能力属于 2.15 发布范围，不能推断旧版、非 NCCL backend 或其他 vendor runtime 都有相同功能。 |
| 在线改变并行布局 | **A**：DynaTrain, arXiv:2605.18815 v1, 2026-05-12，[预印本](https://arxiv.org/abs/2605.18815)。作者在其系统中报告 70B dense 模型重配置低于 2 s、235B MoE 案例 4.36 s，并提出 Virtual Parameter Space 与状态路由机制。 | 这是预印本在作者集群和特定配置下的结果，不是 PyTorch DCP 能力、标准 checkpoint 行为或已在多个系统复现的通用延迟。 |

## 激活重算的分布式边界

**A**：Korthikanti et al., arXiv:2205.05198，[论文](https://arxiv.org/abs/2205.05198)，研究 sequence parallel 与 selective activation recomputation 的组合，并报告模型规模、TP、序列长度和 A100 配置下的内存及 FLOPs 结果。本章第7章只讨论重算触发的 FSDP 参数聚集、TP/CP 通信和 PP stage 负载交互；激活重算机制、分段估算和性能数字由第8章负责，以避免跨章重复。

**A-**：PyTorch FSDP2 文档支持参数聚集与重分片接口语义；PyTorch 2.15 DCP 文档支持分布式状态保存与加载边界。框架文档不能给出目标运行时上的真实 overlap、peak memory 或 tokens/s，须由目标配置测量。
