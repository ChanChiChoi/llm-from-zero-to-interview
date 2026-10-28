# 分布式训练并行策略：来源证据

记录日期：2026-10-28

本文件记录分布式训练机制的来源事实、版本和证据边界，不记录书内修改过程或逐章进度。

## 数据并行与训练状态分片

- Rajbhandari et al., “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models,” arXiv:1910.02054 v3，[论文](https://arxiv.org/abs/1910.02054)，[HTML 正文](https://arxiv.org/html/1910.02054)。
  - 实际阅读范围：§3 model states，§4 ZeRO overview，§5 optimizer/gradient/parameter partitioning，§7 communication analysis。
  - 来源支持：ZeRO-DP 按阶段减少数据并行 rank 之间对 optimizer state、gradient 和 parameter 的冗余；不同阶段对通信、临时状态和峰值内存有不同要求。
  - 证据等级：source-backed（论文定义和实验实现）。
  - 适用边界：论文测量的内存与通信数值限于论文中的模型、精度、系统和并行配置；分阶段概念不能直接决定当前 DeepSpeed 或其他框架的所有实现细节。

- DeepSpeed, “ZeRO,” current documentation v0.19.9，[官方文档](https://deepspeed.readthedocs.io/en/latest/zero3.html)，访问日期 2026-10-28。
  - 实际阅读范围：stage 1–3 描述与 ZeRO-3 参数 gather/partition 说明。
  - 来源支持：Stage 1 切 optimizer states，Stage 2 再切 reduced gradients，Stage 3 再切 model parameters；ZeRO-3 在前向/反向计算期间按需收集和重新切分参数。
  - 证据等级：source-backed（DeepSpeed 当前公开接口文档）。
  - 适用边界：该描述是 DeepSpeed 当前版本接口语义；offload、通信调优和具体 checkpoint 语义需按配置及版本进一步检查。

- PyTorch, “DistributedDataParallel,” v2.14，[官方 API 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)。
  - 实际阅读范围：行为说明、all-reduce、同步点及 gradient accumulation 例子。
  - 来源支持：DDP 在 backward 中同步梯度；优化器由各 rank 在本地执行，文档明确指出参数本身并非每次迭代都在 rank 间广播。相同模型状态、梯度和更新过程是各副本保持一致的前提。
  - 证据等级：source-backed（PyTorch v2.14 API 行为）。
  - 适用边界：梯度求和/平均和最终更新还受 loss reduction、local batch、通信 hook、未使用参数和训练代码影响；“每 rank 一张 GPU”是常见部署方式，不是 DDP 唯一合法拓扑。

- PyTorch, “FullyShardedDataParallel,” v2.14，[FSDP API 文档](https://docs.pytorch.org/docs/2.14/fsdp.html)；“Getting Started with Fully Sharded Data Parallel (FSDP2),” v2.14，[官方教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)；[FSDP2 `fully_shard` API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)。
  - 实际阅读范围：FSDP1/FSDP2 的 API 说明、参数分片、all-gather/reduce-scatter、reshard 策略和分层预取。
  - 来源支持：FSDP2 对参数作分片，计算某模块前通过 all-gather 组成该模块所需参数，反向阶段同步分片梯度；参数是否在 forward 后重新分片可由策略决定，all-gather 可与当前模块计算重叠。PyTorch 文档明确提供 FSDP1 到 FSDP2 的迁移说明，两者调用接口并不相同。
  - 证据等级：source-backed（PyTorch v2.14 API 与教程）。
  - 适用边界：wrap 粒度、reshard、混合精度、设备 mesh、预取和 checkpoint strategy 会改变峰值显存与通信路径；不能把简化 ZeRO-3 估算当作任何 FSDP 配置的精确峰值。

## 张量并行、流水线并行与 activation checkpointing

- Shoeybi et al., “Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism,” arXiv:1909.08053 v4，[论文](https://arxiv.org/abs/1909.08053)，[HTML 正文](https://arxiv.org/html/1909.08053)。
  - 实际阅读范围：§2 data/model parallelism，§3 intra-layer Transformer parallelism，§5 scaling analysis。
  - 来源支持：张量并行把 Transformer 层内的矩阵运算按列/行分区；分区输出布局决定何处需要 reduction 或 gather。论文将该机制与 pipeline model parallelism 视作可组合的维度，并在其 GPT/BERT 与 GPU 配置上报告吞吐结果。
  - 证据等级：source-backed（论文机制与指定硬件/模型实验）。
  - 适用边界：通信量、并行效率和可用切分取决于层结构、设备互联、实现和并行组；论文中的性能数值不能外推为当前系统保证。

- Huang et al., “GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism,” arXiv:1811.06965 v5，[论文](https://arxiv.org/abs/1811.06965)，[HTML 正文](https://arxiv.org/html/1811.06965)。
  - 实际阅读范围：§2 micro-batch pipeline 算法，§3 效率分析，§6 trade-offs。
  - 来源支持：GPipe 把 mini-batch 切成 micro-batches，在多个 accelerator 上流水执行不同层分段；pipeline fill/drain 带来 bubble。论文给出 bubble 开销随 stage 数和 micro-batch 数变化的量级分析，并在其模型和调度下观察到 micro-batch 数足够大时 bubble 较小。
  - 证据等级：source-backed（GPipe 定义、复杂度分析和论文实验）。
  - 适用边界：bubble 量级和吞吐取决于 schedule、stage 计算是否均衡、micro-batch 大小、激活占用和网络传输；不能把单一比例视为任意 pipeline 实现的精确 step time。

- Chen et al., “Training Deep Nets with Sublinear Memory Cost,” arXiv:1604.06174 v2，[论文](https://arxiv.org/abs/1604.06174)；PyTorch v2.14 [`torch.utils.checkpoint`](https://docs.pytorch.org/docs/2.14/checkpoint.html)。
  - 实际阅读范围：论文摘要与 PyTorch checkpointing API 的概念说明。
  - 来源支持：activation checkpointing/recomputation 不保存所有前向中间激活，在反向传播时重新计算部分结果，以额外计算换取更低激活内存。
  - 证据等级：source-backed（原始论文算法和 PyTorch 指定版本 API）。
  - 适用边界：显存和时间折中随切分策略、网络结构、计算算子、保存边界和框架实现变化；论文实验中的额外时间不能当作当前模型固定开销。

## 分布式 checkpoint

- PyTorch, “Distributed Checkpoint,” v2.14，[官方文档](https://docs.pytorch.org/docs/2.14/distributed.checkpoint.html)。
  - 实际阅读范围：distributed load/state-dict 说明及 FSDP checkpoint 示例入口。
  - 来源支持：分布式 load 以 SPMD 方式由各 rank 提供 state dict；读取分片状态时，各 rank 可仅读取满足本地 shard 所需数据；参与 load 的 rank 需要具有一致的 state-dict keys。
  - 证据等级：source-backed（PyTorch v2.14 DCP 文档）。
  - 适用边界：API 文档不能替代应用层对模型、optimizer、scheduler、RNG、数据游标、rank 数和保存/恢复拓扑的集成验证；不能只凭一个权重文件证明训练可续接。

## 生命周期

数据并行、状态分片、张量/流水线并行及激活重计算机制：active。具体 API、状态字典格式、精度、网络性能、并行拓扑和恢复语义均依框架版本及运行配置而定。
