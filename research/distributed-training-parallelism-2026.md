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

## 当前框架实现与异步 RL 系统

- PyTorch contributors, “Graph-based CPU Offloading for TorchTitan Frontier Model Training,” 2026-06-23，[PyTorch DevLog](https://docs.pytorch.org/devlogs/distributed/2026-06-23-cpu-offloading/)。
  - 实际阅读范围：全文，重点包括 traced-graph offload pass、view replay、pinned-memory pool、NUMA、budget/defer/prefetch 定义及各模型配置结果。
  - 来源支持：该 pass 根据 traced graph 决定激活卸载对象、copy-back 时机和 view 重建；所述 8×H100 96 GB 实验中，Llama3 8B、batch=1、sequence=8192、defer=8 的 10–12 GB budget 区间以约 60–70 TPS 下降换得 0.46 GiB 额外峰值显存；DSv3/MoE 的 defer≥3 在该配置中出现 24% 以上开销或 NCCL timeout，而 dense 案例可用 defer=8。
  - 证据等级：source-backed（PyTorch 项目工程记录和指定模型/硬件实验）。
  - 适用边界：所有吞吐、显存和 defer 结果绑定该 DevLog 的模型、batch、序列长度、H100/PCIe/NUMA 拓扑及训练实现。10–12 GB 的局部斜率不能外推为普遍线性关系；defer 值也不是所有 MoE 或 dense 训练的默认建议。

- PyTorch Foundation, “PyTorch 2.13 Release Blog,” 2026-07-08，[官方发布说明](https://pytorch.org/blog/pytorch-2-13-release-blog/)。
  - 实际阅读范围：§ Distributed Training 中 torchcomms Backend 和 FSDP2 Separate Reduce-Scatter Group。
  - 来源支持：发布说明把 `torchcomms` 描述为 PyTorch Distributed 的新通信后端，并列出 graceful timeout、partial-group recovery、structured logging 和 collective tracing；同页将相关 API 标注为 API Unstable。说明也描述 FSDP2 使用独立 reduce-scatter communicator 来与 all-gather 重叠。
  - 证据等级：source-backed（PyTorch 2.13 官方发布说明）。
  - 适用边界：通信后端的恢复与 tracing 不等于训练 optimizer、数据游标和 checkpoint 状态已经一致，也不证明整项训练能自动续跑；发布说明中的 unstable API 需按具体构建和版本复核。

- NVIDIA, “Megatron Core Parallelism Strategies Guide,” [`latest` 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)，访问日期 2026-10-29。
  - 实际阅读范围：并行策略总览、Expert Parallelism 与组合配置部分。
  - 来源支持：Megatron Core 将 EP 定义为把 MoE experts 分布到多个 GPU；其公开配置示例列出 EP 与 TP/PP/CP/DP 的组合，并指出在该实现中 TP 与 EP 同时使用时需要启用 Sequence Parallelism。
  - 证据等级：source-backed（NVIDIA 当前公开框架指南）。
  - 适用边界：TP+EP 的配置约束属于 Megatron Core，不能推广到所有框架；指南中的推荐规模不是目标集群性能保证。`latest` 页面会更新，正文机制解释不依赖某个未固定的性能测量。

- PyTorch contributors, “SPMD types in TorchTitan,” 2026-08-26，[PyTorch DevLog](https://docs.pytorch.org/devlogs/distributed/2026-08-26-spmd-types-in-torchtitan/)。
  - 实际阅读范围：设计动机、显式 collectives、forward/backward 类型关系、local SPMD、mesh reinterpretation、TorchTitan 集成、性能对照和 DTensor/FSDP2 状态说明。
  - 来源支持：该记录称 TorchTitan 已将 `spmd_types` 设为 eager 预训练和 RL 的默认计算后端；开发阶段可用类型检查验证分片合同，运行时可擦除检查、以普通 tensor 执行显式 collective；FSDP2 仍用 DTensor 表示持久化分片状态。MoE EP 示例在稀疏 mesh 内用 all-to-all dispatch 与返回 token。其 debug-model 对照在 8×H100、LBS 8、context length 2048 下报告 eager 最高约 46% 吞吐差异，Inductor 约 5–9%，部分 CUDA Graph 对照的置信区间包含零，峰值内存未见实质变化。
  - 证据等级：source-backed（PyTorch 项目工程记录及其指定调试基准）。
  - 适用边界：这是 TorchTitan 在 2026-08 的项目状态与特定 debug-model 测量，不等于 PyTorch 所有训练栈已采用该后端，也不证明任意模型、编译模式或硬件都会提速。

- Hu et al., “DORA: A Scalable Asynchronous Reinforcement Learning System for Language Model Training,” arXiv:2604.26256v2，2026-07-20，[论文 v2](https://arxiv.org/abs/2604.26256v2)。
  - 实际阅读范围：摘要、§3.2 多版本 streaming training、§3.3 资源编排、§4.1–4.3 实验与消融、§5 限制，以及附录 C 的 staleness/长尾案例。
  - 来源支持：每条轨迹由派发时指定的单个 policy 版本完整生成；滑动窗口给出可配置的最大陈旧度；64-GPU 实验的 rollout-only 阶段由 14.9 分钟降到 1.8 分钟（约 8.2×），128-GPU 实验的端到端吞吐相对同步基线最高约 2.12×。论文还报告生产规模的额外结果，但正文不把它们与开源设置混成同一对照。
  - 证据等级：source-backed（预印本 v2 的方法、实验和作者报告的结果）。
  - 适用边界：论文实验使用受控内部框架，64 与 128 GPU 的阶段/端到端数字不是同一个运行；效果取决于模型、任务、长尾分布、基线与 staleness 窗口，不能作为异步训练的一般加速承诺。

- Zhang et al., “Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale,” arXiv:2604.11554v2，2026-04-14，[论文 v2](https://arxiv.org/abs/2604.11554v2)。
  - 实际阅读范围：§3.2 异步执行与 staleness 控制、§4 多模态 Agent RL 设计、§5.2–5.5 收敛/性能实验、附录 B 的补充实验。
  - 来源支持：Qwen3-4B / DAPO-MATH-17k、16×H800 对照中，on-policy Relax 相对 veRL 报告 1.20×端到端 speedup；另一组同一任务的 async off-policy（staleness=1）相对 colocate 报告 1.76×。Qwen3-Omni-30B 的 Echo Ink 音频—图像任务在 16×H20、staleness=2 下，相对 colocate 报告 2.00×，对应实验最终 reward 约 0.93。
  - 证据等级：source-backed（预印本 v2 的系统设计和指定模型/任务/集群实验）。
  - 适用边界：1.20×、1.76× 和 2.00×使用不同对照、训练模式、模型或任务，不能拼为同一 benchmark 排名；收敛结果仅支持论文所测数据与训练步数，不证明所有多模态或 Agent RL 设置均保持相同质量。

## 生命周期

数据并行、状态分片、张量/流水线并行及激活重计算机制：active。具体 API、状态字典格式、精度、网络性能、并行拓扑和恢复语义均依框架版本及运行配置而定。
