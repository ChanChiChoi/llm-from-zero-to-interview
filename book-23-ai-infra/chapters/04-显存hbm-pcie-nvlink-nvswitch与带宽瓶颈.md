# 第 4 章 显存、HBM、PCIe、NVLink、NVSwitch 与带宽瓶颈

上一章讲了 GPU、NPU、TPU 与 AI 加速器基础。很多同学理解加速器时，第一反应是“算力”。但在大模型训练和推理中，真正限制性能的经常不是算力，而是显存容量、显存带宽和设备之间的数据传输。

这章我们讲硬件里最容易被低估的一组概念：显存、HBM、PCIe、NVLink、NVSwitch 和带宽瓶颈。

先记住一句话：

> 大模型性能不只取决于能算多快，还取决于数据能不能及时送到计算单元，以及多张卡之间能不能高效交换数据。

## 4.0 本讲资料边界与第二轮精修口径

本章讨论显存、HBM、PCIe、NVLink、NVSwitch 和多机网络的稳定工程概念，不绑定某个 GPU 型号、服务器 SKU、云厂商实例名或某一代互联数字。

第二轮精修时，资料边界按官方公开材料校准：NVIDIA Hopper 架构资料说明高端 GPU 会把 HBM、PCIe、NVLink 和 NVSwitch 放在同一套加速器系统设计里；NCCL 文档把 GPU 间 collective 和 point-to-point 通信抽象成拓扑感知通信库；GPUDirect RDMA 文档强调 GPU 与网卡、存储或其他 PCIe peer 直接交换数据仍依赖硬件、驱动、拓扑和权限配置；PCI-SIG 规格库只作为 PCIe 是标准化互联规范的依据，不把某一代 PCIe 的峰值带宽写成所有机器的实际带宽。

因此，本章只抽象四类面试稳定结论：

1. HBM 解决 GPU 计算单元和显存之间的高带宽供给问题。
2. PCIe 解决 CPU、GPU、网卡和存储设备间的通用连接问题，但通常不是高频 GPU 张量通信的最优路径。
3. NVLink / NVSwitch 解决单机多 GPU 之间更高带宽、更均匀拓扑的通信问题。
4. 多机训练还要看 InfiniBand / RoCE、网卡拓扑、NCCL 路由、存储和 checkpoint I/O。

## 4.1 为什么带宽这么重要

大模型的计算过程可以简化成两件事：

1. 从显存读取权重、activation、KV cache 等数据。
2. 在计算单元上做矩阵乘法、attention、MLP 等计算。

如果计算单元很强，但数据送不过来，计算单元就会等待。

这就像厨房里厨师很多、炉灶很多，但食材供应很慢，整体出菜速度还是上不去。

在 AI Infra 中，常见带宽包括：

1. GPU 内部计算单元和显存之间的带宽。
2. GPU 和 CPU 之间通过 PCIe 的带宽。
3. GPU 和 GPU 之间通过 NVLink 的带宽。
4. 多 GPU 之间通过 NVSwitch 的交换带宽。
5. 多机之间通过 InfiniBand 或 RoCE 的网络带宽。
6. GPU 和存储之间间接形成的数据读取带宽。

带宽瓶颈可能发生在任意一层。

## 4.2 显存是什么

显存是 GPU 上的高速内存，用来存放模型计算需要的数据。

训练时显存里通常有：

1. 模型参数。
2. 梯度。
3. 优化器状态。
4. activation。
5. 临时 buffer。
6. 通信 buffer。

推理时显存里通常有：

1. 模型权重。
2. KV cache。
3. 当前 batch 的中间状态。
4. runtime 需要的临时 buffer。
5. 量化或解量化相关数据。

显存容量决定了几个关键上限：

1. 能放多大的模型。
2. 能支持多长上下文。
3. 能支持多大的 batch。
4. 能支持多少并发请求。
5. 训练时是否需要使用 ZeRO、offload、activation checkpointing 等技术。

显存不够不是小问题，很多时候它会直接决定系统方案。

## 4.3 训练时显存为什么消耗大

训练比推理更吃显存，因为训练不只需要模型权重。

以常见 Adam 优化器为例，训练时需要：

1. 参数本身。
2. 参数梯度。
3. Adam 的一阶动量。
4. Adam 的二阶动量。
5. forward 中间 activation。
6. backward 临时张量。

如果使用混合精度，还可能有 FP16/BF16 参数和 FP32 master weight。

因此，一个模型推理能放进一张卡，不代表训练也能放进一张卡。

粗略理解：

```text
推理显存：权重 + KV cache + runtime buffer
训练显存：权重 + 梯度 + 优化器状态 + activation + runtime buffer
```

所以训练平台经常需要分布式策略，例如数据并行、张量并行、pipeline 并行、ZeRO、FSDP 和 checkpointing。

## 4.4 推理时 KV cache 为什么重要

推理阶段尤其要关注 KV cache。

Transformer 自回归生成时，每生成一个新 token，都需要利用前面 token 的 Key 和 Value。为了避免重复计算，系统会把历史 token 的 K/V 存下来，这就是 KV cache。

KV cache 的大小和几个因素有关：

1. batch size。
2. sequence length。
3. layer 数量。
4. hidden size。
5. attention head 数量。
6. KV head 数量。
7. 数据类型，例如 FP16、BF16、INT8。

长上下文和高并发会快速放大 KV cache。

这也是为什么推理平台里，经常出现“模型权重放得下，但并发上不去”。瓶颈可能不是权重，而是 KV cache 占满显存。

## 4.5 HBM 是什么

HBM 是 High Bandwidth Memory，高带宽内存。

现代高端 AI GPU 通常使用 HBM，而不是普通内存。

HBM 的特点：

1. 带宽非常高。
2. 离计算芯片很近。
3. 通过 3D 堆叠提高数据传输能力。
4. 成本高。
5. 容量仍然有限。

为什么 HBM 重要？

因为大模型不仅需要算，还需要不断读写权重、activation、KV cache 和中间结果。

如果 HBM 带宽不够，计算核心就会等数据。

面试里可以这样说：

```text
HBM 的价值在于提供远高于普通内存的显存带宽，让矩阵计算和推理 decode 阶段有足够的数据供给。但 HBM 容量和成本都有限，所以显存管理和 KV cache 管理非常关键。
```

## 4.6 显存容量和显存带宽的区别

显存容量和显存带宽经常被混淆。

显存容量回答的是：能放多少数据？

显存带宽回答的是：单位时间能搬多少数据？

举例：

1. 模型权重太大放不下，这是容量问题。
2. batch 稍微变大就 OOM，这是容量问题。
3. decode 阶段 GPU 利用率不高但延迟高，可能是带宽问题。
4. 大量读取 KV cache 导致吞吐上不去，可能是带宽问题。

简单类比：

```text
容量像仓库大小。
带宽像仓库门口的出入速度。
```

仓库很大但门很窄，取货慢。门很宽但仓库很小，东西放不下。

## 4.7 PCIe 是什么

PCIe 是 CPU、GPU、网卡、存储设备之间常见的高速连接总线。

在 AI 服务器中，PCIe 常用于：

1. CPU 和 GPU 通信。
2. GPU 和网卡通信。
3. GPU 和 NVMe 存储间接通信。
4. 多设备挂载和数据传输。

PCIe 的问题是：相比 GPU 内部 HBM 带宽和 NVLink，PCIe 带宽通常低得多，延迟也更高。

因此，如果训练或推理频繁在 CPU 内存和 GPU 显存之间搬数据，就可能严重拖慢性能。

常见场景：

1. 数据加载太慢，CPU 到 GPU 拷贝阻塞。
2. offload 把参数或优化器状态放到 CPU，PCIe 成为瓶颈。
3. 多 GPU 之间只能通过 PCIe 通信，通信速度不如 NVLink。
4. GPU Direct RDMA 配置不当，网络数据绕 CPU，增加开销。

面试中要记住：PCIe 可以连接设备，但不适合把大量高频张量通信都压在上面。

## 4.8 NVLink 是什么

NVLink 是 NVIDIA 提供的 GPU 高速互联技术。

它的目标是让 GPU 和 GPU 之间以更高带宽、更低延迟交换数据。

相比 PCIe，NVLink 更适合：

1. 张量并行中的中间结果交换。
2. 多 GPU 之间参数和激活传输。
3. 高速 P2P 访问。
4. 单机多卡训练和推理。

为什么它重要？

因为大模型经常需要把一个模型切到多张 GPU 上。

例如张量并行时，一个矩阵乘法被拆到多张卡上，每一步都可能需要 AllReduce 或 AllGather。如果 GPU 间互联慢，计算会被通信拖住。

NVLink 可以显著缓解单机多卡通信瓶颈。

## 4.9 NVSwitch 是什么

NVSwitch 可以理解为 GPU 之间的高速交换芯片。

如果只有点对点 NVLink，GPU 之间的连接拓扑可能不完全均匀。有些 GPU 之间通信快，有些要绕路。

NVSwitch 的目标是让多张 GPU 之间形成更高带宽、更均匀的互联结构。

它的价值：

1. 多 GPU 之间通信更均衡。
2. 支持更大规模单机 GPU 互联。
3. 降低拓扑差异对并行策略的影响。
4. 提升模型并行和集合通信效率。

可以简单理解：

```text
NVLink 像 GPU 之间的高速公路。
NVSwitch 像高速公路交换枢纽。
```

## 4.10 单机多卡里的拓扑问题

不是所有“8 卡机器”都一样。

你需要看 GPU 之间怎么连接。

常见拓扑可能是：

1. 全部 GPU 通过 PCIe 连接。
2. 部分 GPU 之间有 NVLink。
3. 多 GPU 通过 NVSwitch 互联。
4. GPU 和网卡之间距离不同。

拓扑会影响：

1. 哪些 GPU 适合放在同一个 tensor parallel group。
2. 哪些 GPU 之间通信更快。
3. 多机通信是否能通过最近的网卡出去。
4. NCCL 选择什么通信路径。

训练慢的时候，不能只看“我有 8 张卡”，还要看这 8 张卡之间怎么连。

## 4.11 多机训练里的网络带宽

当训练扩展到多机时，瓶颈从单机内部互联扩展到机间网络。

常见网络包括：

1. InfiniBand。
2. RoCE。
3. 高速以太网。

多机训练中常见通信：

1. 数据并行的梯度 AllReduce。
2. ZeRO / FSDP 的参数 shard 同步。
3. 张量并行跨机通信。
4. pipeline 并行 stage 间 activation 传输。
5. checkpoint 保存和加载。

如果网络慢或抖动，GPU 会等待通信完成。

表现可能是：

1. GPU 利用率周期性下降。
2. step time 变长。
3. 某些 rank 明显慢。
4. NCCL timeout。
5. 训练 hang。

这就是为什么大模型集群网络设计非常重要。

## 4.12 带宽瓶颈怎么判断

判断带宽瓶颈，不能只看一个指标。

可以按层排查。

### 4.12.1 显存带宽瓶颈

可能现象：

1. GPU 算力利用率不高。
2. 显存读写利用率高。
3. decode 阶段吞吐低。
4. batch 增大后吞吐提升有限。

常见原因：

1. 权重读取频繁。
2. KV cache 读写压力大。
3. kernel 没有充分融合。
4. 算子 memory-bound。

### 4.12.2 PCIe 瓶颈

可能现象：

1. CPU 到 GPU 拷贝时间长。
2. 数据加载占 step time 比例高。
3. offload 后速度显著下降。
4. GPU 等待 host 数据。

常见原因：

1. dataloader 太慢。
2. pinned memory 没用好。
3. CPU 预处理太重。
4. 参数或优化器状态频繁 offload。

### 4.12.3 GPU 间互联瓶颈

可能现象：

1. 多卡比单卡扩展效率差。
2. tensor parallel 性能差。
3. NCCL 通信占比高。
4. 某些 GPU 之间通信明显慢。

常见原因：

1. 只走 PCIe，没有 NVLink。
2. 拓扑不均匀。
3. parallel group 划分不合理。
4. NCCL 参数或驱动配置问题。

### 4.12.4 机间网络瓶颈

可能现象：

1. 多机扩展效率低。
2. rank 间 step time 差异大。
3. AllReduce 时间长。
4. NCCL timeout 或 hang。

常见原因：

1. 网络带宽不足。
2. 网络拥塞。
3. RDMA 配置问题。
4. 跨机张量并行过重。
5. 拓扑感知调度不足。

## 4.13 如何缓解带宽瓶颈

不同瓶颈有不同手段。

### 4.13.1 缓解显存容量压力

可以使用：

1. 混合精度训练。
2. 参数分片。
3. ZeRO / FSDP。
4. activation checkpointing。
5. gradient accumulation。
6. 更小 batch 或 sequence length。
7. 量化推理。
8. KV cache 压缩或分页管理。

### 4.13.2 缓解显存带宽压力

可以使用：

1. kernel fusion。
2. FlashAttention。
3. 更高效的 attention kernel。
4. 量化降低数据读取量。
5. 改善 batch 和序列调度。
6. 减少不必要的数据拷贝。

### 4.13.3 缓解 PCIe 压力

可以使用：

1. 提前预取数据。
2. 使用 pinned memory。
3. 减少 CPU-GPU 往返。
4. 优化 dataloader。
5. 减少频繁 offload。
6. 使用 GPUDirect 相关能力。

### 4.13.4 缓解 GPU 间通信压力

可以使用：

1. 拓扑感知分组。
2. 优化 tensor parallel 和 pipeline parallel 划分。
3. 使用更高带宽互联。
4. 通信计算重叠。
5. 优化 NCCL 配置。

### 4.13.5 缓解机间网络压力

可以使用：

1. 减少跨机 tensor parallel。
2. 把通信密集 group 放在同机。
3. 使用更适合的并行策略。
4. 拓扑感知调度。
5. 隔离高优先级训练网络。
6. 监控网络拥塞和错误包。

## 4.14 面试中如何回答带宽瓶颈

如果面试官问：

```text
为什么 GPU 算力很强，但训练或推理还是慢？
```

可以这样回答：

```text
大模型性能不只由峰值算力决定，还受显存容量、显存带宽、GPU 间互联、机间网络、数据 I/O 和软件 kernel 影响。

训练时如果 activation、梯度和优化器状态占满显存，就需要分片或 checkpointing；如果多卡通信慢，AllReduce、AllGather 或 tensor parallel 会拖慢 step time；如果数据供给慢，GPU 会等 CPU 或存储。

推理时模型权重可能放得下，但 KV cache 会随着 batch 和 context length 增长，占用大量显存；decode 阶段也经常受显存带宽限制。所以排查时要分层看 HBM、PCIe、NVLink/NVSwitch、机间网络和存储 I/O。
```

## 4.15 带宽瓶颈审计指标与最小 demo

带宽题不能只说“用 NVLink 会更快”。你需要能把显存容量、HBM 带宽、PCIe 拷贝、GPU 间互联、机间网络、存储和 checkpoint 放进同一张审计表。

先定义一个带宽瓶颈审计样本：

```math
q_i=(w_i,m_i,k_i,p_i,n_i,s_i,c_i,o_i,z_i)
```

其中，`w_i` 是工作负载，`m_i` 是显存对象和容量预算，`k_i` 是 kernel 与 HBM 读写模式，`p_i` 是 PCIe / CPU-GPU 拷贝，`n_i` 是 NVLink / NVSwitch / 机间网络拓扑，`s_i` 是存储和 checkpoint I/O，`c_i` 是并行策略与通信量，`o_i` 是观测指标，`z_i` 是风险标签。

显存总预算可以写成：

```math
M_{\mathrm{vram}}=M_{\mathrm{weight}}+M_{\mathrm{grad}}+M_{\mathrm{opt}}+M_{\mathrm{act}}+M_{\mathrm{kv}}+M_{\mathrm{tmp}}+M_{\mathrm{comm}}
```

推理场景中，`M_grad` 和 `M_opt` 通常为 0，但 `M_kv` 会随着并发和上下文长度增长。训练场景中，`M_grad`、`M_opt`、`M_act` 和 `M_comm` 往往让显存压力远大于权重本身。

任意链路的有效带宽可以用传输数据量除以耗时估算：

```math
B_{\mathrm{eff}}=\frac{D}{T}
```

其中，`D` 是实际搬运的数据量，`T` 是这段搬运花费的时间。注意 `B_eff` 通常低于规格峰值，因为有协议开销、拓扑绕行、kernel 调度、并发争用和软件栈开销。

显存带宽是否限制算子，可以继续使用 roofline 直觉：

```math
F_{\mathrm{achievable}}\le \min(F_{\mathrm{peak}}, I B_{\mathrm{hbm}})
```

其中，`I` 是 arithmetic intensity。decode、embedding、归一化、部分小 batch kernel 和频繁 KV cache 访问经常更接近 memory-bound。

通信暴露比例可以写成：

```math
R_{\mathrm{comm}}=\frac{T_{\mathrm{comm}}}{T_{\mathrm{compute}}+T_{\mathrm{comm}}+T_{\mathrm{io}}}
```

PCIe 或 offload 暴露比例可以写成：

```math
R_{\mathrm{pcie}}=\frac{T_{\mathrm{pcie}}}{T_{\mathrm{step}}}
```

多卡扩展效率可以写成：

```math
E_n=\frac{T_1}{nT_n}
```

其中，`T_1` 是单卡 step time，`T_n` 是 `n` 卡 step time。`E_n` 很低时，要检查并行策略、拓扑、通信、I/O 和 load balance，而不是只说“GPU 不够强”。

最后可以定义一个带宽门禁：

```math
G_{\mathrm{bandwidth}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{comm}}\le \rho_{\mathrm{comm}} \land R_{\mathrm{pcie}}\le \rho_{\mathrm{pcie}} \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个带宽审计维度覆盖率，`\rho_comm` 和 `\rho_pcie` 是可接受暴露比例阈值，`P_0` 是 P0 级硬阻断数量。

下面是一个 0 依赖 Python demo。它用 toy 数字估算 KV cache、HBM / PCIe 传输时间、ring all-reduce 通信时间和多卡扩展效率，再用 bad case 检查带宽审计门禁。

```python
METRICS = [
    "vram_capacity_accounting",
    "hbm_bandwidth_model",
    "pcie_transfer_awareness",
    "nvlink_topology_awareness",
    "nvswitch_all_to_all_awareness",
    "inter_node_network_awareness",
    "kv_cache_growth_accounting",
    "training_state_memory_accounting",
    "communication_volume_accounting",
    "topology_aware_parallel_group",
    "dataloader_storage_io_awareness",
    "checkpoint_io_awareness",
    "offload_penalty_awareness",
    "overlap_fusion_optimization",
    "observability_metric_coverage",
    "bandwidth_bottleneck_gate",
]


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def bytes_from_gib(num_gib):
    return num_gib * (1024 ** 3)


def transfer_seconds(num_gib, bandwidth_gbps):
    return round(bytes_from_gib(num_gib) / (bandwidth_gbps * 1_000_000_000), 3)


def kv_cache_gib(layers, batch, seq_len, kv_heads, head_dim, elem_bytes):
    total_bytes = 2 * layers * batch * seq_len * kv_heads * head_dim * elem_bytes
    return round(gib(total_bytes), 2)


def training_state_gib(params_billion, param_bytes, grad_bytes, opt_bytes):
    params = params_billion * 1_000_000_000
    return round(gib(params * (param_bytes + grad_bytes + opt_bytes)), 2)


def ring_allreduce_seconds(num_gib, bandwidth_gbps, world_size):
    traffic_multiplier = 2 * (world_size - 1) / world_size
    return round(traffic_multiplier * bytes_from_gib(num_gib) / (bandwidth_gbps * 1_000_000_000), 3)


def scale_efficiency(single_gpu_step_s, multi_gpu_step_s, world_size):
    return round(single_gpu_step_s / (world_size * multi_gpu_step_s), 3)


def make_case(name, failed_metric=None, p0=False):
    flags = {metric: True for metric in METRICS}
    if failed_metric is not None:
        flags[failed_metric] = False
    return {"name": name, "flags": flags, "p0": p0}


def build_cases():
    bad_cases = [
        ("vram_objects_missing_bad", "vram_capacity_accounting"),
        ("hbm_bandwidth_ignored_bad", "hbm_bandwidth_model"),
        ("pcie_copy_hidden_bad", "pcie_transfer_awareness"),
        ("nvlink_topology_unknown_bad", "nvlink_topology_awareness"),
        ("nvswitch_assumed_bad", "nvswitch_all_to_all_awareness"),
        ("inter_node_network_ignored_bad", "inter_node_network_awareness"),
        ("kv_cache_growth_missing_bad", "kv_cache_growth_accounting"),
        ("training_state_underestimated_bad", "training_state_memory_accounting"),
        ("allreduce_volume_missing_bad", "communication_volume_accounting"),
        ("parallel_group_not_topology_aware_bad", "topology_aware_parallel_group"),
        ("dataloader_storage_unmeasured_bad", "dataloader_storage_io_awareness"),
        ("checkpoint_io_blocking_bad", "checkpoint_io_awareness"),
        ("offload_over_pcie_unbounded_bad", "offload_penalty_awareness"),
        ("no_overlap_or_fusion_plan_bad", "overlap_fusion_optimization"),
        ("metrics_missing_bad", "observability_metric_coverage"),
        ("no_bandwidth_gate_bad", "bandwidth_bottleneck_gate"),
    ]
    cases = [make_case("complete_bandwidth_plan")]
    cases.extend(make_case(name, metric, p0=True) for name, metric in bad_cases)
    return cases


def audit_bandwidth(cases, threshold=0.95):
    metrics = {}
    for metric in METRICS:
        passed = sum(1 for case in cases if case["flags"][metric])
        metrics[metric] = round(passed / len(cases), 3)

    failed_cases = [
        case["name"]
        for case in cases
        if case["p0"] or any(not case["flags"][metric] for metric in METRICS)
    ]
    failed_gates = [
        metric for metric, score in metrics.items() if score < threshold
    ]
    hard_blocker_count = sum(1 for case in cases if case["p0"])
    gate_pass = not failed_gates and hard_blocker_count == 0
    return {
        "metrics": metrics,
        "hard_blocker_count": hard_blocker_count,
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "bandwidth_gate_pass": gate_pass,
    }


bandwidth_examples = {
    "weights_70b_bf16_gib": round(gib(70_000_000_000 * 2), 2),
    "train_state_70b_adamw_gib": training_state_gib(70, 2, 2, 8),
    "kv_cache_8x8192_gib": kv_cache_gib(80, 8, 8192, 8, 128, 2),
    "hbm_read_140gib_at_3000gbps_s": transfer_seconds(140, 3000),
    "pcie_copy_16gib_at_64gbps_s": transfer_seconds(16, 64),
    "nvlink_ring_allreduce_8gib_at_900gbps_s": ring_allreduce_seconds(8, 900, 8),
    "pcie_ring_allreduce_8gib_at_64gbps_s": ring_allreduce_seconds(8, 64, 8),
    "scale_efficiency_8gpu": scale_efficiency(8.0, 1.4, 8),
}

cases = build_cases()
report = audit_bandwidth(cases)
smoke = {
    "complete_case_passes": "complete_bandwidth_plan" not in report["failed_cases"],
    "caught_pcie_copy": "pcie_copy_hidden_bad" in report["failed_cases"],
    "caught_topology_gap": "parallel_group_not_topology_aware_bad" in report["failed_cases"],
    "caught_kv_growth": "kv_cache_growth_missing_bad" in report["failed_cases"],
    "caught_offload_penalty": "offload_over_pcie_unbounded_bad" in report["failed_cases"],
}

print("bandwidth_examples=", bandwidth_examples)
print("smoke=", smoke)
print("metrics=", report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("bandwidth_gate_pass=", report["bandwidth_gate_pass"])
```

这段 demo 故意让 16 个 bad case 分别打穿 16 个审计维度，因此每个维度覆盖率都是 `16/17=0.941`。`bandwidth_examples` 也展示了三个面试直觉：同样是 8 GiB 通信量，走高带宽 GPU 互联和走 PCIe 的耗时量级不同；70B 训练状态显存远大于权重本身；长上下文推理的 KV cache 会把“权重能放下”这个判断变得不够。

## 4.16 常见误区

误区一：显存越大，速度一定越快。

显存容量大能放更多数据，但速度还取决于显存带宽、计算能力和通信效率。

误区二：GPU 利用率低一定是模型代码写得差。

也可能是数据加载慢、PCIe 拷贝慢、通信慢、checkpoint 阻塞或调度问题。

误区三：单机 8 卡都一样。

不同机器的 PCIe、NVLink、NVSwitch 拓扑可能不同，通信性能差异很大。

误区四：推理只要模型权重能放下就行。

长上下文和高并发下，KV cache 可能比你想象中更快占满显存。

误区五：offload 总能解决显存问题。

Offload 可以降低显存占用，但会增加 CPU-GPU 或存储-GPU 数据传输，可能让 PCIe 或 I/O 成为瓶颈。

## 4.17 面试题

### 题 1：显存容量和显存带宽有什么区别？

答：显存容量表示能存多少数据，例如模型权重、activation、KV cache。显存带宽表示单位时间能读写多少数据。容量不足会导致 OOM 或 batch/context 受限，带宽不足会导致计算单元等待数据，吞吐和延迟变差。

### 题 2：为什么训练比推理更吃显存？

答：训练除了模型权重，还要存梯度、优化器状态、activation 和 backward 临时张量。使用 Adam 时还需要一阶和二阶动量。推理通常主要存权重、KV cache 和 runtime buffer，所以同一模型能推理不代表能训练。

### 题 3：KV cache 为什么会限制推理并发？

答：KV cache 随 batch size、sequence length、layer 数、hidden size 和数据类型增长。长上下文和高并发会占用大量显存。即使模型权重能放下，KV cache 也可能让并发受限或触发 OOM。

### 题 4：PCIe、NVLink、NVSwitch 分别解决什么问题？

答：PCIe 是 CPU、GPU、网卡和存储设备之间的通用高速连接。NVLink 是 GPU 之间的高速点对点互联，带宽和延迟通常优于 PCIe。NVSwitch 是 GPU 之间的交换互联，让多 GPU 通信更均匀、更高带宽，适合大规模单机多卡通信。

### 题 5：如何排查多机训练扩展效率差？

答：先看 step time 分解，确认计算、通信、数据加载、checkpoint 各占多少。再看 NCCL 通信时间、rank 间差异、网络带宽和错误、GPU 利用率、拓扑分布和并行策略。常见原因包括网络带宽不足、拓扑不合理、跨机 tensor parallel 过重、数据 I/O 慢和通信计算没有重叠。

## 4.18 小练习

练习一：估算推理显存占用。

要求：列出模型权重、KV cache、batch 中间状态和 runtime buffer，并说明哪些因素会随 context length 增长。

练习二：分析一个训练慢的问题。

假设 8 卡训练 GPU 利用率只有 45%，请从 HBM、PCIe、NVLink、数据加载、NCCL 和 checkpoint 角度列出排查路径。

练习三：比较两种服务器拓扑。

假设一台机器 8 卡只通过 PCIe 互联，另一台机器 8 卡通过 NVSwitch 互联。说明它们在 tensor parallel 和数据并行下的差异。

练习四：设计一个带宽监控面板。

要求：包含 GPU 显存读写、PCIe 传输、GPU 间通信、机间网络、存储读取和 checkpoint 写入指标。

## 4.19 本章小结

本章讲了显存、HBM、PCIe、NVLink、NVSwitch 与带宽瓶颈。

你需要掌握：

1. 大模型性能不只取决于峰值算力，还取决于数据搬运能力。
2. 显存容量决定能放多大模型、多大 batch、多长上下文和多少并发。
3. 显存带宽决定数据能否及时供给计算单元。
4. HBM 是高带宽显存，对训练和推理都很关键。
5. PCIe 连接 CPU、GPU、网卡和存储，但高频张量通信容易受限。
6. NVLink 提供 GPU 间高速互联，NVSwitch 提供更均匀的多 GPU 交换互联。
7. 多机训练还会受到 InfiniBand、RoCE 等机间网络影响。
8. 排查性能问题要分层看显存、PCIe、GPU 间互联、机间网络和存储 I/O。

下一章我们会讲 GPU 利用率、MFU、HFU 与训练效率指标，帮助你用指标量化训练是否真的跑得好。
