# 第 8 章 网络基础：InfiniBand、RoCE、以太网和集合通信

上一章讲了 GPU 集群架构。本章继续往下拆：多机多卡为什么这么依赖网络？InfiniBand、RoCE、以太网有什么区别？集合通信为什么会成为大模型训练的关键瓶颈？

很多训练问题表面上是“GPU 利用率低”“训练 hang”“扩展效率差”，本质却是网络问题。对于大模型训练，网络不是普通后端服务里的辅助组件，而是训练主路径的一部分。

先记住一句话：

> 多机大模型训练的网络不是只负责传文件，而是在每个 step 中参与参数、梯度、activation 和 optimizer state 的同步。

## 8.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径按“AI 集群网络稳定概念”处理，而不是按某个交换机型号、云产品网络规格、网卡 SKU 或 NCCL 环境变量写死。InfiniBand 部分参考 IBTA 对高带宽、低延迟、RDMA fabric 的公开定义；RoCE 部分参考 RDMA over Converged Ethernet 的基本边界，以及 NVIDIA / 以太网数据中心文档中对 lossless fabric、PFC、ECN、拥塞控制和丢包敏感性的工程说明；集合通信部分参考 NVIDIA NCCL 对 AllReduce、AllGather、ReduceScatter、Broadcast、InfiniBand / RoCE / GPUDirect RDMA 和拓扑选择的边界；GPUDirect RDMA 部分延续上一章的 GPU-NIC PCIe 拓扑和直接访问 GPU 显存的边界。

需要注意三点：

1. 本章只讲网络基础和集合通信，不把某个厂商的峰值带宽、交换机端口数、NCCL 参数或云实例网络指标写成通用结论。
2. InfiniBand、RoCE 和普通以太网不是简单“谁更高级”的关系，而是性能、成本、运维复杂度、拥塞控制、团队能力和任务通信强度的取舍。
3. 集合通信瓶颈要和并行策略一起分析。数据并行、FSDP / ZeRO、张量并行和 pipeline 并行对应不同通信频率、通信量和 tail latency 敏感度。

## 8.1 为什么大模型训练需要高速网络

单机多卡可以依赖 NVLink / NVSwitch 做 GPU 间通信。多机训练时，GPU 分散在不同服务器上，就必须通过机间网络通信。

这些通信可能发生在每个训练 step，甚至每个 Transformer layer。

常见场景：

1. 数据并行同步梯度。
2. FSDP / ZeRO 拉取参数分片。
3. 张量并行交换中间结果。
4. pipeline 并行传递 activation。
5. checkpoint 写入远端存储。
6. 分布式推理交换 KV cache 或中间状态。

如果网络慢，GPU 就会等通信完成。

这会导致：

1. step time 变长。
2. GPU 利用率下降。
3. MFU 下降。
4. 多机扩展效率差。
5. 训练任务偶发 timeout。
6. 某些 rank 拖慢整体任务。

所以，大模型集群里的网络是计算系统的一部分。

## 8.2 网络指标怎么看

理解训练网络，需要关注几个指标。

### 8.2.1 带宽

带宽表示单位时间能传多少数据。

常见单位：Gbps、GB/s。

注意 bit 和 byte 的区别：

```text
8 bits = 1 byte
400 Gbps ≈ 50 GB/s 理论上限
```

实际有效带宽会低于理论值，因为有协议开销、拥塞、拓扑、通信模式和实现效率。

### 8.2.2 延迟

延迟表示一次通信从发起到收到响应需要多久。

小消息通信更敏感于延迟，大消息通信更敏感于带宽。

分布式训练中既有大数据传输，也有频繁同步点，因此带宽和延迟都重要。

### 8.2.3 抖动

抖动表示延迟或吞吐不稳定。

训练是同步过程，一个 rank 抖动可能拖慢所有 rank。

### 8.2.4 丢包和重传

网络丢包会导致重传和延迟上升。

在 RoCE 这类基于以太网的 RDMA 场景里，丢包和拥塞控制配置尤其关键。

### 8.2.5 拓扑和拥塞

即使单链路带宽很高，如果多任务共享同一交换路径，也可能拥塞。

训练任务的集合通信会形成同步大流量，对拓扑非常敏感。

## 8.3 RDMA 是什么

RDMA 是 Remote Direct Memory Access，远程直接内存访问。

它允许一台机器直接访问另一台机器的内存，尽量绕过远端 CPU，减少拷贝和内核协议栈开销。

RDMA 的价值：

1. 降低延迟。
2. 提高吞吐。
3. 降低 CPU 开销。
4. 提升分布式训练通信效率。

在 GPU 集群中，还会关注 GPUDirect RDMA。

GPUDirect RDMA 的目标是让网卡直接和 GPU 显存交互，减少数据在 CPU 内存中中转。

简化对比：

```text
普通路径：GPU -> CPU memory -> NIC -> network -> NIC -> CPU memory -> GPU
GPUDirect RDMA：GPU memory -> NIC -> network -> NIC -> GPU memory
```

这对大规模训练非常重要，因为减少一次额外拷贝就能节省大量时间。

## 8.4 InfiniBand 是什么

InfiniBand 是高性能计算和 AI 集群中常见的高速网络技术。

它强调：

1. 高带宽。
2. 低延迟。
3. RDMA 原生支持。
4. 拥塞控制能力。
5. 面向 HPC 和大规模训练优化。

InfiniBand 常用于大规模 GPU 集群，尤其是对通信敏感的预训练任务。

优势：

1. 性能强。
2. 延迟低。
3. 训练生态成熟。
4. 与 NCCL 等通信库结合较好。

挑战：

1. 成本高。
2. 运维要求高。
3. 设备和生态相对专用。
4. 网络规划和故障排查复杂。

面试中可以说：InfiniBand 更像为高性能计算和同步通信设计的网络，而不是普通业务网络。

## 8.5 RoCE 是什么

RoCE 是 RDMA over Converged Ethernet，也就是在以太网上跑 RDMA。

它的目标是利用以太网生态，同时获得接近 RDMA 的低延迟和高吞吐。

RoCE 的优势：

1. 可以复用以太网生态。
2. 成本和通用性可能更好。
3. 带宽可以很高。
4. 适合云厂商和企业数据中心演进。

RoCE 的挑战：

1. 对网络配置要求高。
2. 对拥塞控制敏感。
3. 丢包会严重影响性能。
4. PFC、ECN 等配置复杂。
5. 排查难度高。

RoCE 网络要尽量构建 lossless 或低丢包环境，否则 RDMA 性能会受到很大影响。

不要简单认为“以太网 + RDMA = 自动高性能”。RoCE 的工程质量非常依赖配置和运维。

## 8.6 普通以太网能不能训练大模型

普通以太网也可以训练模型，但适用范围有限。

适合：

1. 小规模训练。
2. 单机多卡为主的任务。
3. 通信不密集的 SFT。
4. 离线 batch 推理。
5. 评估任务。

不适合：

1. 大规模预训练。
2. 强依赖跨机 tensor parallel 的任务。
3. 高频同步通信。
4. 对扩展效率要求很高的任务。

普通以太网的问题：

1. 延迟较高。
2. CPU 协议栈开销大。
3. 拥塞和丢包影响明显。
4. 集合通信效率不如专用高性能网络。

所以是否能用普通以太网，要看任务画像和性能目标。

## 8.7 InfiniBand、RoCE、以太网对比

可以这样对比：

| 网络 | 优点 | 挑战 | 适合场景 |
| --- | --- | --- | --- |
| InfiniBand | 高带宽、低延迟、RDMA 成熟 | 成本高、运维复杂 | 大规模预训练、HPC、高通信任务 |
| RoCE | 复用以太网生态、支持 RDMA | 对配置和拥塞控制敏感 | 大规模云数据中心、AI 集群 |
| 普通以太网 | 通用、便宜、易运维 | 延迟高、通信效率有限 | 小规模训练、评估、普通服务 |

面试里不要说某一种一定最好，要根据规模、预算、团队能力、任务通信强度和现有数据中心条件判断。

## 8.8 集合通信是什么

集合通信是多个进程或设备之间的集体数据交换。

大模型训练中常见集合通信包括：

1. AllReduce。
2. AllGather。
3. ReduceScatter。
4. Broadcast。
5. Gather。
6. Scatter。

它们不是普通的一对一请求，而是所有 rank 共同参与的同步操作。

这意味着：

1. 最慢的 rank 会拖慢所有 rank。
2. 网络抖动会放大成 step time 抖动。
3. 拓扑不合理会显著降低效率。
4. 通信库实现非常关键。

## 8.9 AllReduce

AllReduce 的作用是：每个 rank 都有一份数据，先做 reduce 聚合，再把结果分发给所有 rank。

数据并行训练中，常见用法是同步梯度。

例如 4 张卡各自算出梯度：

```text
rank0: grad0
rank1: grad1
rank2: grad2
rank3: grad3
```

AllReduce 后，每张卡都得到：

```text
grad = grad0 + grad1 + grad2 + grad3
```

再除以 world size，就得到平均梯度。

AllReduce 是数据并行扩展效率的关键。

如果 AllReduce 慢，GPU 就会等梯度同步。

## 8.10 AllGather 和 ReduceScatter

AllGather 的作用是：每个 rank 持有一部分数据，通信后所有 rank 都获得完整数据。

ReduceScatter 的作用是：先 reduce，再把结果分片分给不同 rank。

FSDP / ZeRO 中经常使用这两类通信。

例如参数被分片存储在不同 GPU 上，某一层计算前可能需要 AllGather 拉齐参数。反向传播后，梯度可能通过 ReduceScatter 分散回各个 rank。

这能降低显存占用，但会增加通信。

这就是典型 trade-off：

```text
用通信换显存。
```

## 8.11 NCCL 是什么

NCCL 是 NVIDIA Collective Communications Library，常用于多 GPU 和多机训练的集合通信。

它负责高效实现 AllReduce、AllGather、Broadcast 等操作。

NCCL 会根据硬件拓扑选择通信路径，例如：

1. 单机内走 NVLink / NVSwitch。
2. 机间走 InfiniBand / RoCE。
3. 根据拓扑构建 ring 或 tree。
4. 尽量优化带宽和延迟。

训练中出现 NCCL timeout、NCCL hang、通信占比高，是非常常见的问题。

面试中需要知道 NCCL 是分布式训练通信栈的关键组件。

## 8.12 Ring 和 Tree 通信

集合通信常见实现包括 ring 和 tree。

Ring AllReduce 可以理解为所有 rank 组成一个环，数据沿环传递和归约。

优点：

1. 带宽利用好。
2. 适合大消息。
3. 实现成熟。

缺点：

1. 延迟可能随 rank 数增加。
2. 对某个慢 rank 敏感。

Tree 通信像树结构汇聚和广播。

优点：

1. 延迟较低。
2. 适合某些小消息或大规模场景。

缺点：

1. 带宽利用和拓扑匹配需要优化。
2. 某些节点可能成为瓶颈。

实际通信库会根据消息大小、拓扑和硬件选择不同算法。

## 8.13 通信和并行策略的关系

不同并行策略对应不同通信压力。

数据并行：

1. 每个 step 同步梯度。
2. 通信频率相对低。
3. 通信量和参数量相关。

张量并行：

1. 每层可能通信。
2. 延迟敏感。
3. 更适合同机高速互联。

Pipeline 并行：

1. stage 间传 activation。
2. 有 pipeline bubble。
3. 需要平衡各 stage 负载。

FSDP / ZeRO：

1. 参数、梯度、优化器状态分片。
2. 降低显存占用。
3. 增加 AllGather 和 ReduceScatter。

所以选并行策略时，必须结合网络拓扑。

## 8.14 网络瓶颈的表现

网络瓶颈可能表现为：

1. 多机扩展效率低。
2. GPU 利用率周期性下降。
3. step time 抖动。
4. NCCL timeout。
5. 训练 hang。
6. 某些 rank 明显慢。
7. AllReduce 时间变长。
8. 网络端口错误计数增加。
9. 重传增加。
10. p99 通信延迟升高。

不要只看平均带宽。同步训练对尾延迟非常敏感。

## 8.15 网络问题排查路径

排查网络问题可以按层走。

第一层：训练指标。

1. step time 是否变长。
2. 通信占比是否升高。
3. 哪些 rank 慢。
4. 是否有 NCCL timeout。

第二层：通信库。

1. NCCL 日志。
2. 通信算法选择。
3. 环或树构建是否合理。
4. 是否走了预期网卡。
5. 是否启用 GPUDirect RDMA。

第三层：节点和网卡。

1. 网卡带宽。
2. 错误包。
3. 重传。
4. PCIe 带宽。
5. GPU 到网卡亲和性。

第四层：交换机和拓扑。

1. 链路拥塞。
2. oversubscription。
3. 交换机端口错误。
4. 跨机柜路径。
5. 多任务流量干扰。

第五层：调度。

1. 任务是否被分配得太分散。
2. 通信密集 group 是否跨机柜。
3. 是否和其他大任务共享瓶颈链路。

网络排查需要训练平台、集群网络和调度系统共同配合。

## 8.16 如何优化通信

常见优化方向：

1. 使用更高性能网络，例如 InfiniBand 或高质量 RoCE。
2. 启用 GPUDirect RDMA。
3. 优化 NCCL 参数。
4. 拓扑感知调度。
5. 把 tensor parallel 放在同机。
6. 减少跨机通信密集操作。
7. 通信计算重叠。
8. 梯度累积，降低同步频率。
9. 压缩通信数据。
10. 避免多个大任务争抢同一链路。

这些优化不是越多越好，要结合任务画像。

例如梯度累积可以减少同步频率，但会改变有效 batch size 和训练动态，需要算法侧一起评估。

## 8.17 网络与调度的关系

网络优化不只是网络团队的事，调度也很关键。

调度系统如果随机分配节点，可能把一个训练任务分散到网络距离很远的位置。

结果是：

1. 通信路径长。
2. 跨交换机流量多。
3. 和其他任务互相干扰。
4. 扩展效率下降。

拓扑感知调度可以让任务优先使用：

1. 同机 GPU。
2. 同机柜节点。
3. 同一网络 fabric 下的节点。
4. 拥塞较低的路径。
5. GPU 到网卡亲和性更好的组合。

这就是为什么 AI Infra 里，调度系统必须理解硬件拓扑。

## 8.18 网络通信审计指标与最小 demo

前面讲的是网络概念。真实排查里，还要把“网络好不好”变成可检查指标，否则很容易只盯着一个 400 Gbps 或 800 Gbps 的纸面带宽。

可以把第 `i` 个网络通信样本写成：

```math
n_i=(b_i,\ell_i,j_i,r_i,g_i,f_i,c_i,a_i,t_i,e_i,s_i,z_i)
```

其中，`b_i` 是 bandwidth，`\ell_i` 是 latency，`j_i` 是 jitter，`r_i` 是 RDMA / GPUDirect RDMA 状态，`g_i` 是 collective group，`f_i` 是 fabric 类型，`c_i` 是 congestion / lossless 配置，`a_i` 是通信算法，`t_i` 是拓扑路径，`e_i` 是错误和重传，`s_i` 是调度放置，`z_i` 是风险。

带宽单位先要说清楚：

```math
B_{\mathrm{GiB/s}}=\frac{B_{\mathrm{Gbps}}\cdot 10^9}{8\cdot 2^{30}}
```

其中，`Gbps` 是十进制 bit/s，`GiB/s` 是二进制 GiB/s。面试和排障中经常有人把 bit 和 byte 混用，导致估算差 8 倍。

大消息传输的有效耗时可以写成：

```math
T_{\mathrm{xfer}}=\frac{D}{\eta B_{\mathrm{line}}}
```

其中，`D` 是传输数据量，`B_line` 是链路理论带宽，`\eta` 是有效带宽系数。协议开销、拥塞、拓扑、通信库和消息大小都会让 `eta<1`。

通信耗时可以粗略拆成：

```math
T_{\mathrm{msg}}\approx \alpha H+\frac{D}{B_{\mathrm{eff}}}+T_{\mathrm{queue}}+T_{\mathrm{retx}}
```

其中，`\alpha` 是单跳或启动延迟，`H` 是路径跳数，`D` 是数据量，`B_eff` 是有效带宽，`T_queue` 是拥塞排队，`T_retx` 是丢包重传带来的额外时间。

Ring AllReduce 的每个 rank 近似通信量可以写成：

```math
V_{\mathrm{ring}}\approx 2\frac{P-1}{P}D
```

其中，`P` 是 rank 数，`D` 是每个 rank 参与归约的数据大小。这个公式帮助你估算“参数越大、rank 越多、链路越慢，AllReduce 越贵”。

AllGather 和 ReduceScatter 的每个 rank 近似通信量可以写成：

```math
V_{\mathrm{gather}}\approx \frac{P-1}{P}D
```

其中，`D` 是 gather 后完整张量大小。FSDP / ZeRO 省显存，但会把这类通信放进更多 layer 或 micro-step。

同步训练对 tail latency 很敏感，可以用 jitter ratio 和通信占比辅助判断：

```math
J_{\mathrm{lat}}=\frac{L_{\mathrm{p99}}}{L_{\mathrm{p50}}}
```

```math
R_{\mathrm{comm}}=\frac{T_{\mathrm{comm}}}{T_{\mathrm{step}}}
```

丢包或重传风险可以写成：

```math
R_{\mathrm{retx}}=\frac{N_{\mathrm{retx}}}{N_{\mathrm{packet}}}
```

最后，可以把网络通信门禁写成：

```math
G_{\mathrm{net}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{comm}}\le \rho_{\mathrm{comm}} \land J_{\mathrm{lat}}\le \rho_{\mathrm{jitter}} \land R_{\mathrm{retx}}\le \rho_{\mathrm{retx}} \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个网络审计指标覆盖率，`R_comm` 是通信占比，`J_lat` 是延迟抖动比例，`R_retx` 是重传比例，`P_0` 是 P0 级风险数量。

下面这个 0 依赖 demo 演示如何把网络、RDMA、collective、NCCL、rank skew 和调度放置写成审计规则。它故意构造 1 个完整样本和 16 个坏样本，让每个关键维度各失败一次。

```python
import copy


METRICS = [
    "bandwidth_unit_accounting",
    "latency_jitter_tracking",
    "rdma_capability_fit",
    "gpudirect_rdma_path",
    "infiniband_fabric_readiness",
    "roce_congestion_losslessness",
    "ethernet_fallback_scope",
    "collective_operation_modeling",
    "nccl_topology_runtime_fit",
    "allreduce_cost_estimation",
    "allgather_reducescatter_cost",
    "rank_straggler_detection",
    "packet_error_retransmit_tracking",
    "topology_congestion_awareness",
    "scheduler_network_locality",
    "network_communication_gate",
]


def gbps_to_gib_s(gbps):
    return gbps * 1_000_000_000 / 8 / (1024 ** 3)


def transfer_time_s(gib, gbps, efficiency):
    bytes_total = gib * (1024 ** 3)
    bytes_per_s = gbps * 1_000_000_000 / 8 * efficiency
    return bytes_total / bytes_per_s


def ring_allreduce_time_s(gib, ranks, gbps, efficiency):
    traffic_gib = 2 * (ranks - 1) / ranks * gib
    return transfer_time_s(traffic_gib, gbps, efficiency)


def allgather_time_s(gib_after_gather, ranks, gbps, efficiency):
    traffic_gib = (ranks - 1) / ranks * gib_after_gather
    return transfer_time_s(traffic_gib, gbps, efficiency)


def jitter_ratio(p99_us, p50_us):
    return p99_us / p50_us


def retransmit_rate(retransmits, packets):
    return retransmits / packets


def build_network_cases():
    complete = {
        "name": "complete",
        "bandwidth": {"gbps": 400, "unit_checked": True, "effective_efficiency": 0.8},
        "latency": {"p50_us": 8, "p99_us": 18, "jitter_tracked": True},
        "rdma": {"enabled": True, "validated": True},
        "gdr": {"enabled": True, "gpu_nic_affinity": True},
        "infiniband": {"used_for_pretraining": True, "subnet_health": True, "link_width_ok": True},
        "roce": {"pfc": True, "ecn": True, "cc": True, "loss_rate": 0.0},
        "ethernet": {"scope": "eval_and_small_sft", "large_pretraining": False},
        "collectives": {"ops": ["allreduce", "allgather", "reducescatter"], "message_sizes_known": True},
        "nccl": {"topology_logged": True, "expected_nics": True, "gdr_level_checked": True},
        "allreduce": {"modeled": True, "payload_gib": 8, "ranks": 64},
        "sharded_ops": {"allgather_modeled": True, "reducescatter_modeled": True},
        "rank_times_s": [3.0, 3.05, 3.1, 3.02],
        "packets": {"errors": 0, "retransmits": 5, "total": 1_000_000, "tracked": True},
        "topology": {"rack_local": True, "oversubscription_known": True, "congestion_hotspots": []},
        "scheduler": {"network_locality": True, "tp_within_fast_domain": True},
        "gate": {"enabled": True},
    }

    def bad_case(name, mutator):
        case = copy.deepcopy(complete)
        case["name"] = name
        mutator(case)
        return case

    bad_cases = [
        bad_case("bandwidth_unit_confused_bad", lambda c: c["bandwidth"].update({"unit_checked": False})),
        bad_case("latency_jitter_ignored_bad", lambda c: c["latency"].update({"p99_us": 120, "jitter_tracked": False})),
        bad_case("rdma_missing_bad", lambda c: c["rdma"].update({"enabled": False})),
        bad_case("gdr_path_missing_bad", lambda c: c["gdr"].update({"enabled": False})),
        bad_case("ib_fabric_unverified_bad", lambda c: c["infiniband"].update({"subnet_health": False})),
        bad_case("roce_no_pfc_ecn_bad", lambda c: c["roce"].update({"pfc": False, "ecn": False, "loss_rate": 0.02})),
        bad_case("ethernet_used_for_pretraining_bad", lambda c: c["ethernet"].update({"large_pretraining": True})),
        bad_case("collective_unmodeled_bad", lambda c: c["collectives"].update({"ops": []})),
        bad_case("nccl_topology_unknown_bad", lambda c: c["nccl"].update({"topology_logged": False})),
        bad_case("allreduce_cost_missing_bad", lambda c: c["allreduce"].update({"modeled": False})),
        bad_case("allgather_reducescatter_ignored_bad", lambda c: c["sharded_ops"].update({"allgather_modeled": False})),
        bad_case("rank_straggler_hidden_bad", lambda c: c.update({"rank_times_s": [3.0, 3.1, 4.8, 3.0]})),
        bad_case("packet_errors_untracked_bad", lambda c: c["packets"].update({"tracked": False})),
        bad_case("topology_congestion_ignored_bad", lambda c: c["topology"].update({"oversubscription_known": False})),
        bad_case("scheduler_network_blind_bad", lambda c: c["scheduler"].update({"network_locality": False})),
        bad_case("network_gate_missing_bad", lambda c: c["gate"].update({"enabled": False})),
    ]
    return [complete] + bad_cases


def check_bandwidth(case):
    b = case["bandwidth"]
    return b["unit_checked"] and b["gbps"] > 0 and 0 < b["effective_efficiency"] <= 1


def check_latency(case):
    l = case["latency"]
    return l["jitter_tracked"] and jitter_ratio(l["p99_us"], l["p50_us"]) <= 4


def check_rdma(case):
    return case["rdma"]["enabled"] and case["rdma"]["validated"]


def check_gdr(case):
    return case["gdr"]["enabled"] and case["gdr"]["gpu_nic_affinity"]


def check_infiniband(case):
    ib = case["infiniband"]
    return ib["used_for_pretraining"] and ib["subnet_health"] and ib["link_width_ok"]


def check_roce(case):
    r = case["roce"]
    return r["pfc"] and r["ecn"] and r["cc"] and r["loss_rate"] <= 0.001


def check_ethernet(case):
    e = case["ethernet"]
    return e["scope"] in {"eval_and_small_sft", "batch_inference"} and not e["large_pretraining"]


def check_collectives(case):
    ops = set(case["collectives"]["ops"])
    return {"allreduce", "allgather", "reducescatter"}.issubset(ops) and case["collectives"]["message_sizes_known"]


def check_nccl(case):
    n = case["nccl"]
    return n["topology_logged"] and n["expected_nics"] and n["gdr_level_checked"]


def check_allreduce(case):
    a = case["allreduce"]
    return a["modeled"] and a["payload_gib"] > 0 and a["ranks"] > 1


def check_sharded_ops(case):
    s = case["sharded_ops"]
    return s["allgather_modeled"] and s["reducescatter_modeled"]


def check_rank_straggler(case):
    times = case["rank_times_s"]
    return max(times) / (sum(times) / len(times)) <= 1.2


def check_packets(case):
    p = case["packets"]
    return p["tracked"] and p["errors"] == 0 and retransmit_rate(p["retransmits"], p["total"]) <= 0.001


def check_topology(case):
    t = case["topology"]
    return t["rack_local"] and t["oversubscription_known"] and not t["congestion_hotspots"]


def check_scheduler(case):
    s = case["scheduler"]
    return s["network_locality"] and s["tp_within_fast_domain"]


def check_gate(case):
    return case["gate"]["enabled"]


CHECKS = {
    "bandwidth_unit_accounting": check_bandwidth,
    "latency_jitter_tracking": check_latency,
    "rdma_capability_fit": check_rdma,
    "gpudirect_rdma_path": check_gdr,
    "infiniband_fabric_readiness": check_infiniband,
    "roce_congestion_losslessness": check_roce,
    "ethernet_fallback_scope": check_ethernet,
    "collective_operation_modeling": check_collectives,
    "nccl_topology_runtime_fit": check_nccl,
    "allreduce_cost_estimation": check_allreduce,
    "allgather_reducescatter_cost": check_sharded_ops,
    "rank_straggler_detection": check_rank_straggler,
    "packet_error_retransmit_tracking": check_packets,
    "topology_congestion_awareness": check_topology,
    "scheduler_network_locality": check_scheduler,
    "network_communication_gate": check_gate,
}


def audit_network(cases):
    case_failures = {}
    for case in cases:
        failures = [name for name, check in CHECKS.items() if not check(case)]
        case_failures[case["name"]] = failures

    metrics = {}
    for name, check in CHECKS.items():
        metrics[name] = round(sum(int(check(case)) for case in cases) / len(cases), 3)

    failed_cases = [name for name, failures in case_failures.items() if failures]
    return {
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "network_gate_pass": not failed_cases and min(metrics.values()) >= 0.95,
    }


cases = build_network_cases()
case_by_name = {case["name"]: case for case in cases}
complete = case_by_name["complete"]
network_examples = {
    "gbps_400_to_gib_s": round(gbps_to_gib_s(400), 2),
    "send_16gib_at_400gbps_80eff_s": round(transfer_time_s(16, 400, 0.8), 3),
    "ring_allreduce_8gib_64ranks_200gbps_75eff_s": round(ring_allreduce_time_s(8, 64, 200, 0.75), 3),
    "allgather_8gib_64ranks_200gbps_75eff_s": round(allgather_time_s(8, 64, 200, 0.75), 3),
    "jitter_ratio": round(jitter_ratio(complete["latency"]["p99_us"], complete["latency"]["p50_us"]), 2),
    "retransmit_rate": round(retransmit_rate(complete["packets"]["retransmits"], complete["packets"]["total"]), 6),
}

smoke = {
    "complete_case_passes": all(check(complete) for check in CHECKS.values()),
    "caught_unit_confusion": not check_bandwidth(case_by_name["bandwidth_unit_confused_bad"]),
    "caught_roce_loss_gap": not check_roce(case_by_name["roce_no_pfc_ecn_bad"]),
    "caught_rank_straggler": not check_rank_straggler(case_by_name["rank_straggler_hidden_bad"]),
    "caught_scheduler_blindness": not check_scheduler(case_by_name["scheduler_network_blind_bad"]),
}

audit = audit_network(cases)
print(f"network_examples={network_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"network_gate_pass={audit['network_gate_pass']}")
```

一组典型输出是：

```text
network_examples={'gbps_400_to_gib_s': 46.57, 'send_16gib_at_400gbps_80eff_s': 0.429, 'ring_allreduce_8gib_64ranks_200gbps_75eff_s': 0.902, 'allgather_8gib_64ranks_200gbps_75eff_s': 0.451, 'jitter_ratio': 2.25, 'retransmit_rate': 5e-06}
smoke={'complete_case_passes': True, 'caught_unit_confusion': True, 'caught_roce_loss_gap': True, 'caught_rank_straggler': True, 'caught_scheduler_blindness': True}
metrics={'bandwidth_unit_accounting': 0.941, 'latency_jitter_tracking': 0.941, 'rdma_capability_fit': 0.941, 'gpudirect_rdma_path': 0.941, 'infiniband_fabric_readiness': 0.941, 'roce_congestion_losslessness': 0.941, 'ethernet_fallback_scope': 0.941, 'collective_operation_modeling': 0.941, 'nccl_topology_runtime_fit': 0.941, 'allreduce_cost_estimation': 0.941, 'allgather_reducescatter_cost': 0.941, 'rank_straggler_detection': 0.941, 'packet_error_retransmit_tracking': 0.941, 'topology_congestion_awareness': 0.941, 'scheduler_network_locality': 0.941, 'network_communication_gate': 0.941}
hard_blocker_count=16
failed_cases=['bandwidth_unit_confused_bad', 'latency_jitter_ignored_bad', 'rdma_missing_bad', 'gdr_path_missing_bad', 'ib_fabric_unverified_bad', 'roce_no_pfc_ecn_bad', 'ethernet_used_for_pretraining_bad', 'collective_unmodeled_bad', 'nccl_topology_unknown_bad', 'allreduce_cost_missing_bad', 'allgather_reducescatter_ignored_bad', 'rank_straggler_hidden_bad', 'packet_errors_untracked_bad', 'topology_congestion_ignored_bad', 'scheduler_network_blind_bad', 'network_gate_missing_bad']
network_gate_pass=False
```

这个 demo 的重点是把“网络慢”拆成可定位的证据链。先确认 bit / byte 和有效带宽，再看 latency、jitter、RDMA、GPUDirect RDMA、InfiniBand / RoCE / Ethernet 适用范围；然后把 AllReduce、AllGather、ReduceScatter 的通信量估出来，结合 NCCL 拓扑、rank straggler、错误包、重传、拥塞路径和调度放置一起判断。只说“网卡是 400G，所以网络没问题”在大模型训练排障里是不够的。

## 8.19 面试中如何回答网络问题

如果面试官问：

```text
为什么多机训练扩展效率很差？你怎么排查？
```

可以这样回答：

```text
我会先看 step time 分解，确认通信占比是否升高。然后看 NCCL 日志和 profiler，定位是 AllReduce、AllGather 还是 ReduceScatter 慢。

接着看 rank 间差异，判断是否有 straggler。再检查网络层，包括网卡带宽、错误包、重传、RDMA/GPUDirect 是否生效、GPU 到网卡亲和性、交换机链路拥塞和跨机柜路径。

最后看调度和并行策略：tensor parallel 是否跨机，任务是否被分散到拓扑较远的节点，是否和其他大任务共享瓶颈链路。优化方向包括拓扑感知调度、通信计算重叠、调整并行策略、减少跨机通信和优化 NCCL 配置。
```

如果面试官问：

```text
InfiniBand 和 RoCE 有什么区别？
```

可以回答：

```text
InfiniBand 是面向高性能计算的专用高速网络，RDMA 支持成熟，低延迟、高带宽，常用于大规模训练集群，但成本和运维复杂度较高。

RoCE 是在以太网上承载 RDMA，能复用以太网生态，成本和通用性可能更好，但对拥塞控制、PFC/ECN、丢包和运维配置非常敏感。两者都可用于 AI 集群，选择要看规模、预算、网络团队能力和性能目标。
```

## 8.20 常见误区

误区一：网络只影响数据加载，不影响训练计算。

多机训练中，网络参与梯度、参数和 activation 同步，是训练主路径的一部分。

误区二：带宽高就一定快。

延迟、抖动、拥塞、丢包、拓扑、通信算法和 tail latency 都会影响训练。

误区三：RoCE 是普通以太网，不需要特殊配置。

RoCE 对拥塞控制和低丢包要求很高，配置不好性能会很差。

误区四：NCCL timeout 一定是代码 bug。

也可能是网络抖动、某个 rank 慢、网卡故障、拓扑问题或资源干扰。

误区五：调度和网络无关。

调度决定任务放在哪些节点上，直接影响通信路径、拥塞和扩展效率。

## 8.21 面试题

### 题 1：为什么多机训练需要高速网络？

答：多机训练中，不同 GPU 分布在不同服务器上，需要通过网络同步梯度、参数分片、中间 activation 或 pipeline 数据。这些通信可能发生在每个 step，甚至每层。如果网络慢，GPU 会等待通信，导致 step time 变长、MFU 下降和扩展效率变差。

### 题 2：RDMA 和 GPUDirect RDMA 的价值是什么？

答：RDMA 允许远程直接内存访问，减少 CPU 和内核协议栈开销，降低延迟、提高吞吐。GPUDirect RDMA 进一步让网卡直接访问 GPU 显存，减少 CPU 内存中转，对大规模 GPU 通信非常重要。

### 题 3：AllReduce 在数据并行中做什么？

答：每个 GPU 计算本地 mini-batch 梯度后，需要把各 GPU 的梯度聚合并同步。AllReduce 会对所有 rank 的梯度做 reduce，并把结果分发回所有 rank，让每个 GPU 使用相同的平均梯度更新参数。

### 题 4：FSDP / ZeRO 为什么会增加通信？

答：FSDP / ZeRO 通过分片参数、梯度和优化器状态降低显存占用。但计算某层时可能需要 AllGather 拉取参数，反向传播后需要 ReduceScatter 分发梯度。这是用通信换显存的 trade-off。

### 题 5：如何优化多机通信？

答：可以从网络、通信库、并行策略和调度四层优化。网络层使用高性能 IB/RoCE 和 GPUDirect RDMA；通信库层优化 NCCL 参数；并行策略层减少跨机 tensor parallel、做通信计算重叠；调度层做拓扑感知，避免把通信密集任务分散到远距离节点。

## 8.22 小练习

练习一：画出数据并行 AllReduce 的流程。

要求：用 4 个 rank 举例，说明每个 rank 计算本地梯度后如何得到平均梯度。

练习二：比较 InfiniBand、RoCE 和普通以太网。

要求：从带宽、延迟、RDMA、成本、运维复杂度和适用场景分析。

练习三：分析 NCCL timeout。

假设一个 64 卡训练任务偶发 NCCL timeout，请列出至少 8 个可能原因，并按训练指标、NCCL、网卡、交换机、调度分层排查。

练习四：设计一个网络监控面板。

要求：包含网卡吞吐、错误包、重传、RDMA 状态、NCCL 通信时间、rank 差异、交换机拥塞和跨机柜流量。

## 8.23 本章小结

本章讲了 AI 集群网络基础。

你需要掌握：

1. 多机训练中，网络是训练主路径，不只是传文件。
2. 网络指标包括带宽、延迟、抖动、丢包、重传、拓扑和拥塞。
3. RDMA 和 GPUDirect RDMA 可以降低通信开销，提高 GPU 间跨机通信效率。
4. InfiniBand 性能强但成本和运维复杂度高。
5. RoCE 复用以太网生态，但对拥塞控制和低丢包要求高。
6. 普通以太网适合小规模或低通信任务，不适合高强度大规模预训练。
7. 集合通信包括 AllReduce、AllGather、ReduceScatter 等，是分布式训练的关键。
8. NCCL 是多 GPU 集合通信的重要库。
9. 网络瓶颈要从训练指标、通信库、网卡、交换机、拓扑和调度多层排查。

下一章我们会讲存储体系：本地盘、共享文件系统、对象存储和数据湖。
