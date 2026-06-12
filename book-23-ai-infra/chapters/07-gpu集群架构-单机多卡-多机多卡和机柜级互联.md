# 第 7 章 GPU 集群架构：单机多卡、多机多卡和机柜级互联

前面我们讲了单张加速器、显存带宽和训练效率指标。从本章开始进入第二部分：集群、网络、存储与调度。

大模型训练和推理很少只依赖单张 GPU。真实系统通常是单机多卡、多机多卡，甚至机柜级、集群级互联。GPU 集群架构决定了训练能扩到多大、通信效率如何、推理能承载多少并发、故障影响范围多大，以及调度系统该如何分配资源。

先记住一句话：

> GPU 集群不是把很多卡堆在一起，而是计算、显存、互联、网络、存储、电力、散热和调度共同组成的系统。

## 7.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径按“稳定集群架构抽象”处理，而不是按某一代 GPU 服务器、某个云实例、某个交换机型号或某个调度器实现写死。单机多卡部分参考 NVIDIA HGX / DGX 资料中对 NVLink、NVSwitch、PCIe、CPU、NIC 和本地 NVMe 的系统边界；多机多卡部分参考 NCCL 对 collective communication、PCIe、NVLink / NVSwitch、InfiniBand / RoCE 和多节点通信的工程边界；GPU 到网卡亲和性部分参考 GPUDirect RDMA 文档对 PCIe 拓扑、驱动、权限和 peer-to-peer 访问条件的说明；机柜级 / 集群级部分只抽象 scale-out fabric、oversubscription、故障域、电力散热和调度感知，不把任何单一厂商的数值写成通用标准。

需要注意三点：

1. 本章关注 GPU cluster topology 和 placement，不重复上一章的“任务画像”总览，也不提前展开下一章的 InfiniBand、RoCE 和集合通信细节。
2. NVLink / NVSwitch 是 scale-up 互联，主要解决同机或同一高速互联域内 GPU 之间通信；InfiniBand / RoCE / Ethernet 是 scale-out 互联，主要解决节点之间和机柜之间通信。
3. 集群架构的核心不是“GPU 越多越好”，而是让并行策略、通信模式、故障域、数据位置、资源池和调度策略匹配。

## 7.1 为什么需要 GPU 集群

单张 GPU 再强，也会遇到限制。

主要限制包括：

1. 模型参数放不下。
2. batch size 不够大。
3. context length 受限。
4. 训练时间太长。
5. 推理并发不足。
6. 单卡故障会中断任务。
7. 多实验并行需要更多资源。

为了突破这些限制，需要多张 GPU 共同工作。

多 GPU 可以带来：

1. 更大的总显存。
2. 更高的总算力。
3. 更高训练吞吐。
4. 更高推理并发。
5. 更灵活的资源池。

但多 GPU 也带来新问题：通信、拓扑、调度、同步、故障和成本。

## 7.2 GPU 集群的基本层次

可以把 GPU 集群分成四个层次：

```text
单 GPU
  -> 单机多 GPU
  -> 多机多 GPU
  -> 机柜级 / 集群级 GPU 互联
```

每一层的瓶颈不同。

单 GPU 关注：

1. 显存容量。
2. HBM 带宽。
3. Tensor Core 性能。
4. kernel 效率。

单机多 GPU 关注：

1. GPU 间互联。
2. PCIe / NVLink / NVSwitch 拓扑。
3. 单机 CPU、内存和网卡配置。
4. NUMA 和设备亲和性。

多机多 GPU 关注：

1. 机间网络。
2. RDMA。
3. 集合通信。
4. 拓扑感知调度。
5. 分布式训练容错。

机柜级 / 集群级关注：

1. 交换机拓扑。
2. 机柜内和机柜间带宽。
3. 电力和散热。
4. 故障域。
5. 容量规划。

## 7.3 单机多卡架构

单机多卡是最常见的 GPU 服务器形态。

一台机器可能有 4 张、8 张或更多 GPU。

典型组件包括：

1. 多张 GPU。
2. CPU。
3. 系统内存。
4. PCIe Root Complex。
5. NVLink / NVSwitch。
6. 网卡。
7. 本地 NVMe。
8. 电源和散热系统。

单机多卡的优势：

1. 部署简单。
2. 通信延迟低于多机。
3. 故障域相对小。
4. 适合 SFT、推理、小规模训练。
5. 调试比多机更容易。

单机多卡的限制：

1. GPU 数量有限。
2. 总显存有限。
3. 单机故障会影响整机任务。
4. 机内拓扑可能不均匀。
5. CPU、内存、PCIe 和网卡可能成为瓶颈。

## 7.4 单机多卡拓扑为什么重要

不是所有 8 卡机器都一样。

拓扑决定 GPU 之间通信路径。

常见情况：

1. GPU 只通过 PCIe 通信。
2. 某些 GPU 对之间有 NVLink。
3. 所有 GPU 通过 NVSwitch 高速互联。
4. GPU 到网卡距离不同。
5. GPU 分属不同 CPU socket。

如果拓扑不均匀，调度和并行策略就要考虑设备亲和性。

例如：

1. tensor parallel group 应优先放在高速互联的 GPU 上。
2. 数据加载要尽量靠近对应 GPU。
3. 网卡和 GPU 的 NUMA 亲和性会影响多机通信。
4. 多个任务共享同一机器时，不能随便切分 GPU。

一个常见错误是：只申请“4 张 GPU”，但不关心这 4 张 GPU 是否在同一个高速互联域里。

## 7.5 PCIe-only 多卡机器

PCIe-only 机器中，GPU 之间主要通过 PCIe 和 CPU Root Complex 通信。

优点：

1. 成本相对低。
2. 架构通用。
3. 适合通信较少的任务。

缺点：

1. GPU 间通信带宽较低。
2. 延迟高于 NVLink。
3. 张量并行效率可能差。
4. 多卡训练扩展效率受限。

适合场景：

1. 数据并行为主的小规模训练。
2. LoRA / SFT。
3. 多个独立推理实例。
4. embedding 批处理。
5. 通信不密集的离线任务。

不适合场景：

1. 大规模 tensor parallel。
2. 需要频繁跨卡通信的模型并行。
3. 极致低延迟多卡推理。

## 7.6 NVLink / NVSwitch 多卡机器

高端训练服务器通常配备 NVLink 和 NVSwitch。

它们的目标是提高 GPU 间通信带宽，降低拓扑不均匀性。

适合场景：

1. 大模型预训练。
2. 张量并行。
3. pipeline 并行。
4. 大模型单机推理。
5. 高并发长上下文推理。

优势：

1. GPU 间通信更快。
2. 多卡模型并行效率更高。
3. 拓扑更均匀。
4. 更适合大模型切分。

限制：

1. 成本高。
2. 功耗高。
3. 供应和运维要求高。
4. 仍然受单机 GPU 数量和总显存限制。

面试中可以说：

```text
如果任务主要是数据并行，PCIe-only 机器可能够用；如果任务需要频繁 GPU 间通信，例如 tensor parallel 或大模型推理切分，NVLink/NVSwitch 的价值会明显增加。
```

## 7.7 多机多卡架构

当单机多卡无法满足模型规模或训练时间要求，就需要多机多卡。

多机多卡由多台 GPU 服务器组成，每台服务器内部有多张 GPU，服务器之间通过高速网络连接。

典型结构：

```text
Node 1: GPU x 8
Node 2: GPU x 8
Node 3: GPU x 8
...
Nodes connected by InfiniBand / RoCE / Ethernet
```

多机多卡的优势：

1. 可以扩展到更多 GPU。
2. 总显存更大。
3. 训练吞吐更高。
4. 可以承载更大模型。
5. 可以支持更大实验规模。

多机多卡的挑战：

1. 机间网络成为关键瓶颈。
2. 分布式训练复杂。
3. 故障概率更高。
4. 调度需要 gang scheduling。
5. checkpoint 和恢复更复杂。
6. 集群利用率更难优化。

多机多卡不是简单把单机复制多份，而是进入分布式系统问题。

## 7.8 多机训练中的通信模式

多机训练常见通信模式包括：

1. AllReduce。
2. AllGather。
3. ReduceScatter。
4. Broadcast。
5. Point-to-point。

数据并行主要依赖 AllReduce 或 ReduceScatter。

FSDP / ZeRO 会频繁做参数 shard 的 AllGather 和梯度 ReduceScatter。

张量并行可能在每层都需要跨卡通信。

pipeline 并行需要在 stage 之间传递 activation。

因此，通信模式和并行策略强相关。

如果把通信密集的 parallel group 跨机放置，网络压力会很大。

一个经验原则：

```text
尽量把通信最密集的并行维度放在带宽最高、延迟最低的互联范围内。
```

例如 tensor parallel 通常更适合同机 GPU，data parallel 可以跨机扩展。

## 7.9 机柜级 GPU 互联

当集群规模继续扩大，单台机器不再是主要视角，机柜成为重要单位。

一个机柜可能包含：

1. 多台 GPU 服务器。
2. 高速交换机。
3. 存储或缓存节点。
4. 管理网络。
5. 电力和散热设施。

机柜级设计要考虑：

1. 机柜内带宽。
2. 机柜间带宽。
3. oversubscription ratio。
4. 故障域。
5. 电力密度。
6. 冷却能力。
7. 维护便利性。

如果机柜内通信快、机柜间通信慢，调度系统就应该尽量把同一个大任务放在同一机柜或相邻机柜内。

这就是拓扑感知调度的意义。

## 7.10 Fat-tree、Dragonfly 和其他网络拓扑

大规模集群需要网络拓扑。

常见拓扑包括：

1. Fat-tree。
2. Dragonfly。
3. Torus / Mesh。
4. Clos 网络。

你不需要在算法面试里深入讲交换机设计，但要理解核心问题：网络拓扑决定任意两台机器之间的带宽、延迟和拥塞概率。

关注点包括：

1. 是否全双工。
2. 是否有阻塞。
3. 是否 oversubscribe。
4. 多任务并发时是否互相干扰。
5. 某个链路故障影响多大。

对于大模型训练，网络不是普通后台服务的网络。集合通信会产生高度同步的大流量，对拓扑和拥塞非常敏感。

## 7.11 GPU 集群中的存储位置

GPU 集群不只需要计算和网络，还需要数据和 checkpoint。

常见存储位置：

1. 本地 NVMe。
2. 共享文件系统。
3. 对象存储。
4. 数据湖。
5. 分布式缓存。

不同位置适合不同用途。

本地 NVMe：

1. 速度快。
2. 适合缓存热点数据。
3. 但容量有限，节点故障后数据可能不可用。

共享文件系统：

1. 使用方便。
2. 多节点可访问。
3. 但高并发下可能成为瓶颈。

对象存储：

1. 容量大。
2. 成本相对低。
3. 适合长期存储数据集和 checkpoint。
4. 但直接训练读取可能延迟较高。

AI Infra 常用策略是：对象存储做源数据，本地或分布式缓存做训练热数据。

## 7.12 GPU 集群中的控制面和数据面

可以把 GPU 集群分成控制面和数据面。

控制面负责管理：

1. 任务提交。
2. 调度决策。
3. 资源分配。
4. 权限认证。
5. 配额管理。
6. 任务状态。
7. 监控告警。

数据面负责实际运行：

1. GPU 计算。
2. GPU 间通信。
3. 数据读取。
4. checkpoint 写入。
5. 模型推理。

控制面故障可能导致新任务无法提交，但不一定影响正在运行的任务。

数据面故障会直接影响训练或推理。

成熟平台要区分这两类故障，并设计不同恢复策略。

## 7.13 故障域设计

GPU 集群会出现故障。

常见故障包括：

1. 单张 GPU 故障。
2. 整机故障。
3. 网卡故障。
4. 交换机故障。
5. 存储故障。
6. 电力或散热故障。
7. 驱动或系统软件故障。

故障域是指一个故障会影响的范围。

例如：

1. 单卡故障影响单机上的某些任务。
2. 整机故障影响整台机器上的任务。
3. 机柜交换机故障可能影响一个机柜。
4. 共享存储故障可能影响大量任务。

设计集群时要避免单点故障，并让调度系统知道故障域。

例如高优先级训练任务可以跨故障域放置，降低单点故障影响。但通信密集任务又希望放得近。这里存在 trade-off。

## 7.14 拓扑感知调度

拓扑感知调度是 GPU 集群调度的关键能力。

它不是简单找够 N 张 GPU，而是找“合适的一组 GPU”。

它需要考虑：

1. GPU 是否在同一机器。
2. GPU 之间是否有 NVLink / NVSwitch。
3. GPU 到网卡距离。
4. 机器之间是否在同一机柜。
5. 网络路径是否拥塞。
6. 存储数据是否靠近计算节点。
7. 故障域是否合理。

例如一个 8 卡 tensor parallel 任务，最好分配到同一台 NVSwitch 机器，而不是分散到 8 台机器各一张卡。

一个 256 卡预训练任务，最好分配到网络拓扑紧密的一组节点，而不是横跨整个集群随机分配。

## 7.15 GPU 集群资源池设计

企业通常会把 GPU 集群划分成不同资源池。

常见资源池：

1. 预训练资源池。
2. SFT 资源池。
3. 推理资源池。
4. 评估资源池。
5. 低优先级资源池。
6. 交互式开发资源池。

为什么要分资源池？

1. 任务画像不同。
2. SLO 不同。
3. 抢占策略不同。
4. 成本核算不同。
5. 故障影响不同。

例如在线推理资源池不能被低优先级训练任务随意抢占。预训练资源池需要大规模连续资源，而交互式开发资源池更关注快速启动。

## 7.16 GPU 集群容量规划

容量规划要回答：需要多少 GPU、什么类型 GPU、多少网络带宽、多少存储吞吐。

需要考虑：

1. 预训练计划。
2. SFT 实验数量。
3. 推理峰值流量。
4. 评估任务量。
5. Agent 和 RAG 工作负载。
6. 资源利用率目标。
7. 冗余和故障预留。
8. 未来增长。

容量规划不能只按平均负载做，否则峰值会排队严重。

也不能只按峰值买满，否则大量资源闲置。

这就是为什么需要调度、抢占、混部和成本治理。

## 7.17 一个 GPU 集群架构示例

一个简化企业 GPU 集群可以这样设计：

```text
User / Platform Portal / CLI
  -> Scheduler / Queue / Quota Manager
  -> Resource Pools
      -> Training Pool
          -> NVSwitch GPU Nodes
          -> High-speed Fabric
          -> Checkpoint Storage
      -> Inference Pool
          -> Serving Nodes
          -> Load Balancer
          -> Autoscaler
      -> Eval / Batch Pool
          -> Preemptible GPU Nodes
          -> Object Storage
  -> Observability
  -> Security / Audit
  -> Cost Governance
```

这个架构强调：

1. 训练、推理、评估资源池分开。
2. 调度系统统一管理配额和优先级。
3. 训练池关注拓扑和通信。
4. 推理池关注 SLO 和扩缩容。
5. 评估池可使用低优先级资源。
6. 监控、安全和成本贯穿所有资源池。

## 7.18 GPU 集群拓扑审计指标与最小 demo

前面讲的是架构直觉。真实平台设计里，还需要把 GPU 集群拓扑写成可检查的数据结构，否则调度系统只能看到“空闲 GPU 数量”，看不到这些 GPU 是否适合同一个任务。

可以把第 `i` 个集群候选资源组写成：

```math
g_i=(n_i,u_i,p_i,\ell_i,a_i,r_i,b_i,s_i,f_i,c_i,o_i,z_i)
```

其中，`n_i` 是节点数量，`u_i` 是单节点 GPU 数，`p_i` 是 PCIe / NUMA 拓扑，`\ell_i` 是 NVLink / NVSwitch 等 scale-up 互联，`a_i` 是 GPU 到 NIC 的亲和性，`r_i` 是 rack / pod 位置，`b_i` 是机内和机间带宽，`s_i` 是存储与 checkpoint 路径，`f_i` 是故障域，`c_i` 是电力散热容量，`o_i` 是观测指标，`z_i` 是风险。

总 GPU 和总显存可以先粗略写成：

```math
G_{\mathrm{total}}=N_{\mathrm{node}}G_{\mathrm{node}}
```

```math
M_{\mathrm{total}}=G_{\mathrm{total}}M_{\mathrm{gpu}}
```

其中，`N_node` 是节点数，`G_node` 是单节点 GPU 数，`M_gpu` 是单 GPU 显存容量。这个公式只说明资源上限，不代表这些显存一定能被一个模型像单卡一样自由使用；跨 GPU 使用显存要付通信代价。

通信耗时可以用简化模型表达：

```math
T_{\mathrm{comm}}\approx \alpha H+\frac{V}{B_{\mathrm{eff}}}
```

其中，`\alpha` 是单跳或一次通信启动延迟，`H` 是路径跳数，`V` 是通信量，`B_eff` 是有效带宽。这个式子提醒你：跨机、跨机柜、拥塞链路和 oversubscription 会同时影响延迟和带宽。

通信放置是否合理，可以看跨慢链路通信比例：

```math
R_{\mathrm{slow}}=\frac{V_{\mathrm{slow}}}{V_{\mathrm{total}}}
```

如果 tensor parallel 或专家并行的大量通信被放到跨机链路上，`R_slow` 会升高，扩展效率通常会下降。

网络 oversubscription 可以粗略写成：

```math
O_{\mathrm{net}}=\frac{B_{\mathrm{down}}}{B_{\mathrm{up}}}
```

其中，`B_down` 是下行接入总带宽，`B_up` 是上行汇聚带宽。`O_net=1` 接近无阻塞，`O_net>1` 表示存在超卖；是否能接受取决于任务通信模式和并发负载。

故障域影响可以写成：

```math
F_{\mathrm{rack}}=\frac{G_{\mathrm{rack}}}{G_{\mathrm{total}}}
```

其中，`G_rack` 是单机柜 GPU 数。这个比例越大，单个机柜故障的影响越大；但把任务跨太多故障域铺开，又可能增加通信成本。

最后，可以把 GPU 集群门禁写成：

```math
G_{\mathrm{cluster}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{slow}}\le \rho_{\mathrm{slow}} \land O_{\mathrm{net}}\le \rho_{\mathrm{over}} \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个拓扑审计指标覆盖率，`tau_j` 是最低覆盖率阈值，`rho_slow` 是慢链路通信比例阈值，`rho_over` 是 oversubscription 阈值，`P_0` 是 P0 级风险数量。

下面这个 0 依赖 demo 演示如何把 GPU 集群拓扑、并行组放置和资源池隔离写成审计规则。它故意构造 1 个完整样本和 16 个坏样本，让每个关键维度各失败一次。

```python
import copy


METRICS = [
    "scale_up_domain_fit",
    "pcie_numa_locality",
    "nvlink_nvswitch_locality",
    "gpu_nic_affinity",
    "inter_node_fabric_readiness",
    "rack_locality_awareness",
    "oversubscription_awareness",
    "collective_communication_fit",
    "parallel_group_placement",
    "storage_checkpoint_locality",
    "fault_domain_isolation",
    "power_cooling_capacity_fit",
    "resource_pool_isolation",
    "topology_aware_scheduling",
    "observability_topology_coverage",
    "gpu_cluster_gate",
]


def ring_allreduce_time_s(payload_gib, ranks, bandwidth_gbps):
    bytes_total = payload_gib * (1024 ** 3)
    traffic = 2 * (ranks - 1) / ranks * bytes_total
    return traffic / (bandwidth_gbps * 1_000_000_000)


def total_hbm_tib(nodes, gpus_per_node, hbm_gib_per_gpu):
    return nodes * gpus_per_node * hbm_gib_per_gpu / 1024


def oversubscription_ratio(down_gbps, up_gbps):
    return down_gbps / up_gbps


def rack_fault_blast_radius(gpus_per_rack, total_gpus):
    return gpus_per_rack / total_gpus


def build_cluster_cases():
    complete = {
        "name": "complete",
        "nodes": 16,
        "gpus_per_node": 8,
        "hbm_gib_per_gpu": 80,
        "scale_up": {"domain": "node_nvswitch", "gpus": 8, "fits_task": True},
        "pcie": {"numa_mapped": True, "root_complex_known": True},
        "nvlink": {"present": True, "nvswitch": True, "topology_known": True},
        "gpu_nic": {"affinity_mapped": True, "gdr_ready": True},
        "fabric": {"type": "infiniband_or_roce", "rdma": True, "bandwidth_gbps": 200, "lossless_or_cc": True},
        "rack": {"rack_locality": True, "nodes_per_rack": 4, "rack_spread_limit": 2},
        "network": {"down_gbps": 3200, "up_gbps": 3200, "oversubscription_known": True},
        "collective": {"modeled_ops": ["allreduce", "allgather", "reducescatter"], "nccl_tested": True},
        "parallel_groups": {"tensor_parallel_within_node": True, "data_parallel_cross_node": True},
        "storage": {"checkpoint_nearby": True, "data_cache": True, "write_bandwidth_gbps": 80},
        "fault_domain": {"rack_aware": True, "single_rack_fraction": 0.25, "checkpoint_recoverable": True},
        "power_cooling": {"power_kw_required": 120, "power_kw_available": 150, "cooling_ready": True},
        "resource_pools": {"training": "isolated", "inference": "isolated", "eval": "preemptible"},
        "scheduler": {"topology_aware": True, "gang": True, "quota": True},
        "observability": {"metrics": ["pcie", "nvlink", "nic", "nccl", "rack", "storage"], "traces": True},
        "gate": {"enabled": True},
    }

    def bad_case(name, mutator):
        case = copy.deepcopy(complete)
        case["name"] = name
        mutator(case)
        return case

    bad_cases = [
        bad_case("scale_up_domain_missing_bad", lambda c: c["scale_up"].update({"fits_task": False})),
        bad_case("pcie_numa_unknown_bad", lambda c: c["pcie"].update({"numa_mapped": False})),
        bad_case("nvlink_nvswitch_ignored_bad", lambda c: c["nvlink"].update({"topology_known": False})),
        bad_case("gpu_nic_affinity_missing_bad", lambda c: c["gpu_nic"].update({"affinity_mapped": False})),
        bad_case("inter_node_fabric_unready_bad", lambda c: c["fabric"].update({"rdma": False})),
        bad_case("rack_locality_missing_bad", lambda c: c["rack"].update({"rack_locality": False})),
        bad_case("oversubscription_unknown_bad", lambda c: c["network"].update({"oversubscription_known": False})),
        bad_case("collective_comm_unmodeled_bad", lambda c: c["collective"].update({"modeled_ops": []})),
        bad_case("parallel_group_random_bad", lambda c: c["parallel_groups"].update({"tensor_parallel_within_node": False})),
        bad_case("storage_checkpoint_remote_bad", lambda c: c["storage"].update({"checkpoint_nearby": False})),
        bad_case("fault_domain_single_rack_bad", lambda c: c["fault_domain"].update({"single_rack_fraction": 1.0})),
        bad_case("power_cooling_overbooked_bad", lambda c: c["power_cooling"].update({"power_kw_available": 100})),
        bad_case("resource_pool_mixed_bad", lambda c: c["resource_pools"].update({"inference": "shared_with_training"})),
        bad_case("topology_scheduler_disabled_bad", lambda c: c["scheduler"].update({"topology_aware": False})),
        bad_case("observability_no_topology_metrics_bad", lambda c: c["observability"].update({"metrics": ["gpu_utilization"]})),
        bad_case("gpu_cluster_gate_missing_bad", lambda c: c["gate"].update({"enabled": False})),
    ]
    return [complete] + bad_cases


def check_scale_up(case):
    return case["scale_up"]["fits_task"] and case["scale_up"]["gpus"] <= case["gpus_per_node"]


def check_pcie(case):
    return case["pcie"]["numa_mapped"] and case["pcie"]["root_complex_known"]


def check_nvlink(case):
    return case["nvlink"]["present"] and case["nvlink"]["topology_known"]


def check_gpu_nic(case):
    return case["gpu_nic"]["affinity_mapped"] and case["gpu_nic"]["gdr_ready"]


def check_fabric(case):
    f = case["fabric"]
    return f["rdma"] and f["bandwidth_gbps"] >= 100 and f["lossless_or_cc"]


def check_rack(case):
    return case["rack"]["rack_locality"] and case["rack"]["rack_spread_limit"] >= 1


def check_oversubscription(case):
    n = case["network"]
    return n["oversubscription_known"] and oversubscription_ratio(n["down_gbps"], n["up_gbps"]) <= 2.0


def check_collective(case):
    ops = set(case["collective"]["modeled_ops"])
    return {"allreduce", "allgather", "reducescatter"}.issubset(ops) and case["collective"]["nccl_tested"]


def check_parallel_groups(case):
    return case["parallel_groups"]["tensor_parallel_within_node"] and case["parallel_groups"]["data_parallel_cross_node"]


def check_storage(case):
    s = case["storage"]
    return s["checkpoint_nearby"] and s["data_cache"] and s["write_bandwidth_gbps"] >= 40


def check_fault_domain(case):
    f = case["fault_domain"]
    return f["rack_aware"] and f["single_rack_fraction"] <= 0.5 and f["checkpoint_recoverable"]


def check_power_cooling(case):
    p = case["power_cooling"]
    return p["power_kw_available"] >= p["power_kw_required"] and p["cooling_ready"]


def check_resource_pools(case):
    pools = case["resource_pools"]
    return pools["training"] == "isolated" and pools["inference"] == "isolated" and pools["eval"] in {"preemptible", "batch"}


def check_scheduler(case):
    s = case["scheduler"]
    return s["topology_aware"] and s["gang"] and s["quota"]


def check_observability(case):
    needed = {"pcie", "nvlink", "nic", "nccl", "rack", "storage"}
    return needed.issubset(set(case["observability"]["metrics"])) and case["observability"]["traces"]


def check_gate(case):
    return case["gate"]["enabled"]


CHECKS = {
    "scale_up_domain_fit": check_scale_up,
    "pcie_numa_locality": check_pcie,
    "nvlink_nvswitch_locality": check_nvlink,
    "gpu_nic_affinity": check_gpu_nic,
    "inter_node_fabric_readiness": check_fabric,
    "rack_locality_awareness": check_rack,
    "oversubscription_awareness": check_oversubscription,
    "collective_communication_fit": check_collective,
    "parallel_group_placement": check_parallel_groups,
    "storage_checkpoint_locality": check_storage,
    "fault_domain_isolation": check_fault_domain,
    "power_cooling_capacity_fit": check_power_cooling,
    "resource_pool_isolation": check_resource_pools,
    "topology_aware_scheduling": check_scheduler,
    "observability_topology_coverage": check_observability,
    "gpu_cluster_gate": check_gate,
}


def audit_gpu_cluster(cases):
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
        "gpu_cluster_gate_pass": not failed_cases and min(metrics.values()) >= 0.95,
    }


cases = build_cluster_cases()
case_by_name = {case["name"]: case for case in cases}
complete = case_by_name["complete"]
total_gpus = complete["nodes"] * complete["gpus_per_node"]

cluster_examples = {
    "total_gpus": total_gpus,
    "total_hbm_tib": round(total_hbm_tib(complete["nodes"], complete["gpus_per_node"], complete["hbm_gib_per_gpu"]), 2),
    "scale_up_allreduce_8gib_s": round(ring_allreduce_time_s(8, 8, 900), 3),
    "scale_out_allreduce_8gib_s": round(ring_allreduce_time_s(8, total_gpus, 200), 3),
    "rack_fault_blast_radius": round(rack_fault_blast_radius(32, total_gpus), 2),
    "oversubscription_ratio": round(oversubscription_ratio(3200, 1600), 2),
}

smoke = {
    "complete_case_passes": all(check(complete) for check in CHECKS.values()),
    "caught_pcie_numa_gap": not check_pcie(case_by_name["pcie_numa_unknown_bad"]),
    "caught_gpu_nic_gap": not check_gpu_nic(case_by_name["gpu_nic_affinity_missing_bad"]),
    "caught_parallel_group_gap": not check_parallel_groups(case_by_name["parallel_group_random_bad"]),
    "caught_resource_pool_mixing": not check_resource_pools(case_by_name["resource_pool_mixed_bad"]),
}

audit = audit_gpu_cluster(cases)
print(f"cluster_examples={cluster_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"gpu_cluster_gate_pass={audit['gpu_cluster_gate_pass']}")
```

一组典型输出是：

```text
cluster_examples={'total_gpus': 128, 'total_hbm_tib': 10.0, 'scale_up_allreduce_8gib_s': 0.017, 'scale_out_allreduce_8gib_s': 0.085, 'rack_fault_blast_radius': 0.25, 'oversubscription_ratio': 2.0}
smoke={'complete_case_passes': True, 'caught_pcie_numa_gap': True, 'caught_gpu_nic_gap': True, 'caught_parallel_group_gap': True, 'caught_resource_pool_mixing': True}
metrics={'scale_up_domain_fit': 0.941, 'pcie_numa_locality': 0.941, 'nvlink_nvswitch_locality': 0.941, 'gpu_nic_affinity': 0.941, 'inter_node_fabric_readiness': 0.941, 'rack_locality_awareness': 0.941, 'oversubscription_awareness': 0.941, 'collective_communication_fit': 0.941, 'parallel_group_placement': 0.941, 'storage_checkpoint_locality': 0.941, 'fault_domain_isolation': 0.941, 'power_cooling_capacity_fit': 0.941, 'resource_pool_isolation': 0.941, 'topology_aware_scheduling': 0.941, 'observability_topology_coverage': 0.941, 'gpu_cluster_gate': 0.941}
hard_blocker_count=16
failed_cases=['scale_up_domain_missing_bad', 'pcie_numa_unknown_bad', 'nvlink_nvswitch_ignored_bad', 'gpu_nic_affinity_missing_bad', 'inter_node_fabric_unready_bad', 'rack_locality_missing_bad', 'oversubscription_unknown_bad', 'collective_comm_unmodeled_bad', 'parallel_group_random_bad', 'storage_checkpoint_remote_bad', 'fault_domain_single_rack_bad', 'power_cooling_overbooked_bad', 'resource_pool_mixed_bad', 'topology_scheduler_disabled_bad', 'observability_no_topology_metrics_bad', 'gpu_cluster_gate_missing_bad']
gpu_cluster_gate_pass=False
```

这个 demo 的重点是把“GPU 集群架构”从机房图变成可验证的 placement 规则。单机内要证明 PCIe / NUMA、NVLink / NVSwitch 和 GPU-NIC 亲和性；多机要证明 fabric、collective、rack locality 和 oversubscription；平台层要证明并行组放置、资源池隔离、故障域、电力散热、存储 checkpoint 和拓扑观测都过线。否则即使 GPU 数量足够，训练也可能因为通信、调度或故障域设计失败而跑不稳。

## 7.19 面试中如何回答 GPU 集群架构

如果面试官问：

```text
如何设计一个支撑大模型训练和推理的 GPU 集群？
```

可以这样回答：

```text
我会先按任务画像拆资源池。预训练需要大规模连续 GPU、高速网络、checkpoint 存储和拓扑感知调度；SFT 需要快速启动和实验追踪；推理需要独立资源池、低延迟、自动扩缩容和高可用；评估和离线任务可以使用低优先级或可抢占资源。

底层集群分为单机多卡和多机多卡。单机内要关注 PCIe、NVLink、NVSwitch、GPU 到网卡的 NUMA 亲和性；多机要关注 InfiniBand/RoCE 网络、交换机拓扑、机柜内外带宽和故障域。

调度系统不能只找够 GPU 数量，还要拓扑感知，保证通信密集任务放在高速互联范围内。平台还要提供监控、日志、成本、权限、checkpoint 和故障恢复能力。
```

## 7.20 常见误区

误区一：GPU 集群就是很多 GPU。

真正的集群还包括网络、存储、调度、电力、散热、监控、安全和成本治理。

误区二：只要 GPU 数量够，训练就能线性加速。

通信、拓扑、数据加载、checkpoint 和调度都会影响扩展效率。

误区三：8 卡机器都一样。

PCIe-only、NVLink、NVSwitch 拓扑差异会显著影响模型并行性能。

误区四：训练和推理可以随便混用同一资源池。

训练和推理的任务画像不同。在线推理需要 SLO，不能被训练任务干扰。

误区五：调度只需要看空闲 GPU 数量。

大模型任务需要拓扑感知、故障域感知、优先级、配额和数据位置感知。

## 7.21 面试题

### 题 1：单机多卡和多机多卡的主要区别是什么？

答：单机多卡主要受机内 PCIe、NVLink、NVSwitch、CPU 和网卡拓扑影响，通信延迟较低，适合 SFT、小规模训练和单机推理。多机多卡可以扩展到更大规模，但依赖机间网络、集合通信、拓扑感知调度和容错，复杂度更高。

### 题 2：为什么 NVSwitch 对大模型训练有价值？

答：NVSwitch 提供更均匀、更高带宽的 GPU 间互联，减少 GPU 间拓扑差异，提升张量并行、pipeline 并行和多卡推理的通信效率。对通信密集的大模型任务尤其有价值。

### 题 3：为什么 tensor parallel 通常更适合同机放置？

答：tensor parallel 往往在模型每层都需要跨卡通信，通信频率高、延迟敏感。把 tensor parallel group 放在同机 NVLink/NVSwitch 范围内，可以减少跨机网络通信，提高效率。

### 题 4：什么是拓扑感知调度？

答：拓扑感知调度不是只分配指定数量 GPU，而是根据 GPU 之间互联、机器位置、机柜网络、GPU 到网卡亲和性、故障域和数据位置，选择最适合任务的一组资源。它能提高训练效率并降低通信瓶颈。

### 题 5：为什么推理资源池最好和训练资源池隔离？

答：推理是在线服务，有明确延迟和可用性 SLO；训练任务通常长时间运行、资源占用大、可能产生网络和 I/O 干扰。如果混用同一资源池，训练可能影响推理延迟和稳定性。因此通常需要隔离或至少设置严格优先级和资源保障。

## 7.22 小练习

练习一：画一个 8 卡单机拓扑图。

要求：分别画出 PCIe-only 和 NVSwitch 两种情况，并说明 tensor parallel 性能差异。

练习二：设计一个 128 卡预训练任务的资源分配策略。

要求：说明如何选择节点、是否要求同一机柜、如何考虑网络拓扑和故障域。

练习三：设计一个企业 GPU 资源池。

要求：至少包含预训练、SFT、推理、评估和交互式开发资源池，并说明每个资源池的优先级和抢占策略。

练习四：分析一个多机训练扩展效率低的问题。

要求：从单机拓扑、机间网络、parallel group 划分、NCCL、数据加载和 checkpoint 角度列排查清单。

## 7.23 本章小结

本章讲了 GPU 集群架构。

你需要掌握：

1. GPU 集群不是堆卡，而是计算、互联、网络、存储、调度、电力、散热和治理组成的系统。
2. 单机多卡关注 PCIe、NVLink、NVSwitch、NUMA 和 GPU 到网卡亲和性。
3. 多机多卡关注机间网络、集合通信、拓扑感知调度和容错。
4. tensor parallel 等通信密集任务应优先放在高速互联范围内。
5. 机柜级设计要考虑机柜内外带宽、故障域、电力和散热。
6. GPU 集群通常需要按任务画像划分训练、推理、评估、SFT 和低优先级资源池。
7. 调度系统不能只看 GPU 数量，还要看拓扑、故障域、优先级、配额和数据位置。

下一章我们会深入讲网络基础：InfiniBand、RoCE、以太网和集合通信。
