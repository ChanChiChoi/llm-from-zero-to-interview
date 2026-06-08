# 第八章：并行推理与多 GPU 服务

## 本章目标

理解模型太大或吞吐要求太高时如何进行多 GPU 推理。

## 核心议题

1. Tensor Parallel 推理。
2. Pipeline Parallel 推理。
3. Expert Parallel for MoE。
4. 多副本服务。
5. load balancing。
6. 跨节点通信瓶颈。

## 面试重点

部署中的并行目标和训练不同，常常更关注延迟、吞吐、稳定性和成本。

## 本章资料边界

本章第二轮精修时对照了 Megatron-LM 张量并行与流水线并行论文、NVIDIA TensorRT-LLM 并行推理文档、DeepSpeed Inference / MoE 资料、vLLM 分布式 serving 资料和主流 serving engine 的多 GPU 部署说明。这里聚焦部署面试中最常用的推理并行抽象：

1. Tensor Parallel、Pipeline Parallel、Expert Parallel 和多副本服务分别解决什么问题。
2. 权重、KV Cache、activation 和通信在多 GPU 中如何估算。
3. 为什么跨节点 TP / MoE expert routing 容易成为延迟瓶颈。
4. 如何在“模型放得下”“吞吐够不够”“延迟是否可接受”“故障域多大”之间做选型。

本章不复刻某个框架的具体 launcher 参数，也不把某个 GPU 拓扑上的 benchmark 数字写成通用结论。真正需要掌握的是并行切分改变了哪些显存项、通信项、调度项和故障域。

## 推理并行和训练并行有什么不同

训练并行关注的是：

1. 模型能不能训起来。
2. 梯度和优化器状态怎么同步。
3. 吞吐和收敛效率。

推理并行关注的是：

1. 模型能不能放下。
2. 单请求延迟是否可接受。
3. 多请求吞吐是否足够。
4. KV Cache 怎么分布。
5. 服务是否稳定。
6. 成本是否合理。

训练可以接受较高通信开销，只要总体 tokens/s 高；在线推理更敏感，因为通信可能直接增加用户请求延迟。

面试表达：训练并行关注训练效率和状态同步，推理并行更关注延迟、吞吐、KV Cache 和服务稳定性。

## 1. 为什么需要多 GPU 推理

多 GPU 推理通常有两类原因。

### 1.1 模型太大

单张 GPU 放不下模型权重和 KV Cache。

例如大模型权重本身就超过单卡显存，或者长上下文高并发导致 KV Cache 占用过大。

可用一个粗略门禁判断单卡是否放得下：

```math
G_{\mathrm{single}}=\mathbf{1}[M_w+M_{\mathrm{kv}}+M_{\mathrm{buf}}\le \rho M_{\mathrm{gpu}}]
```

其中 `M_w` 是权重显存，`M_kv` 是 KV Cache 显存，`M_buf` 是 runtime / activation / workspace 预留，`\rho` 是安全系数。这个门禁失败时，才需要优先考虑 TP、PP、量化或换更大显存 GPU。

### 1.2 吞吐要求太高

模型能放进单卡，但单卡服务不了目标 QPS 或 tokens/s。

这时可以部署多个副本，或使用多 GPU 并行提高吞吐。

## 2. Tensor Parallel 推理

Tensor Parallel 把模型内部的大矩阵切到多张 GPU 上。

例如线性层：

```text
Y = X W
```

可以把 `W` 按列或行切分，让多张 GPU 共同计算。

若 TP 大小为 `g_tp`，忽略复制项时，权重显存可粗略写成：

```math
M_{w,\mathrm{tp}}\approx \frac{M_w}{g_{\mathrm{tp}}}
```

如果 K/V head 也按 TP 切分，某个 rank 上的 KV Cache 可粗略写成：

```math
M_{\mathrm{kv,tp}}\approx
2LBT_{\mathrm{ctx}}\frac{H_{\mathrm{kv}}}{g_{\mathrm{tp}}}D_h b
```

其中 `L` 是层数，`B` 是 active sequence 数，`T_ctx` 是平均上下文 token 数，`H_kv` 是 KV heads，`D_h` 是 head dim，`b` 是每个 KV 元素字节数。不同实现可能复制或切分不同中间状态，所以这个公式用于面试估算，不替代框架内存 profiler。

TP 的代价是层内通信，例如 all-reduce / all-gather。可以抽象成：

```math
T_{\mathrm{layer}}\approx T_{\mathrm{compute}}+T_{\mathrm{comm,tp}}
```

通信项越靠近用户请求关键路径，TP 对在线延迟的影响越明显。

### 2.1 优点

1. 能部署单卡放不下的大模型。
2. 单层计算可以并行。
3. 是大模型推理中常见并行方式。

### 2.2 代价

1. 每层都可能有通信。
2. 对 GPU 间带宽要求高。
3. 跨节点 TP 延迟更高。

### 2.3 适用场景

Tensor Parallel 更适合同一节点内多 GPU，例如 NVLink / NVSwitch 互联。

面试表达：TP 切的是层内矩阵，适合模型太大或单层计算太重，但通信频繁，所以更适合同机高速互联。

## 3. Pipeline Parallel 推理

Pipeline Parallel 把模型按层切成多个 stage。

例如：

```text
GPU 0: layer 1-10
GPU 1: layer 11-20
GPU 2: layer 21-30
GPU 3: layer 31-40
```

若 PP stage 数为 `g_pp`，每个 stage 持有的层数约为：

```math
L_{\mathrm{stage}}\approx \left\lceil\frac{L}{g_{\mathrm{pp}}}\right\rceil
```

对应权重和 KV Cache 也大致按层切分：

```math
M_{w,\mathrm{pp}}\approx \frac{M_w}{g_{\mathrm{pp}}}
```

```math
M_{\mathrm{kv,pp}}\approx \frac{M_{\mathrm{kv}}}{g_{\mathrm{pp}}}
```

但单请求必须顺序经过所有 stage：

```math
T_{\mathrm{req}}\approx \sum_{s=1}^{g_{\mathrm{pp}}}T_s+\sum_{s=1}^{g_{\mathrm{pp}}-1}T_{\mathrm{comm},s}+T_{\mathrm{bubble}}
```

这就是 PP 在在线小 batch 场景下经常要谨慎的原因：它节省单卡显存，但可能把跨 stage 延迟直接加到请求路径上。

### 3.1 优点

1. 能把深模型放到多张 GPU 上。
2. 每张 GPU 只保存一部分层。

### 3.2 代价

1. 单请求要顺序经过多个 stage。
2. stage 间通信增加延迟。
3. 可能有 pipeline bubble。
4. 在线小 batch 场景利用率不一定好。

### 3.3 推理中的特点

训练中 pipeline 可以用 micro-batch 填满流水线。

但在线推理请求动态到达，输出长度不一致，pipeline 调度更复杂。

面试表达：PP 切的是层，能降低单卡权重显存，但会增加跨 stage 延迟，在线推理中要谨慎评估。

## 4. Tensor Parallel 和 Pipeline Parallel 怎么选

| 场景 | 倾向 |
| --- | --- |
| 单层矩阵很大 | Tensor Parallel |
| 模型层数很多且权重放不下 | Pipeline Parallel |
| 同机高速互联 | Tensor Parallel 更常见 |
| 跨节点通信较慢 | 减少跨节点 TP |
| 在线低延迟 | 尽量降低通信链路 |

真实系统可能组合 TP 和 PP，但复杂度会上升。

## 5. Expert Parallel for MoE

MoE 模型有多个 expert。

每个 token 由 router 分配到部分 expert。

Expert Parallel 把不同 expert 放到不同 GPU 上。

若 token `x_i` 被 router 分配到 top-k expert 集合 `E_i`，MoE 层可抽象为：

```math
y_i=\sum_{e\in E_i}p_{i,e} f_e(x_i)
```

Expert Parallel 的通信量取决于 token 到 expert 的跨 GPU 路由。负载不均衡可以用最大 expert token 数和平均 token 数的比值粗略衡量：

```math
R_{\mathrm{imbalance}}=\frac{\max_e n_e}{\frac{1}{E}\sum_{e=1}^{E}n_e}
```

`R_imbalance` 越高，某些 expert 越容易成为尾延迟瓶颈。

### 5.1 优点

1. 支持大规模 MoE 模型。
2. 每个 token 只激活部分 expert。
3. 总参数量可以很大。

### 5.2 难点

1. token 路由带来 all-to-all 通信。
2. expert 负载不均衡。
3. 某些 expert 可能成为瓶颈。
4. 部署和扩缩容更复杂。

面试表达：MoE 推理难点不只是参数多，而是 token 到 expert 的动态路由和 all-to-all 通信。

## 6. 多副本服务

如果模型能放进单卡或一组 GPU，可以部署多个副本。

例如：

```text
replica 1: GPU 0-1
replica 2: GPU 2-3
replica 3: GPU 4-5
```

请求由负载均衡器分发到不同 replica。

若每个副本需要 `g_rep` 张 GPU，总 GPU 数为 `G`，可部署副本数为：

```math
R=\left\lfloor\frac{G}{g_{\mathrm{rep}}}\right\rfloor
```

若单副本输出吞吐为 `S_rep`，理想总吞吐上限近似为：

```math
S_{\mathrm{total}}\approx R S_{\mathrm{rep}}
```

真实吞吐还会受路由、负载倾斜、冷启动、限流和故障冗余影响。

### 优点

1. 提高吞吐。
2. 提高可用性。
3. 便于水平扩展。
4. 单个副本故障不影响全部服务。

### 代价

1. 权重重复占显存。
2. 成本更高。
3. 需要负载均衡和健康检查。

面试表达：多副本是提高 QPS 和可用性的常见方式，但不能解决单副本模型放不下的问题。

## 7. Load balancing

负载均衡不是简单 round-robin。

因为 LLM 请求成本差异很大。

一个请求的成本取决于：

1. prompt 长度。
2. max_new_tokens。
3. 当前 decode 状态。
4. KV Cache 占用。
5. replica 当前队列长度。

常见策略：

1. round-robin。
2. least connections。
3. least queue time。
4. token-aware routing。
5. cost-aware routing。
6. tenant-aware routing。

面试表达：LLM 负载均衡最好看 token 和 KV Cache 成本，而不是只看请求数。

## 8. 跨节点通信瓶颈

多机推理比单机多卡更复杂。

原因是跨节点通信带宽和延迟通常不如节点内 NVLink / NVSwitch。

跨节点 TP 或 expert routing 可能显著增加延迟。

优化方向：

1. 尽量把强通信并行放在节点内。
2. 跨节点更多使用副本或较粗粒度切分。
3. 控制 batch 和并行度。
4. 监控通信耗时。

面试表达：跨节点并行推理要非常关注通信拓扑，因为通信延迟会直接影响用户请求延迟。

## 9. KV Cache 在多 GPU 中怎么处理

并行推理不仅要切权重，还要考虑 KV Cache。

在 TP 中，KV Cache 也可能按 head 或 hidden 维度分布在不同 GPU。

在 PP 中，不同层的 KV Cache 跟随对应 stage。

在多副本中，每个副本维护自己的 KV Cache。

这会影响：

1. 显存规划。
2. 调度策略。
3. 请求迁移。
4. 故障恢复。

通常一个正在生成的请求不适合频繁跨 replica 迁移，因为 KV Cache 已经在当前副本上。

## 10. 多 GPU 服务的故障处理

多 GPU 服务中，一个 GPU 故障可能导致整个副本不可用。

需要：

1. 健康检查。
2. 自动摘除故障副本。
3. 请求重试。
4. 限流和降级。
5. 快速拉起新副本。

如果是 TP/PP 组成的副本，任一 rank 挂掉通常会影响整个 replica。

## 11. 并行推理选型流程

可以按下面思路选择：

```text
1. 单卡能否放下模型和 KV Cache？
   能：优先单卡副本，简单稳定。
   不能：考虑 TP/PP/量化。

2. 是否需要提高 QPS？
   是：优先增加副本。

3. 是否单层矩阵太大？
   是：考虑 TP。

4. 是否模型层数太多或权重太大？
   是：考虑 PP。

5. 是否是 MoE？
   是：考虑 expert parallel 和路由开销。

6. 是否跨节点？
   是：谨慎评估通信延迟。
```

## 12. 最小 Python demo：多 GPU 推理选型审计

下面的 0 依赖 demo 用 toy 数字比较单卡、多副本、TP 和 TP+PP 的显存、通信和容量。它不代表真实 benchmark，只演示面试中如何把权重、KV Cache、通信和副本数放到同一张表里。

```python
from math import ceil


MODEL = {
    "params_b": 70,
    "layers": 80,
    "hidden": 8192,
    "kv_heads": 8,
    "head_dim": 128,
    "bytes_weight": 2,
    "bytes_kv": 2,
}

WORKLOAD = {
    "active_seq": 24,
    "ctx_tokens": 4096,
    "gpu_mem_gib": 80,
    "reserve_gib": 8,
    "num_gpus": 8,
}

PLANS = [
    {"name": "single_gpu", "tp": 1, "pp": 1, "replicas": 8},
    {"name": "tp4_two_replicas", "tp": 4, "pp": 1, "replicas": 2},
    {"name": "tp8_one_replica", "tp": 8, "pp": 1, "replicas": 1},
    {"name": "tp4_pp2_one_replica", "tp": 4, "pp": 2, "replicas": 1},
]


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def estimate(plan):
    tp, pp = plan["tp"], plan["pp"]
    gpus_per_replica = tp * pp
    weights = gib(MODEL["params_b"] * 1e9 * MODEL["bytes_weight"] / gpus_per_replica)
    layers_per_stage = ceil(MODEL["layers"] / pp)
    kv_heads_per_rank = ceil(MODEL["kv_heads"] / tp)
    kv = gib(
        2
        * layers_per_stage
        * WORKLOAD["active_seq"]
        * WORKLOAD["ctx_tokens"]
        * kv_heads_per_rank
        * MODEL["head_dim"]
        * MODEL["bytes_kv"]
    )
    total = weights + kv + WORKLOAD["reserve_gib"]
    fits = total <= WORKLOAD["gpu_mem_gib"]
    tp_comm_mib = 0.0
    if tp > 1:
        tp_comm_mib = 2 * MODEL["hidden"] * MODEL["bytes_kv"] * WORKLOAD["active_seq"] / 1024 ** 2
    pp_comm_mib = (pp - 1) * MODEL["hidden"] * MODEL["bytes_kv"] * WORKLOAD["active_seq"] / 1024 ** 2
    relative_latency = round(1.0 + 0.06 * (tp - 1) + 0.10 * (pp - 1), 2)
    aggregate_capacity = plan["replicas"] * WORKLOAD["active_seq"] if fits else 0
    return {
        "gpus_per_replica": gpus_per_replica,
        "replicas": plan["replicas"],
        "weights_gib": round(weights, 2),
        "kv_gib": round(kv, 2),
        "total_gib": round(total, 2),
        "fits": fits,
        "tp_comm_mib_per_decode": round(tp_comm_mib, 3),
        "pp_comm_mib_per_decode": round(pp_comm_mib, 3),
        "relative_latency": relative_latency,
        "aggregate_active_seq": aggregate_capacity,
    }


report = {plan["name"]: estimate(plan) for plan in PLANS}
for name, row in report.items():
    print(name, row)

candidate = max(
    (name for name, row in report.items() if row["fits"]),
    key=lambda name: report[name]["aggregate_active_seq"],
)
print("best_fit_by_capacity=", candidate)

replica_load = [
    {"id": "replica_0", "queued_tokens": 3200, "active_seq": 11},
    {"id": "replica_1", "queued_tokens": 900, "active_seq": 8},
]
new_request = {"prompt": 1200, "max_new": 256}
for replica in replica_load:
    replica["score"] = (
        replica["queued_tokens"] + 0.5 * new_request["prompt"] + 32 * replica["active_seq"]
    )
print("route_to=", min(replica_load, key=lambda x: x["score"])["id"])
```

一组可能输出：

```text
single_gpu {'gpus_per_replica': 1, 'replicas': 8, 'weights_gib': 130.39, 'kv_gib': 30.0, 'total_gib': 168.39, 'fits': False, 'tp_comm_mib_per_decode': 0.0, 'pp_comm_mib_per_decode': 0.0, 'relative_latency': 1.0, 'aggregate_active_seq': 0}
tp4_two_replicas {'gpus_per_replica': 4, 'replicas': 2, 'weights_gib': 32.6, 'kv_gib': 7.5, 'total_gib': 48.1, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.0, 'relative_latency': 1.18, 'aggregate_active_seq': 48}
tp8_one_replica {'gpus_per_replica': 8, 'replicas': 1, 'weights_gib': 16.3, 'kv_gib': 3.75, 'total_gib': 28.05, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.0, 'relative_latency': 1.42, 'aggregate_active_seq': 24}
tp4_pp2_one_replica {'gpus_per_replica': 8, 'replicas': 1, 'weights_gib': 16.3, 'kv_gib': 3.75, 'total_gib': 28.05, 'fits': True, 'tp_comm_mib_per_decode': 0.75, 'pp_comm_mib_per_decode': 0.375, 'relative_latency': 1.28, 'aggregate_active_seq': 24}
best_fit_by_capacity= tp4_two_replicas
route_to= replica_1
```

这段 demo 的结论是：

1. 单卡方案虽然副本最多，但 70B FP16 权重和 KV Cache 放不下。
2. `tp4_two_replicas` 每个副本 4 卡，可在 8 卡机器上放两个副本，toy 容量高于单个 8 卡副本。
3. `route_to` 不按 round-robin，而是按 queued tokens、active sequence 和新请求 prompt 成本做 token-aware routing。

## 13. 面试官会怎么问

### 问法 1：推理中的 Tensor Parallel 解决什么问题？

可以这样答：

```text
Tensor Parallel 把模型层内的大矩阵计算切到多张 GPU 上，解决单卡放不下或单卡算力不足的问题。它常用于大模型推理，但每层可能需要通信，所以更适合同机 NVLink/NVSwitch 这类高速互联环境。
```

### 问法 2：为什么推理中多副本很常见？

可以这样答：

```text
如果单个副本已经能放下模型，多副本是提高 QPS、吞吐和可用性的简单方式。每个副本独立服务请求，负载均衡器分发流量，某个副本故障时可以摘除。但多副本会重复保存权重，成本更高。
```

### 问法 3：TP 和 PP 在推理中怎么选？

可以这样答：

```text
TP 切层内矩阵，适合单层计算或权重太大，通信频繁但在同机高速互联下效果好。PP 切模型层，能把深模型分到多卡，但单请求要经过多个 stage，增加延迟并可能有 pipeline bubble。低延迟在线服务通常要谨慎使用 PP。
```

### 问法 4：MoE 推理为什么难？

可以这样答：

```text
MoE 推理中 token 会被 router 动态分配到不同 expert，不同 expert 可能在不同 GPU 上。这会带来 all-to-all 通信和负载不均衡问题。某些 expert 热点会影响整体延迟，所以要处理 expert parallel、路由和负载均衡。
```

### 问法 5：为什么 LLM 负载均衡不能只按请求数？

可以这样答：

```text
因为不同请求的 token 数、输出长度和 KV Cache 占用差异很大。一个长上下文请求可能比很多短请求更贵。所以负载均衡最好考虑 queue length、active tokens、KV Cache 使用和预计生成长度，而不是只看请求数。
```

## 14. 本章小结

本章核心结论：

1. 推理并行目标不同于训练并行，更重视延迟、吞吐、稳定性和成本。
2. Tensor Parallel 切层内矩阵，适合单层大计算和同机高速互联。
3. Pipeline Parallel 切模型层，能降低单卡权重显存，但可能增加在线延迟。
4. MoE 推理需要 expert parallel，并面对 all-to-all 和负载均衡问题。
5. 多副本服务是提高 QPS 和可用性的常见方式。
6. LLM 负载均衡要看 token、队列和 KV Cache 成本，而不是只看请求数。
7. 跨节点推理要谨慎评估通信瓶颈。
8. 多 GPU 服务必须配合健康检查、故障摘除、重试和降级。
