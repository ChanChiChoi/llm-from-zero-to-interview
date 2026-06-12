# 第 3 章 GPU、NPU、TPU 与 AI 加速器基础

大模型训练和推理离不开 AI 加速器。你可以把加速器理解成专门为大规模矩阵计算、张量计算和并行计算设计的硬件。

很多同学一听到 AI Infra，就先想到“GPU 很贵”“A100、H100 很强”。但面试里不能只背型号。你需要理解：为什么大模型需要加速器？GPU、NPU、TPU 有什么共同点和差异？为什么显存、带宽、互联、软件生态和调度同样重要？

先记住一句话：

> AI 加速器的核心价值，是用高度并行的计算单元和高带宽内存，加速大模型中海量矩阵乘法、attention、MLP 和数据搬运。

## 3.0 本讲资料边界与第二轮精修口径

本讲是硬件基础入口，不追逐某一代 GPU、TPU、NPU 或云实例的最新参数，也不把某个厂商的峰值算力写成通用结论。硬件规格变化很快，面试中更重要的是建立稳定判断框架：计算峰值、显存容量、显存带宽、互联、低精度格式、软件栈、通信库、调试工具、成本和迁移风险。

第二轮精修时，我按 `WRITING_PLAN.md` 做了资料校准，主要参考公开官方资料中的稳定边界：NVIDIA CUDA 文档把 CUDA 定义为 GPU 上的并行计算平台和编程模型；NVIDIA NCCL 文档强调 NCCL 是面向 GPU 间通信的 collective / point-to-point 通信库，并支持 PCIe、NVLink、InfiniBand 等互联；Google Cloud TPU 文档把 TPU 定义为面向机器学习工作负载的 ASIC，核心是快速矩阵运算、MXU、HBM、slice、topology 和 Pod；AMD ROCm 文档说明 ROCm 是面向 AMD GPU 的开源软件栈，覆盖 HIP、OpenCL、OpenMP 和深度学习框架兼容。

因此，本章新增内容只写稳定抽象：GPU、NPU、TPU 和 AI ASIC 都是加速器选择的一部分，差异不只在“谁的 TFLOPS 更高”，而在真实工作负载能否落到硬件、内存、互联和软件生态上。

## 3.1 为什么大模型需要 AI 加速器

大模型的主要计算来自矩阵和张量操作。

例如 Transformer 里最核心的计算包括：

1. Q、K、V 线性投影。
2. Attention score 计算。
3. Attention value 聚合。
4. MLP 中的矩阵乘法。
5. LayerNorm、激活函数和残差连接。
6. 反向传播中的梯度计算。

这些操作有两个特点：

1. 计算量巨大。
2. 并行度很高。

CPU 擅长复杂控制逻辑和通用任务，但大规模矩阵乘法不是 CPU 的强项。GPU 和其他 AI 加速器则有大量并行计算单元，可以同时处理大量乘加操作。

简单对比：

```text
CPU：少量强核心，适合复杂控制和通用任务。
GPU：大量并行核心，适合矩阵、向量和图形/张量计算。
AI ASIC：面向 AI 算子专门设计，追求特定工作负载的效率。
```

大模型不是不能在 CPU 上跑，而是效率和成本通常无法接受。

## 3.2 GPU 是什么

GPU 最初是图形处理器，用来做图形渲染。后来大家发现图形计算和深度学习都大量依赖并行矩阵运算，所以 GPU 成为深度学习训练的主力。

现代数据中心 GPU 通常包含：

1. 大量计算核心。
2. Tensor Core 或类似矩阵加速单元。
3. 高带宽显存，例如 HBM。
4. 高速互联，例如 NVLink。
5. 支持 CUDA、cuDNN、NCCL 等软件栈。

GPU 的优势：

1. 通用性强。
2. 软件生态成熟。
3. 深度学习框架支持完善。
4. 适合训练和推理。
5. 可编程性较好。

GPU 的限制：

1. 成本高。
2. 功耗高。
3. 供应紧张。
4. 多机通信复杂。
5. 对软件栈版本敏感。

面试中提到 GPU，不要只说“算力强”，还要说“显存、带宽、互联和生态”。

## 3.3 NPU 是什么

NPU 通常指 Neural Processing Unit，也就是神经网络处理器。

不同厂商对 NPU 的定义不完全一致。有的 NPU 是手机端 AI 加速单元，有的是数据中心 AI 加速卡，有的是云厂商自研 AI 芯片。

NPU 的共同目标是：针对神经网络算子做专门优化。

它可能更强调：

1. 矩阵乘法加速。
2. 低精度计算。
3. 能效比。
4. 端侧推理。
5. 特定框架和算子融合。

NPU 的优势：

1. 对特定模型或算子能效高。
2. 在端侧设备上可以降低功耗。
3. 在云侧可能降低单位推理成本。
4. 可以和厂商软硬件栈深度结合。

NPU 的挑战：

1. 软件生态通常不如 CUDA 成熟。
2. 算子兼容性可能有限。
3. 模型迁移需要适配。
4. 调试和性能分析工具可能不完善。
5. 多机训练生态可能不如主流 GPU 成熟。

所以，NPU 不是“比 GPU 更高级”，而是面向特定场景的加速器选择。

## 3.4 TPU 是什么

TPU 是 Tensor Processing Unit，最早由 Google 面向 TensorFlow 和大规模机器学习工作负载设计。

TPU 的核心思想是用专门的矩阵乘法单元加速张量计算。

它通常强调：

1. 大规模矩阵乘法吞吐。
2. 高效低精度计算。
3. Pod 级互联。
4. 与 XLA 编译器结合。
5. 面向大规模训练和推理的整体系统设计。

TPU 的优势：

1. 在适配良好的模型和框架下效率很高。
2. 大规模集群互联能力强。
3. 编译器可以做图级优化。
4. 适合大规模训练工作负载。

TPU 的挑战：

1. 生态和使用方式与 GPU 不同。
2. 需要适配 XLA 和相关框架。
3. 某些动态模型或自定义算子迁移成本高。
4. 对非 Google 云环境不一定方便。

面试里可以把 TPU 理解成“专门面向张量计算和大规模 ML 工作负载的加速器体系”，而不是简单理解成“另一种 GPU”。

## 3.5 AI ASIC 与通用 GPU 的差异

ASIC 是 Application-Specific Integrated Circuit，也就是专用集成电路。

AI ASIC 是为 AI 工作负载设计的专用芯片。

和 GPU 相比：

1. GPU 更通用，生态成熟。
2. ASIC 更专用，可能能效更高。
3. GPU 更适合快速变化的模型研究。
4. ASIC 更适合稳定、高规模、可预测的工作负载。
5. GPU 的框架支持通常更好。
6. ASIC 的迁移和调试成本可能更高。

一个简单判断：

```text
如果模型和算子变化快，优先考虑生态成熟的 GPU。
如果工作负载稳定、规模巨大、成本敏感，可以考虑专用加速器。
```

## 3.6 AI 加速器的关键指标

选择加速器时，不能只看“峰值算力”。

需要看以下指标。

### 3.6.1 峰值算力

峰值算力表示理论最大计算能力，常见单位有 TFLOPS、PFLOPS。

但峰值算力不等于真实训练速度。

原因：

1. 算子不一定能跑满硬件。
2. 数据搬运可能成为瓶颈。
3. 显存容量可能限制 batch size。
4. 多机通信可能导致等待。
5. 框架和 kernel 优化影响很大。

所以面试中不要只拿峰值算力比较硬件。

### 3.6.2 显存容量

显存容量决定模型参数、梯度、优化器状态、activation 和 KV cache 能不能放下。

训练时显存主要被以下内容占用：

1. 参数。
2. 梯度。
3. 优化器状态。
4. activation。
5. 临时 buffer。

推理时显存主要被以下内容占用：

1. 模型权重。
2. KV cache。
3. batch 中间状态。
4. runtime buffer。

显存不够会直接限制模型大小、batch size、context length 和并发。

### 3.6.3 显存带宽

显存带宽决定数据从显存读写到计算单元的速度。

大模型很多操作既受计算限制，也受内存带宽限制。

尤其是推理 decode 阶段，每次生成 token 都要读取大量权重和 KV cache，显存带宽非常关键。

### 3.6.4 互联带宽

多 GPU 和多机训练需要大量通信。

互联包括：

1. PCIe。
2. NVLink。
3. NVSwitch。
4. InfiniBand。
5. RoCE。

互联影响：

1. 数据并行梯度同步。
2. 张量并行中间结果交换。
3. pipeline 并行 stage 间传输。
4. checkpoint 和参数加载。

如果互联慢，多卡训练可能变成“多卡一起等”。

### 3.6.5 低精度支持

大模型训练和推理常用 FP16、BF16、FP8、INT8、INT4 等格式。

硬件是否支持这些格式，会影响：

1. 训练速度。
2. 推理吞吐。
3. 显存占用。
4. 数值稳定性。
5. 量化部署成本。

例如 BF16 在训练中很常见，因为它比 FP16 有更大的指数范围，数值稳定性通常更好。

### 3.6.6 软件生态

软件生态包括：

1. 驱动。
2. 编译器。
3. kernel 库。
4. 通信库。
5. 框架支持。
6. profiler。
7. 调试工具。
8. 社区经验。

硬件再强，如果软件生态不成熟，真实落地成本会很高。

## 3.7 为什么 CUDA 生态重要

CUDA 是 NVIDIA 的并行计算平台和编程模型。

在大模型生态中，CUDA 的重要性不只是“能写 GPU 程序”。

它背后有一整套生态：

1. cuDNN。
2. cuBLAS。
3. NCCL。
4. TensorRT。
5. Triton 编译器生态。
6. PyTorch 和 TensorFlow 深度支持。
7. 大量开源项目默认优化 NVIDIA GPU。
8. 完整 profiler 和调试工具。

这也是为什么很多团队即使关注其他加速器，也会把 CUDA 生态作为重要对照。

面试中可以说：

```text
AI 加速器选型不只看芯片峰值算力，还要看框架、算子库、通信库、调试工具和开源生态。如果生态不成熟，迁移和调优成本可能抵消硬件优势。
```

## 3.8 训练和推理对硬件的需求不同

训练和推理都需要加速器，但关注点不同。

### 3.8.1 训练更关注什么

训练关注：

1. 计算吞吐。
2. 显存容量。
3. 多卡互联。
4. 多机网络。
5. checkpoint I/O。
6. 数值稳定性。
7. 分布式训练生态。

训练需要保存梯度、优化器状态和 activation，所以显存压力比推理更大。

训练还需要反向传播和参数更新，通信量也更大。

### 3.8.2 推理更关注什么

推理关注：

1. TTFT。
2. TPOT。
3. 吞吐。
4. 并发。
5. KV cache 显存。
6. 显存带宽。
7. 低精度和量化。
8. 单位 token 成本。

推理尤其关注 decode 阶段的效率。很多时候推理不是峰值算力不够，而是显存带宽、KV cache 和调度策略限制了吞吐。

## 3.9 云上加速器与自建集群

企业使用 AI 加速器有两种常见方式：云上租用和自建集群。

云上优点：

1. 启动快。
2. 弹性好。
3. 不需要自己维护机房。
4. 可以尝试多种硬件。
5. 适合早期探索和弹性需求。

云上缺点：

1. 长期成本可能高。
2. 资源不一定稳定可得。
3. 网络拓扑和性能可控性有限。
4. 数据合规和安全需要额外评估。
5. 深度定制能力有限。

自建优点：

1. 长期大规模使用成本可能更可控。
2. 网络、存储、调度和安全可深度定制。
3. 资源可预测。
4. 适合稳定大规模训练和推理。

自建缺点：

1. 初始投入大。
2. 建设周期长。
3. 运维复杂。
4. 容量规划难。
5. 硬件迭代风险高。

面试回答不要绝对说云上好或自建好，要看规模、合规、成本、资源稳定性和团队运维能力。

## 3.10 加速器选型的思考框架

选择 GPU、NPU、TPU 或其他加速器，可以从八个维度评估：

1. 工作负载：训练、推理、微调、embedding、rerank 还是多模态。
2. 模型规模：参数量、context length、batch size。
3. 性能目标：吞吐、延迟、训练时间。
4. 显存需求：权重、activation、KV cache、优化器状态。
5. 通信需求：单机多卡还是多机多卡。
6. 软件生态：框架、算子、通信库和 profiler。
7. 成本：采购、租用、电力、运维和迁移成本。
8. 风险：供应、兼容性、团队经验和厂商锁定。

一个简化模板：

```text
我不会只看峰值算力，而会从工作负载、显存、带宽、互联、低精度支持、软件生态、成本和运维风险综合选型。
```

## 3.11 加速器容量审计指标与最小 demo

硬件选型题不能只说“这张卡 TFLOPS 更高”。你需要把模型、batch、context length、精度、并行策略、训练 / 推理目标和硬件指标放到同一张表里。

先定义一个加速器选型审计样本：

```math
h_i=(w_i,m_i,b_i,\ell_i,r_i,c_i,u_i,e_i,p_i,z_i)
```

其中，`w_i` 是工作负载类型，`m_i` 是模型规模，`b_i` 是 batch 和并发，`\ell_i` 是 context length，`r_i` 是需要的训练或推理 runtime，`c_i` 是计算峰值和低精度能力，`u_i` 是显存容量、带宽和互联，`e_i` 是软件生态，`p_i` 是成本、电力、供应和云上 / 自建约束，`z_i` 是风险标签。

训练状态显存可以先用粗略下限估算：

```math
M_{\mathrm{train}}\approx N\,(B_{\mathrm{param}}+B_{\mathrm{grad}}+B_{\mathrm{opt}})+M_{\mathrm{act}}+M_{\mathrm{tmp}}
```

其中，`N` 是参数量，`B_param` 是每个参数的存储字节数，`B_grad` 是每个梯度的字节数，`B_opt` 是 optimizer state 的字节数，`M_act` 是 activation 显存，`M_tmp` 是临时 buffer。AdamW 训练时 optimizer state 往往比权重本身更大，所以“模型权重能放下”不等于“训练能跑”。

推理 KV cache 显存可以写成：

```math
M_{\mathrm{kv}}=2LBSH_{\mathrm{kv}}D_hB_{\mathrm{elem}}
```

其中，`L` 是层数，`B` 是 batch 或并发序列数，`S` 是序列长度，`H_kv` 是 KV head 数，`D_h` 是 head dimension，`B_elem` 是每个元素字节数，前面的 `2` 表示 K 和 V。这个公式解释了为什么长上下文和高并发推理很容易被 KV cache 卡住。

判断算子更像 compute-bound 还是 memory-bound，可以用 roofline 的简化直觉：

```math
F_{\mathrm{achievable}}\le \min(F_{\mathrm{peak}}, I B_{\mathrm{mem}})
```

其中，`F_peak` 是峰值算力，`B_mem` 是显存带宽，`I` 是 arithmetic intensity，也就是每读取 1 字节能做多少 FLOPs。`I` 很低时，即使峰值算力很高，也会被显存带宽限制。

多卡训练还要看通信暴露比例：

```math
R_{\mathrm{comm}}=\frac{T_{\mathrm{comm}}}{T_{\mathrm{compute}}+T_{\mathrm{comm}}+T_{\mathrm{io}}}
```

如果 `R_comm` 很高，问题通常不在单卡 TFLOPS，而在并行策略、互联、通信库、拓扑或 batch / bucket 配置。

下面是一个 0 依赖 Python demo。它用 toy 数字估算训练显存、KV cache、roofline 上限，并做加速器选型门禁：

```python
METRICS = [
    "peak_compute_fit",
    "memory_capacity_fit",
    "memory_bandwidth_fit",
    "interconnect_bandwidth_fit",
    "low_precision_support",
    "software_stack_maturity",
    "kernel_library_readiness",
    "distributed_communication_readiness",
    "training_memory_budget",
    "inference_kv_cache_budget",
    "workload_hardware_fit",
    "cloud_self_build_decision",
    "cost_power_capacity_awareness",
    "profiling_observability_readiness",
    "fallback_portability_plan",
    "selection_risk_governance",
]


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def training_state_gib(params_billion, param_bytes, grad_bytes, opt_bytes):
    params = params_billion * 1_000_000_000
    return round(gib(params * (param_bytes + grad_bytes + opt_bytes)), 2)


def kv_cache_gib(layers, batch, seq_len, kv_heads, head_dim, elem_bytes):
    total_bytes = 2 * layers * batch * seq_len * kv_heads * head_dim * elem_bytes
    return round(gib(total_bytes), 2)


def roofline_tflops(peak_tflops, memory_tbps, arithmetic_intensity):
    ridge = peak_tflops / memory_tbps
    achievable = min(peak_tflops, arithmetic_intensity * memory_tbps)
    bound = "compute_bound" if arithmetic_intensity >= ridge else "memory_bound"
    return {
        "ridge_flop_per_byte": round(ridge, 2),
        "achievable_tflops": round(achievable, 2),
        "bound": bound,
    }


def make_case(name, failed_metric=None, p0=False):
    flags = {metric: True for metric in METRICS}
    if failed_metric is not None:
        flags[failed_metric] = False
    return {"name": name, "flags": flags, "p0": p0}


def build_cases():
    bad_cases = [
        ("peak_tflops_only_bad", "peak_compute_fit"),
        ("memory_too_small_bad", "memory_capacity_fit"),
        ("decode_bandwidth_ignored_bad", "memory_bandwidth_fit"),
        ("multi_gpu_interconnect_ignored_bad", "interconnect_bandwidth_fit"),
        ("fp8_int8_missing_bad", "low_precision_support"),
        ("immature_software_stack_bad", "software_stack_maturity"),
        ("kernel_library_gap_bad", "kernel_library_readiness"),
        ("collective_comm_gap_bad", "distributed_communication_readiness"),
        ("training_state_oom_bad", "training_memory_budget"),
        ("kv_cache_oom_bad", "inference_kv_cache_budget"),
        ("workload_mismatch_bad", "workload_hardware_fit"),
        ("cloud_self_build_unreasoned_bad", "cloud_self_build_decision"),
        ("cost_power_ignored_bad", "cost_power_capacity_awareness"),
        ("no_profiler_metrics_bad", "profiling_observability_readiness"),
        ("no_portability_plan_bad", "fallback_portability_plan"),
        ("vendor_lockin_unowned_bad", "selection_risk_governance"),
    ]
    cases = [make_case("complete_accelerator_plan")]
    cases.extend(make_case(name, metric, p0=True) for name, metric in bad_cases)
    return cases


def audit_accelerator_selection(cases, threshold=0.95):
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
        "accelerator_gate_pass": gate_pass,
    }


capacity_examples = {
    "weights_70b_bf16_gib": round(gib(70_000_000_000 * 2), 2),
    "train_state_70b_gib": training_state_gib(70, 2, 2, 8),
    "kv_cache_gib": kv_cache_gib(
        layers=80,
        batch=8,
        seq_len=8192,
        kv_heads=8,
        head_dim=128,
        elem_bytes=2,
    ),
    "roofline": roofline_tflops(
        peak_tflops=1000,
        memory_tbps=3.35,
        arithmetic_intensity=80,
    ),
}

cases = build_cases()
report = audit_accelerator_selection(cases)
smoke = {
    "complete_case_passes": "complete_accelerator_plan" not in report["failed_cases"],
    "caught_peak_only": "peak_tflops_only_bad" in report["failed_cases"],
    "caught_training_oom": "training_state_oom_bad" in report["failed_cases"],
    "caught_kv_oom": "kv_cache_oom_bad" in report["failed_cases"],
    "caught_vendor_lockin": "vendor_lockin_unowned_bad" in report["failed_cases"],
}

print("capacity_examples=", capacity_examples)
print("smoke=", smoke)
print("metrics=", report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("accelerator_gate_pass=", report["accelerator_gate_pass"])
```

这段 demo 故意把 16 个 bad case 分别打穿 16 个维度，因此每个维度覆盖率都是 `16/17=0.941`。前面的 `capacity_examples` 还会展示三个直觉：70B 权重的 BF16 存储只是推理权重下限；训练状态会因为梯度和 optimizer state 明显放大；长上下文推理的 KV cache 可以单独吃掉大量显存。面试时你不需要背具体硬件型号，但要能把这些量级关系讲清楚。

## 3.12 常见误区

误区一：峰值算力越高，训练一定越快。

真实训练速度还受显存、带宽、通信、I/O、kernel 优化和并行策略影响。

误区二：GPU 只要数量够就行。

多 GPU 需要互联、调度、网络、存储和分布式训练栈配合。数量多但系统差，效率可能很低。

误区三：NPU 或 TPU 一定比 GPU 更适合 AI。

要看具体工作负载、软件生态、迁移成本和团队经验。专用硬件在适配良好时效率高，但不一定通用。

误区四：推理主要看算力。

推理尤其是 decode 阶段经常受显存带宽、KV cache 和调度限制。

误区五：硬件选型和算法无关。

模型结构、精度格式、context length、并行策略和量化方式都会影响硬件需求。

## 3.13 面试题

### 题 1：为什么大模型训练主要使用 GPU？

答：大模型训练包含大量矩阵乘法和张量操作，并行度很高。GPU 有大量并行计算单元、Tensor Core、高带宽显存和成熟的软件生态，适合加速 forward、backward 和优化器计算。相比 CPU，GPU 在这类工作负载上的吞吐和能效更高。

### 题 2：GPU、NPU、TPU 的区别是什么？

答：GPU 是通用并行计算加速器，生态成熟，适合多种训练和推理任务。NPU 通常指面向神经网络算子优化的处理器，可能在特定场景有更好能效，但生态和兼容性取决于厂商。TPU 是面向张量计算和大规模 ML 工作负载设计的加速器体系，通常与特定编译器和云生态结合更深。

### 题 3：为什么不能只看峰值 TFLOPS 选 GPU？

答：峰值 TFLOPS 是理论上限，真实性能还受显存容量、显存带宽、互联带宽、低精度支持、kernel 优化、通信库、数据 I/O 和并行策略影响。很多任务瓶颈不是计算，而是内存、通信或存储。

### 题 4：训练和推理对加速器的需求有什么不同？

答：训练需要 forward、backward、梯度和优化器状态，显存压力和通信压力更大，关注训练吞吐、扩展效率和稳定性。推理主要关注延迟、吞吐、并发、KV cache、显存带宽、低精度和单位 token 成本。

### 题 5：为什么软件生态是硬件选型的重要因素？

答：大模型落地依赖框架、算子库、通信库、编译器、profiler 和调试工具。硬件理论性能再强，如果生态不成熟，模型迁移、算子适配、性能调优和故障排查成本都会很高。

## 3.14 小练习

练习一：比较两张加速卡。

要求：不要只比较峰值算力，还要列出显存容量、显存带宽、互联、低精度支持、软件生态、成本和适用场景。

练习二：分析一个推理吞吐低的问题。

假设 GPU 峰值算力很高，但 decode 吞吐不高，请列出可能原因，包括显存带宽、KV cache、batching 和调度。

练习三：设计一个加速器选型表。

要求：列出训练、SFT、embedding、rerank、在线推理、离线批量推理等场景分别关注哪些硬件指标。

练习四：讨论云上 GPU 和自建 GPU 集群的取舍。

要求：从成本、弹性、合规、运维、资源稳定性和性能可控性角度分析。

## 3.15 本章小结

本章讲了 GPU、NPU、TPU 与 AI 加速器基础。

你需要掌握：

1. 大模型需要加速器，是因为矩阵乘法和张量计算规模巨大且并行度高。
2. GPU 的优势是通用性、成熟生态、高带宽显存和强大的矩阵计算能力。
3. NPU 和 TPU 是面向神经网络或张量计算优化的加速器，但生态、迁移和适配成本需要评估。
4. 选型不能只看峰值算力，还要看显存容量、显存带宽、互联、低精度支持、软件生态和成本。
5. 训练更关注吞吐、显存、通信和容错；推理更关注延迟、并发、KV cache、带宽和单位 token 成本。
6. 云上和自建各有优缺点，要结合规模、合规、成本和团队能力判断。

下一章我们会继续往硬件内部走，重点讲显存、HBM、PCIe、NVLink、NVSwitch 与带宽瓶颈。
