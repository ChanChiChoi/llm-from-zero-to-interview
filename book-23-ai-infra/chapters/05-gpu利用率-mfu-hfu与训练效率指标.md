# 第 5 章 GPU 利用率、MFU、HFU 与训练效率指标

前面我们讲了 GPU、显存、HBM、PCIe、NVLink 和 NVSwitch。接下来要回答一个非常工程化的问题：怎么判断一套训练系统跑得好不好？

很多人第一反应是看 GPU 利用率。GPU 利用率当然重要，但它不是唯一指标，也不是最准确的训练效率指标。大模型训练里还经常讨论 MFU、HFU、tokens/s、step time、吞吐、扩展效率、通信占比和 I/O 占比。

先记住一句话：

> GPU 利用率告诉你 GPU 忙不忙，MFU/HFU 更接近告诉你模型训练到底用了多少理论算力，训练效率要结合吞吐、通信、I/O、显存和稳定性一起看。

## 5.0 本讲资料边界与第二轮精修口径

本章讨论训练效率指标的稳定工程口径，不把某个监控面板字段、某个 GPU 型号、某个 profiler 的事件名或某篇论文里的具体数值写成通用标准。

第二轮精修时，资料边界按官方和论文公开材料校准：NVIDIA DCGM / Nsight Systems 代表了 GPU 利用率、显存、SM、带宽、kernel timeline 和系统级 trace 的常见观测入口；PyTorch profiler 代表了框架层 CPU / CUDA activity、算子耗时、shape、memory 和 trace 的观测入口；Megatron-LM 和 PaLM 等大模型训练论文常用 MFU / FLOPs utilization 说明大规模训练有效算力利用情况；前文带宽瓶颈章节已经说明 GPU utilization 需要和 HBM、PCIe、NVLink / NVSwitch、网络、存储和 checkpoint 一起解释。

因此，本章只抽象四类稳定结论：

1. GPU utilization 是入口指标，不能单独证明训练效率高。
2. tokens/s、step time 和 loss 曲线是端到端训练是否快且正确的基础证据。
3. MFU / HFU 需要明确 FLOPs 估算口径、硬件峰值口径和统计窗口。
4. 训练效率排查必须同时看通信、I/O、checkpoint、rank skew、padding、重计算、kernel 和扩展效率。

## 5.1 为什么只看 GPU 利用率不够

GPU 利用率通常表示一段时间内 GPU 是否有 kernel 在运行。

例如你看到：

```text
GPU Utilization: 95%
```

这说明 GPU 大部分时间都在执行任务。但这不等于训练效率一定高。

原因有几个：

1. GPU 忙，不代表算子高效。
2. 可能运行的是低效 kernel。
3. 可能计算单元忙，但显存带宽成为瓶颈。
4. 可能有大量小 kernel，调度开销很大。
5. 可能通信和计算没有充分重叠。
6. 可能 batch 太小，吞吐仍然低。
7. 可能有些 rank 很慢，整体被 straggler 拖住。

反过来，GPU 利用率低也不一定是模型代码错了，可能是数据加载、checkpoint、网络通信或调度问题。

所以 GPU 利用率是入口指标，不是结论。

## 5.2 常见训练效率指标

大模型训练常见指标包括：

1. GPU utilization：GPU 忙碌程度。
2. memory utilization：显存占用比例。
3. memory bandwidth utilization：显存带宽利用情况。
4. step time：每个训练 step 耗时。
5. tokens/s：每秒处理 token 数。
6. samples/s：每秒处理样本数。
7. TFLOPS/GPU：每张 GPU 实际计算吞吐。
8. MFU：Model FLOPs Utilization。
9. HFU：Hardware FLOPs Utilization。
10. scaling efficiency：多卡扩展效率。
11. communication overhead：通信占比。
12. dataloader time：数据加载耗时。
13. checkpoint time：checkpoint 保存耗时。

这些指标要结合起来看。

例如：

1. GPU utilization 高，但 tokens/s 低，可能是 kernel 低效或 batch 太小。
2. tokens/s 高，但 loss 异常，说明快但可能不对。
3. MFU 低，但 GPU utilization 高，可能是大量非有效模型计算。
4. step time 抖动大，可能是 I/O、网络或集群干扰。

## 5.3 tokens/s 和 samples/s

在大模型训练中，tokens/s 通常比 samples/s 更有意义。

因为不同样本长度可能差异很大。一个样本可以是 100 token，也可以是 4096 token。

tokens/s 表示每秒处理多少 token。

常见口径：

```text
global_tokens_per_second = global_batch_size * sequence_length / step_time
```

如果使用 packed sequence 或动态长度，计算会更复杂，需要按实际 token 数统计。

samples/s 表示每秒处理多少样本。

它适合样本长度相对稳定的任务，但在 LLM 训练中不如 tokens/s 稳定。

面试里可以说：

```text
LLM 训练通常更关注 tokens/s，因为模型计算量主要和 token 数、序列长度、模型大小有关，samples/s 容易被样本长度差异误导。
```

## 5.4 step time

step time 是一个训练 step 的耗时。

一个 step 通常包括：

1. 数据读取和预处理。
2. CPU 到 GPU 数据拷贝。
3. forward。
4. loss 计算。
5. backward。
6. 梯度同步。
7. optimizer step。
8. 日志记录。
9. checkpoint 或 eval，如果该 step 触发。

分析 step time 时，不能只看总时间，要拆分。

例如：

```text
step_time = dataloader + h2d_copy + forward + backward + communication + optimizer + logging + checkpoint
```

如果总 step time 是 2 秒，其中通信 0.8 秒，说明通信是大头。如果 dataloader 0.6 秒，说明数据供给可能拖慢训练。

## 5.5 MFU 是什么

MFU 是 Model FLOPs Utilization，模型 FLOPs 利用率。

它想回答：模型训练实际完成的有效计算，占硬件理论峰值算力的比例是多少。

粗略公式：

```text
MFU = actual_model_flops_per_second / theoretical_peak_flops
```

其中 actual_model_flops_per_second 通常根据模型结构、参数量、token 数和 step time 估算。

MFU 越高，说明模型有效计算越接近硬件理论上限。

MFU 的价值：

1. 比 GPU utilization 更接近有效训练计算。
2. 可以比较不同训练配置的效率。
3. 可以判断硬件是否被模型计算充分利用。
4. 可以帮助定位瓶颈是否在通信、I/O 或 kernel。

但 MFU 也不是绝对准确，因为 FLOPs 估算依赖模型结构和统计口径。

## 5.6 HFU 是什么

HFU 是 Hardware FLOPs Utilization，硬件 FLOPs 利用率。

它更关注硬件实际执行了多少 FLOPs，占理论峰值的比例。

可以粗略理解：

```text
HFU = hardware_executed_flops_per_second / theoretical_peak_flops
```

MFU 和 HFU 的区别在于：

1. MFU 更关注模型有效计算。
2. HFU 更关注硬件实际执行计算。
3. HFU 可能包含一些对模型训练不直接有用的额外计算。
4. MFU 更适合衡量训练算法和系统整体效率。

如果 HFU 高但 MFU 低，可能说明硬件在忙，但有很多开销不转化为有效模型训练。

例如：

1. 重计算过多。
2. padding 太多。
3. 低效 kernel。
4. 不必要的数据格式转换。
5. 额外同步和通信导致有效计算比例低。

## 5.7 MFU 和 GPU 利用率的区别

GPU utilization 回答：GPU 有没有在干活？

MFU 回答：GPU 干的活里，有多少是有效模型计算，而且接近理论峰值多少？

举个例子：

```text
场景 A：GPU 利用率 95%，MFU 25%。
场景 B：GPU 利用率 90%，MFU 45%。
```

场景 B 可能更好，因为它把更多硬件能力转化成了有效模型计算。

GPU 利用率高但 MFU 低的常见原因：

1. 小 kernel 太多。
2. kernel fusion 不充分。
3. padding 浪费严重。
4. 通信等待多。
5. activation recompute 过多。
6. batch 或 sequence 配置不合适。
7. attention kernel 不高效。

## 5.8 扩展效率 Scaling Efficiency

扩展效率衡量多卡训练是否真的带来接近线性的加速。

假设单卡 tokens/s 是 1000，8 卡理想情况应该是 8000 tokens/s。

如果实际只有 5600 tokens/s，那么扩展效率是：

```text
scaling_efficiency = actual_throughput / ideal_throughput
                   = 5600 / 8000
                   = 70%
```

扩展效率下降的常见原因：

1. 梯度同步成本增加。
2. 张量并行通信成本高。
3. pipeline bubble。
4. 数据加载跟不上。
5. 网络拥塞。
6. 拓扑不合理。
7. rank 间负载不均。
8. checkpoint 同步阻塞。

多卡训练不是卡越多越快。超过一定规模后，如果通信和调度没有优化，收益会递减。

## 5.9 通信占比

分布式训练中，通信占比非常关键。

常见通信包括：

1. 数据并行 AllReduce。
2. FSDP / ZeRO 的 AllGather 和 ReduceScatter。
3. 张量并行的 AllReduce / AllGather。
4. pipeline 并行的 activation 传输。
5. 参数广播。

通信占比可以这样理解：

```text
communication_ratio = communication_time / step_time
```

如果通信占比很高，优化方向可能是：

1. 调整并行策略。
2. 增大计算粒度。
3. 通信计算重叠。
4. 拓扑感知分组。
5. 减少跨机通信。
6. 优化 NCCL 配置。

面试中要强调：分布式训练性能很大程度取决于通信效率，而不是单卡算力。

## 5.10 数据加载和 I/O 指标

训练慢也可能是数据供给慢。

需要关注：

1. dataloader time。
2. CPU preprocessing time。
3. storage read throughput。
4. cache hit rate。
5. CPU 到 GPU 拷贝时间。
6. batch 组装耗时。
7. tokenizer 或解码耗时。

如果 dataloader 慢，可能出现：

1. GPU 利用率周期性下降。
2. step time 抖动。
3. CPU 使用率高。
4. 存储吞吐打满。
5. 某些 worker 成为瓶颈。

常见优化：

1. 预处理离线化。
2. 数据分片。
3. 本地缓存。
4. 提高 dataloader worker 数。
5. 使用 streaming dataset。
6. 减少训练时动态解析。
7. 使用 pinned memory 和异步拷贝。

## 5.11 Checkpoint 对效率的影响

Checkpoint 是训练可靠性的保障，但也会影响效率。

如果每隔一段时间保存 checkpoint，可能出现：

1. step time 周期性尖刺。
2. GPU 等待 I/O。
3. 存储带宽被打满。
4. 多个训练任务同时保存，造成拥塞。
5. checkpoint 文件过多，管理成本上升。

常见指标：

1. checkpoint save time。
2. checkpoint load time。
3. checkpoint size。
4. checkpoint frequency。
5. checkpoint failure rate。
6. storage write throughput。

优化方向：

1. 异步 checkpoint。
2. 分片 checkpoint。
3. 降低保存频率。
4. 只保留关键版本。
5. 使用更高吞吐存储。
6. 避免全量任务同时保存。

## 5.12 训练效率分析方法

分析训练效率，可以按这个顺序：

1. 看端到端吞吐：tokens/s、step time。
2. 看 GPU 利用率和显存占用。
3. 看 MFU / HFU。
4. 看 step time 分解。
5. 看通信占比。
6. 看数据加载和 I/O。
7. 看 checkpoint 影响。
8. 看 rank 间差异。
9. 看扩展效率。
10. 看 loss 是否正常。

为什么最后还要看 loss？

因为训练快但训练错没有意义。

例如混合精度配置错误可能让训练速度很快，但 loss 不稳定。数据 pipeline 有 bug 也可能让吞吐很好，但模型学不到东西。

## 5.13 常见诊断场景

### 5.13.1 GPU 利用率低

可能原因：

1. 数据加载慢。
2. CPU 预处理慢。
3. CPU 到 GPU 拷贝慢。
4. batch 太小。
5. checkpoint 阻塞。
6. 通信等待。
7. rank 间负载不均。
8. 调度系统导致资源干扰。

### 5.13.2 GPU 利用率高但 MFU 低

可能原因：

1. kernel 低效。
2. 小算子太多。
3. padding 浪费。
4. attention 实现低效。
5. 重计算开销过高。
6. 数据类型转换过多。
7. 计算不是主要瓶颈。

### 5.13.3 单卡快，多卡慢

可能原因：

1. 通信占比高。
2. 网络带宽不足。
3. 拓扑不合理。
4. 并行策略不合适。
5. NCCL 配置问题。
6. 某些 rank 数据加载慢。
7. checkpoint 或日志同步阻塞。

### 5.13.4 step time 抖动大

可能原因：

1. 数据读取不稳定。
2. 共享存储抖动。
3. 网络拥塞。
4. 周期性 checkpoint。
5. 其他任务资源干扰。
6. 动态 padding 导致 batch 计算量变化。
7. 垃圾回收或日志写入阻塞。

## 5.14 一个训练效率面板应该有什么

成熟训练平台应该提供训练效率面板。

至少包含：

1. loss 曲线。
2. learning rate 曲线。
3. tokens/s。
4. step time。
5. GPU utilization。
6. 显存占用。
7. 显存带宽利用。
8. MFU / HFU。
9. 通信时间。
10. dataloader time。
11. checkpoint time。
12. rank 间 step time 分布。
13. 网络吞吐。
14. 存储吞吐。
15. 错误和重试事件。

如果面板只有 loss 和 GPU utilization，定位问题会非常困难。

## 5.15 面试中如何回答训练效率问题

如果面试官问：

```text
一个大模型训练任务 GPU 利用率只有 40%，你怎么排查？
```

可以这样答：

```text
我不会只看 GPU utilization 一个指标，会先看 step time 分解和 tokens/s。

第一步看数据加载是否慢，包括 dataloader time、CPU 预处理、存储读取和 CPU 到 GPU 拷贝。

第二步看通信是否慢，包括 NCCL 时间、AllReduce/AllGather 占比、rank 间 step time 差异和网络带宽。

第三步看显存和 kernel，包括 batch 是否太小、padding 是否严重、attention kernel 是否高效、是否有大量小 kernel。

第四步看 checkpoint、日志和评估是否周期性阻塞。

最后结合 MFU/HFU、扩展效率和 loss 曲线判断是系统瓶颈、配置问题还是模型训练本身的问题。
```

如果面试官问：

```text
GPU 利用率 95%，是否说明训练效率很好？
```

可以回答：

```text
不一定。GPU utilization 只说明 GPU 大部分时间有 kernel 在跑，不代表这些计算都是有效模型计算。还要看 tokens/s、MFU、HFU、step time、通信占比、I/O、padding 浪费和 loss 是否正常。GPU 利用率高但 MFU 低，可能说明硬件忙于低效 kernel、重计算、padding 或其他开销。
```

## 5.16 训练效率审计指标与最小 demo

训练效率题不能只说“GPU 利用率 95%”。你需要能把吞吐、FLOPs、step time、通信、I/O、checkpoint、rank skew 和 loss 正确性放进同一个审计表。

先定义一个训练效率审计样本：

```math
e_i=(u_i,t_i,f_i,h_i,c_i,d_i,r_i,p_i,l_i,z_i)
```

其中，`u_i` 是 GPU / memory / bandwidth utilization，`t_i` 是 tokens/s、samples/s 和 step time，`f_i` 是模型 FLOPs 估算，`h_i` 是硬件峰值和实际执行 FLOPs，`c_i` 是通信时间和通信量，`d_i` 是 dataloader、存储、H2D 和 checkpoint I/O，`r_i` 是 rank 间差异，`p_i` 是 padding、重计算和 kernel 效率，`l_i` 是 loss / grad / numerics 正确性，`z_i` 是风险标签。

端到端吞吐可以写成：

```math
Q_{\mathrm{tok}}=\frac{B_{\mathrm{global}}S_{\mathrm{eff}}}{T_{\mathrm{step}}}
```

其中，`B_global` 是 global batch size，`S_eff` 是每个样本实际参与训练的有效 token 数，`T_step` 是 step time。如果使用 packing 或动态长度，`B_global S_eff` 应替换成真实非 padding token 数。

step time 分解可以写成：

```math
T_{\mathrm{step}}=T_{\mathrm{data}}+T_{\mathrm{h2d}}+T_{\mathrm{fwd}}+T_{\mathrm{bwd}}+T_{\mathrm{comm}}+T_{\mathrm{opt}}+T_{\mathrm{ckpt}}+T_{\mathrm{misc}}
```

MFU 可以写成：

```math
U_{\mathrm{mfu}}=\frac{F_{\mathrm{model}}/T_{\mathrm{step}}}{N_{\mathrm{gpu}}F_{\mathrm{peak}}}
```

其中，`F_model` 是一个 step 的模型有效 FLOPs，`N_gpu` 是 GPU 数量，`F_peak` 是单 GPU 理论峰值 FLOPs/s。粗略 decoder-only 训练估算常用 `F_model \approx 6NT`，其中 `N` 是参数量，`T` 是 step 内有效 token 数。这个估算适合快速量级判断，不应替代 profiler 和模型结构级 FLOPs 统计。

HFU 可以写成：

```math
U_{\mathrm{hfu}}=\frac{F_{\mathrm{hw}}/T_{\mathrm{step}}}{N_{\mathrm{gpu}}F_{\mathrm{peak}}}
```

其中，`F_hw` 是硬件实际执行 FLOPs。若 `U_hfu` 明显高于 `U_mfu`，说明硬件做了不少非模型有效计算，例如 padding、重计算、低效 kernel、格式转换或额外同步。

通信、I/O 和扩展效率可以分别写成：

```math
R_{\mathrm{comm}}=\frac{T_{\mathrm{comm}}}{T_{\mathrm{step}}}
```

```math
R_{\mathrm{io}}=\frac{T_{\mathrm{data}}+T_{\mathrm{h2d}}+T_{\mathrm{ckpt}}}{T_{\mathrm{step}}}
```

```math
E_n=\frac{Q_n}{nQ_1}
```

其中，`Q_n` 是 `n` 卡吞吐，`Q_1` 是单卡吞吐。`E_n` 低时，要优先看通信、I/O、拓扑、batch 粒度和 rank skew。

最后可以定义训练效率门禁：

```math
G_{\mathrm{eff}}=\mathbf{1}\left[U_{\mathrm{mfu}}\ge \tau_{\mathrm{mfu}} \land E_n\ge \tau_{\mathrm{scale}} \land R_{\mathrm{comm}}\le \rho_{\mathrm{comm}} \land R_{\mathrm{io}}\le \rho_{\mathrm{io}} \land P_0=0\right]
```

下面是一个 0 依赖 Python demo。它用 toy step profile 计算 tokens/s、MFU、HFU、通信占比、I/O 占比、rank skew 和扩展效率，并用 bad case 检查训练效率门禁。

```python
METRICS = [
    "tokens_throughput_accounting",
    "step_time_breakdown_coverage",
    "gpu_utilization_interpretation",
    "mfu_estimation",
    "hfu_estimation",
    "model_flops_accounting",
    "hardware_peak_accounting",
    "communication_ratio_tracking",
    "io_dataloader_tracking",
    "checkpoint_overhead_tracking",
    "rank_skew_detection",
    "scaling_efficiency_tracking",
    "padding_waste_awareness",
    "recompute_overhead_awareness",
    "loss_correctness_coupling",
    "efficiency_gate",
]


def peta_flops(flops):
    return round(flops / 1_000_000_000_000_000, 3)


def model_flops(params_billion, tokens):
    return 6 * params_billion * 1_000_000_000 * tokens


def utilization(flops, step_time_s, gpu_count, peak_tflops_per_gpu):
    peak_flops_per_s = gpu_count * peak_tflops_per_gpu * 1_000_000_000_000
    return round((flops / step_time_s) / peak_flops_per_s, 3)


def step_total(parts):
    return round(sum(parts.values()), 3)


def ratio(value, total):
    return round(value / total, 3)


def rank_skew(rank_step_times):
    avg = sum(rank_step_times) / len(rank_step_times)
    return {
        "min": min(rank_step_times),
        "max": max(rank_step_times),
        "avg": round(avg, 3),
        "skew_ratio": round((max(rank_step_times) - min(rank_step_times)) / avg, 3),
    }


def scaling_efficiency(single_gpu_tps, multi_gpu_tps, gpu_count):
    return round(multi_gpu_tps / (single_gpu_tps * gpu_count), 3)


def make_case(name, failed_metric=None, p0=False):
    flags = {metric: True for metric in METRICS}
    if failed_metric is not None:
        flags[failed_metric] = False
    return {"name": name, "flags": flags, "p0": p0}


def build_cases():
    bad_cases = [
        ("tokens_per_second_missing_bad", "tokens_throughput_accounting"),
        ("no_step_breakdown_bad", "step_time_breakdown_coverage"),
        ("gpu_utilization_as_final_answer_bad", "gpu_utilization_interpretation"),
        ("mfu_not_estimated_bad", "mfu_estimation"),
        ("hfu_not_separated_bad", "hfu_estimation"),
        ("model_flops_wrong_bad", "model_flops_accounting"),
        ("peak_flops_unknown_bad", "hardware_peak_accounting"),
        ("communication_ratio_missing_bad", "communication_ratio_tracking"),
        ("dataloader_io_unmeasured_bad", "io_dataloader_tracking"),
        ("checkpoint_spike_ignored_bad", "checkpoint_overhead_tracking"),
        ("rank_skew_hidden_bad", "rank_skew_detection"),
        ("scaling_efficiency_missing_bad", "scaling_efficiency_tracking"),
        ("padding_waste_ignored_bad", "padding_waste_awareness"),
        ("recompute_overhead_ignored_bad", "recompute_overhead_awareness"),
        ("fast_but_loss_wrong_bad", "loss_correctness_coupling"),
        ("no_efficiency_gate_bad", "efficiency_gate"),
    ]
    cases = [make_case("complete_training_efficiency_plan")]
    cases.extend(make_case(name, metric, p0=True) for name, metric in bad_cases)
    return cases


def audit_efficiency(cases, threshold=0.95):
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
        "efficiency_gate_pass": gate_pass,
    }


global_tokens = 262_144
step_parts = {
    "dataloader": 0.25,
    "h2d_copy": 0.08,
    "forward": 0.75,
    "backward": 1.05,
    "communication": 0.45,
    "optimizer": 0.25,
    "checkpoint": 0.12,
    "misc": 0.05,
}
step_time = step_total(step_parts)
model_step_flops = model_flops(params_billion=7, tokens=global_tokens)
hardware_step_flops = 14_640_000_000_000_000
tokens_per_second = round(global_tokens / step_time)

efficiency_examples = {
    "step_time_s": step_time,
    "tokens_per_second": tokens_per_second,
    "model_step_pflops": peta_flops(model_step_flops),
    "mfu": utilization(model_step_flops, step_time, gpu_count=8, peak_tflops_per_gpu=1000),
    "hfu": utilization(hardware_step_flops, step_time, gpu_count=8, peak_tflops_per_gpu=1000),
    "communication_ratio": ratio(step_parts["communication"], step_time),
    "io_ratio": ratio(step_parts["dataloader"] + step_parts["h2d_copy"] + step_parts["checkpoint"], step_time),
    "rank_skew": rank_skew([2.85, 2.9, 2.98, 3.0, 3.05, 3.1, 3.3, 3.42]),
    "scaling_efficiency_8gpu": scaling_efficiency(12_000, tokens_per_second, 8),
}

cases = build_cases()
report = audit_efficiency(cases)
smoke = {
    "complete_case_passes": "complete_training_efficiency_plan" not in report["failed_cases"],
    "caught_gpu_util_only": "gpu_utilization_as_final_answer_bad" in report["failed_cases"],
    "caught_mfu_missing": "mfu_not_estimated_bad" in report["failed_cases"],
    "caught_rank_skew": "rank_skew_hidden_bad" in report["failed_cases"],
    "caught_fast_wrong_loss": "fast_but_loss_wrong_bad" in report["failed_cases"],
}

print("efficiency_examples=", efficiency_examples)
print("smoke=", smoke)
print("metrics=", report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("efficiency_gate_pass=", report["efficiency_gate_pass"])
```

这段 demo 故意让 16 个 bad case 分别打穿 16 个审计维度，因此每个维度覆盖率都是 `16/17=0.941`。`efficiency_examples` 里的 `mfu` 低于 `hfu`，说明硬件执行 FLOPs 里有一部分没有变成模型有效 FLOPs；`communication_ratio`、`io_ratio`、`rank_skew` 和 `scaling_efficiency_8gpu` 则帮助你判断瓶颈更像通信、I/O、负载不均还是扩展效率不足。

## 5.17 常见误区

误区一：GPU 利用率越高越好。

高利用率是好信号，但不是充分条件。还要看有效吞吐和 MFU。

误区二：tokens/s 高就一定好。

如果 loss 异常、数据错误或训练不稳定，tokens/s 再高也没意义。

误区三：MFU 是精确真理。

MFU 依赖 FLOPs 估算口径，不同模型和实现可能有差异。它适合比较和诊断，但不能脱离上下文。

误区四：多卡扩展效率低一定是网络问题。

也可能是并行策略、batch 太小、rank 不均、checkpoint、数据加载或代码同步问题。

误区五：训练效率只归平台团队负责。

训练效率同时受模型结构、数据 pipeline、并行策略、kernel、硬件、网络、存储和调度影响，需要算法和平台一起优化。

## 5.18 面试题

### 题 1：GPU utilization 和 MFU 有什么区别？

答：GPU utilization 表示 GPU 是否忙，通常看一段时间内是否有 kernel 运行。MFU 表示模型有效 FLOPs 占硬件理论峰值 FLOPs 的比例，更接近训练有效计算效率。GPU utilization 高不代表 MFU 高。

### 题 2：HFU 和 MFU 的区别是什么？

答：HFU 更关注硬件实际执行 FLOPs 占峰值的比例，MFU 更关注模型有效计算 FLOPs 占峰值的比例。HFU 高但 MFU 低，可能说明硬件在忙，但有不少计算没有转化为有效模型训练。

### 题 3：为什么 LLM 训练更常用 tokens/s，而不是 samples/s？

答：LLM 样本长度差异很大，samples/s 容易误导。tokens/s 更直接反映模型处理的 token 数和计算量，适合比较训练吞吐。

### 题 4：多机训练扩展效率低怎么办？

答：先计算 scaling efficiency，再看 step time 分解。重点检查通信占比、NCCL 时间、rank 间差异、网络带宽、拓扑、并行策略、batch size、数据加载和 checkpoint。优化方向包括拓扑感知分组、通信计算重叠、调整并行策略、减少跨机通信和优化 I/O。

### 题 5：训练速度很快但 loss 不正常，应该怎么看？

答：说明性能指标不能脱离训练正确性。需要检查数据 pipeline、label、mask、loss 计算、混合精度、梯度裁剪、学习率、分布式同步和 checkpoint resume 是否正确。

## 5.19 小练习

练习一：设计一个训练效率 dashboard。

要求：包含 tokens/s、step time、GPU utilization、MFU、通信占比、dataloader time、checkpoint time、rank 差异和 loss 曲线。

练习二：分析一个 GPU 利用率低的问题。

假设 64 卡训练 GPU 利用率只有 45%，请写出从数据、通信、显存、kernel、checkpoint 和调度六个角度的排查清单。

练习三：计算扩展效率。

单卡吞吐 1200 tokens/s，8 卡实际吞吐 7200 tokens/s，计算扩展效率，并解释可能瓶颈。

练习四：比较两个训练配置。

配置 A：GPU utilization 96%，MFU 28%，tokens/s 10k。配置 B：GPU utilization 88%，MFU 42%，tokens/s 14k。你认为哪个更好？为什么？

## 5.20 本章小结

本章讲了 GPU 利用率、MFU、HFU 与训练效率指标。

你需要掌握：

1. GPU utilization 只说明 GPU 忙不忙，不等于训练效率高。
2. tokens/s 比 samples/s 更适合衡量 LLM 训练吞吐。
3. step time 要拆成数据加载、forward、backward、通信、优化器、checkpoint 等部分。
4. MFU 衡量模型有效 FLOPs 利用率，HFU 衡量硬件 FLOPs 利用率。
5. GPU utilization 高但 MFU 低，说明可能存在低效 kernel、padding、重计算或非有效开销。
6. 多卡训练要看扩展效率和通信占比。
7. 训练效率排查要同时看吞吐、显存、通信、I/O、checkpoint、rank 差异和 loss。

下一章我们会讲大模型任务画像：预训练、SFT、RLHF、评估、推理和 Agent，帮助你理解不同任务对 AI Infra 的资源需求差异。
