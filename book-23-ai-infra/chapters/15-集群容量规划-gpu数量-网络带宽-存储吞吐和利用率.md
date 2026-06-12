# 第 15 章 集群容量规划：GPU 数量、网络带宽、存储吞吐和利用率

前面几章讲了 GPU 集群、网络、存储、Kubernetes、调度和多租户隔离。本章是第二部分的收尾：如何规划一个 AI 集群到底需要多少资源？

很多团队一开始会问：“我们要买多少 GPU？”这个问题本身不完整。真正要问的是：为了支撑哪些任务画像、在什么 SLO、成本和增长假设下，需要多少 GPU、什么型号、多少网络带宽、多少存储吞吐、多少冗余，以及如何保证利用率？

先记住一句话：

> AI 集群容量规划不是按预算买卡，而是按工作负载、性能目标、增长预期、故障冗余和成本约束设计一整套计算、网络、存储和调度容量。

## 15.0 本讲资料边界与第二轮精修口径

第二轮精修时，本章按 `WRITING_PLAN.md` 的要求做了联网资料校准，主要参考 Kubernetes 对 resource request / limit、extended resource、ResourceQuota 和 HorizontalPodAutoscaler 的稳定口径，参考 NVIDIA DCGM 对 GPU profiling metrics、SM activity、显存带宽和 PCIe 传输指标的定义，参考 Google SRE 对 overload、按资源而不是只按 QPS 建模、quota、降级和限流的工程经验。

本章只抽象 AI Infra 集群容量规划的稳定工程问题：如何把 GPU 数量、GPU 型号、训练队列、推理峰值、网络、存储、利用率、故障冗余、增长预测和成本预算放进同一张容量表。它不绑定某个云厂商报价、某一代 GPU 性能参数、某个内部调度器实现或某个固定采购流程。

本章第二轮补强重点有三点：

1. 把原来文字化的 GPU-hours、推理容量、网络和存储估算改成可复用公式。
2. 明确容量规划不能只看平均 GPU utilization，而要同时看 SLO、峰值、余量、成本和预测偏差。
3. 新增一个 0 依赖 Python demo，用 toy case 演示如何把完整容量计划和 16 类失败案例做成容量规划门禁。

## 15.1 为什么容量规划重要

容量规划做不好，会出现两类问题。

第一类是资源不足：

1. 训练任务长期排队。
2. 推理高峰延迟上升。
3. 评估任务无法按时完成。
4. 新项目无法启动。
5. 抢占过于频繁。
6. 团队为了资源互相冲突。

第二类是资源浪费：

1. GPU 闲置。
2. 高端 GPU 被低价值任务占用。
3. 网络和存储过度配置。
4. checkpoint 和日志存储膨胀。
5. 推理实例过度冗余。
6. 成本高但产出低。

容量规划的目标是在“缺资源”和“浪费资源”之间找到可治理的平衡。

## 15.2 容量规划不是只算 GPU

AI 集群容量包括：

1. GPU 数量。
2. GPU 型号和显存。
3. CPU 和内存。
4. 机内互联。
5. 机间网络。
6. 本地盘。
7. 共享存储。
8. 对象存储。
9. checkpoint 吞吐。
10. 模型加载带宽。
11. 推理服务容量。
12. 监控和日志容量。

只买 GPU 不配网络和存储，会导致 GPU 等通信或等数据。

只规划训练不规划推理，会导致模型训出来但上线扛不住流量。

只规划平均负载不规划峰值，会导致关键时刻排队或超时。

## 15.3 从任务画像开始

容量规划第一步是拆任务画像。

至少要分：

1. 预训练。
2. SFT。
3. RLHF / RLAIF。
4. 离线评估。
5. 在线推理。
6. RAG / embedding。
7. Agent。
8. 交互式开发。

每类任务要估算：

1. 每个任务需要多少 GPU。
2. 任务运行多久。
3. 每月任务数量。
4. 是否有截止时间。
5. 是否可抢占。
6. 是否需要高端 GPU。
7. 是否需要高速网络。
8. 是否需要大量存储读写。

没有任务画像，容量规划就是拍脑袋。

## 15.3.1 关键公式与容量规划速查

先把一段时间窗口内的训练需求折算成 GPU-hours：

```math
H_{\mathrm{gpu}}=\sum_{k=1}^{K} n_k g_k t_k r_k
```

其中，`k` 是任务类型，`n_k` 是窗口内任务数，`g_k` 是单任务 GPU 数，`t_k` 是单任务小时数，`r_k` 是重试、失败恢复和实验返工系数。

每张 GPU 在窗口内能提供的有效小时数为：

```math
C_{\mathrm{one}}=D\cdot 24\cdot u_{\mathrm{eff}}
```

其中，`D` 是窗口天数，`u_eff` 是目标有效利用率，不是 DCGM 或 `nvidia-smi` 里看到的瞬时 GPU utilization。容量规划中更关心“这张卡是否在完成有价值工作”。

考虑峰值、故障、碎片和增长之后，训练 GPU 需求可以写成：

```math
G_{\mathrm{need}}=\left\lceil \frac{H_{\mathrm{gpu}}}{D\cdot 24\cdot u_{\mathrm{eff}}}\left(1+h_{\mathrm{peak}}+h_{\mathrm{fail}}+h_{\mathrm{frag}}+h_{\mathrm{growth}}\right) \right\rceil
```

其中，`h_peak` 是峰值余量，`h_fail` 是故障和维护余量，`h_frag` 是调度碎片余量，`h_growth` 是增长余量。

推理实例数不能只按平均 QPS 算。一个简化估算是：

```math
I_{\mathrm{serve}}=\left\lceil \frac{\lambda_{\mathrm{peak}}(L_{\mathrm{in}}+L_{\mathrm{out}})}{\mu_{\mathrm{inst}}\eta_{\mathrm{slo}}} \right\rceil
```

其中，`\lambda_peak` 是峰值 QPS，`L_in` 和 `L_out` 分别是输入和输出 token 数，`\mu_inst` 是单实例在目标延迟下可稳定提供的 token/s，`\eta_slo` 是为了满足 TTFT、TPOT 和 P99 延迟保留的折扣系数。

训练网络容量至少要覆盖关键 collective 的通信量：

```math
B_{\mathrm{net}}\ge \frac{V_{\mathrm{comm}}}{T_{\mathrm{step}}\eta_{\mathrm{net}}}
```

其中，`V_comm` 是每步通信量，`T_step` 是目标 step time，`\eta_net` 是网络有效效率。实际规划还要看 RDMA、GPU-NIC 亲和性、机柜内外 oversubscription、重传和 rank skew。

存储吞吐下限可以取多条路径的最大值：

```math
B_{\mathrm{store}}\ge \max\left(B_{\mathrm{data}},\frac{S_{\mathrm{ckpt}}}{T_{\mathrm{ckpt}}},\frac{S_{\mathrm{model}}}{T_{\mathrm{load}}}\right)
```

其中，`B_data` 是训练数据读取吞吐，`S_ckpt / T_ckpt` 是 checkpoint 写入吞吐，`S_model / T_load` 是模型权重加载吞吐。

利用率目标必须留下弹性：

```math
u_{\mathrm{target}}\le 1-h_{\mathrm{burst}}-h_{\mathrm{maint}}-h_{\mathrm{fail}}
```

如果长期目标利用率高到没有 burst、维护和故障余量，表面上省钱，实际会导致排队、抢占、推理扩容失败和故障恢复困难。

滚动预测要追踪预测偏差：

```math
E_{\mathrm{forecast}}=\frac{|D_{\mathrm{actual}}-D_{\mathrm{forecast}}|}{D_{\mathrm{forecast}}}
```

其中，`D_actual` 是实际需求，`D_forecast` 是预测需求。容量评审不是写完一次文档，而是每月或每季度用预测偏差修正下一轮采购、云上弹性和配额策略。

最后可以把容量规划门禁写成：

```math
G_{\mathrm{capacity}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land G_{\mathrm{actual}}\ge G_{\mathrm{need}} \land I_{\mathrm{actual}}\ge I_{\mathrm{serve}} \land B_{\mathrm{net}}\ge \beta_{\mathrm{net}} \land B_{\mathrm{store}}\ge \beta_{\mathrm{store}} \land K_{\mathrm{est}}\le K_{\mathrm{budget}} \land P_0=0\right]
```

其中，`C_j` 是第 `j` 个治理检查的覆盖率，`G_actual` 是实际可用 GPU 数，`I_actual` 是实际推理实例数，`K_est` 是估算成本，`K_budget` 是预算，`P_0` 是硬阻断数量。

## 15.4 GPU 数量估算

GPU 数量可以从 GPU-hours 估算。

例如一个任务需要 64 张 GPU 训练 48 小时，那么消耗：

```text
64 * 48 = 3072 GPU-hours
```

如果一个月有 20 个类似任务，需要：

```text
3072 * 20 = 61440 GPU-hours
```

如果集群希望平均有效利用率为 60%，一个月按 30 天计算，每张 GPU 可提供：

```text
30 * 24 * 0.60 = 432 effective GPU-hours
```

所需 GPU 数约为：

```text
61440 / 432 ≈ 143 GPUs
```

这只是粗略估算，还要加上：

1. 峰值需求。
2. 排队容忍度。
3. 故障冗余。
4. 调度碎片。
5. 资源池隔离。
6. 增长预期。

## 15.5 平均负载和峰值负载

只按平均负载规划会低估容量。

训练任务可能集中在：

1. 发布前。
2. 论文截止前。
3. 大版本迭代前。
4. 数据刷新后。
5. 新模型启动阶段。

推理流量也有峰值：

1. 白天高峰。
2. 活动期间。
3. 新产品发布。
4. 批量任务集中提交。
5. 外部事件触发。

容量规划要同时看：

1. average demand。
2. p95 demand。
3. peak demand。
4. burst duration。
5. 可延迟任务比例。
6. 可降级任务比例。

训练任务可以排队，推理请求通常不能长时间排队。这两类容量要分开考虑。

## 15.6 GPU 型号规划

不是所有 GPU 都适合所有任务。

需要按任务匹配 GPU 型号。

高端大显存 GPU 适合：

1. 预训练。
2. 大模型 SFT。
3. 长上下文推理。
4. 大 batch 推理。
5. 张量并行任务。

中端 GPU 适合：

1. 小模型 SFT。
2. embedding。
3. rerank。
4. 离线评估。
5. 小模型推理。

低成本或可抢占资源适合：

1. 实验探索。
2. 批量评估。
3. 数据处理。
4. 低优先级训练。

容量规划不能只问“买多少卡”，还要问“每种卡买多少”。

## 15.7 训练容量规划

训练容量规划要考虑：

1. 预训练计划。
2. SFT 并发实验数。
3. RLHF rollout 需求。
4. 评估任务穿插。
5. 失败重试。
6. checkpoint 开销。
7. 调度碎片。
8. 团队配额。

一个训练容量表可以这样设计：

| 任务类型 | 单任务 GPU | 单任务时长 | 月任务数 | GPU-hours | 是否可抢占 |
| --- | --- | --- | --- | --- | --- |
| 7B SFT | 8 | 10h | 80 | 6400 | 是 |
| 70B SFT | 64 | 24h | 10 | 15360 | 否 |
| Eval | 4 | 6h | 200 | 4800 | 是 |
| Pretrain | 256 | 120h | 2 | 61440 | 否 |

表格能帮助你从需求倒推容量，而不是凭感觉估算。

## 15.8 推理容量规划

推理容量规划和训练不同。

训练按 GPU-hours 估算较多，推理要按流量和延迟估算。

关键指标：

1. QPS。
2. 输入 token 长度。
3. 输出 token 长度。
4. TTFT。
5. TPOT。
6. 并发。
7. batch size。
8. KV cache 显存。
9. p95 / p99 延迟。
10. 可用性目标。

推理容量粗略流程：

1. 估算每秒输入和输出 token。
2. 测量单卡或单实例吞吐。
3. 根据延迟 SLO 确定可接受 batch。
4. 估算需要多少实例。
5. 加上峰值和冗余。
6. 规划自动扩缩容。

推理不能只看平均 QPS。长上下文请求、长输出请求和短请求的资源消耗差异很大。

## 15.9 网络容量规划

网络容量规划要看通信模式。

训练网络需求取决于：

1. 数据并行规模。
2. 张量并行是否跨机。
3. FSDP / ZeRO 通信量。
4. Pipeline 并行 stage 放置。
5. checkpoint 写入路径。
6. 数据读取路径。

网络规划要考虑：

1. 单节点网卡带宽。
2. 机柜内带宽。
3. 机柜间带宽。
4. oversubscription ratio。
5. 多任务并发拥塞。
6. RDMA 支持。
7. 网络故障冗余。

如果预训练任务要求高扩展效率，网络不能只按普通业务流量规划。

集合通信会造成同步大流量，尾延迟会拖慢整个训练。

## 15.10 存储容量和吞吐规划

存储要同时规划容量和吞吐。

容量包括：

1. 原始数据。
2. 清洗数据。
3. 训练 shard。
4. checkpoint。
5. 模型权重。
6. eval report。
7. 日志和 trace。
8. RAG 文档和索引。

吞吐包括：

1. 训练数据读取吞吐。
2. checkpoint 写入吞吐。
3. checkpoint 恢复读取吞吐。
4. 模型权重加载吞吐。
5. 多任务并发读写吞吐。

例如 checkpoint 容量估算：

```text
单个 checkpoint 大小 * 保留数量 * 实验数量
```

如果单个训练状态 checkpoint 是 2TB，每个实验保留 10 个，10 个实验并发，就是：

```text
2TB * 10 * 10 = 200TB
```

这还不包括模型发布权重、日志和中间数据。

## 15.11 利用率目标

GPU 利用率不是越高越好。

如果目标利用率设得太高，例如长期 95%，会导致：

1. 新任务排队严重。
2. 高优先级任务没有弹性空间。
3. 推理峰值无法扩容。
4. 维护和故障恢复困难。
5. 抢占频繁。

如果利用率太低，成本浪费。

合理目标要按资源池区分：

1. 预训练资源池：追求较高利用率，但保留维护窗口。
2. 推理资源池：需要冗余，不能压满。
3. 评估资源池：可以使用低优先级资源，提高利用率。
4. 交互式资源池：需要快速响应，利用率不能太满。

容量规划要区分“分配率”和“有效利用率”。GPU 被分配了不代表有效训练。

## 15.12 冗余和故障预留

生产系统必须预留冗余。

原因：

1. GPU 故障。
2. 节点维护。
3. 网络故障。
4. 存储故障。
5. 推理峰值。
6. 突发任务。
7. 灰度发布。

冗余策略：

1. N+1 节点冗余。
2. 推理服务多副本。
3. 跨机柜部署。
4. 关键 checkpoint 多副本。
5. 高优任务预留资源。
6. 低优任务可抢占。

冗余会降低表面利用率，但提高可靠性和业务连续性。

## 15.13 增长预测

AI 资源需求增长通常很快。

容量规划要看：

1. 模型规模增长。
2. context length 增长。
3. 用户流量增长。
4. 团队数量增长。
5. 实验数量增长。
6. 数据规模增长。
7. Agent 调用链增长。
8. 多模态数据增长。

常见做法：

1. 做 3 个月、6 个月、12 个月预测。
2. 按保守、中性、激进三种场景规划。
3. 设置触发扩容阈值。
4. 定期回顾实际用量和预测偏差。

容量规划不是一次性文档，而是持续滚动过程。

## 15.14 云上、混合云和自建容量

容量来源可以是：

1. 自建集群。
2. 公有云 GPU。
3. 混合云。
4. 外部推理 API。
5. 低优先级或竞价资源。

自建适合稳定长期大规模需求。

云上适合弹性峰值、临时实验和快速扩容。

混合云适合把基础负载放自建，把突发负载放云上。

但混合云要注意：

1. 数据迁移成本。
2. 网络延迟。
3. 安全和合规。
4. 镜像和环境一致性。
5. 成本监控。
6. 调度系统统一视图。

## 15.15 容量规划指标体系

一个 AI 集群容量 dashboard 应该包含：

1. GPU 总量。
2. GPU 分配率。
3. GPU 有效利用率。
4. 各队列 pending time。
5. 各租户 GPU-hours。
6. GPU 碎片率。
7. 抢占次数。
8. 推理 QPS 和延迟。
9. 推理实例利用率。
10. 网络吞吐和拥塞。
11. 存储吞吐。
12. checkpoint 容量。
13. 日志和 trace 增长。
14. 成本趋势。
15. 预测需求和实际需求偏差。

没有这些指标，容量规划只能靠感觉。

## 15.16 常见容量规划错误

错误一：只按当前需求买 GPU。

没有考虑增长、峰值和故障冗余。

错误二：只买最高端 GPU。

很多评估、embedding、小模型推理和开发任务不需要最高端 GPU。

错误三：只看 GPU 数量，不看网络和存储。

训练可能被通信和 I/O 拖慢。

错误四：把训练和推理混在一起规划。

训练可以排队，推理需要 SLO 和冗余。

错误五：追求 100% 利用率。

这会牺牲弹性、可靠性和高优任务响应。

错误六：不做成本归因。

不知道谁在用资源，就无法优化。

## 15.17 集群容量规划审计指标与最小 demo

下面这个 demo 不模拟真实 GPU 集群，而是演示容量规划评审应该怎么结构化：一个完整样本要同时通过任务画像、GPU 数量、GPU 型号、训练队列、推理峰值、网络、存储、生命周期、利用率余量、故障冗余、增长预测、成本、资源池隔离、quota / burst 和观测回填；16 个 bad case 分别故意打破一个门禁。

```python
from math import ceil


METRICS = [
    "workload_forecast_coverage",
    "gpu_count_capacity_fit",
    "gpu_type_mix_fit",
    "training_queue_slo_fit",
    "serving_peak_slo_fit",
    "network_bandwidth_capacity_fit",
    "storage_throughput_capacity_fit",
    "storage_capacity_lifecycle_fit",
    "utilization_headroom_fit",
    "failure_redundancy_fit",
    "growth_forecast_fit",
    "cost_budget_fit",
    "pool_separation_fit",
    "quota_burst_governance",
    "observability_forecast_feedback",
    "cluster_capacity_planning_gate",
]


def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * pct / 100.0))
    return ordered[index]


def required_training_gpus(case):
    base = case["monthly_gpu_hours"] / (
        case["days"] * 24.0 * case["target_effective_utilization"]
    )
    multiplier = (
        1.0
        + case["peak_headroom"]
        + case["failure_headroom"]
        + case["fragmentation_headroom"]
        + case["growth_headroom"]
    )
    return int(ceil(base * multiplier))


def required_serving_instances(case):
    peak_tokens_per_s = (
        case["peak_qps"] * (case["input_tokens_p95"] + case["output_tokens_p95"])
    )
    stable_tokens_per_s = (
        case["instance_token_capacity_per_s"] * case["slo_efficiency"]
    )
    return int(ceil(peak_tokens_per_s / stable_tokens_per_s))


def required_network_gibs(case):
    return case["comm_gib_per_step"] / (
        case["target_step_seconds"] * case["network_efficiency"]
    )


def required_storage_gibs(case):
    checkpoint_write = (
        case["checkpoint_tib"] * 1024.0 / case["checkpoint_window_seconds"]
    )
    model_load = case["model_tib"] * 1024.0 / case["model_load_window_seconds"]
    return max(case["data_read_gibs"], checkpoint_write, model_load)


def checkpoint_capacity_tib(case):
    return (
        case["checkpoint_tib"]
        * case["retained_checkpoints"]
        * case["concurrent_experiments"]
    )


def complete_case():
    return {
        "name": "complete_capacity_plan",
        "workload_profiles": True,
        "monthly_gpu_hours": 61440,
        "days": 30,
        "target_effective_utilization": 0.60,
        "peak_headroom": 0.20,
        "failure_headroom": 0.10,
        "fragmentation_headroom": 0.08,
        "growth_headroom": 0.15,
        "actual_gpus": 220,
        "gpu_type_mix_ok": True,
        "queue_wait_minutes": [22, 30, 36, 40, 44],
        "queue_slo_minutes": 60,
        "peak_qps": 42,
        "input_tokens_p95": 900,
        "output_tokens_p95": 220,
        "instance_token_capacity_per_s": 9000,
        "slo_efficiency": 0.72,
        "actual_serving_instances": 10,
        "serving_p99_seconds": 1.6,
        "serving_p99_slo_seconds": 2.0,
        "comm_gib_per_step": 120,
        "target_step_seconds": 40,
        "network_efficiency": 0.75,
        "provisioned_network_gibs": 5.2,
        "data_read_gibs": 3.5,
        "checkpoint_tib": 6.0,
        "checkpoint_window_seconds": 1500,
        "model_tib": 1.5,
        "model_load_window_seconds": 300,
        "provisioned_storage_gibs": 8.0,
        "storage_measured": True,
        "retained_checkpoints": 12,
        "concurrent_experiments": 8,
        "total_checkpoint_capacity_tib": 900,
        "lifecycle_policy": True,
        "burst_headroom": 0.18,
        "maintenance_headroom": 0.08,
        "n_plus_one_ready": True,
        "forecast_period_months": 12,
        "forecast_error": 0.09,
        "estimated_monthly_cost_usd": 470000,
        "monthly_budget_usd": 550000,
        "pool_separation": True,
        "quota_ledger": True,
        "burst_credits": True,
        "observability_metrics": {
            "gpu_allocated",
            "gpu_effective_utilization",
            "queue_wait_p95",
            "serving_p99",
            "network_throughput",
            "storage_throughput",
            "checkpoint_growth",
            "forecast_error",
            "cost_by_tenant",
            "fragmentation_ratio",
        },
        "forecast_review_days": 30,
        "capacity_gate_defined": True,
    }


def build_cases():
    cases = [complete_case()]
    mutations = [
        ("workload_missing_bad", {"workload_profiles": False}),
        ("gpu_count_underplanned_bad", {"actual_gpus": 180}),
        ("gpu_type_mismatch_bad", {"gpu_type_mix_ok": False}),
        ("queue_slo_missing_bad", {"queue_wait_minutes": [70, 80, 95]}),
        ("serving_peak_ignored_bad", {"actual_serving_instances": 6}),
        ("network_underprovisioned_bad", {"provisioned_network_gibs": 2.0}),
        ("storage_throughput_unknown_bad", {"storage_measured": False}),
        ("lifecycle_capacity_missing_bad", {"total_checkpoint_capacity_tib": 500}),
        ("target_util_too_high_bad", {"target_effective_utilization": 0.88}),
        ("no_failure_redundancy_bad", {"n_plus_one_ready": False}),
        ("growth_forecast_missing_bad", {"forecast_period_months": 1}),
        ("cost_budget_missing_bad", {"estimated_monthly_cost_usd": 620000}),
        ("pools_mixed_bad", {"pool_separation": False}),
        ("burst_quota_uncontrolled_bad", {"burst_credits": False}),
        ("observability_missing_bad", {"observability_metrics": {"gpu_allocated"}}),
        ("capacity_gate_missing_bad", {"capacity_gate_defined": False}),
    ]
    for name, patch in mutations:
        item = complete_case()
        item["name"] = name
        item.update(patch)
        cases.append(item)
    return cases


def evaluate_case(case):
    required_gpus = required_training_gpus(case)
    required_instances = required_serving_instances(case)
    network_need = required_network_gibs(case)
    storage_need = required_storage_gibs(case)
    checkpoint_need = checkpoint_capacity_tib(case)
    max_safe_util = 1.0 - (
        case["burst_headroom"]
        + case["maintenance_headroom"]
        + case["failure_headroom"]
    )
    required_metrics = complete_case()["observability_metrics"]

    gates = {
        "workload_forecast_coverage": case["workload_profiles"],
        "gpu_count_capacity_fit": case["actual_gpus"] >= required_gpus,
        "gpu_type_mix_fit": case["gpu_type_mix_ok"],
        "training_queue_slo_fit": percentile(case["queue_wait_minutes"], 95)
        <= case["queue_slo_minutes"],
        "serving_peak_slo_fit": (
            case["actual_serving_instances"] >= required_instances
            and case["serving_p99_seconds"] <= case["serving_p99_slo_seconds"]
        ),
        "network_bandwidth_capacity_fit": (
            case["provisioned_network_gibs"] >= network_need * 1.1
        ),
        "storage_throughput_capacity_fit": (
            case["storage_measured"]
            and case["provisioned_storage_gibs"] >= storage_need * 1.1
        ),
        "storage_capacity_lifecycle_fit": (
            case["total_checkpoint_capacity_tib"] >= checkpoint_need
            and case["lifecycle_policy"]
        ),
        "utilization_headroom_fit": (
            case["target_effective_utilization"] <= max_safe_util
        ),
        "failure_redundancy_fit": (
            case["failure_headroom"] >= 0.08 and case["n_plus_one_ready"]
        ),
        "growth_forecast_fit": (
            case["forecast_period_months"] >= 6 and case["forecast_error"] <= 0.20
        ),
        "cost_budget_fit": (
            case["estimated_monthly_cost_usd"] <= case["monthly_budget_usd"]
        ),
        "pool_separation_fit": case["pool_separation"],
        "quota_burst_governance": case["quota_ledger"] and case["burst_credits"],
        "observability_forecast_feedback": (
            required_metrics.issubset(case["observability_metrics"])
            and case["forecast_review_days"] <= 45
        ),
        "cluster_capacity_planning_gate": case["capacity_gate_defined"],
    }
    return {
        "name": case["name"],
        "required_gpus": required_gpus,
        "required_instances": required_instances,
        "network_need_gibs": network_need,
        "storage_need_gibs": storage_need,
        "checkpoint_need_tib": checkpoint_need,
        "gates": gates,
        "pass": all(gates.values()),
    }


def audit_cluster_capacity_planning(cases):
    results = [evaluate_case(case) for case in cases]
    metrics = {}
    for metric in METRICS:
        passed = sum(1 for result in results if result["gates"][metric])
        metrics[metric] = round(passed / len(results), 3)

    failed_cases = [result["name"] for result in results if not result["pass"]]
    failed_gates = [
        metric for metric in METRICS if any(not r["gates"][metric] for r in results)
    ]
    complete = results[0]
    examples = {
        "training_gpu_need": complete["required_gpus"],
        "serving_instance_need": complete["required_instances"],
        "network_margin": round(
            complete_case()["provisioned_network_gibs"]
            / complete["network_need_gibs"],
            3,
        ),
        "storage_margin": round(
            complete_case()["provisioned_storage_gibs"]
            / complete["storage_need_gibs"],
            3,
        ),
        "checkpoint_capacity_tib": complete["checkpoint_need_tib"],
        "target_effective_utilization": complete_case()[
            "target_effective_utilization"
        ],
    }
    smoke = {
        "complete_case_passes": complete["pass"],
        "caught_underplanned_gpu": "gpu_count_underplanned_bad" in failed_cases,
        "caught_serving_peak": "serving_peak_ignored_bad" in failed_cases,
        "caught_network_gap": "network_underprovisioned_bad" in failed_cases,
        "caught_storage_gap": "storage_throughput_unknown_bad" in failed_cases,
        "caught_headroom_gap": "target_util_too_high_bad" in failed_cases,
    }
    return {
        "capacity_examples": examples,
        "smoke": smoke,
        "metrics": metrics,
        "hard_blocker_count": len(failed_cases),
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "cluster_capacity_planning_gate_pass": all(
            result["pass"] for result in results
        ),
    }


report = audit_cluster_capacity_planning(build_cases())
print("capacity_examples=", report["capacity_examples"], sep="")
print("smoke=", report["smoke"], sep="")
print("metrics=", report["metrics"], sep="")
print("hard_blocker_count=", report["hard_blocker_count"], sep="")
print("failed_cases=", report["failed_cases"], sep="")
print("failed_gates=", report["failed_gates"], sep="")
print(
    "cluster_capacity_planning_gate_pass=",
    report["cluster_capacity_planning_gate_pass"],
    sep="",
)
```

运行后会看到类似输出：

```text
capacity_examples={'training_gpu_need': 218, 'serving_instance_need': 8, 'network_margin': 1.3, 'storage_margin': 1.562, 'checkpoint_capacity_tib': 576.0, 'target_effective_utilization': 0.6}
smoke={'complete_case_passes': True, 'caught_underplanned_gpu': True, 'caught_serving_peak': True, 'caught_network_gap': True, 'caught_storage_gap': True, 'caught_headroom_gap': True}
metrics={'workload_forecast_coverage': 0.941, 'gpu_count_capacity_fit': 0.941, 'gpu_type_mix_fit': 0.941, 'training_queue_slo_fit': 0.941, 'serving_peak_slo_fit': 0.941, 'network_bandwidth_capacity_fit': 0.941, 'storage_throughput_capacity_fit': 0.941, 'storage_capacity_lifecycle_fit': 0.941, 'utilization_headroom_fit': 0.941, 'failure_redundancy_fit': 0.941, 'growth_forecast_fit': 0.941, 'cost_budget_fit': 0.941, 'pool_separation_fit': 0.941, 'quota_burst_governance': 0.941, 'observability_forecast_feedback': 0.941, 'cluster_capacity_planning_gate': 0.941}
hard_blocker_count=16
failed_cases=['workload_missing_bad', 'gpu_count_underplanned_bad', 'gpu_type_mismatch_bad', 'queue_slo_missing_bad', 'serving_peak_ignored_bad', 'network_underprovisioned_bad', 'storage_throughput_unknown_bad', 'lifecycle_capacity_missing_bad', 'target_util_too_high_bad', 'no_failure_redundancy_bad', 'growth_forecast_missing_bad', 'cost_budget_missing_bad', 'pools_mixed_bad', 'burst_quota_uncontrolled_bad', 'observability_missing_bad', 'capacity_gate_missing_bad']
failed_gates=['workload_forecast_coverage', 'gpu_count_capacity_fit', 'gpu_type_mix_fit', 'training_queue_slo_fit', 'serving_peak_slo_fit', 'network_bandwidth_capacity_fit', 'storage_throughput_capacity_fit', 'storage_capacity_lifecycle_fit', 'utilization_headroom_fit', 'failure_redundancy_fit', 'growth_forecast_fit', 'cost_budget_fit', 'pool_separation_fit', 'quota_burst_governance', 'observability_forecast_feedback', 'cluster_capacity_planning_gate']
cluster_capacity_planning_gate_pass=False
```

这里 `cluster_capacity_planning_gate_pass=False` 不是 demo 出错，而是因为我们故意放入了 16 个 bad case。面试里你要强调：容量规划不是“GPU 数字算对就行”，而是要证明需求画像、训练容量、推理峰值、网络、存储、余量、成本、隔离和预测回填都能被观测和治理。

## 15.18 面试中如何回答容量规划题

如果面试官问：

```text
如何规划一个企业大模型 GPU 集群容量？
```

可以这样回答：

```text
我会先从任务画像出发，把需求拆成预训练、SFT、评估、推理、RAG、Agent 和交互式开发。对离线训练任务，用 GPU-hours 估算月度需求；对在线推理任务，用 QPS、输入输出 token、TTFT、TPOT、p95/p99 延迟和单实例吞吐估算实例数。

然后按 GPU 型号分层规划，高端大显存 GPU 给预训练、大模型 SFT 和长上下文推理，中端 GPU 给评估、embedding、rerank 和小模型推理，低优先级资源给可抢占任务。

同时规划网络和存储。网络要看多机训练通信模式、RDMA、机柜内外带宽和 oversubscription；存储要看训练数据读取、checkpoint 写入和恢复、模型权重加载、日志和 trace 增长。

容量上不能只按平均负载，要看 p95 和峰值，保留故障冗余、推理扩容空间和高优任务预留。最后通过 dashboard 持续跟踪 GPU 分配率、有效利用率、队列等待时间、碎片率、抢占次数、存储增长和成本归因，并定期滚动预测 3 到 12 个月需求。
```

## 15.19 常见误区

误区一：容量规划就是买多少 GPU。

还要规划网络、存储、调度、推理冗余、checkpoint、日志和成本。

误区二：GPU 利用率越高越好。

过高利用率会导致无弹性、排队严重和故障恢复困难。

误区三：高端 GPU 适合所有任务。

不同任务应匹配不同 GPU 型号，否则会浪费。

误区四：推理容量可以按平均 QPS 算。

推理要看峰值、长尾延迟、token 长度分布、KV cache 和扩容时间。

误区五：存储容量够就行。

训练还需要吞吐，checkpoint 还需要恢复速度，模型加载还影响扩容。

## 15.20 面试题

### 题 1：如何用 GPU-hours 估算训练容量？

答：先估算每类任务的单任务 GPU 数、运行时长和月任务数，得到总 GPU-hours。再除以每张 GPU 每月可提供的有效 GPU-hours，得到所需 GPU 数。之后还要加上峰值、故障冗余、调度碎片和增长预期。

### 题 2：推理容量规划和训练容量规划有什么不同？

答：训练主要按 GPU-hours、任务并发和等待时间规划；推理是在线服务，要按 QPS、输入输出 token、TTFT、TPOT、p95/p99 延迟、KV cache、可用性和自动扩缩容规划。

### 题 3：为什么容量规划要区分 GPU 型号？

答：不同任务对显存、带宽、互联和成本要求不同。预训练和大模型推理需要高端大显存 GPU，而评估、embedding、小模型推理和开发任务可能使用中端或低成本 GPU 更划算。

### 题 4：为什么不能追求长期 100% GPU 利用率？

答：长期满载会导致任务排队、高优任务无资源、推理无法扩容、维护困难、故障恢复没有冗余。生产平台需要保留弹性和冗余。

### 题 5：容量 dashboard 应该看哪些指标？

答：包括 GPU 总量、分配率、有效利用率、队列等待时间、碎片率、抢占次数、各租户 GPU-hours、推理 QPS 和延迟、网络吞吐、存储吞吐、checkpoint 容量、日志增长和成本趋势。

## 15.21 小练习

练习一：估算训练 GPU 数量。

假设每月有 10 个任务，每个任务使用 32 张 GPU 训练 24 小时，目标有效利用率 60%。估算至少需要多少 GPU，不考虑峰值和冗余。

练习二：设计一个推理容量表。

要求：包含模型名称、QPS、平均输入 token、平均输出 token、p95 延迟目标、单实例吞吐、实例数和冗余比例。

练习三：规划 checkpoint 存储容量。

假设单个 checkpoint 1.5TB，每个实验保留 8 个，同时有 12 个实验。估算容量，并说明如何做生命周期管理。

练习四：设计一个容量规划 dashboard。

要求：包含 GPU、网络、存储、推理、队列、租户和成本指标。

## 15.22 本章小结

本章讲了集群容量规划。

你需要掌握：

1. 容量规划不是按预算买卡，而是按任务画像、SLO、成本和增长设计资源。
2. 需要同时规划 GPU、网络、存储、推理容量、冗余和调度策略。
3. 训练容量可以用 GPU-hours 粗略估算，但要考虑峰值、碎片、故障和增长。
4. 推理容量要看 QPS、token 长度、TTFT、TPOT、延迟、KV cache 和扩缩容。
5. GPU 型号要按任务匹配，不能所有任务都用最高端资源。
6. 网络容量要考虑集合通信、RDMA、机柜内外带宽和拥塞。
7. 存储容量和吞吐都重要，checkpoint、模型加载、日志和 trace 都要纳入规划。
8. 利用率目标要按资源池区分，不能盲目追求 100%。
9. 容量规划需要 dashboard、成本归因和滚动预测。
10. 集群容量规划门禁要把任务画像、训练 GPU 数、推理峰值、网络、存储、生命周期、余量、成本、资源池隔离和预测偏差放在一起审计。

第二部分到这里结束。下一章开始进入第三部分：大模型训练平台工程。我们会从第 16 章“训练平台总览：从脚本训练到平台化训练”开始，讲如何把训练从手工脚本变成可提交、可复现、可监控、可恢复的平台能力。
