# 第 55 章 设计一个多租户 GPU 集群调度系统

上一章设计了企业级 LLMOps 平台。本章继续系统设计题：设计一个多租户 GPU 集群调度系统。

GPU 集群调度是 AI Infra 的核心难题之一。大模型训练需要多机多卡、推理需要低延迟和弹性、评估和数据处理也会抢资源。多租户共享集群时，还要处理公平性、优先级、配额、抢占、碎片和成本。

先记住一句话：

> 多租户 GPU 调度系统的核心，是在有限 GPU 资源下，同时满足公平性、优先级、拓扑约束、任务完整性、利用率和 SLO。

## 55.0 本讲资料边界与第二轮精修口径

第二轮精修时，本章按“稳定调度抽象 + 主流开源实现口径”校准，不绑定某个公司内部平台或某个 Kubernetes 发行版。

参考口径包括：

1. Kubernetes 的 ResourceQuota、Device Plugin / Extended Resource、PriorityClass 和 Pod preemption，用来校准资源声明、命名空间配额、设备暴露和优先级抢占边界。
2. Kueue 的 ClusterQueue、LocalQueue、cohort 借用 / 出借、preemption 和 topology-aware scheduling，用来校准队列、配额借用、准入控制和拓扑感知调度边界。
3. Volcano 的 Queue、PodGroup、gang scheduling、priority 和 preemption，用来校准批训练任务整体调度、队列治理和抢占口径。
4. Dominant Resource Fairness 的 dominant share 思想，用来解释为什么多租户公平不能只看 GPU 数，还要同时看 CPU、内存、网络、存储和稀缺 GPU 型号。

本章的目标不是教你背某个调度器参数，而是把面试题回答成可验证的系统设计：任务资源画像清楚，队列和配额可解释，借用可回收，gang scheduling 能防止半启动，拓扑放置能减少通信损耗，抢占和 checkpoint 有耦合，低优先级资源可回收，推理服务有 SLO 隔离，异构 GPU 有成本策略，调度事件能解释“为什么还没跑”。

## 55.1 题目理解

面试题可能这样问：

1. 请设计一个公司内部多租户 GPU 集群调度系统。
2. 请设计一个支持大模型训练和推理的 GPU 调度平台。
3. 如何设计支持队列、配额、抢占和 gang scheduling 的调度系统？
4. 如何提高 GPU 集群利用率并保证公平性？

这类题考的是资源调度和平台治理能力，不只是 Kubernetes 使用经验。

## 55.2 需求澄清

先问：

1. 集群规模多大？多少节点、多少 GPU？
2. GPU 是否异构？A100、H100、L40、国产加速器？
3. 任务类型有哪些？训练、推理、评估、embedding、数据处理？
4. 是否多机多卡？是否需要 gang scheduling？
5. 是否多租户？租户之间如何隔离？
6. 是否有队列和优先级？
7. 是否允许抢占？
8. 是否需要拓扑感知？
9. 是否支持低优先级队列？
10. 目标是低等待时间、高利用率还是公平性？

不同目标会导向不同调度策略。

## 55.3 核心目标

GPU 调度系统目标：

1. 支持多租户队列和配额。
2. 支持不同 GPU 类型。
3. 支持训练 gang scheduling。
4. 支持拓扑感知调度。
5. 支持优先级和抢占。
6. 支持低优先级任务填充空闲资源。
7. 减少资源碎片。
8. 提高 GPU 有效利用率。
9. 支持公平性和成本归因。
10. 支持可观测性和审计。

系统设计要同时讲调度算法、资源模型和治理机制。

## 55.4 总体架构

可以设计如下架构：

```text
Job Submission
  -> Auth / Quota Check
  -> Queue Manager
  -> Scheduler
      -> Resource Snapshot
      -> Policy Engine
      -> Placement Engine
      -> Preemption Engine
  -> Kubernetes / Cluster Manager
  -> Node / GPU Runtime
  -> Observability / Cost / Audit
```

核心模块：

1. Queue Manager：管理队列、优先级、等待任务。
2. Resource Snapshot：维护集群资源视图。
3. Policy Engine：处理配额、公平性、优先级。
4. Placement Engine：做具体放置决策。
5. Preemption Engine：处理抢占。
6. Observability：提供调度事件和资源指标。

## 55.5 资源模型

资源模型要表示：

1. GPU 类型。
2. GPU 数量。
3. GPU 显存。
4. CPU。
5. 内存。
6. 本地 NVMe。
7. 网络拓扑。
8. 节点标签。
9. 可用区。
10. 设备健康状态。

大模型训练不能只看 GPU 数量。

同样 8 张 GPU，如果跨 NUMA、跨节点或网络慢，性能差别很大。

## 55.6 任务模型

任务要声明：

1. job type。
2. tenant。
3. queue。
4. priority。
5. GPU type。
6. nodes。
7. gpus per node。
8. CPU / memory。
9. storage。
10. expected duration。
11. preemptible。
12. checkpoint support。
13. topology requirement。

训练任务、推理服务和评估任务的调度需求不同。

调度系统要理解任务类型，而不是只看 Pod spec。

## 55.7 多租户队列

队列可以按：

1. 团队。
2. 项目。
3. 优先级。
4. 任务类型。
5. GPU 类型。
6. 生产/实验环境。

划分。

队列属性包括：

1. quota。
2. priority weight。
3. max running jobs。
4. max GPUs。
5. preemptible。
6. allowed GPU types。

队列是多租户公平性的基本单位。

## 55.8 配额设计

配额分为：

1. guaranteed quota。
2. burst quota。
3. borrowed quota。

Guaranteed quota 是保证资源。

Burst quota 是允许短期超过。

Borrowed quota 是借用别人暂时不用的资源。

借用能提高利用率，但必须可回收。

当原 owner 需要资源时，借用方低优先级任务可能被抢占。

## 55.9 公平性策略

常见公平性策略：

1. FIFO。
2. weighted fair sharing。
3. dominant resource fairness。
4. priority scheduling。
5. quota-based scheduling。

FIFO 简单但不公平。

Priority scheduling 能保证高优先级任务，但可能饿死低优先级任务。

Weighted fair sharing 适合多团队共享。

实际系统通常组合使用。

## 55.10 Gang Scheduling

Gang scheduling 是分布式训练核心需求。

一个 64 卡训练任务需要所有 worker 同时启动。

如果只调度了一部分 worker，任务无法正常运行，还会占住资源。

Gang scheduling 要求：

1. 资源够才整体调度。
2. 资源不够就继续排队。
3. 启动失败要整体回滚。
4. 运行中 worker 故障要整体处理。

没有 gang scheduling，多机多卡训练会非常不稳定。

## 55.11 拓扑感知调度

拓扑感知要考虑：

1. 单机内 GPU 互联。
2. NVLink / NVSwitch。
3. PCIe 拓扑。
4. NUMA。
5. 跨节点网络。
6. 机架位置。
7. IB/RoCE 网络域。

目标是让通信密集型任务尽量放在通信更好的拓扑中。

例如优先满足单机 8 卡，再考虑跨机多卡。

## 55.12 Placement 策略

Placement 要决定任务放到哪些节点。

常见策略：

1. pack：尽量紧凑放置，减少碎片。
2. spread：分散放置，提高容错。
3. topology-aware pack。
4. data-locality-aware placement。
5. reserved placement。

训练任务通常偏向拓扑感知 pack。

推理服务可能需要 spread，提高可用性。

不同任务类型策略不同。

## 55.13 资源碎片治理

碎片导致“总资源够，但任务调度不了”。

治理方法：

1. bin packing。
2. 大任务预留。
3. 小任务避开整机资源。
4. 低优先级任务可抢占。
5. 队列重排。
6. 调度模拟。
7. defragmentation。

调度器需要可观测碎片情况。

例如显示有多少空闲 GPU 无法组成 8 卡节点。

## 55.14 抢占机制

抢占用于让高优先级任务拿到资源。

设计要点：

1. 抢占规则。
2. 候选 victim 选择。
3. graceful termination。
4. checkpoint 保存。
5. 重新入队。
6. 抢占次数限制。
7. 抢占审计。
8. 用户通知。

选择 victim 时要考虑：

1. 优先级低。
2. 可抢占。
3. 已运行时间。
4. checkpoint 状态。
5. 释放资源是否足够。

## 55.15 低优先级队列

低优先级队列用于填充空闲资源。

适合：

1. 探索实验。
2. 批量评估。
3. embedding 生成。
4. 数据处理。
5. 可恢复训练。

特点：

1. 不保证启动时间。
2. 可被抢占。
3. 成本更低。
4. 利用闲置 GPU。

低优先级队列是提高利用率的重要手段。

## 55.16 推理服务调度

推理服务调度不同于训练。

关注：

1. 长期运行。
2. 低延迟。
3. 自动扩缩容。
4. 多副本分散。
5. warm pool。
6. 滚动发布。
7. SLO guardrail。

推理服务通常不适合被随意抢占。

可以把推理和训练放在不同资源池，或设置严格优先级和隔离策略。

## 55.17 异构 GPU 调度

集群可能有不同 GPU。

调度要支持：

1. GPU 型号选择。
2. 显存需求。
3. compute capability。
4. driver / CUDA 兼容。
5. 价格差异。
6. 任务适配性。

例如训练 70B 模型可能需要 H100，embedding 批处理可以使用较低规格 GPU。

异构调度可以提高成本效率。

## 55.18 调度事件和可观测性

调度系统必须可观测。

需要记录：

1. 为什么任务排队。
2. 为什么任务无法调度。
3. 选择了哪些节点。
4. 为什么抢占某个任务。
5. 资源碎片情况。
6. 队列等待时间。
7. 配额使用情况。
8. pending job 数。
9. 调度决策耗时。

用户最常问的问题是：我的任务为什么还没跑？

调度系统必须能解释。

## 55.19 成本和利用率

调度系统要输出成本和利用率指标：

1. GPU allocation rate。
2. GPU utilization。
3. idle GPU hours。
4. queue wait time。
5. failed job GPU hours。
6. preempted GPU hours。
7. per-tenant GPU hours。
8. borrowed quota usage。
9. low-priority queue usage。

调度不仅要让任务跑，还要让平台负责人知道资源怎么被使用。

## 55.20 与 Kubernetes 的关系

可以基于 Kubernetes，但需要扩展。

K8s 提供：

1. Pod 调度。
2. 节点管理。
3. 资源声明。
4. 控制器机制。

AI 调度需要补充：

1. gang scheduling。
2. queue。
3. quota。
4. topology-aware scheduling。
5. preemption policy。
6. GPU health。
7. cost attribution。
8. TrainingJob 语义。

所以答案可以说“基于 K8s 扩展”，但不能止步于 K8s。

## 55.21 多租户 GPU 调度系统设计指标和最小 demo

系统设计题里，最容易失分的地方是只画出 Scheduler、Queue 和 Kubernetes，却没有说明怎样验收调度系统真的公平、可回收、可解释、可审计。可以把多租户 GPU 调度样本写成：

```math
S_i=(t_i,q_i,r_i,p_i,g_i,n_i,h_i,b_i,f_i,o_i,c_i,z_i)
```

其中 `t_i` 是 tenant，`q_i` 是 queue，`r_i` 是任务资源画像，`p_i` 是优先级，`g_i` 是 gang 需求，`n_i` 是拓扑需求，`h_i` 是异构 GPU 约束，`b_i` 是借用策略，`f_i` 是公平性状态，`o_i` 是可观测事件，`c_i` 是成本标签，`z_i` 是最终调度状态。

统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[g_j(S_i)=1\right]
```

其中 `g_j` 是第 `j` 个审计条件，例如队列契约、配额借用、gang、拓扑、抢占、推理 SLO 或成本归因。

队列配额占用率可以写成：

```math
U_q=\frac{G_{\mathrm{used},q}}{G_{\mathrm{quota},q}}
```

借用 GPU 数可以写成：

```math
B_q=\max\left(0,G_{\mathrm{used},q}-G_{\mathrm{guaranteed},q}\right)
```

多资源公平可以用 dominant share 表达：

```math
D_q=\max_{r\in\mathcal{R}}\frac{u_{q,r}}{c_r}
```

其中 `u_{q,r}` 是队列 `q` 在资源 `r` 上的使用量，`c_r` 是集群总容量。带权公平偏差可以写成：

```math
H_q=\frac{D_q}{w_q},\qquad \Delta_{\mathrm{fair}}=\max_q H_q-\min_q H_q
```

其中 `w_q` 是队列权重。`Delta_fair` 越大，说明同权重归一化后的 dominant share 越不均衡。

Gang scheduling 可行性可以写成：

```math
G_{\mathrm{gang}}=\mathbf{1}\left[\sum_{n=1}^{N}\left\lfloor\frac{a_n}{g_{\mathrm{pernode}}}\right\rfloor \ge n_{\mathrm{worker}}\right]
```

其中 `a_n` 是节点 `n` 的可用 GPU 数，`g_pernode` 是每个 worker 需要的 GPU 数，`n_worker` 是需要同时启动的 worker 数。

拓扑得分可以写成：

```math
S_{\mathrm{topo}}=w_1C_{\mathrm{block}}+w_2C_{\mathrm{rack}}+w_3C_{\mathrm{nic}}+w_4C_{\mathrm{health}}
```

其中 `C_block`、`C_rack`、`C_nic` 和 `C_health` 分别表示是否处在更好的 GPU / 网络拓扑域、机架域、GPU-NIC 亲和域和健康节点集合中。

碎片率可以写成：

```math
R_{\mathrm{frag}}=1-\frac{\sum_n \left\lfloor a_n/g_{\mathrm{job}}\right\rfloor g_{\mathrm{job}}}{\sum_n a_n}
```

其中 `g_job` 是典型大任务每节点需要的 GPU 数。总空闲 GPU 很多但 `R_frag` 很高，说明资源形状不适合大任务。

抢占损失可以写成：

```math
L_{\mathrm{preempt}}=\sum_{v\in\mathcal{V}}G_v\Delta t_v
```

其中 `V` 是被抢占任务集合，`G_v` 是 victim 占用 GPU 数，`\Delta t_v` 是距离最近 checkpoint 的分钟数。这个量表示最坏丢失 GPU-minutes。

有效利用率可以写成：

```math
U_{\mathrm{eff}}=\frac{H_{\mathrm{alloc}}-H_{\mathrm{idle}}-H_{\mathrm{failed}}-H_{\mathrm{blocked}}}{H_{\mathrm{alloc}}}
```

这里 `H_alloc` 是已分配 GPU-hours，`H_idle` 是分配但空闲，`H_failed` 是失败重跑浪费，`H_blocked` 是碎片或错误调度导致的无效占用。它比普通 GPU utilization 更接近“真实有效产出”。

最终门禁可以写成：

```math
G_{\mathrm{mtgpu}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land \Delta_{\mathrm{fair}}\le \delta \land R_{\mathrm{frag}}\le \rho \land L_{\mathrm{preempt}}\le \lambda \land P_0=0\right]
```

其中 `P_0` 是 P0 级硬阻断数量。这个门禁提醒你：多租户 GPU 调度不是单个算法，而是队列、配额、公平、gang、拓扑、抢占、低优先级、推理 SLO、异构成本和可观测性的综合治理。

下面是一个 0 依赖 demo。它不实现真实调度器，而是模拟面试中应具备的“系统设计审计表”：给定 toy 队列、节点和设计样本，输出配额占用、借用、dominant share、gang 可行性、拓扑得分、碎片率、抢占损失、有效利用率，并检查 16 个系统设计门禁。

```python
from math import floor


class MiniMultiTenantGPUSchedulerDesignAudit:
    def __init__(self):
        self.capacity = {"gpu": 80, "cpu": 960, "memory_gib": 4096}
        self.queues = {
            "research": {
                "guaranteed_gpus": 16,
                "quota_gpus": 32,
                "weight": 0.50,
                "usage": {"gpu": 24, "cpu": 160, "memory_gib": 900},
            },
            "product": {
                "guaranteed_gpus": 16,
                "quota_gpus": 24,
                "weight": 0.30,
                "usage": {"gpu": 14, "cpu": 120, "memory_gib": 512},
            },
            "eval": {
                "guaranteed_gpus": 8,
                "quota_gpus": 16,
                "weight": 0.20,
                "usage": {"gpu": 10, "cpu": 96, "memory_gib": 256},
            },
        }
        self.nodes = [
            {"id": "node-a1", "free_gpus": 8, "block": "nv-a", "rack": "rack-1", "nic": "ib-a", "healthy": True},
            {"id": "node-a2", "free_gpus": 8, "block": "nv-a", "rack": "rack-1", "nic": "ib-a", "healthy": True},
            {"id": "node-b1", "free_gpus": 8, "block": "nv-b", "rack": "rack-2", "nic": "ib-b", "healthy": True},
            {"id": "node-b2", "free_gpus": 4, "block": "nv-b", "rack": "rack-2", "nic": "ib-b", "healthy": True},
            {"id": "node-c1", "free_gpus": 2, "block": "pcie-c", "rack": "rack-3", "nic": "eth-c", "healthy": True},
        ]
        self.gates = [
            "tenant_queue_contract",
            "quota_borrowing_policy",
            "dominant_fairness_accounting",
            "gang_scheduling_fit",
            "topology_aware_placement",
            "fragmentation_control",
            "preemption_checkpoint_safety",
            "low_priority_reclaim",
            "inference_slo_isolation",
            "heterogeneous_gpu_policy",
            "resource_snapshot_health",
            "placement_explainability",
            "cost_utilization_attribution",
            "scheduler_observability_audit",
            "tradeoff_boundary_reasoning",
            "multi_tenant_gpu_scheduler_gate",
        ]

    def quota_usage(self):
        result = {}
        for name, queue in self.queues.items():
            result[name] = round(queue["usage"]["gpu"] / queue["quota_gpus"], 3)
        return result

    def borrowed_gpus(self):
        result = {}
        for name, queue in self.queues.items():
            borrowed = queue["usage"]["gpu"] - queue["guaranteed_gpus"]
            result[name] = max(0, borrowed)
        return result

    def dominant_shares(self):
        shares = {}
        for name, queue in self.queues.items():
            dominant = 0.0
            for resource, capacity in self.capacity.items():
                dominant = max(dominant, queue["usage"][resource] / capacity)
            shares[name] = round(dominant, 3)
        return shares

    def fairness_gap(self):
        shares = self.dominant_shares()
        normalized = []
        for name, share in shares.items():
            normalized.append(share / self.queues[name]["weight"])
        return round(max(normalized) - min(normalized), 3)

    def gang_feasible(self, worker_count, gpus_per_worker):
        fit_nodes = 0
        for node in self.nodes:
            fit_nodes += floor(node["free_gpus"] / gpus_per_worker)
        return fit_nodes >= worker_count

    def topology_score(self, selected_node_ids):
        selected = [node for node in self.nodes if node["id"] in selected_node_ids]
        if not selected:
            return 0.0
        same_block = int(len({node["block"] for node in selected}) == 1)
        same_rack = int(len({node["rack"] for node in selected}) == 1)
        same_nic = int(len({node["nic"] for node in selected}) == 1)
        all_healthy = int(all(node["healthy"] for node in selected))
        score = 0.35 * same_block + 0.25 * same_rack + 0.20 * same_nic + 0.20 * all_healthy
        return round(score, 3)

    def fragmentation_ratio(self, gpus_per_job):
        free_total = sum(node["free_gpus"] for node in self.nodes)
        if free_total == 0:
            return 0.0
        packable = sum(floor(node["free_gpus"] / gpus_per_job) * gpus_per_job for node in self.nodes)
        return round(1 - packable / free_total, 3)

    def preemption_loss_gpu_minutes(self):
        victims = [
            {"job": "low-priority-train", "gpus": 8, "minutes_since_checkpoint": 12},
            {"job": "batch-eval", "gpus": 4, "minutes_since_checkpoint": 5},
        ]
        return sum(victim["gpus"] * victim["minutes_since_checkpoint"] for victim in victims)

    def effective_utilization(self):
        allocated = 1000
        idle = 90
        failed = 20
        blocked = 30
        return round((allocated - idle - failed - blocked) / allocated, 3)

    def scheduler_examples(self):
        return {
            "quota_usage": self.quota_usage(),
            "borrowed_gpus": self.borrowed_gpus(),
            "dominant_shares": self.dominant_shares(),
            "fairness_gap": self.fairness_gap(),
            "gang_feasible": self.gang_feasible(worker_count=3, gpus_per_worker=8),
            "topology_score": self.topology_score(["node-a1", "node-a2"]),
            "fragmentation_ratio": self.fragmentation_ratio(gpus_per_job=8),
            "preemption_loss_gpu_min": self.preemption_loss_gpu_minutes(),
            "effective_utilization": self.effective_utilization(),
        }

    def design_cases(self):
        complete = {gate: True for gate in self.gates}
        bad_specs = [
            ("tenant_queue_missing_bad", "tenant_queue_contract"),
            ("quota_borrowing_unbounded_bad", "quota_borrowing_policy"),
            ("dominant_fairness_missing_bad", "dominant_fairness_accounting"),
            ("gang_partial_start_bad", "gang_scheduling_fit"),
            ("topology_ignored_bad", "topology_aware_placement"),
            ("fragmentation_uncontrolled_bad", "fragmentation_control"),
            ("preemption_without_checkpoint_bad", "preemption_checkpoint_safety"),
            ("low_priority_not_reclaimable_bad", "low_priority_reclaim"),
            ("inference_slo_mixed_with_training_bad", "inference_slo_isolation"),
            ("heterogeneous_gpu_no_policy_bad", "heterogeneous_gpu_policy"),
            ("resource_snapshot_stale_bad", "resource_snapshot_health"),
            ("placement_reason_missing_bad", "placement_explainability"),
            ("cost_utilization_unattributed_bad", "cost_utilization_attribution"),
            ("scheduler_observability_missing_bad", "scheduler_observability_audit"),
            ("tradeoff_boundary_missing_bad", "tradeoff_boundary_reasoning"),
            ("scheduler_gate_missing_bad", "multi_tenant_gpu_scheduler_gate"),
        ]
        cases = [{"name": "complete_scheduler_design", "checks": complete}]
        for name, failed_gate in bad_specs:
            checks = {gate: True for gate in self.gates}
            checks[failed_gate] = False
            cases.append({"name": name, "checks": checks})
        return cases

    def evaluate_cases(self):
        cases = self.design_cases()
        metrics = {}
        failed_cases = []
        failed_gates = set()
        for gate in self.gates:
            passed = sum(1 for case in cases if case["checks"][gate])
            metrics[gate] = round(passed / len(cases), 3)
        for case in cases:
            missing = [gate for gate, ok in case["checks"].items() if not ok]
            if missing:
                failed_cases.append(case["name"])
                failed_gates.update(missing)
        return {
            "metrics": metrics,
            "failed_cases": failed_cases,
            "failed_gates": sorted(failed_gates),
            "hard_blocker_count": len(failed_cases),
            "gate_pass": len(failed_cases) == 0,
        }

    def run(self):
        evaluated = self.evaluate_cases()
        return {
            "scheduler_examples": self.scheduler_examples(),
            "metrics": evaluated["metrics"],
            "hard_blocker_count": evaluated["hard_blocker_count"],
            "failed_case_count": len(evaluated["failed_cases"]),
            "failed_gate_count": len(evaluated["failed_gates"]),
            "multi_tenant_gpu_scheduler_gate_pass": evaluated["gate_pass"],
        }


if __name__ == "__main__":
    audit = MiniMultiTenantGPUSchedulerDesignAudit()
    result = audit.run()
    print("scheduler_examples=" + str(result["scheduler_examples"]))
    print("metrics=" + str(result["metrics"]))
    print("hard_blocker_count=" + str(result["hard_blocker_count"]))
    print("failed_case_count=" + str(result["failed_case_count"]))
    print("failed_gate_count=" + str(result["failed_gate_count"]))
    print("multi_tenant_gpu_scheduler_gate_pass=" + str(result["multi_tenant_gpu_scheduler_gate_pass"]))
```

一组典型输出如下：

```text
scheduler_examples={'quota_usage': {'research': 0.75, 'product': 0.583, 'eval': 0.625}, 'borrowed_gpus': {'research': 8, 'product': 0, 'eval': 2}, 'dominant_shares': {'research': 0.3, 'product': 0.175, 'eval': 0.125}, 'fairness_gap': 0.042, 'gang_feasible': True, 'topology_score': 1.0, 'fragmentation_ratio': 0.2, 'preemption_loss_gpu_min': 116, 'effective_utilization': 0.86}
metrics={'tenant_queue_contract': 0.941, 'quota_borrowing_policy': 0.941, 'dominant_fairness_accounting': 0.941, 'gang_scheduling_fit': 0.941, 'topology_aware_placement': 0.941, 'fragmentation_control': 0.941, 'preemption_checkpoint_safety': 0.941, 'low_priority_reclaim': 0.941, 'inference_slo_isolation': 0.941, 'heterogeneous_gpu_policy': 0.941, 'resource_snapshot_health': 0.941, 'placement_explainability': 0.941, 'cost_utilization_attribution': 0.941, 'scheduler_observability_audit': 0.941, 'tradeoff_boundary_reasoning': 0.941, 'multi_tenant_gpu_scheduler_gate': 0.941}
hard_blocker_count=16
failed_case_count=16
failed_gate_count=16
multi_tenant_gpu_scheduler_gate_pass=False
```

这里的重点不是 `0.941` 这个 toy 分数，而是审计思路：每个 bad case 只破坏一个设计条件，因此每个条件都能被独立解释。真实平台中，这张表可以接入调度事件、队列状态、GPU 资源快照、checkpoint metadata、推理 SLO、成本标签和审计日志，成为“为什么这个任务没跑、为什么这个任务被抢占、为什么这个队列能借用资源”的统一解释层。

## 55.22 关键 trade-off

调度系统 trade-off：

1. 公平性 vs 利用率。
2. 高优先级保障 vs 低优先级饥饿。
3. 紧凑放置 vs 容错。
4. 抢占效率 vs 浪费已完成进度。
5. 资源预留 vs 空闲成本。
6. 调度复杂度 vs 可解释性。
7. 异构成本优化 vs 用户体验复杂度。

系统设计题中一定要讲这些权衡。

## 55.23 面试回答模板

可以这样回答：

```text
我会把多租户 GPU 调度系统设计成基于队列、配额和拓扑感知的调度控制平面。用户提交 TrainingJob 或 InferenceDeployment 后，平台先做身份、权限、队列和配额校验，再进入 Queue Manager。

Scheduler 维护集群 Resource Snapshot，包括 GPU 类型、数量、健康状态、拓扑、CPU、内存、网络和本地盘。调度时 Policy Engine 先根据租户配额、优先级、公平性和任务类型筛选候选任务，Placement Engine 再根据 gang scheduling、拓扑、资源碎片和数据本地性选择节点。

对于高优先级任务，如果资源不足，可以通过 Preemption Engine 抢占低优先级、可恢复任务，抢占前保存 checkpoint，任务重新入队。低优先级队列用于填充空闲 GPU，提高利用率。

系统同时提供调度事件、队列等待、配额使用、资源碎片、GPU 利用率、idle GPU hours、preemption 和成本归因的可观测性，让用户知道任务为什么没跑，让平台负责人知道资源如何使用。
```

这个回答覆盖资源模型、队列、调度、拓扑、抢占、可观测性和成本。

## 55.24 常见扣分点

扣分点一：只说 Kubernetes scheduler。

问题：没有讲 gang scheduling、队列、配额、拓扑和抢占。

扣分点二：不提多租户公平性。

问题：内部平台一定需要公平性和配额。

扣分点三：不提资源碎片。

问题：GPU 集群常见问题就是总资源够但无法满足大任务。

扣分点四：抢占只说 kill 任务。

问题：生产抢占要考虑 checkpoint、重新入队、通知和审计。

扣分点五：不提可观测性。

问题：调度系统必须解释任务为什么排队和为什么被抢占。

## 55.25 小练习

1. 为什么大模型训练需要 gang scheduling？
2. GPU 资源碎片是什么？如何缓解？
3. Guaranteed quota、burst quota 和 borrowed quota 有什么区别？
4. 拓扑感知调度要考虑哪些因素？
5. 抢占 victim 选择时应考虑哪些指标？
6. 推理服务和训练任务调度有什么不同？
7. 如何设计“我的任务为什么没跑”的解释功能？
8. 如何用低优先级队列提高 GPU 利用率？
9. 如何用 dominant share 解释多租户 GPU、CPU 和内存的公平性？
10. 如何把抢占损失从“kill 一个任务”量化成 GPU-minutes 损失？

## 55.26 本章小结

本章系统设计了一个多租户 GPU 集群调度系统。

你需要记住：

1. 多租户 GPU 调度要同时考虑公平性、优先级、配额、拓扑、碎片、抢占和利用率。
2. 大模型训练需要 gang scheduling，推理服务需要低延迟和高可用，两者调度目标不同。
3. 队列和配额是多租户治理基础，借用和低优先级队列能提高利用率。
4. 拓扑感知和资源碎片治理是 GPU 集群调度的核心难点。
5. 调度系统必须可解释、可观测、可审计，否则用户和平台团队都无法信任它。
6. 第二轮新增的公式和 demo 可以把配额占用、借用、dominant share、gang 可行性、拓扑得分、碎片率、抢占损失和有效利用率统一成可验收的系统设计门禁。

下一章我们会设计一个大模型评估与实验平台。
