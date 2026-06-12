# 第 12 章 Kubernetes 与 GPU 资源管理

上一章讲了容器化。本章进入容器编排和资源管理：Kubernetes 如何管理 GPU？为什么很多 AI 平台会基于 Kubernetes 构建？为什么 Kubernetes 又不能直接解决所有大模型调度问题？

很多候选人一被问到 AI Infra，就回答“用 Kubernetes”。这不够。Kubernetes 是重要底座，但大模型训练和推理还需要 GPU device plugin、调度扩展、队列、配额、拓扑感知、gang scheduling、抢占、checkpoint 和多租户治理。

先记住一句话：

> Kubernetes 能管理容器和基础资源，但大模型 GPU 资源管理还需要在它之上补齐 AI 工作负载特有的调度、拓扑、队列和治理能力。

## 12.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，资料口径按“Kubernetes 上的 GPU 资源管理稳定抽象”处理，而不是绑定某个云厂商、某个集群发行版、某个训练平台或某个调度器插件版本。Kubernetes 侧重点参考官方 device plugin、extended resource、ResourceQuota、node affinity、taint / toleration 和 topology spread 等文档口径；NVIDIA 侧重点参考 NVIDIA Kubernetes device plugin 对 GPU、MIG、time-slicing / MPS 共享模式的说明，以及 GPU Operator 对 driver、container toolkit、device plugin、DCGM exporter、MIG manager 和 GPU feature discovery 的管理边界。

需要注意三点：

1. Kubernetes 上游已有 gang scheduling 的 alpha 能力文档，但这不等于所有生产集群默认可用。真实平台仍然要明确是否启用上游能力，或用 Volcano、Kueue、YuniKorn、自研调度层等方式补齐队列、配额、gang、抢占和公平性。
2. 本章讲 GPU resource expression、device plugin、GPU Operator、MIG / 共享、拓扑感知、namespace / quota、多租户和监控排障；更细的队列、优先级、抢占、公平性和配额策略放到下一章展开。
3. 这里的 demo 是教学审计脚本，不调用真实 Kubernetes API。它帮助你把“GPU 能不能被 K8s 管起来”拆成可检查字段，真实平台还需要从 kube-state-metrics、scheduler events、device plugin logs、DCGM、训练平台数据库和审计日志采集证据。

## 12.1 Kubernetes 在 AI Infra 中的位置

Kubernetes，简称 K8s，是容器编排系统。

它负责：

1. 启动容器。
2. 调度 Pod 到节点。
3. 管理服务发现。
4. 管理资源 request / limit。
5. 管理配置和密钥。
6. 做健康检查。
7. 支持滚动升级。
8. 提供声明式 API。

在 AI Infra 中，K8s 常作为底层调度和运行平台。

训练平台、推理平台、评估平台可以把任务转换成 K8s 资源，例如 Pod、Job、Deployment、StatefulSet 或自定义 CRD。

但 K8s 默认更偏通用容器编排，不是专门为大规模分布式训练设计的。

## 12.2 Kubernetes 的基本对象

理解 K8s GPU 管理前，需要知道几个对象。

Pod：K8s 调度的最小单位，里面可以包含一个或多个容器。

Node：集群中的机器。

Deployment：管理无状态服务副本，适合推理服务。

Job：运行一次性任务，适合离线任务。

StatefulSet：管理有状态服务。

ConfigMap：配置。

Secret：密钥。

Service：服务发现和负载均衡入口。

Namespace：逻辑隔离空间。

CRD：自定义资源定义，用来扩展 K8s API。

AI 平台通常会定义自己的 CRD，例如 TrainingJob、InferenceService、RayCluster、ModelDeployment。

## 12.3 Kubernetes 如何表达资源

K8s 使用 request 和 limit 表达资源需求。

例如 CPU 和内存：

```yaml
resources:
  requests:
    cpu: "8"
    memory: "64Gi"
  limits:
    cpu: "8"
    memory: "64Gi"
```

GPU 通常作为 extended resource 表达：

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

这里的意思是这个 Pod 需要 4 张 NVIDIA GPU。

注意：GPU 通常是整数资源。默认情况下，K8s 不会像 CPU 那样自然切分 GPU。

如果要做 GPU 共享、MIG、时间片或虚拟化，需要额外机制。

## 12.4 Device Plugin

K8s 本身不知道某台机器上有哪些 GPU。

Device Plugin 负责把设备资源注册给 K8s。

NVIDIA GPU 常用 NVIDIA device plugin。

它负责：

1. 发现节点上的 GPU。
2. 向 kubelet 注册 GPU 资源。
3. 把 GPU 暴露给容器。
4. 设置设备文件。
5. 设置环境变量。
6. 支持 MIG 等模式。

没有 device plugin，K8s 调度器无法正确识别 GPU 资源。

Device Plugin 是 GPU 容器调度的关键桥梁。

## 12.5 GPU Operator

GPU Operator 是用于自动化管理 GPU 软件栈的工具。

它通常可以管理：

1. NVIDIA driver。
2. Container toolkit。
3. Device plugin。
4. DCGM exporter。
5. MIG manager。
6. GPU feature discovery。

它的价值是减少手工安装和维护 GPU 节点软件栈的复杂度。

但在大型生产集群里，团队仍然需要严格控制驱动版本、升级节奏和兼容矩阵。

不要以为用了 operator 就不用理解驱动、CUDA 和 device plugin。

## 12.6 GPU 调度的基本流程

一个 GPU Pod 的调度流程大致是：

1. 用户提交任务。
2. 平台生成 Pod / Job / CRD。
3. Pod 声明需要 GPU 数量。
4. K8s scheduler 查找有足够 GPU 的节点。
5. Pod 被绑定到节点。
6. kubelet 启动容器。
7. device plugin 分配 GPU 设备。
8. 容器内通过 CUDA 访问 GPU。

这个流程适合简单任务。

但大模型训练需要的不只是“找一台有 8 张空闲 GPU 的机器”。

它还需要：

1. 多个 Pod 同时启动。
2. 多节点 gang scheduling。
3. 拓扑感知。
4. 队列和优先级。
5. 抢占和恢复。
6. 配额和公平性。
7. 多租户隔离。

这些通常需要扩展。

## 12.7 Kubernetes 默认调度的局限

K8s 默认调度器对普通服务很强，但对大模型训练有局限。

主要包括：

1. 不天然支持 gang scheduling。
2. 对 GPU 拓扑理解有限。
3. 不天然理解训练任务整体性。
4. 队列和公平性能力有限。
5. GPU 碎片化问题明显。
6. 多租户成本治理需要额外系统。
7. 对分布式训练失败恢复支持有限。

例如一个 64 卡训练任务需要 8 台 8 卡机器同时启动。如果 K8s 只调度到了 6 台，任务无法正常开始，但资源可能已经被占住。

这就是 gang scheduling 要解决的问题。

## 12.8 Gang Scheduling

Gang scheduling 是指一组相关 Pod 要么一起调度成功，要么都不启动。

分布式训练非常需要它。

截至当前主流 Kubernetes 口径，gang scheduling 不是所有生产集群默认开启的基础能力。即使上游有 alpha 能力，训练平台也要把“是否支持 gang、是否有最小可用 worker 数、失败后是否释放已占资源、是否和队列 / quota / priority 联动”写进调度门禁，而不是只假设 scheduler 会自动理解分布式训练的整体性。

原因：

1. 所有 worker 需要同时参与训练。
2. 缺少部分 worker 任务无法启动。
3. 部分 Pod 先占资源会造成资源浪费。
4. rank 初始化依赖完整 world size。

例如：

```text
任务需要 16 个 worker，每个 worker 8 GPU。
如果只能调度 12 个 worker，任务不能开始。
```

没有 gang scheduling，可能出现资源被部分 Pod 占住但训练无法进行的情况。

AI 平台常用 Volcano、Kueue、YuniKorn 或自研调度层来补齐这类能力。

## 12.9 GPU 碎片化

GPU 碎片化是集群利用率的大问题。

例如集群中有很多节点，每个节点 8 张 GPU。

如果很多任务各占 1 到 2 张 GPU，可能导致每台机器都剩几张卡，但没有一台机器能满足一个需要完整 8 卡的任务。

这就是碎片化。

碎片化影响：

1. 大任务排队。
2. GPU 利用率看似不低但可用性差。
3. 调度等待时间变长。
4. 拓扑变差。

缓解方法：

1. bin packing。
2. 按任务类型划分资源池。
3. 小任务集中放置。
4. 大任务预留完整节点。
5. 抢占低优先级碎片任务。
6. 定期整理或重调度。

## 12.10 拓扑感知调度

第 7 章讲过拓扑感知，这里结合 K8s 再看。

GPU 拓扑包括：

1. 同一节点内 GPU 是否通过 NVLink / NVSwitch 连接。
2. GPU 属于哪个 NUMA 域。
3. GPU 到网卡距离。
4. 节点在哪个机柜。
5. 节点之间网络路径如何。
6. 数据是否靠近节点。

K8s 默认调度器对这些细节理解有限。

AI 调度层需要根据任务特点选择资源。

例如：

1. tensor parallel 任务优先同机 NVSwitch GPU。
2. 多机训练优先同机柜节点。
3. 数据密集任务优先靠近缓存节点。
4. 推理服务要考虑 GPU 型号和显存大小。

拓扑感知调度能显著影响训练效率。

## 12.11 Node Label、Taint 和 Affinity

K8s 提供一些基础调度能力。

Node label 可以标记节点属性：

```text
gpu-type=A100
gpu-count=8
network=ib
zone=rack-a
```

Node selector 可以选择节点。

Affinity 可以表达亲和性和反亲和性。

Taint 和 toleration 可以控制哪些 Pod 能调度到某类节点。

例如：

1. 训练任务调度到 GPU 节点。
2. 推理任务调度到推理资源池。
3. 高优先级任务调度到专用节点。
4. 普通任务不能调度到保留节点。

这些能力是基础，但复杂 AI 调度通常还需要自定义调度器或调度扩展。

## 12.12 GPU 共享、MIG 和时间片

GPU 默认通常按整卡分配。

但有些场景想共享 GPU。

常见方式：

1. MIG。
2. 时间片共享。
3. MPS。
4. 自定义 GPU 虚拟化。

MIG 是 Multi-Instance GPU，可以把支持的 GPU 切成多个硬件隔离实例。

适合：

1. 小模型推理。
2. 评估任务。
3. 交互式开发。
4. 需要隔离的小负载。

不适合：

1. 大模型训练。
2. 需要完整显存的任务。
3. 通信密集多卡任务。

时间片共享可以提高利用率，但隔离性和性能稳定性不如硬件隔离。

推理服务中可以考虑 GPU 共享，预训练通常不适合随意共享。

## 12.13 GPU 资源申请设计

训练平台不应该让用户直接写复杂 YAML。

更好的方式是提供任务配置：

```yaml
resources:
  gpu_type: A100-80G
  gpu_count: 64
  gpu_per_node: 8
  network: ib
  priority: high
  preemptible: false
```

平台再把它转换成 K8s 资源和调度约束。

资源申请应该表达：

1. GPU 型号。
2. GPU 数量。
3. 每节点 GPU 数。
4. 显存需求。
5. 网络需求。
6. 存储需求。
7. 优先级。
8. 是否可抢占。
9. 预计运行时长。
10. 项目和租户。

这样调度系统才能做更好的决策。

## 12.14 K8s 上运行训练任务

在 K8s 上运行分布式训练，常见方式：

1. PyTorchJob。
2. MPIJob。
3. RayJob。
4. DeepSpeed launcher + Job。
5. 自研 TrainingJob CRD。

关键问题：

1. master 地址如何发现。
2. worker rank 如何分配。
3. world size 如何设置。
4. worker 如何同时启动。
5. 某个 worker 失败怎么办。
6. checkpoint 如何保存。
7. 日志如何聚合。
8. 任务状态如何展示。

K8s 提供基础运行能力，但训练语义通常由上层 operator 或平台补齐。

## 12.15 K8s 上运行推理服务

推理服务通常用 Deployment、StatefulSet 或专门的 InferenceService。

关注点：

1. 模型加载。
2. GPU 绑定。
3. 服务发现。
4. 自动扩缩容。
5. 滚动升级。
6. 灰度发布。
7. 健康检查。
8. 请求路由。
9. GPU 利用率和延迟监控。
10. 冷启动。

推理服务和训练任务不同，它是在线服务，要满足 SLO。

所以推理资源池通常需要更严格的隔离和优先级。

## 12.16 Namespace、Quota 和多租户

K8s 可以用 namespace 做逻辑隔离。

ResourceQuota 可以限制资源使用。

例如：

1. 每个团队最多使用多少 GPU。
2. 每个 namespace 最多创建多少 Pod。
3. 限制 CPU、内存和存储。
4. 限制高优先级任务数量。

但 AI 多租户还需要更细能力：

1. 按 GPU 型号配额。
2. 按资源池配额。
3. 按项目成本统计。
4. 按任务优先级排队。
5. 按团队公平分享。
6. 数据权限隔离。
7. 镜像和密钥权限隔离。

所以 namespace 和 quota 是基础，不是完整多租户方案。

## 12.17 GPU 监控

K8s 需要配合 GPU 监控。

常见监控组件包括 DCGM exporter、Prometheus、Grafana 等。

关键指标：

1. GPU utilization。
2. 显存使用。
3. 显存带宽。
4. GPU temperature。
5. power usage。
6. ECC error。
7. PCIe throughput。
8. GPU process。
9. Pod 到 GPU 的映射。
10. GPU 分配率。

AI 平台还要把 GPU 指标和任务关联起来。

例如：

```text
team_a / job_123 / pod_worker_0 / gpu_3 / utilization=92%
```

否则你只能看到某张卡忙，不知道是谁在用。

## 12.18 常见问题和排查

### 12.18.1 Pod 一直 Pending

可能原因：

1. GPU 不足。
2. 指定 GPU 型号没有资源。
3. node selector 太严格。
4. taint 没有 toleration。
5. quota 不够。
6. gang scheduling 条件不满足。
7. 镜像拉取失败。

### 12.18.2 Pod 启动后看不到 GPU

可能原因：

1. device plugin 异常。
2. GPU runtime 配置错误。
3. 容器没有分配 GPU。
4. 驱动问题。
5. MIG 配置不正确。

### 12.18.3 GPU 空闲但任务排队

可能原因：

1. GPU 碎片化。
2. 任务需要完整节点。
3. 配额不足。
4. 优先级较低。
5. 拓扑约束不满足。
6. 资源池隔离导致不可用。

### 12.18.4 训练任务部分 worker 启动失败

可能原因：

1. 没有 gang scheduling。
2. 镜像拉取速度不一致。
3. 某些节点环境异常。
4. 网络或 DNS 问题。
5. rank 配置错误。
6. checkpoint 挂载失败。

## 12.19 Kubernetes GPU 资源管理审计指标与最小 demo

把 Kubernetes GPU 管理讲清楚，不能只说“Pod 里写 `nvidia.com/gpu`”。更工程化的说法是：平台要证明 GPU 设备能注册、资源请求能表达、调度约束能落地、分布式训练能整体启动、碎片和拓扑能被治理、租户配额能被约束、监控和排障能把 Pod 映射回具体 GPU。

可以把一个 GPU 任务或资源池审计样本写成：

```math
k_i=(p_i,n_i,d_i,g_i,m_i,q_i,a_i,t_i,r_i,o_i,s_i,z_i)
```

其中，`p_i` 是 Pod / Job / CRD 画像，`n_i` 是节点与 GPU 拓扑，`d_i` 是 device plugin 状态，`g_i` 是 GPU request / limit，`m_i` 是 MIG / 共享策略，`q_i` 是 namespace / quota，`a_i` 是 affinity / taint / toleration，`t_i` 是 topology 证据，`r_i` 是训练或推理运行时，`o_i` 是监控观测，`s_i` 是排障事件，`z_i` 是最终门禁。

统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(k_i)=1]
```

其中，`g_j(k_i)=1` 表示样本 `i` 通过第 `j` 个 Kubernetes GPU 管理检查。

GPU 分配率可以写成：

```math
U_{\mathrm{gpu}}=\frac{G_{\mathrm{allocated}}}{G_{\mathrm{total}}}
```

其中，`G_allocated` 是已经分配给 Pod / Job 的 GPU 数，`G_total` 是资源池总 GPU 数。

碎片化比例可以写成：

```math
R_{\mathrm{frag}}=1-\frac{\sum_n \lfloor g_n / g_{\mathrm{job}}\rfloor g_{\mathrm{job}}}{\sum_n g_n}
```

其中，`g_n` 是节点 `n` 上的空闲 GPU 数，`g_job` 是目标任务每个放置单元需要的 GPU 数。这个值越高，说明“总空闲 GPU 看起来够，但能满足完整任务形状的 GPU 越少”。

Gang scheduling 可行性可以写成：

```math
C_{\mathrm{gang}}=\mathbf{1}\left[\sum_n \lfloor g_n / g_{\mathrm{pernode}}\rfloor \ge N_{\mathrm{worker}}\right]
```

其中，`g_pernode` 是每个 worker 需要的单节点 GPU 数，`N_worker` 是 worker 数量。

租户配额占用率可以写成：

```math
Q_{\mathrm{tenant}}=\frac{G_{\mathrm{used}}}{G_{\mathrm{quota}}}
```

其中，`G_used` 是租户当前已用 GPU，`G_quota` 是租户配额。

拓扑得分可以写成：

```math
S_{\mathrm{topo}}=w_1 C_{\mathrm{node}}+w_2 C_{\mathrm{nic}}+w_3 C_{\mathrm{rack}}+w_4 C_{\mathrm{storage}}
```

其中，`C_node`、`C_nic`、`C_rack`、`C_storage` 分别表示节点内 GPU 互联、GPU-NIC 亲和性、机柜 locality 和存储 locality 是否满足任务画像，`w_i` 是权重。

最后，可以把 Kubernetes GPU 资源管理门禁写成：

```math
G_{\mathrm{k8s\_gpu}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{frag}}\le \rho \land Q_{\mathrm{tenant}}\le 1 \land P_0=0\right]
```

其中，`\tau_j` 是每个检查项的最低覆盖率，`\rho` 是碎片化阈值，`P_0` 是 P0 级风险数量。

下面这个 0 依赖 demo 演示如何把 Kubernetes GPU 管理拆成审计规则。它故意构造 1 个完整样本和 16 个坏样本，让每个关键维度各失败一次。

```python
import copy


METRICS = [
    "device_plugin_readiness",
    "gpu_extended_resource_fit",
    "gpu_request_limit_integrity",
    "gang_scheduling_readiness",
    "fragmentation_control",
    "topology_aware_placement",
    "node_label_affinity_fit",
    "taint_toleration_fit",
    "mig_sharing_policy_fit",
    "quota_namespace_governance",
    "training_operator_readiness",
    "inference_service_readiness",
    "gpu_monitoring_mapping",
    "multi_tenant_isolation",
    "pending_troubleshooting_coverage",
    "kubernetes_gpu_gate",
]


def gpu_allocation_ratio(allocated, total):
    return allocated / total


def fragmentation_ratio(free_gpus_by_node, gpus_per_job):
    allocatable = sum((free // gpus_per_job) * gpus_per_job for free in free_gpus_by_node)
    total_free = sum(free_gpus_by_node)
    return 1 - allocatable / total_free


def gang_feasible(free_gpus_by_node, gpu_per_node, workers):
    available_workers = sum(free // gpu_per_node for free in free_gpus_by_node)
    return available_workers >= workers


def tenant_quota_ratio(used_gpu, quota_gpu):
    return used_gpu / quota_gpu


def topology_score(components, weights):
    return round(sum(components[name] * weights[name] for name in weights), 3)


def build_kubernetes_gpu_cases():
    complete = {
        "name": "complete",
        "device_plugin": {
            "ready": True,
            "resource_name": "nvidia.com/gpu",
            "node_capacity_reported": True,
            "allocatable_reported": True,
        },
        "resource": {
            "extended_resource_declared": True,
            "limits_gpu": 8,
            "requests_gpu": 8,
            "requests_equal_limits": True,
            "total_requested_gpu": 64,
            "available_gpu": 128,
        },
        "gang": {
            "enabled": True,
            "min_available": 8,
            "worker_count": 8,
            "gpu_per_worker": 8,
            "partial_allocation_allowed": False,
        },
        "fragmentation": {
            "free_gpus_by_node": [8, 8, 8, 8, 4, 4],
            "gpus_per_job": 8,
            "threshold": 0.35,
            "bin_packing": True,
            "large_job_pool": True,
        },
        "topology": {
            "components": {"node": 1.0, "nic": 1.0, "rack": 0.8, "storage": 0.8},
            "weights": {"node": 0.4, "nic": 0.25, "rack": 0.2, "storage": 0.15},
            "threshold": 0.8,
            "gpu_nic_affinity_known": True,
            "nvlink_domain_known": True,
        },
        "placement": {
            "labels": {"gpu-type": "A100-80G", "network": "ib", "pool": "train"},
            "required_labels": {"gpu-type": "A100-80G", "network": "ib"},
            "node_affinity": True,
            "label_governance": True,
        },
        "taints": {
            "dedicated_taints": {"gpu-pool"},
            "tolerations": {"gpu-pool"},
            "toleration_scope": True,
        },
        "sharing": {
            "workload": "pretraining",
            "mode": "exclusive",
            "mig_policy": "none",
            "time_slicing": False,
            "mps": False,
        },
        "quota": {
            "namespace": "team-a",
            "resource_quota": True,
            "per_gpu_type_quota": True,
            "used_gpu": 48,
            "quota_gpu": 64,
            "cost_attribution": True,
        },
        "training_runtime": {
            "training_crd": True,
            "rank_env": True,
            "service_discovery": True,
            "checkpoint_uri": True,
        },
        "inference_runtime": {
            "serving_controller": True,
            "autoscaling": True,
            "slo_metrics": True,
            "health_check": True,
            "cold_start_budget": True,
        },
        "monitoring": {
            "dcgm_exporter": True,
            "prometheus": True,
            "pod_gpu_mapping": True,
            "ecc_error": True,
            "allocation_metric": True,
        },
        "isolation": {
            "namespace": True,
            "rbac": True,
            "network_policy": True,
            "secret_scope": True,
            "registry_scope": True,
        },
        "troubleshooting": {
            "playbook": True,
            "pending_reasons": {
                "gpu_insufficient",
                "quota_exceeded",
                "node_affinity",
                "taint_blocked",
                "image_pull",
                "gang_waiting",
            },
        },
        "gate": {"enabled": True, "thresholds_ok": True},
    }

    def bad_case(name, mutator):
        case = copy.deepcopy(complete)
        case["name"] = name
        mutator(case)
        return case

    bad_cases = [
        bad_case("device_plugin_missing_bad", lambda c: c["device_plugin"].update({"ready": False})),
        bad_case("gpu_resource_not_declared_bad", lambda c: c["resource"].update({"extended_resource_declared": False})),
        bad_case("request_limit_mismatch_bad", lambda c: c["resource"].update({"requests_equal_limits": False})),
        bad_case("no_gang_scheduling_bad", lambda c: c["gang"].update({"enabled": False})),
        bad_case("fragmented_cluster_bad", lambda c: c["fragmentation"].update({"free_gpus_by_node": [4, 4, 4, 4, 4]})),
        bad_case("topology_ignored_bad", lambda c: c["topology"].update({"gpu_nic_affinity_known": False})),
        bad_case("label_affinity_missing_bad", lambda c: c["placement"].update({"node_affinity": False})),
        bad_case("toleration_missing_bad", lambda c: c["taints"].update({"tolerations": set()})),
        bad_case("mig_for_pretraining_bad", lambda c: c["sharing"].update({"mode": "time_slicing", "time_slicing": True})),
        bad_case("quota_missing_bad", lambda c: c["quota"].update({"resource_quota": False})),
        bad_case("training_operator_missing_bad", lambda c: c["training_runtime"].update({"training_crd": False})),
        bad_case("inference_slo_missing_bad", lambda c: c["inference_runtime"].update({"slo_metrics": False})),
        bad_case("gpu_monitoring_unmapped_bad", lambda c: c["monitoring"].update({"pod_gpu_mapping": False})),
        bad_case("tenant_isolation_missing_bad", lambda c: c["isolation"].update({"rbac": False})),
        bad_case("pending_reason_unknown_bad", lambda c: c["troubleshooting"].update({"pending_reasons": {"gpu_insufficient"}})),
        bad_case("kubernetes_gpu_gate_missing_bad", lambda c: c["gate"].update({"enabled": False})),
    ]
    return [complete] + bad_cases


def check_device_plugin(case):
    plugin = case["device_plugin"]
    return (
        plugin["ready"]
        and plugin["resource_name"] == "nvidia.com/gpu"
        and plugin["node_capacity_reported"]
        and plugin["allocatable_reported"]
    )


def check_gpu_resource(case):
    resource = case["resource"]
    return (
        resource["extended_resource_declared"]
        and resource["limits_gpu"] > 0
        and resource["total_requested_gpu"] <= resource["available_gpu"]
    )


def check_request_limit(case):
    resource = case["resource"]
    return resource["requests_equal_limits"] and resource["requests_gpu"] == resource["limits_gpu"]


def check_gang(case):
    gang = case["gang"]
    return (
        gang["enabled"]
        and gang["min_available"] == gang["worker_count"]
        and not gang["partial_allocation_allowed"]
    )


def check_fragmentation(case):
    frag = case["fragmentation"]
    return (
        fragmentation_ratio(frag["free_gpus_by_node"], frag["gpus_per_job"]) <= frag["threshold"]
        and frag["bin_packing"]
        and frag["large_job_pool"]
    )


def check_topology(case):
    topo = case["topology"]
    return (
        topology_score(topo["components"], topo["weights"]) >= topo["threshold"]
        and topo["gpu_nic_affinity_known"]
        and topo["nvlink_domain_known"]
    )


def check_label_affinity(case):
    placement = case["placement"]
    labels = placement["labels"]
    required = placement["required_labels"]
    labels_match = all(labels.get(key) == value for key, value in required.items())
    return placement["node_affinity"] and placement["label_governance"] and labels_match


def check_taints(case):
    taints = case["taints"]
    return taints["dedicated_taints"].issubset(taints["tolerations"]) and taints["toleration_scope"]


def check_sharing(case):
    sharing = case["sharing"]
    if sharing["workload"] == "pretraining":
        return (
            sharing["mode"] == "exclusive"
            and sharing["mig_policy"] == "none"
            and not sharing["time_slicing"]
            and not sharing["mps"]
        )
    return sharing["mode"] in {"exclusive", "mig", "time_slicing", "mps"}


def check_quota(case):
    quota = case["quota"]
    return (
        quota["resource_quota"]
        and quota["per_gpu_type_quota"]
        and tenant_quota_ratio(quota["used_gpu"], quota["quota_gpu"]) <= 1
        and quota["cost_attribution"]
    )


def check_training_runtime(case):
    return all(case["training_runtime"].values())


def check_inference_runtime(case):
    return all(case["inference_runtime"].values())


def check_monitoring(case):
    return all(case["monitoring"].values())


def check_isolation(case):
    return all(case["isolation"].values())


def check_troubleshooting(case):
    required = {
        "gpu_insufficient",
        "quota_exceeded",
        "node_affinity",
        "taint_blocked",
        "image_pull",
        "gang_waiting",
    }
    trouble = case["troubleshooting"]
    return trouble["playbook"] and required.issubset(trouble["pending_reasons"])


def check_gate(case):
    return case["gate"]["enabled"] and case["gate"]["thresholds_ok"]


CHECKS = {
    "device_plugin_readiness": check_device_plugin,
    "gpu_extended_resource_fit": check_gpu_resource,
    "gpu_request_limit_integrity": check_request_limit,
    "gang_scheduling_readiness": check_gang,
    "fragmentation_control": check_fragmentation,
    "topology_aware_placement": check_topology,
    "node_label_affinity_fit": check_label_affinity,
    "taint_toleration_fit": check_taints,
    "mig_sharing_policy_fit": check_sharing,
    "quota_namespace_governance": check_quota,
    "training_operator_readiness": check_training_runtime,
    "inference_service_readiness": check_inference_runtime,
    "gpu_monitoring_mapping": check_monitoring,
    "multi_tenant_isolation": check_isolation,
    "pending_troubleshooting_coverage": check_troubleshooting,
    "kubernetes_gpu_gate": check_gate,
}


def audit_kubernetes_gpu_management(cases):
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
        "kubernetes_gpu_gate_pass": not failed_cases and min(metrics.values()) >= 0.95,
    }


cases = build_kubernetes_gpu_cases()
case_by_name = {case["name"]: case for case in cases}
complete = case_by_name["complete"]
examples = {
    "gpu_allocation_ratio": round(gpu_allocation_ratio(56, 80), 3),
    "fragmentation_ratio": round(fragmentation_ratio([8, 4, 4, 2, 2], 8), 3),
    "gang_feasible": gang_feasible([8, 8, 8, 4], 8, 3),
    "tenant_quota_ratio": round(tenant_quota_ratio(48, 64), 3),
    "topology_score": topology_score(
        {"node": 1.0, "nic": 1.0, "rack": 0.5, "storage": 1.0},
        {"node": 0.4, "nic": 0.25, "rack": 0.2, "storage": 0.15},
    ),
}

smoke = {
    "complete_case_passes": all(check(complete) for check in CHECKS.values()),
    "caught_device_plugin_gap": not check_device_plugin(case_by_name["device_plugin_missing_bad"]),
    "caught_fragmentation": not check_fragmentation(case_by_name["fragmented_cluster_bad"]),
    "caught_topology_gap": not check_topology(case_by_name["topology_ignored_bad"]),
    "caught_quota_gap": not check_quota(case_by_name["quota_missing_bad"]),
    "caught_pending_gap": not check_troubleshooting(case_by_name["pending_reason_unknown_bad"]),
}

audit = audit_kubernetes_gpu_management(cases)
print(f"kubernetes_gpu_examples={examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"kubernetes_gpu_gate_pass={audit['kubernetes_gpu_gate_pass']}")
```

一组典型输出是：

```text
kubernetes_gpu_examples={'gpu_allocation_ratio': 0.7, 'fragmentation_ratio': 0.6, 'gang_feasible': True, 'tenant_quota_ratio': 0.75, 'topology_score': 0.9}
smoke={'complete_case_passes': True, 'caught_device_plugin_gap': True, 'caught_fragmentation': True, 'caught_topology_gap': True, 'caught_quota_gap': True, 'caught_pending_gap': True}
metrics={'device_plugin_readiness': 0.941, 'gpu_extended_resource_fit': 0.941, 'gpu_request_limit_integrity': 0.941, 'gang_scheduling_readiness': 0.941, 'fragmentation_control': 0.941, 'topology_aware_placement': 0.941, 'node_label_affinity_fit': 0.941, 'taint_toleration_fit': 0.941, 'mig_sharing_policy_fit': 0.941, 'quota_namespace_governance': 0.941, 'training_operator_readiness': 0.941, 'inference_service_readiness': 0.941, 'gpu_monitoring_mapping': 0.941, 'multi_tenant_isolation': 0.941, 'pending_troubleshooting_coverage': 0.941, 'kubernetes_gpu_gate': 0.941}
hard_blocker_count=16
failed_cases=['device_plugin_missing_bad', 'gpu_resource_not_declared_bad', 'request_limit_mismatch_bad', 'no_gang_scheduling_bad', 'fragmented_cluster_bad', 'topology_ignored_bad', 'label_affinity_missing_bad', 'toleration_missing_bad', 'mig_for_pretraining_bad', 'quota_missing_bad', 'training_operator_missing_bad', 'inference_slo_missing_bad', 'gpu_monitoring_unmapped_bad', 'tenant_isolation_missing_bad', 'pending_reason_unknown_bad', 'kubernetes_gpu_gate_missing_bad']
kubernetes_gpu_gate_pass=False
```

这个 demo 的重点是把 Kubernetes GPU 管理拆成可验证证据链：device plugin 要就绪，GPU extended resource 要正确声明，request / limit 不能含糊，分布式训练要有 gang 语义，碎片化和拓扑要进入调度决策，node label / affinity / taint / toleration 要可解释，MIG / time slicing 要匹配工作负载，namespace / quota / 成本归因要约束租户，训练和推理运行时要有各自的 CRD / SLO / health 证据，GPU 监控必须能映射到 Pod，最后还要有 Pending 原因排障和统一门禁。

## 12.20 面试中如何回答 K8s 与 GPU 管理

如果面试官问：

```text
如何用 Kubernetes 管理大模型训练和推理的 GPU 资源？
```

可以这样回答：

```text
Kubernetes 可以作为容器编排和资源管理底座。GPU 通过 device plugin 注册为 extended resource，Pod 通过 nvidia.com/gpu 申请 GPU。GPU Operator 可以帮助管理驱动、device plugin、DCGM exporter 等节点组件。

但默认 K8s 调度器不足以完整支持大模型训练。分布式训练需要 gang scheduling，避免部分 worker 占住资源但任务无法启动；还需要拓扑感知调度，考虑 NVLink/NVSwitch、GPU 到网卡亲和性、机柜网络和数据位置；还要处理 GPU 碎片化、队列、优先级、抢占、配额和多租户公平性。

训练任务可以通过 PyTorchJob、MPIJob、RayJob 或自研 TrainingJob CRD 表达；推理服务可以用 Deployment、StatefulSet 或 InferenceService，并配合自动扩缩容、灰度发布和 SLO 监控。平台层应该把用户友好的资源申请转换成 K8s 对象和调度约束，而不是让用户直接写复杂 YAML。
```

## 12.21 常见误区

误区一：用了 Kubernetes 就解决了 GPU 调度。

K8s 只是底座，大模型还需要 gang scheduling、拓扑感知、队列、配额、抢占和多租户治理。

误区二：GPU 和 CPU 一样可以随便切分。

GPU 通常按整卡分配，MIG、时间片和共享需要额外机制，且适用场景有限。

误区三：Pod Pending 就一定是没有 GPU。

也可能是配额、node selector、taint、优先级、拓扑约束或 gang scheduling 条件不满足。

误区四：Namespace 就等于多租户隔离。

Namespace 是逻辑隔离，还需要资源配额、权限、网络策略、数据隔离、镜像权限和成本归因。

误区五：训练和推理可以使用同一套调度策略。

训练关注吞吐、gang scheduling 和 checkpoint，推理关注 SLO、扩缩容和稳定性。

## 12.22 面试题

### 题 1：Kubernetes 如何识别 GPU？

答：K8s 通过 device plugin 机制识别 GPU。NVIDIA device plugin 会发现节点上的 GPU，并向 kubelet 注册 `nvidia.com/gpu` 这类 extended resource。Pod 通过资源声明申请 GPU，kubelet 启动容器时把对应 GPU 设备暴露给容器。

### 题 2：为什么分布式训练需要 gang scheduling？

答：分布式训练需要多个 worker 同时参与。如果只启动部分 worker，任务无法正常开始，但资源可能被占住。Gang scheduling 保证一组 Pod 要么全部调度成功，要么都不启动，避免资源浪费和任务卡死。

### 题 3：什么是 GPU 碎片化？

答：GPU 碎片化是指很多节点上都有少量空闲 GPU，但无法满足需要完整节点或大量连续 GPU 的任务。例如每台 8 卡机器都剩 2 张卡，但一个 8 卡任务无法启动。解决方法包括 bin packing、资源池划分、小任务集中放置、抢占和重调度。

### 题 4：K8s 默认调度器为什么不够？

答：默认调度器对普通容器服务足够，但不天然理解大模型训练的整体性、GPU 拓扑、gang scheduling、通信密集任务、队列公平性、抢占恢复和多租户成本治理。这些需要调度扩展或上层平台补齐。

### 题 5：MIG 适合什么场景？

答：MIG 适合把支持的 GPU 切成多个硬件隔离实例，常用于小模型推理、评估、交互式开发和隔离小负载。不适合需要完整显存和高带宽通信的大模型训练或通信密集多卡任务。

## 12.23 小练习

练习一：写一个 GPU Pod 资源申请示例。

要求：声明需要 2 张 GPU，并说明 device plugin 的作用。

练习二：设计一个 TrainingJob CRD。

要求：包含 gpu_type、gpu_count、gpu_per_node、priority、queue、image、command、checkpoint_uri 和 data_uri。

练习三：分析 Pod Pending。

假设一个 64 卡训练任务一直 Pending，请从 GPU 数量、完整节点、quota、taint、node selector、gang scheduling、拓扑和优先级角度排查。

练习四：设计一个 GPU 碎片化治理策略。

要求：包含 bin packing、小任务资源池、大任务预留、抢占和监控指标。

## 12.24 本章小结

本章讲了 Kubernetes 与 GPU 资源管理。

你需要掌握：

1. Kubernetes 是 AI Infra 常见底座，但不是完整 AI 调度系统。
2. GPU 通过 device plugin 注册为 extended resource。
3. GPU Operator 可以管理驱动、device plugin、监控等节点组件。
4. 默认 K8s 调度器对大模型训练有局限。
5. 分布式训练需要 gang scheduling。
6. GPU 碎片化会影响大任务启动和集群利用率。
7. 拓扑感知调度需要考虑 NVLink/NVSwitch、NUMA、网卡、机柜和数据位置。
8. MIG、时间片和 GPU 共享适合部分小负载，不适合所有任务。
9. Namespace 和 ResourceQuota 是多租户基础，但还需要更完整的权限、配额、队列和成本治理。
10. 平台层应该为用户屏蔽复杂 YAML，提供面向任务的资源申请接口。

下一章我们会讲训练任务调度：队列、优先级、抢占、公平性和配额。
