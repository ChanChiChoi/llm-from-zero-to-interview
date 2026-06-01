# 第 12 章 Kubernetes 与 GPU 资源管理

上一章讲了容器化。本章进入容器编排和资源管理：Kubernetes 如何管理 GPU？为什么很多 AI 平台会基于 Kubernetes 构建？为什么 Kubernetes 又不能直接解决所有大模型调度问题？

很多候选人一被问到 AI Infra，就回答“用 Kubernetes”。这不够。Kubernetes 是重要底座，但大模型训练和推理还需要 GPU device plugin、调度扩展、队列、配额、拓扑感知、gang scheduling、抢占、checkpoint 和多租户治理。

先记住一句话：

> Kubernetes 能管理容器和基础资源，但大模型 GPU 资源管理还需要在它之上补齐 AI 工作负载特有的调度、拓扑、队列和治理能力。

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

## 12.19 面试中如何回答 K8s 与 GPU 管理

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

## 12.20 常见误区

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

## 12.21 面试题

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

## 12.22 小练习

练习一：写一个 GPU Pod 资源申请示例。

要求：声明需要 2 张 GPU，并说明 device plugin 的作用。

练习二：设计一个 TrainingJob CRD。

要求：包含 gpu_type、gpu_count、gpu_per_node、priority、queue、image、command、checkpoint_uri 和 data_uri。

练习三：分析 Pod Pending。

假设一个 64 卡训练任务一直 Pending，请从 GPU 数量、完整节点、quota、taint、node selector、gang scheduling、拓扑和优先级角度排查。

练习四：设计一个 GPU 碎片化治理策略。

要求：包含 bin packing、小任务资源池、大任务预留、抢占和监控指标。

## 12.23 本章小结

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
