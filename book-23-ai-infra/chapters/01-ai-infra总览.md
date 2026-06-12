# 第 1 章 AI Infra 总览：从模型 API 到基础设施底座

很多人第一次接触大模型时，看到的是一个很简单的接口：输入 prompt，调用模型 API，得到回答。

但在真实公司里，模型 API 背后不是一个黑盒，而是一整套基础设施：GPU 集群、网络、存储、调度系统、训练平台、推理平台、数据平台、模型仓库、评估平台、监控告警、权限审计、成本治理和事故处理。

这些东西合起来，就是 AI Infra。

先记住一句话：

> AI Infra 是支撑大模型从数据、训练、评估、部署到在线服务稳定运行的基础设施和平台工程体系。

如果说算法决定“模型能不能变聪明”，那么 AI Infra 决定“模型能不能训得动、推得快、跑得稳、成本可控、出了问题能定位”。

## 1.0 本讲资料边界与第二轮精修口径

本讲是第二十三册入口章，目标不是把 Kubernetes、Slurm、DCGM、OpenTelemetry、MLflow 或某一家云厂商平台逐一讲成教程，而是建立 AI Infra 的分层地图和面试表达框架。

第二轮精修时，我按 `WRITING_PLAN.md` 做了资料校准，主要参考公开官方资料中的稳定边界：Kubernetes device plugin 说明了加速器等厂商设备如何暴露给集群调度；Slurm 是 HPC 场景常见的作业调度和资源管理系统；NVIDIA DCGM 代表 GPU 遥测、健康检查和诊断能力；OpenTelemetry 把 metrics、logs、traces 作为可观测性信号；MLflow Tracking 代表实验记录、参数、指标和 artifact 追踪的一类平台能力。

这些资料只用于校准“AI Infra 应覆盖哪些层”，不能反过来推出“AI Infra 等于 Kubernetes”或“AI Infra 等于某个实验平台”。本讲新增的公式和 demo 也只做教学审计：帮助你把模块清单转成可检查的指标，而不是替代真实公司的容量规划、调度器实现、SRE 体系或安全合规系统。

## 1.1 一个简单 API 背后的复杂系统

用户看到的可能只是：

```text
POST /v1/chat/completions
```

但请求背后可能经过：

1. API Gateway 做认证、限流和路由。
2. 模型路由器选择合适模型和版本。
3. 推理调度器把请求放入队列。
4. 推理引擎做 prefill、decode 和 batching。
5. GPU 执行 attention、MLP、采样和 KV cache 读写。
6. 监控系统记录 TTFT、TPOT、错误率、GPU 利用率。
7. 日志系统记录请求、trace 和异常。
8. 成本系统统计 token、GPU 时间和租户账单。
9. 安全系统检查权限、敏感词、数据留存策略。

训练一个模型更复杂。一个预训练任务可能涉及：

1. 数据从对象存储或数据湖读取。
2. 数据清洗、去重、分片和 shuffle。
3. 训练任务提交到集群调度系统。
4. 容器拉取镜像和依赖环境。
5. 多机多卡建立通信组。
6. GPU 计算 forward 和 backward。
7. 网络传输梯度、参数或激活。
8. checkpoint 定期保存到存储。
9. 监控系统采集 loss、吞吐、显存、网络和 I/O。
10. 节点失败后恢复训练。

所以，大模型不是只运行在代码里，而是运行在基础设施里。

## 1.2 AI Infra 包含哪些模块

AI Infra 可以粗略分成十个模块。

### 1.2.1 计算基础设施

包括 GPU、NPU、TPU、CPU、内存、显存、HBM、PCIe、NVLink、NVSwitch 等。

它决定：

1. 单步训练能放多大 batch。
2. 推理能承载多少并发。
3. 模型能否放进显存。
4. 训练和推理的理论上限。

面试里常见问题是：为什么 GPU 利用率低？为什么显存够但吞吐上不去？为什么多机训练比单机慢很多？这些都和计算基础设施有关。

### 1.2.2 网络基础设施

包括 InfiniBand、RoCE、以太网、交换机、网卡、拓扑、RDMA 和集合通信。

它决定：

1. 多机训练的通信效率。
2. AllReduce、AllGather、ReduceScatter 的耗时。
3. 参数并行和 pipeline 并行的稳定性。
4. checkpoint、数据加载和分布式推理的传输瓶颈。

大模型训练不是 GPU 越多越快。如果网络跟不上，GPU 会互相等待，整体效率会很差。

### 1.2.3 存储基础设施

包括本地 SSD、共享文件系统、对象存储、数据湖、缓存层和 checkpoint 存储。

它决定：

1. 训练数据能否稳定供给。
2. checkpoint 保存和恢复速度。
3. 数据集版本是否可追踪。
4. 多任务并发读取是否会打爆存储。

很多训练慢，不是模型慢，而是数据读不动。

### 1.2.4 调度与资源管理

包括 Kubernetes、Slurm、Ray、队列系统、配额、优先级、抢占、公平调度和 gang scheduling。

它决定：

1. 谁能使用 GPU。
2. 任务什么时候启动。
3. 多租户如何公平共享资源。
4. 训练任务失败后如何恢复。
5. 低优先级任务如何利用碎片资源。

AI Infra 里的调度比普通 Web 服务更难，因为大模型训练通常需要一组 GPU 同时到位，不能只给一部分资源就启动。

### 1.2.5 训练平台

训练平台把“写脚本跑训练”变成“可提交、可复现、可监控、可恢复”的平台能力。

它包括：

1. 任务提交。
2. 镜像和环境管理。
3. 代码版本管理。
4. 数据版本管理。
5. 超参配置。
6. 分布式启动器。
7. 日志和指标。
8. checkpoint 策略。
9. 自动重试和容错。
10. 权限和审计。

训练平台的目标不是让训练脚本消失，而是让训练过程可管理。

### 1.2.6 推理平台

推理平台负责把模型稳定服务给线上用户。

它包括：

1. 模型加载和运行时。
2. Continuous batching。
3. KV cache 管理。
4. 模型路由。
5. 自动扩缩容。
6. 限流、熔断、超时和降级。
7. 灰度发布和回滚。
8. 多模型服务。
9. 监控和告警。
10. 成本统计。

训练关注吞吐和稳定收敛，推理关注延迟、并发、吞吐、可用性和成本。

### 1.2.7 数据平台

数据平台负责数据采集、清洗、存储、版本、血缘、质量监控和访问控制。

在大模型场景里，数据平台尤其重要，因为数据规模大、来源杂、质量差异大，而且训练结果高度依赖数据。

它需要回答：

1. 这批数据从哪里来？
2. 有没有去重？
3. 是否包含敏感信息？
4. 使用了哪个版本？
5. 训练集和评估集是否泄漏？
6. 数据质量变化是否影响模型表现？

### 1.2.8 模型与 Artifact 平台

Artifact 是训练和部署过程中的产物。

包括：

1. 模型权重。
2. tokenizer。
3. 配置文件。
4. checkpoint。
5. adapter。
6. 量化版本。
7. eval report。
8. deployment package。

模型仓库不是简单网盘。它要管理版本、血缘、权限、签名、兼容性、发布状态和回滚。

### 1.2.9 评估与实验平台

评估平台负责离线评测、人工评测、在线 A/B 测试和回归测试。

实验平台负责记录：

1. 使用了什么代码。
2. 使用了什么数据。
3. 使用了什么模型初始化。
4. 使用了什么超参。
5. 得到了什么指标。
6. 产生了什么 artifact。

没有实验追踪，团队很快会陷入“这次为什么变好了”“上次那个模型是怎么训的”这种混乱。

### 1.2.10 可观测性、安全和成本治理

可观测性包括 metrics、logs、traces、events 和 profiles。

安全包括身份、权限、密钥、数据隔离、镜像安全、供应链安全和审计。

成本治理包括 GPU 成本、存储成本、网络成本、推理 token 成本和闲置资源成本。

AI Infra 的生产化水平，很大程度体现在这些“看起来不酷但非常关键”的能力上。

## 1.3 AI Infra 和 MLOps、LLMOps、Platform Engineering 的关系

这些词经常混在一起，需要区分。

### 1.3.1 MLOps

MLOps 关注机器学习模型从开发到上线的生命周期管理。

典型内容包括：

1. 数据版本。
2. 特征管理。
3. 训练流水线。
4. 模型注册。
5. 部署发布。
6. 监控回滚。

MLOps 更早服务于传统机器学习和深度学习。

### 1.3.2 LLMOps

LLMOps 是面向大模型的生命周期管理。

相比 MLOps，它更强调：

1. Prompt 管理。
2. RAG 知识库。
3. 大模型评测。
4. Agent trace。
5. 工具调用日志。
6. 模型供应商管理。
7. token 成本。
8. 安全和内容治理。

LLMOps 更贴近大模型应用层。

### 1.3.3 Platform Engineering

Platform Engineering 关注为内部开发者提供自助式平台。

它的目标是让算法工程师、应用工程师和业务团队不用直接面对底层复杂性。

例如：

1. 自助提交训练任务。
2. 自助部署模型服务。
3. 自助查看指标和日志。
4. 自助申请数据和权限。
5. 自助创建评测任务。

### 1.3.4 AI Infra

AI Infra 更底层，也更综合。

它覆盖硬件、集群、网络、存储、调度、平台、治理和运维。

可以这样理解：

```text
AI Infra 是底座。
MLOps / LLMOps 是生命周期管理方法和平台能力。
Platform Engineering 是把这些能力产品化给内部用户使用的工程方式。
```

## 1.4 为什么算法岗也要懂 AI Infra

很多算法同学会问：我又不是 SRE，为什么要懂 GPU 集群、调度和存储？

原因很现实。

### 1.4.1 训练问题经常不是算法问题

例如：

1. loss 突然变 NaN，可能是混合精度、数据异常或通信问题。
2. 训练吞吐下降，可能是数据加载慢、网络拥塞或 checkpoint 阻塞。
3. 多机扩展效率差，可能是并行策略和网络拓扑不匹配。
4. 任务经常失败，可能是节点稳定性、驱动版本或容器环境问题。

如果完全不懂 Infra，很容易把所有问题都误判为模型或代码问题。

### 1.4.2 推理效果和推理系统有关

同一个模型，不同推理平台可能表现完全不同：

1. batching 策略影响延迟和吞吐。
2. KV cache 策略影响并发和显存。
3. 模型路由影响成本和质量。
4. 超时策略影响用户体验。
5. 降级策略影响可用性。

算法岗如果不了解推理平台，就很难设计可落地的模型方案。

### 1.4.3 成本是模型方案的一部分

大模型项目里，成本不是财务部门的问题，而是技术方案的一部分。

一个模型如果效果提升 1%，但推理成本增加 5 倍，未必值得上线。

一个训练方案如果理论上更好，但需要占用整个集群两周，也可能不现实。

优秀算法工程师需要能回答：这个方案效果、成本、延迟和资源占用之间如何权衡。

## 1.5 AI Infra 的核心矛盾

AI Infra 不是堆机器，而是在多个目标之间做 trade-off。

### 1.5.1 性能与成本

更大的 GPU 集群、更快的网络、更高性能的存储都能提升效率，但成本很高。

问题是：哪些瓶颈值得花钱解决？哪些可以通过软件优化解决？

例如：

1. 如果训练瓶颈是数据读取，买更多 GPU 没用。
2. 如果推理瓶颈是 KV cache 显存，增加 CPU 不解决问题。
3. 如果网络慢，模型并行策略可能需要调整。

### 1.5.2 利用率与隔离性

为了提高资源利用率，平台希望混部、抢占和共享资源。

为了保证稳定性，平台又需要隔离、配额和优先级。

两者天然冲突。

例如低优先级训练任务可以使用空闲 GPU，但高优先级任务来了以后是否抢占？抢占后 checkpoint 如何恢复？这些都是调度系统要解决的问题。

### 1.5.3 灵活性与标准化

算法团队希望自由选择框架、镜像、依赖、启动方式和实验配置。

平台团队希望标准化，方便运维、监控、安全和复现。

好的平台不是禁止灵活性，而是在关键边界上标准化：任务描述、资源申请、日志格式、checkpoint 位置、指标上报和权限模型。

### 1.5.4 速度与治理

研发希望快速试验和上线。

企业需要权限、审计、合规、成本控制和变更管理。

AI Infra 的成熟度体现在：在不拖慢研发太多的情况下，把治理嵌入平台默认流程。

## 1.6 一个完整 AI Infra 视图

可以用下面的图理解：

```text
Users / Teams
  -> Portal / CLI / SDK
  -> Training Platform
  -> Inference Platform
  -> Data Platform
  -> Eval Platform
  -> Model Registry / Artifact Store
  -> Scheduler / Resource Manager
  -> GPU Cluster / Network / Storage
  -> Observability / Security / Cost Governance
```

从上往下看：用户通过平台入口提交任务。

从下往上看：底层硬件资源被抽象成训练、推理、数据和评估能力。

横向看：监控、安全和成本治理贯穿所有层。

## 1.7 面试中如何讲 AI Infra 总览

如果面试官问：

```text
你怎么理解 AI Infra？它包含哪些模块？
```

可以这样回答：

```text
我理解 AI Infra 是支撑大模型训练、推理、评估和应用运行的基础设施与平台工程体系。

底层包括 GPU/加速器集群、网络、存储和调度系统；中间包括训练平台、推理平台、数据平台、模型仓库、实验追踪和评估平台；上层包括面向算法和应用团队的 Portal、CLI、SDK；横向能力包括可观测性、权限审计、安全治理、成本治理和容量规划。

它解决的问题不是单纯把模型跑起来，而是让模型训得动、推得快、服务稳、可复现、可观测、可扩展、成本可控。
```

如果面试官继续追问：

```text
AI Infra 和 MLOps / LLMOps 有什么区别？
```

可以回答：

```text
AI Infra 更偏底座和平台，覆盖硬件、集群、网络、存储、调度、训练和推理服务。MLOps / LLMOps 更偏模型生命周期管理，包括数据、训练、评估、部署、监控和回滚。LLMOps 还会加入 prompt、RAG、Agent trace、工具调用和 token 成本等大模型特有内容。Platform Engineering 则是把这些能力产品化给内部团队自助使用。
```

## 1.8 AI Infra 总览审计指标与最小 demo

总览章节最容易写成“列一堆模块名”。第二轮精修里，更推荐把它转成一组可审计问题：这个平台是否真的覆盖了训练、推理、数据、评估、可观测性、安全、成本和开发者体验？它是否把 AI Infra、MLOps、LLMOps 和 Platform Engineering 的边界讲清楚？算法团队和平台团队是否能用同一张指标表沟通？

先定义一个 AI Infra 总览审计样本：

```math
x_i=(c_i,n_i,s_i,q_i,t_i,r_i,d_i,a_i,e_i,o_i,g_i,p_i,z_i)
```

其中，`c_i` 表示计算和加速器准备度，`n_i` 表示网络和通信准备度，`s_i` 表示存储、数据供给和 checkpoint 准备度，`q_i` 表示调度和资源治理，`t_i` 表示训练平台可复现性，`r_i` 表示推理平台 SLO 能力，`d_i` 表示数据平台血缘，`a_i` 表示模型与 artifact 治理，`e_i` 表示评估和实验追踪，`o_i` 表示可观测性，`g_i` 表示安全治理，`p_i` 表示成本和容量治理，`z_i` 表示边界、协作和开发者自助能力等标签。

对第 `j` 个审计维度，统一写成通过率：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(x_i)=1]
```

其中，`N` 是审计样本数，`g_j(x_i)=1` 表示样本 `x_i` 通过第 `j` 个检查。

常见资源效率可以用 GPU 忙碌时间占比做直觉起点：

```math
U_{\mathrm{gpu}}=\frac{T_{\mathrm{busy}}}{T_{\mathrm{wall}}}
```

但面试中要强调：GPU utilization 只是粗粒度忙碌指标，不能单独证明训练有效率。还要结合 MFU、tokens/sec、step time、通信时间、数据加载时间、checkpoint 阻塞和失败率。

成本治理可以先用单位成功任务成本表达：

```math
K_{\mathrm{success}}=\frac{K_{\mathrm{gpu}}+K_{\mathrm{storage}}+K_{\mathrm{network}}+K_{\mathrm{ops}}}{N_{\mathrm{success}}}
```

其中，`K_gpu`、`K_storage`、`K_network` 和 `K_ops` 分别代表 GPU、存储、网络和运维成本，`N_success` 是通过质量门禁的成功训练、评估或推理任务数。这个公式的重点不是精确财务建模，而是提醒你：只算 GPU 小时或 token 单价，会低估 AI Infra 的真实成本。

最后给一个总览门禁：

```math
G_{\mathrm{ai\_infra}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{p0}}=0\right]
```

其中，`\tau_j` 是每个维度的最低通过阈值，`R_p0` 表示 P0 级硬阻断数量。只要关键维度覆盖不足，或者存在没有缓解方案的 P0 风险，总览门禁就不应该通过。

下面是一个 0 依赖 Python demo。它不模拟真实集群，只把“AI Infra 总览回答是否完整”变成一张 toy 审计表：

```python
METRICS = [
    "compute_accelerator_readiness",
    "network_communication_readiness",
    "storage_data_checkpoint_readiness",
    "scheduler_resource_governance",
    "training_platform_reproducibility",
    "inference_platform_slo_readiness",
    "data_platform_lineage_quality",
    "model_artifact_registry_governance",
    "eval_experiment_tracking_coverage",
    "observability_signal_coverage",
    "security_governance_coverage",
    "cost_capacity_governance",
    "developer_self_service_readiness",
    "ai_infra_boundary_clarity",
    "mlops_llmops_platform_boundary_clarity",
    "algorithm_infra_collaboration_readiness",
]


def make_case(name, failed_metric=None, p0=False):
    flags = {metric: True for metric in METRICS}
    if failed_metric is not None:
        flags[failed_metric] = False
    return {"name": name, "flags": flags, "p0": p0}


def build_cases():
    bad_cases = [
        ("gpu_capacity_missing_bad", "compute_accelerator_readiness"),
        ("network_ignored_bad", "network_communication_readiness"),
        ("storage_checkpoint_missing_bad", "storage_data_checkpoint_readiness"),
        ("scheduler_no_quota_bad", "scheduler_resource_governance"),
        ("training_not_reproducible_bad", "training_platform_reproducibility"),
        ("inference_no_slo_bad", "inference_platform_slo_readiness"),
        ("data_lineage_missing_bad", "data_platform_lineage_quality"),
        ("artifact_registry_missing_bad", "model_artifact_registry_governance"),
        ("eval_tracking_missing_bad", "eval_experiment_tracking_coverage"),
        ("observability_missing_bad", "observability_signal_coverage"),
        ("security_audit_missing_bad", "security_governance_coverage"),
        ("cost_capacity_missing_bad", "cost_capacity_governance"),
        ("no_self_service_bad", "developer_self_service_readiness"),
        ("kubernetes_only_bad", "ai_infra_boundary_clarity"),
        ("boundary_confused_bad", "mlops_llmops_platform_boundary_clarity"),
        ("algorithm_infra_silo_bad", "algorithm_infra_collaboration_readiness"),
    ]

    cases = [make_case("complete_ai_infra_stack")]
    cases.extend(make_case(name, metric, p0=True) for name, metric in bad_cases)
    return cases


def audit_ai_infra_overview(cases, threshold=0.95):
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
        "ai_infra_gate_pass": gate_pass,
    }


cases = build_cases()
report = audit_ai_infra_overview(cases)

smoke = {
    "complete_case_passes": "complete_ai_infra_stack" not in report["failed_cases"],
    "caught_kubernetes_only": "kubernetes_only_bad" in report["failed_cases"],
    "caught_network_ignored": "network_ignored_bad" in report["failed_cases"],
    "caught_inference_no_slo": "inference_no_slo_bad" in report["failed_cases"],
    "caught_boundary_confused": "boundary_confused_bad" in report["failed_cases"],
}

print("smoke=", smoke)
print("metrics=", report["metrics"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("ai_infra_gate_pass=", report["ai_infra_gate_pass"])
```

这段 demo 的设计故意让 16 个 bad case 分别打穿 16 个维度，因此每个覆盖率都是 `16/17=0.941`，低于 `0.95` 阈值。它想表达的不是“真实平台必须有 16 个指标”，而是面试中不能只说“我们用了 Kubernetes”或“我们买了 GPU”。一个可上线的 AI Infra 总览回答，需要同时覆盖计算、网络、存储、调度、训练、推理、数据、artifact、评估、可观测性、安全、成本、开发者体验和跨团队协作。

## 1.9 常见误区

误区一：AI Infra 等于 Kubernetes。

Kubernetes 只是资源管理和容器编排的一部分。AI Infra 还包括 GPU、网络、存储、训练平台、推理平台、数据平台、评估平台、可观测性和治理。

误区二：AI Infra 等于买 GPU。

GPU 是基础，但没有调度、网络、存储、平台和运维，GPU 很容易变成昂贵但低利用率的资源。

误区三：算法岗不用懂 Infra。

大模型训练和推理问题经常跨越算法、系统和平台边界。懂 Infra 能帮助你更快定位问题，也能设计更可落地的方案。

误区四：推理平台只是把模型部署成服务。

推理平台还要处理 batching、KV cache、模型路由、限流、自动扩缩容、灰度、回滚、监控和成本。

误区五：监控是上线之后再补。

大模型系统如果没有可观测性，出了问题很难定位。监控、日志、trace 和事件应该从平台设计一开始就纳入。

## 1.10 面试题

### 题 1：AI Infra 和普通后端基础设施有什么不同？

答：AI Infra 更依赖 GPU/加速器、显存、高速网络、大规模数据读取、checkpoint 存储和模型运行时。训练任务通常长时间运行、资源需求大、需要 gang scheduling 和容错；推理服务则需要处理 token 级延迟、KV cache、batching 和模型版本治理。

### 题 2：为什么 GPU 多不一定训练快？

答：训练速度还受网络通信、并行策略、数据读取、存储吞吐、checkpoint、CPU 预处理和调度影响。如果通信或 I/O 成为瓶颈，增加 GPU 只会增加等待时间，扩展效率可能下降。

### 题 3：训练平台和推理平台的关注点有什么不同？

答：训练平台关注资源申请、分布式启动、数据供给、checkpoint、日志、指标、容错和可复现性。推理平台关注模型加载、请求路由、延迟、吞吐、batching、KV cache、限流、熔断、灰度、回滚和成本。

### 题 4：AI Infra 为什么需要成本治理？

答：GPU、存储、网络和推理 token 都很贵。没有成本治理，容易出现 GPU 闲置、低利用率训练、重复实验、无效 checkpoint、过度扩容和高成本模型滥用。成本治理帮助团队在效果、延迟和资源之间做合理权衡。

### 题 5：为什么可观测性对 AI Infra 特别重要？

答：大模型系统链路长、组件多、故障原因复杂。训练慢可能来自数据、网络、GPU、框架或存储；推理慢可能来自队列、prefill、decode、KV cache 或路由。没有 metrics、logs、traces 和 events，就很难定位问题。

## 1.11 小练习

练习一：画出你理解的大模型训练平台架构图。

要求：至少包含用户入口、任务提交、调度器、GPU 集群、数据存储、checkpoint、日志和监控。

练习二：画出你理解的大模型推理平台架构图。

要求：至少包含 API Gateway、模型路由、推理引擎、batching、KV cache、自动扩缩容、监控和成本统计。

练习三：分析一个 GPU 利用率低的问题。

假设训练任务 GPU 利用率只有 35%，请列出至少 8 个可能原因，并说明你会先看哪些指标。

练习四：区分 AI Infra、MLOps、LLMOps 和 Platform Engineering。

要求：用自己的话分别解释它们关注什么，并举一个具体例子。

## 1.12 本章小结

本章是第二十三册的入口。

你需要掌握：

1. AI Infra 是支撑大模型训练、推理、评估和应用运行的基础设施与平台工程体系。
2. 模型 API 背后包含 GPU、网络、存储、调度、推理引擎、监控、安全和成本治理。
3. AI Infra 主要模块包括计算、网络、存储、调度、训练平台、推理平台、数据平台、模型仓库、评估平台和治理体系。
4. AI Infra 和 MLOps、LLMOps、Platform Engineering 有交集，但 AI Infra 更偏底层和平台底座。
5. 算法岗也要懂 AI Infra，因为训练效率、推理延迟、稳定性和成本都与基础设施密切相关。
6. AI Infra 的核心 trade-off 包括性能与成本、利用率与隔离性、灵活性与标准化、速度与治理。

后面的章节会逐步拆开这些模块：先讲 GPU、显存、网络和存储，再讲调度、训练平台、推理平台、数据平台、可观测性、成本治理和系统设计。
