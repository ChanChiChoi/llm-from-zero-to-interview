# 第 2 章 AI Infra、MLOps、LLMOps、Platform Engineering 的边界

上一章我们从整体上认识了 AI Infra：它是支撑大模型训练、推理、评估和应用运行的基础设施与平台工程体系。

但在真实团队里，你会经常听到几个相近的词：AI Infra、MLOps、LLMOps、Platform Engineering、DevOps、Data Platform、Model Platform。它们有交集，但不是同一个东西。

如果边界讲不清，面试时很容易出现两种问题：要么把 AI Infra 说成 Kubernetes，要么把 LLMOps 说成 prompt 管理工具。

本章目标是把这些概念拆清楚。

先记住一句话：

> AI Infra 更像底座，MLOps / LLMOps 更像模型生命周期管理，Platform Engineering 更像把底座能力产品化给内部团队使用的方法。

## 2.0 本讲资料边界与第二轮精修口径

本讲讨论的是概念边界和职责切分，不是给某个组织画固定组织架构，也不是定义行业统一岗位名称。真实公司里，AI Infra、ML Platform、LLMOps、Data Platform、Model Platform、SRE 和 Platform Engineering 经常会重叠，重要的是交付物、接口和责任边界可审计。

第二轮精修时，我按 `WRITING_PLAN.md` 做了资料校准，主要参考公开官方资料中的稳定边界：Google Cloud 的 MLOps 资料强调 ML 系统开发与运维、自动化、监控、CI/CD/CT、数据和模型验证；MLflow 的 LLM / Agent 文档把 tracing、evaluation、prompt management、AI gateway、agent serving 和 monitoring 放在 LLM 应用工程能力里；CNCF Platforms White Paper 把平台定义成面向内部用户的一组能力、接口、模板、自助 API 和一致体验；OpenTelemetry 用 traces、metrics、logs、baggage 等 signals 描述可观测性基础；Google SRE 的 SLI / SLO 框架强调可靠性目标要由用户关心的指标驱动。

因此，本章新增内容采用“职责边界审计”口径：不说某个工具天然属于某个团队，而是看它解决的是资源运行、模型生命周期、大模型应用生命周期、内部开发者体验、数据底座、模型资产管理还是可靠性治理问题。

## 2.1 为什么边界容易混乱

这些概念混乱，主要有四个原因。

第一，它们都和“让模型上线”有关。

训练、评估、部署、监控、回滚、数据、权限、成本，每个概念都会碰到其中一部分。

第二，大模型改变了传统机器学习平台的边界。

过去 MLOps 主要服务传统 ML 模型，例如特征工程、模型训练、模型部署。大模型出现后，又增加了 prompt、RAG、Agent、工具调用、token 成本、多模型路由、人工评测和安全治理。

第三，公司组织架构不同。

有的公司 AI Infra 团队负责 GPU 集群和训练平台；有的公司叫 ML Platform；有的公司把推理平台放在在线服务团队；有的公司把 LLMOps 做成应用平台。

第四，厂商命名不统一。

云厂商、模型厂商、开源项目和企业内部平台会用不同名字包装相似能力。

所以不要死记名字，要看它解决的问题。

## 2.2 一张总览图

可以用下面的分层理解：

```text
Business / Product / AI Applications
  -> LLMOps: prompt, RAG, Agent, eval, model routing, safety
  -> MLOps: data, training pipeline, experiment, model registry, deployment
  -> Platform Engineering: portal, CLI, SDK, self-service, golden path
  -> AI Infra: GPU cluster, network, storage, scheduler, runtime, observability, security, cost
  -> Hardware / Cloud / Data Center
```

这张图不是严格上下级关系，而是强调侧重点。

1. AI Infra 关心底层资源和平台能力。
2. MLOps 关心机器学习生命周期。
3. LLMOps 关心大模型应用生命周期。
4. Platform Engineering 关心内部开发者体验和自助化。

## 2.3 AI Infra 的核心边界

AI Infra 的关键词是“资源、平台、运行时、治理”。

它负责让大模型工作负载有地方跑、跑得稳、跑得快、可观测、可治理。

典型范围包括：

1. GPU / NPU / TPU 集群。
2. 高速网络和集合通信。
3. 本地盘、共享文件系统、对象存储和数据湖。
4. Kubernetes、Slurm、Ray 等调度系统。
5. 多租户资源配额和优先级。
6. 训练平台。
7. 推理平台。
8. 模型运行时和 serving runtime。
9. 日志、指标、trace、event 和 profile。
10. 权限、安全、审计和成本治理。

AI Infra 关注的问题通常是：

1. GPU 为什么闲着？
2. 多机训练为什么扩展效率低？
3. checkpoint 为什么保存很慢？
4. 推理请求为什么排队？
5. p95 延迟为什么突然升高？
6. 哪些团队占用了最多 GPU？
7. 哪个模型版本造成错误率上升？
8. 如何让训练任务失败后自动恢复？

一句话：AI Infra 管的是“底座能不能支撑 AI 工作负载”。

## 2.4 MLOps 的核心边界

MLOps 的关键词是“机器学习生命周期”。

它关注一个模型从数据到训练、评估、部署、监控和迭代的完整流程。

典型范围包括：

1. 数据集版本管理。
2. 特征工程和 Feature Store。
3. 训练 pipeline。
4. 实验追踪。
5. 模型注册和模型仓库。
6. 模型评估。
7. 模型部署。
8. 在线监控。
9. 数据漂移和模型漂移检测。
10. 回滚和再训练。

传统 MLOps 场景里，模型可能是推荐模型、投放模型、风控模型、CV 模型或小型 NLP 模型。

MLOps 关注的问题通常是：

1. 这次训练用了哪个数据版本？
2. 超参和代码版本是什么？
3. 实验指标有没有提升？
4. 模型是否通过上线门禁？
5. 线上数据分布是否变化？
6. 模型效果下降后如何触发再训练？
7. 如何回滚到上一个稳定模型？

一句话：MLOps 管的是“机器学习模型生命周期是否可复现、可发布、可监控、可迭代”。

## 2.5 LLMOps 的核心边界

LLMOps 是大模型时代的模型和应用生命周期管理。

它继承 MLOps 的一部分，但又有明显新增内容。

典型范围包括：

1. Prompt 管理。
2. Prompt 版本和回归测试。
3. RAG 知识库管理。
4. Embedding 和向量索引管理。
5. 大模型评测。
6. 人工评测和偏好数据。
7. 模型路由和多模型选择。
8. Agent trace。
9. 工具调用日志和安全策略。
10. token 成本治理。
11. 内容安全和输出审核。
12. provider 管理。

LLMOps 关注的问题通常是：

1. 这个 prompt 版本是否比上一个好？
2. RAG 检索结果是否相关？
3. 模型回答是否有 hallucination？
4. Agent 调用了哪些工具？
5. 工具调用是否越权？
6. token 成本是否超预算？
7. 哪个模型更适合当前任务？
8. 安全策略是否拦截了风险输出？

一句话：LLMOps 管的是“大模型应用从 prompt、RAG、Agent 到评估和上线的生命周期”。

## 2.6 Platform Engineering 的核心边界

Platform Engineering 的关键词是“内部开发者平台”和“自助化”。

它不是某个 AI 专属概念，而是一种工程组织方式。

它的目标是把复杂基础设施包装成内部团队容易使用的产品。

典型能力包括：

1. Portal。
2. CLI。
3. SDK。
4. 模板和最佳实践。
5. 自助申请资源。
6. 自助提交任务。
7. 自助部署服务。
8. 自助查看日志和指标。
9. 权限申请和审批流。
10. 标准化 golden path。

Platform Engineering 关注的问题通常是：

1. 算法工程师能否自己提交训练任务？
2. 应用工程师能否自己部署模型服务？
3. 团队是否必须找平台同学手工开资源？
4. 新项目是否有标准模板？
5. 平台是否把安全和监控作为默认能力？
6. 开发者是否知道任务失败原因？

一句话：Platform Engineering 管的是“如何让内部用户高效、安全、标准化地使用平台能力”。

## 2.7 DevOps、SRE 和 AI Infra 的关系

DevOps 强调开发和运维协作，关注自动化交付、CI/CD、环境一致性和发布效率。

SRE 强调可靠性工程，关注 SLO、错误预算、监控告警、事故响应和容量规划。

AI Infra 会借鉴 DevOps 和 SRE，但 AI 工作负载有特殊性：

1. GPU 资源昂贵且稀缺。
2. 训练任务长时间运行。
3. 分布式训练需要多节点同时可用。
4. checkpoint 和数据吞吐压力大。
5. 推理延迟和 token 生成过程强相关。
6. 模型效果不是普通服务健康检查能完全衡量。
7. 数据、模型和 prompt 都需要版本治理。

所以 AI Infra 不是简单把 Web 服务那套搬过来，而是把 DevOps / SRE 思想适配到 AI 工作负载。

## 2.8 Data Platform 和 AI Infra 的关系

Data Platform 关注数据采集、计算、存储、治理和服务。

典型组件包括：

1. 数据湖。
2. 数仓。
3. 流式计算。
4. 离线计算。
5. 数据质量。
6. 数据血缘。
7. 权限和合规。

AI Infra 依赖 Data Platform，因为训练数据、评估数据、RAG 文档和业务日志都来自数据平台。

但 AI Infra 还会额外关注 AI 特有的数据问题：

1. 大规模样本读取吞吐。
2. 训练数据分片和 shuffle。
3. 数据与 checkpoint 的版本对应关系。
4. 训练集和评估集污染。
5. embedding 索引更新。
6. 多模态数据存储。

一句话：Data Platform 提供数据底座，AI Infra 把数据底座接入训练、评估、推理和 Agent 工作负载。

## 2.9 Model Platform 和 AI Infra 的关系

Model Platform 通常指围绕模型资产的管理平台。

包括：

1. 模型注册。
2. 模型版本。
3. 权重存储。
4. tokenizer 和配置管理。
5. adapter 和 LoRA 管理。
6. 量化版本。
7. 模型评测报告。
8. 模型发布状态。
9. 权限和审批。

它和 AI Infra 的关系是：

1. AI Infra 提供存储、训练和推理底座。
2. Model Platform 管理模型 artifact 和生命周期。
3. 推理平台从 Model Platform 拉取模型版本。
4. 评估平台把评测结果回写到 Model Platform。

在成熟系统里，模型不是文件路径，而是带 metadata、权限、血缘、评估结果和发布状态的资产。

## 2.10 一个例子：训练一个新模型

假设团队要训练一个新的客服大模型。

AI Infra 负责：

1. 提供 GPU 集群。
2. 提供网络和存储。
3. 调度训练任务。
4. 提供训练平台。
5. 采集训练指标和日志。
6. 保存 checkpoint。
7. 处理节点失败和重试。

MLOps 负责：

1. 记录数据版本。
2. 记录训练配置。
3. 记录实验指标。
4. 注册模型版本。
5. 管理评估和上线流程。

LLMOps 负责：

1. 评测对话质量。
2. 管理 prompt 模板。
3. 测试 RAG 效果。
4. 跟踪安全和幻觉指标。
5. 管理上线后的 token 成本。

Platform Engineering 负责：

1. 提供自助提交入口。
2. 提供标准训练模板。
3. 提供查看日志和指标的页面。
4. 提供权限申请流程。
5. 降低用户使用底层平台的复杂度。

同一件事，不同层关注点不同。

## 2.11 一个例子：上线一个 RAG 应用

假设团队要上线一个企业知识库问答系统。

AI Infra 负责：

1. 推理服务资源。
2. Embedding 服务。
3. 向量索引运行环境。
4. 存储和网络。
5. 监控、日志、限流和成本统计。

MLOps 可能负责：

1. embedding 模型版本。
2. reranker 模型版本。
3. 模型评估和发布。

LLMOps 负责：

1. prompt 版本。
2. RAG 召回评估。
3. 答案正确性评估。
4. 引用来源检查。
5. Agent 或工具调用 trace。
6. 内容安全和输出审核。

Platform Engineering 负责：

1. 提供知识库接入模板。
2. 提供应用创建向导。
3. 提供默认监控面板。
4. 提供灰度发布流程。
5. 提供开发者文档和 SDK。

这个例子能说明：LLMOps 更接近大模型应用层，而 AI Infra 提供运行底座。

## 2.12 常见组织形态

不同公司会有不同团队拆分。

小团队里，一个平台团队可能同时负责：

1. GPU 集群。
2. 训练平台。
3. 推理平台。
4. LLMOps 工具。
5. 监控和成本。

中型团队可能拆成：

1. Infra 团队：集群、网络、存储、调度。
2. Training Platform 团队：训练任务、checkpoint、实验追踪。
3. Inference Platform 团队：模型服务、路由、batching、扩缩容。
4. LLMOps 团队：prompt、RAG、评估、Agent trace。
5. Data Platform 团队：数据湖、数据质量、数据权限。

大型公司可能进一步拆分出：

1. GPU Fleet 团队。
2. Networking 团队。
3. Storage 团队。
4. Scheduler 团队。
5. Model Runtime 团队。
6. Eval Platform 团队。
7. Safety Platform 团队。
8. Cost Governance 团队。

面试时不要纠结名称，而要能说明每个团队解决什么问题。

## 2.13 如何判断一个需求属于哪一层

可以用下面的问题判断。

如果问题是“资源在哪里、怎么调度、怎么跑稳”，通常属于 AI Infra。

如果问题是“模型从训练到上线怎么管理”，通常属于 MLOps。

如果问题是“prompt、RAG、Agent、工具调用和大模型评测怎么管理”，通常属于 LLMOps。

如果问题是“内部用户怎么自助使用这些能力”，通常属于 Platform Engineering。

举例：

1. GPU 利用率低：AI Infra。
2. 训练实验不可复现：MLOps。
3. Prompt 改动导致线上回答变差：LLMOps。
4. 算法同学提交训练任务太复杂：Platform Engineering。
5. 模型权重版本混乱：MLOps / Model Platform。
6. RAG 检索质量下降：LLMOps / Data Platform。
7. 推理 p95 延迟升高：AI Infra / Inference Platform。
8. 线上模型回答有安全风险：LLMOps / Safety Platform。

## 2.14 面试回答模板

如果面试官问：

```text
AI Infra、MLOps、LLMOps、Platform Engineering 有什么区别？
```

可以这样回答：

```text
我会从关注点区分。

AI Infra 更偏底层基础设施和平台能力，负责 GPU 集群、网络、存储、调度、训练平台、推理平台、可观测性、安全和成本治理，目标是让 AI 工作负载跑得动、跑得快、跑得稳。

MLOps 更偏机器学习生命周期，负责数据版本、训练 pipeline、实验追踪、模型注册、评估、部署、监控和回滚，目标是让模型开发和上线可复现、可管理。

LLMOps 是大模型应用生命周期，除了继承 MLOps 一部分能力，还重点关注 prompt、RAG、Agent、工具调用、模型路由、大模型评测、安全和 token 成本。

Platform Engineering 是一种产品化内部平台的方式，把上述能力通过 Portal、CLI、SDK、模板和自助流程提供给算法和应用团队使用。

这几个概念有交集，但不是同义词。AI Infra 是底座，MLOps / LLMOps 是生命周期管理，Platform Engineering 是面向内部用户的交付方式。
```

## 2.15 边界审计指标与最小 demo

边界题最容易答成“背名词”。更可靠的方式是把每个需求映射到主责层、协作层、交付物和门禁。

先定义一个需求或事故样本：

```math
b_i=(u_i,y_i,h_i,a_i,m_i,l_i,p_i,d_i,o_i,s_i,c_i,z_i)
```

其中，`u_i` 是用户或需求场景，`y_i` 是应该主责的层，`h_i` 是当前回答或系统设计给出的主责层，`a_i` 是 AI Infra 资源运行职责，`m_i` 是 MLOps 生命周期职责，`l_i` 是 LLMOps 应用生命周期职责，`p_i` 是 Platform Engineering 开发者体验职责，`d_i` 是 Data / Model Platform 资产职责，`o_i` 是可观测性和 SLO 证据，`s_i` 是安全、权限和成本治理，`c_i` 是跨团队接口，`z_i` 是标签、风险等级和复盘信息。

分类错误率可以写成：

```math
R_{\mathrm{route}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[h_i\ne y_i]
```

第 `j` 个边界维度的覆盖率仍然写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(b_i)=1]
```

如果要把职责矩阵写清楚，可以用：

```math
A_{k,m}\in\{0,1\}
```

其中，`A_{k,m}=1` 表示第 `k` 个平台层对第 `m` 个能力有主责或硬门禁责任。注意：主责不代表独占。比如模型 registry 可以属于 MLOps / Model Platform，但它要和 AI Infra 的存储、推理平台的模型加载、评估平台的报告和权限系统打通。

边界门禁可以写成：

```math
G_{\mathrm{boundary}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land R_{\mathrm{route}}=0 \land P_0=0\right]
```

其中，`\tau_j` 是各维度最低通过率，`P_0` 是 P0 级边界风险数量。边界风险的典型形式是：出了事故没人主责、上线前缺硬门禁、平台接口没有 owner、模型版本和数据版本断链、安全策略只写在文档里、成本和 SLO 无法归因。

下面是一个 0 依赖 Python demo。它把本章概念做成 toy 边界审计表：

```python
METRICS = [
    "ai_infra_scope_accuracy",
    "mlops_lifecycle_accuracy",
    "llmops_application_accuracy",
    "platform_engineering_dx_accuracy",
    "devops_sre_boundary_accuracy",
    "data_platform_boundary_accuracy",
    "model_platform_boundary_accuracy",
    "primary_owner_clarity",
    "interface_contract_coverage",
    "artifact_lineage_handoff",
    "observability_slo_handoff",
    "security_cost_governance_handoff",
    "lifecycle_stage_mapping",
    "anti_tool_name_confusion",
    "incident_routing_accuracy",
    "collaboration_handoff_readiness",
]


def make_case(name, expected_owner, actual_owner=None, failed_metric=None, p0=False):
    flags = {metric: True for metric in METRICS}
    if failed_metric is not None:
        flags[failed_metric] = False
    return {
        "name": name,
        "expected_owner": expected_owner,
        "actual_owner": actual_owner or expected_owner,
        "flags": flags,
        "p0": p0,
    }


def build_cases():
    bad_cases = [
        ("gpu_queue_called_mlops_bad", "AI Infra", "MLOps", "ai_infra_scope_accuracy"),
        ("model_registry_no_lifecycle_bad", "MLOps", "AI Infra", "mlops_lifecycle_accuracy"),
        ("llmops_prompt_only_bad", "LLMOps", "Prompt Tool", "llmops_application_accuracy"),
        ("platform_as_devops_rename_bad", "Platform Engineering", "DevOps", "platform_engineering_dx_accuracy"),
        ("slo_missing_for_serving_bad", "SRE", "AI Infra", "devops_sre_boundary_accuracy"),
        ("rag_doc_freshness_misrouted_bad", "Data Platform", "LLMOps", "data_platform_boundary_accuracy"),
        ("weight_file_no_model_platform_bad", "Model Platform", "Object Storage", "model_platform_boundary_accuracy"),
        ("no_primary_owner_bad", "AI Infra", "Unknown", "primary_owner_clarity"),
        ("no_interface_contract_bad", "Platform Engineering", "Ad Hoc Ticket", "interface_contract_coverage"),
        ("artifact_lineage_broken_bad", "MLOps", "Unknown", "artifact_lineage_handoff"),
        ("trace_slo_not_connected_bad", "SRE", "Monitoring Only", "observability_slo_handoff"),
        ("cost_security_not_handoff_bad", "AI Infra", "Finance", "security_cost_governance_handoff"),
        ("lifecycle_stage_missing_bad", "MLOps", "Script", "lifecycle_stage_mapping"),
        ("tool_name_confusion_bad", "LLMOps", "Kubernetes", "anti_tool_name_confusion"),
        ("rag_incident_wrong_route_bad", "LLMOps", "AI Infra", "incident_routing_accuracy"),
        ("team_handoff_missing_bad", "Platform Engineering", "Manual Ops", "collaboration_handoff_readiness"),
    ]

    cases = [make_case("complete_boundary_map", "Boundary Board")]
    cases.extend(
        make_case(name, expected, actual, metric, p0=True)
        for name, expected, actual, metric in bad_cases
    )
    return cases


def audit_boundary(cases, threshold=0.95):
    metrics = {}
    for metric in METRICS:
        passed = sum(1 for case in cases if case["flags"][metric])
        metrics[metric] = round(passed / len(cases), 3)

    misrouted_cases = [
        case["name"]
        for case in cases
        if case["expected_owner"] != case["actual_owner"]
    ]
    failed_cases = [
        case["name"]
        for case in cases
        if case["p0"] or any(not case["flags"][metric] for metric in METRICS)
    ]
    failed_gates = [
        metric for metric, score in metrics.items() if score < threshold
    ]
    hard_blocker_count = sum(1 for case in cases if case["p0"])
    route_error_rate = round(len(misrouted_cases) / len(cases), 3)
    gate_pass = not failed_gates and not misrouted_cases and hard_blocker_count == 0

    return {
        "metrics": metrics,
        "route_error_rate": route_error_rate,
        "hard_blocker_count": hard_blocker_count,
        "misrouted_cases": misrouted_cases,
        "failed_cases": failed_cases,
        "failed_gates": failed_gates,
        "boundary_gate_pass": gate_pass,
    }


cases = build_cases()
report = audit_boundary(cases)

smoke = {
    "complete_case_passes": "complete_boundary_map" not in report["failed_cases"],
    "caught_mlops_as_ai_infra": "gpu_queue_called_mlops_bad" in report["failed_cases"],
    "caught_prompt_only_llmops": "llmops_prompt_only_bad" in report["failed_cases"],
    "caught_platform_as_devops": "platform_as_devops_rename_bad" in report["failed_cases"],
    "caught_rag_incident_misroute": "rag_incident_wrong_route_bad" in report["failed_cases"],
}

print("smoke=", smoke)
print("metrics=", report["metrics"])
print("route_error_rate=", report["route_error_rate"])
print("hard_blocker_count=", report["hard_blocker_count"])
print("misrouted_cases=", report["misrouted_cases"])
print("failed_cases=", report["failed_cases"])
print("failed_gates=", report["failed_gates"])
print("boundary_gate_pass=", report["boundary_gate_pass"])
```

这段 demo 里有 1 个完整样本和 16 个边界错误样本，所以每个维度的覆盖率是 `16/17=0.941`，低于 `0.95` 阈值。它想训练的不是固定答案，而是边界判断方法：先识别主责层，再列协作层，再说明交付物和门禁。面试时只要能把事故路由、artifact 血缘、trace / SLO、安全成本和开发者自助讲清楚，就不会停留在名词解释。

## 2.16 常见误区

误区一：MLOps 等于 AI Infra。

MLOps 更关注模型生命周期，AI Infra 更关注底层资源和平台能力。训练平台可能同时属于两者交集，但二者侧重点不同。

误区二：LLMOps 只是 prompt 管理。

Prompt 管理只是 LLMOps 的一部分。LLMOps 还包括 RAG、评估、Agent trace、工具调用、安全、模型路由和 token 成本。

误区三：Platform Engineering 是换个名字的 DevOps。

Platform Engineering 更强调把基础设施能力做成内部产品，提供自助化、标准模板和开发者体验。

误区四：AI Infra 团队只负责买卡和维护集群。

成熟 AI Infra 还要负责训练平台、推理平台、可观测性、容量规划、成本治理和安全治理。

误区五：边界一定要绝对清晰。

真实组织里边界会重叠。重要的是职责清楚、接口清楚、交付物清楚，而不是名字完全统一。

## 2.17 面试题

### 题 1：AI Infra 和 MLOps 的区别是什么？

答：AI Infra 更关注底层资源和平台能力，例如 GPU 集群、网络、存储、调度、训练平台、推理平台和可观测性。MLOps 更关注模型生命周期，例如数据版本、训练 pipeline、实验追踪、模型注册、评估、部署、监控和回滚。训练平台和模型仓库可能是两者交集。

### 题 2：LLMOps 相比 MLOps 多了什么？

答：LLMOps 增加了大模型应用特有能力，例如 prompt 管理、RAG 知识库、embedding 和向量索引、Agent trace、工具调用日志、大模型评测、内容安全、模型路由、provider 管理和 token 成本治理。

### 题 3：Platform Engineering 在 AI Infra 中的价值是什么？

答：它把复杂的底层能力封装成内部开发者可自助使用的平台，例如 Portal、CLI、SDK、训练模板、部署模板、日志面板和权限流程。这样算法和应用团队不用每次都找平台团队手工操作，可以更快、更标准、更安全地使用 AI Infra。

### 题 4：为什么说 AI Infra 不是简单的 DevOps？

答：AI 工作负载有特殊性：GPU 稀缺且昂贵，训练任务长时间运行，分布式训练需要多节点同时可用，checkpoint 和数据吞吐压力大，推理有 token 级延迟和 KV cache 管理，模型效果也需要评估。普通 Web 服务 DevOps 经验需要适配这些特点。

### 题 5：如果一个 RAG 应用回答质量下降，应该从哪些层排查？

答：可以从 LLMOps 排查 prompt、检索、rerank、上下文拼接和评测；从 Data Platform 排查文档更新、数据质量和索引版本；从 AI Infra 排查 embedding 服务、向量索引服务、推理延迟和错误率；从 Model Platform 排查模型版本是否变化。

## 2.18 小练习

练习一：给下面需求分类：GPU 任务排队太久、prompt 改动无回归测试、模型版本无法回滚、算法同学不会写 Kubernetes YAML、RAG 引用文档过期。

要求：分别判断主要属于 AI Infra、MLOps、LLMOps、Platform Engineering、Data Platform 或 Model Platform 的哪一类。

练习二：设计一个内部 AI 平台首页。

要求：从 Platform Engineering 角度列出用户最需要的入口，例如提交训练任务、部署模型服务、查看实验、查看评估、申请权限、查看成本。

练习三：画出一个大模型应用从开发到上线的生命周期。

要求：标出哪些步骤属于 MLOps，哪些属于 LLMOps，哪些依赖 AI Infra。

练习四：假设你是 AI Infra 负责人，如何和 Data Platform、Safety Platform、应用团队分工？

要求：写出每个团队的职责边界和接口。

## 2.19 本章小结

本章拆清了 AI Infra、MLOps、LLMOps、Platform Engineering 的边界。

你需要掌握：

1. AI Infra 更关注底层资源、平台运行时、可观测性、安全和成本治理。
2. MLOps 更关注机器学习模型生命周期，包括数据、训练、实验、模型注册、部署和回滚。
3. LLMOps 更关注大模型应用生命周期，包括 prompt、RAG、Agent、工具调用、评估、安全和 token 成本。
4. Platform Engineering 更关注把平台能力产品化，让内部团队自助、安全、标准化使用。
5. DevOps、SRE、Data Platform、Model Platform 都与 AI Infra 有交集，但关注点不同。
6. 面试时不要纠结名词，而要讲清楚每层解决什么问题、交付什么能力、和其他层如何协作。

下一章开始，我们进入硬件基础，讲 GPU、NPU、TPU 与 AI 加速器。
