# 第 60 章 AI Infra 未来趋势：Serverless GPU、异构算力、边缘推理和自治运维

这是第二十三册《AI Infra、大模型基础设施与平台工程》的最后一章。

前面我们从硬件、集群、训练平台、推理平台、数据模型实验平台、可观测性、成本安全治理，一直讲到系统设计题。本章不再展开某一个具体平台，而是讨论 AI Infra 的未来趋势。

先记住一句话：

> AI Infra 的未来，不是简单拥有更多 GPU，而是让算力更弹性、硬件更多元、推理更靠近用户、平台更自动化、治理更内建。

## 60.0 本讲资料边界与第二轮精修口径

本章讨论的是 AI Infra 的趋势判断方法，不是预测某个厂商、云平台或开源项目一定会采用哪条路线。第二轮精修时，本章按稳定抽象来写：

1. Serverless GPU 参考 Knative 式自动扩缩容、scale-to-zero 和按需服务的抽象，但大模型场景必须额外考虑权重加载、GPU warm pool、KV cache 状态和冷启动尾延迟。
2. 异构算力参考 Kubernetes Dynamic Resource Allocation 对设备资源抽象的方向，以及 GPU 分区、隔离和多实例化这类硬件能力；正文只写调度和适配原则，不绑定具体 CRD、驱动版本或云厂商接口。
3. 边缘推理只讨论低延迟、隐私、离线和成本约束下的云边路由，不把“端侧模型完全替代云端大模型”当成确定结论。
4. 自治运维参考 SRE 的 SLO、error budget、burn-rate 和分阶段自动化思想；自动修复必须保留审批、回滚、爆炸半径和审计边界。
5. 可观测性和 Agent trace 参考 OpenTelemetry signals 与 GenAI 语义约定的方向，但字段命名和生态实现仍会演进，因此本章强调 trace、metrics、logs、events 的契约，不把某个字段名写成唯一标准。

所以，本章的面试表达重点不是“背趋势名词”，而是能把趋势拆成可验证的指标、门禁和失败模式。

## 60.1 为什么要看趋势

AI Infra 变化非常快。

过去几年，大模型基础设施经历了多次变化：

1. 从单机训练到大规模分布式训练。
2. 从脚本训练到平台化训练。
3. 从单模型服务到多模型推理平台。
4. 从普通 API 到 RAG 和 Agent。
5. 从 GPU 堆资源到成本和可靠性治理。
6. 从实验驱动到评估和发布门禁。

未来还会继续变化。

面试中，趋势题常用于考察候选人是否有系统视野。

## 60.2 趋势一：Serverless GPU

Serverless GPU 的目标是让用户不关心底层 GPU 实例，只提交任务或请求，由平台自动分配、启动、伸缩和计费。

理想形态：

```text
用户提交推理或训练任务
  -> 平台自动选择 GPU 类型
  -> 自动加载模型或数据
  -> 自动扩缩容
  -> 按实际使用计费
```

这类似普通 serverless，但 GPU 场景更难。

## 60.3 Serverless GPU 的难点

难点包括：

1. 大模型冷启动慢。
2. 权重加载耗时长。
3. GPU 资源昂贵，不能无限预热。
4. 多模型切换成本高。
5. KV cache 和显存状态难迁移。
6. 训练任务运行时间长。
7. GPU 资源碎片明显。
8. 不同 GPU 型号兼容性不同。

因此 Serverless GPU 更可能先在推理、embedding、评估和短任务中成熟，再逐步扩展到更复杂训练场景。

## 60.4 Serverless 推理

Serverless 推理会关注：

1. 模型按需加载。
2. 权重缓存。
3. 快速冷启动。
4. warm pool。
5. 按 token 计费。
6. 多模型共享 GPU。
7. 自动扩缩容。
8. 空闲自动释放。

对长尾模型特别有价值。

高频模型可以常驻，低频模型按需加载。

这能降低多模型平台的空闲成本。

## 60.5 趋势二：异构算力

未来 AI Infra 不会只有一种 GPU。

异构算力包括：

1. NVIDIA GPU。
2. AMD GPU。
3. TPU。
4. NPU。
5. 国产 AI 加速器。
6. CPU 推理。
7. 边缘端芯片。
8. 专用推理芯片。

异构算力带来的核心问题是：平台如何屏蔽差异，同时利用不同硬件的成本和性能优势。

## 60.6 异构算力调度

异构调度要考虑：

1. 模型是否支持该硬件。
2. runtime 是否支持该硬件。
3. 精度和量化格式是否兼容。
4. 性能是否达标。
5. 成本是否更优。
6. 数据和模型是否能迁移。
7. 可观测性指标是否统一。

未来平台需要一个硬件抽象层，让用户声明目标：成本优先、延迟优先、吞吐优先或合规优先，由平台选择硬件。

## 60.7 Runtime 生态会继续分化

推理 runtime 会继续分化。

可能方向：

1. 通用 LLM serving runtime。
2. 极致性能 runtime。
3. 多模态 runtime。
4. Agent-oriented runtime。
5. 边缘推理 runtime。
6. 硬件厂商专用 runtime。

平台要避免被单一 runtime 锁死。

更好的方式是定义统一 inference interface，并在底层适配不同 runtime。

## 60.8 趋势三：边缘推理

边缘推理指模型在靠近用户或设备侧运行。

场景包括：

1. 手机端助手。
2. 车载智能系统。
3. 工业设备。
4. 本地隐私推理。
5. 低延迟交互。
6. 离线可用应用。

边缘推理的价值：

1. 降低云端成本。
2. 降低网络延迟。
3. 提升隐私保护。
4. 支持离线场景。

但边缘资源受限，需要小模型、量化、蒸馏和本地缓存。

## 60.9 云边协同

未来不是全部云端，也不是全部边缘，而是云边协同。

例如：

1. 简单任务在端侧小模型处理。
2. 复杂任务路由到云端大模型。
3. 敏感数据尽量本地处理。
4. 云端负责模型更新和策略下发。
5. 边缘端上传匿名反馈用于改进。

这会让模型路由从“多模型路由”扩展到“云边路由”。

## 60.10 趋势四：推理和训练更加一体化

未来训练平台和推理平台会更紧密联动。

原因：

1. 在线反馈会进入训练数据。
2. 推理 trace 会进入评估和回归集。
3. 线上问题样本会触发微调。
4. 模型发布会依赖训练血缘。
5. 成本和质量会共同驱动模型选择。

这意味着平台需要闭环：

```text
训练 -> 评估 -> 发布 -> 推理 -> 反馈 -> 数据 -> 再训练
```

AI Infra 会从单点平台走向闭环系统。

## 60.11 趋势五：更强的评估与质量治理

模型能力越强，评估越重要。

未来评估会更强调：

1. 线上真实任务评估。
2. 自动评估和人工评估结合。
3. 多维安全评估。
4. Agent 行为评估。
5. 工具调用评估。
6. 长期记忆评估。
7. 成本质量联合评估。
8. 持续回归测试。

模型发布不会只看 benchmark，而会看质量、安全、成本、SLO 和业务结果。

## 60.12 趋势六：Agent Infra 成为新基础设施

Agent 越来越多后，会需要专门的 Agent Infra。

它包括：

1. Agent runtime。
2. Tool registry。
3. Permission system。
4. Memory store。
5. Execution trace。
6. Human-in-the-loop。
7. Agent evaluation。
8. Cost control。
9. Safety guardrail。

Agent Infra 的难点是多步状态、工具副作用、安全和可回放。

未来很多 LLMOps 平台会向 AgentOps 演进。

## 60.13 趋势七：自治运维

自治运维是让平台自动发现、诊断和修复问题。

可能能力：

1. 自动检测 GPU 异常。
2. 自动摘除不健康 endpoint。
3. 自动回滚异常模型发布。
4. 自动分析 p99 升高原因。
5. 自动调整扩缩容参数。
6. 自动生成事故复盘草稿。
7. 自动推荐成本优化方案。
8. 自动识别数据质量异常。

自治运维不会一开始完全取代人，而是先成为值班工程师的辅助系统。

## 60.14 AIOps 在 AI Infra 中的特殊性

AIOps 在 AI Infra 中更复杂，因为指标更多、链路更长。

它需要理解：

1. GPU。
2. 分布式训练。
3. KV cache。
4. 数据质量。
5. 模型版本。
6. prompt 版本。
7. RAG trace。
8. Agent tool trace。
9. 成本。

普通微服务 AIOps 不能直接迁移。

AI Infra 的自治运维必须理解 AI 任务语义。

## 60.15 趋势八：成本成为一等公民

未来 AI 平台设计会更早考虑成本。

不是上线后再优化，而是在设计阶段就考虑：

1. cost per token。
2. cost per request。
3. cost per agent run。
4. cost per eval sample。
5. tokens per GPU hour。
6. cache saved cost。
7. failed job cost。
8. warm pool idle cost。

模型质量不再是唯一目标。

同等质量下，成本更低的平台会更有竞争力。

## 60.16 趋势九：安全和合规内建

未来安全不会是外置审核，而会内建在平台流程中。

例如：

1. 数据接入自动分级。
2. 训练任务自动校验数据权限。
3. 模型继承训练数据敏感等级。
4. 推理请求自动脱敏。
5. RAG 检索自动权限过滤。
6. 工具调用自动风险分级。
7. 发布自动检查安全门禁。
8. 审计自动记录。

安全治理会从“人工流程”变成“平台默认能力”。

## 60.17 趋势十：标准化和互操作

AI Infra 生态会逐步标准化。

可能标准化对象：

1. 模型服务 API。
2. Tool schema。
3. Agent trace。
4. Eval report。
5. Model metadata。
6. Dataset manifest。
7. Observability signals。
8. Deployment manifest。

标准化能降低平台之间迁移成本。

但短期内，不同厂商和开源生态仍会有差异。

平台设计要保留适配层。

## 60.18 AI Infra 工程师未来能力模型

未来 AI Infra 工程师需要同时具备：

1. 分布式系统能力。
2. GPU 和硬件理解。
3. Kubernetes 和调度能力。
4. 训练和推理机制理解。
5. 数据平台能力。
6. 可观测性能力。
7. 安全和成本治理能力。
8. 产品化平台思维。
9. 模型评估理解。
10. 跨团队协作能力。

单纯会部署模型服务已经不够。

## 60.19 未来趋势判断指标和最小 demo

趋势题最容易回答得很虚。更稳的方式是把每个趋势都转成“指标 + 门禁 + 失败模式”。

可以把一个未来 AI Infra 方案样本记为：

```math
F_i=(s_i,h_i,e_i,l_i,a_i,o_i,k_i,r_i,m_i,z_i)
```

其中 `s` 表示 Serverless GPU 能力，`h` 表示异构算力适配，`e` 表示边缘推理，`l` 表示训练、评估、发布、推理和反馈闭环，`a` 表示 Agent Infra，`o` 表示自治运维，`k` 表示成本治理，`r` 表示安全合规，`m` 表示互操作和标准化，`z` 表示最终上线门禁。

Serverless GPU 最关键的是冷启动和 warm pool 成本：

```math
T_{\mathrm{cold}}=T_{\mathrm{schedule}}+T_{\mathrm{image}}+T_{\mathrm{weight}}+T_{\mathrm{runtime}}+T_{\mathrm{warmup}}
```

```math
K_{\mathrm{warm}}=N_{\mathrm{warm}}K_{\mathrm{gpu}}T_{\mathrm{idle}}
```

其中 `T_cold` 是一次冷启动总耗时，`T_schedule` 是调度时间，`T_image` 是镜像准备时间，`T_weight` 是模型权重加载时间，`T_runtime` 是 runtime 初始化时间，`T_warmup` 是模型预热时间；`K_warm` 是 warm pool 空闲成本，`N_warm` 是常驻 GPU 数量，`K_gpu` 是单 GPU 单位时间成本，`T_idle` 是空闲时长。

异构算力不是简单“支持更多硬件”，而是要做可解释选择。一个简化打分可以写成：

```math
S_h=w_qQ_h-w_lL_h-w_cC_h+w_pP_h+w_mM_h
```

其中 `Q_h` 是质量保持程度，`L_h` 是延迟归一化惩罚，`C_h` 是成本归一化惩罚，`P_h` 是可迁移性，`M_h` 是成熟度；权重 `w_q,w_l,w_c,w_p,w_m` 由业务目标决定。

边缘推理要看本地命中和云端回退：

```math
R_{\mathrm{local}}=\frac{N_{\mathrm{edgeok}}}{N_{\mathrm{total}}},\qquad
R_{\mathrm{fallback}}=\frac{N_{\mathrm{cloud}}}{N_{\mathrm{total}}}
```

其中 `N_edgeok` 是端侧可直接完成的请求数，`N_cloud` 是回退到云端的请求数，`N_total` 是总请求数。隐私敏感任务更看重 `R_local`，复杂推理任务更看重可靠回退。

自治运维必须有安全门：

```math
G_{\mathrm{auto}}=\mathbf{1}\left[
A_{\mathrm{detect}}\land A_{\mathrm{diagnose}}\land A_{\mathrm{rollback}}\land H_{\mathrm{approve}}\land B_{\mathrm{blast}}\le \rho_{\mathrm{blast}}
\right]
```

其中 `A_detect`、`A_diagnose` 和 `A_rollback` 分别表示检测、诊断和回滚能力，`H_approve` 表示高风险动作需要人工审批，`B_blast` 是自动动作影响范围，`\rho_blast` 是允许的最大爆炸半径。

最终趋势判断门禁可以写成：

```math
G_{\mathrm{future}}=\mathbf{1}\left[
G_{\mathrm{serverless}}\land
G_{\mathrm{hetero}}\land
G_{\mathrm{edge}}\land
G_{\mathrm{loop}}\land
G_{\mathrm{agent}}\land
G_{\mathrm{auto}}\land
G_{\mathrm{cost}}\land
G_{\mathrm{security}}\land
G_{\mathrm{interop}}
\right]
```

下面这个 0 依赖 demo 演示如何把趋势题做成可运行审计。它不是生产系统，而是帮助你在面试中把“未来趋势”讲成“能验收的工程能力”。

```python
from copy import deepcopy


class MiniAIInfraFutureTrendAudit:
    def __init__(self):
        self.gates = [
            "serverless_gpu_readiness",
            "heterogeneous_compute_fit",
            "edge_inference_fit",
            "closed_loop_learning",
            "agent_infra_governance",
            "autonomous_ops_safety",
            "cost_first_design",
            "security_builtin",
            "interop_standardization",
        ]
        self.actions = {
            "serverless_gpu_readiness": [
                "separate model weight cache from request path",
                "size warm pool by cold-start SLO and idle cost budget",
                "record scale-to-zero and runtime warmup behavior",
            ],
            "heterogeneous_compute_fit": [
                "build hardware-runtime-precision compatibility matrix",
                "rank devices by quality, latency, cost, portability, and maturity",
            ],
            "edge_inference_fit": [
                "route privacy-sensitive and low-latency tasks to local model first",
                "keep cloud fallback for hard tasks and model update flow",
            ],
            "closed_loop_learning": [
                "link inference traces to eval sets, release gates, and feedback datasets",
                "require human review before feedback enters training data",
            ],
            "agent_infra_governance": [
                "add tool registry, permission model, replayable traces, and agent evals",
                "enforce cost budget per agent run",
            ],
            "autonomous_ops_safety": [
                "separate detection, diagnosis, and repair permissions",
                "cap blast radius and require approval for high-risk actions",
            ],
            "cost_first_design": [
                "define cost per 1k tokens and warm-pool idle budget before launch",
                "attribute cost by model, tenant, workload, and environment",
            ],
            "security_builtin": [
                "make data classification, redaction, audit, and policy gate mandatory",
                "fail closed when permissions or secrets are missing",
            ],
            "interop_standardization": [
                "standardize model, dataset, deployment, trace, and eval manifests",
                "keep adapter layer between platform API and runtime implementations",
            ],
        }

    def cold_start_ms(self, case):
        s = case["serverless"]
        return (
            s["schedule_ms"]
            + s["image_ms"]
            + s["weight_ms"]
            + s["runtime_ms"]
            + s["warmup_ms"]
        )

    def warm_pool_cost(self, case):
        s = case["serverless"]
        return round(s["warm_gpu_count"] * s["gpu_cost_per_hour"] * s["idle_hours"], 2)

    def best_hardware(self, case):
        h = case["heterogeneous"]
        scores = []
        for candidate in h["candidates"]:
            latency_penalty = candidate["latency_ms"] / h["max_latency_ms"]
            cost_penalty = candidate["cost_per_1k"] / h["max_cost_per_1k"]
            score = (
                0.45 * candidate["quality"]
                - 0.20 * latency_penalty
                - 0.20 * cost_penalty
                + 0.10 * candidate["portability"]
                + 0.05 * candidate["maturity"]
            )
            scores.append((round(score, 3), candidate))
        return max(scores, key=lambda item: item[0])

    def edge_rates(self, case):
        e = case["edge"]
        return {
            "local_hit_rate": round(e["edge_ok"] / e["total_requests"], 3),
            "fallback_rate": round(e["cloud_fallback"] / e["total_requests"], 3),
        }

    def gate_serverless_gpu_readiness(self, case):
        s = case["serverless"]
        return (
            self.cold_start_ms(case) <= s["cold_start_budget_ms"]
            and s["has_weight_cache"]
            and s["has_scale_to_zero"]
            and s["runtime_warmup_profiled"]
            and self.warm_pool_cost(case) <= s["warm_pool_budget"]
        )

    def gate_heterogeneous_compute_fit(self, case):
        h = case["heterogeneous"]
        _, best = self.best_hardware(case)
        return (
            h["has_resource_abstraction"]
            and h["has_runtime_matrix"]
            and h["has_metric_parity"]
            and best["runtime_supported"]
            and best["quality"] >= h["min_quality"]
            and best["latency_ms"] <= h["max_latency_ms"]
            and best["cost_per_1k"] <= h["max_cost_per_1k"]
        )

    def gate_edge_inference_fit(self, case):
        e = case["edge"]
        rates = self.edge_rates(case)
        return (
            rates["local_hit_rate"] >= e["min_local_hit_rate"]
            and rates["fallback_rate"] <= e["max_fallback_rate"]
            and e["has_cloud_fallback"]
            and e["privacy_route_enabled"]
            and e["thermal_budget_checked"]
        )

    def gate_closed_loop_learning(self, case):
        l = case["loop"]
        return (
            l["trace_to_eval"]
            and l["eval_to_release_gate"]
            and l["feedback_dataset"]
            and l["human_review"]
            and l["rollback_policy"]
            and l["online_feedback_rate"] >= l["min_feedback_rate"]
        )

    def gate_agent_infra_governance(self, case):
        a = case["agent"]
        return (
            a["tool_registry"]
            and a["permission_model"]
            and a["trace_replay"]
            and a["agent_eval_suite"]
            and a["cost_budget_per_run"]
            and a["side_effect_policy"]
        )

    def gate_autonomous_ops_safety(self, case):
        o = case["ops"]
        return (
            o["detect"]
            and o["diagnose"]
            and o["rollback"]
            and o["human_approval"]
            and o["audit_log"]
            and o["blast_radius"] <= o["max_blast_radius"]
        )

    def gate_cost_first_design(self, case):
        k = case["cost"]
        return (
            k["cost_per_1k_tokens"] <= k["budget_per_1k_tokens"]
            and k["attribution_enabled"]
            and k["cache_policy"]
            and k["unit_cost_slo"]
            and self.warm_pool_cost(case) <= k["warm_pool_idle_budget"]
        )

    def gate_security_builtin(self, case):
        r = case["security"]
        return (
            r["data_classification"]
            and r["secret_management"]
            and r["pii_redaction"]
            and r["permission_filter"]
            and r["audit_log"]
            and r["release_policy_gate"]
        )

    def gate_interop_standardization(self, case):
        m = case["interop"]
        return (
            m["model_manifest"]
            and m["dataset_manifest"]
            and m["deployment_manifest"]
            and m["trace_schema"]
            and m["eval_report_schema"]
            and m["adapter_layer"]
            and m["api_versioning"]
        )

    def audit_case(self, case):
        failed = []
        for gate in self.gates:
            method = getattr(self, "gate_" + gate)
            if not method(case):
                failed.append(gate)
        return failed

    def audit(self, cases):
        results = {}
        failed_cases = {}
        hard_blockers = []
        for case in cases:
            failed = self.audit_case(case)
            results[case["name"]] = failed
            if failed:
                failed_cases[case["name"]] = failed
                hard_blockers.extend((case["name"], gate) for gate in failed)

        metrics = {}
        total = len(cases)
        for gate in self.gates:
            pass_count = sum(1 for failed in results.values() if gate not in failed)
            metrics[gate] = round(pass_count / total, 3)

        failed_gate_names = sorted({gate for failed in results.values() for gate in failed})
        remediation_sample = {
            name: [self.actions[gate][0] for gate in gates]
            for name, gates in list(failed_cases.items())[:3]
        }
        return {
            "metrics": metrics,
            "failed_cases": failed_cases,
            "failed_gate_names": failed_gate_names,
            "hard_blockers": hard_blockers,
            "remediation_sample": remediation_sample,
            "future_trend_gate_pass": not failed_cases,
        }


def build_cases():
    complete = {
        "name": "complete_future_platform",
        "serverless": {
            "schedule_ms": 1200,
            "image_ms": 2200,
            "weight_ms": 18000,
            "runtime_ms": 2800,
            "warmup_ms": 1800,
            "cold_start_budget_ms": 30000,
            "has_weight_cache": True,
            "has_scale_to_zero": True,
            "runtime_warmup_profiled": True,
            "warm_gpu_count": 4,
            "gpu_cost_per_hour": 3.2,
            "idle_hours": 2.5,
            "warm_pool_budget": 40.0,
        },
        "heterogeneous": {
            "has_resource_abstraction": True,
            "has_runtime_matrix": True,
            "has_metric_parity": True,
            "min_quality": 0.92,
            "max_latency_ms": 650,
            "max_cost_per_1k": 0.035,
            "candidates": [
                {
                    "name": "gpu_a",
                    "quality": 0.96,
                    "latency_ms": 420,
                    "cost_per_1k": 0.030,
                    "portability": 0.90,
                    "maturity": 0.95,
                    "runtime_supported": True,
                },
                {
                    "name": "edge_npu",
                    "quality": 0.93,
                    "latency_ms": 610,
                    "cost_per_1k": 0.018,
                    "portability": 0.70,
                    "maturity": 0.75,
                    "runtime_supported": True,
                },
            ],
        },
        "edge": {
            "total_requests": 1000,
            "edge_ok": 760,
            "cloud_fallback": 190,
            "min_local_hit_rate": 0.70,
            "max_fallback_rate": 0.25,
            "has_cloud_fallback": True,
            "privacy_route_enabled": True,
            "thermal_budget_checked": True,
        },
        "loop": {
            "trace_to_eval": True,
            "eval_to_release_gate": True,
            "feedback_dataset": True,
            "human_review": True,
            "rollback_policy": True,
            "online_feedback_rate": 0.18,
            "min_feedback_rate": 0.10,
        },
        "agent": {
            "tool_registry": True,
            "permission_model": True,
            "trace_replay": True,
            "agent_eval_suite": True,
            "cost_budget_per_run": True,
            "side_effect_policy": True,
        },
        "ops": {
            "detect": True,
            "diagnose": True,
            "rollback": True,
            "human_approval": True,
            "audit_log": True,
            "blast_radius": 0.05,
            "max_blast_radius": 0.10,
        },
        "cost": {
            "cost_per_1k_tokens": 0.021,
            "budget_per_1k_tokens": 0.030,
            "attribution_enabled": True,
            "cache_policy": True,
            "unit_cost_slo": True,
            "warm_pool_idle_budget": 40.0,
        },
        "security": {
            "data_classification": True,
            "secret_management": True,
            "pii_redaction": True,
            "permission_filter": True,
            "audit_log": True,
            "release_policy_gate": True,
        },
        "interop": {
            "model_manifest": True,
            "dataset_manifest": True,
            "deployment_manifest": True,
            "trace_schema": True,
            "eval_report_schema": True,
            "adapter_layer": True,
            "api_versioning": True,
        },
    }

    cases = [complete]

    bad = deepcopy(complete)
    bad["name"] = "serverless_gpu_cold_start_bad"
    bad["serverless"]["weight_ms"] = 45000
    bad["serverless"]["has_weight_cache"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "heterogeneous_compute_runtime_gap_bad"
    bad["heterogeneous"]["has_runtime_matrix"] = False
    bad["heterogeneous"]["candidates"][0]["runtime_supported"] = False
    bad["heterogeneous"]["candidates"][1]["quality"] = 0.86
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "edge_without_cloud_fallback_bad"
    bad["edge"]["edge_ok"] = 520
    bad["edge"]["cloud_fallback"] = 410
    bad["edge"]["has_cloud_fallback"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "closed_loop_feedback_untrusted_bad"
    bad["loop"]["trace_to_eval"] = False
    bad["loop"]["human_review"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "agent_infra_permission_gap_bad"
    bad["agent"]["permission_model"] = False
    bad["agent"]["trace_replay"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "autonomous_ops_unbounded_action_bad"
    bad["ops"]["human_approval"] = False
    bad["ops"]["blast_radius"] = 0.30
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "cost_first_design_missing_bad"
    bad["cost"]["cost_per_1k_tokens"] = 0.061
    bad["cost"]["attribution_enabled"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "security_builtin_missing_bad"
    bad["security"]["pii_redaction"] = False
    bad["security"]["release_policy_gate"] = False
    cases.append(bad)

    bad = deepcopy(complete)
    bad["name"] = "interop_standardization_missing_bad"
    bad["interop"]["trace_schema"] = False
    bad["interop"]["adapter_layer"] = False
    cases.append(bad)

    return cases


audit = MiniAIInfraFutureTrendAudit()
cases = build_cases()
example_case = cases[0]
best_score, best_hardware = audit.best_hardware(example_case)
future_examples = {
    "cold_start_ms": audit.cold_start_ms(example_case),
    "warm_pool_cost": audit.warm_pool_cost(example_case),
    "best_hardware": best_hardware["name"],
    "best_hardware_score": best_score,
    **audit.edge_rates(example_case),
}
report = audit.audit(cases)

print("future_examples=", future_examples)
print("metrics=", report["metrics"])
print("hard_blocker_count=", len(report["hard_blockers"]))
print("failed_case_count=", len(report["failed_cases"]))
print("failed_gate_count=", len(report["failed_gate_names"]))
print("remediation_sample=", report["remediation_sample"])
print("future_trend_gate_pass=", report["future_trend_gate_pass"])
```

这段代码输出的核心含义是：

1. `future_examples` 把趋势题变成冷启动、warm pool 成本、异构硬件选择和云边路由指标。
2. `metrics` 说明每类未来能力在 toy case 中的覆盖率。
3. `hard_blocker_count`、`failed_case_count` 和 `failed_gate_count` 能帮助你定位“趋势判断”里最不成熟的模块。
4. `remediation_sample` 给出每类失败的第一步修复动作。
5. `future_trend_gate_pass=False` 说明未来趋势不是口号，只要 Serverless、异构、边缘、闭环、Agent、自治运维、成本、安全和互操作任一关键门禁缺失，就不能说平台已经具备面向未来的完整能力。

## 60.20 面试开放题回答思路

如果面试官问“AI Infra 未来会怎么发展”，可以这样回答：

```text
我认为 AI Infra 会朝几个方向发展。第一是更弹性的 Serverless GPU 和按需推理，降低长尾模型的空闲成本。第二是异构算力和硬件抽象，让平台能在不同 GPU、NPU、TPU 和边缘芯片之间做成本和性能选择。第三是云边协同，让简单和隐私敏感任务在边缘处理，复杂任务上云。第四是训练、评估、发布、推理和反馈形成闭环。第五是 Agent Infra、trace、工具权限和安全治理成为平台核心能力。最后是自治运维和成本治理内建，让平台能自动发现问题、优化资源和控制风险。
```

这个回答覆盖弹性、硬件、边缘、闭环、Agent、安全、成本和运维。

## 60.21 常见误区

误区一：未来只是更多 GPU。

更多 GPU 重要，但弹性、调度、成本、数据、推理、评估和治理同样关键。

误区二：Serverless GPU 会马上解决所有问题。

大模型冷启动、权重加载、显存状态和成本让 Serverless GPU 很难一蹴而就。

误区三：边缘推理会取代云端推理。

更可能是云边协同，不同任务放在不同位置。

误区四：自治运维等于完全无人值守。

短期更现实的是辅助诊断、自动告警、自动回滚和推荐修复。

误区五：标准化会消除平台差异。

标准化会降低迁移成本，但性能、成本、治理和体验仍会形成差异。

## 60.22 给读者的最后建议

学 AI Infra，不要只学工具。

更重要的是理解：

1. 为什么大模型训练需要特殊调度。
2. 为什么推理平台要看 token 而不是只看 QPS。
3. 为什么 KV cache 是核心瓶颈。
4. 为什么数据版本和血缘影响模型质量。
5. 为什么评估和发布门禁是生产必需。
6. 为什么 trace 对 RAG/Agent 很重要。
7. 为什么安全和成本必须内建。
8. 为什么平台工程的最终目标是提升研发效率和生产可靠性。

如果你能把这些问题讲清楚，就已经具备 AI Infra 岗位的核心思维。

## 60.23 第二十三册总结

本册从 AI Infra 总览讲起，覆盖了：

1. GPU、显存、网络、存储和集群基础。
2. 训练平台工程。
3. 推理平台工程。
4. 数据、模型与实验平台。
5. 可观测性、可靠性与成本治理。
6. 安全、审计和变更管理。
7. 多个 AI Infra 系统设计题。

你现在应该能系统回答：

1. 如何设计训练平台？
2. 如何设计推理平台？
3. 如何设计 GPU 调度系统？
4. 如何设计评估平台？
5. 如何设计模型仓库？
6. 如何设计可观测性平台？
7. 如何做成本、安全和可靠性治理？

这就是 AI Infra 面试和实战中的主干能力。

## 60.24 本章小结

本章讲了 AI Infra 的未来趋势。

你需要记住：

1. Serverless GPU 会提升弹性，但受冷启动、权重加载和显存状态限制。
2. 异构算力会成为常态，平台需要硬件抽象和 runtime 适配能力。
3. 边缘推理和云边协同会在隐私、低延迟和成本场景中变重要。
4. 训练、评估、发布、推理和反馈会形成闭环。
5. Agent Infra、自治运维、成本治理、安全合规和标准化会成为未来平台竞争重点。
6. AI Infra 工程师需要同时懂系统、硬件、模型、数据、平台、安全和成本。

第二十三册到这里正文第一版完成。
