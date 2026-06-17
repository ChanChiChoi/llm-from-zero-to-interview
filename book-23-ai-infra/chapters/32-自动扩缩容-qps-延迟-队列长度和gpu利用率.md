# 第 32 章 自动扩缩容：QPS、延迟、队列长度和 GPU 利用率

上一章讲了缓存体系。本章讲推理平台的自动扩缩容。

自动扩缩容看起来像一个很成熟的问题：流量高了扩容，流量低了缩容。但大模型推理和普通 Web 服务不同，不能简单照搬 CPU 利用率或 QPS 驱动的 HPA。

先记住一句话：

> 大模型推理扩缩容的核心不是“实例数够不够”，而是“在冷启动、显存、队列、token 负载和 SLO 约束下，容量是否及时、稳定、低成本地匹配需求”。

## 32.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 Kubernetes Horizontal Pod Autoscaling 对按指标比例计算期望副本数的公开算法，KEDA 对事件驱动和队列型指标扩缩容的工程抽象，Triton Inference Server 对 queue、compute、request、batch、GPU 等推理指标的公开口径，KServe / Knative 对并发、请求量和冷启动相关的 serverless 推理服务口径，以及 Google SRE 关于过载保护、容量、错误预算和控制回路稳定性的系统设计原则。

这些资料共同指向一个稳定事实：普通 HPA 的比例缩放思想有价值，但 LLM serving 不能只用 CPU、QPS 或 GPU utilization 做单指标扩缩容。推理 autoscaler 必须同时看 input / output tokens、TTFT、TPOT、queue wait、KV pressure、active sequences、error / timeout、冷启动、warm pool、draining、路由降级、租户配额和成本预算。

本章只抽象截至 2026-06 仍稳定的 LLM-aware autoscaling 设计口径，不把某个云厂商、Kubernetes 插件、serverless 平台、GPU 型号、runtime 指标字段名或内部平台阈值写成通用标准。正文公式用于面试表达、容量估算和策略审计；真实上线仍要用目标模型、runtime、硬件、流量分布、冷启动时间、warm pool 策略和 SLO 实测校准。

## 32.1 为什么推理服务需要自动扩缩容

推理流量通常有明显波动：

1. 白天和夜晚流量不同。
2. 工作日和周末不同。
3. 活动期间突增。
4. 某个产品功能上线后突增。
5. Agent 批量任务触发突增。
6. 某个租户异常调用导致突增。

如果容量固定，会出现两个问题：

1. 容量不足时延迟升高、排队增加、超时变多。
2. 容量过剩时 GPU 空闲、成本浪费。

自动扩缩容的目标是在服务质量和成本之间平衡。

## 32.2 普通 HPA 为什么不够

Kubernetes HPA 常见依据是 CPU、内存或自定义指标。

普通 Web 服务里，CPU 利用率常常能反映负载。

但 LLM 推理不一样：

1. GPU 利用率不等于用户体验。
2. QPS 不等于 token 负载。
3. Decode 可能显存带宽受限，而不是算力受限。
4. KV cache 接近满时，GPU 利用率可能还不高。
5. 冷启动时间很长，扩容不是立即生效。
6. 长请求和短请求对资源消耗差异巨大。

所以推理平台需要更贴近 LLM 的扩缩容指标。

## 32.3 扩缩容目标

推理服务扩缩容通常要同时满足：

1. TTFT SLO。
2. TPOT SLO。
3. p95 / p99 延迟目标。
4. 错误率和超时率目标。
5. 队列等待时间目标。
6. 成本预算。
7. GPU 利用率目标。
8. 多租户配额目标。

这些目标会冲突。

例如，更激进扩容可以降低延迟，但成本上升。更保守扩容可以省钱，但容易排队和超时。

## 32.4 QPS 指标的局限

QPS 是最常见的流量指标，但对大模型推理不够。

两个 endpoint 都是 50 QPS：

1. Endpoint A 每个请求输入 100 token，输出 50 token。
2. Endpoint B 每个请求输入 8000 token，输出 1000 token。

它们的 QPS 一样，但 B 的资源消耗可能高出几十倍。

所以 QPS 只能作为辅助指标。

更关键的是：

1. input tokens/s。
2. output tokens/s。
3. active sequences。
4. total context tokens。
5. KV cache usage。
6. queue wait time。

## 32.5 延迟指标

延迟指标直接反映用户体验。

常见指标：

1. TTFT。
2. TPOT。
3. end-to-end latency。
4. p50 latency。
5. p95 latency。
6. p99 latency。

扩容触发可以基于：

1. p95 TTFT 超过阈值。
2. p99 延迟持续升高。
3. TPOT 明显退化。
4. 超时率升高。

但延迟是滞后指标。等 p99 已经升高再扩容，可能已经影响用户。

所以需要结合队列和负载指标提前判断。

## 32.6 队列长度和等待时间

队列指标是推理扩缩容非常重要的信号。

常见指标：

1. waiting queue length。
2. queue wait time。
3. admission reject count。
4. active requests。
5. pending prefill requests。
6. pending decode requests。

队列增长通常说明当前处理能力跟不上到达速度。

相比延迟，队列是更早的拥塞信号。

例如：

```text
queue length 持续增长 -> TTFT 开始变差 -> timeout rate 上升
```

扩容系统应该在队列持续增长阶段就行动，而不是等超时爆发。

## 32.7 GPU 利用率的误区

GPU 利用率常被用于扩缩容，但它不是万能指标。

问题包括：

1. GPU 利用率高不一定坏，可能只是吞吐高。
2. GPU 利用率低不一定轻松，可能卡在显存带宽或 KV cache。
3. GPU 利用率不能反映队列等待。
4. GPU 利用率不能反映 p99 延迟。
5. 不同 runtime 指标口径可能不同。

更合理的方式是把 GPU 利用率和其他指标一起看：

1. GPU 利用率。
2. GPU 显存占用。
3. KV cache 使用率。
4. tokens/s。
5. queue wait time。
6. TTFT / TPOT。

## 32.8 KV Cache 水位

KV cache 是 LLM 推理扩缩容中特别重要的资源。

如果 KV cache 水位接近上限，系统可能出现：

1. 新请求无法进入。
2. 长上下文请求被拒绝。
3. cache 驱逐增加。
4. OOM 风险上升。
5. p99 延迟抖动。

因此扩容信号应包括：

1. KV cache usage。
2. free KV blocks。
3. eviction rate。
4. OOM count。
5. max active sequences。

对于长上下文业务，KV cache 水位往往比 GPU 利用率更能反映容量压力。

## 32.9 Token 吞吐指标

Token 吞吐是推理服务容量的重要描述。

常见指标：

1. input tokens/s。
2. output tokens/s。
3. total tokens/s。
4. per-GPU tokens/s。
5. per-model tokens/s。
6. per-tenant tokens/s。

扩容可以基于 tokens/s 接近实例可承载上限。

例如某模型在指定 SLO 下每张 GPU 可稳定处理 3000 output tokens/s，如果当前已经长期接近 2800，并且队列增长，就应该扩容。

注意：最大 tokens/s 不是固定值，会受输入长度、输出长度、batch、上下文长度和 sampling 参数影响。

## 32.10 冷启动问题

大模型扩容的最大难点之一是冷启动慢。

冷启动包括：

1. 调度 Pod。
2. 拉取镜像。
3. 挂载权重。
4. 加载模型。
5. 初始化 CUDA。
6. 分配显存。
7. 预热 kernel。
8. 注册 endpoint。

这个过程可能从几十秒到数分钟。

所以扩容决策必须提前。

如果等流量打满再扩，新的实例可能来不及承接流量。

## 32.11 热池和预热

为了解决冷启动，平台常用 warm pool。

Warm pool 是预先启动好的一组可用实例或半可用实例。

常见方式：

1. 常用模型常驻。
2. 低流量时保留最小副本。
3. 预拉取镜像。
4. 本地缓存权重。
5. 提前加载模型但不接流量。
6. 定时预热 kernel。

Warm pool 能降低扩容响应时间，但会增加空闲成本。

这是典型的成本和可用性权衡。

## 32.12 扩容策略

常见扩容策略：

1. 阈值扩容。
2. 预测扩容。
3. 队列驱动扩容。
4. SLO 驱动扩容。
5. 定时扩容。
6. 手动兜底扩容。

阈值扩容简单，例如 queue wait time 超过 500ms 持续 3 分钟就扩容。

预测扩容根据历史流量提前扩，例如每天上午 9 点流量上涨前先扩容。

SLO 驱动扩容更关注用户体验，但要避免指标滞后。

实际系统通常组合使用。

## 32.13 缩容策略

缩容比扩容更容易出问题。

如果缩得太快，会导致：

1. 刚释放资源又要扩容。
2. 连接被中断。
3. streaming 请求被影响。
4. KV cache 丢失。
5. 热缓存失效。
6. 延迟抖动。

缩容要考虑：

1. 当前是否有 active requests。
2. 是否有长 streaming 请求。
3. 实例是否处于 draining 状态。
4. 是否是 warm pool 实例。
5. 最近流量是否稳定下降。
6. 是否处于业务高峰前。

缩容通常要比扩容更保守。

## 32.14 Draining 机制

缩容前应先让实例进入 draining 状态。

Draining 表示：

1. 不再接收新请求。
2. 继续处理已有请求。
3. 等 active requests 结束。
4. 超过最长等待时间后强制终止或迁移。

没有 draining，缩容可能直接中断用户请求。

对 streaming 生成尤其要小心，因为连接可能持续很久。

## 32.15 多模型扩缩容

多模型平台里，扩缩容更复杂。

问题包括：

1. 每个模型的流量不同。
2. 每个模型的冷启动时间不同。
3. 每个模型需要的 GPU 类型不同。
4. 小模型和大模型的资源利用率不同。
5. 模型之间能否共享节点或实例。
6. 路由策略会影响流量分布。

平台要决定：

1. 给哪个模型扩容？
2. 扩多少副本？
3. 用什么 GPU？
4. 是否从其他模型释放资源？
5. 是否通过路由降级缓解压力？

扩缩容不能和模型路由割裂。

## 32.16 多租户扩缩容

多租户场景还要考虑配额和优先级。

例如：

1. 高优先级租户 SLO 要优先保证。
2. 低优先级批量任务可以延迟。
3. 某个租户突增不能拖垮全平台。
4. 扩容成本要能归因到租户或业务。

平台可以按：

1. 租户级队列。
2. 租户级 token/s。
3. 租户级 p95 延迟。
4. 租户级预算。
5. 租户级优先级。

做扩缩容和限流决策。

## 32.17 扩缩容和路由联动

扩缩容和路由应该联动。

当某个模型过载时，可以：

1. 扩容该模型。
2. 将部分流量路由到其他 endpoint。
3. 将低优先级流量降级到小模型。
4. 开启更严格限流。
5. 使用缓存或结果复用。

如果只扩容不改路由，可能来不及。

如果只改路由不扩容，可能把压力转移到另一个模型。

成熟平台会把 autoscaler、router、admission controller 和 cost controller 放在同一个控制闭环里。

## 32.18 扩缩容控制闭环

一个完整控制闭环：

```text
Metrics -> Autoscaler -> Scheduler -> Deployment -> Router -> Runtime -> Metrics
```

步骤如下：

1. Metrics 收集流量、延迟、队列、GPU 和 KV cache 指标。
2. Autoscaler 判断是否扩缩容。
3. Scheduler 分配 GPU 节点。
4. Deployment 启动或停止 runtime 实例。
5. Router 更新可用 endpoint。
6. Runtime 承接流量并继续上报指标。

这个闭环要避免振荡。

## 32.19 如何避免扩缩容振荡

振荡指系统频繁扩容、缩容、再扩容。

常见原因：

1. 指标窗口太短。
2. 阈值太敏感。
3. 冷启动未纳入考虑。
4. 缩容太激进。
5. 流量本身周期性波动。

缓解方法：

1. 使用滑动窗口。
2. 设置 cooldown 时间。
3. 扩容快、缩容慢。
4. 最小副本和最大副本限制。
5. 预测扩容。
6. 缩容前 draining。
7. 对异常指标做平滑。

扩缩容系统本身也是一个控制系统。

## 32.20 自动扩缩容审计指标与最小 demo

第二轮精修时，需要把自动扩缩容从“看哪个指标扩副本”升级成可解释的控制闭环。

可以把一个观测窗口写成：

```math
a_i=(\lambda_i,x_i,y_i,q_i,w_i,t_i,p_i,g_i,k_i,c_i,h_i,u_i,z_i)
```

其中，`\lambda_i` 是请求到达率，`x_i` 是 input tokens/s，`y_i` 是 output tokens/s，`q_i` 是队列长度，`w_i` 是 queue wait，`t_i` 是 TTFT / TPOT / E2E 延迟，`p_i` 是 p95 / p99 和 timeout 等尾部风险，`g_i` 是 GPU 指标，`k_i` 是 KV cache 水位，`c_i` 是冷启动和 warm pool 状态，`h_i` 是租户或优先级，`u_i` 是成本预算，`z_i` 是 trace 和配置版本。

普通 HPA 的核心比例思想可以抽象成：

```math
N_{\mathrm{des}}=\left\lceil N_{\mathrm{cur}}\max_j \frac{M_j}{B_j}\right\rceil
```

其中，`N_{\mathrm{cur}}` 是当前副本数，`M_j` 是第 `j` 个观测指标，`B_j` 是该指标的目标阈值。这个公式的价值是简单稳定；它的问题是如果只选 CPU 或 GPU utilization，就会漏掉 LLM 的 token 负载、队列和 KV 压力。

LLM 推理应该先把 QPS 转成 token 负载：

```math
\Lambda_{\mathrm{in}}=\lambda_{\mathrm{req}}\mathbb{E}[n_{\mathrm{in}}],\qquad
\Lambda_{\mathrm{out}}=\lambda_{\mathrm{req}}\mathbb{E}[n_{\mathrm{out}}]
```

其中，`\Lambda_{\mathrm{in}}` 是 input tokens/s，`\Lambda_{\mathrm{out}}` 是 output tokens/s，`n_{\mathrm{in}}` 和 `n_{\mathrm{out}}` 是单请求输入 / 输出 token 数。两个 endpoint 的 QPS 一样时，`\Lambda_{\mathrm{in}}` 和 `\Lambda_{\mathrm{out}}` 可能完全不同。

基于 token 的副本需求可以写成：

```math
N_{\mathrm{tok}}=\max\left(
\left\lceil \frac{\Lambda_{\mathrm{in}}}{\mu_{\mathrm{in}}}\right\rceil,
\left\lceil \frac{\Lambda_{\mathrm{out}}}{\mu_{\mathrm{out}}}\right\rceil
\right)
```

其中，`\mu_{\mathrm{in}}` 是单副本在目标 SLO 下可承载的 input tokens/s，`\mu_{\mathrm{out}}` 是单副本在目标 SLO 下可承载的 output tokens/s。它们不是理论峰值，而应该来自压测和线上回放。

队列等待可以用 Little's Law 做直觉估算：

```math
W_q\approx \frac{L_q}{\lambda_{\mathrm{req}}}
```

其中，`L_q` 是等待队列长度，`W_q` 是平均等待时间。线上扩容不应等 timeout 已经爆发，而应在 `L_q` 和 `W_q` 持续增长时提前行动。

冷启动时间可以拆成：

```math
T_{\mathrm{ready}}=T_{\mathrm{sched}}+T_{\mathrm{image}}+T_{\mathrm{weight}}+T_{\mathrm{init}}+T_{\mathrm{warm}}+T_{\mathrm{register}}
```

其中，`T_{\mathrm{sched}}` 是调度 GPU 节点时间，`T_{\mathrm{image}}` 是镜像拉取和解包时间，`T_{\mathrm{weight}}` 是权重加载时间，`T_{\mathrm{init}}` 是 runtime / CUDA 初始化时间，`T_{\mathrm{warm}}` 是 kernel 和 cache 预热时间，`T_{\mathrm{register}}` 是 endpoint 注册和路由生效时间。

因此扩容真正生效的容量不是当前副本，而是：

```math
N_{\mathrm{eff}}(t+T_{\mathrm{ready}})=N_{\mathrm{cur}}+N_{\mathrm{warm}}+N_{\mathrm{cold}}(t+T_{\mathrm{ready}})
```

其中，`N_{\mathrm{warm}}` 是 warm pool 中可以快速接流量的副本，`N_{\mathrm{cold}}` 是冷启动完成后的新增副本。大模型扩容如果不考虑 `T_{\mathrm{ready}}`，就会在流量尖峰期间一直追不上需求。

缩容要比扩容更保守。一个简单的缩容安全条件可以写成：

```math
S_{\mathrm{down}}=\mathbf{1}\left[
L_q=0 \land A_{\mathrm{stream}}=0 \land T_{\mathrm{cool}}\ge B_{\mathrm{cool}} \land D=1
\right]
```

其中，`A_{\mathrm{stream}}` 是活跃流式请求数，`T_{\mathrm{cool}}` 是距离上次缩容或扩容的 cooldown 时间，`B_{\mathrm{cool}}` 是 cooldown 阈值，`D` 表示待缩容实例已经进入 draining 并停止接新请求。没有这个条件，缩容很容易中断流式连接或造成振荡。

最终 autoscaling 门禁可以写成：

```math
G_{\mathrm{autoscale}}=\mathbf{1}\left[\min_j C_j\ge\tau_j \land T_{\mathrm{ttft,p95}}\le B_{\mathrm{ttft}} \land T_{\mathrm{tpot,p95}}\le B_{\mathrm{tpot}} \land W_{q,\mathrm{p95}}\le B_q \land R_{\mathrm{kv}}\le \rho_{\mathrm{kv}} \land N_{\mathrm{eff}}\ge N_{\mathrm{need}} \land P_0=0\right]
```

其中，`C_j` 是各审计维度覆盖率，`R_{\mathrm{kv}}` 是 KV pressure，`N_{\mathrm{need}}` 是多指标推导出的需求副本数，`P_0` 是未关闭的 P0 风险数。这个门禁强调：autoscaler 不是单独的 K8s 配置，而是推理平台容量控制闭环。

下面这个 0 依赖 Python demo 演示一个简化 LLM-aware autoscaler：分别按 QPS、input tokens/s、output tokens/s、queue wait、TTFT、TPOT、KV pressure 和 GPU utilization 计算建议副本数，再结合 warm pool、冷启动、预算上限和 draining 安全做动作判断。

```python
from math import ceil

CHECKS = [
    "traffic_token_profile",
    "slo_latency_contract",
    "queue_backlog_signal",
    "gpu_kv_capacity_signal",
    "cold_start_lead_time",
    "warm_pool_readiness",
    "multi_metric_recommendation",
    "scale_up_down_policy",
    "draining_streaming_safety",
    "router_admission_coupling",
    "tenant_quota_priority",
    "cost_budget_guard",
    "observability_trace_coverage",
    "autoscaling_gate",
]

CAPACITY = {
    "per_replica_qps": 7.0,
    "per_replica_input_tps": 9000,
    "per_replica_output_tps": 900,
    "queue_wait_target_s": 0.8,
    "ttft_target_s": 1.8,
    "tpot_target_ms": 45,
    "kv_pressure_target": 0.78,
    "gpu_util_target": 0.82,
    "max_replicas": 12,
    "min_replicas": 2,
    "current_replicas": 4,
    "warm_pool_ready": 2,
    "cold_start_s": 180,
    "budget_max_replicas": 10,
    "scale_down_cooldown_s": 300,
}

WINDOWS = [
    {
        "id": "normal",
        "qps": 18,
        "input_tps": 42000,
        "output_tps": 2600,
        "queue_len": 12,
        "queue_wait_p95_s": 0.55,
        "ttft_p95_s": 1.3,
        "tpot_p95_ms": 32,
        "kv_pressure": 0.52,
        "gpu_util": 0.68,
        "timeout_rate": 0.002,
        "tenant_burst": False,
    },
    {
        "id": "prefill_spike",
        "qps": 24,
        "input_tps": 82000,
        "output_tps": 3300,
        "queue_len": 45,
        "queue_wait_p95_s": 1.35,
        "ttft_p95_s": 2.7,
        "tpot_p95_ms": 38,
        "kv_pressure": 0.64,
        "gpu_util": 0.74,
        "timeout_rate": 0.01,
        "tenant_burst": False,
    },
    {
        "id": "decode_kv_pressure",
        "qps": 21,
        "input_tps": 50000,
        "output_tps": 6100,
        "queue_len": 52,
        "queue_wait_p95_s": 1.1,
        "ttft_p95_s": 1.9,
        "tpot_p95_ms": 68,
        "kv_pressure": 0.88,
        "gpu_util": 0.79,
        "timeout_rate": 0.018,
        "tenant_burst": False,
    },
    {
        "id": "low_traffic_but_long_streams",
        "qps": 8,
        "input_tps": 12000,
        "output_tps": 1300,
        "queue_len": 4,
        "queue_wait_p95_s": 0.30,
        "ttft_p95_s": 1.0,
        "tpot_p95_ms": 30,
        "kv_pressure": 0.42,
        "gpu_util": 0.28,
        "timeout_rate": 0.002,
        "tenant_burst": False,
        "active_streams": 9,
    },
]


def desired_by_ratio(current, current_value, target_value):
    if target_value <= 0:
        return current
    return ceil(current * current_value / target_value)


def recommendations(window):
    current = CAPACITY["current_replicas"]
    rec = {
        "qps": ceil(window["qps"] / CAPACITY["per_replica_qps"]),
        "input_tokens": ceil(window["input_tps"] / CAPACITY["per_replica_input_tps"]),
        "output_tokens": ceil(window["output_tps"] / CAPACITY["per_replica_output_tps"]),
        "queue_wait": desired_by_ratio(current, window["queue_wait_p95_s"], CAPACITY["queue_wait_target_s"]),
        "ttft": desired_by_ratio(current, window["ttft_p95_s"], CAPACITY["ttft_target_s"]),
        "tpot": desired_by_ratio(current, window["tpot_p95_ms"], CAPACITY["tpot_target_ms"]),
        "kv": desired_by_ratio(current, window["kv_pressure"], CAPACITY["kv_pressure_target"]),
        "gpu": desired_by_ratio(current, window["gpu_util"], CAPACITY["gpu_util_target"]),
    }
    raw = max(rec.values())
    capped = min(
        max(raw, CAPACITY["min_replicas"]),
        CAPACITY["max_replicas"],
        CAPACITY["budget_max_replicas"],
    )
    warm_deficit = max(0, capped - current - CAPACITY["warm_pool_ready"])
    ready_s = 20 if capped <= current + CAPACITY["warm_pool_ready"] else CAPACITY["cold_start_s"]
    if capped > current:
        action = "scale_up"
    elif capped < current and window.get("active_streams", 0) == 0 and window["queue_wait_p95_s"] < 0.4:
        action = "scale_down_candidate"
    elif capped < current:
        action = "hold_for_draining"
    else:
        action = "hold"
    return rec, raw, capped, action, warm_deficit, ready_s


def build_profile_cases():
    complete = {"name": "complete_autoscaling_case"}
    complete.update({check: True for check in CHECKS})
    cases = [complete]
    bad_names = [
        "traffic_token_profile_missing_bad",
        "slo_latency_contract_missing_bad",
        "queue_signal_missing_bad",
        "gpu_kv_signal_missing_bad",
        "cold_start_ignored_bad",
        "warm_pool_missing_bad",
        "single_metric_hpa_bad",
        "scale_policy_missing_bad",
        "draining_missing_bad",
        "router_not_coupled_bad",
        "tenant_quota_missing_bad",
        "cost_guard_missing_bad",
        "autoscaling_trace_missing_bad",
        "autoscaling_gate_missing_bad",
    ]
    for bad_name, failed_check in zip(bad_names, CHECKS):
        case = {"name": bad_name}
        case.update({check: True for check in CHECKS})
        case[failed_check] = False
        cases.append(case)
    return cases


def audit_profiles(cases):
    metrics = {check: round(sum(1 for case in cases if case.get(check)) / len(cases), 3) for check in CHECKS}
    failed_cases = [case["name"] for case in cases if not all(case.get(check) for check in CHECKS)]
    failed_gates = [check for check, value in metrics.items() if value < 1.0]
    hard_blockers = sum(1 for case in cases if case["name"].endswith("_bad"))
    gate_pass = min(metrics.values()) >= 0.95 and hard_blockers == 0
    return metrics, failed_cases, failed_gates, hard_blockers, gate_pass


summary = {}
for window in WINDOWS:
    rec, raw, capped, action, warm_deficit, ready_s = recommendations(window)
    summary[window["id"]] = {
        "max_signal": max(rec, key=rec.get),
        "raw": raw,
        "capped": capped,
        "action": action,
        "warm_deficit": warm_deficit,
        "ready_s": ready_s,
    }

metrics, failed_cases, failed_gates, hard_blockers, gate_pass = audit_profiles(build_profile_cases())

print("autoscaling_recommendations=", summary)
print("normal_metric_replicas=", recommendations(WINDOWS[0])[0])
print("spike_metric_replicas=", recommendations(WINDOWS[1])[0])
print("kv_metric_replicas=", recommendations(WINDOWS[2])[0])
print("draining_decision=", summary["low_traffic_but_long_streams"]["action"])
print("warm_pool_ready=", CAPACITY["warm_pool_ready"])
print("cold_start_s=", CAPACITY["cold_start_s"])
print("metrics=", metrics)
print("hard_blocker_count=", hard_blockers)
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("autoscaling_gate_pass=", gate_pass)
```

输出示例：

```text
autoscaling_recommendations= {'normal': {'max_signal': 'input_tokens', 'raw': 5, 'capped': 5, 'action': 'scale_up', 'warm_deficit': 0, 'ready_s': 20}, 'prefill_spike': {'max_signal': 'input_tokens', 'raw': 10, 'capped': 10, 'action': 'scale_up', 'warm_deficit': 4, 'ready_s': 180}, 'decode_kv_pressure': {'max_signal': 'output_tokens', 'raw': 7, 'capped': 7, 'action': 'scale_up', 'warm_deficit': 1, 'ready_s': 180}, 'low_traffic_but_long_streams': {'max_signal': 'ttft', 'raw': 3, 'capped': 3, 'action': 'hold_for_draining', 'warm_deficit': 0, 'ready_s': 20}}
normal_metric_replicas= {'qps': 3, 'input_tokens': 5, 'output_tokens': 3, 'queue_wait': 3, 'ttft': 3, 'tpot': 3, 'kv': 3, 'gpu': 4}
spike_metric_replicas= {'qps': 4, 'input_tokens': 10, 'output_tokens': 4, 'queue_wait': 7, 'ttft': 6, 'tpot': 4, 'kv': 4, 'gpu': 4}
kv_metric_replicas= {'qps': 3, 'input_tokens': 6, 'output_tokens': 7, 'queue_wait': 6, 'ttft': 5, 'tpot': 7, 'kv': 5, 'gpu': 4}
draining_decision= hold_for_draining
warm_pool_ready= 2
cold_start_s= 180
metrics= {'traffic_token_profile': 0.933, 'slo_latency_contract': 0.933, 'queue_backlog_signal': 0.933, 'gpu_kv_capacity_signal': 0.933, 'cold_start_lead_time': 0.933, 'warm_pool_readiness': 0.933, 'multi_metric_recommendation': 0.933, 'scale_up_down_policy': 0.933, 'draining_streaming_safety': 0.933, 'router_admission_coupling': 0.933, 'tenant_quota_priority': 0.933, 'cost_budget_guard': 0.933, 'observability_trace_coverage': 0.933, 'autoscaling_gate': 0.933}
hard_blocker_count= 14
autoscaling_gate_pass= False
```

这个 demo 的重点不是给出生产阈值，而是训练一套判断方式：LLM 扩缩容要从 QPS 走向 token 负载，从平均延迟走向 TTFT / TPOT / queue / KV，从副本数走向冷启动和 warm pool，从缩容动作走向 draining 和 streaming 安全。

## 32.21 常见误区

误区一：GPU 利用率高就扩容。

GPU 利用率高可能是健康高吞吐，也可能是过载。必须结合队列、延迟、KV cache 和错误率。

误区二：QPS 低就可以缩容。

QPS 低但请求很长，token 负载可能仍然高。

误区三：扩容立即生效。

大模型冷启动慢，扩容需要提前量。

误区四：缩容只是减少副本。

缩容要处理 active request、streaming、KV cache、draining 和路由摘除。

误区五：扩缩容只属于 K8s。

K8s 只是执行层，推理平台需要 LLM-aware autoscaler。

## 32.22 面试常见追问

问题一：为什么 LLM 推理不能只按 QPS 扩容？

可以回答：因为不同请求的输入输出 token 数差异很大，同样 QPS 资源消耗可能相差几十倍。应该结合 input/output tokens/s、队列、TTFT、TPOT、KV cache 和 GPU 指标。

问题二：为什么不能只看 GPU 利用率？

可以回答：GPU 利用率不能反映队列等待、p99 延迟、KV cache 水位和显存带宽瓶颈。decode 阶段可能不是算力瓶颈，因此需要多指标综合判断。

问题三：大模型扩容为什么慢？

可以回答：需要调度 GPU、拉镜像、加载权重、初始化 runtime、分配显存和预热 kernel，模型越大冷启动越慢。

问题四：如何安全缩容？

可以回答：先将实例从路由摘除并进入 draining，不再接新请求，等待已有请求完成，处理 streaming 和超时，再释放实例，同时设置 cooldown 避免振荡。

问题五：为什么 warm pool 是成本和 SLO 的权衡？

可以回答：warm pool 可以把扩容生效时间从模型冷启动时间降到路由接入时间，但会长期占用 GPU、显存或至少占用已加载权重的资源。关键是用历史峰值、SLO、冷启动时间和预算来确定最小热池，而不是无限常驻。

问题六：扩缩容如何和模型路由联动？

可以回答：autoscaler 发现某模型过载时，router 可以临时把低优先级流量降级、转移到健康 endpoint、启用缓存或限流；同时 autoscaler 扩容目标模型。只改路由可能转移故障，只扩容可能来不及，二者要共享指标和策略版本。

## 32.23 小练习

1. 为什么普通 HPA 不适合直接管理 LLM 推理服务？
2. QPS 和 tokens/s 的区别是什么？
3. TTFT 和队列长度哪个更早反映拥塞？
4. KV cache 水位为什么能作为扩容信号？
5. 大模型冷启动包括哪些步骤？
6. Warm pool 的收益和代价是什么？
7. 缩容为什么要 draining？
8. 如何避免扩缩容振荡？
9. 写出一个多指标 autoscaling 公式，至少包含 input tokens/s、output tokens/s、queue wait、TTFT、TPOT 和 KV pressure。
10. 估算一个模型的 `T_ready`，把调度、镜像、权重加载、runtime 初始化、kernel 预热和路由注册拆开。
11. 设计一个缩容 draining 状态机，覆盖 active requests、streaming、timeout、router 摘除和 cooldown。
12. 为多租户推理平台设计 autoscaling trace 字段，要求能解释一次扩容是哪个指标触发的。

## 32.24 本章小结

本章讲了大模型推理平台的自动扩缩容。

你需要记住：

1. LLM 推理扩缩容不能只看 CPU、QPS 或 GPU 利用率。
2. 更关键的指标包括 TTFT、TPOT、队列长度、tokens/s、KV cache 和错误率。
3. 大模型冷启动慢，扩容必须提前，常需要 warm pool 和预测扩容。
4. 缩容要处理 draining、streaming、active requests 和缓存失效。
5. 多模型、多租户场景下，扩缩容必须和路由、配额、限流、成本治理联动。
6. 成熟的推理 autoscaler 是 LLM-aware 的控制闭环，不只是 K8s HPA 配置。
7. Autoscaling 门禁要同时检查 token 画像、SLO、队列、GPU / KV、冷启动、warm pool、多指标推荐、扩缩容策略、draining、路由准入、租户配额、成本和 trace。

下一章我们会讲限流、熔断、重试、超时和降级。
