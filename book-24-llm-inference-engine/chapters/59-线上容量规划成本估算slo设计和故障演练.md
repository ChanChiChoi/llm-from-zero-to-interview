# 第 59 章 线上容量规划、成本估算、SLO 设计和故障演练

第 58 章我们讨论了如何把 LLM serving engine 部署到生产环境。

部署成功只是第一步。

真正上线后，团队马上会遇到更现实的问题：

1. 需要多少张 GPU。
2. 高峰期能不能扛住。
3. TTFT 和 TPOT 应该承诺到什么水平。
4. 成本为什么这么高。
5. 要不要为大客户预留容量。
6. 长上下文请求会不会拖垮系统。
7. 某个 worker 掉了以后容量还够不够。
8. 对象存储、镜像仓库、调度系统出问题时服务怎么降级。
9. 灰度版本有性能退化时如何发现。
10. 如何向业务、产品和面试官解释容量规划方法。

这些问题已经不只是 engine 内部实现。

它们连接了 serving engine、推理平台、业务流量、成本预算和稳定性工程。

本章讨论 LLM serving 线上容量规划、成本估算、SLO 设计和故障演练。

## 59.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 LLM serving 的容量规划、成本估算、SLO 设计和故障演练框架。

公开资料校准主要参考四类口径：

1. Google SRE Workbook 对 SLO、SLI、error budget、burn rate 和告警策略的公开方法论。
2. vLLM benchmark、metrics、optimization / tuning 文档对 TTFT、TPOT、E2E latency、KV cache usage、preemption、queue time、running / waiting requests 和压测指标的公开口径。
3. Kubernetes resource management 和 deployment 文档对 resource request / limit、副本数、滚动发布和可用容量的通用工程语义。
4. 本书第 53 到 58 章对 benchmark framework、异步 serving、多 worker router、分布式 KV、API 层和生产部署门禁的教学抽象。

本章不提供某个 GPU、某个模型、某个云厂商或某个框架版本的通用容量答案，也不替代真实压测、真实账单、真实 SLO 审批、真实值班制度或真实故障演练平台。我们只保留一个能解释、能复算、能演练的闭环：

```text
workload token profile -> benchmark stable capacity -> GPU count -> KV concurrency -> cost attribution -> SLO and error budget -> admission policy -> fault drill -> rollback and runbook
```

第二轮新增 demo 的验收重点是：容量不能只按 QPS 算；要同时用 input token/s、output token/s、request rate 和 KV active tokens 估算 worker 数；规划容量要包含利用率、安全余量、N+1 和发布容量；成本要能按 GPU 小时和 token 粗分摊；SLO 要能计算 error budget；故障演练要证明 worker 丢失、依赖不可用、长上下文突增和新版本退化都有可执行动作。

## 59.1 本章目标

读完本章，你应该能讲清：

1. LLM serving 容量为什么不能只看 QPS。
2. request QPS、input token/s、output token/s 的区别。
3. TTFT、TPOT、E2E latency 和吞吐之间的关系。
4. 如何用 benchmark 结果估算线上 GPU 数量。
5. 为什么上下文长度分布会显著影响容量。
6. KV cache 容量如何限制并发请求数。
7. 成本估算应该按 GPU 小时、token 和租户拆分。
8. SLO 应该如何定义、监控和触发告警。
9. 故障演练应该覆盖哪些场景。
10. 面试中如何回答“如何规划一个 LLM serving 集群容量”。

本章重点不是给出一个万能公式。

重点是建立一套能落地的工程估算框架。

## 59.2 为什么不能只看 QPS

普通 Web 服务经常用 QPS 做容量规划。

例如：

```text
单实例 1000 QPS
高峰 10000 QPS
需要 10 个实例
```

LLM serving 不能这样简单估算。

原因是两个请求的成本可能差几个数量级：

```text
请求 A:
  input tokens: 32
  output tokens: 16

请求 B:
  input tokens: 32000
  output tokens: 2048
```

它们都是 1 个请求。

但对 GPU、KV cache、scheduler、网络和队列的压力完全不同。

LLM serving 至少要同时看：

1. request QPS。
2. input token/s。
3. output token/s。
4. prefill token/s。
5. decode token/s。
6. 平均输入长度。
7. P95/P99 输入长度。
8. 平均输出长度。
9. P95/P99 输出长度。
10. 并发请求数。
11. KV cache 占用。
12. TTFT 和 TPOT。

如果只看 QPS，很容易低估长上下文和长输出的成本。

## 59.3 容量规划的核心单位

LLM serving 的核心资源消耗来自两部分：

1. prefill。
2. decode。

prefill 处理输入 prompt。

decode 逐 token 生成输出。

因此容量规划的基本单位不是 request，而是 token。

可以先建立几个指标：

```text
RPS = 每秒请求数
InputTokensPerRequest = 平均输入 token 数
OutputTokensPerRequest = 平均输出 token 数

InputTokenRate = RPS * InputTokensPerRequest
OutputTokenRate = RPS * OutputTokensPerRequest
```

例如：

```text
RPS = 20
平均输入 = 1000 tokens
平均输出 = 200 tokens

InputTokenRate = 20 * 1000 = 20000 tokens/s
OutputTokenRate = 20 * 200 = 4000 tokens/s
```

这比“20 QPS”有意义得多。

因为 engine 真正要处理的是：

```text
每秒 20000 个 prompt tokens
每秒 4000 个 generated tokens
```

## 59.4 Prefill 和 Decode 的容量不同

prefill 和 decode 的资源画像不同。

prefill 通常更像大矩阵计算。

特点是：

1. 并行度高。
2. 对算力敏感。
3. 输入越长，单次计算越重。
4. 影响 TTFT。

decode 通常逐 token 迭代。

特点是：

1. 每步处理一个或少量 token。
2. 对 batch size、KV cache 读取和调度敏感。
3. 持续占用请求槽位。
4. 影响 TPOT 和整体吞吐。

所以容量规划要分开看：

```text
prefill capacity >= input token rate
decode capacity >= output token rate
KV cache capacity >= active sequence tokens
```

其中任何一个不满足，服务都会出问题。

prefill 不够时，典型现象是：

1. TTFT 升高。
2. 请求排队时间变长。
3. 新请求迟迟拿不到第一个 token。

decode 不够时，典型现象是：

1. TPOT 升高。
2. streaming 变慢。
3. 已经开始的请求长时间占用 worker。

KV cache 不够时，典型现象是：

1. 可接收并发下降。
2. scheduler 频繁 preempt。
3. OOM 或请求被拒绝。

## 59.5 从 Benchmark 到容量估算

容量估算必须建立在 benchmark 上。

不能只看模型参数量或 GPU 理论 FLOPS。

一个最小 benchmark 应该固定：

1. 模型。
2. GPU 型号。
3. tensor parallel size。
4. dtype。
5. max context length。
6. scheduler 参数。
7. max_num_batched_tokens。
8. gpu_memory_utilization。
9. prefix cache 是否开启。
10. 输入长度分布。
11. 输出长度分布。
12. 并发水平。

benchmark 输出至少包括：

1. request throughput。
2. input token throughput。
3. output token throughput。
4. TTFT P50/P95/P99。
5. TPOT P50/P95/P99。
6. E2E latency P50/P95/P99。
7. GPU utilization。
8. GPU memory usage。
9. KV cache usage。
10. preemption 次数。
11. OOM 次数。
12. 失败率。

假设某个 4 卡 worker benchmark 结果是：

```text
model: chat-32b
gpu: 4 * H100
input length: avg 1000, p95 4000
output length: avg 200, p95 800

stable capacity:
  RPS: 18
  input tokens/s: 18000
  output tokens/s: 3600
  TTFT P95: 900 ms
  TPOT P95: 35 ms
```

如果线上高峰预估是：

```text
RPS: 90
input tokens/s: 90000
output tokens/s: 18000
```

只按平均吞吐估算：

```text
workers = ceil(90 / 18) = 5
```

但这只是理论下限。

生产环境还要加冗余。

## 59.6 安全余量

线上不能按 100% benchmark 容量运行。

原因包括：

1. 流量有突刺。
2. 输入长度分布会变化。
3. 输出长度不可完全预测。
4. 某些请求会触发长上下文。
5. GPU 性能存在波动。
6. worker 可能重启或下线。
7. 灰度升级会临时占用容量。
8. 故障恢复需要预留空间。
9. 监控和日志也有开销。
10. benchmark 通常比真实业务更干净。

常见做法是只把单 worker 稳定容量的 60% 到 80% 作为规划容量。

例如：

```text
benchmark stable RPS = 18
planning utilization = 70%
planning RPS per worker = 18 * 0.7 = 12.6

peak RPS = 90
workers = ceil(90 / 12.6) = 8
```

如果一个 worker 需要 4 张 GPU：

```text
GPU count = 8 * 4 = 32
```

这比只按 benchmark 极限算出来的 20 张 GPU 更接近生产现实。

## 59.7 N+1 和故障容量

容量规划还要考虑 worker 故障。

如果集群有 8 个 worker，每个 worker 承载 12.5% 流量。

当一个 worker 掉线，剩余 7 个 worker 要承载全部流量。

剩余 worker 的负载会变成：

```text
new load per worker = 1 / 7 = 14.3%
relative increase = 14.3 / 12.5 - 1 = 14.4%
```

如果原来已经跑到 85% 利用率，一个 worker 掉线后可能直接过载。

因此要定义故障容量目标。

例如：

```text
正常状态：P95 流量下利用率 <= 70%
N-1 状态：损失 1 个 worker 后利用率 <= 85%
N-2 状态：允许降级，但错误率不能失控
```

大规模集群还要考虑机架、可用区或节点池故障。

容量不是只看总 GPU 数。

还要看故障域分布。

## 59.8 上下文长度分布

LLM serving 的容量经常被长上下文请求打穿。

平均输入长度可能很低：

```text
avg input tokens = 800
```

但 P99 可能很高：

```text
p99 input tokens = 28000
```

长 prompt 会带来几类问题：

1. prefill 时间长。
2. TTFT 高。
3. KV cache 占用大。
4. batch packing 更困难。
5. 对其他短请求造成排队影响。

容量规划要按分布而不是平均值。

至少要统计：

1. P50 输入长度。
2. P90 输入长度。
3. P95 输入长度。
4. P99 输入长度。
5. 最大输入长度。
6. P50 输出长度。
7. P90 输出长度。
8. P95 输出长度。
9. P99 输出长度。
10. 最大输出长度。

如果业务同时有短问答和长文档总结，不要把它们混在一个平均数里。

应该分 workload 建模。

## 59.9 Workload 分层

一个线上模型可能服务多类业务：

```text
workload A: 客服短问答
  input avg: 300
  output avg: 120
  latency sensitive: high

workload B: 代码生成
  input avg: 2000
  output avg: 800
  latency sensitive: medium

workload C: 长文档总结
  input avg: 24000
  output avg: 1000
  latency sensitive: low
```

如果都进入同一个 worker pool，长文档总结可能拖慢客服短问答。

常见隔离方式包括：

1. 按模型拆 worker pool。
2. 按租户拆 worker pool。
3. 按上下文长度拆 worker pool。
4. 按优先级拆队列。
5. 按 SLA 拆流量入口。
6. 长任务进入异步 batch 或低优先级队列。

容量规划时要分别估算每类 workload。

不要让低优先级长请求吃掉高优先级短请求的 SLO。

## 59.10 KV Cache 容量估算

KV cache 是 LLM serving 的关键容量约束。

对每个 active sequence，需要保存历史 token 的 key/value。

简化估算可以写成：

```text
KV bytes per token
  = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

其中：

1. `2` 表示 key 和 value。
2. `num_layers` 是层数。
3. `num_kv_heads` 是 KV head 数。
4. `head_dim` 是每个 head 的维度。
5. `bytes_per_element` 取决于 fp16、bf16、fp8 等。

假设：

```text
num_layers = 40
num_kv_heads = 8
head_dim = 128
bytes_per_element = 2
```

则：

```text
KV bytes per token = 2 * 40 * 8 * 128 * 2 = 163840 bytes
                   ≈ 160 KB/token
```

如果一个请求总上下文长度是：

```text
input 4000 + generated 1000 = 5000 tokens
```

则这个请求的 KV cache 约为：

```text
5000 * 160 KB ≈ 800 MB
```

这只是示意，真实模型还要考虑 tensor parallel、GQA/MQA、block size、对齐、碎片和实现细节。

但它能帮助你理解：长上下文并发非常贵。

## 59.11 并发上限不是 QPS 上限

decode 阶段请求会持续停留在系统里。

因此并发上限同样重要。

可以用 Little's Law 做粗略估计：

```text
Concurrency = ArrivalRate * AverageLatency
```

例如：

```text
RPS = 20
平均端到端延迟 = 10s

active requests ≈ 20 * 10 = 200
```

如果每个 active request 平均占用 3000 tokens KV cache：

```text
active tokens = 200 * 3000 = 600000 tokens
```

这会直接决定 KV cache 需要多少显存。

当输出变长时，即使 RPS 不变，并发也会上升。

因此 capacity dashboard 必须同时展示：

1. waiting requests。
2. running requests。
3. active sequences。
4. active tokens。
5. KV cache allocated blocks。
6. KV cache free blocks。
7. preempted sequences。

## 59.12 Admission Control

容量规划不只是买更多 GPU。

还要决定什么时候拒绝请求。

如果系统已经接近极限，继续接收新请求可能导致：

1. 所有请求延迟变差。
2. KV cache 被打满。
3. preemption 频繁发生。
4. worker OOM。
5. 大面积 5xx。

更好的做法是 admission control。

也就是在入口处判断请求是否能进入系统。

常见判断条件包括：

1. 当前队列长度。
2. 预计输入 token 数。
3. `max_tokens`。
4. 当前 KV cache 空闲量。
5. 当前 running request 数。
6. 租户配额。
7. 请求优先级。
8. deadline 或 timeout。

拒绝请求时应该返回明确错误：

```json
{
  "error": {
    "type": "capacity_exceeded",
    "message": "The model is temporarily overloaded. Please retry later.",
    "code": "overloaded"
  }
}
```

有控制地拒绝一部分请求，比让所有请求一起超时更好。

## 59.13 成本估算的基本模型

LLM serving 成本主要来自 GPU。

一个基础成本模型是：

```text
MonthlyCost = GPUCount * GPUHourlyPrice * 24 * 30
```

例如：

```text
GPUCount = 32
GPUHourlyPrice = $4

MonthlyCost = 32 * 4 * 24 * 30 = $92160
```

这还只是 GPU 租用成本。

真实成本还包括：

1. CPU 和内存。
2. 本地 NVMe。
3. 对象存储。
4. 镜像仓库。
5. 跨可用区网络。
6. 日志和 metrics。
7. 负载均衡。
8. 空闲冗余容量。
9. 工程维护成本。
10. 失败重试带来的额外 token。

容量规划时不要只算满载 token 成本。

线上系统必须为峰值、故障和发布留空间，这些都会转化为空闲成本。

## 59.14 Token 成本

业务通常更关心每 1000 token 或每 100 万 token 的成本。

可以从集群视角估算：

```text
CostPerOutputToken = ClusterCostPerSecond / OutputTokensPerSecond
```

例如：

```text
32 GPUs
$4 / GPU-hour

ClusterCostPerHour = 32 * 4 = $128/hour
ClusterCostPerSecond = 128 / 3600 = $0.0356/s

OutputTokensPerSecond = 18000

CostPerOutputToken = 0.0356 / 18000 = $0.00000198
CostPerMillionOutputTokens ≈ $1.98
```

但这个数字容易误导。

因为它假设一直跑在该吞吐下。

如果实际平均利用率只有 35%，成本会变成：

```text
EffectiveCost = IdealCost / Utilization
              = 1.98 / 0.35
              ≈ $5.66 per million output tokens
```

所以成本优化的核心之一是提升有效利用率。

但利用率不能无限提高。

过高利用率会牺牲 SLO 和故障余量。

## 59.15 成本归因

多租户 serving 系统需要成本归因。

否则你只知道总账单很高，不知道谁在消耗。

日志中至少要记录：

1. tenant id。
2. user id 或 service id。
3. model id。
4. model revision。
5. input tokens。
6. output tokens。
7. cached input tokens。
8. TTFT。
9. TPOT。
10. E2E latency。
11. worker id。
12. GPU type。
13. finish reason。
14. error code。

成本归因可以按不同口径计算：

```text
按 token 分摊：简单，适合产品计费
按 GPU time 分摊：更接近真实成本
按 reserved capacity 分摊：适合大客户专属容量
按 peak capacity 分摊：适合需要保障峰值的业务
```

面试中要强调：token 计费简单，但不一定等于资源成本。

长上下文请求的 input token 和 KV cache 占用可能比 output token 更贵。

## 59.16 SLO、SLA 和 SLI

先区分三个概念：

```text
SLI: Service Level Indicator，实际观测指标
SLO: Service Level Objective，内部目标
SLA: Service Level Agreement，对外承诺
```

例如：

```text
SLI: /v1/chat/completions 的 TTFT P95
SLO: TTFT P95 < 1.5s，按 5 分钟窗口统计
SLA: 月度可用性 99.9%，违反后赔付
```

工程上通常先建立 SLI，再设 SLO，最后才谨慎承诺 SLA。

不要一开始就承诺过细的外部 SLA。

LLM serving 的延迟强依赖：

1. 模型大小。
2. 输入长度。
3. 输出长度。
4. 并发水平。
5. 是否命中 prefix cache。
6. 租户优先级。
7. GPU 型号。
8. 当前故障状态。

因此 SLO 应该带条件。

## 59.17 LLM Serving 的关键 SLI

常见 SLI 包括：

1. availability。
2. HTTP 5xx rate。
3. model error rate。
4. timeout rate。
5. throttle rate。
6. TTFT P50/P95/P99。
7. TPOT P50/P95/P99。
8. E2E latency P50/P95/P99。
9. request queue time。
10. input token throughput。
11. output token throughput。
12. running requests。
13. waiting requests。
14. KV cache usage。
15. preemption rate。
16. GPU utilization。
17. GPU memory usage。
18. worker restart count。
19. router no-capacity count。
20. client disconnect rate。

对 streaming 接口来说，TTFT 和 TPOT 比单纯 E2E latency 更重要。

因为用户体感通常是：

1. 多久看到第一个 token。
2. 后续 token 是否流畅。

## 59.18 SLO 示例

一个生产系统可以这样定义 SLO：

```text
Availability:
  99.9% requests receive non-5xx response

For short-chat workload:
  condition: input_tokens <= 1024 and max_tokens <= 512
  TTFT P95 <= 1.5s
  TPOT P95 <= 50ms
  timeout rate <= 0.5%

For long-context workload:
  condition: input_tokens <= 32000 and max_tokens <= 1024
  TTFT P95 <= 8s
  TPOT P95 <= 80ms
  timeout rate <= 1%

Capacity control:
  overload rejection returns 429 or 503 with stable error body
  no silent timeout due to unlimited queueing
```

注意这里按 workload 分开定义。

短请求和长请求不应该共用一个延迟目标。

否则 SLO 要么不现实，要么没有约束力。

## 59.19 Error Budget

SLO 可以转化为 error budget。

例如月可用性 SLO 是 99.9%。

一个 30 天月总时间是：

```text
30 * 24 * 60 = 43200 minutes
```

允许不可用时间是：

```text
43200 * 0.1% = 43.2 minutes
```

这就是 error budget。

如果一个月内已经消耗了 30 分钟，发布策略就应该变保守。

例如：

1. 暂停非紧急发布。
2. 降低灰度速度。
3. 增加人工审批。
4. 优先修稳定性问题。
5. 扩容高风险模型池。

error budget 的价值是把“稳定性”和“迭代速度”连接起来。

## 59.20 告警设计

告警不能只看 HTTP 500。

LLM serving 退化经常先表现为延迟和容量压力。

建议分层告警：

```text
用户可见层:
  availability drop
  5xx rate increase
  timeout rate increase
  TTFT P95/P99 increase
  TPOT P95/P99 increase

容量层:
  queue length high
  no available worker
  KV cache free blocks low
  preemption rate high
  GPU memory near limit

执行层:
  worker restart
  CUDA OOM
  NCCL error
  model load failure
  tokenizer error

依赖层:
  object storage error
  registry pull failure
  metrics pipeline delay
  router registry unavailable
```

告警要避免两个极端：

1. 指标一抖就报警，导致告警疲劳。
2. 只在完全不可用时报警，已经太晚。

比较实用的方式是多窗口告警。

例如：

```text
fast burn:
  5 分钟错误预算消耗过快，立即报警

slow burn:
  1 小时错误预算持续消耗，创建工单或低优告警
```

## 59.21 Dashboard 应该展示什么

一个 LLM serving dashboard 至少应该有这些面板：

1. RPS。
2. input tokens/s。
3. output tokens/s。
4. TTFT P50/P95/P99。
5. TPOT P50/P95/P99。
6. E2E latency P50/P95/P99。
7. queue time。
8. running/waiting requests。
9. input/output length distribution。
10. KV cache usage/free blocks。
11. preemption count。
12. GPU utilization。
13. GPU memory usage。
14. error rate by error code。
15. timeout and cancel rate。
16. worker state distribution。
17. model revision distribution。
18. tenant traffic distribution。
19. prefix cache hit rate。
20. cost per model or tenant。

关键是 dashboard 必须能按维度切分：

1. model。
2. revision。
3. tenant。
4. route。
5. worker pool。
6. GPU type。
7. region or zone。

否则总量指标会掩盖局部故障。

## 59.22 压测策略

上线前压测不能只发固定 prompt。

至少要覆盖：

1. 短输入短输出。
2. 短输入长输出。
3. 长输入短输出。
4. 长输入长输出。
5. 混合输入长度分布。
6. 混合输出长度分布。
7. streaming 和 non-streaming。
8. prefix cache 命中和不命中。
9. 高并发突刺。
10. 慢客户端消费。
11. client cancel。
12. 超过上下文限制的非法请求。
13. 超过 max_tokens 的请求。
14. 多租户混合流量。

压测要回答几个问题：

1. 稳定容量是多少。
2. 拐点在哪里。
3. 过载时系统如何表现。
4. 是否会优雅拒绝。
5. 是否会 OOM。
6. 恢复后是否能回到正常状态。

最重要的不是峰值数字，而是过载行为是否可控。

## 59.23 线上容量预测流程

一个可落地的容量预测流程是：

1. 收集历史流量。
2. 按模型、租户和 workload 切分。
3. 统计 RPS、input tokens/s、output tokens/s。
4. 统计输入和输出长度分布。
5. 估算高峰流量和增长率。
6. 用真实分布做 benchmark。
7. 得到单 worker 稳定容量。
8. 应用利用率目标和安全余量。
9. 应用 N+1 或故障域冗余。
10. 得到 worker 和 GPU 数。
11. 估算成本。
12. 和预算、SLO、发布策略一起评审。

可以写成一个简单表格：

```text
model: chat-32b
workload: short-chat
peak rps: 90
peak input tokens/s: 90000
peak output tokens/s: 18000
single worker planning rps: 12.6
worker count: 8
tp size: 4
gpu count: 32
monthly gpu cost: $92160
normal utilization target: 70%
N-1 utilization target: <= 85%
```

这个表比一句“需要 32 张卡”更有解释力。

## 59.24 自动扩缩容的难点

普通服务可以根据 CPU 或 QPS 自动扩缩容。

LLM serving 的自动扩缩容更难。

原因包括：

1. GPU 资源稀缺。
2. worker cold start 很慢。
3. 模型权重下载慢。
4. warmup 需要时间。
5. 扩容后短时间内不一定能接流量。
6. 缩容需要 drain，不能直接杀 stream。
7. 负载由 token 分布决定，不只是 QPS。
8. 突发流量可能比扩容速度快得多。

因此自动扩缩容通常要结合：

1. 预测式扩容。
2. 定时扩容。
3. 按 token throughput 扩容。
4. 按 queue time 扩容。
5. 按 KV cache pressure 扩容。
6. 保留 warm pool。
7. 限流和降级。

不要指望冷启动扩容解决秒级突刺。

秒级突刺主要靠预留容量、队列、限流和降级。

## 59.25 降级策略

当容量不足时，可以考虑降级。

常见降级方式包括：

1. 降低低优先级租户配额。
2. 拒绝超长上下文请求。
3. 降低 `max_tokens` 上限。
4. 关闭部分非关键模型。
5. 把低优先级任务切到异步队列。
6. 使用更小模型兜底。
7. 关闭 expensive decoding 选项。
8. 降低并发上限。
9. 对重复请求利用缓存。
10. 对非关键场景返回稍后重试。

降级必须是产品和工程共同定义的。

不能线上出事时临时决定。

每个降级策略都要明确：

1. 触发条件。
2. 影响范围。
3. 用户可见行为。
4. 恢复条件。
5. 负责人。

## 59.26 故障演练为什么重要

没有演练的故障预案通常不可用。

LLM serving 的故障演练尤其重要，因为很多问题只在 GPU 和长连接场景下暴露。

演练目标不是制造事故。

演练目标是验证：

1. 故障能否被监控发现。
2. 告警是否发给正确的人。
3. runbook 是否可执行。
4. 系统是否能自动隔离故障。
5. 降级是否生效。
6. 回滚是否足够快。
7. 用户影响是否符合预期。
8. 事后是否能通过日志定位。

如果只写文档不演练，真正故障时通常会发现关键步骤缺失。

## 59.27 常见故障演练场景

建议覆盖这些场景：

1. 单 worker 进程退出。
2. 单 GPU OOM。
3. 多卡 worker 某个 rank hang。
4. NCCL 初始化失败。
5. 模型权重下载失败。
6. tokenizer 或 config 缺失。
7. readiness 一直不成功。
8. router 和 worker 心跳中断。
9. KV cache 接近耗尽。
10. prefix cache 异常膨胀。
11. 长上下文流量突增。
12. 某租户请求量突增。
13. client 大量取消 stream。
14. 下游网络抖动导致慢客户端。
15. 新模型 revision TTFT 退化。
16. 新 engine 版本 OOM 升高。
17. 对象存储不可用。
18. 镜像仓库不可用。
19. metrics pipeline 延迟。
20. 可用区部分故障。

每个演练都应该有明确预期。

例如：

```text
场景：单 worker 进程退出

预期：
  1. liveness 失败。
  2. worker 从 router 摘除。
  3. 新请求不再路由到该 worker。
  4. running stream 失败率受限。
  5. 剩余 worker 利用率上升但不超过 N-1 目标。
  6. 告警触发。
  7. worker 自动重启并完成 warmup 后重新注册。
```

## 59.28 Runbook

runbook 是故障处理手册。

一个好的 runbook 不应该只写“查看日志”。

它应该包含：

1. 故障现象。
2. 可能原因。
3. 影响判断。
4. 查询 dashboard 链接。
5. 关键日志字段。
6. 快速止血步骤。
7. 回滚步骤。
8. 降级步骤。
9. 验证恢复的方法。
10. 升级联系人。

例如 KV cache 压力过高的 runbook 可以写：

```text
现象：
  KV cache free blocks 持续低于 10%
  preemption rate 升高
  TTFT/TPOT 同时退化

可能原因：
  长上下文流量增加
  输出长度增加
  某租户突增
  max_num_batched_tokens 或 memory utilization 配置变化

止血：
  降低低优先级租户 max_tokens
  对超长输入返回 overload 或引导异步处理
  扩容 worker pool
  如由新版本导致，切回旧 revision

验证：
  KV cache free blocks 回升
  preemption rate 下降
  TTFT/TPOT 恢复到 SLO 内
```

runbook 越具体，故障时越有用。

## 59.29 发布和容量的关系

发布会消耗容量。

第 58 章讲滚动升级时提到，新 worker 要先 ready，老 worker 要 drain。

这意味着发布期间可能同时存在：

1. 旧 worker。
2. 新 worker。
3. draining worker。
4. warmup 中的 worker。

如果 GPU 资源没有 surge 空间，发布过程会挤占线上容量。

容量规划要明确：

```text
normal capacity: 正常服务容量
failure capacity: worker 或节点故障后的容量
release capacity: 滚动升级和灰度期间需要的额外容量
```

如果没有额外 GPU，可以采用更保守发布：

1. 低峰发布。
2. 小批次替换。
3. 先扩容再发布。
4. 降低灰度速度。
5. 发布期间限制低优先级流量。

发布策略和容量策略必须一起设计。

## 59.30 常见错误

错误一：只用 QPS 做容量规划。

```text
结果：长上下文和长输出请求打爆 KV cache 和 TTFT。
```

错误二：用 benchmark 极限吞吐直接算 GPU 数。

```text
结果：线上没有安全余量，轻微波动就过载。
```

错误三：只看平均输入长度。

```text
结果：P99 长 prompt 导致少量请求拖垮整体延迟。
```

错误四：忽略 KV cache 容量。

```text
结果：GPU utilization 看起来不高，但系统无法接更多并发。
```

错误五：SLO 不区分 workload。

```text
结果：短问答和长文档使用同一延迟目标，指标不可解释。
```

错误六：告警只看 5xx。

```text
结果：TTFT、TPOT 和队列已经严重退化时才发现。
```

错误七：没有 admission control。

```text
结果：过载时无限排队，最后所有请求一起超时。
```

错误八：成本只按满载 token 算。

```text
结果：忽略峰值冗余和低利用率，实际账单远高于预期。
```

错误九：不做故障演练。

```text
结果：预案在真实故障时不可执行。
```

错误十：发布不考虑容量。

```text
结果：升级过程中 worker drain 和 cold start 导致线上容量骤降。
```

## 59.31 面试高频问题

问题一：为什么 LLM serving 容量规划不能只看 QPS？

回答要点：因为不同请求的输入长度、输出长度和上下文长度差异很大，同样是 1 个请求，prefill、decode 和 KV cache 成本可能差几个数量级。应该同时看 request QPS、input token/s、output token/s、输入输出长度分布、并发数、KV cache 占用、TTFT 和 TPOT。

问题二：如何从 benchmark 估算需要多少张 GPU？

回答要点：先用接近真实业务的输入输出长度分布做 benchmark，得到单 worker 在 SLO 约束下的稳定吞吐，而不是极限吞吐。然后按线上峰值流量计算 worker 数，应用利用率目标和安全余量，再考虑 N+1 故障冗余、发布 surge、可用区分布，最后乘以每个 worker 的 tensor parallel GPU 数得到总 GPU 数。

问题三：KV cache 如何影响容量？

回答要点：每个 active sequence 都需要为历史 token 保存 key/value，KV cache 占用与层数、KV head 数、head dim、dtype 和 active token 数相关。长上下文和长输出会显著增加 KV cache 占用，限制并发请求数。即使 GPU 计算利用率不高，KV cache 不足也会导致 preemption、拒绝请求或 OOM。

问题四：LLM serving 应该定义哪些 SLO？

回答要点：至少包括 availability、5xx rate、timeout rate、TTFT、TPOT、E2E latency，并按 workload 和输入输出长度条件区分。streaming 场景要特别关注 TTFT 和 TPOT。还应该监控 queue time、KV cache pressure、preemption、worker health 等内部指标作为 SLO 退化的前置信号。

问题五：过载时应该怎么处理？

回答要点：不要无限排队。应该通过 admission control 基于队列长度、token 预算、KV cache 空闲量、租户配额和优先级判断是否接收请求。容量不足时可以限流、拒绝低优先级请求、降低 max_tokens、限制超长上下文、切到小模型或异步处理。可控拒绝比全局超时更好。

问题六：如何做故障演练？

回答要点：选择典型故障场景，例如 worker 退出、GPU OOM、NCCL hang、模型下载失败、KV cache 耗尽、长上下文突增、对象存储不可用等。每个演练要定义预期行为、监控告警、止血步骤和恢复验证。演练后更新 runbook，确保真实故障时可执行。

## 59.32 标准回答模板

如果面试官问“你会如何做一个 LLM serving 集群的容量规划和稳定性设计”，可以这样回答：

```text
我不会只用 QPS 来规划容量，因为 LLM 请求的成本主要由 input tokens、output tokens、上下文长度和并发决定。首先我会按模型、租户和 workload 收集线上或预估流量，统计 RPS、input token/s、output token/s，以及输入输出长度的 P50/P95/P99 分布。短问答、代码生成、长文档总结这类 workload 要分开建模。

然后我会用接近真实分布的压测数据得到单 worker 在 SLO 约束下的稳定容量，包括 request throughput、prefill token/s、decode token/s、TTFT、TPOT、KV cache 使用和 preemption，而不是只看极限吞吐。容量计算会按峰值流量除以单 worker 规划容量，并加入安全余量，比如只使用 benchmark 稳定容量的 60% 到 80%。还要考虑 N+1 故障冗余、可用区分布、滚动升级和灰度发布时的额外容量。

对 KV cache 我会单独建模，因为 active sequence 的历史 token 会持续占用显存。长上下文和长输出会限制并发，即使 GPU compute utilization 不高，也可能因为 KV cache pressure 导致 preemption 或 OOM。因此 dashboard 要展示 running/waiting requests、active tokens、KV cache free blocks、preemption rate、TTFT 和 TPOT。

SLO 上我会区分 workload 定义，例如短 chat 的 TTFT P95、TPOT P95、timeout rate，长上下文任务使用另一套目标。告警不能只看 5xx，还要看 queue time、KV cache pressure、worker restart、OOM、NCCL error 和 no-capacity。过载时通过 admission control、租户限流、max_tokens 限制、长上下文降级和小模型兜底来保护系统。

最后我会定期做故障演练，包括 worker 退出、GPU OOM、NCCL hang、模型加载失败、对象存储不可用、长上下文突增和新版本性能退化。每个演练都要验证监控告警、router 摘除、降级、回滚和 runbook 是否可执行。容量、成本、SLO 和故障演练要一起设计，不能分开看。
```

## 59.33 Capacity SLO Fault Drill 公式、容量门禁和可运行 demo

先把 workload 的峰值 token 压力写成：

```math
Q_{\mathrm{in}}=\sum_k \lambda_k X_k
```

```math
Q_{\mathrm{out}}=\sum_k \lambda_k Y_k
```

其中 `lambda_k` 是第 `k` 类请求的峰值请求率，`X_k` 和 `Y_k` 分别是规划输入、输出 token 数。

单 worker 稳定容量要乘以规划利用率：

```math
P_{\mathrm{worker}}^{\mathrm{plan}}=\rho P_{\mathrm{worker}}^{\mathrm{bench}}
```

worker 数可以从请求率、prefill、decode 和 KV 四个方向取最大值：

```math
N_{\mathrm{worker}}=\max(N_{\mathrm{req}},N_{\mathrm{prefill}},N_{\mathrm{decode}},N_{\mathrm{kv}})
```

KV active token 估算：

```math
A_{\mathrm{tok}}=\sum_k C_k(X_k+Y_k)
```

GPU 数量：

```math
N_{\mathrm{gpu}}=N_{\mathrm{worker}}T_{\mathrm{gpu}}
```

月度 error budget：

```math
B_{\mathrm{err}}=(1-A_{\mathrm{slo}})T_{\mathrm{month}}
```

小时成本粗估：

```math
C_{\mathrm{hour}}=N_{\mathrm{gpu}}C_{\mathrm{gpu}}+C_{\mathrm{store}}+C_{\mathrm{net}}+C_{\mathrm{obs}}
```

最终容量与演练门禁：

```math
G_{\mathrm{capslo}}=G_{\mathrm{profile}}G_{\mathrm{bench}}G_{\mathrm{kv}}G_{\mathrm{cost}}G_{\mathrm{slo}}G_{\mathrm{admit}}G_{\mathrm{drill}}G_{\mathrm{runbook}}
```

下面这个 0 依赖 demo 模拟一次最小容量、成本、SLO 和故障演练审计：

```python
from dataclasses import dataclass
from math import ceil


@dataclass
class WorkloadProfile:
    name: str
    peak_rps: float
    input_tokens_p95: int
    output_tokens_p95: int
    active_requests_p95: int
    priority: int


@dataclass
class WorkerBenchmark:
    stable_request_rps: float
    prefill_tokens_per_s: int
    decode_tokens_per_s: int
    kv_active_tokens: int
    gpus_per_worker: int
    utilization_target: float


@dataclass
class CostConfig:
    gpu_hour_price: float
    storage_hour: float
    network_hour: float
    observability_hour: float


@dataclass
class SLOTarget:
    availability: float
    ttft_p95_ms: int
    tpot_p95_ms: int
    timeout_rate_max: float
    window_hours: int


@dataclass
class FaultScenario:
    name: str
    lost_workers: int
    dependency_available: bool
    long_context_multiplier: float
    candidate_tpot_p95_ms: int
    runbook_steps: list[str]


class ToyCapacitySLOFaultDrillAuditor:
    def __init__(self, workloads, benchmark, cost, slo):
        self.workloads = workloads
        self.benchmark = benchmark
        self.cost = cost
        self.slo = slo

    def traffic_profile(self) -> dict:
        peak_rps = sum(w.peak_rps for w in self.workloads)
        input_tps = sum(w.peak_rps * w.input_tokens_p95 for w in self.workloads)
        output_tps = sum(w.peak_rps * w.output_tokens_p95 for w in self.workloads)
        active_tokens = sum(w.active_requests_p95 * (w.input_tokens_p95 + w.output_tokens_p95) for w in self.workloads)
        return {
            "peak_rps": round(peak_rps, 2),
            "input_tps": int(input_tps),
            "output_tps": int(output_tps),
            "active_tokens": int(active_tokens),
        }

    def capacity_plan(self) -> dict:
        traffic = self.traffic_profile()
        rho = self.benchmark.utilization_target
        req_workers = ceil(traffic["peak_rps"] / (self.benchmark.stable_request_rps * rho))
        prefill_workers = ceil(traffic["input_tps"] / (self.benchmark.prefill_tokens_per_s * rho))
        decode_workers = ceil(traffic["output_tps"] / (self.benchmark.decode_tokens_per_s * rho))
        kv_workers = ceil(traffic["active_tokens"] / (self.benchmark.kv_active_tokens * rho))
        base_workers = max(req_workers, prefill_workers, decode_workers, kv_workers)
        safe_workers = ceil(base_workers * 1.25)
        n_plus_one_workers = safe_workers + 1
        release_workers = n_plus_one_workers + 1
        return {
            "traffic": traffic,
            "required_by_axis": {
                "request": req_workers,
                "prefill": prefill_workers,
                "decode": decode_workers,
                "kv": kv_workers,
            },
            "base_workers": base_workers,
            "safe_workers": safe_workers,
            "n_plus_one_workers": n_plus_one_workers,
            "release_workers": release_workers,
            "planned_gpus": release_workers * self.benchmark.gpus_per_worker,
        }

    def cost_report(self, plan: dict) -> dict:
        gpu_hour = plan["planned_gpus"] * self.cost.gpu_hour_price
        total_hour = gpu_hour + self.cost.storage_hour + self.cost.network_hour + self.cost.observability_hour
        tokens_per_hour = (plan["traffic"]["input_tps"] + plan["traffic"]["output_tps"]) * 3600
        cost_per_million_tokens = total_hour / max(1, tokens_per_hour / 1_000_000)
        return {
            "gpu_hour_cost": round(gpu_hour, 2),
            "total_hour_cost": round(total_hour, 2),
            "cost_per_million_tokens": round(cost_per_million_tokens, 3),
        }

    def slo_report(self, observed: dict) -> dict:
        budget_minutes = (1 - self.slo.availability) * self.slo.window_hours * 60
        burned_minutes = observed["incident_minutes"] + observed["timeout_rate"] * self.slo.window_hours * 60
        return {
            "availability_ok": observed["availability"] >= self.slo.availability,
            "ttft_ok": observed["ttft_p95_ms"] <= self.slo.ttft_p95_ms,
            "tpot_ok": observed["tpot_p95_ms"] <= self.slo.tpot_p95_ms,
            "timeout_ok": observed["timeout_rate"] <= self.slo.timeout_rate_max,
            "error_budget_minutes": round(budget_minutes, 2),
            "burned_minutes": round(burned_minutes, 2),
            "burn_ratio": round(burned_minutes / max(1, budget_minutes), 3),
        }

    def admission_decision(self, plan: dict, incoming: WorkloadProfile) -> dict:
        projected_tokens = plan["traffic"]["active_tokens"] + incoming.active_requests_p95 * (
            incoming.input_tokens_p95 + incoming.output_tokens_p95
        )
        kv_limit = plan["release_workers"] * self.benchmark.kv_active_tokens * self.benchmark.utilization_target
        admitted = projected_tokens <= kv_limit or incoming.priority <= 1
        action = "admit" if admitted else "reject_or_degrade"
        return {"projected_active_tokens": int(projected_tokens), "kv_limit": int(kv_limit), "action": action}

    def fault_drill(self, plan: dict, scenarios: list[FaultScenario]) -> dict:
        rows = {}
        for scenario in scenarios:
            remaining_workers = plan["release_workers"] - scenario.lost_workers
            capacity_ok = remaining_workers >= plan["base_workers"]
            dependency_ok = scenario.dependency_available or "use_cached_model_artifacts" in scenario.runbook_steps
            long_context_ok = scenario.long_context_multiplier <= 1.5 or "cap_context_length" in scenario.runbook_steps
            release_ok = scenario.candidate_tpot_p95_ms <= self.slo.tpot_p95_ms or "rollback_revision" in scenario.runbook_steps
            runbook_ok = {"detect", "mitigate", "verify"}.issubset(set(scenario.runbook_steps))
            rows[scenario.name] = {
                "remaining_workers": remaining_workers,
                "capacity_ok": capacity_ok,
                "dependency_ok": dependency_ok,
                "long_context_ok": long_context_ok,
                "release_ok": release_ok,
                "runbook_ok": runbook_ok,
                "pass": capacity_ok and dependency_ok and long_context_ok and release_ok and runbook_ok,
            }
        return rows


workloads = [
    WorkloadProfile("short_chat", 80.0, 256, 96, 120, 1),
    WorkloadProfile("rag_summary", 12.0, 4096, 512, 36, 2),
    WorkloadProfile("code_gen", 18.0, 1024, 512, 54, 2),
]
benchmark = WorkerBenchmark(
    stable_request_rps=50.0,
    prefill_tokens_per_s=180000,
    decode_tokens_per_s=12000,
    kv_active_tokens=120000,
    gpus_per_worker=4,
    utilization_target=0.7,
)
cost = CostConfig(gpu_hour_price=2.8, storage_hour=18.0, network_hour=12.0, observability_hour=5.0)
slo = SLOTarget(availability=0.995, ttft_p95_ms=700, tpot_p95_ms=80, timeout_rate_max=0.01, window_hours=720)
auditor = ToyCapacitySLOFaultDrillAuditor(workloads, benchmark, cost, slo)

plan = auditor.capacity_plan()
cost_report = auditor.cost_report(plan)
slo_report = auditor.slo_report(
    {"availability": 0.997, "ttft_p95_ms": 640, "tpot_p95_ms": 72, "timeout_rate": 0.00005, "incident_minutes": 18}
)
incoming_long_context = WorkloadProfile("tenant_burst", 6.0, 12000, 1024, 48, 3)
admission = auditor.admission_decision(plan, incoming_long_context)
fault_rows = auditor.fault_drill(
    plan,
    [
        FaultScenario("worker_lost", 1, True, 1.0, 72, ["detect", "mitigate", "verify"]),
        FaultScenario("object_store_down", 0, False, 1.0, 72, ["detect", "use_cached_model_artifacts", "mitigate", "verify"]),
        FaultScenario("long_context_burst", 0, True, 2.0, 74, ["detect", "cap_context_length", "mitigate", "verify"]),
        FaultScenario("candidate_tpot_regression", 0, True, 1.0, 110, ["detect", "rollback_revision", "mitigate", "verify"]),
    ],
)

summary = {
    "traffic": plan["traffic"],
    "required_by_axis": plan["required_by_axis"],
    "base_workers": plan["base_workers"],
    "safe_workers": plan["safe_workers"],
    "n_plus_one_workers": plan["n_plus_one_workers"],
    "release_workers": plan["release_workers"],
    "planned_gpus": plan["planned_gpus"],
    "cost_report": cost_report,
    "slo_report": slo_report,
    "admission": admission,
    "fault_rows": fault_rows,
}
gates = {
    "profile_gate": summary["traffic"]["input_tps"] == 88064 and summary["traffic"]["output_tps"] == 23040,
    "benchmark_capacity_gate": summary["required_by_axis"] == {"request": 4, "prefill": 1, "decode": 3, "kv": 4},
    "kv_capacity_gate": summary["base_workers"] == 4 and summary["release_workers"] == 7,
    "cost_gate": summary["cost_report"]["total_hour_cost"] == 113.4,
    "slo_gate": all(summary["slo_report"][key] for key in ["availability_ok", "ttft_ok", "tpot_ok", "timeout_ok"])
    and summary["slo_report"]["burn_ratio"] < 0.1,
    "admission_gate": summary["admission"]["action"] == "reject_or_degrade",
    "fault_drill_gate": all(row["pass"] for row in summary["fault_rows"].values()),
    "runbook_gate": all(row["runbook_ok"] for row in summary["fault_rows"].values()),
}
gates["capacity_slo_fault_drill_gate"] = all(gates.values())

print("capacity_slo_fault_drill_summary=", summary)
print("capacity_slo_fault_drill_gates=", gates)
```

一次运行的核心输出类似：

```text
capacity_slo_fault_drill_summary= {'traffic': {'peak_rps': 110.0, 'input_tps': 88064, 'output_tps': 23040, 'active_tokens': 291072}, 'required_by_axis': {'request': 4, 'prefill': 1, 'decode': 3, 'kv': 4}, 'base_workers': 4, 'safe_workers': 5, 'n_plus_one_workers': 6, 'release_workers': 7, 'planned_gpus': 28, 'cost_report': {'gpu_hour_cost': 78.4, 'total_hour_cost': 113.4, 'cost_per_million_tokens': 0.284}, 'slo_report': {'availability_ok': True, 'ttft_ok': True, 'tpot_ok': True, 'timeout_ok': True, 'error_budget_minutes': 216.0, 'burned_minutes': 20.16, 'burn_ratio': 0.093}, 'admission': {'projected_active_tokens': 916224, 'kv_limit': 588000, 'action': 'reject_or_degrade'}, 'fault_rows': {'worker_lost': {'remaining_workers': 6, 'capacity_ok': True, 'dependency_ok': True, 'long_context_ok': True, 'release_ok': True, 'runbook_ok': True, 'pass': True}, 'object_store_down': {'remaining_workers': 7, 'capacity_ok': True, 'dependency_ok': True, 'long_context_ok': True, 'release_ok': True, 'runbook_ok': True, 'pass': True}, 'long_context_burst': {'remaining_workers': 7, 'capacity_ok': True, 'dependency_ok': True, 'long_context_ok': True, 'release_ok': True, 'runbook_ok': True, 'pass': True}, 'candidate_tpot_regression': {'remaining_workers': 7, 'capacity_ok': True, 'dependency_ok': True, 'long_context_ok': True, 'release_ok': True, 'runbook_ok': True, 'pass': True}}}
capacity_slo_fault_drill_gates= {'profile_gate': True, 'benchmark_capacity_gate': True, 'kv_capacity_gate': True, 'cost_gate': True, 'slo_gate': True, 'admission_gate': True, 'fault_drill_gate': True, 'runbook_gate': True, 'capacity_slo_fault_drill_gate': True}
```

这个 demo 证明了几个关键点：

1. 容量规划要同时看 request、prefill、decode 和 KV active tokens，不能只看 QPS。
2. benchmark 极限值要乘以规划利用率，再叠加安全余量、N+1 和发布容量。
3. KV cache 可能和 decode 一样成为主约束，长上下文 burst 要走 admission 或降级。
4. 成本要从 GPU 小时扩展到存储、网络和观测，并能按 token 粗略归因。
5. SLO 不是一句口号，必须能计算 error budget 和 burn ratio。
6. 故障演练要覆盖 worker、依赖系统、长上下文突增和新版本性能退化，并且每个场景都有 detect、mitigate、verify。

## 59.34 小练习

1. 为 benchmark 脚本增加 input token/s 和 output token/s 统计。
2. 为请求日志增加 input_tokens、output_tokens、tenant_id 和 model_revision。
3. 统计一组请求的输入长度 P50/P95/P99。
4. 统计一组请求的输出长度 P50/P95/P99。
5. 基于 benchmark 结果写一个 GPU 数量估算表。
6. 把规划利用率从 100% 改为 70%，比较 GPU 数变化。
7. 增加一个 KV cache bytes per token 的估算函数。
8. 在 dashboard 中展示 running requests 和 waiting requests。
9. 在 dashboard 中展示 KV cache free blocks。
10. 实现一个简单 admission control：队列过长时返回 overload。
11. 对超长输入请求返回明确错误，而不是进入 engine 后失败。
12. 为不同租户设置不同 QPS 和 token 配额。
13. 为短请求和长请求分别定义 TTFT SLO。
14. 写一个压测场景：短输入短输出。
15. 写一个压测场景：长输入长输出。
16. 写一个混合流量压测场景。
17. 模拟一个 worker 退出，观察 router 是否摘除。
18. 模拟 KV cache 压力升高，观察 preemption 和 TTFT。
19. 写一个 worker OOM 的故障演练 runbook。
20. 写一个新版本 TPOT 退化后的回滚 runbook。

## 59.35 本章总结

LLM serving 的容量规划不能只看 QPS。

真正重要的是 request QPS、input token/s、output token/s、输入输出长度分布、并发请求数和 KV cache 占用。

prefill 主要影响 TTFT，decode 主要影响 TPOT，KV cache 决定长上下文和高并发下能同时容纳多少 active sequence。

容量估算应该基于真实 workload 的 benchmark，并用稳定容量而不是极限吞吐。

生产规划要加入安全余量、N+1 故障冗余、发布容量和可用区分布。

成本估算不仅要看 GPU 小时，也要考虑利用率、空闲冗余、存储、网络、日志和多租户归因。

SLO 应该按 workload 和输入输出长度条件定义，streaming 场景要重点关注 TTFT 和 TPOT。

告警不能只看 5xx，还要覆盖 queue time、KV cache pressure、preemption、OOM、NCCL error、worker health 和依赖系统。

过载时应该通过 admission control、限流、降级和明确错误保护系统，而不是无限排队。

故障演练要覆盖 worker、GPU、模型加载、router、KV cache、依赖系统、长上下文突增和新版本性能退化。

到这里，一个 LLM serving engine 已经不仅能跑起来，也能被容量化、成本化、SLO 化和演练化。

下一章可以继续讨论：如何把这些工程能力沉淀成面试项目、作品集和系统设计表达。
