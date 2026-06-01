# 第 58 章 设计一个 AI Infra 可观测性平台

上一章设计了模型仓库与发布系统。本章继续系统设计题：设计一个 AI Infra 可观测性平台。

AI Infra 的问题往往跨越训练、推理、数据、模型、RAG、Agent、GPU 集群、存储、网络和成本。如果没有统一可观测性平台，故障定位会变成到处翻日志、猜测指标、人工拼时间线。

先记住一句话：

> AI Infra 可观测性平台的核心，是把 metrics、logs、traces、events、cost 和 model quality signals 串起来，让问题能被发现、定位、解释、复盘和治理。

## 58.1 题目理解

面试题可能这样问：

1. 请设计一个 AI Infra 可观测性平台。
2. 如何监控训练、推理、RAG 和 Agent？
3. 如何定位 TTFT、TPOT、loss 异常、GPU 利用率低等问题？
4. 如何统一 metrics、logs、traces 和 events？

这类题既考后端可观测性，也考 AI 系统特有指标。

## 58.2 需求澄清

先问：

1. 观测对象包括哪些？训练、推理、数据、评估、RAG、Agent？
2. 集群规模多大？
3. 是否多租户？
4. 是否有 SLO 和告警体系？
5. 是否需要 trace 回放？
6. 日志是否包含敏感数据？
7. 是否需要成本可观测性？
8. 是否需要模型质量指标？
9. 是否和事故复盘系统联动？
10. 数据留存周期要求是什么？

AI Infra 可观测性不能只按普通微服务指标设计。

## 58.3 核心目标

平台目标：

1. 收集 metrics、logs、traces、events。
2. 覆盖训练、推理、数据、模型、RAG、Agent、集群和成本。
3. 支持统一 ID 关联。
4. 支持多租户隔离和权限。
5. 支持 dashboard 和告警。
6. 支持慢请求和失败任务诊断。
7. 支持 SLO 和错误预算。
8. 支持事故复盘和变更关联。
9. 支持成本可观测性。
10. 支持日志脱敏和留存治理。

## 58.4 总体架构

可以设计如下：

```text
Instrumentation SDK / Agents / Exporters
  -> Metrics Pipeline
  -> Logs Pipeline
  -> Traces Pipeline
  -> Events Pipeline
  -> Cost Pipeline
  -> Correlation Layer
  -> Query / Dashboard / Alerting
  -> Incident / Postmortem / SLO
```

数据来源包括：

1. Training platform。
2. Inference platform。
3. GPU cluster。
4. Kubernetes。
5. Runtime。
6. Data platform。
7. Eval platform。
8. RAG/Agent platform。
9. Cost system。
10. Release system。

## 58.5 Metrics Pipeline

Metrics pipeline 处理时间序列指标。

指标来源：

1. GPU exporter。
2. Kubernetes metrics。
3. training SDK。
4. inference runtime。
5. API Gateway。
6. storage system。
7. network system。
8. eval platform。

指标例子：

1. GPU utilization。
2. train loss。
3. tokens/s。
4. TTFT。
5. TPOT。
6. queue length。
7. KV cache usage。
8. error rate。
9. cost per 1k tokens。

Metrics 用于趋势、告警和 SLO。

## 58.6 Logs Pipeline

Logs pipeline 处理日志。

日志来源：

1. 训练脚本。
2. runtime。
3. API Gateway。
4. scheduler。
5. data loader。
6. checkpoint。
7. RAG retrieval。
8. tool call。
9. security audit。

日志要求：

1. 结构化。
2. 带 request ID / job ID。
3. 支持脱敏。
4. 支持采样。
5. 支持权限控制。
6. 支持 TTL。

日志不是越多越好，关键是可检索、可关联、可安全保存。

## 58.7 Traces Pipeline

Traces pipeline 记录链路。

推理 trace：

```text
gateway -> auth -> router -> cache -> queue -> prefill -> decode -> safety -> response
```

RAG trace：

```text
query rewrite -> retrieval -> filter -> rerank -> prompt assembly -> generation
```

Agent trace：

```text
plan -> model call -> tool call -> observation -> next step -> final answer
```

Trace 适合定位慢在哪、失败在哪、为什么路由到这个模型或工具。

## 58.8 Events Pipeline

Events 是状态变化。

事件包括：

1. TrainingJob submitted。
2. Job scheduled。
3. Pod failed。
4. Checkpoint saved。
5. Model deployed。
6. Canary started。
7. Autoscaler scaled out。
8. Router fallback triggered。
9. Dataset quality gate failed。
10. Security policy denied。

Events 用于构建事故时间线。

指标告诉你发生了异常，事件告诉你异常前后发生了什么。

## 58.9 Cost Pipeline

AI Infra 成本要进入可观测性平台。

成本数据包括：

1. GPU hours。
2. idle GPU hours。
3. failed job cost。
4. storage cost。
5. network cost。
6. inference cost per 1k tokens。
7. eval cost。
8. agent run cost。

成本维度：

1. tenant。
2. team。
3. project。
4. model。
5. endpoint。
6. training job。
7. experiment run。

没有成本可观测性，就无法做成本治理。

## 58.10 统一 ID 体系

可观测性平台必须有统一 ID。

常见 ID：

1. request ID。
2. trace ID。
3. tenant ID。
4. user ID。
5. model version ID。
6. endpoint ID。
7. training job ID。
8. experiment run ID。
9. dataset version ID。
10. deployment ID。

这些 ID 要贯穿 metrics、logs、traces 和 events。

否则无法从一个慢请求跳到对应日志、路由、模型版本和成本记录。

## 58.11 Correlation Layer

Correlation Layer 负责关联不同信号。

例如：

1. 从 p99 告警跳到慢请求 trace。
2. 从 trace 跳到 runtime 日志。
3. 从 model version 跳到发布事件。
4. 从 training job 跳到 checkpoint 和成本。
5. 从 RAG 答案跳到 retrieval trace。

这层是 AI Infra 可观测性平台的关键。

没有关联，数据只是散落的监控碎片。

## 58.12 训练可观测性设计

Training dashboard 应展示：

1. job status。
2. queue wait time。
3. loss curve。
4. learning rate。
5. gradient norm。
6. tokens/s。
7. step time breakdown。
8. GPU utilization。
9. data loader time。
10. NCCL communication。
11. checkpoint events。
12. failed/retry events。

训练问题要能拆到数据、计算、通信、checkpoint 和调度。

## 58.13 推理可观测性设计

Inference dashboard 应展示：

1. QPS。
2. TTFT。
3. TPOT。
4. p95/p99 latency。
5. input/output tokens/s。
6. queue length。
7. active sequences。
8. KV cache usage。
9. GPU utilization。
10. cache hit rate。
11. fallback rate。
12. error/timeout。
13. cost per 1k tokens。

同时支持按 model、tenant、endpoint、runtime、region 切分。

## 58.14 RAG/Agent 可观测性设计

RAG dashboard：

1. retrieval latency。
2. empty result rate。
3. top-k hit rate。
4. rerank latency。
5. citation availability。
6. groundedness 指标。

Agent dashboard：

1. run success rate。
2. step count。
3. tool call count。
4. tool success rate。
5. loop detection。
6. cost per run。
7. human confirmation count。

RAG/Agent 必须支持 trace 查看和回放。

## 58.15 数据和评估可观测性

数据平台：

1. ingestion 成功率。
2. 数据质量指标。
3. dataset build 状态。
4. shard 错误。
5. train/eval overlap。
6. data loader throughput。

评估平台：

1. eval job 状态。
2. samples/s。
3. eval score。
4. judge error rate。
5. report generation time。
6. quality gate pass rate。

模型质量信号也是可观测性的一部分。

## 58.16 告警系统

告警应支持：

1. 阈值告警。
2. SLO burn rate 告警。
3. 异常检测。
4. 事件驱动告警。
5. 组合条件告警。

示例：

```text
TTFT p95 > SLO 且 queue wait time 持续升高
```

比单纯 GPU 利用率高更有价值。

告警要有 owner、severity、runbook 和 dashboard 链接。

## 58.17 SLO 和错误预算

可观测性平台应支持 SLO：

1. 定义 SLI。
2. 配置 SLO。
3. 计算错误预算。
4. 展示 burn rate。
5. 触发告警。
6. 联动发布策略。

推理 SLO 可以包括成功率、TTFT、TPOT、p99。

训练 SLO 可以包括任务成功率、调度等待、checkpoint 恢复成功率。

## 58.18 Dashboard 设计

不同角色需要不同 dashboard。

平台 SRE：

1. SLO。
2. error rate。
3. p99。
4. cluster health。
5. incidents。

训练工程师：

1. loss。
2. GPU。
3. data loader。
4. checkpoint。

推理工程师：

1. TTFT。
2. TPOT。
3. queue。
4. KV cache。

业务负责人：

1. 成本。
2. 用户反馈。
3. 模型质量。
4. 发布状态。

不要用一个 dashboard 覆盖所有需求。

## 58.19 隐私和权限

可观测数据可能包含敏感信息。

要求：

1. 日志脱敏。
2. trace 字段级权限。
3. prompt 和输出 TTL。
4. 审计日志长期保存。
5. 敏感 trace 采样。
6. 多租户隔离。
7. 访问审计。

可观测性平台本身也是敏感数据平台。

## 58.20 高基数控制

Metrics label 不能无限增加。

高基数字段：

1. request ID。
2. user ID。
3. prompt hash。
4. document ID。

这些不适合作为 metrics label。

适合放在 trace 或 logs 中。

低基数字段如 model、endpoint、tenant、status 可以作为 label。

## 58.21 数据留存和成本

可观测性数据本身成本很高。

策略：

1. metrics 聚合降采样。
2. debug logs 短期保留。
3. audit logs 长期保留。
4. trace 采样。
5. 错误 trace 全量保留。
6. 敏感字段脱敏或不落盘。
7. 冷热存储分层。

留存策略要按数据价值和风险分层。

## 58.22 Incident 集成

可观测性平台应和事故管理集成。

告警触发后：

1. 创建 incident。
2. 关联 dashboard。
3. 关联最近变更。
4. 记录时间线。
5. 记录应急操作。
6. 生成复盘模板。
7. 跟踪行动项。

这样事故处理不是散落在聊天记录里。

## 58.23 变更关联

很多故障来自变更。

可观测性平台应展示：

1. 模型发布。
2. prompt 修改。
3. runtime 升级。
4. 路由策略变化。
5. 数据版本变化。
6. 调度策略变化。
7. 集群升级。

排障时第一问题是：最近变了什么？

## 58.24 核心 trade-off

Trade-off：

1. 数据完整性 vs 成本。
2. Trace 细节 vs 隐私。
3. 指标标签丰富度 vs 高基数风险。
4. 告警灵敏度 vs 噪音。
5. 长期留存 vs 存储成本。
6. 统一平台 vs 团队自定义灵活性。

可观测性平台设计必须考虑这些权衡。

## 58.25 面试回答模板

可以这样回答：

```text
我会把 AI Infra 可观测性平台设计成统一收集 metrics、logs、traces、events、cost 和 model quality signals 的系统。数据来源包括训练平台、推理平台、GPU 集群、K8s、runtime、数据平台、评估平台、RAG/Agent 平台和发布系统。

架构上通过 SDK、exporter 和 agent 采集数据，分别进入 metrics、logs、traces、events 和 cost pipeline。平台通过统一 request ID、job ID、model version、tenant、deployment 等 ID 做 correlation，支持从告警跳到 trace、日志、事件、发布记录和成本记录。

训练侧观察 loss、tokens/s、GPU、data loader、NCCL、checkpoint；推理侧观察 TTFT、TPOT、p99、queue、KV cache、tokens/s、runtime、cache hit 和成本；RAG/Agent 侧观察 retrieval trace、tool call trace、execution trace 和 cost per run。

平台提供 dashboard、SLO、burn rate 告警、incident 集成、变更关联和复盘支持，同时通过脱敏、权限、采样、TTL 和高基数控制管理成本和隐私风险。
```

## 58.26 常见扣分点

扣分点一：只讲 Prometheus 和 Grafana。

问题：没有覆盖 logs、traces、events、AI 特有指标和关联层。

扣分点二：不提 TTFT、TPOT、KV cache。

问题：不理解推理可观测性关键指标。

扣分点三：不提训练 loss 和 data loader。

问题：训练平台可观测性不完整。

扣分点四：不提隐私和脱敏。

问题：AI 日志和 trace 可能包含敏感 prompt、文档和工具结果。

扣分点五：不提高基数。

问题：metrics 系统容易被 request ID、user ID 等高基数字段打爆。

## 58.27 小练习

1. AI Infra 可观测性平台和普通微服务监控有什么不同？
2. 为什么需要统一 request ID 和 training job ID？
3. 推理平台 dashboard 应展示哪些关键指标？
4. RAG trace 应该包含哪些阶段？
5. 什么字段适合作为 metrics label，什么字段不适合？
6. 如何设计 TTFT SLO 告警？
7. 可观测性数据如何做留存分层？
8. 如何把变更记录和故障排查关联起来？

## 58.28 本章小结

本章系统设计了一个 AI Infra 可观测性平台。

你需要记住：

1. AI Infra 可观测性要覆盖 metrics、logs、traces、events、cost 和 model quality signals。
2. 训练、推理、数据、评估、RAG、Agent 和成本都有专属指标和 trace。
3. 统一 ID 和 correlation layer 是从告警定位根因的关键。
4. SLO、错误预算、告警、incident 和变更关联让可观测性进入生产治理闭环。
5. 隐私、脱敏、采样、留存和高基数控制是 AI 可观测性平台不可忽视的工程问题。

下一章我们会整理 AI Infra 面试高频问题与标准回答。
