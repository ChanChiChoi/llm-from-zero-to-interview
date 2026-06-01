# 第 44 章 AI Infra 可观测性总览：metrics、logs、traces、events

前面第五部分讲完了数据、模型与实验平台。从本章开始进入第六部分：可观测性、可靠性与成本治理。

AI Infra 的系统链路很长：数据、训练、评估、推理、RAG、Agent、工具、GPU 集群、存储、网络和调度都可能出问题。没有可观测性，平台只能靠猜。

先记住一句话：

> AI Infra 可观测性的目标，不是收集更多指标，而是让训练、推理、数据、模型和平台问题能够被快速发现、定位、解释和复盘。

## 44.1 为什么 AI Infra 更需要可观测性

普通后端服务主要关注请求、延迟、错误率和资源使用。

AI Infra 还要额外关注：

1. GPU 利用率。
2. 显存和 KV cache。
3. 训练 loss。
4. gradient norm。
5. checkpoint。
6. 数据读取吞吐。
7. NCCL 通信。
8. 推理 TTFT 和 TPOT。
9. token 吞吐。
10. 模型版本。
11. prompt 版本。
12. eval 分数。
13. RAG 检索 trace。
14. tool call trace。
15. 成本。

AI 系统的问题常常跨越模型、数据、系统和业务多个层面。

## 44.2 可观测性的四类信号

本章重点讲四类信号：

1. Metrics。
2. Logs。
3. Traces。
4. Events。

Metrics 是数值指标，适合监控趋势和告警。

Logs 是文本或结构化日志，适合排查细节。

Traces 是一次请求或任务的链路，适合定位阶段耗时和调用关系。

Events 是状态变化和关键事件，适合理解系统发生了什么。

四者结合，才能形成完整可观测性。

## 44.3 Metrics 是什么

Metrics 是时间序列指标。

例如：

1. GPU utilization。
2. training loss。
3. QPS。
4. TTFT p95。
5. checkpoint save latency。
6. data loader throughput。
7. queue length。
8. error rate。
9. cost per hour。

Metrics 适合回答：

1. 系统现在是否健康？
2. 指标是否超过阈值？
3. 最近是否有趋势变化？
4. 哪个租户或模型消耗最多？

## 44.4 Logs 是什么

Logs 记录具体发生了什么。

例如：

1. 训练脚本输出。
2. runtime 错误堆栈。
3. 数据加载失败。
4. checkpoint 保存失败。
5. 模型加载日志。
6. 工具调用错误。
7. 权限拒绝原因。
8. 路由决策日志。

Logs 适合回答：

1. 具体错误是什么？
2. 哪个文件失败了？
3. 哪个参数不合法？
4. 哪个节点报错？

日志最好结构化，否则很难检索和聚合。

## 44.5 Traces 是什么

Trace 记录一次请求或任务经过哪些阶段，每个阶段耗时多少，调用了哪些组件。

推理请求 trace 可能包括：

```text
gateway -> auth -> router -> cache -> queue -> prefill -> decode -> safety -> response
```

训练任务 trace 可能包括：

```text
submit -> schedule -> image pull -> data load -> train step -> checkpoint -> eval -> finish
```

Trace 适合回答：

1. 慢在哪里？
2. 哪个组件失败？
3. 调用链路是什么？
4. 哪个模型或工具参与了这次请求？

## 44.6 Events 是什么

Events 是离散事件，通常表示状态变化或关键动作。

例如：

1. TrainingJob created。
2. TrainingJob scheduled。
3. Pod OOMKilled。
4. Checkpoint saved。
5. ModelVersion promoted。
6. Deployment rolled back。
7. Autoscaler scaled out。
8. Router degraded request。
9. Safety policy blocked output。
10. Dataset quality gate failed。

Events 能帮助复盘时间线。

当系统出问题时，你需要知道指标变化前后发生了哪些事件。

## 44.7 四类信号如何协作

假设推理 p99 突然升高。

Metrics 告诉你：p99 高了，queue length 也高了。

Logs 告诉你：某些 runtime 出现 KV cache allocation failed。

Traces 告诉你：请求主要慢在 queue 和 prefill。

Events 告诉你：10 分钟前某个大租户开始灰度新功能，流量突增。

单独看任何一种信号都不够。

可观测性的价值在于把它们关联起来。

## 44.8 训练可观测性

训练场景需要观察：

训练指标：

1. train loss。
2. eval loss。
3. learning rate。
4. gradient norm。
5. step time。
6. tokens/s。
7. samples/s。

系统指标：

1. GPU utilization。
2. GPU memory。
3. CPU utilization。
4. host memory。
5. disk I/O。
6. network bandwidth。
7. NCCL communication time。

平台指标：

1. queue time。
2. scheduling time。
3. checkpoint save time。
4. data load throughput。
5. retry count。
6. node failure count。

训练可观测性既要看模型收敛，也要看系统效率。

## 44.9 推理可观测性

推理场景需要观察：

请求指标：

1. QPS。
2. TTFT。
3. TPOT。
4. p95 / p99 latency。
5. error rate。
6. timeout rate。

token 指标：

1. input tokens/s。
2. output tokens/s。
3. average input tokens。
4. average output tokens。

runtime 指标：

1. queue length。
2. active sequences。
3. batch size。
4. KV cache usage。
5. GPU utilization。
6. GPU memory。

平台指标：

1. route decision。
2. cache hit rate。
3. fallback rate。
4. degradation rate。
5. admission reject rate。

推理可观测性要能把端到端延迟拆到 queue、prefill、decode、streaming 等阶段。

## 44.10 数据平台可观测性

数据平台需要观察：

1. ingestion 成功率。
2. 清洗任务成功率。
3. 样本数量变化。
4. token 数变化。
5. 重复率。
6. PII 命中率。
7. schema 错误率。
8. shard 损坏率。
9. 数据读取吞吐。
10. data loader waiting time。
11. dataset quality gate 状态。

数据问题常常在训练中表现为 loss 异常或效果退化。

所以数据质量指标要和训练结果关联起来。

## 44.11 模型和实验可观测性

模型和实验层需要观察：

1. experiment run 状态。
2. 指标曲线。
3. eval score。
4. eval failure cases。
5. model version 状态。
6. release gate 状态。
7. 灰度指标。
8. A/B 测试结果。
9. 回滚事件。

模型质量问题不一定表现为系统错误。

模型稳定返回 200，但回答质量下降，这也必须被观测。

## 44.12 RAG/Agent 可观测性

RAG/Agent 需要观察：

RAG 指标：

1. retrieval latency。
2. top-k hit rate。
3. empty result rate。
4. rerank latency。
5. citation correctness。
6. groundedness。

Agent 指标：

1. step count。
2. tool call count。
3. tool success rate。
4. tool latency。
5. loop detection count。
6. user confirmation count。
7. run success rate。
8. cost per run。

Trace：

1. retrieval trace。
2. prompt assembly trace。
3. tool call trace。
4. execution trace。

Agent 是多步系统，不看 trace 基本无法排查问题。

## 44.13 成本可观测性

AI Infra 成本高，成本本身也是可观测性的一部分。

需要观察：

1. GPU hour。
2. GPU utilization。
3. storage cost。
4. network cost。
5. training cost per run。
6. inference cost per 1k tokens。
7. cost per tenant。
8. cost per model。
9. cost per endpoint。
10. idle GPU cost。

如果只能看到总账单，无法做成本治理。

成本必须和模型、租户、任务、endpoint、run 关联。

## 44.14 统一 ID 体系

可观测性要靠统一 ID 串联。

常见 ID：

1. request ID。
2. trace ID。
3. tenant ID。
4. user ID。
5. training job ID。
6. experiment run ID。
7. model version ID。
8. dataset version ID。
9. checkpoint ID。
10. deployment ID。

如果日志、指标、trace、事件里的 ID 不一致，就很难关联。

统一 ID 是平台可观测性的基础。

## 44.15 标签体系

Metrics 要有合理标签。

常见标签：

1. cluster。
2. namespace。
3. tenant。
4. model。
5. model_version。
6. endpoint。
7. runtime。
8. gpu_type。
9. job_type。
10. priority。

标签过少，无法定位。

标签过多，会导致指标基数爆炸。

设计标签要平衡查询能力和存储成本。

## 44.16 高基数问题

高基数是 metrics 系统常见问题。

例如把 request ID 作为 metrics label，会导致时间序列数量爆炸。

高基数字段适合放在 logs 或 traces 中，不适合放在 metrics label 中。

Metrics label 应选择有限枚举字段，例如 model、tenant、endpoint、status。

高基数字段如 request ID、prompt hash、user ID，通常进入 trace 或日志。

## 44.17 告警设计

告警不是阈值越多越好。

好的告警应该：

1. 指向用户影响。
2. 可行动。
3. 有上下文。
4. 避免重复。
5. 有优先级。

常见告警：

1. 训练任务 hang。
2. GPU 利用率长期过低。
3. checkpoint 保存失败。
4. 推理 p99 超阈值。
5. timeout rate 上升。
6. KV cache 水位过高。
7. eval gate 失败。
8. cost 异常上升。

不要为每个小波动都告警，否则值班会被噪音淹没。

## 44.18 Dashboard 设计

不同角色需要不同 dashboard。

训练工程师关注：

1. loss。
2. tokens/s。
3. GPU。
4. data loader。
5. checkpoint。

推理工程师关注：

1. QPS。
2. TTFT。
3. TPOT。
4. queue。
5. KV cache。
6. runtime errors。

平台负责人关注：

1. 集群利用率。
2. 成本。
3. SLO。
4. 租户配额。
5. 发布状态。

不要试图用一个 dashboard 服务所有人。

## 44.19 日志脱敏

AI 系统日志可能包含敏感信息：

1. prompt。
2. 用户输入。
3. 检索文档。
4. 工具参数。
5. 模型输出。
6. 密钥。
7. 私有数据。

日志系统必须支持：

1. 脱敏。
2. 字段级权限。
3. TTL。
4. 加密。
5. 审计。

不能为了排障，把所有 prompt 和输出无保护地永久保存。

## 44.20 Trace 采样

Trace 全量保存成本高，尤其是 Agent 和 RAG。

常见策略：

1. 错误请求全量保存。
2. 慢请求全量保存。
3. 高成本请求全量保存。
4. 普通成功请求采样。
5. 敏感请求只保存结构化摘要。

采样策略要兼顾排障能力、成本和隐私。

## 44.21 可观测性平台架构

一个 AI Infra 可观测性平台可以这样设计：

```text
Collectors / SDKs / Agents
  -> Metrics Pipeline
  -> Logs Pipeline
  -> Traces Pipeline
  -> Events Pipeline
  -> Correlation / Indexing
  -> Alerting
  -> Dashboards
  -> Incident / Postmortem
```

数据来源包括训练任务、推理 runtime、K8s、GPU exporter、存储系统、网络系统、评估平台、RAG/Agent runtime 和成本系统。

## 44.22 常见误区

误区一：有 GPU 指标就够了。

GPU 指标只能说明资源状态，不能解释模型质量、数据问题、trace 链路和成本。

误区二：日志越多越好。

日志太多会增加成本和隐私风险，关键是结构化、可检索和有上下文。

误区三：只看平均延迟。

AI 在线服务更要看 p95/p99、TTFT、TPOT 和队列。

误区四：模型质量不属于可观测性。

模型质量、eval 分数、线上反馈和安全指标都应该进入可观测体系。

误区五：可观测性只服务故障排查。

它还服务性能优化、成本治理、发布决策和合规审计。

## 44.23 面试常见追问

问题一：AI Infra 可观测性和普通后端可观测性有什么不同？

可以回答：除了 QPS、延迟、错误率和资源，还要关注 GPU、显存、KV cache、training loss、tokens/s、TTFT、TPOT、数据质量、模型版本、eval 分数、RAG trace、tool trace 和成本。

问题二：Metrics、logs、traces、events 分别解决什么问题？

可以回答：metrics 用于趋势和告警，logs 用于细节排查，traces 用于链路和阶段耗时定位，events 用于状态变化和复盘时间线。

问题三：为什么 request ID 不适合做 metrics label？

可以回答：request ID 基数极高，会导致时间序列爆炸。它适合放在 trace 或日志里，而不是 metrics label。

问题四：如何设计推理服务的可观测性？

可以回答：记录 QPS、TTFT、TPOT、p99、input/output tokens/s、queue length、active sequences、KV cache、GPU、cache hit、fallback、degradation、error/timeout，并通过 trace 拆分 gateway、router、queue、prefill、decode 和 streaming。

## 44.24 小练习

1. AI Infra 为什么比普通后端更依赖可观测性？
2. Metrics、logs、traces、events 各适合回答什么问题？
3. 训练任务需要观察哪些模型指标和系统指标？
4. 推理服务为什么要同时看 TTFT 和 TPOT？
5. RAG/Agent 为什么必须保存 trace？
6. 什么是 metrics 高基数问题？
7. 日志脱敏为什么在 AI 系统中特别重要？
8. 如何设计一个推理 p99 延迟告警？

## 44.25 本章小结

本章讲了 AI Infra 可观测性总览。

你需要记住：

1. AI Infra 可观测性要覆盖训练、推理、数据、模型、RAG、Agent、成本和平台资源。
2. Metrics、logs、traces、events 各有作用，必须关联起来使用。
3. 训练可观测性要同时看 loss、tokens/s、GPU、通信、数据读取和 checkpoint。
4. 推理可观测性要关注 TTFT、TPOT、tokens/s、queue、KV cache、runtime 和路由。
5. RAG/Agent 可观测性离不开 retrieval trace、tool call trace 和 execution trace。
6. 可观测性还必须考虑成本、隐私、脱敏、高基数和告警噪音。

下一章我们会讲训练故障定位：loss 异常、hang、OOM、通信慢和 I/O 慢。
