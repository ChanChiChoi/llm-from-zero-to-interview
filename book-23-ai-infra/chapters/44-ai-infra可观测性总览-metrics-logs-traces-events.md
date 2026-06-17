# 第 44 章 AI Infra 可观测性总览：metrics、logs、traces、events

前面第五部分讲完了数据、模型与实验平台。从本章开始进入第六部分：可观测性、可靠性与成本治理。

AI Infra 的系统链路很长：数据、训练、评估、推理、RAG、Agent、工具、GPU 集群、存储、网络和调度都可能出问题。没有可观测性，平台只能靠猜。

先记住一句话：

> AI Infra 可观测性的目标，不是收集更多指标，而是让训练、推理、数据、模型和平台问题能够被快速发现、定位、解释和复盘。

## 44.0 本讲资料边界与第二轮精修口径

本章按通用可观测性平台抽象来写，不绑定 Prometheus、Grafana、OpenTelemetry Collector、Datadog、Loki、Jaeger、Tempo、云厂商监控或内部平台实现。资料校准时，主要参考 OpenTelemetry 对 traces、metrics、logs、baggage、events / profiles 的信号划分和 metric instrument / aggregation / attribute 口径，参考 Prometheus 对 time series、metric name、labels 和 label 变化生成新时间序列的模型，参考 Google SRE Book 对 SLI、SLO、SLA、error budget 和可行动监控的定义，也结合 AI Infra 前文中的训练、推理、RAG、Agent、评估、成本和安全治理场景。

第二轮精修只做三件事：

1. 把 metrics、logs、traces、events 和 cost signal 变成可审计对象，而不是概念解释。
2. 补齐 SLO、error budget、p95 / p99、标签基数、trace 覆盖、日志脱敏、告警噪音和根因定位公式。
3. 增加一个 0 依赖 Python demo，用 toy telemetry cases 检查 AI Infra 可观测性平台是否真的可排障、可告警、可复盘。

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

一次 AI Infra 运行的可观测对象可以抽象成：

$$
O_i=(m_i,l_i,t_i,e_i,s_i,c_i,a_i,d_i)
$$

其中 `m_i` 是 metrics，`l_i` 是 logs，`t_i` 是 traces，`e_i` 是 events，`s_i` 是 SLO / error budget，`c_i` 是 cost signal，`a_i` 是 alert / dashboard，`d_i` 是 data / model / artifact 版本上下文。

整体覆盖率：

$$
C_{\mathrm{obs}}=\frac{1}{N}\sum_{i=1}^{N}I(m_i,l_i,t_i,e_i,s_i,c_i,a_i,d_i\ \mathrm{present})
$$

这说明可观测性不是“有监控页面”，而是每类关键任务都要有能互相关联的信号。

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

四类信号的关联可以写成：

$$
R_i=(\mathrm{trace\_id}_i,\mathrm{request\_id}_i,\mathrm{tenant}_i,\mathrm{model}_i,\mathrm{run}_i,\mathrm{time}_i)
$$

如果 metrics、logs、traces 和 events 里没有共享的 `R_i`，事故复盘就只能靠人工猜测时间线。

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

一个时间序列可以写成：

$$
x_{m,L}(t)
$$

其中 `m` 是 metric name，`L` 是 label 集合，`t` 是时间。窗口均值：

$$
\bar{x}_{[a,b]}=\frac{1}{b-a}\int_a^b x(t)\,dt
$$

p 分位延迟可以写成：

$$
Q_p=\inf\{q:F(q)\ge p\}
$$

AI 推理里平均延迟不够，通常要同时看 `Q_0.95`、`Q_0.99`、TTFT 和 TPOT。

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

结构化日志覆盖率：

$$
C_{\mathrm{log}}=\frac{\sum_i I(\mathrm{ts}_i,\mathrm{level}_i,\mathrm{service}_i,\mathrm{trace\_id}_i,\mathrm{event}_i,\mathrm{status}_i\ \mathrm{present})}{N_{\mathrm{log}}}
$$

日志脱敏覆盖率：

$$
C_{\mathrm{redact}}=\frac{N_{\mathrm{redacted\ fields}}}{N_{\mathrm{sensitive\ fields}}}
$$

AI 系统里，prompt、检索片段、tool 参数和输出经常含敏感信息，不能只追求“日志越全越好”。

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

端到端推理延迟可以拆成：

$$
T_{\mathrm{e2e}}=T_{\mathrm{gateway}}+T_{\mathrm{auth}}+T_{\mathrm{router}}+T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{safety}}+T_{\mathrm{stream}}
$$

Trace span 覆盖率：

$$
C_{\mathrm{span}}=\frac{|S_{\mathrm{observed}}\cap S_{\mathrm{expected}}|}{|S_{\mathrm{expected}}|}
$$

如果没有 `queue`、`prefill`、`decode` 这些 span，TTFT / TPOT 的退化就只能靠猜。

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

事件时间线完整率：

$$
C_{\mathrm{event}}=\frac{\sum_i I(\mathrm{event\_type}_i,\mathrm{entity}_i,\mathrm{version}_i,\mathrm{actor}_i,\mathrm{ts}_i\ \mathrm{present})}{N_{\mathrm{event}}}
$$

事件要能和指标突变对齐，比如发布、回滚、扩容、降级、质量门禁失败和数据版本切换。

## 44.7 四类信号如何协作

假设推理 p99 突然升高。

Metrics 告诉你：p99 高了，queue length 也高了。

Logs 告诉你：某些 runtime 出现 KV cache allocation failed。

Traces 告诉你：请求主要慢在 queue 和 prefill。

Events 告诉你：10 分钟前某个大租户开始灰度新功能，流量突增。

单独看任何一种信号都不够。

可观测性的价值在于把它们关联起来。

信号关联率：

$$
C_{\mathrm{corr}}=\frac{N_{\mathrm{records\ with\ trace\_or\ run\_id}}}{N_{\mathrm{records}}}
$$

定位根因时，通常先用 metrics 发现异常，再用 events 找时间点，用 traces 定位阶段，用 logs 看细节，用版本和成本信号解释影响面。

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

训练效率可以用 token 吞吐和 GPU 利用率联合衡量：

$$
E_{\mathrm{train}}=\frac{\mathrm{tokens/s}}{N_{\mathrm{gpu}}\cdot U_{\mathrm{gpu}}}
$$

其中 `U_gpu` 是平均 GPU 利用率。`E_train` 下降但 loss 正常，通常是系统瓶颈；loss 异常但系统指标正常，通常要回到数据、配置或优化器排查。

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

推理 SLO 可以写成：

$$
G_{\mathrm{infer}}=I(Q_{0.99}(T_{\mathrm{e2e}})\le S_{\mathrm{e2e}})\cdot I(Q_{0.95}(T_{\mathrm{ttft}})\le S_{\mathrm{ttft}})\cdot I(Q_{0.95}(T_{\mathrm{tpot}})\le S_{\mathrm{tpot}})\cdot I(R_{\mathrm{err}}\le \epsilon)
$$

这比只看 QPS 更接近用户体验，因为用户会先感知 TTFT，再感知流式输出间隔。

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

数据质量异常率：

$$
R_{\mathrm{data\_bad}}=\frac{N_{\mathrm{schema\ fail}}+N_{\mathrm{pii\ hit}}+N_{\mathrm{corrupt}}+N_{\mathrm{quality\ fail}}}{N_{\mathrm{sample}}}
$$

数据平台可观测性要能把 `R_data_bad`、读取吞吐、data wait 和训练 loss / eval 退化连起来。

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

模型质量监控可以写成：

$$
\Delta Q=Q_{\mathrm{online}}-Q_{\mathrm{baseline}}
$$

当 `\Delta Q` 在关键切片上低于阈值时，即使错误率为 0，也应该触发质量告警或发布门禁。

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

RAG / Agent 可观测性门禁：

$$
G_{\mathrm{rag\_agent\_obs}}=I(C_{\mathrm{retr\_trace}}\ge \alpha)\cdot I(C_{\mathrm{tool\_trace}}\ge \beta)\cdot I(C_{\mathrm{exec\_trace}}\ge \gamma)\cdot I(R_{\mathrm{loop}}\le \lambda)
$$

其中 `R_loop` 是循环或重复工具调用异常率。Agent 成功率高但 trace 缺失，仍然不可上线。

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

单位 token 成本：

$$
K_{1k}=\frac{1000\cdot K_{\mathrm{total}}}{N_{\mathrm{input\ token}}+N_{\mathrm{output\ token}}}
$$

闲置 GPU 成本：

$$
K_{\mathrm{idle}}=H_{\mathrm{gpu}}\cdot p_{\mathrm{gpu}}\cdot (1-U_{\mathrm{gpu}})
$$

成本可观测性要能定位“哪个模型、租户、endpoint、run 或工具”造成成本变化。

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

ID 覆盖率：

$$
C_{\mathrm{id}}=\frac{\sum_i I(\mathrm{trace\_id}_i\lor \mathrm{run\_id}_i\lor \mathrm{job\_id}_i)}{N_{\mathrm{record}}}
$$

训练、评估、推理、RAG 和 Agent 不一定都有 request ID，但至少要有可关联的 job / run / trace / model / dataset 版本 ID。

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

时间序列数量近似为：

$$
N_{\mathrm{series}}\approx \sum_{m\in M}\prod_{\ell\in L_m}|V_\ell|
$$

`V_l` 是某个 label 的取值集合。把 `request_id`、`user_id`、`prompt_hash` 放进 metrics label，会让 `N_series` 爆炸。

## 44.16 高基数问题

高基数是 metrics 系统常见问题。

例如把 request ID 作为 metrics label，会导致时间序列数量爆炸。

高基数字段适合放在 logs 或 traces 中，不适合放在 metrics label 中。

Metrics label 应选择有限枚举字段，例如 model、tenant、endpoint、status。

高基数字段如 request ID、prompt hash、user ID，通常进入 trace 或日志。

标签基数门禁：

$$
G_{\mathrm{card}}=I(N_{\mathrm{series}}\le B_{\mathrm{series}})\cdot I(L_{\mathrm{high\ card}}\cap L_{\mathrm{metric}}=\varnothing)
$$

其中 `B_series` 是序列预算，`L_high_card` 是高基数字段集合。

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

告警可行动率：

$$
A_{\mathrm{action}}=\frac{N_{\mathrm{actionable\ alerts}}}{N_{\mathrm{alerts}}}
$$

告警噪音率：

$$
R_{\mathrm{noise}}=\frac{N_{\mathrm{duplicate}}+N_{\mathrm{non\ action}}}{N_{\mathrm{alerts}}}
$$

好的告警要能指向 runbook、owner、影响面和可能根因，而不是只告诉你某条曲线抖了一下。

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

日志与 trace 隐私门禁：

$$
G_{\mathrm{privacy}}=I(C_{\mathrm{redact}}\ge \alpha)\cdot I(\mathrm{ttl}\le \tau_{\max})\cdot I(\mathrm{access\ scoped})\cdot I(\mathrm{audit\ enabled})
$$

## 44.20 Trace 采样

Trace 全量保存成本高，尤其是 Agent 和 RAG。

常见策略：

1. 错误请求全量保存。
2. 慢请求全量保存。
3. 高成本请求全量保存。
4. 普通成功请求采样。
5. 敏感请求只保存结构化摘要。

采样策略要兼顾排障能力、成本和隐私。

Trace 保存概率可以写成：

$$
p_{\mathrm{trace}}(r)=
\begin{cases}
1, & r\in R_{\mathrm{error}}\cup R_{\mathrm{slow}}\cup R_{\mathrm{high\ cost}} \\
p_s, & r\in R_{\mathrm{success}} \\
p_{\mathrm{summary}}, & r\in R_{\mathrm{sensitive}}
\end{cases}
$$

这和上一章 RAG / Agent trace 的隐私策略一致：错误、慢请求和高成本 run 优先保留，普通成功请求采样，敏感请求保留摘要。

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

## 44.22 AI Infra 可观测性审计指标和最小 demo

把本章落到平台验收时，可以用 16 个门禁：

1. Signal Inventory Coverage：metrics、logs、traces、events、costs、alerts、dashboards、versions 是否都有入口。
2. Metric Contract Completeness：metric name、unit、labels、aggregation、owner、SLO / threshold 是否完整。
3. SLO Error Budget Readiness：可用性 SLI、SLO 目标、错误预算、burn rate 是否可计算。
4. Latency Quantile Guard：p95 / p99、TTFT、TPOT 是否有分位指标和阈值。
5. Trace Span Coverage：关键链路 span 是否覆盖 gateway、auth、router、queue、prefill、decode、safety、response。
6. Log Event Structure：日志和事件是否结构化，有时间、级别、服务、trace id、状态、事件类型、actor 和版本。
7. Correlation ID Coverage：metrics、logs、traces、events、costs 是否能通过 trace / run / job / model / dataset ID 关联。
8. Cardinality Budget Control：metrics label 是否避免 request ID、user ID、prompt hash 等高基数字段。
9. Training Observability：loss、learning rate、gradient norm、tokens/s、GPU、NCCL、data wait、checkpoint 是否完整。
10. Inference Observability：QPS、TTFT、TPOT、error、timeout、queue、KV cache、GPU、cache、fallback、degrade 是否完整。
11. Data Quality Observability：ingestion、schema、PII、duplicate、corrupt shard、quality gate、data loader wait 是否完整。
12. RAG Agent Observability：retrieval trace、prompt assembly trace、tool call trace、execution trace、loop rate 和 groundedness 是否完整。
13. Cost Observability：GPU、storage、network、LLM token、tool、trace 成本是否能按 tenant / model / endpoint / run 归因。
14. Privacy Redaction Retention：prompt、检索片段、tool 参数、输出和日志是否有脱敏、TTL、访问控制和审计。
15. Alert Actionability：告警是否有 owner、runbook、severity、影响面、去重和可行动上下文。
16. Observability Platform Gate：最终是否有 owner、dashboard、incident process、postmortem、回滚和 P0 风险阻断。

综合门禁：

$$
G_{\mathrm{observability}}=\prod_{j=1}^{16}G_j
$$

下面是一个 0 依赖 demo，用 toy telemetry cases 检查一个 AI Infra 可观测性平台是否只是“收集了一堆数据”，还是能真正发现、定位、解释和复盘问题。

```python
from copy import deepcopy


class MiniAIInfraObservabilityAudit:
    GATES = [
        "signal_inventory_coverage",
        "metric_contract_completeness",
        "slo_error_budget_readiness",
        "latency_quantile_guard",
        "trace_span_coverage",
        "log_event_structure",
        "correlation_id_coverage",
        "cardinality_budget_control",
        "training_observability",
        "inference_observability",
        "data_quality_observability",
        "rag_agent_observability",
        "cost_observability",
        "privacy_redaction_retention",
        "alert_actionability",
        "observability_platform_gate",
    ]

    METRIC_FIELDS = ["name", "unit", "labels", "aggregation", "owner", "value"]
    LOG_FIELDS = ["ts", "level", "service", "trace_id", "event", "status", "redacted"]
    EVENT_FIELDS = ["ts", "event_type", "entity", "version", "actor"]
    REQUIRED_TRAINING = [
        "train_loss",
        "eval_loss",
        "learning_rate",
        "gradient_norm",
        "tokens_per_s",
        "gpu_util",
        "nccl_time_ratio",
        "data_wait_ratio",
        "checkpoint_failures",
    ]
    REQUIRED_INFERENCE = [
        "qps",
        "ttft_p95_ms",
        "tpot_p95_ms",
        "error_rate",
        "timeout_rate",
        "queue_p95_ms",
        "kv_cache_pressure",
        "gpu_util",
        "cache_hit_rate",
        "fallback_rate",
        "degrade_rate",
    ]
    REQUIRED_DATA = [
        "ingestion_success_rate",
        "schema_error_rate",
        "pii_hit_rate",
        "duplicate_rate",
        "corrupt_shards",
        "quality_gate_pass",
        "data_loader_wait_ratio",
    ]

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    def signal_inventory_coverage(self, case):
        required = {"metrics", "logs", "traces", "events", "costs", "alerts", "dashboards", "versions"}
        return required.issubset(set(case.get("signals", [])))

    def metric_contract_completeness(self, case):
        metrics = case.get("metrics", [])
        return bool(metrics) and all(
            self.coverage(metric, self.METRIC_FIELDS) == 1.0
            and isinstance(metric.get("labels"), dict)
            for metric in metrics
        )

    def slo_error_budget_readiness(self, case):
        slo = case.get("slo", {})
        total = slo.get("requests_total", 0)
        good = slo.get("good_requests", 0)
        target = slo.get("availability_target", 1.0)
        if total <= 0 or target >= 1.0:
            return False
        availability = good / total
        budget = 1 - target
        used = 1 - availability
        return availability >= target and (budget - used) >= 0 and "window_minutes" in slo

    def latency_quantile_guard(self, case):
        latency = case.get("latency", {})
        return (
            latency.get("e2e_p99_ms", 999999) <= latency.get("e2e_slo_ms", 0)
            and latency.get("ttft_p95_ms", 999999) <= latency.get("ttft_slo_ms", 0)
            and latency.get("tpot_p95_ms", 999999) <= latency.get("tpot_slo_ms", 0)
        )

    def trace_span_coverage(self, case):
        traces = case.get("traces", {})
        expected = set(traces.get("expected_spans", []))
        observed = set(traces.get("observed_spans", []))
        if not expected:
            return False
        return len(expected & observed) / len(expected) >= 0.95

    def log_event_structure(self, case):
        logs = case.get("logs", [])
        events = case.get("events", [])
        logs_ok = logs and all(self.coverage(log, self.LOG_FIELDS) == 1.0 for log in logs)
        events_ok = events and all(self.coverage(event, self.EVENT_FIELDS) == 1.0 for event in events)
        return bool(logs_ok and events_ok)

    def correlation_id_coverage(self, case):
        corr = case.get("correlation", {})
        total = corr.get("records_total", 0)
        with_id = corr.get("records_with_id", 0)
        return total > 0 and with_id / total >= 0.95

    def cardinality_budget_control(self, case):
        labels = case.get("labels", {})
        metric_count = labels.get("metric_count", 0)
        values = labels.get("label_value_counts", {})
        series = metric_count
        for count in values.values():
            series *= count
        return (
            series <= labels.get("series_budget", 0)
            and not labels.get("high_cardinality_metric_labels", [])
        )

    def training_observability(self, case):
        training = case.get("training", {})
        return (
            self.coverage(training, self.REQUIRED_TRAINING) == 1.0
            and training["data_wait_ratio"] <= 0.15
            and training["checkpoint_failures"] == 0
        )

    def inference_observability(self, case):
        inference = case.get("inference", {})
        return (
            self.coverage(inference, self.REQUIRED_INFERENCE) == 1.0
            and inference["error_rate"] <= 0.01
            and inference["timeout_rate"] <= 0.005
            and inference["kv_cache_pressure"] <= 0.85
        )

    def data_quality_observability(self, case):
        data = case.get("data_quality", {})
        return (
            self.coverage(data, self.REQUIRED_DATA) == 1.0
            and data["schema_error_rate"] <= 0.01
            and data["pii_hit_rate"] <= 0.001
            and data["corrupt_shards"] == 0
            and data["quality_gate_pass"] is True
        )

    def rag_agent_observability(self, case):
        rag_agent = case.get("rag_agent", {})
        return (
            rag_agent.get("retrieval_trace_coverage", 0.0) >= 0.95
            and rag_agent.get("tool_trace_coverage", 0.0) >= 0.95
            and rag_agent.get("execution_trace_coverage", 0.0) >= 0.95
            and rag_agent.get("loop_rate", 1.0) <= 0.05
            and rag_agent.get("groundedness", 0.0) >= 0.8
        )

    def cost_observability(self, case):
        cost = case.get("cost", {})
        if cost.get("total_tokens", 0) <= 0:
            return False
        cost_per_1k = 1000 * cost["total_cost_usd"] / cost["total_tokens"]
        return (
            cost.get("by_tenant") is True
            and cost.get("by_model") is True
            and cost.get("by_endpoint") is True
            and cost.get("by_run") is True
            and cost_per_1k <= cost.get("max_cost_per_1k_usd", 9999)
        )

    def privacy_redaction_retention(self, case):
        privacy = case.get("privacy", {})
        total = privacy.get("sensitive_fields_total", 0)
        redacted = privacy.get("redacted_fields", 0)
        if total <= 0:
            return False
        return (
            redacted / total >= 0.95
            and privacy.get("ttl_days", 9999) <= 90
            and privacy.get("access_scoped") is True
            and privacy.get("audit_enabled") is True
        )

    def alert_actionability(self, case):
        alerts = case.get("alerts", [])
        if not alerts:
            return False
        actionable = [alert for alert in alerts if alert.get("actionable")]
        duplicates = [alert for alert in alerts if alert.get("duplicate")]
        metadata_ok = all(alert.get("owner") and alert.get("runbook") and alert.get("severity") for alert in alerts)
        return len(actionable) / len(alerts) >= 0.8 and len(duplicates) / len(alerts) <= 0.2 and metadata_ok

    def observability_platform_gate(self, case):
        gate = case.get("platform_gate", {})
        return (
            gate.get("enabled") is True
            and bool(gate.get("owner"))
            and gate.get("dashboards_ready") is True
            and gate.get("incident_process_ready") is True
            and gate.get("p0_open") is False
        )

    def audit_case(self, case):
        return {gate: getattr(self, gate)(case) for gate in self.GATES}

    def run_all(self, cases):
        results = {case["case_id"]: self.audit_case(case) for case in cases}
        metrics = {}
        for gate in self.GATES:
            passed = sum(1 for result in results.values() if result[gate])
            metrics[gate] = round(passed / len(cases), 3)
        failed_cases = [
            case_id
            for case_id, result in results.items()
            if not all(result.values())
        ]
        failed_gates = [
            gate
            for gate in self.GATES
            if any(not result[gate] for result in results.values())
        ]
        return {
            "metrics": metrics,
            "hard_blocker_count": len(failed_cases),
            "failed_cases": failed_cases,
            "failed_gates": failed_gates,
            "observability_gate_pass": metrics["observability_platform_gate"] == 1.0,
        }

    def example_outputs(self, case):
        slo = case["slo"]
        availability = slo["good_requests"] / slo["requests_total"]
        budget = 1 - slo["availability_target"]
        used = 1 - availability
        labels = case["labels"]
        series = labels["metric_count"]
        for count in labels["label_value_counts"].values():
            series *= count
        cost = case["cost"]
        privacy = case["privacy"]
        logs = case["logs"]
        events = case["events"]
        traces = case["traces"]
        span_coverage = len(set(traces["expected_spans"]) & set(traces["observed_spans"])) / len(traces["expected_spans"])
        return {
            "signal_coverage": round(len(case["signals"]) / 8, 3),
            "metric_contract_coverage": round(
                sum(self.coverage(metric, self.METRIC_FIELDS) for metric in case["metrics"])
                / len(case["metrics"]),
                3,
            ),
            "availability": round(availability, 4),
            "error_budget_remaining": round((budget - used) / budget, 3),
            "latency_p99_ms": case["latency"]["e2e_p99_ms"],
            "trace_span_coverage": round(span_coverage, 3),
            "log_structure_coverage": round(
                sum(self.coverage(log, self.LOG_FIELDS) for log in logs) / len(logs),
                3,
            ),
            "event_timeline_coverage": round(
                sum(self.coverage(event, self.EVENT_FIELDS) for event in events) / len(events),
                3,
            ),
            "correlation_id_coverage": round(case["correlation"]["records_with_id"] / case["correlation"]["records_total"], 3),
            "cardinality_estimate": series,
            "training_tokens_per_s": case["training"]["tokens_per_s"],
            "inference_ttft_p95_ms": case["inference"]["ttft_p95_ms"],
            "rag_agent_trace_ready": self.rag_agent_observability(case),
            "cost_per_1k_tokens_usd": round(1000 * cost["total_cost_usd"] / cost["total_tokens"], 3),
            "redaction_coverage": round(privacy["redacted_fields"] / privacy["sensitive_fields_total"], 3),
            "actionable_alerts": sum(1 for alert in case["alerts"] if alert["actionable"]),
        }


def build_good_case():
    return {
        "case_id": "production_ready",
        "signals": ["metrics", "logs", "traces", "events", "costs", "alerts", "dashboards", "versions"],
        "metrics": [
            {
                "name": "inference_e2e_latency_ms",
                "unit": "ms",
                "labels": {"tenant": "acme", "model": "chat", "endpoint": "chat", "status": "ok"},
                "aggregation": "histogram",
                "owner": "serving",
                "value": 890,
            },
            {
                "name": "training_tokens_per_s",
                "unit": "tokens/s",
                "labels": {"cluster": "train-a", "job_type": "pretrain", "gpu_type": "h100"},
                "aggregation": "gauge",
                "owner": "training",
                "value": 178000,
            },
            {
                "name": "rag_groundedness",
                "unit": "ratio",
                "labels": {"tenant": "acme", "app": "support"},
                "aggregation": "gauge",
                "owner": "rag-platform",
                "value": 0.88,
            },
        ],
        "slo": {
            "requests_total": 10000,
            "good_requests": 9970,
            "availability_target": 0.99,
            "window_minutes": 60,
        },
        "latency": {
            "e2e_p99_ms": 890,
            "e2e_slo_ms": 1200,
            "ttft_p95_ms": 620,
            "ttft_slo_ms": 800,
            "tpot_p95_ms": 45,
            "tpot_slo_ms": 80,
        },
        "traces": {
            "expected_spans": ["gateway", "auth", "router", "queue", "prefill", "decode", "safety", "response"],
            "observed_spans": ["gateway", "auth", "router", "queue", "prefill", "decode", "safety", "response"],
        },
        "logs": [
            {
                "ts": "2026-06-15T10:00:00Z",
                "level": "INFO",
                "service": "router",
                "trace_id": "tr_1",
                "event": "route_decision",
                "status": "ok",
                "redacted": True,
            },
            {
                "ts": "2026-06-15T10:00:01Z",
                "level": "WARN",
                "service": "runtime",
                "trace_id": "tr_1",
                "event": "kv_pressure_high",
                "status": "warn",
                "redacted": True,
            },
        ],
        "events": [
            {
                "ts": "2026-06-15T09:55:00Z",
                "event_type": "deployment_promoted",
                "entity": "chat-serving",
                "version": "deploy_7",
                "actor": "release-bot",
            },
            {
                "ts": "2026-06-15T09:58:00Z",
                "event_type": "autoscaler_scaled_out",
                "entity": "chat-serving",
                "version": "hpa_3",
                "actor": "autoscaler",
            },
        ],
        "correlation": {"records_total": 50, "records_with_id": 50},
        "labels": {
            "metric_count": 10,
            "label_value_counts": {"tenant": 12, "model": 20, "endpoint": 10, "status": 4},
            "series_budget": 100000,
            "high_cardinality_metric_labels": [],
        },
        "training": {
            "train_loss": 1.92,
            "eval_loss": 1.98,
            "learning_rate": 0.0002,
            "gradient_norm": 0.74,
            "tokens_per_s": 178000,
            "gpu_util": 0.72,
            "nccl_time_ratio": 0.12,
            "data_wait_ratio": 0.08,
            "checkpoint_failures": 0,
        },
        "inference": {
            "qps": 320,
            "ttft_p95_ms": 620,
            "tpot_p95_ms": 45,
            "error_rate": 0.003,
            "timeout_rate": 0.001,
            "queue_p95_ms": 110,
            "kv_cache_pressure": 0.72,
            "gpu_util": 0.77,
            "cache_hit_rate": 0.34,
            "fallback_rate": 0.02,
            "degrade_rate": 0.01,
        },
        "data_quality": {
            "ingestion_success_rate": 0.998,
            "schema_error_rate": 0.004,
            "pii_hit_rate": 0.0,
            "duplicate_rate": 0.018,
            "corrupt_shards": 0,
            "quality_gate_pass": True,
            "data_loader_wait_ratio": 0.06,
        },
        "rag_agent": {
            "retrieval_trace_coverage": 1.0,
            "tool_trace_coverage": 1.0,
            "execution_trace_coverage": 1.0,
            "loop_rate": 0.01,
            "groundedness": 0.88,
        },
        "cost": {
            "total_cost_usd": 240.0,
            "total_tokens": 20_000_000,
            "max_cost_per_1k_usd": 0.02,
            "by_tenant": True,
            "by_model": True,
            "by_endpoint": True,
            "by_run": True,
            "idle_gpu_cost_usd": 38.4,
        },
        "privacy": {
            "sensitive_fields_total": 5,
            "redacted_fields": 5,
            "ttl_days": 30,
            "access_scoped": True,
            "audit_enabled": True,
        },
        "alerts": [
            {
                "name": "inference_p99_burn",
                "owner": "serving",
                "runbook": "runbook://serving-latency",
                "severity": "page",
                "actionable": True,
                "duplicate": False,
            },
            {
                "name": "checkpoint_failure",
                "owner": "training",
                "runbook": "runbook://checkpoint",
                "severity": "ticket",
                "actionable": True,
                "duplicate": False,
            },
            {
                "name": "cost_spike",
                "owner": "platform-finops",
                "runbook": "runbook://cost-spike",
                "severity": "ticket",
                "actionable": True,
                "duplicate": False,
            },
        ],
        "platform_gate": {
            "enabled": True,
            "owner": "ai-infra-observability",
            "dashboards_ready": True,
            "incident_process_ready": True,
            "p0_open": False,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "signal_inventory_missing_bad"
    case["signals"].remove("events")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "metric_contract_missing_bad"
    case["metrics"][0].pop("owner")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "slo_error_budget_burn_bad"
    case["slo"]["good_requests"] = 9890
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "latency_quantile_bad"
    case["latency"]["e2e_p99_ms"] = 1800
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "trace_span_missing_bad"
    case["traces"]["observed_spans"].remove("decode")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "log_event_structure_bad"
    case["logs"][0].pop("trace_id")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "correlation_id_missing_bad"
    case["correlation"]["records_with_id"] = 40
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cardinality_explosion_bad"
    case["labels"]["high_cardinality_metric_labels"] = ["request_id"]
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "training_observability_gap_bad"
    case["training"]["data_wait_ratio"] = 0.31
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "inference_observability_gap_bad"
    case["inference"]["kv_cache_pressure"] = 0.94
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "data_quality_observability_bad"
    case["data_quality"]["schema_error_rate"] = 0.08
    case["data_quality"]["quality_gate_pass"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "rag_agent_trace_gap_bad"
    case["rag_agent"]["tool_trace_coverage"] = 0.5
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cost_observability_missing_bad"
    case["cost"]["by_tenant"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "privacy_redaction_retention_bad"
    case["privacy"]["redacted_fields"] = 2
    case["privacy"]["ttl_days"] = 365
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "alert_noise_bad"
    case["alerts"][0]["actionable"] = False
    case["alerts"][0]["duplicate"] = True
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "observability_gate_missing_bad"
    case["platform_gate"]["enabled"] = False
    cases.append(case)

    return cases


audit = MiniAIInfraObservabilityAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("observability_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("observability_gate_pass=" + repr(summary["observability_gate_pass"]))
```

参考输出应类似：

```text
observability_examples={'signal_coverage': 1.0, 'metric_contract_coverage': 1.0, 'availability': 0.997, 'error_budget_remaining': 0.7, 'latency_p99_ms': 890, 'trace_span_coverage': 1.0, 'log_structure_coverage': 1.0, 'event_timeline_coverage': 1.0, 'correlation_id_coverage': 1.0, 'cardinality_estimate': 96000, 'training_tokens_per_s': 178000, 'inference_ttft_p95_ms': 620, 'rag_agent_trace_ready': True, 'cost_per_1k_tokens_usd': 0.012, 'redaction_coverage': 1.0, 'actionable_alerts': 3}
metrics={'signal_inventory_coverage': 0.941, 'metric_contract_completeness': 0.941, 'slo_error_budget_readiness': 0.941, 'latency_quantile_guard': 0.941, 'trace_span_coverage': 0.941, 'log_event_structure': 0.941, 'correlation_id_coverage': 0.941, 'cardinality_budget_control': 0.941, 'training_observability': 0.941, 'inference_observability': 0.941, 'data_quality_observability': 0.941, 'rag_agent_observability': 0.941, 'cost_observability': 0.941, 'privacy_redaction_retention': 0.941, 'alert_actionability': 0.941, 'observability_platform_gate': 0.941}
hard_blocker_count=16
failed_cases=['signal_inventory_missing_bad', 'metric_contract_missing_bad', 'slo_error_budget_burn_bad', 'latency_quantile_bad', 'trace_span_missing_bad', 'log_event_structure_bad', 'correlation_id_missing_bad', 'cardinality_explosion_bad', 'training_observability_gap_bad', 'inference_observability_gap_bad', 'data_quality_observability_bad', 'rag_agent_trace_gap_bad', 'cost_observability_missing_bad', 'privacy_redaction_retention_bad', 'alert_noise_bad', 'observability_gate_missing_bad']
failed_gates=['signal_inventory_coverage', 'metric_contract_completeness', 'slo_error_budget_readiness', 'latency_quantile_guard', 'trace_span_coverage', 'log_event_structure', 'correlation_id_coverage', 'cardinality_budget_control', 'training_observability', 'inference_observability', 'data_quality_observability', 'rag_agent_observability', 'cost_observability', 'privacy_redaction_retention', 'alert_actionability', 'observability_platform_gate']
observability_gate_pass=False
```

这个 demo 的面试价值是把“可观测性”从泛泛的监控系统，变成可验收的工程能力：指标有契约，SLO 能算，trace 能拆阶段，日志和事件能关联，标签基数受控，AI 专属训练 / 推理 / RAG / Agent / 成本 / 隐私信号都能进入同一个排障闭环。

## 44.23 常见误区

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

## 44.24 面试常见追问

问题一：AI Infra 可观测性和普通后端可观测性有什么不同？

可以回答：除了 QPS、延迟、错误率和资源，还要关注 GPU、显存、KV cache、training loss、tokens/s、TTFT、TPOT、数据质量、模型版本、eval 分数、RAG trace、tool trace 和成本。

问题二：Metrics、logs、traces、events 分别解决什么问题？

可以回答：metrics 用于趋势和告警，logs 用于细节排查，traces 用于链路和阶段耗时定位，events 用于状态变化和复盘时间线。

问题三：为什么 request ID 不适合做 metrics label？

可以回答：request ID 基数极高，会导致时间序列爆炸。它适合放在 trace 或日志里，而不是 metrics label。

问题四：如何设计推理服务的可观测性？

可以回答：记录 QPS、TTFT、TPOT、p99、input/output tokens/s、queue length、active sequences、KV cache、GPU、cache hit、fallback、degradation、error/timeout，并通过 trace 拆分 gateway、router、queue、prefill、decode 和 streaming。

问题五：如何设计 AI Infra 可观测性平台的上线门禁？

可以回答：按 signal inventory、metric contract、SLO / error budget、latency quantile、trace span、structured log / event、correlation ID、label cardinality、training、inference、data quality、RAG / Agent、cost、privacy、alert actionability 和 platform gate 逐项验收。重点不是收集更多指标，而是证明每个关键事故都能被发现、定位、解释、复盘和改进。

## 44.25 小练习

1. AI Infra 为什么比普通后端更依赖可观测性？
2. Metrics、logs、traces、events 各适合回答什么问题？
3. 训练任务需要观察哪些模型指标和系统指标？
4. 推理服务为什么要同时看 TTFT 和 TPOT？
5. RAG/Agent 为什么必须保存 trace？
6. 什么是 metrics 高基数问题？
7. 日志脱敏为什么在 AI 系统中特别重要？
8. 如何设计一个推理 p99 延迟告警？
9. 写一个 0 依赖 AI Infra 可观测性审计 demo，覆盖 metrics、logs、traces、events、SLO、标签基数、训练、推理、RAG / Agent、成本、隐私和告警。
10. 构造 5 个可观测性失败样本：request ID 进入 metrics label、trace 缺 decode span、日志未脱敏、告警无 runbook、成本只按总账记录，并说明各自如何修复。

## 44.26 本章小结

本章讲了 AI Infra 可观测性总览。

你需要记住：

1. AI Infra 可观测性要覆盖训练、推理、数据、模型、RAG、Agent、成本和平台资源。
2. Metrics、logs、traces、events 各有作用，必须关联起来使用。
3. 训练可观测性要同时看 loss、tokens/s、GPU、通信、数据读取和 checkpoint。
4. 推理可观测性要关注 TTFT、TPOT、tokens/s、queue、KV cache、runtime 和路由。
5. RAG/Agent 可观测性离不开 retrieval trace、tool call trace 和 execution trace。
6. 可观测性还必须考虑成本、隐私、脱敏、高基数和告警噪音。
7. 第二轮精修后，本章新增的核心抓手是 Observability Platform Gate：让 signals、SLO、trace、日志、事件、成本、隐私、告警和 dashboard 都能被结构化验收。

下一章我们会讲训练故障定位：loss 异常、hang、OOM、通信慢和 I/O 慢。
