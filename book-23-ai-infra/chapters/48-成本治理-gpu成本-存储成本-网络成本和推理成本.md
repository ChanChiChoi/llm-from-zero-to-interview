# 第 48 章 成本治理：GPU 成本、存储成本、网络成本和推理成本

上一章讲了 SLO、SLA、错误预算和生产值班体系。本章讲 AI Infra 成本治理。

AI Infra 的成本非常高，尤其是 GPU、存储、网络和推理服务。成本治理不是财务部门月底看账单，而是平台工程中的实时能力：能看清成本从哪里来，能归因到租户、模型和任务，能在不破坏质量和 SLO 的前提下降低浪费。

先记住一句话：

> AI Infra 成本治理的核心，是把不可见的资源消耗变成可观测、可归因、可预算、可优化的工程指标。

## 48.1 为什么 AI Infra 成本治理重要

AI Infra 的成本特点：

1. GPU 单价高。
2. 训练任务持续时间长。
3. 推理流量可能快速增长。
4. 模型权重和 checkpoint 占用大量存储。
5. 数据读取和跨地域传输产生网络成本。
6. RAG/Agent 多轮调用会放大成本。
7. 实验数量多，很多实验不会进入生产。

如果不治理，成本会随着模型规模、用户增长和实验数量快速失控。

## 48.2 成本治理解决什么问题

成本治理要回答：

1. 钱花在哪里？
2. 哪个团队、租户、项目、模型消耗最多？
3. 哪些资源空闲？
4. 哪些训练任务性价比低？
5. 哪些推理请求成本异常？
6. 哪些 checkpoint 可以清理？
7. 哪些模型应该量化或降级？
8. 哪些 SLO 目标导致成本过高？

没有成本归因，就只能粗暴砍资源，容易伤害核心业务。

## 48.3 成本分类

AI Infra 成本大致分为：

1. GPU 成本。
2. CPU 和内存成本。
3. 存储成本。
4. 网络成本。
5. 推理运行成本。
6. 数据处理成本。
7. 评估成本。
8. 人工标注成本。
9. 运维和平台成本。

其中 GPU、存储、网络和推理成本最常被关注。

## 48.4 GPU 成本

GPU 成本通常是最大头。

它来自：

1. 训练任务。
2. 推理服务。
3. 评估任务。
4. embedding 生成。
5. 数据处理中的 GPU 加速。
6. 空闲预留资源。

GPU 成本不能只看总卡时，还要看利用率和有效产出。

例如同样消耗 1000 GPU 小时，一个任务训练出可上线模型，另一个任务因为配置错误失败，价值完全不同。

## 48.5 GPU 成本指标

常见 GPU 成本指标：

1. GPU hours。
2. GPU cost per job。
3. GPU cost per model version。
4. GPU cost per tenant。
5. GPU utilization。
6. MFU。
7. tokens per GPU hour。
8. idle GPU hours。
9. preempted GPU hours。
10. failed job GPU hours。

其中 failed job GPU hours 很重要。

失败任务消耗的 GPU 时间通常是明显浪费。

## 48.6 训练成本治理

训练成本优化方向：

1. 提高 GPU 利用率。
2. 减少失败任务。
3. 减少无效实验。
4. 使用低优先级队列。
5. 支持抢占和恢复。
6. 合理 checkpoint 频率。
7. 数据供给优化。
8. 分布式通信优化。
9. 实验预算控制。
10. 任务级成本预估。

训练平台可以在任务提交时预估成本：

```text
estimated_cost = gpu_count * expected_hours * gpu_hour_price
```

虽然不精确，但可以防止用户无意识提交超大任务。

## 48.7 推理成本

推理成本由多个因素决定：

1. 模型大小。
2. GPU 类型。
3. 输入 token 数。
4. 输出 token 数。
5. 并发量。
6. batch 效率。
7. cache 命中率。
8. 量化方式。
9. SLO 要求。
10. warm pool 规模。

推理成本常用指标：

1. cost per request。
2. cost per 1k tokens。
3. cost per output token。
4. cost per tenant。
5. cost per model。
6. cost per endpoint。
7. GPU utilization。
8. cache saved cost。

推理成本治理要和模型路由、缓存、限流、扩缩容联动。

## 48.8 推理成本优化

常见优化方式：

1. 小模型优先。
2. 分层模型路由。
3. 语义缓存。
4. prompt cache。
5. 结果缓存。
6. 输出长度限制。
7. prompt 压缩。
8. 量化。
9. continuous batching。
10. 自动扩缩容。
11. 降低 warm pool 空闲成本。
12. 低优先级请求离峰处理。

优化要注意质量和 SLO。

例如把所有请求都降到小模型，成本降了，但用户体验可能崩掉。

## 48.9 存储成本

AI Infra 存储成本来自：

1. raw data。
2. cleaned dataset。
3. dataset shards。
4. checkpoint。
5. model weights。
6. quantized weights。
7. eval outputs。
8. logs。
9. traces。
10. vector index。

大模型 checkpoint 和数据集非常占空间。

如果每个实验都长期保留全部 checkpoint，存储成本会迅速上升。

## 48.10 存储成本指标

常见指标：

1. storage by artifact type。
2. storage by tenant。
3. storage by project。
4. checkpoint storage growth。
5. dataset storage growth。
6. hot/cold storage ratio。
7. unused artifact size。
8. duplicate artifact size。
9. log retention size。
10. trace storage size。

存储成本治理的关键是生命周期策略。

## 48.11 存储成本优化

优化手段：

1. Checkpoint retention policy。
2. 删除失败实验中间产物。
3. 只保留 best/latest checkpoint。
4. 冷热分层存储。
5. 压缩。
6. 去重。
7. 权重分片复用。
8. eval output 采样保存。
9. trace 采样。
10. log TTL。

删除前必须检查血缘和部署依赖。

不能因为省存储，把可回滚的生产 artifact 删除。

## 48.12 网络成本

网络成本常被低估。

来源包括：

1. 训练数据从对象存储读取。
2. checkpoint 上传下载。
3. 跨地域复制模型权重。
4. 分布式训练通信。
5. 推理服务跨地域调用。
6. RAG 检索访问远端索引。
7. 日志和 trace 上报。

网络成本不仅是费用，也影响延迟和稳定性。

## 48.13 网络成本优化

优化方法：

1. 数据本地性调度。
2. 本地 NVMe 缓存。
3. 权重预分发。
4. 避免跨地域训练读取。
5. checkpoint 增量上传。
6. 压缩日志和 trace。
7. 就近推理。
8. 向量索引地域化。

数据在哪里，计算就尽量在哪里。

跨地域传大模型权重和数据集通常代价很高。

## 48.14 成本归因

成本归因是成本治理的基础。

要能按以下维度归因：

1. 租户。
2. 团队。
3. 项目。
4. 用户。
5. 模型。
6. 模型版本。
7. 训练任务。
8. 推理 endpoint。
9. dataset。
10. experiment run。

没有归因，就没有治理。

所有资源申请和任务提交都应该带 owner、project、tenant、cost center 等标签。

## 48.15 标签和成本中心

成本标签包括：

1. tenant。
2. project。
3. team。
4. owner。
5. environment。
6. job_type。
7. model。
8. priority。
9. cost_center。

平台应该强制关键标签存在。

没有标签的资源容易变成无人认领成本。

## 48.16 预算和配额

成本治理需要预算和配额。

预算控制的是花多少钱。

配额控制的是能用多少资源。

例如：

1. 每个团队每月 GPU 预算。
2. 每个租户推理 token 预算。
3. 每个项目最大 checkpoint 存储。
4. 每个用户最大并发训练任务。
5. 每个 endpoint 最大 warm pool 副本。

预算和配额可以软限制，也可以硬限制。

软限制用于提醒，硬限制用于保护平台。

## 48.17 成本告警

常见成本告警：

1. 日成本超过预算。
2. 某租户成本突增。
3. idle GPU hours 过高。
4. failed job cost 过高。
5. 推理 cost per 1k tokens 上升。
6. checkpoint 存储增长异常。
7. 跨地域流量异常。
8. 某 Agent run 成本异常。

成本告警要能定位责任方和资源来源。

否则只能知道“贵了”，不知道谁导致。

## 48.18 成本和 SLO 的权衡

成本优化不能孤立进行。

例如：

1. 减少 warm pool 会降低成本，但增加冷启动风险。
2. 降低副本数会省钱，但 p99 可能变差。
3. 减少 checkpoint 会省存储，但恢复成本上升。
4. 使用小模型会省推理成本，但质量可能下降。
5. 降低 trace 留存会省存储，但排障能力下降。

因此成本治理必须和 SLO、质量、安全一起决策。

## 48.19 Unit Economics

Unit economics 是单位经济性。

AI Infra 常见单位成本：

1. cost per 1k tokens。
2. cost per request。
3. cost per successful task。
4. cost per eval sample。
5. cost per training token。
6. cost per model version。
7. cost per RAG query。
8. cost per Agent run。

单位成本比总成本更有用。

总成本上升可能是业务增长，单位成本上升才可能说明效率变差。

## 48.20 成本 Dashboard

成本 dashboard 应该展示：

1. 总成本趋势。
2. GPU 成本分布。
3. 存储成本分布。
4. 网络成本分布。
5. 推理单位成本。
6. 训练任务成本排行。
7. 租户成本排行。
8. idle 和 waste。
9. failed job cost。
10. 预算使用率。

Dashboard 要能下钻到 job、model、tenant 和 artifact。

## 48.21 成本治理系统架构

一个成本治理系统可以这样设计：

```text
Resource Usage Collectors
  -> Usage Metering
  -> Cost Attribution
  -> Budget / Quota Service
  -> Cost Dashboard
  -> Cost Alerts
  -> Optimization Recommendations
```

Usage 来源包括：

1. Kubernetes。
2. GPU exporter。
3. training platform。
4. inference platform。
5. object storage。
6. network billing。
7. artifact store。
8. trace/log system。

成本治理依赖统一标签和资源用量采集。

## 48.22 常见误区

误区一：成本治理就是月底看账单。

月底账单只能事后分析，平台需要实时或准实时用量和预算监控。

误区二：GPU 利用率高就成本健康。

如果任务无效、失败或模型不可用，高利用率也可能是浪费。

误区三：只优化训练成本。

推理、存储、网络、评估和 trace 都可能成为大头。

误区四：成本优化就是砍资源。

真正优化是提高单位产出，例如更高 tokens/GPU hour、更低 cost per 1k tokens。

误区五：存储便宜，不用治理。

大模型 checkpoint、数据集和 trace 规模很大，长期看存储成本不可忽视。

## 48.23 面试常见追问

问题一：AI Infra 成本主要来自哪里？

可以回答：主要来自 GPU 训练和推理、模型和 checkpoint 存储、数据和权重传输网络成本、评估和 embedding 生成、日志 trace 留存，以及人工标注和平台运维。

问题二：如何降低推理成本？

可以回答：通过小模型优先、模型路由、缓存、prompt 压缩、输出长度控制、量化、continuous batching、自动扩缩容、warm pool 优化和单位 token 成本监控降低成本。

问题三：如何做成本归因？

可以回答：所有资源、任务、模型、endpoint、artifact 都必须带 tenant、team、project、owner、model、job_type、cost_center 等标签，再结合 usage metering 归因到租户、团队、模型和任务。

问题四：成本优化和 SLO 冲突怎么办？

可以回答：成本优化必须在质量和 SLO 约束下进行。可以分层服务，不同业务使用不同 SLO、模型、缓存、配额和降级策略，而不是一刀切省钱。

## 48.24 小练习

1. GPU hours 和 tokens per GPU hour 分别说明什么？
2. 为什么 failed job GPU hours 是重要成本指标？
3. 推理 cost per 1k tokens 受哪些因素影响？
4. Checkpoint retention policy 如何设计？
5. 网络成本在 AI Infra 中有哪些来源？
6. 为什么成本标签必须强制填写？
7. 如何设计租户级推理 token 预算？
8. 成本优化如何避免伤害 SLO？

## 48.25 本章小结

本章讲了 AI Infra 成本治理。

你需要记住：

1. 成本治理的目标是让资源消耗可观测、可归因、可预算、可优化。
2. GPU、推理、存储、网络、评估和 trace 都可能产生显著成本。
3. 成本归因依赖统一标签、用量采集和成本中心。
4. 训练成本要关注 GPU hours、失败任务、tokens/GPU hour 和资源利用率。
5. 推理成本要关注 cost per request、cost per 1k tokens、cache 命中率和模型路由。
6. 成本优化必须和 SLO、质量、安全和回滚能力一起权衡。

下一章我们会讲资源利用率优化：混部、抢占、弹性训练和低优先级队列。
