# 第 48 章 成本治理：GPU 成本、存储成本、网络成本和推理成本

上一章讲了 SLO、SLA、错误预算和生产值班体系。本章讲 AI Infra 成本治理。

AI Infra 的成本非常高，尤其是 GPU、存储、网络和推理服务。成本治理不是财务部门月底看账单，而是平台工程中的实时能力：能看清成本从哪里来，能归因到租户、模型和任务，能在不破坏质量和 SLO 的前提下降低浪费。

先记住一句话：

> AI Infra 成本治理的核心，是把不可见的资源消耗变成可观测、可归因、可预算、可优化的工程指标。

## 48.0 本讲资料边界与第二轮精修口径

本章按通用 AI Infra 成本治理抽象来写，不绑定某个云厂商账单、GPU 型号、Kubernetes 成本插件、FinOps 组织形态或内部计费系统。资料校准时，主要参考 FinOps 对 allocation、tagging、预算、showback / chargeback 和单位经济账的通用口径，参考 OpenCost / Kubernetes 成本分摊对 workload、namespace、label 和资源用量计量的抽象，并结合前文集群容量规划、训练调度、推理平台、缓存、可观测性、SLO 值班体系和 artifact 生命周期章节。

第二轮精修只做三件事：

1. 把 GPU、训练、推理、存储、网络、评估、RAG / Agent、日志 trace 和平台运维成本统一成可计量、可归因、可下钻的成本样本。
2. 补齐 GPU-hours、tokens per GPU hour、失败任务浪费、推理 cost per 1k tokens、缓存节省、存储生命周期、网络出站、标签归因、预算使用率、异常成本、单位经济账和 SLO / 成本权衡公式。
3. 增加一个 0 依赖 Python demo，用 toy cost cases 检查成本治理是否只是“月底看账单”，还是能把 usage、attribution、budget、quota、dashboard、告警和优化建议串成闭环。

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

一个成本治理样本可以写成：

$$
C_i=(u_i,g_i,t_i,p_i,s_i,n_i,a_i,b_i,q_i,d_i,o_i,z_i)
$$

其中 `u_i` 是 usage 记录，`g_i` 是 GPU / accelerator 消耗，`t_i` 是训练任务与实验，`p_i` 是推理请求与 token 画像，`s_i` 是存储 artifact，`n_i` 是网络流量，`a_i` 是归因标签，`b_i` 是预算，`q_i` 是配额，`d_i` 是 dashboard 下钻证据，`o_i` 是优化建议，`z_i` 是成本治理结论。

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

成本治理的总门禁可以写成：

$$
G_{\mathrm{cost}}=\mathbf{1}\left[C_{\mathrm{usage}}\ge \tau_u \land C_{\mathrm{attr}}\ge \tau_a \land U_{\mathrm{budget}}\le 1 \land R_{\mathrm{waste}}\le \rho_w \land P_0=0\right]
$$

其中 `C_{\mathrm{usage}}` 是用量计量覆盖率，`C_{\mathrm{attr}}` 是成本归因覆盖率，`U_{\mathrm{budget}}` 是预算使用率，`R_{\mathrm{waste}}` 是浪费成本比例，`P_0` 是未关闭的成本治理硬阻断。

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

总成本可以按类型分解：

$$
K_{\mathrm{total}}=K_{\mathrm{gpu}}+K_{\mathrm{cpu}}+K_{\mathrm{mem}}+K_{\mathrm{storage}}+K_{\mathrm{network}}+K_{\mathrm{infer}}+K_{\mathrm{data}}+K_{\mathrm{eval}}+K_{\mathrm{ops}}
$$

面试中要强调：成本分类必须能回到资源用量和 owner，否则只是财务账单重命名。

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

GPU 任务成本可以粗略写成：

$$
K_{\mathrm{gpu}}=G\cdot T\cdot P_{\mathrm{gpu}}\cdot \eta_{\mathrm{bill}}
$$

其中 `G` 是 GPU 数，`T` 是运行小时数，`P_{\mathrm{gpu}}` 是单 GPU 小时价格，`\eta_{\mathrm{bill}}` 是计费修正系数，例如预留、竞价、包年包月或云上折扣。成本治理不能只看 `K_{\mathrm{gpu}}`，还要看这笔钱产生了多少有效训练 token、模型版本、评估提升或业务收益。

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

训练吞吐效率可以写成：

$$
E_{\mathrm{tok/gpu\_h}}=\frac{N_{\mathrm{token}}}{G\cdot T}
$$

失败任务 GPU 成本可以写成：

$$
K_{\mathrm{failed}}=\sum_{j=1}^{J}G_jT_jP_j
$$

其中 `J` 是失败任务数量。成本 dashboard 至少要能把 `K_{\mathrm{failed}}` 下钻到 owner、project、queue、failure type、node、image、dataset 和 checkpoint 状态。

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

训练任务提交前的估算可以写成：

$$
\hat K_{\mathrm{train}}=G_{\mathrm{req}}\cdot \hat T\cdot P_{\mathrm{gpu}}+K_{\mathrm{storage}}+K_{\mathrm{network}}+K_{\mathrm{eval}}
$$

如果 `\hat K_{\mathrm{train}}` 超过项目预算或配额，平台应要求审批、降级队列、降低资源、拆分实验或补充业务理由。训练成本治理的目标不是阻止训练，而是让高成本实验有清楚假设、预算和失败止损条件。

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

推理单位成本常写成每 1k token 成本：

$$
K_{\mathrm{1k}}=\frac{1000K_{\mathrm{infer}}}{N_{\mathrm{in}}+N_{\mathrm{out}}}
$$

其中 `K_{\mathrm{infer}}` 是窗口内推理运行成本，`N_{\mathrm{in}}` 和 `N_{\mathrm{out}}` 分别是输入和输出 token 数。对产品和平台更有用的是分租户、分模型、分 endpoint、分请求类型计算这个指标。

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

缓存节省成本可以粗略写成：

$$
S_{\mathrm{cache}}=N_{\mathrm{hit}}\cdot K_{\mathrm{prefill\_saved}}-K_{\mathrm{cache}}
$$

其中 `N_{\mathrm{hit}}` 是缓存命中次数，`K_{\mathrm{prefill\_saved}}` 是单次命中节省的 prefill / retrieval / tool 成本，`K_{\mathrm{cache}}` 是缓存存储、维护和一致性成本。只有 `S_{\mathrm{cache}}>0` 且质量和权限不受损时，缓存才是真正的成本优化。

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

存储月成本可以按 artifact 类型分层：

$$
K_{\mathrm{storage}}=\sum_{a=1}^{A}S_aP_aT_a+K_{\mathrm{request}}+K_{\mathrm{restore}}
$$

其中 `S_a` 是第 `a` 类 artifact 的存储大小，`P_a` 是单位存储价格，`T_a` 是保留时间，`K_{\mathrm{request}}` 是请求成本，`K_{\mathrm{restore}}` 是冷归档恢复成本。

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

可删除候选比例可以写成：

$$
R_{\mathrm{delete}}=\frac{S_{\mathrm{unused}}+S_{\mathrm{duplicate}}+S_{\mathrm{expired}}}{S_{\mathrm{total}}}
$$

但 `R_{\mathrm{delete}}` 只能作为候选筛选，真正删除前必须通过血缘和回滚依赖检查。

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

删除安全门禁可以写成：

$$
G_{\mathrm{delete}}=\mathbf{1}\left[D_{\mathrm{deploy}}=0 \land D_{\mathrm{rollback}}=0 \land D_{\mathrm{lineage}}=0 \land A_{\mathrm{owner}}=1\right]
$$

其中 `D_{\mathrm{deploy}}` 表示是否仍被线上部署依赖，`D_{\mathrm{rollback}}` 表示是否仍是回滚目标，`D_{\mathrm{lineage}}` 表示是否仍被训练、评估或审计血缘依赖，`A_{\mathrm{owner}}` 表示 owner 是否批准。

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

网络成本可以写成：

$$
K_{\mathrm{network}}=\sum_{r=1}^{R}D_rP_r
$$

其中 `D_r` 是第 `r` 类数据传输量，例如跨地域 checkpoint、模型权重分发、对象存储读取、日志 trace 上报，`P_r` 是对应单价或内部成本权重。

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

数据本地性得分可以写成：

$$
S_{\mathrm{locality}}=\frac{D_{\mathrm{local}}}{D_{\mathrm{local}}+D_{\mathrm{remote}}}
$$

其中 `D_{\mathrm{local}}` 是本地域或同集群读取的数据量，`D_{\mathrm{remote}}` 是跨地域或远端读取的数据量。成本优化通常先提升 `S_{\mathrm{locality}}`，再考虑压缩、增量同步和缓存。

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

归因覆盖率可以写成：

$$
C_{\mathrm{attr}}=\frac{N_{\mathrm{tagged}}}{N_{\mathrm{resource}}}
$$

未归因成本比例可以写成：

$$
R_{\mathrm{unallocated}}=\frac{K_{\mathrm{unallocated}}}{K_{\mathrm{total}}}
$$

成熟平台应把 `R_{\mathrm{unallocated}}` 控制在很低水平，并对无 owner、无 cost center、无 project 的资源触发拦截或升级。

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

标签完整率可以写成：

$$
C_{\mathrm{tag}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\mathrm{tenant}_i\land \mathrm{project}_i\land \mathrm{owner}_i\land \mathrm{cost\_center}_i]
$$

这里 `tenant`、`project`、`owner` 和 `cost_center` 是正文标签字段，不建议在公式里展开成复杂变量。成本标签要在资源创建时强制，而不是月底人工补账。

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

预算使用率和配额使用率分别可以写成：

$$
U_{\mathrm{budget}}=\frac{K_{\mathrm{actual}}}{K_{\mathrm{budget}}}
$$

$$
U_{\mathrm{quota}}=\frac{R_{\mathrm{used}}}{R_{\mathrm{quota}}}
$$

其中 `K_{\mathrm{actual}}` 是实际成本，`K_{\mathrm{budget}}` 是预算，`R_{\mathrm{used}}` 是资源使用量，`R_{\mathrm{quota}}` 是资源配额。预算回答“花了多少钱”，配额回答“用了多少资源”，两者都要有 owner 和例外审批。

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

成本异常放大系数可以写成：

$$
A_t=\frac{K_t}{\max(\epsilon,\bar K_{t-w:t-1})}
$$

其中 `K_t` 是当前窗口成本，`\bar K_{t-w:t-1}` 是历史窗口均值，`\epsilon` 是防止除零的小常数。一个可行动的成本告警还要附带 tenant、project、model、job、endpoint、artifact、recent change 和建议动作。

## 48.18 成本和 SLO 的权衡

成本优化不能孤立进行。

例如：

1. 减少 warm pool 会降低成本，但增加冷启动风险。
2. 降低副本数会省钱，但 p99 可能变差。
3. 减少 checkpoint 会省存储，但恢复成本上升。
4. 使用小模型会省推理成本，但质量可能下降。
5. 降低 trace 留存会省存储，但排障能力下降。

因此成本治理必须和 SLO、质量、安全一起决策。

成本优化门禁可以写成：

$$
G_{\mathrm{opt}}=\mathbf{1}\left[\Delta K<0 \land \Delta Q\ge -q_0 \land \Delta S_{\mathrm{slo}}\ge -s_0 \land R_{\mathrm{risk}}\le r_0\right]
$$

其中 `\Delta K` 是成本变化，`\Delta Q` 是质量变化，`\Delta S_{\mathrm{slo}}` 是 SLO 变化，`R_{\mathrm{risk}}` 是新增风险。只有成本下降且质量、SLO 和风险都在可接受范围内，优化才应该扩大使用。

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

单位毛利可以写成：

$$
M_{\mathrm{unit}}=V_{\mathrm{unit}}-K_{\mathrm{unit}}
$$

其中 `V_{\mathrm{unit}}` 是一次请求、一次成功任务或 1k token 带来的价值，`K_{\mathrm{unit}}` 是对应单位成本。若 `M_{\mathrm{unit}}<0`，规模越大亏损越大，不能只靠扩大流量解决。

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

dashboard 下钻覆盖率可以写成：

$$
C_{\mathrm{drill}}=\frac{N_{\mathrm{cost\_item\_drillable}}}{N_{\mathrm{cost\_item}}}
$$

如果只能看到总成本曲线，不能下钻到 job、tenant、model、endpoint、dataset、artifact 和 change event，就无法支持真正治理。

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

## 48.22 成本治理审计指标和最小 demo

把本章落到平台验收时，可以用 16 个门禁：

1. Usage Metering Coverage：GPU、CPU、memory、storage、network、request、token、artifact、logs 和 traces 是否都有用量记录。
2. Cost Attribution Labels：tenant、project、team、owner、model、job type、environment 和 cost center 是否强制存在。
3. GPU Cost Efficiency：GPU-hours、tokens per GPU hour、MFU / utilization 和空闲 GPU 小时是否可计算。
4. Training Waste Control：失败任务、抢占丢失、无效实验和 checkpoint / resume 成本是否可定位。
5. Inference Unit Cost：cost per request、cost per 1k tokens、cost per output token、endpoint / model / tenant 成本是否可计算。
6. Cache Savings Accounting：result cache、prompt cache、KV / prefix cache 和 semantic cache 的节省与维护成本是否可量化。
7. Storage Lifecycle Governance：checkpoint、dataset、model、eval output、logs、traces 和 vector index 是否有保留、归档、删除和冷存储策略。
8. Network Egress Governance：跨地域数据、权重、checkpoint、日志 trace 和远端索引访问是否可计量和优化。
9. Artifact Dependency Safety：删除或降冷前是否检查部署、回滚、训练、评估、审计和血缘依赖。
10. Budget Quota Enforcement：团队、租户、项目、endpoint、训练任务和 artifact 是否有预算 / 配额 / 例外审批。
11. Cost Anomaly Alerting：日成本、单位成本、idle、failed job、storage growth、egress 和 agent run 异常是否能告警到 owner。
12. Tenant Model Chargeback：成本是否能按租户、模型、模型版本、endpoint、job 和 experiment 做 showback / chargeback。
13. Dashboard Drilldown Readiness：dashboard 是否能从总成本下钻到 tenant、model、job、artifact、request slice 和 recent change。
14. Optimization Recommendation Trace：优化建议是否带成本节省、质量 / SLO 风险、owner、证据和回滚条件。
15. SLO Quality Cost Tradeoff：降本动作是否同时检查质量、SLO、安全、回滚和用户体验。
16. Cost Governance Gate：最终是否有 owner、预算策略、优化 backlog、审批记录、审计和 P0 风险阻断。

综合门禁：

$$
G_{\mathrm{cost\_governance}}=\prod_{j=1}^{16}G_j
$$

下面是一个 0 依赖 demo，用 toy cost cases 检查成本治理是否只是“月底看账单”，还是能把用量、归因、预算、配额、告警、dashboard 和优化建议串成闭环。

```python
from copy import deepcopy


class MiniAICostGovernanceAudit:
    GATES = [
        "usage_metering_coverage",
        "cost_attribution_labels",
        "gpu_cost_efficiency",
        "training_waste_control",
        "inference_unit_cost",
        "cache_savings_accounting",
        "storage_lifecycle_governance",
        "network_egress_governance",
        "artifact_dependency_safety",
        "budget_quota_enforcement",
        "cost_anomaly_alerting",
        "tenant_model_chargeback",
        "dashboard_drilldown_readiness",
        "optimization_recommendation_trace",
        "slo_quality_cost_tradeoff",
        "cost_governance_gate",
    ]

    USAGE_FIELDS = ["gpu", "cpu", "memory", "storage", "network", "request", "token", "artifact", "logs", "traces"]
    LABEL_FIELDS = ["tenant", "project", "team", "owner", "model", "job_type", "environment", "cost_center"]
    DASHBOARD_FIELDS = ["tenant", "model", "job", "artifact", "endpoint", "request_slice", "recent_change"]

    @staticmethod
    def present(record, key):
        return key in record and record[key] is not None and record[key] != ""

    def coverage(self, record, fields):
        if not record:
            return 0.0
        return sum(1 for field in fields if self.present(record, field)) / len(fields)

    def gpu_job_cost(self, case):
        gpu = case["gpu"]
        return gpu["count"] * gpu["hours"] * gpu["price_per_hour"] * gpu["billing_factor"]

    def tokens_per_gpu_hour(self, case):
        gpu = case["gpu"]
        return gpu["train_tokens"] / max(1e-12, gpu["count"] * gpu["hours"])

    def failed_job_cost(self, case):
        return sum(job["gpu_count"] * job["hours"] * job["price_per_hour"] for job in case["training_waste"]["failed_jobs"])

    def inference_cost_per_1k_tokens(self, case):
        inference = case["inference"]
        total_tokens = inference["input_tokens"] + inference["output_tokens"]
        return 1000.0 * inference["cost"] / total_tokens

    def cache_saved_cost(self, case):
        cache = case["cache"]
        return cache["hits"] * cache["saved_cost_per_hit"] - cache["maintenance_cost"]

    def storage_monthly_cost(self, case):
        storage = case["storage"]
        return sum(item["gib"] * item["price_per_gib_month"] for item in storage["artifacts"])

    def network_egress_cost(self, case):
        return sum(flow["gib"] * flow["price_per_gib"] for flow in case["network"]["flows"])

    def budget_used(self, case):
        budget = case["budget"]
        return budget["actual_cost"] / budget["monthly_budget"]

    def unit_margin(self, case):
        unit = case["unit_economics"]
        return unit["value_per_unit"] - unit["cost_per_unit"]

    def usage_metering_coverage(self, case):
        usage = case.get("usage_metering", {})
        return self.coverage(usage, self.USAGE_FIELDS) == 1.0 and usage.get("freshness_minutes", 10**9) <= usage.get("max_freshness_minutes", 0)

    def cost_attribution_labels(self, case):
        labels = case.get("labels", {})
        return self.coverage(labels, self.LABEL_FIELDS) == 1.0 and labels.get("unallocated_cost_ratio", 1.0) <= labels.get("max_unallocated_cost_ratio", 0.0)

    def gpu_cost_efficiency(self, case):
        gpu = case.get("gpu", {})
        return (
            self.gpu_job_cost(case) <= gpu.get("max_job_cost", 0.0)
            and self.tokens_per_gpu_hour(case) >= gpu.get("min_tokens_per_gpu_hour", 10**18)
            and gpu.get("mfu", 0.0) >= gpu.get("min_mfu", 1.0)
            and gpu.get("idle_gpu_hours", 10**9) <= gpu.get("max_idle_gpu_hours", 0)
        )

    def training_waste_control(self, case):
        waste = case.get("training_waste", {})
        return (
            self.failed_job_cost(case) <= waste.get("max_failed_job_cost", 0.0)
            and waste.get("invalid_experiments", 10**9) <= waste.get("max_invalid_experiments", 0)
            and waste.get("resume_ready") is True
            and waste.get("failure_owner_ready") is True
        )

    def inference_unit_cost(self, case):
        inference = case.get("inference", {})
        return (
            self.inference_cost_per_1k_tokens(case) <= inference.get("max_cost_per_1k_tokens", 0.0)
            and inference.get("cost_per_request", 10**9) <= inference.get("max_cost_per_request", 0.0)
            and inference.get("tenant_breakdown") is True
            and inference.get("model_endpoint_breakdown") is True
        )

    def cache_savings_accounting(self, case):
        cache = case.get("cache", {})
        return (
            self.cache_saved_cost(case) >= cache.get("min_saved_cost", 10**9)
            and cache.get("permission_safe") is True
            and cache.get("quality_regression") <= cache.get("max_quality_regression", 0.0)
            and cache.get("versioned_keys") is True
        )

    def storage_lifecycle_governance(self, case):
        storage = case.get("storage", {})
        return (
            self.storage_monthly_cost(case) <= storage.get("max_monthly_cost", 0.0)
            and storage.get("retention_policy") is True
            and storage.get("hot_cold_tiering") is True
            and storage.get("duplicate_ratio") <= storage.get("max_duplicate_ratio", 0.0)
            and storage.get("log_trace_ttl") is True
        )

    def network_egress_governance(self, case):
        network = case.get("network", {})
        local = network["local_gib"]
        remote = network["remote_gib"]
        locality = local / max(1e-12, local + remote)
        return (
            self.network_egress_cost(case) <= network.get("max_egress_cost", 0.0)
            and locality >= network.get("min_locality", 1.0)
            and network.get("cross_region_reason") is True
            and network.get("compression_or_incremental") is True
        )

    def artifact_dependency_safety(self, case):
        artifact = case.get("artifact_delete", {})
        return (
            artifact.get("deployment_dependency") is False
            and artifact.get("rollback_dependency") is False
            and artifact.get("lineage_dependency") is False
            and artifact.get("owner_approved") is True
            and artifact.get("delete_plan_audited") is True
        )

    def budget_quota_enforcement(self, case):
        budget = case.get("budget", {})
        return (
            self.budget_used(case) <= budget.get("max_budget_used", 0.0)
            and budget.get("quota_used") <= budget.get("quota_limit")
            and budget.get("exception_approval") is True
            and budget.get("hard_limit_for_p0") is True
        )

    def cost_anomaly_alerting(self, case):
        alerts = case.get("alerts", {})
        return (
            alerts.get("daily_cost_ratio", 10**9) <= alerts.get("max_daily_cost_ratio", 0.0)
            and alerts.get("unit_cost_ratio", 10**9) <= alerts.get("max_unit_cost_ratio", 0.0)
            and alerts.get("owner_routed") is True
            and alerts.get("recent_change_attached") is True
            and alerts.get("recommendation_attached") is True
        )

    def tenant_model_chargeback(self, case):
        chargeback = case.get("chargeback", {})
        return (
            chargeback.get("tenant") is True
            and chargeback.get("model") is True
            and chargeback.get("model_version") is True
            and chargeback.get("endpoint") is True
            and chargeback.get("job") is True
            and chargeback.get("experiment") is True
            and chargeback.get("cost_center_count", 0) >= chargeback.get("min_cost_center_count", 10**9)
        )

    def dashboard_drilldown_readiness(self, case):
        dashboard = case.get("dashboard", {})
        return self.coverage(dashboard.get("drilldown", {}), self.DASHBOARD_FIELDS) == 1.0 and dashboard.get("fresh") is True

    def optimization_recommendation_trace(self, case):
        rec = case.get("recommendation", {})
        return (
            rec.get("estimated_savings", 0.0) >= rec.get("min_estimated_savings", 10**9)
            and rec.get("evidence") is True
            and rec.get("owner") is True
            and rec.get("rollback_condition") is True
            and rec.get("quality_slo_risk_reviewed") is True
        )

    def slo_quality_cost_tradeoff(self, case):
        tradeoff = case.get("tradeoff", {})
        return (
            tradeoff.get("cost_delta", 0.0) < 0
            and tradeoff.get("quality_delta", -1.0) >= -tradeoff.get("max_quality_drop", 0.0)
            and tradeoff.get("slo_delta", -1.0) >= -tradeoff.get("max_slo_drop", 0.0)
            and tradeoff.get("risk_score", 1.0) <= tradeoff.get("max_risk_score", 0.0)
        )

    def cost_governance_gate(self, case):
        gate = case.get("platform_gate", {})
        return (
            gate.get("enabled") is True
            and bool(gate.get("owner"))
            and gate.get("budget_policy") is True
            and gate.get("optimization_backlog") is True
            and gate.get("approval_records") is True
            and gate.get("audit_ready") is True
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
            "cost_governance_gate_pass": metrics["cost_governance_gate"] == 1.0,
        }

    def example_outputs(self, case):
        return {
            "gpu_job_cost_usd": round(self.gpu_job_cost(case), 3),
            "tokens_per_gpu_hour": round(self.tokens_per_gpu_hour(case), 3),
            "failed_job_cost_usd": round(self.failed_job_cost(case), 3),
            "cost_per_1k_tokens_usd": round(self.inference_cost_per_1k_tokens(case), 3),
            "cache_saved_cost_usd": round(self.cache_saved_cost(case), 3),
            "storage_monthly_cost_usd": round(self.storage_monthly_cost(case), 3),
            "network_egress_cost_usd": round(self.network_egress_cost(case), 3),
            "budget_used": round(self.budget_used(case), 3),
            "idle_gpu_hours": case["gpu"]["idle_gpu_hours"],
            "unit_margin": round(self.unit_margin(case), 3),
            "cost_center_count": case["chargeback"]["cost_center_count"],
        }


def build_good_case():
    return {
        "case_id": "full_cost_governance",
        "usage_metering": {
            "gpu": True,
            "cpu": True,
            "memory": True,
            "storage": True,
            "network": True,
            "request": True,
            "token": True,
            "artifact": True,
            "logs": True,
            "traces": True,
            "freshness_minutes": 10,
            "max_freshness_minutes": 30,
        },
        "labels": {
            "tenant": "enterprise-a",
            "project": "assistant",
            "team": "infra",
            "owner": "cost-owner",
            "model": "chat-v8",
            "job_type": "sft",
            "environment": "prod",
            "cost_center": "cc-ai-01",
            "unallocated_cost_ratio": 0.02,
            "max_unallocated_cost_ratio": 0.05,
        },
        "gpu": {
            "count": 8,
            "hours": 10,
            "price_per_hour": 32.0,
            "billing_factor": 1.0,
            "train_tokens": 400_000_000,
            "max_job_cost": 3000.0,
            "min_tokens_per_gpu_hour": 4_000_000,
            "mfu": 0.42,
            "min_mfu": 0.35,
            "idle_gpu_hours": 12,
            "max_idle_gpu_hours": 16,
        },
        "training_waste": {
            "failed_jobs": [
                {"gpu_count": 2, "hours": 2, "price_per_hour": 32.0},
                {"gpu_count": 1, "hours": 1, "price_per_hour": 32.0},
            ],
            "max_failed_job_cost": 200.0,
            "invalid_experiments": 1,
            "max_invalid_experiments": 1,
            "resume_ready": True,
            "failure_owner_ready": True,
        },
        "inference": {
            "cost": 90.0,
            "input_tokens": 3_000_000,
            "output_tokens": 2_000_000,
            "max_cost_per_1k_tokens": 0.02,
            "cost_per_request": 0.012,
            "max_cost_per_request": 0.02,
            "tenant_breakdown": True,
            "model_endpoint_breakdown": True,
        },
        "cache": {
            "hits": 1000,
            "saved_cost_per_hit": 0.06,
            "maintenance_cost": 6.0,
            "min_saved_cost": 40.0,
            "permission_safe": True,
            "quality_regression": 0.0,
            "max_quality_regression": 0.01,
            "versioned_keys": True,
        },
        "storage": {
            "artifacts": [
                {"type": "checkpoint", "gib": 5000, "price_per_gib_month": 0.4},
                {"type": "dataset", "gib": 3000, "price_per_gib_month": 0.25},
                {"type": "trace", "gib": 700, "price_per_gib_month": 0.2},
            ],
            "max_monthly_cost": 3200.0,
            "retention_policy": True,
            "hot_cold_tiering": True,
            "duplicate_ratio": 0.04,
            "max_duplicate_ratio": 0.08,
            "log_trace_ttl": True,
        },
        "network": {
            "flows": [
                {"name": "weights_cross_region", "gib": 400, "price_per_gib": 0.12},
                {"name": "logs_trace_export", "gib": 600, "price_per_gib": 0.10},
            ],
            "max_egress_cost": 120.0,
            "local_gib": 9000,
            "remote_gib": 1000,
            "min_locality": 0.85,
            "cross_region_reason": True,
            "compression_or_incremental": True,
        },
        "artifact_delete": {
            "deployment_dependency": False,
            "rollback_dependency": False,
            "lineage_dependency": False,
            "owner_approved": True,
            "delete_plan_audited": True,
        },
        "budget": {
            "actual_cost": 7200.0,
            "monthly_budget": 10000.0,
            "max_budget_used": 0.8,
            "quota_used": 80,
            "quota_limit": 100,
            "exception_approval": True,
            "hard_limit_for_p0": True,
        },
        "alerts": {
            "daily_cost_ratio": 1.2,
            "max_daily_cost_ratio": 1.5,
            "unit_cost_ratio": 1.1,
            "max_unit_cost_ratio": 1.25,
            "owner_routed": True,
            "recent_change_attached": True,
            "recommendation_attached": True,
        },
        "chargeback": {
            "tenant": True,
            "model": True,
            "model_version": True,
            "endpoint": True,
            "job": True,
            "experiment": True,
            "cost_center_count": 3,
            "min_cost_center_count": 3,
        },
        "dashboard": {
            "drilldown": {
                "tenant": True,
                "model": True,
                "job": True,
                "artifact": True,
                "endpoint": True,
                "request_slice": True,
                "recent_change": True,
            },
            "fresh": True,
        },
        "recommendation": {
            "estimated_savings": 650.0,
            "min_estimated_savings": 500.0,
            "evidence": True,
            "owner": True,
            "rollback_condition": True,
            "quality_slo_risk_reviewed": True,
        },
        "tradeoff": {
            "cost_delta": -0.18,
            "quality_delta": -0.005,
            "slo_delta": -0.002,
            "risk_score": 0.1,
            "max_quality_drop": 0.01,
            "max_slo_drop": 0.005,
            "max_risk_score": 0.2,
        },
        "unit_economics": {
            "value_per_unit": 0.10,
            "cost_per_unit": 0.018,
        },
        "platform_gate": {
            "enabled": True,
            "owner": "finops-ai-infra",
            "budget_policy": True,
            "optimization_backlog": True,
            "approval_records": True,
            "audit_ready": True,
            "p0_open": False,
        },
    }


def build_bad_cases(good_case):
    cases = []

    case = deepcopy(good_case)
    case["case_id"] = "usage_metering_missing_bad"
    case["usage_metering"]["token"] = None
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cost_labels_missing_bad"
    case["labels"].pop("cost_center")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "gpu_efficiency_bad"
    case["gpu"]["idle_gpu_hours"] = 40
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "training_waste_bad"
    case["training_waste"]["failed_jobs"].append({"gpu_count": 4, "hours": 4, "price_per_hour": 32.0})
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "inference_unit_cost_bad"
    case["inference"]["cost"] = 130.0
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cache_savings_bad"
    case["cache"]["permission_safe"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "storage_lifecycle_bad"
    case["storage"]["retention_policy"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "network_egress_bad"
    case["network"]["remote_gib"] = 4000
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "artifact_dependency_bad"
    case["artifact_delete"]["rollback_dependency"] = True
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "budget_quota_bad"
    case["budget"]["actual_cost"] = 9300.0
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cost_anomaly_alerting_bad"
    case["alerts"]["owner_routed"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "chargeback_bad"
    case["chargeback"]["model_version"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "dashboard_drilldown_bad"
    case["dashboard"]["drilldown"].pop("request_slice")
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "recommendation_trace_bad"
    case["recommendation"]["quality_slo_risk_reviewed"] = False
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "slo_quality_tradeoff_bad"
    case["tradeoff"]["quality_delta"] = -0.05
    cases.append(case)

    case = deepcopy(good_case)
    case["case_id"] = "cost_governance_gate_missing_bad"
    case["platform_gate"]["enabled"] = False
    cases.append(case)

    return cases


audit = MiniAICostGovernanceAudit()
good = build_good_case()
cases = [good] + build_bad_cases(good)
summary = audit.run_all(cases)

print("cost_governance_examples=" + repr(audit.example_outputs(good)))
print("metrics=" + repr(summary["metrics"]))
print("hard_blocker_count=" + repr(summary["hard_blocker_count"]))
print("failed_cases=" + repr(summary["failed_cases"]))
print("failed_gates=" + repr(summary["failed_gates"]))
print("cost_governance_gate_pass=" + repr(summary["cost_governance_gate_pass"]))
```

参考输出应类似：

```text
cost_governance_examples={'gpu_job_cost_usd': 2560.0, 'tokens_per_gpu_hour': 5000000.0, 'failed_job_cost_usd': 160.0, 'cost_per_1k_tokens_usd': 0.018, 'cache_saved_cost_usd': 54.0, 'storage_monthly_cost_usd': 2890.0, 'network_egress_cost_usd': 108.0, 'budget_used': 0.72, 'idle_gpu_hours': 12, 'unit_margin': 0.082, 'cost_center_count': 3}
metrics={'usage_metering_coverage': 0.941, 'cost_attribution_labels': 0.941, 'gpu_cost_efficiency': 0.941, 'training_waste_control': 0.941, 'inference_unit_cost': 0.941, 'cache_savings_accounting': 0.941, 'storage_lifecycle_governance': 0.941, 'network_egress_governance': 0.941, 'artifact_dependency_safety': 0.941, 'budget_quota_enforcement': 0.941, 'cost_anomaly_alerting': 0.941, 'tenant_model_chargeback': 0.941, 'dashboard_drilldown_readiness': 0.941, 'optimization_recommendation_trace': 0.941, 'slo_quality_cost_tradeoff': 0.941, 'cost_governance_gate': 0.941}
hard_blocker_count=16
failed_cases=['usage_metering_missing_bad', 'cost_labels_missing_bad', 'gpu_efficiency_bad', 'training_waste_bad', 'inference_unit_cost_bad', 'cache_savings_bad', 'storage_lifecycle_bad', 'network_egress_bad', 'artifact_dependency_bad', 'budget_quota_bad', 'cost_anomaly_alerting_bad', 'chargeback_bad', 'dashboard_drilldown_bad', 'recommendation_trace_bad', 'slo_quality_tradeoff_bad', 'cost_governance_gate_missing_bad']
failed_gates=['usage_metering_coverage', 'cost_attribution_labels', 'gpu_cost_efficiency', 'training_waste_control', 'inference_unit_cost', 'cache_savings_accounting', 'storage_lifecycle_governance', 'network_egress_governance', 'artifact_dependency_safety', 'budget_quota_enforcement', 'cost_anomaly_alerting', 'tenant_model_chargeback', 'dashboard_drilldown_readiness', 'optimization_recommendation_trace', 'slo_quality_cost_tradeoff', 'cost_governance_gate']
cost_governance_gate_pass=False
```

这个 demo 的面试价值是把成本治理从“月底账单分析”升级为工程控制闭环：先采集 usage，再强制 attribution，然后按 GPU、训练、推理、缓存、存储、网络和 artifact 拆成本，接着用预算、配额、异常告警和 dashboard 下钻定位责任方，最后用质量 / SLO / 风险门禁决定优化建议能不能落地。

## 48.23 常见误区

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

## 48.24 面试常见追问

问题一：AI Infra 成本主要来自哪里？

可以回答：主要来自 GPU 训练和推理、模型和 checkpoint 存储、数据和权重传输网络成本、评估和 embedding 生成、日志 trace 留存，以及人工标注和平台运维。

问题二：如何降低推理成本？

可以回答：通过小模型优先、模型路由、缓存、prompt 压缩、输出长度控制、量化、continuous batching、自动扩缩容、warm pool 优化和单位 token 成本监控降低成本。

问题三：如何做成本归因？

可以回答：所有资源、任务、模型、endpoint、artifact 都必须带 tenant、team、project、owner、model、job_type、cost_center 等标签，再结合 usage metering 归因到租户、团队、模型和任务。

问题四：成本优化和 SLO 冲突怎么办？

可以回答：成本优化必须在质量和 SLO 约束下进行。可以分层服务，不同业务使用不同 SLO、模型、缓存、配额和降级策略，而不是一刀切省钱。

## 48.25 小练习

1. GPU hours 和 tokens per GPU hour 分别说明什么？
2. 为什么 failed job GPU hours 是重要成本指标？
3. 推理 cost per 1k tokens 受哪些因素影响？
4. Checkpoint retention policy 如何设计？
5. 网络成本在 AI Infra 中有哪些来源？
6. 为什么成本标签必须强制填写？
7. 如何设计租户级推理 token 预算？
8. 成本优化如何避免伤害 SLO？

## 48.26 本章小结

本章讲了 AI Infra 成本治理。

你需要记住：

1. 成本治理的目标是让资源消耗可观测、可归因、可预算、可优化。
2. GPU、推理、存储、网络、评估和 trace 都可能产生显著成本。
3. 成本归因依赖统一标签、用量采集和成本中心。
4. 训练成本要关注 GPU hours、失败任务、tokens/GPU hour 和资源利用率。
5. 推理成本要关注 cost per request、cost per 1k tokens、cache 命中率和模型路由。
6. 成本优化必须和 SLO、质量、安全和回滚能力一起权衡。

下一章我们会讲资源利用率优化：混部、抢占、弹性训练和低优先级队列。
