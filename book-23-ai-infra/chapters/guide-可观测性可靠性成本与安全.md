# 第六章：可观测性、可靠性、成本与安全

## 6.1 四类信号和一个生产问题

AI Infra 事故通常不是“模型完全挂掉”这么简单，更多时候表现为训练吞吐下降、推理尾延迟上升、答案质量回归、GPU 成本失控或敏感数据进入错误租户。要解释这些问题，需要把 metrics、logs、traces 和 events 连接起来：

```text
metrics：发生了多少、变化趋势是什么
logs：某次请求或某个 worker 具体说了什么
traces：一次任务跨哪些组件、每段花了多久
events：版本、扩缩容、抢占、节点故障和策略变更何时发生
```

没有 trace 的指标只能看到症状，没有指标的日志很难发现趋势，没有 events 的 trace 很难解释“为什么此刻变了”。

## 6.2 从用户目标到 SLI/SLO

可靠性指标应从用户可感知的任务开始，而不是从机器指标开始。推理服务可以关注成功请求率、TTFT、TPOT、p99、引用正确率和工具成功率；训练平台可以关注任务成功率、恢复时间、有效 GPU 小时和 checkpoint 恢复成功率。

设窗口内请求数为 `N`，满足业务成功条件的请求数为 `N_good`，可用性 SLI 可以写成：

```math
SLI=\frac{N_{good}}{N}
```

如果 SLO 是 99.9%，允许的错误预算约为窗口请求量的 0.1%。错误预算不是“可以故意失败的额度”，而是决定团队能否继续激进发布、实验或需要优先修复可靠性问题的管理信号。

## 6.3 延迟和资源指标的关联

推理请求的端到端延迟可以拆成：

```math
T_{e2e}=T_{queue}+T_{prefill}+T_{decode}+T_{postprocess}
```

当 p99 上升时，必须按阶段分解。GPU utilization 上升可能意味着有效吞吐变好，也可能意味着长 prompt、cache miss 或无界重试把设备压满。训练平台的 step time 也要拆成 compute、communication、I/O、checkpoint 和 stall。

常见的时间序列标签包括 model revision、engine revision、tenant、GPU type、prompt length bucket、output length bucket 和 error class。标签不能无限增加，否则 metrics cardinality 会造成监控系统自身失控；完整请求级信息应放到 trace 或采样日志。

## 6.4 Trace 设计和隐私边界

一个 LLM 请求 trace 可以包含 gateway、router、retriever、model server、tool executor 和 evaluator span。每个 span 记录开始/结束时间、状态、版本、输入输出大小、重试次数和错误分类；原始 prompt、文件内容和工具结果要按敏感等级决定是否脱敏、采样或只保存 hash/reference。

trace ID 应贯穿 API、队列、GPU worker、外部工具和异步任务。没有稳定 correlation ID，排查“用户看到的答案为什么慢/错”时就无法把前端请求和后端执行拼起来。

## 6.5 成本模型

单位成本不能只除 GPU 价格。对一次成功任务，可以粗略写成：

```math
C_{success}=\frac{C_{gpu}+C_{cpu}+C_{storage}+C_{network}+C_{eval}+C_{human}}{N_{success}}
```

`N_success` 是满足业务完成标准的任务数，而不是 API 调用数。reasoning、tool calling、重试和人工接管都会增加分子；质量回归会降低分母。因此一个更便宜但任务失败率更高的模型，单位成功任务成本可能反而更高。

推理成本还要按 prefill/decode、输入/输出 token、KV cache、模型精度、并发、缓存命中率和空闲容量分解。训练成本则要记录有效 GPU 小时、失败重跑、checkpoint I/O、低利用率等待和评估占用。

## 6.6 事故响应闭环

一个可执行的事故流程包括：检测、分级、止血、定位、修复、验证、复盘和防复发。止血动作可以是回滚模型、关闭某个路由、降低并发、暂停训练、禁用高风险工具或切换人工处理；但每个动作都要记录影响范围和恢复条件。

MTTD（平均发现时间）和 MTTR（平均恢复时间）可以帮助判断响应效率：

```math
MTTR=\frac{\sum_{i=1}^{n}(t_{recover,i}-t_{detect,i})}{n}
```

复盘不应只写“人为疏忽”。要指出触发条件、未被发现的信号、缺失的验收条件、错误的默认值、责任边界和可自动化的防线，并用回归测试或演练证明修复有效。

## 6.7 安全控制面

AI Infra 安全至少包含身份认证、最小权限、租户隔离、密钥管理、数据加密、供应链、模型访问控制、日志审计和高风险动作确认。模型输出是数据，不是权限凭证；一个生成的 SQL、shell 命令或工具参数必须经过独立的策略和参数校验。

安全决策可以抽象为：

```math
Allow(a,r,e)=Auth(a)\land Policy(a,r,e)\land Risk(r,e)\le\tau
```

`a` 是 actor，`r` 是资源或动作，`e` 是环境，`Risk` 是策略引擎计算的风险。风险验收条件应独立于模型是否“看起来很自信”，并在日志中记录 allow/deny 的原因。

## 6.8 供应链和发布条件

容器镜像、CUDA/驱动、Python 依赖、模型权重、tokenizer、adapter 和评估脚本都属于供应链。发布前应记录 digest、来源、扫描结果、许可证、漏洞状态和兼容性测试；生产运行时要限制任意下载和任意网络访问。

模型发布准入条件可以同时检查质量、安全、协议、资源和回滚：

```math
Gate=Q\land Safety\land Protocol\land Resource\land Rollback
```

即使模型质量提升，只要工具权限、数据隔离或回滚包不完整，也不应进入生产。

## 6.9 一个“延迟变慢”的联合排查

线上 p99 TTFT 突然增加，监控显示 GPU 利用率也上升。排查顺序是：

1. 按模型、版本、租户和 prompt length 分桶，确认影响范围。
2. 用 trace 区分 queue、prefill、router 和 gateway 时间。
3. 检查发布 events，确认是否启用了 reasoning、换了量化或降低了 cache 命中。
4. 查看 KV block 使用、preemption、重试和长请求比例。
5. 先执行低风险止血，如回滚版本或限制超长输入，再验证 p99 是否恢复。
6. 用固定 workload 重放，判断修复是改善真实瓶颈还是只改变了流量。

如果只把 GPU 利用率降下来，可能牺牲吞吐而没有解决队列；如果只扩大实例，可能把成本推高而无法恢复 p99。可观测性必须服务于可验证的因果判断。

## 6.10 指标之间的因果链

tokens/s 下降可能来自队列、KV 压力、kernel fallback、网络通信或工具等待。监控要把 request、batch、GPU、cache、模型和工具 trace 关联起来，避免只盯一个平均指标。

安全和隐私还要求字段级采集：可以记录 token 数和错误类型，不代表可以长期保存原始 prompt、检索片段或工具参数。观测 schema 本身也要有权限和 retention。

## 6.11 四类信号如何串起来

Metrics 适合趋势和 SLO，logs 适合局部错误，traces 适合跨服务时间线，events 适合发布、权限和状态变更。训练、推理、RAG 和 Agent 应使用统一 request/job/trace/model/data id，否则成本和错误无法串联。

高基数标签要克制，隐私字段要脱敏，原始 prompt、工具参数和媒体要按风险和保留策略处理。可观测性不能成为新的数据泄漏面。

## 6.12 SLO、错误预算和模型质量

AI 服务的 SLO 不只有 latency/availability，还应包括任务成功率、引用支持、结构化输出、工具副作用和安全漏放。错误预算可以决定何时暂停发布、减少流量或触发人工复核。

质量指标的统计窗口、样本分布和 grader 版本要固定。线上质量下降可能来自模型、检索、数据、用户分布或工具，而不是单一服务错误。

## 6.13 成本和可靠性的共同治理

单位请求成本可以拆成 GPU、存储、网络、检索、工具、重试和人工审核。降低单次 token 价格但增加失败重试，可能提高单位成功成本。成本 trace 要和质量、风险以及租户配额绑定。

故障演练应覆盖 GPU、网络、存储、模型加载、KV 池、外部工具和策略服务，确认系统能降级、回滚、告警和恢复，而不是只验证健康检查。

## 6.14 SLI、SLO 与错误预算

不同服务阶段需要不同 SLI。交互式推理至少看 TTFT、TPOT、p99、错误率、取消率和结构化输出成功率；Agent 还要看任务成功率、工具副作用错误率、人工接管率和单位成功成本；训练平台则看 step success、resume success、checkpoint RPO/RTO 和数据等待时间。

可以把错误预算写成：

~~~math
B_{\mathrm{error}}
=1-\mathrm{SLO}_{\mathrm{target}},
\qquad
\mathrm{burn\ rate}
=\frac{\mathrm{observed\ error}}{B_{\mathrm{error}}}.
~~~

错误预算被快速消耗时，发布策略应收紧、流量降级或冻结变更，而不是继续用平均成功率掩盖尾部故障。不同租户和风险等级可以有不同目标，但定义必须在入口和 trace 中一致。

## 6.15 Trace 的最小字段与隐私

一条推理 trace 需要把 API、router、queue、prefill、decode、cache、tool、policy 和 response event 串起来。建议保存 request id、tenant、model revision、prompt revision、token bucket、状态转移、耗时、错误码、cache handle、工具 call id 和最终 finish reason。

原始 prompt、检索片段、工具参数和媒体默认不应进入普通日志。使用哈希、脱敏字段、短期受控采样和字段级访问策略，既能支持排查，又不把可观测平台变成新的数据泄露面。

## 6.16 事故响应的证据顺序

“延迟变慢”不是根因。排查顺序应先确认影响范围和发布时间，再按 queue wait、prefill/decode、batch、KV/state、GPU memory、kernel fallback、网络、工具和外部依赖分解。每一层都需要相应的 trace/span 和对照 workload。

事故记录要包含时间线、检测信号、用户影响、最后一次正常版本、缓解动作、回滚决定、状态恢复和后续行动。若只是扩大实例而没有定位原因，可能在下一次长请求峰值中再次失败。

## 6.17 成本和安全的联合治理

成本观测要按租户、模型、输入/输出 token、GPU 时间、cache 命中、工具和人工介入归因；安全观测要按身份、权限、风险、策略版本、拒绝/放行和副作用归因。两者共享 trace id，但访问权限和保留周期不应完全相同。

单位成功成本可以写成：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{compute}}+C_{\mathrm{tool}}+C_{\mathrm{review}}}
{N_{\mathrm{accepted\ safe\ tasks}}}.
~~~

若系统为了降低成本关闭验证，安全事故或人工修复成本会在账本中反弹。因此成本优化必须与质量、安全和错误预算一起做验收条件。

## 6.18 统一事件模型：把四类信号连成状态机

metrics、logs、traces 和 events 的区别不是存储格式，而是观察角度。一个 `request_id` 的生命周期可以产生 `admitted`、`prefill_started`、`tool_denied`、`stream_closed` 等事件；每个事件同时更新计数器、产生 span，并在需要时写入详细日志。

统一事件至少包含：

```text
event_id, trace_id, parent_id, subject_id
tenant_id, model_revision, data_revision
event_type, timestamp, outcome, policy_revision
resource_cost, redacted_attributes
```

这样可以把“p99 变慢”与某次发布、模型 revision、长 prompt、cache miss 或策略重试关联起来。高基数原文不能直接作为 metric label；敏感字段应保存受控引用或哈希，并由权限系统决定谁能回读。

## 6.19 故障注入与错误预算动作

可靠性不是健康检查为绿色，而是依赖失败时仍能满足预先定义的边界。推理服务可以注入 worker OOM、KV allocator 失败、模型加载超时、策略服务不可用、外部工具半成功和客户端断流；训练平台可以注入 rank hang、checkpoint 半写入和对象存储暂时不可用。

每个演练都要写出检测信号、自动动作、用户可见结果、数据一致性和恢复条件。例如策略服务超时时，高风险写操作应 fail-closed 或进入人工路径；只读检索可以使用有时效标记的缓存，但不能把缓存结果伪装成最新事实。错误预算消耗到阈值后，应冻结发布或减少流量，不能只增加告警数量。

## 6.20 统一案例：质量下降但延迟变好

一次量化发布后，p99 TTFT 从 1.8 秒降到 1.2 秒，但引用支持率从 93% 降到 86%。如果只看性能仪表盘，发布是成功的；如果只看质量均值，又可能忽略影响集中在高风险租户。正确排查要对齐 model revision、quantization、retriever、prompt、数据切片和引用 verifier，并比较相同样本的逐条结果。

发布准入条件可以要求：

```math
G_{\mathrm{release}}
=I(\Delta Q\ge-\epsilon_q)
 I(\mathrm{p99}\le\tau_l)
 I(\mathrm{unsafe\ leak}=0)
 I(\mathrm{rollback\ ready}=1).
```

这里的 `epsilon_q` 和 `tau_l` 由业务风险确定；对高风险任务，质量下降容忍度可能为零。这个例子说明性能、质量、安全和成本不是四个互不相干的 KPI，必须使用同一版本与样本证据做联合判断。

## 6.21 生产验收的基础验收条件

上线前要做指标缺口检查、隐私抽样、权限审计、故障注入、回滚演练、成本对账和质量回归。Metrics、logs、traces、events 不是四个互不相干的仪表盘，而是同一状态机的不同投影。

可观测性、可靠性、成本与安全的完整闭环，应把 request/job、model/data revision、trace、resource、quality、permission 和 incident 关联起来。OpenTelemetry、MLflow、Kubernetes 和模型服务文档可以支持实现细节，但 SLO、数据保留和事故责任必须由具体系统定义。

## 6.22 先定义事件和指标的实体契约

观测数据的第一问题不是选 Prometheus 还是日志系统，而是每条信号究竟描述什么实体。训练平台的实体可能是 `job`、`attempt`、`rank`、`checkpoint`；推理平台的实体可能是 `request`、`iteration`、`tool_call`、`kv_handle`；发布平台的实体可能是 `release`、`evaluation_run` 和 `gate`。

每条事件至少要能回答：发生在谁身上、哪个版本、哪个租户、哪条状态转移、消耗了什么资源、结果是什么。指标负责聚合趋势，trace 负责一次请求的因果链，日志保存需要回看的细节，事件保存状态变化。若没有稳定的 `entity_id` 和 `revision`，四种信号即使数量很多，也无法连成一个故障故事。

## 6.23 高基数、采样和隐私的三方约束

把 `prompt`、完整 URL、用户 ID 或工具参数直接作为 metric label，会造成高基数和隐私泄露；完全不记录原文，又可能无法复现一条关键失败。实际系统通常把聚合字段与受控 artifact 分开：metrics 使用长度桶、任务类型和哈希；trace 保存脱敏后的结构和引用；原文放在有 ACL、retention 和删除流程的存储中。

采样也不能只按请求随机抽样。长请求、错误请求、权限拒绝、工具半成功、低置信输出和尾延迟请求应提高采样率；正常短请求可以降低采样率。可以把采样权重写成：

```math
p_{\mathrm{keep}}=\mathrm{clip}(p_0
 +\alpha I_{\mathrm{error}}
 +\beta I_{\mathrm{risk}}
 +\gamma I_{\mathrm{tail}},0,1).
```

这不是隐私策略本身，仍需经过数据分级和访问审批；它只是让有限存储优先保留最有诊断价值的证据。

## 6.24 SLO 应绑定用户可见结果

TTFT、TPOT、GPU 利用率是重要指标，但不能独立代表可靠性。对一个带工具的 Agent，请求可能模型延迟正常，却因工具超时、重复执行或引用不支持而失败；对训练任务，step time 正常，却因 checkpoint 不可恢复而失去实验价值。

因此 SLO 应分层：可用性、延迟、质量、协议、安全和恢复分别有门槛。错误预算也要按严重度加权，而不是把一次安全越权与一次普通超时视为同一个错误：

```math
E_{\mathrm{risk}}=\sum_i w_i n_i,
\qquad w_{\mathrm{security}}>w_{\mathrm{latency\ timeout}}.
```

权重必须由业务和安全责任人确定。超过预算后的动作可以是冻结发布、降低高风险能力、扩大人工审核或切换 fallback，而不是简单地让告警变红。

## 6.25 成本、质量和安全要使用同一分母

只按 token 计费会漏掉 cache、工具、人工、重试和回滚成本；只按请求计费会掩盖长上下文和 reasoning 的差异。更有用的单位是“成功且安全的任务”：

```math
C_{\mathrm{unit}}
=\frac{C_{\mathrm{gpu}}+C_{\mathrm{storage}}+C_{\mathrm{network}}
 +C_{\mathrm{tool}}+C_{\mathrm{review}}}
 {N_{\mathrm{quality\ pass\ and\ safe}}}.
```

分子和分母都必须绑定模型、版本、租户和 workload。若关闭 verifier 后 token 成本下降，但安全拒答错误和人工修复增加，单位成功成本可能反而上升。成本优化应在质量和安全 hard gate 通过后进行。

## 6.26 事故复盘要从症状追到控制点

复盘“引用率下降”时，不应停在模型输出。应按时间线查询发布、模板、数据/索引、retriever、模型、策略、cache、worker 和 verifier；按样本切片判断问题集中在哪些租户、长度、语言、风险和工具路径；最后确认哪个控制点本应阻止错误进入用户可见结果。

一个好的事故报告包含影响范围、检测延迟、证据链接、状态机转移、临时缓解、根因、长期修复、回归样本和回滚判断。不要把“加强监控”作为唯一修复；如果错误是未授权工具执行，应补权限门和幂等检查，如果是 cache 版本混用，应修 artifact manifest 和 admission。

观测系统的最终目标不是收集最多字段，而是让团队能在有限时间内做出正确动作：识别真实影响、阻止继续扩散、保留证据、恢复服务并把样本沉淀为下一次发布的验收条件。

## 6.27 小结

AI Infra 可观测性必须把指标、日志、trace、事件、质量、安全和成本放在同一个请求与版本语义下。SLO 和错误预算要按风险分层，单位成本要使用成功且安全的分母，事故复盘要追到具体控制点并生成回归验收条件；否则监控只能描述症状，不能支持恢复和决策。
