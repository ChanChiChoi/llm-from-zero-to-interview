# 第六册第十三章：生产系统设计深度精修研究记录

- 日期：2026-10-27
- 正文：[第六册第十三章](../book-06-llm-deployment/chapters/13-生产系统设计.md)
- 依据：当前 [WRITING_SPEC.md](../WRITING_SPEC.md)、[progress_v2.md](../progress_v2.md) 与仓库 AGENTS.md。
- 范围：逐节复读正文 §1—§16，核对既有概念、公式、示例、系统运行案例、来源和配套导航。重点从生产可靠性与 serving 的一手工程资料中补足“具体机制—证据—适用边界”，不把文档建议或合成 demo 写成目标服务验收。

## 原知识点覆盖图

| 原章范围 | 保留内容 | 本轮补深或校正 |
| --- | --- | --- |
| §1 生产系统目标 | 模型只是请求链路的一环；在线、流式、异步；生命周期图和六类契约 | 将组件责任落实到请求、资源、质量、安全、状态和发布契约；用失败边界解释为什么单独调用模型不等于交付任务。 |
| §2 需求与服务目标 | workload 分类、SLI/SLO、端到端延迟、质量与安全目标 | 增加可复算的 95%/3.5 秒/30 天 SLI 和 10,000 请求中的 500 次 bad-event budget；要求事先固定 eligible 请求、测量边界、取消分类和窗口；质量/安全硬门禁独立于 availability error budget。 |
| §3 容量规划 | 输入/输出 token/s、Little's Law、active sequence、副本、KV 显存、N+1、成本 | 明确 `A_m` 是 runtime 活跃序列，`J_m` 是端到端 job 并发；重试/fallback/并行路径计入模型调用数；prefill、decode、序列容量要在相同混合负载下测。复核输入吞吐例为 18,000 token/s（非 72,000）；与 §11 示例统一 active-sequence 算式。 |
| §4 请求链路 | Gateway、租户/数据边界、限流、路由、上下文预算、四种缓存、runtime 边界、RAG/Agent/多模态 | 修正 token bucket：先补充至容量上限，只有准入时扣预估预算；输出长度未知时先预留、按实际 usage 结算。说明前缀缓存 key 和租户隔离、错误缓存及显存占用的取舍。 |
| §5 可靠边界 | API/流式协议、内部原因、timeout/cancel/retry、commit-unknown | 增全抖动退避、deadline 传播、重试预算和嵌套重试放大（两层各三次总尝试最多九次）；按错误类型决定是否重试；写工具超时保留操作 id 并查询状态。对照幂等键、Transactional Outbox 和 Saga 的原子边界与未解决问题，不承诺端到端 exactly-once。 |
| §6 可观测性 | logs/metrics/traces、RAG/Agent span、指标分母与诊断 | 增 Prometheus label cardinality 的运行代价；高基数 request/user/prompt 信息放受控 trace/log，不进 metrics 标签。将 OTel GenAI `Development` 状态、TTFT 与首 token 后 TPOT 的计量边界写清，区分服务端与客户端 E2E。 |
| §7 评估与发布 | release manifest、offline/shadow/canary、质量切片、回滚 | 增 control/canary 分组和归因例：5% 候选流量的 20% 错误可被全服务聚合稀释到约 1%（假设 control 无错且流量可比）；质量、安全、工具副作用、成本和尾延迟并列作 gate。 |
| §8 故障与过载 | 过载诊断、公平调度、重试/熔断、断线取消 | 补熔断三状态与重试的不同作用，阈值需按依赖、样本量和 workload 配置；取消必须传播到 runtime 释放序列/KV。 |
| §9 扩展 | 横向副本、多 GPU、多区域与状态所有权 | 补 Kubernetes startup/readiness/liveness 的职责差异；将队列或依赖短暂过载设成 liveness 失败可能导致重启级联，过载首先通过 admission/backpressure 处理。 |
| §10 多租户成本 | 多维配额、单成功任务成本、Autoscaling | 保留资源量纲与内部预算换算边界；强调权限和资源额度不可互相替代，扩容需考虑冷启动和 warm pool 成本。 |
| §11 完整架构案例 | FAQ/RAG/退款 Agent、容量推导、失效、发布与回滚 | 保留 35/12/3 QPS 等教学 workload。将任务并发 70/74.4 与 runtime 活跃序列约 35.3/54.9 分开；large pool 主限制为 decode，5 副本的 N+1 输出容量为 7,200 token/s。明确这些数字不证明端到端 SLO。 |
| §12 标准库 demo | 三类 workload、两种模型池、容量/清单/预算门禁和暂停结论 | prefill 服务时间统一用 `effective_prompt / prefill_tps`，不再和池吞吐表使用冲突的固定 `0.04 ms/token`。增加独立 `design_components` 输入，集合逐项核对而非只比数量；把声明型清单与真实 P95/任务质量/安全证据分开。 |
| §13 系统运行日 | 早高峰、P99、候选发布、RAG 故障、退款提交、租户公平六个案例 | 保留贯穿案例，增加分阶段诊断、证据状态和动作理由；trace 示例字段改为 `cache_saved_token_ratio`，避免读成请求级 hit rate。 |
| §14 误区、§15 资料边界、§16 小结 | QPS、引擎、缓存、重试、安全、灰度和均值误区；来源与总结 | 使误区对应到前文的公式、方案、失败条件和测量方法；来源按论文、官方工程文档、框架实现和教学假设区分。 |

## 代表性工程来源：机制、所解决的问题与边界

| 来源 | 解决的问题 / 实际机制 | 本章采用的结论 | 限制与使用范围 |
| --- | --- | --- | --- |
| [Google SRE: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/) | 让用户体验指标成为有分母、窗口、目标和误差预算的操作契约；讨论 SLI/SLO 与发布决策的关系。 | SLI 必须定义请求范围、事件和计量窗口；未达 SLO 的预算可控制常规发布速度。正文用 10k 请求和 95% SLO 演算。 | 书中的组织实践不是所有产品必须照抄的阈值；错误预算不能替代任务质量和安全硬门禁。 |
| [Google SRE Workbook: Canarying Releases](https://sre.google/workbook/canarying-releases/) | 用有限期部分流量部署候选并保留 control，以差异观察候选回归；强调代表性、足够样本和归因。 | 全服务指标会稀释小流量 canary 的问题，必须分组比较同一时段和可比 workload。 | 5%×20%=1% 是流量加权示意，不证明统计显著性；LLM 还需任务质量和高风险动作切片。 |
| [Revisiting SLO and Goodput Metrics in LLM Serving](https://arxiv.org/abs/2410.14257) | 研究 LLM serving 中 request/token 粒度的目标与吞吐指标，提出 smooth-goodput 作为研究型平滑效用视角。 | 区分 raw tokens/s 与满足服务目标的用户任务吞吐；指标分子、分母会改变 scheduler 被奖励的行为。 | smooth-goodput 不是行业统一 SLO；不替代业务定义的延迟、完成和质量硬门槛。论文讨论受作者 workload 与定义约束。 |
| [AWS Builders' Library: Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/) | 让客户端为一次语义意图提供稳定 token；服务端保存请求 token 和结果，识别重放，并防止重复副作用。 | 重试需使用同一操作键；键与规范化参数不匹配应拒绝；去重记录与副作用要有足够原子协调和保留期。 | 仅在入口生成 UUID 不足以保证业务幂等；若下游没有可查询/可去重语义，仍需未知状态和人工处置。 |
| [AWS Builders' Library: Timeouts, retries, and backoff with jitter](https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/) | 限制调用者等待、退避并打散同步重试，避免瞬时错误变成过载反馈环。 | 采用全抖动上界、端到端 deadline、retry budget；多层重试次数相乘，取消也需下传。 | retry 不能解决持久错误；重试本身增加负载，也不能推断已提交写操作已回滚。超时阈值要从实际延迟分布和依赖行为校准。 |
| [Microsoft Transactional Outbox pattern](https://learn.microsoft.com/en-us/power-platform/architecture/reference-architectures/transactional-outbox-pattern) | 通过同一数据库事务保存业务状态与待发布消息，再由异步 dispatcher 投递，避免数据库写入和消息发布之间的双写丢失。 | dispatcher 可能重复投递，模式提供至少一次；消费者必须按 operation/event id 去重。 | outbox 不把多个数据库/外部退款服务变成一个分布式原子事务，不等于端到端 exactly-once。 |
| [Azure Architecture Center: Microservices design patterns](https://learn.microsoft.com/en-us/azure/architecture/microservices/design/patterns) | 区分 Retry、Circuit Breaker、Bulkhead 和 Saga 的职责；前两者常组合，Saga 以局部事务和补偿管理跨服务工作流。 | 重试针对短暂故障，熔断器阻止持续向坏依赖施压；Saga 补偿是业务动作，不保证恢复到原状。 | 模式清单不是阈值建议；错误类别、窗口、最小样本、冷却时间、半开探测和补偿流程需由目标业务设计。 |
| [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/) | 分别提供 startup（初始化完成前等待）、readiness（是否接流量）和 liveness（是否需重启）的信号。 | 权重加载期间用 startup；暂不可接流量时从路由摘除；只有进程确实失去活性才触发 liveness restart。 | probe 只能表达健康策略，不能单独证明模型推理正确；liveness 错设为依赖/队列健康可能引发重启级联。 |
| [Prometheus instrumentation practices](https://prometheus.io/docs/practices/instrumentation/) | 指出每个 label value 组合会创建 time series，并提供控制维度和基数的实践。 | request_id、user_id、邮件或 prompt 等高基数字段留在访问受控的 trace/log，不作为指标标签。 | 不是所有 label 都禁用；应按受控枚举和可接受序列规模设计，敏感信息还需独立隐私治理。 |
| [OpenTelemetry GenAI metrics conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) | 给模型服务请求时长、TTFT 和首 token 后 TPOT 等指标提供共享命名约定。 | 正文区分 server-side 与 client E2E 边界，并记录该页面当前标记为 `Development`。 | 约定仍可能变化，runtime/exporter 不一定全部实现或采用相同边界；它是语义参考，不是性能或兼容性认证。 |
| 既有 [vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics.html)、[Kubernetes HPA](https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/) 与 [OpenTelemetry primer](https://opentelemetry.io/docs/concepts/observability-primer/) | 分别提供 LLM serving 指标入口、基于指标比例建议副本、以及 logs/metrics/traces 基础概念。 | 作为 engine 级监控、控制器机制和观测术语的实现入口。 | 动态文档和实现版本需固定；HPA 默认指标不能直接代表 GPU/KV/队列需求，文档也不等于目标 deployment 实测。 |

本章生产可靠性主题没有用单篇 LLM 系统论文去替代通用分布式系统机制。论文和系统研究（如 SLO/goodput、serving scheduling、KV 管理）用于解释 LLM 负载和指标；SRE/云厂商/Kubernetes/Prometheus/OpenTelemetry 的官方资料用于核验操作语义。二者证据层级与用途不同。

## 公式与数字审计

- SLO 例：10,000 个符合资格的同步请求，SLO=0.95，bad-event budget 为 `(1−0.95)×10,000=500`。超时、服务端断连和用户取消的分母归属要事先写入指标契约。LLM serving 的 goodput 应与 raw tokens/s 分开，并按业务目标明确计数口径。
- Canary 例：候选占 0.05、候选错误率 0.20、control 错误率 0 时，加权总错误率贡献为 0.01；这是聚合稀释的算术，不是显著性结论。
- 生成时间：输出 O≥1 token 时，`TTFT+(O−1)TPOT`；若 TTFT 已覆盖排队和 prefill，就不能再把相同阶段加一次。
- token bucket：余额先按 `min(Bmax, B+rΔt)` 补充；仅准入请求扣除估算成本；否则排队/拒绝，不能让预算变负。
- 重试：全抖动等待 `U(0,min(tmax,t0·2^k))`；两层各允许 3 次总尝试，嵌套最坏 9 次，故须共享总 deadline/预算。
- 案例输入负载：small 输入/输出 25,200/6,300 token/s；large 39,600/6,120 token/s。RAG job 并发 38.4，Agent job 并发 36；总 job 并发 74.4 不等于模型序列。
- demo 当前假设下，small `τmodel=4.8+110+895=1,009.8ms`，active sequence `35×1.0098≈35.3`；large RAG `τmodel≈2,204.3ms`、并发约 26.5；Agent 单模型调用约 3,156.6ms，3 QPS×3 次调用约 28.4；large 合计约 54.9。`model_path_estimate_ms` 输出 chat/RAG/Agent 为 1,105/2,299/9,755 ms。所有时间参数是教学假设。
- 副本：small 的 decode 比值 6,300/7,000=0.9，乘 1.2 余量后需 2 个副本；large decode 比值 6,120/1,800=3.4，ceil(4.08)=5。large 故障一副本后为 7,200 token/s，可覆盖演示输出需求，但仅核对 decode，不覆盖恢复时延/网络/KV 重建。
- `model_path_estimate_ms` 不包含真实队列、检索、工具、网络和长尾分布，不能替代 3.5 秒 end-to-end P95。安全与质量证据在 demo 中明确缺失，决策为 `pause_and_investigate`。

## 代码运行和配套同步

- 提取 §12 唯一标准库 Python demo，`compile` 并执行；核对 stdout 与相邻的 `~~~text` 示例输出逐字一致。
- `PAPERS.md` 增生产可靠性与服务治理官方资料入口；`GLOSSARY_EN_ZH.md` 增 SLI/SLO、幂等键、Outbox、commit-unknown 和任务并发/活跃序列口径，并细化错误预算定义。
- `INTERVIEW_BANK.md` 增生产容量、SLO、事务恢复和 canary 归因面试题；`EXERCISES.md` 增六项容量/SLO/事务/canary/demo/probe 练习。第六册目录已有第十三章导航，无需改动。
- 外部来源页面已按本轮资料核验；未对全章既有所有 URL 执行自动 HTTP 扫描，未执行外部公式渲染、真实压测或 PDF 构建。

## 未解决边界

无目标模型、GPU、请求 trace、队列分布、检索/工具真实延迟、线上 P95、任务成功标签、隐私流程或安全评估数据。正文与 demo 不宣称可直接用于生产 sizing，也不证明目标 SLO、质量、安全或成本。论文和框架指标受各自 workload、硬件、版本及测量边界约束；OTel GenAI 语义约定、vLLM/HPA 文档需按部署版本复核。
