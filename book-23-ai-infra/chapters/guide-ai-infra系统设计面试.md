# 第七章：AI Infra 系统设计面试

## 7.1 系统设计题考什么

AI Infra 系统设计题不只是让候选人画出 API、数据库和 GPU。面试官通常要判断候选人能否把业务目标翻译成资源约束，能否识别控制面和数据面，能否在可靠性、成本、性能、安全和迭代速度之间做取舍。

一个可迁移的回答顺序是：

```text
目标与用户 -> 工作负载 -> SLO/容量 -> 核心状态
-> 数据/控制路径 -> 资源调度 -> 故障恢复
-> 观测/安全 -> 取舍与演进
```

先讲清问题，再画架构；如果没有工作负载和验收指标，架构图很容易变成组件名清单。

## 7.2 先问清楚约束

至少要澄清：用户是谁、任务是训练/推理/评估还是数据处理、峰值和平均流量、输入输出长度、同步还是异步、可接受延迟、成功标准、数据敏感级别、GPU 类型和预算、是否允许抢占、失败后能否重试、需要保存多久以及是否有多租户。

假设题目是“设计一个高并发推理平台”，没有输入长度、输出长度和 TTFT/TPOT 目标，就不能计算 GPU 数量；假设题目是“设计训练平台”，没有任务规模、checkpoint 大小和恢复目标，就不能选择存储和容错策略。

## 7.3 容量估算

推理的输入 token 吞吐可以粗略写成：

```math
R_{in}=\lambda\cdot T_{in}
```

输出 token 吞吐为：

```math
R_{out}=\lambda\cdot T_{out}
```

其中 `\lambda` 是请求到达率。若 reasoning 或工具调用使平均输出长度增加，`R_out` 会随之增加；若上下文包含历史和检索结果，`T_in` 也会增加。

KV cache 的粗略占用为：

```math
M_{KV}\approx2BLTH_{kv}d_hb
```

训练数据读取需求则可写成：

```math
B_{read}\ge W\cdot r\cdot s
```

面试时不必假装这些公式能给出精确机器数，但要说明假设、余量和需要压测校准的参数。

## 7.4 练习题一：设计训练平台

题设：公司有多个团队提交预训练、SFT 和评估任务；任务使用不同 GPU，可能需要多机通信；平台要支持 quota、优先级、抢占、checkpoint 恢复、日志、指标和权限隔离。

核心对象可以设计为：

```text
Job, Attempt, ResourceAllocation, DatasetManifest,
Checkpoint, Artifact, MetricStream, AuditEvent
```

控制面保存 Job 状态和策略；调度器根据资源、拓扑、quota 和优先级分配节点；执行面负责容器、launcher、worker heartbeat 和故障传播；数据面提供代码、数据、checkpoint 和日志。

状态要区分 `QUEUED`、`ALLOCATING`、`RUNNING`、`PREEMPTED`、`RESTORING`、`SUCCEEDED` 和 `FAILED`。每个 attempt 有独立输出目录，checkpoint 使用 manifest 和 checksum，恢复时只读完整提交的版本。

关键 trade-off：

1. 抢占提高集群利用率，但需要可靠 checkpoint，且恢复会消耗 I/O 和重复计算。
2. 高优先级能降低紧急任务延迟，但必须用 aging 或 quota 防止低优先级永久饥饿。
3. 共享文件系统简化路径，但吞吐和故障域可能成为瓶颈；对象存储需要缓存和重试。
4. 强隔离提升安全性，但可能降低资源利用率；混部需要更严格的调度和监控。

验收指标包括队列等待、GPU 利用率、有效 step 吞吐、失败恢复时间、checkpoint 恢复成功率、quota 违约率、租户数据隔离和任务成功率。

## 7.5 练习题二：设计高并发推理平台

题设：请求长度差异很大，需要流式输出，部分任务需要工具调用和结构化 JSON，平台支持多个模型版本，并要求质量、延迟和成本可观测。

请求路径可以画成：

```text
Gateway -> Auth/Quota -> Router -> Tokenizer/Template
-> Scheduler -> Prefill/Decode -> KV Manager
-> Sampler/Grammar -> Stream -> Trace/Billing
```

Scheduler 同时管理 token budget、sequence budget 和 KV block budget；continuous batching 让完成请求退出、新请求进入；prefix cache 只复用版本、权限和 token 序列兼容的 prefix；streaming 只提交已经通过验证的 token。

路由记录模型 revision、engine、template、sampling、降级原因和 cache 命中。故障时可以回滚模型或降低 reasoning budget，但不能跳过权限、结构化校验和安全策略。

验收指标包括 TTFT、TPOT、p95/p99、请求成功率、格式正确率、工具成功率、KV 利用率、cache hit、单位成功任务成本和错误分类。

## 7.6 练习题三：设计评估与实验平台

评估平台的关键不是把模型输出写进 CSV，而是建立可追溯的 run：

```text
dataset revision + model revision + harness revision
+ prompt/template + tool config + output + judge + error slice
```

离线批评估承担回归，人评承担开放质量，在线评估承担真实任务和长期行为。发布条件应采用质量、安全、协议、成本和延迟的合取条件，而不是只比较一个平均分。

对于生成式任务，应保存错误样例和证据引用。对于 RAG，还要记录检索候选、rerank、权限过滤和 source version；对于 Agent，要保存每一轮 tool call、参数校验、结果、重试和人工接管。

## 7.7 可靠性、安全和回滚

任何系统设计答案都应说明至少三类故障：依赖故障、资源故障和数据/协议故障。比如推理平台要处理模型 worker OOM、GPU 节点不可用、tokenizer/template 不匹配、工具超时、流式连接中断和 cache 污染；训练平台要处理 rank hang、对象存储失败、checkpoint 半写入和版本漂移。

安全边界要独立于模型输出。身份、租户、数据访问和高风险工具由策略服务或执行器决定；模型只能提出候选动作。每个 allow/deny、回滚、人工确认和管理员访问都应写入审计事件。

## 7.8 如何讲取舍而不是罗列组件

一个成熟回答会把方案和约束绑定：

```text
因为输入长度长且流量变长，我按 token 和 KV budget 调度；
因为任务可恢复，我用 checkpoint manifest 和 attempt 隔离；
因为高风险工具有副作用，我把权限判断放在 executor；
因为质量和成本同时重要，我用模型路由和单位成功任务成本做验收条件。
```

不要只说“用 Kubernetes、Redis、Kafka、向量库和 GPU”。每个组件都要回答：它保存什么状态、谁拥有状态、失败如何恢复、是否需要幂等、指标是什么以及为什么适合当前负载。

## 7.9 一份可复用的回答骨架

面试现场可以按以下结构组织答案：

1. 复述目标、用户和 SLO。
2. 给出工作负载假设和容量估算。
3. 画控制面、数据面和执行面。
4. 说明核心对象、状态机和资源预算。
5. 解释主链路的读写、调度和版本契约。
6. 选择两个最危险的失败模式并给出恢复流程。
7. 补充权限、审计、监控、成本和回滚。
8. 明确当前方案的边界，以及流量或质量变化后的演进方向。

这份骨架的价值在于让答案可检查。每一步都能被追问，也都能落到指标或测试，而不是停留在口号。

## 7.10 回答系统设计题的验收闭环

完整回答应从需求、工作负载、数据流、状态、容量、故障、安全、成本和指标收束到上线条件。罗列组件不如说明一个请求如何进入、如何消耗资源、如何失败和如何恢复。

面试中给出公式后还要说明假设：平均/峰值 QPS、输入输出长度、并发、SLO、冗余和增长率。假设变化时，指出哪个组件先成为瓶颈。

## 7.11 面试回答的容量推导

系统设计题先给 workload：请求数、输入/输出长度、并发、SLO、峰值比例、模型版本和数据/工具依赖。再估算权重、KV/state、workspace、网络和存储，而不是直接画组件图。

一个简化显存预算是：

~~~math
M_{\mathrm{total}}
=M_{\mathrm{weights}}+M_{\mathrm{KV/state}}
+M_{\mathrm{activation}}+M_{\mathrm{workspace}}
+M_{\mathrm{fragmentation}}
~~~

所有数字都应说明是教学估计还是压测结果。

## 7.12 失败优先的设计

面试中主动回答 GPU 故障、节点失联、数据坏 shard、模型加载失败、KV 耗尽、grader 不可用、权限错误和发布回滚，通常比罗列更多中间件更有价值。每个故障要说明检测、隔离、降级、恢复和数据一致性。

Agent 和推理系统还要讨论副作用、幂等、checkpoint、state rollback 和租户隔离。高风险动作不能通过自动重试无限放大。

## 7.13 取舍和验证

组件选型要围绕 workload 和组织约束：托管服务减少运维但降低控制，开源 runtime 提高可调试性但需维护，实时系统追求 p99 而离线系统追求吞吐。最后给出压测、故障演练、质量回归和灰度方案。

## 7.14 把面试回答组织成证据链

系统设计回答可以按五步展开。先写清 workload 和 SLO，再画请求或训练任务的生命周期，接着做资源估算，随后说明状态、协议和权限边界，最后用压测和故障演练证明设计。每一步都要指出一个可观测指标和一个失败出口。

例如高并发推理题不能只画 API、router 和 GPU。应继续追问输入/输出长度、流量峰值、模型副本、KV 预算、队列策略、短长请求隔离、stream 取消、模型升级和 N+1。若题目带工具，还要加入 idempotency、确认门和副作用审计。

## 7.15 一个可复用的容量推导

假设峰值请求率为 Q，平均输入/输出 token 为 L_in/L_out，目标 GPU 有效吞吐为 R_pre/R_dec，安全利用率为 u，则可先估计：

~~~math
N_{\mathrm{pre}}
\ge\frac{Q\,L_{\mathrm{in}}}{uR_{\mathrm{pre}}},
\qquad
N_{\mathrm{dec}}
\ge\frac{Q\,L_{\mathrm{out}}}{uR_{\mathrm{dec}}}.
~~~

再检查每副本权重、KV/state、workspace、通信和故障余量。这个估算不要求面试中得到精确数字，但必须说明假设、单位和哪个变量会让结果改变一个数量级。

## 7.16 失败优先的回答方式

每个核心组件至少给出一个失败：数据坏 shard、GPU OOM、通信 hang、模型加载失败、KV 耗尽、评估服务不可用、权限错误、客户端断流或工具半成功。回答检测、隔离、降级、恢复和一致性，而不是只说“重试”。

重试还要有边界。无副作用的读取可以有限重试；写操作要先查执行状态；训练任务要依赖 checkpoint；模型服务要防止 stream 重复；安全策略服务失败时通常 fail-closed 或进入受限路径。

## 7.17 worked example：从容量到故障的完整回答

题目是“设计一个支持 1M context、流式输出和工具调用的企业推理平台”。一个完整回答先声明：峰值请求率、输入/输出分布、TTFT/TPOT、租户隔离、工具副作用和 N+1 目标都需要确认；在缺少数据时给出区间，而不是直接报 GPU 数。

接着把请求拆成 preflight、route、admission、prefill、decode、tool、verify 和 stream。preflight 用目标 tokenizer 计算模板展开后的 token；admission 同时检查权重、KV/state、输出预算和队列 SLO；scheduler 用 continuous batching，但对长上下文和交互请求分池；工具动作经过 executor 的权限、schema、幂等和审批门；stream 只提交已验证事件。

容量上先估算：

```math
R_{in}=Q L_{in},
\qquad
R_{out}=Q L_{out},
\qquad
M=M_{weights}+M_{state}+M_{workspace}+M_{slack}.
```

然后说明 `R_in` 需要 profile prefill，`R_out` 需要 profile decode，`M_state` 取决于并发、上下文、KV layout 和精度。最后给出验证：混合长度压测、1M 证据检索、断流恢复、工具半成功、GPU OOM、N+1、权限回归和灰度回滚。这样回答中的每个组件都对应一个资源、状态、指标和失败出口。

## 7.18 追问检查：把“会设计”变成“能验收”

面试官继续追问时，可以用四个问题自检：

1. 当前状态的权威源是什么？如果 worker 崩溃，谁能恢复？
2. 这个动作消耗哪种资源？资源耗尽时如何拒绝或降级？
3. 这个结论由哪个 verifier、样本或 trace 证明？
4. 如果请求已部分执行，重试会不会产生重复副作用？

例如“用 cache 降低延迟”需要继续说明 cache key、版本、租户隔离、失效、命中后的权限和错误回退；“用多 Agent 提高质量”需要说明额外 token、共享状态、合并 verifier、相关错误和单位成功成本。追问的重点不是组件数量，而是系统是否能在边界条件下保持可解释。

## 7.19 面试后的验证方案

把设计落成实验时，先做 tiny workload 验证状态机，再做混合长度压测，最后做故障注入和灰度。记录假设、输入分布、模型 revision、配置、代码 commit、指标和结论；若压测推翻了原估算，应更新模型，而不是修改结果来迎合架构图。

AI Infra 系统设计的核心能力是把模型工作负载翻译成资源、状态、协议和治理约束。架构图只是表达工具，真正决定答案质量的是假设透明、容量可算、失败可恢复、权限独立、指标可验证。

练习：为 8 卡训练任务设计抢占恢复；为 1M context 请求设计 token/KV/成本约束；为工具推理服务画流式提交与回滚状态机；为模型分数下降设计从数据版本到线上版本的血缘查询。可参考 Kubernetes、PyTorch Distributed、主流推理引擎、OpenTelemetry、Google SRE 和 NIST AI RMF 的公开资料，但公开默认值不能替代容量压测、故障演练和安全评估。

## 7.20 先建立资源类型账本

AI Infra 题最容易出现的错误，是把所有资源都说成“GPU 不够”。一个训练或推理系统至少同时消耗计算、显存、主机内存、网络带宽、对象存储吞吐、队列容量、日志/trace 配额和人工审核预算。它们的瓶颈并不总是同一时间出现。

对一个请求或任务，可以记录：

```text
compute: prefill/decode/forward/backward seconds
state: KV pages or recurrent state bytes
network: all-reduce, RPC, tool and storage bytes
storage: dataset/checkpoint/cache bytes and IOPS
control: queue slots, retries, leases and rate limits
governance: review, audit and retention budget
```

例如 1M context 请求可能先受 prefill 计算限制，随后受 KV page 和队列等待限制；训练任务可能先受通信拓扑限制，checkpoint 时又受对象存储带宽限制。容量公式应写出资源单位，不能只给一个“需要 N 张卡”的结论。

## 7.21 控制面、数据面和执行面的边界

控制面负责声明任务、策略、配额和状态；数据面负责传输数据、权重、KV、checkpoint 和 trace；执行面负责真正运行 kernel、worker、executor 或评估程序。三者都可能成功或失败，不能用一个 `job_status=success` 代替。

一次训练任务可能处于“控制面已创建、执行面启动、数据面读取失败”；一次推理请求可能处于“模型已生成 proposal、工具尚未确认、客户端已经断流”。系统设计要为这些中间状态命名，并说明谁是权威状态源、谁可以重试、谁负责补偿。

可以把状态转移抽象成：

```math
s_{t+1}=\delta(s_t,e_t),
\qquad
\mathrm{commit}(e_t)\Rightarrow e_t\text{ 可审计且不可重复产生副作用}.
```

这个抽象同时适用于 checkpoint、工具调用、模型发布和流式输出。它把“系统设计”从服务名词列表变成了可恢复的状态机。

## 7.22 容量估算必须写假设和余量

假设峰值到达率为 `Q`，每请求输入/输出 token 均值为 `L_in/L_out`，并发峰值为 `B`，则输入输出 token 速率分别为：

```math
R_{in}=Q L_{in},
\qquad
R_{out}=Q L_{out}.
```

但真实容量还要乘上长度分布、cache 命中、reasoning、工具轮次、重试、N+1 故障和 p99 余量。一个实用估算是：

```math
N\ge
\left\lceil\frac{R}{uR_{\mathrm{profile}}}\right\rceil
 +N_{\mathrm{failover}},
```

其中 `u` 是目标利用率，不应取 1；`R_profile` 必须绑定模型、engine、硬件、dtype、batch 和 workload。若输入长尾是主导因素，均值估算会严重低估 KV 峰值，应该按 p95/p99 输入长度重新计算。

## 7.23 用证据闭合设计回答

成熟的系统设计回答不止说“使用某框架”，而是为每个关键决策提供证据：容量由压测和公式支持，质量由 paired sample 和 verifier 支持，安全由权限回归和红队样本支持，恢复由故障演练和 replay 支持，成本由账本支持。

证据对象应绑定版本：模型权重、tokenizer、模板、数据、索引、工具 schema、策略、runtime 和硬件。否则一次性能回归无法判断是模型更新、cache 失效、模板膨胀还是调度变化。设计图、实验报告和线上 trace 应共享 `run_id/release_id`，让结论可以从用户请求追到具体 artifact。

这也是面试中“讲得像做过”的关键：不是堆更多组件，而是说明每个组件解决哪个约束、留下哪个指标、失败后如何恢复，以及什么证据会让你改变原先的选择。
