# 第四章：大模型推理平台工程

## 4.1 从模型调用到在线服务

离线调用模型只需要输入 token、执行 forward 和取出输出；在线推理平台还要处理并发请求、变长 prompt、流式输出、取消、限流、版本、权限、故障和成本。平台的基本路径可以表示为：

```text
请求 -> 鉴权/限流 -> tokenizer/template -> 路由
     -> waiting queue -> scheduler -> prefill/decode
     -> KV cache -> sampler/grammar -> stream -> 监控与计费
```

每个箭头都是可能改变行为的协议边界。tokenizer 或 chat template 不匹配会让模型输入改变；scheduler 只按请求数而不按 token 数，会让长 prompt 挤压短请求；streaming 已经发出 token 后才发现工具参数非法，会造成客户端无法回滚。

## 4.2 TTFT、TPOT 和端到端延迟

对流式生成，用户看到的体验至少拆成：

```math
T_{e2e}=T_q+T_{prefill}+T_{first\_decode}+T_{remaining\_decode}
```

首 token 延迟通常称为 TTFT，后续 token 间隔可以用 TPOT 描述。粗略地，若输出有 `N_out` 个 token，平均 token 间延迟为 `t_token`，则：

```math
T_{e2e}\approx TTFT+(N_{out}-1)\cdot TPOT
```

这个式子提醒我们，优化 prefill 主要改善 TTFT，优化 decode、KV cache 和 batching 主要改善 TPOT；只报平均 tokens/s 可能掩盖用户已经等待很久才看到第一个 token 的事实。线上还要按 prompt 长度、输出长度、模型路由和租户切分 p50/p95/p99。

## 4.3 Prefill、decode 和 KV cache

Prefill 处理整段输入，计算量随 prompt 长度增加；decode 每次通常只输入一个或少量新 token，但要读取历史 KV。标准显式 KV 的教学估算是：

```math
M_{KV}\approx2BLTH_{kv}d_hb
```

`B` 是并发序列数，`L` 是层数，`T` 是每条序列的上下文长度，`H_kv` 是 KV head 数，`d_h` 是 head dimension，`b` 是每元素字节数。实际 serving 还要考虑 block metadata、对齐、量化 scale 和不同架构的 state layout。

因此 scheduler 同时受 token budget 和 KV block budget 约束：

```math
N_{prefill}+N_{decode}\le B_{token}
```

```math
B_{used}+B_{new}\le B_{KV}
```

只满足第一条而不检查第二条，可能在执行中 OOM；只满足第二条而让超长 prefill 占满 token budget，又会导致已有流式请求 TPOT 抖动。

## 4.4 Continuous batching 和请求状态

静态 batch 要等最长请求完成，变长生成会造成大量 padding 和等待。Continuous batching 在每个 iteration 重新选择 waiting、running 和 finished 请求，让完成请求退出、新请求进入，并按 token 和 cache 预算生成执行计划。

一个请求至少要维护：

```text
request_id, input_ids, generated_ids, phase,
block_table, sampling_params, grammar_state,
stop_reason, cancel_state, usage, trace_id
```

`phase` 可以是 `WAITING/PREFILL/DECODE/FINISHED/ABORTED`。请求完成后必须释放活动 KV 引用；prefix cache 的共享 block 不能因为一个请求结束就无条件释放。取消请求也要区分“尚未执行”和“外部工具或副作用已经发生”的情况。

## 4.5 路由、缓存和版本契约

模型路由可以根据质量、成本、延迟、上下文长度、模态、工具能力和租户策略选择后端。路由决策应记录实际 model revision、template revision、sampling 参数和降级原因，否则线上结果无法解释。

Prefix cache 适合复用完全相同的 token prefix，但命中必须绑定 tokenizer、template、模型 revision、LoRA adapter、租户权限和多模态输入版本。语义缓存可以复用相似问题的答案，却有时效性、权限和个性化风险，不能把它和 KV cache 混为一谈。

## 4.6 Streaming 和协议正确性

流式响应不只是把字符串切片发送。服务端要处理 item 开始、增量 token、工具参数、完成、错误、取消和 usage；客户端看到的内容应当只包含已经提交的 token。

推测解码或 grammar 约束存在临时候选时，服务端必须区分 `candidate` 与 `committed`。如果候选被 target 拒绝，相关 KV、grammar state、usage 和 stream event 都要回滚。客户端不能先看到候选，再期待服务端撤回。

## 4.7 容量规划示例

假设一个服务平均每个请求输入 4,000 token、输出 500 token，峰值到达率为 20 requests/s，目标 p95 TTFT 为 1 秒。容量估算不能只用 `20` 作为 batch size，还要估算每秒的 prefill token：

```math
R_{prefill}=20\times4000=80{,}000\ mathrm{tokens/s}
```

如果开启 reasoning，平均输出增到 2,000 token，则 decode token rate 从 `10,000` 增到 `40,000 tokens/s`。同一业务流量下，推理预算和 KV 占用都发生变化。若工具调用平均每任务增加 3 轮，每轮都产生新的输入和输出，单位请求成本也不能按一次模型调用估算。

正确的压测应使用真实长度分布、长尾输入、并发、取消、缓存命中率、工具/grammar workload 和多租户混合流量，记录 TTFT、TPOT、p99、GPU 利用率、KV 使用率、排队时间、错误率和单位成功任务成本。

## 4.8 限流、降级和灰度

限流单位可能是 requests/s、input tokens/s、output tokens/s、并发序列数或 KV blocks。只按请求数限流会让一个百万 token 请求和一个短问题占用同样额度；更合理的策略是按 token 和资源成本加权。

降级可以包括降低最大输出、关闭高成本 reasoning、切换小模型、降低并发、排队或返回人工接管。降级必须保留安全验收条件和协议兼容性；为了降延迟而跳过权限过滤或引用校验不是合格的降级。

灰度发布应同时绑定模型、engine、tokenizer、template、量化和路由配置。旧版本和新版本要用相同的离线回归、线上护栏指标和错误样例对照；发现质量或安全回归时要能按 revision 快速回滚。

## 4.9 常见故障及排查顺序

TTFT 上升时，先看队列等待、prefill token、batch token budget 和长请求比例；TPOT 上升时，先看 decode 并发、KV 带宽、抢占和 GPU 访存；OOM 时，检查 cache block 估算、碎片、共享 prefix 引用和临时候选；输出格式错误时，检查 template、tokenizer、grammar parser 和 stream commit。

“GPU 利用率 95%”也不代表服务健康。高利用率可能来自大量无效重算、长尾请求或频繁 preemption；必须与任务成功率、尾延迟、错误率、缓存命中率和成本一起看。

## 4.10 serving 的资源与协议一致性

推理平台同时管理请求状态、KV/state、模型版本、流式响应和工具副作用。scheduler 重新排队时，request owner、position、block table、sampling 参数和 stream sequence 必须一起迁移。

线上回归应有 golden request、长上下文、并发、取消、超时、preemption 和版本滚动测试。只检查最终文本，发现不了流式重复、usage 错误或 cache 错位。

## 4.11 推理平台的阶段化资源画像

Prefill 主要处理输入上下文，适合大矩阵并行，受输入长度和 KV 写入影响；decode 每轮生成少量 token，更容易受 KV 带宽、kernel launch、batch 不规则和调度影响。流式输出还要加入首 token、首包和网络发送。

平台要把 TTFT、TPOT、吞吐、p99、峰值显存、KV bytes、队列等待和错误率分开记录。平均 latency 不能替代长请求和高并发下的尾延迟。

## 4.12 模型发布和运行时契约

推理平台需要把权重、config、tokenizer、chat template、quantization、adapter、runtime、GPU capability 和 safety/eval report 绑定成 release manifest。权重能加载只是第一道门，协议错误可能让模型输出角色标记、错误 JSON 或错位的 position。

发布流程应包含离线 smoke test、结构化输出、长上下文、工具、量化、性能、灰度和 rollback。旧 KV/cache 不能因为 token 前缀相同就跨 model revision 复用。

## 4.13 调度和容量验收条件

调度器要根据输入长度、预计输出、优先级、KV 预算和 SLO 决定 admission、batch、preemption 或拒绝。高优先级短请求和长 reasoning 请求混在一起时，公平性和吞吐会产生冲突。

可以按 workload 分池，例如低延迟交互、长上下文、批量离线和工具 Agent；但分池会降低共享资源效率。选择要由真实请求分布、SLO 和故障演练决定。

## 4.14 请求生命周期必须有可恢复状态

一个在线请求至少经历 admitted、tokenized、prefill、decoding、waiting-for-tool、streaming、completed、cancelled、failed 等状态。状态机的价值在于把“服务暂时没返回”与“已经执行外部副作用”区分开来。每次状态变化要记录 request id、model revision、owner worker、logical token position、KV handle、deadline 和 retry count。

可以把服务提交抽象为：

~~~math
s_{t+1}
=\mathrm{transition}(s_t,e_t)
\quad\text{subject to}\quad
\mathrm{CommittedOutput}_{t+1}
\supseteq\mathrm{CommittedOutput}_{t}.
~~~

输出只能向前提交，不能让客户端先看到候选 token 后再撤回。请求取消时，未提交的 candidate、temporary KV 和工具 proposal 必须清理；已经提交的工具动作则要通过幂等查询或补偿流程处理。

## 4.15 token budget 与 state budget 的双验收条件

调度器每轮至少同时管理计算预算和状态预算。设本轮最多执行 B_tok 个 token，最多新增 B_page 个 KV page，则：

~~~math
\sum_i \Delta T_i\le B_{\mathrm{tok}},
\qquad
\sum_i \Delta P_i\le B_{\mathrm{page}}.
~~~

只控制 token 会在长上下文请求上 OOM；只控制 page 会让 prefill 计算挤压短请求。chunked prefill、continuous batching、speculative verify 和工具结果写回都要纳入同一轮预算。

## 4.16 批处理、公平性与尾延迟

continuous batching 提高吞吐的前提是请求可以在 iteration level 加入、暂停和退出。服务需要在吞吐和公平性之间取舍：长请求持有大量 KV，短请求如果永远排在后面会出现年龄倒置；高优先级租户如果无上限，会耗尽所有 page。

可采用按队列年龄、请求剩余工作量、租户配额和风险等级的组合策略。评估时分别压测短输入/短输出、长输入/短输出、短输入/长输出和长输入/长输出，记录 TTFT、TPOT、queue wait、batch size、page utilization 和 p99。

## 4.17 版本契约与故障演练

模型版本不只是权重。tokenizer、chat template、special token、position scheme、quantization、KV layout、grammar、sampling 和 API event schema 都可能变化。prefix cache、speculative artifact 和已排队请求要明确是否跨版本兼容；不兼容时排空、重算或拒绝，不能静默复用。

故障演练至少覆盖 GPU OOM、worker 重启、模型加载失败、cache checksum 错、客户端断流、工具半成功、策略服务超时和滚动升级。每个场景写出检测、隔离、可见结果、状态恢复和是否允许重试。

## 4.18 worked example：一次长上下文工具请求

假设一个用户提交 80K token 的文档分析请求，要求流式返回结论，并在发现缺失字段时调用只读数据库工具。请求进入网关后，系统先计算模板展开后的输入长度和 4K 输出预留；如果它通过长上下文池的 admission，scheduler 为 prefill 预留 token budget，为后续 decode 预留 KV page。

prefill 完成后，模型输出一段可见草稿和一个 tool proposal。proposal 不能立刻执行：先验证 item 是否完整、参数是否符合 schema、数据库权限是否有效，再把只读查询提交给 executor。工具返回结果后，结果带数据版本和覆盖范围重新进入上下文，模型继续 decode。此时短请求不能因为长请求占满一个 batch 就永久等待，scheduler 需要按 iteration 释放或切片长请求。

若客户端在工具返回后断流，服务端要区分三种状态：模型 token 尚未提交、只读工具已经完成、业务响应尚未发送。重连可以查询 response id 并补发事件；若模型已经产生外部写入 proposal，则必须进入 policy 和 idempotency 流程。若长请求被抢占，只有在 KV handle、logical position 和 sampling state 一起保存时，才能选择恢复或重算。

这个例子把许多“独立模块”连在了一起：tokenizer/template 决定真实长度，admission 决定是否接入，KV allocator 决定能否继续，stream protocol 决定客户端看到什么，tool executor 决定副作用，trace 则把每个边界连接起来。只优化其中一个模块，可能把瓶颈或风险转移到另一个模块。

## 4.19 容量实验与发布条件

容量不是一个静态数字。至少要用四类 workload 做对照：短输入短输出、长输入短输出、短输入长输出和长输入长输出；再分别加入 prefix cache、reasoning、工具轮次、取消和长尾请求。每一类记录 TTFT、TPOT、p99、KV page 使用、队列等待、GPU 利用率和单位成功任务成本。

可以定义一个最小服务验收条件：

```math
G_{\mathrm{serving}}
=G_{\mathrm{protocol}}
 G_{\mathrm{quality}}
 G_{\mathrm{latency}}
 G_{\mathrm{state}}
 G_{\mathrm{safety}}
 G_{\mathrm{rollback}}
```

`G_protocol` 检查 stream event、结构化输出和 usage；`G_quality` 检查任务和引用；`G_latency` 检查分桶后的 SLO；`G_state` 检查 KV、取消和恢复；`G_safety` 检查权限与副作用；`G_rollback` 检查旧版本、cache 和排队请求如何处理。平均 tokens/s 再高，只要 `G_state=0`，就不能发布。

容量压测还要包含 N+1 和滚动发布：一台 worker 失效后，剩余容量是否仍能满足关键 SLO；新版本加载时，旧权重、warmup、cache 冷启动和灰度流量是否同时占用资源。压测结论要绑定模型、engine、量化、GPU、驱动和请求分布，不能把某次机器上的数字写成通用性能承诺。

## 4.20 平台验收的基础结论

上线前的验收应把功能、性能、质量和安全放在一起：golden token replay、结构化输出、取消恢复、混合长度压测、N+1 演练、权限回归、成本账本和灰度回滚缺一不可。一个服务即使 tokens/s 很高，只要 stream event 顺序错误或工具副作用可能重复，就不能称为可靠的 serving 平台。

推理平台的核心是请求状态、调度预算、KV 生命周期、协议正确性、版本治理和可观测性。vLLM、SGLang、TensorRT-LLM 等引擎提供不同实现路径，但平台设计不能只复制项目类名，必须依据 workload、硬件、质量验收条件和 SLO 实测。可参考 vLLM Quantization、主流推理引擎、API streaming 和 Kubernetes 服务治理文档；具体参数与性能必须绑定版本、硬件和 workload。

## 4.21 请求状态要覆盖“等待外部结果”

普通文本请求可以近似为 admitted、prefill、decode、completed；工具 Agent 还要加入 waiting-for-tool、tool-running、tool-verified、streaming、cancelled 和 compensation。每个状态要写清输入、输出、owner、deadline、KV/state handle、已提交事件和是否允许重试。

状态机可以保证一个重要不变量：

```math
\mathrm{CommittedOutput}_{t+1}
\supseteq\mathrm{CommittedOutput}_{t}.
```

客户端已经看到的 token 或 tool event 不能因为 speculative、重试或 worker 切换被撤回。候选 token 只能在 target/grammar/policy 确认后提交；外部写操作即使模型请求超时，也要通过 idempotency key 查询结果，而不是盲目重做。

## 4.22 token、state 和队列是三个预算

请求的输入输出 token 预算决定计算，KV/state 预算决定显存，队列预算决定等待和公平。设本轮请求集合为 `I`，则至少要同时满足：

```math
\sum_{i\in I}\Delta T_i\le B_T,
\qquad
\sum_{i\in I}\Delta S_i\le B_S,
\qquad
Q_{\mathrm{wait}}\le B_Q.
```

只控制 token，长上下文可能 OOM；只控制 state，prefill 可能挤压交互请求；只控制 GPU 利用率，尾延迟和租户公平可能失控。admission 应使用模板展开后的 token、输出预留、reasoning/工具预算和历史 cache 命中率，而不是用户原始字符数。

## 4.23 路由和 cache 的一致性

模型路由不仅按价格和平均质量选择。它还要考虑 tokenizer、chat template、position scheme、quantization、tool schema、grammar、cache layout 和安全策略是否兼容。prefix cache 命中后，权限和版本仍需重新验证；不能因为 token 前缀相同，就认为请求可以共享全部语义状态。

cache key 至少应绑定模型 revision、tokenizer/template、position/quantization schema、租户/权限域和 pattern revision。失效时要区分删除、重算、排空和降级；滚动升级期间旧 cache、旧排队请求和新权重同时存在，必须把资源余量纳入容量计划。

## 4.24 质量、协议和副作用的发布条件

推理平台的 benchmark 不能只测 tokens/s。golden replay 要比较输出、stop reason、stream event、usage、结构化合法率、引用、工具参数和安全策略；压测要比较 TTFT、TPOT、p99、峰值 KV、fallback、取消和重连；故障演练要验证 worker 重启、cache checksum、策略超时和工具半成功。

发布条件可以写成：

```math
G=G_{\mathrm{quality}}G_{\mathrm{protocol}}G_{\mathrm{state}}
 G_{\mathrm{safety}}G_{\mathrm{latency}}G_{\mathrm{rollback}}.
```

平均延迟改善不能抵消一次未授权写操作或已提交 token 回滚错误。高风险工具应在 target 确认、policy 通过和幂等键生成后执行；模型输出的 proposal 永远不等于 executor 的授权。

## 4.25 推理平台的故障回放

最有价值的故障回放不是随机杀进程，而是把请求停在边界：候选已生成但未验证、工具已完成但响应未发送、KV 已迁移但 stream 尚未提交、策略服务超时、模型版本切换时旧 cache 仍在。每个回放都检查客户端看到的事件、外部副作用、状态恢复和是否重复计费。

回放结果应沉淀为 golden trace 和回归测试。这样 serving engine 的改动、量化、speculative decoding、scheduler 调优和路由升级都能在同一套协议与状态验收条件下比较。一个推理平台真正可靠的标志，是它能在正常路径高效，在异常路径明确、可恢复、可审计。

## 4.26 小结

推理平台把模型执行、调度、KV/state、协议、工具权限和发布治理连成一条状态链。平台验收不能只看吞吐，还要验证取消、重试、cache 版本、工具副作用、故障回放和回滚；只有质量、协议、安全、延迟和恢复检查条件同时通过，模型才具备生产可用性。
