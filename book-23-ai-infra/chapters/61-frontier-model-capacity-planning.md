# 第 61 章 Frontier Model Capacity Planning：先算资源，再谈能否上线

## 61.1 新模型接入为什么经常高估容量

模型发布页给出参数量、上下文长度和吞吐宣传，但平台真正要回答的是：多少 GPU 才能装下权重和 KV？长请求会不会挤压短请求？reasoning 和工具调用会把平均输出放大多少？故障和滚动升级需要多少余量？

Capacity planning 不是把一张 GPU 数量表填满，而是把模型、工作负载、服务目标和故障预算放进同一个资源模型。尤其对 MoE、混合 attention、长上下文和 Agent 模型，参数量、active FLOPs 和显存不是同一个数字。

## 61.2 从权重开始

权重存储的粗略估算为：

```math
M_{\mathrm{weights}}=P\times b_w
```

`P` 是实际需要常驻或可加载的参数量，`b_w` 是每个参数的字节数。部署还要加 scale、量化元数据、embedding、LM head、runtime workspace 和复制。

若模型权重需要跨 `n` 张卡分片，理论最低每卡权重约为 `M_weights/n`，但通信、张量并行冗余和可用显存比例会使真实值更大。GPU 标称显存不能全部拿来放权重。

## 61.3 KV、state 和长上下文

显式 KV 的教学估算为：

```math
M_{\mathrm{KV}}\approx2BLTH_{\mathrm{kv}}d_hb_{kv}
```

递归 state、latent cache、local window 和 global layer 需要另行估算。总容量约为：

```math
M_{\mathrm{request}}
=M_{\mathrm{KV}}+M_{\mathrm{state}}+M_{\mathrm{workspace}}+M_{\mathrm{metadata}}
```

一百万 token 的请求即使只占一个 batch slot，也可能消耗远高于普通 32K 请求的状态。容量规划不能只用平均上下文长度，要看 p95/p99 长请求和并发组合。

## 61.4 prefill 与 decode 是两种压力

prefill 处理整段输入，通常受矩阵计算、内存带宽和 context parallelism 影响；decode 每步只产生少量 token，却需要频繁访问 KV、调度和网络。长输入会拖慢 TTFT，长输出会拖慢 TPOT，两者的优化方向不同。

工作负载可表示为：

```text
(input_tokens, output_tokens, concurrency, arrival_rate, task_type)
```

只用 tokens/s 做规划会掩盖短请求 p99 和长请求队头阻塞。应分别压测短/短、长/短、短/长和长/长组合。

## 61.5 MoE 和 active 参数

MoE 的 active 参数更接近每 token 的计算路径，但 total experts 仍可能需要常驻或可快速加载。expert parallel 的 all-to-all 通信、热点专家和 capacity padding 会影响吞吐。

因此资源表至少记录 total weight、active FLOPs、expert dispatch bytes、通信时间、KV/state、batch 和 p99。不能用 active 参数直接决定 GPU 数量。

## 61.6 容量例子

假设一个服务有 8 张 GPU，每张可用于模型和 cache 的显存为 70 GiB。权重分片占每卡 48 GiB，runtime/workspace 和通信预留 8 GiB，则每卡剩余约 14 GiB 给 KV/state。

如果普通请求需要 2 GiB，理论上每卡可容纳 7 个请求；但一个 1M 长请求可能需要 100 GiB 的全局 KV，必须路由到多卡长上下文池或换压缩/检索路径。把普通请求的并发乘到长请求上，会得到完全错误的容量结论。

## 61.7 SLO 与 admission control

服务要先定义 TTFT、TPOT、p95/p99、成功率、取消率和并发 SLO，再反推 admission。超出预算的请求可以排队、降级到 RAG、缩短上下文、切换模型或请求异步任务，但降级要显式记录。

一个长请求不应无限占据资源。scheduler 可以把 long context、high effort 和多工具任务放到专用 pool，避免短请求被队头阻塞。

## 61.8 故障和升级余量

容量不能按 100% 利用率规划。节点故障、滚动升级、模型加载、cache 重建、重试和灰度都需要余量。若目标是 N+1，必须把一张 GPU/一个 worker 下线后的有效容量重新压测，而不是只看静态冗余。

模型 revision、tokenizer、runtime 和量化变更可能同时改变权重、KV 和吞吐，升级要用 shadow workload 验证。

## 61.9 评估与监控

线上至少观测：输入/输出 token、上下文 bucket、KV/state GiB、cache hit、prefill/decode 时间、queue wait、batch size、GPU memory、通信、p95/p99、OOM、取消、fallback 和单位成功成本。

按租户、任务类型、model revision 和 effort 分桶。平均 GPU utilization 很高不代表服务健康，若 queue wait 和 p99 同时上升，说明容量门限已经被突破。

## 61.10 常见失败

用参数量替代容量模型；用平均长度替代尾部；把 active 参数当显存；忽略 KV/state；只测 prefill 或只测 decode；没有 N+1 余量；升级前没有重新压测；把 fallback 计入成功却不计入成本。

## 61.11 面试回答与练习

回答“如何为新 frontier model 做容量规划”时，应从权重、KV/state、prefill/decode、MoE 通信、工作负载分布、SLO、admission、故障余量和成本开始，按长度/并发/effort 分桶压测，不用一个参数量或 tokens/s 下结论。

练习一：给定 GPU 显存、权重、KV 和并发分布，计算可承载的短/长请求组合。

练习二：设计一次 N+1 故障演练。

练习三：列出长请求超预算时的四种显式降级。

### 61.11.1 工作负载向量

容量规划的输入不是一个 QPS 数字，而是一组工作负载向量：

```text
w=(input_tokens, output_tokens, context_length,
   concurrency, reasoning_effort, tool_rate,
   long_request_ratio, multimodal_ratio)
```

同样的 QPS，短问答和百万 token 研究任务对 prefill、KV、队列和带宽的压力完全不同。规划应按 workload bucket 测量，不能用平均 token 把长尾抹掉。

### 61.11.2 从资源到可接纳请求

设单请求权重占用为 `M_w`，KV/state 为 `M_s(T)`，workspace 和通信为 `M_o`，设备可用预算为 `M_cap`，则同时接纳的请求数 `N` 需要满足：

```math
M_w+\sum_{i=1}^{N}M_s(T_i)+M_o
\le M_{\mathrm{cap}}-M_{\mathrm{slack}}
```

还要检查 prefill 算力、decode 算力、网络和队列。显存足够但 prefill 过载，仍然会违反 TTFT；算力足够但 KV 不够，仍然需要拒绝或降级。

### 61.11.3 worked example：混合长度容量

假设服务有 100 个短请求和 2 个长请求的峰值组合。若一个长请求的 prefill 时间相当于 40 个短请求，调度器把两个长请求同时放入统一 batch，短请求的 p99 可能快速上升。可以把长请求放入异步池、使用 chunked prefill 或按预算分片，再比较短请求 SLO、长请求完成时间和总吞吐。

### 61.11.4 故障余量和 admission

容量规划要模拟一张 GPU 或一个 worker 故障、模型加载、滚动升级、KV 泄漏和突发长请求。admission 要在请求进入 GPU 前检查真实 token、预算、租户配额、工具风险和队列，而不是等 OOM 后再重试。拒绝、排队和降级都要向用户返回可解释状态。

## 61.12 容量计划的敏感性分析

新模型接入时至少改变输入长度、输出长度、并发、量化、KV/state、故障余量和流量峰值，画出 GPU 数量、显存和 p99 的敏感性。单点估算无法覆盖真实工作负载。

对 MoE 还要加入专家热点和通信拥塞，对 reasoning/Agent 还要加入思考 token、工具和重试。容量计划必须给出最小、典型和峰值三种场景。

## 61.13 不确定性和峰值规划

frontier 模型发布时常缺少完整的 batch、KV、量化和工具条件。容量规划应使用区间和场景矩阵：短交互、长上下文、长 reasoning、Agent 多轮、批量离线和故障降级分别估算。

对每个场景记录输入、输出、并发、p95/p99 长度、SLO、缓存命中和回退比例。峰值容量不能用平均 token 数代替。

## 61.14 admission、配额和发布条件

当权重、KV/state、workspace 和通信超过预算，系统要排队、限流、缩短上下文、切换模型或拒绝。配额应按租户、优先级和风险分层，不能让一个长任务耗尽整个集群。

模型版本升级时重新测资源和质量，因为 active parameters、cache layout、tokenizer、reasoning budget 和工具调用都可能改变容量。

## 61.15 把容量估算变成资源账本

一个可审计的容量表至少按请求保存四类数字：权重常驻、请求状态、每轮计算和共享开销。不能把所有内存都归入“模型显存”，也不能把所有 token 都归入同一种算力。

~~~text
request_id
model_revision
input_tokens / output_tokens
kv_or_state_bytes
prefill_ms / decode_ms
network_bytes
workspace_bytes
admission_decision
~~~

设设备集群有 G 张卡，每张卡的可用预算为 M_cap，单副本权重和共享 workspace 为 M_fixed，故障与发布预留为 M_slack。对于一组请求，最基本的验收条件是：

~~~math
\sum_{r\in\mathcal{R}}M_{\mathrm{state}}(r)
\le
G\,M_{\mathrm{cap}}
-M_{\mathrm{fixed}}
-M_{\mathrm{slack}}.
~~~

这只是显存条件。还要检查单位时间的 prefill token、decode token、通信字节和外部工具等待；任一资源超过自己的预算，系统都会违反某个 SLO。

## 61.16 混合长度请求的排队效应

容量规划不能只用平均输入长度。假设短请求有 2K 输入、256 输出，长请求有 128K 输入、4K 输出。两个 workload 的 QPS 都是 1，但长请求可能在 prefill 上消耗几十倍计算，并长期占据 KV。把它们混到同一个队列，短请求的等待时间会被长请求的服务时间放大。

可以用简单的工作量 proxy：

~~~math
W_i
=\alpha L_{\mathrm{in},i}
+\beta L_{\mathrm{out},i}
+\gamma M_{\mathrm{state},i}
+\delta B_{\mathrm{network},i}.
~~~

调度器按 W 或分桶后的预算做 admission，并不意味着要把不同请求简单排序。交互式请求需要 TTFT/TPOT，批处理请求需要吞吐，长研究任务可以进入异步池。面试和生产设计都应先说明队列分层，再谈 batch size。

## 61.17 N+1 和滚动升级的真实容量

如果一个服务有 N 张卡，目标是 N+1 容错，必须在一张卡或一个 worker 失效后重新满足关键 SLO。滚动发布期间还要同时容纳旧版本和新版本的权重、warmup 请求、cache 冷启动和灰度流量。静态的“总显存足够”无法证明这件事。

可以把有效容量写成：

~~~math
C_{\mathrm{effective}}
=C_{\mathrm{steady}}
-C_{\mathrm{failure}}
-C_{\mathrm{rollout}}
-C_{\mathrm{fragmentation}}
-C_{\mathrm{headroom}}.
~~~

每一项都应该由演练或历史数据估计。若发布只能在清空全部 cache 后进行，就要把 cache 重建时间和 TTFT 峰值列入发布窗口；若模型有 MoE，还要加入 expert 热点导致的通信余量。

## 61.18 用 admission 保护尾延迟

admission 不是单一的“显存够不够”。请求进入 GPU 前，应依次检查身份和配额、模型版本、输入 token 预算、输出预算、状态预算、队列年龄、风险等级和降级路径。拒绝要有原因码，排队要有预计状态，降级要记录实际使用的模型和证据范围。

一个教学版决策器可以写成：

~~~text
if not quota_ok:
    reject("quota")
elif state_budget > free_state_budget:
    route("long_context_pool")
elif predicted_ttft > ttft_budget:
    route("async_or_smaller_model")
elif risk_requires_confirmation:
    route("human_confirmation")
else:
    admit()
~~~

真实系统还要防止并发请求在检查和提交之间竞争同一块状态预算，因此 admission 与 allocator 必须使用同一个 reservation 事务。

## 61.19 容量计划的 worked example

假设峰值为 12 requests/s，其中 70% 是 4K 输入、512 输出，25% 是 32K 输入、2K 输出，5% 是 256K 输入、4K 输出。平均长度看起来并不夸张，但长请求占用了大量 prefill 和 state。先分别计算输入 token 速率：

```math
R_{in}
=12(0.70\times4K+0.25\times32K+0.05\times256K).
```

这个数比只使用全体平均值更能说明队列压力，因为 256K 请求会在一个较长窗口内持续占用 KV。接着按模型 profile 得到每卡的 prefill/decode 有效吞吐，并用真实长度分布压测，而不是把厂商宣传的单请求 tokens/s 直接乘到 QPS 上。

如果长请求进入交互池，会拉高短请求的 TTFT；把它们切到异步或长上下文池，可能提高交互 SLO，却牺牲长任务完成时间。容量计划不是寻找一个最大 QPS，而是明确每个 workload pool 的目标、配额、降级和拒绝边界。

## 61.20 证据等级与计划变更

容量表中的每个字段都要标注证据来源：模型卡或 API 文档是声明，理论公式是估算，单请求 profile 是局部观测，混合压力测试是服务证据，N+1 故障演练才是余量证据。不同来源不能在同一列里伪装成同等确定性。

当模型升级、tokenizer 改变、reasoning budget 增加、量化替换、工具轮数变化或 cache 命中下降时，计划必须重新计算。可以保存参数敏感性表：

```text
assumption -> observed value -> confidence -> impact -> remeasure trigger
```

这样“容量不够”可以被定位到具体假设，而不是在上线前临时增加机器。容量计划还要跟成本、SLO、故障半径和业务增长预测联动；预留太少导致频繁扩容，预留太多则产生长期空闲成本。

## 61.21 敏感性分析与实验设计

新模型接入前，至少绘制输入长度、输出长度、并发、量化、KV 精度、cache 命中、工具轮数和峰值流量对 GPU 数、p99 与单位成本的敏感性。一次只改一个变量可以看局部影响；再用真实混合 workload 检查变量之间的交互。

容量结论要附带证据等级：模型卡是声明，理论公式是估算，单请求 profile 是局部观测，混合压力测试才接近服务结论，N+1 故障演练才能验证余量。不同模型的参数量、上下文上限和宣传吞吐不能跨版本直接比较。

## 61.22 容量计划是一份可变更的契约

容量表不应只是一个写死 GPU 数的电子表格。每个结论要绑定 workload revision、模型 revision、tokenizer、量化、硬件、引擎、并发、SLO 和证据等级。任何一项改变，都要触发重新 profile 或明确记录“仍然适用”的理由。

可以把一个容量假设表示为：

```text
assumption_id
workload/model/runtime revision
observed value + confidence
capacity impact
remeasure trigger
owner + expiry
```

例如“每卡 decode 吞吐为 1200 token/s”只能在指定 batch、上下文、dtype 和 kernel 下成立。若 reasoning budget 提高、工具轮数增加或 prefix cache 命中下降，这条假设就需要重新验证，而不是继续乘到新的 QPS 上。

## 61.23 从平均请求转向混合 workload 仿真

容量最容易被平均数欺骗。应把请求表示为输入 token、输出 token、并发保持时间、工具轮数、媒体 token、优先级、取消概率和风险动作的向量，然后按比例采样混合流量。短请求和长请求共享一个队列时，还要模拟到达时间和调度策略，因为长 prefill 会阻塞短请求的 TTFT。

一个粗略的输入 token 速率为：

~~~math
R_{\mathrm{input}}=\sum_{k}q_k\lambda_k\bar T_{k},
~~~

其中 `q_k` 是 workload bucket 占比，`\lambda_k` 是该 bucket 的到达率，`\bar T_k` 是平均输入 token。这个式子只是流量起点；真正的容量还要用每个 bucket 的 p95/p99 长度、burst 和状态持有时间压测。

## 61.24 reservation 和 admission 必须原子化

两个请求可能同时看到足够的空闲 KV，然后都被允许进入，实际分配时却发生 OOM 或抢占。解决办法是让 admission、状态 reservation 和 allocator 使用同一事务或可回滚预留：

```text
estimate -> reserve -> recheck policy/SLO -> commit admission
         \-> reject/degrade and release reservation
```

预留还要区分正式 KV、临时 speculative state、媒体 buffer 和输出上限。请求取消、超时、worker 失联时必须释放 reservation；否则监控看到的“空闲显存”与调度器可用容量会逐渐分离。

## 61.25 N+1、滚动升级和故障半径

在线容量不能只按正常状态规划。N+1 要回答丢失一台 worker 后是否还能满足关键 SLO；滚动升级要回答同时 draining 的 worker 数、cache warm-up、模型加载和流量迁移会占用多少余量。若系统在满载时才开始升级，排队和恢复通常会互相放大。

故障演练应分别注入 worker 丢失、长上下文突发、KV allocator 失败、对象存储不可用、网络分区、模型加载失败和 tokenizer 不兼容。检查的不是“还有没有 200”，而是高风险动作是否被保护、短请求是否有公平份额、状态是否可恢复、p99 是否仍在预算内。

## 61.26 容量、SLO 和成本要统一分母

增加预留 GPU 可能降低 p99，却提高空闲成本；缩短 max output 可能提高吞吐，却损害任务成功率；把长请求转异步可能保护交互 SLO，却增加完成时间。比较方案时至少报告：每百万 token 成本、单位成功任务成本、SLO error budget、拒绝/降级率、峰值显存和故障恢复时间。

如果只用“最大 QPS”作为指标，系统会偏好短请求和容易成功的流量。容量计划应按 workload bucket 报告可接纳率，并明确每个 bucket 在过载时是排队、降级、异步还是拒绝。

## 61.27 敏感性分析与实验设计

Frontier model capacity planning 的目标是把模型声明转成可运行的资源和 SLO。权重、active 计算、KV/state、通信、prefill、decode、Agent 工具和故障余量必须一起算；本章公式是容量估算起点，不是部署承诺。

容量结论只有在版本、硬件、workload、SLO 和证据等级都写清时才可复用。模型卡是声明，理论公式是估算，单请求 profile 是局部观测，混合压力测试才接近服务结论，N+1 故障演练才能验证余量；价格、吞吐和上下文上限不能脱离日期与条件横向比较。

## 61.28 从模型卡到容量计划的转换表

模型卡给出参数、上下文、精度或最大长度时，容量工程仍要把它们转换成权重、激活、KV/state、通信、队列和故障余量。每个字段都要标记证据等级：声明、理论估算、单请求 profile、混合压测或故障演练。不同等级不能在同一张表里伪装成同样确定的数字。

## 61.29 敏感性分析比单点估算更有用

对输入长度、输出长度、并发、命中率、reasoning effort、工具轮数、active 参数、dtype 和故障余量分别做上下浮动，观察可接纳 QPS、显存和单位成本如何变化。若只要输出长度增加 20% 就超过 HBM，系统应优先优化路由和预算，而不是把平均吞吐写成容量承诺。

## 61.30 容量计划的验收材料

上线前应有 workload trace、硬件拓扑、版本矩阵、压测结果、p50/p95/p99、N+1 演练、降级路径、扩容时间和成本账。容量结论的有效期也要写明，因为模型、kernel、价格和流量分布都会变化。

## 61.31 小结

Frontier 模型容量规划必须把权重、active compute、KV/state、通信、工具和故障余量放到同一张资源账本中，并绑定 workload、版本、硬件和 SLO。单点 benchmark 只能提供估算入口，混合流量压测、N+1 演练和单位成功成本才决定是否具备上线容量。
