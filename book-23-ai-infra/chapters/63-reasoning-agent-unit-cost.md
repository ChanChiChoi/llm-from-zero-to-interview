# 第 63 章 Reasoning/Agent Unit Cost：成功完成一次任务到底花多少钱

## 63.1 token 单价为什么不够

普通聊天可以用输入输出 token 粗略估算成本。reasoning model 还可能消耗隐藏推理 token、verifier、工具、重试和更长 GPU 时间；Agent 还会产生浏览器、代码执行、存储、人工复核和外部 API 费用。

如果只看 token 单价，一个失败后重试三次的 Agent 可能被误判为便宜；一个高 effort 但一次成功的路径可能被误判为昂贵。正确单位应该是“完成一个成功任务的总成本”。

## 63.2 任务级成本公式

对任务 `i`：

```math
C_i=C_{\mathrm{model},i}+C_{\mathrm{reason},i}+C_{\mathrm{tool},i}
+C_{\mathrm{gpu},i}+C_{\mathrm{storage},i}+C_{\mathrm{retry},i}+C_{\mathrm{human},i}
```

单位成功成本：

```math
C_{\mathrm{success}}=\frac{\sum_i C_i}{\max(1,N_{\mathrm{success}})}
```

如果失败任务仍消耗资源，分子要包含它们；如果人工只复核少数高风险任务，人工成本要按实际分配，而不是平均摊成 0。

## 63.3 reasoning 成本

reasoning 成本不一定等于可见输出 token。记录 input token、output token、reasoning usage（若 provider 暴露）、候选数量、verifier 次数、工具时间、deadline 和模型路由。不同 provider 对内部 token 的计费和 usage 字段可能不同，必须保存价格版本和日期。

高 effort 的收益应写成质量-成本曲线：

```math
\Delta U=V\Delta P-\lambda\Delta C-\mu\Delta T-\nu\Delta R
```

任务价值 `V` 不同，最优 effort 也不同。

## 63.4 Agent 工具和重试

工具成本包括浏览器时间、代码沙箱、搜索、数据库、外部 API 和数据传输。失败重试可能重复工具调用；对不可逆动作还会增加风险成本，而不仅是钱。

每个工具都应标注 read、idempotent effect 或 irreversible effect，并记录执行状态。一次 timeout 若可能已经提交，重试前要查询状态；否则单位成本和副作用都会失真。

## 63.5 一个单 Agent 与多 Agent 例子

低 effort 单 Agent 每任务模型成本 1、工具成本 0.5，成功率 0.70；高 effort 多 Agent 模型成本 3、工具和协调成本 2，成功率 0.85。对 100 个任务，前者总成本约 150、成功 70，单位成功成本 `150/70≈2.14`；后者总成本约 500、成功 85，单位成功成本 `500/85≈5.88`。

若任务失败损失极高，后者可能值得；若是低价值 FAQ，前者更合理。不能用“成功率更高”单独决定路由。

## 63.6 成本与质量联合路由

路由器可以先用低 effort 处理低风险任务；verifier 失败、证据不足、用户价值高或风险高时，升级到高 effort、工具或人工。路由策略要用成功任务成本训练和监控，而不是只用 token 数。

高风险任务还要设硬安全约束：即使高 effort 成功率高，只要越权或敏感外发率超过阈值，就不能进入自动路径。

## 63.7 容量成本

单位任务成本还受到 GPU 利用率和队列影响。长 reasoning 占用 decode slot，长上下文占用 KV，Agent 并行占用协调和网络。价格表之外，要计算：

```math
C_{\mathrm{gpu}}=\mathrm{reserved\_gpu\_time}\times\mathrm{effective\_rate}
```

峰值容量和闲时利用率会改变真实单位成本。高平均 GPU 利用率如果带来 p99 和重试，可能反而变贵。

## 63.8 评估表

| 路径 | success | model token | tool time | retries | p95 | cost/success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct | 0.70 | 1.0x | 0.2x | 0.1x | 1.5 s | 2.1 |
| verified | 0.79 | 1.6x | 0.8x | 0.2x | 3.4 s | 3.0 |
| multi-agent | 0.85 | 3.0x | 2.0x | 0.4x | 6.8 s | 5.9 |

数字只是记录格式。还要按任务类型、用户价值、风险、长度和 model revision 分桶。

## 63.9 常见失败

只算 token 不算工具/GPU/人工；把失败重试从分子删掉；把平均成本当成功成本；用模型单价替代容量；忽略缓存和 batch；把人工接管记成成功却不计成本；高 effort 成本没有上限。

## 63.10 面试回答与练习

回答“如何比较两个 Agent 方案”时，应以成功任务为分母，完整计算模型、reasoning、工具、GPU、存储、重试和人工成本，同时看延迟、容量、风险和失败类型。高成功率只有在价值和 SLO 足以支付成本时才有意义。

练习一：为 direct、verified、multi-agent 建立成本表。

练习二：计算成功率从 0.72 提升到 0.80 时，允许增加多少成本仍然值得。

练习三：列出三种常被漏算的 Agent 成本。

### 63.10.1 成本账本的分项

一个成功任务的成本应拆成：

```math
C_{\mathrm{task}}
=C_{\mathrm{input}}+C_{\mathrm{output}}
 +C_{\mathrm{reason}}+C_{\mathrm{tool}}
 +C_{\mathrm{gpu}}+C_{\mathrm{storage}}
 +C_{\mathrm{retry}}+C_{\mathrm{human}}
```

若模型按 token 计费，输入中的重复上下文、tool result 和 reasoning token 都要计入；若自建 GPU，还要分摊权重常驻、KV、空闲容量、网络和故障余量。工具的搜索、浏览器、代码执行和人工审核不能被写成“免费环境”。

### 63.10.2 单位成功成本的例子

假设 direct 路径 100 次请求成功 72 次，总成本 140；verified 路径成功 80 次，总成本 250，则：

```math
C_{\mathrm{success,direct}}=140/72\approx1.94
```

```math
C_{\mathrm{success,verified}}=250/80=3.125
```

verified 的成功率更高，但单位成功成本约为 1.61 倍。是否值得取决于一次失败的业务损失、SLO 和人工成本；支付、发布和安全审计可能愿意支付，普通闲聊可能不愿意。

### 63.10.3 长任务和多 Agent 的隐藏成本

长周期 Agent 还要计算 checkpoint、workspace 存储、trace、回放、并发 worker、失败恢复和人工接管。多 Agent 可能减少 wall-clock，却增加总 token、通信、merge、重复读取和验证。容量规划要同时报告 cost per task、cost per success、p95、peak GPU 和失败严重度。

### 63.10.4 成本优化的正确顺序

先去掉无效工具调用和重复上下文，再做缓存、路由、预算和量化；最后才用降低验证或安全的方式换成本。每次优化都要重算单位成功成本和危险失败，而不是只看 token 数下降。

## 63.11 失败成本和成功成本要分开

一个 Agent 任务可能在最后一步失败，却已经消耗大量推理、工具和人工时间；也可能低成本成功。单位成功成本应把重试、人工审查、回滚和失败副作用纳入，而不是只除以模型输出 token。

路由实验比较 small/large、low/high reasoning、单 Agent/多 Agent 和有无 verifier，记录质量、总耗时、工具调用、失败恢复和人工介入。

## 63.12 成功成本的分解

Reasoning 和 Agent 任务的成本不只是输入输出 token，还包括思考 token、工具调用、检索、重试、verifier、人工确认和失败副作用。可定义：

~~~math
C_{\mathrm{success}}
=\frac{C_{\mathrm{model}}+C_{\mathrm{tool}}
+C_{\mathrm{retrieval}}+C_{\mathrm{retry}}
+C_{\mathrm{verification}}+C_{\mathrm{human}}}
{P_{\mathrm{success}}}
~~~

分母低时，单次便宜的模型也可能更昂贵。

## 63.13 预算控制和动态路由

系统可以按任务风险和难度设置 thinking budget、工具轮数、并发 Agent 数和 verifier 次数。预算耗尽时，选择澄清、摘要、人工接管或安全拒答，而不是无限重试。

路由实验要比较 fast/thinking、单 Agent/多 Agent、少工具/多工具和不同 verifier。报告质量、延迟、token、工具副作用、人工介入和单位成功成本。

## 63.14 成本 trace 和优化顺序

先用 trace 找出主要成本：模型 decode、长上下文 prefill、工具 round trip、检索、验证还是重试。再分别尝试 cache、程序化工具调用、模型路由、并行 Agent、量化或减少输出。没有归因的优化容易把成本从模型转移到工具和人工。

线上要按租户、任务类型、模型 revision 和风险记录成本与成功率，防止平均值掩盖少数高成本任务。

## 63.15 先建立可核对的成本账本

一个任务的 trace 应把模型、检索、工具、重试、验证和人工事件分开记账。至少保存 request id、tenant、model revision、input/output/thinking token、GPU 时间、工具名称、工具耗时、失败原因、人工介入和最终成功状态。没有这些字段，成本优化很容易把一项支出隐藏到另一项里。

设一次任务的直接成本为：

~~~math
C_{\mathrm{direct}}
=C_{\mathrm{input}}
+C_{\mathrm{output}}
+C_{\mathrm{thinking}}
+C_{\mathrm{tool}}
+C_{\mathrm{retrieval}}
+C_{\mathrm{gpu}}
+C_{\mathrm{storage}}.
~~~

如果任务失败后重试，前一次消耗不能被覆盖；如果工具产生可逆或不可逆副作用，还要单独计入补偿、人工复核和事故成本。

## 63.16 思考预算与验证预算

reasoning 模型的输出 token 可能分成可见答案、隐藏思考、工具参数和工具结果。把它们全部按“回答长度”统计，会低估高 effort 请求。预算应至少分成最大思考 token、最大工具轮数、最大验证次数和最大墙钟时间。

对于一个要求正确率至少为 q 的任务，可以比较不同策略：

~~~math
U
=\Pr(\mathrm{success})V
-C_{\mathrm{direct}}
-\Pr(\mathrm{failure})C_{\mathrm{recovery}},
~~~

其中 V 是成功价值，C_recovery 是失败后的恢复或人工成本。高 effort 只有在成功概率增加足以覆盖额外消耗时才合理；不能把“思考得更久”当成无条件的质量提升。

## 63.17 Agent 工具调用的期望成本

工具调用会带来模型 round trip、网络、鉴权、序列化、执行和结果写回上下文的成本。若工具在第 i 轮被调用的概率为 p_i，单次成本为 c_i，重试概率为 r_i，则工具成本可用：

~~~math
\mathbb{E}[C_{\mathrm{tool}}]
=\sum_i p_i\,c_i(1+r_i+\cdots).
~~~

真正的增长还可能来自工具结果变长，导致下一轮 prefill 和 KV 增长。把工具成本只记成 API 价格，会漏掉上下文膨胀、并发占用和权限审计。

## 63.18 失败调整后的单位成功成本

最重要的分母是成功任务数，而不是请求数。一个简单的单位成功成本为：

~~~math
C_{\mathrm{per\ success}}
=\frac{C_{\mathrm{model}}+C_{\mathrm{tool}}+C_{\mathrm{retry}}
+C_{\mathrm{verification}}+C_{\mathrm{human}}+C_{\mathrm{incident}}}
{N_{\mathrm{successful\ tasks}}}.
~~~

如果系统把“安全拒答”算作失败，把“危险动作成功”算作成功，账本会鼓励错误的策略。因此 success 定义必须与业务目标、安全验收条件和人工验收绑定；高风险任务即使输出文本看起来完成，也可能不算成功。

## 63.19 动态路由和优化顺序

路由器可以根据任务难度、证据完整性、风险和剩余预算选择 fast model、thinking model、verifier、人工或异步队列。路由实验要先固定成功定义，再比较质量、延迟、工具副作用、人工介入和单位成功成本。

优化顺序通常是先删除无效重试和重复上下文，再做 prefix/cache、结果裁剪、模型路由、并行化和批处理，最后才考虑降低验证或安全保护。每次改动都要在 trace 中确认成本是否真的下降，以及失败是否只是转移到了人工或工具层。

## 63.20 worked example：更贵的路线反而更便宜

假设 100 个任务有三种路线：

| 路线 | 直接成本/次 | 成功率 | 人工/失败补偿 | 说明 |
| --- | ---: | ---: | ---: | --- |
| fast model | 1.0 | 0.70 | 0.30 | 输出短，验证弱 |
| thinking + verifier | 2.4 | 0.90 | 0.15 | 思考和验证更充分 |
| multi-agent | 4.5 | 0.92 | 0.50 | 多轮协作，工具多 |

若只比较每次请求价格，fast model 最便宜；若按成功任务计算，fast model 的直接成本约为 `1.0/0.70=1.43`，thinking 路线约为 `2.4/0.90=2.67`。再把失败人工和补偿计入，真实差距可能缩小；如果失败任务会造成昂贵的人工复核或业务损失，fast model 可能不再是最佳路线。

正确决策还要按风险分桶。普通草稿允许 fast model，代码发布、财务核对和高风险工具动作则要求 verifier 或人工确认。路由器不能用全局平均成功率覆盖这些差异。

## 63.21 预算状态机与路由验收条件

成本控制不只是请求前设置一个 `max_tokens`。任务可以经历 `budget_reserved`、`thinking`、`tooling`、`verification`、`human_review` 和 `settled`。每个阶段消耗独立预算，剩余预算不足时转入澄清、异步、降级或安全拒答。

可以把路由目标写成效用：

```math
U(a\mid x)
=P_{\mathrm{success}}(a\mid x)V(x)
-C_{\mathrm{direct}}(a\mid x)
-C_{\mathrm{failure}}(a\mid x).
```

`a` 是候选路线，`x` 是任务和风险上下文。这个式子不要求线上准确估计所有概率，但要求账本显式包含失败后果。高 reasoning budget 只有在提升成功概率或减少昂贵错误时有价值；更多 token 本身不是价值。

成本账本还要记录失败尝试、取消后资源、缓存命中、GPU 空闲预留、工具结果写回带来的 prefill、人工审阅和事故补偿。否则优化一个模型 API 单价，可能只是把支出转移到 verifier、工具或人工。

## 63.22 成本实验的基础结论

最小成本实验可以用同一批任务比较：低/高 reasoning budget、单 Agent/多 Agent、有无 verifier、有无工具缓存和不同模型路由。报告每个任务的输入、输出、思考、工具、重试、成功、人工和 p95，不要只报告平均 token。

## 63.23 成本账本要能和 trace 对账

每个任务建立一个 immutable cost ledger，至少记录 request id、model revision、输入/输出/推理 token、工具调用、检索、验证、重试、GPU 时间、人工处理、缓存命中、失败补偿和最终成功状态。账本总额要能和 provider usage、serving metrics、工具账单及人工工单对账。

如果 token 账和 GPU 账对不上，可能是 batch padding、prefill 重算、reasoning 字段没有计入、工具结果重复回填或取消请求仍占用资源。成本分析先解决可核对性，再谈优化。

## 63.24 失败成本不能藏在成功成本之外

一次失败任务可能重新调用模型、重复检索、占用队列、触发人工复核或产生补偿。设尝试路线 `a` 的直接成本为 `C_d`，失败概率为 `p_f`，失败后的平均补偿为 `C_f`，则期望成本至少应包含：

~~~math
E[C\mid a]=C_d(a)+p_f(a)C_f(a).
~~~

这仍未包含严重事故的非线性影响，但比只看 API 单价更接近真实决策。对支付、生产发布、数据删除等任务，失败成本应单独作为 hard gate；不能为了少用 token 而选择会放大副作用的路线。

## 63.25 动态路由的预算状态机

成本控制不是请求前设置一个 `max_tokens` 就结束。任务可以经历 `budget_reserved`、`thinking`、`tooling`、`verification`、`human_review` 和 `settled`。每一阶段有独立上限，剩余预算不足时转为澄清、异步、降级或安全拒绝。

路由器还要防止重试风暴：同一个外部动作应使用 action id 和幂等键；模型超时而工具状态未知时先查询状态；多个 Agent 的总预算要有 coordinator 级上限。失败后继续加预算不一定提高成功率，可能只是重复同一个错误路径。

## 63.26 成本和质量的 Pareto 边界

低成本路线、高质量路线和低风险路线常常不能同时最优。可以对每个候选方案记录 `(quality, success_cost, p99, risk)`，删除被另一方案在所有维度支配的点，保留 Pareto 前沿。普通草稿、代码修复、企业问答和生产动作应使用不同前沿，而不是用一个全局平均选择。

实验要固定质量验收条件：引用支持、测试通过、结构化合法、权限正确、无重复副作用。若成本下降是靠放宽验收条件，必须把它描述为质量/风险变化，而不是成本优化。

## 63.27 一个可复现的单位成本实验

在固定任务集上比较 fast、thinking+verifier 和 multi-agent 三条路线。每条路线固定最大总预算和工具权限，逐任务记录 token、调用次数、工具等待、失败、人工、p50/p95、质量和最终状态。再按简单/困难、高风险/低风险、短/长上下文分桶。

结果应回答四个问题：哪条路线成功率更高，额外成本花在哪里，失败是否更容易恢复，路由器能否在任务开始前识别适合的路线。若只能回答“某模型更便宜”，说明还没有建立任务级成本模型。

## 63.28 成本实验与证据层级

Reasoning/Agent unit cost 把完成一次真实任务作为成本单位，连接模型预算、工具轨迹、GPU 容量、失败补偿和人工复核。价格、usage 和模型计费会随 provider、地区和日期变化；本章公式用于建立可核对的成本账本，不能替代实际账单、容量压测和风险评估。

任何成本结论都应同时写明成功判据、失败分母、模型/引擎版本、硬件、并发、缓存、工具和人工条件。更贵的路线可能因为减少失败而降低单位成功成本，更便宜的路线也可能因副作用和人工补偿而更贵。

## 63.29 成本分解要保留失败补偿

一次 Agent 任务的成本不仅是模型输入输出，还包括 reasoning、工具、检索、验证、重试、GPU 排队、人工复核和失败补偿。可以写成：

```math
C_{\mathrm{task}}
=C_{\mathrm{model}}+C_{\mathrm{tool}}+C_{\mathrm{retrieve}}
+C_{\mathrm{verify}}+C_{\mathrm{retry}}+C_{\mathrm{human}}.
```

若某路线单次调用便宜，却让失败率和人工复核上升，单位成功成本可能更高。成本表必须把失败任务也放在分母解释中。

## 63.30 预算路由的反事实

比较固定 low、固定 high、规则路由和带 verifier 的 adaptive 路由，固定总任务集、权限、工具和成功判据。记录成功率、每成功任务成本、p95、升级率、重试、人工接管和安全事件。路由器只有在相同质量/风险验收条件下节省成本，才算有效。

## 63.31 单位成本和容量相互约束

reasoning effort 或工具轮数增加，会同时增加 token、KV/state、队列等待和 GPU 并发压力。成本模型不能独立于容量模型；在低并发实验中便宜的路径，可能在峰值流量下因为排队和降级变贵。报告应把单位成本曲线和可接纳 QPS 一起给出。

## 63.32 小结

Reasoning/Agent 的成本单位应是成功且安全地完成一次任务，而不是一条 API 调用或一段输出 token。预算状态机要覆盖思考、工具、验证、人工和失败补偿；路线比较要固定成功判据、权限、版本和 workload，再用 Pareto 与 hard gate 选择，而不是用单价替代系统成本。
