# 第 65 章 Acceptance Length：推测解码到底省了多少调用

## 65.1 不能只看 tokens/s

推测解码最直观的指标是 draft 候选有多少被 target 接受。但 acceptance length 不是一个脱离 workload 的模型属性：tokenizer、chat template、temperature、grammar、position/cache、batch 和 target revision 都会改变它。

更重要的是，接受得多不一定端到端更快。draft 生成、target 验证、候选 cache、调度和 stream parser 都要花成本。只有把这些开销放进同一张表，才知道推测解码是否值得上线。

## 65.2 三个基础指标

接受率是被保留的候选 token 比例：

```math
R_{\mathrm{accept}}=\frac{N_{\mathrm{accepted}}}{\max(1,N_{\mathrm{drafted}})}
```

平均接受长度是每次 verify 前进的 token 数：

```math
\bar A=\frac{N_{\mathrm{accepted}}}{\max(1,C_{\mathrm{verify}})}
```

target call reduction 可以写成：

```math
G_{\mathrm{call}}=\frac{C_{\mathrm{baseline}}}{\max(1,C_{\mathrm{spec}})}
```

三个指标回答不同问题。接受率低，说明候选和 target 分歧；平均接受长度还受窗口和拒绝位置影响；call reduction 只是 target 层指标，不能代替延迟和成本。

## 65.3 从接受长度估算收益

假设每轮 draft 产生 `K` 个 token，平均接受 `A` 个，target verify 成本为 `L_v`，draft 成本为 `L_d`，调度成本为 `L_s`，教学化的每轮前进效率可以写成：

```math
E_{\mathrm{round}}=\frac{A+1}{L_v+L_d+L_s}
```

`+1` 表示拒绝位置或 verify 产生的 target token。真实引擎可能使用 tree verify、共享 hidden state 或不同的采样协议，因此最终结论必须来自端到端压测。

举例：baseline 每个 token 成本 1；spec 每轮接受 3 个，draft+verify+schedule 成本为 2.2，则每个 token 的近似成本为 `2.2/4=0.55`，理论上有收益。若接受长度降到 0.5，成本变成 `2.2/1.5≈1.47`，就应关闭 speculative。

## 65.4 为什么 workload 会改变 acceptance

代码、JSON 和模板化文本有较强重复结构，候选更容易被 target 接受。开放写作的候选空间大，temperature 和 top-p 也会放大分歧。数学 reasoning 可能在公式附近有稳定模式，却在分支选择处突然分歧；工具调用还受到 grammar 和参数值的影响。

因此 benchmark 至少按任务、输出长度、temperature、grammar、effort、模型 revision 和 batch 分桶。一个全局平均值可能掩盖“代码加速、开放写作变慢、工具协议不兼容”的真实情况。

## 65.5 拒绝处理的正确性

候选被拒绝后，只能提交 accepted prefix。rejected token 不能写入正式 KV、grammar state、stream event 或 usage。对工具调用，半个 JSON 不能被 executor 接收；对多模态 placeholder，位置状态也不能错位。

可以构造一个回放：候选为 `A B C D`，target 只接受 `A`。检查正式输出长度、KV 长度、parser 状态、stream event 和下一次 target logits。如果任何对象包含 `B`、`C` 或 `D`，就是状态污染。

## 65.6 动态调整 speculative window

高接受率请求可以逐渐增加候选窗口，低接受率请求减少窗口或关闭模块。一个简单控制器为：

```math
k_{t+1}=\mathrm{clip}(k_t+\gamma(\bar A_t-A_{\mathrm{target}}),k_{\min},k_{\max})
```

控制器还要考虑最小样本数、p95、grammar 失败、峰值显存和 batch。否则流量改变时窗口会来回振荡，或因为一次偶然高接受率申请过多 cache。

## 65.7 一个压测表

| workload | accept rate | mean accept | TPOT baseline | TPOT spec | peak memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 0.72 | 3.1 | 42 ms | 25 ms | +8% |
| JSON | 0.66 | 2.7 | 40 ms | 27 ms | +10% |
| open text | 0.28 | 0.8 | 41 ms | 47 ms | +12% |
| tool call | 0.51 | 1.6 | 44 ms | 43 ms | +15% |

数字只是结果记录格式示例。结论应同时看输出等价性、p99、并发、单位成功成本和 fallback 比例。

## 65.8 常见失败

只展示全接受的 happy path；把 target call 减少解释成延迟必然下降；忽略候选 cache 的峰值；不同 tokenizer 或模板混跑；高温度任务的接受率被平均掉；streaming 重复输出；低接受率请求仍然占用大窗口；只看平均不看 p99。

## 65.9 面试回答与练习

回答“如何判断 speculative decoding 是否值得”时，应先测 acceptance rate 和平均接受长度，再把 draft、verify、调度、cache、grammar 和 fallback 成本加入端到端 benchmark；按 workload 分桶，检查输出正确性、TTFT、TPOT、p99、显存、并发和单位成功成本，低接受率时动态关闭。

练习一：给出三种 workload 的 acceptance 和成本，计算近似 target call reduction 与单位 token 成本。

练习二：设计拒绝回滚的五个断言。

练习三：说明为什么 p99 可能在平均 TPOT 下降时反而变差。

### 65.9.1 从接受长度到 speedup

设 draft 每步生成 `k` 个候选，平均接受长度为 `a`，target 单次验证成本为 `C_T`，draft 成本为 `C_D`，额外调度和 cache 成本为 `C_O`。每接受一个 token 的粗略成本可以写成：

```math
C_{\mathrm{spec/token}}
\approx\frac{C_T+C_D+C_O}{\max(1,a)}
```

baseline 的成本是 `C_T` 加上每 token 调度成本。只有当候选平均接受足够长，且 `C_D+C_O` 没有吃掉 `C_T` 的节省时，speculative 才有收益。

### 65.9.2 接受率的分布比平均值更重要

平均接受长度为 3 可能来自所有请求都稳定接受 3 个，也可能来自一半请求接受 6 个、另一半完全不接受。两种情况下 p99、cache 分配和动态开关策略不同。应报告 p50/p90/p99 acceptance、按任务桶的分布和低接受率请求比例。

### 65.9.3 质量和协议验收条件

正确性不只是最终文本相同。要比较 baseline/speculative 的 sampling 分布、stop reason、grammar 合法性、tool arguments、usage、stream event、引用和安全拒绝。若 speculative 在 JSON 或工具调用中偶尔产生非法前缀，即使平均 TPOT 变好，也不能上线。

### 65.9.4 p99 变差的原因

候选树可能在长请求上占用临时 KV，触发 preemption；低接受率请求支付额外 draft 成本；不同长度请求在 scheduler 中产生 batch fragmentation；回滚和 fallback 的慢路径集中在尾部。压测要同时看平均和尾部，并做高并发、取消和 OOM 注入。

## 65.10 接受长度的分布

平均接受长度不足以规划服务。应按语言、任务、温度、输出位置、代码/JSON、上下文长度和请求并发画分布，记录 p50、p90、拒绝比例和 target verify 时间。

如果少数长请求接受长度很高但大量短请求接近零，平均 tokens/s 可能误导容量计划；调度器还要防止 verify batch 造成尾延迟。

## 65.11 acceptance length 的估算

设一次 speculative round 产生 k 个候选，平均接受长度为 a，target 验证一次的成本为 C_t，draft 和准备成本为 C_d，则每轮平均推进 token 数接近 a+1，粗略收益可写成：

~~~math
S
\approx\frac{a+1}{C_t+C_d}
\Big/
\frac{1}{C_t}
~~~

这只是直觉模型。真实收益还受 batch、并行验证、KV 写入、回滚、网络和调度影响。a 越大不一定越快，如果候选树和验证 workspace 很重，收益可能被抵消。

## 65.12 acceptance 不是单一质量指标

平均 acceptance length 要按任务、长度、采样温度、模型版本、语言、代码、数字、工具 JSON 和 reasoning 分桶。某路径在普通聊天上接受很长，在格式敏感任务上频繁拒绝，整体平均值会隐藏风险。

还要检查拒绝后的输出是否和普通 decode 一致。推测解码的正确性目标是分布或采样语义保持，而不是只让 token 看起来相似。

## 65.13 收益递减和关闭条件

候选长度、draft 规模和分支数量增加会提高预测覆盖，也会增加显存、验证和回滚。线上可以按 acceptance、p99、错误率和成本设置动态关闭条件，保证 speculative 失败时回到普通 decode。

灰度阶段记录每个请求的路径、候选和提交结果，不能只在聚合层看平均 TPOT。高风险任务可使用更保守的候选长度或直接关闭。

## 65.14 接受长度的期望值如何展开

设 A 是一轮被 target 接受的连续 token 数，最大候选窗口为 K。不需要假设 A 服从几何分布，也可以用尾概率表示期望：

~~~math
\mathbb{E}[A]
=\sum_{j=1}^{K}\Pr(A\ge j).
~~~

这个公式说明为什么只报“平均接受率”不够。若第一位置经常正确、后面迅速失败，Pr(A>=1) 很高但长前缀很少；如果接受长度分布有一小部分极长尾，平均值又可能掩盖大量零收益请求。服务应保存每个位置的接受概率和整轮 histogram。

## 65.15 从接受长度到端到端 speedup

设普通 decode 每推进一个 token 的成本为 C_t，候选和准备成本为 C_d，验证一次窗口的成本为 C_v，平均每轮推进 E[A]+1 个 token，则教学估算为：

~~~math
S_{\mathrm{e2e}}
\approx
\frac{C_t\,(\mathbb{E}[A]+1)}
{C_d+C_v+C_{\mathrm{rollback}}+C_{\mathrm{schedule}}}.
~~~

这个式子不是 benchmark 结果，而是检查量纲的工具。C_v 可能因为 batched verification 小于多个单 token step，但也可能被长上下文的 KV 读写、候选树和低并发抵消。若把 C_d 或失败回退成本漏掉，就会系统性高估收益。

## 65.16 接受长度的分布漂移

接受长度会随模型版本、tokenizer、模板、上下文位置、语言、温度、top-p、重复惩罚和任务改变。reasoning 模型的隐藏思考 token、工具 JSON 的 grammar 和代码缩进都会改变局部 token 相关性。

线上应保存脱敏后的 bucket 统计，而不是长期保存原始 prompt：请求类型、输入长度、输出长度、温度、候选窗口、A 的 p50/p90、拒绝位置、fallback、p99 和最终质量。新版本若平均 A 提高但高风险工具请求的拒绝率上升，不能只看总均值决定发布。

## 65.17 接受率控制器的迟滞

如果每个请求都依据上一轮的 acceptance 立即调大或调小窗口，系统会在边界 workload 上振荡。可以使用带迟滞的控制器：连续多个窗口超过上阈值才增加候选长度，连续多个窗口低于下阈值才缩短，并设置冷却时间。

~~~text
if low_acceptance_for_n_rounds and p99_ok:
    window = max(min_window, window - step)
elif high_acceptance_for_n_rounds and memory_headroom_ok:
    window = min(max_window, window + step)
else:
    keep(window)
~~~

候选窗口控制必须服从 hard gate：grammar 错误、token 等价性错误、cache 回滚失败和安全策略异常直接关闭路径，不能由平均收益“抵消”。

## 65.18 一个数值例子：平均接受长度不等于收益

假设 baseline 每个 token 的 target 成本为 1.0，speculative 每轮候选和验证固定成本为 2.8。若平均每轮推进 `a+1` 个 token，`a=1` 时每 token 成本约为 `2.8/2=1.4`，反而比 baseline 慢；`a=3` 时约为 `2.8/4=0.7`，才可能有收益。若 20% 请求的 acceptance 为 0、80% 请求为 4，平均接受长度仍可能看起来不错，但失败桶会支付额外候选成本并拉高 p99。

因此实验要保存每请求的 acceptance histogram、fallback 和临时 state，而不是只保存全局均值。调度器可以在低 acceptance 请求上关闭 speculative，把资源留给高 acceptance 且 SLO 合格的请求。

## 65.19 关闭控制与发布条件

线上控制器应有最小/最大窗口、连续观测轮数、冷却时间和 hard gate。连续多个窗口 acceptance 低且 p99 变差时缩短候选；连续多个窗口 acceptance 高且显存有余量时才扩大。grammar 错误、rejected token 泄漏、cache 回滚失败和副作用不确定时直接关闭，不由平均加速抵消。

发布比较至少包括 baseline、speculative、动态关闭和故障回退四条路径。按普通文本、代码、JSON、工具、reasoning 和长上下文分桶，检查质量、stream、usage、取消、p99、峰值显存和单位成功成本。

## 65.20 接受长度的条件分布

平均 acceptance length 会掩盖 workload 差异。应按模型 revision、语言、任务、上下文位置、temperature、top-p、grammar、候选窗口和请求长度保存直方图。代码、JSON、工具参数和 reasoning token 的分布可能完全不同。

对每个 bucket 记录 `A` 的 p50、p90、p99、零接受比例、拒绝位置和回退。若总体平均值提高，但高风险工具 bucket 的零接受比例上升，系统不能据此扩大 speculative 流量。接受长度是条件变量，不是一个跨任务可比较的模型能力分数。

## 65.21 从接受长度推导收益要补齐成本

设每轮接受 `a` 个候选并额外提交一个 target token，候选和验证成本为 `C_c+C_v`，单 token baseline 成本为 `C_d`。粗略的每 token 成本为：

~~~math
C_{\mathrm{token}}
\approx\frac{C_c+C_v}{a+1}
 +C_{\mathrm{rollback}}+C_{\mathrm{overhead}}.
~~~

当 `a` 低时，候选成本无法摊薄；当 `a` 高时，验证树、临时 KV、网络和调度开销可能增长。公式用于检查量纲，不是 speedup 承诺。必须用端到端 trace 验证 `C_rollback`、preemption 和排队对 p99 的影响。

## 65.22 接受率控制器需要迟滞和硬性条件

如果每轮都依据上一轮 acceptance 调整窗口，边界 workload 会在大窗口和小窗口之间振荡。控制器可以要求连续多个窗口超过上阈值才增加候选，连续多个窗口低于下阈值才缩短，并设置冷却时间。

硬硬性条件优先于均值收益：grammar 错误、rejected token 泄漏、cache rollback 失败、stream event 不一致、工具状态未知和未授权动作直接关闭路径。平均接受长度再高，也不能抵消协议和安全错误。

## 65.23 p99 变差的常见原因

接受长度提高但 p99 变差，常见原因包括候选临时 state 造成抢占、验证 batch 等待、长请求占用 scheduler、低 acceptance 请求重复付出候选成本、模块冷启动和回退路径重新 prefill。排查应按请求 trace 分解 prefill、candidate、verify、commit、queue、preemption 和 fallback，而不是只看 tokens/s。

短请求和长请求还要分池或设置公平策略。对低 acceptance bucket 可以关闭 speculative，把 GPU 预算留给真正能摊薄验证成本的请求。

## 65.24 结构化输出和工具任务的专门验收

普通文本接受率不能代替 JSON、代码和工具任务验收。结构化任务要验证 grammar state、字段完整性、tool call id、参数 round-trip、stop reason 和事件顺序；工具任务还要验证 candidate 阶段不执行副作用，target 确认后不重复提交。

质量回归应按任务成功而不是字符相似度判断。一个 JSON 文本看起来更短不代表调用正确；一次工具请求返回 200 也不代表动作只执行了一次。

## 65.25 一个可复现的 acceptance 实验

固定模型、tokenizer、模板、采样、硬件、并发和任务集，比较 target-only、固定候选窗口、动态窗口和关闭回退四条路径。逐请求保存接受直方图、target call、候选/验证时间、临时 bytes、TTFT、TPOT、p99、显存、错误、取消恢复、最终质量和单位成功成本。

对长上下文、代码、JSON、工具、reasoning 和多语言分别切片，并改变 temperature、prefix cache 命中和请求混合比例。这样才能找到“平均速度提升但某个关键 bucket 退化”的情况。

## 65.26 验收标准的证据范围

一个可以上线的 acceptance 实验至少要有 baseline、固定 workload、相同质量验收条件、接受长度分布、target call、端到端 TTFT/TPOT、p99、峰值显存、错误率、取消恢复和单位成功成本。随机采样要验证分布语义，结构化输出要验证完整 event sequence，工具任务要验证没有重复副作用。

Acceptance length 只是推测解码收益的中间变量。它必须和候选成本、验证成本、协议正确性、显存、SLO 和任务成功率一起解释。参考 speculative decoding、EAGLE/MTP 公开资料以及 vLLM/SGLang 文档；任何具体数值都要绑定模型、采样、模板、硬件和 runtime 版本。

## 65.27 接受长度的条件分布

令 `A` 表示一轮连续接受的候选 token 数，最大候选窗口为 `K`。其期望可以写成尾概率之和：

```math
\mathbb{E}[A]=\sum_{j=1}^{K}\Pr(A\ge j).
```

这个形式说明为什么平均接受率不够。第一 token 几乎总是被接受、后续迅速拒绝时，`Pr(A\ge1)` 很高但收益有限；少量长候选也可能把平均值抬高，却让大多数请求支付无收益的 draft 成本。线上应保存按位置的接受概率、零接受比例和完整 histogram。

## 65.28 从 acceptance 到 p99

端到端时间可粗略拆为：

```math
T_{\mathrm{req}}
=T_{\mathrm{queue}}+T_{\mathrm{prefill}}
 +R(T_{\mathrm{candidate}}+T_{\mathrm{verify}}
 +T_{\mathrm{commit}})
 +T_{\mathrm{fallback}}.
```

提高平均 `A` 可能降低 `R`，但临时 KV、验证 batch、低收益请求和回退会增加其他项。p99 变差时要沿 trace 查 queue、candidate、verify、preemption、rollback 和 fallback，而不是只看全局 TPOT。长请求与短请求通常需要分池或公平调度。

## 65.29 接受率控制器的迟滞

基于单轮 acceptance 立即改变候选窗口，容易在边界 workload 上振荡。更稳定的控制器需要连续观测窗口、上下阈值、冷却时间和显存条件：连续多轮高接受且 p99/显存满足时增大，连续多轮低接受或尾延迟恶化时缩小。

控制器必须服从 hard gate。grammar 错误、rejected token 泄漏、cache rollback 失败、工具状态未知和未授权动作直接关闭 speculative，不能让平均 acceptance 抵消协议/安全错误。调节决策和原因要写进 trace，便于解释版本切换后的行为。

## 65.30 接受长度的分布比平均数更有用

设一轮候选窗口为 `K`，接受长度为随机变量 `A`。平均值 `E[A]` 不能说明长尾：两个系统都可能平均接受 2 个 token，一个稳定地接受 2 个，另一个一半请求接受 0 个、一半接受 4 个。后者的 p99、调度公平和失败恢复可能更差。

因此要报告 `P(A=0)`、`P(A\ge k)`、拒绝位置、workload bucket 和窗口大小。对结构化输出，还要报告 grammar 约束前后的接受长度，避免把格式过滤造成的退化隐藏在总体平均里。

## 65.31 速度模型中的固定成本

若普通 decode 每 token 成本为 `C_t`，speculative 每轮固定成本为 `C_0`、平均提交 `a` 个 token，则粗略每 token 成本为：

```math
C_{\mathrm{spec/token}}
\approx\frac{C_0}{a}+C_{\mathrm{verify/token}}.
```

当 `a` 很小时，候选生成和调度固定成本无法摊薄；当 `a` 足够大且 verify 能批量化，才可能超过普通 decode。这个模型用于解释趋势，最终数字必须来自固定硬件和 workload 的压测。

## 65.32 acceptance controller 的迟滞和冷启动

如果每一轮 acceptance 下降就立即关闭，系统会在短期抖动中频繁切换；如果关闭太晚，又会持续浪费显存。控制器可以使用滑动窗口、最小启用时长和连续失败阈值，并为新请求设置冷启动探测。切换动作必须落在事务边界，且要保留前后指标。

## 65.33 接受长度不是跨模型能力分数

不同 tokenizer、chat template、temperature、grammar、reasoning effort、上下文位置、模型 revision 和 draft artifact 会改变 `A`。因此 acceptance length 只能在同一条件下比较，不能作为跨模型的通用智力或质量分数。

最终上线报告至少包含 target-only baseline、固定/动态候选、关闭回退、按 workload 分桶的接受分布、输出等价性、质量、安全、TTFT/TPOT、p99、显存、取消恢复和单位成功成本。只有这些指标共同通过，才能把“接受得更多”解释成“系统真正更快且可靠”。
