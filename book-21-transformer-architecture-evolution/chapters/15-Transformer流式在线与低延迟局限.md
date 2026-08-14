# 第十五章：Transformer 在流式、在线与低延迟场景的局限

## 15.1 离线生成和在线生成不是同一个问题

把一整段文章一次性送进模型，和每隔 20 毫秒接收一小段语音、日志或传感器数据，虽然都可以写成“序列建模”，但系统约束完全不同。离线场景可以批量化、预填充、等待更多上下文；在线场景要持续输出，不能因为等待整段输入而失去时效。

Transformer 在训练上擅长并行处理序列，在自回归推理上却要维护历史状态。每一次新 token 到达，都可能需要读取越来越长的 KV cache；如果输入和输出交替到达，还会出现频繁的小请求、动态 batch、状态迁移和尾延迟放大。

## 15.2 小白直觉：一边读一边回答的三种等待

在线助手通常有三种等待：

1. **首包等待**：模型先处理已经积累的上下文，才能生成第一个 token。
2. **每 token 等待**：生成过程中，用户等待下一个 token。
3. **输入等待**：模型还没收到足够的新输入，无法判断是否继续回答。

如果把这三种等待混成一个平均响应时间，就找不到真正的瓶颈。语音助手可能更关心端到端延迟和打断响应，代码补全更关心 p95 首 token，日志告警更关心持续吞吐和状态恢复。

## 15.3 自回归推理的成本拆解

设已有上下文长度为 `P`，本次输出长度为 `O`。预填充阶段处理输入，解码阶段逐 token 生成。一个教学化的时间模型是：

```math
T_{\mathrm{total}}=T_{\mathrm{prefill}}(P)+\sum_{i=1}^{O}T_{\mathrm{decode}}(P+i-1)
```

显式 KV cache 令模型不必重新计算历史的 K/V，但每个新 query 仍要读取历史 K/V。若每层每个 token 的 KV payload 为 `m_kv` 字节，读取量可以粗略写成：

```math
M_{\mathrm{read}}\approx\sum_{i=1}^{O}(P+i-1)m_{kv}
```

当 `P` 很大、`O` 较短时，decode 可能是 memory-bandwidth bound；当 `P` 也很大时，首 token 又会受到 prefill compute、attention workspace 和调度影响。GQA、MQA、量化 KV 和分页管理降低的是不同部分的成本。

## 15.4 为什么流式输入会让调度更难

一个请求可能交替经历：接收输入 chunk、更新状态、判断是否触发工具、生成几个 token、被用户打断、再接收新 chunk。传统“一个请求跑完再换下一个”的 batch 方式会让短请求等待长请求；只做 continuous batching 又要保证不同请求的状态不会错位。

状态机可以写成：

```text
RECEIVING -> PREFILLING -> DECODING -> WAITING_INPUT
     ^           |            |             |
     |           v            v             v
   CANCEL <- PREEMPTED <- TOOL_WAIT <- INTERRUPTED
```

每次状态转换都要明确：哪些 token 已提交、KV 是否可复用、位置 id 到哪里、是否允许回滚，以及调度器是否可以把请求迁移到其他 worker。

## 15.5 分块预填充和 token budget

对长输入一次性 prefill 会阻塞正在 decode 的请求。chunked prefill 把输入拆成多个块，在每轮给 decode 留出预算。若当前 iteration 的 token budget 为 `B`，可以约束：

```math
N_{\mathrm{decode}}+N_{\mathrm{prefill\_chunk}}\le B
```

这不是免费优化。chunk 太小会增加 kernel launch 和调度开销，chunk 太大则仍会造成 decode stall。应分别测 TTFT、TPOT、ITL、吞吐和 p99，而不是只看总 tokens/s。

## 15.6 一个低延迟排查 demo

下面用纯 Python 模拟输入长度和输出步数对延迟的影响，帮助建立指标直觉。系数只是教学参数。

```python
def profile_request(prompt_tokens, output_tokens, kv_bytes_per_token):
    prefill_ms = 0.004 * prompt_tokens + 0.00000002 * prompt_tokens**2
    decode_ms = sum(0.18 + 0.000015 * (prompt_tokens + i) * kv_bytes_per_token
                    for i in range(output_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": round(prefill_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "total_ms": round(prefill_ms + decode_ms, 3),
    }


for item in ((500, 40), (20_000, 40), (20_000, 800)):
    print(profile_request(*item, kv_bytes_per_token=2_048))
```

真实服务应从 trace 中记录实际 prompt token、命中 prefix cache 的 token、output token、batch size、GPU 利用率、KV 分配和网络等待，再把请求按 workload 分桶。

## 15.7 在线场景为什么重视状态而不是只重视吞吐

离线 benchmark 可以把整段输入放在一个进程中；在线服务要处理重启、抢占、扩缩容、连接断开和多租户。一个可迁移的请求状态至少包括 model revision、token offset、position contract、KV block 或递归 state、已提交输出和工具协议状态。只保存文本历史并重新 prefill，可能延长恢复时间；只保存 KV 却没有 tokenizer/template 版本，可能恢复出错误 logits。

状态的所有权也很重要。请求取消后，scheduler 必须释放 KV；batch 重排不能把 A 的 cache 绑定到 B；speculative decoding 拒绝候选时要回滚临时状态。状态 bug 往往表现为偶发重复 token、跨请求泄露或恢复后结果不一致。

## 15.8 与递归状态模型的取舍

Transformer + KV cache 适合需要精确访问上下文、工具结果和长文档证据的任务，但缓存随长度增长。递归状态模型可把每个请求的状态控制在近似固定大小，更适合连续流和高并发，但历史经过压缩，任意回看能力下降。混合模型通过少量 global attention 或外部 retrieval 给精确访问留出口。

比较必须同时包含：

```text
质量：精确回忆、跨段组合、打断后继续
系统：TTFT、TPOT、p99、显存、状态迁移、恢复时间
业务：单位成功成本、并发、数据隔离、可观测性
```

## 15.9 低延迟优化的边界

减少 token 数并不总能降低端到端延迟。输入压缩可能增加摘要模型调用；更激进量化可能触发 kernel fallback；批量变大提升吞吐却恶化首 token；跨节点 PD 分离降低互相干扰却增加 KV 传输；流式输出更早可见，却可能产生无效或需要撤回的中间结果。

因此在线优化应先画出请求时间线，再定位等待：客户端网络、tokenizer、排队、prefill、KV 分配、decode、工具、后处理和发送。每个阶段都要有可观测的开始/结束时间，不能用一个平均 latency 代替。

## 15.10 机制与边界：在线 attention 的真正瓶颈

decode 阶段常见瓶颈是低算术强度的 KV 读取和小 batch，prefill 阶段更接近高吞吐矩阵计算。长上下文还会让 KV cache 的布局、页表、跨 GPU 访问和内存碎片显著影响实际表现。FlashAttention 优化的是 attention 的中间读写，PagedAttention 优化的是 KV 的内存管理，两者不属于同一层问题。

对于流式输入，position id 和 mask 的增量一致性是正确性验收条件。若每个 chunk 独立从位置 0 开始，模型会把续接内容当成新文档；若 padding token 更新 state，batch 内请求会互相污染。对每次恢复运行前后比较 logits 或 cache checksum，通常比肉眼观察输出更可靠。

## 15.11 面试追问、误区与练习

**问：有了 KV cache，Transformer 就天然适合在线服务吗？**

标准回答：KV cache 避免重算历史 K/V，显著降低自回归重复计算，但 cache 随上下文增长，decode 仍要读取历史；流式服务还要解决 scheduler、状态迁移、抢占、打断、p99 和多租户隔离。

**问：chunked prefill 一定提升性能吗？**

标准回答：它通常减少长 prefill 对 decode 的阻塞，但会引入更多调度和 kernel 开销。应在固定 workload 下比较 TTFT、TPOT、ITL、吞吐和尾延迟，并调节 token budget。

常见误区包括只报告平均 latency、把首 token 和每 token 延迟混淆、把 prefix cache 当作永久状态、把理论线性复杂度当成实际低延迟，以及忽略请求取消后的 cache 回收。

练习：为语音输入、代码补全、长文档问答各画一条请求时间线，标注 prefill、decode、工具等待、打断和恢复点，再设计对应的 SLO 与 trace 字段。

## 15.12 流式服务的时间预算

在线系统的总延迟不是一个数字。可以拆成：

~~~math
T_{\mathrm{e2e}}
=T_{\mathrm{queue}}
+T_{\mathrm{prefill}}
+T_{\mathrm{decode}}
+T_{\mathrm{tool}}
+T_{\mathrm{postprocess}}
~~~

流式输出还要考虑首 token 时间、每 token 间隔和尾延迟。一个模型即使平均 tokens/s 很高，如果首 token 等待很久，交互体验仍然不好；如果偶发长上下文请求阻塞短请求，p99 会明显恶化。

## 15.13 增量状态和取消语义

流式请求可以在任意 token 后取消，服务端必须释放 KV、递归 state、临时 buffer 和工具 lease。若已经发送部分输出，重试时要明确从哪个 committed prefix 继续，不能把未发送的 speculative token 当成已提交结果。

多轮对话还要区分用户已看到的文本、模型已经算出的内部状态和工具已经提交的副作用。取消只停止生成，不一定能撤销外部写操作，所以工具执行器要有独立事务语义。

## 15.14 把低延迟拆成可控制的时间预算

在线请求的延迟不是一个“模型速度”数字。可以拆成：

```math
T_{\mathrm{e2e}}=T_{\mathrm{queue}}+T_{\mathrm{prefill}}
 +T_{\mathrm{first\ token}}+N_{\mathrm{out}}T_{\mathrm{decode}}
 +T_{\mathrm{tool}}+T_{\mathrm{post}}
```

用户关心的 TTFT 主要受排队和 prefill 影响，TPOT 主要受 decode、KV 读取和 batch 调度影响；在线语音或视频还要加采集、编码、网络抖动和播放缓冲。只优化 kernel 而不看 queue time，可能让单请求 benchmark 变快，线上 p95 却不变。

流式服务应为每个阶段设置预算。输入还没结束时可以用 chunked prefill，输出 token 生成到一半时要允许取消，工具返回慢时要返回明确的等待状态。deadline 逼近时，系统可以缩短输出、减少验证、切换只读路径或转异步，但不能为了赶上延迟直接提交未授权副作用。

## 15.15 增量状态与取消语义

流式推理的状态至少包括已消费输入位置、KV/state、已发送输出、未提交工具动作和客户端 ack。取消请求到达时，服务要区分“停止继续生成”和“撤销已经执行的动作”。已经发送给客户端的 token 不能从网络上收回；已经提交的外部写操作也不能仅靠取消生成回滚。

一个安全状态机可以是：

```text
running -> cancel_requested -> stop_decode -> drain_events
         -> release_state -> completed(cancelled)
```

若处于 `tool_started` 或 `write_committed`，要先查询工具状态并记录结果；若状态未知，转人工或补偿流程。batch reorder、preemption 和 snapshot restore 也要把每个请求的逻辑 token 位置与物理 cache/page 对齐，不能用全局 batch index 作为身份。

## 15.16 Transformer 与递归状态模型的在线取舍

Transformer 的显式 KV 提供精确历史访问，但长会话会增加 cache 读写和 admission 压力；递归状态大小固定，更适合持续流，却把历史压缩进有限表示。在线系统还要考虑断线恢复：KV 可以按页保存或重算，递归 state 需要保证 step、输入和 reset 语义一致。

一个合理的实验矩阵包括短突发、长流、多租户混合长度、随机取消、工具等待和节点迁移。指标包括 TTFT、TPOT、p99、每请求状态字节、恢复延迟、重复副作用率和质量切片。不能用离线长序列吞吐代替在线交互体验。

## 15.17 Admission、deadline 与公平性

流式服务不能只在请求进入后才发现它会占满 cache。接入层应根据输入 token、预估输出、租户优先级和剩余 deadline 计算 admission；否则一个超长 prefill 可能让大量短请求在队列里等待。可以把请求的预算写成：

```math
B_i = B_time_i + B_token_i + B_kv_i
```

其中每一项都不是越大越好。时间预算决定 deadline，token 预算限制输出和 prefill，KV 预算决定是否能继续保留历史。调度器应在这些预算和租户公平之间做选择，例如给实时语音保留较小的 prefill chunk，为离线摘要分配更大的吞吐份额；但降级必须显式记录，不能静默截断用户输入。

公平性也不能只看请求数。一个请求输出 20 个 token，另一个请求输出 20,000 个 token，二者占用的 decode 时间、KV 和网络资源完全不同。可按 token、GPU 时间和 active state 记账，并把取消后释放的资源及时归还到租户配额。否则高并发租户会通过发送大量短请求或保持空闲连接，绕过按请求数设计的限流。

## 15.18 已提交状态、临时状态和副作用

流式生成中至少有三种状态：已经发送给客户端的 committed output，模型已经算出但尚未发送的 speculative output，以及工具已经开始执行的 external side effect。它们的回滚能力不同。取消可以停止后续 decode，通常不能撤回已发送 token，也不能凭空撤销已经提交的数据库写入。

可以用事件序列表达一次安全的取消：

```text
decode -> candidate_token -> client_ack?
       -> tool_prepare -> policy_check -> tool_commit
       -> cancel_requested -> drain -> release
```

如果 `tool_commit` 已发生，结束状态必须包含工具结果和补偿动作；如果只到 `tool_prepare`，可以安全丢弃临时参数；如果 candidate token 尚未发送，应该从 committed prefix 重新生成，而不是把它当成历史继续追加。这个区分同样适用于 speculative decoding：被 target 拒绝的 token 和已经接受的 token 不能共享同一份可提交 state。

对多租户系统，日志还要避免把一个请求的原始音频、代码或工具参数写入另一个请求的恢复快照。状态快照必须绑定 request id、tenant id、model revision、template revision、position offset 和权限上下文，并在恢复时再次做当前权限检查。

## 15.19 从单请求 benchmark 到混合流量演练

低延迟实验至少需要三组 workload：短 prompt/短 output 的交互请求，长 prompt/短 output 的检索请求，以及中等 prompt/长 output 的生成请求。再加入随机到达、随机取消、工具等待、节点迁移和慢客户端，才能看到真实调度压力。

结果表不要只列平均时间，应至少包括 TTFT p50/p95/p99、TPOT p50/p95/p99、队列时间、active requests、KV 使用率、取消清理时间、输出重复率和单位成功成本。对每个失败样本保存状态机事件，判断是 admission 放行过多、chunk budget 太大、客户端 backpressure，还是模型/网络本身慢。

一个实用的上线门可以写成：

```math
G_stream = I[TTFT_p99 <= tau_1] I[TPOT_p99 <= tau_2]
            I[P_leak = 0] I[P_duplicate <= tau_3]
            I[C_cleanup <= tau_4]
```

这里的验收条件值要根据业务确定。`P_leak` 可以表示跨租户状态泄露率，安全系统应直接要求为零；`P_duplicate` 则覆盖重试、恢复和重复工具提交。这组条件比“tokens/s 提升 20%”更接近流式系统是否可用。

## 15.20 流式状态机的不变量

流式调度的难点不是状态名称多，而是每次转换都必须保持不变量。对一个请求，至少要满足：已消费 token 只增长不回退；已提交输出不会被临时候选覆盖；物理 KV/page 或递归 state 与逻辑位置一致；工具副作用要有明确的 prepared/committed 状态；请求 owner 和租户隔离始终成立。

可以把状态写成：

~~~math
\Sigma_i=(r_i, p_i, s_i, o_i, a_i),
~~~

其中 `r_i` 是请求阶段，`p_i` 是逻辑 token 位置，`s_i` 是 KV/state 句柄，`o_i` 是已提交输出，`a_i` 是外部动作状态。一次事件 `e` 应满足：

~~~math
\Sigma_{i+1}=\delta(\Sigma_i,e),
\qquad
\mathrm{owner}(s_{i+1})=\mathrm{owner}(r_{i+1}).
~~~

这比“把请求从 decode 列表移到 cancel 列表”更严格。取消、抢占、超时和恢复都要验证位置、状态和副作用是否同步；否则最终文本可能正常，下一次请求却复用了错误历史。

## 15.21 混合流量下的 admission 和公平性

在线服务不能用平均 prompt 长度做容量规划。对请求 `i`，可以估算它的输入 token `P_i`、输出上限 `O_i`、KV 状态 `K_i` 和 deadline `D_i`，接入条件至少要满足：

~~~math
\sum_{i\in\mathcal{A}}K_i\le K_{\mathrm{free}},
\qquad
\widehat T_i(P_i,O_i)\le D_i.
~~~

但这些估计会有误差。若所有长请求都被放行，短语音请求的 TTFT p99 会恶化；若只按请求数限流，长输出租户可以占满 decode 时间。资源记账应同时考虑 token、GPU 时间、active state 和工具等待，调度器还要给实时请求保留预算。

chunked prefill 的 token budget 也应和 deadline 联动。接近 deadline 时，可以减小 prefill chunk、限制输出、转异步或升级模型；不能静默丢弃输入。每次降级都记录原因和质量变化，才能区分调度策略有效还是服务在悄悄牺牲用户结果。

## 15.22 流式上线条件不只是 tokens/s

一个可操作的流式发布验收条件至少包含四类指标：首 token 和每 token 延迟，状态一致性和跨租户隔离，取消/恢复的资源回收，任务质量和副作用安全。可以用示意式验收条件表示：

~~~math
G_{\mathrm{stream}}
=I[T_{\mathrm{TTFT},p99}\le\tau_1]
 I[T_{\mathrm{TPOT},p99}\le\tau_2]
 I[E_{\mathrm{restore}}\le\epsilon]
 I[P_{\mathrm{leak}}=0]
 I[P_{\mathrm{duplicate\_side\ effect}}\le\tau_3].
~~~

压测场景应覆盖短 prompt/短 output、长 prompt/短 output、中等 prompt/长 output、随机取消、工具等待、慢客户端和节点迁移。对每个失败样本保存状态机事件、queue time、KV/page 变化和客户端 ack，才能判断是模型慢、调度不公平、状态回收错误还是网络 backpressure。

## 15.23 小结与资料边界

Transformer 的并行训练与自回归结构参见 *Attention Is All You Need*；FlashAttention-2 的硬件和访存讨论见 https://arxiv.org/abs/2307.08691；在线服务中的 KV 管理应结合具体 serving engine 文档核对。本文中的时间模型是容量规划的近似，不可替代实际硬件压测。

流式场景的核心不是把模型“变快”一个指标，而是让输入、状态、调度和输出在有限延迟内持续一致。只要历史状态增长、读取带宽或恢复协议成为瓶颈，就需要在显式 KV、压缩状态、检索和混合架构之间做选择。
