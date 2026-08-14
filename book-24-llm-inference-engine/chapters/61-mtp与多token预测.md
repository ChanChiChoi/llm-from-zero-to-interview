# 第 61 章 MTP：多 token 预测如何成为推测解码的候选来源

## 61.1 自回归 decode 为什么慢

decoder-only 模型生成文本时，每一轮通常只提交一个新 token。下一轮的 logits 依赖这个 token 的 hidden state，因此即使 GPU 能够并行矩阵乘法，时间轴上仍然存在一个无法消除的串行依赖。

假设输出需要生成 `N` 个 token，普通 decode 大致需要 `N` 次 target step。若系统能先提出一段候选，再让 target 一次验证多个位置，成功接受的 token 就可以在一次 target 调用中提交。这个想法就是 speculative decoding 的基础。

Multi-Token Prediction（MTP）通常指模型在训练或结构上增加未来多个位置的预测头。MTP 可以帮助模型提出候选，但它本身不是完整的 speculative decoding runtime。候选如何生成、target 如何验证、拒绝后如何采样和 cache 如何回滚，都属于 serving 算法。

## 61.2 MTP 头在预测什么

设主干在位置 `t` 输出 hidden state `h_t`，第 `k` 个未来预测头给出：

```math
p_k(\cdot\mid h_t)=\mathrm{softmax}(W_k h_t+b_k)
```

它试图预测 `y_{t+k}`。如果有 `K` 个头，训练时可以加入多位置交叉熵：

```math
\mathcal{L}_{\mathrm{MTP}}=\sum_{k=1}^{K}\lambda_k\mathcal{L}_{\mathrm{CE}}(p_k,y_{t+k})
```

`\lambda_k` 控制不同未来距离的权重。距离越远，预测不确定性通常越大；头之间是否共享参数、是否使用额外 hidden transform、是否和主 logits 共享输出层，取决于具体架构。

## 61.3 训练目标和推理目标不是同一件事

训练 MTP 头只说明它在训练分布上学习了未来 token 预测。Serving 时，候选会进入 target 验证，面对真实的 sampling、temperature、grammar、tool call 和不同 batch。训练损失下降不自动等于 acceptance length 增加。

尤其要注意 tokenizer 和 template。候选头预测的是 token id；如果 runtime 使用了不同 tokenizer、特殊 token 或 chat template，候选的语义边界就会错位。多模态 placeholder、reasoning channel 和工具 JSON 也可能让“下一个 token”的含义改变。

## 61.4 一轮推测解码发生了什么

可以把一轮过程写成：

```text
context x
  -> MTP proposes d1 ... dK
  -> target verifies all compatible positions
  -> accept prefix d1 ... dA
  -> sample or regenerate at rejection position
  -> commit accepted tokens and cache
```

设候选为 `A B C D`，target 接受 `A B`，在 `C` 处拒绝。正式输出只提交 `A B` 和按 target 规则产生的下一个 token；`C D` 不能进入 committed KV。临时 candidate state 必须清理或回滚。

对 greedy decoding，可以用“接受前缀长度”直观理解收益；对随机采样，runtime 还要处理 rejection sampling 或等价的 residual distribution，不能直接把被拒绝 token 替换成任意 token，否则改变了目标分布。

## 61.5 接受长度决定什么

若每次候选窗口为 `K`，平均接受 `A` 个 token，target 每轮能够推进约 `A+1` 个 token 的教学化估算，但实际收益还要支付 MTP 头、target verify、调度、grammar 和临时 cache 成本。

可以粗略写成：

```math
E_{\mathrm{round}}\approx\frac{A+1}{L_{\mathrm{draft}}+L_{\mathrm{verify}}+L_{\mathrm{sched}}}
```

当 `A` 很低时，额外候选只增加开销；当 `A` 高且 verify kernel 能有效批量化时，TPOT 才可能下降。代码和结构化 JSON 通常有更多重复模式，开放写作或高温度采样的接受长度可能更低。

## 61.6 scheduler 如何管理候选

scheduler 不能把 candidate token 当作正式输出 token。每个请求要区分候选长度、已验证长度、已提交长度和可用 KV block。批处理时，一个低接受率请求不应永久占用很大的候选空间；可以根据历史接受率动态调整窗口或关闭 speculative。

grammar state 也只能随 accepted prefix 前进。若候选包含半个 JSON 参数，target 在中途拒绝，parser 不能先发出一个不完整的 tool call event。streaming 客户端只应看到 committed token。

## 61.7 worked example：计算 target call 减少

假设普通 decode 生成 100 token，需要约 100 次 target step。MTP 每轮提出 4 token，平均接受 2.5 个，拒绝位置再产生一个 target token，平均每轮前进约 3.5 token，于是 target verify 轮数约为 `100/3.5≈29`。

这不等于端到端加速 100/29。每轮还要运行 MTP 头、一次批量 target forward，并产生调度和 cache 管理开销。若普通 target step 是 1 单位，speculative 每轮是 3 单位，端到端理论成本反而可能接近；只有在 target 验证能高效利用并行计算，或者 draft 成本很低时才有明显收益。

## 61.8 如何做 benchmark

固定 target model、MTP checkpoint、tokenizer、chat template、sampling 和输出长度，分别测试代码、JSON、数学、开放写作和工具调用。记录：

```text
drafted_tokens, accepted_tokens, rejected_tokens,
target_calls, MTP_time, verify_time,
TTFT, TPOT, p99, peak_memory, output_equivalence
```

必须和普通 decode 做对照。只报告 target call 减少会高估收益；只报告 tokens/s 又可能漏掉质量变化、峰值显存和 batch 下降。

## 61.9 常见失败

MTP head 与 target checkpoint、tokenizer 或 position 不匹配；拒绝 token 被错误写入正式 KV；grammar state 和 stream event 没有回滚；低接受率 workload 仍然强制启用；candidate cache 造成 OOM；usage 把候选和提交 token 重复计费。

## 61.10 面试回答与练习

回答“MTP 是什么”时，应说它是未来多个 token 的训练目标或附加预测头，可以作为 speculative decoding 的候选来源；target 仍负责批量验证和最终提交。收益取决于接受长度、draft 成本、验证 kernel、cache、采样和调度，不是头数越多越快。

练习一：给定 `K=4`、平均接受 2 个、输出 100 token，估算 target verify 轮数，并说明为什么这不是端到端 speedup。

练习二：设计拒绝发生在第二 token 时的 KV、grammar 和 streaming 回滚测试。

练习三：比较代码、JSON 和开放写作中接受长度可能不同的原因。

### 61.10.1 接受长度与 target 调用

若每次 target 验证平均接受 `a` 个 token，输出长度为 `N`，理想 target 调用次数约为：

```math
N_{\mathrm{target}}\approx\left\lceil\frac{N}{a}\right\rceil
```

但一次 speculative step 还要支付 draft/MTP 计算、候选组织、验证 kernel、KV 管理、grammar 和调度成本。减少 target calls 只说明一个中间收益，不能直接推出端到端 tokens/s 提高。

### 61.10.2 约束生成会改变候选质量

代码、JSON 和自然语言的 token 分布不同。JSON grammar 可能屏蔽 draft 的大部分候选，代码中的固定缩进和关键词可能提高接受长度，开放写作的高熵采样可能降低接受率。评测要固定 sampling、grammar、stop reason 和输出格式，按 workload 分桶。

### 61.10.3 rejected prefix 的一致性

target 拒绝候选后，只有 accepted prefix 可以进入正式 KV；被拒绝 token 的临时 KV、position、grammar state 和 stream delta 都要回滚。客户端不能先看到被拒绝 token 再让服务端撤回，否则输出协议已经不一致。

### 61.10.4 MTP head 的训练和部署

多 token head 的训练目标可能预测未来不同距离的 token。head 数量增多会增加训练和推理计算，也不保证远距离候选质量单调提升。发布 artifact 要绑定 base model、head 参数、tokenizer/template、position、quantization 和 engine 支持。

## 61.11 MTP 的接受长度和回滚

MTP 预测更远 token 不代表 target 一定接受。要分别测 draft 质量、接受长度、verify 成本、rejected prefix 的 state 回滚和 grammar 约束下的退化。

线上指标应按请求和长度分布统计，平均 acceptance 可能掩盖某些语言、代码或工具 JSON 任务几乎没有收益。

## 61.12 MTP 训练目标和服务目标

训练时 MTP 让主干同时预测多个未来位置，增加监督信号；服务时可以把多个预测头当作候选来源，再由 target model 验证。两者相关但不等价：训练中预测得准，不代表候选 token 在目标分布下连续可接受。

不同 head 的远期预测通常不确定性更高。系统应记录各 head 的 top-1、一致性、候选长度、验证拒绝位置和额外参数，而不是只看 MTP loss。

## 61.13 MTP 与 cache 的状态契约

MTP 头使用主干 hidden 和位置状态生成候选。候选验证期间，draft 分支可能暂时更新 KV/state；拒绝后必须回滚。候选 token、position、page、head 输出和 target logits 都属于一次事务。

prefix cache 复用还要确认 MTP head、模型 revision、tokenizer 和模板一致。只复用主干 KV 而忽略附加头版本，可能得到合法但错误的候选。

## 61.14 MTP 的质量验收条件

评估至少包含普通文本、代码、数字、结构化输出、长上下文和工具调用。多 token 候选若在格式敏感任务上更容易被拒绝，平均 acceptance 可能掩盖真实收益。

推测收益要与额外 head 的显存、带宽、验证 kernel、回滚成本和失败回退一起报告。MTP 是候选来源，不是跳过 target verification 的理由。

## 61.15 MTP 训练实现中的位置对齐

MTP 最容易被忽略的地方是监督位置。对于长度为 T 的训练样本，主语言模型通常用位置 t 的 hidden 预测 y_{t+1}；第 k 个未来头则应使用同一个定义清楚的上下文去预测 y_{t+k}。如果数据管线在 shift、padding 或 packed sequence 上多移动了一位，训练 loss 仍然可以下降，但候选会在 serving 中整体错位。

一个最小的对齐表应显式写出：

| 主干位置 | 可见上下文 | 一步头目标 | 两步头目标 | 是否跨样本 |
| --- | --- | --- | --- | --- |
| t | y_0...y_t | y_{t+1} | y_{t+2} | 否 |
| padding 位置 | mask 后不计 loss | 不计 loss | 不计 loss | 否 |
| packed 边界 | 只看当前样本 | 不得读下一样本 | 不得读下一样本 | 否 |

对于第 k 个头，可以写成带有效位置掩码的损失：

~~~math
\mathcal{L}_{k}
=-\frac{1}{\sum_t m_{t,k}}
\sum_t m_{t,k}\log p_k(y_{t+k}\mid h_t),
\qquad
m_{t,k}=\mathbf{1}[t+k<T_{\mathrm{sample}}].
~~~

m_{t,k} 不是装饰。若把跨样本的下一个 token 当成监督目标，MTP 头会学到错误的边界模式；若只在短样本上训练远期头，长输出时 acceptance 可能迅速下降。训练日志应分别报告每个头的有效 token 数、loss、top-1 accuracy 和在真实连续候选上的前缀命中率。

## 61.16 从单点预测到连续候选

未来头的 top-1 准确率并不等于 speculative acceptance。假设每个头独立地以概率 q 预测正确，连续接受 a 个位置的概率近似为：

~~~math
\Pr(A\ge a)\approx q^a.
~~~

真实模型并不满足独立假设：第一个候选一旦错误，后面的候选上下文也会错；反过来，代码中的固定模式会让多个位置高度相关。因此服务评估应直接统计 A 的分布，而不是从各头的单 token accuracy 外推速度。

还要区分两种候选生成方式。独立多头可以从同一个 h_t 同时提出多个未来位置；递归候选则把前一个候选反馈给下一步。前者并行度高但可能不自洽，后者候选更连贯却增加 draft 依赖。一个实现若只写“有多个预测头”，无法判断它属于哪一种，也不能据此推断它的 runtime 成本。

## 61.17 一轮 MTP 的事务伪代码

下面的伪代码刻意把候选状态和正式状态分开。它没有实现具体 logits 或 sampling kernel，却展示了最不可省略的提交边界：

~~~text
state = target.snapshot_committed_state()
candidate = mtp.propose(state.hidden, window=k)
trial = target.verify(candidate, state)

if trial.protocol_ok and trial.accepted_length > 0:
    accepted = trial.accepted_prefix
    target.commit(state, trial.snapshot_after(accepted))
    stream.emit(accepted)
else:
    target.restore(state)
    accepted = target.sample_one(state)
    target.commit(state, accepted.state)
    stream.emit(accepted.token)

discard(trial.uncommitted_pages)
record(candidate, trial, accepted)
~~~

这里的 snapshot 不一定是复制完整 KV；实际引擎可以使用 page 引用、写时复制或可回退的 block cursor。但抽象上的不变量相同：客户端只看 committed token，正式 KV 只包含 committed token，usage 只计算正式 token。取消请求、抢占和超时都必须能在 candidate 阶段安全终止。

## 61.18 MTP 与结构化输出的冲突

开放文本中，候选 token 的局部正确性经常足够；JSON、代码语法和工具参数则需要整个前缀处于合法 grammar state。一个候选可能在自然语言意义上合理，却在字符串、转义或数字小数点位置上不可提交。

因此 grammar 不能在 MTP 之后再“修一下字符串”。正确顺序是让候选生成知道允许集合，或者在验证时把 grammar state 作为 target 的一部分共同检查。测试至少包含字符串转义、Unicode、嵌套数组、缺失字段、stop token 和工具调用半途拒绝。只检查最终 JSON 可解析，会漏掉流式协议已经提前发出非法片段的情况。

## 61.19 MTP artifact 与启停条件

MTP 的发布对象不是一个孤立的 head 文件，而是一组和 target 强绑定的 artifact：base model revision、head revision、hidden shape、position、tokenizer/template、dtype、量化、sampling、grammar 和 engine ABI。启动时可以先做结构检查，再用固定 prefix 比较候选 token；任何 shape 或 token 序列不一致，都应关闭 MTP 回到 target-only。

运行时还需要动态验收条件。低 acceptance、临时 KV 不足、p99 超标、grammar 错误或 target 回滚失败时，不能继续为了平均加速而保持候选路径。关闭动作应在一轮事务提交或回滚后发生，清理候选 page，并在 trace 中记录 `disable_reason`。普通 decode 和 MTP 路径的 usage、stream event 与最终协议要保持同一语义。

## 61.20 实验设计的证据范围

MTP 可以表示训练目标、附加预测头或某个模型的 speculative 候选接口；这些含义不能在没有模型卡或论文的情况下混用。公开资料能确认的通常是目标、接口和部分评估，head 的内部共享方式、hidden 读取位置和 runtime 优化若未披露，就只能写成待验证假设。

一个可复现实验应固定 target checkpoint、MTP artifact、tokenizer、chat template、sampling、grammar、硬件和 batch，比较普通 decode 与 MTP 路径的接受长度分布、target 调用次数、head/verify 时间、峰值显存、p99 和 token-level 等价性。这样才能把“多 token 预测”从一句结构描述推进到可证伪的系统结论。

MTP 的价值不是让模型跳过验证，而是用额外预测信号提出更长的候选；最终收益取决于候选连续性、target 验证、采样正确性、状态回滚和 workload。参考 speculative decoding 的公开论文、MTP 技术报告以及 vLLM/SGLang 的推理文档；具体 head 形态和兼容版本以对应发布物为准。

## 61.21 MTP 的训练目标与 serving 语义

MTP 可以在训练时让 hidden state 预测未来多个位置，也可以在 serving 中被实现为附加预测头。两者相关但不等价：训练目标提供额外的未来 token 信号，serving 还要把候选送入 target 验证、grammar、KV 提交和 stream 协议。没有 target 验证，未来预测只是 draft，不是正式输出。

若第 `k` 个 head 预测 `x_{t+k}`，训练损失可以抽象为：

```math
\mathcal{L}_{\mathrm{MTP}}
=\sum_{k=1}^{K}\lambda_k
\left[-\log p_{\theta,k}(x_{t+k}\mid x_{\le t})\right].
```

不同实现的 head 共享方式、hidden 读取位置和权重并不一定相同。正文能确认的是目标和接口，未公开的内部结构不能由名称推断。评估时要把 head 的候选质量与 target-only 的最终采样语义分开。

## 61.22 候选提交的事务边界

一次 MTP round 至少有 `drafted`、`verified`、`committed` 和 `reclaimed` 四种状态。只有 accepted prefix 进入正式 KV、grammar state、usage 和 stream；rejected suffix 必须释放。候选中的 tool call 或外部写操作在 committed 前只能是 proposal，不能被 executor 执行。

可以用不变量检查实现：

```math
|\mathrm{KV}_{\mathrm{committed}}|
=|\mathrm{output}_{\mathrm{visible}}|,
\qquad
\mathrm{usage}=\mathrm{count}(\mathrm{committed\ tokens}).
```

取消、抢占和 target 超时都必须从 committed state 恢复，不能把临时 candidate 当作可重放输出。golden replay 要覆盖全接受、首 token 拒绝、中间拒绝、stop token 和 grammar 错误。

## 61.23 MTP 的收益分母

MTP 可能减少 target decode step，但会增加 head 计算、验证、临时 KV 和调度复杂度。设每轮平均提交 `a+1` 个 token，候选和验证成本为 `C_c+C_v`，粗略每 token 成本为：

```math
C_{\mathrm{token}}
\approx\frac{C_c+C_v+C_{\mathrm{rollback}}}
{a+1}+C_{\mathrm{schedule}}.
```

`a` 越大不一定越好，因为候选 workspace、验证 batch 和尾延迟可能同步增加。实验要分普通文本、代码、JSON、工具、reasoning 和长上下文，保存 acceptance histogram、TTFT/TPOT、p99、峰值显存、fallback 和单位成功成本。

## 61.24 MTP 的候选窗口如何和 scheduler 配合

候选窗口越大，理论上一次验证能推进更多 token，但临时 KV、grammar 状态和验证计算也随之增加。scheduler 可以根据历史 acceptance、剩余显存、请求 deadline 和输出类型动态调整窗口；低接受率请求应快速缩小窗口或关闭 MTP，不能让一个请求长期占据候选页。

## 61.25 MTP 的 streaming 语义

客户端只能看到 committed prefix。target 在候选中途拒绝时，被拒绝 token、对应 usage、grammar state 和 tool event 都必须被丢弃；若客户端已经收到候选，再试图发送撤回事件，协议就已经失真。端到端测试要检查全接受、首 token 拒绝、中间拒绝、stop token 和取消。

## 61.26 MTP 的收益要按工作负载分桶

代码、JSON、数学和开放写作的候选连续性不同。报告平均 acceptance 会掩盖某些请求几乎没有收益；应给出接受长度分布、draft/verify 时间、target calls、TTFT、TPOT、p99、峰值显存和最终输出等价性。只有验证成本被摊薄时，target call 减少才会变成端到端 speedup。

## 61.27 MTP 与训练 artifact 的绑定

MTP head、base model、tokenizer、template、position、dtype、量化和 engine 支持范围应一起发布。head 的 shape 能加载，只能证明文件可读；固定 prefix 的候选 token、接受长度和 grammar 行为一致，才说明它和 target 语义匹配。

## 61.28 MTP 的故障注入

注入候选全拒绝、target 超时、临时 KV 不足、grammar 错误、客户端断流、取消和 worker 重启，检查 committed prefix、usage、stream、工具事件和资源回收。任一 rejected token 泄露到客户端或正式 KV，都是 hard failure。

## 61.29 MTP 的回退策略

可以按请求、模型、输出类型或 acceptance 动态关闭；关闭时必须在事务提交/回滚后发生，并清理候选页。target-only 路径应和普通服务共用质量、协议和成本监控，不能把回退当成没有指标的黑箱。

## 61.30 MTP 的发布与回退

MTP artifact 至少绑定 base model、head、tokenizer/template、position、dtype、quantization、grammar 和 engine ABI。启动时用固定 prefix 做 candidate/target 对照，运行时按 acceptance、p99、显存和协议错误动态关闭。关闭发生在一轮提交或回滚之后，保留正式 KV，丢弃临时候选。

MTP 的合理定位是一个可回退的候选路径。它在目标 workload 上同时通过 token-level 等价性、结构化输出、工具安全、SLO、恢复和成本约束后才有生产意义；模型卡中的“支持多 token 预测”不能替代这些端到端证据。
