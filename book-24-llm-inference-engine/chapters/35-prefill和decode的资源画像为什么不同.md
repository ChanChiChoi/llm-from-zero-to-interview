# 第 35 章 Prefill 和 Decode 的资源画像为什么不同

上一章结束了 SGLang 架构详解部分。从本章开始进入第五部分：PD 分离与高级 Serving 架构。

PD 分离里的 P 是 Prefill，D 是 Decode。

要理解 PD 分离，不能一上来就画分布式架构图。必须先理解一个更底层的问题：Prefill 和 Decode 虽然都在跑同一个 Transformer 模型，但它们的资源画像完全不同。

如果把二者放在同一个 GPU worker 里混合调度，系统要不断在 TTFT、TPOT、吞吐、显存和公平性之间折中。PD 分离的基本动机，就是把这两类不同形态的工作负载拆到更合适的资源池里执行。

一句话概括：

> Prefill 是一次处理大量输入 token、偏大矩阵计算和 KV 写入的阶段，主要影响 TTFT；Decode 是逐 token 生成、频繁读取 KV cache、重复调度的阶段，主要影响 TPOT。二者资源画像不同，是 PD 分离架构成立的根本原因。

## 35.1 本章目标

读完本章，你应该能讲清：

1. Prefill 和 Decode 在输入、输出和执行频率上的差异。
2. 为什么 Prefill 更偏大计算，Decode 更偏 KV cache 读取和调度。
3. TTFT、TPOT 分别由哪些阶段主导。
4. 为什么长 Prefill 会干扰 Decode streaming。
5. 为什么 Decode-first 会导致 Prefill 饥饿。
6. PD 分离为什么要先从资源画像差异说起。
7. 面试中如何解释 Prefill/Decode disaggregation 的动机。

## 35.2 再看一次生成生命周期

一个 LLM 请求不是一次 forward 完成。

完整生命周期是：

```text
prompt tokens
  -> prefill
  -> first token
  -> decode token 2
  -> decode token 3
  -> decode token 4
  -> ...
  -> finish
```

Prefill 处理输入 prompt。

Decode 处理输出生成。

如果用户输入 4000 tokens，输出 800 tokens，那么：

```text
prefill: 处理 4000 个输入 tokens，通常一次或分 chunk 完成
decode: 生成 800 个输出 tokens，通常需要 800 轮左右
```

这两段的形态完全不同。

Prefill 是“宽”的：一次处理很多 token。

Decode 是“长”的：每次处理少量 token，但重复很多轮。

## 35.3 Prefill 的执行画像

Prefill 阶段输入是整个 prompt 或 prompt chunk。

例如：

```text
input_ids: [t1, t2, ..., t4096]
```

模型要计算这些 token 的 hidden states，并为每层 attention 写入 KV cache。

Prefill 的特点：

1. 单次 token 数多。
2. 矩阵乘法规模较大。
3. GPU 算力利用率通常更容易做高。
4. 计算量随 prompt 长度上升。
5. 会产生大量 KV cache 写入。
6. 对 TTFT 影响直接。

从资源倾向看，Prefill 常被认为更偏 compute-bound。

这不是说它没有显存和带宽压力，而是说相对 Decode，它更容易把 GPU 计算单元喂饱。

## 35.4 Decode 的执行画像

Decode 阶段每轮通常只输入上一步生成的 token。

例如 running batch 有 128 个请求：

```text
decode input: 每个请求 1 个 token，总计 128 tokens
```

模型要为每个请求读取历史 KV cache，计算下一个 token 的 logits，再采样。

Decode 的特点：

1. 每个请求每轮 token 数很少。
2. 需要重复很多轮。
3. 每轮都要读取历史 KV cache。
4. 上下文越长，KV 读取越重。
5. 对调度开销敏感。
6. 对 streaming 平滑度影响直接。
7. 主要影响 TPOT。

Decode 常被认为更偏 memory-bound 或 bandwidth-sensitive。

因为它不断读取历史 KV，单轮计算量相对小，GPU 计算单元不一定容易吃满。

## 35.5 一张表总结差异

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 处理内容 | 输入 prompt | 输出 token |
| 每请求每轮 token 数 | 多个 token | 通常 1 个 token |
| 执行频率 | 每个请求通常一次或少数 chunks | 每个输出 token 一轮 |
| 主要指标 | TTFT | TPOT、ITL、输出吞吐 |
| 资源倾向 | 更偏 compute-bound | 更偏 memory-bandwidth / scheduling-sensitive |
| KV cache 行为 | 大量写入 prompt KV | 读取历史 KV，并追加新 KV |
| 对 batch 的需求 | 大 token batch 提高算力利用 | 大 sequence batch 提高 decode 吞吐 |
| 风险 | 长 prompt 阻塞 decode | 长输出占用 KV 和调度轮次 |
| 常见优化 | prefix cache、chunked prefill、PD prefill pool | continuous batching、KV layout、speculative decoding |

这张表是 PD 分离的基础。

## 35.6 TTFT 为什么主要看 Prefill

TTFT 是 time to first token。

可以拆成：

```text
TTFT = queue time + preprocess time + prefill time + first token sampling + network flush
```

其中 prefill time 经常是大头，尤其是长 prompt 场景。

影响 prefill time 的因素：

1. 输入 token 数。
2. 是否命中 prefix cache。
3. 是否使用 RadixAttention。
4. prefill batch 大小。
5. 是否 chunked prefill。
6. GPU compute 能力。
7. attention backend。
8. 是否被 decode workload 干扰。

如果一个请求排队很久，TTFT 也会高。

如果 prompt 很长但 prefix cache 命中大段，TTFT 可能下降。

所以优化 TTFT 时，要看 queue 和 prefill，而不是只看 decode kernel。

## 35.7 TPOT 为什么主要看 Decode

TPOT 是 time per output token。

可以理解为：

```text
TPOT = decode 阶段总耗时 / 输出 token 数
```

影响 TPOT 的因素：

1. Decode step latency。
2. Running batch size。
3. KV cache 读取效率。
4. 上下文长度。
5. Attention kernel。
6. Sampling 开销。
7. Structured output grammar mask。
8. Streaming 输出阻塞。
9. Speculative decoding 接受率。

TPOT 直接决定流式输出是否顺滑。

如果 decode step 被长 prefill 打断，用户会看到输出卡顿。

这就是 Prefill 和 Decode 混跑时最常见的冲突。

## 35.8 长 Prefill 如何干扰 Decode

假设系统中有 100 个请求正在流式 decode。

每轮 decode 都应该稳定输出 token。

突然来了一个 64000-token prompt。

如果 scheduler 把这个长 prefill 一次性放进 GPU 执行，那么 decode 请求可能长时间得不到调度。

用户看到的现象是：

```text
前面输出很顺滑
突然卡住几秒
然后继续输出
```

这不是网络问题，而是 decode 被长 prefill 阻塞。

缓解办法包括：

1. Chunked prefill。
2. Decode-first scheduling。
3. 限制单 step prefill tokens。
4. 隔离长 prompt。
5. PD 分离，把 prefill 放到独立资源池。

PD 分离是更架构化的解决方式。

## 35.9 Decode-first 为什么也有问题

为了保护 streaming，很多 scheduler 会偏向 decode。

策略大概是：

```text
先调度 running decode
再用剩余 token budget 调度 waiting prefill
```

这能让已有请求输出更平滑。

但如果系统 decode 压力长期很高，新请求可能一直等不到 prefill。

用户看到的现象是：

```text
请求发出去后很久没有首 token
但一旦开始输出，后续速度还可以
```

这就是 prefill 饥饿。

所以 Prefill/Decode 混合调度天然是矛盾的：

```text
偏 prefill -> TTFT 好，TPOT 可能差
偏 decode -> TPOT 好，TTFT 可能差
```

PD 分离试图通过资源隔离减少这个矛盾。

## 35.10 GPU 利用率视角

Prefill 和 Decode 对 GPU 利用率的表现不同。

Prefill：

1. 大矩阵乘法更容易吃满 Tensor Core。
2. 单次 batch token 多时吞吐高。
3. 长 prompt 可能让 GPU 忙于计算。

Decode：

1. 每请求每轮 token 少。
2. 小 batch decode 可能 GPU 利用率低。
3. 大 running batch 能提高利用率，但可能增加延迟。
4. KV cache 读取和调度开销更明显。

所以一个 GPU 同时跑 prefill 和 decode 时，资源使用会不断切换形态。

这对 kernel、batch shape、CUDA graph、attention backend 都不友好。

## 35.11 KV cache 视角

Prefill 和 Decode 都和 KV cache 有关，但行为不同。

Prefill：

```text
写入 prompt 的 KV cache
```

Decode：

```text
读取已有 KV cache
追加新 token KV cache
```

Prefill 的 KV 写入可能一次很多。

Decode 的 KV 读取频繁且持续。

长上下文 decode 时，每轮都要关注大量历史 KV。

这导致 Decode 对 KV layout、memory bandwidth 和 cache locality 更敏感。

PD 分离后，还会引入新问题：Prefill 生成的 KV cache 要如何交给 Decode worker？

这就是后面章节要讲的 KV cache 迁移、共享和路由。

## 35.12 Batch shape 视角

Prefill batch 和 Decode batch 的 shape 也不同。

Prefill batch 更像：

```text
total prefill tokens = sum(prompt_or_chunk_lengths)
```

Decode batch 更像：

```text
decode tokens = number_of_running_sequences
```

举例：

```text
Prefill batch:
  request A: 4000 tokens
  request B: 2000 tokens
  request C: 1000 tokens
  total = 7000 tokens

Decode batch:
  128 running requests
  total = 128 tokens this step
```

这两个 batch 对 kernel 和调度的要求完全不同。

混在一起会造成 batch shape 不稳定。

## 35.13 Latency SLO 视角

在线系统通常同时有 TTFT SLO 和 TPOT SLO。

例如：

```text
p95 TTFT < 2s
p95 TPOT < 80ms/token
```

Prefill 主要威胁 TTFT。

Decode 主要威胁 TPOT。

混部时，一个策略很难同时让两者最优。

如果为了降低 TTFT，积极接纳长 prefill，TPOT 可能抖。

如果为了稳定 TPOT，decode-first，TTFT 可能升。

PD 分离的目标不是魔法消除成本，而是让两个 SLO 可以分别用不同资源池和调度策略治理。

## 35.14 成本视角

成本不是只看 GPU 数。

要看每类 GPU 在做什么。

如果 decode workload 在一张适合大矩阵计算的 GPU 上小 batch 跑，GPU 可能利用率不高。

如果 prefill workload 被 decode-heavy 调度打断，prefill 吞吐也上不去。

资源画像不同意味着可以考虑：

1. Prefill pool 使用更适合大计算的配置。
2. Decode pool 使用更适合低延迟和 KV 读取的配置。
3. 不同 pool 设置不同 batch/token budget。
4. 不同 pool 独立扩缩容。
5. 根据流量输入输出比例调整 P/D 比例。

这就是 PD 分离走向资源池化的原因。

## 35.15 输入输出比例决定压力

不同业务的 prefill/decode 压力不同。

短输入长输出：

```text
prompt 100 tokens, output 2000 tokens
```

Decode 压力大。

长输入短输出：

```text
prompt 16000 tokens, output 100 tokens
```

Prefill 压力大。

长输入长输出：

```text
prompt 32000 tokens, output 4000 tokens
```

Prefill 和 Decode 压力都大。

因此 PD 分离资源配比不能固定照抄。

要看真实流量的输入输出 token 分布。

## 35.16 三类典型 workload

RAG 问答：

```text
长文档上下文 + 相对短答案
```

更偏 prefill 压力，prefix cache 和 prefill pool 重要。

代码生成：

```text
中等 prompt + 长输出
```

更偏 decode 压力，TPOT、speculative decoding 和 decode pool 重要。

Agent：

```text
多轮短生成 + 工具等待 + 上下文逐步增长
```

既要看多轮 TTFT，也要看工具延迟和 cache locality。

不同 workload 的 PD 分离收益不同。

## 35.17 为什么 chunked prefill 还不够

Chunked prefill 可以缓解长 prefill 阻塞 decode。

它把长 prompt 拆成多个 chunk，让 decode 在 chunk 之间穿插执行。

但它仍然是在同一资源池里调度：

```text
同一组 GPU 同时承担 prefill 和 decode
```

这意味着：

1. P/D 仍然争抢同一 GPU。
2. Scheduler 仍然要做复杂折中。
3. Batch shape 仍然混合。
4. 资源扩缩容不能按 P/D 独立做。
5. 长尾流量仍可能互相影响。

PD 分离是更进一步：

```text
不同资源池分别处理 prefill 和 decode
```

当然，它也引入 KV 迁移和系统复杂度。

## 35.18 PD 分离的基本动机

现在可以概括 PD 分离的动机。

因为 Prefill 和 Decode：

1. 资源倾向不同。
2. 延迟指标不同。
3. batch shape 不同。
4. 调度目标不同。
5. KV cache 行为不同。
6. 扩缩容压力不同。

所以可以考虑把它们拆开：

```text
Prefill workers:
  负责处理 prompt，生成 prompt KV cache 和 first token

Decode workers:
  负责接收 KV cache，持续生成后续 tokens
```

这样做的目标是：

1. Prefill 不阻塞 Decode。
2. Decode 不让 Prefill 饥饿。
3. P/D 资源可以独立扩缩容。
4. P/D scheduler 可以各自优化。
5. 长尾影响更容易隔离。

## 35.19 PD 分离不是免费午餐

PD 分离也有代价。

最大问题是 KV cache 如何从 Prefill worker 到 Decode worker。

需要考虑：

1. KV cache 传输带宽。
2. 网络延迟。
3. GPU-GPU、GPU-CPU、跨节点通信。
4. KV cache layout 兼容。
5. 请求路由和状态同步。
6. 失败恢复。
7. 多租户隔离。
8. 调试复杂度。

如果 KV 迁移成本太高，PD 分离收益可能被抵消。

所以 PD 分离适合在 P/D 干扰明显、资源池化收益明显、网络和工程能力能支撑时使用。

## 35.20 什么时候不需要 PD 分离

并不是所有系统都需要 PD 分离。

不一定需要的情况：

1. 流量规模小。
2. 请求长度短。
3. P/D 干扰不明显。
4. 单机 continuous batching 已经满足 SLO。
5. 团队还没有足够监控和调度能力。
6. KV 迁移成本高于收益。
7. 系统复杂度优先级低。

过早引入 PD 分离可能让系统更难稳定。

正确顺序通常是：

```text
先做好单机 serving 和指标
  -> 观察 P/D 干扰
  -> 优化 scheduler 和 chunked prefill
  -> 再评估 PD 分离
```

## 35.21 观测指标

要判断是否需要 PD 分离，先看指标。

Prefill 指标：

1. prefill latency。
2. input tokens/s。
3. scheduled prefill tokens。
4. prefill queue time。
5. prefix cache hit length。
6. chunked prefill 次数。

Decode 指标：

1. decode step latency。
2. TPOT / ITL。
3. output tokens/s。
4. running sequences。
5. KV cache read pressure。
6. speculative accepted tokens。

干扰指标：

1. 长 prefill 到来时 TPOT 是否抖动。
2. decode-heavy 时 waiting queue 是否增长。
3. p99 TTFT 和 p99 TPOT 是否交替恶化。
4. GPU utilization 是否随 batch shape 大幅波动。
5. KV cache allocation failure 是否集中发生。

没有指标就谈 PD 分离，很容易拍脑袋。

## 35.22 面试官会怎么问

问题一：为什么 Prefill 和 Decode 的资源画像不同？

回答要点：Prefill 一次处理大量 prompt tokens，更偏大矩阵计算和 KV 写入，影响 TTFT；Decode 每轮每请求通常一个 token，反复读取历史 KV，调度频繁，更影响 TPOT 和 streaming。

问题二：为什么长 Prefill 会影响 Decode？

回答要点：如果在同一 GPU worker 上一次性执行长 prompt prefill，会占用计算资源，running decode 请求得不到及时调度，导致 TPOT/ITL 抖动，用户看到 streaming 卡顿。

问题三：Decode-first 有什么问题？

回答要点：它保护 running 请求输出平滑，但在 decode 压力高时 waiting prefill 可能长期得不到调度，导致新请求 TTFT 变差。

问题四：PD 分离为什么能缓解这个矛盾？

回答要点：把 prefill 和 decode 放到不同资源池，各自用适合的 batch、调度和扩缩容策略，减少 P/D 互相干扰。

问题五：PD 分离的主要代价是什么？

回答要点：KV cache 需要从 prefill worker 迁移或共享给 decode worker，会带来网络带宽、延迟、layout、路由、状态同步和故障恢复复杂度。

## 35.23 标准回答模板

如果面试官问“为什么要做 PD 分离”，可以这样回答：

```text
PD 分离的根本原因是 Prefill 和 Decode 的资源画像不同。Prefill 处理输入 prompt，通常一次处理很多 tokens，矩阵乘法规模较大，更偏 compute-bound，并且会大量写入 KV cache，主要影响 TTFT。Decode 则是逐 token 生成，每个请求每轮通常只处理一个 token，要频繁读取历史 KV cache，重复调度很多轮，更偏 memory bandwidth 和调度敏感，主要影响 TPOT 和 streaming 平滑度。

如果二者在同一个 GPU worker 上混跑，长 prefill 可能阻塞 running decode，导致输出卡顿；反过来，如果 scheduler 长期 decode-first，新请求 prefill 会饥饿，TTFT 变差。Chunked prefill 能缓解，但仍然是在同一资源池里折中。

PD 分离把 prefill 和 decode 拆到不同资源池，让 prefill pool 专注处理输入和生成 prompt KV，让 decode pool 专注持续生成输出 token。这样 P/D 可以使用不同调度策略、batch 形态和扩缩容比例。但它不是免费午餐，核心代价是 KV cache 要在 P/D worker 之间迁移或共享，网络带宽、延迟、状态同步和故障恢复都会变复杂。
```

## 35.24 小练习

1. 给定 prompt 8000 tokens、输出 200 tokens，判断主要压力偏 prefill 还是 decode。
2. 给定 prompt 200 tokens、输出 4000 tokens，判断主要压力偏 prefill 还是 decode。
3. 画出长 prefill 阻塞 decode 的时间线。
4. 解释 decode-first 为什么会造成 prefill 饥饿。
5. 设计一个 dashboard，用于判断是否需要 PD 分离。
6. 列出 PD 分离后 KV cache 迁移需要解决的 5 个问题。
7. 对比 chunked prefill 和 PD 分离分别缓解什么。

## 35.25 本章总结

Prefill 和 Decode 的差异，是理解高级 serving 架构的起点。

Prefill 更像大块输入计算，影响 TTFT；Decode 更像长时间逐 token 服务，影响 TPOT 和 streaming。二者混合调度会天然产生冲突：prefill 过多会卡 decode，decode-first 又会让 prefill 饥饿。

PD 分离的动机就是把这两类资源画像不同的 workload 拆开，用不同资源池、不同调度策略和不同扩缩容方式分别治理。

下一章会正式讲 PD 分离解决什么问题，并进一步展开它的收益、代价、适用场景和反模式。
