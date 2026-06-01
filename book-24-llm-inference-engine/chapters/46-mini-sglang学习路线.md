# 第 46 章 mini-sglang 学习路线

前两章分别讲了 nano-vLLM 和 tiny-llm。

nano-vLLM 更适合学习 vLLM-like engine 的骨架：`Sequence`、`Scheduler`、`BlockManager`、`ModelRunner`、attention、sampler。

tiny-llm 更适合从 attention、RoPE、GQA、generate、KV cache、continuous batching、chunked prefill、paged attention 这条线理解推理系统如何从模型 forward 演进出来。

这一章看 `mini-sglang`。

本书第 34 章已经讲过 mini-sglang 的源码学习路径，重点是如何沿请求生命周期读源码。本章不重复那条源码阅读清单，而是把 mini-sglang 放在第六部分的教学项目序列里，回答一个更系统的问题：

```text
读完 nano-vLLM 和 tiny-llm 后，为什么还需要读 mini-sglang？它能补上哪一块能力？
```

一句话概括：

> mini-sglang 的学习价值在于把你从“通用 vLLM-like serving engine”带到“SGLang-style runtime”：重点理解 Radix Cache、prefix sharing、overlap scheduling、online serving、structured generation、tool/agent serving，以及这些能力如何围绕复杂 LLM program 的执行效率展开。

## 46.1 本章目标

读完本章，你应该能讲清：

1. mini-sglang 和 nano-vLLM、tiny-llm 的学习重点有什么不同。
2. 为什么 mini-sglang 是理解 SGLang 的过渡项目。
3. 应该重点看 mini-sglang 的哪些能力，而不是重复读 vLLM-like 基础模块。
4. Radix Cache 和普通 prefix cache 的区别。
5. Overlap scheduling 解决什么问题。
6. OpenAI-compatible online serving、shell、offline benchmark 分别适合验证什么。
7. mini-sglang 和完整 SGLang 的差距在哪里。
8. 如何基于 mini-sglang 做一个有辨识度的推理系统项目。

## 46.2 三个教学项目的分工

可以把 nano-vLLM、tiny-llm、mini-sglang 看成三块拼图。

| 项目 | 重点 | 最适合回答的问题 |
|---|---|---|
| tiny-llm | 从算子到 serving 优化 | KV cache、FlashAttention、continuous batching 为什么出现 |
| nano-vLLM | vLLM-like engine 骨架 | 一个轻量 serving engine 如何组织请求和 KV block |
| mini-sglang | SGLang-style runtime | 复杂 LLM program、prefix sharing、结构化输出和在线服务如何高效执行 |

如果只读 tiny-llm，可能会偏底层算子和单机推理。

如果只读 nano-vLLM，可能会理解 engine 骨架，但对 SGLang 的 prefix/program runtime 特色不够敏感。

如果只读 mini-sglang，可能会看到很多 runtime 能力，但缺少前面两个项目提供的基础坐标系。

推荐顺序是：

```text
tiny-llm -> nano-vLLM -> mini-sglang
```

对应认知升级：

```text
模型如何算
  -> engine 如何调度
  -> runtime 如何执行复杂 LLM program
```

## 46.3 mini-sglang 的核心特征

mini-sglang 是 SGLang 的紧凑实现，目标是用更小代码解释现代 LLM serving 系统的复杂性。

从公开介绍看，它的核心能力包括：

1. Radix Cache。
2. Chunked Prefill。
3. Overlap Scheduling。
4. Tensor Parallelism。
5. FlashAttention / FlashInfer 等优化 kernel。
6. OpenAI-compatible online serving。
7. Interactive shell。
8. Offline / online benchmark。

这些能力说明它已经不只是教学级 generate loop，而是一个更接近真实 serving runtime 的项目。

学习时应该抓住一个主线：

```text
mini-sglang 如何在复杂请求、共享前缀、长上下文和在线服务场景下，尽量复用已有计算并稳定推进 decode？
```

## 46.4 和第 34 章的关系

第 34 章已经给过源码阅读路径：

```text
入口 API
  -> 请求状态
  -> scheduler
  -> KV cache
  -> RadixAttention / prefix sharing
  -> model runner
  -> sampler
  -> structured output / streaming
```

本章的角度不同。

本章更关注：

1. 读 mini-sglang 前你应该已经从 tiny-llm、nano-vLLM 得到什么基础。
2. 读 mini-sglang 时应该特别强化哪些 SGLang 特有概念。
3. 如何把它变成一个可展示的系统项目。
4. 如何从 mini-sglang 过渡到完整 SGLang。

也就是说，第 34 章是“怎么读源码”，本章是“为什么读、读出什么、怎么用它做项目”。

## 46.5 先带着 SGLang 问题读

读 mini-sglang 前，先回忆 SGLang 要解决的问题。

vLLM-like engine 主要关注：

```text
大量独立请求如何高吞吐、低延迟地生成。
```

SGLang 进一步关注：

```text
复杂 LLM programs 如何高效执行。
```

复杂 LLM programs 可能包括：

1. 多轮对话。
2. few-shot prompt。
3. 多分支生成。
4. structured output。
5. tool calling。
6. agent loop。
7. RAG prompt assembly。
8. 多请求共享长前缀。

这些场景有一个共同点：

```text
很多请求之间存在可复用前缀，很多请求内部存在结构化执行状态。
```

所以读 mini-sglang 时，不要只问它是不是有 scheduler 和 KV cache。

要问：

```text
它如何发现和复用 prefix？
它如何把复杂请求映射成 runtime 可调度的状态？
它如何让 decode 稳定推进，同时隐藏 CPU 调度开销？
```

## 46.6 第一重点：Radix Cache

Radix Cache 是 mini-sglang 最值得重点看的能力。

普通 prefix cache 可以理解成：

```text
hash(prefix tokens) -> cached KV
```

如果完整 prefix 命中，就复用。

Radix Cache 更像一棵前缀树：

```text
root
  -> shared prefix A
      -> branch B
      -> branch C
```

它适合处理部分共享前缀。

例如三个请求：

```text
R1: system prompt + tool spec + question A
R2: system prompt + tool spec + question B
R3: system prompt + tool spec + question C
```

它们共享：

```text
system prompt + tool spec
```

Radix Cache 的目标是让这段共享前缀的 KV 不要重复 prefill。

读 Radix Cache 时，重点看七个操作：

1. prefix match。
2. longest prefix lookup。
3. node split。
4. insert。
5. pin / unpin。
6. eviction。
7. cached KV 和 request KV 的 ownership。

最重要的问题是 ownership：

```text
命中的 cached KV 是被当前 request 独占，还是多个 request 共享？
当前 request 结束时，哪些 KV 可以释放，哪些要留在 radix cache？
```

如果这个问题没想清楚，很容易写出 KV 泄漏或错误释放。

## 46.7 Radix Cache 和 PagedAttention 的关系

很多人会混淆 Radix Cache 和 PagedAttention。

它们解决的问题不同。

PagedAttention 解决：

```text
一个请求的 KV 如何用不连续 block 管理，减少显存浪费和碎片。
```

Radix Cache 解决：

```text
多个请求之间的共享前缀 KV 如何复用，减少重复 prefill。
```

可以这样理解：

```text
PagedAttention: logical positions -> physical blocks
Radix Cache: prefix token path -> reusable KV blocks
```

两者可以组合：

```text
Radix tree node 持有一段 prefix 对应的 KV blocks；request 命中 prefix 后，把这些 blocks 接到自己的 block table 或 KV reference 上。
```

面试中常见错误是说：

```text
有了 PagedAttention 就不需要 prefix cache。
```

这是错的。

PagedAttention 主要是内存管理方式，prefix/Radix Cache 是计算复用策略。

## 46.8 第二重点：Chunked Prefill

mini-sglang 支持 chunked prefill，这和前面第 39 章、tiny-llm 第 2 周内容形成呼应。

在 SGLang 场景中，chunked prefill 更重要，因为 prompt 可能非常长：

1. system prompt。
2. few-shot examples。
3. tool definitions。
4. RAG retrieved chunks。
5. conversation history。
6. schema / grammar spec。

如果一次性 prefill，decode 会被长时间阻塞。

Chunked prefill 的关键问题是：

1. chunk size 如何定。
2. chunk 之间是否允许 decode 插队。
3. prefix cache 命中后剩余 suffix 如何 chunk。
4. position 如何连续。
5. KV 如何逐 chunk 写入。
6. 如果请求取消，已经写入的 partial KV 如何清理。

读 mini-sglang 时，要把 chunked prefill 和 Radix Cache 一起看：

```text
先命中共享 prefix，再对未命中的 suffix 做 chunked prefill。
```

这才是长 prompt serving 的真实路径。

## 46.9 第三重点：Overlap Scheduling

mini-sglang 的一个重要能力是 overlap scheduling。

它要解决的是：

```text
CPU 侧调度、请求处理、metadata 构造，不要阻塞 GPU 计算。
```

一个 naive engine 可能是：

```text
CPU schedule
  -> prepare batch metadata
  -> GPU forward
  -> CPU update states
  -> next schedule
```

这会导致 GPU 在 CPU 调度期间空等。

Overlap scheduling 的思路是：

```text
GPU 正在执行 batch N 时，CPU 准备 batch N+1 的调度和 metadata。
```

简化表示：

```text
time ---&gt;

CPU: schedule N      schedule N+1      schedule N+2
GPU:        forward N        forward N+1        forward N+2
```

读 overlap scheduling 时，重点看：

1. 哪些工作在 CPU 侧。
2. 哪些工作在 GPU 侧。
3. 两者之间如何同步。
4. 请求状态何时可见。
5. batch metadata 是否可以提前构造。
6. 取消、abort、完成请求如何处理。
7. 关闭 overlap 后性能如何变化。

这个能力很适合做实验：

```text
开启 overlap scheduling vs 关闭 overlap scheduling
比较吞吐、TPOT 和 GPU utilization
```

## 46.10 第四重点：Online Serving

nano-vLLM 和 tiny-llm 更偏 offline 或教学执行路径。

mini-sglang 更值得看的一个点是 online serving。

Online serving 和 offline inference 的差异是：

```text
offline: 一批请求已经给定，系统尽快处理完
online: 请求持续到达，要处理排队、streaming、取消、超时、负载和错误
```

读 online serving 时，重点看：

1. OpenAI-compatible API 如何接入。
2. chat messages 如何转 prompt。
3. sampling params 如何解析。
4. streaming response 如何返回。
5. 请求如何进入 runtime 队列。
6. 客户端断开如何 abort。
7. 错误如何返回。

这部分能补上教学项目常缺的服务化视角。

如果你想做简历项目，online serving 比单纯 offline generate 更有说服力。

## 46.11 第五重点：Interactive Shell

mini-sglang 支持 interactive shell。

Shell 看起来只是一个 demo，但它能帮助你观察多轮对话状态。

重点看：

1. 多轮 history 如何保存。
2. `/reset` 如何清理状态。
3. 每轮 prompt 如何拼接。
4. 多轮共享前缀是否能命中 Radix Cache。
5. streaming token 如何打印。
6. 用户中断如何处理。

多轮对话是 prefix sharing 的天然场景。

例如：

```text
turn 1: system + user1
turn 2: system + user1 + assistant1 + user2
turn 3: system + user1 + assistant1 + user2 + assistant2 + user3
```

后续轮次包含前面轮次的大量 prefix。

这正是 Radix Cache 可以发挥作用的地方。

## 46.12 第六重点：Tensor Parallelism

mini-sglang 支持 tensor parallel。

这让它比很多教学项目更接近生产 serving。

读 TP 时，不需要一开始看所有通信细节。

先看：

1. `--tp` 参数如何进入配置。
2. worker 或 model runner 如何知道自己的 rank。
3. 权重如何 shard。
4. linear 层如何切分。
5. attention heads 如何分配。
6. forward 中哪些地方需要通信。

然后再连接第 24 章和第 41 章：

```text
TP 可以承载更大模型，但 decode 阶段每层通信会进入 TPOT 关键路径；跨节点 TP 尤其要谨慎。
```

mini-sglang 的 TP 适合用来理解生产系统里的模型并行入口。

## 46.13 第七重点：Kernel Backend

mini-sglang 集成 FlashAttention、FlashInfer 等优化 kernel。

读 kernel backend 时，不要陷入每个 CUDA kernel 的实现细节。

先看接口边界：

1. runtime 如何选择 attention backend。
2. prefill 和 decode 是否使用不同 kernel。
3. attention metadata 如何传入 kernel。
4. KV layout 对 kernel 有什么要求。
5. fallback 路径是什么。

Serving engine 的关键不是“会写所有 kernel”，而是知道：

```text
scheduler 和 KV manager 产出的 metadata 必须满足 kernel backend 的输入约束。
```

如果 metadata 错了，kernel 再快也没用。

## 46.14 第八重点：Structured Generation

完整 SGLang 很强调 structured generation 和 constrained decoding。

mini-sglang 是否完整覆盖所有 grammar 能力，会随版本变化。

学习时应该关注接口边界，而不是强行要求教学项目覆盖全部能力。

结构化生成的核心链路是：

```text
schema / regex / grammar
  -> grammar state
  -> valid token mask
  -> sampler
  -> update grammar state
```

读 mini-sglang 时，重点问：

1. 请求中是否能携带结构化约束。
2. 约束在哪里解析。
3. 每个 request 的 grammar state 存在哪里。
4. valid token mask 是否进入 sampler。
5. streaming 时如何保证结构合法。
6. 约束失败如何处理。

如果 mini-sglang 没有完整实现，也可以把它作为设计练习：

```text
如何给 mini-sglang 增加一个最小 regex constrained decoding？
```

## 46.15 第九重点：Tool / Agent Serving

SGLang 的另一个价值是更适合复杂 LLM program。

Tool / Agent serving 不是简单多一个 HTTP API。

它会影响 runtime：

1. Prompt 里有 tool schema。
2. 输出可能是 tool call JSON。
3. Streaming 时 tool arguments 需要增量拼接。
4. Tool result 会回灌成下一轮上下文。
5. 多轮 agent loop 会产生大量共享 prefix。
6. structured generation 可能约束 tool call 格式。

读 mini-sglang 时，可以关注：

1. chat template 是否支持 tools。
2. tool schema 如何拼进 prompt。
3. 输出 parser 是否和 runtime 解耦。
4. tool call 是否只是解析，不负责真实执行。
5. agent loop 是否留给应用层。

真实工具执行通常不应该放进推理引擎核心，因为它涉及权限、网络、超时、安全和审计。

推理 runtime 更适合负责：

```text
生成和解析 tool call，并高效维护上下文。
```

## 46.16 mini-sglang 和完整 SGLang 的差距

mini-sglang 是教学项目，不是完整生产系统。

它可能省略或简化：

1. 更多模型架构。
2. 更多硬件后端。
3. 更完整的分布式部署。
4. 更复杂的 grammar backend。
5. 更完整的 speculative decoding。
6. 更复杂的 tool parser。
7. 多租户隔离。
8. 完整 metrics、tracing、dashboard。
9. 故障恢复和自动扩缩容。
10. 更复杂的 PD 分离和跨节点 serving。

但它保留了学习 SGLang 的关键骨架：

```text
runtime + scheduler + KV cache + Radix Cache + model runner + sampler + online serving
```

这就是它的价值。

## 46.17 和生产 SGLang 源码怎么衔接

读完 mini-sglang 后，不要直接在完整 SGLang 里乱搜。

应该带着映射去读。

映射方式：

| mini-sglang 认知 | 完整 SGLang 中要找的能力 |
|---|---|
| request state | 更完整的请求对象、会话状态、abort 状态 |
| scheduler | 更复杂的调度策略、token budget、prefill/decode 策略 |
| Radix Cache | 更完整的 prefix cache、eviction、ref count |
| model runner | 多模型、多后端、CUDA graph、attention backend |
| sampler | penalties、logprobs、grammar mask、speculative verify |
| online serving | OpenAI API、streaming、metrics、错误处理 |
| TP | 分布式 worker、通信、rank 管理 |

这样读完整 SGLang 时，每个复杂模块都有锚点。

## 46.18 适合做的改造练习

mini-sglang 的练习应该围绕 SGLang 特色展开。

练习一：Radix Cache 可视化。

构造多个共享 system prompt 的请求，打印 radix tree split、insert、match、evict。

练习二：prefix 命中率统计。

统计请求的 prefix hit length、hit ratio、节省 prefill token 数。

练习三：chunked prefill ablation。

对比不同 chunk size 下 TTFT、TPOT p99、吞吐和 GPU utilization。

练习四：overlap scheduling ablation。

关闭 overlap scheduling，对比吞吐和 GPU 空闲时间。

练习五：online serving 压测。

用持续到达的请求测 TTFT、TPOT、QPS、错误率和 streaming 平滑度。

练习六：多轮对话 prefix reuse。

用 shell 或 chat API 构造多轮对话，观察后续轮次是否命中历史 prefix。

练习七：最小 structured decoding。

实现一个简单 digit-only 或 JSON-like token mask，接入 sampler。

练习八：tool call parser。

让模型输出一个简化 tool call JSON，并实现 streaming parser。

练习九：KV cache 泄漏测试。

构造 abort、timeout、client disconnect，检查 KV blocks 和 radix cache pin 是否释放。

练习十：mini-sglang 到完整 SGLang 对照报告。

列出 mini-sglang 简化了哪些生产能力，以及完整 SGLang 如何补齐。

## 46.19 适合写进简历的项目描述

如果你基于 mini-sglang 做过改造，可以这样写：

```text
基于 mini-sglang 阅读和改造 SGLang-style LLM serving runtime，重点分析 scheduler、KV cache、Radix Cache、chunked prefill、overlap scheduling 和 online serving；实现 prefix hit ratio、节省 prefill token 数、TTFT/TPOT p99、KV block 使用量等指标，并通过共享前缀请求、多轮对话和长 prompt 压测验证 Radix Cache 与 chunked prefill 对延迟和吞吐的影响。
```

如果更偏 structured generation / agent：

```text
在 mini-sglang 基础上实现简化 structured decoding 和 tool call parser，将 grammar/token mask 接入 sampler，并支持流式解析 tool call arguments；结合 Radix Cache 观察多轮 agent prompt 的 prefix reuse，分析结构化输出、工具调用和 prefix sharing 对 serving runtime 的要求。
```

这比单纯写“读过 SGLang 源码”更有说服力。

因为它说明你理解：

1. SGLang 的特色在哪里。
2. 你做过可观测指标。
3. 你做过 ablation。
4. 你理解 prefix、scheduler、KV、sampler 之间的关系。

## 46.20 常见误区

误区一：把 mini-sglang 当成另一个 vLLM clone。

它确实有通用 serving engine 能力，但学习重点应该放在 SGLang runtime 特色上。

误区二：只看 benchmark，不看请求生命周期。

性能数字没有请求状态、cache 命中、调度策略支撑，很难解释。

误区三：把 Radix Cache 理解成普通 hash cache。

Radix Cache 的关键是部分前缀匹配、tree split、节点复用和 eviction。

误区四：忽略 overlap scheduling。

很多 serving 性能问题不只在 GPU kernel，也在 CPU 调度和 metadata 准备是否阻塞 GPU。

误区五：把 tool execution 放进 engine 核心。

推理 runtime 可以解析和生成 tool call，但真实工具执行通常属于应用层或 agent runtime。

误区六：忘记和完整 SGLang 对照。

mini-sglang 是学习桥梁，不是终点。

## 46.21 面试官会怎么问

问题一：mini-sglang 相比 nano-vLLM 和 tiny-llm，最值得学什么？

回答要点：tiny-llm 适合理解底层算子和 serving 优化动机，nano-vLLM 适合理解 vLLM-like engine 骨架，mini-sglang 适合理解 SGLang-style runtime，重点是 Radix Cache、prefix sharing、chunked prefill、overlap scheduling、online serving、structured generation 和 tool/agent serving。

问题二：Radix Cache 和普通 prefix cache 有什么不同？

回答要点：普通 prefix cache 更像完整 prefix hash 命中；Radix Cache 用前缀树支持最长前缀匹配、部分共享、节点 split/insert/evict。它适合 system prompt、tool schema、few-shot、多轮对话等大量共享前缀场景。

问题三：Overlap scheduling 解决什么问题？

回答要点：它让 CPU 侧调度和 batch metadata 准备与 GPU forward 重叠，避免 GPU 等 CPU。GPU 执行 batch N 时，CPU 可以准备 batch N+1，从而提升吞吐并降低调度开销对 TPOT 的影响。

问题四：mini-sglang 为什么适合学习在线 serving？

回答要点：它支持 OpenAI-compatible server、interactive shell、offline/online benchmark，可以观察持续请求到达、streaming、取消、错误、排队、TTFT/TPOT、prefix cache 命中和 scheduler 行为，比只跑 offline generate 更接近真实服务。

问题五：mini-sglang 和完整 SGLang 的关系是什么？

回答要点：mini-sglang 是紧凑教学实现，保留 runtime、scheduler、KV cache、Radix Cache、model runner、sampler、online serving 等骨架；完整 SGLang 有更多模型、后端、分布式、grammar、speculative decoding、tool parser、metrics、故障恢复和生产工程能力。

## 46.22 标准回答模板

如果面试官问“你怎么通过 mini-sglang 学 SGLang”，可以这样回答：

```text
我会把 mini-sglang 放在 tiny-llm 和 nano-vLLM 之后读。tiny-llm 帮我理解 attention、RoPE、GQA、KV cache、FlashAttention、continuous batching、chunked prefill 这些能力为什么出现；nano-vLLM 帮我理解一个 vLLM-like engine 的 sequence、scheduler、block manager、model runner 骨架；mini-sglang 则用来理解 SGLang-style runtime 的特色。

读 mini-sglang 时，我不会只看它有没有普通 scheduler 和 KV cache，而会重点看 Radix Cache、chunked prefill、overlap scheduling、online serving、structured generation 和 tool/agent serving。Radix Cache 用前缀树支持最长前缀匹配和部分共享，适合 system prompt、tool schema、few-shot、多轮对话这类共享前缀场景；chunked prefill 处理长 prompt，避免 prefill 长时间阻塞 decode；overlap scheduling 让 CPU 侧调度和 metadata 准备与 GPU forward 重叠，减少 GPU 空等。

我会做几个实验验证理解：构造共享 system prompt 的请求观察 radix tree split 和 prefix hit length；关闭 overlap scheduling 做 ablation；对比不同 chunk size 下 TTFT、TPOT p99 和吞吐；用 online benchmark 观察持续请求到达下的 streaming 平滑度、错误率和 KV 使用量。

mini-sglang 不是完整生产 SGLang，它会简化更多模型、后端、分布式、grammar、speculative decoding、metrics 和故障恢复能力。但它能建立 SGLang Runtime 的骨架，读完后再去看完整 SGLang，就可以把复杂模块挂回 request state、scheduler、Radix Cache、model runner、sampler 和 online serving 这条主线。
```

## 46.23 小练习

1. 对比 tiny-llm、nano-vLLM、mini-sglang 三者的学习重点。
2. 画出 mini-sglang 中一个 online request 的生命周期。
3. 构造三个共享前缀请求，画出 Radix Cache 的 tree split 和 insert。
4. 统计 prefix hit length 和节省的 prefill token 数。
5. 解释 Radix Cache 和 PagedAttention 的区别。
6. 对比 chunked prefill 开关或不同 chunk size 下的 TTFT、TPOT 和吞吐。
7. 关闭 overlap scheduling，观察吞吐和 GPU utilization 的变化。
8. 用 interactive shell 分析多轮对话中的 prefix reuse。
9. 设计一个最小 structured decoding token mask，并说明如何接入 sampler。
10. 设计一个 tool call streaming parser。
11. 构造 abort 请求，检查 KV cache 和 Radix Cache pin 是否释放。
12. 写一页 mini-sglang 到完整 SGLang 的差距分析。

## 46.24 本章总结

mini-sglang 是学习 SGLang-style runtime 的关键过渡项目。

它和 tiny-llm、nano-vLLM 的分工不同：tiny-llm 帮你理解底层计算和 serving 优化动机，nano-vLLM 帮你理解 vLLM-like engine 骨架，mini-sglang 帮你理解复杂 LLM program runtime、prefix sharing、Radix Cache、overlap scheduling、online serving、structured generation 和 tool/agent serving。

读 mini-sglang 时，重点不是重复学习普通 scheduler 或 KV cache，而是看 SGLang 的特色如何落到工程实现：共享前缀如何命中和复用，长 prompt 如何 chunked prefill，CPU 调度如何与 GPU forward 重叠，在线请求如何进入 runtime，结构化输出和工具调用如何影响 sampler 和 streaming。

读完这三个教学项目后，你应该能把“模型计算、engine 调度、runtime 语义”三层打通。

下一章会进入第 47 章：如何从这些教学项目中抽象出推理框架的核心模块，形成自己的 mini serving engine 设计图。
