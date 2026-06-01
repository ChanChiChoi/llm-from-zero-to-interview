# 第 45 章 tiny-llm 学习路线

上一章讲了 nano-vLLM 学习路线。nano-vLLM 更像一个浓缩版 vLLM-like serving engine：重点是 `Sequence`、`Scheduler`、`BlockManager`、`ModelRunner`、attention 和 sampler 如何组成一个推理引擎。

这一章看另一个教学项目：`tiny-llm`。

`tiny-llm` 的定位有所不同。它是一个面向系统工程师的 LLM serving 课程项目，目标是在 Apple Silicon 和 MLX 环境中，从底层数组和矩阵 API 开始，一步步实现 Qwen 类模型推理和 serving 系统。

它的路线大致是：

1. 第 1 周：attention、RoPE、GQA、RMSNorm、MLP、模型加载、generate、sampling。
2. 第 2 周：KV cache、量化 matmul、FlashAttention、continuous batching、chunked prefill。
3. 第 3 周：paged attention、MoE、speculative decoding、RAG、Agent、long context 等高级主题。

一句话概括：

> 学 tiny-llm 的重点是从“模型算子和单请求 generate”一路走到“KV cache、batching、chunked prefill、paged attention 这些 serving 优化”，理解推理系统不是凭空出现的，而是在模型 forward 的每个瓶颈上逐步长出来的。

## 45.1 本章目标

读完本章，你应该能讲清：

1. tiny-llm 和 nano-vLLM 的学习重点有什么不同。
2. 为什么 tiny-llm 适合系统工程师学习 LLM inference serving。
3. 应该按什么顺序读 tiny-llm 的课程和源码。
4. 如何从 attention、RoPE、GQA、RMSNorm、MLP 过渡到完整模型推理。
5. 如何从单请求 generate 过渡到 KV cache 和 continuous batching。
6. 如何理解 FlashAttention、chunked prefill、paged attention 在 serving 中的位置。
7. 如何基于 tiny-llm 设计改造练习和面试项目。

## 45.2 tiny-llm 和 nano-vLLM 的区别

两个项目都适合学习推理框架，但侧重点不同。

| 项目 | 更适合学习 | 阅读主线 |
|---|---|---|
| nano-vLLM | vLLM-like engine 骨架 | API、engine、scheduler、sequence、block manager、model runner |
| tiny-llm | 从模型实现到 serving 优化 | attention、RoPE、GQA、模型加载、generate、KV cache、batching、paged attention |

nano-vLLM 更像：

```text
给你一个已经成型的轻量 serving engine，学习它的模块边界。
```

tiny-llm 更像：

```text
从 attention 和模型 forward 开始，逐步把它改造成 serving engine。
```

所以两者互补。

推荐学习顺序是：

1. 先用 tiny-llm 理解模型推理和底层算子。
2. 再用 nano-vLLM 理解 vLLM-like engine 的整体骨架。
3. 最后用 mini-sglang 理解 SGLang 的 runtime、prefix sharing 和 structured generation。

如果你已经读过 nano-vLLM，再读 tiny-llm，就要反过来问：

```text
nano-vLLM 里的 ModelRunner、Attention、BlockManager、Scheduler，分别是从 tiny-llm 哪些基础步骤演进来的？
```

## 45.3 tiny-llm 的课程式结构

tiny-llm 的一个优点是路线非常课程化。

它不是一上来给你一个完整框架，而是分周推进。

第 1 周主要回答：

```text
如何不用高级神经网络封装，从基础矩阵 API 实现一个能生成文本的 Qwen-like decoder-only 模型？
```

包括：

1. Attention。
2. RoPE positional encoding。
3. Grouped Query Attention。
4. RMSNorm 和 MLP。
5. 模型权重加载。
6. Generate response。
7. Sampling。

第 2 周主要回答：

```text
如何把能跑的单请求模型，变成更高效的 inference serving 系统？
```

包括：

1. KV cache。
2. Quantized matmul and linear。
3. FlashAttention。
4. Continuous batching。
5. Chunked prefill。

第 3 周主要回答：

```text
如何继续向生产级推理框架靠近？
```

包括：

1. Paged attention。
2. MoE。
3. Speculative decoding。
4. RAG pipeline。
5. Agent / tool calling。
6. Long context。

这条路线和本书前面章节高度对应。

## 45.4 推荐阅读顺序

tiny-llm 最好按课程顺序读，而不是按目录随便跳。

推荐顺序是：

1. 先读 README 和 book overview，确认课程目标和环境。
2. Week 1 overview：建立“从算子到模型”的全局图。
3. Attention：理解最小 attention 计算。
4. RoPE：理解 position 如何进入 Q/K。
5. GQA：理解 query heads 和 KV heads 为什么可以不同。
6. RMSNorm and MLP：补齐 decoder layer。
7. Qwen3 model：把模块组装成完整模型。
8. Generate response：实现自回归生成。
9. Sampling：从 logits 到 token。
10. Week 2 overview：切换到 serving 优化视角。
11. KV cache：理解为什么 decode 不能每步重算全上下文。
12. Quantized matmul：理解权重量化和推理算子。
13. FlashAttention：理解 memory-efficient attention。
14. Continuous batching：理解多请求调度。
15. Chunked prefill：理解长 prompt 和 decode 抖动。
16. Paged attention：理解 block-based KV 管理。

这个顺序有一个清晰主线：

```text
模型能算 -> 模型能生成 -> 生成能复用 KV -> 多请求能 batch -> 长 prompt 能切 chunk -> KV 能分页管理
```

这也是一个 naive inference 程序演进成 serving engine 的过程。

## 45.5 第一阶段：从 Attention 开始

tiny-llm 的第一站是 attention。

不要觉得 attention 公式已经学过就跳过。

在 serving 语境下，attention 需要关注的不只是：

```text
softmax(QK^T / sqrt(d))V
```

还要关注：

1. Q/K/V 的 shape。
2. batch 维度如何组织。
3. causal mask 如何生效。
4. prefill 和 decode 的 attention shape 有什么不同。
5. 中间 attention score 是否占用大量内存。
6. 后续 KV cache 会缓存 K/V 的哪一部分。

建议你在读 attention 时写下两个 shape。

Prefill：

```text
Q: [batch, prompt_len, num_heads, head_dim]
K: [batch, prompt_len, num_kv_heads, head_dim]
V: [batch, prompt_len, num_kv_heads, head_dim]
```

Decode：

```text
Q: [batch, 1, num_heads, head_dim]
K/V cache: [batch, past_len + 1, num_kv_heads, head_dim]
```

这个差异是后面 KV cache、FlashAttention、continuous batching 和 PD 分离的基础。

## 45.6 第二阶段：RoPE 和位置

RoPE 是很多 decoder-only LLM 的关键组件。

读 tiny-llm 的 RoPE 时，重点不是背公式，而是理解：

```text
为什么 decode 时 position 必须和 KV cache 长度严格对齐？
```

在自回归生成中，第 `t` 个 token 的 position 不能错。

如果 position 错了，表面上模型仍然能 forward，但输出质量会明显异常。

Serving 系统里 position 相关 bug 很常见：

1. prompt length 计算错。
2. padding 后 position 没处理好。
3. chunked prefill 后 position 没连续。
4. KV cache 复用后 position 偏移错。
5. sliding window 或 long context 扩展后 position 策略不一致。

所以读 RoPE 时，要把它和后面的 KV cache 连接起来：

```text
KV cache 记录历史 token 的 K/V，RoPE 决定这些 K/V 对应的位置信息。
```

## 45.7 第三阶段：Grouped Query Attention

GQA 是现代 LLM serving 非常重要的结构。

它的核心是：

```text
num_query_heads > num_kv_heads
```

也就是说，多个 query heads 共享更少的 K/V heads。

这对 serving 很重要，因为 KV cache 大小和 `num_kv_heads` 成正比。

KV bytes 近似为：

```text
KV bytes ~= 2 * num_layers * seq_len * num_kv_heads * head_dim * bytes_per_elem
```

所以 GQA 可以显著降低 KV cache 内存。

读 tiny-llm 的 GQA 时，重点看：

1. Q heads 和 KV heads 的 shape 差异。
2. K/V 如何 repeat 或 broadcast 给 Q heads。
3. GQA 对 attention 计算的影响。
4. GQA 对 KV cache 容量的影响。

这也是面试常问点：

```text
为什么 GQA/MQA 对推理更友好？
```

答案不是“参数少”这么简单，而是 KV cache 更小，decode 阶段读 KV 的内存压力更低。

## 45.8 第四阶段：RMSNorm 和 MLP

RMSNorm 和 MLP 看起来更偏模型结构，但也影响推理性能。

读这一部分时，重点看：

1. RMSNorm 是否在 attention 和 MLP 前使用。
2. Norm 是否是逐 token 计算。
3. MLP 是普通 FFN 还是 gated MLP。
4. 中间维度 expansion 有多大。
5. 激活函数是什么。
6. 权重 shape 如何加载。

Serving 中，MLP 往往贡献大量矩阵乘计算。

Attention 在长上下文时很显眼，但每层 MLP 的 GEMM 也非常重要。

后面读量化 matmul 时，要记住：

```text
量化优化的主要对象之一就是 linear/MLP 权重矩阵。
```

## 45.9 第五阶段：组装 Qwen-like 模型

读到 Qwen3 model 时，要从组件视角切换到完整 forward 视角。

一个 decoder-only 模型大致是：

```text
token ids
  -> token embedding
  -> N x decoder layer
       -> RMSNorm
       -> attention with RoPE/GQA/KV
       -> residual
       -> RMSNorm
       -> MLP
       -> residual
  -> final norm
  -> lm head
  -> logits
```

读模型组装代码时，要回答：

1. config 如何定义层数、hidden size、head 数、kv head 数。
2. 权重文件如何映射到模块。
3. 每层 forward 的输入输出 shape 是什么。
4. attention 如何接收 position 和 KV cache。
5. logits 只取最后一个 token 还是所有 token。

推理服务通常只关心最后位置的 logits：

```text
prefill: 需要 prompt 最后一个位置的 logits 来采样首 token
decode: 每步输入一个 token，输出这个位置的 logits
```

理解这一点，才能理解为什么 prefill 和 decode 可以分阶段优化。

## 45.10 第六阶段：Generate Response

有了模型 forward，还不等于有推理服务。

下一步是自回归生成：

```text
input prompt
  -> model forward
  -> logits
  -> sample next token
  -> append token
  -> repeat until eos or max_tokens
```

这就是最小 generate loop。

读 tiny-llm 的 generate 时，要重点看：

1. prompt 如何 tokenize。
2. 第一次 forward 输入的是完整 prompt 还是最后 token。
3. 新 token 如何 append。
4. eos 如何判断。
5. max tokens 如何限制。
6. 是否每一步都重算完整上下文。

如果没有 KV cache，naive generate 的复杂度会很差：

```text
step 1: compute prompt_len
step 2: compute prompt_len + 1
step 3: compute prompt_len + 2
...
```

这就是第 2 周 KV cache 出现的动机。

## 45.11 第七阶段：Sampling

Sampling 把 logits 变成 next token。

读 tiny-llm 的 sampling 时，重点看：

1. greedy 如何实现。
2. temperature 如何缩放 logits。
3. top-k 如何截断候选集合。
4. top-p 如何按累计概率截断。
5. 随机数如何采样。
6. 采样结果如何写回 generate loop。

在 serving 系统里，sampling 还有一个重要工程问题：

```text
同一个 batch 里的不同请求，可能有不同 sampling params。
```

例如：

1. 请求 A 使用 greedy。
2. 请求 B 使用 temperature 0.8。
3. 请求 C 使用 top-p 0.9。

所以生产级 sampler 通常要支持 per-request sampling params。

如果 tiny-llm 的实现更简单，也可以把它作为改造练习。

## 45.12 第八阶段：KV Cache

KV cache 是从“模型推理”进入“推理系统”的分水岭。

没有 KV cache，decode 每一步都要重算历史 token。

有了 KV cache，流程变成：

```text
prefill:
  input full prompt
  compute K/V for prompt
  store K/V in cache
  output next-token logits

decode:
  input only last token
  compute K/V for new token
  append to cache
  attend over cached K/V
  output next-token logits
```

读 tiny-llm 的 KV cache 时，要重点看：

1. cache 的 shape。
2. 每层是否有独立 cache。
3. K 和 V 是否分开存。
4. prefill 如何写入 cache。
5. decode 如何 append cache。
6. current position 如何维护。
7. cache 最大长度如何限制。

这部分要和第 9 章、第 18 章、第 21 章联系起来。

KV cache 不是一个模型小优化，而是 serving engine 的核心资源。

## 45.13 第九阶段：量化 Matmul

tiny-llm 包含 quantized matmul and linear，这是系统工程师很值得看的部分。

LLM 推理里，大量时间花在矩阵乘：

1. Q/K/V projection。
2. Attention output projection。
3. MLP up/gate/down projection。
4. LM head。

量化的目标是降低：

1. 权重显存。
2. 内存带宽压力。
3. 部分计算成本。

读量化 matmul 时，重点看：

1. 权重如何存储。
2. scale 和 zero point 如何表示。
3. 反量化是在 matmul 前做，还是融合进 matmul。
4. CPU 和 GPU 实现差异。
5. 精度和速度如何测试。

要记住一个工程判断：

```text
量化不是只看模型文件变小，还要看端到端 TTFT、TPOT、吞吐和质量是否可接受。
```

## 45.14 第十阶段：FlashAttention

FlashAttention 的核心价值是减少 attention 中间矩阵的内存读写。

普通 attention 可能显式形成：

```text
attention scores: [batch, heads, query_len, key_len]
```

长上下文下，这个矩阵非常大。

FlashAttention 通过分块和在线 softmax，避免把完整 attention score materialize 到显存。

读 tiny-llm 的 FlashAttention 时，重点看：

1. 为什么要分 block。
2. softmax 的 max 和 sum 如何在线维护。
3. causal mask 如何处理。
4. CPU 和 GPU 实现差异。
5. prefill 和 decode 哪个更受益。

和 serving 的关系是：

```text
prefill 阶段 query_len 大，FlashAttention 通常更关键；decode 阶段 query_len 通常为 1，瓶颈更多在读 KV 和小 batch 效率。
```

## 45.15 第十一阶段：Continuous Batching

Continuous batching 是从单请求 generate 到 serving engine 的关键一步。

Naive batching 是：

```text
一批请求一起开始，一起生成到结束。
```

问题是不同请求输出长度不同，短请求会等待长请求，或者 batch 很快变空。

Continuous batching 是：

```text
每一轮 decode 都重新组织 active requests，新请求可以在合适时机加入，完成请求可以立刻退出。
```

读 tiny-llm 的 continuous batching 时，重点看：

1. 请求状态如何表示。
2. waiting 和 running 如何维护。
3. 每轮 batch 如何构造。
4. prefill 和 decode 是否混合。
5. 新请求何时加入。
6. 完成请求何时移除。
7. KV cache 如何跟随请求状态变化。

这部分可以和第 19 章、第 20 章、第 44 章 nano-vLLM 对照。

## 45.16 第十二阶段：Chunked Prefill

Chunked prefill 是处理长 prompt 和 decode 抖动的重要技术。

长 prompt prefill 会占用较长计算时间。

如果它和 decode 混在同一个 engine 中，可能导致：

```text
decode token 间隔变大，streaming 卡顿，TPOT p99 变差。
```

Chunked prefill 的思路是：

```text
不要一次处理完整长 prompt，而是拆成多个 chunk，让 scheduler 可以在 chunk 之间插入 decode。
```

读 tiny-llm 的 chunked prefill 时，重点看：

1. prompt 如何切 chunk。
2. chunk 之间 position 如何连续。
3. KV cache 如何逐 chunk append。
4. scheduler 如何在 prefill chunk 和 decode 之间取舍。
5. chunk size 如何影响 TTFT、TPOT 和吞吐。

这部分可以直接连接第 39 章。

也是后续 PD 分离之前最值得尝试的优化之一。

## 45.17 第十三阶段：Paged Attention

Paged attention 是 vLLM 的核心思想之一。

tiny-llm 的第 3 周进入 paged attention，非常适合作为从 fixed KV cache 到 block-based KV cache 的过渡。

普通 KV cache 通常按 request 分配连续空间：

```text
request A: [0 ... max_len]
request B: [0 ... max_len]
```

这会浪费显存，因为很多请求不会生成到 max_len。

Paged attention 把 KV cache 拆成固定大小 block：

```text
logical tokens -> block table -> physical KV blocks
```

读 paged attention 时，重点看：

1. block size 如何选择。
2. logical token position 如何映射到 block id 和 offset。
3. block table 如何传给 attention。
4. decode 追加 token 时何时分配新 block。
5. 请求结束时如何释放 block。
6. attention kernel 如何按 block table 读取不连续 KV。

这部分和第 17、18、21 章完全对应。

## 45.18 tiny-llm 和本书章节映射

可以把 tiny-llm 当成本书前半部分的代码练习索引。

| tiny-llm 主题 | 对应本书章节 | 学习重点 |
|---|---|---|
| Attention | 第 4、9、17 章 | Q/K/V、causal mask、KV cache 入口 |
| RoPE | 第 4、9 章 | position 和 KV cache 对齐 |
| GQA | 第 4、15、21 章 | KV cache 显存和 decode 读带宽 |
| RMSNorm/MLP | 第 1、15 章 | 模型层计算成本 |
| Generate | 第 7 章 | 最小自回归循环 |
| Sampling | 第 8 章 | logits 到 token |
| KV Cache | 第 9、18、21 章 | prefill/decode 分离基础 |
| Quantized Matmul | 第 15、25 章 | 显存、带宽、精度取舍 |
| FlashAttention | 第 15、17 章 | attention IO 优化 |
| Continuous Batching | 第 19、20 章 | iteration-level scheduling |
| Chunked Prefill | 第 39 章 | 长 prompt 和 TPOT 抖动 |
| Paged Attention | 第 17、18、21 章 | block-based KV 管理 |

这样读 tiny-llm 时，每个课程章节都能对应到本书的系统概念。

## 45.19 适合做的改造练习

tiny-llm 很适合做从底层到系统的改造练习。

练习一：画 shape trace。

对 attention、RoPE、GQA、MLP、lm head 画出每一步 tensor shape。

练习二：实现并对比 naive generate 和 KV cache generate。

记录每生成一个 token 的耗时，观察上下文变长后的差异。

练习三：给 generate loop 增加 TTFT 和 TPOT 统计。

区分 prompt prefill 时间和 decode token 时间。

练习四：实现 per-request sampling params。

让同一批请求支持不同 temperature、top-k、top-p。

练习五：对比不同量化 matmul 的速度和误差。

记录输出误差、perplexity 近似指标或生成质量差异。

练习六：给 continuous batching 增加 scheduler 日志。

每轮记录 waiting、running、batch size、prefill token 数和 decode request 数。

练习七：实现 chunked prefill 的不同 chunk size。

比较 TTFT、TPOT p95/p99 和吞吐。

练习八：实现简单 paged KV block manager。

支持 allocate、append、free 和 block table 查询。

练习九：模拟 KV cache OOM。

观察 scheduler 如何拒绝新请求或等待 finished 请求释放 KV。

练习十：写一份从 tiny-llm 到 vLLM 的差距分析。

列出缺失的 API server、streaming、block manager、worker/executor、metrics、分布式和故障恢复能力。

## 45.20 适合写进简历的项目描述

如果你基于 tiny-llm 做过实战，可以这样写：

```text
基于 tiny-llm 从底层矩阵 API 实现 Qwen-like LLM 推理流程，完成 attention、RoPE、GQA、RMSNorm、MLP、模型加载、generate 和 sampling；进一步实现 KV cache、continuous batching 和 chunked prefill，并增加 TTFT/TPOT 统计，对比 naive generate、KV cache decode 和不同 chunk size 下的延迟与吞吐表现。
```

更偏系统优化可以写：

```text
在 tiny-llm 基础上实现推理 serving 优化实验，包括量化 matmul、FlashAttention、KV cache、continuous batching 和 paged KV block manager；通过 scheduler 日志和 KV 使用统计分析长 prompt、decode 抖动、显存利用率和 batch size 对 TTFT/TPOT 的影响。
```

面试时要强调：

1. 你不是只跑了 demo。
2. 你理解每个模型组件的 shape 和推理作用。
3. 你知道 KV cache 为什么改变复杂度。
4. 你做过 batching、chunked prefill 或 paged attention 相关实验。
5. 你能把 tiny-llm 的实现映射到 vLLM-like serving engine。

## 45.21 常见误区

误区一：只把 tiny-llm 当模型实现教程。

它确实从模型组件开始，但后半部分重点是 inference serving。

误区二：跳过 attention/RoPE/GQA，直接看 batching。

这样很容易不理解 KV cache 的 shape、position 和内存成本。

误区三：只看能不能生成文本。

生成文本只是第一步，更重要的是 TTFT、TPOT、吞吐、显存和 batch 行为。

误区四：把 MLX 细节当成唯一重点。

MLX 是实现环境，真正可迁移的是推理系统思想：KV cache、batching、attention 优化、block 管理。

误区五：认为 tiny-llm 可以替代生产框架。

它更适合学习和实验，不是完整生产 serving 系统。

## 45.22 面试官会怎么问

问题一：tiny-llm 适合学什么？

回答要点：适合从底层实现理解 LLM inference serving 的演进路径：先实现 attention、RoPE、GQA、RMSNorm、MLP、模型加载、generate 和 sampling，再加入 KV cache、量化 matmul、FlashAttention、continuous batching、chunked prefill 和 paged attention。

问题二：tiny-llm 和 nano-vLLM 有什么区别？

回答要点：nano-vLLM 更像一个已经成型的轻量 vLLM-like engine，重点看 engine、scheduler、sequence、block manager、model runner；tiny-llm 更课程化，从模型算子和单请求 generate 开始，逐步引入 KV cache、batching、chunked prefill、paged attention 等 serving 优化。

问题三：为什么 GQA 对推理重要？

回答要点：GQA 让多个 query heads 共享更少的 KV heads，降低 KV cache 大小和 decode 阶段读取 KV 的带宽压力。KV cache 大小与 `num_kv_heads` 成正比，所以 GQA 对长上下文和高并发 serving 很重要。

问题四：KV cache 如何改变 generate 的复杂度？

回答要点：没有 KV cache 时，每生成一个 token 都要重算完整上下文；有 KV cache 后，prefill 计算 prompt 并缓存 K/V，decode 每步只输入新 token，追加新 K/V，并 attend over cached K/V，从而避免重复计算历史 token。

问题五：chunked prefill 解决什么问题？

回答要点：长 prompt prefill 会阻塞 decode，导致 streaming TPOT 抖动。Chunked prefill 把长 prompt 拆成多个 chunk，让 scheduler 能在 chunk 之间插入 decode，改善 decode 尾延迟，但会引入 chunk size、position 连续性和 KV 逐步写入的复杂度。

## 45.23 标准回答模板

如果面试官问“你怎么通过 tiny-llm 学推理系统”，可以这样回答：

```text
我会把 tiny-llm 当成从模型 forward 演进到 serving engine 的课程来读。第一阶段先实现 attention、RoPE、GQA、RMSNorm 和 MLP，重点不是背公式，而是把 Q/K/V shape、position、num_heads 和 num_kv_heads 搞清楚，因为这些直接决定后面 KV cache 的 shape 和显存成本。

第二阶段把这些组件组装成 Qwen-like decoder-only 模型，完成模型加载、generate loop 和 sampling。这里我会重点区分 prefill 和 decode：prefill 输入完整 prompt，decode 每次生成一个 token。没有 KV cache 时，每步都重算历史上下文，复杂度很差。

第三阶段进入 serving 优化，引入 KV cache，让 prefill 写入 prompt K/V，decode 每步只计算新 token 的 K/V 并追加到 cache。然后继续看量化 matmul、FlashAttention、continuous batching、chunked prefill 和 paged attention。量化主要降低权重显存和带宽，FlashAttention 优化长 prompt attention 的 IO，continuous batching 让多个请求按 iteration 动态组成 batch，chunked prefill 缓解长 prompt 阻塞 decode，paged attention 则用 block table 管理不连续 KV，降低显存浪费。

相比 nano-vLLM，tiny-llm 更适合理解这些能力是怎么从底层一步步长出来的；nano-vLLM 更适合看一个 vLLM-like engine 的模块边界。两个结合起来，一个解释底层计算和优化动机，一个解释 engine 架构。

如果做项目，我会基于 tiny-llm 增加 TTFT/TPOT 统计、scheduler 日志、KV cache 使用统计，比较 naive generate、KV cache decode、continuous batching 和不同 chunked prefill size 下的延迟、吞吐和显存变化。
```

## 45.24 小练习

1. 画出 tiny-llm 中 attention、RoPE、GQA 的 tensor shape。
2. 推导 GQA 对 KV cache 大小的影响。
3. 对比 naive generate 和 KV cache generate 的计算流程。
4. 给 generate loop 增加 TTFT 和 TPOT 统计。
5. 解释 prefill 和 decode 在输入 shape、计算量和 KV 写入上的差异。
6. 实现 per-request sampling params，并说明 batch sampler 如何处理不同参数。
7. 对比量化 matmul 前后的显存、速度和生成质量。
8. 解释 FlashAttention 为什么更适合长 prefill。
9. 设计一个 continuous batching 的状态机。
10. 实验不同 chunk size 对 TTFT、TPOT 和吞吐的影响。
11. 设计一个最小 paged KV block manager。
12. 写一个面试回答：tiny-llm 和 nano-vLLM 应该如何结合学习？

## 45.25 本章总结

tiny-llm 是一个从底层模型实现走向 LLM serving 的课程型项目。

它的价值在于把推理系统拆成一条清晰演进线：先实现 attention、RoPE、GQA、RMSNorm、MLP、模型加载、generate 和 sampling，再引入 KV cache、量化 matmul、FlashAttention、continuous batching、chunked prefill 和 paged attention。

读 tiny-llm 时，不要只关注“能不能生成文本”，而要关注每一步为什么会成为 serving 优化的前置条件：GQA 影响 KV cache 大小，RoPE 影响 position 对齐，KV cache 区分 prefill/decode，continuous batching 改变请求调度，chunked prefill 缓解长 prompt 阻塞，paged attention 解决 KV 显存管理。

nano-vLLM 和 tiny-llm 应该结合起来学：tiny-llm 帮你理解底层计算和优化动机，nano-vLLM 帮你理解 vLLM-like engine 的模块边界。

下一章会进入 `mini-sglang` 学习路线，重点会从 vLLM-like engine 转向 SGLang 的 runtime、prefix sharing、structured generation 和 agent/tool serving 支持。
