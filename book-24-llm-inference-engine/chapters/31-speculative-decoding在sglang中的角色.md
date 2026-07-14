# 第 31 章 Speculative Decoding 在 SGLang 中的角色

上一章讲了 Structured Generation 与 constrained decoding：runtime 可以在每一步采样前用 grammar mask 限制合法 token，让输出满足 JSON schema、regex、EBNF 或 structural tag。

本章看另一个直接影响 decode 性能的技术：Speculative Decoding。

前面的章节反复讲过，LLM 自回归生成的 decode 阶段通常每轮只生成一个 token。即使 prefill 很快，长答案仍然要经过很多轮主模型 forward。Speculative decoding 的核心目标就是减少主模型逐 token decode 的等待，把“一个 token 一轮”尽量变成“一轮验证多个候选 token”。

一句话概括：

> Speculative decoding 用较便宜的 draft 来源先猜多个候选 token，再用 target model 并行验证这些候选；如果接受率足够高，就能用较少的 target model decode 轮数生成更多 token，从而改善 TPOT 和输出吞吐。

## 31.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，主要参考四类公开资料：

1. SGLang 官方 Speculative Decoding 文档，对 EAGLE、EAGLE3、MTP、DFLASH、Standalone draft model、NGRAM、Speculative Decoding V2 / overlap scheduler 等入口和参数给出了公开说明。
2. SGLang 官方 Adaptive Speculative Decoding 文档，对用近期 accept length 动态调整 speculative steps 的口径给出了说明。
3. SGLang server arguments / sampling / structured output 相关文档，对 speculative 参数、grammar 约束、streaming、metrics 和 runtime 组合边界提供工程背景。
4. Speculative Decoding / Speculative Sampling 论文，对 draft propose、target verify、接受-拒绝采样、修正分布和“保持 target 分布”的理论目标提供基础口径；EAGLE、EAGLE3、MTP 等论文用于理解不同 draft 来源的差异。

本章的写作边界也要说清：

1. 本章讲 SGLang-like serving runtime 中 speculative decoding 的角色，不绑定某个 SGLang 版本的内部类名、源码文件、CUDA kernel、attention backend 细节或 benchmark 数值。
2. 本章把 EAGLE / EAGLE3 / MTP / Standalone / NGRAM 作为 draft 来源和工程路径来解释，不展开它们完整训练算法。
3. 本章 demo 是 greedy 教学版，用来证明 draft / verify / accept / fallback / metrics / KV cleanup 的执行链路；严格保持 target sampling 分布需要论文中的 modified rejection sampling，不能把教学 demo 当成完整论文算法。
4. 本章对 structured output 的讨论只覆盖 grammar state 如何影响 draft 候选和 accept 过程，不实现真实 XGrammar / Outlines / llguidance。
5. 本章所有性能公式都是容量规划和面试解释用的简化模型，真实收益必须用业务 workload replay、GPU profiler、SGLang metrics 和灰度实验验证。

## 31.1 本章目标

读完本章，你应该能讲清：

1. Speculative decoding 主要解决 decode 阶段什么问题。
2. Draft model、target model、verify、accept length 分别是什么。
3. 为什么 speculative decoding 通常优化 TPOT，而不是 prefill。
4. SGLang 中 EAGLE、EAGLE3、MTP、Standalone、NGRAM 等方案的大致定位。
5. 接受率、draft 成本、verify 成本如何共同决定收益。
6. Speculative decoding 和 scheduler、KV cache、structured output 的交互。
7. 面试中如何回答“speculative decoding 为什么能加速”。

## 31.2 为什么 decode 阶段慢

自回归生成的基本流程是：

```text
prefill prompt
  -> decode token 1
  -> decode token 2
  -> decode token 3
  -> ...
```

每个 decode step 都需要主模型做一次 forward。

如果生成 1000 个 token，朴素做法至少需要 1000 轮 decode forward。

Decode 的问题是：

1. 每轮只推进很少 token。
2. 轮数多。
3. 每轮都要读历史 KV cache。
4. 调度、采样、streaming 都要重复发生。
5. 用户会直接感受到 TPOT。

Prefill 可以通过 batching、prefix cache、RadixAttention、chunked prefill 等优化减少重复计算。

Decode 则更难，因为下一个 token 依赖上一个 token，天然串行。

Speculative decoding 要打破的是这个串行瓶颈的一部分。

## 31.3 核心思想：先猜，再验

Speculative decoding 的核心流程是：

```text
1. draft source 快速生成多个候选 token
2. target model 一次性验证这些候选 token
3. 接受其中与 target model 一致或可接受的前缀
4. 如果某个 token 被拒绝，从 target model 分布采样修正 token
5. 重复下一轮
```

用更直观的例子：

```text
当前上下文: The capital of France is

draft 猜: Paris . It is
target 验证: Paris . 被接受，It 被拒绝

本轮实际前进: Paris . + target 修正 token
```

如果 draft 猜得准，一轮 target verify 可以接受多个 token。

原来：

```text
target forward 1 -> token A
target forward 2 -> token B
target forward 3 -> token C
```

现在：

```text
draft guesses A B C
target verify once -> accepts A B C
```

主模型轮数减少，TPOT 就可能下降。

## 31.4 Draft 和 target

Speculative decoding 中通常有两个角色。

Target model：

1. 真正要服务的主模型。
2. 输出质量以它为准。
3. 参数量大、forward 贵。
4. 负责验证 draft token。

Draft source：

1. 用来提出候选 token。
2. 必须比 target 更便宜或更快。
3. 可以是小模型、特殊 draft head、MTP head、EAGLE draft model、ngram cache 等。
4. 猜得越准，接受率越高。

最经典版本是 small draft model + large target model：

```text
small draft model: 快速猜 4 个 token
large target model: 一次 verify 4 个 token
```

但现代 serving engine 中，draft source 不一定是普通小模型，也可能是 EAGLE、MTP、NGRAM 等不同机制。

## 31.5 Verify 和 accept length

Verify 是 target model 对 draft tokens 的验证过程。

假设 draft 提出：

```text
[d1, d2, d3, d4]
```

Target model 会计算这些位置上的分布，并判断 draft token 是否可以接受。

接受前缀长度叫 accept length：

```text
draft:  [d1, d2, d3, d4]
accept: [d1, d2]
reject:         d3

accept length = 2
```

如果 accept length 越大，本轮 target verify 的收益越高。

如果 accept length 经常是 0 或 1，speculative decoding 可能不但没收益，还会增加 draft 开销。

所以接受率是 speculative decoding 最重要的指标之一。

## 31.6 为什么它不改变输出分布

经典 speculative decoding 的一个重要目标是：加速的同时保持 target model 的输出分布不变或尽量等价。

直觉上，draft 只是提出候选，最终是否接受由 target model 决定。

如果 draft 猜错，target 会拒绝并从正确分布中采样修正 token。

因此 draft 不应该决定最终质量，target 才是最终裁判。

工程实现中，不同算法对“严格保持分布”和“近似加速”有不同取舍。

面试回答时可以说：

```text
Speculative decoding 的基本思想是 draft propose、target verify；在标准算法中，通过接受/拒绝规则可以保持 target 分布，工程变体可能在性能和严格性之间做取舍。
```

不要简单说“小模型直接生成，所以质量会变差”。更准确是：小模型只是 draft，target model 负责验证。

## 31.7 最小伪代码

一个简化 speculative decoding loop：

```python
while not finished:
    draft_tokens = draft_model.generate(context, k=4)

    target_logits = target_model.verify(context, draft_tokens)
    accepted, correction = accept_or_reject(target_logits, draft_tokens)

    output.extend(accepted)
    if correction is not None:
        output.append(correction)

    context.extend(accepted)
    if correction is not None:
        context.append(correction)
```

这个伪代码省略了很多工程细节：

1. KV cache 如何写入。
2. Draft model 的 KV cache 如何维护。
3. Target verify 的 attention metadata 如何构造。
4. 多请求 batch 如何一起 speculative。
5. 被拒绝 token 之后的 draft KV 如何丢弃或修正。
6. Streaming 如何返回多个 accepted tokens。
7. Structured output 如何约束候选 token。

但它已经表达了主干：draft 提案，target 验证，接受前缀。

## 31.8 主要优化什么指标

Speculative decoding 主要优化 decode 阶段。

相关指标包括：

1. TPOT。
2. output tokens/s。
3. decode step 有效产出 token 数。
4. accepted tokens/s。
5. target forward per output token。

它通常不直接优化 prefill。

原因是 prefill 本来就是一次处理 prompt 多 token，而 speculative decoding 解决的是 decode 每轮只出一个 token的问题。

所以面试中要避免说：

```text
Speculative decoding 加速 prompt prefill。
```

更准确：

```text
它主要减少主模型 decode 轮数，提高每轮 verify 产出的 token 数，因此改善 TPOT 和输出吞吐。
```

## 31.9 收益取决于什么

Speculative decoding 的收益不是必然的。

它取决于：

1. Draft source 是否足够快。
2. Draft token 接受率是否足够高。
3. Target verify 是否能高效并行验证多个 token。
4. 额外 KV cache 和 memory 开销是否可接受。
5. Batch size 和调度是否适配 speculative 路径。
6. 采样参数是否让 draft 更难预测。
7. 结构化约束是否降低或提高候选可接受性。
8. 硬件是否能吃满 verify 的并行度。

可以用一个直觉公式：

```text
收益 ≈ accepted tokens per target verify / extra draft overhead
```

如果 draft 很慢，或者 target 每次只接受 1 个 token，收益就会很差。

## 31.10 接受率为什么会变化

接受率不是固定常数。

它会随 workload 变化。

接受率较高的场景：

1. 确定性任务。
2. 低 temperature。
3. 格式固定输出。
4. 代码或模板化文本。
5. Draft model 和 target model 分布接近。
6. 上下文模式重复。

接受率较低的场景：

1. 高 temperature。
2. 创意写作。
3. 多样化开放问答。
4. Draft model 太弱。
5. 目标模型分布很尖锐但 draft 猜偏。
6. 强约束和 sampling 规则冲突。

这就是为什么 SGLang 文档中会有 adaptive speculative decoding：真实流量的 accept length 会随时间变化，固定 speculative steps 不一定总是最优。

## 31.11 SGLang 支持的 speculative decoding 类型

SGLang 文档中列出多种 speculative decoding 方案，包括：

1. EAGLE-2。
2. EAGLE-3。
3. MTP。
4. DFLASH。
5. Standalone draft model。
6. NGRAM。
7. Speculative Decoding V2 / overlap scheduler。
8. Adaptive speculative decoding。

学习时不需要一开始记住所有参数。

先把它们按 draft 来源分类：

| 方法 | Draft 来源 | 直觉 |
| --- | --- | --- |
| Standalone | 更小的 draft LLM | 小模型先猜，大模型验证 |
| EAGLE / EAGLE3 | 专门训练或适配的 draft 模块 | 更高效地产生候选 token/tree |
| MTP | 模型内置多 token prediction 能力 | 一个模型结构中直接预测后续多个 token |
| DFLASH | 专门 draft checkpoint | 线性 draft block 验证 |
| NGRAM | 之前 token 的 ngram 匹配 | 不用额外模型，从重复片段中猜 |

核心都是：

```text
产生候选 token -> target verify -> 接受一段 -> 减少 target decode 轮数
```

## 31.12 EAGLE / EAGLE3 的定位

EAGLE 是 SGLang 中推荐的 speculative decoding 路径之一。

它的直觉是：用更适合 speculative 的 draft 机制生成候选 token，target model 再验证。

文档中给出的启用方式通常包含：

```bash
python3 -m sglang.launch_server \
  --model <target-model> \
  --speculative-algorithm EAGLE \
  --speculative-draft-model-path <eagle-draft-model> \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16
```

EAGLE3 是更新的方案，文档中也把它作为高性能推荐选项之一。

对学习者来说，关键不是背具体命令，而是理解这些参数含义：

1. `speculative_algorithm`：使用哪种 speculative 方法。
2. `speculative_draft_model_path`：draft 模型或模块来源。
3. `speculative_num_steps`：draft 往前猜多少步。
4. `speculative_eagle_topk`：每步分支候选数。
5. `speculative_num_draft_tokens`：一次 verify 最多处理多少 draft tokens。

这些参数共同决定候选树大小、接受率、显存占用和计算开销。

## 31.13 Standalone draft model

Standalone 是最容易理解的 speculative decoding。

结构是：

```text
target model: Qwen2.5-7B
draft model:  Qwen2.5-1.5B
```

Draft model 更小，所以生成候选更便宜。

Target model 更大，所以负责最终验证。

优点：

1. 概念简单。
2. 容易解释。
3. 适合有同系列小模型的场景。

缺点：

1. 需要额外加载 draft model，占显存。
2. Draft 和 target tokenizer、模型族要匹配。
3. 接受率依赖两个模型分布接近程度。
4. 多模型调度更复杂。

如果 draft model 太弱，target 会频繁拒绝，收益会下降。

## 31.14 NGRAM speculative decoding

NGRAM 方法不依赖额外 draft model。

它从已有上下文或历史生成中找重复片段，作为 draft tokens。

直觉：

```text
如果模型刚生成过类似 ngram，后面可能再次出现类似片段。
```

适合场景：

1. 代码生成中的重复模式。
2. 长文本中重复短语。
3. 模板化输出。
4. 不想额外加载 draft model 的场景。

优点：

1. 不需要额外模型权重。
2. 显存压力可能低于 standalone draft model。
3. 对重复性强的输出有用。

缺点：

1. 依赖重复模式。
2. 对开放式新内容帮助有限。
3. 命中和接受率不稳定。
4. 适用硬件和配置可能有限。

## 31.15 MTP：Multi-Token Prediction

MTP 指模型自身具备预测多个未来 token 的能力。

如果模型结构或权重支持 MTP，就可以把多 token prediction 接入 speculative decoding 流程。

直觉是：

```text
普通模型 head: 预测下一个 token
MTP heads: 预测后面多个 token
```

然后 target 路径验证这些多 token 候选。

优点：

1. 不一定需要单独小模型。
2. Draft 和 target 更紧密。
3. 可能减少额外模型加载成本。

限制：

1. 需要模型本身支持。
2. 不同模型启用方式不同。
3. 参数调优仍然重要。

## 31.16 Adaptive speculative decoding

Adaptive speculative decoding 解决的是固定 speculative steps 不适合动态 workload 的问题。

如果 `speculative_num_steps` 太小：

```text
draft 本来还能猜更多会被接受的 token，但提前停了
```

如果 `speculative_num_steps` 太大：

```text
draft 猜了很多 token，但 target 拒绝，浪费 draft 计算和显存
```

Adaptive 策略会根据近期 accept length 调整 steps。

SGLang 文档中提到，它会用 EMA 追踪平均接受长度，并在候选 tier 中切换，例如：

```text
candidate_steps = [1, 3, 7]
```

如果近期接受长度高，就倾向更大 steps。

如果近期接受长度低，就降到更小 steps。

这个思路本质是：

```text
让 speculative depth 跟随当前 workload，而不是固定写死。
```

## 31.17 Speculative decoding 和 scheduler

Speculative decoding 会让 scheduler 更复杂。

普通 decode：

```text
每个 running request 本轮生成 1 token
```

Speculative decode：

```text
每个 running request 本轮可能 draft 多个 token
target verify 多个 token
最终接受 0 到 N 个 token
```

Scheduler 要考虑：

1. Draft 阶段的计算成本。
2. Verify 阶段的 target 计算成本。
3. 每个请求的 speculative depth。
4. Draft tokens 对 KV cache 的临时占用。
5. 接受 tokens 后如何更新 request state。
6. 拒绝 tokens 后如何丢弃临时状态。
7. Streaming 一次返回多个 accepted tokens。
8. 不同请求 accept length 不同导致 batch 动态变化。

这就是为什么 speculative decoding 不是简单在 sampler 前加一个小模型，而是 runtime 调度路径的一部分。

## 31.18 和 KV cache 的关系

Speculative decoding 对 KV cache 有额外要求。

Draft 阶段可能会产生候选 token 的 KV。

Target verify 也需要对候选 token 计算或读取 KV。

如果 token 被接受，对应 KV 可以成为正式上下文的一部分。

如果 token 被拒绝，对应临时 KV 不能继续作为正式历史。

因此需要区分：

```text
committed KV: 已接受 token 的 KV
speculative KV: 候选 token 的临时 KV
discarded KV: 被拒绝 token 相关 KV
```

KV cache 管理要保证：

1. 接受的 token KV 被保留。
2. 拒绝的 token KV 被释放或覆盖。
3. Draft model 和 target model 的 KV 不混淆。
4. RadixAttention prefix cache 不错误缓存未接受 token。
5. Memory pool 能承受额外临时 KV 峰值。

这一点很重要：speculative decoding 可能提升 decode 吞吐，但也可能增加显存压力。

## 31.19 和 RadixAttention 的关系

RadixAttention 主要优化 prefill 重复计算。

Speculative decoding 主要优化 decode 主模型轮数。

二者互补：

```text
RadixAttention: 让请求更快完成 prefill，降低 TTFT
Speculative decoding: 让请求 decode 更快，降低 TPOT
```

一个复杂 LLM program 可以同时使用：

1. RadixAttention 复用共享 prompt 前缀。
2. Scheduler 接纳未命中 suffix prefill。
3. Speculative decoding 加速后续长输出。
4. Structured output 约束 token 合法性。

但要注意：speculative 生成的未接受 token 不应该插入 RadixAttention 作为可复用正式 prefix。

只有 committed tokens 才能安全进入长期 cache。

## 31.20 和 structured output 的关系

Structured output 会影响 speculative decoding。

原因是 draft token 也必须满足 grammar 约束，否则 target verify 很可能拒绝或 runtime 无法接受。

两种可能设计：

1. Draft 阶段也使用 grammar mask，只产生合法候选。
2. Draft 阶段先猜，verify/accept 阶段再结合 grammar 过滤。

无论哪种，最终 committed tokens 必须满足 grammar state。

困难点包括：

1. Grammar state 要随 speculative token 前进。
2. 一次接受多个 token 时，要连续更新 grammar state。
3. 拒绝后，要回滚 speculative grammar state。
4. 合法 token 集合可能很小，影响 draft 接受率。
5. Structural tag 和 tool call 的结束条件要和 accept 逻辑一致。

所以 structured output + speculative decoding 是一个更复杂的 runtime 组合，不只是两个开关同时打开。

## 31.21 和 streaming 的关系

普通 decode 每轮通常返回一个 token 或一个文本增量。

Speculative decoding 一轮可能接受多个 token。

这对 streaming 有两个影响。

第一，用户可能一次收到更大的 chunk。

```text
普通 decode: token token token
spec decode: [token token token] [token token] [token]
```

第二，TTFT 不一定明显改善，但后续输出可能更快。

如果 speculative round 本身有额外 draft/verify 开销，first token 甚至可能不变或略受影响。

因此评估时要分开看：

1. TTFT。
2. TPOT。
3. inter-token latency。
4. chunk size。
5. accepted tokens per round。

不要只看主观“流式更快”。

## 31.22 参数调优直觉

Speculative decoding 常见参数包括：

1. speculative algorithm。
2. draft model path。
3. speculative num steps。
4. draft tokens 数量。
5. top-k 或分支数。
6. accept threshold。
7. draft model quantization。
8. draft attention backend。

调参目标是：

```text
提高 accepted tokens per target verify，同时控制 draft overhead 和显存占用
```

`num_steps` 太小：接受率可能高，但每轮前进少。

`num_steps` 太大：候选多，但后面很多被拒绝，浪费计算。

`topk` 太小：候选多样性不足。

`topk` 太大：候选树和 verify 成本上升。

Draft model 太大：draft overhead 高。

Draft model 太小：接受率低。

所以 speculative decoding 必须压测，不适合只凭经验开最大参数。

## 31.23 关键监控指标

上线 speculative decoding 后，至少要监控：

1. accept length 平均值。
2. acceptance rate。
3. accepted tokens/s。
4. drafted tokens/s。
5. rejected tokens/s。
6. target verify latency。
7. draft latency。
8. TPOT。
9. output tokens/s。
10. GPU memory usage。
11. OOM 或 allocation failure。
12. p95/p99 latency。

如果只看 output tokens/s，可能忽略 p99 变差或显存风险。

如果只看 acceptance rate，也可能忽略 draft 模型太慢。

要同时看收益和成本。

## 31.24 常见性能现象

现象一：开启 speculative 后吞吐没有提升。

可能原因：

1. 接受率低。
2. Draft model 太慢。
3. Verify 路径没有并行收益。
4. batch 太小，硬件没吃满。
5. 额外显存导致 batch size 下降。
6. workload 输出短，收益不明显。

现象二：平均吞吐提升，但 p99 变差。

可能原因：

1. 某些请求接受率极低。
2. Speculative depth 太大。
3. Draft/verify 临时 KV 造成显存压力。
4. Scheduler 对 speculative workload 成本估计不足。

现象三：低 temperature 收益明显，高 temperature 收益下降。

这是常见现象。高 temperature 下目标分布更随机，draft 更难猜中。

现象四：代码生成收益明显，开放写作收益一般。

代码和模板化文本重复模式强，draft 更容易预测；开放写作多样性更强，接受率可能更低。

## 31.25 OOM 和显存问题

Speculative decoding 可能增加显存占用。

来源包括：

1. Draft model 权重。
2. Draft model KV cache。
3. Target verify 临时 KV。
4. Candidate tree 或 draft token metadata。
5. CUDA graph 和 workspace。
6. 更大的 batch 或 verify token 数。

如果 OOM，可以考虑：

1. 降低 `speculative_num_steps`。
2. 降低 `speculative_num_draft_tokens`。
3. 降低 top-k 或分支数。
4. 使用更小或量化 draft model。
5. 降低 batch size。
6. 降低 KV cache 预算。
7. 关闭某些不兼容优化。
8. 分离 speculative 和非 speculative 流量。

Speculative decoding 的本质是用额外计算和显存换更少 target decode 轮数。显存不足时，收益可能被反噬。

## 31.26 什么时候适合开启

适合场景：

1. 输出较长。
2. Decode 是主要瓶颈。
3. Draft 接受率高。
4. 有合适 draft 模型或 EAGLE/MTP 支持。
5. 显存有余量。
6. 低 temperature 或确定性任务较多。
7. 代码、模板、结构化输出等可预测任务。

不适合或要谨慎：

1. 输出很短。
2. 瓶颈在 prefill、RAG、tokenizer 或网络。
3. 接受率低。
4. 显存紧张。
5. 高 temperature 创意生成。
6. p99 要求极严但 speculative 路径波动大。
7. 没有合适 draft source。

## 31.27 和 PD 分离的关系

后面章节会讲 PD 分离。

这里先给一个直觉：speculative decoding 主要优化 decode，所以它更直接影响 decode worker 或 decode 资源池。

在 PD 分离架构中：

```text
prefill pool: 处理长 prompt prefill
decode pool: 处理逐 token generation
```

Speculative decoding 通常更适合放在 decode pool 的优化路径中。

但它也会影响系统整体资源：

1. Decode worker 显存需求变化。
2. Output tokens/s 上升后，下游 streaming 压力上升。
3. 更快 decode 可能改变 prefill/decode 队列平衡。
4. Draft model 也需要部署和调度。

所以它不是单机小优化，而会影响 serving 架构容量规划。

## 31.28 常见误解

误解一：Speculative decoding 是用小模型替代大模型。

不对。Draft source 只是提出候选，target model 负责验证，最终质量以 target 为准。

误解二：Speculative decoding 一定提升性能。

不一定。接受率低、draft 太慢、显存紧张或输出太短时，可能没有收益甚至变慢。

误解三：Speculative decoding 主要优化 TTFT。

通常不对。它主要优化 decode 阶段 TPOT 和输出吞吐；TTFT 更多受 queue、tokenize、prefill、prefix cache 影响。

误解四：参数越大越好。

不对。更多 draft steps 和更大 top-k 会增加候选和 verify 成本，接受率不够时会浪费。

误解五：和 structured output 没关系。

不对。Grammar 约束会影响 draft token 合法性、接受率和状态回滚。

## 31.29 面试官会怎么问

问题一：Speculative decoding 为什么能加速？

回答要点：它用便宜 draft source 先生成多个候选 token，再用 target model 并行验证，若接受多个 token，就相当于一次 target verify 前进多个 token，减少主模型 decode 轮数。

问题二：它主要优化 TTFT 还是 TPOT？

回答要点：主要优化 TPOT 和 output tokens/s，因为它作用在 decode 阶段；TTFT 主要由 queue、tokenization、prefill 和 prefix cache 决定。

问题三：收益取决于什么？

回答要点：取决于 draft 速度、接受率、target verify 并行效率、额外显存开销、batch 调度和 workload 采样特征。

问题四：SGLang 支持哪些 speculative 方案？

回答要点：包括 EAGLE/EAGLE3、MTP、DFLASH、Standalone draft model、NGRAM、adaptive speculative decoding 和实验性 overlap scheduler 等。

问题五：为什么 speculative decoding 可能变慢？

回答要点：如果 draft token 被大量拒绝，或者 draft 模型太慢、显存压力降低 batch、verify 开销高，额外成本会抵消减少 target 轮数的收益。

## 31.30 标准回答模板

如果面试官问“SGLang 中 speculative decoding 的角色是什么”，可以这样回答：

```text
Speculative decoding 在 SGLang 中主要是 decode 阶段的加速手段。普通自回归生成每轮 target model 只能生成一个 token，长输出会有很多 decode 轮。Speculative decoding 用一个更便宜的 draft source 先提出多个候选 token，再让 target model 一次性 verify 这些 token。如果 target 接受了多个 draft token，就能用一次 target verify 前进多个 token，从而降低 TPOT、提高 output tokens/s。

SGLang 支持多种 draft 来源和算法，比如 EAGLE/EAGLE3、MTP、DFLASH、Standalone 小 draft model、NGRAM，以及 adaptive speculative decoding。它们的共同点都是 propose and verify，只是候选 token 的来源和验证组织方式不同。

这个优化是否有效取决于接受率和额外开销。接受率高、draft 便宜、输出较长、decode 是瓶颈时收益明显；如果接受率低、draft 太慢、显存紧张或输出很短，可能没有收益。工程上还要处理 scheduler、KV cache、streaming 和 structured output 的交互，尤其是只把 accepted tokens 提交到正式 KV 和 prefix cache，rejected speculative tokens 必须丢弃或回滚。
```

## 31.31 Speculative Decoding 公式、runtime 门禁和可运行 demo

为了把上面的概念落到可验证链路，可以把一个请求抽象成：

```math
r_i=(X_i,Y_i,K_i,g_i)
```

其中 `X_i` 是已提交上下文，`Y_i` 是还要生成的目标 token 序列，`K_i` 是每轮最多 draft token 数，`g_i` 表示是否带 structured output grammar state。

第 `i` 个请求在某轮得到 draft 候选：

```math
D_i=(d_{i,1},d_{i,2},\ldots,d_{i,K_i})
```

在 greedy 教学版中，target verify 计算同一段位置上的目标 token：

```math
Y_i^*=(y^*_{i,1},y^*_{i,2},\ldots,y^*_{i,K_i})
```

接受前缀长度可以写成：

```math
A_i=\max\{a\in\{0,\ldots,K_i\}\mid d_{i,j}=y^*_{i,j},\ 1\le j\le a\}
```

如果第 `A_i+1` 个 draft token 被拒绝，就提交前 `A_i` 个 accepted tokens，并从 target 分布产生一个 fallback / correction token。对采样版 speculative sampling，常见接受概率写成：

```math
\alpha_j=\min\left(1,\frac{p_{\mathrm{target}}(d_j\mid c_j)}{\max(\epsilon,p_{\mathrm{draft}}(d_j\mid c_j))}\right)
```

拒绝后需要从修正分布采样：

```math
r_j(x)=\frac{\max(0,p_{\mathrm{target}}(x\mid c_j)-p_{\mathrm{draft}}(x\mid c_j))}{Z_j}
```

这里 `c_j` 是当前位置上下文，`epsilon` 是避免除零的小常数，`Z_j` 是归一化常数。直觉是：draft 只是提案，target 的概率分布仍然是最终裁判。

工程上至少要看这些指标：

```math
R_{\mathrm{acc}}=\frac{N_{\mathrm{accepted}}}{\max(1,N_{\mathrm{drafted}})}
```

```math
T_{\mathrm{verify}}=\frac{N_{\mathrm{output}}}{\max(1,C_{\mathrm{target}})}
```

```math
G_{\mathrm{call}}=\frac{C_{\mathrm{baseline}}}{\max(1,C_{\mathrm{spec}})}
```

其中 `R_acc` 是接受率，`T_verify` 是每次 target verify 平均提交多少 output tokens，`G_call` 是 target 调用次数下降倍数。粗略延迟收益可以写成：

```math
S_{\mathrm{latency}}=\frac{C_{\mathrm{baseline}}L_{\mathrm{target}}}{C_{\mathrm{spec}}L_{\mathrm{verify}}+N_{\mathrm{drafted}}L_{\mathrm{draft}}+L_{\mathrm{sched}}}
```

如果 `S_latency` 大于 1，说明简化模型里端到端延迟下降；如果小于或接近 1，draft 开销、低接受率或调度开销可能抵消收益。

Speculative decoding 的上线门禁可以写成：

```math
G_{\mathrm{spec}}=G_{\mathrm{draft}}G_{\mathrm{verify}}G_{\mathrm{accept}}G_{\mathrm{fallback}}G_{\mathrm{kv}}G_{\mathrm{grammar}}G_{\mathrm{metric}}
```

这些因子分别要求 draft 来源可解释、target verify 可复盘、accept length 可统计、fallback 正确、临时 KV 可清理、grammar state 不越界、指标能支持灰度和回滚。

下面是一个 0 依赖教学 demo。它同时模拟三类请求：

1. `code_fast`：draft 和 target 很接近，接受率高，target 调用明显下降。
2. `creative_low_accept`：draft 经常猜错，收益很差，说明 speculative decoding 不是必然加速。
3. `json_tool`：draft 试图产生不满足 grammar 的 token，runtime 先阻断，再由 target fallback 修正。

```python
from dataclasses import dataclass


@dataclass
class ToyRequest:
    request_id: str
    target_tokens: list
    draft_tokens: list
    grammar: dict | None = None


class ToySpeculativeDecoder:
    def __init__(self, requests, num_steps=4, target_ms=6.0, draft_ms=0.8):
        self.requests = requests
        self.num_steps = num_steps
        self.target_ms = target_ms
        self.draft_ms = draft_ms

    def _valid(self, request, pos, token):
        if request.grammar is None:
            return True
        return token in request.grammar.get(pos, set())

    def _draft_window(self, request, pos, metrics):
        drafted = []
        for offset in range(self.num_steps):
            draft_pos = pos + offset
            if draft_pos >= len(request.target_tokens):
                break
            token = request.draft_tokens[draft_pos]
            if not self._valid(request, draft_pos, token):
                metrics["grammar_blocked_draft"] += 1
                break
            drafted.append(token)
        metrics["drafted_tokens"] += len(drafted)
        return drafted

    def baseline(self, request):
        return list(request.target_tokens), len(request.target_tokens)

    def speculative(self, request):
        pos = 0
        output = []
        trace = []
        metrics = {
            "target_verify_calls": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "fallback_tokens": 0,
            "grammar_blocked_draft": 0,
            "temp_kv_peak": 0,
            "discarded_temp_tokens": 0,
        }
        while pos < len(request.target_tokens):
            drafted = self._draft_window(request, pos, metrics)
            metrics["target_verify_calls"] += 1
            metrics["temp_kv_peak"] = max(metrics["temp_kv_peak"], len(drafted))
            if not drafted:
                fallback = request.target_tokens[pos]
                output.append(fallback)
                metrics["fallback_tokens"] += 1
                trace.append({"pos": pos, "draft": [], "accepted": [], "fallback": fallback})
                pos += 1
                continue

            target_window = request.target_tokens[pos : pos + len(drafted)]
            accepted = []
            for draft_token, target_token in zip(drafted, target_window):
                if draft_token == target_token:
                    accepted.append(draft_token)
                else:
                    break

            accepted_len = len(accepted)
            output.extend(accepted)
            metrics["accepted_tokens"] += accepted_len
            fallback = None
            if accepted_len < len(drafted):
                fallback = target_window[accepted_len]
                output.append(fallback)
                rejected = len(drafted) - accepted_len
                metrics["rejected_tokens"] += rejected
                metrics["discarded_temp_tokens"] += rejected
                metrics["fallback_tokens"] += 1
                pos += accepted_len + 1
            else:
                pos += accepted_len
            trace.append({"pos": pos, "draft": drafted, "accepted": accepted, "fallback": fallback})
        metrics["committed_tokens"] = len(output)
        return output, metrics, trace

    def run(self):
        rows = []
        summary = {
            "baseline_target_calls": 0,
            "spec_target_calls": 0,
            "drafted_tokens": 0,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "fallback_tokens": 0,
            "grammar_blocked_draft": 0,
            "discarded_temp_tokens": 0,
            "temp_kv_peak": 0,
        }
        for request in self.requests:
            baseline_tokens, baseline_calls = self.baseline(request)
            spec_tokens, metrics, trace = self.speculative(request)
            summary["baseline_target_calls"] += baseline_calls
            summary["spec_target_calls"] += metrics["target_verify_calls"]
            for key in [
                "drafted_tokens",
                "accepted_tokens",
                "rejected_tokens",
                "fallback_tokens",
                "grammar_blocked_draft",
                "discarded_temp_tokens",
            ]:
                summary[key] += metrics[key]
            summary["temp_kv_peak"] = max(summary["temp_kv_peak"], metrics["temp_kv_peak"])
            rows.append(
                {
                    "id": request.request_id,
                    "match_target": spec_tokens == baseline_tokens,
                    "baseline_calls": baseline_calls,
                    "spec_calls": metrics["target_verify_calls"],
                    "accepted": metrics["accepted_tokens"],
                    "drafted": metrics["drafted_tokens"],
                    "fallback": metrics["fallback_tokens"],
                    "grammar_blocked": metrics["grammar_blocked_draft"],
                    "trace_tail": trace[-2:],
                }
            )

        summary["output_tokens"] = sum(len(req.target_tokens) for req in self.requests)
        summary["acceptance_rate"] = round(summary["accepted_tokens"] / max(1, summary["drafted_tokens"]), 3)
        summary["tokens_per_target_call"] = round(summary["output_tokens"] / max(1, summary["spec_target_calls"]), 3)
        summary["target_call_reduction"] = round(summary["baseline_target_calls"] / max(1, summary["spec_target_calls"]), 3)
        baseline_latency = summary["baseline_target_calls"] * self.target_ms
        spec_latency = (
            summary["spec_target_calls"] * self.target_ms
            + summary["drafted_tokens"] * self.draft_ms
            + summary["grammar_blocked_draft"] * 0.2
            + summary["spec_target_calls"] * 0.3
        )
        summary["latency_speedup_est"] = round(baseline_latency / spec_latency, 3)

        gates = {
            "target_output_preserved": all(row["match_target"] for row in rows),
            "acceptance_rate_measured": 0.0 <= summary["acceptance_rate"] <= 1.0,
            "target_calls_reduced": summary["target_call_reduction"] > 1.0,
            "low_accept_case_visible": any(row["accepted"] <= 1 and row["spec_calls"] >= row["baseline_calls"] for row in rows),
            "grammar_blocks_invalid_draft": summary["grammar_blocked_draft"] > 0,
            "temp_kv_cleanup_visible": summary["discarded_temp_tokens"] == summary["rejected_tokens"]
            and summary["discarded_temp_tokens"] > 0,
            "metrics_ready": summary["tokens_per_target_call"] > 1.0,
        }
        gates["speculative_decoding_gate"] = all(gates.values())
        return rows, summary, gates


requests = [
    ToyRequest(
        request_id="code_fast",
        target_tokens=["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b", "<eos>"],
        draft_tokens=["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b", "<eos>"],
    ),
    ToyRequest(
        request_id="creative_low_accept",
        target_tokens=["The", "answer", "is", "not", "obvious", "<eos>"],
        draft_tokens=["A", "very", "different", "path", "appears", "<eos>"],
    ),
    ToyRequest(
        request_id="json_tool",
        target_tokens=["{", '"city"', ":", '"Paris"', "}"],
        draft_tokens=["{", '"city"', ":", "Paris", "}"],
        grammar={
            0: {"{"},
            1: {'"city"'},
            2: {":"},
            3: {'"Paris"', '"London"'},
            4: {"}"},
        },
    ),
]

rows, summary, gates = ToySpeculativeDecoder(requests).run()
print("speculative_rows=", rows)
print("speculative_summary=", summary)
print("speculative_gates=", gates)
```

运行后可以看到类似输出：

```text
speculative_summary= {'baseline_target_calls': 24, 'spec_target_calls': 13, 'drafted_tokens': 35, 'accepted_tokens': 18, 'rejected_tokens': 17, 'fallback_tokens': 6, 'grammar_blocked_draft': 2, 'discarded_temp_tokens': 17, 'temp_kv_peak': 4, 'output_tokens': 24, 'acceptance_rate': 0.514, 'tokens_per_target_call': 1.846, 'target_call_reduction': 1.846, 'latency_speedup_est': 1.306}
speculative_gates= {'target_output_preserved': True, 'acceptance_rate_measured': True, 'target_calls_reduced': True, 'low_accept_case_visible': True, 'grammar_blocks_invalid_draft': True, 'temp_kv_cleanup_visible': True, 'metrics_ready': True, 'speculative_decoding_gate': True}
```

这个 demo 的关键不是把真实 SGLang 内部实现复刻出来，而是让你能复盘 speculative decoding 的工程证据链：

1. target 输出没有被 draft 替代。
2. 接受率和每次 target verify 产出 token 数可观测。
3. 低接受率请求会暴露出来，不能只看总体平均收益。
4. structured output 的 grammar state 会影响 draft 候选。
5. rejected speculative token 的临时 KV 必须丢弃，不能进入正式上下文或 prefix cache。
6. 是否值得开启，要同时看 target call reduction、draft overhead、scheduler overhead、TPOT、显存和 p99。

## 31.32 小练习

1. 用伪代码写一个 draft/verify/accept 的 speculative decoding loop。
2. 解释为什么 speculative decoding 主要优化 TPOT。
3. 给出 3 个接受率高的 workload 和 3 个接受率低的 workload。
4. 对比 Standalone draft model 和 NGRAM draft source。
5. 设计一个 speculative decoding dashboard，包含 accept length、draft latency、verify latency、TPOT、显存和 p99。
6. 解释为什么 rejected token 不能进入 RadixAttention prefix cache。
7. 说明 structured output 如何影响 speculative decoding 的 grammar state。

## 31.33 本章总结

Speculative decoding 的核心是“先猜，再验”。它通过便宜的 draft source 生成候选 token，再用 target model 并行验证，从而在接受率足够高时减少主模型 decode 轮数。

在 SGLang 中，它是和 RadixAttention、scheduler、KV cache、structured output 并列的重要 runtime 优化。RadixAttention 更偏 prefill 和 TTFT，speculative decoding 更偏 decode 和 TPOT。

但 speculative decoding 不是无脑开启的银弹。它需要合适的 draft source、较高接受率、足够显存和正确调度。真正上线时必须用真实 workload 压测，看 accepted tokens、draft overhead、verify latency、TPOT、显存和 p99。

下一章会进入 Multi-turn、tool use 和 agent serving 支持，讨论 SGLang 如何服务多轮状态、工具调用、agent 轨迹和复杂程序执行。
