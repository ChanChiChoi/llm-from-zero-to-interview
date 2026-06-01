# 第 4 章 Prefill、Decode、KV Cache 和 Token Streaming

上一章我们从请求生命周期出发，走完了接入、排队、调度、执行、输出和资源释放。本章把生命周期中最关键的四个概念单独拎出来：Prefill、Decode、KV Cache 和 Token Streaming。

这四个概念决定了 LLM serving 的基本性能边界。后面无论讲 vLLM 的 PagedAttention、SGLang 的 RadixAttention，还是 PD 分离、chunked prefill、多级 KV Cache，本质上都是围绕它们做优化。

一句话概括：

> Prefill 决定首 token 等多久，Decode 决定后续 token 多快，KV Cache 决定显存和并发，Token Streaming 决定用户如何感知生成过程。

## 4.1 为什么这四个概念最重要

大模型推理看起来像一次 `generate()` 调用，但内部不是一次性计算完整答案。

它更像下面这条流水线：

```text
prompt tokens
  -> prefill
  -> KV Cache
  -> first token
  -> decode one token
  -> stream one chunk
  -> decode next token
  -> stream next chunk
  -> ...
  -> finish
```

如果你只知道“模型输入 prompt，输出答案”，就很难理解推理性能为什么复杂。

如果你知道 prefill、decode、KV cache 和 streaming，就能解释很多线上现象：

1. 为什么长 prompt 首 token 慢。
2. 为什么输出过程中有时一卡一卡的。
3. 为什么显存够放模型权重，却撑不住高并发。
4. 为什么 batch 越大不一定体验越好。
5. 为什么 PD 分离要把 prefill 和 decode 放到不同资源池。

## 4.2 Prefill 是什么

Prefill 是处理输入 prompt 的阶段。

如果用户输入了 2000 个 token，模型需要先对这 2000 个 token 做一次前向计算，得到每一层的 KV Cache，并产生下一个 token 的 logits。

简化流程：

```text
input_ids[0:prompt_len]
  -> embedding
  -> transformer layers
  -> logits at last position
  -> sample first output token
```

Prefill 的输入长度通常等于 prompt 长度，计算量和 prompt token 数强相关。

对用户来说，prefill 最直接影响的是 TTFT，也就是 time to first token。

如果 TTFT 很高，用户会觉得服务“半天没反应”。即使后面 decode 很快，体验也已经被首 token 等待拖差了。

## 4.3 Prefill 的计算特点

Prefill 阶段通常有这些特点：

1. 一次处理多个 prompt token。
2. 矩阵乘法规模较大。
3. 更容易利用 GPU 算力。
4. 计算量随 prompt 长度上升。
5. 会产生大量 KV Cache 写入。

从资源角度看，prefill 更偏 compute-bound，也就是更依赖 GPU 计算能力。当然这不是绝对的，长上下文、注意力实现和 cache 写入也会带来内存压力。

Prefill 的主要优化方向包括：

1. 提升 batched prefill 的吞吐。
2. 控制单个长 prompt 对系统的冲击。
3. 使用 chunked prefill 把长 prompt 拆成多段。
4. 复用 prefix cache，避免重复 prefill。
5. 在 PD 分离架构中把 prefill 放到更适合大计算的资源池。

## 4.4 Decode 是什么

Decode 是逐 token 生成阶段。

prefill 之后，模型已经有了 prompt 的 KV Cache。此后每一轮只输入上一个 token，并读取历史 KV Cache，生成下一个 token。

简化流程：

```text
last_token + past KV Cache
  -> transformer layers
  -> logits
  -> sample next token
  -> append token
  -> update KV Cache
```

Decode 会重复很多轮。用户看到的答案就是 decode 一轮一轮积累出来的。

如果生成 500 个 token，decode 至少要进行 500 轮模型前向。每轮可能很小，但重复次数很多。

因此 decode 对 TPOT 影响很大，也就是 time per output token。

## 4.5 Decode 的资源特点

Decode 阶段通常有这些特点：

1. 每个请求每轮只处理一个新 token。
2. 单轮计算量比 prefill 小。
3. 需要频繁读取 KV Cache。
4. 对内存带宽和调度开销敏感。
5. batch 中请求数量会动态变化。

Decode 常被认为更偏 memory-bound，因为模型需要不断读取历史 KV Cache。上下文越长，KV Cache 越大，读取压力越明显。

Decode 的主要优化方向包括：

1. continuous batching，提高每轮执行的有效请求数。
2. 高效 KV Cache 布局，减少碎片和无效读取。
3. kernel 优化，降低小 batch decode 的开销。
4. speculative decoding，减少主模型 decode 轮数。
5. 在 PD 分离架构中使用适合 decode 的资源池。

## 4.6 Prefill 和 Decode 的差异

可以用一张表总结：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 输入 | prompt 中的多个 token | 上一轮生成的 token |
| 输出 | 第一个输出 token 和 KV Cache | 下一个输出 token 和更新后的 KV Cache |
| 频率 | 每个请求通常一次 | 每个输出 token 一次 |
| 主要影响 | TTFT | TPOT、持续吞吐 |
| 资源倾向 | 更偏大计算 | 更偏 cache 读取和调度 |
| 优化重点 | 长 prompt、prefix 复用、chunking | continuous batching、cache 管理、decode kernel |

这个差异非常重要。很多系统设计问题都来自 prefill 和 decode 混在一起带来的冲突。

例如，一个长 prompt prefill 可能占用大量计算资源，导致正在 decode 的请求输出卡顿。反过来，如果系统一直优先 decode，新请求的 TTFT 又会变差。

Scheduler 要做的，就是在 prefill 和 decode 之间不断做权衡。

## 4.7 KV Cache 是什么

Transformer 自注意力中，每个 token 会产生 key 和 value。生成下一个 token 时，模型需要关注历史 token。如果每轮都重新计算历史 token 的 key/value，成本会非常高。

KV Cache 的作用就是保存历史 token 的 key/value，让 decode 阶段只计算新 token。

没有 KV Cache：

```text
第 1 轮：计算 prompt + token1
第 2 轮：重新计算 prompt + token1 + token2
第 3 轮：重新计算 prompt + token1 + token2 + token3
```

有 KV Cache：

```text
prefill：计算 prompt，保存 KV
decode 1：只计算 token1，读取历史 KV
decode 2：只计算 token2，读取历史 KV
decode 3：只计算 token3，读取历史 KV
```

KV Cache 是自回归推理能高效运行的关键。

## 4.8 KV Cache 为什么占显存

KV Cache 保存的是每层、每个历史 token、每个 attention head 的 key/value。

粗略看，它和这些因素正相关：

1. 层数。
2. hidden size 或 head dimension。
3. attention head 数量。
4. batch 中请求数。
5. 每个请求的上下文长度。
6. cache 的数据类型，比如 FP16、BF16、FP8。

可以记一个直觉：模型权重是相对固定的，KV Cache 会随着并发和上下文长度动态增长。

这解释了一个常见现象：模型权重明明能放进 GPU，为什么一上并发就 OOM？原因往往不是权重变大了，而是 KV Cache 占满了剩余显存。

## 4.9 KV Cache 管理的难点

单请求时，KV Cache 可以简单地随请求增长。多请求 serving 中，它变成了内存管理问题。

难点包括：

1. 分配：新请求进入时，如何分配 cache 空间。
2. 扩容：decode 每生成一个 token，都要追加 cache。
3. 释放：请求结束后必须归还 cache。
4. 碎片：不同请求长度不同，连续分配容易浪费。
5. 抢占：资源不足时是否暂停或重算部分请求。
6. 复用：相同 prefix 能否共享 cache。
7. 迁移：PD 分离或多级缓存中，cache 是否要跨设备移动。

vLLM 的 PagedAttention 就是为了解决 KV Cache 连续分配和碎片问题。它把 KV Cache 分成块，类似操作系统里的分页内存，让请求的逻辑 token 序列可以映射到非连续物理块。

## 4.10 Token Streaming 是什么

Token Streaming 是边生成边返回。

非流式输出是：

```text
生成完整答案 -> 一次性返回
```

流式输出是：

```text
生成 token1 -> 返回片段1
生成 token2 -> 返回片段2
生成 token3 -> 返回片段3
```

对用户体验来说，streaming 非常重要。哪怕总耗时一样，如果用户能很快看到第一个 token，并持续看到内容增长，就会感觉系统更快。

Streaming 和 TTFT、TPOT 都有关：

1. TTFT 决定多久看到第一段输出。
2. TPOT 决定后续输出是否顺滑。
3. 抖动决定输出是否一卡一卡。

## 4.11 Streaming 的工程细节

Token Streaming 不是简单 `print(token)`。

工程上要处理：

1. token 到文本的增量解码。
2. Unicode 字符边界。
3. stop sequence 跨 token 匹配。
4. SSE、WebSocket 或 gRPC stream。
5. 客户端断开检测。
6. backpressure，也就是客户端读得慢。
7. 错误和结束事件。
8. usage 统计和 finish reason。

一个典型流式事件可能包括：

```json
{
  "id": "request-123",
  "delta": "Cache",
  "finish_reason": null
}
```

结束时再返回：

```json
{
  "id": "request-123",
  "delta": "",
  "finish_reason": "stop"
}
```

如果客户端中途断开，engine 必须把请求标记为 aborted，并释放 KV Cache。否则看起来用户已经走了，GPU 还在继续为这个请求生成。

## 4.12 四者如何串起来

把四个概念放到同一条链路中：

```text
1. Prompt tokenization
2. Prefill prompt tokens
3. 建立 KV Cache
4. 采样第一个 token
5. Streaming 返回第一个文本片段
6. Decode 下一 token
7. 更新 KV Cache
8. Streaming 返回下一文本片段
9. 重复 6-8
10. 请求结束，释放 KV Cache
```

可以看到，KV Cache 贯穿 prefill 和 decode，streaming 则把 decode 结果及时暴露给客户端。

Prefill、decode、cache、streaming 不是四个孤立模块，而是一条请求执行链路中的不同职责。

## 4.13 对 Scheduler 的影响

Scheduler 必须理解这些阶段，否则无法做出合理决策。

它要知道：

1. 新请求 prefill 会消耗多少 token budget。
2. running 请求 decode 需要多少 cache。
3. 当前 KV Cache 是否足够容纳新请求。
4. 哪些请求已经接近结束。
5. 哪些请求启用了 streaming，需要输出平滑。
6. 长 prefill 是否应该拆成 chunk。

一个简单但实用的调度目标是：

1. 不让新请求饿死，控制 TTFT。
2. 不让老请求卡住，控制 TPOT。
3. 不超出 KV Cache 显存预算。
4. 尽量让 GPU 每轮都有足够有效工作。

这也是为什么推理框架的 scheduler 比普通任务队列复杂得多。

## 4.14 常见性能现象解释

现象一：prompt 变长后，首 token 明显变慢。

原因通常是 prefill token 数增加，prefill 延迟上升。

现象二：输出前面快，后面越来越慢。

原因可能是上下文越来越长，KV Cache 读取压力增大，也可能是 batch 内其他请求影响。

现象三：并发上来后显存爆掉。

原因通常是 KV Cache 随并发和上下文长度增长。

现象四：总耗时没变，但用户觉得更快了。

原因可能是 TTFT 降低或 streaming 更平滑。

现象五：GPU 利用率不低，但用户输出一卡一卡。

原因可能是 decode scheduling 不稳定，或者流式输出通道阻塞。

## 4.15 常见误区

误区一：Prefill 和 Decode 只是代码里的两个阶段，不影响架构。

实际上它们资源画像不同，直接影响 scheduler、PD 分离、扩缩容和性能调优。

误区二：KV Cache 只是优化项，不是必须关注的核心。

没有 KV Cache，自回归推理成本会非常高；KV Cache 管理不好，高并发 serving 会很快遇到显存瓶颈。

误区三：Streaming 只影响前端体验。

Streaming 还影响连接管理、取消、backpressure、finish reason 和资源释放。

误区四：TTFT 只和模型大小有关。

TTFT 还和排队、prompt 长度、prefill batch、prefix cache、调度策略和系统负载有关。

误区五：TPOT 只和 GPU 算力有关。

TPOT 还受 KV Cache 读取、batch 动态变化、kernel、调度开销和输出通道影响。

## 4.16 面试追问

1. Prefill 和 Decode 的区别是什么？
2. 为什么 Prefill 主要影响 TTFT，Decode 主要影响 TPOT？
3. KV Cache 为什么能加速自回归生成？
4. KV Cache 为什么会成为高并发 serving 的显存瓶颈？
5. Streaming 输出有哪些工程难点？
6. 为什么长 prompt 会拖慢首 token？
7. 为什么输出过程中可能一卡一卡？
8. 如果显存充足但 TPOT 很差，你会从哪些方向排查？

参考回答思路：

1. 先定义 prefill、decode、KV cache 和 streaming。
2. 再说明资源画像：prefill 偏大计算，decode 更受 KV Cache 和调度影响。
3. 然后关联指标：prefill 对 TTFT 关键，decode 对 TPOT 关键，KV Cache 对并发关键，streaming 对感知体验关键。
4. 最后补工程 trade-off：scheduler 要在 TTFT、TPOT、吞吐、显存和公平性之间平衡。

## 4.17 小练习

1. 画出一个请求从 prefill 到 decode 再到 streaming 的流程图。
2. 假设 prompt 长度从 1k 增加到 8k，TTFT 可能发生什么变化？为什么？
3. 假设并发从 10 增加到 100，KV Cache 会带来什么压力？
4. 写一段伪代码，表示 decode 一轮后把 token 推送到 stream。
5. 解释为什么客户端断开后必须释放 KV Cache。

## 4.18 本章小结

本章讲清了 LLM serving 中最重要的四个执行概念。

Prefill 处理输入 prompt，决定首 token 等待；Decode 逐 token 生成，决定持续输出速度；KV Cache 复用历史 key/value，决定显存和并发边界；Token Streaming 把生成结果增量返回给用户，决定感知体验和连接生命周期。

下一章我们会把这些概念转成可量化指标，系统讲 TTFT、TPOT、吞吐、并发、显存和成本之间的关系。
