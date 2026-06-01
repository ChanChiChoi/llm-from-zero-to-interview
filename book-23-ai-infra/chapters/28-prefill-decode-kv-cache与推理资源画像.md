# 第 28 章 Prefill、Decode、KV Cache 与推理资源画像

上一章讲了模型服务运行时。本章继续往下拆，讲大模型推理中最关键的三个概念：Prefill、Decode 和 KV Cache。

很多人说推理慢、吞吐低、GPU 利用率不高，但说不清慢在哪里。真正要理解推理平台，必须把一次生成请求拆成阶段，分别分析它们消耗什么资源。

先记住一句话：

> 大模型推理不是一次普通函数调用，而是由 prefill、decode 和 KV cache 管理共同决定的在线计算过程。

## 28.1 一次大模型生成请求发生了什么

用户发起一个请求：

```text
请解释 Transformer 的 attention 机制。
```

模型返回一段文本：

```text
Attention 机制可以理解为...
```

从 runtime 内部看，大致发生了这些步骤：

1. 接收请求。
2. tokenizer 把文本转成 token。
3. 模型处理输入 token。
4. 生成第一枚输出 token。
5. 继续逐个生成后续 token。
6. 达到停止条件后结束。
7. 回收 KV cache。

其中第 3 步主要对应 prefill，第 4 到第 6 步主要对应 decode。

## 28.2 Prefill 是什么

Prefill 是处理输入 prompt 的阶段。

假设输入有 2000 个 token，模型需要一次性处理这 2000 个 token，并为每一层 attention 生成对应的 KV cache。

Prefill 的输入是完整 prompt。

Prefill 的输出包括：

1. 下一 token 的 logits。
2. 每一层的 KV cache。

可以把 prefill 理解为：模型先读完整个题目，建立上下文。

## 28.3 Decode 是什么

Decode 是逐 token 生成输出的阶段。

生成第一个 token 后，模型会把这个 token 加入上下文，再生成下一个 token。这个过程不断重复，直到遇到停止符、达到最大输出长度，或者请求超时。

Decode 的特点是：

1. 每一步通常只生成一个 token。
2. 每一步都要读取已有 KV cache。
3. 输出越长，decode 步数越多。
4. streaming 返回主要发生在 decode 阶段。

可以把 decode 理解为：模型一边想，一边一个词一个词往外说。

## 28.4 Prefill 和 Decode 的核心差异

Prefill 和 decode 的资源特征不同。

| 阶段 | 主要处理对象 | 主要瓶颈 | 延迟影响 | 吞吐影响 |
| --- | --- | --- | --- | --- |
| Prefill | 输入 token | 计算 | 影响 TTFT | 影响输入吞吐 |
| Decode | 输出 token | 显存带宽、KV cache 访问 | 影响 TPOT | 影响输出吞吐 |

这张表非常重要。

如果输入很长，TTFT 可能很高。

如果输出很长，整体延迟和 GPU 占用时间会很高。

如果并发很高，KV cache 会成为主要瓶颈。

## 28.5 TTFT 和 TPOT

TTFT 是 Time To First Token，即从请求进入到第一个 token 返回的时间。

TTFT 通常包括：

1. 网关耗时。
2. 路由耗时。
3. 排队耗时。
4. tokenizer 耗时。
5. prefill 耗时。
6. 第一个 token 采样耗时。

TPOT 是 Time Per Output Token，即平均每生成一个输出 token 的耗时。

TPOT 通常受 decode 阶段影响更大。

用户体验上：

1. TTFT 决定“多久开始响应”。
2. TPOT 决定“输出流不流畅”。
3. 总延迟约等于 TTFT 加上输出 token 数乘以 TPOT。

粗略表达：

```text
total_latency ~= TTFT + output_tokens * TPOT
```

真实系统还要考虑网络、队列、sampling、后处理和 streaming flush。

## 28.6 KV Cache 是什么

Transformer 每层 attention 会计算 Query、Key、Value。

在自回归生成中，历史 token 的 Key 和 Value 不需要每一步重复计算，可以缓存起来。这就是 KV cache。

KV cache 的作用是：

1. 减少重复计算。
2. 加速 decode。
3. 支持长上下文生成。

没有 KV cache，每生成一个 token 都要重新处理全部历史上下文，成本会高得不可接受。

## 28.7 KV Cache 为什么占显存

KV cache 会保存每一层、每个 token、每个 head 的 Key 和 Value。

它的规模大致和这些因素相关：

1. batch size。
2. 序列长度。
3. 模型层数。
4. hidden size。
5. attention head 数。
6. 数据类型精度。

粗略理解：

```text
KV cache memory ∝ batch_size * sequence_length * num_layers * hidden_size * dtype_size
```

这不是精确公式，但足够帮助你理解瓶颈。

长上下文和高并发都会快速放大 KV cache 显存占用。

## 28.8 一个直观例子

假设有两个请求：

请求 A：输入 200 token，输出 100 token。

请求 B：输入 20000 token，输出 100 token。

它们的输出长度一样，但 B 的 TTFT 会明显更高，因为 prefill 要处理更长输入。

再看两个请求：

请求 C：输入 500 token，输出 50 token。

请求 D：输入 500 token，输出 2000 token。

C 和 D 的 prefill 成本接近，但 D 的 decode 时间远长于 C，还会更长时间占用 KV cache 和 GPU 资源。

所以不能只用 QPS 衡量推理负载。

必须看 token 维度。

## 28.9 推理资源画像

推理资源画像就是描述不同请求消耗哪些资源。

至少要看：

1. 输入 token 数。
2. 输出 token 数。
3. 上下文长度。
4. 并发数。
5. batch size。
6. prefill latency。
7. decode latency。
8. KV cache 占用。
9. GPU 利用率。
10. 显存带宽。
11. 网络和 streaming 开销。

一个推理平台如果不知道请求画像，就无法做正确调度。

## 28.10 为什么 QPS 不够用

两个系统都处理 100 QPS，但负载可能完全不同。

系统 A：每个请求输入 100 token，输出 50 token。

系统 B：每个请求输入 8000 token，输出 1000 token。

它们的 QPS 一样，但系统 B 的计算量、显存占用和响应时间远高于系统 A。

所以大模型推理更应该关注：

1. input tokens/s。
2. output tokens/s。
3. active sequences。
4. total tokens in KV cache。
5. cost per token。
6. SLO 下可承载吞吐。

QPS 只是表层指标。

## 28.11 Prefill 的优化方向

Prefill 优化常见方向：

1. 提高矩阵计算效率。
2. 使用更高效 kernel。
3. 优化 attention 实现。
4. prompt cache。
5. prefix cache。
6. 输入截断或压缩。
7. 分离 prefill 和 decode 资源池。
8. 对长 prompt 请求单独调度。

长上下文场景中，prefill 往往是 TTFT 的主要来源。

如果业务强依赖长文档、RAG、代码仓库上下文或多轮对话，prefill 优化会非常关键。

## 28.12 Decode 的优化方向

Decode 优化常见方向：

1. continuous batching。
2. PagedAttention。
3. KV cache 复用。
4. KV cache 量化。
5. speculative decoding。
6. 更高效 sampling。
7. 限制最大输出长度。
8. 分离短输出和长输出请求。
9. 优化 streaming flush 策略。

Decode 阶段的特点是每步计算粒度小、步骤多、频繁访问 KV cache。

这也是为什么 decode 不一定能把 GPU 算力打满，显存带宽和调度开销可能成为瓶颈。

## 28.13 KV Cache 的管理策略

KV cache 管理常见问题：

1. 如何分配。
2. 如何释放。
3. 如何避免碎片。
4. 如何处理长上下文。
5. 如何处理请求取消。
6. 如何处理超时。
7. 如何做前缀复用。
8. 如何在显存不足时驱逐。

常见策略包括：

1. 分块管理。
2. 分页管理。
3. 引用计数。
4. prefix cache。
5. LRU 驱逐。
6. 按租户或优先级隔离。
7. 显存水位控制。

KV cache 管理不好，系统会出现延迟抖动和 OOM。

## 28.14 PagedAttention 的直觉

PagedAttention 的直觉类似操作系统分页。

传统方式可能为每个请求预留连续 KV cache 空间，容易浪费和碎片化。

PagedAttention 把 KV cache 拆成固定大小的块，通过映射表管理逻辑序列和物理块。

好处是：

1. 减少显存浪费。
2. 支持更高并发。
3. 更容易复用前缀。
4. 更好处理不同长度请求。

你不需要一开始就掌握所有实现细节，但要理解它解决的是 KV cache 显存管理问题。

## 28.15 长上下文的特殊问题

长上下文会同时放大 prefill 和 KV cache 压力。

它会带来：

1. TTFT 增加。
2. KV cache 显存占用增加。
3. batch 中长度差异更大。
4. 短请求被长请求影响。
5. 成本显著上升。
6. p99 延迟变差。

长上下文优化手段包括：

1. prompt 压缩。
2. RAG 召回控制。
3. sliding window attention。
4. prefix cache。
5. 长短请求分队列。
6. 长上下文专用实例。
7. 限制上下文配额。

长上下文不是免费能力，平台必须显式治理。

## 28.16 多租户下的 KV Cache 问题

多租户环境中，KV cache 还涉及公平性和隔离。

问题包括：

1. 一个租户的长请求占满显存。
2. 高优先级业务被低优先级业务阻塞。
3. KV cache 驱逐影响用户体验。
4. cache 复用可能带来数据隔离风险。
5. 成本归因不清。

平台应该支持：

1. 租户级上下文长度限制。
2. 租户级并发限制。
3. 租户级 token 配额。
4. 优先级队列。
5. cache 隔离策略。
6. 详细用量统计。

多租户推理平台不能只追求整体吞吐，还要保证公平性和安全隔离。

## 28.17 Prefill/Decode 分离

有些系统会把 prefill 和 decode 分离到不同资源池。

原因是两者资源特征不同：

1. Prefill 更偏计算密集。
2. Decode 更偏显存带宽和 KV cache。
3. Prefill 请求长度差异大。
4. Decode 持续时间受输出长度影响。

分离后的可能架构：

```text
Request -> Prefill Pool -> KV Transfer -> Decode Pool -> Streaming Response
```

好处：

1. 更细粒度调度资源。
2. 长 prompt 不一定阻塞 decode。
3. 可按阶段独立扩缩容。
4. 更适合超大规模推理集群。

代价：

1. 系统复杂度增加。
2. KV cache 传输成本增加。
3. 调度和一致性更难。
4. 故障排查更复杂。

所以它更适合大规模和高优化场景，不一定适合所有团队。

## 28.18 如何定位推理慢

当用户说“模型慢”时，不要直接说加 GPU。

应该按阶段排查：

1. 请求是否在网关排队？
2. router 是否选择了拥塞实例？
3. runtime 队列长度是否过高？
4. 输入 token 是否变长？
5. prefill latency 是否上升？
6. output token 是否变长？
7. TPOT 是否上升？
8. KV cache 是否接近水位？
9. GPU 显存是否不足？
10. batch 策略是否变化？
11. streaming 是否被网络或客户端拖慢？

只有定位到阶段，才能做正确优化。

## 28.19 面试常见追问

问题一：为什么长 prompt 会影响 TTFT？

可以回答：长 prompt 会增加 prefill 计算量，模型需要处理全部输入 token 并生成 KV cache，所以第一个 token 返回前的时间会变长。

问题二：为什么长输出会影响总延迟？

可以回答：输出是逐 token decode 的，输出越长 decode 步数越多，整体延迟约等于 TTFT 加 output tokens 乘以 TPOT。

问题三：为什么 KV cache 会成为瓶颈？

可以回答：KV cache 与并发数、序列长度、层数和 hidden size 相关，会占用大量显存。高并发和长上下文下，显存容量、显存带宽和 cache 管理都会成为瓶颈。

问题四：为什么不能只用 QPS 评估推理系统？

可以回答：不同请求 token 数差异巨大，同样 QPS 下输入输出 token 数可能相差几十倍。应该结合 input tokens/s、output tokens/s、TTFT、TPOT、KV cache 和成本来评估。

## 28.20 小练习

1. Prefill 和 decode 的区别是什么？
2. TTFT 和 TPOT 分别受哪些因素影响？
3. 为什么 KV cache 能加速 decode？
4. 为什么 KV cache 会占大量显存？
5. 长上下文会带来哪些平台问题？
6. 为什么高 QPS 不一定代表高负载？
7. Prefill/decode 分离有什么收益和代价？
8. 用户反馈“模型慢”时，你会按什么顺序排查？

## 28.21 本章小结

本章讲了 Prefill、Decode、KV Cache 和推理资源画像。

你需要记住：

1. Prefill 处理输入 prompt，主要影响 TTFT。
2. Decode 逐 token 生成输出，主要影响 TPOT 和总延迟。
3. KV cache 避免重复计算，但会消耗大量显存。
4. 长上下文会同时放大 prefill 成本和 KV cache 压力。
5. 大模型推理不能只看 QPS，要看 token 吞吐、延迟、cache 和成本。
6. 推理优化必须按阶段定位瓶颈，而不是盲目加机器。

下一章我们会继续讲 Continuous Batching、PagedAttention 和队列调度，理解 runtime 如何在高并发下提升吞吐并控制延迟。
