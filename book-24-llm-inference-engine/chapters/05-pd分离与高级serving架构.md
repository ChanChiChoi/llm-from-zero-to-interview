# 第 5 章 TTFT、TPOT、吞吐、并发、显存和成本

前几章讲了 serving engine 的生命周期和核心执行阶段。本章开始把这些机制转成可量化指标：TTFT、TPOT、吞吐、并发、显存和成本。

做 LLM serving，不能只问“QPS 多少”。大模型服务的性能是多维的：用户关心等多久、输出是否顺滑；平台关心能服务多少并发；业务关心每百万 token 成本；工程师关心显存、KV Cache、batch、调度和稳定性。

一句话概括：

> LLM serving 的优化目标不是单一 QPS，而是在 TTFT、TPOT、吞吐、并发、显存和成本之间做系统权衡。

## 5.1 为什么传统 QPS 不够用

在传统 Web 服务里，我们经常看 QPS、平均延迟、p95 延迟、错误率。这些指标仍然重要，但不足以描述 LLM serving。

原因是 LLM 请求不是一次固定计算：

1. 输入 prompt 长度不同。
2. 输出 token 数不同。
3. 有的请求流式返回，有的请求一次性返回。
4. Prefill 和 decode 的资源画像不同。
5. KV Cache 会随上下文和并发动态增长。
6. 同一个请求会占用多轮 decode 调度。

两个请求的 QPS 都是 1，但成本可能完全不同：

```text
请求 A：输入 50 tokens，输出 20 tokens
请求 B：输入 8000 tokens，输出 2000 tokens
```

如果只看请求数，A 和 B 一样。如果看 token，它们差几个数量级。

所以 LLM serving 必须转向 token 视角和阶段视角。

## 5.2 TTFT：Time To First Token

TTFT 是 time to first token，表示从客户端发起请求，到收到第一个输出 token 的时间。

可以拆成：

```text
TTFT = 网络接入时间 + 排队时间 + tokenization 时间 + prefill 时间 + 首 token 采样与返回时间
```

其中最常见的大头是排队时间和 prefill 时间。

TTFT 对用户体验非常关键。因为用户点下发送后，第一反应是看系统有没有响应。即使最终答案生成很快，如果首 token 等了很久，用户也会觉得服务慢。

影响 TTFT 的因素包括：

1. prompt 长度。
2. waiting queue 长度。
3. prefill batch 策略。
4. 是否有 prefix cache 命中。
5. 是否启用 chunked prefill。
6. GPU 是否繁忙。
7. tokenizer 和 chat template 处理开销。
8. 调度策略是否优先新请求。

## 5.3 如何优化 TTFT

优化 TTFT 的常见方向：

1. 控制排队时间。
2. 对长 prompt 做 chunked prefill。
3. 使用 prefix cache 复用共享前缀。
4. 限制超长输入，避免单请求拖垮队列。
5. 优化 tokenization 和请求预处理。
6. 在 scheduler 中给 waiting 请求一定机会。
7. 使用独立 prefill 资源池，减少和 decode 互相干扰。

但 TTFT 优化也有代价。

如果过度优先新请求 prefill，正在 decode 的请求可能输出变慢，TPOT 变差。用户会看到首 token 很快，但后续一卡一卡。

所以 TTFT 不能孤立优化，要和 TPOT 一起看。

## 5.4 TPOT：Time Per Output Token

TPOT 是 time per output token，表示生成阶段每个输出 token 的平均耗时。

简化计算：

```text
TPOT = decode 总耗时 / 输出 token 数
```

如果一个请求首 token 后又生成了 100 个 token，decode 阶段耗时 5 秒，那么平均 TPOT 是 50ms/token。

TPOT 反映持续输出速度。流式场景下，TPOT 越低，用户看到的输出越快。TPOT 抖动越小，输出越平滑。

影响 TPOT 的因素包括：

1. decode batch 大小。
2. KV Cache 读取效率。
3. 上下文长度。
4. attention kernel 和模型 kernel。
5. sampling 开销。
6. scheduler 每轮开销。
7. streaming 输出是否阻塞。
8. GPU 显存带宽和计算能力。

## 5.5 如何优化 TPOT

优化 TPOT 的常见方向：

1. 使用 continuous batching，提高每轮 decode 的有效 batch。
2. 优化 KV Cache 布局，减少碎片和无效访问。
3. 使用高效 attention kernel。
4. 降低 scheduler 每轮开销。
5. 避免网络 streaming 阻塞模型执行。
6. 对长上下文使用更合适的 cache 或 attention 策略。
7. 使用 speculative decoding 减少主模型 decode 轮数。

TPOT 优化的代价也很明显。

如果为了提高 decode 吞吐，让 batch 尽量大，可能会让新请求等待更久，TTFT 变差。如果为了减少每轮调度开销，把调度粒度做粗，也可能降低响应灵活性。

## 5.6 吞吐：tokens/s 比 QPS 更重要

LLM serving 的吞吐通常用 tokens/s 衡量，而不是只看 QPS。

常见吞吐指标包括：

1. input tokens/s：每秒处理多少输入 token。
2. output tokens/s：每秒生成多少输出 token。
3. total tokens/s：输入和输出 token 总量。
4. per-GPU tokens/s：每张 GPU 的 token 吞吐。
5. accepted tokens/s：如果使用 speculative decoding，还会看被接受 token 的吞吐。

为什么 tokens/s 重要？因为 LLM 的成本和计算量主要跟 token 相关。

同样 10 QPS：

```text
场景 A：每个请求输入 100，输出 50，总计 1500 tokens/s
场景 B：每个请求输入 4000，输出 1000，总计 50000 tokens/s
```

QPS 一样，系统压力完全不同。

面试里如果候选人只说“系统支持 100 QPS”，但不说明输入输出 token 分布、上下文长度和延迟目标，这个数字基本没有意义。

## 5.7 并发：不是连接数那么简单

并发在 LLM serving 中至少有三种含义：

1. 同时连接的客户端数量。
2. waiting queue 中等待的请求数量。
3. running set 中正在占用 KV Cache 和 decode slot 的请求数量。

真正影响 engine 的，主要是 active sequences，也就是正在被调度或占用 KV Cache 的序列数。

高并发会带来几个压力：

1. waiting queue 增长，TTFT 变差。
2. running sequences 增多，KV Cache 占用变大。
3. decode batch 变大，吞吐可能提高但单请求延迟可能变差。
4. scheduler 决策更复杂。
5. streaming 连接和网络输出压力上升。

所以并发不是越高越好。一个系统标称支持 1000 并发，如果 p99 TTFT 已经不可接受，就不算有效并发。

## 5.8 显存：权重、KV Cache 和临时 buffer

LLM serving 的显存占用通常包括：

1. 模型权重。
2. KV Cache。
3. activation 或临时计算 buffer。
4. CUDA graph 或 kernel workspace。
5. 通信 buffer，比如 tensor parallel 场景。
6. 框架运行时开销。

其中模型权重相对固定，KV Cache 随请求动态变化。

一个常见错误是只估算模型权重：

```text
7B 模型，FP16 权重大约 14GB，所以 24GB GPU 肯定够。
```

这个判断不完整。剩余显存还要容纳 KV Cache、临时 buffer 和运行时开销。上下文长、并发高时，KV Cache 可能很快吃掉全部剩余空间。

显存上限会反过来限制：

1. 最大并发序列数。
2. 最大上下文长度。
3. 最大 batch token 数。
4. 是否能启用更大的 prefill batch。
5. 是否需要量化、分片或 offload。

## 5.9 成本：每 token 成本才是核心

推理成本通常可以用每百万 token 成本来估算。

粗略公式：

```text
每 token 成本 = 单位时间 GPU 成本 / 单位时间有效 token 吞吐
```

如果一张 GPU 每小时成本固定，那么 tokens/s 越高，每 token 成本越低。但这不意味着只追求最大吞吐。

原因是业务还要满足延迟 SLO。

一个系统可以通过大 batch 把 tokens/s 做高，但 TTFT 和 TPOT 变差，用户体验无法接受。另一个系统延迟很好，但 GPU 利用率太低，成本无法承受。

成本优化的目标是：在满足 SLO 的前提下，提高有效吞吐和资源利用率。

常见手段包括：

1. continuous batching。
2. prefix cache。
3. KV Cache 高效管理。
4. 模型量化。
5. speculative decoding。
6. 请求路由到合适大小的模型。
7. 自动扩缩容。
8. 不同优先级请求分层服务。

## 5.10 指标之间的冲突

这些指标经常互相冲突。

常见冲突包括：

1. 大 batch 提高吞吐，但可能增加 TTFT。
2. 优先 prefill 降低 TTFT，但可能让 decode 变慢。
3. 优先 decode 保证输出顺滑，但新请求排队更久。
4. 提高并发可以摊薄成本，但 KV Cache 显存压力上升。
5. 限制上下文长度降低成本，但影响业务能力。
6. 量化降低显存和成本，但可能影响质量或引入兼容问题。
7. prefix cache 提升重复 prompt 性能，但需要额外 cache 管理和失效策略。

所以性能调优不是单指标优化，而是多目标优化。

你需要先定义目标：

1. 是聊天助手，重视 TTFT 和输出平滑。
2. 是离线批处理，重视总 tokens/s 和成本。
3. 是代码生成，可能重视长输出 TPOT 和稳定性。
4. 是 RAG 问答，可能重视长 prompt prefill 和 prefix cache。
5. 是 agent 场景，可能重视多轮短请求和工具调用延迟。

场景不同，调优方向不同。

## 5.11 p50、p95、p99 为什么重要

平均值经常掩盖问题。

例如平均 TTFT 是 800ms，但 p99 TTFT 是 15s，说明有少量用户体验非常差。

LLM serving 中长尾延迟很常见，原因包括：

1. prompt 长度分布长尾。
2. 输出长度分布长尾。
3. 某些请求等待 KV Cache。
4. scheduler 对长请求不公平。
5. GPU 上出现短时拥塞。
6. streaming 连接阻塞。
7. tokenizer 或预处理有异常慢样本。

因此线上指标至少要看：

1. p50 TTFT、p95 TTFT、p99 TTFT。
2. p50 TPOT、p95 TPOT、p99 TPOT。
3. p95 total latency。
4. p99 queue waiting time。
5. p99 prefill latency。
6. p99 decode iteration latency。

面向生产，p99 往往比平均值更能反映系统质量。

## 5.12 如何做压测

LLM serving 压测不能只发固定短 prompt。

合理压测应该定义 workload：

1. 输入 token 长度分布。
2. 输出 token 长度分布。
3. 请求到达模式，比如恒定 QPS、突发流量、泊松到达。
4. 是否 streaming。
5. 并发上限。
6. 采样参数。
7. 是否有共享 prefix。
8. SLO 阈值。

压测输出要包含：

1. TTFT 分位数。
2. TPOT 分位数。
3. tokens/s。
4. active sequences。
5. waiting queue 长度。
6. KV Cache 使用率。
7. GPU 利用率和显存。
8. 错误率和超时率。
9. 每百万 token 成本估算。

如果压测报告只给 QPS 和平均延迟，对 LLM serving 来说是不够的。

## 5.13 面试中如何讲性能指标

面试官问“如何评估一个大模型推理服务性能”，不要只答 QPS。

推荐回答结构：

1. 先说用户侧指标：TTFT、TPOT、总延迟、流式平滑度。
2. 再说系统侧指标：input/output tokens/s、active sequences、queue length、GPU 利用率、KV Cache 使用率。
3. 再说稳定性指标：p95/p99、错误率、超时率、OOM 次数。
4. 最后说成本指标：每百万 token 成本、每 GPU tokens/s、SLO 下的资源利用率。

这样回答能体现你理解 LLM serving 的特殊性，而不是把它当普通 HTTP 服务。

## 5.14 常见误区

误区一：QPS 越高，系统越好。

如果请求 token 很短，QPS 高不代表能处理长 prompt；如果 p99 延迟很差，QPS 高也没有意义。

误区二：GPU 利用率越高越好。

GPU 利用率高可能伴随严重排队，用户体验很差。要结合 TTFT、TPOT 和 SLO 看。

误区三：吞吐和延迟可以同时无限优化。

通常不可能。大 batch 提升吞吐，往往增加等待；低延迟策略可能牺牲吞吐和成本。

误区四：显存只由模型大小决定。

KV Cache、临时 buffer、并发和上下文长度都会影响显存。

误区五：平均延迟足够说明问题。

LLM 请求长度分布长尾明显，必须看 p95 和 p99。

## 5.15 面试追问

1. TTFT 和 TPOT 分别是什么？
2. 为什么 LLM serving 不能只看 QPS？
3. input tokens/s 和 output tokens/s 有什么区别？
4. 为什么 KV Cache 会限制最大并发？
5. 如何估算每 token 推理成本？
6. 大 batch 对吞吐和延迟分别有什么影响？
7. 线上 TTFT p99 突然升高，你会怎么排查？
8. 如果 GPU 利用率很高但用户体验差，可能是什么原因？

参考回答思路：

1. 先定义指标：TTFT、TPOT、tokens/s、并发、显存、成本。
2. 再说明 LLM 请求长度不固定，必须按 token 和阶段分析。
3. 然后解释 trade-off：吞吐、延迟、显存和成本互相制约。
4. 最后给排查路径：queue、prefill、decode、KV Cache、GPU、streaming、p95/p99。

## 5.16 小练习

1. 设计一个压测 workload，包含输入长度、输出长度、QPS、并发和是否 streaming。
2. 假设某服务 TTFT 高但 TPOT 正常，你会优先排查哪些阶段？
3. 假设 TPOT 高但 TTFT 正常，你会优先排查哪些阶段？
4. 解释为什么 batch size 增大可能降低每 token 成本但增加首 token 延迟。
5. 画一张图，把 TTFT、TPOT、tokens/s、KV Cache 使用率和成本联系起来。

## 5.17 本章小结

本章把 LLM serving 的核心机制转成了可量化指标。

TTFT 衡量首 token 等待，主要受排队和 prefill 影响；TPOT 衡量持续生成速度，主要受 decode、KV Cache 和调度影响；tokens/s 比 QPS 更能反映真实吞吐；并发会放大 KV Cache 和调度压力；显存不只由模型权重决定，还受 KV Cache 和临时 buffer 影响；成本优化要在满足 SLO 的前提下提高有效 token 吞吐。

下一章我们会回到系统边界，讲清推理框架、推理平台和 AI Infra 的区别，避免把 engine、平台治理和基础设施混在一起。
