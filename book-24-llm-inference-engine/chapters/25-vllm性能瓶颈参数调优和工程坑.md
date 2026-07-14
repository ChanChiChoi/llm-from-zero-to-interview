# 第 25 章 vLLM 性能瓶颈、参数调优和工程坑

上一章讲了 Tensor Parallel、Pipeline Parallel 与 serving 并行：TP、PP、DP、EP 如何影响 worker 数量、显存分布、通信开销、吞吐和延迟。

本章做 vLLM 部分的阶段性收尾：当线上 vLLM-like 服务 TTFT 高、TPOT 抖动、吞吐低、显存爆、prefix cache 没效果、preemption 频繁、GPU 利用率低时，应该怎么拆指标、怎么判断瓶颈、怎么调参数？

一句话概括：

> vLLM 调优不是盲目把 batch、并发和 GPU 数开大，而是把请求生命周期拆成 queue、tokenize、prefill、decode、KV cache、scheduler、worker、streaming 和 network，再针对瓶颈调整 token budget、KV budget、并行度、缓存和限流策略。

## 25.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考五类公开资料：

1. vLLM Optimization and Tuning 文档对 KV cache 空间不足、preemption、`gpu_memory_utilization`、`max_num_batched_tokens`、`max_num_seqs`、chunked prefill、decode / prefill 平衡和 attention backend 的说明。
2. vLLM metrics 文档对 TTFT、TPOT / inter-token latency、E2E latency、queue / prefill / decode 时间、running / waiting / swapped requests、KV cache usage、prefix cache hit rate 和 Prometheus / logging 指标的说明。
3. vLLM Automatic Prefix Caching 文档对 prefix cache hit、saved prefill、cached-free blocks、cache salt、LoRA、多模态 hash 和 eviction 的说明。
4. vLLM Parallelism and Scaling / Data Parallel / Expert Parallel 文档对 TP / PP / DP / EP 拓扑、跨节点通信、每个 DP rank 独立 KV cache 和 MoE expert parallel 的说明。
5. vLLM multimodal / production 相关文档对多模态输入 profile、processor cache、API server、streaming、部署与指标观测的公开口径。

本章只讲 vLLM-like serving 的教学版性能调优方法，不给出某个 GPU、某个模型、某个版本的通用最优参数，不替代真实压测平台、线上 SLO、NCCL / NUMA 排障、云成本模型、Kubernetes 编排、生产安全审计或业务质量评估。本章 demo 用纯 Python trace 表模拟 TTFT、TPOT、KV pressure、preemption、prefix cache locality、CPU / streaming 瓶颈和调参建议，不等同于真实性能预测。

参考资料：

1. vLLM Optimization and Tuning：<https://docs.vllm.ai/en/latest/configuration/optimization/>
2. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics/>
3. vLLM Automatic Prefix Caching：<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>
4. vLLM Parallelism and Scaling：<https://docs.vllm.ai/en/latest/serving/parallelism_scaling/>
5. vLLM Data Parallel Deployment：<https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/>
6. vLLM Expert Parallel Deployment：<https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/>

## 25.1 本章目标

读完本章，你应该能讲清：

1. vLLM-like 服务性能问题应该按哪些指标拆解。
2. TTFT 高、TPOT 高、吞吐低、显存 OOM 分别怎么排查。
3. `max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`max_model_len` 分别影响什么。
4. chunked prefill、prefix cache、preemption、TP/PP/DP 分别如何参与调优。
5. 为什么 GPU 利用率低不一定是模型 kernel 慢。
6. 线上常见工程坑和回滚策略。
7. 面试中如何给出系统化的 vLLM 调优方案。

## 25.2 调优前先拒绝一个误区

常见错误是：

```text
服务慢 -> 调大 batch -> 加 GPU -> 换量化 -> 换框架
```

这不是工程调优，而是碰运气。

正确顺序是：

```text
定义指标 -> 拆请求生命周期 -> 定位瓶颈 -> 改一个参数 -> 压测对比 -> 灰度上线 -> 继续观测
```

调优要回答的不是“哪个参数更快”，而是：

1. 当前瓶颈在哪里？
2. 调这个参数会改善哪个指标？
3. 会牺牲哪个指标？
4. 对 p95/p99 是否有副作用？
5. 是否对不同请求长度分布都有效？

## 25.3 指标总览：不要只看 QPS

vLLM-like 服务至少要看这些指标。

延迟类：

1. TTFT。
2. TPOT 或 ITL。
3. E2E latency。
4. queue time。
5. prefill time。
6. decode step time。
7. streaming flush time。

吞吐类：

1. input tokens/s。
2. output tokens/s。
3. total tokens/s。
4. requests/s。
5. successful requests/s。

调度类：

1. waiting queue length。
2. running requests。
3. scheduled prefill tokens。
4. scheduled decode tokens。
5. running batch size。
6. preempted requests。

显存类：

1. total KV blocks。
2. free KV blocks。
3. used KV blocks。
4. cached-but-free blocks。
5. allocation failures。
6. GPU memory reserved/used。

缓存类：

1. prefix cache hit rate。
2. hit prefix length。
3. saved prefill tokens。
4. eviction count。
5. cache hit by tenant/model/route。

系统类：

1. GPU utilization。
2. GPU memory bandwidth。
3. CPU utilization。
4. tokenizer latency。
5. network send latency。
6. NCCL communication time。
7. error/timeout/abort rate。

如果只有平均 QPS 和平均 latency，你几乎无法判断真正瓶颈。

## 25.4 把请求生命周期拆开

一次请求可以拆成：

```text
client send
  -> gateway / API server
  -> auth / rate limit / validation
  -> tokenizer / chat template / multimodal processing
  -> waiting queue
  -> scheduler admission
  -> prefill
  -> first token sampling
  -> decode loop
  -> detokenization
  -> streaming flush
  -> finish / cleanup
```

对应延迟：

```text
total_latency = frontend_time
              + tokenize_time
              + queue_time
              + prefill_time
              + decode_time
              + output_processing_time
              + network_time
```

TTFT 主要由这些部分组成：

```text
TTFT = frontend_time + tokenize_time + queue_time + prefill_time + first_token_output_time
```

TPOT 主要由这些部分组成：

```text
TPOT = decode_step_time + sampling_time + detokenize_time + streaming_flush_time
```

排查时要先判断慢在哪里，而不是直接看 GPU。

## 25.5 TTFT 高怎么排查

TTFT 高表示用户等第一个 token 很久。

常见原因：

1. waiting queue 堆积。
2. prompt 太长，prefill 重。
3. tokenizer 或 chat template 慢。
4. 多模态预处理慢。
5. scheduler 过度优先 decode，prefill 饥饿。
6. prefix cache 命中率低。
7. `max_num_batched_tokens` 太小，长 prompt 被切太碎。
8. KV blocks 不足，新请求进不来。
9. RAG、工具、安全审核等前置链路慢。

排查顺序：

1. 看 queue time 是否升高。
2. 看 input token 分布是否变长。
3. 看 prefill tokens/s 是否下降。
4. 看 prefix cache hit length 是否下降。
5. 看 waiting 请求是否因为 free blocks 不足无法调度。
6. 看 tokenizer CPU 和 API server CPU 是否满。
7. 看长 prompt 是否被 chunked prefill 多轮处理。

对应调优：

1. 提高 `max_num_batched_tokens`，让更多 prefill tokens 能进入一轮。
2. 启用或调优 prefix cache，减少重复 prefill。
3. 对长 prompt 使用 chunked prefill，避免大尖刺。
4. 增加 API server 或 tokenizer 处理能力。
5. 长短请求分队列或分实例。
6. 对 RAG 输入做 rerank、压缩和上下文预算控制。

注意：提高 `max_num_batched_tokens` 可能改善 TTFT，但也可能让 decode TPOT 抖动，所以必须同时看 streaming 体验。

## 25.6 TPOT 高怎么排查

TPOT 高表示后续 token 输出慢，用户看到“打字很慢”。

常见原因：

1. decode batch 太大。
2. 每轮混入太多长 prefill，阻塞 decode。
3. KV Cache 读取效率差。
4. 上下文过长，attention 读 KV 成本高。
5. TP 通信开销过大。
6. CPU scheduler loop 或 sampling 慢。
7. detokenization 或 streaming flush 慢。
8. speculative decoding 接受率低，额外开销抵消收益。
9. 量化 kernel 不适合当前 batch 或硬件。

排查顺序：

1. 区分 engine TPOT 和 client-observed TPOT。
2. 看 decode step time 是否增加。
3. 看每轮是否混入大 prefill。
4. 看 running batch size 和上下文长度分布。
5. 看 GPU memory bandwidth 和 KV cache block metrics。
6. 看 NCCL 通信时间，尤其 TP 跨节点或 TP 太大时。
7. 看 detokenizer、HTTP SSE、网关和客户端接收时间。

对应调优：

1. 降低 `max_num_batched_tokens`，减少 prefill 对 decode 的阻塞。
2. 使用 decode-first 或更严格的 chunked prefill 策略。
3. 降低 `max_num_seqs`，减少每轮 decode 压力。
4. 优化 TP 拓扑，把 TP group 放在 NVLink/NVSwitch 内。
5. 检查量化 kernel 和 attention backend 是否合适。
6. 对超长上下文请求限流或单独实例。

## 25.7 吞吐低怎么排查

吞吐低不一定等于延迟高。

常见现象：

1. GPU utilization 低。
2. output tokens/s 低。
3. QPS 上不去。
4. waiting queue 很短但 GPU 不满。
5. batch size 总是很小。

可能原因：

1. 流量不足，batch 填不满。
2. scheduler 没有有效 continuous batching。
3. `max_num_seqs` 太小。
4. `max_num_batched_tokens` 太小。
5. CPU 或 tokenizer 跟不上。
6. API server 成为瓶颈。
7. worker kernel launch 或 model runner 准备 metadata 慢。
8. TP/PP 通信让 GPU 等待。
9. 请求输出太短，decode batch 难以维持。

对应调优：

1. 提高 `max_num_seqs`，让更多 running 请求同时 decode。
2. 提高 `max_num_batched_tokens`，增加每轮 token 工作量。
3. 增加 API server count 或输入处理并行度。
4. 使用 DP 扩展副本吞吐。
5. 合理开启 CUDA graph/compile 优化。
6. 对流量做聚合、批处理或离线 batch serving。

但要注意：吞吐提升可能以 p99 延迟恶化为代价。

## 25.8 显存 OOM 怎么排查

vLLM-like 服务的显存通常包括：

```text
模型权重 + KV Cache + activation/workspace + CUDA graph + communication buffers + runtime overhead
```

常见 OOM 原因：

1. 模型权重本身太大。
2. `max_model_len` 太大，KV cache profile 预算过高。
3. `max_num_seqs` 太大，同时活跃请求过多。
4. 长 prompt 或长输出太多。
5. prefix cache 保留过多但命中低。
6. preemption/recompute 频繁。
7. CUDA graph capture sizes 占用额外显存。
8. 多模态输入 profile 预留过大。
9. 取消、超时、失败路径没有释放 KV blocks。

对应调优：

1. 降低 `max_model_len`。
2. 降低 `max_num_seqs`。
3. 降低 `max_num_batched_tokens`。
4. 限制 max output tokens。
5. 提高 `gpu_memory_utilization`，但要保留安全余量。
6. 使用 TP/PP 分摊权重显存。
7. 使用量化降低权重显存。
8. 调整 CUDA graph capture sizes 或启用 eager 模式换显存。
9. 限制多模态输入数量和尺寸。
10. 检查 free KV blocks 是否在请求结束后恢复。

不要只看 `nvidia-smi`。vLLM 可能预分配 KV cache pool，显存不回落不一定是泄漏；要看 free blocks、used blocks、allocation failures 和 active requests。

## 25.9 preemption 频繁意味着什么

vLLM 优化文档中提到，当 KV cache 空间不足以处理所有 batched requests 时，系统可能 preempt 某些请求，后续 recompute。

preemption 的意义是保护系统不直接 OOM。

但频繁 preemption 是坏信号。

它通常说明：

1. KV cache budget 不够。
2. 并发设置过高。
3. 请求上下文过长。
4. `max_num_batched_tokens` 或 `max_num_seqs` 太激进。
5. prefix cache 占用过多。
6. TP/PP 设置导致每卡可用 KV 空间不足。

优化方向：

1. 提高 `gpu_memory_utilization`。
2. 降低 `max_num_seqs`。
3. 降低 `max_num_batched_tokens`。
4. 缩短 `max_model_len`。
5. 增加 TP 或 PP，让权重占用下降。
6. 对长上下文请求限流。
7. 排查 prefix cache 命中率和 eviction。

面试里要说清楚：preemption 是鲁棒性机制，不是性能优化。频繁 preemption 会伤害 E2E latency。

## 25.10 `max_num_batched_tokens` 怎么调

`max_num_batched_tokens` 控制每个 engine step 最多处理多少 tokens。

它同时影响 prefill 和 decode。

调大：

1. 更容易接纳长 prefill。
2. TTFT 可能变好。
3. input tokens/s 可能提升。
4. GPU 利用率可能提升。
5. 但单轮执行时间变长，decode TPOT 可能抖动。

调小：

1. 单轮延迟更稳定。
2. decode 更平滑。
3. 长 prompt 可能被 chunk 成更多轮。
4. TTFT 可能变差。
5. 吞吐可能下降。

经验判断：

1. 低延迟聊天：偏小，优先稳定 TPOT。
2. 长文档 RAG：适中，配合 prefix cache 和 chunked prefill。
3. 离线 batch 生成：偏大，优先吞吐。

vLLM 文档中也提到，chunked prefill 场景下调小 `max_num_batched_tokens` 可以改善 inter-token latency，调大则可能改善 TTFT 和吞吐。

## 25.11 `max_num_seqs` 怎么调

`max_num_seqs` 控制同时参与调度的最大 sequence 数。

调大：

1. 并发能力更高。
2. decode batch 更大。
3. GPU 利用率可能提升。
4. KV Cache 压力更大。
5. scheduler/sampling/output processing 开销更大。

调小：

1. KV Cache 压力下降。
2. p99 可能更稳定。
3. 但吞吐上限下降。
4. 高峰时 queue time 可能升高。

如果 OOM 或 preemption 频繁，`max_num_seqs` 是最直接的保护阀之一。

如果 GPU 很空、queue 很短但 batch 很小，适当提高它可能有效。

## 25.12 `gpu_memory_utilization` 怎么调

`gpu_memory_utilization` 控制 vLLM 可使用 GPU 显存的比例，通常影响 KV cache 预留空间。

调大：

1. KV Cache 空间增加。
2. 可支持更多并发或更长上下文。
3. preemption 可能减少。
4. 但留给 CUDA graph、临时 buffer、通信和系统波动的余量减少。

调小：

1. 安全余量增加。
2. OOM 风险可能下降。
3. 但可用 KV blocks 减少，并发能力下降。

不要盲目设成接近 1。生产中要考虑：

1. 模型加载波动。
2. CUDA graph 额外内存。
3. NCCL buffer。
4. 多模态临时 tensor。
5. 其他进程占用。

## 25.13 `max_model_len` 和上下文预算

`max_model_len` 决定模型允许的最大上下文长度，也会影响 memory profiling 和容量估算。

设置过大：

1. KV Cache 预算压力增加。
2. 最大并发估算下降。
3. 长请求可能拖慢调度。
4. 用户可能无意中提交超长 prompt。

设置过小：

1. 长文档任务无法完成。
2. RAG 召回内容被截断。
3. 多轮对话上下文不足。

正确做法是按业务分层：

1. 普通聊天实例：较短上下文。
2. 长文档 RAG 实例：长上下文，单独限流。
3. 离线分析实例：更长上下文，弱化实时 SLA。

不要让所有业务共用一个最大上下文实例。

## 25.14 Chunked prefill 怎么看

Chunked prefill 把长 prompt prefill 拆成多个小块，和 decode 混合调度。

收益：

1. 长 prompt 不会一次性霸占 GPU。
2. decode 请求 TPOT 更稳定。
3. 更容易填充 token budget。
4. 适合长短请求混合负载。

代价：

1. 某些请求完成 prefill 需要多轮。
2. TTFT 可能受 chunk 策略影响。
3. scheduler 状态更复杂。

观察指标：

1. chunked prefill 次数。
2. 每个请求 prefill chunks 数。
3. decode step time 是否更稳定。
4. TTFT 是否恶化。
5. input tokens/s 是否提升。

调优核心是平衡 TTFT 和 TPOT。

## 25.15 Prefix cache 怎么调

Prefix cache 的目标是减少重复前缀 prefill。

要看这些指标：

1. hit rate。
2. hit prefix length。
3. saved prefill tokens。
4. eviction count。
5. cached-but-free blocks。
6. cache hit by tenant/model/route。
7. TTFT by hit length bucket。

命中率低的常见原因：

1. prompt 前缀实际不一致。
2. chat template 插入动态字段。
3. RAG chunk 顺序不稳定。
4. cache salt 粒度过细。
5. 请求被 DP 路由打散。
6. cache 容量太小，频繁 eviction。
7. LoRA 或多模态 hash 不一致。

调优方式：

1. 稳定 system prompt 和工具说明。
2. 固定 RAG 文档排序和版本。
3. 对相同会话或租户做 cache-aware routing。
4. 避免不必要动态字段进入 prefix。
5. 监控 prefix cache 占用和收益，命中低时考虑关闭或隔离。

## 25.16 并行策略怎么参与调优

TP：

1. 解决单卡放不下和权重显存压力。
2. 可给 KV cache 腾空间。
3. 但增加通信，可能伤害 TPOT。
4. 优先放在同机高速互联内。

PP：

1. 解决单节点放不下或 TP 跨节点太贵。
2. 但可能增加单请求延迟和 pipeline bubble。

DP：

1. 解决总吞吐不足。
2. 每个 rank KV cache 独立。
3. 路由会影响 prefix cache locality 和 p99。

EP：

1. 解决 MoE expert 权重和计算分布。
2. 但要关注 expert 负载不均和 routing 通信。

调优原则：

```text
单副本放不下 -> TP/PP
单副本能跑但 QPS 不够 -> DP
MoE experts 成为瓶颈 -> EP
TPOT 被通信拖慢 -> 降低跨节点 TP 或调整拓扑
```

## 25.17 CPU 瓶颈和 NUMA 问题

vLLM V1 多进程架构下，至少有 API server、engine core 和每个 GPU worker 需要 CPU。

CPU 不足会导致：

1. tokenizer 慢。
2. chat template 慢。
3. 多模态加载慢。
4. scheduler loop 慢。
5. kernel launch 间隔变大。
6. detokenization 慢。
7. streaming flush 慢。

典型现象：

```text
GPU utilization 不高，但 queue time 和 TPOT 都在抖。
```

排查：

1. 看 API server CPU。
2. 看 engine core CPU。
3. 看 worker CPU。
4. 看 tokenization latency。
5. 看 scheduler step interval。
6. 看 NUMA 绑定是否合理。

vLLM 文档中提到，多 socket GPU 机器上 worker 如果跑在离 GPU 远的 NUMA node 上，性能会下降。可以通过 NUMA binding 让 worker CPU 和内存靠近对应 GPU。

## 25.18 Streaming 和输出链路坑

模型生成 token 不代表用户收到 token。

Streaming 链路包括：

```text
sample token -> detokenize -> output processor -> API server -> gateway/proxy -> client
```

常见坑：

1. 网关缓冲 SSE。
2. gzip 或代理合并 chunk。
3. 客户端渲染太频繁。
4. detokenizer 慢。
5. logprobs 或结构化输出处理慢。
6. 网络 backpressure。
7. 客户端断开没有及时 abort。

排查方式：

1. 记录 model token time。
2. 记录 server send time。
3. 记录 client receive time。
4. 三段时间对齐。

如果 model token time 正常但 client receive time 抖动，继续优化 model runner 没意义。

## 25.19 多模态 serving 的特殊调优

多模态模型多了几类资源：

1. media loading CPU。
2. processor cache。
3. encoder cache。
4. vision/audio encoder activation。
5. placeholder tokens。
6. 多模态 embedding 注入。

常见问题：

1. 图片或视频过多导致预处理慢。
2. dummy profile 预留过大导致显存浪费。
3. processor cache 占用 CPU memory。
4. 多 API server 下 IPC cache 效果变化。
5. 多模态 hash 不一致导致 prefix cache 失效。

调优方式：

1. 限制每个 prompt 的图片、视频、音频数量。
2. 配置合理的多模态尺寸 hint。
3. 调整 processor cache 大小。
4. 对文本-only 请求使用文本-only 实例。
5. 分开部署多模态和纯文本流量。

## 25.20 量化不是只看显存

量化可以降低权重显存，但不一定提升速度。

原因：

1. kernel 不一定更快。
2. dequant 开销可能抵消收益。
3. 小 batch 下通信或调度仍是瓶颈。
4. KV Cache 没有同步变小。
5. 质量退化可能带来重试和人工修正成本。

上线量化前要测：

1. 业务真实 prompt。
2. 长上下文。
3. JSON/工具调用/结构化输出。
4. 代码和数学。
5. 安全边界。
6. TTFT、TPOT、tokens/s、显存和 p99。

量化目标是单位质量成本更低，而不是单看显存数字更小。

## 25.21 调优实验应该怎么做

每次只改一个主要变量。

推荐实验矩阵：

1. baseline。
2. 改 `max_num_batched_tokens`。
3. 改 `max_num_seqs`。
4. 改 `gpu_memory_utilization`。
5. 开关 prefix cache。
6. 开关或调整 chunked prefill。
7. 改 TP/DP 配置。
8. 改 quantization 或 attention backend。

每组都记录：

1. TTFT p50/p95/p99。
2. TPOT p50/p95/p99。
3. input/output tokens/s。
4. queue time。
5. GPU utilization。
6. free KV blocks。
7. preemption count。
8. prefix cache hit saved tokens。
9. error/timeout rate。

压测流量必须覆盖真实长度分布：短 prompt、长 prompt、短输出、长输出、RAG、多轮对话、峰值并发。

## 25.22 常见线上事故和处理

事故一：突然 TTFT 飙升。

处理：先看 queue、route、API server CPU、input length、prefix cache hit、RAG 前置耗时。不要先换 kernel。

事故二：TPOT 抖动，用户看到输出卡顿。

处理：看 decode step time、chunked prefill、running batch size、NCCL、streaming send/receive。必要时降低 `max_num_batched_tokens`。

事故三：频繁 OOM 或 allocation failure。

处理：限流长请求，降低 `max_num_seqs`/`max_model_len`，检查 KV free blocks，确认 abort cleanup，临时降级 max output tokens。

事故四：preemption 日志大量出现。

处理：降低并发和 token budget，提高 KV cache 空间，隔离长上下文流量，评估 TP/PP。

事故五：prefix cache 命中突然下降。

处理：检查 chat template、system prompt 版本、RAG 排序、cache salt、DP 路由和 LoRA/multimodal hash。

事故六：GPU 利用率低但排队严重。

处理：检查 CPU、tokenizer、engine core loop、API server、worker hang、调度是否被 KV blocks 卡住。

## 25.23 上线调优的安全策略

不要直接全量调参。

建议：

1. 先离线 replay 真实流量。
2. 再小流量灰度。
3. 分业务线、分 prompt 长度观察。
4. 设置 p99、OOM、error rate 自动回滚阈值。
5. 保留旧配置和旧模型副本。
6. 监控质量指标，不只监控性能。
7. 对长上下文和高价值租户单独保护。

调优可能改善平均吞吐，但伤害尾延迟或质量。生产系统要以 SLA 和成本综合判断。

## 25.24 面试官会怎么问

问题一：vLLM TTFT 高怎么排查？

回答要点：拆 queue、tokenizer、prefill、prefix cache、KV block admission、前置 RAG/tool 链路。看 input length、prefill tokens/s、waiting queue 和 cache hit length。

问题二：TPOT 高怎么排查？

回答要点：看 decode step time、running batch、长 prefill 是否阻塞 decode、KV cache 读取、TP 通信、detokenization 和 streaming 链路。

问题三：`max_num_batched_tokens` 调大有什么风险？

回答要点：可能提升 prefill 吞吐和 TTFT，但单轮执行时间变长，decode ITL/TPOT 和 p99 可能变差。

问题四：preemption 频繁说明什么？

回答要点：通常说明 KV cache 空间不足或并发/token budget 太激进。可以调高 `gpu_memory_utilization`，降低 `max_num_seqs`/`max_num_batched_tokens`，缩短上下文或增加 TP/PP。

问题五：GPU 利用率低一定是模型太小吗？

回答要点：不一定。可能是 CPU/tokenizer/API server/scheduler 慢，waiting queue 被 KV block 卡住，流量不足，batch 组不起来，或者 streaming/network 阻塞。

问题六：怎么做 vLLM 调优实验？

回答要点：固定真实流量分布，每次只改一个关键参数，同时记录 TTFT/TPOT/tokens/s/queue/free blocks/preemption/cache hit/error rate，并看 p95/p99，不只看平均吞吐。

## 25.25 标准回答模板

如果面试官问“你会如何调优 vLLM serving”，可以这样回答：

```text
我不会先盲目调大 batch 或加 GPU，而是先把请求生命周期拆开观测。核心指标包括 TTFT、TPOT、E2E latency、queue time、prefill tokens/s、decode tokens/s、running batch size、KV block 使用、preemption、prefix cache hit、GPU/CPU 利用率和错误率。

如果 TTFT 高，我会先看 queue time、input token 分布、prefill 性能、prefix cache 命中、tokenizer 和 RAG/tool 前置链路。调优可能包括提高 max_num_batched_tokens、优化 prefix cache、启用 chunked prefill、增加 API server 或隔离长 prompt。

如果 TPOT 高，我会看 decode step time、running batch size、长 prefill 是否阻塞 decode、KV cache 读取、TP 通信和 streaming 链路。调优可能是降低 max_num_batched_tokens、降低 max_num_seqs、优化 TP 拓扑、调整 attention backend 或减少超长上下文混跑。

如果 OOM 或 preemption 频繁，我会看 free KV blocks、allocation failures、max_model_len、max_num_seqs、prefix cache 占用和请求长度分布。处理方式包括降低并发和上下文预算、提高 gpu_memory_utilization、使用 TP/PP 分摊权重、限制长请求和检查 abort cleanup。

最后所有调参都要基于真实流量 replay 和灰度，比较 p50/p95/p99、tokens/s、错误率和成本，而不是只看平均 QPS。
```

## 25.26 vLLM 调优公式、事故归因和可运行 demo

把调优需要的观测量先写成稳定公式。对请求 `i`：

$$
T_{\mathrm{ttft},i}=T_{\mathrm{front},i}+T_{\mathrm{tok},i}+T_{\mathrm{queue},i}+T_{\mathrm{prefill},i}+T_{\mathrm{first},i}
$$

$$
T_{\mathrm{tpot},i}=\frac{T_{\mathrm{decode},i}+T_{\mathrm{sample},i}+T_{\mathrm{detok},i}+T_{\mathrm{flush},i}}{\max(1,Y_i)}
$$

其中 `Y_i` 是输出 token 数。压测窗口内的 token 吞吐可以写成：

$$
Q_{\mathrm{out}}=\frac{\sum_i Y_i}{T_{\mathrm{window}}}
$$

KV pressure 可以用 used blocks 和 total blocks 表示：

$$
P_{\mathrm{kv}}=\frac{B_{\mathrm{used}}}{B_{\mathrm{total}}}
$$

preemption rate 可以写成：

$$
R_{\mathrm{preempt}}=\frac{N_{\mathrm{preempt}}}{N_{\mathrm{req}}}
$$

prefix cache 的 saved prefill ratio 可以写成：

$$
R_{\mathrm{save}}=\frac{\sum_i S_i}{\sum_i X_i}
$$

其中 `S_i` 是 prefix cache 省掉的 prefill tokens，`X_i` 是输入 tokens。

教学版 vLLM 调优门禁可以写成：

$$
G_{\mathrm{tune}}=G_{\mathrm{metric}}G_{\mathrm{ttft}}G_{\mathrm{tpot}}G_{\mathrm{kv}}G_{\mathrm{cache}}G_{\mathrm{config}}G_{\mathrm{rollback}}
$$

其中：

1. `G_{\mathrm{metric}}`：TTFT、TPOT、queue、prefill、decode、KV、cache、CPU、streaming 指标完整。
2. `G_{\mathrm{ttft}}`：能区分 queue、tokenize、prefill、prefix cache miss 和前置链路导致的 TTFT。
3. `G_{\mathrm{tpot}}`：能区分 decode、KV 读取、TP 通信、chunked prefill、sampling 和 streaming 导致的 TPOT。
4. `G_{\mathrm{kv}}`：能识别 KV pressure、allocation failure、preemption 和 cleanup 问题。
5. `G_{\mathrm{cache}}`：能解释 prefix cache hit rate、saved tokens、DP route locality 和 eviction。
6. `G_{\mathrm{config}}`：调参方案说明 `max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`max_model_len`、TP / DP 的 trade-off。
7. `G_{\mathrm{rollback}}`：灰度、p99、OOM、error rate 和质量回滚阈值明确。

下面这个 0 依赖 demo 构造 5 条 toy request trace 和一个当前配置，模拟一次 vLLM 性能事故审计。它故意让 TTFT、TPOT、KV pressure、prefix cache 和 streaming 都有可见问题，目标不是让服务通过 SLO，而是证明审计能定位瓶颈并给出有 trade-off 的调参建议。

```python
from dataclasses import dataclass


@dataclass
class RequestTrace:
    request_id: str
    input_tokens: int
    output_tokens: int
    front_ms: int
    tokenize_ms: int
    queue_ms: int
    prefill_ms: int
    first_ms: int
    decode_ms: int
    sample_ms: int
    detok_ms: int
    flush_ms: int
    kv_blocks: int
    prefix_lookups: int
    prefix_hits: int
    saved_prefill_tokens: int
    preemptions: int
    dp_rank: int
    route_key: str
    error: bool = False


@dataclass
class ServingConfig:
    name: str
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    data_parallel_size: int
    chunked_prefill: bool
    prefix_cache: bool
    sticky_routing: bool
    tp_cross_node: bool


class ToyVLLMTuningAuditor:
    def __init__(self, traces, config, total_kv_blocks, ttft_budget_ms, tpot_budget_ms):
        self.traces = traces
        self.config = config
        self.total_kv_blocks = total_kv_blocks
        self.ttft_budget_ms = ttft_budget_ms
        self.tpot_budget_ms = tpot_budget_ms

    def percentile(self, values, ratio):
        ordered = sorted(values)
        index = int((len(ordered) - 1) * ratio + 0.999999)
        return ordered[min(index, len(ordered) - 1)]

    def per_request_metrics(self, trace):
        ttft = trace.front_ms + trace.tokenize_ms + trace.queue_ms + trace.prefill_ms + trace.first_ms
        tpot = (
            trace.decode_ms + trace.sample_ms + trace.detok_ms + trace.flush_ms
        ) / max(1, trace.output_tokens)
        e2e = ttft + trace.decode_ms + trace.sample_ms + trace.detok_ms + trace.flush_ms
        return {
            "request_id": trace.request_id,
            "ttft_ms": round(ttft, 1),
            "tpot_ms": round(tpot, 2),
            "e2e_ms": round(e2e, 1),
            "queue_ms": trace.queue_ms,
            "tokenize_ms": trace.tokenize_ms,
            "prefill_ms": trace.prefill_ms,
            "kv_blocks": trace.kv_blocks,
            "preemptions": trace.preemptions,
            "prefix_hits": trace.prefix_hits,
            "prefix_lookups": trace.prefix_lookups,
            "saved_prefill_tokens": trace.saved_prefill_tokens,
        }

    def summarize(self):
        per_request = [self.per_request_metrics(trace) for trace in self.traces]
        ttfts = [item["ttft_ms"] for item in per_request]
        tpots = [item["tpot_ms"] for item in per_request]
        queues = [item["queue_ms"] for item in per_request]
        e2es = [item["e2e_ms"] for item in per_request]
        window_s = max(e2es) / 1000.0
        prefix_lookups = sum(trace.prefix_lookups for trace in self.traces)
        prefix_hits = sum(trace.prefix_hits for trace in self.traces)
        saved_tokens = sum(trace.saved_prefill_tokens for trace in self.traces)
        input_tokens = sum(trace.input_tokens for trace in self.traces)
        output_tokens = sum(trace.output_tokens for trace in self.traces)
        preemptions = sum(trace.preemptions for trace in self.traces)
        kv_used = sum(trace.kv_blocks for trace in self.traces)

        route_to_ranks = {}
        for trace in self.traces:
            route_to_ranks.setdefault(trace.route_key, set()).add(trace.dp_rank)
        scattered_routes = sorted(key for key, ranks in route_to_ranks.items() if len(ranks) > 1)

        return {
            "per_request": per_request,
            "summary": {
                "ttft_p95_ms": self.percentile(ttfts, 0.95),
                "tpot_p95_ms": self.percentile(tpots, 0.95),
                "queue_p95_ms": self.percentile(queues, 0.95),
                "output_tokens_per_s": round(output_tokens / window_s, 2),
                "kv_pressure": round(kv_used / self.total_kv_blocks, 3),
                "preemption_rate": round(preemptions / len(self.traces), 3),
                "prefix_hit_rate": round(prefix_hits / prefix_lookups, 3) if prefix_lookups else 0.0,
                "saved_prefill_ratio": round(saved_tokens / input_tokens, 3),
                "scattered_prefix_routes": scattered_routes,
                "error_rate": round(sum(1 for trace in self.traces if trace.error) / len(self.traces), 3),
            },
        }

    def diagnose(self, summary):
        reasons = []
        if summary["ttft_p95_ms"] > self.ttft_budget_ms and summary["queue_p95_ms"] > 500:
            reasons.append("queue_or_prefill_ttft")
        if summary["tpot_p95_ms"] > self.tpot_budget_ms:
            reasons.append("decode_or_streaming_tpot")
        if summary["kv_pressure"] > 0.75 or summary["preemption_rate"] > 0:
            reasons.append("kv_pressure_preemption")
        if summary["prefix_hit_rate"] < 0.25 and summary["saved_prefill_ratio"] < 0.05:
            reasons.append("prefix_cache_ineffective")
        if summary["scattered_prefix_routes"]:
            reasons.append("dp_route_breaks_cache_locality")
        if self.config.tp_cross_node:
            reasons.append("cross_node_tp_risk")
        if any(trace.tokenize_ms > 200 or trace.flush_ms > 180 for trace in self.traces):
            reasons.append("cpu_or_streaming_path")
        return reasons

    def recommendations(self, reasons):
        actions = []
        if "queue_or_prefill_ttft" in reasons:
            actions.append("raise_or_rebalance_max_num_batched_tokens_with_replay")
            actions.append("enable_chunked_prefill_for_long_prompts")
        if "decode_or_streaming_tpot" in reasons:
            actions.append("cap_prefill_per_step_or_lower_max_num_batched_tokens")
            actions.append("profile_detokenize_and_sse_flush")
        if "kv_pressure_preemption" in reasons:
            actions.append("lower_max_num_seqs_or_max_model_len")
            actions.append("increase_gpu_memory_utilization_with_oom_guard")
        if "prefix_cache_ineffective" in reasons:
            actions.append("stabilize_prompt_prefix_and_measure_saved_tokens")
        if "dp_route_breaks_cache_locality" in reasons:
            actions.append("enable_tenant_or_session_sticky_routing")
        if "cross_node_tp_risk" in reasons:
            actions.append("keep_tp_inside_node_or_use_tp_plus_pp")
        if "cpu_or_streaming_path" in reasons:
            actions.append("scale_api_tokenizer_workers_and_check_gateway_buffering")
        return actions

    def rollback_policy(self):
        return {
            "ttft_p99_ms": 3500,
            "tpot_p99_ms": 90,
            "oom_or_allocation_failures": 0,
            "error_rate": 0.01,
            "quality_regression": 0,
        }

    def audit(self):
        metrics = self.summarize()
        summary = metrics["summary"]
        reasons = self.diagnose(summary)
        actions = self.recommendations(reasons)
        rollback = self.rollback_policy()
        gates = {
            "metrics_complete": all(
                {"ttft_ms", "tpot_ms", "queue_ms", "kv_blocks"}.issubset(item)
                for item in metrics["per_request"]
            ),
            "ttft_bottleneck_detected": "queue_or_prefill_ttft" in reasons,
            "tpot_bottleneck_detected": "decode_or_streaming_tpot" in reasons,
            "kv_preemption_detected": "kv_pressure_preemption" in reasons,
            "cache_issue_detected": "prefix_cache_ineffective" in reasons and "dp_route_breaks_cache_locality" in reasons,
            "config_tradeoff_ready": (
                "enable_chunked_prefill_for_long_prompts" in actions
                and "lower_max_num_seqs_or_max_model_len" in actions
                and "keep_tp_inside_node_or_use_tp_plus_pp" in actions
            ),
            "rollback_guard_ready": all(value is not None for value in rollback.values()),
        }
        gates["vllm_tuning_gate"] = all(gates.values())
        return {
            "summary": summary,
            "root_causes": reasons,
            "recommendations": actions,
            "rollback_policy": rollback,
            "gates": gates,
        }


traces = [
    RequestTrace("short_chat", 80, 20, 20, 8, 40, 60, 10, 600, 30, 20, 40, 3, 1, 1, 64, 0, 0, "tenantA"),
    RequestTrace("long_rag_a", 5000, 60, 30, 70, 900, 1700, 20, 2400, 90, 80, 120, 18, 2, 0, 0, 1, 0, "manual_v7"),
    RequestTrace("long_rag_b", 4800, 32, 30, 60, 850, 1600, 20, 1400, 60, 50, 80, 16, 2, 0, 0, 1, 1, "manual_v7"),
    RequestTrace("long_decode", 500, 120, 20, 20, 150, 180, 15, 7200, 150, 120, 300, 14, 1, 0, 0, 0, 0, "tenantB"),
    RequestTrace("cpu_streaming", 300, 30, 25, 260, 80, 120, 15, 900, 60, 80, 220, 5, 1, 0, 0, 0, 1, "tenantC"),
]

config = ServingConfig(
    name="baseline_bad_mix",
    max_num_batched_tokens=4096,
    max_num_seqs=64,
    gpu_memory_utilization=0.84,
    max_model_len=32768,
    tensor_parallel_size=16,
    data_parallel_size=2,
    chunked_prefill=False,
    prefix_cache=True,
    sticky_routing=False,
    tp_cross_node=True,
)

auditor = ToyVLLMTuningAuditor(
    traces=traces,
    config=config,
    total_kv_blocks=70,
    ttft_budget_ms=1500,
    tpot_budget_ms=55,
)
result = auditor.audit()

print("vllm_tuning_summary=", result["summary"])
print("vllm_tuning_root_causes=", result["root_causes"])
print("vllm_tuning_recommendations=", result["recommendations"])
print("vllm_tuning_gates=", result["gates"])
```

这份输出应当体现：

1. `ttft_p95_ms=2720`，`queue_p95_ms=900`，说明 TTFT 问题主要落在 queue / prefill 一侧。
2. `tpot_p95_ms=64.75`，说明 decode、sampling、detokenize 或 streaming 链路也在拖慢输出。
3. `kv_pressure=0.8`、`preemption_rate=0.4`，说明 KV budget 和并发参数过激。
4. `prefix_hit_rate=0.143`、`saved_prefill_ratio=0.006`，且 `manual_v7` 路由到多个 DP rank，说明 prefix cache 没有形成有效收益。
5. `root_causes` 同时包含 `cross_node_tp_risk` 和 `cpu_or_streaming_path`，说明调优不能只改一个 batch 参数。
6. `recommendations` 同时覆盖 chunked prefill、`max_num_seqs` / `max_model_len`、`gpu_memory_utilization`、sticky routing、TP 拓扑和 API / tokenizer / gateway 链路。
7. `vllm_tuning_gate=True`，说明这份审计能把指标、瓶颈、配置 trade-off 和回滚阈值串起来。

## 25.27 小练习

1. 设计一个 vLLM dashboard，包含 TTFT、TPOT、queue、KV blocks、prefix cache、preemption、GPU/CPU 指标。
2. 给定现象“TTFT p99 飙升但 TPOT 正常”，列出 8 个排查方向。
3. 给定现象“TPOT 抖动但 TTFT 正常”，列出 8 个排查方向。
4. 设计一组实验比较 `max_num_batched_tokens=2048/4096/8192/16384` 的效果。
5. 解释为什么 prefix cache 命中率高但吞吐不一定提升。
6. 设计一个长上下文流量的限流和降级策略。

## 25.28 本章总结

vLLM 性能调优的核心是指标拆解和 trade-off。

TTFT 主要受 queue、tokenizer、prefill、prefix cache 和前置链路影响；TPOT 主要受 decode step、KV cache 读取、通信、调度和 streaming 链路影响；吞吐受 batch 填充、token budget、GPU/CPU 协同和并行策略影响；显存则由权重、KV cache、CUDA graph、workspace、prefix cache 和多模态缓存共同决定。

`max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`max_model_len`、chunked prefill、prefix cache、preemption、TP/PP/DP 都不是孤立参数。它们共同决定吞吐、TTFT、TPOT、显存和 p99。真正的工程能力，是能根据现象定位瓶颈，根据瓶颈选择参数，并用真实流量压测和灰度验证结果。

下一章开始进入 SGLang 架构详解，我们会对比 vLLM 的 scheduler/block manager 思路，理解 SGLang 为什么强调 RadixAttention、structured generation、runtime 和前端语言表达。
