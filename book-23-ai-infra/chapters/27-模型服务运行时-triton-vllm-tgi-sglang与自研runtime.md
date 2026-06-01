# 第 27 章 模型服务运行时：Triton、vLLM、TGI、SGLang 与自研 runtime

上一章讲了推理平台总览。本章聚焦一个更底层也更关键的问题：模型服务运行时，也就是 inference runtime。

推理平台负责把请求接进来、路由、限流、扩缩容、灰度和治理；runtime 负责真正把模型跑起来，把 token 生成出来。

先记住一句话：

> 推理平台管“服务化和治理”，模型服务运行时管“高性能执行”。

如果 runtime 选型错误，平台层做得再漂亮，也很难获得好的延迟、吞吐和成本。

## 27.1 什么是模型服务运行时

模型服务运行时是一个负责执行模型推理的系统组件。

它通常负责：

1. 加载模型权重。
2. 加载 tokenizer。
3. 管理 GPU 显存。
4. 接收推理请求。
5. 执行 prefill。
6. 执行 decode。
7. 管理 KV cache。
8. 做 batching。
9. 输出流式 token。
10. 暴露指标和健康检查。

从外部看，runtime 可能只是一个 HTTP 或 gRPC 服务。

从内部看，它是一个复杂的 GPU 执行引擎。

## 27.2 Runtime 和平台层的边界

很多团队会犯一个错误：把平台治理逻辑塞进 runtime，或者把 runtime 的高性能执行逻辑塞进平台层。

更合理的边界是：

平台层负责：

1. 模型注册。
2. 版本发布。
3. 请求鉴权。
4. 限流配额。
5. 模型路由。
6. 灰度回滚。
7. 自动扩缩容。
8. 成本归因。
9. 审计治理。

Runtime 负责：

1. 模型加载。
2. GPU kernel 执行。
3. batching。
4. KV cache。
5. token 生成。
6. streaming。
7. 单实例指标。

这条边界很重要。平台层要能替换 runtime，runtime 也不要依赖某个业务平台才能运行。

## 27.3 一个 Runtime 的内部结构

一个典型 LLM runtime 内部可能包含：

```text
API Server
  -> Request Parser
  -> Tokenizer
  -> Scheduler
  -> KV Cache Manager
  -> Batch Manager
  -> Model Executor
  -> Sampler
  -> Streamer
  -> Metrics Exporter
```

每个模块的作用：

1. API Server：接收请求。
2. Request Parser：解析 prompt、参数和停止条件。
3. Tokenizer：把文本转成 token。
4. Scheduler：决定哪些请求进入下一步执行。
5. KV Cache Manager：分配和回收 KV cache。
6. Batch Manager：组织 batch 或 continuous batch。
7. Model Executor：执行模型前向计算。
8. Sampler：执行 temperature、top-p、top-k 等采样。
9. Streamer：流式返回 token。
10. Metrics Exporter：输出延迟、吞吐、显存等指标。

面试中如果能讲出这些内部模块，说明你不是只会“部署模型服务”。

## 27.4 Triton 的定位

Triton Inference Server 是一个通用推理服务框架。

它的优势是：

1. 支持多种模型格式。
2. 支持多后端。
3. 支持 HTTP 和 gRPC。
4. 支持模型管理。
5. 支持动态 batching。
6. 支持 Prometheus 指标。
7. 适合传统深度学习模型服务。

Triton 对以下场景很常见：

1. CV 模型。
2. 推荐模型。
3. embedding 模型。
4. 多模型统一 serving。
5. TensorRT 优化模型。

但对生成式大模型，Triton 通常需要结合 TensorRT-LLM 或其他后端使用，否则很难直接覆盖现代 LLM 推理所需的 continuous batching、KV cache 管理和长上下文优化。

简单说：Triton 是通用 serving 框架，不是只为 LLM 生成推理设计的 runtime。

## 27.5 vLLM 的定位

vLLM 是面向大语言模型生成推理的 runtime。

它最知名的能力之一是 PagedAttention。

PagedAttention 的核心思想是：像操作系统管理虚拟内存一样管理 KV cache，把 KV cache 拆成块，减少显存碎片，提高并发能力。

vLLM 的常见优势：

1. 适合 LLM 文本生成。
2. 支持 continuous batching。
3. KV cache 管理能力强。
4. OpenAI-compatible API 生态友好。
5. 部署和使用门槛相对低。
6. 社区活跃。

vLLM 适合：

1. 通用文本生成服务。
2. Chat API。
3. 多并发在线推理。
4. 对吞吐和易用性都有要求的团队。

但 vLLM 也不是万能的。遇到极致性能、特定硬件、深度定制调度、复杂多模态或特殊模型结构时，可能需要额外工程改造或选择其他 runtime。

## 27.6 TGI 的定位

TGI 是 Text Generation Inference，常用于 Hugging Face 模型生态的文本生成服务。

它的优势是：

1. Hugging Face 模型生态集成好。
2. 文本生成 API 完整。
3. 支持 streaming。
4. 支持 batching。
5. 支持常见生成参数。
6. 工程化程度较高。

TGI 适合：

1. Hugging Face 模型快速上线。
2. 标准文本生成服务。
3. 团队希望少做底层改造。
4. 与 HF 生态深度结合的场景。

如果团队的模型、权重、tokenizer、部署流程大量围绕 Hugging Face，TGI 是一个自然选项。

但如果你需要更细粒度的调度控制、极致吞吐优化或复杂 agent runtime 集成，仍然要评估它和其他 runtime 的差异。

## 27.7 SGLang 的定位

SGLang 既可以看作推理 runtime，也可以看作面向结构化生成和复杂 LLM 程序的执行系统。

它关注的不只是单次 chat completion，还包括更复杂的 LLM 调用模式，例如：

1. 多轮生成。
2. 结构化输出。
3. 分支推理。
4. 并行采样。
5. 受约束生成。
6. agent 或 workflow 式调用。

SGLang 的价值在于：当上层应用不是简单的一问一答，而是有复杂控制流和生成模式时，runtime 可以更理解这些调用结构，从而做更好的执行优化。

适合场景包括：

1. 复杂 agent 服务。
2. 结构化生成服务。
3. 多阶段推理任务。
4. 需要高效复用上下文的任务。

如果说 vLLM 更像高性能 LLM serving engine，那么 SGLang 更强调 LLM program 的执行效率和表达能力。

## 27.8 TensorRT-LLM 的定位

TensorRT-LLM 更偏极致性能优化。

它常用于 NVIDIA GPU 上的大模型推理优化，关注 kernel、图优化、量化、并行和推理吞吐。

它的优势是：

1. 性能潜力高。
2. 与 NVIDIA 生态结合深。
3. 支持多种优化策略。
4. 适合大规模生产优化。
5. 可结合 Triton 做服务化。

但它的使用门槛通常更高：

1. 构建流程更复杂。
2. 模型转换成本更高。
3. 调试难度更大。
4. 对硬件和版本依赖更强。

如果目标是快速上线，可能优先考虑 vLLM 或 TGI。如果目标是大规模稳定流量下压低单位 token 成本，TensorRT-LLM 可能更有价值。

## 27.9 自研 Runtime 什么时候值得做

大多数团队不应该一开始就自研 runtime。

自研 runtime 成本很高，因为你要处理：

1. 模型加载。
2. 分布式推理。
3. kernel 优化。
4. KV cache 管理。
5. continuous batching。
6. speculative decoding。
7. 量化。
8. 多模型并发。
9. 长上下文。
10. 故障恢复。
11. 指标和调试。

但在一些场景下，自研是合理的：

1. 流量规模巨大，单位成本下降收益明显。
2. 模型结构高度定制。
3. 硬件不是主流 GPU。
4. 需要特殊调度策略。
5. 需要深度结合业务调用模式。
6. 开源 runtime 无法满足稳定性或性能要求。

自研 runtime 的前提是团队具备系统、CUDA、编译优化、分布式和模型结构理解能力。

## 27.10 Runtime 选型维度

选择 runtime 时，不要只问“哪个最快”。

应该看这些维度：

1. 支持哪些模型结构。
2. 是否支持目标硬件。
3. 是否支持 continuous batching。
4. KV cache 管理能力如何。
5. 长上下文支持如何。
6. streaming 能力如何。
7. OpenAI-compatible API 支持如何。
8. 量化支持如何。
9. 多 GPU 和多节点支持如何。
10. 指标是否完整。
11. 部署复杂度。
12. 社区活跃度。
13. 调试难度。
14. 与平台层集成成本。
15. 单位 token 成本。

runtime 选型是工程决策，不是排行榜决策。

## 27.11 性能指标怎么比较

比较 runtime 性能时，至少要固定以下条件：

1. 同一模型。
2. 同一权重格式。
3. 同一硬件。
4. 同一精度或量化方式。
5. 同一输入长度分布。
6. 同一输出长度分布。
7. 同一并发模型。
8. 同一 SLO。
9. 同一采样参数。
10. 同一 streaming 设置。

否则比较没有意义。

常见错误是只测一个短 prompt 的 tokens/s，然后宣称某个 runtime 更快。

生产环境中真正重要的是：

1. p95 TTFT。
2. p95 TPOT。
3. p99 latency。
4. 达到 SLO 时的最大吞吐。
5. 显存占用。
6. 失败率。
7. 成本 per 1k tokens。

## 27.12 Runtime 的平台集成方式

平台接入 runtime 时，不应该依赖某个 runtime 的所有细节。

常见方式是定义统一推理接口：

```text
InferenceEndpoint
  model
  version
  runtime
  endpoint_url
  max_context_length
  supported_features
  health_status
  metrics
```

请求层统一抽象：

```text
GenerateRequest
  prompt/messages
  max_tokens
  temperature
  top_p
  stop
  stream
  user/tenant
  trace_id
```

响应层统一抽象：

```text
GenerateResponse
  text/tokens
  finish_reason
  usage
  latency
  model_version
  runtime_instance
```

这样平台可以在 vLLM、TGI、SGLang、自研 runtime 之间切换，而不需要重写全部上层逻辑。

## 27.13 Runtime 指标必须暴露什么

一个可生产使用的 runtime 至少要暴露：

1. 请求数。
2. 成功率。
3. 错误率。
4. TTFT。
5. TPOT。
6. p95 / p99 latency。
7. queue length。
8. active requests。
9. waiting requests。
10. input tokens/s。
11. output tokens/s。
12. GPU utilization。
13. 显存占用。
14. KV cache 使用率。
15. batch size 分布。
16. OOM 次数。
17. timeout 次数。

没有这些指标，线上问题会很难排查。

比如用户说“模型变慢了”，你需要判断慢在哪里：

1. 网关慢？
2. 路由慢？
3. 排队慢？
4. prefill 慢？
5. decode 慢？
6. streaming 慢？
7. 下游客户端慢？

runtime 指标是定位问题的关键。

## 27.14 Runtime 的稳定性问题

Runtime 常见稳定性问题包括：

1. 模型加载失败。
2. tokenizer 不匹配。
3. 权重格式错误。
4. CUDA 版本不兼容。
5. 显存不足。
6. KV cache 碎片化。
7. 长请求拖慢短请求。
8. streaming 连接异常。
9. batch 调度饥饿。
10. 特定 prompt 触发异常。
11. 指标失真。
12. 版本升级引入性能退化。

因此 runtime 上线前要做：

1. 冷启动测试。
2. 压测。
3. 长上下文测试。
4. 高并发测试。
5. streaming 测试。
6. OOM 测试。
7. 升级回滚测试。

## 27.15 Runtime 和量化

量化可以降低显存占用和推理成本。

常见量化方向包括：

1. FP16。
2. BF16。
3. INT8。
4. INT4。
5. FP8。
6. weight-only quantization。
7. KV cache quantization。

量化的收益：

1. 显存占用更低。
2. 可部署更大模型。
3. batch 或并发更高。
4. 单位 token 成本更低。

量化的风险：

1. 质量下降。
2. 特定任务退化。
3. 兼容性问题。
4. 调试难度增加。
5. 与 runtime 支持强相关。

所以量化不是“越低越好”，而是要在质量、延迟、吞吐和成本之间权衡。

## 27.16 Runtime 和分布式推理

大模型太大时，单张 GPU 放不下，或者单机吞吐不够，就需要分布式推理。

常见方式包括：

1. Tensor Parallelism。
2. Pipeline Parallelism。
3. Expert Parallelism。
4. Data Parallel serving。
5. Prefill/decode 分离。

分布式推理会引入新的问题：

1. 通信开销。
2. 拓扑感知。
3. 多卡故障影响。
4. 实例启动变慢。
5. 调度复杂度上升。
6. 成本归因更复杂。

推理平台要知道 runtime 是否支持这些能力，以及需要什么 GPU 拓扑。

## 27.17 面试常见追问

问题一：vLLM 和 Triton 有什么区别？

可以回答：Triton 是通用推理服务框架，支持多模型和多后端；vLLM 更专注于 LLM 生成推理，强调 continuous batching 和 KV cache 管理。生成式大模型在线服务常优先评估 vLLM，传统模型或 TensorRT 生态可能更多使用 Triton。

问题二：为什么 runtime 需要 continuous batching？

可以回答：生成式推理每个请求输出长度不同，静态 batch 会出现等待和浪费。continuous batching 允许请求动态加入和退出，提高 GPU 利用率，同时控制延迟。

问题三：为什么不能只用 OpenAI-compatible API 判断 runtime 能力？

可以回答：API 兼容只说明接口像，不说明性能、KV cache、batching、长上下文、量化、分布式、指标和稳定性满足生产要求。

问题四：什么时候考虑自研 runtime？

可以回答：当流量规模足够大、单位成本优化收益明显，或者模型结构、硬件、调度和业务调用模式高度特殊，开源 runtime 无法满足要求时，才考虑自研。

## 27.18 小练习

1. Runtime 和推理平台层的边界是什么？
2. Triton 更适合哪些场景？
3. vLLM 的核心优势是什么？
4. TGI 为什么适合 Hugging Face 生态？
5. SGLang 和普通 LLM serving engine 的差异是什么？
6. 为什么 runtime benchmark 必须固定输入和输出长度分布？
7. 一个生产 runtime 至少要暴露哪些指标？
8. 什么时候才值得自研 runtime？

## 27.19 本章小结

本章讲了模型服务运行时的核心概念和选型。

你需要记住：

1. Runtime 负责高性能推理执行，平台层负责服务化和治理。
2. Triton 是通用推理服务框架，vLLM 更专注 LLM 生成推理。
3. TGI 适合 Hugging Face 文本生成生态，SGLang 更强调复杂 LLM 程序执行。
4. TensorRT-LLM 适合追求极致性能和深度 NVIDIA 生态优化的场景。
5. 自研 runtime 成本很高，只有在规模、硬件或业务模式足够特殊时才值得做。
6. runtime 选型要综合模型、硬件、batching、KV cache、长上下文、量化、指标、稳定性和成本。

下一章我们会深入 Prefill、Decode、KV Cache 与推理资源画像，理解为什么大模型推理和普通在线服务完全不同。
