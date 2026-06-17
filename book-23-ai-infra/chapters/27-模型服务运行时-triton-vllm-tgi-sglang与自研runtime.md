# 第 27 章 模型服务运行时：Triton、vLLM、TGI、SGLang 与自研 runtime

上一章讲了推理平台总览。本章聚焦一个更底层也更关键的问题：模型服务运行时，也就是 inference runtime。

推理平台负责把请求接进来、路由、限流、扩缩容、灰度和治理；runtime 负责真正把模型跑起来，把 token 生成出来。

先记住一句话：

> 推理平台管“服务化和治理”，模型服务运行时管“高性能执行”。

如果 runtime 选型错误，平台层做得再漂亮，也很难获得好的延迟、吞吐和成本。

## 27.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 NVIDIA Triton Inference Server 官方文档中 model repository、dynamic batching、statistics / metrics、backend 与多模型推理服务的边界；vLLM 官方文档中 PagedAttention、automatic prefix caching、production metrics、OpenAI-compatible server、量化和分布式 serving 的边界；Hugging Face Text Generation Inference 官方文档中 streaming、Prometheus 指标、PagedAttention 和当前 maintenance mode 的说明；SGLang 官方文档中 structured output、RadixAttention / cache、multi-GPU serving、PD disaggregation 和 serving API 的边界；以及 TensorRT-LLM 官方文档中 NVIDIA GPU 上的高性能 LLM inference、batching、KV cache 和 TensorRT / Triton 生态集成边界。

这些资料说明一件事：runtime 选型不是“哪个框架名气最大”，而是要把模型结构、权重格式、tokenizer / chat template、硬件拓扑、prefill / decode、continuous batching、KV cache、streaming、量化、分布式、指标、发布回滚、维护状态和成本放到同一张表里判断。

本章只抽象截至 2026-06 仍稳定的工程口径，不把某个 benchmark 排名、某个版本的默认参数、某个云厂商镜像或某次社区测评写成通用结论。Triton 更偏通用多模型推理服务框架；vLLM、SGLang 更偏现代 LLM serving runtime；TGI 对 Hugging Face 生态仍有学习价值，但当前官方维护状态会影响新项目选型；TensorRT-LLM 更偏 NVIDIA 生态下的深度性能优化；自研 runtime 只有在规模、模型结构、硬件或调度模式足够特殊时才值得投入。

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

但如果你需要更细粒度的调度控制、极致吞吐优化或复杂 agent runtime 集成，仍然要评估它和其他 runtime 的差异。尤其要注意版本边界：当前官方文档已把 TGI 标注为维护模式，新项目选型时不能只看历史生态成熟度，还要看后续维护、兼容性和安全更新责任。

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

## 27.17 模型服务运行时审计指标与最小 demo

面试里谈 runtime 选型，最好不要直接回答“用 vLLM”或“用 Triton”。更稳的回答是：先定义 workload 和 SLO，再把候选 runtime 放进同一套审计表。

可以把一个 runtime 选型样本写成：

```math
r_i=(m_i,h_i,f_i,t_i,p_i,b_i,k_i,s_i,q_i,d_i,o_i,e_i,c_i,a_i,u_i,z_i)
```

其中 `m_i` 表示模型与任务类型，`h_i` 表示硬件和拓扑，`f_i` 表示权重格式和 tokenizer，`t_i` 表示 runtime 与平台边界，`p_i` 表示 prefill / decode 和调度，`b_i` 表示 batching，`k_i` 表示 KV cache 管理，`s_i` 表示 streaming，`q_i` 表示量化，`d_i` 表示分布式推理，`o_i` 表示 observability，`e_i` 表示 benchmark 口径，`c_i` 表示成本容量，`a_i` 表示发布回滚，`u_i` 表示自研门槛，`z_i` 表示最终门禁。

对第 `j` 个审计维度，统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(r_i)=1]
```

runtime 内部延迟可以拆成：

```math
T_{\mathrm{runtime}}=T_{\mathrm{tokenize}}+T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{sample}}+T_{\mathrm{stream}}
```

如果只看 `tokens/s`，就会漏掉用户最敏感的首 token 等待和流式卡顿。因此 TTFT 与 TPOT 仍然要单独看：

```math
T_{\mathrm{ttft}}=T_{\mathrm{tokenize}}+T_{\mathrm{queue}}+T_{\mathrm{prefill}}+T_{\mathrm{first}}
```

```math
T_{\mathrm{tpot}}=\frac{T_{\mathrm{decode}}+T_{\mathrm{sample}}+T_{\mathrm{stream}}}{N_{\mathrm{out}}}
```

KV cache 的单请求显存近似可以写成：

```math
M_{\mathrm{kv,req}}=2LSH_{\mathrm{kv}}D_hB_{\mathrm{elem}}
```

给定可用于 KV cache 的预算 `B_kv`，粗略并发上限是：

```math
N_{\mathrm{kv}}=\left\lfloor \frac{B_{\mathrm{kv}}}{M_{\mathrm{kv,req}}}\right\rfloor
```

把功能、SLO、成本和风险放在一起，runtime 选型门禁可以写成：

```math
G_{\mathrm{runtime}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land T_{\mathrm{ttft,p95}}\le B_{\mathrm{ttft}} \land T_{\mathrm{tpot,p95}}\le B_{\mathrm{tpot}} \land S_{\mathrm{feature}}\ge \tau_{\mathrm{feature}} \land R_{\mathrm{compat}}=1 \land P_0=0\right]
```

下面的 0 依赖 demo 不是为了复现真实 runtime benchmark，而是演示如何把 runtime 选型变成可审计指标。你可以把 Triton、vLLM、TGI、SGLang、TensorRT-LLM 或自研 runtime 都抽象成同样的 profile，再用统一门禁比较。

```python
# Runtime Selection Audit: 0-dependency teaching demo.
from copy import deepcopy


GATES = [
    "model_hardware_fit",
    "runtime_boundary_clarity",
    "model_format_tokenizer_fit",
    "prefill_decode_scheduler_fit",
    "continuous_batching_fit",
    "kv_cache_memory_manager_fit",
    "streaming_api_fit",
    "quantization_accuracy_fit",
    "distributed_inference_fit",
    "observability_metrics_fit",
    "benchmarking_method_fit",
    "deployment_rollback_fit",
    "ecosystem_maintenance_fit",
    "cost_capacity_fit",
    "self_development_threshold",
    "runtime_selection_gate",
]


def has_keys(obj, keys):
    return isinstance(obj, dict) and all(obj.get(k) for k in keys)


def percentile(values, p):
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(len(values) * p + 0.999999) - 1))
    return values[idx]


def kv_cache_mib(layers, batch, seq, kv_heads, head_dim, dtype_bytes):
    bytes_total = 2 * layers * batch * seq * kv_heads * head_dim * dtype_bytes
    return bytes_total / (1024 * 1024)


def gate_results(profile):
    metrics = set(profile.get("metrics", []))
    benchmark = profile.get("benchmark", {})
    ecosystem = profile.get("ecosystem", {})
    self_dev = profile.get("self_development", {})
    kv = profile.get("kv_cache", {})

    results = {
        "model_hardware_fit": has_keys(profile, ["model_class", "hardware", "runtime_name"]),
        "runtime_boundary_clarity": has_keys(profile.get("boundary"), ["platform_controls", "runtime_controls"]),
        "model_format_tokenizer_fit": has_keys(profile.get("model_artifact"), ["weight_format", "tokenizer_version", "chat_template", "stop_tokens"]),
        "prefill_decode_scheduler_fit": has_keys(profile.get("scheduler"), ["prefill_decode_split", "queue_policy", "phase_metrics"]),
        "continuous_batching_fit": has_keys(profile.get("continuous_batching"), ["dynamic_admit", "dynamic_evict", "max_batch_tokens", "long_request_isolation"]),
        "kv_cache_memory_manager_fit": has_keys(kv, ["block_size", "admission_control", "prefix_cache", "eviction_policy", "kv_pressure_metric"]),
        "streaming_api_fit": has_keys(profile.get("streaming"), ["transport", "backpressure", "cancellation", "finish_reason"]),
        "quantization_accuracy_fit": has_keys(profile.get("quantization"), ["dtype", "quality_eval", "rollback_threshold"]),
        "distributed_inference_fit": has_keys(profile.get("distributed"), ["parallel_mode", "topology", "rank_health"]),
        "observability_metrics_fit": {"ttft", "tpot", "queue", "prefill", "decode", "kv", "oom", "error", "version"}.issubset(metrics),
        "benchmarking_method_fit": has_keys(benchmark, ["same_model", "same_hardware", "same_precision", "input_distribution", "output_distribution", "concurrency", "slo"]),
        "deployment_rollback_fit": has_keys(profile.get("deployment"), ["image_digest", "runtime_version", "canary", "rollback", "health_probe"]),
        "ecosystem_maintenance_fit": has_keys(ecosystem, ["status", "release_tracking", "compatibility_matrix", "owner"]),
        "cost_capacity_fit": has_keys(profile.get("cost"), ["gpu_hour_usd", "tokens_in", "tokens_out", "kv_budget_mib", "target_cost_per_1k"]),
        "self_development_threshold": (not self_dev.get("self_develop")) or has_keys(self_dev, ["scale_benefit", "systems_expertise", "cuda_expertise", "fallback_runtime"]),
        "runtime_selection_gate": bool(profile.get("final_gate")),
    }
    return results


def audit_runtime_selection(profiles):
    failed_cases = []
    failed_gates = set()
    for profile in profiles:
        results = gate_results(profile)
        bad = [name for name, ok in results.items() if not ok]
        if bad:
            failed_cases.append(profile["case"])
            failed_gates.update(bad)

    n = len(profiles)
    metrics = {}
    for gate in GATES:
        metrics[gate] = round(sum(gate_results(p)[gate] for p in profiles) / n, 3)

    return {
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": [g for g in GATES if g in failed_gates],
        "hard_blocker_count": len(failed_gates),
        "runtime_selection_gate_pass": not failed_cases,
    }


complete = {
    "case": "complete_runtime_profile_ok",
    "runtime_name": "vllm_or_sglang_style_llm_runtime",
    "model_class": "decoder_llm",
    "hardware": "8x_gpu_nvlink",
    "boundary": {
        "platform_controls": ["auth", "routing", "quota", "release"],
        "runtime_controls": ["prefill", "decode", "kv_cache", "sampling"],
    },
    "model_artifact": {
        "weight_format": "safetensors",
        "tokenizer_version": "tok-v3",
        "chat_template": "chatml-v2",
        "stop_tokens": ["eos"],
    },
    "scheduler": {
        "prefill_decode_split": True,
        "queue_policy": "priority_with_token_budget",
        "phase_metrics": ["queue", "prefill", "decode"],
    },
    "continuous_batching": {
        "dynamic_admit": True,
        "dynamic_evict": True,
        "max_batch_tokens": 8192,
        "long_request_isolation": True,
    },
    "kv_cache": {
        "block_size": 16,
        "admission_control": True,
        "prefix_cache": True,
        "eviction_policy": "lru_per_tenant",
        "kv_pressure_metric": True,
    },
    "streaming": {
        "transport": "sse",
        "backpressure": True,
        "cancellation": True,
        "finish_reason": True,
    },
    "quantization": {
        "dtype": "bf16_or_fp8",
        "quality_eval": "slice_regression",
        "rollback_threshold": "quality_drop_lt_1pct",
    },
    "distributed": {
        "parallel_mode": "tensor_parallel",
        "topology": "nvlink_aware",
        "rank_health": "per_rank_heartbeat",
    },
    "metrics": ["ttft", "tpot", "queue", "prefill", "decode", "kv", "oom", "error", "version"],
    "benchmark": {
        "same_model": True,
        "same_hardware": True,
        "same_precision": True,
        "input_distribution": "prod_prompt_histogram",
        "output_distribution": "prod_output_histogram",
        "concurrency": "prod_qps_sweep",
        "slo": "ttft_tpot_p99",
    },
    "deployment": {
        "image_digest": "sha256:runtime",
        "runtime_version": "pinned",
        "canary": True,
        "rollback": True,
        "health_probe": True,
    },
    "ecosystem": {
        "status": "active_or_risk_accepted",
        "release_tracking": "monthly",
        "compatibility_matrix": "gpu_cuda_driver_model",
        "owner": "serving_team",
    },
    "cost": {
        "gpu_hour_usd": 0.90,
        "tokens_in": 18000,
        "tokens_out": 24000,
        "kv_budget_mib": 32768,
        "target_cost_per_1k": 0.03,
    },
    "self_development": {
        "self_develop": False,
        "fallback_runtime": "open_source_runtime",
    },
    "final_gate": True,
}


def make_bad(case, mutator):
    item = deepcopy(complete)
    item["case"] = case
    mutator(item)
    return item


profiles = [complete]
profiles.append(make_bad("model_hardware_missing_bad", lambda p: p.pop("hardware")))
profiles.append(make_bad("runtime_boundary_blurred_bad", lambda p: p["boundary"].pop("runtime_controls")))
profiles.append(make_bad("model_format_tokenizer_gap_bad", lambda p: p["model_artifact"].pop("chat_template")))
profiles.append(make_bad("prefill_decode_scheduler_gap_bad", lambda p: p["scheduler"].pop("phase_metrics")))
profiles.append(make_bad("continuous_batching_gap_bad", lambda p: p["continuous_batching"].pop("dynamic_evict")))
profiles.append(make_bad("kv_cache_manager_gap_bad", lambda p: p["kv_cache"].pop("admission_control")))
profiles.append(make_bad("streaming_api_gap_bad", lambda p: p["streaming"].pop("backpressure")))
profiles.append(make_bad("quantization_eval_gap_bad", lambda p: p["quantization"].pop("quality_eval")))
profiles.append(make_bad("distributed_inference_gap_bad", lambda p: p["distributed"].pop("topology")))
profiles.append(make_bad("observability_metrics_gap_bad", lambda p: p["metrics"].remove("decode")))
profiles.append(make_bad("benchmark_method_gap_bad", lambda p: p["benchmark"].pop("output_distribution")))
profiles.append(make_bad("deployment_rollback_gap_bad", lambda p: p["deployment"].pop("rollback")))
profiles.append(make_bad("ecosystem_maintenance_unknown_bad", lambda p: p["ecosystem"].pop("status")))
profiles.append(make_bad("cost_capacity_gap_bad", lambda p: p["cost"].pop("kv_budget_mib")))
profiles.append(make_bad("self_development_threshold_gap_bad", lambda p: p.update({"self_development": {"self_develop": True, "scale_benefit": "unclear"}})))
profiles.append(make_bad("runtime_selection_gate_missing_bad", lambda p: p.update({"final_gate": False})))

latency_samples = [
    {"ttft_ms": 180, "tpot_ms": 18, "e2e_ms": 1280},
    {"ttft_ms": 220, "tpot_ms": 22, "e2e_ms": 1500},
    {"ttft_ms": 270, "tpot_ms": 24, "e2e_ms": 1720},
    {"ttft_ms": 310, "tpot_ms": 26, "e2e_ms": 1980},
    {"ttft_ms": 320, "tpot_ms": 28, "e2e_ms": 2300},
]
feature_score = sum(gate_results(complete)[g] for g in GATES[:-1]) / (len(GATES) - 1)
kv_mib = kv_cache_mib(layers=32, batch=16, seq=4096, kv_heads=8, head_dim=128, dtype_bytes=2)
per_request_kv_mib = kv_cache_mib(layers=32, batch=1, seq=4096, kv_heads=8, head_dim=128, dtype_bytes=2)
cost_per_1k = 1000 * (0.90 + 0.03 + 0.02) / (complete["cost"]["tokens_in"] + complete["cost"]["tokens_out"])
audit = audit_runtime_selection(profiles)

runtime_selection_examples = {
    "ttft_p95_ms": percentile([x["ttft_ms"] for x in latency_samples], 0.95),
    "tpot_p95_ms": percentile([x["tpot_ms"] for x in latency_samples], 0.95),
    "e2e_p99_ms": percentile([x["e2e_ms"] for x in latency_samples], 0.99),
    "kv_cache_mib": round(kv_mib, 1),
    "kv_concurrency_limit": int(complete["cost"]["kv_budget_mib"] // per_request_kv_mib),
    "feature_score": round(feature_score, 3),
    "cost_per_1k_tokens": round(cost_per_1k, 4),
}
smoke = {
    "complete_case_passes": not any(v is False for v in gate_results(complete).values()),
    "caught_boundary_gap": "runtime_boundary_blurred_bad" in audit["failed_cases"],
    "caught_kv_gap": "kv_cache_manager_gap_bad" in audit["failed_cases"],
    "caught_observability_gap": "observability_metrics_gap_bad" in audit["failed_cases"],
    "caught_self_dev_gap": "self_development_threshold_gap_bad" in audit["failed_cases"],
    "caught_gate_gap": "runtime_selection_gate_missing_bad" in audit["failed_cases"],
}

print(f"runtime_selection_examples={runtime_selection_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"failed_gates={audit['failed_gates']}")
print(f"runtime_selection_gate_pass={audit['runtime_selection_gate_pass']}")
```

这段 demo 想强调三点：

1. Triton、vLLM、TGI、SGLang、TensorRT-LLM 和自研 runtime 不能只按名称比较，要统一成 profile。
2. runtime benchmark 必须固定模型、硬件、精度、输入输出长度、并发和 SLO，并同时看 TTFT、TPOT、E2E、KV、功能覆盖和成本。
3. 自研 runtime 不是“更高级”的默认答案，只有规模收益、系统能力、CUDA / kernel 能力和 fallback 方案都成立时才进入候选。

## 27.18 面试常见追问

问题一：vLLM 和 Triton 有什么区别？

可以回答：Triton 是通用推理服务框架，支持多模型和多后端；vLLM 更专注于 LLM 生成推理，强调 continuous batching 和 KV cache 管理。生成式大模型在线服务常优先评估 vLLM，传统模型或 TensorRT 生态可能更多使用 Triton。

问题二：为什么 runtime 需要 continuous batching？

可以回答：生成式推理每个请求输出长度不同，静态 batch 会出现等待和浪费。continuous batching 允许请求动态加入和退出，提高 GPU 利用率，同时控制延迟。

问题三：为什么不能只用 OpenAI-compatible API 判断 runtime 能力？

可以回答：API 兼容只说明接口像，不说明性能、KV cache、batching、长上下文、量化、分布式、指标和稳定性满足生产要求。

问题四：什么时候考虑自研 runtime？

可以回答：当流量规模足够大、单位成本优化收益明显，或者模型结构、硬件、调度和业务调用模式高度特殊，开源 runtime 无法满足要求时，才考虑自研。

## 27.19 小练习

1. Runtime 和推理平台层的边界是什么？
2. Triton 更适合哪些场景？
3. vLLM 的核心优势是什么？
4. TGI 为什么适合 Hugging Face 生态？
5. SGLang 和普通 LLM serving engine 的差异是什么？
6. 为什么 runtime benchmark 必须固定输入和输出长度分布？
7. 一个生产 runtime 至少要暴露哪些指标？
8. 什么时候才值得自研 runtime？
9. 写一个 runtime profile，至少覆盖模型、硬件、权重格式、tokenizer、prefill / decode、batching、KV cache、streaming、量化、分布式、指标、发布和成本。
10. 为什么 TGI 的维护状态、TensorRT-LLM 的硬件绑定和自研 runtime 的团队能力都应该进入选型门禁？

## 27.20 本章小结

本章讲了模型服务运行时的核心概念和选型。

你需要记住：

1. Runtime 负责高性能推理执行，平台层负责服务化和治理。
2. Triton 是通用推理服务框架，vLLM 更专注 LLM 生成推理。
3. TGI 适合 Hugging Face 文本生成生态，SGLang 更强调复杂 LLM 程序执行。
4. TensorRT-LLM 适合追求极致性能和深度 NVIDIA 生态优化的场景。
5. 自研 runtime 成本很高，只有在规模、硬件或业务模式足够特殊时才值得做。
6. runtime 选型要综合模型、硬件、batching、KV cache、长上下文、量化、指标、稳定性和成本。
7. 生产选型要用统一审计表比较候选 runtime，避免只看单点 benchmark 或框架流行度。

下一章我们会深入 Prefill、Decode、KV Cache 与推理资源画像，理解为什么大模型推理和普通在线服务完全不同。
