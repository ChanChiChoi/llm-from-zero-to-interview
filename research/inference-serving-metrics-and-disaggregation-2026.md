# 推理服务指标与阶段解耦：来源证据

检索与阅读日期：2026-10-29。本记录只保存来源支持的测量定义、实现行为和证据边界；章节审阅、修改决定与工作进度记录在 progress/。

## vLLM v0.32.0 在线服务指标

- **来源与版本**：[vLLM v0.32.0 Production Metrics](https://docs.vllm.ai/en/v0.32.0/usage/metrics/)、[Metrics](https://docs.vllm.ai/en/v0.32.0/design/metrics/) 与 [Per-Request Metrics](https://docs.vllm.ai/en/v0.32.0/features/per_request_metrics/)，访问日期 2026-10-29。vLLM [v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) 发布于 2026-10-18。
- **阅读位置**：通用请求延迟指标、KV connector metrics、KV cache residency、TPOT/ITL 定义和 metrics deprecation 部分。
- **来源支持的事实**：v0.32.0 区分请求排队、Prefill、Decode、TTFT、inter-token latency、request-level TPOT、抢占、KV block 使用和传输/connector 指标；部分 KV 生命周期直方图需显式打开 KV cache metrics。ITL histogram 按流式 output event 记录间隔；request-level TPOT 为 `(end-to-end latency - TTFT) / (output tokens - 1)`，仅一个或零输出 token 的请求记为 0，而 `vllm bench serve` 的 TPOT 统计排除这些请求。per-request metrics 不能简单用于 `n>1` 或多次生成 turn 的响应归因，且高并发下计算这些指标可能增加 CPU 开销。
- **证据等级与边界**：source-backed（固定发布版本的官方文档）。指标名称、标签和特殊值绑定 vLLM v0.32.0；更换 release 时应重核。

## vLLM benchmark CLI 的客户端测量点

- **来源与版本**：[vLLM Benchmark CLI](https://github.com/vllm-project/vllm/blob/main/docs/benchmarking/cli.md)，main 动态文档，访问日期 2026-10-29。
- **阅读位置**：在线 benchmark 指标示例、Understanding the Latency Metrics、server-side per-request metrics 和 prefix-cache 警告。
- **来源支持的事实**：客户端 TTFT 从发出请求到收到首个 streamed output；该工具的 ITL 统计相邻 streamed outputs 之间的间隔；TPOT 对每条请求计算 (end-to-end latency - TTFT) / (output_tokens - 1) 后再聚合。标准解码中每个 streamed output 通常是一 token，ITL 与 TPOT 常接近；speculative decoding 可一次发送多个 token，同一事件中的 token 不会产生额外 ITL 样本，因此 ITL 与 TPOT 可明显不同。重复使用相同 benchmark 数据时，prefix cache 命中可能抬高结果；工具建议清 cache、重启或使用会重置缓存的 sweep。
- **证据等级与边界**：source-backed（vLLM main 的公开工具文档）。这定义的是该 CLI 的测量口径，不是跨 benchmark 的术语标准；跨工具比较应核对事件起止点、聚合方式和输出分组。

## GuideLLM goodput 与 TPOT SLO 映射

- **来源与版本**：[GuideLLM goodput API reference](https://vllm-project.github.io/guidellm/stable/reference/guidellm/schemas/benchmark/goodput/)，stable 动态文档，访问日期 2026-10-29。
- **阅读位置**：GoodputSLO 描述和字段注释。
- **来源支持的事实**：goodput 是满足已配置逐请求 latency objectives 的请求率；请求需满足全部已设置目标。GuideLLM 的 tpot SLO 对照 inter_token_latency_ms（不含首 token），而不是 GuideLLM 自身包含首 token 的 time_per_output_token_ms；它是最接近 vLLM TPOT 的字段，但仍不完全相同。输出不超过一个 token 的请求没有 inter-token latency，会被该字段置为 undetermined。
- **证据等级与边界**：source-backed（GuideLLM 公开 API 文档）。stable 文档与实现会演进；goodput 须同时报告采用的 SLO 字段和工具版本，不能只按字段名认定两个工具的合格率可比。

## vLLM Prefill/Decode 解耦工程说明

- **来源与版本**：Martin Hickey, “Taking vLLM Apart: A Practical Guide to Disaggregated Serving,” vLLM Blog，2026-09-29；文章讨论 vLLM v0.31.0 或更高版本的实现路径。[工程说明](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html)。
- **阅读位置**：阶段拆分、goodput/成本、单机测量、KV transfer、render/derender 与 limitations 部分。
- **来源支持的事实**：文章把 render、Prefill、Decode 和 derender 拆成可独立部署的服务层，并强调 Prefill/Decode 分离在固定 GPU 数下不保证 raw token throughput 提升；其目标是改善指定 TTFT/ITL 目标下的 goodput。文章报告的 L40S/Qwen2.5-7B 单机实验绑定约 8K prompt、256 输出、Poisson arrivals、prefix cache 关闭和特定 NIXL proxy。该测试也显示无 GPU peer-to-peer 的 KV 传输可能显著抬高 TTFT；collocated baseline 的数据并行拓扑不是所有可比较 baseline 的上限。
- **证据等级与边界**：source-backed（vLLM 项目工程博客及其自测配置）。这是一个实现案例，不是独立论文或跨框架基准；原始性能数字只适用于文章的模型、GPU、网络、请求分布和代理配置。

## NVIDIA Triton 推测解码说明

- **来源与版本**：[NVIDIA Triton Inference Server speculative decoding 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/llm_features/speculative_decoding_by_backend_type.html)，访问日期 2026-10-29。
- **阅读位置**：About Speculative Decoding 与 Performance Improvements。
- **来源支持的事实**：该文档概述 draft 多 token 候选、target 一次前向验证，并将收益条件关联到小 batch 下目标 GPU 未充分利用以及候选可接受程度；文档明确指出收益高度依任务而变。
- **证据等级与边界**：source-backed（NVIDIA 当前公开文档）。这属于 Triton 文档中的一般机制说明，不证明某个特定后端配置、模型或任务达到指定 speedup。

## MoE LLM 推理的多层调度

- **来源与版本**：Sun et al., “Multi-Layer Scheduling for MoE-Based LLM Reasoning,” arXiv:2602.21626 v2（2026-03-02），[预印本](https://arxiv.org/abs/2602.21626)。
- **阅读位置**：arXiv v2 论文页摘要；未在本记录中复核全文实验表和集群配置。
- **来源支持的事实**：摘要将 MoE serving 调度分为 request、engine、expert 三层；列举请求级 SJF/priority-aware aging、考虑 prefix token load 与 KV 使用情况的 engine dispatch，以及处理 expert hotspot 的策略。作者报告 100 多组实验中，相对其 vLLM baseline，TTFT 最多降低 17.8%、TPOT 最多降低 13.3%。
- **证据等级与边界**：source-backed（arXiv v2 预印本摘要中的作者系统结果，尚非同行评审定论）。峰值收益绑定论文模型、GPU/集群、请求分布、调度配置与 baseline 版本；不能由 router load 统计推出所有 MoE serving 的延迟改进。

## AgentPerfBench 的多轮推理负载

- **来源与版本**：Cheuk Hang Lau et al., “AgentPerfBench: A Benchmarking and Evaluation Suite for Inference Performance of Agentic LLMs,” arXiv:2609.34683v1，提交于 2026-09-28；[固定版本摘要](https://arxiv.org/abs/2609.34683v1)。
- **阅读位置**：arXiv v1 摘要；未在本记录中核对全文实验设置和数值表。
- **来源支持的事实**：作者以 SWE-Bench、TerminalBench 等 agent benchmark traces 建立多轮 workload，并从经验分布抽样输入长度、输出长度和 turn count 构造 synthetic profile；作者指出普通单轮聊天负载难以反映上下文累积和硬件饱和。
- **证据等级与边界**：source-backed（arXiv v1 摘要）。支持以真实任务轨迹和多轮请求分布构造 serving 压测；不支持把论文中的模型、GPU 或速度数字外推成通用预测。

## 生命周期

vLLM v0.32.0 的客户端/服务端指标定义：active，已固定对应 release；部署到其他版本时仍需重核。GuideLLM 与 benchmark CLI 的动态资料仍须绑定实际工具版本。Prefill/Decode 分离和 speculative decoding 的收益数字必须保留其模型、硬件、请求分布、网络和评估协议边界。

## 2026-10-30 推理 runtime、协议与部署来源

### Runtime 维护状态与量化能力

- **来源与版本**：[vLLM stable 文档](https://docs.vllm.ai/en/stable/)、[SGLang main 文档](https://github.com/sgl-project/sglang/blob/main/docs/index.mdx)、[TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview)、[TensorRT-LLM quantization](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html)，以及 [Hugging Face TGI 文档](https://huggingface.co/docs/text-generation-inference/main/en/index)和 [Inference Endpoints TGI 迁移说明](https://huggingface.co/docs/inference-endpoints/engines/tgi)；访问日期 2026-10-30。TensorRT-LLM overview 页面当时显示更新于 2026-10-26；其余为 stable/main/latest 动态文档。
- **阅读范围**：各项目的 serving/runtime 概览、量化支持目录，以及 TGI 文档中的维护状态提示。
- **来源支持的事实**：vLLM、SGLang 和 TensorRT-LLM 分别提供当前官方 runtime 文档入口；TGI 官方页面称其处于 maintenance mode，Inference Endpoints 迁移说明原样标注 “as of 12/11/2025” 并将下游引擎作为新部署选择。TensorRT-LLM 当前量化文档列出 FP4、FP8、FP8 KV、NVFP4 KV、W4A16/W4A8 GPTQ/AWQ 等支持路线，并将部分格式能力限定于后端与 GPU 代际。
- **证据等级与边界**：source-backed（项目官方文档）。stable/main/latest 页面不是不可变版本合同；TGI 维护状态不说明所有已部署模型均可直接迁移；量化条目不证明目标模型、GPU、kernel 或质量门禁可用。生产引用须固定软件版本、模型和硬件。

### NCCL collective 类型

- **来源与版本**：[NCCL User Guide: Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)，访问日期 2026-10-30。
- **阅读范围**：AllReduce、Broadcast、Reduce、AllGather、ReduceScatter、AlltoAll 的操作语义与调用约束。
- **来源支持的事实**：NCCL 文档定义各 collective 的输入/输出关系，并描述 rank、count、datatype 等调用约束。使用哪类 collective 由模型切分、runtime 与 kernel 实现决定。
- **证据等级与边界**：source-backed（NVIDIA 官方文档）。API 语义不能推出某个并行策略的真实通信时间；估算和性能结论需要目标拓扑与运行时 profile。

### Prefill/Decode 解耦与分层 KV 的论文范围

- **来源与版本**：[DistServe](https://arxiv.org/abs/2401.09670) 与 [Mooncake](https://arxiv.org/abs/2407.00079)，arXiv 论文页面；访问日期 2026-10-30。
- **阅读范围**：论文摘要、架构与系统设计描述；论文性能数字仅按各自设定引用。
- **来源支持的事实**：DistServe 研究将 prefill 与 decode 放入独立资源池以针对 TTFT/TPOT 目标配置资源；Mooncake 以 KV 为中心描述 disaggregated serving 与多层 KV 管理。
- **证据等级与边界**：source-backed（原始研究论文/预印本）。论文结论和收益受作者模型、GPU、网络、trace、SLO、对照系统及实验配置约束；支持架构设计动机，不构成任意 workload 下的性能保证。

### SSE、Kubernetes probes 与跨服务 tracing

- **来源与版本**：[WHATWG Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html)，HTML Standard 动态页面，访问日期 2026-10-30；[Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/)，官方文档，访问日期 2026-10-30；[OpenTelemetry traces](https://opentelemetry.io/docs/concepts/signals/traces/) 与 [context propagation](https://opentelemetry.io/docs/concepts/context-propagation/)，官方动态文档，访问日期 2026-10-30。
- **阅读范围**：SSE event-stream media type、字段与解析；startup/liveness/readiness probe 生命周期；trace spans 与跨服务 context propagation 概念。
- **来源支持的事实**：WHATWG 定义 text/event-stream 格式、data/event/id/retry 字段及浏览器 EventSource 的连接处理，但不定义模型 API 的 error payload 或 finish reason。Kubernetes 文档说明配置 startup probe 后，startup 成功前不执行 liveness/readiness；liveness 和 startup 失败可触发容器重启，readiness 失败用于停止向 Pod 发送流量。OpenTelemetry traces 使用 span 记录操作并可传播 trace context，以关联分布式服务中的操作。
- **证据等级与边界**：source-backed（规范和项目官方文档）。模型 API 的事件名字、终止标记、取消与重试须由 API 自己约定；镜像拉取在容器进程探针之前；trace 标准本身不保证指标口径、数据脱敏或完整跨服务采样。
