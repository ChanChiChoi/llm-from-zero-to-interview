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
