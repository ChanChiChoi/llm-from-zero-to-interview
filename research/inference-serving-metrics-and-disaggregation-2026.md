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

## 2026-11-02 Prefill/Decode 解耦与工具状态合同

### SGLang PD disaggregation

- **来源与版本**：[SGLang v0.5.22 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.22)，访问日期 2026-11-02；[PD Disaggregation 官方文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)，current 动态文档，访问日期 2026-11-02。
- **实际阅读位置**：release highlight；官方文档的 “Why and What is PD Disaggregation?”、decode allocation policy、HTTP PD Responses API 限制。
- **来源支持的事实**：v0.5.22 release notes 将 “up to 27% more throughput and half the p95 TTFT on long-prompt workloads” 列为 PD disaggregation 的 release highlight；该行没有给出可据以独立复算的完整模型、硬件、到达过程和比较协议，因此只能作为发布方摘要主张引用。current 官方指南解释 Prefill 处理完整输入、偏计算密集；Decode 逐 token 生成并管理 KV，偏内存密集，并列出统一调度中的 Prefill interruption 与 DP attention imbalance。指南称当前 PD transfer engine 支持 Mooncake 和 NIXL。
- **实现与边界**：current 指南描述 `prefill_complete` decode allocation policy：将 destination KV slots 的预留延迟到 Prefill 完成信号之后；页面明确称它改变逐请求的 KV reservation 时机，不改变全局 KV pool 大小，也不是通用吞吐提升。该初始实现列有 Mooncake、attention TP/CP 和 pipeline size 等限制；两侧 worker 必须使用匹配 revision 和 policy。使用前应按部署的确切 SGLang revision 检查兼容项。
- **Responses under HTTP PD**：current 文档称 HTTP PD 可对 foreground `POST /v1/responses` 请求进行流式生成，但不支持 response persistence/history、`previous_response_id` chaining、background Responses、按 response ID 检索/取消等依赖共享进程内状态的路径；文档也称内建 `web_search`、`code_interpreter` tools 在该 PD 路由下不支持。这个限制源于 Prefill/Decode 是分开的 serving processes，HTTP PD router 每个请求只派发一组 Prefill/Decode pair；不能外推到所有 SGLang serving 模式。
- **证据等级与边界**：`source-backed`（项目 release note 和 current 官方文档）。性能数字是未附齐实验条件的发布摘要；功能细节来自动态文档而非不可变 release tag。不得据此推断任意请求分布或任意 release 下的收益与 API 兼容性。

### vLLM v0.32.0 disaggregated-serving 文档入口

- **来源与版本**：[vLLM v0.32.0 Disaggregated Serving examples](https://docs.vllm.ai/en/v0.32.0/examples/disaggregated/disaggregated_serving/) 和 [v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，访问日期 2026-11-02。
- **实际阅读位置**：disaggregated-serving 页面导航中的 encoder disaggregation、disaggregated serving examples、disaggregated-prefill experimental entry 与 connector examples。
- **来源支持的事实**：固定版本文档把 disaggregated encoder、disaggregated serving、experimental disaggregated prefill 及若干 KV connector 示例列为不同入口；仅出现在文档/示例列表中不意味着某个 connector、模型和硬件组合已经达到 production readiness。
- **证据等级与边界**：`source-backed`（固定 v0.32.0 官方文档结构）。该记录只支持文档明确标注的能力类别和 experimental 标记，不代替目标配置的运行验证、兼容性表或吞吐测量。

## 2026-11-02 推理阶段、指标定义与 roofline 测量

### Prefill 生成首 token 的阶段边界

- **来源与版本**：[Sarathi-Serve arXiv v3 HTML](https://arxiv.org/html/2403.02310)（2024-06-17）及 [Splitwise arXiv v2 HTML](https://arxiv.org/html/2311.18677)（2024-05-20），访问日期 2026-11-02。
- **实际阅读位置**：Sarathi-Serve 摘要与 §1；Splitwise 摘要、§I 与 §II-B。
- **来源支持的事实**：两份 serving 系统论文均把 prefill/prompt-computation 描述为处理完整输入并产生首个输出 token；后续 decode/token-generation 前向再以最近生成 token 和已缓存上下文产生后续 token。该表述支持常见 decoder-only 自回归路径：prompt 最后位置的输出 logits 可用于采样首 token。
- **证据等级与边界**：source-backed（原始系统论文）。论文支持阶段模型及其调度讨论，不定义所有 runtime 的 trace span 命名；具体实现可能把采样和输出后处理归入不同 span。不能仅凭接口名称断言额外存在一次完整 Transformer 首 token decode 前向。

### vLLM v0.32.0 benchmark CLI 指标与负载生成

- **来源与版本**：[Benchmark CLI](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)、[Paged Attention](https://docs.vllm.ai/en/v0.32.0/design/paged_attention/)、[Production Metrics](https://docs.vllm.ai/en/v0.32.0/usage/metrics/) 与 [Metrics design](https://docs.vllm.ai/en/v0.32.0/design/metrics/)，访问日期 2026-11-02。
- **实际阅读位置**：Benchmark CLI 的 serving 结果、Understanding the Latency Metrics 和 Load Pattern Configuration；Paged Attention 页面；指标设计文档中的请求计时与输出 histogram 口径。
- **来源支持的事实**：v0.32.0 benchmark CLI 将客户端 TTFT 定义为发送请求至收到首个 streamed output；ITL 对成功请求内相邻 streamed outputs 的 gap 聚合；TPOT 则先按请求用 (E2E - TTFT)/(output_tokens - 1) 计算，再对请求聚合。若 speculative decoding 的一个事件携带多个 token，ITL 不产生事件内部的额外 gap，而 TPOT 的分母仍使用输出 token 数。该 CLI 的 request-rate 默认为 infinity，所有请求会在 benchmark 开始时发出；有限 request rate 可配 burstiness 与 max concurrency。文档也提示重复向同一 server 发送 prompts 可能因 prefix-cache 命中抬高吞吐。Roofline 与延迟字段均须按该工具版本和具体采样边界解释。
- **证据等级与边界**：source-backed（固定版本官方文档）。这只是 vLLM v0.32.0 的 client benchmark 定义，不是通用指标标准；线性插值、nearest-rank、histogram 聚合以及输出事件合并方式会产生不同统计结果。固定版本链接仍不能代替目标设备和运行时的实测。

### TensorRT-LLM benchmark 文档中的 ITL 分母表述

- **来源与版本**：[TensorRT-LLM trtllm-serve benchmark guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/commands/trtllm-serve/run-benchmark-with-trtllm-serve.md)，GitHub main 动态页面，访问日期 2026-11-02。
- **实际阅读位置**：Key Metrics 下的 TPOT、ITL 文字说明和跨请求聚合公式。
- **来源支持的事实**：页面说明 request-level TPOT 先对每请求计算后再做请求等权平均；它将跨请求的 ITL 描述为所有 ITL gap 聚合的 token-weighted 平均，但公式分母写作跨请求的 Num Output Tokens。若每个请求有 n_i 个输出 token 且 ITL 样本仅为相邻输出 gap，其样本总数应为 sum(n_i - 1)，与 sum(n_i) 不同。页面没有在这一处解释首 token 是否作为额外样本或如何处理单 token 请求。
- **证据等级与边界**：source-backed（动态官方文档的当前公开文字），但定义存在需按实现澄清的内部歧义；在未阅读相应固定 revision 的 harness 代码前，不据此确定其实际统计分母，也不把其公式当作任意工具的 ITL 定义。

### Nsight Compute 的 kernel roofline 和 profiler overhead

- **来源与版本**：[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)，当前官方文档，访问日期 2026-11-02。
- **实际阅读位置**：§2.2.6 Overhead 与 §2.9.1—2.9.2 Roofline Chart / Analysis。
- **来源支持的事实**：profiling 开销受所采集 metric 的类型和数量、section set、section 数、被 profile 的 kernel 数及 GPU 架构影响；软件 patch 和需要多轮 replay 的指标尤其可能增加开销。Roofline chart 将 kernel achieved value 与计算和内存带宽上界及 ridge point 对照，主要用于 kernel 级分析，不直接覆盖排队、tokenizer、网络和服务协议开销。
- **证据等级与边界**：source-backed（NVIDIA 官方工具文档）。具体 counter 的定义、内存层级与 replay 行为依 Nsight Compute/GPU 版本和所选 section 而定；profile run 的数字不可未经说明等同于未插桩服务性能。

## 2026-11-02 推理 runtime 选型与服务接口

### Transformers Serve CLI

- **来源与版本**：[Transformers v5.22.1 Serve CLI](https://huggingface.co/docs/transformers/v5.22.1/serve-cli/serving)，v5.22.1 于 2026-10-30 发布，访问日期 2026-11-02；[release note](https://github.com/huggingface/transformers/releases/tag/v5.22.1)。
- **实际阅读位置**：Serve CLI 的定位、continuous batching 和 large-scale deployment 指引。
- **来源支持的事实**：官方将 `transformers serve` 描述为轻量的 local/self-hosted server，适用于 evaluation、experimentation 和 moderate load deployments；列有 continuous batching，并明确建议 large-scale production 使用 vLLM 或 SGLang 作为 Transformer backend。v5.22.1 页面没有把 Serve CLI 标成 experimental。该结论只描述文档的当前措辞，不证明任一模型、流量或运营配置已经满足生产 SLO。
- **证据等级与边界**：source-backed（固定 Transformers 文档版本及 release）。文档建议不能代替目标模型协议回放、负载容量和故障恢复验收。

### vLLM V1 scheduler 与 benchmark

- **来源与版本**：[vLLM v0.32.0 scheduler API/source reference](https://docs.vllm.ai/en/v0.32.0/api/vllm/v1/core/sched/scheduler/)、[V1 architecture](https://docs.vllm.ai/en/v0.32.0/design/arch_overview/)、[optimization and tuning](https://docs.vllm.ai/en/v0.32.0/configuration/optimization/) 与 [benchmark CLI](https://docs.vllm.ai/en/v0.32.0/benchmarking/cli/)，访问日期 2026-11-02。
- **实际阅读位置**：scheduler 的 `schedule` 说明与实现注释、V1 API server/engine core/GPU worker 职责、chunked prefill，以及 benchmark 的请求率和延迟口径。
- **来源支持的事实**：v0.32.0 scheduler 按 step 安排已运行和等待请求的 token 计算进度；其源码注释说明调度循环不把执行拆成独立的 prefill phase 和 decode phase，并在该循环中处理 chunked prefill、prefix caching 和 speculative decoding。v0.32.0 架构文档把 HTTP/input processing 放在 API server，把 scheduler/KV manager 放在 engine core，把模型前向放在 GPU worker。固定版本 benchmark CLI 分别暴露请求到达率、burstiness 与 max concurrency，并警告重复 prompts 会因 prefix-cache 命中影响结果。
- **证据等级与边界**：source-backed（固定 v0.32.0 官方文档和源码引用）。具体调度细节绑定 v0.32.0；不同版本、模型执行路径和配置需重核，文档/示例不证明目标硬件实测表现。

### SGLang program、session cache 与 eviction

- **来源与版本**：[SGLang 原论文 v2](https://arxiv.org/abs/2312.07104v2)；[current Session-Aware Radix Cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache) 和 [Radix Cache Eviction Policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)，访问日期 2026-11-02。
- **实际阅读位置**：论文摘要中的前端/runtime、RadixAttention 与 compressed FSM；current docs 的 session ID、`/close_session` 与 eviction 行为。
- **来源支持的事实**：论文将 SGLang 描述为前端语言与 runtime 的组合，以 RadixAttention 复用 KV cache、以压缩有限状态机加速结构化输出解码；作者报告的最高 6.4× throughput 绑定论文任务、模型、硬件与对照系统。current session 文档说明 `session_id` 只登记 cache reference，不会补全对话历史；请求仍需携带所需 prompt，引用属于 soft protection，内存压力下仍可驱逐，关闭 session 仅移除引用。current eviction 文档列出默认 `lru` 和其他策略，但策略不决定需要释放多少空间，也不能硬性 pin KV。
- **证据等级与边界**：论文机制和实验为 source-backed（原始论文 v2）；session/eviction 为 source-backed（访问日的动态官方文档）。不得把论文峰值推广为现行 runtime 的通用速度，也不得把 session ID 当成授权、历史存储或强制 cache pin。

### TensorRT-LLM backend migration

- **来源与版本**：[TensorRT backend removal migration guide](https://nvidia.github.io/TensorRT-LLM/latest/legacy/tensorrt-backend-removal.html) 与 [PyTorch backend architecture](https://nvidia.github.io/TensorRT-LLM/latest/torch/arch_overview.html)，访问日期 2026-11-02。迁移页显示 last updated 2026-10-26，生成自 TensorRT-LLM commit `b96ba43`。
- **实际阅读位置**：breaking change、removed/replacement table、`trtllm-serve` example 与 PyExecutor component/scheduler description。
- **来源支持的事实**：迁移页称 TensorRT engine backend 已移除、PyTorch 是唯一 execution backend；旧 `LLM(backend="tensorrt")` 会报错，`trtllm-build` 和 per-model checkpoint conversion 不再是该路径的步骤，`trtllm-serve <hf_model>` 可直接加载 HF checkpoint。架构页把 PyExecutor 拆为 Model Engine、Decoder、Scheduler，并描述 CapacityScheduler 和 MicroBatchScheduler 两步。
- **证据等级与边界**：source-backed（NVIDIA 官方页面的指定更新时间/commit）。这只描述该页面所代表的新 backend 路径；旧部署和历史版本仍可能使用 legacy TensorRT engine 流程，升级前须固定 package/image 与 release。

### LMDeploy backend matrix

- **来源与版本**：[LMDeploy stable Supported Models](https://lmdeploy.readthedocs.io/en/stable/supported_models/supported_models.html)、[TurboMind architecture](https://lmdeploy.readthedocs.io/en/stable/inference/turbomind.html) 与 [PyTorch Engine architecture](https://lmdeploy.readthedocs.io/en/stable/inference/pytorch.html)，访问日期 2026-11-02；架构页显示最近更新 2026-05-12。
- **实际阅读位置**：supported-models 的 backend/platform 表格和脚注；两种 engine 的架构和运行路径。
- **来源支持的事实**：官方支持矩阵按 TurboMind/PyTorch engine 和平台分表；TurboMind 描述 persistent batch 与 KV Cache Manager，PyTorch engine 描述由 Engine/EngineInstance、ModelAgent、Scheduler 和 RequestManager 构成的批处理路径。矩阵脚注列出具体例外，例如启用 sliding-window 的某些模型不能走 TurboMind 对应路径、特定 head dimension 会限制 TurboMind 的低位 KV cache 支持。
- **证据等级与边界**：source-backed（官方 stable 文档快照）。stable 页面和 model matrix 会变化；模型出现在一张表中不等于另一 backend、device 或精度组合可用，也不证明质量和性能。

### llama.cpp server 与 TGI lifecycle

- **来源与版本**：[llama.cpp server README on `master`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) 与 [server developer README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README-dev.md)，访问日期 2026-11-02；[TGI official docs](https://huggingface.co/docs/text-generation-inference/index) 与 [Inference Endpoints migration notice](https://huggingface.co/docs/inference-endpoints/engines/tgi)，访问日期 2026-11-02。
- **实际阅读位置**：llama-server feature list、`--parallel`/`--cont-batching`、LoRA request behavior 和 developer batching constraints；TGI maintenance banner 与迁移说明。
- **来源支持的事实**：llama.cpp server README 列出并发 multi-user slots、continuous batching、OpenAI-compatible route、JSON-schema output 等；当前 README 将 `--cont-batching` 默认设为 enabled。官方说明不同 LoRA 配置的请求不会被放进同一个 batch，因此有多 adapter 流量时批次兼容性影响性能。TGI 官方文档称其自 2025-12-11 起处于 maintenance mode，只接受小型 bug 修复、文档和轻量维护；Inference Endpoints 建议新迁移考虑 vLLM/SGLang。
- **证据等级与边界**：source-backed（项目官方文档）。llama.cpp `master` 为动态实现；TGI maintenance banner 的服务范围与 Hugging Face Inference Endpoints 的迁移建议不是所有自托管用户必须立即迁移的命令。

## 2026-11-02 批处理、排队与自适应调度

### vLLM v0.32.0 scheduler budgets and admission

- **来源与版本**：[vLLM v0.32.0 `serve` CLI](https://docs.vllm.ai/en/v0.32.0/cli/serve/)，访问日期 2026-11-02；[scheduler API](https://docs.vllm.ai/en/v0.32.0/api/vllm/v1/core/sched/scheduler/)。
- **实际阅读位置**：SchedulerConfig 中 `max_num_batched_tokens`、`max_num_scheduled_tokens`、`max_num_seqs`、`max_num_active_seqs`、`max_num_queued_reqs`、`max_num_queued_tokens`、`long_prefill_token_threshold`、`scheduling_policy`、`scheduler_reserve_full_isl`、`watermark` 和 `prefill_schedule_interval` 的说明；scheduler source 中 running/admission/token budget 的约束。
- **来源支持的事实**：固定 v0.32.0 文档把单 iteration token budget、单 iteration sequence 上限、RUNNING admission 上限、API server 侧 in-flight request cap 与 prefill prompt-token queue cap 分开定义。`max_num_seqs` 还用于 model-runner buffers/CUDA graph sizing，`max_num_active_seqs` 可收紧 admission 而不缩小 runner capacity；`max_num_queued_reqs` 到限后会让 API server 拒绝新请求，`max_num_queued_tokens` 则限制 prefill backlog，页面将其解释为 TTFT QoS 阀。`max_num_queued_tokens` 对 chunked-prefill 部分进度及 prefix-cache 命中的计数可能偏保守。v0.32.0 的 `scheduling_policy` 有 FCFS 与 priority，priority 数值越小越先处理；`watermark` 是 admission 时留出的 KV block 余量，`prefill_schedule_interval` 会间隔接纳新 prefill。
- **证据等级与边界**：source-backed（固定版本官方文档与 API/source reference）。这些是参数在 v0.32.0 的职责，不构成各参数的通用容量建议；API admission 计数与 GPU scheduler 状态有不同边界，外层 router、DP fan-out 与部署配置都会改变实际行为。

### SGLang current scheduler priority and retraction semantics

- **来源与版本**：[SGLang Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)，current 动态官方文档，访问日期 2026-11-02。
- **实际阅读位置**：Memory and scheduling argument table 的 `schedule-policy`、`enable-priority-scheduling`、`schedule-low-priority-values-first`、`priority-scheduling-preemption-threshold`、`retraction-policy`、`disable-priority-preemption`、`max-prefill-tokens` 与 `prefill-delayer-*`。
- **来源支持的事实**：文档默认 `schedule-policy=fcfs`；启用 priority 后默认较大的整数先处理，可用 `schedule-low-priority-values-first` 反转；抢占阈值定义新请求相对运行请求的最小优先级差。KV cache 满时 retraction 可按 `length`（文档说明先 retract 短输出、长输入请求）或 `priority` 选择；priority retraction 使用和排队相同的方向。页面还把调度 priority preemption 与是否关闭 priority preemption 分开呈现。
- **证据等级与边界**：source-backed（访问日动态官方参数文档）。不能据此推断所有 SGLang release 的默认值；priority 顺序、retraction 策略与实际 starvation/恢复成本需要在目标 revision 上核验。

### LAPS and P-PAS: heterogeneity inside the prefill queue

- **来源与版本**：[LAPS arXiv:2601.11589](https://arxiv.org/abs/2601.11589)，2026-01-04 提交；[P-PAS arXiv:2608.15171](https://arxiv.org/abs/2608.15171)，2026-08-15 提交；访问日期 2026-11-02。
- **实际阅读位置**：两篇预印本摘要与引言；LAPS 对 long/short prefill 分队列、temporal/spatial disaggregation 和 short-prefill batching 的说明；P-PAS 对 max batched tokens 与 scheduling pressure、prefill/decode 并发状态关系的说明。
- **来源支持的事实**：LAPS 指出即使 prefill/decode 已分离，长、短 prefill 混在同一个 prefill pool 仍可能互相干扰；其方案按 prompt 长度分离多轮长 prefill 与短 prefill，并用双队列、长度感知 batching 和 CUDA Graph clustering 组织短请求。其摘要报告的 prefill latency、SLO violation 与 throughput 改善均绑定其 LMsys-Chat-1M/多轮 workload、Qwen2.5-32B、SGLang、H200/多实例配置，不是通用增益。P-PAS 把最大 batched-token (MBT) 当作 iteration 调度预算：低调度压力时较大的 prefill chunk 可能减少执行开销；并发 prefill/decode 压力增大时较小预算可能降低对 decode 的干扰，因此静态 MBT 不保证在所有负载区间最优。其主张来自作者预印本中的特定模型、workload、GPU 和 profiling。
- **证据等级与边界**：source-backed（原始预印本的摘要/引言，未经本地复现）。两篇工作提供可比较的设计假设，不构成现有框架默认能力，也不能横向比较作者报告的不同指标。LAPS 页内仍呈现 under-review/匿名稿样式，引用时按预印本而非已发表定论处理；P-PAS 亦为 arXiv preprint。

### Agentix and Raincheck: adjacent request-level and multimodal scheduling questions

- **来源与版本**：[Agentix, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/luo)，USENIX 正式会议论文页；[Raincheck arXiv:2610.39562](https://arxiv.org/abs/2610.39562)，2026-10-29 提交；访问日期 2026-11-02。
- **实际阅读位置**：Agentix 正式论文页摘要与出版信息；Raincheck 摘要、设计动机、§3.3 latency results 和 SGLang fixed-trace appendix excerpts。
- **来源支持的事实**：Agentix 研究将 agent program 的调用依赖纳入调度上下文，而不是只看孤立 LLM request；其 USENIX 页报告在所测 agentic workloads 上，相同 latency 下 program throughput 提升 4–15×。Raincheck 通过保留初始剪枝未选中的多模态 token、把其 prefill 延后到生成期间或回答后执行来兼顾首轮延迟与后续轮次信息。其 MultiVerse 上两种模型报告 mean TTFT 降低 24.11%/17.90%、mean TPOT 增加 0.13%/0.47%；在 MT-Video-Bench 上 TTFT 降低约 11.9%–13.1%，mean TPOT 增加约 14.9%–19.9%。该论文另报告 deferred work 未必能在回答完成前做完；视频 workload 中 41.67%/68.75% 的 trace 在回答完成时尚有 deferred prefill，且提供 post-response completion time。
- **证据等级与边界**：Agentix 为正式会议论文页的 source-backed 摘要结论，数字不可推广到普通 request scheduler；Raincheck 为 arXiv preprint，结果绑定 Qwen3-VL-8B/Qwen3.5-9B、数据集、30% retention、chunk size 与作者实现。它们分别将调度上下文推向 Agent 程序依赖和多模态延迟计算，不说明通用 runtime 已支持相同能力。

## 2026-11-02 并行推理与 P/D 拓扑证据

### vLLM EP backend、DP/TP 组合与 EPLB

- **来源与版本**：[vLLM Expert Parallel Deployment](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)，主分支动态文档，访问日期 2026-11-02；[DeepEP](https://github.com/deepseek-ai/DeepEP) 项目入口。
- **实际阅读位置**：backend selection guide、single-node deployment 中 TP/DP/EP 的关系、EPLB 参数、Expert Distribution Formula、Memory Footprint Overhead、Performance Optimization 与 benchmarking。
- **来源支持的事实**：文档列出 `deepep_high_throughput` 用于多节点 prefill/high-throughput 工作负载，`deepep_low_latency` 用于多节点 decode/low-latency 工作负载；同时列出 allgather/reducescatter、FlashInfer NVLink、MoRI 等其他 backend。其单节点说明给出 EP size=TP size×DP size；TP=2、DP=4 时 expert 层由 8 rank 的 EP group 分片，attention 在每个 DP group 内采用 TP=2。EPLB 按 forward 收集负载统计并周期性重平衡；冗余 expert 权重驻留占用 GPU memory，文档用 DeepSeek-V3 举例估算每 EP rank 增加一个冗余 expert 约 2.4 GB。
- **证据等级与边界**：`source-backed`（vLLM 官方主分支动态文档）。backend 名称、平台、依赖和配置会变化；文档的 use-case 提示不构成目标硬件上的性能保证，2.4 GB 绑定所述模型/配置。不得外推到其他 runtime 或模型。

### SGLang DP-attention 阶段不齐

- **来源与版本**：[SGLang PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)，官方 current 动态文档，访问日期 2026-11-02。
- **实际阅读位置**：DP Attention Imbalance 部分。
- **来源支持的事实**：页面指出 DP attention 中一个 worker 可能同时处理 prefill batch、另一个处理 decode batch，造成 decode latency 增加。该说明支持把 worker phase skew 纳入调度观测；它不是所有 DP attention 实现的通用行为定律。
- **证据等级与边界**：`source-backed`（访问日的 SGLang 官方动态文档）；使用时需固定 SGLang revision、调度器和 P/D 配置。

### Mooncake v3 trace replay 对照

- **来源与版本**：[Mooncake v3 全文](https://arxiv.org/html/2407.00079v3)，arXiv 标注 2024-07-09；访问日期 2026-11-02。
- **实际阅读位置**：§8.1 evaluation setup、§8.1.3 Real Workload，以及系统/trace 描述。
- **来源支持的事实**：端到端评估使用与 Llama 2 70B 同架构的 dummy model；节点配置为 8 张 NVIDIA A800-SXM4-80GB、NVLink，节点间 RDMA 网络最高 800 Gbps。真实 trace 使用约 23,000 条带 arrival timestamp 请求；对照为 Mooncake 10 prefill+10 decode instances 与 vLLM 20 instances，TTFT 上限 30 s、TBT 上限 0.1 s/token。作者报告两者几乎 100% 请求达到 TTFT 门槛；Mooncake 约 100% 达到 TBT，而 vLLM 约 57%；该 trace replay 中 Mooncake 约多处理 75% 满足 SLO 的请求。
- **证据等级与边界**：`source-backed`（固定论文版本的作者报告）。结果绑定 dummy model 架构、A800/RDMA testbed、实例数量、trace 和宽松延迟阈值；不得改写成普遍的每 GPU 吞吐倍率或任意 workload 的服务保证。

### DistServe v3 与 TaiChi v1 的 SLO 结果口径

- **来源与版本**：[DistServe v3](https://arxiv.org/abs/2401.09670v3)（2024-06-06）和 [TaiChi v1](https://arxiv.org/abs/2508.01989v1)（2025-08-04）；访问日期 2026-11-02。
- **实际阅读位置**：DistServe 摘要、§6 evaluation 及 OPT-66B summarization 结果；TaiChi 摘要、SLO 区域动机及性能评估摘要。
- **来源支持的事实**：DistServe v3 报告多种模型、任务与 latency requirement 中最高 7.4× request rate 或 12.6× tighter SLO，并以超过 90% SLO attainment 为目标；其 OPT-66B/LongBench summarization 对 vLLM 的具体比较为 4.3× request rate 和 12.6× 更严格 SLO。TaiChi v1 报告在 balanced TTFT/TPOT SLO 条件下 goodput 相对 state-of-the-art 最高提高 77%；其论文机制是在 prefill-heavy/decode-heavy instance、chunk 与请求调度之间调节，使系统在共置与完全解耦之间形成 hybrid。
- **证据等级与边界**：`source-backed`（论文/预印本作者结果）。两个系统使用不同模型、trace、SLO 和指标，不可合成单一排行榜；倍数只在论文对照和 attainment 口径下成立。
