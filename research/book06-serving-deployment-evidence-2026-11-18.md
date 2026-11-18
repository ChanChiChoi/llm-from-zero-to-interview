# 第六册部署与推理工程：外部证据补充（2026-11-18）

本记录只保存本轮联网核查到的主题事实、来源、证据类型和适用边界，不记录本册审阅进度或章节取舍。动态网页按访问日理解；公开论文中的结果均绑定作者的模型、硬件、软件、数据集和测量定义。

## 1. 教学组织与服务系统主题

### 1.1 课程公开结构

- Stanford CS349D《AI Inference Infrastructure》（Spring 2026）将课程设计为研究研讨和逐步推进的 mini serving engine 项目。公开里程碑包括数据并行/张量并行、continuous batching 与 PagedAttention、context caching 与 chunked prefill，再从 speculative decoding、hierarchical cache、prefill/decode disaggregation 等高级方向中选择扩展。页面标注这些安排可能调整。
  - 来源：[Stanford CS349D](https://web.stanford.edu/class/cs349d/)
  - 证据类型：课程主页和项目里程碑；可作为主题依赖关系与实践项目设计的例子，不能证明所有课程或书籍必须采用相同顺序。
- CMU《Large Language Model Systems》2026 课程公开大纲包含生成基础、GPU 加速、serving、调度与 speculative decoding、量化等系统主题；CMU 11968 的样例大纲还列有请求调度器、continuous batching、KV/radix-tree/cache-aware load balancing、speculative decoding 和 LMCache。
  - 来源：[CMU 11-868 Spring 2026 syllabus](https://llmsystem.github.io/llmsystem2026spring/docs/Syllabus/)、[CMU 11-868 Fall 2026 syllabus](https://llmsystem.github.io/llmsystem2026fall/docs/Syllabus/)、[CMU 11968 sample syllabus](https://www.cmu.edu/online/welcome/pdfs/11968-llm-systems-syllabus-f25.pdf)
  - 证据类型：课程大纲；展示概念、执行机制和系统专题可逐层连接，不提供唯一的章节数量或适用于所有读者的课程顺序。

### 1.2 Serving 研究所覆盖的机制层次

- Orca 把生成式模型服务的调度粒度推进到模型 iteration；PagedAttention 讨论 KV 状态的分页分配和共享；Sarathi-Serve 研究 chunked prefill 与 decode 调度；DistServe、Splitwise 研究 prefill 与 decode 的资源池拆分。它们处理调度、状态管理、阶段干扰和资源组织等不同问题，论文结果各自绑定系统和测试负载。
  - 来源：[Orca, OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[Sarathi-Serve](https://arxiv.org/abs/2403.02310)、[DistServe, OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)、[Splitwise, ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf)
  - 证据类型：同行评审论文；用来划分机制问题，不可直接据论文峰值推算当前 runtime 的生产性能。
- Mooncake 的 FAST 2025 论文研究面向 KV cache 的分层内存和传输系统；其报告的成本、吞吐与延迟结果绑定论文实现、硬件和工作负载。
  - 来源：[Mooncake, FAST 2025 proceedings](https://www.usenix.org/system/files/fast25_full_proceedings_interior.pdf)
  - 证据类型：会议论文；适合作为 KV 管理从单卡分配延伸到集群传输/存储的例子。

## 2. 性能指标与压测证据

- vLLM 当前动态 Metrics 文档分别定义 streamed-output 事件间隔 `inter_token_latency_seconds` 和 request-level `request_time_per_output_token_seconds`。后者按 `(end-to-end latency - TTFT) / (output_tokens - 1)` 每请求记录一次；前者每个流式输出事件记录一次。一次输出携带多个 token（如 speculative decoding）或聚合权重不同时，两者不同。单 token 或零 token 请求的 Prometheus TPOT 记录规则与 `vllm bench serve` 的排除规则也可能造成差异。
  - 来源：[vLLM Metrics](https://docs.vllm.ai/en/latest/design/metrics/)
  - 证据类型：当前框架实现文档（2026-11-18 访问）；必须固定部署 release，不能视为跨引擎标准。
- vLLM 当前 Benchmark CLI 文档把客户端 TTFT 定义为发送请求至收到首个 streamed output；server queue time 和 server TTFT 与客户端 TTFT 之间还包含 tokenizer/chat template、多模态预处理、detokenization 与网络传输等前端开销。启用 per-request metrics 会增加 server CPU 开销，文档建议把此模式用于诊断。重复压测可能因 prefix cache 复用而高估吞吐。
  - 来源：[vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
  - 证据类型：当前框架 benchmark 文档（2026-11-18 访问）；计时边界与 cache reset 应和被测 workload 一起记录。
- OpenTelemetry GenAI metrics 约定目前标注为 Development。它给 server request duration、request-level time-to-first-token 和成功响应的 time-per-output-token 提供建议性语义；其 TTFT/TPOT 服务端边界不能自动替代客户端体验指标，也不能被当作已冻结的跨产品合同。
  - 来源：[OpenTelemetry GenAI metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md)
  - 证据类型：持续演进的语义约定（2026-11-18 访问）。
- MLPerf Inference 规则区分 Single Stream、Server、Offline、MultiStream 等场景，并规定数据集、质量门槛、延迟/吞吐计量及整条系统流程的规则；它说明 benchmark 结果依场景和完整测量合同而异。
  - 来源：[MLPerf Inference Rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)
  - 证据类型：公开 benchmark 规则；适用于理解标准化比较的组成，不意味着内部压测必须采用 MLPerf 场景。

## 3. 推理平台扩缩容和请求路由

- KServe 0.20 Workload Variant Autoscaler（WVA）文档给 LLMInferenceService 提供基于 KV cache 利用率、队列深度和请求饱和度的扩缩容路径，可由 HPA 或 KEDA 执行动作；对于 disaggregated prefill/decode 部署，prefill 与 decode 可分别设伸缩范围，但两者必须使用相同 actuator 后端。其文档还列出模型加载初始 cooldown、idle scale-down、指标故障 fallback 等配置差异。
  - 来源：[KServe 0.20 LLMInferenceService autoscaling](https://kserve.github.io/website/docs/0.20/model-serving/generative-inference/llmisvc/autoscaling/llmisvc-autoscaling)
  - 证据类型：特定 KServe 版本实现文档（2026-11-18 访问）；不是所有 Kubernetes/serving stack 的默认行为。
- Kubernetes Gateway API Inference Extension 的 InferencePool API 自 v1.0.0 标为 GA；其 Endpoint Picker Extension 可跟踪模型服务的 KV-cache 利用率、待处理队列和活跃 LoRA adapter 等信息，并据此选择模型服务副本。Pod 需要提供该项目定义的 model-server protocol。
  - 来源：[InferencePool API](https://gateway-api-inference-extension.sigs.k8s.io/api-types/inferencepool/)
  - 证据类型：当前项目 API 文档（2026-11-18 访问）；路由能力依实现和协议接入，不能据此推定任何 InferencePool 都能作 KV-aware routing。

## 4. 多模态输入的 CPU、显存与安全边界

- vLLM 当前多模态输入文档说明，`pyav` 逐帧 Python generator 在并发解码时可能受 Python/C 调用与 GIL 争用影响；`torchcodec` 单次处理完整 stream 并释放 GIL。`auto` 的具体 fallback 受 CUDA、CPU、XPU、ROCm/TPU 构建和系统 FFmpeg 等环境影响。
  - 来源：[vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/)
  - 证据类型：动态框架文档（2026-11-18 访问）；应在目标部署平台上测解码 backend、CPU 利用率和端到端时延。
- vLLM 当前 Security 文档列出远程媒体下载、解码尺寸、图像像素数、单请求解码总字节、音频文件大小/时长/PCM bytes 和 dense embedding bytes 等输入限制，用于降低压缩媒体或稀疏 embedding 展开造成的资源耗尽风险；将相应限制设为 0 会停用保护。
  - 来源：[vLLM Security](https://docs.vllm.ai/en/latest/usage/security/)
  - 证据类型：动态框架默认配置文档（2026-11-18 访问）；这些是 vLLM 参数与默认值，其他 runtime 需要检查各自的解码和资源限制。

## 5. FP4 KV Cache：动态实现与前沿结果

- SGLang 当前 `quantized_kv_cache` 文档把 FP4 KV 明确标为 experimental，列出 NVFP4 和 `fp4_mx_block16` 两种 E2M1 配置；后者与 OCP MXFP4 block-size-32 格式不同。页面记录了特定架构/数据集的精度差异：例如 GPT-OSS-120B 的 AIME25 样例中 BF16、FP8 E4M3、FP4 E2M1 分别为 0.7533、0.7667、0.3533；GPQA Diamond 分别为 0.5081、0.5434、0.3202。文档提醒复杂任务和长上下文需自行评估。
  - 来源：[SGLang Quantized KV Cache](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/quantized_kv_cache.mdx)
  - 证据类型：动态主分支实现和初步评测文档（2026-11-18 访问）；版本、硬件、模型与数据集绑定，不能将“支持 dtype”当作生产成熟度或质量保证。
- NVFP4-MLA 预印本（2026-11-13 提交）提出面向 MLA 的 FP4 KV 存储与 attention kernel 联合设计。作者报告在 DeepSeek-R1/Kimi-K3 的九个 benchmark 上平均精度匹配 BF16 MLA、KV 相对 BF16 压缩约 3.20x/3.05x；Rubin 上的 kernel、p90 TTFT 和 per-GPU throughput 结果仍是作者自己的模型/硬件/系统设置。
  - 来源：[NVFP4-MLA, arXiv:2611.18972](https://arxiv.org/abs/2611.18972)
  - 证据类型：未同行评审预印本；摘要结果不能推导一般模型兼容、生产可用或独立复现。
- BitDecoding-FP4 预印本（2026-11-06 提交）报告其 NVFP4/MXFP4 KV 精度与 BF16 KV 的差距、以及 NVIDIA B300 上 Qwen3 模型 serving throughput；作者将结果绑定其量化算法、kernel、模型和平台。
  - 来源：[BitDecoding-FP4, arXiv:2611.09825](https://arxiv.org/abs/2611.09825)
  - 证据类型：未同行评审预印本；作者报告的数据不代表其他 runtime、GPU 或任务。

## 6. GPT-6.2 Astra 当前公开 API 合同（核对日：2026-11-18）

- 模型页公开 `gpt-6.2-astra` 的 1,050,000 context window、922,000 max input、128,000 max output，知识截止日期 May 18, 2026；接受文本和图像输入、返回文本；音频和视频不支持。页面列出 Responses 与 Chat Completions 等 endpoint，并说明模型页所列内建工具在 Responses API 下支持。模型拒绝 `reasoning.effort=none`。
  - 来源：[GPT-6.2 Astra model page](https://developers.openai.com/api/docs/models/gpt-6.2-astra)
- 迁移指南要求工具调用使用 Responses API，即使模型也支持 Chat Completions；迁移时应省略部分不支持参数，并按模型与请求分别核对数据驻留、prompt cache、存储和处理档位合同。
  - 来源：[Using GPT-6.2](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.2-astra)
- 短上下文 Standard 公布价为每百万 input $8、output $40、cached input $0.40、30 分钟 cache write $10、12 小时 write $20。超过 272K input token 时，整次请求 input/cache rate 倍增、output rate 为短上下文 Standard 的 1.5 倍；具体账单仍需看费率页、服务档位、区域和账户合同。Batch 为 Standard 的一半、Fast 为两倍；GPT-6.2 Ultrafast 至 2027-02-28 按 Standard 三倍，自 2027-03-01（Pacific time）起按六倍。eligible background completion window 对 GPT-6.2 Astra 的模型 token 费率在 2 小时窗口低 30%、8 小时窗口低 40%。
  - 来源：[OpenAI API pricing](https://developers.openai.com/api/docs/pricing)、[GPT-6.2 Astra model page](https://developers.openai.com/api/docs/models/gpt-6.2-astra)
  - 证据类型：官方动态费率页面（2026-11-18 访问）；地区、时间和合同可能影响实际账单，教学算例以 Standard global short-context rates 为准。
- Reasoning 指南公开 standard/pro reasoning mode 与 `reasoning.effort` 是不同维度；支持的 GPT-6 模型可在 standard、single-agent 对话中用 `configuration_update` 更新后续 effort。配置更新可与 `context_management` 自动 compaction 同用，服务端会在 compaction item 后发出最新设置；显式 `compaction_trigger` 后要补一条更新。自动 truncation 和独立 `/responses/compact` 不接受含更新项的历史；独立 compact 流程需先保存最新设置、从送入的历史中移除更新项、再在返回窗口中恢复设置。手工 replay 时还要留意 GPT-6.2 对前缀编辑、model、user/safety identifier 和 organization 变化的 reasoning-item 清除规则。
  - 来源：[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)、[GPT-6.2 migration guide](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6.2-astra)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)
- Async tool calling 由应用执行 function/custom tools，要求应用保存 call/task 状态并回传结果。Steering 的 `accepted` 事件只代表输入已排队，后续 successor `response.created` 是该输入进入续接响应的提交点；若响应停在应用侧工具结果或审批，父响应完成后会收到 `response.steer.pending` 和 `required_input`，应用应填入已保存的结果/审批，并沿同一 WebSocket lane 对每个父响应显式续接一次，不能重跑工具或重发 steering。Background Responses、Async tool jobs 与 standalone compaction 管理不同状态；GPT-6.2 的 Responses tool calling、公共 beta background windows 与当前模型 ID/资格需分别核对。
  - 来源：[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling)、[Steering](https://developers.openai.com/api/docs/guides/steering)、[Responses WebSocket events](https://developers.openai.com/api/reference/resources/responses/websocket-events)、[Background mode](https://developers.openai.com/api/docs/guides/background)、[Compaction](https://developers.openai.com/api/docs/guides/compaction)
  - 证据类型：官方 API 文档（2026-11-18 访问）；文档示例不替代目标 organization、endpoint 与 model ID 的实时可用性确认。
