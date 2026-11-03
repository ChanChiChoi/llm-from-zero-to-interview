# KV Cache 容量、分页管理与压缩证据

检索与阅读日期：2026-10-29—2026-11-02。本文只记录来源支持的知识点和证据边界，不记录书册审阅状态或正文修改过程。

## 1. PagedAttention

- **来源与版本**：Woosuk Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention,” SOSP 2023，arXiv:2309.06180v1。来源：[arXiv 摘要与版本](https://arxiv.org/abs/2309.06180v1)、[论文 HTML](https://arxiv.org/html/2309.06180)。
- **阅读位置**：论文摘要、§§1–4.6、§§6–7、§10。
- **来源支持的事实**：论文把每条序列的 KV 状态切成固定 token 数的逻辑 block，并允许它们映射到不连续物理显存；attention kernel 依映射读取，KV cache manager 分配物理块，scheduler 负责请求批次与抢占。按需分配可避免为最大生成长度保留连续空间，块级引用也允许部分请求共享前缀或解码分支。块越大，kernel 可并行处理的位置更多，但请求尾部潜在空槽也会增多。论文中的系统吞吐结果来自 vLLM 与特定历史 baseline、模型、GPU、到达率和 workload 的比较。
- **证据等级与边界**：`source-backed`。支持原论文的方法和其 2023 评测结论，不能单独说明当前 vLLM 或其他 serving 引擎的代码、性能、默认 block 大小或调度策略。

## 2. vLLM 当前 Paged Attention 文档的时效说明

- **来源与版本**：[vLLM Paged Attention 设计页](https://docs.vllm.ai/en/latest/design/paged_attention/)，`latest` 可变文档，访问日期 2026-10-29。
- **阅读位置**：页面标题后的 Warning 和 Inputs/Concepts 开头（网页行 497–507）。
- **来源支持的事实**：页面明示其内容基于原始论文，是历史文档，已不描述当前 vLLM 代码实现；并指出当前代码使用与 paged KV cache 布局配合的 attention kernel。
- **证据等级与边界**：`source-backed`。页面说明本身是时效边界，不应用来反推当前 kernel 的完整实现。研究某个发布版本必须固定相应 release 或源码 revision。

## 3. KVSET 工作集容量分析

- **来源与版本**：Luchang Li et al., “The KV Cache Working Set: Online Capacity Planning for LLM Inference Systems,” arXiv:2609.27746v1，提交于 2026-09-23；作者单位列为 Kingsoft Cloud。[论文版本](https://arxiv.org/abs/2609.27746v1)、[PDF](https://arxiv.org/pdf/2609.27746)。
- **阅读位置**：摘要、§§1–3、§5 和限制说明。
- **来源支持的事实**：KVSET 关注 page-granular LRU **prefix cache** 的工作集，即为达到目标命中率需保留的容量；用 Mattson stack distance 与 Fenwick tree 从访问 trace 估算多个 cache 容量下的命中率。论文使用 10,000 条内部 coding-agent trace，在 SGLang/Mooncake、NVIDIA H20、GLM-5.2 W4A8、FP8 KV（约 60 KB/token）设置中，把估计与已部署 cache 在部分容量点上的测量作比较。另一个 24,000 请求的内部 trace 展示不同 request-coverage 目标对应显著不同的推荐容量；作者报告该工作负载在 95% 和 99% coverage 下分别约需 1 TiB 和 5 TiB，99.9% coverage 的容量在观测 trace 中没有收敛。论文说明实现仅分析 LRU eviction 和模型的 full-attention KV 部分。
- **证据等级与边界**：`source-backed`。以上容量和命中结果属于论文所用的内部 coding-agent traces、模型、引擎、cache policy 与 coverage 定义；不能外推至其他工作负载。它估算跨请求可复用前缀的留存容量，不替代根据活动请求 token 总数计算的单请求/批次 KV payload 公式。

## 4. UltraQuant 4-bit KV serving

- **来源与版本**：Inesh Chakrabarti et al., “UltraQuant: 4-bit KV Caching for Context-Heavy Agents,” arXiv:2606.20474v3，最后修订于 2026-09-11；arXiv 评论列出 EMNLP 2026 Industry Track。[论文 v3](https://arxiv.org/abs/2606.20474v3)、[HTML 全文](https://arxiv.org/html/2606.20474)。
- **阅读位置**：摘要、§§2.1–3、§§5–9、附录 A.3、B.3–B.4、C 和 E。
- **来源支持的事实**：作者研究 context-heavy agent serving 中的 4-bit KV，讨论旋转/量化、FP4 加组尺度的硬件路径、任务质量和缓存驻留/服务吞吐。自适应 SLO 实验使用匿名化且来源于 Claude Code 生产 sessions 的 trace，模型为 MiniMax M2.5 与 Qwen3-235B，AMD MI355X、TP=2；并发会在 6–48 sessions 间增加，滚动 P95 TTFT 门槛为 120 秒，评估窗口 600 秒。作者报告 UltraQuant 的 TTFT-qualified goodput 分别为 BF16 的 2.71× 与 4.38×；同实验的 FP8 KV 分别为 2.18× 与 2.16×。论文另以 MiniMax M2.5、TP=2、2×MI355X、32K/1K 输入/输出、并发 64 测 decode kernel/throughput；结果不是跨硬件通用速度系数。论文称 UltraQuant 的相对 FP8 收益主要出现在 context/concurrency 足以超过 FP8 resident-cache capacity 的情形；低压力短上下文不能保证出现该优势。Retrieval 与 reasoning 质量结果依模型和任务变化。
- **实现背景**：该工作区分 Ultra-TQ（codebook/dequant path）与 UltraQuant（FP4 E2M1、UE8M0 group scales、AMD CDNA4 scaled-MFMA 路径）；4-bit 表示并不自动意味着更快，解量化 kernel 和 cache capacity 都会影响端到端结果。
- **证据等级与边界**：`source-backed`。属于作者报告的论文实验，样本为 production-derived replay 而非公开线上基准；作者主要来自 AMD，方法含特定 AMD GPU kernel 优化。性能不能外推到其他 GPU、服务框架、SLO、模型或请求分布；低比特缓存的任务质量应在目标模型和任务上单独评测。

## 5. Tessera 检索驱动的跨前缀 KV 复用

- **来源与版本**：Fei Fang et al., “Tessera: Demand-Driven KV Cache Management for Retrieval-Augmented LLM Serving,” arXiv:2609.32999v1，提交于 2026-09-26。[固定版本摘要](https://arxiv.org/abs/2609.32999v1)、[HTML 正文](https://arxiv.org/html/2609.32999v1)。
- **阅读位置**：摘要、§§2.2–2.4 的可组合 KV 动机、§3.2 的非前缀重复、§§4–5 的系统与缓存/路由设计、§6 的评估结果、§8 的兼容性与限制。
- **来源支持的事实**：RAG 文档块和 Agent memory 记录可以在不同请求中重复出现，但前置上下文或 RoPE 位置变化会使常规 prefix cache 无法复用。论文提出由检索提供所需 context unit，再结合其 KV 驻留和生成负载决定请求路由、共享状态获取及缺失状态重算；generation node 可并发处理本地、远端和缺失状态，并把新状态保留移出请求关键路径。论文在自己的 RAG 与 Agent memory 工作负载上、按相同请求率比较，报告 mean TTFT 最高降低 3.6×，并称答案质量与底层组合策略相当。
- **证据等级与边界**：`source-backed`。这是 arXiv v1 原型报告，不是生产服务的普遍性能保证。论文兼容范围为 RoPE 全注意力模型、常规 per-token KV 和所述组合/重算策略；hybrid attention、MLA、non-RoPE 需要扩展且未评估。多租户需把命名空间对齐到授权可见范围并在解析 KV 前授权；论文所用 Agent memory snapshot 固定，在线变更/撤销的完整语义不在其评估范围。

## 6. SmartGen 选择性 KV 传输

- **来源与版本**：Xuchuan Luo et al., “SmartGen: Seamless Disaggregated LLM Inference with Selective KV Cache Transfer,” arXiv:2607.28150v1，提交于 2026-07-30。[固定版本摘要](https://arxiv.org/abs/2607.28150v1)、[HTML 正文](https://arxiv.org/html/2607.28150v1)。
- **阅读位置**：摘要、§§3–4 的传输瓶颈和三条数据路径、§5.1 的工作负载与硬件设置、§5.2 的 TTST/TBT、§5.5 的 LongBench 准确率。
- **来源支持的事实**：SmartGen 使用 profile-based proactive transfer 先发位置上较重要的 KV，Decode 阶段并行按需获取缺失 KV，并在空闲资源上 speculative-transfer 其余项。其评估使用 L20/eRDMA 实例，包含 MultiFieldQA、GovReport、SAMSum、LCC 等长上下文工作负载；论文报告 GovReport 上 TTST 最多达到 full-transfer 对照的 4.3× 降低，并称后续 decode 与 LongBench 准确率接近全量 KV 基线。
- **证据等级与边界**：`source-backed`。4.3× 绑定论文的模型、GovReport 请求和 L20/eRDMA testbed；准确率来自所测 LongBench 子集与具体 KV 选择算法。不能解释为通用 P/D serving 加速，也不能只看 TTST 而忽略后续 TBT、质量与最终状态完整性。

## 7. 固定版本运行时文档与 cache 层差异

- **来源与版本**：[Transformers v5.17.0 cache strategies](https://huggingface.co/docs/transformers/v5.17.0/en/kv_cache)、[vLLM v0.32.0 Engine Arguments](https://docs.vllm.ai/en/v0.32.0/configuration/engine_args/) 和 [v0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)，访问日期 2026-10-29。
- **来源支持的事实**：Transformers v5.17.0 文档把 DynamicCache 描述为随生成增长；对 sliding-window 或 chunked attention 层，cache 达到窗口/chunk 上限后不再继续增长。该版本还提供 offloaded、quantized cache 路线，文档明确量化 cache 在短上下文、显存充足时可能损害 latency。vLLM v0.32.0 EngineArgs 将 `gpu_memory_utilization` 定义为当前实例的设备内存使用上限比例，并可由该限制推算 KV 容量；同一文档列出多种 `kv_cache_dtype` 选项，包括平台/模型相关的 FP8、NVFP4、UltraQuant 与其他低位宽类型。v0.32.0 release notes 说明 Model Runner V2 成为默认路径。
- **历史文档边界**：[vLLM v0.32.0 Paged Attention 设计页](https://docs.vllm.ai/en/v0.32.0/design/paged_attention/) 明示该页是基于原始论文的历史说明，不再描述当前 vLLM 代码；它并指出当前代码使用兼容 paged KV layout 的 MQA attention kernel。PagedAttention 原论文支持分页机制动机，不能替代指定 release 的当前实现文档或 runtime 结果。
- **证据等级与边界**：source-backed（固定版本官方文档及 release notes）。cache 上界、dtype 可选项和参数语义绑定所标版本；某项可配置不表示对所有 GPU/模型组合可用，也不证明质量或性能收益。不同层可以有不同的历史状态长度，full-attention 的统一 (T) 公式不能直接覆盖滑窗、chunked 或 hybrid state。

## 8. vAttention ASPLOS ’25 论文 v3

- **来源与版本**：Ramya Prabhu et al., “vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention,” arXiv:2405.04437v3，2025-01-29 修订；正式发表于 ASPLOS ’25，DOI [10.1145/3669940.3707256](https://doi.org/10.1145/3669940.3707256)。[固定版本全文](https://arxiv.org/html/2405.04437v3)、[摘要](https://arxiv.org/abs/2405.04437v3)。
- **阅读位置**：摘要、§§5–7，重点核对 §7 的模型/硬件、kernel baseline 和端到端结果。
- **来源支持的事实**：评测使用 vLLM v0.2.7 作为共同框架，集成 FlashAttention-2 v2.5.9、FlashInfer v0.4.0；模型为 Yi-6B（1 张 A100）、Llama-3-8B 与 Yi-34B（各 2 张 A100）。以 FlashAttention-2 非分页 kernel 配合 vAttention 时，论文报告 decode throughput 相对 vLLM 最高 1.99×。在长上下文端到端设置中，相对分页 FlashAttention-2 和 FlashInfer kernel，throughput 分别最高 1.18× 与 1.23×；decode 图以每条请求初始上下文 16K token、400 次迭代的平均 latency 计算。
- **证据等级与边界**：source-backed。1.99× 的 baseline 同时改变 attention kernel，不能归因于 VMM 分配机制本身；各数值只属于论文给定模型、A100 数量、框架/内核版本和请求设置，不能外推到其他硬件或 runtime。

## 9. vLLM v0.32.0 KV 管理文档

- **来源与版本**：[Paged Attention Design](https://docs.vllm.ai/en/v0.32.0/design/paged_attention/)、[Automatic Prefix Caching Design](https://docs.vllm.ai/en/v0.32.0/design/prefix_caching/)、[Quantized KV Cache](https://docs.vllm.ai/en/v0.32.0/features/quantization/quantized_kvcache/)，访问日期 2026-11-02。
- **阅读位置**：Paged Attention 页首 Warning；Prefix Caching 的 block hash、full block、cache salt 和数据结构段；Quantized KV Cache 的 FP8 scheme、scale calibration 与 layer skip 段。
- **来源支持的事实**：Paged Attention Design 页明确说明它基于原始论文，是历史说明，不再描述当前 vLLM 代码。Prefix Caching 页说明 v1 block hash 使用 parent hash、当前 block token 和 extra hashes；extra hash 可包含 LoRA ID、多模态处理 hash、cache salt；只缓存完整 block。可选的 per-request cache_salt 被注入首 block hash，以限制不同 salt 间共享，文档将其描述为降低 timing-based attack 的措施。Quantized KV Cache 页列出 per-tensor 和 per-attention-head FP8 scales；该版本 per-head 路径仅支持 Flash Attention backend 且要求 llm-compressor calibration；未校准路径默认 scale 为 1.0，FlashAttention 3 路径也会把 Q 纳入 FP8 attention，并提供跳过敏感层量化的选项。
- **证据等级与边界**：source-backed，绑定 v0.32.0 文档。页面公开能力不代表任何模型/backend/GPU 组合均受支持；cache_salt 是 reuse namespace 隔离方式，不等价于加密、删除证明或完整威胁模型防护；历史 kernel 讲解页不作当前实现证据。

## 10. 有损保留策略：H₂O 与 StreamingLLM

- **来源与版本**：[H₂O arXiv:2306.14048v2](https://arxiv.org/html/2306.14048v2)、[StreamingLLM arXiv:2309.17453v2](https://arxiv.org/html/2309.17453v2)。
- **阅读位置**：H₂O 摘要、问题定义、§§3–4 的 cumulative-attention heavy hitters 和保留策略；StreamingLLM 摘要、§§3.1–3.2 的 attention sink 观察与 rolling-cache 方法。
- **来源支持的事实**：H₂O 将累计 attention score 用作 heavy-hitter 代理，并把其与近期 KV 共同保留；论文把这一方法研究为有固定 KV 预算的逐 token eviction policy。StreamingLLM 将初始 attention-sink token 与近期窗口一起保留；摘要报告其评估的 Llama-2、MPT、Falcon、Pythia 等模型在流式语言建模中可稳定处理很长输入，并指出其方法与长文档任意位置检索不是同一个目标。论文摘要报告最多 4 个初始 token 的配置，但该数字属于其模型和实验设置。
- **证据等级与边界**：source-backed。二者通过删除部分旧状态改变后续可见历史，属于有损 eviction，不等于无损 host offload 或数值量化；实验质量不支持对所有任务、模型或随机检索能力的保证。

## 11. 多租户 prefix 共享侧信道

- **来源与版本**：Guanlong Wu et al., “I Know What You Asked: Prompt Leakage via KV-Cache Sharing in Multi-Tenant LLM Serving,” NDSS 2025。[会议论文页](https://www.ndss-symposium.org/ndss-paper/i-know-what-you-asked-prompt-leakage-via-kv-cache-sharing-in-multi-tenant-llm-serving/)、[论文 PDF](https://dev.ndss-symposium.org/wp-content/uploads/2025-1772-paper.pdf)。
- **阅读位置**：论文摘要和攻击场景概述。
- **来源支持的事实**：作者提出 PROMPTPEEK，研究互不信任用户间共享相同 token sequence 的 KV 状态可能暴露的 timing side channel，并在三类具有不同先验信息假设的场景中展示 prompt reconstruction。
- **证据等级与边界**：source-backed。它支持“cache sharing 可能产生可利用的侧信道”这一威胁模型，不证明所有 runtime、配置、租户边界或部署环境都可按同样成功率攻击；缓存盐值、鉴权与日志审查需要结合目标系统单独评估。

## 12. TensorRT-LLM KV Cache System 动态文档

- **来源与版本**：[TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)，页面未显示本文使用的固定 release，访问日期 2026-11-02。
- **阅读位置**：Reuse Across Requests、Eviction、Selecting the KV Cache Manager 和该页的限制说明。
- **来源支持的事实**：页面描述 radix search tree、优先级 LRU、primary KV state 淘汰到 secondary memory 的 offload/reuse；还提示访问时所述实现仅能淘汰 radix-tree leaf，并认为其对 limited-attention layers 不理想。该页面同时介绍 V1/V2 cache manager 和多种模型的 manager 选择。
- **证据等级与边界**：source-backed，但文档为动态页面。leaf-only caveat 在页面中没有绑定到明确 manager/release，因此不能把它外推成所有 TensorRT-LLM 版本或配置的普遍行为；部署结论应核对目标 release 和实际 manager。

## 13. InnerQ inner-dimension KV cache quantization

- **来源与版本**：[InnerQ arXiv v2](https://arxiv.org/abs/2602.23200v2)，2026-05-20 修订，访问于 2026-11-02。[HTML 正文](https://arxiv.org/html/2602.23200)。
- **阅读位置**：摘要、§§3.3、4.1–4.4、§5.3、Tables 4–5 与结论。
- **来源支持的事实**：InnerQ 沿 quantized K/V 矩阵的 inner dimension 分组，让反量化 GEMV 复用 scale；方法包含按组选择对称/非对称的 hybrid quantization、sink/recent 高精度窗口，以及在 K 上执行并折叠进模型权重的 per-channel normalization。作者报告在 Jetson Xavier NX 上，batch=1、单层 fused dequantize-GEMV 测量相对 FP16 cache 平均 2.7× 加速；其评测设置说明经过 10 次 warm-up、记录 100 次测量。该数据是特定 kernel microbenchmark，不是完整模型、端到端请求或服务吞吐。
- **来源内部差异**：§5.3 把该延迟图的单层模型写为 Llama 3.2-8B，而 Table 4 的同组 latency 标题写为 Llama 3.1-8B；Table 5 延迟分解也使用 Llama 3.1-8B。型号存在内部不一致，引用时不应将表述合并为一个确定版本。
- **证据等级与边界**：source-backed（作者预印本 v2）；作者报告的 kernel 速度和任务指标绑定该平台、batch、序列长度和 kernel 实现。没有据此确认第三方 runtime 支持、线上服务收益或其他硬件上的性能。

## 14. TensorRT-LLM KV Cache Compression 动态文档

- **来源与版本**：[KV Cache Compression 文档](https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html)，访问于 2026-11-02；页面标注最后更新 2026-10-26、文档生成 commit b96ba43。
- **阅读位置**：Overview、When Compression Runs、Cold-Page Quantization、TriAttention、Support 和页面末尾硬件条件。
- **来源支持的事实**：文档把两类机制分开：NVFP4 cold-page quantization 在 KV Page 从 GPU 移到 Host/Disk 时编码冷层表示，回到 GPU 后在 attention 复用前解码成模型 runtime KV type；GPU 上 active attention 不直接读取冷层 NVFP4。TriAttention 周期性为生成 KV token 打分、保留选择出的 token 并物理压紧 cache，prompt 保持不变，之后仍由标准 attention 读取 compacted cache；它需要模型专属离线 calibration file。文档当前列出的共同条件为 PyTorch backend、KVCM V2、SM100/SM103/SM107；NVFP4 cold-page 另需 native C++ KVCM V2 和非零 Host/Disk cache。表格列出 cold-page 与 TriAttention 在 MHA/MQA/GQA 上支持情况不同，TriAttention 不支持 MLA。
- **证据等级与边界**：source-backed（NVIDIA 当前官方文档）。页面动态更新，条件只代表访问时列出的 TensorRT-LLM 路径；cold-page 表示冷层数据编码，TriAttention 改变被保留的 token 集合，两者均不能泛化为所有 runtime 的 active KV 量化行为。

## 15. SpectrumKV 混合精度 KV transfer

- **来源与版本**：Pengju Yang, “SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill–Decode Disaggregated LLM Serving,” arXiv:2606.08635v1，2026-06-07 提交；访问日期 2026-11-02。[固定版本全文](https://arxiv.org/html/2606.08635v1)、[摘要](https://arxiv.org/abs/2606.08635v1)。
- **实际阅读位置**：摘要、§§2–4 的预算与 token precision assignment、§5 的实验模型/数据/指标、§§6.1–6.9 的 PPL/检索/延迟结果、§9 limitations。
- **来源支持的事实**：SpectrumKV 按 token 重要性在 FP16、INT8、INT4 之间分配 transfer precision；INT4 tolerance 通过轻量部署探测判定。论文结果中 Qwen2.5-7B 的 INT4 probe 3 次均失败，退回 FP16+INT8；Mistral-7B-Instruct-v0.3 和 Gemma-2-9B-it 的 probe 均 3/3 通过。seq=2048、50% normalized KV budget 下，WikiText-2 PPL 相对 full FP16 KV 的变化：SpectrumKV Greedy 对 Qwen/Mistral/Gemma 分别为 +1.97%、−0.06%、−0.44%，PDTrim 分别为 +25.85%、+22.07%、+35.63%。作者另报告该预算下 sink-protection 变体、不同 budget、NIAH retrieval 和 transfer-path TTFT 数据，均为各自实验口径。
- **证据等级与边界**：`source-backed`（arXiv v1 作者实验）。以上 PPL 数字绑定三种 7B—9B 模型、seq=2048、WikiText-2、b=0.5 与作者实现；quality delta 不能替代端到端 decode/服务质量。Qwen INT4 失败说明模型间 tolerance 不可外推，需对部署模型单独探测；TTFT 结果也不能单独代表 SLO goodput。

## 16. vLLM FP8 KV Cache 与 attention 的工程验证报告

- **来源与版本**：vLLM 团队，“The State of FP8 KV-Cache and Attention Quantization in vLLM”，官方工程博客，2026-04-22。[报告全文](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)，访问日期 2026-11-02。
- **实际阅读位置**：引言、Kernel and vLLM Improvements、Performance Benchmarking 的 Tables 1–5、Accuracy Benchmarking、When to Avoid FP8 KV-Cache。
- **来源支持的事实**：报告明确说明 `--kv-cache-dtype fp8` 在 FlashAttention 3 路径中会让 QK 与 score-value attention matmul 也在 FP8 E4M3 域执行，因此该路径不等于只改变 KV 存储字节数。H100/FA3、Llama-3.1-8B 的 128K needle-in-a-haystack 评测中，BF16 为 91%、早期 FP8 路径为 13%；two-level accumulation 把部分和写入真正的 FP32 register 后恢复到 89%，但增加寄存器压力并拖慢部分长 prefill。报告的质量评测使用 per-tensor、未校准 `scale=1.0`，其质量结果不能直接当作 calibrated scale 或其它 attention/backend 的保证。
- **来源支持的性能数据**：H100 单请求 Llama-3.1-8B 的拟合 ITL slope/intercept 从 BF16 的 `4.37×10^-5 ms/token`、`6.44 ms` 变为 FP8 的 `2.37×10^-5 ms/token`、`6.58 ms`；表中报告约 7,010-token break-even。H100 上 gpt-oss-20b 的历史 v0.10.2 FP8 slope 为 BF16 的 96%、break-even 741,565 token；报告的修复后 v0.19.1 全 FP8 为 80%、22,109 token，跳过 128-token sliding-window 层后为 71%、7,659 token。并发 8、150 个约 20K input/2K output 请求的 Llama-3.1-8B throughput 试验中，output throughput 从 450.3 增至 517.5 token/s，median ITL 从 15.18 降至 12.93 ms。报告还给出 B200/FlashInfer 与 `head_dim=256` 模型结果，显示 backend、硬件、滑窗比例和 head dimension 会改变性能/质量取舍。
- **证据等级与边界**：`source-backed`，属于作者团队的工程验证报告，不是跨 runtime 的独立 benchmark。所有数字绑定报告指定的模型、H100/B200、FlashAttention 3/FlashInfer、并发、上下文长度和软件路径；未校准 scale 下的报告结果不能代表所有校准方案。ITL slope、break-even、TTFT 和并发 output throughput 口径不同，不可互换或按单一倍数外推。
