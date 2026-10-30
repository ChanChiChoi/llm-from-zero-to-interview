# KV Cache 容量、分页管理与压缩证据

检索与阅读日期：2026-10-29。本文只记录来源支持的知识点和证据边界，不记录书册审阅状态或正文修改过程。

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
