# LLM 推理服务：注意力、prefill 调度、推测解码、前缀缓存与量化证据

检索与阅读日期：2026-11-05。此记录只保存联网核验得到的技术事实及其证据范围。

## 推测解码

- [Leviathan、Kalman、Matias，Fast Inference from Transformers via Speculative Decoding，ICML 2023 / arXiv:2211.17192](https://arxiv.org/abs/2211.17192)：论文描述由近似模型提出候选、目标模型并行验证并通过修正采样保持目标分布的方法；作者在 T5-XXL/T5X 实验中报告 2–3 倍加速。该数值属于论文配置，不能外推到其他模型、批量、硬件或推理引擎。
- [Liu 等，Speculative Decoding: Performance or Illusion?，arXiv:2601.11580](https://arxiv.org/abs/2601.11580)：2026 年预印本对 vLLM 中的多种 speculative decoding 变体、工作负载和批量配置做系统研究。作者报告接受长度随输出位置、请求和数据集变化，目标模型验证会占据显著执行成本，实测表现与理论上界之间存在差距。证据范围仅限论文研究的引擎版本、方法、模型和负载；不能据此断言 speculative decoding 普遍无效。
- [Xia 等，Acceptance-Aware Draft Model Training for Speculative Decoding，arXiv:2609.24150](https://arxiv.org/abs/2609.24150)：2026 年预印本提出按 decoding mode 优化接受长度的 draft-model 训练目标，并报告在其目标模型、draft 模型、任务和采样设置下改善接受长度。公开摘要支持“训练目标可直接优化接受行为”这一研究方向；它不能单独证明线上延迟或吞吐必然改善，也不能外推到未评估的模型和 runtime。

## vLLM 前缀缓存与量化文档

- [vLLM Automatic Prefix Caching（stable 文档）](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)：官方文档说明，具有相同 token 前缀的新请求可复用已有 KV Cache，从而跳过共享前缀的 prefill；该机制不缩短 decode，生成阶段占主导或没有可复用前缀时收益有限。`stable` 文档会随版本变化；部署时应固定引擎版本并在目标模型和负载上测量。
- [vLLM Quantization（stable 文档）](https://docs.vllm.ai/en/stable/features/quantization/)：官方文档列出若干量化格式、实现和硬件支持矩阵，并明确兼容关系会随项目变化。文档中的格式或硬件支持不能证明某一具体模型、序列形状和任务集已满足质量、延迟或成本目标，仍须对目标配置做基准和质量回归。

## Prefill 分块与解耦调度

- [Agrawal 等，Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve，arXiv:2403.02310v3](https://arxiv.org/abs/2403.02310)：论文提出 chunked-prefill 与 stall-free batching。调度在 iteration 级优先容纳正在运行的 decode，再按 token budget 加入 prefill chunk；作者依据目标 TBT/SLO 和测量结果确定预算。论文的吞吐增益来自指定模型、A100 配置、并行策略、trace 与 SLO，不能外推成通用容量系数。
- [Song 等，Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving（SLOWeave），arXiv:2609.07883v1](https://arxiv.org/abs/2609.07883)：控制器根据活跃 decode 的最早下一 token deadline 和迭代耗时预测选择最大 chunk。其逐轮安全/最大性命题要求 decode-only iteration 可行、耗时关于 chunk 单调且预测准确；论文提出预测误差裕量，但不能据此保证持续过载下的端到端 SLO。评估覆盖有限的模型、加速器、负载和目标；作者列出 cost model 校准漂移、非平滑 kernel/通信/显存成本、单一共享 TPOT 目标、每轮单个 prefill 及未覆盖多节点 KV 传输等边界。
- [Arun 等，Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving（Kairos），arXiv:2607.02043v1](https://arxiv.org/abs/2607.02043)：调度器比较 prefill 节点排队时的预测 TTFT 与 decode 节点执行 chunked prefill 的 TTFT，并检查活跃 decode 的 TBT 约束。论文实现基于 vLLM 0.18.1，使用 DeepSeek-V2-Lite、A100 及 2P2D 实验配置和生产派生 trace；论文报告的百分比结果只适用该研究设置。限制包括 decode cost model 误差会造成 TBT 风险、状态心跳陈旧、当前估算不考虑 prefill 节点上的 prefix-cache 命中，以及高请求率下集中 dispatcher 可能成为瓶颈。

## FlashAttention

- [Dao 等，FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness，arXiv:2205.14135v2](https://arxiv.org/abs/2205.14135)：原论文将 FlashAttention 定义为使用 tiling、减少 GPU HBM 与片上 SRAM 间读写的 IO-aware exact attention 算法，并分析其 IO 复杂度。论文没有给出完整服务系统中的排队、KV 分配器、集群容量或 SLA 保证；特定训练实验的速度数字不应挪作服务容量承诺。

## 证据边界

论文支持作者在特定设置下报告的机制与结果；官方文档支持被记录版本公开的功能和兼容说明。以上来源均不替代目标模型、GPU、引擎版本、批量和任务切片上的本地 profile 或质量验证。
