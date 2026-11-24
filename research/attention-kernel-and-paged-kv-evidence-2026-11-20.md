# Attention kernel 与分页 KV 的来源证据

研究日期：2026-11-20。研究问题为：dense exact attention kernel 如何减少中间数据搬运；FlashAttention 家族和 Flash-Decoding 分别怎样改变工作划分；PagedAttention 如何在物理非连续 KV blocks 上保持逻辑序列；共享 block 与后端 dispatch 的正确性和结论边界是什么。

检索词包括 `FlashAttention`, `FlashAttention-2`, `FlashAttention-3`, `FlashAttention-4`, `Flash-Decoding`, `PagedAttention block table`, `paged KV prefix sharing`, `SDPA backend dispatch`, `attention kernel lecture notes`。检索了原始论文、作者/团队解释、官方框架文档与课程讲义；技术事实优先按论文及版本文档核对。以下材料支持各自列出的主张，不能互相代替。

## FlashAttention 与 attention kernel

| 来源与版本 | 阅读位置及支持内容 | 不能据此外推 |
| --- | --- | --- |
| Dao 等，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，NeurIPS 2022 | §3.1–3.2、Algorithm 1、Theorems 1–2：tile Q/K/V，在片上存储更新 online-softmax 行统计；精确 dense attention 为 $O(N^2d)$ FLOPs，需 $O(N)$ 额外内存；论文 IO 模型给出标准物化 attention 与 FlashAttention 的 HBM 访问差异。backward 会重算 tile。 | “精确”指相同 attention 语义的算法，不保证浮点逐位一致；IO 下界和速度数字受论文的存储模型、GPU、shape 与基线约束。该论文另有 block-sparse 扩展，因改变连接图而不属于 dense exact 路径。 |
| Dao，[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)，2023-07 | 摘要与算法部分：减少 non-matmul FLOPs；跨 thread blocks 对单 head 沿序列并行；调整 warp 分工以减少 shared-memory 通信。论文在 A100 报告的效率/训练速度绑定所测配置。 | 不推出所有 GPU、shape、dtype、batch 均获相同比例加速。 |
| Dao，[New features: head dimensions up to 256, multi-query attention](https://crfm.stanford.edu/2023/07/17/flash2)，Stanford CRFM 作者/团队说明 | 解释 FA2 的支持范围和 A100 基准设置；适合作为论文的讲解入口和实现背景。 | 页面中的倍速仅属于其列出的 A100、head dimension、mask、dtype、software snapshot 与 baseline。 |
| Shah 等，[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)，2024-07 | §1–3：面向 Hopper，利用 TMA/Tensor Core 异步和 warp specialization 重叠数据移动与计算，交错 block matmul/softmax，并研究 FP8 block quantization/incoherent processing。论文报告的 H100 数值及 FP8 误差对照绑定实现和比较对象。 | 不推广至 Ampere、Blackwell 或未支持的算子路径，也不将某个 FP8 对照的误差比例作为任意低精度 attention 的一般保证。 |
| Zadouri 等，[FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)，2026-03-05 | §1、§3、§5：针对 Blackwell 的硬件吞吐不对称，重新安排 fully asynchronous MMA、tile/pipeline、softmax 指数计算及共享内存/Tensor Memory 工作。§5.1 在 B200、BF16/FP16、若干长度、head size 与固定 batch-token 设定中报告相对 cuDNN 9.13 和 Triton 的结果。 | 论文还说明后续 cuDNN 版本吸收相关技术后性能接近 FA4；不能将其早期相对基线数字当作当前库的固定排名或通用加速率。 |
| PyTorch，[Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)，作者/团队说明 | 解释 decode query 行少时沿 batch/query 并行可能不足；将 K/V 长度切成分段并行计算，各段保留可合并的 log-sum-exp/attention 状态，再作一次 reduction。 | 并行分段有额外 reduction；收益依赖历史长度、batch、设备和实现，不是所有 decode 请求都更快。 |
| Berkeley Scalable AI，[Lecture 6: Attention and MoE Kernels](https://scalable-ai.eecs.berkeley.edu/F2026/assets/lecture_slides/lecture6.pdf)，Fall 2026 | 第 3 页及 attention slides：用 S/P 在 HBM 往返说明 materialized attention 的流量，再组织 FlashAttention 的分块路径；作为课程讲解路线。 | 幻灯片不替代论文的数学证明或版本行为，也不证明某种教学顺序对所有读者有效。 |

## PagedAttention 与 KV block

| 来源与版本 | 阅读位置及支持内容 | 不能据此外推 |
| --- | --- | --- |
| Kwon 等，[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，SOSP 2023 | §4.1–4.4：每个请求的逻辑 KV 序列由固定 token 数的 blocks 组成；block table 映射逻辑 block 到物理 block；kernel 逐 block 访问非连续 K/V；按需分配减少为最大长度预留的浪费，且可用于并行 sampling/beam search 的共享前缀。§7.1 报告其特定 kernel 与 FasterTransformer 比较中 attention-kernel latency 高 20–26%，同时需结合系统级结果理解。 | 不改变活跃 token 的基础 KV payload；kernel-level overhead 与 system throughput 是不同指标。20–26% 是论文比较下的历史 observation，不是当前引擎或所有实现的常数。 |
| vLLM，[Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)，2026-11-20 访问 | 官方文档说明 prefix-cache 可复用重复文本前缀的 KV，给出 hash/block-size 相关实现入口；用作当前 engine 的版本化行为参考。 | API、cache key、淘汰、跨模型/adapter 或跨模态策略会随版本改变，不把某个实现策略写成 PagedAttention 的数学定义。 |
| vLLM，[Paged Attention](https://docs.vllm.ai/en/v0.12.0/design/paged_attention/)，v0.12.0 文档 | 文档明确称其是基于原论文的历史说明，且不再描述当时的当前代码；其 thread/warp 分工可帮助理解单 query paged kernel。 | 不可当作 vLLM 当前实现路径的证据。 |
| Nalla，[PagedAttention: Breaking the KV Cache Memory Wall](https://ramunalla.github.io/posts/paged_attention/index.html)，2026 | 用逻辑序列、物理块和 block table 的小 token 轨迹组织讲解；用于教学结构参考，和原论文对照核验。 | 页面中的“gold standard”等价值判断、广泛的产品能力/性能结论不作为事实依据；其概念例子不能替代当前 kernel 细节。 |
| vAttention，[Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437)，2024 预印本 / ASPLOS 2025 | 提供不同于软件显式 PagedAttention block management 的动态虚拟内存方向，可作为 KV 分配并非只有一种路线的来源入口。 | 性能比较绑定作者的 A100、模型、kernel 与评测设置，不支持一般优劣排名。 |

## 后端与 API 事实

- [PyTorch 2.15 SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 与 [SDPA API 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)：融合后端受 shape、dtype、mask、设备和构建支持范围影响；选择/限制 backend 可以显式控制，但调用高层 API 本身不能证明运行了特定 kernel。文档还指出不同融合与 math backend 可能因浮点顺序产生小差异；API/dropout 细节仅按该固定版本引用。
- [PyTorch SDPA backend context manager](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.sdpa_kernel.html)：用于固定版本的后端选择示例，不代表所有 framework 具有相同 dispatch 行为。

## 本记录不支持的结论

- FlashAttention、Flash-Decoding 和 PagedAttention 不是可互换的同一类算法；一个减少 exact attention 中间数据搬运，一个改变 decode 的并行轴，一个管理 KV 状态的物理映射。
- FlashAttention 的 dense exact 路径不删除允许的 query-key pairs；若使用 block-sparse 扩展，则 attention pattern 已改变，应分别陈述。
- FlashAttention 版本论文中的硬件性能、利用率、FP8 误差和 speedup 绑定各自 GPU、dtype、shape、mask、代码版本及 baseline。
- PagedAttention 减少预留和尾块浪费并可能增加可接纳请求数，但 block table lookup、非连续访存和 kernel 分支有成本；不能只凭 kernel latency 或只凭 allocator 利用率断言端到端优劣。
- prefix sharing 只在 KV 状态兼容且共享页遵循只读/引用生命周期约束时成立；prefix-cache 的实际 key 与回收策略是 engine 版本行为。
- 当前 framework API 名称、backend 默认次序、支持范围与 fallback 均可能变化。未经固定版本源码/文档与实测，不推断当前服务必然启用特定注意力 kernel。
