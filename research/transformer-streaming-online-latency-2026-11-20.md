# Transformer 流式、在线与低延迟：主题证据记录

研究日期：2026-11-20。范围：第二十一册第十五章，区分增量输入、逐 token 输出与在线 serving；核对自回归依赖、prefill/decode、KV 历史状态、流式输入模型设计及调度层缓解。目标读者已理解因果注意力和 KV Cache 基本概念。本记录只记外部知识与证据，不记章节修改过程。

## 检索范围

- 平台：web 搜索；原文阅读包括论文/官方文档页面和解释性教材页面。
- 检索词（中英文）：`Transformer autoregressive inference streaming input online latency prefill decode sequential token`；`Orca iteration-level scheduling LLM serving paper`；`Sarathi-Serve chunked prefill throughput latency`；`vLLM latest chunked prefill documentation`；`streaming simultaneous speech translation partial input Transformer memory`；`Mamba recurrent mode parallel scan state space`；`Chip Huyen inference optimization TTFT TPOT`；`Modular LLM inference prefill decode KV cache`。
- 检索日期与版本边界：页面于 2026-11-20 查阅。vLLM 使用标为 v0.25.1 的 V1 optimization 文档；参数、默认值和策略只支持该文档版本的描述，不应外推到其他 runtime 版本。

## 外部主题及章节去向

| 外部主题 | 来源和支持范围 | 读者结果与前置 | 章节处理 |
| --- | --- | --- | --- |
| 流式输出、流式输入和在线服务是不同条件 | Vaswani 等人说明 decoder 按自回归步骤消费先前输出；Ma 等人的同步语音翻译工作将增量输入、部分输入下开始输出和超长连续输入分别作为设计要求，并报告特定 augmented-memory encoder 与 wait-k 策略 | 读者能判断“逐 token 吐出”并不等于“可处理任意连续输入”；需理解 causal generation | 本章先定义三种约束，并以同步翻译论文说明处理部分输入需要额外结构/策略；不把一个任务方法推广到所有语音或多模态系统 |
| Prefill、decode 与用户可见延迟 | Sarathi-Serve 描述 prompt prefill 与逐 token decode 的差别；NVIDIA benchmark 文档分别定义 TTFT、ITL，并指出工具对计时边界有差异；Chip Huyen 的《AI Engineering》章节摘要用 TTFT/TPOT 组织推理延迟，并区分模型级与 serving 级优化 | 读者能按第一个 token、后续 token 和整条请求拆解等待；前置为 token 生成 | 保留简短时间线和指标表。计时起止点必须声明；不把单个 TTFT/TPOT 数字作为通用定义或模型固有常数 |
| KV 缓存减少重复计算但显式保存历史 | Mamba 论文引言将 attention 的显式上下文存储与有限递归状态对比；第十一章已主讲 KV 容量、读取和分页 | 读者能说明缓存避免重算历史 K/V，但历史访问和增长状态仍有代价；前置为本册第十一章 | 本章只解释对低延迟的结构含义并链接第十一章；容量公式、量化、分页与 kernel 细节留在第十一、二十、二十四章 |
| 自回归步序、prefill/decode 干扰与分块调度 | Orca（OSDI 2022）把调度粒度推进到模型迭代；Sarathi-Serve（2024）用 chunked prefill 与 stall-free scheduling 处理 prompt 与 decode 的吞吐/延迟取舍；vLLM v0.25.1 V1 文档说明可调 token budget 影响 ITL/TTFT | 读者能区分模型的逐 token 依赖和 runtime 的排队/批次干扰；前置为 prefill/decode | 只用分块 prefill 作为“调度可缓和、但不改变自回归依赖”的一个例子；算法和生产调参回链第六册第 2、5、6 章 |
| 在线处理完整输入的架构条件 | Ma 等人的同步语音翻译论文在特定语音翻译任务中以分段编码、memory bank、部分输入读写策略处理连续输入，并联合报告质量与延迟 | 读者能认识到双向编码/等待完整输入与增量处理之间的结构冲突，也能指出 chunk/window/state 会改变可见历史 | 说明普通完整上下文编码器无法天然给出流式输入策略；因果 decoder 可逐块追加输入但需正确管理位置、状态和读取策略。避免扩写成语音系统教程 |
| 有限递归状态的替代路径和边界 | Mamba 原论文第 2–3 节描述递归与卷积计算表示；作者报告硬件感知 selective scan，并用其特定模型、数据和硬件评估速度。论文同时指出有限状态会把过去压缩，效果取决于状态的表示能力 | 读者能把逐位置显式历史与递归压缩状态作结构比较；前置为第十四章及第二十九至三十三章 | 只做结论性对比并交接相关章节；不引用论文中的吞吐倍数作普遍结论，不把线性复杂度等同于低 wall-clock latency |

## 核心证据和边界

1. **Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.** 原论文第 3 节描述 decoder 每步消费先前生成符号；第 3.1 节给出 decoder 的因果 mask；第 4 节/Table 1 比较不同层的序列计算复杂度。来源：[NeurIPS PDF](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)、[arXiv 摘要](https://arxiv.org/abs/1706.03762)。支持原始 Transformer 的自回归依赖和复杂度分析，不直接给出现代推理服务的 wall-clock 性能。
2. **Ma et al., “Streaming Simultaneous Speech Translation with Augmented Memory Transformer,” arXiv:2011.00033, 2020.** 摘要、引言和第 2–5 节讨论 partial-input translation、增量分段编码、memory banks、wait-k，以及算法延迟和计算延迟；这是特定 MuST-C 任务、模型与实验设置，不是所有 Transformer 输入流的性能保证。[arXiv 页面](https://arxiv.org/abs/2011.00033)。
3. **Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models,” OSDI 2022.** 第 3–4 节说明逐迭代执行、请求池、KV 管理和 iteration-level FCFS scheduler；评估只支持论文模型/硬件/负载条件。[USENIX 正式论文 PDF](https://www.usenix.org/system/files/osdi22-yu.pdf)。
4. **Agrawal et al., “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve,” arXiv:2403.02310, 2024.** 摘要和第 1、4 节说明 prefill/decode 负载差异、chunked prefill、stall-free batch 的设计；论文性能数字取决于其指定模型、GPU、并行和 workload，不转写成通用预期。[arXiv 页面](https://arxiv.org/abs/2403.02310)。
5. **vLLM V1 Optimization and Tuning, v0.25.1 文档。** 2026-11-20 访问的该版本文档称 chunked prefill 会把较长 prefill 切成小块与 decode 同批，`max_num_batched_tokens` 的调整对 ITL、TTFT、吞吐有取舍。[官方文档](https://docs.vllm.ai/en/v0.25.1/configuration/optimization/)。这是版本特定接口说明，不是通用调度定律。
6. **NVIDIA, “LLM Inference Benchmarking: Fundamental Concepts.”** 第 2–3 节说明其工具中的 TTFT、ITL/TPOT、E2E 和 throughput 口径，并指出 GenAI-Perf 与 LLMPerf 对指标分母/计时范围不同。[NVIDIA Technical Blog](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/)。这是指定工具定义及作者建议；跨工具必须检查本工具协议。
7. **Gu & Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” arXiv:2312.00752, 2023.** 第 2、3.3、4.5 节说明 recurrent/convolutional 计算表示、selective SSM 的硬件感知 scan，以及论文内速度/内存评估；第 1 节将有限状态压缩与显式上下文对比。[arXiv 页面](https://arxiv.org/abs/2312.00752)。作者报告的 throughput 对应论文所列模型、序列、实现和设备，不外推为任意部署更快。
8. **教学材料：** Chip Huyen, *AI Engineering*, Chapter 9 summary 将 TTFT 与 prefill、TPOT 与 decode 对齐，并明确 model-level 与 service-level 优化区别，[公开章节摘要](https://github.com/chiphuyen/aie-book/blob/main/chapter-summaries.md)；Modular *LLM Inference Handbook* 对 prompt prefill、token decode、KV cache 和 causal mask 给出逐步讲解，[公开讲义](https://github.com/modular/llm-inference-handbook/blob/main/docs/llm-inference-basics/how-does-llm-inference-work.md)。二者用于选解释顺序，不能单独支持性能比较。

## 待核边界

- latency 数字必须随模型、输入/输出长度分布、硬件、runtime、负载、计时端点和分位数报告；本章不列未经本地测量的性能数字。
- full attention 的每步历史读取与 KV 容量随实际注意力形式、GQA/MQA、量化、分页、共享和缓存实现变化；精确容量/带宽分析留在第十一章与第二十章。
- 递归/SSM 的有限状态、并行 scan、线性时间和作者报告速度分别是架构机制、算法计算形式与有条件实验结果，不得合写成“天然更低延迟”。
- 输入流任务（例如同步翻译）需要同时评估读写策略、输出修订、质量和算法/计算延迟；不能仅用文本生成的 TTFT/TPOT 代表该任务。
