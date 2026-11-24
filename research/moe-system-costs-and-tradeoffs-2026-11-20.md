# MoE 系统成本与 Dense/MoE 比较：证据记录

检索与核验日期：2026-11-20。研究问题是如何解释稀疏 expert 路径对训练与服务资源的影响，并区分可复核的系统案例与可推广的架构结论。读者与范围限定为第二十一册第二十二章的系统级比较；不测量本地集群，不核算当前产品账单，也不重做 router、capacity、Expert Parallel 或完整推理服务教程。

## 检索范围

- 研究线索："DeepSeek-V3 training cost official report 671B 37B activated H800 GPU hours"；"MegaBlocks MoE training same validation loss dense A100"；"Megatron Core MoE communication overlap exposed iteration time"；"MoE decode memory-bound activated experts per GPU"；"MoE prefill decode phase expert offload QoS"。
- 教学线索："MoE system costs training versus inference expert weights data movement prefill decode"；"Mixture of Experts inference data movement tutorial"。
- 检索材料：MoE 原始系统/训练论文与技术报告；MoE serving 研究；NVIDIA Megatron-Core 官方文档及其技术报告；SemiAnalysis 的系统讲解文章。关键词用于定位材料，性能数字只按原研究的模型、硬件、负载和计时口径记录。
- 检索平台与筛选：英文搜索引擎、arXiv HTML、NVIDIA Megatron-Core 在线文档和 SemiAnalysis 页面；优先保留原始论文/报告核对机制、配置与实验数字，博客/文档用于补齐系统视角和讲解顺序。页面与版本按 2026-11-20 实际访问状态读取。
- 未研究：未公开模型的部署布局、目标集群吞吐、通用 MoE 成本系数、跨论文性能排名，以及将 API 价格换算成架构成本。

## 来源卡与证据边界

| 来源 | 版本与阅读位置 | 支持 | 不支持 |
| --- | --- | --- | --- |
| DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2) | arXiv v2，2025-02-18；摘要、§§2.1、3.1–3.2、训练成本 Table 1 与结论 | 报告披露 671B total/37B activated、14.8T 预训练 token；作者报告官方训练共 2.788M H800 GPU hours，并说明金额按每 GPU hour $2 估算；其系统结合 FP8、节点限制路由、DualPipe、跨节点 all-to-all 和工程优化 | 该数字不是 MoE 对 matched Dense 的直接比较；官方训练成本明确不含此前架构、算法和数据研究/消融成本；不能据此证明 MoE 的普遍成本或收益 |
| Gale et al., [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/html/2211.15841) | arXiv v1，2022-11-29；§§5–6，尤其 §6.1 | 针对动态、不均匀 expert assignment 的 block-sparse 表示和 kernel 可避免固定 padding/drop 取舍中的部分成本；作者在 The Pile、XS/Small/Medium、每层 64 expert/top-1、8×A100 80GB、EP8 等设定下报告了端到端训练结果 | 该实现、模型尺度、A100/CUDA 版本及容量基线不代表所有 MoE；其同验证损失与 Tutel/Dense Megatron 对照不能移作现代大模型普遍加速系数 |
| Yan et al., [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/html/2603.07685v2) | arXiv v2，2026-03-10；摘要、§§4.2.2–4.2.3、8、9.2 | 该报告说明 MoE 的 memory/communication/compute 效率需要协同优化；DeepSeek-V3 EP64 案例中，优化 dispatcher 后仍报告 EP communication 暴露占迭代的 30–40%，加入相邻 microbatch 的 forward/backward overlap 与梯度拆分后，H100 设置下报告该暴露份额降至 5% 以下 | 结果属于报告中的 DeepSeek-V3/H100/EP64 训练设置；减少的是暴露在关键路径上的通信时间份额，不等于通信数据量消失，也不证明所有工作负载的端到端成本按同一比例下降 |
| NVIDIA, [Megatron-Core MoE documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) | 动态官方文档，访问于 2026-11-20；“Training Optimizations”及 Parallel Folding、memory/communication sections | 公开 feature/config 入口展示 EP 并行组合、通信重叠、内存优化与 compute-kernel 优化怎样共同构成当前 Megatron-Core MoE 栈 | 文档会随版本更新；参数/feature 可用不代表目标硬件上的收益，不能代替固定配置 benchmark 或集群 trace |
| Zhang et al., [DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance](https://arxiv.org/html/2509.07379v2) | arXiv v2，2026-04-09；摘要、§§II、V–VI，模型配置表及性能设置 | 资源受限的 expert offload 场景中，prefill 对专家权重批量/驻留和峰值显存的压力，与 decode 的逐步传输延迟不同；作者据此为两阶段设计不同的 expert 调度。在其 A5000、SQuAD 与四种开放 MoE 配置的实验中报告特定延迟/内存结果 | 该单 GPU offload/runtime 研究不代表大规模常驻多 GPU 推理、任意 prompt 分布或其他 QoS；预测准确率及速度收益不能外推 |
| Yu et al., [Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens](https://arxiv.org/html/2512.09277v1) | arXiv v1，2025-12-10；§§III、VI，摘要与限制讨论 | 作者指出 memory-bound decode 中每 GPU 的 activated expert 数可能比 token 数更能解释权重流量；在其 vLLM 8×A100 实机与 8–16×B200 模拟、Qwen3/DeepSeek-V3、prefill/decode 共部署设定下，报告其 METRO 相对 EPLB 的 decode 延迟和总 token throughput 变化 | 这是特定 router、模型、硬件与负载下的系统研究；不证明所有 MoE decode 都由同一项主导，也不证明均衡 token 目标在其他阶段或系统中无效 |
| Bennett, Shan & Patel, [Computation and Data Movement for Inference](https://newsletter.semianalysis.com/p/computation-and-data-movement-for) | 2026-09-21；可见页面开头及 §3 prefill/midfill/decode（页面标注付费，仅使用已显示内容） | 以服务全链路和数据移动为起点，分别描述 prefill、midfill 与 decode 的计算/状态读取差异，作为组织 MoE serving 解释的教学参考 | 页面付费且本次只读取可见部分；文章框架不能代替论文/系统报告，也不支持其未核验的模型或硬件性能外推 |

## 可复用的研究关系

1. total parameters 表示模型容量与需要在系统中表示/保存的完整权重规模；active parameters 描述某 token 经过的部分参数路径。二者都不能单独给出真实 GPU 显存、FLOPs、训练步时或线上延迟；具体驻留内存还取决于分片、复制、缓存、offload 和精度。
2. 稀疏路由改变了 expert GEMM 的工作量，但训练成本还包括优化器/检查点的全部模型状态、路由后的负载分布、buffer 与通信、kernel 形状以及可重叠的空隙。优化 dispatch 和 schedule 可以降低 padding、drop 或暴露等待，却可能把瓶颈移到内存、计算效率、调度或其他通信。
3. 训练和 serving 的批量形态不同。prefill 的多 token 可能形成较大的 expert batch；decode 每条请求每步只引入少量 token，而权重迁移、expert locality 和 KV 读取会与 latency/throughput 目标共同作用。MoE 稀疏 FFN 不决定 attention KV cache 的大小。
4. “同 active 参数”或“相同 FLOPs”不是公平比较的充分条件。比较训练时间须说明达到的质量、token 数据、训练配置、硬件、精度和计时范围；serving 比较还需说明输入/输出分布、并发、runtime、量化与目标 latency。不同研究各回答一个受限问题，不应拼成架构普遍排名。
5. DeepSeek-V3 的作者报告显示低于 active 参数比例的训练成本需要结合模型、硬件和系统共设计；MegaBlocks 展示另一种具体稀疏计算实现；DuoServe 与 METRO 则表明服务不同阶段的主导成本和均衡目标可以不同。这些案例证明系统条件重要，不证明单一路线更优。

## 待核边界

- DeepSeek-V3 的训练 GPU-hour 数按作者报告口径引用；技术报告没有提供一套同质量、同硬件、同数据、同训练阶段的 Dense baseline，且明确排除先前研究与消融成本。
- Megatron-Core、DuoServe 与 METRO 的新版本、测量环境或配置变化可能改变性能比例；正文只采用固定版本中的方法和实验报告，不写成当前生产部署建议。
- 本记录中的论文/技术报告性能结果均为 source-backed，不代表本地复现；SemiAnalysis 仅用作已读可见部分的教学组织来源。
