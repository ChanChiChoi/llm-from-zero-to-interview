# 生产系统设计与异质 SLO Serving 证据（2026）

记录日期：2026-11-02。本文仅记录联网阅读所得的论文事实、来源版本和证据边界，不记录书籍审阅进度。

## JITServe：不同请求形态需要不同服务目标

- **来源与版本**：[USENIX NSDI 2026 正式论文页](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)，Wei Zhang 等，*JITServe: SLO-aware LLM Serving with Imprecise Request Information*，NSDI ’26，pp. 825–848；同时阅读[正式论文 PDF](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf)。访问日期：2026-11-02。
- **阅读位置**：论文摘要、§2.1 请求类型、§6.1 实验设置和 §7 讨论。
- **论文支持的事实**：作者将 LLM 请求分为 latency-sensitive、deadline-sensitive 与 compound 三类。交互流式请求关注 TTFT/TBT；deadline-sensitive 请求关注完整响应时间；compound 请求由多个有依赖关系的模型调用组成。论文将 JITServe 实现为 vLLM 调度层扩展，并在 16 张 NVIDIA A100 上评测 Llama-3.1-8B、Qwen2.5-14B、Qwen3-30B-MoE-A3B 与 Llama-3.1-70B。负载包括聊天、深度研究、代码 Agent 和数学推理；作者结合公开数据集、Microsoft 服务 trace、用户研究形成请求与到达分布。论文摘要报告相对于其基线的 1.4×–6.3× service-goodput 提升，另报告等 goodput 下的资源节省范围。
- **证据边界**：请求分类和调度结果属于该系统论文的定义与实验。模型、GPU 集群、合成/回放负载、SLO 和基线限定性能结果；这些倍率不是其他服务的容量承诺。论文讨论指出 all-or-nothing goodput 不表达未达 deadline 但仍有部分效用的请求。
- **证据状态**：source-backed（正式系统论文；未在本地复现）。

## QoServe：共享资源池中的 QoS 区分与过载降级

- **来源与版本**：[ASPLOS 2026 论文 DOI](https://doi.org/10.1145/3779212.3790206)；[Microsoft Research 托管论文 PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/10/QoServe_CC_final_v2.pdf)。Kanishk Goel 等，*QoServe: Breaking the Silos of LLM Inference Serving*，ASPLOS ’26，2026-03。访问日期：2026-11-02。
- **阅读位置**：摘要、§3.1 系统工作流、§4 评测和 artifact appendix。
- **论文支持的事实**：QoServe 为不同延迟目标设置细粒度 QoS 类，在共享资源池中结合 dynamic chunking、hybrid prioritization 与 selective request relegation。评测使用 ShareGPT 和 Azure 代码/对话 trace，并测试 A100 与 H100 配置。摘要分别报告：A100 集群上相对隔离部署的容量提高 23%；共享集群配置中相对 Sarathi 的 per-replica goodput 最高提高 2.4×；极端负载下 SLO violation 降低一个数量级。
- **证据边界**：论文摘要中的不同改善数字分别对应不同配置和比较基线，不能互相拼接或与 JITServe 的结果横向排名。论文的 trace、模型、GPU、QoS 类别、负载生成和 artifact 规模限定结果；这些指标不证明同一调度策略适用于任意在线服务。
- **证据状态**：source-backed（正式系统论文；未在本地复现）。

## 相邻章节候选（只读摘要，未纳入本章结论）

- **PackServe**：[arXiv v1](https://arxiv.org/abs/2609.33224)，2026-09-27。摘要聚焦 Agentic serving 的请求调度、prefix KV 复用和 TPOT；作者报告了特定 64 张 H20 GPU 评测及生产集群观察。当前只据摘要识别主题，没有核对正文实验表和生产部署口径；性能数字不能外推。具体调度机制更适合第六册第六章后续扫描，本章只保留服务目标和容量方案的高层设计。
- **InferSlice**：[arXiv v1](https://arxiv.org/abs/2610.32638)，2026-10-23。摘要介绍面向电信 WAN 的参考架构，组合网络切片、GPU 分区、弹性 serving 与可观测性，并报告多站点并行方式的实验观察。当前仅读摘要，尚未核对全文架构和实验细节；其 telecom/WAN 前提具专门性，若全书复审纳入，优先映射至第二十三册 AI Infra 的跨区域网络与异构集群内容。
- **证据状态**：候选发现（摘要层）；未完成全文核验，不作为本章正文证据。
