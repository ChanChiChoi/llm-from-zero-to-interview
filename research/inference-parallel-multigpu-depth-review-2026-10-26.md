# 第六册第八章：并行推理与多 GPU 服务精修研究记录

日期：2026-10-26  
正文：[第八章：并行推理与多 GPU 服务](../book-06-llm-deployment/chapters/08-并行推理与多gpu服务.md)

## 研究问题

原章已覆盖 TP/PP/EP、副本、负载均衡、通信、KV 所有权、故障恢复和 8 卡选型案例。精修重点是把概念和运营决策连起来：

1. 每种并行切分减少哪一类账本，又把通信和故障放到哪里？
2. MoE 的 all-to-all 在当前 serving 实现里如何处理吞吐、低延迟和热 expert？
3. 共置、prefill/decode 解耦和 KV-centric serving 分别在哪些 SLO/网络/缓存条件下有优势？
4. KV 搬运、请求迁移与 owner fencing 怎样避免“张量复制成功但请求状态不一致”？
5. 教学 demo 是否把通信单位、层数和真实链路字节说明清楚？

## 代表性研究与系统证据

### P/D 解耦及 KV transfer

- [DistServe, OSDI 2024](https://arxiv.org/abs/2401.09670)：识别共置 P/D 的资源干扰和并行策略耦合；把阶段放到独立 GPU 池，并按 TTFT/TPOT 共同优化资源与并行布局，再考虑集群带宽以放置节点。摘要报告在其多模型、应用和 SLO 评测中可服务最多 7.4× 请求或支持严格 12.6× SLO，同时超过 90% 请求满足约束。此为论文作者结果，不是通用吞吐比例。
- [Mooncake, 2024](https://arxiv.org/abs/2407.00079)：将 CPU、DRAM、SSD 和 RDMA 组织成分层 KV cache，以 KV-aware scheduler 做 prefix 复用、负载均衡、热点块复制和过载早拒绝。作者报告 10P+10D 对 20 个 vLLM 实例的 trace replay 中，在 TTFT 30 秒、TBT 0.1 秒阈值下，多处理约 75% 请求，Mooncake 近 100% 达到 TBT 而 vLLM 约 57%。数据绑定其采样 trace、硬件与 SLO；论文也有特定公开/模拟数据测试，不拿不同 workload 的最大倍率横向排名。
- [TaiChi, arXiv 2025](https://arxiv.org/abs/2508.01989)：指出 aggregation 更适合严格 TTFT/宽松 TPOT，disaggregation 更适合严格 TPOT/宽松 TTFT；为平衡 SLO 设计 P-heavy/D-heavy 组合、比例/块大小滑杆和 latency shifting。作者在 balanced-SLO 实验报告 goodput 最高提升 77%。它说明系统设计需按 SLO 位置选型；代价是混合资源池和调度器复杂度。
- [SmartGen, arXiv 2026](https://arxiv.org/abs/2607.28150)：在 KV 成为 P→D 网络 payload 后，用 profile-based proactive transfer 先推重要 cache，Decode 并行按需拉取缺失项，再利用后台 speculative transfer 传其余 KV。论文以 L20 GPU 和 25-Gbps RDMA 等设置评测，报告 GovReport time-to-second-token 最多下降 4.3×；收益绑定模型、KV 选择方法与网络，选择错误和远程按需拉取仍要检查质量/TPOT。
- [SpectrumKV, arXiv 2026](https://arxiv.org/abs/2606.08635)：按 token 重要性选择 FP16/INT8/INT4，并以部署期短探测判断能否启用 INT4。摘要中 Qwen2.5-7B 对 INT4 KV 不稳定，而 Mistral-7B、Gemma-2-9B 相对稳定；50% normalized KV budget、WikiText-2 PPL 下，三模型 SpectrumKV 变化分别 +1.97%、−0.06%、−0.44%，PDTrim 为 +25.85%、+22.07%、+35.63%。这组预印本数据展示 model-specific tolerance，不能替目标任务验收。

这些结果的指标不可混比：DistServe/Mooncake 看约束内请求吞吐，TaiChi 看 goodput，SmartGen 看第二 token 延迟，SpectrumKV 看固定传输预算下的困惑度。正文将它们放在同一设计问题中比较机制和成本，但没有排成横向性能榜。

### 当前工程路径和 EP

- [vLLM disaggregated prefilling 文档](https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md)：截至本次查阅，该功能仍标为 experimental；文档区分独立调 TTFT/ITL 和 tail ITL 控制，且明确 disaggregated prefill 本身不提升 throughput，chunked prefill 也可用于控制 tail ITL。它与论文在特定 SLO 下的系统 goodput 结果不是同一评测层级。
- [SGLang PD disaggregation 文档源码](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)：当前公开 transfer engine 包括 Mooncake/NIXL，强调 phase worker、KV transfer timeout、heartbeat 与状态交接；具体 flags 和 connector compatibility 随版本变化。
- [DeepEP 官方仓库](https://github.com/deepseek-ai/DeepEP)：当前主线描述 EP dispatch/combine 的高吞吐与低延迟 all-to-all kernel、FP8 dispatch、NVLink/RDMA 传输等能力。README 同时注明具体硬件/软件条件、一些 PP/Engram/Bucket 能力仍是 experimental；项目能力不能外推为任意 runtime 默认启用。
- [vLLM Expert Parallel Deployment 文档](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)：EPLB 在 forward 期间收集 expert token load 并周期性重映射，可加入冗余 expert；文档对 DeepSeek-V3 估算每 EP rank 增一个冗余 expert 约增加 2.4 GB。该数字是模型/配置说明，不作通用 expert 成本常数。文档提示用 uniform random routing 让 benchmark 负载均衡，因此产品评测还要补真实 trace 的 skew。
- [vLLM parallelism and scaling 文档](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) 与 [NVIDIA NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)：用于核对当前配置入口和 collective 语义；不证明目标拓扑链路速度、tail latency 或目标 workload SLO。

## 原稿知识点覆盖图

| 原稿部分 | 原有知识 | 本轮处理 |
| --- | --- | --- |
| 开头、推理/训练对比、§1 | 多 GPU 动机、容量账本、吞吐/可用性 | 保留；明确单副本容量、横向扩展和故障冗余是不同目标。 |
| §2—§4 | TP、PP、TP/PP 选择 | 保留矩阵切分、通信、stage bubble、KV 分布与判断顺序；完善通信数量示例，使单位按层/按 decode 迭代明确。 |
| §5 | MoE EP、dispatch/combine 和 expert imbalance | 保留机制；补 vLLM EPLB、冗余 expert 内存成本及 DeepEP 的 prefill/decode kernel 选择。 |
| §6—§8 | replica、路由、通信模型和拓扑 | 保留生命周期、queue/token aware routing、collective/all-to-all/activation 关键路径和观测项。 |
| §9 | KV 布局、prefix reuse 与迁移成本 | 保留布局/缓存/搬运账本；增加其与 P/D pool transfer 的关系。 |
| §10—§11 | 故障路径、请求状态和恢复语义 | 保留重算/状态迁移/副本恢复；将 §11 收束为 owner epoch、freeze/transfer/validate/commit/fallback 交接协议，减少和 §9—§10 的重复。 |
| §12—§12.1 | 选型流程与配置表 | 保留原流程；新增共置、DistServe、Mooncake、TaiChi、SmartGen 和 SpectrumKV 的 P/D 比较与 SLO/网络边界。 |
| §13—§16 | demo、8 卡决策、资料和总结 | 保留 8 卡场景；修复 demo 输出与 TP 统计范围，增加当前官方来源和 P/D 主结论。 |

## 修改与配套同步

- demo 原来把一个 decoder block 的两次 collective payload 标为完整 decode pass；现按 80 层×每层两个 hidden-size collective 计算，字段命名为每 rank 每 decode iteration 的逻辑输入 payload，明确不等于链路实际收发字节。原正文把 TP4×PP2 的 PP payload 写成 0.0 MiB，实际为 0.375 MiB；现更新代码输出。
- 新增 P/D 解耦研究比较；明确 phase split 与 TP/PP/EP 的切分对象不同，并把 KV transfer bytes、可隐藏比例、SLO asymmetry 和 destination KV reservation 放进决策条件。
- 重写 §11 的重复总结为带 epoch 的状态 owner handoff 设计建议。它是一个明确标注的工程设计方案，不声称所引用 runtime 都实现该协议。
- `PAPERS.md`、`GLOSSARY_EN_ZH.md`、`INTERVIEW_BANK.md`、`EXERCISES.md` 已按新增研究对照和术语完成同步；第六册目录已有章节入口，无需修改。

## 验收范围与遗留风险

- `formula_status`：依正文 demo 配置复核权重/每 rank KV 账本、TP collective payload 与 PP boundary payload。70B FP16 原始权重约 130.39 GiB；TP4 两副本 toy 每 rank 32.6 GiB 权重+7.5 GiB KV+8 GiB reserve=48.1 GiB；TP8/TP4×PP2 每 rank约28.05 GiB。TP 逻辑 payload `2×80×8192×2×24 / 2^20 = 60 MiB/rank/decode iteration`；PP2 boundary payload `8192×2×24 / 2^20 = 0.375 MiB/iteration`。这不包含 collective 算法的实际链路流量系数、overlap、其他 buffer 或模型布局差异。
- `code_status`：从 Markdown 提取运行 §13 标准库 Python demo，退出码 0、stderr 为空，stdout 与正文 fenced `text` 示例逐字一致；另核对 PP2 边界 payload 为 0.375 MiB、TP=1/PP=1 不产生 collective 或 pipeline 边界 payload。仅验证教学账本和代码路径。
- `link_status`：论文与官方文档入口按本次资料记录核对；正文和研究笔记的 3 个本地 Markdown 链接目标均存在。主分支文档仍可能变化。
- `validation`：核对 §1—§16 连续、28 对 Markdown 围栏闭合、11 张表格列数一致、内部链接目标存在、demo 输出、关键公式量纲与 `git diff --check`；均通过。
- `open_risks`：没有在目标模型、GPU、拓扑、请求 trace 或 runtime 上复现 TP/PP/EP/P-D/Mooncake/SmartGen。论文的 SLO/吞吐/TTST/PPL 指标不可混比；当前 vLLM/SGLang/DeepEP 主分支文档会变化。EPLB 冗余 expert 显存取决于模型结构和配置。owner epoch handoff 是系统设计建议，需结合实际控制面协议、流式 API 与下游副作用实现。demo 的 60 MiB 是设定假设下每 rank 的逻辑 collective 输入量，不是物理网络流量或性能测量。
