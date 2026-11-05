# 大模型系统设计架构证据记录

资料核验日期：2026-11-04

本记录汇总系统设计场景中近期涉及推理编排、模型路由、Agent 状态与评估平台生命周期的来源事实。它不代表任何单一架构适用于所有流量或云环境。

## 推理阶段拆分、KV 状态与调度

截至 2026-11-04 核验的版本为 v3（2024-06-06）。DistServe 将 prefill 和 decode 放到不同 GPU 资源上，按 TTFT/TPOT 服务目标分别配置并行度和资源；论文的收益绑定其模型、工作负载、集群带宽、放置方式和 SLO attainment 条件，不能概括为拆分后必然更快。[DistServe v3](https://arxiv.org/abs/2401.09670v3)

截至 2026-11-04 核验的最新版本为 v5（2026-10-08）。Mooncake 描述 KVCache 中心的分离式架构，把 prefill、decode 分池，并利用 GPU 集群中的 CPU、DRAM、SSD 和 RDMA 资源构建分层缓存；调度同时考虑缓存复用、负载和延迟目标，在过载条件下还使用早拒绝策略。论文报告结果受其采样请求轨迹、集群配置与模拟/实测设置限制。[Mooncake v5](https://arxiv.org/abs/2407.00079v5)

截至 2026-11-04，TaiChi 为 v1（2025-08-04）。它直接比较 prefill-decode 聚合与分离：作者报告聚合和分离分别能在不同 TTFT/TPOT 目标组合下达到较好 goodput，并提出混合架构。该工作支持“由工作负载与目标决定是否拆分”的问题框架，不支持把某一实现的结果推广为通用生产结论。[TaiChi v1](https://arxiv.org/abs/2508.01989v1)

NVIDIA Dynamo 最新文档索引在本次访问时列出 v1.6.1。其架构将请求执行、扩缩容与放置控制、KV 复用和失败恢复分成协作路径，并把 prefill/decode 分离、KV-aware routing、KV 生命周期和过载处理纳入同一 runtime 架构。此为特定 runtime 的设计描述和目标，不等于跨后端的独立性能比较。[NVIDIA Dynamo 架构文档](https://docs.nvidia.com/dynamo/latest/knowledge-base/overview)

## Kubernetes 推理网关接口演进

截至核验日，Kubernetes Gateway API Inference Extension 仓库说明：EPP 与关联 API、Body-Based Router 已迁往 llm-d 旗下仓库；本仓库继续维护 Lightweight EPP、InferencePool API 和一致性测试。仓库将推理网关描述为依据服务指标和能力信息选择模型服务副本，并将 KV 前缀缓存状态、LoRA 可用性、灰度发布、A/B 流量和可观测性列为接口/能力方向。仓库迁移状态应绑定本次快照，不能当作所有 Kubernetes 网关的统一 API 合同。[Gateway API Inference Extension 官方仓库](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

## Agent runtime 的状态所有权

OpenAI Agents SDK 文档区分了直接调用 Responses API 与使用高层 Agent runtime：直接调用时应用自行持有循环、工具分发和状态处理；SDK runtime 可以管理 turn、工具执行、guardrails、handoff 和 sessions，并提供 human-in-the-loop 与 tracing 等能力。这是 SDK 当前公开接口边界，不代表 Agent 系统必须由 SDK 代管状态。[Agents SDK 官方文档](https://openai.github.io/openai-agents-python/)

## 评估平台生命周期

截至 2026-11-04，OpenAI Evals 官方迁移说明已将现有 Evals 平台设为自 2026-10-31 起只读，并计划在 2026-11-30 关闭；该指南建议新用户使用 Datasets。此时间表只描述该产品表面，不能据此推断评估平台的一般设计已经过时。讨论平台架构时应优先讲稳定的任务定义、数据集、grader、运行与比较、trace、反馈和持续回归环节，再把具体厂商 API 当作带版本和生命周期的实现选项。[OpenAI Evals 官方指南](https://developers.openai.com/api/docs/guides/evals)

OpenAI 的官方评估实践页面把评估流程拆为目标定义、数据收集、指标定义、运行比较和持续评估；Agent 评估资料补充了 trace、graders、datasets 和重复 eval runs。它们是厂商给出的应用评估工作流示例，不替代评估方法本身或独立验证。[Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices), [Agent evals](https://developers.openai.com/api/docs/guides/agent-evals)

## 设计审视维度

AWS Generative AI Lens 的 2025-11-19 版本将生成式 AI 系统实践放在运营、可靠性、安全、性能、成本和可持续性维度下，并覆盖 scope、model selection、customization、development、deployment 和 continuous improvement。它是 AWS Well-Architected 的生成式 AI 指南，具体服务举例带有 AWS 语境；这些维度可作为检查清单来源，但并非普适法规或唯一标准。[AWS Generative AI Lens](https://docs.aws.amazon.com/wellarchitected/latest/generative-ai-lens/generative-ai-lens.html)

## 服务目标与观测指标

Google SRE 的《Service Level Objectives》章节区分服务等级指标（SLI）、服务等级目标（SLO）与服务等级协议（SLA）：SLI 是对服务某个方面的定量测量，SLO 是基于 SLI 的目标值或范围，SLA 是与用户达成且包含目标达成/未达成后果的协议。该章也说明延迟、错误率、吞吐和可用性指标需要结合测量窗口、请求集合、采集位置和聚合口径解读。它是 Google SRE 方法的权威说明，可作为系统设计术语入口，不是对所有合同或企业流程的法律定义。[Google SRE《Service Level Objectives》](https://sre.google/sre-book/service-level-objectives/)

## 证据等级与边界

- DistServe、Mooncake、TaiChi 是系统论文，结果适用于各自报告的模型、数据、硬件、SLO 与实验设置。
- Dynamo、Gateway API Inference Extension、Agents SDK 和 OpenAI Evals 是官方实现/产品文档；可支持公开架构或接口事实，不证明其相对性能、跨产品可移植性或生产适用性。
- AWS Well-Architected Lens 是厂商参考框架，适用于理解一个云服务商如何组织设计问题，不作为标准认证或普遍最佳架构证明。
- 所有网页和版本状态按 2026-11-04 查阅记录；API、仓库归属、产品生命周期和文档内容可能继续变化。
