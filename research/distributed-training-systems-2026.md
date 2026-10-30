# 分布式训练系统 API 与平台边界：来源证据

检索与阅读日期：2026-10-30。

## PyTorch 2.15 分布式训练接口

- **来源与版本**：[torch.distributed](https://docs.pytorch.org/docs/2.15/distributed.html)、[DDP internal design](https://docs.pytorch.org/docs/2.15/notes/ddp.html)、[FSDP1](https://docs.pytorch.org/docs/2.15/fsdp.html)、[FSDP2 fully_shard](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)、[pipeline parallel](https://docs.pytorch.org/docs/2.15/distributed.pipelining.html)、[data](https://docs.pytorch.org/docs/2.15/data.html)、[Distributed Checkpoint API](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)，访问日期 2026-10-30。
- **阅读位置**：DDP reducer、gradient bucket 和 collective 顺序说明；FSDP1 state-dict / shard 接口；FSDP2 migration 与 fully_shard 参数表示、通信分组说明；distributed checkpoint API。
- **来源支持的事实**：DDP 将梯度组织为 bucket 并在 backward 中触发 all-reduce；各进程 collective 顺序不一致可能产生错误或 hang。FSDP1 使用 wrapper 与 flat-parameter 分片；FSDP2 使用 DTensor 的逐参数 dim-0 分片，模块边界决定通信分组，且 FSDP2 不直接提供与 FSDP1 相同的 full-state-dict 接口。两者的状态表示和迁移 API 不能混为同一行为。
- **证据等级与边界**：source-backed（PyTorch 2.15 官方版本文档）。文档描述 API 语义，不证明目标模型、设备和通信拓扑上的速度、峰值显存或故障恢复结果。

## Distributed Checkpoint 的拓扑变化能力

- **来源与版本**：[PyTorch Distributed Checkpoint tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)，页面标注 PyTorch 2.15.0+cu132；[DCP API](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)，访问日期 2026-10-30。
- **阅读位置**：教程的 “How DCP works” 与 loading 说明。
- **来源支持的事实**：DCP 支持多 rank 并行保存和加载，并可在加载时针对不同 cluster topology 重新分片；state-dict API 帮助映射模型和 optimizer 状态。
- **证据等级与边界**：source-backed。DCP 不会自动恢复未放入应用状态的随机数、数据游标、外部数据副作用或任意框架状态；教程页面和 API 仍应按部署 PyTorch 版本复核。

## Kubernetes device plugin 与 Kueue 的职责边界

- **来源与版本**：[Kubernetes device plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)，官方动态文档，访问日期 2026-10-30；[Kueue v0.20 admission](https://kueue.sigs.k8s.io/v0.20/docs/concepts/admission/) 与 [Kueue overview](https://kueue.sigs.k8s.io/docs/overview/)，访问日期 2026-10-30。
- **阅读位置**：device plugin registration、extended resources、Allocate 与 API compatibility；Kueue admission phases 和 core design principle。
- **来源支持的事实**：device plugin 向 kubelet 注册厂商设备，kubelet 将设备数量发布到节点资源；extended resources 按整数计量且不可 overcommit，Allocate 可提供容器运行时设备配置。Kueue 的 admission 可检查逻辑配额、可选的 topology-aware physical availability 和额外 AdmissionChecks；Kueue 文档将 Pod-to-node placement 与 job lifecycle 分别归于 kube-scheduler 和 kube-controller-manager。
- **证据等级与边界**：source-backed。Kubernetes 与 Kueue 动态文档可能随版本变化；拓扑准入、gang 和设备分配行为还受 scheduler 插件、workload integration、版本及集群配置约束，不能把队列准入等同于完整 placement。

## OpenTelemetry event 的表示

- **来源与版本**：[Signals](https://opentelemetry.io/docs/concepts/signals/)、[Logs Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/) 与 [Tracing API](https://opentelemetry.io/docs/specs/otel/trace/api/)，官方动态文档，访问日期 2026-10-30。
- **阅读位置**：signals 分类；Log Data Model 的 Events 与 EventName；Tracing API 的 Span Event。
- **来源支持的事实**：OpenTelemetry LogRecord 可表示事件并带 EventName；trace 中的 Event 记录在 Span 上。平台也可以使用自己的事件流存放作业状态、发布和故障变更。
- **证据等级与边界**：source-backed。不同产品对“event”一词的模型和保留语义可能不同；不要把平台事件类别直接表述为 OpenTelemetry 独立顶层 signal。

## NIST AI RMF 的性质

- **来源与版本**：[NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)，动态官方页面，访问日期 2026-10-30。
- **阅读位置**：Overview 与 Framework status。
- **来源支持的事实**：NIST 将 AI RMF 定位为自愿使用，旨在帮助把可信赖性考虑纳入 AI 产品、服务和系统的设计、开发、使用与评估；页面显示 AI RMF 1.0 正在修订。
- **证据等级与边界**：source-backed。根据 NIST 将 AI RMF 定位为自愿使用，本章把“采用框架不等于合规证明”作为适用范围推论，而非 NIST 页面直接声明；实际义务和风险控制取决于适用辖区、业务和威胁模型。

## MFU 与 dominant share 的原始来源

- **来源与版本**：[PaLM: Scaling Language Modeling with Pathways, arXiv v5](https://arxiv.org/abs/2204.02311v5)，2022-10-05；[Dominant Resource Fairness](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types)，NSDI 2011。
- **阅读位置**：PaLM 的 Training Efficiency 与 MFU/HFU 定义；DRF 论文的多资源公平分配机制。
- **来源支持的事实**：PaLM 将 MFU 描述为观察到的模型 FLOPs 吞吐相对理论峰值吞吐的比例，并报告特定 TPU/模型训练配置下的结果。DRF 使用各类资源份额中的最大值作为 dominant share，进一步在多用户和异构资源需求条件下构造分配机制。
- **证据等级与边界**：source-backed（原始论文）。PaLM 的观测数值不构成其他设备的性能目标；本章的 DRF 代码仅计算静态份额，不实现完整的 DRF 分配过程。
