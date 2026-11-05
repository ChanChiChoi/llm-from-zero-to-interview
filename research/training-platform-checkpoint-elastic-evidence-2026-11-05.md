# 训练平台 checkpoint 与弹性恢复证据记录

资料核验日期：2026-11-05

本记录只保留分布式 checkpoint、worker 重启和实验追踪资料所支持的实现事实及边界。框架能力会随版本变化，不能直接外推为任意训练平台的通用承诺。

## PyTorch Distributed Checkpoint

核验版本：PyTorch 2.15 官方文档。

- Distributed Checkpoint 支持多 rank 并行保存和加载，并说明可在加载时执行 reshard，使 checkpoint 能从一种集群拓扑加载到另一种拓扑；保存会产生多文件结构。[PyTorch Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)
- 同一文档明确指出，保存的 state_dict 不保证跨 PyTorch 版本向后兼容；文档提供的通用 state_dict API 仍标为 experimental。平台需要绑定框架版本、checkpoint 格式和使用 API，升级时实际验证恢复路径。
- 以上内容支持“特定 API 能在其支持范围内重新分片”，不支持“任意并行策略、模型格式或框架之间均可无损迁移”，也不支持“改变 world size 后优化轨迹保持等价”。训练状态恢复仍须检查 optimizer、scheduler、随机状态、数据游标和 effective batch。

## PyTorch torchrun 与 worker group

核验版本：PyTorch 2.15 官方文档。

- worker 故障时，worker group 中的其他 worker 会停止并按配置重启；节点成员变化会形成新的 worker group，`RANK` 与 `WORLD_SIZE` 可以改变，`RANK` 不稳定。[PyTorch torchrun 2.15](https://docs.pytorch.org/docs/2.15/elastic/run.html)
- 这描述的是 launcher/worker 管理行为，不等于应用的 optimizer、数据读取位置或 RNG 自动恢复。应用仍须保存并加载 checkpoint，重新初始化进程组，并处理 rank、world size 和数据分片变化。
- 该事实支持将 attempt、membership-change event 和 task 状态分别建模；不能据此推断固定 world size 与弹性训练有相同的样本顺序或优化轨迹。

## DeepSpeed Universal Checkpoint

核验版本：DeepSpeed 当前在线 Model Checkpointing 文档（页面显示 0.19.10）。

- 文档将 Universal Checkpoint 标记为 under development，并说明其目标是支持跨 GPU 数量/3D 并行布局恢复。[DeepSpeed Model Checkpointing](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- 这只说明当前文档中功能的状态和目标，不说明它已适用于所有组合，也不构成与 PyTorch Distributed Checkpoint 格式互通的证据。

## MLflow Tracking 的职责范围

MLflow 官方 Tracking 文档将其描述为记录运行参数、代码版本、指标和输出文件的 API/UI，并以 runs、models、experiments 组织实验记录。[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)

这是产品文档中的公开功能说明。由此只能说明它可承担实验记录组件；平台是否另有租户授权、GPU 调度、资源账本、恢复控制和数据血缘，要由实际系统设计决定。

## 证据边界

- 以上来源均为特定版本的官方文档，能说明公开接口与维护者声明的行为，不能单独证明特定集群的性能、存储一致性或灾难恢复正确性。
- 教学算例中的 batch、有效 token、step 时间、保存/评估耗时和 GPU 单价均为构造输入，不是来源报告或生产测量值。
