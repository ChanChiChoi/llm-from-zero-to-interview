# Checkpoint、评估与实验管理：来源证据

核验日期：2026-11-02。

本记录只保存联网核验到的技术事实、来源版本、实际阅读位置和适用边界；章节审阅决定与工作进度写入 `progress/`。

## PyTorch 2.15 Distributed Checkpoint 与异步保存

- **来源与版本：** [Distributed Checkpoint 2.15](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)、[Asynchronous Saving with DCP 2.15 教程](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)。官方页面标注 PyTorch 2.15 / 2.15.0+cu132。
- **阅读范围：** DCP state-dict saver/load API、rank-local shard 保存说明、`async_save` 的 staging 和 future 文档、`AsyncSaveResponse` 的字段说明，以及异步教程的 checkpoint 排队和 pinned-memory staging 例子。
- **来源支持的事实：** `async_save` 默认先把 state dict staging 到 CPU 内存，再由独立线程执行保存；返回的 `staging_completion` 表示本地 state dict 副本就绪，`upload_completion` 表示 checkpoint 保存完成。默认 staging 副本使训练随后对模块状态的更新不应改变正在序列化的快照。官方教程示例在启动下一次保存前等待上一保存 future，以避免无界排队；也展示使用缓存 pinned-memory buffer 的 staging 优化。文档明确不保证已保存 state dict 在不同 PyTorch 版本间向后兼容。
- **证据等级：** source-backed（指定 PyTorch 版本 API 与官方教程）。
- **适用边界：** CPU/pinned-memory 用量、异步重叠收益和在途保存数取决于模型状态、stager、设备复制和 storage writer。staging 完成不等于持久化上传完成；DCP 接口也不自动定义应用自己的数据 cursor、manifest、对象存储事务或版本发布协议。版本升级必须使用真实训练状态做恢复演练。

## PyTorch 2.15 随机性与可复现范围

- **来源：** [PyTorch 2.15 Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)。
- **阅读范围：** 开头关于版本、commit、平台和 CPU/GPU 的可复现声明，以及随机源和确定性算法说明。
- **来源支持的事实：** 官方不保证跨 PyTorch release、单独 commit、平台或 CPU/GPU 完全可复现，即使使用相同随机种子也可能不同；确定性选项可以限制部分非确定来源，但通常会降低性能。
- **证据等级：** source-backed（指定版本官方限制说明）。
- **适用边界：** 这说明不能承诺任意环境逐 bit 相同，不代表可以省略数据顺序、RNG、依赖、硬件、checkpoint 和评估脚本的记录。

## Hugging Face Transformers Trainer 当前接口文档

- **来源：** [Trainer 当前官方文档](https://huggingface.co/docs/transformers/main/main_classes/trainer) 与[训练配方](https://huggingface.co/docs/transformers/trainer_recipes)，访问日期 2026-11-02。当前页面说明其 `main` 版本要求从源码安装，并把最新稳定版显示为 v5.22.1。
- **阅读范围：** `resume_from_checkpoint` 参数说明、保存与 best-model 相关配置、恢复训练配方。
- **来源支持的事实：** `resume_from_checkpoint` 可指定目录或选择 `output_dir` 最近 checkpoint；文档称恢复会加载 model、optimizer 和 scheduler 状态。配方另说明恢复训练依赖 checkpoint 中 optimizer 和 scheduler 状态文件，`save_only_model=True` 等省略这些状态的保存方式不能作为完整优化器续训点。
- **证据等级：** source-backed（访问日官方当前文档；页面为动态路径）。
- **适用边界：** 这是 Trainer 的公开接口语义，不规定自定义训练循环或其他分布式框架的 checkpoint 格式；`main` 页面会随发布变化，具体运行应依赖锁定的 Transformers 版本重新核对。

## HELM 的场景与指标覆盖口径

- **来源：** Liang et al., [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110v2)，arXiv v2，2023-10-01；本次核对 arXiv 摘要，访问日期 2026-11-02。
- **来源支持的事实：** HELM 论文定义 16 个 core scenarios，并在可行时对每个场景测 7 类指标；另执行覆盖 26 个 targeted scenarios 的 7 组 targeted evaluations。论文评估 30 个模型的 42 个场景，其中 21 个此前未用于主流模型评测；作者报告，HELM 之前模型平均只覆盖 core scenarios 的 17.9%，其统一评估将所测 30 个模型的覆盖提高到 96.0%。
- **证据等级：** source-backed（论文摘要中的 benchmark 定义和作者报告）。
- **适用边界：** 这些比例描述论文当时定义的场景集合、抽样到的模型与可行性协议；不表示覆盖了所有现今任务或生产场景，也不证明不同指标可以合成无损总排名。

## 生命周期

PyTorch DCP 2.15 与 Trainer 页面记载为访问日的版本化接口事实；后续正文升级应以目标依赖版本的官方文档和恢复演练重新确认。本文不把异步 API 描述成对象存储事务，也不把固定 seed 描述成跨环境逐位复现保证。

## TierCheck 分层 checkpoint 研究

- **来源与版本：** Han et al., “TierCheck: Tiered Checkpointing for Fault Tolerance in Large Language Model Training,” arXiv:2605.17821，2026-05-18 发布；[arXiv 正文](https://arxiv.org/abs/2605.17821)，2026-11-02 查阅。
- **实际阅读范围：** §3 的保存、检索与回收设计，§5.1 测试环境，§5.2 checkpoint 频率和失败恢复结果。
- **来源支持的事实：** TierCheck 将高频差分 checkpoint 放入本机与物理邻机的易失内存，将完整基点异步迁移到远端持久层；使用跨 rank 全局提交标记及 watermark 控制恢复和回收。作者测试平台为四节点、16 张 A800 80 GiB、200 Gbps InfiniBand，远端存储为 NVMe-backed Lustre，经聚合 400 Gbps Ethernet；软件栈包括 PyTorch 2.6.0 与 DeepSpeed 0.16.4。作者评估 10B—40B 配置（最高 40B/16 卡）。20B 模型在不超过 3.5% 训练速度下降的约束下，最短 checkpoint 间隔为 5 个训练迭代；20B GPT-2 的软件/节点/机架故障恢复时间为 8.8/18.4/23.9 秒。
- **证据等级与边界：** source-backed（预印本中的系统设计和作者测试）。吞吐、保存频率、恢复延迟受其硬件、网络、存储、差分格式和故障注入协议约束；不是 PyTorch DCP 功能承诺，也不是其他集群可直接采用的 SLA。论文的 124M GPT-2 收敛实验与 20B—40B 系统压力测试承担不同验证目的，不能把系统规模结果误解为完整大模型质量训练复现。
