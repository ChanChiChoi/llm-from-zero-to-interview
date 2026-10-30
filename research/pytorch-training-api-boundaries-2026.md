# PyTorch 训练组件 API 边界

检索与阅读日期：2026-10-30。以下结论限定于列明的 PyTorch 2.12 官方文档；不据此推断其他版本、设备或生产配置。

## 归一化、优化与混合精度

- [BatchNorm1d](https://docs.pytorch.org/docs/2.12/generated/torch.nn.BatchNorm1d.html)：训练前向方差使用有偏估计（`correction=0`），写入 running variance 的批统计使用无偏估计（`correction=1`）；`track_running_stats=False` 时评估也使用当前 batch 统计。其 `momentum` 是运行统计的更新权重，定义不同于优化器 momentum。
- [AdamW](https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html)：文档公式将 weight decay 直接施加到参数，不把它累积进一阶/二阶矩；foreach、fused、capturable 等配置属于具体实现选项。
- [Automatic Mixed Precision](https://docs.pytorch.org/docs/2.12/amp.html)：设备类型作为 `torch.amp.GradScaler` / `torch.amp.autocast` 的显式参数；旧的 `torch.cuda.amp` 与 `torch.cpu.amp` scaler/autocast 命名空间已标记弃用。CPU BF16 autocast 与 CUDA FP16 loss scaling 是不同用法；GradScaler 示例需按设备和精度配置。

## 数据、分布式、重计算和测量

- [Data Loading](https://docs.pytorch.org/docs/2.12/data.html)：多进程加载会为 worker 建立 Dataset 副本；IterableDataset 若未自行分片可能重复产生数据。worker 访问父进程中的大型 Python 对象会增加 CPU 内存，`pin_memory=True` 用于页锁定 host tensor 并支持更快的 CUDA 数据传输。
- [DistributedDataParallel](https://docs.pytorch.org/docs/2.12/generated/torch.nn.parallel.DistributedDataParallel.html)：在 `no_sync()` 中必须把 forward 和 backward 都包含在上下文内；DDP 初始化前应注册完整参数与 buffer，进程间参数注册顺序须一致，之后不应增删。
  - DDP 默认对各进程间的梯度做平均；若各 rank 传入的是 loss sum，而非各自的样本均值，梯度尺度相对单进程全局 batch 目标会多一个 world-size 因子。含 token mask 的语言模型还需以全局有效 token 总数作为共同分母；最后不均匀输入配合 `join()` 时，权重取决于 `divide_by_initial_world_size`。
- [Activation checkpoint](https://docs.pytorch.org/docs/2.12/checkpoint.html)：反向阶段会重算所选前向段；默认会保存/恢复 RNG 状态以使包含随机操作的重算输出与普通前向一致，但这会产生开销。默认状态管理覆盖 CPU 和一种推断出的非 CPU 设备类型，函数内部设备迁移或多种设备可能超出追踪范围。
- [CUDA semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html)：GPU 操作默认异步入队；主机时钟若未等待设备完成，不能给出设备执行时间。精确测量可使用 `torch.cuda.synchronize()` 或 CUDA event。
- [Profiler](https://docs.pytorch.org/docs/2.12/profiler.html)：采集 operator shape 或调用栈会增加开销；shape tracing 会暂时保留 tensor 引用，并可能阻止依赖引用计数的优化或产生额外拷贝。
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/2.12/torch_cuda_memory.html)：PyTorch allocator 提供活动 tensor 与缓存分配器内存的分层视图；这类统计只覆盖 PyTorch allocator 可见的分配，不自动等于设备上所有外部 CUDA 分配。

证据等级：以上为 source-backed 的版本化 API 语义，不是本地执行结果、硬件性能保证或算法普遍结论。
