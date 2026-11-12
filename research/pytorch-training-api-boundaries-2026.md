# PyTorch 训练组件 API 边界

原始检索与阅读日期：2026-10-30，以下条目限定于各自列明的 PyTorch 2.12 官方文档。2026-11-09 补充的 PyTorch 2.15 自动微分、编译和性能分析资料单独标明版本；不据此推断其他版本、设备或生产配置。

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

## DataLoader 与 Dataset（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[torch.utils.data](https://docs.pytorch.org/docs/2.15/data.html)，PyTorch 2.15 官方文档；[Data Loading Optimization 教程](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)，标示为 PyTorch Tutorials 2.15.0+cu132。
- 实际阅读位置：Dataset Types / Data Loading Order / Automatic Batching / Multi-process Data Loading / Randomness / Memory Pinning / DataLoader 参数与 DistributedSampler。另读教程中的 `__getitems__`、`in_order` 与性能测量章节。
- 来源支持的事实：
  - map-style Dataset 必须支持 `__getitem__`；`__len__` 对 sampler 和 DataLoader 默认行为常用但可选；Dataset 可选实现 `__getitems__(indices)` 供 DataLoader 批量取样。
  - IterableDataset 的顺序由用户定义的数据流决定，不支持 map-style 的 sampler / batch_sampler；多进程时每个 worker 都有 Dataset 副本，需显式配置不同 shard。多 worker 流式加载的 `drop_last` 对每个 worker 副本分别丢弃尾部不完整 batch；`len(dataloader)` 对 iterable 数据集是基于长度的估计，分片后可能不准。
  - DataLoader worker 的 PyTorch seed 默认是 base seed 加 worker id；base seed 可由 `generator` 控制。Python/NumPy 等其他库可能在 worker 间得到相同随机状态，文档建议通过 `worker_init_fn` 和 `get_worker_info().seed` 或 `torch.initial_seed()` 单独初始化。
  - 多进程 worker 访问父进程的大型 Python 对象时，CPU 内存可能随 worker 数上升。默认 memory pinning 识别 tensor 及常见映射/容器；自定义 batch 类型需实现 `pin_memory()`。
  - 2.15 的 DataLoader 有 `in_order` 参数，默认 FIFO。关闭后可按 worker 完成次序交付 batch；文档警告这会影响可复现性，并可能在不平衡数据上造成训练分布偏斜。`prefetch_factor` 表示每个 worker 预取的 batch 数。
  - `DistributedSampler` 的各 rank 应使用相同 seed 并在 epoch 开始前调用 `set_epoch()`；`drop_last=False` 会补索引，`True` 会截去数据集尾部使样本数可整除。
- 证据等级：source-backed（固定 2.15 API 文档及教程）；教程给出的性能结果只支持该教程实验环境。
- 适用边界：文档不证明目标 OS、Python 启动方式、实际数据源、真实随机增强、rank/worker 拓扑、内存峰值或吞吐性能；这些都需要对应环境和数据源的独立证据。
- 未解决的事实问题：`in_order=False` 在具体 workload 上是否有收益、远程流的故障与恢复边界，以及自定义 batch pinning 的实际行为，都不能从 API 文档单独推出。

## 反向自动微分（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)，PyTorch 2.15 官方开发者文档。
- 实际阅读位置：§“How autograd encodes the history”（页面第 417–422 行），包括反向自动微分、前向操作图、按链式法则从输出回传梯度，以及每次迭代重新创建计算图的说明。
- 来源支持的事实：PyTorch autograd 在前向执行时记录操作图，并在反向阶段沿图用链式法则计算梯度；动态图在每次迭代重新建立。
- 证据等级：固定版本官方文档所述的公开 autograd 语义。
- 适用边界：该文档解释 PyTorch 的自动微分机制，不证明特定自定义算子、融合 kernel、低精度梯度或硬件执行细节；数学例子的导数结果由本章手算验证。

## 编译、性能分析与混合精度接口（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[torch.compile 入门教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)，PyTorch 2.15 教程。
  - 实际阅读位置：基本用法、性能比较、Graph Breaks 与 Troubleshooting。
  - 来源支持的事实：教程描述 `torch.compile` 捕获并编译可支持的 PyTorch 计算；遇到不支持的 Python 代码时可中断图编译、执行该部分后继续，而 graph break 会损失优化机会。`fullgraph=True` 可让首个 graph break 暴露为错误。教程指出性能变化取决于模型结构、batch size 和瓶颈，示例速度不能外推为一般加速保证。
  - 证据等级：source-backed（PyTorch 2.15 官方教程的公开行为说明和教学示例）。
  - 适用边界：具体编译覆盖、重编译、冷启动、图断点和性能取决于代码、输入形状、硬件、后端及版本；教程示例不能支持目标生产 workload 的收益结论。
- 来源：[torch.profiler API](https://docs.pytorch.org/docs/2.15/profiler.html) 与 [Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)，PyTorch 2.15 官方文档，2026-11-09 阅读。
  - 实际阅读位置：Profiler 概览、采样参数、周期调度、operator/kernel 时间、shape/stack 和 memory 分析。
  - 来源支持的事实：Profiler 可采集 CPU 与设备活动、算子输入 shape、调用栈和 tensor 内存事件；收集 shape/stack 会引入测量开销，shape tracing 还可能暂时持有 tensor 引用并影响优化或导致额外拷贝。
  - 证据等级：source-backed（版本化 API 文档与教程）。
  - 适用边界：trace 的开销和可见范围受采集项、设备、编译模式及版本影响；Profiler 结果是诊断信号，不等同于无扰动的端到端基准。
- 来源：[PyTorch Benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)，PyTorch 2.15 官方教程，2026-11-09 阅读。
  - 实际阅读位置：Python `timeit` 与 `torch.utils.benchmark.Timer` 对照、线程管理和 CUDA 同步注意事项。
  - 来源支持的事实：PyTorch 基准测量需处理线程数与 CUDA 异步执行；仅用未经同步的主机计时不能可靠代表 GPU 运算完成时间。
  - 证据等级：source-backed（官方基准教程）；适用范围限于其测量说明。
- 来源：[torch.amp API](https://docs.pytorch.org/docs/2.15/amp.html)，PyTorch 2.15 官方 API 文档，2026-11-09 阅读。
  - 实际阅读位置：AMP 概览、autocast 与 GradScaler、弃用警告，以及 AMP 与 `torch.compile` 组合说明。
  - 来源支持的事实：2.15 文档将 `torch.cuda.amp` / `torch.cpu.amp` 的 autocast 与 GradScaler 旧入口标为弃用，建议使用带设备类型参数的 `torch.amp` 入口；该版本还记录了 compiled backward autocast 行为与推荐的 AMP 前向/反向上下文不同步时的配置边界。
- 证据等级：source-backed（PyTorch 2.15 官方 API 文档）。
- 适用边界：编译与 AMP 的组合配置须按具体版本和训练结构核验；不把内部配置项写成跨版本稳定接口，也不据 API 文档推出数值稳定性或吞吐结论。

## 张量 shape、索引与存储视图（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[Tensor Views](https://docs.pytorch.org/docs/2.15/tensor_view.html)、[`Tensor.view`](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.view.html)、[`torch.reshape`](https://docs.pytorch.org/docs/2.15/generated/torch.reshape.html)、[`Tensor.contiguous`](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.contiguous.html) 和 [`Tensor.expand`](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.expand.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：view 与原 storage 的关系、stride 兼容条件、reshape/flatten 的 view-or-copy 行为、basic/advanced indexing、expand 的 stride 0 和 `contiguous()`。
  - 来源支持的事实：`view` 需满足原 size/stride 的兼容条件并共享数据；`reshape` 可能返回 view 或复制，官方要求调用者不要依赖其中任一种；`contiguous()` 在输入已符合请求布局时可返回原对象，否则复制；basic indexing 通常返回 view，advanced indexing 返回 copy；`expand` 通过零 stride 创建共享存储视图，多个逻辑元素可能对应同一内存位置，因此原地写入有风险。
  - 证据等级：source-backed（PyTorch 2.15 版本文档）。
  - 适用边界：是否复制、具体 stride 和支持的布局与输入、memory format、设备后端有关；文中连续布局例子限 dense strided tensor，不推广到所有稀疏、嵌套或非 strided 后端。
- 来源：[Broadcasting semantics](https://docs.pytorch.org/docs/2.15/notes/broadcasting.html)、[`torch.matmul`](https://docs.pytorch.org/docs/2.15/generated/torch.matmul.html) 与 [`torch.bmm`](https://docs.pytorch.org/docs/2.15/generated/torch.bmm.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：广播维度从尾部对齐、matmul 的 batch 维规则，以及 bmm 的 batch 和无广播限制。
  - 来源支持的事实：支持广播的逐元素操作从尾维对齐，并允许维度相等、为 1 或缺失；`matmul` 会广播前置 batch dimensions；`bmm` 不做广播。
  - 证据等级：source-backed（固定版本 API 语义）。
  - 适用边界：张量 shape 可广播只说明算子可执行，不证明 batch/head/query/key 维度的语义正确；本记录不推导算子吞吐。
- 来源：[`torch.as_tensor`](https://docs.pytorch.org/docs/2.15/generated/torch.as_tensor.html) 与 [Tensor Attributes](https://docs.pytorch.org/docs/2.15/tensor_attributes.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：数据共享条件、dtype/device 转换行为和设备自动迁移说明。
  - 来源支持的事实：`as_tensor` 在条件允许时共享输入数据；dtype 或 device 不同会发生转换/复制。官方文档指出设备间一般不会自动搬运非标量 tensor；PyTorch 文档对 0 维标量有特例说明。
  - 证据等级：source-backed（固定版本 API 语义）。
  - 适用边界：是否共享取决于输入类型、dtype、device 和转换要求；不要把 `as_tensor` 视为总是零拷贝。
- 来源：[`torch.tensor`](https://docs.pytorch.org/docs/2.15/generated/torch.tensor.html) 与 [`torch.as_tensor`](https://docs.pytorch.org/docs/2.15/generated/torch.as_tensor.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：构造语义与 See also 对照说明。
  - 来源支持的事实：`torch.tensor(data)` 复制输入并创建新的 leaf tensor；`torch.as_tensor` 尽可能共享已有数据并保留可保留的 autograd 历史，遇到 dtype/device 转换时复制。
  - 证据等级：source-backed（固定版本 API 描述）。
  - 适用边界：Python list 等输入没有现成 tensor storage 可共享；本条不把“尽可能共享”扩展为所有输入都零拷贝。
- 来源：[`torch.nn.Linear`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Linear.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：前向公式、输入/输出形状及 `weight` 参数定义。
  - 来源支持的事实：模块计算 `y=xA^T+b`；参数 `weight` 的形状为 `(out_features,in_features)`，输入最后一维为 `in_features`，其余前置维度保留。
  - 证据等级：source-backed（固定版本 API 语义）。
  - 适用边界：这是 `nn.Linear` 的公开接口约定；不用于推断某个设备上的矩阵乘法 kernel、累积精度或性能。
- 来源：[`torch.nn.functional.cross_entropy`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cross_entropy.html) 与 [`CrossEntropyLoss`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：input/target shape、类别数、class-index 目标范围、`ignore_index` 和 reduction 定义。
  - 来源支持的事实：logits 输入支持 `(C)`、`(N,C)` 或 `(N,C,d_1,...,d_K)`；对多维输入分类维为 `dim=1`，一维 `(C)` 输入仅有类别轴 `dim=0`。类索引目标需在合法类别范围内，或按约定等于 `ignore_index`；忽略目标不参加梯度，mean reduction 对未忽略目标做平均。
  - 证据等级：source-backed（PyTorch 2.15 官方 API 文档）。
  - 适用边界：这不意味着任意 `(B,T,V)` logits 都可直接传入；需调整类别轴或展平。全目标被忽略时均值没有有效分母，具体异常/非有限行为应按实际版本核验。

## 反向梯度与计算图生命周期（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[Autograd mechanics](https://docs.pytorch.org/docs/2.15/notes/autograd.html)，PyTorch 2.15 官方说明。
  - 实际阅读位置：“Setting `requires_grad`”“Grad Modes”“In-place operations with autograd”及反向 hooks 章节。
  - 来源支持的事实：只有在梯度模式开启且至少一个输入需要梯度时，前向操作才记录到反向图；默认只有 `requires_grad=True` 的叶子张量在 `.backward()` 后累积 `.grad`；每次前向按实际执行路径重建动态图。原地操作可能覆盖反向需要的值，版本计数用于检测被保存张量在前向与反向之间是否改变；官方文档不建议在多数场景普遍使用原地操作，因为可观内存收益有限且会增加图管理复杂度。
  - 证据等级：source-backed（固定版本官方开发者文档）。
  - 适用边界：版本计数只检测被 Autograd 保存并检查的值；源码无报错不证明某种原地写法有性能收益。自定义算子和不同后端需另行验证。
- 来源：[`torch.autograd.grad`](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad.html) 与 [`torch.autograd.backward`](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.backward.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：函数摘要、`grad_outputs`、输入/输出参数、`create_graph` 与 `retain_graph` 描述。
  - 来源支持的事实：`grad` 返回指定输入的梯度而不累积到输入 `.grad`；`grad_outputs` 提供向量—雅可比积中的向量；`create_graph=True` 会为导数继续建图，`retain_graph` 默认跟随 `create_graph`，且官方说明大多数情况下无需手工保留图。`.backward()` 则将叶子张量梯度累积至 `.grad`。
  - 证据等级：source-backed（版本化公开函数语义）。
  - 适用边界：`autograd.grad` 与标准 DDP 的参数梯度累积路径不同，不可从函数返回梯度推断 DDP 会自动同步它们。
- 来源：[`torch.no_grad`](https://docs.pytorch.org/docs/2.15/generated/torch.no_grad.html) 与 [`torch.inference_mode`](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：上下文说明、示例和两种 grad mode 的限制。
  - 来源支持的事实：`no_grad` 中普通运算的结果不需要反向梯度，但带 `requires_grad` 参数的 factory function 不受影响，文档举例 `nn.Parameter` 在该上下文中仍需要梯度；inference mode 另外关闭 view tracking 与版本计数，其创建的 Tensor 不能进入 Autograd 记录的计算，并且不会自动调用 `model.eval()`。官方建议只有在确认不会与 Autograd 交互时使用 inference mode，难以保证时应比较其收益并考虑 `no_grad`。
  - 证据等级：source-backed（版本化公开上下文语义）。
  - 适用边界：官方接口说明不保证特定模型上的速度收益；训练前向、view/原地交互和 Tensor 跨上下文使用需按目标路径检查。
- 来源：[`Optimizer.zero_grad`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：`set_to_none` 参数及其行为说明。
  - 来源支持的事实：`set_to_none=True` 是该版本默认值，通常降低内存占用并可能略微提高性能；未收到梯度的参数保持 `None`，与显式零梯度在访问结果及优化器是否跳过参数更新方面可能不同。
  - 证据等级：source-backed（固定版本参数默认值与语义）。
  - 适用边界：性能幅度因工作负载而异；自定义优化逻辑需区分 `None` 与全零梯度。
- 来源：[`Tensor.register_hook`](https://docs.pytorch.org/docs/2.15/generated/torch.Tensor.register_hook.html) 与 Autograd mechanics 中的“Backward Hooks execution”，PyTorch 2.15 官方文档。
  - 实际阅读位置：Tensor hook 签名和执行顺序。
  - 来源支持的事实：Tensor hook 在计算该 Tensor 梯度时触发；hook 不应原地修改收到的 `grad`，但可返回替代梯度；post-accumulate-grad hook 仅适用于叶子 Tensor。
  - 证据等级：source-backed（公开 API 文档）。
  - 适用边界：每个 Tensor hook 只提供该 Tensor 的梯度，不自动构成跨参数全局范数；记录 CUDA 数值到 Python 可能产生设备同步开销。
- 来源：[Activation checkpointing](https://docs.pytorch.org/docs/2.15/checkpoint.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：`checkpoint()` 参数说明和 reentrant / non-reentrant 比较。
  - 来源支持的事实：文档要求显式传 `use_reentrant`，并推荐 `False`；non-reentrant 版本保留 Autograd 图、支持 `torch.autograd.grad` 并默认可提前停止多余重算；reentrant 版本在前向时使用 `no_grad`，且反向接口和输入/输出梯度要求有所不同。
  - 证据等级：source-backed（版本化 API 用法和实现差异）。
  - 适用边界：重算一致性、随机状态和具体模型支持依赖调用路径；不推断目标硬件节省的显存/时间。
- 来源：[DistributedDataParallel](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：DDP 与 `torch.autograd.grad` 限制、`no_sync()` 说明和示例。
  - 来源支持的事实：标准 DDP 路径要求梯度累积到参数 `.grad`，不支持直接使用 `torch.autograd.grad`；`no_sync()` 中梯度跨 micro-step 累积，在离开上下文后的同步 forward/backward 中同步，且 forward 也必须处于上下文中。
  - 证据等级：source-backed（版本化 DDP 公开行为）。
  - 适用边界：API 约束不证明生产集群中的通信拓扑、性能、容错或故障恢复行为。
- 来源：[Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)，PyTorch 2.15 官方说明。
  - 实际阅读位置：“Working with Unscaled Gradients / Gradient clipping”及“Gradient accumulation”。
  - 来源支持的事实：梯度裁剪前需先将 GradScaler 管理的梯度 unscale；`unscale_` 每个 optimizer 每次 step 调用一次，且应在该有效 batch 的梯度全部累积之后；累积期间应保持同一 scale，scaler 的 step/update 应按有效 batch 更新。
- 证据等级：source-backed（官方 AMP 示例和注意事项）。
- 适用边界：该结论限于使用 GradScaler 的混合精度梯度缩放路径，不等同于 BF16 或未缩放训练的通用步骤。

## Module 注册、checkpoint 与运行包装（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[`nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)、[`nn.Parameter`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parameter.Parameter.html)、[`ModuleList`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleList.html) 与 [`ModuleDict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleDict.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：参数和子模块注册、buffer 的 `persistent`、`state_dict()`、`get_extra_state()` / `set_extra_state()`、`load_state_dict(assign=...)`、设备迁移和 `train()` / `eval()`。
  - 来源支持的事实：赋给 Module 属性的子模块和 `Parameter` 会注册；`ModuleList` / `ModuleDict` 会注册所含子模块，普通 Python `Tensor` 赋值不具有 `Parameter` 的自动注册行为。默认 state dict 包含注册参数和持久 buffer，并以浅拷贝形式返回对它们的引用；自定义 module 可用 extra-state 方法加入额外值。`load_state_dict(assign=False)` 默认复制到目标对象，`assign=True` 更多保留 state dict 张量属性，官方说明除启用参数交换机制外应在加载后创建 optimizer。`Module.cuda()` 会使相关参数/buffer 成为不同对象，文档要求对随后要优化的模块先迁移、再构造 optimizer。`eval()` 只影响会响应 `training` 状态的模块行为。
  - 证据等级：source-backed（固定版本公开 API 语义）。
  - 适用边界：自定义 extra state 需可序列化；PyTorch 对跨版本序列化的兼容承诺主要覆盖 tensor。其他后端的模块迁移与参数交换应核对相应 API。`track_running_stats=False` 的 BatchNorm 会在 eval 时继续使用当前 batch 统计，依据 [`BatchNorm1d`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html) 文档；不能从 `eval()` 推出输出必定对 batch 组成不敏感。
- 来源：[`torch.load`](https://docs.pytorch.org/docs/2.15/generated/torch.load.html) 与 [serialization semantics](https://docs.pytorch.org/docs/2.15/notes/serialization.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：`weights_only` 参数、受限 unpickler、加载设备映射和自定义 safe globals。
  - 来源支持的事实：PyTorch 2.15 `torch.load` 的默认 `weights_only` 为 `True`；文档说明该模式限制可反序列化的对象类型并禁止动态导入，较普通 pickle 路径缩小执行风险，但不防御拒绝服务或所有下游内存风险。PyTorch 2.6 起在未显式提供 pickle module 时默认启用 `weights_only=True`。自定义类或 tensor subclass 可能需要 allowlist；文档明确警告不要加载不可信来源的数据。
  - 证据等级：source-backed（固定版本序列化 API 与安全说明）。
  - 适用边界：受限加载不是对任意 checkpoint 的完整安全证明；配置和 extra state 的具体对象图需按运行版本核对，不能把受限路径描述成“安全加载任意文件”。
- 来源：[`DistributedDataParallel`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：参数注册顺序、DDP 构造时梯度归约 hook、包装后的模块变更警告、`module.` state dict 前缀。
  - 来源支持的事实：DDP 假设所有参数和 buffer 在构造前已注册，之后不应增删；各 rank 的参数注册顺序必须一致。DDP 构造时为当时参数安装梯度归约函数。加载 DDP state dict 到非 DDP 模型时，官方说明使用 `consume_prefix_in_state_dict_if_present()` 去掉 `module.` 前缀。
  - 证据等级：source-backed（版本化分布式 API 约束）。
  - 适用边界：这是 DDP 的模块树和同步前提，不替代多进程、通信拓扑或故障恢复测试。
- 来源：[torch.compile troubleshooting](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)，PyTorch 2.15 官方开发者文档。
  - 实际阅读位置：“Setting Expectations”“Graph break” 和 “Where to apply torch.compile?”。
  - 来源支持的事实：graph break 会将受支持图段编译、执行不支持片段后继续捕获；初次调用可能包含编译冷启动，输入或 guard 变化可能触发重编译。文档建议选取问题较少的高层函数、顶层 module 或子 module；针对 DDP/FSDP wrapper，建议考虑编译传给 wrapper 的内部模型。
  - 证据等级：source-backed（版本化编译器排障建议）。
  - 适用边界：不保证特定模块组合的兼容性、数值等价或加速；目标代码与硬件仍需按相应证据独立核验。

## 训练循环、优化器与 checkpoint（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[`Optimizer.zero_grad`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.zero_grad.html)、[`torch.optim` 调度说明](https://docs.pytorch.org/docs/2.15/optim.html#how-to-adjust-learning-rate)、[`LRScheduler`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.LRScheduler.html) 与 [`Optimizer.load_state_dict`](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Optimizer.load_state_dict.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：`set_to_none` 行为、调度器的调用顺序和版本说明、scheduler/optimizer state 恢复警告。
  - 来源支持的事实：`zero_grad(set_to_none=True)` 通常节省内存且可能改善性能；在后续 backward 未产生梯度的参数上保留 `.grad=None`，而全零梯度与 `None` 对 optimizer 的处理可能不同。PyTorch 1.1.0 起文档要求 scheduler 在 optimizer 更新后调用；反向调用会跳过 schedule 的首个值。恢复时应先初始化 scheduler，再加载 optimizer state，以免 scheduler 初始化覆盖已保存学习率。
  - 证据等级：source-backed（PyTorch 2.15 公开 API 语义和迁移说明）。
  - 适用边界：优化器对缺失梯度和零梯度的具体更新差异取决于 optimizer 算法及其状态；scheduler 的调用频率还须匹配配置所说的 epoch、batch 或 optimizer update。
- 来源：[`clip_grad_norm_`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.utils.clip_grad_norm_.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：返回值、原地梯度修改及 `error_if_nonfinite` 参数。
  - 来源支持的事实：全局 norm 把各参数梯度视为拼接后的向量；函数原地缩放梯度，返回裁剪前总范数；`error_if_nonfinite=True` 会在总范数为 NaN/Inf 时抛错，默认值为 `False`。
  - 证据等级：source-backed（固定版本函数接口和语义）。
  - 适用边界：文档描述 API 行为，不证明特定阈值能稳定目标训练；GradScaler 路径还要按 AMP 规则先 unscale，再裁剪。
- 来源：[`torch.Generator`](https://docs.pytorch.org/docs/2.15/generated/torch.Generator.html) 与 [Reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：`get_state()` / `set_state()`、DataLoader 的 `generator` / `worker_init_fn` 和确定性算法说明。
  - 来源支持的事实：Generator state 是 `ByteTensor`，可交给 `set_state()` 恢复到特定生成器状态；DataLoader 文档将 generator 与 map-style 随机采样及 worker base seed 联系起来。`torch.use_deterministic_algorithms(True)` 会在有确定性实现时选用该实现，若某操作只有非确定实现则报错；该开关本身不足以保证整个应用可复现。
  - 证据等级：source-backed（版本化 API 和复现说明）。
  - 适用边界：恢复一个 Generator 不能自动恢复已启动 worker 的内部随机状态、预取队列、数据源游标或 sampler 正在消费的位置；这些依赖训练边界与数据管线协议。
- 来源：[Python 3.12 `os.replace`](https://docs.python.org/3.12/library/os.html#os.replace)，Python 官方标准库文档。
  - 实际阅读位置：替换目标文件、跨文件系统失败条件及原子重命名说明。
  - 来源支持的事实：目标存在且为文件时，`os.replace` 可替换它；操作可能因源和目标位于不同文件系统而失败；在 POSIX 上成功重命名是原子操作。
  - 证据等级：source-backed（Python 3.12 标准库 API 说明）。
  - 适用边界：**推论：** 原子名称替换不等同于文件内容和目录元数据已持久写入稳定存储；该文档没有因此证明断电持久性、对象存储原子性或多文件 checkpoint 一致性。

## 混合精度、激活重算与显存统计（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[Automatic Mixed Precision](https://docs.pytorch.org/docs/2.15/amp.html) 与 [Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：autocast context、FP16 gradient scaling 说明、gradient clipping 与 gradient accumulation 示例。
  - 来源支持的事实：autocast 按算子策略选择执行 dtype，应包住前向与 loss 而非 backward；启用 autocast 时不应手动把模型/输入整体转为 half 或 bfloat16。GradScaler 会检查梯度中的 Inf/NaN 并可跳过 optimizer 更新；scale 不保证大于 1，文档特别指出部分 BF16 预训练模型可能无法在最大有限值约 65,504 的 FP16 范围内工作。累积窗口内梯度应保持缩放且 scale 不变，只在窗口完整后 unscale 一次并更新 scale。
  - 证据等级：source-backed（固定版本公开 API 与示例）。
  - 适用边界：文档语义不保证特定模型、硬件或 batch 下兼容或加速；有效 token 不等长时，窗口 loss 分母仍由训练目标决定，不能从固定 batch 示例推成通用归一化方式。
- 来源：[Activation checkpointing](https://docs.pytorch.org/docs/2.15/checkpoint.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：checkpoint 变体差异、RNG state save/restore、`preserve_rng_state` 与 `use_reentrant` 参数说明。
  - 来源支持的事实：默认 RNG 管理保存 CPU 和一种推断出的非 CPU device type 状态；多个 device type 中其余类型未全部追踪，函数内部迁移到未覆盖的新设备也无法保证重算确定性。`preserve_rng_state=False` 可省去 RNG 保存/恢复；`torch.compile` 下该 flag 不生效且始终保留 RNG。PyTorch 2.15 推荐显式传 `use_reentrant=False`；该路径对 input/output 的 `requires_grad` 没有 reentrant 路径的限制。
  - 证据等级：source-backed（PyTorch 2.15 官方 API 文档）。
  - 适用边界：这些界限不等于对自定义随机源或所有设备组合的完整复现保证；目标版本和模型前向仍需单独核验。
- 来源：[CUDA semantics: memory management](https://docs.pytorch.org/docs/2.15/notes/cuda.html#memory-management)、[`max_memory_allocated`](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.max_memory_allocated.html) 与 [`reset_peak_memory_stats`](https://docs.pytorch.org/docs/2.15/generated/torch.cuda.memory.reset_peak_memory_stats.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：CUDA caching allocator 的 allocated/reserved 区分、`empty_cache()` 说明和 peak/reset API。
  - 来源支持的事实：allocated/max allocated 追踪 tensor 占用，reserved/max reserved 追踪 caching allocator 管理的总量；峰值默认从程序开始累计，可通过 reset 重置起点。`empty_cache()` 释放 allocator 未使用的缓存供其他 GPU 应用使用，不释放 tensor 占用且不增加 PyTorch 可用显存。
  - 证据等级：source-backed（版本化官方开发者/API 文档）。
  - 适用边界：PyTorch allocator 读数不覆盖 CUDA context、NCCL 或自定义分配的全部设备内存；峰值读数也不单独判定泄漏、碎片或实际可行的 workload。
- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，论文 v2（2022-06-23）。
  - 实际阅读位置：摘要及算法对 HBM 与片上 SRAM 访问的说明。
  - 来源支持的事实：FlashAttention 是 IO-aware 的精确 attention 算法，以 tiling 减少 HBM 与 SRAM 之间的读写；论文将常规 self-attention 的计算和内存问题与序列长度平方关系联系，并报告其算法降低 HBM 访问。
- 证据等级：source-backed（原始论文摘要/方法说明）。
- 适用边界：论文算法结果不能推广为任意 attention kernel 的实际速度、显存峰值或训练吞吐保证；稠密 attention 的 query-key 配对计算仍随序列长度平方增长。

## DDP、分布式采样与排错（PyTorch 2.15）

补充检索与阅读日期：2026-11-09。

- 来源：[Distributed overview](https://docs.pytorch.org/docs/2.15/distributed.html)、[DistributedDataParallel API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)、[DDP design note](https://docs.pytorch.org/docs/2.15/notes/ddp.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：DDP 初始化、`init_sync`、梯度 reducer、参数顺序/包装后增删限制、buffer 同步、`no_sync()`、unused parameters 和 checkpoint 提示。
  - 来源支持的事实：DDP 不负责切分输入，用户须自行划分数据；默认 `init_sync=True` 会校验参数形状并同步初始化参数/buffer；训练中的主要同步单位是梯度，DDP 假设各 rank 参数登记次序相同且包装后不添加/删除参数；每个 rank 的 buffer 默认在 forward 同步，但不等于跨 rank 合并 BatchNorm 的本地统计。标准 DDP reducer 的梯度累积目标是参数 `.grad`；`no_sync()` 必须同时包住 forward 和 backward，其梯度会在离开上下文后的同步 backward 中归约。`find_unused_parameters=True` 可以处理一部分未使用参数图，但会增加图遍历开销。官方另说明 `torch.autograd.grad()` 路径不由标准 DDP reducer 支持。
  - 梯度归一化推论：默认 reducer 对 rank 梯度平均；若 rank `r` 保留未归一化 loss sum `S_r`，有效项全局总数为 `N=Σ_r n_r`，则本地 backward 使用 `W*S_r/N` 可使归约后的梯度对应全局 token mean（假设无自定义 hook、无额外框架缩放且参与 rank 数为 `W`）。这是由默认平均语义推得的数学缩放；API 文档本身不规定语言模型 loss 分母。
  - 证据等级：source-backed（公开 API 语义）；loss 归一化部分为从 API reduction 推导的教学结论。
  - 适用边界：文档不证明目标 batch、模型、backend、节点拓扑或网络上的吞吐、数值复现和容错表现。
- 来源：[torchrun elastic launch](https://docs.pytorch.org/docs/2.15/elastic/run.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：单机/多机启动、worker 参数和 environment variables。
  - 来源支持的事实：`torchrun` 在每节点按 `--nproc-per-node` 启动 worker；公开环境变量包括 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`、`LOCAL_WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`。PyTorch 2.0 起脚本接收 `--local-rank` 参数形式；文档说明读取环境变量 `LOCAL_RANK` 的方式。弹性 worker group 可能重新启动并重新分配 rank/world size。
  - 证据等级：source-backed（PyTorch 2.15 launch 行为与公开变量）。
  - 适用边界：launcher 参数不验证实际网络可达性、设备映射正确性、共享存储可见性或训练脚本的数据切分。
- 来源：[DistributedSampler](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.distributed.DistributedSampler)，PyTorch 2.15 官方文档。
  - 实际阅读位置：sampler 参数、`seed`、`drop_last` 和示例中的 `set_epoch()`。
  - 来源支持的事实：各进程 sampler 的 seed 应相同；`drop_last=False` 为均匀分 rank 而补入额外索引，`drop_last=True` 则截掉尾部使长度可整除；在每个 epoch 创建 DataLoader iterator 前调用 `set_epoch(epoch)`，否则 shuffle 顺序会重复。
  - 证据等级：source-backed（固定版本 API 契约）。
  - 适用边界：假设 Dataset 为长度固定、索引稳定的 map-style dataset；流式/IterableDataset 需要用户按 rank/worker 实现分片。
- 来源：[DDP `join()`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join) 与 [generic Join context](https://docs.pytorch.org/docs/2.15/distributed.algorithms.join.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：uneven inputs、shadow collectives、`divide_by_initial_world_size` 和额外 collective 警告。
  - 来源支持的事实：`join()` 可让已耗尽输入的 DDP rank shadow DDP forward/backward 通信，以便仍在运行的 rank 不因典型 DDP collectives 失配而 hang；它对 `SyncBatchNorm` 或其他应用 collective 无法自动 shadow，文档建议此类情况启用 early termination；梯度分母可按初始或当前有效 world size 处理，样本权重含义不同。
  - 证据等级：source-backed（公开 DDP join API 语义）。
  - 适用边界：join 不提供任意应用状态恢复，也不能修复数据污染、rank 不一致的 optimizer step 或自定义 collective 协议。
- 来源：[分布式 debug 与 monitored barrier](https://docs.pytorch.org/docs/2.15/distributed.html#debugging-torch-distributed-applications)，PyTorch 2.15 官方文档。
  - 实际阅读位置：`TORCH_DISTRIBUTED_DEBUG`、`TORCH_CPP_LOG_LEVEL`、`monitored_barrier` 和 process-group shutdown。
  - 来源支持的事实：`TORCH_DISTRIBUTED_DEBUG` 支持 `OFF`、`INFO`、`DETAIL`；与 `TORCH_CPP_LOG_LEVEL=INFO` 配合可启用额外日志，`DETAIL` 会报告选定迭代时间统计并可能降低性能。`monitored_barrier` 可在超时时暴露未确认 rank，使用 host-side Gloo group。所有 trainer process 应在通信结束时调用 `destroy_process_group()`；特别是多 process group 的 NCCL 应在各 rank 一致清理，以避免析构次序不一致导致退出 hang。
  - 证据等级：source-backed（固定版本 debug/shutdown 指南）。
  - 适用边界：调试日志和 monitored barrier 不能替代 rank/step 应用日志或目标网络的独立诊断；verbose profiling 结果不是无扰动性能基线。
- 来源：[FSDP wrapper](https://docs.pytorch.org/docs/2.15/fsdp.html) 与 [`fully_shard`](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：FULL_SHARD 与 FSDP2 full-shard 的参数、梯度、optimizer state shard 及 all-gather / reduce-scatter 生命周期。
  - 来源支持的事实：full-shard 路径将参数、梯度和 optimizer state 分片；计算模块前需要 all-gather 参数，依策略在 forward 后重分片，backward 后以 reduce-scatter 同步分片梯度。`fully_shard` 按 module group 执行 collective，分组粒度影响每次聚合 tensor 和通信/计算 overlap 机会。
  - 证据等级：source-backed（固定版本 API 与公开数据流）。
  - 适用边界：API 细节随 FSDP1/FSDP2、strategy、wrapper 边界和 PyTorch 版本不同；文档算法步骤不构成目标显存峰值或速度保证。

## Debug、Profiler、Benchmark 与编译排障（PyTorch 2.15）

补充检索与阅读日期：2026-11-11。

- 来源：[Autograd anomaly detection](https://docs.pytorch.org/docs/2.15/autograd.html#anomaly-detection)，[Profiler API](https://docs.pytorch.org/docs/2.15/profiler.html)，[Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 和 [CUDA semantics](https://docs.pytorch.org/docs/2.15/notes/cuda.html)，PyTorch 2.15 官方文档。
  - 实际阅读位置：anomaly context 的前向/反向作用范围与 `check_nan`；profiler schedule 的 wait/warmup/active 周期、`prof.step()`、`record_shapes` 与 shape/stack 开销；CUDA 异步队列和计时同步说明。
  - 来源支持的事实：anomaly detection 应在前向与反向范围内启用，才能让 backward 异常关联创建节点的前向 traceback；`check_nan=True` 会在 backward 产生 NaN 时抛错。Profiler schedule 按 `wait`、`warmup`、`active` 步数运行，并依赖每轮 `prof.step()` 推进；shape tracing 会增加开销、暂时保留 tensor 引用并可能影响优化或触发额外拷贝。CUDA 默认异步执行，精确 host-side timing 需同步或使用 CUDA events。
  - 证据等级：source-backed（固定版本 API 语义和官方 recipe）。
  - 适用边界：anomaly/profiler 都会改变运行成本；CPU trace 不能支持 CUDA kernel 性能结论，schedule 配置也不能证明采样区间代表整体 workload。
- 来源：[Benchmark Utils](https://docs.pytorch.org/docs/2.15/benchmark_utils.html)，PyTorch 2.15 官方 API 文档。
  - 实际阅读位置：`torch.utils.benchmark.Timer` 的 timer 实现说明、`blocked_autorange` 和 `Measurement`。
  - 来源支持的事实：有 accelerator 时，`Timer` 默认在计时前同步 accelerator；`blocked_autorange` 会先预热，再调节 block size，使 timer 调用开销相对被测代码足够小，并重复采样返回统计对象。
  - 证据等级：source-backed（版本化 Benchmark API 说明）。
  - 适用边界：benchmark 结果只代表指定语句、线程数、设备、输入和环境；自动同步与预热不替代合理的代表性 shape、稳态窗口和完整训练 step 测量。
- 来源：[torch.compile Troubleshooting](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html) 与 [`fullgraph=True` graph-break guidance](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)，PyTorch 2.15 官方开发者文档。
  - 实际阅读位置：graph-break 语义/影响差异、`TORCH_LOGS` 的 `graph_breaks`/`recompiles`、guard 与 shape 变化、`fullgraph=True`、wrapper 编译入口。
  - 来源支持的事实：默认编译路径可结束当前图，在 eager 中执行不支持代码后继续追踪；graph break 导致优化机会损失，但影响程度因发生位置而异。`fullgraph=True` 要求单个 FX graph，否则在断图时出错，可用于诊断而非性能承诺。`TORCH_LOGS` 可报告断点位置/原因和 guard 失效重编译；输入条件变化可能使函数重新编译。PyTorch 文档建议按问题将 `torch.compile` 放在较高层级或 module 内部，并对分布式 wrapper 评估编译内部 module。
  - 证据等级：source-backed（版本化编译器排障指南）。
  - 适用边界：graph-break 日志不证明断点一定是主要瓶颈；日志、`fullgraph=True` 和特定 dynamic 设置不保证兼容性、数值等价或目标模型加速。实际结果仍需固定输入与环境独立测量。
