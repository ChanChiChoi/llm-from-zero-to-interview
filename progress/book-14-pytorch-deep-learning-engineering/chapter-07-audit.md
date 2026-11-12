# 第十四册第七章审计：分布式训练入门

审阅日期：2026-11-11

## Scope 与章级扫描

- 完整复读《分布式训练入门》§§0–7.23，并按 DDP 梯度归约与 loss 分母、sampler 尾部处理、torchrun 多节点 rendezvous、`no_sync()` 累积、DDP/FSDP 状态边界和集体通信排障定向核验。章级来源扫描于 2026-11-09 完成，主要使用 PyTorch 2.15 官方 distributed/DDP/torchrun/data/FSDP 文档；结论、来源版本及证据边界见[PyTorch API 边界研究记录](../../research/pytorch-training-api-boundaries-2026.md)。
- 本章保留既有骨架，将 rank/process group、数据并行与梯度尺度、启动与采样、通信与累积、状态分片与恢复、故障诊断串成六条教学链。没有新证据要求改变第十四册范围；完整集群调度、并行策略和生产 FSDP 系统设计仍交第五册。
- 更新后的范围覆盖 2.15 文档所述 DDP 默认初始化同步、梯度 reduction、buffer 行为、`no_sync()` 上下文范围、sampler 补齐/截断、Join 和 debug API；正文分别限定固定拓扑、等步数/等分母假设和目标硬件测量边界。研究发现只写入 research；审阅决定与章节状态只记于 progress。

## Length status

按连续解释性正文计数，统计汉字；排除标题、围栏内容、表格、引用状态、行内代码和链接。连续编号的小节按六条上位教学链归并。

| 教学链 | 覆盖范围 | 实计汉字 | 类型与门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| 分布式训练动机、rank 与 process group | §§7.1–7.3 | 2,076 | 核心，≥1,800 | 通过 |
| 数据并行、all-reduce 与 DDP 梯度语义 | §§7.4–7.6 | 1,907 | 核心，≥1,800 | 通过 |
| torchrun、分布式 sampler 与 global batch | §§7.7–7.9 | 1,990 | 核心，≥1,800 | 通过 |
| DataParallel 对照、bucket 通信与 `no_sync()` | §§7.10–7.12 | 1,846 | 核心，≥1,800 | 通过 |
| FSDP 状态分片与分布式 checkpoint | §§7.13–7.14 | 1,960 | 核心，≥1,800 | 通过 |
| 常见故障、排查、最小骨架与机制 demo | §§7.15–7.19 | 2,908 | 核心，≥1,800 | 通过 |

字数由逐行过滤标题、代码/数学围栏、表格、引用状态、行内代码与链接后的计数脚本统计；字符达标后仍人工检查了各链是否围绕同一问题解释机制、中间步骤和限制。练习、资料目录和总结没有作为独立讲解链计入。

## Evidence 与 example status

- 主要事实使用固定版本 PyTorch 2.15 官方文档。结论没有从官方 API 外推到 NCCL 带宽、生产速度、集群可靠性或 FSDP 实测峰值；相应收益必须在目标环境测量。`TORCH_DISTRIBUTED_DEBUG=DETAIL`、`monitored_barrier`、`join()`、FSDP wrapper/`fully_shard` 的版本与行为均限定在引用页面范围内。
- §7.2：两台同构节点、每节点 4 worker 时，`WORLD_SIZE=8`；节点 0 的 local rank 0–3 对应 global rank 0–3，节点 1 对应 4–7。明确 rank 计算假设每节点进程数相同。
- §7.4.1：标量模型 `θ=2`、`x=1`、损失 `0.5(θx-y)^2`；两 rank 的样本梯度分别为 `[2,0]` 与 `[1,-1]`，本地均值为 1 和 0，DDP 平均梯度为 0.5，学习率 0.1 后 `θ'=1.95`。另以 4/16 个有效 token、rank mean 2/1 对照 token mean 1.2，推导 DDP 默认平均梯度下的全局分母缩放。
- §7.8–7.9：`N=10, world_size=3` 的 sampler 补齐/截尾索引账本区分 sampler 与 DataLoader 两层 `drop_last`；完整与尾部累积窗分别按 24/12 个名义样本计算，语言模型另按真实有效 token 统计。
- §7.12：16M 个 FP32 梯度约 64 MB（约 61 MiB）；四个 micro-step 每步同步与前三步 `no_sync()`、窗口末同步一次作次数比较。性能例的 120 ms/50 ms 与 60% scaling efficiency 明确为构造的教学假设，不是 PyTorch 测量结果。
- §7.13：假设 120M 参数、BF16 参数/梯度、FP32 AdamW 两组 moment，状态账本为约 1,440 MB（1.34 GiB）；四路 full shard 理想常驻约 0.335 GiB/rank。排除了 activation、临时 all-gather、通信 buffer 和 allocator/context 开销，不把理想状态量当作峰值承诺。
- §7.18：纯 Python demo 包含确定性索引切分、补齐和丢弃、rank mean 与 token mean、梯度平均、`no_sync` 次数和 rank 0 写出审计；样例输出检查覆盖空交集/完整覆盖、epoch shuffle 变化、小数据集补齐、梯度同步、collective 次序错配检测及 token 权重差异。

## Code status

全章 18 个 Python 与 3 个 Bash 代码围栏均有相邻状态，共 21 个。静态块均未运行；第 18 个 Python 块是纯 Python demo。以下以 Python/Bash 各自的顺序编号，便于回到正文定位：

| 代码块 | 正文位置 | 状态 | 环境/边界 |
| --- | --- | --- | --- |
| Python 01 | §7.2.3 local rank | 静态审阅，未运行 | 需要分布式环境变量和目标设备 |
| Python 02–03 | §7.3 process group / collective | 静态审阅，未运行 | PyTorch 2.15 API 片段 |
| Python 04–06 | §7.6 DDP 初始化、训练循环、rank 0 保存 | 静态审阅，未运行 | 需要模型、数据和多进程环境 |
| Python 07 | §7.7 读取 launcher 环境 | 静态审阅，未运行 | 依赖 `torchrun` 注入环境变量 |
| Python 08–10 | §7.8 sampler、epoch 和 DataLoader | 静态审阅，未运行 | 依赖 map-style dataset 和 PyTorch 2.15 |
| Python 11 | §7.12 `no_sync()` 累积窗口 | 静态审阅，未运行 | 依赖长度一致的 rank loader 和 DDP |
| Python 12 | §7.14 DDP checkpoint 保存 | 静态审阅，未运行 | 依赖模型、optimizer 和进程组 |
| Python 13 | §7.15.2 unused parameter 检查 | 静态审阅，未运行 | 依赖 PyTorch 模型与 Autograd |
| Python 14–16 | §7.16 rank/collective 排障辅助片段 | 静态审阅，未运行 | API/日志片段；真实 collective 未测 |
| Python 17 | §7.17 最小 DDP 训练骨架 | 静态审阅，未运行 | 依赖自定义模型/数据及 CUDA 多进程 |
| Python 18 | §7.18 纯 Python 分布式机制审计 demo | 已验证 | Python 3.12.9、CPU、标准库；不代表 PyTorch/NCCL/DDP |
| Bash 01 | §7.7 单机 `torchrun` 启动 | 静态审阅，未运行 | 需在实际设备环境启动 |
| Bash 02 | §7.16 distributed debug 启动 | 静态审阅，未运行 | 需在实际设备和进程组环境启动 |
| Bash 03 | §7.17 最小骨架启动 | 静态审阅，未运行 | 需在实际 CUDA 多进程环境启动 |

全部 18 个 Python 围栏在本轮以 `python3` 执行 `ast.parse`，语法检查通过。

§7.18 demo 于 2026-11-09 在 Python 3.12.9 CPU 环境用标准库执行。实际输出与正文预期一致：sampler 全覆盖且无交集、epoch 顺序改变、10 样本/3 rank 的 pad/drop 结果正确；rank mean loss 为 1.5、token mean 为 1.2，all-reduce 平均梯度为 0.2；同步副本参数一致，未同步 toy 参数出现 0.25 漂移；有 `no_sync` 时通信调用从 6 次降到 2 次；事件审计能定位 rank 2 的 collective 次序错配；只有 rank 0 写文件。10 项 demo 检查均为 True。此 local demo 不验证 PyTorch、CUDA、NCCL、多进程或真实 DDP。

## Formula、link、visual 与 sync status

- 10 个 `~~~math` 围栏均配对，花括号计数平衡；80 个三反引号标记构成 40 个闭合代码围栏。章节共 18 个 Markdown 链接，均为 PyTorch 2.15 官方文档；没有相对路径链接，源码中未残留 `docs/stable`。未声称外链 HTTP crawl 或 GitHub MathJax 页面渲染通过。
- 视觉表达检查：§7.4 数据流表按输入/forward、backward、optimizer step、日志验证分开 rank 本地状态与跨 rank 通信，读者可用它定位同步阶段；sampler 的补齐/截尾示例和 DDP/FSDP 预算以可复算步骤展开。当前没有必须再用 Mermaid 才能理解的流程；未添加装饰图。
- 第七章摘要已同步第十四册[简介](../../book-14-pytorch-deep-learning-engineering/简介.md)和[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)。`INTERVIEW_BANK.md`、`EXERCISES.md` 和 `GLOSSARY_EN_ZH.md` 已同步 DDP/rank、sampler、有效 token 分母、`no_sync()`、FSDP 状态及排障要点。来源事实在现有研究记录中；书籍修改与审计结论只记录于本文件和 progress handoff。
- 修改范围内的 `git diff --check` 通过。未更新 PDF 或 `plan.md`。

## Open risks 与 handoff

- 未在 PyTorch 2.15 runtime、CUDA/NCCL、多 rank 或真实多节点网络中运行代码；所有 GPU launcher/API 片段均明确标为静态审阅，未声称端到端分布式验证。
- 没有测量目标 GPU/网络上的 DDP scaling、bucket overlap、`no_sync()` 速度收益、FSDP 峰值显存或 checkpoint 跨节点可见性。分片预算只是参数状态的理想近似。
- rank-local RNG、DataLoader worker 状态、数据游标和弹性重启后的精确样本续接依赖实际数据管线与恢复协议；章节说明所需状态，但没有实现生产 checkpoint 系统。
- 第七章摘要、题库、练习、术语和来源记录均已同步；本章遗留边界如上，不阻止入门级正文验收。
- **下一步：** 按第十四册目录进入第八章《Debug 与 Profiling》，重点扫描 `torch.compile`/graph break、Profiler 观测扰动和稳态基准方法；若出现改变整册边界的证据再回到册级审视。
