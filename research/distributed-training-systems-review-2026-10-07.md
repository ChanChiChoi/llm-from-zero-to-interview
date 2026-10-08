# 大规模训练系统进阶精修记录（2026-10-07）

## 范围与结论

- `chapter`：`book-02-advanced-100/chapters/04-大规模训练系统进阶.md`。
- `scope`：修订前 1,184 行、覆盖第 41—50 讲；本轮从头复读至末尾，修订后 1,280 行。第二册目录和讲次编号保持一致。
- `evidence_level`：`paper`、`official`、`local-demo`、`inference`。
- `status`：完成；结论范围限于引用资料、公式推导和记录的本地 CPU 教学示例。

## 主要修订

- 细化 DDP 的梯度语义：区分等量样本的 rank mean 与变长序列的 token mean；推导有效 token 加权的本地 loss 缩放，并加入 100/10 token 的数值例子。
- 将原只支持 CUDA/NCCL 的 DDP 代码骨架改成可用 `torchrun` 启动的双进程 CPU/Gloo 示例，直接观察本地梯度平均和一致参数更新。
- 为 ZeRO/FSDP 状态显存估算补足输入类型和有限值检查，并在示例后解释 7B 参数、8 ranks 的估算结果与遗漏项。
- 修复 Tensor Parallel 矩阵公式中的 LaTeX 拼写；补全 blockwise softmax 的局部统计量定义，避免把不同 block 的局部 softmax 结果简单相加。
- 将 activation checkpoint demo 改为比较 checkpoint 与普通路径的输出、输入梯度和参数梯度；明确它没有测量显存或速度。
- 为吞吐/MFU 例子补充有效 token 的统计口径、输入边界检查和估算范围；为事故分类器补充字段契约及非有限值处理。
- 将 checkpoint 提交描述限定为文件系统原子 rename 或对象存储 manifest/pointer 等适用语义；不把 rename 当作所有存储的一般保证。
- 在 `PAPERS.md` 增加并行训练核心论文入口，在 `EXERCISES.md` 增加 DDP 变长 token 加权推导题。术语表已有 DDP、ZeRO、FSDP、TP、PP、SP、CP、MFU 等条目，无需重复新增。

## 资料与证据范围

- 论文入口覆盖 ZeRO、ZeRO-Offload、Megatron-LM 张量并行与集群训练、GPipe、Sequence Parallel、FlashAttention、Ring Attention、梯度检查点和 GPU 测量背景。
- 官方文档入口覆盖 PyTorch DDP、Distributed Overview、FSDP、activation checkpoint、Profiler、Distributed Checkpoint 和 NVIDIA NCCL。
- 本轮检查的 21 个唯一外部 URL 均返回 HTTP 200。PyTorch 文档链接使用 `docs.pytorch.org` 当前入口；外部论文链接状态确认不等于逐篇全文复核。论文结论仅用于解释文中相应机制；吞吐、显存峰值、通信比例和具体硬件配置仍须在目标系统实测。
- FSDP、Profiler 和 DDP 的章节描述按官方文档解释接口/一般机制，不声称目标框架部署结果。故障分类、检查点验收和恢复流程是工程建议，阈值示例不代表生产告警标准。

## 本地教学代码

- 10 个 Python 代码块全部通过语法检查和正常路径执行；运行环境为 Python 3.12.9、PyTorch 2.12.0+cpu，使用仓库 `.tmp/llm-book-cpu-check/`，`.tmp/` 已加入 `.gitignore`。PyTorch 报告 `torch.cuda.is_available() == False`，未安装 CUDA 库。
- DDP 例子通过 `torchrun --standalone --nproc_per_node=2 .tmp/ddp_cpu_demo.py` 两进程运行，输出 `world_size=2 local_batch=2 global_batch=4 weight=1.5000`。这是 CPU/Gloo 语义演示，不是 GPU/NCCL 性能测试。
- 其余 9 个示例逐块执行通过：分片显存估算、TP 矩阵切分、PP 时间线、online attention、rank 映射、activation checkpoint 等价性、通信重叠、吞吐/MFU 估算和事故信号分类均与正文描述一致。
- 另做 7 项边界检查，覆盖 bool/非有限显存参数、非法吞吐参数、非有限 loss、非有限显存上限和零梯度范数处理。

## 格式、链接与限制

- 43 个 `math` 围栏、114 个代码/文本/数学围栏配对；公式花括号和 `begin/end` 环境检查通过。两张 Markdown 表格列数一致；未运行外部 MathJax 渲染器。
- 章节没有相对 Markdown 链接；第二册目录中第 41—50 讲范围与正文一致；21 个外链目标均返回 HTTP 200。
- `git diff --check` 通过。尝试运行 `python3 scripts/build_pdfs.py --dry-run`，但环境缺少 Pandoc，未构建 PDF。
- 未测试 CUDA/NCCL、GPU 吞吐或 MFU、FSDP/ZeRO 实际分片、TP/PP/CP 集群运行、多节点训练、真实模型 checkpoint 或生产故障恢复。CPU 输出只说明教学代码在当前环境中的局部行为。
