# 第八章精修审计：Debug 与 Profiling

审计日期：2026-11-11

## 范围与章级扫描（scope / chapter_scan）

- 完成《第八章：Debug 与 Profiling》§§0–8.23 全文审阅与精修；本章仍服务于“把训练故障和性能疑问变成可复现、可区分、可测量的工程问题”，未改变第十四册十章骨架。
- 2026-11-11 定向查阅 PyTorch 2.15 官方 Autograd anomaly、Profiler API/recipe、Benchmark Utils、CUDA 异步执行与显存管理、DataLoader，以及 `torch.compile` troubleshooting / `fullgraph=True` 指南。研究证据及其版本和边界见[PyTorch API 研究记录](../../research/pytorch-training-api-boundaries-2026.md)“Debug、Profiler、Benchmark 与编译排障”小节。
- 本次核验问题：如何沿数据/shape/label、finite、loss、grad、hook、OOM 顺序找首个异常；如何区别 padded 与有效目标吞吐；Profiler schedule 和观测开销怎样影响 trace；如何排除 CUDA 异步、编译冷启动、graph break 和 guard 失效造成的误判。
- 候选去向：将 `torch.compile` 的 eager 对照、代表 shape 预热、graph-break/recompile 排查和稳态基准并入 §8.14.5，不另设编译器章节；以 schedule 示例和独立计时说明 profiler 的采样与观测边界；保留 `Nsight`、NCCL 深度 profiling、生产 trace 聚合和 Inductor 内部优化为范围外入口。未发现改变本册边界或十章次序的新主题。

## 篇幅与解释链（length_status）

计数仅统计连续中文解释文字，剔除标题、代码、数学围栏、表格、引用、行内代码和链接；以下三条按各自完整论证链计为核心教学单元，门槛均为 1,800 汉字。

| 教学链 | 范围 | 实计汉字 | 类型 / 门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| 诊断信号与首个异常定位 | §§8.1–8.9 | 2,957 | 核心 / 1,800 | 达标；从张量契约和标签语义递进到 finite、anomaly、梯度与 hook 扰动。 |
| 显存、数据与性能测量 | §§8.10–8.14 | 2,797 | 核心 / 1,800 | 达标；涵盖 OOM 账本、DataLoader 稳态窗口、CUDA 同步、Profiler schedule 与 compile 核验。 |
| 训练结果排查与最小复现 | §§8.15–8.19 | 2,343 | 核心 / 1,800 | 达标；将排查次序、rank 日志、checklist 与运行 demo 收束为可复现流程。 |

## 数值例子（example_status）

- §8.3：`B=1,T=4,V=6`，labels `[2,5,3,-100]`；shift 后 logits `[1,3,6]`、labels `[1,3]`，flatten 成 `[3,6]` 和 `[3]`；只有两个目标有效。目标概率 `0.5`、`0.25` 对应 `0.6931`、`1.3863`，忽略项不进分母，平均 CE 为 `1.0397`。
- §8.10：FP32 logits `[4,2048,32000]` 含 `262,144,000` 个元素；乘 4 bytes 得 `1,048,576,000` bytes，约 `0.977 GiB`。保留 10 份输出约 `9.77 GiB`，尚未包括 autograd 中间量和 allocator 开销。
- §8.11–8.12：`B=8,T=1024,t=0.20s` 时 padded throughput 为 `40,960 token/s`；若 shift/mask 后只有 75% 有效，则 `6,144/0.20=30,720 valid target token/s`。
- §8.8、8.13、8.14.5：`[3,4]` 梯度的 L2 norm 为 5；`wait=1,warmup=1,active=2` 至少走四个 step（skip 1、warmup 1、record 2）；假设的 eager `1.20ms` / compiled `0.90ms` 得 `1.33×` 与 25% latency reduction，明确标为假设值。
- §8.15：`B=2,T=8` 共 14 个 shift 后候选目标，若忽略 3 个应有 11 个有效标签；语义断言可发现 shape 合法但 mask 错位。
- `EXERCISES.md` 同步补充 batch A 两个目标、mean loss 2 与 batch B 六个目标、mean loss 1：batch mean 为 1.5，token mean 为 1.25，说明聚合口径不同会改变结论。

## 证据、公式与链接（evidence / formula_status / link_status）

- 正文关键 API 事实采用固定 PyTorch 2.15 官方资料；重点收窄了 anomaly context 必须覆盖前向和反向、`check_nan` 的边界、Profiler schedule 需逐步 `prof.step()`、shape tracing 会引入开销/引用影响、benchmark 预热和 accelerator 同步、compile 默认 graph-break 语义、`fullgraph=True` 的诊断用途及 `TORCH_LOGS` 重编译日志范围。正文结论不把这些 API 描述外推为目标任务的速度或显存保证。
- 本章 9 个 display-math 围栏均配对，变量及单位在邻近正文给出；已对围栏和公式分组作静态检查。全章 33 组三反引号围栏（66 个标记）配对；其中 28 个 Python 和 1 个 Bash 代码块均有紧邻状态，另有 4 个 text 示例块。没有声称 MathJax 页面渲染或外部链接完整 crawl 已通过。
- 9 个章末外链指向 PyTorch 官方资料；唯一章内相对链接指向第七章分布式主讲，目标存在。没有 `docs/stable` 引用。
- 视觉机会已审视：本章保留“假设—最小实验—观测—下一步”对照表与有序排查路径；各故障类别使用独立输入、症状和观察量，未再叠加一张把异质分支压成单一路径的图。现有表格和分步案例承担读者需要比较/追踪的关系。

## 代码状态（code_status）

正文代码状态按稳定位置记录。Python 片段共 28 个，均通过 `ast.parse`；Bash 启动片段 1 个静态审阅。静态片段的目标 API 版本为 PyTorch 2.15，没有标作运行验证。

| 稳定位置 / 代码块 | 状态与验证记录 |
| --- | --- |
| §§8.2–8.6 代码块 01–13 | 静态审阅，未运行；覆盖 metadata、shift loss、shape/label 断言、dtype/device 与 finite 诊断。 |
| §8.7 代码块 14–15 | 静态审阅，未运行；anomaly 示例依赖本节的 `model`、`batch` 和 `compute_loss` 约定。 |
| §§8.8–8.9 代码块 16–18 | 静态审阅，未运行；梯度和 hook 片段按示例模型结构取用。 |
| §8.10 代码块 19 | 静态审阅，未运行；显存查询依赖 CUDA；设备未验证。 |
| §8.10 代码块 20 | 静态审阅，未运行；需要 CUDA 设备以及本节 `fixed_batch`、`train_step`。 |
| §8.11 代码块 21 | 静态审阅，未运行；DataLoader 预热/计时片段。 |
| §8.12 代码块 22 | 静态审阅，未运行；计时片段依赖本节训练函数和固定 batch。 |
| §8.13 代码块 23–24 | 静态审阅，未运行；Profiler 活动选择与 schedule 片段。 |
| §8.14.5 代码块 25 | **已验证**：2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU；从正文代码块提取为临时脚本，以 `.tmp/llm-book-cpu-check/bin/python` 执行，设置绝对 `TORCHINDUCTOR_CACHE_DIR=/data/zzc/llm-from-zero-to-interview/.tmp/torchinductor-cache`。eager/compiled `assert_close` 通过。当地这次玩具前向中位数约为 eager 0.251 ms、compiled 0.373 ms；这里只记录该 CPU 观察，代码没有宣称一般加速。无 CUDA/PyTorch 2.15 runtime 结果。 |
| §8.14.5 Bash 代码块 26 | 静态审阅，未运行；`TORCH_LOGS="graph_breaks,recompiles" python train.py` 需由读者在实际训练入口执行。 |
| §8.16 代码块 27、§8.18 代码块 28 | 静态审阅，未运行；日志与最小复现片段依赖章节内声明的模型输出结构。 |
| §8.19 代码块 29 | **已验证**：2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU；保存为 `debug_audit.py` 后运行 `python debug_audit.py`，默认路径检查通过；设置 `RUN_TORCH_PROFILER=1` 再运行，Profiler 路径检查通过，能得到正事件数和 `aten::addmm`。默认参考 loss `3.1516`，valid label count 为 5，有限值、梯度参数覆盖、hook 移除和正计时检查通过。未验证 CUDA、PyTorch 2.15 runtime 或多 rank。 |

本轮代码块可执行语句没有在状态标记后再修改；状态位置、依赖和验证边界已与正文一致。编译微基准只覆盖固定小型前向和 CPU，不覆盖训练 step、DataLoader、CUDA 或生产模型；本地结果反而显示此玩具负载并未因编译变快。

## 配套同步与遗留项（sync_status / open_risks）

- 已同步第十四册[简介](../../book-14-pytorch-deep-learning-engineering/简介.md)和[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)中的第八章摘要/产出；加入指向第七章的分布式排错主讲入口。
- 已更新 `INTERVIEW_BANK.md` Q62–64 的测量/Profiler/demo 边界并加入 Q64A `torch.compile`；在 `EXERCISES.md` 新增本章 11 题；在 `GLOSSARY_EN_ZH.md` 加入 Graph Break、Recompilation、`torch.compile`、`fullgraph=True`、Profiler Schedule 和 Benchmark Timer；PyTorch 2.15 新发现写入既有研究记录。
- 限制：没有 PyTorch 2.15 本地 runtime、CUDA/GPU、NCCL/多 rank 或真实训练 workload 验证；玩具 CPU compile benchmark 不能支撑整体训练性能判断。后续若第九章引入编译或 AMP+compile 组合，必须按该章实际代码与配置重新核验。
- 本章改动未更新 PDF 和 `plan.md`。

## 工作交接（work_handoff）

第十四册第一至第八章均已完成章级扫描、精修与逐章审计。下一步按目录进入[第九章《实现 Transformer 组件》](../../book-14-pytorch-deep-learning-engineering/chapters/09-实现transformer组件.md)，重点核验 attention / mask / cache 的张量边界、训练目标与生成状态复用，并按内容需要读取相应规范模块后开始正文工作。
