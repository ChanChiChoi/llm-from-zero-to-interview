# 第十四册第六章审计：混合精度与显存优化

审阅日期：2026-11-09

## Scope 与章级扫描

- 完整复读《混合精度与显存优化》§§0–6.20，围绕 AMP dtype/scaler 一致性、FP16/BF16 范围、累积窗口、checkpoint 重算、optimizer/activation 显存账本、CUDA allocator 统计和分阶段 OOM 诊断定向核验；保留十章骨架。
- 主要修正原 AMP 模板中 `autocast` 固定 FP16、scaler 和分支开关各自漂移的问题；补齐 `window_size`、`is_window_end` 的片段边界；更正有效 batch 重复乘 micro-batch 样本数的公式；厘清 BF16 autocast 不必启用 GradScaler、AMP skipped update 与真实 `global_step`/scheduler 的对应关系。
- 章级候选：FlashAttention IO-aware tiling 与 full attention 显存关系并入 §6.1/6.9 作为估算边界，不新增 kernel 教学单元；FP8、ZeRO/FSDP、TP/SP、CUDA allocator 内部和 Nsight 仍按章节契约留给分布式/性能专题。此次扫描未发现改变书册边界或新增独立主题的证据。
- 证据来源为 PyTorch 2.15 AMP、AMP examples、checkpoint、CUDA memory/API 官方文档及 FlashAttention 原论文；来源事实、版本与证据边界见[PyTorch API 边界研究记录](../../research/pytorch-training-api-boundaries-2026.md)。

## Length status

按解释性中文计数，排除标题、代码、公式、表格、代码状态、行内代码和链接；连续论证归为两条教学链。

| 教学链 | 覆盖范围 | 实计汉字 | 类型与门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| dtype、GradScaler、激活重算与累积窗口 | §§6.1–6.8 | 3,739 | 核心，≥1,800 | 通过 |
| 显存账本、测量、OOM 决策与精度选择 | §§6.9–6.16 | 2,300 | 核心，≥1,800 | 通过 |

两条教学链通过篇幅下限后，仍以可复算的范围例、窗口例和 OOM 阶段表承载机制；没有将代码、表格或引用计入解释性文字。

## Evidence 与 example status

- PyTorch 2.15 AMP 文档核验 autocast 算子策略、前向/loss 包裹边界、模型不需手动整体转 half/bfloat16、FP16 GradScaler skip/nonfinite 行为、scale 可低于 1 及部分 BF16 预训练模型超出 FP16 范围。AMP examples 核验裁剪前 unscale、累积时保持 scale 不变、窗口结束后一次 unscale/update。
- PyTorch 2.15 checkpoint 文档核验 `use_reentrant=False`、CPU+一种推断的非 CPU device RNG 范围、多设备/函数内新设备迁移边界及 `torch.compile` 下忽略 `preserve_rng_state=False`。CUDA 文档核验 allocated/reserved/peak 与 `empty_cache()` 语义；FlashAttention 原论文支持 IO-aware tiling、减少 HBM 读写的陈述。结论不推广为所有设备或实现的速度保证。
- §6.1.2：假设 `P=10^9`、BF16 参数/梯度 2 bytes、AdamW 两个 FP32 moment，各为约 1.86、1.86、7.45 GiB，三项合计约 11.17 GiB；不含 activation、workspace、CUDA context、碎片或可能的 master copy。
- §6.5.1：真实梯度范数 10、scale 1,024，scaled norm 为 10,240。直接在 scaled 梯度上按 1 裁剪再 unscale 得约 0.00098；正确先 unscale 再裁剪应得范数 1，故错误顺序缩小 1,024 倍。
- §6.8：`B_micro=2`、`K=4`、`R=3` 时 nominal effective batch 为 `2×4×3=24` 样本，假设 rank 样本数一致；尾窗与 token-weighted loss 不按此简式替代第五章的分母约定。
- §6.9：构造 `B=2,T=2048,d=4096,L=32,H=32,b=2 bytes`。每层一个 hidden state 汇总约 1 GiB；每层一个朴素 score 矩阵约 0.5 GiB，32 层均物化约 16 GiB。将 `T` 减半后 hidden proxy 减半至 0.5 GiB，score proxy 降至每层 0.125 GiB、32 层合计 4 GiB。数值只说明维度关系，不是实际峰值。
- §6.15 CPU demo：100 万 BF16 参数、梯度、FP32 AdamW 状态和 200 万 BF16 activation 元素的估算输出分别为 1.907、1.907、7.629、3.815 MiB，总计 15.259 MiB；CPU autocast 输出 BF16，梯度裁剪前范数 0.746，普通/checkpoint block 调用次数为 1/2。该 demo 不是 CUDA 显存或性能测量。

## Code status

全章 14 个 Python fence 均有相邻状态；除完整 CPU demo 外均为非独立片段并标记静态审阅。

| Python 块 | 位置 | 状态 |
| --- | --- | --- |
| 01–02 | §6.4 autocast 与嵌套上下文 | 静态审阅，未运行 |
| 03 | §6.5 FP16 GradScaler step | 静态审阅，未运行 |
| 04–05 | §6.6 AMP 模板与累积窗口片段 | 静态审阅，未运行 |
| 06 | §6.7 checkpoint 包装 | 静态审阅，未运行 |
| 07 | §6.8 等大小 micro-batch 累积 | 静态审阅，未运行 |
| 08–09 | §6.12 allocated/reserved 与 peak 片段 | 静态审阅，未运行 |
| 10–11 | §6.13 带图 loss 与标量记录示例 | 静态审阅，未运行 |
| 12 | §6.13 分段 peak snapshot | 静态审阅，未运行 |
| 13 | §6.14 checkpointed model AMP step | 静态审阅，未运行 |
| 14 | §6.15 CPU BF16/autograd/checkpoint demo | 已验证 |

14 个 Python fence 均通过 `ast.parse`。块 14 于 Python 3.12.9、PyTorch 2.12.0+cpu 的 CPU 环境执行，输出与正文相符。未执行 CUDA 专用块，不声称 PyTorch 2.15 runtime、GPU memory、吞吐、真实 checkpoint 重算节省或多 rank 结果已验证。

## Formula、link、visual 与 sync status

- 10 个 `~~~math` 围栏配对，逐块花括号平衡；30 个三反引号标记配对，14 个 Python fence 的状态数相同。
- 23 个 Markdown 链接中包含 PyTorch 2.15 官方 AMP/checkpoint/CUDA API、FlashAttention 原论文及第五章本地链接；源码级核验本地目标存在。未宣称全外链 HTTP crawl 或 GitHub MathJax 页面渲染通过。
- 视觉机会已审视：§6.9 用维度预算表对照 hidden/attention 随 `T` 的线性/平方变化；§6.13 用阶段决策表连接 OOM 信号、假设和第一步观测。当前无需新增 Mermaid 图。
- 已同步第十四册[简介](../../book-14-pytorch-deep-learning-engineering/简介.md)和[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)的第六章摘要；`INTERVIEW_BANK.md` Q41–44、47–48 校准范围、更新和诊断表达，`EXERCISES.md` 133–140 补入显存复算与阶段排障，术语表补充 AMP/scaler/checkpoint/memory metric 定义。
- 新发现的 PyTorch 2.15 API 与 FlashAttention 来源事实加入既有研究记录；章节审阅结论、字数、代码状态和同步修改仅记录于此 `progress/` 审计。

## Open risks 与 handoff

- CUDA 显存采样、FP16/BF16 GPU kernel 和性能、PyTorch 2.15 runtime、GradScaler overflow skip、真实多 rank 及生产 OOM 均未本地实测；相关正文使用固定版本来源说明并保留目标设备验证边界。
- checkpointing 仅以 CPU 可观察调用次数说明重算；没有测量目标 GPU 的显存—吞吐权衡。PyTorch 2.12 CPU demo 不代表 2.15 或其他 device 的运行结果。
- `reserved-allocated`、设备进程用量和 allocator snapshot 的差异仍需在具体 GPU 上结合测量解释；源码链接检查不等于外链完整性或渲染验收。
- 第六章摘要、题库、练习和术语表已同步；未更新 PDF 或 `plan.md`。
- **下一步：** 按目录进入第七章《分布式训练入门》，检查 DDP rank 步调、梯度同步、loss 分母和多 rank 故障边界；全套书 goal 继续 active。
