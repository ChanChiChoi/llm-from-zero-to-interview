# 第十四册第五章审计：训练循环工程

审阅日期：2026-11-09

## Scope 与章节决定

- 完整复读《训练循环工程》§§0–5.23，并按章级问题检索 PyTorch 2.15 官方 API 文档及 Python 3.12 标准库文档；研究资料见[PyTorch API 边界记录](../../research/pytorch-training-api-boundaries-2026.md)。本章十章骨架不变。
- 核心问题是怎样让 loss 的统计分母、参数更新次数、验证指标和恢复状态采用同一训练约定。正文保持三条连续教学链：训练 step 与参数更新；验证、检查点与恢复；可复现诊断与端到端训练循环。
- 章级扫描重点核对：`zero_grad(set_to_none=True)` 与全零梯度的区别；scheduler 的调用和恢复顺序；全局梯度范数与非有限检查；shift 后有效目标数及 class-weight reduction 分母；`eval()` 与 inference mode 的边界；`weights_only=True`、随机状态和数据游标；原子替换的文件系统边界。
- 扫描未发现需要新增独立主题或改变本册边界的候选。将有证据的 API 边界并入 §§5.2、5.6–5.14；AMP、GradScaler、DDP collective 和多文件 checkpoint 仅保留边界与后续章节入口。

## Length status

按连续解释链计数汉字，排除标题、代码、公式、表格、状态标记、行内代码和 URL；不把练习、资料清单或总结当成独立教学单元。

| 教学链 | 覆盖范围 | 实计汉字 | 类型与门槛 | 状态 |
| --- | --- | ---: | --- | --- |
| 训练 step 与参数更新 | §§5.1–5.8 | 3,136 | 核心，≥1,800 | 通过 |
| 验证、检查点与恢复 | §§5.9–5.13 | 2,168 | 核心，≥1,800 | 通过 |
| 可复现诊断与端到端训练循环 | §§5.14–5.19 | 1,932 | 核心，≥1,800 | 通过 |

## Evidence 与 example status

- API 事实使用固定版本 PyTorch 2.15 官方文档；文件替换语义使用 Python 3.12 官方文档。核对范围包括 optimizer/scheduler、梯度裁剪、CrossEntropyLoss、module/autograd、torch.load/serialization、Generator/DataLoader 与 deterministic algorithms。研究记录只保留来源事实、版本和证据边界，不记章节进度。
- §5.7.3 手算：两个 micro-batch 分别有 4、16 个有效目标，平均 loss 为 2、1；batch mean 平均为 1.5，token mean 为 `(4×2+16×1)/20=1.2`。加权 CE 的分母必须改用目标权重和。
- §5.8 手算：10 个 micro-batch、`accum_steps=4`，含两个完整窗口和一个 2-batch 尾窗，共 3 次 optimizer/scheduler 更新；100 次 warmup 若误按 micro-step 更新，会在 25 次参数更新后结束。
- §5.10 复用 4/16 个目标的聚合例，明确验证集按有效目标数加权。
- §5.17 demo：初始学习率 0.08、`gamma=0.8`、两次真实更新后 `final_lr=0.0512`；两个累积窗口有 17 和 13 个有效目标。恢复前后固定 batch loss 一致，且两条分支执行下一步后权重、optimizer state 和 scheduler state 均一致。数据为同一组 toy batch，不能作为 held-out 泛化证据。
- §5.19.4 跨 rank 手算：`(loss_sum,count)=(9,3)` 和 `(8,8)`；平均 rank mean 为 2，全局 token mean 为 `17/11≈1.545`，说明应先全局累加分子和分母再相除。

## Code status

本章首次为 39 个 Python 代码块补相邻状态，之前没有可追溯的逐块验证记录。静态审阅块均明确标为未运行；已验证块均标注日期、环境和输入边界。

| Python 块编号 | 位置 | 状态 |
| --- | --- | --- |
| 01 | §5.1 最小循环 | 静态审阅，未运行 |
| 02 | §5.2 最小训练 step | 已验证 |
| 03 | §5.2 `zero_grad` 调用片段 | 静态审阅，未运行 |
| 04–08 | §5.3 device 与递归搬运片段 | 静态审阅，未运行 |
| 09–10 | §5.4 模型输出与 loss 接口片段 | 静态审阅，未运行 |
| 11–12 | §5.5 shift loss 片段 | 静态审阅，未运行 |
| 13–14 | §5.6 backward 与梯度裁剪片段 | 静态审阅，未运行 |
| 15 | §5.7.1 micro-batch 等权累积 | 静态审阅，未运行 |
| 16 | §5.7.3 token 加权累积 | 已验证 |
| 17–18 | §5.8 optimizer/scheduler 更新片段 | 静态审阅，未运行 |
| 19–22 | §5.9 日志与 loss 记录片段 | 静态审阅，未运行 |
| 23 | §5.10 token-weighted validation | 已验证 |
| 24 | §5.10 shift 后有效目标计数 | 静态审阅，未运行 |
| 25–26 | §5.11 checkpoint 字段及 CUDA RNG 片段 | 静态审阅，未运行 |
| 27 | §5.11 Python/NumPy/Torch/loader RNG 保存 | 已验证 |
| 28 | §5.12 模型、scheduler、optimizer 状态恢复 | 静态审阅，未运行 |
| 29 | §5.12 Python/NumPy/Torch/loader RNG 恢复 | 已验证 |
| 30 | §5.12 推理权重加载 | 静态审阅，未运行 |
| 31 | §5.13 best/last 指标状态更新片段 | 静态审阅，未运行 |
| 32 | §5.13 单文件原子保存函数 | 已验证 |
| 33–34 | §5.14 seed 与确定性设置 | 静态审阅，未运行 |
| 35–37 | §5.15 异常检测和 bad-batch 保存片段 | 静态审阅，未运行 |
| 38 | §5.16 工程化训练循环骨架 | 已验证 |
| 39 | §5.17 完整训练循环与恢复 demo | 已验证 |

八个验证块的执行记录：

| 块 | 命令与输入 | 结果及边界 |
| --- | --- | --- |
| 02 | `.tmp/llm-book-cpu-check/bin/python` 抽取并执行该 fenced block；输入随机生成的 batch `[32,10]`、3 类标签 | step 完成，loss 和参数有限；一次观察到 loss=1.1638。 |
| 16 | 同一 Python 环境；4 个合成 batch 的 shift 后有效目标数为 `[1,0,1,1]`，`accum_steps=2`，用计数 optimizer 检查边界 | 两次 optimizer step；含零目标 micro-batch 的窗口仍按位置结束，最终标量参数为约 0.8（包括裁剪影响）。 |
| 23 | 同一 Python 环境；验证 batch 分别有 4/16 个有效目标，均值 loss 为 2/1；再传入全忽略目标 batch | 聚合为 1.2，原训练模式恢复；全忽略 batch 抛出 `ValueError`。 |
| 27、29 | 同一 Python 环境，PyTorch 2.12.0+cpu、NumPy 2.4.6；保存并用 `weights_only=True` 加载，比较 Python/NumPy/Torch/DataLoader generator 后续随机值 | 状态可受限加载，四种随机流的后续值一致。CUDA 分支未执行。 |
| 32 | 同一 Python 环境；临时目录保存含 tensor 的状态，再以 `weights_only=True` 回读 | `os.replace` 目标文件存在且状态相等；只验证本地临时目录单文件路径。 |
| 38 | 同一 Python 环境；5 个合成 micro-batch，`accum_steps=2` | optimizer 共更新 3 次，scheduler `last_epoch=3`，参数有限；CPU、map-style sized loader 情形。 |
| 39 | 同一 Python 环境执行完整 demo | 输出与正文预期一致：2 次更新、scheduler step 2、final lr 0.0512、final token loss 1.1381；恢复后模型/optimizer/scheduler 的下一步状态匹配。 |

另用同一环境对 39 个 Python 块全部执行 `ast.parse`，语法检查通过。环境为 Python 3.12.9、PyTorch 2.12.0+cpu；除 RNG 块外不依赖 NumPy。上述本地观察不代表 PyTorch 2.15 runtime、CUDA、多 rank、multi-worker 断点恢复或生产训练结果。

## Formula、link、visual 与 sync status

- 12 个 `~~~math` 围栏成对，逐块花括号计数平衡；正文没有行内 `$` 数学标记。尚未在 GitHub/浏览器进行 MathJax 渲染。
- 共 34 个 Markdown 链接：31 个 PyTorch 2.15 / Python 3.12 官方外链，3 个本地链接；本地目标为第四章两处和第七章一处，均存在。未宣称完整外链 HTTP crawl。
- 视觉机会已审视：§5.8 更新窗口表呈现 micro-batch、optimizer step、global step 和 scheduler 更新的对应关系；其他顺序在编号流程、公式和状态字段中已可直接追踪，本章未增加 Mermaid 图。
- 已同步第十四册[简介](../../book-14-pytorch-deep-learning-engineering/简介.md)与[目录](../../book-14-pytorch-deep-learning-engineering/目录.md)的第五章摘要；`INTERVIEW_BANK.md` Q33–40、`EXERCISES.md` 125–132 和术语表训练循环条目已按 loss 分母、更新步、checkpoint 和错误边界更新。
- 修改判断和本地验证写入本审计；PyTorch API 来源事实写入既有 research 记录。

## Open risks 与 handoff

- PyTorch 2.15 runtime、CUDA、AMP/GradScaler、真实多 rank、multi-worker 预取队列和外部存储均未实测；正文明确这些边界并交给相应章节。
- `torch.load(weights_only=True)` 的受限状态表示已用 PyTorch 2.12 CPU 本地验证；其他自定义 config/data cursor 类型仍须转成受支持的基础类型或单独审查。
- 后续整册复读仍需检查 5→6 的训练状态交接、目录主路径，以及 MathJax/GitHub 页面渲染条件。
- 下一步按目录精修第六章《混合精度与显存优化》。全套书 goal 继续 active；不更新 PDF 或 `plan.md`。
