# 第十九册第四章《训练不稳定排查宝典》精修审计

精修日期：2026-11-17。状态：首轮正文、章级扫描、练习/面试题同步完成；下一入口为第五章。

## 范围与读者路径

- **scope**：复核第 4 章 4.1—4.21 的训练时间线、first bad step、loss spike、更新停滞、NaN/Inf、梯度裁剪、有效监督目标、scheduler、AMP、异常 batch、train/eval、checkpoint、rank 差异、监控、复现和恢复验证。
- **reader_path**：从一次可定位的 step 记录出发，先确认监督目标和 loss 分母，再沿 forward、backward、optimizer update、rank 归约和 checkpoint 恢复找到首次分歧；以同 checkpoint 的固定 batch 做反事实复现，最后用已声明的训练策略、小规模训练和目标规模回归验证修复。
- **本章贡献**：承接第二章的数据分布和第三章的序列/监督协议，把原始记录追到模型实际优化的目标、更新状态和恢复连续性；分布式通信机制交第五章，行为变化和对齐目标交第六章，通用训练理论回链第五册。
- **结构决定**：保留原有 21 节。所有故障类型仍由“首次异常—关联状态—最小复现—恢复验证”串联，没有发现需要拆为独立章节的互不依赖读者结果。

## 章级扫描与候选去向

检索/核验日期：2026-11-17。问题：训练故障指标能否定位实际 loss 目标，而不是只检查原始 labels；跨 rank loss、AMP overflow 和 resume 状态应如何解释；当前研究中的 spike 机制是否支持可迁移的固定阈值或默认修复？范围：PyTorch 2.15 autograd、AMP、gradient clipping、DDP、Distributed Checkpoint 官方文档；Transformers Trainer 当前公开恢复文档；已有训练稳定性与优化器研究记录中的 Adam、AMP 和输出头有限精度论文。此扫描聚焦第四章的诊断与恢复，不把论文候选提升成新章节或普遍训练处方。

| 候选知识 | 去向与判断 |
| --- | --- |
| labels 的原始非忽略计数、causal shift 后目标位置和 loss reduction 分母可能不同 | 核心内容留在 4.7、4.11、4.16.2，并修订 toy log 字段。它直接承接第三章 `T_loss` 定义，避免把 `labels != -100` 误作实际 loss 分母。 |
| rank local mean 的可比性和全局 token mean 的加权 | 扩写 4.13，加入两 rank 手算；每 rank 同时记录 loss numerator、denominator 和有效目标数。DDP 的默认梯度平均是公开 API 语义；全局 token mean 缩放公式明确标为基于默认 reducer 假设的推论。通信拓扑和 collective 故障交第五章。 |
| AMP 的 unscale/clip 顺序和 overflow 跳过更新 | 纳入 4.6、4.9。单次 overflow 作为事件，不作为普遍硬失败；是否超过容忍度由该次运行的 scaler、更新计数和已声明策略决定。 |
| checkpoint 恢复、数据跳过和重分片 | 纳入 4.12，引用 Transformers Trainer 对 optimizer/scheduler/RNG/data skip 的当前公开说明，以及 PyTorch Distributed Checkpoint 的 resharding 和跨版本兼容边界。框架行为不外推到自定义数据管线或优化轨迹等价。 |
| Adam/预条件器不稳定及有限精度造成迟发 spike 的研究假设 | 在 4.3、4.9 作为有边界的候选机制入口，使用 Molybog、Bai 和 Chen 等论文；强调不同实验设置，不能单凭 loss 曲线推断根因或复制超参。 |
| 诊断阈值 | 原代码中的 128 有效 token、loss +1、grad norm 5、LR 误差 `1e-5`、rank spread 1 被改成可选的 toy run policy 输入；代码默认不设阈值，只保留测量值、硬失败、策略违规和 unknown 分类。 |

## 覆盖图与正文修改

| 知识点 | 读者作用及解释层级 | 本轮处理 |
| --- | --- | --- |
| step 时间线和 first event | 让读者绑定 batch、目标、loss、梯度、更新、rank、版本和恢复状态 | 4.1、4.2 的日志合同补入 loss contract、denominator 和 rank 目标数。 |
| loss spike、停滞、NaN/Inf 和异常 batch | 区分现象与根因，沿前向/反向/更新定位 | 保留排障顺序；补 anomaly detection 的公开边界和 spike 论文的实验限制。 |
| 梯度范数、裁剪、loss normalization | 解释裁剪前梯度、effective batch 和目标分母的关系 | 4.6 加入 AMP 累积后 unscale 再裁剪、返回值为裁剪前总范数及非有限中止边界。 |
| 有效监督目标 | 解释模型真正参与损失计算的位置与 count | 4.7 以 `T_loss` 定义 `n_t`，区分 raw labels 与实际 denominator；提供六 token causal shift 手算。 |
| AMP、scheduler、checkpoint、rank | 判断更新跳过、恢复跳变和 rank 不一致 | 4.9、4.12、4.13 加入 PyTorch/HF API 事实和边界；rank 例算为 `(3+6)/(2+6)=1.125`，并指出两个 rank mean 的简单平均为 1.25。 |
| 监控、状态分类和示例审计器 | 把观察值转成可复核告警，避免把未知误判为稳定 | 4.14—4.17 分开测量、硬失败、策略违规和 unknown；移除通用阈值默认值。 |

主要正文修改集中在 4.1—4.17 与 4.20—4.21：明确 loss numerator/denominator 与有效监督目标数；train/eval 汇总采用相同 loss contract；rank loss 先按各自分母理解，跨 rank 的 token mean 按总 numerator/总 denominator 计算；checkpoint 恢复区分框架声明和自定义数据游标；状态 `failed` 只由硬失败或本次运行显式配置的策略触发。目录中的本章摘要仍覆盖训练异常定位和恢复验证，章节职责未变，无需改简介/目录。

## 证据与例子状态

- **证据**：PyTorch 2.15 anomaly detection、AMP examples、`clip_grad_norm_`、DDP design note、Distributed Checkpoint 和 FSDP 官方页面；Transformers Trainer 当前滚动文档和 checkpoint resume 教程，检索快照日期 2026-11-17；Molybog et al. 2023、Bai et al. 2026、Chen et al. 2026 论文。公开 API 仅支持相应版本的行为，论文实验不构成目标任务复现或通用处方。事实与来源证据存于[训练稳定性与调试研究记录](../../research/training-stability-and-debug-2026.md)及[PyTorch 训练 API 边界记录](../../research/pytorch-training-api-boundaries-2026.md)。
- **example_status**：4.7 六 token 序列在内部 shift 下，raw labels 非忽略数为 6、有效目标位置数为 5；仅监督答案与 EOS 时有效目标数为 2。4.13 rank0 loss sum 3/目标 2、rank1 loss sum 6/目标 6，rank means 为 1.5/1.0，全局 token mean 为 1.125。均为手算教学输入。
- **code_status**：4.17 标为“静态审阅，未运行”。toy 日志记录 raw count、shift 后目标数、loss contract、denominator、各 rank numerator/denominator/target count；默认无阈值，样例策略仅供演示。未运行章节代码或测试。
- **formula_status**：已核对 causal shift 计数、token mean 分母、rank spread 与 DDP 梯度缩放假设；关键变量均在相邻正文说明。未做 GitHub MathJax 页面渲染。

## 配套、链接与未决事项

- **sync_status**：更新 `EXERCISES.md` 第 173 题，要求显式 loss contract、shift 后目标数、分母和 per-rank 口径；更新 `INTERVIEW_BANK.md` 第 78 题，区分事件、硬失败、策略违规和 unknown。简介/目录摘要与本章读者结果一致，无需改动。
- **link_status**：正文和配套文件的本地链接待本轮静态目标检查后记录；外部链接只做本章相关来源定向打开，不代表全量可达性确认。
- **open_risks**：代码未运行，策略数值不适用于真实任务；HF `main` 文档是滚动页面，本轮记录的是 2026-11-17 快照；MathJax 页面渲染、外链长期可达性和第十九册整册连续复读仍待后续门禁。第五章需承接本章 rank loss 数学口径，但避免重复说明本章通用 loss 时间线。
- **下一入口**：第五章《分布式训练事故宝典》，重点检查 rank、collective、数据/通信关键路径、checkpoint 拓扑和本章已交出的 per-rank 分母口径。
