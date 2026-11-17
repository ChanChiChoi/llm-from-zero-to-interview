# 第十九册第五章《分布式训练事故宝典》精修审计

精修日期：2026-11-17。状态：首轮正文、章级扫描、练习/面试题同步完成；下一入口为第六章。

## 范围与读者路径

- **scope**：复核 5.1—5.20 的分布式状态、step phase、rank 与 collective、hang 定位、token 负载、straggler、关键路径、环境证据、checkpoint、resume、FSDP/ZeRO、pipeline bubble、监控、逐级复现和恢复。
- **reader_path**：先用同一观测窗口重建预期 rank、当前 step/phase 与 collective 合同；再区分输入计算负载、shift/mask 后有效监督目标和实际 loss denominator；沿数据、计算、通信阶段定位延迟；核对 checkpoint 的预期/实际 shard 和训练状态；最后在固定环境逐级复现并验证恢复。
- **本章贡献**：承接第四章已经定义的目标 shift、loss mask、有效目标数、loss numerator/denominator 和 per-rank 加权，将它们用于多 rank 事故定位。FSDP/ZeRO、collective 和 checkpoint API 以版本文档为边界；第六章接手 SFT/对齐行为变化，第七章接手评估比较。
- **结构决定**：保留原有 20 节，没有发现需拆章的互不依赖读者结果。主要修复集中在 5.3—5.16 的定义、证据边界和示例审计器。

## 章级扫描与候选去向

检索/核验日期：2026-11-17。问题：如何从当前分布式训练 API 准确定位 rank 失步、collective hang、负载倾斜和恢复偏差；哪些监控量能直接比较，哪些必须绑定共同窗口、loss contract 或运行策略？范围：PyTorch 2.15 DDP、分布式调试、NCCL 环境变量、FSDP、Distributed Checkpoint 与 torchrun 文档；PyTorch 2.15 发布说明；NVIDIA NCCL troubleshooting；ZeRO 原论文；GPipe 原论文。证据快照与未验证边界记入[分布式训练系统 API 与平台边界](../../research/distributed-training-systems-2026.md)及[分布式训练并行研究记录](../../research/distributed-training-parallelism-2026.md)。

| 候选知识 | 去向与判断 |
| --- | --- |
| rank 日志只比较成员或 collective 名称会漏掉缺 rank、乱序、重复调用、group/shape/dtype 不同 | 扩写 5.3、5.13、5.15 和 5.16。审计器接收显式预期 rank 集与逐 rank ordered signature；只对完整的同一观测窗口计算全 rank 比率。缺 rank 日志列为 unknown，协议可比的 step/collective 不一致单独报告。 |
| input token、有效监督目标数与 loss denominator 不同 | 扩写 5.1、5.5、5.13、5.15、5.16。input token 说明计算负载；shift/mask 后目标数说明监督曝光；loss sum 与实际 denominator 按第四章 loss contract 汇总，不以 input token 替代。 |
| 固定 token 不均衡、straggler、通信占比和 LR 容差没有通用适用性 | 改写 5.10、5.16：分别保留观测差值和比率，阈值只能由调用方显式提供本次运行的 RunPolicy；不输出综合分或默认 gate pass。 |
| checkpoint 预期 shard 的子集齐全仍可能存在额外 shard；拓扑重分片不保证轨迹连续 | 扩写 5.9—5.10、5.15—5.16：并列报告 missing / unexpected，另检 hash、metadata、提交标志、optimizer/scheduler、数据游标和版本；将 DCP reshard 与应用级恢复分开。 |
| PyTorch 2.15 hang 诊断与 pipeline bubble 公式的边界 | 5.4 引入 flight recorder、NCCL trace/debug 环境变量和 Gloo `monitored_barrier`，明确它们用于定位而不恢复 worker；5.12 将 bubble 公式限定在 GPipe 式均匀 stage 理想调度。 |

## 覆盖图与正文修改

| 知识点 | 读者作用及解释层级 | 本轮处理 |
| --- | --- | --- |
| rank 状态、step/phase 与 collective 合同 | 判断各进程是在相同训练状态推进，还是在数据、条件分支或 collective 上分叉 | 保留逐级调查路径；collective 由成员检查改为按位置比较类型、group、shape、dtype 和完整顺序。 |
| hang 取证 | 将最后进度定位到 dataloader、backward、collective、保存或异常退出 | 加入 PyTorch 2.15 公开诊断入口及 `monitored_barrier` 的 Gloo/开销边界。 |
| 负载与 loss 口径 | 分别解释 token 计算负载、监督覆盖与训练目标 reduction | 输入 token 手算为 8192/8192/4096/16384，`I_input=4`；复用第四章手算，rank loss sum/denominator 为 3/2 与 6/6，全局 mean 1.125、局部 mean 简单平均 1.25。 |
| straggler、通信和关键路径 | 先报告比率，再结合阶段时间和共同观测窗口查原因 | 保留并逐 rank 输出 `T_data`、`T_compute`、`T_comm`、`T_step`；代码不再内置 1.5、0.35 等通用门槛。 |
| checkpoint 与恢复连续性 | 区分文件集合完整、状态可加载和训练轨迹连续 | 同时报 missing 与 unexpected shard；resume 分别报告 `Δstep`、`Δlr`，不合成加权分数。 |
| FSDP/ZeRO 与 pipeline | 解释分片带来的常驻状态变化、临时峰值与理想 bubble | 区分 ZeRO stage 和 FSDP1/2；用 `P=4`、`M=4/8` 复算理想 bubble 42.9%/27.3%，并加上 GPipe 调度假设。 |
| 指标代码与结论 | 保留机器可审计的输入输出，同时避免 unknown 被误判为通过 | 5.16 按 rank coverage、观测量、protocol errors、run-policy violations 和 unknown 分栏；checkpoint coverage 0.75 且报告额外 rank4 shard；step 不齐时不汇总全局 loss。 |

## 证据与例子状态

- **evidence**：PyTorch 2.15 DDP design/debug、NCCL 环境变量、FSDP1/FSDP2、Distributed Checkpoint、torchrun 官方版本文档及 2.15 发布说明；NVIDIA NCCL Troubleshooting；ZeRO 原论文和 GPipe 原论文。官方材料只支持对应版本的公开接口与机制，不证明目标集群性能或特定事故根因。来源事实、日期和范围见[研究记录](../../research/distributed-training-systems-2026.md)与[并行研究记录](../../research/distributed-training-parallelism-2026.md)。
- **example_status**：四 rank toy 快照中输入 token 不均衡 4，step 不齐（rank2 为 121，其余为 122），rank1 collective 首个位置不匹配；step-time straggler 比率约 1.903，rank3 comm/step 为 0.392；checkpoint coverage 为 3/4，另有 `rank4.bin`；resume LR 差值为 0.0001；`P=4,M=2` 的理想 bubble 为 0.6。均为手算/构造输入，不代表生产阈值或结果。
- **code_status**：5.16 已重写，状态为“静态审阅，未运行”；只依赖 Python 标准库。未运行代码、测试、模型、API 或 benchmark。此状态不能证明章节代码已经执行。
- **formula_status**：分别定义 input-token imbalance、straggler、communication ratio、checkpoint coverage、resume 向量差和理想化 pipeline bubble；复核主要算例。静态检查到 15 个 math block，花括号均配对；未发现禁用公式写法或未配对行内 `$`。未做 GitHub MathJax 实际渲染。

## 配套、链接与未决事项

- **sync_status**：目录第五章摘要补入 input token、有效目标与 loss 分母；更新 `EXERCISES.md` 第 148A 题和 `INTERVIEW_BANK.md` 第 56A 题，移除综合 gate pass 要求并要求显式运行策略。更新第十九册架构审计；新核验的 PyTorch hang 诊断来源写入 research，不将审阅流水放入 research。
- **link_status**：本章、目录、架构审计、本审计和分布式系统研究记录共检查 42 个本地 Markdown 链接目标，缺失 0；`EXERCISES.md` 与 `INTERVIEW_BANK.md` 的本轮变更未新增链接。`git diff --check` 通过；本章 5.1—5.20 连续，Markdown 围栏 19 对，未发现正文中的维护路径残留。未做外链全量可达性或 GitHub 页面渲染检查。
- **open_risks**：代码未运行；NCCL/FSDP/DCP 行为绑定 PyTorch 2.15 和目标环境；GPipe bubble 是理想化近似；GitHub MathJax 实际渲染、外链全量可达性和第十九册整册连续复读尚未完成。
- **下一入口**：第六章《SFT 与对齐训练坑》，围绕监督、偏好、奖励与优化设置检查能力回退、过度拒答、风格漂移和奖励捷径，并与第七章评估方法衔接。
