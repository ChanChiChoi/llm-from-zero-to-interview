# 第十九册第六章《SFT 与对齐训练的行为事故》精修审计

精修日期：2026-11-17。状态：首轮正文、章级扫描、目录和配套题同步完成；下一入口为第七章。

## 范围与读者路径

- **scope**：复核 6.1—6.19 的后训练行为契约、SFT shift/mask/denominator、能力回归、过度拒答、工具格式、偏好数据、Reward Model 捷径、RLHF reward hacking、DPO reference/beta/序列长度、指标定义域、诊断和恢复。
- **reader_path**：先声明允许变化和需要保留的行为；从渲染后样本与 causal shift 后 labels 确认训练目标；用固定评估切片定位行为变化；分别审计安全/风险请求、偏好方向和 reward 反事实；核对 DPO reference 与序列归约、工具 schema 版本；最后将观测变化、协议错误、任务策略违规和 unknown 分开，按预先声明的任务策略决定下一步。
- **本章贡献**：承接第二至五章的数据、tokenizer/template、shift/mask、有效监督目标和实际 loss denominator，把这些口径用于 SFT 与对齐行为事故；向第七章交接固定评估集、样本分母和不确定性。方法原理回链第五、八、十六册。
- **结构决定**：保留 19 节与 6.14.1—6.14.5 五个定义小节；覆盖偏好数据、reward、RLHF、DPO、安全和工具格式的必要主题仍能围绕“训练究竟改变了哪些行为”串联，无需拆章。将 6.15 的长 Python 审计器改为合成手算账本，移除任意通用阈值和综合决策状态。

## 章级扫描与候选去向

检索/核验日期：2026-11-17。问题：如何区分后训练后的真实行为变化、数据/协议错误和评估口径变化；SFT labels 与 DPO 序列概率的 reduction 如何影响目标解释；RM 与人类偏好何时可比较； reward hacking 结论能外推到什么范围？资料范围：InstructGPT 原论文、DPO 原论文、PyTorch 2.14 CrossEntropyLoss、TRL 固定版本 SFT/DPO 文档与源码、Gao 等人的 reward overoptimization 原论文、HUMAINE ICLR 2026 研究及 reward-hacking 跨任务预印本。来源事实和证据边界记于相关 research 文件。

| 候选知识 | 去向与判断 |
| --- | --- |
| causal shift 后的 assistant 目标、prompt/PAD mask 和实际 loss denominator | 扩写 6.2、6.14.1、6.15；区分 input tokens、有效目标数和 reduction 分母；按固定版本文档说明 `ignore_index` 与 assistant-only template 条件。 |
| 后训练可能同时改善某些任务并损伤其他切片 | 保留 6.3、6.12；用同口径 before/after 任务分子分母和不确定性，避免单一总分替代回归切片。 |
| 误拒与漏拒的目标集合不同 | 保留 6.4、6.14.3；分开分母，空集合与缺失观测分别报告。 |
| 偏好标签不是绝对质量；评审方向、rubric 和人群可能不同 | 扩写 6.6—6.7；用盲审方向、平局和无法判断项替代未校准的质量分差。 |
| reward 与人工分数不一定处于同一量尺 | 改写 6.8、6.14.4；不跨尺度相减，使用同 prompt 反事实和同 rubric 的排序一致性作为筛查。 |
| reward overoptimization 与跨任务效应受实验设置限制 | 扩写 6.9；引用合成 RM 研究和 2025 年预印本时明确各自条件，不写成普遍定律或线上发生率。 |
| DPO sequence log-prob 求和、reference 和长度归一化会定义不同目标 | 扩写 6.10、6.14.5、6.15；手算 reference-adjusted margin/loss，并区分 TRL v1.14.1 `sigmoid` 与 `sigmoid_norm` 的口径。 |
| 工具训练 schema 与线上执行器必须精确对齐 | 保留 6.5、6.15；以工具名和参数字段的版本化比较发现协议漂移。 |
| 任意固定阈值和综合 gate 会误把异质指标压成通过结论 | 删除原 6.15 审计代码；改用观测、协议问题、运行策略违规和 unknown 四类报告。 |

## 覆盖图与正文修改

| 知识点 | 读者作用及解释层级 | 本轮处理 |
| --- | --- | --- |
| 后训练行为契约与能力回归 | 区分目标行为变化与旧能力损失，并固定可比评估条件 | 改写 6.1、6.3、6.12；增加 InstructGPT 指定实验的回归证据和范围。 |
| SFT 监督与 reduction | 从目标 token、mask 到 loss numerator/denominator 复核训练目标 | 明确 shift 后 target 口径、等权 token mean 假设及权重/序列归约例外；链接 PyTorch/TRL 固定版本文档。 |
| 误拒、漏拒及安全行为 | 按应回答/应拒绝集合分别诊断用户可用性和风险 | 保留两个明确分母；手算例中安全集合 2/3 误拒、风险集合 1/3 不安全漏出，标明小样本限制。 |
| 偏好数据方向与人类评审 | 判断 chosen 标签与声明 rubric 是否一致，并保留 tie/unknown | 删除默认数值 quality margin；用方向审查及 1/2 chosen-win 教学账本，单列 1 个 tie 和 1 个 unknown。 |
| Reward Model 捷径和过度优化 | 检查模型是否因长度、免责声明或模板得到高代理分 | 不再计算不可比 reward-human gap；用同 prompt 长度反事实（RM 0.42 vs 0.88）作为扩大审查的线索，并说明单 pair 不能估计总体偏差。 |
| DPO reference 与序列长度 | 复算 margin 和 sigmoid loss，核对引用版本、tokenizer/template 和归约方式 | policy gap 0.7、reference gap 0.2、margin 0.5；beta 0.8 时 loss 约 0.513；说明 sft-v2/sft-v3 元数据差异需重算。 |
| 工具 schema | 发现训练调用与执行接口之间的名字/参数不一致 | 对比 `search(query)`/`calculator(expression)` 和线上 `web_search(query)`/`calculator(expr)`。 |
| 证据报告和恢复顺序 | 让工程结论可复核，不把任务策略伪装成普遍门槛 | 取消通用 gate pass；要求预先声明本次风险策略并把 unknown 保留为未知。 |

## 证据、公式与示例状态

- **evidence**：InstructGPT 支持该论文的 SFT—RM—PPO 流程和指定评测回退；PyTorch 2.14/TRL v1.14.1 支持相应固定版本的 ignore_index、assistant-only template 要求和分母口径；DPO 原论文及 TRL v1.14.1 固定源码支持 reference-adjusted objective 与 sequence-sum/normalized loss 差异；Gao 等人提供合成 RM overoptimization 案例；HUMAINE 和 reward-hacking 预印本仅支持各自样本/环境下的观察。事实边界与版本快照在[后训练损失与遗忘研究记录](../../research/sft-adaptation-loss-and-forgetting-2026.md)、[DPO 目标与长度研究记录](../../research/dpo-preference-objective-and-length-2026.md)、[Reward Model 过度优化研究记录](../../research/reward-model-evaluation-and-overoptimization-2026.md)、[偏好评审协议研究记录](../../research/human-preference-evaluation-and-rater-protocols-2026-11-03.md)和[reward-hacking 跨任务研究记录](../../research/reward-hacking-cross-task-generalization-2026-11-03.md)。
- **example_status**：全部为合成教学数据。SFT 输入 tokens 26；shift 后 assistant 目标 10、已标注 9（覆盖 9/10），prompt 目标 9 中误标 1，padding 目标 4 中误标 1，EOS 覆盖 2/3；给定分项 CE 后 numerator 9.1、实际有效 label denominator 11、mean 约 0.827。能力切片按 100 样本展示 −15、−17、−8 个百分点回退；误拒与漏拒分别为 2/3 和 1/3。偏好账本方向明确 pair 中 chosen 胜出 1/2，另有 tie 1、unknown 1。Reward 反事实只演示一个匹配 pair。DPO 手算如上；工具 schema 例仅用于显示接口字段差异。均不构成总体性能或发布标准。
- **code_status**：删除 6.15 原有约 300 行、含任意阈值和综合 decision 的 Python 审计器；正文不再包含可执行代码。未运行原代码、新代码、测试、模型、API 或 benchmark。
- **formula_status**：核心 SFT、能力变化、拒答分母、偏好方向、RM 排序一致性和 DPO margin/loss 均给出定义或算例；静态检查 2 个 math 围栏配对、98 个行内 `$` 标记成对、公式花括号平衡，未发现禁用写法；未进行 GitHub 页面实际渲染。

## 配套同步、链接与未决事项

- **sync_status**：第六章目录摘要已更新；`EXERCISES.md` 第 43A 题改为 shift 后 denominator、盲审方向、reward 反事实、DPO reference/schema 及四类报告；`INTERVIEW_BANK.md` 第 20A 题同步为无通用 gate pass 的诊断流程；第十九册架构地图和阶段状态已更新。
- **link_status**：本章、目录、架构审计、单篇审计和五份研究记录共检查 48 个本地 Markdown 链接目标，缺失 0；本次修改的练习和面试题文本没有新增链接。`git diff --check` 通过。GitHub MathJax 渲染和全量外链可达性未做。对 `EXERCISES.md` 与 `INTERVIEW_BANK.md` 全文件的宽范围扫描另发现 21 个位于其他条目中的既有缺失目标，本轮题目改写没有涉及这些链接，待相关条目/书册审阅时逐项确认。
- **open_risks**：手算样本只演示诊断口径，不代表总体统计；HUMAINE 结论限于其研究参与者和任务；reward-hacking 跨任务证据为预印本且依赖特定编码训练设计；TRL v1.14.1 是固定版本案例，文档页面提示已有更新版本；实际项目的 collator、loss reduction 和 DPO reference 仍需按其锁定版本回放；`EXERCISES.md` 与 `INTERVIEW_BANK.md` 的 21 个既有断链尚待核实。
- **下一入口**：第七章《评估指标陷阱》，从后训练变化交接到任务/样本/评分规则、分母、judge 偏差和线上可比性审计。
