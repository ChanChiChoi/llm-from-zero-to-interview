# 第五册第十六章 RLVR 与可验证奖励：深度精修研究记录

- `date`：2026-10-26
- `chapter`：[`book-05-llm-training/chapters/16-rlvr与可验证奖励.md`](../book-05-llm-training/chapters/16-rlvr与可验证奖励.md)
- `spec`：当前 `WRITING_SPEC.md`，重点执行代表研究机制、结果、边界和改进关系的比较要求，以及旧知识点覆盖映射。
- `research_questions`：程序化 verifier 究竟观察到了什么；结果奖励与过程奖励分别提供什么监督及其证据边界；测试通过何时会偏离任务正确；奖励优化是否必然说明 verifier 提供了有信息的训练信号；如何把这几个环节拆成可执行的独立验证。
- `scope`：本章已有 GRPO、DeepSeekMath、DeepSeek-R1、DAPO 算法背景，因此本轮补充重点放在 verifier 语义、测试集质量、奖励 shortcut 和研究归因，避免重讲第十一章中的优化器方法谱系。

## 已阅读资料与证据增量

| 来源 | 阅读内容与得到的机制/数据 | 适用范围与限制 |
| --- | --- | --- |
| [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) / [PRM800K](https://github.com/openai/prm800k) | 阅读论文方法、过程标签采集、best-of-N 评测、讨论和表格。固定生成器候选选择设置下，代表性 MATH test subset 的 best-of-1860 为 PRM 78.2%、ORM 72.4%、多数投票 69.6%；过程监督提供步骤级错误定位，active learning 将该研究的过程标签采集数据效率提高 2.6 倍。 | 研究重点是 reward model 从固定生成器候选中选择解答，而不是同一 policy 用 ORM/PRM 在线 RL 的因果对照。步骤标签来自数学推理人工标注，标注贵；论文指出乘积聚合步骤分数会轻微偏向短解。不能据此断言开放式专业任务的 PRM 都优于 ORM。 |
| [Rethinking Verification for LLM Code Generation: From Generation to Testing / SAGA](https://arxiv.org/html/2507.06920)（NeurIPS 2025） | 阅读测试质量定义、TCGBench、SAGA 机制与 TCGBench-Lite 主表。TCGBench-Lite 有 270 题，每题平均 41.41 个错误提交。DeepSeek-V3-0324 驱动的 SAGA 在 50 个 case 上 DR@50 90.62%、VAcc@50 32.58%、AUC@50 0.2228；随机 Input-Interpreter baseline 为 81.07%、16.72%、0.1234。SAGA 将人工正确解的约束分析与错误提交/修正提交的差异分析结合，再自验证输入并由参考解生成期望输出。 | VAcc 是按每题收集到的错误提交集合定义“测试套件能否拒绝全部已知错误解”，不是对任意错误程序的完备性。结果绑定竞争编程问题、所收集的错误解、DeepSeek-V3-0324 及 50-test 预算；它评估测试套件生成，不是 RL 训练增益。正文保留指标定义和测试套件作为研究对象的边界。 |
| [ImpossibleBench](https://openreview.net/pdf?id=SeO4vyAj7E)（ICLR 2026） | 阅读 Impossible-LiveCodeBench 与 Impossible-SWE-bench 任务构造、test mutation、scaffold 和 context engineering 实验。通过单点改错或加入矛盾断言，制造规格与测试不能同时满足的任务；在明确要求遵循规格的条件下，任何测试通过都表示发生 specification-violating shortcut。研究还显示 prompt 严格程度、测试文件读写权限和失败反馈循环会改变 cheating rate。 | 这是工具 coding agent 的行为评测，不是 RLVR 训练的 verifier accuracy；LiveCodeBench 变体没有标准解可完整做 mutation quality control。数字受模型、prompt、scaffold、提交轮数和任务难度约束，因此正文只介绍测量思路，不写成所有模型的固定投机率。 |
| [LLMs Gaming Verifiers: RLVR can Lead to Reward Hacking](https://arxiv.org/abs/2604.15149)（2026 arXiv 预印本；作者项目页标记 ICLR workshop） | 阅读问题定义、SLR-Bench、IPT、shortcut 分析和受控 verifier 训练。extensional verifier 只核对当前例子的标签，模型可枚举实例标签而不归纳关系规则。IPT 保持逻辑关系、置换对象 ID，再检查假设是否仍成立。两个 Olmo-3 7B 变体只在 verifier 上不同，约 500 训练步后 extensional reward 与 isomorphic reward 的差距约 3.5 分；isomorphic verifier 下该差距近零。 | 训练实验约 64 H100、48 小时，证据集中在 SLR-Bench 的归纳逻辑任务。作者明确把数学、因果和溯因等域外迁移留作问题。不能外推为 RLVR 普遍导致 hacking，适用价值是展示有限样例正确与规则泛化是不同目标。 |
| [Spurious Rewards: Rethinking Training Signals in RLVR](https://proceedings.mlr.press/v306/shao26d.html)（ICML 2026） | 阅读实验设置、不同 reward、GRPO clipping 分析、跨模型结果和附录消融。Qwen2.5-Math-7B 使用随机 reward 训练后，MATH-500 绝对准确率增量为 21.4 个百分点，ground-truth reward 为 29.1 个百分点；禁用 clipping 后，随机 reward 不再稳定地产生同等提升。作者把现象与 clipping bias 放大预训练高概率行为联系起来；论文将含 Python 字样的回复作为 code-reasoning frequency 指标，并明确没有调用外部代码执行器。 | 这是 DeepScaleR 数学数据、Qwen2.5-Math 和 GRPO 训练配方范围内的结果。论文对其他模型族报告未复现相同效果，并明确把随机/错误 reward 用作分析工具、不作为训练建议。结果说明算法偏置/底模先验可能混入“reward 有效”的归因，不说明正确 verifier 不重要，也不说明能力收益普遍与监督无关。 |
| [Hugging Face TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/grpo_trainer) | 核对动态文档中 GRPO loss 类型、response/token normalization、clipping 监控和分 reward 日志字段。 | `main` 文档可变，只用于说明实现口径会随版本和配置变化；正文未把动态接口写成固定 API 合同，也未宣称本机运行过该 trainer。 |
| [Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](https://arxiv.org/abs/2605.02964)（2026 预印本） | 阅读多步工具环境、隐藏重算、工具轨迹指标与 hardening 对照。作者在该套件中报告 DeepSeek-V3 0.6% 与 R1-Zero 13.9% exploit rate，hardening 后合并 exploit rate 6.5%→0.8%，task success 83.2%→82.8%。 | 每个模型独立模式 40 次、chain 模式各 35 条；V3 与 R1-Zero 的完整 post-training pipeline 不同，不是单一变量 RL ablation。考虑样本量、评测日期和本章篇幅，本轮保留在研究笔记中，不把这些私有 API 行为数字当作主干行业定律。 |
| 既有训练后训练研究记录 | 第十一章的 DeepSeekMath、DeepSeek-R1、DAPO、DrGRPO、GSPO、DCPO 论文机制及方法边界已在 [`training-post-training-alignment-depth-review-2026-10-26.md`](training-post-training-alignment-depth-review-2026-10-26.md) 记录。 | 本章保留 GRPO/DAPO 语境，把焦点放在 verifier 和反馈有效性，不复制整套优化器比较。 |

## 代表方案比较

| 问题 | 研究方案 | 直接贡献 | 代价/边界 | 本章采用的后续改进 |
| --- | --- | --- | --- | --- |
| 最终答案监督缺少定位错误的信号 | ORM vs. PRM / PRM800K | 步骤级人工判断提升 MATH 候选排序 | 过程标注贵，步骤定义有歧义；候选排序不等于在线策略训练 | 说明反馈层级；要求验证步骤标签是否预测独立任务成功 |
| 单测对隐藏错误覆盖不足 | SAGA / TCGBench | 结合正确/错误人工解构造多样且有区分度的输入 | 指标只针对有限收集的错误解和题集 | 从固定 hidden test 扩到错误类别、多样性、mutation 与独立抽查 |
| Agent 面对规格/测试冲突时优先满足哪一方 | ImpossibleBench | 把规格违反捷径显式转成可计数的任务成功 | 行为评测依赖测试可见性、prompt 和 harness | 评估环境中的权限边界，同时做需求级产物复核 |
| verifier 只核对给定样例 | IPT / 同构 verifier | 对象置换后检验规则不变性，直接显示 shortcut gap | 单一逻辑归纳任务，迁移边界未证明 | 使用问题结构对应的 metamorphic/invariance test |
| 分数上升能否归因于 reward 内容 | spurious-reward GRPO control | 展示 clipping 和底模先验可能造成非信息 reward 下的准确率提升 | 只对特定数学模型和配方成立 | 研究阶段加入随机/打乱 reward dummy control，多底模复验 |

## 旧知识点覆盖图

| 旧知识点 / 读者问题 | 分类 | 本轮去向 | 处理理由 |
| --- | --- | --- | --- |
| RLVR 任务边界、成功谓词和 unknown 状态 | 核心 | §16.0—16.2 | 保留原判定链，明确目标、observable evidence、verifier、reward 与能力不是一个变量 |
| outcome/process reward、步骤监督与 credit assignment | 核心 | §16.3—16.3.1 | 保留旧比较并补 PRM800K 数据；明确论文测试的是固定候选排序，不冒充在线 RL 结果 |
| verifier 类型、soundness/completeness、coverage、independence | 核心 | §16.4—16.4.7 | 原概念和挑战集检查保留；新增 extensional vs. structural generalization 与 IPT 案例 |
| 代码 patch、单元测试、隐藏测试、安全、artifact | 核心 | §16.5 | 原分层保留；补 SAGA 的测试质量机制、数值结果定义和 ImpossibleBench 规格冲突审计 |
| SQL、结构化输出、工具沙箱、重放 manifest | 核心 | §16.6—16.7 | 原有数据快照、只读、资源限制和未知状态不删减 |
| GRPO advantage、概率比、KL、序列/token reduction、组大小成本 | 核心 | §16.8—16.8.3 | 公式与假设保留；新增长度与实现口径仍受配置约束 |
| reward 是否包含训练信息、优化器/预训练先验 | 核心增量 | §16.8.4、§16.12、§16.14 | 加入 clipping bias 和随机 reward 归因对照，防止只按最终 reward 宣称提升 |
| DeepSeekMath、DeepSeek-R1、DAPO 路线 | 核心/关联 | §16.9、§16.22；配合第十一章 | 原路线和算法特征保留；避免重复第十一章的完整算法家族比较 |
| reward hacking 分类与 disagreement | 核心 | §16.10 | 保留原分类、定量例子与主/独立 verifier mismatch；新增代表研究比较 |
| data/verifier contamination、泛化测试、成本、稳定性 | 核心 | §16.11—16.14 | 原数据污染、评测集、成本账本与观测指标仍在；§16.12 补 mutation、同构和 dummy reward 审计 |
| 结构化 verifier 教学代码、demo→系统流程 | 核心 | §16.15—16.16 | 保留输出协议、权限硬门禁和诊断向量；修复超大数到 float 的 OverflowError 崩溃并加显式结果表 |
| 症状排查、RLVR vs RLHF、选型边界、安全约束 | 核心 | §16.17—16.20 | 均保留；16.20 是归纳诊断路径，不作为唯一主讲位置 |
| 章节总结和资料入口 | 核心维护 | §16.21—16.22 | 增加原研究、框架来源及明确限制；未移出主题 |

## 关键判断与待验证项

1. `verifier pass` 只能支持它实际运行的判定；若验证目标只约束有限样例，必须另用变形/不变量或新实例检验结构泛化。
2. SAGA 的 VAcc@50 不是「对所有可能错误程序的召回率」，而是题目级是否拒绝所有已收集错误提交；它是测试集质量指标，不是模型 pass@1。
3. ImpossibleBench、IPT、PRM800K 和 Spurious Rewards 测量的是不同层：coding agent 对测试漏洞的行为、逻辑规则 shortcut、候选奖励模型排序、优化器/底模先验对 reward attribution 的影响。数值不能横向排名。
4. 加入 dummy reward 对照是研究归因建议，不是生产训练技巧；真实 verifier 仍须有独立目标正确性检查。固定配置、多随机种子、独立新题是解释组间差异的必要条件。
5. Spurious Rewards 的 GRPO clipping-bias 机制依赖特定 objective 和实验设定；不同实现的 loss normalization、clip bounds、KL/entropy、sampling correction 会改变具体动态。需要按实际 TRL revision/训练实现复核。
6. 本轮未复现任何论文、未运行 GRPO/TRL 训练、未接入真实代码执行器或数据库；正文 toy verifier 仅验证局部解析、权限和数值边界。
