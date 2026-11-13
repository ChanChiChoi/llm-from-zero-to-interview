# 第四章《Verifier 与 Reward Model》精修记录

审阅日期：2026-11-12  
状态：章节正文、章级扫描、代码、公式、链接、视觉与简介摘要已同步；按用户要求不更新 PDF。

## 范围与章级扫描

- 完整复读 §§0.1–19，包括验证对象、outcome/process/programmatic verifier、reward model、pointwise/pairwise/listwise 目标、数据与 hard negative、reranking、指标、reward hacking、安全、练习、资料索引及独立 Python 示例；对照第三章多候选聚合、第五章过程监督和第六章搜索的分工。
- 章级扫描日期为 2026-11-12，时间窗覆盖 2021—2026；阅读原始论文、固定版本预印本与 ICLR/NeurIPS 正式会议页面。联网发现的新知识均已有研究记录；本轮只在本文件记录审阅决定，没有将章节进度写入 research。

| 候选主题 | 来源与正文去向 |
|---|---|
| 数学候选筛选和过程监督 | Cobbe 等人的固定版论文与 Lightman 等人的过程监督论文作为结果/过程信号的代表路线，引用限定于论文任务和设置；完整步骤标注问题留给第五章。 |
| 代码测试套件质量不等于单题完整判别 | Ma 等人的 TCGBench-Lite 指标纳入 §2.4。其 270 道题平均每题 41.41 个已收集错误提交；在 @50 设置下 SAGA 的 DR@50 为 90.62%、VAcc@50 为 32.58%。解释两个分母/目标不同，数字绑定论文数据与测试预算，不作为通用检错率。 |
| 规格与测试相冲突时的测试利用 | ICLR 2026 ImpossibleBench 纳入 §4.2，说明通过测试不必然符合自然语言规格；清楚标记它是人为构造的诊断基准，不代表生产系统发生率。 |
| RewardBench 的偏好评估及其限制 | 在 §0.1、§9.5、§10.1 引用固定 v2；保留聊天、推理、安全切片以及半自动偏好对、格式伪相关和下游相关性未定等边界。 |
| 代理分数过度优化 | Gao 等人的合成 gold RM 研究纳入 §10.5；明确其 fixed 6B gold RM、best-of-N/RL 设置和非人类 ground truth，不推通用阈值。 |
| Verifier 的隐式捷径 | Helff 等人的 2026 预印本纳入 §10.6，描述归纳逻辑中逐例标签枚举与同构规则迁移的差别；明确证据只覆盖 SLR-Bench 和受控 7B 训练，不外推至数学、代码或所有 RLVR。 |
| 通用 LLM judge 的位置/重复性等协议研究 | 不扩成独立讲次；本章保留 reward model / verifier 的目标与下游选择，通用评审协议与统计设计交第七册。 |

## 正文修改

- 整册连续复读（2026-11-13）发现 §11.3 的总成本求和未显式声明各分量单位一致；已补充统一换算单位与异质原始量改用成本向量的条件。可执行代码、依赖、输入输出和预期结果未变，沿用原验证记录。
- 将 verifier 与 reward model 的关系改为“检查角色”和“反馈训练来源”两条轴，不把二者误写成互斥模型类别；新增 outcome/process 与 programmatic/learned 对照表。
- 补充测试覆盖与自然语言规格的一致性、候选分布、reward overoptimization 和隐式 verifier shortcut 的研究证据；各数字和结论均紧邻来源，并注明数据范围。
- 为测试集通过率、过程均分、多目标线性分数、pointwise/pairwise/listwise 损失、pairwise/top-1/hard-negative/ECE、分数间隔和 token 单位成功成本补充逐步数值代入。合同算例拆出折扣、税率乘法的中间值。
- 固定并核对资料索引版本；新增 TCGBench、ImpossibleBench、verifier shortcut 与 overoptimization 原始来源，移除与本章无关的 Self-Consistency 条目，并删除作者联网访问日志。
- 修复 demo：RM score 对浮点单独检查有限性，避免巨大 Python 整数进入 math.isfinite 时触发浮点溢出；hard-negative 指标无 hard negative 时返回 None，不再对空正确候选集执行 max；ECE 最后一桶明确含 1.0；将 cost_per_hybrid_correct 改为 tokens_per_hybrid_correct，以匹配实际只统计 token 的账本。
- 手算复核五道 toy case 的 pairwise accuracy：10 个正确—错误候选对中有 9 个排序正确，因此原预期 0.9 保持不变。纯 RM top-1 为 4/5，混合选择为 5/5，greedy 为 1/5；均只描述固定 toy 数据。
- 更新简介第四章摘要；正文添加第三章入口及第五、第六章交接。章节编号和目录顺序未变。

## 验收字段

- **length_status：** 单篇核心讲解，剔除代码、公式围栏、表格、标题、练习、总结和资料索引后，连贯解释性中文实计 7,378 汉字，高于 1,800 字下限；连续复读补入成本单位条件后计数增加 43 字；统计不替代人工深度判断。
- **example_status：** 覆盖 4 个测试 3 个通过的 \(3/4\) 测试集率、三步 \(1,0.5,1\) 的过程均分、多目标分数 0.69、三种训练损失、distractor_math 的成对排序和 top-1、五题整体选择准确率、4 个 hard negative 的 \(3/4\)、三桶 ECE 的 0.165、分差 0.13、693/5=138.6 生成 tokens/正确案例，以及合同计算 95,000×1.06 和 95,000×1.60。均为可复算的教学输入。
- **evidence：** 数学候选验证为 arXiv:2110.14168v2；过程监督为 arXiv:2305.20050v1；InstructGPT 为 v1；摘要偏好研究为 v3；RewardBench 为 v2；代码测试覆盖为 arXiv:2507.06920v2；ImpossibleBench 使用 ICLR 2026 正式会议页；RLVR shortcut 为 arXiv:2604.15149v1；reward overoptimization 为 arXiv:2210.10760v1。来源主张均限定到对应论文数据、模型、标注和协议。
- **formula_status：** 复核 19 个 display-math 块，变量、非空分母、分数范围、并列处理与概率边界均有说明；总成本求和显式要求统一单位，异质资源量应保留为向量；代码和数学围栏配对，公式花括号计数平衡。
- **code_status：** §13 标准库示例于 2026-11-12 在 Python 3.12.9 运行；从 Markdown 提取代码，捕获 stdout 与预期输出逐行完全一致。另做 12 项边界检查：bool/NaN/无穷/负值/越界/超大整数 RM 分数拒绝；0 和 1 两个分数端点接受；bool token 拒绝；没有 hard negative 时返回 None；缺少正确对照的 hard negative 抛出 ValueError；分数 1.0 进入含上端点的 ECE 桶。只验证 5 个预制 toy case 和分数/账本合同，不运行模型、API、真实测试沙箱或生产服务。
- **link_status：** 本章 18 个 Markdown 链接中 5 个本地链接全部存在，13 个论文/会议外链已按固定版本核对；简介本地链接同步检查无断链。
- **visual_status：** 新增“检查粒度 × 实现形式”对照表，避免把 outcome/process 与 programmatic/learned 混成同一分类轴；本章其余数据流由案例和算式表达，不需新增 Mermaid。
- **sync_status：** 简介第四章摘要和第三、五、六章交叉入口已同步；目录与编号未变。未修改 plan.md、PDF、题库或其他全局索引。
- **open_risks：** TCGBench 指标基于已收集错误提交，不能覆盖所有可能程序；ImpossibleBench 是人为构造的冲突任务；RLVR shortcut 结果仅来自特定归纳逻辑 benchmark/受控 7B 训练；RM toy 分数和 ECE 不是概率校准实验；示例不证明真实模型、沙箱或生产部署的安全性和准确率。

## 下一步

进入第五章《Process Supervision》，继续检查步骤标签、过程奖励模型、正确轨迹多样性、步骤粒度与搜索使用；第 1—4 章已完成，第 5—20 章和整册最终门禁待处理。
