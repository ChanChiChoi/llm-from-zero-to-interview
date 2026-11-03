# Benchmark 生命周期、评测器偏差与 Agent Harness 评估来源

核验日期：2026-11-02。本记录只保存联网核验到的知识点、来源版本、阅读范围和证据边界；书册定位、候选去向和修改进度见 `progress/book-07-evaluation-experiments/architecture-audit.md`。

## Benchmark 饱和与生命周期

- **来源与版本：** Akhtar et al., [When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](https://arxiv.org/abs/2602.16763)，arXiv v4，2026-08-06。
- **阅读范围：** 摘要、§2 饱和定义与测量、§3 数据集选择、§4 的多因素分析、§5 保护措施和生命周期建议、§6 局限。
- **来源支持的事实：** 作者按 14 项属性分析 60 个语言模型 benchmark，报告近半数表现出饱和，饱和率随 benchmark 年龄增加；在该样本和分析中，专家策划与较低饱和相关，而仅将测试集设为私有没有显示同等保护作用。作者讨论扩大测试集、报告不确定性、细分能力和生命周期更新等做法。
- **证据边界：** 这是基于公开排行榜与作者定义的饱和指数的观察性研究，不证明年龄或策划方式具有因果效应，也不代表所有 benchmark。论文指出其不确定性方法适用于 accuracy 类固定测试集；Elo、pass@k 和 judge 评估需要各自适配的方差估计。私人测试集可能仍降低直接泄漏风险，不能把“在此样本中未降低饱和”写成“保密无用”。

## LLM Judge 的位置偏差

- **来源与版本：** Shi et al., [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18/)，IJCNLP-AACL 2025，页 292–314。
- **阅读范围：** ACL Anthology 官方论文摘要与出版信息。
- **来源支持的事实：** 论文在 pairwise 和 list-wise 比较中提出 repetition stability、position consistency、preference fairness 三项度量；实验覆盖 15 个 judge、MTBench 和 DevBench 的 22 个任务、约 40 个答案生成模型和超过 150,000 个评判实例。作者报告位置偏差会因 judge、候选答案和任务而异，并受到答案质量差距影响。
- **证据边界：** 这是探索性研究，覆盖指定 judge、数据和协议；摘要不足以推出所有 judge 均有同等偏差，也不证明交换顺序或多数投票就能消除偏差。具体评估仍须按目标任务检查重测稳定性、方向偏好和人工校准。

## Agent Harness 与完整工作流评测

- **来源与版本：** Yao et al., [Harness-Bench: Evaluating Agent Harnesses across Models and Realistic Workflows](https://arxiv.org/abs/2605.27922)，arXiv v2，2026-10-15。
- **阅读范围：** 摘要、§3 任务与环境构造、§4 模型/harness 设置和主要结果、§5 分析、附录 A 局限与附录 D 轨迹标注。
- **来源支持的事实：** v2 包含 300 个跨六类工作的任务、1,408 个交付要求评分模块；274 个通用任务与 26 个多模态任务分别使用 10 个和 7 个模型后端。比较 7 个原生 harness，固定任务输入与初始环境，并由 oracle 检查最终 artifact；轨迹用于行为和资源分析。论文报告模型与 harness 的组合会改变交付质量、完整成功率和成本排序。
- **证据边界：** 附录说明每个任务—模型—harness 配置只运行一次，研究覆盖特定 7 个 harness、所测版本及所选 300 个任务，不能证明真实业务中的普遍排名或波动范围。最终 artifact 的模块分数和轨迹描述的是完整配置；若没有额外的随机化、消融或机制验证，不应据此把效果归因到单一 harness 功能或裸模型。

- **来源与版本：** [Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution](https://arxiv.org/abs/2610.32285)，arXiv v1，2026-10-23。
- **阅读范围：** 摘要、§3 两层评估与轨迹/证据方法、失败归因的手工验证规则、§4 设置和 §5 结果概述。
- **来源支持的事实：** TraceTrial-Bench 将 harness reliability 分为 context、execution、governance、lifecycle、observability、tools、verification 七类；用 95 个机制任务比较 4 个 harness 与 9 个模型，共计 3,420 次运行。其框架把 Agent 任务表现与 harness 标准分开计分，并将候选失败机制与经过人工核验的传播链分开报告。摘要报告宏观 harness 分数为 59.8%—72.0%，且 31.1% 的运行完成任务但至少违反一项 harness 标准。
- **证据边界：** 这是近期预印本和特定任务/系统配置的结果。论文将候选 link 和人工核验的因果传播分开；只有经端点、事件顺序、可观测中间变化和竞争解释复核的 link 才支持其定义下的传播结论。不能把候选轨迹关联直接当成普遍根因，judge 辅助标注也不替代独立人工核验。

## 长上下文任务的表面匹配与有效长度

- **来源与版本：** Modarressi et al., [NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167)，arXiv v3，2025-07-09。
- **阅读范围：** 摘要、§3 任务构造、§4 实验指标和主要结果、附录 E 延伸设置。
- **来源支持的事实：** NoLiMa 调整 needle-in-a-haystack 问题与材料间的词面重合，使样本需要隐式关联；作者在其 13 个模型的实验中报告，随长度增加成绩下降，32K 时 11 个模型低于各自短上下文基线的一半。论文用相对短上下文基线的阈值定义 effective length，并在部分模型上扩展到 64K/128K。
- **证据边界：** 这些结果绑定论文的 needle、haystack、prompt、模型版本和长度设置；其有效长度是该基准与阈值下的指标，不能当成某模型的一般上下文上限。词面去重也只处理特定捷径，不等于测到所有长文档任务。

## 动态 benchmark 设计实例

- **来源与版本：** Stanford Digital Economy Lab, [Agent Island: A Saturation- and Contamination-Resistant Benchmark from Multiagent Games](https://digitaleconomy.stanford.edu/publication/agent-island-a-saturation-and-contamination-resistant-benchmark-from-multiagent-games/)，工作论文页面，2026-05-05。
- **阅读范围：** Stanford 作者机构页面摘要。
- **来源支持的事实：** 页面介绍一个多智能体合作、冲突和说服的动态博弈环境，并使用 Bayesian Plackett–Luce 模型估计玩家相对技能与不确定性。
- **证据边界：** 页面将其标为工作论文；摘要描述的是一个特定游戏环境和排名方案，不能证明动态博弈能普遍解决污染或饱和，也不能替代具有外部效度的专业任务评估。
