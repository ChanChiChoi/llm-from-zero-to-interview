# On-policy distillation 新论文：2026-11

本记录只摘录近期论文中的方法、实验与证据边界。数值是作者在论文设定下报告的结果，未在本地复现，也不构成跨论文排名或通用训练建议。

## Bernoulli On-Policy Distillation

- **来源与版本：** Cao et al., [“Bernoulli On-Policy Distillation: Don’t Ignore High-Confidence Student Tokens”](https://arxiv.org/abs/2611.02093v1)，arXiv v1，2026-11-01。访问日期 2026-11-17；阅读摘要、§§3–5、主结果表和 Appendix D。
- **来源支持：** 论文指出 vanilla sampled reverse-KL 的 sampled-token 梯度含有额外的学生置信度抑制项；Bernoulli OPD 用 Bernoulli cross-entropy 表示已采样 token 的教师/学生概率差。主实验的 Qwen3-4B-Instruct-2507 教师 → Qwen3-1.7B 学生设置中，五项数学基准汇总从 vanilla OPD 的 44.38 提高到 51.76；其余三种师生配置的增益从 0.98 到 2.08 个百分点，主设置增益为 7.38。前四项指标按 Avg@32 报告，MATH500 按 Avg@4。
- **证据等级：** source-backed（单篇预印本的算法推导与作者实验）。
- **边界：** 任务集中在数学推理，结果随教师配置显著变化；论文不能证明修正梯度分配会让任意教师信号更有用，也没有形成跨团队复现。准确率数字绑定论文的学生初始化、提示、生成模式、训练和评估协议。

## Learning Dynamics of On-policy Distillation

- **来源与版本：** Hong et al., [“Learning Dynamics of On-policy Distillation”](https://arxiv.org/abs/2611.08016v1)，arXiv v1，2026-11-05。访问日期 2026-11-17；阅读摘要、§§2–4、Conclusion 及相关机制分析。
- **来源支持：** 作者把 OPD 反馈环拆成两环节：**conversion** 描述教师监督如何经学生参数化与优化器变成 policy update；**broadcast** 描述更新如何影响其他前缀的预测、后续 rollout 和监督来源。分析指出，较大的 teacher-student disagreement 不保证产生有效更新；局部 teacher agreement 或 loss 改善也不保证终局解题能力改善。论文还报告，离线准备有时可改善随后 OPD 的表现，即使准备阶段本身未明显提高即时 held-out accuracy。
- **证据等级：** source-backed（预印本的机制分析和作者实验）。
- **边界：** 作者明确将实证范围限制在少数模型和数学推理任务。conversion/broadcast 是解释和诊断框架，不是可直接套用的优化算法、统一指标或性能保证；离线准备结果也绑定论文的教师、数据和训练设置。

## Bootstrapped On-Policy Distillation

- **来源与版本：** Wu et al., [“Bootstrapped On-Policy Distillation”](https://arxiv.org/abs/2611.13680v1)，arXiv v1，2026-11-10。访问日期 2026-11-17；阅读摘要、§§3–5、主结果与消融。
- **来源支持：** BOPD 分两阶段运行。Stage I 使用参考答案条件下的 OPSD，在学生自己生成的轨迹上训练一个任务适配中间模型；Stage II 冻结该模型作为教师，把学生重置为初始模型，从不带参考答案的状态上继续 OPD。作者的解释是，中间教师已把特权参考信息吸收进自身模型，因而更适合指导 reference-free 学生。
- **作者报告的结果：** Qwen3 4B/8B 数学三基准 Avg@12 mean 分别从初始模型的 18.06%/18.70% 提高到 23.98%/22.04%；四项代码基准 Pass@1 mean 分别从 44.42%/48.73% 提高到 46.54%/48.82%。论文也评估 SmolLM3-3B，并报告更多能力保持与消融分析。
- **证据等级：** source-backed（单篇预印本的算法定义与指定模型、数据、训练预算下的作者实验）。
- **边界：** Stage I 仍需参考答案和额外训练，BOPD 是两阶段成本；模型、数学/代码数据、rollout 和评测都受论文配置约束。报告结果尚未证明其他领域、不同规模或不同参考质量下也有效，不能仅以两阶段分数变化推断中间教师一定“更匹配”。
