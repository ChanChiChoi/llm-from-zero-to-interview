# Reward Model 评估与过度优化：来源证据

检索日期：2026-10-29。

本记录只保存联网核验得到的知识点、来源版本和证据范围，不记录书籍修改过程、逐讲状态或工作交接。InstructGPT 与摘要反馈流程的来源范围另见 [`rlhf-preference-and-policy-optimization-2026.md`](rlhf-preference-and-policy-optimization-2026.md)。

## RewardBench 的数据覆盖与限制

- Lambert et al., “RewardBench: Evaluating Reward Models for Language Modeling,” [arXiv:2403.13787v2](https://arxiv.org/abs/2403.13787v2)，v2 于 2024-06-08 修订；论文 HTML：<https://arxiv.org/html/2403.13787>。
  - 实际阅读范围：摘要、§1 Introduction、§5.2 Limits of Current Reward Models、附录 A Limitations。
  - 来源支持：RewardBench 使用 prompt/chosen/rejected 样本，覆盖 chat、reasoning 和 safety；部分比较针对细微但可验证的错误（如 bug 或事实错误）。作者报告受测 reward models 在 refusal、reasoning 和 instruction following 等方面有不同表现。
  - 限制：作者指出，除特定子集外，chosen/rejected 样本主要通过半自动方式构造并人工复核；部分领域格式可能产生对人或模型的伪相关；benchmark 结果与下游训练效果的相关关系尚未解决；使用过 AlpacaEval 或 MT Bench 数据的模型可能存在污染风险。
  - 证据等级：source-backed（该版本评测集设计、受测类别及作者明确列出的限制）。
  - 适用边界：基准只能说明受测数据、模型和评分设置下的表现，不能单独证明 RM 对当前 policy rollout 的可靠性或下游训练收益。

## Reward Model 过度优化的合成研究

- Gao, Schulman, and Hilton, “Scaling Laws for Reward Model Overoptimization,” [arXiv:2210.10760v1](https://arxiv.org/abs/2210.10760)，2022-10-19；论文 HTML：<https://arxiv.org/html/2210.10760>。
  - 实际阅读范围：摘要、§1、§2.1 Synthetic Data Setup、§3.2–3.6 与 KL penalty 相关结果。
  - 来源支持：论文用固定的 6B reward model 作为合成 gold 信号，为较小 proxy reward models 生成偏好标签，再通过 RL 或 best-of-n 优化 proxy。作者报告，在该合成设定中 gold reward 对优化强度可呈先升后降的关系；函数形式依优化方法变化，结果也受 reward model 参数量、数据集大小、policy 尺寸和 KL 系数影响。
  - 限制：gold RM 代替人类判断；方法、数据、模型和训练超参数均特定于论文。作者关于 KL penalty 的观察在其 RL 配置下未显示 gold-reward/KL frontier 有可测改善，并指出结果可能对超参数敏感。因此它支持监测 proxy 与 gold 指标的分离，不支持通用优化阈值，也不证明 KL 是普适修复办法。
  - 证据等级：source-backed（合成环境的方法和论文报告的曲线/条件）。
  - 适用边界：论文不能单独证明所有真实人类偏好任务都存在同形曲线；gold RM 不是独立人类评审。

## 生命周期

RewardBench v2 的评估类别和限制：active；Gao 等人的实验结论：historical（方法论示例，适用于理解代理优化风险，不能外推为当前模型排名）。跨任务的 RM 校准、policy rollout 泛化和 KL 风险仍须按具体数据、标注协议、模型及独立任务 verifier 核验。
