# Human Preference Evaluation and Rater Protocols

**核验日期：** 2026-11-03  
**范围：** 人类偏好评估的目标人群、语言本地化、LLM judge 的反馈协议，以及 Bradley--Terry 排名的最坏情况敏感性。下列证据来自所列论文的指定版本与实验，不构成跨产品的一般效果承诺。

## HUMAINE：目标人群与偏好异质性

- **来源与版本：** Nora Petrova、Andrew Gordon、Enzo Blindow，*Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework*，ICLR 2026 正式论文；[会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9c1bcfe6edf7e47d83d2e99492d851e4-Abstract-Conference.html)、[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/9c1bcfe6edf7e47d83d2e99492d851e4-Paper-Conference.pdf)。
- **阅读范围：** 摘要、方法第 3.1—3.4 节、结果第 4.2—4.4 节和局限第 5.1 节。
- **来源支持的事实：** 研究招募 23,404 名美国和英国参与者，按人口组组织与 28 个模型进行的短多轮对话；收集 119,890 条多维人工判断。作者以分层 Bayesian Bradley--Terry--Davidson 模型汇总 pairwise 结果，并按美英人口普查数据后分层。作者报告模型排名在年龄组之间存在变化；Trust, Ethics & Safety 维度的平局比例为 65%，高于 Overall Winner 维度的 10%。
- **适用边界：** 地理范围限于美国和英国；人口分层只覆盖研究定义的年龄、族裔和政治维度；模型配置是论文快照；对话较短、开放式任务复杂度未控制，且是文本交互。年龄组差异是该样本中的观察结果，不是年龄导致偏好的因果结论。65% 平局表明此设计下该维度难以区分，不是安全评估一般都应有的平局率。
- **证据等级：** ICLR 2026 同行评审研究，支持一项具体的人群分层、偏好汇总和任务设计案例。

## MENLO：多语质量维度与本地化标注

- **来源与版本：** Chenxi Whitehouse 等，*MENLO: From Preferences to Proficiency - Evaluating and Modeling Native-like Quality Across 47 Languages*，arXiv:2509.26601，2025 年预印本；[论文](https://arxiv.org/abs/2509.26601)。
- **阅读范围：** 摘要、引言、数据集设计、标注流程与数据统计，以及语言变体附录。
- **来源支持的事实：** 数据集有 6,423 组回答比较和 81,014 条人工标注，涵盖 47 种语言变体；四类维度为 Fluency、Tone、Localized Tone 和 Localized Factuality。提示模板经翻译与本地化，评审员按具体语言变体选取；每组回答至少由三人标注，作者报告总体 Krippendorff’s alpha 为 0.84。
- **适用边界：** 研究测量的是本地语言质量，其样本、四类 rubric、模板和评审流程由作者特定设计；平均一致性不意味着各语言相同，也不能证明跨语言分数可直接比较。该来源为预印本，headline 结果不应外推至一般质量、事实性或高风险任务。
- **证据等级：** 预印本与公开数据集报告，支持特定多语标注方案的设计与报告结果。

## Pairwise 与 pointwise：LLM judge 协议效应

- **来源与版本：** Tuhina Tripathi 等，*Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation*，arXiv:2504.14716，2025 年预印本；[论文](https://arxiv.org/abs/2504.14716)。
- **阅读范围：** 实验设置、受控 IFEval-TweakSet 与 MT-Bench 人工判断切分上的结果、以及局限附录。
- **来源支持的事实：** 研究比较 LLM judge 的 pairwise 偏好与逐条绝对评分；测试的干扰包括 assertiveness、prolixity 和 sycophancy，主要评估目标是 instruction following。作者报告在其 MT-Bench 设置中引入风格干扰后的 pairwise 偏好翻转平均约 35%，绝对评分约 9%；受控 IFEval-TweakSet 中也观察到 pairwise 对同质质量答案较少选择 tie。
- **适用边界：** 这是对 LLM evaluators 的实验，不是人类评审的因果结论；数据集中包含作者构造的 IFEval-TweakSet，实验不覆盖 GPT 系列，干扰特征与目标能力范围有限。不得用此结果断言所有 pairwise 人评都比 pointwise 脆弱。
- **证据等级：** 预印本，支持所测 LLM judge、干扰项和数据上的协议敏感性观察；与人工评估设计的关系是候选问题，不是直接替代证据。

## Bradley--Terry 排名的 worst-case 数据敏感性

- **来源与版本：** Jenny Huang 等，*Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings*，ICLR 2026 正式论文；[会议页面](https://proceedings.iclr.cc/paper_files/paper/2026/hash/92d777fb721936613b6444b07c4ffc18-Abstract-Conference.html)、[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/92d777fb721936613b6444b07c4ffc18-Paper-Conference.pdf)。
- **阅读范围：** 方法、Chatbot Arena 结果、bootstrap 区间排名附录与数据说明。
- **来源支持的事实：** 作者以影响函数近似筛选高影响 preference，再精确重拟合 Bradley--Terry 模型核验。其固定的 Chatbot Arena 人类偏好数据快照中，最坏情况移除 2/57,477 条记录足以翻转点估计前两名；将 bootstrap 区间重叠视为并列时，最坏情况移除 29/57,477 条可改变 top-one 集合。
- **适用边界：** 这是针对特定固定数据集寻找最坏情况删除集合的敏感性分析，不是随机删除概率、常规置信区间、记录错误率或恶意投票证据。AMIP 用于候选筛选，论文随后精确重拟合验证。
- **证据等级：** ICLR 2026 同行评审研究，支持公开快照中的排名敏感性审计范例。
