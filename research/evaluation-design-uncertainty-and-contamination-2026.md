# 开放技术问题评测设计、不确定性与污染检测来源

核验日期：2026-10-29。

本记录只保存联网核验得到的技术事实、来源版本、阅读位置和证据边界；不记录章节审阅、正文修改或工作进度。

## 预期信息增益与主动学习

- **来源：** Kirsch and Gal, [Unifying Approaches in Active Learning and Active Sampling via Fisher Information and Information-Theoretic Quantities](https://arxiv.org/abs/2208.00549v2)，arXiv v2，2022-11-06。
- **阅读范围：** 摘要、§2 的信息量分类与 acquisition quantities、预期信息增益定义、§5 的近似方法概述。
- **支持事实：** 作者把主动学习/采样中多种 informativeness 目标联系到模型预测与参数之间的互信息、预期信息增益等信息论量，并研究 Fisher information、observed information 等近似。
- **证据边界：** 论文的正式量和近似依赖其贝叶斯/模型/数据假设。把“先选择能区分假设的实验”用于一般工程排障是概念迁移；`G×R/C` 一类收益/风险/成本排序不是论文定义，也不是 EIG 估计。

## 语言模型评测的协议敏感性与复现

- **来源：** Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782v3)，arXiv v3，2026-05-31。
- **阅读范围：** 摘要、§2.1–2.3、§3.1、§4 评估挑战与建议、附录 D 检查项。
- **支持事实：** 作者讨论 prompt、微小实现细节、benchmark 实现和报告不完整如何改变得分、导致比较不可复现，并提出固定与公开关键评测设置、实现及统计信息的建议。
- **证据边界：** 这篇论文汇总 lm-eval 团队经验和案例，不保证某个协议适用于所有任务；工具本身不能替使用者选择有效 benchmark 或保证标签、抽样与因果解释正确。

## 评测统计不确定性：深度强化学习范围

- **来源：** Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264v4)，NeurIPS 2021 / arXiv v4，2022-01-05。
- **阅读范围：** 摘要、§1、§4 建议与工具，尤其是分层 bootstrap 区间、跨任务和运行性能分布、IQM 的建议与限制。
- **支持事实：** 作者在 Atari 100k、ALE、Procgen 和 DeepMind Control Suite 等深度 RL 基准上显示，有限运行下只报告点估计会掩盖不确定性；建议报告区间估计、跨任务/运行分布和较稳健的聚合统计。
- **证据边界：** 这是深度 RL 任务和运行层级的统计研究，不直接验证 LLM 题目、prompt、生成种子或用户聚类的最佳重采样单位。LLM 评估需要按自身的配对题目、来源簇、运行随机性和目标估计量选方法。

## 原题与扰动题污染信号

- **来源：** Golchin and Surdeanu, [Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://aclanthology.org/2025.tacl-1.37/)，TACL 2025，13:809–830。
- **阅读范围：** ACL Anthology 摘要和论文 §1 引言中 DCQ 构造说明。
- **支持事实：** DCQ 对每个原始样本构造三个词级扰动版本并加入“均非原文”选项，形成选择题；作者把模型从选项中识别原文的行为作为特定数据分区的污染检测信号，并讨论位置偏差。
- **证据边界：** 该方法检测的是其协议下的行为信号，不提供模型训练文件的来源证明；扰动是否保留题意、难度和答案需要逐项检查，原题/变体性能差也可能由难度变化造成。未观察到信号不能证明未污染。

## 推理模型中检测信号的脆弱性

- **来源：** Wang et al., [On The Fragility of Benchmark Contamination Detection in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html)，ICLR 2026；arXiv v2 于 2026-03-02 更新并标记为已接收。
- **阅读范围：** ICLR 官方会议摘要；本次未据摘要外推未读的实现细节。
- **支持事实：** 作者报告，在其 base model 经 benchmark 污染 SFT 后的实验中，短程 GRPO 可显著隐藏多数检测方法使用的污染信号；对 advanced reasoning model 的特定 CoT 污染场景，多数检测器结果接近随机猜测。
- **证据边界：** 结论限于论文构造的模型、训练流程、污染方式和检测器，不证明所有 RL 或所有推理模型都会抹除污染信号。它说明“没有检测到”不能视作“没有污染”的一般充分证据。

## 成员推断假设与生成式评测边界

- **来源：** [Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions](https://aclanthology.org/2025.findings-naacl.291/)，Findings of NAACL 2025；[正式论文 PDF](https://aclanthology.org/2025.findings-naacl.291.pdf)。
- **阅读范围：** 摘要、引言、综述分类、直接 instance-level membership-inference 假设的实验方法和 §6 案例结果。
- **支持事实：** 论文综述 50 篇检测研究并归纳 8 类假设，随后实证检验 3 个直接成员推断假设。在论文评测的预训练数据集上，若干检测方法接近随机猜测；作者指出较高检测率可能受训练/测试领域分布差异影响。
- **证据边界：** 实验只检验其中三类直接样本级成员推断假设，不足以否定所有语料 overlap、benchmark-level 或行为检测方法。高分也不能未经控制就解释为成员信号；领域差异可能构成替代解释。

- **来源：** [BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22b4e30a7a660b17f1fb58ab49671e77-Abstract-Conference.html)，ICLR 2026；[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/22b4e30a7a660b17f1fb58ab49671e77-Paper-Conference.pdf)。
- **阅读范围：** 官方摘要、方法和任务构造、实验汇总、限制章节。
- **支持事实：** 论文定义 44 个数学/组合任务族和 117 种变化形式，生成空间超过 $10^{15}$ 个实例；采用确定性 verifier，并在三折设置中评测 101 个模型。
- **证据边界：** 任务仅覆盖作者实现的算法与数学结构，使用单一 prompt；生成器和 verifier 决定构念覆盖。它降低固定题面记忆风险，不代表一般知识、开放任务或真实产品工作流都具有同等污染抵抗力。

## OpenAI 托管 Evals 平台生命周期

- **来源：** OpenAI 官方 [Working with evals](https://developers.openai.com/api/docs/guides/evals) 与 [Deprecations](https://developers.openai.com/api/docs/deprecations)，访问日期 2026-10-30；网页未标固定文档版本。
- **阅读范围：** Evals 指南的弃用提示；下线表中 “2026-06-03: Evals platform” 条目。
- **支持事实：** 官方页面说明既有 Evals 将于 2026-10-31 转为只读，Evals dashboard 与 API 计划于 2026-11-30 关闭；相关 graders 在迁移范围内。指南建议新建或迭代评测时考虑 Datasets。
- **证据边界：** 这是访问日的托管平台时间表，可能变更；它不表示开源 `openai/evals` GitHub 仓库或评测方法本身关闭。依赖该托管服务的项目应在迁移时再次查看官方时间表。

- **Agent 工作流指南（2026-10-30）**：[Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)建议先用单条 trace grading 排查工具选择、handoff、instruction/safety policy 等工作流问题；“good”标准明确后再转为 datasets 和 eval runs 做重复比较。该指南属于动态产品文档，页面中的 dashboard、grader 和 Agent Builder 路径不能当作稳定 API 契约。

## 尚未解决的证据问题

- 工程诊断中的“实验区分度”如何转成可信的先验概率、结果似然和行动效用，需要目标任务自身的数据与决策模型；通用 `G×R/C` 分数没有校准依据。
- 等价改写的语义、难度和分布保持应由领域专家或经验证的流程检查；词级扰动、数字/实体替换和组合新题不是同一种干预。
- 污染检测方法依赖访问条件和可观测行为，异质信号不能合并成一个普适污染率；未检出、分数差和语料近重复分别支持不同命题。
