# 实验、事故复盘与污染检测来源记录

核验日期：2026-11-17。用途：支持第十九册第十四章的 DORA 交付指标、NIST AI RMF 版本状态与评估集污染检测边界。本文只记来源事实、版本和证据限制；章节审阅进度及修改记录保存在 `progress/`。

## Google SRE 事件响应

- [Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/) 及 [Managing Incidents](https://sre.google/sre-book/managing-incidents/)：用于核对角色、沟通、影响控制和事件协调的实践入口。
- [Workbook: Incident Response](https://sre.google/workbook/incident-response/) 与 [Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)：用于事件响应演练和监控语境。
- **边界：** 这些是 SRE 实践资料，不定义生成模型任务正确率、RAG 证据分数、工具授权或法律适用结论。第十四章将其用于事件组织和复盘实践，不视为对所有团队强制的固定流程。

## DORA 软件交付指标

- [DORA 软件交付绩效指标](https://dora.dev/guides/dora-metrics/) 页面截至核验时列出五项：change lead time、deployment frequency、failed deployment recovery time、change fail rate、deployment rework rate。页面将变更失败率描述为部署后需要立即干预的部署比例，将失败部署恢复时间限定为恢复失败部署的时间；并建议按单个应用或服务、结合情境解释，避免背景不同的应用间机械比较。
- [DORA 指标历史](https://dora.dev/insights/dora-metrics-history/) 记述口径演变：MTTR 在 2023 年被重命名并收窄为 failed deployment recovery time；2024 年加入 deployment rework rate，形成当前五指标。
- **版本和边界：** 两个页面均标注最后更新于 2026-01。指标用于交付过程观察，不替代任务质量、模型正确性、用户价值或安全控制。不同团队须记录服务范围、部署单位、观察窗口、分类规则和指标版本；旧版 MTTR 口径不能无说明地并入当前口径。

## NIST AI RMF

- [NIST AI RMF 官方页面](https://www.nist.gov/itl/ai-risk-management-framework) 截至 2026-11-17 显示 AI RMF 1.0 正在修订，并说明其为自愿使用框架；页面记载 AI RMF 1.0 于 2023-01-26 发布、NIST AI 600-1 Generative AI Profile 于 2024-07-26 发布。
- [NIST AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/) 将建议动作对应到 Govern、Map、Measure、Manage；页面说明它既非必须完整照做的 checklist，也不是强制步骤。
- **边界：** 框架与 Playbook 可用于治理、测量、管理结构，不构成法律合规证明或事故因果证据。状态按核验日期固定；后续复用章节时应重新查官方版本。

## 基准污染检测

- Golchin 与 Surdeanu，[Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://aclanthology.org/2025.tacl-1.37.pdf), *TACL* 13 (2025), pp. 809–830：以原始样本和词级扰动构造选择题，使用模型选择原文的记忆信号估计检测到的逐字污染；研究指出需考虑选项位置偏差。
- Yang et al., [Rethinking Benchmark and Contamination for Language Models with Rephrased Samples](https://arxiv.org/abs/2311.04850), arXiv v2 (2023-11-11)：在 MMLU、GSM8K、HumanEval 等实验设置下报告释义或翻译样本可绕过字符串匹配，并提出更强的 LLM-based decontamination 方法。
- Wang et al., [On The Fragility of Benchmark Contamination Detection in Reasoning Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a3860475ddbfb8c644c43f0dcd266b66-Abstract-Conference.html), ICLR 2026：论文报告其研究的特定 SFT/GRPO 和 CoT 污染情境会削弱若干现有检测方法信号；结果基于所述 reasoning model 训练设置。
- **共同边界：** 这些工作测量特定检测器、模型和样本构造下的信号，不是对训练语料的完整审计。阴性检测结果不能证明没有语义重叠或污染；阳性检测信号也不单独证明具体数据来源、发生时间或责任归属。不能把论文中的结果外推为所有模型或训练管线的普遍规律。
