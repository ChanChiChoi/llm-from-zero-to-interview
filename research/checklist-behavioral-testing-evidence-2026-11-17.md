# CheckList 行为测试来源记录

核验日期：2026-11-17。用途：为面试题中 MFT、INV、DIR 三类行为测试提供一手来源和术语边界。本文仅记录论文事实及由此引出的工程注意，不记录章节审阅进度。

## CheckList 论文

- Ribeiro et al., [Beyond Accuracy: Behavioral Testing of NLP Models with CheckList](https://aclanthology.org/2020.acl-main.442/), ACL 2020, pp. 4902–4912, DOI 10.18653/v1/2020.acl-main.442。
- 论文将 CheckList 描述为按能力和测试类型组织的行为测试方法，报告三类测试：Minimum Functionality Tests (MFT) 用简单用例检查特定行为；Invariance (INV) 对输入施加保留标签的扰动并期望预测不变；Directional Expectation (DIR) 施加预期模型输出沿特定方向变化的扰动。
- **边界：** 三类测试用于生成更系统的行为测试，不是正确性证明、安全认证或完备测试集。论文中的示例以 NLP 分类/问答任务为主；将它们迁移到 Agent 安全时，必须重新定义任务标签、动作状态和外部副作用观察。
- **工程注意（推论）：** 语义保持扰动应由人工或独立规则复核；同一源请求生成的多个变体共享来源，可在统计不确定性估计中按源请求聚类，避免把相关改写当作独立样本。此聚类建议是对评估样本相关性的工程处理，不是 CheckList 论文给出的统计保证。
