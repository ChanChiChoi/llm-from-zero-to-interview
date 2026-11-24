# 训练总览章节：规模估算与 checkpoint 证据

检索日期：2026-11-24。以下来源用于核对第五册第一章中的通用事实和适用范围；其中公开报告的模型结论按报告自身设置理解，不扩展为其他项目的保证。

## 参数、token 与计算规模

- Kaplan 等人的研究拟合语言模型验证交叉熵随参数、数据与计算变化的经验关系，适用范围是论文模型和训练设定：[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)。
- Chinchilla 研究报告训练了 400 多个约 70M 至 16B 参数模型，并用 70B 参数、1.4T token 的模型对其计算最优预测做模型级验证。该比例不应脱离模型、语料和优化设定作为跨项目口诀：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)。
- DeepSeek LLM 报告用 $C\\approx6ND$ 作为预训练计算量的近似口径；此式适合数量级估算，不能替代具体架构、序列长度、重计算、通信和硬件实测：[DeepSeek LLM](https://arxiv.org/abs/2401.02954)。
- 数据受限训练研究考察固定数据量下重复训练的边际收益：[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)。
- inference-aware scaling 研究在给定训练和推理需求下调整模型参数与预训练数据规模，并指出高推理需求会改变其成本最优方案：[Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)。
- Llama 3 报告称其旗舰模型约为本项目预算下的 compute-optimal 规模，较小模型训练时间超过该预算下的 compute-optimal 点，以在相同推理预算下取得更好表现。这是厂商报告中的特定规划案例：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)。

## 评估指标与数据质量

- ICLR 2026 的生成式评测 scaling 研究比较了以计算量、参数与训练 token、参考解 log-likelihood 为协变量的拟合方式，并报告 pass-at-$k$ 会影响拟合参数与预测稳定性。结论绑定论文任务、采样数和验证器：[Pretraining Scaling Laws for Generative Evaluations of Language Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html)。
- ICLR 2026 的数据质量研究使用 23 个 QualityPajama 数据集、100M–8B 参数模型和 100M–200B 训练 token，报告去重、启发式过滤与 LLM 改写等干预会改变拟合的 scaling law 参数。结果适用于论文语料与干预范围，不构成通用质量分数：[How Text Quality Interventions Reshape Neural Scaling Laws for LLMs](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)。
- 2025 年 scaling-law 综述检查 51 项研究，并指出方程、变量、训练设置、评估和拟合方法常未充分报告；据此检查预测时应先核对口径：[ (Mis)Fitting: A Survey of Scaling Laws](https://arxiv.org/abs/2502.18969)。

## Checkpoint 与训练阶段

- PyTorch 当前教程区分只用于推理的模型参数保存与一般训练 checkpoint；继续训练的 checkpoint 还应保存 optimizer state 和其他恢复所需状态。教程示例是单机 API 说明，分布式训练恢复仍由框架和作业语义决定：[Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models)。
- InstructGPT 报告将示范监督、偏好奖励模型与强化学习串成后训练流程：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)。
- DPO 论文定义了直接使用偏好对与参考策略进行策略优化的目标：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)。
- DeepSeek-R1 报告是推理强化训练与蒸馏的一个公开案例；其做法不能视为所有 reasoning model 的通用配方：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)。
