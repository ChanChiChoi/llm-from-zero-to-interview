# 大模型训练与后训练系统研究资料（2026）

检索与阅读日期：2026-10-30。本文只记录公开论文/技术报告中的训练机制和证据边界，不记录书册审阅进度或正文修改决定。

## 多模态预训练的阶段化课程

**来源**：[Qwen3-VL Technical Report](https://arxiv.org/html/2511.21631v1)，Qwen Team，arXiv v1，2025-11-26。阅读摘要、引言和 §3.1。

- 报告把预训练分为四段：S0 只训练 vision-language merger，约 67B token、8K 序列；S1 全模型约 1T token、8K；S2 全模型约 1T token、32K；S3 全模型约 100B token、262K。
- 报告将后训练描述为长 CoT SFT、较强教师蒸馏和 RL 三段；正文还报告区分 thinking 与 non-thinking 变体。
- 这是 Qwen3-VL 团队对其单一模型系列的自述训练方案。阶段长度、数据配比和效果不能直接当作其他多模态模型的通用最优配方，也不是独立复现实验。

## 优化器分组、batch scaling 与稳定性评估

**来源**：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/html/2608.30320v1)，arXiv:2608.30320，2026-08。阅读 §3.1–§3.3 中优化器、超参数缩放、稳定性压力测试及生产配置检查。

- 报告把 Muon 用在若干矩阵形状的投影和专家权重上，而将 embedding、输出头、MoE router 及特定低秩投影交给 AdamW；报告说明其 router 的早期波动是采用这一分组的原因之一。
- 在其 10.8B-A0.89B MoE、4T token 对照中，报告称按 scaling-law 预测从 12.6M token batch 调到 25.2M 优于旧设置；37.7M 的变化较小。该报告中的 batch-size ramp 没有优于固定目标 batch，并多用了 18.8% optimizer steps。
- 在另一个 156B-A7B、419B-token 对照中，作者报告预测值附近的若干 batch/LR 配置结果接近，训练曲线稳定。以上结论绑定该报告的模型、预算、优化器、数据顺序和实现，不构成 Muon、batch ramp 或 LR 的普遍规则；报告也提醒窄幅 benchmark 排序可能处于运行/测量噪声内。

## 多教师 on-policy distillation

**来源**：[MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/html/2606.30406v1)，arXiv:2606.30406，2026-06-29。阅读 §§1、3–5 和附录训练细节。

- 方法先从一个通用 SFT 初始化分别训练领域 RL teachers，再让 student 生成自己的 rollout，由对应冻结 teacher 在 student 访问到的状态提供逐 token 监督；论文重点是多领域能力整合，不是提出新的通用 RL objective。
- 论文在 Qwen3-30B-A3B 的数学、指令遵循和软件工程设置，以及作者报告的 MiMo-V2-Flash 案例中比较集成路线。归一化分数、模型、任务、数据与训练预算均属于论文设置；这些结果不能外推为任意 teacher 数量或任务组合的保证。

## 大规模 RL/OPD 的异步采样与状态恢复

**来源**：[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/html/2609.19969v1)，DeepSeek-AI，arXiv v1，2026-09-17。阅读 §§4.1–4.2、§5.1–§5.2。

- 报告称该模型使用 45T 多模态 token 预训练；稀疏注意力从头在 64K 序列长度训练，并在约 34T token 时把上下文扩到 1M。作者报告的这些训练设置不代表对其他架构的推荐配置。
- 后训练使用 SFT、RL 和 OPD 的既有组合。报告强调其变化集中在合成数据与交互环境：任务构造、参考解/奖励、过滤、去重和难度校准。
- 异步后训练设施以并发生成缓解 rollout 长尾；报告还描述了控制在途样本和等待条件以限制 policy lag、对过度陈旧 token 使用 loss mask，以及在 checkpoint 切换时逐 token 中断 rollout、保存 KV/routing 状态并从新 checkpoint 恢复。
- 最终全词表 OPD 阶段据报告使用 40 余个 teacher，并支持不同训练阶段或不同架构的教师。相关吞吐与能力结论是报告作者的训练系统披露，不等同于独立复现或通用生产保证。

## 近期 MOPD 改进方向：仍属早期预印本

**来源一**：[UP-MOPD: Update Projection in Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2610.08398v1)，arXiv v1，2026-10-06。阅读摘要、§§1–3 和附录 A。

- 该工作指出 AdamW 的动量、坐标自适应缩放和 weight decay 可能使“梯度层面不伤害各领域”的修正，在实际 optimizer displacement 上仍伤害某个领域；其方法对 optimizer 生成的候选位移作最小投影修正。
- 论文报告了医疗/通用两教师和数学、代码、指令遵循三教师设置；作者明确指出教师数量增加时的梯度计算成本和长期能力保持仍需研究。因此这些结果不支持把方法写成成熟通用标准。

**来源二**：[Nash-MOPD: Optimizing Task Trade-offs in Multi-Teacher On-Policy Distillation](https://arxiv.org/html/2610.26124v1)，arXiv v1，2026-10-19。阅读摘要、§§1–5、§6 及 limitations。

- 该工作根据各领域 teacher–student reverse-KL 差距及其近期变化估计剩余训练需求，再用局部 minimax 目标动态分配 prompt 采样比例。
- 论文报告的实验限于 DeepSeek-R1-Distill-Qwen-1.5B（三个领域）和 Qwen3-8B-Base（数学/代码）两种 student；作者把跨领域迁移、rate 不确定性、延迟反馈、更多模型规模和训练预算列为后续问题。
- 两篇均为近期预印本，适合作为 MOPD 主题的跟踪方向；单次论文对照不能替代跨实现、跨任务的复现或形成行业默认结论。
