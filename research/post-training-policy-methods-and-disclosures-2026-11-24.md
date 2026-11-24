# 后训练策略方法与公开披露：原始来源证据

- **检索日期：** 2026-11-24
- **范围：** 后训练中的偏好优化、RLVR 方法以及厂商公开的训练信息。以下只记录来源支持的知识点、版本与证据边界。

## 方法论文

| 主题 | 原始来源与版本 | 来源支持的事实 | 证据边界 |
| --- | --- | --- | --- |
| GRPO | [DeepSeekMath，arXiv:2402.03300v3](https://arxiv.org/html/2402.03300) | GRPO 针对同一问题采样多条回答，以组内 reward 建立 baseline，省去 PPO 的单独 critic；论文目标按 token importance ratio 优化。 | 这是论文算法与数学任务实验的描述，不表示所有 GRPO 实现采用完全相同的 reward、KL 或 loss reduction。 |
| DPO | [DPO，arXiv:2305.18290v3](https://arxiv.org/html/2305.18290v3) | 在偏好模型假设下，DPO 直接以 policy 的分类目标学习偏好，避免单独拟合 reward model 并在训练步骤中运行在线 RL。 | “不训练单独 RM”不等于无需偏好标签；论文实验和理论依赖其偏好建模假设。 |
| SimPO | [SimPO，arXiv:2405.14734v3](https://arxiv.org/html/2405.14734) | 以回答长度归一化的平均 log probability 构造 reference-free 的隐式 reward，并在偏好目标中加入目标 reward margin。 | 长度归一化不等同于消除全部长度偏差；效果限于论文模型和基准。 |
| DAPO | [DAPO，arXiv:2503.14476v2](https://arxiv.org/html/2503.14476v2) | 论文将 Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping 组合为长推理 RL 配方。Qwen2.5-32B base 在 AIME24 Avg@32 的逐步实验中由 naive GRPO 的 30 到组合配方的 50。 | 该分数依赖特定模型、数据、采样、实现和评测设置；逐步消融不能把组合结果等同于每一组件的独立因果效应。 |
| DrGRPO | [Understanding R1-Zero-Like Training，arXiv:2503.20783v2](https://arxiv.org/abs/2503.20783v2) | 论文分析 GRPO 的响应长度优化偏差，并提出 DrGRPO，以固定生成长度分母和不使用组内 reward 标准差缩放来降低该偏差。 | 这些改动不消除 reward 误差、无组内差异或跨任务 reward 尺度问题。 |
| GSPO | [GSPO，arXiv:2507.18071v2](https://arxiv.org/abs/2507.18071v2) | 重要性比率基于序列似然，并在序列级执行 clipping、reward 和优化；长度归一化的 log ratio 等价于 token ratios 的几何平均。 | 序列聚合降低局部 token 变化的可见性；论文性能主张限于报告的模型与实验条件。 |
| Dynamic Clipping Policy Optimization（DCPO） | [DCPO，arXiv:2509.02333v2](https://arxiv.org/html/2509.02333v2) | 论文按旧策略 token 概率动态调整 clipping bounds；将当前组标准化优势和同 prompt 累计标准化优势分别加权平滑，再取绝对值较小者；OTM 对每条回答内的有效 token loss 求均值，并对回答值求和。论文在四个 Qwen 设置、400 个训练 step 的 RUR 汇总中报告平均 71.8%，GRPO 为 43.8%。 | RUR 是论文定义的非零 advantage 回答比例，不是正确率或有益梯度比例。以上分数来自论文设置，不能外推至其他任务。 |
| 同名 DCPO | [Distribution-Centric Policy Optimization，arXiv:2601.12730](https://arxiv.org/abs/2601.12730) | 该论文也使用缩写 DCPO，但全称、目标和方法不同，研究以分布级正则调节探索。 | 引用时需写出全称；不可与 Dynamic Clipping Policy Optimization 混为一谈。 |
| DHPO | [DHPO，Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1883/)；[对应 arXiv:2601.05607v1](https://arxiv.org/abs/2601.05607v1) | 已收录于 Findings of ACL 2026。它在一个 clipped surrogate 中混合 token-level 与 sequence-level ratio，采用平均或 entropy-guided mixing，并对两条分支使用各自 clipping；论文报告七个数学基准及 Qwen3 dense/MoE 实验。 | 论文中的对比结论限于所报告的数学数据、模型和训练条件。 |
| Group Adaptive Clipping Policy Optimization（GAPO） | [GAPO，EMNLP 2026 主会](https://aclanthology.org/2026.emnlp-main.1882/) | 该方法按 rollout advantage 自适应 clipping threshold，旨在给较强学习信号的 rollout 更多更新空间；论文在部分 Qwen、Llama 数学和代码任务报告 Pass@1/Pass@k 改善。 | 缩写 GAPO 被不同方法复用，例如 [Group Adaptive Policy Optimization，Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.875/) 与 [Group-Aware Policy Optimization，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1649/)。仅写缩写会造成歧义。 |
| MOPD | [MOPD，arXiv:2606.30406v1](https://arxiv.org/abs/2606.30406v1) | 提出的路线先训练多个领域 RL teacher，再让学生在自己的 rollout 上接受这些 teacher 的蒸馏；论文报告 Qwen3-30B-A3B 设置，并称已用于 MiMo-V2-Flash 后训练。 | 该报告不能据此推出一般场景的训练成本、收益或可复现程度。 |

## 厂商公开的模型训练信息

| 模型与来源 | 发布方公开说明 | 未披露或不能由该说明单独推出 |
| --- | --- | --- |
| [GLM-5.3，Z.ai，2026-08-14](https://z.ai/blog/glm-5.3) | 发布方称其沿用 GLM-5.2 base model，并扩大长周期任务环境、任务多样性与 post-training 计算。 | 完整数据、优化器、reward 配方、超参数及独立复现结果。 |
| [GLM-5.4，Z.ai，2026-11-23](https://z.ai/blog/glm-5.4) | 发布方称以预训练 base 继续训练，将其改造为 Loop Transformer；也披露了 30B 同参数量对照。 | 不能据此推断偏好/RL 的具体配方，也不能把全部产品表现归因于某一训练阶段。 |
| [gpt-oss，OpenAI，2025-08-05](https://openai.com/index/introducing-gpt-oss/) | 发布方称模型经过 SFT 和高计算量 RL 后训练，并使用来自内部先进模型的训练技术。 | 发布页没有给出 PPO/GRPO 选择、完整 reward 设计或所有超参数。 |
| [Grok 4.7，xAI，2026-09-21](https://x.ai/news/grok-4-7) | 发布方称使用更大的 base model，并以更长 RL run 和更难、偏长时任务的混合数据训练。 | 不能由产品评测单独分离 base model、任务分布和 RL 各自贡献。 |

这些厂商页面是发布方披露，不构成独立控制实验。所列发布日期与来源版本用于核对截至检索日可查到的公开信息。
