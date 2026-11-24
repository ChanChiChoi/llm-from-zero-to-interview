# 大模型架构近期报告中的长上下文与稳定性对照

检索与复核日期：2026-11-24。本文只记录两份 2026 年公开技术报告中与训练前架构决策有关的具体设置；报告数字用于说明如何读取单项实验，不构成其他模型的配方或收益承诺。

## Qwen3.8-Next 的门控稳定性压力测试

- 来源：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/html/2608.30320)，arXiv:2608.30320v1，2026-08-31，§3.3，Figure 12。
- 报告用 28 层 25B-A3B MoE 做稳定性压力测试。门控单变量对照将学习率固定在各自最优值的 3 倍，保持 AdamW、结构和数据顺序不变，只切换 GatedNorm。开启门控后，loss spike 从每 10,000 步 32.0 次降至 3.2 次；超过梯度裁剪阈值的累计次数从 256 降至 20。该压力测试的全局梯度裁剪阈值为 0.5。
- 这组数据是高学习率压力测试下的门控消融，不能与 Qwen3.8-Flash-Next 完整生产 recipe 的表现混为一谈，也不能推导其他架构的默认门控、学习率或稳定性收益。

## DeepSeek-V4.1-Flash 的长度课程

- 来源：[DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/html/2609.19969)，arXiv:2609.19969v1，2026-09-17，§4.2.2。
- 报告称该模型从头以 64K 序列长度训练稀疏注意力，并在训练数据达到 34T tokens 时将序列长度扩至 1M。该模型最长上下文支持和训练长度变化是两项不同信息，描述课程时应保留阶段节点。
- 这是 DeepSeek-V4.1-Flash 的特定训练配方，不能证明其他架构在相同长度安排下能稳定获得同等长上下文能力；效果还取决于稀疏注意力实现、数据、训练量与评测。
