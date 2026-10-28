# Transformer Block 与逐位置前馈层：事实和证据

检索日期：2026-10-28。

## 原始 Transformer 的层结构

- Vaswani 等人的 Transformer 原论文 v7（2023-08-02）§3.1 描述 encoder 层由 multi-head self-attention 与逐位置 feed-forward 子层组成；每个子层外有 residual connection，之后执行 LayerNorm。论文指出各子层输出维度保持为 d_model=512，以支持残差相加。Decoder 在这两类子层外增加 encoder-decoder attention，并对 decoder self-attention 遮罩未来位置。来源：[论文 HTML §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1)。证据等级：原始论文中的架构定义。边界：这是原论文 encoder-decoder Transformer 的设计，不能当成所有 decoder-only 模型的逐项规格。
- 同一论文 §3.3 将 feed-forward network 定义为逐位置、各位置共享参数的两层线性变换，中间使用 ReLU；论文配置为 d_model=512、d_ff=2048。来源：[论文 HTML §3.3](https://arxiv.org/html/1706.03762v7#S3.SS3)。证据等级：原始论文中的方法定义与配置。边界：512/2048 和 ReLU 是该论文配置，不代表现代 Transformer 的固定宽度比例或激活函数。
- 同一论文 §3.2.3 说明 decoder 的 attention 组合包括 masked self-attention 和读取 encoder 输出的 encoder-decoder attention；后者的 Query 来自 decoder，Key/Value 来自 encoder 输出。来源：[论文 HTML §3.2.3](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS3)。证据等级：原始论文的结构描述。边界：decoder-only 模型没有独立 encoder 输出时，不应据此假定存在该 Cross-Attention 子层。

## Pre-LN 与 Post-LN

- Xiong 等人的论文 v2（2020-06-29，发表于 ICML 2020）摘要说明其 mean-field 分析比较了 Post-LN 与 Pre-LN：在该论文研究的原始 Post-LN 架构中，初始化时靠近输出层的预期梯度较大；其所分析的 Pre-LN 架构初始化梯度较平稳。作者还报告了特定应用实验中 Pre-LN 可在不使用 warm-up 时达到与 baseline 可比的结果。来源：[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745v2)。证据等级：原始论文摘要中的理论分析与作者报告的实验结论。边界：结论限于论文的架构、分析假设与实验设置；不能外推为所有模型均无需或必须使用 warm-up。该证据也不说明 Pre-LN 与 Post-LN 只在数值公式上有无关紧要的差异。

## 门控前馈变体

- Shazeer 的 GLU Variants Improve Transformer（2020，arXiv v1）摘要将 GLU 描述为两个线性投影的逐元素乘积，其中一个投影先经 sigmoid；其变体研究在 Transformer 序列到序列模型的 feed-forward 子层中测试不同门控激活，并报告部分变体相对于该实验中的 ReLU/GELU 有质量提升。来源：[论文摘要](https://arxiv.org/abs/2002.05202)。证据等级：原始论文摘要中的方法和作者实验报告。边界：支持门控前馈是标准 ReLU/GELU 前馈的一个结构变化，不证明所有模型、任务或参数预算都能提升质量或降低资源成本。

## 未验证边界

- 未运行或 profile 任何真实 Transformer 实现；未核对具体模型的 Norm 顺序、激活函数、门控宽度、bias、dropout、Cross-Attention 路径和残差形状。
- 文中的两位置向量、attention 更新、MLP 权重与参数账本均为自行构造的教学输入；不是论文实验或模型测量。
