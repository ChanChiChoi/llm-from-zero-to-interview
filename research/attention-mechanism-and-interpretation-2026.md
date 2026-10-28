# Attention 计算、实现与解释边界：事实和证据

检索日期：2026-10-28。

## Self-attention 的标准复杂度与并行性

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。阅读位置：§4。
- 支持事实：论文按标准序列层比较 self-attention、recurrent 和 convolutional 的单层复杂度、最少顺序运算数与最大路径长度；标准 self-attention 表格给出 $O(n^2 d)$ 每层复杂度、$O(1)$ 顺序运算数；recurrent 给出 $O(n d^2)$ 与 $O(n)$ 顺序运算数。
- 证据等级：原始论文对其所比较标准层形式的分析。
- 边界：该表不是任意现代架构、稀疏注意力或具体 fused kernel 的运行时间保证。正文的 T=4/T=8 位置对数量是按全连接/因果可见关系自行计算的教学例。

## Exact attention 与 GPU IO 优化

- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2)，Dao 等，arXiv v2（2022-06-23）。阅读位置：§2.2、§3.1–3.3。
- 支持事实：§2.2 描述标准 attention 实现形成并写回 $N\times N$ 的分数矩阵 $S$ 与概率矩阵 $P$；§3.1 的 FlashAttention 用 tiling 和 recomputation 计算 exact attention，减少 HBM 读写并避免保存这些大中间矩阵。论文另行提出 block-sparse 近似扩展。
- 证据等级：原始论文算法、IO 复杂度分析和特定 GPU 实验。
- 边界：FlashAttention 的 exact attention 仍有全连接注意力的二次位置关系；其收益是内存访问/中间状态方面，不能将其说成线性位置对数或通用任意 kernel 的行为。block-sparse 扩展是另一种近似注意力算法。

## Attention 权重与解释

- 来源：[Attention is not Explanation](https://arxiv.org/html/1902.10186v3)，Jain 与 Wallace，arXiv v3（2019-05-08）。阅读位置：摘要、§1、§4.1–4.2、§6。
- 支持事实：作者在文本分类、问答和自然语言推理设定中检验 attention weights 与梯度/留一特征重要性、反事实权重的关系；论文报告 BiLSTM 标准 attention 在这些研究中未能稳定通过相关检验，且某些替代分布产生近似输出。
- 证据等级：原始论文的研究问题、作者实验和结论。
- 边界：作者明确把主要发现限定于其 RNN-based 模型和具体任务；不应把它直接推广成所有 Transformer/LLM attention 权重都没有解释价值。

- 来源：[Attention is not not Explanation](https://arxiv.org/html/1908.04626v2)，Wiegreffe 与 Pinter，arXiv v2（2019-09-05）。阅读位置：摘要、§1–§5。
- 支持事实：作者指出注意力可否构成解释取决于解释定义、模型组成和测试设计，并提出 uniform baseline、随机种子方差、冻结权重诊断和模型一致的对抗测试；其 RNN 实验中提供了支持注意力在部分设定可具有解释价值的结果。
- 证据等级：原始论文的方法论论证与作者实验。
- 边界：此文同样主要研究 RNN-with-attention 与特定分类任务，不证明任一注意力热图都忠实解释 Transformer/LLM 的输出。两篇工作共同支持谨慎区分局部权重计算和模型整体解释。
