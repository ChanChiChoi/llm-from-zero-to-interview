# Attention 计算、实现与解释边界：事实和证据

检索日期：2026-10-28。

## Transformer attention 的定义、来源与结构边界

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。阅读位置：§§3.2.1–3.2.3、§3.5、§4。
- 支持事实：论文将查询与各 Key 的点积除以 $\sqrt{d_k}$，沿 Key 位置作 softmax，再对 Value 加权求和；多头版本使用不同的可学习线性投影并行计算，再拼接、输出投影。Self-attention 的 Q/K/V 来自同一序列表示；encoder-decoder attention 的 Query 来自 decoder，而 K/V 来自 encoder。Decoder 自注意力在 softmax 前将非法未来位置对应的分数设为负无穷。因为该模型没有 recurrence 或 convolution，论文在输入表示中加入位置信息。
- 证据等级：原始论文对算子的定义、架构说明和复杂度分析。
- 适用边界：这些描述对应论文提出的 Transformer 形式；不同模型可替换位置表示、投影、可见性模式或 attention kernel。论文关于 self-attention 与 recurrent/convolutional 层的比较见上方复杂度小节，仅支持表中定义的层形式。

## 早期序列到序列 attention 与双向编码器

- 来源：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473)，Bahdanau、Cho 与 Bengio，arXiv v1（2014）；[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/html/1810.04805)，Devlin 等，arXiv v2（2019-05-24）。阅读位置：Bahdanau 论文 §§2.1、3.1；BERT 摘要、引言和 §3。
- 支持事实：Bahdanau 等让 decoder 在生成每个目标词时，根据当前 decoder 状态对 source annotations 计算 soft alignment，并加权形成该步 context vector；这不同于把全句压成一个固定向量。BERT 使用 masked-language-modeling 预训练深层双向 encoder 表示，使其每层能结合左右上下文。
- 证据等级：原始论文中的模型定义和作者实验范围。
- 适用边界：Bahdanau 论文研究的是英语—法语神经机器翻译中的 RNN encoder-decoder 注意力；BERT 的双向表示与下游效果对应其模型和预训练目标。二者说明 attention 和可见性模式的具体历史实例，不构成对所有任务统一优劣的结论。

## Attention head 裁剪证据

- 来源：[Are Sixteen Heads Really Better than One?](https://arxiv.org/html/1905.10650)，Michel、Levy 与 Neubig，arXiv v3（2019-11-04）。阅读位置：摘要、§§3.2–3.3、§§4.2–5。
- 支持事实：在论文所测机器翻译模型与 BERT 自然语言推理模型中，许多训练后的 head 可分别在测试时裁剪而不显著影响该实验的结果；某些层可降为单头。跨全网共同裁剪时，大部分 head 仍须保留才能避免明显性能下降，且 encoder-decoder attention 比 self-attention 对裁剪更敏感。
- 证据等级：原始论文的消融与裁剪实验。
- 适用边界：这表明 head 重要性在论文模型、任务和裁剪方法中并不均匀；不能推出所有模型中的 head 都冗余，也不单独证明 attention weights 对最终预测没有解释价值。

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
