# Attention 计算、实现与解释边界：事实和证据

检索日期：2026-10-28。

以下 Transformer attention、早期序列到序列 attention、双向 encoder 与 head 裁剪条目于 2026-11-19 补充核验。

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

## Multi-head 投影、head 角色与剪枝证据

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。阅读位置：§3.2.2、§4.2。
- 支持事实：每个 head 对完整的 query/key/value 输入分别应用独立的可学习线性投影，输出拼接后再过输出投影；论文将其设计动机表述为让模型联合利用不同位置、不同表示子空间的信息。其 WMT 英德实验在近似固定计算量下改变 head 数及 key/value 维度；论文报告单头配置比最佳配置低 0.9 BLEU，head 数过多时质量也会下降。
- 证据等级：原始论文中的结构定义和指定机器翻译配置的消融。
- 适用边界：不同投影并不证明子空间彼此正交或对应固定语义功能；实验结果只适用于该论文模型、数据、训练设定和指标，不能推出通用最佳 head 数。

- 来源：[What Does BERT Look at? An Analysis of BERT's Attention](https://aclanthology.org/W19-4828/)，Clark 等，BlackboxNLP 2019。阅读位置：摘要和论文中的 BERT head 分析。
- 支持事实：研究在 BERT 中观察到若干 head 呈现分隔符、位置偏移和广域关注等模式；部分注意力分布与句法依存或共指关系相关。
- 证据等级：针对 BERT 模型的注意力模式分析。
- 适用边界：观测到模式或相关性不等于证明该 head 对任务必需、具有单一语义功能或在其他模型中位置相同。

- 来源：[Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/html/1905.09418v2)，Voita 等，arXiv v2（2019-06-07）。阅读位置：摘要、§§3–6。
- 支持事实：在英俄、英德、英法机器翻译编码器以及英俄 OpenSubtitles 域中，作者用重要性分析和带可微 $L_0$ 门的剪枝观察位置、句法关系、低频词等 head pattern；其剪枝方法会优先保留部分较专门的 head。
- 证据等级：论文指定翻译模型、数据域和门控剪枝方法下的原始实验。
- 适用边界：观察到的模式和保留顺序依赖任务、数据域、模型及其重要性/剪枝准则，不能当作所有 Transformer 的固定 head 角色清单。

- 来源：[Are Sixteen Heads Really Better than One?](https://arxiv.org/html/1905.10650v3)，Michel 等，arXiv v3（2019-11-04）。阅读位置：摘要、§§3.2–4.2、§5。
- 支持事实：在论文测试的机器翻译和 BERT 自然语言推理任务中，许多 head 可逐个在测试时屏蔽而不显著降低指标，有些层在该单层分析中只保留一个 head 也影响有限；但跨层迭代剪枝时，若要避免明显性能下降，大多数 head 仍需保留。机器翻译的 encoder-decoder attention 比 self-attention 对剪枝更敏感。
- 证据等级：指定模型、任务和屏蔽/联合剪枝方法下的原始消融。
- 适用边界：单个 head 可被移除不等于模型从训练开始就不需要多头，也不说明该 head 没有局部作用；单层逐个屏蔽与全网联合移除不是同一个问题。

- 来源：[On the weak link between importance and prunability of attention heads](https://aclanthology.org/2020.emnlp-main.260/)，Budhraja 等，EMNLP 2020。阅读位置：摘要及论文 §§2–3。
- 支持事实：作者在 Transformer 翻译模型和 BERT 多任务设定中发现，可随机剪除相当比例的 head 而仅有限影响准确率；按既有解释研究推断的 head importance 并不稳定预测可剪枝性，微调时其余 head 对被剪部分的补偿大致分散。
- 证据等级：论文设定下的原始剪枝实验。
- 适用边界：结果受模型、任务、剪枝比例及微调协议限制；它提醒解释性重要度与性能损失是不同测量量，不证明 head pattern 一概无用。

- 来源：[Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](https://papers.nips.cc/paper_files/paper/2025/file/a7f530e11fa19e9551b7a51dbd0f336f-Paper-Conference.pdf)，Nam 等，NeurIPS 2025。阅读位置：摘要、§§3–4、讨论。
- 支持事实：作者在四个 Llama 3 系列变体以及句法、常识和数学任务数据上，用 head 输出门控和消融估计任务相关性，并在各模型—数据集组合上使用多个随机种子。论文报告同一 head 的作用会随任务和其他 head 是否启用而变化；该方法可帮助筛选任务相关子电路，但作者明确说明它本身不揭示单个 head 的精确计算。
- 证据等级：会议论文中的特定现代因果语言模型和任务实验。
- 适用边界：论文所测 Llama 3 变体、任务集和 CHG 流程不代表全部模型或 head 角色；门控分数是任务相关干预证据，不是跨任务的语义标签。

- 来源：[Quantifying LLM Attention-Head Stability: Implications for Circuit Universality](https://arxiv.org/abs/2602.16740)，Bali 等，arXiv v1（2026-02-17，预印本）。阅读位置：摘要及跨独立初始化重训的实验方法与结果。
- 支持事实：作者跨独立初始化的 GPT-like Transformer 重训比较 head 表示稳定性，报告其测试设置中中间层 head 稳定性最低、residual stream 相对稳定；优化设置改变时测得的稳定性也会变化。
- 证据等级：尚处预印本阶段的模型重训研究。
- 适用边界：结论受所测试的模型尺寸/架构、训练数据、随机重训数量和相似度度量限制，不能推广成所有前沿模型或所有随机种子的普遍规律。

- 来源：[Entropy Meets Importance: A Unified Head Importance–Entropy Score for Stable and Efficient Transformer Pruning](https://aclanthology.org/2026.findings-emnlp.2191/)，Choi 等，Findings of EMNLP 2026。阅读位置：摘要、§§4.1–4.2、§§5.1–5.3、附录 C.2、D.2 和 E.3；直接核对 ACL Anthology 全文 PDF。
- 支持事实：论文提出 HIES，将归一化的梯度式 head importance score 与注意力熵缺口组合为剪枝评分；实验覆盖作者所列的 9 种模型架构、21 个语言/视觉/多模态数据集，并与随机剪枝、HIS 等基线比较。作者报告的相对提升绑定论文的质量/稳定性指标、剪枝比例、模型、校准数据和汇总口径。论文还指出 GQA 中单独移除一个 query head 不会释放共享 K/V cache，除非整个共享组也被移除。
- 证据等级：已发表会议论文中的方法、分析和指定设置实验。
- 适用边界：HIES 是论文提出的一种剪枝准则，不是 head 语义功能的测量真值，也不能证明其在其他模型/任务中的通用收益。缓存结论依赖 GQA 的共享分组结构，属于架构和 serving 账本问题。

- 来源：[Examining the Universality Assumption of Attention Head Pruning across Datasets](https://doi.org/10.1587/transinf.2026PAL0001)，Gong 等，IEICE Transactions on Information and Systems，2026-06-15 advance online publication。
- 支持事实：可访问的期刊页面摘要称，作者跨多个数据集和模型检查“不同数据集可复用相同剪枝 head”的假设，并报告实际待剪 head 具有较强数据集依赖性。
- 证据等级：当前仅核对出版方页面摘要；页面提供 PDF 入口，但本轮浏览器工具无法打开 PDF 正文，尚未核实实验协议和结果细节。
- 适用边界：摘要结果不作为正文的核心证据；只有后续能阅读全文并核对模型、数据集、剪枝方法和量化指标后，才引用其详细结论。
