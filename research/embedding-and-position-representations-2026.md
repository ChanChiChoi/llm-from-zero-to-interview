# Embedding 与位置表示：事实和证据

检索日期：2026-10-28；补充检索日期：2026-11-19。

## 检索范围与筛选

- **问题范围：** 位置表示如何注入序列顺序；相对位置方法有哪些不同实现；位置表示的公式性质能否支持长度泛化结论；哪些内容适合基础位置编码章节、哪些属于长上下文扩展；公开教材/课程如何组织同一主题。
- **平台和检索词：** 2026-11-19 使用 Web 搜索检查 `CS224N 2025 lecture 8 transformers positional encoding`、`Dive into Deep Learning positional information transformer chapter sinusoidal learned RoPE ALiBi NoPE`；同时按标题、arXiv 编号、ACL Anthology 条目和 PMLR 论文页直达原始/官方来源。主要筛选英语来源；按经典方法、相对位置变体、RoPE 扩展和教材/课程组织分别核对。时间范围为 2017–2026，检索目标是方法定义与主张边界，不以搜索结果摘要单独支持技术结论。
- **筛选边界：** 纳入实际阅读的原始论文、会议论文页面、官方课程讲义和公开教材章节；搜索发现但未直接阅读原文的材料不作为结论依据。教材/课程只支持其实际采用的编排样本，不支持普遍学习效果。

## Token embedding 与权重共享

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。
- 阅读位置：§3.4 “Embeddings and Softmax”。作者说明其 Transformer 使用学习式输入/输出 token embedding，并在该模型中让两组 embedding 与 pre-softmax 线性变换共享权重；原文还说明 embedding weights 乘以 $\sqrt{d_{model}}$。
- 证据等级：原始论文对其模型设计的直接描述。
- 边界：共享权重和缩放是该论文的具体模型设定，不代表所有 Transformer 都采用相同做法。

## 绝对位置编码

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。
- 阅读位置：§3.5 “Positional Encoding”。论文把与 token embedding 同维度的位置编码加到 encoder/decoder 输入；比较固定正弦/余弦位置编码与 learned positional embeddings，并将固定编码可能帮助长度外推表述为设计假设。公式定义每个位置和维度的正弦/余弦值，频率按几何级数变化。
- 证据等级：原始论文的方法说明和其特定翻译实验。
- 边界：论文所述“可能外推”是提出方法时的动机，不等于模型在未见长度上一定可靠。该论文报告的近似性能只适用于其测试配置。

## 相对位置表示

- 来源：[Self-Attention with Relative Position Representations](https://arxiv.org/html/1803.02155v2)，Shaw、Uszkoreit、Vaswani，arXiv v2（2018-04-12；NAACL 2018）。
- 阅读位置：§3.1–3.3，尤其 §3.2；实验范围见 §4.1–4.3。
- 支持事实：该方法按 query/key 两位置的相对距离构造表示，将相对表示加入 key 侧匹配，并将 value 侧相对表示加入加权输出；对距离截断到 $[-k,k]$。作者在 WMT 2014 英德、英法机器翻译配置下报告相对表示相对于其绝对位置基线的结果。
- 证据等级：原始论文的方法公式和作者报告的实验。
- 边界：相对位置方法可有不同实现。论文结果受翻译任务、模型配置、裁剪距离和训练设置限制，不能推出其他任务或所有相对位置机制的收益。

## Attention with Linear Biases (ALiBi)

- 来源：[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/html/2108.12409v2)，Press、Smith、Lewis，arXiv v2（2022-04-22）。
- 阅读位置：§1、§3 和 §4。
- 支持事实：ALiBi 不向 token embedding 加位置向量，而是向 query-key 注意力分数加入与两者距离成比例的惩罚。论文在其具体语言模型、数据和设置中报告了训练长度 1,024、测试长度 2,048 的结果，以及与同设置正弦位置模型的速度、内存比较。
- 证据等级：原始论文的方法定义和作者报告的实验。
- 边界：该论文结果不证明所有任务、模型或更长长度均能同样外推；“长度支持”与“有效利用上下文”仍须按任务测量。

## 未验证边界

- 未测量特定模型的 embedding 初始化、position ID 约定、padding 行为、权重共享或可用上下文长度。此类接口和配置须以对应模型文件及实现为准。

## 2026-11-19 补充：相对位置、因果掩码与教学组织

### 相对位置方法不是单一实现

- **Shaw 等，Self-Attention with Relative Position Representations，NAACL 2018。** [ACL Anthology 页面](https://aclanthology.org/N18-2074/)及[论文 PDF](https://aclanthology.org/N18-2074.pdf)，阅读摘要、方法和实验说明。论文将相对距离表示纳入 attention，并在 WMT 2014 英德、英法任务上分别报告相对绝对位置基线增加 1.3 和 0.3 BLEU；将相对与绝对位置一起使用，在这组实验中没有额外提升。证据是特定机器翻译设定下的原始方法与作者实验；不能外推为所有任务或所有相对位置方法的收益。
- **Dai 等，Transformer-XL，arXiv v3（2019-06-02）。** [论文 HTML](https://arxiv.org/html/1901.02860)，阅读 §1、§3.3 和消融表。论文把相对位置编码与跨 segment 的状态复用配合，理由是复用旧 segment 的状态时，纯绝对位置会造成时间位置混淆。其长依赖和性能结果属于 Transformer-XL 的组合架构，不能单独归因于位置编码。
- **Raffel 等，Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer，arXiv v4（2020）。** [论文 HTML](https://arxiv.org/html/1910.10683v4)，阅读 §2.1。T5 把 query-key 相对 offset 映射为加到 attention logit 的标量 bias；其具体模型使用 32 个按距离范围对数分桶的位置参数，超过 128 的 offset 共用最远桶。论文说明单层在超过该距离时不再区分 offset，但后续层可组合局部信息获得更大范围的敏感性。上述分桶数值是 T5 的配置，不是相对位置偏置的通用定义。

### 位置关系不等于长度泛化

- **RoPE 的代数性质：** Su 等，RoFormer，arXiv v4（2022-08-09），[论文 HTML](https://arxiv.org/html/2104.09864v4)，阅读 §§3.1–3.3。对 query 和 key 的成对维度作按位置旋转后，其点积的位置项可以写成两位置之差；旋转矩阵正交。论文还分析其指定频率表的长距离衰减性质。相对位移出现在打分公式中，并不证明模型会在训练未覆盖的位置上做出可靠预测；衰减分析也不应表述为任意内容向量的 attention 分数必然随距离单调下降。
- **ALiBi：** Press 等，Train Short, Test Long，arXiv v2（2022-04-22），[论文 HTML](https://arxiv.org/html/2108.12409v2)，阅读 §§2–4。ALiBi 向每个 query-key score 加入 head-specific 的线性距离惩罚；由于偏置可按任意距离计算，它没有固定位置表的行数上限。论文在其 WikiText-103 与 1.3B 模型实验中报告了训练短序列、推理长序列的结果；例如 1.3B 设置把 1,024 token 训练序列用于 2,048 token 推理，并报告与 2,048 token 正弦模型接近的困惑度且训练更快、内存更少。该结果受模型、数据、实现和评估口径限制；偏置可计算至远距离不代表模型能有效利用任意长度上下文。
- **Position Interpolation：** Chen 等，arXiv v2（2023-06-28），[论文 HTML](https://arxiv.org/html/2306.15595v2)，阅读 §§2.2–2.3、§3。PI 将扩展后的相对位置缩放到原始 RoPE 位置范围，避免直接把新位置索引送入训练未覆盖的角度范围。作者在 LLaMA 7B–65B 的特定实验中报告由 2K 扩至 32K，并使用短轮次微调。该方法证明的是该实验中的扩展路径，不构成所有 RoPE 模型的无损窗口扩大保证。
- **YaRN：** Peng 等，arXiv v3（2026-02-06），[论文 HTML](https://arxiv.org/html/2309.00071v3)，阅读摘要、§§3.1–3.4 和 §4。论文组合不同频率区段的 RoPE 缩放与 attention 温度调整；Llama 2 的 7B/13B 实验在其数据和配置下把 4K 预训练窗口扩到 128K，并报告以数百训练步微调和从较短微调窗口继续外推的结果。论文摘要的“比先前方法少 10 倍 token、少 2.5 倍步数”是其比较基线和设置下的报告，不是一般训练成本规律。
- **LongRoPE：** Ding 等，ICML 2024，PMLR 235，[论文页面](https://proceedings.mlr.press/v235/ding24i.html)，阅读摘要和方法/实验概述。作者对 LLaMA 2、Mistral 采用非均匀位置插值、渐进扩展，并在长窗口后用 8K 再调整短窗口表现；报告在最多 256K 的中间训练长度上扩展到 2,048K。该 token 窗口是特定模型与评估设置的名义长度；不能单凭窗口数值推断任意任务的有效容量。

### 因果掩码可提供隐式位置线索

- **Haviv 等，Transformer Language Models without Positional Encodings Still Learn Positional Information，Findings of EMNLP 2022。** [ACL Anthology 页面](https://aclanthology.org/2022.findings-emnlp.99/)及[论文 PDF](https://aclanthology.org/2022.findings-emnlp.99.pdf)，阅读摘要和论文实验部分。作者报告在其不同数据集、模型规模和序列长度实验中，不带显式位置编码的 causal LM 仍有竞争表现，并通过 probing 观察到隐式绝对位置信息；作者提出因果注意力中每个位置可访问的前驱数量是可能来源。它支持“因果掩码可能提供位置信号”的解释，不证明所有任务、模型都能去掉位置表示。

### 教材与课程的组织样本

- **Dive into Deep Learning，§10.4 Positional Information。** [章节页面](https://d2l.smola.org/chapter_attention/positional-information.html)，访问日期 2026-11-19。章节从无位置/无掩码 self-attention 的置换等变性出发，依次讲绝对位置表、Sinusoidal 与 Learned、正弦位置的旋转关系、RoPE，再以长度外推任务并列 ALiBi 与 NoPE；之后用固定小模型实验比较长度变化，并给出机制练习。它是一个教材组织实例，不是该顺序能改善学习效果的对照研究。
- **Stanford CS224N/Ling284 Winter 2025，Lecture 8: Self-Attention and Transformers。** [官方讲义 PDF](https://web.stanford.edu/class/cs224n/slides_w25/cs224n-2025-lecture08-transformers.pdf)，访问日期 2026-11-19。讲义先指出 self-attention 缺少顺序信息，再说明加入位置表示，随后讲 causal mask 如何阻断未来信息并保留并行计算。它可作为基础依赖和直觉入口，不覆盖相对位置方法或 RoPE，也不能单独证明教学顺序的效果。

### 供章节结构审视使用的主题候选

- **无位置表示与因果 mask：** 读者结果是能区分“显式位置编码缺失”与“模型完全没有顺序线索”；前置是 self-attention 对输入排列的性质和 causal mask。可以作简短边界说明，是否进入主线仍需看本章篇幅与后续章节分工。
- **相对表示的实现差异：** Shaw、Transformer-XL、T5 分别代表相对向量进入 key/value、含相对项的分解打分、以及分桶标量 bias。读者需能按“位置进入哪一步、参数随什么索引、距离如何截断”分类；不能把这些统一写成一个公式。
- **长窗口扩展方法：** PI、YaRN、LongRoPE 展示了改变 RoPE 位置映射并继续适配的不同路径。它们的共同读者结果是区分原始 RoPE 的相对位移性质、未训练长度上的泛化和经过扩展训练后的模型能力；具体长窗口案例宜与长上下文专题分工。
