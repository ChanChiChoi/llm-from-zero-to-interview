# MHA、MQA、GQA 与 MLA 的结构和证据边界

检索与原文核验日期：2026-11-19。本文记录架构机制、论文实验范围和来源边界；章节职责、正文修改与交接见 `progress/book-21-transformer-architecture-evolution/chapter-04-audit.md`。

## 多头注意力的基线

- **来源与版本**：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7。阅读位置：§§3.2.2、4。
- **支持事实**：各 head 使用各自学习到的 Q/K/V 投影并行计算，拼接后经输出投影组合。标准 head 是投影后的表示，不是把原始 hidden coordinates 预先切成互不重叠的语义槽。
- **证据边界**：原论文的投影与机器翻译实验不证明 head 表示相互正交、具有固定语义，也不建立某个 head 数对所有任务最优。

## Multi-Query Attention（MQA）

- **来源与版本**：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)，Noam Shazeer，arXiv（2019-11-06）。阅读位置：§§2.2–2.4、§§3–4，重点核对增量解码内存/算术分析、MQA 定义、质量与速度实验。
- **支持事实**：MQA 保留多个 query heads，但让它们共享一组 key 和 value heads。原论文把增量解码反复加载历史 K/V 的内存带宽成本作为动机；其特定 TPUv2/v3、固定形状和模型/数据设置下，MQA 解码明显更快，作者报告质量只出现较小变化。
- **证据边界**：实验包括英语—德语翻译和 Billion-Word 语言模型，属于论文当时的 encoder-decoder/decoder 模型与 TPU 环境；不能外推为现代 decoder-only 模型、其他 GPU kernel 或任意任务中的固定速度倍数或质量损失。

## Grouped-Query Attention（GQA）与 checkpoint uptraining

- **来源与版本**：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://aclanthology.org/2023.emnlp-main.298/)，Ainslie 等，EMNLP 2023。阅读位置：§§2–4、表 1、§5 Limitations；直接核对 ACL Anthology 全文 PDF。
- **支持事实**：GQA 将 query heads 分成若干组，每组共享一套 K/V；一组时退化为 MQA，组数等于 query heads 数时退化为 MHA。论文的转换方案把原 checkpoint 组内 K/V 投影作 mean pooling，再继续预训练；其 T5 uptraining 实验使用原预训练计算量约 5% 的附加训练，作者报告 GQA 在这些 encoder-decoder 任务中质量接近 MHA、推理速度接近 MQA。
- **限制**：作者指出评测仅覆盖 encoder-decoder 模型，没有对 decoder-only 模型实验；uptraining、质量和延迟数字均绑定 T5 checkpoint、训练 recipe、输入/输出长度、任务和硬件。不能把 5% 当成通用转换配方，也不能说所有 GQA 一定接近 MHA 质量。
- **交叉证据**：[DeepSeek-V2](https://arxiv.org/html/2405.04434v5) 附录 D.1 的 7B dense MHA/GQA-8/MQA 对比在四个 hard benchmarks 上由 MHA 得分更高。三种模型为近似总参数规模，并通过调整层数对齐参数；该结果同样受特定模型、训练数据和评测范围约束，不能单独判定一般优劣。

## Multi-head Latent Attention（MLA）

- **来源与版本**：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)，DeepSeek-AI，arXiv v5（2024-06-19）。阅读位置：§2.1.1–2.1.4、附录 C、附录 D.1–D.2。
- **支持事实**：MLA 对每个 token 的 K/V 相关信息作联合低秩下投影，缓存 latent $c_t^{KV}$；由于普通 RoPE 会使 key 投影依赖位置，DeepSeek-V2 将 content 部分和位置相关的 RoPE key 分开处理，推理缓存包含 latent 和共享的位置 key。按论文表示，每 token、每层缓存元素数为 $d_c+d_h^R$，而常规等宽显式 K/V 为 $2H_{kv}d_h$。论文也说明上投影可以吸收到 query/output 路径，不必在推理时显式恢复完整 K/V 张量。
- **作者实验范围**：附录 D.2 比较两个规模的 MLA/MHA MoE 模型，作者报告 MLA 缓存分别约为对应 MHA 变体的 14% 和 4%，且在所列四项 hard benchmarks 得分更高；两组模型的 activated parameters 不完全相等。摘要中的相对 DeepSeek 67B 的 93.3% KV-cache 降幅和 5.76× 最大生成吞吐属于 MLA、MoE 及其它模型配置共同组成的完整 DeepSeek-V2 对比，不是对 MLA 单因素的因果估计。
- **证据边界**：不将单一模型报告提升写成 MLA 在所有设置中优于 MHA；压缩比例取决于 latent 与 RoPE 维度、层数、数据类型和模型配置，实际延迟还受实现与 kernel 影响。

## 将现有 checkpoint 转成 MLA 的研究

- **来源与版本**：[Towards Economical Inference: Enabling DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs](https://aclanthology.org/2025.acl-long.1597/)，Ji 等，ACL 2025。阅读位置：§§1–4、Llama2-7B LongBench 表、附录；直接核对 ACL Anthology 全文 PDF。
- **支持事实**：MHA2MLA 使用贡献感知的 partial-RoPE 选择和 K/V 联合 SVD 低秩近似，将已有模型适配到 MLA，并继续微调恢复性能。论文报告使用其特定模型和数据规模时，仅需原训练 token 的约 0.6%–1%；Llama2-7B 示例报告 KV cache 减少 92.19%、LongBench 总体下降约 1%。
- **证据边界**：这是特定适配算法和论文模型上的训练结果，不是零样本替换，也不是任意 MHA checkpoint 的直接转换保证；论文结果不能代替目标模型上的质量与 serving 复核。

## 公开课程的组织样本

- **Stanford CME295 Lecture 2**：[2025 课程讲义](https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf) 与[课程大纲](https://cme295.stanford.edu/syllabus/)。阅读位置：讲义“Sharing attention heads”图表（PDF 第 15 页附近）；大纲 Lecture 2 项。讲义以 query-head 组数 $G$ 展示 MQA（$G=1$）、GQA（中间组数）与 MHA（$G=h$），并将其纳入 Transformer 变体课程单元。
- **University of Edinburgh, Model Inference Systems**：[2026 年 10 月讲义](https://auth.opencourse.inf.ed.ac.uk/sites/default/files/2026-10/model_inference_systems.pdf)。阅读位置：PDF 第 12 页的 KV cache/GQA 例子、第 38 页的 MLA cache 图、第 39 页关于 tensor-parallel cache 复制的案例，以及 FlashAttention 页。该课把 cache 容量、架构表示与 kernel 分开处理。
- **证据边界**：课程材料说明这些课程采用了何种呈现和主题分工；它们不是对固定教学顺序或学习成效的因果研究。
