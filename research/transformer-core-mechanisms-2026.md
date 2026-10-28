# Transformer 核心机制的来源证据

检索日期：2026-10-28

## Grouped-query attention

- 来源：[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/html/2305.13245v1)，Ainslie 等，arXiv v1（2023-05-22）。
- 阅读位置：摘要；§2.1–2.2；§3.1–3.2；§3.3 的 group 数消融。
- 支持事实：GQA 将 Query heads 分组，每组共享一个 Key head 和 Value head；组数为 1 时对应 MQA，等于 Query head 数时对应 MHA。作者报告其 uptrained T5.1.1 GQA 在论文实验中接近 MHA 的质量，并接近 MQA 的推理速度；共享 KV heads 减少 KV cache 的大小及读取量。作者还讨论了 MQA 在其部分任务中的质量退化和训练不稳定，以及 GQA 组数对质量/速度的折中。
- 证据等级：原始论文的方法定义和作者报告的实验结果。
- 边界：质量、速度和 uptraining 成本来自论文模型、硬件及实验设置，不能当作任意模型都具有相同比例的收益。

## Multi-query attention

- 来源：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/html/1911.02150v1)，Shazeer，arXiv v1（2019-11-06）。
- 阅读位置：摘要；§2.4/§2.4.1；§3/§3.1；§4.1–4.3。
- 支持事实：自回归增量推理每步需要复读先前的 K/V 状态；MQA 保持多个 Query heads，但让它们共享同一组 Key/Value heads，以减少 K/V 张量大小和增量解码的内存带宽需求。论文在 WMT14 En-De 翻译和 Billion Word 语言模型上报告了相对当时 MHA 基线的质量和速度结果。
- 证据等级：原始论文的结构定义、成本分析和受控实验。
- 边界：实验使用论文所述 Transformer 尺寸、TPUv2、批量和序列长度；作者报告的性能与轻微质量变化不能直接外推到现代模型、GPU kernel 或生产 serving。

## FlashAttention 与标准注意力的 IO 区别

- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，Dao 等，arXiv v1（2022-05-27）。
- 阅读位置：摘要；§3.1–3.2 和定理 1（HTML 第 84–145 行）。
- 支持事实：FlashAttention 使用 tiling 和 recomputation，在减少 GPU 高带宽内存（HBM）读写的同时计算 exact attention；论文给出其额外工作内存复杂度，并分析标准实现与分块算法的 IO 复杂度差异。
- 证据等级：原始论文算法、复杂度分析和作者实验。
- 边界：具体运行时间取决于硬件、head dimension、序列长度和实现；论文中某些 speedup 数字不可直接外推。Exact 描述针对 FlashAttention 的算法，不应把所有近似/稀疏 attention 都归为等价实现。

## SentencePiece、子词算法与原始文本处理

- 来源：[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)，Kudo 与 Richardson，arXiv v1（2018-08-19）。
- 阅读位置：摘要；§3.2–3.4（第 94–107 行）；结论（第 277–279 行）。
- 支持事实：SentencePiece 是 tokenizer/detokenizer 工具，能够从原始句子训练子词模型；论文讨论 BPE 与 unigram language model 两类模型，因此工具名称不能和单一切分算法混为一谈。论文所述默认配置包含 NFKC 字符归一化，归一化方式可以配置。
- 证据等级：原始论文的工具定义、实现说明和实验。
- 边界：默认归一化行为及命令行配置属于论文所述 SentencePiece 实现；不同 tokenizer、模型文件或版本可以采用不同规范化规则。

## Rotary Position Embedding

- 来源：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)，Su 等，arXiv v5（2023-11-08）。
- 阅读位置：摘要（第 16–18 行）。
- 支持事实：RoPE 以旋转矩阵编码绝对位置，并在 self-attention 形式中体现相对位置依赖；作者在长文本分类基准上报告其实验比较。
- 证据等级：原始论文摘要与作者报告的 benchmark 结论。
- 边界：摘要中的序列长度灵活性和实验结果不证明模型能在训练范围外可靠使用任意长上下文，也不等于位置扩展技术的普遍保证。

## Selective state space models 作为架构边界

- 来源：[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)，Gu 与 Dao，arXiv v1（2023-12-01）。
- 阅读位置：§3.2–3.3（HTML 第 101–125 行）；§4.5–4.6（第 414–443 行）；结论（第 507–508 行）。
- 支持事实：Mamba 让若干 SSM 参数依赖输入以形成选择机制，因此转为 time-varying SSM；论文提出硬件感知的 selective scan 处理其无法直接使用卷积的计算挑战，并评估单独及混合架构配置。
- 证据等级：原始论文的模型定义、算法及论文实验。
- 边界：论文的训练、推理速度和质量数字受模型规模、数据、硬件与对照实现约束。此来源支持其作为另一类序列模型机制，不支持对 Transformer 或 Mamba 的普遍优劣结论。

- 来源：[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)，Dao 与 Gu，arXiv v1（2024-05-31），发表于 ICML 2024。
- 阅读位置：arXiv HTML 摘要与 §1（第 149–182 行）。
- 支持事实：作者提出 structured state space duality 框架，说明部分 SSM 与 attention 变体之间存在结构联系，并据此提出 Mamba-2；论文报告其核心层比 Mamba 快 2–8 倍且语言建模表现保持竞争力。
- 证据等级：原始论文摘要和作者报告的比较。
- 边界：该结果属于论文模型与实验设置；不能简化成所有 SSM 与标准 softmax attention 等价，或可互换。
