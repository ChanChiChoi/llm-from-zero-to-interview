# Attention、卷积与 SSM 的信息路径和组合位置

研究日期：2026-11-23。研究问题：如何根据历史信息的表示与读取方式区分 Attention、卷积和状态空间模型（SSM），并避免把“局部、长程、全局”直接当作能力或性能承诺。范围限于因果序列模型中的内部序列混合路径，不涉及模型外检索系统、服务协议或生产运行手册。

## 检索与阅读范围

通过网页搜索发现资料并直接阅读原论文、会议论文库和技术报告。检索词包括 `Qwen3 convolution depthwise Conv1D ablation architecture local convolution`、`Simple Hardware-Efficient Long Convolutions for Sequence Modeling H3 attention layers`、`StripedHyena 7B technical report hybrid attention convolution primary paper`、`Hybrid Architectures for Language Models Systematic Analysis and Design Insights`、`Mamba selective state spaces`、`causal convolution receptive field dilation` 和 `Attention Is All You Need query key value`。筛选重点是能直接支持运算机制、因果可见性和具体组合位置的一手资料；没有运行代码、模型或 benchmark。

## 研究发现

### 1. 模块职责要按历史怎样被表示和读取来判断

在因果 full attention 中，query 与可见 keys 的兼容度形成权重，对相应 values 加权；因果 mask 规定当前位置只能读取允许的前缀位置。该机制提供显式的内容相关读取路径，但“有读取路径”不等于训练后能准确召回相关内容。窗口或稀疏 mask 会改变可读位置集合。

因果卷积依据相对位移共享卷积核权重，输出只能依赖其核或网络结构可达的过去位置。短核给出局部范围；层叠、扩张卷积或长核可以扩大乃至覆盖整段序列。因此“卷积就是局部”的说法不成立。卷积也不自动提供 query 对任意历史位置重新分配权重的机制；带门控或输入控制的具体算子还须按其实际结构分析。

SSM 以递归状态表示已读前缀。固定维度的递归状态不随历史 token 数逐项增长，但它保存哪些任务相关信息取决于参数化和训练。固定线性时不变系统与固定卷积核存在等价表示；Mamba 则让部分 SSM 参数依赖输入，使状态更新能按当前内容选择性传播或遗忘。两者均不能被泛化为“只记趋势”，也不能仅凭固定大小状态断言所有精确信息都不可能编码。

这些路径描述的是结构偏置：相对位置滤波、递归状态更新和 query 条件读取并不等同于一套固定的“局部/长程/全局”职能表，也不能仅由机制推出 task score、训练速度或生成吞吐。

### 2. 公开实例显示卷积可以进入不同位置

《Convolution for Large Language Models》在 Qwen3 block 的 17 个位置比较 Conv1D；该位置消融中，联合处理 Q/K/V 投影的 P5（Attention 聚合之前）取得该设置下最低平均 loss 与 perplexity。作者将其解释为先为 Attention 输入提供局部上下文，同时明确位置消融本身没有隔离证明这一中间解释。此报告的模型、训练数据预算与 benchmark 条件，以及摘要和参数表中的参数增量差异，详见[卷积序列模型研究记录](convolution-sequence-models-in-llms-2026-11-23.md)。

Fu 等人的 LongConv 语言建模实验沿 H3 设置保留两层 Attention，以长卷积替换 H3 层中的 SSM 核。这证明长卷积可与 Attention 共同位于一个序列模型中，不是纯卷积主干的替代证据；数据集和训练 token 条件见来源原文及上方研究记录。

因此需要区分局部卷积修正 Attention 投影、长卷积担任序列混合算子、卷积和 Attention 在主干中共存等结构位置。名称同为“卷积”不表示信息路径、范围或效果相同。

## 来源与边界

| 来源 | 版本、位置 | 可支持 | 不支持的外推 |
| --- | --- | --- | --- |
| Vaswani et al., [Attention Is All You Need](https://arxiv.org/html/1706.03762) | arXiv v7，2023-08-02；§3.1–3.2 | Attention 通过 query-key 兼容度对 values 加权；decoder mask 限制可见前缀 | 所有注意力变体均可读取整段历史，或注意力保证正确召回 |
| Bai, Kolter & Koltun, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) | arXiv v2，2018-04-19；§3.2–3.3 | causal convolution 的过去输入约束，dilation 扩大 TCN 感受野 | TCN 在现代 LLM 或所有任务上优于 RNN/Attention |
| Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/html/2312.00752) | arXiv v2，2024-05-31；§3.1–3.5 | 输入依赖的状态选择机制及按当前 token 传播/遗忘信息 | 任意 SSM 都有同样选择性，或模型能无损保存所有输入 |
| Tian et al., [Convolution for Large Language Models](https://arxiv.org/html/2607.18413) | arXiv v1，2026-07-20；§3.2–3.4、Tables 1–3 | Qwen3 block 的 17 处位置比较及 post-QKV P5 的本设置结果 | 局部卷积在其他架构和训练配方中必然提升质量 |
| Fu et al., [Simple Hardware-Efficient Long Convolutions for Sequence Modeling](https://proceedings.mlr.press/v202/fu23a.html) 与 [arXiv 原文](https://arxiv.org/html/2302.06646) | ICML 2023 / PMLR 202；arXiv v1；文本建模 §4.1.3、Tables 7–8 | H3+LongConv 的层替换方式、保留两层 Attention 和所报数据集结果 | 纯卷积主干胜过 Transformer，或结果只由卷积核单独造成 |

技术主张均为 `source-backed`；没有本地复现。跨位置机制、性能及具体模型结果的证据边界见各来源原文和[卷积序列模型研究记录](convolution-sequence-models-in-llms-2026-11-23.md)。
