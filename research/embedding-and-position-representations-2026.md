# Embedding 与位置表示：事实和证据

检索日期：2026-10-28。

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
