# Scaled dot-product attention：计算与实现证据

检索日期：2026-10-28。

## 缩放点积、方差动机与因果 mask

- 来源：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，Vaswani 等，arXiv v7（2023-08-02）。阅读位置：§3.2.1、§3.2.3。
- 支持事实：scaled dot-product attention 对 Q/K 点积除以 $\sqrt{d_k}$，再沿 Key 位置 softmax 并加权 V；论文以各维独立、均值为 0、方差为 1 的假设说明未缩放点积方差随 $d_k$ 增长，因此提出该缩放项。解码器因果遮罩把非法连接对应的 softmax 输入设为负无穷。
- 证据等级：原始论文的方法定义和作者给出的设计动机。
- 边界：方差推导依赖论文写明的分布假设，不代表真实网络每层、每头的 Q/K 都符合该分布；作者给出的训练动机不是对任意初始化、精度或模型配置的保证。

## FlashAttention 与标准注意力中间矩阵

- 来源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135v2)，Dao 等，arXiv v2（2022-06-23）。阅读位置：§2.2、§3.1–3.3。
- 支持事实：论文的标准 attention 流程显式形成 $N\times N$ 分数矩阵 $S$ 和概率矩阵 $P$ 并在 HBM 中读写。FlashAttention 使用分块（tiling）和重计算来计算 exact attention，避免保存这些大型中间矩阵并减少 HBM 读写；论文将 block-sparse 扩展另列为稀疏近似算法。
- 证据等级：原始论文算法、IO 复杂度分析和论文实验。
- 边界：IO 优势、运行时间和内存占用取决于硬件、序列长度、head dimension、dtype 与 kernel；论文中的速度数字不外推到未测配置。exact 指算法针对同一 attention 数学目标，不意味着不同浮点实现逐位一致；本仓库未测量 kernel 性能。

## Local / sparse attention

- 来源：[Longformer: The Long-Document Transformer](https://arxiv.org/html/2004.05150v2)，Beltagy 等，arXiv v2（2020-12-02）。阅读位置：摘要、§3.1。
- 支持事实：Longformer 将滑动窗口局部注意力与任务相关的全局注意力组合；固定窗口大小时，其论文所述 attention pattern 的计算随序列长度线性增长。
- 证据等级：原始论文中的结构定义和作者分析。
- 边界：这是 Longformer 的具体可见关系设计，不是所有稀疏注意力方案的通用模式；减少位置交互意味着改变标准全连接注意力的连接图。

## Linear attention 的特征映射边界

- 来源：[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/html/2006.16236v2)，Katharopoulos 等，arXiv v2（2020-06-30）。阅读位置：§3.2.1。
- 支持事实：论文用 kernel feature map 重写注意力相似度并利用矩阵乘法结合律改变求和顺序；作者指出与标准 softmax 对应的 exponential kernel 需要无限维精确 feature map，而所用有限维 feature map 构造的是该论文的 linear attention 形式。
- 证据等级：原始论文中的方法推导与特定实验。
- 边界：论文中的任务指标和速度结论依其模型、任务和硬件设置；“linear attention”覆盖多种方法，不能把一个 feature map 的性质推广到全部方法，也不能默认与标准 softmax attention 等价。
