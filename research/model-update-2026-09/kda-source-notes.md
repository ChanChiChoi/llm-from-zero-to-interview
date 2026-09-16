# Kimi Delta Attention（KDA）原始论文核验

核验日期：2026-09-09。

来源：[Kimi Linear arXiv 摘要](https://arxiv.org/abs/2510.26692)、其 LaTeX 源码、[MoonshotAI FlashKDA 搜索到的官方仓库名](https://github.com/MoonshotAI/FlashKDA)。本笔记依据论文源码，不代表 Kimi K3 已公开全部采用的配置；Kimi K3 官方文章只明确提到 KDA。

## 论文定位

论文于 2025-10-30 首次提交，标题为 *Kimi Linear: An Expressive, Efficient Attention Architecture*。摘要称 KDA 是 Gated DeltaNet 的线性注意力扩展，加入更细粒度 gating，并使用 DPLR transition 的专门变体实现 chunkwise 算法。

## 核心递推

论文给出的每头状态为 `S_t ∈ R^{d_k × d_v}`：

```math
S_t=(I-\beta_t k_t k_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_t k_t v_t^\top,
\qquad o_t=S_t^\top q_t.
```

`alpha_t` 是逐通道衰减门，`beta_t` 是标量更新门；与只用标量衰减的 GDN 相比，KDA 可以对不同 key 通道施加不同记忆衰减。第一项先衰减已有状态，再进行 rank-1 的纠正和写入。不能把它简化为普通 attention 的 `softmax(QK^T)V`。

论文将状态更新解释为 delta rule：根据当前 key/value 关系修正一个有限状态的关联记忆。KDA 把 DPLR 结构中的两组向量绑定到同一个 key，以减少二级 chunking 和矩阵乘法；源码声称相对通用 DPLR 的 operator 效率约提升 100%，这是论文实现和测量语境，不能泛化为所有硬件。

## 模型层面的组合

源码称 Kimi Linear 使用 KDA 与少量全局 MLA 层混合，实验中采用 3 个 KDA 层接 1 个全局 MLA 层。MLA 层使用 NoPE，把位置和新近性主要交给 KDA；这是一种具体架构实验选择，不是 KDA 的必要条件。

KDA 的 `q/k/v` 经过 ShortConv 和 Swish，`q/k` 再做 L2Norm；论文实验头维度为 128。逐通道衰减和输出 gate 使用低秩参数化。这里的 head dimension、低秩维度和 3:1 比例都必须与 Kimi K3 区分：K3 官方文章没有给出这些完整配置。

## 效率与局限

线性状态的大小与序列长度无关，解码阶段可保持固定状态；但纯线性注意力的长程精确检索仍是瓶颈，因此论文引入全局注意力层。预填充与自回归解码使用不同 kernel 形态：前者更偏 chunk 并行，后者使用递归更新。教学章节应把“固定状态”与“不会遗忘”区分开，衰减和有限容量仍会造成信息损失。

## 后续章节设计

先从“把一叠历史笔记压缩成一个可更新的记忆矩阵”讲起，再手算一个小维度状态更新；随后推导衰减、rank-1 写入和 delta correction，最后实现简化版 PyTorch recurrent KDA。再对照 full attention 的 KV cache，解释为何 KDA 解码状态固定而检索能力需要混合全局层。正式代码必须标注为教学简化实现，不冒充 FlashKDA kernel。
