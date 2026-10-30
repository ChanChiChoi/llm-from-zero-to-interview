# Transformer 架构方向的新证据（2026）

检索日期：2026-10-30

## 深度方向残差路由

- 来源：[Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630)，arXiv（2026-10-09）。
- 阅读位置：arXiv 摘要。
- 支持事实：MSAR 将 AttnRes 的深度方向选择扩展为多头深度路由；通过多尺度因果卷积形成不同的上下文视角，并按需展开多路表示。摘要称其不需要持续保存或通信多路状态；作者报告在 18B 与 28B MoE 规模、12 个下游基准上优于其比较的标准残差、mHC 与 AttnRes，并报告相对标准残差 1.58× 的 iso-loss compute speedup。
- 证据等级：原始预印本摘要；未阅读完整论文。
- 边界：实验规模、基线、数据、实现和计算口径均限于论文设置；不得据此宣称通用训练加速或生产开销。后续纳入正文前须阅读完整论文并核对实验表、消融和代码。

## 残差与归一化

- 来源：[SpanNorm: Reconciling Training Stability and Performance in Deep Transformers](https://proceedings.mlr.press/v306/chao26b.html)，ICML 2026，PMLR 306:12948–12966。
- 阅读位置：PMLR 官方论文页摘要。
- 支持事实：SpanNorm 保留 PreNorm 的直接残差路径，在残差输出端采用 PostNorm 式归一化，并配合缩放策略；论文摘要称其提供信号方差有界的理论分析，并报告 deep dense 与 MoE Transformer 实验。
- 证据等级：同行评审会议论文的官方出版页与摘要。
- 边界：本记录仅依据论文页摘要，不记录未核对的模型尺寸、指标或逐项消融；作者结果不能直接建立通用默认配置。

## FFN 激活

- 来源：[PowLU: An Activation Function for Stable Pre-Training of LLMs](https://arxiv.org/html/2605.25704)，arXiv v1（2026-05-25）。
- 阅读位置：摘要；§4.2.2 的 7.9B 实验表；§4.2.3 的 124B 实验表；§4.3.1 loss-spike 分析。
- 支持事实：论文指出 SwiGLU 在大正输入下近似二次增长，并提出 rational-power PowLU。Ling MoE 实验分别报告 7.9B 总参数、600B token 和 124B 总参数、800B token 设置。
- 证据等级：原始预印本的方法说明和作者实验。
- 边界：结果属于 Ling 与论文训练配方；不同实验的精度和激活替换时点并不完全相同，不能把所有质量或稳定性差异单独归因于激活函数，也不代表其他模型可直接替换。

- 来源：[More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647)，arXiv v2（2026-08-28）。
- 阅读位置：arXiv 摘要及版本历史；完整 HTML 正文未能从该入口取得。
- 支持事实：摘要称 Mixture of Activations（MoA）用输入相关轻量门控混合多种激活并共享线性投影；作者报告 0.12B—2B dense 与 MoE 模型上的预训练实验，并报告相对调优基线的终点损失结果。
- 证据等级：原始预印本摘要；未核对正文实验表。
- 边界：正文只采用摘要能够支持的机制、模型范围和作者报告，不扩写未读到的门控细节、benchmark 数字或计算开销；MoA 与 PowLU 的任务和规模不同，不作横向排名。

## 递归状态与混合注意力

- 来源：[Log-Linear Attention](https://arxiv.org/html/2506.04761v3)，arXiv v3（2026-03-01）。
- 阅读位置：摘要；§1；§3 的 Fenwick-tree 状态组织与解码描述；§4 的实验概述；§5 讨论与限制。
- 支持事实：该框架把固定大小状态扩展为随前缀长度对数增长的多尺度状态；论文给出的特定形式为 $O(T\log T)$ 训练计算和 $O(\log T)$ 状态/解码成本，并将其应用到 Mamba-2 与 Gated DeltaNet。作者的语言模型实验约为 700M—800M 参数、使用 H100；报告有些合成检索和长上下文任务改善。
- 证据等级：原始预印本的机制、算法和作者实验。
- 边界：作者说明部分任务没有优于线性状态基线，整体仍与 Transformer 存在差距；仅运行一次的 700M—800M 模型限制了调参分析。Fenwick 状态数不等于常数大小的字节成本；区间内计算和反向传播需要更复杂实现。结果不支持普遍的长文档问答或服务加速结论。

- 来源：[Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention](https://arxiv.org/html/2605.22791)，arXiv v1（2026-05-21）。
- 阅读位置：摘要；§3.1；§4 实验设置；§4 的语言建模、检索结果与 RULER 切片。
- 支持事实：Gated DeltaNet-2 将 key 侧 channel-wise erase gate $b_t$ 与 value 侧 channel-wise write gate $w_t$ 分开；作者报告 1.3B 参数、100B FineWeb-Edu token 训练设置，并分别研究纯递归模型与加入 sliding-window attention 的 hybrid 模型。论文在若干常识和多键检索切片上报告改进。
- 证据等级：原始预印本的算法说明和作者实验。
- 边界：结果限于论文模型、数据、长度、基线与评测协议；hybrid 结果包含显式窗口 attention，不能全归因于递归更新，也不证明其普遍优于 full attention。
