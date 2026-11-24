# Gated Attention 的 SDPA 输出门研究记录

检索日期：2026-11-24。研究对象是 Qiu 等人的 softmax attention 输出门 G1、论文消融涉及的 G2—G5，以及后续关于 attention sink、长上下文与梯度机制的研究。重点核实门的位置和粒度、实验条件，以及 output feature gate 与 token attention/实际执行成本之间的区别。

## 核心机制与可支持主张

### G1 的输入、位置与张量含义

Qiu 等人将 gate 写为 $Y'=Y\odot\sigma(XW_\theta)$，其中 $X$ 是 pre-normalization hidden state，$Y$ 是被调节的表示。G1 将该乘法 sigmoid gate 放在 SDPA 输出后、最终多头输出投影前；论文实验分别采用 headwise（每头一个标量）或 elementwise（每头每个特征维一个值），并区分 head-specific 与 head-shared。

在固定某层的 Q/K/V、mask 和参数时，G1 作用于已经由 softmax attention 聚合的向量，不会改写该次 softmax 的 token 权重或可见集合。训练后模型的 hidden states/参数不同，后续 attention maps 可以发生经验变化。因此“当前前向 G1 不改 attention weights”与“训练后的 gated model 有不同 sink mass”是不同层次的主张。

### G1—G5 位置消融及非线性解释

原论文比较 G1（SDPA 输出）、G2（value projection 后）、G3（key projection 后）、G4（query projection 后）、G5（最终多头输出投影后）。统一对照还包含 gate 颗粒度、跨头 sharing、乘法/加法形式和激活函数。不能把这些轴合并成单一“用了 gate”的设定。

论文第 4.1 节在固定 attention weights 下写出 value projection $W_V^h$ 与对应输出子投影 $W_O^h$ 的复合映射。其秩至多为 head dimension $d_k$；当 $d_k<d_{\mathrm{model}}$ 时该线性 value/output 子路径受低秩限制，GQA 的共享 value projection 会进一步约束头间表达。论文将 G1/G2 插入的非线性视为增广该映射表达力的可能原因；该分析不等于整个含 query-key softmax 的 attention block 都为低秩，也不构成所有架构上的普遍保证。

G1 gate 由当前 query hidden state 决定，作用于聚合后的结果；G2 gate 由来源位置的表示决定，作用于 value 聚合之前。原论文把 G1 的 query dependency/head-specific sparsity 与其较强结果联系起来。此解释不是 G1 已执行 semantic token selection 或因果证据检索的证明。

### 15A2B MoE 消融

15A2B 实验模型为 15B total、2.54B activated。表 1 的门控对照训练 400B tokens，sequence length 4096，global batch 1024，100k optimization steps。Baseline average PPL/MMLU 为 6.026/58.79；SDPA elementwise G1 为 5.761/60.82；SDPA headwise G1 为 5.792/60.05。PPL 越低越好，MMLU 越高越好。这些结果来自论文指定模型、数据、训练和评测设置。

表 1 的 headwise G1 added params 数值为 1.6；论文正文也称 15A2B 的 headwise gate 增加少于 2M 参数。论文还报告 gate 引入的 wall-time latency 在其测试设置中少于 2%。该 latency 是作者所测的运行路径结果，不是任意硬件/实现的统一上界；output gate 本身不会跳过此前的 attention 计算。

### Dense 模型训练稳定性

论文在 1.7B dense 模型上报告 SDPA sigmoid gate 降低 loss spikes，并在部分更高学习率的设置中保留训练稳定性。一个 28-layer、400B-token、batch 1024、maximum LR $4\times10^{-3}$ 的比较中，baseline 的 PPL/MMLU 是 7.499/50.21，elementwise gate 是 7.404/51.15。3.5T-token 组使用 batch 2048、maximum LR $4.5\times10^{-3}$。表 2 还包含增加层数、batch 与 maximum LR 的消融；相应训练条件和 baseline 是否 divergence 均应随主张说明。结果支持论文所报配置中的稳定性观察，不足以证明 G1 对所有规模、优化器和训练数据都能避免发散。

### Attention sink 与 massive activation 是分离测量

论文 Figure 2 报告测试数据上的首 token 平均 attention mass 从 baseline 46.7% 降至 elementwise G1 的 4.8%；layer 21 从 83% 降至 4%。Table 4 对 15A2B 模型分别报告 mean maximum hidden activation 与 first-token attention mass：

| Variant | Mean maximum activation | First-token attention mass |
| --- | ---: | ---: |
| Baseline | 1053 | 0.467 |
| Value elementwise G2 | 125 | 0.297 |
| SDPA head-shared G1 | 286 | 0.301 |
| SDPA elementwise G1 | 94 | 0.048 |

G2 降低 maximum activation、但仍保留较高 sink mass，显示两者不是同一指标，也不应互相代替。论文关于 input-dependent sparsity、massive activation 与 sink 的关联属于其机制分析；正文应把作者观察与普遍因果定律区分开。

### 长上下文 RULER 设置

原论文从 3.5T-token 训练模型出发，将 RoPE base 从 10k 提升到 1M，在 sequence length 32K 上继续训练 80B tokens，然后使用 YaRN 扩至 128K，按 RULER 评估。YaRN 扩展后的 64K 分数为 baseline 37.51、SDPA gate 66.60；128K 为 31.65 与 58.82。扩展后 32K 分数也下降：baseline 79.50 至 37.94，gate 79.77 至 72.88。结果比较的是包含 RoPE 改动、80B-token 继续训练和 YaRN 的完整配置，不能归因于 gate 单因素，也不证明其不经训练即可支持任意长度。

### 后续研究与证据边界

Rizwan、Abdus Salam 与 Uddin 的 [SinkProbe arXiv v2](https://arxiv.org/abs/2609.08574v2) 同时包括短序列受控训练模型与对 released Kimi K3 的独立长上下文诊断。受控模型在长度 96 上训练、评估到 768；其 softmax+gate 对照的 sink mass 从 0.316 到 0.261，差异处于作者报告的 seed spread 内。该模型的 gate mean 为 0.686，作者以此解释 gate 未学出原论文中约 0.116 的稀疏分布。短序列受控结果不能作为与 Qiu 15B/1.7B 设置等尺度的复现或反证；Kimi K3 diagnostics 也不是隔离 G1 因果效果的同模型训练消融。

Kim、Shin 与 Park 的 [Attention Sinks as Spectral Spikes arXiv v1](https://arxiv.org/abs/2611.20403v1)（arXiv comments 列明 NeurIPS 2026）提出 sink-induced spectral modes，并分析 post-SDPA gate 对 forward spectral energy 和 value-path gradient aggregation 的影响。其局部 theorem 以固定 attention weights 与 upstream gradients 为条件，说明 gate 对 sink column 的 value-path gradient contributions 可产生收缩；这是条件化局部结论，不是训练后网络所有梯度都必然减小的定理。文中的谱-spike 解释也含理想化随机矩阵假设。

## 来源与证据边界

| 来源 | 本轮查阅内容 | 可支持 | 边界 |
| --- | --- | --- | --- |
| Qiu et al., [Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free, NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html) 与[最终版 PDF](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf) | 方法 §2；15A2B 表 1；dense 表 2；机制分析 §§4.1—4.4；limitations/conclusion | G1 公式、G1—G5、15A2B/dense 设置、gate 参数/延迟、sink/activation 及 RULER 数值 | 只支持论文报告的模型、数据、训练、评测和运行设置；作者未提供 attention-sink 导致长上下文结果的完整理论证明 |
| Sebastian Raschka, [Gated Attention](https://sebastianraschka.com/llm-architecture-gallery/gated-attention/) | “The gate after attention”及 Qwen3-Next 说明 | 适合教学的 SDPA→gate→output projection 表达及模型布局说明 | 教学/架构解释来源；实验数值回到 Qiu 论文，模型事实使用固定 revision |
| Qwen, [Qwen3-Next-80B-A3B-Instruct model card, revision c7646fbcc61b04c51c1dacf7abf1d2e4df785524](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/c7646fbcc61b04c51c1dacf7abf1d2e4df785524/README.md) | Model Overview | 48 层、12 组 3:1 GDN/Gated Attention 布局、80B total/3B activated 和 gated attention 配置 | 指定 revision 的模型卡披露；不支持把整体效率归给 output gate |
| Rizwan et al., [Do New Attention Mechanisms Actually Fix Attention Sinks at Million-Token Context?, arXiv v2](https://arxiv.org/abs/2609.08574v2) | 方法、controlled short-sequence results、seed spread、gate score、limitations | 该小型受控训练比较和独立 Kimi K3 diagnostics 的设置与作者报告 | 不作为 Qiu 规模的反证；小模型、短训练/长度及其 stand-in 有清楚限制 |
| Kim et al., [Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention, arXiv v1](https://arxiv.org/abs/2611.20403v1) | abstract、条件化梯度分析及谱解释 | 固定 attention/upstream gradients 时的局部 value-path gradient contraction；作者报告的 spectral diagnostics | arXiv v1 comments 列明 NeurIPS 2026；局部条件结果和理想化谱模型不等于全网通用梯度保证 |

检索词包括 “Gated Attention G1 SDPA output gate”, “G1 G2 headwise elementwise”, “attention sink massive activation Gated Attention”, “Gated Attention RULER YaRN 128k”, “Gated Attention explained tutorial”, “Qwen3-Next model card”, “attention sink gate pilot”, “value-path gradient gated attention”。研究解释来源用于确认清晰的机制组织方式；事实和实验主张回到论文、官方固定 revision 或对应预印本原文。

