# Transformer 架构方向的新证据（2026）

检索日期：2026-10-30；补充检索：2026-11-06

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

## 稀疏专家负载均衡

- 来源：[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)，arXiv v2（2025-02-18）。
- 阅读位置：§2.2 的 DeepSeekMoE 与辅助损失说明；模型规模和激活参数说明。
- 支持事实：报告的 DeepSeek-V3 有 671B 总参数、每个 token 激活约 37B 参数。作者采用无辅助损失的负载均衡策略，通过按 batch 专家负载更新路由 bias 来调节专家使用，同时保留 sequence-wise auxiliary loss 以降低单序列内部的极端失衡风险。
- 证据等级：原始技术报告的模型与训练方案说明；参数和策略为作者披露。
- 边界：参数规模、路由配置和训练报告属于 DeepSeek-V3 特定模型与数据、集群设置；不证明辅助损失在其他 MoE 中无用，也不说明 bias 更新可脱离 capacity、expert placement 或 token dropping 单独保证均衡。本记录不将作者方案转写成通用默认配方。

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

## 深度聚合、递归状态与压缩注意力

- 来源：[Attention Residuals](https://arxiv.org/abs/2603.15031)，arXiv 原始论文；2026-03。
- 阅读位置：摘要、引言、AttnRes 与 Block AttnRes 方法说明。
- 支持事实：AttnRes 用 softmax 权重从先前层表示中构造深度方向聚合；Block AttnRes 先在块内累加，再选择块表示，以降低保存历史表示和跨流水线阶段通信的负担。论文报告将方法用于 Kimi Linear，并给出其模型与训练规模。
- 证据等级：原始预印本的方法说明和作者实验。
- 边界：论文的模型规模、训练条件及收益属于作者配置；不能据此推导 token 维 KV cache 下降，也不能把结果直接移植到其他硬件或模型。

- 来源：[DeepCrossAttention](https://proceedings.mlr.press/v267/heddes25a.html)，ICML 2025 官方论文页。
- 阅读位置：官方出版页摘要。
- 支持事实：DCA 使用输入相关权重和 depth-wise cross-attention 重组历史层输出；论文摘要报告其语言建模实验在相同质量目标下最高达到 3 倍速度。
- 证据等级：同行评审会议论文的官方页面和摘要。
- 边界：本记录未复核完整实验表；“最高 3 倍”绑定论文模型、训练时间、基线和测量口径，不是通用训练加速保证。

- 来源：[Kimi Linear v2](https://arxiv.org/html/2510.26692v2)，arXiv 原始论文。
- 阅读位置：Kimi Delta Attention 方法与 Kimi Linear hybrid 架构章节；长上下文实验设置和结果切片。
- 支持事实：KDA 对递归状态使用通道级衰减和 delta correction；Kimi Linear 将 KDA 路径与全局 MLA 结合。论文的模型级比较覆盖 RULER、RepoQA、LongBench V2 等指定任务。
- 证据等级：原始预印本的方法说明和作者实验。
- 边界：模型级结果不能视为只替换 KDA 算子的消融，也不能外推到所有任务、Kimi K3 的层配置或生产服务表现。

- 来源：[DeepSeek V4 Technical Report](https://arxiv.org/html/2606.19348)，arXiv 技术报告。
- 阅读位置：注意力机制概述、CSA/HCA 说明与 serving cache 章节。
- 支持事实：CSA 压缩历史 KV 后进行稀疏选择；HCA 采取更强压缩并对压缩表示进行 dense attention。报告还分别管理压缩 KV、索引状态、滑动窗口和未完成尾部等状态。
- 证据等级：原始技术报告。
- 边界：模型级 FLOPs、cache 和 benchmark 数字是完整系统组合结果，不能单独归因给 CSA/HCA；压缩精度、索引召回和 cache 生命周期应分别评测。

- 来源：[mHC v2](https://arxiv.org/html/2512.24880v2)，arXiv 原始论文；前序方法见 [Hyper-Connections](https://arxiv.org/abs/2409.19606)。
- 阅读位置：mHC 的双随机映射、Sinkhorn-Knopp 近似与大规模实验段落。
- 支持事实：mHC 将多流残差映射约束在双随机矩阵集合，并使用有限轮 Sinkhorn-Knopp 迭代近似投影。论文报告在 `n=4` 的特定大规模训练设置下增加 6.7% wall-time。
- 证据等级：原始预印本的方法说明和作者实验。
- 边界：6.7% 属于论文的模型、基础设施和实现；双随机约束只作用于残差流线性映射，不保证整个非线性网络的范数恒定，也不代表通用开销。

## 架构论文线：注意力压缩、递归状态与混合设计

### Multi-head Latent Attention

- 来源：[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)，arXiv v5（2024-06-19）。
- 阅读位置：摘要、§2.1.2—§2.1.4、附录 D.1—D.2；缓存比较的整体系统数字见摘要。
- 支持事实：MLA 将 key/value 联合投影到低秩 latent $c_t^{KV}$，推理时缓存该 latent 与解耦 RoPE key；论文给出的缓存量为 $(d_c+d_h^R)l$。在 DeepSeek-V2 配置中 $d_c=4d_h$、$d_h^R=d_h/2$，相当于 2.25 个 GQA 组。附录 D.2 报告小模型与大模型两组 MLA 变体的 KV 缓存分别是对应 MHA 变体的约 14% 与 4%，并在四项 hard benchmarks 上得分更高。
- 证据等级：原始技术报告的机制说明及作者控制的两尺度比较。
- 边界：各尺度变体的激活参数不同，结果只覆盖所列模型、token 预算和基准；摘要中相对 DeepSeek 67B 的 93.3% KV-cache 降幅是完整系统对比，不能作为 MLA 的单独因果效应或任意 serving 加速保证。

### Low-Rank Key Value Attention（LRKV）

- 来源：[Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471)，arXiv 论文（2026-01-16）。
- 阅读位置：摘要、引言、方法中的 KV cache 公式及实验设置/结果段。
- 支持事实：LRKV 使用共享的完整 KV 投影作为基础，再为每个 attention head 学习低秩残差；论文给出的每 token KV 数量为 `2(d_h+H r)`，标准 MHA 为 `2H d_h`。作者在 128M—6.3B 参数的预训练实验中报告，LRKV 使用标准 MHA 约 45%—53% 的 KV cache，并在其测试设置中取得较低的 held-out cross-entropy。
- 证据等级：原始论文的方法说明和作者实验。
- 边界：实验规模、数据、rank、模型实现和比较基线限于论文设置；公式是元素数估算，不包含 dtype、元数据、kernel 与系统开销，论文结果不代表任意模型或 serving workload 的质量和延迟收益。

### Multi-Head Low-Rank Attention（MLRA）

- 来源：[Multi-Head Low-Rank Attention](https://proceedings.iclr.cc/paper_files/paper/2026/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html)，ICLR 2026；[论文全文](https://arxiv.org/abs/2603.02188)。
- 阅读位置：摘要、MLA 背景、MLRA-4 方法和实验设置/结果。
- 支持事实：MLRA 将 MLA 的单个低秩 latent 分成四个可分片 latent head，使方法可支持 4 路 tensor parallel decode；作者在 2.9B 模型设置中报告了 perplexity、常识任务和长上下文解码结果。
- 证据等级：ICLR 同行评审论文的方法说明和作者实验。
- 边界：关键比较限于论文所述 2.9B 规模、4 路并行、GPU、kernel 和评测；不能据此推断更大模型、其他并行拓扑或生产 serving 的收益。

### Structured State Space Duality / Mamba-2

- 来源：[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/html/2405.21060)，ICML 2024 论文。
- 阅读位置：摘要、§3.2—§3.4、§5.1—§5.3 和 §6 的 SSD 及分块算法说明。
- 支持事实：论文将状态大小为 $N$ 的结构化状态空间序列变换表示为 $N$-semiseparable 矩阵，并证明特定 structured masked attention 与特定 SSM 存在对偶表示；同一运算可采用递归线性形式或矩阵二次形式。作者报告 Mamba-2 核心层在其实现和比较设置中比 Mamba-1 快 2—8 倍。
- 证据等级：同行评审论文中的形式化、算法与作者效率实验。
- 边界：对偶关系针对明确的结构化子类，不等于标准 softmax attention 与所有 SSM 等价；2—8 倍是论文核心层结果，不代表完整模型训练、不同 GPU 或生产 serving 的加速比。

### Jamba 的层间混合

- 来源：[Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/html/2403.19887v2)，arXiv v2（2024-07-03）。
- 阅读位置：摘要、混合层比例实验和纯 Attention/Mamba 对照表（§4 实验）。
- 支持事实：模型交错使用 Transformer 与 Mamba 层，并在部分层加入 MoE。作者在 1.3B 参数、250B 训练 token 下比较 1:3 与 1:7 的 Attention:Mamba 比例，报告性能差别不大并选择 1:7 进行较大规模实验；7B、50B token 的一个对照中，混合模型优于该论文里的纯 attention 与纯 Mamba 变体。
- 证据等级：原始预印本的架构说明和作者实验。
- 边界：层比例结论受模型规模、数据、训练预算和实现限制；不是可移植的固定最优比例。

### Native Sparse Attention

- 来源：[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/html/2502.11089v2)，arXiv v2（2025-02-27）。
- 阅读位置：方法概述、长上下文评估、§5.1 训练速度、§5.2 解码速度。
- 支持事实：NSA 使用压缩 token、选择 token 与局部窗口组成层级稀疏路径，同时设计原生训练与硬件对齐算子。作者在 8×A100 设置中报告 64K 序列下训练前向最高 9.0×、反向最高 6.0×速度提升；解码最高 11.6×是其基于 KV 访问量给出的最高报告，论文也报告对应延迟优势。效率配置包括 GQA group 4、16 个 group 内 heads、$d_k=192$、$d_v=128$。
- 证据等级：原始预印本的方法、指定任务实验和作者 GPU/kernel 测量。
- 边界：速度与模型、8×A100、Triton 实现、基线与长度相关；LongBench 基线统一激活 token 预算，论文还排除部分子集。合成检索或单一长度结果不能推出一般长文档能力或其他 runtime 的收益。

### Gated Delta Networks 与 Mamba-3

- 来源：[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)，arXiv 预印本（2024-12-09）；[Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)，arXiv 预印本（2026-03-16）。
- 阅读位置：Gated Delta Networks 摘要及 §4 的真实世界回忆和长上下文结果；Mamba-3 摘要、§§3.1—3.3、§§4.1—4.4 与表 4—7。
- 支持事实：Gated DeltaNet 将自适应遗忘门与 delta residual 写入结合，并提出分块并行训练；论文指出纯递归模型在真实回忆任务上仍与 Transformer 有差距。Mamba-3 组合指数梯形离散化、复值状态和 MIMO；作者在 1.5B 语言模型评估中报告相对 Gated DeltaNet 平均下游准确率 +0.6 个百分点，MIMO 版本再 +1.2 个百分点。在 440M、FineWeb-Edu、2× Chinchilla token 预算的 state-size 曲线上，作者报告 Mamba-3 用约一半状态维度达到相近 perplexity；kernel 对照使用单张 H100、batch 128 和论文指定 DSL。
- 证据等级：原始预印本的算法说明、任务实验和作者 kernel 对照。
- 边界：不同论文的模型、数据和 harness 不同，不横向合并分数；Mamba-3 的真实数据检索切片对半结构化/非结构化抽取仍有弱项，状态维数 proxy 不等于目标服务的延迟；H100 kernel 对照不等同于生产 P99。
