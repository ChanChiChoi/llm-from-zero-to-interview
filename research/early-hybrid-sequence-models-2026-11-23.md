# Mega、Gated State Space 与 H3 的早期混合设计

研究日期：2026-11-23
研究范围：Mega、GSS 与 H3 的原始算子、相互关系及论文实验；重点区分 EMA、状态空间卷积、输入投影门控和保留 attention 层分别承担的作用。本文只记录联网发现的技术知识与来源证据，不记录章节编辑过程。

## 研究结论

1. **Mega 把 EMA 纳入 gated attention 的上下文路径。** 多维 damped EMA 对每个输入通道扩维后递归更新状态，再投影回该通道；EMA 输出构成共享表示，用于生成 query/key、reset gate、update gate 和候选表征，而 value 从原输入投影。候选表征将 EMA 上下文与 reset-gate 调制后的 attention 输出合并；update gate 再在候选表征与原始输入之间插值。因此 Mega 不是“独立 EMA 分支与 attention 输出二选一”的简单加权式。Mega 论文还提出以误差函数构造的 Laplace attention function，作为平方 ReLU attention 的有界替代选择；它与 EMA 的时间结构偏置是不同设计。完整 Mega 保留全长 attention；Mega-chunk 将 attention 限制在固定长度的分块内，注意力项变成 $O(Lc)$，固定 chunk 长度 $c$ 时随序列长度线性增长。贯穿整序列的 EMA 能将分块边界外的上下文压缩到输入表示，但不等价于跨块任意 query-key 寻址。
2. **GSS 用门控缩小 DSS 上下文计算的通道宽度。** 给定输入 $X$，两路非线性投影 $U,V$；DSS 只作用于宽度较小的 $U$，再投影到 $V$ 的宽度，与 $V$ 做逐点乘法并映射回模型维度。其数据依赖来自流间乘法门控，DSS 的状态参数仍为共享的线性时不变结构；不应把它写成每 token 动态产生 $A_t,B_t,C_t$ 的 selective SSM。作者另测了每四层插入 Transformer block 的 GSS-Hybrid。
3. **H3 从两类合成语言任务中提出两类 SSM。** Induction Head 任务要求模型记住特定事件之后的 token；Associative Recall 要求在多组 key-value 关联中按末尾 query 取值。H3 将输入投影为 $Q,K,V$，以 shift SSM 让 key 路径保留前序 token，再将其与 value 做乘法交互；diagonal SSM 累积更长程的键值交互，query 路径对状态进行比较与读取。单一 head 维度为 1 时论文写为 $Q\odot\mathrm{SSM}_{\mathrm{diag}}(\mathrm{SSM}_{\mathrm{shift}}(K)\odot V)$；多 head 情形对 key/value 做外积再由 query 读取。这里的输入依赖来自乘法交互，不意味着状态转移矩阵逐 token 动态变化。
4. **三者的混合位置不同。** Mega 在 attention 子层内部融合 EMA；GSS 在状态空间上下文流外接乘法门控，另有插入 Transformer block 的 hybrid 变体；H3 在两个 SSM 与乘法交互构成的层外，可再保留少量标准 attention。Hyena 原论文把 GSS 与 H3 分别视作其特定滤波器参数化下的 Hyena-1/Hyena-2；这是算子级关系，不表示完整 block、训练配方或实验可以互换。
5. **结果绑定各论文设置。** Mega 在 LRA 六项任务（输入长度约 1K—16K）报告平均准确率 88.21；其模型规模与 S4-v1 近似并对每项任务取 5 个随机种子平均，Mega-chunk 的 chunk size 为 128，Path-X 为 4096；S4-v2 使用更大的模型和更好的超参数。GSS 在 PG19 固定参数比较中，约 192M 参数的 GSS 为 5.3 steps/s、4K 评测 PPL 12.84；约 209M 的 DSS 为 1.8 steps/s、PPL 13.53。非基线模型采用 64 TPUv4 cores、125K 训练 steps 和 4K 训练序列；GSS 初始化在 PG19 上重调，DSS 沿用其默认初始化，因此长度外推比较并非完全匹配。
6. **H3 的机制证据与语言模型证据应分开。** 两层合成任务中，按 Induction Head（长度 30、词表 20）与 Associative Recall（长度 20、词表 10）的顺序，H3 accuracy 为 100.0/99.8，S4D 为 35.6/86.0，GSS 为 6.8/78.0，attention 两项均为 100.0。论文还报告 125M Hybrid H3 在 Pile 的 PPL 8.8、GPT-Neo-125M 为 9.4；Hybrid H3 模型在 Pile 上训练 400B tokens，而公开 baseline checkpoint 的训练 recipe 不完全等价。推理测试的最高约 2.4x 是 1.3B 模型、A100 80GB、batch 64、prompt 512/1024/1536、每序列生成 128 tokens 的指定设置；该报告还依赖 H3 的 recurrent 路径和系统实现，不能外推为架构普遍加速。

## 来源与证据边界

| 来源 | 类型、版本与阅读位置 | 可支持 | 不单独支持 |
| --- | --- | --- | --- |
| [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655v3)，Ma et al.，ICLR 2023，v3（2023-01-28） | 原始论文；§3.1—3.5 的多维 damped EMA、gated attention、block 与 chunk 变体；§4.1、§4.3 的 LRA 与语言建模结果 | Mega 的运算结构、Mega-chunk 复杂度及指定任务/配置结果 | 任意长序列、任意硬件或任意模型配方的质量/速度保证 |
| [Long Range Language Modeling via Gated State Spaces](https://arxiv.org/abs/2206.13947v3)，Mehta et al.，v3（2022-07-02） | 原始论文；Fig. 1、§3.2 的 GSS layer；§4.1—4.2 的 DSS/GSS 和 GSS-Hybrid 实验 | GSS 的两路投影、DSS、逐点乘法；指定 TPUv4 与语言模型实验 | Mamba 式输入依赖状态转移或通用部署吞吐结论 |
| [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](https://arxiv.org/abs/2212.14052v3)，Fu et al.，ICLR 2023 spotlight，v3（2023-04-29） | 原始论文；§3.1—3.3 的合成任务、H3 与表达性；§5.1 的语言模型/推理数据；§6 的 FlashConv | H3 的 shift/diagonal SSM、乘法交互和各自实验配置 | 合成任务成功即代表一般 ICL；硬件速度可只由 H3 架构单独造成 |
| [H3: Language Modeling with State Space Models and (Almost) No Attention](https://hazyresearch.stanford.edu/blog/2023-01-20-h3)，Hazy Research（2023-01-23） | 方法作者团队的解释文章；associate recall 示例、两类记忆/比较能力的教学路线 | 作者对机制动机和适合入门的组织解释 | 替代 camera-ready 论文版本的精确指标与设置 |
| [Hyena Hierarchy](https://proceedings.mlr.press/v202/poli23a.html)，Poli et al.，ICML 2023 | 原始论文；§3 的算子关系；既有 [Hyena 研究记录](hyena-implicit-long-convolution-2026-11-23.md)保留详细定位 | GSS/H3 在指定滤波器参数化下与 Hyena 阶算子的关系 | 完整架构、训练配方、实验结果同一 |

研究未运行代码、模型、API 或 benchmark；实验结论均为 `source-backed`，不是本地复现。
