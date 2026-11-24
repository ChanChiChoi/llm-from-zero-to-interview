# 第 73 章 Gated Attention：SDPA 输出后的输入依赖门控

本章专指 Qiu 等人于 NeurIPS 2025 系统比较的 softmax Gated Attention，重点是 head-specific sigmoid gate G1 在 SDPA 输出后、输出投影前的作用。它改变 attention 聚合后输出特征的映射，不承担显式 attention 与递归/局部路径之间的路由。读者应能区分 attention weights 与 output gate、说明位置/粒度/共享方式如何影响配置，并按原论文实验设置判断质量、稳定性、attention sink 和长上下文结果。第 62 章讲 Gated DeltaNet 的状态更新，第 63 章讲 Gated MLA，第 65 章讲 hybrid attention 路径，第 74 章讲压缩历史读取；本章不重复这些机制，也不讨论 speculative decode、cache 回滚或通用 serving 运维。

## 73.1 Gated Attention 出现在 SDPA 的哪个位置

本节确定方法对象和边界：标准 SDPA 先用 query-key scores 产生归一化权重，再将可见 value 混成每个 query/head 的向量；G1 使用当前 query 位置的 pre-normalization hidden state 产生 sigmoid gate，调节 SDPA 结果，然后再进入多头输出投影。应明确 gate 的索引位于 query/head/feature 轴，attention weight 位于 query/source-token 轴；固定 Q/K/V 的当前层前向中，G1 调节输出而不改写 softmax 权重。

### 73.1.1 标准 SDPA 把值向量混成每个 query 的输出

围绕 scaled dot-product score、mask 后的 softmax 权重 $\alpha_{ij}^h$ 及聚合向量 $a_i^h=\sum_j\alpha_{ij}^h v_j^h$，界定 query、可见 key/value、head 输出和归一化权重之间的关系。中心在于每个头输出是多个 value 的加权特征组合，而不是一个来源 token 的身份或离散选择；这是辨认 output gate 作用对象所需的 attention 前置。

### 73.1.2 G1 在 SDPA 后、输出投影前逐元素调节结果

本节聚焦论文中的 G1：由 pre-normalization hidden state $x_i$ 产生 head-specific sigmoid gate $g_i^h=\sigma(x_iW_g^h)$，逐元素调节已聚合的 $a_i^h$，再通过输出投影 $W_O$。关键是非线性位于聚合结果与输出投影之间，调节的是 head feature vector；论文统一表示 $Y'=Y\odot\sigma(XW_\theta)$，应与 Raschka 的教学表达及原论文定义交叉核对。

### 73.1.3 gate 调节注意力结果，不负责选择另一条计算路径

本节限定 G1 的机制含义：它按 query/head 调节聚合结果的特征分量，不在 value 来源间重新分配 softmax 权重，也不在显式 attention 与递归、局部或其他分支间选择。连续输出被压低不代表 attention kernel 已跳过执行；固定一层的 Q/K/V 时，当前 softmax map 不会因 G1 后乘而改变。

## 73.2 位置、粒度与共享方式共同定义 gate

本节整理论文的消融维度，避免将门控位置、shape、跨 head sharing 和非线性形式合并成同一种机制。重点术语包括 G1—G5、head-specific/head-shared、headwise/elementwise、multiplicative/additive、sigmoid/SiLU，以及 value/key/query projections；不同位置会改变 gate 所操作的表示和它与 attention 聚合的关系。

### 73.2.1 G1—G5 放在注意力计算的不同阶段

本节区分 G1（SDPA 输出）、G2（value projection 输出）、G3（key projection 输出）、G4（query projection 输出）和 G5（最终多头输出投影后）。这些位置分别关系到聚合结果、被加权的 value、query/key score inputs 或最终 block 输出；论文中 G1 与 G2 形成投影间非线性的主要比较，G5 则不能作为 G1 的等价替换。

### 73.2.2 head-specific/shared 和 headwise/elementwise 是两组独立选择

本节区分两条配置轴：head-specific/shared 决定各注意力头是否独立生成或共用 gate，headwise/elementwise 决定 gate 是每头一个标量还是每维一个值。研究抓手是 gate score 的张量 shape、各 head 的独立性及对照结果；这些术语不能与 GQA 的 KV head sharing 混同。

### 73.2.3 乘法 sigmoid、加法与激活变体不可混作同一设置

本节比较乘法 gate $Y\odot\sigma(\cdot)$、加法 gate $Y+\sigma(\cdot)$ 及论文测试的激活函数和 non-sparse sigmoid 变体。sigmoid 的有界缩放与 SiLU 等非线性在数值范围和稀疏倾向上不同，论文结果显示它们并非可互换配置；主结论应绑定被测的 head-specific multiplicative sigmoid G1。

## 73.3 为什么论文偏向 G1

本节梳理论文对 G1/G2 差异的机制解释：固定 attention weights 后，value projection 与输出子投影的复合映射秩受 head dimension 限制；在两线性映射之间加入输入依赖非线性可能提高表达能力。G1 与 G2 都能插入非线性，但一个依赖 query 并在聚合后调节，另一个依赖来源 token 并在聚合前调节；解释不得扩大为 gate 已实施语义证据检索。

### 73.3.1 value 与输出投影之间的低秩复合限制 attention 输出映射

本节围绕每个头的 value/output 映射 $W_V^hW_O^h$ 及其秩至多为 $d_k$ 的性质展开，并说明 $d_k<d_{\mathrm{model}}$ 与 GQA 共享 value projection 的关系。该低秩描述针对固定 attention weights 下的线性 value/output 子映射，不代表整个含 query-key softmax 的 attention block 都是低秩；论文第 4.1 节提供这一解释及相应公式。

### 73.3.2 G1 把 query-dependent 非线性放在加权聚合之后

本节说明 G1 的 gate 由当前 query hidden state $x_i$ 产生，在 $a_i^h$ 已汇总来源 value 后调节其特征，并位于输出投影之前。由此它在 value 与 output projection 之间增加非线性，同时把 gate 变成 query-dependent；这与 G2 按来源表示调节 value、再由 attention 权重做聚合的关系不同。

### 73.3.3 G1、G2 的效果差别涉及查询依赖与稀疏性

本节连接 15A2B 实验里 G1/G2 的差异与论文的 gate-score 分析：G1 的分数来自 query state，G2 来自 key/value 位置，论文把 G1 更强的结果与 query-dependency 和输入依赖稀疏联系起来。结论应保持为作者在指定配置上的实证解释；gate score 稀疏与 token-level evidence selection 是不同主张。

## 73.4 论文实验报告的质量、稳定性和额外成本

本节把质量、训练稳定性和计算代价分成各自有条件的证据。15A2B MoE ablation、1.7B dense experiments 与 wall-time latency 使用不同设置；应标明模型、tokens、sequence length、batch、学习率和指标，避免从一组结果推出普遍架构保证。

### 73.4.1 15A2B 消融显示 G1 改善 PPL 与基准结果

本节依据论文表 1 比较 baseline、elementwise G1 和 headwise G1 的 average PPL 与 MMLU。数值属于 15B total、2.54B active 的 15A2B MoE，在 400B training tokens、sequence length 4096、global batch 1024 和 100k optimization steps 条件下；读者目标是说明这些指标在该对照中如何变化，而非归纳为所有任务上的固定增益。

### 73.4.2 Dense 实验报告更稳定训练与较大的学习率容忍度

本节聚焦 1.7B dense 模型的作者报告，覆盖 loss spikes、不同训练 token/batch/最大学习率的配置比较，以及 gate 在 baseline 较不稳定设置中的表现。需区分作者在固定实验里的 stability observation、参数量匹配时对 FFN width 的调整和跨模型/优化器的普遍结论；文中不应把“加 gate”写成无条件的训练稳定方案。

### 73.4.3 参数与延迟成本属于论文的实验配置

本节分开说明 gate 参数数、额外计算和 wall-time latency。论文报告 headwise G1 少于 2M additional parameters，且门控 latency 在其实验设置中小于 2%；这些数字不覆盖任意实现、硬件和 workload。由于 G1 后乘不取消前面的 QK、softmax 与 value aggregation，它不自动减少 attention 的渐近计算量或 KV cache。

## 73.5 Attention sink 与长上下文证据的边界

本节区分当前前向的代数机制、训练后 attention map 的变化、hidden activation 统计与位置扩展 benchmark。Qiu 等人的 sink 结果属于训练模型观测；RULER 外推还包含 RoPE 设置、继续训练与 YaRN，不能单独归因 G1。

### 73.5.1 G1 训练后减少首 token attention mass

本节说明论文在指定 15B 模型/test data 上报告的 first-token attention mass 变化，并区分整体层平均和个别层值。中心关系是：在固定 Q/K/V 的当前 SDPA 计算中，G1 后乘不改变该次 softmax weights；论文报告的 sink reduction 是训练后模型的 attention map 观察，不能称作 G1 在代数上直接删掉了某个 attention column。

### 73.5.2 Massive activation 与 attention sink 是关联而非同一指标

本节对照论文的 mean maximum hidden activation 与 first-token attention mass，利用 G2、head-shared G1 和 head-specific elementwise G1 的分化结果说明两种测量不可互换。即使 gate 降低某项 hidden-state magnitude，也不能仅凭该项断言 sink 或任务读取质量同步改善；指标须按各自定义及表 4 设置解释。

### 73.5.3 RULER 外推结果绑定 32K 继续训练和 YaRN 设置

本节交代长上下文实验的条件链：基于 3.5T-token 模型，RoPE base 从 10k 调至 1M，在 32K sequence length 上继续训练 80B tokens，再使用 YaRN 扩至 128K 并在 RULER 上评估。64K/128K 结果及扩展后 32K 区间的退化都要与这套过程绑定；不能把它们写成 output gate 单独创造长上下文能力。

## 73.6 Qwen3-Next 把 output-gated SDPA 放进 3:1 hybrid stack

本节使用 Qwen3-Next 固定模型卡说明原论文中的 gated softmax attention 如何出现在 Gated DeltaNet 与 MoE 交替的实际架构中。案例职责是核对层布局及术语，不替模型卡或研究论文做单因素性能归因；完整 hybrid 路径取舍由第 65 章主讲。

### 73.6.1 48 层周期布局包含 36 个 GDN 与 12 个 gated attention

本节依据指定 revision 的 Qwen3-Next-80B-A3B-Instruct 模型卡，说明 48 层由 12 组“三层 Gated DeltaNet 加一层 Gated Attention”构成，并记录 80B total、3B activated 的卡片口径。相关配置应绑定模型卡版本；层数比例描述的是架构布局，不是 gate 单独产生的性能证据。

### 73.6.2 混合效率不能归结为 gate 单独节省 attention 计算

本节把 output gate 与 hybrid schedule 的计算作用分开：Gated Attention 层仍执行显式 attention，G1 不代表动态 kernel skipping；多数层使用 GDN、MoE 和实际执行实现共同决定系统资源结果。不得从模型卡整体长上下文/throughput 主张反推出输出 gate 单独降低 KV 或 attention FLOPs。

## 73.7 使用门控证据时不能跨越的推断

本节界定本章中不同可观测量的因果边界，并将前沿研究作为有条件的机制证据。gate 值、attention weights、训练后分布、任务正确率和实际 kernel 工作量需分开记录；实验比较应保持训练与任务条件可比。

### 73.7.1 gate 输出稀疏不等于 softmax token sparse

本节区分按输出 feature 维度的 G1 gate sparsity 与按来源 token 维度的 attention sparsity。sigmoid gate 集中在零附近表示输出特征被缩放，不代表 softmax 权重矩阵被 top-k、mask 或删除 token，也不意味着计算图省略相应位置的 QK/value 运算。

### 73.7.2 gate 值和 attention weight 都不能单独证明证据选择

本节说明 query/head/feature gate 与 query/source-token attention weight 具有不同索引和作用对象。二者的可视化或相关性不能独立证明模型利用了正确证据；因果解释需要与可控证据干预、对照模型和任务结果相结合，且不把中间量直接当作人类语义重要性。

### 73.7.3 2026 后续研究扩展机制解释但未解决大规模外推

本节覆盖两类后续证据：2026 年 9 月 SinkProbe v2 的受控短序列训练及 Kimi K3 长上下文诊断，指出其 gate 对照未学出稀疏分数、不能作为等规模反证；2026 年 11 月 arXiv v1 的谱模式与固定 attention/upstream-gradient 条件下的局部 value-path gradient contraction。后者的局部条件结论不等于所有训练梯度普遍收缩，也不能替代大模型上的直接机制验证。

### 73.7.4 按相同训练/任务对齐比较 gate 位置与计算收益

本节归纳可比较的研究问题：在相同模型规模、训练数据与预算下分别改变 gate position、head sharing、granularity 和激活形式；同步测量质量、gate distribution、attention map、activation statistics 与真实执行成本。理论 feature sparsity 与执行稀疏必须分开；只有实施了实际跳过计算的系统并以匹配 workload 测量，才可主张相应的计算收益。

