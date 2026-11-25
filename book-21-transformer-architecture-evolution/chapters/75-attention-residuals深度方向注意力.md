# Attention Residuals：让网络沿深度选择历史表示

Attention Residuals（AttnRes）把问题放在残差流的“深度轴”上：一个子层如何从此前多次计算中取得输入？标准残差将历史增量固定相加；Full AttnRes 为每次读入设置伪查询，并对历史来源做深度 softmax；Block AttnRes 则在块内累加、块间选择，以减少需要长期保存和传输的表示。本章按一次残差写入作为一个深度步骤。熟悉的 Transformer block 通常有 attention 与 FFN 两次写入，因此下文的子层数可能是 block 数的两倍。原始定义与实验来自 [Kimi Team 的 AttnRes 技术报告](https://arxiv.org/pdf/2603.15031)；Kimi K3 的公开配置是一个单独的模型实例，后文会与原论文实验分开讨论。

## 75.1 标准残差把历史计算累积到同一条状态

理解 AttnRes 前，先把“残差流”拆成每次读取与写入。标准加法为 Transformer 提供直接的梯度路径，也让每个子层都能接收此前累积的状态；代价是历史各项没有独立的读入权重。AttnRes 改的是历史怎样汇总到下一次读入，并不替换网络中实际执行的 attention 或 FFN。

### 75.1.1 Transformer block 内有 attention 与 FFN 两次残差写入

一个常见的 PreNorm Transformer block 先规范化当前状态，再执行 attention，并把 attention 的输出加回残差流；随后 FFN 再读取更新后的状态，并做第二次残差写入。用 $a_l$ 表示第 $l$ 个 block 中 attention 的输出增量，用 $m_l$ 表示 FFN 的输出增量，可以写成：

```math
a_l=\mathrm{Attn}(\mathrm{Norm}_1(h_l)),\qquad h'_l=h_l+a_l
```

```math
m_l=\mathrm{FFN}(\mathrm{Norm}_2(h'_l)),\qquad h_{l+1}=h'_l+m_l
```

所以，一个 block 是两次先后发生的残差更新，而不是一次不可分的“层输出”。为了讲清 AttnRes 的来源集合，后文把这两次更新分别编号为深度子层。这个粒度也解释了为什么同一个目标 block 内的 attention 与 FFN 可以有不同的读入结果。

### 75.1.2 固定单位系数让旧增量无法在读入时单独调节

令 $v_0$ 为当前 token 的 embedding；第 $i$ 个残差子层读取 $h_i$，产生输出增量 $v_i=f_i(h_i)$。以 $h_1=v_0$ 作为第一个子层的输入，标准残差的递推与展开为：

```math
h_{i+1}=h_i+v_i\quad(i\ge1),\qquad h_l=v_0+\sum_{i=1}^{l-1}v_i
```

每项历史增量的显式系数都是 $1$。这条恒等路径有助于信息和梯度越过变换直接传递，是标准残差的重要优点；但某个后续子层不能单独把“早期 attention 输出”调高、把“较近 FFN 输出”调低。过去各项一旦进入相加结果，就不再作为可分别寻址的来源留给该子层。AttnRes 把这个受限点作为研究动机，不意味着固定残差在所有模型或任务中都不够用。

### 75.1.3 AttnRes 改的是深度信息汇总，不是 token 注意力或执行层数

序列 self-attention 在同一个网络深度位置上读取不同 token 的表示；AttnRes 则针对一个 token，比较它在此前不同深度步骤产生的表示。前者沿序列位置选择，后者沿网络深度选择。两者可以同时存在：一个 Transformer 子层仍照常执行 token attention，AttnRes 决定它接收怎样汇总的残差输入。

每个 attention 和 FFN 子层仍然会运行并产生输出增量。AttnRes 用权重改变后续子层读到的历史混合，不会自动跳过不重要的层，也不是 early-exit。它同样不扩大 token 的可见上下文；序列上的长程交互仍由模型采用的 self-attention 或递归状态机制负责。

## 75.2 Full AttnRes：以深度 softmax 读取历史来源

Full AttnRes 保留 token embedding 和每个已完成子层的输出增量。目标子层为这些来源打分，再用归一化权重组成自己的输入。关键关系是：同一历史向量既参与“该读多少”的打分，也作为“读到什么”的内容进入加权和；打分路径对它做 RMSNorm，输出路径使用原向量。

### 75.2.1 source bank 包含 embedding 与子层输出增量

令 $v_0$ 表示当前 token 的 embedding，令 $v_i=f_i(h_i)$ 表示第 $i$ 个残差子层计算出的增量。目标子层 $l$ 读入时只能看到 $v_0,\ldots,v_{l-1}$；当前子层尚未产生的 $v_l$ 不在候选中。这样可区分两种容易混为一谈的对象：$h_l$ 是某次运算的输入状态，$v_i$ 是一项可以被单独检索和混合的历史写入。

### 75.2.2 每个读入位置用一个可学习伪查询

每个目标深度位置 $l$ 有一个可学习向量 $w_l$，论文将它作为伪查询 $q_l=w_l$。这个参数在该位置对所有 token 共用，不是从当前 token 的 $h_l$ 经过查询投影得到的动态向量。因此，同一子层位置的 token 使用同一组查询参数；不同目标深度位置则可以学习不同的查询。

静态查询也有工程作用：同一块内多个目标位置的查询在前向过程中已经确定，可以批量计算它们对已完成历史块的注意力。它简化了读入调度，但并不表示每个 token 的历史权重都相同；权重是否随 token 变化，要看来源向量本身。

### 75.2.3 同一历史向量同时作为 key 与 value

AttnRes 把历史增量同时用作 key 和 value，即 $k_i=v_i$。用 $w_l$ 与经 RMSNorm 处理的 $k_i$ 算分，再用未经归一化的 $v_i$ 贡献内容：

```math
s_{i\to l}=w_l^\top \mathrm{RMSNorm}(k_i),\qquad k_i=v_i
```

RMSNorm 作用于匹配分数的路径，降低仅因某个来源幅度较大就让它支配 softmax 的风险；它不表示最后的加权残差输出也要按同样方式归一化。原论文给出的分数是 $w_l^\top\mathrm{RMSNorm}(k_i)$，没有额外除以 $\sqrt{d}$；不要因为它用了 attention 一词，就自动套入序列 self-attention 的所有实现细节。

### 75.2.4 Softmax 沿深度来源归一化

对目标子层 $l$，softmax 在所有已生成的深度来源 $i<l$ 上归一化。于是每个来源的权重非负，且可见来源的权重和为 $1$：

```math
\alpha_{i\to l}=
\frac{\exp(s_{i\to l})}{\sum_{j=0}^{l-1}\exp(s_{j\to l})},
\qquad
h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}v_i
```

这里的归一化轴是历史来源的深度索引 $i$，不是序列 token 位置。虽然 $w_l$ 对 token 共享，$v_i$ 与 $k_i$ 随 token 的中间计算而变，所以不同 token 可以得到不同的 $s_{i\to l}$ 和 $\alpha_{i\to l}$。换句话说，伪查询参数是静态的，路由结果可以依赖输入。

## 75.3 用一个 token 复走 Full AttnRes 的读入

下面用两个 token 的历史向量追踪一次完整的 Full 读取。数值是教学设定，用来逐项复算 RMSNorm、分数、softmax 和 value 聚合，不代表论文实验。为简化手算，忽略 RMSNorm 中很小的数值稳定项。

### 75.3.1 从三条历史来源算出深度权重与混合结果

取一个目标子层的伪查询 $w_l=(1,0)$。对 token A，三个来源值依次为 $v_0=(1,1)$、$v_1=(0,1)$、$v_2=(-1,1)$。它们经 RMSNorm 后的第一维分别为 $1$、$0$ 和 $-1$，因此打分为 $[1,0,-1]$。softmax 的权重约为 $[0.665,0.245,0.090]$：

| 来源 | value $v_i$ | score $s_{i\to l}$ | 权重 $\alpha_{i\to l}$ |
| --- | --- | ---: | ---: |
| $v_0$ | $(1,1)$ | $1$ | $0.665$ |
| $v_1$ | $(0,1)$ | $0$ | $0.245$ |
| $v_2$ | $(-1,1)$ | $-1$ | $0.090$ |

因此目标子层的输入是：

```math
h_l\approx 0.665(1,1)+0.245(0,1)+0.090(-1,1)=(0.575,1.000)
```

这个结果对应三件事：score 表示来源与伪查询的匹配，softmax 把匹配转换成来源间的份额，value 加权和则把历史内容送给目标子层。算式不是“挑中一个层”；softmax 通常会让多个历史来源共同贡献。

### 75.3.2 静态伪查询仍会产生 token 相关的权重

保持同一个 $w_l=(1,0)$，另一个 token B 的三个来源可以是 $(-2,2)$、$(0,2)$、$(3,3)$。它们方向与 token A 的对应来源相同，因此 RMSNorm 后的分数仍分别为 $[-1,0,1]$；但原始 value 的幅度不同。权重随之反转为约 $[0.090,0.245,0.665]$，加权结果约为 $(1.815,2.665)$。

这个对照遵守了 $k_i=v_i$：分数来自同一批向量的归一化版本，输出来自它们的原值。token A 和 B 用的是同一查询参数，但来源向量不同，所以选择权重和最终输入不同。若仅说“查询向量是静态的”，便误以为路由也静态了，就漏掉了输入依赖所在的这条 key/value 路径。

### 75.3.3 零初始化伪查询得到均匀平均，不等于恒等残差初始化

原论文将所有伪查询向量初始化为零。此时对任意来源都有 $s_{i\to l}=0$，于是每个可见来源在初始化时权重相同：

```math
w_l=0\quad\Longrightarrow\quad
\alpha_{i\to l}=\frac{1}{l},\qquad
h_l=\frac{1}{l}\sum_{i=0}^{l-1}v_i
```

这是对来源的均匀平均，不是标准残差的恒等传递。标准残差会把历史增量按单位系数相加；零查询 AttnRes 会把它们按来源数归一化后平均。即使某种讲解或其他变体使用 one-hot 权重来近似恒等初始化，也不能把它写成原论文的 zero-init 设置。[原论文](https://arxiv.org/pdf/2603.15031)明确说明零初始化使初始注意力权重均匀。

## 75.4 Block AttnRes：块间保留摘要，块内保留 partial sum

Full AttnRes 的历史列表随深度增长。Block AttnRes 把连续子层分组：一个块内，新产生的增量先通过普通加法累积成 partial sum；块完成时，这个和成为一个可被后续深度读取的表示。读者需要追踪四个时点：新块的第一次读入、块内后续读入、块边界提交，以及模型最终输出读入。

### 75.4.1 连续子层增量形成一个已完成块表示

设 $B_n$ 是第 $n$ 个块所含的残差子层，$v_j$ 是其中各子层的输出增量。该块完成时保存的表示为 $b_n$；块还未完成时，已产生增量的部分和记作 $p_n$：

```math
b_0=e,\qquad
b_n=\sum_{j\in B_n}v_j,\qquad
p_n^{(r)}=\sum_{j=1}^{r}v_{n,j}
```

这里 $b_0=e$ 单独表示 token embedding；$b_n$ 是一个完成块的增量摘要，$p_n^{(r)}$ 是当前块前 $r$ 次写入的和。第 $n$ 块完成后，将最终 partial $p_n$ 作为 $b_n$ 放入历史来源。块内先前单个 $v_j$ 不再作为独立的跨块来源保留，块内加法正是 Block 版换取较小状态的压缩步骤。

### 75.4.2 首个子层与后续子层看到不同候选集合

块开始时，尚无本块 partial sum，因此第一个子层只能读取 embedding 和之前完成的块。它写入输出增量后，partial sum 才出现；同块下一次读入便可以把 partial 也列为来源。每次读取发生在当前子层生成新增量之前，因此不会把本次输出泄漏回自己的输入。

以两个两次写入的块为例，$u_1$ 至 $u_4$ 分别表示四个子层的输出增量：

| 读入位置 | 本次可读来源 | 写入后发生什么 |
| --- | --- | --- |
| 块 $B_1$ 的 attention | $b_0$ | 写入 $u_1$，得到 partial $p_1^{(1)}=u_1$ |
| 块 $B_1$ 的 FFN | $b_0,p_1^{(1)}$ | 写入 $u_2$，完成 $b_1=u_1+u_2$ |
| 块 $B_2$ 的 attention | $b_0,b_1$ | 写入 $u_3$，得到 partial $p_2^{(1)}=u_3$ |
| 块 $B_2$ 的 FFN | $b_0,b_1,p_2^{(1)}$ | 写入 $u_4$，完成 $b_2=u_3+u_4$ |

这个表给出 Block AttnRes 的完整候选来源与更新顺序：块间读旧摘要，块内后续读入再看到当前 partial。实际模型的块宽和层数可以不同，来源可见性的规则不变。

### 75.4.3 Attention 与 FFN 分别读取，输出头也需最终聚合

Attention 子层和 FFN 子层是两个深度读入位置，各有自己的可学习伪查询。两者可以对同一历史 bank 产生不同权重；每次读入之后，各自的输出增量再写入当前块的 partial sum。不能把一个 Transformer block 简化成只执行一次 AttnRes。

最后一个子层完成后，模型还需要将 embedding 与各个完成块表示交给最终输出读入，再接最终 normalization 和 language-model head。若只追踪 block 内子层的输入而不说明最终输出如何组合，就没有交代模型怎样从这些历史来源得到预测所用的表示。

### 75.4.4 区分 block 数、block 宽度与 residual sublayer 数

用 $L$ 表示残差写入位置总数，用 $N$ 表示把这些位置分成的块数，用 $S$ 表示每块包含的深度更新宽度。概念上 $S$ 约为 $L/N$，但末块可以不足一个完整宽度；embedding 是额外来源，不并入块宽度。若按熟悉的 Transformer block 计数，一个 block 往往包含 attention 与 FFN 两个残差子层，因此必须读清资料采用的层计数口径。

后文会分别说明三种数字：原论文 scaling sweep 中约 $8$ 是块数 $N$；原论文的 48B Kimi Linear 设置以 $54$ 个残差子层、每块 $6$ 个子层形成 $9$ 块；Kimi K3 报告使用“8 个块、12-layer size”的模型特定说法。不能把块数、块宽和 Transformer block 数互换。

## 75.5 让深度读取在训练与推理中可承受

方法定义说明了要读哪些向量，系统实现还要回答这些向量存在哪里、每层重复读取多少、何时跨设备传递。复杂度估算用于看懂成本来源；真实延迟还取决于 checkpointing、流水线并行、内存访问、通信融合和 kernel 调度。

### 75.5.1 Full AttnRes 的算术与历史存储随深度增长

对一个 token，目标深度越深，可读来源越多。把每个目标子层的分数计算与向量加权都计入，$L$ 次读入总计约需 $O(L^2d)$ 运算，并要保留约 $O(Ld)$ 的历史表示；这里 $d$ 是隐藏宽度。一个长度为 $T$ 的序列、一个 batch 处理 $B$ 个序列时，这部分表示状态还会按 token 数 $BT$ 扩展。

在普通训练中，反向传播本来就会保存或重算不少中间激活，因此论文指出 Full 的历史表示可与原本留存的激活重叠。采用激活重算后，原可释放的 layer outputs 可能必须另外保留；流水线并行还要让后续 stage 获得历史来源，内存和通信成本便会显出来。算术复杂度、激活存留和跨 stage 传输是相关但不同的账目。

### 75.5.2 Block AttnRes 以块级历史源降低访存压力

Block AttnRes 把需要长期保留的来源由逐子层的 $L$ 项降至约 $N$ 项，加上当前 partial 和独立 embedding，状态量约为 $O(Nd)$。但每个残差读入仍要比较可见的完成块，并在块内后续位置考虑 partial。若逐个统计所有深度读入与约 $N$ 个来源之间的 score/value 操作，粗略的算术项是 $O(LNd)$，而状态保存项是 $O(Nd)$。

实现会把同一块中的查询批量化，减少重复读取源张量，所以理论上的逐项运算数不等于实际内存流量或延迟。块更宽会减少块数和存储项，却让块内更多输出只能以 partial sum 的方式累积；块更细保留更多可单独选择的历史摘要，也要付出更多来源维护和读取成本。只看“存了几个向量”不能推出加速了同样倍数。

### 75.5.3 两阶段计算批量处理已完成块并在线合并 partial

同一块的伪查询是已知参数，可以先批量查询之前已完成的块表示。进入块内逐层计算后，partial sum 按顺序产生，不能在第一阶段提前查询。第二阶段对新 partial 的分数和值做在线 softmax 合并，避免每次都重新扫描旧块。

具体说，对一组来源 $A$，用最大分数 $m_A$、稳定化分母 $\ell_A$ 和加权分子 $z_A$ 保存其 softmax 统计：

```math
m_A=\max_{i\in A}s_i,\qquad
\ell_A=\sum_{i\in A}\exp(s_i-m_A),\qquad
z_A=\sum_{i\in A}\exp(s_i-m_A)v_i
```

若另有包含当前 partial 的来源组 $B$，两组可先调整到共同最大值，再合并分母和分子，得到与一次性 softmax 全部来源相同的结果：

```math
m=\max(m_A,m_B)
```

```math
\ell=\exp(m_A-m)\ell_A+\exp(m_B-m)\ell_B,\qquad
z=\exp(m_A-m)z_A+\exp(m_B-m)z_B,\qquad
h=\frac{z}{\ell}
```

这里的等价性来自对同一批分数和值重写归一化，不是近似地忽略旧来源。论文的两阶段调度先批量处理块间来源，再顺序加入 partial 并合并在线 softmax 统计；随后当前子层才生成自己的 $v_i$ 并更新 partial。

### 75.5.4 Pipeline cache 与 GPU kernel 决定实际代价

流水线训练中，若每次 stage 转换都传整个历史 bank，同一个块表示会被重复传输。论文的 cross-stage caching 方案让已完成块只在新边界出现时传递和缓存，并将此类状态成本与逐层激活缓存分开。作者报告：未启用 pipeline parallelism 时训练额外开销很小；启用流水线并行时，在其测量设置下端到端训练开销低于 $4\%$。这不是任何硬件和流水线拓扑的保证。

推理侧的两阶段计算批量处理已完成块的查询，并将同块 partial 的顺序依赖留给第二阶段。论文报告典型推理 workload 的延迟额外开销低于 $2\%$。这些数值只说明报告中的模型、负载、实现与并行设置；具体部署还要检查重复访存、编译形状、寄存器压力以及 collective 通信是否能与融合 kernel 配合。

## 75.6 论文证据：损失、梯度、下游表现和开销分别判断

原论文提供了缩放曲线、组件消融、48B Kimi Linear 训练与开销分析。它们分别回答“损失随预算怎样变化”“哪些机制选择在该消融中有效”“加入模型后的测试结果如何”和“系统成本多大”；不能将一种证据当成另一种证据，也不能超出实验配置外推。

### 75.6.1 Scaling law 比较报告特定预算下的 compute 等效

论文在五个模型尺度上比较 PreNorm baseline、Full AttnRes 与约 $8$ 个块的 Block AttnRes。Scaling sweep 使用 $8192$ token context；各变体沿用根据 baseline 选择的同一组超参数。论文按拟合曲线报告一个预算点：Block AttnRes 的 validation loss 为 $1.692$，baseline 为 $1.714$，并将对应关系表述为约 $1.25\times$ compute advantage。

这个 compute 等效绑定论文使用的模型尺度、上下文、超参数和损失曲线。它不是说部署 Block AttnRes 就能少花 $20\%$ 计算，也不能直接解释为单步更快、推理吞吐增加或所有任务都提高同样比例。

### 75.6.2 16-layer 消融分离 Full、Block 与 baseline

在论文的 16-layer ablation 中，PreNorm baseline 的验证损失为 $1.766$，Full AttnRes 为 $1.737$，Block AttnRes（$S=4$）为 $1.746$。论文说明这组变体使用相同超参数和 compute budget，因此可以读作该实验中深度读取及块化取舍的对照。

这组数字不证明任意数据、规模和块宽都保持同样排序。它也不代表 $S=4$ 是固定推荐配置；改变任务深度或每块更新数，都需要新的同预算消融来判断质量与资源的平衡。

### 75.6.3 Kimi Linear 48B 实验报告训练动态和下游指标

论文在 Kimi Linear 架构上训练一组 48B 总参数、3B 激活参数的模型。该模型有 $27$ 个 Transformer block，即 $54$ 个 attention/FFN 残差子层；AttnRes 按每块 $6$ 个残差子层分组，共 $9$ 个块，另加 embedding 来源。训练包括约 $1$T token 的预训练与约 $400$B token 的高质量 mid-training，总训练量约 $1.4$T token。

论文的下游表中，MMLU 从 $73.5$ 到 $74.6$，GPQA-Diamond 从 $36.9$ 到 $44.4$，HumanEval 从 $59.1$ 到 $62.2$；MMLU-Pro 则为 $52.2$，没有变化。它支持的是这套 Kimi Linear 训练和评测流程中的对照结果，不支持“每个能力都会提升”或把分差直接外推到另一种模型。

### 75.6.4 成本数字和 PreNorm dilution 解释都保留实验范围

作者把标准 PreNorm 残差中随深度增长的 hidden-state magnitude、较早层梯度更大等训练观察，解释为贡献稀释问题；AttnRes 的选择性汇总在该实验中使输出幅度和梯度分布更均匀。这是论文对指定训练设置的机制解释，不是深层模型不稳定的唯一原因，也不代表所有任务都必然出现同样的曲线。

同理，低于 $4\%$ 的流水线训练额外开销和低于 $2\%$ 的典型推理延迟额外开销，须连同其通信策略、硬件、负载和 measurement protocol 一起引用。架构收益、训练 wall-clock 与推理 latency 应分开记录；没有对齐的 baseline 和测量条件，就不应拿单个数字下部署结论。[原始报告](https://arxiv.org/pdf/2603.15031)的实验章节给出这些条件与限制。

## 75.7 Kimi K3：架构采用与总体模型成绩分开归因

Kimi K3 把 Block AttnRes 用在一个更大的模型中，为“此机制在什么配置中被采用”提供了官方实例。但 K3 同时包含序列注意力、MoE、量化、训练和 serving 系统的多项设计，所以模型级能力或吞吐不能单独归因于 AttnRes。

### 75.7.1 K3 报告给出 8 个块、12-layer size 与 embedding 来源

[Kimi K3 技术报告 v2](https://arxiv.org/html/2607.24653v2)描述将层划分为 $8$ 个块、每块为 “12-layer size”，末块可以是不完整块，并把 embedding 作为单独来源；报告按包含 embedding 来源计为 $9$ 个深度来源。固定的 [Hugging Face config revision](https://huggingface.co/moonshotai/Kimi-K3/blob/2496450e92e425c886db095102a52a6682ca3970/config.json)也包含 `attn_res_block_size` 字段，值为 $12$。

应保留报告原有的 “12-layer size” 与配置字段名，不擅自将它改写成原论文所有实验统一的残差子层数。尤其不要把这组 K3 配置和原 AttnRes scaling sweep 的“约 $8$ 个块”混为一谈：一个报告 $8$ 个块和 “12-layer size”，另一个只说块数约为 $8$。原论文 48B Kimi Linear 实验又是 $9$ 个块、每块 $6$ 个残差子层。

### 75.7.2 K3 整体能力不能单独证明 AttnRes 因果收益

K3 报告同时介绍 KDA、Gated MLA、Stable LatentMoE、量化、预训练和后训练等组件；固定配置只能确认公开字段，不能替代同一 K3 架构上关闭 AttnRes 的对照。K3 的 benchmark、Agent 表现或总体吞吐只能作为完整模型结果，不能拆成 AttnRes 的独立贡献。

若要估计 AttnRes 本身的因果影响，需要在其他条件相同的前提下只改变残差机制，并对齐模型、训练 token、数据、硬件、评测 harness 与预算。这也是为什么本章引用原论文中受控对照结果，却不把 K3 发布成绩写成 AttnRes 的单因素证明。K3 全模型架构由[第 88 章](88-kimi-k3-kda-stable-latentmoe与百万token-agent.md)统筹。

### 75.7.3 推理 kernel 的具体优化回链实现证据

AMD 的 [Kimi K3 on MI355X with TokenSpeed 工程记录](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus-with-tokenspeed.html)描述一种具体执行路径：单遍访问快照并融合 score、加权混合与 normalization；对符合条件的 TP8 batch 采用 token-row 分片，在 reduce-scatter 之后分摊 AttnRes 与 normalization 工作，再为后续 MoE 汇集结果。文章还指出一次保留所有候选会带来寄存器压力，所以按快照逐项读取。

该材料报告的 kernel 数字是特定 MI355X、TokenSpeed 与测量形状下和直接 PyTorch 路径的比较；它展示的是实现如何处理访存、寄存器与通信，不能外推为一般硬件上的模型级加速比。它也不能替代 Kimi 训练论文中的架构对照。

## 75.8 后续变体与残差路线的选择边界

AttnRes 后续工作分别调整路由 key 的维数、查询与 value 的角色关系，或可读取的多流视图。它们都在研究深度方向的信息汇总，但改变点和证据规模不同；应称为各自论文提出的变体，不要回写成原始 AttnRes 定义。

### 75.8.1 Low-Rank AttnRes 将路由 key 维数与 value 宽度拆开

[Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694)保留全宽度 value，用低维 key 计算深度路由分数；其一项方案使用 value 的末尾 $r$ 个维度作为 key。作者在 $1$B 与 $4$B 参数设置中报告，$r=d/4$ 时相对标准 AttnRes 有更低的最终验证损失、更高的平均下游分数及更高的训练 step throughput。

这些数字属于该预印本的训练、模型、kernel 与测量协议，不能推导为任何模型把 key 缩到四分之一就会获得固定提速。这个变体的主要问题是：路由判断是否需要完整隐藏维度，还是可以用更窄的表示而保留完整内容值。

### 75.8.2 RD-AttnRes 将匹配路径与 value 内容路径解耦

[Role-Decoupled Attention Residuals](https://arxiv.org/pdf/2608.01075)让 Q/K 共用一种深度路由，再为 V 学另一种读取分布，以检验“在哪里匹配”与“取回什么内容”是否应该采用同一份权重。其报告在 FineWeb-Edu、$120$M 和 $343$M 两种规模、每种五组配对 seed、约 $2$B token 的设置中，十组配对的验证 NLL 都下降。

作者将结论限定在该训练 regime，并明确需要更长训练、跨数据集和更强 baseline。它是支持进一步研究角色解耦的受控小规模证据，不是普适优越性证明。

### 75.8.3 Multi-Stream AttnRes 为不同来源视图增加多流读取

[Multi-Stream Attention Residuals](https://arxiv.org/pdf/2610.13630)先用多尺度 causal depthwise convolution 形成不同序列上下文的 stream，再让这些 stream 分别读取历史深度表示，并通过轻量混合得到单流子层输入。作者报告在 $18$B 与 $28$B MoE 设置中比较 $12$ 项下游 benchmark，同时给出自己的 scaling-law 实验。

这些结果属于该预印本的模型、训练预算和评测协议，不是和原 AttnRes 论文统一复现后的跨论文排名。它的结构问题是以额外的上下文视图换取不同的深度读取；是否值得，需要连同新增卷积、参数、计算和任务收益一起测量。

### 75.8.4 与 KDA、mHC 对齐问题后再决定采用哪条路线

AttnRes 沿网络深度从历史子层或块表示中选择；[KDA](61-kda递归注意力.md)沿 token 序列递归更新状态；[mHC](78-mhc双随机残差连接.md)维护同一深度上的多条残差流并约束它们的混合矩阵。三者都影响信息流，但状态轴和作用机制不同。KDA 与全局 MLA 在 Kimi Linear 中的组合见[第 76 章](76-kda从delta规则到线性注意力.md)。比较时应固定模型、训练预算、数据、并行和任务，分别测量质量、保存状态、通信与推理延迟；不能因为它们都修改“残差流”就当作同一类注意力。
