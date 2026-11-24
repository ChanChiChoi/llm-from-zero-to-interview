# 第 73 章 Gated Attention：SDPA 输出后的输入依赖门控

本章的 Gated Attention 专指 Qiu 等人在 NeurIPS 2025 系统研究的 softmax attention 变体：在 Scaled Dot-Product Attention（SDPA）输出之后、输出投影之前，用当前输入产生的 sigmoid gate 调节每个注意力头的输出特征。它不是在 Gated DeltaNet 的递归状态更新门，也不泛指在局部、全局和递归路径之间做路由；第 63 章另讲 Gated MLA。本章默认读者了解 Q/K/V、softmax 和多头注意力，读完应能定位 G1、解释它改变和不改变的对象，并把论文结果限制在对应训练与评测条件内。

## 73.1 Gated Attention 出现在 SDPA 的哪个位置

这一节把标准注意力的 token 加权聚合与 G1 的输出特征调节连起来。关键区别是 gate 乘在已经混合好的注意力结果上；它不会在当前 SDPA 中重新归一化 token 权重，也不表示模型转去执行另一条路径。

### 73.1.1 标准 SDPA 把值向量混成每个 query 的输出

对 query 位置 $i$ 和注意力头 $h$，softmax 根据 $q_i^h$ 与可见 key 的相似度生成权重，再对对应的 value 加权求和。本小节围绕这个聚合对象说明 attention weight 表示“本头在可见位置间如何分配读取权重”，而头输出是一个由多个 value 混合成的特征向量；它为之后辨认 gate 所调节的对象提供必要前置。

~~~math
a_i^h=\sum_{j\in\mathcal{V}(i)}\alpha_{ij}^h v_j^h,
\qquad
\sum_{j\in\mathcal{V}(i)}\alpha_{ij}^h=1.
~~~

### 73.1.2 G1 在 SDPA 后、输出投影前逐元素调节结果

Qiu 等人把 G1 定义为 head-specific sigmoid gate，输入取 pre-normalization hidden state；对每个 query/head，gate 逐维调节刚刚形成的 SDPA 输出，再交给多头输出投影。它在注意力加权聚合之后插入输入依赖的非线性，因此调节的是混合结果的特征维，而不是单个来源 token 的 softmax 权重。[论文最终版](https://papers.neurips.cc/paper_files/paper/2025/file/904e89bb4e632e75fb47f093b620b257-Paper-Conference.pdf)给出统一的门控形式 $Y'=Y\odot\sigma(XW_\theta)$；[Raschka 的讲解](https://sebastianraschka.com/llm-architecture-gallery/gated-attention/)也把 gate 放在 SDPA 与输出投影之间。

~~~math
g_i^h=\sigma(x_iW_g^h),
\qquad
\widetilde{a}_i^h=a_i^h\odot g_i^h,
\qquad
o_i=\operatorname{Concat}_h(\widetilde{a}_i^h)W_O.
~~~

### 73.1.3 gate 调节注意力结果，不负责选择另一条计算路径

对两个可见 value $v_1=(0,2)$、$v_2=(4,0)$，若 attention weights 为 $(0.25,0.75)$，SDPA 先得到 $a=(3,0.5)$。假设 G1 给出 gate $(0.2,0.8)$，调节后是 $(0.6,0.4)$；原来的 $(0.25,0.75)$ 仍是本次 SDPA 使用的权重。这个小算例只说明“先聚合、再调节特征”，不能被读成 gate 已找出哪一个来源 token 更重要，也不能推导出注意力 kernel 被跳过。

## 73.2 位置、粒度与共享方式共同定义 gate

“门控”不是一个单独的配置轴。Qiu 等人分别改变门控所在阶段、每个头输出上使用标量还是向量、各头是否共用 gate，以及乘法/加法和激活函数；比较结果必须确认这些轴，而不能只按“用了 sigmoid gate”归类。

### 73.2.1 G1—G5 放在注意力计算的不同阶段

论文将 G1 放在 SDPA 输出后，将 G2 放在 value 投影后，并把 G3、G4 分别放在 key、query 投影后；G5 则位于最终多头输出投影之后。G3/G4 会影响产生 attention scores 的输入，G2 改变送入加权聚合的 value，G1 改变加权聚合后的头输出，而 G5 作用于已通过输出投影的结果。这些位置具有不同的计算关系，不能用一个通用公式互相替代。

### 73.2.2 head-specific/shared 和 headwise/elementwise 是两组独立选择

head-specific/shared 描述不同注意力头是否有各自或共享的 gate；headwise/elementwise 描述一个头是共用一个标量，还是对输出向量的各维分别给值。前者改变头之间的独立调节能力，后者改变每头内部的调节分辨率。二者是正交选择，也不等于 GQA 中 query head 对 key/value head 的共享关系。

### 73.2.3 乘法 sigmoid、加法与激活变体不可混作同一设置

G1 的乘法 sigmoid 形式把各 gate 值限制在 $[0,1]$，用于缩放已经形成的输出特征；论文也比较了加法形式及 SiLU、非稀疏 sigmoid 等变体。其消融显示，设置会改变稀疏程度与结果，SDPA 后的乘法 sigmoid 表现最稳定。不能把某个变体的指标、稀疏性或训练表现归给所有名为 Gated Attention 的模块。

## 73.3 为什么论文偏向 G1

论文提出的解释聚焦于 value 投影与输出投影之间的低秩复合，以及 G1 把输入依赖的非线性放在加权聚合之后。这个解释说明 G1 与 G2 为什么值得比较，但它是论文分析对结果的机制解释，不意味着单个 gate 在任意模型上必然提升表达能力或检索质量。

### 73.3.1 value 与输出投影之间的低秩复合限制 attention 输出映射

先固定某头的 attention weights。value 投影把模型维映射到头维，头输出再经对应的输出投影回到模型维；两个线性映射可以合并成一个秩至多为 $d_k$ 的复合映射。因为通常 $d_k<d_{\mathrm{model}}$，这一 value/output 子路径受低秩限制；GQA 还会在一组 query heads 间共享 value 映射。这里讨论的是固定 attention weights 后 $W_V^hW_O^h$ 的低秩复合，不是把整个含输入依赖 softmax 权重的 attention 层都称为低秩。

~~~math
o_i^h
=\left(\sum_j\alpha_{ij}^h x_jW_V^h\right)W_O^h
=\sum_j\alpha_{ij}^h x_j\left(W_V^hW_O^h\right).
~~~

### 73.3.2 G1 把 query-dependent 非线性放在加权聚合之后

G1 的 gate 来自当前 query 所在位置的 hidden state $x_i$，而 attention 输出已经汇总了可见 value。逐维乘法在聚合结果与输出投影之间增加非线性，使同一组线性投影能够随当前 query 改变保留哪些输出特征。它调节的是 query 对应的聚合结果，并没有把聚合拆回对各历史 token 的离散选取。

### 73.3.3 G1、G2 的效果差别涉及查询依赖与稀疏性

G2 在 value 进入加权求和前调节各来源表示，gate 依赖各 key/value 位置；G1 则依赖当前 query，并在求和后调节结果。论文的 15A2B 对照中两者都改善基线，但 G1 的 PPL 和整体基准结果更好。作者进一步把差异与 query-dependent、head-specific 的稀疏 gate 联系起来；这些观察不是“G1 识别到金标准证据”的因果证明。

## 73.4 论文实验报告的质量、稳定性和额外成本

Gated Attention 的收益应按模型规模、训练数据、序列长度、优化配置、指标和消融项阅读。不同实验回答不同问题：MoE 表格比较门控位置与形式，dense 实验报告训练稳定性与学习率容忍度，参数和延迟则只表示论文实际测量的实现设置。

### 73.4.1 15A2B 消融显示 G1 改善 PPL 与基准结果

论文在 15B 总参数、2.54B activated 的 15A2B MoE 上训练了 400B tokens；序列长度为 4096，global batch 为 1024，共 100k steps。表 1 中的 baseline 平均 PPL/MMLU 为 6.026/58.79；elementwise G1 为 5.761/60.82；headwise G1 为 5.792/60.05。PPL 越低越好，MMLU 越高越好。它们是该模型和训练设置中的作者报告结果，并不代表所有下游任务或规模都能获得同样差值。

| 15A2B 设置 | 平均 PPL | MMLU |
| --- | ---: | ---: |
| Baseline | 6.026 | 58.79 |
| SDPA elementwise G1 | 5.761 | 60.82 |
| SDPA headwise G1 | 5.792 | 60.05 |

### 73.4.2 Dense 实验报告更稳定训练与较大的学习率容忍度

在 1.7B dense 模型上，作者报告 SDPA sigmoid gate 减少训练 loss spikes，并在较大的学习率设置下改善稳定性。一个 400B-token、batch 1024、最大学习率 $4\times10^{-3}$ 的 28-layer 对照中，PPL 从 7.499 到 7.404，MMLU 从 50.21 到 51.15；3.5T-token 组另使用 batch 2048 和最大学习率 $4.5\times10^{-3}$。论文也报告某些更大学习率的 baseline 训练失败，而带 gate 的对应组可训练。结果属于作者给出的配置消融，不能简化为加 gate 就自动稳定训练。

### 73.4.3 参数与延迟成本属于论文的实验配置

在 15A2B 的表格中，headwise G1 增加的参数约为 1.6M，论文正文称少于 2M；作者报告门控带来的 wall-time latency 在其实验设置中小于 2%。这两个数字都不应外推为任意模型、硬件、kernel 的保证。G1 仍要先完成 QK 相似度、softmax 与 value 聚合；它不通过关闭显式 attention 来降低二次 attention 计算，也不自行缩减 KV cache。

## 73.5 Attention sink 与长上下文证据的边界

论文报告训练后的模型出现较低的首 token attention mass 和较小的 massive activation，并在一个包含继续训练与 YaRN 的扩展实验中得到更好的 RULER 结果。必须把同一次前向中的代数作用、训练后观察到的 attention 图变化，以及长上下文实验里同时改变的配置分开理解。

### 73.5.1 G1 训练后减少首 token attention mass

论文在测试语言模型数据上观察到，baseline 各层平均有 46.7% 的 attention score 分配给第一个 token，SDPA gate 模型为 4.8%；baseline 的第 21 层为 83%，对应 gated 模型为 4%。这里比较的是训练后模型的 attention map。对一个固定层输入和固定 Q/K/V，G1 位于 SDPA 之后，不会代数地改写该次 softmax 权重；训练参数与层间表示变化后，模型学到的 attention map 可以不同。

### 73.5.2 Massive activation 与 attention sink 是关联而非同一指标

论文分别报告 massive activation（跨层 hidden state 最大绝对值的统计量）与 first-token attention mass。15A2B 表 4 中，value 位置的 G2 将最大 activation 从 1053 降到 125，但首 token mass 仍为 0.297；head-shared G1 的相应指标为 286 与 0.301；head-specific elementwise G1 则为 94 与 0.048。G2 这组结果说明较小的 hidden activation 本身不能代替 sink 指标，两种现象相关但不可互相当作同一测量。

| 变体 | 平均最大 activation | 首 token attention mass |
| --- | ---: | ---: |
| Baseline | 1053 | 0.467 |
| Value elementwise G2 | 125 | 0.297 |
| SDPA head-shared G1 | 286 | 0.301 |
| SDPA elementwise G1 | 94 | 0.048 |

### 73.5.3 RULER 外推结果绑定 32K 继续训练和 YaRN 设置

长上下文实验先以 3.5T-token 模型为起点，将 RoPE base 从 10k 调到 1M，并在序列长度 32K 上继续训练 80B tokens，随后用 YaRN 扩展到 128K，再测 RULER。扩展后的 64K 分数为 baseline 37.51、SDPA gate 66.60；128K 分数为 31.65 与 58.82。延展同时包含继续训练、RoPE 配置和 YaRN，不能把差异归因于 gate 单因素；两组在原 32K 范围内也都经历扩展后的退化，只是 gated 组较轻。

## 73.6 Qwen3-Next 把 output-gated SDPA 放进 3:1 hybrid stack

Qwen3-Next 是论文所述 SDPA output gating 的一个具体模型案例，模型卡同时披露了它与 Gated DeltaNet 的层间组合。这个例子展示 gate 如何出现在含普通 softmax attention 的 block 中，不用它推断全部混合架构的性能来源。

### 73.6.1 48 层周期布局包含 36 个 GDN 与 12 个 gated attention

Qwen3-Next-80B-A3B-Instruct 的固定模型卡 revision 给出 48 层，重复 12 次“3 个 Gated DeltaNet → MoE，再 1 个 Gated Attention → MoE”；因此共有 36 个 GDN 层和 12 个 gated attention 层。卡片列出的模型规模是 80B total、3B activated。它把递归状态路径与显式 attention block 交错起来；Gated Attention 层中的 output gate 仍是 attention block 内的特征调节机制。[固定 revision 模型卡](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/c7646fbcc61b04c51c1dacf7abf1d2e4df785524/README.md)

### 73.6.2 混合效率不能归结为 gate 单独节省 attention 计算

Gated Attention 层仍执行显式 attention；G1 只调节其 SDPA 输出。该模型总体效率是包括 GDN 层比例、MoE、模型配置和实现路径在内的混合系统属性。输出门本身不是动态跳过 attention kernel 的路由器，也不能单独解释模型卡中的吞吐或长上下文主张；相关性能需保留官方报告的比较条件。

## 73.7 使用门控证据时不能跨越的推断

G1 的 feature gate、softmax token weights、训练后 attention map 和真实执行成本是不同对象。阅读结果或设计消融时，需分别观察这些变量并对齐任务和训练条件；不从其中一项的变化单独推出其余结论。

### 73.7.1 gate 输出稀疏不等于 softmax token sparse

当许多 G1 特征值接近零时，表示某些 query/head 的 SDPA 输出特征被压低；softmax attention 仍对 mask 允许的 key 计算归一化权重。feature sparsity 不等同于删除 token、稀疏化 attention matrix、减少可见位置或节省 attention FLOPs。

### 73.7.2 gate 值和 attention weight 都不能单独证明证据选择

G1 的索引是 query、head 和输出特征维；attention weights 的索引则是 query 与来源 token。两者都可用于描述某种中间计算，但任何单张 gate heatmap 或 attention map 都不能单独说明哪条证据导致了正确答案。对具体任务的因果解释还需要与证据扰动、对照模型及任务结果相配的分析。

### 73.7.3 2026 后续研究扩展机制解释但未解决大规模外推

2026 年 9 月的 SinkProbe 预印本 v2 同时包含短长度受控训练模型与另行进行的 Kimi K3 长上下文诊断。其受控 gate 模型在长度 96 上训练、评估至 768；作者称 gate 的 sink-mass 差异处于 seed spread 内，平均 gate score 为 0.686，未学成 Qiu 等人模型中观察到的稀疏 gate（0.116）。这不能视为对 15B/1.7B 原始实验的等规模反证，也不能说明该机制普遍失效。[论文 v2](https://arxiv.org/abs/2609.08574v2)

2026 年 11 月的机制分析 arXiv v1（页面列明 NeurIPS 2026）提出 sink-induced spectral modes，并给出固定 attention weights 与 upstream gradients 条件下的 value-path gradient 局部收缩分析。该结论有明确条件；它没有证明训练后网络中的所有梯度都普遍缩小，也不能把理想化谱模型当作已确立的唯一机制。[论文 v1](https://arxiv.org/abs/2611.20403v1)

### 73.7.4 按相同训练/任务对齐比较 gate 位置与计算收益

比较 gate 位置、粒度与共享方式时，应固定模型规模、数据、训练 token、序列长度和优化预算，并分别报告语言模型质量、下游任务、gate 分布、attention map 与资源测量。若问题是计算节省，还要区分“结果乘小”与“实际不执行某 kernel”：只有后者才可能减少相应计算，而且必须在明确的实现、硬件和 workload 上测量。论文的 G1 结果并不证明 gate 的统计稀疏能自动变成执行稀疏。

### 本章回顾

Gated Attention 的 G1 是 SDPA 后、输出投影前的 query-dependent sigmoid 输出门。它在固定 Q/K/V 的单层前向中调节聚合后的特征，不改写该次 softmax token 权重，也不选择另一条路径。Qiu 等人的质量、稳定性、sink 与长上下文结果有各自的实验条件；Qwen3-Next 提供的是一项具体模型实现与混合层布局。读这些证据时，应始终区分 gate 特征、token attention、训练后行为和实际执行成本。

