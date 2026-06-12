# 第九章：Transformer 的并行性、表达力和 scaling 优势

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修主要对齐 Transformer 原论文中关于摆脱 recurrence / convolution、提高序列并行度和缩短最大路径长度的论述，Kaplan scaling laws 对 cross-entropy loss、参数量、数据量和计算量的经验幂律拟合，GPT-3 对 few-shot / in-context learning 的展示，Chinchilla 对固定 compute 下参数和训练 token 配比的修正，PaLM 对大规模系统训练的案例，以及 emergent abilities 相关论文对“涌现”现象的边界讨论。

写作边界如下：

1. scaling law 是经验规律，不是跨模型族、跨数据分布、跨训练 recipe 都固定成立的物理定律。
2. 本讲不把 Kaplan、Chinchilla 或 PaLM 中的具体指数、token/parameter 比例、硬件规模写成通用标准，只讲它们带来的方法论。
3. 第二轮精修会把并行性、路径长度、attention 成本、训练 FLOPs 和 scaling gate 写成稳定 MathJax 公式，并补一个 0 依赖 demo，用 toy 数字解释“最低 toy loss”不等于“最适合上线或最 compute-balanced”。

## 9.1 本章定位

第一章已经从全局角度回答过“Transformer 为什么能成为大模型基础架构”。本章进入第二部分，开始系统拆解 Transformer 的优势和瓶颈。

本章先讲优势：并行性、表达力和 scaling。

Transformer 能在大模型时代胜出，不只是因为 attention 很优雅，而是因为它同时满足了三类条件：

```text
算法上：能让 token 之间做灵活的信息路由。
工程上：能把大部分计算变成大矩阵乘法，适合 GPU/TPU 并行。
规模上：随着参数、数据和计算增加，loss 和能力通常能稳定改善。
```

这三点叠加，形成了大模型发展的主线：

```text
更大模型 + 更多数据 + 更多计算 + 稳定训练 recipe + Transformer 主干
```

本章要回答的问题是：

1. Transformer 为什么比 RNN 更适合大规模并行训练。
2. Self-Attention 的表达力来自哪里。
3. 多层、多头、FFN 和 residual stream 如何共同提升表达能力。
4. 为什么 Transformer 对 GPU/TPU 友好。
5. scaling law 到底说明了什么。
6. Chinchilla 为什么改变了“只堆参数”的观念。
7. GPT-3、PaLM 等模型展示的 few-shot、reasoning 和 emergent abilities 该如何理解。
8. scaling 的边界是什么，为什么不能把 scaling law 当成万能定律。

本章的核心观点是：

```text
Transformer 的成功不是单一结构胜利，而是表达力、并行计算和可预测扩展三者共振的结果。
```

## 9.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 Transformer，强调完全基于 attention，摆脱 recurrence 和 convolution，并在机器翻译中实现更高并行性和更低训练成本。
2. Kaplan et al., 2020, *Scaling Laws for Neural Language Models*。系统研究语言模型 cross-entropy loss 随模型规模、数据规模和训练 compute 的幂律关系。
3. Brown et al., 2020, *Language Models are Few-Shot Learners*。GPT-3 展示大规模 autoregressive Transformer 的 few-shot/in-context learning 能力。
4. Hernandez et al., 2021, *Scaling Laws for Transfer*。研究预训练迁移中的 scaling 规律，说明预训练可转化为有效数据增益。
5. Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*。Chinchilla 工作指出固定 compute 下模型参数和训练 token 需要共同扩展，很多早期大模型 undertrained。
6. Chowdhery et al., 2022, *PaLM: Scaling Language Modeling with Pathways*。展示大规模 dense Transformer 在多任务、few-shot、多步推理、代码和多语言上的 scaling 效果。
7. Wei et al., 2022, *Emergent Abilities of Large Language Models*。讨论一些能力在小模型中不明显、在大模型中突然出现的现象。

需要说明的是，scaling law 是经验规律，不是物理定律。它依赖模型族、数据分布、训练 recipe、优化稳定性和评估指标。本章会同时讲 scaling 的价值和边界。

## 9.3 Transformer 的并行性优势

先看 RNN/LSTM。

RNN 的计算有强时间依赖：

```math
h_t=f(h_{t-1},x_t)
```

第 `t` 个 hidden state 必须等第 `t-1` 个 hidden state 算完。即使 batch 很大，单条序列内部仍然难以完全并行。

Transformer 不一样。对一整段序列：

```math
X=[x_1,x_2,\ldots,x_n]\in\mathbb{R}^{n\times d}
```

它可以一次性计算：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

再一次性计算所有 token pair 的 attention score：

```math
S=\frac{QK^\top}{\sqrt{d_k}}
```

这意味着训练时，序列中所有位置可以并行计算。

并行性带来的不是小优化，而是大模型时代的根本前提。因为预训练需要处理万亿级 token，如果序列内部强串行，训练墙钟时间会非常难接受。

## 9.4 并行性不等于计算量少

Transformer 并不是每个维度都更省。

Self-Attention 的 score matrix 是 `n x n`：

```math
|S|=n^2
```

如果 head dimension 是 `d_h`，attention score 的主要矩阵乘成本可以粗略写成：

```math
C_{\mathrm{score}}=O(n^2d_h)
```

所以 attention 在长序列下有 O(n^2) 计算和显存瓶颈。这也是下一章要重点讲的问题。

但 Transformer 的优势在于：

```text
虽然总计算量可能大，但这些计算高度并行、规则、密集，适合 GPU/TPU。
```

相比之下，RNN 的计算量可能看起来是 O(n)，但时间步依赖导致并行度低。

可以用一句话概括：

```text
Transformer 用更多可并行计算，换掉了 RNN 的串行依赖。
```

在大规模硬件上，这个 trade-off 非常划算。

## 9.5 GPU/TPU 友好性

现代加速器最擅长的是大规模矩阵乘法。

Transformer 的核心操作大多是矩阵乘法：

1. Q/K/V projection。
2. attention score 的 `QK^T`。
3. attention output 的 `AV`。
4. output projection。
5. FFN 中的两个或三个大 linear layers。

这些操作可以高效映射到 GPU tensor cores 或 TPU matrix units。

同时，Transformer 的结构很规则：

```text
attention block + FFN block + residual + norm
```

重复堆叠几十层、上百层，工程系统可以围绕这种规则结构做大量优化：

1. fused kernels。
2. FlashAttention。
3. tensor parallelism。
4. pipeline parallelism。
5. sequence parallelism。
6. activation checkpointing。
7. ZeRO/FSDP optimizer sharding。
8. KV cache 推理优化。

这就是为什么 Transformer 不只是一个算法结构，也是一个硬件生态友好的计算模板。

## 9.6 Self-Attention 的表达力：动态信息路由

Self-Attention 的核心表达力来自动态路由。

对每个 query token，模型会根据当前内容动态决定从哪些 token 读取信息：

```math
a_{ij}=
\frac{\exp(q_i^\top k_j/\sqrt{d_k})}
{\sum_{\ell\in\mathcal{V}_i}\exp(q_i^\top k_\ell/\sqrt{d_k})}
```

其中 `V_i` 是第 `i` 个 query 允许看到的 key 集合。这个集合由上一章讲的 attention mask 决定。

这和固定卷积核不同。

CNN 的局部卷积通常在固定邻域内混合信息：

```text
当前位置看附近几个位置
```

Self-Attention 则可以让当前位置直接看任意位置：

```text
当前位置可以看前一个词、句首主题、函数定义、远处括号、系统提示、示例答案
```

更重要的是，这个“看哪里”的模式不是手写规则，而是由 Q/K 内容匹配动态决定。

所以 attention 更像一个可学习的信息检索系统：

```text
Q：我现在需要什么信息？
K：每个位置提供什么索引？
V：每个位置真正携带什么内容？
```

这使 Transformer 很适合语言、代码、数学推理、对话上下文、工具调用等需要动态依赖的任务。

## 9.7 一跳长距离交互

在 RNN 中，位置 `i` 的信息传到位置 `j`，需要经过很多时间步：

```text
i -> i+1 -> i+2 -> ... -> j
```

路径长，梯度和信息都容易衰减或被覆盖。

在 CNN 中，如果卷积核很小，也需要多层堆叠才能扩大感受野。

Self-Attention 中，任意两个 token 可以在一层内直接交互：

```text
i -> j
```

这对长距离依赖非常重要。

例如：

1. 代码中函数调用要回看函数定义。
2. 数学证明中后文要引用前文假设。
3. 多轮对话中当前回答要遵守系统提示。
4. 长文摘要中结论要整合前面多个段落。
5. JSON/XML/括号结构中远距离符号要匹配。

当然，一跳可达不等于一定能用好。长上下文中的 lost-in-the-middle、精确检索和干扰问题会在后续章节专门讨论。

## 9.8 多头注意力：多个子空间的并行路由

单个 attention head 只能学习一种 Q/K/V 投影方式。多头注意力把 hidden representation 切成多个子空间：

```text
head_1, head_2, ..., head_h
```

每个 head 可以学习不同的信息路由模式。

例如：

1. 某些 head 关注局部邻近 token。
2. 某些 head 关注句法依赖。
3. 某些 head 关注分隔符或特殊 token。
4. 某些 head 关注复制、引用、括号匹配。
5. 某些 head 在特定任务中承担检索或路由作用。

不要把这些解释绝对化。attention head 的可解释性有启发价值，但 raw attention weight 不是完整因果解释。

从表达力角度看，多头注意力的意义是：

```text
同一层中并行学习多种关系，而不是用一个相似度函数解释所有依赖。
```

## 9.9 FFN 的表达力：逐 token 的非线性计算

Attention 负责 token 间信息交换，FFN 负责每个 token 内部的非线性变换。

一个 Transformer block 可以粗略理解为：

```text
Attention：从上下文读取信息
FFN：对读取后的信息做非线性加工
```

FFN 通常占 Transformer 参数量的大头。现代 LLM 中的 SwiGLU/Gated MLP 进一步提升了 FFN 的表达能力。

所以不能把 Transformer 的表达力全部归因于 attention。更准确地说：

```text
Attention 提供上下文路由，FFN 提供局部计算，残差流负责在层间积累和传递状态。
```

多层堆叠后，模型能进行复杂的上下文组合。

## 9.10 Residual Stream：逐层写入和读取的共享工作区

Transformer 中每层都通过 residual connection 更新 hidden states：

```math
x_{\ell+1}=x_\ell+F_\ell(\mathrm{Norm}(x_\ell))
```

可以把 residual stream 想成一个共享工作区。

每层 attention 和 FFN 都可以从这个工作区读取信息，再写入新的增量信息。

这带来一个重要直觉：

```text
Transformer 不是每层重新表示所有东西，而是在 residual stream 中逐层积累特征、关系和中间计算结果。
```

这有助于理解为什么 depth 有价值：更深层可以在前面层构造出的表示基础上继续组合。

但 depth 也带来训练稳定性问题，所以需要 Pre-Norm、RMSNorm、初始化、学习率 warmup、residual scaling 等配套机制。第六章已经详细讲过这部分。

## 9.11 Scaling Law 到底说了什么

Scaling law 研究的是：当模型参数量、训练数据量和训练计算量变大时，语言模型 loss 如何变化。

Kaplan 等人的工作发现，在一定范围内，cross-entropy loss 和规模变量之间可以用幂律关系描述：

```math
L(N)\approx A_NN^{-\alpha}+B_N
```

```math
L(D)\approx A_DD^{-\beta}+B_D
```

```math
L(C)\approx A_CC^{-\gamma}+B_C
```

其中：

1. `N` 是模型参数量。
2. `D` 是训练数据量。
3. `Compute` 是训练计算量。
4. α、β、γ 是经验拟合出来的指数。

不要死记公式。核心意义是：

```text
在相当大的尺度范围内，扩大模型、数据和计算，loss 会以相对平滑、可预测的方式下降。
```

这给大模型训练带来了工程上的可规划性。

训练一个百亿、千亿参数模型前，可以先做一系列小模型实验，拟合趋势，再预测更大模型可能达到的 loss 区间。

## 9.12 Scaling Law 为什么重要

Scaling law 的影响不只是学术上的。

它改变了大模型研发方式。

在没有 scaling law 之前，大规模训练更像豪赌：

```text
投入巨大算力，训练完才知道结果。
```

有了 scaling 经验规律后，可以做更系统的资源规划：

1. 给定 compute budget，选多大模型。
2. 给定模型大小，训练多少 token。
3. 训练曲线是否偏离预期。
4. 数据质量提升是否带来 loss 改善。
5. 是否值得继续扩大模型或数据。
6. 小规模实验能否预测大规模趋势。

这也是为什么 scaling law 巩固了 Transformer 路线。它让产业和研究更愿意沿着同一主干持续优化，而不是频繁更换架构。

## 9.13 Chinchilla：从“更大”到“compute-optimal”

早期 scaling 叙事容易被简化成：

```text
模型越大越好。
```

Chinchilla 修正了这个观念。

Hoffmann 等人的工作指出，固定 compute budget 下，很多已有大模型参数量很大，但训练 token 不足，也就是 undertrained。

他们的核心启发是：

```text
参数量和训练 token 数要一起讨论。
```

在同样训练计算量下，一个更小但训练 token 更多的模型，可能超过一个更大但训练不足的模型。

常见的 dense Transformer 训练 FLOPs 粗估是：

```math
C_{\mathrm{train}}\approx 6ND
```

其中 `N` 是参数量，`D` 是训练 token 数。这个公式只用于粗略估算，不替代真实 profiler、激活重算、并行效率、optimizer 开销和硬件利用率分析。

Chinchilla 用 70B 参数、更多训练数据，在很多任务上超过了 Gopher 280B、GPT-3 175B 等更大模型。

面试里不要把 Chinchilla 机械背成某个固定比例。更准确的表达是：

```text
Chinchilla 提供的是 compute-optimal 思路：在给定算力下，需要平衡模型容量和训练 token，而不是只追求参数量。
```

这个思想也推动了后来的高质量小模型路线：参数不一定最大，但训练更充分、数据质量更高、recipe 更好，实际能力可以很强。

## 9.14 Scaling 不只是参数，还包括数据质量

Scaling law 常用参数量、token 数和 compute 描述趋势，但真实训练中，数据质量同样关键。

两个模型都训练 1T token，不代表它们获得了同样有效的信息。

数据质量会影响：

1. 重复内容比例。
2. 垃圾网页和模板文本比例。
3. 代码、数学、知识、对话等数据配比。
4. 多语言覆盖。
5. 数据污染和 benchmark leakage。
6. harmful/toxic 内容分布。
7. 训练样本格式一致性。

所以更准确的说法是：

```text
Scaling 需要参数、有效数据、计算、优化稳定性和评估体系共同成立。
```

单纯增加低质量 token，可能只是浪费 compute，甚至损害能力和安全性。

## 9.15 GPT-3：Scaling 带来的 In-Context Learning

GPT-3 的重要性在于，它展示了大规模 decoder-only Transformer 的 few-shot 能力。

传统范式是：

```text
预训练模型 + 任务数据 fine-tuning
```

GPT-3 展示的是：

```text
在 prompt 中给任务说明和少量示例，不更新参数，模型也能完成新任务。
```

例如：

```text
English: cat
French: chat

English: dog
French: chien

English: book
French:
```

模型可以从上下文示例中推断任务模式。

这背后至少涉及三件事：

1. 预训练中见过大量任务、格式和文本模式。
2. attention 能在上下文中读取示例并匹配当前 query。
3. 大模型容量足够强，可以把 prompt 当作临时任务说明。

In-context learning 不等于模型真正像人一样学习，也不等于参数更新。它更像是在上下文中做模式识别、检索、归纳和条件生成。

后续第 12 章会专门展开 In-Context Learning 与显式 token 检索能力。

## 9.16 PaLM：大规模训练和系统能力

PaLM 展示了另一个角度：大规模 dense Transformer 需要系统工程配套。

PaLM 540B 使用 Pathways 在 6144 个 TPU v4 上训练。它说明大模型 scaling 不只是模型结构问题，还包括：

1. 分布式训练系统。
2. 数据管线。
3. 稳定性控制。
4. 高效并行策略。
5. 训练监控和故障恢复。
6. 大规模评估。

PaLM 报告了多语言、代码、few-shot、多步推理等能力，并观察到一些 BIG-bench 任务随规模出现不连续提升。

这说明 Transformer scaling 的成功，是算法和系统共同作用的结果。

如果没有高效并行系统，大模型无法训练；如果没有稳定训练 recipe，scaling 曲线会偏离；如果没有足够数据，参数会 undertrained。

## 9.17 Emergent Abilities：涌现能力怎么理解

涌现能力通常指：

```text
小模型上几乎看不到，模型变大后突然表现出来的能力。
```

例如某些多步推理、复杂指令跟随、少样本任务、符号操作，在小模型中接近随机，到了某个规模后明显提升。

这个现象很吸引人，但要谨慎理解。

第一，涌现可能和评估指标有关。

如果指标是离散 accuracy，模型从 49% 到 51% 可能看起来像突然跨过阈值。连续指标下趋势可能更平滑。

第二，涌现不一定只由参数量决定。

训练数据、prompt 格式、chain-of-thought、评估任务、采样策略都会影响是否观察到涌现。

第三，涌现不代表能力完全不可研究。

它提醒我们小规模实验可能无法预测所有能力，但仍可以通过更细的指标、更好的 scaling 实验和机制分析来理解。

更稳妥的表述是：

```text
Scaling 通常带来平滑 loss 改善，但部分下游能力可能在某些评估上表现出非线性或阈值式变化。
```

## 9.18 Transformer 的表达力和 Scaling 为什么能配合

一个架构适合 scaling，至少需要满足几个条件。

第一，容量可扩展。

```text
增加层数、宽度、head 数、FFN hidden size 后，模型能有效利用新增参数。
```

第二，优化可稳定。

```text
模型变深变宽后，loss 仍能稳定下降，不频繁发散。
```

第三，硬件可扩展。

```text
新增计算能高效分摊到更多 GPU/TPU 上。
```

第四，数据可吸收。

```text
模型能从海量多样文本、代码、数学、对话中学习可迁移模式。
```

Transformer 恰好在这些方面都表现不错。

这不是说 Transformer 是理论上唯一选择，而是说它在过去几年形成了最成熟的 scaling 证据和工程生态。

## 9.19 为什么不是 CNN 或 RNN 成为 LLM 主干

RNN/LSTM 的问题主要是串行依赖和长距离路径。

```text
优点：有状态、天然适合流式。
缺点：训练并行性差，长距离信息需要多步传递，大规模训练生态弱。
```

CNN 的问题主要是局部感受野。

```text
优点：并行、局部性强、硬件友好。
缺点：远距离依赖需要多层或大卷积核，动态内容路由不如 attention 灵活。
```

Transformer 的优势是：

```text
既能并行训练，又能让任意 token 直接交互，还能把计算变成大矩阵乘法。
```

但这不代表 RNN/CNN 思想无价值。后续 Mamba、RWKV、RetNet、Hyena 等架构，很多都在重新探索状态、卷积、线性时间和硬件效率，只是它们需要在表达力、并行训练和生态上追赶 Transformer。

## 9.20 Scaling 的边界

Scaling 很强，但不是万能。

它至少有以下边界。

第一，loss 改善不等于所有能力等比例改善。

```text
困惑度下降，不保证数学推理、安全拒答、工具调用、长上下文检索都同步提升。
```

第二，数据质量可能成为瓶颈。

低质量、高重复、污染数据会浪费 token budget。

第三，训练稳定性是前提。

loss spike、NaN、分布式错误、mask 错误、优化器状态异常都会破坏 scaling 趋势。

第四，推理成本会成为约束。

更大模型训练后，部署延迟、显存、吞吐、KV cache 都可能成为瓶颈。

第五，安全和对齐不自动随规模解决。

规模变大可能提高能力，也可能放大幻觉、偏见、越狱、隐私记忆等风险。

第六，架构瓶颈仍然存在。

Transformer 的 O(n^2) attention、长上下文检索失败、状态记忆不足、流式低延迟困难，都是后续章节要讲的核心瓶颈。

## 9.21 并行与 Scaling 审计指标与最小 demo

工程里判断一个 Transformer scaling 方案，不能只看“参数更大”或“toy loss 更低”。至少要同时看四类门禁：

```math
G_{\mathrm{scale}}=
\mathbf{1}[C_{\mathrm{path}}=1]\cdot
\mathbf{1}[C_{\mathrm{balance}}=1]\cdot
\mathbf{1}[C_{\mathrm{data}}=1]\cdot
\mathbf{1}[C_{\mathrm{serve}}=1]
```

其中：

```math
C_{\mathrm{path}}=\mathbf{1}[P_{\mathrm{model}}\le P_{\mathrm{max}}]
```

表示序列内信息路径是否足够短。Transformer 的一层 attention 可以让任意可见 token 直接交互；RNN 的最坏路径随序列长度增长；小卷积核 CNN 需要多层扩大感受野。

```math
C_{\mathrm{balance}}=
\mathbf{1}[\tau_{\min}\le D/N\le\tau_{\max}]
```

表示训练 token 数和参数量是否大致平衡。这里的 `tau_min` 和 `tau_max` 是教学 demo 的经验门槛，不是 Chinchilla 原论文的通用比例。

```math
C_{\mathrm{serve}}=\mathbf{1}[S_{\mathrm{infer}}\le S_{\max}]
```

表示候选模型的推理压力是否在上线预算内。训练 loss 更低的模型，如果推理成本太高，也未必是业务上最优选择。

下面 demo 用纯 Python 做一个简化审计。它不训练模型，也不拟合真实 scaling law，只用 toy 公式展示三件事：

1. Transformer 用 `n^2` 个 attention pair 换来一层内长距离交互。
2. 固定训练 compute 时，参数变大意味着可训练 token 变少。
3. toy loss 最低的候选，不一定同时满足 token/parameter balance 和 serving gate。

```python
import math


def train_flops(params_b, tokens_b):
    return 6 * params_b * tokens_b


def toy_loss(params_b, tokens_b, data_quality=1.0):
    model_term = 0.55 * params_b ** -0.32
    data_term = 0.90 * (tokens_b * data_quality) ** -0.28
    return 1.35 + model_term + data_term


def architecture_cost(seq_len, kernel=5):
    return {
        "rnn_steps": seq_len,
        "cnn_layers_for_full_range": math.ceil((seq_len - 1) / (kernel - 1)),
        "transformer_path_length": 1,
        "attention_score_millions": round(seq_len * seq_len / 1_000_000, 2),
        "attention_vs_rnn_pair_ratio": seq_len,
    }


def compute_candidates(budget_flops_b2, data_quality):
    rows = []
    for params_b in [1, 3, 7, 13, 30]:
        tokens_b = budget_flops_b2 / (6 * params_b)
        tokens_per_param = tokens_b / params_b
        loss = toy_loss(params_b, tokens_b, data_quality)
        serving_pressure = min(1.0, params_b / 30)
        balanced = 10 <= tokens_per_param <= 40
        rows.append({
            "params_b": params_b,
            "tokens_b": round(tokens_b, 1),
            "tokens_per_param": round(tokens_per_param, 1),
            "toy_loss": round(loss, 4),
            "serving_pressure": round(serving_pressure, 3),
            "balanced": balanced,
        })
    return rows


def scaling_gate(best, data_quality, train_stability, serving_limit):
    failed = []
    if not best["balanced"]:
        failed.append("compute_balance")
    if data_quality < 0.8:
        failed.append("data_quality")
    if not train_stability:
        failed.append("train_stability")
    if best["serving_pressure"] > serving_limit:
        failed.append("serving_pressure")
    return failed, not failed


seq_len = 4096
budget = train_flops(7, 140)
records = compute_candidates(budget, data_quality=0.92)
loss_best = min(records, key=lambda row: row["toy_loss"])
ready_candidates = [
    row for row in records
    if row["balanced"] and row["serving_pressure"] <= 0.7
]
ready_best = min(ready_candidates, key=lambda row: row["toy_loss"])
failed, gate_pass = scaling_gate(
    ready_best,
    data_quality=0.92,
    train_stability=True,
    serving_limit=0.7,
)

print("arch_cost=", architecture_cost(seq_len))
print("compute_budget_1e21_flops=", round(budget / 1000, 3))
print("records=", records)
print("loss_best=", loss_best)
print("ready_best=", ready_best)
print("failed_gates=", failed)
print("scaling_gate_pass=", gate_pass)
```

一组固定输出如下：

```text
arch_cost= {'rnn_steps': 4096, 'cnn_layers_for_full_range': 1024, 'transformer_path_length': 1, 'attention_score_millions': 16.78, 'attention_vs_rnn_pair_ratio': 4096}
compute_budget_1e21_flops= 5.88
records= [{'params_b': 1, 'tokens_b': 980.0, 'tokens_per_param': 980.0, 'toy_loss': 2.0339, 'serving_pressure': 0.033, 'balanced': False}, {'params_b': 3, 'tokens_b': 326.7, 'tokens_per_param': 108.9, 'toy_loss': 1.9191, 'serving_pressure': 0.1, 'balanced': False}, {'params_b': 7, 'tokens_b': 140.0, 'tokens_per_param': 20.0, 'toy_loss': 1.876, 'serving_pressure': 0.233, 'balanced': True}, {'params_b': 13, 'tokens_b': 75.4, 'tokens_per_param': 5.8, 'toy_loss': 1.8667, 'serving_pressure': 0.433, 'balanced': False}, {'params_b': 30, 'tokens_b': 32.7, 'tokens_per_param': 1.1, 'toy_loss': 1.8823, 'serving_pressure': 1.0, 'balanced': False}]
loss_best= {'params_b': 13, 'tokens_b': 75.4, 'tokens_per_param': 5.8, 'toy_loss': 1.8667, 'serving_pressure': 0.433, 'balanced': False}
ready_best= {'params_b': 7, 'tokens_b': 140.0, 'tokens_per_param': 20.0, 'toy_loss': 1.876, 'serving_pressure': 0.233, 'balanced': True}
failed_gates= []
scaling_gate_pass= True
```

这组结果可以这样读：

1. `seq_len=4096` 时，RNN 最坏串行步数是 4096，小卷积核 CNN 要很多层扩大感受野，而 Transformer 的可见 token 一层内路径长度为 1。
2. Transformer 的代价是 attention score 有 1678 万个 pair，因此下一章要继续讲 O(n^2) 成本和长上下文瓶颈。
3. 在固定 toy compute 下，1B/3B 模型 token 很多但容量小，13B/30B 模型 token-per-parameter 太低，可能 undertrained。
4. `loss_best` 是 13B，但它没有通过 balance gate；`ready_best` 是 7B，因为它同时满足 toy loss、tokens-per-parameter 和 serving pressure。
5. demo 数字只用于教学，不是任何真实模型的 benchmark。真实训练要结合数据质量、优化稳定性、硬件效率、评估和上线成本。

## 9.22 面向专家：Scaling Law 的正确用法

在真实项目中，scaling law 更像一个资源规划工具，而不是最终真理。

正确用法包括：

1. 用小模型和中模型拟合 loss vs compute 趋势。
2. 检查大模型训练曲线是否偏离预测。
3. 比较不同数据配比的有效性。
4. 估算给定预算下模型大小和 token 数。
5. 判断模型是否 undertrained。
6. 评估继续训练、扩模型、换数据哪个收益更高。

错误用法包括：

1. 把某篇论文的指数当成跨所有模型固定常数。
2. 只看参数量，不看 token 数。
3. 只看 token 数，不看数据质量。
4. 用 loss 直接预测所有复杂 benchmark。
5. 忽略训练稳定性和系统瓶颈。
6. 把 Chinchilla 简化成绝对比例公式。

面试中能说清这些边界，会比只背“loss 幂律下降”更有深度。

## 9.23 面向专家：表达力、优化和归纳偏置的张力

Transformer 表达力强，但归纳偏置相对弱。

它不像 CNN 那样强假设局部平移不变，也不像 RNN 那样天然维护递推状态。它更多依赖数据和规模去学出结构。

这带来两面性。

好处是：

```text
足够通用，可以统一文本、代码、多模态 token、工具调用轨迹等多种序列。
```

代价是：

```text
需要大量数据和计算来学出本可由强归纳偏置提供的规律。
```

这也是为什么后续架构创新经常在做两件事：

1. 保留 Transformer 的通用性和并行性。
2. 引入更合适的归纳偏置或效率改进。

例如：

1. RoPE 引入相对位置结构。
2. Sliding window 引入局部性。
3. GQA/MQA 降低 KV cache 成本。
4. MoE 扩大参数容量但保持稀疏激活。
5. Mamba/RWKV/RetNet 重新引入状态和线性时间处理。

## 9.24 常见误区

### 误区 1：Transformer 比 RNN 更省计算

不一定。Transformer attention 有 O(n^2) 成本。它的优势主要是并行性和硬件友好，而不是所有情况下计算量更少。

### 误区 2：Scaling law 说明只要模型变大就一定更好

不对。参数、数据、compute、数据质量、训练稳定性需要匹配。Chinchilla 已经说明很多大模型可能 undertrained。

### 误区 3：涌现能力说明大模型能力完全不可预测

不准确。loss scaling 往往可预测，但部分下游能力在特定指标上可能呈现非线性或阈值效应。

### 误区 4：Attention 是 Transformer 表达力的全部来源

不对。Attention 负责路由，FFN 负责非线性计算，residual stream 负责跨层状态积累，norm 和初始化保证训练稳定。

### 误区 5：Scaling law 是理论定律

Scaling law 是经验规律，依赖实验条件，不能脱离模型族、数据分布和训练 recipe 使用。

### 误区 6：更低 loss 一定意味着更安全、更可靠

不一定。安全、事实性、鲁棒性、隐私、对齐需要专门数据、训练和评估。

## 9.25 面试高频问题

### 题 1：Transformer 为什么适合大规模并行训练？

参考回答：

```text
Transformer 没有 RNN 那种强时间递推依赖。训练时整段序列可以一次性计算 Q/K/V 和所有 attention scores，大部分操作都是矩阵乘法，适合 GPU/TPU 并行。虽然 attention 有 O(n^2) 成本，但这些计算规则、密集、并行度高，因此在大规模硬件上非常高效。
```

### 题 2：Self-Attention 的表达力来自哪里？

参考回答：

```text
Self-Attention 的表达力来自动态信息路由。每个 token 根据自己的 query 和其他 token 的 key 动态决定读取哪些 value。相比固定卷积邻域，它可以一层内让任意两个 token 交互，并且路由模式依赖内容。多头注意力还能在多个子空间并行学习不同关系。
```

### 题 3：Transformer 中 attention 和 FFN 分别负责什么？

参考回答：

```text
可以粗略理解为 attention 负责 token 间信息交换和上下文路由，FFN 负责每个 token 位置上的非线性计算。Residual stream 在层间传递和积累状态，norm 和初始化保证深层训练稳定。Transformer 的表达力不是 attention 单独提供的，而是这些模块共同作用。
```

### 题 4：Scaling law 对大模型训练有什么意义？

参考回答：

```text
Scaling law 说明在一定模型族、数据和训练设置下，语言模型 loss 随参数量、数据量和计算量增加呈现相对平滑的幂律下降。这让大模型训练从纯经验堆规模变成更可规划的资源分配问题，可以用小规模实验预测大模型趋势、选择模型大小和 token 数，并监控训练是否偏离预期。
```

### 题 5：Chinchilla 的核心启发是什么？

参考回答：

```text
Chinchilla 的核心启发是固定 compute 下参数量和训练 token 数需要平衡。很多早期大模型参数很大但训练 token 不足，因此 undertrained。一个较小但训练更充分的模型，可能超过更大但数据不足的模型。它推动行业从只追求参数量转向 compute-optimal training 和高质量数据。
```

### 题 6：GPT-3 的 few-shot 能力说明了什么？

参考回答：

```text
GPT-3 展示了大规模 decoder-only Transformer 可以通过 prompt 中的任务说明和少量示例完成新任务，而不需要梯度更新。这说明 scaling 后模型能利用上下文进行模式识别、检索和条件生成，也推动了 in-context learning 范式。但这不等同于参数级学习，也不代表模型像人一样真正理解任务。
```

### 题 7：如何理解 emergent abilities？

参考回答：

```text
涌现能力指一些能力在小模型上不明显，到了更大规模后突然出现。它说明部分下游能力可能不是简单线性可外推的。但要谨慎，因为涌现现象可能受评估指标、prompt、数据和阈值影响。更稳妥的说法是，loss 往往平滑改善，但某些任务能力可能呈现非线性或阈值式变化。
```

### 题 8：Scaling 的边界是什么？

参考回答：

```text
Scaling 不是万能。loss 改善不保证所有任务、安全性、事实性、长上下文能力都同步提升。数据质量、训练稳定性、推理成本和对齐都会成为瓶颈。Scaling law 也是经验规律，依赖模型族、数据、训练 recipe 和规模范围，不能当成无条件定律。
```

## 9.26 小练习

1. 用一张表比较 RNN、CNN、Transformer 在并行性、长距离依赖和硬件友好性上的差异。
2. 解释为什么 Transformer 的 attention 是 O(n^2)，但仍然适合大规模训练。
3. 用自己的话说明 Q/K/V 如何构成动态信息路由。
4. 解释多头注意力为什么比单头更有表达力。
5. 阅读 Kaplan scaling laws 摘要，说明 loss、参数、数据、compute 之间的关系。
6. 阅读 Chinchilla 摘要，解释为什么更大模型不一定 compute-optimal。
7. 讨论 GPT-3 few-shot learning 和 fine-tuning 的区别。
8. 举一个例子说明“loss 下降不等于某项能力必然提升”。

## 9.27 本章总结

本章讲了 Transformer 的并行性、表达力和 scaling 优势。

核心结论：

1. Transformer 摆脱 RNN 的时间步串行依赖，训练时序列内部可以高度并行。
2. Transformer 的核心计算是规则的大矩阵乘法，非常适合 GPU/TPU 和分布式训练。
3. Self-Attention 提供动态内容路由，让任意 token 可以一层内直接交互。
4. 多头注意力、FFN、residual stream、normalization 共同构成 Transformer 的表达力。
5. Scaling law 让大模型训练从经验扩展走向更可预测的资源规划。
6. Chinchilla 强调 compute-optimal training，参数量和训练 token 数必须匹配。
7. GPT-3、PaLM 等工作展示了大规模 Transformer 的 few-shot、reasoning 和多任务能力。
8. Scaling 有边界，不能忽视数据质量、训练稳定性、推理成本、安全对齐和架构瓶颈。

下一章会进入 O(n^2) Attention 的计算与显存瓶颈，解释 Transformer 的核心优势为什么也带来了长上下文和推理成本问题。
