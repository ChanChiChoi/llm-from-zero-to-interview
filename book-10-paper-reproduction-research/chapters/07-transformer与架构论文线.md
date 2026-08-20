# 第七章：Transformer 与架构论文线

读一篇架构论文，最容易做的事情是画出模块图，最难做的事情是判断它到底改变了什么。一个新模块可能减少理论复杂度，却因为 kernel、通信和内存访问不成熟而更慢；一个模型可能在 benchmark 上提高，却同时使用更多训练 token、更大的数据集和更宽松的推理预算；一个结构可能在长序列复制任务上有效，却在需要精确随机访问的任务上失效。

因此，架构论文不应被读成“新名字列表”。它们更像一条围绕瓶颈不断移动的实验史：RNN 的递归限制了并行训练，Transformer 用全局 attention 换取并行性；decoder-only 与 next-token prediction 形成统一生成接口；RoPE、ALiBi 和各种长上下文方法处理顺序与距离；GQA、MLA 和其他状态压缩方法处理推理内存；MoE 用路由换取稀疏容量；SSM 和混合架构尝试把长序列成本从二次路径转移到线性状态更新。

初学者可以把每篇架构论文先问成三个问题：它想修哪一个瓶颈？它改变了计算图的哪一段？这个变化带来了什么新代价？专家还要继续问：在相同数据、参数、训练 token、总计算、硬件和评估 harness 下，收益是否仍然存在？论文是否做了足够的消融，证明提升来自架构而不是训练配方？

## 1. 架构论文首先是瓶颈论文

### 1.1 结构名称不是研究问题

“提出一种新的 attention”不是完整的研究问题。完整的问题应当包含工作负载和约束，例如：

- 在训练阶段，如何让长序列的 token 交互保持可并行，同时减少中间激活显存？
- 在 decode 阶段，如何在质量接近的情况下减少每个请求的 KV Cache 和内存带宽？
- 当模型规模变大时，如何增加参数容量，却不让每个 token 都承担完整的矩阵计算？
- 当序列长度增长时，如何保持状态更新便宜，同时不丢失精确的远距离信息？

同一个模块在不同问题下可能有不同结论。GQA 主要改变 decode 阶段的 K/V 状态规模，不应被描述成“解决了所有长上下文问题”；MoE 主要改变容量与激活计算的关系，不应被描述成“参数越多计算越便宜”；SSM 主要提供线性状态路径，不自动保证对任意文本任务都有 attention 一样的随机访问能力。

### 1.2 三类架构收益

架构改动的收益可以分成三类：

| 收益类别 | 观察对象 | 典型指标 | 典型风险 |
| --- | --- | --- | --- |
| 表达能力 | 可表示的依赖、记忆和组合关系 | 任务质量、长距离检索、泛化 | 训练配方和数据影响难分离 |
| 训练效率 | 并行度、显存、收敛和吞吐 | tokens/s、MFU、训练曲线、GPU 小时 | 理论 FLOPs 不等于实测时间 |
| 推理效率 | decode 状态、带宽、延迟和容量 | TTFT、TPOT、P99、KV 显存、单位成本 | batch、硬件和 runtime 影响很大 |

一篇论文可能同时声称三类收益，但证据往往只覆盖其中一类。若论文只报告 BLEU 或准确率，不能自动推出训练更快；若只报告 kernel 吞吐，不能自动推出任务质量更高。读者要把每个 claim 和对应证据配对。

### 1.3 先画变量账本

在读实验表前，先记下：

1. 模型结构和每层参数。
2. 总参数量与 active parameters。
3. 训练 token、数据 mixture 和数据过滤。
4. batch、序列长度、优化器、学习率和训练步数。
5. 硬件、精度、kernel、并行方式和通信拓扑。
6. 推理 batch、输入/输出长度、缓存策略和采样参数。
7. 评估任务、prompt、后处理、重试和 judge。

如果其中很多字段没有公开，结论的证据等级就应相应收窄。架构论文的图可以告诉你“作者想改变哪一段计算”，但不能仅凭图证明端到端收益。

## 2. Transformer：把序列建模改成可并行的信息路由

### 2.1 Transformer 之前的约束

在 Transformer 之前，序列到序列模型常依赖 RNN 或 CNN。RNN 在时间步上递归：当前状态由上一个状态和当前输入共同决定。它的优点是天然有顺序，但训练阶段必须等待前一个时间步，长距离信息还要通过很多次状态传递。

CNN 可以并行处理一段序列，但有限卷积核只能直接看到局部邻域；要让两个相距很远的 token 交互，需要增加层数、扩大 kernel 或设计空洞卷积。这样会增加路径长度和结构复杂度。

Transformer 的关键变化是让每个位置直接根据整段序列中的其他位置计算加权信息。训练时，整段序列可以作为矩阵同时处理；代价是标准全 attention 对长度 $T$ 有二次的 token-pair 交互。

### 2.2 Self-attention 的数学对象

设输入表示矩阵为 $X\in\mathbb{R}^{T\times d_{model}}$，通过三个投影得到：

~~~math

Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.

~~~

$Q$ 可以理解为当前位置在寻找什么，$K$ 表示每个位置能提供什么索引，$V$ 是真正被聚合的内容。单头 attention 写成：

~~~math

\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V.

~~~

$d_k$ 是 key 的维度，$M$ 是 mask 或其他位置偏置。因果语言模型中，当前位置不能读取未来 token，$M$ 会把上三角位置设为极小值；encoder 的双向 attention 则可以让合法位置互相读取。

缩放因子 $1/\sqrt{d_k}$ 防止点积随维度增大而过大，避免 softmax 进入过饱和区域。这个公式的重要之处不是记住三个字母，而是看清信息流：先计算所有位置之间的相似性，再用权重对 $V$ 做内容聚合。

### 2.3 Multi-head 为什么不是简单复制

多头 attention 将表示投影到多个子空间：

~~~math

\mathrm{MHA}(X)
=\mathrm{Concat}(H_1,\ldots,H_h)W_O,
\qquad
H_j=\mathrm{Attention}(XW_Q^{(j)},XW_K^{(j)},XW_V^{(j)}).

~~~

不同 head 可以学习不同的关系，例如局部语法、实体指代、位置对应或长距离依赖。多头不是保证每个 head 都有清晰语义，也不是 head 越多越强；它提供的是多个可学习的路由子空间，实际是否形成有用分工要通过分析和消融验证。

### 2.4 路径长度和复杂度的取舍

标准 self-attention 的分数矩阵包含 $T\times T$ 个位置关系，粗略计算和显存复杂度可写为：

~~~math

\mathrm{AttentionCost}=O(T^2d),
\qquad
\mathrm{ScoreMemory}=O(T^2).

~~~

它的优势是任意两个位置之间的路径长度接近常数；代价是长度增加时 pairwise 交互迅速膨胀。FlashAttention 等方法可以减少中间矩阵的物化和内存访问，但不会把标准 attention 的信息依赖自动变成线性复杂度。读论文时必须区分算法复杂度、实际 kernel 内存和端到端服务成本。

### 2.5 Transformer 的贡献边界

Attention Is All You Need 的历史贡献可以拆成：

1. 用 attention 作为主要序列交互机制，去掉 recurrent/convolutional 主干。
2. 用 multi-head 提供多个关系子空间。
3. 用位置表示补回顺序信息。
4. 用残差、归一化和 FFN 组成可堆叠的 block。
5. 在序列到序列任务中同时获得并行训练和较短的依赖路径。

这并不意味着 Transformer 在所有维度都优于 RNN/CNN。在线流式场景、极长序列、严格低功耗设备和需要固定状态的场景，attention 的缓存和带宽成本可能成为主要瓶颈。架构论文的结论应绑定 workload，而不是绑定一个抽象的“先进/落后”标签。

## 3. GPT 与 BERT：同一个家族的不同训练问题

### 3.1 架构、目标和接口要分开

Transformer 只是一个结构模板。GPT 和 BERT 的差异至少有三层：

| 层次 | GPT 路线 | BERT 路线 |
| --- | --- | --- |
| 主体结构 | decoder-only，因果 mask | encoder-only，双向可见 |
| 预训练目标 | next-token prediction | masked language modeling 等 |
| 下游接口 | 生成序列、prompt、completion | 表示、分类、抽取和双向编码 |

把“GPT 是 decoder、BERT 是 encoder”说完还不够，因为训练目标决定了模型学到的条件分布和使用方式。把 BERT 的 masked LM 直接当成 GPT 的 next-token prediction，也会误解输入可见性、输出位置和推理接口。

### 3.2 GPT：把语言建模变成统一生成接口

decoder-only 模型对序列 $x_1,\ldots,x_T$ 最大化：

~~~math

\log P(x_1,\ldots,x_T)
=\sum_{t=1}^{T}\log P(x_t\mid x_{1:t-1}).

~~~

这里的因果条件使训练目标和生成过程天然一致：训练时预测下一个 token，推理时把已经生成的 token 放回上下文继续预测。prompt、对话历史、工具返回和答案都可以序列化到同一接口中。

这个统一性带来工程优势，也带来代价。模型必须通过自回归目标间接学习双向关系；推理是逐 token 的，KV Cache、解码调度和长上下文内存成为核心系统问题。

### 3.3 BERT：双向表示与预训练迁移

BERT 让每个被 mask 的位置读取左右上下文，再通过 fine-tuning 适配分类、问答、序列标注等任务。它证明了深层双向预训练表示可以迁移到多个理解任务，但其原始接口不是自然语言生成器。

读 GPT/BERT 论文时，要把“预训练效果强”拆成三个问题：预训练目标是否更适合任务？模型结构是否提供了必要的信息可见性？下游适配是否使用了额外标注和参数？只有这样，才能避免把训练目标的贡献误写成架构贡献。

### 3.4 Encoder-decoder 仍然有自己的工作负载

encoder-decoder 允许 encoder 双向读取输入，再由 decoder 自回归生成输出，适合翻译、摘要和条件生成。decoder-only 的流行并不意味着 encoder-decoder 在所有场景都被淘汰；当输入理解和输出生成的职责差异明显，encoder 的双向表示仍可能具有优势。

架构比较应问“输入和输出是什么”，而不是只问“哪个架构更先进”。一个统一大模型接口可以简化产品，但可能用更多生成 token 或更大的上下文来弥补 encoder 的特长；这属于系统级 trade-off。

## 4. Decoder-only 为什么成为大模型主流

### 4.1 统一序列接口

decoder-only 可以把 instruction、conversation、retrieved evidence、tool result 和 answer 都表示成一条带边界标记的序列。训练目标仍然是 next-token prediction，数据工程和推理 runtime 因此容易共享。

规模化还需要统一的并行策略、tokenizer、checkpoint、KV Cache 和 serving 生态。decoder-only 在这些方面形成了强大的路径依赖，使新模型能快速复用已有工具链。

### 4.2 主流不是没有代价

decoder-only 的代价包括：

1. 标准 attention 对长上下文的计算和显存压力。
2. decode 阶段必须维护历史 K/V 状态。
3. 双向输入关系要通过训练数据和上下文组织学习。
4. 生成任务的 token 成本会随答案长度增长。
5. 多轮、工具和 reasoning 会把上下文变成不断扩大的状态。

因此后续架构论文大量围绕 attention、位置、KV Cache、稀疏激活、状态空间和混合 block 展开。所谓“架构演进”，很大一部分是在保持 decoder-only 接口的同时重塑内部信息流和资源账本。

### 4.3 不同任务需要不同比较

在纯文本生成 benchmark 上，decoder-only 的目标匹配优势明显；在检索编码、分类、双向理解或低延迟流式任务上，encoder 或状态模型可能更合适。论文若只在单一任务上比较，不应被外推成架构家族的全面排名。

## 5. 位置编码：把顺序和距离注入 attention

### 5.1 为什么 attention 需要位置

如果只给 attention 输入 token 表示而不加入位置，序列中 token 的排列可能被视为一个集合，模型很难区分“狗咬人”和“人咬狗”。位置机制提供了顺序、距离或相对偏置，但不同方法把位置信息放在不同位置：输入表示、attention 分数、Q/K 旋转或相对距离函数。

### 5.2 Sinusoidal 与 learned absolute position

经典正弦位置编码用不同频率构造位置向量：

~~~math

PE_{(p,2i)}=\sin\left(p/10000^{2i/d}\right),
\qquad
PE_{(p,2i+1)}=\cos\left(p/10000^{2i/d}\right).

~~~

$p$ 是位置，$i$ 是维度索引，$d$ 是表示维度。它不需要为每个位置学习独立参数，理论上可以计算更长位置；learned absolute embedding 则为训练范围内的位置学习向量，超出训练长度时通常需要额外处理。

这两类方法的关键问题不是公式是否漂亮，而是训练长度与推理长度不一致时会发生什么。位置外推、插值和重新训练都可能改变远距离注意力的分布。

### 5.3 RoPE：让相对位置信息进入 Q/K 几何

RoPE 对 Q、K 的二维子空间做随位置变化的旋转。用二维向量表示一个局部子空间时：

~~~math

R(p)=
\begin{bmatrix}
\cos(p\theta)&-\sin(p\theta)\\
\sin(p\theta)&\phantom{-}\cos(p\theta)
\end{bmatrix}.

~~~

对位置 $p$ 的 query 和 key 分别旋转后，它们的内积会携带与位置差相关的相位关系。RoPE 的优势是把相对位置性质融入 attention 点积，并与 decoder-only 模型自然结合；长上下文时，频率分布、缩放策略、训练覆盖和数值范围仍然决定能否稳定外推。

读 RoPE 改进论文，要追问：是改变频率、缩放位置、重新训练长序列，还是改变 attention 结构？如果只看到最大 context length 增大，却没有长距离任务、位置分层和成本数据，不能把接口长度当成有效理解能力。

### 5.4 ALiBi：直接给 attention 分数加距离偏置

ALiBi 的思想是在 attention logits 中加入与相对距离相关的线性惩罚：

~~~math

S_{ij}=\frac{q_i k_j^{\mathsf T}}{\sqrt{d_k}}-m_h|i-j|,

~~~

其中 $m_h$ 是第 $h$ 个 head 的斜率。它不需要把绝对位置向量加入 token 表示，而是直接改变不同距离的注意力偏好。

RoPE、ALiBi 和插值类方法体现了不同归纳偏置：一个强调旋转后的相对几何，一个强调距离惩罚，一个可能重新映射训练位置到更长范围。架构论文不能只按“支持多少 K”排序，还要看中间位置、跨段组合、冲突证据和真实 decode 成本。

### 5.5 位置论文的实验应如何读

一个有说服力的位置实验至少应包含：

1. 训练长度和推理长度的明确关系。
2. 短、中、长长度的质量曲线，而不是只报最长点。
3. 证据位于开头、中间和结尾的分层结果。
4. 单证据、双证据和冲突证据任务。
5. 位置变化是否需要继续训练、额外 token 或新的 checkpoint。
6. prefill、decode、KV 和服务成本。

如果方法只在 synthetic needle 任务上有效，结论应限制为定位能力信号；如果真实文档中的多证据推理没有改善，不能把 synthetic 结果写成通用长上下文能力。

## 6. 归一化和残差：决定深层训练能否工作

### 6.1 残差让信息和梯度有短路径

设一个 block 的变换为 $F$，残差输出可写成：

~~~math

x_{l+1}=x_l+F(x_l).

~~~

这条恒等路径让信息可以跨层传递，也让梯度不必完全穿过每个非线性变换。没有残差，深层网络更容易出现优化困难；但残差的尺度、归一化位置和初始化仍然影响稳定性。

### 6.2 LayerNorm 与 RMSNorm

LayerNorm 对一个 token 的特征维度做中心化和缩放：

~~~math

\mathrm{LN}(x)=\gamma\odot\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}+\beta.

~~~

RMSNorm 不减去均值，只使用均方根进行缩放：

~~~math

\mathrm{RMSNorm}(x)=g\odot\frac{x}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}.

~~~

它通常减少部分计算和参数，但“更便宜”不等于“必然更强”。归一化方法还会影响表示尺度、残差累积、混合精度稳定性和训练超参。

### 6.3 Pre-Norm 与 Post-Norm

Pre-Norm 把归一化放在子层之前，示意为：

~~~math

x_{l+1}=x_l+F(\mathrm{Norm}(x_l)).

~~~

Post-Norm 则先做残差再归一化：

~~~math

x_{l+1}=\mathrm{Norm}(x_l+F(x_l)).

~~~

Pre-Norm 往往更容易训练很深的网络，因为残差路径保持直接；Post-Norm 可能有不同的表示和最终质量特性，通常需要更细致的初始化与优化设置。论文若声称“归一化改动提升稳定性”，需要同时看 loss 曲线、梯度、可训练深度、学习率和最终质量，不能只看一次最终 benchmark。

## 7. FFN 与 SwiGLU：attention 之外的容量来源

### 7.1 FFN 做什么

一个 Transformer block 通常在 attention 后接逐 token 的前馈网络：

~~~math

\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2.

~~~

attention 负责跨 token 路由，FFN 在每个位置上进行通道混合和非线性变换。它不直接把不同 token 聚合起来，却承载大量参数和表示变换，不能把模型能力简单归因给 attention。

### 7.2 门控线性单元

SwiGLU 一类门控 FFN 可以写成教学形式：

~~~math

\mathrm{SwiGLU}(x)
=\left(\mathrm{Swish}(xW_g)\odot xW_v\right)W_o.

~~~

门控分支决定哪些通道被保留，value 分支提供内容，最后投影回模型维度。它可能改善优化和表达能力，但通常伴随不同的中间维度和参数量。公平比较必须重新对齐参数、FLOPs 或训练预算；如果只把 GELU 换成 SwiGLU 同时扩大隐藏层，结果混入容量变化。

### 7.3 FFN 论文的证据

应检查：激活函数改变后，参数和计算是否匹配；训练曲线是否更稳定；不同模型规模趋势是否一致；代码、多语言和安全等护栏是否退化；实测 kernel 是否支持目标硬件。FFN 也可能成为 MoE 的专家单元，因此一个局部 FFN 改动会影响路由、容量和通信，不应脱离系统看。

## 8. LLaMA：现代开源 LLM 配方的组合价值

### 8.1 组合配方而不是单点模块

LLaMA 论文的教学价值不应被缩减成“用了 RMSNorm、SwiGLU 和 RoPE”。它展示的是一个相互配合的 foundation language model 配方：decoder-only 主体、预训练数据、训练 token、优化设置、模型规模和评估一起构成结果。

论文中的结构选择后来成为许多开源模型的基线，但这不意味着每个局部组件都是 LLaMA 首次提出，也不意味着只复制结构就能复制能力。数据质量、训练 token、去重、计算预算和评估设置同样影响最终结论。

### 8.2 规模与 token 的关系

当两个模型参数量不同，比较时还要看每个模型训练了多少 token。模型小而 token 多，可能比模型大而 token 少更接近 compute-optimal；若只看参数量，容易把训练充分程度误当成结构优势。

可以把每个实验对象写成：

~~~math

M=(P,\;D,\;F,\;A,\;H),

~~~

其中 $P$ 是参数量，$D$ 是训练 token，$F$ 是训练计算，$A$ 是架构，$H$ 是训练配方。架构 claim 要尽量在 $P,D,F,H$ 相近时改变 $A$；如果论文展示的是完整产品方案，则应诚实地把它写成组合收益。

### 8.3 开放性也是系统贡献

LLaMA 的影响还来自可获得的权重、论文信息和社区复用。开放性不是架构机制，但它改变了研究生态、复现成本和下游工程。读论文时可以把它作为 ecosystem contribution 记录，却不要把开放性自动当成质量证据。

## 9. Mistral：GQA 与 Sliding Window 的资源取舍

### 9.1 GQA 改变 KV 状态的复制方式

标准 MHA 为每个 query head 配置独立的 K/V head。GQA 将多个 query head 共享较少的 K/V head，MQA 则进一步让所有 query head 共享一组 K/V。设 query head 数为 $n_q$、KV head 数为 $n_{kv}$、每个 head 维度为 $d_h$，则每层每个 token 的 KV 元素数量近似为：

~~~math

N_{KV}=2n_{kv}d_h.

~~~

前面的 2 对应 K 和 V。把 $n_{kv}$ 从 $n_q$ 降低，会降低 decode 阶段需要读取和保存的 KV 状态，但 query 端仍可保留多个子空间。质量是否保持，要看训练和结构配置，不能只依据内存公式。

### 9.2 KV Cache 的数量关系

对 $L$ 层、上下文长度 $T$、每个元素 $b$ 字节的模型，KV Cache 显存近似为：

~~~math

\mathrm{Memory}_{KV}
=2L T n_{kv}d_h b.

~~~

公式忽略页表、对齐、临时 buffer 和 batch，但能说明一个关键事实：KV Cache 随层数、长度和 KV head 数线性增长。GQA 的主要系统价值常常在 decode memory bandwidth 和并发容量，而不是改变训练阶段的 $O(T^2)$ attention 依赖。

### 9.3 Sliding Window 的收益与边界

滑动窗口 attention 只让每个 token 直接读取局部窗口 $w$，理论交互量从全局的 $T^2$ 降到近似 $Tw$。但信息要跨越很远距离，需要通过多层传播、特殊全局 token、稀疏连接或其他状态路径传递。

因此它适合局部语法、长文本中的局部处理和资源受限推理，却可能损害一次性远距离复制、多证据合并或精确跨段引用。论文读者应查看窗口大小、层间结构、任务长度曲线和真实硬件收益，而不是只把“窗口 attention”翻译成“长上下文已经解决”。

## 10. MoE：把容量、计算和通信拆开

### 10.1 Conditional computation 的基本形式

MoE 将一个 dense FFN 替换为多个 expert，由 router 为每个 token 选择少数专家。若第 $e$ 个 expert 的输出为 $f_e(x)$，router 权重为 $p_e(x)$，top-k 集合为 $S(x)$，可以写成：

~~~math

y=\sum_{e\in S(x)}p_e(x)f_e(x).

~~~

总参数量随 expert 数量增长，但每个 token 的激活计算只涉及选中的专家。于是必须同时记录 total parameters、active parameters、每 token FLOPs 和通信成本。

### 10.2 路由不是免费选择

router 需要把 token 分发到不同 expert。若某些 expert 被过多 token 选择，会出现 load skew：部分 expert 超载、其他 expert 空闲，导致 token 丢弃、padding 浪费、通信拥塞和训练不稳定。

一个简化的负载不均衡度可以写成：

~~~math

\mathrm{Skew}=\frac{\max_e c_e}{\mathrm{mean}_e(c_e)},

~~~

其中 $c_e$ 是 expert $e$ 收到的 token 数。理想情况下它接近 1，但真实任务和训练阶段可能出现偏斜。负载均衡辅助损失、capacity factor、token dropping 和 router z-loss 等机制都有质量与吞吐取舍。

### 10.3 MoE 的公平比较

把 8x7B 模型直接与 7B dense 模型比较“谁强”是不完整的。需要至少列出：

1. 总参数量和激活参数量。
2. 每 token 的 FLOPs。
3. 专家数、top-k 和 capacity。
4. all-to-all 通信量和拓扑。
5. 训练总计算与 token 数。
6. decode 吞吐、P99、显存和失败率。

MoE 可能在相近每 token 计算下提供更大容量，也可能因为通信和路由开销在小 batch 或单卡上不划算。结论应绑定集群和 workload。

## 11. SSM 与 Mamba：把长序列建模改成状态更新

### 11.1 线性状态空间模型

一个离散状态空间模型可以表示为：

~~~math

h_t=\bar A h_{t-1}+\bar B x_t,
\qquad
y_t=\bar C h_t.

~~~

$h_t$ 是固定大小的状态，$x_t$ 是输入，$y_t$ 是输出。与保存全部历史 token 的 attention 不同，SSM 试图把历史压缩到状态中，因此序列推进的状态成本与 $T$ 线性相关。

固定状态的优点是长流和逐步推理便宜；代价是状态压缩可能丢失任意细节，模型如何选择、保留和恢复信息成为核心问题。

### 11.2 Mamba 的 selective state space

传统 SSM 的参数通常固定或与位置规则相关。Mamba 的关键方向是让部分状态参数依赖当前输入，使模型可以选择哪些信息写入、保留或遗忘；同时用面向硬件的 scan 算法把递推结构变成训练可接受的并行实现。

这里要区分两个层次：

1. **数学层**：状态转移和选择参数提供了输入依赖的记忆机制。
2. **系统层**：scan、kernel、融合和内存访问决定理论线性复杂度是否变成实测吞吐。

一篇论文在数学层提出新状态更新，不等于所有硬件、batch 和长度上都更快。Mamba 的效率优势尤其需要看序列长度、训练/推理阶段、设备、kernel 和比较基线。

### 11.3 SSM 的能力边界

SSM 可能擅长流式处理、长序列扫描和低状态成本，但固定或压缩状态对精确随机访问、远程复制、多个冲突证据和复杂 in-context learning 的表现需要单独评估。不能把“线性复杂度”直接等同于“更好的长上下文理解”。

好的比较应包括：局部语法、长距离复制、变量绑定、多证据合并、顺序敏感、工具轨迹和真实流式任务；同时比较质量、状态内存、吞吐、首 token、每 token 延迟和训练稳定性。

## 12. 混合架构：不是二选一

### 12.1 不同模块承担不同信息任务

Attention、卷积、SSM 和 MoE 并非只能互相替代。混合架构的基本思想是让不同模块分担不同信息流：

- attention 处理精确、动态、跨位置的随机交互；
- SSM 或递归状态处理长流、局部扫描和压缩历史；
- convolution 处理局部模式和硬件友好的邻域混合；
- MoE FFN 扩大内容变换容量；
- cache 或 latent state 保存服务侧需要复用的历史表示。

混合结构的难点是模块边界、状态接口、训练稳定性、kernel 组合和调度复杂度。它可能在质量和成本之间取得更好折中，也可能因为每个模块都不够成熟而失去整体收益。

### 12.2 读混合架构要追踪状态

至少画出三条状态路径：

1. 当前 token 如何进入 attention 或 SSM。
2. 历史信息以完整 KV、压缩 latent 还是递归 state 保存。
3. 不同层之间如何交换、更新和淘汰状态。

如果论文只说“混合了 attention 和 SSM”，却没有说明状态形状、更新时机和训练/推理差异，读者还没有真正理解它的系统含义。

## 13. 架构论文的统一评价框架

### 13.1 九个问题

对 Transformer、MoE、SSM、位置编码和推理优化论文，可以沿着九个问题阅读：

1. **Bottleneck**：原系统的瓶颈是质量、计算、显存、带宽、通信还是延迟？
2. **Treatment**：论文具体改变了哪一段计算图或状态路径？
3. **Inductive bias**：它偏好局部、全局、递归、稀疏、平滑还是精确匹配？
4. **Budget**：参数、token、FLOPs、显存和调用次数怎样变化？
5. **Mechanism**：中间变量是否按假设改变？
6. **Evidence**：实验是否覆盖主张，而非只覆盖一个 benchmark？
7. **Scaling**：趋势是否跨模型、长度、数据和硬件规模？
8. **Deployment**：kernel、通信、缓存和生态是否能支持真实工作负载？
9. **Boundary**：在哪些任务、长度、batch、设备或风险条件下会失效？

### 13.2 把主张写成受控比较

例如，“GQA 提升推理效率”应拆成：

> 在相同模型层数、query head 数、精度、上下文长度、并发和质量容差下，减少 KV head 数是否降低 KV 状态读取和 P99 延迟？

“Mamba 更适合长上下文”应拆成：

> 在相同训练 token、模型规模和任务 harness 下，Mamba 或混合结构是否在指定长度范围内保持任务质量，同时降低状态内存或每 token 成本？

问题越具体，越容易发现论文没有测量什么。

### 13.3 常见混淆变量

架构论文中最常见的混淆包括：

1. 数据更多或质量不同。
2. 训练 token 更多或训练更久。
3. 参数量、active parameters 和 FLOPs 不匹配。
4. 新方法调参更充分。
5. baseline 使用旧实现或弱配置。
6. 评估 prompt、后处理和 judge 不同。
7. 新 kernel、编译器或硬件不同。
8. batch、上下文和请求长度不同。
9. 只报告吞吐，不报告质量和尾延迟。
10. 只报告最长上下文，不报告有效任务能力。

如果论文没有排除这些因素，读者可以认可“完整方案在当前设置下表现更好”，但不应无条件认可“架构机制本身更好”。

## 14. Worked case：比较 MHA、GQA 和 MQA

### 14.1 先算状态账本

设模型有 $L=32$ 层、每个 head 维度 $d_h=128$、上下文长度 $T=8192$，使用 BF16，每个元素 2 字节。比较三种 query/KV head 配置：

| 结构 | $n_q$ | $n_{kv}$ | 共享方式 |
| --- | ---: | ---: | --- |
| MHA | 32 | 32 | 每个 query head 一组 K/V |
| GQA | 32 | 8 | 四个 query head 共享一组 K/V |
| MQA | 32 | 1 | 所有 query head 共享一组 K/V |

KV Cache 显存用：

~~~math

\mathrm{Memory}_{KV}=2LTn_{kv}d_hb.

~~~

代入后，MHA 约为 4 GiB，GQA 约为 1 GiB，MQA 约为 0.125 GiB。这里没有计入 batch、页表、对齐和临时 buffer，实际 serving 账本会更大；但数量关系已经说明 GQA 为什么能显著释放 KV 状态空间。

### 14.2 质量和成本不能只看显存

GQA 可能降低 KV 读取，但它也改变了 K/V 表示容量。要验证“质量保持”，需要比较短上下文、长上下文、代码、数学、多语言和工具参数等切片；要验证“服务更快”，需要固定硬件、batch、输入/输出长度、编译和 cache 策略，测 TTFT、TPOT、吞吐和 P99。

如果 GQA 在质量容差内节省 75% KV 显存，并在目标 batch 下改善 P99，它的架构证据比较完整；如果只在单请求 benchmark 上显存下降、P99 没变，则更准确的结论是“状态占用下降”，不是“端到端服务更快”。

## 15. Worked case：MoE 与 dense 模型的公平比较

设 dense 模型有 7B 参数，MoE 模型有 8 个 7B 级 expert，每个 token top-2 激活。MoE 的总参数量约为 56B 加共享部分，但每个 token 只经过两个 expert 的 FFN 路径。下面三个数字必须分开：

- total parameters：影响权重存储和可能的容量。
- active parameters：影响每 token 的主要计算量。
- communication cost：影响跨卡路由、all-to-all 和尾延迟。

如果只报告 active 参数，可能低估权重加载、专家放置和通信；如果只报告 total 参数，又会误以为每个 token 承担 56B dense 计算。公平报告应同时给出质量、训练 FLOPs、每 token 激活、expert load skew、通信量、吞吐、P99 和显存。

在小 batch 单机上，通信可能占主要开销，dense 反而更快；在大规模集群和足够 batch 下，MoE 的容量优势可能更明显。这不是矛盾，而是 workload 改变了成本项权重。

## 16. Worked case：SSM 与 attention 的任务边界

假设一个 SSM 在百万 token 流式扫描上显著节省状态内存，但在远程复制和两个冲突实体的任务上下降。一个合理的报告表应是：

| 任务 | attention | SSM | 解释 |
| --- | ---: | ---: | --- |
| 局部语法 |  |  | 局部混合是否足够 |
| 远程复制 |  |  | 精确保存细节的能力 |
| 变量绑定 |  |  | 长距离关系和状态压缩 |
| 多证据合并 |  |  | 随机访问和冲突处理 |
| 长流吞吐 |  |  | 状态大小和硬件实现 |
| 每 token 成本 |  |  | 计算、带宽和调度 |

若 SSM 只在长流吞吐上占优，结论应是“它提供低成本的状态路径”，而不是“它全面替代 attention”。若混合架构在远程检索任务恢复质量，却引入复杂 cache 和 kernel fallback，则还要记录系统维护成本。

## 17. 一份架构论文阅读记录应该留下什么

读完论文后，至少保留一张证据卡：

| 字段 | 要回答的问题 |
| --- | --- |
| 原问题 | 原系统的瓶颈是什么，为什么重要 |
| 架构变化 | 哪个张量、路径、状态或路由发生改变 |
| 计算关系 | 时间、空间、通信和状态复杂度如何变化 |
| 归纳偏置 | 更偏向局部、全局、递归、稀疏还是精确访问 |
| 训练配方 | 数据、token、参数、优化和精度是否匹配 |
| 推理账本 | KV、状态、带宽、batch、P99 和单位成本 |
| 主结果 | 在哪些任务、规模和长度上出现效果 |
| 中间证据 | 哪些机制指标支持作者解释 |
| 消融 | 是否排除了数据、规模、调参和实现因素 |
| 失败边界 | 哪些任务、长度、硬件或负载没有受益 |
| 结论等级 | 论文事实、作者推断、自己复现或教学抽象 |

这张卡的价值不在于把论文压缩成十行，而在于避免把不同证据等级混成一句“这个架构更先进”。

## 18. 可运行的架构资源比较 demo

下面的代码用简化账本比较 MHA、GQA、MQA 和一种不维护显式 KV 的状态模型。它只计算教学用的 KV 显存和粗略状态成本，不替代真实 kernel benchmark；真实测量还要加入 batch、页表、编译、通信、硬件和质量结果。

~~~python
from dataclasses import dataclass


@dataclass
class Architecture:
    name: str
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int
    context: int
    bytes_per_value: int
    state_width: int = 0


def kv_cache_gib(model):
    values = (
        2
        * model.layers
        * model.context
        * model.kv_heads
        * model.head_dim
        * model.bytes_per_value
    )
    return values / (1024 ** 3)


def state_mib(model):
    if model.state_width == 0:
        return 0.0
    values = model.layers * model.state_width * model.bytes_per_value
    return values / (1024 ** 2)


models = [
    Architecture("MHA", 32, 32, 32, 128, 8192, 2),
    Architecture("GQA", 32, 32, 8, 128, 8192, 2),
    Architecture("MQA", 32, 32, 1, 128, 8192, 2),
    Architecture("SSM_proxy", 32, 0, 0, 0, 8192, 2, state_width=4096),
]

for model in models:
    print(
        f"{model.name}: kv_cache_gib={kv_cache_gib(model):.3f}, "
        f"state_mib={state_mib(model):.1f}"
    )
~~~

在这个账本中，MHA、GQA 和 MQA 的 KV 显存分别按 `n_kv` 成比例下降；SSM proxy 没有显式 token-by-token KV，但有固定宽度的递归状态。它没有说明谁的答案质量更好，也没有说明实际 kernel 谁更快。正确的下一步是把这些资源数字与相同 harness 下的质量、长距离任务和实测 trace 对齐。

## 19. 如何判断一篇架构论文的结论强度

可以把结论分成四层：

1. **接口事实**：论文或模型卡明确披露了结构、参数或最大长度。
2. **受控结果**：在相近预算和明确硬件下，某个指标出现差异。
3. **机制解释**：中间变量、消融和反事实支持作者提出的原因。
4. **部署结论**：在目标 workload、并发、SLO 和维护约束下，确实值得采用。

很多架构论文能支持第一和第二层，却不足以支持第三或第四层。论文中的“更高效”可能指单个 kernel，“更强”可能指某组 benchmark，“可扩展”可能指理论复杂度。读者必须把修饰词还原成可测量条件。

## 20. 技术讨论中的架构表达

在组会、架构评审或技术决策中，一个成熟的说明顺序是先给瓶颈，再给机制，最后给证据和边界：

> 这项改动针对的是 decode 阶段 KV 状态和带宽，而不是训练 attention 的二次复杂度。它通过减少 K/V head 让多个 query head 共享状态，所以 KV 显存随 `n_kv` 下降；但质量、通信和 kernel 仍需在相同长度、batch、硬件和评估协议下比较。当前证据可以支持状态占用降低，如果没有端到端 P99 和任务质量数据，还不能直接说服务全面更快。

这样的说明比“GQA 更高效”更有用，因为它同时说明作用范围、公式来源、未知项和验证方法。对于 MoE、Mamba、RoPE 或混合架构也可以使用同一结构：瓶颈是什么，计算图怎么变，预算怎么变，证据到哪一层，边界在哪里。若这段内容来自已经完成的实验，还应补上版本、样本、硬件和原始 trace；若只是研究计划，则要明确标注为待验证假设。

## 21. 资料与证据边界

Transformer 的原始结构、self-attention、multi-head、位置编码和 encoder-decoder 设计，参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。论文支持其序列到序列结构和机器翻译实验，不自动支持后续所有 decoder-only、长上下文或高效 kernel 结论。

GPT 与 BERT 的预训练路线分别参考 [Improving Language Understanding by Generative Pre-Training](https://arxiv.org/abs/1801.06146) 和 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)。它们支持不同预训练目标、可见性和下游适配的历史事实；本章关于现代产品接口的讨论属于后续工程归纳。

位置编码部分参考 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 与 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)。论文中的外推结果依赖训练长度、任务、模型和实现，不能把任何最大长度声明直接转成通用理解能力。

归一化和 FFN 部分参考 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) 与 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。公式和机制用于解释模块差异，实际收益仍需在参数、训练预算和硬件条件可比时验证。

LLaMA 与 Mistral 的架构和训练配方分别参考 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 与 [Mistral 7B](https://arxiv.org/abs/2310.06825)。这些论文支持公开版本中的模型、训练和评估描述；参数、上下文、吞吐和产品性能应绑定具体 revision、硬件和运行协议。

MoE 部分参考 [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668) 和 [Switch Transformers](https://arxiv.org/abs/2101.03961)。它们支持稀疏路由、容量和负载均衡的研究脉络；本章的资源公式是教学近似，不替代集群通信 benchmark。

SSM 与 Mamba 部分参考 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)。论文支持 selective state space 和相关实验主张；“适合哪些真实工作负载”仍需结合任务、硬件、kernel、训练预算和独立评估。

本章中的 GQA/MQA 显存数字、MoE/SSM 对照和 Python demo 是教学构造。它们帮助读者建立资源账本和证据层次，不能被引用为任何模型的实测 benchmark。真实比较应保存模型配置、数据、训练 token、checkpoint、runtime、硬件、batch、输入/输出长度、评估脚本和原始 trace。

## 22. 本章小结

架构论文的主线不是不断出现的新模块，而是不断重新分配信息、计算和状态的成本。阅读时应：

1. 先找原系统瓶颈，再确认论文改变了哪一段计算图。
2. 把架构、训练目标、数据、规模和工程实现分开。
3. 用公式追踪 attention、位置、归一化、FFN、KV、路由和状态的数量关系。
4. 同时记录表达能力、训练效率、推理效率和维护成本。
5. 对 GQA/MQA 区分 query 与 KV head，对 MoE 区分 total/active parameters 和通信，对 SSM 区分线性状态成本与精确随机访问能力。
6. 对长上下文论文检查中间证据、多证据冲突、真实文档和资源约束，而不只看最大 context length。
7. 对所有 benchmark 提升追问数据、token、compute、调参、硬件和评估协议是否公平。
8. 最后把结论标记为接口事实、受控结果、机制解释或部署结论，避免把其中一层扩大成另一层。

当你能说清一个架构改变了什么、为什么可能有效、为此付出了什么、哪些实验仍然缺失时，你才真正读懂了这条论文线，而不是记住了一串模型名字。
