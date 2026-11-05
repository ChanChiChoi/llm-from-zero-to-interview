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

下面的图把“先找瓶颈”变成一条可执行的筛查路径。它把常见改动放到最可能影响的资源项上；实际论文可能同时改变多个位置，所以最后仍要回到对照实验和完整预算。

~~~mermaid
flowchart TD
    A["定义工作负载、质量目标与预算"] --> B{"主要受限在哪里？"}
    B -->|训练时注意力中间量与数据搬运| C["IO-aware attention kernel"]
    B -->|长序列 token 对交互量| D["稀疏或局部 attention"]
    B -->|decode 阶段的 KV Cache| E["GQA、MLA 等状态压缩"]
    B -->|总容量与每 token 激活计算| F["MoE 路由"]
    B -->|长流历史如何保留| G["SSM、Mamba 或 Delta rule"]
    C --> H["在目标硬件与 harness 下核对质量和端到端成本"]
    D --> H
    E --> H
    F --> H
    G --> H
~~~

图中的分支是首轮诊断，不是互斥分类。例如 MoE 同时涉及参数驻留和跨卡通信；混合架构也可能把稀疏 attention、递归状态和专家路由组合起来。读者应沿着箭头先定位待验证的变量，再检查论文是否把其余因素控制住。

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

可用一个朴素实现估算中间分数矩阵的量级。设单层、batch 为 1、32 个 head、长度 $T=2048$，并假设每个 query-key 分数以 BF16（2 字节）完整物化，则分数元素数为 $32\times2048^2=134,217,728$，占用约 $256\ \mathrm{MiB}$。若长度增至 8192，长度变为 4 倍，分数矩阵变为 $32\times8192^2=2,147,483,648$ 个元素，占用约 $4\ \mathrm{GiB}$，正好是 16 倍。这个估算只针对一层的分数，不包括 Q/K/V、反向传播和临时张量；使用 FlashAttention 一类分块 kernel 时不会按这个数字保存完整矩阵，但 token-pair 交互的计算仍随 $T^2$ 增长。于是“显存省了”可能指不再物化分数矩阵，“复杂度降为线性”则是另一个需要单独证明的主张。

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

如果方法只在 synthetic needle 任务上有效，结论应限制为定位能力信号；如果真实文档中的多证据推理没有改善，不能把 synthetic 结果写成通用长上下文能力。长上下文任务如何按证据位置、任务难度和有效长度分层，可继续看[第七册第十三章：1M Context 有效能力评测](../../book-07-evaluation-experiments/chapters/13-1m-context有效能力评测.md)。

近年的稀疏注意力还说明，理论上跳过了多少分数，不足以预测硬件实际快多少。Native Sparse Attention（NSA，原生稀疏注意力）把粗粒度压缩、细粒度 token 选择和局部精确读取组合成层级稀疏路径，并把训练可用的算子与硬件对齐一并设计。该论文报告在自己的 64K 序列设置中，decode 延迟最高达到相对全 attention 的 11.6 倍加速；这个数字绑定其模型、GPU、kernel 和基线，不能外推成任意稀疏模式或任意长度的加速承诺，见 [NSA 原文](https://arxiv.org/abs/2502.11089v2)。读这一类论文时，应同时检查 token 选择机制、训练是否原生支持稀疏路径、选择开销、质量切片以及相同 runtime 上的端到端测量。

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

模型的研究影响还取决于论文、训练细节、代码、权重和许可分别公开到什么程度。它们改变复现成本和社区复用，但不属于架构机制本身，也不能自动当作模型质量证据。读者应对照具体模型版本，逐项记录“能读到什么、能下载什么、许可允许做什么、缺了什么”，避免用一个“开放”标签把几种不同的可复现条件混在一起。

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

### 9.4 MLA：把多头 K/V 联合压进潜变量

GQA 通过减少 K/V head 数压缩缓存；Multi-head Latent Attention（MLA，多头潜变量注意力）采用另一条路线：先把当前表示投影成低维 KV 潜变量，再由它恢复内容 key 和 value。设第 $t$ 个 token 的隐藏表示为 $h_t\in\mathbb{R}^{d}$，压缩维度为 $d_c$，则教学化地写成：

~~~math

c_t^{KV}=W^{D}_{KV}h_t,\qquad k_t^{C}=W^{U}_{K}c_t^{KV},\qquad v_t^{C}=W^{U}_{V}c_t^{KV}.

~~~

推理时，缓存的主体是每个 token 的 $c_t^{KV}$，而不是所有 head 的完整 K、V。位置编码带来一个额外约束：若直接把 RoPE 旋转放进待压缩的 key，旋转矩阵会阻止一部分投影在推理中合并。DeepSeek-V2 的做法是把内容 key 与携带位置的 RoPE key 分开；后一条共享位置路径仍须缓存。这里 $d_h^{R}$ 是论文中解耦位置 key 的每头维度，因此每层每 token 的缓存宽度按论文记为 $d_c+d_h^{R}$，而不是只有 $d_c$。

在 DeepSeek-V2 的配置中，$d_c=4d_h$、$d_h^{R}=d_h/2$，所以 MLA 的缓存宽度是 $4.5d_h$。对比每组 GQA 的 $2d_h$，它相当于 2.25 组 GQA 的缓存量。作者在附录 D.2 用两个规模的 MoE 变体比较 MLA 与 MHA：两组模型分别共享各自架构、只更换 attention，训练 token 数分别为 1.33T 和 420B；报告 MLA 变体的 KV 缓存分别约为 MHA 的 14% 和 4%，并在四个所选 benchmark 上获得更高分。该证据比只看整模型对比更接近 attention 消融，但激活参数等配置仍有差异，不代表所有模型中 MLA 都优于任意 GQA 配置。论文摘要中“KV Cache 降低 93.3%”则是 DeepSeek-V2 与 DeepSeek 67B 的系统级比较，不能直接归因为 MLA 单一因素，详见 [DeepSeek-V2 原文](https://arxiv.org/abs/2405.04434v5)。

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

Mamba-2 把“递归更新”和“并行训练”之间的关系讲得更具体。Structured State Space Duality（SSD，结构化状态空间对偶）把一类结构化状态空间变换写成半可分矩阵；同一计算既可沿 token 顺序做线性递归，也可改写成分块矩阵运算。这个视角让模型在推理时保留递归状态，同时让训练用更适合矩阵硬件的分块计算。论文题名“Transformers are SSMs”是对特定结构关系的概括，不表示标准 softmax attention 与任意 SSM 可以无损互换；文中证明和算法针对明确定义的结构化子类。论文报告核心层在其设定下比 Mamba-1 快 2–8 倍，这个数不能直接当作整模型训练或线上 serving 的统一加速比，详见 [Mamba-2 原文](https://arxiv.org/abs/2405.21060)。

### 11.3 SSM 的能力边界

SSM 可能擅长流式处理、长序列扫描和低状态成本，但固定或压缩状态对精确随机访问、远程复制、多个冲突证据和复杂 in-context learning 的表现需要单独评估。不能把“线性复杂度”直接等同于“更好的长上下文理解”。

还有一条相邻路线把递归状态看成可读写的快速权重矩阵。普通 Delta rule 的简化形式是：

~~~math

S_t=S_{t-1}+\beta_t k_t\left(v_t-S_{t-1}^{\mathsf T}k_t\right)^{\mathsf T},\qquad o_t=S_t^{\mathsf T}q_t.

~~~

其中 $k_t$ 是写入地址，$v_t$ 是新内容，$q_t$ 是读取地址，$\beta_t$ 控制更新幅度。取 $S_0=0$、$k_1=(1,0)^{\mathsf T}$、$v_1=(4,2)^{\mathsf T}$、$\beta_1=0.5$，旧预测 $S_0^{\mathsf T}k_1=(0,0)^{\mathsf T}$，待写残差仍为 $(4,2)^{\mathsf T}$，所以 $S_1=\begin{bmatrix}2&1\\0&0\end{bmatrix}$；若 $q_1=k_1$，读出 $o_1=(2,1)^{\mathsf T}$，即这一步把该地址的目标内容写入一半。这个小算例只解释 Delta 更新的读写直觉，不是完整 Gated DeltaNet 公式；Gated Delta Networks 再加入自适应遗忘门，并给出分块并行训练算法。作者在真实世界回忆任务中也报告纯递归模型与 Transformer 存在差距，而门控 DeltaNet 及混合模型在特定 20K 长度测试上改善，这说明机制能力必须按任务切片核验，详见 [Gated Delta Networks 原文](https://arxiv.org/abs/2412.06464)。

2026 年的 Mamba-3 预印本继续沿着状态更新机制改造：作者组合指数梯形离散化、复数状态转移和多输入多输出（MIMO）形式，以提升状态跟踪表达能力，并试图在不增加 decode 延迟的情况下提高质量。在论文报告的 1.5B 对照中，Mamba-3 的下游平均准确率比其 Gated DeltaNet 对照高 0.6 个百分点；MIMO 版本再提高 1.2 个百分点。论文还报告，在其 440M、FineWeb-Edu、2× Chinchilla token 预算的 state-size 对照中，Mamba-3 用约一半 Mamba-2 状态维度达到相近 perplexity。它们是论文特定模型、数据、基线和任务套件的结果；作者另以单张 H100、batch 128 比较 kernel latency，这并不是生产服务 P99。是否能迁移到目标服务，仍需单独测状态字节数、吞吐、尾延迟和检索失败切片，详见 [Mamba-3 原文](https://arxiv.org/abs/2603.15569)。

同一状态路线仍在快速迭代。2026 年的 Gated DeltaNet-2 预印本把通道级 erase gate 与 write gate 分开，并比较纯递归和带滑窗 attention 的 hybrid；作者报告的 1.3B、100B FineWeb-Edu token 实验只说明这套模型与协议中的差异，不形成递归家族的一般排名。它可以作为后续追踪入口，但本章不把最新单篇预印本写成稳定默认配方，见 [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)。

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

### 12.3 混合比例是实验变量，不是配方常数

Jamba 是一个可读的组合案例：它交错排列 Transformer attention 层与 Mamba 层，并在部分层加入 MoE。作者在 1.3B 参数、训练 250B token 的实验中比较 1:3 与 1:7 的 attention:Mamba 层比例，报告二者性能差异很小，并因计算效率选择 1:7 做较大规模实验；在 7B、训练 50B token 的对照里，混合模型也优于论文中的纯 attention 和纯 Mamba 对照。正确的读法不是“1:7 最优”，而是作者在给定规模、token 预算和实现中证明这个比例值得继续验证。不同 state size、窗口 attention、MoE 位置或 GPU kernel 都可能改变最优比例，具体结果见 [Jamba 原文](https://arxiv.org/abs/2403.19887v2)。

后续混合方案也不止把两类 block 交错。Kimi Linear 将 Kimi Delta Attention 的递归状态路径与全局 MLA 组合；DeepSeek-V4 技术报告则描述压缩历史 KV 后进行稀疏选择的 CSA，以及对压缩表示做 dense attention 的 HCA，并分别管理压缩 KV、索引状态、滑动窗口和未完成尾部。这些是特定报告披露的结构，不代表所有模型都采用相同实现；整模型质量或成本结果也不能单独归因于某条路径。设计表应明确列出每种路径负责什么信息、缓存何种状态、怎样跨路径通信，以及每增加一种路径后引入多少算子与调度负担。论文如果只报告整套模型胜过一个基线，不能据此断言每种模块都必要；还要查看单独移除 attention、递归状态或 MoE 的消融，并检查质量收益是否足以抵偿多状态管理和 kernel fallback 的维护成本。具体结构见 [Kimi Linear](https://arxiv.org/abs/2510.26692v2) 与 [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)。

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

## 14. Worked case：比较 MHA、GQA、MQA 和 MLA

### 14.1 先算状态账本

设模型有 $L=32$ 层、query head 数 $n_q=32$、每个 head 维度 $d_h=128$、上下文长度 $T=8192$，使用 BF16，每个缓存元素 2 字节；先按单条请求计算，不计 allocator、页表和临时工作区。比较三种普通 KV head 配置，再加入 DeepSeek-V2 的 MLA 维度：

| 结构 | $n_q$ | $n_{kv}$ 或 MLA 维度 | 每层每 token 的缓存元素 |
| --- | ---: | ---: | ---: |
| MHA | 32 | $n_{kv}=32$ | $2\times32\times128=8,192$ |
| GQA | 32 | $n_{kv}=8$ | $2\times8\times128=2,048$ |
| MQA | 32 | $n_{kv}=1$ | $2\times1\times128=256$ |
| MLA | 32 | $d_c=512,\ d_h^R=64$ | $d_c+d_h^R=576$ |

对 MHA、GQA 和 MQA，KV Cache 显存用：

~~~math

\mathrm{Memory}_{KV}=2LTn_{kv}d_hb.

~~~

每层每 token 先缓存 K 和 V，因此元素数为 $2n_{kv}d_h$；乘上层数后，再乘 token 数与每元素字节数。MHA 的总元素数是 $8,192\times32\times8,192=2,147,483,648$，占用 $4,294,967,296$ 字节，即 4 GiB。GQA 的对应计算为 $2,048\times32\times8,192\times2=1,073,741,824$ 字节，即 1 GiB；MQA 为 $256\times32\times8,192\times2=134,217,728$ 字节，即 0.125 GiB 或 128 MiB。仅减少 KV head 数，缓存就按 $n_{kv}$ 线性下降。

MLA 则按每层缓存 $d_c+d_h^R=512+64=576$ 个元素，32 层共有 $18,432$ 个元素/token；在 8,192 token 和 BF16 下，占用 $18,432\times8,192\times2=301,989,888$ 字节，即 288 MiB。相对本例 MHA，减少约 $1-288/4,096=92.97\%$；相对单个 GQA 组的 $2d_h=256$ 个元素，它相当于 2.25 组缓存。这个结果与论文中 DeepSeek-V2 配置给出的维度关系相符，但本例是教学账本，不是模型实测。论文报告的 93.3% 是 DeepSeek-V2 对 DeepSeek 67B 的系统级对比，不应和本例 92.97% 的纯公式结果混为一谈。

### 14.2 质量和成本不能只看显存

GQA 可能降低 KV 读取，但它也改变了 K/V 表示容量。要验证“质量保持”，需要比较短上下文、长上下文、代码、数学、多语言和工具参数等切片；要验证“服务更快”，需要固定硬件、batch、输入/输出长度、编译和 cache 策略，测 TTFT、TPOT、吞吐和 P99。

缓存缩小首先意味着单请求状态字节数下降；它可能让同一 GPU 放入更多并发请求，也可能因读写流量下降而改善 TPOT，但这两项都要在目标 runtime 中测量。还要同时做质量护栏：固定模型深度、query heads、训练数据、token 与微调配方，只改变 KV 组织，再按短/长、检索/生成等切片比较质量。若总模型、训练预算和推理 kernel 同时改变，论文只支持“这一组合方案在本设置下更好”；若只看到缓存下降而 P99 未变，结论应停在“状态占用下降”。

把单请求账本换成一个并发容量例子：假设 16 条请求都达到 8,192 token，且都按上面的配置保存完整历史，MHA 的 KV 合计为 $16\times4=64$ GiB，GQA 为 16 GiB，MQA 为 2 GiB，MLA 为 $16\times288=4,608$ MiB，即 4.5 GiB。这个乘法说明减少状态可能让服务接纳更多活跃序列，但还没有计算权重、临时激活、分页碎片、prefix sharing、请求长度不均和安全余量。真实调度器会按请求增长逐步分配缓存，也可能批内请求同时达到峰值，因此“16 条满长请求”的上界要与生产请求分布区分。还要记住，MLA 压缩 KV 不会自动消除 attention 的 token-pair 计算；它首先改变 decode 缓存与读带宽，prefill 仍需要单独测算。只有质量曲线、内存峰值与目标 runtime 的吞吐/P99 同时齐备，才可以把资源公式推进到部署判断。

缓存公式的输入也必须在论文之间对齐。这里默认所有层都保存每个历史 token、所有值同精度、没有窗口淘汰；若一方用量化 KV，$b$ 会变小，但量化误差和反量化开销要另测；若一方采用 sliding window，$T$ 应换成实际保留长度；若采用分页，元素数不变，但碎片和分配策略会影响峰值。GQA 减少 K/V head 组数，可能改变每组可表达的信息；MLA 用低秩 latent 做压缩，可能在减少缓存时引入瓶颈。因而比较表应同时保存 head 维度、cache precision、缓存生命周期和 RoPE 路径，避免把不同机制的“每 token 缓存”当成同一个指标。

若 batch 中请求长度不同，不能拿最大长度乘请求数冒充实际平均值。MHA/GQA 的总缓存应按 $2Ln_{kv}d_hb\sum_iT_i$ 求和，MLA 则按 $L(d_c+d_h^R)b\sum_iT_i$；只有请求长度全相同时才可简化为 $B$ 倍单请求结果。论文若只给“最大并发”而不披露长度分布和 KV dtype，读者无法复算平均成本。这个分母和缓存生命周期同样是架构比较的一部分。

最后，容量是字节数，不是延迟的同义词。更小的 cache 可能缓解 memory-bound decode，也可能被投影计算或调度开销抵消；须测同样 batch 和输出长度下的 TPOT、P99 与并发吞吐。

## 15. Worked case：MoE 与 dense 模型的公平比较

构造一个只用于算账的模型，不把它当成真实模型配置：dense 模型有 5B 个非 FFN 参数和一个 2B 参数 FFN，总计 7B。MoE 把这个 2B FFN 替换成 8 个同尺寸 expert，并令每个 token 只路由到 top-1 expert。此时总参数是 $5+8\times2=21$B，激活参数近似为 $5+1\times2=7$B。它的每 token 矩阵计算量可能接近 dense 对照，但权重存储约为 3 倍；实际 FLOPs 还受 router、padding、稀疏 kernel 和专家放置影响，不能只凭 active parameters 断言相等。

再把 routing 负载算出来。若一个 batch 有 1,024 个 token，均匀分给 8 个 expert，理想负载是每个 expert 128 个 token，按本章定义的 $\mathrm{Skew}=\max_e(c_e)/\mathrm{mean}_e(c_e)$ 得到 1。若最忙的 expert 收到 256 个、其余 expert 平均分配另外 768 个，则平均仍是 128，Skew 为 $256/128=2$。假设实现把 capacity factor 设为 1.25，则名义容量是 $\lceil1.25\times1,024/8\rceil=160$ 个 token；最忙 expert 超出 96 个位置。不同实现可能丢弃、padding 或重新路由这些 token，因此相同的参数规模和 top-k 仍可能得到不同质量与吞吐。

因此比较 MoE 与 dense 至少要分开报告 total parameters（权重存储和容量）、active parameters（每 token 的主要矩阵计算代理）、路由和通信（跨卡 all-to-all、负载偏斜和尾延迟）。还须对齐预训练数据、token 数、训练 FLOPs、质量切片和硬件。DeepSeekMoE 的细粒度专家切分与共享专家隔离说明 MoE 内部也有不同设计；DeepSeek-V2 同时改变注意力和 FFN，并使用新的数据与训练预算，模型总结果不能单独证明其中某个组件的因果收益，见 [DeepSeek-V2 原文](https://arxiv.org/abs/2405.04434v5)。小 batch 单机上通信可能让 dense 更快；集群和负载改变后成本排序也可能改变，差异来自约束变化，不是指标矛盾。

刚才的 7B active 只是把参数数目当作计算代理；同样数量的参数如果位于 embedding、attention 或 FFN，实际每 token FLOPs、访存和并行方式并不相同。MoE 的 expert 权重还必须放在某些设备上，token 路由可能跨节点；即使每个 token 只执行一个 expert，权重驻留、all-to-all 和热点设备仍会限制批量吞吐。因此一篇公平的比较应明确哪些参数共享、哪些专家激活，是否在 dense 侧通过加宽或加深来匹配预算，并报告每层实际 FLOPs 或至少可复算的层形状。

容量例子也说明为什么只给一个总准确率会漏掉机制。可把上面的 1,024 token batch 按 expert 画成负载直方图：如果专家 0 有 256 个 token，而平均容量是 128，那么它的队列可能决定整个同步 step 的尾部时间；capacity factor 1.25 时只有 160 个位置，超出的 96 个 token 会按具体实现被丢弃、padding 或转发。审阅者应记录这些 token 的比例和落在哪些任务切片，而不只看平均 expert load。若总质量变化同时伴随 token drop 变化，观察到的差异可能来自信息没有进 expert，而非 expert 参数不足。跨设备比较还需固定网络拓扑、expert parallel 组大小、微批次、精度和 kernel，否则所谓“MoE 比 dense 便宜”很可能只描述了某个集群的吞吐点。

再把参数数目换成一个静态存储下界。假设所有权重都以 BF16 常驻，dense 的 7B 参数约需 $7\times10^9\times2=14$ GB 十进制字节；MoE 的 21B 总参数约需 42 GB。这个差异不等于每张卡都要装下 42 GB：expert parallel 可以把权重分散到多卡，量化和 offload 也会改变驻留，但它说明 active parameter 数不能替代集群总权重账本。报告时至少区分单卡峰值、所有专家聚合占用和每 token 实际读取字节，并说明单位按 GB 还是 GiB。否则一项方案可能在 FLOPs 上近似 dense，却因权重通信或显存容量无法满足目标并发。

公平实验应把训练和部署分开做。训练消融固定 token 数、数据顺序、优化器、总计算预算和质量目标，至少给出不同随机种子的分布；部署基准则固定 checkpoint、请求长度分布、并发、拓扑和生成配置，记录吞吐、P95/P99、网络字节、专家负载分位数与 token drop。若无法完全匹配总参数和 active 参数，应把差额列成设计变量，并把结论限定为“在此预算下的组合比较”。这样读者才能判断收益来自额外容量、条件计算，还是路由实现，而不会将三个不同问题合成一句“MoE 更便宜”。

top-k 也是重要条件：若改为 top-2，每个 token 会生成两份 expert assignment，路由流量和容量公式随之变化；即使激活了相似数量的 FFN 参数，通信和负载曲线也不同。容量是在整个 batch、设备组还是单个 expert-parallel group 上计算，取决于实现，必须写清。对研究者来说，最小报告还应保留每层、每设备的负载分布和 overflow 处理策略；这样才能区分结构在任务上有用，还是运行时恰好把压力落到了少数热点专家。

## 16. Worked case：SSM 与 attention 的任务边界

为避免把能力压成单一分数，先构造一个 300 题的审阅案例，分成三个各 100 题的切片。下表中的正确数是教学假设，不是论文或模型测量：

| 任务切片 | attention | SSM | 要检查的机制 |
| --- | ---: | ---: | --- |
| 局部语法（100 题） | 94/100 | 93/100 | 局部混合与短程状态更新是否足够 |
| 远距离原文复制（100 题） | 88/100 | 63/100 | 固定状态能否保留精确细节 |
| 两条冲突事实的多证据问答（100 题） | 82/100 | 60/100 | 是否能按查询随机访问并处理冲突 |

三切片等权时，attention 为 $(94+88+82)/300=264/300=88\%$，SSM 为 $(93+63+60)/300=216/300=72\%$。整体差为 16 个百分点，但它掩盖了差异集中于远程复制和冲突证据；因此报告要同时保留分层分母，并用配对转移表记录共同答对、仅一方答对和共同答错的题目。

若目标业务的题型比例不同，overall 会随分布改变。再设一个合成业务流：80% 是局部语法类请求，10% 是远程复制，10% 是冲突事实问答。用同一组假设切片正确率加权，attention 为 $0.8\times94+0.1\times88+0.1\times82=92.2\%$，SSM 为 $0.8\times93+0.1\times63+0.1\times60=86.7\%$，差距变为 5.5 个百分点，而不是等权测试的 16 点。改变题型权重并没有改变单个切片结果，却改变了决策相关的总体估计；因此要报告目标 workload 的权重来源，以及把分布改变到什么程度会翻转方案选择。所有比例和正确率仍是教学构造值。

为了让切片能复现，远程复制题应规定唯一键、目标原文长度、干扰项位置和准确匹配规则；冲突事实题应给两条内容相近但时间或来源不同的陈述，并预先规定查询到底要求最新、最早还是指定来源。每题都固定 tokenization、输入长度、生成预算和后处理。若只有最终对错，无法区分模型没有读取证据、状态把旧值覆盖，还是生成器读到了却选错；保留错误类别、证据位置和干扰跨度，能把架构机制接回可观测结果。对比时先用完全相同的题与提示，随后再做真实文档测试，才能知道合成检索差异能否迁移。

例如，可以在位置 100 写入“项目代号=琥珀”，再插入 4,000 个无关 token，最后询问项目代号；下一题在位置 100 和 3,000 分别写入两个带时间戳的负责人值，再询问最新值。把目标位置、干扰跨度和事实先后次序系统变化，attention 与 SSM 的输入完全相同，只改变待读取距离或状态覆盖条件。统计每种条件下的正确数、证据引用位置和失败类型，才能看出差异是否随距离增加、是否集中在冲突更新，以及失败后是否恢复。若要比较生成的资源成本，还要固定答案长度或单独报告生成 token，否则输出长短会混入 TPOT 与吞吐。最后用真实文档复核合成任务中观察到的趋势；合成结果解释机制，真实集检查迁移，两者都不能替代另一方。

固定状态也不是“记住的信息量固定为某个字节数”这么简单：状态维度、每层头数、精度和并发数都会改变总占用，状态压缩还可能把多个历史事实映射到相同表示。长流测试应在多个距离点重复同一任务，并显式重置每条请求的状态，避免前一条样本泄漏到下一条。报告中若只给一个 256K 或 1M 长度的成功率，就看不出模型从什么长度开始遗忘，也无法区分位置编码外推、递归状态容量和数据分布变化。建议同时画质量—长度曲线和状态字节—长度曲线，再把拐点与失效类型对应起来。

再单独算资源尺度。设一个长度 $T=8,192$ 的单请求，比较前例的 32 层、32 个 head、每 head 128 维、BF16 MHA，其 KV 缓存是 4 GiB。构造一个仅用于缩放说明的 SSM proxy：每层只保存 4,096 个 16-bit 状态值，则总状态为 $32\times4,096\times2=262,144$ 字节，即 256 KiB，且不随 $T$ 增长。这个 proxy 没有声称等同于 Mamba 的实际状态形状；真实模型还要计入 inner dimension、状态维度、门控和临时 buffer。它只说明“历史作为逐 token KV 保存”与“压进固定状态”具有不同长度依赖，不能用 4 GiB/256 KiB 直接宣布 SSM 的实际质量或端到端成本胜出。

所以，若一个研究问题把质量和长流成本放在同一目标里，应同时报告：任务切片及分母、远程信息可检索性、状态形状与字节数、预填充/解码时延、吞吐和目标硬件。若 SSM 只在长流资源指标占优，结论应是“它提供低成本状态路径”，而不是“全面替代 attention”；若混合架构在远程检索恢复质量，却引入额外缓存、kernel fallback 和调度复杂度，也要把这些成本记入同一决策。

## 17. 一份架构论文阅读记录应该留下什么

证据卡的目的不是把论文摘要抄成表格，而是把作者的机制主张、测量结果和读者自己的推断拆开。填卡时先写可以被证伪的问题，例如“在质量容差内降低 decode 的每请求 KV 字节”；再记录结构到底改了哪条张量或状态路径，最后把论文实验中的同名指标、对照条件和未测范围填进去。

| 字段 | 要回答的问题 |
| --- | --- |
| 原问题 | 原系统的瓶颈是什么，为什么重要？ |
| 架构变化 | 哪个张量、路径、状态或路由发生改变？ |
| 计算关系 | 时间、空间、通信和状态复杂度如何变化？ |
| 归纳偏置 | 更偏向局部、全局、递归、稀疏还是精确访问？ |
| 训练配方 | 数据、token、参数、优化和精度是否匹配？ |
| 推理账本 | KV、状态、带宽、batch、P99 和单位成本是什么？ |
| 主结果 | 哪些任务、模型规模和长度上出现效果？ |
| 中间证据 | 哪些机制指标支持作者的解释？ |
| 消融 | 是否排除了数据、规模、调参和实现因素？ |
| 失败边界 | 哪些任务、长度、硬件或负载没有受益？ |
| 结论等级 | 这是论文事实、作者推断、自己复现还是教学抽象？ |

以 §14 的 MLA 算例为例，卡片中的“架构变化”应写低秩 K/V 联合压缩和独立的 RoPE key；“计算关系”写每层缓存宽度 $d_c+d_h^R$；“主结果”只能填论文报告的特定模型和消融；“部署边界”还需注明本地目标 runtime 的 P99 与质量尚未测。这样一张卡能让另一个研究者知道哪些结论可追溯，哪些仍是假设，而不把证据压扁为“MLA 更高效”。

## 18. 可运行的架构资源比较 demo

下面的代码把 §14 的同一组输入放入简化账本，比较 MHA、GQA、MQA、MLA 与一个不维护逐 token KV 的固定状态 proxy。输入单位是“每层每 token 的缓存元素数”、层数、上下文 token 数和每元素字节数；输出是单请求 GiB 与假设状态的 MiB。它只做公式核对，不测 kernel 吞吐、质量或真实模型的状态形状。

状态：静态审阅，未运行（2026-11-04；Python 3.7+，仅标准库）

~~~python
from dataclasses import dataclass


@dataclass
class Architecture:
    name: str
    layers: int
    context: int
    kv_width_per_token: int
    bytes_per_value: int
    state_width: int = 0


def kv_cache_gib(model):
    values = (
        model.layers
        * model.context
        * model.kv_width_per_token
        * model.bytes_per_value
    )
    return values / (1024 ** 3)


def state_mib(model):
    if model.state_width == 0:
        return 0.0
    values = model.layers * model.state_width * model.bytes_per_value
    return values / (1024 ** 2)


models = [
    Architecture("MHA", 32, 8192, 8192, 2),
    Architecture("GQA", 32, 8192, 2048, 2),
    Architecture("MQA", 32, 8192, 256, 2),
    Architecture("MLA", 32, 8192, 576, 2),
    Architecture("SSM_proxy", 32, 8192, 0, 2, state_width=4096),
]

for model in models:
    print(
        f"{model.name}: kv_cache_gib={kv_cache_gib(model):.3f}, "
        f"state_mib={state_mib(model):.3f}"
    )
~~~

从 §14 的公式可手算预期：MHA 4.000 GiB、GQA 1.000 GiB、MQA 0.125 GiB、MLA 0.281 GiB；proxy 的逐 token KV 是 0，假设固定状态为 0.250 MiB。把上下文从 8,192 加到 16,384 时，前四种缓存都翻倍，proxy 的状态字节不变。由于 proxy 只假设每层 4,096 个 16-bit 状态值，这个对比不能代表真实 Mamba，也不表示答案质量相等。用于论文判断时，还应把这些数与同一 harness 的质量切片、TPOT、并发吞吐、内存峰值和原始 trace 对齐；任何实际执行结果都应另行记录版本、环境和验证状态。

## 19. 如何判断一篇架构论文的结论强度

可以把结论分成四层：

1. **接口事实**：论文或模型卡明确披露了结构、参数或最大长度。
2. **受控结果**：在相近预算和明确硬件下，某个指标出现差异。
3. **机制解释**：中间变量、消融和反事实支持作者提出的原因。
4. **部署结论**：在目标 workload、并发、SLO 和维护约束下，确实值得采用。

可以沿一个主张逐层检查。第一层只记录接口事实：MLA 的缓存由潜变量和 RoPE key 组成。第二层用公式或对照实验核实缓存元素数；§14 的 288 MiB 是本章教学输入的公式结果，而 DeepSeek-V2 论文另有自己的配置和对照。第三层要看消融能否把 K/V 压缩与数据、训练量、MoE、模型规模和 kernel 区分开。第四层则需在目标服务硬件上固定请求长度、batch、质量容差和 runtime，测 P99、吞吐、内存峰值与失败切片。

若证据到第二层，稳妥结论是“该结构在给定公式或论文设置中降低缓存”；若有受控消融但没有目标服务结果，可以说“机制证据支持作者的解释”，仍不应建议直接替换线上架构。论文中的“更高效”可能指单个 kernel，“更强”可能指某组 benchmark，“可扩展”可能指理论复杂度。每个修饰词都要还原成测量对象、基线、硬件、预算和允许误差，缺一项就把结论停在相应证据层。

## 20. 技术讨论中的架构表达

在组会、架构评审或技术决策中，可以用四句话把分析交付给团队：先定义瓶颈和工作负载，再说明机制怎样改变状态或计算图，然后给可复算的资源数字，最后指出证据层级和下一项能区分替代解释的实验。数字要带单位和分母；理论值、论文测量和自己的测量要分别标明。

> 这项改动针对的是 decode 阶段的 KV 状态占用。按本章假设，32 层、32 个 query head、128 维 head、8K 上下文和 BF16 的 MHA 需要 4 GiB；GQA 的 8 个 KV head 约 1 GiB，MLA 的 576 个缓存元素/层/token 约 288 MiB。这些是公式账本，不是端到端速度测量。下一步应在同一 checkpoint 或可比训练预算下固定 runtime、batch 和长度，检查任务质量、TPOT、并发、P99 和显存峰值；在这些数据出来前，只能主张缓存规模下降，不能据此下部署结论。

这样的表达能让团队复核所用的条件，也能自然接出实验任务：同一数据和质量门槛下比较模型，再记录单请求与并发成本。对于 MoE、Mamba、RoPE 或混合架构也使用同一顺序：瓶颈是什么、状态/计算图怎么变、预算怎么变、证据到哪一层、失败边界在哪里。若这段结论来自已完成的实验，说明 revision、样本、硬件、输入范围和 trace；若只是计划，就把它写成待验证假设，并明确什么观察结果会支持或推翻它。

## 21. 资料与证据边界

Transformer 的原始结构、self-attention、multi-head、位置编码和 encoder-decoder 设计，参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。论文支持其序列到序列结构和机器翻译实验，不自动支持后续所有 decoder-only、长上下文或高效 kernel 结论。

长序列稀疏注意力的当代系统案例参考 [Native Sparse Attention](https://arxiv.org/abs/2502.11089v2)。论文同时讨论层级 token 选择、原生训练和硬件对齐算子；11.6 倍是其 64K 解码实验中的最高报告，不是跨硬件、任务或模型的通用 speedup。

GPT 与 BERT 的预训练路线分别参考 [Improving Language Understanding by Generative Pre-Training](https://arxiv.org/abs/1801.06146) 和 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)。它们支持不同预训练目标、可见性和下游适配的历史事实；本章关于现代产品接口的讨论属于后续工程归纳。

位置编码部分参考 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 与 [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)。论文中的外推结果依赖训练长度、任务、模型和实现，不能把任何最大长度声明直接转成通用理解能力。

归一化和 FFN 部分参考 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) 与 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。公式和机制用于解释模块差异，实际收益仍需在参数、训练预算和硬件条件可比时验证。

LLaMA 与 Mistral 的架构和训练配方分别参考 [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) 与 [Mistral 7B](https://arxiv.org/abs/2310.06825)。这些论文支持公开版本中的模型、训练和评估描述；参数、上下文、吞吐和产品性能应绑定具体 revision、硬件和运行协议。

MoE 部分参考 [Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668) 和 [Switch Transformers](https://arxiv.org/abs/2101.03961)。它们支持稀疏路由、容量和负载均衡的研究脉络；本章的资源公式是教学近似，不替代集群通信 benchmark。

SSM 与 Mamba 部分参考 [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)。论文支持 selective state space 和相关实验主张；“适合哪些真实工作负载”仍需结合任务、硬件、kernel、训练预算和独立评估。

状态空间与递归架构的演进另参考 [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060)、[Gated Delta Networks](https://arxiv.org/abs/2412.06464) 和 [Mamba-3](https://arxiv.org/abs/2603.15569)。SSD 连接特定 SSM 与结构化 masked attention 的计算形式；Gated DeltaNet 研究门控遗忘与 delta 写入；Mamba-3 报告离散化、复状态和 MIMO 的作者实验。各自的速度和质量结果绑定论文模型、数据、对照及任务。

混合路线参考 [Jamba](https://arxiv.org/abs/2403.19887v2)：论文交错 Mamba/Transformer 层并在部分层使用 MoE；作者对不同层比例的结果仅支持论文中的模型规模和训练预算，不构成通用混合比例。

本章中的 GQA/MLA 显存账本、MoE/SSM 对照和 Python demo 是教学构造。它们帮助读者建立资源账本和证据层次，不能被引用为任何模型的实测 benchmark。真实比较应保存模型配置、数据、训练 token、checkpoint、runtime、硬件、batch、输入/输出长度、评估脚本和原始 trace。

## 22. 本章小结

架构论文的主线不是不断出现的新模块，而是不断重新分配信息、计算和状态的成本。阅读时应：

1. 先找原系统瓶颈，再确认论文改变了哪一段计算图。
2. 把架构、训练目标、数据、规模和工程实现分开。
3. 用公式追踪 attention、位置、归一化、FFN、KV、路由和状态的数量关系。
4. 同时记录表达能力、训练效率、推理效率和维护成本。
5. 对 GQA/MQA/MLA 区分共享 head 与潜变量缓存，对 MoE 区分 total/active parameters 和通信。
6. 对 SSM、Delta rule 与混合架构，区分状态更新、训练并行算法、精确检索能力和系统状态管理。
7. 对长上下文论文检查中间证据、多证据冲突、真实文档和资源约束，而不只看最大 context length。
8. 对所有 benchmark 提升追问数据、token、compute、调参、硬件和评估协议是否公平。
9. 最后把结论标记为接口事实、受控结果、机制解释或部署结论，避免把其中一层扩大成另一层。

当你能说清一个架构改变了什么、为什么可能有效、为此付出了什么、哪些实验仍然缺失时，你才真正读懂了这条论文线，而不是记住了一串模型名字。
