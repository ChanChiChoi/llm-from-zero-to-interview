# 第一章：Transformer 为什么能成为大模型基础架构

## 0. 本讲范围与资料

本章以公开论文、技术报告和前序章节为资料边界：Transformer 原论文、GPT-3、Scaling Laws、Chinchilla、LLaMA、FlashAttention、Mamba / Mamba-2，以及本项目第一册、第二册、第六册、第十三册、第十四册和第二十册中关于 self-attention、decoder-only、next-token prediction、KV Cache、分布式训练、推理部署和 Agent context window 的内容。

本章只解释“为什么 Transformer 在当前大模型生态里成为基础架构”，不把某个闭源模型的内部结构、训练数据、隐藏 scaling recipe、routing 策略或系统优化细节写成确定事实；示例推演也不代表真实模型 benchmark。后续章节会专门讨论 MQA/GQA/MLA、FFN、RMSNorm、RoPE、attention 复杂度、KV Cache、SSM/Mamba、RWKV、RetNet、Hyena 和混合架构。

本章重点有三点：

1. 解释 self-attention、next-token prediction、attention 成本和 KV Cache 的关键口径。
2. 比较 Transformer、经典 RNN/LSTM、CNN 和现代状态空间路线的信息路由、训练并行性、状态/缓存取舍。
3. 根据原始论文、实验范围和生态路径解释 Transformer 成为基础架构的原因，不把主观打分写成通用排名。

## 1.1 本章定位

很多人学 Transformer 时，会先记住几个模块：Self-Attention、Multi-Head Attention、FFN、LayerNorm、Residual、Position Encoding。这样能应付基础题，但还不够回答更高阶的问题：

```text
为什么最后是 Transformer 成为大模型的基础架构，而不是 RNN、LSTM、CNN、纯检索系统或者其他序列模型？
```

这个问题是架构理解的入口。因为 Transformer 的成功不是某个单点技巧，而是多个因素叠加：并行训练、长距离信息路由、统一自回归目标、硬件适配、规模实验、上下文学习和生态积累。

学完本章，你应该能回答：

1. Transformer 诞生前，RNN/CNN 序列模型主要瓶颈是什么。
2. Self-Attention 为什么改变了信息路由方式。
3. Transformer 为什么适合大规模并行训练。
4. Decoder-only Transformer 为什么成为通用 LLM 主流。
5. Scaling law、GPT-3、Chinchilla、LLaMA 等工作如何共同巩固 Transformer 路线。
6. Transformer 成为基础架构的同时，留下了哪些瓶颈，以及当前证据支持到什么范围。

## 1.2 资料来源和可信边界

本章主要参考以下公开论文和技术报告：

1. [Vaswani et al., 2017, *Attention Is All You Need*](https://arxiv.org/html/1706.03762v7)。原始实验比较的是机器翻译模型；作者报告 Transformer 在所测任务上质量更好、可并行化程度更高、训练时间更短。
2. [Kaplan et al., 2020, *Scaling Laws for Neural Language Models*](https://arxiv.org/abs/2001.08361)。论文研究指定语言模型实验范围内的 held-out cross-entropy 与模型规模、数据量和训练计算量之间的经验幂律关系。
3. [Brown et al., 2020, *Language Models are Few-Shot Learners*](https://arxiv.org/html/2005.14165)。GPT-3 论文展示了不更新参数、仅用文本任务描述和示例完成多种任务的行为，同时报告了任务差异和失败项。
4. [Hoffmann et al., 2022, *Training Compute-Optimal Large Language Models*](https://arxiv.org/html/2203.15556)。Chinchilla 研究在其模型、token 与训练预算范围内估计给定计算预算下的参数量和训练 token 配比。
5. [Touvron et al., 2023, *LLaMA: Open and Efficient Foundation Language Models*](https://arxiv.org/html/2302.13971)。论文报告使用公开可得数据集训练，并向研究社区发布模型；这支持该论文的具体数据与模型披露，不表示训练语料文件全部公开。
6. [Gu and Dao, 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*](https://arxiv.org/html/2312.00752) 与 [Dao and Gu, 2024, *Transformers are SSMs*](https://arxiv.org/html/2405.21060)。两篇论文分别讨论选择性状态更新及其扫描算法、以及特定结构化状态空间模型的递归与矩阵计算形式。

本章重点解释“为什么 Transformer 成为主流”，不是说 Transformer 没有缺点，也不是说它一定永远不可替代。后续章节会专门讲 $O(T^2d)$ 的 full-attention 计算、SSM/Mamba、RWKV、RetNet、Hyena 和混合架构。

## 1.3 Transformer 之前的问题：序列建模不只是“读文本”

语言、代码、语音、时间序列、DNA 都可以看作序列。

序列建模要解决的问题是：

```text
给定一串 token，模型如何让每个位置理解其他位置的信息，并预测、生成或分类？
```

Transformer 之前，主流路线包括：

1. RNN。
2. LSTM/GRU。
3. Seq2Seq + Attention。
4. CNN/Temporal Convolution。

经典 RNN/LSTM 的核心思想是递推：

```math
h_t=f(h_{t-1},x_t)
```

也就是说，第 $t$ 个位置的状态来自上一个状态和当前输入。

这个设计很自然，但在标准的按时间展开实现中有三个取舍。

第一，序列维度上的训练并行性受限。

因为 $h_t$ 依赖 $h_{t-1}$，这一类实现要沿时间步推进，难以像 Transformer 那样一次并行计算整段序列的各位置。这里描述的是经典 RNN/LSTM 的递推形式，不应泛化成所有循环或状态空间结构都无法并行：某些结构化递推可以用并行扫描算法计算，Mamba 等模型也专门设计了硬件感知的 scan。

第二，跨很远位置的信息要经过多步状态传递。

如果第 1000 个 token 要使用第 10 个 token 的信息，经典递推要经过许多状态更新。LSTM/GRU 的门控可以帮助训练长依赖，但不会把这条信息路径变成一步。

第三，固定维度状态需要压缩历史。

这类模型不会像标准 Transformer 那样为上下文中的每个 token 保留一份独立的 K/V 表示，而是将历史汇总进状态。状态维度有限，因此当任务要求按需找回许多离散细节时，会面临状态容量与计算/内存之间的取舍；这不是说递推状态不能保留长期信息，具体能力取决于状态结构、规模、训练和任务。

CNN 序列模型解决了一部分并行问题。卷积可以并行处理局部窗口，但它的感受野需要堆很多层才能覆盖长距离。扩大 kernel 或 dilation 可以缓解，但仍然不像 attention 那样让任意两个 token 直接交互。

因此，Transformer 出现前的一项重要取舍可以概括为：

经典 RNN/LSTM 的状态逐步传递，而局部 CNN 需要经多层扩大感受野；Transformer 选择以更重的位置交互计算换取全局内容路由和序列维并行性。

Transformer 选择了另一种解法：在一个注意力层中，让每个位置根据当前内容对可见位置加权读取信息。它减少了跨位置通信的层数，但增加了位置两两交互的计算与存储代价。

## 1.4 Self-Attention 的信息路由本质

Self-Attention 的公式通常写成：

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

这不是单纯数学形式，它代表一种信息路由机制。

对每个 token 来说：

1. Query 表示“我想找什么信息”。
2. Key 表示“我能被什么查询匹配”。
3. Value 表示“如果别人关注我，我提供什么内容”。

注意力权重就是：

```text
当前位置应该从其他位置读取多少信息。
```

Self-Attention 的一个关键变化是跨位置的计算图路径长度。

在 RNN 中，两个相距很远的 token 通信需要经过很多步：

$x_i\to h_{i+1}\to h_{i+2}\to\cdots\to h_j$

在一个 full self-attention 层中，若掩码允许，两个位置可以有一条直接的信息路径：

$x_i\leftrightarrow x_j$

“一跳”说的是计算图中路径更短，不等于信息无损传递、模型一定找对相关内容，或注意力权重就是可靠解释。它提供了一种让内容决定读取位置的机制。对语言任务，这种机制有用，因为依赖关系不总是局部的：

1. 主语和谓语可能相隔很远。
2. 代词可能指向前文实体。
3. 代码中的变量定义和使用可能隔很多行。
4. 数学推理需要反复引用条件。
5. 长上下文问答需要从远处检索证据。

Transformer 能成为大模型基础架构，一个重要原因是 attention 把“当前位置读取哪些可见位置”变成了可学习的内容路由问题；训练数据和参数仍决定这种路由能否服务具体任务。

## 1.5 并行训练：大模型时代的硬件友好性

Transformer 原论文标题叫 *Attention Is All You Need*，但摘要里同样重要的一句话是：在论文所测的机器翻译任务上，Transformer 更可并行化、训练时间更少，翻译质量也更好；这不是对所有任务和架构的普遍排名。[原始论文](https://arxiv.org/html/1706.03762v7)

大模型不是只靠好想法，还要靠能训练。

训练大模型需要：

1. 大 batch。
2. 大矩阵乘法。
3. 高 GPU/TPU 利用率。
4. 可分布式扩展。
5. 可稳定堆深和堆宽。

在 next-token prediction 训练中，目标序列已经给定。对长度为 $T$ 的输入，模型可以并行计算各位置的 Q、K、V、attention 和 FFN；因果掩码限制每个位置能读取哪些前文。网络层之间仍需依次计算。相比之下，经典 RNN/LSTM 的状态更新在序列维度上逐步依赖前一步。

从硬件角度看，Transformer 的主干计算是：

```text
Linear projection
Matrix multiplication
Softmax
MLP / FFN
Residual / normalization
```

这些操作很适合 GPU/TPU 加速，也方便做 kernel 优化、并行切分和混合精度训练。

所以 Transformer 的成功不是“算法与硬件无关”。矩阵乘法适配加速器，是重要的工程条件。后续的 FlashAttention、分页 KV cache、张量并行、流水线并行和分片优化进一步降低了特定训练/推理瓶颈；这些是围绕 Transformer 的系统演进，不代表原始 attention 没有长序列成本。

一句话总结：

```text
Transformer 不只是效果好，它还足够适合被现代加速器大规模训练。
```

## 1.6 统一目标：Next-Token Prediction 的威力

原始 Transformer 是 encoder-decoder，用于机器翻译。

但今天通用大模型主流是 decoder-only Transformer。

为什么？

因为 decoder-only 可以和自回归语言建模自然结合：

```math
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{1:t-1})
```

也就是每一步预测下一个 token。

这个目标有几个巨大优势。

第一，文本来源广泛。

网页、书籍、论文、代码、对话和文档都可以整理成 next-token prediction 样本；具体数据是否可用、质量如何、如何配比，仍是独立的训练数据问题。

第二，许多任务可以改写成续写任务。

翻译、摘要、问答和代码生成可以用“给定上下文，继续生成答案”来表达；分类等任务也可用标签文本或评分形式适配，但结果质量仍依赖数据、提示和评测。Next-token prediction 并非 Transformer 独有，其他序列架构也可以使用同一目标。

第三，训练目标形式统一。

模型可以在各位置预测后续 token，并用交叉熵学习。统一目标减少了为每类文本任务分别设计训练目标的需要，但不意味着训练过程天然稳定，也不排除任务专用的输出头或后训练。

第四，生成接口直接。

推理时模型逐 token 续写，直接支持开放式生成和对话文本。工具调用还需要外部协议和运行时配合；自回归生成也带来逐 token 解码的延迟。

因此，decoder-only Transformer 和 next-token prediction 形成了一个便于扩展的组合：

```text
统一架构 + 统一目标 + 海量数据 + 可扩展训练
```

GPT 系列为这条路线提供了语言模型实例。GPT-3 论文进一步展示，在其 few-shot 评估中，模型可以不更新参数，仅通过文本任务说明和示例完成多种任务；作者也报告了任务差异和表现较弱的项目。[GPT-3 原始论文](https://arxiv.org/html/2005.14165)

## 1.7 Scaling law 让扩展投入更可估算

Transformer 路线更容易获得持续投入的另一个原因，是研究者能在模型规模、数据和计算量上做定量实验。

Kaplan 等人研究了 held-out cross-entropy 随模型参数量、数据量和训练计算量变化的经验幂律。这个结论描述论文所测语言模型、训练流程和规模范围内的 loss，不直接保证所有下游能力随规模单调提高。[Scaling Laws 原始论文](https://arxiv.org/abs/2001.08361) Chinchilla 随后在其计算预算和模型族中重新估计参数量与训练 token 的配比：

```text
给定训练计算预算，参数量与训练 token 数要一起规划；只增加参数未必能得到更低 loss。
```

这对产业非常重要。

如果一个架构小模型很好，但放大后不稳定，企业不会轻易投入巨额训练成本。

这些结果让 Transformer 路线的训练投入更可估算：

1. 可以测量 held-out loss 随模型、数据和计算变化的趋势。
2. 可以比较固定计算预算下不同的参数/token 配比。
3. 可以据这些特定实验提出下一轮训练预算假设，再用目标模型和数据验证。

Chinchilla 论文用 70B 参数和更多训练 token 的模型，在与 Gopher 相同的计算预算下做了实验比较，并在多项评估中报告较好结果。[Chinchilla 原始论文](https://arxiv.org/html/2203.15556) 这支持该预算设定下的重新配比，不意味着任意模型、数据质量或下游任务都有同一最优比例。

这些工作没有证明 Transformer 是唯一可扩展的架构；它们为当时已经广泛研究的 Transformer 路线提供了更多可检验的训练证据。研究者随后继续改进数据配比、tokenizer、长上下文、MoE、注意力变体和推理系统。

## 1.8 In-Context Learning：上下文示例提供任务信号

GPT-3 的重要性在于 few-shot learning。

用户在 prompt 中写几个例子：

```text
English: cat
French: chat

English: dog
French: chien

English: apple
French:
```

在 GPT-3 论文的 few-shot 设定中，模型不更新参数，而是根据输入中的任务说明和示例继续输出。[GPT-3 原始论文](https://arxiv.org/html/2005.14165) 报告多种任务上的进步，也记录了少数任务表现较弱或 few-shot 收益不明显的情况；因此，这个结果说明模型能在上下文中利用示例，不代表它对所有任务都能可靠适配。

这类行为称为 in-context learning（上下文学习）。它是预训练参数、训练数据、模型结构和提示共同产生的行为。仅凭观察到 few-shot 行为，不能证明它由 attention 单独造成，也不必把上下文理解成模型内部真的执行了一段“临时程序”。

Attention 与上下文学习有什么关系？

在 decoder-only Transformer 中，生成某个位置时，因果 attention 允许该位置从提示前缀的可见 token 表示中读取信息。这提供了一条让示例、格式或约束影响后续 token 的信息路径，但它本身不能保证模型选对信息或遵循示例。

可以把 prompt 看成一次请求中显式提供给模型的上下文输入：

```text
Prompt examples / instructions / retrieved documents / conversation history
-> The model processes the visible context
-> Model generates answer
```

检索增强生成（RAG）可以把检索到的文本放进 prompt；Agent 系统也可以把工具返回值加入后续上下文。这些是围绕模型输入/输出建立的系统设计，不表示 Transformer 内部自带检索库或长期记忆。上下文长度、信息位置、干扰内容和提示格式都会影响可用结果，后续章节再讨论这些边界。

GPT-3 论文还明确记录 few-shot 任务之间表现不一，并讨论了 benchmark 污染等评估限制。上下文学习是 Transformer 路线重要的能力展示之一，但它不是“注意力会精确检索”的证据。

## 1.9 生态优势：基础设施围绕 Transformer 成熟

架构能不能成为基础架构，不只取决于论文指标，还取决于生态。

长期围绕 Transformer 的研究和部署形成了可复用的基础设施积累。这个路径依赖是结合论文演进和工具迁移成本作出的解释，不是某一项论文实验单独测出的因果效应：

1. 公开模型的结构定义、配置和权重加载路径。
2. 分布式训练、混合精度、检查点和并行切分的实现经验。
3. Attention/FFN 的融合算子、推理时 KV cache 管理和批处理方案。
4. 评估脚本、量化工具、模型格式及部署流程。
5. 围绕 RoPE、RMSNorm、SwiGLU、GQA、MoE、LoRA 等方法积累的可复用实现和经验。

LLaMA 是这条研究路线的一个例子。论文报告其 7B–65B 模型使用公开可得数据集训练，并向研究社区发布模型；作者还在指定基准上比较其模型与 GPT-3、Chinchilla 和 PaLM。[LLaMA 原始论文](https://arxiv.org/html/2302.13971) 这支持“公开可得数据可用于该论文训练”及其模型结果，不等于训练语料文件全部开放，也不能单独证明 Transformer 架构造成了这些结果。

这些积累会形成路径依赖。新的架构即使在某项论文实验中表现更好，要成为可替换的基础组件，还需逐项回答：

1. 在相同训练数据、预算、模型规模和评估条件下，质量与计算成本怎样？
2. 训练算法、并行扫描或 kernel 是否能在目标硬件上稳定运行？
3. 长上下文、逐 token 解码和需要精确访问前文的任务表现如何？
4. 现有权重、工具链、评估与部署流程需要改多少？
5. 在目标工作负载下，端到端成本和质量是否都满足要求？

Mamba 等选择性状态空间模型说明，递推路线可以通过输入相关状态更新和硬件感知扫描改善传统瓶颈，并在论文实验中取得有竞争力的语言建模结果。它们拓展了架构选择，而不是对 Transformer 优劣的一次性裁决；两类方法的质量、状态表示、硬件和生态仍需按相同任务与预算比较。

## 1.10 为什么不是 RNN/LSTM 成为大模型主干

面试里经常会问：为什么不是 LSTM？

不能用一句“RNN 无法并行”概括所有递推架构。更准确的比较要限定在经典 RNN/LSTM 和标准 full-attention Transformer。

第一，经典 RNN/LSTM 的状态沿序列递推，时间步之间有依赖；标准 Transformer 在给定整段训练文本后，可以并行计算各位置的表示。前者的顺序路径较长，后者用更多位置交互计算换取较短的跨 token 路径。

第二，两者保存历史信息的方式不同。RNN/LSTM 将过去汇总进固定维度状态；Transformer 在上下文窗口内保留逐 token 表示，并由 attention 加权读取。前者需要在有限状态中压缩信息，后者需要随上下文保存 KV cache，且 full attention 的训练计算随位置对数二次增长。哪种取舍更合适取决于任务是否需要访问长前缀中的离散细节。

第三，历史证据与工具积累不同。Transformer 原论文先在机器翻译上展示质量和并行训练优势；之后 scaling law 与 GPT-3 等工作又在 Transformer 语言模型上研究规模扩展和上下文学习。后来出现的 SSM/Mamba 研究针对长序列成本和内容选择提出另一条路线，并已显示有竞争力的结果。[Mamba 原始论文](https://arxiv.org/html/2312.00752) 这是研究与工程路线逐步形成的历史，不是“递推模型不能扩展”的证明。

状态空间模型提醒我们，递推和并行不是互斥选项：结构化递推可以改写成并行扫描或矩阵计算。Transformer 的优势在于提供通用、直接的上下文路由及成熟的训练路径；SSM 等方法则探索怎样以较小状态和更低序列成本处理输入。它们的差异要按模型、硬件和任务实测，不能把某一类方法的性质套给全部递推模型。

## 1.11 为什么不是 CNN 成为大模型主干

CNN 的优势是局部连接、位置维度并行和规整的卷积计算。Transformer 原论文也比较了标准卷积与 self-attention 的层复杂度、顺序运算数和最长路径；这些结论限定于论文讨论的层形式。[原始论文 §4](https://arxiv.org/html/1706.03762v7)

但单层宽度有限的卷积只读取局部窗口；扩大有效感受野要增加层数、kernel 或 dilation。由此带来几项与 attention 不同的取舍：

1. 远距离信息要沿多层卷积路径传递，路径长度取决于卷积窗口和网络结构。
2. 卷积按位置共享局部滤波器；标准 attention 则根据当前 token 内容，为可见位置计算不同的读取权重。
3. 因此，卷积可在局部模式和计算规整性上有优势，attention 则提供更直接的全局内容路由；具体质量与效率仍需在同一任务下比较。

卷积并没有消失。长卷积、局部卷积和混合架构继续利用其局部性与规整计算；这说明架构演进常是重新组合计算路径，而不是简单地由一种结构彻底取代另一种。

在通用语言模型中，attention 的全局动态路由曾成为一条有效主干路线；这不证明卷积无法处理语言，也不保证全局 attention 在长序列下始终更省资源。

## 1.12 Transformer 核心优势的证据与取舍

可以把 Transformer 成为基础架构的原因归纳为几项相互配合、各有边界的条件。

第一，信息路由灵活。

在掩码允许的位置之间，attention 可以建立直接的信息路径并按内容分配权重。这提供了处理长距离依赖和代码引用的机制，但不保证模型找到正确内容或生成正确答案。

第二，训练并行性强。

在整段文本已知的训练阶段，序列位置可并行计算；矩阵运算适合现代加速器。自回归生成阶段仍要逐 token 解码。

第三，统一目标简单。

Decoder-only 与 next-token prediction 让不同文本任务能共享一种训练目标；该目标也可用于其他架构，数据质量和任务迁移效果另需验证。

第四，scaling 行为好。

在已研究的模型与训练范围内，held-out loss 呈经验 scaling 规律；给定预算下参数和 token 的分配需要共同优化。不能据此断言所有下游能力都随规模单调增长。

第五，迁移能力强。

预训练模型可以通过上下文示例和后训练适配任务；具体效果取决于模型、数据、方法和评估，不是单由架构决定。

第六，推理形态自然。

自回归生成自然地产生聊天、代码和长文本；工具调用、多轮状态和记忆还依赖外部协议与运行时。

第七，生态成熟。

训练、推理、压缩和部署已经累积了大量 Transformer 相关实现与经验，形成路径依赖，但新架构可以借用或重建其中一部分工具链。

## 1.13 Transformer 的代价：成功不等于完美

Transformer 的主流地位不能掩盖它的瓶颈。

标准 dense full attention 的主要代价随序列长度二次增长。

对一层、单个样本、长度为 $T$ 的序列，注意力分数矩阵有 $T\times T$ 个位置对。计算分数 $QK^\top$ 和加权汇总 $PV$ 的总量级为 $O(T^2d)$，其中 $d$ 是各注意力头维度之和；这里不包括投影和 FFN 等整层计算，后者还会产生与宽度有关的额外成本。若批大小为 $B$，上述注意力计算量还要乘上 $B$。

```math
S=QK^\top,\qquad S\in\mathbb{R}^{T\times T},\qquad C_{\mathrm{attn}}=O(T^2d)
```

朴素实现可能把大分数矩阵写入显存；FlashAttention 等 exact attention 实现通过分块降低中间矩阵的读写与存储需求，但仍需计算 full attention 的位置交互，不能据此把二次计算说成线性。[FlashAttention 原始论文](https://arxiv.org/html/2205.14135v2)

自回归推理时，为避免每生成一个 token 都重新计算整个前缀，模型通常缓存每层各历史 token 的 Key 和 Value。只计算 K/V 张量本身，近似内存为：

```math
M_{\mathrm{KV}}\approx 2BLTH_{\mathrm{kv}}d_hb
```

其中 $B$ 是并行请求数，$L$ 是层数，$T$ 是每个请求已缓存的 token 数，$H_{\mathrm{kv}}$ 是 Key/Value head 数，$d_h$ 是每个 head 的维度，$b$ 是每个缓存元素占用的字节数，系数 2 对应 Key 和 Value 两份张量。标准 MHA 中 $H_{\mathrm{kv}}d_h$ 通常等于模型宽度；GQA/MQA 通过减少 KV head 数降低这部分存储。该估算不含分页元数据、对齐开销、量化尺度，也未处理张量并行下分片或复制差异；实际请求长度不同时还要按每条序列分别估算。

这解释了为什么 MQA、GQA、MLA、分页 KV cache、KV cache 量化、滑动窗口 attention 和上下文压缩会成为重要优化方向；它们分别改变缓存表示、存储方式或可见位置范围，不能混为同一种复杂度优化。

另外，Transformer 的归纳偏置也不强：

1. 不天然知道局部性。
2. 不天然有持久状态。
3. 不天然适合无限流式输入。
4. 对位置编码和长度外推敏感。
5. 某些模型和任务中，相关信息位于长上下文中间时性能可能下降；该现象受模型、任务和评估设置影响。[原始研究](https://arxiv.org/html/2307.03172)
6. 精确算法推理仍需要训练和后训练增强。

因此，后 Transformer 架构研究的核心不是“Transformer 一无是处”，而是试图补它的短板：

```text
更低复杂度、更强长上下文、更便宜 KV cache、更好状态记忆、更适合流式推理。
```

## 1.14 一个小例子：为什么 attention 适合代码补全

看一个简化代码片段：

```python
def normalize_scores(scores):
    total = sum(scores)
    return [score / total for score in scores]

values = [1, 2, 3]
result = normalize_scores(values)
```

示例状态：静态审阅，未运行。这个片段用于观察函数、参数和变量在多行上下文中的关系。

如果模型要理解最后一行，它需要关联：

1. `normalize_scores` 的函数定义。
2. `scores` 参数的含义。
3. `total` 的计算。
4. 返回值是列表。
5. `values` 是输入列表。

这些信息分布在不同位置。Self-Attention 允许最后一行的 token 直接关注函数定义、参数、变量和返回语句。按本例手算，`total` 为 6，`result` 的三个元素依次是 $1/6$、$1/3$、$1/2$；这些数值只用于说明程序里的对象关系，不是模型评测结果。

当相关代码已放入模型可见的上下文时，attention 提供了在这些 token 表示间建立信息路径的机制；检索工具则由外部系统选择要放入的代码片段。这说明 Transformer 可以利用跨行上下文，但不保证模型找到正确代码、执行正确补全，或单凭架构就使 coding agent 更可靠。

## 1.15 如何比较候选架构

“哪种架构更好”没有脱离任务与预算的统一分数。比较时先对齐训练数据、模型规模、计算预算、硬件和评估任务，再看信息怎样跨位置传递、训练如何并行、推理保存什么状态，以及证据覆盖到什么范围。下表比较的是几类代表性结构，不是对所有变体的排名。

| 架构路线 | 前文信息怎样到达当前位置 | 训练时的序列维并行性 | 自回归推理保存什么 | 主要取舍 |
| --- | --- | --- | --- | --- |
| 经典 RNN/LSTM | 通过前一步状态逐步传递 | 标准递推沿时间步串行；特定结构化递推可设计并行扫描 | 递推状态，不为每个前缀 token 保存一组独立 K/V | 状态紧凑，适合流式处理；按需找回多条离散历史信息时受状态表示与容量约束 |
| CNN/TCN | 局部卷积逐层扩大感受野 | 卷积窗口内的位置可并行 | 由具体因果卷积实现保留状态或历史激活 | 局部模式和规整计算有优势；全局信息需经过由深度、kernel 和 dilation 决定的路径 |
| Decoder-only Transformer | attention 根据当前位置内容，对因果掩码允许的前缀位置加权读取 | 输入序列已知时，可并行计算各位置；网络层仍依次执行 | 每层前缀 token 的 Key/Value 表示 | 全局动态路由和并行训练有吸引力；full attention 计算随位置对数二次增长，KV cache 随前缀增长 |
| 选择性 SSM（如 Mamba） | 由输入相关的状态更新选择传播或遗忘信息 | Mamba 为选择性状态更新设计硬件感知扫描；特定 SSM 还可采用矩阵形式 | 递推状态，而非标准逐 token KV cache | 长序列计算与状态紧凑有吸引力；记忆质量、任务适配和实际硬件效率需要按配置实测 |

这张表刻意不为“内容路由”“生态成熟度”或“上下文能力”赋主观分数。比如 Transformer 的一跳路径表示计算图连接较短；若要断言它在某个检索任务上更准，还需要对齐数据、模型和评测的实验证据。Mamba-2 论文进一步展示，特定结构化状态空间运算可以写成线性递归形式，也可以写成二次矩阵形式；因此“递推”不自动等于训练时只能逐 token 串行计算。[Mamba-2/SSD 原始论文](https://arxiv.org/html/2405.21060)

对实际模型选择，可以把表中的问题改成逐项核验：目标任务是否需要显式访问长前缀中的细节？训练瓶颈是计算、内存还是通信？推理时保存 K/V 还是较小的递推状态，哪种方式更适合请求长度和硬件？已有实验是否在可比预算下覆盖了目标任务？需要重建哪些 kernel、权重格式和部署工具？答案应落在具体模型配置和工作负载上，不能由这张概念表直接算出一个“赢家”。

## 1.16 面试题

### 题 1：Transformer 为什么能成为大模型基础架构？

参考回答：

> Transformer 的优势来自几方面配合：attention 为可见位置提供直接、内容相关的信息路径；在整段文本已知时，各位置可并行训练；decoder-only 结构能用 next-token prediction 学习多类文本续写；后续研究又在这一架构上积累了规模实验、训练配方和工具链。证据要按论文范围理解：scaling law 研究的是特定设定下的 loss，GPT-3 展示的 few-shot 行为有任务差异，Chinchilla 的配比也绑定其预算和模型范围。Transformer 不是所有单项都最优的架构。

### 题 2：Transformer 相比 RNN/LSTM 的核心优势是什么？

参考回答：

> 对经典 RNN/LSTM 来说，序列状态逐步依赖前一状态；Transformer 在训练时可以并行计算已知序列的各位置，并在 attention 层建立较短的跨 token 路径。代价是 full attention 的位置交互计算和逐 token KV cache。结构化状态模型也能设计并行扫描，所以比较时要说清模型变体、任务和硬件。

### 题 3：为什么 decoder-only 成为通用 LLM 主流？

参考回答：

> Decoder-only 的因果掩码与自回归 next-token prediction 配合自然，可以把多种文本任务整理成给定前缀后的续写训练。这个目标并非 Transformer 独有；它的吸引力还包括实现简单、能利用大量未标注文本，以及生成接口直接。逐 token 解码、训练数据和后训练仍会影响最终能力。

### 题 4：Scaling law 对 Transformer 路线有什么影响？

参考回答：

> Kaplan 等人报告了特定语言模型实验中 held-out cross-entropy 与模型规模、数据和计算量之间的经验关系；Chinchilla 则在其预算设定下重新估计参数量和 token 数的配比。这些结果让规模实验更可规划，但不保证所有下游能力随规模单调上升，也不提供跨数据和架构通用的最优比例。

### 题 5：既然 Transformer 这么强，为什么还研究 Mamba、SSM、RWKV、RetNet？

参考回答：

> 标准 dense full attention 的计算随序列位置对数二次增长，decoder-only 推理还要为前缀保存每层的 KV 表示。SSM、Mamba、RWKV、RetNet 等路线探索递推状态、长序列成本和流式计算的不同取舍；Mamba 已展示有竞争力的论文结果。是否适合某个场景要在同一任务、预算和硬件下评估，不能靠架构名称或渐近复杂度直接判胜负。

## 1.17 小练习

1. 用一张表比较 RNN、CNN、Transformer 在并行性、长距离依赖、状态表示、硬件友好性上的差异。
2. 手写 Self-Attention 的 Q、K、V 直觉解释，不使用公式也能讲清楚。
3. 解释为什么 decoder-only + next-token prediction 可以统一翻译、摘要、问答和代码生成。
4. 阅读 *Attention Is All You Need* 摘要，找出作者强调的三个关键词：attention、parallelizable、training cost。
5. 阅读 GPT-3 摘要，解释 few-shot learning 和 fine-tuning 的区别。
6. 阅读 Chinchilla 摘要，解释为什么“更大模型”不一定 compute-optimal。
7. 用 2 分钟回答“Transformer 会不会被 Mamba 替代”，要求同时讲 Transformer 优势和瓶颈。
8. 选一种本章没有展开的序列架构，按表中的信息路由、训练并行、推理状态和证据范围四项写一份定性比较；把来源直接支持的事实与自己的推断分开标明。

## 1.18 本章总结

本章回答了“Transformer 为什么能成为大模型基础架构”。

核心结论：

1. 相较经典递推层，Transformer 在已知整段序列时能并行计算各位置，并提供较短的跨 token 信息路径；这不等于所有递推架构都无法并行。
2. Self-Attention 提供内容相关的加权读取机制，但不保证信息无损、检索正确或回答可靠。
3. 矩阵运算与加速器适配、训练规模证据、统一文本目标和生态积累共同推动 Transformer 成为主干路线。
4. Decoder-only 与 next-token prediction 组合自然，但训练目标也可用于其他序列架构。
5. GPT-3 展示了上下文示例在许多任务中可支持不更新参数的任务适配，同时存在任务差异与评估边界。
6. Scaling law 研究的是特定训练范围内的 loss 规律；Chinchilla 的参数/token 配比也绑定论文预算与模型设定。
7. LLaMA 论文报告其模型使用公开可得数据集训练，并向研究社区发布模型；这不等于完整训练语料文件均已开放。
8. Dense full attention 的注意力计算随序列位置对数二次增长；自回归推理 KV cache 的估算要纳入 batch、层数、KV head、head 维度和数据类型。
9. Mamba、SSM、RWKV、RetNet、Hyena 等路线探索不同的状态、长序列计算和硬件取舍，需按具体任务与配置评估。

下一章会深入 Self-Attention 的信息路由本质，重点讲 Q、K、V、attention score、softmax 和多 token 交互到底在学什么。
