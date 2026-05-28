# 第七章：Transformer 与架构论文线

从这一章开始，第十册进入具体论文线。本章讨论 Transformer 与架构演化。目标不是把每篇论文讲成百科条目，而是训练一种读架构论文的能力：它解决了什么结构性瓶颈？它改的是 attention、FFN、归一化、位置编码、训练目标、稀疏激活还是硬件执行？它的收益来自架构本身，还是来自数据、规模、训练配方和工程实现？

大模型架构论文最容易被误读。很多人看到新名字就以为是革命，看到 benchmark 提升就认为架构更强。但在真实研究中，架构贡献必须和训练数据、token 数、参数量、compute、推理成本、实现复杂度一起看。

本章重点：Attention Is All You Need、GPT、BERT、LLaMA、Mistral、MoE、SSM。

---

## 1. 架构论文要读什么？

读架构论文，不要只问“结构长什么样”，而要问：

1. 它解决哪个瓶颈？
2. 它改变了哪部分计算图？
3. 它带来什么归纳偏置？
4. 它的训练和推理成本如何变化？
5. 它和数据、规模、硬件的关系是什么？
6. 它在什么任务和规模下验证？
7. 它有没有真实 ablation 证明架构贡献？

架构论文通常有三类贡献：

1. 表达能力：模型能表示更复杂依赖或更长记忆。
2. 训练效率：更容易并行、更稳定、更少显存或更快收敛。
3. 推理效率：更低延迟、更高吞吐、更低 KV cache 或更长上下文。

你要先判断论文主打哪一类。

---

## 2. Attention Is All You Need：真正的新问题是什么？

Transformer 论文的历史意义不只是提出 self-attention，而是把序列建模从 RNN/CNN 的时序递归和局部卷积中解放出来。

当时主流 sequence transduction 模型依赖 recurrent 或 convolutional 网络，虽然 attention 已经存在，但通常作为 encoder-decoder 之间的辅助机制。Transformer 的关键主张是：完全基于 attention，不需要 recurrence 和 convolution，也能做高质量序列建模，并且更容易并行训练。

它解决的核心 trade-off 是：序列依赖建模 vs 并行效率。

RNN 擅长按时间推进，但训练并行性差，长距离依赖困难。CNN 并行性好，但长距离依赖需要堆很多层。Self-attention 让任意位置之间直接交互，同时训练阶段可以并行计算。

---

## 3. Transformer 的核心贡献怎么判断？

Transformer 的贡献可以拆成几层：

1. Self-attention 作为主干计算。
2. Multi-head attention 提供多子空间关系建模。
3. Position encoding 补充序列顺序信息。
4. Encoder-decoder attention 保留 seq2seq 能力。
5. FFN 提供逐 token 非线性变换。
6. Residual + LayerNorm 提升深层训练稳定性。
7. 完全并行化训练提升效率。

读这篇论文时，不要只背 QKV。要抓住它的证据链：在机器翻译任务上，Transformer 达到更好 BLEU，同时训练时间显著降低。这说明贡献不只是效果，也包括可并行训练的系统价值。

---

## 4. GPT 和 BERT：同一架构，不同预训练范式

Transformer 之后，GPT 和 BERT 代表了两种重要方向。

GPT 路线使用 decoder-only 自回归语言模型，目标是预测下一个 token。它天然适合生成，并逐渐成为大模型主流形态。

BERT 路线使用 encoder-only 双向表示，通过 masked language modeling 等目标学习上下文表示，适合理解类任务和表示学习。

BERT 的关键贡献不是“用了 Transformer”，而是证明深层双向 Transformer 预训练可以在多个 NLP 理解任务上通过简单 fine-tuning 获得强结果。

读 GPT/BERT 这类论文时，要把架构和训练目标分开：

1. 架构是 encoder、decoder 还是 encoder-decoder？
2. 训练目标是 causal LM、masked LM、seq2seq denoising 还是别的？
3. 下游适配方式是 prompting、fine-tuning 还是 instruction tuning？

很多面试回答混淆“架构贡献”和“预训练目标贡献”，这是常见错误。

---

## 5. Decoder-only 为什么成为主流？

现代大语言模型多采用 decoder-only 架构。原因不是 encoder 不好，而是 decoder-only 与生成式预训练和统一接口高度匹配。

优势包括：

1. 训练目标简单：next-token prediction。
2. 生成能力天然匹配。
3. prompt、completion、multi-turn dialogue 可以统一成序列建模。
4. scaling law 研究和工程栈成熟。
5. 推理时 KV cache 机制清晰。

但 decoder-only 也有成本：长上下文 attention 成本高，KV cache 占用大，双向理解能力需要通过数据和目标弥补。

架构论文线的一个核心主题，就是在 decoder-only 主流框架下继续优化 attention、位置、FFN、归一化、稀疏激活和长上下文。

---

## 6. 位置编码论文线：模型如何知道顺序？

Transformer 本身没有 RNN 的天然顺序结构，因此需要位置编码。

位置相关方法可以分几类：

1. Absolute position embedding。
2. Sinusoidal position encoding。
3. Relative position encoding。
4. RoPE。
5. ALiBi。
6. Long-context position interpolation 或 extrapolation。

读位置编码论文，要问：

1. 它解决训练长度和推理长度不一致的问题吗？
2. 它对长上下文外推是否稳定？
3. 它是否改变 attention score？
4. 它对不同频率和距离的归纳偏置是什么？
5. 它是否需要继续训练或位置插值？

位置编码不是细节。长上下文能力很大程度受位置建模影响。

---

## 7. 归一化和残差：训练稳定性的架构线

LayerNorm、Residual、Pre-LN、Post-LN、RMSNorm 这些改动看起来不如 attention 新潮，但对大模型训练稳定性非常关键。

读这类论文时，要关注：

1. 梯度流是否更稳定。
2. 深层网络是否更容易训练。
3. 是否减少数值不稳定。
4. 是否降低计算或参数。
5. 是否影响收敛速度和最终性能。

很多现代 LLM 使用 RMSNorm、Pre-Norm 等设计，不是因为它们概念华丽，而是因为大规模训练中稳定、便宜、有效。

架构论文不一定要提出巨大模块，小而关键的稳定性改动也可能很有价值。

---

## 8. FFN 和激活函数：被低估的主力模块

Transformer block 中 FFN 占大量参数和计算。很多能力并不只来自 attention，FFN 也承载大量知识和非线性变换。

常见演化包括：

1. ReLU 到 GELU。
2. SwiGLU / GEGLU。
3. 更大 FFN expansion ratio。
4. MoE 将 FFN 替换为多个 expert。

读 FFN 相关论文，要问：

1. 激活函数是否改善表达能力或梯度？
2. 参数增加是否公平控制？
3. 计算成本是否增加？
4. 效果来自 FFN 设计还是整体训练配方？

---

## 9. LLaMA：现代开源 LLM 架构配方

LLaMA 的论文价值不只是“开源模型”，还在于展示了高效训练 foundation language models 的现代配方：公开数据、大量 token、相对高效的架构选择和严谨评估。

LLaMA 使用一系列后来很常见的设计，包括 decoder-only、Pre-normalization、RMSNorm、SwiGLU、RoPE 等。它的结论强调：在合适数据和训练 token 下，较小模型也能非常强。

读 LLaMA 时要注意：

1. 架构改动并非唯一贡献，数据和训练配方同样重要。
2. 模型大小和训练 token 的平衡影响很大。
3. 开源可复现性和社区影响也是贡献。
4. 评估要看多任务、多模型规模和与强 baseline 的比较。

面试中不要把 LLaMA 讲成单一架构创新。更准确地说，它是现代 LLM 训练配方和开放研究生态的重要节点。

---

## 10. Mistral：小模型高效率的架构与系统取舍

Mistral 7B 的论文强调 7B 参数模型在性能和效率上的强表现。它使用 GQA 提升推理效率，使用 sliding window attention 处理较长序列并降低推理成本。

读 Mistral 时，要抓住它解决的 trade-off：小模型性能 vs 推理效率 vs 长上下文成本。

GQA 的价值在于减少 KV cache 和推理带宽压力。Sliding window attention 的价值在于降低长序列 attention 成本，但它也引入局部窗口限制，需要配合机制处理远距离信息。

判断这类论文贡献时，不能只看 benchmark，还要看：

1. 吞吐和延迟。
2. KV cache 占用。
3. 长上下文质量。
4. 训练和推理实现复杂度。
5. 与同尺寸和更大模型的公平比较。

---

## 11. MoE：用稀疏激活扩大容量

MoE 的核心思想是 conditional computation：模型有很多参数，但每个 token 只激活其中一部分 expert，从而在不同比例增加计算的情况下扩大容量。

早期 Sparsely-Gated MoE 工作展示了大幅增加模型容量的潜力。后来 Switch Transformer、GShard、Mixtral、DeepSeekMoE 等继续推动 MoE 在 LLM 中落地。

读 MoE 论文，要问：

1. Router 如何分配 token？
2. 每个 token 激活几个 expert？
3. load balancing 如何保证 expert 不塌缩？
4. 通信成本如何处理？
5. 训练是否稳定？
6. 推理时吞吐、延迟和显存如何？
7. 参数量和激活参数量如何比较才公平？

MoE 的贡献不是简单“参数更多”，而是容量、计算、通信和路由稳定性的系统取舍。

---

## 12. SSM 和 Mamba：Transformer 替代路线

长序列建模中，attention 的二次复杂度一直是瓶颈。SSM、linear attention、gated convolution、recurrent model 等都试图提供更低复杂度的替代。

Mamba 的核心主张是 selective state spaces：让 SSM 参数依赖输入，从而增强内容选择能力，并通过 hardware-aware parallel algorithm 实现高效训练和推理。

读 Mamba 这类论文，要抓住两个问题：

1. 它如何弥补传统 subquadratic 模型在语言上不如 attention 的问题？
2. 它的效率优势是否在真实硬件和真实任务中成立？

Mamba 论文中强调线性序列长度扩展、快速推理和多模态适用性。但读这类论文也要看边界：复杂 reasoning、in-context learning、长距离精确检索、与 Transformer 同等训练预算下的比较，都需要仔细评估。

---

## 13. 架构论文的共同评价框架

评价架构论文，可以用这个框架：

1. Bottleneck：它解决什么瓶颈？
2. Mechanism：它改了哪部分结构？
3. Inductive bias：它引入什么偏置？
4. Cost：训练和推理成本如何？
5. Evidence：实验是否证明架构贡献？
6. Ablation：是否排除数据、规模、调参影响？
7. Scaling：不同模型大小是否稳定？
8. Deployment：是否容易落地？
9. Boundary：在哪些场景可能失效？

这个框架适用于 Transformer、MoE、SSM、长上下文和推理优化论文。

---

## 14. 架构论文最容易混淆的变量

架构论文中常见混淆变量包括：

1. 数据更多。
2. 训练 token 更多。
3. 模型参数更多。
4. 激活参数和总参数混淆。
5. 调参更充分。
6. 训练稳定性 trick 不透明。
7. 评估任务偏向新方法。
8. 硬件实现不同。
9. 推理 batch 和序列长度设置不同。

如果这些变量没有控制好，很难判断提升是否来自架构本身。

---

## 15. 面试中如何讲 Transformer 论文线？

可以按这样讲：

第一，Transformer 解决 RNN/CNN 序列建模并行效率和长距离依赖问题，用 self-attention 做主干。

第二，GPT/BERT 把 Transformer 和不同预训练目标结合，形成生成式 decoder-only 和理解式 encoder-only 两条路线。

第三，现代 LLM 主要沿 decoder-only 发展，并在位置编码、归一化、FFN、attention 变体、长上下文和数据配方上持续优化。

第四，LLaMA、Mistral 代表现代高效 LLM 配方；MoE 代表稀疏容量扩展；Mamba/SSM 代表 attention 替代路线。

第五，判断架构论文不能只看 SOTA，要看训练 token、数据、compute、推理成本、ablation 和可部署性。

---

## 16. 典型面试追问

### 16.1 Transformer 相比 RNN 的关键优势是什么？

关键是 self-attention 让任意位置直接交互，同时训练阶段更容易并行。RNN 递归依赖时间步，难以高效并行；Transformer 更适合大规模训练。

### 16.2 GPT 和 BERT 的核心区别是什么？

GPT 是 decoder-only 自回归模型，适合生成；BERT 是 encoder-only 双向表示模型，适合理解任务。区别不仅是架构，还有训练目标和下游使用方式。

### 16.3 LLaMA 的贡献是架构创新吗？

不完全是。LLaMA 的价值更多在于现代 LLM 训练配方、公开数据、高效架构选择、较充分训练 token 和开放研究影响。它不是单一模块创新。

### 16.4 GQA 有什么作用？

GQA 通过减少 key/value head 数量降低 KV cache 和推理带宽压力，在保持较好质量的同时提高推理效率。

### 16.5 MoE 为什么能扩大模型容量？

因为每个 token 只激活部分 expert，总参数量很大，但激活计算相对有限。难点是路由、负载均衡、通信成本和训练稳定性。

### 16.6 Mamba 想解决什么问题？

它试图用 selective SSM 提供线性时间序列建模，缓解 attention 长序列成本，同时通过输入依赖参数增强内容选择能力。

---

## 17. 常见误区

误区一：架构论文只看结构图。

结构图只是表面，关键是瓶颈、机制、成本和证据。

误区二：benchmark 高就是架构更好。

提升可能来自数据、token 数、训练配方或调参。

误区三：总参数量越大模型越贵。

MoE 中要区分总参数量和每 token 激活参数量。

误区四：长上下文方法只看最大 context length。

还要看长上下文质量、推理成本、KV cache、needle retrieval 和真实任务表现。

误区五：Transformer 替代架构一定会取代 Transformer。

替代路线需要在质量、成本、生态、硬件和训练稳定性上全面比较。

---

## 18. 本章小结

Transformer 与架构论文线是大模型研究的主干之一。

本章要记住几句话：

1. Transformer 的核心贡献是用 attention 主干提升序列建模并行性和长距离交互。
2. GPT/BERT 的差异不仅是架构，也是预训练目标和使用方式。
3. 现代 LLM 架构演化围绕位置、归一化、FFN、attention 效率、长上下文和稀疏激活展开。
4. LLaMA 和 Mistral 体现了现代高效 LLM 配方与工程取舍。
5. MoE 用稀疏激活扩大容量，但引入路由和通信挑战。
6. SSM/Mamba 是 attention 替代路线，但需要谨慎评估质量和边界。
7. 判断架构论文必须同时看机制、数据、规模、成本、ablation 和部署价值。

如果只能记住一个框架：架构论文不是问“结构新不新”，而是问“它解决了什么瓶颈，代价是什么，证据是否排除了数据和规模因素”。
