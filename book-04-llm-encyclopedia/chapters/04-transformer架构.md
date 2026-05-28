# D. Transformer 架构

条目：Transformer、Token Embedding、Embedding Matrix、Position Embedding、Positional Encoding、Sinusoidal Positional Encoding、RoPE、RoPE Scaling、ALiBi、Long Context、Position Extrapolation、Lost in the Middle、Weight Tying、Contextual Representation、Self-Attention、Query、Key、Value、Attention Weight、Causal Self-Attention、Causal Mask、Padding Mask、Causal LM、Masked LM、Scaled Dot-Product Attention、Full Attention、Local Attention、Sparse Attention、Linear Attention、FlashAttention、Attention Complexity、Multi-Head Attention、Attention Head、MQA、GQA、KV Cache、Transformer Block、MLP / FFN、Residual Connection、Pre-LN、Post-LN、Decoder-Only Transformer、Encoder-Only Transformer、Encoder-Decoder Transformer、Cross-Attention。

## Transformer

一句话定义：Transformer 是一种以 attention 为核心的信息交互架构，用堆叠的 attention、MLP、normalization 和 residual connection 对序列进行建模。

提出背景：早期 NLP 常用 RNN、LSTM、GRU 等循环结构建模序列。它们天然按时间步递推，适合表达顺序，但训练并行度差，长距离依赖难学。Transformer 用 self-attention 让任意位置之间可以直接交互，同时让训练过程在序列维度上高度并行。

解决的问题：Transformer 主要解决三件事。第一，增强长距离依赖建模能力。第二，提高训练并行效率。第三，让模型可以通过堆叠层数和扩大宽度形成强大的通用表示能力。

基本组成：一个典型 Transformer block 包含 attention 子层、MLP 子层、残差连接和归一化层。大语言模型中最常见的是 decoder-only Transformer，也就是 GPT 类自回归架构。

为什么适合大模型：attention 能根据上下文动态选择相关信息，MLP 提供非线性变换和知识存储能力，残差和归一化让深层网络可训练，整体结构规整，适合大规模 GPU/TPU 并行。

局限：标准 attention 的复杂度随序列长度平方增长，长上下文训练和推理成本很高；同时 Transformer 本身没有强归纳偏置，需要大量数据和算力学习语言规律。

面试表达：Transformer 的核心不是“多头”本身，而是用 self-attention 替代循环递推，让 token 之间可以全局交互，并把序列建模转化为高度并行的矩阵计算。

## Token Embedding

一句话定义：token embedding 是把 token id 映射成连续向量的可训练查表层。

形状：embedding matrix 通常是 `[vocab_size, hidden_size]`。输入 token id 序列形状为 `[B, T]`，查表后变成 `[B, T, hidden_size]`。

为什么需要它：token id 只是离散编号，本身没有语义距离。模型不能直接对 id 的大小做语义计算，所以需要把每个离散 token 映射成连续向量。

训练方式：embedding 向量通过语言模型 loss 反向传播更新。频繁出现的 token 通常得到更充分训练，低频 token 和新增 token 更容易表示不足。

工程注意：tokenizer、vocabulary 和 embedding matrix 必须对齐。换 tokenizer 或新增 special token 后，通常需要 resize embedding，并决定新 token 向量如何初始化。

常见误区：embedding 不是固定词典解释，而是模型在训练中学习到的参数。一个 token 的静态 embedding 只表示初始输入，经过 Transformer 层后会变成上下文相关表示。

面试表达：token embedding 是离散文本进入神经网络的第一层接口，负责把 token id 转为模型可计算的稠密向量。

## Embedding Matrix

一句话定义：embedding matrix 是存储所有 token 向量表示的参数矩阵。

参数规模：如果词表大小为 `V`，隐藏维度为 `d`，输入 embedding 参数量约为 `V * d`。对于大词表和大 hidden size，这部分参数并不小。

与输出层关系：语言模型最后需要把 hidden state 投影到词表维度得到 logits。很多模型会使用 weight tying，让输入 embedding 和输出 projection 共享权重。

新增 token 的影响：新增 token 会改变词表大小，需要扩展 embedding matrix 和输出层。如果只改 tokenizer 而不改模型参数，会出现 id 越界或语义错位。

工程坑：加载 checkpoint 时如果 tokenizer 和 embedding size 不一致，常见报错是 shape mismatch。正确做法是确认 tokenizer 版本、special token 数量、vocab size 和模型 config 是否一致。

面试表达：embedding matrix 是模型理解 token 的参数入口，tokenizer 的 id 映射必须和这个矩阵逐行对应。

## Position Embedding

一句话定义：position embedding 是向模型注入 token 位置信息的机制。

为什么需要它：self-attention 本身对输入顺序没有天然感知。如果不加入位置信息，序列中的 token 交换顺序后，attention 计算缺少区分位置的依据。

常见形式：learned absolute position embedding、sinusoidal positional encoding、RoPE、ALiBi。现代 decoder-only LLM 中 RoPE 非常常见。

绝对位置与相对位置：绝对位置强调第几个 token，相对位置强调两个 token 之间相隔多远。长上下文外推通常更关注相对位置信息是否稳定。

工程注意：位置编码会影响上下文长度扩展。仅把 config 中的 max position 改大，通常不能保证模型能稳定处理更长文本。

面试表达：position embedding 解决的是 Transformer 对序列顺序不敏感的问题，它决定模型如何理解“谁在前、谁在后、相隔多远”。

## Positional Encoding

一句话定义：positional encoding 是让模型知道 token 在序列中位置的编码方法。

核心问题：语言顺序很重要，但 attention 主要根据 Q/K 内容相似度计算。如果没有位置编码，模型很难区分“狗咬人”和“人咬狗”这类顺序变化。

主要类型：第一类是可学习绝对位置向量，直接为每个位置学习一个 embedding。第二类是固定函数位置编码，如正弦余弦编码。第三类是把相对位置融入 attention 计算，如 RoPE 和 ALiBi。

评价维度：一个位置编码方法通常要看训练稳定性、短上下文效果、长度外推能力、实现复杂度和与 KV Cache 的兼容性。

常见误区：支持更长位置编码不等于模型真正学会长程依赖。长上下文能力还依赖训练数据长度分布、attention 机制、优化过程和评估方式。

面试表达：位置编码不是附属细节，它直接影响模型对顺序、距离和长上下文的理解。

## Sinusoidal Positional Encoding

一句话定义：sinusoidal positional encoding 是 Transformer 原论文中使用的固定正弦/余弦位置编码。

核心形式：不同维度使用不同频率的正弦和余弦函数表示位置。低频维度变化慢，高频维度变化快。

为什么提出：它不需要为每个位置学习参数，理论上可以计算训练长度之外的位置编码，具有一定外推直觉。

优点：无额外可训练参数，实现简单，原始 Transformer 中效果良好。

局限：在现代大规模 decoder-only LLM 中，它不如 RoPE 等相对位置方法常见；实际长上下文表现还取决于模型是否在长序列上训练过。

面试表达：正弦位置编码用固定周期函数表示位置，优势是简单和可外推，但现代 LLM 更常用能表达相对位置关系的方案。

## RoPE / Rotary Position Embedding

一句话定义：RoPE 通过按位置旋转 Q/K 向量来注入位置信息。

核心直觉：不同位置对应不同旋转角度。两个位置的 Q/K 点积不仅包含内容相似度，也包含相对位置差带来的相位关系。

作用位置：RoPE 通常作用在 attention 的 query 和 key 上，而不是直接加到 token embedding 上。

为什么流行：RoPE 能自然建模相对位置信息，和 decoder-only Transformer、KV Cache 兼容较好，因此被许多现代 LLM 采用。

优点：表达相对位置关系清晰，短上下文效果好，实现上可以高效融合到 attention 中。

局限：训练长度之外的位置外推仍可能退化。长上下文扩展时需要 RoPE scaling、继续训练或其他长上下文技术配合。

工程坑：RoPE 的 base、维度分配、scaling 策略、max position 设置必须和 checkpoint 对齐。推理框架中 RoPE 参数不一致，会导致模型输出质量明显异常。

面试表达：RoPE 不是简单给 embedding 加位置向量，而是在 Q/K 空间做旋转，使 attention score 自然携带相对位置信息。

## RoPE Scaling

一句话定义：RoPE scaling 是通过调整 RoPE 频率或位置映射来扩展模型上下文长度的方法。

为什么需要它：模型预训练时通常只见过有限长度。如果直接推理更长序列，高位置对应的旋转相位可能超出训练分布，导致注意力模式不稳定。

常见思路：一种是位置插值，把更长的位置压缩映射到训练长度范围附近；一种是调整频率 base 或不同维度的缩放方式；还有一些方法会结合长上下文继续训练。

优点：相对轻量，常可在不从零训练模型的情况下扩展可用上下文。

风险：可能影响短上下文能力，也不解决 attention 计算和 KV Cache 显存随长度增长的问题。

面试表达：RoPE scaling 解决的是位置编码外推问题，不等于完整解决长上下文问题；长上下文还要处理训练分布、检索能力、显存和评估。

## ALiBi

一句话定义：ALiBi 是在 attention score 中加入与距离相关的线性偏置的位置建模方法。

核心机制：距离越远的 token，会根据 head 对应的斜率受到不同程度的 score 惩罚，从而把相对距离信息注入 attention。

优点：不需要显式位置 embedding，外推到更长长度时形式简单，推理时实现相对直接。

局限：线性距离偏置是一种较强归纳假设，不一定适合所有任务和模型规模。现代 LLM 中 RoPE 使用更广。

面试表达：ALiBi 通过给远距离 attention 加惩罚来编码相对位置，重点是把距离偏置直接加到 attention logits 上。

## Long Context

一句话定义：long context 是模型处理远超传统窗口长度输入的能力。

核心挑战：第一，标准 attention 计算和显存复杂度是 `O(T^2)`。第二，推理时 KV Cache 随上下文长度线性增长。第三，位置编码需要外推或重新训练。第四，模型不一定能有效利用很远位置的信息。

能力分层：能接收长输入只是第一层；能检索长输入中的关键信息是第二层；能在长输入中做跨段推理、冲突处理和多证据整合才是更高层能力。

典型方案：位置编码扩展、长上下文继续预训练、稀疏或滑窗 attention、分块注意力、检索增强、KV Cache 压缩、外部记忆机制。

工程评估：不能只看最大 context length。还要看长文问答、needle-in-a-haystack、多跳检索、跨段总结、位置鲁棒性和实际延迟成本。

面试表达：长上下文不是把窗口参数调大，而是位置编码、训练数据、attention 成本、KV Cache 和评估共同决定的系统能力。

## Position Extrapolation

一句话定义：position extrapolation 是模型在训练长度之外处理更长位置的能力。

为什么困难：训练时未见过的位置可能对应未学习过的编码模式，attention score 分布和层间表示都可能偏离训练分布。

常见现象：短上下文正常，超长上下文开始复读、忽略远端信息、答案不稳定、对中间信息不敏感。

改善方向：RoPE scaling、长上下文继续训练、更合理的位置插值、使用相对位置偏置、加入长文任务数据。

面试表达：位置外推关注的是“位置编码和模型是否能泛化到更长位置”，不是单纯显存能否装下长输入。

## Lost in the Middle

一句话定义：lost in the middle 是长上下文模型更容易忽略上下文中间部分信息的现象。

典型表现：关键信息放在开头或结尾时模型更容易回答正确，放在中间时准确率下降。

原因直觉：训练数据和注意力模式可能让模型更偏好近邻、开头指令和结尾问题；长序列中中间信息竞争注意力资源，也更容易被稀释。

评估意义：它说明支持长窗口不等于能可靠利用所有位置的信息。

工程缓解：把关键约束放在开头和结尾，使用结构化提示，分段摘要，检索重排，显式引用证据位置，对长文任务做专项微调。

面试表达：长上下文评估不能只测是否能塞进去，还要测不同位置的信息是否都能被稳定使用。

## Weight Tying

一句话定义：weight tying 是输入 embedding matrix 和输出词表投影矩阵共享参数。

为什么使用：语言模型输入端需要 token embedding，输出端需要把 hidden state 映射到 vocabulary logits。两者都和 token 语义空间相关，因此可以共享权重。

优点：减少参数量，提升参数利用率，让输入 token 表示和输出 token 分类共享空间。

局限：共享权重是一种约束，不一定适合所有结构；新增 token 或 resize embedding 时，需要同时关注输入和输出权重是否同步。

工程注意：某些模型 config 中会有 `tie_word_embeddings`。做 LoRA、量化、保存和加载时，要确认共享权重没有被错误拆开。

面试表达：weight tying 让模型输入词表表示和输出词表分类器使用同一套 token 参数，既省参数又形成语义约束。

## Contextual Representation

一句话定义：contextual representation 是 token 经过 Transformer 层并结合上下文后得到的动态表示。

与 embedding 的区别：token embedding 是静态查表结果，同一个 token 在不同句子中初始向量相同；contextual representation 会随上下文变化。

例子：`bank` 在 “river bank” 和 “bank account” 中初始 token embedding 可能相同，但经过 attention 后的表示会分别偏向河岸和银行语义。

为什么重要：大模型的理解、推理和生成能力主要来自多层上下文表示，而不是单个静态词向量。

面试表达：Transformer 的每一层都在更新 token 的上下文表示，越高层越融合全局语境和任务信息。

## Self-Attention

一句话定义：self-attention 是序列中每个 token 根据同一序列内其他 token 的信息更新自身表示的机制。

核心直觉：每个 token 先生成 query、key、value。query 表示“我想找什么”，key 表示“我能被如何匹配”，value 表示“我提供什么内容”。query 和 key 算相似度，softmax 得到权重，再加权汇总 value。

解决问题：self-attention 让任意两个位置可以直接交互，适合建模长距离依赖；同时矩阵计算可并行，训练效率高于逐步递推的 RNN。

标准公式：`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`。

优点：动态上下文选择、全局信息交互、并行训练、表达能力强。

局限：标准 attention 需要计算 `[T, T]` attention matrix，复杂度为 `O(T^2)`，长序列成本高。

常见误区：attention weight 可以提供解释线索，但不等于完整因果解释。模型行为还受 value、MLP、残差、多层交互影响。

面试表达：self-attention 的本质是基于内容相似度的信息路由机制，每个 token 都能动态决定从哪些上下文位置聚合信息。

## Query

一句话定义：query 是 attention 中表示“当前 token 想找什么信息”的向量。

来源：由当前 hidden state 通过线性投影得到，通常记为 `Q = X W_Q`。

类比：query 像搜索请求，表示当前位置要向上下文检索哪些信息。

形状：在多头注意力中，query 常被 reshape 成 `[B, num_heads, T, head_dim]`。

面试表达：query 不是人工写的查询语句，而是模型根据当前 token 表示动态生成的向量。

## Key

一句话定义：key 是 attention 中表示“当前 token 可被如何匹配”的索引向量。

来源：由 hidden state 通过线性投影得到，通常记为 `K = X W_K`。

类比：key 像文档索引，用来和 query 计算匹配分数。

在 KV Cache 中：自回归推理会缓存历史 token 的 key，避免每生成一步都重复计算历史部分。

面试表达：key 决定一个 token 在 attention 匹配中以什么方式被其他 token 找到。

## Value

一句话定义：value 是 attention 中真正被加权汇总的信息向量。

来源：由 hidden state 通过线性投影得到，通常记为 `V = X W_V`。

类比：value 像文档内容。query 和 key 决定查哪些位置，value 决定取回什么信息。

在 KV Cache 中：历史 token 的 value 也会被缓存，因为生成后续 token 时仍需聚合历史信息。

面试表达：attention score 只决定权重，最终进入下一层的信息来自 value 的加权和。

## Attention Weight

一句话定义：attention weight 是当前 token 从其他 token 聚合信息时使用的权重。

计算方式：query 和 key 点积得到 attention score，经过缩放、mask 和 softmax 后得到权重。

性质：同一个 query 对所有可见 key 的权重通常归一化为 1。mask 掉的位置权重应接近 0。

解释风险：较大的 attention weight 说明某层某 head 在该步从某位置聚合较多 value，但不能直接等同于模型最终答案的因果依据。

工程用途：可用于可视化、debug 长上下文关注位置、检查 mask 是否生效、分析重复生成或信息泄漏。

面试表达：attention weight 是信息聚合权重，不是模型解释的全部。

## Causal Self-Attention

一句话定义：causal self-attention 是当前位置只能关注自己和之前位置，不能关注未来位置的 self-attention。

典型场景：GPT 类自回归语言模型。

为什么需要：语言模型训练目标是预测下一个 token。如果当前位置能看到未来 token，就会产生信息泄漏，训练 loss 虽低但生成时不可用。

实现方式：使用 causal mask，在 softmax 前把未来位置的 attention scores 置为 `-inf` 或很大的负数。

面试表达：causal attention 保证第 `t` 个位置只使用 `<= t` 的上下文，从而匹配自回归生成过程。

## Causal Mask

一句话定义：causal mask 是防止当前位置关注未来 token 的下三角 attention mask。

作用方式：在 softmax 前把未来位置的 attention scores 置为 `-inf`，使其 softmax 权重变为 0。

为什么需要它：保证 causal LM 学习 `P(x_t | x_{1:t-1})`，防止训练时信息泄漏。

常见形状：训练时常对应 `[T, T]` 下三角矩阵，批量和多头维度可广播。

工程坑：mask 方向写反会导致模型看不到历史或看到未来；padding mask 和 causal mask 需要正确组合；混合精度中使用极小负数时要避免数值溢出。

面试表达：causal mask 是自回归模型成立的必要约束，不是简单为了减少计算。

## Padding Mask

一句话定义：padding mask 是防止模型关注 padding token 的 attention mask。

为什么需要：batch 内样本长度不同，通常会 pad 到相同长度。padding token 不是有效文本，不能参与语义建模。

与 causal mask 的区别：padding mask 处理无效 token，causal mask 处理未来 token。decoder-only 训练中经常需要二者同时生效。

工程坑：左 padding 和右 padding 对 position id、KV Cache、生成对齐都有影响。很多 decoder-only 模型推理时更偏好左 padding 或需要框架特别处理。

面试表达：padding mask 解决的是 batch 对齐带来的无效 token 干扰问题。

## Causal LM

一句话定义：causal LM 是从左到右预测下一个 token 的语言模型。

训练目标：最大化 `P(x_1, x_2, ..., x_T) = Π_t P(x_t | x_{1:t-1})`，实现上常把输入右移一位作为 label。

典型模型：GPT 系列、LLaMA 系列、Qwen 系列等 decoder-only LLM。

特点：只能使用左侧上下文，天然适合续写、对话和代码生成。

局限：对纯理解类任务需要通过 prompt 转成生成任务；双向上下文建模不如 masked LM 直接。

面试表达：现代对话 LLM 大多是 causal LM，它们通过不断预测下一个 token 来完成生成。

## Masked LM

一句话定义：masked LM 是随机 mask 输入 token，并用双向上下文预测被 mask token 的语言模型。

典型模型：BERT。

特点：可以同时利用左右上下文，适合理解、分类、检索表示等任务。

与 causal LM 的区别：masked LM 不是按从左到右方式生成完整文本，因此不天然适合开放式自回归生成。

局限：预训练中使用 `[MASK]` token，而下游真实输入通常没有 `[MASK]`，存在一定预训练和下游分布差异。

面试表达：BERT 类 masked LM 更偏理解表示，GPT 类 causal LM 更偏生成，两者的 attention mask 和训练目标不同。

## Scaled Dot-Product Attention

一句话定义：scaled dot-product attention 是使用 `softmax(QK^T / sqrt(d_k))V` 计算注意力输出的方法。

为什么缩放：如果 `d_k` 很大，query 和 key 的点积方差会变大，softmax 容易饱和，导致梯度变小。除以 `sqrt(d_k)` 可以稳定分数尺度。

计算步骤：先算 `QK^T` 得到每个 query 对每个 key 的匹配分数；再加 mask；再 softmax；最后乘以 `V` 得到加权汇总结果。

复杂度：标准实现需要构造 `[T, T]` 分数矩阵，计算和显存随序列长度平方增长。

面试表达：缩放点积注意力的关键是用点积做匹配、用 softmax 做归一化、用 value 加权和做信息聚合。

## Full Attention

一句话定义：full attention 是每个 token 可以关注所有可见 token 的 attention。

优点：信息交互充分，表达力强，适合需要全局依赖的任务。

缺点：标准实现复杂度为 `O(T^2)`，长序列成本高。

在 decoder-only 中：由于 causal mask，当前位置可关注所有历史和当前位置，但不能关注未来。

面试表达：full attention 的优势是全局交互，代价是长度平方级成本。

## Local Attention / Sliding Window Attention

一句话定义：local attention 是每个 token 只关注局部窗口内 token 的 attention。

为什么提出：长序列中很多依赖是局部的，只计算附近窗口可以显著降低 attention 成本。

优点：降低计算和显存，适合长文本、语音、视频等长序列任务。

缺点：远距离依赖受限。如果没有全局 token、跨窗口连接或层间信息传递，远端信息很难直接交互。

面试表达：sliding window attention 用局部性换效率，适合长序列，但需要额外机制弥补全局信息不足。

## Sparse Attention

一句话定义：sparse attention 是只计算部分 token pair 注意力连接的 attention。

常见模式：local、global、block sparse、strided、random。

核心 trade-off：通过稀疏连接降低成本，但可能损失全局信息和实现通用性。

工程问题：理论复杂度降低不一定等于实际加速。GPU 上稀疏算子是否高效、是否支持训练和推理框架非常关键。

面试表达：sparse attention 的重点不是简单少算，而是设计一种既省计算又保留关键信息路径的连接模式。

## Linear Attention

一句话定义：linear attention 是通过特征映射或计算重排避免显式构造 `T*T` attention matrix 的方法。

目标：把复杂度从 `O(T^2)` 降到接近 `O(T)`。

核心思路：用核函数或特征映射近似 softmax attention，使 `softmax(QK^T)V` 可以被重排为先聚合 K/V，再与 Q 交互。

优点：长序列理论成本低。

局限：可能改变 attention 数学形式，质量、数值稳定性和硬件效率需要具体分析。很多线性 attention 在通用 LLM 上替代标准 attention 并不容易。

面试表达：linear attention 试图从算法形式上降低长度平方复杂度，但是否能保持 LLM 质量是关键问题。

## FlashAttention

一句话定义：FlashAttention 是通过 IO-aware 分块计算优化精确 attention 的高效实现。

核心思想：标准 attention 的瓶颈不只是 FLOPs，还有 HBM 显存读写。FlashAttention 通过 tiling、在线 softmax 和重计算策略，减少中间 attention matrix 的显存读写。

重要特点：它通常不改变 attention 的数学结果，而是改变计算实现方式。

优点：显著降低显存占用，提高训练和推理速度，尤其适合长序列和大 batch。

局限：依赖硬件、CUDA kernel、数据类型和框架支持；不同版本对 mask、dropout、GQA、变长序列的支持可能不同。

面试表达：FlashAttention 是 exact attention 的高效 kernel，不是稀疏 attention，也不是近似 attention。

## Attention Complexity

一句话定义：attention complexity 描述 attention 计算和显存如何随序列长度、head 数和维度增长。

标准成本：对于序列长度 `T`，标准 full attention 的 score matrix 是 `[T, T]`，所以计算和中间显存核心项通常是 `O(T^2)`。

训练影响：长序列训练时，attention 激活显存和计算成本快速上升，需要 checkpointing、FlashAttention、序列并行等技术。

推理影响：prefill 阶段需要处理整个 prompt，attention 成本高；decode 阶段每步只生成一个 token，但 KV Cache 随历史长度增长。

面试表达：长上下文的瓶颈分 prefill 和 decode 两部分，prefill 关注 attention 计算，decode 关注 KV Cache 和逐步访存。

## Multi-Head Attention / MHA

一句话定义：multi-head attention 是并行使用多个 attention head，在不同子空间中计算注意力的机制。

核心直觉：不同 head 可以学习不同类型的信息交互，例如局部语法、指代关系、位置模式、格式边界等。

形状：通常从 `[B, T, d_model]` 投影为 Q/K/V，再 reshape 到 `[B, num_heads, T, head_dim]`，其中 `d_model = num_heads * head_dim`。

为什么需要多头：单个 attention 分布可能表达能力有限，多头允许模型同时从多个关系视角聚合上下文信息。

局限：head 的功能不一定清晰可解释；head 数越多也不一定越好，还要考虑 head_dim、算力和 KV Cache。

面试表达：MHA 是把隐藏维度拆成多个子空间并行做 attention，再拼接回去，从而增强信息交互的多样性。

## Attention Head

一句话定义：attention head 是 multi-head attention 中的一组独立 Q/K/V attention 子空间。

组成：每个 head 有自己的 Q/K/V 投影子空间，输出后与其他 head 拼接，再经过输出投影。

可能功能：一些 head 可能偏位置，一些偏复制，一些偏语法关系，一些可能冗余。

剪枝问题：研究中可以剪掉部分 head，但生产模型不能简单假设某个 head 无用，需通过评估验证。

面试表达：head 是 attention 的并行子通道，提供多种匹配和聚合模式。

## Multi-Query Attention / MQA

一句话定义：multi-query attention 是多个 query heads 共享一组 key/value heads 的 attention 变体。

为什么提出：自回归推理的 decode 阶段需要保存历史 K/V。标准 MHA 每个 query head 都有自己的 K/V，KV Cache 很大。MQA 让所有 query heads 共享 K/V，大幅减少缓存。

核心价值：显著减少 KV Cache 显存和访存，提高长上下文推理效率。

代价：K/V 表达能力下降，可能影响模型质量，尤其是模型较小或训练不足时。

适用场景：推理效率要求高、上下文长、服务并发高的生成模型。

面试表达：MQA 是用共享 K/V 换推理效率，本质优化的是 KV Cache，而不是减少 query head 数。

## Grouped-Query Attention / GQA

一句话定义：grouped-query attention 是 query heads 分组共享 key/value heads 的 attention 变体。

位置关系：GQA 介于 MHA 和 MQA 之间。MHA 是每个 query head 有独立 K/V；MQA 是所有 query head 共享一组 K/V；GQA 是一组 query heads 共享一组 K/V。

核心价值：在 MHA 的质量和 MQA 的推理效率之间折中。

工程收益：减少 KV Cache 大小和 decode 访存压力，适合现代 LLM 在线推理。

常见追问：为什么 GQA 能加速生成？因为 decode 阶段每步要读取历史 K/V，K/V head 数减少后缓存读写减少。

面试表达：GQA 不是 attention 近似，而是改变 Q head 与 K/V head 的共享关系，在质量和推理成本之间折中。

## KV Cache

一句话定义：KV Cache 是自回归生成时缓存历史 token 的 key/value，避免重复计算历史部分。

为什么需要：生成第 `t+1` 个 token 时，历史 token 的 K/V 不会变。如果每步都重新计算全部历史，会造成大量重复计算。

缓存内容：每一层 attention 的历史 key 和 value。形状通常与 batch、layer、kv_head、sequence length、head_dim 相关。

收益：decode 阶段从重复处理整个前缀，变成只处理新 token 的 Q，并与缓存的 K/V 做 attention。

成本：KV Cache 显存随 batch size、层数、上下文长度、K/V head 数和 head_dim 线性增长。长上下文和高并发服务中，KV Cache 往往是主要瓶颈。

与 MQA/GQA 的关系：减少 K/V head 数可以显著降低 KV Cache 显存和访存。

工程坑：不同请求长度不一会导致 cache 管理复杂；beam search、speculative decoding、prefix caching、paged attention 都需要正确处理 cache 生命周期。

面试表达：KV Cache 是 LLM 推理加速的核心机制，但它把计算压力转化为显存和访存压力。

## Transformer Block

一句话定义：Transformer block 是 Transformer 中反复堆叠的基本模块，通常由 attention、MLP、normalization 和 residual connection 组成。

Decoder-only 常见形式：`x = x + Attention(Norm(x))`，然后 `x = x + MLP(Norm(x))`。这属于 Pre-LN 结构。

各部分作用：attention 负责 token 间信息交互，MLP 负责逐 token 的非线性变换和容量扩展，normalization 稳定训练，residual connection 保留原信息并改善梯度传播。

堆叠意义：单层只能做有限的信息变换，多层堆叠后模型可以逐步形成从局部模式到高级语义、推理和任务格式的表示。

工程注意：不同模型的 block 细节不同，例如 LayerNorm/RMSNorm、GELU/SwiGLU、MHA/GQA、是否使用 bias、RoPE 位置等。

面试表达：Transformer block 是 LLM 的基本积木，attention 负责混合 token，MLP 负责变换表示，残差和归一化保证深层可训练。

## MLP / FFN

一句话定义：MLP 或 FFN 是 Transformer block 中对每个 token 位置独立作用的前馈网络。

核心作用：增加非线性表达能力和模型容量。attention 混合不同 token 的信息，MLP 则在每个位置上对混合后的表示进行复杂变换。

常见结构：传统 FFN 是 `d_model -> d_ff -> d_model`，中间接激活函数。现代 LLM 常用 SwiGLU、GEGLU 等门控 FFN 变体。

参数占比：在很多 LLM 中，MLP 参数量占比很大，甚至超过 attention。因此不能把 Transformer 能力只归因于 attention。

工程注意：MLP 是训练和推理中的主要计算来源之一，量化、张量并行和算子融合都会重点优化它。

面试表达：attention 解决信息从哪里来，MLP 解决信息如何非线性加工和存储。

## Residual Connection

一句话定义：residual connection 是把子层输出加回输入的连接方式。

核心公式：`x = x + sublayer(x)`。

为什么需要：深层网络如果每层都完全重写表示，梯度传播和信息保留会很困难。残差连接让模型可以在原表示基础上学习增量修改。

作用：保留信息、改善梯度传播、支持深层训练、让子层可以学习接近恒等映射的变换。

工程注意：残差分支的数值尺度很重要。深层模型中初始化、normalization、residual scaling 都会影响训练稳定性。

面试表达：残差连接让深层 Transformer 更容易优化，本质是让每层学习对表示的增量更新。

## Pre-LN

一句话定义：Pre-LN 是在子层之前做 normalization 的 Transformer 结构。

形式：`x = x + sublayer(Norm(x))`。

特点：现代 LLM 常用，深层训练更稳定，梯度更容易沿残差路径传播。

优点：训练稳定性好，适合堆叠很多层。

潜在问题：某些情况下最终输出表示可能需要额外 final norm；不同结构在收敛速度和最终效果上仍需实验比较。

面试表达：Pre-LN 把归一化放在 attention/MLP 前面，是现代大模型更常用的稳定训练结构。

## Post-LN

一句话定义：Post-LN 是在残差相加之后做 normalization 的 Transformer 结构。

形式：`x = Norm(x + sublayer(x))`。

历史背景：原始 Transformer 使用 Post-LN。

特点：浅层或中等层数模型中可用，但很深模型训练时梯度稳定性通常不如 Pre-LN。

对比：Post-LN 每层输出被归一化，形式直观；Pre-LN 残差主路径更干净，深层优化更友好。

面试表达：Post-LN 是原始结构，Pre-LN 是现代 LLM 更常见的工程选择，主要差异在训练稳定性。

## Decoder-Only Transformer

一句话定义：decoder-only Transformer 是只使用带 causal mask 的 Transformer decoder block 进行自回归生成的架构。

典型模型：GPT、LLaMA、Qwen、Mistral 等主流对话 LLM。

训练方式：输入 token 序列，模型在每个位置预测下一个 token。训练时使用 causal mask，推理时逐 token 生成。

优点：结构统一，训练和推理目标一致，天然适合开放式生成、对话、代码、工具调用和指令跟随。

局限：只能从左到右使用上下文。对纯理解、双向编码、句向量检索等任务，可能不如专门 encoder 模型高效。

为什么成为主流：大规模预训练加上指令微调和对齐后，decoder-only 架构可以把大量 NLP 任务统一成文本生成问题，扩展性和工程生态最好。

面试表达：现代 LLM 主流是 decoder-only causal LM，因为它把预训练、对话和生成统一到 next token prediction 上。

## Encoder-Only Transformer

一句话定义：encoder-only Transformer 是使用双向 self-attention 编码输入序列的 Transformer 架构。

典型模型：BERT、RoBERTa、DeBERTa 等。

训练目标：常见是 masked language modeling，也可以结合 sentence order、对比学习等目标。

优点：每个位置可以同时看左右上下文，适合理解、分类、抽取、匹配、检索 embedding 等任务。

局限：不天然适合从左到右生成长文本；如果要生成，通常需要额外解码器或改造目标。

面试表达：encoder-only 更像强大的文本理解编码器，输出的是输入文本的上下文表示，而不是直接自回归生成器。

## Encoder-Decoder Transformer

一句话定义：encoder-decoder Transformer 是用 encoder 编码输入、decoder 通过 cross-attention 条件生成输出的架构。

典型模型：原始 Transformer、T5、BART 等。

结构：encoder 使用双向 self-attention 理解输入；decoder 使用 causal self-attention 生成输出，并通过 cross-attention 读取 encoder 表示。

适用任务：机器翻译、摘要、文本改写、结构化输入到文本输出等明确输入到输出转换任务。

优点：输入理解和输出生成分工清晰，适合 seq2seq 任务。

局限：结构比 decoder-only 更复杂，预训练和推理路径也更复杂；在通用对话 LLM 时代，很多任务被 decoder-only 统一处理。

面试表达：encoder-decoder 的核心是 encoder 负责读输入，decoder 负责写输出，中间通过 cross-attention 连接。

## Cross-Attention

一句话定义：cross-attention 是 decoder 的 query 去关注 encoder 输出的 key/value 的 attention。

与 self-attention 的区别：self-attention 的 Q/K/V 来自同一个序列；cross-attention 的 Q 来自 decoder 当前状态，K/V 来自 encoder 表示或外部条件。

典型场景：机器翻译中 decoder 生成目标语言时关注源语言句子；多模态模型中文本 decoder 关注图像 encoder 特征；RAG 或工具场景中也可把外部记忆作为 K/V 来源。

作用：把条件信息注入生成过程，让输出依赖输入或外部模态。

工程注意：cross-attention 的 mask、缓存和 batch 对齐比普通 self-attention 更复杂，尤其在多模态和变长输入中。

面试表达：cross-attention 是条件生成的桥梁，让生成端在每一步读取编码端或外部信息。

## 本章小结

Transformer 架构可以从四条线理解。

第一条线是输入表示：token id 先进入 embedding，再通过位置编码获得顺序信息。

第二条线是 attention：Q/K/V、mask、多头、MHA/MQA/GQA、KV Cache 共同决定信息如何在 token 之间流动。

第三条线是 block 结构：attention 负责混合上下文，MLP 负责非线性加工，residual 和 normalization 保证深层网络可训练。

第四条线是模型形态：decoder-only 适合自回归生成，encoder-only 适合理解表示，encoder-decoder 适合输入到输出转换。

面试中如果被问 Transformer，不要只背公式。更好的回答是先说明它解决 RNN 难并行和长距离依赖的问题，再讲 self-attention 的 Q/K/V 机制，随后补充 mask、位置编码、block 结构、复杂度和现代 LLM 中的 MQA/GQA、KV Cache、FlashAttention 等工程优化。

下一章，我们进入大语言模型基础。
