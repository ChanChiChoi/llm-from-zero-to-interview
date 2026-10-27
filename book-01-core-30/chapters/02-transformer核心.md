# 第二部分：Transformer 核心

## 本部分要回答的问题

一个语言模型怎样把一段文本变成“下一个 token 应该是什么”的分布？答案不是一组互不相干的模块名称，而是一条有输入、有状态、有约束的数据流：文本先被离散化，离散编号被查表变成向量，注意力让每个位置读取允许看到的上下文，多个子层组成可堆叠的 Transformer Block，最后语言模型头把上下文表示映射到词表上的 logits。

本部分沿 decoder-only 文本模型这条主线讲清基础机制。它不试图收录所有 Transformer 变体，也不把推理框架、训练系统和长上下文评测展开成平行百科；这些专题分别由第二册、第五册、第六册和第二十四册接手。

| 讲次 | 本讲回答的问题 | 留给下一讲的结果 |
| --- | --- | --- |
| 第 6 讲 | 文本如何变成模型可读的 token ID？ | 一个有序的整数序列 |
| 第 7 讲 | 整数如何变成向量，模型怎样知道位置？ | 带有位置处理的初始表示 |
| 第 8 讲 | 一个位置怎样从其他位置读取信息？ | Q、K、V 的信息路由直觉 |
| 第 9 讲 | 注意力怎样从直觉变成计算？ | 可逐项检查的分数、权重与输出 |
| 第 10 讲 | 多个注意力头怎样组织，K/V 共享会改变什么？ | 多头张量与 KV Cache 的账本 |
| 第 11 讲 | 模型如何只预测下一个 token？ | 训练标签、因果可见性和生成循环 |
| 第 12 讲 | Attention、MLP 如何成为一个 Block？ | 可以重复堆叠的层结构 |
| 第 13 讲 | 深层网络怎样保持信息与梯度通路？ | 对残差、Pre-LN、LayerNorm 和 RMSNorm 的理解 |
| 第 14 讲 | 位置信息怎样进入 Q/K，长上下文又受什么限制？ | RoPE 的机制及其能力边界 |
| 第 15 讲 | 怎样把这些部件接成一个最小 GPT？ | 一次前向、训练和逐 token 生成 |

后文统一使用以下 shape 账本：B 表示 batch 大小，T 表示每条序列的 token 数，D 表示隐藏维度，H 表示 Query 头数，H_kv 表示 Key/Value 头数，d_h=D/H 表示每个头的维度，V 表示词表大小。贯穿例子使用示意性序列“我 喜欢 模型。”，假设 tokenizer 将它映射为 [12, 44, 9, 31]。这些编号只用于追踪 shape，不代表任何真实 tokenizer 词表。讲训练数据时取前三个 ID 作为 input，后三个 ID 作为右移 target，因此示例 shape 账本中的 T=3。

## 从文本到下一个 token 的总路径

下图把十讲放回一次 decoder-only 语言模型计算中。训练和生成共享模型前向路径，但目标不同：训练把预测与真实后续 token 比较并更新参数；生成取最后位置的分布，选择一个 token 后再追加到上下文。

~~~mermaid
flowchart LR
    A[文本] --> B[Tokenizer]
    B --> C[Token IDs B×T]
    C --> D[Token Embedding B×T×D]
    D --> E[Transformer Blocks：位置机制、Attention 与 MLP]
    E --> F[Logits B×T×V]
    F --> G[下一 token 概率]
    Y[右移后的目标 token] --> H[交叉熵 loss]
    F --> H
    H --> I[梯度与参数更新]
    I -.更新 embedding.-> D
    I -.更新 block 参数.-> E
    I -.更新输出层.-> F
    G --> J[选择一个 token]
    J --> K[追加到上下文]
    K --> D
~~~

Shape 是读代码时的导航工具。输入 [B,T] 经 embedding 后成为 [B,T,D]；注意力内部暂时拆成多个头，输出再回到 [B,T,D]；语言模型头最后产生 [B,T,V]。位置可以加在 embedding 上，也可以作用于 Q/K 或 attention score，因此图中把位置机制放在 Transformer Blocks 内说明。后续残差分支都必须回到相同的 [B,T,D]，否则不能相加。

## 第 6 讲：Tokenization——文本怎样变成离散编号

### 模型接收的是编号，不是字符串

神经网络不能直接把“我喜欢模型”当作文本对象处理。Tokenizer 按固定规则把文本切成 token，再把每个 token 映射到词表中的整数 ID：

~~~text
文本 -> token 序列 -> token ID 序列
~~~

Token 可以是词、子词、字符、字节片段或专用控制符。ID 只是词表项的索引：12 不比 4 更大、更接近“喜欢”，也没有天然的语义顺序。真正的表示要到下一讲从 embedding 表中查出向量后才出现。

### 为什么常见做法是子词或字节级切分

| 切分粒度 | 好处 | 代价 |
| --- | --- | --- |
| 词级 | 常见词直观、序列较短 | 词表膨胀；新词、拼写变化和无空格语言难处理 |
| 字符级 | 词表小，组合能力强 | 序列变长，模型需要更多步组合词义 |
| 子词 | 在词表规模与序列长度之间折中 | 切分规则与训练词表绑定 |
| 字节级 | 可表示任意文本字节，减少未知符号 | 序列长度和 Unicode 处理成本可能增加 |

BPE 的基本思路是在训练语料的符号序列中反复合并高频相邻单元。例如仅为解释 merge 过程，若玩具序列是 a b a b，频率最高的相邻对是 a b，合并后得到 ab ab。真实 tokenizer 还涉及预切分、边界标记、字节映射、词表停止条件和编码规则；这个两步例子不等于任何模型的实际词表构造流程。[Sennrich 等人的 BPE 论文](https://arxiv.org/abs/1508.07909)讨论了用子词单元处理罕见词的动机。

WordPiece 和 BPE 都会得到子词词表，但不能把它们说成同一种 merge 算法。SentencePiece 是直接面向原始文本训练和应用子词模型的工具，支持的模型类型不等于“SentencePiece 这一种切分算法”。[BERT 论文](https://arxiv.org/abs/1810.04805)是 WordPiece 的代表性实例；[SentencePiece 论文](https://arxiv.org/abs/1808.06226)介绍了其原始文本处理和子词建模方式。

### Tokenizer 是模型输入契约的一部分

Tokenizer 会影响一段文本占多少 token、固定上下文窗口能容纳多少文本、训练样本如何构造，以及特殊 token 如何表示。这个影响必须针对具体 tokenizer 和语料测量，不能直接总结为“某种语言总是更费 token”。

训练和推理应使用兼容的 tokenizer、词表、特殊 token 配置和模型 embedding。若 tokenizer 新增或重排词表项，模型的 embedding 行也必须按约定同步；仅仅重新编号会改变每个 ID 对应的向量。评估 token 长度或 token 成本时，也必须记录使用的 tokenizer。

**交接：**现在得到的是一个有序 ID 序列。下一讲把它变为 [B,T,D] 的向量序列，并处理“模型怎样区分位置”。

## 第 7 讲：Embedding 与位置信息——编号怎样变成表示

### 查表把离散 ID 变成向量

Embedding 是一个可训练矩阵 $E \in \mathbb{R}^{V \times D}$。输入 ID 为 i 时，模型取出矩阵第 i 行作为该 token 的初始向量。对一批长度为 T 的序列查表后，shape 从 [B,T] 变成 [B,T,D]。

~~~math
X_{b,t,:} = E[\mathrm{token\_id}_{b,t}]
~~~

编号本身没有语义距离：ID 12 和 13 不代表两个 token 的意思相近。训练通过 loss 和梯度调整 embedding 行，使它们参与后续计算时有用。若输出层与输入 embedding 权重共享，词表 logits 可以由 $X E^\top$ 得到；权重共享是一种架构选择，不是所有模型都必须采用的规则。

### 为什么还需要位置

Self-Attention 依据向量内容计算 token 间关系。若完全不提供位置或顺序约束，模型没有足够信息区分相同 token 的不同排列。不同位置机制把位置信息放在不同位置：

| 位置机制 | 位置信息进入哪里 | 本章需要记住的区别 |
| --- | --- | --- |
| Learned absolute embedding | 加到 token embedding 上 | 位置向量由训练学习，表长与最大位置有关 |
| Sinusoidal absolute encoding | 加到 token embedding 上 | 用不同频率的正弦/余弦构造位置向量 |
| Relative bias / ALiBi | 加到 attention score 上 | 按相对距离调整注意力分数 |
| RoPE | 旋转 Query 和 Key | 让 Q/K 点积随相对位置变化；第 14 讲再推导 |

原始 Transformer 使用正弦位置编码；后续模型采用了多种不同方案。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)是原始结构与 scaled dot-product attention 的来源。此处只需要知道位置机制放在计算路径的哪里；RoPE 的公式要等 Q/K 定义清楚后再讲。

ALiBi 是一种在 attention score 中按距离加入线性 bias 的方案；此处只作为位置机制分类，具体效果和长度外推结论应限于论文设置。[Train Short, Test Long](https://arxiv.org/abs/2108.12409)介绍了该方法。

对于常见的绝对位置相加方式，令位置向量为 P，则初始状态是 $X_0 = E[\mathrm{ids}] + P$，shape 仍为 [B,T,D]。注意力之后，位置和上下文信息会逐层混合，不能把最终 hidden state 继续理解成静态的“词向量”。

**交接：**现在每个位置有一个向量，但还没有从同一句中的其他位置汇入信息。第 8 讲提出这种信息路由的直觉。

## 第 8 讲：Self-Attention 直觉——当前位置怎样读取上下文

### 表示需要依赖上下文

同一个 token 在不同上下文中的作用可能不同。比如“苹果”可以指水果，也可以指公司；单看 token 本身无法确定当前语境。模型需要根据当前序列，把其他位置的信息有选择地汇入当前位置表示。

Self-Attention 的每个位置都会从同一序列的表示中生成三种向量：

- **Query（查询）**：当前位置用来计算“我和哪些位置相关”的向量；
- **Key（键）**：每个位置用来参与匹配的向量；
- **Value（值）**：匹配后实际被加权汇入的信息向量。

这只是对学习到的投影功能的直觉，不意味着模型内部显式执行人类语言中的“提问、查找、取答案”。

### 从匹配到加权汇总

对某个 Query，模型计算它与所有可见 Key 的相似分数，把分数归一化成权重，再按这些权重混合 Value。权重高的位置对当前输出贡献更大。计算过程稍后会写成矩阵公式。

“Attention 权重”说明的是模型在这一层如何分配 Value 的加权系数，不能单独当作完整的因果解释或特征归因。网络有多层、多个头和后续非线性，输出由整条计算路径共同决定。

### 为什么它适合并行，也为什么代价随长度上升

对一条长度为 T 的序列，所有 Query、Key、Value 投影和注意力矩阵都可以通过矩阵运算并行计算，不需要像标准 RNN 那样必须先完成第 t-1 步再算第 t 步。但每个位置都要和许多其他位置比较，完整注意力会产生与 $T^2$ 成比例的分数项。并行计算并不等于长序列没有成本。

Self-Attention 也不应简单等同于“关键词搜索”：Key/Query 的匹配空间是模型学习出来的，Value 经过多层投影，最终信息影响不止来自某一张注意力图。它解决的是可学习的信息混合问题，代价和可见范围由具体结构决定。

**交接：**直觉已经建立。下一讲固定符号，手算一次分数、softmax 和 Value 汇总，并把“未来不可见”写成 mask。

## 第 9 讲：Scaled Dot-Product Attention——从公式到数值

### 张量怎样变成 Query、Key、Value

令输入表示 $X \in \mathbb{R}^{B \times T \times D}$。线性投影生成 Q、K、V，再将隐藏维度拆成头：

~~~math
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
~~~

先以标准 MHA 为例，三个投影矩阵的输出宽度都是 D；将 D 拆成 H 个头，每个头宽度为 $d_h=D/H$。GQA/MQA 中 K/V 投影的输出宽度会按 KV 头数缩小。

| 张量 | Shape | 每一维表示什么 |
| --- | --- | --- |
| Q、K、V（拆头前） | [B,T,D] | 每个 token 的投影表示 |
| Q、K、V（拆头后） | [B,H,T,d_h] | 每个头在每个位置的向量 |
| QK 转置 | [B,H,T,T] | 每个 Query 对每个 Key 的分数 |
| Softmax 权重 | [B,H,T,T] | 每个 Query 在可见位置上的归一化权重 |
| 权重乘 V | [B,H,T,d_h] | 每个头聚合后的上下文表示 |

Query 和 Key 在最后一维做点积，得到 Query-Key 相关分数。缩放因子来自一个简化假设：若各维 Q、K 独立、均值为 0、方差为 1，则点积是 $d_h$ 个乘积之和，方差为 $d_h$；除以 $\sqrt{d_h}$ 后，分数方差约回到 1。这样可减轻维度增大时 softmax 过早变尖锐的问题。真实网络中的分布未必满足这些假设，但这解释了缩放项的设计动机。

~~~math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}} + M
\right)V
~~~

这里 M 是可选的加性 mask 或 bias。Softmax 沿 Key 位置这一维逐行归一化。对不允许看到的位置填入负无穷，softmax 后其权重为 0；若没有可见位置的某一行被全部屏蔽，softmax 会遇到未定义的数值行为，因此 padding、变长 batch 和 causal mask 的组合必须按具体实现处理。

### 手算一个 Query 的输出

取一个 Query $q=(1,0)$，两个 Key $k_1=(1,0)$、$k_2=(0,1)$，两个 Value $v_1=(2,0)$、$v_2=(0,4)$，令 $d_h=2$。

1. 原始点积是 [1, 0]。
2. 缩放后分数是 [1/√2, 0] ≈ [0.7071, 0]。
3. Softmax 权重约为 [0.6698, 0.3302]。
4. 加权 Value 得到 $0.6698(2,0)+0.3302(0,4)\approx(1.3396,1.3208)$。

每一步都能定位输出变化来自哪里：点积决定相对分数，缩放影响 softmax 的尖锐程度，Value 决定最终被混合的内容。这个小例子只演示定义，不是模型测量。

### 因果 mask 怎样改变可见位置

自回归语言模型中，位置 t 预测后续 token 时不能读到未来位置。对长度 4 的序列，Query 行只能看同位置和之前的位置：

| Query 位置 | Key 1 | Key 2 | Key 3 | Key 4 |
| --- | --- | --- | --- | --- |
| 1 | 可见 | 屏蔽 | 屏蔽 | 屏蔽 |
| 2 | 可见 | 可见 | 屏蔽 | 屏蔽 |
| 3 | 可见 | 可见 | 可见 | 屏蔽 |
| 4 | 可见 | 可见 | 可见 | 可见 |

用 1-based 下标，causal mask 的定义为：

~~~math
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
~~~

将 mask 加到分数上，再做 softmax，未来位置就不会向当前输出贡献 Value。第 11 讲会解释这种可见性如何和训练标签、逐 token 生成配合。

### 计算量与实现优化不是同一问题

标准全量注意力的主要计算量约为 $O(BT^2D)$，朴素实现还会显式保存 $O(BHT^2)$ 的注意力分数或权重。随着序列长度增长，计算和中间状态都会变贵。

| 方法 | 改变什么 | 本章范围内的结论 |
| --- | --- | --- |
| FlashAttention | 分块计算与数据搬运方式 | 保留精确 attention 数学目标；减少对高带宽显存的中间读写，不是稀疏近似 |
| Local / Sparse Attention | 哪些位置彼此可见 | 可减少交互数，但改变了注意力连接模式 |
| Linear Attention 类方法 | 对注意力计算作重构或近似 | 需单独看假设、精度和任务结果，不能只凭名字断言等价 |

[FlashAttention 论文](https://arxiv.org/abs/2205.14135)提出 IO-aware 的精确注意力实现。其实际速度和内存收益仍依赖硬件、张量形状、dtype 和 kernel。稀疏或线性注意力的结构选择属于 Transformer 进阶主题，参见[第二册 Transformer 架构进阶](../../book-02-advanced-100/chapters/02-transformer架构进阶.md)。

**交接：**单个头完成了“算分数—归一化—混合 Value”。下一讲把隐藏维度拆成多个头，再看减少 Key/Value 头数为什么主要影响推理缓存。

## 第 10 讲：Multi-Head Attention——多个子空间与共享 KV

### 多头做的是并行投影，不是复制同一个判断

Multi-Head Attention（MHA）用多组投影，让不同头在各自的子空间中计算注意力。每头输出 shape [B,T,d_h]；沿最后一维拼接回 [B,T,D]，再经输出投影得到模块输出。注意力权重本身并不自动对应一个可命名的“语法头”或“事实头”。

~~~math
\mathrm{head}_i = \mathrm{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}),
\qquad
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W_O
~~~

当隐藏宽度 D 固定时，仅改变 H 并相应改变每头宽度，并不会自动增加 Q/K/V/O 的投影矩阵参数量；参数规模主要由投影的输入、输出宽度决定。实际实现还可能包含 bias 或不同的投影布局。

~~~mermaid
flowchart LR
    X[输入表示 B×T×D] --> Q[Query heads Hq]
    X --> K[Key heads Hkv]
    X --> V[Value heads Hkv]
    Q --> A[每个 Query 头独立计算注意力]
    K --> A
    V --> A
    A --> C[拼接各头输出]
    C --> O[输出投影 Wo]
~~~

如果有 $H_q$ 个 Query 头，不一定要有同样多的 Key/Value 头。MHA、GQA 和 MQA 的关系是 KV 头数量不同：

| 结构 | Query 头数 | Key/Value 头数 | 结构含义 |
| --- | --- | --- | --- |
| MHA | $H_q$ | $H_q$ | 每个 Query 头有自己的 K/V 头 |
| GQA | $H_q$ | $1 < H_{kv} < H_q$ | 一组 Query 头共享一组 K/V |
| MQA | $H_q$ | 1 | 所有 Query 头共享一组 K/V |

GQA 是 MHA 与 MQA 之间的结构折中。它减少 KV 表示数量，同时保留多组 Query；具体质量、吞吐和显存收益依赖模型、训练及 serving 实现。[GQA 论文](https://arxiv.org/abs/2305.13245)报告了其研究配置中的质量和速度取舍，不能推出所有模型都获得相同收益。

### KV Cache 为什么受 KV 头数影响

逐 token 生成时，早先位置的 Key 和 Value 可以缓存起来，后续步骤复用。单条序列的原始 KV Cache 载荷可按下式估算：

~~~math
\mathrm{KV\ bytes}
=
2 \times L \times T \times H_{kv} \times d_h \times b
~~~

其中 L 是层数，T 是缓存长度，前面的 2 对应 K 与 V，b 是每个元素的字节数。以 L=32、T=8192、d_h=128、b=2 字节为例：

| 结构 | H_kv | 单序列原始 KV 载荷 |
| --- | ---: | ---: |
| MHA，32 个头 | 32 | 4 GiB |
| GQA，8 个 KV 头 | 8 | 1 GiB |
| MQA，1 个 KV 头 | 1 | 128 MiB |

这是公式估算，不含 batch、对齐、量化元数据、分页开销、workspace 或框架额外状态。它只说明在这些固定假设下，KV 头数与缓存载荷成正比。它不代表实际吞吐提升，也不表示 attention 的全部计算量按相同比例下降。更完整的 Cache 布局和生命周期见[第六册 KV Cache 与内存管理](../../book-06-llm-deployment/chapters/03-kv-cache与内存管理.md)和[第二十四册最小 KV Cache 实现](../../book-24-llm-inference-engine/chapters/09-实现kv-cache的最小版本.md)。

**交接：**多个头说明怎样组织一次上下文混合；下一讲把因果可见性放回语言模型目标中，明确训练和生成各自在做什么。

## 第 11 讲：Causal Mask 与自回归生成——训练时并行，生成时逐步

### 标签右移与因果可见性必须同时成立

给定 token 序列 [12, 44, 9, 31]，训练窗口可写成：

~~~text
input:  [12, 44, 9]
target: [44,  9, 31]
~~~

位置 i 的 logits 预测下一个 token $x_{i+1}$，但位置 i 的表示只能读取 $x_1,\ldots,x_i$。目标右移定义“预测谁”，causal mask 定义“预测时能看见什么”；这两个约束各司其职，不能互相替代。

语言模型的 token 级交叉熵目标可以写成：

~~~math
\mathcal{L}
=
-\frac{1}{T}
\sum_{i=1}^{T}
\log p_\theta(x_{i+1}\mid x_1,\ldots,x_i)
~~~

首个预测位置常通过 BOS 等起始约定处理。训练时一次前向可以并行计算所有位置的 logits，因为 mask 已让每个位置只能看到允许的前缀；并行计算没有泄漏未来目标。

### Teacher Forcing 不等于允许看到未来

Teacher Forcing 是训练时把真实历史 token 作为模型输入。对位置 i，它拿到的是此前的真实前缀，而不是此前模型采样出来的 token；causal mask 仍然限制该位置看不到未来 token。推理时没有真实 target，模型必须从当前分布选择一个 token，把它追加到上下文，再继续计算。

~~~mermaid
sequenceDiagram
    participant X as 已有上下文
    participant M as Causal Transformer
    participant P as 下一个 token 分布
    X->>M: 一次处理提示词各位置
    M->>P: 取最后位置 logits
    P->>X: 选择并追加一个 token
    X->>M: 带新 token 的上下文再次前向
~~~

提示词阶段通常称为 Prefill，可以对提示词位置并行计算；之后每一步 Decode 只需要产生下一个 token。KV Cache 复用历史 K/V，避免每次都为整段历史重新计算，但它改变的是计算复用方式，不改变“只能基于已有前缀预测下一个 token”的因果协议。具体服务阶段与缓存管理见[第六册推理基础](../../book-06-llm-deployment/chapters/02-推理基础.md)。

Causal LM 和 Masked LM 的目标也不同：Causal LM 按顺序预测后续 token，Masked LM 预测输入中被遮盖的 token，允许使用两侧上下文；BERT 是 masked language modeling 的代表性实例。[BERT 论文](https://arxiv.org/abs/1810.04805)介绍了其训练目标。两种训练目标不能只根据是否出现“mask”这个词混为一谈。

**交接：**训练目标规定了模型沿着前缀预测；现在需要一层可以重复堆叠的模块来不断更新上下文表示，这就是 Transformer Block。

## 第 12 讲：Transformer Block——Attention 与 MLP 如何组合

### 一个 Block 有两种互补计算

Attention 在 token 之间混合信息；MLP 对每个 token 的特征维度做非线性变换。每个 block 先后执行这两种计算，并通过残差支路保持输入输出宽度一致。多层 block 堆叠后，每个位置的表示会逐层融合更复杂的上下文特征。

下面以常见的 Pre-LN 写法表示一个 decoder-only block：

~~~math
x' = x + \mathrm{Attention}(\mathrm{Norm}_1(x))
~~~

~~~math
x_{\mathrm{next}}
=
x' + \mathrm{MLP}(\mathrm{Norm}_2(x'))
~~~

~~~mermaid
flowchart TD
    X[输入 x B×T×D] --> N1[Norm 1]
    N1 --> A[Causal Self-Attention]
    A --> R1[与 x 做残差相加]
    X --> R1
    R1 --> N2[Norm 2]
    N2 --> F[逐 token MLP]
    F --> R2[与前一状态做残差相加]
    R1 --> R2
    R2 --> Y[输出 B×T×D]
~~~

逐 token MLP 的典型形式是先把宽度扩展到 $D_{ff}$，经过激活，再投影回 D。它不在 token 轴上混合位置，所以输出仍为 [B,T,D]。原始 Transformer 论文使用了特定的前馈层和 encoder-decoder 结构；现代 decoder-only 模型在激活函数、归一化位置、bias、dropout 和注意力实现上会有不同配置。[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)不应被当成每个现代 GPT 的逐项规格。

Decoder-only block 使用因果 self-attention；经典 encoder-decoder Transformer 的 decoder 还可以包含读取 encoder 输出的 cross-attention，GPT 类语言模型通常没有这条 encoder 输出支路。最终的 Transformer 堆叠之外，模型通常还有 final normalization 与语言模型头：从 [B,T,D] 映射为 [B,T,V]。

**交接：**Block 的结构已经闭合，但“残差为什么有用、Norm 具体归一化什么、Pre-LN 和 Post-LN 差在哪里”仍需要解释。第 13 讲只深入这些稳定性问题，不再重复 Block 结构。

## 第 13 讲：LayerNorm、RMSNorm 与残差——让深层计算保持可训练

### 残差支路保留直接的信息通路

残差块输出 $x+F(x)$。若 $F(x)$ 暂时接近零，信息仍可沿 x 这条直接支路传递；求梯度时也多出一条直接路径。这有助于深层优化，但不保证任意深度、初始化或学习率下训练都稳定。

~~~math
\frac{\partial (x+F(x))}{\partial x}
=
I + \frac{\partial F(x)}{\partial x}
~~~

Block 采用残差后仍需要控制每层数值尺度。Normalization 会在指定维度上重标定表示，帮助训练过程管理激活统计；它不是“让所有 token 变得一样”，也不会替代良好的数据、初始化和优化设置。

### LayerNorm 与 RMSNorm 的具体区别

LayerNorm 对每个 token 的隐藏维度 D 计算均值和方差，再标准化并应用可学习的缩放与偏移：

~~~math
\mu = \frac{1}{D}\sum_{j=1}^{D}x_j,
\qquad
\sigma^2 = \frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2
~~~

~~~math
\mathrm{LayerNorm}(x)_j
=
\gamma_j
\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta_j
~~~

RMSNorm 不减均值，而是按均方根缩放，再应用可学习缩放参数：

~~~math
\mathrm{RMSNorm}(x)_j
=
\gamma_j
\frac{x_j}{\sqrt{\frac{1}{D}\sum_{k=1}^{D}x_k^2+\epsilon}}
~~~

以 x=[1,2,3,10] 且暂不应用可学习参数为例，LayerNorm 的均值为 4，标准差约为 3.5355，结果约为 [-0.8485,-0.5657,-0.2828,1.6971]；RMSNorm 的分母约为 5.3385，结果约为 [0.1873,0.3747,0.5620,1.8734]。差别来自是否中心化，而不是两种公式仅有名称不同。[RMSNorm 论文](https://arxiv.org/abs/1910.07467)提出了省略均值项的归一化形式及其实验；不应据此宣称 RMSNorm 在所有模型上都优于 LayerNorm。

### Pre-LN 和 Post-LN 改变了计算顺序

~~~math
\mathrm{Pre\text{-}LN}: \quad y = x + F(\mathrm{Norm}(x))
~~~

~~~math
\mathrm{Post\text{-}LN}: \quad y = \mathrm{Norm}(x + F(x))
~~~

两种结构把归一化放在残差分支的不同位置，梯度路径和训练行为因此不同。Pre-LN 常被现代 decoder-only 模型采用，但仅凭这一个差异无法判断训练是否稳定；初始化、深度、学习率、归一化变体和具体实现都可能相关。[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)分析了其研究设置下 Pre-LN 与 Post-LN 的梯度路径差异。

**交接：**深层 block 如何保持数值和梯度路径已经说明。接下来回到 Attention 中的位置：RoPE 并不把位置向量加到输入，而是旋转 Q/K。

## 第 14 讲：RoPE 与长上下文——位置关系不等于长度能力

### RoPE 怎样让 Q/K 点积感知相对位移

在二维子空间中，位置 m 对向量的旋转矩阵为：

~~~math
R(m\theta)
=
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
~~~

RoPE 对位置 m 的 Query 和位置 n 的 Key 分别旋转。它们的点积可以改写为：

~~~math
(R(m\theta)q)^\top(R(n\theta)k)
=
q^\top R((n-m)\theta)k
~~~

因此点积中出现的是相对位移 n-m。实际 RoPE 会把不同维度对分配给不同频率 $\theta_i$，并在每层 Q/K 的对应维度对上执行旋转；V 不做这项位置旋转。[RoFormer 论文](https://arxiv.org/abs/2104.09864)给出了 RoPE 的构造和相对位置性质。

若取二维单位向量 q=k=(1,0)，每个位置步长角为 0.25 弧度，旋转后的点积为 $\cos((n-m)\times0.25)$。不同绝对位置但相同相对距离会得到相同的这一项。这只展示公式性质，不代表完整模型注意力分布，也不意味着把模型位置无限延长后性能不变。

### 位置表示、最大窗口与有效使用能力是不同问题

位置机制解决“怎样把顺序和距离表示进模型”；上下文窗口表示模型一次可以处理多少 token；有效长上下文能力还取决于训练长度与数据、位置处理、注意力计算、模型是否能找到相关证据，以及任务本身。

| 扩展方向 | 主要处理的问题 | 仍需单独验证什么 |
| --- | --- | --- |
| RoPE scaling 等位置变换 | 调整超出训练位置范围时的位置映射 | 短上下文回归、远近位置、不同任务效果 |
| FlashAttention 等 kernel | 减少注意力中间数据搬运开销 | 实际硬件、shape、dtype 下的吞吐和内存 |
| Local / Sparse Attention | 减少部分 token 间交互 | 被移除的长距离连接是否影响目标任务 |
| RAG 或外部检索 | 不把所有资料都塞进一次上下文 | 检索召回、证据定位和答案使用是否正确 |

长上下文不是某个位置编码单独带来的属性。模型可能接受很长的输入，却对其中间区域或特定任务表现不稳；[Lost in the Middle](https://arxiv.org/abs/2307.03172)在其研究设置中观察到证据位置会影响长上下文任务表现。它是评测设计应关注的一类现象，不是所有模型和任务上的固定曲线。更深的位置扩展、长上下文 RAG 与评估见[第二册相关章节](../../book-02-advanced-100/chapters/07-长上下文rag与agent.md)。

**交接：**本章已分别解释输入、上下文混合、因果训练、深层 block 和位置。最后一讲把这些机制放回同一个小模型，并沿实际 shape 走一遍。

## 第 15 讲：从零实现一个小 GPT——把主线接成程序

### 代码要验证什么

下面给出一份完整的字符级 decoder-only GPT 教学脚本。它把字符串映射为字符 ID，用可学习绝对位置 embedding、Pre-LN Transformer Block、因果 self-attention 和交叉熵训练，再逐 token 采样生成。

这段代码的目标是让输入、mask、残差和 logits 的 shape 可检查。它不实现 RoPE、GQA、KV Cache、混合精度、FlashAttention、停止 token 或分布式训练，因此不是生产 GPT。运行环境以仓库锁定的 Python 3.13+ 与 PyTorch 2.12.0 为准；脚本没有依赖预训练模型或外部数据，代码中的形状由构造参数决定。正文给出代码预期观察，不声称本轮已运行。

### 一次前向的 shape 路径

| 阶段 | Shape |
| --- | --- |
| 字符 ID 输入 | [B,T] |
| Token 与位置 embedding 相加 | [B,T,D] |
| 单头 Q/K/V | [B,H,T,d_h] |
| 注意力分数和权重 | [B,H,T,T] |
| 每个 block 的输出 | [B,T,D] |
| 语言模型头 logits | [B,T,V] |
| 训练目标 | [B,T] |

### 完整教学脚本

> **代码状态：静态审阅，未运行。**

~~~python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal", causal[None, None, :, :])

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(channels, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        allowed = self.causal[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.norm2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=2, n_head=2, n_embd=32):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.position_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.norm_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        if seq_len == 0 or seq_len > self.block_size:
            raise ValueError("sequence length must be in [1, block_size]")
        if idx.min() < 0 or idx.max() >= self.vocab_size:
            raise ValueError("token ID outside vocabulary")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_emb(idx) + self.position_emb(positions)[None, :, :]

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


torch.manual_seed(0)

text = "hello transformer\n" * 64
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

block_size = 16
batch_size = 8


def get_batch():
    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


model = MiniGPT(vocab_size=len(chars), block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(100):
    x, y = get_batch()
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("logits shape:", tuple(logits.shape))
print("loss is finite:", bool(torch.isfinite(loss)))

model.eval()
prompt = torch.tensor([[stoi["h"]]], dtype=torch.long)
generated = model.generate(prompt, max_new_tokens=40)[0].tolist()
print("generated:", "".join(itos[token_id] for token_id in generated))
~~~

按构造参数，前向 logits 的 shape 应为 [8,16,V]，loss 为标量，生成序列长度应为 41；loss 的具体数值和生成文本取决于随机初始化与采样过程，不能当作固定预期。脚本打印有限性检查结果供运行时观察。训练样本的 x 与 y 右移一位；因果 mask 让每个位置只读当前前缀；生成时每次只取最后位置的 logits。

### 沿着不变量检查小模型

| 若观察到的问题 | 先检查的约束 |
| --- | --- |
| loss 形状异常或无法计算 | logits 是 [B,T,V]，target 是 [B,T]，两者位置按 next-token 目标对齐 |
| 未来 token 影响较早位置 | causal mask 的方向、切片和 softmax 前屏蔽 |
| 残差相加报维度错误 | Attention 与 MLP 输出是否都回到 D |
| 输入长度超过支持范围 | position embedding 和 causal mask 是否覆盖当前 T |
| loss 下降但生成不合理 | 数据量、训练步数、字符级粒度、采样和任务难度 |

这段示例每一步重新处理截断后的上下文，没有 KV Cache；字符级 tokenization 也比常见子词词表产生更长序列。它适合验证结构，不代表真实模型质量或推理性能。更多训练实现和数据处理见[第五册模型架构与初始化](../../book-05-llm-training/chapters/04-模型架构与初始化.md)。

## 本部分收束：能从输入一路解释到 logits

读完后应能不看正文画出一条完整路径，并为每一步说出它解决的问题：

1. Tokenizer 把文本变成离散 ID；词表和特殊 token 是模型输入契约的一部分。
2. Embedding 把 ID 变为 [B,T,D]；位置机制把顺序或距离纳入计算。
3. Attention 通过 Q/K 分数、缩放、softmax 和 Value 汇总上下文；causal mask 限制未来可见性。
4. 多头提供多个投影子空间；KV 头数改变缓存载荷，但不等于全部计算成本。
5. Attention、MLP、Norm 和残差组成可堆叠 Block；最后映射为 [B,T,V] logits。
6. 训练把 logits 与右移目标比较并更新参数；生成逐次选择 token 并追加到上下文。

若一个模型在训练 loss、生成质量或推理内存上出现问题，可以先沿这条路径定位到具体接口：ID 与 embedding 是否一致、mask 是否正确、shape 是否守恒、位置是否在支持范围、KV 头与缓存预算是否符合目标。更深入的训练目标交给[第三部分](03-训练对齐推理.md)，Serving 和 Cache 交给第六册与第二十四册，长上下文和架构变体交给第二册。

## 资料入口与证据边界

- Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)：scaled dot-product attention、多头结构与原始 Transformer 设计。
- Sennrich 等，[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词单元与罕见词处理动机。
- Kudo 与 Richardson，[SentencePiece](https://arxiv.org/abs/1808.06226)：原始文本上的子词模型工具与处理方式。
- Su 等，[RoFormer](https://arxiv.org/abs/2104.09864)：RoPE 构造和相对位置性质。
- Zhang 与 Sennrich，[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)：RMSNorm 形式与论文实验。
- Dao 等，[FlashAttention](https://arxiv.org/abs/2205.14135)：IO-aware 精确注意力实现。
- Ainslie 等，[GQA](https://arxiv.org/abs/2305.13245)：Grouped-Query Attention 结构与论文配置下的取舍。
- Liu 等，[Lost in the Middle](https://arxiv.org/abs/2307.03172)：特定长上下文任务中证据位置效应的研究实例。

这些论文支撑其明确说明的机制和研究设置，不足以证明所有现代模型使用相同配置、所有实现有相同性能，或某个位置机制可以保证任意长度泛化。模型规格、API mask 语义、kernel 性能和长上下文能力应依据目标模型与版本单独核验。
