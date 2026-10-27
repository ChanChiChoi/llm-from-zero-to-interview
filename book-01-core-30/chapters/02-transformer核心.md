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

看同一段“模型”在三种刻意简化的切法下会有多少单元：字符级切成“模”“型”，是 2 个 token；原始 UTF-8 字节级（每个字节单独作为 token）中每个汉字占 3 字节，是 6 个 token；若某个玩具子词词表恰好有完整 token“模型”，则是 1 个 token。这里的字节级示例特指不合并字节的原始字节 tokenizer，真实 byte-level BPE 还可能把常见字节串合并。一个字符、一个字节和一个 token 是三种不同计量单位，不能互相换算成固定比例。

BPE 的基本思路是在训练语料的符号序列中反复合并高频相邻单元。用一个能手算的小语料看一次 merge：

| 阶段 | 符号序列 | 相邻对计数 |
| --- | --- | --- |
| 初始 | a b a b | ab 出现 2 次，ba 出现 1 次 |
| 合并最常见的 ab | ab ab | 序列从 4 个符号缩短为 2 个 |

若语料中还有别的序列，计数要在整个训练语料上累计；同频时如何打破平局、何时停止、是否先做预切分，都会影响最终词表。这个小例子只展示“一次按频率合并”的计算，不等于任何模型的实际 tokenizer 训练流程。[Sennrich 等人的 BPE 论文](https://arxiv.org/abs/1508.07909)讨论了用子词单元处理罕见词的动机。

WordPiece 和 BPE 都会得到子词词表，但不能把它们说成同一种 merge 算法。SentencePiece 是直接面向原始文本训练和应用子词模型的工具，支持的模型类型不等于“SentencePiece 这一种切分算法”。[BERT 论文](https://arxiv.org/abs/1810.04805)是 WordPiece 的代表性实例；[SentencePiece 论文](https://arxiv.org/abs/1808.06226)介绍了其原始文本处理和子词建模方式。

### Tokenizer 是模型输入契约的一部分

Tokenizer 会影响一段文本占多少 token、固定上下文窗口能容纳多少文本、训练样本如何构造，以及特殊 token 如何表示。这个影响必须针对具体 tokenizer 和语料测量，不能直接总结为“某种语言总是更费 token”。

训练和推理应使用兼容的 tokenizer、词表、特殊 token 配置和模型 embedding。若 tokenizer 新增或重排词表项，模型的 embedding 行也必须按约定同步；仅仅重新编号会改变每个 ID 对应的向量。评估 token 长度或 token 成本时，也必须记录使用的 tokenizer。

例如，训练时“模型”被编码成 ID 9，推理时若换了 tokenizer，结果变成 ID 31，模型收到的就不是“同一个意思的另一种写法”，而是查了 embedding 的另一行。除非词表与 embedding 按明确映射一起转换，否则原来学到的向量不会自动跟着文字搬家。这也是 tokenizer 属于模型输入契约、不能像可替换的前置清洗脚本一样随意更换的原因。

**交接：**现在得到的是一个有序 ID 序列。下一讲把它变为 [B,T,D] 的向量序列，并处理“模型怎样区分位置”。

## 第 7 讲：Embedding 与位置信息——编号怎样变成表示

### 查表把离散 ID 变成向量

ID 是词表里的地址，不是词义本身。Embedding 是一个可训练矩阵 $E \in \mathbb{R}^{V \times D}$；输入 ID 为 i 时，模型取出第 i 行。若一批输入的 shape 是 [B,T]，查表会在每个 batch、每个位置各取一行，结果变成 [B,T,D]：新增的 D 维就是每个 token 的向量表示。

~~~math
X_{b,t,:} = E[\mathrm{token\_id}_{b,t}]
~~~

可以把查表理解成 one-hot 矩阵乘法的高效实现。ID 为 12 时，one-hot 向量只有第 12 位为 1；它与 E 相乘后恰好取出 E 的第 12 行。实际实现通常直接做索引，不会真的构造一个宽度为 V 的稀疏向量。

用贯穿本章的序列 [12,44,9] 做一次数值查表。假设这三个 ID 分别对应“我”“喜欢”“模型”，下面的二维向量是为了演示而构造的玩具参数，不是任何已训练模型的真实 embedding：

| 词表 ID | token | Embedding 行 | 查表后的序列位置 |
| ---: | --- | --- | ---: |
| 12 | 我 | (0.2, -0.1) | 1 |
| 44 | 喜欢 | (-0.3, 0.4) | 2 |
| 9 | 模型 | (0.5, 0.2) | 3 |

因此，单条长度为 3 的输入从整数张量 [1,3] 变成：

~~~text
[[[ 0.2, -0.1],
  [-0.3,  0.4],
  [ 0.5,  0.2]]]
shape: [1, 3, 2]
~~~

若 batch 中有第二条序列 [44,12,9]，相同 ID 仍查到相同的行，只是行的排列不同；整体 shape 成为 [2,3,2]。这说明 D 维是每个 token 的特征宽度，B 和 T 两个轴仍分别表示样本和位置。

这些向量不是手工写入的定义。训练时，预测损失的梯度会沿计算图传回被用到的 embedding 行；同一个 ID 若在样本中出现多次，其梯度贡献会累加。训练开始时，向量通常还没有可解释的语义；它们通过后续 attention、MLP 和预测目标共同调整，最终是否形成有用的表示要看训练结果，不能从 ID 数值或初始化向量直接读出。

### 为什么还需要位置

只查表会让相同 token 在任何位置都拿到同一行。例如 ID 12 出现在句首还是句尾，单看 E[12] 是无法区分的。对不带位置编码的双向 self-attention，单纯重排输入会相应重排输出，模型没有额外的绝对位置标记。Decoder-only 模型的 causal mask 本身会规定哪些前缀可见，因此它并非完全没有顺序约束；但“能看见前面、不能看见后面”和“相距一位还是一百位、现在是第几个位置”不是同一类信号。位置机制把后者交给模型学习或计算。

以两维玩具向量为例，设位置向量依次是 $P_0=(0.1,0)$、$P_1=(0,0.2)$、$P_2=(-0.1,0.1)$。常见绝对位置相加得到：

| 位置 | Token embedding | 位置向量 | 相加后的初始表示 |
| ---: | --- | --- | --- |
| 0：“我” | (0.2, -0.1) | (0.1, 0) | (0.3, -0.1) |
| 1：“喜欢” | (-0.3, 0.4) | (0, 0.2) | (-0.3, 0.6) |
| 2：“模型” | (0.5, 0.2) | (-0.1, 0.1) | (0.4, 0.3) |

把前两个 token 换成 [44,12] 时，ID 12 现在位于位置 1，它的初始表示变为 (0.2,-0.1)+(0,0.2)=(0.2,0.1)，不再是位置 0 的 (0.3,-0.1)。输入向量因此同时携带 token 身份和位置线索。表中的值只是为了展示相加，不表示真实模型中 token 有这些语义坐标。

对于绝对位置编码，本次序列取出位置 0 至 T-1 的向量 $P_{0:T-1}$，shape 是 [T,D]；把它加到 [B,T,D] 的 embedding 上时，位置向量在 batch 轴广播，输出 shape 仍是 [B,T,D]。若使用 learned absolute embedding，底层位置表通常按最大支持长度分配，本次只取前 T 行。两者最后一维必须兼容。位置向量的数值会随着层间计算被混合，经过 attention 和 MLP 后的 hidden state 不再是静态的“词向量”。

### 位置方案改变的是计算路径

“加入位置”不是一种固定操作。比较时先问它改动了哪一步，而不是只背方法名称：

| 方案 | 位置信息进入哪里 | 读者应理解的计算差异 |
| --- | --- | --- |
| Learned absolute embedding | 加到 token embedding 上 | 每个位置有可学习的向量表；表长和训练位置范围需要考虑 |
| Sinusoidal absolute encoding | 加到 token embedding 上 | 由固定频率的正弦/余弦生成，不为每个位置学习独立向量 |
| Relative bias / ALiBi | 加到 attention score 上 | 根据 Query 与 Key 的相对距离调整匹配分数 |
| RoPE | 旋转 Query 和 Key | 改变 Q/K 点积，使相对位移进入注意力；第 14 讲推导 |

原始 Transformer 的正弦位置编码可以用二维最小例子读懂：取最高频率对应的角度步长为 1 弧度，则位置 0 的两个分量是 [sin 0, cos 0]=[0,1]，位置 1 是 [sin 1, cos 1]≈[0.8415,0.5403]。实际模型有更多维度和频率，不能把这个二维演示当作完整配置。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)介绍原始结构及正弦位置编码。

ALiBi 不把位置向量加到输入，而是在 attention score 中按距离加入线性 bias。[Train Short, Test Long](https://arxiv.org/abs/2108.12409)介绍了该方法及其论文设置；这里不据此推断所有任务都能外推到任意长度。RoPE 的具体旋转要等 Q/K 定义清楚后再讲，因为它作用于注意力计算内部，而不是简单加在输入向量上。

**本讲小结：**Embedding 回答“这个 ID 查到哪组可学习参数”，位置机制回答“同一 token 的计算发生在序列什么位置”。读者现在应该能从 [B,T] 手算出一组 [B,T,D] 初始表示，并说明为什么仅有 embedding 不能表达所有位置关系。

**交接：**每个位置已有带位置线索的向量，但还没有从同一句其他位置汇入内容。第 8 讲先看一次信息混合长什么样，第 9 讲再推导模型如何算出混合权重。

## 第 8 讲：Self-Attention 直觉——当前位置怎样读取上下文

### 表示需要依赖上下文

同一个 token 在不同上下文中的作用可能不同。比如“苹果”可以指水果，也可以指公司；单看 token 本身无法确定当前语境。模型需要根据当前序列，把其他位置的信息有选择地汇入当前位置表示。

Self-Attention 的每个位置都会从同一序列的表示中生成三种向量：

- **Query（查询）**：当前位置用来计算“我和哪些位置相关”的向量；
- **Key（键）**：每个位置用来参与匹配的向量；
- **Value（值）**：匹配后实际被加权汇入的信息向量。

这只是对学习到的投影功能的直觉，不意味着模型内部显式执行人类语言中的“提问、查找、取答案”。

### 从匹配到加权汇总

对某个 Query，模型计算它与所有可见 Key 的匹配分数，把分数归一化成权重，再按这些权重混合 Value。Query 和 Key 决定“从哪儿取多少”，Value 决定“取回来的内容是什么”。它不是按关键词找到一个位置后照抄，而是可以同时从多个位置取信息。

先把匹配分数暂时当作已算好，单看混合这一步。假设位置 3 的 Query 对前三个可见位置给出权重 [0.2,0.3,0.5]，对应 Value 分别是 (2,0)、(0,2)、(1,1)：

| 来源位置 | Attention 权重 | Value | 加权贡献 |
| ---: | ---: | --- | --- |
| 1 | 0.2 | (2,0) | (0.4,0) |
| 2 | 0.3 | (0,2) | (0,0.6) |
| 3 | 0.5 | (1,1) | (0.5,0.5) |
| 合计 | 1.0 | — | (0.9,1.1) |

因此当前位置的新向量是三个加权贡献之和 (0.9,1.1)。权重和为 1，是因为 softmax 会在可见 Key 之间归一化。表中的向量和权重是教学用假设值，不代表某个模型对这句话的真实注意力；第 9 讲会从具体 Q、K 算出分数、softmax 权重和输出。

“Attention 权重”说明的是模型在这一层如何分配 Value 的加权系数，不能单独当作完整的因果解释或特征归因。网络有多层、多个头和后续非线性，输出由整条计算路径共同决定。

### 为什么它适合并行，也为什么代价随长度上升

在一层的训练前向中，整条序列的 token 表示已经给定，所以 Q、K、V 可以一次投影；所有 Query 与 Key 的点积也能组成矩阵乘法并行计算。这和标准 RNN 必须先得到第 t-1 步状态才能计算第 t 步不同。但“能并行”不等于“计算量不随长度增长”：每个 Query 都要和很多 Key 比较。

具体数一数，一个头、长度 T=4 的完整分数矩阵有 $4\times4=16$ 个位置对；使用 causal mask 后只允许对角线及其下三角，共 $4+3+2+1=10$ 个可见位置对。长度翻倍到 T=8 时，完整矩阵有 64 个位置对，因果可见位置有 $8+7+\cdots+1=36$ 个，约为原来的 3.6 倍；长度更大时，二次增长会逐渐接近四倍。朴素 dense 实现仍可能先算完整 64 个分数再屏蔽上三角；某些专用实现可以跳过被屏蔽的计算。多头和 batch 会再乘上对应规模，优化实现可以减少中间读写，但不会让基础全连接注意力的关系数变成线性。

训练时并行的是一次前向中不同位置的计算；生成时，新 token 依赖刚选出的前一个 token，因此输出 token 仍要逐步产生。KV Cache 可以复用过去的 K/V，省掉部分重复计算，却不会让未来 token 在逻辑上提前生成。这个区别帮助我们把“注意力层内部可并行”和“自回归生成必须逐 token”分开理解。

Self-Attention 也不应简单等同于“关键词搜索”：Key/Query 的匹配空间是模型学习出来的，Value 经过多层投影，最终信息影响不止来自某一张注意力图。它解决的是可学习的信息混合问题，代价和可见范围由具体结构决定。

**交接：**我们已经手算了“权重已知时怎样混合 Value”，也估算了序列变长时要比较多少位置对。下一讲补上最关键的缺环：这些权重是怎样从 Q/K 的点积和 softmax 中算出来的，并用 mask 把不允许读取的位置排除。

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

看 shape 有助于确定“对谁做点积”：单个 Query 是 [d_h]，一组 Key 是 [T,d_h]，二者相乘得到该 Query 对 T 个位置的 T 个分数。整组 Query 是 [T,d_h]，与 Key 转置相乘后得到 [T,T]；第 i 行表示 Query i 对各个 Key 的分数。多 batch、多头时前面再带上 B 和 H，因此 shape 为 [B,H,T,T]。随后 softmax 沿最后的 Key 位置维度逐行归一化，不会把不同 Query 行混在一起。

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

1. 原始点积为 $qk_1^\top=1$、$qk_2^\top=0$，所以分数是 [1,0]。
2. 这里 $d_h=2$，除以 $\sqrt 2\approx1.4142$ 后，缩放分数是 [0.7071,0]。
3. Softmax 先对两个分数取指数。为避免大数溢出，可以先减去最大值 0.7071，得到 [0,-0.7071]；指数约为 [1,0.4931]，总和约 1.4931，因此权重为 [1/1.4931,0.4931/1.4931]≈[0.6698,0.3302]。
4. 逐维加权 Value：第一维 $0.6698\times2+0.3302\times0=1.3396$；第二维 $0.6698\times0+0.3302\times4=1.3208$。输出约为 (1.3396,1.3208)。

每一步都能定位输出变化来自哪里：点积产生匹配分数，缩放影响 softmax 的尖锐程度，softmax 决定权重，Value 决定被混合的内容。如果不除以 $\sqrt{d_h}$，更高维下点积可能随维度增大而变大，softmax 更容易集中在少数位置；如果交换两个 Value，权重不变但输出会交换对应的信息贡献。这个小例子只演示定义，不是模型测量。

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

例如，长度为 3 时，位置 2 的 Query 可以看位置 1 和 2，但不能看位置 3。假设原始分数是 [1,2,100]，第三个位置即使分数极大，mask 后也变为 [1,2,-∞]；softmax 权重约为 [0.269,0.731,0]。屏蔽不是给未来位置一个“比较小”的分数，而是让其权重严格为 0。实际实现还要正确处理 padding mask 和变长样本，避免某一行所有位置都被屏蔽。

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

用一组小 shape 看清拆头与拼回去：取 B=1、T=3、D=8、Query 头数 H_q=4，则每头维度 $d_h=D/H_q=2$。输入 X 的 shape 是 [1,3,8]；拆头后 Q、K、V 都是 [1,4,3,2]；每个头各自产生 [1,3,2] 的输出，沿最后一维拼接回 [1,3,8]，再经过输出投影。拆头不是把序列切成四段，而是把每个 token 的 8 维表示分成四个 2 维子空间。

当隐藏宽度 D 固定时，仅改变 Query 头数 H_q 并相应改变每头宽度，只要 Q/K/V/O 的投影总宽度仍为 D，就不会自动增加这些投影矩阵的参数量。例如 D=8 时，W_Q 仍是 8×8；H_q=2 时是两个 4 维头，H_q=4 时是四个 2 维头。参数形状主要由投影输入、输出宽度决定，头数决定这些通道怎样分组。实际实现还可能包含 bias 或不同投影布局。

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

继续沿用 D=8、H_q=4、$d_h=2$ 的例子：MHA 有 4 个 KV 头，K/V shape 各为 [1,4,3,2]；若采用 2 个 KV 头的 GQA，K/V shape 各为 [1,2,3,2]，可以让 Query 头 0、1 共享 KV 头 0，Query 头 2、3 共享 KV 头 1；MQA 则把四个 Query 头都映射到同一个 KV 头。Query 输出仍需拼回 D=8，所以改变 KV 头数与改变 Query 头数不是一回事。

### KV Cache 为什么受 KV 头数影响

逐 token 生成时，早先位置的 Key 和 Value 可以缓存起来，后续步骤复用。单条序列的原始 KV Cache 载荷可按下式估算：

~~~math
\mathrm{KV\ bytes}
=
2 \times L \times T \times H_{kv} \times d_h \times b
~~~

其中 L 是层数，T 是缓存长度，前面的 2 对应 K 与 V，b 是每个元素的字节数。以 L=32、T=8192、d_h=128、b=2 字节为例：

先看一个 KV 头在一个位置、一个层上占多少：K 和 V 各有 128 个元素，每个元素 2 字节，所以 $2\times128\times2=512$ 字节。GQA 的 8 个 KV 头在 8192 个位置、32 层上的载荷是 $512\times8\times8192\times32=1,073,741,824$ 字节，即 1 GiB。MHA 用 32 个 KV 头，按相同比例是 4 GiB；MQA 用 1 个 KV 头，则是 128 MiB。表中用二进制单位换算：1 GiB=$2^{30}$ 字节，1 MiB=$2^{20}$ 字节。

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

把三组位置逐项对齐，可以检查标签有没有错位：

| 位置 i | 当前输入前缀 | 该位置要预测的 target |
| ---: | --- | ---: |
| 1 | [12] | 44 |
| 2 | [12,44] | 9 |
| 3 | [12,44,9] | 31 |

位置 i 的 logits 预测下一个 token $x_{i+1}$，但位置 i 的表示只能读取 $x_1,\ldots,x_i$。目标右移定义“预测谁”，causal mask 定义“预测时能看见什么”；这两个约束各司其职，不能互相替代。若把 input 和 target 错写成完全相同的序列，模型就会被要求复制当前位置，而不是预测下一个 token；若 target 右移正确但 mask 允许看到未来，训练也会发生信息泄漏。

语言模型的 token 级交叉熵目标可以写成：

~~~math
\mathcal{L}
=
-\frac{1}{T}
\sum_{i=1}^{T}
\log p_\theta(x_{i+1}\mid x_1,\ldots,x_i)
~~~

首个预测位置常通过 BOS 等起始约定处理。训练时一次前向可以并行计算所有位置的 logits，因为 mask 已让每个位置只能看到允许的前缀；并行计算没有泄漏未来目标。

再用一组假设的预测概率算一次 loss。若三个位置对真实 target 的概率分别是 0.5、0.25、0.8，则每个位置的负对数似然约为 0.6931、1.3863、0.2231，平均交叉熵为 $(0.6931+1.3863+0.2231)/3\approx0.7675$。第二个目标只分到 0.25 的概率，因此对平均 loss 贡献最大。这里的概率是为解释公式而构造的，不是模型运行结果；训练通过梯度调整参数，让正确 target 的概率在数据分布上提高。

### Teacher Forcing 不等于允许看到未来

Teacher Forcing 是训练时把真实历史 token 作为模型输入。对位置 i，它拿到的是此前的真实前缀，而不是此前模型采样出来的 token；causal mask 仍然限制该位置看不到未来 token。推理时没有真实 target，模型必须从当前分布选择一个 token，把它追加到上下文，再继续计算。

沿用 [12,44,9,31]：训练时把 [12,44,9] 一次送入模型，位置 1 预测 44、位置 2 预测 9、位置 3 预测 31；虽然一次前向并行算出三个位置，后两个位置的输入仍是数据中的真实历史。生成时从提示 [12] 开始，模型先选出一个 token（假设选到 44），再把 [12,44] 作为下一轮上下文。若生成第二步选错了，后续上下文也会包含这个生成结果，而不会自动改回训练语料中的真实 token。这就是训练与推理输入分布可能不同的直观来源之一。

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

这两步不是把同一件事做两次。Attention 的输入按 token 位置互相读取，因而能把前文信息写入当前位置；MLP 随后对每个位置分别变换特征维度，所有位置复用同一组 MLP 权重。两步的输出宽度都回到 D，才能分别与对应的残差输入相加。堆叠时，下一层收到的是已经混合上下文、再经过逐位置变换后的表示。

用一个单 batch、两个位置、二维隐藏状态把结构走一遍。设进入 Attention 残差支路的表示是 $x_1=(1,0)$、$x_2=(0,1)$；为单独检查 block 的组合方式，假设 Attention 子层（含其输出投影）给出 $a_1=(0.5,0.25)$、$a_2=(0.25,0.5)$：

| 位置 | 残差输入 x | Attention 子层输出 a | 相加后的 x' |
| ---: | --- | --- | --- |
| 1 | (1,0) | (0.5,0.25) | (1.5,0.25) |
| 2 | (0,1) | (0.25,0.5) | (0.25,1.5) |

假设第二个 MLP 子层分别对这两个位置输出 $m_1=(-0.1,0.2)$、$m_2=(0.2,-0.1)$，第二次残差相加后得到 $x_{\mathrm{next},1}=(1.4,0.45)$、$x_{\mathrm{next},2}=(0.45,1.4)$。这些子层输出是为了展示相加而给定的玩具值，并非完整网络前向的测量结果；这个例子要读出的不变量是每次相加都逐元素进行，shape 从 [1,2,2] 保持不变。真实 Attention 如何由 Q/K/V 算出输出，已在第 9 讲推导。

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

逐 token MLP 的典型形式是先把宽度扩展到 $D_{ff}$，经过激活，再投影回 D。比如 D=8、$D_{ff}=32$ 时，每个 token 的特征先从 8 维扩到 32 维，再回到 8 维；T=3 时，这个相同的变换分别作用于 3 个位置，不会让位置 1 直接读取位置 2。它与 Attention 的关键区别是：Attention 在 token 轴上交换信息，MLP 在每个 token 的特征轴上加工信息。

原始 Transformer 论文使用了特定的前馈层和 encoder-decoder 结构；现代 decoder-only 模型在激活函数、归一化位置、bias、dropout 和注意力实现上会有不同配置。[原始 Transformer 论文](https://arxiv.org/abs/1706.03762)不应被当成每个现代 GPT 的逐项规格。D_ff 常比 D 宽，但具体倍数和门控结构取决于模型配置。

不同 Transformer 类型的 Block 也会有不同的注意力支路：

| 结构 | 输入序列内部的注意力 | 额外读取的信息 |
| --- | --- | --- |
| Encoder | Self-Attention 通常可看整个输入 | 通常没有自回归输出要求 |
| Decoder-only | Causal Self-Attention 只能看当前位置及之前 | 直接预测下一个 token |
| Encoder-decoder 的 Decoder | Causal Self-Attention 处理已生成前缀 | Cross-Attention 读取 Encoder 编码的输入 |

GPT 类语言模型通常没有 encoder 输出，因此没有 cross-attention 这条支路。Transformer 堆叠之外，模型通常还有 final normalization 与语言模型头：它把每个位置的 [D] 表示映射成 [V] 个词表 logits，所以整体 shape 从 [B,T,D] 变为 [B,T,V]。

**交接：**Block 的结构已经闭合，但“残差为什么有用、Norm 具体归一化什么、Pre-LN 和 Post-LN 差在哪里”仍需要解释。第 13 讲只深入这些稳定性问题，不再重复 Block 结构。

## 第 13 讲：LayerNorm、RMSNorm 与残差——让深层计算保持可训练

### 残差支路保留直接的信息通路

残差块输出 $x+F(x)$。它不是把旧表示丢掉后完全依赖新变换，而是保留输入 x，再叠加当前子层学到的更新 $F(x)$。例如 x=(1,2)、某个子层给出小更新 $F(x)=(-0.05,0.1)$，残差输出为 (0.95,2.1)；子层只需表示“相对输入要改多少”，输入本身仍在结果中。这是解释残差相加的玩具算例，不表示训练中每次更新都很小。

~~~math
\frac{\partial (x+F(x))}{\partial x}
=
I + \frac{\partial F(x)}{\partial x}
~~~

公式中的 I 是沿输入直通的恒等路径；梯度还包含子层的局部导数 $\partial F/\partial x$。直通路径为梯度提供一条不必穿过 F 的通路，有利于深层优化，但其他梯度项仍可能放大、抵消或变得不稳定。因此残差不是“加了就一定能训练”的保证。Block 采用残差后仍需要控制每层数值尺度。Normalization 会在指定维度上重标定表示，帮助训练过程管理激活统计；它不是“让所有 token 变得一样”，也不会替代良好的数据、初始化和优化设置。

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

以 x=[1,2,3,10] 且暂不应用可学习参数、忽略很小的 $\epsilon$ 为例。LayerNorm 先求均值 4，中心化后得到 [-3,-2,-1,6]；平方和为 50，除以 D=4 得方差 12.5，标准差约 3.5355，再把中心化分量逐个除以它。RMSNorm 不中心化：平方和是 $1+4+9+100=114$，均方为 28.5，分母 $\sqrt{28.5}\approx5.3385$，再把原向量逐项除以该分母。

| 方法 | 计算中间量 | 归一化结果（约） |
| --- | --- | --- |
| LayerNorm | 均值 4；中心化 [-3,-2,-1,6]；标准差 3.5355 | [-0.8485,-0.5657,-0.2828,1.6971] |
| RMSNorm | 均方 28.5；RMS 分母 5.3385 | [0.1873,0.3746,0.5620,1.8732] |

差别来自是否先减均值，不是两种公式只换了名称。LayerNorm 的可学习 $\gamma$ 和 $\beta$、RMSNorm 的可学习 $\gamma$ 会再改变结果；实际计算还保留 $\epsilon$ 以避免除零。[RMSNorm 论文](https://arxiv.org/abs/1910.07467)提出了省略均值项的归一化形式及其实验；不应据此宣称 RMSNorm 在所有模型上都优于 LayerNorm。

### Pre-LN 和 Post-LN 改变了计算顺序

~~~math
\mathrm{Pre\text{-}LN}: \quad y = x + F(\mathrm{Norm}(x))
~~~

~~~math
\mathrm{Post\text{-}LN}: \quad y = \mathrm{Norm}(x + F(x))
~~~

两种结构把归一化放在残差分支的不同位置，梯度路径和训练行为因此不同。沿着公式逐步看：Pre-LN 先把 x 送进 Norm，再只把 F 的结果加回原始 x；Post-LN 先把 x 与 F(x) 相加，最后对总和归一化。两者改变了哪些值进入归一化，也改变了残差支路通往下一层的形式。Pre-LN 的残差支路更直接地保留未归一化的 x；Post-LN 则在每个子层相加后都重新归一化。这里讲的是计算顺序差异，不据此给所有模型排一个稳定性高低名次。[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)分析了其研究设置下 Pre-LN 与 Post-LN 的梯度路径差异。

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

若取二维单位向量 q=k=(1,0)，每个位置步长角为 0.25 弧度，旋转后的点积为 $\cos((n-m)\times0.25)$。把两个位置对实际代入：

| 位置对 (m,n) | Query 旋转角 | Key 旋转角 | 相对距离 | 旋转后点积 |
| --- | ---: | ---: | ---: | ---: |
| (1,3) | 0.25 | 0.75 | 2 | $\cos(0.5)\approx0.8776$ |
| (8,10) | 2.00 | 2.50 | 2 | $\cos(0.5)\approx0.8776$ |

例如第一行的旋转向量约为 q'=(0.9689,0.2474)、k'=(0.7317,0.6816)，点积 $0.9689\times0.7317+0.2474\times0.6816\approx0.8776$。第二行的绝对角度更大，但相减仍是 0.5，因此这个玩具例子里点积相同。这说明相对位移如何进入 Q/K 点积；它不代表完整模型的注意力分布，也不意味着把模型位置无限延长后性能不变。

### 位置表示、最大窗口与有效使用能力是不同问题

位置机制解决“怎样把顺序和距离表示进模型”；上下文窗口表示模型一次可以处理多少 token；有效长上下文能力还取决于训练长度与数据、位置处理、注意力计算、模型是否能找到相关证据，以及任务本身。

| 扩展方向 | 主要处理的问题 | 仍需单独验证什么 |
| --- | --- | --- |
| RoPE scaling 等位置变换 | 调整超出训练位置范围时的位置映射 | 短上下文回归、远近位置、不同任务效果 |
| FlashAttention 等 kernel | 减少注意力中间数据搬运开销 | 实际硬件、shape、dtype 下的吞吐和内存 |
| Local / Sparse Attention | 减少部分 token 间交互 | 被移除的长距离连接是否影响目标任务 |
| RAG 或外部检索 | 不把所有资料都塞进一次上下文 | 检索召回、证据定位和答案使用是否正确 |

长上下文不是某个位置编码单独带来的属性。模型可能接受很长的输入，却对其中间区域或特定任务表现不稳；[Lost in the Middle](https://arxiv.org/abs/2307.03172)在其研究设置中观察到证据位置会影响长上下文任务表现。它是评测设计应关注的一类现象，不是所有模型和任务上的固定曲线。更深的位置扩展、长上下文 RAG 与评估见[第二册相关章节](../../book-02-advanced-100/chapters/07-长上下文rag与agent.md)。

把这几种限制拆开看：假设一个服务的硬窗口是 4096 token，证据位于输入第 5000 个 token，那么这次请求根本没有把证据送进模型；改用 8192 token 窗口后，证据才进入可见输入，但模型是否能定位、理解并使用它仍需任务评测。RoPE 处理的是位置关系如何参与注意力分数，窗口上限由模型和服务配置决定，长文本中的检索与推理能力则要通过位置分布、任务和答案正确性单独评估。这个假设例子说明三个问题不能互相替代，不是某一模型的实测表现。

**交接：**本章已分别解释输入、上下文混合、因果训练、深层 block 和位置。最后一讲把这些机制放回同一个小模型，并沿实际 shape 走一遍。

## 第 15 讲：从零实现一个小 GPT——把主线接成程序

### 代码要验证什么

下面给出一份完整的字符级 decoder-only GPT 教学脚本。它把字符串映射为字符 ID，用可学习绝对位置 embedding、Pre-LN Transformer Block、因果 self-attention 和交叉熵训练，再逐 token 采样生成。

这段代码的目标是让输入、mask、残差和 logits 的 shape 可检查。它不实现 RoPE、GQA、KV Cache、混合精度、FlashAttention、停止 token 或分布式训练，因此不是生产 GPT。运行环境以仓库锁定的 Python 3.13+ 与 PyTorch 2.12.0 为准；脚本没有依赖预训练模型或外部数据，代码中的形状由构造参数决定。正文给出代码预期观察，不声称本轮已运行。

运行前先按源码把尺寸算出来。重复字符串“hello transformer\n”每份有 18 个字符，重复 64 次后数据长度是 1152；其中共有 13 种不同字符，所以字符词表大小 V=13。`block_size=16`、`batch_size=8`，`get_batch` 每次从数据中抽 8 个起点，每个起点取 16 个字符作为 x，再取右移一位的 16 个字符作为 y。因此 x、y 都是 [8,16]。采样上界是 $1152-16=1136$，Python 的上界不包含在候选值里，最大起点为 1135；这样 y 的最后一个索引刚好是 1151，不会越过 1152 个数据元素。模型宽度 D=32、Query 头数 H=2，每头宽度 $d_h=32/2=16$；两个 Block 输出仍保持 [8,16,32]。

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

代入本脚本的参数，最后一层 logits 是 [8,16,13]，表示 8 条序列、每条 16 个预测位置、每个位置对 13 种字符各有一个分数。计算交叉熵前把前两维合并成 [128,13]，target 对应变成 128 个类别编号；所以一次更新汇总了 8×16=128 个 next-character 预测位置。这个 shape 核算能帮助定位常见错误：如果把序列长度或词表维度放错，loss 输入与 target 就无法一一对齐。

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

生成从一个字符提示开始，循环 40 次，每次追加 1 个 ID，所以返回序列长度为 $1+40=41$。输入长度超过 block_size=16 后，代码只把最近 16 个 ID 送进模型；由于没有 KV Cache，每轮仍会重新计算当前截断上下文中所有位置的表示。这个脚本里的文本只是重复的短字符串，100 次参数更新只能展示训练循环怎样闭合，不足以支持关于语言能力、泛化或推理性能的结论。

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
