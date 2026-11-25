# 第 2 章 Transformer 组件实战

**正文对应：** [输入表示](../../../book-03-practical-handbook/chapters/02-transformer组件实战-1-输入表示.md)、[Attention 与 Mask](../../../book-03-practical-handbook/chapters/02-transformer组件实战-2-attention与mask.md)、[Decoder Block 与现代组件](../../../book-03-practical-handbook/chapters/02-transformer组件实战-3-decoder-block与现代组件.md)

本章围绕一条可检查的张量路径，把离散 token 变为上下文化 hidden state，再变为词表分数。读者需要能追踪维度、掩码可见性、位置编号、残差与归一化，解释组件改变了哪些信息路径，并能区分局部实现正确与整模型能力结论。

## 2.1 Token Embedding 与位置表示：让离散序列进入连续空间

本节解释 token ID 如何通过查表形成向量、位置如何进入表示，以及 padding、输出头和批处理约定怎样共同决定输入语义。

### 2.1.1 token id 是索引，不是有序特征

说明词表 ID 只标识类别，不代表 token 之间的距离或大小关系。

### 2.1.2 查表与 one-hot 矩阵乘法是同一个数学运算

从 embedding 查表与 one-hot 乘嵌入矩阵的等价关系解释高效实现。

### 2.1.3 参数量和梯度的稀疏访问

覆盖嵌入表参数规模、批次实际访问行及稀疏梯度语义。

### 2.1.4 padding_idx、attention mask 和 loss mask

区分填充 token 的 embedding 行、注意力可见性和监督损失屏蔽三个用途。

### 2.1.5 位置表示为什么必不可少

说明无位置输入时自注意力对序列排列的限制，以及位置机制需要提供的信息。

### 2.1.6 Learned positional embedding 的实现

解释可学习位置表的 shape、长度边界、参数成本与超出训练长度时的限制。

### 2.1.7 Sinusoidal encoding 是固定函数

说明正弦位置编码的频率构造和固定参数属性，并限定其与现代位置机制的关系。

### 2.1.8 输出头与 token embedding 权重共享

比较输入嵌入表与输出投影的维度关系、权重共享条件及参数规模影响。

### 2.1.9 左右 padding 与增量位置

分析 padding 方向、有效 token 位置编号和增量生成时位置 offset 的一致性要求。

### 2.1.10 从输入 id 到 hidden state，逐项核对

汇总词表索引、embedding、位置和形状检查，形成后续 attention 可复用的输入约定。

## 2.2 缩放点积注意力（scaled dot-product attention）：从相似度到信息汇聚

本节将 Query、Key、Value、缩放、softmax 和掩码连成明确计算对象，并区分手写实现与 PyTorch SDPA 的接口语义。

### 2.2.1 Q、K、V 是三个不同的角色

说明 Query 发起读取、Key 决定匹配、Value 承载被汇聚内容的线性投影角色。

### 2.2.2 点积、缩放和 softmax

推导分数矩阵、维度缩放与按 key 归一化的权重，解释各步的数值含义。

### 2.2.3 手写 attention，并把 mask 语义写进函数

界定输入 shape、缩放、掩码极性及输出权重/上下文，避免函数名掩盖约定。

### 2.2.4 mask 应在 softmax 前应用

说明屏蔽 logits 与事后屏蔽权重的差别，以及全屏蔽行可能造成的数值异常。

### 2.2.5 PyTorch 缩放点积注意力接口（SDPA）的边界

核对 SDPA 的输入布局、布尔 mask 约定、因果标志、dropout 和后端差异。

## 2.3 多头注意力（multi-head attention, MHA）：把一个相似度空间拆成多个子空间

本节连接投影、head 拆分/合并、self/cross attention 与资源复杂度，重点是通过形状追踪验证实现。

### 2.3.1 从单头到多头

解释多头将特征维分组计算多个注意力映射，再投影合并回模型维度。

### 2.3.2 逐步核对张量形状

跟踪 `[B,T,D]` 到 `[B,H,T,d_h]`、注意力分数和最终输出的维度变化。

### 2.3.3 一个可读的 MultiHeadAttention 实现

覆盖投影层、reshape、transpose、attention、合并 head 与输出投影的实现契约。

### 2.3.4 自注意力与交叉注意力

比较 Q/K/V 来自同一序列或不同序列时的语义及 mask 需求。

### 2.3.5 参数量与复杂度

估算投影参数量与序列长度相关计算/内存开销，明确 head 数不会自动降低总维度成本。

### 2.3.6 针对实现的局部测试

说明 shape、因果不泄漏、输出有限值及小规模数值对照可分别捕获哪些缺陷。

## 2.4 注意力掩码（attention mask）：把可见性写成明确的协议

本节将“哪些位置可被读取”视为计算协议，区分因果、padding 与加性掩码，并追踪不同 API 的布尔方向。

### 2.4.1 因果掩码与填充掩码解决不同问题

区分防止未来 token 泄漏与屏蔽 padding key 的目的及组合关系。

### 2.4.2 从 mask 形状推导广播

通过 batch、head、query 和 key 轴分析掩码广播，而不是依赖隐式形状猜测。

### 2.4.3 keep mask、block mask 和 additive mask

统一描述保留语义、屏蔽语义与 logits 加偏置表示之间的转换。

### 2.4.4 PyTorch 不同接口的布尔语义

指出 SDPA、MultiheadAttention 等接口可能采用不同布尔含义，要求按目标版本查证。

### 2.4.5 全 mask 行是一个独立异常状态

说明全屏蔽 query 行会令 softmax 不稳定或语义未定义，需明确处理策略。

### 2.4.6 在 softmax 前应用 mask 的完整函数

将 mask 极性、扩维和屏蔽位置处理组合为可审核的 attention 实现。

### 2.4.7 mask 的测试矩阵

以 padding、因果、混合批次和异常行组成覆盖矩阵，区分形状通过与语义通过。

## 2.5 Transformer 模块（block）：残差、归一化与逐 token 的非线性

本节把 attention、前馈网络、残差与归一化组合成一个 decoder block，连接梯度路径、资源估算和诊断方式。

### 2.5.1 Attention 与 FFN 的分工

解释跨 token 信息混合与逐 token 特征变换的职责区别。

### 2.5.2 FFN 的数学形态

覆盖升维、激活与降维投影，说明前馈子层如何改变通道表示。

### 2.5.3 Pre-LN 与 Post-LN：归一化放在残差分支的哪里

比较两种残差块中的归一化位置及其信号/梯度路径差别。

### 2.5.4 LayerNorm、Dropout 与训练模式

说明归一化统计轴、Dropout 随机性和模块训练模式之间的关系。

### 2.5.5 一个 Pre-LN decoder block

界定注意力子层和 FFN 子层在残差流中的输入、输出与顺序。

### 2.5.6 残差连接的梯度路径

用链式法则解释恒等分支对梯度传播的作用及不能据此保证训练稳定的边界。

### 2.5.7 参数量与计算量

按模型维、前馈维、层数和序列长度估算 block 规模与主要计算项。

### 2.5.8 Block 的诊断顺序

围绕逐层 shape、有限值、mask、残差尺度和梯度检查定位 block 故障。

## 2.6 旋转位置编码（RoPE）：把位置关系写进 Q、K 的几何变换

本节解释 RoPE 的二维旋转构造、相对位置点积性质和工程实现约定，重点核对 Q/K 布局、offset 与缓存位置。

### 2.6.1 为什么 self-attention 需要位置机制

说明纯内容相似度不编码顺序，以及位置机制与序列索引的关系。

### 2.6.2 二维旋转和偶奇维配对

定义成对通道旋转、角度频率与位置相关变换。

### 2.6.3 位置旋转如何产生相对距离

说明 Q、K 同时旋转后点积依赖相对位移的数学关系及其适用条件。

### 2.6.4 cos/sin cache 与 offset

说明预计算三角值如何降低重复开销，以及增量解码需从正确位置切片。

### 2.6.5 应用到 q 和 k，不应用到 v

解释位置变换作用于注意力寻址分数的 Q/K 路径，而 Value 保持内容载荷角色。

### 2.6.6 把 RoPE 接入 attention

将位置变换接入投影输出，核对维度、广播、dtype 和缓存位置。

### 2.6.7 位置布局和工程边界

比较 interleaved 与 split-half 等配对布局，并界定 checkpoint 与实现的兼容要求。

## 2.7 均方根归一化（RMSNorm）与 SwiGLU：现代 decoder block 的两个改造

本节把归一化和门控前馈结构放回 block 组合，说明它们与标准 Transformer 的联系及配置兼容边界。

### 2.7.1 LayerNorm 和 RMSNorm 的数学差异

比较中心化、均方根缩放和可学习尺度参数，避免将实现差异说成效果保证。

### 2.7.2 低精度下的实现

讨论累积精度、epsilon 和输入/参数 dtype 对归一化数值稳定性的影响。

### 2.7.3 SwiGLU 的门控结构

说明两个投影分支、SiLU 门控和逐元素交互怎样形成前馈输出。

### 2.7.4 参数量为什么常用约 $8D/3$

从 SwiGLU 三个线性映射与中间维度设定推导近似参数量，并标明近似假设。

### 2.7.5 SwiGLU 实现

核对门控分支顺序、隐藏维、投影偏置与输出 shape。

### 2.7.6 现代 decoder block 的组合

组合 RMSNorm、RoPE、因果 attention、残差和 SwiGLU，明确结构选择必须与 checkpoint 配置一致。

### 2.7.7 组件级证据和整机级结论

区分单组件局部验证、端到端生成质量和性能测量的证据范围。

### 2.7.8 一条完整的 shape 路径

从 token 输入一路追踪 batch、序列、head 与 hidden 维，形成可调试的维度契约。

### 2.7.9 组件回归实验的最小集合

界定组件改动后需要复核的数值、mask、缓存与行为不变量。

## 继续阅读与资料边界

连接原始论文、PyTorch API 与框架文档，指出 API 布尔语义和 checkpoint 布局需要按固定版本核对。
