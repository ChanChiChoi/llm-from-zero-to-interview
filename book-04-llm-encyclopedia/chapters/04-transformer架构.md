# D. Transformer 架构：从信息路由到长序列系统

## 阅读边界

Transformer 不是一个单独的 attention 公式，而是一套把离散 token 变成上下文表示、再变成下一个 token 概率的架构。它由 embedding、位置表示、Q/K/V 投影、mask、attention、MLP、归一化、残差、输出头和训练目标共同组成。理解架构时，不能只记住“Q 看 K，V 被加权”，还要追踪每一个张量的形状、每一条信息路径和每一种成本。

本章沿着 decoder-only Transformer 的数据流展开，同时对照 encoder-only、encoder-decoder、稀疏 attention、线性 attention、递归状态架构和 MoE。全章分为十三个独立主题：

~~~text
token id
 -> embedding
 -> position representation
 -> Q / K / V
 -> mask and attention
 -> MHA / MQA / GQA
 -> MLP and residual block
 -> logits and language-model loss
 -> prefill / decode / KV cache
~~~

每个主题都从初学者直觉、专家变量、公式、工程例子和失败边界讲起。需要先学习组件实现的读者，可以阅读[第一册第二章：Transformer 核心](../../book-01-core-30/chapters/02-transformer核心.md)；需要追踪架构演进和真实模型配置的读者，可以继续阅读[第二十一册：Transformer 架构演进](../../book-21-transformer-architecture-evolution/目录.md)。本章中所有复杂度和数字示例都是推导或教学构造，不能替代目标模型、目标硬件上的实测。

## 4.1 Transformer 解决了什么问题

### 从递归到全局信息路由

RNN、LSTM 和 GRU 按时间步递归更新状态。递归结构有明确的顺序归纳偏置，但一段长度为 T 的序列需要经过 T 次依赖前一步的计算，训练并行度受到限制；远距离信息还必须穿过许多步状态传递。

Transformer 让每个位置通过 attention 直接读取其他可见位置的表示。训练时，整段序列可以同时计算 Q、K、V 和 attention score，序列维度上的主要矩阵运算可以并行执行。它并没有消除顺序问题，而是把顺序信息交给位置表示和 causal mask 处理。

### 一个 block 的信息流

以常见的 Pre-LN decoder block 为例：

~~~math
\begin{aligned}
u_l &= x_l+\operatorname{Attention}_l(\operatorname{Norm}_l(x_l)),\\
x_{l+1} &= u_l+\operatorname{FFN}_l(\operatorname{Norm}'_l(u_l)).
\end{aligned}
~~~

x_l 是第 l 层输入，u_l 是 attention 残差后的中间表示。attention 让位置之间交换信息，FFN 对每个位置分别做非线性变换，Norm 控制表示尺度，residual 让主路径近似恒等映射。不同模型可能使用 RMSNorm、LayerNorm、SwiGLU、GQA 或不同的残差缩放，但这条主线仍然适合作为形状和功能的第一张地图。

### 为什么适合规模化

Transformer 的矩阵乘法规则整齐，适合 GPU/TPU 的批量计算；层可以重复堆叠，宽度、深度和训练 token 可以独立扩展；attention 能动态选择上下文，FFN 提供大量逐 token 参数容量。代价也同样明确：标准 full attention 的序列长度成本近似平方增长，推理时 KV cache 线性占用显存，模型还需要大量数据才能学习顺序和语义规律。

### 从输入到 loss

decoder-only causal LM 的简化链路是：

~~~text
input_ids [B,T]
 -> token embedding [B,T,d]
 -> position-aware hidden [B,T,d]
 -> L 个 Transformer blocks
 -> logits [B,T,V]
 -> labels 右移一位
 -> masked cross-entropy
~~~

B 是 batch size，T 是输入长度，d 是 hidden size，V 是词表大小。若任何一步的 T、mask 或 label shift 不一致，代码可能仍然产生一个 loss 数字，但这个数字不再对应预期的 next-token 任务。

### 架构问题的三种层次

看到“模型支持长上下文”时，至少要区分：

1. 接口和 runtime 是否接受这么多 token。
2. 位置表示和 attention 是否能处理这么长的序列。
3. 模型是否能检索、整合并推理远处的信息，同时承受可接受的计算和显存成本。

这三层不能用一个 context length 配置代替。后面的章节会分别讨论位置外推、长上下文评估和服务成本。

## 4.2 Token Embedding 与位置表示

### Embedding 是查表，不是数值编码

token id 是离散索引。给定词表大小 V 和隐藏宽度 d，embedding 矩阵 E 的形状是 [V,d]：

~~~math
h^{(0)}_t=E[x_t],
\qquad
E\in\mathbb{R}^{V\times d}.
~~~

id 为 7 并不意味着它在语义空间中靠近 id 为 8 的 token。连续向量的关系由训练得到。低频 token 的向量得到的更新较少，新增 special token 若没有训练数据也不会自动拥有所需语义。

### Weight tying

输出端通常要把 hidden state 投影到词表：

~~~math
z_t=h_tW_{\mathrm{lm}}^\top+b.
~~~

若 W_lm=E，输入 embedding 和输出 LM head 共享权重。这会减少约 Vd 个参数，并让输入和输出使用相同的 token 空间。共享不是“两个矩阵数值碰巧相同”，而是同一组参数的两条梯度路径。扩展词表、量化、LoRA 和 checkpoint 加载时必须检查共享关系是否仍然存在。

### 为什么需要位置

self-attention 只根据 Q/K 内容计算匹配。如果把一组输入向量整体换序，而没有额外位置信息，模型缺少区分顺序的直接来源。位置表示可以进入 embedding 相加、attention score，也可以通过旋转 Q/K 或递归状态间接表达。

常见路线包括：

~~~text
learned absolute position embedding
sinusoidal positional encoding
RoPE
ALiBi
relative position bias
local / block position
递归状态或状态更新中的顺序
~~~

“位置编码”是一个家族名，不同方法的输入位置、可外推性和 cache 语义并不相同。

### 正弦位置编码

原始 Transformer 使用固定的正弦和余弦函数。对位置 p 和维度索引 i，可以写成：

~~~math
\begin{aligned}
PE(p,2i)&=\sin\left(p/10000^{2i/d}\right),\\
PE(p,2i+1)&=\cos\left(p/10000^{2i/d}\right).
\end{aligned}
~~~

不同维度使用不同频率，因此位置向量携带多尺度周期信息。它没有可训练的位置参数，形式上可以计算训练长度之外的位置；但能计算不等于模型在更长位置上已经学会稳定使用，训练数据和 attention 分布仍然重要。

### RoPE

RoPE 把二维子空间中的 Q、K 按位置旋转。若一个二维子向量为 q，位置 p 对应旋转矩阵 R(θ_p)，则：

~~~math
q_p=R(\theta_p)q,
\qquad
k_q=R(\theta_q)k.
~~~

旋转后的点积具有相对相位信息，直观上能让 q_p 与 k_q 的匹配依赖位置差。RoPE 通常作用于 Q 和 K，而不是把一个位置向量直接加到 embedding 上。base、旋转维度、频率和 position offset 都是 checkpoint 契约的一部分。

### RoPE scaling 与外推

训练长度为 T_train 时，直接把推理位置扩大到远高于 T_train，可能导致旋转相位分布偏离训练范围。位置插值、频率调整、分段 scaling 和长上下文继续训练都试图缓解这一问题，但各有短上下文、长上下文、数值和实现代价。

RoPE scaling 只解决位置映射的一部分问题，不会自动解决：

~~~text
attention 的平方级 prefill 成本
KV cache 显存
训练数据是否真的包含长序列
远端证据是否能被检索
跨段推理是否稳定
~~~

### ALiBi

ALiBi 把与相对距离有关的线性偏置直接加到 attention logits。不同 head 可以使用不同斜率，远距离位置受到不同程度的惩罚。它不需要显式的位置 embedding，形式简单，但线性距离偏置本身是一种归纳假设，质量和外推表现应在目标模型上验证。

### 位置表示的工程账本

换位置方案时，要同时记录：

1. 它作用在 embedding、Q/K 还是 score。
2. 训练最大长度和推理最大长度。
3. prefill 与 decode 的 position id 生成方式。
4. KV cache 中历史位置的 offset。
5. padding 后 position id 是否仍正确。
6. scaling 参数和 checkpoint 是否一致。

只修改 max_position_embeddings 或服务配置，不能证明模型已获得新的长度能力。

## 4.3 Self-Attention：Q、K、V 如何完成信息路由

### 直觉

对当前位置来说，query 表示“我想找什么”；key 表示“我可以如何被匹配”；value 表示“如果被选中，我提供什么内容”。这只是帮助初学者建立概念的比喻，真实 Q/K/V 都是由隐藏向量和可训练矩阵产生的：

~~~math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
~~~

如果 X 的形状是 [B,T,d]，单头 Q、K、V 的最后一维可以是 d_h。多头实现会把 d 拆成 H 个子空间。

### Scaled Dot-Product Attention

标准公式为：

~~~math
A(Q,K,V)
=\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt{d_h}}+M
\right)V.
~~~

Q 的形状是 [B,H,T_q,d_h]，K/V 的形状是 [B,H,T_k,d_h]。QK^T 的结果是 [B,H,T_q,T_k]。M 是 mask 或位置偏置。softmax 在 key 维度上归一化，最后把 V 按权重相加。

### 为什么除以 sqrt(d_h)

若 Q 和 K 的各维近似独立、均值为零、方差为一，点积的方差大致随 d_h 增长。d_h 较大时，未经缩放的 score 可能让 softmax 过早饱和，几乎只选择一个位置，梯度变得不稳定。除以 sqrt(d_h) 把分数尺度拉回较可控范围。

这是一种统计近似，不意味着所有实际激活都满足独立同分布假设。低精度、归一化、量化和特殊 attention 变体都可能改变 score 的统计，需要实际监控。

### Attention weight

softmax 后的权重通常满足每个 query 行和约为 1：

~~~math
\sum_{s=1}^{T_k}P_{t,s}=1.
~~~

权重大的位置表示当前 head 在这一层对相应 value 做了更大的聚合，但它不等于模型最终答案的完整因果解释。后续 MLP、残差、其他层和不同 head 都会改变信息路径。

### 计算与存储复杂度

对长度 T 的 full attention，score 矩阵含 T^2 个位置对。若忽略 batch 和常数，计算和中间存储的关键项分别近似为：

~~~math
\operatorname{cost}_{\mathrm{score}}=O(T^2d_h),
\qquad
\operatorname{memory}_{\mathrm{score}}=O(T^2).
~~~

实际 kernel 可能不显式存完整 score 矩阵，但数学依赖关系仍然存在；FlashAttention 减少 IO 和中间存储，不会把精确 full attention 的连接数变成线性。

### 一个可手算的 attention 例子

设一个 query 对三个 key 的未归一化 score 为 [1, 0, -1]，softmax 权重为：

~~~math
p_i=\frac{\exp(s_i)}{\exp(1)+\exp(0)+\exp(-1)}.
~~~

若三个 value 是标量 [10, 0, 4]，输出就是 p_1×10+p_2×0+p_3×4。这个例子只展示“score 决定权重、value 决定内容”，不能据此解释真实模型的语义。

## 4.4 Causal Mask、Padding Mask 与信息边界

### Causal mask

decoder-only causal LM 在位置 t 预测下一个 token 时，只允许读取位置 s≤t 的信息。用一个 T×T 的允许矩阵表示：

~~~math
C_{t,s}=
\begin{cases}
0,&s\le t,\\
-\infty,&s>t.
\end{cases}
~~~

把 C 加到 score 后再 softmax，未来位置的概率就会被压到零。mask 必须在 softmax 之前加入；先 softmax 再把结果置零会破坏归一化，除非重新归一化。

### Padding mask

padding mask 处理的是 batch 对齐带来的无效 token。若样本被补到长度 T，padding key 位置不应被其他 query 读取；某些实现还需要处理 padding query 的输出和 loss mask。causal mask 和 padding mask 解决不同问题，通常需要组合：

~~~math
M=C+P.
~~~

实际实现可能使用 bool mask、加性 mask 或 fused kernel 的不同 API，必须确认其 True/False 语义，不能只凭变量名猜。

### 左 padding 与右 padding

右 padding 让有效 token 在左侧，左 padding 让有效 token 对齐序列末端。两者都可以工作，但 position id、最后一个有效位置、生成 logits 和 KV cache 必须配套。许多 decoder-only 批量生成实现偏好左 padding，因为不同长度样本的最后一个有效 token 可以对齐；这不是普遍定律，应以具体框架接口为准。

### 全屏蔽行

若某个 query 的所有 key 都被 mask，softmax 的输入可能全为负无穷，产生 NaN。这个错误经常被误认为是低精度随机不稳定。调试时要统计每行有效 key 数，并为全屏蔽情况定义明确行为。

### mask 的反事实测试

一个比肉眼查看矩阵更强的测试是未来信息泄漏实验：

1. 固定模型参数、dropout 和随机状态。
2. 构造两份只在未来位置不同的输入。
3. 比较更早位置的 logits。
4. 若早期 logits 也变化，检查 causal mask、position id 或缓存路径。

同样，可以把一个 padding 位置的 token 替换成明显不同的内容，检查有效 query 的输出是否变化。测试应同时覆盖训练 full sequence 和 decode query length 为 1 的路径。

### 纯 Python 生成一个 causal mask

~~~python
def causal_mask(length):
    if (isinstance(length, bool) or not isinstance(length, int)
            or length < 0):
        raise ValueError("length must be a non-negative integer")
    return [
        [s <= t for s in range(length)]
        for t in range(length)
    ]


for row in causal_mask(3):
    print(row)
~~~

预期三行表示每个位置能看到自己和之前的位置。这个布尔矩阵没有 batch、head、padding 和 dtype 语义，不能直接当作高性能 attention mask 使用。

## 4.5 MHA、MQA、GQA 与 MLA：改变 K/V 的共享关系

### Multi-Head Attention

MHA 把 hidden width 拆成 H 个 query head、H 个 key head 和 H 个 value head。若 d=H d_h：

~~~text
X              [B,T,d]
Q,K,V projection
Q,K,V reshape  [B,H,T,d_h]
attention      [B,H,T,d_h]
merge          [B,T,d]
~~~

多头让模型在不同子空间同时建立信息路由，但每个 head 都保存历史 K/V，decode 时缓存较大。

### Multi-Query Attention

MQA 保留 H_q 个 query head，但只使用一个 K head 和一个 V head。所有 query head 共享 K/V，KV cache 的 head 维从 H_q 降到 1。它节省显存和历史 K/V 读取，代价是 K/V 表达容量减少，是否损失质量需要训练实验确认。

### Grouped-Query Attention

GQA 介于 MHA 和 MQA 之间。H_q 个 query head 被划成 G 组，每组共享一个 K/V head，通常 H_q 是 G 的整数倍。每个 K/V head 被若干 query head 复用：

~~~math
r=\frac{H_q}{H_{kv}}.
~~~

r 是每个 K/V head 服务的 query head 数。KV cache 和 decode 读取压力大致按 H_kv 而不是 H_q 增长，实际速度还取决于 kernel、内存布局和 batch。

### KV cache 的内存估算

忽略量化和分页，一个层的 K/V cache 元素数近似为：

~~~math
N_{\mathrm{KV}}
=2B T H_{kv}d_h.
~~~

若有 L 层、每元素 b 字节：

~~~math
M_{\mathrm{KV}}
\approx 2LBT H_{kv}d_h b.
~~~

前面的 2 是 K 和 V 两份。GQA 把 H_kv 降低，可以直接减少这部分显存；它并不减少 query 投影的所有计算，也不自动减少 MLP 或输出层成本。

### MLA 与低维 latent cache

一些架构把 K/V 信息压缩到低维 latent，再在 attention 中恢复或组合，以减少 cache 体积。低维缓存可能换来更复杂的投影、数值误差、kernel 和训练约束。不能把“cache 元素更少”直接等价为“端到端延迟一定更低”，还要测 decode 访存、投影 FLOPs 和 kernel 融合。

### 选择 attention 形式的变量

架构比较至少记录：

~~~text
H_q、H_kv、d_h
训练与推理的 batch
prefill / decode 的序列长度
KV cache dtype 或量化方式
kernel 是否支持 GQA / MLA
质量变化与长上下文分桶
实际显存和 P95 延迟
~~~

MHA、MQA、GQA、MLA 是不同的设计选择，不存在脱离模型规模和硬件的单一排序。

## 4.6 KV Cache、Prefill 与 Decode

### 为什么缓存有效

生成第 t+1 个 token 时，历史 token 的 K/V 在模型参数和历史输入不变的情况下不会改变。若每一步都重新计算完整前缀，会重复计算历史投影。KV cache 保存每一层历史 K/V，使新 query 只需读取缓存并追加当前 token 的 K/V。

### 两个阶段

Prefill 处理整段输入 prompt，通常 query 长度和 key 长度都较大，主要压力是矩阵计算和显存带宽。Decode 每次生成一个或少量 token，query 长度很小，但需要读取越来越长的 K/V，主要压力逐渐转向访存和调度。

~~~text
prefill:
Q [B,H,T_prompt,d_h]
K/V [B,H,T_prompt,d_h]

decode:
new Q [B,H,1,d_h]
cached K/V [B,H,T_cache,d_h]
new K/V appended to cache
~~~

TTFT 主要受 prefill 影响；TPOT、ITL 和吞吐则受到 decode、cache、batch 调度和输出长度共同影响。

### cache 与全量重算对齐

正确实现应满足：对同一前缀，prefill 后逐 token decode 得到的 logits，与每一步把完整前缀重新送入模型得到的 logits 在允许的浮点误差内一致。若不一致，优先检查：

1. RoPE position offset。
2. query/key 的 mask 形状。
3. 新 K/V 的拼接顺序。
4. padding 和 position id。
5. cache 是否跨请求污染。
6. 低精度或 fused kernel 的误差阈值。

### cache 生命周期

服务端还要处理请求取消、最大长度、beam 分支、speculative decoding、prefix reuse、分页 block 和不同租户的隔离。cache 是请求状态，不是永久模型参数；复用时必须确认前缀 token、模型版本、LoRA adapter、位置偏移和安全上下文都一致。

### 一个 cache 内存估算函数

~~~python
def kv_cache_bytes(
    layers,
    batch,
    tokens,
    kv_heads,
    head_dim,
    bytes_per_value,
):
    values = (layers, batch, tokens, kv_heads, head_dim, bytes_per_value)
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value <= 0 for value in values):
        raise ValueError("cache dimensions and dtype bytes must be positive integers")
    values = 2 * layers * batch * tokens * kv_heads * head_dim
    return values * bytes_per_value


print(kv_cache_bytes(32, 4, 8192, 8, 128, 2) / 1024**3)
~~~

输出约为 4.0 GiB。这个结果只计算 K/V 元素，不包括 allocator、workspace、权重、激活、分页碎片和并发请求；它的作用是建立量级直觉。

## 4.7 Full、Local、Sparse、Linear 与 FlashAttention

### Full attention

Full attention 允许每个 query 访问所有可见 key，信息路径最直接，但 score 连接数量随 T² 增长。对于需要跨段推理的任务，它提供充分的连接；对于极长序列，计算和内存成本成为主要限制。

### Local 或 sliding-window attention

Local attention 只让 query 访问邻近窗口。若窗口宽度为 w，理论连接数从 T² 降到约 Tw。它适合局部语法、音频帧或长文档的局部关联，但远端信息需要通过层间传播、全局 token、摘要或外部检索才能到达。

窗口并不自动等价于“长上下文理解”。若一个证据和问题距离超过所有可达路径，模型可能无法直接整合它。

### Sparse attention

Sparse attention 只保留预先设计的连接模式，可以是 block、global、strided、random 或 local/global 混合。理论上减少连接并不保证 GPU 实际加速；稀疏索引、kernel、负载均衡和训练框架支持都可能消耗收益。

### Linear attention

Linear attention 试图利用特征映射或计算重排，把显式的 QK^T 连接改成先聚合 K/V，再与 Q 交互。抽象地说，若 softmax kernel 可以近似为 φ(q)^Tφ(k)，则：

~~~math
\sum_j
\phi(q)^\top\phi(k_j)v_j^\top
=
\phi(q)^\top
\left(\sum_j\phi(k_j)v_j^\top\right).
~~~

括号内的状态可以沿序列递推，避免显式保存 T×T 矩阵。代价是它通常改变了 softmax attention 的精确形式，表达、数值、归一化和硬件效率需要单独评估。

### FlashAttention

FlashAttention 是 exact attention 的 IO-aware 实现。它通过 tiling、在线 softmax 和分块重计算，减少 HBM 与片上存储之间的往返和中间矩阵写入。它改变的是计算组织，而不是把 attention 连接模式变成稀疏或把 softmax 近似成另一种函数。

因此要区分：

~~~text
FlashAttention：精确 attention 的 kernel / IO 优化
Sparse attention：改变可计算的连接集合
Linear attention：改变或重排 attention 的数学形式
KV cache：减少 decode 时的历史重复计算
~~~

不同版本对 mask、dropout、变长序列、GQA 和硬件的支持不同，实际收益必须在目标形状和目标 GPU 上测量。

### Prefill 与 attention 优化的关系

FlashAttention 对长序列 prefill 往往有明显帮助；decode 的主要瓶颈可能是读取 KV cache 和请求调度，此时单纯优化 prefill kernel 不会等比例降低 TPOT。部署评估要把 prefill、decode、并发和输出长度分开报告。

## 4.8 Transformer Block：attention、FFN、Norm 和 residual 的组合

### Pre-LN 与 Post-LN

Pre-LN 的一个形式是：

~~~math
y=x+F(\operatorname{Norm}(x)).
~~~

Post-LN 的一个形式是：

~~~math
y=\operatorname{Norm}(x+F(x)).
~~~

Pre-LN 让残差主路径更接近恒等映射，通常更容易训练深层模型；Post-LN 是原始 Transformer 中的常见形式，在特定初始化和训练设置下也可以工作。两者的差异不只是代码位置，还会影响梯度、初始化、最终 norm 和学习率策略。

### FFN 的作用

Attention 把其他位置的信息引入当前位置，FFN 再在当前位置内部做非线性变换。普通 FFN 可以写成：

~~~math
\operatorname{FFN}(x)
=W_2\phi(W_1x+b_1)+b_2.
~~~

SwiGLU 等门控 FFN 使用多个投影分支，增加特征选择能力和参数容量。很多 LLM 的 FFN 参数量大于 attention 投影，因此只优化 attention 不能代表优化了整个模型。

### 残差尺度

深层 block 中，许多分支输出会反复叠加。初始化、Norm、residual scaling 和层深共同决定激活方差。只要一个分支长期比主路径大很多，就可能造成训练不稳；如果分支几乎为零，模型容量又没有被有效利用。监控每层 residual 分支范数和主路径范数，比只看总 loss 更容易发现问题。

### Decoder block 的形状账本

~~~text
x                    [B,T,d]
norm(x)              [B,T,d]
Q,K,V                 [B,H,T,d_h] or GQA variant
attention output      [B,T,d]
residual              [B,T,d]
norm                   [B,T,d]
FFN intermediate       [B,T,d_ff]
FFN output             [B,T,d]
block output           [B,T,d]
~~~

任何 residual add 都必须形状兼容；GQA 只改变 K/V head 关系，不改变最终 residual 的 d。

## 4.9 三种 Transformer 形态与 Cross-Attention

### Decoder-only

Decoder-only 使用 causal self-attention，从左到右预测下一个 token。GPT、LLaMA、Qwen 等生成式 LLM 属于这一大类。训练和推理目标一致，输入、工具结果、检索证据和答案都可以序列化到一条 token 流中。

它的局限是每个位置只能读取左侧上下文；若目标是双向文本表示或固定长度分类，专门的 encoder 可能更直接。

### Encoder-only

Encoder-only 使用双向 self-attention，输入中的每个位置可以读取左右上下文。BERT、RoBERTa、DeBERTa 等常用于分类、抽取、匹配和 embedding。Masked LM 等预训练目标让它学到上下文表示，但它不天然提供从左到右生成长文本的路径。

### Encoder-decoder

Encoder 先双向编码输入，decoder 使用 causal self-attention 生成输出，并用 cross-attention 读取 encoder 表示。翻译、摘要和明确的输入到输出转换任务可以自然地使用这种结构。它的两个序列长度和两套 mask 使工程更复杂，但输入理解与输出生成的分工清晰。

### Cross-attention 的公式

若 decoder query 为 Q_d，encoder 输出产生 K_e、V_e：

~~~math
\operatorname{CrossAttn}
=\operatorname{softmax}
\left(\frac{Q_dK_e^\top}{\sqrt{d_h}}+M_{de}\right)V_e.
~~~

self-attention 的 Q/K/V来自同一序列；cross-attention 的 Q 来自生成端，K/V 来自条件端。多模态模型可以让文本 query 读取图像 encoder 特征，检索系统也可以把外部表示作为条件，但具体架构必须区分“真正的 cross-attention”和“把证据文本拼到 prompt 中”。

### 混合架构的判断

当一个模型同时包含 full attention、local attention、递归 state、cross-attention 或外部 memory 时，要画出信息路径和状态生命周期。不要只按“Transformer”或“非 Transformer”二分；关键问题是每个 token 如何读取历史、哪些状态会被缓存、状态是否可精确检索，以及训练和推理是否使用同一条路径。

## 4.10 长上下文：窗口、位置和有效利用

### 能接收与能使用

最大上下文长度是接口、模型配置、训练分布和 runtime 共同形成的边界。能接受一个很长请求，只能说明请求没有在 tokenizer、协议、显存或 runtime 层被拒绝。有效使用还要回答：

~~~text
远端证据能否被找到
多个证据能否跨段合并
冲突证据能否被识别
中间位置是否被忽略
增加长度后的延迟和成本是否可接受
~~~

### Position extrapolation

训练长度之外，位置编码可能进入未见过的相位或距离范围。RoPE scaling、位置插值、长上下文继续训练、局部窗口和外部检索都可能帮助，但它们解决的问题不同。位置编码修补不能保证模型会进行跨段推理，检索增强也不能自动修复错误的 position id。

### Lost in the middle

长上下文任务中，关键信息位于开头或结尾时，模型有时比关键信息位于中间时表现更好。评估应轮换证据位置，并控制文档内容、问题、token 长度和干扰信息。只在首尾放一个 needle，不能代表完整长上下文能力。

### 长上下文的分层评估

建议分别测：

1. 单点位置检索。
2. 多点证据合并。
3. 跨段因果链。
4. 冲突和版本判断。
5. 长文本摘要的覆盖率。
6. 输入长度、TTFT、TPOT、显存和单位成功成本。

质量曲线和成本曲线要同时画。一个长度翻倍但质量不变的系统，可能已经因为 prefill、cache 和并发下降而不适合生产。

## 4.11 递归状态、Delta 变体与 NoPE 设计

### 为什么出现递归状态

标准 attention 显式保留历史 token 的 K/V，长度增加时 cache 线性增长；递归状态架构试图把历史压缩到固定或较小的状态中。抽象形式可以写成：

~~~math
S_t=f_\theta(S_{t-1},x_t),
\qquad
y_t=g_\theta(S_t,x_t).
~~~

这类状态像一个持续更新的记忆，而不是可逐项寻址的完整 token 表。它可能降低长序列的内存和计算，但也可能丢失精确的远端细节。

### Delta rule 与 gated update

一种抽象的 delta 更新是：

~~~math
S_t
=\alpha_t\odot S_{t-1}
+u_t v_t^\top
-\Delta_t.
~~~

α_t 可以控制保留历史的程度，u_t v_t^T 是低秩写入，Δ_t 表示基于当前输入的修正。不同论文对状态、门和值的定义不同，不能把一个示意式当作所有 DeltaNet/KDA 实现的精确公式。

### KDA、Gated DeltaNet 与混合注意力

近期架构把 gated delta state、局部或全局 attention、MLA 等组合起来，目标通常是在表达能力、长序列效率和 cache 体积之间寻找折中。Kimi Linear 论文中的 Kimi Delta Attention 是一个具体实例；Gated DeltaNet 则代表另一类门控递归状态设计。阅读这些模型时，应逐层记录：

~~~text
状态 S 的 shape 和 dtype
状态是否按请求隔离
是否支持精确 token 级检索
prefill 和 decode 的更新顺序
状态重置、截断和分支复制
训练时的并行扫描方式
目标硬件上的 kernel 和吞吐
~~~

“cache 更小”不等于“信息容量相同”；递归状态可能更适合流式和长序列，也可能在精确引用、随机访问或复杂多跳任务上出现不同错误。

### NoPE 的含义

NoPE 通常表示某个路径不显式使用 RoPE、ALiBi 或其他传统位置注入。它不表示模型完全没有顺序信息。causal mask 提供可见性方向，递归更新的先后、局部窗口、内容模式、训练数据和 token 边界也能携带顺序线索。但 causal mask 单独不能表示任意距离关系，因此“NoPE”应理解为一种架构设计标签，而不是长上下文能力结论。

### 证据边界

对于 KDA、Gated DeltaNet、Gated MLA 或 NoPE 这类较新的名称，应以原始论文、模型卡和实现代码核对，不要从二手新闻中的一句“无需 RoPE”推断：

1. 所有层都不使用位置表示。
2. 模型不需要 position id。
3. 可以无限上下文。
4. 精确保留所有历史信息。
5. 一定比 attention 更快或更强。

架构名描述一个机制局部，能力结论需要目标模型和任务上的实测。

## 4.12 Mixture-of-Experts：总参数与每 token 激活参数

### 稀疏路由

MoE 把一个 FFN 替换为多个 expert，并由 router 为每个 token 选择 top-k expert。若共有 E 个 expert，每个 expert 参数量为 P_e，共享部分参数为 P_shared，其他参数为 P_r，则总参数可粗略写成：

~~~math
P_{\mathrm{total}}
=P_{\mathrm{shared}}+EP_e+P_r.
~~~

若每个 token 只激活 k 个 expert，单 token 的激活参数近似为：

~~~math
P_{\mathrm{active}}
=P_{\mathrm{shared}}+kP_e+P_r.
~~~

P_active 不是模型实际延迟的完整公式。路由、token dispatch、跨设备通信、expert 负载不均衡、KV cache、kernel 和权重加载都会影响服务成本。

### Router 与负载均衡

router 可以根据 hidden state 产生 expert logits，再选择 top-k。若所有 token 都集中到少数 expert，其他 expert 空闲，吞吐和训练稳定性都会下降。容量因子、辅助负载损失、token dropping 或 shared expert 都是常见处理，但每种机制都有质量或通信代价。

### MoE 的误读

总参数多不代表每次计算都使用全部参数；active 参数少也不代表显存只需要存 active 参数。部署时通常仍需存储或按需加载多个 expert，通信和权重驻留策略会改变实际成本。比较 dense 与 MoE，要同时报告：

~~~text
total parameters
active parameters per token
expert 数与 top-k
路由负载分布
通信量
权重显存
KV/state cache
tokens/s 和 P95
质量与长尾任务分桶
~~~

## 4.13 复杂度、验证与资料边界

### 一张成本对照表

~~~text
Full attention       连接充分，score 关系约 O(T²)
Local attention      约 O(Tw)，远端路径需要补充机制
Linear/state model   目标接近 O(T)，但数学形式或状态容量改变
FlashAttention       exact attention，主要优化 IO 和中间存储
MQA/GQA/MLA          减少或压缩 K/V cache，影响投影和质量
KV Cache             避免 decode 重算历史，换来显存和访存
MoE                  增加总参数，按 token 稀疏激活，带来路由通信
~~~

这些是复杂度主项，不是端到端 benchmark。真实结果还取决于 batch、序列长度、dtype、kernel、硬件、并发和调度。

### 组件级验证

Transformer 实现至少需要以下测试：

~~~text
Embedding 与 tokenizer 的 vocab size 对齐
position id 和 RoPE offset 对齐
causal mask 阻止未来信息
padding mask 不读取无效 token
attention weight 每行归一化且 finite
MHA reshape 后元素顺序正确
GQA 的 query/KV head 映射正确
KV cache decode 与全量重算对齐
weight tying 确实共享参数
MoE router 的 top-k 和容量约束正确
~~~

这些测试证明实现语义，不证明模型已经具备语言能力。能力评估要另用任务数据、分桶指标、失败样例和成本测量。

### 形状与参数账本

~~~python
def attention_shapes(batch, tokens, model_dim, query_heads, kv_heads):
    values = (batch, tokens, model_dim, query_heads, kv_heads)
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value <= 0 for value in values):
        raise ValueError("shape dimensions must be positive integers")
    if model_dim % query_heads != 0:
        raise ValueError("query_heads must divide model_dim")
    if query_heads % kv_heads != 0:
        raise ValueError("kv_heads must divide query_heads")
    head_dim = model_dim // query_heads
    return {
        "hidden": (batch, tokens, model_dim),
        "q": (batch, query_heads, tokens, head_dim),
        "k": (batch, kv_heads, tokens, head_dim),
        "v": (batch, kv_heads, tokens, head_dim),
        "group_size": query_heads // kv_heads,
    }


print(attention_shapes(2, 16, 64, 8, 2))
~~~

这个片段只验证 MHA/GQA 的形状关系，不执行 attention。它也没有验证 kernel 是否真的按照 group_size 复制或广播 K/V，真实实现仍需数值对齐测试。

### 资料与证据边界

建议优先阅读以下资料：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 原始架构。
- [RoFormer](https://arxiv.org/abs/2104.09864)：RoPE。
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409)：ALiBi。
- [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)：MQA。
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)：GQA。
- [FlashAttention](https://arxiv.org/abs/2205.14135) 与 [FlashAttention-2](https://arxiv.org/abs/2307.08691)：IO-aware exact attention。
- [Longformer](https://arxiv.org/abs/2004.05150)：局部/全局稀疏 attention。
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)：线性 attention 路线。
- [Lost in the Middle](https://arxiv.org/abs/2307.03172)：长上下文位置偏差。
- [Retentive Network](https://arxiv.org/abs/2307.08621)：递归状态与并行/递归形式。
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)：门控 delta state。
- [Kimi Linear](https://arxiv.org/abs/2510.26620)：Kimi Delta Attention 与混合长序列架构。
- [PyTorch scaled dot product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：官方 attention 接口。
- [PyTorch MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)：官方多头接口。

论文中的实验结论、框架 API 的行为、教学代码的输出和目标模型在目标硬件上的吞吐，属于不同证据层。尤其是 KDA、Gated DeltaNet、Gated MLA、NoPE 等较新架构，必须核对具体论文和实现版本，不应从名称或一条宣传语推出通用能力。

## 本章回顾

理解 Transformer，应该能把一个 token 从输入走到输出，并在每一步写出形状和约束：

~~~text
token id
 -> embedding
 -> position representation
 -> Q/K/V
 -> score / scale / mask / softmax
 -> value aggregation
 -> MHA / GQA / MQA
 -> residual + norm + FFN
 -> logits
 -> shifted loss
~~~

进一步比较架构时，还要回答三个问题：历史信息是通过完整 KV、压缩 cache 还是递归 state 保存的；信息连接是 full、local、sparse、linear 还是混合的；总参数、每 token 激活参数、显存、通信和质量如何共同决定成本。Transformer 的名字只是入口，真正需要理解的是信息从哪里来、沿什么路径传播、在哪里被截断，以及这些路径在目标任务和目标硬件上是否得到证据支持。
