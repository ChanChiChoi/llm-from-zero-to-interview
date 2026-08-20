# 第二部分：Transformer 架构进阶

Transformer 的关键不在于记住一张组件清单，而在于理解信息如何流动、计算为什么变贵、哪些近似会改变模型行为，以及架构选择如何传导到训练和部署。

本部分把一个 decoder-only Transformer 拆成一条完整链路：

1. Self-Attention 如何在 token 之间做内容寻址和信息路由；
2. 为什么 dense attention 出现平方复杂度，以及算法近似与 kernel 优化的区别；
3. residual、normalization 和深度如何共同决定训练稳定性；
4. FFN、门控 MLP 和 MoE 如何提供特征容量与条件计算；
5. RoPE、ALiBi 和相对位置 bias 如何影响长度泛化；
6. MHA、MQA、GQA 如何改变 KV Cache 和解码带宽；
7. encoder-only、decoder-only、encoder-decoder、SSM 与混合架构分别适合什么问题；
8. 模型尺寸、硬件整除、并行方式和真实模型配置如何统一分析。

贯穿全章的一个区分是：理论复杂度、实际 kernel 时间、模型质量和单位成功任务成本不是同一个指标。把 O(T²) 写成 O(T) 不代表 GPU 一定更快；把最大上下文窗口写大不代表模型会使用远处证据；把总参数量做大不代表每个 token 的激活计算也同比增长。每个主题都要把机制、数量关系、适用条件、反例和评估方法放在一起。

## 11. Self-Attention：从相关性计算到动态信息路由

### 11.1 一个 token 怎样读取其他 token

设一段序列的表示矩阵为 X∈R^(T×d)，T 是序列长度，d 是模型宽度。对一个 head，线性投影得到：

~~~math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
~~~

其中 Q、K、V∈R^(T×d_h)，d_h 是单个 head 的维度。注意力分数、归一化权重和输出分别为：

~~~math
S=\frac{QK^\mathsf T}{\sqrt{d_h}},
\qquad
A=\mathrm{softmax}_{\mathrm{row}}(S),
\qquad
O=AV
~~~

第 i 行的输出可以展开为：

~~~math
O_i=\sum_{j=1}^{T}A_{ij}V_j
~~~

A_ij 表示当前位置 i 从位置 j 读取多少 value。它不是固定卷积核，而是由当前输入内容决定的权重矩阵。

初学者可以把 Q 理解为“我现在想找什么”，把 K 理解为“我能被什么查询匹配”，把 V 理解为“匹配成功后真正提供的内容”。这三个比喻有助于读公式，但 Q/K/V 并不是人工规定的语义字段，而是通过训练学出来的投影空间。

### 11.2 为什么要除以平方根

如果 Q 和 K 的每个维度均值接近 0、方差接近 1，那么点积是 d_h 个乘积之和，其方差大致随 d_h 增长。维度越大，未经缩放的 logits 越容易进入 softmax 的饱和区。

softmax 饱和会让一两个位置获得几乎全部概率，其他位置梯度很小，训练早期尤其不稳定。除以 √d_h 的作用是把分数尺度拉回可学习范围，而不是人为规定“每个 head 只能看多少位置”。

数值实现还会在 softmax 前减去每行最大值：

~~~math
\mathrm{softmax}(s)_j
=\frac{\exp(s_j-\max_k s_k)}
{\sum_u\exp(s_u-\max_k s_k)}
~~~

这个变换不改变结果，却避免指数溢出。

### 11.3 Attention matrix 是动态路由图

A∈R^(T×T) 可以看成一张随输入变化的有向加权图：第 i 行是从查询位置 i 指向所有来源位置的边权。标准 dense attention 让所有位置对都有候选边；causal mask 则删除指向未来的边。

如果第 i 个 token 需要找到前文中某个实体、变量定义或约束，它可以通过 Q 与候选 K 的内容匹配，把权重集中到相关位置。与固定窗口卷积相比，attention 的连接模式依赖内容；与 RNN 相比，远处位置可以在一层内直接交互，不必经过逐时间步的隐藏状态传递。

这也解释了 attention 的代价：每个查询都要和 T 个 key 比较，天然产生 T×T 的交互表。

### 11.4 因果 mask 改变的是信息流

自回归模型在训练时一次处理整段序列，但位置 i 不能读取未来位置 j>i。设未加 mask 的分数为 S，则：

~~~math
\widetilde S_{ij}
=\begin{cases}
S_{ij},&j\le i\\
-\infty,&j>i
\end{cases}
~~~

softmax 后，未来位置的权重为零。mask 必须在 softmax 之前施加；若先 softmax 再把权重清零而不重新归一化，概率质量会丢失，若实现错误地让未来权重参与归一化，训练就会发生信息泄漏。

一个反事实测试很有价值：在同一前缀中只改变未来 token，检查当前位置的 hidden state 和 logits 是否完全不变。如果会变，mask、position id 或缓存实现至少有一处不符合因果协议。

### 11.5 Multi-Head 不只是重复计算

多头注意力把模型宽度切成多个子空间：

~~~math
\mathrm{head}_h
=\mathrm{Attention}(XW_Q^{(h)},XW_K^{(h)},XW_V^{(h)})
~~~

~~~math
\mathrm{MHA}(X)
=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W_O
~~~

不同 head 可以学习不同的匹配几何和信息路由：局部邻近、实体指代、代码分隔符、长程复制和格式约束都可能出现。但不能把这种可能性写成每个 head 都有清晰人类语义。head 会冗余、重组和随训练阶段变化。

### 11.6 Attention map 不是完整解释

attention 权重可以告诉我们某一层、某一个 head 在聚合 value 时使用了哪些权重，但它不能直接证明某个 token 对最终答案具有因果贡献，原因有四个：

1. V 已经经过线性投影，权重高不代表原文本被原样读取；
2. 多层多头会继续混合和重写表示；
3. MLP 与 residual stream 也能改变最终 logits；
4. 输出可能由多个路径共同造成。

若要研究因果作用，需要做遮挡或替换实验、activation patching、路径追踪，并检查输出变化是否稳定。解释性分析应当报告干预方法和反事实结果，而不是只展示一张热力图。

### 11.7 Attention 与外部检索的边界

把 attention 叫作“内部检索器”是有帮助的类比：Q 类似查询，K 类似索引，V 类似内容。但它的知识来源仍然受限于当前上下文和参数：

- attention 只能访问输入窗口中的 token 表示；
- RAG 可以从外部索引取回新文档；
- attention 的 key/value 是连续向量；
- RAG 通常还要处理文档来源、时间、权限和引用。

上下文没有某个事实时，attention 不能凭空把它检索出来。参数中可能存有相关知识，但那是生成模型的记忆路径，不是当前上下文检索。

### 11.8 Residual stream 连接各个子层

现代 Transformer 常把每个子层看成对主干表示的增量更新。以 Pre-Norm 为例：

~~~math
x_{l+1}=x_l+\mathrm{Attention}(\mathrm{Norm}(x_l))
~~~

~~~math
x_{l+2}=x_{l+1}+\mathrm{MLP}(\mathrm{Norm}(x_{l+1}))
~~~

attention 负责跨位置的信息混合，MLP 负责每个位置的特征变换，normalization 控制数值尺度，residual stream 则承载跨层累积的表示。只讲 attention 而不讲 residual、MLP 和 norm，会得到一个不完整的架构模型。

### 11.9 一个纯 Python 的单头实现

下面的代码用列表实现小规模 causal attention，适合手工检查每一步。它没有使用高效矩阵库，所以只用于教学。

~~~python
import math


def dot(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("dot product requires non-empty vectors of equal length")
    return sum(a * b for a, b in zip(left, right))


def softmax(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("softmax input must be non-empty and finite")
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("softmax normalizer must be positive and finite")
    return [weight / total for weight in weights]


def causal_attention(query, keys, values):
    if not query or len(query) != len(keys) or len(keys) != len(values):
        raise ValueError("query, keys and values must be non-empty and aligned")
    query_dim = len(query[0])
    value_dim = len(values[0])
    if query_dim == 0 or value_dim == 0:
        raise ValueError("query and value dimensions must be positive")
    if any(len(row) != query_dim for row in query + keys):
        raise ValueError("query and key dimensions must be consistent")
    if any(len(row) != value_dim for row in values):
        raise ValueError("value dimensions must be consistent")
    outputs = []
    routes = []
    scale = math.sqrt(query_dim)
    for i, current_query in enumerate(query):
        visible_scores = [dot(current_query, key) / scale for key in keys[: i + 1]]
        visible_weights = softmax(visible_scores)
        weights = visible_weights + [0.0] * (len(keys) - i - 1)
        output = [
            sum(weight * value[dimension] for weight, value in zip(weights, values))
            for dimension in range(len(values[0]))
        ]
        outputs.append(output)
        routes.append(weights)
    return outputs, routes


q = [[1.0, 0.0], [0.8, 0.2], [0.1, 0.9]]
k = [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]]
v = [[10.0, 0.0], [0.0, 10.0], [3.0, 7.0]]
outputs, routes = causal_attention(q, k, v)
print("last route:", [round(value, 4) for value in routes[-1]])
print("last output:", [round(value, 4) for value in outputs[-1]])
assert all(routes[i][j] < 1e-8 for i in range(3) for j in range(i + 1, 3))
~~~

最后一个断言验证了因果性。真正的 Transformer 还需要多头投影、batch 维、位置编码、残差、归一化和高效 kernel；这段代码只把“分数—mask—权重—value 聚合”这条主线显式化。

### 11.10 分层理解

初学者先回答：Q/K 决定从哪里读，V 决定读到什么，softmax 形成读取比例，causal mask 限制可见范围。

进阶学习者要继续追问：

1. attention 权重的统计是否随层和 head 改变；
2. 高权重是否真的对应因果重要性；
3. full attention 的交互图是否必要；
4. KV Cache、位置编码和 kernel 如何改变同一个公式的工程成本。

## 12. Attention 的平方瓶颈：稀疏、线性与 IO 优化

### 12.1 复杂度从哪里来

单个 head 的 Q、K、V 形状为 T×d_h。计算 QK^T 和 AV 的主要乘法量分别约为：

~~~math
C_{QK}=O(T^2d_h),\qquad C_{AV}=O(T^2d_h)
~~~

H 个 head 合起来可近似写成 O(T²d_model)。Q/K/V 投影和输出投影是 O(Td_model²)，在短序列、大宽度模型中也可能占很大比例，但长上下文的独特瓶颈是 T²。

长度从 4K 变成 32K，T 增加 8 倍，交互数增加 64 倍。长度从 32K 变成 1M，交互数再增加约 976 倍。这个数量级足以说明：把窗口声明得很大，必须同时说明采用了什么注意力结构、训练方式和推理成本。

### 12.2 训练显存与推理显存

训练反向传播需要保存或重算中间值。若直接保存 attention logits 或 probabilities，形状通常是 [B,H,T,T]。单个 head、T=32768 时约有：

~~~math
32768^2=1,073,741,824
~~~

个元素。即使每个元素 2 字节，也约 2 GiB，还未计入 batch、多头、反向保存和临时 buffer。FlashAttention 通过分块和在线 softmax 避免把完整矩阵写回高层显存，但并没有让精确 dense attention 的数学交互数消失。

自回归解码有另一种瓶颈。使用 KV Cache 后，第 t 步的新 query 仍需读取前 t 个 key/value，单步读量随 t 增长；缓存本身大致需要：

~~~math
M_{\mathrm{KV}}
=2BLT H_{\mathrm{KV}}d_h s
~~~

其中 B 是 batch，L 是层数，T 是缓存长度，H_KV 是 K/V head 数，d_h 是 head dimension，s 是每个元素的字节数，2 表示 K 与 V。GQA/MQA 会直接缩小 H_KV，分页管理则减少碎片和调度浪费。

### 12.3 局部和滑动窗口注意力

如果每个位置只关注 w 个邻近位置，交互量近似为：

~~~math
C_{\mathrm{window}}=O(Twd_h)
~~~

当 w 固定且远小于 T 时，它从平方增长变为线性增长。代价是远处信息不能在一层直接交互，必须通过多层传播、全局 token、摘要或外部检索传递。

局部窗口适合局部依赖强、流式处理或可以接受有限记忆的任务；对跨文档精确引用、长距离变量绑定和多段证据合并，需要额外的全局路径。

### 12.4 Block sparse 和 global token

完全随机的稀疏模式不一定能被 GPU 高效执行，因此实际设计常把保留的连接组织成块。典型模式包括局部块、跨块稀疏连接、全局 token 和少量随机边。

global token 的直觉是：局部 token 负责细节，全局位置负责汇聚和广播。它能缩短某些信息传播路径，却也会产生汇聚瓶颈：如果所有证据都压到少数全局位置，容量不足时可能损失细节。

Longformer、BigBird 等路线说明，稀疏模式既是算法归纳偏置，也是硬件布局。需要同时报告理论保留的边、实际 kernel 是否利用稀疏、以及任务质量是否下降。

### 12.5 低秩和线性 attention

若认为 attention 交互矩阵存在可压缩结构，可以用低秩投影、landmark token 或 kernel feature map 减少显式 T×T 计算。线性 attention 的抽象形式是：

~~~math
\mathrm{Attention}(Q,K,V)
\approx\phi(Q)\left(\phi(K)^\mathsf TV\right)
~~~

它尝试先聚合 K 与 V，再让 Q 查询聚合结果，从而避免显式构造完整交互矩阵。但 softmax attention 的归一化、因果前缀和数值稳定性并不会自动保留；任何近似都要验证检索、组合推理和语言建模质量。

### 12.6 FlashAttention 与稀疏 attention 必须分开

两者解决的层次不同：

| 路线 | 是否改变精确连接图 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| dense attention | 不改变 | 表达完整、生态成熟 | T² 交互 |
| sparse/window | 改变 | 减少实际连接 | 远程信息可能丢失、kernel 更难 |
| linear/低秩 | 近似或改写 | 降低复杂度阶数 | 可能改变 softmax 行为 |
| FlashAttention | 不改变 | 减少 IO 和中间矩阵写回 | 仍有 dense 交互，依赖硬件 kernel |

因此“FlashAttention 把 attention 变成线性复杂度”是错误的；更准确的描述是它对精确 attention 做 IO-aware 的分块实现。

### 12.7 加 mask 不等于省计算

如果实现仍然构造完整 T×T logits，只是把远处位置设为负无穷，数学结果可能是稀疏的，底层乘法却仍然是 dense。要真正降低计算，需要：

1. 让 kernel 只读取保留的块；
2. 使用专门的 sliding-window 或 block-sparse 实现；
3. 改变计算图，避免生成完整交互矩阵；
4. 检查端到端 profiler，而不是只看理论公式。

### 12.8 一个量级估算

~~~python
def gibibytes(elements, bytes_per_element=2):
    if elements < 0 or bytes_per_element <= 0:
        raise ValueError("elements must be non-negative and bytes must be positive")
    return elements * bytes_per_element / (1024 ** 3)


def attention_accounting(seq_len, heads=32, layers=32, kv_heads=8,
                         head_dim=128, window=4096):
    if any(value <= 0 for value in [seq_len, heads, layers, kv_heads, head_dim]):
        raise ValueError("sequence and model dimensions must be positive")
    if window <= 0:
        raise ValueError("window must be positive")
    if kv_heads > heads:
        raise ValueError("kv_heads cannot exceed query heads")
    dense_pairs = heads * seq_len * seq_len
    window_pairs = heads * seq_len * min(seq_len, window)
    kv_elements = 2 * layers * seq_len * kv_heads * head_dim
    return {
        "dense_pairs_per_layer": dense_pairs,
        "window_pairs_per_layer": window_pairs,
        "dense_score_gib_bf16": gibibytes(dense_pairs),
        "kv_cache_gib_bf16": gibibytes(kv_elements),
    }


for length in [4096, 32768]:
    result = attention_accounting(length)
    print("length =", length)
    for name, value in result.items():
        print(" ", name, "=", round(value, 4) if isinstance(value, float) else value)
~~~

这段代码只能做数量级估算，不能替代实际 profiler。真实显存还包括权重、激活、通信 buffer、allocator 碎片和并发请求。

### 12.9 评估长上下文优化

至少要把以下测试分开：

- 语言建模 loss 随长度的曲线；
- 远处单证据检索；
- 多个远处证据的合并；
- 中间位置证据，避免只测开头和结尾；
- 长代码中的跨文件依赖；
- 长对话中的约束保持；
- 首 token 延迟、每 token 延迟、吞吐和缓存占用。

needle-in-a-haystack 可以作为入口，但它只测一种检索形式，不能代表复杂长上下文推理。

## 13. Residual、Pre-LN、Post-LN 与 DeepNorm

### 13.1 归一化位置是信息流设计

一个残差子层可以抽象为：

~~~math
x_{l+1}=x_l+F_l(x_l)
~~~

把 norm 放在 F 的输入前或 residual add 后，会改变主干和子层的梯度路径：

~~~math
\mathrm{PreLN}(x)=x+F(\mathrm{Norm}(x))
~~~

~~~math
\mathrm{PostLN}(x)=\mathrm{Norm}(x+F(x))
~~~

这不是代码风格差异，而是改变了每层表示和梯度的递推关系。

### 13.2 为什么 Pre-LN 通常更稳

Pre-LN 保留了较直接的 identity path。即使子层 F 的梯度不理想，主干仍然可以把一部分梯度传过许多层。深层模型通常更容易使用较大的学习率，也较少依赖极精细的 warmup。

但 Pre-LN 的 residual stream 没有在每个 add 之后立即被中心化，层数增加时表示尺度仍需监控，输出前常见 final norm。训练稳定也不自动意味着下游质量更高。

### 13.3 Post-LN 的边界

Post-LN 让每层输出尺度被 norm 约束，在浅层模型和原始 Transformer 配置中可以工作；深层时，梯度要经过多层归一化变换，训练对初始化、学习率和 warmup 更敏感。它不是错误架构，而是深度扩展时工程裕量较小。

### 13.4 Sandwich-LN 和额外控制

Sandwich-LN 的一种抽象形式是：

~~~math
x_{l+1}=x_l+\mathrm{Norm}_2
\left(F_l(\mathrm{Norm}_1(x_l))\right)
~~~

它同时控制子层输入和输出，可能改善尺度漂移，却增加归一化算子、内存读写和结构复杂度。是否值得使用，要看融合 kernel、训练稳定性和质量收益。

### 13.5 DeepNorm 与 residual scaling

若每层都加入增量 Δx_l，则：

~~~math
x_L=x_0+\sum_{l=1}^{L}\Delta x_l
~~~

深度变大时，增量累积可能使表示或梯度失控。DeepNorm/DeepNet 的核心思想是让 residual 缩放和初始化随深度配套设计，从而控制累积效应。应把它理解为一组架构与初始化规则，而不是简单在某一行代码乘一个常数。

论文中的具体系数依赖 encoder/decoder、层数和残差定义。移植时需要核对公式、参数初始化、学习率和最终 norm，不能只复制一个 α。

### 13.6 训练监控

深层模型除了总 loss，还应记录：

1. 每层 residual stream 的 RMS 或 L2 norm；
2. attention 和 MLP 分支输出相对主干的比例；
3. 每层梯度范数和参数更新比率；
4. norm 的缩放参数是否极端；
5. 不同深度的激活分布；
6. loss spike 与具体 batch 的对应关系。

如果只有总 loss，无法判断是底层梯度消失、上层激活爆炸，还是数据和优化器问题。

### 13.7 一个 toy 递推

~~~python
def residual_stack(value, updates, scale=1.0):
    history = [value]
    for update in updates:
        value = value + scale * update
        history.append(value)
    return history


updates = [1.0] * 8
for scale in [1.0, 0.25, 0.125]:
    print("scale =", scale, "last =", residual_stack(0.0, updates, scale)[-1])
~~~

这个例子不模拟深度网络，却显示相同的每层增量在深度累积时会迅速放大；残差缩放的作用是控制累积尺度，而不是消除学习能力。

## 14. LayerNorm、RMSNorm、ScaleNorm 与统计维度

### 14.1 LayerNorm 的统计对象

对一个 token 的 hidden vector x∈R^d，LayerNorm 在 hidden 维度上计算：

~~~math
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
~~~

~~~math
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
~~~

它不依赖 batch 统计，因此适合变长序列和自回归推理。

### 14.2 RMSNorm 做了什么

RMSNorm 只控制向量的均方根：

~~~math
r(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
~~~

~~~math
y_i=\gamma_i\frac{x_i}{r(x)}
~~~

它不保证输出均值为零，但在许多 decoder-only LLM 中，尺度控制已经足以获得稳定训练。减少均值计算还可能带来更轻的实现，实际收益取决于融合 kernel 和内存访问。

### 14.3 ScaleNorm 的不同取舍

ScaleNorm 直接按向量 L2 范数缩放：

~~~math
y=g\frac{x}{\lVert x\rVert_2+\epsilon}
~~~

其中 g 通常是一个可学习标量。与 RMSNorm 的逐维 γ 相比，ScaleNorm 对各维的可调自由度更少，强调整体向量尺度。

### 14.4 为什么 BatchNorm 不常用于 decoder LLM

BatchNorm 的统计跨 batch 或空间维度，语言模型的动态 batch、padding、分布式同步和自回归单 token 解码会造成训练—推理统计差异。它在视觉模型中仍然有价值，但不能因为它经典，就直接移植到所有语言模型。

### 14.5 epsilon 与精度

ε 既影响除零保护，也影响小方差时的归一化尺度。过小可能在低精度下不稳定，过大则会让归一化偏离原本的统计量。替换 norm 时应固定或记录 ε、参数精度、累积精度和 kernel 实现。

### 14.6 norm 参数的 weight decay

norm 的 γ/β 主要控制尺度和平移，通常从 weight decay 参数组中排除；这是经验配置，不是普适定理。若模型包含特殊 norm、门控 scale 或 adapter，应按参数功能和对照实验决定。

### 14.7 一个无框架比较

~~~python
import math


def validate_vector(values, eps):
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("values must be non-empty and finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def layer_norm(values, eps=1e-6):
    validate_vector(values, eps)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + eps)
    return [(value - mean) / scale for value in values]


def rms_norm(values, eps=1e-6):
    validate_vector(values, eps)
    rms = math.sqrt(sum(value * value for value in values) / len(values) + eps)
    return [value / rms for value in values]


def scale_norm(values, target=2.0, eps=1e-6):
    validate_vector(values, eps)
    if not math.isfinite(target) or target < 0:
        raise ValueError("target must be finite and non-negative")
    norm = math.sqrt(sum(value * value for value in values))
    return [target * value / (norm + eps) for value in values]


x = [5.0, 6.0, 4.0, 5.0]
for name, function in [
    ("LayerNorm", layer_norm),
    ("RMSNorm", rms_norm),
    ("ScaleNorm", scale_norm),
]:
    y = function(x)
    mean = sum(y) / len(y)
    rms = math.sqrt(sum(value * value for value in y) / len(y))
    norm = math.sqrt(sum(value * value for value in y))
    print(name, "mean=", round(mean, 4), "rms=", round(rms, 4), "l2=", round(norm, 4))
~~~

输出会显示 LayerNorm 的均值接近 0，RMSNorm 的 RMS 接近 1，ScaleNorm 的 L2 接近设定目标。三者控制的统计量不同，不能只看“输出看起来都差不多”就认为等价。

## 15. FFN、GeLU、SwiGLU 与 MLP 容量

### 15.1 Attention 之外的另一条主干

Transformer block 通常交替包含 attention 和 MLP。attention 让不同位置交换信息，MLP 对每个位置的表示做非线性变换：

~~~math
\mathrm{FFN}(x)
=W_{\mathrm{down}}\,
\phi(W_{\mathrm{up}}x+b_{\mathrm{up}})
+b_{\mathrm{down}}
~~~

如果没有激活函数，多层线性矩阵可以合并成一个线性矩阵，网络深度不会带来同等的非线性表达能力。MLP 的作用不是“再做一次 attention”，而是在已经混入上下文的 token 表示上进行高维特征组合。

### 15.2 参数量与计算量

普通 FFN 从 d_model 扩展到 d_ff，再投影回来，两个主要矩阵的参数量近似为：

~~~math
P_{\mathrm{FFN}}
\approx 2d_{\mathrm{model}}d_{\mathrm{ff}}
~~~

每个 token 的矩阵乘法量也与 d_model d_ff 成正比。若 d_ff=4d_model，参数量约为 8d_model²；一个带 Q/K/V/O 四个投影的 attention 模块约为 4d_model²。因此在许多 dense Transformer 中，MLP 的参数和 FLOPs 都不容忽略。

### 15.3 激活函数的架构含义

ReLU 是：

~~~math
\mathrm{ReLU}(x)=\max(0,x)
~~~

GeLU 是：

~~~math
\mathrm{GELU}(x)=x\Phi(x)
~~~

SiLU 是：

~~~math
\mathrm{SiLU}(x)=x\sigma(x)
~~~

GeLU 和 SiLU 都是平滑的输入相关调制；它们不会像 ReLU 那样把全部负半轴硬切成零。具体效果会受初始化、norm、数据和训练预算影响，不能脱离对照实验把某个激活函数写成普遍最优。

### 15.4 GLU 与 SwiGLU

门控 MLP 把中间表示拆成内容分支和控制分支：

~~~math
u=W_{\mathrm{up}}x,\qquad
g=W_{\mathrm{ctrl}}x
~~~

~~~math
h=u\odot\mathrm{SiLU}(g)
~~~

~~~math
\mathrm{SwiGLU}(x)=W_{\mathrm{down}}h
~~~

逐元素乘法让控制分支根据输入调节内容特征。它提供了比单分支激活更丰富的乘法交互，但同时多了一条投影矩阵。

### 15.5 为什么 hidden size 常接近 8/3

SwiGLU 有 up、control、down 三个矩阵，若中间宽度为 h：

~~~math
P_{\mathrm{SwiGLU}}\approx3d_{\mathrm{model}}h
~~~

要与普通 d_ff=4d_model 的 FFN 参数量 8d_model² 接近，需要：

~~~math
3d_{\mathrm{model}}h\approx8d_{\mathrm{model}}^2
\quad\Longrightarrow\quad
h\approx\frac{8}{3}d_{\mathrm{model}}
~~~

现实配置还会向上取整，以满足 tensor parallel 和矩阵 kernel 的维度约束。比较 GeLU 与 SwiGLU 时，必须同时报告参数量和 FLOPs。

### 15.6 FFN 与知识表示

FFN 具有大量参数和高维非线性变换，因此可能编码许多知识模式。把它类比为 key-value memory 有助于研究特征，但不能据此断定某条事实只存储在某一个神经元或某一层。embedding、attention、MLP、residual stream 和输出头共同构成分布式表示。

### 15.7 一个参数量比较器

~~~python
def ffn_parameters(d_model, hidden, projections):
    return projections * d_model * hidden


for d_model in [1024, 4096]:
    gelu_hidden = 4 * d_model
    swiglu_hidden = round(8 * d_model / 3)
    gelu = ffn_parameters(d_model, gelu_hidden, 2)
    swiglu = ffn_parameters(d_model, swiglu_hidden, 3)
    print(
        "d_model =", d_model,
        "GeLU =", round(gelu / d_model**2, 3), "d^2",
        "SwiGLU =", round(swiglu / d_model**2, 3), "d^2",
    )
~~~

这里输出的是以 d_model² 为单位的相对账本，方便比较结构，而不是完整模型参数量。embedding、norm、bias、MoE router 和输出头需要单独计算。

## 16. RoPE：旋转几何与相对位置

### 16.1 纯 attention 为什么不知道顺序

如果 attention 只依据 token 内容做匹配，序列中 token 的排列需要额外注入。位置编码可以加在 embedding 上、作为 attention bias，也可以改变 Q/K 的几何表示。

“绝对位置”告诉模型一个 token 位于第 i 个位置；“相对位置”关注两个 token 的距离 i−j。语言中的局部依赖、变量引用和跨句关系，往往更依赖后者。

### 16.2 二维旋转

把 hidden vector 的相邻两个维度看成二维平面。位置 i 的旋转矩阵为：

~~~math
R(\theta_i)=
\begin{bmatrix}
\cos\theta_i&-\sin\theta_i\\
\sin\theta_i&\cos\theta_i
\end{bmatrix}
~~~

对 query 和 key 分别应用：

~~~math
q_i'=R(\theta_i)q_i,\qquad
k_j'=R(\theta_j)k_j
~~~

旋转矩阵满足正交性质 R(θ_i)^T R(θ_j)=R(θ_j−θ_i)，因此：

~~~math
(q_i')^\mathsf Tk_j'
=q_i^\mathsf T
R(\theta_j-\theta_i)k_j
~~~

这说明 Q/K 的点积中出现了相对相位差。多维 RoPE 会在多个二维子空间使用不同频率，让模型同时感知短距离和长距离变化。

### 16.3 频率与位置

通常第 r 个二维子空间的角度与位置 i 和频率 ω_r 的乘积有关：

~~~math
\theta_{i,r}=i\omega_r
~~~

不同频率提供不同周期。高频对短距离变化敏感，低频变化慢，更适合表示较长距离。实际实现还要考虑 base、维度配对顺序、数据类型和 kernel。

### 16.4 RoPE 与长度泛化

RoPE 的相对结构为长度外推提供了更自然的先验，但不是无限长度保证。若训练只覆盖 4K，直接推理到 128K 可能出现相位分布、attention score、训练数据和有效检索能力的共同漂移。

位置插值、频率缩放和继续预训练的目标，是把更长的实际位置映射到模型更熟悉的相位范围，或者让模型在长序列上重新适应。任何 scaling 都可能影响短上下文，因此必须做长度分段回归。

### 16.5 RoPE 与 ALiBi 的区别

RoPE 改变 Q/K 的表示，使点积带有相对相位；ALiBi 直接在 logits 上加入距离 bias。两者都能让位置进入 attention，但一个修改几何表示，一个修改分数先验，不能混为同一方法。

### 16.6 一个旋转等价性实验

~~~python
import math


def rotate(vector, angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y = vector
    return [x * cosine - y * sine, x * sine + y * cosine]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


q = [0.7, -0.2]
k = [0.1, 0.9]
frequency = 0.37

for i, j in [(1, 4), (3, 6), (2, 9)]:
    absolute = dot(rotate(q, i * frequency), rotate(k, j * frequency))
    relative = dot(q, rotate(k, (j - i) * frequency))
    print(i, j, round(absolute, 8), round(relative, 8))
    assert abs(absolute - relative) < 1e-9
~~~

这个二维例子只验证旋转矩阵的代数性质，不代表完整 LLM 的长度能力。真正的 RoPE 还包括多频率维度、批量计算、缓存和精度处理。

### 16.7 评估位置扩展

位置编码改动至少要测：

1. 训练长度以内的 perplexity 和任务回归；
2. 训练长度附近的平滑退化；
3. 更长位置上的单证据和多证据检索；
4. 开头、中间、结尾的证据位置；
5. 长代码中的变量和括号依赖；
6. 长对话中的约束保持；
7. prefill 时间、decode 时间和 KV Cache。

最大可运行长度只是接口能力，不能代替有效使用能力。

## 17. ALiBi 与相对位置 bias

### 17.1 直接把距离写进分数

ALiBi 在 attention logits 上加入距离相关的线性偏置：

~~~math
\widetilde S_{ij}
=\frac{q_i^\mathsf Tk_j}{\sqrt{d_h}}
-m_h(i-j),\qquad j\le i
~~~

m_h 是第 h 个 head 的斜率。距离越远，默认分数越低，但这只是软偏置，不是把远处位置硬掩码。

### 17.2 多头斜率提供多尺度

如果所有 head 使用相同斜率，所有路由都会有同样的距离偏好。为不同 head 设置不同斜率后，部分 head 更偏局部，部分 head 可以保留远程访问能力。斜率分布是架构超参数，不能随意更改后只观察一个 benchmark。

### 17.3 相对 bias 的其他形式

相对位置 bias 可以是固定函数、可学习距离表或分桶后的 embedding。T5 风格方法把距离分到若干 bucket，近距离分得细、远距离分得粗，以控制参数量和长距离泛化。ALiBi 使用固定线性函数，参数更简单，但先验更强。

### 17.4 ALiBi 的适用条件与代价

它的优点是没有固定长度的位置 embedding 表，距离函数可以应用到更长位置；代价是单调的近处偏置可能伤害某些需要精确远程检索的任务。长度外推形式好，不等于模型经过训练就能整合远处证据。

### 17.5 一个 bias 生成器

~~~python
import math


def alibi_bias(seq_len, slope):
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError("seq_len must be a positive integer")
    if not math.isfinite(slope) or slope < 0:
        raise ValueError("slope must be finite and non-negative")
    matrix = []
    for i in range(seq_len):
        row = []
        for j in range(seq_len):
            row.append(float("-inf") if j > i else -slope * (i - j))
        matrix.append(row)
    return matrix


for slope in [0.25, 1.0]:
    bias = alibi_bias(5, slope)
    print("slope =", slope, "last row =", bias[-1])
    assert bias[0][1] == float("-inf")
    assert bias[-1][0] < bias[-1][-1]
~~~

### 17.6 RoPE、ALiBi 与绝对位置的选择表

| 方法 | 注入位置的位置 | 长度外推形式 | 主要风险 |
| --- | --- | --- | --- |
| 学习式绝对 embedding | 输入表示 | 未见位置没有自然参数 | 长度绑定 |
| RoPE | Q/K 几何 | 相对相位结构 | 超长相位与训练分布漂移 |
| ALiBi | attention logits | 距离函数可继续计算 | 距离先验可能过强 |
| 分桶相对 bias | attention logits | 依赖 bucket 设计 | 远距离分辨率有限 |

选择应由目标任务、训练长度、kernel、模型家族和回归评估共同决定。

## 18. MHA、MQA、GQA 与 KV Cache

### 18.1 三种 head 组织

标准 MHA 有 H 个 query heads，也有 H 个 K/V heads；MQA 仍保留 H 个 query heads，但只使用一个 K/V head；GQA 则使用介于 1 和 H 之间的 H_KV 个 K/V heads。

~~~math
\begin{aligned}
\mathrm{MHA}:&\quad H_Q=H,\ H_{\mathrm{KV}}=H\\
\mathrm{MQA}:&\quad H_Q=H,\ H_{\mathrm{KV}}=1\\
\mathrm{GQA}:&\quad H_Q=H,\ 1<H_{\mathrm{KV}}<H
\end{aligned}
~~~

当 H_Q 能被 H_KV 整除时，每个 K/V head 服务 H_Q/H_KV 个 query heads。

### 18.2 KV Cache 为什么受 head 数影响

增量解码时，历史 K/V 不需要重复投影，因此保存在缓存中。单个请求的缓存元素量近似为：

~~~math
N_{\mathrm{KV}}
=2LTH_{KV}d_h
~~~

再乘 batch 和 dtype 字节数。GQA 把 H_KV 从 H 降到 G 后，缓存约变为原来的 G/H；MQA 的理论 head 维度则只剩 1/H。

### 18.3 质量与系统的折中

共享 K/V 减少了每个 query head 的独立内容空间，可能影响模型质量、长程检索或复杂任务。GQA 比 MQA 保留更多 K/V 表达，因此常被作为折中。不能把训练好的 MHA 直接改配置就当成无损 GQA；权重映射、继续训练和目标任务评估都需要说明。

### 18.4 Prefill 与 Decode

推理通常分为两个阶段：

- prefill：一次处理用户输入，计算并写入整段 prompt 的 KV Cache；
- decode：逐 token 生成，每一步读取历史缓存并追加一个位置。

GQA/MQA 主要缓解 decode 阶段的缓存体积和带宽。若请求 prompt 很长，prefill 的 dense attention 仍然可能很贵；优化一阶段不等于优化另一阶段。

### 18.5 Paged KV 管理的边界

GQA/MQA 改变每 token 的缓存大小；分页 KV 管理解决缓存如何分块、复用、回收和减少碎片。二者可以组合：

~~~text
GQA/MQA       减少每个 token 的 K/V 体积
KV paging     改善缓存的分配、共享和调度
量化/压缩      进一步减少每个元素的字节数
~~~

实际吞吐还取决于 batch 调度、并发请求长度、attention kernel 和内存带宽。

### 18.6 估算缓存

~~~python
def kv_cache_mib(batch, seq_len, layers, kv_heads, head_dim, bytes_per_value=2):
    dimensions = [batch, seq_len, layers, kv_heads, head_dim]
    if any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("cache dimensions must be positive integers")
    if bytes_per_value <= 0:
        raise ValueError("bytes_per_value must be positive")
    elements = 2 * batch * seq_len * layers * kv_heads * head_dim
    return elements * bytes_per_value / (1024 ** 2)


query_heads = 32
for kv_heads in [32, 8, 1]:
    size = kv_cache_mib(1, 8192, 32, kv_heads, 128)
    print("kv_heads =", kv_heads, "cache_mib =", round(size, 1))

assert kv_cache_mib(1, 8192, 32, 8, 128) == kv_cache_mib(1, 8192, 32, 32, 128) / 4
~~~

这个账本没有包括权重、激活、allocator 碎片和多请求排队。线上容量规划要用真实引擎的峰值显存与 P95/P99 延迟验证。

### 18.7 分层理解

初学者先记住：MHA 每个 head 有自己的 K/V，MQA 共享一组，GQA 分组共享。

进一步分析时要继续问：共享后质量损失在哪些 slice 出现？prefill 和 decode 的瓶颈是否相同？缓存节省是否转化为更大 batch，还是被通信和调度吃掉？这些问题决定 GQA 是否真的带来单位成功任务成本下降。

## 19. Mixture of Experts：用稀疏激活扩展容量

### 19.1 从一个 FFN 到多个 expert

Dense Transformer 的每个 token 都经过同一个 FFN。MoE 把它替换成 E 个 expert，每个 expert 通常仍是一个 FFN 或 SwiGLU MLP。router 根据 token 表示决定使用哪几个 expert。

设第 t 个 token 的路由 logits 为：

~~~math
z_t=W_r x_t,\qquad
p_t=\mathrm{softmax}(z_t)
~~~

若选择的 expert 集合为 S_t，则一个 top-k MoE 输出可以写成：

~~~math
y_t=\sum_{e\in S_t}w_{t,e}F_e(x_t),
\qquad
w_{t,e}=\frac{p_{t,e}}
{\sum_{u\in S_t}p_{t,u}}
~~~

总参数量随 E 增长，但每个 token 只激活 k 个 expert。这里要区分 total parameters 和 activated parameters：前者说明模型潜在容量，后者更接近每 token 计算。

### 19.2 Top-1 与 Top-2

top-1 路由只选择最大概率 expert，计算和通信较省，但单次路由错误的影响更直接。top-2 选择两个 expert 并加权合并，通常提供更平滑的训练信号，却增加计算、通信和容量管理。

路由并不是外部多个模型投票。所有 expert 和 router 是同一个网络的一部分，梯度、通信和容量约束都在训练图中发生。

### 19.3 MoE 为什么不是免费扩容

MoE 把部分算力压力换成了：

1. token dispatch 和 combine；
2. expert parallel 的 all-to-all 通信；
3. 路由不均衡；
4. expert capacity 和 token dropping；
5. 动态 batch 与 tail latency；
6. checkpoint、容错和 serving 调度复杂度。

当 expert 分布在不同设备时，token 需要被发送到拥有目标 expert 的设备，计算完还要按原 token 顺序合并。GPU 的矩阵乘法变快，不代表整个 MoE step 变快。

### 19.4 Expert specialization 需要验证

人们常把不同 expert 解释成代码、数学或多语言专家，但 router 的分配可能受到 token 频率、位置、格式和 batch 组成影响。要研究 specialization，至少需要：

- 统计不同领域和任务的路由分布；
- 固定输入做 expert ablation；
- 比较 expert 替换对输出的影响；
- 分析路由熵、负载和质量是否相关。

均匀使用不等于每个 expert 学到了有用功能；高度专门化也不等于没有冗余。

### 19.5 一个 top-1 路由模拟

~~~python
def top1_route(logits):
    if not logits or any(not row for row in logits):
        raise ValueError("logits must contain non-empty rows")
    return [max(range(len(row)), key=row.__getitem__) for row in logits]


def count_assignments(assignments, num_experts):
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if any(not isinstance(expert, int) or not 0 <= expert < num_experts
           for expert in assignments):
        raise ValueError("expert assignments are out of range")
    counts = [0] * num_experts
    for expert in assignments:
        counts[expert] += 1
    return counts


logits = [
    [2.0, 0.1, 0.2],
    [1.9, 0.2, 0.1],
    [0.1, 2.0, 0.2],
    [1.8, 0.1, 0.3],
    [0.2, 0.4, 1.7],
]
assignments = top1_route(logits)
counts = count_assignments(assignments, 3)
print("assignments =", assignments)
print("counts =", counts)
~~~

结果中第一个 expert 可能获得过多 token。这个 toy 现象就是下一节需要处理的负载不均衡问题。

## 20. MoE 负载均衡、容量与通信

### 20.1 为什么 router 会塌缩

训练早期的微小偏好可能被反馈放大：某个 expert 得到更多 token，更新更多，随后更容易得到高 router 分数；其他 expert 因为样本少，学习机会变少。这会形成 expert collapse。

collapse 的表现不仅是平均利用率下降，还包括：

- 最大 expert 负载远高于平均；
- 某些领域几乎总走同一 expert；
- overflow 和 token drop 增加；
- device 之间 all-to-all 等待不平衡；
- 某些 expert 梯度长期接近零。

### 20.2 Auxiliary load-balancing loss

设 f_e 是被路由到 expert e 的 token 比例，P_e 是 router 对 expert e 的平均概率，E 是 expert 数，一个常见的辅助损失形式为：

~~~math
\mathcal L_{\mathrm{aux}}
=\alpha E\sum_{e=1}^{E}f_eP_e
~~~

当 f 和 P 都均匀时，乘积和达到较理想的平衡尺度；当 token 和概率都集中到少数 expert 时，损失变大。不同论文和实现的归一化、top-k 处理和系数可能不同，这个公式用于说明机制，不应替代具体实现文档。

辅助损失只能鼓励均衡，不能保证任务质量。过强的均衡约束可能把本应由某个 expert 处理的 token 强行分散，损害 specialization。

### 20.3 Expert capacity 与 token dropping

一个 batch 有 N 个 token，E 个 expert，capacity factor 为 c，则每个 expert 的槽位通常近似：

~~~math
C=\left\lceil c\frac{N}{E}\right\rceil
~~~

若分配给某 expert 的 token 数超过 C，就发生 overflow。系统可以丢弃、转发到备选 expert、增加 capacity 或采用更复杂的合并策略。

c 太小会增加 token dropping 和训练信号损失；c 太大则需要更多 padding、显存和通信。capacity 不是越大越安全，而是一个资源—丢失风险的折中。

### 20.4 Router logits 的数值稳定

router logits 过大时，softmax 过于尖锐，路由很早锁死。z-loss 的抽象目标是约束 log-sum-exp 的尺度：

~~~math
\mathcal L_z
=\frac{1}{T}\sum_{t=1}^{T}
\left(\log\sum_{e=1}^{E}\exp z_{t,e}\right)^2
~~~

训练中也可能加入 jitter 或噪声，让早期路由保留探索空间。噪声太大则会使 dispatch 不稳定，因此要结合路由熵、drop rate 和验证质量一起调。

### 20.5 All-to-all 的端到端账本

一次 MoE 层通常包含：

1. 本地计算 router；
2. 按 expert 排序 token；
3. 跨设备 dispatch；
4. expert MLP 计算；
5. 跨设备 combine；
6. 恢复原 token 顺序。

当 expert parallel 跨越多个节点时，网络拓扑、消息大小和同步等待可能决定 step 时间。模型 FLOPs 低而通信时间高，是 MoE 工程中非常常见的结果。

### 20.6 推理中的热点和尾延迟

在线请求的 token 分布不均，会让某个 expert 变成热点。即使平均 utilization 正常，P95/P99 也可能被最慢 expert、跨节点通信或某个长请求主导。评估不能只报告平均 token/s，应包括：

- expert 负载分位数；
- dispatch/combine 时间；
- 每个 expert 的排队；
- token drop 或 fallback；
- 请求级 P50/P95/P99；
- 质量和单位成功成本。

### 20.7 capacity 模拟

~~~python
import math


def apply_capacity(assignments, num_experts, capacity_factor):
    if not assignments:
        raise ValueError("assignments must not be empty")
    if not isinstance(num_experts, int) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if not math.isfinite(capacity_factor) or capacity_factor <= 0:
        raise ValueError("capacity_factor must be finite and positive")
    if any(not isinstance(expert, int) or not 0 <= expert < num_experts
           for expert in assignments):
        raise ValueError("expert assignments are out of range")
    capacity = math.ceil(capacity_factor * len(assignments) / num_experts)
    used = [0] * num_experts
    kept = []
    dropped = []
    for token, expert in enumerate(assignments):
        if used[expert] < capacity:
            used[expert] += 1
            kept.append(token)
        else:
            dropped.append(token)
    return capacity, used, kept, dropped


assignments = [0, 0, 0, 0, 0, 1, 1, 2, 3, 3]
for factor in [1.0, 1.5, 2.0]:
    result = apply_capacity(assignments, 4, factor)
    print("factor =", factor, "capacity =", result[0],
          "used =", result[1], "dropped =", result[3])
~~~

这个实验把负载不均衡、capacity 和 drop 的关系变成可以手算的结果。真实 top-k 路由还要处理每个 token 的权重、备选 expert、padding 和跨设备通信。

## 21. Encoder-only、Decoder-only 与 Encoder–Decoder

### 21.1 三种信息流

Encoder-only 通常使用双向 self-attention，位置 i 可以读取完整输入；decoder-only 使用 causal self-attention，位置 i 只能读取左侧上下文；encoder–decoder 则组合了 source encoder 和 target decoder：

~~~math
\begin{aligned}
\text{Encoder-only:}&\quad H=\mathrm{Enc}(X)\\
\text{Decoder-only:}&\quad y_t\sim P(y_t\mid y_{1:t-1},x)\\
\text{Encoder-decoder:}&\quad H=\mathrm{Enc}(X),\
Y=\mathrm{Dec}(Y_{1:t-1},H)
\end{aligned}
~~~

mask 和训练目标共同决定这些信息流，不是只看模块名字。

### 21.2 Encoder-only 与 MLM

BERT 类 encoder 用 masked language modeling 或类似的去噪目标，在输入中遮挡一部分 token，让模型利用左右上下文预测它们。它适合分类、embedding、reranking、抽取和审核，但训练时的双向可见性与逐 token 生成不完全一致。

这不是能力高低的简单排序，而是目标匹配问题：如果产品需要一个高质量句向量，encoder-only 可能比大 decoder 更经济；如果需要开放式多轮生成，decoder-only 更自然。

### 21.3 Decoder-only 与统一生成

decoder-only 的 next-token 目标可以把问答、翻译、摘要、代码和工具调用组织成 prompt→continuation。训练和自回归推理形态一致，数据与 serving 也较统一，这是通用 LLM 采用它的重要原因。

它并不意味着 decoder-only 在所有 source-to-target 任务上都理论最优。encoder–decoder 对输入完整双向理解、输出独立生成的任务仍然很自然。

### 21.4 Encoder–Decoder 的 cross-attention

encoder 先把 source 编码为 H，decoder 的每个位置通过 cross-attention 读取 H：

~~~math
\mathrm{CrossAttn}(Q_{\mathrm{dec}},K_H,V_H)
=\mathrm{softmax}
\left(\frac{Q_{\mathrm{dec}}K_H^\mathsf T}{\sqrt{d_h}}\right)V_H
~~~

翻译、摘要和某些语音/视觉到文本任务都可以利用这种清晰的输入—输出边界。代价是需要管理 encoder states、decoder self-attention cache 和 cross-attention 的系统形态。

### 21.5 选择架构的实际问题

| 需求 | 适合优先考虑 | 原因 |
| --- | --- | --- |
| 句向量、分类、reranking | encoder-only | 双向表示，推理短 |
| 通用对话、代码、工具调用 | decoder-only | 统一自回归生成 |
| 翻译、摘要、强条件生成 | encoder–decoder | source 双向理解、target 独立生成 |
| 图像/语音 encoder 到文本 | encoder + decoder | 模态编码与文本生成边界清楚 |

最终还要看数据标注、模型规模、延迟、缓存和部署生态。

### 21.6 mask 账本

~~~python
def encoder_mask(length):
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    return [[1 for _ in range(length)] for _ in range(length)]


def causal_mask(length):
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    return [[1 if column <= row else 0 for column in range(length)]
            for row in range(length)]


def encoder_decoder_masks(source, target):
    if (not isinstance(source, int) or source <= 0
            or not isinstance(target, int) or target <= 0):
        raise ValueError("source and target lengths must be positive integers")
    return {
        "encoder_self": encoder_mask(source),
        "decoder_self": causal_mask(target),
        "cross": [[1 for _ in range(source)] for _ in range(target)],
    }


print("encoder =", encoder_mask(3))
print("decoder =", causal_mask(3))
print("enc-dec =", encoder_decoder_masks(4, 2))
~~~

把 mask 画出来比背定义更容易发现训练目标和信息泄漏问题。

## 22. State Space Models：压缩历史的序列算子

### 22.1 为什么需要非 attention 路线

Transformer 的优势是显式读取任意历史 token，代价是 dense 交互、训练显存和解码缓存。状态空间模型（SSM）使用隐状态持续压缩历史，目标是在长序列和流式推理中获得更接近线性的成本。

### 22.2 连续与离散状态方程

经典离散形式可以写为：

~~~math
h_t=Ah_{t-1}+Bx_t,\qquad
y_t=Ch_t+Dx_t
~~~

A 控制历史状态如何衰减和混合，B 决定输入如何写入，C 决定状态如何读出，D 是输入的直接通路。对线性、时不变系统，可以把递推展开为卷积；训练时用卷积并行，推理时用状态递推。

离散化连续系统会得到带步长的 A_bar、B_bar 等参数。实际 S4、Mamba 等模型有更复杂的结构化参数化和数值处理，不能用四个矩阵的 toy 公式替代论文实现。

### 22.3 与 RNN、CNN、attention 的边界

SSM 和 RNN 都维护状态，但现代 SSM 更强调结构化参数、长序列并行算法和硬件友好的 scan。线性 SSM 与卷积存在联系，因此可以兼顾训练并行与推理递推。

attention 显式保留 token-to-token 访问；SSM 把历史压缩进状态。前者更适合精确检索，后者更省长序列成本。状态压缩的核心风险是：如果多个重要事实被压到有限状态中，细节可能丢失。

### 22.4 Mamba 的选择性

固定状态更新可能无法区分“必须长期保留的约束”和“可以遗忘的噪声”。选择性 SSM 让部分状态更新参数依赖输入，让模型动态决定写入、保留和遗忘哪些信息。Mamba 的 selective scan 同时要解决两件事：内容相关的状态控制，以及并行/硬件效率。

Mamba-2/SSD 等工作还展示了 SSM 与 attention 在更一般的结构空间中存在联系，因此更合理的趋势判断是“出现融合和可互换的序列算子”，而不是简单预测谁会完全淘汰谁。

### 22.5 一个标量状态实验

~~~python
def scalar_ssm(inputs, a, b=1.0, c=1.0, d=0.0):
    state = 0.0
    outputs = []
    for value in inputs:
        state = a * state + b * value
        outputs.append(c * state + d * value)
    return outputs


inputs = [1.0, 0.0, 0.0, 0.0, 0.0]
for a in [0.2, 0.8]:
    print("a =", a, "outputs =", [round(v, 4) for v in scalar_ssm(inputs, a)])
~~~

a 越接近 1，历史衰减越慢；这只是状态记忆的直觉，不是 Mamba 的完整实现。

### 22.6 SSM 的评价维度

不能只看 O(T) 或接近线性复杂度，还要评估：

1. 长序列语言建模；
2. 精确远距检索；
3. in-context learning；
4. 组合和复制任务；
5. 流式状态的重置与并发；
6. kernel 吞吐和端到端延迟；
7. 大规模训练稳定性；
8. 生态与工具支持。

## 23. Attention、SSM 与卷积的混合架构

### 23.1 为什么混合比单一替代更现实

attention 适合精确读取，SSM 适合连续状态传播，卷积适合局部模式。长上下文任务往往同时需要“持续记忆”和“精确找证据”，因此可以把三种算子放在不同层或同一 block 中。

常见布局包括：

- 每隔若干 SSM 层插入 attention；
- 前层用局部/卷积，后层用全局 attention；
- 大多数层用 SSM，少数层承担精确检索；
- 在不同 token 类型或模态上选择不同算子。

### 23.2 需要固定的公平账本

比较混合架构时要尽量固定参数量、训练 token、数据、上下文和硬件。只减少 attention 层而不补偿容量，观察到的质量变化不能归因于“SSM 更好”。

同时要拆分成本：

- 训练 FLOPs；
- 激活显存；
- attention KV Cache；
- SSM 状态大小；
- kernel launch 和融合；
- 通信与并行；
- 单请求和批量延迟。

### 23.3 一个粗略成本模型

~~~python
def layer_cost(length, kind, width=64):
    if kind == "attention":
        return length * length
    if kind in {"ssm", "conv"}:
        return length * width
    raise ValueError(kind)


def stack_cost(length, kinds):
    return sum(layer_cost(length, kind) for kind in kinds)


length = 8192
stacks = {
    "dense": ["attention"] * 12,
    "hybrid": ["attention", "ssm", "ssm", "conv"] * 3,
    "mostly_ssm": ["attention"] * 2 + ["ssm"] * 8 + ["conv"] * 2,
}
baseline = stack_cost(length, stacks["dense"])
for name, kinds in stacks.items():
    print(name, round(stack_cost(length, kinds) / baseline, 4))
~~~

这是一个交互数 proxy，不包含每种算子的真实常数、宽度、kernel 和质量。它只能帮助形成“减少二次层会降低某类成本”的直觉。

### 23.4 混合架构的失败模式

常见失败包括：

1. SSM 状态压缩丢失关键事实；
2. attention 层太少，精确检索退化；
3. 不同算子尺度不匹配，训练不稳定；
4. kernel 不成熟，理论成本没有转化为速度；
5. reset、缓存和并发状态管理错误；
6. 短 benchmark 变好，长任务和真实部署变差。

因此混合架构的主张必须配套任务矩阵和系统 profiler。

## 24. 深度、宽度、head 与硬件友好尺寸

### 24.1 结构变量

一个 decoder-only 模型的主要尺寸包括层数 L、模型宽度 d_model、query head 数 H_Q、K/V head 数 H_KV、head dimension d_h、MLP 中间宽度 d_ff、词表大小 V 和上下文长度 T。

通常：

~~~math
d_{\mathrm{model}}=H_Qd_h
~~~

GQA/MQA 只改变 H_KV，不改变 query 的总宽度关系。

### 24.2 深度和宽度

增加深度会增加逐层组合、顺序延迟和训练稳定性压力；增加宽度会提高每层表示容量，但主要矩阵参数和 FLOPs 近似按 d_model² 增长。两者不是互换完全等价的旋钮。

在一个带 dense attention 与普通 FFN 的 block 中，主要参数粗略为：

~~~math
P_{\mathrm{attn}}\approx4d_{\mathrm{model}}^2
~~~

~~~math
P_{\mathrm{ffn}}\approx2d_{\mathrm{model}}d_{\mathrm{ff}}
~~~

如果 d_ff=4d_model，则每层约 12d_model²，再乘 L；embedding、输出头、norm、bias 和 MoE 需要另算。

### 24.3 Head 数和 head dimension

head 太少可能限制路由子空间，head 太多则让 d_h 过小、投影和 kernel 更碎。64、80、96、128 等 head dimension 常见于硬件友好配置，但它不是语义规律，而是表达、并行和 kernel 的折中。

### 24.4 FFN 与 SwiGLU 尺寸

普通 FFN d_ff 常接近 4d_model；SwiGLU 的三投影结构常把中间宽度调到约 8d_model/3。tensor parallel 要求 intermediate size 能均匀切分，因此配置中经常出现经过整除取整的数字。

### 24.5 硬件约束

模型设计还要检查：

- hidden size 是否能被 tensor parallel 度数整除；
- query/KV heads 是否能被并行度整除；
- intermediate size 是否适合矩阵 kernel；
- vocab 是否做 padding；
- 序列长度是否适合 attention tile；
- GQA 是否被 serving kernel 原生支持。

一个理论更优但维度不友好的结构，可能在真实 GPU 上更慢。

### 24.6 计算最优不是只增大参数

在固定计算预算下，要平衡模型参数、训练 token、数据质量和训练步数。模型过大而训练不足会欠训练；模型过小而数据很多会受容量限制。架构尺寸必须和 scaling law、数据配比、优化器状态显存与推理成本一起决定。

### 24.7 参数量估算

~~~python
def round_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def block_params(d_model, d_ff, projections=4):
    return projections * d_model * d_model + 2 * d_model * d_ff


d_model = 4096
layers = 32
gelu_ff = 4 * d_model
swiglu_ff = round_multiple(8 * d_model // 3, 256)

for name, hidden, projection_count in [
    ("GeLU", gelu_ff, 4),
    ("SwiGLU", swiglu_ff, 4),
]:
    total = block_params(d_model, hidden, projection_count) * layers
    print(name, "intermediate =", hidden, "block stack params ~", round(total / 1e9, 3), "B")
~~~

它是架构账本，不是完整模型统计。真实配置应使用框架的参数计数，并把 tied embedding、MoE expert、router 和输出头说明清楚。

## 25. 从配置文件拆解现代 LLM

### 25.1 先看整体范式

分析一个未知模型时，先回答它是 encoder-only、decoder-only 还是 encoder–decoder，再看信息流和生成协议。不要从模型名称推断全部架构，因为同一系列的不同版本可能改变 norm、KV head、MoE、位置缩放和上下文。

### 25.2 再看六组字段

1. attention：H_Q、H_KV、d_h、是否稀疏或混合；
2. position：RoPE base、scaling、ALiBi 或 relative bias；
3. normalization：LayerNorm、RMSNorm、Pre/Post-LN、ε；
4. MLP：激活函数、intermediate size、SwiGLU 或 MoE；
5. scale：layers、hidden size、vocab、context；
6. serving：KV Cache、quantization、kernel 和并行切分。

配置字段只能告诉你设计意图，不能证明训练质量、长上下文有效性或产品行为。

### 25.3 LLaMA 风格组合的读法

如果配置包含 decoder-only、RMSNorm、RoPE、SwiGLU 和 H_KV<H_Q，它可以被描述为一种现代 decoder-only 组合，并且使用 GQA。这个判断说明架构结构，不应扩展成“因此一定更强”。质量还取决于数据、训练 token、后训练和评估。

### 25.4 MoE 配置的读法

如果看到 num_experts、top_k、capacity factor 或 auxiliary router loss，要把 total parameters 与 activated parameters 分开，并继续检查 expert parallel、通信拓扑、token drop、路由熵和热点延迟。

### 25.5 长上下文配置的读法

看到 max context 或 rope scaling 时，要追问：

- 长序列训练是否存在；
- 远处证据是否真的能被找到；
- 中间信息是否退化；
- prefill/decode 成本如何；
- KV Cache 是否可承载目标并发；
- 评估是否只测接口接收而没有测任务成功。

### 25.6 配置解析器

~~~python
def inspect_config(config):
    required = {
        "num_attention_heads",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"missing config fields: {sorted(missing)}")
    query_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", query_heads)
    if any(
        not isinstance(value, int) or value <= 0
        for value in [query_heads, kv_heads, config["num_hidden_layers"],
                      config["hidden_size"], config["intermediate_size"]]
    ):
        raise ValueError("model dimensions must be positive integers")
    if kv_heads > query_heads or query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if kv_heads == query_heads:
        attention = "MHA"
        cache_ratio = 1.0
    elif kv_heads == 1:
        attention = "MQA"
        cache_ratio = 1.0 / query_heads
    else:
        attention = "GQA"
        cache_ratio = kv_heads / query_heads

    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    return {
        "layers": config["num_hidden_layers"],
        "attention": attention,
        "kv_cache_ratio_vs_mha": round(cache_ratio, 4),
        "mlp_ratio": round(intermediate / hidden, 4),
        "norm": config.get("norm", "unknown"),
        "position": config.get("position", "unknown"),
        "experts": config.get("num_experts", 0),
    }


llama_like = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "norm": "RMSNorm",
    "position": "RoPE",
}
print(inspect_config(llama_like))
~~~

### 25.7 一个完整架构审计问题集

读完配置后，按以下顺序写一页审计记录：

1. 信息从哪里到哪里流动；
2. 一次训练 step 激活哪些参数；
3. 一次 decode 保存和读取哪些状态；
4. 长度增加时哪一项成本先爆；
5. 路由、位置或归一化改变了什么归纳偏置；
6. 哪些结论来自配置，哪些必须用实验验证；
7. 最小回归集应该包括哪些短任务、长任务和系统指标。

这比单纯列出组件名称更能检验架构理解。

### 25.8 综合案例

假设一个模型有 32 层、4096 hidden、32 个 query heads、8 个 KV heads、11008 intermediate、RMSNorm、RoPE 和 SwiGLU。可以得到：

- H_KV/H_Q=1/4，缓存 head 维度约为同样 MHA 的四分之一；
- intermediate/hidden≈2.69，接近参数匹配下的 SwiGLU 宽度；
- decoder-only 说明默认是 causal 生成；
- RMSNorm 与 Pre-Norm 的组合需要关注 final norm 和 residual 尺度；
- RoPE 配置不能单独证明长上下文有效；
- GQA 只说明缓存结构，不能证明质量无损。

如果再加入 64 experts、top-k=2，就要增加 capacity、router loss、expert parallel 和通信的审计线。一个配置拆解的结论必须停在证据允许的范围内。

## 26. 架构选择的统一评估

### 26.1 四本账

比较 attention、SSM、MoE 或不同 norm 时，至少维护四本账：

1. 质量账：验证 loss、任务成功、长上下文、鲁棒性；
2. 计算账：FLOPs、激活、参数和通信；
3. 系统账：显存、带宽、kernel、并行、P95/P99；
4. 风险账：训练失败、数据泄漏、能力回归、未知边界。

只有四本账都改善，才可以把一个架构改动称为整体收益。

### 26.2 最小 ablation

架构实验要固定数据、token budget、随机种子范围和评估协议，至少保留 dense baseline。SwiGLU 要匹配参数量或 FLOPs；GQA 要比较质量和缓存；MoE 要报告 activated parameters、通信和 drop；SSM/混合要加入精确检索任务。

### 26.3 章节主线

本部分的知识关系可以写成：

~~~text
Q/K/V 路由
  -> T^2 交互与 KV Cache
  -> sparse/IO/kernel 取舍
  -> residual/norm 稳定性
  -> FFN/SwiGLU 容量
  -> RoPE/ALiBi 位置泛化
  -> MQA/GQA 解码缓存
  -> MoE 条件容量与通信
  -> encoder/decoder/SSM/混合范式
  -> 尺寸、硬件与配置审计
~~~

读者最终应能同时回答两个问题：模型为什么这样计算，以及这种计算在真实数据、硬件和任务上是否值得。

## 参考资料与证据边界

本部分优先使用原始论文、正式教材和公开技术报告。数学定义与复杂度推导属于可直接检查的内容；具体模型在某个 benchmark 上的收益属于实验结果；闭源模型的内部实现不能由外部配置或产品宣传反推。

1. Vaswani et al., Attention Is All You Need：Transformer、scaled dot-product attention 和 multi-head attention。
   https://arxiv.org/abs/1706.03762
2. Dao et al., FlashAttention：Fast and Memory-Efficient Exact Attention with IO-Awareness：精确 attention 的 IO 优化。
   https://arxiv.org/abs/2205.14135
3. Dao, FlashAttention-2：Faster Attention with Better Parallelism and Work Partitioning：并行与 work partitioning。
   https://arxiv.org/abs/2307.08691
4. Beltagy, Peters and Cohan, Longformer：The Long-Document Transformer：局部与全局稀疏注意力。
   https://arxiv.org/abs/2004.05150
5. Zaheer et al., Big Bird：Transformers for Longer Sequences：块状、全局和随机稀疏连接。
   https://arxiv.org/abs/2007.14062
6. Choromanski et al., Rethinking Attention with Performers：线性 attention 近似。
   https://arxiv.org/abs/2009.14794
7. Xiong et al., On Layer Normalization in the Transformer Architecture：Pre-LN 与梯度分析。
   https://arxiv.org/abs/2002.04745
8. Wang et al., DeepNet：Scaling Transformers to 1,000 Layers：DeepNorm/DeepNet 深层稳定性。
   https://arxiv.org/abs/2203.00555
9. Zhang and Sennrich, Root Mean Square Layer Normalization：RMSNorm。
   https://arxiv.org/abs/1910.07467
10. Shazeer, GLU Variants Improve Transformer：GeLU/SiLU 门控 MLP。
    https://arxiv.org/abs/2002.05202
11. Su et al., RoFormer：Enhanced Transformer with Rotary Position Embedding：RoPE。
    https://arxiv.org/abs/2104.09864
12. Press et al., Train Short, Test Long：Attention with Linear Biases Enables Input Length Extrapolation：ALiBi。
    https://arxiv.org/abs/2108.12409
13. Shazeer, Fast Transformer Decoding：One Write-Head is All You Need：MQA。
    https://arxiv.org/abs/1911.02150
14. Ainslie et al., GQA：Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints：GQA。
    https://arxiv.org/abs/2305.13245
15. Shazeer, Sparsely-Gated Mixture-of-Experts：MoE 路由与稀疏激活。
    https://arxiv.org/abs/1701.06538
16. Lepikhin et al., GShard：Scaling Giant Models with Conditional Computation and Automatic Sharding：专家并行。
    https://arxiv.org/abs/2006.16668
17. Fedus, Zoph and Shazeer, Switch Transformers：稀疏激活与 top-1 routing。
    https://arxiv.org/abs/2101.03961
18. Gu and Dao, Mamba：Linear-Time Sequence Modeling with Selective State Spaces：选择性 SSM。
    https://arxiv.org/abs/2312.00752
19. Gu and Goel, Efficiently Modeling Long Sequences with Structured State Spaces：S4。
    https://arxiv.org/abs/2111.00396
20. Dao and Gu, Transformers are SSMs：Generalized Models and Efficient Algorithms Through Structured State Space Duality：Mamba-2/SSD 视角。
    https://arxiv.org/abs/2405.21060
21. Gulati et al., Conformer：Convolution-augmented Transformer for Speech Recognition：attention 与卷积混合。
    https://arxiv.org/abs/2005.08100
22. Devlin et al., BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding：encoder-only 与 MLM。
    https://arxiv.org/abs/1810.04805
23. Radford et al., Language Models are Unsupervised Multitask Learners：GPT 范式的早期公开说明。
    https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
24. Raffel et al., Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer：T5 与 encoder-decoder。
    https://arxiv.org/abs/1910.10683
25. Touvron et al., LLaMA：Open and Efficient Foundation Language Models：现代 decoder-only 配置与训练路线。
    https://arxiv.org/abs/2302.13971
26. Hoffmann et al., Training Compute-Optimal Large Language Models：模型尺寸、数据和计算预算的平衡。
    https://arxiv.org/abs/2203.15556
