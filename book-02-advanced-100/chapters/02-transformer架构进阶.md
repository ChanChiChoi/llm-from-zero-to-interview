# 第二部分：Transformer 架构进阶

## 第 11 讲：Self-Attention 的矩阵视角与信息路由

### 本讲目标

第一册里我们已经从直觉和公式层面学习过 Self-Attention。本讲进一步用矩阵视角理解 attention：它不只是“算相关性”，更像是在序列内部做内容寻址、动态信息路由和可微检索。

你需要掌握：

1. Self-Attention 的矩阵形式。
2. Q、K、V 在信息路由中的角色。
3. attention matrix 为什么可以看成动态路由矩阵。
4. attention 和固定卷积、RNN 的关键差异。
5. 多头 attention 为什么不是简单重复。
6. attention 权重能解释什么，不能解释什么。
7. 面试中如何从“内容寻址”角度讲清楚 attention 的本质。

### 从公式重新看 Self-Attention

给定输入序列 hidden states：

```text
X ∈ R^{T × d_model}
```

其中 `T` 是序列长度，`d_model` 是隐藏维度。

Self-Attention 首先通过线性投影得到：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

其中：

```math
Q,K,V\in\mathbb{R}^{T\times d_h}
```

其中 `d_h` 表示单个 attention head 的维度。

然后计算 attention logits：

```math
S=\frac{QK^T}{\sqrt{d_h}}
```

再经过 softmax 得到 attention matrix：

```math
A=\mathrm{softmax}(S)
```

最后输出：

```math
O=AV
```

这几个矩阵的核心含义是：

1. `Q` 表示当前位置想找什么信息。
2. `K` 表示每个位置提供什么索引。
3. `V` 表示每个位置真正被读取的内容。
4. `A` 表示每个位置从其他位置读取多少信息。

### Attention Matrix 是什么

`A` 的 shape 是：

```math
A\in\mathbb{R}^{T\times T}
```

其中第 `i` 行表示第 `i` 个 token 如何从所有 token 读取信息。

也就是说：

```math
O_i=\sum_j A_{ij}V_j
```

这句话很重要。

它说明 attention 输出不是固定模板，而是输入相关的加权求和。

每个 token 会根据当前上下文动态决定：

1. 该关注谁。
2. 从谁那里读信息。
3. 每个来源读多少。

所以 `A` 可以理解为一个动态信息路由矩阵。

### 内容寻址的直觉

传统数组访问是按位置寻址。

例如：

```text
读取第 5 个元素
```

Attention 更像内容寻址。

当前位置生成一个 query，去和所有 key 匹配：

```text
谁的 key 和我的 query 最相关，我就从谁那里读取 value
```

这和检索系统很像：

1. query 是检索请求。
2. key 是文档索引。
3. value 是文档内容。
4. attention weight 是相关性权重。

区别是 attention 是端到端可微的，Q、K、V 都是模型自己学出来的表示。

### 为什么除以 sqrt(d_head)

如果 `Q` 和 `K` 每个维度的方差近似为 1，那么点积 `Q_i · K_j` 的方差会随 `d_head` 增大。

如果不缩放，logits 可能很大，softmax 会变得过于尖锐。

softmax 过于尖锐会导致：

1. 注意力几乎集中在少数位置。
2. 梯度变小或不稳定。
3. 训练初期难以优化。

所以使用：

```text
QK^T / sqrt(d_head)
```

目的是控制 logits 的尺度，让 softmax 处在更合适的区域。

### Attention 和卷积的差异

卷积的连接模式通常是固定的。

例如一维卷积会看固定窗口：

```text
x_{i-1}, x_i, x_{i+1}
```

不管输入内容是什么，窗口结构不变。

Attention 的连接模式是动态的。

第 `i` 个 token 可以根据内容选择关注任意位置。

这带来几个优势：

1. 长距离依赖路径短。
2. 不需要固定局部窗口。
3. 能根据语义动态建立连接。
4. 适合处理语言中的指代、依赖和跨句关系。

但代价是：

```text
标准 attention 的时间和显存复杂度是 O(T²)
```

这也是后面要讨论稀疏 attention、FlashAttention、MQA/GQA 和长上下文优化的原因。

### Attention 和 RNN 的差异

RNN 按时间步递归处理序列：

```text
h_t = f(h_{t-1}, x_t)
```

信息从远处 token 传到当前位置，需要经过很多步。

这会带来：

1. 长距离依赖难学。
2. 并行性差。
3. 梯度传播路径长。

Self-Attention 则让任意两个 token 可以在一层内直接交互。

路径长度更短，也更容易并行。

但 attention 没有天然顺序递归结构，所以需要位置编码提供位置信息。

### Causal Mask 下的信息路由

GPT 类自回归模型不能看到未来 token。

所以会加 causal mask，让位置 `i` 只能关注 `j <= i` 的位置。

形式上：

```text
S_{ij} = -∞, if j > i
```

softmax 后未来位置权重为 0。

这意味着 causal self-attention 中，第 `i` 个 token 的信息来源只能是：

```text
x_1, x_2, ..., x_i
```

这个约束保证训练目标和自回归生成一致。

### Multi-Head Attention 的矩阵视角

单头 attention 只有一套 Q、K、V 投影。

Multi-Head Attention 使用多套投影：

```text
head_h = Attention(XW_Q^h, XW_K^h, XW_V^h)
```

然后拼接所有 head：

```text
O = concat(head_1, ..., head_H) W_O
```

多头的意义不是简单平均多次 attention。

每个 head 可以学习不同的信息路由模式。

例如：

1. 某些 head 关注局部邻近 token。
2. 某些 head 关注句法依赖。
3. 某些 head 关注实体指代。
4. 某些 head 关注分隔符或格式 token。
5. 某些 head 负责复制或聚合特定信息。

当然，这些解释不是每个 head 都天然清晰，也不能过度神话 attention head。

### Attention Head 是否可解释

attention 权重可以提供一定解释线索，但不能直接等同于模型解释。

原因包括：

1. attention 权重只显示 value 聚合比例。
2. V 向量本身已经是复杂变换后的表示。
3. 多层、多头之间会叠加和重组信息。
4. MLP 子层也会进行大量非线性变换。
5. 高 attention weight 不一定表示该 token 对最终答案因果重要。

所以更准确的说法是：

```text
attention map 是有用的分析工具，但不是完整可解释性证明
```

如果要判断因果重要性，需要配合 ablation、activation patching、causal tracing 等方法。

### Attention 作为可微检索

从系统角度看，attention 像一个内部检索器。

对于每个位置，它在当前上下文中检索相关信息。

与外部 RAG 相比：

1. attention 检索的是上下文窗口内 token。
2. RAG 检索的是外部知识库文档。
3. attention 的 key/value 是连续向量。
4. RAG 的文档通常是离散文本或 chunk。
5. attention 是端到端训练得到的内部机制。

这个类比很有用，但不能混淆两者。

Attention 不能凭空访问上下文外的信息。

如果上下文里没有事实证据，attention 再强也可能幻觉。

### 信息路由和 residual stream

现代 Transformer 可以理解为不断更新 residual stream。

每层 attention 从上下文中路由信息，MLP 对特征做非线性变换，然后都通过 residual 加回主干。

简化形式：

```text
x <- x + Attention(Norm(x))
x <- x + MLP(Norm(x))
```

从这个角度看：

1. residual stream 是信息主干。
2. attention 决定跨 token 信息如何流动。
3. MLP 决定每个 token 内部特征如何变换。
4. norm 控制尺度和训练稳定性。

这比只说“attention 算相关性”更接近现代 LLM 架构理解。

### 一个最小 PyTorch 例子

下面是一个简化版单头 causal self-attention。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return self.out_proj(out)
```

shape 对应关系：

```text
x:      [batch, seq_len, d_model]
q/k/v:  [batch, seq_len, d_head]
scores: [batch, seq_len, seq_len]
attn:   [batch, seq_len, seq_len]
out:    [batch, seq_len, d_head]
```

这段代码体现了三件事：

1. `QK^T` 形成 token-to-token 路由分数。
2. causal mask 阻止当前位置读取未来。
3. `attn @ V` 根据路由权重聚合内容。

### 最小 demo：验证 causal attention 的 shape 和 mask

下面这个 demo 是上面代码的可运行版本，额外返回 attention matrix，检查未来位置的 attention 权重是否为 0。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        self.out_proj = nn.Linear(d_head, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, seq_len, seq_len), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = self.out_proj(attn @ v)
        return out, attn


x = torch.randn(2, 4, 8)
attn_layer = TinyCausalSelfAttention(d_model=8, d_head=4)
out, attn = attn_layer(x)

future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
assert out.shape == x.shape
assert torch.allclose(attn[:, future_mask], torch.zeros_like(attn[:, future_mask]))

print("out shape:", out.shape)
print("attn shape:", attn.shape)
```

这个 demo 强调一个工程细节：causal mask 必须在 softmax 前加到 logits 上，否则未来 token 仍可能泄漏信息。

### 面试中怎么讲

如果面试官问“Self-Attention 的本质是什么”，可以这样回答：

```text
Self-Attention 本质上是一种输入相关的内容寻址和动态信息路由机制。每个 token 通过 query 表示自己想找什么信息，所有 token 通过 key 表示自己能被如何匹配，通过 value 提供被读取的内容。QK^T 得到 token 之间的匹配分数，softmax 后形成 attention matrix，每一行表示当前位置从所有位置读取信息的权重，最后用这个权重对 V 做加权求和。因此 attention 可以让序列中任意位置在一层内直接交互。
```

如果面试官问“attention 权重能不能解释模型决策”，可以这样回答：

```text
attention 权重可以提供一定线索，因为它显示某一层某个 head 中 value 聚合的比例。但它不能直接等同于模型解释，因为 value 已经是变换后的表示，多层多头和 MLP 会继续重组信息，高 attention weight 也不一定代表对最终输出有因果贡献。要做更严格解释，需要结合 ablation、activation patching 或 causal tracing。
```

### 真实项目中的坑

#### 坑 1：把 attention 简单说成相似度计算

相似度只是第一步，更重要的是 softmax 后形成动态路由矩阵，并用它聚合 value。

#### 坑 2：忽略 mask

GPT 类模型如果 causal mask 错了，训练会泄露未来信息，评估结果会失真。

#### 坑 3：过度解释 attention map

attention map 有分析价值，但不是完整因果解释。

#### 坑 4：只关注 attention，忽略 MLP 和 residual stream

Transformer 的能力来自 attention、MLP、normalization、residual、position encoding 和训练目标的组合。

#### 坑 5：忽略 O(T²) 成本

标准 attention 的路由能力强，但长上下文场景下计算和显存成本很高。

### 面试官会怎么问

#### 问题 1：Q、K、V 分别是什么作用？

回答框架：

1. Q 表示当前位置的查询需求。
2. K 表示每个位置被匹配的索引。
3. V 表示每个位置提供的内容。
4. Q 和 K 决定读谁，V 决定读到什么。

#### 问题 2：为什么 attention 可以处理长距离依赖？

回答框架：

1. 任意两个 token 可以在一层内直接交互。
2. 路径长度比 RNN 短。
3. 权重由内容动态决定，而不是固定局部窗口。
4. 代价是标准 attention 复杂度为 O(T²)。

#### 问题 3：Multi-Head Attention 为什么有用？

回答框架：

1. 不同 head 有不同 Q/K/V 投影。
2. 可以学习不同子空间的信息路由。
3. 有些 head 可能关注局部、句法、指代或格式。
4. 多头结果拼接后再融合，表达能力更强。

#### 问题 4：attention matrix 的每一行代表什么？

回答框架：

1. 第 `i` 行表示第 `i` 个 token 对所有 token 的读取权重。
2. 这些权重经过 softmax，通常和为 1。
3. 输出 `O_i` 是对所有 `V_j` 的加权求和。

#### 问题 5：为什么 attention 需要位置编码？

回答框架：

1. 纯 attention 对输入集合本身近似置换等变。
2. 语言序列顺序非常重要。
3. 位置编码让模型知道 token 的顺序和相对位置。
4. RoPE、ALiBi 等方法会进一步影响长度泛化。

### 常见误区

1. 误区：attention 权重就是最终答案的解释。
   纠正：attention 权重只是某层某头的聚合权重，不等于因果解释。

2. 误区：Q、K、V 是人工定义的语义角色。
   纠正：Q、K、V 是学出来的投影空间，语义解释只是帮助理解。

3. 误区：多头 attention 只是把同一件事重复多次。
   纠正：不同 head 可以学习不同子空间和路由模式。

4. 误区：attention 能解决所有长上下文问题。
   纠正：标准 attention 有 O(T²) 成本，且长上下文还涉及位置外推、检索、训练分布和评估问题。

5. 误区：只要有 attention，模型就能访问所有知识。
   纠正：attention 只能在当前上下文中路由信息，模型参数知识和外部知识是另一回事。

### 小练习

1. 写出 `Q = XW_Q`、`K = XW_K`、`V = XW_V` 的 shape。
2. 解释 attention matrix 中第 `i` 行的含义。
3. 用自己的话解释“content addressing”。
4. 比较 attention、CNN、RNN 处理长距离依赖的差异。
5. 实现一个最小 causal self-attention，并标注每一步 shape。
6. 用 2 分钟回答“Self-Attention 的本质是什么”。

### 本讲总结

本讲最重要的结论：

1. Self-Attention 可以看成内容寻址和动态信息路由机制。
2. Q 和 K 决定 token 之间的匹配与路由权重，V 提供被聚合的内容。
3. attention matrix 的每一行表示当前位置从所有位置读取信息的分布。
4. 相比 CNN 和 RNN，attention 能让任意 token 在一层内直接交互，但标准复杂度是 O(T²)。
5. Multi-Head Attention 允许不同 head 学习不同子空间的信息路由。
6. attention map 有解释线索，但不能直接等同于因果解释。
7. 现代 LLM 的信息流要结合 attention、MLP、residual stream、normalization 和位置编码一起理解。

## 第 12 讲：Attention 复杂度瓶颈与稀疏化思路

### 本讲目标

Self-Attention 很强，但标准 dense attention 的计算和显存复杂度随序列长度平方增长。本讲讨论这个瓶颈从哪里来，以及稀疏 attention、局部 attention、低秩近似、线性 attention 和工程优化分别在解决什么问题。

你需要掌握：

1. 标准 attention 的时间复杂度和显存复杂度。
2. 为什么长上下文会让 attention 成为瓶颈。
3. 稀疏 attention 的基本思想。
4. local、sliding window、block sparse、global token 等模式。
5. 低秩和线性 attention 的直觉。
6. FlashAttention 和稀疏 attention 的区别。
7. 面试中如何讨论长上下文 attention 的 trade-off。

### 标准 Attention 的复杂度来自哪里

标准 self-attention 的核心计算是：

```text
S = QK^T
A = softmax(S)
O = AV
```

如果序列长度是 `T`，单个 head 的维度是 `d_h`，那么：

```text
Q, K, V ∈ R^{T × d_h}
S, A ∈ R^{T × T}
```

计算 `QK^T` 的复杂度是：

```text
O(T² d_h)
```

计算 `AV` 的复杂度也是：

```text
O(T² d_h)
```

所以单个 head 的核心 attention 计算是：

```text
O(T² d_h)
```

所有 head 加起来大致是：

```text
O(T² d_model)
```

这里的 `d_model` 约等于 `heads × d_h`。严格计数时还要加上 Q/K/V/out projection 和 MLP 的计算，但讨论 attention 瓶颈时，最关键的是 `T × T` token 交互项会随上下文长度平方增长。

真正的问题在于 `T²`。

当上下文长度从 4K 增加到 32K，`T²` 会增加 64 倍。

### 显存瓶颈

训练时不仅要算 attention，还要保存中间结果用于反向传播。

最直接的问题是 attention matrix：

```text
A ∈ R^{T × T}
```

如果 batch、head 数也算进去，attention logits 或 attention probabilities 的 shape 通常是：

```text
[batch, heads, seq_len, seq_len]
```

这在长上下文下非常大。

例如 `seq_len=32768` 时，单个 head 的 attention matrix 元素数约为：

```text
32768 × 32768 ≈ 1.07B
```

即使用 BF16，每个元素 2 字节，单个 head 就接近 2GB，更不用说 batch、多头和反向传播。

这就是长上下文训练中 attention 显存瓶颈的来源。

### 推理时的瓶颈

自回归推理时，每次生成一个新 token。

如果没有 KV Cache，每步都要重新计算所有历史 token 的 K/V，非常浪费。

使用 KV Cache 后，每步只计算新 token 的 Q/K/V，并把新的 K/V 追加到缓存。

但每生成一个新 token，仍然要让新 token 的 Q 和所有历史 K 做 attention：

```text
O(T d_h)
```

生成完整长度 `T` 的序列，总体仍然接近：

```text
O(T² d_h)
```

同时 KV Cache 本身显存也随序列长度线性增长：

```text
O(T × layers × heads × d_h)
```

更精确地说，KV Cache 需要同时保存 K 和 V，元素量约为：

```text
2 × layers × batch × T × kv_heads × d_h
```

如果使用 MQA 或 GQA，`kv_heads` 会小于 query heads，因此 KV Cache 会明显变小。

所以长上下文推理的瓶颈包括：

1. 每步读取越来越长的 KV Cache。
2. KV Cache 占用显存。
3. attention 带宽压力变大。
4. batch 调度更困难。

### 稀疏 Attention 的核心思想

标准 attention 让每个 token 关注所有 token。

稀疏 attention 的想法是：

```text
不是所有 token 对都同等重要，能不能只计算一部分连接？
```

如果每个 token 只关注 `k` 个 token，而不是 `T` 个 token，那么复杂度可以从：

```text
O(T²)
```

降到：

```text
O(Tk)
```

当 `k << T` 时，节省明显。

但问题是：

```text
哪些连接可以删，删掉后能力损失多少？
```

这就是所有稀疏 attention 方法的核心 trade-off。

### Local Attention

Local attention 只让每个 token 关注附近窗口。

例如窗口大小为 `w`：

```text
token i 只关注 [i-w, ..., i]
```

复杂度从 `O(T²)` 变成：

```text
O(Tw)
```

优点：

1. 简单。
2. 适合局部依赖强的任务。
3. 显存和计算可控。
4. 容易实现 sliding window KV Cache。

缺点：

1. 远距离 token 不能直接交互。
2. 长程依赖需要多层逐步传递。
3. 对跨文档、跨段落信息整合不友好。

### Sliding Window Attention

Sliding window attention 是 local attention 的常见形式。

每个 token 只看固定长度的历史窗口。

例如：

```text
window_size = 4096
```

生成第 `t` 个 token 时，只关注：

```text
[t-4096, ..., t]
```

这会让每步推理成本不再随完整历史无限增长，而是被窗口大小限制。

### Global Token 和稀疏全局连接

只用局部窗口会切断长程连接。

一种改进是加入 global token。

global token 可以关注所有位置，所有位置也可以关注 global token。

这样信息可以通过 global token 汇聚和广播。

类似结构在 Longformer、BigBird 等模型思想中出现过。

直觉是：

```text
局部 attention 负责近邻细节，全局 token 负责长程汇总
```

### Block Sparse Attention

Block sparse attention 把 attention matrix 划分成块，只计算部分块。

例如：局部块、间隔块、全局块和随机块。

这样做的原因是硬件更喜欢块状计算，而不是完全随机稀疏。

### Random Sparse Attention

一些方法会加入随机连接。

直觉是图论中的小世界网络：只要加入少量长程随机边，信息传播路径可能明显变短。

优点是增强长程连接，缺点是随机连接不一定对具体任务有意义。

### 低秩近似思路

另一类方法认为 attention matrix 可能存在低秩结构。

如果 `T × T` 的 attention 不需要完整表示，可以用低秩形式近似。

例如用较少的 landmark token 或 projection 把序列压缩，再做 attention。

### 线性 Attention 的直觉

标准 attention 是：

```math
\mathrm{softmax}(QK^T)V
```

线性 attention 试图通过 kernel feature map 改写 attention，使计算顺序变成：

```math
\phi(Q)\left(\phi(K)^T V\right)
```

这样可以先聚合 `K` 和 `V`，再和 `Q` 交互，从而避免显式构造 `T × T` 矩阵。

### FlashAttention 解决的不是复杂度阶数

FlashAttention 的核心不是把 `O(T²)` 变成 `O(T)`。

它仍然计算精确 attention，但主要优化显存读写、attention 中间矩阵 materialization 和分块 softmax。

换句话说：dense attention、sparse attention、linear attention 和 FlashAttention 解决的是不同层面的问题。

```text
dense attention: 连接最完整，理论交互数是 T²
sparse attention: 改连接图，只保留局部、全局、随机或块状连接
linear attention: 改写或近似 attention 形式，避免显式 T×T 矩阵
FlashAttention: 不改数学结果，优化精确 dense attention 的 IO 和中间矩阵存储
```

### 为什么稀疏 Attention 没有完全取代标准 Attention

原因包括：

1. Full attention 简单、通用、效果强。
2. 稀疏模式会引入 inductive bias，可能损失能力。
3. 很多任务需要精确长程检索。
4. 稀疏 kernel 不一定比 dense kernel 更快。
5. 工程生态对 dense attention 优化更成熟。

### 长上下文不只是 Attention 问题

长上下文还涉及位置编码是否能外推、训练时是否见过长序列、模型是否会 lost in the middle、KV Cache 是否可承载，以及评估是否覆盖真实能力。

### 一个最小示例：Sliding Window Mask

下面是一个简化版 sliding window causal mask。

```python
import torch


def build_sliding_window_mask(seq_len, window_size, device):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)

    future = j > i
    too_far = j < i - window_size + 1
    mask = future | too_far
    return mask


mask = build_sliding_window_mask(seq_len=8, window_size=3, device="cpu")
print(mask.int())
```

注意：这个 demo 只构造了 mask，帮助你理解哪些连接被禁止。它本身不会让 attention 变快；如果底层仍然计算完整 `[T,T]` logits，再把远处位置 mask 掉，计算量仍接近 dense attention。真正省计算需要配套 sliding-window attention kernel、block-sparse kernel，或者改变 attention 的计算图。

### 一个最小示例：Attention 成本估算

下面这个 demo 用粗略元素量估算 dense attention、sliding window attention 和 KV Cache 的增长趋势。它不是 profiler，但能帮助你在面试中快速量级估算。

```python
def gb(num_elements, bytes_per_element=2):
    return num_elements * bytes_per_element / 1024**3


def estimate_attention_cost(seq_len, layers=32, heads=32, kv_heads=32, d_h=128, window_size=4096):
    dense_pairs_per_layer = heads * seq_len * seq_len
    window_pairs_per_layer = heads * seq_len * min(seq_len, window_size)
    kv_cache_elements = 2 * layers * seq_len * kv_heads * d_h

    return {
        "seq_len": seq_len,
        "dense_attention_pairs_per_layer": dense_pairs_per_layer,
        "sliding_window_pairs_per_layer": window_pairs_per_layer,
        "attention_matrix_gb_per_layer_bf16": gb(dense_pairs_per_layer),
        "kv_cache_gb_all_layers_bf16": gb(kv_cache_elements),
    }


for seq_len in [4096, 32768]:
    stats = estimate_attention_cost(seq_len)
    print(f"seq_len={stats['seq_len']}")
    print(f"  dense pairs/layer: {stats['dense_attention_pairs_per_layer']:,}")
    print(f"  window pairs/layer: {stats['sliding_window_pairs_per_layer']:,}")
    print(f"  dense attention matrix/layer: {stats['attention_matrix_gb_per_layer_bf16']:.2f} GB")
    print(f"  KV cache/all layers: {stats['kv_cache_gb_all_layers_bf16']:.2f} GB")
```

你会看到两个关键现象：

1. dense attention 的 pair 数从 4K 到 32K 增加 64 倍。
2. 当 window size 固定为 4096 时，sliding window 的 pair 数随 `T` 近似线性增长。

### 面试中怎么讲

如果面试官问“标准 attention 的瓶颈是什么”，可以这样回答：

```text
标准 self-attention 需要计算 QK^T 和 AV，其中 attention logits 和 attention matrix 的大小是 T×T。单个 head 的核心计算约为 O(T² d_h)，多头合起来约为 O(T² d_model)，训练显存也会随 T² 增长。长上下文下，T 从 4K 到 32K 会让 attention 交互数增加 64 倍。推理时虽然有 KV Cache，但每个新 token 仍要和历史 K 做 attention，KV Cache 本身也会随长度线性增长。
```

如果面试官问“FlashAttention 和稀疏 attention 有什么区别”，可以这样回答：

```text
FlashAttention 是 IO-aware 的精确 dense attention，它通过分块计算和避免显式 materialize 完整 attention matrix 来减少显存读写和提升速度，但并没有改变 attention 的 O(T²) 交互数量。稀疏 attention 或线性 attention 则会改变连接模式或近似 softmax attention，目标是降低复杂度阶数或实际连接数，但可能引入能力损失和实现复杂度。
```

### 真实项目中的坑

#### 坑 1：以为加 mask 就能加速

如果仍然构造完整 `T × T` logits，只是把一部分位置 mask 成 `-inf`，计算量并没有真正省掉。

真正加速需要 sparse kernel、block-wise 实现或改变计算图。

#### 坑 2：把 FlashAttention 说成线性 attention

FlashAttention 是精确 dense attention 的 IO 优化，不是稀疏或线性近似。

#### 坑 3：只看理论复杂度，不看硬件效率

某些稀疏算法理论复杂度更低，但 GPU 上可能因为访存不连续、kernel 不成熟而不快。

#### 坑 4：以为长上下文等于改 attention

长上下文还需要位置编码、长序列训练数据、推理系统、KV Cache 管理和真实评估。

#### 坑 5：忽略能力损失

稀疏 attention 删除连接后，可能影响长程检索、跨段推理和代码依赖分析。

### 面试官会怎么问

#### 问题 1：标准 attention 的时间复杂度和显存复杂度是多少？

回答框架：

1. `QK^T` 是 `O(T² d_h)`。
2. `AV` 也是 `O(T² d_h)`。
3. 多头合起来约为 `O(T² d_model)`。
4. attention logits/probabilities 是 `T × T`，训练显存也随 `T²` 增长。

#### 问题 2：为什么长上下文推理仍然贵？

回答框架：

1. KV Cache 避免重复计算历史 K/V。
2. 但每个新 token 仍要和所有历史 K 做 attention。
3. KV Cache 本身随上下文长度、层数和 head 数增长。
4. 长上下文推理常受显存容量和带宽限制。

#### 问题 3：local attention 的优缺点是什么？

回答框架：

1. 优点是复杂度从 `O(T²)` 降到 `O(Tw)`。
2. 适合局部依赖和流式场景。
3. 缺点是远距离 token 不能直接交互。
4. 长程依赖需要多层传递或额外全局连接。

#### 问题 4：线性 attention 为什么能降复杂度？

回答框架：

1. 它用 kernel feature map 近似或改写 softmax attention。
2. 把计算顺序从 `softmax(QK^T)V` 变成类似 `φ(Q)[φ(K)^T V]`。
3. 避免显式构造 `T × T` attention matrix。
4. 代价是可能不是精确 softmax attention，能力和稳定性要验证。

#### 问题 5：如何设计一个支持长上下文的 LLM 系统？

回答框架：

1. 架构上考虑 dense attention 优化、稀疏 attention、GQA/MQA 或 sliding window。
2. 位置编码上考虑 RoPE scaling、ALiBi 或长上下文位置策略。
3. 训练上加入长序列数据和长程依赖任务。
4. 推理上管理 KV Cache、分页、压缩和调度。
5. 应用上结合 RAG、chunking、摘要和外部 memory。
6. 评估上覆盖 needle-in-a-haystack、跨段推理和真实长文任务。

### 常见误区

1. 误区：稀疏 attention 一定比 dense attention 快。
   纠正：理论复杂度低不等于硬件上更快，kernel 和访存模式很关键。

2. 误区：FlashAttention 解决了 attention 的平方复杂度。
   纠正：FlashAttention 优化 IO 和显存，不改变 dense attention 的交互数量。

3. 误区：只要 context window 很大，模型就能使用所有信息。
   纠正：模型可能 lost in the middle，也可能没有经过长程依赖训练。

4. 误区：local attention 适合所有长文本任务。
   纠正：需要远距离证据整合的任务可能明显受损。

5. 误区：长上下文可以完全替代 RAG。
   纠正：长上下文和 RAG 解决的问题不同，RAG 还能提供外部知识更新、引用和过滤。

### 小练习

1. 推导标准 attention 中 `QK^T` 和 `AV` 的复杂度。
2. 估算 `seq_len=32768` 时单个 attention matrix 的元素数量。
3. 写一个 sliding window causal mask，并说明它是否真的减少计算。
4. 比较 local attention、block sparse attention 和 linear attention 的优缺点。
5. 用自己的话解释 FlashAttention 和稀疏 attention 的区别。
6. 用 2 分钟回答“如何支持 1M context”。

### 本讲总结

本讲最重要的结论：

1. 标准 attention 的核心瓶颈来自 `T × T` attention matrix，计算和训练显存都随序列长度平方增长。
2. KV Cache 能减少自回归推理中的重复计算，但不能消除长上下文下读取历史 K/V 的成本。
3. 稀疏 attention 通过减少 token-to-token 连接降低复杂度，但会引入能力和实现 trade-off。
4. local、sliding window、global token、block sparse、低秩和线性 attention 是不同方向的近似或结构设计。
5. FlashAttention 是精确 dense attention 的 IO 优化，不是线性 attention，也不改变复杂度阶数。
6. 长上下文能力不只取决于 attention，还取决于位置编码、训练数据、推理系统、RAG/压缩策略和评估方法。

## 第 13 讲：Pre-LN、Post-LN、Sandwich-LN 与 DeepNorm

### 本讲目标

Transformer 能不能稳定训练到很深，和 LayerNorm 放在哪里、残差分支如何缩放密切相关。第一册和第 7 讲已经讲过 Pre-LN 与 Post-LN 的基础，本讲进一步比较 Pre-LN、Post-LN、Sandwich-LN 和 DeepNorm，重点理解它们如何影响梯度传播、训练稳定性和深层 Transformer 扩展。

你需要掌握：

1. Pre-LN 和 Post-LN 的公式差异。
2. 为什么 Post-LN 早期常见但深层训练更难。
3. 为什么 Pre-LN 更稳定，但也有 trade-off。
4. Sandwich-LN 的动机。
5. DeepNorm 通过 residual scaling 稳定深层 Transformer 的思想。
6. 面试中如何把 normalization placement 和训练稳定性讲清楚。

### 为什么 LayerNorm 位置重要

Transformer block 不是单个函数，而是多层残差结构反复堆叠。

简化表示为：

```text
x_{l+1} = x_l + F_l(x_l)
```

其中 `F_l` 可以是 attention 或 MLP。

LayerNorm 放在 `F_l` 前面还是后面，会改变：

1. 前向激活尺度。
2. residual stream 的传播路径。
3. 反向梯度路径。
4. 对学习率和初始化的敏感性。
5. 深层模型训练稳定性。

所以归一化位置不是实现细节，而是架构设计问题。

### Post-LN Transformer

Post-LN 的形式是：

```text
x_{l+1} = LN(x_l + F_l(x_l))
```

也就是先做子层变换和 residual add，再做 LayerNorm。

早期 Transformer 使用 Post-LN。

它的优点是每一层输出都会被归一化，进入下一层时尺度更受控。

但它的问题是：

```text
residual identity path 被 LayerNorm 包住了
```

反向传播时，梯度需要穿过很多 LayerNorm。

当模型很深时，梯度传播可能更困难，训练对 warmup、初始化和学习率更敏感。

### Pre-LN Transformer

Pre-LN 的形式是：

```text
x_{l+1} = x_l + F_l(LN(x_l))
```

也就是先对输入归一化，再进入 attention 或 MLP，最后加回 residual。

Pre-LN 的关键优势是 residual 主路径更直接。

梯度可以沿着 identity path 传播：

```text
x_l -> x_l + F_l(LN(x_l))
```

这让深层模型更容易训练。

现代 decoder-only LLM 大多采用 Pre-LN 或 RMSNorm + Pre-Norm 形式。

### Pre-LN 的优点

Pre-LN 常见优点：

1. 深层训练更稳定。
2. 对学习率 warmup 不那么敏感。
3. 梯度可以更直接地穿过 residual stream。
4. loss spike 和发散风险更低。
5. 更适合大规模 decoder-only LLM。

从工程角度看，Pre-LN 降低了训练很深 Transformer 的难度。

如果你要训练几十层甚至上百层模型，Pre-LN 通常比 Post-LN 更稳。

### Pre-LN 的 trade-off

Pre-LN 也不是完美方案。

常见 trade-off 包括：

1. 每个子层的输出没有立即被归一化后再进入 residual 主干。
2. residual stream 的尺度可能随层数累积。
3. 最终输出前通常需要 final norm。
4. 某些研究认为 Pre-LN 可能让深层更新更像小扰动，影响有效深度利用。

可以直观理解为：

```text
Pre-LN 让训练更稳，但稳定不等于一定最优表达
```

实践中，现代 LLM 更看重大规模训练可控性，所以 Pre-LN 成为主流。

### Post-LN 的优点和风险

Post-LN 的优点：

1. 每层输出尺度被强制归一化。
2. 原始 Transformer 结构清晰。
3. 在较浅模型或合适初始化下可以工作。
4. 某些设置下可能有较强表示更新。

风险：

1. 深层训练更不稳定。
2. 对 warmup 和学习率更敏感。
3. 梯度传播经过多层 norm，容易出现训练困难。
4. 大模型中调参成本高。

所以面试中不要说 Post-LN 错了，而要说它在深层大模型训练中更难稳定。

### Sandwich-LN 是什么

Sandwich-LN 的思想是在子层前后都放 normalization。

一种简化形式可以写成：

```text
x_{l+1} = x_l + LN_2(F_l(LN_1(x_l)))
```

直觉是：

1. 前面的 norm 稳定子层输入。
2. 后面的 norm 控制子层输出尺度。
3. residual 主路径仍然保留。

它试图结合 Pre-LN 和 Post-LN 的部分优点。

但代价是：

1. 额外 norm 带来计算开销。
2. 结构更复杂。
3. 不一定在所有规模和任务上收益明显。
4. 可能改变 residual update 的尺度和表达。

Sandwich-LN 更像一种稳定性增强设计，而不是所有 LLM 的默认标准。

### DeepNorm 的动机

当 Transformer 非常深时，只靠普通 residual 和 norm 可能仍然不够稳定。

DeepNorm 的核心动机是：

```text
通过特定 residual scaling 和初始化，让非常深的 Transformer 仍能稳定训练
```

深层网络中，每层 residual update 都会累加到主干。

如果每层更新尺度不受控，层数越多越容易不稳定。

DeepNorm 通过给 residual 分支和参数初始化加入和层数相关的缩放系数，控制深层累积效应。

DeepNet 论文把 DeepNorm 定位为一种稳定极深 Transformer 的 normalization/residual 设计：它修改 residual connection，并配套理论推导的初始化，使模型更新在很深层数下仍更可控。面试中不需要背具体系数，但要讲清楚它不是单独“换一个 LayerNorm”，而是 residual scaling 和 initialization 的组合设计。

### DeepNorm 的直觉

普通 residual 是：

```text
x_{l+1} = x_l + F_l(x_l)
```

DeepNorm 类思想会引入缩放：

```text
x_{l+1} = α x_l + F_l(x_l)
```

或者对 residual branch 和初始化做配套缩放。

这里不要求死记具体论文中的所有系数。

面试中更重要的是讲清楚直觉：

```text
层数越深，residual 更新的累积越需要控制；DeepNorm 用和深度相关的缩放规则稳定前向表示和反向梯度
```

### Residual Scaling 为什么有效

假设每一层都往 residual stream 中加入一个更新 `Δx_l`。

如果每个 `Δx_l` 的尺度都不小，堆叠很多层后：

```text
x_L = x_0 + Σ_l Δx_l
```

表示尺度可能随层数增大。

residual scaling 的想法是让每层更新更温和，避免深层累积过强。

类似思想也出现在：

1. 初始化缩放。
2. residual branch scaling。
3. attention output projection 缩放。
4. DeepNet/DeepNorm 系列设计。

### 和现代 LLM 的关系

现代 LLM 常见稳定组合包括：

1. Pre-Norm 或 RMSNorm。
2. residual connection。
3. 合理初始化。
4. learning rate warmup。
5. gradient clipping。
6. BF16 训练。
7. attention 和 MLP 输出投影的尺度控制。

DeepNorm、Sandwich-LN 等方法代表的是更广义的稳定深层 Transformer 的设计空间。

实际是否采用，要看模型规模、训练数据、框架实现和实验结果。

### 如何选择 Norm 结构

如果是训练通用 decoder-only LLM，一个稳妥默认选择通常是：

```text
RMSNorm + Pre-Norm + residual connection + final norm
```

如果是研究极深 Transformer，可以考虑：

1. residual scaling。
2. DeepNorm 类设计。
3. 更谨慎的初始化。
4. 更长 warmup。
5. 更细粒度的激活和梯度监控。

如果是复现原始 Transformer 或较浅模型，Post-LN 也可能正常工作。

关键不是背“哪个最好”，而是说清楚场景和约束。

### 一个最小代码对比

下面是 Pre-LN 和 Post-LN block 的简化对比。

```python
import torch.nn as nn


class PreLNBlock(nn.Module):
    def __init__(self, norm, sublayer):
        super().__init__()
        self.norm = norm
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(self.norm(x))


class PostLNBlock(nn.Module):
    def __init__(self, norm, sublayer):
        super().__init__()
        self.norm = norm
        self.sublayer = sublayer

    def forward(self, x):
        return self.norm(x + self.sublayer(x))
```

差异只是一行位置变化，但对深层训练影响很大。

### 最小 demo：比较 Pre-LN 和 Post-LN 的梯度流

下面这个 demo 构造多层简化 block，并比较输入端梯度范数。它不是严谨实验，只用于观察：normalization 放在 residual path 的不同位置，会改变梯度传播行为。

```python
import torch
import torch.nn as nn


torch.manual_seed(0)


class TinyPreLNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class TinyPostLNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, x):
        return self.norm(x + self.ffn(x))


def input_grad_norm(block_cls, depth=12, d_model=16):
    model = nn.Sequential(*[block_cls(d_model) for _ in range(depth)])
    x = torch.randn(2, 4, d_model, requires_grad=True)
    loss = model(x)[..., :4].pow(2).mean()
    loss.backward()
    return x.grad.norm().item()


print("Pre-LN input grad norm:", round(input_grad_norm(TinyPreLNBlock), 6))
print("Post-LN input grad norm:", round(input_grad_norm(TinyPostLNBlock), 6))
```

这个 toy demo 不能替代真实大模型实验，但它能提醒你：Pre-LN/Post-LN 的差别不是语法差别，而是会改变 residual identity path 和梯度穿过多层 block 的方式。

### 面试中怎么讲

如果面试官问“Pre-LN 和 Post-LN 有什么区别”，可以这样回答：

```text
Post-LN 是先做子层变换和 residual add，再做 LayerNorm，形式是 LN(x + F(x))。Pre-LN 是先归一化再进入子层，形式是 x + F(LN(x))。Pre-LN 的 residual identity path 更直接，梯度更容易穿过很多层，所以深层 Transformer 训练更稳定。Post-LN 每层输出尺度更受控，但深层训练对 warmup、初始化和学习率更敏感。
```

如果面试官问“DeepNorm 解决什么问题”，可以这样回答：

```text
DeepNorm 关注非常深的 Transformer 训练稳定性。深层 residual 网络中，每层 residual update 会不断累积，可能导致前向表示和反向梯度不稳定。DeepNorm 通过和层数相关的 residual scaling 以及配套初始化，控制 residual 累积效应，让更深的 Transformer 也能稳定训练。
```

### 真实项目中的坑

#### 坑 1：只改 norm 位置，不调其他超参

Pre-LN、Post-LN、DeepNorm 对学习率、warmup、初始化都有影响，不能孤立比较。

#### 坑 2：把训练稳定等同于最终效果

Pre-LN 更稳，但最终效果仍要看数据、规模、训练预算和评估。

#### 坑 3：忽略 final norm

Pre-LN 架构中，最终输出前通常需要 final norm 来稳定输出表示。

#### 坑 4：照搬论文系数

DeepNorm 类缩放系数和具体架构、层数、encoder/decoder 设置相关，照搬可能不适配。

#### 坑 5：没有监控 residual stream 尺度

深层训练中，activation norm、gradient norm、update norm 都应该监控。

### 面试官会怎么问

#### 问题 1：Pre-LN 为什么更稳定？

回答框架：

1. Pre-LN 形式是 `x + F(LN(x))`。
2. residual identity path 更直接。
3. 梯度可以沿主路径传播，不必每层都穿过 post norm。
4. 深层训练对学习率和 warmup 不那么敏感。

#### 问题 2：Post-LN 有什么问题？

回答框架：

1. 形式是 `LN(x + F(x))`。
2. 每层输出尺度受控。
3. 但 residual path 被 norm 包住。
4. 深层时梯度传播更难，训练更敏感。

#### 问题 3：Sandwich-LN 的直觉是什么？

回答框架：

1. 子层输入前做 norm，稳定输入。
2. 子层输出后再做 norm，控制输出尺度。
3. 尝试结合 Pre-LN 和 Post-LN 的优点。
4. 代价是额外计算和结构复杂度。

#### 问题 4：DeepNorm 和普通 residual 有什么区别？

回答框架：

1. 普通 residual 直接累加每层更新。
2. 深层时 residual update 累积可能不稳定。
3. DeepNorm 用和层数相关的 residual scaling 和初始化控制尺度。
4. 目标是稳定训练更深 Transformer。

#### 问题 5：训练很深 Transformer 时你会监控什么？

回答框架：

1. training loss 和 validation loss。
2. activation norm 和 residual stream norm。
3. gradient norm 和 update norm。
4. 每层 norm 参数变化。
5. loss spike、NaN、Inf。
6. 不同层的梯度是否消失或爆炸。

### 常见误区

1. 误区：Post-LN 是错误设计。
   纠正：Post-LN 在原始 Transformer 和较浅模型中可以工作，只是在深层大模型中更难稳定。

2. 误区：Pre-LN 全面优于 Post-LN。
   纠正：Pre-LN 更稳定，但也有 residual 尺度累积和有效深度利用等 trade-off。

3. 误区：DeepNorm 只是多乘一个常数。
   纠正：DeepNorm 的核心是 residual scaling 和初始化配套，目标是稳定深层训练动态。

4. 误区：归一化位置只是代码风格。
   纠正：norm placement 会改变梯度路径和训练稳定性。

5. 误区：训练不稳定只靠换 norm 解决。
   纠正：还要看学习率、warmup、初始化、数据、精度和 optimizer。

### 小练习

1. 写出 Pre-LN 和 Post-LN 的公式。
2. 画出两者的残差路径和梯度路径。
3. 解释为什么 Pre-LN 更适合深层 Transformer。
4. 用自己的话解释 Sandwich-LN 的动机。
5. 解释 DeepNorm 为什么需要和层数相关的缩放。
6. 用 2 分钟回答“训练 100 层 Transformer 时 norm placement 为什么重要”。

### 本讲总结

本讲最重要的结论：

1. LayerNorm 放在 residual 前还是后，会显著影响深层 Transformer 的梯度传播和训练稳定性。
2. Post-LN 每层输出尺度更受控，但深层训练更敏感。
3. Pre-LN 保留更直接的 residual identity path，因此更适合现代深层 LLM。
4. Sandwich-LN 尝试同时稳定子层输入和输出，但会增加复杂度。
5. DeepNorm 通过 residual scaling 和配套初始化稳定非常深的 Transformer。
6. norm placement 不能孤立讨论，必须结合初始化、学习率、warmup、残差缩放和训练监控一起看。

## 第 14 讲：RMSNorm、ScaleNorm 与归一化简化

### 本讲目标

LayerNorm 是 Transformer 的关键组件，但现代 LLM 中越来越常见的是 RMSNorm 等更简化的归一化方法。本讲讨论为什么归一化可以简化、RMSNorm 和 ScaleNorm 分别做了什么、它们和 LayerNorm 的差异，以及为什么很多 LLaMA 类模型选择 RMSNorm。

你需要掌握：

1. LayerNorm 的完整计算过程。
2. RMSNorm 为什么去掉均值中心化。
3. ScaleNorm 的核心思想。
4. 归一化简化如何影响速度、稳定性和表达。
5. 为什么现代 LLM 常用 RMSNorm。
6. 面试中如何比较 LayerNorm、RMSNorm 和 BatchNorm。

### 先回顾 LayerNorm

给定一个 token 的 hidden state：

```text
x ∈ R^d
```

LayerNorm 会先计算均值和方差：

```text
mu = mean(x)
var = mean((x - mu)^2)
```

然后归一化：

```text
y = (x - mu) / sqrt(var + eps) * gamma + beta
```

它做了两件事：

1. mean centering：减去均值。
2. variance normalization：除以标准差。

最后再用可学习参数 `gamma` 和 `beta` 调整尺度和平移。

### LayerNorm 的作用

LayerNorm 的核心作用是控制每个 token 的 hidden state 尺度。

它不依赖 batch 统计，而是在 hidden dimension 上做归一化。

这非常适合语言模型，因为：

1. batch size 可能动态变化。
2. 序列长度可能不同。
3. 自回归推理时不能依赖训练 batch 统计。
4. 每个 token 都需要独立稳定的表示尺度。

LayerNorm 帮助深层 Transformer 减少激活尺度漂移，提高训练稳定性。

### 为什么要简化 LayerNorm

既然 LayerNorm 很好，为什么还要简化？

原因包括：

1. 大模型中 norm 会被调用非常多次。
2. 每次 norm 都涉及 reduce 操作。
3. 均值和方差计算有一定开销。
4. 一些研究发现均值中心化不是总是必要。
5. 简化 norm 可能在保持稳定性的同时提高效率。

在超大模型中，一个看似小的算子差异，乘以层数、token 数和训练步数后，都会变成显著成本。

### RMSNorm 的公式

RMSNorm 去掉了均值中心化，只保留 root mean square 归一化。

公式是：

```text
rms(x) = sqrt(mean(x^2) + eps)
y = x / rms(x) * gamma
```

它不计算 `mu`，也不做 `x - mu`。

相比 LayerNorm：

```text
LayerNorm:  normalize by mean and variance
RMSNorm:    normalize by vector magnitude
```

RMSNorm 的核心思想是：

```text
对 Transformer hidden state 来说，控制表示尺度可能比强制均值为 0 更关键
```

更严谨地说，RMSNorm 论文的表述不是“均值永远没用”，而是假设 LayerNorm 里的 re-centering invariance 可以被去掉，保留 re-scaling invariance 仍然足以在许多模型里稳定训练。也就是说，RMSNorm 控制的是向量幅度，不保证输出均值为 0。

### RMSNorm 的优点

RMSNorm 常见优点：

1. 计算更简单。
2. 少一次均值中心化。
3. 对大模型训练足够稳定。
4. 和 Pre-Norm decoder-only LLM 配合很好。
5. 被 LLaMA 等现代模型广泛采用。

它尤其适合大规模训练，因为它用更少操作保留了最关键的尺度控制。

### RMSNorm 的潜在 trade-off

RMSNorm 不是无条件更好。

它的 trade-off 包括：

1. 不做均值中心化，表示分布可能和 LayerNorm 不同。
2. 某些任务或架构中不一定优于 LayerNorm。
3. 需要和初始化、学习率、residual 设计一起验证。
4. 不能简单从公式判断最终效果。

所以更准确的说法是：

```text
RMSNorm 是一种更简洁、工程上高效、在现代 LLM 中被验证有效的归一化选择
```

### ScaleNorm 的思想

ScaleNorm 进一步强调向量范数控制。

它的思想是把向量缩放到固定尺度附近。

简化形式可以写成：

```text
y = g * x / ||x||_2
```

其中 `g` 是一个标量尺度参数，原始 ScaleNorm 设计强调用 L2 normalization 加单一 scale 来替代 LayerNorm 中逐维的 `gamma` 和 `beta`。实际实现通常会加 `eps`，避免输入范数过小时除零。

ScaleNorm 的直觉是：

```text
与其分别控制每个维度，不如控制整个 hidden vector 的范数
```

它比 LayerNorm 更简单，但在现代通用 LLM 中没有 RMSNorm 那么主流。

理解 ScaleNorm 的价值在于看到归一化设计的一个方向：从复杂统计量转向尺度控制。

### RMSNorm、ScaleNorm 和 LayerNorm 对比

可以用下面方式比较：

```text
LayerNorm: 减均值 + 除标准差 + gamma/beta
RMSNorm:   不减均值，只除 RMS + gamma
ScaleNorm: 用 L2 范数缩放 + 单一 scale
```

从复杂度看：

1. LayerNorm 统计最完整。
2. RMSNorm 保留尺度控制，去掉均值中心化。
3. ScaleNorm 更强调整体范数。

从实践看：

1. LayerNorm 是经典 Transformer 默认选项。
2. RMSNorm 是现代 decoder-only LLM 常见选项。
3. ScaleNorm 是重要思想，但不是当前最主流 LLM 默认配置。

### 为什么 LLaMA 类模型常用 RMSNorm

LLaMA 类模型常见组合是：

```text
RMSNorm + Pre-Norm + SwiGLU + RoPE
```

RMSNorm 适合这个组合的原因包括：

1. Pre-Norm 已经需要稳定子层输入尺度。
2. RMSNorm 足够控制 hidden state 的幅度。
3. 省掉均值中心化，计算更轻。
4. 大规模实验验证效果好。
5. 与 SwiGLU、RoPE 等现代组件配合稳定。

面试中可以强调：

```text
RMSNorm 的流行不是因为理论上永远优于 LayerNorm，而是因为它在大规模 decoder-only LLM 中提供了很好的稳定性、效率和效果平衡
```

### Norm 和数值稳定性

归一化层中通常有一个 `eps`：

```text
sqrt(var + eps)
sqrt(mean(x^2) + eps)
```

`eps` 的作用是避免除以非常小的数。

如果 `eps` 太小，在混合精度训练中可能引入数值风险。

如果 `eps` 太大，又可能改变归一化行为。

真实训练中，norm 相关问题可能表现为：

1. activation norm 异常。
2. loss spike。
3. NaN 或 Inf。
4. 某些层输出尺度漂移。
5. mixed precision 下不稳定。

所以 norm 虽然看起来简单，但实现细节很重要。

### Norm 参数是否做 Weight Decay

训练 LLM 时，norm 参数通常不做 weight decay。

原因是 norm 的 `gamma` 主要控制尺度，不像普通 Linear weight 那样承担特征变换。

如果对 norm 参数做 decay，可能干扰尺度调节。

常见参数分组是：

```text
decay:    linear weights
no_decay: bias, norm weights
```

这不是数学定律，而是被大量工程实践采用的经验。

### 一个最小 RMSNorm 实现

下面是简化版 PyTorch RMSNorm。

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

如果使用较新版本 PyTorch，也可以直接查看 `torch.nn.RMSNorm`。但面试和学习时最好能手写最小版本，因为这能说明 RMSNorm 和 LayerNorm 的统计维度差异。

输入输出 shape：

```text
x:   [batch, seq_len, hidden_dim]
out: [batch, seq_len, hidden_dim]
```

它只在最后一个 hidden dimension 上计算 RMS，不依赖 batch 或 seq_len 的统计。

### 可运行对比 demo

下面这个 demo 同时实现 LayerNorm、RMSNorm 和 ScaleNorm，并打印每个 token 输出的均值、RMS 和 L2 norm。它展示三个差异：LayerNorm 会把均值拉到接近 0；RMSNorm 不保证均值为 0，但会把 RMS 控制到接近 1；ScaleNorm 控制的是整个向量的 L2 norm。

```python
import torch
import torch.nn as nn


class ManualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class ManualRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ManualScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(dim ** 0.5))

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


def summarize(name, y):
    token_mean = y.mean(dim=-1)
    token_rms = torch.sqrt((y * y).mean(dim=-1))
    token_l2 = torch.linalg.vector_norm(y, ord=2, dim=-1)
    print(f"{name:10s} mean={token_mean.flatten().tolist()}")
    print(f"{name:10s} rms ={token_rms.flatten().tolist()}")
    print(f"{name:10s} l2  ={token_l2.flatten().tolist()}")


torch.manual_seed(0)
x = torch.randn(2, 3, 4) * 2.0 + 5.0

layers = {
    "LayerNorm": ManualLayerNorm(dim=4),
    "RMSNorm": ManualRMSNorm(dim=4),
    "ScaleNorm": ManualScaleNorm(dim=4),
}

for name, layer in layers.items():
    y = layer(x)
    assert y.shape == x.shape
    summarize(name, y)
```

典型输出会接近下面这样：

```text
LayerNorm  mean=[0.0, 0.0, ...]
LayerNorm  rms =[1.0, 1.0, ...]
RMSNorm    mean=[0.93, 0.99, ...]
RMSNorm    rms =[1.0, 1.0, ...]
ScaleNorm  mean=[0.93, 0.99, ...]
ScaleNorm  l2  =[2.0, 2.0, ...]
```

这里 hidden dimension 是 4，所以 ScaleNorm 的默认 `scale = sqrt(4) = 2` 时，输出 L2 norm 接近 2；RMSNorm 输出 RMS 接近 1，因此它的 L2 norm 也会接近 `sqrt(4)`。两者数值上可能接近，但参数化不同：RMSNorm 通常有逐维 `weight`，ScaleNorm 是单一标量 scale。

### 面试中怎么讲

如果面试官问“RMSNorm 和 LayerNorm 有什么区别”，可以这样回答：

```text
LayerNorm 会对每个 token 的 hidden dimension 计算均值和方差，先减均值再除以标准差，然后用可学习参数缩放和平移。RMSNorm 去掉了均值中心化，只用 root mean square 控制向量尺度，形式是 `x / rms(x) * gamma`。它更简单、计算更轻，在很多 decoder-only LLM 中足够稳定，因此被 LLaMA 类模型广泛使用。
```

如果面试官问“为什么不用 BatchNorm”，可以这样回答：

```text
BatchNorm 依赖 batch 维度统计，而语言模型有变长序列、动态 batch、自回归推理和分布式训练问题，训练和推理统计也不容易一致。LayerNorm 和 RMSNorm 都是在单个 token 的 hidden dimension 上归一化，不依赖 batch，因此更适合 LLM。
```

### 真实项目中的坑

#### 坑 1：把 RMSNorm 当成 LayerNorm 的完全等价替代

RMSNorm 不做均值中心化，表示分布会不同。替换时需要重新验证训练稳定性和效果。

#### 坑 2：忽略 epsilon

`eps` 影响数值稳定性，尤其在 mixed precision 训练中不能随便设置。

#### 坑 3：对 norm 参数做 weight decay

这可能干扰尺度参数，通常 norm weights 会放入 no_decay 参数组。

#### 坑 4：只看理论 FLOPs

norm 的实际速度还受 kernel、融合算子、内存访问和框架实现影响。

#### 坑 5：孤立比较 norm

norm 效果和 Pre-LN/Post-LN、初始化、学习率、激活函数、residual scaling 都相关。

### 面试官会怎么问

#### 问题 1：RMSNorm 为什么可以去掉均值中心化？

回答框架：

1. 它假设控制 hidden vector 的整体尺度是最关键的。
2. 不强制表示均值为 0。
3. 在很多 Transformer LLM 中，尺度控制已经足够稳定训练。
4. 省掉均值计算可以降低开销。

#### 问题 2：ScaleNorm 的直觉是什么？

回答框架：

1. 控制整个 hidden vector 的范数。
2. 用 `x / ||x||_2` 再乘尺度参数。
3. 比 LayerNorm 更简化。
4. 体现归一化可以从统计标准化转向尺度控制。

#### 问题 3：为什么 norm 参数通常不做 weight decay？

回答框架：

1. norm 参数主要控制尺度。
2. decay 可能干扰尺度调节。
3. 它们不是普通特征变换矩阵。
4. 工程上通常把 norm weights 和 bias 放进 no_decay 组。

#### 问题 4：RMSNorm 对训练稳定性有什么帮助？

回答框架：

1. 控制每个 token hidden state 的尺度。
2. 减少激活尺度漂移。
3. 和 Pre-Norm residual 结构配合，稳定子层输入。
4. 有助于深层 Transformer 训练。

#### 问题 5：LayerNorm、RMSNorm、BatchNorm 如何比较？

回答框架：

1. BatchNorm 用 batch 统计，不适合 LLM 自回归场景。
2. LayerNorm 用 token 内 hidden 维度统计，经典稳定。
3. RMSNorm 去掉均值中心化，只做 RMS 尺度控制，更轻量。
4. 现代 decoder-only LLM 常用 RMSNorm。

### 常见误区

1. 误区：RMSNorm 一定比 LayerNorm 好。
   纠正：RMSNorm 在很多 LLM 中效果好，但不是所有架构和任务都必然更优。

2. 误区：归一化只是为了防止 NaN。
   纠正：归一化还影响梯度传播、激活尺度、学习率可用范围和深层训练稳定性。

3. 误区：BatchNorm、LayerNorm、RMSNorm 只是实现不同。
   纠正：它们统计维度和训练/推理行为不同，适用场景不同。

4. 误区：去掉均值中心化一定会损失表达能力。
   纠正：是否损失要看架构和训练结果，现代 LLM 中 RMSNorm 已被广泛验证。

5. 误区：norm 参数很少，可以随便处理。
   纠正：norm 参数虽少，但对稳定性敏感，weight decay 和精度处理都要注意。

### 小练习

1. 写出 LayerNorm 和 RMSNorm 的公式。
2. 解释 RMSNorm 为什么更轻量。
3. 实现一个 RMSNorm，并标注输入输出 shape。
4. 说明为什么 BatchNorm 不适合自回归 LLM。
5. 设计一个实验比较 LayerNorm 和 RMSNorm 的训练稳定性。
6. 用 2 分钟回答“为什么 LLaMA 类模型常用 RMSNorm”。

### 本讲总结

本讲最重要的结论：

1. LayerNorm 通过减均值和除标准差控制每个 token 的 hidden state 分布。
2. RMSNorm 去掉均值中心化，只控制 RMS 尺度，计算更简单。
3. ScaleNorm 进一步强调整体向量范数控制。
4. 现代 decoder-only LLM 常用 RMSNorm，是稳定性、效率和效果之间的工程平衡。
5. norm 参数通常不做 weight decay，epsilon 和 mixed precision 实现也会影响稳定性。
6. 归一化方法必须和 Pre-Norm、初始化、学习率、残差结构和训练监控一起理解。

## 第 15 讲：FFN、GeLU、SwiGLU 与门控 MLP

### 本讲目标

第 8 讲已经从深度学习基础角度讲过激活函数和 SwiGLU。本讲放在 Transformer 架构进阶里，重点从架构设计、参数量、FLOPs、表示能力和现代 LLM 案例角度理解 FFN/MLP 子层。

你需要掌握：

1. Transformer FFN 的标准结构。
2. FFN 在 token mixing 和 feature transformation 中的角色。
3. GeLU、SiLU、SwiGLU 的架构意义。
4. 门控 MLP 为什么提升表达能力。
5. SwiGLU hidden dimension 如何影响参数量和计算量。
6. FFN 与 attention 的参数和 FLOPs 对比。

### 标准 FFN 结构

Transformer 中的 FFN 通常对每个 token 独立作用。

给定 hidden state：

```text
x shape: [T, d_model]
```

标准 FFN 是：

```text
FFN(x) = W_2 activation(W_1 x)
```

其中：

```text
W_1: d_model -> d_ff
W_2: d_ff -> d_model
```

常见设置是：

```text
d_ff = 4 * d_model
```

FFN 的作用不是在 token 之间传信息，而是在每个 token 的 hidden dimension 上做非线性特征变换。

### Attention 和 FFN 的分工

一个简洁说法是：

```text
attention mixes tokens, FFN transforms features
```

attention 负责不同 token 之间的信息路由，FFN 负责对每个 token 的表示做非线性特征变换。

如果没有 FFN，Transformer 只靠 attention 做线性加权聚合，表达能力会明显受限。

### FFN 参数量为什么大

标准 FFN 参数量大约是：

```text
d_model * d_ff + d_ff * d_model = 2 * d_model * d_ff
```

如果 `d_ff = 4 * d_model`：

```text
params = 8 * d_model * d_model
```

而一个标准 multi-head attention 的 Q/K/V/O 投影参数量大约是：

```text
params = 4 * d_model * d_model
```

所以在很多 Transformer block 中，FFN 参数量比 attention 更大。

这也是为什么不能只关注 attention。LLM 的大量容量其实在 MLP/FFN 子层里。

### FFN FLOPs

对每个 token，标准 FFN 需要两次大矩阵乘法。

如果序列长度为 `T`，大致 FLOPs 是：

```text
O(T * d_model * d_ff)
```

在长上下文中，attention 的 `T^2` 复杂度会变得突出。但在普通上下文长度下，FFN 仍然占据大量计算。

### GeLU FFN

早期 GPT、BERT 类 Transformer 常用 GeLU：

```text
FFN(x) = W_2 GeLU(W_1 x)
```

GeLU 的优点是平滑，不像 ReLU 那样硬截断负值。

它可以理解为一种输入相关的软门控：

```text
GeLU(z) = z * Phi(z)
```

其中 `Phi(z)` 是标准正态 CDF。更准确地说，GeLU 不是像 ReLU 那样根据符号硬截断，而是用输入值对应的高斯 CDF 做平滑加权。

### 从 GeLU 到门控 MLP

普通 GeLU FFN 只有一条中间分支。

门控 MLP 会把中间层拆成内容分支和门控分支。

普通 FFN：

```text
hidden = activation(W_1 x)
out = W_2 hidden
```

门控 MLP：

```text
up = W_up x
gate = W_gate x
hidden = up * activation(gate)
out = W_down hidden
```

门控分支决定哪些特征通道被放大或抑制。

### GLU、GEGLU 和 SwiGLU

GLU 的基本形式是：

```text
GLU(x) = (W_a x) * sigmoid(W_g x)
```

GEGLU 使用 GeLU 门控：

```text
GEGLU(x) = (W_a x) * GeLU(W_g x)
```

SwiGLU 使用 SiLU/Swish 门控：

```text
SwiGLU(x) = (W_a x) * SiLU(W_g x)
```

现代 LLM 中常见的是 SwiGLU 变体：

```text
MLP(x) = W_down [SiLU(W_gate x) * W_up x]
```

这里的 `*` 表示逐元素相乘，不是矩阵乘法。GLU 原始形式来自门控卷积语言模型，SwiGLU/GEGLU 则是在 Transformer FFN 子层里把 sigmoid gate 换成 SiLU/GeLU gate 的变体；面试中不要把它说成 attention 机制，它仍然是逐 token 的 MLP 子层。

### 为什么门控有效

门控结构有效的原因包括：

1. 它提供输入相关的特征选择。
2. 它引入乘法交互，表达能力比单分支激活更强。
3. 它让 MLP 能动态调节不同特征通道。
4. 它和大规模语言数据中的上下文依赖模式匹配。

直觉上，普通 FFN 是把输入映射到一组中间特征再激活；门控 MLP 是一条分支生成候选特征，另一条分支决定这些特征通过多少。

### SwiGLU hidden dimension 为什么常不是 4 倍

普通 FFN 有两个矩阵，参数量约：

```text
2 * d_model * d_ff
```

SwiGLU 有三个矩阵，参数量约：

```text
3 * d_model * d_hidden
```

如果 `d_hidden = 4 * d_model`，SwiGLU 参数量会比普通 FFN 大很多。

为了让参数量接近普通 `4d` FFN，通常设置：

```text
3 * d_model * d_hidden = 8 * d_model * d_model
d_hidden = 8/3 * d_model
```

所以很多 LLaMA 类模型会使用接近 `2.67 * d_model` 的 intermediate size，并做硬件友好的取整。

### FFN 和模型容量

FFN 层是 Transformer 容量的重要来源。

原因是：

1. FFN 参数量大。
2. 每个 token 都经过 FFN 非线性变换。
3. FFN 扩展维度后再压缩，形成丰富特征组合。
4. 一些研究认为 FFN 可能存储大量事实和模式。

但要注意：知识不是只存在 FFN 中，而是分布在 embedding、attention、MLP、residual stream 和整体参数空间里。

### FFN 和 MoE 的关系

MoE 可以看成对 FFN 的一种扩展。

普通 Transformer 每个 token 都经过同一个 FFN。

MoE 中有多个 FFN expert，router 决定每个 token 去哪些 expert。

简化理解：

```text
Dense FFN: 所有 token 用同一个 MLP
MoE FFN:   不同 token 路由到不同 expert MLP
```

这能在增加参数量的同时控制每个 token 的激活计算量。

### 一个最小 SwiGLU MLP 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)
```

输入输出 shape：

```text
x:      [batch, seq_len, d_model]
gate:   [batch, seq_len, hidden_dim]
up:     [batch, seq_len, hidden_dim]
hidden: [batch, seq_len, hidden_dim]
out:    [batch, seq_len, d_model]
```

下面这个 demo 对比普通 `4d` GeLU FFN 和 `8/3 d` SwiGLU MLP 的 shape 与参数量。为了模拟真实模型里的硬件友好取整，`round_to_multiple` 会把 hidden dimension 向上取整到 8 的倍数。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def round_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


class GeLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up
        return self.down_proj(hidden)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


d_model = 24
d_ff = 4 * d_model
swiglu_hidden = round_to_multiple(8 * d_model // 3, 8)

gelu_ffn = GeLUFFN(d_model, d_ff)
swiglu_mlp = SwiGLUMLP(d_model, swiglu_hidden)

x = torch.randn(2, 5, d_model)
gelu_out = gelu_ffn(x)
swiglu_out = swiglu_mlp(x)

assert gelu_out.shape == x.shape
assert swiglu_out.shape == x.shape

print("GeLU FFN hidden:", d_ff)
print("SwiGLU hidden:", swiglu_hidden)
print("GeLU FFN params:", count_params(gelu_ffn))
print("SwiGLU params:", count_params(swiglu_mlp))
print("output shape:", list(swiglu_out.shape))
```

这个 demo 的重点不是证明 SwiGLU 一定更好，而是验证两个工程事实：第一，SwiGLU 增加了一条投影分支；第二，如果把 hidden dimension 调到接近 `8/3 * d_model`，参数量可以和普通 `4d` FFN 接近。

### 面试中怎么讲

如果面试官问“FFN 在 Transformer 中有什么作用”，可以这样回答：

```text
Attention 主要负责 token 之间的信息路由，而 FFN 主要负责对每个 token 的 hidden state 做非线性特征变换。FFN 通常先把维度从 d_model 扩展到更大的 d_ff，再经过激活或门控结构，最后投影回 d_model。它承担了大量参数和计算，是 Transformer 表达能力的重要来源，不能把模型能力全部归因于 attention。
```

如果面试官问“为什么现代 LLM 常用 SwiGLU”，可以这样回答：

```text
SwiGLU 是门控 MLP 的一种形式，它用一条 up 分支生成候选特征，用一条 gate 分支经过 SiLU 后控制这些特征通过多少。相比普通 GeLU FFN，SwiGLU 引入了输入相关的乘法调制，表达能力更强。因为它有三组投影矩阵，实践中通常会把 hidden dimension 调到约 8/3 d_model，使参数量接近普通 4d FFN。
```

### 真实项目中的坑

#### 坑 1：只比较效果，不比较参数量

SwiGLU 如果 hidden dimension 不调小，参数量和 FLOPs 会明显超过普通 FFN，比较不公平。

#### 坑 2：忽略 MLP 的推理成本

在普通上下文长度下，FFN/MLP 是 Transformer block 的主要计算来源之一。

#### 坑 3：把 FFN 说成简单的两层全连接

形式上是两层或三层投影，但它承担了高维非线性特征变换和大量模型容量。

#### 坑 4：认为知识只存在 FFN

FFN 可能编码大量模式，但知识通常是分布式的，不能简单定位到某个模块。

#### 坑 5：忽略硬件友好维度

intermediate size 通常会为了 tensor core、并行切分和 kernel 效率做取整。

### 面试官会怎么问

#### 问题 1：FFN 和 attention 的分工是什么？

回答框架：

1. attention 做 token mixing。
2. FFN 做 feature transformation。
3. attention 建立上下文信息流。
4. FFN 提供非线性特征组合和模型容量。

#### 问题 2：标准 FFN 的参数量是多少？

回答框架：

1. 标准 FFN 有 `d_model -> d_ff -> d_model` 两个矩阵。
2. 参数量约 `2 * d_model * d_ff`。
3. 如果 `d_ff = 4 * d_model`，约为 `8 * d_model * d_model`。
4. 通常比 attention 的 Q/K/V/O 投影参数量更大。

#### 问题 3：SwiGLU 和 GeLU FFN 有什么区别？

回答框架：

1. GeLU FFN 是单分支激活。
2. SwiGLU 有 up 分支和 gate 分支。
3. gate 分支经过 SiLU 后和 up 分支逐元素相乘。
4. 它提供输入相关的特征门控。

#### 问题 4：为什么 SwiGLU 的 hidden dimension 常用 8/3 倍？

回答框架：

1. 普通 4d FFN 参数量约 `8 * d * d`。
2. SwiGLU 有三个投影矩阵，参数量约 `3d*h`。
3. 为了参数量接近，令 `3 * d * h = 8 * d * d`。
4. 得到 `h = 8/3 * d`。

#### 问题 5：MoE 和 FFN 有什么关系？

回答框架：

1. MoE 通常把 dense FFN 替换为多个 expert FFN。
2. router 决定每个 token 去哪些 expert。
3. 它增加总参数量，但每个 token 只激活部分 expert。
4. 所以 MoE 是 FFN 层的条件计算扩展。

### 常见误区

1. 误区：Transformer 的关键只有 attention。
   纠正：FFN/MLP 承担大量参数、计算和非线性表达能力。

2. 误区：SwiGLU 是免费提升。
   纠正：它增加投影矩阵，必须调整 hidden dimension 才能公平比较参数量和 FLOPs。

3. 误区：FFN 不影响长上下文效率。
   纠正：attention 是 `T^2`，但 FFN 也随 token 数线性增长，在长序列下同样很贵。

4. 误区：GeLU 已经过时。
   纠正：GeLU 仍是经典稳定选择，SwiGLU 是现代 LLM 中更常见的高效表达结构。

5. 误区：MoE 只是把模型变大。
   纠正：MoE 的关键是条件计算，增加总参数量但控制每 token 激活计算。

### 小练习

1. 推导标准 FFN 的参数量。
2. 推导 SwiGLU hidden dimension 约为 `8/3 d_model` 的原因。
3. 手写一个 SwiGLU MLP，并标注每一步 shape。
4. 比较 attention 和 FFN 在参数量、FLOPs 和功能上的差异。
5. 用自己的话解释门控 MLP 为什么有效。
6. 用 2 分钟回答“为什么不能只关注 attention”。

### 本讲总结

本讲最重要的结论：

1. FFN/MLP 是 Transformer block 中负责非线性特征变换的核心模块。
2. 标准 FFN 参数量通常大于 attention 投影参数量，是模型容量的重要来源。
3. GeLU FFN 是经典结构，SwiGLU 通过门控分支提供更强表达能力。
4. SwiGLU 有三组投影矩阵，因此 hidden dimension 通常调到约 `8/3 d_model` 来控制参数量。
5. MoE 可以看成 FFN 的条件计算扩展。
6. 面试中要把 attention 的 token mixing 和 FFN 的 feature transformation 一起讲，才是完整 Transformer 视角。

## 第 16 讲：RoPE 的数学直觉与位置外推

### 本讲目标

位置编码是 Transformer 理解序列顺序的关键。本讲重点讲 RoPE 的数学直觉、它为什么比绝对位置编码更适合现代 LLM、以及它如何帮助长上下文外推。你还需要理解 RoPE 和 attention 的关系，以及为什么很多长上下文优化都围绕 RoPE 展开。

你需要掌握：

1. 为什么 attention 需要位置编码。
2. 绝对位置编码和相对位置编码的区别。
3. RoPE 的核心思想。
4. RoPE 为什么被称为旋转位置编码。
5. RoPE 如何把相对位置信息注入 Q 和 K。
6. RoPE 为什么更适合长度外推。
7. 面试中如何解释 RoPE 的数学直觉。

### 为什么 attention 需要位置编码

Self-Attention 本身对 token 顺序并不天然敏感。

如果只看 token 集合，不考虑位置，那么交换序列顺序后，模型可能看不到区别。

但语言是有顺序的：

1. “狗咬人”和“人咬狗”不同。
2. 代码顺序会影响语义。
3. 对话历史顺序会影响上下文。

所以必须把位置注入模型。

### 绝对位置编码

早期做法是给每个位置一个独立向量，和 token embedding 相加：

```text
x_i = token_embedding_i + position_embedding_i
```

这就是绝对位置编码的基本思路。

优点：

1. 简单。
2. 直观。
3. 让模型知道第几个 token。

缺点：

1. 位置表示和具体序列长度绑定较强。
2. 对超出训练长度的外推通常不够自然。
3. 相对距离信息需要模型自己间接学习。

### 相对位置编码的思路

相对位置编码关注的是 token 之间的距离，而不是绝对编号。

例如：

```text
pos(j) - pos(i)
```

在语言中，很多关系更关心相对位置：

1. 最近前文。
2. 跨句依赖。
3. 引用与被引用。
4. 代码中括号的匹配距离。

RoPE 就是把相对位置信息更自然地注入 attention 的一种方法。

### RoPE 的核心思想

RoPE 是 Rotary Position Embedding，核心思想是：

```text
把位置信息编码成对 Q 和 K 的旋转
```

它不是简单把位置向量加到 embedding 上，而是对 Q、K 做位置相关的旋转变换。

直觉上，RoPE 让同一个 token 在不同位置对应不同的几何方向，从而让 attention 计算天然感知相对距离。

### RoPE 为什么叫旋转

RoPE 通常把每两个维度看成一个二维平面上的向量。

对于第 `i` 个位置，在每个二维子空间上施加一个角度和位置相关的旋转：

```text
[x1, x2] -> [x1 * cos(angle) - x2 * sin(angle), x1 * sin(angle) + x2 * cos(angle)]
```

其中 `angle` 与位置 `i` 相关。

也就是说，不同位置对应不同旋转角。

这就是“Rotary”的来源。

### RoPE 怎么进入 attention

对每个位置 `i` 的 query 和 key，先做旋转：

```text
q_i' = R(i) q_i
k_i' = R(i) k_i
```

然后再计算 attention logits：

```text
q_i'^T k_j'
```

因为 query 和 key 的旋转角都和位置有关，二者相乘时会自然出现相对位置差异。

这意味着 attention 分数不只是内容相似度，还带有相对位置信息。

### 为什么 RoPE 是相对位置机制

RoPE 的一个重要性质是：

```text
R(i)^T R(j) = R(j - i)
```

这意味着位置 `i` 和 `j` 之间的交互可以写成只与相对距离有关的形式。

这就是 RoPE 最关键的数学直觉之一：

```text
attention score 自然携带相对位置信息
```

因此它比简单绝对位置向量加法更适合建模相对顺序关系。

### RoPE 对长上下文有什么帮助

RoPE 的一个重要优点是相对位置结构更自然。

如果模型在训练时学会了基于相对距离的模式，那么在一定程度上可以对更长上下文进行外推。

更严谨地说，RoPE 原论文强调的是：用旋转矩阵编码绝对位置，同时让 self-attention 公式中出现显式的相对位置依赖。它给长度泛化提供了更好的结构先验，但不是保证任意长度都能稳定工作的充分条件。

这也是为什么很多长上下文扩展工作会围绕 RoPE 做文章，例如：

1. RoPE scaling。
2. NTK scaling。
3. YaRN 类调整。
4. 长上下文继续预训练。

但要注意：

```text
RoPE 有利于长度外推，不等于自动解决所有长上下文问题
```

### RoPE 和频率尺度

RoPE 通常在不同维度使用不同频率的旋转。

低维度对应高频或低频，形成多尺度位置表示。

直觉上可以把它理解为：

1. 某些维度捕捉短距离位置变化。
2. 某些维度捕捉长距离位置变化。
3. 多频率结合后，模型能在不同尺度上感知相对位置。

这种多尺度编码也是 RoPE 受欢迎的原因之一。

### RoPE 和绝对位置编码的差异

可以粗略比较：

```text
绝对位置编码：告诉模型你在第几个位置
RoPE：告诉模型当前位置和其他位置之间的相对关系
```

绝对编码更像给 token 打坐标标签。

RoPE 更像让 attention 在几何空间中自然感知距离。

对语言建模而言，相对关系往往更重要，所以 RoPE 更适合现代 LLM。

### RoPE 的局限性

RoPE 很强，但也有局限。

常见局限包括：

1. 直接外推到远超训练长度时可能退化。
2. 高频旋转在极长长度下可能失真。
3. 训练和推理长度不一致时，效果可能下降。
4. 需要和长上下文训练策略配套。

所以后续很多方法都在改造 RoPE，使它更适合超长上下文。

### 长上下文外推问题

模型训练时如果只见过 4K context，推理时突然用 32K 甚至 128K context，可能出现：

1. attention 分数分布漂移。
2. 远距离 token 的位置感知失真。
3. 中间信息遗忘。
4. 生成质量下降。

RoPE scaling 的目标就是缓解这类问题。

Position Interpolation 这类方法的核心边界是：不要让推理位置无约束地跑到训练位置范围之外，而是把更长序列的位置索引压缩或映射回模型更熟悉的范围。YaRN 等方法进一步围绕 RoPE 做更细的频率/温度/插值设计，并通常仍需要评估和一定训练预算。面试时不要把这些方法说成“把 max length 参数改大就行”。

### RoPE scaling 的直觉

当推理长度超过训练长度时，可以通过重新缩放位置频率，让模型感知到的“位置跨度”变慢一些。

直觉上就是：

```text
让更长的实际序列，在模型看来不要旋转得太快
```

这样可以减少超长位置上的相位失真。

不同缩放方法的具体实现细节各不相同，但核心目标是一致的：

```text
改善 RoPE 在训练长度外的相对位置泛化
```

### RoPE 和长上下文训练

仅靠改位置编码通常不够。

要真正让模型支持长上下文，还需要：

1. 训练数据中有长序列样本。
2. 模型在训练中见过长距离依赖任务。
3. 注意力实现支持更长 KV Cache。
4. 评估覆盖跨段推理和长文检索。

RoPE 是长上下文能力的重要一环，但不是唯一一环。

### 一个最小 RoPE 直觉例子

可以把二维向量旋转想成复平面上的相位变化。

如果把两个维度看成一个复数：

```text
z = x_1 + i x_2
```

那么位置相关旋转相当于：

```text
z_new = z * exp(i * angle)
```

不同位置对应不同相位。

当 query 和 key 相乘时，相位差就反映了相对位置差。

这就是 RoPE 的几何直觉。

### 一个最小 PyTorch 直觉实现

下面是简化的二维旋转示意，不是完整高效实现，只帮助理解。

```python
import torch


def rotate_2d(x, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x1, x2 = x[..., 0], x[..., 1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1)
```

如果把位置映射成角度，不同位置就会得到不同旋转结果。

真实 RoPE 会在多组维度上使用不同频率的旋转。

下面这个 demo 验证 RoPE 的核心性质：把 `q` 旋转到位置 `i`，把 `k` 旋转到位置 `j` 后的点积，只和相对距离 `j - i` 对应的旋转有关。

```python
import torch


def rotate_2d(x, angle):
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    x1, x2 = x[..., 0], x[..., 1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1)


torch.manual_seed(0)
q = torch.randn(2)
k = torch.randn(2)
freq = torch.tensor(0.25)

for i, j in [(1, 3), (4, 6), (2, 7)]:
    qi = rotate_2d(q, torch.tensor(float(i)) * freq)
    kj = rotate_2d(k, torch.tensor(float(j)) * freq)
    score_by_absolute_positions = torch.dot(qi, kj)

    relative_angle = torch.tensor(float(j - i)) * freq
    k_relative = rotate_2d(k, relative_angle)
    score_by_relative_distance = torch.dot(q, k_relative)

    print(i, j, round(score_by_absolute_positions.item(), 6), round(score_by_relative_distance.item(), 6))
    assert torch.allclose(score_by_absolute_positions, score_by_relative_distance, atol=1e-6)
```

输出中每一行后两个数相同，说明二维旋转下 `R(i) q` 和 `R(j) k` 的点积可以等价写成 `q` 与按相对距离旋转后的 `k` 的点积。这就是 RoPE 能把相对位置信息注入 attention score 的最小数学直觉。

### 面试中怎么讲

如果面试官问“RoPE 的核心思想是什么”，可以这样回答：

```text
RoPE 不是简单把位置向量加到 embedding 上，而是对 attention 里的 Q 和 K 做位置相关旋转。这样 query 和 key 的点积会自然包含相对位置信息，因为不同位置的旋转相位差会进入 attention score。它比绝对位置编码更自然地表达相对距离，所以非常适合现代 decoder-only LLM。
```

如果面试官问“为什么 RoPE 对长上下文有帮助”，可以这样回答：

```text
RoPE 用多频率旋转把相对位置编码进 Q/K 的几何结构里，模型更容易学习距离模式，因此在一定程度上可以向训练长度外外推。但真正支持超长上下文还需要配合长序列训练、位置缩放、KV Cache 优化和评估策略，RoPE 只是其中一个关键组件。
```

### 真实项目中的坑

#### 坑 1：以为 RoPE 自动解决长上下文

RoPE 有助于位置外推，但模型是否真的会用长上下文，还取决于训练数据、注意力实现和评估任务。

#### 坑 2：随意改 RoPE scaling

位置缩放会改变 attention 分数分布，可能影响短上下文能力和生成稳定性，需要做系统评估。

#### 坑 3：只看最大 context length

最大长度变大不代表长文检索、跨段推理和中间信息利用都变强。

#### 坑 4：忽略训练长度和推理长度差异

训练从未见过长序列，推理直接拉长到很大 context，往往会出现位置分布漂移。

#### 坑 5：把 RoPE 和 ALiBi 混为一谈

RoPE 是旋转 Q/K，ALiBi 是给 attention logits 加距离相关 bias，两者机制不同。

### 面试官会怎么问

#### 问题 1：为什么 Transformer 需要位置编码？

回答框架：

1. Self-Attention 本身主要根据内容做匹配。
2. 纯 attention 不天然知道 token 顺序。
3. 语言、代码和对话都依赖顺序。
4. 位置编码把顺序信息注入模型。

#### 问题 2：RoPE 和绝对位置编码有什么区别？

回答框架：

1. 绝对位置编码通常把位置向量加到 embedding 上。
2. RoPE 对 Q/K 做位置相关旋转。
3. RoPE 的 attention score 能自然体现相对位置差。
4. 因此 RoPE 更适合相对距离建模和长度外推。

#### 问题 3：RoPE 为什么能表示相对位置？

回答框架：

1. 位置 `i` 和 `j` 分别对应旋转矩阵 `R(i)` 和 `R(j)`。
2. Q/K 点积中会出现 `R(i)^T R(j)`。
3. 这个项等价于和 `j-i` 相关的旋转。
4. 因此 attention 分数携带相对位置信息。

#### 问题 4：RoPE scaling 解决什么问题？

回答框架：

1. 推理长度超过训练长度时，旋转相位可能超出模型熟悉范围。
2. scaling 调整位置频率或位置映射。
3. 让长序列在模型内部的相位变化更平滑。
4. 目标是改善训练长度外的泛化。

#### 问题 5：支持长上下文除了 RoPE 还要做什么？

回答框架：

1. 长序列继续训练或微调。
2. 构造长程依赖数据。
3. 优化 attention 和 KV Cache 系统。
4. 做 lost-in-the-middle、needle 和真实长文评估。
5. 必要时结合 RAG、压缩和分块策略。

### 常见误区

1. 误区：RoPE 是一种绝对位置 embedding。
   纠正：RoPE 通过旋转 Q/K，让 attention 分数自然包含相对位置关系。

2. 误区：RoPE scaling 只会提升长上下文，不会有副作用。
   纠正：缩放可能影响短上下文分布和 attention 行为，需要评估。

3. 误区：context window 变大就说明长上下文能力强。
   纠正：还要看模型是否能检索、引用、推理和利用远处信息。

4. 误区：RoPE 可以替代 RAG。
   纠正：RoPE 处理位置表示，RAG 处理外部知识访问，两者不是同一类问题。

5. 误区：位置编码只是输入层的小模块。
   纠正：位置编码会影响每层 attention 的路由模式和长度泛化。

### 小练习

1. 用自己的话解释为什么纯 attention 需要位置编码。
2. 写出二维旋转矩阵，并解释它如何作用在 Q/K 上。
3. 解释为什么 `R(i)^T R(j)` 和相对位置有关。
4. 比较绝对位置编码、RoPE 和 ALiBi 的直觉差异。
5. 设计一个评估 RoPE 长度外推能力的实验。
6. 用 2 分钟回答“RoPE 为什么适合现代 LLM”。

### 本讲总结

本讲最重要的结论：

1. Self-Attention 需要位置编码来理解 token 顺序。
2. RoPE 通过对 Q/K 做位置相关旋转，把位置信息注入 attention score。
3. RoPE 的核心数学直觉是旋转相位差能表达相对位置。
4. RoPE 比简单绝对位置编码更自然地建模相对距离。
5. RoPE scaling 试图改善训练长度外的位置泛化，但可能有副作用。
6. 长上下文能力还需要长序列数据、训练策略、推理系统和评估一起配合。

## 第 17 讲：ALiBi、相对位置编码与长度泛化

### 本讲目标

上一讲讲了 RoPE。本讲继续讨论位置建模，重点是 ALiBi 和相对位置编码。ALiBi 的思路和 RoPE 很不一样：它不是旋转 Q/K，而是在 attention logits 上直接加入和距离相关的线性偏置。这个设计简单、外推友好，也很适合用来理解“长度泛化”到底依赖什么。

你需要掌握：

1. 相对位置编码为什么重要。
2. ALiBi 的核心公式和直觉。
3. ALiBi 和 RoPE 的差异。
4. 为什么 ALiBi 有较好的长度外推性质。
5. 长度泛化和长上下文能力的区别。
6. 面试中如何比较不同位置编码方案。

### 相对位置为什么重要

很多语言关系不只依赖绝对位置，而是依赖相对距离。

例如：

1. 当前 token 和前一个 token 的关系。
2. 当前变量和最近定义位置的关系。
3. 引用和被引用文本之间的距离。
4. 长文中标题、段落和证据之间的距离。

绝对位置编码告诉模型“我在第几个位置”。

相对位置编码更关注：

```text
我和你相隔多远
```

对 attention 来说，这很自然，因为 attention 本身是在计算 token 对之间的交互。

### ALiBi 的核心思想

ALiBi 是 Attention with Linear Biases。

它的核心做法是在 attention logits 上加入一个和距离相关的线性 bias。

标准 attention logits 是：

```text
score[i, j] = dot(q_i, k_j) / sqrt(d)
```

ALiBi 改成：

```text
score[i, j] = dot(q_i, k_j) / sqrt(d) + bias(i, j)
```

对 causal LM，通常 `j <= i`。

距离可以写成：

```text
distance = i - j
```

ALiBi 使用类似下面的形式：

```text
bias(i, j) = -m_h * (i - j)
```

其中 `m_h` 是第 `h` 个 attention head 的斜率。

距离越远，bias 越负，attention 越倾向于近处 token。

更贴近原论文的表述是：ALiBi 不把位置 embedding 加到 token embedding 上，而是直接给 query-key attention score 加一个和距离成比例的惩罚项。这个设计带有 recency inductive bias，也就是默认鼓励模型更容易关注近处 token，但不是硬性禁止远处 token。

### ALiBi 的直觉

ALiBi 给模型加入一个简单先验：

```text
越近的 token 默认越重要，越远的 token 默认越需要更强内容匹配才能被关注
```

这不是禁止关注远处 token。

远处 token 仍然可以被关注，只是需要内容相似度足够高，抵消距离惩罚。

所以 ALiBi 是一种软位置偏置。

### 为什么不同 head 使用不同斜率

ALiBi 通常给不同 attention head 分配不同斜率。

有些 head 的距离惩罚强，更偏向局部。

有些 head 的距离惩罚弱，可以关注更远位置。

这形成多尺度位置偏好：

1. 局部 head 捕捉短距离依赖。
2. 长程 head 保留远距离访问能力。
3. 多个 head 合起来覆盖不同距离范围。

这和多头 attention 的子空间分工是一致的。

### ALiBi 为什么适合长度外推

ALiBi 不需要为每个位置学习单独 embedding。

它的 bias 是距离的简单函数：

```text
-m * distance
```

这个函数天然可以应用到比训练更长的位置。

例如训练时最长看到 4K，推理时看到 8K，距离变大后 bias 仍然可以计算。

这就是 ALiBi 长度外推友好的原因之一。

相比之下，学习式绝对位置 embedding 对未见位置没有天然定义。

### ALiBi 和 RoPE 的差异

RoPE 的做法是：

```text
对 Q/K 做位置相关旋转
```

ALiBi 的做法是：

```text
对 attention logits 加距离相关 bias
```

两者都能让 attention 感知相对位置，但方式不同。

RoPE 更像把位置编码进 Q/K 的几何结构。

ALiBi 更像在 attention 分数上加入一个距离先验。

可以粗略比较：

```text
RoPE: 位置影响 Q/K 表示和点积几何
ALiBi: 位置直接影响 attention logits
```

### ALiBi 的优点

ALiBi 的优点包括：

1. 简单，不需要位置 embedding 表。
2. 对更长序列有天然定义。
3. 加在 logits 上，实现概念清楚。
4. 不增加 token embedding 维度。
5. 不同 head 可以形成多尺度距离偏好。

这些优点让它在长度外推研究中很有代表性。

### ALiBi 的局限

ALiBi 也不是万能的。

局限包括：

1. 它引入了单调距离惩罚，偏向近处 token。
2. 对某些需要远距离精确检索的任务，距离惩罚可能不理想。
3. 它没有 RoPE 那种旋转几何结构。
4. 长度外推好不代表长上下文推理一定强。
5. 具体效果依赖模型、数据和训练方式。

所以 ALiBi 是一个优秀的位置偏置方案，但不是所有现代 LLM 的唯一选择。

### 长度泛化和长上下文能力不同

这是面试中非常容易混淆的点。

长度泛化指模型在比训练更长的位置上仍然保持合理位置行为。

长上下文能力则包括：

1. 能否读完整上下文。
2. 能否找到远处证据。
3. 能否整合多段信息。
4. 能否避免 lost in the middle。
5. 能否在长文中保持推理一致性。

位置编码只解决其中一部分。

真正的长上下文能力还依赖训练数据、attention 实现、KV Cache、检索策略和评估。

### 相对位置编码的其他形式

除了 RoPE 和 ALiBi，还有很多相对位置编码方式。

常见思路包括：

1. 在 attention logits 上加相对位置 bias。
2. 为不同相对距离学习 embedding。
3. 使用 bucket，把距离分桶后学习偏置。
4. 在 Q/K/V 中加入相对位置信息。

T5 风格的 relative position bias 就是一个重要例子。

它把相对距离分桶，然后给每个 bucket 一个可学习 bias。

和 ALiBi 相比，T5 风格 relative position bias 是学习式 bucket bias；ALiBi 是手工设定斜率的线性距离 bias。二者都作用在 attention logits 上，但外推行为和参数化不同。

### Bucket 相对位置偏置

直接为每个距离学习一个 bias 会有问题：距离太多，参数和外推都麻烦。

bucket 方法会把距离分组。

例如：

1. 很近的距离分得细。
2. 很远的距离分得粗。
3. 超过某个范围的距离共用 bucket。

这样可以兼顾近距离精度和远距离泛化。

但它仍然依赖预设 bucket 设计，外推能力取决于分桶策略。

### 如何比较位置编码方案

比较位置编码不能只看论文分数。

要从几个维度看：

1. 是否支持长度外推。
2. 是否增加参数。
3. 是否影响 attention kernel。
4. 是否适合自回归推理。
5. 是否和 FlashAttention 等优化兼容。
6. 是否在目标任务上稳定。
7. 是否影响短上下文能力。

RoPE、ALiBi、relative bias 各有取舍。

### 一个最小 ALiBi bias 示例

下面是简化版 causal ALiBi bias 构造。

```python
import torch


def build_alibi_bias(seq_len, slope, device):
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = i - j
    bias = -slope * distance.clamp(min=0)
    mask = j > i
    bias = bias.masked_fill(mask, float("-inf"))
    return bias
```

这个示例展示了 ALiBi 的核心：

1. 对未来位置做 causal mask。
2. 对历史位置按距离加入负 bias。
3. 距离越远，默认分数越低。

真实实现会为不同 head 使用不同 slope，并注意数值和 kernel 兼容。

下面这个 demo 构造多个 head 的 ALiBi bias，并验证三件事：未来位置是 `-inf`，历史距离越远 bias 越负，不同 head 的 slope 会产生不同距离惩罚强度。

```python
import torch


def build_alibi_bias(seq_len, slopes, device="cpu"):
    positions = torch.arange(seq_len, device=device)
    query_pos = positions.view(1, seq_len, 1)
    key_pos = positions.view(1, 1, seq_len)
    distance = query_pos - key_pos

    slopes = torch.tensor(slopes, device=device).view(-1, 1, 1)
    bias = -slopes * distance.clamp(min=0)
    causal_mask = key_pos > query_pos
    return bias.masked_fill(causal_mask, float("-inf"))


seq_len = 5
slopes = [0.25, 1.0]
bias = build_alibi_bias(seq_len, slopes)

assert list(bias.shape) == [2, seq_len, seq_len]
assert torch.isinf(bias[0, 0, 1])
assert bias[0, 4, 0] < bias[0, 4, 3]
assert bias[1, 4, 0] < bias[0, 4, 0]

print("head 0 bias for last query:", bias[0, -1].tolist())
print("head 1 bias for last query:", bias[1, -1].tolist())
print("future bias at query 0 key 1:", bias[0, 0, 1].item())
```

典型输出：

```text
head 0 bias for last query: [-1.0, -0.75, -0.5, -0.25, -0.0]
head 1 bias for last query: [-4.0, -3.0, -2.0, -1.0, -0.0]
future bias at query 0 key 1: -inf
```

这说明 slope 越大，远距离 token 的默认惩罚越强；但这只是加在 logits 上的 soft bias，不是把远处 token mask 掉。

### 面试中怎么讲

如果面试官问“ALiBi 的核心思想是什么”，可以这样回答：

```text
ALiBi 不学习位置 embedding，也不旋转 Q/K，而是在 attention logits 上加入和相对距离成线性关系的 bias。对于 causal LM，距离越远的历史 token 会得到越大的负偏置，这给模型一个近处更重要的先验，同时仍允许远处 token 通过足够高的内容匹配获得关注。因为这个 bias 是距离函数，所以能自然扩展到比训练更长的位置。
```

如果面试官问“ALiBi 和 RoPE 怎么比较”，可以这样回答：

```text
RoPE 是对 Q/K 做位置相关旋转，让点积天然包含相对位置相位差；ALiBi 是直接在 attention logits 上加距离相关线性偏置。RoPE 更像几何式相对位置编码，ALiBi 更像显式距离先验。两者都支持相对位置建模，但机制、外推行为和工程适配不同，需要结合模型和任务验证。
```

### 真实项目中的坑

#### 坑 1：把长度外推当成长上下文能力

位置编码能外推，不代表模型能可靠检索和推理超长上下文。

#### 坑 2：忽略短上下文回归

改位置编码或 bias 后，短上下文任务也可能受影响，需要同时评估。

#### 坑 3：ALiBi 斜率设置随便改

不同 head 的 slope 控制距离偏好，改动会影响 attention 行为。

#### 坑 4：只看最大长度

需要看不同长度区间的表现，而不是只看能不能跑到某个最大 context。

#### 坑 5：忽略 kernel 兼容性

相对 bias 是否能和 FlashAttention、PagedAttention、推理引擎兼容，是工程落地问题。

### 面试官会怎么问

#### 问题 1：ALiBi 为什么能外推到更长位置？

回答框架：

1. 它不依赖学习的位置 embedding 表。
2. bias 是相对距离的线性函数。
3. 更长距离仍然可以计算 bias。
4. 因此比固定长度绝对位置 embedding 更自然外推。

#### 问题 2：ALiBi 是否会阻止模型关注远处 token？

回答框架：

1. 不会硬性阻止。
2. 它只是给远处 token 更大的负偏置。
3. 如果内容匹配足够强，远处 token 仍可被关注。
4. 这是软距离先验。

#### 问题 3：相对位置 bias 和 RoPE 的主要区别是什么？

回答框架：

1. 相对 bias 直接加到 attention logits。
2. RoPE 改变 Q/K 表示，使点积包含相对位置信息。
3. bias 方法更显式，RoPE 更几何化。
4. 两者都需要结合长度外推和任务效果评估。

#### 问题 4：什么是长度泛化？

回答框架：

1. 模型在比训练更长的位置上保持合理行为。
2. 位置编码是否有外推定义很关键。
3. 但长度泛化不等于长上下文理解。
4. 长上下文还要看数据、训练和任务能力。

#### 问题 5：如何评估位置编码的长上下文效果？

回答框架：

1. 在不同长度区间测试 perplexity 或 loss。
2. 做 needle-in-a-haystack 检索测试。
3. 做跨段推理和多证据问答。
4. 检查 lost-in-the-middle。
5. 同时评估短上下文任务是否退化。

### 常见误区

1. 误区：ALiBi 是另一种绝对位置 embedding。
   纠正：ALiBi 是加在 attention logits 上的相对距离 bias。

2. 误区：ALiBi 让模型只能看近处。
   纠正：它是软偏置，不是 hard mask。

3. 误区：长度外推好就说明长上下文能力强。
   纠正：还要看训练数据、检索能力、推理能力和系统实现。

4. 误区：RoPE 和 ALiBi 只是公式不同，效果一定差不多。
   纠正：它们注入位置的机制不同，实际行为和外推特性也不同。

5. 误区：相对位置编码一定优于绝对位置编码。
   纠正：要看任务、长度范围、模型结构和训练设置。

### 小练习

1. 写出 ALiBi bias 的简化公式。
2. 解释为什么不同 head 可以使用不同 slope。
3. 比较 ALiBi、RoPE、绝对位置 embedding 的差异。
4. 设计一个测试长度泛化的实验。
5. 解释为什么长度泛化不等于长上下文推理能力。
6. 用 2 分钟回答“ALiBi 为什么外推友好”。

### 本讲总结

本讲最重要的结论：

1. ALiBi 通过给 attention logits 加相对距离线性 bias 注入位置信息。
2. 距离越远，默认负偏置越大，但远处 token 仍可通过内容匹配获得关注。
3. ALiBi 不依赖固定位置 embedding 表，因此天然具备较好的长度外推形式。
4. RoPE 和 ALiBi 都建模相对位置，但一个旋转 Q/K，一个修改 attention logits。
5. 长度泛化不等于长上下文能力，后者还需要训练数据、推理系统和真实任务评估。
6. 位置编码方案要从外推能力、短上下文回归、kernel 兼容和任务效果多维比较。

## 第 18 讲：MQA、GQA 与推理效率优化

### 本讲目标

大模型推理时，attention 的瓶颈不只在计算，也在 KV Cache 的显存和带宽。MQA 和 GQA 是现代 LLM 中非常重要的推理优化设计：它们通过减少 K/V head 数量，显著降低 KV Cache 成本，同时尽量保留多头 query 的表达能力。

你需要掌握：

1. MHA、MQA、GQA 的区别。
2. KV Cache 为什么是推理瓶颈。
3. MQA 如何减少 K/V 缓存。
4. GQA 为什么是 MHA 和 MQA 的折中。
5. MQA/GQA 对质量、速度和显存的 trade-off。
6. 面试中如何从推理系统角度解释 GQA。

### 先回顾 MHA

标准 Multi-Head Attention 中，每个 head 都有自己的 Q、K、V。

如果有 `H` 个 attention heads，那么：

```text
Q heads: H
K heads: H
V heads: H
```

每个 head 独立计算 attention，然后拼接输出。

这种设计表达能力强，因为不同 head 可以学习不同子空间的信息路由。

但推理时，每一层都需要为每个 head 缓存 K/V。

### KV Cache 为什么贵

自回归生成时，每生成一个新 token，都需要和历史 token 做 attention。

为了避免重复计算历史 K/V，推理系统会缓存每层的 K 和 V。

KV Cache 大小大致和下面因素成正比：

```text
batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2
```

最后的 `2` 表示 K 和 V。

当上下文变长、batch 变大、层数变多时，KV Cache 会非常大。

它不仅占显存，还会带来显存带宽压力。

很多推理场景中，瓶颈不是算不动矩阵乘法，而是读写 KV Cache 太慢。

MQA 原论文关注的正是增量解码里的内存带宽问题：每一步生成都要反复读取历史 keys 和 values，K/V 张量越大，带宽压力越高。减少 KV heads 不只是省显存，也是在减少每步解码读取 KV Cache 的数据量。

### MQA 的核心思想

MQA 是 Multi-Query Attention。

它的核心是：

```text
多个 query heads 共享同一组 K/V heads
```

最极端的 MQA 中：

```text
Q heads: H
K heads: 1
V heads: 1
```

也就是说，query 仍然有多个 head，但所有 query head 共享同一套 K/V。

这样 KV Cache 可以从 `H` 份减少到 1 份。

如果 `H=32`，理论上 K/V cache 的 head 维度部分可以减少约 32 倍。

注意这不是把 query head 减少到 1。MQA 仍然保留多个 query heads，只是这些 query heads 共享同一套 K/V 表示。

### MQA 的优点

MQA 的优点很直接：

1. KV Cache 显著变小。
2. 推理显存压力降低。
3. 读取 KV Cache 的带宽压力降低。
4. 长上下文和大 batch 推理更友好。
5. 解码吞吐可能提升。

这对在线服务非常重要。

当服务要同时处理很多用户请求时，KV Cache 往往决定能放下多少并发序列。

### MQA 的潜在问题

MQA 的代价是 K/V 表达能力下降。

因为所有 query heads 共享同一组 K/V，不同 head 在读取内容时少了一部分独立性。

可能问题包括：

1. 模型质量下降。
2. attention head 多样性变弱。
3. 对复杂任务或长上下文能力有影响。
4. 训练时需要专门适配，而不是随便把 MHA 改成 MQA。

所以 MQA 是强推理优化，但质量 trade-off 需要验证。

### GQA 的核心思想

GQA 是 Grouped-Query Attention。

它是 MHA 和 MQA 之间的折中。

GQA 把 query heads 分成若干组，每组共享一套 K/V。

GQA 论文把它看成 MQA 的泛化：KV heads 的数量介于 1 和 query heads 数量之间。论文还讨论了从 MHA checkpoint uptrain 成 MQA/GQA 的路线，重点不是“无训练直接改配置”，而是用少量继续训练恢复质量。

例如：

```text
Q heads: 32
KV heads: 8
```

这意味着每 4 个 query heads 共享 1 组 K/V。

相比 MHA，KV Cache 减少 4 倍。

相比 MQA，K/V 表达能力更强。

### MHA、MQA、GQA 对比

可以这样比较：

```text
MHA: Q heads = H, KV heads = H
MQA: Q heads = H, KV heads = 1
GQA: Q heads = H, KV heads = G, 1 < G < H
```

其中 `G` 是 KV head 数。

从质量到效率：

```text
MHA: 表达强，KV Cache 大
MQA: KV Cache 小，表达压缩强
GQA: 折中方案
```

现代 LLM 很多采用 GQA，因为它在质量和推理效率之间更平衡。

### 为什么 GQA 成为主流折中

GQA 的吸引力在于它不是极端压缩。

它保留多个 K/V heads，让不同 query group 仍然有不同的信息读取空间。

同时又把 KV Cache 降到原来的 `G/H`。

例如 `H=32, G=8`：

```text
KV Cache = MHA 的 1/4
```

这对长上下文推理非常有价值。

相比 MQA，GQA 更容易保持模型质量。

### GQA 的 shape 直觉

假设：

```text
num_query_heads = 32
num_kv_heads = 8
head_dim = 128
```

那么 Q 的 shape 可以理解为：

```text
[batch, seq_len, 32, 128]
```

K/V 的 shape 是：

```text
[batch, seq_len, 8, 128]
```

计算 attention 时，每 4 个 query heads 使用同一个 KV head。

实现中通常会把 K/V 通过 repeat 或更高效的 kernel 映射到 query heads。

### GQA 和训练

GQA 最好从预训练阶段就作为模型架构的一部分。

如果训练好一个 MHA 模型后强行改成 GQA，需要做权重转换或继续训练，质量可能受影响。

常见做法包括：

1. 从头用 GQA 训练。
2. 从 MHA checkpoint 转换后继续训练。
3. 对 K/V heads 做平均或分组初始化。

但这些都需要实验验证。

### GQA 和长上下文

长上下文推理中，KV Cache 是关键瓶颈。

GQA 降低 `num_kv_heads`，所以对长上下文特别有用。

如果上下文长度增加 4 倍，KV Cache 也近似增加 4 倍。

如果同时把 KV heads 从 32 降到 8，就可以抵消一部分增长。

所以很多长上下文 LLM 会同时使用：

1. RoPE 或位置扩展。
2. FlashAttention 或高效 attention kernel。
3. GQA/MQA 降低 KV Cache。
4. PagedAttention 管理 KV Cache。

### MQA/GQA 和 PagedAttention 的关系

MQA/GQA 减少的是每个 token、每层需要缓存的 K/V head 数。

PagedAttention 解决的是 KV Cache 如何分页、复用、调度和减少碎片。

两者不冲突。

可以理解为：

```text
GQA/MQA: 减少 KV Cache 体积
PagedAttention: 更高效地管理 KV Cache
```

在线推理系统通常需要组合使用多种优化。

### 一个简单估算

假设：

```text
layers = 32
seq_len = 8192
head_dim = 128
query_heads = 32
kv_heads = 32 or 8
dtype = BF16
```

KV Cache 元素数大致是：

```text
layers * seq_len * kv_heads * head_dim * 2
```

如果 `kv_heads` 从 32 变成 8，KV Cache 直接降为 1/4。

这就是 GQA 对推理服务的实际价值。

下面这个 demo 同时估算 KV Cache 大小，并展示 GQA 中 query head 到 KV head 的分组映射。

```python
def kv_cache_mib(batch, seq_len, layers, kv_heads, head_dim, bytes_per_value):
    elements = batch * seq_len * layers * kv_heads * head_dim * 2
    return elements * bytes_per_value / 1024 / 1024


def query_to_kv_mapping(query_heads, kv_heads):
    assert query_heads % kv_heads == 0
    group_size = query_heads // kv_heads
    return [q // group_size for q in range(query_heads)]


batch = 1
seq_len = 8192
layers = 32
query_heads = 32
head_dim = 128
bytes_per_value = 2  # BF16/FP16

for kv_heads in [32, 8, 1]:
    cache_mib = kv_cache_mib(batch, seq_len, layers, kv_heads, head_dim, bytes_per_value)
    print(f"kv_heads={kv_heads:2d}, KV cache MiB={cache_mib:.1f}")

print("GQA mapping for 32 query heads and 8 KV heads:")
print(query_to_kv_mapping(query_heads=32, kv_heads=8))
```

典型输出：

```text
kv_heads=32, KV cache MiB=4096.0
kv_heads= 8, KV cache MiB=1024.0
kv_heads= 1, KV cache MiB=128.0
GQA mapping for 32 query heads and 8 KV heads:
[0, 0, 0, 0, 1, 1, 1, 1, ..., 7, 7, 7, 7]
```

这个估算只算 KV Cache，不算模型权重、激活、allocator 碎片和框架开销。真实服务中还要看 batching、paged KV 管理、attention kernel 和并行切分。

### 面试中怎么讲

如果面试官问“MQA 和 GQA 是什么”，可以这样回答：

```text
标准 MHA 中每个 query head 都有自己的 K/V head，推理时 KV Cache 很大。MQA 保留多个 query heads，但所有 query heads 共享一组 K/V，从而显著降低 KV Cache。GQA 是折中方案，把 query heads 分组，每组共享一个 K/V head，比如 32 个 query heads 配 8 个 KV heads。这样比 MHA 更省显存和带宽，又比 MQA 保留更多 K/V 表达能力。
```

如果面试官问“为什么 GQA 对推理效率重要”，可以这样回答：

```text
自回归推理时，每层都要保存历史 token 的 K/V。KV Cache 大小和 num_kv_heads 成正比，长上下文和高并发下会成为显存和带宽瓶颈。GQA 通过减少 KV heads，把 KV Cache 降到 MHA 的一部分，因此可以提升吞吐、支持更长上下文或更大 batch，同时比 MQA 更好保留质量。
```

### 真实项目中的坑

#### 坑 1：只看参数量，不看 KV Cache

推理服务里，KV Cache 往往比权重参数更直接限制并发和上下文长度。

#### 坑 2：把 MQA 当成无损优化

MQA 压缩 K/V 表达，可能影响质量，需要训练和评估验证。

#### 坑 3：训练后随意改 KV head 数

MHA、GQA、MQA 是架构设计，不能在不适配的情况下随便替换。

#### 坑 4：只优化单请求 latency

实际服务还要看吞吐、batching、显存占用、KV Cache 碎片和排队延迟。

#### 坑 5：忽略 kernel 支持

GQA/MQA 的实际加速依赖推理引擎和 attention kernel 是否高效支持。

### 面试官会怎么问

#### 问题 1：MHA、MQA、GQA 的区别是什么？

回答框架：

1. MHA 每个 query head 都有独立 K/V。
2. MQA 所有 query heads 共享一组 K/V。
3. GQA 把 query heads 分组，每组共享 K/V。
4. GQA 是质量和推理效率之间的折中。

#### 问题 2：为什么 KV Cache 是推理瓶颈？

回答框架：

1. 自回归生成需要缓存每层历史 K/V。
2. KV Cache 随 batch、seq_len、layers、kv_heads、head_dim 增长。
3. 长上下文和高并发下显存压力大。
4. 每步解码还要读取历史 K/V，带宽压力也大。

#### 问题 3：GQA 会不会影响模型质量？

回答框架：

1. 可能会，因为多个 query heads 共享 K/V。
2. KV 表达自由度比 MHA 少。
3. 但比 MQA 保留更多 K/V heads。
4. 实际影响取决于模型规模、数据和训练方式。

#### 问题 4：为什么很多现代 LLM 用 GQA 而不是 MQA？

回答框架：

1. MQA 最省 KV Cache，但压缩最强。
2. GQA 在减少 KV Cache 的同时保留多个 K/V heads。
3. 质量和效率更平衡。
4. 对长上下文推理和服务部署更友好。

#### 问题 5：如何估算 KV Cache 大小？

回答框架：

1. 大致和 `batch * seq_len * layers * kv_heads * head_dim * 2` 成正比。
2. `2` 表示 K 和 V。
3. 再乘 dtype 字节数。
4. GQA/MQA 通过减少 `kv_heads` 降低缓存。

### 常见误区

1. 误区：GQA 只是少几个 attention head。
   纠正：GQA 减少的是 K/V heads，query heads 仍然可以很多。

2. 误区：MQA/GQA 只影响显存，不影响速度。
   纠正：减少 KV Cache 也能降低带宽压力，影响吞吐和延迟。

3. 误区：KV Cache 只在长上下文下重要。
   纠正：高并发、多 batch 服务中，即使中等上下文，KV Cache 也可能是瓶颈。

4. 误区：GQA 一定无损。
   纠正：它是折中，仍需质量评估。

5. 误区：PagedAttention 和 GQA 是同一个东西。
   纠正：GQA 减少 KV 体积，PagedAttention 管理 KV 分页和调度。

### 小练习

1. 写出 MHA、MQA、GQA 中 Q heads 和 KV heads 的关系。
2. 估算 `layers=32, seq_len=8192, kv_heads=32, head_dim=128, BF16` 的 KV Cache 大小。
3. 比较 `kv_heads=32` 和 `kv_heads=8` 的缓存差异。
4. 解释为什么 GQA 比 MQA 更可能保持质量。
5. 设计一个实验评估从 MHA 改成 GQA 后的质量变化。
6. 用 2 分钟回答“为什么 GQA 是现代 LLM 推理优化关键组件”。

### 本讲总结

本讲最重要的结论：

1. 自回归推理中 KV Cache 是显存和带宽的重要瓶颈。
2. MHA 每个 query head 有独立 K/V，表达强但 KV Cache 大。
3. MQA 让所有 query heads 共享 K/V，极大降低缓存但可能影响质量。
4. GQA 让一组 query heads 共享 K/V，是 MHA 和 MQA 之间的折中。
5. GQA 对长上下文、高并发和在线服务非常重要。
6. 推理效率优化要同时看模型结构、KV Cache 管理、attention kernel、batching 和评估质量。

## 第 19 讲：MoE 架构基础

### 本讲目标

MoE 是 Mixture of Experts，中文常叫混合专家模型。它是现代大模型扩展参数量的重要路线：模型可以拥有很多专家参数，但每个 token 只激活其中少数专家，从而在增加总参数量的同时控制计算量。本讲先讲 MoE 的基本架构，下一讲再专门讲训练稳定性和负载均衡。

你需要掌握：

1. MoE 和 dense FFN 的关系。
2. expert、router、top-k routing 分别是什么。
3. MoE 为什么能增加参数量但不同比例增加 FLOPs。
4. sparse activation 的核心思想。
5. MoE 的优点、代价和适用场景。
6. 面试中如何解释 MoE 的系统 trade-off。

### 从 Dense FFN 到 MoE

标准 Transformer 中，每个 block 都有一个 FFN。

所有 token 都经过同一个 FFN：

```text
y = FFN(x)
```

MoE 的想法是把一个 FFN 换成多个 expert FFN。

对于每个 token，router 决定它应该送到哪些 expert。

简化形式：

```text
y = sum_over_selected_experts gate_e(x) * Expert_e(x)
```

其中只有少数 `gate_e(x)` 非零。

更准确地说，MoE 的核心是 conditional computation：模型总参数量可以很大，但每个 token 只激活少数 expert 路径，因此每 token 计算不会随 expert 总数同比例增长。稀疏门控 MoE、GShard 和 Switch Transformer 都围绕这个思想发展，只是 routing、并行和稳定化设计不同。

### Expert 是什么

在 Transformer MoE 中，expert 通常就是一个 FFN/MLP。

例如每个 expert 都可以是：

```text
Expert_e(x) = W_down^e activation(W_up^e x)
```

或者 SwiGLU MLP。

不同 expert 有不同参数。

如果有 64 个 experts，那么这一层的总 FFN 参数量可以接近 dense FFN 的 64 倍。

但每个 token 不会经过所有 experts。

### Router 是什么

Router 是决定 token 去哪个 expert 的模块。

通常 router 会对每个 token 输出一组 expert 分数：

```text
logits = W_router x
```

然后通过 softmax 得到每个 expert 的概率：

```text
p_e = softmax(logits)_e
```

再选择 top-k experts。

例如 top-1 routing：每个 token 只去一个 expert。

top-2 routing：每个 token 去两个 experts，再加权合并输出。

### Sparse Activation

MoE 的关键不是总参数量大，而是稀疏激活。

Dense 模型中，每个 token 都使用所有参数路径。

MoE 中，每个 token 只使用少数 expert。

所以：

```text
总参数量很大，但每 token 激活参数量较小
```

这让模型可以提升容量，同时控制每个 token 的计算成本。

### 为什么 MoE 能扩展模型容量

在 dense 模型中，如果把 FFN hidden size 增大 4 倍，每个 token 的计算也会明显增加。

MoE 则可以增加 expert 数量，让总参数量变大，但每个 token 只走 top-k experts。

例如：

```text
experts = 64
top_k = 2
```

每个 token 只激活 2 个 experts，而不是 64 个。

这就是 MoE 的核心扩展优势。

### MoE 的直觉

可以把 MoE 理解为一种条件计算。

不同 token 根据内容被路由到不同 expert。

例如模型可能学到：

1. 某些 expert 更处理代码。
2. 某些 expert 更处理数学。
3. 某些 expert 更处理多语言。
4. 某些 expert 更处理格式化文本。

但这只是直觉，不一定每个 expert 都有清晰语义分工。

实际 expert specialization 需要通过分析验证。

### Top-1 和 Top-2 Routing

Top-1 routing：

```text
每个 token 只送到得分最高的 expert
```

优点：计算更省，路由简单。

缺点：容量利用和训练稳定性可能更难。

Switch Transformer 的一个重要取舍就是采用更简单的 top-1 routing，降低通信和计算复杂度，但需要配合负载均衡、capacity 和 router 稳定化技巧，不能理解成“只选一个 expert 就自然稳定”。

Top-2 routing：

```text
每个 token 送到得分最高的两个 experts
```

优点：表达更强，路由更平滑。

缺点：计算和通信成本更高。

不同 MoE 模型会根据训练规模和系统约束选择不同路由策略。

### MoE 和通信成本

MoE 的难点不只是算法，还有分布式系统。

如果 experts 分布在不同 GPU 上，token 需要被发送到对应 expert 所在设备。

这会产生 all-to-all 通信。

GShard 这类系统工作的重点之一，就是让 MoE 的条件计算能通过自动分片和专家并行在大规模设备上运行。MoE 的瓶颈经常不是单个 expert 的 MLP 计算，而是 token dispatch、combine 和跨设备 all-to-all。

MoE 训练和推理常见瓶颈包括：

1. token dispatch。
2. expert parallelism。
3. all-to-all 通信。
4. load balancing。
5. expert capacity 管理。

所以 MoE 是模型架构和系统工程强耦合的设计。

### MoE 的优点

MoE 的主要优点：

1. 大幅增加总参数量。
2. 每个 token 只激活少数专家，控制计算量。
3. 可能提升多任务、多领域容量。
4. 在相近推理 FLOPs 下获得更大模型容量。
5. 可以通过 expert specialization 提升表达能力。

这也是很多前沿大模型采用 MoE 的原因。

### MoE 的代价

MoE 的代价也很明显：

1. 训练更复杂。
2. 路由可能不均衡。
3. all-to-all 通信成本高。
4. batch 内 token 分布会影响专家负载。
5. 推理部署和 serving 调度更难。
6. checkpoint、并行策略和容错更复杂。

所以 MoE 不是“免费变大”。

它把一部分计算压力转换成路由、通信和系统复杂度。

### MoE 和 Dense 模型如何取舍

Dense 模型优点：

1. 架构简单。
2. 训练稳定。
3. 推理部署成熟。
4. 每个 token 使用全部模型能力。

MoE 模型优点：

1. 总参数量更大。
2. 每 token 计算可控。
3. 容量扩展效率高。

如果算力和系统能力足够，MoE 能提供很强扩展路线。

如果团队系统能力有限，dense 模型更容易训练和部署。

### 一个最小 MoE 伪代码

下面是极简 top-1 MoE 伪代码，只展示逻辑。

```python
def moe_layer(x, router, experts):
    router_logits = router(x)
    expert_id = router_logits.argmax(dim=-1)

    outputs = empty_like(x)
    for e, expert in enumerate(experts):
        mask = expert_id == e
        if mask.any():
            outputs[mask] = expert(x[mask])

    return outputs
```

真实 MoE 实现会复杂很多，需要处理 top-k 权重、容量限制、all-to-all、token 排序和负载均衡 loss。

下面这个 demo 是一个可直接运行的 top-1 MoE。它验证三个基本事实：每个 token 只路由到一个 expert，输出 shape 和输入 shape 相同，不同 expert 的 token 负载可能不均衡。

```python
import torch
import torch.nn as nn


class TinyTop1MoE(nn.Module):
    def __init__(self, d_model, d_hidden, num_experts):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        flat_x = x.reshape(-1, x.shape[-1])
        router_logits = self.router(flat_x)
        expert_id = router_logits.argmax(dim=-1)

        flat_out = torch.empty_like(flat_x)
        token_counts = torch.bincount(expert_id, minlength=len(self.experts))

        for expert_idx, expert in enumerate(self.experts):
            mask = expert_id == expert_idx
            if mask.any():
                flat_out[mask] = expert(flat_x[mask])

        return flat_out.reshape_as(x), expert_id.reshape(x.shape[:-1]), token_counts


torch.manual_seed(0)
moe = TinyTop1MoE(d_model=6, d_hidden=12, num_experts=4)
x = torch.randn(2, 5, 6)
out, expert_id, token_counts = moe(x)

assert out.shape == x.shape
assert expert_id.shape == x.shape[:-1]
assert token_counts.sum().item() == x.shape[0] * x.shape[1]

print("output shape:", list(out.shape))
print("expert ids:")
print(expert_id)
print("token counts per expert:", token_counts.tolist())
```

这个 demo 故意不实现 capacity、token dropping、top-2 combine 和 all-to-all，因为这些属于下一讲的训练稳定性和系统实现重点。这里先把 MoE 的核心路径讲清楚：router 先选 expert，token 只经过被选中的 expert。

### 面试中怎么讲

如果面试官问“MoE 是什么”，可以这样回答：

```text
MoE 是混合专家模型，通常把 Transformer 中的 dense FFN 替换成多个 expert FFN。每个 token 通过 router 选择少数 experts，比如 top-1 或 top-2，然后只经过这些 experts。这样模型可以拥有很大的总参数量，但每个 token 的激活计算量只对应少数专家，因此实现稀疏激活和条件计算。
```

如果面试官问“MoE 为什么难训练和部署”，可以这样回答：

```text
MoE 的难点在于 router 会把 token 分配到不同 experts，如果分配不均，会出现某些 expert 过载、某些 expert 空闲的问题。分布式训练中 token 还要跨设备发送到对应 expert，带来 all-to-all 通信成本。推理时也要处理动态路由、expert 并行、batching 和负载均衡，所以 MoE 是架构和系统强耦合的设计。
```

### 真实项目中的坑

#### 坑 1：只看总参数量

MoE 的总参数量大，但每个 token 只激活部分参数。比较模型时要看 activated parameters 和 FLOPs。

#### 坑 2：忽略通信成本

MoE 的 all-to-all 通信可能抵消一部分计算收益。

#### 坑 3：router 负载不均衡

如果大量 token 被路由到少数 expert，会造成过载、丢 token 或训练不稳定。

#### 坑 4：以为 expert 一定自动分工明确

expert specialization 需要分析验证，不是每个 expert 都有清楚语义标签。

#### 坑 5：推理部署低估复杂度

MoE serving 要处理动态路由、并发请求、专家分布、缓存和负载均衡。

### 面试官会怎么问

#### 问题 1：MoE 和 dense FFN 的区别是什么？

回答框架：

1. dense FFN 每个 token 都走同一个 FFN。
2. MoE 有多个 expert FFN。
3. router 为每个 token 选择少数 experts。
4. MoE 总参数量大，但每 token 激活计算较小。

#### 问题 2：router 的作用是什么？

回答框架：

1. 对 token hidden state 计算 expert 分数。
2. 选择 top-k experts。
3. 决定 token 的计算路径。
4. router 质量影响负载均衡和模型效果。

#### 问题 3：MoE 为什么能提高容量效率？

回答框架：

1. 增加 expert 数量可以增加总参数。
2. 每个 token 只激活少数 experts。
3. 因此总容量增长快于每 token 计算增长。
4. 这是稀疏激活的核心优势。

#### 问题 4：MoE 的主要系统瓶颈是什么？

回答框架：

1. token dispatch。
2. all-to-all 通信。
3. expert 负载不均衡。
4. expert parallelism 和 serving 调度。
5. capacity 和 token dropping。

#### 问题 5：MoE 一定比 dense 模型好吗？

回答框架：

1. 不一定。
2. MoE 有容量优势，但训练和部署复杂。
3. dense 模型更简单稳定。
4. 选择取决于算力、系统能力、任务和成本约束。

### 常见误区

1. 误区：MoE 就是多个模型投票。
   纠正：MoE 是在模型内部通过 router 对 token 进行专家路由，不是外部 ensemble。

2. 误区：MoE 总参数大，所以每 token 计算也同等变大。
   纠正：MoE 的关键是 sparse activation，每 token 只激活少数 experts。

3. 误区：router 只是辅助模块，不重要。
   纠正：router 决定计算路径，直接影响质量、负载和稳定性。

4. 误区：MoE 只解决算法问题。
   纠正：MoE 强依赖分布式系统、通信和 serving 工程。

5. 误区：expert 越多越好。
   纠正：expert 数越多，路由、通信、负载均衡和部署复杂度也越高。

### 小练习

1. 用自己的话解释 MoE 的 sparse activation。
2. 比较 dense FFN、top-1 MoE 和 top-2 MoE。
3. 写一个 top-1 MoE 的伪代码。
4. 解释为什么 MoE 需要负载均衡。
5. 分析 MoE 推理部署相比 dense 模型多了哪些复杂度。
6. 用 2 分钟回答“MoE 为什么是架构和系统强耦合设计”。

### 本讲总结

本讲最重要的结论：

1. MoE 通常把 Transformer FFN 替换为多个 expert FFN。
2. Router 根据 token 表示选择 top-k experts，形成条件计算路径。
3. MoE 的核心优势是总参数量大、每 token 激活计算可控。
4. MoE 的核心代价是路由、负载均衡、all-to-all 通信和 serving 复杂度。
5. MoE 不是简单比 dense 更好，而是在容量、计算、通信、稳定性和部署成本之间做 trade-off。

## 第 20 讲：MoE 训练稳定性与负载均衡

### 本讲目标

上一讲讲了 MoE 的基本架构。本讲继续讲 MoE 最难的部分：训练稳定性和负载均衡。MoE 的核心问题不是“有很多 experts”这么简单，而是 router 会动态分配 token，如果分配不均、通信瓶颈严重或 expert 容量设置不当，训练和推理都会出问题。

你需要掌握：

1. 为什么 MoE 容易负载不均衡。
2. load balancing loss 的作用。
3. expert capacity、token dropping 和 overflow 是什么。
4. router z-loss、router jitter 等稳定化方法的直觉。
5. MoE 训练中的 all-to-all 通信瓶颈。
6. MoE 推理服务为什么比 dense 模型更难。
7. 面试中如何讲 MoE 的工程 trade-off。

### 为什么 MoE 会负载不均衡

MoE 中 router 会为每个 token 选择 expert。

理想情况下，不同 experts 都能被充分使用。

但真实训练中，router 可能偏向少数 experts。

例如 64 个 experts 中，很多 token 都被路由到前几个 experts。

这会导致：

1. 热门 expert 过载。
2. 冷门 expert 学不到东西。
3. 设备间负载不均。
4. token 被丢弃或延迟变高。
5. 训练不稳定。

所以 MoE 必须显式处理负载均衡。

### Expert Collapse

Expert collapse 指大量 token 被路由到少数 experts，其他 experts 几乎不用。

这类似一种路由塌缩。

可能原因包括：

1. router 初期随机偏差被放大。
2. 某些 expert 早期训练得更快，吸引更多 token。
3. router 缺少均衡约束。
4. 数据分布不均。
5. learning rate 或初始化不合适。

一旦 expert collapse 出现，MoE 的大容量优势就会下降。

### Load Balancing Loss

load balancing loss 的目标是鼓励不同 experts 使用更均匀。

它通常会约束两件事：

1. 每个 expert 被分配到的 token 数。
2. router 给每个 expert 的平均概率。

直觉是：

```text
不要让所有 token 都挤到少数专家，也不要让 router 只偏好少数专家
```

load balancing loss 是 MoE 训练中的关键辅助损失。

Sparsely-Gated MoE、GShard 和 Switch Transformer 都强调：MoE 不是只把 expert 数量堆大，还必须让 router 的分配足够均衡。否则条件计算的容量优势会被 expert collapse、overflow 和设备热点抵消。

### Expert Capacity

每个 expert 在一个 batch 中能处理的 token 数通常有限。

这个上限叫 expert capacity。

如果某个 expert 被分配的 token 超过 capacity，就会出现 overflow。

处理 overflow 的方式包括：

1. 丢弃多余 token。
2. 送到备选 expert。
3. 增大 capacity factor。
4. 改善 router 均衡。

capacity 太小会丢 token。

capacity 太大又会浪费显存和计算。

### Capacity Factor

capacity factor 控制每个 expert 预留多少处理容量。

如果 batch 中 token 总数是 `N`，expert 数是 `E`，理想情况下每个 expert 处理：

```text
N / E
```

实际 capacity 可能设为：

```text
capacity = capacity_factor * N / E
```

capacity factor 越大，越不容易 overflow，但浪费越多。

capacity factor 越小，效率更高，但更容易丢 token。

工程上常用整数 capacity，可以理解为：

```text
capacity = ceil(capacity_factor * total_tokens / num_experts)
```

这里的 `ceil` 表示向上取整。capacity 不是越大越好，它是在 token dropping 风险和 padding/空槽浪费之间做折中。

### Token Dropping

当 expert overflow 时，一些 token 可能被 drop。

这会带来训练信号损失。

如果 token dropping 很多，模型质量会受到影响。

常见监控指标包括：

1. token drop rate。
2. 每个 expert 的 token count。
3. router entropy。
4. load balancing loss。
5. 每个 expert 的 utilization。

MoE 训练必须持续监控这些指标。

### Router z-loss

router logits 如果过大，softmax 会变得非常尖锐。

这会让 router 过早做出极端选择，降低探索和均衡性。

router z-loss 的直觉是约束 router logits 的尺度。

它帮助：

1. 防止 router logits 爆炸。
2. 减少过度自信路由。
3. 提升训练稳定性。
4. 缓解 expert collapse 风险。

不需要在面试中死记具体公式，理解它是在稳定 router 即可。

Switch Transformer 中的 router z-loss 关注 router logits 的数值尺度，目标是避免 logits 过大导致 softmax 过度尖锐。它不是替代 load balancing loss，而是另一个稳定 router 的辅助项。

### Router Jitter 和噪声

训练早期，router 可能过早偏向某些 experts。

加入噪声或 jitter 可以鼓励探索。

直觉是：

```text
不要让 router 太早锁死到固定专家，让不同 experts 都有机会学习
```

这类似探索机制。

但噪声太大也会影响收敛和稳定性。

### Top-1 和 Top-2 的稳定性差异

Top-1 routing 计算更省，但路由更硬。

每个 token 只走一个 expert，容易受到 router 错误选择影响。

Top-2 routing 更平滑。

每个 token 走两个 experts，输出按权重合并。

优点是：

1. 训练信号更丰富。
2. 路由更鲁棒。
3. expert collapse 风险可能降低。

缺点是计算和通信成本更高。

### All-to-All 通信瓶颈

在分布式 MoE 中，不同 experts 通常分布在不同 GPU 上。

一个 batch 的 token 需要按 router 结果发送到对应设备。

这会产生 all-to-all 通信。

通信流程通常包括：

1. token dispatch 到 expert 所在设备。
2. expert 计算。
3. expert 输出再 combine 回原 token 顺序。

如果通信慢，MoE 的计算优势会被抵消。

### Expert Parallelism

expert parallelism 是把不同 experts 放到不同设备上。

它和 data parallel、tensor parallel、pipeline parallel 可能同时存在。

MoE 训练中常见并行组合很复杂。

例如：

1. dense 层用 tensor parallel。
2. experts 用 expert parallel。
3. 不同 batch 用 data parallel。
4. 多层模型用 pipeline parallel。

所以 MoE 不是只改模型代码，还要重做分布式并行策略。

### MoE 推理服务难点

MoE 推理比 dense 模型更难。

原因包括：

1. 每个 token 的 expert 路由不同。
2. 不同请求的 token 分布不同。
3. 某些 experts 可能成为热点。
4. batching 更难做规则化。
5. expert 权重分布在多设备上。
6. 延迟受最慢 expert 或通信影响。

在线服务要考虑的不只是平均 FLOPs，还有 tail latency。

### MoE 监控指标

MoE 训练和服务需要额外监控：

1. expert utilization。
2. token drop rate。
3. load balancing loss。
4. router entropy。
5. 每个 expert 的 token count。
6. all-to-all 通信时间。
7. 每个 expert 的延迟。
8. 热点 expert 分布。
9. 质量指标和稳定性指标。

没有这些监控，MoE 出问题很难定位。

### 一个最小 capacity 和 overflow demo

下面这个 demo 模拟 top-1 routing 后的 expert capacity。它展示同一批 token 在不同 `capacity_factor` 下会产生多少 overflow 和 drop。

```python
import math
import torch


def route_with_capacity(expert_id, num_experts, capacity_factor):
    total_tokens = expert_id.numel()
    capacity = math.ceil(capacity_factor * total_tokens / num_experts)
    kept = torch.zeros_like(expert_id, dtype=torch.bool)
    counts = torch.zeros(num_experts, dtype=torch.long)
    dropped = torch.zeros(num_experts, dtype=torch.long)

    for token_idx, expert in enumerate(expert_id.tolist()):
        if counts[expert] < capacity:
            kept[token_idx] = True
            counts[expert] += 1
        else:
            dropped[expert] += 1

    return capacity, kept, counts, dropped


expert_id = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2, 3, 3])
num_experts = 4

for capacity_factor in [1.0, 1.5, 2.0]:
    capacity, kept, counts, dropped = route_with_capacity(
        expert_id, num_experts, capacity_factor
    )
    print("capacity_factor:", capacity_factor)
    print("capacity per expert:", capacity)
    print("kept tokens:", int(kept.sum().item()), "dropped tokens:", int((~kept).sum().item()))
    print("processed counts:", counts.tolist())
    print("dropped counts:", dropped.tolist())
```

典型输出会显示：当 `capacity_factor=1.0` 时，热门 expert 0 会 overflow；capacity factor 增大后 drop 变少，但每个 expert 预留的槽位也更多，资源浪费风险更高。

### 面试中怎么讲

如果面试官问“MoE 为什么需要负载均衡”，可以这样回答：

```text
MoE 中 router 会把 token 分配给 experts。如果没有负载均衡，router 可能把大量 token 分到少数 experts，导致这些 experts 过载，而其他 experts 学不到东西。这会造成 expert collapse、token dropping、设备负载不均和训练不稳定。因此 MoE 通常需要 load balancing loss、capacity 控制、router 稳定化和负载监控。
```

如果面试官问“MoE 的系统瓶颈是什么”，可以这样回答：

```text
MoE 的系统瓶颈主要来自动态路由和 all-to-all 通信。token 要根据 router 结果被发送到不同设备上的 experts，expert 计算后再把结果合并回原顺序。如果路由不均衡，某些 expert 会成为热点；如果通信慢，MoE 的稀疏计算优势会被抵消。推理时还要处理动态 batching、专家分布和 tail latency。
```

### 真实项目中的坑

#### 坑 1：只看平均 expert utilization

平均值可能掩盖热点 expert，需要看分布、最大值和长尾。

#### 坑 2：capacity factor 设置过小

会导致 token dropping，训练信号丢失，模型质量下降。

#### 坑 3：capacity factor 设置过大

虽然减少 overflow，但会浪费显存和计算。

#### 坑 4：忽略 router logits 尺度

router 过度自信会导致早期路由塌缩，需要 z-loss 或其他稳定手段。

#### 坑 5：低估推理 tail latency

MoE 服务中最慢 expert、通信和热点请求会显著影响尾延迟。

### 面试官会怎么问

#### 问题 1：什么是 expert collapse？

回答框架：

1. 大量 token 被路由到少数 experts。
2. 其他 experts 使用率很低。
3. 模型容量没有被充分利用。
4. 可能导致过载和训练不稳定。

#### 问题 2：load balancing loss 解决什么问题？

回答框架：

1. 鼓励 token 更均匀分配到 experts。
2. 防止 router 偏向少数 experts。
3. 提高 expert utilization。
4. 缓解 expert collapse 和设备负载不均。

#### 问题 3：capacity factor 是什么？

回答框架：

1. 控制每个 expert 在 batch 中能处理多少 token。
2. 通常相对平均 token 数设置冗余。
3. 太小会 overflow 和 drop token。
4. 太大会浪费资源。

#### 问题 4：MoE 为什么通信成本高？

回答框架：

1. experts 分布在不同设备。
2. token 要按路由结果发送到 expert 所在设备。
3. 计算后还要 combine 回原顺序。
4. 这个 all-to-all 通信可能成为瓶颈。

#### 问题 5：Top-1 和 Top-2 routing 怎么取舍？

回答框架：

1. Top-1 更省计算和通信。
2. Top-2 更平滑、更鲁棒。
3. Top-2 可能改善训练信号和稳定性。
4. 但代价是更高成本。

### 常见误区

1. 误区：MoE 只要加 load balancing loss 就稳定。
   纠正：还要看 capacity、router logits、通信、并行策略和数据分布。

2. 误区：expert utilization 均匀就一定质量好。
   纠正：均衡只是必要条件之一，还要看任务效果和专家学习质量。

3. 误区：token dropping 很少可以不管。
   纠正：少量 dropping 也可能集中在关键样本或特定领域，需要分布分析。

4. 误区：MoE 推理一定比 dense 快。
   纠正：通信、调度和热点 expert 可能抵消稀疏计算收益。

5. 误区：router 不需要单独监控。
   纠正：router 是 MoE 稳定性的核心，必须监控 entropy、logits 和分配分布。

### 小练习

1. 解释 expert collapse 的原因和后果。
2. 设计一组 MoE 训练监控指标。
3. 说明 capacity factor 太大和太小分别有什么问题。
4. 比较 top-1 和 top-2 routing 的稳定性与成本。
5. 画出 MoE all-to-all 的 token dispatch 和 combine 流程。
6. 用 2 分钟回答“MoE 为什么难训练、难部署”。

### 本讲总结

本讲最重要的结论：

1. MoE 的核心难点是动态路由带来的负载不均衡和通信复杂度。
2. expert collapse 会让少数 experts 过载，其他 experts 学不到东西。
3. load balancing loss、capacity factor、router z-loss 和 routing 噪声都是稳定 MoE 的常见手段。
4. all-to-all 通信和 expert parallelism 是 MoE 系统工程的核心瓶颈。
5. MoE 推理服务要关注热点 expert、batching、KV Cache、通信和 tail latency。
6. 面试中讲 MoE，必须同时覆盖算法、训练稳定性和分布式系统 trade-off。

## 第 21 讲：Encoder-only、Decoder-only 与 Encoder-Decoder 再比较

### 本讲目标

第一册里我们已经比较过 BERT、GPT 和 T5 类架构。本讲从进阶角度重新比较 Encoder-only、Decoder-only 和 Encoder-Decoder：它们的 attention mask、训练目标、信息流、推理方式和适用任务有什么本质差异？为什么当前通用大模型主要采用 decoder-only？

你需要掌握：

1. 三类 Transformer 架构的结构差异。
2. bidirectional attention 和 causal attention 的差异。
3. MLM、自回归 LM、seq2seq 目标的差异。
4. 为什么 decoder-only 成为 LLM 主流。
5. Encoder-Decoder 仍然适合哪些任务。
6. 面试中如何从训练、推理和产品形态比较架构。

### Encoder-only 架构

Encoder-only 代表模型是 BERT。

它的核心特点是使用双向 self-attention。

每个 token 可以看到左右两侧上下文：

```text
x_i can attend to x_1 ... x_T
```

这非常适合理解类任务，例如：

1. 文本分类。
2. 句子匹配。
3. token classification。
4. embedding 表示学习。
5. reranking。

Encoder-only 的典型训练目标是 masked language modeling，也就是 MLM。

BERT 论文的关键点是 deep bidirectional representations：模型在各层联合利用左右上下文，更适合表示学习和理解任务。它不是为逐 token 自回归生成设计的架构。

### MLM 的特点

MLM 会随机 mask 一部分 token，让模型预测被 mask 的 token。

优点：

1. 可以利用双向上下文。
2. 适合理解任务。
3. 学到较强语义表示。

缺点：

1. 训练目标和自回归生成不一致。
2. 不天然适合逐 token 生成。
3. mask token 在预训练和下游场景之间存在 mismatch。

所以 BERT 类模型强在理解，不是现代对话式生成 LLM 的主流架构。

### Decoder-only 架构

Decoder-only 代表模型是 GPT 类模型。

它使用 causal self-attention。

第 `i` 个 token 只能看到自己和之前的 token：

```text
x_i can attend to x_1 ... x_i
```

训练目标通常是 next-token prediction。

推理时也是逐 token 生成。

这带来一个重要优势：

```text
训练目标和生成方式高度一致
```

### Decoder-only 为什么适合 LLM

现代通用 LLM 多采用 decoder-only，原因包括：

1. next-token prediction 可以利用海量文本自监督训练。
2. 训练和推理形式一致。
3. 所有任务都可以转成 prompt 到 continuation。
4. 对话、多轮交互、代码生成、工具调用都自然适配。
5. KV Cache 支持高效自回归推理。
6. 架构简单，易于扩展和部署。

这不是说 decoder-only 在所有任务上都理论最优，而是它在大规模预训练、通用生成和产品化上形成了最强闭环。

GPT 类 decoder-only 的核心工程优势，是训练目标、推理方式和产品交互形式高度一致：训练时预测下一个 token，推理时也是逐 token 续写；prompt、对话历史、工具说明和输出都能组织成一条序列。

### Encoder-Decoder 架构

Encoder-Decoder 代表模型包括原始 Transformer、T5、BART 等。

它有两个部分：

1. encoder 编码输入序列。
2. decoder 自回归生成输出序列。

decoder 中通常有 cross-attention，用来读取 encoder 输出。

T5 的代表性思路是把多种 NLP 任务统一成 text-to-text 形式。它说明 encoder-decoder 并不过时，而是在输入输出边界明确、需要强条件生成时仍然非常自然。

典型任务包括：

1. 翻译。
2. 摘要。
3. 文本改写。
4. 条件生成。
5. 某些结构化 seq2seq 任务。

### Encoder-Decoder 的优势

Encoder-Decoder 的优势是输入和输出角色分明。

encoder 可以完整双向理解输入，decoder 再条件生成输出。

这适合明确的 source-to-target 任务。

例如机器翻译中：

```text
source sentence -> target sentence
```

encoder 可以双向看完整 source，decoder 再逐步生成 target。

这种结构在很多传统 NLP 任务中非常自然。

### Encoder-Decoder 的代价

Encoder-Decoder 也有代价：

1. 架构更复杂。
2. encoder 和 decoder 都要计算。
3. cross-attention 增加实现复杂度。
4. 对通用对话 prompt-continuation 形式不如 decoder-only 简洁。
5. KV Cache 设计和 serving 形态更复杂。

在大规模通用 LLM 时代，统一 prompt 到 continuation 的 decoder-only 方案更容易扩展。

### Attention Mask 的差异

三类架构最核心差异之一是 mask。

Encoder-only：

```text
双向 attention，无 causal mask
```

Decoder-only：

```text
causal attention，只看左侧上下文
```

Encoder-Decoder：

```text
encoder 双向 attention
decoder causal attention
decoder cross-attention 读取 encoder 输出
```

mask 决定了信息流，也决定了适合的训练目标和推理方式。

### 可运行 mask 对比 demo

下面这个 demo 不实现完整 attention，只打印三类架构中 token 之间的可见性矩阵。`1` 表示 query token 可以 attend 到 key token，`0` 表示不能。

```python
import torch


def encoder_only_mask(seq_len):
    return torch.ones(seq_len, seq_len, dtype=torch.int)


def decoder_only_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.int))


def encoder_decoder_masks(source_len, target_len):
    encoder_self = torch.ones(source_len, source_len, dtype=torch.int)
    decoder_self = torch.tril(torch.ones(target_len, target_len, dtype=torch.int))
    cross_attention = torch.ones(target_len, source_len, dtype=torch.int)
    return encoder_self, decoder_self, cross_attention


seq_len = 5
print("encoder-only self-attention")
print(encoder_only_mask(seq_len))

print("decoder-only causal self-attention")
print(decoder_only_mask(seq_len))

enc_self, dec_self, cross = encoder_decoder_masks(source_len=4, target_len=3)
print("encoder-decoder encoder self-attention")
print(enc_self)
print("encoder-decoder decoder self-attention")
print(dec_self)
print("encoder-decoder cross-attention")
print(cross)
```

这个 demo 能直接看出三点：encoder-only 每个 token 可以看全句；decoder-only 第 `i` 个 token 只能看自己和左侧 token；encoder-decoder 中 encoder 双向看 source，decoder 自回归看 target，同时通过 cross-attention 读取完整 source 表示。

### 训练目标差异

Encoder-only 常见目标：

```text
masked language modeling
```

Decoder-only 常见目标：

```text
next-token prediction
```

Encoder-Decoder 常见目标：

```text
conditional generation / denoising seq2seq
```

训练目标不仅影响能力，还影响数据组织和产品形态。

### 推理方式差异

Encoder-only 通常不直接用于开放式生成。

它更多用于输出表示、分类分数或 token-level 标签。

Decoder-only 推理是逐 token 生成，非常适合对话和生成。

Encoder-Decoder 推理先编码输入，再由 decoder 逐 token 生成输出。

因此在 serving 中：

1. decoder-only 的 prompt prefill + decode 形态最统一。
2. encoder-decoder 需要管理 encoder states 和 decoder KV Cache。
3. encoder-only 更像表示模型或判别模型。

### 为什么当前通用 LLM 以 Decoder-only 为主

核心原因是统一性。

几乎所有任务都能表示为：

```text
prompt -> continuation
```

例如：

1. 问答：问题 -> 答案。
2. 翻译：翻译指令和原文 -> 译文。
3. 摘要：摘要指令和文档 -> 摘要。
4. 代码：上下文和需求 -> 代码。
5. 工具调用：对话和工具说明 -> function call。

这种统一形式让数据、训练、推理和产品交互都变得简单。

### Encoder-only 现在还有什么价值

Encoder-only 并没有过时。

它在很多场景仍然重要：

1. embedding 模型。
2. reranker。
3. 检索系统。
4. 分类审核模型。
5. token-level NER 或抽取。
6. 小而快的理解模型。

RAG 系统中，encoder-only 或双塔模型常用于 embedding，cross-encoder 常用于 reranking。

### Encoder-Decoder 现在还有什么价值

Encoder-Decoder 仍适合强条件生成任务。

例如：

1. 输入输出边界清楚的翻译。
2. 长文档摘要。
3. 语音识别或多模态 encoder 到文本 decoder。
4. 某些高质量 seq2seq 数据充足的任务。

多模态模型中，也经常出现 encoder + decoder 的组合思想。

例如 vision encoder 编码图像，LLM decoder 生成文本。

### 面试中怎么讲

如果面试官问“为什么 GPT 类模型用 decoder-only”，可以这样回答：

```text
Decoder-only 使用 causal self-attention 和 next-token prediction，训练目标和自回归生成方式一致。它可以用海量文本做自监督训练，并且几乎所有任务都能转成 prompt 到 continuation 的形式。对话、代码生成、工具调用和多轮交互都自然适配这种统一生成框架，所以现代通用 LLM 多采用 decoder-only。
```

如果面试官问“三类 Transformer 如何比较”，可以这样回答：

```text
Encoder-only 使用双向 attention，适合理解和表示学习；decoder-only 使用 causal attention，适合自回归生成和通用 LLM；encoder-decoder 用 encoder 双向理解输入，再由 decoder 条件生成输出，适合翻译、摘要等 source-to-target 任务。选择哪种架构取决于训练目标、推理方式和产品形态。
```

### 真实项目中的坑

#### 坑 1：说 decoder-only 全面替代 encoder

检索、embedding、reranking、分类等场景中 encoder-only 仍然非常有价值。

#### 坑 2：只从模型能力比较，不看服务形态

架构选择会影响 KV Cache、batching、延迟、吞吐和系统复杂度。

#### 坑 3：把 Encoder-Decoder 看成过时架构

它在翻译、摘要、多模态 encoder 到文本 decoder 等场景仍然自然。

#### 坑 4：忽略训练目标和推理目标一致性

decoder-only 的优势很大程度来自 next-token training 和 autoregressive inference 一致。

#### 坑 5：把 mask 当成实现细节

mask 决定信息流，是架构本质差异之一。

### 面试官会怎么问

#### 问题 1：BERT 和 GPT 架构差异是什么？

回答框架：

1. BERT 是 encoder-only，双向 attention。
2. GPT 是 decoder-only，causal attention。
3. BERT 常用 MLM，GPT 常用 next-token prediction。
4. BERT 强理解，GPT 强生成。

#### 问题 2：Encoder-Decoder 为什么适合翻译？

回答框架：

1. encoder 双向理解完整源句。
2. decoder 自回归生成目标句。
3. cross-attention 让 decoder 读取 source 表示。
4. source-to-target 边界清晰。

#### 问题 3：为什么 decoder-only 适合 instruction following？

回答框架：

1. 指令和上下文可以放在 prompt 中。
2. 模型按 continuation 生成答案。
3. 训练和推理都是 next-token 形式。
4. 多轮对话也能串成单一序列。

#### 问题 4：Encoder-only 在 LLM 时代还有什么用？

回答框架：

1. embedding。
2. reranking。
3. 分类和审核。
4. 抽取和 token-level 理解。
5. RAG 检索系统组件。

#### 问题 5：三种架构如何影响推理系统？

回答框架：

1. encoder-only 通常输出表示或分数。
2. decoder-only 是 prefill + decode。
3. encoder-decoder 需要 encoder states 和 decoder KV Cache。
4. 系统复杂度和 batching 方式不同。

### 常见误区

1. 误区：encoder-only 不能用于生成，所以没价值。
   纠正：它在表示、检索、分类和 reranking 中很重要。

2. 误区：decoder-only 一定在所有任务上最优。
   纠正：它是通用生成闭环最强，不代表所有任务理论最优。

3. 误区：Encoder-Decoder 只是老架构。
   纠正：多模态和强条件生成中仍然常见 encoder-decoder 思想。

4. 误区：架构差异只是有没有 decoder。
   纠正：核心还包括 attention mask、训练目标、推理方式和系统形态。

5. 误区：MLM 和 next-token prediction 都是语言模型目标，差不多。
   纠正：它们的信息条件和推理一致性不同。

### 小练习

1. 画出 encoder-only、decoder-only、encoder-decoder 的信息流。
2. 比较 MLM 和 next-token prediction。
3. 用自己的话解释为什么 decoder-only 成为通用 LLM 主流。
4. 举三个 encoder-only 仍然适合的真实场景。
5. 举两个 encoder-decoder 仍然有优势的任务。
6. 用 2 分钟回答“三类 Transformer 如何选择”。

### 本讲总结

本讲最重要的结论：

1. Encoder-only 使用双向 attention，适合理解和表示学习。
2. Decoder-only 使用 causal attention，训练和自回归生成高度一致，是现代通用 LLM 主流。
3. Encoder-Decoder 用 encoder 理解输入、decoder 条件生成，适合 source-to-target 任务。
4. 架构选择不是只看模型能力，还要看训练目标、推理方式、数据组织和产品形态。
5. LLM 时代 encoder-only 和 encoder-decoder 仍然有重要应用场景。

## 第 22 讲：状态空间模型与 Transformer 替代路线

### 本讲目标

Transformer 的核心瓶颈是 attention 的序列长度平方复杂度。状态空间模型，简称 SSM，是近几年重要的替代路线之一，代表包括 S4、Mamba 等。本讲不追求推导完整控制理论，而是帮助你理解：SSM 想解决什么问题、它和 attention 的信息流有什么不同、为什么 Mamba 类模型受到关注，以及为什么它们还没有完全替代 Transformer。

你需要掌握：

1. 什么是状态空间模型。
2. SSM 如何建模序列。
3. SSM 和 RNN、CNN、Attention 的关系。
4. S4、Mamba 的核心直觉。
5. SSM 的长序列效率优势。
6. SSM 的局限和工程 trade-off。
7. 面试中如何评价 Transformer 替代路线。

### Transformer 为什么需要替代路线

Transformer 很强，但也有明显代价：

1. 标准 attention 是 `O(T^2)`。
2. 长上下文训练显存压力大。
3. 推理 KV Cache 随长度增长。
4. 超长序列下吞吐和延迟压力大。

因此研究者一直在寻找更高效的序列建模方式。

替代路线包括：

1. 稀疏 attention。
2. 线性 attention。
3. 状态空间模型。
4. 卷积序列模型。
5. 混合架构。

SSM 是其中非常重要的一类。

### 什么是状态空间模型

状态空间模型用一个隐含状态来描述系统随时间演化。

经典形式可以写成：

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t + D x_t
```

其中：

1. `x_t` 是输入。
2. `h_t` 是隐藏状态。
3. `y_t` 是输出。
4. `A` 控制状态如何随时间演化。
5. `B` 控制输入如何写入状态。
6. `C` 控制状态如何读出。
7. `D` 是输入到输出的直接通路。

直觉上，SSM 是一种带记忆的序列模型。

### SSM 和 RNN 的关系

SSM 看起来很像 RNN，因为都有状态递推。

RNN 是：

```text
h_t = f(h_{t-1}, x_t)
```

SSM 也维护状态，但它通常有更结构化的状态转移，并且可以通过数学变换实现高效并行或卷积计算。

可以粗略理解为：

```text
SSM 是更结构化、更适合长序列建模和高效计算的一类递推/卷积模型
```

### SSM 和 CNN 的关系

线性 SSM 在某些条件下可以转化为长卷积。

也就是说，序列输出可以看成输入和一个长卷积核的卷积。

这带来一个重要优势：

1. 训练时可以并行卷积。
2. 推理时可以递推更新状态。
3. 兼顾并行训练和线性推理。

这也是 SSM 吸引人的地方。

### SSM 和 Attention 的差异

Attention 的信息流是显式 token-to-token 交互。

每个 token 可以直接读取其他 token。

SSM 的信息流更像通过状态压缩历史：

```text
past tokens -> hidden state -> current output
```

差异可以这样理解：

```text
Attention: 显式检索历史 token
SSM: 把历史压缩进状态中持续更新
```

这带来 trade-off：
1. SSM 长序列效率更好。
2. Attention 精确检索任意历史 token 更直接。
3. SSM 依赖状态是否能保留关键信息。
4. Attention 代价更高但表达更灵活。

### S4 的核心直觉

S4 是结构化状态空间模型的重要代表。

它试图让 SSM 能高效处理长序列，并通过结构化矩阵设计解决训练和计算问题。

你不需要在面试中推导完整 S4 公式。

需要理解的是：

```text
S4 用结构化状态空间模型把长程依赖建模变成高效序列变换，目标是在长序列任务上比 attention 更高效
```

S4 对后续 Mamba 等模型有重要影响。

从 S4 论文的边界看，它不是简单“把 RNN 换个名字”，而是通过结构化状态空间参数化，让 SSM 在长序列上更高效、更可训练。面试中可以把 S4 说成现代 SSM 路线的重要起点，而不是要求手推完整连续时间控制方程。

### Mamba 的核心直觉

Mamba 是选择性状态空间模型的代表。

它的关键思想是 selective scan。

传统 SSM 的参数对所有输入可能比较固定。

Mamba 引入输入相关的选择机制，让模型根据当前 token 动态决定：

1. 什么信息写入状态。
2. 什么信息保留。
3. 什么信息输出。

这让 SSM 更像有输入相关控制的序列模型。

Mamba 论文强调的关键问题是：很多 subquadratic 序列模型在语言这类离散模态上不如 attention，一个原因是内容相关推理能力不足。Mamba 通过让 SSM 参数依赖输入，选择性地传播或遗忘信息，并用硬件感知 parallel scan 解决效率问题。

Mamba-2 / SSD 进一步说明 SSM 和 attention 在结构上并不是完全割裂的两类模型，而是可以通过 structured state space duality 建立联系。这个边界很重要：新架构趋势更像“融合和泛化”，而不只是“谁淘汰谁”。

### Selective Scan 为什么重要

如果状态转移完全固定，模型对不同内容的适应性有限。

语言序列中，不同 token 的重要性差异很大。

例如：

1. 关键词需要长期保留。
2. 停用词可以快速遗忘。
3. 代码变量定义需要后续引用。
4. 对话中的用户约束需要持续记忆。

Selective scan 的直觉是：

```text
让状态更新规则依赖输入内容，从而更灵活地选择记什么和忘什么
```

### SSM 的优势

SSM 类模型的主要优势：

1. 对长序列更高效。
2. 推理时不需要像 attention 那样保存完整 KV Cache。
3. 复杂度通常接近线性。
4. 适合流式处理。
5. 在某些长序列任务上表现强。

这些优势让 SSM 成为 Transformer 替代路线的重要候选。

### 一个最小 SSM 递推 demo

下面这个 demo 实现一维 SSM 递推。它展示两个直觉：状态会把历史输入压缩成一个滚动记忆；`A` 越接近 1，历史保留越久，`A` 越小，遗忘越快。

```python
def run_scalar_ssm(inputs, A, B=1.0, C=1.0, D=0.0):
    state = 0.0
    outputs = []
    states = []
    for x_t in inputs:
        state = A * state + B * x_t
        y_t = C * state + D * x_t
        states.append(round(state, 4))
        outputs.append(round(y_t, 4))
    return states, outputs


inputs = [1.0, 0.0, 0.0, 0.0, 0.0]

for A in [0.2, 0.8]:
    states, outputs = run_scalar_ssm(inputs, A=A)
    print("A =", A)
    print("states:", states)
    print("outputs:", outputs)
```

典型输出：

```text
A = 0.2
states: [1.0, 0.2, 0.04, 0.008, 0.0016]
A = 0.8
states: [1.0, 0.8, 0.64, 0.512, 0.4096]
```

这不是 Mamba 的完整 selective scan，只是 SSM 的最小机制演示：历史不是以 KV Cache 形式逐 token 保存，而是被不断压缩进状态。Mamba 的 selective scan 可以理解为让类似的状态更新规则根据输入动态变化。

### SSM 的局限

SSM 也有局限：

1. 精确检索远处 token 不如 attention 直观。
2. 历史信息被压缩到状态中，可能丢失细节。
3. 大规模通用 LLM 生态不如 Transformer 成熟。
4. 训练 recipe、并行实现和硬件优化仍在发展。
5. 对复杂 in-context learning 的表现需要持续验证。

所以不能简单说 SSM 会直接替代 Transformer。

### 为什么 Transformer 仍然强

Transformer 的优势包括：

1. attention 可以显式读取上下文中任意 token。
2. in-context learning 能力强。
3. 工程生态成熟。
4. 训练 recipe 成熟。
5. 硬件和 kernel 优化充分。
6. 已经在超大规模上验证。

SSM 要替代 Transformer，不只要理论复杂度更好，还要在质量、扩展性、生态和部署上全面竞争。

### 混合路线

现实中很可能不是纯替代，而是混合。

例如：

1. 部分层用 attention。
2. 部分层用 SSM。
3. 局部用卷积。
4. 长程记忆用 SSM。
5. 精确检索用 attention。

这种混合架构试图同时获得 attention 的表达能力和 SSM 的长序列效率。

下一讲会专门讲混合架构。

### 如何评价 Transformer 替代路线

面试中不要只说“复杂度更低”。

应该从多个维度评价：

1. 训练复杂度。
2. 推理复杂度。
3. 长上下文能力。
4. 精确检索能力。
5. in-context learning 能力。
6. 硬件友好性。
7. 工程生态成熟度。
8. 大规模验证结果。

一个替代路线真正成功，需要在这些维度形成整体优势。

### 面试中怎么讲

如果面试官问“SSM 和 attention 有什么区别”，可以这样回答：

```text
Attention 是显式 token-to-token 交互，每个 token 可以直接读取上下文中其他 token，因此表达灵活但复杂度高。SSM 通过隐藏状态递推或等价长卷积来建模序列，把历史信息压缩进状态中，通常能以接近线性复杂度处理长序列。它更高效，但精确检索远处 token 和复杂 in-context learning 是否能完全达到 Transformer 水平，需要具体模型和规模验证。
```

如果面试官问“Mamba 为什么受到关注”，可以这样回答：

```text
Mamba 是选择性状态空间模型，它在 SSM 中引入输入相关的选择机制，让模型根据当前 token 动态决定写入、保留和输出哪些信息。相比固定 SSM，它更适合语言这种内容依赖很强的序列，同时保持较好的长序列效率，因此成为 Transformer 替代或混合路线中的重要方向。
```

### 真实项目中的坑

#### 坑 1：只看 O(T) 复杂度

复杂度低不代表实际效果、硬件效率和生态都更好。

#### 坑 2：忽略精确检索能力

很多 LLM 任务需要从上下文中精确找证据，attention 在这方面很直接。

#### 坑 3：把 SSM 当成 RNN 简单复活

现代 SSM 有结构化设计、并行训练和 selective scan，不等同于传统 RNN。

#### 坑 4：认为替代路线一定完全替代 Transformer

混合架构可能比纯替代更现实。

#### 坑 5：忽略大规模验证

小规模长序列任务表现好，不代表超大规模通用 LLM 一定更强。

### 面试官会怎么问

#### 问题 1：状态空间模型是什么？

回答框架：

1. 用隐藏状态描述序列历史。
2. 状态随输入递推更新。
3. 输出由状态和当前输入生成。
4. 可以看成结构化的序列记忆模型。

#### 问题 2：SSM 为什么适合长序列？

回答框架：

1. 不需要显式构造 `T × T` attention matrix。
2. 训练可通过卷积或 scan 并行化。
3. 推理可递推更新状态。
4. 复杂度通常更接近线性。

#### 问题 3：Mamba 的 selective scan 解决什么问题？

回答框架：

1. 传统 SSM 参数较固定。
2. 语言序列需要根据内容选择记忆和遗忘。
3. selective scan 让状态更新依赖输入。
4. 提升内容适应性。

#### 问题 4：SSM 为什么没有完全替代 Transformer？

回答框架：

1. Transformer 显式上下文检索能力强。
2. in-context learning 表现成熟。
3. 工程生态和训练 recipe 完善。
4. SSM 在超大规模通用能力上仍需更多验证。

#### 问题 5：混合架构为什么有吸引力？

回答框架：

1. attention 负责精确 token 交互。
2. SSM 负责高效长程状态建模。
3. 卷积可处理局部模式。
4. 混合可以在效率和表达力之间折中。

### 常见误区

1. 误区：SSM 就是老 RNN。
   纠正：现代 SSM 有结构化状态、并行训练和输入选择机制。

2. 误区：线性复杂度一定更好。
   纠正：质量、硬件效率、任务需求和生态同样重要。

3. 误区：attention 的 O(T^2) 说明它必然会被淘汰。
   纠正：attention 的表达能力和生态优势仍然很强。

4. 误区：Mamba 适合所有 LLM 任务。
   纠正：不同任务对检索、推理和上下文交互需求不同，需要评估。

5. 误区：替代路线只看模型结构。
   纠正：还要看训练数据、优化、kernel、分布式和 serving。

### 小练习

1. 写出状态空间模型的基本递推形式。
2. 比较 SSM、RNN、CNN 和 attention 的信息流。
3. 用自己的话解释 selective scan。
4. 设计一个任务来比较 SSM 和 Transformer 的长上下文能力。
5. 分析混合 attention + SSM 架构可能的优缺点。
6. 用 2 分钟回答“SSM 会不会替代 Transformer”。

### 本讲总结

本讲最重要的结论：

1. SSM 用隐藏状态递推建模序列，是 Transformer 替代路线的重要方向。
2. SSM 的优势是长序列效率和流式推理友好。
3. Attention 的优势是显式 token-to-token 交互和成熟的 in-context learning 能力。
4. Mamba 通过 selective scan 提升 SSM 对输入内容的适应性。
5. SSM 是否替代 Transformer 不能只看复杂度，还要看质量、规模验证、硬件效率和生态。
6. 混合架构可能是兼顾效率和表达能力的现实路线。

## 第 23 讲：混合架构：Attention、SSM 与卷积的结合

### 本讲目标

上一讲讲了 SSM 作为 Transformer 替代路线。本讲进一步讨论更现实的方向：混合架构。很多时候不是 attention、SSM、卷积三选一，而是把它们组合起来，让不同模块承担不同功能。本讲帮助你理解混合架构为什么有吸引力，以及如何从效率、表达能力和工程复杂度角度评价它。

你需要掌握：

1. 为什么需要混合架构。
2. attention、SSM、卷积分别擅长什么。
3. 常见混合方式有哪些。
4. 混合架构对长上下文和推理效率有什么帮助。
5. 混合架构的训练和部署难点。
6. 面试中如何评价新架构，而不是只看宣传词。

### 为什么不是简单替代

Transformer 的 attention 很贵，但它有很强的显式上下文读取能力。

SSM 更高效，但历史信息被压缩进状态中。

卷积很擅长局部模式，但长程依赖能力有限。

所以现实选择往往不是：

```text
用 SSM 完全替代 attention
```

而是：

```text
让不同模块处理它们擅长的问题
```

这就是混合架构的动机。

从近期架构演进看，混合不只是工程拼装。Mamba-2/SSD 强调 attention 与 SSM 在结构上存在联系；Conformer 则是卷积增强 Transformer 的经典例子，用 attention 捕捉全局依赖，用卷积补局部模式。面试中更稳的说法是：新架构通常是在不同序列算子的归纳偏置、效率和硬件实现之间重新组合，而不是简单把某个模块替换掉。

### 三类模块的分工

可以粗略理解为：

```text
Attention: 精确 token-to-token 信息读取
SSM: 高效长程状态建模
Convolution: 局部模式和短程归纳偏置
```

attention 擅长从上下文中找特定 token。

SSM 擅长以线性或接近线性的方式处理长序列。

卷积擅长捕捉局部 n-gram、局部语音/视觉模式或短程结构。

### Attention + SSM

一种混合方式是在部分层使用 attention，部分层使用 SSM。

例如：

1. 每隔几层插入 attention。
2. 大多数层用 SSM，少数层用 attention。
3. 前几层用卷积/SSM，后几层用 attention。

直觉是：

1. SSM 提供高效长程状态传播。
2. attention 提供精确上下文检索。
3. 两者组合比单独使用更平衡。

### Attention + Convolution

卷积曾经在很多高效 Transformer 中出现。

加入卷积的原因包括：

1. 提供局部归纳偏置。
2. 捕捉短程模式。
3. 降低纯 attention 的压力。
4. 对语音、视觉、多模态输入尤其有用。

在文本中，局部顺序模式也很重要。

例如词组、代码局部结构、标点和格式模式，都可能受益于局部算子。

### SSM + Convolution

SSM 本身和卷积有紧密关系。

线性 SSM 可以转化为长卷积。

很多现代序列模型会把卷积作为局部混合模块，再用 SSM 处理更长范围。

这种组合的直觉是：

```text
卷积处理局部细节，SSM 处理更长程的状态传播
```

### 混合架构的优势

混合架构可能带来：

1. 比 full attention 更低的长序列成本。
2. 比纯 SSM 更强的精确检索能力。
3. 比纯 attention 更好的局部归纳偏置。
4. 更灵活的效率和质量 trade-off。
5. 更适合多模态或流式场景。

它的核心价值是组合不同模块的优势。

### 混合架构的代价

代价也很明显：

1. 架构更复杂。
2. 训练 recipe 不成熟。
3. kernel 和框架支持更难。
4. 模块比例需要大量实验。
5. 推理系统需要适配不同算子。
6. 可解释性和 debug 难度更高。

所以混合架构不是简单“堆模块”，而是系统设计。

### 如何设计混合比例

常见问题是：多少层用 attention，多少层用 SSM？

没有固定答案。

需要考虑：

1. 目标上下文长度。
2. 任务是否需要精确检索。
3. 训练数据是否包含长程依赖。
4. 推理延迟和吞吐要求。
5. 硬件对不同算子的支持。
6. 模型规模和预算。

如果任务强依赖精确引用，attention 比例不能太低。

如果任务是流式长序列建模，SSM 或卷积比例可以更高。

### 一个最小混合比例估算 demo

下面这个 demo 不模拟真实 kernel，只用粗略成本帮助理解：如果一部分层用 attention，一部分层用 SSM 或卷积，长序列 token 交互成本会如何变化。这里把 attention 近似为 `seq_len * seq_len`，SSM 和卷积近似为 `seq_len * window_or_state`。

```python
def estimate_layer_cost(seq_len, layer_type, width=64):
    if layer_type == "attention":
        return seq_len * seq_len
    if layer_type == "ssm":
        return seq_len * width
    if layer_type == "conv":
        return seq_len * width
    raise ValueError(f"unknown layer type: {layer_type}")


def estimate_stack_cost(seq_len, layers):
    return sum(estimate_layer_cost(seq_len, layer_type) for layer_type in layers)


seq_len = 8192
all_attention = ["attention"] * 12
hybrid = ["attention", "ssm", "ssm", "conv"] * 3
mostly_ssm = ["attention"] * 2 + ["ssm"] * 8 + ["conv"] * 2

for name, layers in [
    ("all_attention", all_attention),
    ("hybrid", hybrid),
    ("mostly_ssm", mostly_ssm),
]:
    cost = estimate_stack_cost(seq_len, layers)
    print(name, "layers=", layers.count("attention"), "attention +", layers.count("ssm"), "ssm +", layers.count("conv"), "conv")
    print(name, "relative cost:", round(cost / estimate_stack_cost(seq_len, all_attention), 4))
```

典型输出会显示：减少 full attention 层数可以显著降低粗略长序列成本，但这不代表质量一定更好。真正设计时还要看是否需要精确检索、是否有高效 kernel、以及长上下文评估是否通过。

### 长上下文场景中的混合架构

长上下文任务有两类需求：

1. 大范围持续记忆。
2. 精确定位某个证据。

SSM 更像持续记忆。

Attention 更像精确检索。

卷积更像局部模式提取。

因此长上下文模型可能需要三者组合，而不是单一机制。

### 推理系统视角

从 serving 看，不同模块的瓶颈不同。

Attention 的瓶颈常在 KV Cache 和显存带宽。

SSM 的瓶颈可能在 scan kernel、状态管理和算子融合。

卷积的瓶颈可能在 kernel 实现和序列长度适配。

混合架构必须考虑实际硬件吞吐，而不是只看论文复杂度。

### 评估混合架构

评估混合架构要覆盖：

1. perplexity 或 validation loss。
2. 常规 benchmark。
3. 长上下文检索。
4. lost-in-the-middle。
5. 代码和数学推理。
6. 推理 latency。
7. tokens/s。
8. 显存占用。
9. 训练稳定性。

如果只看一个指标，很容易误判。

### 面试中怎么讲

如果面试官问“为什么要做混合架构”，可以这样回答：

```text
Attention、SSM 和卷积各有优势。Attention 擅长显式 token-to-token 检索，但长序列成本高；SSM 更适合线性复杂度的长程状态建模，但精确读取上下文 token 不如 attention 直接；卷积擅长局部模式。混合架构的目标是把它们组合起来，在长上下文效率、精确检索能力和局部建模之间取得更好的 trade-off。
```

如果面试官问“混合架构怎么评估”，可以这样回答：

```text
不能只看理论复杂度或单个 benchmark。需要同时看模型质量、长上下文检索、推理能力、训练稳定性、推理延迟、吞吐、显存和 kernel 支持。特别是替代 attention 的模块，需要验证它是否还能完成需要精确上下文读取的任务。
```

### 真实项目中的坑

#### 坑 1：只看复杂度，不看任务能力

线性复杂度模型如果不能完成精确检索，对很多 LLM 场景就不够。

#### 坑 2：混合比例靠拍脑袋

attention、SSM、卷积比例需要通过 ablation 和任务评估决定。

#### 坑 3：忽略 kernel 成熟度

新架构如果没有高效 kernel，理论优势可能无法落地。

#### 坑 4：只测短 benchmark

混合架构的优势常在长序列和流式场景，短 benchmark 不足以说明问题。

#### 坑 5：部署复杂度估计不足

不同算子混合会增加推理引擎、并行策略和监控复杂度。

### 面试官会怎么问

#### 问题 1：Attention、SSM、卷积分别适合什么？

回答框架：

1. Attention 适合精确上下文读取。
2. SSM 适合高效长程状态建模。
3. 卷积适合局部模式。
4. 混合架构试图组合三者优势。

#### 问题 2：混合架构为什么可能比纯 SSM 更好？

回答框架：

1. 纯 SSM 历史信息被压缩到状态。
2. 精确检索某个历史 token 可能困难。
3. 加入 attention 可以补充显式读取能力。
4. 因此混合更平衡。

#### 问题 3：混合架构有哪些工程难点？

回答框架：

1. 算子种类更多。
2. kernel 支持复杂。
3. 并行和 serving 适配更难。
4. 训练 recipe 和稳定性需要重新调。

#### 问题 4：如何做 ablation？

回答框架：

1. 固定参数量和计算预算。
2. 改变 attention/SSM/卷积比例。
3. 比较短任务、长任务和推理指标。
4. 看质量、速度、显存和稳定性。

### 常见误区

1. 误区：混合架构一定比纯 Transformer 好。
   纠正：要看任务、规模、训练和硬件实现。

2. 误区：复杂度低就是架构先进。
   纠正：还要看精确检索、推理能力和生态成熟度。

3. 误区：卷积只适合图像。
   纠正：卷积也能建模文本、语音和多模态中的局部模式。

4. 误区：新模块可以直接插入现有 LLM。
   纠正：通常需要重新训练、调参和系统适配。

### 小练习

1. 比较 attention、SSM 和卷积的信息流。
2. 设计一个 attention + SSM 的混合层级方案。
3. 说明混合架构适合哪些长上下文任务。
4. 设计一组评估指标比较混合架构和 dense Transformer。
5. 用 2 分钟回答“为什么混合架构可能是现实路线”。

### 本讲总结

本讲最重要的结论：

1. 混合架构不是简单堆模块，而是在不同序列建模机制之间做 trade-off。
2. Attention 擅长精确检索，SSM 擅长长程状态建模，卷积擅长局部模式。
3. 混合架构可能改善长上下文效率，同时保留部分 attention 表达能力。
4. 它的代价是训练、kernel、推理系统和调参复杂度更高。
5. 评价新架构必须同时看质量、长上下文能力、速度、显存、稳定性和生态成熟度。

## 第 24 讲：模型宽度、深度、head 数和 hidden size 的设计

### 本讲目标

设计一个 Transformer 不是只决定参数量，还要决定层数、隐藏维度、head 数、head dimension、FFN 中间维度、KV head 数等结构超参数。本讲讨论这些设计变量如何影响模型能力、训练稳定性、推理效率和硬件利用率。

你需要掌握：

1. 深度和宽度分别影响什么。
2. `d_model`、`num_heads`、`head_dim` 的关系。
3. FFN hidden size 为什么常是 `d_model` 的数倍。
4. head 数不是越多越好。
5. 架构尺寸如何影响参数量和 FLOPs。
6. 为什么模型设计要考虑硬件友好性。

### 关键结构超参数

一个 decoder-only Transformer 常见结构参数包括：

1. `num_layers`：层数，也叫深度。
2. `d_model`：hidden size，也叫模型宽度。
3. `num_heads`：attention query head 数。
4. `head_dim`：每个 head 的维度。
5. `d_ff` 或 `intermediate_size`：FFN 中间维度。
6. `num_kv_heads`：GQA/MQA 中的 KV head 数。
7. vocabulary size。
8. context length。

这些参数共同决定模型的参数量、计算量、显存和推理速度。

### d_model、head 数和 head_dim

通常有：

```text
d_model = num_heads * head_dim
```

例如：

```text
d_model = 4096
num_heads = 32
head_dim = 128
```

head 数决定有多少个 attention 子空间。

head_dim 决定每个 head 的表示容量。

如果 head 太少，每层的信息路由子空间可能不足。

如果 head 太多，head_dim 过小，每个 head 的表达能力可能不足，同时 kernel 和并行效率也会受影响。

### 深度和宽度的区别

加深模型意味着增加层数。

加宽模型意味着增加 hidden size。

深度更影响逐层组合和抽象能力。

宽度更影响每层表示容量和并行计算规模。

粗略来说：

```text
深度增加 sequential computation
宽度增加 per-layer capacity
```

深模型可能更难训练，需要更好的 normalization、初始化和 residual scaling。

宽模型计算更密集，可能更适合 GPU 矩阵乘法，但显存和参数量增长快。

### 参数量粗略估算

一个 Transformer block 中，attention 投影参数约为：

```text
attention_params = 4 * d_model * d_model
```

标准 FFN 如果 `d_ff = 4 * d_model`，参数约为：

```text
ffn_params = 8 * d_model * d_model
```

所以每层大约：

```text
block_params = 12 * d_model * d_model
```

总参数量粗略为：

```text
total_block_params = num_layers * 12 * d_model * d_model
```

这只是估算，实际还要考虑 embedding、norm、bias、GQA、SwiGLU 和 vocab size。

### FFN hidden size

传统 FFN 常用：

```text
d_ff = 4 * d_model
```

SwiGLU 结构因为有三组投影，常用接近：

```text
intermediate_size = 8/3 * d_model
```

并做硬件友好的取整。

这个 `8/3` 是为了让三投影的 SwiGLU MLP 参数量接近普通 `4 * d_model` FFN 的常见经验值，不是所有模型都必须严格等于它。真实配置还会按 tensor core、tensor parallel 和框架 kernel 做取整。

FFN hidden size 决定每个 token 的非线性特征变换容量。

它太小会限制表达能力，太大则增加参数、FLOPs 和推理延迟。

### Head 数怎么选

head 数需要和 `d_model`、`head_dim` 一起看。

常见 head_dim 是 64、80、96、128 等硬件友好的值。

如果 `d_model=4096`，`head_dim=128`，则 `num_heads=32`。

设计时要考虑：

1. head_dim 是否足够表达。
2. num_heads 是否有足够路由子空间。
3. 是否适配 FlashAttention 和推理 kernel。
4. 是否使用 GQA 降低 KV heads。
5. tensor parallel 切分是否方便。

### 深度太深的问题

增加层数可以提升能力，但会带来：

1. 训练更难稳定。
2. 激活显存增加。
3. 推理延迟增加。
4. pipeline parallel 更复杂。
5. gradient checkpointing 成本增加。

深层模型通常需要 Pre-Norm、RMSNorm、合适初始化、warmup 和 residual scaling。

### 宽度太宽的问题

增加 `d_model` 会让参数量按平方增长。

如果 `d_model` 翻倍，每层主要矩阵参数量约增加 4 倍。

宽模型的优点是矩阵乘法更大，GPU 利用率可能好。

缺点是：

1. 参数显存增加。
2. optimizer state 增加。
3. 每 token 计算增加。
4. 通信成本增加。

所以宽度设计要和硬件预算匹配。

### Scaling Law 视角

模型结构设计不能脱离数据和计算预算。

在固定计算预算下，模型太大而训练 token 太少，会欠训练。

模型太小而数据太多，又无法充分吸收数据。

因此架构尺寸要和：

1. 参数量。
2. 训练 token 数。
3. 计算预算。
4. 数据质量。
5. 目标任务。

一起设计。

这就是 compute-optimal training 的基本思想。

Chinchilla 这类 scaling 结果提醒我们：结构尺寸不是孤立变量。固定 compute 下，盲目增大参数但训练 token 不足会欠训练；更小但训练更充分的模型可能在质量和推理成本上更好。LLaMA 类模型也体现了“用更多公开 token 充分训练相对高效模型”的路线。

### 一个可运行参数量估算 demo

下面这个 demo 粗略估算 decoder-only Transformer block 的参数量，并比较普通 GeLU FFN 和 SwiGLU MLP 的 intermediate size 取整效果。

```python
def round_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def estimate_block_params(d_model, d_ff, mlp_type="gelu"):
    attention = 4 * d_model * d_model
    if mlp_type == "gelu":
        mlp = 2 * d_model * d_ff
    elif mlp_type == "swiglu":
        mlp = 3 * d_model * d_ff
    else:
        raise ValueError(f"unknown mlp_type: {mlp_type}")
    return attention + mlp


d_model = 4096
num_layers = 32
gelu_d_ff = 4 * d_model
swiglu_d_ff = round_to_multiple(int(8 * d_model / 3), 256)

gelu_block = estimate_block_params(d_model, gelu_d_ff, "gelu")
swiglu_block = estimate_block_params(d_model, swiglu_d_ff, "swiglu")

print("GeLU d_ff:", gelu_d_ff)
print("SwiGLU intermediate_size:", swiglu_d_ff)
print("GeLU block params:", round(gelu_block / 1e6, 2), "M")
print("SwiGLU block params:", round(swiglu_block / 1e6, 2), "M")
print("GeLU 32-layer block params:", round(gelu_block * num_layers / 1e9, 2), "B")
print("SwiGLU 32-layer block params:", round(swiglu_block * num_layers / 1e9, 2), "B")
```

这个估算不包含 embedding、LM head、norm、bias、GQA 对 K/V projection 的影响和 vocab size，因此只能作为架构直觉工具，不能替代真实配置统计。

### 硬件友好性

真实模型设计必须考虑硬件。

常见硬件约束包括：

1. hidden size 是否能被 tensor parallel size 整除。
2. head 数是否能被并行度整除。
3. intermediate size 是否适合 tensor core。
4. vocab size 是否需要 padding。
5. sequence length 是否适合 attention kernel。
6. GQA 的 KV heads 是否适合推理引擎。

很多看似奇怪的维度，其实是为了硬件效率和并行切分。

### 架构设计的 trade-off

常见 trade-off：

1. 更深：表达更强，但延迟更高、训练更难。
2. 更宽：容量更大，但参数和 FLOPs 按平方增长。
3. 更多 heads：路由子空间更多，但 head_dim 可能变小。
4. 更大 FFN：非线性容量更强，但计算更贵。
5. GQA：推理更省，但可能影响质量。
6. 更长上下文：能力更强，但 attention 和 KV Cache 更贵。

好的架构设计是在这些约束中取平衡。

### 面试中怎么讲

如果面试官问“如何设计一个 Transformer 的宽度和深度”，可以这样回答：

```text
我会先确定目标参数量、训练 token 数、计算预算和推理约束。深度影响逐层组合能力，但增加顺序计算和训练稳定性压力；宽度影响每层表示容量，主要矩阵参数和 FLOPs 近似按 `d_model * d_model` 增长。head 数要和 d_model、head_dim、GQA、kernel 和并行切分一起设计。FFN intermediate size 决定 MLP 容量，SwiGLU 下通常不是简单 4 倍，而是考虑参数量和硬件取整。
```

### 真实项目中的坑

#### 坑 1：只按参数量设计模型

同样参数量，不同深宽比、head 数和 FFN 比例会有不同速度、稳定性和效果。

#### 坑 2：忽略推理约束

训练时能跑不代表线上 serving 成本可接受。

#### 坑 3：head 数随便定

head 数必须和 head_dim、kernel、GQA 和 tensor parallel 一起考虑。

#### 坑 4：不考虑硬件取整

不友好的 hidden size 或 intermediate size 会浪费算力。

#### 坑 5：只追求长上下文

context length 变大后，attention、KV Cache 和评估都会变复杂。

### 面试官会怎么问

#### 问题 1：d_model、num_heads、head_dim 的关系是什么？

回答框架：

1. 通常 `d_model = num_heads * head_dim`。
2. num_heads 决定 attention 子空间数量。
3. head_dim 决定每个 head 的容量。
4. 设计时还要考虑 kernel 和并行切分。

#### 问题 2：深模型和宽模型怎么取舍？

回答框架：

1. 深度增强逐层组合能力。
2. 宽度增强每层表示容量。
3. 深度增加顺序延迟和稳定性压力。
4. 宽度让参数和 FLOPs 快速增长。

#### 问题 3：为什么 FFN hidden size 常比 d_model 大？

回答框架：

1. FFN 先扩展维度再压回。
2. 扩展维度提供更丰富非线性特征。
3. 标准 FFN 常用 4 倍。
4. SwiGLU 常用约 8/3 倍以控制参数量。

#### 问题 4：为什么架构尺寸要考虑硬件？

回答框架：

1. 矩阵维度影响 tensor core 利用率。
2. hidden size/head 数影响并行切分。
3. GQA/KV heads 影响推理引擎效率。
4. 不友好维度会浪费显存和算力。

### 常见误区

1. 误区：参数量一样，模型能力就差不多。
   纠正：深宽比、FFN 比例、head 数和训练 recipe 都会影响能力。

2. 误区：head 越多越好。
   纠正：head 太多会让 head_dim 变小，也可能影响效率。

3. 误区：越深一定越好。
   纠正：深度增加训练稳定性和推理延迟压力。

4. 误区：架构设计只看论文指标。
   纠正：还要看硬件、serving、并行和成本。

### 小练习

1. 估算一个 `num_layers=32, d_model=4096, d_ff=16384` 模型的 block 参数量。
2. 比较加深和加宽对参数量、延迟和稳定性的影响。
3. 设计一组 `d_model、num_heads、head_dim`，并说明理由。
4. 解释为什么 SwiGLU 的 intermediate size 常接近 `8/3 d_model`。
5. 用 2 分钟回答“如何在固定计算预算下设计模型尺寸”。

### 本讲总结

本讲最重要的结论：

1. Transformer 尺寸设计包括深度、宽度、head 数、head_dim、FFN hidden size 和 KV head 数。
2. 参数量大致随层数线性增长、随 `d_model * d_model` 增长。
3. 深度、宽度、head 数和 FFN 比例都有不同 trade-off。
4. 模型尺寸必须和数据规模、计算预算、推理约束和硬件效率一起设计。
5. 真实架构设计不是追求单个指标最大，而是在能力、成本、稳定性和部署之间平衡。

## 第 25 讲：现代 LLM 架构案例拆解

### 本讲目标

本讲是第二部分“Transformer 架构进阶”的收尾。前面我们分别讲了 attention、归一化、FFN、RoPE、GQA、MoE、SSM 和架构尺寸设计。本讲把这些组件串起来，用现代 LLM 的典型架构特征做一次系统拆解，帮助你在面试中从整体视角分析一个模型，而不是只背单个模块。

你需要掌握：

1. 现代 decoder-only LLM 的常见组件组合。
2. LLaMA 类架构为什么有代表性。
3. GPT 类、MoE 类、长上下文类模型分别关注什么。
4. 如何从配置文件判断模型架构。
5. 如何面试式拆解一个未知 LLM 架构。
6. 第二部分的完整知识图谱。

### 现代 LLM 的典型组合

很多现代 decoder-only LLM 会采用类似组合：

```text
Decoder-only
Causal Self-Attention
Pre-Norm / RMSNorm
RoPE
SwiGLU MLP
GQA 或 MQA
BF16 训练
FlashAttention 类高效 attention
```

这些组件不是随机拼起来的。

它们分别解决不同问题：

1. decoder-only 统一生成范式。
2. RMSNorm 提升训练稳定性和效率。
3. RoPE 提供相对位置建模。
4. SwiGLU 增强 MLP 表达能力。
5. GQA 降低推理 KV Cache。
6. FlashAttention 降低 attention IO 成本。

按资料边界看，LLaMA 风格架构的代表性不在于某个单一 trick，而在于把 decoder-only、RMSNorm、RoPE、SwiGLU 等选择组合成了稳定、可扩展、易复现的基线。GQA/MQA 这类改动主要服务推理效率，RoPE 则服务位置建模和一定长度泛化；它们都需要结合训练数据、训练长度和评估结果理解。

### LLaMA 类架构为什么有代表性

LLaMA 类模型的代表性在于它把现代 decoder-only LLM 的很多常见选择组合到一起。

典型特征包括：

1. Decoder-only。
2. Pre-Norm。
3. RMSNorm。
4. RoPE。
5. SwiGLU。
6. 无 bias 或较少 bias。
7. GQA 在较新版本中常见。

这套设计简单、稳定、可扩展，也被很多开源模型采用或借鉴。

### GPT 类模型的核心特征

GPT 类模型的核心不是某个单一 trick，而是：

```text
decoder-only + next-token prediction + 大规模数据和算力扩展
```

架构层面，GPT 类模型强调：

1. 自回归生成。
2. causal attention。
3. 大规模预训练。
4. prompt 到 continuation 的统一任务形式。
5. 后训练对齐成助手行为。

很多现代 LLM 都可以看成 GPT 范式的扩展。

### MoE 类模型的架构重点

MoE 类模型的重点是用 sparse activation 增加总参数量。

典型结构是：

```text
部分或全部 FFN 层替换为 MoE experts
router 为 token 选择 top-k experts
```

它关注的问题是：

1. 总容量扩展。
2. 每 token 计算控制。
3. expert 负载均衡。
4. all-to-all 通信。
5. 推理 serving 调度。

面试中看到 MoE，要立刻想到 router、load balancing、capacity、通信和 activated parameters。

### 长上下文模型的架构重点

长上下文模型不只是把 `max_position_embeddings` 改大。

它通常需要同时考虑：

1. 位置编码外推，例如 RoPE scaling 或 ALiBi。
2. attention 显存和计算优化。
3. KV Cache 管理，例如 GQA/MQA、PagedAttention。
4. 长序列继续训练。
5. long-context evaluation。
6. RAG 或上下文压缩。

如果一个模型宣称支持 128K context，面试中要追问：

```text
它是否真的能利用远处信息？是否评估 lost-in-the-middle？推理成本如何？
```

### 如何读模型配置

看到一个 LLM config，可以重点看：

1. `num_hidden_layers`：层数。
2. `hidden_size`：模型宽度。
3. `intermediate_size`：FFN 宽度。
4. `num_attention_heads`：query head 数。
5. `num_key_value_heads`：KV head 数，判断是否 GQA/MQA。
6. `max_position_embeddings` 或 context length。
7. `rope_theta`、rope scaling 配置。
8. `hidden_act`：GeLU、SiLU、SwiGLU 等。
9. `rms_norm_eps` 或 norm 类型。
10. vocab size。

这些字段能帮助你快速判断模型架构风格。

### 架构拆解模板

面试中拆一个 LLM 架构，可以按下面顺序：

1. 先判断整体范式：decoder-only、encoder-only 还是 encoder-decoder。
2. 看 attention：MHA、GQA、MQA、稀疏 attention 还是混合结构。
3. 看位置编码：RoPE、ALiBi、绝对位置还是相对 bias。
4. 看 norm：LayerNorm、RMSNorm、Pre-LN 还是 Post-LN。
5. 看 MLP：GeLU FFN、SwiGLU、MoE。
6. 看尺寸：层数、hidden size、head 数、intermediate size。
7. 看推理优化：KV Cache、GQA、FlashAttention、量化兼容。
8. 看训练和后训练：预训练目标、SFT、偏好优化。

这套模板比零散背组件更适合系统设计和架构面试。

### 案例：一个 LLaMA 风格模型

假设一个模型配置是：

```text
decoder-only
32 layers
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
intermediate_size = 11008
RMSNorm
RoPE
SwiGLU
```

你可以分析：

1. 这是 decoder-only 通用生成模型。
2. `num_kv_heads < num_attention_heads`，说明使用 GQA。
3. RMSNorm + RoPE + SwiGLU 是现代 LLaMA 风格组合。
4. intermediate size 接近 `8/3 * hidden_size`，符合 SwiGLU 参数量控制。
5. 架构重点是训练稳定、推理效率和通用生成能力平衡。

### 一个可运行 config 解析 demo

下面这个 demo 用一个 LLaMA 风格配置，自动判断 attention 类型、KV Cache 节省比例、MLP 风格和基础架构标签。真实项目里可以把同样逻辑用在 Hugging Face `config.json` 上。

```python
def describe_llm_config(config):
    query_heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", query_heads)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]

    if kv_heads == query_heads:
        attention_type = "MHA"
        kv_cache_ratio = 1.0
    elif kv_heads == 1:
        attention_type = "MQA"
        kv_cache_ratio = 1 / query_heads
    else:
        attention_type = "GQA"
        kv_cache_ratio = kv_heads / query_heads

    mlp_ratio = intermediate_size / hidden_size
    mlp_type = "SwiGLU-like" if 2.0 < mlp_ratio < 3.5 else "standard-or-custom"
    norm_type = "RMSNorm" if "rms_norm_eps" in config else "LayerNorm-or-custom"
    position_type = "RoPE" if "rope_theta" in config else "unknown-or-absolute"

    return {
        "layers": config["num_hidden_layers"],
        "hidden_size": hidden_size,
        "attention_type": attention_type,
        "kv_cache_ratio_vs_mha": round(kv_cache_ratio, 3),
        "mlp_ratio": round(mlp_ratio, 3),
        "mlp_type": mlp_type,
        "norm_type": norm_type,
        "position_type": position_type,
    }


config = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 11008,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}

summary = describe_llm_config(config)
for key, value in summary.items():
    print(f"{key}: {value}")
```

典型输出会显示该配置是 32 层、4096 hidden size、GQA、KV Cache head 维度约为 MHA 的 1/4、RMSNorm、RoPE，并且 MLP ratio 接近 SwiGLU 常见范围。这个 demo 只根据配置做架构推断，不能替代真实模型代码、训练 recipe 和 benchmark 结果。

### 案例：一个 MoE 风格模型

假设模型配置包含：

```text
num_experts = 64
top_k = 2
router_aux_loss
expert_capacity
```

你应该立刻分析：

1. 这是 sparse MoE。
2. 每 token 只激活 2 个 experts。
3. 总参数量和 activated parameters 要分开看。
4. 训练要关注 load balancing 和 expert collapse。
5. 推理要关注 all-to-all、expert parallelism 和 tail latency。

### 案例：一个长上下文模型

如果模型宣称支持 128K context，你应该看：

1. 用什么位置编码或 scaling。
2. 训练是否包含长序列。
3. attention 是否使用 FlashAttention、sparse attention 或 sliding window。
4. KV Cache 是否用 GQA/MQA 或分页管理。
5. 长上下文 benchmark 是否包含真实推理，而不只是 needle retrieval。

真正的长上下文能力不是只看最大窗口，而是看能否可靠利用远处信息。

### 第二部分知识图谱

第二部分可以串成一条架构主线：

```text
Self-Attention -> 复杂度瓶颈 -> Norm/Residual -> FFN/SwiGLU -> 位置编码 -> GQA/KV Cache -> MoE -> 架构范式 -> SSM/混合架构 -> 尺寸设计 -> 案例拆解
```

对应能力是：

1. 能解释 Transformer 信息流。
2. 能分析训练稳定性设计。
3. 能理解推理效率瓶颈。
4. 能比较 dense、MoE、SSM 和混合架构。
5. 能从配置推断模型架构。
6. 能做系统化面试表达。

### 面试中怎么讲

如果面试官问“如何拆解一个现代 LLM 架构”，可以这样回答：

```text
我会先看整体范式是不是 decoder-only，然后看 attention 结构是 MHA、GQA 还是 MQA，位置编码是 RoPE 还是 ALiBi，norm 是 LayerNorm 还是 RMSNorm，MLP 是 GeLU FFN、SwiGLU 还是 MoE。接着看层数、hidden size、head 数和 intermediate size 估算参数和计算，再结合 KV Cache、FlashAttention、量化和 serving 约束判断推理效率。最后还要看训练目标、后训练方式和评估结果，因为架构只是能力的一部分。
```

### 真实项目中的坑

#### 坑 1：只看模型名字

同一个系列不同版本可能在 GQA、context length、MoE、RoPE scaling 上差异很大。

#### 坑 2：只看参数量

参数量不能说明 activated parameters、推理成本、长上下文能力和后训练质量。

#### 坑 3：忽略配置细节

`num_key_value_heads`、`intermediate_size`、`rope_theta` 这类字段往往能暴露重要架构选择。

#### 坑 4：把架构和模型能力等同

能力还取决于数据、训练规模、后训练、评估和产品系统。

#### 坑 5：只讲优点，不讲 trade-off

现代 LLM 架构设计本质上是在质量、成本、稳定性和部署之间平衡。

### 面试官会怎么问

#### 问题 1：现代 LLM 常见架构组件有哪些？

回答框架：

1. decoder-only。
2. causal attention。
3. RMSNorm/Pre-Norm。
4. RoPE。
5. SwiGLU。
6. GQA/MQA。
7. FlashAttention 类优化。

#### 问题 2：如何从 config 判断是否用了 GQA？

回答框架：

1. 看 `num_attention_heads`。
2. 看 `num_key_value_heads`。
3. 如果二者相等，是 MHA。
4. 如果 KV heads 更少，是 GQA 或 MQA。

#### 问题 3：为什么 LLaMA 风格架构影响大？

回答框架：

1. 结构简洁稳定。
2. 使用 RMSNorm、RoPE、SwiGLU 等现代组件。
3. 易扩展、易复现。
4. 成为很多开源 LLM 的参考范式。

#### 问题 4：评估一个新架构要看什么？

回答框架：

1. 模型质量。
2. 训练稳定性。
3. 长上下文能力。
4. 推理速度和显存。
5. 硬件 kernel 支持。
6. 大规模验证和生态成熟度。

### 常见误区

1. 误区：架构先进就一定能力强。
   纠正：数据、训练、后训练和评估同样关键。

2. 误区：所有现代 LLM 架构都差不多。
   纠正：GQA、MoE、位置编码、context length 和 MLP 设计会带来显著差异。

3. 误区：配置文件只是工程细节。
   纠正：配置字段能反映核心架构选择。

4. 误区：只要支持长上下文就是好架构。
   纠正：还要看能否利用长上下文，以及推理成本是否可接受。

### 小练习

1. 找一个开源 LLM config，标注它的 norm、position、attention 和 MLP 设计。
2. 判断一个模型是否使用 GQA，并估算 KV Cache 节省比例。
3. 比较一个 dense LLM 和 MoE LLM 的 activated parameters。
4. 设计一个架构拆解面试回答模板。
5. 用 2 分钟回答“如何评价一个现代 LLM 架构”。

### 本讲总结

本讲最重要的结论：

1. 现代 LLM 架构通常是多种组件的组合，而不是单一 trick。
2. LLaMA 风格架构代表了 decoder-only、RMSNorm、RoPE、SwiGLU、GQA 等现代设计的组合。
3. MoE、长上下文模型、SSM 和混合架构分别针对容量、上下文长度和效率问题。
4. 拆解模型架构要从整体范式、attention、position、norm、MLP、尺寸和推理优化逐层分析。
5. 第二部分的核心能力是把 Transformer 架构组件和训练、推理、部署 trade-off 连起来。
