# 第二部分：Transformer 组件实战

6. 实现 Token Embedding 与 Positional Embedding
## 第 7 讲：从零实现 Scaled Dot-Product Attention

### 本讲目标

学完本讲，你应该能做到六件事：

1. 用 PyTorch 从零实现 scaled dot-product attention。
2. 说清楚 Query、Key、Value 的 shape 和含义。
3. 理解为什么 attention score 要除以 `sqrt(d_k)`。
4. 正确处理 attention mask。
5. 能解释 attention weights 和输出的 shape。
6. 能把本讲代码连接到 Multi-Head Attention 和 Transformer Block。

前面第一部分完成了 PyTorch 基础训练闭环。

从本讲开始，我们进入 Transformer 组件实战。

Transformer 最核心的操作是 attention。

而 attention 的最小核心就是：

```text
Scaled Dot-Product Attention
```

公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

这行公式看起来短，但面试中经常能问出很多细节。

本讲目标就是把它写成可运行代码，并彻底搞清楚每个 shape。

---

### 一、Q、K、V 分别是什么

在 self-attention 中，每个 token 会产生三个向量：

1. Query，简称 Q。
2. Key，简称 K。
3. Value，简称 V。

直觉上可以这样理解：

```text
Query：我想找什么信息。
Key：我能提供什么索引。
Value：如果你关注我，我实际提供什么内容。
```

对一个序列来说，每个 token 都会拿自己的 Query 去和所有 token 的 Key 做相似度。

相似度越高，说明当前 token 越应该关注那个 token。

然后用这个关注权重对所有 Value 加权求和。

最终得到当前 token 的新表示。

---

### 二、最常见的 shape 约定

本讲先使用单头 attention。

输入 shape 约定为：

```text
q: [batch, seq_len, d_k]
k: [batch, seq_len, d_k]
v: [batch, seq_len, d_v]
```

通常在 Transformer 中：

```text
d_k = d_v = head_dim
```

为了简单，设：

```text
batch = 2
seq_len = 4
d_k = 8
d_v = 8
```

那么：

```text
q: [2, 4, 8]
k: [2, 4, 8]
v: [2, 4, 8]
```

attention 输出 shape 是：

```text
out: [2, 4, 8]
```

也就是每个 token 输出一个新的表示。

---

### 三、第一步：计算 attention scores

公式第一部分是：

```text
QK^T
```

代码写法：

```python
scores = q @ k.transpose(-2, -1)
```

为什么要 `transpose(-2, -1)`？

因为 `q` shape 是：

```text
[batch, seq_len, d_k]
```

`k` shape 是：

```text
[batch, seq_len, d_k]
```

我们希望每个 query token 和每个 key token 做点积。

所以要把 k 的最后两个维度转置成：

```text
[batch, d_k, seq_len]
```

矩阵乘法结果：

```text
[batch, seq_len, d_k] @ [batch, d_k, seq_len]
= [batch, seq_len, seq_len]
```

也就是说：

```text
scores[i, a, b] 表示第 i 个 batch 中，第 a 个 token 对第 b 个 token 的关注分数。
```

---

### 四、为什么要除以 sqrt(d_k)

如果不缩放，点积值会随着维度 `d_k` 增大而变大。

假设 q 和 k 的每个维度均值为 0、方差为 1。

点积是：

```text
q · k = sum(q_i * k_i)
```

这个和的方差大约会随 `d_k` 增大。

当 `d_k` 很大时，scores 的数值会很大。

softmax 遇到很大的正负值，会变得非常尖锐。

例如：

```text
softmax([20, 1, -5]) 几乎全压到第一个位置
```

这会导致梯度很小，训练不稳定。

所以缩放：

```text
scores = QK^T / sqrt(d_k)
```

代码：

```python
scores = scores / math.sqrt(q.size(-1))
```

面试中可以这样回答：

```text
除以 sqrt(d_k) 是为了控制点积方差，避免维度变大后 attention logits 过大，softmax 过于饱和，从而影响梯度和训练稳定性。
```

---

### 五、第二步：softmax 得到 attention weights

scores shape 是：

```text
[batch, seq_len, seq_len]
```

我们要在最后一个维度做 softmax：

```python
attn_weights = torch.softmax(scores, dim=-1)
```

含义是：

```text
对每个 query token，它对所有 key token 的关注权重和为 1。
```

shape 仍然是：

```text
[batch, seq_len, seq_len]
```

例如第 2 个 token 对所有 token 的 attention 权重可能是：

```text
[0.1, 0.2, 0.6, 0.1]
```

表示它最关注第 3 个 token。

---

### 六、第三步：加权求和 Value

最后一步：

```python
out = attn_weights @ v
```

shape 是：

```text
[batch, seq_len, seq_len] @ [batch, seq_len, d_v]
= [batch, seq_len, d_v]
```

这表示每个 token 都得到一个新的表示。

这个新表示是所有 Value 的加权平均。

权重由 Query-Key 相似度决定。

完整流程就是：

```text
Q 和 K 算相似度 -> softmax 成权重 -> 用权重汇聚 V
```

---

### 七、完整实现：无 mask 版本

```python
import math
import torch


def scaled_dot_product_attention(q, k, v):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights


torch.manual_seed(42)

batch = 2
seq_len = 4
d_k = 8

q = torch.randn(batch, seq_len, d_k)
k = torch.randn(batch, seq_len, d_k)
v = torch.randn(batch, seq_len, d_k)

out, attn_weights = scaled_dot_product_attention(q, k, v)

print(out.shape)          # torch.Size([2, 4, 8])
print(attn_weights.shape) # torch.Size([2, 4, 4])
print(attn_weights.sum(dim=-1))
```

最后一行应该接近全 1。

因为每个 query 对所有 key 的 attention 权重和为 1。

---

### 八、加入 mask

实际 Transformer 中经常需要 mask。

常见 mask 有两类：

1. Padding mask：不关注 padding token。
2. Causal mask：不关注未来 token。

mask 的核心做法是：

```text
把不允许关注的位置的 score 设为一个极小值。
```

例如：

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

这样 softmax 后，这些位置的概率会接近 0。

完整版本：

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights
```

mask shape 要能 broadcast 到 scores：

```text
scores: [batch, seq_len, seq_len]
mask:   [batch, seq_len, seq_len]
```

或者：

```text
mask: [1, seq_len, seq_len]
```

用于所有 batch 共用一个 causal mask。

---

### 九、实现一个 causal mask

Causal mask 用于自回归语言模型。

它保证第 t 个 token 只能看见自己和之前的 token，不能看未来。

```python
def make_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
```

例如 `seq_len = 4`：

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

含义：

1. 第 0 个 token 只能看第 0 个 token。
2. 第 1 个 token 可以看第 0、1 个 token。
3. 第 2 个 token 可以看第 0、1、2 个 token。
4. 第 3 个 token 可以看所有历史 token。

测试代码：

```python
mask = make_causal_mask(seq_len)
out, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

print(mask)
print(attn_weights[0])
```

你应该看到未来位置的 attention 权重接近 0。

---

### 十、和 PyTorch 内置函数对齐

PyTorch 提供了：

```python
torch.nn.functional.scaled_dot_product_attention
```

可以用于高效 attention。

简单对齐：

```python
import torch.nn.functional as F


manual_out, _ = scaled_dot_product_attention(q, k, v)
torch_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

print(torch.allclose(manual_out, torch_out, atol=1e-6))
```

注意不同 PyTorch 版本和后端可能会使用 FlashAttention 或 memory-efficient attention。

它们数学目标相同，但实现更高效。

本讲的手写版本适合理解原理。

真实大模型训练中会使用更优化的 kernel。

---

### 十一、常见工程坑

#### 坑 1：忘记除以 `sqrt(d_k)`

训练可能更不稳定，softmax 更容易饱和。

#### 坑 2：softmax 维度写错

应该在 key 维度做 softmax：

```python
torch.softmax(scores, dim=-1)
```

如果写成 `dim=1`，语义就错了。

#### 坑 3：mask 方向反了

有的代码约定 mask 中 `1` 表示保留。

有的代码约定 `True` 表示要 mask 掉。

一定要看清接口约定。

#### 坑 4：mask shape 不能 broadcast

scores 是：

```text
[batch, seq_len, seq_len]
```

mask 要么同形状，要么能 broadcast。

#### 坑 5：全 mask 导致 NaN

如果某一行所有位置都被 mask，softmax 会对全 `-inf` 做归一化，可能产生 NaN。

要保证每个 query 至少有一个可见 key。

#### 坑 6：用 `-inf` 在低精度下出问题

bf16/fp16 中有时用很小负数更稳，例如：

```python
torch.finfo(scores.dtype).min
```

或框架内置 attention mask 处理。

---

### 十二、面试怎么讲 Scaled Dot-Product Attention

如果面试官问“attention 怎么实现”，可以这样回答：

```text
给定 Q、K、V，先计算 Q 和 K 的点积得到 [batch, seq_len, seq_len] 的 attention scores，再除以 sqrt(d_k) 控制数值尺度。如果有 mask，就把不可见位置设为很小的负数。然后在最后一维做 softmax 得到 attention weights，最后用 weights 乘 V 得到每个 token 的加权表示。
```

如果追问“为什么要 scale”，可以回答：

```text
因为 QK 点积的方差会随 d_k 增大而增大，导致 softmax 输入过大、分布过尖、梯度变小。除以 sqrt(d_k) 可以稳定 logits 尺度，让训练更稳定。
```

如果追问“causal mask 做什么”，可以回答：

```text
Causal mask 用于自回归语言模型，保证当前位置只能关注自己和历史 token，不能看到未来 token，防止训练时信息泄漏。
```

---

### 十三、小练习

#### 练习 1

打印 `scores`、`attn_weights` 和 `out` 的 shape。

解释每个维度的含义。

#### 练习 2

把 `softmax(dim=-1)` 改成 `softmax(dim=1)`。

观察 attention weights 的归一化维度发生了什么变化。

#### 练习 3

实现 padding mask。

假设 batch 中某些 token 是 padding，让其他 token 不关注 padding 位置。

#### 练习 4

构造 `seq_len=5` 的 causal mask，手动确认每一行可见位置是否正确。

#### 练习 5

用 `F.scaled_dot_product_attention` 和手写版本对比输出是否一致。

---

### 本讲总结

这一讲从零实现了 Scaled Dot-Product Attention。

核心结论如下：

1. Attention 的核心公式是 `softmax(QK^T / sqrt(d_k))V`。
2. `QK^T` 产生每个 token 对其他 token 的关注分数。
3. 除以 `sqrt(d_k)` 是为了控制点积尺度，避免 softmax 饱和。
4. softmax 应该在 key 维度上做，让每个 query 的关注权重和为 1。
5. mask 通过把不可见位置设为极小值，让 softmax 后概率接近 0。
6. causal mask 防止自回归模型看到未来 token。
7. 手写 attention 适合理解原理，真实大模型训练通常使用 FlashAttention 等高效实现。

下一讲，我们实现 Multi-Head Attention。

也就是把单头 attention 扩展成多个子空间并行关注不同关系。

## 第 8 讲：实现 Multi-Head Attention

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解为什么要从单头 attention 扩展到多头 attention。
2. 手写一个可运行的 `MultiHeadAttention` 模块。
3. 熟练掌握 `view`、`transpose`、`contiguous` 在多头拆分和合并中的作用。
4. 能解释多头 attention 中每个张量的 shape 变化。
5. 正确处理 attention mask 的 broadcast。
6. 能把 Multi-Head Attention 接入后续 Transformer Block。

上一讲我们实现了单头 Scaled Dot-Product Attention。

单头 attention 的核心是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Multi-Head Attention 不是另一个完全不同的机制。

它本质上是：

```text
把隐藏维度切成多个 head，每个 head 单独做 attention，最后再拼回去。
```

---

### 一、为什么需要多头 Attention

单头 attention 只有一个相似度空间。

它可以学到“当前 token 应该关注哪些 token”，但表达能力有限。

多头 attention 希望不同 head 关注不同关系。

例如在一句话中：

```text
The animal didn't cross the street because it was tired.
```

不同 head 可能分别关注：

1. 主谓关系。
2. 指代关系。
3. 局部短语关系。
4. 长距离依赖。
5. 标点或位置结构。

当然，真实模型里的 head 不一定能被人类清晰解释。

但工程直觉是：

```text
多个 head 让模型在多个子空间里并行计算 token 间关系。
```

---

### 二、多头 Attention 的整体流程

输入：

```text
x: [batch, seq_len, d_model]
```

假设：

```text
d_model = 512
num_heads = 8
head_dim = d_model / num_heads = 64
```

Multi-Head Attention 的流程是：

```text
x -> 线性层得到 q, k, v
q, k, v -> 拆成多个 head
每个 head 单独做 scaled dot-product attention
多个 head 的输出拼接
再过一个输出线性层
```

也就是：

```text
[B, T, D]
-> q/k/v: [B, T, D]
-> 拆 head: [B, H, T, Dh]
-> attention: [B, H, T, Dh]
-> 合并 head: [B, T, D]
-> output projection: [B, T, D]
```

其中：

```text
B  = batch size
T  = sequence length
D  = d_model
H  = num_heads
Dh = head_dim
```

---

### 三、先实现单头 Attention 函数

我们复用上一讲的核心函数，但支持多头 shape。

这次 q、k、v 的 shape 是：

```text
q: [batch, num_heads, seq_len, head_dim]
k: [batch, num_heads, seq_len, head_dim]
v: [batch, num_heads, seq_len, head_dim]
```

代码仍然成立：

```python
import math
import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights
```

shape 推导：

```text
q:      [B, H, T, Dh]
k^T:    [B, H, Dh, T]
scores: [B, H, T, T]
v:      [B, H, T, Dh]
out:    [B, H, T, Dh]
```

这说明 scaled dot-product attention 天然支持多头。

因为 PyTorch 的矩阵乘法会把前面的维度当作 batch 维度处理。

---

### 四、实现 MultiHeadAttention 类

完整代码如下：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)

        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_out = self._merge_heads(attn_out, batch_size, seq_len)
        out = self.out_proj(attn_out)
        return out, attn_weights

    def _split_heads(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x, batch_size, seq_len):
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
```

这是一个最小但完整的多头 self-attention 实现。

---

### 五、逐步解释 shape 变化

输入：

```python
x = torch.randn(2, 4, 16)
```

表示：

```text
batch = 2
seq_len = 4
d_model = 16
```

如果：

```python
num_heads = 4
```

那么：

```text
head_dim = 16 / 4 = 4
```

投影后：

```text
q: [2, 4, 16]
k: [2, 4, 16]
v: [2, 4, 16]
```

拆 head：

```python
x.view(batch_size, seq_len, num_heads, head_dim)
```

得到：

```text
[2, 4, 4, 4]
```

维度含义是：

```text
[batch, seq_len, num_heads, head_dim]
```

但 attention 计算希望 head 维度在 seq_len 前面。

所以做：

```python
x.transpose(1, 2)
```

得到：

```text
[2, 4, 4, 4]
```

这里数字刚好一样，容易误导。

真实含义已经变成：

```text
[batch, num_heads, seq_len, head_dim]
```

为了避免混淆，可以换一个例子：

```text
batch = 2
seq_len = 5
d_model = 12
num_heads = 3
head_dim = 4
```

拆分前：

```text
[2, 5, 12]
```

view 后：

```text
[2, 5, 3, 4]
```

transpose 后：

```text
[2, 3, 5, 4]
```

这就是每个 head 单独处理一个 `[seq_len, head_dim]` 的子空间。

---

### 六、为什么合并 head 前要 contiguous

合并 head 的代码是：

```python
x = x.transpose(1, 2).contiguous()
x = x.view(batch_size, seq_len, self.d_model)
```

attention 输出 shape：

```text
[batch, num_heads, seq_len, head_dim]
```

先 transpose：

```text
[batch, seq_len, num_heads, head_dim]
```

然后合并最后两维：

```text
[batch, seq_len, d_model]
```

为什么要 `contiguous()`？

因为 `transpose` 通常只改变张量的 stride，不一定重新排列底层内存。

而 `view` 要求内存布局连续。

所以常见写法是：

```python
x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
```

也可以用：

```python
x = x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
```

但面试和手写实现中，显式 `contiguous().view(...)` 更能说明你理解内存布局。

---

### 七、运行一个最小测试

```python
torch.manual_seed(42)

batch_size = 2
seq_len = 5
d_model = 12
num_heads = 3

x = torch.randn(batch_size, seq_len, d_model)
mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

out, attn_weights = mha(x)

print(out.shape)          # torch.Size([2, 5, 12])
print(attn_weights.shape) # torch.Size([2, 3, 5, 5])
print(attn_weights.sum(dim=-1).shape)
print(attn_weights.sum(dim=-1))
```

输出含义：

```text
out: [batch, seq_len, d_model]
attn_weights: [batch, num_heads, seq_len, seq_len]
```

`attn_weights.sum(dim=-1)` 应该接近全 1。

因为每个 head 中，每个 query token 对所有 key token 的关注权重和为 1。

---

### 八、加入 causal mask

多头 attention 的 scores shape 是：

```text
[batch, num_heads, seq_len, seq_len]
```

所以 causal mask 需要能 broadcast 到这个 shape。

可以写成：

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
```

shape 是：

```text
[1, 1, seq_len, seq_len]
```

它可以 broadcast 到：

```text
[batch, num_heads, seq_len, seq_len]
```

测试：

```python
mask = make_causal_mask(seq_len, device=x.device)
out, attn_weights = mha(x, mask=mask)

print(mask.shape)         # torch.Size([1, 1, 5, 5])
print(attn_weights.shape) # torch.Size([2, 3, 5, 5])
print(attn_weights[0, 0])
```

你应该看到每一行未来位置的权重接近 0。

---

### 九、实现 padding mask

实际 batch 中经常有 padding token。

例如：

```text
tokens:
[12, 35, 98, 0, 0]
```

其中 `0` 是 padding。

我们不希望任何 query 关注 padding key。

假设 padding mask 是：

```python
padding_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
])
```

shape 是：

```text
[batch, seq_len]
```

要变成可 broadcast 的 attention mask：

```python
padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
```

shape 是：

```text
[batch, 1, 1, seq_len]
```

它可以 broadcast 到：

```text
[batch, num_heads, seq_len, seq_len]
```

含义是：

```text
对所有 head、所有 query，都不允许关注 padding key。
```

代码：

```python
padding_mask = torch.tensor([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
], device=x.device)

padding_mask = padding_mask.view(batch_size, 1, 1, seq_len)
out, attn_weights = mha(x, mask=padding_mask)
```

如果既要 padding mask，又要 causal mask，可以相乘：

```python
causal_mask = make_causal_mask(seq_len, device=x.device)
combined_mask = causal_mask * padding_mask
out, attn_weights = mha(x, mask=combined_mask)
```

---

### 十、和 PyTorch 官方模块对照

PyTorch 提供了：

```python
nn.MultiheadAttention
```

但它默认输入 shape 是：

```text
[seq_len, batch, embed_dim]
```

如果设置：

```python
batch_first=True
```

则输入可以是：

```text
[batch, seq_len, embed_dim]
```

示例：

```python
official_mha = nn.MultiheadAttention(
    embed_dim=d_model,
    num_heads=num_heads,
    batch_first=True,
)

official_out, official_weights = official_mha(x, x, x)

print(official_out.shape)
print(official_weights.shape)
```

注意：

```text
官方模块默认返回的 attention weights 可能会对 head 求平均。
```

如果想保留每个 head 的权重，可以查看版本对应参数，例如 `average_attn_weights=False`。

工程中经常使用官方模块、FlashAttention 或框架封装。

但面试和源码阅读中，手写版本更重要。

---

### 十一、参数量是多少

多头 attention 常见有四个线性层：

```text
W_q: [d_model, d_model]
W_k: [d_model, d_model]
W_v: [d_model, d_model]
W_o: [d_model, d_model]
```

忽略 bias，总参数量约为：

```text
4 * d_model * d_model
```

很多人以为 head 数越多参数量越大。

在 `d_model` 固定时，这通常不对。

因为 head 数增加时，单个 head 的 `head_dim` 会变小。

例如：

```text
d_model = 512, num_heads = 8, head_dim = 64
d_model = 512, num_heads = 16, head_dim = 32
```

投影矩阵仍然是 `[512, 512]`。

所以在标准实现里，固定 `d_model` 时改变 `num_heads` 通常不改变 QKV 和输出投影的参数量。

但它会影响计算划分、head_dim、kernel 效率和模型表达方式。

---

### 十二、计算复杂度

最核心的开销来自 attention scores：

```text
QK^T: [B, H, T, Dh] @ [B, H, Dh, T]
```

复杂度约为：

```text
O(B * H * T^2 * Dh)
```

因为：

```text
H * Dh = D
```

所以也可以写成：

```text
O(B * T^2 * D)
```

这说明 attention 的主要瓶颈是序列长度平方项 `T^2`。

当上下文长度从 4K 增加到 32K，attention score 矩阵会急剧变大。

这就是 FlashAttention、稀疏 attention、线性 attention、KV Cache、长上下文优化的重要背景。

---

### 十三、常见工程坑

#### 坑 1：`d_model` 不能被 `num_heads` 整除

必须满足：

```text
d_model % num_heads == 0
```

否则无法平均拆成多个 head。

#### 坑 2：拆 head 后忘记 transpose

如果只做：

```python
x.view(batch, seq_len, num_heads, head_dim)
```

但没有转成 `[batch, num_heads, seq_len, head_dim]`，后续 attention 语义会错。

#### 坑 3：合并 head 前忘记 `contiguous`

`transpose` 后直接 `view` 可能报错或得到错误布局。

#### 坑 4：mask 少了 head 维度

多头 scores 是：

```text
[B, H, T, T]
```

常用 mask shape 是：

```text
[1, 1, T, T]
[B, 1, 1, T]
[B, 1, T, T]
```

要确保能 broadcast。

#### 坑 5：把 `num_heads` 当成增加参数量的主要原因

固定 `d_model` 时，标准 MHA 的主要线性层参数量通常不随 head 数线性增加。

#### 坑 6：没有区分 self-attention 和 cross-attention

本讲实现的是 self-attention。

self-attention 中：

```text
q, k, v 都来自同一个 x。
```

cross-attention 中：

```text
q 来自 decoder hidden states
k, v 来自 encoder outputs 或外部模态特征
```

---

### 十四、面试怎么讲 Multi-Head Attention

如果面试官问“Multi-Head Attention 怎么实现”，可以这样回答：

```text
输入 x 先经过三个线性层得到 Q、K、V，shape 都是 [B, T, D]。然后把 D 拆成 H 个 head，每个 head 的维度是 D/H，得到 [B, H, T, Dh]。每个 head 独立计算 scaled dot-product attention，输出 [B, H, T, Dh]。之后把 head 维度转回并拼接成 [B, T, D]，最后经过输出线性层得到最终结果。
```

如果追问“为什么要多头”，可以回答：

```text
多头 attention 让模型在多个子空间中并行建模 token 间关系。不同 head 可以学习不同类型的依赖，比如局部关系、长距离关系、语法关系或指代关系。相比单头，它提高了表达能力。
```

如果追问“head 数增加参数量会不会增加”，可以回答：

```text
在标准实现里，如果 d_model 固定，Q、K、V 和输出投影矩阵仍是 d_model 到 d_model，所以参数量通常不随 head 数线性增加。head 数增加会减小每个 head 的 head_dim，影响表示划分和计算效率。
```

如果追问“复杂度瓶颈在哪里”，可以回答：

```text
主要瓶颈是 attention score 矩阵，复杂度约为 O(B * T^2 * D)，显存也需要保存 [B, H, T, T] 级别的权重或中间结果，因此长序列时开销很大。
```

---

### 十五、小练习

#### 练习 1

把 `d_model=16`、`num_heads=4`，打印每一步 q 的 shape。

确认它从 `[B, T, D]` 变成 `[B, H, T, Dh]`。

#### 练习 2

把 `num_heads` 改成不能整除 `d_model` 的值，观察报错。

#### 练习 3

去掉 `_merge_heads` 中的 `contiguous()`，观察是否报错。

解释原因。

#### 练习 4

实现一个 padding mask，并确认 padding key 对应列的 attention 权重接近 0。

#### 练习 5

把本讲实现改造成 cross-attention：`q` 来自 `x_q`，`k` 和 `v` 来自 `x_kv`。

---

### 本讲总结

这一讲实现了 Multi-Head Attention。

核心结论如下：

1. Multi-Head Attention 是多个 scaled dot-product attention 的并行组合。
2. 输入 `[B, T, D]` 会被拆成 `[B, H, T, Dh]`。
3. 每个 head 在自己的子空间里计算 attention。
4. 多个 head 的输出会拼接回 `[B, T, D]`。
5. `transpose` 和 `contiguous` 是实现中最容易出错的地方。
6. mask 必须能 broadcast 到 `[B, H, T, T]`。
7. 固定 `d_model` 时，标准 MHA 的参数量通常不随 head 数线性增加。
8. attention 的主要瓶颈是序列长度平方复杂度。

下一讲，我们专门实现 Causal Mask。

虽然本讲已经用到了 causal mask，但下一讲会把 padding mask、causal mask、组合 mask、bool mask、float mask 和 PyTorch 官方接口中的 mask 差异讲清楚。

## 第 9 讲：实现 Causal Mask

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解 causal mask 为什么是自回归语言模型的核心约束。
2. 从零实现 causal mask、padding mask 和二者组合。
3. 掌握 mask 在 attention score 中的应用方式。
4. 分清 bool mask、0/1 mask、additive mask 的语义差异。
5. 理解 PyTorch `nn.MultiheadAttention` 和 `scaled_dot_product_attention` 中 mask 接口的常见差异。
6. 能排查 mask 方向反了、shape 不匹配、全 mask 导致 NaN 等工程问题。

前两讲我们已经实现了 Scaled Dot-Product Attention 和 Multi-Head Attention。

这两讲都用到了 mask。

但 mask 是 Transformer 实战里最容易写错的组件之一。

尤其是 causal mask。

它直接决定自回归语言模型训练时是否发生“偷看未来”。

本讲专门把 mask 讲透。

---

### 一、Causal Mask 解决什么问题

自回归语言模型的训练目标是：

```text
根据前面的 token 预测下一个 token。
```

例如序列：

```text
我 爱 机器 学习
```

模型在不同位置应该看到的信息是：

```text
预测 爱：只能看到 我
预测 机器：只能看到 我、爱
预测 学习：只能看到 我、爱、机器
```

如果第 1 个位置能直接看到第 3 个位置，那训练就变成作弊。

模型会利用未来 token，而不是学习语言建模规律。

所以 causal mask 的核心作用是：

```text
让当前位置只能关注自己和历史位置，不能关注未来位置。
```

---

### 二、Causal Mask 长什么样

假设 `seq_len = 5`。

causal mask 是一个下三角矩阵：

```text
[[1, 0, 0, 0, 0],
 [1, 1, 0, 0, 0],
 [1, 1, 1, 0, 0],
 [1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1]]
```

行表示 query 位置。

列表示 key 位置。

第 `i` 行第 `j` 列表示：

```text
第 i 个 token 是否允许关注第 j 个 token。
```

如果用 0-based index：

```text
允许关注：j <= i
禁止关注：j > i
```

也就是说，矩阵上三角部分是未来位置，要被 mask 掉。

---

### 三、最简单的实现

```python
import torch


def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


mask = make_causal_mask(5)
print(mask)
```

输出：

```text
tensor([[1., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1.]])
```

如果要用于多头 attention，通常返回：

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
```

shape 是：

```text
[1, 1, seq_len, seq_len]
```

可以 broadcast 到 attention scores：

```text
[batch, num_heads, seq_len, seq_len]
```

---

### 四、mask 如何作用到 attention scores

attention scores 是：

```text
scores = QK^T / sqrt(d_k)
```

shape 是：

```text
[batch, num_heads, seq_len, seq_len]
```

mask 的作用是在 softmax 之前完成。

如果某个位置不允许关注，就把对应 score 设成一个很小的负数：

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

然后再做：

```python
attn_weights = torch.softmax(scores, dim=-1)
```

因为 `exp(-1e9)` 接近 0，所以 softmax 后该位置权重接近 0。

完整 attention 函数：

```python
import math


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights
```

注意：

```text
mask 必须在 softmax 前使用。
```

如果 softmax 后再把权重置 0，剩余权重的和就不一定为 1。

---

### 五、验证 causal mask 是否生效

```python
torch.manual_seed(42)

batch = 1
num_heads = 1
seq_len = 5
head_dim = 4

q = torch.randn(batch, num_heads, seq_len, head_dim)
k = torch.randn(batch, num_heads, seq_len, head_dim)
v = torch.randn(batch, num_heads, seq_len, head_dim)

mask = make_causal_mask(seq_len, device=q.device)
out, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

print(attn_weights[0, 0])
```

你应该观察到：

```text
第 0 行只有第 0 列非 0。
第 1 行只有第 0、1 列非 0。
第 2 行只有第 0、1、2 列非 0。
未来列的权重接近 0。
```

还可以直接检查上三角：

```python
future_positions = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
print(attn_weights[0, 0][future_positions])
```

这些值应该接近 0。

---

### 六、Padding Mask 解决什么问题

batch 内序列长度通常不同。

为了组成 batch，会把短序列补齐：

```text
样本 A: [12, 35, 98, 77, 64]
样本 B: [21, 43,  0,  0,  0]
```

其中 `0` 可能是 padding token。

padding token 不是真实内容。

所以模型不应该关注它。

padding mask 的作用是：

```text
禁止所有 query 关注 padding key。
```

假设：

```python
tokens = torch.tensor([
    [12, 35, 98, 77, 64],
    [21, 43,  0,  0,  0],
])
```

构造 padding mask：

```python
pad_token_id = 0
padding_mask = (tokens != pad_token_id).long()
print(padding_mask)
```

输出：

```text
tensor([[1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0]])
```

shape 是：

```text
[batch, seq_len]
```

为了作用到 attention scores，要 reshape：

```python
padding_mask = padding_mask.view(tokens.size(0), 1, 1, tokens.size(1))
```

shape 变成：

```text
[batch, 1, 1, seq_len]
```

它可以 broadcast 到：

```text
[batch, num_heads, query_len, key_len]
```

---

### 七、组合 causal mask 和 padding mask

语言模型训练中经常两个 mask 都要用。

causal mask 控制：

```text
不能看未来。
```

padding mask 控制：

```text
不能看 padding。
```

二者都用 1 表示可见、0 表示不可见时，可以相乘：

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def make_padding_mask(tokens, pad_token_id=0):
    mask = (tokens != pad_token_id).long()
    return mask.view(tokens.size(0), 1, 1, tokens.size(1))


tokens = torch.tensor([
    [12, 35, 98, 77, 64],
    [21, 43,  0,  0,  0],
])

causal_mask = make_causal_mask(tokens.size(1), device=tokens.device)
padding_mask = make_padding_mask(tokens, pad_token_id=0)
combined_mask = causal_mask * padding_mask

print(combined_mask.shape)
```

输出 shape：

```text
[2, 1, 5, 5]
```

为什么是 `[2, 1, 5, 5]`？

因为：

```text
causal_mask:  [1, 1, 5, 5]
padding_mask: [2, 1, 1, 5]
combined:     [2, 1, 5, 5]
```

它可以继续 broadcast 到：

```text
[batch, num_heads, seq_len, seq_len]
```

---

### 八、0/1 mask、bool mask、additive mask

工程里最容易混乱的是 mask 语义。

常见三种形式。

#### 形式 1：0/1 mask

我们前面使用的是：

```text
1 表示可见
0 表示不可见
```

用法：

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

#### 形式 2：bool keep mask

也可以用 bool 表示可见：

```text
True 表示可见
False 表示不可见
```

用法：

```python
scores = scores.masked_fill(~mask, -1e9)
```

#### 形式 3：bool block mask

有些接口使用相反语义：

```text
True 表示要 mask 掉
False 表示允许关注
```

用法：

```python
scores = scores.masked_fill(mask, -1e9)
```

#### 形式 4：additive mask

additive mask 直接加到 scores 上：

```text
可见位置加 0
不可见位置加一个很小的负数
```

例如：

```python
additive_mask = (1 - mask) * -1e9
scores = scores + additive_mask
```

很多大模型实现喜欢 additive mask。

因为它和 attention logits 更容易统一处理。

---

### 九、低精度下不要随便写 `-inf`

很多教程会写：

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

数学上没问题。

但在 fp16、bf16、不同 kernel 或某些全 mask 场景中，可能更容易出现 NaN。

更稳妥的写法是用 dtype 对应的最小值：

```python
mask_value = torch.finfo(scores.dtype).min
scores = scores.masked_fill(mask == 0, mask_value)
```

或者使用框架内置 attention 接口，让它处理数值细节。

但也要注意：

```text
如果某一行所有 key 都不可见，softmax 仍然可能出 NaN 或无意义结果。
```

所以构造 mask 时要保证每个 query 至少有一个可见 key。

---

### 十、PyTorch 接口里的 mask 差异

PyTorch 有多个 attention 接口。

mask 语义不完全一样，必须查文档或看源码。

#### 1. `torch.nn.functional.scaled_dot_product_attention`

常见调用：

```python
import torch.nn.functional as F

out = F.scaled_dot_product_attention(
    q,
    k,
    v,
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
)
```

它还支持：

```python
is_causal=True
```

这表示直接使用 causal mask，不一定需要手动传 causal mask。

但要注意：

```text
不要在不了解版本行为的情况下，同时传复杂 attn_mask 又设置 is_causal=True。
```

不同版本对组合行为、mask dtype、形状要求可能有细节差异。

#### 2. `nn.MultiheadAttention`

它有两个常见 mask 参数：

```python
attn_mask
key_padding_mask
```

直觉上：

```text
attn_mask：通常用于控制 query-key 位置关系，比如 causal mask。
key_padding_mask：通常用于屏蔽 padding key。
```

但 bool mask 的 True/False 语义可能和你手写代码相反。

所以工程里必须明确：

```text
当前接口的 True 到底表示保留，还是表示屏蔽？
```

建议在接入官方接口时写一个小测试：

```text
构造 seq_len=4 的简单输入，打印 attention weights，确认未来位置和 padding 位置是否真的为 0。
```

不要只靠直觉。

---

### 十一、一个更稳的手写 Attention Mask 实现

下面把 mask 逻辑封装得稍微清楚一点。

约定：

```text
传入 mask 中 True 表示允许关注。
```

代码：

```python
def apply_attention_mask(scores, mask):
    if mask is None:
        return scores

    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    mask_value = torch.finfo(scores.dtype).min
    return scores.masked_fill(~mask, mask_value)
```

配合 attention 使用：

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    scores = apply_attention_mask(scores, mask)
    attn_weights = torch.softmax(scores, dim=-1)
    out = attn_weights @ v
    return out, attn_weights
```

然后 causal mask 可以直接返回 bool：

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.view(1, 1, seq_len, seq_len)
```

padding mask：

```python
def make_padding_mask(tokens, pad_token_id=0):
    mask = tokens != pad_token_id
    return mask.view(tokens.size(0), 1, 1, tokens.size(1))
```

组合：

```python
combined_mask = make_causal_mask(seq_len, x.device) & make_padding_mask(tokens)
```

使用 bool mask 的好处是语义更明确。

但前提是团队内部必须统一约定。

---

### 十二、常见工程坑

#### 坑 1：上三角和下三角写反

Causal mask 应该允许看历史。

如果 `i` 是 query，`j` 是 key，则允许：

```text
j <= i
```

对应下三角。

#### 坑 2：行列语义搞反

attention scores 的最后两维通常是：

```text
[query_len, key_len]
```

不是 `[key_len, query_len]`。

#### 坑 3：mask 在 softmax 后才用

应该在 softmax 前把不可见位置设为极小负数。

#### 坑 4：padding mask 只 mask 了 query，没有 mask key

多数情况下，我们要禁止关注 padding key。

所以 padding mask 应 reshape 成：

```text
[batch, 1, 1, key_len]
```

#### 坑 5：bool mask 语义反了

有的代码 `True` 表示保留。

有的接口 `True` 表示屏蔽。

必须写测试确认。

#### 坑 6：全 mask 行导致 NaN

如果某个 query 没有任何可见 key，softmax 会出问题。

常见原因是 padding、causal 和截断逻辑组合错了。

#### 坑 7：mask device 不一致

例如 q 在 GPU 上，mask 在 CPU 上，会报错。

构造 mask 时要传：

```python
device=q.device
```

#### 坑 8：mask dtype 不匹配

有些接口要求 bool mask，有些接受 float additive mask。

不要混用语义。

---

### 十三、面试怎么讲 Causal Mask

如果面试官问“causal mask 是什么”，可以这样回答：

```text
Causal mask 是自回归语言模型中的注意力遮罩，用来保证第 t 个位置只能关注自己和 t 之前的位置，不能关注未来 token。它通常是一个下三角矩阵，作用在 attention softmax 之前，把未来位置的 attention logits 设为极小负数，使 softmax 后权重接近 0。
```

如果追问“padding mask 和 causal mask 区别是什么”，可以回答：

```text
causal mask 控制时间方向，防止当前位置看到未来；padding mask 控制有效 token，防止模型关注 padding token。语言模型训练中经常要把二者组合起来，同时满足不看未来和不看 padding。
```

如果追问“mask shape 怎么设计”，可以回答：

```text
如果 attention scores 是 [B, H, Tq, Tk]，causal mask 常做成 [1, 1, T, T]，padding mask 常做成 [B, 1, 1, Tk]，二者可以 broadcast 后组合成 [B, 1, T, T]，再 broadcast 到所有 head。
```

如果追问“为什么 mask 要在 softmax 前”，可以回答：

```text
因为 softmax 是归一化操作。只有在 softmax 前把不可见位置的 logits 设为极小负数，才能让这些位置概率接近 0，同时让可见位置重新归一化。如果 softmax 后再置 0，权重和会被破坏。
```

---

### 十四、小练习

#### 练习 1

手写 `seq_len=6` 的 causal mask。

确认第 3 行只能看到第 0、1、2、3 列。

#### 练习 2

构造一个 batch：

```text
[[1, 2, 3, 0, 0],
 [4, 5, 6, 7, 0]]
```

实现 padding mask，并打印 shape。

#### 练习 3

把 causal mask 和 padding mask 组合起来。

确认最终 shape 可以 broadcast 到 `[B, H, T, T]`。

#### 练习 4

故意把 `torch.tril` 改成 `torch.triu`。

观察 attention weights 如何错误地关注未来。

#### 练习 5

用 `F.scaled_dot_product_attention(..., is_causal=True)` 和手写 causal mask 版本对比输出。

---

### 本讲总结

这一讲专门实现并梳理了 Causal Mask。

核心结论如下：

1. causal mask 防止自回归语言模型看到未来 token。
2. causal mask 通常是下三角矩阵，允许 `key_position <= query_position`。
3. mask 应作用在 softmax 前，而不是 softmax 后。
4. padding mask 用来阻止模型关注 padding key。
5. causal mask 常用 shape 是 `[1, 1, T, T]`。
6. padding mask 常用 shape 是 `[B, 1, 1, T]`。
7. 多个 mask 可以通过逻辑与或相乘组合。
8. bool mask、0/1 mask、additive mask 的语义必须统一。
9. PyTorch 不同 attention 接口的 mask 约定可能不同，工程中要写小测试验证。

下一讲，我们把前面实现的 embedding、attention、mask、MLP、归一化和残差连接组合起来，实现一个完整 Transformer Block。

## 第 10 讲：实现 Transformer Block

### 本讲目标

学完本讲，你应该能做到五件事：

1. 理解 Transformer Block 由哪些组件组成。
2. 从零实现一个可运行的 decoder-only Transformer Block。
3. 讲清 Attention、MLP、LayerNorm、残差连接之间的关系。
4. 分清 Post-LN 和 Pre-LN 两种结构。
5. 能用面试语言解释为什么现代大模型更常用 Pre-LN 结构。

前面几讲我们已经实现了：

```text
Token Embedding
Positional Encoding
Scaled Dot-Product Attention
Multi-Head Attention
Causal Mask
```

这些组件单独看都不复杂。

但真正的大模型不是只调用一次 attention。

它会把多个 Transformer Block 堆叠起来。

每个 block 负责一次信息混合和一次非线性变换。

本讲我们把这些组件组装成一个完整的 Transformer Block。

---

### 一、Transformer Block 解决什么问题

一个 decoder-only Transformer Block 通常包含两类核心计算：

```text
1. Self-Attention：让 token 之间交换信息。
2. Feed Forward Network：对每个 token 独立做非线性特征变换。
```

如果只做 attention，模型能根据上下文聚合信息，但表达能力不够。

如果只做 MLP，每个 token 只能独立变换，无法感知上下文。

所以 Transformer Block 把二者结合起来：

```text
Attention 负责跨 token 通信。
MLP 负责逐 token 特征加工。
Residual 负责保留原信息并改善梯度传播。
LayerNorm 负责稳定训练。
```

一个 block 的输入输出 shape 通常相同：

```text
输入 x: [batch, seq_len, d_model]
输出 y: [batch, seq_len, d_model]
```

这就允许我们反复堆叠很多层。

---

### 二、Post-LN 与 Pre-LN

经典 Transformer 论文使用的是 Post-LN。

结构可以写成：

```text
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + MLP(x))
```

现代大模型更常见的是 Pre-LN。

结构可以写成：

```text
x = x + SelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

二者差别在于 LayerNorm 放在子层前还是子层后。

Post-LN：

```text
先残差相加，再归一化。
```

Pre-LN：

```text
先归一化，再进入 attention 或 MLP，最后残差相加。
```

现代大模型更偏向 Pre-LN，主要原因是训练深层网络时更稳定。

直观理解：

```text
Pre-LN 中残差主干更接近一条干净的信息高速路，梯度可以更顺畅地沿残差路径传播。
```

本讲实现 Pre-LN 版本。

---

### 三、先实现 Feed Forward Network

Transformer 里的 MLP 通常叫 FFN，也就是 Feed Forward Network。

它对每个 token 独立执行同一个两层网络。

输入输出：

```text
[B, T, D] -> [B, T, D]
```

中间维度通常更大：

```text
D -> 4D -> D
```

代码：

```python
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

为什么用 GELU？

很多 Transformer 模型使用 GELU 或 SwiGLU。

GELU 相比 ReLU 更平滑，是 BERT、GPT 系列早期架构中的常见选择。

后面的课程会单独实现 SwiGLU。

---

### 四、复用前面实现的 Multi-Head Attention

这里使用一个简化版 Multi-Head Attention。

它接收：

```text
x:    [B, T, D]
mask: [B, 1, T, T] 或 [1, 1, T, T]
```

返回：

```text
out:          [B, T, D]
attn_weights: [B, H, T, T]
```

完整代码：

```python
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask=None):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, mask_value)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, attn_weights
```

这里约定：

```text
mask 中 True 表示允许关注。
```

这个约定和上一讲保持一致。

---

### 五、实现 Transformer Block

Pre-LN Transformer Block 的核心结构：

```text
attn_out = SelfAttention(LayerNorm(x))
x = x + attn_out

mlp_out = MLP(LayerNorm(x))
x = x + mlp_out
```

代码：

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        hidden_dim = d_model * mlp_ratio

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout=dropout)

    def forward(self, x, mask=None, return_attn=False):
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out

        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        if return_attn:
            return x, attn_weights
        return x
```

这就是一个最小但完整的 Transformer Block。

输入输出 shape 都是：

```text
[B, T, D]
```

---

### 六、构造 Causal Mask 并运行

先写 mask 函数：

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.view(1, 1, seq_len, seq_len)
```

运行一个 block：

```python
torch.manual_seed(42)

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)
mask = make_causal_mask(seq_len, device=x.device)

block = TransformerBlock(
    d_model=d_model,
    num_heads=num_heads,
    mlp_ratio=4,
    dropout=0.1,
)

y, attn_weights = block(x, mask=mask, return_attn=True)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("attn shape:", attn_weights.shape)
```

期望输出：

```text
x shape: torch.Size([2, 5, 16])
y shape: torch.Size([2, 5, 16])
attn shape: torch.Size([2, 4, 5, 5])
```

这说明：

```text
Transformer Block 不改变序列长度，也不改变 hidden size。
```

它只改变每个 token 的表示。

---

### 七、检查 causal mask 是否仍然生效

```python
print(attn_weights[0, 0])
```

你应该看到第 0 个 head 的 attention 权重中，上三角未来位置接近 0。

也可以写断言：

```python
future_positions = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1,
).bool()

future_weights = attn_weights[0, 0][future_positions]
print(future_weights)
```

在没有 dropout 或 eval 模式下，未来位置应该非常接近 0。

如果处于训练模式，并且 attention dropout 开启，可见位置权重可能被 dropout 改变。

为了验证 mask，建议：

```python
block.eval()
```

或者把 dropout 设为 0。

---

### 八、加入 Padding Mask

如果输入 token 有 padding，需要组合 mask。

```python
def make_padding_mask(tokens, pad_token_id=0):
    mask = tokens != pad_token_id
    return mask.view(tokens.size(0), 1, 1, tokens.size(1))
```

示例：

```python
tokens = torch.tensor([
    [10, 11, 12, 13, 14],
    [20, 21,  0,  0,  0],
])

causal_mask = make_causal_mask(tokens.size(1), device=tokens.device)
padding_mask = make_padding_mask(tokens, pad_token_id=0)
combined_mask = causal_mask & padding_mask

print(combined_mask.shape)
```

输出：

```text
torch.Size([2, 1, 5, 5])
```

这个 mask 可以直接传给 block：

```python
x = torch.randn(tokens.size(0), tokens.size(1), d_model)
y = block(x, mask=combined_mask)
```

注意：

```text
这里的 padding mask 屏蔽的是 key 位置。
```

也就是说，其他 token 不会关注 padding token。

实际训练时，loss 计算也要忽略 padding 位置。

mask attention 和 mask loss 是两件不同的事。

---

### 九、为什么要有残差连接

残差连接写作：

```python
x = x + sublayer(x)
```

它有三个作用。

第一，保留原始信息。

如果某层暂时学不好，模型至少还能通过残差路径传递输入。

第二，改善梯度传播。

深层网络中，梯度可以沿着加法路径更直接地往回传。

第三，让每层学习“增量更新”。

Transformer Block 不需要每一层都从零重写表示。

它只需要学习：

```text
在当前表示基础上应该补充什么上下文信息。
```

这也是为什么大模型可以堆叠几十层甚至上百层。

---

### 十、为什么要有 LayerNorm

LayerNorm 对每个 token 的 hidden 维度做归一化。

输入：

```text
[B, T, D]
```

它对最后一维 `D` 归一化。

也就是说，每个 token 独立归一化。

它的作用是稳定激活分布，让训练更容易。

Pre-LN 中，进入 attention 和 MLP 前都会先归一化：

```python
self.attn(self.ln1(x), mask=mask)
self.ffn(self.ln2(x))
```

直观理解：

```text
每个子层拿到的输入分布更稳定，优化难度更低。
```

现代 LLM 中，经常把 LayerNorm 换成 RMSNorm。

后面的课程会实现 RMSNorm。

---

### 十一、为什么 Attention 后还需要 MLP

Attention 本质上是加权求和。

它擅长从其他 token 取信息。

但如果没有 MLP，每个位置的非线性加工能力会不足。

MLP 的作用可以理解为：

```text
对 attention 聚合后的上下文表示做逐 token 的复杂变换。
```

Attention 和 MLP 的分工：

```text
Attention：token mixing，跨位置信息交互。
MLP：channel mixing，hidden 维度上的非线性变换。
```

这也是很多架构分析里常说的：

```text
Transformer Block = Token Mixing + Channel Mixing
```

---

### 十二、参数量估算

假设：

```text
d_model = D
mlp_ratio = 4
hidden_dim = 4D
```

Self-Attention 主要参数：

```text
Q projection: D * D
K projection: D * D
V projection: D * D
O projection: D * D
合计约 4D^2
```

MLP 主要参数：

```text
第一层: D * 4D = 4D^2
第二层: 4D * D = 4D^2
合计约 8D^2
```

所以一个标准 Transformer Block 的主参数量约为：

```text
12D^2
```

LayerNorm 和 bias 参数相比很小。

这也是为什么在很多 LLM 中，MLP 的参数量通常比 attention 更大。

---

### 十三、计算复杂度

Self-Attention 的主要复杂度来自：

```text
QK^T: [T, D] x [D, T] -> [T, T]
```

复杂度约为：

```text
O(T^2 D)
```

MLP 的复杂度约为：

```text
O(T D hidden_dim)
```

如果 `hidden_dim = 4D`，则约为：

```text
O(4T D^2)
```

所以：

```text
长序列时，attention 的 T^2 会成为瓶颈。
大 hidden size 时，MLP 也会占大量计算。
```

这解释了为什么长上下文优化、FlashAttention、Linear Attention、SSM 和 Mamba 会成为重要方向。

---

### 十四、常见工程坑

#### 坑 1：忘记残差连接

没有残差，深层 Transformer 很难训练。

#### 坑 2：LayerNorm 放错位置但自己没意识到

Post-LN 和 Pre-LN 都能写，但训练稳定性和学习率敏感性不同。

面试和工程中要明确自己实现的是哪一种。

#### 坑 3：attention 输出 shape 不等于输入 shape

Transformer Block 必须保证输入输出都是 `[B, T, D]`，否则无法堆叠。

#### 坑 4：mask shape 不能 broadcast

传入 block 的 mask 最好能 broadcast 到：

```text
[B, H, T, T]
```

#### 坑 5：dropout 影响验证 attention 权重

如果想检查 mask 是否生效，建议切换到 eval 模式或设置 dropout 为 0。

#### 坑 6：把 loss mask 和 attention mask 混为一谈

attention mask 控制模型能看哪里。

loss mask 控制哪些位置参与损失。

这两个 mask 经常都需要，但作用不同。

---

### 十五、面试怎么讲 Transformer Block

如果面试官问“Transformer Block 由什么组成”，可以这样回答：

```text
一个典型 Transformer Block 由 self-attention、feed-forward network、LayerNorm 和 residual connection 组成。Self-attention 负责不同 token 之间的信息交互，FFN 负责对每个 token 的 hidden 表示做非线性变换，残差连接改善梯度传播并保留原信息，LayerNorm 稳定训练。
```

如果追问“Pre-LN 和 Post-LN 有什么区别”，可以回答：

```text
Post-LN 是先做子层和残差相加，再 LayerNorm；Pre-LN 是先 LayerNorm，再进入子层，最后与原输入做残差相加。现代大模型更常用 Pre-LN，因为深层训练更稳定，残差路径更利于梯度传播。
```

如果追问“Attention 和 MLP 分别负责什么”，可以回答：

```text
Attention 主要负责 token mixing，也就是跨位置的信息交互；MLP 主要负责 channel mixing，也就是在 hidden 维度上做非线性特征变换。二者结合后，模型既能聚合上下文，又能提升表示能力。
```

如果追问“一个 block 参数量大概是多少”，可以回答：

```text
在标准设置下，attention 的 Q、K、V、O 投影约 4D^2，MLP 如果使用 4D hidden size，则两层线性约 8D^2，所以一个 block 主参数量约 12D^2，不含很小的 bias 和 norm 参数。
```

---

### 十六、小练习

#### 练习 1

运行本讲代码，确认 `TransformerBlock` 输入输出 shape 都是 `[B, T, D]`。

#### 练习 2

把 `mlp_ratio` 从 4 改成 2，观察参数量如何变化。

#### 练习 3

把 Pre-LN 改成 Post-LN，写出对应代码。

#### 练习 4

设置 `dropout=0`，验证 causal mask 上三角 attention weights 是否为 0。

#### 练习 5

堆叠两个 `TransformerBlock`，确认输出 shape 仍然不变。


---

### 本讲总结

这一讲实现了一个完整的 decoder-only Transformer Block。

核心结论如下：

1. Transformer Block 的输入输出 shape 通常都是 `[B, T, D]`。
2. Attention 负责跨 token 信息交互。
3. MLP 负责 hidden 维度上的非线性变换。
4. 残差连接保留信息并改善梯度传播。
5. LayerNorm 稳定训练。
6. 现代大模型更常用 Pre-LN，而不是原始论文中的 Post-LN。
7. 一个标准 block 的主参数量大约是 `12D^2`。
8. 长序列下 attention 的 `T^2` 复杂度是核心瓶颈。

下一讲，我们实现 RoPE，也就是现代大模型里非常常见的旋转位置编码。

## 第 11 讲：实现 RoPE

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解为什么 Transformer 需要位置编码。
2. 讲清 RoPE 和绝对位置编码的核心区别。
3. 从零实现 RoPE 的 `cos`、`sin` 缓存。
4. 正确把 RoPE 应用到 attention 的 `q` 和 `k` 上。
5. 理解 RoPE 的 shape、广播和偶奇维旋转逻辑。
6. 能解释 RoPE 为什么适合 decoder-only 大语言模型。

RoPE 全称是 Rotary Position Embedding，中文常译为旋转位置编码。

它是现代大语言模型中最常见的位置编码方案之一。

LLaMA、Qwen、ChatGLM 等很多模型都采用了 RoPE 或 RoPE 变体。

本讲我们从零实现 RoPE，并把它接入 Multi-Head Attention。

---

### 一、为什么需要位置编码

Self-Attention 本身对 token 顺序不敏感。

如果没有位置编码，attention 只看到一组 token 表示。

对于下面两个序列：

```text
我 喜欢 你
你 喜欢 我
```

如果只看 token 集合，它们包含的词类似。

但语义完全不同。

所以 Transformer 必须知道：

```text
每个 token 在序列中的位置。
```

常见位置编码方案包括：

1. 绝对位置编码：给第 0、1、2、3 个位置分别加一个位置向量。
2. 相对位置编码：让 attention 显式感知两个 token 的相对距离。
3. RoPE：通过旋转 q 和 k，把位置信息注入 attention 内积。

RoPE 的特点是：

```text
不直接把位置向量加到 x 上，而是在 attention 里旋转 q 和 k。
```

---

### 二、RoPE 的核心直觉

在二维平面里，一个向量可以旋转。

例如向量：

```text
[x1, x2]
```

旋转角度 `theta` 后变成：

```text
[x1 * cos(theta) - x2 * sin(theta),
 x1 * sin(theta) + x2 * cos(theta)]
```

RoPE 的想法是：

```text
把 hidden 维度两两分组，每一组看成二维向量，然后按照 token 位置旋转不同角度。
```

位置越靠后，旋转角度越大。

不同维度组使用不同频率。

这样 q 和 k 做内积时，内积结果不仅和内容有关，也和二者位置关系有关。

这就是 RoPE 的关键：

```text
位置信息进入了 q 和 k 的相似度计算。
```

---

### 三、RoPE 作用在哪里

标准 attention 是：

```text
q = Wq x
k = Wk x
v = Wv x

scores = q @ k^T / sqrt(d)
```

RoPE 的接入位置是：

```text
q = RoPE(q)
k = RoPE(k)
v 不做 RoPE
```

然后再计算：

```text
scores = q @ k^T / sqrt(d)
```

为什么不旋转 v？

因为 attention score 由 q 和 k 决定。

位置关系应该影响“关注谁”。

v 是被加权汇总的内容本身，通常不需要注入这种相对位置相似度结构。

---

### 四、RoPE 的频率设计

假设 head_dim 是 `D`。

RoPE 会把维度按两两一组处理：

```text
(0, 1), (2, 3), (4, 5), ...
```

每组使用一个频率。

常见实现：

```python
inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

其中 `base` 通常是：

```text
10000
```

如果 `head_dim = 8`，则 `torch.arange(0, head_dim, 2)` 是：

```text
[0, 2, 4, 6]
```

对应 4 个二维组。

每个位置 `pos` 的角度是：

```python
theta = pos * inv_freq
```

所以位置矩阵 shape 是：

```text
[seq_len, head_dim / 2]
```

---

### 五、实现 RoPE 缓存

为了避免每次 forward 都重新计算 `cos` 和 `sin`，通常预先缓存。

代码：

```python
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )

        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)

        cos = freqs.cos()
        sin = freqs.sin()

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len, device=None):
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        if device is not None:
            cos = cos.to(device)
            sin = sin.to(device)

        return cos, sin
```

这里的缓存 shape 是：

```text
cos: [max_seq_len, head_dim / 2]
sin: [max_seq_len, head_dim / 2]
```

`register_buffer` 的作用是：

```text
把 cos/sin 注册为模块状态，但不作为可训练参数。
```

`persistent=False` 表示它们不一定保存进 checkpoint。

实际工程中可以根据需要选择是否持久化。

---

### 六、实现 rotate_half

RoPE 要对偶数维和奇数维成对旋转。

如果一个二维向量是：

```text
[x_even, x_odd]
```

旋转公式是：

```text
new_even = x_even * cos - x_odd * sin
new_odd  = x_even * sin + x_odd * cos
```

先写一个清楚版本：

```python
def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = cos.view(1, 1, cos.size(0), cos.size(1))
    sin = sin.view(1, 1, sin.size(0), sin.size(1))

    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos

    x_out = torch.empty_like(x)
    x_out[..., 0::2] = x_rotated_even
    x_out[..., 1::2] = x_rotated_odd
    return x_out
```

这里假设 `x` 的 shape 是：

```text
[batch, num_heads, seq_len, head_dim]
```

所以：

```text
x_even: [B, H, T, Dh/2]
x_odd:  [B, H, T, Dh/2]
cos:    [1, 1, T, Dh/2]
sin:    [1, 1, T, Dh/2]
```

最后 broadcast 后逐元素相乘。

---

### 七、另一种常见 rotate_half 写法

很多开源实现会写：

```python
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
```

然后：

```python
def apply_rope(x, cos, sin):
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
    cos = cos.view(1, 1, cos.size(0), cos.size(1))
    sin = sin.view(1, 1, sin.size(0), sin.size(1))
    return x * cos + rotate_half(x) * sin
```

两种写法本质相同。

第一种更适合教学，因为偶奇维逻辑非常直观。

第二种更接近很多工程代码。

本讲后续使用第一种写法。

---

### 八、把 RoPE 接入 Multi-Head Attention

现在改造上一讲的 Multi-Head Attention。

变化只有一个：

```text
q、k 拆成多头之后，计算 scores 之前，应用 RoPE。
```

代码：

```python
import math


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        cos, sin = self.rope(seq_len, device=x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, mask_value)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, attn_weights
```

注意：

```text
RoPE 应用在 q 和 k 上，不应用在 v 上。
```

---

### 九、完整运行示例

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.view(1, 1, seq_len, seq_len)


torch.manual_seed(42)

batch_size = 2
seq_len = 6
d_model = 32
num_heads = 4

x = torch.randn(batch_size, seq_len, d_model)
mask = make_causal_mask(seq_len, device=x.device)

attn = RoPEMultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads,
    max_seq_len=128,
    dropout=0.0,
)

out, attn_weights = attn(x, mask=mask)

print("x shape:", x.shape)
print("out shape:", out.shape)
print("attn shape:", attn_weights.shape)
```

期望输出：

```text
x shape: torch.Size([2, 6, 32])
out shape: torch.Size([2, 6, 32])
attn shape: torch.Size([2, 4, 6, 6])
```

RoPE 不改变 q、k 的 shape。

它只改变 q、k 的数值。

---

### 十、RoPE 的 shape 逐步检查

假设：

```text
B = 2
T = 6
D = 32
H = 4
Dh = 8
```

输入：

```text
x: [2, 6, 32]
```

线性投影后：

```text
q: [2, 6, 32]
k: [2, 6, 32]
v: [2, 6, 32]
```

拆头后：

```text
q: [2, 4, 6, 8]
k: [2, 4, 6, 8]
v: [2, 4, 6, 8]
```

RoPE 缓存：

```text
cos: [6, 4]
sin: [6, 4]
```

reshape 后：

```text
cos: [1, 1, 6, 4]
sin: [1, 1, 6, 4]
```

偶奇维拆分：

```text
x_even: [2, 4, 6, 4]
x_odd:  [2, 4, 6, 4]
```

旋转后再交错放回：

```text
q: [2, 4, 6, 8]
k: [2, 4, 6, 8]
```

attention scores：

```text
scores: [2, 4, 6, 6]
```

---

### 十一、RoPE 和绝对位置编码的区别

绝对位置编码通常是：

```python
x = token_embedding + position_embedding
```

也就是说，位置向量直接加到 token 表示上。

RoPE 不是这样。

RoPE 是：

```python
q = Wq(x)
k = Wk(x)
q = rope(q)
k = rope(k)
```

区别可以总结为：

```text
绝对位置编码：位置影响 token 表示本身。
RoPE：位置影响 q 和 k 的相似度计算。
```

RoPE 的重要性质是，它能让 attention score 自然地包含相对位置信息。

直观上：

```text
两个 token 的 q/k 旋转角度差，和它们的位置距离有关。
```

这使它比简单绝对位置 embedding 更适合很多自回归语言模型场景。

---

### 十二、RoPE 的优点

#### 优点 1：适合自回归语言模型

RoPE 直接作用在 q/k 的匹配上。

这和语言模型“当前位置关注历史位置”的机制非常契合。

#### 优点 2：自然表达相对位置信息

attention score 与 q/k 的相对旋转角度有关。

这让模型更容易感知两个 token 的距离关系。

#### 优点 3：实现简单

不需要额外可训练位置 embedding 参数。

只需要 cos/sin 缓存。

#### 优点 4：推理时方便配合 KV Cache

自回归推理时，新 token 的位置是递增的。

只需要取当前位置对应的 cos/sin 即可。

---

### 十三、RoPE 的工程注意点

#### 注意 1：head_dim 必须是偶数

因为 RoPE 两两维度一组做旋转。

所以：

```python
assert head_dim % 2 == 0
```

#### 注意 2：RoPE 作用在每个 head 的 head_dim 上

不是对整个 `d_model` 直接做旋转。

常见 shape 是：

```text
[B, H, T, Dh]
```

#### 注意 3：只旋转 q 和 k

v 通常不做 RoPE。

#### 注意 4：缓存长度要覆盖最大上下文

如果 `max_seq_len=2048`，但输入长度是 4096，会越界或截断。

#### 注意 5：KV Cache 下要处理 position offset

训练时通常一次处理完整序列，位置是：

```text
0, 1, 2, ..., T-1
```

推理时如果已经缓存了前 `past_len` 个 token，新 token 的位置应该从：

```text
past_len
```

开始，而不是从 0 重新开始。

简单扩展：

```python
def forward(self, seq_len, offset=0, device=None):
    cos = self.cos_cached[offset:offset + seq_len]
    sin = self.sin_cached[offset:offset + seq_len]
    if device is not None:
        cos = cos.to(device)
        sin = sin.to(device)
    return cos, sin
```

#### 注意 6：长上下文外推不是免费午餐

RoPE 相比可学习绝对位置 embedding 有更好的外推潜力。

但直接把训练长度 4K 的模型拉到 32K，也可能退化。

实际大模型会用 RoPE scaling、NTK scaling、YaRN 等方法改造。

这些属于后续进阶内容。

---

### 十四、把 RoPE 接入 Transformer Block

只需要把上一讲 block 里的 attention 替换成 `RoPEMultiHeadAttention`。

```python
class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4, max_seq_len=2048, dropout=0.1):
        super().__init__()

        hidden_dim = d_model * mlp_ratio

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPEMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout=dropout)

    def forward(self, x, mask=None, return_attn=False):
        attn_out, attn_weights = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out

        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out

        if return_attn:
            return x, attn_weights
        return x
```

如果使用 RoPE，一般就不再需要单独加绝对位置 embedding。

输入可以是：

```text
x = token_embedding(tokens)
```

而不是：

```text
x = token_embedding(tokens) + position_embedding(positions)
```

---

### 十五、常见工程坑

#### 坑 1：把 RoPE 加到 x 上

RoPE 不是普通 position embedding。

它应该作用在 q/k 上。

#### 坑 2：忘记只对 q/k 做 RoPE

v 一般不旋转。

#### 坑 3：head_dim 是奇数

RoPE 要两两配对，head_dim 必须是偶数。

#### 坑 4：cos/sin shape 广播错误

对于 `[B, H, T, Dh]`，cos/sin 通常 reshape 成：

```text
[1, 1, T, Dh/2]
```

#### 坑 5：推理时 position 从 0 重置

KV Cache 场景下，新 token 的 position 必须接在历史 token 后面。

否则 q/k 的相对位置关系会错。

#### 坑 6：训练长度外直接暴力推到超长上下文

RoPE 有一定外推能力，但长上下文通常还需要 scaling 方法和对应训练。

---

### 十六、面试怎么讲 RoPE

如果面试官问“RoPE 是什么”，可以这样回答：

```text
RoPE 是旋转位置编码，它不是把位置向量直接加到 token embedding 上，而是在 attention 中对 q 和 k 按位置做二维旋转。这样 q 和 k 的内积会自然包含相对位置信息，因此很适合 decoder-only 语言模型。
```

如果追问“RoPE 作用在哪里”，可以回答：

```text
RoPE 通常作用在 multi-head attention 中拆头后的 q 和 k 上，shape 是 [B, H, T, Dh]，不作用在 v 上。应用 RoPE 后再计算 qk^T attention score。
```

如果追问“为什么 RoPE 能表达相对位置”，可以回答：

```text
因为不同位置会对应不同旋转角度。两个 token 的 q/k 内积与二者旋转角度差有关，而角度差由相对位置决定，所以 attention score 中会自然包含相对位置信息。
```

如果追问“RoPE 工程上最容易错什么”，可以回答：

```text
常见错误包括把 RoPE 加到 embedding 上、错误地旋转 v、head_dim 不是偶数、cos/sin broadcast 维度不对，以及 KV Cache 推理时没有使用正确的 position offset。
```

---

### 十七、小练习

#### 练习 1

用 `head_dim=8`、`seq_len=4` 打印 RoPE 的 `cos` 和 `sin` shape。

#### 练习 2

构造一个 `[2, 4, 6, 8]` 的 q，应用 `apply_rope`，确认输出 shape 不变。

#### 练习 3

把 `apply_rope` 改成 `rotate_half` 版本，并验证输出和本讲清晰版接近。

#### 练习 4

把 RoPE 接入上一讲的 `TransformerBlock`，确认输出 shape 仍是 `[B, T, D]`。

#### 练习 5

模拟 KV Cache 场景：设 `offset=10`，只处理长度为 1 的新 token，取第 10 个位置的 cos/sin。

---

### 本讲总结

这一讲实现了 RoPE。

核心结论如下：

1. RoPE 是现代大模型中常见的旋转位置编码。
2. RoPE 不直接加到 token embedding 上，而是作用在 q 和 k 上。
3. RoPE 把 head_dim 两两配对，按位置角度做二维旋转。
4. q/k 的旋转角度差让 attention score 自然包含相对位置信息。
5. RoPE 不改变 tensor shape，只改变 q/k 数值。
6. 工程实现中要注意 cos/sin 缓存、broadcast、head_dim 偶数和 KV Cache offset。
7. 使用 RoPE 后，通常不再额外使用可学习绝对位置 embedding。

下一讲，我们实现 RMSNorm 和 SwiGLU，它们是现代 LLM 中非常常见的归一化与 MLP 改造。

## 第 12 讲：实现 RMSNorm 和 SwiGLU

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解 LayerNorm 和 RMSNorm 的区别。
2. 从零实现 RMSNorm。
3. 理解普通 FFN 和 SwiGLU FFN 的区别。
4. 从零实现 SwiGLU。
5. 把 RMSNorm、RoPE Attention、SwiGLU 组合成现代 LLM 风格 Transformer Block。
6. 能在面试中解释为什么 LLaMA 类模型常用 RMSNorm 和 SwiGLU。

前面我们已经实现了 Transformer Block 和 RoPE。

但如果你看现代 LLM 的结构，会发现它和最原始 Transformer 还有一些差异。

两个非常常见的改造是：

```text
LayerNorm -> RMSNorm
普通 FFN -> SwiGLU FFN
```

本讲就把这两个组件手写出来。

---

### 一、RMSNorm 解决什么问题

LayerNorm 会对 hidden 维度做均值和方差归一化。

公式直观写成：

```text
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm 更简单。

它不减均值，只用均方根做缩放：

```text
y = x / RMS(x) * weight
```

其中：

```text
RMS(x) = sqrt(mean(x^2) + eps)
```

也就是说，RMSNorm 只关心向量的尺度，不显式中心化。

它的目标是：

```text
用更简单、更便宜的归一化方式稳定训练。
```

很多 LLaMA 类模型使用 RMSNorm。

---

### 二、LayerNorm 和 RMSNorm 的区别

假设输入是：

```text
x: [batch, seq_len, d_model]
```

LayerNorm 对最后一维做：

```text
减均值，除标准差，再做可学习缩放和平移。
```

RMSNorm 对最后一维做：

```text
不减均值，只除以 root mean square，再做可学习缩放。
```

对比：

```text
LayerNorm: 有 mean，有 variance，有 gamma，通常有 beta。
RMSNorm:   没有 mean，只用 RMS，通常只有 weight，没有 bias。
```

直观理解：

```text
LayerNorm 同时控制中心和尺度。
RMSNorm 主要控制尺度。
```

RMSNorm 少做一些操作，工程上更简洁，训练大模型时表现也很好。

---

### 三、从零实现 RMSNorm

代码：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
```

输入输出 shape：

```text
输入: [B, T, D]
输出: [B, T, D]
```

`self.weight` 的 shape 是：

```text
[D]
```

它会 broadcast 到 `[B, T, D]`。

---

### 四、更贴近工程的 RMSNorm 写法

有些实现会先把输入转成 float32 做归一化，再转回原 dtype。

这是为了提升 fp16/bf16 下的数值稳定性。

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return (x_norm.to(input_dtype) * self.weight)
```

这里用的是：

```python
torch.rsqrt(variance + self.eps)
```

它等价于：

```text
1 / sqrt(variance + eps)
```

很多大模型代码会采用这种写法。

---

### 五、验证 RMSNorm

```python
torch.manual_seed(42)

x = torch.randn(2, 3, 8)
norm = RMSNorm(d_model=8)
y = norm(x)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("weight shape:", norm.weight.shape)
```

期望输出：

```text
x shape: torch.Size([2, 3, 8])
y shape: torch.Size([2, 3, 8])
weight shape: torch.Size([8])
```

RMSNorm 不改变 shape。

---

### 六、SwiGLU 解决什么问题

原始 Transformer 的 FFN 通常是：

```text
Linear(D -> 4D) -> GELU/ReLU -> Linear(4D -> D)
```

SwiGLU 是一种带门控的 FFN。

它的核心形式是：

```text
SwiGLU(x) = Swish(xW_gate) * (xW_up)
```

然后再接一个 down projection：

```text
out = SwiGLU(x) W_down
```

其中 Swish 常用 SiLU 实现：

```text
SiLU(x) = x * sigmoid(x)
```

所以工程里常写成：

```python
F.silu(gate_proj(x)) * up_proj(x)
```

SwiGLU 的作用是：

```text
用门控机制控制哪些 hidden 通道应该通过，从而增强 MLP 表达能力。
```

---

### 七、普通 FFN 和 SwiGLU 的区别

普通 FFN：

```text
x -> Linear -> activation -> Linear
```

SwiGLU FFN：

```text
x -> gate_proj -> SiLU -> *
x -> up_proj   ----------^
然后 -> down_proj
```

也就是说，SwiGLU 有两条上投影分支：

```text
gate 分支：决定哪些信息通过。
up 分支：提供候选特征。
```

二者逐元素相乘。

直观理解：

```text
up_proj 负责生成内容，gate_proj 负责控制开关。
```

这比普通激活函数多了一层动态门控能力。

---

### 八、从零实现 SwiGLU

代码：

```python
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return self.dropout(out)
```

输入输出 shape：

```text
输入: [B, T, D]
gate: [B, T, H]
up:   [B, T, H]
乘积: [B, T, H]
输出: [B, T, D]
```

其中 `H` 是 FFN hidden dimension。

---

### 九、SwiGLU 的 hidden_dim 为什么不一定是 4D

普通 FFN 使用 `D -> 4D -> D` 时，参数量约为：

```text
D * 4D + 4D * D = 8D^2
```

SwiGLU 有三组投影：

```text
gate_proj: D * H
up_proj:   D * H
down_proj: H * D
合计:      3DH
```

如果直接设 `H = 4D`，参数量是：

```text
12D^2
```

比普通 FFN 大很多。

所以很多 LLaMA 类模型会把 SwiGLU 的 hidden_dim 设成约：

```text
8D / 3
```

因为：

```text
3D * (8D / 3) = 8D^2
```

这样 SwiGLU 的参数量大致和普通 4D FFN 持平。

工程中还会把 hidden_dim 调整为某个倍数的整数，方便硬件计算。

例如：

```python
def make_swiglu_hidden_dim(d_model, multiple_of=256):
    hidden_dim = int(8 * d_model / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim
```

教学时可以先直接使用：

```python
hidden_dim = int(8 * d_model / 3)
```

---

### 十、验证 SwiGLU

```python
torch.manual_seed(42)

d_model = 12
hidden_dim = int(8 * d_model / 3)

x = torch.randn(2, 5, d_model)
mlp = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=0.0)
y = mlp(x)

print("x shape:", x.shape)
print("hidden_dim:", hidden_dim)
print("y shape:", y.shape)
```

期望输出：

```text
x shape: torch.Size([2, 5, 12])
hidden_dim: 32
y shape: torch.Size([2, 5, 12])
```

SwiGLU 和普通 FFN 一样，不改变 `[B, T, D]` 的整体 shape。

---

### 十一、现代 LLM 风格 Transformer Block

现在我们把前几讲的组件组合起来：

```text
RMSNorm
RoPE Multi-Head Attention
SwiGLU
Residual Connection
Pre-Norm
```

结构：

```text
x = x + RoPEAttention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

代码：

```python
class LLMTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, dropout=0.1):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)

        self.attn_norm = RMSNorm(d_model)
        self.attn = RoPEMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x, mask=None, return_attn=False):
        attn_out, attn_weights = self.attn(self.attn_norm(x), mask=mask)
        x = x + attn_out

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out

        if return_attn:
            return x, attn_weights
        return x
```

这个 block 已经非常接近现代 decoder-only LLM 的基础层。

当然真实大模型还会包含：

```text
KV Cache
Grouped-Query Attention
FlashAttention
更复杂的 RoPE scaling
并行残差或特殊初始化
```

但主干结构已经在这里了。

---

### 十二、完整运行示例

```python
def make_causal_mask(seq_len, device=None):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask.view(1, 1, seq_len, seq_len)


torch.manual_seed(42)

batch_size = 2
seq_len = 8
d_model = 48
num_heads = 6

x = torch.randn(batch_size, seq_len, d_model)
mask = make_causal_mask(seq_len, device=x.device)

block = LLMTransformerBlock(
    d_model=d_model,
    num_heads=num_heads,
    max_seq_len=128,
    dropout=0.0,
)

y, attn_weights = block(x, mask=mask, return_attn=True)

print("x shape:", x.shape)
print("y shape:", y.shape)
print("attn shape:", attn_weights.shape)
```

期望输出：

```text
x shape: torch.Size([2, 8, 48])
y shape: torch.Size([2, 8, 48])
attn shape: torch.Size([2, 6, 8, 8])
```

这个例子说明：

```text
RMSNorm、RoPE、SwiGLU 都不会改变 block 的输入输出 shape。
```

---

### 十三、RMSNorm 与 SwiGLU 的面试重点

RMSNorm 的重点不是“比 LayerNorm 永远更好”。

而是：

```text
它用更简单的尺度归一化替代均值方差归一化，在大模型训练中足够稳定且计算更简洁。
```

SwiGLU 的重点不是“多写两个 Linear”。

而是：

```text
它在 FFN 中引入门控机制，用 gate 分支控制 up 分支的信息通过，提高 MLP 表达能力。
```

现代 LLM block 的重点是：

```text
Pre-RMSNorm + RoPE Attention + SwiGLU FFN + Residual
```

如果你能把这条主线讲清楚，面试官通常会认为你不是只会背概念，而是真的理解现代大模型结构。

---

### 十四、常见工程坑

#### 坑 1：RMSNorm 误写成 LayerNorm

RMSNorm 不减均值。

核心是：

```python
x * torch.rsqrt(mean(x ** 2) + eps)
```

#### 坑 2：RMSNorm 忘记 `keepdim=True`

如果没有 `keepdim=True`，最后一维 shape 会丢失，广播可能出错。

#### 坑 3：低精度下直接归一化

fp16/bf16 下建议用 float32 做 variance 计算，再转回原 dtype。

#### 坑 4：SwiGLU 的两个分支写反或少写一个分支

SwiGLU 至少需要：

```text
gate_proj 和 up_proj 两个上投影。
```

#### 坑 5：忘记 SiLU

SwiGLU 不是简单的：

```python
gate_proj(x) * up_proj(x)
```

而是：

```python
F.silu(gate_proj(x)) * up_proj(x)
```

#### 坑 6：hidden_dim 直接照搬 4D 导致参数量增大

SwiGLU 有三组矩阵。

如果要和普通 FFN 参数量接近，可以使用约 `8D/3`。

#### 坑 7：误以为 RMSNorm 和 SwiGLU 会改变 shape

二者都保持输入输出 `[B, T, D]`。

---

### 十五、面试怎么讲 RMSNorm 和 SwiGLU

如果面试官问“RMSNorm 和 LayerNorm 有什么区别”，可以这样回答：

```text
LayerNorm 会减均值并除以标准差，而 RMSNorm 不做中心化，只用 hidden 维度的 root mean square 来缩放输入，通常再乘一个可学习 weight。RMSNorm 计算更简单，在很多 LLaMA 类大模型中被用来替代 LayerNorm。
```

如果追问“RMSNorm 的公式是什么”，可以回答：

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight，mean 是在 hidden 维度上计算的。
```

如果问“SwiGLU 是什么”，可以回答：

```text
SwiGLU 是一种门控 FFN。它有 gate_proj 和 up_proj 两个上投影分支，用 SiLU(gate_proj(x)) 作为门控，与 up_proj(x) 逐元素相乘，再通过 down_proj 投回 d_model。它比普通 FFN 有更强的表达能力。
```

如果追问“为什么 SwiGLU hidden_dim 常用 8D/3”，可以回答：

```text
普通 4D FFN 的参数量约是 8D^2。SwiGLU 有 gate、up、down 三个投影，参数量约 3D * hidden_dim。令 hidden_dim 约等于 8D/3，就能让 SwiGLU 参数量和普通 4D FFN 大致持平。
```

如果问“现代 LLM block 和原始 Transformer block 有哪些常见差异”，可以回答：

```text
现代 decoder-only LLM 常用 Pre-Norm 结构，归一化层使用 RMSNorm，位置编码使用 RoPE，FFN 使用 SwiGLU 或类似门控结构，同时还可能配合 KV Cache、GQA 和 FlashAttention 等工程优化。
```

---

### 十六、小练习

#### 练习 1

手写 RMSNorm，并和 `nn.LayerNorm` 比较输出均值是否一定接近 0。

#### 练习 2

把 RMSNorm 的 `keepdim=True` 去掉，观察 shape 报错或广播异常。

#### 练习 3

实现 SwiGLU，打印 `gate_proj(x)`、`up_proj(x)` 和输出 shape。

#### 练习 4

比较普通 FFN 和 SwiGLU 在 `D=1024` 时的参数量。

#### 练习 5

把 `LLMTransformerBlock` 堆叠 4 层，确认最终输出 shape 仍是 `[B, T, D]`。

---

### 本讲总结

这一讲实现了 RMSNorm 和 SwiGLU。

核心结论如下：

1. RMSNorm 不减均值，只用 hidden 维度的 RMS 做尺度归一化。
2. RMSNorm 通常只有可学习 `weight`，没有 bias。
3. 低精度训练中，RMSNorm 常用 float32 计算归一化再转回原 dtype。
4. SwiGLU 是带门控的 FFN，由 gate、up、down 三个投影组成。
5. SwiGLU 的核心是 `SiLU(gate_proj(x)) * up_proj(x)`。
6. 为了让参数量接近普通 4D FFN，SwiGLU hidden_dim 常设为约 `8D/3`。
7. `Pre-RMSNorm + RoPE Attention + SwiGLU FFN + Residual` 是现代 LLM block 的常见主干。

至此，第三册第二部分“Transformer 组件实战”正文第一版完成。
