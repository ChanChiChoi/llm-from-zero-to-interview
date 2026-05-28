# 第一章：PyTorch 张量基础

PyTorch 的核心对象是 tensor。大模型工程里的输入 token、embedding、hidden states、attention scores、logits、loss mask、梯度和参数，本质上都是不同 shape、dtype、device 和 stride 的 tensor。

很多人学 PyTorch 时只记住了几个 API，但一到面试或调试训练脚本，就会卡在更底层的问题上：为什么 `view` 报错、为什么广播后 shape 不对、为什么 `matmul` 得到的维度和预期不同、为什么 `transpose` 后要 `contiguous`、为什么同一段代码在 CPU 上能跑到 GPU 上就报 device mismatch。

本章目标不是罗列 PyTorch API，而是建立大模型工程中最常用的 tensor 基础：shape、dtype、device、broadcast、矩阵乘法、einsum、索引、reshape、view、transpose、contiguous，以及常见调试方法。

## 1.1 Tensor 是什么

Tensor 可以理解为多维数组，但在深度学习框架里，它不只是数组。一个 PyTorch tensor 至少包含几类关键信息：

1. 数据本身。
2. Shape，也就是每个维度的长度。
3. Dtype，也就是数据类型。
4. Device，也就是数据存放在 CPU 还是 GPU。
5. Stride，也就是如何从底层存储中按维度访问数据。
6. 是否参与梯度计算。

例如：

```python
import torch

x = torch.randn(2, 3, 4)

print(x.shape)   # torch.Size([2, 3, 4])
print(x.dtype)   # torch.float32
print(x.device)  # cpu
print(x.stride())
```

在大模型中，一个最常见的 hidden states shape 是：

```text
X: [B, T, d]
```

其中：

1. `B` 是 batch size。
2. `T` 是 sequence length。
3. `d` 是 hidden size。

如果你能稳定地推导每一步 tensor 的 shape，大部分模型实现和 debug 都会变简单。

## 1.2 Shape 是第一优先级

写深度学习代码时，最重要的习惯是先看 shape，再看数值。

以语言模型为例，输入通常是 token id：

```text
input_ids: [B, T]
```

经过 embedding 层后：

```text
hidden_states: [B, T, d]
```

经过 LM head 后：

```text
logits: [B, T, vocab_size]
```

训练时 labels 通常是：

```text
labels: [B, T]
```

如果做 next-token prediction，常见处理是：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

此时：

```text
shift_logits: [B, T - 1, vocab_size]
shift_labels: [B, T - 1]
```

很多 loss 报错都来自 shape 没有对齐。比如 `cross_entropy` 通常希望输入是 `[N, C]`，标签是 `[N]`，因此语言模型训练里常写成：

```python
loss = torch.nn.functional.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)
```

这里的含义是把 `[B, T - 1, vocab_size]` 展平为 `[B * (T - 1), vocab_size]`，把 `[B, T - 1]` 展平为 `[B * (T - 1)]`。

面试回答：

```text
我写 PyTorch 模型时会优先跟踪 shape。语言模型里 input_ids 是 [B, T]，embedding 后是 [B, T, d]，LM head 后是 [B, T, vocab]。算 cross entropy 前通常要把 logits reshape 成 [B*T, vocab]，labels reshape 成 [B*T]。很多训练 bug 本质上是 shape 没有按 loss 或 matmul 的要求对齐。
```

## 1.3 Dtype：精度、性能和数值稳定性

Tensor 的 dtype 决定数值精度和计算性能。

常见 dtype：

1. `torch.float32`：默认浮点类型，精度较高，显存占用较大。
2. `torch.float16`：半精度，显存更省，吞吐更高，但更容易溢出或下溢。
3. `torch.bfloat16`：常用于大模型训练，动态范围接近 fp32，精度低于 fp32。
4. `torch.int64`：常用于 token id、labels、索引。
5. `torch.bool`：常用于 mask。

示例：

```python
input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
mask = torch.tensor([[True, True, False]])
x = torch.randn(2, 3, dtype=torch.float32)
y = x.to(torch.bfloat16)
```

大模型工程中要特别注意：

1. Embedding 输入必须是整数 token id，通常是 `torch.long`。
2. Attention mask 通常是 bool 或可以加到 logits 上的浮点 mask。
3. 模型参数和激活可能是 fp32、fp16 或 bf16。
4. Loss 计算、softmax、归一化等操作可能需要更稳定的 dtype。

常见错误：

```text
RuntimeError: expected scalar type Long but found Float
```

这类错误通常说明你把浮点 tensor 传给了需要整数索引的模块，例如 `nn.Embedding`。

## 1.4 Device：CPU 和 GPU 必须一致

Tensor 的 device 表示它在哪个设备上。

```python
x = torch.randn(2, 3)
print(x.device)  # cpu

if torch.cuda.is_available():
    x = x.cuda()
    print(x.device)  # cuda:0
```

PyTorch 不会自动在 CPU 和 GPU 之间搬运参与同一次计算的 tensor。下面代码会报错：

```python
x = torch.randn(2, 3, device="cuda")
y = torch.randn(2, 3, device="cpu")
z = x + y
```

正确做法是让参与计算的 tensor 在同一个 device 上：

```python
y = y.to(x.device)
z = x + y
```

训练脚本里常见写法：

```python
device = next(model.parameters()).device
batch = {k: v.to(device) for k, v in batch.items()}
```

注意，不是所有 batch 字段都一定是 tensor。如果 batch 里混有字符串、列表或元数据，需要先判断类型：

```python
batch = {
    k: v.to(device) if torch.is_tensor(v) else v
    for k, v in batch.items()
}
```

## 1.5 创建 Tensor 的常用方式

常见创建方式：

```python
torch.tensor([1, 2, 3])
torch.zeros(2, 3)
torch.ones(2, 3)
torch.randn(2, 3)
torch.arange(0, 10)
torch.empty(2, 3)
```

需要注意 `torch.tensor` 和 `torch.as_tensor` 的区别：

```python
data = [1, 2, 3]
x = torch.tensor(data)
y = torch.as_tensor(data)
```

`torch.tensor` 通常会拷贝数据并创建新 tensor。`torch.as_tensor` 在某些输入类型下会尽量共享数据，避免额外拷贝。

`torch.empty` 只分配内存，不初始化数值：

```python
x = torch.empty(2, 3)
```

它的内容是不确定的，不能当作全 0 使用。只有你马上会覆盖所有元素时，才适合用 `empty`。

## 1.6 索引与切片

Tensor 支持类似 NumPy 的索引和切片。

```python
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

print(x[0].shape)        # [3, 4]
print(x[:, 1].shape)     # [2, 4]
print(x[:, :, -1].shape) # [2, 3]
```

语言模型里常见切片：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

含义是：用第 `0` 到第 `T-2` 个位置的 logits 预测第 `1` 到第 `T-1` 个 token。

如果要保留维度，可以使用范围切片而不是单点索引：

```python
x[:, 0, :].shape    # [B, d]
x[:, 0:1, :].shape  # [B, 1, d]
```

这在拼接、广播和 attention mask 处理中很常见。

## 1.7 Broadcasting：自动扩展维度

Broadcasting 是 PyTorch 自动对齐 shape 的规则。它允许某些不同 shape 的 tensor 做逐元素运算。

规则可以简化为：从右往左对齐维度，每一维要么相等，要么其中一个是 1，要么其中一个维度不存在。

示例：

```python
x = torch.randn(2, 3, 4)
bias = torch.randn(4)
y = x + bias
print(y.shape)  # [2, 3, 4]
```

这里 `bias: [4]` 会被看成 `[1, 1, 4]`，再广播到 `[2, 3, 4]`。

LayerNorm 中常见类似行为：

```text
x:      [B, T, d]
weight: [d]
bias:   [d]
```

`weight` 和 `bias` 会沿着 batch 和 sequence 维度广播。

Attention mask 里也经常使用 broadcasting：

```python
scores = torch.randn(2, 4, 8, 8)      # [B, H, T, T]
mask = torch.ones(2, 1, 1, 8).bool()  # [B, 1, 1, T]
scores = scores.masked_fill(~mask, float("-inf"))
```

`mask` 会沿着 head 维和 query 维广播。

常见坑是某个维度刚好可以广播，但语义不对。例如你想让 mask 对齐 token 维，却把它写成了 `[B, T, 1]`，代码可能能跑，但实际 mask 的方向错了。

## 1.8 unsqueeze、squeeze 和 expand

为了让 tensor 满足 broadcasting，需要经常增加或删除长度为 1 的维度。

```python
x = torch.randn(2, 3)

x1 = x.unsqueeze(1)
print(x1.shape)  # [2, 1, 3]

x2 = x1.squeeze(1)
print(x2.shape)  # [2, 3]
```

`unsqueeze(dim)` 会在指定位置插入一个长度为 1 的维度。

`squeeze(dim)` 会删除指定位置上长度为 1 的维度。

注意不要随便使用无参数 `squeeze()`：

```python
x = torch.randn(1, 8, 1, 64)
y = x.squeeze()
print(y.shape)  # [8, 64]
```

它会删除所有长度为 1 的维度，可能把 batch 维也删掉。训练脚本中更安全的写法是显式指定维度：

```python
y = x.squeeze(2)
```

`expand` 可以创建广播视图：

```python
x = torch.randn(1, 3)
y = x.expand(4, 3)
print(y.shape)  # [4, 3]
```

`expand` 不会真正复制数据，而是通过 stride 让多个位置指向同一份底层存储。因此不能把它理解成物理拷贝。如果需要真实拷贝，可以使用 `repeat`，但会占更多内存。

## 1.9 matmul：大模型里最常见的计算

矩阵乘法是 Transformer 的核心。

二维矩阵乘法规则：

```text
A: [m, n]
B: [n, p]
C = A @ B
C: [m, p]
```

PyTorch 中：

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B
print(C.shape)  # [3, 5]
```

对更高维 tensor，`torch.matmul` 会把最后两维当作矩阵维度，前面的维度按 batch 维处理并尝试 broadcast。

```python
Q = torch.randn(2, 4, 8, 64)  # [B, H, T, d_h]
K = torch.randn(2, 4, 8, 64)  # [B, H, T, d_h]

scores = Q @ K.transpose(-2, -1)
print(scores.shape)  # [2, 4, 8, 8]
```

这里：

1. `Q` 的最后两维是 `[T, d_h]`。
2. `K.transpose(-2, -1)` 的最后两维是 `[d_h, T]`。
3. 相乘后得到 `[T, T]`。
4. 前面的 `[B, H]` 作为 batch 维保留下来。

这就是 attention score 的 shape 来源。

## 1.10 bmm、mm 和 matmul 的区别

PyTorch 有多个矩阵乘法 API：

1. `torch.mm`：只处理两个二维矩阵。
2. `torch.bmm`：处理两个三维 tensor 的 batch 矩阵乘法。
3. `torch.matmul`：更通用，支持一维、二维和高维 batch 矩阵乘法。
4. `@`：通常等价于调用 matmul 语义。

示例：

```python
A = torch.randn(10, 3, 4)
B = torch.randn(10, 4, 5)
C = torch.bmm(A, B)
print(C.shape)  # [10, 3, 5]
```

`bmm` 不做 batch 维广播，要求 batch size 相同。`matmul` 更灵活，能处理更多 broadcasting 场景。

工程里常用建议：

1. 写普通二维矩阵乘法时，用 `@` 或 `matmul`。
2. 写 attention 这种高维 batch 矩阵乘法时，用 `@` 或 `matmul`。
3. 如果明确是三维 batch 矩阵乘法，并且不需要广播，可以用 `bmm`。

## 1.11 einsum：把维度关系写清楚

`einsum` 可以用字符串显式描述维度之间的计算关系，适合表达复杂张量运算。

例如矩阵乘法：

```python
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.einsum("mn,np->mp", A, B)
print(C.shape)  # [3, 5]
```

Attention score 可以写成：

```python
Q = torch.randn(2, 4, 8, 64)  # [B, H, T, D]
K = torch.randn(2, 4, 8, 64)  # [B, H, T, D]

scores = torch.einsum("bhtd,bhsd->bhts", Q, K)
print(scores.shape)  # [2, 4, 8, 8]
```

这里 `t` 表示 query position，`s` 表示 key position，`d` 是被求和消掉的 head dimension。

Value 加权求和可以写成：

```python
attn = torch.softmax(scores, dim=-1)  # [B, H, T, S]
V = torch.randn(2, 4, 8, 64)          # [B, H, S, D]

context = torch.einsum("bhts,bhsd->bhtd", attn, V)
print(context.shape)  # [2, 4, 8, 64]
```

`einsum` 的优点是表达清晰，尤其适合面试手写和解释维度。缺点是字符串写错时不如普通矩阵乘法直观，而且在某些场景下性能需要实际 profiling。

## 1.12 reshape、view 和 flatten

改变 tensor shape 时常用 `reshape`、`view` 和 `flatten`。

```python
x = torch.randn(2, 3, 4)

y = x.reshape(6, 4)
z = x.view(6, 4)
w = x.flatten(0, 1)
```

它们都可以把 `[2, 3, 4]` 变成 `[6, 4]`，但底层语义不同。

`view` 要求 tensor 在内存布局上兼容目标 shape。`reshape` 更宽松，如果不能返回 view，可能会创建拷贝。

常见错误：

```python
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)
z = y.view(2, 12)  # 可能报错
```

原因是 `transpose` 后 tensor 通常不再 contiguous，`view` 无法按目标 shape 解释底层存储。

更稳妥的写法：

```python
z = y.reshape(2, 12)
```

或者显式让 tensor 连续：

```python
z = y.contiguous().view(2, 12)
```

工程建议：

1. 如果只是想安全改变 shape，优先用 `reshape`。
2. 如果明确知道 tensor 是 contiguous，并且想避免潜在拷贝，可以用 `view`。
3. 如果要合并连续维度，用 `flatten(start_dim, end_dim)` 可读性更好。

## 1.13 transpose、permute 和 contiguous

`transpose` 用来交换两个维度：

```python
x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)
print(y.shape)  # [2, 4, 3]
```

`permute` 可以重新排列多个维度：

```python
x = torch.randn(2, 3, 4, 5)
y = x.permute(0, 2, 1, 3)
print(y.shape)  # [2, 4, 3, 5]
```

多头注意力中常见维度变换：

```python
B, T, d_model = 2, 8, 64
num_heads = 4
head_dim = d_model // num_heads

x = torch.randn(B, T, d_model)
q = x.reshape(B, T, num_heads, head_dim).transpose(1, 2)

print(q.shape)  # [B, H, T, d_h]
```

这一步把 `[B, T, d_model]` 拆成 `[B, T, H, d_h]`，再交换维度得到 attention 更方便计算的 `[B, H, T, d_h]`。

计算完 attention 后，通常要变回 `[B, T, d_model]`：

```python
context = torch.randn(B, num_heads, T, head_dim)
out = context.transpose(1, 2).contiguous().view(B, T, d_model)
print(out.shape)  # [B, T, d_model]
```

这里的 `contiguous()` 很关键。`transpose` 只是改变 stride 视图，不一定重新排列底层内存。后续使用 `view` 前，常常需要先调用 `contiguous()`。

面试回答：

```text
transpose 和 permute 通常不会真的复制数据，而是改变 tensor 的 stride，因此结果可能不是 contiguous。view 要求内存布局兼容目标 shape，所以在 transpose 后经常需要 contiguous().view(...)，或者直接使用 reshape，让 PyTorch 在必要时处理拷贝。
```

## 1.14 Stride：理解 contiguous 的关键

Stride 表示沿某个维度移动一步，在底层存储中要跳过多少个元素。

```python
x = torch.arange(2 * 3 * 4).reshape(2, 3, 4)
print(x.stride())  # 通常是 (12, 4, 1)

y = x.transpose(1, 2)
print(y.shape)     # [2, 4, 3]
print(y.stride())  # 通常是 (12, 1, 4)
```

原始 `x` 中，最后一维是连续的。`transpose` 后，shape 变了，但底层数据没有真正重排，只是 stride 变了。

这也是为什么有些 tensor 看起来 shape 没问题，但 `view` 会失败。因为 `view` 想用新的 shape 直接解释同一块连续内存，而当前 stride 不满足要求。

判断 tensor 是否连续：

```python
print(x.is_contiguous())
print(y.is_contiguous())
```

让 tensor 变连续：

```python
z = y.contiguous()
```

注意，`contiguous()` 可能触发真实内存拷贝。在性能敏感代码中，不要无脑到处加；但在模型原型、面试手写和调试阶段，它是解决 layout 问题的常见手段。

## 1.15 cat 和 stack

`torch.cat` 和 `torch.stack` 都能拼接 tensor，但语义不同。

`cat` 在已有维度上拼接：

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

c = torch.cat([a, b], dim=0)
print(c.shape)  # [4, 3]

d = torch.cat([a, b], dim=1)
print(d.shape)  # [2, 6]
```

`stack` 会新增一个维度：

```python
s = torch.stack([a, b], dim=0)
print(s.shape)  # [2, 2, 3]
```

数据集 collate 时，常用 `stack` 把多个样本堆成 batch：

```python
samples = [torch.randn(10) for _ in range(4)]
batch = torch.stack(samples, dim=0)
print(batch.shape)  # [4, 10]
```

如果每个样本长度不同，不能直接 stack，需要先 padding 或自定义 collate function。

## 1.16 mask 的常见写法

Mask 是大模型代码里最容易出错的 tensor 之一。

常见 attention scores：

```text
scores: [B, H, T, T]
```

Padding mask 可能来自输入：

```text
attention_mask: [B, T]
```

其中 `1` 表示有效 token，`0` 表示 padding。为了加到 scores 上，常见转换方式是：

```python
attention_mask = torch.tensor([[1, 1, 1, 0, 0]])  # [B, T]
mask = attention_mask[:, None, None, :].bool()    # [B, 1, 1, T]

scores = torch.randn(1, 4, 5, 5)
scores = scores.masked_fill(~mask, float("-inf"))
```

Causal mask 用于禁止当前位置看未来 token：

```python
T = 5
causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
print(causal_mask.shape)  # [T, T]
```

扩展到 attention scores：

```python
scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))
```

真实工程里通常会把 padding mask 和 causal mask 结合起来。关键是始终确认 mask 的维度语义：哪个维度是 batch，哪个维度是 query position，哪个维度是 key position。

## 1.17 in-place 操作的风险

PyTorch 中以下划线结尾的方法通常是 in-place 操作：

```python
x.add_(1)
x.masked_fill_(mask, 0)
```

In-place 操作会直接修改原 tensor，可能节省内存，但也可能影响 autograd 或后续复用。

例如：

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
# 某些复杂场景下，对计算图中还需要的 tensor 做 in-place 修改会导致 backward 报错。
```

在训练代码中，除非你明确知道某个 in-place 操作不会破坏计算图，否则优先使用非 in-place 写法。

常见建议：

1. 面试手写时，少用 in-place，避免引入额外解释成本。
2. 写模型 forward 时，谨慎修改会参与梯度计算的中间激活。
3. 优化显存时，可以有意识地使用 in-place，但需要配合测试和 anomaly detection。

## 1.18 调试 Tensor 的实用清单

当 PyTorch 代码报错时，不要只看最后一行错误。优先打印关键 tensor 的元信息：

```python
def debug_tensor(name, x):
    if not torch.is_tensor(x):
        print(name, type(x))
        return
    print(
        name,
        "shape=", tuple(x.shape),
        "dtype=", x.dtype,
        "device=", x.device,
        "contiguous=", x.is_contiguous(),
        "stride=", x.stride(),
    )
```

重点检查：

1. Shape 是否符合预期。
2. Dtype 是否符合模块要求。
3. Device 是否一致。
4. 是否有 NaN 或 Inf。
5. Mask 的方向是否正确。
6. `view` 前是否 contiguous。

检查 NaN 和 Inf：

```python
torch.isnan(x).any()
torch.isinf(x).any()
torch.isfinite(x).all()
```

定位异常值：

```python
bad = ~torch.isfinite(x)
print(bad.nonzero())
```

对大模型训练来说，tensor debug 的核心不是“会不会调用 API”，而是能不能快速判断错误属于 shape、dtype、device、layout、mask 还是数值稳定性问题。

## 1.19 手写一个简化 attention shape 流程

下面用最少代码串起本章最重要的 tensor 操作。

```python
import math
import torch

B, T, d_model = 2, 8, 64
num_heads = 4
head_dim = d_model // num_heads

x = torch.randn(B, T, d_model)
Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)

q = x @ Wq  # [B, T, d_model]
k = x @ Wk  # [B, T, d_model]
v = x @ Wv  # [B, T, d_model]

q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]
k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]
v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)     # [B, H, T, T]

causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

attn = torch.softmax(scores, dim=-1)                       # [B, H, T, T]
context = attn @ v                                         # [B, H, T, d_h]

out = context.transpose(1, 2).contiguous().view(B, T, d_model)
print(out.shape)                                           # [B, T, d_model]
```

这段代码覆盖了本章最核心的点：

1. 线性投影是矩阵乘法。
2. 多头拆分依赖 `reshape` 和 `transpose`。
3. Attention scores 来自 `q @ k.transpose(-2, -1)`。
4. Causal mask 依赖 broadcasting。
5. 多头合并时需要理解 `contiguous().view(...)`。

真正写 Transformer 时会使用 `nn.Linear`、更完整的 mask、dropout、输出投影和更严谨的初始化，但 tensor shape 主线是不变的。

## 1.20 常见面试题

问题一：`view` 和 `reshape` 有什么区别？

答：

```text
view 要求 tensor 的内存布局和目标 shape 兼容，通常要求 contiguous 或 stride 可以直接解释；reshape 更灵活，如果不能返回 view，可能会创建拷贝。工程里如果只是想安全改 shape，常用 reshape；如果追求明确的 view 语义，就要确认 tensor 是否 contiguous。
```

问题二：为什么 `transpose` 后经常要 `contiguous`？

答：

```text
transpose 通常不会重排底层数据，而是改变 stride，所以结果可能不是 contiguous。后续如果用 view 合并维度，可能因为内存布局不兼容报错。因此常见写法是 transpose 后先 contiguous 再 view，或者直接使用 reshape。
```

问题三：PyTorch broadcasting 规则是什么？

答：

```text
Broadcasting 会从右往左对齐维度，每一维要么相等，要么其中一个是 1，要么其中一个维度不存在。它常用于 bias、LayerNorm 参数、attention mask 等场景。调试时不仅要看能不能广播，还要确认广播方向是否符合语义。
```

问题四：Attention score 的 shape 为什么是 `[B, H, T, T]`？

答：

```text
多头注意力中 Q 和 K 的 shape 通常是 [B, H, T, d_h]。计算 Q @ K.transpose(-2, -1) 时，最后两维是 [T, d_h] 乘 [d_h, T]，得到 [T, T]，前面的 B 和 H 作为 batch 维保留，所以 score 是 [B, H, T, T]。
```

问题五：`cat` 和 `stack` 有什么区别？

答：

```text
cat 是在已有维度上拼接，不增加新维度；stack 会先新增一个维度，再把多个 tensor 沿这个新维度堆起来。把多个样本组成 batch 时常用 stack；把多个片段沿序列维或特征维拼起来时常用 cat。
```

## 1.21 本章小结

PyTorch 张量基础的重点不是背 API，而是形成稳定的工程判断：

1. 任何模型代码先看 shape。
2. Dtype 决定精度、性能和模块输入要求。
3. Device 必须一致，否则 CPU/GPU 混用会报错。
4. Broadcasting 很强大，但要确认语义方向正确。
5. `matmul` 和 `einsum` 是理解 attention 的核心工具。
6. `reshape`、`view`、`transpose`、`permute`、`contiguous` 背后是内存布局和 stride。
7. Mask、loss reshape、多头拆分和合并，是大模型工程最常见的 tensor 基础考点。

下一章会在 tensor 基础上进入 autograd，理解 PyTorch 如何构建计算图、保存中间结果、执行 backward，以及为什么 `detach`、`no_grad`、梯度累积和 in-place 操作会影响训练行为。
