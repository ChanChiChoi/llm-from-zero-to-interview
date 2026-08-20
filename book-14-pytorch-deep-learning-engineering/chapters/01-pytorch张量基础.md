# 第一章：PyTorch 张量基础

PyTorch 的核心对象是 tensor。大模型工程里的输入 token、embedding、hidden states、attention scores、logits、loss mask、梯度和参数，本质上都是不同 shape、dtype、device 和 stride 的 tensor。

很多人学 PyTorch 时只记住了几个 API，但一到真实训练或调试脚本，就会卡在更底层的问题上：为什么 `view` 报错、为什么广播后 shape 不对、为什么 `matmul` 得到的维度和预期不同、为什么 `transpose` 后要 `contiguous`、为什么同一段代码在 CPU 上能跑到 GPU 上就报 device mismatch。

本章目标不是罗列 PyTorch API，而是建立大模型工程中最常用的 tensor 基础：shape、dtype、device、broadcast、矩阵乘法、einsum、索引、reshape、view、transpose、contiguous，以及常见调试方法。

## 1.0 本章范围与资料

本章以 PyTorch 官方文档为 API 语义的主要依据，参考张量属性、广播、`torch.matmul`、`torch.bmm`、`torch.einsum`、`Tensor.view`、`torch.reshape`、`Tensor.contiguous`、`Tensor.stride` 和 `torch.nn.functional.cross_entropy` 文档，并结合前序 Transformer、attention、LM loss、mask 和数学基础章节的 shape 推导口径。文末列出可直接核验的链接。

本章只聚焦大模型工程最常见的 tensor 基础：shape、dtype、device、broadcasting、matmul / bmm / einsum、索引切片、view / reshape / flatten、transpose / permute / contiguous、stride、mask、loss reshape 和 tensor debug。它不展开 CUDA kernel、autograd graph、distributed tensor、Tensor Core、编译优化或 profiler 细节；这些会在后续章节中处理。文中的小例子是可复现的教学构造，不等于对任意 PyTorch 版本、GPU 型号或混合精度配置的性能承诺；涉及性能的判断必须在目标环境中复测。

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

### 1.1.1 语言模型的 shape contract

语言模型中最常见的张量主线：

~~~math
input\_ids \in \mathbb{Z}^{B \times T}
~~~

~~~math
X = Emb(input\_ids),\qquad X \in \mathbb{R}^{B \times T \times d}
~~~

LM head 输出：

~~~math
logits \in \mathbb{R}^{B \times T \times V}
~~~

next-token prediction 的 shift：

~~~math
shift\_logits = logits[:,0:T-1,:]
~~~

~~~math
shift\_labels = labels[:,1:T]
~~~

交叉熵前展平：

~~~math
shift\_logits \rightarrow \mathbb{R}^{B(T-1) \times V},\qquad shift\_labels \rightarrow \mathbb{Z}^{B(T-1)}
~~~

多头注意力拆头：

~~~math
d = H d_h
~~~

~~~math
Q,K,V \in \mathbb{R}^{B \times H \times T \times d_h}
~~~

attention score：

~~~math
S = \frac{QK^T}{\sqrt{d_h}},\qquad S \in \mathbb{R}^{B \times H \times T \times T}
~~~

attention mask 常见广播：

~~~math
mask_{pad}: [B,T] \rightarrow [B,1,1,T]
~~~

~~~math
mask_{causal}: [T,T] \rightarrow [1,1,T,T]
~~~

broadcasting 从右往左对齐维度：

~~~math
[B,T,d] + [d] \rightarrow [B,T,d]
~~~

LoRA 或线性层常见矩阵乘法：

~~~math
X [B,T,d_{in}] \times W [d_{in},d_{out}] \rightarrow Y [B,T,d_{out}]
~~~

这些 shape 还不是完整的正确性证明。通常要求 `B>0`、`T>0`、`d>0`、`V>0`、`H>0`，并且 `d=H*d_h`；next-token loss 若要产生至少一个位置，还需要 `T>=2`。每一个符号都对应一个约束：`V` 必须和词表索引范围一致，`labels` 的整数值必须落在 `[0, V)` 或等于明确约定的 `ignore_index`，而 mask 的最后一维必须确实对应 key position。工程上应把这些约束写成断言或单元测试，而不是只把 shape 打印出来。特别是 `T=1` 或一个 batch 的所有 label 都被忽略时，`mean` reduction 没有有效监督位置，不能把产生的 `NaN` 当成模型质量信号。

`view`、`reshape` 和 `contiguous` 的关系可以这样理解：

1. `view`：只在当前 stride 能兼容目标 shape 时直接改视图。
2. `reshape`：能返回 view 就返回 view，不行时可能创建拷贝。
3. `transpose` / `permute`：通常只改 stride，不重排底层数据，因此结果常常不是 contiguous。

这些公式不是为了背维度，而是为了形成一个调试顺序：先看 shape，再看 dtype，再看 device，再看 stride / contiguous，最后检查 mask 方向和数值稳定性。若只证明“代码能运行”，仍可能留下广播方向错、padding 被当成真实 token 或无效位置参与 loss 等语义错误。

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

很多 loss 报错都来自 shape 没有对齐。`cross_entropy` 的分类输入把类别放在最后一维之外的“类别维”上；最容易理解的二维形式是 logits 为 `[N, C]`、标签为 `[N]`。因此语言模型训练里常把时间和 batch 合并成样本维：

```python
loss = torch.nn.functional.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
)
```

这里的含义是把 `[B, T - 1, vocab_size]` 展平为 `[B * (T - 1), vocab_size]`，把 `[B, T - 1]` 展平为 `[B * (T - 1)]`。如果 batch 中有 padding，还要同时把无效位置标记成 `ignore_index`，否则模型会被要求预测人为补上的 token。也就是说，shape 对齐只解决“张量能否传入 loss”，并不自动保证监督信号的语义正确。

建议在模型边界写出显式检查。下面的断言把维度约定转成了可执行的 contract：

```python
def check_lm_contract(input_ids, logits, labels, vocab_size, ignore_index):
    assert input_ids.ndim == 2
    assert labels.shape == input_ids.shape
    assert logits.shape[:2] == input_ids.shape
    assert logits.size(-1) == vocab_size
    assert input_ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids.numel() > 0
    assert vocab_size > 0
    assert int(input_ids.min()) >= 0
    assert int(input_ids.max()) < vocab_size
    valid = labels.ne(ignore_index)
    assert valid.any()
    assert int(labels[valid].min()) >= 0
    assert int(labels[valid].max()) < vocab_size
```

这类检查不应只在调试时临时打印。数据格式、词表版本或 padding 策略一旦变化，contract 就是最早暴露错误的地方；当模型进入大规模训练后，尽早失败通常比训练数小时后才发现 loss 语义错更便宜。

## 1.3 Dtype：精度、性能和数值稳定性

Tensor 的 dtype 决定数值精度和计算性能。

常见 dtype：

1. `torch.float32`：默认浮点类型，精度较高，显存占用较大。
2. `torch.float16`：半精度，显存更省，吞吐更高，但更容易溢出或下溢。
3. `torch.bfloat16`：常用于大模型训练，动态范围接近 fp32，精度低于 fp32。
4. `torch.int64`：常用于 token id、labels、索引。
5. `torch.bool`：常用于 mask。

dtype 还决定算子是否能够执行，以及不同 dtype 混合时结果会被提升到什么类型。整数 token id 不能直接拿来做浮点矩阵乘法；`bool` mask 适合做逻辑筛选，但如果把它转换成加性 mask，就必须使用和 score 兼容的浮点类型。不同版本、设备和算子可能采用不同的 promotion 或内部累积策略，因此不要仅凭“输入是 bf16”就断言整个算子都以 bf16 完成。

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

一个典型的数值边界是 attention 的缩放和 softmax。半精度 score 如果绝对值过大，可能在指数运算前后出现溢出；如果所有 key 都被 mask，整行 logits 都可能是负无穷，softmax 便会产生 NaN。选择 `-1e9` 还是负无穷，不是脱离算子和 dtype 的固定教条：要确认后续 kernel、dtype、是否存在全屏蔽行，以及框架版本对 mask 的处理方式。工程实现应在单元测试中覆盖“至少一个有效 key”和“全是 padding”的边界，而不是只测正常句子。

常见错误：

```text
RuntimeError: expected scalar type Long but found Float
```

这类错误通常说明你把浮点 tensor 传给了需要整数索引的模块，例如 `nn.Embedding`。

另一个常见误区是把 `.to(dtype)` 当成数值校验。它只执行转换，不会检查浮点值是否适合作为索引，也不会保证转换后的数值仍然是合法 token。数据进入 embedding 或 loss 前，应分别检查 dtype、取值范围和特殊 token 约定。

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

模型中的常量也有 device 问题。直接在 `forward` 中写 `torch.arange(T)`、`torch.ones(...)` 或一个 Python 数字，可能产生 CPU tensor 或与输入 dtype 不一致的 tensor。更稳妥的做法是使用输入的 device 和 dtype，或者把长期复用的 mask 注册为 buffer：

```python
class CausalMask(torch.nn.Module):
    def __init__(self, max_length):
        super().__init__()
        mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, length, like):
        if not 0 <= length <= self.mask.size(0):
            raise ValueError("length exceeds the preallocated mask")
        return self.mask[:length, :length].to(device=like.device)
```

`register_buffer` 的要点不是“把 mask 变成参数”，而是让它跟随模块迁移到目标 device，并参与模块状态管理；它不应出现在优化器的可训练参数列表中。若 mask 的 dtype 需要与某个运算严格匹配，应在使用点明确转换，而不是依赖隐式 promotion。

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

由于多个逻辑位置可能指向同一存储，不能对 `expand` 返回的视图做依赖独立元素的原地写入；这类写入可能报错，也可能产生难以察觉的别名语义。需要修改每个展开位置时，应先 `clone()`，或改用 `repeat()`，并在实际路径中衡量额外内存。

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

`einsum` 的优点是把“哪些维度保留、哪些维度求和”写在表达式里，适合教学、原型和复杂的多路收缩。它不是性能保证：字符串写错时可能得到形状正确但语义错误的结果，某些路径也可能比专门的矩阵乘法慢。使用它时至少应做两件事：用一个小尺寸输入和 `matmul` 对照数值，用 profiler 在目标设备和真实尺寸上比较耗时、显存和 kernel 行为。

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

原因不是“所有非 contiguous tensor 都不能 view”，而是目标 shape 必须满足当前 stride 的可合并条件。`transpose` 后通常不满足这个条件，所以该例可能报错；也存在某些切片或维度变换仍可 `view` 的情况。把规则简单记成“非 contiguous 一定不能 view”会在更复杂的 layout 中误导排查。

更稳妥的写法：

```python
z = y.reshape(2, 12)
```

或者显式让 tensor 连续：

```python
z = y.contiguous().view(2, 12)
```

工程上要同时考虑语义和拷贝成本：

1. 如果只是想安全改变 shape，优先用 `reshape`，但要意识到它在必要时可能复制数据。
2. 如果明确知道 tensor 是 contiguous，并且想表达“不得复制”的视图语义，可以用 `view` 并配合 `is_contiguous()` 或 stride 测试。
3. 如果要合并连续维度，用 `flatten(start_dim, end_dim)` 可读性更好。
4. 在性能敏感路径中，对疑似复制的操作记录内存峰值和耗时；不能把 `reshape` 视为永远零拷贝。

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

注意，`contiguous()` 可能触发真实内存拷贝。在性能敏感代码中，不要无脑到处加；但在模型原型和调试阶段，它是解决 layout 问题的常见手段。更重要的是区分两个事实：stride 描述访问方式，contiguous 描述是否符合某个约定的连续布局；它们都不说明 tensor 的数值是否正确，也不说明某个 kernel 一定更快。某些算子可以直接处理非连续输入，另一些算子会在内部复制；最终成本要通过目标路径测量。

可以用一个小实验观察“同一数值、不同 layout”与“显式复制”的区别：

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
z = y.contiguous()

assert torch.equal(y, z)
assert y.shape == z.shape
assert not y.is_contiguous()
assert z.is_contiguous()
assert y.stride() != z.stride()
```

这里 `z` 与 `y` 的值相同，但访问路径不同。排查 view 错误时，应先确认是否只需要一个 view，还是确实需要一份连续存储；排查数值错误时，则应比较值和索引语义，不能用 `is_contiguous()` 代替数值验证。

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

如果每个样本长度不同，不能直接 stack，需要先 padding 或自定义 collate function。padding 后必须同时产出有效 token mask 或 labels mask；否则“为了得到矩形 batch”这一步会悄悄改变 loss 和 attention 的监督对象。

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

真实工程里通常会把 padding mask 和 causal mask 结合起来。关键是始终确认 mask 的维度语义：哪个维度是 batch，哪个维度是 query position，哪个维度是 key position。上面的组合 mask 形状是 `[B, 1, T, T]`：padding mask 限制 key 位置，causal mask 同时限制 query-key 的相对位置。如果还要屏蔽 query 侧的 padding，需要另加 query mask；只屏蔽 key 侧并不会阻止 padding query 产生输出。

一个容易被忽略的边界是全屏蔽行。对右 padding 的 batch，若把 padding query 也送进 attention，某些行可能没有任何合法 key。实现可以在进入 softmax 前保证每个 query 至少有一个合法 key，也可以在 softmax 后把无效 query 的输出清零；选择哪种策略取决于模型的 padding 约定和 kernel。测试时应专门构造全 padding、长度为 1 和 batch 内长度不一致的样本，并检查输出与 loss 都是有限值。

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

更稳妥的原则是先说明别名关系，再决定是否原地修改。`y = x` 只增加一个 Python 引用；`y = x.clone()` 才创建独立存储；`detach()` 切断梯度关系但仍可能与原 tensor 共享存储。若一个 tensor 还会被计算图、残差分支或日志线程使用，对它做 in-place 修改就可能导致反向结果错误或直接报错。优化显存时可以使用原地操作，但必须用梯度检查、有限值检查和目标 batch 的回归测试证明它没有改变语义。

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

## 1.19 最小可运行 PyTorch 张量审计 demo

下面用一段 demo 串起本章最重要的 tensor 操作：shape、dtype、device、broadcasting、matmul、einsum、mask、reshape、view、contiguous、stride、stack 和语言模型 loss reshape。

```python
import math
import torch
import torch.nn.functional as F


torch.manual_seed(7)

B, T, d_model = 2, 5, 16
num_heads = 4
vocab_size = 32
head_dim = d_model // num_heads

input_ids = torch.tensor(
    [
        [1, 5, 9, 2, 0],
        [1, 7, 8, 0, 0],
    ],
    dtype=torch.long,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)

embedding_table = torch.randn(vocab_size, d_model, device=device)
x = embedding_table[input_ids]                         # [B, T, d_model]
bias = torch.arange(d_model, device=device).float() / 1000
x_with_bias = x + bias                                 # [B, T, d_model]

Wq = torch.randn(d_model, d_model)
Wk = torch.randn(d_model, d_model)
Wv = torch.randn(d_model, d_model)
Wq = Wq.to(device)
Wk = Wk.to(device)
Wv = Wv.to(device)

q = x @ Wq  # [B, T, d_model]
k = x @ Wk  # [B, T, d_model]
v = x @ Wv  # [B, T, d_model]

q = q.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]
k = k.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]
v = v.reshape(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, d_h]

scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)     # [B, H, T, T]
scores_einsum = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(head_dim)
raw_scores = scores

padding_mask = input_ids.ne(0)                             # [B, T]
causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
combined_mask = padding_mask[:, None, None, :] & causal_mask[None, None, :, :]
query_mask = padding_mask[:, None, :, None]
valid_query = query_mask.any(dim=-1, keepdim=True)
scores = scores.masked_fill(~combined_mask, -1e9)
scores = torch.where(valid_query, scores, torch.zeros_like(scores))

attn = torch.softmax(scores, dim=-1)                       # [B, H, T, T]
attn = torch.where(query_mask, attn, torch.zeros_like(attn))
context = attn @ v                                         # [B, H, T, d_h]

transposed = context.transpose(1, 2)                        # [B, T, H, d_h]
view_failed = False
try:
    transposed.view(B, T, d_model)
except RuntimeError:
    view_failed = True

out = transposed.contiguous().view(B, T, d_model)           # [B, T, d_model]

lm_logits = torch.randn(B, T, vocab_size, device=device)
labels = input_ids
shift_logits = lm_logits[:, :-1, :].reshape(-1, vocab_size)
shift_labels = labels[:, 1:].reshape(-1)
loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=0)

sample_a = torch.randn(d_model, device=device)
sample_b = torch.randn(d_model, device=device)
stacked = torch.stack([sample_a, sample_b], dim=0)
concated = torch.cat([sample_a, sample_b], dim=0)

report = {
    "meta": {
        "input_shape": tuple(input_ids.shape),
        "input_dtype": str(input_ids.dtype),
        "hidden_shape": tuple(x.shape),
        "hidden_dtype": str(x.dtype),
        "device": str(x.device),
    },
    "broadcast": {
        "bias_shape": tuple(bias.shape),
        "x_plus_bias_shape": tuple(x_with_bias.shape),
    },
    "attention": {
        "q_shape": tuple(q.shape),
        "scores_shape": tuple(scores.shape),
        "einsum_matches_matmul": torch.allclose(
            scores_einsum,
            raw_scores,
            rtol=1e-5,
            atol=1e-5,
        ),
        "mask_shape": tuple(combined_mask.shape),
        "attn_row_sum": round(float(attn[0, 0, 2].sum().item()), 6),
        "out_shape": tuple(out.shape),
    },
    "layout": {
        "transposed_contiguous": transposed.is_contiguous(),
        "transposed_stride": transposed.stride(),
        "view_failed_before_contiguous": view_failed,
        "out_contiguous": out.is_contiguous(),
    },
    "loss": {
        "shift_logits_shape": tuple(shift_logits.shape),
        "shift_labels_shape": tuple(shift_labels.shape),
        "loss_is_finite": bool(torch.isfinite(loss).item()),
    },
    "cat_stack": {
        "stacked_shape": tuple(stacked.shape),
        "concated_shape": tuple(concated.shape),
    },
    "checks": {
        "embedding_rank_3": x.dim() == 3,
        "broadcast_shape_ok": x_with_bias.shape == x.shape,
        "attention_shape_ok": scores.shape == (B, num_heads, T, T),
        "mask_broadcast_shape_ok": combined_mask.shape == (B, 1, T, T),
        "view_requires_contiguous": view_failed,
        "lm_loss_flatten_ok": shift_logits.shape[0] == shift_labels.shape[0],
    },
}

print(report)
```

这段代码覆盖了本章最核心的点：

1. `input_ids` 是 `torch.long`，embedding 后变成 `[B,T,d_model]` 的浮点 hidden states。
2. `[d_model]` 的 bias 可以 broadcast 到 `[B,T,d_model]`。
3. 线性投影是矩阵乘法，多头拆分依赖 `reshape` 和 `transpose`。
4. Attention scores 可以用 `matmul` 或 `einsum` 表达。
5. Padding mask 和 causal mask 通过 broadcasting 合成 `[B,1,T,T]`。
6. 多头合并时，`transpose` 后通常需要 `contiguous().view(...)`。
7. LM loss 前要把 `[B,T-1,V]` 和 `[B,T-1]` 展平成 `[B(T-1),V]` 和 `[B(T-1)]`。

真正写 Transformer 时会使用 `nn.Linear`、更完整的 mask、dropout、输出投影和更严谨的初始化，但 tensor shape 主线是不变的。

## 1.20 一个完整的张量审计过程

前面的概念在真实项目中通常同时出现。假设训练突然出现 `loss=nan`，或者一个 attention 模块的输出形状正确但效果明显下降，可以按以下顺序审计，而不是只在报错行前面加 `print`。

第一步是确认输入和标签的语义：`input_ids` 是否为整数，padding id 是否和 mask 使用同一个约定，labels 是否发生了正确的 next-token shift，词表范围是否覆盖所有非忽略标签。第二步是沿着模块边界记录 shape，特别是 `[B, T, d]`、`[B, H, T, d_h]` 和 `[B, H, T, T]` 的转换。第三步是检查参与同一计算的 dtype 与 device，避免 CPU 常量、浮点索引和隐式 promotion 混在一起。第四步才是检查 stride、contiguous、mask 数值和有限值。

可以把这个顺序写成一个小型审计函数：

```python
def audit_tensor(name, value):
    if not torch.is_tensor(value):
        return {"name": name, "type": type(value).__name__}
    report = {
        "name": name,
        "shape": tuple(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "contiguous": bool(value.is_contiguous()),
        "stride": tuple(value.stride()),
    }
    if value.is_floating_point() or value.is_complex():
        report["finite"] = bool(torch.isfinite(value).all().item())
    return report
```

审计结果应和输入样本、模型版本、PyTorch 版本一起保存。这样才能回答“哪个边界第一次出现异常”，而不是只知道最后的 loss 已经变成 NaN。对大型训练任务，建议在小 batch、单卡和固定随机种子下先执行同一套 contract；只有基础语义通过后，才增加序列长度、GPU 数量和混合精度。

## 1.21 章末练习

### 练习一：shape contract

给定 `B=3`、`T=17`、`d_model=768`、`H=12`、`V=32000`，写出 embedding、Q/K/V、attention score、合并多头和 LM loss flatten 后的 shape。说明为什么 `d_model` 必须能被 `H` 整除。

### 练习二：广播方向

构造 `scores: [2, 4, 5, 5]` 和 `padding_mask: [2, 5]`，分别写出 key mask 与 query mask 的广播形状。解释为什么 `[B, T, 1]` 不是这个问题的等价写法。

### 练习三：layout 与复制

创建一个 `[2, 3, 4]` tensor，交换两个维度后比较 `view`、`reshape` 和 `contiguous().view` 的行为。记录 shape、stride、是否 contiguous，并说明哪一步可能产生数据拷贝。

### 练习四：dtype 和取值

让一个浮点 tensor 经过 `.long()` 后作为 embedding 索引。分别构造合法整数、负数和超出词表的值，记录错误类型，并说明为什么 dtype 正确仍然不代表索引合法。

### 练习五：mask 的有限值

构造一个 query 没有任何合法 key 的 attention 行，比较使用 `-inf` 和有限大负数时 softmax 的结果。解释为什么“没有 NaN”也不一定表示 mask 语义正确。

### 练习六：cat、stack 与 padding

给出三个不同长度的 token 序列，设计一个 collate 函数返回 padded ids、attention mask 和 labels mask。说明 `stack` 为什么不能直接处理变长序列。

### 练习七：端到端审计

运行本章 demo，把 `num_heads`、padding 位置和 dtype 至少各改一次。为每次改动写出一个预期 contract，并记录实际错误或输出变化。不要只修到代码能跑，要说明修复后哪个语义约束重新成立。

## 1.22 资料与证据边界

本章优先依据 PyTorch 官方 API 和说明文档：

1. 张量属性与基础操作：https://pytorch.org/docs/stable/tensors.html
2. 广播语义：https://pytorch.org/docs/stable/notes/broadcasting.html
3. `torch.matmul` 的维度规则：https://pytorch.org/docs/stable/generated/torch.matmul.html
4. `torch.bmm` 的批量矩阵乘法：https://pytorch.org/docs/stable/generated/torch.bmm.html
5. `torch.einsum` 的下标表达式：https://pytorch.org/docs/stable/generated/torch.einsum.html
6. Tensor view、stride 与存储关系：https://pytorch.org/docs/stable/tensor_view.html
7. `Tensor.view` 的限制：https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
8. `torch.reshape` 的 view/copy 语义：https://pytorch.org/docs/stable/generated/torch.reshape.html
9. `torch.nn.functional.cross_entropy` 的输入和 target 约定：https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html

这些页面适合核对 API 的形状、dtype、device 和返回值语义，不等于对某个模型实现的正确性证明。性能、显存、非连续输入是否触发内部复制、混合精度下的数值稳定性，都依赖 PyTorch 版本、后端、硬件和具体输入；书中的 demo 只能证明教学构造在当前环境可运行，不能替代目标部署的 benchmark。

## 1.23 本章小结

PyTorch 张量基础的重点不是背 API，而是形成稳定的工程判断：

1. 任何模型代码先看 shape。
2. Dtype 决定精度、性能和模块输入要求。
3. Device 必须一致，否则 CPU/GPU 混用会报错。
4. Broadcasting 很强大，但要确认语义方向正确。
5. `matmul` 和 `einsum` 是理解 attention 的核心工具。
6. `reshape`、`view`、`transpose`、`permute`、`contiguous` 背后是内存布局和 stride。
7. Mask、loss reshape、多头拆分和合并，是大模型工程最常见的 tensor 基础。
8. 能运行不等于语义正确；contract、边界样本和有限值检查要一起验证。

下一章会在 tensor 基础上进入 autograd，理解 PyTorch 如何构建计算图、保存中间结果、执行 backward，以及为什么 `detach`、`no_grad`、梯度累积和 in-place 操作会影响训练行为。
