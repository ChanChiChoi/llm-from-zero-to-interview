# 第二章：Autograd 与反向传播

Autograd 是 PyTorch 把数学公式变成可训练模型的核心机制。你写 `loss.backward()` 时，PyTorch 会沿着前向计算留下的计算图，从 loss 反向追踪到模型参数，自动计算每个参数的梯度。

很多人能写训练代码，但一遇到梯度为 `None`、显存莫名增长、`Trying to backward through the graph a second time`、`one of the variables needed for gradient computation has been modified by an inplace operation` 这类问题，就不知道从哪里查。原因通常不是反向传播公式不会，而是没有真正理解 PyTorch 的动态计算图、叶子张量、梯度累积、`detach`、`no_grad` 和 in-place 操作之间的关系。

本章目标不是重新推导所有神经网络反向传播公式，而是把 PyTorch Autograd 在工程里的工作方式讲清楚：哪些 tensor 会记录梯度，计算图什么时候创建和释放，为什么梯度会累积，为什么推理阶段要关掉梯度，什么时候应该 `detach`，什么时候不该 `detach`，以及如何排查训练脚本里的梯度问题。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修时，参考了 PyTorch 官方 automatic differentiation tutorial、autograd mechanics、leaf / non-leaf tensor tutorial、`torch.autograd.grad`、`Tensor.backward`、`Tensor.detach`、`torch.no_grad`、`torch.inference_mode`、`detect_anomaly`、`Optimizer.zero_grad` 和 activation checkpointing 相关文档，并结合前序数学基础、张量基础和训练循环章节的公式与工程口径。

本章只聚焦 PyTorch Autograd 的工程主线：动态计算图、链式法则、`requires_grad`、叶子张量、`.grad`、`backward`、vector-Jacobian product、梯度累积、`zero_grad`、`detach`、`no_grad` / `inference_mode`、in-place 风险、`autograd.grad`、hook、训练循环顺序和梯度 debug。它不展开完整自动微分理论、CUDA backward kernel、functorch / `torch.func`、分布式 autograd 或编译器级图优化。

## 2.1 为什么需要 Autograd

训练神经网络的目标是最小化 loss。以最简单的线性模型为例：

```math
\hat{y}=wx+b
```

如果使用均方误差：

```math
L=(\hat{y}-y)^2
```

训练时需要知道：

```math
\frac{\partial L}{\partial w},\qquad \frac{\partial L}{\partial b}
```

手写小模型的梯度还可以接受，但真实大模型有 embedding、attention、MLP、layer norm、residual connection、dropout、mask、loss reshape、混合精度和分布式通信。如果每个算子都手写反向传播，几乎不可维护。

Autograd 解决的是这个问题：你只写前向计算，框架自动记录前向过程，并根据链式法则自动生成反向计算。

这就是 PyTorch 训练代码通常长这样：

```python
logits = model(input_ids)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

前两行是你显式写出的前向计算，`backward()` 会触发反向传播，`optimizer.step()` 使用梯度更新参数。

面试回答：

```text
Autograd 的作用是自动求导。PyTorch 在前向计算时动态构建计算图，每个可求导操作都会记录反向函数。调用 loss.backward() 时，PyTorch 从标量 loss 出发按链式法则反向遍历计算图，把梯度累积到叶子参数的 .grad 上。这样我们只需要写 forward，不需要手写每个算子的 backward。
```

### 2.1.1 关键公式与 Autograd 调试速查

标量 loss 对参数的梯度：

$$
g_\theta = \frac{\partial L}{\partial \theta}
$$

链式法则：

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

多层计算图中的反向传播：

$$
x \rightarrow h_1 \rightarrow h_2 \rightarrow L
$$

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial h_2}
\frac{\partial h_2}{\partial h_1}
\frac{\partial h_1}{\partial x}
$$

向量输出的 `backward(gradient)` 本质是 vector-Jacobian product：

$$
v^T J,\qquad J=\frac{\partial y}{\partial x}
$$

梯度累积：

$$
grad_t = grad_{t-1} + \frac{\partial L_t}{\partial \theta}
$$

梯度累积模拟大 batch 时的平均 loss：

$$
L_{\mathrm{micro}}^{scaled} = \frac{L_{\mathrm{micro}}}{K}
$$

参数更新必须在不建图环境中做：

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

叶子张量与 `.grad` 的工程口径：

1. 模型参数通常是 leaf tensor，`backward()` 后梯度累积到参数的 `.grad`。
2. 中间 tensor 默认不保留 `.grad`，需要调试时用 `retain_grad()` 或 hook。
3. `detach()` 切断某个 tensor 和当前计算图的连接。
4. `no_grad()` / `inference_mode()` 让一段代码不记录新的 autograd 图。
5. `retain_graph=True` 不是普通训练的默认解法，通常应该重新 forward 或重构 loss。

## 2.2 动态计算图：前向时构建，反向后释放

PyTorch 使用动态计算图，也常被称为 define-by-run。意思是：代码每执行一次前向，就根据实际执行路径构建一张新的计算图。

示例：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x * x + 3 * x + 1

print(y)          # tensor(11., grad_fn=<AddBackward0>)
print(y.grad_fn)  # 记录 y 是由什么反向函数产生的

y.backward()
print(x.grad)     # tensor(7.)，因为 dy/dx = 2x + 3 = 7
```

这里 `x` 是需要梯度的输入，`y` 是由一系列计算得到的结果。因为 `x.requires_grad=True`，所以 `x * x`、`3 * x`、加法这些操作会被 autograd 记录下来。

动态计算图的特点是：

1. 每次 forward 都会重新构图。
2. Python 控制流可以自然参与建图，例如 `if`、`for`、递归。
3. backward 默认会释放中间激活和计算图，节省显存。
4. 如果需要对同一张图反向多次，必须显式指定 `retain_graph=True`，但工程中应谨慎使用。

例如：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x

y.backward()
y.backward()  # RuntimeError
```

第二次 `backward()` 会报错，典型信息是：

```text
Trying to backward through the graph a second time
```

原因是第一次反向传播后，PyTorch 已经释放了计算图中保存的中间结果。通常正确做法不是随手加 `retain_graph=True`，而是重新执行一次 forward：

```python
x = torch.tensor(2.0, requires_grad=True)

y = x * x
y.backward()

x.grad = None
y = x * x
y.backward()
```

`retain_graph=True` 适用于确实需要从同一张图上做多次 backward 的场景，例如某些多 loss 分开反传、梯度惩罚、二阶梯度实验。但在普通训练循环里，如果加它只是为了让报错消失，往往会造成显存持续增长。

## 2.3 requires_grad 决定是否跟踪梯度

`requires_grad` 表示这个 tensor 是否需要参与梯度计算。

```python
x = torch.randn(3, requires_grad=True)
w = torch.randn(3, requires_grad=True)

y = (x * w).sum()
print(y.requires_grad)  # True

y.backward()
print(x.grad)
print(w.grad)
```

只要一个操作的输入中有 tensor 需要梯度，并且这个操作本身可求导，输出通常也会带上 `requires_grad=True`。

对模型参数来说，`requires_grad=True` 通常由 `nn.Parameter` 自动设置：

```python
import torch.nn as nn

linear = nn.Linear(4, 2)

for name, param in linear.named_parameters():
    print(name, param.requires_grad)
```

输出一般是：

```text
weight True
bias True
```

如果要冻结某些参数，可以设置：

```python
for param in model.backbone.parameters():
    param.requires_grad = False
```

冻结参数后，这些参数不会累积梯度，也不会被优化器更新。微调大模型时，常见做法是冻结 base model，只训练 LoRA adapter 或某些 head；或者先冻结底层，再逐步解冻。

需要注意：冻结参数不等于整段 forward 都不建图。如果后续仍有可训练参数依赖这些中间结果，PyTorch 仍可能需要保留部分计算用于梯度传递。真正的推理阶段应该使用 `torch.no_grad()` 或 `torch.inference_mode()`。

## 2.4 叶子张量、非叶子张量和 .grad

理解 `.grad` 时，必须理解叶子张量。

简单说：

1. 用户直接创建、且没有由其他可求导操作产生的 tensor，通常是叶子张量。
2. `nn.Parameter` 是叶子张量。
3. 由计算产生的中间结果通常是非叶子张量。
4. 默认只有叶子张量的 `.grad` 会在 `backward()` 后被保留下来。

示例：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y * y

print(x.is_leaf)  # True
print(y.is_leaf)  # False
print(z.is_leaf)  # False

z.backward()

print(x.grad)  # 有值
print(y.grad)  # 默认是 None，并可能出现 warning
```

这不是 `y` 没有梯度，而是 PyTorch 默认不保存非叶子张量的 `.grad`。原因很直接：中间激活太多，如果都保存 `.grad`，显存会爆。

如果确实想观察某个中间 tensor 的梯度，可以使用 `retain_grad()`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
y.retain_grad()
z = y * y

z.backward()

print(y.grad)
```

面试中如果被问“为什么我的中间 tensor `.grad` 是 None”，可以这样答：

```text
PyTorch 默认只把梯度累积到叶子张量的 .grad 上，模型参数就是典型叶子张量。中间 tensor 虽然参与反向传播，但默认不保留 .grad，是为了节省显存。如果需要调试中间梯度，可以对该 tensor 调用 retain_grad()，或者注册 hook。
```

## 2.5 backward 做了什么

`loss.backward()` 的本质是从 loss 对自己梯度为 1 开始，沿计算图反向应用链式法则。

以标量函数为例：

```math
y=x^2,
\qquad z=3y
```

那么：

```math
\frac{\partial z}{\partial x}
=\frac{\partial z}{\partial y}\frac{\partial y}{\partial x}
=3\cdot 2x
```

对应代码：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x
z = 3 * y

z.backward()
print(x.grad)  # tensor(12.)
```

如果输出不是标量，就不能直接 `backward()`，除非提供外部梯度：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * x

# y.backward()  # RuntimeError: grad can be implicitly created only for scalar outputs
y.backward(torch.ones_like(y))

print(x.grad)  # tensor([2., 4., 6.])
```

为什么标量 loss 可以不传 gradient？因为默认是：

```math
\frac{\partial L}{\partial L}=1
```

而向量输出需要告诉 autograd 你想计算哪个向量-雅可比积。PyTorch 的反向传播不是默认显式构造完整 Jacobian，而是高效计算 vector-Jacobian product。这一点在大模型里很重要，因为完整 Jacobian 可能大到不可接受。

## 2.6 梯度会累积，不会自动清零

PyTorch 的梯度默认是累积的。

```python
x = torch.tensor(1.0, requires_grad=True)

y = x * 2
y.backward()
print(x.grad)  # tensor(2.)

y = x * 3
y.backward()
print(x.grad)  # tensor(5.)，不是 tensor(3.)
```

这就是训练循环里每一步都要清梯度的原因：

```python
for batch in dataloader:
    optimizer.zero_grad()

    logits = model(batch["input_ids"])
    loss = criterion(logits, batch["labels"])

    loss.backward()
    optimizer.step()
```

如果忘记 `zero_grad()`，梯度会把多个 step 的结果加在一起，训练会变得不可控。

但梯度累积也可以被主动利用。比如显存只能放下小 batch，但想模拟大 batch：

```python
accum_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(dataloader):
    logits = model(batch["input_ids"])
    loss = criterion(logits, batch["labels"])
    loss = loss / accum_steps
    loss.backward()

    if (step + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

这里把 loss 除以 `accum_steps` 是为了让累积后的梯度尺度接近真实大 batch 的平均梯度。如果不除，等价于把学习率放大了约 `accum_steps` 倍。

更推荐的清梯度写法是：

```python
optimizer.zero_grad(set_to_none=True)
```

它会把 `.grad` 设为 `None`，通常比填零更省内存和更快。需要注意，某些手写逻辑如果假设 `param.grad` 一定是 tensor，就要处理 `None` 情况。

## 2.7 detach：切断计算图

`detach()` 会返回一个与原 tensor 共享数据、但不再连接当前计算图的新 tensor。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()

print(y.requires_grad)  # True
print(z.requires_grad)  # False
```

如果后续 loss 只依赖 `z`，梯度不会再传回 `x`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach()
loss = z * z

# loss.backward()  # 会报错，因为 loss 不需要梯度
```

常见使用场景：

1. 日志记录时把 tensor 从图里拿出来，避免保留整张计算图。
2. 生成样本、缓存 hidden states、更新指标时不需要梯度。
3. 某些算法需要阻断一条分支的梯度，例如 target network、stop-gradient、对比学习里的某些 teacher 分支。
4. RNN 或长序列训练中做 truncated backpropagation through time。

典型日志写法：

```python
loss_value = loss.detach().item()
```

或者：

```python
metrics["loss"] += loss.item()
```

`.item()` 会把单元素 tensor 转成 Python 数字，也会脱离计算图。不要把带图的 `loss` 直接长期放进 list：

```python
losses = []

for batch in dataloader:
    loss = compute_loss(batch)
    losses.append(loss)  # 风险：可能保留每一步的计算图
```

更安全的是：

```python
losses.append(loss.detach().cpu())
```

或只保存数值：

```python
losses.append(loss.item())
```

`detach` 的风险也很大：如果误用在模型输出、loss 或中间 hidden states 上，梯度会被切断，导致参数不更新或部分模块学不到东西。

## 2.8 no_grad 和 inference_mode

`torch.no_grad()` 用于关闭梯度记录，常用于验证和推理：

```python
model.eval()

with torch.no_grad():
    logits = model(input_ids)
    probs = torch.softmax(logits, dim=-1)
```

在 `no_grad` 作用域里，即使输入或参数 `requires_grad=True`，新产生的结果也不会被 autograd 记录。这样可以：

1. 减少显存占用。
2. 加快推理速度。
3. 避免验证阶段误保留计算图。

`torch.inference_mode()` 比 `no_grad()` 更激进，进一步关闭版本计数等 autograd 相关机制，推理性能可能更好：

```python
model.eval()

with torch.inference_mode():
    logits = model(input_ids)
```

一般经验：

1. 验证、推理、离线 embedding 抽取，优先使用 `torch.inference_mode()`。
2. 如果代码中需要创建后续还可能参与 autograd 的 tensor，或者存在特殊的 view/in-place 交互，使用 `torch.no_grad()` 更保守。
3. 训练阶段不要用 `no_grad()` 包住 forward，否则 loss 无法反向传播到参数。

`model.eval()` 和 `torch.no_grad()` 不是一回事：

1. `model.eval()` 影响 dropout、batch norm 等模块行为。
2. `torch.no_grad()` 影响 autograd 是否记录计算图。

验证时通常两个都要用：

```python
model.eval()
with torch.no_grad():
    val_loss = evaluate(model, val_loader)
model.train()
```

面试回答：

```text
model.eval() 只是切换模块行为，比如关闭 dropout、使用 batch norm 的 running statistics；no_grad() 是关闭梯度记录，减少显存和计算图开销。验证或推理时通常两者都需要。只写 eval 不会自动关闭 autograd，只写 no_grad 也不会改变 dropout 的行为。
```

## 2.9 in-place 操作为什么容易破坏反向传播

PyTorch 中以下划线结尾的操作通常是 in-place 操作，例如：

```python
x.add_(1)
x.relu_()
x.masked_fill_(mask, 0)
```

in-place 操作会直接修改原 tensor 的数据。它可能更省内存，但也可能破坏 autograd 需要保存的中间值。

典型报错：

```text
one of the variables needed for gradient computation has been modified by an inplace operation
```

直觉上，反向传播需要用前向时的某些值计算梯度。如果你在 backward 之前把这些值原地改掉，PyTorch 就无法保证梯度正确。

示例：

```python
x = torch.randn(4, requires_grad=True)
y = x * 2
z = y * y

y.add_(1)  # 修改了反向传播可能需要的中间值
loss = z.sum()
loss.backward()
```

并不是所有 in-place 操作都会报错，也不是所有带下划线的操作都绝对不能用。但在训练代码里，尤其是刚开始写模型时，建议优先使用非 in-place 版本：

```python
x = x + 1
x = torch.relu(x)
x = x.masked_fill(mask, 0)
```

Transformer 实现中还要特别注意 mask：

```python
scores = scores.masked_fill(causal_mask, float("-inf"))
```

比下面这种写法更安全：

```python
scores.masked_fill_(causal_mask, float("-inf"))
```

面试中可以这样说：

```text
in-place 操作会修改 tensor 本身，而 autograd 反向传播可能依赖前向保存的中间值。PyTorch 会用版本计数检查某些 tensor 是否被原地改动，如果发现反向所需变量被修改，就会报 inplace 相关错误。工程上我会优先使用非 inplace 写法，只有在确认不会破坏计算图且确实有显存收益时才使用 inplace。
```

## 2.10 常见 Autograd 报错与定位方法

### 2.10.1 grad 是 None

常见原因：

1. 参数没有参与 loss 计算。
2. 参数的 `requires_grad=False`。
3. 中间某处用了 `detach()`、`.item()`、`torch.no_grad()` 切断计算图。
4. 你查看的是非叶子 tensor 的 `.grad`。
5. 优化器里没有包含这个参数。
6. 分支逻辑导致某个模块在当前 batch 没有被调用。

排查代码：

```python
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is None:
        print("missing grad:", name)
```

如果某些参数长期没有梯度，要检查 forward 路径和 loss 是否真的依赖它。

### 2.10.2 loss 不下降

loss 不下降不一定是 autograd 问题，但可以先排除几类梯度问题：

```python
total_norm = 0.0
for param in model.parameters():
    if param.grad is not None:
        total_norm += param.grad.detach().norm().item()

print("grad_norm", total_norm)
```

如果梯度全是 0 或 `None`，可能是图断了、loss 写错、mask 把所有 token 都忽略了、学习率过小、激活饱和或初始化有问题。

如果梯度是 `nan` 或 `inf`，要检查学习率、混合精度、loss scale、softmax 前是否出现极端值、mask 是否产生全 `-inf` 行。

### 2.10.3 backward through graph a second time

常见原因：

1. 对同一个 loss 调用了两次 `backward()`。
2. 多个 loss 共用同一张计算图，但分开 backward 时没有保留图。
3. 循环里保存了带图 tensor，并在后续 step 重复参与计算。

可能修复：

1. 重新 forward 后再 backward。
2. 把多个 loss 加起来一次 backward：`(loss1 + loss2).backward()`。
3. 如果确实需要多次 backward，再考虑 `retain_graph=True`。
4. 对跨 step 保存的状态使用 `detach()`。

### 2.10.4 element does not require grad

典型报错：

```text
element 0 of tensors does not require grad and does not have a grad_fn
```

常见原因：

1. loss 是在 `torch.no_grad()` 里算出来的。
2. loss 经过 `.item()` 变成 Python 数字后又包装回 tensor。
3. 模型输出被 `detach()` 了。
4. 所有参与计算的 tensor 都不需要梯度。

错误写法：

```python
loss_value = loss.item()
loss = torch.tensor(loss_value, requires_grad=True)
loss.backward()
```

这样新建的 `loss` 已经和原模型计算图没有关系，即使能 backward，也不会更新模型参数。

## 2.11 autograd.grad、二阶梯度和高阶求导

除了 `loss.backward()`，PyTorch 还提供 `torch.autograd.grad`，用于直接返回某些输入的梯度，而不是累积到 `.grad` 上。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

grad_x = torch.autograd.grad(y, x)
print(grad_x)  # (tensor(12.),)
```

如果要继续对梯度求导，需要 `create_graph=True`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3

grad_x = torch.autograd.grad(y, x, create_graph=True)[0]
grad2_x = torch.autograd.grad(grad_x, x)[0]

print(grad_x)   # tensor(12., grad_fn=...)
print(grad2_x)  # tensor(12.)
```

常见场景包括：

1. 梯度惩罚。
2. 元学习。
3. 隐式优化。
4. 某些物理信息神经网络。
5. 研究二阶优化或 Hessian 相关性质。

高阶梯度会显著增加显存和计算开销。大模型训练主流程中很少直接使用完整二阶梯度，更多是使用一阶优化器，或者用近似方法估计曲率信息。

## 2.12 hook：观察和修改梯度

hook 可以在反向传播时观察梯度，常用于 debug。

```python
def print_grad(grad):
    print("grad mean:", grad.mean().item())

x = torch.randn(4, requires_grad=True)
y = (x * x).sum()

handle = x.register_hook(print_grad)
y.backward()
handle.remove()
```

对模型参数也可以注册 hook：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        param.register_hook(lambda grad, name=name: print(name, grad.norm().item()))
```

hook 的用途：

1. 查看某层梯度是否为 0、`nan` 或异常大。
2. 定位梯度从哪一层开始消失或爆炸。
3. 实现简单的梯度裁剪、mask 或自定义调试逻辑。

但训练主逻辑中不要滥用 hook。hook 会让梯度路径变得隐蔽，降低代码可读性，也可能引入性能开销。正式工程里，优先用清晰的 forward/loss 写法和显式日志；hook 更适合作为临时 debug 工具。

## 2.13 训练循环里的正确顺序

标准训练 step 通常是：

```python
model.train()

for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids)
    loss = criterion(logits, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

关键点：

1. `zero_grad` 通常放在 forward 前，避免旧梯度残留。
2. forward 和 loss 计算必须在梯度开启的环境里。
3. `backward` 后参数的 `.grad` 才有值。
4. gradient clipping 要放在 `backward` 后、`optimizer.step` 前。
5. `optimizer.step` 后参数发生更新。
6. 日志记录时用 `loss.item()` 或 `loss.detach()`。

验证循环则不同：

```python
model.eval()
total_loss = 0.0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()

model.train()
```

如果验证时忘记 `no_grad()`，结果通常仍然正确，但会构建计算图，导致显存更高、速度更慢。如果还把验证 loss tensor 存起来，可能造成显存持续增长。

## 2.14 大模型工程中的 Autograd 边界

在大模型训练里，autograd 问题通常不只出现在简单的 `loss.backward()`，而是出现在模块边界和工程优化里。

### 2.14.1 LoRA 和参数冻结

LoRA 微调时，常见目标是只训练 adapter 参数：

```python
for name, param in model.named_parameters():
    param.requires_grad = "lora" in name
```

排查时要确认：

1. 冻结的 base 参数 `requires_grad=False`。
2. LoRA 参数 `requires_grad=True`。
3. 优化器只包含可训练参数。
4. LoRA 分支确实参与 forward。

```python
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
```

### 2.14.2 梯度检查点

Gradient checkpointing 会在 forward 时不保存部分中间激活，backward 时重新计算它们，以计算换显存。它仍然依赖 autograd，但改变了中间激活的保存策略。

直觉：

1. 普通训练：forward 保存激活，backward 直接使用。
2. checkpoint：forward 少保存激活，backward 重新执行部分 forward。

优点是省显存，缺点是训练更慢。大模型训练里经常用它换更长序列、更大 batch 或更大模型。

### 2.14.3 混合精度和 GradScaler

混合精度训练中，autograd 仍然负责求导，但部分 forward/backward 以 fp16/bf16 执行。fp16 容易出现 underflow，因此常和 loss scaling 配合。

典型 AMP 训练结构：

```python
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast():
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["labels"])

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

这里梯度裁剪前要先 `unscale_`，否则裁剪的是被 scale 过的梯度。bf16 动态范围更大，很多大模型训练中不一定需要 GradScaler。

### 2.14.4 分布式训练

在 DistributedDataParallel 中，每个 rank 前向和反向自己的 mini-batch。反向传播过程中，DDP 会通过 autograd hook 在梯度 ready 时触发 all-reduce，把各 rank 的梯度同步。

工程上常见问题：

1. 某些参数在某些 rank 没参与 forward，导致梯度同步异常。
2. 条件分支让不同 rank 走了不同计算路径。
3. 梯度累积时忘记使用 `no_sync()`，导致每个 micro-step 都同步，性能变差。
4. 手动 `detach` 或 `no_grad` 切断了需要同步的参数梯度。

这些内容会在后续分布式训练章节展开，本章只需要记住：DDP 的梯度同步也是挂在 autograd 反向过程上的。

## 2.15 最小可运行 Autograd 审计 demo

下面用一个 demo 同时检查标量梯度、向量输出的外部梯度、leaf / non-leaf、`retain_grad`、梯度累积、`detach`、`no_grad`、optimizer 训练顺序和缺失梯度排查。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(13)

# 1. Scalar backward: y = x^3 + 2x, dy/dx at x=2 is 14.
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 2 * x
y.backward()
scalar_grad = float(x.grad)

# 2. Vector-Jacobian product: y = x^2, v=[1,2,3].
vjp_x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
vjp_y = vjp_x ** 2
vjp_y.backward(torch.tensor([1.0, 2.0, 3.0]))
vjp_grad = vjp_x.grad.detach().tolist()

# 3. Leaf / non-leaf and retain_grad.
leaf = torch.tensor(3.0, requires_grad=True)
middle = leaf * 4
middle.retain_grad()
out = middle ** 2
out.backward()
leaf_grad = float(leaf.grad)
middle_grad = float(middle.grad)
middle_is_leaf = middle.is_leaf

# 4. Gradients accumulate unless cleared.
accum = torch.tensor(1.0, requires_grad=True)
(accum * 2).backward()
first_grad = float(accum.grad)
(accum * 3).backward()
second_grad = float(accum.grad)
accum.grad = None
(accum * 3).backward()
after_clear_grad = float(accum.grad)

# 5. detach cuts graph on that branch.
base = torch.tensor(2.0, requires_grad=True)
tracked = base * 5
stopped = tracked.detach()
detach_loss = tracked + stopped
detach_loss.backward()
detach_grad = float(base.grad)

# 6. Minimal training loop with optimizer and zero_grad.
model = nn.Linear(2, 1)
inputs = torch.randn(16, 2)
true_w = torch.tensor([[2.0], [-3.0]])
targets = inputs @ true_w + 0.5

optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
loss_trace = []
grad_norm_trace = []

for _ in range(8):
    pred = model(inputs)
    loss = F.mse_loss(pred, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(
        p.grad.detach().norm().item()
        for p in model.parameters()
        if p.grad is not None
    )
    optimizer.step()

    loss_trace.append(round(float(loss.detach()), 4))
    grad_norm_trace.append(round(grad_norm, 4))

# 7. no_grad / eval: validation should not build graph.
model.eval()
with torch.no_grad():
    val_pred = model(inputs)
    val_loss = F.mse_loss(val_pred, targets)
model.train()

missing_grad = []
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is None:
        missing_grad.append(name)

report = {
    "scalar_grad": round(scalar_grad, 3),
    "vjp_grad": [round(v, 3) for v in vjp_grad],
    "leaf_nonleaf": {
        "middle_is_leaf": middle_is_leaf,
        "leaf_grad": round(leaf_grad, 3),
        "middle_grad_after_retain": round(middle_grad, 3),
    },
    "grad_accumulation": {
        "first": round(first_grad, 3),
        "second_without_clear": round(second_grad, 3),
        "after_clear": round(after_clear_grad, 3),
    },
    "detach": {
        "base_grad": round(detach_grad, 3),
        "expected_only_tracked_branch": 5.0,
    },
    "training": {
        "loss_trace": loss_trace,
        "grad_norm_trace": grad_norm_trace,
        "final_loss": round(float(val_loss), 4),
        "val_requires_grad": val_loss.requires_grad,
        "missing_grad_after_last_step": missing_grad,
    },
    "checks": {
        "scalar_grad_ok": abs(scalar_grad - 14.0) < 1e-6,
        "vjp_grad_ok": [round(v, 3) for v in vjp_grad] == [2.0, 8.0, 18.0],
        "nonleaf_grad_retained": middle_grad == 24.0,
        "grad_accumulates": second_grad == first_grad + 3.0,
        "detach_blocks_one_branch": detach_grad == 5.0,
        "training_loss_decreases": loss_trace[-1] < loss_trace[0],
        "no_grad_validation": val_loss.requires_grad is False,
    },
}

print(report)
```

这段代码的关键点：

1. 标量 `backward()` 默认使用 `dL/dL=1`。
2. 向量输出必须传入外部梯度，本质是 vector-Jacobian product。
3. 非叶子 tensor 默认不保留 `.grad`，调用 `retain_grad()` 后才能观察。
4. `.grad` 默认累积，清空后再 backward 才是当前图的梯度。
5. `detach()` 只切断被 detach 的分支，未 detach 的分支仍能反传。
6. 验证阶段 `no_grad()` 下的 loss 不应继续挂在计算图上。
7. 真实训练通常用 optimizer 更新参数，而不是在图里手动构造新 tensor。

## 2.16 Debug 清单

遇到梯度问题时，可以按下面顺序查：

1. `loss.requires_grad` 是否为 `True`。
2. `loss.grad_fn` 是否为 `None`。
3. 关键参数的 `requires_grad` 是否为 `True`。
4. `loss` 是否真的依赖目标参数。
5. 是否误用了 `detach()`、`.item()`、`no_grad()`。
6. 是否对同一张图 backward 了多次。
7. 是否忘记 `optimizer.zero_grad()`。
8. 是否在 `backward()` 前做了破坏计算图的 in-place 操作。
9. 是否把非叶子 tensor 的 `.grad is None` 误判成没有梯度。
10. optimizer 是否包含了需要训练的参数。
11. 梯度中是否出现 `nan` 或 `inf`。
12. 分布式训练中各 rank 是否走了相同的参数使用路径。

常用检查代码：

```python
print("loss requires_grad:", loss.requires_grad)
print("loss grad_fn:", loss.grad_fn)

for name, param in model.named_parameters():
    if param.requires_grad:
        grad_status = None if param.grad is None else param.grad.norm().item()
        print(name, grad_status)
```

如果怀疑有 in-place 问题，可以临时开启 anomaly detection：

```python
torch.autograd.set_detect_anomaly(True)
```

或者：

```python
with torch.autograd.detect_anomaly():
    loss.backward()
```

它会让 PyTorch 尝试报告导致 backward 出错的前向位置，但会明显降低速度，只适合 debug，不适合长期训练开启。

## 2.17 面试官会怎么问

### 问题一：PyTorch 的 autograd 是怎么工作的？

回答模板：

```text
PyTorch 使用动态计算图。每次 forward 时，如果输入 tensor 需要梯度，相关操作会被记录成计算图节点，并保存 backward 所需的中间信息。调用 loss.backward() 后，autograd 从 loss 开始按链式法则反向遍历图，把梯度累积到叶子 tensor，尤其是模型参数的 .grad 上。默认 backward 后图会释放，所以普通训练每一步都会重新 forward 构建新图。
```

### 问题二：为什么每次训练都要 zero_grad？

回答模板：

```text
因为 PyTorch 的 .grad 默认是累积的，不会在 backward 前自动清零。这样设计可以支持梯度累积、多 loss 反传等场景。普通训练中如果不 zero_grad，当前 batch 的梯度会和历史 batch 的梯度叠加，导致更新方向和尺度错误。所以每个 optimizer step 前通常要先 optimizer.zero_grad(set_to_none=True)。
```

### 问题三：detach 和 no_grad 有什么区别？

回答模板：

```text
detach 是对某个 tensor 切断它和当前计算图的连接，返回一个不再追踪梯度的新 tensor，常用于日志、缓存或 stop-gradient。no_grad 是一个上下文管理器，在作用域内关闭新操作的梯度记录，常用于验证和推理。detach 更像是切断某条边，no_grad 更像是让一段代码不建图。
```

### 问题四：为什么中间 tensor 的 .grad 是 None？

回答模板：

```text
默认只有叶子 tensor 的 .grad 会被保留，模型参数就是叶子 tensor。中间 tensor 虽然有梯度流过，但为了节省显存，PyTorch 不会默认保存它的 .grad。如果需要调试中间梯度，可以调用 retain_grad() 或注册 hook。
```

### 问题五：为什么验证时要同时写 model.eval() 和 no_grad()？

回答模板：

```text
model.eval() 改变模块行为，比如关闭 dropout、让 batch norm 使用 running statistics；no_grad() 关闭 autograd 记录，减少显存和计算开销。它们解决的是不同问题。验证或推理时通常两者都需要，并且验证后要切回 model.train()。
```

### 问题六：in-place 操作为什么可能导致 backward 报错？

回答模板：

```text
反向传播可能需要前向时保存的中间值。in-place 操作会直接改写 tensor，如果改掉了 backward 需要的值，梯度就可能不正确。PyTorch 会用版本计数检测这类修改，发现问题时会报变量被 inplace 修改的错误。工程上我会优先使用非 inplace 操作，除非明确知道它安全且确实能节省显存。
```

## 2.18 常见误区

1. 以为 `model.eval()` 会关闭梯度。它不会，关闭梯度要用 `no_grad()` 或 `inference_mode()`。
2. 以为 `loss.item()` 后还能反向传播。`.item()` 得到的是 Python 数字，已经脱离计算图。
3. 以为 `.grad is None` 就一定没有梯度。中间 tensor 默认不保留 `.grad`。
4. 以为 `retain_graph=True` 是解决 backward 报错的常规方案。多数时候应该重新 forward 或修正图复用逻辑。
5. 以为冻结参数后整个 forward 都不建图。如果后面还有可训练参数，仍可能需要保留部分图。
6. 以为 in-place 一定更好。它可能省一点内存，但会增加 autograd 风险。
7. 以为梯度累积只要多次 backward 就行。通常还要按累积步数缩放 loss，并控制 optimizer step 的频率。
8. 以为手动 `param = param - lr * grad` 能更新模型参数。这样可能创建新 tensor 而不是更新原参数，正确做法是 optimizer 或在 `no_grad` 下原地更新参数数据。

## 2.19 小练习

1. 写一个标量函数 `y = x ** 3 + 2 * x`，用 autograd 验证在 `x=2` 时的梯度。
2. 构造一个两层 MLP，打印每个参数的 `requires_grad`、`is_leaf` 和 backward 后的梯度范数。
3. 故意在训练循环中去掉 `optimizer.zero_grad()`，观察 loss 和梯度范数变化。
4. 写一个例子保存每一步的 `loss` tensor 到 list，再改成保存 `loss.item()`，比较显存或对象引用差异。
5. 在一个中间激活上调用 `retain_grad()`，观察它的 `.grad`。
6. 把验证循环分别写成只用 `model.eval()`、只用 `no_grad()`、两者都用，解释差异。
7. 构造一个误用 `detach()` 导致某层参数没有梯度的例子，并用 `named_parameters()` 排查。

## 2.20 本章总结

Autograd 的核心是动态计算图和链式法则。PyTorch 在 forward 时记录可求导操作，在 `loss.backward()` 时沿图反向传播，把梯度累积到叶子参数的 `.grad` 上。默认情况下，计算图在 backward 后释放，梯度不会自动清零，中间 tensor 的 `.grad` 不会被保留。

工程上最重要的不是背 API，而是知道 autograd 的边界：`requires_grad` 决定是否追踪梯度，`detach` 会切断某个 tensor 的图连接，`no_grad` 和 `inference_mode` 会关闭一段代码的建图，in-place 操作可能破坏反向传播，`retain_graph=True` 不能滥用。

如果能稳定回答“这个 loss 是否依赖这个参数”“这段代码有没有建图”“这个 tensor 是叶子还是中间结果”“梯度为什么是 None”“验证为什么要关梯度”，就已经具备了写训练脚本和排查大部分 PyTorch 梯度问题的基础。
