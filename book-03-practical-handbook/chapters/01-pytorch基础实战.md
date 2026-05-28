# 第一部分：PyTorch 基础实战

## 第 1 讲：实现线性回归与梯度下降

### 本讲目标

学完本讲，你应该能做到五件事：

1. 用 PyTorch 从零实现一个线性回归训练循环。
2. 理解 `Tensor`、`requires_grad`、`backward`、`grad` 和 `no_grad` 的关系。
3. 说清楚梯度下降每一步在更新什么。
4. 能定位线性模型训练中常见的 shape、梯度和学习率问题。
5. 能把这个小实验讲成一个面试中的“最小训练系统”。

这一讲看起来很基础。

但它是后面训练 Transformer、微调 LLM、做 DPO 和推理优化的共同底座。

如果你连一个线性回归训练循环都不能手写清楚，面试官很难相信你真正理解大模型训练。

---

### 一、问题设定

我们先做一个最简单的监督学习任务。

假设真实数据来自：

```text
y = 3x + 2 + noise
```

模型不知道真实参数。

它只看到一批 `(x, y)` 样本。

我们要训练一个模型：

```text
y_hat = w * x + b
```

目标是让 `y_hat` 尽量接近 `y`。

损失函数使用均方误差：

```text
loss = mean((y_hat - y)^2)
```

训练过程就是不断调整 `w` 和 `b`，让 loss 下降。

---

### 二、完整最小代码

下面是一份可以直接运行的 PyTorch 版本。

```python
import torch


torch.manual_seed(42)

# 1. 构造数据：y = 3x + 2 + noise
n = 100
x = torch.randn(n, 1)
y = 3.0 * x + 2.0 + 0.1 * torch.randn(n, 1)

# 2. 初始化参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
epochs = 100

for epoch in range(epochs):
    # 3. 前向计算
    y_hat = x @ w + b
    loss = ((y_hat - y) ** 2).mean()

    # 4. 反向传播
    loss.backward()

    # 5. 参数更新
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # 6. 清空梯度
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 10 == 0:
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print("final w:", w.item())
print("final b:", b.item())
```

理想情况下，最后 `w` 会接近 `3`，`b` 会接近 `2`。

这段代码很短，但已经包含了深度学习训练的完整闭环：

```text
data -> forward -> loss -> backward -> update -> zero_grad
```

后面无论是训练 MLP、Transformer 还是 LLM，本质上都离不开这条链路。

---

### 三、逐行理解训练循环

#### 1. 为什么参数需要 `requires_grad=True`

```python
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

`requires_grad=True` 表示 PyTorch 要追踪这个张量参与的计算图。

当调用：

```python
loss.backward()
```

PyTorch 会沿着计算图反向传播，把 loss 对 `w` 和 `b` 的梯度写入：

```python
w.grad
b.grad
```

如果忘了设置 `requires_grad=True`，参数不会收到梯度，也就无法训练。

#### 2. 为什么用 `x @ w + b`

这里 `x` 的 shape 是：

```text
[100, 1]
```

`w` 的 shape 是：

```text
[1, 1]
```

所以：

```text
x @ w -> [100, 1]
```

`b` 的 shape 是 `[1]`，会通过 broadcasting 加到每个样本上。

面试中如果被问 shape，应该能立刻说清楚。

#### 3. 为什么 loss 要取 mean

```python
loss = ((y_hat - y) ** 2).mean()
```

如果不取 mean，loss 会随 batch size 增大而增大。

这会让梯度尺度依赖 batch size。

取 mean 后，不同 batch size 下的学习率更容易比较。

#### 4. `backward` 做了什么

`loss.backward()` 会计算：

```text
d loss / d w
d loss / d b
```

并写入 `w.grad` 和 `b.grad`。

线性回归中，梯度可以手推。

如果：

```text
loss = mean((xw + b - y)^2)
```

那么：

```text
d loss / d w = mean(2 * (xw + b - y) * x)
d loss / d b = mean(2 * (xw + b - y))
```

PyTorch 自动帮我们做了这件事。

#### 5. 为什么更新参数要用 `torch.no_grad()`

```python
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
```

参数更新本身不是模型前向计算的一部分。

如果不放在 `no_grad` 里，PyTorch 会把更新操作也记录进计算图，导致图越来越复杂，甚至报错。

面试中可以这样说：

```text
forward 和 loss 需要构建计算图，optimizer step 不应该构建计算图，所以参数更新要放在 no_grad 里。
```

#### 6. 为什么每一步要清空梯度

```python
w.grad.zero_()
b.grad.zero_()
```

PyTorch 默认会累积梯度。

也就是说，第二次 `backward()` 得到的梯度会加到第一次的 `grad` 上。

这在梯度累积训练中很有用。

但普通训练循环中，如果忘记清零，等价于把多个 step 的梯度错误叠加，训练会不稳定。

---

### 四、用 `nn.Module` 和 `optim` 改写

上面的写法适合理解原理。

真实工程中，通常用 `nn.Module` 和优化器。

```python
import torch
from torch import nn


torch.manual_seed(42)

n = 100
x = torch.randn(n, 1)
y = 3.0 * x + 2.0 + 0.1 * torch.randn(n, 1)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_hat = model(x)
    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        weight = model.weight.item()
        bias = model.bias.item()
        print(f"epoch={epoch}, loss={loss.item():.4f}, w={weight:.4f}, b={bias:.4f}")
```

这版代码和手写版本本质一致。

对应关系如下：

| 手写版本 | 工程版本 |
|---|---|
| `w`, `b` | `model.parameters()` |
| `x @ w + b` | `model(x)` |
| 手写 MSE | `nn.MSELoss()` |
| 手写 `w -= lr * grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

大模型训练只是把 `nn.Linear(1, 1)` 换成了巨大的 Transformer，把 MSE 换成了 next-token cross entropy，把 SGD 换成 AdamW 或分布式优化器。

---

### 五、最容易踩的坑

#### 坑 1：忘记清空梯度

表现：loss 抖动，参数更新越来越怪。

原因：梯度默认累积。

解决：每个 step 调用：

```python
optimizer.zero_grad()
```

或手动：

```python
w.grad.zero_()
```

#### 坑 2：在有梯度追踪时原地更新参数

错误写法：

```python
w -= lr * w.grad
```

如果没有 `torch.no_grad()`，可能破坏 autograd 图。

正确写法：

```python
with torch.no_grad():
    w -= lr * w.grad
```

#### 坑 3：shape 不一致但被 broadcasting 掩盖

例如 `y` 是 `[100]`，`y_hat` 是 `[100, 1]`。

这时相减可能 broadcast 成 `[100, 100]`，loss 看起来能算，但含义完全错。

建议训练前打印：

```python
print(x.shape, y.shape, y_hat.shape)
```

#### 坑 4：学习率过大

表现：loss 变成 `nan` 或越来越大。

原因：每一步跨过了最优点太远。

解决：减小学习率，例如从 `0.1` 降到 `0.01`。

#### 坑 5：误把 `.item()` 用在训练图中

`.item()` 会把单元素 Tensor 转成 Python 数字。

它常用于日志打印。

不要把 `.item()` 后的值再拿去参与需要反传的 loss。

---

### 六、面试怎么讲这个实验

如果面试官让你“手写一个最小训练循环”，可以这样回答：

```text
我会先构造输入和标签，然后定义需要训练的参数，并设置 requires_grad=True。每个 step 先做 forward，计算 loss，然后调用 backward 让 autograd 计算梯度。接着在 no_grad 环境下根据学习率更新参数，最后清空梯度，避免梯度累积影响下一个 step。
```

如果继续追问“PyTorch 为什么要 zero_grad”，可以回答：

```text
因为 PyTorch 的 grad 默认是累积的，这样可以支持 gradient accumulation。但普通训练中每个 mini-batch 应该使用自己的梯度，所以每个 step 前或 step 后都要清空梯度。
```

如果追问“大模型训练和线性回归训练循环有什么共同点”，可以回答：

```text
共同点是训练闭环完全一致：forward 计算预测，loss 度量目标差异，backward 计算梯度，optimizer step 更新参数，zero_grad 清理梯度。区别在于模型结构、损失函数、数据规模、优化器、并行策略和数值稳定性复杂得多。
```

---

### 七、从这个实验连接到大模型训练

线性回归里的每个概念，在 LLM 训练中都有对应物。

| 线性回归 | LLM 训练 |
|---|---|
| 输入 `x` | token ids / embeddings |
| 参数 `w`, `b` | Transformer 参数 |
| 预测 `y_hat` | next-token logits |
| MSE loss | cross entropy loss |
| SGD step | AdamW / fused optimizer step |
| 单机训练 | DDP / FSDP / ZeRO |
| shape 检查 | batch、sequence、hidden、vocab 检查 |

所以不要轻视这个小实验。

它是理解所有训练工程的最小闭环。

---

### 八、小练习

#### 练习 1

把真实函数改成：

```text
y = -5x + 0.7 + noise
```

观察训练后的 `w` 和 `b` 是否接近真实值。

#### 练习 2

把学习率从 `0.1` 改成：

```text
1.0, 0.01, 0.001
```

观察 loss 曲线变化。

回答：学习率过大和过小分别会出现什么现象？

#### 练习 3

故意删除 `zero_grad()`，观察参数和 loss 的变化。

回答：为什么梯度累积会改变训练行为？

#### 练习 4

把 `y` 从 `[n, 1]` 改成 `[n]`，打印 `y_hat - y` 的 shape。

回答：为什么 broadcasting bug 在训练中很危险？

---

### 本讲总结

这一讲实现了 PyTorch 中最小的训练系统。

核心结论如下：

1. 一个训练循环包括 forward、loss、backward、update 和 zero_grad。
2. `requires_grad=True` 决定参数是否被 autograd 追踪。
3. `loss.backward()` 会把梯度写入参数的 `.grad` 字段。
4. 参数更新应放在 `torch.no_grad()` 中。
5. PyTorch 默认累积梯度，所以普通训练要清空梯度。
6. shape、学习率和 broadcasting 是最常见的基础坑。
7. 大模型训练本质上仍然遵循同一个训练闭环，只是模型、数据和系统规模更复杂。

下一讲，我们继续实现 MLP 分类器。

也就是从线性模型进入多层神经网络，并理解激活函数、隐藏层和分类损失如何组合成一个完整分类模型。

## 第 2 讲：实现 MLP 分类器

### 本讲目标

学完本讲，你应该能做到六件事：

1. 用 PyTorch 实现一个最小 MLP 分类器。
2. 理解输入层、隐藏层、激活函数和输出层分别做什么。
3. 说清楚分类任务中 logits、probability 和 label 的关系。
4. 正确使用 `nn.CrossEntropyLoss()`，避免重复 `softmax`。
5. 能定位分类训练中的 shape、dtype、label 编码和过拟合问题。
6. 能把 MLP 分类器讲成 Transformer 前馈网络和分类头的基础版本。

上一讲我们实现了线性回归。

线性回归的输出是一个连续值。

这一讲我们进入分类任务。

分类任务更接近大模型里的很多核心问题：

```text
给定输入 -> 计算每个类别的分数 -> 用交叉熵训练模型把正确类别分数拉高
```

LLM 的 next-token prediction 本质上也是一个超大词表分类问题。

---

### 一、问题设定

我们构造一个二维平面上的三分类任务。

每个样本有两个特征：

```text
x = [x1, x2]
```

标签是三类之一：

```text
y in {0, 1, 2}
```

模型要输出三个分数：

```text
logits = [score_class_0, score_class_1, score_class_2]
```

训练目标是让真实类别对应的分数最高。

注意这里先输出的是 `logits`，不是概率。

`logits` 可以是任意实数。

概率是对 logits 做 softmax 之后得到的：

```text
prob = softmax(logits)
```

但训练时通常不要手动 softmax，因为 `nn.CrossEntropyLoss()` 内部已经包含了 `log_softmax + NLLLoss`。

---

### 二、完整最小代码

下面是一份可以直接运行的 MLP 三分类代码。

```python
import torch
from torch import nn


torch.manual_seed(42)

# 1. 构造三类二维数据
n_per_class = 100

x0 = torch.randn(n_per_class, 2) + torch.tensor([-2.0, -2.0])
x1 = torch.randn(n_per_class, 2) + torch.tensor([2.0, -2.0])
x2 = torch.randn(n_per_class, 2) + torch.tensor([0.0, 2.0])

x = torch.cat([x0, x1, x2], dim=0)
y = torch.cat([
    torch.zeros(n_per_class, dtype=torch.long),
    torch.ones(n_per_class, dtype=torch.long),
    torch.full((n_per_class,), 2, dtype=torch.long),
])

# 2. 定义 MLP 分类器
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

# 3. 训练循环
for epoch in range(200):
    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        print(f"epoch={epoch}, loss={loss.item():.4f}, acc={acc.item():.4f}")
```

这段代码里，训练闭环仍然是：

```text
forward -> loss -> zero_grad -> backward -> optimizer.step
```

和上一讲完全一致。

区别在于：

1. 模型从线性层变成了多层网络。
2. 任务从回归变成了分类。
3. 损失函数从 MSE 变成了交叉熵。
4. 输出从一个连续值变成了多个类别 logits。

---

### 三、MLP 到底在做什么

MLP 是 Multi-Layer Perceptron，多层感知机。

本讲代码中的模型是：

```python
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
```

可以拆成三步。

#### 1. 第一层线性变换

```python
nn.Linear(2, 16)
```

输入是二维特征。

输出是 16 维隐藏表示。

shape 变化是：

```text
[batch, 2] -> [batch, 16]
```

它做的是：

```text
h = xW + b
```

这一层把原始特征映射到更高维空间。

#### 2. 非线性激活

```python
nn.ReLU()
```

如果没有激活函数，多层线性层叠加后仍然等价于一个线性层。

也就是说：

```text
Linear -> Linear -> Linear
```

本质还是：

```text
one Linear
```

ReLU 引入非线性，让模型可以拟合更复杂的决策边界。

#### 3. 输出层

```python
nn.Linear(16, 3)
```

输出三个类别的 logits。

shape 是：

```text
[batch, 16] -> [batch, 3]
```

每一行对应一个样本，每一列对应一个类别分数。

---

### 四、logits、softmax 和 label

这是分类任务里最容易混淆的地方。

假设一个样本输出：

```python
logits = torch.tensor([1.2, -0.3, 2.1])
```

这三个数不是概率。

它们没有限制必须大于 0，也不要求和为 1。

如果要转成概率，可以做：

```python
probs = torch.softmax(logits, dim=-1)
```

预测类别是分数最大的类别：

```python
pred = logits.argmax(dim=-1)
```

标签 `y` 不是 one-hot，而是类别 id：

```text
0, 1, 2
```

对于 `nn.CrossEntropyLoss()`，输入要求是：

```text
logits: [batch, num_classes], dtype=float
labels: [batch], dtype=long
```

在本讲中就是：

```text
logits: [300, 3]
y:      [300]
```

如果把 `y` 写成 `[300, 1]`，或者写成 float，都会导致错误或训练含义不对。

---

### 五、为什么不要在 CrossEntropyLoss 前手动 softmax

错误写法：

```python
probs = torch.softmax(logits, dim=-1)
loss = nn.CrossEntropyLoss()(probs, y)
```

正确写法：

```python
loss = nn.CrossEntropyLoss()(logits, y)
```

原因是 `CrossEntropyLoss` 内部已经做了：

```text
log_softmax + negative log likelihood
```

如果你提前 softmax，会带来两个问题：

1. 数值稳定性变差。
2. 损失函数接收到的不是它期望的 logits。

面试中可以这样回答：

```text
PyTorch 的 CrossEntropyLoss 期望输入 raw logits，而不是 softmax 后的概率。它内部会用更稳定的 log_softmax 实现交叉熵，所以训练时不应该先手动 softmax。
```

---

### 六、手动计算一次 accuracy

分类任务通常会看准确率。

```python
preds = logits.argmax(dim=-1)
acc = (preds == y).float().mean()
```

逐行解释：

```python
preds = logits.argmax(dim=-1)
```

从每个样本的类别分数中取最大值所在位置。

```python
preds == y
```

得到布尔向量。

```python
.float().mean()
```

把 `True/False` 转成 `1/0`，再求平均。

注意 accuracy 不参与反向传播。

它只是评估指标。

训练真正优化的是交叉熵 loss。

---

### 七、加入 train/eval 划分

上面的代码为了简单，直接在全部数据上训练和评估。

真实实验中至少要分 train/test。

```python
perm = torch.randperm(x.size(0))
x = x[perm]
y = y[perm]

train_size = int(0.8 * x.size(0))
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

for epoch in range(200):
    model.train()
    logits = model(x_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            test_preds = test_logits.argmax(dim=-1)
            test_acc = (test_preds == y_test).float().mean()
        print(f"epoch={epoch}, loss={loss.item():.4f}, test_acc={test_acc.item():.4f}")
```

这里有两个重要习惯：

1. 训练时调用 `model.train()`。
2. 验证时调用 `model.eval()` 并使用 `torch.no_grad()`。

本讲的模型没有 Dropout 和 BatchNorm，所以 `train/eval` 差别不明显。

但真实模型中这是必须养成的习惯。

---

### 八、常见工程坑

#### 坑 1：标签 dtype 错误

`CrossEntropyLoss` 要求 label 是 `torch.long`。

错误写法：

```python
y = y.float()
```

正确写法：

```python
y = y.long()
```

#### 坑 2：手动 softmax 后再传入交叉熵

不要这样做。

训练时传 logits。

推理展示概率时才 softmax。

#### 坑 3：类别维度搞错

`CrossEntropyLoss` 默认认为输入 shape 是：

```text
[batch, num_classes]
```

如果你写成 `[num_classes, batch]`，语义就错了。

#### 坑 4：label 从 1 开始编号

PyTorch 分类标签通常应是：

```text
0 到 num_classes - 1
```

如果三分类标签写成 `1, 2, 3`，类别 `3` 会越界。

#### 坑 5：只看训练准确率

训练准确率高不等于泛化好。

如果训练集准确率接近 100%，测试集很差，说明过拟合或数据分布不一致。

---

### 九、从 MLP 连接到 Transformer

MLP 不是过时模型。

它是 Transformer 里的核心组件之一。

Transformer block 通常包含：

```text
self-attention + MLP/FFN
```

其中 FFN 可以理解成对每个 token 单独应用的 MLP：

```text
hidden -> larger hidden -> activation -> hidden
```

典型结构是：

```text
Linear(d_model, 4 * d_model)
Activation
Linear(4 * d_model, d_model)
```

本讲的 MLP 是：

```text
Linear(2, 16)
ReLU
Linear(16, 3)
```

区别只是维度和使用场景不同。

所以理解 MLP，对理解 Transformer FFN、分类 head、reward model head 都有帮助。

---

### 十、面试怎么讲这个实验

如果面试官问“如何用 PyTorch 写一个分类器”，可以这样回答：

```text
我会先准备输入特征和 long 类型类别标签，然后定义一个 MLP，例如 Linear + ReLU + Linear，输出 shape 是 [batch, num_classes] 的 logits。训练时直接把 logits 和标签传给 CrossEntropyLoss，不手动 softmax。每个 step 执行 zero_grad、backward 和 optimizer.step，并用 argmax 计算 accuracy。
```

如果追问“为什么分类用交叉熵而不是 MSE”，可以回答：

```text
分类任务本质是在学习类别分布。交叉熵直接最大化正确类别的对数似然，和 softmax 分类模型匹配；MSE 把类别当连续数值处理，优化目标和分类概率建模不匹配，通常训练效果和梯度性质都更差。
```

如果追问“MLP 和 Transformer 有什么关系”，可以回答：

```text
Transformer block 里的 FFN 本质就是对每个 token 的 hidden state 应用一个 MLP，通常先升维、经过激活函数，再降回 d_model。它负责逐 token 的非线性特征变换，而 attention 负责 token 间信息交互。
```

---

### 十一、小练习

#### 练习 1

把隐藏层维度从 `16` 改成：

```text
4, 32, 128
```

观察训练速度、训练准确率和测试准确率变化。

#### 练习 2

把 `ReLU` 替换成：

```text
Tanh, GELU, Sigmoid
```

比较 loss 下降速度。

#### 练习 3

故意在 `CrossEntropyLoss` 前加 `softmax`。

观察训练是否变慢或不稳定。

#### 练习 4

把标签改成 float 类型，观察报错信息。

回答：为什么分类标签必须是 long 类型类别 id？

#### 练习 5

把训练集样本减少到每类 5 个，隐藏层改成 128。

观察训练集和测试集 accuracy。

回答：这是否出现了过拟合？

---

### 本讲总结

这一讲实现了一个最小 MLP 分类器。

核心结论如下：

1. MLP 由线性层、激活函数和输出层组成。
2. 激活函数提供非线性，否则多层线性层仍等价于一个线性层。
3. 分类模型输出 logits，而不是直接输出类别概率。
4. `CrossEntropyLoss` 接收 raw logits 和 long 类型类别标签。
5. 训练时不要在交叉熵前手动 softmax。
6. accuracy 是评估指标，不是训练目标。
7. MLP 是 Transformer FFN、分类头和 reward head 的基础结构。

下一讲，我们手写交叉熵损失。

也就是把 `CrossEntropyLoss` 拆开，看清楚 softmax、log、负对数似然和数值稳定性之间的关系。

## 第 3 讲：手写交叉熵损失

### 本讲目标

学完本讲，你应该能做到六件事：

1. 手写一个基础版 cross entropy loss。
2. 解释 logits、softmax、log probability 和 label 的关系。
3. 说清楚交叉熵为什么等价于正确类别的负对数概率。
4. 理解为什么要用 `logsumexp` 做数值稳定。
5. 验证手写版本和 `nn.CrossEntropyLoss()` 的结果一致。
6. 把交叉熵连接到 LLM 的 next-token prediction。

上一讲我们使用了：

```python
nn.CrossEntropyLoss()
```

这一讲把它拆开。

你会看到，分类训练最核心的目标其实很简单：

```text
让正确类别的概率尽可能高。
```

交叉熵就是这个目标的标准实现。

---

### 一、从一个样本开始

假设模型对一个三分类样本输出 logits：

```python
logits = torch.tensor([2.0, 1.0, -1.0])
```

真实标签是：

```python
label = 0
```

意思是正确类别是第 0 类。

logits 不是概率。

要变成概率，需要 softmax：

```text
prob_i = exp(logit_i) / sum_j exp(logit_j)
```

如果第 0 类概率越高，模型越好。

交叉熵 loss 对单样本就是：

```text
loss = -log(prob_correct_class)
```

也就是：

```text
正确类别概率越接近 1，loss 越接近 0。
正确类别概率越接近 0，loss 越大。
```

---

### 二、最朴素的手写版本

先写一个最直观版本。

```python
import torch


def cross_entropy_naive(logits, labels):
    probs = torch.softmax(logits, dim=-1)
    correct_probs = probs[torch.arange(labels.shape[0]), labels]
    loss = -torch.log(correct_probs)
    return loss.mean()


logits = torch.tensor([
    [2.0, 1.0, -1.0],
    [0.1, 0.2, 3.0],
])
labels = torch.tensor([0, 2])

loss = cross_entropy_naive(logits, labels)
print(loss)
```

这里的 shape 是：

```text
logits: [batch, num_classes]
labels: [batch]
```

`probs` 也是：

```text
[batch, num_classes]
```

这一行最关键：

```python
correct_probs = probs[torch.arange(labels.shape[0]), labels]
```

它的意思是：

```text
对每个样本，取出真实类别对应的概率。
```

如果 batch 有两个样本，labels 是 `[0, 2]`，那就取：

```text
第 0 个样本的第 0 类概率
第 1 个样本的第 2 类概率
```

---

### 三、和 PyTorch 内置实现对齐

验证手写版本和 PyTorch 是否一致。

```python
import torch
from torch import nn


def cross_entropy_naive(logits, labels):
    probs = torch.softmax(logits, dim=-1)
    correct_probs = probs[torch.arange(labels.shape[0]), labels]
    loss = -torch.log(correct_probs)
    return loss.mean()


torch.manual_seed(42)

logits = torch.randn(4, 5)
labels = torch.tensor([0, 3, 1, 4])

loss_manual = cross_entropy_naive(logits, labels)
loss_torch = nn.CrossEntropyLoss()(logits, labels)

print(loss_manual)
print(loss_torch)
print(torch.allclose(loss_manual, loss_torch))
```

通常会输出 `True`。

这说明我们对交叉熵的理解是对的。

但这个 naive 版本有一个问题：数值稳定性不够。

---

### 四、为什么 naive softmax 不稳定

softmax 里有指数函数。

如果 logits 很大：

```python
logits = torch.tensor([[1000.0, 999.0, 998.0]])
```

直接计算：

```text
exp(1000)
```

会溢出。

计算机会得到 `inf`。

然后再除法，就可能得到 `nan`。

但数学上 softmax 有一个重要性质：

```text
softmax(x) = softmax(x - max(x))
```

也就是说，给所有 logits 同时减去最大值，概率不变。

例如：

```text
[1000, 999, 998] -> [0, -1, -2]
```

这样指数就不会爆炸。

---

### 五、稳定版 softmax

先手写一个稳定版 softmax。

```python
def stable_softmax(logits):
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    exp_values = torch.exp(shifted)
    probs = exp_values / exp_values.sum(dim=-1, keepdim=True)
    return probs
```

再写稳定版 cross entropy：

```python
def cross_entropy_stable_softmax(logits, labels):
    probs = stable_softmax(logits)
    correct_probs = probs[torch.arange(labels.shape[0]), labels]
    loss = -torch.log(correct_probs)
    return loss.mean()
```

这比 naive 版本安全一些。

但工业实现通常更进一步：直接计算 `log_softmax`，避免先得到概率再取 log。

---

### 六、用 logsumexp 写稳定交叉熵

交叉熵可以直接写成：

```text
loss = -log_softmax(logits)[correct_class]
```

而：

```text
log_softmax(logits_i) = logits_i - log(sum_j exp(logits_j))
```

其中：

```text
log(sum_j exp(logits_j))
```

叫 `logsumexp`。

稳定版实现是：

```python
def cross_entropy_logsumexp(logits, labels):
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.exp(shifted).sum(dim=-1))
    correct_logits = logits[torch.arange(labels.shape[0]), labels]
    loss = log_sum_exp - correct_logits
    return loss.mean()
```

为什么 loss 是：

```text
log_sum_exp - correct_logit
```

因为：

```text
-log_softmax(correct)
= -(correct_logit - log_sum_exp)
= log_sum_exp - correct_logit
```

这就是交叉熵的稳定核心形式。

---

### 七、用 torch.logsumexp 更简洁

PyTorch 已经提供了稳定的 `torch.logsumexp`。

所以可以写得更短：

```python
def cross_entropy_with_logsumexp(logits, labels):
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    correct_logits = logits[torch.arange(labels.shape[0]), labels]
    loss = log_sum_exp - correct_logits
    return loss.mean()
```

验证：

```python
torch.manual_seed(42)

logits = torch.randn(8, 10) * 5
labels = torch.randint(0, 10, (8,))

manual_loss = cross_entropy_with_logsumexp(logits, labels)
torch_loss = torch.nn.functional.cross_entropy(logits, labels)

print(manual_loss)
print(torch_loss)
print(torch.allclose(manual_loss, torch_loss))
```

如果实现正确，结果应该非常接近。

---

### 八、为什么交叉熵会推动正确类别 logit 变大

直觉上，loss 是：

```text
loss = -log(prob_correct)
```

要让 loss 变小，就要让 `prob_correct` 变大。

而 softmax 概率由 logits 决定。

所以训练会推动：

1. 正确类别 logit 上升。
2. 错误类别 logit 相对下降。

对于单样本，交叉熵对 logits 的梯度有一个经典形式：

```text
grad = probs - one_hot(label)
```

这非常重要。

它说明：

1. 正确类别：`prob - 1`，通常是负数，梯度下降会提高正确 logit。
2. 错误类别：`prob - 0`，是正数，梯度下降会降低错误 logit。

这也是 softmax + cross entropy 梯度简洁稳定的原因。

---

### 九、和 LLM next-token prediction 的关系

LLM 训练时，每个位置都在做分类。

如果词表大小是 `vocab_size`，模型输出：

```text
logits: [batch, seq_len, vocab_size]
```

标签是下一个 token：

```text
labels: [batch, seq_len]
```

训练时通常把前两维展平：

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
)
```

这和本讲的分类器完全一样。

只不过：

1. 类别数从 3 变成几万或几十万。
2. 每个 token 位置都是一个分类样本。
3. 通常要 mask 掉 padding 或不需要训练的位置。

所以理解交叉熵，就是理解 LLM 预训练 loss 的核心。

---

### 十、ignore_index 与 mask

真实训练中，有些位置不应该计算 loss。

例如 padding token。

PyTorch 支持：

```python
loss = torch.nn.functional.cross_entropy(
    logits,
    labels,
    ignore_index=-100,
)
```

如果某个 label 是 `-100`，这一项就不会参与 loss。

在 LLM SFT 中，常用做法是：

1. 用户 prompt 位置 label 设为 `-100`。
2. assistant answer 位置保留真实 token id。

这样模型只学习生成 assistant 答案，而不学习复述用户输入。

这和第二册多次讲过的 loss mask 是同一件事。

---

### 十一、常见工程坑

#### 坑 1：对 logits 先 softmax 再 CrossEntropyLoss

不要这么做。

传 raw logits。

#### 坑 2：label 是 one-hot

`CrossEntropyLoss` 默认需要类别 id，不是 one-hot。

如果 label 是 one-hot，要么转成类别 id，要么用其他合适 loss。

#### 坑 3：label dtype 不是 long

分类 label 应该是：

```python
torch.long
```

#### 坑 4：类别维度放错

普通分类：

```text
[batch, num_classes]
```

语言模型展平后：

```text
[batch * seq_len, vocab_size]
```

#### 坑 5：没有处理 padding

padding 位置参与 loss，会污染训练。

#### 坑 6：logits 太大导致 naive 实现溢出

手写时要用稳定版 `logsumexp`。

---

### 十二、面试怎么讲交叉熵

如果面试官问“交叉熵怎么实现”，可以这样回答：

```text
对分类任务，模型输出 raw logits。交叉熵等于正确类别概率的负对数，也就是 -log softmax(logits)[label]。为了数值稳定，实际实现不会先显式 softmax 再 log，而是使用 logsumexp，写成 logsumexp(logits) - correct_logit。
```

如果追问“为什么 CrossEntropyLoss 前不要 softmax”，可以回答：

```text
因为 PyTorch 的 CrossEntropyLoss 内部已经包含 log_softmax 和 NLLLoss，并且做了数值稳定优化。提前 softmax 不仅重复计算，还会降低数值稳定性，让 loss 接收到非预期输入。
```

如果追问“LLM 的 loss 是什么”，可以回答：

```text
LLM 的 next-token prediction 本质是对词表做分类。每个位置输出 vocab_size 维 logits，用下一个 token id 作为 label，计算 cross entropy。训练时通常把 batch 和 sequence 维展平，并用 ignore_index mask 掉不参与训练的位置。
```

---

### 十三、小练习

#### 练习 1

用 `torch.softmax + torch.log` 实现 naive cross entropy。

再和 `F.cross_entropy` 对比结果。

#### 练习 2

构造 logits：

```python
torch.tensor([[1000.0, 999.0, 998.0]])
```

观察 naive softmax 是否溢出。

再用 `torch.logsumexp` 实现稳定版本。

#### 练习 3

手写一个支持 `ignore_index=-100` 的 cross entropy。

提示：先构造 mask，只对 label 不等于 `-100` 的样本求平均。

#### 练习 4

假设 logits shape 是：

```text
[batch, seq_len, vocab_size]
```

labels shape 是：

```text
[batch, seq_len]
```

写出如何 reshape 后传入 `F.cross_entropy`。

---

### 本讲总结

这一讲手写了交叉熵损失。

核心结论如下：

1. 分类交叉熵等于正确类别概率的负对数。
2. `CrossEntropyLoss` 接收 raw logits 和 long 类型类别 id。
3. 朴素 softmax 版本容易数值溢出。
4. 稳定实现使用 `logsumexp(logits) - correct_logit`。
5. `CrossEntropyLoss = log_softmax + NLLLoss`。
6. LLM 的 next-token prediction 本质是对词表做交叉熵分类。
7. `ignore_index` 和 loss mask 是 SFT、padding 和对话训练中的关键细节。

下一讲，我们手写一个反向传播小例子。

也就是不依赖 PyTorch autograd，亲手算一遍前向、局部梯度和链式法则。

## 第 4 讲：手写反向传播小例子

### 本讲目标

学完本讲，你应该能做到六件事：

1. 不依赖 PyTorch autograd，手算一个小计算图的反向传播。
2. 理解局部梯度、上游梯度和链式法则的关系。
3. 手写一个两参数线性模型的梯度。
4. 用 PyTorch autograd 验证手算结果。
5. 说清楚为什么深度学习框架需要计算图。
6. 把反向传播和大模型训练中的 loss backward 联系起来。

前三讲我们一直在用：

```python
loss.backward()
```

这一讲把 `backward` 拆开。

你会看到，反向传播不是魔法。

它就是链式法则在计算图上的系统应用。

---

### 一、从最小计算图开始

考虑一个简单函数：

```text
z = x * y + y
```

可以拆成两个中间变量：

```text
a = x * y
z = a + y
```

假设：

```text
x = 2
y = 3
```

前向计算：

```text
a = 2 * 3 = 6
z = 6 + 3 = 9
```

我们想求：

```text
dz/dx
dz/dy
```

手算：

```text
z = xy + y
dz/dx = y = 3
dz/dy = x + 1 = 3
```

这里 `dz/dy = x + 1`，因为 y 走了两条路径：

1. 通过 `x * y` 影响 z。
2. 直接通过 `+ y` 影响 z。

这就是反向传播中“梯度累加”的来源。

---

### 二、用局部梯度理解链式法则

计算图：

```text
x ----\
       (*) -> a ----\
y ----/              (+) -> z
y ------------------/
```

局部梯度：

```text
a = x * y
da/dx = y
da/dy = x

z = a + y
dz/da = 1
dz/dy_direct = 1
```

反向传播从输出开始。

输出对自己的梯度是：

```text
dz/dz = 1
```

先传到 `a`：

```text
dz/da = 1
```

再传到 `x`：

```text
dz/dx = dz/da * da/dx = 1 * y = 3
```

传到 `y` 有两条路径。

第一条通过乘法：

```text
dz/dy_via_a = dz/da * da/dy = 1 * x = 2
```

第二条直接加法：

```text
dz/dy_direct = 1
```

所以：

```text
dz/dy = 2 + 1 = 3
```

反向传播的核心就是：

```text
每个节点把上游梯度乘以局部梯度，再把来自多条路径的梯度相加。
```

---

### 三、用 PyTorch 验证

```python
import torch


x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

a = x * y
z = a + y

z.backward()

print(x.grad)  # tensor(3.)
print(y.grad)  # tensor(3.)
```

PyTorch 的结果和手算一致。

这说明 autograd 做的事就是：

1. 记录前向计算图。
2. 从 loss 开始反向遍历。
3. 对每个操作应用局部梯度。
4. 把多路径梯度累加到叶子张量的 `.grad`。

---

### 四、线性回归的一步反向传播

现在回到第 1 讲的线性模型。

单样本：

```text
y_hat = w * x + b
loss = (y_hat - y)^2
```

设：

```text
x = 2
y = 7
w = 1
b = 0
```

前向计算：

```text
y_hat = 1 * 2 + 0 = 2
error = y_hat - y = 2 - 7 = -5
loss = error^2 = 25
```

我们要求：

```text
dloss/dw
dloss/db
```

拆成计算图：

```text
mul = w * x
y_hat = mul + b
error = y_hat - y
loss = error^2
```

局部梯度：

```text
dloss/derror = 2 * error = -10
derror/dy_hat = 1
dy_hat/dmul = 1
dy_hat/db = 1
dmul/dw = x = 2
```

所以：

```text
dloss/dw = dloss/derror * derror/dy_hat * dy_hat/dmul * dmul/dw
          = -10 * 1 * 1 * 2
          = -20
```

```text
dloss/db = dloss/derror * derror/dy_hat * dy_hat/db
          = -10 * 1 * 1
          = -10
```

梯度为负，说明如果使用梯度下降：

```text
w = w - lr * grad_w
```

那么 `w` 会增大。

这符合直觉：当前预测 `2`，真实值 `7`，模型输出太小，需要增大 `w` 或 `b`。

---

### 五、用代码手写这一轮梯度

```python
x = 2.0
y = 7.0
w = 1.0
b = 0.0

# forward
mul = w * x
y_hat = mul + b
error = y_hat - y
loss = error ** 2

# backward
d_loss = 1.0
d_error = d_loss * 2 * error
d_y_hat = d_error * 1.0
d_b = d_y_hat * 1.0
d_mul = d_y_hat * 1.0
d_w = d_mul * x

print("loss:", loss)
print("d_w:", d_w)
print("d_b:", d_b)
```

输出应该是：

```text
loss: 25.0
d_w: -20.0
d_b: -10.0
```

这就是一个手写反向传播。

---

### 六、用 PyTorch 验证线性回归梯度

```python
import torch


x = torch.tensor(2.0)
y = torch.tensor(7.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_hat = w * x + b
loss = (y_hat - y) ** 2

loss.backward()

print("loss:", loss.item())
print("w.grad:", w.grad.item())
print("b.grad:", b.grad.item())
```

结果应该是：

```text
loss: 25.0
w.grad: -20.0
b.grad: -10.0
```

这再次说明：PyTorch autograd 和手算链式法则是一回事。

---

### 七、批量样本的梯度

真实训练通常是 batch。

如果：

```text
loss = mean((xw + b - y)^2)
```

那么梯度是所有样本梯度的平均。

```text
dloss/dw = mean(2 * error_i * x_i)
dloss/db = mean(2 * error_i)
```

这解释了为什么 batch size 会影响梯度估计。

如果取 `sum` 而不是 `mean`：

```text
loss = sum((xw + b - y)^2)
```

梯度大小会随 batch size 增大。

所以工程中常用 mean，使不同 batch size 下 loss 尺度更稳定。

---

### 八、为什么需要计算图

深度模型包含大量操作。

例如 Transformer 里有：

1. embedding。
2. matmul。
3. attention。
4. softmax。
5. layer norm。
6. MLP。
7. residual connection。
8. cross entropy。

如果手写所有梯度，会非常复杂。

PyTorch 的 autograd 做了两件事：

#### 1. 前向时记录计算图

每个 Tensor 记录自己由哪个操作得到。

#### 2. 反向时自动应用链式法则

从 loss 开始，沿图反向传播梯度。

这让我们可以专注模型结构和 loss，而不用手写每个参数的梯度。

但作为算法工程师，必须理解 autograd 背后的链式法则。

否则遇到梯度为 `None`、梯度爆炸、detach、no_grad、in-place op 报错时就很难 debug。

---

### 九、常见 autograd 坑

#### 坑 1：非叶子节点的 grad 默认不保留

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y ** 2
z.backward()

print(x.grad)  # 有
print(y.grad)  # 通常是 None
```

`x` 是叶子张量。

`y` 是中间结果。

PyTorch 默认只把梯度存到叶子张量。

如果想看中间变量梯度，需要：

```python
y.retain_grad()
```

#### 坑 2：误用 detach

```python
y = model(x).detach()
loss = criterion(y, target)
```

`detach()` 会切断计算图。

模型参数收不到梯度。

#### 坑 3：在 no_grad 里做 forward

```python
with torch.no_grad():
    y_hat = model(x)
    loss = criterion(y_hat, y)
loss.backward()
```

这样不会构建计算图，无法反向传播。

`no_grad` 适合验证和参数更新，不适合训练 forward。

#### 坑 4：in-place 操作破坏计算图

某些原地操作可能覆盖反向传播需要的值。

例如带下划线的操作：

```python
x.relu_()
```

不是所有 in-place 都错，但出错时要警惕。

#### 坑 5：重复 backward 没有 retain_graph

默认情况下，`backward()` 后计算图会被释放。

如果要对同一图多次 backward，需要：

```python
loss.backward(retain_graph=True)
```

但通常不建议随便这么做，因为会增加显存占用。

---

### 十、和大模型训练的关系

大模型训练中：

```python
loss.backward()
```

背后就是同样的链式法则。

区别是计算图巨大得多。

例如一次 LLM forward 包含：

1. token embedding。
2. 多层 Transformer block。
3. attention score。
4. softmax attention。
5. MLP。
6. layer norm。
7. logits projection。
8. cross entropy loss。

反向传播会计算每个参数对 loss 的梯度。

然后优化器用这些梯度更新参数。

如果训练不稳定，可能来自：

1. 梯度爆炸。
2. 梯度消失。
3. mixed precision 溢出。
4. loss scale 不合适。
5. in-place 操作。
6. 错误 detach。
7. 梯度同步问题。

所以理解小计算图，有助于 debug 大模型训练。

---

### 十一、面试怎么讲反向传播

如果面试官问“反向传播是什么”，可以这样回答：

```text
反向传播是链式法则在计算图上的高效应用。前向时模型构建计算图并保存必要中间量；反向时从 loss 的梯度 1 开始，沿计算图反向传播，每个节点用上游梯度乘以局部梯度，并把多条路径的梯度累加到参数上。
```

如果追问“为什么梯度会累加”，可以回答：

```text
因为一个变量可能通过多条路径影响最终 loss。根据多元链式法则，总梯度等于所有路径贡献之和。PyTorch 中同一个参数被多处使用时，grad 也会累加。
```

如果追问“autograd 为什么需要计算图”，可以回答：

```text
因为模型由很多基础操作组合而成，反向传播需要知道每个中间变量由什么操作产生，以及局部梯度怎么计算。计算图记录了这些依赖关系，使框架能自动按拓扑顺序应用链式法则。
```

---

### 十二、小练习

#### 练习 1

手算下面函数在 `x=2` 时的梯度：

```text
z = (x^2 + 3x)^2
```

再用 PyTorch 验证。

#### 练习 2

手写一个两样本 MSE loss 的 `dw` 和 `db`。

```text
x = [1, 2]
y = [3, 5]
w = 1
b = 0
loss = mean((xw + b - y)^2)
```

#### 练习 3

构造一个例子，让某个变量通过两条路径影响 loss。

验证它的梯度等于两条路径贡献之和。

#### 练习 4

写一段代码使用 `detach()` 切断计算图，观察参数梯度变成 `None` 或无法更新。

---

### 本讲总结

这一讲手写了反向传播小例子。

核心结论如下：

1. 反向传播就是链式法则在计算图上的应用。
2. 每个节点把上游梯度乘以局部梯度，再传给前面的节点。
3. 一个变量通过多条路径影响 loss 时，梯度要相加。
4. PyTorch autograd 会自动记录计算图并执行反向传播。
5. 参数梯度默认存放在叶子张量的 `.grad` 中。
6. `detach`、`no_grad`、in-place 操作都可能影响计算图。
7. 大模型训练中的 `loss.backward()` 本质上仍然是同样的机制，只是计算图更大、更复杂。

下一讲，我们比较 SGD、Adam、AdamW。

也就是从“梯度怎么算”进入“梯度怎么用来更新参数”。

## 第 5 讲：比较 SGD、Adam、AdamW

### 本讲目标

学完本讲，你应该能做到六件事：

1. 说清楚优化器在训练循环中的作用。
2. 手写 SGD 的参数更新公式。
3. 理解 Momentum 为什么能加速收敛、减少震荡。
4. 理解 Adam 的一阶矩、二阶矩和自适应学习率。
5. 解释 AdamW 为什么比 Adam 加 L2 更适合训练大模型。
6. 在 PyTorch 中正确配置 `SGD`、`Adam`、`AdamW`。

上一讲我们解决了一个问题：

```text
梯度怎么算？
```

这一讲解决下一个问题：

```text
梯度算出来以后，参数怎么更新？
```

深度学习训练的核心循环可以写成：

```python
loss = model(x)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

其中：

1. `loss.backward()` 负责计算梯度。
2. `optimizer.step()` 负责用梯度更新参数。
3. `optimizer.zero_grad()` 负责清空旧梯度，避免梯度累加。

优化器决定了训练能不能稳定、能不能快速收敛、最终性能能不能充分发挥。

---

### 一、最朴素的梯度下降

假设参数是 `w`，loss 对 `w` 的梯度是：

```text
grad = dloss/dw
```

梯度下降更新公式：

```text
w = w - lr * grad
```

其中 `lr` 是学习率。

直觉是：

1. 梯度指向 loss 增大的方向。
2. 所以参数要沿负梯度方向走。
3. 学习率控制每一步走多远。

如果梯度为正：

```text
w = w - lr * 正数
```

`w` 会变小。

如果梯度为负：

```text
w = w - lr * 负数
```

`w` 会变大。

这和上一讲线性回归例子一致：预测太小，`dw` 为负，更新后 `w` 增大。

---

### 二、SGD：随机梯度下降

SGD 全称是 Stochastic Gradient Descent。

它的核心公式是：

```text
theta_t = theta_{t-1} - lr * grad_t
```

其中：

1. `theta` 表示参数。
2. `grad_t` 表示当前 batch 上的梯度。
3. `lr` 表示学习率。

为什么叫“随机”？

因为真实训练通常不是每次用全量数据算梯度，而是随机采样一个 mini-batch。

所以每一步的梯度只是全量梯度的有噪声估计。

SGD 的特点：

1. 实现简单。
2. 显存和额外状态少。
3. 泛化能力常常不错。
4. 对学习率比较敏感。
5. 在病态曲面上容易震荡。

所谓病态曲面，可以想象成一个狭长山谷。

SGD 可能在山谷两侧来回横跳，沿真正下降方向前进很慢。

---

### 三、手写 SGD 更新

下面手写一个最小 SGD。

```python
w = 1.0
lr = 0.1
grad = -20.0

w = w - lr * grad

print(w)  # 3.0
```

如果放到 PyTorch 里：

```python
import torch


w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
y = torch.tensor(7.0)

loss = (w * x - y) ** 2
loss.backward()

with torch.no_grad():
    w -= 0.1 * w.grad
    w.grad.zero_()

print(w)
```

注意更新参数时要放在 `torch.no_grad()` 里。

否则 PyTorch 会把参数更新本身也记录进计算图，导致图越来越乱。

---

### 四、PyTorch 中使用 SGD

标准写法：

```python
import torch
import torch.nn as nn


model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

x = torch.randn(16, 1)
y = 2 * x + 1

pred = model(x)
loss = criterion(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

推荐顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

也可以用：

```python
optimizer.zero_grad(set_to_none=True)
```

它会把梯度设为 `None`，通常更省显存和更快一点。

---

### 五、Momentum：给 SGD 加惯性

SGD 的问题是每一步只看当前 batch 梯度。

如果梯度噪声大，方向会抖动。

Momentum 的想法是：

```text
不要只看当前梯度，也参考过去一段时间的平均方向。
```

常见公式：

```text
v_t = momentum * v_{t-1} + grad_t
theta_t = theta_{t-1} - lr * v_t
```

其中 `v_t` 可以理解为速度。

如果连续很多步梯度方向一致，速度会积累，参数更新更快。

如果某个方向来回震荡，正负梯度会互相抵消，震荡会减弱。

PyTorch 写法：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
)
```

Momentum 适合：

1. CNN 训练。
2. 中小模型。
3. 对最终泛化比较敏感的传统监督训练。

但在大模型预训练和微调中，更常用的是 AdamW。

---

### 六、Adam：自适应学习率优化器

Adam 全称是 Adaptive Moment Estimation。

它同时维护两个状态：

1. 一阶矩：梯度的指数滑动平均，类似 Momentum。
2. 二阶矩：梯度平方的指数滑动平均，用来估计梯度尺度。

公式可以简化理解为：

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2

theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + eps)
```

其中：

1. `m_t` 表示梯度方向的平滑估计。
2. `v_t` 表示梯度大小的平滑估计。
3. `eps` 防止除零。

Adam 的直觉是：

```text
如果某个参数的梯度经常很大，就给它小一点的有效步长；如果某个参数的梯度经常很小，就给它相对大一点的有效步长。
```

所以 Adam 对不同参数有不同的自适应学习率。

这在稀疏特征、NLP、Transformer 中很有用。

---

### 七、Adam 的 bias correction

Adam 还有一个细节：偏置修正。

因为刚开始时：

```text
m_0 = 0
v_0 = 0
```

所以前几步的 `m_t` 和 `v_t` 会偏小。

Adam 使用：

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

最终更新：

```text
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

面试不一定要求你完整推导，但要知道：

```text
Adam 用一阶矩估计方向，用二阶矩调整尺度，并用 bias correction 修正初始阶段的估计偏差。
```

---

### 八、PyTorch 中使用 Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)
```

常见默认值：

1. `beta1 = 0.9`。
2. `beta2 = 0.999`。
3. `eps = 1e-8`。

Adam 的优点：

1. 收敛快。
2. 对学习率相对不那么敏感。
3. 对稀疏梯度友好。
4. 在 Transformer 和 NLP 任务中表现稳定。

Adam 的缺点：

1. 需要保存一阶矩和二阶矩，显存开销更大。
2. 有时泛化不如 SGD。
3. 如果 weight decay 处理不当，会和自适应缩放耦合。

这就引出了 AdamW。

---

### 九、L2 正则和 weight decay

训练模型时，我们常希望参数不要无限变大。

一种做法是在 loss 里加入 L2 正则：

```text
loss_total = loss_task + lambda * ||theta||^2
```

它对应的梯度会多一项：

```text
grad_total = grad_task + 2 * lambda * theta
```

在普通 SGD 中，L2 正则和 weight decay 很接近。

更新可以写成：

```text
theta = theta - lr * (grad + lambda * theta)
```

也就是：

```text
theta = (1 - lr * lambda) * theta - lr * grad
```

这看起来像每步把参数衰减一点，所以叫 weight decay。

但是在 Adam 里，事情不一样。

因为 Adam 会对梯度除以 `sqrt(v)`。

如果把 `lambda * theta` 直接加进梯度，它也会被 Adam 的自适应学习率缩放。

这会导致 weight decay 的效果和参数梯度统计耦合，不再是干净的“权重衰减”。

---

### 十、AdamW：解耦权重衰减

AdamW 的核心改动是：

```text
把 weight decay 从梯度更新中解耦出来。
```

Adam 的一种错误直觉写法是：

```text
grad = grad + weight_decay * theta
theta = AdamUpdate(theta, grad)
```

AdamW 的思路是：

```text
theta = theta - lr * AdamDirection
theta = theta - lr * weight_decay * theta
```

或者合并理解为：

```text
先按 Adam 的自适应方向更新，再直接对参数做衰减。
```

这样 weight decay 不会被二阶矩 `v_t` 缩放。

这就是“decoupled weight decay”。

大模型中常用 AdamW，因为它在 Transformer 训练中更稳定、更可控。

---

### 十一、PyTorch 中使用 AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)
```

大模型训练中常见配置可能是：

```text
optimizer = AdamW
lr = 1e-4 到 5e-4
betas = (0.9, 0.95) 或 (0.9, 0.999)
weight_decay = 0.01 到 0.1
```

具体取值取决于：

1. 模型规模。
2. batch size。
3. 训练 token 数。
4. 是否预训练、SFT、LoRA 微调。
5. 学习率调度器。
6. 数据质量和任务难度。

面试中不要死记某个唯一参数。

更重要的是解释为什么用 AdamW。

---

### 十二、哪些参数不做 weight decay

工程中通常不会对所有参数都做 weight decay。

常见不衰减的参数包括：

1. bias。
2. LayerNorm 的 weight。
3. RMSNorm 的 weight。
4. embedding 有时也会特殊处理。

原因是这些参数不是普通矩阵权重。

对归一化层参数做衰减，可能影响数值尺度控制。

典型分组写法：

```python
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if name.endswith("bias") or "norm" in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.95),
)
```

这是大模型训练代码中非常常见的细节。

---

### 十三、SGD、Adam、AdamW 对比

| 优化器 | 核心思想 | 优点 | 缺点 | 常见场景 |
|---|---|---|---|---|
| SGD | 沿负梯度方向更新 | 简单、省内存、泛化好 | 收敛慢、对学习率敏感 | 传统 CV、中小模型 |
| SGD + Momentum | 使用历史梯度方向 | 减少震荡、加速收敛 | 仍需调学习率 | CNN、监督训练 |
| Adam | 一阶矩 + 二阶矩自适应学习率 | 收敛快、稳定、适合稀疏梯度 | 状态开销大、weight decay 耦合问题 | NLP、Transformer |
| AdamW | Adam + 解耦 weight decay | 更适合大模型、正则更可控 | 状态开销大、超参仍需调 | LLM 预训练、SFT、微调 |

如果面试只能记一句话：

```text
SGD 直接用梯度更新，Momentum 加历史方向，Adam 用一阶矩和二阶矩做自适应学习率，AdamW 在 Adam 基础上解耦 weight decay，是大模型训练最常见选择。
```

---

### 十四、学习率和优化器不是独立的

优化器必须和学习率调度一起看。

大模型训练中常见策略：

1. warmup。
2. cosine decay。
3. linear decay。
4. constant with warmup。

为什么需要 warmup？

训练早期参数和优化器状态都不稳定。

如果一开始学习率过大，容易 loss spike 或直接发散。

所以先从小学习率线性升高到目标学习率。

为什么后期 decay？

训练后期希望参数在较优区域细调。

学习率太大会导致 loss 震荡，无法进一步收敛。

所以实际训练常见组合是：

```text
AdamW + warmup + cosine decay
```

---

### 十五、混合精度下的优化器细节

大模型训练通常使用 bf16 或 fp16。

这会影响优化器。

常见做法：

1. forward 和 backward 用 bf16/fp16。
2. optimizer state 通常保存为 fp32。
3. master weights 可能保存为 fp32。
4. fp16 训练需要 loss scaling 防止 underflow。

为什么优化器状态常用 fp32？

因为 AdamW 的一阶矩、二阶矩是长期累积统计。

如果精度太低，统计会不稳定，影响训练。

这也是 AdamW 比 SGD 更吃显存的原因之一。

每个参数至少还要保存：

1. 参数本身。
2. 梯度。
3. 一阶矩 `m`。
4. 二阶矩 `v`。

在大模型训练中，优化器状态显存经常比参数本身还大。

---

### 十六、常见工程坑

#### 坑 1：忘记 `zero_grad`

PyTorch 默认梯度会累加。

如果忘记清空：

```python
loss.backward()
optimizer.step()
```

下一轮梯度会叠加上一轮梯度。

除非你刻意做 gradient accumulation，否则这通常是 bug。

#### 坑 2：`zero_grad` 放错位置

推荐训练循环：

```python
for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
```

#### 坑 3：对所有参数都做 weight decay

LayerNorm、bias 通常不做 weight decay。

大模型代码里要做参数分组。

#### 坑 4：学习率照搬不调

AdamW 稳定不代表不用调学习率。

模型规模、batch size、训练 token 数变化时，学习率也要重新评估。

#### 坑 5：gradient clipping 缺失

训练不稳定时，可以加梯度裁剪：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

常见位置是：

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

#### 坑 6：混淆 Adam 的 weight_decay 和 AdamW

现在 PyTorch 里 `torch.optim.AdamW` 才是明确的解耦权重衰减实现。

训练 Transformer 时，优先选 AdamW。

---

### 十七、面试怎么回答

#### 问题 1：SGD 和 Adam 有什么区别？

可以回答：

```text
SGD 直接用当前 batch 的梯度按统一学习率更新参数；Adam 会维护梯度的一阶矩和二阶矩，一阶矩类似动量，二阶矩用于估计每个参数的梯度尺度，从而实现自适应学习率。Adam 通常收敛更快、对学习率没那么敏感，但需要额外优化器状态，显存开销更大。
```

#### 问题 2：AdamW 相比 Adam 改进了什么？

可以回答：

```text
AdamW 的核心是解耦 weight decay。Adam 如果把 L2 正则直接加到梯度里，这一项会被 Adam 的二阶矩自适应缩放，导致权重衰减和梯度统计耦合。AdamW 把参数衰减作为独立步骤应用，使正则效果更可控，因此在 Transformer 和大模型训练中更常用。
```

#### 问题 3：为什么大模型常用 AdamW？

可以回答：

```text
大模型参数量大、梯度尺度差异明显，AdamW 的自适应学习率有助于稳定训练；同时它解耦了 weight decay，更适合 Transformer 的正则化需求。实际训练中通常会配合 warmup、cosine decay、gradient clipping 和参数分组使用。
```

#### 问题 4：AdamW 有什么缺点？

可以回答：

```text
AdamW 需要为每个参数保存一阶矩和二阶矩，显存和通信开销大；超参仍然敏感，学习率、betas、weight decay、调度器都需要配合；在某些任务上泛化未必一定优于 SGD。
```

---

### 十八、小练习

#### 练习 1

用纯 Python 实现一次 SGD 更新。

输入：

```text
w = 1.0
grad = -20.0
lr = 0.1
```

输出更新后的 `w`。

#### 练习 2

用 PyTorch 分别训练同一个线性回归模型，比较：

1. `SGD(lr=0.01)`。
2. `SGD(lr=0.01, momentum=0.9)`。
3. `Adam(lr=0.01)`。
4. `AdamW(lr=0.01, weight_decay=0.01)`。

观察 loss 下降速度。

#### 练习 3

写一个 AdamW 参数分组函数。

要求：

1. bias 不做 weight decay。
2. norm 参数不做 weight decay。
3. 其他参数做 weight decay。

#### 练习 4

故意忘记 `optimizer.zero_grad()`，观察 loss 和梯度变化。

解释为什么梯度会越来越大。

---

### 本讲总结

这一讲比较了 SGD、Adam 和 AdamW。

核心结论如下：

1. 优化器负责把梯度转化为参数更新。
2. SGD 直接沿负梯度方向更新，简单省内存，但对学习率敏感。
3. Momentum 给 SGD 加历史方向，可以减少震荡、加速收敛。
4. Adam 使用一阶矩估计方向，使用二阶矩调整每个参数的有效学习率。
5. AdamW 把 weight decay 从梯度更新中解耦出来，更适合训练 Transformer 和大模型。
6. 大模型训练常见组合是 `AdamW + warmup + cosine decay + gradient clipping`。
7. 工程中要注意 `zero_grad`、参数分组、混合精度和优化器状态显存。

下一讲，我们进入 learning rate scheduler。

也就是学习率如何随训练步数变化。

## 第 6 讲：Learning Rate Scheduler

### 本讲目标

学完本讲，你应该能做到六件事：

1. 解释为什么训练中学习率不能一直固定不变。
2. 说清楚 warmup、decay、cosine scheduler 的直觉。
3. 在 PyTorch 中正确使用 learning rate scheduler。
4. 区分按 step 更新和按 epoch 更新的 scheduler。
5. 理解大模型训练中 `AdamW + warmup + cosine decay` 的常见组合。
6. 能根据 loss 曲线判断学习率是否过大、过小或调度不合理。

上一讲我们讲了优化器：

```text
梯度算出来以后，参数怎么更新？
```

这一讲讲学习率调度器：

```text
更新步长应该如何随训练过程变化？
```

优化器决定“往哪里走”。

学习率决定“每一步走多远”。

学习率调度器决定“不同训练阶段走多远”。

---

### 一、为什么不能一直用固定学习率

最简单的训练写法是固定学习率：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

这表示从训练第一步到最后一步，学习率都等于 `3e-4`。

固定学习率的问题是：训练不同阶段对学习率的需求不同。

训练早期：

1. 参数还很随机。
2. loss 很大。
3. 梯度方向不稳定。
4. 优化器的一阶矩、二阶矩还没稳定。

这时学习率太大，容易 loss spike，甚至直接发散。

训练中期：

1. 模型已经找到大致下降方向。
2. 需要较大的学习率快速降低 loss。
3. 如果学习率太小，训练会非常慢。

训练后期：

1. 参数接近较优区域。
2. 需要更小步长精细调整。
3. 学习率太大可能在最优点附近震荡。

所以实际训练常常使用：

```text
先升高，再保持或逐渐降低。
```

这就是 learning rate scheduler 的核心作用。

---

### 二、学习率过大和过小的现象

学习率过大时，常见现象：

1. loss 剧烈震荡。
2. loss 突然变成 `nan`。
3. 梯度范数暴涨。
4. validation loss 不稳定。
5. 生成模型输出很快退化。

学习率过小时，常见现象：

1. loss 下降很慢。
2. 训练很多 step 后几乎没有改善。
3. 梯度方向看起来正常，但参数变化太小。
4. 同样算力下模型欠拟合。

学习率调度不合理时，常见现象：

1. warmup 太短，训练初期不稳定。
2. warmup 太长，前期学习太慢。
3. decay 太快，模型还没学够学习率就变小。
4. decay 太慢，后期 loss 震荡。

面试中如果问训练不收敛，学习率和 scheduler 一定是优先排查项。

---

### 三、Warmup：先小步走稳

Warmup 是大模型训练中非常常见的策略。

它的做法是：

```text
训练前若干步，学习率从 0 或很小的值线性升高到目标学习率。
```

例如目标学习率是 `3e-4`，warmup steps 是 `1000`。

第 0 步：

```text
lr = 0
```

第 500 步：

```text
lr = 1.5e-4
```

第 1000 步：

```text
lr = 3e-4
```

线性 warmup 公式：

```text
lr_t = base_lr * t / warmup_steps
```

其中 `t <= warmup_steps`。

为什么大模型需要 warmup？

1. 初始参数还没有形成稳定表示。
2. AdamW 的动量统计刚开始不可靠。
3. 大 batch 训练中早期梯度可能很尖锐。
4. 混合精度训练中早期数值更容易不稳定。

Warmup 的直觉很简单：

```text
刚开始不要迈大步，先让模型和优化器状态热起来。
```

---

### 四、StepLR：每隔一段时间降一次

`StepLR` 是最容易理解的 scheduler。

它每隔固定 epoch 把学习率乘以一个系数。

例如：

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1,
)
```

含义是：

```text
每 10 个 epoch，学习率乘以 0.1。
```

如果初始学习率是 `1e-2`：

```text
epoch 0-9:   lr = 1e-2
epoch 10-19: lr = 1e-3
epoch 20-29: lr = 1e-4
```

典型训练循环：

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    scheduler.step()
```

注意：这里 `scheduler.step()` 放在每个 epoch 结束后。

StepLR 常见于传统 CV 训练。

但在 LLM 训练中，更常用按 step 更新的 warmup + cosine decay。

---

### 五、ExponentialLR：指数衰减

`ExponentialLR` 每一步或每个 epoch 都按固定比例衰减学习率。

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.99,
)
```

更新形式：

```text
lr_t = lr_0 * gamma^t
```

如果 `gamma=0.99`，学习率会缓慢下降。

如果 `gamma=0.9`，下降会很快。

指数衰减的特点：

1. 曲线平滑。
2. 实现简单。
3. `gamma` 不直观，需要调。
4. 训练后期可能衰减得过小。

它适合一些中小模型实验，但大模型训练中通常更偏好 cosine decay。

---

### 六、Cosine Decay：余弦退火

Cosine decay 是大模型训练中非常常见的策略。

它让学习率按余弦曲线从最大值平滑下降到最小值。

直觉是：

```text
前期下降慢一点，中后期平滑降低，最后以很小学习率收尾。
```

简化公式：

```text
lr_t = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
```

其中：

```text
progress = 当前步数 / 总训练步数
```

当 `progress = 0`：

```text
cos(0) = 1
lr = base_lr
```

当 `progress = 1`：

```text
cos(pi) = -1
lr = min_lr
```

Cosine decay 的优点：

1. 学习率变化平滑。
2. 后期自然变小，利于收敛。
3. 不需要手动指定多个下降节点。
4. 和 warmup 组合非常自然。

大模型训练常见组合：

```text
linear warmup + cosine decay
```

---

### 七、手写 warmup + cosine scheduler

先写一个纯 Python 版本。

```python
import math


def get_lr(step, total_steps, base_lr, min_lr, warmup_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


for step in [0, 100, 1000, 5000, 10000]:
    lr = get_lr(
        step=step,
        total_steps=10000,
        base_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=1000,
    )
    print(step, lr)
```

这个函数分两段：

1. warmup 阶段线性升高。
2. warmup 后按 cosine decay 降低。

真实训练中，只要每个 step 根据当前步数更新 optimizer 的 lr 即可。

---

### 八、用 LambdaLR 实现自定义 scheduler

PyTorch 中可以用 `LambdaLR` 自定义学习率倍率。

```python
import math
import torch


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + cosine * (1.0 - min_lr_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

使用方式：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = build_warmup_cosine_scheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    min_lr_ratio=0.1,
)

for step, batch in enumerate(dataloader):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

这里 `scheduler.step()` 每个训练 step 调一次。

这和 `StepLR` 常见的每个 epoch 调一次不同。

---

### 九、scheduler.step 放在哪里

常见顺序是：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
scheduler.step()
```

也就是先更新参数，再推进学习率调度器。

为什么？

因为当前 step 的参数更新使用当前学习率。

更新完成后，scheduler 进入下一 step 的学习率。

实际中要特别注意两件事：

1. 有些 scheduler 按 epoch 调用。
2. 有些 scheduler 按 optimizer step 调用。

大模型训练通常按 optimizer step 调用。

如果使用 gradient accumulation，要注意：

```text
scheduler.step() 应该和 optimizer.step() 对齐，而不是和每个 micro-batch 对齐。
```

错误示例：

```python
for micro_batch in loader:
    loss = compute_loss(model, micro_batch) / grad_accum_steps
    loss.backward()
    scheduler.step()  # 错：optimizer 还没 step
```

正确示例：

```python
for step, batch in enumerate(loader):
    loss = compute_loss(model, batch) / grad_accum_steps
    loss.backward()

    if (step + 1) % grad_accum_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```

---

### 十、Hugging Face 中的 scheduler

实际大模型训练中，经常使用 Hugging Face Transformers 的 scheduler。

典型写法：

```python
from transformers import get_cosine_schedule_with_warmup


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000,
)
```

训练循环：

```python
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

常见 scheduler 包括：

1. `get_linear_schedule_with_warmup`。
2. `get_cosine_schedule_with_warmup`。
3. `get_constant_schedule_with_warmup`。

SFT 和 LoRA 微调中，这些接口很常见。

---

### 十一、如何计算 total_steps

很多 scheduler 需要 `num_training_steps`。

如果没有 gradient accumulation：

```text
total_steps = num_epochs * len(dataloader)
```

如果有 gradient accumulation：

```text
total_steps = num_epochs * len(dataloader) // grad_accum_steps
```

如果是分布式训练，`len(dataloader)` 通常已经是当前进程看到的 step 数。

还要注意 drop last、断点续训、max_steps 等因素。

更稳妥的写法是显式设置：

```python
max_steps = 10000
warmup_steps = int(0.03 * max_steps)
```

很多大模型训练配置会写：

```text
warmup_ratio = 0.03
```

表示 warmup steps 占总训练步数的 3%。

---

### 十二、学习率调度和 batch size 的关系

学习率不是孤立超参。

它和 batch size 强相关。

常见经验是：

```text
batch size 变大，可以适当增大学习率。
```

原因是大 batch 的梯度估计更稳定，可以承受更大的步长。

但这不是无限成立。

batch size 太大可能带来：

1. 泛化变差。
2. 训练早期更需要 warmup。
3. 学习率过大导致 loss spike。
4. 通信和吞吐瓶颈。

大模型训练中，常见调参顺序是：

1. 固定模型和数据。
2. 选一个合理 global batch size。
3. 选 AdamW。
4. 选 base learning rate。
5. 设 warmup ratio。
6. 选 cosine 或 linear decay。
7. 观察 loss 曲线和梯度范数。

---

### 十三、不同任务的常见选择

#### 1. 从零预训练 LLM

常见组合：

```text
AdamW + linear warmup + cosine decay
```

原因：

1. 训练步数长。
2. 模型规模大。
3. 需要稳定起步。
4. 后期需要平滑收敛。

#### 2. SFT 微调

常见组合：

```text
AdamW + warmup + cosine decay 或 linear decay
```

SFT 学习率通常比预训练小。

因为模型已经有能力，微调主要是调整行为和格式。

#### 3. LoRA 微调

常见组合：

```text
AdamW + warmup + cosine/linear decay
```

LoRA 只训练少量参数，学习率有时可以比全参微调更大。

#### 4. 小模型教学实验

可以先用：

```text
固定学习率 或 StepLR
```

重点是先理解训练闭环，不必一开始就堆复杂调度器。

---

### 十四、常见工程坑

#### 坑 1：scheduler.step 调用次数错了

如果 scheduler 设计为按 step 更新，却按 epoch 更新，学习率变化会太慢。

如果设计为按 epoch 更新，却按 step 更新，学习率会衰减太快。

#### 坑 2：gradient accumulation 下 step 数算错

如果每 8 个 micro-batch 才 `optimizer.step()` 一次，那么 scheduler 也应该每 8 个 micro-batch 调一次。

否则 warmup 和 decay 都会被压缩。

#### 坑 3：恢复训练时没有恢复 scheduler state

断点续训时，不只要恢复：

1. model state。
2. optimizer state。
3. scheduler state。
4. global step。

如果 scheduler 没恢复，学习率会从头开始，训练曲线可能异常。

#### 坑 4：warmup_steps 大于 total_steps

这会导致训练全程都在 warmup。

模型还没到目标学习率，训练就结束了。

#### 坑 5：只看最终 loss，不看 lr 曲线

调参时建议记录：

1. train loss。
2. eval loss。
3. learning rate。
4. grad norm。

只看 loss 很难判断是优化器问题、学习率问题还是数据问题。

---

### 十五、面试怎么回答

#### 问题 1：为什么要做学习率调度？

可以回答：

```text
训练不同阶段需要不同步长。早期参数和优化器状态不稳定，学习率太大容易发散，所以常用 warmup；中期需要较大学习率快速下降；后期接近较优区域，需要降低学习率减少震荡并精细收敛。因此学习率调度能提升稳定性和最终效果。
```

#### 问题 2：warmup 的作用是什么？

可以回答：

```text
Warmup 是在训练初期把学习率从很小的值逐步升到目标学习率，避免一开始参数随机、梯度不稳定、Adam 统计不充分时使用过大学习率导致 loss spike 或发散。大模型训练中 warmup 几乎是标配。
```

#### 问题 3：cosine decay 为什么常用？

可以回答：

```text
Cosine decay 能让学习率从峰值平滑下降到较小值，不需要手动设多个下降节点；前中期保持足够训练步长，后期自然降低学习率帮助收敛。它和 linear warmup 组合简单稳定，所以在 LLM 预训练和微调中常见。
```

#### 问题 4：gradient accumulation 下 scheduler 怎么 step？

可以回答：

```text
Scheduler 应该和 optimizer.step 对齐，而不是和每个 micro-batch 对齐。因为只有 optimizer.step 才真正更新了一次参数。如果每个 micro-batch 都 scheduler.step，会导致 warmup 和 decay 过快，实际学习率曲线和预期不一致。
```

---

### 十六、小练习

#### 练习 1

手写一个函数，输入 `step`、`total_steps`、`base_lr`、`warmup_steps`，输出 linear warmup + linear decay 的学习率。

#### 练习 2

用 `LambdaLR` 实现 warmup + cosine decay，并打印前 20 个 step 的学习率。

#### 练习 3

构造一个有 gradient accumulation 的训练循环，确保 `scheduler.step()` 和 `optimizer.step()` 次数一致。

#### 练习 4

训练同一个小线性模型，比较固定学习率、StepLR、cosine decay 的 loss 曲线。

---

### 本讲总结

这一讲讲了 learning rate scheduler。

核心结论如下：

1. 学习率决定参数每一步更新多远，scheduler 决定不同训练阶段的步长变化。
2. 固定学习率简单，但不能适配训练早期、中期、后期的不同需求。
3. Warmup 可以提升训练早期稳定性，是大模型训练常见标配。
4. StepLR 适合传统训练，cosine decay 更常见于 LLM 预训练和微调。
5. `scheduler.step()` 要和 scheduler 类型匹配，按 step 还是按 epoch 不能混淆。
6. Gradient accumulation 下，scheduler 应该和 `optimizer.step()` 对齐。
7. 断点续训要恢复 scheduler state，否则学习率曲线会错。

下一讲，我们进入 Transformer 组件实战。

也就是从 PyTorch 基础训练闭环，进入 Attention、LayerNorm、MLP 等模块的手写实现。
