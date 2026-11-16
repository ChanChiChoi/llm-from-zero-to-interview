# 第 1 章 PyTorch 基础实战：把数学变成可运行的训练过程

神经网络的代码看起来常常只有几行：定义模型、计算损失、调用 `backward()`，再让优化器更新参数。让人困惑的往往不是 API 名字，而是一次更新里究竟发生了什么：损失为什么是这个数？`backward()` 算出了什么？参数又是谁改的？

我们先从四条简单数据开始，亲手算出一次线性回归的预测、误差、梯度和参数变化。接下来把输出换成类别分数，再把同一个训练循环接到语言模型的 token 预测上。最后再看优化器、学习率、混合精度和 checkpoint 怎样改变训练过程。每一步都围绕同一件事展开：模型给出预测后，训练系统怎样依据目标信号更新参数。

本章不要求 GPU，也不要求读者先熟悉大模型框架。代码中的数据和数值是专门构造的小例子，用来复核公式；它们不是模型效果或硬件性能的实测结论。若使用不同 PyTorch 版本，优先核对张量形状、梯度方向和更新顺序；随机数和浮点实现可能让末尾小数不同。

```mermaid
flowchart LR
    A["输入与参数"] --> B["前向计算：预测"]
    B --> C["损失：衡量误差"]
    C --> D["反向传播：计算梯度"]
    D --> E["优化器：更新参数"]
    E --> F["学习率调度与日志"]
    F --> B
```

这张图先给出本章的路线。最容易混淆的两步是求梯度和改参数：`loss.backward()` 只把梯度写入参数的 `.grad`，真正改变参数的是 `optimizer.step()`。我们先用手算把这两步分开，之后再逐渐补上批次、数值精度和恢复训练等实际问题。

## 1.1 从线性回归开始：一个参数更新到底改变了什么

### 1.1.1 先写清楚输入、目标和预测

先看一个熟悉的任务：用几项已知信息预测一个数值。一个样本有 $d$ 个特征，把 $B$ 个样本排成矩阵 $X$，形状就是 $[B,d]$。线性模型有一组权重 $w\in\mathbb{R}^{d}$ 和一个偏置 $b\in\mathbb{R}$。模型对这批样本的预测是：

```math
\hat{y}=Xw+b
```

偏置 $b$ 会加到这批样本的每一个预测上，因此 $\hat y$ 的形状是 $[B]$。本章约定样本放在第 0 维、特征放在第 1 维，预测和目标都用 $[B]$。这个约定看似细枝末节，却能防止后面出现“代码可以运行，实际却把不同样本两两相减”的形状错误。

把矩阵运算展开到单个样本，就得到：

```math
\hat y_i=\sum_{j=1}^{d}x_{ij}w_j+b
```

如果第 $i$ 个样本的真实目标是 $y_i$，先定义误差为预测减去目标。最直观的衡量方式之一是均方误差（Mean Squared Error，MSE）：

```math
L(w,b)=\frac{1}{B}\sum_{i=1}^{B}(\hat y_i-y_i)^2
```

平方把正误差和负误差都变成非负数，也让较大的错误占更大比重。代价是，异常值可能主导总损失。以后看到“loss 下降”，要记住它说的是这个数学目标变小了；还要检查是不是每个样本都变好，以及这个目标是否符合实际任务。

### 1.1.2 从误差算到梯度，再看参数怎样移动

现在把模型暂时固定为 $w=0,b=0$，输入取 $[1,2,3,4]$，目标取 $[3,5,7,9]$。模型当前还不会预测任何变化，所以四个预测都是 0。要让参数知道该往哪边改，先对损失求导。记第 $i$ 个样本的误差为 $e_i=\hat y_i-y_i$。因为预测对权重 $w_j$ 的变化率是对应特征 $x_{ij}$，链式法则给出：

```math
\frac{\partial L}{\partial w_j}
=\frac{2}{B}\sum_{i=1}^{B}e_i x_{ij}
```

偏置对每个预测的变化率都是 1，因此它的梯度是：

```math
\frac{\partial L}{\partial b}
=\frac{2}{B}\sum_{i=1}^{B}e_i
```

把这组数据代进去，MSE 是 $(9+25+49+81)/4=41$。每个样本对权重和偏置梯度的贡献如下：

| $x_i$ | $y_i$ | 当前预测 $\hat y_i$ | 误差 $e_i$ | 对 $w$ 的贡献 $2e_ix_i$ | 对 $b$ 的贡献 $2e_i$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0 | -3 | -6 | -6 |
| 2 | 5 | 0 | -5 | -20 | -10 |
| 3 | 7 | 0 | -7 | -42 | -14 |
| 4 | 9 | 0 | -9 | -72 | -18 |
| 平均 | — | — | — | -35 | -12 |

所以 $\partial L/\partial w=-35$、$\partial L/\partial b=-12$。负号表示：在当前点附近，把 $w$ 和 $b$ 调大，会让损失下降。参数按负梯度方向更新：

```math
w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}
```

学习率 $\eta>0$ 决定移动多远。它不是模型能力的分数。步子太小，有限训练时间里变化不明显；步子太大，可能越过损失较低的位置。下面的代码照着表格计算一次更新；手算和 PyTorch 会得到同一组梯度。

PyTorch 的自动微分系统会根据参与运算的张量建立动态计算图。当一个需要梯度的标量损失调用 `backward()` 时，系统从输出向输入反向应用局部导数，并把结果累积到叶子张量的 `.grad` 属性中。下面的代码同时展示了形状、梯度和一次手工更新：

> 代码状态：静态审阅，未运行（2026-11-16）；本次调整了学习率和输出，预期值由上表手算。

```python
import torch

torch.manual_seed(7)
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([3.0, 5.0, 7.0, 9.0])

w = torch.tensor([0.0], requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
prediction = x[:, 0] * w[0] + b
loss = ((prediction - y) ** 2).mean()
loss.backward()

print(prediction.shape)
print(round(loss.item(), 4))
print(round(w.grad.item(), 4), round(b.grad.item(), 4))

with torch.no_grad():
    w -= 0.01 * w.grad
    b -= 0.01 * b.grad

next_prediction = x[:, 0] * w[0] + b
next_loss = ((next_prediction - y) ** 2).mean()
print(round(w.item(), 4), round(b.item(), 4))
print(round(next_loss.item(), 4))
```
这个片段的输出为：

```text
torch.Size([4])
41.0
-35.0 -12.0
0.35 0.12
28.4532
```

表格把链式法则拆成了可核对的四行。PyTorch 输出的梯度正好是各行贡献的平均。学习率取 $0.01$ 后，参数变为 $w=0.35,b=0.12$；新预测为 $[0.47,0.82,1.17,1.52]$，损失从 41 降到约 28.45。它还没有拟合好，但这一步确实沿着降低当前 MSE 的方向移动。

注意，代码里的 `.grad` 并不会因为调用 `backward()` 自动变成参数的新值。反向传播计算并保存梯度；`torch.no_grad()` 里的两行才手动执行一次更新。实际训练通常把这一步交给优化器，后面会说明优化器还会保存哪些历史状态。

这里还做了一个选择：先把样本误差求和，再除以样本数。若改成求和，样本数翻倍时梯度也可能翻倍，原来的学习率就不再对应相同的更新尺度。看到梯度变大，既可能是模型状态变化，也可能只是有效样本数或损失归约方式变了。

所以比较两次训练时，要一并固定 batch 定义和尾批次处理。一个 epoch 的最后一批若只剩 3 条，而其他批次有 32 条，直接对每批 mean loss 再做等权平均，会让尾批次中的每条样本得到更大的 epoch 权重；按样本数汇总则不会。数据采样、梯度累积和日志聚合最好使用同一套有效样本计数。

目标值的单位也会影响损失尺度：把“元”改写为“分”，平方误差会放大一万倍，梯度也随之变化。预处理或目标缩放应记录在模型契约里，推理时再按相反变换还原。

### 1.1.3 梯度默认会相加：独立更新前要清零

刚才我们只反向传播了一次。训练循环会反复处理不同批次，而 PyTorch 不会猜测你要不要保留上一批的梯度：每次 `backward()` 都把新梯度加到现有 `.grad` 上。这样既能把多个损失的贡献相加，也能把几个小批次的梯度累积起来；如果你打算每个批次都单独更新，就必须在下一轮开始前清零。

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

parameter = torch.tensor(2.0, requires_grad=True)
first_loss = parameter ** 2
first_loss.backward()
first_grad = parameter.grad.item()

second_loss = 3.0 * parameter
second_loss.backward()
accumulated_grad = parameter.grad.item()

parameter.grad = None
third_loss = 4.0 * parameter
third_loss.backward()
cleared_grad = parameter.grad.item()

print(first_grad, accumulated_grad, cleared_grad)
```
输出是：

```text
4.0 7.0 4.0
```

第一次梯度是 $2w=4$，第二次梯度是 3，累积后得到 7。将 `.grad` 设为 `None` 是常用清零方式；`optimizer.zero_grad()` 则会对优化器管理的参数执行相应操作。是否使用 `set_to_none=True` 会影响清零后的状态表现和少量性能细节，实际项目应以目标 PyTorch 版本的文档和内存行为为准。

### 1.1.4 广播可以自动配形状，却不知道样本是否配对

下面两个张量看起来只差一个维度，但相减结果完全不同：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

prediction = torch.zeros(4)
target_vector = torch.ones(4)
target_column = torch.ones(4, 1)

print((prediction - target_vector).shape)
print((prediction - target_column).shape)
```
输出为：

```text
torch.Size([4])
torch.Size([4, 4])
```

第二种情况里，形状 $[4]$ 会先按广播规则看作 $[1,4]$，再和 $[4,1]$ 组合成 $[4,4]$。代码仍能运行，损失却不再比较同一个样本的预测和目标，而是在四个预测与四个目标之间做了两两组合。广播只负责让运算的形状兼容，不知道数据行之间应该怎样配对。对于回归任务，最好在损失前检查：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

prediction = torch.zeros(4)
target = torch.ones(4)
assert prediction.shape == target.shape
print("shape contract passed")
```
输出为：

```text
shape contract passed
```

`target` 应该与预测逐项对应。如果数据管道返回列向量，就在进入损失前明确地压缩或重塑维度。断言把这个数据契约写进代码；它尤其能挡住“没有报错，但训练目标已经变了”的问题。

### 1.1.5 把公式写成循环，看看框架替你省了什么

刚才的更新由 `torch.no_grad()` 里的两行手动完成。若把同一个计算重复多次，程序只需循环计算误差、梯度和参数更新，不需要 PyTorch 的计算图。下面用纯 Python 重写同一条规则。数据按 $y=2x+1$ 构造，因此这个小任务的目标参数是 $w=2,b=1$：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
x_values = [0.0, 1.0, 2.0, 3.0]
y_values = [1.0, 3.0, 5.0, 7.0]
w = 0.0
b = 0.0
learning_rate = 0.1

for _ in range(50):
    errors = [w * x + b - y for x, y in zip(x_values, y_values)]
    dw = 2.0 * sum(error * x for error, x in zip(errors, x_values)) / len(x_values)
    db = 2.0 * sum(errors) / len(errors)
    w -= learning_rate * dw
    b -= learning_rate * db

print(round(w, 3), round(b, 3))
```
输出为：

```text
2.001 0.998
```

纯 Python 版本没有计算图，也没有 `.grad`，却完成了同一组数学操作。PyTorch 替你做的是记录运算关系、按链式法则求导和管理梯度；它没有替你选对损失，也没有判断这组数据是否代表真实任务。

回归例子把一次更新拆成了数字，但它的输出仍是连续值。分类任务要把模型输出解释为有限个类别；困难在于，类别本身不能直接提供平滑的更新方向。下一节先看模型实际输出什么，再决定怎样把标签变成训练信号。

## 1.2 MLP 分类器：从连续预测到类别分数

回归模型回答“数值大约是多少”。分类模型看起来要回答另一个问题：“这张图片属于猫还是狗？”但如果模型直接输出“猫”，我们很难根据这个离散答案计算一个有用的梯度。实际做法是先给每个候选类别一个分数，再用标签和损失推动正确类别的分数上升。分类结果是训练目标的解释，分数才是训练过程实际操作的对象。

### 1.2.1 分类模型先输出分数，而不是类别

回归的输出是一个连续数。分类问题通常有 $C$ 个类别，模型对每个类别输出一个实数分数，称为 logit。对一个批次，logits 的形状通常是 $[B,C]$，第 $i$ 行对应一个样本，第 $c$ 列对应一个类别。logit 可以是负数，也不需要加起来等于 1；它只是后续概率计算的输入。

如果需要概率，可以使用 softmax：

```math
p_{ic}=\frac{\exp(z_{ic})}{\sum_{k=1}^{C}\exp(z_{ik})}
```

其中 $z_{ic}$ 是第 $i$ 个样本对类别 $c$ 的 logit，$p_{ic}$ 是对应概率。预测类别通常取最大 logit 的索引：

```math
\hat c_i=\arg\max_c z_{ic}
```

softmax 保持每一行概率和为 1，但在训练时通常不需要先手动调用 softmax。`torch.nn.CrossEntropyLoss` 接收未经归一化的 logits，并在内部完成数值更稳定的对数 softmax 与负对数似然组合。把 softmax 后的概率再次传给它，往往会改变数值行为，也让极端 logit 下的稳定性变差。

### 1.2.2 为什么需要隐藏层和 ReLU

一个单层线性分类器为：

```math
z=XW+b
```

无论输入特征怎样组合，决策边界仍然是线性的。两层网络在中间加入非线性函数：

```math
h=\mathrm{ReLU}(XW_1+b_1),\qquad
z=hW_2+b_2
```

ReLU 定义为：

```math
\mathrm{ReLU}(a)=\max(0,a)
```

如果没有 ReLU，两个线性变换可以合并成一个线性变换，增加层数却没有增加表达能力。ReLU 把输入空间切成不同区域，在每个区域内保持线性，但不同区域可以有不同的斜率，因此能够表达更复杂的决策边界。

“神经元死亡”来自 ReLU 在 $a<0$ 时梯度为零：如果某个单元长期落在负半轴，它可能很少收到更新。初始化、学习率、归一化和激活函数选择都会影响这个现象；不能看到一个负激活就断言模型已经失效，因为不同样本上的激活状态可能不同。

### 1.2.3 写一个可观察的 MLP

下面的模型使用二维输入和三个类别，数据仍是用于说明训练流程的人工构造。`forward` 返回 logits，不在模型内部做 softmax，这样损失函数可以直接使用原始分数。

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

torch.manual_seed(3)

features = torch.tensor([
    [-1.0, -1.0], [-0.8, -1.2],
    [1.0, -1.0], [1.2, -0.8],
    [0.0, 1.0], [0.2, 1.2],
])
labels = torch.tensor([0, 0, 1, 1, 2, 2])

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(80):
    logits = model(features)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    final_logits = model(features)
    final_loss = loss_fn(final_logits, labels)
    predicted = final_logits.argmax(dim=1)
    accuracy = (predicted == labels).float().mean()

print(predicted.tolist())
print(round(final_loss.item(), 4))
print(round(accuracy.item(), 4))
```
在本地 PyTorch 2.12.0 环境和上述随机种子下，输出为：

```text
[0, 0, 1, 1, 2, 2]
0.0869
1.0
```

这里的 loss 在第 80 次参数更新后重新前向计算，与预测和准确率对应同一组最终参数。输出是在本地 PyTorch 2.12.0 CPU 环境实测得到的教学结果；更换 PyTorch 版本、硬件或随机数实现后，小数位可能变化。类别完全可分时，训练集准确率达到 1.0 并不说明模型学会了普遍规律；它只说明这六个样本在当前评估方式下被正确分类。

`CrossEntropyLoss` 对目标张量的要求也很具体。对于 `[B,C]` 的 logits，类别索引目标通常应是形状 `[B]`、整数类型 `torch.long`，取值范围为 `0` 到 `C-1`。如果传入 one-hot 浮点标签，某些版本可能支持概率目标，但语义与类别索引不同，不能把两种格式混为一谈。训练前检查形状和 dtype，比在损失变成 `nan` 后猜原因更有效。

### 1.2.4 `train()` 和 `eval()` 改变的是模块行为

`model.train()` 和 `model.eval()` 不会自动开始或停止梯度计算，它们切换的是模块的训练/评估行为。Dropout 在训练时随机丢弃激活，在评估时关闭随机丢弃；BatchNorm 在训练时更新运行统计量，在评估时使用已有统计量。若评估阶段忘记调用 `eval()`，指标可能带有随机性；若只调用 `eval()` 却仍保留计算图，则仍会浪费推理内存。

推理阶段常见的组合是：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 3),
)
features = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])

model.eval()
with torch.no_grad():
    logits = model(features)
    predictions = logits.argmax(dim=-1)

print(logits.shape, predictions.shape)
```
输出形状为：

```text
torch.Size([2, 3]) torch.Size([2])
```

这里有两件独立的事：第一行改变模块模式，第二行关闭 autograd 记录。`torch.inference_mode()` 也是单独的梯度模式，它不会自动调用 `model.eval()`；它比 `no_grad()` 额外关闭 view tracking 和 version counter，但其中创建的 tensor 不能任意转入后续需要 autograd 记录的计算。因此在确认输出只用于推理时再使用，若后续还要把结果接入训练图，应选择 `no_grad()` 或复制到普通 tensor。具体限制以目标版本的 [PyTorch inference_mode 文档](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html) 为准。

### 1.2.5 MLP 与 Transformer 前馈网络的联系

Transformer block 中的前馈网络（Feed-Forward Network，FFN）通常也采用“线性变换—非线性—线性变换”的形态：

```math
\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
```

与这里的 MLP 相比，主要差别在输入包含序列位置，且实际模型通常使用更大的隐藏维度、残差连接、归一化以及诸如 GELU 或门控激活等组件。数学骨架仍然相同：先把表示投影到较高维空间，在高维空间施加非线性，再投影回模型维度。

这个联系把本节接回第一章开头的训练循环。张量形状、logits、损失、自动微分和参数更新都会进入 Transformer；变化的是序列维、模型规模以及内存和数值稳定性要求。第二章会把这些同样的对象放到注意力和 decoder block 里。

一个最小的非线性例子能看出隐藏层增加了什么。考虑输入只取 0 或 1 的四个点，目标是异或：$(0,0)\to0$、$(0,1)\to1$、$(1,0)\to1$、$(1,1)\to0$。单个线性分类边界无法把两种标签分开。两个隐藏单元可以构造 $h_1=\mathrm{ReLU}(x_1-x_2)$ 与 $h_2=\mathrm{ReLU}(x_2-x_1)$，输出 $h_1+h_2$ 在四个点上恰好是 0、1、1、0。这里的权重是为展示表达能力而手工指定的，不是训练结果；实际分类器还会把分数映射为 logits，并用标签与损失来更新参数。该例说明“增加层数”本身不够，真正改变函数族的是中间的非线性。

分类实验还要把训练样本与未参与更新的验证样本分开。若在同一批四个 XOR 点上反复调结构，最后报告这四点全部正确，只能证明它们被当前参数拟合；要判断模型是否学到适用于任务分布的规则，需要固定划分、额外样本或明确的合成数据生成机制。小型教学集可以核对前向、梯度和决策边界，但样本很少时，不应把训练准确率写成泛化能力。

评估时也要把“输出分数”和“最终决策”分开。argmax 只在类别成本相同、阈值已经确定时给出一个默认决策；如果漏掉少数类的代价更高，可能需要按验证集选择阈值或使用代价敏感指标。阈值必须在训练外的数据上确定，并保留原始 logits 或概率，方便在业务成本改变时重新决策，而不必重新训练模型。

以刚才 18 比 2 的样本为例，即使把少数类阈值调低，也不能在训练集上反复试到看起来最好再报告同一集的分数。应先确定验证协议，并查看混淆矩阵里的假阳性与假阴性分别增加多少；如果正类判断触发真实操作，阈值还要结合错误代价和人工复核能力。这个过程把模型输出、决策规则和业务后果分成三层，避免把“概率变了”误说成“分类系统一定更好”。

如果验证样本太少，阈值附近的一条记录就可能让 precision 大幅波动；此时要报告计数或区间，不能只保留一个百分比。课程示例关注实现链路，正式任务还需要为阈值和指标预先定义切片。

部署分类器时，阈值和类别映射也应随模型版本保存；只有模型权重而没有标签顺序，输出索引就无法可靠解释。

进入损失函数前，先确认这段 MLP 的数据流：二维特征先变成隐藏表示，经过非线性，再投影成三个 logits。标签并不是“要把模型改成某个类别”的命令；它告诉损失函数，当前 logits 对应的概率分布离目标有多远。

## 1.3 交叉熵：分数怎样变成可优化的目标

为什么不直接用准确率训练？准确率只记录“当前预测对不对”。只要预测类别没有翻转，它就不变；而梯度下降需要知道当前分数应该往哪个方向、移动多少。交叉熵保留了概率差异：同样预测正确，给正确类别 51% 和 99% 概率，得到的损失不同；预测错了而且非常自信，损失会更大。接下来从最大似然推导这个目标，再看它为什么可以直接接收 logits。

### 1.3.1 从最大似然得到负对数似然

分类器对一个样本产生 $C$ 个 logits，记为 $z=(z_1,\ldots,z_C)$。logit 是尚未归一化的类别分数；softmax 把它们变成总和为 1 的概率：

```math
p_c=\frac{e^{z_c}}{\sum_{k=1}^{C}e^{z_k}}
```

若真实类别是 $y$，模型给真实类别的概率是 $p_y$。最大似然希望这个概率尽可能大；为了把乘法概率转化为可相加、可优化的量，使用负对数似然：

```math
\ell(z,y)=-\log p_y
```

把 softmax 代入并整理，可以得到只依赖 logits 的形式：

```math
\ell(z,y)=\log\left(\sum_{k=1}^{C}e^{z_k}\right)-z_y
```

第一项是所有类别分数的整体尺度，第二项奖励真实类别的分数变大。若真实类别的 logit 比其他类别都高，损失会变小；若某个错误类别的 logit 远高于真实类别，损失会迅速增加。

这也解释了交叉熵与准确率的差别。准确率只看最大类别是否正确，而交叉熵还关心概率分布的尖锐程度：一个预测正确但只给真实类别 0.51 概率的样本，损失仍然不小；一个预测正确且给真实类别 0.99 概率的样本，损失更小。训练时使用交叉熵，评估时同时看准确率，才能看到这两种信息。

### 1.3.2 数值稳定性来自先平移 logits

直接计算 `exp(z)` 会在 logits 较大时溢出，在 logits 较小时下溢。这个问题不是抽象的边角情况：深度网络的 logit 尺度可能随着初始化、归一化、精度格式和训练阶段变化。利用恒等式，可以先令 $m=\max_k z_k$：

```math
\log\sum_k e^{z_k}
=m+\log\sum_k e^{z_k-m}
```

平移后至少有一个指数是 $e^0=1$，其余指数不大于 1。这个公式是 `torch.logsumexp` 这类实现的核心思想。对真实类别 $y$，稳定的交叉熵可以写成：

```math
\ell(z,y)=m+\log\sum_k e^{z_k-m}-z_y
```

下面的纯 Python 版本不依赖 PyTorch，显式展示稳定计算：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import math

logits = [2.0, 1.0, -1.0]
target = 0
shift = max(logits)
log_z = shift + math.log(sum(math.exp(value - shift) for value in logits))
loss = log_z - logits[target]
probabilities = [math.exp(value - log_z) for value in logits]
gradient = [probability - float(index == target)
            for index, probability in enumerate(probabilities)]

print(round(loss, 4))
print([round(value, 4) for value in probabilities])
print([round(value, 4) for value in gradient])
```
输出为：

```text
0.349
[0.7054, 0.2595, 0.0351]
[-0.2946, 0.2595, 0.0351]
```

不要把“先做 softmax 再取 log”当作等价的工程写法。数学上它们可以化简，数值上却不是同一件事。`CrossEntropyLoss` 将 `log_softmax` 与负对数似然组合起来，避免中间概率先被舍入到 0。

### 1.3.3 logits 的梯度为什么是“概率减去 one-hot”

对 logits 的第 $c$ 个分量求导，可得：

```math
\frac{\partial \ell}{\partial z_c}=p_c-\mathbf{1}[c=y]
```

其中 $\mathbf{1}[c=y]$ 在 $c=y$ 时为 1，否则为 0。因此真实类别的梯度是 $p_y-1$，通常为负值，梯度下降会把对应 logit 往上推；错误类别的梯度是 $p_c$，为正值，梯度下降会把对应 logit 往下推。

这个推导还揭示了一个尺度性质：所有类别的 logits 同时加上同一个常数，softmax 概率不变，损失也不变。模型真正需要学习的是类别之间的相对分数，而不是一个绝对基准。数值稳定的 log-sum-exp 正是在利用这一平移不变性。

例如 logits 为 $[2,1,0]$，真实类别是第 0 类。平移最大值 2 后得到 $[0,-1,-2]$，稳定的 loss 是 $\ln(1+e^{-1}+e^{-2})\approx0.4076$。若原始 logits 同时加 1000，概率和 loss 理论上不变；直接计算 $e^{1002}$ 却可能溢出，先减最大值仍只需计算 $1,e^{-1},e^{-2}$。这是数值稳定技巧与数学目标一致、但实际运算路径不同的例子。

batch 内 `reduction="mean"` 的平均对象也必须明确。分类任务通常对样本平均；自回归语言模型的目标则可能是对所有未屏蔽 token 平均。若第一条样本有 10 个有效 token、第二条只有 2 个，把各自 token 平均损失再平均会让两条样本权重相同；把 12 个 token 的损失总和除以 12，则长样本贡献更大。两种目标都可能合理，但在变长数据、padding 和梯度累积时，分子与分母必须按同一约定计算。

如果一个类别的概率已经非常接近 1，它的梯度接近 0；这不是自动微分失效，而是当前样本对该参数方向的局部信号确实很小。多个样本、多个层和参数共享会共同决定最终梯度，不能只凭单个 logit 的变化判断整个模型是否还在学习。

这也解释了为什么训练日志应同时保留平均 loss、样本或 token 分母，以及关键类别的错误数。两个 batch 的 loss 同为 0.4，可能一个包含 8 个样本、另一个包含 800 个；它们的估计稳定性并不相同。把均值、总和和计数一起保存，后续才能重算加权指标，判断变化来自模型、采样比例还是归约口径。

### 1.3.4 `reduction`、类别权重与无效标签

对一个批次的逐样本损失 $\ell_i$，常见的聚合方式有三种：

```math
L_{\text{none}}=[\ell_1,\ldots,\ell_B],\qquad
L_{\text{sum}}=\sum_i\ell_i,\qquad
L_{\text{mean}}=\frac{1}{B}\sum_i\ell_i
```

`reduction="mean"` 会让梯度规模与批次样本数大致保持可比；`sum` 则会随着有效样本数量增加而增大。在使用梯度累积、动态 padding 或分布式训练时，不能默认“每张卡的 mean 再平均”一定等于“全局有效 token 的 mean”。如果不同设备上的有效 token 数量不同，应明确分子和分母，按有效元素总数归一化。

当 `CrossEntropyLoss` 使用类别权重且目标为类别索引时，`reduction="mean"` 的分母是本批次未忽略目标对应权重之和，而不是简单的样本数；没有类别权重时，每个有效目标的权重为 1，才退化为按有效样本数平均。若目标使用概率分布格式，接口的 reduction 语义不同，应以对应 PyTorch 版本文档为准。

类别不均衡时可以引入类别权重 $\alpha_c$：

```math
\ell_i=-\alpha_{y_i}\log p_{i,y_i}
```

权重改变了不同类别对梯度的贡献，并不会神奇地创造新数据。权重的尺度还会影响整体梯度大小，所以调整权重后应重新观察学习率、损失和验证集指标。

序列任务常有 padding。若某些位置不是有效目标，可以使用 `ignore_index`，使这些位置不贡献损失。等价地，也可以先得到逐位置损失，再用 mask 做加权平均：

```math
L=\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}
{\max(1,\sum_{i,t}m_{i,t})},\qquad m_{i,t}\in\{0,1\}
```

分母中的 `max(1, ·)` 是工程保护，避免一个批次全是 padding 时出现除零；它并不意味着这个批次包含有效训练信号。全无效批次更合理的处理方式可能是跳过更新，并在数据管道中追查原因。

### 1.3.5 语言模型中的 `[B,T,V]`

自回归语言模型通常为每个位置预测下一个 token。设输入 token 形状为 $[B,T]$，词表大小为 $V$，模型输出 logits 形状为 $[B,T,V]$。标签需要向左移动一位：位置 $t$ 的输出对应目标 $t+1$。抽象写法是：

```math
L=-\frac{1}{N_{\text{valid}}}
\sum_{b=1}^{B}\sum_{t=1}^{T-1}
m_{b,t}\log p(x_{b,t+1}\mid x_{b,\le t})
```

其中 $m_{b,t}$ 表示该目标位置是否有效，且

```math
N_{\text{valid}}=\sum_{b=1}^{B}\sum_{t=1}^{T-1}m_{b,t}
```

是有效目标数量。这个平均式假定 $N_{\text{valid}}>0$；如果一个 batch 全是 padding，就不应把“除以 1 后得到的数”误当作有效训练信号，而应跳过更新并追查数据管道。实现时常把 logits 重排为 $[B\times T,V]$，标签重排为 $[B\times T]$：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

batch_size, sequence_length, vocabulary_size = 2, 4, 7
logits = torch.randn(batch_size, sequence_length, vocabulary_size)
input_ids = torch.tensor([
    [1, 2, 3, 4],
    [2, 5, 6, 0],
])

shifted_logits = logits[:, :-1, :].contiguous()
shifted_labels = input_ids[:, 1:].contiguous()
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
loss = loss_fn(
    shifted_logits.view(-1, vocabulary_size),
    shifted_labels.view(-1),
)
print(shifted_logits.shape, shifted_labels.shape)
print(loss.ndim)
```
输出形状为：

```text
torch.Size([2, 3, 7]) torch.Size([2, 3])
0
```

`view` 要求底层存储连续；在切片、转置或排列维度后，使用 `.contiguous()` 或 `reshape` 可以避免因为内存布局造成的问题。这里 `ignore_index=0` 只是一个教学约定，真实词表中 0 是否是 padding、是否可作为正常 token，必须以 tokenizer 和数据管道的定义为准。

### 1.3.6 从平均 token loss 到 perplexity

语言模型常把平均负对数似然指数化为困惑度（perplexity）：

```math
\mathrm{PPL}=\exp(L_{\text{token}})
```

只有当 $L_{\text{token}}$ 确实按有效 token 平均，并且对数使用自然对数时，这个换算才有上述形式；还要保证损失是有限值。若有效 token 数为零，困惑度没有定义，不能用一个人为的保护分母制造可比较的 PPL。如果损失包含 padding、不同样本权重或跨批次归一化方式不同，直接比较 PPL 会产生误导。PPL 也不是所有生成质量的充分指标；它偏向评估概率建模目标，无法单独代表事实性、指令遵循或长文本结构能力。

交叉熵现在能告诉我们每个 logit 应该怎样变化。但 logits 是许多参数经过层层运算得到的；要知道权重矩阵该怎样改，还必须把这份信号沿前向计算的路径传回去。这就是反向传播要做的事。

## 1.4 反向传播：计算图、局部导数与梯度的路径

### 1.4.1 反向传播不是“把公式倒着算一遍”

自动微分首先执行前向计算，并记录足以计算局部导数的信息。反向阶段从损失开始，沿着图的反方向传播一个上游梯度；每经过一个运算节点，就把上游梯度乘以该节点对输入的局部导数。

例如：

```math
u=xy,\qquad v=\sin x,\qquad L=u+v
```

从 $L$ 到 $x$ 有两条路径：

```math
\frac{\partial L}{\partial x}
=\frac{\partial u}{\partial x}\frac{\partial L}{\partial u}
+\frac{\partial v}{\partial x}\frac{\partial L}{\partial v}
=y+\cos x
```

“多条路径的梯度相加”是理解残差连接、参数共享和注意力结构的关键。一个参数被多个分支使用时，它收到的是所有使用位置贡献的总和，而不是最后一条路径的结果。

更简单的例子是：

```math
L=x^2+3x,\qquad
\frac{\partial L}{\partial x}=2x+3
```

如果把 $x$ 同时送入两个分支后再相加，反向传播必须把两个分支的局部导数都保留下来。计算图的“动态”含义是：每次前向运行都可以根据 Python 控制流构造不同的图，而不是先声明一张固定的静态图。

可以用一个只有两步的计算图手算局部导数。令 $u=wx+b$、$v=\mathrm{ReLU}(u)$、$L=(v-y)^2$，并取 $w=2,x=1,b=0,y=1$。前向得到 $u=v=2,L=1$；从末端开始，$\partial L/\partial v=2(v-y)=2$，由于 $u>0$，ReLU 的局部导数是 1，所以 $\partial L/\partial u=2$；再乘上 $\partial u/\partial w=x=1$，得到 $\partial L/\partial w=2$，同理 $\partial L/\partial b=2$。若学习率为 0.1，更新后 $w=1.8,b=-0.2$，新 $u=1.6$、损失为 0.36。这里每一步只是“上游梯度乘局部导数”；若 $u<0$，ReLU 路径会把这条梯度截为 0，说明梯度能否到达参数取决于前向经过的分支。

再看一个共享参数例子。令 $z=wx+w$，取 $w=1,x=2$，所以 $z=3$；若 $L=z^2$，则上游梯度 $\partial L/\partial z=6$，而 $z$ 对 $w$ 的总导数是 $x+1=3$，最终 $\partial L/\partial w=18$。如果把两处使用误当成两个独立参数，就会漏掉一个分支的贡献。残差路径、共享 embedding 和输出头都可能复用同一参数，PyTorch 会沿每条路径计算并将贡献累加。排查“梯度为什么比预期大”时，应先检查参数是否共享、loss 是否求和、同一张量是否经过多次调用，再怀疑自动微分本身。

梯度检查也有适用条件。中心有限差分用 $f(w+\epsilon)-f(w-\epsilon)$ 估计当前斜率；$\epsilon$ 太小会被浮点舍入主导，太大则测到较大区间的平均变化。ReLU 恰在零点不可导，dropout 每次前向会改变路径，低精度会放大数值噪声。因而检查自定义算子时，应固定随机状态、选取远离折点的简单输入、使用较高精度并报告容差；通过一个标量局部检查只支持该算子的局部梯度正确，不能证明整个模型训练有效。

反向传播之后，框架通常会释放本轮图中为求导保存的中间值；如果需要对同一次前向结果再次求导，必须显式保留图，但这会占用额外内存。多数训练循环应重新做一次前向，而不是长期保留旧图。把训练循环中的图保存在列表里、只为了稍后打印 loss，可能让计算图和激活无法释放；若只需记录数字，应保存标量副本，而不是仍连接计算图的张量。

### 1.4.2 标量损失与非标量输出

最常见的训练写法是让损失变成标量，再调用：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

loss = torch.tensor(2.0, requires_grad=True)
loss.backward()
print(loss.grad.item())
```
输出为：

```text
1.0
```

如果输出不是标量，例如逐样本损失形状为 `[B]`，它有一个向量到标量的 Jacobian 问题。此时不能无参数地调用 `backward()`，需要明确传入上游向量，或先求和/求平均：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

parameter = torch.tensor([2.0, 3.0], requires_grad=True)
values = parameter ** 2
values.backward(torch.ones_like(values))
print(parameter.grad.tolist())
```
输出为：

```text
[4.0, 6.0]
```

这里传入的全 1 向量表示对 $v_1+v_2$ 求导。如果传入 `[1.0, 0.0]`，只会选择第一项对参数的贡献。这个接口让用户能够表达向量输出的加权组合，但也意味着“调用 `backward()`”本身并不说明你对哪个标量目标求了梯度。

### 1.4.3 叶子张量、`.grad` 与中间结果

由用户直接创建、且 `requires_grad=True` 的参数通常是叶子张量，优化器关注的就是这些张量。中间运算结果也可能需要梯度，但默认不会把它保存在 `.grad` 属性中；如果确实需要观察中间激活，可以在反向前调用 `retain_grad()`：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

weight = torch.tensor(2.0, requires_grad=True)
hidden = weight * 3.0
hidden.retain_grad()
loss = hidden ** 2
loss.backward()

print(weight.grad.item(), hidden.grad.item())
```
输出为：

```text
36.0 12.0
```

因为 $hidden=3w=6$，$L=hidden^2$，所以 $\partial L/\partial hidden=12$，再乘以 $\partial hidden/\partial w=3$，得到 $36$。如果只打印 `hidden.grad` 却没有调用 `retain_grad()`，得到 `None` 并不表示反向传播没有经过该节点。

### 1.4.4 `detach()`、`no_grad()` 和参数更新

这三个概念经常被混在一起，但作用不同。

`detach()` 从一个已有张量创建一个不再向前连接的视图或张量关系：后续对 detached 结果的运算不会把梯度传回原图。它适合表达“把这个表示当作常量使用”的语义，例如目标网络、缓存特征或某些强化学习计算。

`torch.no_grad()` 是一个上下文管理器，表示在这段代码执行期间不记录大多数 autograd 运算。优化器更新参数时通常使用它，因为更新动作本身不应成为下一轮计算图的一部分：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

parameter = torch.tensor(2.0, requires_grad=True)
loss = parameter ** 2
loss.backward()
learning_rate = 0.1

with torch.no_grad():
    parameter -= learning_rate * parameter.grad

print(parameter.item())
```
输出为：

```text
1.6
```

`detach()` 关注一个张量与已有图的连接；`no_grad()` 关注一段运算是否建立图。下面的例子说明 detach 后的损失仍可计算，却不会给原参数提供梯度：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

parameter = torch.tensor(2.0, requires_grad=True)
detached = parameter.detach()
loss = (detached * 5.0) ** 2
print(loss.requires_grad)
```
输出为：

```text
False
```

如果需要在训练中暂时冻结某个模块，可以将其参数的 `requires_grad` 设为 `False`，也可以在前向路径中使用 `no_grad()`；两种写法对优化器参数组、激活保存和后续重新解冻的影响并不完全相同，应明确设计，而不是到处插入 `detach()` 让错误“消失”。

### 1.4.5 原地操作为什么可能破坏反向传播

反向传播有时需要前向阶段保存的中间值。例如 $y=x^2$ 的梯度需要知道前向时的 $x$。如果在反向前原地修改了这个值，系统无法保证使用的是正确版本，通常会抛出版本计数错误。

参数更新也是原地修改，但它发生在 `no_grad()` 中，并且通常在本轮反向完成之后，所以不会把更新操作接入当前计算图。对需要梯度的叶子张量直接做原地运算，可能立即报错；对中间激活做原地运算，可能破坏之后某个节点需要的保存值。

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

value = torch.tensor(2.0, requires_grad=True)
result = value * value

with torch.no_grad():
    value.add_(1.0)

print(result.item())
try:
    result.backward()
except RuntimeError:
    print("RuntimeError: in-place modification detected")
```
输出为：

```text
4.0
RuntimeError: in-place modification detected
```

这里 `result` 的前向值仍是 4；在本地 PyTorch 2.12.0 中，随后反向会检测到 `value` 的版本计数已经变化。具体异常文本会随版本变化，所以示例只固定异常类型和教学提示。工程代码不应依赖这种错误处理方式；更稳妥的是让参数更新只出现在优化步骤中，避免在图仍被使用时修改参与计算的张量。

### 1.4.6 用有限差分检查一个梯度

当自定义算子、手写损失或复杂张量变换出现可疑结果时，可以用有限差分做小规模检查。对标量函数 $f(\theta)$，中心差分近似为：

```math
f'(\theta)\approx
\frac{f(\theta+\varepsilon)-f(\theta-\varepsilon)}{2\varepsilon}
```

$\varepsilon$ 太大时截断误差明显，太小时浮点舍入误差明显；它不是一个对所有 dtype 和尺度都固定的常数。下面只检查一个标量二次函数：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
def function(value):
    return value * value + 3.0 * value

point = 1.5
epsilon = 1e-5
finite_difference = (
    function(point + epsilon) - function(point - epsilon)
) / (2.0 * epsilon)
analytic = 2.0 * point + 3.0

print(round(finite_difference, 4), round(analytic, 4))
```
输出通常为：

```text
6.0 6.0
```

有限差分只能在小问题上提供局部证据，不能替代端到端训练验证。对于随机算子、非连续函数、混合精度或包含 dropout 的模型，需要先固定随机性、选择合适的容差，并理解数值梯度与解析梯度比较的条件。

梯度回答“当前往哪个方向移动，局部损失会下降”。它没有规定实际要走多远，也没有利用过去几步的梯度。优化器把这些选择变成参数更新规则；SGD、Momentum、Adam 和 AdamW 的区别，主要体现在怎样处理当前梯度、历史信息和参数衰减。

## 1.5 优化器：梯度告诉方向，状态决定怎样走

### 1.5.1 SGD 是一个更新规则，不是训练的全部

最基本的随机梯度下降（SGD）使用当前小批次估计的梯度：

```math
\theta_{t+1}=\theta_t-\eta_t g_t,
\qquad g_t=\nabla_\theta L_{\mathcal B_t}(\theta_t)
```

这里 $\theta$ 表示所有可训练参数，$\mathcal B_t$ 是第 $t$ 次使用的小批次，$g_t$ 是该小批次损失对参数的梯度。小批次梯度通常不是全数据梯度的精确值，而是一个带噪声的估计。噪声有时帮助模型离开狭窄区域，但也会让损失曲线抖动。

批次大小改变了梯度估计的方差、每次更新的计算量以及单位样本的吞吐。不能只说“批次越大越好”：在固定总 token 数下，大批次意味着更新次数更少；在固定更新次数下，大批次又意味着看过更多样本。比较实验时需要说明究竟固定了样本数、token 数、更新次数，还是墙钟时间。

PyTorch 的 `SGD` 还支持动量、dampening、Nesterov 和 weight decay 等选项。只写出类名并不能确定实际算法；完整实验记录至少要包含学习率、动量、weight decay、nesterov、是否最大化目标以及参数组设置。

### 1.5.2 Momentum 为什么能平滑小批次噪声

一种常见的动量形式是：

```math
v_t=\mu v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta_t v_t
```

$\mu$ 是动量系数，$v_t$ 是梯度的指数加权累积。沿着长期一致的方向，历史梯度会叠加；方向频繁变化的噪声则部分抵消，因此参数轨迹通常比裸 SGD 更平滑。

不同框架对动量缓存的定义、初始时刻和 dampening 处理可能略有差别。上面的式子是理解机制的简化形式，不应拿来逐项复刻某个版本的全部边界行为。研究复现实验时，应查看目标版本源码或官方文档，特别是第一次更新、Nesterov 和 weight decay 的组合。

### 1.5.3 Adam 的一阶矩、二阶矩和偏差修正

Adam 同时维护梯度的一阶矩和平方梯度的二阶矩：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

因为两个缓存初始为零，训练早期的估计会偏向零，所以要做偏差修正：

```math
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
```

参数更新为：

```math
\theta_{t+1}=\theta_t-
\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

其中 $\epsilon$ 防止分母为零，也会影响极小二阶矩时的有效步长。`betas`、`eps` 和学习率共同决定更新尺度；只比较“Adam 比 SGD 收敛快”而不报告这些设置，结论是不完整的。

看一次单参数更新，就能分清偏差修正的作用。令初始参数 $\theta_0=1$，当前梯度 $g_1=2$，并取 $\beta_1=0.9,\beta_2=0.999$。第一次更新时，$m_1=0.2$、$v_1=0.004$；偏差修正后 $\hat m_1=2$、$\hat v_1=4$。若学习率 $\eta=0.1$ 且 $\epsilon$ 很小，Adam 的更新约为 $0.1\times2/2=0.1$，于是参数从 1 降到约 0.9。相同梯度下，学习率为 0.1 的普通 SGD 会把参数更新到 0.8。这个单步算例说明 Adam 会按梯度历史的尺度重新调整步幅；它不证明 Adam 在任意任务上都更好。

Adam 的一阶矩和二阶矩都要为每个参数保存一个同形状缓存。因此优化器状态可能接近参数本体的两倍；使用 FP32 参数、梯度和状态时，显存估算不能只看模型参数数量。分布式训练、参数分片和 8-bit optimizer state 会改变这个账本，必须按实际实现核算。

### 1.5.4 Adam 与 AdamW：L2 正则和 weight decay 不总是同一件事

把 L2 正则加入目标函数，得到：

```math
L_{\text{reg}}(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2
```

对应梯度是：

```math
\nabla L_{\text{reg}}(\theta)=g_t+\lambda\theta_t
```

对于普通 SGD，直接把 $\lambda\theta$ 加入梯度与乘法形式的 weight decay 有紧密关系。但在 Adam 中，这个正则项会进入自适应的一阶、二阶统计，被不同参数的梯度尺度重新缩放，因而不再等同于“统一收缩参数”。

AdamW 把衰减从自适应梯度更新中解耦。简化写法是：

```math
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

这里的 $\lambda$ 是参数收缩强度，$\eta_t$ 是当前学习率。两者相乘意味着学习率调度也会影响每一步实际衰减量。实践中常把 bias、LayerNorm/RMSNorm 的 scale 和某些特殊参数排除在 weight decay 之外；这种参数分组是建模选择，不是 AdamW 自动推断出来的。

下面的代码展示参数分组的结构。它不声称某个任务的最优配置，只说明如何让不同参数使用不同衰减策略：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, values):
        return self.norm(self.projection(values))

model = TinyBlock()
decay_parameters = []
no_decay_parameters = []

for name, parameter in model.named_parameters():
    if parameter.ndim == 1 or name.endswith("bias"):
        no_decay_parameters.append(parameter)
    else:
        decay_parameters.append(parameter)

optimizer = torch.optim.AdamW([
    {"params": decay_parameters, "weight_decay": 0.01},
    {"params": no_decay_parameters, "weight_decay": 0.0},
], lr=1e-3)

print(len(decay_parameters), len(no_decay_parameters))
```
对于这个模型，输出为：

```text
1 3
```

权重矩阵属于衰减组；线性层 bias、LayerNorm 的 weight 和 bias 都是一维参数，进入不衰减组。真实模型可能有卷积参数、嵌入参数、门控参数或共享权重，不能机械地把“维度为 1”当作永远正确的规则。

### 1.5.5 一个可靠的训练循环需要明确顺序

典型的单批次训练按以下顺序执行。下面的代码包含了运行它所需的最小对象；在较大的训练程序中，`inputs`、`labels`、`model`、`loss_fn` 和 `optimizer` 通常由数据管道与训练初始化代码提供：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

model = nn.Linear(2, 2)
inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
labels = torch.tensor([0, 1])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)
loss = loss_fn(logits, labels)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```
每一行都有独立语义：`train()` 设置模块模式；清零防止梯度跨批次意外累积；前向建立计算图；损失提供标量目标；反向填充梯度；梯度裁剪限制整体范数；`step()` 使用当前梯度和优化器状态更新参数。裁剪并非所有任务都需要，而且它改变了实际更新方向，使用时应记录阈值和裁剪比例。

如果把 `optimizer.zero_grad()` 放在 `backward()` 之后，会把刚算出的梯度清掉；如果把 `optimizer.step()` 放在 `backward()` 之前，更新使用的可能是上一个批次的梯度或空梯度。将这些操作包装成函数可以减少重复，但不应因此失去对顺序的理解。

### 1.5.6 梯度累积与有效批次大小

显存不足时，可以让 $K$ 个 micro-batch 共用一次参数更新。若每个 micro-batch 的损失已经是该批次的平均值，且每个 micro-batch 的有效样本数相同，通常应将每次损失除以 $K$：

```math
L_{\text{acc}}=\frac{1}{K}\sum_{k=1}^{K}L_k
```

代码中不会真的先构造一个跨 micro-batch 的 $L_{\text{acc}}$，而是对每个 $L_k/K$ 分别反向，最后调用一次 `optimizer.step()`；两种写法在计算图仍可保留且归一化条件相同时等价。否则梯度大约会放大 $K$ 倍，等价的学习率也随之改变。若各 micro-batch 有效 token 数不同，简单除以 $K$ 可能仍不等于全局有效 token 平均；应累计未归一化损失和有效数量，再按总有效数量归一化。

下面的最小示例保留最后一个不完整组。它假定每个 micro-batch 的有效样本数相同；如果序列长度或有效 token 数不同，应把 `loss_fn` 改为 `reduction="sum"`，另外累计有效数量，再按总有效数量归一化：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Linear(2, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
features = torch.tensor([
    [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
    [-1.0, 0.0], [0.0, -1.0],
])
labels = torch.tensor([0, 1, 0, 1, 0])
loader = DataLoader(TensorDataset(features, labels), batch_size=1)

accumulation_steps = 4
total_micro_batches = len(loader)
optimizer.zero_grad(set_to_none=True)
optimizer_steps = 0

for micro_step, (inputs, labels) in enumerate(loader):
    group_start = (micro_step // accumulation_steps) * accumulation_steps
    group_size = min(
        accumulation_steps,
        total_micro_batches - group_start,
    )
    logits = model(inputs)
    loss = loss_fn(logits, labels) / group_size
    loss.backward()

    is_group_end = (
        (micro_step + 1) % accumulation_steps == 0
        or micro_step + 1 == total_micro_batches
    )
    if is_group_end:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

print(optimizer_steps)
```
输出为 `2`：5 个 micro-batch 被分成大小为 4 和 1 的两组，因此产生两次参数更新。代码按当前组大小分别除以 4 和 1；若把最后一个损失仍除以 4，第二次更新的梯度就会被缩小到应有值的四分之一。这个边界在 `drop_last=True` 时可以通过丢弃不完整组来回避，但是否丢弃数据必须成为明确的训练决策。

有效批次大小常写成：

```math
B_{\text{effective}}
=B_{\text{micro}}\times K\times N_{\text{data-parallel}}
```

这是在每个设备上的 micro-batch 大小、累积步数和数据并行设备数定义清楚，且每次更新都覆盖相同数量有效样本时的简化表达。序列任务中更准确的单位往往是有效 token 数，而不是样本数。

### 1.5.7 混合精度改变的是数值路径，不是数学目标

混合精度通常让部分矩阵运算使用 FP16 或 BF16，同时保留某些参数、归约和优化器状态的更高精度。它减少内存和带宽压力，但也缩小了可表示的数值范围或有效精度。FP16 训练常配合 loss scaling：先把损失乘以一个较大的尺度 $s$，反向得到 $sg$，更新前再除以 $s$，从而避免小梯度下溢；如果检测到无穷或 NaN，则跳过本次更新并调整尺度。

可以把它看成一段临时的数值放大：假设某条梯度约为 $10^{-10}$，直接放进 FP16 的计算路径时可能被舍入成 0；若先把损失乘以 $65{,}536$，反向得到的梯度约为 $6.55\times10^{-6}$，可先在较大的数值范围内传递，再在更新前除回原尺度。实际的缩放值会动态调整，遇到溢出时也可能跳过这一步更新。这个小算例只解释缩放的目的，不代表每个模型都需要同一个尺度。

Autocast 是这条数值路径的另一半：它按算子选择精度，通常包住前向计算和损失计算；反向放在 autocast 区域外。使用缩放器时，要先反向，再取消缩放、裁剪梯度，最后让缩放器检查梯度并决定是否执行优化器更新。PyTorch 2.15 推荐从 `torch.amp` 使用 autocast 和 GradScaler；旧的 `torch.cuda.amp.*` 与 `torch.cpu.amp.*` 入口已标为弃用。具体顺序和设备边界见 [PyTorch 2.15 AMP 文档](https://docs.pytorch.org/docs/2.15/amp.html)。

BF16 的指数范围接近 FP32，通常更不容易溢出，但尾数精度较低；这不是说 BF16 在所有任务上都无需数值检查。CPU BF16 autocast 的常见路径不需要 GradScaler；FP16 是否需要缩放以及哪些算子用何种精度，则应按目标设备、PyTorch 版本和算子支持确定。混合精度能否提速也必须在目标硬件上测量，不能从 dtype 名字直接推断。

训练日志至少应同时记录：损失是否有限、梯度范数、学习率、跳过更新次数以及有效 token 数。只记录一个下降的 loss，无法区分“模型正常学习”和“某些 batch 被悄悄跳过”。

即使优化器和梯度都固定，同一个更新方向也会因学习率不同而走出不同距离。训练开始、稳定推进和接近预算终点时，合适的步长可能不同，因此还需要规定学习率如何随训练进展变化。

## 1.6 学习率调度：让更新尺度与训练阶段相匹配

### 1.6.1 学习率是时间的函数

把固定学习率写成 $\eta$ 只是最简单的情况。更一般地，学习率是优化器更新步 $s$ 的函数：

```math
\theta_{s+1}=\mathrm{Update}
(\theta_s,g_s,\eta_s,\mathrm{state}_s)
```

调度器改变 $\eta_s$，并不直接改变损失、数据顺序或模型结构。训练前期通常需要较小步幅以建立稳定状态，随后可以提高到目标学习率；训练后期则常降低步幅，让参数在已有解附近细化。是否需要 warmup、何时衰减和最终降到多低，都依赖优化器、模型规模、数据量与训练预算。

### 1.6.2 线性 warmup 的含义

设目标学习率为 $\eta_{\max}$，warmup 更新步数为 $S_w>0$，用 1-based 的更新步 $s$ 表示：

```math
\eta_s=\eta_{\max}\min\left(1,\frac{s}{S_w}\right)
```

当 $1\le s\le S_w$ 时，学习率线性增加；之后保持 $\eta_{\max}$，直到后续调度改变它。若初始学习率设置为 0，第一步的更新也可能为 0；不同调度器对 `last_epoch` 和首次调用时刻的定义会影响具体序列，因此应把实际打印的学习率作为事实依据。

warmup 并非“越长越安全”。过长会减少高学习率阶段的训练预算，过短又可能无法缓冲初始化、较大批次或混合精度带来的早期不稳定。比较 warmup 实验时，应固定总更新步数，并报告 warmup 占比，而不只报告最终 loss。

例如目标学习率为 0.2、warmup 为 4 次参数更新，按 $\eta_s=0.2s/4$ 得到前四次更新使用的步长是 0.05、0.10、0.15、0.20。若每两份 micro-batch 才累积出一次参数更新，调度器就应通常在两份数据处理完、优化器执行一次更新后前进一次；若误在每份 micro-batch 后前进，四个 warmup 点会在两次参数更新内消耗完，训练预算轴被缩短一半。这个差异不一定导致程序报错，但它改变了实际优化过程，因此日志里要同时记录调度器和参数更新的计数单位。

### 1.6.3 常见衰减函数

Step decay 在每隔 $S$ 步把学习率乘以 $\gamma$：

```math
\eta_s=\eta_0\gamma^{\lfloor s/S\rfloor},
\qquad 0<\gamma<1
```

指数衰减连续地变化：

```math
\eta_s=\eta_0\gamma^s
```

余弦退火在一个周期内平滑下降到 $\eta_{\min}$：

```math
\eta_s=\eta_{\min}
+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi s}{S}\right),
\quad 0\le s\le S
```

其中 $S$ 是余弦周期的更新步数。若在 $S$ 之后继续使用同一公式，必须说明是否重启、截断或进入另一个周期；不能只写“用了 cosine”就认为实验设置完整。

PyTorch 提供 `StepLR`、`ExponentialLR`、`CosineAnnealingLR`、`LambdaLR` 等调度器。它们记录的是调度状态和计数器，不会替代优化器状态。恢复训练时只加载模型参数而不加载 optimizer/scheduler state，可能使动量、二阶矩和学习率序列突然回到错误状态。

### 1.6.4 调度器的步长必须与参数更新对齐

如果一个 batch 被拆成 $K$ 个 micro-batch，但只在最后一个 micro-batch 调用 `optimizer.step()`，那么学习率调度器通常也应每次参数更新调用一次。下面是一个包含最小上下文的示意：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

model = nn.Linear(2, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=2,
    gamma=0.5,
)
loader = [
    (torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
    (torch.tensor([[0.0, 1.0]]), torch.tensor([1])),
    (torch.tensor([[1.0, 1.0]]), torch.tensor([0])),
]
accumulation_steps = 1

def compute_loss(batch):
    inputs, labels = batch
    return loss_fn(model(inputs), labels)

for micro_step, batch in enumerate(loader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (micro_step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
```
把 scheduler 每个 micro-batch 调用一次，会让学习率在参数还未更新时提前走完；这可能是有意按 token 调度，也可能只是计数单位混淆。两种方案都可以设计，但必须明确横轴究竟是 micro-batch、optimizer update、样本、token 还是 epoch。

对大语言模型，按 token 数调度有时比按 epoch 更自然，因为不同 batch 的序列长度和有效 token 数可能差异很大。此时需要维护有效 token 计数，并把学习率函数定义在这个计数轴上，而不是把“一个 dataloader 迭代”默认当成相同训练量。

### 1.6.5 `LambdaLR` 与分段策略

当内置调度器不能表达目标曲线时，可以定义一个相对初始学习率的倍率函数：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch

parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.SGD([parameter], lr=0.1)

def multiplier(step):
    if step < 2:
        return (step + 1) / 2.0
    return 0.5 ** ((step - 2) // 3)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
rates = []
for _ in range(8):
    optimizer.step()
    scheduler.step()
    rates.append(round(optimizer.param_groups[0]["lr"], 4))

print(rates)
```
在本地 PyTorch 2.12.0、先调用 `optimizer.step()` 再调用 `scheduler.step()` 的顺序下，学习率序列为：

```text
[0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025]
```

这个例子特意展示一个容易混淆的边界：调度器在创建时会处理初始状态，第一次 `scheduler.step()` 后的倍率不一定对应直觉中的“第零步”。不同 PyTorch 版本和调用顺序可能改变观察到的第一项；若实验需要精确复现，应打印每次 `optimizer.step()` 前后的学习率，并将 scheduler 的 `state_dict()` 一并保存。

在通常的训练循环中，官方建议先调用 `optimizer.step()`，再调用 `scheduler.step()`，避免跳过调度序列的第一项。`ReduceLROnPlateau` 是例外类型：它根据验证指标决定是否下降，通常在验证结束并传入指标之后调用，而不是按每个优化更新调用。

### 1.6.6 checkpoint 必须包含训练状态

可恢复训练至少要考虑以下状态：模型参数、优化器状态、调度器状态、当前更新步、随机数状态、数据采样位置以及混合精度缩放器状态。下面先构造一个最小的模型、优化器和调度器，再展示保存结构；真实项目还应按使用的混合精度和数据加载方式补充相应状态：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn
from pathlib import Path

model = nn.Linear(2, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.1,
)
global_step = 0

checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
}
Path(".tmp").mkdir(exist_ok=True)
torch.save(checkpoint, ".tmp/pytorch-chapter-checkpoint.pt")
```
示例把临时 checkpoint 写入仓库中已忽略的 `.tmp/`，避免运行教学代码在项目根目录留下生成物；真实训练应按项目的 checkpoint 目录和保留策略配置路径。

恢复时先构造相同的模型、优化器和调度器，再分别加载对应状态。若先加载 scheduler 状态却改变了 optimizer 参数组数量或顺序，可能出现难以察觉的学习率错配；参数组结构应在恢复前保持兼容。

随机状态和数据位置尤其重要。只恢复模型权重，可以做“从这个权重继续微调”，但它不等于“从中断处无缝续训”。两者的目标不同，报告结果时应该用不同措辞。

### 1.6.7 用日志判断调度是否真的生效

学习率调度问题常常不在公式，而在调用位置、参数组或恢复逻辑。下面的示例先完成一次真实更新，再读取当前参数组学习率和 Adam 的状态步数：

> 代码状态：已验证（2026-10-19，Python 3.12.9，PyTorch 2.12.0+cpu）

```python
import torch
from torch import nn

model = nn.Linear(2, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
inputs = torch.tensor([[1.0, 2.0]])
targets = torch.tensor([[1.0]])
loss = ((model(inputs) - targets) ** 2).mean()
loss.backward()
optimizer.step()

global_step = 1
current_rates = [group["lr"] for group in optimizer.param_groups]
optimizer_state_step = optimizer.state[model.weight]["step"].item()
print({
    "step": global_step,
    "loss": float(loss.detach()),
    "lr": current_rates,
    "optimizer_step": optimizer_state_step,
})
```
观察时可以问几个具体问题：学习率是否按预期单调变化？多个参数组是否有不同曲线？恢复 checkpoint 后第一步是否突然跳变？loss 下降变慢时，梯度范数是变小了还是出现了 NaN？这些问题把“训练不稳定”拆成了可测量的变量。

如果损失长期不动，可能是学习率太小，也可能是数据标签错误、梯度被 detach、参数没有加入优化器或所有样本都被 mask。若损失突然爆炸，可能是学习率过大，也可能是 loss 缩放、归一化、异常 batch 或溢出。调度器只是其中一个因素，不能把所有曲线异常都归因于学习率。

### 1.6.8 从六个机制回到一条训练链

现在可以把一个最小训练系统完整地读一遍：数据先形成有明确形状的张量；模型把输入映射为 logits；损失把 logits 与目标转成一个可微标量；反向传播沿计算图累积参数梯度；优化器结合梯度和历史状态生成更新；调度器为下一次更新提供学习率。任何一个环节都可能改变最终结果。

当这条链迁移到 Transformer 或语言模型时，线性层变多了，输入从 `[B, d]` 变成 `[B, T]`，输出从 `[B, C]` 变成 `[B, T, V]`，训练还会加入注意力、归一化、并行和混合精度，但基本问题没有变化：每个张量是什么形状，损失对哪些位置求平均，梯度从哪里来，参数在何时更新，学习率沿什么横轴变化。把这些问题逐一写清楚，才可能把“代码能跑”推进到“结果可解释”。

下一章会沿着这条训练链搭出 Transformer 组件，继续追踪 token 表示怎样变成下一层可用的向量。

## 延伸阅读与证据边界

本章的 PyTorch 接口按 2.15 文档核对；已有代码验证记录仍绑定各状态标记注明的 PyTorch 2.12.0+cpu 环境，不能把文档核对当作在 2.15 上运行。算例用于复算数学关系，Adam 和 AdamW 的算法动机可回到原论文查证。

1. PyTorch, [*Autograd mechanics, 2.15*](https://docs.pytorch.org/docs/2.15/notes/autograd.html)：计算图、梯度记录和反向传播。
2. PyTorch, [*inference_mode, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.autograd.grad_mode.inference_mode.html)：它相对 no_grad 的限制，以及它为何不替代 model.eval()。
3. PyTorch, [*CrossEntropyLoss, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html)：logits、目标形状、ignore_index、reduction 和概率目标。
4. PyTorch, [*torch.logsumexp, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.logsumexp.html)：稳定的 log-sum-exp 运算。
5. PyTorch, [*SGD, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.optim.SGD.html)、[*Adam, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.optim.Adam.html) 与 [*AdamW, 2.15*](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)：实现参数、优化器状态和更新规则。
6. PyTorch, [*How to Adjust Learning Rate, 2.15*](https://docs.pytorch.org/docs/2.15/optim.html#how-to-adjust-learning-rate)：scheduler 的调用顺序和状态。
7. D. P. Kingma and J. Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)：一阶/二阶矩与偏差修正的原始算法表述。
8. I. Loshchilov and F. Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101)，ICLR 2019：AdamW 与 L2 正则解耦的算法动机。
9. PyTorch, [*Automatic Mixed Precision, 2.15*](https://docs.pytorch.org/docs/2.15/amp.html)：autocast、梯度缩放及其数值边界。
