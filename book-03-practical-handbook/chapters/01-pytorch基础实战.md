# 第 1 章 PyTorch 基础实战：把数学变成可运行的训练过程

神经网络的代码看起来常常只有几行：定义一个模型，计算损失，调用 `backward()`，再让优化器更新参数。真正困难的地方不在于记住这些 API，而在于知道每一行代码对应哪一个数学对象，知道张量的形状如何流动，也知道一个“训练成功”的数字究竟说明了什么。

这一章从一个最小的回归问题开始，把训练过程拆成六个互相连接的层次：模型如何产生预测，损失如何衡量误差，自动微分如何得到梯度，优化器如何使用梯度，学习率如何随训练阶段变化，以及这些机制在语言模型训练中如何重新出现。读者可以先把示例当作可运行的实验，再回头读公式；有经验的读者则可以把每个公式与 PyTorch 的实现细节对照起来。

代码中的超参数、数据和输出是教学构造，不能被误读为任何真实数据集上的基准结果。PyTorch 文档对 API 行为的说明属于官方接口资料；优化算法的数学表达则来自论文或教材。二者在这里并列使用，但证据等级不同：一个 API 的默认值需要以目标版本的文档为准，一个算法是否有效则还需要看任务、数据和训练规模。

## 1.1 从线性回归开始：一个参数更新到底改变了什么

### 1.1.1 先把对象和形状写在纸上

假设每个样本有 \(d\) 个特征，输入矩阵记为 \(X\)，批量中有 \(B\) 个样本，那么它的形状是 \([B,d]\)。线性模型有权重向量 \(w\in\mathbb{R}^{d}\) 和偏置 \(b\in\mathbb{R}\)，对整个批次的预测为：

```math
\hat{y}=Xw+b
```

这里的 \(b\) 会沿着批次维广播，所以 \(\hat y\) 的形状是 \([B]\)。如果把偏置写成形状 \([1]\) 或 \([B,1]\)，也可以得到等价的计算，但不能只凭“能广播”就认为形状设计正确。最安全的做法是先约定：样本维是第 0 维，特征维是第 1 维，目标值与预测值都采用 \([B]\)；代码中再用断言把这个约定固定下来。

初学者可以把 \(w\) 看成每个特征的“影响力度”，把 \(b\) 看成所有特征都为零时的基线。这个比喻只是帮助建立直觉。数学上，\(w_j\) 是函数对第 \(j\) 个输入分量的偏导数，前提是其余变量保持不变；当模型不再是线性的时，“一个特征固定影响多少”就不能用一个常数完整描述了。

对第 \(i\) 个样本，预测写成：

```math
\hat y_i=\sum_{j=1}^{d}x_{ij}w_j+b
```

如果真实目标是 \(y_i\)，最常用的教学损失是均方误差（Mean Squared Error，MSE）：

```math
L(w,b)=\frac{1}{B}\sum_{i=1}^{B}(\hat y_i-y_i)^2
```

平方有两个作用。第一，它把正负误差都变成非负量；第二，大误差会被放大，优化器会更重视离目标很远的样本。它也带来一个限制：异常值可能主导损失，因此“损失更小”不自动等价于“所有样本都更好”。

### 1.1.2 手推一次梯度，理解 `backward()` 的结果

令 \(e_i=\hat y_i-y_i\)。因为 \(\hat y_i\) 对 \(w_j\) 的偏导是 \(x_{ij}\)，链式法则给出：

```math
\frac{\partial L}{\partial w_j}
=\frac{2}{B}\sum_{i=1}^{B}e_i x_{ij}
```

对偏置则有：

```math
\frac{\partial L}{\partial b}
=\frac{2}{B}\sum_{i=1}^{B}e_i
```

这两个式子值得停下来读一遍。权重的梯度是“误差乘以对应特征”的平均值；如果某个特征在所有样本中都接近零，它对当前批次的权重更新就会很小。偏置的梯度只看平均误差，因为偏置对每个样本的贡献都是 1。

梯度下降用负梯度方向更新参数：

```math
w\leftarrow w-\eta\frac{\partial L}{\partial w},\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b}
```

\(\eta>0\) 是学习率。它不是“模型学习能力”的固定分数，而是当前参数空间中每次移动的步长尺度。学习率太大可能越过低损失区域，太小则可能在有限训练时间内几乎看不到变化。

PyTorch 的自动微分系统会根据参与运算的张量建立动态计算图。当一个需要梯度的标量损失调用 `backward()` 时，系统从输出向输入反向应用局部导数，并把结果累积到叶子张量的 `.grad` 属性中。下面的代码同时展示了形状、梯度和一次手工更新：

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
    w -= 0.1 * w.grad
    b -= 0.1 * b.grad

print(round(w.item(), 4), round(b.item(), 4))
```

这个片段的输出为：

```text
torch.Size([4])
41.0
-35.0 -12.0
3.5 1.2
```

初始预测全为零，目标平均为 6；偏置梯度是样本误差的平均值乘以 2，因此为 -12。权重梯度还要乘以每个样本的输入，得到 -35。这里的形状、梯度和更新后参数都可以直接由上面的公式复核。


这个小例子说明了一个重要的工作习惯：书稿中的示例输出也是可验证的事实，不应凭印象填写。对教学代码来说，输出、形状和梯度都是读者检查推导的锚点。若把学习率改成 \(0.01\)，同一轮更新会得到 \(w=0.35,b=0.12\)；参数数值随学习率改变，梯度本身则由当前批次和当前参数决定。

### 1.1.3 梯度为何要清零

PyTorch 默认会把多次反向传播得到的梯度相加，而不是自动覆盖。这样设计是为了支持一个损失由多个反向路径组成、或多个小批次累积后再更新参数的场景。若每个批次都要独立更新，却忘记清零，实际使用的就不是当前批次梯度，而是从上一次更新开始不断累积的结果。

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

第一次梯度是 \(2w=4\)，第二次梯度是 3，累积后得到 7。将 `.grad` 设为 `None` 是常用清零方式；`optimizer.zero_grad()` 则会对优化器管理的参数执行相应操作。是否使用 `set_to_none=True` 会影响清零后的状态表现和少量性能细节，实际项目应以目标 PyTorch 版本的文档和内存行为为准。

### 1.1.4 广播让代码简短，也可能掩盖错误

下面两个表达式的形状并不等价：

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

第二种情况中，形状 \([4]\) 会被视为 \([1,4]\)，再与 \([4,1]\) 广播成 \([4,4]\)。如果损失函数随后对全部元素求平均，程序可能仍然运行，却已经把四个样本两两组合了。对于回归任务，建议在损失前明确断言：

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

这里的 `target` 是与预测值配对的规范目标张量；如果数据管道实际返回的是列向量，就应该在进入损失前明确 `squeeze` 或重塑，而不是让广播替你做决定。断言不是装饰，而是把数据契约写进程序。大型训练任务最昂贵的错误往往不是立即抛出异常的错误，而是形状合法、结果却悄悄改变的错误。

### 1.1.5 一个不依赖 PyTorch 的梯度下降实验

为了把“优化器只是按照梯度移动”这件事从框架 API 中剥离出来，可以用纯 Python 写一个标量回归。数据由教学者构造，目标关系为 \(y=2x+1\)：

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

纯 Python 版本没有计算图，也没有 `.grad`，却完成了同一组数学操作。这种对照能帮助读者建立边界：自动微分减少的是求导和图管理的手工工作，并没有改变损失函数、梯度方向或学习率的含义。

## 1.2 MLP 分类器：从连续预测到离散决策

### 1.2.1 分类模型先输出分数，而不是类别

回归的输出是一个连续数。分类问题通常有 \(C\) 个类别，模型对每个类别输出一个实数分数，称为 logit。对一个批次，logits 的形状通常是 \([B,C]\)，第 \(i\) 行对应一个样本，第 \(c\) 列对应一个类别。logit 可以是负数，也不需要加起来等于 1；它只是后续概率计算的输入。

如果需要概率，可以使用 softmax：

```math
p_{ic}=\frac{\exp(z_{ic})}{\sum_{k=1}^{C}\exp(z_{ik})}
```

其中 \(z_{ic}\) 是第 \(i\) 个样本对类别 \(c\) 的 logit，\(p_{ic}\) 是对应概率。预测类别通常取最大 logit 的索引：

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

“神经元死亡”来自 ReLU 在 \(a<0\) 时梯度为零：如果某个单元长期落在负半轴，它可能很少收到更新。初始化、学习率、归一化和激活函数选择都会影响这个现象；不能看到一个负激活就断言模型已经失效，因为不同样本上的激活状态可能不同。

### 1.2.3 写一个可观察的 MLP

下面的模型使用二维输入和三个类别，数据仍是用于说明训练流程的人工构造。`forward` 返回 logits，不在模型内部做 softmax，这样损失函数可以直接使用原始分数。

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
    predicted = model(features).argmax(dim=1)
    accuracy = (predicted == labels).float().mean()

print(predicted.tolist())
print(round(loss.item(), 4))
print(round(accuracy.item(), 4))
```

在本地 PyTorch 2.12.0 环境和上述随机种子下，输出为：

```text
[0, 0, 1, 1, 2, 2]
0.0891
1.0
```

这里的损失和预测是在本地 PyTorch 2.12.0 环境实测得到的教学输出；更换 PyTorch 版本、硬件或随机数实现后，小数位可能变化。类别完全可分时，训练集准确率达到 1.0 并不说明模型学会了普遍规律；它只说明这六个样本在当前评估方式下被正确分类。

`CrossEntropyLoss` 对目标张量的要求也很具体。对于 `[B,C]` 的 logits，类别索引目标通常应是形状 `[B]`、整数类型 `torch.long`，取值范围为 `0` 到 `C-1`。如果传入 one-hot 浮点标签，某些版本可能支持概率目标，但语义与类别索引不同，不能把两种格式混为一谈。训练前检查形状和 dtype，比在损失变成 `nan` 后猜原因更有效。

### 1.2.4 `train()` 和 `eval()` 改变的是模块行为

`model.train()` 和 `model.eval()` 不会自动开始或停止梯度计算，它们切换的是模块的训练/评估行为。Dropout 在训练时随机丢弃激活，在评估时关闭随机丢弃；BatchNorm 在训练时更新运行统计量，在评估时使用已有统计量。若评估阶段忘记调用 `eval()`，指标可能带有随机性；若只调用 `eval()` 却仍保留计算图，则仍会浪费推理内存。

推理阶段常见的组合是：

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

这里有两件独立的事：第一行改变模块模式，第二行关闭 autograd 记录。若代码运行在需要更严格推理优化的场景，还可以研究 `torch.inference_mode()`；它的约束和行为应以目标版本官方文档为准，不能简单理解成“更快的 `no_grad()`”。

### 1.2.5 MLP 与 Transformer 前馈网络的联系

Transformer block 中的前馈网络（Feed-Forward Network，FFN）通常也采用“线性变换—非线性—线性变换”的形态：

```math
\mathrm{FFN}(x)=\phi(xW_1+b_1)W_2+b_2
```

与这里的 MLP 相比，主要差别在输入包含序列位置，且实际模型通常使用更大的隐藏维度、残差连接、归一化以及诸如 GELU 或门控激活等组件。数学骨架仍然相同：先把表示投影到较高维空间，在高维空间施加非线性，再投影回模型维度。

这个联系有助于理解“基础实战”与大模型工程并不是两套互不相干的知识。张量形状、logits、损失、自动微分和参数更新，都会原样进入 Transformer；变化的是规模、并行方式和稳定性要求。

## 1.3 交叉熵：为什么训练分类器时直接使用 logits

### 1.3.1 从最大似然得到负对数似然

分类器对一个样本产生 \(C\) 个 logits，记为 \(z=(z_1,\ldots,z_C)\)。softmax 把它们变成概率：

```math
p_c=\frac{e^{z_c}}{\sum_{k=1}^{C}e^{z_k}}
```

若真实类别是 \(y\)，模型给真实类别的概率是 \(p_y\)。最大似然希望这个概率尽可能大；为了把乘法概率转化为可相加、可优化的量，使用负对数似然：

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

直接计算 `exp(z)` 会在 logits 较大时溢出，在 logits 较小时下溢。这个问题不是抽象的边角情况：深度网络的 logit 尺度可能随着初始化、归一化、精度格式和训练阶段变化。利用恒等式，可以先令 \(m=\max_k z_k\)：

```math
\log\sum_k e^{z_k}
=m+\log\sum_k e^{z_k-m}
```

平移后至少有一个指数是 \(e^0=1\)，其余指数不大于 1。这个公式是 `torch.logsumexp` 这类实现的核心思想。对真实类别 \(y\)，稳定的交叉熵可以写成：

```math
\ell(z,y)=m+\log\sum_k e^{z_k-m}-z_y
```

下面的纯 Python 版本不依赖 PyTorch，显式展示稳定计算：

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

对 logits 的第 \(c\) 个分量求导，可得：

```math
\frac{\partial \ell}{\partial z_c}=p_c-\mathbf{1}[c=y]
```

其中 \(\mathbf{1}[c=y]\) 在 \(c=y\) 时为 1，否则为 0。因此真实类别的梯度是 \(p_y-1\)，通常为负值，梯度下降会把对应 logit 往上推；错误类别的梯度是 \(p_c\)，为正值，梯度下降会把对应 logit 往下推。

这个推导还揭示了一个尺度性质：所有类别的 logits 同时加上同一个常数，softmax 概率不变，损失也不变。模型真正需要学习的是类别之间的相对分数，而不是一个绝对基准。数值稳定的 log-sum-exp 正是在利用这一平移不变性。

如果一个类别的概率已经非常接近 1，它的梯度接近 0；这不是自动微分失效，而是当前样本对该参数方向的局部信号确实很小。多个样本、多个层和参数共享会共同决定最终梯度，不能只凭单个 logit 的变化判断整个模型是否还在学习。

### 1.3.4 `reduction`、类别权重与无效标签

对一个批次的逐样本损失 \(\ell_i\)，常见的聚合方式有三种：

```math
L_{\text{none}}=[\ell_1,\ldots,\ell_B],\qquad
L_{\text{sum}}=\sum_i\ell_i,\qquad
L_{\text{mean}}=\frac{1}{B}\sum_i\ell_i
```

`reduction="mean"` 会让梯度规模与批次样本数大致保持可比；`sum` 则会随着有效样本数量增加而增大。在使用梯度累积、动态 padding 或分布式训练时，不能默认“每张卡的 mean 再平均”一定等于“全局有效 token 的 mean”。如果不同设备上的有效 token 数量不同，应明确分子和分母，按有效元素总数归一化。

类别不均衡时可以引入类别权重 \(\alpha_c\)：

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

自回归语言模型通常为每个位置预测下一个 token。设输入 token 形状为 \([B,T]\)，词表大小为 \(V\)，模型输出 logits 形状为 \([B,T,V]\)。标签需要向左移动一位：位置 \(t\) 的输出对应目标 \(t+1\)。抽象写法是：

```math
L=-\frac{1}{N_{\text{valid}}}
\sum_{b=1}^{B}\sum_{t=1}^{T-1}
m_{b,t}\log p(x_{b,t+1}\mid x_{b,\le t})
```

其中 \(m_{b,t}\) 表示该目标位置是否有效，且

```math
N_{\text{valid}}=\sum_{b=1}^{B}\sum_{t=1}^{T-1}m_{b,t}
```

是有效目标数量。这个平均式假定 \(N_{\text{valid}}>0\)；如果一个 batch 全是 padding，就不应把“除以 1 后得到的数”误当作有效训练信号，而应跳过更新并追查数据管道。实现时常把 logits 重排为 \([B\times T,V]\)，标签重排为 \([B\times T]\)：

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

只有当 \(L_{\text{token}}\) 确实按有效 token 平均，并且对数使用自然对数时，这个换算才有上述形式；还要保证损失是有限值。若有效 token 数为零，困惑度没有定义，不能用一个人为的保护分母制造可比较的 PPL。如果损失包含 padding、不同样本权重或跨批次归一化方式不同，直接比较 PPL 会产生误导。PPL 也不是所有生成质量的充分指标；它偏向评估概率建模目标，无法单独代表事实性、指令遵循或长文本结构能力。

## 1.4 反向传播：计算图、局部导数与梯度的路径

### 1.4.1 反向传播不是“把公式倒着算一遍”

自动微分首先执行前向计算，并记录足以计算局部导数的信息。反向阶段从损失开始，沿着图的反方向传播一个上游梯度；每经过一个运算节点，就把上游梯度乘以该节点对输入的局部导数。

例如：

```math
u=xy,\qquad v=\sin x,\qquad L=u+v
```

从 \(L\) 到 \(x\) 有两条路径：

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

如果把 \(x\) 同时送入两个分支后再相加，反向传播必须把两个分支的局部导数都保留下来。计算图的“动态”含义是：每次前向运行都可以根据 Python 控制流构造不同的图，而不是先声明一张固定的静态图。

### 1.4.2 标量损失与非标量输出

最常见的训练写法是让损失变成标量，再调用：

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

这里传入的全 1 向量表示对 \(v_1+v_2\) 求导。如果传入 `[1.0, 0.0]`，只会选择第一项对参数的贡献。这个接口让用户能够表达向量输出的加权组合，但也意味着“调用 `backward()`”本身并不说明你对哪个标量目标求了梯度。

### 1.4.3 叶子张量、`.grad` 与中间结果

由用户直接创建、且 `requires_grad=True` 的参数通常是叶子张量，优化器关注的就是这些张量。中间运算结果也可能需要梯度，但默认不会把它保存在 `.grad` 属性中；如果确实需要观察中间激活，可以在反向前调用 `retain_grad()`：

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

因为 \(hidden=3w=6\)，\(L=hidden^2\)，所以 \(\partial L/\partial hidden=12\)，再乘以 \(\partial hidden/\partial w=3\)，得到 \(36\)。如果只打印 `hidden.grad` 却没有调用 `retain_grad()`，得到 `None` 并不表示反向传播没有经过该节点。

### 1.4.4 `detach()`、`no_grad()` 和参数更新

这三个概念经常被混在一起，但作用不同。

`detach()` 从一个已有张量创建一个不再向前连接的视图或张量关系：后续对 detached 结果的运算不会把梯度传回原图。它适合表达“把这个表示当作常量使用”的语义，例如目标网络、缓存特征或某些强化学习计算。

`torch.no_grad()` 是一个上下文管理器，表示在这段代码执行期间不记录大多数 autograd 运算。优化器更新参数时通常使用它，因为更新动作本身不应成为下一轮计算图的一部分：

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

反向传播有时需要前向阶段保存的中间值。例如 \(y=x^2\) 的梯度需要知道前向时的 \(x\)。如果在反向前原地修改了这个值，系统无法保证使用的是正确版本，通常会抛出版本计数错误。

参数更新也是原地修改，但它发生在 `no_grad()` 中，并且通常在本轮反向完成之后，所以不会把更新操作接入当前计算图。对需要梯度的叶子张量直接做原地运算，可能立即报错；对中间激活做原地运算，可能破坏之后某个节点需要的保存值。

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

当自定义算子、手写损失或复杂张量变换出现可疑结果时，可以用有限差分做小规模检查。对标量函数 \(f(\theta)\)，中心差分近似为：

```math
f'(\theta)\approx
\frac{f(\theta+\varepsilon)-f(\theta-\varepsilon)}{2\varepsilon}
```

\(\varepsilon\) 太大时截断误差明显，太小时浮点舍入误差明显；它不是一个对所有 dtype 和尺度都固定的常数。下面只检查一个标量二次函数：

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

## 1.5 优化器：梯度告诉方向，状态决定怎样走

### 1.5.1 SGD 是一个更新规则，不是训练的全部

最基本的随机梯度下降（SGD）使用当前小批次估计的梯度：

```math
\theta_{t+1}=\theta_t-\eta_t g_t,
\qquad g_t=\nabla_\theta L_{\mathcal B_t}(\theta_t)
```

这里 \(\theta\) 表示所有可训练参数，\(\mathcal B_t\) 是第 \(t\) 次使用的小批次，\(g_t\) 是该小批次损失对参数的梯度。小批次梯度通常不是全数据梯度的精确值，而是一个带噪声的估计。噪声有时帮助模型离开狭窄区域，但也会让损失曲线抖动。

批次大小改变了梯度估计的方差、每次更新的计算量以及单位样本的吞吐。不能只说“批次越大越好”：在固定总 token 数下，大批次意味着更新次数更少；在固定更新次数下，大批次又意味着看过更多样本。比较实验时需要说明究竟固定了样本数、token 数、更新次数，还是墙钟时间。

PyTorch 的 `SGD` 还支持动量、dampening、Nesterov 和 weight decay 等选项。只写出类名并不能确定实际算法；完整实验记录至少要包含学习率、动量、weight decay、nesterov、是否最大化目标以及参数组设置。

### 1.5.2 Momentum 为什么能平滑小批次噪声

一种常见的动量形式是：

```math
v_t=\mu v_{t-1}+g_t,\qquad
\theta_{t+1}=\theta_t-\eta_t v_t
```

\(\mu\) 是动量系数，\(v_t\) 是梯度的指数加权累积。沿着长期一致的方向，历史梯度会叠加；方向频繁变化的噪声则部分抵消，因此参数轨迹通常比裸 SGD 更平滑。

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

其中 \(\epsilon\) 防止分母为零，也会影响极小二阶矩时的有效步长。`betas`、`eps` 和学习率共同决定更新尺度；只比较“Adam 比 SGD 收敛快”而不报告这些设置，结论是不完整的。

Adam 的二阶矩会为每个参数保存一个与参数同形状的缓存，通常还要保存一阶矩。因此优化器状态可能接近参数本体的两倍；使用 FP32 参数、梯度和状态时，显存估算不能只看模型参数数量。分布式训练、参数分片和 8-bit optimizer state 会改变这个账本，必须按实际实现核算。

### 1.5.4 Adam 与 AdamW：L2 正则和 weight decay 不总是同一件事

把 L2 正则加入目标函数，得到：

```math
L_{\text{reg}}(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2
```

对应梯度是：

```math
\nabla L_{\text{reg}}(\theta)=g_t+\lambda\theta_t
```

对于普通 SGD，直接把 \(\lambda\theta\) 加入梯度与乘法形式的 weight decay 有紧密关系。但在 Adam 中，这个正则项会进入自适应的一阶、二阶统计，被不同参数的梯度尺度重新缩放，因而不再等同于“统一收缩参数”。

AdamW 把衰减从自适应梯度更新中解耦。简化写法是：

```math
\theta_{t+1}=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

这里的 \(\lambda\) 是参数收缩强度，\(\eta_t\) 是当前学习率。两者相乘意味着学习率调度也会影响每一步实际衰减量。实践中常把 bias、LayerNorm/RMSNorm 的 scale 和某些特殊参数排除在 weight decay 之外；这种参数分组是建模选择，不是 AdamW 自动推断出来的。

下面的代码展示参数分组的结构。它不声称某个任务的最优配置，只说明如何让不同参数使用不同衰减策略：

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

典型的单批次训练顺序是。下面的代码包含了运行它所需的最小对象；在较大的训练程序中，`inputs`、`labels`、`model`、`loss_fn` 和 `optimizer` 通常由数据管道与训练初始化代码提供：

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

显存不足时，可以让 \(K\) 个 micro-batch 共用一次参数更新。若每个 micro-batch 的损失已经是该批次的平均值，且每个 micro-batch 的有效样本数相同，通常应将每次损失除以 \(K\)：

```math
L_{\text{acc}}=\frac{1}{K}\sum_{k=1}^{K}L_k
```

代码中不会真的先构造一个跨 micro-batch 的 \(L_{\text{acc}}\)，而是对每个 \(L_k/K\) 分别反向，最后调用一次 `optimizer.step()`；两种写法在计算图仍可保留且归一化条件相同时等价。否则梯度大约会放大 \(K\) 倍，等价的学习率也随之改变。若各 micro-batch 有效 token 数不同，简单除以 \(K\) 可能仍不等于全局有效 token 平均；应累计未归一化损失和有效数量，再按总有效数量归一化。

下面的最小示例保留最后一个不完整组。它假定每个 micro-batch 的有效样本数相同；如果序列长度或有效 token 数不同，应把 `loss_fn` 改为 `reduction="sum"`，另外累计有效数量，再按总有效数量归一化：

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

输出为 `2`：前四个 micro-batch 产生一次更新，最后一个 micro-batch 单独产生一次更新。若简单地把最后一个损失仍除以 4，第二次更新的梯度就会被缩小到应有值的四分之一。这个边界在 `drop_last=True` 时可以通过丢弃不完整组来回避，但是否丢弃数据必须成为明确的训练决策。

有效批次大小常写成：

```math
B_{\text{effective}}
=B_{\text{micro}}\times K\times N_{\text{data-parallel}}
```

这是在每个设备上的 micro-batch 大小、累积步数和数据并行设备数定义清楚，且每次更新都覆盖相同数量有效样本时的简化表达。序列任务中更准确的单位往往是有效 token 数，而不是样本数。

### 1.5.7 混合精度改变的是数值路径，不是数学目标

混合精度通常让部分矩阵运算使用 FP16 或 BF16，同时保留某些参数、归约和优化器状态的更高精度。它减少内存和带宽压力，但也缩小了可表示的数值范围或有效精度。FP16 训练常配合 loss scaling：先把损失乘以一个较大的尺度 \(s\)，反向得到 \(sg\)，更新前再除以 \(s\)，从而避免小梯度下溢；如果检测到无穷或 NaN，则跳过本次更新并调整尺度。

如果训练还使用梯度缩放器，梯度裁剪前必须先把梯度还原到未缩放的尺度；否则裁剪阈值会同时受到缩放因子影响。常见流程是先反向，再取消缩放，再裁剪，最后让缩放器执行参数更新并更新自己的尺度。

BF16 的指数范围接近 FP32，通常更不容易溢出，但尾数精度较低；这不是说 BF16 在所有任务上都无需数值保护。自动混合精度上下文、梯度缩放器和优化器的组合应以目标设备、PyTorch 版本及算子支持为准。

训练日志至少应同时记录：损失是否有限、梯度范数、学习率、跳过更新次数以及有效 token 数。只记录一个下降的 loss，无法区分“模型正常学习”和“某些 batch 被悄悄跳过”。

## 1.6 学习率调度：让更新尺度与训练阶段相匹配

### 1.6.1 学习率是时间的函数

把固定学习率写成 \(\eta\) 只是最简单的情况。更一般地，学习率是优化器更新步 \(s\) 的函数：

```math
\theta_{s+1}=\mathrm{Update}
(\theta_s,g_s,\eta_s,\mathrm{state}_s)
```

调度器改变 \(\eta_s\)，并不直接改变损失、数据顺序或模型结构。训练前期通常需要较小步幅以建立稳定状态，随后可以提高到目标学习率；训练后期则常降低步幅，让参数在已有解附近细化。是否需要 warmup、何时衰减和最终降到多低，都依赖优化器、模型规模、数据量与训练预算。

### 1.6.2 线性 warmup 的含义

设目标学习率为 \(\eta_{\max}\)，warmup 更新步数为 \(S_w>0\)，用 1-based 的更新步 \(s\) 表示：

```math
\eta_s=\eta_{\max}\min\left(1,\frac{s}{S_w}\right)
```

当 \(1\le s\le S_w\) 时，学习率线性增加；之后保持 \(\eta_{\max}\)，直到后续调度改变它。若初始学习率设置为 0，第一步的更新也可能为 0；不同调度器对 `last_epoch` 和首次调用时刻的定义会影响具体序列，因此应把实际打印的学习率作为事实依据。

warmup 并非“越长越安全”。过长会减少高学习率阶段的训练预算，过短又可能无法缓冲初始化、较大批次或混合精度带来的早期不稳定。比较 warmup 实验时，应固定总更新步数，并报告 warmup 占比，而不只报告最终 loss。

### 1.6.3 常见衰减函数

Step decay 在每隔 \(S\) 步把学习率乘以 \(\gamma\)：

```math
\eta_s=\eta_0\gamma^{\lfloor s/S\rfloor},
\qquad 0<\gamma<1
```

指数衰减连续地变化：

```math
\eta_s=\eta_0\gamma^s
```

余弦退火在一个周期内平滑下降到 \(\eta_{\min}\)：

```math
\eta_s=\eta_{\min}
+\frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1+\cos\frac{\pi s}{S}\right),
\quad 0\le s\le S
```

其中 \(S\) 是余弦周期的更新步数。若在 \(S\) 之后继续使用同一公式，必须说明是否重启、截断或进入另一个周期；不能只写“用了 cosine”就认为实验设置完整。

PyTorch 提供 `StepLR`、`ExponentialLR`、`CosineAnnealingLR`、`LambdaLR` 等调度器。它们记录的是调度状态和计数器，不会替代优化器状态。恢复训练时只加载模型参数而不加载 optimizer/scheduler state，可能使动量、二阶矩和学习率序列突然回到错误状态。

### 1.6.4 调度器的步长必须与参数更新对齐

如果一个 batch 被拆成 \(K\) 个 micro-batch，但只在最后一个 micro-batch 调用 `optimizer.step()`，那么学习率调度器通常也应每次参数更新调用一次。下面是一个包含最小上下文的示意：

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

```python
import torch
from torch import nn

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
torch.save(checkpoint, "checkpoint.pt")
```

恢复时先构造相同的模型、优化器和调度器，再分别加载对应状态。若先加载 scheduler 状态却改变了 optimizer 参数组数量或顺序，可能出现难以察觉的学习率错配；参数组结构应在恢复前保持兼容。

随机状态和数据位置尤其重要。只恢复模型权重，可以做“从这个权重继续微调”，但它不等于“从中断处无缝续训”。两者的目标不同，报告结果时应该用不同措辞。

### 1.6.7 用日志判断调度是否真的生效

学习率调度问题常常不在公式，而在调用位置、参数组或恢复逻辑。下面的示例先完成一次真实更新，再读取当前参数组学习率和 Adam 的状态步数：

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

## 延伸阅读与证据边界

本章的 API 行为以 PyTorch 官方文档为主，算法部分用经典论文和官方实现文档交叉核对。网页版本会更新，读者复现实验时应记录访问日期、PyTorch 版本、CUDA/CPU 环境和关键默认值。

1. PyTorch, *Autograd mechanics*：<https://pytorch.org/docs/stable/notes/autograd.html>。用于核对计算图、梯度记录、`no_grad` 与 `detach` 的行为。
2. PyTorch, *CrossEntropyLoss*：<https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>。用于核对 logits、目标形状、`ignore_index`、`reduction` 和概率目标的接口语义。
3. PyTorch, *torch.logsumexp*：<https://pytorch.org/docs/stable/generated/torch.logsumexp.html>。用于核对稳定 log-sum-exp 运算的 API。
4. PyTorch, *SGD*、*Adam* 与 *AdamW*：<https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>、<https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>、<https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>。用于核对实现参数、状态和更新顺序；页面中的默认值不能脱离目标版本使用。
5. D. P. Kingma and J. Ba, *Adam: A Method for Stochastic Optimization*，arXiv:1412.6980。用于一阶/二阶矩和偏差修正的原始算法表述。
6. I. Loshchilov and F. Hutter, *Decoupled Weight Decay Regularization*，ICLR 2019。用于 AdamW 与 L2 正则解耦的算法动机。
7. PyTorch, *How to Adjust Learning Rate* 与 scheduler API：<https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>。用于核对 scheduler 的调用顺序、状态和不同调度器的接口。
8. Hugging Face Transformers, *Optimization and Schedules*：<https://huggingface.co/docs/transformers/main_classes/optimizer_schedules>。用于对照语言模型训练中 warmup、更新步和调度器封装的工程语境；它是框架文档，不等同于某个模型训练结果的独立证据。
