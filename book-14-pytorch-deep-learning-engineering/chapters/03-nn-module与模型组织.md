# 第三章：nn.Module 与模型组织

`nn.Module` 是 PyTorch 组织模型的核心抽象。第一章讲的是 tensor，第二章讲的是 autograd；到了真实工程里，我们不会把所有参数、矩阵乘法和激活函数都散写在训练脚本中，而是把它们封装成一个个 module，再组合成完整模型。

很多 PyTorch 初学者会把 `nn.Module` 理解成“写神经网络时必须继承的类”，但这只是表层。`nn.Module` 真正重要的地方在于：它负责注册参数、注册子模块、管理 buffer、递归切换训练/推理模式、递归移动 device、导出和加载 `state_dict`，并为优化器、分布式训练、混合精度和模型保存提供统一接口。

如果你不理解这些机制，很容易遇到这类问题：明明定义了参数但 optimizer 没有更新、保存的 checkpoint 里缺字段、`model.eval()` 后结果仍然随机、加载权重时报 missing key、把 tensor 放进 list 里导致参数没有注册、推理时 batch norm 或 dropout 行为不对、LoRA 参数没有进入 optimizer、DDP 包装后 key 多了 `module.` 前缀。

本章目标是把 `nn.Module` 在工程中的规则讲清楚：如何写 `forward`，什么是 `Parameter`，子模块如何注册，`state_dict` 保存什么，`train()` 和 `eval()` 到底影响什么，模型保存加载应该怎么做，以及如何组织一个可维护的大模型代码结构。

可沿三条连续链阅读：先从对象注册树走到参数、buffer 和 optimizer；再从模块状态走到 `state_dict`、checkpoint 和恢复；最后把这些规则放进大模型结构、故障排查、分布式和编译包装中。遇到“参数没更新、模型没迁移或 checkpoint 不匹配”时，都回到同一问题：该状态由哪个对象拥有，是否注册，谁在读取或更新它。

## 0. 本章范围与资料

本章以 PyTorch 2.15 官方 `nn.Module`、`nn.Parameter`、`register_buffer`、`state_dict`、`load_state_dict`、容器、训练/评估模式、模型保存加载和 `torch.compile` 文档为 API 语义依据。文末列出可直接核验的资料入口，并区分 API 事实、工程惯例和目标系统复测；本章的本地代码运行记录会单独标明实际 PyTorch 版本。

本章聚焦大模型工程最常用的 Module 组织问题：参数注册、子模块注册、buffer、`state_dict`、保存加载、训练/推理模式、device / dtype 迁移、参数冻结、参数分组、hook、DDP / FSDP 依赖的模块边界，以及最小可运行的 Module 审计 demo。

本章不展开 C++ dispatcher、底层 autograd engine、完整分布式训练实现、FSDP wrap policy 细节、`torch.export` / AOTAutograd 编译器内部机制或生产级 checkpoint shard 格式。这些内容会分别放到分布式训练、profiling、推理部署和 AI Infra 相关章节中讲。文中的小模型实验用于验证注册、保存和模式切换等机制，不代表大型模型在特定硬件上的吞吐或显存结果。

## 3.1 nn.Module 解决什么问题

先看一个不用 `nn.Module` 的极简线性模型：

```python
import torch

w = torch.randn(10, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

x = torch.randn(4, 10)
y = x @ w + b
loss = y.pow(2).mean()
loss.backward()
```
> 代码片段，静态审阅，未运行。

这段代码可以求梯度，但它不是一个好工程结构。问题在于：

1. 参数 `w`、`b` 散落在外部变量中。
2. 训练脚本需要手动知道哪些 tensor 是参数。
3. 保存加载时需要自己维护字段名。
4. 复杂模型会有大量层、buffer 和子结构，无法靠几个变量管理。
5. 分布式、混合精度、设备迁移都需要统一遍历模型内部对象。

`nn.Module` 提供了统一容器：

```python
import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(4, 10)
y = model(x)
loss = y.pow(2).mean()
loss.backward()
optimizer.step()
```
> 代码片段，静态审阅，未运行。

这里 `self.linear = nn.Linear(10, 1)` 不只是普通赋值。因为 `LinearModel` 继承了 `nn.Module`，PyTorch 会把 `linear` 识别为子模块并注册起来。后续 `model.parameters()`、`model.state_dict()`、`model.to(device)`、`model.train()` 都会递归作用到它；规则见 [PyTorch 2.15 `nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)。

### 3.1.1 注册树、状态和参数更新

`nn.Module` 本身不是一个数学层，而是一套工程组织规则。理解它的关键不是记住很多类名，而是能回答三个问题：这个对象是否在模块树上、它是否会出现在保存状态中、优化器是否拿到了它。

第一，参数量来自注册参数，而不是来自 Python 变量名。以下参数量公式假定各维度是正整数，并且只统计实际创建的参数，不把临时激活或未注册 tensor 算进去。线性层参数量为：

~~~math
N = d_{in} d_{out} + d_{out}
~~~

如果 `bias=False`，则只有：

~~~math
N = d_{in} d_{out}
~~~

Embedding 表的参数量为：

~~~math
N_{emb} = V d
~~~

其中 `V` 是词表大小，`d` 是 hidden size。一个两层 MLP 的参数量常写成：

~~~math
N_{mlp} = d_{in} d_h + d_h + d_h d_{out} + d_{out}
~~~

第二，optimizer 更新的是它拿到的 `Parameter`。单个参数的 SGD 更新可以写成：

~~~math
\theta_{t+1} = \theta_t - \eta g_t
~~~

~~~math
g_t = \frac{\partial L}{\partial \theta_t}
~~~

如果某个 tensor 没有注册成 `Parameter`，或者某个子模块藏在普通 list / dict 中，它即使参与 forward，也不会自动出现在 `model.parameters()` 里，optimizer 通常不会更新它。

第三，默认 `state_dict` 不是保存整个 Python 对象，而是保存注册权重和持久化 buffer：

~~~math
S_{default}(M) = P(M) \cup B_p(M)
~~~

其中 `P(M)` 表示 module 树上的注册参数，`B_p(M)` 表示 `persistent=True` 的 buffer。普通属性和临时 tensor 不会自动保存；`persistent=False` 的 buffer、`forward` 代码和 Python 类定义也不在默认模型 `state_dict` 中。模块还可以实现 `get_extra_state()` / `set_extra_state()` 或注册状态字典 hook，显式加入额外状态，所以实际字典不保证只含 tensor。额外对象必须可序列化；PyTorch 对跨版本序列化的兼容承诺主要针对 tensor，详见 [PyTorch 2.15 `nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)。若两个模块共享同一个 `Parameter`，`state_dict` 可能为两个访问路径保留两个 key；因此 key 的数量不一定等于唯一参数存储的数量，参数统计时还要考虑别名。

下表把对象所在位置与框架能管理的状态对应起来。`ModuleList` / `ModuleDict` 中的子模块按递归注册处理；普通 Python 容器不会自动注册其中的 module 或 `Parameter`。

| 对象 | `parameters()` | `buffers()` | 默认 `state_dict()` | `Module.to()` | 从 `model.parameters()` 交给 optimizer |
| --- | --- | --- | --- | --- | --- |
| 注册的 `Parameter` | 是 | 否 | 是 | 是 | 是 |
| 属性、`ModuleList` 或 `ModuleDict` 中的已注册子模块 | 递归包含其参数 | 递归包含其 buffer | 递归包含参数和持久 buffer | 递归迁移 | 递归包含其参数 |
| 持久 buffer | 否 | 是 | 是 | 是 | 否 |
| `persistent=False` buffer | 否 | 是 | 否 | 是 | 否 |
| 普通 tensor 属性 | 否 | 否 | 否 | 否 | 否 |
| 普通 list/dict 中的 module 或 `Parameter` | 否 | 否 | 否 | 否 | 否 |

表中的 optimizer 列说的是“通过 `model.parameters()` 取得”。优化器也能接收调用者手工传入的独立参数，但这不会让它们自动进入模型的保存、迁移或分布式注册树。

第四，参数高效微调或冻结模型时，先看可训练参数比例：

~~~math
r = \frac{\sum_{p \in P_{train}} |p|}{\sum_{p \in P_{all}} |p|}
~~~

其中 `|p|` 是一个参数 tensor 的元素个数，`P_train` 和 `P_all` 按 `model.parameters()` 默认的对象去重结果统计，且分母必须大于 0；如果模型没有参数，这个比例没有定义。LoRA、adapter、prompt tuning 的核心工程检查就是：应该训练的参数是否都被注册、`requires_grad=True`、进入 optimizer，并且保存时能从 `state_dict` 或 `named_parameters()` 中筛出来。

第五，`train()` / `eval()` 改的是模块状态，不是梯度开关：

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```
> 代码片段，静态审阅，未运行。

`eval()` 递归设置 `training=False`，影响 Dropout、BatchNorm 等层；`no_grad()` / `inference_mode()` 关闭 autograd 记录。验证和推理通常两者都要用。

最小 Module 审计清单：

1. `list(model.named_parameters())` 中是否出现目标层参数。
2. `list(model.named_buffers())` 中是否出现固定 mask、统计量或位置编码。
3. `model.state_dict().keys()` 是否包含该保存的权重，是否排除了临时缓存。
4. `sum(p.numel() for p in model.parameters() if p.requires_grad)` 是否等于 optimizer 参数组里的元素数。
5. `model.train()` / `model.eval()` 是否递归影响所有子模块。
6. 保存、加载后同一输入在 eval 模式下输出是否一致。
7. `strict=False` 加载时的 missing / unexpected keys 是否符合预期，而不是被忽略。

这套检查把“定义了一个对象”和“框架会管理这个对象”区分开来。普通 Python 属性可以被 forward 读取，但只有注册为参数、子模块或 buffer 后，才会自动参与相应的遍历、迁移、保存和分布式处理。

## 3.2 最小 Module 写法

一个标准的 module 通常包含两部分：

1. `__init__`：定义层、参数、buffer 和配置。
2. `forward`：定义前向计算逻辑。

示例：

```python
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```
> 代码片段，静态审阅，未运行。

几个关键习惯：

1. 必须先调用 `super().__init__()`。
2. 在 `__init__` 中创建需要长期持有的层和参数。
3. 在 `forward` 中写真实计算，不要在里面反复创建可训练层。
4. `forward` 可以使用 Python 控制流，PyTorch 动态图会按实际路径构图。

常见错误是把层写在 `forward` 里：

```python
class BadMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x):
        layer = nn.Linear(self.hidden_size, self.hidden_size)
        return layer(x)
```
> 代码片段，静态审阅，未运行。

这段代码的问题很严重：每次 forward 都会新建一组随机参数，这些参数不会被稳定训练，optimizer 也不知道它们应该被长期更新。除非你明确要创建临时无参数对象，否则可训练层应该放在 `__init__` 中。

正确写法：

```python
class GoodMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)
```
> 代码片段，静态审阅，未运行。

## 3.3 forward 和 __call__ 的关系

使用模型时，我们通常写：

```python
output = model(x)
```
> 代码片段，静态审阅，未运行。

而不是：

```python
output = model.forward(x)
```
> 代码片段，静态审阅，未运行。

虽然最终会执行 `forward`，但两者不完全等价。`model(x)` 经过 `nn.Module` 的调用路径并运行已注册的 forward pre/post hooks，然后进入 `forward`；直接写 `model.forward(x)` 会绕过这层调用路径。有关 hook 的签名和调用时机见 [PyTorch 2.15 `nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)。

因此工程中建议：

1. 外部调用永远写 `model(inputs)`。
2. 不要直接调用 `model.forward(inputs)`，除非是在非常明确的内部调试场景。
3. 自定义逻辑写在 `forward` 中，不要重写 `__call__`。

例如：

```python
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


model = ToyModel()
out = model(torch.randn(3, 4))
```
> 代码片段，静态审阅，未运行。

## 3.4 Parameter：什么会被 optimizer 更新

`nn.Parameter` 是一种特殊 tensor。它的作用是告诉 `nn.Module`：这个 tensor 是模型参数，需要被注册，应该出现在 `model.parameters()` 和 `state_dict()` 中。

示例：

```python
import torch
from torch import nn


class ScaleLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.scale
```
> 代码片段，静态审阅，未运行。

检查参数：

```python
layer = ScaleLayer(4)

for name, param in layer.named_parameters():
    print(name, param.shape, param.requires_grad)
```
> 代码片段，静态审阅，未运行。

输出类似：

```text
scale torch.Size([4]) True
```

如果你只是写普通 tensor：

```python
class BadScaleLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.scale = torch.ones(hidden_size, requires_grad=True)

    def forward(self, x):
        return x * self.scale
```
> 代码片段，静态审阅，未运行。

这个 `scale` 虽然 `requires_grad=True`，但它不是 `Parameter`，不会出现在 `model.parameters()` 中，optimizer 通常也不会更新它。

可以验证：

```python
bad = BadScaleLayer(4)
print(list(bad.named_parameters()))  # []
```
> 代码片段，静态审阅，未运行。

核心规则：

1. 需要训练的长期权重，用 `nn.Parameter` 或标准层如 `nn.Linear`。
2. 中间激活不要定义成 `Parameter`。
3. 不需要训练但要随模型保存的状态，用 buffer。
4. `requires_grad=False` 的 `Parameter` 仍是参数，只是不计算梯度。

## 3.5 子模块注册：赋值不是普通赋值

当你把一个 `nn.Module` 赋值给另一个 module 的属性时，它会被自动注册为子模块。

```python
class Block(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))
```
> 代码片段，静态审阅，未运行。

`self.norm` 和 `self.mlp` 都会被注册。`self.mlp` 里面的 `Linear` 也会被递归注册。

可以查看模块树：

```python
block = Block(768)

for name, module in block.named_modules():
    print(name, type(module).__name__)
```
> 代码片段，静态审阅，未运行。

输出类似：

```text
 Block
norm LayerNorm
mlp Sequential
mlp.0 Linear
mlp.1 GELU
mlp.2 Linear
```

这里第一个空 name 表示根模块本身。

常见坑是把子模块放进普通 Python list：

```python
class BadStack(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = [Block(hidden_size) for _ in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```
> 代码片段，静态审阅，未运行。

这段代码能 forward，但 `self.layers` 里的 block 不会被注册为子模块。后果是：

1. `model.parameters()` 找不到这些参数。
2. `model.to("cuda")` 不会移动这些参数。
3. `state_dict()` 不会保存这些权重。
4. DDP 不会正确同步这些参数。

正确写法是 `nn.ModuleList`：

```python
class GoodStack(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Block(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```
> 代码片段，静态审阅，未运行。

用一个两层小模型把注册、前向和状态大小连起来。令第一层权重为 `[2, -1]`、偏置为 `1`，第二层权重为 `2`、偏置为 `-1`，输入为 `[3, 2]`：

```python
import torch
from torch import nn


class HandCalcStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 1),
            nn.Linear(1, 1),
        ])
        self.register_buffer("offset", torch.tensor(0.0))

    def forward(self, x):
        x = self.layers[0](x)
        return self.layers[1](x) + self.offset


model = HandCalcStack()
with torch.no_grad():
    model.layers[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
    model.layers[0].bias.copy_(torch.tensor([1.0]))
    model.layers[1].weight.copy_(torch.tensor([[2.0]]))
    model.layers[1].bias.copy_(torch.tensor([-1.0]))

x = torch.tensor([[3.0, 2.0]])
state = model.state_dict()
print("output:", model(x).item())
print("parameter elements:", sum(p.numel() for p in model.parameters()))
print("state tensor elements:", sum(v.numel() for v in state.values()))
print("state keys:", list(state))
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

手算第一层得到 `2×3 + (-1)×2 + 1 = 5`；第二层得到 `2×5 - 1 + 0 = 9`。第一层有两个权重和一个偏置，共 3 个参数；第二层有一个权重和一个偏置，共 2 个参数，因此模型有 5 个可训练标量。`offset` 是一个持久 buffer，不参加 optimizer，却和参数一起保存，所以 state dict 有 6 个 tensor 元素，key 为两层的 weight/bias 和 `offset`。若把 `ModuleList` 换成普通 list，前向仍可调用这些层，但这 5 个参数不会由 `model.parameters()`、`state_dict()` 或 `model.to()` 管理。

如果是按名称组织，可以用 `nn.ModuleDict`：

```python
class MultiHeadModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleDict({
            "classification": nn.Linear(hidden_size, 2),
            "regression": nn.Linear(hidden_size, 1),
        })

    def forward(self, x, task):
        return self.heads[task](x)
```
> 代码片段，静态审阅，未运行。

## 3.6 Sequential、ModuleList、ModuleDict 怎么选

三者都能组织子模块，但语义不同。

`nn.Sequential` 表示线性流水线：前一个模块的输出直接作为后一个模块的输入。

```python
mlp = nn.Sequential(
    nn.Linear(128, 512),
    nn.GELU(),
    nn.Linear(512, 128),
)
```
> 代码片段，静态审阅，未运行。

适合简单串联结构。缺点是不适合复杂控制流、多输入多输出或 residual connection。

`nn.ModuleList` 表示一组已注册模块，但不规定 forward 如何执行。

```python
self.layers = nn.ModuleList([
    Block(hidden_size) for _ in range(num_layers)
])
```
> 代码片段，静态审阅，未运行。

适合 Transformer block 堆叠，因为 forward 通常要加入 cache、mask、gradient checkpointing、输出 hidden states 等逻辑。

`nn.ModuleDict` 表示按字符串 key 管理子模块。

```python
self.adapters = nn.ModuleDict({
    "domain_a": Adapter(hidden_size),
    "domain_b": Adapter(hidden_size),
})
```
> 代码片段，静态审阅，未运行。

适合多任务 head、多 adapter、多分支结构。

选择建议：

1. 简单串联用 `Sequential`。
2. 多层堆叠但 forward 需要自己控制，用 `ModuleList`。
3. 按名称动态选择子模块，用 `ModuleDict`。
4. 不要用普通 `list` 或 `dict` 存放可训练子模块。

## 3.7 ParameterList 和 ParameterDict

如果你存的是参数而不是子模块，可以用 `nn.ParameterList` 和 `nn.ParameterDict`。

示例：

```python
class WeightedSum(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(())) for _ in range(num_inputs)
        ])

    def forward(self, xs):
        out = 0
        for weight, x in zip(self.weights, xs):
            out = out + weight * x
        return out
```
> 代码片段，静态审阅，未运行。

如果直接用普通 list：

```python
self.weights = [nn.Parameter(torch.ones(())) for _ in range(num_inputs)]
```
> 代码片段，静态审阅，未运行。

这些参数不会被注册。

`ParameterDict` 适合按名字管理参数：

```python
class TaskBias(nn.Module):
    def __init__(self, tasks, hidden_size):
        super().__init__()
        self.biases = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(hidden_size))
            for task in tasks
        })

    def forward(self, x, task):
        return x + self.biases[task]
```
> 代码片段，静态审阅，未运行。

在大模型工程中，直接使用 `ParameterList` 和 `ParameterDict` 的机会不如 `ModuleList` 多，但在自定义 embedding、adapter、可学习 prompt、可学习加权融合时很有用。

## 3.8 buffer：不是参数，但属于模型状态

有些 tensor 不需要训练，但仍然属于模型状态，需要随模型移动 device、保存和加载。例如：

1. BatchNorm 的 running mean 和 running variance。
2. 固定的位置编码。
3. attention mask 的固定模板。
4. 量化时的 scale、zero point。
5. 一些统计量或缓存状态。

这类 tensor 应该注册为 buffer：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len]
```
> 代码片段，静态审阅，未运行。

buffer 的特点：

1. 不会出现在 `model.parameters()` 中。
2. 默认会出现在 `state_dict()` 中。
3. 会随 `model.to(device)` 移动。
4. 可以通过 `persistent=False` 设置为不保存。

例如：

```python
self.register_buffer("causal_mask", mask, persistent=False)
```
> 代码片段，静态审阅，未运行。

`persistent=False` 适合可重新生成的大缓存，避免 checkpoint 过大。

查看 buffer：

```python
for name, buf in model.named_buffers():
    print(name, buf.shape, buf.device)
```
> 代码片段，静态审阅，未运行。

## 3.9 parameters、named_parameters 和参数分组

`model.parameters()` 返回模型中所有注册参数，通常直接传给 optimizer：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```
> 代码片段，静态审阅，未运行。

但真实训练里经常需要参数分组，例如：

1. 对 bias 和 LayerNorm 不使用 weight decay。
2. backbone 和 head 使用不同学习率。
3. 只训练 LoRA 或 adapter 参数。
4. 冻结一部分参数。

这时需要 `named_parameters()`：

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

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=1e-4)
```
> 代码片段，静态审阅，未运行。

为什么 LayerNorm 和 bias 常常不做 weight decay？直觉上，weight decay 更适合约束大矩阵权重，而归一化层的 scale、bias 和偏置参数承担的是平移缩放作用，强行衰减可能影响训练稳定性。这个是常见工程惯例，不是 PyTorch 的硬性规则；是否采用它取决于优化器、模型结构和实验结果。参数分组后还应检查每个参数只进入一个 group，避免重复引用或遗漏。

冻结参数：

```python
for name, param in model.named_parameters():
    if name.startswith("backbone"):
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)
```
> 代码片段，静态审阅，未运行。

如果在创建 optimizer 后冻结参数，优化器仍保留这些参数引用；通常不会为已冻结参数生成新梯度，但旧 `.grad`、显式零梯度、weight decay 或动量状态会影响后续行为。要让参数组清楚地对应当前训练目标，应清理旧梯度并重建参数组，或明确调用 `add_param_group` 管理后来解冻的参数。仅仅设置 `requires_grad=False` 不会自动重写 optimizer 的参数组。

## 3.10 state_dict 保存什么

`state_dict()` 是 PyTorch 保存模型权重的标准方式。

```python
state = model.state_dict()

for key, value in state.items():
    description = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
    print(key, description)
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

默认保存的是：

1. 所有注册参数。
2. 默认持久化的 buffer。

它不会自动保存：

1. Python 类定义本身。
2. `forward` 代码逻辑。
3. optimizer 状态，除非单独保存 optimizer 的 `state_dict()`。
4. 未注册在 module 上的普通 tensor。
5. `persistent=False` 的 buffer。

若自定义 module 通过 `get_extra_state()` 或 state-dict hook 加入额外值，这些值也可能进入返回字典；因此不能假定每个 value 都是权重 tensor。

以一个简单模型为例：

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.LayerNorm(8),
    nn.Linear(8, 2),
)

print(model.state_dict().keys())
```
> 代码片段，静态审阅，未运行。

可能输出：

```text
odict_keys([
    '0.weight', '0.bias',
    '1.weight', '1.bias',
    '2.weight', '2.bias'
])
```

key 来自模块层级路径。大模型里的 key 可能是：

```text
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.mlp.gate_proj.weight
model.norm.weight
lm_head.weight
```

理解 key 的层级非常重要，因为加载权重、排查 missing key、做 LoRA merge、切分 checkpoint 都依赖这些名称。

还要注意 `state_dict()` 的对象生命周期。它返回的是一个保存 key 的映射，里面的 tensor 通常仍与当前模块参数共享底层内容；把它赋给 `best_state` 后继续训练，并不等于自动冻结了“最佳时刻”的数值。这个浅拷贝语义见 [PyTorch 2.15 API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)。若模型使用标准 tensor 参数和 buffer，保留最佳权重时应显式复制到 CPU 或 clone：

```python
best_state = {
    name: value.detach().cpu().clone()
    for name, value in model.state_dict().items()
}
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

自定义 extra state 若不是 tensor，应按它自己的类型建立独立副本；不能直接对任意 state value 调用 tensor 方法。

否则常见的“验证集最好时保存了权重，训练结束后加载却变成最后一步”的问题，可能不是 `load_state_dict` 失效，而是保存的映射一直指向正在变化的存储。

## 3.11 保存和加载模型

推荐保存方式是保存 `state_dict`，而不是直接保存整个模型对象。

保存：

```python
torch.save(model.state_dict(), "model.pt")
```
> 代码片段，静态审阅，未运行。

加载：

```python
model = MyModel(config)
state = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
```
> 代码片段，静态审阅，未运行。

为什么不推荐直接保存整个模型？

```python
torch.save(model, "model.pt")
```
> 代码片段，静态审阅，未运行。

这种方式依赖 Python pickle，需要加载时能找到完全一致的类路径和代码结构。工程迭代后很容易因为类名、文件路径或依赖变化加载失败。保存 `state_dict` 更稳定，也更符合跨版本和跨工程迁移习惯。

PyTorch 2.6 起，在未传入其他 pickle module 时，`torch.load` 默认采用 `weights_only=True`；本章显式写出该参数以便读者看清加载边界。受限加载适合 tensor、基本类型、字典等受支持对象组成的 checkpoint；自定义配置类、tensor subclass 或由 `get_extra_state()` 写入的对象可能需要审查并显式 allowlist。模型配置宜存成基本类型组成的字典，再由代码重建模型。该限制会缩小 pickle 动态反序列化的范围，但不防御所有拒绝服务或下游风险。不要对来源不可信的文件改用 `weights_only=False`；具体限制见 [PyTorch 2.15 `torch.load`](https://docs.pytorch.org/docs/2.15/generated/torch.load.html) 和[序列化说明](https://docs.pytorch.org/docs/2.15/notes/serialization.html)。

完整训练 checkpoint 通常包含：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "epoch": epoch,
    "config": config_dict,  # 用数字、字符串、列表和字典等受支持的基本类型
}

torch.save(checkpoint, "checkpoint.pt")
```
> 代码片段，静态审阅，未运行。

恢复：

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
step = checkpoint["step"]
```
> 代码片段，静态审阅，未运行。

推理只需要模型权重时，通常不需要 optimizer 和 scheduler。

恢复训练与恢复推理是两个不同的目标。若要尽量复现中断前的训练轨迹，至少还要考虑混合精度 scaler、随机数状态、数据采样器位置、梯度累积边界、当前学习率和配置版本：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict() if scaler is not None else None,
    "rng_torch": torch.get_rng_state(),
    "step": step,
    "config": config_dict,
}
```
> 代码片段，静态审阅，未运行。

这里的 `rng_torch` 只示意 CPU PyTorch 随机状态。若训练还使用 Python `random`、NumPy、CUDA 随机数或带 worker 的采样器，应根据恢复目标保存相应状态和数据进度。这仍然不是绝对的 bitwise reproducibility 保证：算子确定性、worker 时序、分布式通信和外部数据版本也会影响后续轨迹。好的 checkpoint 设计应明确写出“用于推理”“用于继续训练”还是“用于审计复现”，三者需要保存的状态不同。

## 3.12 strict=True 和 strict=False

`load_state_dict` 默认 `strict=True`，表示 checkpoint 中的 key 必须和当前模型完全匹配。

```python
model.load_state_dict(state, strict=True)
```
> 代码片段，静态审阅，未运行。

如果不匹配，会报 missing key 或 unexpected key：

1. missing key：当前模型需要某些参数，但 checkpoint 没有。
2. unexpected key：checkpoint 有某些参数，但当前模型不需要。

常见原因：

1. 模型结构改了。
2. 层名称改了。
3. DDP 保存时带了 `module.` 前缀。
4. 加载 backbone 到带分类头的新模型。
5. 只加载 LoRA、adapter 或部分模块。

如果确实只想部分加载，可以用 `strict=False`：

```python
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)
```
> 代码片段，静态审阅，未运行。

`strict=False` 只放宽缺失 key 和多余 key 的检查；同名参数的 shape 不一致仍会报错。[`load_state_dict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) 的 `strict` 只定义 key 是否严格匹配。若有意跳过尺寸变化的层，应先按 key 和 shape 筛选 checkpoint，并逐项记录被跳过或需要裁剪/扩展的参数。不要把 `strict=False` 当成万能修复：例如加载 backbone 到新分类模型时，分类头 missing 可能合理；但如果 Transformer block 大量 missing，说明 key 对不上或模型结构不一致。

处理 DDP 的 `module.` 前缀时，优先使用 PyTorch 提供的 helper：

```python
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

consume_prefix_in_state_dict_if_present(state, "module.")
model.load_state_dict(state)
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

更好的习惯是在 DDP 训练时保存未包装模型或保存 `model.module.state_dict()`。

PyTorch 2.15 的 `load_state_dict` 还提供 `assign=False` 参数。默认值会把 checkpoint 值复制到目标模型现有的参数和 buffer；若选择 `assign=True`，张量属性更多地沿用 checkpoint 一侧，官方文档提醒除非启用了参数交换机制，否则应在加载后再创建 optimizer。本章示例使用默认 `assign=False`，避免把两种参数对象生命周期混在一起；参数和 optimizer 的顺序约束见 [API 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)。

## 3.13 train() 和 eval() 到底改变什么

`model.train()` 和 `model.eval()` 用于切换模块的训练/推理模式。

```python
model.train()
```
> 代码片段，静态审阅，未运行。

等价于：

```python
model.train(True)
```
> 代码片段，静态审阅，未运行。

推理模式：

```python
model.eval()
```
> 代码片段，静态审阅，未运行。

等价于：

```python
model.train(False)
```
> 代码片段，静态审阅，未运行。

它们会递归设置每个子模块的 `training` 标志：

```python
print(model.training)
```
> 代码片段，静态审阅，未运行。

最受影响的层是：

1. Dropout。
2. BatchNorm。

Dropout 在训练时会随机置零部分激活，在 eval 时不再随机丢弃：

```python
dropout = nn.Dropout(p=0.5)
x = torch.ones(10)

dropout.train()
print(dropout(x))  # 随机有些位置为 0

dropout.eval()
print(dropout(x))  # 通常保持不变
```
> 代码片段，静态审阅，未运行。

BatchNorm 在训练时使用当前 batch 的统计量并更新 running statistics，在 eval 时使用保存的 running statistics。

例外是 `track_running_stats=False`：这时 BatchNorm 没有保存运行统计量，评估时也会使用当前 batch 的统计，所以同一个样本放在不同 batch 中可能得到不同结果，见 [PyTorch 2.15 BatchNorm 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html)。`eval()` 只设置模块的 `training` 标志；自定义层若不检查该标志，或推理代码仍进行随机采样，输出仍可能变化。

重要区别：`model.eval()` 不等于关闭梯度。

```python
model.eval()
with torch.no_grad():
    output = model(x)
```
> 代码片段，静态审阅，未运行。

推理时通常两者都需要：

1. `model.eval()`：改变 dropout、batch norm 等模块行为。
2. `torch.no_grad()` 或 `torch.inference_mode()`：关闭 autograd，节省显存和计算开销。

## 3.14 Module.to、device 和 dtype 管理

`model.to(device)` 会递归移动模型参数和 buffer。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
> 代码片段，静态审阅，未运行。

`.to()` 会原地修改模块。设备迁移完成后再创建 optimizer 是稳妥顺序；PyTorch 2.15 对 [`Module.cuda()`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.cuda) 明确提醒迁移可能让参数对象改变，因此应在 optimizer 构造前完成迁移。

输入数据也必须在同一个 device：

```python
x = x.to(device)
output = model(x)
```
> 代码片段，静态审阅，未运行。

常见错误：

```text
Expected all tensors to be on the same device
```

通常是模型在 GPU、输入在 CPU，或者某个未注册 tensor 没有被 `model.to(device)` 移动。

这也是为什么固定 mask、位置编码这类 tensor 应该注册为 buffer，而不是普通属性：

```python
class BadMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = torch.ones(1, 1, 1024, 1024)

    def forward(self, x):
        return x.masked_fill(self.mask == 0, 0)
```
> 代码片段，静态审阅，未运行。

`self.mask` 不会随 `model.to("cuda")` 移动，容易 device mismatch。

正确写法：

```python
class GoodMask(nn.Module):
    def __init__(self):
        super().__init__()
        mask = torch.ones(1, 1, 1024, 1024)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        return x.masked_fill(self.mask == 0, 0)
```
> 代码片段，静态审阅，未运行。

`model.to(dtype=torch.bfloat16)` 可以转换浮点参数和 buffer 的 dtype：

```python
model = model.to(dtype=torch.bfloat16)
```
> 代码片段，静态审阅，未运行。

注意：整数类型 tensor 不会被转换成浮点；token id 通常仍然是 `torch.long`。

## 3.15 apply、children、modules 和递归遍历

`nn.Module` 提供了多种遍历方式。

直接子模块：

```python
for child in model.children():
    print(type(child).__name__)
```
> 代码片段，静态审阅，未运行。

所有递归子模块，包括自身：

```python
for module in model.modules():
    print(type(module).__name__)
```
> 代码片段，静态审阅，未运行。

带名称遍历：

```python
for name, module in model.named_modules():
    print(name, type(module).__name__)
```
> 代码片段，静态审阅，未运行。

这些接口常用于：

1. 自定义初始化。
2. 替换某些层。
3. 给特定模块加 hook。
4. 收集特定模块的参数。
5. 打印模型结构。

`modules()`、`named_modules()`、`parameters()` 和 `named_parameters()` 默认会去重同一个对象。权重共享或同一子模块挂在多个访问路径时，遍历到的数量可能少于路径 key 数；需要审计别名时，应同时检查对象身份和 state dict key，而不能只看名称数量。

自定义初始化示例：

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


model.apply(init_weights)
```
> 代码片段，静态审阅，未运行。

`apply` 会递归访问所有子模块，并对每个模块执行函数。

替换层时要小心，不要一边遍历一边直接修改复杂层级导致遗漏。简单场景可以这样：

```python
for name, module in model.named_children():
    if isinstance(module, nn.ReLU):
        setattr(model, name, nn.GELU())
```
> 代码片段，静态审阅，未运行。

复杂模型替换通常需要写递归函数，或者在模型构建阶段就通过配置决定层类型。

## 3.16 自定义初始化

初始化对训练稳定性很重要。PyTorch 标准层自带默认初始化，但大模型项目经常会按论文或架构要求自定义初始化。

一个常见模式：

```python
class TinyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Block(config.hidden_size) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
```
> 代码片段，静态审阅，未运行。

初始化要注意：

1. 不要在每次 forward 时初始化。
2. 初始化通常放在模型构造结束后。
3. 加载预训练权重后不要再次随机初始化覆盖权重。
4. 特定结构可能需要特殊初始化，例如 residual 分支缩放。

如果要做权重绑定，例如语言模型里 embedding 和 LM head 共享权重：

```python
self.lm_head.weight = self.embed_tokens.weight
```
> 代码片段，静态审阅，未运行。

这表示两个模块引用同一个 `Parameter`。保存、加载、优化器更新时要理解它们是共享权重，而不是两份独立参数。共享权重会让参数统计、checkpoint key 和优化器状态出现别名关系：逻辑上可能有两个使用位置，实际只应有一份可训练存储。`model.parameters()` 默认会去重同一个参数对象，而 `state_dict` 可能保留两个名称，因此不能用 key 数量代替参数量。

加载也不会凭空建立绑定。若模型构造函数先执行了 `self.lm_head.weight = self.embed_tokens.weight`，再加载 state dict，两个名称最终仍然写入同一份参数存储；若目标模型原本是两份独立参数，即使 checkpoint 中的两个 tensor 数值相同，`load_state_dict` 也只会分别拷贝数值，不会自动把它们变成别名。需要共享权重时，应在模型结构定义阶段显式绑定，并在加载后检查：

```python
assert model.lm_head.weight is model.embed_tokens.weight
```
> 代码片段，静态审阅，未运行。

如果手工构造 optimizer 参数列表，还要避免把共享参数重复加入；直接使用 `model.parameters()` 通常会得到去重后的参数迭代器。

## 3.17 大模型中的 Module 组织方式

一个简化版 decoder-only Transformer 通常可以按下面层级组织：

```text
LanguageModel
embed_tokens
layers
  0 DecoderLayer
    self_attn
      q_proj
      k_proj
      v_proj
      o_proj
    mlp
      gate_proj
      up_proj
      down_proj
    input_layernorm
    post_attention_layernorm
  1 DecoderLayer
  ...
norm
lm_head
```

代码骨架：

```python
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
```
> 代码片段，静态审阅，未运行。

这种组织方式的优点：

1. 层级清楚，便于读 `state_dict`。
2. 参数命名稳定，便于加载和迁移。
3. 每个子模块职责明确，便于单元测试。
4. 可以局部替换 attention、MLP、norm 或 adapter。
5. 便于 FSDP、DDP、checkpointing 这类工程工具按模块粒度处理。

## 3.18 forward 返回值怎么设计

小模型可以直接返回 tensor：

```python
return logits
```
> 代码片段，静态审阅，未运行。

但大模型常常需要返回更多信息：

1. `loss`。
2. `logits`。
3. `hidden_states`。
4. `attentions`。
5. `past_key_values`。

简单项目可以返回 dict：

```python
return {
    "loss": loss,
    "logits": logits,
    "hidden_states": hidden_states,
}
```
> 代码片段，静态审阅，未运行。

也可以使用 dataclass：

```python
from dataclasses import dataclass


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
```
> 代码片段，静态审阅，未运行。

设计原则：

1. 训练脚本需要什么就返回什么，不要过度复杂化。
2. 返回结构要稳定，避免不同分支返回完全不同类型。
3. 如果支持 `output_hidden_states`，默认可以关闭，避免额外显存。
4. loss 可以在 model 内部算，也可以在 trainer 外部算，但项目内要统一。

例如语言模型 forward：

```python
def forward(self, input_ids, labels=None):
    logits = self.compute_logits(input_ids)
    loss = None

    if labels is not None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    return {"loss": loss, "logits": logits}
```
> 代码片段，静态审阅，未运行。

这里要注意：如果使用 `view`，前面常常需要 `contiguous()`；如果不想关心连续性，可以用 `reshape`。

## 3.19 hooks：观察和修改中间过程

`nn.Module` 支持 hook，用于观察或修改 forward、backward 过程。常见用途：

1. 抓取中间激活。
2. 调试某一层输出是否有 NaN。
3. 统计激活范围。
4. 做可视化。
5. 在研究代码中临时修改输出。

forward hook 示例：

```python
activations = {}


def save_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach()
    return hook


handle = model.layers[0].register_forward_hook(save_activation("layer0"))

with torch.no_grad():
    _ = model(input_ids)

print(activations["layer0"].shape)

handle.remove()
```
> 代码片段，静态审阅，未运行。

hook 使用注意：

1. 用完要 `remove()`，否则可能影响后续实验。
2. 保存激活时通常要 `detach()`，否则可能保留计算图导致显存增长。
3. hook 会增加调试复杂度，不适合作为核心业务逻辑长期依赖。
4. 分布式和编译场景下 hook 行为可能更复杂。

上例假设被观察层返回单个 tensor，且该层本轮只调用一次；若输出是 tuple/dict，或同一层会被多次调用，应显式选择要保存的张量并给调用位置编号，避免后一次输出覆盖前一次记录。

## 3.20 register_module、setattr 和动态添加层

有时模型结构需要根据配置动态构建。只要用 `setattr` 把 module 赋给 `nn.Module` 属性，也会注册。

显式调用 `add_module(name, module)` 也可注册子模块；`register_module` 是它的别名。一般模型用属性赋值或 `ModuleList` 更易读，手工注册接口主要用于确实需要动态命名的构建器。

```python
class DynamicModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, f"layer_{i}", nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        for i in range(self.num_layers):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
        return x
```
> 代码片段，静态审阅，未运行。

这样才会按照构造时的层数执行。它能工作，但不如 `ModuleList` 清晰，因为层名和遍历逻辑被拆开了，改结构时还要同步维护 `getattr`。更好的写法：

```python
self.layers = nn.ModuleList([
    nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
])
```
> 代码片段，静态审阅，未运行。

动态添加层要注意：如果 optimizer 已经创建，再向模型里添加新参数，optimizer 默认不知道这些新参数。

```python
model.new_head = nn.Linear(768, 2)
```
> 代码片段，静态审阅，未运行。

如果这发生在 optimizer 创建之后，需要重新创建 optimizer，或者显式 `add_param_group`。大多数情况下，应在构建 optimizer 前完成模型结构定义。

## 3.21 常见 bug：参数没有注册

症状：loss 在变，但某些层参数不更新；或者 `state_dict` 里没有预期权重。

常见原因：

1. 子模块放在普通 list 或 dict 中。
2. 可训练 tensor 没有包装成 `nn.Parameter`。
3. 在 `forward` 中临时创建层。
4. 创建 optimizer 后才添加新层。
5. 手动覆盖了已经注册的参数或模块。

排查方法：

```python
for name, param in model.named_parameters():
    print(name, param.shape, param.requires_grad)
```
> 代码片段，静态审阅，未运行。

检查目标层是否出现。如果没有出现，说明没有注册。

再检查 optimizer：

```python
num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_optim_params = sum(
    p.numel()
    for group in optimizer.param_groups
    for p in group["params"]
)

print(num_model_params, num_optim_params)
```
> 代码片段，静态审阅，未运行。

如果两者不一致，要确认是否有意冻结、是否参数分组遗漏、是否创建 optimizer 的时机不对。

## 3.22 常见 bug：eval 后结果仍然变化

如果 `model.eval()` 后同一个输入输出仍然变化，可能原因包括：

1. forward 中手动调用了随机函数，例如 `torch.randn`、随机采样。
2. 推理逻辑里使用了采样解码，例如 multinomial sampling。
3. 模型内部有自定义 dropout 没有遵守 `self.training`。
4. 输入本身不同，例如 padding 或 mask 不一致。
5. 某些 CUDA 算子存在非确定性。

自定义 dropout 类逻辑应该检查 `self.training`：

```python
class CustomDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return torch.zeros_like(x)
        mask = torch.rand_like(x) > self.p
        return x * mask / (1 - self.p)
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

如果只是推理忘了关梯度，输出通常不会因此随机变化，但会浪费显存：

```python
model.eval()
with torch.inference_mode():
    output = model(x)
```
> 代码片段，静态审阅，未运行。

## 3.23 常见 bug：加载 checkpoint 报错

典型报错包括：

```text
Missing key(s) in state_dict
Unexpected key(s) in state_dict
size mismatch for ...
```

排查顺序：

1. 打印当前模型 `state_dict().keys()`。
2. 打印 checkpoint 的 keys。
3. 比较是否有 `module.` 前缀差异。
4. 检查模型配置是否一致，例如 hidden size、层数、词表大小。
5. 检查是否只加载部分模块。

size mismatch 通常比 missing key 更严重，因为名称对上了但 shape 不一致。例如词表大小变化：

```text
size mismatch for embed_tokens.weight
```

如果只是扩展词表，需要特殊处理：

```python
old_weight = state["embed_tokens.weight"]
new_weight = model.embed_tokens.weight

num = min(old_weight.size(0), new_weight.size(0))
with torch.no_grad():
    new_weight[:num].copy_(old_weight[:num])
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

这类操作要明确知道语义，不能随便忽略。

## 3.24 LoRA、Adapter 与可训练参数筛选

在参数高效微调中，经常冻结 base model，只训练 LoRA 或 adapter 参数。

本章只检查这些参数是否注册并进入 optimizer；LoRA 的低秩分解和参数量推导见[第十三册第五章《矩阵分解与低秩》](../../book-13-math-foundations/chapters/05-矩阵分解与低秩.md)。

典型逻辑：

```python
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "lora_" in name or "adapter" in name:
        param.requires_grad = True

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
```
> 代码片段，静态审阅，未运行。

检查可训练参数：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
```
> 代码片段，静态审阅，未运行。

LoRA 参数能否被 optimizer 看到，仍然取决于注册机制：LoRA 层里的 A、B 矩阵必须是注册的 `nn.Parameter`，LoRA 子模块必须挂在模型模块树上，否则 optimizer、state_dict、DDP 都不会自动处理它们。只按 `requires_grad=True` 筛选还不够；还要核对参数命名、target module、adapter 配置和保存/加载后的结构。

保存 LoRA 时也常常只保存可训练参数：

```python
lora_state = {
    name: param.detach().cpu().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}

torch.save(lora_state, "lora.pt")
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

真实项目中还要保存 LoRA 配置，例如 rank、alpha、target modules，否则只保存权重不够恢复结构。

## 3.25 DDP、FSDP 与 Module 边界

分布式训练工具依赖 `nn.Module` 的参数树。

DDP 常见写法：

```python
model = model.to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
)
```
> 代码片段，静态审阅，未运行。

DDP 会遍历模型参数，注册梯度同步 hook。没有注册到 module 树上的参数不会被同步。

构造 DDP 前必须完成参数和 buffer 的注册；包装后不得再增删参数或 buffer，而且各进程的参数注册顺序必须一致。否则 DDP 已安装的梯度归约 hook 与真实参数集合/顺序不再匹配。动态加层要发生在 DDP 包装之前，并在所有 rank 上以相同结构执行；这是 [PyTorch 2.15 DDP API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html) 的显式前提。本册的进程组、数据分片和梯度同步主讲见[第七章《分布式训练入门》](07-分布式训练入门.md)。

保存时常用：

```python
if rank == 0:
    torch.save(model.module.state_dict(), "model.pt")
```
> 代码片段，静态审阅，未运行。

因为 DDP 包装后的外层对象有 `.module` 指向原始模型。

FSDP 更依赖模块边界，因为它会按模块切分、扁平化、分片参数。模型组织得越清晰，越容易指定 wrap policy，例如按 Transformer block 包装。

工程建议：

1. 不要把整个大模型写在一个巨大 `forward` 函数里。
2. Transformer block、attention、MLP、norm 分成清晰子模块。
3. 参数创建在 `__init__` 中完成。
4. 避免未注册参数和临时层。
5. 保存时明确是 DDP 包装前还是包装后的 `state_dict`。

## 3.26 torch.compile 对 Module 的影响

`torch.compile` 会尝试捕获并编译模型可支持的计算段，是否加速取决于代码、输入、后端和运行环境：

```python
model = torch.compile(model)
```
> 代码片段，静态审阅，未运行。

它通常包装 module 的调用过程，而不是改变你写 `forward` 的基本规则。为了更容易被编译和优化，模型代码应尽量：

1. 避免在 `forward` 中创建新的可训练层。
2. 避免过多依赖 Python 副作用。
3. 避免每次 forward 动态改变模块结构。
4. 对 shape 变化和控制流保持可预期。

不是所有模型都适合直接 compile。遇到 graph break、编译时间过长、显存变化或数值不一致时，需要先与 eager 输出对照，再逐步缩小范围，例如只 compile 某个 block 或训练/推理函数。PyTorch 2.15 的 [compiler troubleshooting 文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)建议选择问题较少的高层函数、顶层 module 或子 module；对 DDP/FSDP 包装场景，文档建议优先考虑编译传给分布式 wrapper 的内部 module。这个建议是版本化排障入口，不是所有组合都已在本章运行验证。

本册的最小复现、Profiler 和编译性能验证流程放在[第八章《Debug 与 Profiling》](08-debug与profiling.md)；本节只交代 Module 边界及版本化排查入口。

这里的关键不是背 compile 参数，而是知道：`nn.Module` 仍然是模型结构入口，`forward` 的 Python 写法会影响编译器是否能稳定捕获计算图。compile 前后都应比较数值、输出结构和状态切换行为，不应把一次成功编译当作所有动态输入都已兼容。

## 3.27 最小可运行 Module 审计 demo

下面的 demo 把本章最容易踩坑的机制放到一个小例子里：普通 list 不注册子模块、`ModuleList` 会注册子模块、`Parameter` 会进入参数表、buffer 会进入 `state_dict`、`persistent=False` buffer 不保存、`train/eval` 会递归切换、冻结参数后 optimizer 需要重新按可训练参数创建、保存加载后 eval 输出应一致。

```python
import tempfile
from pathlib import Path

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, x):
        return x + self.net(x)


class BadStack(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GoodStack(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TinyClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_blocks):
        super().__init__()
        self.input_scale = nn.Parameter(torch.ones(input_size))
        self.register_buffer("feature_mean", torch.zeros(input_size))
        self.register_buffer("scratch_mask", torch.ones(1, input_size), persistent=False)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        x = (x - self.feature_mean) * self.input_scale
        hidden = self.encoder(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.dropout(hidden)
        logits = self.head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "hidden": hidden}


torch.manual_seed(7)

input_size = 6
hidden_size = 12
num_classes = 2

bad_stack = BadStack(hidden_size, num_layers=2)
good_stack = GoodStack(hidden_size, num_layers=2)
model = TinyClassifier(input_size, hidden_size, num_classes, num_blocks=2)

bad_stack_param_names = [name for name, _ in bad_stack.named_parameters()]
good_stack_param_names = [name for name, _ in good_stack.named_parameters()]
param_names = [name for name, _ in model.named_parameters()]
buffer_names = [name for name, _ in model.named_buffers()]
state_keys = list(model.state_dict().keys())
module_names = [name for name, _ in model.named_modules() if name]

model.train()
recursive_train = all(module.training for module in model.modules())
model.eval()
recursive_eval = not any(module.training for module in model.modules())

x = torch.randn(64, input_size)
labels = (x[:, 0] + 0.5 * x[:, 1] > 0).long()

model.train()
model.dropout.p = 0.0
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
losses = []

for _ in range(40):
    optimizer.zero_grad(set_to_none=True)
    outputs = model(x, labels=labels)
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    losses.append(round(float(loss.detach()), 4))

for name, param in model.named_parameters():
    if name.startswith("encoder") or name.startswith("blocks"):
        param.requires_grad = False

trainable_params = [param for param in model.parameters() if param.requires_grad]
frozen_optimizer = torch.optim.AdamW(trainable_params, lr=0.01)
num_model_trainable = sum(param.numel() for param in trainable_params)
num_optimizer_params = sum(
    param.numel()
    for group in frozen_optimizer.param_groups
    for param in group["params"]
)
trainable_ids = {id(param) for param in trainable_params}
optimizer_param_ids = [
    id(param)
    for group in frozen_optimizer.param_groups
    for param in group["params"]
]
optimizer_identity_match = (
    len(optimizer_param_ids) == len(set(optimizer_param_ids))
    and set(optimizer_param_ids) == trainable_ids
)

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "tiny_classifier.pt"
    torch.save(model.state_dict(), path)

    clone = TinyClassifier(input_size, hidden_size, num_classes, num_blocks=2)
    clone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    model.eval()
    clone.eval()
    with torch.inference_mode():
        same_after_load = torch.allclose(
            model(x)["logits"],
            clone(x)["logits"],
            atol=1e-6,
        )

partial_state = dict(model.state_dict())
partial_state.pop("head.weight")
partial_state.pop("head.bias")
fresh = TinyClassifier(input_size, hidden_size, num_classes, num_blocks=2)
missing, unexpected = fresh.load_state_dict(partial_state, strict=False)

report = {
    "bad_stack_param_names": bad_stack_param_names,
    "good_stack_param_count": len(good_stack_param_names),
    "module_count": len(module_names),
    "has_input_scale": "input_scale" in param_names,
    "buffer_names": buffer_names,
    "state_has_feature_mean": "feature_mean" in state_keys,
    "state_has_scratch_mask": "scratch_mask" in state_keys,
    "loss_first_last": (losses[0], losses[-1]),
    "trainable_after_freeze": num_model_trainable,
    "optimizer_param_count": num_optimizer_params,
    "optimizer_identity_match": optimizer_identity_match,
    "missing_keys": sorted(missing),
    "unexpected_keys": unexpected,
}

checks = {
    "bad_stack_unregistered": len(bad_stack_param_names) == 0,
    "modulelist_registered": len(good_stack_param_names) == 4,
    "parameter_registered": "input_scale" in param_names,
    "buffer_registered": "feature_mean" in buffer_names,
    "buffer_saved": "feature_mean" in state_keys,
    "nonpersistent_buffer_skipped": "scratch_mask" not in state_keys,
    "train_eval_recursive": recursive_train and recursive_eval,
    "loss_decreased": losses[-1] < losses[0],
    "optimizer_matches_trainable": optimizer_identity_match,
    "strict_false_reports_missing": (
        sorted(missing) == ["head.bias", "head.weight"] and unexpected == []
    ),
    "load_roundtrip_same_logits": same_after_load,
}

print("module audit report:")
for key, value in report.items():
    print(f"{key}: {value}")

print("checks:")
for key, value in checks.items():
    print(f"{key}: {value}")

assert all(checks.values())
```
> 代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

如果这个 demo 在有 PyTorch 的环境里运行，应该看到：

1. `BadStack` 的参数名为空，说明普通 list 没有注册子模块。
2. `GoodStack` 有 4 个参数 tensor，来自两个 Linear 的 weight 和 bias。
3. `input_scale` 出现在 `named_parameters()` 中。
4. `feature_mean` 出现在 buffer 和 `state_dict` 中。
5. `scratch_mask` 是非持久化 buffer，不出现在 `state_dict` 中。
6. `train()` / `eval()` 递归切换所有子模块。
7. 冻结后 optimizer 参数量和可训练参数量一致。
8. 保存加载后 eval 模式下 logits 一致。
9. `strict=False` 只报告有意删除的 `head.weight` 和 `head.bias`。

这个例子的重点不是分类任务本身，而是建立 Module 工程审计习惯：每次遇到“参数没更新、checkpoint 加载不对、eval 行为异常、LoRA 参数没保存”这类问题，先从注册树、`state_dict`、`requires_grad` 和 optimizer 参数组查起。

## 3.28 章末练习

练习 1：写一个 `ResidualMLP`，包含两层 Linear、GELU、Dropout 和 residual connection，要求所有层都在 `__init__` 中定义。

练习 2：写一个 `Stack`，用 `ModuleList` 堆叠 4 个 `ResidualMLP`，并打印所有参数名，确认每层参数都被注册。

练习 3：给模型注册一个固定 buffer `scale`，在 forward 中使用它，然后调用 `model.to(device)`，确认 buffer 跟着移动。

练习 4：保存模型 `state_dict`，重新构造同结构模型并加载，检查同一输入下输出是否一致。注意推理时使用 `eval()` 和 `inference_mode()`。

练习 5：故意把子模块放进普通 list，观察 `named_parameters()` 和 `state_dict()` 的变化，再改成 `ModuleList`。

练习 6：在训练过程中保存 `best_state = model.state_dict()`，继续修改参数，再比较它与 `model.state_dict()` 的值。随后用 `detach().cpu().clone()` 建立独立快照，解释两种结果为什么不同。

练习 7：设计一个“继续训练” checkpoint，列出模型、优化器、调度器、AMP scaler、随机数、数据进度和配置版本分别解决什么问题，并说明哪些字段只对审计复现有帮助。

练习 8：构造一个 embedding 与 lm_head 共享权重的微型模型，检查参数数量、state_dict key 和加载后绑定关系。再把它改成两份独立权重，比较参数量和更新行为。

练习 9：给一个自定义模块加 forward hook，记录一次激活后移除 hook；再故意不移除并运行两次，观察回调次数和状态污染。说明为什么 hook 不应承担核心业务逻辑。

## 3.29 资料与证据边界

本章关于模块注册、参数、buffer、模式切换、状态字典和编译包装的 API 语义，以 PyTorch 2.15 官方资料为基准：

1. [`nn.Module`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html)、[`nn.Parameter`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parameter.Parameter.html) 和 [`register_buffer`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.register_buffer)
2. [`ModuleList`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleList.html)、[`ModuleDict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ModuleDict.html)、[`ParameterList`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ParameterList.html) 和 [`ParameterDict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.ParameterDict.html)
3. [保存与加载模型教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) 与 [`torch.load`](https://docs.pytorch.org/docs/2.15/generated/torch.load.html)
4. [序列化说明](https://docs.pytorch.org/docs/2.15/notes/serialization.html) 与 [`load_state_dict`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict)
5. [`Dropout`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Dropout.html) 和 [`BatchNorm1d`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BatchNorm1d.html)
6. [`DistributedDataParallel`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.parallel.DistributedDataParallel.html)
7. [`torch.compile`](https://docs.pytorch.org/docs/2.15/torch.compiler.html) 与 [torch.compile troubleshooting](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)

官方资料能确认版本化接口语义，但不能替代目标 checkpoint、硬件和分布式拓扑上的验证。共享权重、FSDP state dict、量化状态、LoRA adapter 合并和 `torch.compile` 的性能及兼容性，应以对应版本的实现和项目实测为准。`weights_only=True` 也不是对不可信 checkpoint 的完整安全保证；不要对不可信来源直接执行 pickle 加载。

## 3.30 本章总结

`nn.Module` 是 PyTorch 工程化的中心。它不是简单的“模型基类”，而是参数、子模块、buffer、状态切换、保存加载和分布式工具之间的连接点。

本章需要记住几条主线：

1. 可训练权重必须是注册的 `Parameter`，或者属于已注册子模块。
2. 子模块要用属性赋值、`ModuleList`、`ModuleDict` 等方式注册，不能藏在普通 list 或 dict 中。
3. 不训练但属于模型状态的 tensor 应该注册为 buffer。
4. `state_dict` 保存参数和持久化 buffer，是推荐的模型保存加载方式。
5. `train()` 和 `eval()` 改变模块行为，但不负责关闭梯度。
6. `model.to(device)` 只会递归处理注册过的参数和 buffer。
7. optimizer 只会更新它拿到的参数，参数注册和 optimizer 创建时机都很重要。
8. 大模型代码应按 embedding、block、attention、MLP、norm、head 等模块清晰组织。

如果你能解释清楚 `nn.Module` 的注册机制、`Parameter` 和 buffer 的区别、`state_dict` 的内容、`train/eval` 的作用，以及常见加载报错的排查路径，就已经具备了阅读和维护大多数 PyTorch 模型代码的基础。
