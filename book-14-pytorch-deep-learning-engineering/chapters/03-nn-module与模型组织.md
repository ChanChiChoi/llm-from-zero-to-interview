# 第三章：nn.Module 与模型组织

`nn.Module` 是 PyTorch 组织模型的核心抽象。第一章讲的是 tensor，第二章讲的是 autograd；到了真实工程里，我们不会把所有参数、矩阵乘法和激活函数都散写在训练脚本中，而是把它们封装成一个个 module，再组合成完整模型。

很多 PyTorch 初学者会把 `nn.Module` 理解成“写神经网络时必须继承的类”，但这只是表层。`nn.Module` 真正重要的地方在于：它负责注册参数、注册子模块、管理 buffer、递归切换训练/推理模式、递归移动 device、导出和加载 `state_dict`，并为优化器、分布式训练、混合精度和模型保存提供统一接口。

如果你不理解这些机制，很容易遇到这类问题：明明定义了参数但 optimizer 没有更新、保存的 checkpoint 里缺字段、`model.eval()` 后结果仍然随机、加载权重时报 missing key、把 tensor 放进 list 里导致参数没有注册、推理时 batch norm 或 dropout 行为不对、LoRA 参数没有进入 optimizer、DDP 包装后 key 多了 `module.` 前缀。

本章目标是把 `nn.Module` 在工程中的规则讲清楚：如何写 `forward`，什么是 `Parameter`，子模块如何注册，`state_dict` 保存什么，`train()` 和 `eval()` 到底影响什么，模型保存加载应该怎么做，以及如何组织一个可维护的大模型代码结构。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已核对 PyTorch 官方 `nn.Module`、`nn.Parameter`、`register_buffer`、`state_dict`、`load_state_dict`、`ModuleList`、`ModuleDict`、`ParameterList`、`ParameterDict`、`train()` / `eval()`、模型保存加载教程和 `torch.compile` 文档口径。

本章聚焦大模型工程最常用的 Module 组织问题：参数注册、子模块注册、buffer、`state_dict`、保存加载、训练/推理模式、device / dtype 迁移、参数冻结、参数分组、hook、DDP / FSDP 依赖的模块边界，以及最小可运行的 Module 审计 demo。

本章不展开 C++ dispatcher、底层 autograd engine、完整分布式训练实现、FSDP wrap policy 细节、`torch.export` / AOTAutograd 编译器内部机制或生产级 checkpoint shard 格式。这些内容会分别放到分布式训练、profiling、推理部署和 AI Infra 相关章节中讲。

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

这里 `self.linear = nn.Linear(10, 1)` 不只是普通赋值。因为 `LinearModel` 继承了 `nn.Module`，PyTorch 会把 `linear` 识别为子模块并注册起来。后续 `model.parameters()`、`model.state_dict()`、`model.to(device)`、`model.train()` 都会递归作用到它。

面试回答：

```text
nn.Module 是 PyTorch 中组织模型的核心抽象。它不仅封装 forward，还会自动注册 Parameter、子模块和 buffer，并支持递归遍历参数、移动 device、切换 train/eval、保存和加载 state_dict。optimizer、DDP、AMP、checkpoint 基本都依赖 Module 提供的统一结构。
```

## 3.1.1 关键机制公式与 Module 调试速查

`nn.Module` 本身不是一个数学层，而是一套工程组织规则。面试和 debug 时，可以把它拆成几条稳定口径。

第一，参数量来自注册参数，而不是来自 Python 变量名。线性层参数量为：

$$
N = d_{in} d_{out} + d_{out}
$$

如果 `bias=False`，则只有：

$$
N = d_{in} d_{out}
$$

Embedding 表的参数量为：

$$
N_{emb} = V d
$$

其中 `V` 是词表大小，`d` 是 hidden size。一个两层 MLP 的参数量常写成：

$$
N_{mlp} = d_{in} d_h + d_h + d_h d_{out} + d_{out}
$$

第二，optimizer 更新的是它拿到的 `Parameter`。单个参数的 SGD 更新可以写成：

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

$$
g_t = \frac{\partial L}{\partial \theta_t}
$$

如果某个 tensor 没有注册成 `Parameter`，或者某个子模块藏在普通 list / dict 中，它即使参与 forward，也不会自动出现在 `model.parameters()` 里，optimizer 通常不会更新它。

第三，`state_dict` 不是保存整个 Python 对象，而是保存注册权重和持久化 buffer：

$$
S(M) = P(M) \cup B_p(M)
$$

其中 `P(M)` 表示 module 树上的注册参数，`B_p(M)` 表示 `persistent=True` 的 buffer。普通属性、临时 tensor、`persistent=False` 的 buffer、`forward` 代码和 Python 类定义都不在模型 `state_dict` 里。

第四，参数高效微调或冻结模型时，先看可训练参数比例：

$$
r = \frac{\sum_{p \in P_{train}} |p|}{\sum_{p \in P_{all}} |p|}
$$

其中 `|p|` 是一个参数 tensor 的元素个数。LoRA、adapter、prompt tuning 的核心工程检查就是：应该训练的参数是否都被注册、`requires_grad=True`、进入 optimizer，并且保存时能从 `state_dict` 或 `named_parameters()` 中筛出来。

第五，`train()` / `eval()` 改的是模块状态，不是梯度开关：

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

`eval()` 递归设置 `training=False`，影响 Dropout、BatchNorm 等层；`no_grad()` / `inference_mode()` 关闭 autograd 记录。验证和推理通常两者都要用。

最小 Module 审计清单：

1. `list(model.named_parameters())` 中是否出现目标层参数。
2. `list(model.named_buffers())` 中是否出现固定 mask、统计量或位置编码。
3. `model.state_dict().keys()` 是否包含该保存的权重，是否排除了临时缓存。
4. `sum(p.numel() for p in model.parameters() if p.requires_grad)` 是否等于 optimizer 参数组里的元素数。
5. `model.train()` / `model.eval()` 是否递归影响所有子模块。
6. 保存、加载后同一输入在 eval 模式下输出是否一致。
7. `strict=False` 加载时的 missing / unexpected keys 是否符合预期，而不是被忽略。

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

## 3.3 forward 和 __call__ 的关系

使用模型时，我们通常写：

```python
output = model(x)
```

而不是：

```python
output = model.forward(x)
```

虽然最终会执行 `forward`，但两者不完全等价。`model(x)` 会调用 `nn.Module.__call__`，它在内部处理 hook、autograd 相关状态、编译包装以及其他框架机制，然后再进入 `forward`。

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

面试回答：

```text
PyTorch 里应该调用 model(x)，而不是 model.forward(x)。model(x) 会走 Module 的 __call__ 逻辑，里面会处理 hook、状态和框架扩展，然后调用 forward。直接调用 forward 可能绕过这些机制，工程上不推荐。
```

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

检查参数：

```python
layer = ScaleLayer(4)

for name, param in layer.named_parameters():
    print(name, param.shape, param.requires_grad)
```

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

这个 `scale` 虽然 `requires_grad=True`，但它不是 `Parameter`，不会出现在 `model.parameters()` 中，optimizer 通常也不会更新它。

可以验证：

```python
bad = BadScaleLayer(4)
print(list(bad.named_parameters()))  # []
```

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

`self.norm` 和 `self.mlp` 都会被注册。`self.mlp` 里面的 `Linear` 也会被递归注册。

可以查看模块树：

```python
block = Block(768)

for name, module in block.named_modules():
    print(name, type(module).__name__)
```

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

适合简单串联结构。缺点是不适合复杂控制流、多输入多输出或 residual connection。

`nn.ModuleList` 表示一组已注册模块，但不规定 forward 如何执行。

```python
self.layers = nn.ModuleList([
    Block(hidden_size) for _ in range(num_layers)
])
```

适合 Transformer block 堆叠，因为 forward 通常要加入 cache、mask、gradient checkpointing、输出 hidden states 等逻辑。

`nn.ModuleDict` 表示按字符串 key 管理子模块。

```python
self.adapters = nn.ModuleDict({
    "domain_a": Adapter(hidden_size),
    "domain_b": Adapter(hidden_size),
})
```

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

如果直接用普通 list：

```python
self.weights = [nn.Parameter(torch.ones(())) for _ in range(num_inputs)]
```

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

buffer 的特点：

1. 不会出现在 `model.parameters()` 中。
2. 默认会出现在 `state_dict()` 中。
3. 会随 `model.to(device)` 移动。
4. 可以通过 `persistent=False` 设置为不保存。

例如：

```python
self.register_buffer("causal_mask", mask, persistent=False)
```

`persistent=False` 适合可重新生成的大缓存，避免 checkpoint 过大。

查看 buffer：

```python
for name, buf in model.named_buffers():
    print(name, buf.shape, buf.device)
```

面试回答：

```text
Parameter 是需要被优化器更新的模型权重；buffer 是不训练但属于模型状态的 tensor，比如 BatchNorm running mean 或固定 mask。buffer 不会出现在 parameters() 里，但会随 model.to(device) 移动，并且默认保存在 state_dict 中。
```

## 3.9 parameters、named_parameters 和参数分组

`model.parameters()` 返回模型中所有注册参数，通常直接传给 optimizer：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

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

为什么 LayerNorm 和 bias 常常不做 weight decay？直觉上，weight decay 更适合约束大矩阵权重，而归一化层的 scale、bias 和偏置参数承担的是平移缩放作用，强行衰减可能影响训练稳定性。不同项目会有不同策略，但面试中要能说明这是参数分组问题。

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

注意：修改 `requires_grad` 后，最好重新创建 optimizer，避免 optimizer 里仍然持有不该更新的参数组。

## 3.10 state_dict 保存什么

`state_dict()` 是 PyTorch 保存模型权重的标准方式。

```python
state = model.state_dict()

for key, value in state.items():
    print(key, value.shape)
```

它保存的是：

1. 所有注册参数。
2. 默认持久化的 buffer。

它不保存：

1. Python 类定义本身。
2. `forward` 代码逻辑。
3. optimizer 状态，除非单独保存 optimizer 的 `state_dict()`。
4. 未注册在 module 上的普通 tensor。
5. `persistent=False` 的 buffer。

以一个简单模型为例：

```python
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.LayerNorm(8),
    nn.Linear(8, 2),
)

print(model.state_dict().keys())
```

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

## 3.11 保存和加载模型

推荐保存方式是保存 `state_dict`，而不是直接保存整个模型对象。

保存：

```python
torch.save(model.state_dict(), "model.pt")
```

加载：

```python
model = MyModel(config)
state = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state)
```

为什么不推荐直接保存整个模型？

```python
torch.save(model, "model.pt")
```

这种方式依赖 Python pickle，需要加载时能找到完全一致的类路径和代码结构。工程迭代后很容易因为类名、文件路径或依赖变化加载失败。保存 `state_dict` 更稳定，也更符合跨版本和跨工程迁移习惯。

完整训练 checkpoint 通常包含：

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": step,
    "epoch": epoch,
    "config": config,
}

torch.save(checkpoint, "checkpoint.pt")
```

恢复：

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
step = checkpoint["step"]
```

推理只需要模型权重时，通常不需要 optimizer 和 scheduler。

## 3.12 strict=True 和 strict=False

`load_state_dict` 默认 `strict=True`，表示 checkpoint 中的 key 必须和当前模型完全匹配。

```python
model.load_state_dict(state, strict=True)
```

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

但不要把 `strict=False` 当成万能修复。正确做法是检查 missing 和 unexpected 是否符合预期。例如加载预训练 backbone 到下游分类模型时，分类头 missing 是合理的；但如果 Transformer block 大量 missing，说明 key 对不上或模型结构不一致。

处理 `module.` 前缀：

```python
new_state = {}
for key, value in state.items():
    if key.startswith("module."):
        key = key[len("module."):]
    new_state[key] = value

model.load_state_dict(new_state)
```

更好的习惯是在 DDP 训练时保存未包装模型或保存 `model.module.state_dict()`。

## 3.13 train() 和 eval() 到底改变什么

`model.train()` 和 `model.eval()` 用于切换模块的训练/推理模式。

```python
model.train()
```

等价于：

```python
model.train(True)
```

推理模式：

```python
model.eval()
```

等价于：

```python
model.train(False)
```

它们会递归设置每个子模块的 `training` 标志：

```python
print(model.training)
```

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

BatchNorm 在训练时使用当前 batch 的统计量并更新 running statistics，在 eval 时使用保存的 running statistics。

重要区别：`model.eval()` 不等于关闭梯度。

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

推理时通常两者都需要：

1. `model.eval()`：改变 dropout、batch norm 等模块行为。
2. `torch.no_grad()` 或 `torch.inference_mode()`：关闭 autograd，节省显存和计算开销。

面试回答：

```text
model.train() 和 model.eval() 改的是 Module 的 training 状态，会递归影响 Dropout、BatchNorm 这类依赖训练模式的层。它们不会自动关闭 autograd，所以推理时还要配合 torch.no_grad() 或 inference_mode()。
```

## 3.14 Module.to、device 和 dtype 管理

`model.to(device)` 会递归移动模型参数和 buffer。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

输入数据也必须在同一个 device：

```python
x = x.to(device)
output = model(x)
```

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

`model.to(dtype=torch.bfloat16)` 可以转换浮点参数和 buffer 的 dtype：

```python
model = model.to(dtype=torch.bfloat16)
```

注意：整数类型 tensor 不会被转换成浮点；token id 通常仍然是 `torch.long`。

## 3.15 apply、children、modules 和递归遍历

`nn.Module` 提供了多种遍历方式。

直接子模块：

```python
for child in model.children():
    print(type(child).__name__)
```

所有递归子模块，包括自身：

```python
for module in model.modules():
    print(type(module).__name__)
```

带名称遍历：

```python
for name, module in model.named_modules():
    print(name, type(module).__name__)
```

这些接口常用于：

1. 自定义初始化。
2. 替换某些层。
3. 给特定模块加 hook。
4. 收集特定模块的参数。
5. 打印模型结构。

自定义初始化示例：

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


model.apply(init_weights)
```

`apply` 会递归访问所有子模块，并对每个模块执行函数。

替换层时要小心，不要一边遍历一边直接修改复杂层级导致遗漏。简单场景可以这样：

```python
for name, module in model.named_children():
    if isinstance(module, nn.ReLU):
        setattr(model, name, nn.GELU())
```

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

初始化要注意：

1. 不要在每次 forward 时初始化。
2. 初始化通常放在模型构造结束后。
3. 加载预训练权重后不要再次随机初始化覆盖权重。
4. 特定结构可能需要特殊初始化，例如 residual 分支缩放。

如果要做权重绑定，例如语言模型里 embedding 和 LM head 共享权重：

```python
self.lm_head.weight = self.embed_tokens.weight
```

这表示两个模块引用同一个 `Parameter`。保存、加载、优化器更新时要理解它们是共享权重，而不是两份独立参数。

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

也可以使用 dataclass：

```python
from dataclasses import dataclass


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
```

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

hook 使用注意：

1. 用完要 `remove()`，否则可能影响后续实验。
2. 保存激活时通常要 `detach()`，否则可能保留计算图导致显存增长。
3. hook 会增加调试复杂度，不适合作为核心业务逻辑长期依赖。
4. 分布式和编译场景下 hook 行为可能更复杂。

面试回答：

```text
Module hook 可以在 forward 或 backward 过程中插入回调，常用于抓中间激活、排查 NaN 或统计层输出。使用时要注意 remove handle，并且保存激活时最好 detach，避免无意保留计算图造成显存泄漏。
```

## 3.20 register_module、setattr 和动态添加层

有时模型结构需要根据配置动态构建。只要用 `setattr` 把 module 赋给 `nn.Module` 属性，也会注册。

```python
class DynamicModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        for i in range(num_layers):
            setattr(self, f"layer_{i}", nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        for i in range(3):
            layer = getattr(self, f"layer_{i}")
            x = layer(x)
        return x
```

这能工作，但不如 `ModuleList` 清晰，因为 forward 中还要知道层数。更好的写法：

```python
self.layers = nn.ModuleList([
    nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
])
```

动态添加层要注意：如果 optimizer 已经创建，再向模型里添加新参数，optimizer 默认不知道这些新参数。

```python
model.new_head = nn.Linear(768, 2)
```

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
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = torch.rand_like(x) > self.p
        return x * mask / (1 - self.p)
```

如果只是推理忘了关梯度，输出通常不会因此随机变化，但会浪费显存：

```python
model.eval()
with torch.inference_mode():
    output = model(x)
```

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
new_weight = model.embed_tokens.weight.data

num = min(old_weight.size(0), new_weight.size(0))
new_weight[:num].copy_(old_weight[:num])
```

这类操作要明确知道语义，不能随便忽略。

## 3.24 LoRA、Adapter 与可训练参数筛选

在参数高效微调中，经常冻结 base model，只训练 LoRA 或 adapter 参数。

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

检查可训练参数：

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.numel())
```

面试中常见追问是：LoRA 参数为什么能被 optimizer 看到？答案仍然回到 `nn.Module` 注册机制：LoRA 层里的 A、B 矩阵必须是注册的 `nn.Parameter`，LoRA 子模块必须挂在模型模块树上，否则 optimizer、state_dict、DDP 都不会自动处理它们。

保存 LoRA 时也常常只保存可训练参数：

```python
lora_state = {
    name: param.detach().cpu()
    for name, param in model.named_parameters()
    if param.requires_grad
}

torch.save(lora_state, "lora.pt")
```

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

DDP 会遍历模型参数，注册梯度同步 hook。没有注册到 module 树上的参数不会被同步。

保存时常用：

```python
if rank == 0:
    torch.save(model.module.state_dict(), "model.pt")
```

因为 DDP 包装后的外层对象有 `.module` 指向原始模型。

FSDP 更依赖模块边界，因为它会按模块切分、扁平化、分片参数。模型组织得越清晰，越容易指定 wrap policy，例如按 Transformer block 包装。

工程建议：

1. 不要把整个大模型写在一个巨大 `forward` 函数里。
2. Transformer block、attention、MLP、norm 分成清晰子模块。
3. 参数创建在 `__init__` 中完成。
4. 避免未注册参数和临时层。
5. 保存时明确是 DDP 包装前还是包装后的 `state_dict`。

## 3.26 torch.compile 对 Module 的影响

`torch.compile` 可以编译模型以获得性能提升：

```python
model = torch.compile(model)
```

它通常包装 module 的调用过程，而不是改变你写 `forward` 的基本规则。为了更容易被编译和优化，模型代码应尽量：

1. 避免在 `forward` 中创建新的可训练层。
2. 避免过多依赖 Python 副作用。
3. 避免每次 forward 动态改变模块结构。
4. 对 shape 变化和控制流保持可预期。

不是所有模型都适合直接 compile。遇到 graph break、编译时间过长、显存变化或数值不一致时，需要逐步缩小范围，例如只 compile 某个 block 或推理路径。

对面试来说，关键不是背 compile 参数，而是知道：`nn.Module` 仍然是模型结构入口，`forward` 的 Python 写法会影响编译器是否能稳定捕获计算图。

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

with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "tiny_classifier.pt"
    torch.save(model.state_dict(), path)

    clone = TinyClassifier(input_size, hidden_size, num_classes, num_blocks=2)
    clone.load_state_dict(torch.load(path, map_location="cpu"))

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
    "optimizer_matches_trainable": num_model_trainable == num_optimizer_params,
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

## 3.28 面试高频问题

问题 1：`nn.Module` 和 `nn.Parameter` 的关系是什么？

参考回答：

```text
nn.Module 是模型容器，负责注册参数、子模块和 buffer。nn.Parameter 是特殊 Tensor，被赋值为 Module 属性时会自动注册为参数，出现在 model.parameters() 和 state_dict() 中。optimizer 通常通过 model.parameters() 拿到这些 Parameter 并更新。
```

问题 2：为什么子模块不能放普通 list？

参考回答：

```text
普通 Python list 里的 Module 不会被 nn.Module 自动注册，因此 parameters()、state_dict()、to(device)、train/eval、DDP 都无法递归处理它们。应该使用 ModuleList 或 ModuleDict 来保存子模块。
```

问题 3：`model.eval()` 会关闭梯度吗？

参考回答：

```text
不会。model.eval() 只切换模块的 training 标志，影响 Dropout、BatchNorm 等层的行为。关闭梯度需要使用 torch.no_grad() 或 torch.inference_mode()。推理时通常两者都要用。
```

问题 4：`state_dict` 里保存哪些内容？

参考回答：

```text
state_dict 保存模型注册的 Parameter 和持久化 buffer，不保存 Python 类定义、forward 代码和 optimizer 状态。optimizer 状态需要单独保存 optimizer.state_dict()。
```

问题 5：加载 checkpoint 时 missing key 和 unexpected key 怎么处理？

参考回答：

```text
missing key 表示当前模型需要的参数 checkpoint 没有，unexpected key 表示 checkpoint 有但当前模型不用。要先比较模型和 checkpoint 的 key，检查是否结构变化、名称变化、DDP 的 module. 前缀或只加载部分模块。strict=False 可以用于有意部分加载，但必须确认缺失和多余的 key 符合预期。
```

问题 6：Parameter 和 buffer 的区别是什么？

参考回答：

```text
Parameter 是可训练权重，通常会被 optimizer 更新；buffer 是不训练但属于模型状态的 Tensor，比如 BatchNorm running statistics 或固定 mask。buffer 会随 model.to(device) 移动，并默认保存在 state_dict 中，但不会出现在 parameters() 里。
```

问题 7：为什么创建 optimizer 后再添加新层可能有问题？

参考回答：

```text
optimizer 在创建时拿到的是当时的参数列表。之后如果给 model 新增一层，新层参数虽然可能注册到 Module 上，但 optimizer 的 param_groups 不会自动包含它们。需要重新创建 optimizer 或手动 add_param_group。
```

## 3.29 本章小练习

练习 1：写一个 `ResidualMLP`，包含两层 Linear、GELU、Dropout 和 residual connection，要求所有层都在 `__init__` 中定义。

练习 2：写一个 `Stack`，用 `ModuleList` 堆叠 4 个 `ResidualMLP`，并打印所有参数名，确认每层参数都被注册。

练习 3：给模型注册一个固定 buffer `scale`，在 forward 中使用它，然后调用 `model.to(device)`，确认 buffer 跟着移动。

练习 4：保存模型 `state_dict`，重新构造同结构模型并加载，检查同一输入下输出是否一致。注意推理时使用 `eval()` 和 `inference_mode()`。

练习 5：故意把子模块放进普通 list，观察 `named_parameters()` 和 `state_dict()` 的变化，再改成 `ModuleList`。

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
