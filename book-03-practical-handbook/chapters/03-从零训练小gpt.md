# 第三部分：从零训练小 GPT

## 第 13 讲：准备字符级语言模型数据集

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解字符级语言模型的数据格式。
2. 从原始文本构建字符词表。
3. 实现字符到 id、id 到字符的编码解码。
4. 划分训练集和验证集。
5. 构造 GPT 训练所需的输入 `x` 和标签 `y`。
6. 写出一个可直接用于训练的小型数据加载器。

从这一讲开始，我们进入第三册第三部分：从零训练小 GPT。

前两部分我们已经实现了 PyTorch 基础和 Transformer 组件。

现在要把这些组件变成一个完整训练项目。

第一步不是写模型，而是准备数据。

很多初学者训练不起来，不是模型写错，而是数据格式、输入标签错位、batch 构造或验证集划分出了问题。

所以本讲先把字符级语言模型的数据管线讲清楚。

本讲精修时按 `WRITING_PLAN.md` 核对了 PyTorch 官方 `nn.Embedding`、`torch.randint`、`torch.stack`、`torch.utils.data.Dataset` 和 `DataLoader` 文档资料边界，确认 embedding 输入应是整数 id，`torch.randint` 的上界是右开区间，`Dataset.__len__` 应返回可采样样本数，`DataLoader` 会按 batch 维堆叠样本。

---

### 一、什么是字符级语言模型

字符级语言模型把文本拆成一个个字符。

例如：

```text
hello
```

会被拆成：

```text
h, e, l, l, o
```

如果语料是中文：

```text
我爱机器学习
```

会被拆成：

```text
我, 爱, 机, 器, 学, 习
```

字符级模型的优点是实现简单。

它不需要 BPE、SentencePiece 或 WordPiece tokenizer。

只要构造一个字符表，就能把文本变成整数序列。

缺点也明显：

```text
序列更长，语义粒度更细，训练效率和效果都不如成熟 tokenizer。
```

但作为从零训练小 GPT 的第一步，字符级语言模型非常适合教学。

---

### 二、语言模型训练数据长什么样

语言模型的目标是预测下一个 token。

更形式化地说，给定 token 序列 `t_0,t_1,\ldots,t_{N-1}`，训练目标是让模型在每个位置预测下一个 token：

```math
p(t_0,\ldots,t_{N-1})
=
\prod_{i=0}^{N-1}p(t_i\mid t_0,\ldots,t_{i-1})
```

假设文本是：

```text
hello
```

字符序列是：

```text
h e l l o
```

如果 block size 是 4，那么输入和标签可以是：

```text
x: h e l l
y: e l l o
```

也就是说：

```text
y 是 x 向右移动一位。
```

模型在每个位置都预测下一个字符：

```text
看到 h，预测 e
看到 h e，预测 l
看到 h e l，预测 l
看到 h e l l，预测 o
```

在实现里，字符会先变成 id。

例如：

```text
h -> 3
e -> 2
l -> 4
o -> 5
```

则：

```text
x: [3, 2, 4, 4]
y: [2, 4, 4, 5]
```

---

### 三、准备一段小语料

真实项目中可以使用小说、诗歌、代码、百科或领域文档。

本讲先用一个小文本演示完整流程。

```python
text = """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
"""
```

为了训练更稳定，通常需要更大的语料。

但数据管线的逻辑和小语料完全一样。

---

### 四、构建字符词表

先取出语料中出现过的所有字符：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)

print("chars:", chars)
print("vocab_size:", vocab_size)
```

`set(text)` 会去重。

`sorted` 让词表顺序固定，方便复现实验。

如果不排序，每次运行可能得到不同 id 映射。

然后构造两个字典：

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

其中：

```text
stoi: string to integer
itos: integer to string
```

也就是：

```text
字符 -> id
id -> 字符
```

---

### 五、实现 encode 和 decode

编码函数：

```python
def encode(s):
    return [stoi[ch] for ch in s]
```

解码函数：

```python
def decode(ids):
    return "".join([itos[i] for i in ids])
```

测试：

```python
sample = "hello"
ids = encode(sample)
recovered = decode(ids)

print("sample:", sample)
print("ids:", ids)
print("recovered:", recovered)
```

期望看到：

```text
recovered: hello
```

这一步很重要。

训练前必须确认：

```text
decode(encode(text)) == text
```

如果编码解码不一致，后面的训练和生成都会出问题。

---

### 六、把全文编码成 tensor

```python
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape)
print(data[:20])
```

语言模型训练时，输入必须是整数 id。

embedding 层会把这些 id 映射成向量。

所以 `data` 的 dtype 应该是：

```text
torch.long
```

不要用 float。

`nn.Embedding` 接收的是整数索引。

---

### 七、划分训练集和验证集

通常按比例划分：

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("train length:", len(train_data))
print("val length:", len(val_data))
```

为什么不随机打散每个字符？

因为语言模型依赖连续上下文。

如果把字符全部随机打散，文本顺序就被破坏了。

所以常见做法是：

```text
在长文本维度上切分前 90% 作为训练，后 10% 作为验证。
```

当然，如果你的语料由很多独立文档组成，也可以按文档划分 train/val。

更严谨的做法是：

```text
验证集文档不要出现在训练集中。
```

---

### 八、构造一个训练样本

设定上下文长度：

```python
block_size = 8
```

从某个位置 `i` 开始取一段：

```python
x = train_data[i:i + block_size]
y = train_data[i + 1:i + block_size + 1]
```

注意：

```text
x 长度是 block_size。
y 长度也是 block_size。
y 比 x 向右移动一位。
```

如果起点是 `i`、上下文长度是 `T`，可以写成：

```math
x_j = d_{i+j},\qquad y_j=d_{i+j+1},\qquad j=0,\ldots,T-1
```

示例：

```python
i = 0
x = train_data[i:i + block_size]
y = train_data[i + 1:i + block_size + 1]

print("x ids:", x.tolist())
print("y ids:", y.tolist())
print("x text:", decode(x.tolist()))
print("y text:", decode(y.tolist()))
```

你应该看到 `y text` 是 `x text` 的下一个字符版本。

---

### 九、构造 batch

训练时不会一次只喂一个样本。

我们需要随机采样多个起点，组成 batch。

```python
def get_batch(split, batch_size=4, block_size=8):
    source = train_data if split == "train" else val_data

    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")

    ix = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    return x, y
```

调用：

```python
xb, yb = get_batch("train", batch_size=4, block_size=8)

print("xb shape:", xb.shape)
print("yb shape:", yb.shape)
print(xb)
print(yb)
```

输出 shape 应该是：

```text
xb shape: torch.Size([4, 8])
yb shape: torch.Size([4, 8])
```

这就是 GPT 训练最常见的数据形状：

```text
input_ids: [batch, seq_len]
labels:    [batch, seq_len]
```

模型 forward 后 logits 通常是：

```text
logits: [batch, seq_len, vocab_size]
```

然后和 labels 计算交叉熵。

---

### 十、完整可运行数据脚本

下面是一份完整脚本，可以直接运行。

```python
import torch


torch.manual_seed(42)

text = """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size=4, block_size=8):
    source = train_data if split == "train" else val_data
    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    print("vocab_size:", vocab_size)
    print("data_dtype:", data.dtype)
    print("train_len:", len(train_data))
    print("val_len:", len(val_data))
    print("chars:", chars)

    sample = "hello"
    ids = encode(sample)
    print("sample:", sample)
    print("ids:", ids)
    print("decode:", decode(ids))
    print("roundtrip_ok:", decode(encode(text)) == text)

    xb, yb = get_batch("train", batch_size=4, block_size=8)
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("shift_ok:", torch.equal(xb[:, 1:], yb[:, :-1]))
    print("first x:", repr(decode(xb[0].tolist())))
    print("first y:", repr(decode(yb[0].tolist())))
```

参考输出：

```text
vocab_size: 22
data_dtype: torch.int64
train_len: 121
val_len: 14
chars: ['\n', ' ', 'a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x']
sample: hello
ids: [8, 5, 11, 11, 14]
decode: hello
roundtrip_ok: True
xb shape: torch.Size([4, 8])
yb shape: torch.Size([4, 8])
shift_ok: True
first x: 'model\ntr'
first y: 'odel\ntra'
```

这份脚本可以作为后续训练小 GPT 的数据准备部分。

---

### 十一、封装成 Dataset 类

上面的 `get_batch` 很适合教学。

但如果你想更接近 PyTorch 标准流程，可以写成 `Dataset`。

```python
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
```

使用：

```python
from torch.utils.data import DataLoader

train_dataset = CharDataset(train_data, block_size=8)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)
```

两种方式都可以。

小 GPT 教学项目中，经常使用 `get_batch`。

真实训练项目中，更常使用 `Dataset + DataLoader` 或更复杂的数据管线。

---

### 十二、为什么 y 要右移一位

这是语言模型训练中最关键的细节。

模型输入：

```text
x = [t0, t1, t2, t3]
```

标签：

```text
y = [t1, t2, t3, t4]
```

模型第 0 个位置输出用来预测 `t1`。

第 1 个位置输出用来预测 `t2`。

第 2 个位置输出用来预测 `t3`。

第 3 个位置输出用来预测 `t4`。

这叫 next-token prediction。

如果你错误地设置：

```python
y = x
```

模型就变成预测当前 token 本身。

这不是自回归语言模型训练目标。

---

### 十三、数据准备常见工程坑

#### 坑 1：忘记把 dtype 设为 long

Embedding 输入必须是整数 id。

应该使用：

```python
torch.tensor(ids, dtype=torch.long)
```

#### 坑 2：`x` 和 `y` 没有错位

语言模型标签必须是输入右移一位。

#### 坑 3：采样上界写错

如果写成：

```python
torch.randint(0, len(source), ...)
```

后面切片可能长度不足。

正确上界要留出 `block_size + 1`。

因为 `torch.randint(low, high, ...)` 的 `high` 不会被取到，所以常用写法是：

```python
max_start = len(source) - block_size
ix = torch.randint(0, max_start, (batch_size,))
```

这样最大起点是 `len(source) - block_size - 1`，刚好还能取到长度为 `block_size` 的 `x` 和右移一位的 `y`。

#### 坑 4：验证集太短

小语料按 9:1 切分后，验证集可能短于 `block_size + 1`。

这时 `get_batch("val")` 会报错。

真实训练时要保证 train/val 都足够长。

#### 坑 5：字符级词表无法处理未见字符

如果推理时输入了训练语料中没出现过的字符，`stoi[ch]` 会 KeyError。

教学项目可以先忽略。

工程项目要加入 `<unk>` 或使用成熟 tokenizer。

#### 坑 6：把 train 和 val 混在一起

验证集应该用于估计泛化能力。

不要在训练时从验证集采样。

---

### 十四、面试怎么讲这部分

如果面试官问“你从零训练小 GPT，数据怎么准备”，可以这样回答：

```text
我会先准备纯文本语料，构建字符级或 BPE 级词表，把文本编码成 token id 序列。然后按文档或连续区间划分 train/val。训练样本中输入 x 是长度为 block_size 的 token 片段，标签 y 是 x 向右移动一位，用于 next-token prediction。batch 的 shape 是 [B, T]，模型输出 logits 是 [B, T, vocab_size]，再和 y 计算交叉熵。
```

如果追问“为什么标签要右移”，可以回答：

```text
因为自回归语言模型的目标是根据当前位置及之前的上下文预测下一个 token。输入第 t 个位置对应的标签应该是第 t+1 个 token，所以 y 要比 x 向右移动一位。
```

如果追问“字符级 tokenizer 有什么优缺点”，可以回答：

```text
字符级 tokenizer 实现简单，不需要复杂分词，适合教学和小实验；缺点是序列会更长，语义粒度太细，训练效率和效果通常不如 BPE、SentencePiece 等子词 tokenizer。
```

---

### 十五、小练习

#### 练习 1

换一段中文文本，构建字符级词表，打印 `vocab_size`。

#### 练习 2

验证 `decode(encode(text)) == text`。

#### 练习 3

把 `block_size` 改成 16，观察 `xb` 和 `yb` 的 shape。

#### 练习 4

故意把 `y = x`，思考这会导致模型学到什么错误目标。

#### 练习 5

用 `Dataset + DataLoader` 重写 `get_batch` 版本。

---

### 本讲总结

这一讲完成了字符级语言模型的数据准备。

核心结论如下：

1. 字符级语言模型把文本拆成字符，并为每个字符分配 id。
2. `stoi` 负责字符到 id，`itos` 负责 id 到字符。
3. `encode` 和 `decode` 必须互相一致。
4. 语言模型训练样本中，`y` 是 `x` 右移一位。
5. batch 输入 shape 是 `[B, T]`，标签 shape 也是 `[B, T]`。
6. 模型输出 logits 通常是 `[B, T, vocab_size]`。
7. train/val 要严格分开，验证集不能参与训练采样。

下一讲，我们使用这套数据管线，训练一个字符级 GPT。

## 第 14 讲：训练一个字符级 GPT

### 本讲目标

学完本讲，你应该能做到六件事：

1. 使用上一讲的数据管线训练一个字符级 GPT。
2. 从零实现最小可运行 GPT 模型。
3. 理解 token embedding、position embedding、Transformer Block、lm head 的连接方式。
4. 写出完整训练循环，包括 forward、loss、backward、optimizer step。
5. 实现简单文本生成函数。
6. 能解释训练 loss、生成效果和常见训练问题。

上一讲我们完成了字符级语言模型数据准备。

本讲开始真正训练一个小 GPT。

这个模型不会很强。

但它具备 GPT 的核心结构：

```text
token embedding
position embedding
causal self-attention
Transformer blocks
language modeling head
next-token prediction loss
autoregressive generation
```

这比只看论文或框架调用更重要。

因为你会亲手跑通一次从文本到生成的完整闭环。

本讲精修时按 `WRITING_PLAN.md` 核对了 PyTorch 官方 `nn.Embedding`、`nn.Linear`、`nn.ModuleList`、`nn.Sequential`、`F.cross_entropy`、`torch.optim.AdamW` 和 `Module.train/eval` 文档资料边界，确认 embedding 输入是整数 id，`F.cross_entropy` 可接收 class index target，常见训练写法需要把 `[B,T,V]` logits 展平成 `[B*T,V]`、把 `[B,T]` labels 展平成 `[B*T]`，`AdamW` 是解耦 weight decay 优化器，`model.eval()` / `model.train()` 会切换 dropout 等模块行为。

---

### 一、整体训练流程

字符级 GPT 的训练流程如下：

```text
1. 准备文本语料。
2. 构建字符词表。
3. 把文本编码成 token id。
4. 构造 x/y batch，y 是 x 右移一位。
5. 把 x 输入 GPT，得到 logits。
6. 用 logits 和 y 计算交叉熵。
7. 反向传播并更新参数。
8. 定期生成文本观察效果。
```

模型输入：

```text
x: [B, T]
```

模型输出：

```text
logits: [B, T, vocab_size]
```

标签：

```text
y: [B, T]
```

训练目标：

```text
每个位置预测下一个字符。
```

如果 `V` 是词表大小，模型输出 logits 为 `z_{b,t,c}`，标签为 `y_{b,t}`，next-token 交叉熵可以写成：

```math
L=
\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\left(
-z_{b,t,y_{b,t}}+
\log\sum_{c=1}^{V}\exp(z_{b,t,c})
\right)
```

---

### 二、准备数据代码

先复用上一讲的数据准备逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


text = """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
attention mixes information across tokens
small gpt learns to generate characters
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

定义训练超参数：

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
block_size = 32
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4

d_model = 64
num_heads = 4
num_layers = 2
dropout = 0.1
```

小语料上这些参数已经够演示。

如果没有 GPU，也可以把 `max_iters` 调小。

---

### 三、get_batch

```python
def get_batch(split):
    source = train_data if split == "train" else val_data
    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

`x` 和 `y` 的 shape 都是：

```text
[batch_size, block_size]
```

这里 `y` 比 `x` 右移一位。

这是 next-token prediction 的核心。

---

### 四、实现一个 Attention Head

为了教学清楚，我们先实现单个 attention head。

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)
        mask = self.tril[:seq_len, :seq_len]
        wei = wei.masked_fill(~mask, torch.finfo(wei.dtype).min)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out
```

输入输出：

```text
输入 x:  [B, T, D]
输出 out: [B, T, head_size]
```

这就是一个 causal self-attention head。

---

### 五、实现 Multi-Head Attention

多个 head 并行，然后拼接。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
```

如果：

```text
d_model = 64
num_heads = 4
head_size = 16
```

则 4 个 head 拼接后仍是：

```text
[B, T, 64]
```

---

### 六、实现 Feed Forward

```python
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

FFN 不改变 shape：

```text
[B, T, D] -> [B, T, D]
```

---

### 七、实现 Transformer Block

这里使用 Pre-LN 结构。

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = d_model // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

这个 block 和上一部分讲过的 Transformer Block 是一致的。

区别只是这里写成更紧凑的训练脚本版本。

---

### 八、实现 GPTLanguageModel

完整模型：

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size_ = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size_)
            targets_flat = targets.reshape(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
```

关键 shape：

```text
idx:       [B, T]
token_emb: [B, T, D]
pos_emb:   [T, D]
x:         [B, T, D]
logits:    [B, T, vocab_size]
targets:   [B, T]
```

为什么 loss 前要 flatten？

因为 `F.cross_entropy` 通常接收：

```text
input:  [N, C]
target: [N]
```

所以把：

```text
[B, T, vocab_size] -> [B*T, vocab_size]
[B, T]             -> [B*T]
```

这里用 `reshape` 而不是强依赖连续内存的 `view`，可以减少因为前面张量布局变化带来的误用风险。

---

### 九、实现 generate

自回归生成逻辑：

```text
1. 输入当前 token 序列。
2. 模型预测每个位置的下一个 token 分布。
3. 取最后一个位置的 logits。
4. softmax 得到概率。
5. 采样一个 token。
6. 拼到序列末尾。
7. 重复。
```

代码：

```python
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

注意：

```text
idx_cond = idx[:, -block_size:]
```

这是因为 position embedding 只支持最大长度 `block_size`。

生成长文本时，每次只取最近 `block_size` 个 token 作为上下文。

---

### 十、完整模型类代码

把 `generate` 放回类中：

```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, seq_len, vocab_size_ = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size_)
            targets_flat = targets.reshape(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

---

### 十一、估计训练集和验证集 loss

训练时不要只看一个 batch 的 loss。

可以定期估计平均 loss。

```python
@torch.no_grad()
def estimate_loss(model, eval_iters=20):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

`model.eval()` 会关闭 dropout。

`model.train()` 会恢复训练模式。

这一步很重要。

否则评估 loss 会受 dropout 随机性影响。

---

### 十二、训练循环

```python
torch.manual_seed(42)

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {iter}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

这就是最小训练闭环。

每一步做四件事：

```text
1. 取 batch。
2. forward 计算 loss。
3. backward 计算梯度。
4. optimizer step 更新参数。
```

---

### 十三、生成文本

训练结束后，从一个起始 token 生成文本。

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))
```

这里 `0` 对应哪个字符，取决于你的词表。

更可控的写法是从指定字符串开始：

```python
prompt = "hello"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))
```

注意：

如果 prompt 中包含训练词表没有的字符，`encode(prompt)` 会报错。

---

### 十四、完整可运行训练脚本

下面是一份可以直接复制运行的短脚本。

它不是追求效果最强，而是验证从数据、模型、loss、训练到生成的完整闭环。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)

device = "cpu"
batch_size = 8
block_size = 16
max_iters = 40
eval_iters = 5
learning_rate = 1e-3
d_model = 32
num_heads = 4
num_layers = 2
dropout = 0.0

text = """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
attention mixes information across tokens
small gpt learns to generate characters
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.size(-1))
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = d_model // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[Block() for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        if seq_len > block_size:
            raise ValueError("sequence length exceeds block_size")

        token_emb = self.token_embedding_table(idx)
        pos = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = targets.reshape(batch_size * seq_len)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        if was_training:
            self.train()
        return idx


@torch.no_grad()
def estimate_loss(model):
    was_training = model.training
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    if was_training:
        model.train()
    return out


model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

xb, yb = get_batch("train")
logits, loss = model(xb, yb)
initial_losses = estimate_loss(model)

for step in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_losses = estimate_loss(model)
context = torch.tensor([encode("hello")], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=40)

print("vocab_size=", vocab_size)
print("batch_shape=", tuple(xb.shape))
print("logits_shape=", tuple(logits.shape))
print("initial_train_loss=", round(initial_losses["train"], 4))
print("final_train_loss=", round(final_losses["train"], 4))
print("initial_val_loss=", round(initial_losses["val"], 4))
print("final_val_loss=", round(final_losses["val"], 4))
print("param_count=", sum(p.numel() for p in model.parameters()))
print("generated=", repr(decode(generated[0].tolist())))
```

参考输出：

```text
vocab_size= 22
batch_shape= (8, 16)
logits_shape= (8, 16, 22)
initial_train_loss= 3.281
final_train_loss= 2.5838
initial_val_loss= 3.1977
final_val_loss= 2.8169
param_count= 27222
generated= 'hello rfwngeootn ioterehial\nephifr mouordnmrt'
```

先不要急着拆很多文件。

对小白来说，第一版最好放在一个脚本里跑通。

等你确认训练和生成都正常，再拆成：

```text
data.py
model.py
train.py
generate.py
```

---

### 十五、训练时你应该观察什么

#### 1. loss 是否下降

随机初始化时，初始 loss 大约接近：

```math
L_0 \approx \log V
```

其中 `V` 是 vocab size。

如果 `vocab_size = 32`，则：

```math
\log 32 \approx 3.47
```

训练后 loss 应该逐渐下降。

#### 2. train loss 和 val loss 的差距

如果 train loss 持续下降，但 val loss 上升，说明过拟合。

小语料非常容易过拟合。

#### 3. 生成文本是否从乱码变得更像语料

训练早期生成内容通常接近随机字符。

训练一段时间后，可能开始出现语料中的单词、空格和换行模式。

---

### 十六、常见工程坑

#### 坑 1：targets 没有 flatten

`F.cross_entropy` 需要 logits 是 `[N, C]`，targets 是 `[N]`。

所以要把 `[B, T, V]` 和 `[B, T]` 展平。

#### 坑 2：position embedding 越界

如果输入序列长度大于 `block_size`，position id 会超过 embedding 范围。

生成时要使用：

```python
idx_cond = idx[:, -block_size:]
```

#### 坑 3：mask 方向写反

causal mask 应该禁止看未来。

`torch.tril` 是下三角，通常是正确方向。

#### 坑 4：忘记 `model.train()` 和 `model.eval()` 切换

评估时要关闭 dropout。

评估结束后要恢复训练模式。

#### 坑 5：小语料验证集太短

如果 `val_data` 长度小于 `block_size + 1`，`get_batch("val")` 会报错。

可以增大语料或减小 `block_size`。

#### 坑 6：生成时没有 `torch.no_grad()`

生成不需要梯度。

否则会浪费显存和计算。

---

### 十七、面试怎么讲这个项目

如果面试官问“你怎么从零训练一个小 GPT”，可以这样回答：

```text
我先准备纯文本语料，构建字符级词表，把文本编码成 token id。训练样本中 x 是连续 token 片段，y 是右移一位的标签。模型结构包括 token embedding、position embedding、多层 causal Transformer block、最终 LayerNorm 和 lm head。forward 后得到 [B, T, vocab_size] logits，将 logits 和 labels 展平后计算交叉熵，用 AdamW 训练。生成时自回归地取最后一个位置 logits，softmax 后采样下一个 token，再拼回上下文。
```

如果追问“为什么需要 causal mask”，可以回答：

```text
因为自回归语言模型在第 t 个位置只能利用 t 及之前的 token 预测 t+1，不能看到未来 token。causal mask 会把未来位置的 attention logits 置为负无穷，softmax 后权重为 0。
```

如果追问“loss 怎么算”，可以回答：

```text
模型输出 logits shape 是 [B, T, V]，标签是 [B, T]。计算交叉熵前把 logits reshape 成 [B*T, V]，把 labels reshape 成 [B*T]，相当于对 batch 中每个位置都做 next-token classification。
```

---

### 十八、小练习

#### 练习 1

把 `num_layers` 从 2 改成 4，观察 loss 下降速度和过拟合情况。

#### 练习 2

把 `block_size` 从 32 改成 8，观察生成文本是否更短视。

#### 练习 3

把 dropout 设为 0，观察 train loss 和 val loss 差距。

#### 练习 4

把 `torch.multinomial` 改成 `argmax`，观察生成文本是否更重复。

#### 练习 5

用一段中文文本训练字符级 GPT，观察生成结果。

---

### 本讲总结

这一讲训练了一个字符级 GPT。

核心结论如下：

1. GPT 训练目标是 next-token prediction。
2. 输入 `x` 和标签 `y` 都是 `[B, T]`，其中 `y` 是 `x` 右移一位。
3. 模型输出 logits 是 `[B, T, vocab_size]`。
4. 计算 loss 前要把 logits 展平成 `[B*T, vocab_size]`。
5. causal self-attention 保证模型不能看未来。
6. 训练循环包括 forward、loss、backward、optimizer step。
7. 生成阶段是自回归采样，每次只使用最后一个位置预测下一个 token。

下一讲，我们专门实现 Top-k、Top-p 和 Temperature Sampling，让生成结果更可控。

## 第 15 讲：实现 Top-k、Top-p 和 Temperature Sampling

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解生成阶段为什么不能只用训练 loss 衡量效果。
2. 从零实现 temperature sampling。
3. 从零实现 top-k sampling。
4. 从零实现 top-p sampling。
5. 把采样策略接入上一讲的 `generate` 函数。
6. 能解释不同采样参数对生成文本的影响。

上一讲我们训练了一个字符级 GPT。

生成时使用的是最简单的做法：

```python
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

这能工作，但不够可控。

实际大模型生成时，常见参数包括：

```text
temperature
top_k
top_p
```

本讲专门实现这些采样策略。

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 PyTorch 官方 `F.softmax`、`torch.multinomial`、`torch.topk`、`torch.sort`、`torch.cumsum` 文档，以及 Holtzman 等人的 nucleus sampling 论文。这里重点讲教学版实现：数学目标与常见框架接口一致，但不会覆盖 beam search、repetition penalty、presence penalty、grammar constrained decoding 等更完整的生产解码系统。

---

### 一、为什么需要采样策略

语言模型输出的是 logits。

logits 经过 softmax 后得到词表上每个 token 的概率。

例如：

```text
token A: 0.50
token B: 0.20
token C: 0.10
token D: 0.08
其他 token: 0.12
```

生成时我们要从这个分布里选下一个 token。

常见选择方式有三类。

#### 方式 1：贪心解码

每次选概率最大的 token。

```text
优点：稳定、确定。
缺点：容易重复、缺少多样性。
```

#### 方式 2：完全随机采样

按完整 softmax 概率采样。

```text
优点：有多样性。
缺点：可能采到低质量长尾 token。
```

#### 方式 3：受控采样

用 temperature、top-k、top-p 控制随机性。

```text
目标：在稳定性和多样性之间折中。
```

这就是本讲要实现的内容。

---

### 二、Temperature Sampling

temperature 控制分布的尖锐程度。

给定最后一个位置的 logits $z_1,\ldots,z_V$，temperature sampling 先用正数 $\tau$ 缩放 logits，再做 softmax：

```math
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_{j=1}^{V}\exp(z_j/\tau)}
```

其中 `V` 是词表大小，$\tau$ 就是代码里的 `temperature`。

如果：

```text
temperature < 1
```

logits 被放大，分布更尖锐，高概率 token 更容易被选中。

如果：

```text
temperature > 1
```

logits 被压平，低概率 token 更容易被采到，生成更随机。

如果：

```text
temperature -> 0
```

接近贪心解码。

---

### 三、实现 Temperature

```python
import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
```

使用：

```python
logits = apply_temperature(logits, temperature=0.8)
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

常见经验：

```text
temperature = 0.7 或 0.8：更稳，更适合问答。
temperature = 1.0：原始随机性。
temperature > 1.0：更发散，更适合创意生成，但更容易胡说。
```

---

### 四、Top-k Sampling

top-k 的思想是：

```text
只保留概率最高的 k 个 token，其余 token 禁止采样。
```

记 $S_k$ 为 logits 最大的 `k` 个 token 索引集合。过滤后的 logits 可以写成：

```math
\tilde z_i=
\begin{cases}
z_i, & i\in S_k \\
-M, & i\notin S_k
\end{cases}
```

这里 $M$ 是一个足够大的正数。工程实现中不建议手写 `-1e9` 或 `float("-inf")` 到处传递，更稳妥的做法是用当前 dtype 的最小有限值 `torch.finfo(logits.dtype).min`，再让 softmax 把被 mask 的位置变成接近 0 的概率。

例如词表有 10000 个 token，`top_k=50`。

模型每一步只会从概率最高的 50 个 token 中采样。

这样可以过滤掉大量低概率长尾 token。

---

### 五、实现 Top-k

输入 logits shape：

```text
[B, vocab_size]
```

代码：

```python
def apply_top_k(logits, top_k=None):
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k, dim=-1)

    mask_value = torch.finfo(logits.dtype).min
    filtered_logits = torch.full_like(logits, mask_value)
    filtered_logits.scatter_(dim=-1, index=indices, src=values)
    return filtered_logits
```

解释：

```text
torch.topk 找到每行最大的 k 个 logits。
torch.full_like 先把全部位置设成当前 dtype 的最小有限值。
scatter_ 再把 top-k 位置的原始 logits 放回去。
softmax 后非 top-k token 的概率接近 0。
```

注意，`torch.topk` 在遇到相同 logit 时不承诺稳定返回哪几个并列 token；这不影响采样教学逻辑，但生产系统如果强依赖 tie-breaking，需要自己定义确定性规则。

输出 shape 仍然是 `[B, vocab_size]`，只是大部分位置已经被 mask。

---

### 六、Top-p Sampling

top-p 又叫 nucleus sampling，中文常译为核采样。

它的思想是：

```text
不固定保留 k 个 token，而是保留累计概率达到 p 的最小 token 集合。
```

例如排序后概率是：

```text
0.40, 0.25, 0.15, 0.08, 0.04, ...
```

如果 `top_p=0.8`，则保留：

```text
0.40 + 0.25 + 0.15 = 0.80
```

前三个 token。

更正式地说，先把概率从大到小排序为：

```math
p_{(1)}\ge p_{(2)}\ge \cdots \ge p_{(V)}
```

再找到最小的 $m$：

```math
m=\min\left\{r:\sum_{i=1}^{r}p_{(i)}\ge p\right\}
```

top-p 候选集合就是排序后的前 $m$ 个 token。它保留的是“累计概率质量”，不是固定 token 数。

如果分布很尖锐，保留 token 少。

如果分布很平坦，保留 token 多。

所以 top-p 比 top-k 更自适应。

---

### 七、实现 Top-p

代码：

```python
def apply_top_p(logits, top_p=None):
    if top_p is None:
        return logits

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    mask_value = torch.finfo(logits.dtype).min
    sorted_logits = sorted_logits.masked_fill(sorted_mask, mask_value)

    filtered_logits = torch.full_like(logits, mask_value)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits
```

关键点是这几行：

```python
sorted_mask = cumulative_probs > top_p
sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
sorted_mask[:, 0] = False
```

为什么要右移一位？

因为第一个让累计概率超过 `top_p` 的 token 也应该保留。

例如：

```text
累计概率: 0.40, 0.65, 0.80, 0.88
top_p = 0.8
```

第三个 token 使累计概率达到 0.80。

它应该保留。

超过边界之后的 token 才应该 mask。

---

### 八、组合 Temperature、Top-k、Top-p

常见顺序：

```text
1. 取最后一个位置 logits。
2. 应用 temperature。
3. 应用 top-k。
4. 应用 top-p。
5. softmax。
6. multinomial 采样。
```

代码：

```python
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)

    if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
        raise RuntimeError("invalid probability distribution after filtering")

    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next
```

输入输出：

```text
logits:   [B, vocab_size]
idx_next: [B, 1]
```

下面是一个可以独立复制运行的最小 demo，用来同时验证 temperature、top-k 和 top-p 的效果：

```python
import torch
import torch.nn.functional as F


torch.manual_seed(7)
vocab = ["A", "B", "C", "D", "E", "F"]
logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -0.5, -1.0]])


def apply_temperature(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def apply_top_k(logits, top_k=None):
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k, dim=-1)
    mask_value = torch.finfo(logits.dtype).min
    filtered_logits = torch.full_like(logits, mask_value)
    filtered_logits.scatter_(dim=-1, index=indices, src=values)
    return filtered_logits


def apply_top_p(logits, top_p=None):
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    mask_value = torch.finfo(logits.dtype).min
    sorted_logits = sorted_logits.masked_fill(sorted_mask, mask_value)
    filtered_logits = torch.full_like(logits, mask_value)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = apply_temperature(logits, temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)

    if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
        raise RuntimeError("invalid probability distribution after filtering")

    return torch.multinomial(probs, num_samples=1)


def rounded(probs):
    return [round(x, 4) for x in probs.squeeze(0).tolist()]


def entropy(probs):
    safe_probs = probs.clamp_min(1e-12)
    return float(-(safe_probs * safe_probs.log()).sum(dim=-1).item())


def kept_tokens(filtered_logits):
    mask_value = torch.finfo(filtered_logits.dtype).min
    return [
        vocab[i]
        for i, value in enumerate(filtered_logits[0])
        if value.item() != mask_value
    ]


base_probs = F.softmax(logits, dim=-1)
cold_probs = F.softmax(apply_temperature(logits, 0.5), dim=-1)
hot_probs = F.softmax(apply_temperature(logits, 2.0), dim=-1)
topk_logits = apply_top_k(logits, top_k=3)
topp_logits = apply_top_p(logits, top_p=0.8)
topk_probs = F.softmax(topk_logits, dim=-1)
topp_probs = F.softmax(topp_logits, dim=-1)
idx_next = sample_next_token(logits, temperature=0.8, top_k=4, top_p=0.9)

print("base_probs=", rounded(base_probs))
print("cold_entropy=", round(entropy(cold_probs), 4))
print("hot_entropy=", round(entropy(hot_probs), 4))
print("topk_kept=", kept_tokens(topk_logits))
print("topk_nonzero=", int((topk_probs > 0).sum().item()))
print("topp_kept=", kept_tokens(topp_logits))
print("topp_nonzero=", int((topp_probs > 0).sum().item()))
print("sampled_shape=", tuple(idx_next.shape))
```

输出应类似：

```text
base_probs= [0.6121, 0.2252, 0.0828, 0.0502, 0.0185, 0.0112]
cold_entropy= 0.4827
hot_entropy= 1.5681
topk_kept= ['A', 'B', 'C']
topk_nonzero= 3
topp_kept= ['A', 'B']
topp_nonzero= 2
sampled_shape= (1, 1)
```

这段输出说明三件事：低 temperature 会降低熵、让分布更尖；高 temperature 会提高熵、让分布更平；`top_k=3` 只保留 3 个候选，而 `top_p=0.8` 会根据当前概率质量只保留 `A` 和 `B`。

---

### 九、改造 generate 函数

把上一讲的 `generate` 改造成可配置版本。

```python
@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = model(idx_cond)
            logits = logits[:, -1, :]

            idx_next = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    finally:
        if was_training:
            model.train()
```

也可以把这个方法放进 `GPTLanguageModel` 类里。

为了教学清楚，这里写成外部函数。

---

### 十、不同采样方式对比

假设已经训练好了模型。

```python
prompt = "hello"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
```

#### 1. 低温度生成

```python
out = generate(
    model,
    context,
    max_new_tokens=200,
    block_size=block_size,
    temperature=0.7,
)
print(decode(out[0].tolist()))
```

特点：

```text
更稳定，更保守，更容易重复。
```

#### 2. 高温度生成

```python
out = generate(
    model,
    context,
    max_new_tokens=200,
    block_size=block_size,
    temperature=1.3,
)
print(decode(out[0].tolist()))
```

特点：

```text
更多样，更随机，更容易出现奇怪字符。
```

#### 3. Top-k 生成

```python
out = generate(
    model,
    context,
    max_new_tokens=200,
    block_size=block_size,
    temperature=1.0,
    top_k=10,
)
print(decode(out[0].tolist()))
```

特点：

```text
过滤掉低概率字符，生成更稳。
```

#### 4. Top-p 生成

```python
out = generate(
    model,
    context,
    max_new_tokens=200,
    block_size=block_size,
    temperature=1.0,
    top_p=0.9,
)
print(decode(out[0].tolist()))
```

特点：

```text
根据概率分布自适应选择候选集合。
```

#### 5. 组合策略

```python
out = generate(
    model,
    context,
    max_new_tokens=200,
    block_size=block_size,
    temperature=0.8,
    top_k=20,
    top_p=0.9,
)
print(decode(out[0].tolist()))
```

实际 LLM 应用中，经常组合使用这些参数。

---

### 十一、贪心解码怎么写

有时你希望完全确定性输出。

可以使用 argmax。

```python
def greedy_next_token(logits):
    return torch.argmax(logits, dim=-1, keepdim=True)
```

贪心解码不需要 softmax。

因为 softmax 不改变 logits 的大小顺序。

贪心生成的特点是：

```text
确定、稳定、低多样性、容易重复。
```

对于开放式生成，贪心通常不是最佳选择。

对于某些结构化任务，贪心可能更合适。

---

### 十二、为什么不能 top_k 太小

如果 `top_k=1`，它等价于贪心解码。

如果 `top_k` 很小，模型只能从极少数 token 中选。

结果可能变得机械、重复。

例如字符级模型可能生成：

```text
the the the the the
```

或者不断重复某些高频字符。

所以 top-k 的作用不是越小越好。

它是一个质量和多样性的折中。

---

### 十三、为什么 top_p 常比 top_k 灵活

top-k 固定候选数量。

不管模型是否很确定，都保留 k 个。

top-p 固定累计概率阈值。

当模型很确定时，候选集合可能很小。

当模型不确定时，候选集合会变大。

例如：

```text
场景 A：概率分布很尖锐
0.90, 0.03, 0.02, ...
top_p=0.9 可能只保留 1 个 token。

场景 B：概率分布很平坦
0.10, 0.09, 0.08, ...
top_p=0.9 会保留很多 token。
```

所以 top-p 更自适应。

---

### 十四、常见工程坑

#### 坑 1：temperature 设为 0

`logits / 0` 会出错。

如果想做贪心解码，应该单独使用 `argmax`。

#### 坑 2：top-k 后没有重新 softmax

过滤 logits 后必须重新 softmax。

被设为 `-inf` 的位置概率才会变成 0。

#### 坑 3：top-p 没有保留第一个超过阈值的 token

这样可能导致候选集合累计概率不足，甚至极端情况下被 mask 空。

#### 坑 4：对 `[B, T, V]` 直接采样

生成时通常只用最后一个位置：

```python
logits = logits[:, -1, :]
```

shape 应该是 `[B, V]`。

#### 坑 5：`top_k > vocab_size`

应使用：

```python
top_k = min(top_k, logits.size(-1))
```

#### 坑 6：采样时忘记 `model.eval()`

生成阶段最好关闭 dropout。

否则同样参数下输出会额外受 dropout 影响。

---

### 十五、面试怎么讲采样策略

如果面试官问“temperature 是什么”，可以这样回答：

```text
temperature 是生成时调节 softmax 分布尖锐程度的参数。它通过 logits / temperature 实现。temperature 小于 1 会让分布更尖锐，输出更确定；大于 1 会让分布更平坦，输出更多样但更容易不稳定。
```

如果问“top-k 是什么”，可以回答：

```text
top-k sampling 每一步只保留概率最高的 k 个 token，把其他 token 的 logits 设为负无穷，然后在剩余 token 上重新 softmax 并采样。它可以过滤低概率长尾 token，提高生成质量。
```

如果问“top-p 是什么”，可以回答：

```text
top-p 又叫 nucleus sampling，它不是固定保留 k 个 token，而是按概率从高到低排序，保留累计概率达到 p 的最小 token 集合。相比 top-k，top-p 会根据模型当前的不确定性自适应调整候选集合大小。
```

如果追问“这些参数怎么组合”，可以回答：

```text
常见做法是先对最后一步 logits 应用 temperature，再做 top-k 或 top-p 截断，最后 softmax 并用 multinomial 采样。低 temperature、较小 top-k/top-p 会更保守，高 temperature、较大 top-k/top-p 会更多样但风险更高。
```

---

### 十六、小练习

#### 练习 1

写一个 logits 向量，分别用 `temperature=0.5` 和 `temperature=2.0` 观察 softmax 概率变化。

#### 练习 2

实现 `top_k=3`，确认只有 3 个 token 的概率非 0。

#### 练习 3

实现 `top_p=0.9`，打印每一步保留了多少个 token。

#### 练习 4

比较贪心解码和 multinomial 采样生成结果的重复程度。

#### 练习 5

在上一讲字符级 GPT 上分别测试：

```text
temperature=0.7, top_k=10
temperature=1.0, top_p=0.9
temperature=1.2, top_p=0.95
```

观察生成文本差异。

---

### 本讲总结

这一讲实现了 Top-k、Top-p 和 Temperature Sampling。

核心结论如下：

1. temperature 控制 softmax 分布的尖锐程度。
2. temperature 越低，输出越确定；temperature 越高，输出越随机。
3. top-k 只保留 logits 最高的 k 个 token。
4. top-p 保留累计概率达到 p 的最小 token 集合。
5. top-p 比 top-k 更自适应。
6. 生成时通常只使用最后一个位置的 logits。
7. 过滤 logits 后必须重新 softmax 再采样。
8. 贪心解码稳定但容易重复，随机采样多样但需要控制。

下一讲，我们给训练脚本加入 checkpoint、日志和验证集评估，让项目更接近真实工程。

## 第 16 讲：加入 checkpoint、日志和验证集评估

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解为什么训练脚本需要 checkpoint。
2. 保存和恢复模型、优化器、训练步数和配置。
3. 定期记录训练集 loss 和验证集 loss。
4. 实现 best checkpoint 保存逻辑。
5. 把训练日志保存为 CSV，便于后续画图分析。
6. 能把一个教学脚本升级成更接近真实项目的训练脚本。

前几讲我们已经完成了：

```text
字符级数据集
小 GPT 模型
训练循环
采样策略
```

这些已经能跑通。

但还不够工程化。

真实训练最怕三件事：

```text
训练中断后不能恢复。
只看训练 loss，不知道泛化情况。
没有日志，无法复盘实验。
```

本讲把训练脚本升级为可恢复、可观察、可对比的版本。

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 PyTorch 官方 saving/loading tutorial、`torch.save`、`torch.load`、`state_dict` / `load_state_dict`、`torch.no_grad` 和 `Module.train/eval` 资料边界。这里采用教学项目最常见的字典式 checkpoint：保存模型参数、优化器状态、训练步数、验证集最优指标、配置和词表元信息；生产训练还会额外保存随机数状态、分布式 rank 状态、学习率调度器状态、混合精度 scaler、数据迭代器进度和实验追踪信息。

---

### 一、为什么需要 checkpoint

checkpoint 是训练过程中的快照。

它通常包含：

```text
model state
optimizer state
current step
best validation loss
training config
vocab metadata
```

如果用数学符号表示，一个最小 checkpoint 可以看成：

```math
C_t=(\theta_t,s_t,t,L_{\mathrm{best}},h,v)
```

其中 $\theta_t$ 是第 `t` 步的模型参数，$s_t$ 是优化器内部状态，$L_{\mathrm{best}}$ 是目前最好的验证集 loss，$h$ 是训练配置，$v$ 是词表或 tokenizer 元信息。

如果训练中断，比如：

```text
机器重启
显存错误
任务超时
手动停止
```

有 checkpoint 就可以继续训练。

没有 checkpoint，只能从头开始。

对于大模型训练，这通常不可接受。

即使是小 GPT 项目，也应该养成保存 checkpoint 的习惯。

---

### 二、项目目录结构

建议给小 GPT 项目准备一个简单目录：

```text
mini_char_gpt/
  train.py
  checkpoints/
  logs/
  samples/
```

在脚本里创建目录：

```python
from pathlib import Path


out_dir = Path("mini_char_gpt")
ckpt_dir = out_dir / "checkpoints"
log_dir = out_dir / "logs"
sample_dir = out_dir / "samples"

ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)
sample_dir.mkdir(parents=True, exist_ok=True)
```

为什么要分目录？

```text
checkpoints 保存模型权重。
logs 保存训练指标。
samples 保存生成样例。
```

这样后续复盘实验更清楚。

---

### 三、训练配置 config

不要把所有超参数散落在脚本里。

可以集中成一个字典：

```python
config = {
    "batch_size": 16,
    "block_size": 32,
    "max_iters": 1000,
    "eval_interval": 100,
    "eval_iters": 20,
    "learning_rate": 3e-4,
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "seed": 42,
}
```

保存 checkpoint 时把 config 一起保存。

这样以后看到一个模型文件时，能知道它是怎么训练出来的。

---

### 四、验证集评估函数

上一讲已经写过 `estimate_loss`。

这里稍微工程化一下。

如果每次评估抽取 `K` 个 batch，验证集平均 loss 可以写成：

```math
L_{\mathrm{val}}(t)=
\frac{1}{K}
\sum_{k=1}^{K}
L_{\mathrm{batch}}^{(k)}(t)
```

训练集评估同理。注意这里的 train loss 是“评估模式下抽样估计的训练集 loss”，不是训练循环刚刚反传的那个单 batch loss；这样 train/val 两条曲线更可比。

```python
@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters):
    out = {}
    was_training = model.training
    model.eval()

    try:
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xb, yb = get_batch(split)
                logits, loss = model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
    finally:
        if was_training:
            model.train()

    return out
```

这里记录了 `was_training`。

如果函数调用前模型处于训练模式，评估后恢复训练模式。

这比无脑 `model.train()` 更稳。

---

### 五、保存 checkpoint

保存函数：

```python
def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    best_val_loss,
    config,
    stoi,
    itos,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }
    torch.save(checkpoint, path)
```

为什么要保存 `stoi` 和 `itos`？

因为字符 id 映射必须和训练时一致。

如果模型训练时：

```text
'a' -> 3
```

但推理时变成：

```text
'a' -> 7
```

生成结果会完全错乱。

所以 tokenizer 或词表元信息必须和模型一起保存。

---

### 六、加载 checkpoint

加载函数：

```python
def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    config = checkpoint.get("config", None)
    stoi = checkpoint.get("stoi", None)
    itos = checkpoint.get("itos", None)

    return {
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }
```

如果只是推理，可以不传 optimizer。

加载外部来源 checkpoint 时要有安全边界：PyTorch checkpoint 本质上经过 Python 反序列化流程，生产环境不要直接加载不可信文件；教学项目里显式使用 `weights_only=True`，可以减少不必要对象反序列化，但仍应只加载自己训练或可信来源的文件。

如果要恢复训练，必须加载 optimizer state。

原因是 AdamW 内部有动量和二阶矩估计。

简化写成：

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
```

```math
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
```

如果只恢复 $\theta_t$，不恢复 $m_t$ 和 $v_t$，下一步更新方向和尺度都会变。

只恢复模型参数、不恢复优化器，训练轨迹会变。

---

### 七、记录 CSV 日志

最简单可用的日志格式是 CSV。

```python
import csv


log_path = log_dir / "train_log.csv"


def init_log_file(path):
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "best_val_loss"])


def append_log(path, step, train_loss, val_loss, best_val_loss):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([step, train_loss, val_loss, best_val_loss])
```

CSV 的好处是：

```text
可以用 Excel 打开。
可以用 pandas 读取。
可以很容易画 loss 曲线。
```

后面第 17 讲会用它分析训练 loss。

---

### 八、保存生成样例

定期保存生成文本也很有用。

因为 loss 下降不一定代表生成质量好。

```python
@torch.no_grad()
def save_sample(model, step, prompt, max_new_tokens, path):
    was_training = model.training
    model.eval()

    try:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=max_new_tokens)
        text_out = decode(generated[0].tolist())

        with path.open("w", encoding="utf-8") as f:
            f.write(text_out)
    finally:
        if was_training:
            model.train()
```

如果你的 `generate` 是外部函数，也可以改成：

```python
generated = generate(model, context, max_new_tokens, block_size)
```

保存样例后，可以比较：

```text
step 0 生成什么
step 100 生成什么
step 500 生成什么
step 1000 生成什么
```

这对理解模型学习过程非常有帮助。

---

### 九、best checkpoint 逻辑

训练过程中通常保存两类 checkpoint：

```text
last.pt：最近一次 checkpoint，用于恢复训练。
best.pt：验证集 loss 最好的 checkpoint，用于最终评估或推理。
```

逻辑：

```math
t^*=
\underset{t\in E}{\mathrm{argmin}}\ L_{\mathrm{val}}(t)
```

```math
L_{\mathrm{best}}=
\min_{t\in E}L_{\mathrm{val}}(t)
```

其中 `E` 是已经执行过评估的 step 集合。也就是说，`best.pt` 应该由验证集 loss 决定，而不是由训练集 loss 或最近一次 loss 决定。

```python
best_val_loss = float("inf")

if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_checkpoint(
        ckpt_dir / "best.pt",
        model,
        optimizer,
        step,
        best_val_loss,
        config,
        stoi,
        itos,
    )
```

每次评估也可以保存 `last.pt`：

```python
save_checkpoint(
    ckpt_dir / "last.pt",
    model,
    optimizer,
    step,
    best_val_loss,
    config,
    stoi,
    itos,
)
```

这样即使最新模型不是最佳模型，也能恢复最近训练状态。

---

### 十、完整训练循环改造

把评估、日志、checkpoint 放入训练循环：

```python
torch.manual_seed(config["seed"])

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

init_log_file(log_path)
best_val_loss = float("inf")
start_step = 0

for step in range(start_step, config["max_iters"]):
    if step % config["eval_interval"] == 0:
        losses = estimate_loss(model, get_batch, config["eval_iters"])
        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"step {step}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, "
            f"best val {best_val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                config,
                stoi,
                itos,
            )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            step,
            best_val_loss,
            config,
            stoi,
            itos,
        )

        append_log(log_path, step, train_loss, val_loss, best_val_loss)

        sample_path = sample_dir / f"sample_step_{step}.txt"
        save_sample(
            model,
            step,
            prompt="hello",
            max_new_tokens=200,
            path=sample_path,
        )

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

这个循环已经具备真实训练脚本的基本能力。

下面是一份可以直接运行的最小 checkpoint demo。为了让代码短一些，它用一个小回归模型代替 GPT，但演示的工程动作完全一样：评估 train/val loss、写 CSV 日志、保存 `best.pt` 和 `last.pt`、恢复模型与优化器状态、确认词表元信息一起保存。

```python
import csv
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)

config = {
    "learning_rate": 0.05,
    "max_iters": 10,
    "eval_interval": 5,
    "eval_iters": 3,
    "seed": 0,
}

x_train = torch.linspace(-1.0, 1.0, steps=16).unsqueeze(1)
y_train = 2.0 * x_train + 1.0
x_val = torch.tensor([[-0.75], [-0.25], [0.25], [0.75]])
y_val = 2.0 * x_val + 1.0
stoi = {"x": 0, "y": 1}
itos = {0: "x", 1: "y"}


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x, y=None):
        pred = self.net(x)
        loss = None if y is None else F.mse_loss(pred, y)
        return pred, loss


def get_batch(split):
    if split == "train":
        idx = torch.randint(0, len(x_train), (4,))
        return x_train[idx], y_train[idx]
    return x_val, y_val


@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    was_training = model.training
    model.eval()
    try:
        for split in ["train", "val"]:
            losses = []
            for _ in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    finally:
        if was_training:
            model.train()
    return out


def save_checkpoint(path, model, optimizer, step, best_val_loss, config, stoi, itos):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "stoi": stoi,
        "itos": itos,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "step": checkpoint["step"],
        "best_val_loss": checkpoint["best_val_loss"],
        "config": checkpoint["config"],
        "stoi": checkpoint["stoi"],
        "itos": checkpoint["itos"],
    }


def init_log_file(path):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_loss", "val_loss", "best_val_loss"])


def append_log(path, step, train_loss, val_loss, best_val_loss):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            step,
            round(train_loss, 6),
            round(val_loss, 6),
            round(best_val_loss, 6),
        ])


with tempfile.TemporaryDirectory() as tmpdir:
    out_dir = Path(tmpdir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir()
    log_dir.mkdir()
    log_path = log_dir / "train_log.csv"
    init_log_file(log_path)

    model = TinyRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    best_val_loss = float("inf")

    for step in range(config["max_iters"] + 1):
        if step % config["eval_interval"] == 0:
            losses = estimate_loss(model, config["eval_iters"])
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    best_val_loss,
                    config,
                    stoi,
                    itos,
                )
            save_checkpoint(
                ckpt_dir / "last.pt",
                model,
                optimizer,
                step,
                best_val_loss,
                config,
                stoi,
                itos,
            )
            append_log(log_path, step, losses["train"], losses["val"], best_val_loss)

        if step == config["max_iters"]:
            break
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    resumed_model = TinyRegressor()
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=config["learning_rate"],
    )
    meta = load_checkpoint(ckpt_dir / "last.pt", resumed_model, resumed_optimizer)
    best_meta = load_checkpoint(ckpt_dir / "best.pt", TinyRegressor())
    log_rows = list(csv.DictReader(log_path.open(encoding="utf-8")))

    print("log_steps=", [int(row["step"]) for row in log_rows])
    print("last_step=", meta["step"])
    print("resume_start_step=", meta["step"] + 1)
    print("best_step=", best_meta["step"])
    print("best_val_loss=", round(best_meta["best_val_loss"], 6))
    print("vocab_roundtrip=", meta["itos"][meta["stoi"]["x"]] == "x")
    print("optimizer_state_nonempty=", len(resumed_optimizer.state_dict()["state"]) > 0)
    print("files=", sorted(path.name for path in ckpt_dir.iterdir()))
```

参考输出：

```text
log_steps= [0, 5, 10]
last_step= 10
resume_start_step= 11
best_step= 10
best_val_loss= 0.128653
vocab_roundtrip= True
optimizer_state_nonempty= True
files= ['best.pt', 'last.pt']
```

这段输出说明：日志按评估步写入，`last.pt` 能恢复到最近 step，恢复训练应从 `step+1` 开始，`best.pt` 由验证集 loss 选择，词表映射和优化器状态都确实被保存并加载回来。

---

### 十一、恢复训练

如果要从 `last.pt` 恢复训练：

```python
resume_path = ckpt_dir / "last.pt"

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

start_step = 0
best_val_loss = float("inf")

if resume_path.exists():
    meta = load_checkpoint(
        resume_path,
        model,
        optimizer=optimizer,
        map_location=device,
    )
    start_step = meta["step"] + 1
    best_val_loss = meta["best_val_loss"]
    print(f"resumed from step {start_step}, best val loss {best_val_loss:.4f}")
```

然后训练循环写成：

```python
for step in range(start_step, config["max_iters"]):
    ...
```

注意：

```text
恢复训练时要从 step + 1 开始。
```

否则可能重复训练同一个 step 的日志和 checkpoint。

---

### 十二、只加载模型做推理

如果只想加载最好的模型生成文本：

```python
model = GPTLanguageModel().to(device)
meta = load_checkpoint(
    ckpt_dir / "best.pt",
    model,
    optimizer=None,
    map_location=device,
)
model.eval()

prompt = "hello"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))
```

真实项目中，`encode` 和 `decode` 应该使用 checkpoint 中保存的 `stoi` 和 `itos`。

否则词表不一致会导致错误。

---

### 十三、日志里应该记录什么

最小日志：

```text
step
train_loss
val_loss
best_val_loss
```

更完整的日志可以包括：

```text
learning_rate
grad_norm
tokens_seen
elapsed_time
samples_per_second
```

小 GPT 项目里，先记录 loss 就够。

但面试项目如果想更像真实训练，可以加上：

```text
训练速度
参数量
显存占用
生成样例
```

这些信息能体现工程意识。

---

### 十四、验证集评估为什么重要

训练 loss 下降只能说明模型越来越拟合训练数据。

验证 loss 才能帮助判断泛化。

常见情况：

```text
train loss 下降，val loss 也下降：正常学习。
train loss 下降，val loss 上升：过拟合。
train loss 不下降，val loss 不下降：模型、数据或学习率可能有问题。
train loss 很低，生成很差：可能数据太少、采样策略差或模型只是记忆局部模式。
```

小语料训练字符级 GPT 时，过拟合很常见。

这不是坏事。

它反而是理解训练动态的好机会。

---

### 十五、常见工程坑

#### 坑 1：只保存模型参数

如果要恢复训练，还必须保存 optimizer state、step 和 best_val_loss。

#### 坑 2：没有保存词表

字符级模型的 `stoi/itos` 必须保存。

否则推理时 id 映射可能错乱。

#### 坑 3：评估时忘记 `model.eval()`

dropout 会让验证 loss 不稳定。

#### 坑 4：评估后忘记恢复 `model.train()`

如果一直处于 eval 模式，dropout 不生效，训练行为会改变。

#### 坑 5：best checkpoint 用 train loss 判断

best 模型通常应该根据 validation loss 保存。

#### 坑 6：覆盖掉所有历史结果

至少保留 `best.pt` 和 `last.pt`。

更复杂实验可以按时间戳或实验名建目录。

#### 坑 7：日志不记录 config

没有 config，就很难复现实验。

---

### 十六、面试怎么讲 checkpoint 和评估

如果面试官问“训练脚本怎么支持断点恢复”，可以这样回答：

```text
我会保存 checkpoint，其中包括 model state_dict、optimizer state_dict、当前 step、best validation loss、训练 config 和 tokenizer 或词表信息。恢复训练时先构建同样结构的模型和优化器，再加载这些状态，并从 step+1 继续训练。
```

如果问“为什么要保存 optimizer state”，可以回答：

```text
因为 AdamW 这类优化器内部有一阶和二阶动量估计。如果只恢复模型参数，不恢复优化器状态，训练虽然能继续，但优化轨迹会改变，尤其在大模型训练中会影响稳定性和可复现性。
```

如果问“best checkpoint 应该按什么保存”，可以回答：

```text
通常应该按 validation loss 保存 best checkpoint，而不是按 train loss。train loss 只表示训练集拟合程度，validation loss 更能反映泛化能力。
```

如果问“你会记录哪些日志”，可以回答：

```text
最少记录 step、train loss、val loss、best val loss。更完整的训练日志还会记录 learning rate、grad norm、tokens seen、吞吐、耗时和定期生成样例，便于复盘训练过程和定位问题。
```

---

### 十七、小练习

#### 练习 1

给上一讲训练脚本加入 `best.pt` 和 `last.pt` 保存。

#### 练习 2

中断训练后从 `last.pt` 恢复，确认 step 能正确接上。

#### 练习 3

把训练日志保存为 CSV，并用表格软件打开。

#### 练习 4

每 100 step 保存一个生成样例，对比不同 step 的生成质量。

#### 练习 5

故意只加载模型参数、不加载 optimizer state，对比恢复后的 loss 波动。

---

### 本讲总结

这一讲给小 GPT 训练脚本加入了 checkpoint、日志和验证集评估。

核心结论如下：

1. checkpoint 是训练过程快照，用于恢复训练和保存最佳模型。
2. 恢复训练时应保存并加载 model、optimizer、step、best_val_loss 和 config。
3. 字符级模型还必须保存 `stoi/itos`，保证词表一致。
4. 验证集 loss 比训练集 loss 更适合选择 best checkpoint。
5. CSV 日志简单实用，便于后续画 loss 曲线。
6. 定期保存生成样例能帮助理解模型质量变化。
7. 一个工程化训练脚本应该是可恢复、可观察、可复现的。

下一讲，我们分析训练 loss 与生成样例，学习如何判断模型是否真的学到了东西。

## 第 17 讲：分析训练 loss 与生成样例

### 本讲目标

学完本讲，你应该能做到六件事：

1. 读取训练日志并画出 train/val loss 曲线。
2. 判断模型是否正常学习、欠拟合或过拟合。
3. 理解 loss 数值和生成质量之间的关系。
4. 对比不同 step 的生成样例。
5. 根据 loss 曲线和样例定位常见训练问题。
6. 把训练分析整理成可写入项目报告和面试表达的结论。

前一讲我们把小 GPT 训练脚本升级成了工程化版本：

```text
checkpoint
CSV 日志
验证集评估
生成样例保存
```

这些信息保存下来以后，下一步就是分析它们。

训练一个模型不是只看最后 loss 多低。

更重要的是回答：

```text
模型有没有学到东西？
有没有过拟合？
生成质量是否真的变好？
如果训练异常，问题可能在哪里？
```

本讲就围绕这些问题展开。

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 PyTorch 官方 `CrossEntropyLoss`、`Module.train/eval`、`torch.no_grad`、`clip_grad_norm_` 文档，以及 pandas `read_csv` 和 Matplotlib `savefig` 文档。这里的分析脚本仍定位为教学项目：重点是读懂 loss 曲线、best checkpoint 和生成样例，不引入 TensorBoard、W&B 或复杂实验追踪系统。

---

### 一、为什么要分析训练曲线

训练 loss 是模型在训练集上的平均预测错误。

验证 loss 是模型在未参与训练的数据上的平均预测错误。

设第 `t` 次评估时模型参数为 `theta_t`，单个样本的 next-token 负对数似然为 `ell_i(theta_t)`，则可以把训练集和验证集 loss 写成：

```math
L_{\mathrm{train}}(t)=\frac{1}{N_{\mathrm{train}}}\sum_{i=1}^{N_{\mathrm{train}}}\ell_i(\theta_t)
```

```math
L_{\mathrm{val}}(t)=\frac{1}{N_{\mathrm{val}}}\sum_{i=1}^{N_{\mathrm{val}}}\ell_i(\theta_t)
```

这里 `N_train` 和 `N_val` 分别是训练集和验证集用于评估的 token 或样本数量。直觉上，train loss 看模型能否拟合训练分布，val loss 看这种拟合是否能迁移到未参与训练的数据。

二者一起看，才能判断训练状态。

只看 train loss 有风险。

因为模型可能只是记住训练文本。

只看 val loss 也不够。

因为你还需要知道模型是否正在有效拟合训练数据。

一个常用的辅助量是 train/val gap：

```math
G(t)=L_{\mathrm{val}}(t)-L_{\mathrm{train}}(t)
```

`G(t)` 越来越大，通常说明模型对训练集越来越熟，但对验证集没有同步变好。选择 best checkpoint 时，通常不看最后一步，而看验证 loss 最低的 step：

```math
t_{\mathrm{best}}=\arg\min_{t\in\mathcal{T}} L_{\mathrm{val}}(t)
```

其中 `\mathcal{T}` 是所有做过验证集评估的 step 集合。

常见判断：

```text
train loss 下降，val loss 下降：正常学习。
train loss 下降，val loss 上升：过拟合。
train loss 和 val loss 都很高：欠拟合或训练没跑起来。
train loss 剧烈震荡：学习率、batch、数据或实现可能有问题。
```

---

### 二、读取 CSV 日志

上一讲保存的日志格式是：

```text
step,train_loss,val_loss,best_val_loss
0,3.50,3.48,3.48
100,2.80,2.95,2.95
200,2.20,2.60,2.60
...
```

用 pandas 读取：

```python
import pandas as pd


log_path = "mini_char_gpt/logs/train_log.csv"
df = pd.read_csv(log_path)

print(df.head())
print(df.tail())
```

如果不想引入 pandas，也可以用标准库 `csv`。

但为了画图和分析，pandas 更方便。

---

### 三、画 loss 曲线

```python
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["step"], df["train_loss"], label="train loss")
ax.plot(df["step"], df["val_loss"], label="val loss")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("Training and Validation Loss")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("mini_char_gpt/logs/loss_curve.png", dpi=150)
plt.close(fig)
```

如果是在服务器上没有图形界面，可以只保存图片：

```python
plt.savefig("mini_char_gpt/logs/loss_curve.png", dpi=150)
```

然后下载或打开图片查看。

---

### 四、初始 loss 应该是多少

随机初始化模型在一开始接近均匀猜测。

如果词表大小为 `V`，均匀分布下每个 token 的概率是 `1/V`。此时单 token 交叉熵为：

```math
L_0=-\log\frac{1}{V}=\log V
```

这就是为什么初始 loss 往往接近 `log V`。这个判断只适合 logits 近似均匀、没有严重初始化异常、label 构造正确的情况。

例如：

```python
import math

V = 32
print(math.log(V))
```

输出约为：

```text
3.47
```

所以如果你的词表大小是 32，初始 loss 在 3.4 附近是正常的。

如果初始 loss 极端异常，比如：

```text
几十、几百、NaN
```

就要检查：

```text
logits shape
labels 范围是否在 [0, V)
cross entropy 输入
学习率
mask
数据 dtype
```

---

### 五、典型曲线 1：正常学习

正常学习时，曲线通常是：

```text
train loss 逐步下降
val loss 逐步下降
二者差距不大
```

这说明模型正在学习语料中的统计规律，并且验证集上也有改善。

小 GPT 项目中，正常曲线可能类似：

```text
step 0:    train 3.50, val 3.48
step 100:  train 2.90, val 3.00
step 300:  train 2.30, val 2.60
step 600:  train 1.90, val 2.35
step 1000: train 1.60, val 2.20
```

生成样例也应该从乱码逐渐变得像训练语料。

---

### 六、典型曲线 2：过拟合

过拟合时，常见现象是：

```text
train loss 持续下降
val loss 下降一段后开始上升
```

这说明模型越来越记住训练集，但对验证集泛化变差。

小语料训练字符级 GPT 非常容易过拟合。

可能曲线：

```text
step 0:    train 3.50, val 3.49
step 300:  train 2.00, val 2.40
step 600:  train 1.20, val 2.60
step 1000: train 0.60, val 3.10
```

应对方法：

```text
增大数据集。
减小模型规模。
增大 dropout。
减少训练步数。
使用 best checkpoint，而不是 last checkpoint。
```

注意：

```text
过拟合不是训练失败，它是模型容量相对数据过大的自然现象。
```

关键是你要能识别它。

---

### 七、典型曲线 3：欠拟合

欠拟合时，常见现象是：

```text
train loss 和 val loss 都高
下降很慢
二者差距不大
```

可能原因：

```text
模型太小。
训练步数太少。
学习率太低。
上下文长度太短。
数据模式太复杂。
```

应对方法：

```text
增大 d_model。
增加 num_layers。
训练更久。
适当提高学习率。
增大 block_size。
```

但要注意，增大模型后也更容易过拟合。

所以要结合 train/val gap 判断。

---

### 八、典型曲线 4：loss 震荡或发散

如果 loss 剧烈震荡，甚至变成 NaN，常见原因是：

```text
学习率过大。
梯度爆炸。
mask 造成全 -inf 行。
输入标签越界。
数据里有异常。
混合精度数值不稳定。
```

小 GPT 项目中，最常见的是学习率过大。

可以尝试：

```text
把 learning_rate 从 3e-4 降到 1e-4。
加入 gradient clipping。
检查 labels 的最大值是否小于 vocab_size。
检查 loss 是否从第一步就 NaN。
```

gradient clipping 示例：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 九、分析生成样例

loss 是数字，生成样例是行为。

二者都要看。

假设你保存了：

```text
samples/sample_step_0.txt
samples/sample_step_100.txt
samples/sample_step_500.txt
samples/sample_step_1000.txt
```

可以写脚本对比：

```python
from pathlib import Path


sample_dir = Path("mini_char_gpt/samples")

for path in sorted(sample_dir.glob("sample_step_*.txt")):
    print("=" * 80)
    print(path.name)
    print(path.read_text(encoding="utf-8")[:500])
```

观察重点：

```text
是否学会空格和换行。
是否出现训练语料中的高频单词。
是否能生成局部合理片段。
是否大量重复。
是否出现完全随机字符。
是否只是在背诵训练集。
```

---

### 十、不同阶段的生成表现

#### step 0

模型随机初始化。

生成通常是乱码。

```text
qzae  lmx\ntrrpp  oo...
```

#### 训练早期

模型开始学会字符频率。

可能出现空格、换行和少量常见字母组合。

```text
he lo  th  lae to...
```

#### 训练中期

模型开始生成类似语料的短词和局部模式。

```text
hello langua model the text...
```

#### 训练后期

如果数据很小，模型可能开始背诵训练语料。

```text
hello transformer
hello large language model
...
```

这在小语料字符级模型中很常见。

---

### 十一、loss 降低但生成很差怎么办

可能原因包括：

```text
训练步数还不够。
采样参数太随机。
模型太小。
数据太少或太杂。
上下文长度太短。
prompt 不在训练分布内。
```

排查顺序：

```text
1. 先用低 temperature，例如 0.7。
2. 加 top_k 或 top_p。
3. 用训练语料中的 prompt 开始生成。
4. 检查 train/val loss 是否合理。
5. 检查 decode 是否正确。
6. 确认模型处于 eval 模式，并在生成时使用 `torch.no_grad()`。
```

生成差不一定说明训练错。

小模型、小数据、字符级 tokenizer 本身就会限制生成质量。

---

### 十二、生成重复怎么办

重复是语言模型生成中的常见问题。

例如：

```text
hello hello hello hello hello
```

可能原因：

```text
贪心解码或 temperature 太低。
top_k 太小。
训练数据重复度高。
模型过拟合。
上下文中已有重复模式。
```

可以尝试：

```text
提高 temperature。
增大 top_k。
使用 top_p。
加入 repetition penalty。
清洗重复训练数据。
```

本实战项目先不实现 repetition penalty。

但面试中可以提到它是解决重复问题的一种常见方法。

---

### 十三、验证 labels 是否正确

如果 loss 不下降，第一件事是检查数据。

打印一个 batch：

```python
xb, yb = get_batch("train")

print("x:", decode(xb[0].tolist()))
print("y:", decode(yb[0].tolist()))
```

你应该看到：

```text
y 是 x 右移一位。
```

如果不是，训练目标就错了。

这是小 GPT 项目中最常见的 bug 之一。

---

### 十四、检查 logits 和 labels shape

在 forward 中临时打印：

```python
print("logits:", logits.shape)
print("targets:", targets.shape)
```

应该是：

```text
logits:  [B, T, V]
targets: [B, T]
```

计算交叉熵前：

```python
logits_flat = logits.reshape(B * T, V)
targets_flat = targets.reshape(B * T)
```

shape 应该是：

```text
logits_flat:  [B*T, V]
targets_flat: [B*T]
```

PyTorch 的 `CrossEntropyLoss` 接收的是未归一化 logits，目标 label 如果是类别索引，取值范围应该在 `[0, V)`，并且通常是整数类型。教学代码里先把 `[B, T, V]` 展平到 `[B*T, V]`，再把 `[B, T]` 展平到 `[B*T]`，是为了让每个 token 位置都变成一个独立分类样本。

如果这里错了，loss 可能无意义。

---

### 十五、一个简单分析脚本

先看一个 0 依赖、可直接运行的最小 demo。它不画图，而是演示如何从 CSV 日志中找 best step、计算 train/val gap、识别过拟合起点，并用一个很粗的重复率检查生成样例。

```python
import csv
import math
from io import StringIO


CSV_LOG = """
step,train_loss,val_loss,best_val_loss
0,3.47,3.46,3.46
100,2.80,2.90,2.90
300,1.95,2.35,2.35
600,1.10,2.60,2.35
1000,0.55,3.10,2.35
"""

SAMPLES = {
    0: "qzae lmx trrpp oo",
    300: "hello model the text",
    1000: "hello hello hello hello language model",
}


def read_log(text):
    rows = []
    for row in csv.DictReader(StringIO(text.strip())):
        rows.append(
            {
                "step": int(row["step"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
            }
        )
    return rows


def first_overfit_step(rows, min_delta=0.05):
    best_val = float("inf")
    previous_train = None
    for row in rows:
        improved = row["val_loss"] < best_val - min_delta
        if improved:
            best_val = row["val_loss"]
        train_still_falling = (
            previous_train is not None and row["train_loss"] < previous_train
        )
        if (not improved) and train_still_falling:
            return row["step"]
        previous_train = row["train_loss"]
    return None


def adjacent_repeat_ratio(text):
    words = text.split()
    if len(words) < 2:
        return 0.0
    repeats = sum(a == b for a, b in zip(words, words[1:]))
    return repeats / (len(words) - 1)


rows = read_log(CSV_LOG)
best = min(rows, key=lambda row: row["val_loss"])
last = rows[-1]
overfit_start = first_overfit_step(rows)
diagnosis = "overfitting" if overfit_start is not None else "normal_or_underfit"

print(f"expected_uniform_loss={math.log(32):.4f}")
print(f"best_step={best['step']}")
print(f"best_val_loss={best['val_loss']:.2f}")
print(f"last_gap={last['val_loss'] - last['train_loss']:.2f}")
print(f"overfit_start={overfit_start}")
print(f"diagnosis={diagnosis}")

for step, text in SAMPLES.items():
    ratio = adjacent_repeat_ratio(text)
    print(f"sample_{step}_repeat_ratio={ratio:.2f}")
```

运行后会看到类似输出：

```text
expected_uniform_loss=3.4657
best_step=300
best_val_loss=2.35
last_gap=2.55
overfit_start=600
diagnosis=overfitting
sample_0_repeat_ratio=0.00
sample_300_repeat_ratio=0.00
sample_1000_repeat_ratio=0.60
```

这段 demo 说明三件事：

```text
初始 loss 可以和 log V 对齐，先检查训练是否在合理尺度。
best checkpoint 应该选 validation loss 最低的 step。
train loss 继续下降、val loss 反弹、重复率升高时，要警惕过拟合和背诵。
```

实际项目中，可以再加 pandas 和 Matplotlib 画曲线。可以创建：

```text
analyze_training.py
```

内容：

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


run_dir = Path("mini_char_gpt")
log_path = run_dir / "logs" / "train_log.csv"
sample_dir = run_dir / "samples"

df = pd.read_csv(log_path)

print("last rows:")
print(df.tail())

best_row = df.loc[df["val_loss"].idxmin()]
print("best step:", int(best_row["step"]))
print("best val loss:", float(best_row["val_loss"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["step"], df["train_loss"], label="train")
ax.plot(df["step"], df["val_loss"], label="val")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("Mini Char GPT Loss Curve")
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig(run_dir / "logs" / "loss_curve.png", dpi=150)
plt.close(fig)

for path in sorted(sample_dir.glob("sample_step_*.txt")):
    print("=" * 80)
    print(path.name)
    print(path.read_text(encoding="utf-8")[:500])
```

这个脚本做三件事：

```text
读取日志。
画 loss 曲线。
打印不同 step 的生成样例。
```

---

### 十六、如何写项目分析结论

训练结束后，可以写一个简短实验结论。

示例：

```text
本实验从零训练了一个字符级 GPT。训练初始 loss 接近 log V，其中 V 是词表大小，说明模型初始输出接近均匀分布。随着训练进行，train loss 和 val loss 均下降，生成样例从随机字符逐渐变成包含空格、换行和语料中高频词的文本。后期 train loss 继续下降但 val loss 开始上升，说明小语料上出现过拟合，因此最终采用 validation loss 最低的 best checkpoint 作为生成模型。
```

这类结论很适合写到项目 README 或简历项目说明里。

它体现你不仅会训练，还会分析。

---

### 十七、常见工程坑

#### 坑 1：只看最终生成样例

应该结合 loss 曲线和多个 step 的样例。

#### 坑 2：只看 train loss

train loss 低不代表泛化好。

#### 坑 3：val loss 上升还继续用 last checkpoint

应该使用 best checkpoint。

#### 坑 4：生成差就以为模型代码错了

小数据、小模型、字符级 tokenizer 都会限制效果。

要结合数据规模和模型容量判断。

#### 坑 5：loss NaN 还继续训练

一旦出现 NaN，要立即排查学习率、mask、label 范围和数值稳定性。

#### 坑 6：样例 prompt 不在词表里

字符级模型遇到未见字符会编码失败。

---

### 十八、面试怎么讲训练分析

如果面试官问“你怎么判断小 GPT 训练是否正常”，可以这样回答：

```text
我会同时看 train loss、validation loss 和定期保存的生成样例。初始 loss 应该接近 log V，其中 V 是词表大小；训练过程中 train loss 应该下降。如果 val loss 也下降，说明模型在泛化上有提升。如果 train loss 下降但 val loss 上升，说明过拟合。生成样例则用于观察模型是否从随机字符逐渐学会空格、换行、词形和局部语法模式。
```

如果追问“loss 下降但生成不好怎么办”，可以回答：

```text
我会先检查采样参数，比如降低 temperature、加入 top-k 或 top-p；然后检查 prompt 是否在训练分布内，再检查数据管线，比如 y 是否是 x 右移一位、decode 是否正确、logits 和 labels shape 是否正确。如果这些没问题，可能是模型太小、数据太少或字符级 tokenizer 表达能力有限。
```

如果问“如何识别过拟合”，可以回答：

```text
典型过拟合表现是 train loss 持续下降，但 validation loss 在某个 step 后开始上升，同时生成样例可能越来越像背诵训练语料。应对方式包括使用 best checkpoint、增加数据、减小模型、提高 dropout 或提前停止。
```

---

### 十九、小练习

#### 练习 1

读取 `train_log.csv`，画出 train loss 和 val loss 曲线。

#### 练习 2

计算 `log V`，和 step 0 的 loss 对比。

#### 练习 3

找出 validation loss 最低的 step。

#### 练习 4

对比 step 0、step 100、step 500、step 1000 的生成样例。

#### 练习 5

故意减小训练数据，观察过拟合是否更早出现。

---

### 本讲总结

这一讲分析了训练 loss 与生成样例。

核心结论如下：

1. train loss 和 val loss 要一起看。
2. 初始 loss 通常接近 `log V`。
3. train/val 同时下降通常表示正常学习。
4. train loss 下降但 val loss 上升通常表示过拟合。
5. loss 曲线能反映优化状态，生成样例能反映模型行为。
6. 生成差不一定是模型错，也可能是采样、数据、模型容量或 tokenizer 限制。
7. 一个完整项目应该包含训练曲线、生成样例和分析结论。

下一讲，我们把字符级 tokenizer 扩展到 BPE tokenizer，让小 GPT 项目更接近真实大模型训练流程。

## 第 18 讲：扩展到 BPE tokenizer

### 本讲目标

学完本讲，你应该能做到六件事：

1. 理解字符级 tokenizer 的局限。
2. 理解 BPE tokenizer 的基本思想。
3. 使用现成 tokenizer 把文本编码成 token id。
4. 将前面的小 GPT 数据管线从字符级改成 BPE 级。
5. 处理 vocab size、特殊 token、decode 和 checkpoint 元信息。
6. 能在面试中解释字符级、子词级和真实 LLM tokenizer 的差异。

前面几讲我们用字符级 tokenizer 训练了一个小 GPT。

字符级 tokenizer 的好处是简单。

但真实大模型一般不会直接用字符级 tokenizer。

它们通常使用 BPE、SentencePiece、Unigram 或类似的子词 tokenizer。

本讲把前面的项目扩展到 BPE tokenizer，让小 GPT 项目更接近真实训练流程。

资料边界说明：本讲第二轮精修时按 `WRITING_PLAN.md` 核对 Hugging Face Transformers tokenizer、GPT-2 tokenizer、`add_special_tokens` / `resize_token_embeddings` 文档，Hugging Face tokenizers 训练接口，以及 PyTorch `nn.Embedding`、`nn.Linear`、`torch.randint` 文档。这里的代码仍定位为教学项目：重点是讲清 tokenizer 切换、id 序列、vocab size、embedding/lm head shape 和 checkpoint 元信息，不把 tokenizer 训练扩展成完整工程。

---

### 一、字符级 tokenizer 的局限

字符级 tokenizer 把文本拆成字符。

例如：

```text
language model
```

会变成：

```text
l, a, n, g, u, a, g, e,  , m, o, d, e, l
```

优点：

```text
实现简单。
词表小。
不会遇到未知单词问题。
```

缺点：

```text
序列很长。
语义粒度太细。
训练效率低。
模型需要很多步才能组合出词和短语。
```

对于中文，字符级还算能表达基本字。

对于英文、代码、多语言场景，字符级效率很低。

真实 LLM 更常使用子词级 tokenizer。

---

### 二、BPE 的核心思想

BPE 全称 Byte Pair Encoding。

它的核心思想是：

```text
从小单位开始，不断合并高频相邻片段，形成更大的 token。
```

一个简化例子：

语料中经常出现：

```text
l o w
l o w e r
l o w e s t
```

BPE 可能会先合并高频 pair：

```text
l + o -> lo
lo + w -> low
```

最后 `low` 可能成为一个 token。

这样：

```text
low
```

不再需要拆成 `l, o, w` 三个字符。

序列长度变短。

模型训练更高效。

更形式化地说，BPE 每一步会统计当前 token 序列中相邻 pair 的频次，选择最高频的 pair 合并：

```math
(a^*,b^*)=\underset{(a,b)\in P}{\mathrm{argmax}}\ c(a,b)
```

这里 `P` 是当前所有相邻 token pair 的集合，`c(a,b)` 是 pair `(a,b)` 在语料中的出现次数。合并后，所有相邻的 `a,b` 会被替换成一个新 token `ab`。重复这个过程直到达到目标词表大小，或没有值得继续合并的高频 pair。

这只是 BPE 的核心直觉。真实 tokenizer 还会涉及 byte-level 预处理、normalization、pre-tokenization、special tokens、未知字符处理和训练语料选择。

---

### 三、BPE 和字符级的对比

假设文本是：

```text
transformer
```

字符级可能是：

```text
t r a n s f o r m e r
```

长度 11。

BPE 可能切成：

```text
trans former
```

或者：

```text
transform er
```

长度 2。

这带来两个直接影响：

```text
同样 block_size 下，BPE 能覆盖更长文本。
同样文本长度下，BPE 的训练和推理步数更少。
```

如果字符级长度为 `T_char`，BPE 后长度为 `T_bpe`，在标准 full attention 里，attention score 矩阵大小大致从 `T_char*T_char` 变成 `T_bpe*T_bpe`：

```math
R_{\mathrm{attn}}=
\frac{T_{\mathrm{bpe}}^2}{T_{\mathrm{char}}^2}
```

当 `T_bpe` 明显小于 `T_char` 时，attention 计算和显存压力都会下降。这个结论只比较序列长度，不代表 BPE 永远更省，因为 BPE 的词表更大，会增加 embedding、lm head 和 softmax 相关成本。

代价是：

```text
词表更大。
tokenizer 实现更复杂。
需要保存 tokenizer 文件或元信息。
```

---

### 四、选择一个现成 tokenizer

本讲不从零训练 BPE tokenizer。

因为我们的目标是训练小 GPT，而不是实现 tokenizer 算法本身。

实战中可以使用 Hugging Face 的 tokenizer。

例如 GPT-2 tokenizer：

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

如果环境无法联网，可以提前下载 tokenizer 文件，或使用本地路径：

```python
tokenizer = AutoTokenizer.from_pretrained("./tokenizers/gpt2")
```

也可以使用 `tiktoken`：

```python
import tiktoken


enc = tiktoken.get_encoding("gpt2")
```

本讲用 Hugging Face 写法讲解，因为它和后续微调实战衔接更自然。

---

### 五、用 tokenizer 编码和解码

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "hello transformer, hello language model"
ids = tokenizer.encode(text)
recovered = tokenizer.decode(ids)

print(ids)
print(recovered)
```

输出可能类似：

```text
[31373, 28247, 11, 23748, 3303, 2746]
hello transformer, hello language model
```

注意：

```text
BPE token 不等于字符，也不一定等于单词。
```

一个 token 可能是：

```text
一个词
一个子词
一个空格加词片段
一个标点
一个字节片段
```

---

### 六、vocab_size 怎么来

字符级项目里，我们自己构造：

```python
vocab_size = len(chars)
```

BPE 项目里，词表来自 tokenizer：

```python
vocab_size = tokenizer.vocab_size
```

或者更稳妥地使用：

```python
vocab_size = len(tokenizer)
```

二者有时不同。

原因是某些 tokenizer 可能额外添加 special tokens。

如果你给 tokenizer 新增了 special token，应该使用：

```python
vocab_size = len(tokenizer)
```

并确保模型 embedding 大小匹配。

为了避免符号混乱，后面统一记：

```math
V=\operatorname{len}(\mathrm{tokenizer})
```

其中 `V` 是模型可见的 token id 总数。训练数据里的所有 token id 都必须满足：

```math
0 \le d_i < V
```

如果某个输入 id 大于等于 `V`，`nn.Embedding(V, D)` 查表时就会越界。

---

### 七、改造数据准备代码

字符级版本：

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

BPE 版本：

```python
ids = tokenizer.encode(text)
data = torch.tensor(ids, dtype=torch.long)
```

给定 BPE 后的连续 token id 序列 `d_0,d_1,...,d_{N-1}`，第 `i` 个训练样本仍然是 next-token prediction：

```math
x_j=d_{i+j},\qquad y_j=d_{i+j+1},\qquad j=0,\ldots,T-1
```

这里 `T` 就是 `block_size`。换成 BPE 后，公式不变，只是 `d_i` 的含义从字符 id 变成 BPE token id。

完整示例：

```python
import torch
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = (
    """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
"""
    * 20
)

ids = tokenizer.encode(text, add_special_tokens=False)
data = torch.tensor(ids, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

vocab_size = len(tokenizer)
```

后续 `get_batch` 不需要大改。

因为它只关心 token id 序列。

```python
def get_batch(split):
    source = train_data if split == "train" else val_data
    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
```

这说明模型训练管线本质上只需要：

```text
一串 token ids。
```

字符级和 BPE 级的差别主要发生在 tokenizer 层。

---

### 八、改造 decode

字符级版本：

```python
def decode(ids):
    return "".join([itos[i] for i in ids])
```

BPE 版本：

```python
def decode(ids):
    return tokenizer.decode(ids)
```

如果 ids 是 tensor，可以先转 list：

```python
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids)
```

生成后：

```python
generated = model.generate(context, max_new_tokens=100)
print(decode(generated[0]))
```

---

### 九、处理特殊 token

GPT-2 tokenizer 默认没有单独的 padding token。

如果只是做连续文本语言模型训练，可以不需要 padding。

因为我们从长 token 序列中切固定长度 block。

但如果后续要 batch 多个不同长度文本，就可能需要 pad token。

常见做法：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

这表示用 EOS token 兼作 padding token。

注意：

```text
如果添加了新的 special token，模型 embedding size 要同步调整。
```

在从零训练模型时，直接用：

```python
vocab_size = len(tokenizer)
```

然后创建 embedding 即可。

如果是加载已有预训练模型后添加 token，则需要：

```python
model.resize_token_embeddings(len(tokenizer))
```

这是后续 Hugging Face 微调部分会遇到的问题。

---

### 十、模型结构需要改哪里

从字符级切到 BPE 级，模型结构基本不需要变。

只需要改：

```text
vocab_size
encode/decode
数据 ids
checkpoint 中保存 tokenizer 信息
```

模型里的 embedding：

```python
self.token_embedding_table = nn.Embedding(vocab_size, d_model)
```

lm head：

```python
self.lm_head = nn.Linear(d_model, vocab_size)
```

必须使用新的 `vocab_size`。

如果 hidden size 记为 `D`，词表大小记为 `V`，那么 token embedding 参数量是：

```math
N_{\mathrm{embed}}=V D
```

普通未绑权重的 lm head 参数量约为：

```math
N_{\mathrm{head}}=D V + V
```

其中最后的 `+V` 是 bias。如果使用 tied embedding，lm head 可以复用 token embedding 权重，参数量会少一块 `D V`，但输出 softmax 的类别数仍然是 `V`。

如果 GPT-2 tokenizer 的词表约 50257，而字符级词表只有几十个，那么 lm head 会大很多。

这意味着：

```text
BPE 版本参数量会显著增加。
```

尤其是小模型中，embedding 和 lm head 可能占很大比例。

---

### 十一、BPE 版本的 prompt 编码

字符级 prompt：

```python
context = torch.tensor([encode("hello")], dtype=torch.long, device=device)
```

BPE prompt：

```python
prompt = "hello"
context = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device,
)
```

如果使用 Hugging Face tokenizer 的批处理接口：

```python
encoded = tokenizer(prompt, return_tensors="pt")
context = encoded["input_ids"].to(device)
```

生成后解码：

```python
text_out = tokenizer.decode(generated[0].tolist())
```

---

### 十二、完整 BPE 数据脚本

```python
import torch
from transformers import AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16
block_size = 8

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = (
    """
hello world
hello transformer
hello large language model
transformer learns patterns from text
language model predicts the next token
attention mixes information across tokens
small gpt learns to generate text
"""
    * 20
)

ids = tokenizer.encode(text, add_special_tokens=False)
data = torch.tensor(ids, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

vocab_size = len(tokenizer)


def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids)


def get_batch(split):
    source = train_data if split == "train" else val_data
    max_start = len(source) - block_size
    if max_start <= 0:
        raise ValueError(f"{split} split is too short for block_size={block_size}")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    print("num tokens:", len(data))
    print("vocab_size:", vocab_size)
    xb, yb = get_batch("train")
    print("xb shape:", xb.shape)
    print("yb shape:", yb.shape)
    print("sample x:", decode(xb[0]))
    print("sample y:", decode(yb[0]))
```

注意：

```text
这个示例语料太小，BPE 后 token 数可能很少。
实际训练 BPE 小 GPT 时，需要比字符级更多文本。
```

因为 BPE 词表很大，小语料很难覆盖足够 token。

如果当前环境没有 `transformers`，也可以先用下面这个纯 Python toy demo 理解“从字符级切到 BPE 级”到底改了什么。它不是完整 BPE 训练器，只手动给出一组 merge 规则，用来演示 token 数缩短、id 序列构造、右移标签和词表相关参数量变化。

```python
TEXT = "low lower lowest low lower transformer transformer low"
MERGES = [
    ("l", "o"), ("lo", "w"), ("low", "e"), ("lowe", "r"),
    ("lowe", "s"), ("lowes", "t"), ("t", "r"), ("tr", "a"),
    ("tra", "n"), ("tran", "s"), ("trans", "f"), ("transf", "o"),
    ("transfo", "r"), ("transfor", "m"), ("transform", "e"),
    ("transforme", "r"),
]
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def split_initial(text):
    return [" " if ch == " " else ch for ch in text]


def apply_merges(tokens, merges):
    tokens = list(tokens)
    for left, right in merges:
        merged = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                merged.append(left + right)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged
    return tokens


def build_vocab(tokens):
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def decode(ids, id_to_token):
    pieces = []
    for idx in ids:
        token = id_to_token[idx]
        if token not in {"<pad>", "<bos>", "<eos>"}:
            pieces.append(token)
    return "".join(pieces)


def vocab_dependent_params(vocab_size, d_model, tied_embedding=False, bias=True):
    embedding_params = vocab_size * d_model
    head_params = 0 if tied_embedding else d_model * vocab_size
    if bias and not tied_embedding:
        head_params += vocab_size
    return embedding_params + head_params


char_tokens = split_initial(TEXT)
bpe_tokens = ["<bos>"] + apply_merges(char_tokens, MERGES) + ["<eos>"]
vocab = build_vocab(bpe_tokens)
ids = encode(bpe_tokens, vocab)
id_to_token = {idx: token for token, idx in vocab.items()}

block_size = 8
max_start = len(ids) - block_size
starts = [0, max_start - 1]
x = [ids[i:i + block_size] for i in starts]
y = [ids[i + 1:i + block_size + 1] for i in starts]

char_vocab_size = len(set(char_tokens)) + len(SPECIAL_TOKENS)
bpe_vocab_size = len(vocab)
d_model = 32

print("char_token_count=", len(char_tokens))
print("bpe_token_count=", len(bpe_tokens))
print("compression_ratio=", round(len(bpe_tokens) / len(char_tokens), 3))
print("bpe_tokens=", bpe_tokens)
print("ids=", ids)
print("roundtrip_ok=", decode(ids, id_to_token) == TEXT)
print("vocab_size=", bpe_vocab_size)
print("x_shape=", (len(x), block_size))
print("shift_ok=", all(row_x[1:] == row_y[:-1] for row_x, row_y in zip(x, y)))
print("char_vocab_params=", vocab_dependent_params(char_vocab_size, d_model))
print("toy_bpe_vocab_params=", vocab_dependent_params(bpe_vocab_size, d_model))
print("gpt2_vocab_params=", vocab_dependent_params(50257, d_model))
```

运行后会看到类似输出：

```text
char_token_count= 54
bpe_token_count= 17
compression_ratio= 0.315
bpe_tokens= ['<bos>', 'low', ' ', 'lower', ' ', 'lowest', ' ', 'low', ' ', 'lower', ' ', 'transformer', ' ', 'transformer', ' ', 'low', '<eos>']
ids= [1, 4, 5, 6, 5, 7, 5, 4, 5, 6, 5, 8, 5, 8, 5, 4, 2]
roundtrip_ok= True
vocab_size= 9
x_shape= (2, 8)
shift_ok= True
char_vocab_params= 1040
toy_bpe_vocab_params= 585
gpt2_vocab_params= 3266705
```

这个 demo 说明四件事：

```text
BPE 合并后 token 数可以明显少于字符数。
模型训练仍然只看到 token id 序列，x/y 仍然是右移一位。
词表大小会直接改变 embedding 和 lm head 相关参数量。
大词表 tokenizer 对小模型不一定划算，因为大量参数会花在很少出现的 token 上。
```

---

### 十三、checkpoint 里保存 tokenizer 信息

字符级时我们保存 `stoi/itos`。

BPE 时应该保存 tokenizer 名称或本地路径。

例如：

```python
config = {
    "tokenizer_name": "gpt2",
    "vocab_size": len(tokenizer),
    "block_size": 64,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
}
```

checkpoint：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": step,
    "best_val_loss": best_val_loss,
    "config": config,
}
```

如果你训练了自己的 tokenizer，应保存 tokenizer 文件：

```text
tokenizer.json
vocab.json
merges.txt
special_tokens_map.json
tokenizer_config.json
```

推理时必须加载同一个 tokenizer。

否则 token id 对不上，模型输出会失效。

---

### 十四、BPE 训练时的注意事项

#### 注意 1：BPE 词表更大

字符级 vocab 可能只有几十个。

BPE vocab 通常是几万。

这会显著增加：

```text
embedding 参数
lm_head 参数
softmax 计算量
```

#### 注意 2：小语料不适合大词表

如果语料只有几百行，却用 5 万词表，很多 token 很少出现。

模型很难学好。

教学时可以使用 GPT-2 tokenizer 演示流程。

真正训练时，应增大语料，或训练一个更小 vocab 的 tokenizer。

#### 注意 3：block_size 含义变了

字符级 `block_size=64` 表示 64 个字符。

BPE 级 `block_size=64` 表示 64 个 BPE token。

通常覆盖的文本更长。

#### 注意 4：decode 可能清理空格

BPE tokenizer 对空格、换行和特殊符号有自己的规则。

不要期待 token 和字符一一对应。

#### 注意 5：special tokens 要一致

训练、验证、生成时使用的 BOS/EOS/PAD 配置必须一致。

---

### 十五、如果想训练自己的 BPE tokenizer

可以使用 Hugging Face `tokenizers` 库。

这里只给一个简化方向，不展开成完整项目。

```python
from tokenizers import ByteLevelBPETokenizer


tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus.txt"],
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)

tokenizer.save_model("my_bpe_tokenizer")
```

然后用：

```python
from transformers import GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("my_bpe_tokenizer")
```

对于小 GPT 项目，自训练一个 8000 或 16000 vocab 的 tokenizer，通常比直接用 GPT-2 的 50257 vocab 更合适。

但这需要更多语料。

---

### 十六、字符级到 BPE 级的改造清单

从前面代码迁移到 BPE，需要改这些地方：

```text
1. 删除 chars、stoi、itos 的手动构造。
2. 使用 AutoTokenizer 加载 tokenizer。
3. 用 tokenizer.encode(text) 得到 ids。
4. 设置 vocab_size = len(tokenizer)。
5. decode 改成 tokenizer.decode。
6. prompt 编码改成 tokenizer(prompt, return_tensors="pt") 或 tokenizer.encode。
7. checkpoint 保存 tokenizer_name 或 tokenizer 文件路径。
8. 确认 train/val 数据长度大于 block_size + 1。
```

模型结构基本不变。

这说明：

```text
GPT 模型本质上不关心 token 是字符、子词还是字节，它只关心 token id 序列。
```

---

### 十七、面试怎么讲 BPE 扩展

如果面试官问“字符级 tokenizer 和 BPE 有什么区别”，可以这样回答：

```text
字符级 tokenizer 把文本拆成单个字符，实现简单但序列长、语义粒度细；BPE 从小单位出发，通过合并高频相邻片段形成子词 token，可以显著缩短序列长度，提高训练和推理效率，更接近真实 LLM 的 tokenizer 方案。
```

如果问“从字符级小 GPT 切到 BPE 要改模型吗”，可以回答：

```text
主体 Transformer 结构不需要改，因为模型只处理 token id。需要改的是 tokenizer、数据编码解码、vocab_size、embedding 和 lm_head 的输出维度，以及 checkpoint 中保存 tokenizer 信息。
```

如果追问“为什么 tokenizer 必须和 checkpoint 一起保存”，可以回答：

```text
因为模型学到的是 token id 和语义之间的关系。如果推理时 tokenizer 不一致，同一个 id 对应的文本片段可能变了，模型输出就会错乱。所以必须保存并复用训练时的 tokenizer。
```

如果问“直接用 GPT-2 tokenizer 训练小模型有什么问题”，可以回答：

```text
GPT-2 tokenizer 词表大约 5 万，对小语料和小模型来说词表过大，embedding 和 lm_head 参数量明显增加，而且很多 token 训练中很少出现。教学上可以用它演示流程，真正小模型训练更适合使用较小 vocab 的自训练 tokenizer。
```

---

### 十八、常见工程坑

#### 坑 1：忘记改 vocab_size

如果模型还是字符级 vocab size，但输入 id 来自 BPE tokenizer，会出现 id 越界。

#### 坑 2：训练和推理 tokenizer 不一致

这会导致生成结果乱码或完全无意义。

#### 坑 3：数据太短

BPE 后 token 数可能比字符数少很多。

如果 `len(data) <= block_size + 1`，无法采样 batch。

#### 坑 4：误以为 BPE token 等于单词

BPE token 是子词片段，不一定是完整词。

#### 坑 5：新增 special token 后没有同步 embedding

从零训练时用 `len(tokenizer)` 初始化模型。

加载预训练模型后新增 token 时，要 resize embedding。

#### 坑 6：小模型使用过大词表

大词表会让 embedding 和输出层占用过多参数，训练更难。

---

### 十九、小练习

#### 练习 1

用 GPT-2 tokenizer 编码一句中文和一句英文，观察 token 数量差异。

#### 练习 2

比较同一段文本的字符级长度和 BPE token 长度。

#### 练习 3

把第 14 讲的小 GPT 数据管线改成 BPE 版本。

#### 练习 4

打印 `vocab_size = len(tokenizer)`，估算 embedding 和 lm head 参数量。

#### 练习 5

尝试训练一个较小 vocab 的自定义 BPE tokenizer，并替换 GPT-2 tokenizer。

---

### 本讲总结

这一讲把字符级小 GPT 扩展到了 BPE tokenizer。

核心结论如下：

1. 字符级 tokenizer 简单，但序列长、效率低。
2. BPE 通过合并高频片段得到子词 token，能缩短序列。
3. GPT 模型本质上处理 token id，不关心 token 来自字符级还是 BPE。
4. 从字符级切到 BPE，主要改 tokenizer、encode/decode、vocab_size 和 checkpoint 元信息。
5. BPE 词表更大，会增加 embedding 和 lm head 参数量。
6. 训练和推理必须使用同一个 tokenizer。
7. 小模型训练时，使用较小 vocab 的自训练 tokenizer 往往比直接用大词表更合适。

至此，第三册第三部分“从零训练小 GPT”正文第一版完成。

## 与第一册第 15 讲的关系

第一册第 15 讲提供 miniGPT 的最小教学实现。本实战部分后续应在此基础上扩展为完整项目，包括：

1. 独立项目目录结构。
2. 数据下载和预处理脚本。
3. 训练脚本。
4. 验证集评估。
5. checkpoint 保存和恢复。
6. 采样策略。
7. loss 曲线分析。
8. 可写入简历的项目总结。
