# 第四章：Dataset 与 DataLoader

Dataset 和 DataLoader 是 PyTorch 里把“原始数据”变成“可训练 batch”的核心组件。前面三章解决的是 tensor、autograd 和模型组织，到了真实训练里，最容易出问题的往往不是模型本身，而是数据管线：样本格式不统一、长度不一致、padding 方向错了、shuffle 没生效、分布式下数据重复、collate_fn 写错、num_workers 太大导致卡死、pin_memory 和 device 搬运时机不合理。

本章目标不是背 API，而是把一个可维护的数据管线讲清楚：Dataset 负责描述“单个样本怎么取”，DataLoader 负责描述“样本怎么组成 batch 并高效送到训练循环”，collate_fn 负责处理变长样本和复杂样本结构，sampler 负责控制采样顺序，padding 和 packing 负责把不同长度样本整理成模型可吃的张量。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已核对 PyTorch 官方 `torch.utils.data`、`Dataset`、`IterableDataset`、`DataLoader`、default collate、sampler、`DistributedSampler`、worker seed、`worker_init_fn`、`pin_memory`、`persistent_workers` 和 `prefetch_factor` 文档口径。

本章聚焦大模型训练最常见的数据管线问题：map-style dataset、iterable-style dataset、`__len__` / `__getitem__`、`collate_fn`、padding、labels ignore index、packing、shuffle、sampler、batch sampler、length bucket、DataLoader worker、pinned memory、分布式数据切分，以及最小可运行的数据管线审计 demo。

本章不展开生产级数据湖、远程对象存储流式读取、WebDataset / Datasets / DataPipes、复杂多模态数据解码、GPU 数据预取、分布式 checkpoint 数据状态恢复或大规模训练数据治理。这些内容分别放在数据工程、训练系统、AI Infra 和多模态章节中展开。

## 4.1 为什么数据管线很重要

很多训练问题表面看像模型问题，根因其实在数据管线。

例如：

1. loss 不下降，最后发现 labels 和 input_ids 错位了。
2. 模型输出乱码，最后发现 padding token 被算进 loss。
3. 训练很慢，最后发现每个 batch 都在 Python 里做重活。
4. 分布式训练各卡数据一样，最后发现 sampler 没设对。
5. 显存突然爆了，最后发现 collate_fn 把超长样本直接堆进 batch。

数据管线的职责是把“外部数据格式”稳定变成“模型训练格式”。一个好的管线要做到：

1. 样本读取逻辑清楚。
2. batch 组装逻辑统一。
3. 变长输入可处理。
4. 顺序控制可复现。
5. 分布式场景不重复、不漏样。
6. 尽量减少 CPU 成为瓶颈。

面试回答：

```text
Dataset 和 DataLoader 负责把原始数据组织成训练可用的 batch。Dataset 定义单个样本如何读取，DataLoader 负责并行取样、打乱顺序、组 batch 和调用 collate_fn。对于大模型训练，数据管线的重要性不亚于模型本身，因为很多训练 bug 都来自样本格式、padding、shuffle、sampler 或分布式取样设置错误。
```

## 4.1.1 关键 shape 公式与数据管线调试速查

第一，map-style Dataset 可以理解成从索引到样本的映射：

```math
\mathcal{D}:\{0,\ldots,N-1\}\rightarrow \mathcal{S},\qquad s_i=\mathcal{D}(i)
```

其中 `N` 是样本数，`s_i` 是第 `i` 个样本。`__len__()` 给出 `N`，`__getitem__(i)` 给出 `s_i`。

第二，`collate_fn` 是从样本列表到 batch 的函数：

```math
C(\{s_1,\ldots,s_B\})=(X,Y,M)
```

在 causal LM 训练中，常见 batch shape 是：

```math
X\in\mathbb{Z}^{B\times T_b},\qquad
Y\in\mathbb{Z}^{B\times T_b},\qquad
M\in\{0,1\}^{B\times T_b}
```

其中 `B` 是 batch size，`T_b` 是当前 batch 内 padding 后长度，`X` 是 `input_ids`，`Y` 是 `labels`，`M` 是 `attention_mask`。

第三，变长序列 padding 后的长度通常是当前 batch 内最长样本：

```math
T_b=\max_{1\le i\le B} l_i
```

其中 `l_i` 是第 `i` 个样本的真实 token 长度。padding token 利用率可以粗略写成：

```math
R_{\mathrm{valid}}=\frac{\sum_{i=1}^{B} l_i}{B T_b},\qquad
R_{\mathrm{pad}}=1-R_{\mathrm{valid}}
```

`R_valid` 越低，说明 batch 里浪费在 padding 上的计算越多。length bucket 的目标就是让同一个 batch 内的 `l_i` 更接近，从而降低 `R_pad`。

第四，causal LM 的 loss 通常使用右移后的 logits 和 labels：

```math
\mathrm{logits}_{\mathrm{shift}}\in\mathbb{R}^{B\times (T_b-1)\times V},\qquad
Y_{\mathrm{shift}}\in\mathbb{Z}^{B\times (T_b-1)}
```

其中 `V` 是词表大小。padding 位置的 label 通常设为 `-100`，让 `CrossEntropyLoss(ignore_index=-100)` 忽略它们。

第五，分布式采样器要把样本索引分给不同 rank。理想情况下：

```math
S_r\cap S_q=\varnothing,\qquad r\ne q
```

其中 `S_r` 是第 `r` 个 rank 在一个 epoch 中看到的索引集合。实际 PyTorch `DistributedSampler` 为了让每个 rank 样本数一致，可能在数据量不能整除时补样本；所以要理解 `drop_last`、样本数和重复样本之间的取舍。

## 4.2 Dataset 的两种基本范式

PyTorch 里最常见的是两类 Dataset：map-style dataset 和 iterable-style dataset。

### 4.2.1 map-style Dataset

map-style Dataset 需要实现两个核心方法：

1. `__len__()`：返回样本数。
2. `__getitem__(idx)`：按索引返回第 `idx` 个样本。

一个最小例子：

```python
from torch.utils.data import Dataset


class NumberDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        x = self.values[idx]
        return {"x": x, "y": x * 2}
```

使用时：

```python
dataset = NumberDataset([1, 2, 3])
print(len(dataset))
print(dataset[0])
```

map-style Dataset 的特点是：

1. 能随机访问。
2. 适合监督学习、固定语料、离线数据集。
3. 容易配合 shuffle、sampler 和 batch sampler。

### 4.2.2 iterable-style Dataset

iterable-style Dataset 适合流式数据或无法预先知道总长度的数据。它更像一个迭代器，而不是索引表。

示意：

```python
from torch.utils.data import IterableDataset


class StreamDataset(IterableDataset):
    def __iter__(self):
        for i in range(5):
            yield {"x": i, "y": i * 2}
```

适用场景：

1. 超大规模语料流式读取。
2. 在线日志数据流。
3. 远程对象存储逐条读取。
4. 无法随机访问的数据源。

区别要点：

1. map-style 更适合有索引的数据。
2. iterable-style 更适合流式和超大规模数据。
3. iterable-style 通常不能直接依赖 `shuffle=True`，而要自己设计打乱逻辑。

## 4.3 一个最小可训练文本 Dataset

大模型训练里最常见的数据结构不是单个数值，而是 token 序列。一个简单的文本 Dataset 通常会把文本样本转成 `input_ids`、`labels` 或 `attention_mask`。

示例：

```python
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=16):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer(text)[: self.max_length]
        return {
            "input_ids": ids,
            "length": len(ids),
        }
```

这里先不急着 padding，因为不同样本长度往往不同，更合理的做法是在 `collate_fn` 里统一处理。

如果是监督训练，还可能直接返回：

```python
{
    "input_ids": [...],
    "labels": [...],
    "attention_mask": [...],
}
```

设计 Dataset 时常见原则：

1. `__getitem__` 只负责单样本读取和轻量预处理。
2. 不要把大批量 padding 放在 `__getitem__` 里。
3. 不要把昂贵的 batch 级逻辑塞进 Dataset。
4. 复杂的 batch 拼接放到 `collate_fn`。

## 4.4 DataLoader 解决什么问题

DataLoader 负责把 Dataset 取出的样本变成训练循环可直接消费的 batch。

典型用法：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
)

for batch in loader:
    ...
```

DataLoader 常见职责：

1. 批量取样。
2. 可选 shuffle。
3. 多进程并行读取数据。
4. 调用 collate_fn 组 batch。
5. 可选 pinned memory 加速 GPU 搬运。
6. 配合 sampler 控制采样顺序。

可以把它理解成“Dataset 之上的 batch 组装层”。

## 4.5 batch 是怎么拼起来的

如果样本结构简单，DataLoader 默认会尝试把样本堆成张量。

例如 Dataset 返回：

```python
{ "x": tensor([1, 2]), "y": tensor(3) }
```

DataLoader 可能自动把一个 batch 变成：

```python
{
    "x": tensor([[1, 2], [4, 5]]),
    "y": tensor([3, 6]),
}
```

但默认 collate 只适合“结构一致、shape 一致”的样本。如果样本长度不一样，就会报错。

例如变长 token 序列：

```python
[1, 2, 3]
[4, 5]
[6, 7, 8, 9]
```

不能直接 `stack`。这时就需要自定义 `collate_fn`。

## 4.6 collate_fn：batch 组装的关键

`collate_fn` 接收一个样本列表，输出一个 batch。

典型签名：

```python
def collate_fn(samples):
    ...
    return batch
```

### 4.6.1 变长序列 padding

最常见的做法是把一个 batch 里所有序列 pad 到同样长度。

示例：

```python
import torch


def pad_sequences(seqs, pad_value=0):
    max_len = max(len(seq) for seq in seqs)
    batch = []
    mask = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        batch.append(seq + [pad_value] * pad_len)
        mask.append([1] * len(seq) + [0] * pad_len)
    return (
        torch.tensor(batch, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )
```

对应 `collate_fn`：

```python
def collate_fn(samples):
    input_ids, attention_mask = pad_sequences(
        [s["input_ids"] for s in samples],
        pad_value=0,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
```

这里注意两点：

1. `pad_value` 要和 tokenizer 的 pad token 一致。
2. `attention_mask` 要正确标记有效 token 和 padding token。

### 4.6.2 labels 的 padding

监督学习时 labels 也可能需要 padding。但很多训练任务里，padding 部分不应该参与 loss。

常见做法：

```python
def pad_labels(seqs, pad_value=-100):
    max_len = max(len(seq) for seq in seqs)
    out = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        out.append(seq + [pad_value] * pad_len)
    return torch.tensor(out, dtype=torch.long)
```

`-100` 是 PyTorch 交叉熵里常见的 ignore index。这样 padding 位置不会计入 loss。

### 4.6.3 一个更完整的 collate_fn

```python
def collate_fn(samples):
    max_len = max(len(s["input_ids"]) for s in samples)

    input_ids = []
    labels = []
    attention_mask = []

    for s in samples:
        ids = s["input_ids"]
        lab = s.get("labels", ids)
        pad_len = max_len - len(ids)

        input_ids.append(ids + [0] * pad_len)
        labels.append(lab + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }
```

这就是很多语言模型训练脚本里最核心的数据拼 batch 逻辑。

## 4.7 padding 的方向和坑

padding 不只是“补零”，还要注意方向。

### 4.7.1 right padding

右侧 padding 最常见：

```text
[1, 2, 3, 0, 0]
```

优点：

1. 最直观。
2. 和大多数训练代码兼容。
3. attention mask 容易构造。

### 4.7.2 left padding

左侧 padding 常见于某些生成推理场景：

```text
[0, 0, 1, 2, 3]
```

它有时更方便把不同长度 prompt 对齐到最右侧。但在训练里，如果模板、mask 或位置编码没有处理好，容易引入错位。

### 4.7.3 常见坑

1. padding token 被算进 loss。
2. attention mask 方向反了。
3. labels 和 input_ids 没有对齐。
4. 左 padding 与位置编码或模板不一致。
5. 只 pad 了 input 没 pad labels。

一个简单检查方法是：打印一个 batch 的 `input_ids`、`labels`、`attention_mask`，确认 padding 的位置完全符合预期。

## 4.8 packing：把多个短样本拼进一个长序列

除了 padding，还有一种提高 token 利用率的方法叫 packing。

### 4.8.1 为什么需要 packing

如果 batch 中很多样本都很短，单纯 padding 会浪费大量计算。

例如：

```text
样本 A: 20 tokens
样本 B: 18 tokens
样本 C: 22 tokens
```

如果都 pad 到 22，浪费不大；但如果一个 batch 中长度分布很散，浪费会很明显。

packing 的思路是把多个短样本拼接到同一个固定长度序列里，减少 padding 浪费。

### 4.8.2 简化 packing 示例

```python
def pack_sequences(seqs, max_length, eos_id=2):
    packed = []
    cur = []

    for seq in seqs:
        if len(cur) + len(seq) + 1 > max_length:
            packed.append(cur)
            cur = []
        cur.extend(seq + [eos_id])

    if cur:
        packed.append(cur)

    return packed
```

这里每个样本之间加 `eos_id`，表示样本边界。

### 4.8.3 packing 的风险

1. 样本边界要处理清楚。
2. labels 和 mask 要正确切分。
3. 不适合所有任务。
4. 对对话数据或有严格轮次结构的数据，packing 要更谨慎。

简单说：padding 更稳，packing 更省，但实现复杂度更高。

## 4.9 shuffle、sampler 和 batch_sampler

DataLoader 里控制“怎么取样”的几个概念容易混淆。

### 4.9.1 shuffle

`shuffle=True` 表示每个 epoch 随机打乱样本顺序。

```python
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

它适合单机单卡的简单场景。

### 4.9.2 sampler

sampler 决定“按什么顺序取样本索引”。

例如：

```python
from torch.utils.data import SequentialSampler, RandomSampler
```

在复杂场景里你可能需要：

1. 按长度排序后再分桶。
2. 按类别均衡采样。
3. 按权重采样。
4. 分布式训练下只取某个 rank 的样本子集。

### 4.9.3 batch_sampler

batch_sampler 直接产出一批索引，而不是单个索引。

适合：

1. 按长度 bucket 组 batch。
2. 自定义 batch 大小策略。
3. 复杂采样逻辑。

一般优先级是：

1. 简单场景用 `shuffle=True`。
2. 需要控制采样顺序用 `sampler`。
3. 需要控制“每个 batch 里有哪些样本”用 `batch_sampler`。

## 4.10 按长度分桶：减少 padding 浪费

语言模型训练里，长度差异很大时常用 bucket sampler。

直觉是：把长度相近的样本放进同一个 batch，这样 pad 更少。

简化思路：

```python
def bucket_by_length(samples, bucket_size=4):
    samples = sorted(samples, key=lambda x: len(x["input_ids"]))
    buckets = []
    for i in range(0, len(samples), bucket_size):
        buckets.append(samples[i : i + bucket_size])
    return buckets
```

好处：

1. 减少 padding。
2. 提高 token 利用率。
3. 常常能提升训练吞吐。

代价：

1. batch 不再完全随机。
2. 可能引入轻微分布偏差。
3. 实现更复杂。

## 4.11 DataLoader 的性能参数

几个最常见的参数：

```python
DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

### 4.11.1 num_workers

`num_workers` 决定用多少个子进程并行读数据。

直觉：

1. 数据读取和预处理放到多个 worker。
2. 主进程专注训练。
3. 适合 CPU 预处理较重的数据集。

但不是越大越好：

1. worker 太多会争抢 CPU。
2. 某些数据读取库不适合多进程。
3. Windows、notebook、共享环境里可能更脆弱。

### 4.11.2 pin_memory

`pin_memory=True` 可以让 CPU 内存页锁定，通常有助于加快数据拷贝到 GPU。

训练时常见写法：

```python
for batch in loader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```

这和 pin_memory 配合时更有效。

### 4.11.3 persistent_workers

`persistent_workers=True` 可以让 worker 在多个 epoch 间保持存活，减少反复启动开销。

适合：

1. 多 epoch 训练。
2. worker 启动成本高。

### 4.11.4 prefetch_factor

`prefetch_factor` 控制每个 worker 预先加载多少个 batch。

如果数据集读取慢，可以通过预取隐藏一部分 IO 延迟；但太大也会增加内存占用。

## 4.12 分布式训练中的数据切分

多卡训练时，不能让每张卡都看到完全一样的数据顺序，否则等于重复训练。

这时常用 `DistributedSampler`。

示例：

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)
```

要点：

1. 每个 rank 拿到不同子集。
2. 每个 epoch 需要调用 `sampler.set_epoch(epoch)`，否则 shuffle 可能不变。
3. sampler 和 `shuffle=True` 通常不能同时乱配。

示例：

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

如果忘记 `set_epoch`，每个 epoch 的样本顺序可能完全一样，削弱随机性。

## 4.13 一个完整的数据管线例子

下面把 Dataset、collate_fn 和 DataLoader 串起来。

```python
import torch
from torch.utils.data import Dataset, DataLoader


class ToyTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.tokenizer(self.texts[idx])
        return {"input_ids": ids, "labels": ids.copy()}


def collate_fn(samples):
    max_len = max(len(s["input_ids"]) for s in samples)
    input_ids, labels, attention_mask = [], [], []

    for s in samples:
        ids = s["input_ids"]
        lab = s["labels"]
        pad_len = max_len - len(ids)

        input_ids.append(ids + [0] * pad_len)
        labels.append(lab + [-100] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
    }


texts = ["hello world", "tiny dataset", "pytorch"]
tokenizer = lambda s: [ord(c) % 100 for c in s][:8]

dataset = ToyTextDataset(texts, tokenizer)
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
)

for batch in loader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"])
```

这个例子虽然简单，但结构已经接近真实训练脚本：

1. Dataset 负责单样本读取。
2. collate_fn 负责 batch 组装和 padding。
3. DataLoader 负责批量加载和打乱。

## 4.14 常见 bug 与排查顺序

### 4.14.1 collate_fn 报错

常见原因：

1. 样本字段不一致。
2. 某个样本长度为 0。
3. 直接尝试 stack 变长序列。
4. 返回了 Python 对象而不是 tensor。

排查方法：先打印前几个样本的原始结构，再打印 batch 结构。

### 4.14.2 数据加载特别慢

常见原因：

1. `__getitem__` 里做了太重的预处理。
2. `num_workers` 太少。
3. 磁盘 IO 慢。
4. 远程读取或压缩解压成本太高。
5. collate_fn 里有大量 Python 循环。

### 4.14.3 每个 epoch 顺序都一样

常见原因：

1. 没开 shuffle。
2. 分布式 sampler 没设 epoch。
3. 自己写的 sampler 没有随机化。

### 4.14.4 loss 异常偏小

常见原因：

1. labels 被错误地 padding 成有效 token。
2. padding token 参与了 loss。
3. shift 操作对齐错了。
4. mask 把大部分 token 忽略了。

### 4.14.5 多卡训练样本重复

常见原因：

1. 没有使用 DistributedSampler。
2. sampler 配置不对。
3. 每个 rank 都读了完整数据集。

## 4.15 最小可运行 DataLoader 审计 demo

下面这个 demo 把本章核心点串起来：Dataset 只返回单样本，`collate_fn` 负责 padding、`labels=-100` 和 `attention_mask`，DataLoader 用固定 `generator` 做可复现 shuffle，最后用 `DistributedSampler` 模拟两个 rank 的数据切分。

```python
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

PAD_ID = 0
IGNORE_INDEX = -100


class ToyCausalDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [list(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx]
        if len(ids) < 2:
            raise ValueError("causal LM sample must contain at least 2 tokens")
        return {"input_ids": ids, "labels": ids.copy(), "length": len(ids)}


def collate_causal_lm(samples):
    max_len = max(s["length"] for s in samples)
    input_ids, labels, attention_mask = [], [], []

    for s in samples:
        ids = s["input_ids"]
        lab = s["labels"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [PAD_ID] * pad_len)
        labels.append(lab + [IGNORE_INDEX] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "lengths": torch.tensor([s["length"] for s in samples], dtype=torch.long),
    }


def padding_waste(lengths, batch_size):
    total_slots = 0
    real_tokens = 0
    for i in range(0, len(lengths), batch_size):
        batch = lengths[i : i + batch_size]
        total_slots += len(batch) * max(batch)
        real_tokens += sum(batch)
    return round((total_slots - real_tokens) / total_slots, 3)


sequences = [
    [11, 12, 13, 14],
    [21, 22],
    [31, 32, 33, 34, 35, 36],
    [41, 42, 43],
    [51, 52, 53, 54, 55],
    [61, 62, 63],
]

dataset = ToyCausalDataset(sequences)
generator = torch.Generator().manual_seed(7)
loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    generator=generator,
    collate_fn=collate_causal_lm,
    num_workers=0,
)

batch = next(iter(loader))
valid_tokens = int(batch["attention_mask"].sum().item())
all_slots = batch["attention_mask"].numel()
ignore_pad_ok = bool((batch["labels"][~batch["attention_mask"]] == IGNORE_INDEX).all())
shift_logits_shape = (batch["input_ids"].size(0), batch["input_ids"].size(1) - 1, 128)
shift_labels_shape = tuple(batch["labels"][:, 1:].shape)

lengths = [len(x) for x in sequences]
plain_waste = padding_waste(lengths, batch_size=3)
bucketed_waste = padding_waste(sorted(lengths), batch_size=3)

sampler_rank0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=13)
sampler_rank1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=13)
sampler_rank0.set_epoch(0)
sampler_rank1.set_epoch(0)
rank0_indices = list(iter(sampler_rank0))
rank1_indices = list(iter(sampler_rank1))

print("batch_input_shape=", tuple(batch["input_ids"].shape))
print("batch_lengths=", batch["lengths"].tolist())
print("attention_mask=", batch["attention_mask"].int().tolist())
print("valid_token_ratio=", round(valid_tokens / all_slots, 3))
print("ignore_pad_ok=", ignore_pad_ok)
print("shift_logits_shape=", shift_logits_shape)
print("shift_labels_shape=", shift_labels_shape)
print("padding_waste_plain=", plain_waste)
print("padding_waste_bucketed=", bucketed_waste)
print("rank0_indices=", rank0_indices)
print("rank1_indices=", rank1_indices)
print("distributed_overlap=", sorted(set(rank0_indices) & set(rank1_indices)))
```

期望输出类似：

```text
batch_input_shape= (3, 6)
batch_lengths= [2, 6, 3]
attention_mask= [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
valid_token_ratio= 0.611
ignore_pad_ok= True
shift_logits_shape= (3, 5, 128)
shift_labels_shape= (3, 5)
padding_waste_plain= 0.303
padding_waste_bucketed= 0.148
rank0_indices= [4, 0, 2]
rank1_indices= [3, 5, 1]
distributed_overlap= []
```

这段输出要检查几件事：

1. `input_ids` 是 `[B,T_b]`，本例为 `[3,6]`。
2. `attention_mask=0` 的位置，`labels` 必须是 `-100`。
3. next-token loss 使用 `T_b-1` 个位置，因此 shift 后 shape 是 `[B,T_b-1,V]` 和 `[B,T_b-1]`。
4. length bucket 把 padding waste 从 `0.303` 降到 `0.148`，说明长度相近的样本放在一起能减少浪费。
5. 两个 rank 的索引无交集，说明这个 toy 场景下没有重复样本。

## 4.16 面试官会怎么问

### 问题一：Dataset 和 DataLoader 有什么区别？

回答模板：

```text
Dataset 负责定义单个样本如何读取和返回，DataLoader 负责把样本按 batch 组织起来，并支持 shuffle、并行加载、collate_fn 和 sampler。Dataset 更偏“样本级”，DataLoader 更偏“batch 级”。
```

### 问题二：为什么变长序列不能直接用默认 DataLoader？

回答模板：

```text
默认 DataLoader 会尝试把样本 stack 成规则张量，但变长序列长度不一致，无法直接 stack。需要自定义 collate_fn，在 batch 级做 padding、mask 和 labels 对齐。
```

### 问题三：collate_fn 的作用是什么？

回答模板：

```text
collate_fn 接收一个样本列表，负责把它们组装成 batch。对于大模型训练，它通常要做 padding、构造 attention_mask、处理 labels 的 ignore index，有时还会做 packing 或把复杂字段整理成统一结构。
```

### 问题四：num_workers、pin_memory、persistent_workers 分别有什么作用？

回答模板：

```text
num_workers 控制并行加载数据的 worker 数量；pin_memory 可以提升 CPU 到 GPU 的拷贝效率；persistent_workers 可以让 worker 跨 epoch 保持存活，减少重复启动开销。它们主要影响数据吞吐，不直接改变模型逻辑。
```

### 问题五：为什么分布式训练要用 DistributedSampler？

回答模板：

```text
因为多卡训练时每张卡应该看到不同的数据子集，否则会重复训练。DistributedSampler 会按 rank 切分数据，保证各卡数据不重复，并且通常需要每个 epoch 调用 set_epoch 来刷新 shuffle。
```

### 问题六：为什么 `collate_fn` 通常比在 Dataset 里 padding 更合适？

回答模板：

```text
Dataset 更适合处理单样本逻辑，而 padding 是 batch 级决策，因为 pad 到多长取决于当前 batch 的最长样本。如果在 Dataset 里提前 pad 到全局 max length，容易浪费大量计算；放到 collate_fn 里可以按 batch 动态 padding，也更方便同时构造 attention_mask 和 labels ignore index。
```

### 问题七：DataLoader 里如何保证随机性可复现？

回答模板：

```text
单进程场景可以给 DataLoader 传入固定 seed 的 torch.Generator；多 worker 场景还要理解 worker seed，并在需要时使用 worker_init_fn 给 Python random、NumPy 或自定义随机逻辑设种子。分布式场景中 DistributedSampler 通常每个 epoch 调用 set_epoch(epoch)，否则不同 epoch 的 shuffle 顺序可能不变。
```

## 4.17 常见误区

1. 以为 Dataset 要负责 batch 组装。通常单样本逻辑放 Dataset，batch 逻辑放 collate_fn。
2. 以为 DataLoader 只是迭代器。它还负责并行加载、采样和 batch 组装。
3. 以为变长序列一定先在 Dataset 里 pad。更常见是交给 collate_fn。
4. 以为 shuffle=True 就能解决分布式数据切分。多卡时通常还要配合 sampler。
5. 以为 padding token 可以随便算进 loss。多数任务里 padding 应该被 mask 掉。
6. 以为 num_workers 越大越好。它需要和 CPU、IO 和预处理成本一起权衡。
7. 以为 packing 只是把序列拼起来。其实还要处理样本边界、labels 和 mask。

## 4.18 小练习

1. 写一个 `Dataset`，返回文本和长度两个字段。
2. 写一个 `collate_fn`，把变长 token 序列 pad 成 batch。
3. 给 padding 部分构造 `attention_mask` 和 `labels=-100`。
4. 用 `DataLoader` 读取你的数据集，打印一个 batch 的 shape。
5. 把 `shuffle=True` 改成 `DistributedSampler` 形式，并在伪代码里写出 `set_epoch`。
6. 写一个简单的 length bucket 函数，比较 bucket 前后 batch 的平均 padding 比例。

## 4.19 本章总结

Dataset 负责单样本逻辑，DataLoader 负责 batch 组装和并行加载，collate_fn 负责把变长或复杂样本整理成训练可用的张量结构。对于大模型训练来说，最常见的数据处理任务是 padding、attention mask、labels 对齐和分布式采样。

要记住的主线是：

1. map-style Dataset 适合随机访问，IterableDataset 适合流式数据。
2. 变长样本通常在 collate_fn 里处理，而不是在 Dataset 里硬 pad。
3. padding、labels 和 attention_mask 必须一起设计。
4. shuffle、sampler 和 batch_sampler 是不同层次的采样控制。
5. num_workers、pin_memory、persistent_workers 影响吞吐和延迟。
6. 分布式训练必须保证样本切分正确，避免重复训练。
7. packing 能提升 token 利用率，但实现和调试复杂度更高。

下一章会进入训练循环工程，重点讲一个完整训练 step 应该怎么写，怎么组织 optimizer、scheduler、梯度裁剪、日志、验证、checkpoint 和异常恢复。
