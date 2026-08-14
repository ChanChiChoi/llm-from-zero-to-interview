# 第四章：Dataset 与 DataLoader

Dataset 和 DataLoader 是 PyTorch 里把“原始数据”变成“可训练 batch”的核心组件。前面三章解决的是 tensor、autograd 和模型组织，到了真实训练里，最容易出问题的往往不是模型本身，而是数据管线：样本格式不统一、长度不一致、padding 方向错了、shuffle 没生效、分布式下数据重复、collate_fn 写错、num_workers 太大导致卡死、pin_memory 和 device 搬运时机不合理。

本章目标不是背 API，而是把一个可维护的数据管线讲清楚：Dataset 负责描述“单个样本怎么取”，DataLoader 负责描述“样本怎么组成 batch 并高效送到训练循环”，collate_fn 负责处理变长样本和复杂样本结构，sampler 负责控制采样顺序，padding 和 packing 负责把不同长度样本整理成模型可吃的张量。

## 0. 本章范围与资料

本章以 PyTorch 官方 `torch.utils.data`、`Dataset`、`IterableDataset`、`DataLoader`、default collate、sampler、`DistributedSampler`、worker seed、`worker_init_fn`、`pin_memory`、`persistent_workers` 和 `prefetch_factor` 文档为 API 语义依据。文末列出可直接核验的资料入口，并区分框架行为、教学估算和目标数据管线实测。

本章聚焦大模型训练最常见的数据管线问题：map-style dataset、iterable-style dataset、`__len__` / `__getitem__`、`collate_fn`、padding、labels ignore index、packing、shuffle、sampler、batch sampler、length bucket、DataLoader worker、pinned memory、分布式数据切分，以及最小可运行的数据管线审计 demo。

本章不展开生产级数据湖、远程对象存储流式读取、WebDataset / Datasets / DataPipes、复杂多模态数据解码、GPU 数据预取、分布式 checkpoint 数据状态恢复或大规模训练数据治理。这些内容分别放在数据工程、训练系统、AI Infra 和多模态章节中展开。文中的吞吐、padding 比例和样本顺序示例只说明机制，不对真实存储、CPU、GPU 或网络环境作性能承诺。

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

### 4.1.1 样本、batch 和有效监督的约束

第一，map-style Dataset 可以理解成从索引到样本的映射：

~~~math
\mathcal{D}:\{0,\ldots,N-1\}\rightarrow \mathcal{S},\qquad s_i=\mathcal{D}(i)
~~~

其中 `N` 是样本数，`s_i` 是第 `i` 个样本。`__len__()` 给出 `N`，`__getitem__(i)` 给出 `s_i`。

第二，`collate_fn` 是从样本列表到 batch 的函数：

~~~math
C(\{s_1,\ldots,s_B\})=(X,Y,M)
~~~

在 causal LM 训练中，常见 batch shape 是：

~~~math
X\in\mathbb{Z}^{B\times T_b},\qquad
Y\in\mathbb{Z}^{B\times T_b},\qquad
M\in\{0,1\}^{B\times T_b}
~~~

其中 `B` 是 batch size，`T_b` 是当前 batch 内 padding 后长度，`X` 是 `input_ids`，`Y` 是 `labels`，`M` 是 `attention_mask`。

第三，变长序列 padding 后的长度通常是当前 batch 内最长样本：

~~~math
T_b=\max_{1\le i\le B} l_i
~~~

其中 `l_i` 是第 `i` 个样本的真实 token 长度。padding token 利用率可以粗略写成：

~~~math
R_{\mathrm{valid}}=\frac{\sum_{i=1}^{B} l_i}{B T_b},\qquad
R_{\mathrm{pad}}=1-R_{\mathrm{valid}}
~~~

`R_valid` 越低，说明 batch 里浪费在 padding 上的计算越多。length bucket 的目标就是让同一个 batch 内的 `l_i` 更接近，从而降低 `R_pad`。

第四，causal LM 的 loss 通常使用右移后的 logits 和 labels：

~~~math
\mathrm{logits}_{\mathrm{shift}}\in\mathbb{R}^{B\times (T_b-1)\times V},\qquad
Y_{\mathrm{shift}}\in\mathbb{Z}^{B\times (T_b-1)}
~~~

其中 `V` 是词表大小。padding 位置的 label 通常设为 `-100`，让 `CrossEntropyLoss(ignore_index=-100)` 忽略它们。

第五，分布式采样器要把样本索引分给不同 rank。理想情况下：

~~~math
S_r\cap S_q=\varnothing,\qquad r\ne q
~~~

其中 `S_r` 是第 `r` 个 rank 在一个 epoch 中看到的索引集合。实际 PyTorch `DistributedSampler` 为了让每个 rank 样本数一致，可能在数据量不能整除时补样本；所以要理解 `drop_last`、样本数和重复样本之间的取舍。

这里的“batch 正确”至少有三层含义：shape 能进入模型，mask 与 padding 语义一致，loss 的分母和样本权重符合训练目标。前两层通过而第三层错误时，训练仍然可以正常运行，却在优化一个不同的目标。

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

IterableDataset 还有一个多 worker 陷阱：每个 worker 都会执行自己的 `__iter__()`。如果 `__iter__()` 无条件从头读取同一个数据源，worker 之间就会重复产出样本。应使用 `torch.utils.data.get_worker_info()` 按 worker id 分片，分布式场景还要同时结合 rank：

```python
from torch.utils.data import IterableDataset, get_worker_info


class ShardedRange(IterableDataset):
    def __init__(self, total):
        self.total = total

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        worker_count = info.num_workers if info is not None else 1
        for value in range(worker_id, self.total, worker_count):
            yield value
```

这段代码只解决单进程内的 worker 分片；多 rank 流式读取还需要把 rank/world size 加入分片函数。对于远程流，必须进一步考虑连接重试、样本边界、断点和 epoch 长度，不能把 `IterableDataset` 当成自动去重的分布式 sampler。

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

这就是很多语言模型训练脚本里最核心的数据拼 batch 逻辑。但 `-100` 只是标签层面的 ignore index；它不会自动让 attention 屏蔽同一个位置，也不会修复已经错位的 shift。input mask、label mask 和 loss reduction 必须分别验证。

在 causal LM 中，还要注意 padding 出现在 shift 前还是 shift 后。若 `labels` 的有效 token 数为 `N_valid`，loss 的有效分母应与 `N_valid` 的定义一致：

~~~math
L_{\mathrm{token}}=
\frac{\sum_{b,t} M^{label}_{b,t}\,\ell_{b,t}}
{\max\left(1,\sum_{b,t}M^{label}_{b,t}\right)}
~~~

其中 `M^{label}` 表示真正参与监督的 label mask。用普通 `mean` 时，框架会按照它的 `ignore_index` 语义处理；如果手动做 token 加权或跨 micro-batch 累积，就要自己维护分子和分母，避免全 padding batch 产生除零或把短样本权重放大。

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

一个简单检查方法是：打印一个 batch 的 `input_ids`、`labels`、`attention_mask`，确认 padding 的位置完全符合预期；再随机选一个有效位置，手工核对它的 label 是否是目标 token，而不是同位置或前一位置的 token。对左 padding，还要同时检查 position ids 和 generation 时的最后一个有效位置。

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

这里每个样本之间加 `eos_id`，表示样本边界。这个简化函数还需要补一个边界保护：如果单个 `seq` 本身比 `max_length` 长，不能把空 chunk 先 append，也不能静默截断；应该在数据预处理阶段明确截断、拒绝或拆分策略。

### 4.8.3 packing 的风险

1. 样本边界要处理清楚。
2. labels 和 mask 要正确切分。
3. 不适合所有任务。
4. 对对话数据或有严格轮次结构的数据，packing 要更谨慎。

简单说：padding 更稳，packing 更省，但实现复杂度更高。对 decoder-only LM，直接把两个样本拼接后使用普通 causal mask，后一个样本可能看到前一个样本的 token；如果训练目标要求样本彼此独立，就必须使用 document boundary mask、reset position ids 或等价的隔离机制。只插入 `eos_id` 并不能阻止跨样本 attention。

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

长度分桶还会改变随机性结构。一个常见折中是先在较大的窗口内随机打乱，再在窗口内按长度组成 batch，而不是对全数据做完全排序。评估吞吐时，应同时报告有效 token/s、padding ratio 和样本顺序策略；只报告 wall-clock batch/s 可能把更小的有效工作量误认为更快。

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

数据加载性能应通过时间线测量，而不是凭参数名称猜测。至少分别记录：取 batch 的等待时间、CPU collate 时间、CPU 到 device 的拷贝时间和 GPU 计算时间。`pin_memory=True` 只有在实际存在 CPU 到 CUDA 的拷贝且目标后端支持时才可能带来收益；在 CPU 训练、极小 batch 或 collate 已经成为瓶颈的场景，开启它可能只是增加内存压力。`persistent_workers=True` 还要求 `num_workers>0`，否则没有可持久化的 worker。

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

`DistributedSampler` 的“不同 rank 不重复”不是无条件保证。设数据集大小为 `N`、world size 为 `R`；当 `N` 不能被 `R` 整除时，为了让每个 rank 拿到相同数量，sampler 可能补齐索引。补齐的样本会在同一 epoch 的全局索引集合中重复。若更关心不重复，可以设置合适的 `drop_last`，但代价是丢掉尾部样本；若更关心每个 rank 的 step 数一致，则需要接受 padding/补样本，并在有效样本数和 loss 统计中记录它。

示例：

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

如果忘记 `set_epoch`，每个 epoch 的样本顺序可能完全一样，削弱随机性。

一个可复现的分布式数据实验至少固定三件事：sampler 的 `seed`、当前 `epoch` 和数据集版本。只固定 DataLoader 的 `generator`，不能替代 `DistributedSampler.set_epoch()`；只固定 sampler，也不能让 Dataset 内部使用的 Python `random` 或 NumPy 随机操作自动复现。数据管线的随机性应当在 worker、sampler 和数据增强三个层级分别说明。

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

排查时不要只看 rank 0 的第一个 batch。收集一个完整 epoch 的样本 id，检查跨 rank 交集、每个 rank 的数量、补样本计数和 `drop_last` 行为；对 IterableDataset 则需要在数据源层记录 shard/record id，因为它可能没有可直接比较的整数索引。

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

## 4.16 从数据异常反推管线问题

数据管线的排查应从“单样本、单 batch、单 epoch、全分布式”逐层扩大。

### 4.16.1 单样本 contract

先固定几个样本，检查字段是否齐全、token 是否为整数、长度是否在允许范围内、空样本如何处理、label 是否与输入共享同一 tokenizer 和版本。单样本不稳定时，不要直接调 `num_workers` 或模型 batch size。

### 4.16.2 单 batch contract

检查每个字段的 shape、dtype、device、有效 token 数和 padding 位置。对 causal LM，随机抽取一行有效位置，确认 `input_ids[:, :-1]` 的预测目标确实是 `labels[:, 1:]`；确认 `labels == -100` 的位置不会进入 loss；确认 attention mask 的 key/query 方向符合模型实现。

### 4.16.3 单 epoch contract

记录样本 id、有效 token 数、padding ratio、空/异常样本数和 batch 等待时间。若使用 sampler，检查每个 rank 的数量、交集和补样本；若使用流式 dataset，记录 shard、worker 和 record id。这样才能区分“数据真的缺了”与“统计只看到了一个 rank”。

### 4.16.4 全训练 contract

再观察吞吐、GPU 利用率、CPU 利用率、内存峰值、异常样本重试和 epoch 边界。数据管线优化不能只看 batch/s，应至少同时看：

~~~math
\mathrm{effective\ tokens/s}
=\frac{\text{有效 token 数}}{\text{wall-clock seconds}}
~~~

如果通过增加 padding 把 batch/s 提高，却让有效 token/s 下降，模型并没有真正变快。

## 4.17 常见误区

1. 以为 Dataset 要负责 batch 组装。通常单样本逻辑放 Dataset，batch 逻辑放 collate_fn。
2. 以为 DataLoader 只是迭代器。它还负责并行加载、采样和 batch 组装。
3. 以为变长序列一定先在 Dataset 里 pad。更常见是交给 collate_fn。
4. 以为 shuffle=True 就能解决分布式数据切分。多卡时通常还要配合 sampler。
5. 以为 padding token 可以随便算进 loss。多数任务里 padding 应该被 mask 掉。
6. 以为 num_workers 越大越好。它需要和 CPU、IO 和预处理成本一起权衡。
7. 以为 packing 只是把序列拼起来。其实还要处理样本边界、labels 和 mask。

## 4.18 章末练习

1. 写一个 `Dataset`，返回文本和长度两个字段。
2. 写一个 `collate_fn`，把变长 token 序列 pad 成 batch。
3. 给 padding 部分构造 `attention_mask` 和 `labels=-100`。
4. 用 `DataLoader` 读取你的数据集，打印一个 batch 的 shape。
5. 把 `shuffle=True` 改成 `DistributedSampler` 形式，并在伪代码里写出 `set_epoch`。
6. 写一个简单的 length bucket 函数，比较 bucket 前后 batch 的平均 padding 比例。

7. 实现一个 `IterableDataset`，用 `get_worker_info()` 把整数范围分给多个 worker，验证同一 epoch 内没有重复值；再说明多 rank 时还缺少什么信息。
8. 令 `N=10`、`world_size=3`，分别推导 `DistributedSampler` 在补齐和 `drop_last` 下每个 rank 可能看到的样本数与重复/丢弃行为。
9. 构造两个有效 token 数差异很大的 micro-batch，比较“每个 batch 的 mean loss 再平均”和“按有效 token 总数归一化”的梯度权重。
10. 对一个 DataLoader 记录 batch 等待时间、有效 token/s、padding ratio 和 GPU 计算时间，判断瓶颈在读取、collate、搬运还是模型。
11. 把多个样本 packing 到固定长度，设计 document boundary mask，使后一个样本不能看到前一个样本；说明只加入 `eos_id` 为什么不够。

## 4.19 资料与证据边界

本章关于 Dataset、DataLoader、collate、worker、sampler 和 pinned memory 的接口语义，优先依据 PyTorch 官方资料：

1. Dataset 类型：https://pytorch.org/docs/stable/data.html#dataset-types
2. `DataLoader`：https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
3. `default_collate`：https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
4. DataLoader 多进程：https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
5. `DistributedSampler`：https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
6. `get_worker_info`：https://pytorch.org/docs/stable/data.html#torch.utils.data.get_worker_info
7. `worker_init_fn`：https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

这些文档可以确认框架的调用约定和 sampler 行为，但不能替代目标数据源上的重复率、吞吐、内存和随机性验证。padding ratio 是教学指标，effective tokens/s 也必须明确有效 token 的定义；packing 的跨样本注意力隔离则取决于模型和 mask 实现，不会由 DataLoader 自动完成。

## 4.20 本章总结

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
