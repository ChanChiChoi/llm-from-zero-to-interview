# 第四章：Dataset 与 DataLoader

Dataset 和 DataLoader 是 PyTorch 里把“原始数据”变成“可训练 batch”的核心组件。前面三章解决的是 tensor、autograd 和模型组织，到了真实训练里，最容易出问题的往往不是模型本身，而是数据管线：样本格式不统一、长度不一致、padding 方向错了、shuffle 没生效、分布式下数据重复、collate_fn 写错、num_workers 太大导致卡死、pin_memory 和 device 搬运时机不合理。

本章目标不是背 API，而是把一个可维护的数据管线讲清楚：Dataset 负责描述“单个样本怎么取”，DataLoader 负责描述“样本怎么组成 batch 并高效送到训练循环”，collate_fn 负责处理变长样本和复杂样本结构，sampler 负责控制采样顺序，padding 和 packing 负责把不同长度样本整理成模型可吃的张量。

阅读时沿三条连续问题推进：先从单样本字段走到 batch 的 shape、mask 和有效 loss 目标；再从样本 key 走到 sampler、worker 与 rank 的分片；最后从 batch 等待时间反推 CPU 预处理、内存搬运和预取是否值得调整。三条链在训练循环汇合：模型接收的张量必须符合 contract，分布式各 rank 必须拿到预期数据，吞吐变化必须用有效 token 和计时证据判断。

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

第一，对于有限的 map-style 数据集，可以把样本访问写成从 key 到样本的映射：

~~~math
\mathcal{D}:\mathcal{I}\rightarrow \mathcal{S},\qquad s_k=\mathcal{D}(k),\quad k\in\mathcal{I}
~~~

其中 `I` 是一组可访问的 key，`s_k` 是 key `k` 对应的样本。有限、连续编号的数据集常用 `I={0,...,N-1}`；但 map-style key 不一定是整数。`__getitem__(key)` 是取样协议，有限数据集通常还实现 `__len__()`，因为常用 sampler 和 `len(dataloader)` 需要知道数据量。PyTorch 2.15 也允许 Dataset 可选实现 `__getitems__(keys)`，让 DataLoader 一次请求一批 key；它适合能合并磁盘读取、数据库查询或向量化预处理的场景，见 [PyTorch 2.15 Dataset API](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.Dataset)。空数据集本身可以存在，但不能产出训练 batch，训练循环必须明确定义跳过或报错策略。

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

其中 `l_i` 是第 `i` 个样本的真实 token 长度，且本式假定 `B>0`、每个样本长度为正。padding token 利用率可以粗略写成：

~~~math
R_{\mathrm{valid}}=\frac{\sum_{i=1}^{B} l_i}{B T_b},\qquad
R_{\mathrm{pad}}=1-R_{\mathrm{valid}}
~~~

`R_valid` 越低，说明 batch 里浪费在 padding 上的计算越多。length bucket 的目标就是让同一个 batch 内的 `l_i` 更接近，从而降低 `R_pad`。如果 batch 为空，或所有样本都是空序列，分母为零，不能把利用率当成 0；应在 collate 或数据预处理阶段拒绝这类输入，或者单独定义跳过策略。

第四，causal LM 的 loss 通常使用右移后的 logits 和 labels：

~~~math
\mathrm{logits}_{\mathrm{shift}}\in\mathbb{R}^{B\times (T_b-1)\times V},\qquad
Y_{\mathrm{shift}}\in\mathbb{Z}^{B\times (T_b-1)}
~~~

其中 `V` 是词表大小。padding 位置的 label 通常设为 `-100`，让 `CrossEntropyLoss(ignore_index=-100)` 忽略它们。`attention_mask` 和 label 的 ignore mask 服务于不同位置：前者按模型约定控制注意力可见性，后者决定哪些预测目标进入 loss；某些模型还需要 position ids 或专门的 document boundary mask。

第五，分布式采样器要把样本索引分给不同 rank。理想情况下：

~~~math
S_r\cap S_q=\varnothing,\qquad r\ne q
~~~

其中 `S_r` 是第 `r` 个 rank 在一个 epoch 中看到的索引集合。实际 PyTorch `DistributedSampler` 为了让每个 rank 样本数一致，可能在数据量不能整除时补样本；所以要理解 `drop_last`、样本数和重复样本之间的取舍。

这里的“batch 正确”至少有三层含义：shape 能进入模型，mask 与 padding 语义一致，loss 的分母和样本权重符合训练目标。前两层通过而第三层错误时，训练仍然可以正常运行，却在优化一个不同的目标。

用两条 causal LM 样本把这些量手算一次。令 A=`[11,12,13,14]`、B=`[21,22]`，本例假设 pad id 为 0，padding 放在右侧。当前最长长度 `T_b=max(4,2)=4`，所以 `input_ids` 为 `[[11,12,13,14],[21,22,0,0]]`，`attention_mask` 为 `[[1,1,1,1],[1,1,0,0]]`。一共 6 个真实 token、8 个 batch 槽位，故有效 token 利用率为 `6/8=0.75`，padding 比例为 `1-0.75=0.25`。

若 labels 在模型前向前与输入位置对齐并把 padding 写成 `-100`，则未 shift 的 labels 是 `[[11,12,13,14],[21,22,-100,-100]]`。next-token loss 使用 `logits[:, :-1, :]` 对 `labels[:, 1:]`，目标变成 `[[12,13,14],[22,-100,-100]]`，实际参与监督的是 4 个目标 token。此处 padding 后的 input mask 有 2 个零，但被忽略的 shifted labels 有 2 个 `-100`；两者数值恰好相等只是这个例子的结果，不能据此把 attention mask 直接当 label mask。只要输入、标签、shift 和分母逐项对齐，读者就能从 batch 输出判断模型到底在预测哪些 token。

## 4.2 Dataset 的两种基本范式

PyTorch 里最常见的是两类 Dataset：map-style dataset 和 iterable-style dataset。

### 4.2.1 map-style Dataset

map-style Dataset 必须能按 key 取样；有限数据集通常还会提供长度：

1. `__getitem__(key)`：返回 key 对应的一个样本。
2. `__len__()`：返回样本数；常用 sampler 和 `len(dataloader)` 依赖它，但不是所有可能的 map-style Dataset 都必须实现。

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
代码片段，静态审阅，未运行。

使用时：

```python
dataset = NumberDataset([1, 2, 3])
print(len(dataset))
print(dataset[0])
```
代码片段，静态审阅，未运行。

map-style Dataset 的特点是：

1. 能随机访问。
2. 适合监督学习、固定语料、离线数据集。
3. 容易配合 shuffle、sampler 和 batch sampler。

PyTorch 2.15 的 Dataset 还可选实现 `__getitems__(indices)`。DataLoader 在 map-style 自动 batching 时可以用它一次取一批样本；简单实现可以逐项调用 `__getitem__`，而能批量读文件或做向量化预处理时才可能减少每样本固定开销。它不改变输出 batch 的 contract，也不会自动保证吞吐提升。

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
代码片段，静态审阅，未运行。

适用场景：

1. 超大规模语料流式读取。
2. 在线日志数据流。
3. 远程对象存储逐条读取。
4. 无法随机访问的数据源。

区别要点：

1. map-style 更适合有索引的数据。
2. iterable-style 更适合流式和超大规模数据。
3. iterable-style 的顺序由 `__iter__()` 控制；它没有 key/index，因此不能使用 map-style 的 `sampler`、`batch_sampler` 或 DataLoader `shuffle=True`。需要打乱流式样本时，要在数据源或有限缓冲区里实现，并明确其随机性和覆盖边界；参见 [PyTorch 2.15 数据集类型与采样约定](https://docs.pytorch.org/docs/2.15/data.html#dataset-types)。

| 维度 | map-style | iterable-style |
| --- | --- | --- |
| 取样 | 用 key 调用 `__getitem__`，可选批量 `__getitems__` | `__iter__` 从数据源逐条或分块产出 |
| 顺序与分批 | sampler/batch_sampler 控制 key；DataLoader 可 shuffle | 顺序与分片由输入流负责；不能直接用 sampler 或 `shuffle=True` |
| 多 worker | 主进程 sampler 分 key 给 worker | 每个 worker 都有 Dataset 副本，需显式分片避免重复 |
| 常见场景 | 固定文件、索引表、可随机读取语料 | 远程流、日志、顺序读取或不适合随机 seek 的语料 |

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
代码片段，静态审阅，未运行。

这段代码只解决单进程内的 worker 分片；多 rank 流式读取还需要把 rank/world size 加入分片函数，例如把全局槽位按 `global_worker_id = rank * worker_count + worker_id` 重新计算。对于远程流，必须进一步考虑连接重试、样本边界、断点和 epoch 长度，不能把 `IterableDataset` 当成自动去重的分布式 sampler。

例如 `total=8`、两个 worker 时，worker 0 读取 `0,2,4,6`，worker 1 读取 `1,3,5,7`；样本集合的并集是原范围且交集为空。扩展到分布式时，若每个 rank 有 `W` 个 worker，先定义全局 worker 数 `R×W` 和唯一的全局 worker id，再按该 id 分配 shard；只使用本地 worker id 会让每个 rank 都重复读取相同分片。流式数据的 shard 若不是等长，训练前还要检查每个 rank 的 batch/step 数是否一致。

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
代码片段，静态审阅，未运行。

这里先不急着 padding，因为不同样本长度往往不同，更合理的做法是在 `collate_fn` 里统一处理。

如果是监督训练，还可能直接返回：

```python
{
    "input_ids": [...],
    "labels": [...],
    "attention_mask": [...],
}
```
代码片段，静态审阅，未运行。

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
代码片段，静态审阅，未运行。

DataLoader 常见职责：

1. 批量取样。
2. 可选 shuffle。
3. 多进程并行读取数据。
4. 调用 collate_fn 组 batch。
5. 可选 pinned memory 加速 GPU 搬运。
6. 配合 sampler 控制采样顺序。

可以把它理解成“Dataset 之上的 batch 组装层”。

map-style 的默认路径可以读成“主进程 sampler 产出 key → Dataset 逐项或批量取样 → `collate_fn` 组 batch → 训练循环消费”；多 worker 只改变取样与组装在哪些进程执行，不会替你修正字段、mask 或标签语义。IterableDataset 则由数据流本身产出样本，DataLoader 不能替它决定全局顺序和分片。

## 4.5 batch 是怎么拼起来的

如果样本结构简单，DataLoader 默认会尝试把样本堆成张量。

例如 Dataset 返回：

```python
{ "x": tensor([1, 2]), "y": tensor(3) }
```
代码片段，静态审阅，未运行。

DataLoader 可能自动把一个 batch 变成：

```python
{
    "x": tensor([[1, 2], [4, 5]]),
    "y": tensor([3, 6]),
}
```
代码片段，静态审阅，未运行。

但默认 collate 只适合“结构一致、shape 一致”的样本。如果样本长度不一样，就会报错。

例如变长 token 序列：

```python
[1, 2, 3]
[4, 5]
[6, 7, 8, 9]
```
代码片段，静态审阅，未运行。

不能直接 `stack`。这时就需要自定义 `collate_fn`。

`batch_size=1` 仍属于自动 batching，默认 collate 会在样本最外层增加 batch 维。若显式把 `batch_size=None` 且不传 `batch_sampler`，自动 batching 被关闭，`collate_fn` 每次收到一个样本而不是样本列表；这个模式适合 Dataset 本身已经返回一个预组好的 batch 或样本大小动态变化的情况。默认 collate 保留映射、元组等结构并递归堆叠 tensor、数组和数值，但同一字段 shape 不一致时不会自动替你 padding。

关闭自动 batching 时，若没有自定义 `collate_fn`，PyTorch 使用 `default_convert` 转换单样本中的 NumPy 数组，而不会再增加 batch 维；启用自动 batching 时才使用 `default_collate` 递归堆叠样本。具体调用区别见 [PyTorch 2.15 automatic batching 文档](https://docs.pytorch.org/docs/2.15/data.html#loading-batched-and-non-batched-data)。

## 4.6 collate_fn：batch 组装的关键

`collate_fn` 接收一个样本列表，输出一个 batch。默认 `default_collate` 会为支持的字段递归堆叠 batch 维，但不会替变长序列 pad，见 [PyTorch 2.15 `default_collate`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.default_collate)。

典型签名：

```python
def collate_fn(samples):
    ...
    return batch
```
伪代码片段，未运行。

### 4.6.1 变长序列 padding

最常见的做法是把一个 batch 里所有序列 pad 到同样长度。

示例：

```python
import torch


def pad_sequences(seqs, pad_value=0):
    if not seqs:
        raise ValueError("cannot collate an empty sample list")
    if any(len(seq) == 0 for seq in seqs):
        raise ValueError("empty sequence needs an explicit filtering policy")
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
代码片段，静态审阅，未运行。

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
代码片段，静态审阅，未运行。

这里注意两点：

1. `pad_value` 要和 tokenizer 的 pad token 一致。
2. `attention_mask` 要正确标记有效 token 和 padding token。

### 4.6.2 labels 的 padding

监督学习时 labels 也可能需要 padding。但很多训练任务里，padding 部分不应该参与 loss。

常见做法：

```python
def pad_labels(seqs, pad_value=-100):
    if not seqs:
        raise ValueError("cannot collate an empty sample list")
    max_len = max(len(seq) for seq in seqs)
    if max_len == 0:
        raise ValueError("all label sequences are empty")
    out = []
    for seq in seqs:
        pad_len = max_len - len(seq)
        out.append(seq + [pad_value] * pad_len)
    return torch.tensor(out, dtype=torch.long)
```
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

`-100` 是 PyTorch 交叉熵里常见的 ignore index。这样 padding 位置不会计入 loss。

### 4.6.3 一个更完整的 collate_fn

```python
def collate_fn(samples):
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    max_len = max(len(s["input_ids"]) for s in samples)

    input_ids = []
    labels = []
    attention_mask = []

    for s in samples:
        ids = s["input_ids"]
        lab = s.get("labels", ids)
        if len(ids) == 0:
            raise ValueError("empty input_ids needs an explicit filtering policy")
        if len(lab) != len(ids):
            raise ValueError("labels and input_ids must have the same length")
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
代码片段，静态审阅，未运行。

这就是很多语言模型训练脚本里最核心的数据拼 batch 逻辑。但 `-100` 只是标签层面的 ignore index；它不会自动让 attention 屏蔽同一个位置，也不会修复已经错位的 shift。input mask、label mask 和 loss reduction 必须分别验证。

在 causal LM 中，还要注意 padding 出现在 shift 前还是 shift 后。若 `labels` 的有效 token 数为 `N_valid`，loss 的有效分母应与 `N_valid` 的定义一致：

~~~math
L_{\mathrm{token}}=
\frac{\sum_{b,t} M^{label}_{b,t}\,\ell_{b,t}}
{\max\left(1,\sum_{b,t}M^{label}_{b,t}\right)}
~~~

其中 `M^{label}` 表示真正参与监督的 label mask。用普通 `mean` 时，框架会按照它的 `ignore_index` 语义处理；如果手动做 token 加权或跨 micro-batch 累积，就要自己维护分子和分母，避免全 padding batch 产生除零或把短样本权重放大。

回到前面的 A、B 样本，padding 在右侧时，`labels[:, 1:]` 的第二行是 `[22,-100,-100]`，因此这个 batch 的目标数是 `3+1=4`，不是输入的 6 个有效 token。若训练代码的标签已提前右移，或模型 `forward` 内部会 shift labels，就不能再在外部重复 shift。无论 API 把 shift 放在哪一层，都应从一个具体位置核对“输入位置 t 预测标签位置 t+1”，并把 ignore mask 放在实际参与交叉熵的标签张量上。

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
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    packed = []
    cur = []

    for seq in seqs:
        if len(seq) + 1 > max_length:
            raise ValueError(
                "a sequence including eos_id is longer than max_length; "
                "choose truncate, reject, or split explicitly"
            )
        if len(cur) + len(seq) + 1 > max_length:
            if cur:
                packed.append(cur)
            cur = []
        cur.extend(seq + [eos_id])

    if cur:
        packed.append(cur)

    return packed
```
代码片段，静态审阅，未运行。

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

对 map-style Dataset，`shuffle=True` 会让 DataLoader 选择随机采样顺序；它不能与显式 `sampler` 或 `batch_sampler` 同时指定。对 iterable-style Dataset，顺序由 `__iter__()` 自己控制，Loader 没有索引可供随机排列。

```python
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```
代码片段，静态审阅，未运行。

这是 map-style 数据集的便捷随机顺序入口；要做按 rank 分片、类别重采样或长度分桶时，应显式交给相应 sampler。

### 4.9.2 sampler

sampler 决定“按什么顺序取样本索引”。

例如：

```python
from torch.utils.data import SequentialSampler, RandomSampler
```
代码片段，静态审阅，未运行。

一次普通 sampler 产出一个 key；DataLoader 再按 `batch_size` 将相邻取出的样本组成一批。map-style 的 `sampler` 和 `batch_sampler` 不能同时传入，IterableDataset 也不支持它们，因为流式样本没有 key/index。若在采样器外自行调用 `shuffle`，应确认最终只有一个组件拥有顺序控制权，避免以为换了 seed 实际却重复使用同一顺序。

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
代码片段，静态审阅，未运行。

这个函数只展示“相似长度放在一起”的机制，返回的是样本组，并不是可直接交给 DataLoader 的 `batch_sampler`；真正的 batch sampler 应产出数据集 key 的列表。若把全数据永久按长度排序，训练顺序会长期与长度相关。工程实现通常每个 epoch 先打乱，再在较大窗口内按长度组成 batch，或在 bucket 内打乱样本，并通过固定种子、epoch 与数据版本记录可复现条件。

好处：

1. 减少 padding。
2. 提高 token 利用率。
3. 常常能提升训练吞吐。

代价：

1. batch 不再完全随机。
2. 可能引入轻微分布偏差。
3. 实现更复杂。

长度分桶还会改变随机性结构。一个常见折中是先在较大的窗口内随机打乱，再在窗口内按长度组成 batch，而不是对全数据做完全排序。评估吞吐时，应同时报告有效 token/s、padding ratio 和样本顺序策略；只报告 wall-clock batch/s 可能把更小的有效工作量误认为更快。

用审计 demo 中的六条长度 `[4,2,6,3,5,3]` 手算一次。按原顺序以 3 条组成两批：第一批长度 `[4,2,6]` 需要 `3×6=18` 个槽位，其中 12 个有效，浪费 6 个；第二批 `[3,5,3]` 需要 `3×5=15` 个槽位，其中 11 个有效，浪费 4 个。合计 23 个真实 token 占 33 个槽位，padding waste 是 `(33-23)/33≈0.303`。先按长度排列为 `[2,3,3]` 和 `[4,5,6]` 后，两批共需 `9+18=27` 个槽位，仍有 23 个真实 token，浪费 `(27-23)/27≈0.148`。节省来自更短的 batch 最大长度，不是减少训练目标 token；实际 wall-clock 是否变快仍取决于读取、计算和 kernel 形状。

## 4.11 DataLoader 的性能参数

几个最常见的参数：

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
```
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU，多 worker batch 形状通过；无加速器，pin_memory 未实测）。

`persistent_workers=True` 和 `prefetch_factor` 只适用于 `num_workers>0`。从 `num_workers=0` 开始建立可调试基线，再根据读取/变换是否拖慢训练增加 worker；进程数增加也会扩大 Dataset 中被 worker 访问的 Python 对象和预取 batch 的内存占用。不要把某个固定 worker 数当成通用最佳配置。

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

多进程模式会在 worker 中执行 Dataset 访问、transform 和 `collate_fn`。被 worker 访问的大型 Python 对象可能在多个 worker 进程中各自带来接近父进程的内存占用，所以把庞大的文件名列表或已解码数据直接放在 Python list 里，可能让增加 worker 后 CPU 内存成倍增长。Windows/macOS 的 spawn 模式要求 Dataset、`collate_fn`、`worker_init_fn` 等可 pickle 的定义放在模块顶层，创建 loader 的主程序用 `if __name__ == "__main__":` 保护；短 notebook 示例要先用 `num_workers=0` 排查，不能假设切到多 worker 后仍可运行。平台和内存行为见 [PyTorch 2.15 多进程加载说明](https://docs.pytorch.org/docs/2.15/data.html#single-and-multi-process-data-loading)。

每个 worker 的 PyTorch seed 默认由 DataLoader 的 base seed 加 worker id 得到；Python `random`、NumPy 等其他库的随机状态仍需在 `worker_init_fn` 中分别播种。`generator` 可同时参与 map-style `RandomSampler` 的索引随机化和 worker base seed 生成；它不能代替 `DistributedSampler` 的 `seed` 与 `set_epoch()`。如果数据增强使用多个随机库，固定一个全局 seed 并不足以证明 worker 输出可复现；接口关系见 [worker 随机性说明](https://docs.pytorch.org/docs/2.15/data.html#randomness-in-multi-process-data-loading)。

例如使用 Python `random` 和 NumPy 做随机数据增强时，可从每个 worker 已分配的 PyTorch seed 派生这两个库的种子：

```python
import random
import numpy as np
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
```
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu、NumPy 2.4.6；CPU 双 worker，worker 随机流可复现且不同）。

把这个函数和固定的 `torch.Generator` 一起传给 DataLoader，例如 `DataLoader(dataset, num_workers=2, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(7))`，能让同一环境和配置下的 worker 随机流可复现且彼此不同；不能据此保证外部数据源、非确定性算子或跨版本执行都完全一致。

### 4.11.2 pin_memory

`pin_memory=True` 可以让 CPU 内存页锁定，通常有助于加快数据拷贝到 GPU。

训练时常见写法：

```python
for batch in loader:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
```
代码片段，静态审阅，未运行。

这和 pin_memory 配合时更有效。

默认 pinning 会递归识别 tensor、映射和包含 tensor 的常见容器；若 `collate_fn` 返回自定义 batch 类，需要为该类实现 `pin_memory()` 才会按预期锁页。锁页内存会占用不可随意换出的主机 RAM，`non_blocking=True` 也只有在传输条件合适时才可能与计算重叠；应测量 host-to-device 时间，而不是仅凭参数组合推断收益。多进程 worker 通常应返回 CPU tensor，再交给 DataLoader pinning，不要在 worker 里直接构造 CUDA tensor。细节见 [PyTorch 2.15 memory pinning](https://docs.pytorch.org/docs/2.15/data.html#memory-pinning)。

### 4.11.3 persistent_workers

`persistent_workers=True` 可以让 worker 在多个 epoch 间保持存活，减少反复启动开销。

适合：

1. 多 epoch 训练。
2. worker 启动成本高。

### 4.11.4 prefetch_factor

`prefetch_factor` 控制每个 worker 预先加载多少个 batch。PyTorch 2.15 中，`num_workers>0` 时默认值为 2，即每个 worker 预取两个 batch；`num_workers=0` 时默认值为 `None`。

如果数据集读取慢，可以通过预取隐藏一部分 IO 延迟；但太大也会增加内存占用。

例如 4 个 worker、`prefetch_factor=2` 时，队列最多预取约 `4×2=8` 个 batch。若每个 collated batch 约 64 MiB，单是这些队列就可能占用约 `8×64=512 MiB`，还未计入 Dataset 副本、解码临时对象、共享内存和主进程当前 batch。这个是容量估算而非精确峰值；实际队列深度和 batch 大小会变化。调参时同时观察 batch 等待时间、内存、有效 token/s 和 GPU 空闲区间。

PyTorch 2.15 的 `DataLoader` 还提供 `in_order`。默认 `True` 会按 sampler 顺序返回 batch；`False` 允许先返回先完成的 batch，可能减轻慢 worker 造成的队头阻塞，但不保证提高平均吞吐，还会削弱可复现顺序；当样本类别与读取耗时相关时，先完成的 batch 可能短时偏向某类，造成训练数据分布偏斜。除非测量出队头阻塞且评估过分布风险，不应只为“更快”默认关闭顺序约束；参数与警告见 [PyTorch 2.15 DataLoader API](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.DataLoader)。

数据加载性能应通过时间线测量，而不是凭参数名称猜测。至少分别记录：取 batch 的等待时间、CPU collate 时间、CPU 到 device 的拷贝时间和 GPU 计算时间。`pin_memory=True` 只有在实际存在 CPU 到 CUDA 的拷贝且目标后端支持时才可能带来收益；在 CPU 训练、极小 batch 或 collate 已经成为瓶颈的场景，开启它可能只是增加内存压力。`persistent_workers=True` 还要求 `num_workers>0`，否则没有可持久化的 worker。

## 4.12 分布式训练中的数据切分

多卡训练时，不能让每张卡都看到完全一样的数据顺序，否则等于重复训练。

这时常用 `DistributedSampler`。它适用于大小与元素顺序稳定的 map-style Dataset；流式 Dataset 的 rank/worker shard 需由数据源管理，见 [PyTorch 2.15 `DistributedSampler`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.distributed.DistributedSampler)。

示例：

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=13,
    drop_last=False,
)
loader = DataLoader(dataset, batch_size=8, sampler=sampler, drop_last=False)
```
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU；单进程模拟 rank）。

要点：

1. 每个 rank 拿到不同子集。
2. 每个 epoch 需要调用 `sampler.set_epoch(epoch)`，否则 shuffle 可能不变。
3. 显式设置 `sampler` 后不要再给同一个 DataLoader 传 `shuffle=True`；采样顺序由 sampler 唯一控制。所有 rank 使用相同的 sampler seed，但在每个 epoch 创建 iterator 前调用 `sampler.set_epoch(epoch)`，以得到各 rank 相容而跨 epoch 变化的排列。

`DistributedSampler` 的“不同 rank 不重复”不是无条件保证。设数据集大小为 `N`、world size 为 `R`。默认 `drop_last=False` 时，每个 rank 通常需要相同的样本数

~~~math
N_{\mathrm{sampled}}=\left\lceil\frac{N}{R}\right\rceil R,
\qquad
N_{\mathrm{pad}}=N_{\mathrm{sampled}}-N
~~~

多出来的 `N_pad` 个位置由已有索引补齐，所以全局索引列表可能包含重复样本。设置 `drop_last=True` 时，通常只保留

~~~math
N_{\mathrm{sampled}}=\left\lfloor\frac{N}{R}\right\rfloor R
~~~

个位置，尾部最多丢弃 `N mod R` 个样本。这里的 `drop_last` 是 `DistributedSampler` 自己的参数，它在 rank 分片前处理数据集尾部。`DataLoader(drop_last=...)` 是另一个参数，在每个 rank 已取得自己的索引序列后，决定是否丢弃不足一个 batch 的尾批；两者可以分别设置，不能把它们的丢弃数量混为一谈。

| `drop_last` 所属对象 | 处理阶段 | `False` | `True` |
| --- | --- | --- | --- |
| `DistributedSampler` | rank 切分前 | 补索引，使各 rank 样本数相等；可能重复 | 截掉数据集尾部，使总数可被 rank 数整除 |
| `DataLoader` | 每个 rank 已拿到 sampler 序列后 | 保留不满 batch 的尾批 | 丢弃每个 rank 的不完整尾批 |

用 `N=10`、`R=3`、`shuffle=False` 手算。sampler 默认补齐时，目标总数是 `ceil(10/3)×3=12`，把索引 `0,1` 复制到尾部得到 `[0,1,2,3,4,5,6,7,8,9,0,1]`。按 rank 交错取位后，rank 0 得 `[0,3,6,9]`，rank 1 得 `[1,4,7,0]`，rank 2 得 `[2,5,8,1]`；每 rank 4 条、step 数相同，但有两个重复位置。设置 `DistributedSampler(drop_last=True)` 则只保留 9 条，三个 rank 分别得到 `[0,3,6]`、`[1,4,7]`、`[2,5,8]`，丢弃索引 9。若再设置 `DataLoader(batch_size=2, drop_last=True)`，每 rank 的 3 条又各自少一个尾样本；这是第二次、按 batch 发生的丢弃。

对 IterableDataset，不能用 `DistributedSampler`，也不能从 `len(dataloader)` 推出精确覆盖数。每个 worker 是独立的 Dataset 副本，程序必须把 rank 和 worker id 一起映射到全局 shard；此外，多 worker 下 DataLoader 的 `drop_last=True` 会对每个 worker 副本分别丢弃不满 batch 的尾部，累积丢失量可能大于全局最后一个 batch。若 IterableDataset 实现了 `__len__()`，框架给 iterable loader 的长度仍只是基于 `len(dataset)/batch_size` 的估计，多 worker 分片时尤其需要直接记录实际消费的 sample/shard id。

示例：

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        ...
```
伪代码片段，未运行。

如果忘记 `set_epoch`，每个 epoch 的样本顺序可能完全一样，削弱随机性。

一个可复现的分布式数据实验至少固定三件事：所有 rank 共用的 sampler `seed`、当前 `epoch` 和数据集版本。只固定 DataLoader 的 `generator`，不能替代 `DistributedSampler.set_epoch()`；只固定 sampler，也不能让 Dataset 内部使用的 Python `random` 或 NumPy 随机操作自动复现。数据管线的随机性应当在 worker、sampler 和数据增强三个层级分别说明。

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
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    max_len = max(len(s["input_ids"]) for s in samples)
    input_ids, labels, attention_mask = [], [], []

    for s in samples:
        ids = s["input_ids"]
        lab = s["labels"]
        if not ids:
            raise ValueError("empty input_ids needs an explicit filtering policy")
        if len(lab) != len(ids):
            raise ValueError("labels and input_ids must have the same length")
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


def tokenizer(s):
    return [ord(c) % 100 for c in s][:8]

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
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

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
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    max_len = max(s["length"] for s in samples)
    input_ids, labels, attention_mask = [], [], []

    for s in samples:
        ids = s["input_ids"]
        lab = s["labels"]
        if len(ids) < 2:
            raise ValueError("causal LM sample must contain at least 2 tokens")
        if len(lab) != len(ids) or len(ids) != s["length"]:
            raise ValueError("input_ids, labels, and length must agree")
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
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not lengths or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in lengths):
        raise ValueError("lengths must be non-empty positive integers")
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
repeat_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    generator=torch.Generator().manual_seed(7),
    collate_fn=collate_causal_lm,
    num_workers=0,
)
repeat_batch = next(iter(repeat_loader))
assert torch.equal(batch["input_ids"], repeat_batch["input_ids"])
valid_tokens = int(batch["attention_mask"].sum().item())
all_slots = batch["attention_mask"].numel()
ignore_pad_ok = bool((batch["labels"][~batch["attention_mask"]] == IGNORE_INDEX).all())
shift_logits_shape = (batch["input_ids"].size(0), batch["input_ids"].size(1) - 1, 128)
shift_labels_shape = tuple(batch["labels"][:, 1:].shape)
supervised_tokens = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum().item())

lengths = [len(x) for x in sequences]
plain_waste = padding_waste(lengths, batch_size=3)
bucketed_waste = padding_waste(sorted(lengths), batch_size=3)

sampler_rank0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=True, seed=13)
sampler_rank1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=True, seed=13)
sampler_rank0.set_epoch(0)
sampler_rank1.set_epoch(0)
rank0_indices = list(iter(sampler_rank0))
rank1_indices = list(iter(sampler_rank1))

nondivisible_dataset = ToyCausalDataset(sequences[:5])
pad_rank0 = DistributedSampler(nondivisible_dataset, num_replicas=2, rank=0, shuffle=False)
pad_rank1 = DistributedSampler(nondivisible_dataset, num_replicas=2, rank=1, shuffle=False)
trim_rank0 = DistributedSampler(
    nondivisible_dataset, num_replicas=2, rank=0, shuffle=False, drop_last=True
)
trim_rank1 = DistributedSampler(
    nondivisible_dataset, num_replicas=2, rank=1, shuffle=False, drop_last=True
)
pad_indices = [list(pad_rank0), list(pad_rank1)]
trim_indices = [list(trim_rank0), list(trim_rank1)]
assert not (set(rank0_indices) & set(rank1_indices))
assert sorted(rank0_indices + rank1_indices) == list(range(len(dataset)))
assert sorted(set(pad_indices[0]) & set(pad_indices[1])) == [0]
assert trim_indices == [[0, 2], [1, 3]]

print("batch_input_shape=", tuple(batch["input_ids"].shape))
print("batch_lengths=", batch["lengths"].tolist())
print("attention_mask=", batch["attention_mask"].int().tolist())
print("valid_token_ratio=", round(valid_tokens / all_slots, 3))
print("ignore_pad_ok=", ignore_pad_ok)
print("supervised_tokens_after_shift=", supervised_tokens)
print("shift_logits_shape=", shift_logits_shape)
print("shift_labels_shape=", shift_labels_shape)
print("padding_waste_plain=", plain_waste)
print("padding_waste_bucketed=", bucketed_waste)
print("rank0_indices=", rank0_indices)
print("rank1_indices=", rank1_indices)
print("distributed_overlap=", sorted(set(rank0_indices) & set(rank1_indices)))
print("nondivisible_sampler_indices=", pad_indices)
print("nondivisible_sampler_overlap=", sorted(set(pad_indices[0]) & set(pad_indices[1])))
print("nondivisible_drop_last_indices=", trim_indices)
```
代码片段，已验证（2026-11-09；Python 3.12.9、PyTorch 2.12.0+cpu，CPU）。

期望输出类似：

```text
batch_input_shape= (3, 6)
batch_lengths= [2, 6, 3]
attention_mask= [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
valid_token_ratio= 0.611
ignore_pad_ok= True
supervised_tokens_after_shift= 8
shift_logits_shape= (3, 5, 128)
shift_labels_shape= (3, 5)
padding_waste_plain= 0.303
padding_waste_bucketed= 0.148
rank0_indices= [4, 0, 2]
rank1_indices= [3, 5, 1]
distributed_overlap= []
nondivisible_sampler_indices= [[0, 2, 4], [1, 3, 0]]
nondivisible_sampler_overlap= [0]
nondivisible_drop_last_indices= [[0, 2], [1, 3]]
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

先固定几个样本，检查字段是否齐全、token 是否为整数、长度是否在允许范围内、空样本如何处理、label 是否与输入共享同一 tokenizer 和版本。map-style 样本还要确认 Dataset key 到 record id 的映射稳定；IterableDataset 则记录 shard、worker 和 epoch。单样本不稳定时，不要直接调 `num_workers` 或模型 batch size。

### 4.16.2 单 batch contract

检查每个字段的 shape、dtype、device、有效 token 数和 padding 位置。对 causal LM，随机抽取一行有效位置，确认 `input_ids[:, :-1]` 的预测目标确实是 `labels[:, 1:]`；确认 `labels == -100` 的位置不会进入 loss；确认 attention mask 的 key/query 方向符合模型实现。

### 4.16.3 单 epoch contract

记录样本 id、有效 token 数、padding ratio、空/异常样本数和 batch 等待时间。若使用 sampler，检查每个 rank 的数量、交集、补样本和丢样本；同时区分 `DistributedSampler.drop_last` 与 `DataLoader.drop_last`。若使用流式 dataset，记录 shard、worker 和 record id，并比较每个 worker 实际消费的条数。这样才能区分“数据真的缺了”与“统计只看到了一个 rank”。

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
4. 以为 `shuffle=True` 就能解决分布式数据切分。多卡 map-style 通常要使用 `DistributedSampler` 并在每个 epoch 调用 `set_epoch()`；IterableDataset 要自己组合 rank 与 worker 分片。
5. 以为 padding token 可以随便算进 loss。多数任务里 padding 应该被 mask 掉。
6. 以为 num_workers 越大越好。它需要和 CPU、IO 和预处理成本一起权衡。
7. 以为 packing 只是把序列拼起来。其实还要处理样本边界、labels 和 mask。
8. 以为 `DistributedSampler(drop_last=True)` 和 `DataLoader(drop_last=True)` 是同一个开关。前者先按 rank 截掉数据集尾部，后者再按每个 rank 的 batch 丢弃不足一批的样本。
9. 以为 `in_order=False` 一定能提高吞吐，或固定 DataLoader generator 就能复现所有随机增强。前者有顺序/分布风险；后者还要播种 worker 内的其他随机库并固定 sampler epoch。

## 4.18 章末练习

1. 写一个 `Dataset`，返回文本和长度两个字段。
2. 写一个 `collate_fn`，把变长 token 序列 pad 成 batch。
3. 给 padding 部分构造 `attention_mask` 和 `labels=-100`。
4. 用 `DataLoader` 读取你的数据集，打印一个 batch 的 shape。
5. 把 `shuffle=True` 改成 `DistributedSampler` 形式，并在伪代码里写出 `set_epoch`。
6. 写一个简单的 length bucket 函数，比较 bucket 前后 batch 的平均 padding 比例。

7. 实现一个 `IterableDataset`，用 `get_worker_info()` 把整数范围分给多个 worker，验证同一 epoch 内没有重复值；再把 rank 与 worker 合成全局 shard id。
8. 令 `N=10`、`world_size=3`、`shuffle=False`，逐步写出 `DistributedSampler` 在补齐和 `drop_last=True` 下三个 rank 的索引；再设每 rank `batch_size=2`，解释 DataLoader 的 `drop_last` 还会丢掉什么。
9. 构造两个有效 token 数差异很大的 micro-batch，比较“每个 batch 的 mean loss 再平均”和“按有效 token 总数归一化”的梯度权重。
10. 对一个 DataLoader 记录 batch 等待时间、有效 token/s、padding ratio 和 GPU 计算时间；以 4 workers、`prefetch_factor=2`、每批 64 MiB 估算排队 batch 的数据量，再判断瓶颈在读取、collate、搬运还是模型。
11. 把多个样本 packing 到固定长度，设计 document boundary mask，使后一个样本不能看到前一个样本；说明只加入 `eos_id` 为什么不够。

## 4.19 资料与证据边界

本章接口事实固定按 PyTorch 2.15 官方文档核验。它们支持公开 API 语义，不等于目标平台、真实数据或训练吞吐已经测试：

1. [Dataset 类型、map/iterable 约定与 `__getitems__`](https://docs.pytorch.org/docs/2.15/data.html#dataset-types)
2. [`DataLoader`、自动 batching 与 `collate_fn`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.DataLoader)
3. [`default_collate`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.default_collate)
4. [单进程/多进程加载、worker 内存与随机种子](https://docs.pytorch.org/docs/2.15/data.html#single-and-multi-process-data-loading)
5. [`DistributedSampler`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.distributed.DistributedSampler)
6. [`get_worker_info`](https://docs.pytorch.org/docs/2.15/data.html#torch.utils.data.get_worker_info)
7. [PyTorch 2.15 数据加载优化教程](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)，用于核对 `in_order`、`__getitems__` 和测量边界；其速度数字仅对应教程环境。

文档说明的 DataLoader 参数不保证在所有系统上的性能收益。IterableDataset 多 worker 时 `drop_last` 会作用于每个 worker 副本，`len(dataloader)` 只是估算；`in_order=False` 可能损害可复现性并在不平衡数据上造成顺序偏斜。目标数据源上的重复率、吞吐、内存和随机性仍须另测。padding ratio 是教学指标，effective tokens/s 必须明确有效 token 的定义；packing 的跨样本注意力隔离取决于模型和 mask 实现，不由 DataLoader 自动完成。

## 4.20 本章总结

Dataset 负责单样本逻辑，DataLoader 负责 batch 组装和并行加载，collate_fn 负责把变长或复杂样本整理成训练可用的张量结构。对于大模型训练来说，最常见的数据处理任务是 padding、attention mask、labels 对齐和分布式采样。

要记住的主线是：

1. map-style Dataset 通过 key 取样，IterableDataset 通过迭代产样；流式多 worker/rank 必须显式分片。
2. 变长样本通常在 collate_fn 里处理，而不是在 Dataset 里硬 pad。
3. padding、labels 和 attention_mask 必须一起设计。
4. shuffle、sampler 和 batch_sampler 是不同层次的采样控制。
5. sampler 的补齐与 DataLoader 的 batch 丢尾是两层决策；IterableDataset 的估算长度和逐 worker 丢尾要单独检查。
6. num_workers、pin_memory、persistent_workers、prefetch_factor 和 in_order 影响内存、顺序与延迟，是否提速要用测量确认。
7. packing 能减少 padding，但必须实现样本边界的 label、position 与 attention 隔离。

下一章会进入训练循环工程，重点讲一个完整训练 step 应该怎么写，怎么组织 optimizer、scheduler、梯度裁剪、日志、验证、checkpoint 和异常恢复。
