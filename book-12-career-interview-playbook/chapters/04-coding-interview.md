# 第四章：Coding Interview：把模型语义写成可靠程序

大模型算法岗的 coding 面试，表面上是在共享编辑器里写几段 Python 或 PyTorch，实际上是在观察一件更难的事：你能不能把一个含糊的模型想法，翻译成输入输出明确、形状自洽、数值稳定、可以验证的程序。

只刷传统算法题，很容易忽略大模型代码中最危险的错误。一个函数可能能够运行，却把 prompt 也算进了偏好损失；一个 attention 可能能够返回正确的 shape，却让 token 看到了未来；一个训练循环可能让 loss 下降，却因为标签没有错位而学会复制当前 token；一个缓存可能提高命中率，却把不同租户的答案混在一起。

因此，本章不把 coding 当作题型目录，而把它当作一门小型的程序设计课程。我们从程序契约开始，逐步走到数据结构、张量语义、自回归损失、mask、multi-head attention、位置编码、采样、生成缓存、DPO、归一化、调试、数值稳定性、性能和测试。代码以教学实现为主；涉及生产系统时，会明确哪些地方需要交给 PyTorch 的融合算子、推理引擎或更完整的工程组件。

读者可以把本章看成两条同时推进的路线。初学者先借助具体的 shape 和小例子建立直觉，知道每一行代码在计算图中的位置；有经验的读者则应继续追问契约是否完整、梯度是否流向正确、复杂度是否符合服务目标，以及这个实现的结果能否被测试和解释。

## 4.1 Coding 面试考的不是代码长度

### 4.1.1 同一个错误会同时伤害语义和工程

考虑一个语言模型训练函数。输入 logits 的形状是 [B, T, V]，其中 B 是 batch size，T 是序列长度，V 是词表大小。标签的形状是 [B, T]。如果直接把同一位置的 logits 和 labels 对齐，程序在形状上可能没有问题，但训练目标已经变了。

在标准的因果语言模型中，位置 t 的输出用来预测位置 t+1 的 token。也就是说，logits 的最后一个时间位置没有下一个 token 可以预测，而 labels 的第一个位置没有前文输出可以对应。错位一位不是代码风格问题，而是监督信号的定义错误。

类似的错误还有很多：

- 把 [B, T, H] 当成 [B, H, T]，矩阵乘法仍可能在某些尺寸下运行，但每个维度含义已经交换。
- 把 padding mask 和 causal mask 混在一起，模型会在不该看的位置取信息，或者把一整行分数都屏蔽成 NaN。
- 把 log probability 当成 probability 再调用 multinomial，采样函数会收到负数。
- 在 DPO 中把 prompt 的 log probability 也纳入 chosen/rejected 比较，长度和模板差异会被误当成偏好信号。
- 在 decode 阶段把已经缓存的 key 和 value 重复计算，答案也许正确，但吞吐和显存成本完全不同。

这些问题的共同点是：程序表面完成了运算，内部不变量却被破坏了。成熟的 coding 解答首先会写清不变量，然后用代码实现它。

### 4.1.2 四个问题组成一份最小契约

一个模型小模块至少应在开始编码前回答四个问题。

第一，数据是什么。要写明张量的 rank、每个轴的含义、dtype、device，以及是否允许变长。

第二，结果是什么。要写明返回 shape、返回的是 logits、probability、log probability 还是 loss，以及 reduction 是 sum、mean 还是逐 token 返回。

第三，哪些位置有效。padding、prompt、未来 token、已经结束的序列，可能分别由不同 mask 控制。mask 的 True 到底代表保留还是屏蔽，也必须固定。

第四，资源和梯度如何处理。函数是否参与反向传播，是否会建立大中间张量，是否要求 cache，是否允许原地修改，都会影响实现。

例如，自回归语言模型损失可以写成这样一份契约：

| 项目 | 约定 |
| --- | --- |
| logits | 浮点张量，形状 [B, T, V] |
| labels | long 张量，形状 [B, T]；无效标签为 -100 |
| 对齐 | logits[:, t] 预测 labels[:, t + 1] |
| reduction | 只在有效的目标 token 上取平均 |
| 返回值 | 标量浮点 loss，保留对 logits 的梯度 |
| 边界 | 没有有效目标 token 时不返回 NaN |

契约不是多余的文档。它使实现、测试和口头解释使用同一套语言。后文的每个小模块都会先说明这样的契约。

### 4.1.3 现场表达应当跟随程序的因果顺序

在现场实现时，最有价值的说明通常不是背诵 API，而是把决策与原因连起来。先确认输入输出，再提出最简单的正确方案；写一个核心路径后，用最小例子手动检查；最后说明复杂度、资源和可替换的工程实现。

这并不要求把每一行代码都念出来。真正需要说清楚的是那些会改变语义的地方：为什么要错位、为什么 mask 要扩展到这个形状、为什么这里必须用 long、为什么一个缓存可以复用而另一个不能复用。这样的说明也方便在程序出错时快速回到假设，而不是随机改动。

## 4.2 先学会读张量：shape 是代码中的名词

### 4.2.1 用字母给每个轴命名

大模型代码中的数字通常没有独立意义。32 可能是 batch size，也可能是 head 数；2048 可能是序列长度，也可能是 hidden size。只看数字而不看轴的含义，是 shape bug 的起点。

本章使用以下记号：

| 记号 | 含义 | 常见例子 |
| --- | --- | --- |
| B | batch size | 一次处理的序列数 |
| T | query 序列长度 | 当前输入 token 数 |
| S | key/value 序列长度 | 可能包含历史 cache |
| V | vocabulary size | 输出 logits 的类别数 |
| H | attention head 数 | query 的 head 数 |
| H_kv | key/value head 数 | GQA 或 MQA 中较小 |
| D | hidden size | 模型主表示宽度 |
| d | head dimension | 通常 D / H |

一个 decoder block 中常见的形状链是：

~~~text
[B, T, D]
    -> q_proj, k_proj, v_proj
[B, T, H * d]
    -> reshape + transpose
[B, H, T, d]
    -> attention
[B, H, T, d]
    -> transpose + merge
[B, T, D]
    -> lm_head
[B, T, V]
~~~

在写矩阵乘法前，先确认最后两个轴。q 与 k 的最后一个轴都是 d，因此 k 需要交换最后两个轴，得到 [B, H, d, S]；两者相乘后才得到 [B, H, T, S]。这比死记一行 matmul 更可靠。

### 4.2.2 rank、dtype 和 device 是同一份接口

一个张量的 shape 只描述了大小，不描述它能否参与计算。语言模型里有几条应当形成条件反射的规则：

- input_ids 通常是 torch.long，因为它们被 embedding 或 gather 当作索引。
- labels 通常也是 torch.long，因为 cross entropy 的类别目标不是浮点概率。
- logits、hidden、q、k、v 通常是浮点张量。
- attention mask 和 loss mask 最清楚的表示是 torch.bool。
- 参与同一算子的张量通常必须在同一 device；CPU 上的 mask 不能直接和 CUDA 分数相加或填充。
- 混合精度下，累加和归一化的中间结果有时需要用 float32。

下面这个小工具只做形状和类型检查，不替代真正的业务验证：

~~~python
from __future__ import annotations

import torch


def require_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, int]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have rank 3, got {logits.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have rank 2, got {labels.shape}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"sequence dimensions disagree: logits={logits.shape}, "
            f"labels={labels.shape}"
        )
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")
    batch, seq_len, vocab = logits.shape
    return batch, seq_len, vocab
~~~

一个好的检查函数应当指出违反了哪条契约。只写 assert logits.ndim == 3 在本地可以帮助自己，但在更大的系统中，带有实际 shape 的错误信息更容易定位数据管道问题。

### 4.2.3 reshape、view、transpose 和 permute 的区别

view 只是在已有存储上重新解释形状。它要求底层 stride 能够支持这种解释，所以 transpose 或 permute 后直接 view 可能报错。contiguous 会按照当前逻辑顺序建立一份连续存储，随后才可以安全地 view，但这可能产生额外内存和拷贝。

reshape 会尽量复用原存储，必要时自行创建连续副本，因此通常更方便；不过它并不意味着没有拷贝。性能敏感代码仍应知道数据是否连续。

transpose 通常交换两个轴，permute 可以任意重排多个轴。它们一般只改变视图和 stride，不会立即搬动数据。比如：

~~~python
# x: [B, T, H, d]
x_heads = x.permute(0, 2, 1, 3)
# x_heads: [B, H, T, d]

# 后续若需要按 [B, T, H, d] 的连续顺序合并：
x_back = x_heads.transpose(1, 2).contiguous().reshape(
    x.size(0), x.size(1), -1
)
~~~

这里的 contiguous 不是装饰。没有它，某些版本和某些输入布局下的 view 会失败；即使用 reshape 让程序跑通，也可能暗中复制一个很大的张量。面试中说出“我在这里需要连续存储，代价是一份潜在的拷贝”，比单纯背 API 更能体现工程判断。

### 4.2.4 expand 和 repeat 不是同一个操作

expand 通过 stride 为零的视图广播一个尺寸为 1 的轴，通常不复制数据；repeat 会真正复制数据。比如把一个 [B, 1, T] 的 mask 扩展到 head 维，通常只需要 expand 或让算子自动 broadcast，不应为每个 head 复制一份。

expand 出来的多个位置可能指向同一块存储，因此不能对它做会改变值的原地写入。若需要独立可写的副本，使用 repeat 或 clone，并承担相应的内存成本。

broadcast 的规则可以概括为：从最后一个轴向前对齐，两个尺寸相等，或者其中一个为 1，才可以广播。attention 中的 [B, 1, 1, S] key mask 可以广播到 [B, H, T, S]；[B, T] 则不能在所有情形下自动变成你想要的形状，最好显式写成 [B, 1, 1, T]。

### 4.2.5 gather 表达“每行取一个类别”

给定 log_probs [B, T, V] 和 labels [B, T]，想取每个位置对应标签的 log probability，正确的索引张量要增加最后一维：

~~~python
log_probs = torch.randn(2, 4, 100).log_softmax(dim=-1)
labels = torch.randint(0, 100, (2, 4))

selected = log_probs.gather(
    dim=-1,
    index=labels.unsqueeze(-1),
).squeeze(-1)
# selected: [B, T]
~~~

gather 的 index 必须和输入在非索引轴上相容，index 的 dtype 必须是 long。labels 中的 -100 不能直接拿来 gather，因为它不是一个有效的词表索引；正确做法是先把无效位置替换成 0，再用 mask 清除对应结果。这个细节会在 DPO 和 sequence log probability 中反复出现。

## 4.3 Python：先把数据处理写成可追溯的程序

### 4.3.1 字符长度不是 token 长度

训练数据清洗常常从 JSONL 开始。一个初学者版本只统计 len(text)，这对快速发现空样本有用，但它统计的是 Python 字符数，不是 tokenizer token 数。中文、代码、URL 和混合语言的字符长度与 token 长度关系不同；如果任务真正受上下文窗口和训练 token 预算约束，就必须使用目标 tokenizer 重新统计。

下面的代码故意把两件事分开：它用字符长度做一个不依赖外部模型的初筛，同时记录坏行、字段缺失和精确重复。真实数据量很大时，seen 集合应换成外部去重存储或分桶哈希，不能无条件把全部文本留在进程内存中。

~~~python
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any


def _text_key(obj: dict[str, Any]) -> str | None:
    prompt = obj.get("prompt", "")
    response = obj.get("response", "")
    if (
        not isinstance(prompt, str)
        or not isinstance(response, str)
        or not prompt.strip()
        or not response.strip()
    ):
        return None
    return prompt + "\n" + response


def analyze_jsonl(path: str) -> dict[str, Any]:
    total_lines = 0
    valid_records = 0
    malformed_lines = 0
    non_object_lines = 0
    missing = Counter()
    lengths: list[int] = []
    seen: set[str] = set()
    duplicate_records = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            total_lines += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue
            if not isinstance(obj, dict):
                non_object_lines += 1
                continue

            valid_records += 1
            for key in ("prompt", "response"):
                value = obj.get(key)
                if not isinstance(value, str) or not value.strip():
                    missing[key] += 1

            text = _text_key(obj)
            if text is None:
                continue
            lengths.append(len(text))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                duplicate_records += 1
            else:
                seen.add(digest)

    return {
        "total_lines": total_lines,
        "valid_records": valid_records,
        "malformed_lines": malformed_lines,
        "non_object_lines": non_object_lines,
        "missing": dict(missing),
        "duplicate_records": duplicate_records,
        "char_length_mean": mean(lengths) if lengths else 0.0,
        "char_length_median": median(lengths) if lengths else 0.0,
    }
~~~

这里的 digest 只是为了减少内存占用和避免把完整文本暴露在日志中。重复统计只针对 prompt 和 response 都有效的记录；字段缺失样本会单独进入 missing，而不会因为都映射成空字符串而互相误报。digest 仍然只代表“规范化规则下的精确重复”，不代表语义重复，也不代表样本已经通过质量审核。若 prompt 前后空格、模板版本和系统消息会影响训练，规范化规则必须在数据说明中记录。

### 4.3.2 数据结构题要把抽象结构连回模型系统

哈希表、堆、队列和排序不是与大模型无关的基础题。它们分别对应 token 统计、Top-K 候选、生成队列、检索结果合并和 batch 调度。重要的不是在每个题目里强行套一个模型名词，而是能识别数据结构真正解决的约束。

例如，计算频率最高的 k 个 token 或字符串，可以先统计，再用大小为 k 的堆保持候选。下面的实现对频率相同时使用字典序，便于测试和复现：

~~~python
from collections import Counter
import heapq


def top_k_frequent(items: list[str], k: int) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    counts = Counter(items)
    # key 越小越应该出现在结果中：频率高优先，字典序小优先。
    return heapq.nsmallest(
        min(k, len(counts)),
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
~~~

假设输入长度为 n，不同元素数为 m。计数是 O(n)，堆选择是 O(m log k)，结果排序已经由 nsmallest 的 key 约定确定。若 k 接近 m，完整排序 O(m log m) 可能更简单；复杂度分析不是为了背一个最优答案，而是为了让实现选择与输入规模相称。

在检索系统中，如果同时按相关性、文档版本和权限排序，就不能只说“用一个堆”。需要先定义比较关系：哪些字段是硬约束，哪些字段只用于 tie-break；权限过滤是在候选进入堆之前还是之后；版本冲突如何处理。数据结构只有放进完整契约，才会产生正确的系统行为。

### 4.3.3 生成器和流式处理解决的是峰值内存

读取大文件时，list comprehension 会一次性保留全部记录；生成器则把“产生一个样本”和“消费一个样本”连接起来。它不自动降低总计算量，但可以降低峰值内存，并让坏样本处理和统计更容易分阶段进行。

~~~python
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not an object")
            yield value
~~~

这里选择在遇到坏行时抛出异常，而不是静默跳过，是因为数据集构建和线上日志分析的责任不同。离线探索可以收集坏行继续跑，发布训练集前则通常应该让数据质量问题显式失败。一个好的答案会先说明场景，再选择错误策略。

## 4.4 自回归语言模型损失：先对齐，再求概率

### 4.4.1 一位错位为何决定了训练目标

给出序列：

~~~text
<bos> 机器 学习 很 有趣
~~~

在 teacher forcing 训练中，模型读到 <bos> 后预测“机器”，读到“机器”后预测“学习”，以此类推。若把输入 token 记为 x_0 到 x_{T-1}，训练目标是：

~~~math
\mathcal{L}(\theta)
= - \frac{1}{|M|}
\sum_{t=1}^{T-1}
\mathbf{1}[t \in M]
\log p_\theta(x_t \mid x_{<t})
~~~

其中 M 是有效目标位置集合，theta 是模型参数；这个公式要求 \(|M|>0\)。代码实现中，logits[:, 0:T-1] 对应 labels[:, 1:T]。最后一个 logits 没有被使用，是因为输入序列没有给出它之后的真实 token。若一个 batch 没有任何有效目标，工程实现可以返回带梯度的零作为显式的“无训练信号”约定，但不能把它解释成真正的平均损失。

用表格看更清楚：

| 模型读取的位置 | 它应该预测的目标 |
| --- | --- |
| <bos> | 机器 |
| 机器 | 学习 |
| 学习 | 很 |
| 很 | 有趣 |
| 有趣 | 不参与本条序列的 next-token loss |

如果训练数据已经单独准备好了 next-token labels，就不能再次 shift。代码本身无法猜测 labels 的约定，所以必须把“labels 是原序列还是已经错位的目标”写入接口。

### 4.4.2 padding、prompt 和无效目标不是同一个概念

padding token 在 batch 对齐时出现，它通常不应贡献 loss；prompt token 在 SFT 或 preference training 中可能只是条件，不应被当成需要模型模仿的 completion；未来 token 则是 attention 层面的不可见位置。三者都可能使用 mask，但它们处在不同的计算阶段。

可以这样区分：

| 名称 | 作用位置 | 被屏蔽的对象 |
| --- | --- | --- |
| causal mask | attention score | 未来的 key/value |
| padding/key mask | attention score | 不存在的上下文 token |
| query mask | attention 输出或后处理 | 不需要产生表示的 query |
| loss mask | logits 与 labels 对齐后 | 不计入训练目标的 token |

把 attention mask 传给 loss，或者把 loss mask 误传给 attention，程序可能仍然运行，但模型学习到的对象已经改变。

### 4.4.3 一个带边界检查的实现

下面的实现假定 labels 是与输入序列等长的原始 token 序列，-100 表示不计入损失。它在计算前检查 rank、dtype、词表范围，并在整批没有有效目标时返回带梯度的零，而不是让 mean reduction 产生 NaN。

~~~python
from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    batch, seq_len, vocab = require_logits_and_labels(logits, labels)
    del batch

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    valid_count = int(valid.sum().item())

    if valid_count == 0:
        # 保留计算图，使调用方仍可安全执行 backward。
        return shift_logits.sum() * 0.0

    active_labels = shift_labels.masked_select(valid)
    if active_labels.numel():
        if int(active_labels.min()) < 0 or int(active_labels.max()) >= vocab:
            raise ValueError("active labels contain an invalid vocabulary index")

    per_token = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(shift_labels)
    return per_token.masked_select(valid).mean()
~~~

这里使用 reshape 而不是直接对 permute 后的张量 view，是为了避免把连续性假设藏在调用者身上。F.cross_entropy 接收 [N, C] 的分类输入，所以 [B, T-1, V] 要展平为 [B(T-1), V]；类别标签则展平为 [B(T-1)]。

### 4.4.4 loss 能下降，不代表目标正确

一个常见故障是 loss 下降得很快，生成却出现重复、复制输入或对模板变化极其敏感。排查时不能只看 loss 曲线，至少要检查四件事：

1. 随机抽一个位置，打印它的输入 token、logits 对应的预测位置和 label。
2. 用一个很小的序列手工确认第一、最后一个有效目标。
3. 检查 padding 和 prompt 的 mask 是否真的进入 reduction。
4. 在一个极小数据集上过拟合，确认模型能记住正确的 next-token 关系。

极小数据集过拟合是一个很有用的诊断，但它只能证明梯度路径在这组数据上能工作，不能证明大规模训练的泛化和数据质量。它与真实验证集是两种不同的证据。

## 4.5 Mask：先规定语义，再规定形状

### 4.5.1 causal mask 的含义

对长度为 T 的自回归序列，第 i 个 query 只能访问位置 j <= i 的 key。若约定 True 表示“保留”，因果 mask 是下三角矩阵：

~~~math
C_{i,j} = \mathbf{1}[j \leq i]
~~~

代码如下：

~~~python
import torch


def make_causal_mask(
    query_len: int,
    key_len: int | None = None,
    *,
    past_len: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    if query_len < 0:
        raise ValueError("query_len must be non-negative")
    if key_len is None:
        key_len = past_len + query_len
    if key_len < 0:
        raise ValueError("key_len must be non-negative")
    if past_len < 0:
        raise ValueError("past_len must be non-negative")

    query_positions = (
        torch.arange(query_len, device=device) + past_len
    )[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    return key_positions <= query_positions
~~~

当 query 是带历史 cache 的新 token 时，query 的绝对位置不再是 0 到 query_len-1，因此需要把历史长度传给 past_len。比如 past_len=4、query_len=1、key_len=5 时，唯一的 query 可以访问 0 到 4 的全部 key。忽略这个 offset 会让增量解码的新 token 看不到一部分合法历史。若 key 还包含未来位置，仍需保证 key_len 与 cache 状态一致。

### 4.5.2 key mask 和 query mask 的职责不同

假设一个 batch 中有两条长度不同的序列，短序列末尾用 padding 补齐。key mask 应防止任何有效 query 读取 padding key；query mask 则说明 padding query 的输出不应被下游使用。只用 key mask 不一定能让 padding query 的输出变成零，因为它仍可能对有效 key 做 softmax。

一种清晰的做法是：

1. attention 计算中屏蔽 padding key；
2. attention 输出完成后，用 query mask 将无效 query 的表示清零，或者在 loss 中完全忽略这些位置；
3. 对需要严格零输出的模块，单独测试 padding query。

这也是为什么“attention mask 是一个布尔矩阵”这个说法不够完整。必须说明它控制 query 还是 key，以及 True 的含义。

### 4.5.3 手写 attention：适合解释，不一定适合部署

scaled dot-product attention 的分数和输出为：

~~~math
S = \frac{QK^\mathsf{T}}{\sqrt{d}}
~~~

~~~math
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}(S + A)V
~~~

Q 的形状是 [B, H, T, d]，K 和 V 的 key/value 长度可以是 S，A 是加性 mask；若使用布尔 keep mask，则被屏蔽位置在实现中填入一个足够小的值。除以 sqrt(d) 是为了抑制点积随维度增大而变得过大的趋势，使 softmax 不至于过早饱和。

下面是一个便于在白板上解释的版本。它把 softmax 的归一化放到 float32 中，减少低精度下的溢出风险；生产代码通常优先使用框架提供的 scaled dot-product attention，因为融合实现可以避免显式存储完整的 [B, H, T, S] 分数矩阵。

~~~python
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape [B, H, length, head_dim]")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimensions disagree")
    if q.shape[1] != k.shape[1] or k.shape[1] != v.shape[1]:
        raise ValueError("head dimensions disagree")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions disagree")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v sequence dimensions disagree")
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        raise TypeError("attention inputs must be floating point")
    if not (q.dtype == k.dtype == v.dtype):
        raise TypeError("q, k, and v must use the same dtype")
    if not (q.device == k.device == v.device):
        raise ValueError("attention inputs must be on one device")

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))

    if keep_mask is not None:
        if keep_mask.dtype != torch.bool:
            raise TypeError("keep_mask must be boolean; True means keep")
        if keep_mask.device != scores.device:
            raise ValueError("keep_mask must be on the attention device")
        try:
            expanded = torch.broadcast_to(keep_mask, scores.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"mask {keep_mask.shape} is not broadcastable to {scores.shape}"
            ) from exc
        if not bool(expanded.any(dim=-1).all()):
            raise ValueError("at least one key must remain for every query")
        scores = scores.masked_fill(
            ~expanded,
            torch.finfo(scores.dtype).min,
        )

    weights = F.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
    return torch.matmul(weights, v)
~~~

如果一整行 key 都被屏蔽，softmax 的分母可能变成 0，结果出现 NaN。用 -1e9 替代 -inf 并不能从根本上解决“整行没有合法 key”的逻辑错误；在 fp16 中，常量的表示范围也需要考虑。更可靠的做法是保证每个有效 query 至少有一个合法 key，并对 padded query 单独处理。

### 4.5.4 使用 PyTorch 的融合算子

当前 PyTorch 提供 torch.nn.functional.scaled_dot_product_attention。它可以根据 device、dtype、输入规模和后端选择不同实现，并可能使用 FlashAttention 等融合路径。调用时仍要确认布尔 mask 的语义和版本行为，不能因为函数名相同就跳过契约。

~~~python
def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=keep_mask,
        dropout_p=0.0,
        is_causal=False,
    )
~~~

如果使用 is_causal=True，就不应再传一个相互冲突的 attn_mask；如果 query 和 key 长度不同，还应确认该版本对非方形 causal mask 的定义。面试中的简化实现用于表达算法，工程实现则要以当前框架文档、profile 结果和回归测试为准。

## 4.6 Multi-Head Attention：从线性投影到轴的合并

### 4.6.1 为什么要拆成多个 head

一个 hidden 表示可以通过不同的 query、key、value 投影，在不同子空间中寻找关系。第 h 个 head 只处理 d 维，多个 head 的结果再拼回 D 维。若 D = H * d，计算图是：

~~~text
[B, T, D]
    -> q_proj, k_proj, v_proj
[B, T, H * d]
    -> reshape
[B, H, T, d]
    -> attention
[B, H, T, d]
    -> transpose + merge
[B, T, D]
    -> o_proj
~~~

拆分本身不产生新的参数；参数来自 q、k、v 和输出投影。真正的成本主要由投影矩阵、attention 分数和 KV cache 共同决定。

### 4.6.2 一个最小但完整的实现

~~~python
from __future__ import annotations

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size <= 0 or num_heads <= 0:
            raise ValueError("hidden_size and num_heads must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        if hidden != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden}")
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        if heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(f"unexpected attention output shape: {x.shape}")
        return (
            x.transpose(1, 2)
            .contiguous()
            .reshape(batch, seq_len, self.hidden_size)
        )

    def forward(
        self,
        x: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, D]")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        attended = manual_attention(q, k, v, keep_mask=keep_mask)
        return self.o_proj(self._merge_heads(attended))
~~~

这段实现先保证每一个轴都能解释，再谈优化。它没有 dropout、RoPE、cache、GQA 和 FlashAttention，因此不能直接当作生产 decoder。这样的限制反而适合教学：基础实现的边界明确，后续每一项优化都可以问“它改变了哪个轴、哪个存储、哪个计算阶段”。

### 4.6.3 GQA 和 MQA 改变的是 K/V 的头数

标准 MHA 使用 H 个 query head、H 个 key head 和 H 个 value head。Grouped-Query Attention 让多个 query head 共享一个 key/value head 组，Multi-Query Attention 更进一步让所有 query head 共享一个 K/V head。

当 batch、缓存长度、层数和 head_dim 固定时，KV cache 的元素量近似为：

~~~math
N_{\mathrm{KV}}
= 2 \cdot L \cdot B \cdot T
\cdot H_{\mathrm{kv}} \cdot d
~~~

2 来自 key 和 value，L 是层数。把 H_kv 从 H 降到较小的数，主要节省 decode 阶段的缓存和内存带宽，但可能改变质量、并行和模型结构约束。代码题中，如果只要求写 MHA，不应未经说明擅自把 k/v 复制或共享；如果要讨论 GQA，先把 H、H_kv 和每组 query 数写出来。

## 4.7 位置编码与增量解码：位置是 cache 契约的一部分

### 4.7.1 位置编码不是一个可以随意替换的装饰

attention 本身只比较 token 表示，不知道 token 的先后顺序。位置编码把顺序信息注入计算图。绝对位置 embedding、相对位置 bias、RoPE 和 ALiBi 都可以表达位置关系，但它们的参数、外推行为和 cache 接口不同。

因此，一个通用的 attention 模块不应假定所有模型都使用 RoPE。实现可以把“得到带位置的 q/k”作为一个可替换策略；模型配置则负责说明实际策略、最大训练长度和推理时的缩放方法。

### 4.7.2 RoPE 的几何直觉

RoPE 把 q 和 k 的二维子空间成对旋转，旋转角度随位置变化。对第 i 个位置和频率 omega，旋转矩阵可以写成：

~~~math
R(\theta_i)
=
\begin{bmatrix}
\cos \theta_i & -\sin \theta_i \\
\sin \theta_i & \cos \theta_i
\end{bmatrix}
~~~

它的一个重要性质是，旋转后 q 与 k 的内积包含相对位置信息，而不是简单地把一个位置向量加到 hidden 上。不同实现对偶数/奇数维的排列可能不同，代码必须和 checkpoint 的训练约定一致。

下面使用“前半维与后半维配对”的教学约定：

~~~python
from __future__ import annotations

import torch


def build_rope_cache(
    max_seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_seq_len < 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError("max_seq_len must be non-negative and head_dim even")
    if base <= 0 or not torch.isfinite(torch.tensor(base)):
        raise ValueError("base must be a finite positive number")

    indices = torch.arange(
        0,
        head_dim,
        2,
        device=device,
        dtype=torch.float32,
    )
    inverse_frequency = 1.0 / (base ** (indices / head_dim))
    positions = torch.arange(
        max_seq_len,
        device=device,
        dtype=torch.float32,
    )
    angles = torch.outer(positions, inverse_frequency)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    first, second = x[..., :half], x[..., half:]
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError("x must have shape [B, H, T, d]")
    if position_ids is None:
        selected_cos = cos[:, :, : x.size(-2), :]
        selected_sin = sin[:, :, : x.size(-2), :]
    else:
        if (
            position_ids.ndim != 1
            or position_ids.numel() != x.size(-2)
            or position_ids.dtype != torch.long
            or position_ids.device != cos.device
        ):
            raise ValueError("position_ids must have shape [T]")
        selected_cos = cos.index_select(2, position_ids)
        selected_sin = sin.index_select(2, position_ids)

    x_float = x.float()
    result = x_float * selected_cos + _rotate_half(x_float) * selected_sin
    return result.to(dtype=x.dtype)
~~~

如果增量解码已经生成了 past_len 个 token，新 token 的 position_ids 应从 past_len 开始，而不是每一步重新从 0 开始。已经写入 cache 的 k 不应在下一步再次旋转；它们已经带着自己的绝对位置。这个错误有时不会造成 shape 报错，却会让长序列的质量逐渐下降，因此必须用“全量前向”和“逐 token cache 前向”做数值对齐测试。

### 4.7.3 位置外推不能只改一个常数

把 max_position_embeddings 改大，或把 RoPE base 改成另一个数，并不自动等价于支持更长上下文。训练长度、角度频率、checkpoint 约定、attention kernel、位置缩放、数据分布和评估任务共同决定长上下文行为。

在 coding 题中，先实现与配置一致的短序列版本；若讨论长上下文，要把“模型能接收更长输入”和“在更长输入上保持任务能力”分开验证。代码能索引到第 100000 个位置，只能证明索引和内存路径存在，不能证明模型在该位置理解证据。

## 4.8 采样：从 logits 到下一个 token

### 4.8.1 logits、probability 和 log probability

模型输出 logits [B, V]，它们不是概率，也不要求和为 1。softmax 后才得到概率：

~~~math
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
~~~

数值稳定实现会先减去每行最大值，或者直接使用框架的 softmax/log_softmax。temperature 对 logits 做缩放：

~~~math
p_i(T)
=
\operatorname{softmax}(z_i / T)
~~~

当 T 增大，分布通常变平；当 T 减小时，分布通常更集中。T = 0 在工程实现中通常作为 greedy argmax 的特殊分支，而不是把除法真的执行。

top-k 保留固定数量的候选；top-p 按排序后的累计概率保留最小候选集合。它们都是对分布支持集的截断，不能改变已经被模型赋予的候选排序之外的事实。

### 4.8.2 一个边界明确的 top-k/top-p 实现

~~~python
from __future__ import annotations

import torch
import torch.nn.functional as F


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, V]")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("logits cannot contain NaN or positive infinity")
    if not bool(torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError("each row needs at least one finite logit")

    filtered = logits
    vocab = logits.size(-1)
    floor = torch.finfo(logits.dtype).min

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        k = min(top_k, vocab)
        _, indices = torch.topk(filtered, k=k, dim=-1)
        keep = torch.zeros_like(filtered, dtype=torch.bool)
        keep.scatter_(dim=-1, index=indices, value=True)
        filtered = filtered.masked_fill(~keep, floor)

    if top_p is not None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        # Keep the first token that crosses the probability threshold.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, floor)
        filtered = torch.full_like(filtered, floor)
        filtered.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_logits,
        )

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, V]")
    if not torch.isfinite(torch.tensor(temperature)) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if temperature == 0:
        if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
            raise ValueError("greedy sampling cannot use NaN or positive infinity")
        if not bool(torch.isfinite(logits).any(dim=-1).all()):
            raise ValueError("each row needs at least one finite logit")
        return torch.argmax(logits, dim=-1)

    scaled = logits / temperature
    filtered = filter_logits(scaled, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered.float(), dim=-1)
    return torch.multinomial(
        probs,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)
~~~

实现 top-k 时使用 topk 返回的索引，而不是用“低于第 k 大值就屏蔽”的阈值，可以保证在并列 logits 时最多保留 k 个 token。top-p 的移位操作则保证至少保留累计概率阈值前的第一个 token。两者组合时先 top-k 还是先 top-p 会影响结果，代码应固定顺序并在配置中记录。

### 4.8.3 greedy、temperature 和随机种子

greedy 选择最大 logit，对重复和确定性调试很有用，但不等于质量一定最高。随机采样需要概率非负且总和为正；不能把 log_softmax 的输出直接传给 multinomial。temperature 只是改变分布尖锐程度，不能替代安全策略、停止条件和内容过滤。

随机种子也不是完整的复现保证。设备、并行归约、算子后端、采样 generator 的 device、模型 dropout 状态和版本都可能影响结果。一个可复现的采样实验至少应记录模型版本、tokenizer、生成参数、随机种子、输入文本和输出截断规则。

### 4.8.4 生成循环的状态比一个采样函数复杂

真正的生成过程还要维护已生成序列、attention mask、KV cache、结束状态和最大长度。下面用接近 Hugging Face 风格的接口展示状态变化；不同模型返回对象的字段可能不同，因此它是教学骨架，不是对所有模型的通用适配器。

~~~python
from __future__ import annotations

import torch


@torch.inference_mode()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a long tensor of shape [B, T]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    elif (
        attention_mask.shape != input_ids.shape
        or attention_mask.dtype != torch.bool
        or attention_mask.device != input_ids.device
    ):
        raise ValueError(
            "attention_mask must be bool, on input_ids.device, and [B, T]"
        )

    model.eval()
    generated = input_ids
    finished = torch.zeros(
        input_ids.size(0),
        dtype=torch.bool,
        device=input_ids.device,
    )
    past_key_values = None

    for _ in range(max_new_tokens):
        # 首次调用处理整个 prompt；之后只把新 token 送入模型。
        model_input = generated if past_key_values is None else generated[:, -1:]
        outputs = model(
            input_ids=model_input,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = getattr(outputs, "past_key_values", None)
        next_logits = logits[:, -1, :]
        next_ids = sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if eos_token_id is not None:
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, eos_token_id),
                next_ids,
            )
            finished = finished | next_ids.eq(eos_token_id)

        generated = torch.cat((generated, next_ids[:, None]), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    input_ids.size(0),
                    1,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
            ),
            dim=1,
        )
        if eos_token_id is not None and bool(finished.all()):
            break

    return generated
~~~

这段代码有一个重要的接口假设：当 past_key_values 不为空时，模型能够把完整 attention_mask 与单个新 token 配合使用，并且能够从 cache 推断新 token 的位置。调用者可以传入初始 padding mask；新生成 token 会追加 True。某些模型需要显式 position_ids，某些模型使用 cache_position；如果模型接口不同，必须按其实现调整。不能把“看起来像 Hugging Face 的参数”当成跨模型协议。

代码中的 torch.cat 每一步都会创建更长的 generated 张量，适合作为面试中的清晰实现；高吞吐引擎会预分配输出缓冲区，并用分页或块状方式管理 KV cache。正确性和性能是两层问题，先用小输入对齐结果，再替换存储策略。

## 4.9 从 token log probability 到 DPO loss

### 4.9.1 序列 log probability 必须只统计目标 token

对一个 completion y = (y_1, ..., y_m)，给定 prompt x，序列 log probability 是：

~~~math
\log \pi_\theta(y \mid x)
=
\sum_{t=1}^{m}
\log \pi_\theta(y_t \mid x, y_{<t})
~~~

在 batch 中，prompt、padding 和 completion 通常混在同一条 labels 序列里。最安全的做法是把不参与比较的位置设置为 -100，然后 gather 前替换成安全索引。

~~~python
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    normalize_by_length: bool = False,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be [B, T, V] and labels must be [B, T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels have different [B, T]")
    if labels.dtype != torch.long:
        raise TypeError("labels must use torch.long")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(ignore_index)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    active_labels = shift_labels.masked_select(valid)
    if active_labels.numel():
        if int(active_labels.min()) < 0 or int(active_labels.max()) >= logits.size(-1):
            raise ValueError("active labels contain an invalid vocabulary index")

    # float32 的 log_softmax 更适合教学和审计；实际大模型可使用融合实现。
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logps = log_probs.gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    sums = (token_logps * valid).sum(dim=-1)

    if not normalize_by_length:
        return sums
    counts = valid.sum(dim=-1)
    return torch.where(
        counts > 0,
        sums / counts.clamp_min(1),
        torch.zeros_like(sums),
    )
~~~

总 log probability 会随着 completion 变长而更负。这个现象不是实现错误，而是乘积概率在 log 空间中的自然结果。若把总和改成平均值，长度偏置会改变，但“每个 token 的平均偏好”和“整段 completion 的概率偏好”也不再是同一个目标。选择哪一种，取决于训练目标、数据构造和既有代码库约定，不能为了让数字好看而临时决定。

### 4.9.2 DPO 比较的是相对 reference 的偏好

DPO 的一个常用形式把 policy 与 reference 的序列 log-ratio 进行比较：

~~~math
r_\theta(x,y)
=
\log \pi_\theta(y \mid x)
-
\log \pi_{\mathrm{ref}}(y \mid x)
~~~

对于 chosen completion y_c 和 rejected completion y_r：

~~~math
\mathcal{L}_{\mathrm{DPO}}
=
-
\mathbb{E}
\left[
\log \sigma
\left(
\beta
\left(
r_\theta(x,y_c)
-
r_\theta(x,y_r)
\right)
\right)
\right]
~~~

beta 控制相对 reference 的偏好差异进入 sigmoid 的尺度。reference 的 log probability 在计算时通常不需要梯度；它不是“一个额外的 chosen 标签”，而是提供 policy 偏移的参照。

~~~python
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive")
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("all sequence log-probability tensors must be [B]")
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("chosen/rejected tensors must have equal batch shapes")

    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    return -F.logsigmoid(logits).mean()
~~~

### 4.9.3 prompt mask 是偏好数据的关键

如果 chosen 和 rejected 共享同一个 prompt，比较 prompt 部分的 log probability在理想情况下会抵消；但真实 batch 中的模板、padding、截断和拼接方式可能不同，直接把整段序列相加会引入额外差异。更稳妥的做法是明确记录 completion 起点，只给 completion token 建立 loss mask。

~~~python
def mask_prompt_tokens(
    labels: torch.Tensor,
    prompt_lengths: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    if labels.ndim != 2 or prompt_lengths.ndim != 1:
        raise ValueError("labels must be [B, T], prompt_lengths must be [B]")
    if labels.size(0) != prompt_lengths.size(0):
        raise ValueError("batch dimensions disagree")
    if prompt_lengths.dtype != torch.long or prompt_lengths.device != labels.device:
        raise TypeError("prompt_lengths must be long and on labels.device")
    if bool((prompt_lengths < 0).any()) or bool(
        (prompt_lengths > labels.size(1)).any()
    ):
        raise ValueError("prompt_lengths must lie in [0, sequence_length]")
    positions = torch.arange(
        labels.size(1),
        device=labels.device,
    )[None, :]
    prompt_mask = positions < prompt_lengths[:, None]
    return labels.masked_fill(prompt_mask, ignore_index)
~~~

一个偏好损失的调试样本至少应包含 prompt 长度、chosen/rejected completion 长度、有效 token 数、policy/reference 的四个 log probability，以及最终 margin。只记录最终 loss，很难判断问题来自模型、mask、长度、reference 还是 preference 数据。

## 4.10 归一化和 MLP：简单模块也有精确边界

### 4.10.1 LayerNorm 和 RMSNorm 处理的对象不同

对 hidden 维度上的向量 x，LayerNorm 会减去均值并除以标准差：

~~~math
\operatorname{LN}(x)
=
\gamma
\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\;+\;\beta
~~~

RMSNorm 不减均值，只按均方根缩放：

~~~math
\operatorname{RMSNorm}(x)
=
\gamma
\odot
\frac{x}
\sqrt{\operatorname{mean}(x^2)+\epsilon}
~~~

其中 mu 和 sigma^2 分别是 x 在最后一个 hidden 维度上的均值和方差，gamma 是逐通道缩放参数，beta 是 LayerNorm 的逐通道偏置，epsilon 用于避免除零。RMSNorm 的计算路径更短，很多 decoder-only 架构采用它；这不表示它在所有任务和所有实现中都必然更好。归一化位置也重要，Pre-Norm 和 Post-Norm 会改变残差路径、梯度传播和稳定性。

~~~python
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.weight.numel():
            raise ValueError("last dimension does not match hidden_size")
        rms_inv = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        ).to(dtype=x.dtype)
        weight = self.weight.to(dtype=x.dtype)
        return x * rms_inv * weight
~~~

这里在 float32 中计算均方根，再转换回输入 dtype，是一种清晰的数值稳定取舍。生产内核可能采用更专门的累加和融合方式，但仍应满足“最后一维归一化、参数逐通道缩放、epsilon 防止除零”这三个语义。

### 4.10.2 SwiGLU 不只是把 GELU 换成 SiLU

一个常见的 gated MLP 先生成两支中间表示，一支作为门，一支作为被调制的值：

~~~math
\operatorname{SwiGLU}(x)
=
W_{\mathrm{down}}
\left(
\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot
(W_{\mathrm{up}}x)
\right)
~~~

~~~python
from torch import nn
import torch
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)
~~~

中间维度不是一个可以随便照抄的常数。不同模型会为了保持参数量、硬件对齐或 checkpoint 兼容而使用不同的 intermediate_size。实现时应该从配置读出它，并在初始化时检查权重形状，而不是用“hidden size 的四倍”作为所有架构的真理。

## 4.11 Debug：把“这里错了”变成可验证的定位过程

### 4.11.1 先分类症状

读训练或推理代码时，可以先把现象归入几类：

| 现象 | 优先检查 |
| --- | --- |
| 立即 shape error | rank、轴顺序、view/permute、batch 对齐 |
| loss 为 NaN | 全屏蔽行、logits/labels、低精度、学习率 |
| loss 不变 | 梯度、参数是否更新、label mask、数据是否为空 |
| loss 下降但生成复制输入 | label shift、训练/推理模板、位置和停止条件 |
| CPU/CUDA 错误 | model、input、mask、cache 的 device |
| 评估结果随机波动 | train/eval、dropout、seed、数据切分 |
| decode 越生成越慢 | cache 未复用、重复 prefill、输出拼接和调度 |
| DPO loss 合理但偏好不变 | completion mask、reference、长度偏置、数据方向 |

分类不是为了省略检查，而是为了把检查顺序与症状匹配。先复现一个最小输入，再只改变一个变量，通常比一次性打开十个开关更快。

### 4.11.2 一个有问题的训练步骤

下面的代码看起来像训练，但至少有五个需要确认的地方：

~~~python
def broken_train_step(model, batch, optimizer):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    return loss.item()
~~~

问题不在于它“少写了几行”，而在于每个省略都隐藏了假设：

- 没有清理旧梯度，除非调用方明确在做梯度累积，否则梯度会跨 step 相加。
- logits 若是 [B, T, V]，cross entropy 的类别轴和样本轴需要明确整理。
- 因果语言模型通常需要 shift；如果 labels 已经错位，则不能重复 shift。
- padding 或 prompt 是否应该计入 loss 没有说明。
- batch 是否和模型在同一 device 没有说明。
- model 可能返回带有 logits 字段的对象，而不是裸 tensor。
- 没有 model.train()，dropout 和 batch-dependent 行为可能处于错误模式。

一个基础但契约更完整的版本是：

~~~python
def train_step(
    model,
    batch: dict[str, torch.Tensor],
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    input_ids = batch["input_ids"].to(device=device, non_blocking=True)
    labels = batch["labels"].to(device=device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids)
    logits = getattr(outputs, "logits", outputs)
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
~~~

如果使用梯度累积，zero_grad 和 optimizer.step 的周期要改成累积周期，且每次 backward 前通常将 loss 除以 accumulation_steps。若使用梯度裁剪，要明确裁剪发生在 unscale 之后还是之前；若使用混合精度，还要按当前 PyTorch 版本的 amp 接口处理 scaler。一个“通用训练函数”若把这些策略全部隐藏，反而难以复核。

### 4.11.3 用不变量而不是 print 海量张量

有效的 debug 输出应当围绕不变量：

~~~python
def check_training_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    *,
    ignore_index: int = -100,
) -> None:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids and labels must both be [B, T]")
    if input_ids.dtype != torch.long or labels.dtype != torch.long:
        raise TypeError("token ids and labels must be long")
    if input_ids.numel():
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= vocab_size:
            raise ValueError("input_ids contain an invalid token id")
    active = labels.ne(ignore_index)
    active_labels = labels.masked_select(active)
    if active_labels.numel():
        if int(active_labels.min()) < 0:
            raise ValueError("active labels contain a negative id")
        if int(active_labels.max()) >= vocab_size:
            raise ValueError("active labels exceed vocabulary size")
~~~

对整数 token id 检查 finite 没有意义；范围检查才是这里的不变量。更重要的是，检查应在数据进入模型的地方发生，而不是等 CUDA kernel 报错后才看最后一批数据。

### 4.11.4 最小复现要保留触发条件

一个好的最小复现不是把完整训练工程复制到另一个目录，而是保留导致错误的最小状态：

1. 固定一个 batch 和 tokenizer/model 配置。
2. 关闭无关的数据增强、随机采样和并行。
3. 记录输入 shape、dtype、device、有效 token 数。
4. 把错误缩小到一个算子或一层模块。
5. 加入一个预期结果或参考实现。

例如 attention 结果不一致时，先用 B=1、H=1、T=3、d=2 的手工 q/k/v，比较逐项分数、mask 后分数、softmax 权重和最终输出。直接打印一个七十亿参数模型的全部 hidden，信息量反而更低。

## 4.12 数值稳定性：正确的公式仍可能得到错误的浮点数

### 4.12.1 softmax 的稳定形式

直接计算 exp(x) 可能溢出。对每行减去最大值 m 不改变 softmax：

~~~math
\operatorname{softmax}(x_i)
=
\frac{\exp(x_i-m)}
{\sum_j \exp(x_j-m)},
\quad
m=\max_j x_j
~~~

在 PyTorch 中优先使用 torch.softmax、torch.log_softmax 和 torch.logsumexp，而不是把这几个操作拆成多个临时张量。若必须说明原理，可以写一个稳定的 log-softmax：

~~~python
def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x - torch.logsumexp(x, dim=dim, keepdim=True)
~~~

log_softmax 比先 softmax 再 log 更稳定，因为后者可能先把很小的概率下溢为 0，再得到 -inf。DPO、交叉熵、语言模型评分都更适合在 log 空间中计算。

### 4.12.2 mask 的最小值不是逻辑修复

常见代码会把屏蔽位置填成 -1e9。这个数在 float32 中看起来足够小，但在 float16、bfloat16 或不同算子中未必是合适的表示。使用 torch.finfo(dtype).min 可以适配 dtype 的有限范围，却仍然无法修复“整行没有合法 key”的错误。

因此，mask 的检查顺序应当是：

1. 先确认每个有效 query 是否至少有一个合法 key。
2. 再选择与 score dtype 兼容的填充值或使用布尔 mask。
3. 对输出和 loss 检查 isfinite。
4. 对全 padding 的 query 明确规定输出和下游行为。

### 4.12.3 NaN 排查要保留第一个异常点

loss NaN 可能在很早的 logits、attention score 或梯度中产生，直到 reduction 后才被看见。可以在调试版本中加入：

~~~python
def require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        bad = (~torch.isfinite(value)).sum().item()
        raise FloatingPointError(f"{name} contains {bad} non-finite values")


with torch.autograd.detect_anomaly():
    outputs = model(input_ids)
    logits = getattr(outputs, "logits", outputs)
    require_finite("logits", logits)
    loss = causal_lm_loss(logits, labels)
    require_finite("loss", loss)
    loss.backward()
~~~

detect_anomaly 会显著增加开销，适合最小复现而不适合常规训练。排查时还要区分输入数据异常、学习率过高、梯度爆炸、混合精度溢出、非法 labels、全屏蔽 mask 和 inplace 操作破坏 autograd。用 torch.nan_to_num 把 NaN 隐藏起来通常不是修复，因为它会让错误继续传播并污染评估。

### 4.12.4 混合精度改变的是数值路径，不是模型目标

autocast 可以让适合低精度的算子使用 fp16 或 bfloat16，同时让部分敏感操作保持更高精度。它不会自动告诉你哪些 labels 有效，也不会替你处理 overflow。学习率、loss scaling、梯度裁剪和 checkpoint 保存都必须与训练策略一致。

在面试中，最稳妥的表达是先说明“我会在 fp32 reference 上验证语义，再启用 autocast 比较误差、吞吐和显存”。如果误差只在低精度出现，就记录第一个发生差异的模块，而不是直接把所有计算强制转成 fp32。这样才能看清代价来自哪里。

## 4.13 内存和复杂度：把一行 matmul 放回资源账本

### 4.13.1 attention 为什么随序列长度变贵

标准 self-attention 要形成 [B, H, T, T] 的 score 矩阵，因此分数计算和临时存储都随 T 的平方增长：

~~~math
\operatorname{work}_{\mathrm{score}}
\propto BHT^2d
~~~

~~~math
\operatorname{memory}_{\mathrm{score}}
\propto BHT^2
~~~

在训练中还要保存反向传播需要的中间结果；在推理 decode 中，query 往往只有一个新 token，主要增长的是它访问历史 K/V 的长度。FlashAttention 一类融合算法的价值不只是“写得更快”，还在于避免把完整 score 和 probability 矩阵写回高带宽内存。

### 4.13.2 KV cache 的估算

每一层、每一个 batch 样本、每一个缓存 token 的 K/V 元素数近似为：

~~~math
N_{\mathrm{one\ token, layer}}
=
2H_{\mathrm{kv}}d
~~~

若元素占 b 字节，L 层、B 个序列、T 个缓存 token 的字节数近似为：

~~~math
M_{\mathrm{KV}}
=
2LBTH_{\mathrm{kv}}d b
~~~

这个估算没有包括 allocator 对齐、分页元数据、临时 workspace、beam 或并发请求的额外副本。它的用途是判断量级和比较 MHA/GQA/MQA，而不是代替实际 profile。

例如，从 MHA 改成 GQA 时，H_kv 变小，KV cache 和内存读写压力可能显著下降；query head 仍然可以很多，模型质量和并行方式则需要单独评估。不要把“active parameters 较少”“KV head 较少”和“总模型参数较少”混成一个概念。

### 4.13.3 不同的节省方式解决不同瓶颈

- view/expand 主要避免不必要的数据复制，但不会减少算子真正需要访问的逻辑元素。
- repeat 会产生真实副本，可能让显存突然增加。
- no_grad 关闭梯度记录；inference_mode 还会关闭更多 autograd 相关开销，但某些需要 autograd 的后处理不能在其中运行。
- model.eval() 影响 dropout、batch normalization 等模块行为，和 no_grad 不是一回事。
- 量化减少权重或 cache 的存储，但可能引入反量化和精度回归。
- fused attention 减少中间张量和 kernel launch，不一定在所有短序列、所有 batch 下都更快。

性能讨论必须指出目标是 TTFT、TPOT、吞吐、峰值显存还是单位成功任务成本。只说“向量化会更快”不够，因为一个大张量的向量化实现可能产生巨大的中间结果。

### 4.13.4 向量化 token accuracy

下面是一个典型的逐 token 统计。labels 为 -100 的位置不参与分母：

~~~python
def masked_token_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    mask = labels.ne(ignore_index)
    correct = (predictions.eq(labels) & mask).sum()
    total = mask.sum()
    return correct.float() / total.clamp_min(1)
~~~

相对于 Python 双重循环，这个版本把比较交给张量算子。它仍然需要说明“没有有效 token 时返回 0”只是一个约定，汇总整个数据集时更严谨的方式是累积 correct 和 total，最后再相除，而不是对每个 batch 的 accuracy 直接取平均。

## 4.14 训练与评估模式：状态改变必须显式发生

### 4.14.1 train、eval、no_grad 的三个层次

model.train() 和 model.eval() 是模块行为的切换，典型影响包括 dropout 和 batch normalization。torch.no_grad() 是暂时不记录梯度；torch.inference_mode() 是更强的推理上下文，进一步减少 autograd 的元数据开销。三者解决不同问题，不能相互替代。

训练循环通常需要：

~~~python
model.train()
for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=batch["input_ids"])
    loss = causal_lm_loss(outputs.logits, batch["labels"])
    loss.backward()
    optimizer.step()
~~~

评估循环则需要：

~~~python
model.eval()
with torch.inference_mode():
    for batch in validation_loader:
        outputs = model(input_ids=batch["input_ids"])
        loss = causal_lm_loss(outputs.logits, batch["labels"])
~~~

忘记 eval 可能让同一输入在两次评估中得到不同 dropout 结果；忘记 inference_mode 则可能仍然为每个 batch 保留不需要的 autograd 状态。反过来，在需要计算输入梯度的解释性实验中，不能盲目使用 inference_mode。

批量生成时还要额外确认 padding 方向。若 batch 使用右侧 padding，最后一列不一定是每条序列的最后一个有效 token，直接取 logits[:, -1, :] 可能从 padding 位置取分布；工程实现要么使用左侧 padding，要么根据 attention_mask 找到每条序列的最后有效位置。下面的教学生成器假定输入已经是左侧 padding 或没有 padding，这个假设必须和 tokenizer 的配置一致。

### 4.14.2 梯度累积的缩放

如果每次设备只能容纳 micro_batch，而目标 effective batch 更大，可以累积若干次梯度：

~~~python
def accumulated_step(
    model,
    micro_batches,
    optimizer,
    accumulation_steps: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(micro_batches):
        outputs = model(input_ids=batch["input_ids"])
        loss = causal_lm_loss(outputs.logits, batch["labels"])
        (loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
~~~

除以 accumulation_steps 是为了让累积梯度近似目标 batch 的平均梯度。这个教学循环还假定 `micro_batches` 的数量正好是 `accumulation_steps` 的整数倍；如果最后一组不足一个完整累积周期，调用方必须显式执行一次尾批次 `optimizer.step()`，或者丢弃并记录这组样本，不能静默留下未更新的梯度。若每个 micro-batch 的有效 token 数差异很大，按 batch 平均和按 token 平均并不等价；严格训练还要根据有效 token 数做全局归一化。这是一个容易被“代码能运行”掩盖的目标定义问题。

## 4.15 评估小模块：正确性要有参考答案

### 4.15.1 先测形状和不变量

模型小模块的第一层测试不需要大数据，可以覆盖：

- input rank 不正确时是否快速报错；
- hidden size 不能被 head 数整除时是否拒绝初始化；
- mask 是否是 bool，且能广播到 score；
- causal mask 的上三角是否全部为 False；
- labels 的有效值是否落在词表范围；
- 没有有效 loss token 时是否不会产生 NaN；
- top-p 是否至少保留一个有限 logit；
- cache 前向和全量前向的结果是否接近。

测试这些不变量的价值，在于它们不会因为某一组随机数字恰好通过而失效。

### 4.15.2 用小尺寸 reference 检查优化实现

对 attention，可以用 manual_attention 作为小尺寸 reference，再和融合算子比较：

~~~python
def compare_attention_implementations(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
) -> None:
    reference = manual_attention(q, k, v, keep_mask)
    fused = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=keep_mask,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(
        reference,
        fused,
        rtol=2e-3,
        atol=2e-3,
    )
~~~

容差不是越大越好。它应当和 dtype、后端、序列长度及运算顺序有关。fp32 reference 的误差通常可以更紧；低精度和融合 kernel 需要合理放宽，但不能因为“不相等”就无限放宽。测试说明里应记录为什么选择这个容差。

### 4.15.3 属性测试比单个示例更有力量

一个 top-k 测试可以检查“返回的概率支持集不超过 k”；一个 causal attention 测试可以把未来 token 的 value 改成完全不同的数，确认当前位置输出不变；一个 permutation 测试可以确认你显式加入位置编码后顺序变化确实会影响表示。

以 causal attention 为例，固定 q、k，只改变未来位置的 v。如果位置 i 的输出发生变化，说明 mask 方向、broadcast 或 query/key 轴至少有一个错误。这个反事实测试比只检查输出 shape 更接近模型的实际语义。

### 4.15.4 极小数据的手算测试

在损失测试中，可以构造一个只有两个有效目标的 batch，手动计算两个负对数概率的平均值，再与实现比较。在采样测试中，可以把一个 token 的 logit 设置得远高于其他 token，验证 temperature=0 一定返回它；把 top_p 设置为很小的值，验证最可能 token 仍被保留。

手算测试的目的不是覆盖所有浮点细节，而是把“这个函数究竟算什么”固定下来。随机测试则补充不同 batch、长度、padding 和 dtype 的形状组合。

## 4.16 一个完整的 attention debug 案例

假设某个 decoder 在全量前向时输出正常，改成 KV cache 后，生成前几个 token 一致，长度变长后逐渐分叉。这个现象不能直接归因于采样随机性，因为可以先用 greedy 解码排除随机采样。

一个有条理的分析会按以下因果链推进：

1. 让全量前向和 cache 前向使用同一组 q/k/v 权重、同一输入和 temperature=0。
2. 在每一个 decode step 比较新 token 的 hidden 和 logits，而不是只比较最后的字符串。
3. 检查 cache 中 K/V 的形状是否是 [B, H_kv, past_len, d]，并确认新 token 只追加一次。
4. 检查新 query 的 position_id 是否为 past_len，而不是从 0 重启。
5. 检查矩形 causal mask 是否允许新 query 读取全部历史和自己，且没有读取未来位置。
6. 确认 RoPE 只应用于新生成的 q/k，历史 cache 没有被再次旋转。
7. 最后才检查 fused kernel、dtype 和随机种子。

如果第 4 步发现 position_id 每次都是 0，修复后全量与 cache logits 对齐，说明根因是位置状态；如果 logits 已经对齐而字符串不同，再检查采样 generator 和 EOS 处理。把根因、诱因和表现分开，才能避免把所有问题都归结为“cache 有 bug”。

## 4.17 代码质量：清楚地表达取舍

### 4.17.1 正确版本和优化版本要分层

在现场写 attention 时，先写一个显式的 score、mask、softmax、加权求和版本，便于检查形状和语义。确认正确后，再讨论使用 scaled_dot_product_attention、FlashAttention、GQA、KV cache 或量化。

这不是回避工程问题。相反，优化版本只有在 reference 定义清楚后才有可比较的对象。直接写一段复杂 fused wrapper，却说不清 mask 是什么，通常很难定位错误。

### 4.17.2 变量名应当携带语义

q_len、kv_len、head_dim、completion_mask、reference_logps 比 x1、x2、mask 更有价值。变量名不能替代 shape 注释，但可以在阅读代码时提醒读者某个轴的责任。

对于相似的量，名称更要区分：

- logits 与 log_probs；
- token mask 与 sequence mask；
- policy log probability 与 reference log probability；
- prompt length 与 completion length；
- total parameters 与 active parameters；
- latency p50 与 p95。

这些量数值上可能都像浮点张量，语义上却不能互换。

### 4.17.3 错误信息也属于程序接口

与其让一个后续 matmul 抛出“尺寸不匹配”，不如在模块入口说明 q、k、v 的预期维度和实际维度。与其在训练数百 step 后发现 loss 是 NaN，不如在 batch 进入模型时报告有效 token 数、label 范围和 device。

工程代码中的异常信息、日志字段和测试失败信息，都是未来定位问题的材料。coding 面试中的可读性，最终对应的是团队维护时的可诊断性。

## 4.18 练习：把一个小模块写完整

### 4.18.1 练习一：实现 masked mean

给定 values [B, T, D] 和 mask [B, T]，只对有效 token 求平均。需要处理某条序列没有有效 token 的情况，并说明返回零向量还是报错。

一种实现如下：

~~~python
def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.ndim != 3 or mask.ndim != 2:
        raise ValueError("values must be [B, T, D], mask must be [B, T]")
    if values.shape[:2] != mask.shape or mask.dtype != torch.bool:
        raise ValueError("values and mask are incompatible")
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    totals = (values * weights).sum(dim=1)
    counts = weights.sum(dim=1)
    return totals / counts.clamp_min(1)
~~~

这里的零向量约定通过 totals / 1 实现，因为无效位置已经被乘成零。若零向量会在下游被误认为真实表示，也可以返回额外的 has_value 标记或直接拒绝空序列。边界行为应由调用方选择，而不是由实现者偷偷决定。

### 4.18.2 练习二：实现 token budget 的截断

截断对 prompt、completion 和监督标签的影响不同。简单地取 input_ids[:, :max_length] 可能把答案截断成半句话，却仍然保留一个看似有效的标签。练习时应先规定：

- 从左侧还是右侧截断；
- prompt 是否必须完整；
- completion 被截断时是否丢弃样本；
- EOS 是否需要预留位置；
- truncation 后 labels 和 attention mask 如何同步变化。

这道题的关键不是写切片，而是说明数据契约。训练数据处理中的一个 token 偏差，可能在数百万样本上变成系统性目标偏差。

### 4.18.3 练习三：对比全量与 cache

构造一个小 decoder，在相同权重下分别执行：

1. 一次性输入完整序列；
2. 先输入 prompt，再逐 token 输入后续 token，并传递 cache。

比较每个位置的 logits。允许的差异应根据 dtype 和算子后端设置，不能只比较最终 argmax。若结果不同，依次检查 cache 拼接顺序、position_ids、mask、RoPE 和 dropout 状态。

这个练习把 attention、位置编码、生成和测试连接起来，是比单独背诵 KV cache 定义更完整的理解方式。

## 4.19 如何安排准备：从手写到解释

准备 coding 面试不应只有刷题数量，也不应每天机械抄一段 attention。更有效的训练是让同一个实现经历四个阶段。

第一阶段是脱离框架的语义描述：写出输入、输出、shape、不变量和一个手算例子。此时不追求代码短。

第二阶段是最小可运行实现：使用普通 Python 或 PyTorch 基础算子，先让结果在小尺寸上正确。

第三阶段是故障注入：主动把 shift、mask、dtype、device、contiguous 或 reduction 改错，再观察测试如何失败。能解释失败，比只记住正确代码更稳定。

第四阶段是工程替换：把显式实现换成融合算子、cache、向量化或低精度，再与 reference 做数值和资源比较。

如果只有一周，可以按这个顺序组织材料，但每天都应留下运行记录和一个失败案例：

| 时间 | 主题 | 需要留下的证据 |
| --- | --- | --- |
| 第 1 天 | Python 数据结构、JSONL、Top-K | 复杂度说明和边界测试 |
| 第 2 天 | shape、gather、broadcast、dtype | shape ledger 和错误样例 |
| 第 3 天 | causal loss、attention、mask | 手算 loss 与 reference 对齐 |
| 第 4 天 | MHA、RoPE、KV cache | 全量/增量前向差异 |
| 第 5 天 | sampling、generation、停止条件 | 固定 seed 的采样记录 |
| 第 6 天 | DPO、归一化、NaN debug | policy/reference 和有效 token 账本 |
| 第 7 天 | 综合实现和口头讲解 | 一次限时实现、一份复盘 |

表格只是安排的外壳。真正的学习产出应是能够打开的代码、失败前后的差异、测试结果和对边界的解释，而不是一张“已经复习过”的清单。

## 4.20 资料与证据边界

本章的 API 语义和工程建议分别来自不同等级的资料，不能混成一个“权威结论”。

1. Python 官方教程的数据结构部分：[https://docs.python.org/3/tutorial/datastructures.html](https://docs.python.org/3/tutorial/datastructures.html)
2. PyTorch Tensor Views：[https://docs.pytorch.org/docs/stable/tensor_view.html](https://docs.pytorch.org/docs/stable/tensor_view.html)
3. PyTorch Broadcasting Semantics：[https://docs.pytorch.org/docs/stable/notes/broadcasting.html](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
4. PyTorch gather：[https://docs.pytorch.org/docs/stable/generated/torch.gather.html](https://docs.pytorch.org/docs/stable/generated/torch.gather.html)
5. PyTorch cross entropy：[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)
6. PyTorch scaled dot-product attention：[https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
7. PyTorch inference mode：[https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)
8. Attention Is All You Need：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
9. RoFormer: Enhanced Transformer with Rotary Position Embedding：[https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
10. Root Mean Square Layer Normalization：[https://arxiv.org/abs/1910.07467](https://arxiv.org/abs/1910.07467)
11. Direct Preference Optimization：[https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
12. Efficient Memory Management for Large Language Model Serving with PagedAttention：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

Python 和 PyTorch 官方文档用于支持 API、shape view、广播、索引、交叉熵和推理上下文的语义；Transformer、RoPE、RMSNorm、DPO 和 PagedAttention 的论文用于支持原始方法的定义或系统动机。本文中的代码是教学实现，经过语法和逻辑层面的人工检查，但不宣称覆盖所有 PyTorch 版本、模型协议或硬件后端。

案例中的数据、损失值、长度和性能比较均为教学构造，不能当作真实项目指标。手写 attention 与融合 attention 的数值容差、低精度行为和速度，必须在目标设备、目标版本和目标模型上重新测量。公开论文的实验结果也不能直接等价为某个 checkpoint、某个服务配置或某个岗位的统一要求。

## 4.21 结语：能运行只是起点

大模型 coding 的核心能力，可以浓缩成一条完整的程序链：

~~~text
明确契约
    -> 命名 shape 和 dtype
    -> 实现最小正确版本
    -> 用手算例子和不变量测试
    -> 观察数值与梯度
    -> 计算复杂度和内存
    -> 再替换成工程优化
~~~

初学者应先能够指出每个轴和每个 mask 的含义，知道 labels 为什么错位，知道 log probability 为什么不能直接当 probability。专家则要继续把实现放回训练目标、cache 状态、融合算子、低精度和单位任务成本中，解释一个优化究竟节省了什么，又引入了什么新边界。

优秀的现场代码不一定最长，也不一定最短。它应该让别人能够看出输入如何流动、语义在哪里被保证、错误如何被发现，以及当规模从一个小例子扩大到真实模型时，哪些假设仍然成立。下一章进入 ML 与 LLM 基础面试，从这些可运行的模块回到概率、优化、泛化和训练目标，解释代码背后的数学。
