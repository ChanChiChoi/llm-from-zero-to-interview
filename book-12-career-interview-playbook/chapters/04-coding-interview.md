# 第四章：Coding Interview

很多人准备大模型算法岗 coding 面试时，只想到刷 LeetCode。刷题当然有用，但大模型算法岗的 coding 面试通常不只是考“能不能写出二叉树遍历”。它更关心你能否把算法想法写成可靠代码，能否处理张量 shape，能否实现模型核心模块，能否定位训练或推理代码中的 bug。

换句话说，coding interview 不是单独的一关，而是对你“工程化算法能力”的现场验证。

本章重点：Python、数据结构、张量操作、attention、sampling、DPO loss、debug 代码。

## 4.1 大模型算法岗 coding 面试考什么

大模型算法岗 coding 面试通常覆盖四类能力：

1. 通用编程基本功：Python、数据结构、复杂度、边界条件。
2. 数值计算能力：NumPy、PyTorch、张量 shape、broadcast、mask、稳定性。
3. 模型实现能力：attention、MLP、LayerNorm、sampling、loss、训练循环。
4. Debug 能力：读代码、找 bug、解释错误、修复 shape、定位数值异常。

不同岗位侧重不同。

| 岗位方向 | Coding 面试重点 |
| --- | --- |
| 基础模型研究 | 张量操作、模型模块、loss、实验代码清晰度 |
| 训练工程 | 分布式训练、性能、显存、checkpoint、debug |
| Post-training | SFT/DPO loss、数据处理、采样、人评数据处理 |
| 推理部署 | decoding、batching、KV cache、性能优化 |
| RAG / Agent | 字符串处理、检索、排序、缓存、状态管理 |
| 多模态 | 图像张量、padding、mask、embedding 对齐 |

所以准备 coding 面试时，不要只做传统算法题，也要练大模型相关的小实现题。

## 4.2 面试官在观察什么

面试官不只看最终代码是否通过，还会观察过程。

他们通常看：

1. 你能否先澄清输入输出。
2. 你能否处理边界条件。
3. 你能否写出简洁、可读、可测试的代码。
4. 你是否理解时间和空间复杂度。
5. 你是否能解释张量 shape。
6. 你是否能识别数值稳定性问题。
7. 代码出错后能否系统 debug。
8. 你是否知道什么时候应该向量化，什么时候可以先写清楚。

一个常见误区是沉默写代码。coding 面试不是独立考试，而是协作问题解决。你应该边写边解释关键决策，但不要每一行都碎碎念。

推荐节奏：

1. 先复述题意。
2. 明确输入输出和边界。
3. 给出初步方案。
4. 写代码。
5. 用小例子手动跑一遍。
6. 分析复杂度。
7. 如果有时间，讨论优化。

## 4.3 Python 基本功

大模型算法岗大量代码仍然是 Python。Python 写不好，会直接影响面试官对你工程能力的判断。

需要熟练掌握：

1. list、tuple、dict、set。
2. defaultdict、Counter、deque、heapq。
3. 排序、自定义 key、稳定排序。
4. 迭代器、生成器、enumerate、zip。
5. 字符串处理、正则基础。
6. 文件和 JSONL 数据处理。
7. 类型标注和简单类设计。
8. 异常处理和边界判断。

例如，处理一批 JSONL 训练数据时，常见需求是统计字段缺失、长度分布和重复样本。你应该能快速写出清晰代码：

```python
from collections import Counter
import json


def analyze_jsonl(path: str) -> dict:
    total = 0
    missing = Counter()
    lengths = []
    seen = set()
    duplicates = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            obj = json.loads(line)
            for key in ["prompt", "response"]:
                if key not in obj or not obj[key]:
                    missing[key] += 1

            text = obj.get("prompt", "") + "\n" + obj.get("response", "")
            lengths.append(len(text))
            if text in seen:
                duplicates += 1
            else:
                seen.add(text)

    return {
        "total": total,
        "missing": dict(missing),
        "duplicates": duplicates,
        "avg_length": sum(lengths) / max(len(lengths), 1),
    }
```

这个例子不复杂，但体现了数据处理、边界保护和可读性。

## 4.4 通用数据结构与算法

大模型算法岗不一定会大量考 hard LeetCode，但常见数据结构仍然要熟。

高频主题包括：

1. 数组和双指针。
2. 哈希表。
3. 栈和队列。
4. 堆和 Top-K。
5. 二分查找。
6. 图搜索。
7. 动态规划基础。
8. 前缀和和滑动窗口。

为什么这些和大模型有关？因为很多模型系统问题最终也会退化成这些基本结构。

例如：

1. Top-K sampling 用到 top-k 选择。
2. beam search 用到候选队列。
3. RAG 检索结果合并用到排序、去重和堆。
4. tokenizer 可能涉及 trie 或贪心匹配。
5. batch packing 可能涉及区间和贪心。
6. 日志分析常用哈希统计和滑动窗口。

以 Top-K 为例：

```python
import heapq


def top_k_frequent(items: list[str], k: int) -> list[tuple[str, int]]:
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return heapq.nlargest(k, counts.items(), key=lambda x: x[1])
```

面试时不仅要能写，还要能说复杂度：统计是 O(n)，堆选择是 O(m log k)，其中 m 是不同元素数量。

## 4.5 张量操作是核心基本功

大模型 coding 面试中，张量操作非常重要。很多候选人概念懂，但一写 PyTorch 就出错，尤其是 shape、mask、broadcast 和 dtype。

你需要熟练掌握：

1. reshape、view、transpose、permute。
2. unsqueeze、squeeze、expand、repeat。
3. matmul、einsum、bmm。
4. gather、scatter、index_select。
5. masked_fill、where。
6. softmax、log_softmax。
7. dtype 转换和 device 管理。
8. contiguous 的含义。

张量题一定要先写 shape 注释。

例如，给定 logits 和 labels，计算 next-token cross entropy：

```python
import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [batch, seq_len, vocab]
    # labels: [batch, seq_len]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
```

这个题看似简单，但面试官可以追问：

1. 为什么 logits 去掉最后一个 token？
2. 为什么 labels 去掉第一个 token？
3. 为什么要 contiguous？
4. ignore_index 的作用是什么？
5. 如果 labels 中 padding 没有设成 -100，会发生什么？

能答清这些问题，说明你真正理解自回归训练目标。

## 4.6 Attention 实现

Attention 是大模型 coding 面试最高频的模型实现题之一。

最基础的 scaled dot-product attention：

```python
import math
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    # q: [batch, heads, q_len, dim]
    # k: [batch, heads, k_len, dim]
    # v: [batch, heads, k_len, dim]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    # scores: [batch, heads, q_len, k_len]

    if mask is not None:
        # mask should be broadcastable to scores, True means keep.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, v)
```

这段代码必须能解释：

1. 为什么除以 sqrt(d)。
2. mask 的 shape 如何 broadcast。
3. causal mask 和 padding mask 有什么区别。
4. 为什么不用很大的负数常量更安全地适配 dtype。
5. softmax 维度为什么是最后一维。

如果面试官要求写 causal mask：

```python
def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
```

用于 attention 时通常需要变成 `[1, 1, seq_len, seq_len]`：

```python
mask = causal_mask(seq_len, device=q.device)[None, None, :, :]
```

常见 bug 是 mask 语义反了。有的代码里 True 表示保留，有的代码里 True 表示屏蔽。面试时一定要说清楚。

## 4.7 Multi-Head Attention

如果要求写完整 multi-head attention，可以实现一个简化版本：

```python
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, heads * head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        out = scaled_dot_product_attention(q, k, v, mask)
        return self.o_proj(self._merge_heads(out))
```

这个实现适合面试讲解，但真实生产中还要考虑：

1. dropout。
2. attention bias。
3. RoPE。
4. KV cache。
5. FlashAttention。
6. grouped-query attention。
7. dtype 和数值稳定性。

面试中先写清楚正确版本，再讨论工程优化，通常比一开始追求复杂实现更好。

## 4.8 Sampling 实现

推理相关岗位很可能考 sampling。你需要理解 greedy、temperature、top-k、top-p 的区别。

一个基础 sampling 函数：

```python
import torch
import torch.nn.functional as F


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    # logits: [batch, vocab]
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < threshold, torch.finfo(logits.dtype).min)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

可以继续实现 top-p：

```python
def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
    filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
    return filtered.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
```

面试官可能追问：

1. temperature 越大输出越随机还是越确定？
2. top-k 和 top-p 的区别是什么？
3. 为什么 top-p 至少要保留一个 token？
4. greedy decoding 是否等价于 temperature 为 0？
5. sampling 如何影响安全性和幻觉？

## 4.9 DPO Loss 实现

Post-training 方向很可能考 DPO loss。你不一定要背公式，但要理解 chosen、rejected、policy、reference 的关系。

DPO 的核心直觉是：相对于 reference model，policy model 应该更偏向 chosen，而不是 rejected。

简化实现如下：

```python
import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    # All tensors: [batch]
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
```

面试官可能继续要求你从 token logits 计算 sequence log probability：

```python
def sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [batch, seq_len, vocab]
    # labels: [batch, seq_len], ignored tokens are -100
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)

    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)
```

要注意，这里返回的是总 log probability，不是平均 log probability。实际实现中是否按长度归一化，需要根据训练目标和代码库约定决定。

常见追问包括：

1. 为什么需要 reference model？
2. beta 控制什么？
3. chosen 和 rejected 长度不同怎么办？
4. sequence log probability 是否应该除以长度？
5. 如果 labels shift 错了会发生什么？

## 4.10 Debug 代码题

大模型 coding 面试中，debug 题很常见。面试官可能给一段训练代码，让你找 bug。

常见 bug 类型：

1. shape 不匹配。
2. mask 方向错误。
3. labels 没有 shift。
4. padding token 参与 loss。
5. dtype 不一致。
6. device 不一致。
7. 梯度没有清零。
8. eval 时忘记 no_grad。
9. dropout 在 eval 中仍然开启。
10. softmax 维度错误。
11. log probabilities 和 probabilities 混用。
12. inplace 操作破坏梯度。

例如下面这段代码有多个问题：

```python
def train_step(model, batch, optimizer):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(logits, batch["labels"])
    loss.backward()
    optimizer.step()
    return loss.item()
```

你应该能指出：

1. 没有 `optimizer.zero_grad()`。
2. `cross_entropy` 的输入 shape 不对，需要展平成 `[batch * seq, vocab]`。
3. 自回归 LM 通常需要 shift logits 和 labels。
4. padding token 需要 ignore_index。
5. batch tensor 可能没搬到 model device。
6. 如果 model 输出对象不是 tensor，还要取 `.logits`。

修正版本：

```python
def train_step(model, batch, optimizer, device):
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Debug 时的表达很重要。不要只说“这里错了”，要说明为什么错、会导致什么现象、如何验证。

## 4.11 数值稳定性

大模型训练和推理中，数值稳定性非常关键。coding 面试中常见考点包括 softmax、logsumexp、混合精度和溢出。

错误写法：

```python
def naive_softmax(x):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

如果 x 很大，`torch.exp(x)` 会溢出。稳定写法：

```python
def stable_softmax(x):
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

更推荐实际使用 `torch.softmax` 或 `F.log_softmax`，因为框架实现已经处理了稳定性。

常见数值问题包括：

1. fp16 下 attention scores 过大。
2. mask 使用 `-1e9` 在低精度下不合适。
3. loss 出现 NaN。
4. 梯度爆炸。
5. log(0) 导致 inf。
6. softmax 后再 log 不如直接 log_softmax。

如果面试官问 loss NaN 怎么排查，可以按顺序回答：

1. 检查输入数据是否有 NaN 或异常 token。
2. 检查 logits、loss、grad norm 从哪一步开始异常。
3. 检查学习率、warmup、梯度裁剪和混合精度 scaler。
4. 检查 mask 是否导致整行全被屏蔽。
5. 检查 labels 是否越界。
6. 尝试缩小 batch、关闭混合精度或保存最小复现 batch。

## 4.12 性能与内存意识

大模型代码不仅要正确，还要有性能和内存意识。

面试中不一定要求你写 CUDA kernel，但会看你是否理解基本原则：

1. 避免 Python 循环处理大张量。
2. 尽量使用向量化操作。
3. 注意中间张量的显存占用。
4. 区分 view、expand、repeat 的内存行为。
5. 知道 batch size、seq_len、hidden_size 对显存的影响。
6. 理解 attention 的 O(seq_len^2) 复杂度。
7. eval 时使用 no_grad 或 inference_mode。

例如，不推荐这样计算 batch 内 token accuracy：

```python
correct = 0
total = 0
for i in range(labels.size(0)):
    for j in range(labels.size(1)):
        if labels[i, j] != -100:
            correct += int(preds[i, j] == labels[i, j])
            total += 1
```

更好的写法：

```python
mask = labels != -100
correct = ((preds == labels) & mask).sum().item()
total = mask.sum().item()
accuracy = correct / max(total, 1)
```

这不仅更快，也更清晰。

## 4.13 手写小模块训练

建议准备几个常见小模块的手写实现：

1. LayerNorm。
2. RMSNorm。
3. GELU 或 SwiGLU。
4. MLP block。
5. scaled dot-product attention。
6. causal mask。
7. token-level cross entropy。
8. sequence log probability。
9. top-k / top-p sampling。
10. DPO loss。

以 RMSNorm 为例：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., hidden_size]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
```

面试官可能追问 RMSNorm 和 LayerNorm 的区别：LayerNorm 会减均值再除标准差，RMSNorm 不减均值，只按均方根归一化，计算更简单，在很多 LLM 架构中常用。

## 4.14 面试中的代码风格

好的 coding 面试代码不一定最短，但一定清晰。

建议：

1. 函数签名先写清楚。
2. 对张量写 shape 注释。
3. 变量命名直接表达含义。
4. 边界条件显式处理。
5. 不确定 API 时说明思路。
6. 写完用小例子手动检查。
7. 有 bug 时不要慌，先定位再修改。

不建议：

1. 上来就写复杂优化版本。
2. 不解释 shape 变化。
3. 大量复制模板但讲不清。
4. 把所有逻辑塞进一行。
5. 出错后随机修改。

对于大模型题，shape 注释尤其重要。比如：

```python
# hidden: [batch, seq_len, hidden]
# q: [batch, heads, seq_len, head_dim]
# scores: [batch, heads, seq_len, seq_len]
```

这些注释能显著降低你自己和面试官的认知负担。

## 4.15 如何准备

建议按三条线准备。

第一条线：通用算法题。

重点练 easy 到 medium，确保哈希、双指针、堆、二分、图搜索、动态规划基础不掉链子。不需要把大量时间投入偏竞赛的 hard 题，除非目标岗位明确强调传统算法。

第二条线：张量和模型题。

每天手写一个小模块：attention、loss、sampling、LayerNorm、DPO、sequence log probability。重点不是背代码，而是每次都解释 shape、mask、dtype 和边界。

第三条线：debug 和读代码。

找一些训练脚本、推理脚本或开源实现，练习快速回答：输入输出是什么，哪一步最可能错，如何插入检查，如何构造最小复现。

一周训练计划可以这样安排：

| 天数 | 重点 |
| --- | --- |
| 第 1 天 | Python 数据结构、Top-K、字符串和 JSONL 处理 |
| 第 2 天 | PyTorch shape、gather、mask、cross entropy |
| 第 3 天 | attention、causal mask、multi-head attention |
| 第 4 天 | sampling、top-k、top-p、beam search 思路 |
| 第 5 天 | SFT loss、sequence log probability、DPO loss |
| 第 6 天 | debug 训练代码、NaN 排查、device 和 dtype 问题 |
| 第 7 天 | 模拟面试，限时讲解并手写完整题 |

## 4.16 常见现场失误

### 4.16.1 不澄清题意

比如输入是否为空、是否有 padding、是否 batch 化、是否需要支持不同长度。如果不问清楚，很容易写偏。

### 4.16.2 Shape 不写清楚

大模型题中，shape 错误是第一大问题。每次 transpose、view、gather 都要知道前后 shape。

### 4.16.3 忽略 mask

padding mask、causal mask、attention mask、loss mask 是不同概念。很多错误来自 mask 语义混乱。

### 4.16.4 忽略 dtype 和 device

CPU tensor 和 GPU tensor 混用、fp16 下使用不合适的常量、labels dtype 错误，都是常见 bug。

### 4.16.5 只会调用库，不会解释原理

调用 `F.cross_entropy` 没问题，但你要知道它内部相当于 `log_softmax` 加负对数似然。

### 4.16.6 过早优化

面试中先写正确清晰版本，再讨论优化。直接写复杂版本出错，通常得不偿失。

## 4.17 本章小结

大模型算法岗 coding 面试考的不是单纯刷题数量，而是你能否把算法、模型和工程基本功落到代码里。

你需要同时准备 Python 数据结构、通用算法、PyTorch 张量操作、attention、sampling、loss、DPO、debug 和数值稳定性。写代码时要先明确输入输出和边界，张量题要写 shape，模型题要讲清 mask 和 loss，debug 题要有系统排查路径。

下一章进入 ML 与 LLM 基础面试。coding 证明你能写出来，基础面试会继续考察你是否真正理解这些代码背后的概率、优化、泛化、Transformer、tokenization 和训练目标。
