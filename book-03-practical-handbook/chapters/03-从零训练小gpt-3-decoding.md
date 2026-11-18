# 第 3 章（第三篇）：从概率分布到生成序列

训练目标给每个位置一个概率分布，但真正生成时还要规定怎样从中选 token。贪心、随机采样、temperature、top-k 和 top-p 改变的是解码过程；为了比较它们，prompt、checkpoint、随机种子和停止条件都要固定。

## 3.3 解码控制：Temperature、Top-k 与 Top-p

### 3.3.1 logits、概率和选择规则

语言模型最后输出的是 $V$ 个 logits。softmax 把它们变成概率：

```math
p_i=\frac{\exp(z_i)}{\sum_{j=0}^{V-1}\exp(z_j)}
```

贪心解码选择 $\arg\max_i z_i$，完全采样则按照整个 $p$ 分布抽样。两种方法各有问题：贪心容易重复，按完整分布采样又可能选中概率很低的 token。temperature、top-k 和 top-p 可以调整或筛选候选分布，但不能补上模型没有学到的事实关系或文本结构。

### 3.3.2 Temperature 改变分布尖锐程度

给定温度 $\tau>0$：

```math
p_i(\tau)=\frac{\exp(z_i/\tau)}{\sum_{j=0}^{V-1}\exp(z_j/\tau)}
```

$\tau<1$ 放大 logits 差异，分布更尖；$\tau>1$ 压低 logits 差异，分布更平；$\tau\to0^+$ 在数值上接近贪心，但不应真的把代码写成除以 0。需要确定性输出时，直接使用 argmax 更清楚。

分布熵可以帮助量化这种变化：

```math
H(p)=-\sum_{i=0}^{V-1}p_i\log p_i
```

熵降低表示概率集中，不等于事实性提高；熵升高表示选择更多，不等于创造性或质量提高。不同任务需要用任务指标、格式通过率、重复率或人工评价验证。

Temperature 只改变推理时的概率分布，不会更新模型参数，也不会改变训练目标。可以用它比较不同的生成行为，但不能把它当成修复训练质量的手段。

### 3.3.3 Top-k 保留固定数量的候选

令 $S_k$ 为 logits 最大的 $k$ 个 token 的集合。过滤后的 logits 是：

```math
\tilde z_i=\begin{cases}
z_i,&i\in S_k\\
m_{\min},&i\notin S_k
\end{cases}
```

其中 $m_{\min}$ 是当前 dtype 的最小有限值。过滤后必须重新 softmax；不能把已经归一化的概率简单截断后忘记重新归一化。

### 3.3.4 Top-p 保留概率质量

先按概率从大到小排序：

```math
p_{(1)}\ge p_{(2)}\ge\cdots\ge p_{(V)}
```

再定义最小的候选数量：

```math
m=\min\left\{r:\sum_{i=1}^{r}p_{(i)}\ge p\right\}
```

top-p 保留排序后的前 $m$ 个 token。分布很尖时，候选集合可能只有一个或几个；分布平坦时，候选集合会扩大。因此它不是固定的多样性保证，而是依据当前分布的概率质量动态改变候选数量。Nucleus sampling 的原始研究也把候选集合定义为随上下文分布变化的概率质量“核”；论文报告的是其 GPT-2/WebText 开放式生成设置，不代表所有任务的质量排序：[Holtzman 等，The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751v2)。

实现时要保留第一个使累计概率达到阈值的 token。过滤函数支持 `[..., vocab_size]`；抽样前把所有前导维展平成行，调用只接收向量或矩阵的 `torch.multinomial`，再恢复原 batch 维：

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import math

import torch
import torch.nn.functional as F


def filter_top_k(logits, top_k=None):
    if logits.ndim < 1 or logits.size(-1) <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if top_k is None:
        return logits
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer")
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k, dim=-1)
    mask_value = torch.finfo(logits.dtype).min
    filtered = torch.full_like(logits, mask_value)
    filtered.scatter_(-1, indices, values)
    return filtered


def filter_top_p(logits, top_p=None):
    if logits.ndim < 1 or logits.size(-1) <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    if top_p is None:
        return logits
    if (
        not isinstance(top_p, (int, float))
        or isinstance(top_p, bool)
        or not math.isfinite(float(top_p))
        or not 0.0 < top_p <= 1.0
    ):
        raise ValueError("top_p must be in (0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative >= top_p

    shifted_mask = sorted_mask.clone()
    shifted_mask[..., 1:] = sorted_mask[..., :-1]
    shifted_mask[..., 0] = False
    mask_value = torch.finfo(logits.dtype).min
    sorted_logits = sorted_logits.masked_fill(shifted_mask, mask_value)

    filtered = torch.full_like(logits, mask_value)
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def sample_next_token(
    logits,
    temperature=1.0,
    top_k=None,
    top_p=None,
    generator=None,
):
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be a finite positive number")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype")
    logits = logits / temperature
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)
    probabilities = F.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities contain NaN or infinity")
    if (probabilities.sum(dim=-1) <= 0).any():
        raise ValueError("probability mass is empty")
    flat_probabilities = probabilities.reshape(-1, probabilities.size(-1))
    sampled = torch.multinomial(
        flat_probabilities,
        num_samples=1,
        generator=generator,
    )
    return sampled.reshape(*probabilities.shape[:-1], 1)


def greedy_next_token(logits):
    return logits.argmax(dim=-1, keepdim=True)
```
这里先做 temperature，再做 top-k/top-p，是一种常见的教学顺序。top-p 的候选集合依赖 softmax 后的相对概率，因此如果先 top-k 再 top-p，top-p 看到的是截断后的分布，结果会和先 top-p 再 top-k 不同。解码器必须固定顺序并在实验记录中写清楚。`sample_next_token` 可接收与 logits 所在设备匹配的随机数生成器，便于把采样随机性与其他随机操作分开；PyTorch 的 `torch.multinomial` 按非负权重抽样：[2.12 接口文档](https://docs.pytorch.org/docs/2.12/generated/torch.multinomial.html)。第 3.3.5 的代码依赖本节已经定义的 helper；若要单独复制运行，应把本节完整代码一并复制。

### 3.3.5 一个可验证的采样实验

下面的实验使用固定 logits，观察熵和保留集合。它依赖上一节的 helper；将两段代码放在同一脚本中运行。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import torch
import torch.nn.functional as F


generator = torch.Generator().manual_seed(7)
tokens = ["A", "B", "C", "D", "E", "F"]
logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -0.5, -1.0]])


def entropy(probabilities):
    probabilities = probabilities.clamp_min(1e-12)
    return float(
        -(probabilities * probabilities.log()).sum(dim=-1).item()
    )


def kept_tokens(filtered_logits):
    minimum = torch.finfo(filtered_logits.dtype).min
    return [
        tokens[index]
        for index, value in enumerate(filtered_logits[0])
        if value.item() != minimum
    ]


base = F.softmax(logits, dim=-1)
cold = F.softmax(logits / 0.5, dim=-1)
hot = F.softmax(logits / 2.0, dim=-1)
top_k_logits = filter_top_k(logits, top_k=3)
top_p_logits = filter_top_p(logits, top_p=0.8)
threshold_top_p_logits = filter_top_p(
    torch.log(torch.tensor([[0.5, 0.3, 0.2]])),
    top_p=0.79,
)
batched_logits = logits.unsqueeze(0).expand(2, -1, -1)
sampled = sample_next_token(
    batched_logits,
    temperature=0.8,
    top_k=4,
    top_p=0.9,
    generator=generator,
)

print("base_probs=", [round(x, 4) for x in base[0].tolist()])
print("cold_entropy=", round(entropy(cold), 4))
print("hot_entropy=", round(entropy(hot), 4))
print("top_k_kept=", kept_tokens(top_k_logits))
print("top_p_kept=", kept_tokens(top_p_logits))
assert torch.isfinite(threshold_top_p_logits[0, :2]).all()
assert threshold_top_p_logits[0, 2].item() == torch.finfo(threshold_top_p_logits.dtype).min
assert sampled.shape == (2, 1, 1)
print("sample_shape=", tuple(sampled.shape))
```
固定 logits 的实验只能证明过滤函数的局部行为。真实生成还需要观察任务成功率、格式通过率、重复率、长度分布和事实性；“看起来更像训练语料”不是单独可靠的质量指标。

### 3.3.6 找出解码问题出在哪一步

| 现象 | 首先检查的量 | 不能直接推出的结论 |
| --- | --- | --- |
| 输出高度重复 | temperature、top-k、top-p、训练数据重复度 | 模型一定过拟合 |
| 输出乱码 | prompt 是否在词表、decode 是否一致、checkpoint/tokenizer 是否匹配 | 训练一定失败 |
| 输出过于随机 | 温度、候选过滤、logits 是否有限 | 模型一定缺乏知识 |
| 采样报 NaN | logits、mask、dtype、过滤后候选数量 | 只调 temperature 就能修复 |
| greedy 和 sampling 差异很大 | 固定 seed、分布熵、候选集合 | 某种策略普遍更好 |

比较解码配置时，应使用同一 checkpoint 和 prompt，并固定生成预算、温度、过滤顺序、停止规则与随机种子。最好对多条 prompt、多个 seed 重复比较；单个 seed 的结果受随机数调用顺序、设备和采样实现影响，增加一次随机调用就可能改变后续整段生成。报告多次结果的分布，比挑一条最满意的输出更能说明策略差异。

看一个分布如何同时受 temperature 与过滤影响。logits 为 $[2,1,0]$ 时，temperature=1 的概率约为 $[0.665,0.245,0.090]$；top-p=0.7 会保留前两个，截断后归一化为 $[0.731,0.269,0]$。temperature=2 后概率约为 $[0.506,0.307,0.186]$，此时 top-k=2 和 top-p=0.8 都保留前两个，归一化后约为 $[0.622,0.378,0]$。温度越低，最高项越容易独自达到 top-p 阈值；温度越高，达到同一概率质量可能要保留更多候选。top-k 控制候选数量，top-p 控制概率质量，二者不能互相替代；先后顺序也会改变结果，因为 top-p 看到的是过滤前还是过滤后的分布。

过滤后至少要留下一个有限分数的候选。当前 helper 的 top-k/top-p 会保留至少一个位置，但若 logits 全是负无穷或 NaN，softmax 仍无法产生有效概率；外加 EOS 屏蔽、词表白名单等约束时，也要重新检查候选集合。top-k 的边界遇到并列分数时，保留哪个 token 可能取决于排序实现；若可复现性重要，应固定 tie-breaking 或记录实现。确定性贪心应单独走 `argmax`，不要把 temperature=0 当成通用定义：对 $[2,2,0]$，最高分并列，而 argmax 通常只返回其中一个索引。

停止规则同样属于解码配置。EOS 可结束生成；若它也被用作 padding，必须区分真实结束和填充值。最大新 token 数是硬上限，停止词的识别依赖 tokenizer 的编码规则。如果 top-p 的候选集合只剩概率为 0.95 的 EOS，模型可能立即停止；最小生成长度或屏蔽 EOS 会改变采样分布，因此要记录下来。评估还应规定空输出或只输出 EOS 时如何计算长度、重复率和格式分，不能为了好看临时丢弃这类样本。



固定了解码协议后，生成样例才具有可比较性。下一篇会把这些样例和验证 loss 放回训练状态中，处理 checkpoint、恢复和模型选择。
