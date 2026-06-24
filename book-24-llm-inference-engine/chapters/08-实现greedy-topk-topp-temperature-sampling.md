# 第 8 章 实现 greedy、top-k、top-p 和 temperature sampling

上一章我们实现了最小 generate loop，但 token 选择只用了 greedy，也就是每次都选 logits 最大的 token。本章把 token 选择模块拆出来，逐步实现 greedy、temperature、top-k 和 top-p sampling。

一句话概括：

> Sampling 决定模型在“最确定”和“更多样”之间如何取舍，是生成质量、稳定性、可控性和成本体验的重要开关。

## 8.0 本讲资料边界与第二轮精修口径

本章第二轮精修前，先用公开资料校准口径：Transformers 的 generation 文档把 greedy、sampling、beam search、logits processor / warper、stopping criteria 和 cache 放在同一个生成控制体系中；top-k、top-p 和 temperature 都属于对 logits 或候选概率分布的控制策略；nucleus sampling 的经典论文动机是减少低质量长尾 token 被采样，同时避免固定 top-k 在不同分布形状下过度或不足截断。

因此，本章不讨论 beam search、contrastive search、speculative decoding、structured generation 和安全 logits mask，只聚焦单步 token sampling。正文里的 PyTorch 代码用于贴近工程 API，新增的 0 依赖 demo 用纯 Python 展示 stable softmax、temperature、top-k、top-p、seeded multinomial 和采样审计指标。

## 8.1 为什么要单独实现 sampling

在 generate loop 中，模型 forward 只负责给出 logits。真正决定下一个 token 的，是 sampling 模块。

上一章的核心代码是：

```python
logits = model_wrapper.forward(generated_ids)
next_token_id = greedy_select(logits)
generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
```

本章要把 `greedy_select` 升级成可配置 sampler：

```python
next_token_id = sampler.sample(logits)
```

这样做的好处是：

1. generate loop 不关心具体采样策略。
2. 后续可以轻松加入 temperature、top-k、top-p。
3. serving engine 可以为每个请求设置不同 sampling params。
4. debug 时可以固定 greedy，线上可以启用随机采样。

## 8.2 logits、概率和采样

模型输出 logits，logits 不是概率，而是未归一化分数。

要得到概率，需要 softmax：

```python
probs = torch.softmax(logits, dim=-1)
```

然后从概率分布中采样：

```python
next_token_id = torch.multinomial(probs, num_samples=1)
```

生成的核心链路可以理解为：

```text
hidden states -> logits -> probability distribution -> sampled token
```

不同 sampling 策略，本质上都是在 softmax 和采样前后修改这个概率分布。

## 8.3 Greedy Decoding

Greedy decoding 每次选择 logits 最大的 token。

```python
def greedy_sample(logits):
    next_token_logits = logits[:, -1, :]
    return torch.argmax(next_token_logits, dim=-1, keepdim=True)
```

优点：

1. 确定性强。
2. 实现简单。
3. 适合单元测试和系统 debug。
4. 不需要随机数种子。

缺点：

1. 容易输出重复内容。
2. 多样性较差。
3. 不一定选择全局最优序列。
4. 对开放式写作和创意任务不友好。

Greedy 适合知识问答、分类式输出、格式稳定任务，也适合排查 serving engine 的确定性问题。

## 8.4 Temperature

Temperature 用来控制概率分布的尖锐程度。

做法是把 logits 除以 temperature：

```python
scaled_logits = logits / temperature
```

temperature 越低，分布越尖锐，模型越保守。

temperature 越高，分布越平坦，模型越随机。

常见直觉：

1. `temperature = 0`：近似 greedy。
2. `temperature < 1`：更保守。
3. `temperature = 1`：保持原分布。
4. `temperature > 1`：更多样，但更容易跑偏。

实现时要注意 `temperature=0` 不能直接做除法，通常单独走 greedy。

```python
def apply_temperature(logits, temperature):
    if temperature is None or temperature == 1.0:
        return logits
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature
```

生产系统里也常把 `temperature=0` 解释为 deterministic decoding。

## 8.5 Top-k Sampling

Top-k 的思想是：只保留概率最高的 k 个 token，其他 token 直接屏蔽。

例如词表有 50000 个 token，但 `top_k=50`，那么每一步只从分数最高的 50 个 token 中采样。

实现方式：

```python
def apply_top_k(logits, top_k):
    if top_k is None or top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
```

Top-k 的优点：

1. 简单直观。
2. 能过滤大量低概率 token。
3. 减少胡乱采样的概率。

缺点：

1. k 是固定数量，不适应分布形状。
2. 如果分布很尖锐，k 太大没意义。
3. 如果分布很平坦，k 太小可能过度截断。

## 8.6 Top-p Sampling

Top-p 也叫 nucleus sampling。它不是保留固定数量 token，而是保留累计概率达到 p 的最小 token 集合。

例如 `top_p=0.9` 表示：按概率从高到低排序，只保留累计概率达到 0.9 的候选 token。

实现步骤：

1. 对 logits 排序。
2. softmax 得到排序后的概率。
3. 计算累计概率。
4. 屏蔽累计概率超过 top-p 之后的 token。
5. 映射回原词表顺序。

示例代码：

```python
def apply_top_p(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0:
        raise ValueError("top_p must be positive")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return filtered_logits
```

Top-p 的优点是能适应分布形状。分布尖锐时候选集合小，分布平坦时候选集合大。

## 8.7 组合顺序

常见组合流程是：

```text
取最后位置 logits
  -> temperature scaling
  -> top-k filtering
  -> top-p filtering
  -> softmax
  -> multinomial sample
```

代码如下：

```python
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=1.0):
    next_token_logits = logits[:, -1, :]

    if temperature == 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    next_token_logits = apply_temperature(next_token_logits, temperature)
    next_token_logits = apply_top_k(next_token_logits, top_k)
    next_token_logits = apply_top_p(next_token_logits, top_p)

    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

不同框架在细节上可能略有差异，但核心思想一致：先修改候选分布，再采样。

## 8.8 Sampler 类

为了更接近推理框架，可以把采样封装成类：

```python
class SamplingParams:
    def __init__(self, temperature=1.0, top_k=None, top_p=1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class Sampler:
    def __init__(self, params):
        self.params = params

    def sample(self, logits):
        return sample_next_token(
            logits,
            temperature=self.params.temperature,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
        )
```

然后 generate loop 变成：

```python
def generate(tokenizer_wrapper, model_wrapper, sampler, prompt, max_new_tokens=128):
    generated_ids = tokenizer_wrapper.encode(prompt).to(model_wrapper.device)

    for _ in range(max_new_tokens):
        logits = model_wrapper.forward(generated_ids)
        next_token_id = sampler.sample(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

    return tokenizer_wrapper.decode(generated_ids[0].tolist())
```

这样结构更清楚，后续加入 repetition penalty、stop token、logprobs 也更方便。

## 8.9 随机种子和可复现性

一旦使用 sampling，输出就可能不同。

如果需要复现，可以设置随机种子：

```python
torch.manual_seed(1234)
```

但生产系统里，可复现并不总是简单。因为还会受到这些因素影响：

1. batch 内请求顺序。
2. GPU kernel 的确定性。
3. 多卡并行通信。
4. speculative decoding 的接受路径。
5. 框架版本和浮点误差。

所以面试里不要轻易承诺“设置 seed 就一定完全可复现”。更稳妥的说法是：单机单请求简单路径可以尽量复现，复杂 serving 场景需要额外控制执行顺序和底层确定性。

## 8.10 参数如何影响输出

常见参数组合：

| 场景 | temperature | top_p | top_k | 特点 |
| --- | --- | --- | --- | --- |
| 稳定问答 | 0 或 0.2 | 1.0 | None | 更确定 |
| 通用聊天 | 0.7 | 0.9 | None 或 50 | 平衡稳定和多样 |
| 创意写作 | 0.9-1.2 | 0.95 | None | 更多样 |
| 格式化输出 | 0 或 0.1 | 1.0 | None | 更可控 |

这只是经验，不是硬规则。不同模型、任务和业务对参数的敏感度不同。

## 8.11 Sampling 和 Serving Engine

在 serving engine 中，sampling 不是全局固定的。不同请求可能有不同参数。

例如：

```text
request_1: temperature=0, top_p=1.0
request_2: temperature=0.7, top_p=0.9
request_3: temperature=1.0, top_k=40
```

这会带来工程复杂度：

1. batch 中每个请求的 sampling params 不同。
2. 有些请求需要 logprobs，有些不需要。
3. 有些请求需要 deterministic，有些需要 random。
4. structured generation 还会加入额外 logits mask。

因此真实 engine 的 sampler 往往是批量化且按请求配置执行的，而不是本章这种单请求函数。

## 8.12 采样公式、数值稳定和可运行 demo

temperature softmax 可以写成：

```math
p_i(\tau)=\frac{\exp((z_i-m)/\tau)}{\sum_j \exp((z_j-m)/\tau)},\qquad m=\max_j z_j
```

其中 $z_i$ 是第 $i$ 个 token 的 logits，$\tau$ 是 temperature。减去 $m$ 不改变 softmax 结果，但能避免 $\exp(z_i)$ 因 logits 过大而数值溢出。

当 $\tau\to 0$ 时，分布会越来越尖锐，工程上通常直接走 greedy：

```math
y=\mathrm{argmax}_{i\in\mathcal{V}}z_i
```

Top-k 的候选集合可以写成：

```math
\mathcal{S}_k=\{i\mid z_i\ \mathrm{is\ in\ the\ top}\ k\ \mathrm{values}\}
```

Top-p 的候选集合先按概率从大到小排序，取最小的前缀集合：

```math
n^*=\min\left\{n:\sum_{r=1}^{n}p_{(r)}\ge \rho\right\},\qquad \mathcal{S}_{p}=\{(1),\ldots,(n^*)\}
```

其中 $\rho$ 是 `top_p`，$p_{(r)}$ 是排序后的第 $r$ 个概率。过滤后要重新归一化：

```math
\tilde{p}_i=\frac{p_i\mathbf{1}[i\in\mathcal{S}]}{\sum_j p_j\mathbf{1}[j\in\mathcal{S}]}
```

采样模块的验收门禁可以写成：

```math
G_{\mathrm{sampling}}=G_{\mathrm{softmax}}G_{\mathrm{temperature}}G_{\mathrm{topk}}G_{\mathrm{topp}}G_{\mathrm{seed}}
```

下面是一个 0 依赖 demo。它用固定 logits 展示 greedy、temperature、top-k、top-p 和组合过滤如何改变候选集合、熵和最终采样 token。

```python
import math
import random


VOCAB = ["safe", "clear", "creative", "risky", "off_topic", "loop"]
LOGITS = [4.0, 3.0, 2.0, 0.5, -1.0, -2.0]


def stable_softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax")
    max_logit = max(logits)
    exp_values = [math.exp((value - max_logit) / temperature) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def normalize(probs):
    total = sum(probs)
    if total <= 0:
        raise ValueError("all probabilities were filtered out")
    return [value / total for value in probs]


def apply_top_k(probs, top_k):
    if top_k is None or top_k <= 0 or top_k >= len(probs):
        return probs[:]
    keep = {idx for idx, _ in sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:top_k]}
    return normalize([value if idx in keep else 0.0 for idx, value in enumerate(probs)])


def apply_top_p(probs, top_p):
    if top_p is None or top_p >= 1.0:
        return probs[:]
    if top_p <= 0:
        raise ValueError("top_p must be positive")

    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    keep = set()
    cumulative = 0.0
    for idx, value in ranked:
        keep.add(idx)
        cumulative += value
        if cumulative >= top_p:
            break
    return normalize([value if idx in keep else 0.0 for idx, value in enumerate(probs)])


def entropy(probs):
    return -sum(value * math.log(value) for value in probs if value > 0)


def sample_from_probs(probs, seed):
    rng = random.Random(seed)
    threshold = rng.random()
    cumulative = 0.0
    for idx, value in enumerate(probs):
        cumulative += value
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def sample_once(name, temperature=1.0, top_k=None, top_p=1.0, seed=6):
    if temperature == 0:
        selected = max(range(len(LOGITS)), key=lambda idx: LOGITS[idx])
        probs = [1.0 if idx == selected else 0.0 for idx in range(len(LOGITS))]
    else:
        probs = stable_softmax(LOGITS, temperature=temperature)
        probs = apply_top_k(probs, top_k)
        probs = apply_top_p(probs, top_p)
        selected = sample_from_probs(probs, seed)

    candidates = [VOCAB[idx] for idx, value in enumerate(probs) if value > 0]
    return {
        "name": name,
        "selected": VOCAB[selected],
        "candidates": candidates,
        "candidate_count": len(candidates),
        "entropy": round(entropy(probs), 3),
        "probs": {VOCAB[idx]: round(value, 3) for idx, value in enumerate(probs) if value > 0},
    }


cases = [
    sample_once("greedy", temperature=0),
    sample_once("temperature_0_7", temperature=0.7),
    sample_once("temperature_1_4", temperature=1.4),
    sample_once("top_k_3", temperature=1.0, top_k=3),
    sample_once("top_p_0_8", temperature=1.0, top_p=0.8),
    sample_once("combined", temperature=0.8, top_k=4, top_p=0.85),
]

repeat_a = sample_once("repeat", temperature=1.0, top_k=3, seed=42)["selected"]
repeat_b = sample_once("repeat", temperature=1.0, top_k=3, seed=42)["selected"]

audit = {
    "softmax_normalized": all(abs(sum(item["probs"].values()) - 1.0) <= 0.002 for item in cases),
    "temperature_changes_entropy": cases[2]["entropy"] > cases[1]["entropy"],
    "top_k_reduces_candidates": cases[3]["candidate_count"] == 3,
    "top_p_adapts_candidates": cases[4]["candidate_count"] < len(VOCAB),
    "seed_reproducible": repeat_a == repeat_b,
}
audit["sampling_gate"] = all(audit.values())

print("cases=", cases)
print("audit=", audit)
```

一组典型输出如下：

```text
cases= [{'name': 'greedy', 'selected': 'safe', 'candidates': ['safe'], 'candidate_count': 1, 'entropy': -0.0, 'probs': {'safe': 1.0}}, {'name': 'temperature_0_7', 'selected': 'clear', 'candidates': ['safe', 'clear', 'creative', 'risky', 'off_topic', 'loop'], 'candidate_count': 6, 'entropy': 0.686, 'probs': {'safe': 0.766, 'clear': 0.184, 'creative': 0.044, 'risky': 0.005, 'off_topic': 0.001, 'loop': 0.0}}, {'name': 'temperature_1_4', 'selected': 'clear', 'candidates': ['safe', 'clear', 'creative', 'risky', 'off_topic', 'loop'], 'candidate_count': 6, 'entropy': 1.187, 'probs': {'safe': 0.54, 'clear': 0.264, 'creative': 0.129, 'risky': 0.044, 'off_topic': 0.015, 'loop': 0.007}}, {'name': 'top_k_3', 'selected': 'clear', 'candidates': ['safe', 'clear', 'creative'], 'candidate_count': 3, 'entropy': 0.832, 'probs': {'safe': 0.665, 'clear': 0.245, 'creative': 0.09}}, {'name': 'top_p_0_8', 'selected': 'clear', 'candidates': ['safe', 'clear'], 'candidate_count': 2, 'entropy': 0.582, 'probs': {'safe': 0.731, 'clear': 0.269}}, {'name': 'combined', 'selected': 'clear', 'candidates': ['safe', 'clear'], 'candidate_count': 2, 'entropy': 0.53, 'probs': {'safe': 0.777, 'clear': 0.223}}]
audit= {'softmax_normalized': True, 'temperature_changes_entropy': True, 'top_k_reduces_candidates': True, 'top_p_adapts_candidates': True, 'seed_reproducible': True, 'sampling_gate': True}
```

注意最后一个 combined 示例：先做 temperature 和 top-k，再做 top-p，会让候选集和概率发生二次变化。实际框架要清楚记录这些参数，否则线上生成差异很难复盘。

## 8.13 常见误区

误区一：temperature 越高越聪明。

temperature 只会增加随机性，不会提升模型能力。太高反而更容易胡说。

误区二：top-k 和 top-p 是一回事。

top-k 固定保留 k 个 token；top-p 保留累计概率达到 p 的动态集合。

误区三：greedy 一定最好。

greedy 稳定，但可能重复、死板，不适合所有开放式任务。

误区四：sampling 只影响文本质量，不影响系统。

sampling 会影响输出长度、是否重复、是否命中 stop，从而影响 decode 轮数、TPOT 和成本。

误区五：设置 seed 就能保证任何线上场景完全复现。

复杂 serving 里还有 batch 顺序、kernel、并行和浮点误差等因素。

## 8.14 面试追问

1. logits 和概率有什么区别？
2. greedy decoding 怎么实现？
3. temperature 如何影响概率分布？
4. top-k 和 top-p 的区别是什么？
5. 为什么 top-p 更能适应分布形状？
6. sampling 参数如何影响输出稳定性和多样性？
7. serving engine 中为什么每个请求可能需要不同 sampling params？
8. 为什么线上完全复现随机采样并不容易？

参考回答思路：

1. 先说 logits 是未归一化分数，softmax 后才是概率。
2. 再说 greedy 取最大，temperature 调分布尖锐程度，top-k/top-p 过滤候选集合。
3. 然后说明参数影响质量、输出长度和成本。
4. 最后补 serving 复杂度：batch 内每个请求参数不同，sampler 需要批量化和可配置。

## 8.15 小练习

1. 在第 7 章代码中替换 greedy，加入 `sample_next_token`。
2. 对同一个 prompt 分别设置 `temperature=0`、`0.7`、`1.2`，观察输出差异。
3. 对比 `top_k=10` 和 `top_p=0.9` 的候选集合变化。
4. 设置相同 seed 多次运行，观察是否完全一致。
5. 思考为什么结构化 JSON 输出通常更适合低 temperature。
6. 扩展本章纯 Python demo，打印每种策略的候选 token 集合和 entropy，解释为什么高 entropy 不等于高质量。

## 8.16 本章小结

本章实现了 greedy、temperature、top-k 和 top-p sampling。

Greedy 适合确定性和 debug；temperature 控制分布随机性；top-k 固定保留高分 token；top-p 根据累计概率动态保留候选集合。把采样逻辑封装成 sampler 后，generate loop 的结构更接近真实 serving engine。

下一章我们会加入 KV Cache，让 naive generate loop 不再每轮重复计算完整历史序列。
