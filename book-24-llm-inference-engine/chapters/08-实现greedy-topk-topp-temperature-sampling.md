# 第 8 章 实现 greedy、top-k、top-p 和 temperature sampling

上一章我们实现了最小 generate loop，但 token 选择只用了 greedy，也就是每次都选 logits 最大的 token。本章把 token 选择模块拆出来，逐步实现 greedy、temperature、top-k 和 top-p sampling。

一句话概括：

> Sampling 决定模型在“最确定”和“更多样”之间如何取舍，是生成质量、稳定性、可控性和成本体验的重要开关。

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

## 8.12 常见误区

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

## 8.13 面试追问

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

## 8.14 小练习

1. 在第 7 章代码中替换 greedy，加入 `sample_next_token`。
2. 对同一个 prompt 分别设置 `temperature=0`、`0.7`、`1.2`，观察输出差异。
3. 对比 `top_k=10` 和 `top_p=0.9` 的候选集合变化。
4. 设置相同 seed 多次运行，观察是否完全一致。
5. 思考为什么结构化 JSON 输出通常更适合低 temperature。

## 8.15 本章小结

本章实现了 greedy、temperature、top-k 和 top-p sampling。

Greedy 适合确定性和 debug；temperature 控制分布随机性；top-k 固定保留高分 token；top-p 根据累计概率动态保留候选集合。把采样逻辑封装成 sampler 后，generate loop 的结构更接近真实 serving engine。

下一章我们会加入 KV Cache，让 naive generate loop 不再每轮重复计算完整历史序列。
