# 第 10 章 实现 batched prefill

上一章我们实现了单请求 KV Cache：第一次 prefill 输入完整 prompt，后续 decode 只输入最新 token。本章继续升级，把多个请求的 prompt 合成一个 batch，一次性做 prefill。

Batched prefill 是推理框架从“单请求 demo”走向“多请求 serving”的第一步。它能让 GPU 一次处理多个 prompt，提高吞吐，但也引入 padding、attention mask、长度差异、首 token logits 选择、padding waste 和 cache 拆分等问题。

一句话概括：

> Batched prefill 的目标是把多个请求的 prompt 同时送进模型，得到每个请求的首 token logits 和初始 KV Cache。

## 10.0 本讲资料边界与第二轮精修口径

本章第二轮精修前，先用公开资料校准口径：Transformers tokenizer 文档把 padding / truncation 作为把不同长度样本整理成 batch tensor 的基本步骤；decoder-only generation 中，padding side、attention mask 和最后位置 logits 选择会直接影响生成正确性；Transformers generation 路径会处理一批输入的生成参数、mask 和 cache；vLLM 等 serving engine 进一步在生产中引入 continuous batching、chunked prefill、max batch tokens、KV block 管理和调度策略。

因此，本章只实现教学版 batched prefill：多个 prompt padding 成 batch，一次 forward，按最后真实 token 取首 token logits，并得到 batch 维度上的初始 KV Cache。它不实现 continuous batching、chunked prefill、packed sequence、paged KV block 或 prefix cache。新增 0 依赖 demo 专门验证左 padding、右 padding、attention mask、gather last logits 和 padding waste。

## 10.1 为什么需要 batched prefill

单请求 prefill 的流程是：

```text
request_1 prompt -> model forward -> request_1 KV Cache -> first token
```

如果有多个请求，一个个处理会浪费 GPU 并行能力：

```text
request_1 prefill
request_2 prefill
request_3 prefill
```

Batched prefill 把它们合起来：

```text
[request_1, request_2, request_3] prompts -> model forward -> 每个请求的 KV Cache 和 first token logits
```

这样能提高 GPU 利用率，尤其是在多个 prompt 长度接近时。

但 LLM prompt 长度往往不同，这就是本章的复杂度来源。

## 10.2 最小输入示例

假设有三个 prompt：

```text
request_1: "Hello"
request_2: "Explain KV Cache in LLM inference"
request_3: "Write a short poem"
```

tokenize 后长度可能是：

```text
request_1: 2 tokens
request_2: 6 tokens
request_3: 3 tokens
```

模型需要张量输入，不能直接输入三个不同长度的 list。因此要 padding 到同一长度。

```text
request_1: [t1, t2, pad, pad, pad, pad]
request_2: [t1, t2, t3,  t4,  t5,  t6 ]
request_3: [t1, t2, t3,  pad, pad, pad]
```

然后用 attention mask 告诉模型哪些位置是真实 token，哪些位置是 padding。

## 10.3 padding_side：左 padding 还是右 padding

在 decoder-only LLM 推理中，padding 方向很重要。

右 padding：

```text
[A, B, C, PAD, PAD]
[D, E, F, G,   H  ]
```

左 padding：

```text
[PAD, PAD, A, B, C]
[D,   E,   F, G, H]
```

很多生成场景更偏向左 padding，因为每个序列的最后一个真实 token 对齐在最右侧，取 `logits[:, -1, :]` 更方便。

如果使用右 padding，`logits[:, -1, :]` 对短序列取到的可能是 PAD 位置的 logits，而不是最后一个真实 token 的 logits。

所以教学版 batched prefill 推荐使用左 padding，或者显式根据每个请求长度 gather 最后一个真实 token 的 logits。

## 10.4 tokenizer 批量编码

Hugging Face tokenizer 支持批量编码：

```python
encoded = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
)
```

如果希望左 padding：

```python
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
```

很多 causal LM 没有单独的 pad token，教学和推理中常用 eos token 作为 pad token。但要注意：pad token 只是填充，不应该被 attention 当成有效上下文。

因此必须传 `attention_mask`。

## 10.5 Batched TokenizerWrapper

我们可以给 `TokenizerWrapper` 增加批量接口。

```python
class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def encode_batch(self, prompts):
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return encoded.input_ids, encoded.attention_mask
```

返回两个张量：

1. `input_ids`: `[batch_size, max_prompt_len]`。
2. `attention_mask`: `[batch_size, max_prompt_len]`。

attention mask 中 1 表示真实 token，0 表示 padding。

## 10.6 修改 ModelWrapper 支持 attention_mask

第 9 章的 `ModelWrapper` 已经支持 `past_key_values`。现在 prefill 要支持 padding，所以要传 `attention_mask`。

```python
class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values
```

这里的重点是：padding 不是简单补齐，还必须让模型知道 padding 不应该参与注意力。

## 10.7 实现 batched_prefill

最小 batched prefill：

```python
def batched_prefill(tokenizer_wrapper, model_wrapper, sampler, prompts):
    input_ids, attention_mask = tokenizer_wrapper.encode_batch(prompts)
    input_ids = input_ids.to(model_wrapper.device)
    attention_mask = attention_mask.to(model_wrapper.device)

    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
    )

    last_logits = gather_last_token_logits(logits, attention_mask)
    next_token_ids = sampler.sample(last_logits)
    return input_ids, attention_mask, next_token_ids, past_key_values
```

如果确定使用左 padding，也可以让 sampler 内部取 `logits[:, -1, :]`。但教学上更推荐保留 `gather_last_token_logits`，因为它能显式处理右 padding 和长度差异。

## 10.8 右 padding 下如何取最后真实位置

真实长度可以从 attention mask 得到：

```python
lengths = attention_mask.sum(dim=-1)
```

最后真实 token 的位置是：

```python
last_indices = lengths - 1
```

从 logits 中取每个请求最后真实位置：

```python
def gather_last_token_logits(logits, attention_mask):
    batch_size = logits.size(0)
    lengths = attention_mask.sum(dim=-1)
    last_indices = lengths - 1
    batch_indices = torch.arange(batch_size, device=logits.device)
    return logits[batch_indices, last_indices, :]
```

然后 sampler 可以对 `[batch_size, vocab_size]` 的 logits 采样。

这就是为什么 padding side 和 logits 位置不能忽略。

## 10.9 Sampler 兼容 batch

第 8 章的 sampler 从 `logits[:, -1, :]` 取下一 token。为了支持通用 batch，可以让 sampler 接收两种形状：

1. `[batch, seq_len, vocab]`：内部取最后位置。
2. `[batch, vocab]`：直接采样。

示例：

```python
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=1.0):
    if logits.dim() == 3:
        next_token_logits = logits[:, -1, :]
    else:
        next_token_logits = logits

    if temperature == 0:
        return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    next_token_logits = apply_temperature(next_token_logits, temperature)
    next_token_logits = apply_top_k(next_token_logits, top_k)
    next_token_logits = apply_top_p(next_token_logits, top_p)

    probs = torch.softmax(next_token_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

如果右 padding 输入没有先 gather，sampler 内部取 `logits[:, -1, :]` 就会出错。

## 10.10 生成每个请求的首 token

Batched prefill 的输出 `next_token_ids` 形状是：

```text
[batch_size, 1]
```

它表示每个请求的第一个输出 token。

例如：

```text
request_1 -> token 318
request_2 -> token 262
request_3 -> token 198
```

这些 token 后续会作为 batched decode 的输入。

对每个请求，我们还需要维护：

1. 原始 prompt tokens。
2. 已生成 tokens。
3. 当前 attention mask 或有效长度。
4. 对应的 KV Cache。
5. 是否已结束。

所以 batched prefill 的结果不能只是一个 tensor，还要回填到 request object。

## 10.11 Request 对象如何保存 prefill 结果

教学版可以这样设计：

```python
class RequestState:
    def __init__(self, request_id, prompt):
        self.request_id = request_id
        self.prompt = prompt
        self.input_ids = None
        self.prompt_len = 0
        self.output_ids = []
        self.next_token_id = None
        self.finished = False
```

batched prefill 后：

```python
for i, request in enumerate(requests):
    request.input_ids = input_ids[i]
    request.prompt_len = int(attention_mask[i].sum().item())
    request.next_token_id = next_token_ids[i]
    request.output_ids.append(next_token_ids[i].item())
```

真实 serving engine 还会记录每个请求对应的 cache block，而不是直接把整个 `past_key_values` 放在 request 里。

## 10.12 past_key_values 的 batch 维度

batched prefill 返回的 `past_key_values` 是一个 batch cache。

直觉形状类似：

```text
layer_i.key:   [batch, num_kv_heads, max_prompt_len, head_dim]
layer_i.value: [batch, num_kv_heads, max_prompt_len, head_dim]
```

如果用了 padding，cache 的 `max_prompt_len` 包含 padded positions；有效 KV 长度仍然要由 attention mask 或 per-request length 来解释。

教学版可以先整体保存它，后续 batched decode 继续用同一个 batch 顺序。

但真实 serving engine 中，请求会动态加入和退出，不能永远依赖固定 batch 顺序。这就是后续 continuous batching 和 cache manager 要解决的问题。

## 10.13 padding 带来的浪费

Batched prefill 的问题是 padding 会浪费计算。

设 batch size 为 `B`，每个请求真实 prompt 长度为 `L_i`，batch 最大长度为：

```math
L_{\max}=\max_i L_i
```

实际张量 token 数：

```math
T_{\mathrm{tensor}}=B L_{\max}
```

真实有效 token 数：

```math
T_{\mathrm{real}}=\sum_i L_i
```

padding waste ratio：

```math
R_{\mathrm{pad}}=\frac{T_{\mathrm{tensor}}-T_{\mathrm{real}}}{T_{\mathrm{tensor}}}
```

如果一个 batch 长度分布是：

```text
10, 20, 30, 4000
```

padding 到 4000 后，前三个请求会有大量 padding。

```text
有效 token = 10 + 20 + 30 + 4000 = 4060
实际张量 token = 4 * 4000 = 16000
```

所以生产系统常按长度分桶，或用更高效的变长 attention / packed prefill，减少 padding 浪费。

教学版用 padding 是为了简单，真实系统会进一步优化。

## 10.14 长 prompt、token budget 和 chunked prefill

如果一个请求 prompt 很长，把它和短请求放在同一个 prefill batch 里，会拖慢整个 batch。

常见策略包括：

1. 长短请求分桶。
2. 限制单轮 prefill token budget。
3. 对长 prompt 做 chunked prefill。
4. 让 decode 请求穿插执行，避免输出卡顿。
5. 使用 prefix cache 复用共享前缀。

一个简单 token budget 可以写成：

```math
\sum_{i\in\mathcal{B}} L_i \le B_{\mathrm{prefill}}
```

其中 `B_prefill` 是本轮允许处理的真实 prompt tokens 预算。生产系统还会结合 padded tokens、KV budget、queue wait 和 decode TPOT 做更复杂的调度。

本章先实现基础 batched prefill，后续讲 vLLM 和高级 serving 时会继续展开这些策略。

## 10.15 batched prefill 公式、padding 风险和可运行 demo

下面这个 demo 不依赖 PyTorch。它实现一个 toy tokenizer、左右 padding、attention mask、toy model logits 和最后真实位置 gather。

它验证三件事：

1. 左 padding 下取最后位置 logits 可以得到每个请求的首 token。
2. 右 padding 下直接取最后位置会让短请求读到 PAD 位置，结果错误。
3. 用 attention mask gather 最后真实 token 后，右 padding 也能得到正确首 token。

```python
VOCAB = {
    "<pad>": 0,
    "<eos>": 1,
    "Hello": 2,
    "LLM": 3,
    "Explain": 4,
    "KV": 5,
    "Cache": 6,
    "in": 7,
    "inference": 8,
    "Write": 9,
    "short": 10,
    "poem": 11,
    "!": 12,
    ".": 13,
}
ID_TO_TOKEN = {idx: tok for tok, idx in VOCAB.items()}
NEXT_TOKEN = {
    VOCAB["LLM"]: VOCAB["!"],
    VOCAB["inference"]: VOCAB["."],
    VOCAB["poem"]: VOCAB["."],
    VOCAB["<pad>"]: VOCAB["<eos>"],
}


def encode(prompt):
    return [VOCAB[token] for token in prompt.split()]


def pad_batch(sequences, side):
    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    for seq in sequences:
        pad_count = max_len - len(seq)
        if side == "left":
            row = [VOCAB["<pad>"]] * pad_count + seq
            mask = [0] * pad_count + [1] * len(seq)
        else:
            row = seq + [VOCAB["<pad>"]] * pad_count
            mask = [1] * len(seq) + [0] * pad_count
        input_ids.append(row)
        attention_mask.append(mask)
    return input_ids, attention_mask


def toy_logits_for_token(token_id):
    logits = [-10.0] * len(VOCAB)
    logits[NEXT_TOKEN.get(token_id, VOCAB["<eos>"])] = 10.0
    return logits


def toy_model_forward(input_ids):
    return [[toy_logits_for_token(token_id) for token_id in row] for row in input_ids]


def argmax(logits):
    return max(range(len(logits)), key=lambda idx: logits[idx])


def sample_from_last_position(logits_batch):
    return [argmax(row[-1]) for row in logits_batch]


def gather_last_token_logits(logits_batch, attention_mask):
    gathered = []
    last_indices = []
    for logits, mask in zip(logits_batch, attention_mask):
        length = sum(mask)
        last_index = length - 1
        last_indices.append(last_index)
        gathered.append(logits[last_index])
    return gathered, last_indices


def decode_ids(ids):
    return [ID_TO_TOKEN[idx] for idx in ids]


prompts = [
    "Hello LLM",
    "Explain KV Cache in LLM inference",
    "Write short poem",
]
encoded = [encode(prompt) for prompt in prompts]
lengths = [len(seq) for seq in encoded]

left_ids, left_mask = pad_batch(encoded, "left")
right_ids, right_mask = pad_batch(encoded, "right")

left_logits = toy_model_forward(left_ids)
right_logits = toy_model_forward(right_ids)

left_next = sample_from_last_position(left_logits)
right_naive_next = sample_from_last_position(right_logits)
right_last_logits, right_last_indices = gather_last_token_logits(right_logits, right_mask)
right_gather_next = [argmax(logits) for logits in right_last_logits]

batch_size = len(prompts)
max_len = max(lengths)
real_tokens = sum(lengths)
tensor_tokens = batch_size * max_len
padding_waste = round((tensor_tokens - real_tokens) / tensor_tokens, 3)
layers = 2
kv_heads = 1
head_dim = 4
dtype_bytes = 2
kv_bytes_real = 2 * layers * real_tokens * kv_heads * head_dim * dtype_bytes
kv_bytes_padded = 2 * layers * tensor_tokens * kv_heads * head_dim * dtype_bytes

print("lengths=", lengths)
print("left_input_ids=", left_ids)
print("left_attention_mask=", left_mask)
print("left_next=", decode_ids(left_next))
print("right_naive_next=", decode_ids(right_naive_next))
print("right_last_indices=", right_last_indices)
print("right_gather_next=", decode_ids(right_gather_next))
print("padding_waste=", padding_waste)
print("kv_bytes_real=", kv_bytes_real)
print("kv_bytes_padded=", kv_bytes_padded)
print("batched_prefill_gate=", left_next == right_gather_next and right_naive_next != right_gather_next)
```

一组稳定输出如下：

```text
lengths= [2, 6, 3]
left_input_ids= [[0, 0, 0, 0, 2, 3], [4, 5, 6, 7, 3, 8], [0, 0, 0, 9, 10, 11]]
left_attention_mask= [[0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
left_next= ['!', '.', '.']
right_naive_next= ['<eos>', '.', '<eos>']
right_last_indices= [1, 5, 2]
right_gather_next= ['!', '.', '.']
padding_waste= 0.389
kv_bytes_real= 352
kv_bytes_padded= 576
batched_prefill_gate= True
```

输出怎么读：

1. 左 padding 下，所有请求最后位置都是真实 token，所以 `left_next` 正确。
2. 右 padding 下，短请求最后位置是 PAD，`right_naive_next` 错误地输出 `<eos>`。
3. 用 attention mask 找最后真实位置后，`right_gather_next` 和左 padding 结果一致。
4. `padding_waste=0.389` 说明这个 batch 里接近 39% 的张量位置是 padding。
5. `kv_bytes_padded` 大于 `kv_bytes_real`，说明简单 padding 版 cache / 计算会引入额外浪费。

## 10.16 最小完整代码骨架

把关键部分组合起来：

```python
def batched_prefill(tokenizer_wrapper, model_wrapper, sampler, prompts):
    input_ids, attention_mask = tokenizer_wrapper.encode_batch(prompts)
    input_ids = input_ids.to(model_wrapper.device)
    attention_mask = attention_mask.to(model_wrapper.device)

    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=None,
    )

    last_logits = gather_last_token_logits(logits, attention_mask)
    next_token_ids = sampler.sample(last_logits)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "next_token_ids": next_token_ids,
        "past_key_values": past_key_values,
    }
```

如果明确只使用左 padding，可以简化为：

```python
next_token_ids = sampler.sample(logits)
```

但只要后续要兼容右 padding、packed prefill 或复杂 request state，显式 gather 最后真实位置会更稳。

## 10.17 和生产 engine 的差距

这个版本还很简化。

它没有处理：

1. 动态请求加入。
2. 长 prompt chunking。
3. packed sequence。
4. cache block 分配。
5. prefill 和 decode 混合调度。
6. prefix cache。
7. 不同请求 sampling params。
8. 单请求 prefill 失败后的局部错误处理。
9. 显存预算和 admission control。

但它已经把多个 prompt 合成 batch，完成了从单请求到多请求执行的第一步。

## 10.18 常见误区

误区一：batch 只是把 prompts 放进 list。

真正执行前还要 padding、attention mask、设备放置、logits 位置选择和 cache 管理。

误区二：有 padding 就一定没问题。

padding 必须配合 attention mask，否则模型会把 padding 当成上下文。

误区三：总是可以取 `logits[:, -1, :]`。

只有在左 padding 或无 padding 等情况下通常成立；右 padding 下短序列最后位置可能是 PAD。

误区四：batch 越大越好。

batch 大可能提高吞吐，但也可能增加 padding 浪费、TTFT 和显存压力。

误区五：batched prefill 就等于 continuous batching。

不是。batched prefill 只是批量处理 prompt，continuous batching 是每轮动态维护 running batch。

## 10.19 面试追问

1. Batched prefill 的目标是什么？
2. 为什么不同长度 prompt 需要 padding？
3. attention mask 在 batched prefill 中有什么作用？
4. 左 padding 和右 padding 对取 logits 有什么影响？
5. 为什么长 prompt 会导致 padding 浪费？
6. batched prefill 返回的 KV Cache 在 batch 维度上如何理解？
7. batched prefill 和 continuous batching 有什么区别？
8. 如何减少 batched prefill 中的 padding 浪费？

参考回答思路：

1. 先说 batched prefill 把多个 prompt 合并，一次 forward 得到首 token logits 和初始 KV Cache。
2. 再说不同长度需要 padding，attention mask 屏蔽 padding。
3. 然后强调 logits 位置选择，左 padding 可以简化最后位置取 logits，右 padding 要按长度 gather。
4. 最后补充工程 trade-off：batch 提升吞吐，但可能增加 padding、显存、TTFT 和 cache 管理复杂度。

## 10.20 小练习

1. 实现 `encode_batch`，打印 `input_ids` 和 `attention_mask`。
2. 分别用左 padding 和右 padding，观察 `logits[:, -1, :]` 的含义差异。
3. 写出 `gather_last_token_logits` 并在右 padding 下测试。
4. 构造长度差异很大的 prompts，计算 padding waste ratio。
5. 思考如何按 prompt 长度分桶以减少 padding。
6. 给一个 token budget，设计一个简单规则，把等待队列中的请求组成 prefill batch。

## 10.21 本章小结

本章实现了最小 batched prefill。

Batched prefill 把多个 prompt 合并成一个 batch，一次模型 forward 得到每个请求的首 token logits 和初始 KV Cache。关键难点包括 padding、attention mask、最后真实 token logits 的选择、batch 维度上的 cache 理解，以及长短 prompt 混合带来的 padding 浪费。

下一章我们会继续实现 batched decode，让多个已经完成 prefill 的请求能够一起逐 token 生成。
