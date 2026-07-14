# 第 9 章 实现 KV Cache 的最小版本

上一章我们实现了 sampling。本章解决 naive generate loop 最大的性能问题：每生成一个 token，都重新计算完整历史序列。解决办法就是 KV Cache。

KV Cache 是 LLM 推理框架的核心。vLLM 的 PagedAttention、SGLang 的 prefix sharing、PD 分离里的 KV 迁移，本质上都围绕 KV Cache 做文章。本章先不讲复杂管理，只实现最小版本：单请求、连续缓存、使用 `past_key_values` 复用历史。

一句话概括：

> KV Cache 让 decode 阶段不再重复计算历史 token 的 key/value，而是复用缓存，只为新 token 计算 query/key/value。

## 9.0 本讲资料边界与第二轮精修口径

本章第二轮精修前，先用公开资料校准口径：Transformers 文档把 cache 解释为自回归生成中保存历史 key/value states，用来避免每一步重复计算完整上下文；Transformers `generate()` 和模型 forward 路径通常通过 `use_cache`、`past_key_values`、`Cache` 对象、`attention_mask` 和位置相关信息协同工作；vLLM 的 PagedAttention 论文和文档进一步把问题推进到生产 serving 中的 KV cache block 管理、非连续物理存储、按需分配、共享前缀和碎片控制。

因此，本章只讨论最小单请求 KV Cache：prefill 建 cache，decode 追加 cache，验证输出等价和重复计算下降。它不实现 PagedAttention、block manager、prefix cache、多请求调度、KV offload、KV quantization 或 PD 分离。正文里的 Hugging Face / PyTorch 代码用于贴近工程接口，新增 0 依赖 demo 用纯 Python 展示最小 cache 机制和审计指标。

## 9.1 naive loop 的重复计算

第 7 章的 generate loop 每一轮都把完整 `generated_ids` 送进模型：

```python
for _ in range(max_new_tokens):
    logits = model_wrapper.forward(generated_ids)
    next_token_id = sampler.sample(logits)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
```

如果 prompt 有 1000 个 token，生成 100 个 token，模型处理的序列长度大致是：

```text
1000, 1001, 1002, ..., 1099
```

历史 token 会被一遍遍重新计算。自回归生成真正需要的是：历史已经算过的 key/value 保存起来，每轮只处理新 token。

可以用一个简单工作量近似看差异。设 prompt 长度为 `N`，生成 `M` 个 token。

naive loop 每轮处理完整上下文：

```math
W_{\mathrm{naive}}=\sum_{t=0}^{M-1}(N+t)
=MN+\frac{M(M-1)}{2}
```

带 KV Cache 的最小 loop 先 prefill prompt，再每轮 decode 一个 token：

```math
W_{\mathrm{kv}}=N+M
```

这个公式不是精确 FLOPs，只是说明重复计算规模。真实 Transformer 中 attention、MLP、kernel、batch 和访存都会影响耗时，但方向很稳定：cache 用额外显存换掉大量历史 token 的重复计算。

## 9.2 KV Cache 保存了什么

Transformer 的 attention 会为每个 token 计算 query、key、value。

生成新 token 时，新 token 的 query 需要和历史 token 的 key/value 做 attention。历史 token 的 key/value 在 causal decoder 中不会因为后面生成新 token 而改变，所以可以缓存。

KV Cache 保存的是：

1. 每一层的 key cache。
2. 每一层的 value cache。
3. 每个历史 token 对应的 key/value。
4. 每个 KV head 对应的 key/value。

简化结构：

```text
past_key_values[layer] = (past_key, past_value)
```

常见 shape 可以写成：

```text
key/value: [batch, num_kv_heads, seq_len, head_dim]
```

注意这里是 `num_kv_heads`，不一定等于 query heads。MHA 中二者相同；GQA / MQA 中 KV head 更少，KV Cache 显存也更低。

## 9.3 Prefill 和 Decode 中的 cache

KV Cache 的使用通常分成两步。

第一步，prefill：

```text
输入完整 prompt -> 模型 forward -> 返回 logits 和 past_key_values
```

第二步，decode：

```text
输入上一个 token + past_key_values -> 模型 forward -> 返回新 logits 和更新后的 past_key_values
```

也就是说，prefill 建立初始 cache，decode 不断扩展 cache。

设 prompt token 为 `x_1,...,x_N`，生成 token 为 `y_1,...,y_M`。prefill 后 cache 长度为：

```math
T_0=N
```

第 `t` 次 decode 后 cache 长度为：

```math
T_t=N+t
```

decode 的 logits 应该只从最后一个有效位置取：

```math
z_t=f_{\theta}(y_{t-1},C_{t-1})_{\mathrm{last}}
```

其中 `C_{t-1}` 是已经缓存的历史 key/value。

## 9.4 Hugging Face 中的 past_key_values

很多 causal LM 支持 `use_cache=True`，并在输出中返回 `past_key_values` 或新版 cache 对象。

第一次 prefill：

```python
outputs = model(input_ids=input_ids, use_cache=True)
logits = outputs.logits
past_key_values = outputs.past_key_values
```

后续 decode：

```python
outputs = model(
    input_ids=next_token_id,
    past_key_values=past_key_values,
    use_cache=True,
)
logits = outputs.logits
past_key_values = outputs.past_key_values
```

注意，decode 阶段传入的 `input_ids` 通常只包含最新 token，而不是完整历史序列。

版本边界也要注意：不同 Transformers 版本和不同模型可能使用 tuple 形式的 `past_key_values`，也可能使用新的 `Cache` / `DynamicCache` 抽象。面试中不需要背具体类名，但要讲清接口语义：cache 是模型 forward 的状态输入和状态输出。

## 9.5 修改 ModelWrapper

我们先把 `ModelWrapper.forward` 改成支持 cache。

```python
class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values=None, attention_mask=None, position_ids=None):
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values
```

对外接口从：

```python
logits = model_wrapper.forward(input_ids)
```

变成：

```python
logits, past_key_values = model_wrapper.forward(input_ids, past_key_values)
```

这一步看起来小，但意义很大：模型执行从无状态调用变成了带缓存状态的调用。

## 9.6 带 KV Cache 的 generate loop

最小带 cache 版本如下：

```python
def generate_with_kv_cache(tokenizer_wrapper, model_wrapper, sampler, prompt, max_new_tokens=128):
    input_ids = tokenizer_wrapper.encode(prompt).to(model_wrapper.device)
    generated_ids = input_ids

    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        past_key_values=None,
    )

    for _ in range(max_new_tokens):
        next_token_id = sampler.sample(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

        logits, past_key_values = model_wrapper.forward(
            input_ids=next_token_id,
            past_key_values=past_key_values,
        )

    return tokenizer_wrapper.decode(generated_ids[0].tolist())
```

核心变化是：

1. 第一次输入完整 prompt。
2. 保存 `past_key_values`。
3. 后续每轮只输入上一个 token。
4. 每轮更新 `past_key_values`。

这段代码已经体现了 prefill 和 decode 的区别：第一次 forward 输入完整 prompt，后续 forward 只输入新 token。

## 9.7 更清晰的 prefill/decode 写法

为了贴近 serving engine，可以把 prefill 和 decode 拆成两个函数。

```python
def prefill(model_wrapper, input_ids, attention_mask=None):
    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        past_key_values=None,
        attention_mask=attention_mask,
    )
    return logits, past_key_values


def decode_one_step(model_wrapper, next_token_id, past_key_values, attention_mask=None, position_ids=None):
    logits, past_key_values = model_wrapper.forward(
        input_ids=next_token_id,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    return logits, past_key_values
```

generate loop 变成：

```python
logits, past_key_values = prefill(model_wrapper, input_ids)

while not finished:
    next_token_id = sampler.sample(logits)
    logits, past_key_values = decode_one_step(
        model_wrapper,
        next_token_id,
        past_key_values,
    )
```

这就是后续实现 batched prefill 和 batched decode 的基础。

## 9.8 cache 的显存公式

KV Cache 会随着 `seq_len` 增长。

设：

1. `B` 是 active batch size。
2. `T` 是当前 cache token 长度。
3. `L` 是 transformer 层数。
4. `H_kv` 是 KV heads 数。
5. `D_h` 是每个 head 的维度。
6. `b` 是每个元素的字节数，例如 FP16 / BF16 常用 2 字节。

单个 active batch 的 KV Cache 显存近似为：

```math
M_{\mathrm{kv}}=2BLTH_{\mathrm{kv}}D_hb
```

前面的 `2` 来自 key 和 value 两份缓存。

如果多个请求长度不同，更准确的估算应该按活跃请求求和：

```math
M_{\mathrm{kv}}=2L H_{\mathrm{kv}}D_hb\sum_i T_i
```

这也解释了为什么 KV Cache 会影响并发：上下文越长、输出越长、活跃请求越多，显存压力越大。

## 9.9 为什么 cache 会影响并发

单请求时，KV Cache 是一个优化。多请求时，它是并发上限的重要约束。

如果每个请求都保存自己的 cache，那么并发越高，cache 总量越大。

```text
总 KV Cache 约正比于：active_requests * average_context_length
```

所以 serving engine 不只是“用不用 cache”，还要管理 cache：分配、释放、复用、抢占、分页、迁移。

最小单请求 cache 是连续增长的列表或张量；生产系统里的 cache manager 需要回答更多问题：

1. 新请求能不能准入。
2. 还有多少 KV block 可用。
3. 请求结束或取消后是否及时释放。
4. prefix cache 是否能复用公共前缀。
5. cache 过高时是拒绝、等待、抢占、swap 还是降级。

本章只实现最小版本，后续讲 vLLM 时会进入 block manager 和 PagedAttention。

## 9.10 attention_mask 和 position_ids

真实模型调用里，除了 `input_ids` 和 `past_key_values`，常常还要处理 `attention_mask` 和 `position_ids`。

简化理解：

1. `attention_mask` 告诉模型哪些 token 是有效 token。
2. `position_ids` 告诉模型当前 token 的位置。

在单请求、无 padding、使用标准 `generate()` 路径时，很多模型会自动处理。

但在自己实现 batch decode 时，这些不能忽略。多个请求长度不同，新 token 的位置也不同。若 position 错位，RoPE / absolute position embedding 看到的位置就错了；若 mask 长度和 cache 长度不一致，模型可能看不到该看的历史，或者错误看见 padding。

decode 阶段的关键约束可以写成：

```math
\mathrm{len}(\mathrm{attention\_mask})=T_{\mathrm{cache}}+T_{\mathrm{new}}
```

单 token decode 时：

```math
\mathrm{position\_id}=T_{\mathrm{cache}}
```

这里的 `T_cache` 是当前新 token 之前已经缓存的历史长度。

## 9.11 cache 和输出一致性

理论上，在相同采样策略、相同随机种子、相同模型实现下，使用 KV Cache 和不使用 KV Cache 应该得到等价结果。

但实际可能出现微小差异，原因包括：

1. 浮点计算顺序不同。
2. kernel 实现不同。
3. attention mask 或 position_ids 处理错误。
4. 模型是否正确支持 cache。
5. batch 和单请求路径不同。

如果使用 cache 后输出明显变坏，优先排查：

1. 是否只在 decode 阶段传入最新 token。
2. `past_key_values` 是否正确更新。
3. position 是否正确递增。
4. attention mask 是否和 cache 长度匹配。
5. EOS 和停止逻辑是否被破坏。

## 9.12 KV Cache 公式、机制和可运行 demo

下面这个 demo 不依赖 PyTorch。它用一个极小的单头 causal attention 模拟两条路径：

1. `full_last_hidden`：每次对完整序列重新计算 Q/K/V，再取最后 token 输出。
2. `cached_decode_step`：prefill 缓存历史 K/V，decode 时只为新 token 计算 Q/K/V，并把新 K/V 追加进 cache。

demo 的目标不是复刻真实 Transformer，而是验证三个核心点：cache 输出和全量重算一致、cache 长度随 token 追加增长、重复工作量明显下降。

```python
from math import exp, sqrt


TOKENS = ["<eos>", "LLM", "inference", "uses", "KV", "cache", "."]
EMB = {
    1: [0.20, 0.10, 0.00, 0.30],
    2: [0.00, 0.30, 0.10, 0.20],
    3: [0.40, 0.00, 0.20, 0.10],
    4: [0.10, 0.40, 0.30, 0.00],
    5: [0.30, 0.20, 0.40, 0.10],
    6: [0.10, 0.00, 0.20, 0.50],
}


def matvec(vec, mat):
    return [sum(vec[i] * mat[i][j] for i in range(len(vec))) for j in range(len(mat[0]))]


W_Q = [
    [0.5, 0.1, 0.0, 0.0],
    [0.0, 0.4, 0.1, 0.0],
    [0.0, 0.0, 0.3, 0.2],
    [0.1, 0.0, 0.0, 0.5],
]
W_K = [
    [0.4, 0.0, 0.1, 0.0],
    [0.1, 0.5, 0.0, 0.0],
    [0.0, 0.1, 0.4, 0.1],
    [0.0, 0.0, 0.2, 0.5],
]
W_V = [
    [0.3, 0.1, 0.0, 0.0],
    [0.0, 0.2, 0.1, 0.0],
    [0.1, 0.0, 0.4, 0.1],
    [0.0, 0.0, 0.1, 0.3],
]


def softmax(xs):
    m = max(xs)
    exps = [exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def project(token_id):
    x = EMB[token_id]
    return {
        "q": matvec(x, W_Q),
        "k": matvec(x, W_K),
        "v": matvec(x, W_V),
    }


def attend_last(query, keys, values):
    scale = sqrt(len(query))
    scores = [sum(q * k for q, k in zip(query, key)) / scale for key in keys]
    probs = softmax(scores)
    out = [
        sum(prob * value[j] for prob, value in zip(probs, values))
        for j in range(len(values[0]))
    ]
    return [round(x, 6) for x in out], [round(p, 6) for p in probs]


def full_last_hidden(token_ids):
    projected = [project(tid) for tid in token_ids]
    keys = [item["k"] for item in projected]
    values = [item["v"] for item in projected]
    return attend_last(projected[-1]["q"], keys, values)


def prefill(token_ids):
    projected = [project(tid) for tid in token_ids]
    return {
        "keys": [item["k"] for item in projected],
        "values": [item["v"] for item in projected],
        "project_calls": len(token_ids),
    }


def cached_decode_step(token_id, cache):
    projected = project(token_id)
    cache["keys"].append(projected["k"])
    cache["values"].append(projected["v"])
    cache["project_calls"] += 1
    hidden, probs = attend_last(projected["q"], cache["keys"], cache["values"])
    return hidden, probs, cache


prompt = [1, 2, 3]          # LLM inference uses
generated = [4, 5, 6]       # KV cache .
cache = prefill(prompt)

rows = []
for token_id in generated:
    full_ids = prompt + [row["token_id"] for row in rows] + [token_id]
    full_hidden, full_probs = full_last_hidden(full_ids)
    cache_hidden, cache_probs, cache = cached_decode_step(token_id, cache)
    rows.append({
        "token_id": token_id,
        "token": TOKENS[token_id],
        "seq_len": len(full_ids),
        "match": full_hidden == cache_hidden,
        "hidden": cache_hidden,
        "attn_last": cache_probs[-1],
    })

prompt_len = len(prompt)
new_tokens = len(generated)
naive_project_calls = sum(prompt_len + i for i in range(1, new_tokens + 1))
kv_project_calls = cache["project_calls"]
d_h = len(EMB[1])
layers = 2
kv_heads = 1
dtype_bytes = 2
kv_bytes = 2 * layers * len(cache["keys"]) * kv_heads * d_h * dtype_bytes

print("rows=", rows)
print("final_cache_len=", len(cache["keys"]))
print("naive_project_calls=", naive_project_calls)
print("kv_project_calls=", kv_project_calls)
print("saved_project_calls=", naive_project_calls - kv_project_calls)
print("kv_bytes_toy=", kv_bytes)
print("kv_cache_gate=", all(row["match"] for row in rows) and len(cache["keys"]) == prompt_len + new_tokens)
```

一组稳定输出如下：

```text
rows= [{'token_id': 4, 'token': 'KV', 'seq_len': 4, 'match': True, 'hidden': [0.06739, 0.057621, 0.09523, 0.059904], 'attn_last': 0.252303}, {'token_id': 5, 'token': 'cache', 'seq_len': 5, 'match': True, 'hidden': [0.080177, 0.060045, 0.114252, 0.061983], 'attn_last': 0.202015}, {'token_id': 6, 'token': '.', 'seq_len': 6, 'match': True, 'hidden': [0.074965, 0.051437, 0.116638, 0.080451], 'attn_last': 0.17009}]
final_cache_len= 6
naive_project_calls= 15
kv_project_calls= 6
saved_project_calls= 9
kv_bytes_toy= 192
kv_cache_gate= True
```

输出怎么读：

1. `match=True` 说明在这个 toy attention 中，缓存路径和完整重算路径的最后 token hidden state 完全一致。
2. `final_cache_len=6` 说明 prompt 3 个 token 加生成 3 个 token 后，cache 长度增长到 6。
3. `naive_project_calls=15`，`kv_project_calls=6` 说明 cache 避免了重复投影历史 token。
4. `kv_bytes_toy=96` 展示 KV Cache 会真实占用额外内存；真实模型要再乘以层数、KV head、head dim、batch 和 dtype bytes。

## 9.13 这个版本还缺什么

最小 KV Cache 版本仍然不是 serving engine。

它缺少：

1. batch prefill。
2. batch decode。
3. 多请求 cache 管理。
4. cache 分配和释放。
5. 长度不同请求的 padding 和 position 处理。
6. 显存预算控制。
7. cache block 管理。
8. prefix cache 复用。
9. 请求取消后的 cache 清理。

但它已经解决了最关键的重复计算问题，并为后续批处理打下基础。

## 9.14 常见误区

误区一：KV Cache 会减少模型权重显存。

不会。KV Cache 是额外缓存，会增加显存占用，但减少重复计算。

误区二：有了 KV Cache，推理一定更省显存。

不是。KV Cache 通常用显存换速度。高并发和长上下文时，KV Cache 可能成为显存瓶颈。

误区三：decode 阶段还要传完整历史 `input_ids`。

使用 cache 时，decode 通常只传最新 token，同时传入 `past_key_values`。完整历史已经体现在 cache 中。

误区四：单请求 cache 管理就等于生产 cache manager。

生产系统要处理多请求、碎片、释放、抢占、复用、准入和显存预算。

误区五：cache 只影响性能，不影响正确性。

position_ids、attention_mask、cache 更新错误都会直接影响输出正确性。

## 9.15 面试追问

1. KV Cache 保存的是什么？
2. 为什么 KV Cache 能加速自回归 decode？
3. Prefill 和 Decode 在 cache 使用上有什么区别？
4. 使用 `past_key_values` 后，decode 阶段为什么只输入最新 token？
5. KV Cache 为什么会增加显存占用？
6. 多请求 serving 中 KV Cache 管理有哪些难点？
7. 为什么 position_ids 和 attention_mask 在 batched decode 中很重要？
8. 使用 KV Cache 后输出异常，你会怎么排查？

参考回答思路：

1. 先说 cache 保存每层历史 token 的 key/value。
2. 再说 prefill 建 cache，decode 复用 cache，只计算新 token。
3. 然后说明它用显存换计算，能降低重复计算但增加显存压力。
4. 最后补充生产难点：多请求分配释放、碎片、prefix 复用、position / mask 正确性和 KV capacity admission。

## 9.16 小练习

1. 在第 7 章代码基础上加入 `past_key_values`。
2. 打印每层 cache 的 shape，观察 seq_len 如何增长。
3. 对比使用 cache 和不使用 cache 的生成耗时。
4. 故意在 decode 阶段传完整 `generated_ids`，观察性能和行为变化。
5. 思考如果 batch 中每个请求长度不同，position_ids 应该如何设置。
6. 用本章显存公式估算一个 32 层、32 KV heads、head dim 128、FP16、上下文 4096 的单请求 KV Cache 显存。

## 9.17 本章小结

本章实现了最小 KV Cache。

naive generate loop 每轮重复计算完整历史序列，KV Cache 通过保存历史 key/value，让 decode 阶段只计算新 token。最小实现依赖 `past_key_values`：prefill 输入完整 prompt 并建立 cache，decode 输入最新 token 并更新 cache。

KV Cache 的 trade-off 很清楚：它用额外显存换更少重复计算。单请求里这是性能优化；生产 serving 里它会进一步演变成 KV block、admission、eviction、prefix reuse、paged memory 和跨节点迁移等系统问题。

下一章我们会继续升级，把单请求 prefill 扩展成 batched prefill，让多个请求的 prompt 能一起进入模型执行。
