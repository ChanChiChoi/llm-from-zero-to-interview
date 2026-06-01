# 第 9 章 实现 KV Cache 的最小版本

上一章我们实现了 sampling。本章解决 naive generate loop 最大的性能问题：每生成一个 token，都重新计算完整历史序列。解决办法就是 KV Cache。

KV Cache 是 LLM 推理框架的核心。vLLM 的 PagedAttention、SGLang 的 prefix sharing、PD 分离里的 KV 迁移，本质上都围绕 KV Cache 做文章。本章先不讲复杂管理，只实现最小版本：单请求、连续缓存、使用 `past_key_values` 复用历史。

一句话概括：

> KV Cache 让 decode 阶段不再重复计算历史 token，而是复用历史 key/value，只计算新 token。

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

这意味着历史 token 被一遍遍重新计算。

自回归生成真正需要的是：历史已经算过的部分保存起来，每轮只算新 token。

## 9.2 KV Cache 保存了什么

Transformer 的 attention 会为每个 token 计算 query、key、value。

生成新 token 时，新 token 的 query 需要和历史 token 的 key/value 做 attention。

历史 token 的 key/value 不会因为后面生成新 token 而改变，所以可以缓存。

KV Cache 保存的是：

1. 每一层的 key cache。
2. 每一层的 value cache。
3. 每个历史 token 对应的 key/value。
4. 每个 attention head 对应的 key/value。

简化理解：

```text
past_key_values[layer] = (past_key, past_value)
```

有了它，decode 时就不必重新计算 prompt 的 key/value。

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

这和第 4 章讲的生命周期完全一致。

## 9.4 Hugging Face 中的 past_key_values

很多 causal LM 支持 `use_cache=True`，并在输出中返回 `past_key_values`。

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

## 9.5 修改 ModelWrapper

我们先把 `ModelWrapper.forward` 改成支持 cache。

```python
class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values=None):
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
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
    past_key_values = None

    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        past_key_values=None,
    )

    next_token_id = sampler.sample(logits)
    generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    for _ in range(max_new_tokens - 1):
        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

        logits, past_key_values = model_wrapper.forward(
            input_ids=next_token_id,
            past_key_values=past_key_values,
        )

        next_token_id = sampler.sample(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    return tokenizer_wrapper.decode(generated_ids[0].tolist())
```

核心变化是：

1. 第一次输入完整 prompt。
2. 保存 `past_key_values`。
3. 后续每轮只输入上一个 token。
4. 每轮更新 `past_key_values`。

## 9.7 更清晰的 prefill/decode 写法

为了贴近 serving engine，可以把 prefill 和 decode 拆成两个函数。

```python
def prefill(model_wrapper, input_ids):
    logits, past_key_values = model_wrapper.forward(
        input_ids=input_ids,
        past_key_values=None,
    )
    return logits, past_key_values


def decode_one_step(model_wrapper, next_token_id, past_key_values):
    logits, past_key_values = model_wrapper.forward(
        input_ids=next_token_id,
        past_key_values=past_key_values,
    )
    return logits, past_key_values
```

generate loop 变成：

```python
logits, past_key_values = prefill(model_wrapper, input_ids)
next_token_id = sampler.sample(logits)

while not finished:
    logits, past_key_values = decode_one_step(model_wrapper, next_token_id, past_key_values)
    next_token_id = sampler.sample(logits)
```

这就是后续实现 batched prefill 和 batched decode 的基础。

## 9.8 cache 的形状直觉

不同模型实现细节不同，但常见 KV Cache 形状类似：

```text
num_layers 个元素
每层包含 key 和 value
key/value shape 类似 [batch, num_heads, seq_len, head_dim]
```

也就是说，cache 会随着 `seq_len` 增长。

如果 prompt 长度是 1000，生成了 100 个 token，那么 cache 中的序列长度大约是 1100。

这也解释了为什么 KV Cache 会占用大量显存：它和层数、head 数、head_dim、batch、上下文长度都有关。

## 9.9 为什么 cache 会影响并发

单请求时，KV Cache 只是一个优化。多请求时，它是并发上限的重要约束。

如果每个请求都保存自己的 cache，那么并发越高，cache 总量越大。

```text
总 KV Cache 约正比于：active_requests * average_context_length
```

上下文越长、并发越高，显存压力越大。

所以 serving engine 不只是“用不用 cache”，还要管理 cache：分配、释放、复用、抢占、分页、迁移。

本章只实现最小版本，后续讲 vLLM 时会进入 block manager 和 PagedAttention。

## 9.10 attention_mask 和 position_ids

真实模型调用里，除了 `input_ids` 和 `past_key_values`，有时还要处理 `attention_mask` 和 `position_ids`。

简化理解：

1. `attention_mask` 告诉模型哪些 token 是有效 token。
2. `position_ids` 告诉模型当前 token 的位置。

在单请求、无 padding、使用标准 Hugging Face generate 路径时，很多模型会自动处理。

但在自己实现 batch decode 时，这些就不能忽略。

例如多个请求长度不同，position_ids 必须对应每个请求当前的新 token 位置，否则模型会误解 token 的位置。

这也是为什么 batched decode 比单请求 decode 复杂得多。

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

## 9.12 最小 KV Cache 代码骨架

把本章代码合起来：

```python
class ModelWrapper:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def forward(self, input_ids, past_key_values=None):
        input_ids = input_ids.to(self.device)
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values


def generate_with_kv_cache(tokenizer_wrapper, model_wrapper, sampler, prompt, max_new_tokens=128):
    input_ids = tokenizer_wrapper.encode(prompt).to(model_wrapper.device)
    generated_ids = input_ids
    past_key_values = None

    logits, past_key_values = model_wrapper.forward(input_ids, past_key_values)

    for _ in range(max_new_tokens):
        next_token_id = sampler.sample(logits)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

        if next_token_id.item() == tokenizer_wrapper.eos_token_id:
            break

        logits, past_key_values = model_wrapper.forward(next_token_id, past_key_values)

    return tokenizer_wrapper.decode(generated_ids[0].tolist())
```

这段代码已经体现了 prefill 和 decode 的区别：第一次 forward 输入完整 prompt，后续 forward 只输入新 token。

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

不是。KV Cache 通常用显存换速度。

误区三：decode 阶段还要传完整历史 input_ids。

使用 cache 时，decode 通常只传最新 token，同时传入 past_key_values。

误区四：单请求 cache 管理就等于生产 cache manager。

生产系统要处理多请求、碎片、释放、抢占、复用和显存预算。

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
4. 最后补充生产难点：多请求分配释放、碎片、prefix 复用、position 和 mask 正确性。

## 9.16 小练习

1. 在第 7 章代码基础上加入 `past_key_values`。
2. 打印每层 cache 的 shape，观察 seq_len 如何增长。
3. 对比使用 cache 和不使用 cache 的生成耗时。
4. 故意在 decode 阶段传完整 `generated_ids`，观察性能和行为变化。
5. 思考如果 batch 中每个请求长度不同，position_ids 应该如何设置。

## 9.17 本章小结

本章实现了最小 KV Cache。

naive generate loop 每轮重复计算完整历史序列，KV Cache 通过保存历史 key/value，让 decode 阶段只计算新 token。最小实现依赖 `past_key_values`：prefill 输入完整 prompt 并建立 cache，decode 输入最新 token 并更新 cache。

下一章我们会继续升级，把单请求 prefill 扩展成 batched prefill，让多个请求的 prompt 能一起进入模型执行。
