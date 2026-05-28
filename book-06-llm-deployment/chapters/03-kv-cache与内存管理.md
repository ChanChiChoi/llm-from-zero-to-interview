# 第三章：KV Cache 与内存管理

## 本章目标

理解 KV Cache 对推理速度和显存占用的影响。

## 核心议题

1. KV Cache 缓存了什么。
2. KV Cache 显存估算。
3. MHA、MQA、GQA 对 KV Cache 的影响。
4. PagedAttention。
5. KV Cache 量化。
6. 长上下文下的内存压力。

## 面试重点

KV Cache 是部署面试高频题，要能讲清楚它为什么加速、为什么占显存、如何优化。

## 为什么 KV Cache 是推理优化核心

LLM 自回归生成时，每一步只生成一个新 token。

如果没有 KV Cache，模型每生成一个 token，都要重新计算所有历史 token 的 key 和 value。

这会造成大量重复计算。

KV Cache 的核心思想是：

```text
历史 token 的 K/V 已经算过，就缓存起来，后续 decode 直接复用。
```

它能显著降低 decode 阶段的重复计算，但代价是占用大量显存。

面试表达：KV Cache 用显存换时间，是 LLM decode 加速的核心机制。

## 1. KV Cache 缓存了什么

在 self-attention 中，每层都会计算：

```text
Q = X W_Q
K = X W_K
V = X W_V
```

生成过程中，新 token 的 query 要和所有历史 token 的 key/value 交互。

历史 token 的 K/V 在后续步骤不会变，所以可以缓存。

常见 cache 结构：

```text
past_key_values[layer_id] = (k_cache, v_cache)
```

常见 shape：

```text
k_cache: [batch, num_kv_heads, seq_len, head_dim]
v_cache: [batch, num_kv_heads, seq_len, head_dim]
```

注意这里是 `num_kv_heads`，因为 MQA/GQA 中 K/V head 数可能少于 query head 数。

## 2. Prefill 和 Decode 中的 KV Cache

### 2.1 Prefill

Prefill 阶段处理完整 prompt。

它会一次性计算 prompt 中所有 token 的 K/V，并写入 KV Cache。

例如 prompt 长度是 1024，则 prefill 后每层 cache 长度就是 1024。

### 2.2 Decode

Decode 阶段每次生成一个新 token。

每一步只计算新 token 的 K/V，再 append 到 cache：

```text
k_cache = concat(k_cache, k_new)
v_cache = concat(v_cache, v_new)
```

然后新 token 的 query attend 到完整 cache。

面试表达：prefill 负责建立初始 KV Cache，decode 负责逐步追加并复用 KV Cache。

## 3. KV Cache 为什么加速

假设已经生成到长度 `T`。

没有 KV Cache，每一步都要对长度 `T` 的完整序列重新算 K/V。

有 KV Cache，每一步只需要计算新 token 的 K/V。

这避免了历史 token 的重复投影计算。

但注意：KV Cache 不代表 attention 成本完全消失。

新 token 仍然要 attend 到所有历史 K/V，所以随着上下文变长，decode 每步仍会变慢。

面试表达：KV Cache 省掉的是历史 token K/V 的重复计算，但不会消除新 token 对历史上下文的 attention 访问。

## 4. KV Cache 显存估算

KV Cache 显存大致和这些因素成正比：

```text
batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2 * bytes
```

其中 `2` 表示 K 和 V 两份缓存。

### 4.1 解释每一项

1. `batch_size`：并发请求数。
2. `seq_len`：prompt + 已生成 token 的总长度。
3. `num_layers`：每层都有 KV Cache。
4. `num_kv_heads`：K/V head 数。
5. `head_dim`：每个 head 的维度。
6. `bytes`：数据类型大小，例如 FP16/BF16 是 2 bytes。

### 4.2 为什么长上下文贵

`seq_len` 越长，KV Cache 线性增长。

如果并发也高，显存压力会非常大。

这就是为什么长上下文在线服务经常先被 KV Cache 显存限制，而不是被模型权重限制。

## 5. MHA、MQA、GQA 对 KV Cache 的影响

### 5.1 MHA

标准 Multi-Head Attention 中，每个 query head 都有独立 K/V。

如果 query head 数是 `H`，那么 K/V head 数也是 `H`。

优点是表达能力强，缺点是 KV Cache 大。

### 5.2 MQA

Multi-Query Attention 中，所有 query head 共享一组 K/V。

KV Cache 显著变小。

缺点是表达能力可能受影响。

### 5.3 GQA

Grouped-Query Attention 是折中方案。

多个 query head 共享一组 K/V。

它在效果和 KV Cache 成本之间取得平衡，因此现代 LLM 很常用。

面试表达：MHA、MQA、GQA 的重要部署差异之一，就是 K/V head 数不同，从而直接影响 KV Cache 显存。

## 6. KV Cache 和并发

在线服务中，每个请求都有自己的 KV Cache。

并发越高，cache 越多。

如果请求长度差异很大，会出现两个问题：

1. 短请求很快结束，长请求一直占显存。
2. 如果连续分配显存，容易产生碎片和浪费。

这也是为什么 LLM serving 需要专门的 KV Cache 管理系统。

## 7. PagedAttention

PagedAttention 的核心是用分页思想管理 KV Cache。

传统方式可能要求每个请求的 cache 连续存放。

但请求长度动态变化时，连续分配容易浪费。

PagedAttention 把 KV Cache 切成固定大小 block：

```text
request A -> block 1, block 7, block 9
request B -> block 2, block 3
```

逻辑上连续，物理上可以不连续。

### 7.1 好处

1. 减少显存碎片。
2. 按需分配 cache block。
3. 请求结束后及时回收 block。
4. 更适合 continuous batching。

### 7.2 面试表达

PagedAttention 不是新的 attention 数学公式，而是 KV Cache 的分页式内存管理方法。

## 8. KV Cache 量化

KV Cache 也可以量化。

例如从 FP16/BF16 降到 INT8 或更低精度。

### 8.1 好处

1. 降低显存占用。
2. 提升可支持并发。
3. 降低显存带宽压力。

### 8.2 风险

1. 可能影响生成质量。
2. 长上下文下误差可能累积。
3. 对注意力分布敏感。

所以 KV Cache 量化必须做质量评估，不能只看显存收益。

## 9. Prefix Cache 和 Prompt Cache

很多线上请求共享相同前缀。

例如：

1. 固定 system prompt。
2. 企业助手固定规则。
3. RAG 模板前缀。
4. 多轮对话历史的一部分。

Prefix cache 可以复用公共前缀的 KV Cache，减少重复 prefill。

但复用的条件通常是 token 序列完全一致。

如果 tokenizer、chat template 或 system prompt 版本变化，cache 就可能失效。

还要注意权限隔离，避免不同用户之间缓存泄漏。

## 10. 长上下文下的内存压力

长上下文会让 KV Cache 成为主要显存瓶颈。

例如上下文从 4k 增加到 32k，KV Cache 大约线性增长 8 倍。

常见缓解方法：

1. 限制最大上下文长度。
2. 限制最大输出长度。
3. 使用 GQA/MQA。
4. KV Cache 量化。
5. PagedAttention。
6. Prefix cache。
7. 对超长请求单独调度。

面试表达：长上下文服务的难点不只是位置编码，而是 KV Cache 显存、decode 带宽和调度成本。

## 11. KV Cache OOM 怎么排查

如果线上出现显存 OOM，可以按下面顺序排查：

```text
1. 当前并发数是多少？
2. 平均 prompt 长度和 P95 prompt 长度是多少？
3. 平均输出长度和 P95 输出长度是多少？
4. 是否有超长请求占用大量 cache？
5. 是否及时释放已完成请求的 cache？
6. 是否存在 cache 碎片？
7. 是否开启了分页或量化？
8. 模型是 MHA、MQA 还是 GQA？
```

不要只看模型权重大小。很多时候权重放得下，但 KV Cache 放不下。

## 12. 面试官会怎么问

### 问法 1：KV Cache 为什么能加速？

可以这样答：

```text
自回归生成每一步都会用到历史 token。如果没有 KV Cache，每一步都要重新计算历史 token 的 key 和 value。KV Cache 把历史 K/V 存起来，decode 时只计算新 token 的 K/V，再让新 query attend 到历史 cache，从而避免大量重复计算。
```

### 问法 2：KV Cache 为什么占显存？

可以这样答：

```text
因为每一层都要为每个请求保存历史 token 的 K 和 V。显存大致和 batch size、sequence length、layer 数、KV head 数、head dim 和 dtype 成正比。长上下文和高并发下，KV Cache 可能成为主要显存瓶颈。
```

### 问法 3：MHA、MQA、GQA 对 KV Cache 有什么影响？

可以这样答：

```text
MHA 中每个 query head 都有独立 K/V，KV Cache 最大。MQA 中多个 query head 共享一组 K/V，cache 最小。GQA 是折中，若干 query head 共享一组 K/V，在效果和 cache 成本之间平衡。部署时 GQA/MQA 可以明显降低 KV Cache 显存和带宽压力。
```

### 问法 4：PagedAttention 解决什么问题？

可以这样答：

```text
PagedAttention 用分页思想管理 KV Cache，把 cache 切成固定大小 block，逻辑上连续但物理上可以分散。这样可以减少连续分配带来的显存碎片，按需分配和回收 cache，更适合高并发和 continuous batching。
```

### 问法 5：长上下文服务为什么难？

可以这样答：

```text
长上下文会让 prefill 变慢，也会让 KV Cache 线性增长。decode 阶段每步都要访问更长的历史 K/V，所以显存和带宽压力都上升。部署上要结合 GQA/MQA、PagedAttention、KV Cache 量化、prefix cache 和请求调度来控制成本。
```

## 13. 本章小结

本章核心结论：

1. KV Cache 缓存每层历史 token 的 K/V。
2. KV Cache 避免重复计算历史 K/V，从而加速 decode。
3. KV Cache 不消除 attention 对历史上下文的访问成本。
4. KV Cache 显存随 batch、seq_len、layer、kv_heads、head_dim 和 dtype 增长。
5. MQA/GQA 可以显著降低 KV Cache 成本。
6. PagedAttention 通过分页式 cache 管理减少显存碎片。
7. KV Cache 量化能省显存，但必须评估质量影响。
8. Prefix cache 可以复用公共前缀，降低重复 prefill 成本。
9. 长上下文和高并发下，KV Cache 往往是部署瓶颈。
