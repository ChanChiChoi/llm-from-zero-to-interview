# 第十一章：KV Cache、长上下文推理和显存增长问题

## 11.0 本讲资料边界与第二轮精修口径

截至 2026-06-10，本讲用公开论文、官方文档和主流框架资料校准 KV cache 的通用概念。这里讨论 decoder-only LLM 推理中的 K/V 缓存、显存估算、decode 带宽、PagedAttention 和 MQA / GQA / MLA 的结构差异，不把某个 serving 框架的 block size、scheduler 策略、cache layout、量化配置或某个模型报告中的压缩比例写成通用标准。

本讲重点区分五层问题：

1. 推理语义：KV cache 是推理时的运行时状态，不是模型权重，也通常不用于训练反向传播。
2. 张量规模：每层历史 token 的 K 和 V 都要缓存，大小随层数、batch、上下文长度、KV head 数、head dim 和精度线性增长。
3. Decode 带宽：decode 单步计算量不再是完整 `T^2`，但每步要读取历史 K/V，长上下文和高并发下常受显存容量与带宽限制。
4. 架构压缩：MQA / GQA 减少 KV head 数，MLA 压缩 KV 表示，属于模型结构层面的 KV cache 优化。
5. 系统管理：PagedAttention、prefix cache、continuous batching 和 block manager 管理动态请求和 KV blocks，不改变 scaled dot-product attention 的数学定义。

后面的公式默认使用：

1. `L` 表示 Transformer 层数。
2. `B` 表示活跃请求 batch size。
3. `T` 表示每个请求当前上下文长度。
4. `H_q` 表示 query head 数，`H_{\mathrm{kv}}` 表示 K/V head 数。
5. `D_h` 表示每个 head 的维度。
6. `s` 表示每个 cache 元素的字节数，例如 fp16/bf16 通常粗略取 `s=2`。

## 11.1 本章定位

上一章讲了训练和 prefill 阶段中 O(n^2) attention 的计算与显存瓶颈。本章进入自回归推理阶段的核心问题：KV Cache。

对 decoder-only LLM 来说，推理不是一次性输出完整答案，而是逐 token 生成：

```text
prompt -> 生成第 1 个 token -> 追加到上下文 -> 生成第 2 个 token -> ...
```

如果每生成一个 token 都重新计算所有历史 token 的 K/V，成本会非常高。KV cache 的作用就是缓存历史 token 在每一层的 key 和 value，后续生成时直接复用。

KV cache 让自回归推理变快，但也带来了新的问题：

```text
上下文越长、batch 越大、层数越多、KV heads 越多，显存占用越大。
```

本章要回答的问题是：

1. KV cache 缓存的到底是什么。
2. 没有 KV cache 时推理为什么会重复计算。
3. KV cache 显存如何估算。
4. 为什么长上下文 serving 经常被 KV cache 卡住。
5. prefill、decode、continuous batching 的瓶颈有什么区别。
6. MHA、MQA、GQA、MLA 如何影响 KV cache。
7. PagedAttention/vLLM 解决了什么问题。
8. 长上下文推理中延迟、吞吐、显存和质量如何 trade-off。

本章的核心观点是：

```text
KV cache 是 decoder-only LLM 高效推理的必要条件，也是长上下文、高并发 serving 的核心显存和带宽瓶颈。
```

## 11.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 Transformer attention 结构，decoder 生成时需要读取历史上下文。
2. Shazeer, 2019, *Fast Transformer Decoding: One Write-Head is All You Need*。提出 Multi-Query Attention，指出 incremental decoding 中反复加载 K/V 的内存带宽成本很高。
3. Ainslie et al., 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。提出 GQA，在质量和推理速度之间折中。
4. Kwon et al., 2023, *Efficient Memory Management for Large Language Model Serving with PagedAttention*。提出 PagedAttention/vLLM，用分页式管理 KV cache，降低碎片和重复。
5. Dao et al., 2022, *FlashAttention*。作为对比：FlashAttention 优化 attention kernel 和 IO，不从架构上减少 KV cache 总量。
6. DeepSeek-AI, 2024, *DeepSeek-V2*。提出 MLA，压缩 KV cache 到 latent vector，并报告大幅减少 KV cache 和提升生成吞吐。
7. Hugging Face Transformers cache 官方文档。用于校准 `past_key_values`、cache position、attention mask 和生成时 cache 更新的实现口径。
8. vLLM 官方文档与设计说明。用于校准 KV cache block、block table、PagedAttention 和 continuous batching 的系统边界。

需要说明的是，KV cache 的具体实现和显存占用会受框架、并行策略、量化、分页粒度、batch scheduler、模型结构影响。本章重点讲通用原理和估算方法。

## 11.3 自回归推理为什么需要 KV Cache

decoder-only LLM 每一步都要预测下一个 token。

假设 prompt 是：

```text
x_1, x_2, ..., x_n
```

生成第一个新 token 时，需要对这 `n` 个 token 做前向计算。

生成第二个新 token 时，上下文变成：

```text
x_1, x_2, ..., x_n, y_1
```

如果没有 cache，模型会重新计算所有 `x_1...x_n` 的 K/V。生成第 `t` 个 token 时，又会重复计算前面所有历史 token。

这很浪费，因为历史 token 的 hidden state 在 causal attention 中不会因为未来 token 出现而改变。

对于已经处理过的 token，它们在每层产生的 K/V 可以缓存下来：

```text
历史 token 的 K/V 不变，新 token 只需要计算自己的 Q/K/V。
```

这就是 KV cache。

## 11.4 KV Cache 缓存什么

在每一层 attention 中，模型会计算：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

对新 token 来说，当前层的 query `q_t` 要和所有历史 key 做点积：

```math
s_{t,j}=\frac{q_t^\top k_j}{\sqrt{D_h}}
```

然后用 attention weights 加权所有历史 value：

```math
o_t=\sum_{j=1}^{t} a_{t,j}v_j
```

因此推理时需要保存每一层、每个历史 token 的：

```text
K cache
V cache
```

通常 shape 可以理解为：

```math
K_{\mathrm{cache}},V_{\mathrm{cache}}\in\mathbb{R}^{B\times H_{\mathrm{kv}}\times T\times D_h}
```

实际系统可能为了 kernel 访问效率使用不同布局，例如按 block、page、layer 或 head 组织。

## 11.5 有无 KV Cache 的复杂度对比

假设 prompt 长度为 `n`，要生成 `m` 个 token。

### 没有 KV Cache

每一步都重新计算完整上下文。

第 1 步上下文长度约 `n`。
第 2 步上下文长度约 `n+1`。
第 m 步上下文长度约 `n+m-1`。

每一步都要重新跑历史 token 的 K/V 和 attention。

这会造成大量重复计算。

### 有 KV Cache

prefill 阶段：

```text
一次性处理 prompt，缓存 prompt 的 K/V。
```

decode 阶段：

```text
每步只计算新 token 的 Q/K/V。
新 token 的 Q 读取历史 K/V cache。
把新 token 的 K/V 追加进 cache。
```

这样每步不再重复计算历史 token 的 K/V。

KV cache 的收益非常大，是现代 decoder-only LLM serving 的基础。

## 11.6 Prefill 和 Decode

LLM 推理一般分为两个阶段。

### Prefill

prefill 处理输入 prompt。

如果 prompt 长度是 `n`：

```math
T_q=T_k=n
```

它需要对 prompt 内部做 causal attention，成本近似 O(n^2)。

prefill 的输出包括：

1. 最后一个位置的 logits，用来生成第一个 token。
2. prompt 中所有 token 的 KV cache。

prefill 影响首 token 延迟，也就是 TTFT：

```text
time to first token
```

### Decode

decode 每次生成一个新 token。

此时：

```math
T_q=1,\qquad T_k=T_{\mathrm{past}}+1
```

单步 attention 成本约 O(past_len)，但需要读取历史 K/V cache。

decode 影响输出速度，也就是：

```text
tokens per second
```

长上下文推理中，prefill 和 decode 都可能成为瓶颈，只是瓶颈类型不同。

## 11.7 KV Cache 显存公式

KV cache 粗略大小：

```math
M_{\mathrm{kv}}=L B T H_{\mathrm{kv}} D_h \times 2 \times s
```

其中：

1. `num_layers` 是层数。
2. `batch_size` 是同时服务的请求数。
3. `seq_len` 是每个请求当前上下文长度。
4. `num_kv_heads` 是 K/V head 数。
5. `head_dim` 是每个 head 的维度。
6. `2` 是 K 和 V 两份。
7. `bytes_per_element` 对 fp16/bf16 通常是 2。

注意，这里用的是 `num_kv_heads`，不是一定等于 `num_q_heads`。MHA、MQA、GQA 的差异主要就在这里。

如果只看单个 token 给一个请求追加到所有层的 KV cache，粗略是：

```math
M_{\mathrm{kv/token}}=L H_{\mathrm{kv}}D_h\times 2\times s
```

这也是长输出 decode 时 cache 持续增长的基本单位。

## 11.8 KV Cache 粗略估算

假设一个模型：

```text
num_layers = 32
batch_size = 16
seq_len = 8192
num_kv_heads = 32
head_dim = 128
dtype = bf16 = 2 bytes
```

KV cache 大小：

```math
M_{\mathrm{kv}}=32\times 16\times 8192\times 32\times 128\times 2\times 2\ \mathrm{bytes}
```

```math
M_{\mathrm{kv}}\approx 68.7\ \mathrm{GB}\approx 64\ \mathrm{GiB}
```

这只是 KV cache，不包括：

1. 模型权重。
2. temporary buffers。
3. logits。
4. runtime workspace。
5. memory fragmentation。
6. 框架和通信开销。

如果上下文扩到 32K，其他不变：

```text
seq_len 变成 4 倍
KV cache 也变成 4 倍
```

这就是为什么长上下文 + 高并发 serving 会非常吃显存。

## 11.9 KV Cache 是线性增长，但仍然可怕

和训练 attention matrix 的 O(n^2) 不同，KV cache 对序列长度是线性增长：

```math
M_{\mathrm{kv}}\propto T
```

但它仍然可怕，因为常数很大：

```math
L B H_{\mathrm{kv}}D_h\times 2
```

而且 serving 时需要同时服务多个请求。

单个请求 128K 上下文可能还能勉强放下，但如果要并发服务几十个请求，KV cache 会迅速吃光显存。

所以长上下文产品化的难点不是“单条样本能不能跑”，而是：

```text
在可接受延迟和成本下，能不能高吞吐稳定服务大量长请求。
```

## 11.10 Decode 为什么常受内存带宽限制

decode 阶段每步只处理一个新 token，矩阵乘法规模较小，不一定能充分利用 GPU 的峰值算力。

同时，每一步都要读取历史 K/V：

```text
new query -> read all historical K/V -> compute attention -> produce next token
```

当上下文很长、batch 很大时，读取 KV cache 的内存带宽会成为瓶颈。

这也是 Shazeer 提出 MQA 的关键动机：

```text
incremental decoding 慢，很多时候不是因为 FLOPs 不够，而是因为反复加载巨大的 K/V tensors。
```

所以优化 decode，不只是优化计算，也要减少：

1. KV cache 大小。
2. KV cache 读取带宽。
3. cache 碎片。
4. 重复 cache。
5. batch 内 padding 和无效计算。

## 11.11 MHA、MQA、GQA 对 KV Cache 的影响

标准 MHA：

```math
H_q=H_{\mathrm{kv}}=H
```

每个 query head 有独立 K/V head，表达能力完整，但 KV cache 最大。

MQA：

```math
H_q=H,\qquad H_{\mathrm{kv}}=1
```

所有 query heads 共享一组 K/V，KV cache 约降到 MHA 的 `1/H`。

GQA：

```math
H_q=H,\qquad H_{\mathrm{kv}}=G,\qquad 1<G<H
```

每组 query heads 共享一个 K/V head，是 MHA 和 MQA 之间的折中。

例如：

```text
H = 32
GQA num_kv_heads = 8
```

KV cache 约为 MHA 的：

```math
\frac{H_{\mathrm{kv}}}{H_q}=\frac{8}{32}=\frac{1}{4}
```

这就是为什么现代 LLM 很常用 GQA。它在质量和推理成本之间更平衡。

## 11.12 MLA：从减少 KV Heads 到压缩 KV 表示

MLA，也就是 Multi-head Latent Attention，是更激进的 KV cache 优化路线。

GQA/MQA 的思路是减少 K/V head 数。

MLA 的思路更像是：

```text
不要直接缓存完整 K/V，而是缓存更低维的 latent 表示，需要时再恢复或投影出 attention 所需信息。
```

DeepSeek-V2 报告中，MLA 显著压缩 KV cache，并提升最大生成吞吐。

从架构演进角度看：

```text
MHA：每个 head 都缓存 K/V。
MQA/GQA：减少 K/V head 数。
MLA：压缩 K/V 表示本身。
```

MLA 的代价是实现复杂度更高，对训练、推理 kernel、权重格式、生态兼容性都有更高要求。

## 11.13 PagedAttention 解决什么

KV cache 不只是大，还会动态变化。

Serving 中每个请求长度不同：

```text
请求 A：prompt 200 tokens，生成 50 tokens
请求 B：prompt 8000 tokens，生成 500 tokens
请求 C：prompt 1000 tokens，生成 20 tokens
```

请求会不断加入、结束、扩展。KV cache 会动态增长和释放。

如果用连续大块内存为每个请求分配 KV cache，容易出现：

1. 内存碎片。
2. 预留过多。
3. batch size 受限。
4. prefix 共享困难。
5. beam search 或 parallel sampling 中重复 cache。

PagedAttention 借鉴操作系统分页思想，把 KV cache 划分为固定大小的 blocks/pages。

逻辑上请求有连续 token 序列，物理上 KV cache 可以分散存储在不同 blocks 中，通过 block table 映射。

这样可以：

1. 减少碎片。
2. 接近按需分配。
3. 更灵活地释放和复用 blocks。
4. 支持 prefix cache sharing。
5. 提升可容纳 batch size。

vLLM 就是围绕 PagedAttention 构建的高吞吐 LLM serving 系统。

## 11.14 Continuous Batching

传统 batching 可能要求一批请求一起开始、一起结束。

但 LLM 生成长度不同，如果等待最长请求完成，会浪费大量算力。

Continuous batching 的思想是：

```text
每个 decode step 动态调度请求。
完成的请求移出 batch。
新的请求加入 batch。
```

这能提高 GPU 利用率和吞吐。

但它也要求系统高效管理 KV cache：

1. 请求动态加入。
2. 请求动态结束。
3. 每个请求 past length 不同。
4. KV cache blocks 频繁分配和释放。
5. attention kernel 要支持 varlen batch。

所以 continuous batching 和 PagedAttention 往往是配套问题。

## 11.15 Prefix Sharing

很多请求共享相同前缀。

例如：

1. 同一个 system prompt。
2. 同一个长文档问多个问题。
3. beam search 中多个候选共享前缀。
4. parallel sampling 中多个采样分支共享 prompt。
5. Agent 多次调用模型时共享历史上下文。

如果每个请求都复制一份 prefix KV cache，会浪费显存。

Prefix sharing 的目标是：

```text
共享相同 prefix 的 KV cache，只为分叉后的 token 单独存储。
```

PagedAttention 的 block/page 设计有利于实现这种共享。

但 prefix sharing 也带来工程复杂度：

1. 引用计数。
2. block 生命周期。
3. cache 命中判断。
4. 不同请求 position ids 对齐。
5. 安全隔离和多租户问题。

## 11.16 Sliding Window Cache

如果模型使用 sliding window attention，每个新 token 只关注最近 `w` 个历史 token。

那么理论上不需要保留全部历史 K/V 给每层 attention 使用。

可以只保留窗口内 cache：

```text
cache length ≈ window size
```

这能显著降低长上下文 decode 的 cache 显存。

但代价是：

1. 模型无法直接访问窗口外历史。
2. 长距离信息需要通过其他机制保留。
3. 需要和位置编码、mask、cache eviction 配套。
4. 对任务质量有影响。

有些模型会结合 local window、global tokens、summary memory 或 retrieval 来弥补窗口限制。

## 11.17 Cache Quantization

KV cache 可以量化。

如果 fp16/bf16 cache 每元素 2 bytes，量化到 int8 可以近似减半，量化到更低 bit 可以进一步减少。

好处：

1. 降显存。
2. 降内存带宽。
3. 提高可并发 batch size。

风险：

1. attention score 或 value 聚合误差。
2. 长上下文中误差累积。
3. 不同层、不同 head 对量化敏感性不同。
4. kernel 实现复杂。
5. 质量和延迟收益需要实测。

KV cache quantization 是实用工程优化，但不能把它当成无损免费的优化。

## 11.18 长上下文推理的延迟拆解

一个请求的总延迟可以粗略拆成：

```math
T_{\mathrm{e2e}}=T_{\mathrm{prefill}}+T_{\mathrm{decode}}+T_{\mathrm{queue}}+T_{\mathrm{post}}
```

prefill latency 主要受：

1. prompt length。
2. O(n^2) attention。
3. model size。
4. batching。
5. attention kernel。

decode latency 主要受：

1. output length。
2. 每步读取 KV cache 的成本。
3. batch size。
4. sampling/beam search。
5. memory bandwidth。
6. KV cache 管理。

长 prompt + 长输出是最难的组合：

```text
prefill 很重，decode cache 也越来越大。
```

## 11.19 吞吐、延迟、显存的三角关系

Serving 系统经常要在三者之间折中：

```text
吞吐：单位时间生成多少 tokens。
延迟：单个请求等多久。
显存：能放多少权重、cache、batch。
```

提高 batch size 通常能提升吞吐，但会增加每个请求等待和 KV cache 显存。

支持更长上下文能提升产品能力，但会降低并发和吞吐。

压缩 KV cache 能提高并发，但可能引入质量或实现复杂度风险。

所以真实工程里通常要按场景优化：

1. 聊天助手：更重视低延迟和稳定吞吐。
2. 长文档分析：更重视长 prompt prefill 能力。
3. 批量离线生成：更重视吞吐和成本。
4. 代码仓库分析：长上下文和检索结合更重要。
5. Agent 场景：多轮上下文、prefix sharing、工具调用格式很重要。

## 11.20 面向专家：KV Cache 和 Tensor Parallel

大模型常用 tensor parallel 把 attention heads 分到多张 GPU 上。

如果按 head 切分，每张卡只保存自己负责 heads 的 KV cache。

这能分摊 cache，但也带来通信和调度问题。

例如：

1. logits 需要跨卡聚合。
2. attention output 可能需要 all-reduce。
3. cache layout 要和 tensor parallel partition 对齐。
4. GQA/MQA 的 KV heads 较少时，head 切分可能不均衡。
5. pipeline parallel 下不同层的 cache 分布在不同 stage。

所以 KV cache 优化不能只看单卡公式，还要考虑分布式并行布局。

## 11.21 面向专家：为什么 KV Cache 影响架构设计

过去很多架构设计更关注训练效果。

现代 LLM 越来越多地把 serving 成本前置考虑。

原因是：

```text
模型训练一次，推理服务可能运行数十亿次。
```

如果一个结构训练时效果稍好，但 KV cache 大很多，线上成本可能不可接受。

这就是为什么 MQA、GQA、MLA 变得重要。

它们都在回答同一个问题：

```text
能否减少每个历史 token 必须长期保存和反复读取的信息量？
```

这也是后续架构演进的重要方向：

1. 更少 KV heads。
2. 更低维 KV cache。
3. KV cache quantization。
4. sliding window cache。
5. retrieval/memory 替代一部分上下文。
6. state-space/recurrent 架构减少显式缓存。

## 11.22 KV Cache 成本审计指标与最小 demo

第二轮精修时，本章建议把 KV cache 问题落到一组可审计指标：

1. `kv_cache_gib`：不同 KV head 数、上下文长度和 batch 下的 KV cache 显存。
2. `kv_cache_per_token`：每个新 token 会给每个请求追加多少 cache。
3. `decode_read_per_step`：decode 单步需要读取的历史 K/V 量级，用来估算带宽压力。
4. `gqa_vs_mha_saving`：GQA / MQA 相比 MHA 的 cache 节省比例。
5. `paged_block_waste`：按 fixed-size blocks 管理 KV cache 时尾部浪费多少 token。
6. `kv_cache_gate`：把显存容量、decode 带宽、分页浪费和上下文预算组合成上线门禁。

下面是一个 0 依赖 demo。它不模拟真实 kernel，也不代表某个 serving 框架的默认配置，只演示 KV cache 的线性增长、KV head 压缩收益和分页块尾部浪费。

```python
BYTES_PER_ELEM = 2


def gib(num_bytes):
    return num_bytes / (1024 ** 3)


def kv_cache_bytes(layers, batch, seq_len, kv_heads, head_dim, bytes_per_elem):
    return layers * batch * seq_len * kv_heads * head_dim * 2 * bytes_per_elem


def block_stats(prompt_lens, block_size):
    total_blocks = sum((length + block_size - 1) // block_size for length in prompt_lens)
    total_capacity = total_blocks * block_size
    used_tokens = sum(prompt_lens)
    waste_tokens = total_capacity - used_tokens
    return total_blocks, waste_tokens, waste_tokens / total_capacity if total_capacity else 0.0


layers = 32
batch = 16
seq_len = 8192
head_dim = 128
q_heads = 32
mha_kv_heads = 32
gqa_kv_heads = 8
mqa_kv_heads = 1

mha_gib = gib(kv_cache_bytes(layers, batch, seq_len, mha_kv_heads, head_dim, BYTES_PER_ELEM))
gqa_gib = gib(kv_cache_bytes(layers, batch, seq_len, gqa_kv_heads, head_dim, BYTES_PER_ELEM))
mqa_gib = gib(kv_cache_bytes(layers, batch, seq_len, mqa_kv_heads, head_dim, BYTES_PER_ELEM))
long_gqa_gib = gib(kv_cache_bytes(layers, batch, seq_len * 4, gqa_kv_heads, head_dim, BYTES_PER_ELEM))
per_token_gqa_kib = kv_cache_bytes(layers, 1, 1, gqa_kv_heads, head_dim, BYTES_PER_ELEM) / 1024

decode_read_gib_per_step = gib(kv_cache_bytes(layers, batch, seq_len, gqa_kv_heads, head_dim, BYTES_PER_ELEM))

prompt_lens = [210, 8192, 997, 64, 4097]
blocks, waste, waste_ratio = block_stats(prompt_lens, block_size=16)

failed_gates = []
if gqa_gib > 20:
    failed_gates.append("gqa_cache_over_20gib")
if long_gqa_gib > 80:
    failed_gates.append("long_context_cache_over_80gib")
if waste_ratio > 0.08:
    failed_gates.append("paged_block_waste_over_8pct")
if decode_read_gib_per_step > 25:
    failed_gates.append("decode_bandwidth_pressure")

print("mha_8k_cache_gib=", round(mha_gib, 2))
print("gqa_8k_cache_gib=", round(gqa_gib, 2))
print("mqa_8k_cache_gib=", round(mqa_gib, 2))
print("gqa_vs_mha_saving=", round(1 - gqa_gib / mha_gib, 3))
print("gqa_32k_cache_gib=", round(long_gqa_gib, 2))
print("gqa_cache_per_token_kib=", round(per_token_gqa_kib, 1))
print("decode_read_gib_per_step=", round(decode_read_gib_per_step, 2))
print("paged_blocks=", blocks)
print("paged_waste_tokens=", waste)
print("paged_waste_ratio=", round(waste_ratio, 4))
print("failed_gates=", failed_gates)
print("kv_cache_gate_pass=", len(failed_gates) == 0)
```

示例输出：

```text
mha_8k_cache_gib= 64.0
gqa_8k_cache_gib= 16.0
mqa_8k_cache_gib= 2.0
gqa_vs_mha_saving= 0.75
gqa_32k_cache_gib= 64.0
gqa_cache_per_token_kib= 128.0
decode_read_gib_per_step= 16.0
paged_blocks= 850
paged_waste_tokens= 40
paged_waste_ratio= 0.0029
failed_gates= []
kv_cache_gate_pass= True
```

这个 demo 的读法是：

1. 同样 8K 上下文、16 并发、32 层、`D_h=128`，MHA cache 约 64 GiB，GQA 8 个 KV heads 约 16 GiB，MQA 约 2 GiB。
2. GQA 从 32 个 KV heads 降到 8 个 KV heads，cache 节省约 75%。
3. GQA 下上下文从 8K 扩到 32K，cache 仍然线性变成 4 倍，约 64 GiB。
4. 每个新 token 给单请求追加约 128 KiB cache；长输出和多并发会持续吃掉显存。
5. block/page 管理可以把不同长度请求映射到固定块，尾部浪费由 block size 和请求长度分布共同决定。

## 11.23 常见误区

### 误区 1：KV cache 会减少模型权重显存

不会。KV cache 是额外的推理状态，不是模型权重压缩。

### 误区 2：KV cache 越大，模型效果一定越好

不一定。更长上下文提供更多可见信息，但模型是否能有效利用，还受位置编码、训练长度、attention pattern、干扰信息和任务影响。

### 误区 3：FlashAttention 能解决 KV cache 过大

不准确。FlashAttention 优化 attention kernel 的 IO 和中间显存，不从源头减少历史 K/V cache 总量。

### 误区 4：PagedAttention 是 GQA 的替代品

不是。PagedAttention 管理 KV cache 内存，GQA 减少 KV cache 大小。两者可以组合使用。

### 误区 5：长上下文只影响 prefill

不对。长 prompt 影响 prefill；长历史和长输出还会增加 decode 阶段的 KV cache 读取和显存占用。

### 误区 6：MQA/GQA 只是减少参数

主要不是。它们的关键收益是减少 KV cache 和 decode 内存带宽。

## 11.24 面试高频问题

### 题 1：KV cache 是什么？为什么需要它？

参考回答：

```text
KV cache 是 decoder-only LLM 推理时缓存每层历史 token 的 key 和 value。自回归生成每步只新增一个 token，历史 token 的 K/V 不会因为未来 token 改变。如果不缓存，每一步都要重复计算所有历史 token 的 K/V，成本很高。KV cache 让 decode 只计算新 token 的 Q/K/V，并复用历史 K/V。
```

### 题 2：KV cache 显存如何估算？

参考回答：

```text
KV cache 大小约等于 num_layers * batch_size * seq_len * num_kv_heads * head_dim * 2 * bytes_per_element。其中 2 表示 K 和 V 两份。它随层数、batch、上下文长度和 KV head 数线性增长。长上下文和高并发 serving 时，KV cache 往往会成为显存瓶颈。
```

### 题 3：Prefill 和 decode 的区别是什么？

参考回答：

```text
Prefill 是处理输入 prompt 的阶段，query_len 和 key_len 都是 prompt 长度，需要构建 prompt 的 KV cache，长 prompt 下 attention 成本接近 O(n^2)，影响首 token 延迟。Decode 是逐 token 生成阶段，每步 query_len 为 1，key_len 是历史长度，单步 attention 是 O(t)，主要受 KV cache 读取、显存和内存带宽影响。
```

### 题 4：为什么长上下文推理会被 KV cache 卡住？

参考回答：

```text
KV cache 随上下文长度、batch size、层数、KV head 数线性增长。长上下文下每个请求都需要保存大量历史 K/V，高并发时显存很快被占满。同时 decode 每步都要读取历史 K/V，内存带宽也会成为瓶颈。因此长上下文 serving 不只是计算问题，也是显存容量和带宽问题。
```

### 题 5：MQA/GQA 为什么能提升推理速度？

参考回答：

```text
MQA/GQA 减少 K/V head 数。标准 MHA 每个 query head 都有独立 K/V head，KV cache 大且 decode 时读取带宽高。MQA 让所有 query heads 共享一组 K/V，GQA 让一组 query heads 共享 K/V，从而显著减少 KV cache 大小和读取带宽，提升长上下文和大 batch 推理效率。
```

### 题 6：PagedAttention 解决什么问题？

参考回答：

```text
PagedAttention 解决 LLM serving 中 KV cache 动态内存管理问题。不同请求长度不同、不断加入和结束，连续分配容易造成碎片和预留浪费。PagedAttention 借鉴操作系统分页，把 KV cache 分成 blocks/pages，通过映射表管理，减少碎片，支持按需分配、释放和 prefix sharing，从而提升 batch size 和吞吐。
```

### 题 7：FlashAttention 和 PagedAttention 有什么区别？

参考回答：

```text
FlashAttention 是 attention kernel 优化，通过 tiling 和 IO-aware 计算减少 HBM 读写和中间显存，不改变 attention 数学结果。PagedAttention 是 serving 系统中的 KV cache 管理方法，解决 cache 分配、碎片和共享问题。前者偏 kernel/计算，后者偏系统/内存管理。
```

### 题 8：MLA 和 GQA 的区别是什么？

参考回答：

```text
GQA 通过减少 KV head 数降低 KV cache。MLA 更进一步，把 K/V 信息压缩到 latent 表示中缓存，需要时再恢复或投影出 attention 所需信息。GQA 是减少 head 数，MLA 是压缩表示本身，因此 MLA 的 cache 压缩潜力更大，但实现复杂度也更高。
```

## 11.25 小练习

1. 推导 KV cache 显存公式，并解释每一项含义。
2. 计算 32 层、batch=8、seq_len=32768、num_kv_heads=8、head_dim=128、bf16 的 KV cache 大小。
3. 画出 prefill 和 decode 阶段的 query/key 长度关系。
4. 比较 MHA、MQA、GQA 的 KV cache 大小。
5. 解释为什么 decode 阶段常常受内存带宽限制。
6. 用自己的话解释 PagedAttention 为什么像操作系统分页。
7. 讨论 prefix sharing 在多轮对话和多样本采样中的作用。
8. 设计一个长上下文 serving 的监控面板，至少包含 TTFT、tokens/s、KV cache 使用率、batch size、cache fragmentation、OOM 数量。
9. 运行 11.22 的 KV Cache 成本审计 demo，把 `gqa_kv_heads` 从 8 改成 4，解释 cache 显存、decode 读取量和质量风险之间的 trade-off。
10. 把 `block_size` 从 16 改成 64，观察 `paged_waste_ratio` 的变化，并解释 block size 为什么不是越小越好。

## 11.26 本章总结

本章讲了 KV Cache、长上下文推理和显存增长问题。

核心结论：

1. KV cache 缓存每层历史 token 的 K/V，是 decoder-only LLM 高效自回归推理的基础。
2. Prefill 负责处理 prompt 并构建 cache，长 prompt 下受 O(n^2) attention 影响。
3. Decode 每步只处理一个新 token，但要读取历史 K/V，常受显存和内存带宽限制。
4. KV cache 大小随层数、batch、seq_len、num_kv_heads、head_dim 线性增长。
5. 长上下文和高并发 serving 中，KV cache 往往比模型权重更限制 batch size。
6. MQA/GQA 通过减少 KV head 数降低 cache；MLA 通过压缩 KV 表示进一步降低 cache。
7. PagedAttention/vLLM 通过分页式 cache 管理减少碎片和重复，提高吞吐。
8. 长上下文推理需要在延迟、吞吐、显存、质量和实现复杂度之间做系统 trade-off。
9. KV cache 审计要同时看 cache GiB、per-token cache 增长、decode 读取带宽、GQA/MQA 节省、block/page 浪费和上线门禁。

下一章会进入 In-Context Learning 与显式 token 检索能力，解释模型如何在上下文中利用示例、指令和证据 token 完成临时任务。
