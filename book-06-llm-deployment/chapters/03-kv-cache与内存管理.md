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

## 本章资料边界

本章第二轮精修时对照了 Hugging Face Transformers KV Cache 文档、vLLM / PagedAttention 论文与实现说明、TensorRT-LLM paged KV cache / KV cache manager 资料，以及 GQA / MQA 相关论文资料。这里聚焦部署面试和工程估算中最常用的 KV Cache 抽象：

1. KV Cache 的 shape、单位 token 显存和总显存。
2. MHA、MQA、GQA 如何通过 `num_kv_heads` 改变 cache 成本。
3. PagedAttention 如何把逻辑连续的 cache 映射到物理 block。
4. KV Cache 量化、prefix cache 和 OOM 排查的边界。

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

写成数学形式：

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
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

对第 `l` 层，常见 KV Cache shape 可以写成：

```math
K_l,V_l
\in
\mathbb{R}^{B\times H_{\mathrm{kv}}\times T_{\mathrm{ctx}}\times D_h}
```

其中 `B` 是 batch size，`H_kv` 是 K/V head 数，`T_ctx` 是当前上下文长度，`D_h` 是 head dimension。

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

如果输入 token 数是 `T_in`，已经生成了 `t` 个 token，则当前 cache 长度是：

```math
T_{\mathrm{ctx}}(t)=T_{\mathrm{in}}+t
```

然后新 token 的 query attend 到完整 cache。

面试表达：prefill 负责建立初始 KV Cache，decode 负责逐步追加并复用 KV Cache。

## 3. KV Cache 为什么加速

假设已经生成到长度 `T`。

没有 KV Cache，每一步都要对长度 `T` 的完整序列重新算 K/V。

有 KV Cache，每一步只需要计算新 token 的 K/V。

这避免了历史 token 的重复投影计算。

可以把单步 K/V 投影成本粗略对比成：

```math
C_{\mathrm{no\_cache}}
\propto
T_{\mathrm{ctx}}\cdot d^2
```

```math
C_{\mathrm{cache}}
\propto
d^2
```

这里的 `d` 是 hidden size。这个对比只说明 K/V 投影重复计算被省掉，不代表 attention 访问成本被省掉。

但注意：KV Cache 不代表 attention 成本完全消失。

新 token 仍然要 attend 到所有历史 K/V，所以随着上下文变长，decode 每步仍会变慢。

面试表达：KV Cache 省掉的是历史 token K/V 的重复计算，但不会消除新 token 对历史上下文的 attention 访问。

## 4. KV Cache 显存估算

KV Cache 显存大致和这些因素成正比：

```text
batch_size * seq_len * num_layers * num_kv_heads * head_dim * 2 * bytes
```

其中 `2` 表示 K 和 V 两份缓存。

更稳定的公式写法是：

```math
M_{\mathrm{token}}
=
2\cdot L\cdot H_{\mathrm{kv}}\cdot D_h\cdot b
```

```math
M_{\mathrm{KV}}
=
B\cdot T_{\mathrm{ctx}}\cdot M_{\mathrm{token}}
=
2\cdot L\cdot B\cdot T_{\mathrm{ctx}}\cdot H_{\mathrm{kv}}\cdot D_h\cdot b
```

其中 `M_token` 是单个 token 在所有层的 KV Cache 显存，`b` 是每个 cache 数值的字节数。

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

如果 query head 数是 `H`，KV head 数是 `H_kv`，在其他配置相同的情况下：

```math
R_{\mathrm{KV}}
=
\frac{M_{\mathrm{KV}}(H_{\mathrm{kv}})}{M_{\mathrm{KV}}(H)}
=
\frac{H_{\mathrm{kv}}}{H}
```

例如 `H=32` 且 `H_kv=8` 时，GQA 的 KV Cache 约为 MHA 的 `1/4`；MQA 可以近似看作 `H_kv=1`。

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

如果 block size 是 `S_block`，某个请求当前长度是 `T_ctx`，需要的 block 数是：

```math
N_{\mathrm{block}}
=
\left\lceil \frac{T_{\mathrm{ctx}}}{S_{\mathrm{block}}} \right\rceil
```

最后一个 block 的 token 浪费可以粗略写成：

```math
W_{\mathrm{last}}
=
N_{\mathrm{block}}\cdot S_{\mathrm{block}}-T_{\mathrm{ctx}}
```

分页式管理不能让 KV Cache 数学上消失，但可以减少连续预留、碎片和动态增长带来的浪费。

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

如果从 BF16/FP16 的 2 bytes 降到 INT8 的 1 byte，理想显存比例是：

```math
M_{\mathrm{KV,int8}}
\approx
\frac{1}{2}M_{\mathrm{KV,bf16}}
```

实际收益还会受 scale、zero point、block metadata 和 kernel 支持影响。

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

如果共享前缀 token 数是 `T_shared`，完整 prompt token 数是 `T_prompt`，粗略 prefill 复用比例可以写成：

```math
S_{\mathrm{prefix}}
\approx
\frac{T_{\mathrm{shared}}}{T_{\mathrm{prompt}}}
```

这只是成本直觉；真实系统要求 token 序列、模板版本、权限边界和 cache key 完全可控。

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

## 13. 最小可运行 KV Cache 内存审计 demo

下面这个 0 依赖 demo 用 3 个 toy 请求估算 MHA / GQA / MQA 的 KV Cache 显存差异、PagedAttention block 分配浪费、连续预留和分页分配的差异，以及 INT8 KV Cache 的理想收益。它不是框架实现，只用于把内存账算清楚。

```python
import math


requests = [
    {"id": "chat", "prompt_tokens": 600, "generated_tokens": 80, "reserved_tokens": 1024},
    {"id": "rag_long", "prompt_tokens": 3200, "generated_tokens": 160, "reserved_tokens": 4096},
    {"id": "agent", "prompt_tokens": 1200, "generated_tokens": 220, "reserved_tokens": 2048},
]

layers = 32
query_heads = 32
head_dim = 128
bytes_bf16 = 2
block_size = 128
gpu_budget_gib = 9.3
weight_gib = 7.5
workspace_gib = 1.0
kv_head_options = {"MHA": 32, "GQA": 8, "MQA": 1}


def kv_bytes(tokens, kv_heads, dtype_bytes=bytes_bf16):
    return 2 * layers * tokens * kv_heads * head_dim * dtype_bytes


def gib(num_bytes):
    return num_bytes / 1024**3


active_tokens = sum(req["prompt_tokens"] + req["generated_tokens"] for req in requests)
reserved_tokens = sum(req["reserved_tokens"] for req in requests)

memory = {
    name: round(gib(kv_bytes(active_tokens, kv_heads)), 3)
    for name, kv_heads in kv_head_options.items()
}
reserved_memory_gqa = round(gib(kv_bytes(reserved_tokens, kv_head_options["GQA"])), 3)
active_memory_gqa = memory["GQA"]

page_rows = []
for req in requests:
    active = req["prompt_tokens"] + req["generated_tokens"]
    blocks = math.ceil(active / block_size)
    allocated = blocks * block_size
    page_rows.append(
        {
            "id": req["id"],
            "active": active,
            "blocks": blocks,
            "waste": allocated - active,
        }
    )

paged_tokens = sum(row["blocks"] * block_size for row in page_rows)
paged_memory_gqa = round(gib(kv_bytes(paged_tokens, kv_head_options["GQA"])), 3)
last_block_waste_tokens = sum(row["waste"] for row in page_rows)
fragmentation_saved_gib = round(reserved_memory_gqa - paged_memory_gqa, 3)
int8_gqa_gib = round(active_memory_gqa / 2, 3)
fit_gate = weight_gib + workspace_gib + paged_memory_gqa <= gpu_budget_gib
reserved_gate = weight_gib + workspace_gib + reserved_memory_gqa <= gpu_budget_gib

print(f"active_tokens={active_tokens}")
print(f"kv_memory_gib={memory}")
print(f"gqa_vs_mha_ratio={memory['GQA'] / memory['MHA']:.2f}")
print(f"mqa_vs_mha_ratio={memory['MQA'] / memory['MHA']:.3f}")
print(f"page_rows={page_rows}")
print(f"paged_tokens={paged_tokens}")
print(f"last_block_waste_tokens={last_block_waste_tokens}")
print(f"reserved_memory_gqa_gib={reserved_memory_gqa}")
print(f"paged_memory_gqa_gib={paged_memory_gqa}")
print(f"fragmentation_saved_gib={fragmentation_saved_gib}")
print(f"int8_gqa_gib={int8_gqa_gib}")
print(f"fit_gate={fit_gate}")
print(f"reserved_gate={reserved_gate}")
print(f"gate_pass={fit_gate and not reserved_gate and fragmentation_saved_gib > 0}")
```

这段 demo 的关键输出应该是：

```text
active_tokens=5460
kv_memory_gib={'MHA': 2.666, 'GQA': 0.667, 'MQA': 0.083}
gqa_vs_mha_ratio=0.25
mqa_vs_mha_ratio=0.031
page_rows=[{'id': 'chat', 'active': 680, 'blocks': 6, 'waste': 88}, {'id': 'rag_long', 'active': 3360, 'blocks': 27, 'waste': 96}, {'id': 'agent', 'active': 1420, 'blocks': 12, 'waste': 116}]
paged_tokens=5760
last_block_waste_tokens=300
reserved_memory_gqa_gib=0.875
paged_memory_gqa_gib=0.703
fragmentation_saved_gib=0.172
int8_gqa_gib=0.334
fit_gate=True
reserved_gate=False
gate_pass=True
```

这个 demo 想说明：KV Cache 显存主要由 `H_kv`、活跃 token 数和 dtype 决定；GQA 相比 MHA 直接降到 `1/4`，分页式管理会有最后一个 block 浪费，但能避免按最大长度连续预留带来的更大浪费。

## 14. 本章小结

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
