# 第九章：实现 Transformer 组件

前面几章讲了 PyTorch 工程基础：tensor、autograd、Module、DataLoader、训练循环、混合精度、分布式和 debug。本章把这些能力集中到一个目标上：手写 Transformer 的核心组件。大模型算法岗面试里，最常见的代码题和工程追问往往不是让你完整复现一个大模型，而是让你能讲清并写出 embedding、attention、causal mask、MLP、LayerNorm、RMSNorm、RoPE、KV Cache 和 decoder block 的最小实现。

本章代码以教学清晰为第一目标，不追求 FlashAttention、PagedAttention、张量并行或极致性能。真实大模型会有更复杂的 kernel、并行策略和缓存管理，但只要能稳定手写本章组件，你就能理解大多数 decoder-only LLM 的主体结构。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已核对 Transformer 原论文中的 scaled dot-product attention / multi-head attention 口径、PyTorch 官方 `scaled_dot_product_attention`、`nn.MultiheadAttention`、`nn.Embedding`、`nn.LayerNorm`、`nn.RMSNorm` 和 `nn.Transformer` 文档，以及 RoPE / RoFormer 论文中旋转位置编码的基本定义。

本章聚焦面试和个人项目中最常需要手写、解释和 debug 的 decoder-only Transformer 组件：token embedding、LM head 权重绑定、RMSNorm、causal mask、scaled dot-product attention、multi-head self-attention、SwiGLU MLP、Pre-Norm decoder block、RoPE、causal LM shift loss 和教学版 KV Cache。

本章不展开 FlashAttention kernel、PagedAttention、MQA / GQA、tensor parallel、pipeline parallel、FSDP / ZeRO、`torch.compile`、生产级 serving scheduler、block table、prefix cache 或复杂长上下文外推策略。这些内容会在训练、部署、架构演进和推理框架章节中继续展开。

## 9.1 Transformer 组件总览

一个简化 decoder-only Transformer 通常包含：

1. Token embedding。
2. 位置编码或 RoPE。
3. 多层 Transformer block。
4. 每个 block 包含 causal self-attention、MLP、norm 和 residual。
5. 最后的 norm。
6. LM head，把 hidden states 投到词表 logits。

数据流：

```text
input_ids [B, T]
-> embedding [B, T, d]
-> block 1
-> block 2
-> ...
-> final norm
-> lm_head logits [B, T, V]
```

面试回答：

```text
Decoder-only Transformer 的主干是 token embedding、多层 causal self-attention 和 MLP block、final norm 以及 LM head。每个 block 通常包含 pre-norm、self-attention、residual、pre-norm、MLP、residual。训练时输出 [B,T,V] logits，用当前位置预测下一个 token。
```

## 9.1.1 关键公式与 Transformer 组件速查

第一，decoder-only LM 的输入输出 shape 主线是：

```math
I\in\mathbb{Z}^{B\times T},\qquad X=E[I]\in\mathbb{R}^{B\times T\times d},\qquad Z=XW_{\mathrm{lm}}\in\mathbb{R}^{B\times T\times V}
```

其中 `B` 是 batch size，`T` 是序列长度，`d` 是 hidden size，`V` 是词表大小。

第二，多头拆分要求：

```math
d = H D_h,\qquad Q,K,V\in\mathbb{R}^{B\times H\times T\times D_h}
```

如果 `hidden_size % num_heads != 0`，就无法把 hidden state 均匀拆成多个 head。

第三，scaled dot-product attention 是：

```math
S=\frac{QK^\top}{\sqrt{D_h}}+M,\qquad A=\mathrm{softmax}(S),\qquad O=AV
```

其中 `S` 的 shape 是 `[B,H,T_q,T_k]`，`A` 的最后一维应逐行求和为 1。

第四，普通 causal mask 可以写成：

```math
M_{t,s}=
\begin{cases}
0, & s\le t\\
-\infty, & s>t
\end{cases}
```

如果使用 KV Cache，当前 query 只包含新 token，key/value 包含历史 token，此时更稳的形式是：

```math
M_{t,s}=
\begin{cases}
0, & s\le p+t\\
-\infty, & s>p+t
\end{cases}
```

其中 `p` 是已经缓存的历史 key/value 长度。这个细节能避免 decode 阶段 `query_len=1,key_len>1` 时 mask 方向错。

第五，RMSNorm 的核心是按最后一维均方根缩放：

```math
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\qquad y_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}
```

它不减均值，计算比 LayerNorm 更简单。

第六，SwiGLU MLP 可以抽象为：

```math
\mathrm{SwiGLU}(x)=W_d\left(\mathrm{SiLU}(xW_g)\odot xW_u\right)
```

输入输出 shape 都是 `[B,T,d]`，中间维度通常是 `intermediate_size`。

第七，Pre-Norm decoder block 的主线是：

```math
x' = x+\mathrm{Attn}(\mathrm{Norm}(x)),\qquad x_{\mathrm{out}}=x'+\mathrm{MLP}(\mathrm{Norm}(x'))
```

这也是现代 decoder-only LLM 中最常见的 block 结构。

第八，RoPE 在每个二维子空间上做旋转：

```math
\begin{bmatrix}
q'_{2i}\\
q'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{m,i} & -\sin\theta_{m,i}\\
\sin\theta_{m,i} & \cos\theta_{m,i}
\end{bmatrix}
\begin{bmatrix}
q_{2i}\\
q_{2i+1}
\end{bmatrix}
```

同样的旋转也作用到 `K`。因为旋转矩阵保持向量范数，所以 RoPE 改变位置信息，不应改变每个二维子向量的长度。

第九，causal LM loss 需要右移：

```math
L=-\frac{1}{N}\sum_{b,t}\log p_\theta(y_{b,t+1}\mid y_{b,\le t})
```

实现时通常是 `shift_logits = logits[:, :-1, :]`，`shift_labels = labels[:, 1:]`，并用 `ignore_index=-100` 屏蔽 padding 或 prompt-only 位置。

第十，KV Cache 的单层显存量级可以粗略写成：

```math
M_{\mathrm{kv,layer}}=2 B H_{\mathrm{kv}} T_{\mathrm{cache}} D_h b
```

其中因子 2 来自 K 和 V，`b` 是每个元素字节数。

## 9.2 配置对象

为了避免到处传参数，先定义一个简单配置。

```python
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    dropout: float = 0.0
```

要求：

```python
assert config.hidden_size % config.num_heads == 0
```

因为每个 head 的维度是：

```python
head_dim = hidden_size // num_heads
```

## 9.3 Token Embedding 和 LM Head

Token embedding 把离散 token id 变成连续向量。

```python
import torch
from torch import nn


embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

input_ids = torch.randint(0, config.vocab_size, (2, 8))
x = embed_tokens(input_ids)
print(x.shape)  # [B, T, d]
```

LM head 把 hidden states 投到词表大小：

```python
lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
logits = lm_head(x)
print(logits.shape)  # [B, T, V]
```

很多语言模型会做权重绑定：

```python
lm_head.weight = embed_tokens.weight
```

含义是输入 embedding 和输出词表投影共享同一份权重。优点是减少参数量，并让输入输出 token 表示空间更一致。

## 9.4 手写 RMSNorm

现代 LLM 经常使用 RMSNorm。它和 LayerNorm 的区别是：RMSNorm 不减均值，只按均方根归一化。

公式直觉：

```text
rms = sqrt(mean(x * x) + eps)
y = x / rms * weight
```

PyTorch 实现：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
```

这里把 `x` 临时转成 fp32，是为了数值更稳。最后再转回原 dtype。

Shape：

```text
x: [B, T, d]
weight: [d]
output: [B, T, d]
```

## 9.5 Causal Mask

Causal mask 防止当前位置看到未来 token。

最小实现：

```python
def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask[None, None, :, :]  # [1, 1, T, T]
```

使用：

```python
scores = torch.randn(2, 8, 16, 16)
mask = make_causal_mask(16, scores.device)
scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
```

注意：不要在 fp16 中随便写很大的负数，比如 `-1e30`，可能溢出。更稳妥的是使用：

```python
torch.finfo(scores.dtype).min
```

## 9.6 Scaled Dot-Product Attention

输入：

```text
q: [B, H, T, D]
k: [B, H, S, D]
v: [B, H, S, D]
```

输出：

```text
context: [B, H, T, D]
```

实现：

```python
import math


def make_causal_mask(query_len, key_len, device, past_key_values_length=0):
    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    return key_positions <= query_positions + past_key_values_length


def scaled_dot_product_attention(q, k, v, causal=True, past_key_values_length=0):
    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    if causal:
        query_len = q.size(-2)
        key_len = k.size(-2)
        mask = make_causal_mask(query_len, key_len, q.device, past_key_values_length)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    context = attn @ v
    return context
```

这里的关键是：

1. `q @ k.transpose(-2, -1)` 得到 attention scores。
2. 除以 `sqrt(head_dim)` 稳定 softmax。
3. causal mask 禁止看未来。
4. softmax 后乘 `v` 得到上下文表示。

这里额外传入 `past_key_values_length`，是为了兼容 KV Cache decode：当 `q` 只有当前 token、`k/v` 包含历史 token 时，当前 query 应该能看到全部历史 key，而不是只看到 key 序列的第 0 个位置。

## 9.7 Multi-Head Self-Attention

把输入 `[B, T, d]` 投影成多头 Q/K/V。

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, T, D]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)

        context = scaled_dot_product_attention(q, k, v, causal=True)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)
```

Shape 主线：

```text
x: [B, T, d]
q/k/v after projection: [B, T, d]
q/k/v after reshape: [B, H, T, D]
context: [B, H, T, D]
merged context: [B, T, d]
```

## 9.8 MLP 和 SwiGLU

Transformer block 里的 MLP 负责逐 token 的非线性变换。

普通 MLP：

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

现代 LLM 常用 SwiGLU 变体：

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

Shape 不变：

```text
input: [B, T, d]
output: [B, T, d]
```

## 9.9 Decoder Block

现代 decoder-only LLM 通常用 Pre-Norm 结构。

```python
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x
```

核心结构：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

面试中要能解释：attention 负责 token 间通信，MLP 负责每个 token 表示的非线性加工，residual 帮助深层训练和信息保留，norm 稳定激活尺度。

## 9.10 RoPE 的最小实现

RoPE 作用在 Q/K 上，通过旋转注入位置信息。

先构造频率：

```python
def build_rope_cache(seq_len, head_dim, device, base=10000):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # [T, D/2]
    cos = freqs.cos()[None, None, :, :]       # [1, 1, T, D/2]
    sin = freqs.sin()[None, None, :, :]       # [1, 1, T, D/2]
    return cos, sin
```

旋转函数：

```python
def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)
```

输入输出：

```text
x: [B, H, T, D]
cos/sin: [1, 1, T, D/2]
output: [B, H, T, D]
```

在 attention 中使用：

```python
cos, sin = build_rope_cache(seq_len, self.head_dim, x.device)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

RoPE 的重点不是背代码，而是知道它通过对 Q/K 按位置旋转，让 attention score 包含相对位置信息。

## 9.11 加入 RoPE 的 Attention

```python
class RoPEMultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)

        cos, sin = build_rope_cache(seq_len, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        context = scaled_dot_product_attention(q, k, v, causal=True)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)
```

真实工程里通常会缓存 `cos` 和 `sin`，而不是每次 forward 都重新构造。

## 9.12 最小 Decoder-only LM

把组件组合起来：

```python
class MiniDecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}
```

测试 shape：

```python
config = TransformerConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4, intermediate_size=512)
model = MiniDecoderLM(config)

input_ids = torch.randint(0, config.vocab_size, (2, 16))
labels = input_ids.clone()
outputs = model(input_ids, labels=labels)

print(outputs["logits"].shape)  # [2, 16, 1000]
print(outputs["loss"])
```

## 9.13 KV Cache 的直觉

训练时通常一次输入完整序列。推理生成时，每次只生成一个新 token。如果每一步都重新计算整个历史序列的 K/V，会浪费大量计算。

KV Cache 的思路是：

1. 第一次 prefill 时计算 prompt 的 K/V。
2. 后续 decode 每生成一个 token，只计算新 token 的 K/V。
3. 把新 K/V 拼到历史 cache 后面。
4. query 只来自当前 token，但 key/value 来自完整历史。

没有 cache：每步重复算历史。

有 cache：历史 K/V 复用。

## 9.14 简化 KV Cache Attention

注意：下面是教学版，只展示核心逻辑。

```python
class CachedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x, batch_size, seq_len):
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape

        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        context = scaled_dot_product_attention(
            q,
            k,
            v,
            causal=True,
            past_key_values_length=past_len,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)

        new_cache = (k, v) if use_cache else None
        return output, new_cache
```

KV Cache shape：

```text
k: [B, H, T_cache, D]
v: [B, H, T_cache, D]
```

真实推理框架还要处理 batch 内不同长度、分页管理、prefix cache、内存复用和连续批处理。

## 9.15 最小可运行 Transformer 组件审计 demo

下面这个 demo 用很小的配置把本章关键组件串起来。它的目标不是训练出好模型，而是帮助你在面试或项目 debug 时快速确认：shape、mask、loss shift、RoPE 范数、权重绑定和 KV Cache 长度都符合预期。

```python
import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(7)
torch.set_num_threads(1)


@dataclass
class TransformerConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 32
    max_position_embeddings: int = 16
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


def make_causal_mask(query_len, key_len, device, past_key_values_length=0):
    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    mask = key_positions <= query_positions + past_key_values_length
    return mask[None, None, :, :]


def scaled_dot_product_attention(
    q,
    k,
    v,
    causal=True,
    past_key_values_length=0,
    return_attn=False,
):
    head_dim = q.size(-1)
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)

    if causal:
        mask = make_causal_mask(
            q.size(-2),
            k.size(-2),
            q.device,
            past_key_values_length=past_key_values_length,
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    context = attn @ v
    if return_attn:
        return context, attn
    return context


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x, return_attn=False):
        batch_size, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)
        context, attn = scaled_dot_product_attention(q, k, v, return_attn=True)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(context)
        if return_attn:
            return out, attn
        return out


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


def build_rope_cache(seq_len, head_dim, device, base=10000):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class MiniDecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, labels=None):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}


class CachedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x, batch_size, seq_len):
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x), batch_size, seq_len)
        k = self._shape(self.k_proj(x), batch_size, seq_len)
        v = self._shape(self.v_proj(x), batch_size, seq_len)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        context, attn = scaled_dot_product_attention(
            q,
            k,
            v,
            causal=True,
            past_key_values_length=past_len,
            return_attn=True,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        new_cache = (k, v) if use_cache else None
        return self.o_proj(context), new_cache, attn


config = TransformerConfig()
batch_size, seq_len = 2, 5
x = torch.randn(batch_size, seq_len, config.hidden_size)
input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
labels = input_ids.clone()

rms = RMSNorm(config.hidden_size)
rms_out = rms(x)
rms_mean_square = rms_out.pow(2).mean(dim=-1)[0, 0].item()

mask = make_causal_mask(4, 4, x.device)
mask_rows = mask[0, 0].int().tolist()

q = torch.randn(batch_size, config.num_heads, seq_len, config.hidden_size // config.num_heads)
k = torch.randn_like(q)
v = torch.randn_like(q)
context, attn = scaled_dot_product_attention(q, k, v, return_attn=True)
future_mask = ~make_causal_mask(seq_len, seq_len, x.device)
future_weight_max = attn.masked_select(future_mask).max().item()

mha = MultiHeadSelfAttention(config)
mha_out, mha_attn = mha(x, return_attn=True)
mlp_out = SwiGLUMLP(config)(x)
block_out = DecoderBlock(config)(x)

cos, sin = build_rope_cache(seq_len, config.hidden_size // config.num_heads, x.device)
rope_in = q.clone()
rope_out = apply_rope(rope_in, cos, sin)
rope_norm_preserved = torch.allclose(rope_in.norm(dim=-1), rope_out.norm(dim=-1), atol=1e-5)

model = MiniDecoderLM(config)
outputs = model(input_ids, labels=labels)
tied_weights = model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()

cache_attn = CachedSelfAttention(config)
prefill_out, cache, prefill_attn = cache_attn(x, use_cache=True)
next_x = torch.randn(batch_size, 1, config.hidden_size)
decode_out, cache, decode_attn = cache_attn(next_x, past_key_value=cache, use_cache=True)
decode_can_see_all_cache = decode_attn[0, 0, 0].numel() == seq_len + 1

summary = {
    "rms_shape": tuple(rms_out.shape),
    "rms_mean_square_first": round(rms_mean_square, 4),
    "mask_rows": mask_rows,
    "context_shape": tuple(context.shape),
    "attn_row_sums": [round(v, 4) for v in attn[0, 0].sum(dim=-1).tolist()],
    "future_weight_max": round(future_weight_max, 4),
    "mha_shape": tuple(mha_out.shape),
    "mha_attn_shape": tuple(mha_attn.shape),
    "swiglu_shape": tuple(mlp_out.shape),
    "block_shape": tuple(block_out.shape),
    "rope_shape": tuple(rope_out.shape),
    "rope_norm_preserved": bool(rope_norm_preserved),
    "logits_shape": tuple(outputs["logits"].shape),
    "loss": round(outputs["loss"].item(), 4),
    "tied_weights": tied_weights,
    "cache_lengths": (prefill_attn.size(-1), cache[0].size(2)),
    "decode_attn_shape": tuple(decode_attn.shape),
    "decode_can_see_all_cache": decode_can_see_all_cache,
}

checks = {
    "rms_shape": summary["rms_shape"] == (2, 5, 16),
    "mask_direction": mask_rows[-1] == [1, 1, 1, 1],
    "attention_shape": summary["context_shape"] == (2, 4, 5, 4),
    "attention_rows_sum_to_one": all(v == 1.0 for v in summary["attn_row_sums"]),
    "future_mask_zero": summary["future_weight_max"] == 0.0,
    "mha_shape": summary["mha_shape"] == (2, 5, 16),
    "swiglu_shape": summary["swiglu_shape"] == (2, 5, 16),
    "block_shape": summary["block_shape"] == (2, 5, 16),
    "rope_norm": summary["rope_norm_preserved"],
    "lm_shape": summary["logits_shape"] == (2, 5, 32),
    "weight_tying": summary["tied_weights"],
    "cache_grows": summary["cache_lengths"] == (5, 6),
    "decode_cache_visible": summary["decode_can_see_all_cache"],
}

print("summary=", summary)
print("checks=", checks)
print("gate_pass=", all(checks.values()))
```

运行后应看到所有 checks 为 `True`，并且 `gate_pass=True`。如果某一项失败，优先按下面顺序排查：先看 shape，再看 mask 方向，再看 `contiguous().view(...)`，最后看 loss 是否正确右移。

## 9.16 常见实现 bug

### 9.16.1 head_dim 不整除

```python
assert hidden_size % num_heads == 0
```

否则无法把 hidden size 平均拆到多头。

### 9.16.2 mask 方向错

Causal mask 应该允许当前位置看自己和过去，禁止看未来。

长度为 4 的 mask 应该类似：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

KV Cache decode 时还要检查 `query_len` 和 `key_len` 不相等的情况。当前 token 的 query 应该能看见完整历史 cache，否则输出会只依赖很短的历史。

### 9.16.3 transpose 后忘记 contiguous

```python
context = context.transpose(1, 2).contiguous().view(B, T, d)
```

如果直接 `view`，可能因为内存 layout 不连续而报错。

### 9.16.4 loss 没 shift

语言模型训练需要：

```python
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

### 9.16.5 padding 参与 loss

padding 或 prompt 中不训练的位置要设成 `-100`，并在 cross entropy 里使用 `ignore_index=-100`。

## 9.17 面试官会怎么问

### 问题一：Self-attention 的 shape 怎么变化？

回答模板：

```text
输入 hidden states 是 [B,T,d]。经过 Q/K/V 投影后仍是 [B,T,d]，再 reshape 成 [B,H,T,D]，其中 D=d/H。计算 Q @ K.transpose(-2,-1) 得到 scores [B,H,T,T]，softmax 后乘 V 得到 context [B,H,T,D]，最后合并多头回到 [B,T,d]。
```

### 问题二：为什么需要 causal mask？

回答模板：

```text
Decoder-only 语言模型要按自回归方式预测下一个 token，当前位置不能看到未来 token。训练时虽然整段序列并行输入，但 causal mask 会在 attention scores 上屏蔽未来位置，防止信息泄漏。
```

### 问题三：RMSNorm 和 LayerNorm 有什么区别？

回答模板：

```text
LayerNorm 会减均值并除以标准差，RMSNorm 不减均值，只按均方根缩放。RMSNorm 计算更简单，是很多现代 LLM 常用的归一化方式。
```

### 问题四：RoPE 作用在哪里？

回答模板：

```text
RoPE 通常作用在 attention 的 Q 和 K 上，对它们按位置做旋转。这样 Q 和 K 的点积会自然包含相对位置信息，因此模型可以在 attention score 中感知 token 之间的相对距离。
```

### 问题五：KV Cache 为什么能加速推理？

回答模板：

```text
自回归推理每步只新增一个 token，历史 token 的 K/V 不变。KV Cache 缓存历史 K/V，后续 decode 只计算新 token 的 K/V，并和历史 cache 拼接，避免每一步重复计算完整上下文。
```

## 9.18 小练习

1. 手写 RMSNorm，并验证输入输出 shape 不变。
2. 写出长度为 5 的 causal mask。
3. 手写 scaled dot-product attention，输入 `[B,H,T,D]`。
4. 写一个 MultiHeadSelfAttention，并打印每一步 shape。
5. 把普通 MLP 改成 SwiGLU MLP。
6. 写一个 DecoderBlock，确认 residual 前后 shape 一致。
7. 给 attention 加 RoPE。
8. 写一个最小 decoder-only LM，并跑一次 forward 和 loss。
9. 给 attention 加一个教学版 KV Cache。
10. 故意去掉 causal mask，解释为什么训练 loss 可能异常偏低。
11. 修改本章审计 demo，让 `seq_len=1` 的 decode token 使用长度为 6 的 KV Cache，并验证 attention 权重长度也是 6。

## 9.19 本章总结

本章把 PyTorch 基础能力落到了 Transformer 组件实现上。需要掌握的主线是：embedding 把 token id 转成 hidden states，attention 负责 token 间通信，MLP 负责逐 token 非线性变换，norm 和 residual 保证深层训练稳定，RoPE 注入位置信息，KV Cache 加速自回归推理。

核心记忆点：

1. 输入 `input_ids` 是 `[B,T]`，embedding 后是 `[B,T,d]`。
2. 多头 attention 中 Q/K/V 是 `[B,H,T,D]`。
3. attention scores 是 `[B,H,T,T]`。
4. causal mask 禁止看未来。
5. decoder block 常见结构是 `x = x + Attention(Norm(x))` 和 `x = x + MLP(Norm(x))`。
6. RoPE 作用在 Q/K 上。
7. KV Cache 缓存历史 K/V，减少 decode 阶段重复计算。

下一章会进入工程面试题，集中整理 PyTorch 工程、训练循环、显存、分布式和 Transformer 实现相关的高频问答。
