# 第 2 章 Transformer 组件实战：从张量形状到现代 LLM Block

Transformer 的实现并不是把几个类名拼在一起。一个输入 token 经过 embedding、位置处理、注意力、归一化和前馈网络，期间会反复改变张量的维度组织；任何一个维度的误读，都可能让程序继续运行，却让模型学习到另一种函数。

本章把这些组件放在同一条数据路径中讲解。先从离散 token id 进入连续表示开始，再逐步建立 scaled dot-product attention、multi-head attention、mask、Transformer block、RoPE、RMSNorm 和 SwiGLU。每个主题都给出数学对象、PyTorch 中的 shape、可运行的最小实现和边界条件。代码采用教学构造，示例输出只说明形状或不变量，不代表任何真实模型的能力指标。

资料分成三类。Transformer 和 RoPE 的公式来自论文；张量操作和模块行为以 PyTorch 官方文档为准；模块组合、断言和诊断实验是教学实现。论文中的结构描述、框架的接口语义和某个目标系统的实测性能不能互相替代。

## 2.1 Token Embedding 与位置表示：让离散序列进入连续空间

### 2.1.1 token id 是索引，不是有序特征

Tokenizer 把文本变成整数序列，例如：

```text
我 -> 125
爱 -> 876
你 -> 42
```

这些数字的大小没有语义。编号 876 不比编号 42 “更强”，相邻编号也不意味着两个 token 相似。如果把 [125, 876, 42] 当作普通浮点特征输入线性层，模型会被迫把编号的数值关系当成可学习规律的一部分。

Embedding 的职责是建立一个从离散索引到连续向量的映射。设词表大小为 \(V\)，模型隐藏维度为 \(D\)，token embedding 矩阵为：

```math
E\in\mathbb{R}^{V\times D}
```

输入 token id 为 \(i\) 时，输出就是矩阵的第 \(i\) 行：

```math
e_i=E_{i,:}
```

若一个 batch 的 token id 形状为 \([B,T]\)，查表后得到：

```math
\mathrm{Embedding}(X)\in\mathbb{R}^{B\times T\times D}
```

其中 \(B\) 是批次大小，\(T\) 是序列长度，\(D\) 是 hidden size。这个形状会成为后续 Transformer block 的接口契约。

初学者可以把 embedding 想成一本有 \(V\) 行的词典，每一行是一张 \(D\) 维的“特征卡片”；专家则应进一步注意，embedding 查表是稀疏访问：一个 batch 只会让出现过的行获得来自当前损失的直接梯度。

### 2.1.2 查表与 one-hot 矩阵乘法是同一个数学运算

设词表只有 5 个 token，id 为 2 的 one-hot 向量是：

```math
u_2=[0,0,1,0,0]
```

用行向量表示时：

```math
u_2E=E_{2,:}
```

所以 one-hot 乘法与索引查表在数学上等价。工程实现不应真的构造 \([B,T,V]\) 的 one-hot 张量，因为 \(V\) 可能达到数万乃至数十万；直接读取需要的行可以避免大量零值存储和乘法。

下面用一个小矩阵验证两种写法的结果一致：

```python
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
vocab_size = 5
hidden_size = 3
input_ids = torch.tensor([[0, 2, 4], [1, 2, 3]], dtype=torch.long)

embedding = nn.Embedding(vocab_size, hidden_size)
lookup = embedding(input_ids)
one_hot = F.one_hot(input_ids, num_classes=vocab_size)
one_hot = one_hot.to(dtype=embedding.weight.dtype)
matmul = one_hot @ embedding.weight

print("lookup_shape=", tuple(lookup.shape))
print("one_hot_shape=", tuple(one_hot.shape))
print("outputs_close=", torch.allclose(lookup, matmul))
```

输出的关键不变量是：

```text
lookup_shape= (2, 3, 3)
one_hot_shape= (2, 3, 5)
outputs_close= True
```

nn.Embedding 的输入应是整数索引，通常使用 torch.long；每个索引必须满足 \(0\le i<V\)。将索引先转成浮点数不会让 embedding 获得更丰富的信息，反而会导致接口错误。

### 2.1.3 参数量和梯度的稀疏访问

Embedding 矩阵的参数量为：

```math
N_{\mathrm{token}}=VD
```

例如 \(V=100000,D=4096\) 时：

```math
N_{\mathrm{token}}=100000\times4096=409{,}600{,}000
```

如果权重以 BF16 保存，每个元素占 2 bytes，裸权重约为 819.2 MB。训练显存还要加上梯度、优化器状态、临时激活以及可能的 FP32 master copy；不能把这个数字当成训练显存总量。

下面观察哪些行收到梯度：

```python
import torch
from torch import nn

torch.manual_seed(3)
embedding = nn.Embedding(10, 4)
input_ids = torch.tensor([[1, 1, 4], [7, 4, 4]], dtype=torch.long)
loss = embedding(input_ids).pow(2).mean()
loss.backward()

row_norms = embedding.weight.grad.norm(dim=-1)
used_rows = [i for i, value in enumerate(row_norms) if value > 0]
print("used_rows=", used_rows)
print("unused_row_0_norm=", round(row_norms[0].item(), 6))
```

在没有额外正则项、且只计算这一个 batch 损失时，used_rows 应包含 1、4、7，未出现的行没有这次损失的直接梯度。若使用权重衰减、共享输出头或其他正则项，未出现行仍可能被更新；“未被索引访问”和“最终参数绝对不变”是两个不同命题。

### 2.1.4 padding_idx、attention mask 和 loss mask

变长序列通常需要 padding：

```text
样本 A: [12, 35, 98, 77]
样本 B: [21, 43,  0,  0]
```

如果 0 是 padding token，可以写：

```python
from torch import nn

vocab_size = 100
d_model = 32
token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=d_model,
    padding_idx=0,
)
```

padding_idx 作用在 embedding 参数这一层：该行在常规反向传播中不累积梯度，并通常被初始化为零。它不负责修改 attention 的可见性，也不负责改变 labels。三个机制的职责如下：

| 机制 | 解决的问题 | 作用位置 |
| --- | --- | --- |
| padding_idx | padding 行是否由常规 loss 更新 | embedding 参数 |
| attention mask | query 能否读取某个 key | attention score |
| loss mask | 某个目标位置是否计入损失 | loss reduction |

一个完整的 batch 往往需要同时使用它们。只设置 padding_idx，padding key 仍可能被 attention 读取；只使用 attention mask，padding 位置仍可能进入 token loss。实际的 pad_token_id 必须从 tokenizer 和数据配置读取，不能把 0 写死为普遍规则；decoder-only tokenizer 甚至可能没有独立的 padding token。

### 2.1.5 位置表示为什么必不可少

Token embedding 只编码“是什么”，不编码“在哪里”。如果把同一组 token 重新排序，单纯的 token 表示集合并不会自动告诉 attention 这些 token 的顺序：

```text
我 喜欢 你
你 喜欢 我
```

自注意力如果没有任何位置机制，对输入 token 的排列缺少足够的顺序信息。因此常见做法是把 token 表示与位置表示组合：

```math
H_{b,t}=E_{x_{b,t}}+P_t
```

其中 \(x_{b,t}\) 是第 \(b\) 个样本第 \(t\) 个 token 的 id，\(P_t\) 是位置 \(t\) 的向量。这个公式描述的是 absolute position embedding；RoPE 会在 attention 的 q、k 上注入位置信息，接入位置不同，不能把两者简单叠加。

可学习位置表为：

```math
P\in\mathbb{R}^{T_{\max}\times D}
```

它有 \(T_{\max}D\) 个位置参数，超过 \(T_{\max}\) 时不能直接索引。position_offset 可以让增量推理从历史长度开始取位置，但不能突破位置表的最大长度。

### 2.1.6 Learned positional embedding 的实现

```python
import torch
from torch import nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_token_id=None):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model and max_seq_len must be positive")
        kwargs = {}
        if pad_token_id is not None:
            kwargs["padding_idx"] = pad_token_id
        self.token = nn.Embedding(vocab_size, d_model, **kwargs)
        self.position = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, position_offset=0):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        batch_size, seq_len = input_ids.shape
        end = position_offset + seq_len
        if end > self.max_seq_len:
            raise ValueError("learned position table is too short")
        positions = torch.arange(
            position_offset,
            end,
            device=input_ids.device,
            dtype=torch.long,
        )
        positions = positions.view(1, seq_len).expand(batch_size, seq_len)
        return self.token(input_ids) + self.position(positions)

torch.manual_seed(42)
input_ids = torch.tensor([[4, 8, 2, 0], [9, 3, 0, 0]], dtype=torch.long)
module = TokenAndPositionEmbedding(
    vocab_size=20,
    d_model=8,
    max_seq_len=16,
    pad_token_id=0,
)
hidden = module(input_ids)
offset_hidden = module(input_ids[:, :2], position_offset=4)

print("hidden_shape=", tuple(hidden.shape))
print("offset_hidden_shape=", tuple(offset_hidden.shape))
print("position_table_shape=", tuple(module.position.weight.shape))
```

输出形状为：

```text
hidden_shape= (2, 4, 8)
offset_hidden_shape= (2, 2, 8)
position_table_shape= (16, 8)
```

这里 offset 只改变位置索引；token embedding 的 padding 行仍由 padding_idx 管理。若采用左 padding，还要决定真实 token 的 position ids 是否跳过 padding；不同模型和 serving runtime 的约定并不一致，不能只把 torch.arange(seq_len) 当成普适答案。

### 2.1.7 Sinusoidal encoding 是固定函数

Transformer 原论文给出了一种不增加可训练位置参数的编码。对位置 \(p\)、维度索引 \(i\) 和模型维度 \(D\)：

```math
PE_{p,2i}=\sin\left(\frac{p}{10000^{2i/D}}\right)
```

```math
PE_{p,2i+1}=\cos\left(\frac{p}{10000^{2i/D}}\right)
```

它可以在运行时计算更长的位置，但“可计算”不等于“模型在更长位置上保持同样质量”。频率范围、训练长度、数值精度和任务分布都会影响外推。

```python
import math
import torch

def sinusoidal_position_encoding(seq_len, d_model, device=None, dtype=torch.float32):
    if seq_len < 0 or d_model <= 0:
        raise ValueError("seq_len must be non-negative and d_model positive")
    if d_model % 2 != 0:
        raise ValueError("this implementation expects an even d_model")
    positions = torch.arange(
        seq_len, device=device, dtype=torch.float32
    ).view(seq_len, 1)
    pair_index = torch.arange(
        0, d_model, 2, device=device, dtype=torch.float32
    )
    inv_frequency = torch.exp(
        -math.log(10000.0) * pair_index / d_model
    )
    angles = positions * inv_frequency.view(1, -1)
    encoding = torch.zeros(
        seq_len, d_model, device=device, dtype=torch.float32
    )
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles)
    return encoding.to(dtype=dtype)

encoding = sinusoidal_position_encoding(seq_len=4, d_model=8)
print("encoding_shape=", tuple(encoding.shape))
print("position_zero=", encoding[0].tolist())
```

位置 0 的偶数维为 0，奇数维为 1，这是由该公式直接得到的，不是训练结果。固定 encoding 与 token states 相加后，整体 shape 仍是 \([B,T,D]\)。

### 2.1.8 输出头与 token embedding 权重共享

语言模型最终需要把 hidden state 映射回词表 logits。设 \(H\) 的形状为 \([B,T,D]\)，输出矩阵为 \(W_{\mathrm{out}}\in\mathbb{R}^{D\times V}\)：

```math
Z=HW_{\mathrm{out}}+b
```

权重共享（weight tying）常令：

```math
W_{\mathrm{out}}=E^\top
```

这样输入 embedding 和输出分类空间使用同一组参数，减少一份 \(VD\) 参数，并施加一种表示空间约束。它不是 nn.Embedding 的默认行为；共享时必须让两个计算路径真正引用同一个 Parameter，而不是复制数值后各自注册。

```python
import torch
from torch import nn

vocab_size = 100
d_model = 16
embedding = nn.Embedding(vocab_size, d_model)
output_bias = nn.Parameter(torch.zeros(vocab_size))
hidden = torch.randn(2, 4, d_model)
logits = hidden @ embedding.weight.transpose(0, 1) + output_bias

print("logits_shape=", tuple(logits.shape))
print("tied_parameter_count=", embedding.weight.numel() + output_bias.numel())
```

共享输入输出权重会让输入侧和输出侧梯度相加到同一个参数上；这也是它与“初始化相同的两份矩阵”在训练语义上的区别。是否共享取决于模型架构和 tokenizer 设计，不能由参数量估算反推。

### 2.1.9 左右 padding 与增量位置

右 padding 常见形式是：

```text
真实 token: [12, 35, 98, 77, 0, 0]
```

左 padding 则是：

```text
padding:    [0, 0, 12, 35, 98, 77]
```

对 absolute position embedding，左 padding 会带来一个选择：真实 token 使用物理列位置 2、3、4、5，还是重新编号为 0、1、2、3。对 causal mask、position ids 和 KV Cache 来说，这个选择必须一致；否则模型看到的“第一个真实 token”在不同 batch 中可能获得不同位置。

至少要把五件事写成同一个数据契约：

1. padding 是左侧还是右侧；
2. position ids 是否跳过 padding；
3. attention mask 如何屏蔽 padding key 和无效 query；
4. loss mask 是否保留有效目标；
5. 增量解码时新 token 的 cache position 如何计算。

在没有确认目标模型训练和 runtime 约定前，不能因为某个实现使用 arange 就把它推广为所有 decoder-only 模型的规则。

### 2.1.10 Embedding 输入的最小审计

对输入模块，最有价值的检查不是只看程序是否返回一个 Tensor，而是检查以下不变量：

```python
import torch
from torch import nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, pad_token_id=0):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model and max_seq_len must be positive")
        self.token = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        self.position = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, position_offset=0):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer dtype")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        batch_size, seq_len = input_ids.shape
        if position_offset + seq_len > self.max_seq_len:
            raise ValueError("position range is outside the table")
        position_ids = torch.arange(
            position_offset,
            position_offset + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        return self.token(input_ids) + self.position(position_ids)

torch.manual_seed(11)
input_ids = torch.tensor([[5, 8, 0, 0], [3, 7, 4, 2]])
module = InputEmbedding(32, 12, 32, pad_token_id=0)
hidden = module(input_ids)
hidden.square().mean().backward()

assert hidden.shape == (2, 4, 12)
assert module.token.weight.grad[0].abs().sum().item() == 0.0
assert module.position.weight.grad is not None
print("embedding_contract_ok=True")
```

```text
embedding_contract_ok=True
```

这个断言只证明 embedding 层的两个局部契约；它没有证明 attention mask 或 token loss mask 正确。系统验证必须把三个 mask 的测试分别写出来。

## 2.2 Scaled Dot-Product Attention：从相似度到信息汇聚

### 2.2.1 Q、K、V 是三个不同的角色

对输入表示做三组线性投影：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

Query 可以理解为当前位置要寻找的模式，Key 是每个位置可被匹配的索引，Value 是被选中后真正汇聚的内容。这些是理解用的角色描述，不是说网络内部存有可直接解释的“问题”和“答案”。

单头 self-attention 常用 shape 为：

| 张量 | shape | 含义 |
| --- | --- | --- |
| \(Q\) | \([B,T_q,d_k]\) | 每个 query 位置的向量 |
| \(K\) | \([B,T_k,d_k]\) | 每个 key 位置的向量 |
| \(V\) | \([B,T_k,d_v]\) | 每个 key 位置的内容向量 |

Self-attention 中 \(T_q=T_k=T\) 且三者都来自同一段 \(X\)；cross-attention 中 query 端和 key/value 端可以来自不同序列，因此 \(T_q\) 与 \(T_k\) 不必相同。

### 2.2.2 点积、缩放和 softmax

先计算 query 与每个 key 的相似度：

```math
S_{b,a,c}=\frac{Q_{b,a,:}K_{b,c,:}^{\top}}{\sqrt{d_k}}
```

这里 \(a\) 是 query 位置，\(c\) 是 key 位置。将所有位置写成矩阵：

```math
S=\frac{QK^\top}{\sqrt{d_k}}
```

得到 \(S\in\mathbb{R}^{B\times T_q\times T_k}\)。再对 key 维度做 softmax：

```math
A_{b,a,c}=\frac{\exp(S_{b,a,c})}{\sum_{r=1}^{T_k}\exp(S_{b,a,r})}
```

最后用注意力权重汇聚 Value：

```math
O_{b,a,:}=\sum_{c=1}^{T_k}A_{b,a,c}V_{b,c,:}
```

因此 \(A\) 的每个 query 行在没有 dropout 时和为 1，\(O\) 的形状是 \([B,T_q,d_v]\)。

为什么要除以 \(\sqrt{d_k}\)？如果每个 \(q_j,k_j\) 近似零均值、单位方差且相互独立，则点积的方差近似为 \(d_k\)：

```math
\mathrm{Var}(q^\top k)\approx d_k
```

缩放后：

```math
\mathrm{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)\approx 1
```

这不是所有训练状态下的精确统计定律，而是说明 softmax logits 随维度放大的原因。归一化层、初始化和相关性会改变真实分布，但缩放仍是标准 attention 定义的一部分。

### 2.2.3 手写 attention，并把 mask 语义写进函数

下面的实现把布尔 mask 定义为 keep mask：True 表示允许 query 读取该 key，False 表示屏蔽。函数只处理 shape 可广播到 scores 的 mask；它不替调用者验证 query 是否至少有一个可见 key。

```python
import math
import torch

def scaled_dot_product_attention(q, k, v, keep_mask=None):
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same key dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key length")

    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])

    if keep_mask is not None:
        keep_mask = keep_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(
            ~keep_mask,
            torch.finfo(scores.dtype).min,
        )

    weights = torch.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights, scores

torch.manual_seed(7)
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 6)
output, weights, scores = scaled_dot_product_attention(q, k, v)

assert output.shape == (2, 4, 6)
assert weights.shape == (2, 4, 4)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(2, 4),
    atol=1e-6,
)
print("output_shape=", tuple(output.shape))
print("weights_shape=", tuple(weights.shape))
```

这个例子只使用 self-attention 的 \(T_q=T_k=4\)，但函数本身也支持 \(T_q\ne T_k\)。注意 torch.softmax 的最后一维是 key 维；如果在 query 维归一化，权重就不再表示“一个 query 如何分配对各 key 的注意力”。

### 2.2.4 mask 应在 softmax 前应用

设一个位置不可见。将其 score 设为很小的有限数 \(m_{\mathrm{min}}\)，再做 softmax：

```math
A_{b,a,c}\approx0\quad (M_{b,a,c}=0)
```

教学实现常使用 torch.finfo(scores.dtype).min；某些实现使用负无穷或 additive mask。选择哪一个要看 dtype 和 kernel 的行为，但所有方案都要面对同一个边界：如果某个 query 的所有 key 都被屏蔽，归一化没有合法的分母，结果可能是 NaN 或无意义的均匀分布。

不要在 softmax 之后简单把权重置零来代替 mask。这样会破坏可见位置的归一化，除非你随后重新归一化，而且重新归一化又要处理全零行。更稳妥的契约是：在进入 softmax 前保证每个有效 query 至少有一个可见 key。

### 2.2.5 与 PyTorch SDPA 的边界

PyTorch 提供 torch.nn.functional.scaled_dot_product_attention。它可能选择普通、memory-efficient 或 Flash 风格的后端，具体 kernel 取决于设备、dtype、输入形状和版本。数学目标仍是缩放点积注意力，但中间张量是否显式物化不能从 API 名字推断。

布尔 mask 的语义必须核对目标接口：SDPA 的布尔 attn_mask 中 True 表示参与 attention，False 表示屏蔽；float mask 则作为 additive bias 加到 attention score 上。手写函数若采用同样的 keep 语义，可以做数值对照：

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, keep_mask=None):
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same key dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same key length")
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    if keep_mask is not None:
        keep_mask = keep_mask.to(
            device=scores.device,
            dtype=torch.bool,
        )
        scores = scores.masked_fill(
            ~keep_mask,
            torch.finfo(scores.dtype).min,
        )
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights, scores

torch.manual_seed(13)
q = torch.randn(2, 3, 5, 8)
k = torch.randn(2, 3, 5, 8)
v = torch.randn(2, 3, 5, 8)
keep_mask = torch.tril(
    torch.ones(5, 5, dtype=torch.bool)
).view(1, 1, 5, 5)

manual, _, _ = scaled_dot_product_attention(q, k, v, keep_mask)
official = F.scaled_dot_product_attention(
    q, k, v, attn_mask=keep_mask, dropout_p=0.0
)
print("shape=", tuple(official.shape))
print("manual_matches_official=", torch.allclose(
    manual, official, atol=1e-6
))
```

这段代码把对照所需的手写函数一并放入围栏，因此可以单独复制运行。两种实现都使用相同的 keep-mask 语义；差异只应来自数值误差和 PyTorch 选择的 attention kernel。

## 2.3 Multi-Head Attention：把一个相似度空间拆成多个子空间

### 2.3.1 从单头到多头

单头 attention 只有一个 \(d_k\) 维的相似度空间。多头 attention 将模型维度 \(D\) 分成 \(H\) 个 head，每个 head 使用 \(d_h=D/H\) 个通道独立计算，再把结果拼回去：

```math
D=H d_h
```

第 \(h\) 个 head 可以写成：

```math
\mathrm{head}_h=
\mathrm{Attention}(QW_h^Q,KW_h^K,VW_h^V)
```

所有 head 拼接后再投影：

```math
\mathrm{MHA}(X)=
\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W^O
```

不同 head 可以在不同参数子空间中学习局部关系、长距离关系或其他统计模式。这是结构上的可能性，不代表每个 head 都能被人类稳定地赋予一个语义标签；是否出现可解释的 head，需要通过激活、消融和下游任务实验验证。

标准实现通常用三个 \(D\to D\) 的线性层一次性产生 Q、K、V，再重新组织最后一维，而不是为每个 head 单独创建 Python 模块。只要 \(D\) 固定，改变 \(H\) 会改变每个 head 的维度和计算组织，但不会自动把四个投影矩阵的参数量乘以 \(H\)。

### 2.3.2 shape 账本

以 self-attention 为例：

| 张量 | shape | 含义 |
| --- | --- | --- |
| \(X\) | \([B,T,D]\) | 输入 hidden states |
| \(Q,K,V\) | \([B,T,D]\) | 三组线性投影 |
| 拆头后的 \(Q,K,V\) | \([B,H,T,d_h]\) | 每个 head 的序列表示 |
| scores | \([B,H,T,T]\) | query-key 分数 |
| head output | \([B,H,T,d_h]\) | 每个 head 汇聚后的值 |
| 合并后的 output | \([B,T,D]\) | 拼接并输出投影前 |

拆头是：

```math
[B,T,D]\to[B,T,H,d_h]\to[B,H,T,d_h]
```

合并是：

```math
[B,H,T,d_h]\to[B,T,H,d_h]\to[B,T,D]
```

transpose 通常只改变 stride，不保证新视图在物理内存中连续。合并前调用 contiguous 再 view，或者直接使用 reshape，是为了明确处理这一点。

### 2.3.3 一个可读的 MultiHeadAttention 实现

```python
import math
import torch
import torch.nn.functional as F
from torch import nn

def scaled_dot_product_attention(
    q,
    k,
    v,
    keep_mask=None,
    dropout_p=0.0,
    training=True,
):
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(q.shape[-1])
    if keep_mask is not None:
        keep_mask = keep_mask.to(dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(
            ~keep_mask,
            torch.finfo(scores.dtype).min,
        )
    weights = torch.softmax(scores, dim=-1)
    weights = F.dropout(
        weights,
        p=dropout_p,
        training=training,
    )
    return weights @ v, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        return x.transpose(1, 2)

    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x, keep_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        output, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            keep_mask=keep_mask,
            dropout_p=self.dropout.p,
            training=self.training,
        )
        output = self.merge_heads(output)
        output = self.out_proj(output)
        return output, weights

torch.manual_seed(42)
x = torch.randn(2, 5, 12)
mask = torch.tril(
    torch.ones(5, 5, dtype=torch.bool)
).view(1, 1, 5, 5)
mha = MultiHeadAttention(d_model=12, num_heads=3)
output, weights = mha(x, keep_mask=mask)

print("output_shape=", tuple(output.shape))
print("weights_shape=", tuple(weights.shape))
print("row_sum_shape=", tuple(weights.sum(dim=-1).shape))
```

输出形状为：

```text
output_shape= (2, 5, 12)
weights_shape= (2, 3, 5, 5)
row_sum_shape= (2, 3, 5)
```

在 dropout 为 0 且每个 query 至少有一个可见 key 时，weights 沿最后一维应接近全 1。若在训练模式下对 weights 使用 dropout，dropout 后的权重行不一定仍然和为 1；检查 mask 时应关闭 dropout，避免把正则化行为误判成 mask 错误。

### 2.3.4 self-attention 与 cross-attention

本节代码的 q、k、v 都来自同一个 x，因此是 self-attention。cross-attention 的接口应允许 query 序列与 key/value 序列长度不同：

```math
Q\in\mathbb{R}^{B\times T_q\times d_k},\qquad
K\in\mathbb{R}^{B\times T_k\times d_k},\qquad
V\in\mathbb{R}^{B\times T_k\times d_v}
```

其分数形状为 \([B,H,T_q,T_k]\)，而不是强行写成 \([B,H,T,T]\)。decoder 生成时，query 可以来自 decoder 当前 hidden，key/value 可以来自 encoder 输出；多模态模型中，key/value 也可能来自视觉或音频编码器。

如果只把 self-attention 类的 seq_len 同时传给 q 和 kv，cross-attention 在 \(T_q\ne T_k\) 时就会出现 reshape 或 mask 对齐错误。工程接口应把 query length 和 key length 分开记录。

### 2.3.5 参数量与复杂度

标准 MHA 有四个主要投影矩阵：

```math
W_Q,W_K,W_V,W_O\in\mathbb{R}^{D\times D}
```

忽略 bias 时：

```math
N_{\mathrm{MHA}}\approx4D^2
```

如果 \(D=512,H=8\)，改成 \(H=16\) 后，四个矩阵的形状仍然是 \([512,512]\)，参数量不会自动翻倍；head_dim 从 64 变成 32，可能改变 kernel 效率和每个子空间的表达容量。

attention 分数矩阵的计算复杂度近似为：

```math
O(BHT^2d_h)=O(BT^2D)
```

Q、K、V 和输出投影的线性层还带来：

```math
O(BTD^2)
```

朴素实现通常显式保存 \([B,H,T,T]\) 的权重或中间分数；FlashAttention 一类 kernel 可以减少中间显存和读写，但不应被描述为自动消除了标准 attention 的二次计算项。

### 2.3.6 针对实现的局部测试

下面的测试包含一个紧凑的 `MultiHeadAttention` 实现，因此可以独立运行。它保留 2.3.3 的 shape、keep-mask 和输出契约，省略 dropout 以便测试结果确定。

```python
import math
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, keep_mask):
        batch_size, seq_len, _ = x.shape
        def split_heads(values):
            values = values.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            )
            return values.transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~keep_mask.to(device=x.device, dtype=torch.bool),
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), weights

torch.manual_seed(5)
mha = MultiHeadAttention(d_model=16, num_heads=4)
x = torch.randn(2, 6, 16)
mask = torch.tril(
    torch.ones(6, 6, dtype=torch.bool)
).view(1, 1, 6, 6)
output, weights = mha(x, keep_mask=mask)

future = torch.triu(
    torch.ones(6, 6, dtype=torch.bool),
    diagonal=1,
).view(1, 1, 6, 6).expand_as(weights)

assert output.shape == x.shape
assert weights.shape == (2, 4, 6, 6)
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones_like(weights.sum(dim=-1)),
)
assert weights.masked_select(future).abs().max().item() < 1e-6
print("mha_contract_ok=True")
```

这个测试没有证明模型训练有效，但能把 shape、归一化、因果方向和输出接口四个问题从端到端 loss 中隔离出来。若某一项失败，先修复局部契约，再讨论模型质量。

## 2.4 Attention Mask：把可见性写成明确的协议

### 2.4.1 causal mask 与 padding mask 是两种约束

Causal mask 解决时间方向的信息泄漏。令 \(i\) 是 query 位置，\(j\) 是 key 位置，keep mask 为：

```math
M^{\mathrm{causal}}_{i,j}=
\begin{cases}
1,&j\le i\\
0,&j>i
\end{cases}
```

它是下三角矩阵。位置 \(i\) 可以读取自己和历史 key，不能读取未来 key。

Padding mask 解决 batch 对齐产生的无效 key。设 \(p_{b,j}=1\) 表示第 \(b\) 个样本的 key 位置 \(j\) 是有效 token：

```math
M^{\mathrm{pad}}_{b,j}=p_{b,j}
```

为了广播到 scores，通常把它变成 \([B,1,1,T_k]\)。两个 keep mask 组合时使用逻辑与：

```math
M^{\mathrm{combined}}_{b,h,i,j}
=
M^{\mathrm{causal}}_{i,j}\land M^{\mathrm{pad}}_{b,j}
```

这只约束 key 维度。padding query 仍然可能对有效 key 产生输出；通常由 loss mask 忽略这些 query 的训练目标，或者在下游把无效位置清理掉。不要把“不能读取 padding key”和“padding query 不运行”混成一个机制。

### 2.4.2 从 mask 形状推导广播

当 scores 形状为 \([B,H,T_q,T_k]\)：

| mask | 形状 | 作用 |
| --- | --- | --- |
| causal keep mask | \([1,1,T_q,T_k]\) | 所有样本、所有 head 共用 |
| padding key mask | \([B,1,1,T_k]\) | 每个样本屏蔽自己的 padding key |
| 局部 query-key mask | \([B,1,T_q,T_k]\) | 每个样本独立约束 |
| per-head mask | \([B,H,T_q,T_k]\) | 每个 head 单独约束 |

只要广播规则成立，mask 不必物理复制到每个 batch 和 head。过早 expand 可能制造很大的逻辑视图或中间张量；优先保留最小广播形状，交给 kernel 处理。

```python
import torch

batch_size = 2
num_heads = 4
query_len = 5
key_len = 5

causal = torch.tril(
    torch.ones(query_len, key_len, dtype=torch.bool)
).view(1, 1, query_len, key_len)

tokens = torch.tensor([
    [12, 35, 98, 0, 0],
    [21, 43, 77, 64, 0],
])
padding = (tokens != 0).view(batch_size, 1, 1, key_len)
combined = causal & padding

scores = torch.randn(batch_size, num_heads, query_len, key_len)
masked_scores = scores.masked_fill(
    ~combined,
    torch.finfo(scores.dtype).min,
)

print("scores_shape=", tuple(scores.shape))
print("causal_shape=", tuple(causal.shape))
print("padding_shape=", tuple(padding.shape))
print("combined_shape=", tuple(combined.shape))
print("masked_shape=", tuple(masked_scores.shape))
```

输出的 shape 依次是 \([2,4,5,5]\)、\([1,1,5,5]\)、\([2,1,1,5]\)、\([2,1,5,5]\)、\([2,4,5,5]\)。

### 2.4.3 keep mask、block mask 和 additive mask

同一个“不可见位置”可以用三种常见协议表达：

| 形式 | 可见位置 | 屏蔽位置 | softmax 前的处理 |
| --- | --- | --- | --- |
| keep bool | True | False | 对 False 填最小值 |
| block bool | False | True | 对 True 填最小值 |
| additive float | 0 | 很小负数 | 直接加到 score |

最危险的不是选择哪一种，而是把一种语义传给期待另一种语义的接口。例如手写函数若使用反向 keep mask 填值，而调用者传入的是 block mask，结果会把可见位置全部屏蔽。

建议在函数名或变量名中带上语义，例如 keep_mask、block_mask、additive_bias，不要只写 mask。在跨模块传递时，把 shape 和 True 的含义写进注释或数据结构。

### 2.4.4 PyTorch 不同接口的布尔语义

PyTorch 的不同 attention 接口并不自动共享布尔 mask 约定：

| 接口 | 参数 | bool True 的含义 |
| --- | --- | --- |
| functional SDPA | attn_mask | 允许参与 attention |
| nn.MultiheadAttention | attn_mask | 禁止该 query-key 位置 |
| nn.MultiheadAttention | key_padding_mask | 该 key 是 padding，忽略 |

因此，一个 keep causal mask：

```python
import torch

keep_causal = torch.tril(
    torch.ones(4, 4, dtype=torch.bool)
)
block_causal = ~keep_causal
print("keep_true_count=", int(keep_causal.sum()))
print("block_true_count=", int(block_causal.sum()))
```

传给官方模块前还要核对 dtype、device 和 batch/head 广播规则。尤其不要因为两个接口都叫 attn_mask，就默认 True 的语义一致。

### 2.4.5 全 mask 行是一个独立异常状态

如果某个 query 的所有 key 都被屏蔽，softmax 没有合法的概率分布。可能的结果包括 NaN、全零或由特定 kernel 产生的其他边界值。常见原因是 padding 与 causal mask 组合错误、首个有效 query 被屏蔽、key length 为零，或外部过滤器把所有候选都删掉。

```python
import torch

keep_mask = torch.tensor([
    [[
        [True, False, False],
        [True, True, False],
        [False, False, False],
    ]]
])
visible_count = keep_mask.sum(dim=-1)
bad_rows = visible_count == 0
print("bad_rows=", bad_rows.tolist())
```

系统应先决定空候选的语义：跳过该 query、返回显式错误、使用安全 sentinel key，还是让上层重新规划。把全 mask 行默默交给 softmax，会把数据错误变成 NaN，增加归因难度。

### 2.4.6 在 softmax 前应用 mask 的完整函数

```python
import torch

def apply_keep_mask(scores, keep_mask):
    keep_mask = keep_mask.to(
        device=scores.device,
        dtype=torch.bool,
    )
    try:
        keep_mask = torch.broadcast_to(keep_mask, scores.shape)
    except RuntimeError as exc:
        raise ValueError("query/key mask shape is not broadcastable") from exc
    if (keep_mask.sum(dim=-1) == 0).any():
        raise ValueError("every query must keep at least one key")
    return scores.masked_fill(
        ~keep_mask,
        torch.finfo(scores.dtype).min,
    )

scores = torch.randn(1, 2, 3, 3)
mask = torch.tensor(
    [[
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]]
).view(1, 1, 3, 3)
masked = apply_keep_mask(scores, mask)
weights = torch.softmax(masked, dim=-1)
print("masked_shape=", tuple(masked.shape))
print("row_sums=", weights.sum(dim=-1).tolist())
```

这里显式广播后再检查每个 query 是否有可见 key，因此既支持
`[1,1,T_q,T_k]` 的 causal mask，也支持 `[B,1,1,T_k]` 的 padding-key mask。
这个检查只确认每个 query 有一个 key；它不确认该 key 是语义上正确的 key。mask 的正确性仍需用确定性样例验证。

### 2.4.7 mask 的测试矩阵

一个可复用的单元测试至少覆盖：

1. 未加 mask 时，权重每行归一化；
2. causal mask 后，所有未来位置权重为零；
3. padding mask 后，所有 padding key 列权重为零；
4. 两种 mask 组合后，既不看未来也不看 padding；
5. keep mask 转 additive bias 后，输出在允许的数值误差内一致；
6. 全 mask 行抛出明确错误，而不是产生静默 NaN；
7. CPU 与目标 GPU 上的 dtype、device 和广播行为一致；
8. 接入官方模块时，True 的语义通过短序列测试确认。

测试应关注权重和状态，而不只比较一个最终 logits。最终 logits 可能因为初始化、dropout 或低精度差异而不同，但未来位置出现非零权重是明确的协议失败。

## 2.5 Transformer Block：残差、归一化与逐 token 的非线性

### 2.5.1 Attention 与 FFN 的分工

Attention 做 token mixing：当前位置从其他位置读取信息。FFN 做 channel mixing：对每个位置的 hidden 维度独立进行非线性变换。一个 block 的输入输出通常保持相同：

```math
X\in\mathbb{R}^{B\times T\times D}
\longrightarrow
Y\in\mathbb{R}^{B\times T\times D}
```

输入输出 shape 相同，才可以堆叠多个 block。若 attention 输出、FFN 输出或残差分支的维度不一致，应在模块边界立即报错，不要等到几十层之后才从 loss 异常倒推。

### 2.5.2 FFN 的数学形态

普通两层 FFN 可以写成：

```math
\mathrm{FFN}(X)=
\phi(XW_1+b_1)W_2+b_2
```

若中间维度为 \(D_{\mathrm{ff}}\)，则：

```math
W_1\in\mathbb{R}^{D\times D_{\mathrm{ff}}},
\qquad
W_2\in\mathbb{R}^{D_{\mathrm{ff}}\times D}
```

每个 token 使用同一组 \(W_1,W_2\)，但 token 之间不在 FFN 内直接交互。GELU、SiLU 或 SwiGLU 是不同的非线性和门控选择，不改变 FFN 作为逐位置变换的基本角色。

### 2.5.3 Pre-LN 与 Post-LN

Post-LN 的一种写法是：

```math
Y=\mathrm{LN}\left(X+\mathrm{Attn}(X)\right)
```

```math
Z=\mathrm{LN}\left(Y+\mathrm{FFN}(Y)\right)
```

Pre-LN 则是：

```math
Y=X+\mathrm{Attn}(\mathrm{LN}(X))
```

```math
Z=Y+\mathrm{FFN}(\mathrm{LN}(Y))
```

两者都保留残差，但归一化所在位置不同。深层 decoder-only 模型经常采用 Pre-LN 或其变体，因为残差主干上存在更直接的梯度路径；这是一种常见架构选择，不是对所有深度、初始化和训练方案的无条件保证。比较两者时，应固定参数量、学习率和训练预算，并观察梯度范数、激活范围和验证损失。

### 2.5.4 LayerNorm、Dropout 与训练模式

LayerNorm 通常沿最后一个 hidden 维度归一化。对一个 token 向量 \(x\in\mathbb{R}^{D}\)：

```math
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j,\qquad
\sigma^2=\frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2
```

```math
\mathrm{LN}(x)_j=
\gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_j
```

它不跨 batch 估计统计量，因此输入可以是 \([B,T,D]\)，每个 token 的最后一维独立处理。Dropout 在训练模式下随机置零并做尺度补偿，在评估模式下通常表现为恒等映射。检查 attention mask 或残差等价性时要调用 eval 或将 dropout 设为 0，否则随机性会污染对照。

### 2.5.5 一个 Pre-LN decoder block

```python
import math
import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class BlockAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keep_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        if keep_mask is not None:
            scores = scores.masked_fill(
                ~keep_mask.to(dtype=torch.bool, device=x.device),
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.d_model)
        return self.out_proj(out), weights

class PreLNBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        if d_model <= 0 or mlp_ratio <= 0:
            raise ValueError("d_model and mlp_ratio must be positive")
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BlockAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * mlp_ratio, dropout)

    def forward(self, x, keep_mask=None, return_weights=False):
        attn_out, weights = self.attn(self.ln1(x), keep_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        if return_weights:
            return x, weights
        return x

torch.manual_seed(42)
x = torch.randn(2, 5, 16)
mask = torch.tril(
    torch.ones(5, 5, dtype=torch.bool)
).view(1, 1, 5, 5)
block = PreLNBlock(16, 4, mlp_ratio=4, dropout=0.0)
block.eval()
y, weights = block(x, keep_mask=mask, return_weights=True)

assert y.shape == x.shape
assert weights.shape == (2, 4, 5, 5)
print("block_output_shape=", tuple(y.shape))
print("block_weights_shape=", tuple(weights.shape))
```

这个 block 尚未包含 RoPE、KV Cache 或 RMSNorm；组合新组件时要逐层替换，并保留这些 shape 断言。

### 2.5.6 残差连接的梯度路径

残差更新 \(Y=X+F(X)\) 对输入的 Jacobian 为：

```math
\frac{\partial Y}{\partial X}
=I+\frac{\partial F(X)}{\partial X}
```

其中 \(I\) 是恒等映射。即使子层的局部 Jacobian 在某些方向上很小，恒等项也提供了一条直接路径；这解释了残差为什么有助于深层优化，但不意味着任何残差网络都不会梯度消失或爆炸。初始化、归一化、学习率和激活仍然会影响第二项。

残差还让每个子层更像在当前表示上学习增量。若要研究某个子层是否真正贡献信息，可以比较完整 block 与只保留残差路径的消融，而不是只看某一层的激活均值。

### 2.5.7 参数量与计算量

标准 self-attention 的四个投影忽略 bias 约为：

```math
N_{\mathrm{attn}}\approx4D^2
```

普通 FFN 的主要参数量约为：

```math
N_{\mathrm{ffn}}\approx2D D_{\mathrm{ff}}
```

当 \(D_{\mathrm{ff}}=4D\)：

```math
N_{\mathrm{ffn}}\approx8D^2
```

因此一个标准 block 的主参数量约为：

```math
N_{\mathrm{block}}\approx12D^2
```

这只是估算，不包括 bias、norm、嵌入、输出头和门控变体。计算复杂度还要区分序列长度 \(T\) 与 hidden size \(D\)：

```math
C_{\mathrm{attn}}\approx O(BT^2D)+O(BTD^2)
```

```math
C_{\mathrm{ffn}}\approx O(BTDD_{\mathrm{ff}})
```

长序列主要放大 \(T^2\) 项；较大 hidden size 和 FFN 扩展比例则放大线性投影。显存瓶颈还受到是否物化 attention matrix、激活保存、KV Cache 和并行策略影响。

### 2.5.8 Block 的诊断顺序

当一个 block 的 loss、梯度或输出异常时，可以按数据路径检查：

1. 输入是否真的是 \([B,T,D]\)；
2. q/k/v 拆头后是否为 \([B,H,T,d_h]\)；
3. mask 是否能广播到 \([B,H,T,T]\)；
4. 每个有效 query 是否至少有一个可见 key；
5. norm 是否沿最后一维工作；
6. residual 相加前两项 shape 是否相同；
7. 训练和评估模式是否符合当前测试；
8. 参数量是否与配置中的 \(D,H,D_{\mathrm{ff}}\) 相符。

这套顺序先检查局部契约，再检查数值范围，最后才比较训练曲线。它能把“Transformer 不收敛”拆成可定位的问题。

## 2.6 RoPE：把位置关系写进 q、k 的几何变换

### 2.6.1 为什么 self-attention 需要位置机制

纯 self-attention 对输入集合的排列缺少天然的顺序偏好。位置机制可以直接把位置向量加到 hidden states，也可以改变 q、k 的相似度几何。RoPE 属于后者：它不把一张位置表简单相加，而是对每个 head 的 q、k 进行按位置的二维旋转。

### 2.6.2 二维旋转和偶奇维配对

对二维向量 \((x_0,x_1)\)，角度为 \(\theta\) 的旋转是：

```math
\begin{bmatrix}
x'_0\\
x'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
\begin{bmatrix}
x_0\\
x_1
\end{bmatrix}
```

即：

```math
x'_0=x_0\cos\theta-x_1\sin\theta,\qquad
x'_1=x_0\sin\theta+x_1\cos\theta
```

RoPE 把 head_dim 的维度按 \((0,1),(2,3),\ldots\) 两两分组。对位置 \(p\) 和第 \(i\) 个二维组，常见频率为：

```math
\omega_i=\mathrm{base}^{-\frac{2i}{d_h}},\qquad
\theta_{p,i}=p\omega_i
```

因此 cos/sin 缓存的形状是 \([T,d_h/2]\)，应用到 q、k 后 shape 不变。

### 2.6.3 位置旋转如何产生相对距离

设 \(R_p\) 是位置 \(p\) 对每个二维平面施加的旋转。旋转矩阵满足正交性和复合关系：

```math
R_p^\top R_q=R_{q-p}
```

于是：

```math
(R_p q)^\top(R_q k)
=q^\top R_p^\top R_q k
=q^\top R_{q-p}k
```

内积中出现了 \(q-p\)，也就是两个位置的相对距离。这个推导说明 RoPE 具有相对位置结构；它不等于证明模型在任意长度、任意任务上都会保持性能，训练长度和频率外推仍然需要实验。

### 2.6.4 cos/sin cache 与 offset

```python
import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if max_seq_len <= 0 or base <= 0:
            raise ValueError("max_seq_len and base must be positive")
        self.max_seq_len = max_seq_len
        pair_index = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_index / head_dim)
        positions = torch.arange(
            max_seq_len, dtype=torch.float32
        ).view(-1, 1)
        frequencies = positions * inv_freq.view(1, -1)
        self.register_buffer(
            "cos_cached",
            frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            frequencies.sin(),
            persistent=False,
        )

    def forward(self, seq_len, offset=0, device=None, dtype=None):
        if seq_len < 0 or offset < 0:
            raise ValueError("seq_len and offset must be non-negative")
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError("RoPE cache is too short")
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        if device is not None:
            cos = cos.to(device=device)
            sin = sin.to(device=device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

rope = RotaryEmbedding(head_dim=8, max_seq_len=32)
cos, sin = rope(seq_len=6, offset=4, dtype=torch.float32)
print("cos_shape=", tuple(cos.shape))
print("sin_shape=", tuple(sin.shape))
```

register_buffer 让缓存随模块迁移到设备，但不把它们当作可训练 Parameter。persistent=False 表示它们不进入 state_dict；恢复模型时需要用同样配置重新构造。是否采用 persistent=False 是 checkpoint 设计选择，不能据此断言所有实现都一样。

### 2.6.5 应用到 q 和 k，不应用到 v

假设 q、k 形状为 \([B,H,T,d_h]\)，cos、sin 为 \([T,d_h/2]\)。先把缓存扩成 \([1,1,T,d_h/2]\)，再对偶奇维旋转：

```python
import torch

def apply_rope(x, cos, sin):
    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError("x must have shape [B,H,T,D] with even D")
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("cos and sin must share shape [T,D/2]")
    if (x.shape[-2], x.shape[-1] // 2) != tuple(cos.shape):
        raise ValueError("RoPE cache shape does not match x")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos.to(device=x.device, dtype=x.dtype).view(
        1, 1, cos.shape[0], cos.shape[1]
    )
    sin = sin.to(device=x.device, dtype=x.dtype).view(
        1, 1, sin.shape[0], sin.shape[1]
    )
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    output = torch.empty_like(x)
    output[..., 0::2] = rotated_even
    output[..., 1::2] = rotated_odd
    return output

q = torch.randn(2, 4, 6, 8)
cos = torch.ones(6, 4)
sin = torch.zeros(6, 4)
rotated = apply_rope(q, cos, sin)
assert torch.equal(rotated, q)
print("rotated_shape=", tuple(rotated.shape))
```

在 sin 为 0、cos 为 1 的特例下，输出应与输入完全相同；这是一个简单但有价值的实现测试。RoPE 作用在 q、k，是因为位置要改变 query-key 匹配；v 表示被汇聚的内容，通常不做同样的旋转。具体架构若旋转 v，必须以该架构的定义为准，不能把常见实现写成数学必然性。

### 2.6.6 把 RoPE 接入 attention

下面的示例把 `RotaryEmbedding` 和 `apply_rope` 的最小定义一并放入围栏，可以单独运行。生产实现还应根据目标模型的旋转布局、缓存策略和 dtype 约定补充更严格的接口检查。

```python
import math
import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if max_seq_len <= 0 or base <= 0:
            raise ValueError("max_seq_len and base must be positive")
        self.max_seq_len = max_seq_len
        pair_index = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_index / head_dim)
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        ).view(-1, 1)
        frequencies = positions * inv_freq.view(1, -1)
        self.register_buffer(
            "cos_cached",
            frequencies.cos(),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            frequencies.sin(),
            persistent=False,
        )

    def forward(self, seq_len, offset=0, device=None, dtype=None):
        if seq_len < 0 or offset < 0:
            raise ValueError("seq_len and offset must be non-negative")
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError("RoPE cache is too short")
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        if device is not None:
            cos = cos.to(device=device)
            sin = sin.to(device=device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin

def apply_rope(x, cos, sin):
    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError("x must have shape [B,H,T,D] with even D")
    if cos.shape != sin.shape or cos.ndim != 2:
        raise ValueError("cos and sin must share shape [T,D/2]")
    if (x.shape[-2], x.shape[-1] // 2) != tuple(cos.shape):
        raise ValueError("RoPE cache shape does not match x")
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos.to(device=x.device, dtype=x.dtype).view(
        1, 1, cos.shape[0], cos.shape[1]
    )
    sin = sin.to(device=x.device, dtype=x.dtype).view(
        1, 1, sin.shape[0], sin.shape[1]
    )
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, keep_mask=None, position_offset=0):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rope(
            seq_len,
            offset=position_offset,
            device=x.device,
            dtype=q.dtype,
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        if keep_mask is not None:
            scores = scores.masked_fill(
                ~keep_mask.to(device=x.device, dtype=torch.bool),
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), weights

torch.manual_seed(42)
x = torch.randn(2, 6, 32)
mask = torch.tril(
    torch.ones(6, 6, dtype=torch.bool)
).view(1, 1, 6, 6)
attention = RoPEAttention(32, 4, max_seq_len=64)
output, weights = attention(x, keep_mask=mask)

assert output.shape == x.shape
assert weights.shape == (2, 4, 6, 6)
print("output_shape=", tuple(output.shape))
print("weights_shape=", tuple(weights.shape))
```

增量推理时，若 KV Cache 已包含 past_len 个位置，新 q/k 的 position_offset 应从 past_len 开始。offset 只改变 cos/sin 的切片，不改变 q/k 的局部 shape；同时 attention mask 的 key length 也要与“历史缓存加新 token”的长度一致。只给 RoPE 传 offset、却仍用长度为 1 的局部 mask，是另一个独立的 shape 错误。

### 2.6.7 位置布局和工程边界

不同代码库可能采用偶奇交错布局，也可能先把前半维和后半维配对。两种实现都可以正确，但 cos/sin 的排列和 rotate_half 必须匹配。将一个布局的缓存直接喂给另一个布局，会产生看似有限却语义错误的输出。

RoPE cache 需要覆盖实际位置。把训练长度从 4K 直接推到 32K，不等价于获得了 32K 的有效理解能力。RoPE scaling、频率变换、重新训练或长上下文微调可能改善某些任务，但应区分论文提出的方法、框架提供的参数和目标模型实测结果。不要把“可以生成 cos/sin 到更长位置”当成质量保证。

## 2.7 RMSNorm 与 SwiGLU：现代 decoder block 的两个改造

### 2.7.1 LayerNorm 和 RMSNorm 的数学差异

对单个 token 的 hidden 向量 \(x\in\mathbb{R}^{D}\)，LayerNorm 先求均值和方差：

```math
\mu=\frac{1}{D}\sum_{j=1}^{D}x_j,\qquad
\sigma^2=\frac{1}{D}\sum_{j=1}^{D}(x_j-\mu)^2
```

```math
\mathrm{LN}(x)_j=
\gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_j
```

RMSNorm 不做中心化，只计算 root mean square：

```math
r(x)=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}
```

```math
\mathrm{RMSNorm}(x)_j=
w_j\frac{x_j}{r(x)}
```

因此 LayerNorm 有 scale 和 bias 两组参数，RMSNorm 的常见实现只有一个 scale \(w\)。RMSNorm 的“更简单”不等价于“所有任务都更好”；它改变了归一化的对称性和数值路径，架构选择应通过训练稳定性、质量和吞吐实验判断。

### 2.7.2 低精度下的实现

归一化的平方、求均值和开方容易受到低精度舍入影响。常见工程实现先用 FP32 计算平方均值，再把结果转换回输入 dtype：

```python
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        input_dtype = x.dtype
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)
        normalized = normalized.to(dtype=input_dtype)
        return normalized * self.weight.to(dtype=input_dtype)

torch.manual_seed(42)
x = torch.randn(2, 3, 8)
norm = RMSNorm(8)
y = norm(x)
rms = torch.sqrt(y[0, 0].pow(2).mean())
print("output_shape=", tuple(y.shape))
print("first_token_rms=", round(rms.item(), 5))
```

如果 weight 初始为 1 且 eps 很小，输出 token 的 RMS 会接近 1；它不保证 token 的均值接近 0，这正是它与 LayerNorm 的一个可观察差别。

### 2.7.3 SwiGLU 的门控结构

普通 FFN 是一条激活路径：

```math
Y=W_2\phi(XW_1+b_1)+b_2
```

SwiGLU 使用两条上投影分支：

```math
G=\mathrm{SiLU}(XW_g),\qquad
U=XW_u
```

再逐元素门控并下投影：

```math
Y=(G\odot U)W_d
```

其中：

```math
\mathrm{SiLU}(z)=z\sigma(z)
```

\(G\) 可以理解为门控信号，\(U\) 是候选特征；这个解释帮助理解计算图，但不应把每个通道的数值直接解释成离散“开关”。门控是连续乘法，梯度也会同时经过两条路径。

### 2.7.4 参数量为什么常用约 \(8D/3\)

普通 \(4D\) FFN 的两层线性参数量忽略 bias 为：

```math
N_{\mathrm{FFN}}\approx D(4D)+(4D)D=8D^2
```

SwiGLU 有 gate、up、down 三个矩阵，若中间维度为 \(H_{\mathrm{ff}}\)：

```math
N_{\mathrm{SwiGLU}}\approx DH_{\mathrm{ff}}+DH_{\mathrm{ff}}+H_{\mathrm{ff}}D
=3DH_{\mathrm{ff}}
```

令两者近似相等：

```math
3DH_{\mathrm{ff}}\approx8D^2
\quad\Longrightarrow\quad
H_{\mathrm{ff}}\approx\frac{8D}{3}
```

真实配置还会把 \(H_{\mathrm{ff}}\) 向硬件友好的倍数取整，也可能加入 bias、张量并行切分或不同的扩展比例。这个公式用于参数账本，不是规定所有 LLM 必须使用同一个 hidden dimension。

### 2.7.5 SwiGLU 实现

```python
import torch
import torch.nn.functional as F
from torch import nn

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.dropout(self.down_proj(gate * value))

def swiglu_hidden_dim(d_model, multiple_of=256):
    if d_model <= 0 or multiple_of <= 0:
        raise ValueError("d_model and multiple_of must be positive")
    raw = int(8 * d_model / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)

torch.manual_seed(42)
x = torch.randn(2, 5, 12)
hidden_dim = swiglu_hidden_dim(12, multiple_of=8)
mlp = SwiGLU(12, hidden_dim)
y = mlp(x)

print("hidden_dim=", hidden_dim)
print("output_shape=", tuple(y.shape))
print("parameter_count=", sum(p.numel() for p in mlp.parameters()))
```

输入输出都保持 \([B,T,D]\)。当 \(D=12\)、multiple_of=8 时，向上取整后的 hidden_dim 是 32；不带 bias 的参数量是 \(3\times12\times32=1152\)。

### 2.7.6 现代 decoder block 的组合

Pre-RMSNorm、RoPE attention、SwiGLU 和 residual 可以写成：

```math
Y=X+\mathrm{Attn}(\mathrm{RMSNorm}(X))
```

```math
Z=Y+\mathrm{SwiGLU}(\mathrm{RMSNorm}(Y))
```

下面组合前面三个组件。为便于读者复制验证，围栏中包含这四个组件的最小实现；它们沿用前文已经解释过的公式，但示例不依赖前文代码块的执行状态：

```python
import math
import torch
import torch.nn.functional as F
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        if d_model <= 0 or eps <= 0:
            raise ValueError("d_model and eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_float = x.float()
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(mean_square + self.eps)
        return normalized.to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        self.max_seq_len = max_seq_len
        pair_index = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = base ** (-pair_index / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32).view(-1, 1)
        frequencies = positions * inv_freq.view(1, -1)
        self.register_buffer("cos_cached", frequencies.cos(), persistent=False)
        self.register_buffer("sin_cached", frequencies.sin(), persistent=False)

    def forward(self, seq_len, offset=0, device=None, dtype=None):
        end = offset + seq_len
        if seq_len < 0 or offset < 0 or end > self.max_seq_len:
            raise ValueError("RoPE position range is invalid")
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        if device is not None:
            cos, sin = cos.to(device=device), sin.to(device=device)
        if dtype is not None:
            cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
        return cos, sin

def apply_rope(x, cos, sin):
    if x.ndim != 4 or x.shape[-1] % 2 != 0:
        raise ValueError("x must have shape [B,H,T,D] with even D")
    if tuple(cos.shape) != (x.shape[-2], x.shape[-1] // 2):
        raise ValueError("RoPE cache shape does not match x")
    even, odd = x[..., 0::2], x[..., 1::2]
    cos = cos.to(device=x.device, dtype=x.dtype).view(1, 1, *cos.shape)
    sin = sin.to(device=x.device, dtype=x.dtype).view(1, 1, *sin.shape)
    output = torch.empty_like(x)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output

class RoPEAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        if d_model <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, keep_mask=None, position_offset=0):
        batch_size, seq_len, _ = x.shape
        def split_heads(values):
            return values.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        cos, sin = self.rope(
            seq_len,
            offset=position_offset,
            device=x.device,
            dtype=q.dtype,
        )
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if keep_mask is not None:
            scores = scores.masked_fill(
                ~keep_mask.to(device=x.device, dtype=torch.bool),
                torch.finfo(scores.dtype).min,
            )
        weights = torch.softmax(scores, dim=-1)
        output = (weights @ v).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(output), weights

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.dropout(self.down_proj(gate * value))

def swiglu_hidden_dim(d_model, multiple_of=256):
    if d_model <= 0 or multiple_of <= 0:
        raise ValueError("d_model and multiple_of must be positive")
    raw = int(8 * d_model / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)

class ModernDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, dropout=0.0):
        super().__init__()
        hidden_dim = swiglu_hidden_dim(d_model, multiple_of=8)
        self.attn_norm = RMSNorm(d_model)
        self.attn = RoPEAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, hidden_dim, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x, keep_mask=None, position_offset=0):
        attn_out, weights = self.attn(
            self.attn_norm(x),
            keep_mask=keep_mask,
            position_offset=position_offset,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, weights

torch.manual_seed(7)
x = torch.randn(2, 8, 48)
mask = torch.tril(
    torch.ones(8, 8, dtype=torch.bool)
).view(1, 1, 8, 8)
block = ModernDecoderBlock(48, 6, max_seq_len=64)
block.eval()
y, weights = block(x, keep_mask=mask)

assert y.shape == x.shape
assert weights.shape == (2, 6, 8, 8)
print("block_shape=", tuple(y.shape))
print("weights_shape=", tuple(weights.shape))
print("ffn_hidden_dim=", block.hidden_dim)
```

这个 block 仍然缺少完整模型的 token embedding、输出头、KV Cache 和训练 loss；它是结构实验，不应被称为一个可直接部署的语言模型。

### 2.7.7 组件级证据和整机级结论

到这里可以区分三种结论：

- 形状结论：输入 \([B,T,D]\) 是否能经过各模块并返回同形状输出；
- 数值结论：mask 后未来权重是否为零、RMSNorm 的尺度是否有限、两种实现是否在误差内一致；
- 能力结论：模型是否在真实语言任务上提升，需要训练集、验证集、长上下文和生成质量实验。

前两种可以用本章的小测试验证，第三种不能由某个 toy forward 得出。把 shape 断言通过写成“Transformer 已经理解语言”，是把证据等级偷换了。

### 2.7.8 一条完整的 shape 路径

对现代 decoder block，可以沿下面的路径追踪：

```text
input_ids                 [B,T]
token embedding           [B,T,D]
q/k/v projection          [B,T,D]
split heads               [B,H,T,d_h]
RoPE on q/k               [B,H,T,d_h]
attention scores          [B,H,T,T]
attention output          [B,H,T,d_h]
merge heads               [B,T,D]
RMSNorm + residual        [B,T,D]
SwiGLU hidden             [B,T,H_ff]
SwiGLU output             [B,T,D]
```

这个账本应该与代码中的每次 view、transpose、矩阵乘法和残差相加对应。遇到 shape error 时，从这条路径反向定位，通常比直接修改 reshape 参数更可靠。

### 2.7.9 组件回归实验的最小集合

一个小型 Transformer 组件库至少应保留以下回归测试：

1. embedding 输入为 long，输出为浮点且形状为 \([B,T,D]\)；
2. padding_idx 行在无额外正则的测试中梯度为零；
3. sinusoidal 位置 0 的偶数维为 0、奇数维为 1；
4. attention 权重沿 key 维归一化；
5. causal mask 的未来权重为零；
6. MHA 拆头后再合并能恢复形状；
7. RoPE 的单位旋转与输入相同；
8. RoPE offset 取到 cache 的对应切片；
9. RMSNorm 输出尺度有限，且不被误判为零均值；
10. SwiGLU 输出形状与输入相同；
11. block 的 residual 输出仍为 \([B,T,D]\)；
12. CPU、目标 GPU 和目标 dtype 的数值差异在预设容差内。

这些测试不能替代训练和评估，但能为后续实现 KV Cache、GQA、FlashAttention 和量化提供稳定的局部基线。

## 延伸阅读与资料边界

本章的 API 行为以目标版本的 PyTorch 官方文档为准，架构公式和历史背景以原始论文为准。复现实验应记录 PyTorch 版本、设备、dtype、是否启用 dropout、attention 后端以及 tokenizer 的 padding 配置。

1. Vaswani 等，Attention Is All You Need：<https://arxiv.org/abs/1706.03762>。Transformer、scaled dot-product attention、multi-head attention 和 sinusoidal encoding。
2. PyTorch nn.Embedding：<https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html>。查表、padding_idx 和索引输入。
3. PyTorch scaled_dot_product_attention：<https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>。SDPA、布尔/浮点 mask 和 causal 参数。
4. PyTorch MultiheadAttention：<https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>。官方 MHA 的输入、输出和 mask 语义。
5. Su 等，RoFormer: Enhanced Transformer with Rotary Position Embedding：<https://arxiv.org/abs/2104.09864>。RoPE 的旋转构造和相对位置性质。
6. Zhang 与 Sennrich，Root Mean Square Layer Normalization：<https://arxiv.org/abs/1910.07467>。RMSNorm。
7. Shazeer，GLU Variants Improve Transformer：<https://arxiv.org/abs/2002.05202>。GLU 变体与 SwiGLU。
8. Touvron 等，LLaMA: Open and Efficient Foundation Language Models：<https://arxiv.org/abs/2302.13971>。RMSNorm、RoPE、SwiGLU 组合的公开架构背景。
