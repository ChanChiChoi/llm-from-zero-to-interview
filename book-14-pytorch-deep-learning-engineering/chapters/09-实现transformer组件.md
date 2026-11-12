# 第九章：实现 Transformer 组件

前面几章分别讨论了张量、自动求导、`nn.Module`、数据加载、训练循环、混合精度、分布式训练和调试。到了这里，我们把这些能力放进一个完整而又足够小的对象中：一个可以完成前向计算、计算语言模型损失，并在生成时复用历史状态的 decoder-only Transformer。

这件事的价值不在于用几百行 Python 取代成熟的 Transformer 实现。生产系统会使用融合 kernel、专门的内存布局、分布式并行和推理调度器；本章的实现不会与它们竞争。手写一个小模型的价值在于把“张量形状、数学运算、状态边界和验证方法”放在同一张图里。当输出不对时，我们能判断问题来自投影、掩码、位置编码、残差、损失对齐，还是缓存的历史长度，而不是只看到一个最终的 loss。

本章还刻意把训练和推理放在同一条叙事中。训练阶段把一段序列一次性送入模型，所有位置并行计算；生成阶段每次只增加一个 token，历史的 key/value 被保存起来。两种模式使用相同的参数，却有不同的张量形状和因果掩码。理解这条分界线，比记住某个类的几行代码更重要。

## 9.0 阅读范围、资料与证据边界

本章实现的是一个教学版的 decoder-only Transformer，包含：token embedding、绑定权重的 LM head、RMSNorm、scaled dot-product attention、多头 self-attention、causal mask、SwiGLU MLP、Pre-Norm decoder block、RoPE、causal language-model loss，以及带 RoPE 位置偏移的 KV Cache。

代码使用 PyTorch 张量运算，优先保证可读性和可测试性。它没有实现 FlashAttention、PagedAttention、MQA/GQA、MLA、张量并行、流水线并行、prefix cache 或生产级连续批处理。这些技术解决的是更大模型或更高并发下的计算与内存问题，本章会说明它们与教学实现的边界，但不把它们假装成几行 `torch` 代码就能完整复现的功能。

本章的资料分为三类：

1. Transformer 的原始论文，用来确认 scaled dot-product attention、多头 attention 和 encoder/decoder 结构的数学定义。
2. PyTorch 官方文档，用来确认 `Embedding`、`RMSNorm`、`scaled_dot_product_attention` 等 API 的参数和行为。官方 API 文档说明的是库的语义，不等同于某个具体大模型的完整架构。
3. RoFormer 论文，用来确认旋转位置编码作用于 query/key 的基本构造。不同模型可能在旋转维度、频率基数、位置缩放和缓存方式上有自己的配置，因此下面的 `base=10000` 只是教学默认值。

资料可以证明公式、API 和公开论文中的定义；它们不能单独证明某个闭源模型一定采用本章的所有组件，也不能从一个小模型的运行速度推导出生产系统的吞吐。章节中的运行输出属于目标环境下的教学实验，硬件、PyTorch 版本和 dtype 改变后，数值的最后几位可能变化。

## 9.1 先看完整的数据流

设输入 token id 为 `input_ids`，形状是 `[B, T]`：`B` 表示 batch size，`T` 表示序列长度。模型的主干可以画成：

```text
input_ids [B, T]
    │
    ▼
token embedding [B, T, d]
    │
    ▼
重复 L 次：Pre-Norm → causal self-attention → residual
          Pre-Norm → SwiGLU MLP → residual
    │
    ▼
final RMSNorm [B, T, d]
    │
    ▼
LM head [B, T, V]
    │
    ▼
每个位置上的下一个 token 分布
```

这里 `d` 是 hidden size，`L` 是层数，`V` 是词表大小。最后的 `LM head` 把每个位置的隐藏向量投影为 `V` 个 logits；`logits[b,t,:]` 不是一个 token，而是位置 `t` 对整个词表的打分。

这条图把“模型里有哪些模块”压成了一个可以逐段检查的数据流。若某一层把 `[B,T,d]` 错改成 `[B,d,T]`，后面的线性层有时仍能接受这个张量，却会把时间和特征维当反；因此每次 reshape 或 transpose 之后，最好同时写出维度含义，而不只写总形状。可以把每个模块想成输入输出契约：embedding 负责把整数索引变成向量，attention 负责跨位置读取，MLP 负责逐位置变换，Norm 和 residual 组合更新，LM head 把特征轴映到词表轴。这样发生错误时，定位对象会落到第一处 contract 不成立的位置。

训练和生成也应在这张图上区分。训练时常用 teacher forcing：整段真实 token 一次送入，因果 mask 保证位置 `t` 看不到 `t+1`，但所有位置仍可并行算出 logits。生成时，模型先处理 prompt，再用已生成 token 逐步延长状态；每个新 token 的 query 长度变为 `1`，历史 key/value 长度继续增长。两者共享参数，不代表可以复用同一个方阵 mask 或假定输出形状不变。后面的 cache 测试会把这个形状变化作为接口，而不是实现细节。

### 9.1.1 给初学者的直觉：模型究竟在做什么

可以把一行 token 看成一列带编号的词或子词。embedding 先为每个编号取出一个向量。attention 让当前位置根据因果规则读取前面的位置；例如正在处理“北京是中国的”时，当前位置可以利用“北京”“中国”等 token 的表示。MLP 不负责读取别的位置，而是在每一个位置独立地把当前表示变换得更丰富。residual 把变换前的表示加回来，使每一层都可以在已有信息上增量修改。

经过许多层后，位置 `t` 的隐藏状态同时包含了当前 token 和可见历史的上下文。LM head 再把这个向量翻译为词表上的分数，训练目标要求它给真实的下一个 token 更高分。

### 9.1.2 从实现者角度看：shape 是第一份接口契约

主线公式是：

~~~math
I\in\mathbb{Z}^{B\times T},\qquad
X=E[I]\in\mathbb{R}^{B\times T\times d},\qquad
Z=XW_{\mathrm{lm}}^\top\in\mathbb{R}^{B\times T\times V}.
~~~

其中 `E` 的形状是 `[V,d]`。如果采用权重绑定，`W_lm` 与 `E` 共享同一个参数；如果不绑定，则 LM head 另有一个 `[V,d]` 参数。实现中最常见的错误不是矩阵乘法公式写错，而是在 reshape、transpose 或 cache 拼接后忘记某一维代表什么。

多头 attention 要求：

~~~math
d=H D_h,\qquad
Q,K,V\in\mathbb{R}^{B\times H\times T\times D_h},
~~~

`H` 是 head 数，`D_h` 是每个 head 的维度。比如 `d=16,H=4` 时，`D_h=4`；把 `[B,T,16]` reshape 成 `[B,T,4,4]` 后，再交换维度得到 `[B,4,T,4]`。

在完整序列训练中，query、key、value 的长度都为 `T`；在单 token decode 中，query 长度通常是 `1`，key/value 长度是历史长度加 `1`。因此，任何只在 `[T,T]` 方阵上验证过的 mask，都不能自动证明 cache decode 是正确的。

## 9.2 配置对象与 shape contract

把维度集中放在配置对象里，既能避免散落的数字，也能在模型构造时尽早拒绝不一致的配置。

```python
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 32_000
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 1_376
    max_position_embeddings: int = 2_048
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size // self.num_heads % 2 != 0:
            raise ValueError("head_dim must be even for this RoPE implementation")
        if min(self.vocab_size, self.hidden_size, self.num_layers,
               self.num_heads, self.intermediate_size) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("norm_eps and rope_theta must be positive")
```
> 代码片段，静态审阅，未运行；配置校验示例，使用 Python 标准库 `dataclasses`。

本章后面的代码都遵循下面的形状约定：

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B,T]` | 整数 token id |
| hidden states | `[B,T,d]` | 每个位置的连续表示 |
| 单个 head 的 `q/k/v` | `[B,H,T,D_h]` | 多头拆分后的表示 |
| attention score | `[B,H,T_q,T_k]` | 每个 query 对每个 key 的分数 |
| causal mask | `[1,1,T_q,T_k]` | 可广播到 batch 和 head |
| logits | `[B,T,V]` | 每个位置对词表的打分 |
| 每层 K/V cache | `[B,H,T_cache,D_h]` | 已经计算过的历史状态 |

这里的 `T_q` 和 `T_k` 故意分开写。训练时通常 `T_q=T_k=T`；decode 时通常 `T_q=1`、`T_k=T_cache+1`。把这两个符号混写，是缓存实现中最容易隐藏 bug 的原因之一。

这份契约还包含几个不能从形状表中省略的条件。`input_ids` 必须是整数张量，取值范围为 `[0,V)`，并且本教学实现不接受空 batch 或空序列；`labels` 必须与 `input_ids` 形状相同，除了 `-100` 之外也必须落在同一词表范围内。cache 则必须按层成对提供 key 和 value，所有层的 batch、head、head dimension、设备、dtype 与当前模型一致，历史长度也必须一致。把这些约束写在入口处，错误会停在“数据契约”这一层，而不是等到 embedding、矩阵乘法或 `torch.cat` 深处才出现一个难以定位的异常。

### 9.2.1 用一个具体尺寸检查数量级

假设 `B=2,T=5,d=16,H=4,V=32`，则 `D_h=4`。embedding 输出是 `[2,5,16]`；每个 head 的 query 是 `[2,4,5,4]`；attention score 是 `[2,4,5,5]`；合并 head 后回到 `[2,5,16]`；LM head 输出 `[2,5,32]`。这组尺寸很小，却已经覆盖了所有维度转换。

### 9.2.2 参数量也属于接口的一部分

不考虑 bias 时，一层普通多头 attention 的四个投影大约有 `4d^2` 个参数；SwiGLU 的三个投影大约有 `3d m` 个参数，其中 `m` 是 `intermediate_size`。embedding 与独立 LM head 合计约 `2Vd`；绑定权重后只保留约 `Vd`。所以权重绑定不只是一个 API 写法，它会改变参数量和输入、输出表示空间的关系。

这些是数量级估算，不是任何具体模型的完整参数统计。是否有 bias、是否使用 GQA、MLA、MoE 或额外的 embedding，会改变实际结果。

把估算代入一组小配置：令 `V=32,d=16,m=32,L=2`，且所有线性层都不带 bias。绑定的 embedding/LM head 有 `32×16=512` 个参数；每层 attention 的 Q/K/V/O 投影各有 `16×16=256` 个参数，共 `1,024`；SwiGLU 有 `16×32` 的 gate、up、down 三个矩阵，共 `1,536`；每层两个 RMSNorm 各有 16 个 scale，因此该层合计 `1,024+1,536+32=2,592`。两层是 `5,184`，加上 embedding 的 512 和 final norm 的 16，合计 `5,712`。若 LM head 不绑定，再加 `Vd=512`，就是 `6,224`。这不是凭“模型有两层”猜参数，而是把每个矩阵形状和是否共享逐项入账；真实模型还要再核对 bias、额外 embedding、位置参数和词表实现。

## 9.3 Token embedding、LM head 与权重绑定

### 9.3.1 离散编号如何进入连续空间

`nn.Embedding(V,d)` 可以理解为一个查表操作：输入的每个整数 `i` 选择参数矩阵第 `i` 行。它不是对 token id 做普通的线性插值，也不会因为 id 相邻就自动认为两个 token 相似；相似性由训练更新出来。

```python
import torch
from torch import nn


vocab_size, hidden_size = 32, 16
embed_tokens = nn.Embedding(vocab_size, hidden_size)
input_ids = torch.tensor([[1, 4, 7], [2, 3, 5]])
x = embed_tokens(input_ids)
assert x.shape == (2, 3, hidden_size)
```
> 代码片段，静态审阅，未运行；可独立在 CPU 上运行，依赖 PyTorch 张量与 `nn.Embedding`。

embedding 的输入必须是整数类型，且每个值都在 `[0,V)` 内。把浮点 token id 传进去，或把 padding id 误设成词表之外的值，属于数据接口错误，不是模型表达能力问题。

可以用一张只有三行的表把 lookup 和分类矩阵联系起来。假设 `E=[[1,0],[0,1],[1,1]]`，输入 id `2` 直接查得向量 `[1,1]`；如果隐藏状态是 `[1,0]` 且输出权重绑定，同一矩阵对三个 token 的 logits 是 `E·[1,0]=[1,0,1]`。输入侧把行号当地址，输出侧把同一行当分类向量参与点积；id 本身没有“大小相近就语义相近”的数学性质，向量关系来自训练目标不断调整参数。绑定时两条路径的梯度汇入同一个 Parameter，所以权重共享会改变优化状态和 checkpoint 内容，而不只是节省一次存储。

若数据有 PAD token，`nn.Embedding(..., padding_idx=pad_id)` 只规定该 embedding 行不从 lookup 梯度更新；新建模块时该行默认初始化为零。它不会自动把 PAD key 从 attention 中遮住，也不会自动把 PAD label 从语言模型 loss 中排除。实际 batch 还需要根据 padding 方向组合 attention mask，并把不参与预测的 label 设成 loss 的 ignore index。右侧 padding 的因果序列里，padding 出现在有效 query 之后，通常不会被有效位置读取；左侧 padding 则会处于有效 query 的历史范围内，若不加 key mask 就可能污染表示。PyTorch 2.15 的 `Embedding` 文档明确说明 `padding_idx` 的梯度语义，不能据此推断 attention 或 loss 也已忽略该位置。

### 9.3.2 LM head 的方向

若隐藏状态为 `x∈R^d`，词表矩阵为 `W∈R^{V×d}`，则一个位置的 logits 是：

~~~math
z=W x + b\in\mathbb{R}^{V}.
~~~

PyTorch 的 `nn.Linear(d,V)` 内部权重形状是 `[V,d]`，因此代码可以直接写：

```python
lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
logits = lm_head(x)
assert logits.shape == (2, 3, vocab_size)
```
> 代码片段，静态审阅，未运行；依赖前文定义的 `x`、`hidden_size`、`vocab_size` 和 `nn`。

### 9.3.3 为什么可以共享权重

语言模型常把输入 embedding 和输出投影绑定：

```python
lm_head.weight = embed_tokens.weight
```
> 代码片段，静态审阅，未运行；权重绑定片段，依赖本节已创建的 `embed_tokens` 与 `lm_head`。

这表示两个模块引用同一个 `Parameter`，不是把当前数值复制一份。训练时来自输入侧和输出侧的梯度会共同更新同一个矩阵。好处是减少约 `Vd` 个参数，并让 token 的输入表示与输出分类向量处于同一个坐标系；代价是输入、输出必须使用兼容的词表和维度，不能随意给两者使用不同的词表。

验证绑定关系要检查对象身份或数据指针，而不是只比较一次数值：

```python
assert lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()
```
> 代码片段，静态审阅，未运行；指针断言片段，依赖同一组 embedding 与 LM head。

保存和加载时还要遵循所用框架的 tied-weight 约定。若一个工程把绑定关系破坏成两个独立参数，模型仍然可以运行，但参数量、梯度和 checkpoint 语义已经改变。

## 9.4 归一化：LayerNorm 与 RMSNorm

### 9.4.1 LayerNorm 做了什么

对最后一维的输入 `x∈R^d`，LayerNorm 通常计算：

~~~math

\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,qquad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
~~~

它既移除均值，又缩放方差，并通过可学习的 `gamma`、`beta` 恢复表达能力。

### 9.4.2 RMSNorm 的不同

RMSNorm 不减均值，而是按均方根缩放：

~~~math
\mathrm{RMS}(x)=
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},qquad
y_i=\gamma_i\frac{x_i}{\mathrm{RMS}(x)}.
~~~

这减少了计算步骤，并保留了输入均值方向。它不是“更强”的 LayerNorm，而是另一种归一化假设；某些模型使用 RMSNorm，某些模型仍使用 LayerNorm 或其他变体。

### 9.4.3 一个数值稳定的实现

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must be floating point")
        input_dtype = x.dtype
        # fp16/bf16 输入用 fp32 累积；fp32/fp64 保留原计算精度。
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_compute = x.to(compute_dtype)
        variance = x_compute.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_compute * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.to(compute_dtype)).to(input_dtype)
```
> 代码片段，静态审阅，未单独运行；完整实现中的对应归一化逻辑随下方 CPU demo 检查 fp16、bf16、fp32、fp64。

对 `[B,T,d]` 输入，均方根的归一化维度是最后一维 `d`，不是 batch 维或时间维。可以用下面的性质做快速检查：当 `weight=1` 且忽略 `eps` 的微小影响时，输出每个位置的均方值接近 `1`。

低精度输入转成 fp32 是教学实现中的数值保护措施；fp32 与 fp64 输入则保留各自的计算 dtype。实际 [PyTorch 2.15 `nn.RMSNorm`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html) 在 `eps=None` 时根据 computation dtype 选择 machine epsilon，而本例显式使用 `1e-6`。不能因为两个实现都叫 RMSNorm，就假定默认 epsilon 与所有边界行为完全一致。

用同一个二元向量看两种归一化到底改了什么。取 `x=[3,4]`、scale `gamma=[1,1]`、bias `beta=[0,0]`，先忽略很小的 epsilon。LayerNorm 的均值是 `(3+4)/2=3.5`，有偏方差是 `((3-3.5)^2+(4-3.5)^2)/2=0.25`，标准差为 `0.5`，所以输出约为 `[-1,1]`。RMSNorm 不减均值，均方值是 `(3^2+4^2)/2=12.5`，RMS 为 `sqrt(12.5)≈3.5355`，输出约为 `[0.8485,1.1314]`。前者把中心移到零，后者保留了输入的正向均值；两者都把尺度控制在可学习的 affine 变换前。若用本节代码的 `eps=1e-6`，上述舍入只会在最后几位略变。

工程中不能只比较一组向量的输出。LayerNorm 额外减均值，RMSNorm 少一步统计；训练稳定性还受 residual 路径、初始化、学习率、数值精度和 epsilon 共同影响。加载 checkpoint 时，Norm 类型与 scale 参数形状属于架构契约，不能把一个 Norm 的参数直接解释成另一个 Norm 的参数。本节的实现只展示最后特征维的计算，没有试图证明 RMSNorm 在所有任务上更快或更好。[PyTorch 2.15 `LayerNorm` 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)说明其方差采用 `correction=0` 的有偏估计；本节教学 RMSNorm 显式使用 `eps=1e-6`，而官方 `nn.RMSNorm(eps=None)` 按 computation dtype 选择 machine epsilon，因此相同模块名不代表默认数值完全一致。

## 9.5 从 Q/K/V 到 scaled dot-product attention

### 9.5.1 先理解一次注意力

设 query、key、value 的形状分别为 `[B,H,T_q,D_h]`、`[B,H,T_k,D_h]`、`[B,H,T_k,D_h]`。注意力的计算为：

~~~math
S=\frac{QK^\top}{\sqrt{D_h}}+M,qquad
A=\mathrm{softmax}(S),qquad
O=AV.
~~~

`QK^T` 在最后两维相乘，因此 `S` 的形状是 `[B,H,T_q,T_k]`。对固定的 query 位置，`A` 在 key 维上求和为 `1`；输出 `O` 是 value 的加权和。

### 9.5.2 为什么要除以 `sqrt(D_h)`

如果 query 和 key 的分量近似独立、均值为零、方差为一，那么点积的方差会随 `D_h` 增长。维度变大时，未经缩放的分数更容易落入 softmax 的极端区域，导致一个位置几乎独占权重，梯度也更难训练。除以 `sqrt(D_h)` 把分数的量级拉回较稳定的范围。

这不是让 softmax “变得正确”的魔法，而是对随机初始化下分数尺度的控制。训练后向量分布不一定满足这些理想假设，仍需通过归一化、初始化和数值监控共同保持稳定。

可以从方差把缩放因子算出来：若每一维的 `q_i`、`k_i` 均值为 0、方差为 1，且各维近似独立，则乘积 `q_i k_i` 的方差约为 1。四维点积把四项相加，方差约为 4，标准差约为 2；除以 `sqrt(4)=2` 后，分数标准差回到约 1。维度从 4 增为 64 时，未缩放点积的标准差近似从 2 增至 8；softmax 对这种尺度变化很敏感，较大的分数会让概率集中、反向信号变窄。这个推导依赖独立、单位方差的近似，是解释公式的初始化直觉，不是对训练后 Q/K 分布的精确断言。

### 9.5.3 causal mask 的数学含义

自回归语言模型在位置 `t` 预测下一个 token 时，不能读取未来位置。对完整序列，允许矩阵为：

~~~math
M_{t,s}=\begin{cases}
0,&s\le t,\\
-\infty,&s>t.
\end{cases}
~~~

把被禁止位置加上负无穷后，softmax 的权重理论上为零。实际低精度代码不必写 `-1e30`；这个数字可能超出 fp16 的表示范围。使用 `torch.finfo(scores.dtype).min`，或者让 PyTorch 的 attention kernel 处理 mask，更稳妥。

把缩放、mask 和 softmax 放在一次手算中，能看见这三个操作分别做什么。假设某个 head 已经投影得到 `q=[0,1]`，两个 key 是 `k_0=[1,0]`、`k_1=[0,1]`，对应 value 为 `v_0=[10,0]`、`v_1=[0,20]`。在位置 1，未缩放分数为 `[0,1]`，除以 `sqrt(2)` 后为 `[0,0.7071]`；softmax 权重大约是 `[0.3302,0.6698]`，加权输出约为 `[3.302,13.396]`。若这是位置 0，因果 mask 会把第二个 key 设为负无穷，权重变成 `[1,0]`，输出就是 `[10,0]`。所以 mask 不会改 Q/K/V 本身，而是改 query 能把多少概率分给各个 key；scale 控制分数范围，softmax 把合法分数变为归一化权重，最后再由 value 决定输出内容。

### 9.5.4 教学版 attention

```python
import math
import torch.nn.functional as F


def make_causal_mask(query_len, key_len, device, past_key_values_length=0):
    if past_key_values_length < 0:
        raise ValueError("past_key_values_length must be non-negative")
    if key_len != past_key_values_length + query_len:
        raise ValueError(
            "for this contiguous cache, key_len must equal past_len + query_len"
        )

    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    allowed = key_positions <= query_positions + past_key_values_length
    return allowed[None, None, :, :]  # [1, 1, T_q, T_k]


def scaled_dot_product_attention(
    q,
    k,
    v,
    causal=True,
    past_key_values_length=0,
    return_attn=False,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, T, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
        raise ValueError("batch and head dimensions must match")
    if q.size(-1) != k.size(-1):
        raise ValueError("query and key head dimensions must match")

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        mask = make_causal_mask(
            q.size(-2), k.size(-2), q.device, past_key_values_length
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    # 用 fp32 做 softmax，再回到 q 的 dtype，减少低精度下的舍入误差。
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    context = attn @ v
    if return_attn:
        return context, attn
    return context
```
> 代码片段，静态审阅，未单独运行；依赖前文的 `torch` 与 PyTorch 2.15 API 语义，目标是说明显式 causal mask。

`mask` 的形状是 `[1,1,T_q,T_k]`，会广播到 `[B,H,T_q,T_k]`。广播不会复制出一个完整的 batch/head mask，因此既清晰又节省内存。代码中的检查还把一个隐含前提显式写出来：本实现只支持连续的 cache，即 key 序列由长度为 `past_len` 的历史和当前 `query_len` 个新 token 顺序拼接而成。

### 9.5.5 PyTorch 的融合实现与教学实现的关系

PyTorch 提供 `torch.nn.functional.scaled_dot_product_attention`。它可以根据设备、dtype 和参数选择不同的后端，通常比直接保存完整的 attention 矩阵更适合生产计算。教学实现保留 `attn` 是为了检查每行和、未来位置是否为零；真实服务通常不会返回这块 `[B,H,T_q,T_k]` 的矩阵，否则会额外占用 `O(BHT_qT_k)` 内存。

使用官方函数时，需要明确 `is_causal`、显式 `attn_mask` 和非方阵 query/key 的语义。完整序列的方阵 attention 可以直接使用 causal 后端；cache decode 的 `T_q != T_k` 则应根据当前 PyTorch 版本文档选择正确的 mask 形式。本章手写显式 mask，是为了把“第几个 query 能看见第几个 key”直接呈现出来，而不是暗中依赖一个特定版本的默认行为。

具体到[PyTorch 2.15 SDPA 文档](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)，`is_causal=True` 在非方阵上使用左上对齐的 causal bias；例如 `T_q=1,T_k=6` 时，唯一一行只开放 key 0。这和 cache decode 中“新 query 位置为 5，因此可以看见 key 0–5”的绝对位置规则不同。2.15 提供的 [`causal_lower_right`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_lower_right.html) 函数在调用 `causal_lower_right(1, 6)` 时会给出右下对齐的全可见行，但该 bias API 标为 prototype；也可以像本章这样直接按 `key_position <= past_len + query_index` 构造布尔 mask。SDPA 不允许同时设置 `attn_mask` 和 `is_causal=True`，所以有 padding mask 或 cache offset 时应明确合成正确 mask，再按 API 约定传入。布尔 SDPA mask 中 True 表示参与 attention，而 [`nn.MultiheadAttention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html) 中的 mask 语义相反，迁移代码时要核对方向。

SDPA 的 `dropout_p` 是显式函数参数，它不会因为外层 module 调用 `.eval()` 就自动变成 0。训练态应传模型配置的 dropout 概率，评估和生成态则传 `0.0`；否则同一个 `eval()` 模型每次输出仍可能随机变化。不同后端会融合不同浮点运算，数值可能有小差异，测试应使用与 dtype 和后端相称的容差。融合 kernel 的选择由输入条件决定；调用 SDPA 并不保证一定运行 FlashAttention，也不保证在目标硬件上更快。上述 dropout、mask 和后端选择边界均以[PyTorch 2.15 SDPA API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html)为准。

## 9.6 多头 self-attention：把一条表示拆成多个子空间

单头 attention 只有一套相似度函数。多头 attention 先用不同的线性投影得到多个子空间，在每个子空间中独立计算，再把结果拼接回来：

~~~math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
~~~

~~~math
Q,K,V:[B,T,d]
\xrightarrow{\mathrm{reshape+transpose}}
[B,H,T,D_h].
~~~

不同 head 可以学习不同的关系，例如局部搭配、长距离指代或标点结构；这是一种解释性直觉，不意味着每个 head 都能被稳定地命名为某种功能。`o_proj` 把拼接后的 `[B,T,d]` 再做一次混合。

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

    def _shape(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, T, D_h]

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x))
        k = self._shape(self.k_proj(x))
        v = self._shape(self.v_proj(x))
        context = scaled_dot_product_attention(q, k, v, causal=True)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(context)
```
> 代码片段，静态审阅，未运行；依赖前文的 `torch`、`nn` 和 `TransformerConfig`。

`transpose` 只改变视图的步长，通常不会让内存变成连续布局；所以在 `view` 前写 `contiguous()` 是一种明确的布局转换。也可以使用 `reshape`，但不能把它当成“所有 layout 问题自动消失”的理由：在大张量上，`reshape` 可能隐式复制，仍要知道它的成本。

固定 `d=4`、`H=2` 时，`D_h=2`。一个 `[1,3,4]` 的隐藏状态先经四个独立投影，Q/K/V 的张量形状仍各为 `[1,3,4]`；reshape 后为 `[1,3,2,2]`，transpose 后成为 `[1,2,3,2]`。于是每个 head 对三个 query 和三个 key 形成 `[1,2,3,3]` 的分数，两个 head 各自归一化、加权 value，拼接回 `[1,3,4]`，最后由 `o_proj` 混合两个子空间。head 数量不是“模型自动发现的功能标签”；它决定投影后如何切分表示。把 `H` 从 1 改成 2、保持 `d` 不变，投影参数量仍约 `4d²`，但每个 head 的维数减半，attention 权重张量的元素数从 `T²` 增成 `2T²`。总点积 FLOPs 的量级仍随 `T²d` 变化，score 中间量和 kernel 调度形态却会改变，因此 head 配置要结合表达能力与硬件实测，而不是只看 `H` 越大越好。

多头投影还解释了为何“每个 head 可以专注一种关系”只是一种便于理解的直觉。所有 head 都由同一训练损失共同驱动，head 之间可能冗余、组合或随层变化；把某个 head 命名为“语法头”需要额外的因果干预或可解释性证据。工程排错时，先检查 reshape 和 transpose 后的 head 轴，再检查每个 head 的 attention 行和与 mask；不能仅根据某个 head 的权重图给出能力结论。

把张量形状串起来看，attention 不是把 `[B,T,d]` 变成另一个任意表示：它对每个 query 位置产生一个 value 加权和，仍保留 batch、head 和 query 轴；最后 transpose、concat 与 `o_proj` 把 head 维折回 `d`。若 query 序列长度为 `T_q` 而 key/value 为 `T_k`，分数的 `T_q×T_k` 是真实的关系表，不是隐藏维。缓存 decode 只把 `T_q` 从整段长度缩成新 token 数，`T_k` 仍覆盖历史。这也是为什么形状表里把两种长度分开，是 attention 与 generation 接合处的核心约定。

## 9.7 RoPE：把位置放进 Q/K 的几何关系

### 9.7.1 为什么需要位置机制

没有位置机制时，self-attention 看到的是一组向量和它们之间的相似度，单凭内容无法区分“甲追上乙”和“乙追上甲”这样的顺序变化。位置机制要让模型知道 token 在序列中的位置，同时尽量保留 attention 对相对距离的敏感性。

### 9.7.2 二维旋转

RoPE 把 head 维度两两分组。位置 `m`、第 `i` 个二维子空间的角度为 `theta_{m,i}`，然后使用旋转矩阵：

~~~math
\begin{bmatrix}q'_{2i}\\q'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_{m,i}&-\sin\theta_{m,i}\\
\sin\theta_{m,i}&\cos\theta_{m,i}
\end{bmatrix}
\begin{bmatrix}q_{2i}\\q_{2i+1}\end{bmatrix}.
~~~

同样的旋转作用于 key。常见频率构造为：

~~~math
\theta_{m,i}=m\cdot\text{base}^{-2i/D_h}.
~~~

旋转矩阵是正交矩阵，所以每个二维分量的范数保持不变。更重要的是：

~~~math
(R_mq)^\top(R_nk)=q^\top R_m^\top R_n k
=q^\top R_{n-m}k,
~~~

在二维旋转可交换的条件下，点积含有位置差 `n-m`。这解释了“相对位置信息自然进入 Q/K 点积”的说法；它不意味着模型对任意长度外推都没有误差。

令 head_dim 为 2，只看一个频率对，取基向量 `q=k=[1,0]`，并按本节教学默认 `base=10000`。第一个频率 `i=0` 的角速度是 `10000^0=1` 弧度/位置。query 在位置 1 旋转后为 `[cos1,sin1]≈[0.5403,0.8415]`；key 在位置 3 旋转后为 `[cos3,sin3]≈[-0.9900,0.1411]`。两者点积约为 `cos(3-1)=cos2≈-0.4161`，若两者都在位置 0，点积则是 1。绝对位置改变了向量方向，相对位移 `2` 决定两者内积。head_dim=4 时还会有第二个频率 `10000^{-1/2}=0.01`，同一位置移动 2 步只转 `0.02` 弧度；多个频带让模型同时拥有快、慢位置变化，但具体频率分布是架构配置。

这也提示实现时要区分数学等价与张量布局兼容。当前代码将相邻偶数/奇数维组成二维对，依次作用旋转；其他实现可能把前半维和后半维配对，公式能转换但权重布局与 cache 不能直接互换。迁移预训练参数时要和目标模型的 `rotary_dim`、频率基数、位置缩放及 `position_ids` 约定一起核对。使用长绝对位置时，fp32 位置和频率计算比低精度更稳，但 RoPE 的相对点积结构本身不保证训练窗口以外的位置分布仍可靠。

从技术谱系看，原始 Transformer 使用正弦/余弦位置向量加到 token 表示上；RoPE 改为在每一层 attention 的 Q/K 上施加位置相关旋转。前一种方法让位置信号直接进入每层可见的 hidden 表示，后一种方法把相对位移效应放进 query-key 点积。它们回答的是“位置怎样参与 attention”的不同设计，不能仅凭公式有正弦或余弦就视作同一个实现。不同 RoPE 变体会改动旋转维度、频率、位置缩放或插值方法；是否适合长上下文，要看模型训练时的位置范围、权重来源和目标数据，而不是从旋转矩阵范数守恒直接推出。

本章的 cache 保存已旋转的 `k`，因此位置 offset 属于状态 contract。若历史长度 `p=4`，下一 token 的 q/k 都应使用位置 4；若误用位置 0，张量 shape、范数和 attention 行和可能全部正常，只有相对位置关系和 prefill/decode 输出对照能揭示错误。一个有效的 RoPE 单元测试应至少包括两类性质：对任意输入检查旋转前后范数近似相等；再取同一组 q/k 放到不同绝对位置，检查点积随相对差变化。第一类只证明旋转数值结构，第二类才能说明位置索引进入了 attention。

### 9.7.3 生成 cos/sin cache

```python
def build_rope_cache(
    seq_len,
    head_dim,
    device,
    base=10_000.0,
    start_pos=0,
):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for pairwise RoPE")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(
        start_pos, start_pos + seq_len, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)  # [T, D_h / 2]
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x, cos, sin):
    # x: [B, H, T, D_h], cos/sin: [1, 1, T, D_h/2]
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack(
        (x_even * cos - x_odd * sin,
         x_even * sin + x_odd * cos),
        dim=-1,
    )
    return rotated.flatten(-2)
```
> 代码片段，静态审阅，未单独运行；完整模型复用该 RoPE 算法，CPU demo 检查范数和 prefill/decode 位置对齐。

把 `cos`、`sin` 作为缓存保存下来，可以避免每次 forward 重新计算频率。cache 的 dtype 选择需要和工程的数值策略一致：用 fp32 保存通常更精确，应用时再转换到 Q/K 的 dtype；如果序列非常长，频率计算、位置缩放和外推策略还会成为单独的设计问题。

### 9.7.4 RoPE 与 KV Cache 的位置偏移

prefill 的第一个 token 位置通常是 `0`，长度为 `P` 的历史 cache 之后，新 token 的绝对位置从 `P` 开始。因此 decode 时不能每次都用位置 `[0]` 旋转新的 query/key，否则相同内容出现在不同生成步会得到错误的位置关系。正确关系是：

~~~text
prefill: positions = [0, 1, ..., P-1]
decode : positions = [P, P+1, ..., P+T_new-1]
~~~

本章的完整实现把 `past_len` 作为 `start_pos`，并且只对新产生的 `k` 做旋转；缓存中的历史 `k` 已经按旧位置旋转过，不能再次旋转。

## 9.8 MLP 与 SwiGLU

attention 负责位置之间的信息交换；MLP 对每个位置独立应用相同的参数。普通两层 MLP 可以写成：

~~~math
\mathrm{MLP}(x)=W_2\,\phi(W_1x+b_1)+b_2.
~~~

SwiGLU 使用一个门分支和一个值分支：

~~~math
\mathrm{SwiGLU}(x)=W_d\left[
\mathrm{SiLU}(W_gx)\odot(W_ux)
\right].
~~~

`SiLU(z)=z·sigmoid(z)`，`⊙` 是逐元素乘法。门分支可以控制哪些中间特征被放大或抑制；它的代价是三个线性投影，而不是普通 MLP 的两个，所以 `intermediate_size` 往往会相应调整，不能只把普通 MLP 的中间维度原样照搬。

```python
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```
> 代码片段，静态审阅，未运行；依赖 `nn` 与 `TransformerConfig`。

因为 MLP 对最后一维做线性变换，`[B,T,d]` 经过它仍然是 `[B,T,d]`。这并不代表它没有时间复杂度：每个 token 都要做三次投影，计算量大约与 `B·T·d·intermediate_size` 成正比。

用一个标量例子观察门控如何改变值。取输入 `x=1`，gate 权重 `W_g=1`、value 权重 `W_u=2`、输出权重 `W_d=1`，忽略 bias。则 gate 为 `SiLU(1)=1/(1+e^{-1})≈0.7311`，up 分支为 `2`，逐元素乘积是 `1.4622`，down 投影后输出仍约 `1.4622`。若把 gate 换成 ReLU，这个输入会得到 `ReLU(1)×2=2`；两者数值不同，说明非线性与门控共同决定中间通道，不是把两个线性层简单并排。真实向量场景是在 `m` 个通道逐维做同样的乘法，再由 down projection 混合回 hidden size。

按参数量比较，设 `d=4,m=6` 且无 bias，普通两层 MLP 的矩阵参数为 `4×6+6×4=48`；SwiGLU 的 gate、up、down 三个矩阵合计 `3×4×6=72`，多出 50%。这个多出的投影容量需要更合适的 `m` 才能与普通 MLP 做公平对照。两种 MLP 都是逐 token 的位置变换，不在 token 之间通信；attention 负责跨位置读取，二者串联后才构成 block 的一次完整更新。代码中的 `F.silu` 作用于 gate 的结果，而不是先把 `up` 分支做激活。

若 `x` 是 `[B,T,d]`，三个线性层都只变换最后一维：gate、up 得到 `[B,T,m]`，逐元素乘法仍为 `[B,T,m]`，down 后回到 `[B,T,d]`。例如 `B=2,T=3,d=4,m=6`，每层中间 gate/up 各有 `2×3×6=36` 个激活值，乘完仍是 36 个，输出则有 24 个。这种逐位置共享权重让不同 token 可以并行通过 MLP；它不能代替 attention，因为其中没有任何 `t` 与 `s` 之间的混合。长序列上，即便 attention 已融合，MLP 的矩阵乘仍会在每个 token 重复执行，所以 benchmark 时不能只看 attention kernel。

模型设计里的 `intermediate_size` 也不是由某个固定整数决定。经典两层 MLP 常见约 `4d` 的扩展宽度，SwiGLU 为了容纳三个投影，论文/模型配置可能选更窄的中间维以控制参数预算；不同结构如果没有固定参数量、训练 token、优化配置和数据，就不能只比较验证 loss 断言激活函数胜负。实际迁移 checkpoint 时还要看 gate/up/down 的权重键名与矩阵方向；把 `gate_proj` 和 `up_proj` 对调可能仍能运行，但函数已改变。

## 9.9 Pre-Norm Decoder Block

把 attention 和 MLP 放进残差结构，常见的 Pre-Norm 写法是：

~~~math
h=x+\mathrm{Attn}(\mathrm{Norm}_1(x)),
\qquad
y=h+\mathrm{MLP}(\mathrm{Norm}_2(h)).
~~~

直觉上，attention 和 MLP 各自只负责一次增量更新，residual 保留一条从输入到输出的直接路径。对深层网络而言，这条路径有利于梯度传播。Post-Norm 把 norm 放在加法之后，数学上也是一种合理结构，但初始化、训练稳定性和可迁移 checkpoint 不能混用；实现时必须以目标模型的配置为准。

```python
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = MultiHeadSelfAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x
```
> 代码片段，静态审阅，未运行；依赖本节定义的 self-attention 与 MLP 模块。

这个版本用于说明 block 的结构。后面的完整实现会把带 RoPE 和 cache 的 attention 注入同样的残差骨架。不要把“attention 负责 token 间通信、MLP 负责逐 token 变换”误读为严格的功能隔离；训练后的两个子模块会共同编码信息，这只是分析计算路径的有用分解。

可以把残差路径按具体数值展开。假设输入 hidden 是 `[1,2]`，Norm 与 attention 子层合起来返回 `[0.5,-1]`，第一次相加后 `h=[1.5,1]`；第二个 Norm 与 MLP 子层合起来返回 `[-0.25,0.5]`，最终 `y=[1.25,1.5]`。若实现者误把第二个加数写成原始 `x` 上的 MLP 输出，就会得到不同的依赖关系；残差相加的位置决定了第二个子层读取的是已经吸收 attention 更新的 `h`。这个小算例不声称子层本身会产生给定数值，而是把 block 的状态更新规则逐步走完，便于和代码中的两次 `x = x + ...` 一一核对。

Pre-Norm 与 Post-Norm 的差异会贯穿深层训练。Pre-Norm 在每个子层前归一化，identity residual 从 block 输入直达输出；其梯度路径较直接，但输出表示的尺度由多个层共同决定。Post-Norm 在残差相加后再做 Norm，归一化位置改变了前向与反向的状态路径。二者不是只改一个类名就能互换的“实现风格”：学习率、初始化、Norm 参数、checkpoint 结构以及训练配方都可能依赖原架构。若只复用本章代码拼 toy 模型，可以实验比较；若加载已有权重，应先服从模型的配置与实现。

把本节数值继续到形状层面：对 `x:[B,T,d]`，第一次归一化不改 shape，attention 输出也回到 `[B,T,d]`，相加后仍是 `[B,T,d]`；第二个 norm、SwiGLU 和相加同样保持该 shape。假设前面算出的 `h=[1.5,1]`，下一个 MLP 分支给 `[-0.25,0.5]`，结果 `[1.25,1.5]` 不是把两种特征平均，而是保留原状态并叠加更新。梯度从输出对 `x` 至少有 residual 的直接导数路径；与此同时，也会穿过两个子层的归一化和线性投影。若把 residual 漏掉，forward 形状依旧不变，但信息和梯度路径都不同，所以测试不能只看维度。

这一分解也给 profile 结果一个合理解释。attention 包含 Q/K/V/O 投影和按 token 关系构造的加权操作；MLP 包含 gate/up/down 矩阵；Norm、残差和激活则是相对小的逐元素计算。短序列上大矩阵投影往往重要，长序列上 attention 的 `T_q×T_k` 项也可能占主导。模型使用的 hidden、head、intermediate 与序列长度共同决定每个算子的工作量。Profiler 中某一算子时间高，说明当前输入下它在 trace 里显眼，不直接证明改成另一种 activation 或 Norm 就会更快；必须固定相同层宽、工作量和硬件再比较。

## 9.10 Causal LM 的输出与损失对齐

给定 token 序列 `y_0,y_1,...,y_{T-1}`，语言模型在位置 `t` 使用 `y_0...y_t` 预测 `y_{t+1}`。因此 logits 和 labels 要错开一格：

~~~math
L=-\frac{1}{N}
\sum_{b,t:\,y_{b,t+1}\ne-100}
\log p_\theta(y_{b,t+1}\mid y_{b,\le t}).
~~~

实现为：

```python
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()
loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=-100,
)
```
> 代码片段，静态审阅，未运行；loss 片段依赖 `logits`、`labels` 与 PyTorch `F`。

`-100` 是 PyTorch 交叉熵常用的忽略标签值。padding、只作为上下文而不应计入监督的 prompt 部分，都可以设为 `-100`，但必须确认数据处理阶段没有把真正的 token 错误地屏蔽掉。

还要区分“有被忽略的位置”和“没有任何有效目标”。交叉熵的 mean reduction 本质上是对有效目标的损失求和，再除以有效目标数；如果 shift 之后所有标签都是 `-100`，分母为零，PyTorch 可能返回 `NaN`。这不是“这一批样本 loss 恰好为零”，而是这批数据没有定义训练信号。数据管线可以选择丢弃这批、重新组成 batch，或在统计时跳过它；本章的最小实现直接拒绝这种输入。

同理，长度为 `T=1` 的序列在右移后没有任何 next-token 目标，即使 embedding 和 Transformer 主干能够完成前向，也不能计算本章定义的 causal-LM loss。形状合法不等于监督目标存在，这个区别在对话数据含有空答案、截断样本或全是 prompt 的 batch 中尤其重要。

手算一组 batch 可以同时检查 shift、忽略值和 reduction 分母。令输入 token 是 `[3,1,2,0]`，labels 是 `[3,1,2,-100]`。logits 的 shape 为 `[1,4,4]`，丢掉最后一行后得到 `[1,3,4]`；labels 向左移一位得到 `[1,2,-100]`，展平后分别是三行 logits 和三个类别 id。假设第一行对目标 1 的概率为 `0.8`，第二行对目标 2 的概率为 `0.5`，第三行被忽略，则两项交叉熵为 `-ln(0.8)≈0.2231` 和 `-ln(0.5)≈0.6931`，token-mean loss 是 `(0.2231+0.6931)/2≈0.4581`，不是除以 3。理想化的 PPL 是 `exp(0.4581)≈1.581`。这组概率是教学输入，不是模型实测；读者可用同样的两行概率复算 `ignore_index=-100` 如何改变分母。

代码应把原始 logits 交给 `F.cross_entropy`，因为它内部会做 log-softmax 与负对数似然。先 softmax 再把概率送入 cross entropy 会被函数当成 logits 再归一化一次，目标就变了。还要将 padding label 和 attention mask 分开：`ignore_index` 只控制 loss 的目标计数，不控制某个 key 能否被模型读取。当前实现只有 causal mask，因此假设输入采用右侧 padding；若换成左侧 padding或任意稀疏可见关系，还必须显式合成 key-padding mask，并保证每个有效 query 至少有一个可见 key。

训练和生成的调用边界也由此明确。训练时 `labels` 必须与输入序列对应，模型一次产生整段 logits，再用 shift 形成监督；生成时不提供 labels，最后位置 logits 进入采样或解码器。带 cache 的输入只包含新 token，若直接拿完整原始 labels 与增量 logits 做同一套 shift，长度虽然可能能凑到相同，却不再表示同一批目标位置。本教学模型对“有 cache 同时传 labels”直接报错，把目标对齐留给调用者显式设计。

对话微调还要区分 prompt loss 和 completion loss。假设输入 `[system,user,assistant_1,assistant_2]`，模型仍需要完整 prompt 上下文，但不一定训练它预测 system/user token；相应 `labels` 可以在 prompt 区间置为 `-100`，仅保留 assistant 目标。若不做这个 mask，模型优化目标就包含了“重建用户问题”，统计的 token-mean loss 也不再代表预期监督区间。反过来，若 mask 边界多右移一格，第一条 assistant token 可能被漏掉。对话模板中的特殊 token 是否参与监督，属于数据目标定义，必须与 tokenizer/template 约定一起核对。

当 batch 含不同数量的有效目标时，逐 batch mean 再平均和全局按 token 平均不是一回事。若 batch A 有 2 个目标、平均 loss 为 2，batch B 有 6 个目标、平均 loss 为 1，简单 batch mean 是 `(2+1)/2=1.5`；按 token 聚合则 loss sum 为 `2×2+6×1=10`，除以 8 得 `1.25`。如果不同 batch padding 比例不同，报告值和学习率调度可能因 reduction 口径而变化。训练日志、验证 loss 与 perplexity 要共用有效目标分母；否则曲线变化可能来自 label mask 或 batch size，而不是模型泛化。

因果约束与 shift loss共同防止“答案从哪里来”的信息泄漏。causal mask 约束位置 `t` 的 hidden state 只能依赖 `0...t`；label shift 再要求该状态去预测 `t+1`。若 mask 方向反了，位置 0 可以读取未来 token；若忘记 shift，位置 `t` 会直接拿自己的 token 当目标。两种错误都可能让训练 loss 很快降低，所以需要分别检查允许矩阵和目标对齐，而不能把一条漂亮的 loss 曲线当作正确性证据。

如果 loss 使用自然对数，理想化条件下困惑度为：

~~~math
\mathrm{PPL}=\exp(L).
~~~

只有在分母是有效 token 数、不同 batch 的 loss 口径一致时，才能把不同实验的 loss 或 PPL 放在一起比较。一个忘记 shift 的实现可能仍然得到有限数值，甚至得到看似很好的 loss，但目标已经泄漏或错位。

## 9.11 KV Cache：从完整序列到增量生成

### 9.11.1 没有 cache 会重复什么

生成到第 `t` 个 token 时，朴素做法把长度为 `t` 的完整前缀重新送入每一层。历史 token 的 Q/K/V 会被反复计算，随着生成长度增加，重复工作迅速变大。

有 cache 时，prefill 阶段一次计算 prompt 的 K/V；之后每个 decode step 只为新 token 计算 Q/K/V，把新的 K/V 接到历史末尾。当前 query 仍要与全部历史 key 做点积，因为新 token 需要读取历史信息；cache 消除的是历史 K/V 投影和重复的前缀计算，不是把 attention 变成完全与上下文长度无关的操作。

### 9.11.2 形状与内存公式

单层、普通 MHA、每个元素 `b` 字节时，cache 的粗略内存为：

~~~math
M_{\mathrm{kv,layer}}
=2\times B\times H\times T_{\mathrm{cache}}\times D_h\times b.
~~~

`2` 来自 K 和 V。若有 `L` 层，总量近似为：

~~~math
M_{\mathrm{kv,total}}
=2L B H_{\mathrm{kv}}T_{\mathrm{cache}}D_h b.
~~~

这是假设每层都缓存完整 K/V 的教学公式。GQA/MQA 会让 `H_kv` 小于 query head 数；MLA 或线性状态模型可能保存不同的状态，不能把本公式不加修改地套用。

带入具体预算：若 `L=32,B=2,H_kv=4,T_cache=512,D_h=64`，使用 bf16 或 fp16（每元素 `b=2` bytes），每层需要 `2×2×4×512×64×2=1,048,576` bytes，也就是 1 MiB；32 层共 32 MiB。其他条件不变、KV head 减到 2 时则约为 16 MiB。这个差异能从公式直接读出：缓存容量与 `H_kv` 成正比，但 GQA 还改变训练时 K/V 投影和模型质量约束，不能只因缓存更小就断言整体模型更好。该估算也没有包括 allocator 对齐、临时 buffer、请求元数据和其他层状态。

### 9.11.3 cache mask 为什么不是普通下三角矩阵

设历史长度为 `p`，本次有 `q` 个新 token，那么 key 长度为 `p+q`。第 `r` 个新 query 的绝对位置是 `p+r`，它可见的 key 下标是 `0...p+r`：

~~~math
M_{r,s}=\begin{cases}
0,&s\le p+r,\\
-\infty,&s>p+r.
\end{cases}
~~~

当 `p=5,q=1` 时，mask 只有一行 `[1,1,1,1,1,1]`；如果错误地生成一个长度为 `1` 的普通下三角矩阵，当前 token 就只能看到第一个 key，历史信息全部丢失。

对一个较短的生成过程逐次记账，能区分“少重算”与“不要算 attention”。假设 prompt 长度 `P=3`，希望再生成两个 token。无 cache 时，第一轮用长度 3 的完整前缀计算下一个 token，因果位置对数是 `1+2+3=6`；第二轮把新增 token 拼回去，再对长度 4 的序列全部重算，位置对数是 `1+2+3+4=10`，合计 16。带 cache 时先对 prompt 做一次 prefill，仍是 6 对；下一轮只处理新 token 的一个 query，它读取 4 个 key，合计 4 对，总计 10。历史投影也从重复处理 `3+4=7` 个位置变成 `3+1=4` 个位置。cache 降低了重复前缀投影和 query 行数，但新 query 仍要读取所有 4 个 key；当历史变长，单步 attention 的读写量仍随 cache 长度线性增加。

### 9.11.4 生成中的 batch 约束

教学实现假定同一个 batch 中所有样本的 cache 长度相同。真实生成服务经常同时处理不同长度的请求，需要 padding、block table、分页 cache 或请求级索引。那是调度和内存管理问题，不能只在 `torch.cat([past_k,new_k],dim=2)` 外面加一层循环就解决。

状态的所有权也是 cache 契约的一部分。每层返回的 `(K,V)` 是该请求当前前缀的状态；下一步需要把它原样传回对应层，而不是跨请求复用或在样本重排后忘记同步重排。beam search、连续 batching 或多请求调度可能会重排 batch 行、删除已结束请求、改变每个序列的长度；本章的 tuple cache 只支持同长、固定 batch，不提供这些索引管理。若将 cache 与其他请求混用，即使 tensor shape 完全相同，位置和内容已属于不同序列，结果仍是语义错误。

内存和复制也要分开看。`torch.cat((past_k,k_new), dim=2)` 每次构造一份更长的连续张量；旧长度为 `p` 时，每个 decode step 至少要读写与 `p` 成正比的历史存储。重复追加 `N` 步会累计搬移约 `1+2+...+N=N(N+1)/2` 个长度单位，因此教学 `cat` 的实现方便观察状态，却不是生产 cache allocator。服务引擎通常使用预分配块、分页或请求级索引来减少全量复制与碎片；这些机制和本章“保存历史 K/V”的算法概念有关，却是另一个内存管理层次。

多 token 分块 decode 也可以用绝对位置逐行检查。若 `past_len=p=3`、本次 `q=2`，key 长度为 5；query 0 的绝对位置是 3，所以可见 key `[0,1,2,3]`，mask 行是 `[1,1,1,1,0]`；query 1 的绝对位置是 4，mask 行为 `[1,1,1,1,1]`。这说明“非方形 causal”不是一律全可见：只有 `q=1` 且正好追加在缓存末尾时，新 query 才能看到当下全部 K；一次输入多枚新 token 时，较早的新 query 仍不能偷看同一块里的未来 token。测试 `q=2` 能抓出只针对单 token decode 写出的过度开放 mask。

缓存按层保存的是 attention 的中间状态，不是模型参数，也不是已经完成的 token 文本。每一层用当前输入计算新的 K/V，和该层历史 K/V 沿序列轴拼接；下一层仍处理新的 hidden state，但可复用的是各自层的投影结果。把最后一层的 cache 错传给所有层、只缓存 K 忘了 V，或把 RoPE 位置偏移和 cache 长度分开管理，都会产生形状看似合理但不可用的状态。当前实现按 layer index 一一对应，明确要求每层 batch/head/dtype/device 相同；如果模型增删层或改变 KV head 结构，旧 cache 不再兼容。

训练与推理的资源目标不同。训练通常需要对整段序列保留反向传播所需的中间量，不能简单把无梯度生成 cache 的内存公式当作训练峰值；推理 cache 常在请求生命周期内持有，并随已生成长度增长。`torch.no_grad()` 可以省去 autograd 图，但不会自动减少 K/V 的元素数量。容量规划还应乘上活跃请求数、每层数、KV head、cache dtype 和实际序列长度，并加上 allocator 与临时 workspace；单个请求的公式只是预算起点，不是 serving SLO。

## 9.12 一份连贯的最小实现

前面的片段分别展示了局部组件。下面把它们合成一份可运行实现，并特别处理两个容易被局部代码掩盖的条件：RoPE 的位置偏移，以及 `query_len != key_len` 时的 causal mask。代码没有 dropout，便于做 prefill/decode 一致性测试；生产模型应按自己的训练配置加入 dropout、初始化和其他结构。

```python
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int = 32
    hidden_size: int = 16
    num_layers: int = 2
    num_heads: int = 4
    intermediate_size: int = 32
    max_position_embeddings: int = 16
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2 != 0:
            raise ValueError("head_dim must be even")
        if min(self.vocab_size, self.hidden_size, self.num_layers,
               self.num_heads, self.intermediate_size,
               self.max_position_embeddings) <= 0:
            raise ValueError("model dimensions and position limit must be positive")
        if self.norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("norm_eps and rope_theta must be positive")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must be floating point")
        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_compute = x.to(compute_dtype)
        variance = x_compute.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_compute * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.to(compute_dtype)).to(input_dtype)


def make_causal_mask(query_len, key_len, device, past_key_values_length=0):
    if min(query_len, key_len, past_key_values_length) < 0:
        raise ValueError("mask lengths must be non-negative")
    if key_len != past_key_values_length + query_len:
        raise ValueError("key_len must equal past_len + query_len")
    query_positions = torch.arange(query_len, device=device)[:, None]
    key_positions = torch.arange(key_len, device=device)[None, :]
    allowed = key_positions <= query_positions + past_key_values_length
    return allowed[None, None]


def scaled_dot_product_attention(
    q, k, v, causal=True, past_key_values_length=0, return_attn=False
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape [B, H, T, D]")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    if q.shape[:2] != k.shape[:2] or q.size(-1) != k.size(-1):
        raise ValueError("q, k and v batch/head dimensions are incompatible")
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        if k.size(-2) != past_key_values_length + q.size(-2):
            raise ValueError("key_len must equal past_len + query_len")
        mask = make_causal_mask(
            q.size(-2), k.size(-2), q.device, past_key_values_length
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    context = attn @ v
    if return_attn:
        return context, attn
    return context


def build_rope_cache(seq_len, head_dim, device, base=10_000.0, start_pos=0):
    if seq_len < 0 or start_pos < 0:
        raise ValueError("seq_len and start_pos must be non-negative")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even number")
    if base <= 0:
        raise ValueError("RoPE base must be positive")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(
        start_pos, start_pos + seq_len, device=device, dtype=torch.float32
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos()[None, None], freqs.sin()[None, None]


def apply_rope(x, cos, sin):
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)


class CachedRoPEAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _shape(self, x):
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x,
        past_key_value=None,
        use_cache=False,
        position_offset=None,
        return_attn=False,
    ):
        if x.ndim != 3 or x.size(-1) != self.hidden_size:
            raise ValueError("x must have shape [B, T, hidden_size]")
        batch_size, seq_len, _ = x.shape
        q = self._shape(self.q_proj(x))
        k_new = self._shape(self.k_proj(x))
        v_new = self._shape(self.v_proj(x))

        if past_key_value is None:
            past_len = 0
        else:
            past_k, past_v = past_key_value
            if past_k.shape != past_v.shape:
                raise ValueError("cached key and value must have the same shape")
            if past_k.ndim != 4 or past_k.size(0) != batch_size:
                raise ValueError("cache must have shape [B, H, T_cache, D_h]")
            if past_k.size(1) != self.num_heads or past_k.size(3) != self.head_dim:
                raise ValueError("cache head dimensions do not match the model")
            if past_k.device != x.device or past_v.device != x.device:
                raise ValueError("cache and input must be on the same device")
            if past_k.dtype != x.dtype or past_v.dtype != x.dtype:
                raise ValueError("cache and input must use the same dtype")
            past_len = past_k.size(2)

        if position_offset is None:
            position_offset = past_len
        if position_offset != past_len:
            raise ValueError("position_offset must equal the contiguous cache length")

        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            x.device,
            base=self.rope_theta,
            start_pos=position_offset,
        )
        q = apply_rope(q, cos, sin)
        k_new = apply_rope(k_new, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k_new), dim=2)
            v = torch.cat((past_v, v_new), dim=2)
        else:
            k, v = k_new, v_new

        context, attn = scaled_dot_product_attention(
            q,
            k,
            v,
            causal=True,
            past_key_values_length=past_len,
            return_attn=True,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        output = self.o_proj(context)
        present = (k, v) if use_cache else None
        return output, present, (attn if return_attn else None)


class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = CachedRoPEAttention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, past_key_value=None, use_cache=False,
                position_offset=None, return_attn=False):
        attn_out, present, attn = self.self_attn(
            self.input_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_offset=position_offset,
            return_attn=return_attn,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attn_norm(x))
        return x, present, attn


class MiniDecoderLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids,
        labels=None,
        past_key_values=None,
        use_cache=False,
        position_offset=None,
        output_attentions=False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use int32 or int64 dtype")
        if input_ids.device != self.embed_tokens.weight.device:
            raise ValueError("input_ids and model parameters must be on the same device")
        batch_size, seq_len = input_ids.shape
        if batch_size == 0 or seq_len == 0:
            raise ValueError("input_ids must contain at least one batch item and token")
        if input_ids.min().item() < 0 or input_ids.max().item() >= self.config.vocab_size:
            raise ValueError("input_ids contain a token outside [0, vocab_size)")
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError("one cache entry is required for every layer")
            has_cache = [item is not None for item in past_key_values]
            if any(has_cache) and not all(has_cache):
                raise ValueError("cache must be present for every layer or none")
            model_device = self.embed_tokens.weight.device
            cache_lengths = []
            for layer_index, item in enumerate(past_key_values):
                if item is None:
                    continue
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(f"cache entry {layer_index} must be a (key, value) pair")
                key, value = item
                if key.ndim != 4 or key.shape != value.shape:
                    raise ValueError(
                        f"cache entry {layer_index} must have matching 4-D key/value tensors"
                    )
                if key.size(0) != batch_size or key.size(1) != self.config.num_heads:
                    raise ValueError(f"cache entry {layer_index} batch/head shape is invalid")
                if key.size(3) != self.config.hidden_size // self.config.num_heads:
                    raise ValueError(f"cache entry {layer_index} head dimension is invalid")
                if key.device != model_device or value.device != model_device:
                    raise ValueError("cache and model parameters must be on the same device")
                cache_lengths.append(key.size(2))
            if cache_lengths and len(set(cache_lengths)) != 1:
                raise ValueError("all cache layers must have the same sequence length")
            past_len = cache_lengths[0] if cache_lengths else 0

        if position_offset is None:
            position_offset = past_len
        if not isinstance(position_offset, int) or position_offset < 0:
            raise ValueError("position_offset must be a non-negative Python int")
        if position_offset != past_len:
            raise ValueError("position_offset must match cache length")
        if position_offset + seq_len > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype != torch.int64:
                raise ValueError("labels must use int64 dtype")
            if labels.device != input_ids.device:
                raise ValueError("labels and input_ids must be on the same device")
            invalid_labels = (labels != -100) & (
                (labels < 0) | (labels >= self.config.vocab_size)
            )
            if invalid_labels.any().item():
                raise ValueError("labels must be -100 or lie in [0, vocab_size)")
            if not (labels[:, 1:] != -100).any().item():
                raise ValueError("labels contain no valid next-token targets")
            if any(item is not None for item in past_key_values):
                raise ValueError("this demo computes labels only for a fresh sequence")

        x = self.embed_tokens(input_ids)
        presents = []
        attentions = []
        for layer, past in zip(self.layers, past_key_values):
            x, present, attn = layer(
                x,
                past_key_value=past,
                use_cache=use_cache,
                position_offset=position_offset,
                return_attn=output_attentions,
            )
            if use_cache:
                presents.append(present)
            if output_attentions:
                attentions.append(attn)

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

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": tuple(presents) if use_cache else None,
            "attentions": tuple(attentions) if output_attentions else None,
        }
```
> **已验证：** 2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU。与下一代码块合并为一个脚本运行；覆盖前向、loss、shape、mask、RoPE、权重绑定和 prefill/decode 对照。未验证 PyTorch 2.15 runtime、CUDA 或其他设备。

这份代码有几个值得特别注意的地方。

第一，cache 中的 key 已经是经过 RoPE 的历史 key；只有 `k_new` 使用当前的 `position_offset` 旋转。第二，`past_len` 同时参与 cache mask 和位置编号，二者不能各自猜一个长度。第三，`labels` 与 cache decode 没有在这个最小实现中混用，因为带历史的 logits 和 labels 的 shift 语义需要由调用者明确规定；主动报错比静默算出错误 loss 更安全。第四，`output_attentions=True` 只为审计服务，不应在高并发推理路径上默认打开。

入口范围检查同样是教学版的取舍。`input_ids.min().item()`、`labels.any().item()` 会把设备上的判断结果带回 Python；在 CUDA 上这可能触发同步。如果每一步都对完整 batch 做所有取值检查，检查本身会进入训练关键路径。本实现把它们放在 forward 中，是为了让读者得到清楚、靠近错误边界的异常；正式训练通常在数据落盘或 CPU collate 阶段验证一次，运行中保留 shape/dtype 等廉价断言，再通过小 batch 的 debug 模式打开逐值检查。这里的 CPU demo 不能用于估计 CUDA 训练吞吐。

## 9.13 审计 demo：不只检查能不能跑

一个 forward 成功只能证明 Python 路径没有抛异常，不能证明模型语义正确。下面的测试分别检查：

1. RMSNorm 输出形状和均方值。
2. causal mask 的方向。
3. attention 行和是否为 `1`，未来权重是否为 `0`。
4. RoPE 是否保持二维子空间范数。
5. logits、loss 和绑定权重。
6. prefill 后逐步 decode 是否与一次性 forward 的对应位置一致。
7. decode 的 query 是否真正看到全部历史 cache。

```python
torch.manual_seed(7)
torch.set_num_threads(1)

config = TransformerConfig()
model = MiniDecoderLM(config).eval()
batch_size, seq_len = 2, 5
input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
labels = input_ids.clone()
x = model.embed_tokens(input_ids)

with torch.no_grad():
    # 1. RMSNorm
    rms = RMSNorm(config.hidden_size)
    rms_out = rms(x)
    rms_mean_square = rms_out.pow(2).mean(dim=-1)[0, 0].item()
    rms_dtype_checks = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        norm = RMSNorm(4).to(dtype=dtype)
        sample = torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=dtype)
        normalized = norm(sample)
        rms_dtype_checks[str(dtype)] = (
            normalized.dtype == dtype and torch.isfinite(normalized).all().item()
        )

    # 2-3. mask 和 attention
    mask = make_causal_mask(4, 4, x.device)
    mask_rows = mask[0, 0].int().tolist()
    cached_block_mask_rows = make_causal_mask(
        2, 5, x.device, past_key_values_length=3
    )[0, 0].int().tolist()
    q = torch.randn(batch_size, config.num_heads, seq_len, config.hidden_size // config.num_heads)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    context, attn = scaled_dot_product_attention(q, k, v, return_attn=True)
    future_mask = ~make_causal_mask(seq_len, seq_len, x.device)
    future_weight_max = attn.masked_select(future_mask).max().item()
    attn_row_sums = attn[0, 0].sum(dim=-1).tolist()

    # 4. RoPE 的范数保持
    cos, sin = build_rope_cache(
        seq_len, config.hidden_size // config.num_heads, x.device
    )
    rope_out = apply_rope(q, cos, sin)
    rope_norm_preserved = torch.allclose(
        q.norm(dim=-1), rope_out.norm(dim=-1), atol=1e-5
    )

    # 5. 完整序列、loss 和权重绑定
    full = model(input_ids, labels=labels, use_cache=True, output_attentions=True)
    tied_weights = (
        model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    )

    # 6-7. 前四个 token 做 prefill，最后一个 token 做 decode
    prefix = model(input_ids[:, :4], use_cache=True, output_attentions=True)
    decoded = model(
        input_ids[:, 4:],
        past_key_values=prefix["past_key_values"],
        use_cache=True,
        output_attentions=True,
    )
    prefill_decode_match = torch.allclose(
        full["logits"][:, 4:], decoded["logits"], atol=1e-5, rtol=1e-5
    )
    decode_attn = decoded["attentions"][0]
    decode_can_see_all_cache = decode_attn.shape[-1] == 5

summary = {
    "rms_shape": tuple(rms_out.shape),
    "rms_mean_square_first": round(rms_mean_square, 4),
    "rms_dtype_checks": rms_dtype_checks,
    "mask_rows": mask_rows,
    "cached_block_mask_rows": cached_block_mask_rows,
    "context_shape": tuple(context.shape),
    "attn_row_sums": [round(value, 4) for value in attn_row_sums],
    "future_weight_max": round(future_weight_max, 6),
    "rope_norm_preserved": bool(rope_norm_preserved),
    "logits_shape": tuple(full["logits"].shape),
    "loss": round(full["loss"].item(), 4),
    "tied_weights": tied_weights,
    "prefill_decode_match": bool(prefill_decode_match),
    "decode_attn_shape": tuple(decode_attn.shape),
    "decode_can_see_all_cache": decode_can_see_all_cache,
}

checks = {
    "rms_shape": summary["rms_shape"] == (2, 5, 16),
    "rms_scale": abs(summary["rms_mean_square_first"] - 1.0) < 1e-3,
    "rms_supported_dtypes": all(rms_dtype_checks.values()),
    "mask_direction": mask_rows == [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    "cached_block_mask": cached_block_mask_rows == [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ],
    "attention_shape": summary["context_shape"] == (2, 4, 5, 4),
    "attention_rows_sum_to_one": all(
        abs(value - 1.0) < 1e-5 for value in attn_row_sums
    ),
    "future_mask_zero": summary["future_weight_max"] == 0.0,
    "rope_norm": summary["rope_norm_preserved"],
    "lm_shape": summary["logits_shape"] == (2, 5, 32),
    "finite_loss": torch.isfinite(full["loss"]).item(),
    "weight_tying": summary["tied_weights"],
    "prefill_decode": summary["prefill_decode_match"],
    "decode_cache_visible": summary["decode_can_see_all_cache"],
}

print("summary=", summary)
print("checks=", checks)
all_checks_pass = all(checks.values())
print("all_checks_pass=", all_checks_pass)
if not all_checks_pass:
    raise AssertionError(f"audit checks failed: {checks}")
```
> **已验证：** 2026-11-11，Python 3.12.9 / PyTorch 2.12.0+cpu，CPU。完整审计 demo 的 dtype、mask、attention、RoPE、loss、cache 与 prefill/decode 检查全部通过；结果不外推到 CUDA 或 PyTorch 2.15。

最有价值的检查是 `prefill_decode_match`。在没有 dropout、参数没有变化的前提下，把同一段序列一次性计算，和“前缀 prefill + 最后一个 token decode”计算，最后位置的 logits 应近似相等。这个测试同时覆盖了：历史 K/V 拼接、RoPE 的位置偏移、非方阵 causal mask 和每层 cache 的传递。仅检查 `cache.shape` 增长到 `5`，无法发现位置旋转错位或 mask 只开放了第一个 key 的问题。

在低精度、不同 kernel 或不同设备上，`allclose` 的阈值需要按 dtype 调整。验证目标是语义一致，而不是要求所有浮点数逐位相等。

这些检查覆盖的是相互独立的语义不变量：mask 方向测试关注“谁可见”，future weight 关注 softmax 之后被禁止的位置确实没有概率，RoPE 范数关注二维旋转的正交性，权重指针关注 Parameter 是否真正共享，prefill/decode 对照则同时关注位置偏移、历史 K/V 追加和非方形 mask。把它们压成一个“loss 有限”检查会漏掉大量错误，因为错误 mask、重复旋转或未绑定参数都可能仍产生有限 loss。

开发者还可以按最早失败的不变量缩小范围：若 shape 断言失败，先看 config 与 reshape；若 attention 行和非 1 或未来权重非零，先看 mask 的布尔含义与 key/query 绝对位置；若 RoPE 范数改变，先看偶/奇维配对与 cos/sin 广播；若完整前向正确而 prefill/decode 不一致，比较每层 cache 长度、`position_offset` 和 RoPE 是否只作用于新 K。每次只改一个因素，然后用同一固定输入重跑全部检查。这样的分层复现，比换一套随机 prompt 后观察文本“看起来合理”更可判定。

## 9.14 常见失败模式与定位顺序

### 9.14.1 `hidden_size` 不能被 head 数整除

如果 `d % H != 0`，没有整数的 `D_h`，reshape 就没有定义。即使整除，使用本章的二维 RoPE 时还要求 `D_h` 为偶数。配置阶段直接抛出异常，比在 forward 中遇到一个难懂的 view 错误更容易定位。

### 9.14.2 mask 方向反了

正确的长度为 `4` 的允许矩阵是：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

如果写成上三角，当前位置会读取未来 token，训练 loss 可能异常偏低；如果把 bool mask 的含义写反，所有正确位置都会被屏蔽。先打印一个极小矩阵，比在大模型上观察 loss 更直接。

### 9.14.3 `query_len=1` 时仍使用普通下三角矩阵

decode 时 key 长度大于 query 长度。普通 `torch.tril(torch.ones(1, key_len))` 只会开放第一个 key，而不是开放完整历史。应按绝对 query 位置 `past_len + r` 生成 mask，并测试最后一行的可见长度。

### 9.14.4 cache 中的 K/V 被重复旋转

如果每次把完整 `past_k` 和 `k_new` 拼起来再对整个序列应用 RoPE，历史 key 会第二次旋转。结果仍有正确的 shape，甚至 loss 也可能是有限的，但 prefill/decode 一致性会失败。正确做法是只旋转新产生的 K/V，并保存已经旋转后的历史值。

### 9.14.5 decode 的位置从零重新开始

当 cache 长度为 `p` 时，新 token 的位置是 `p` 而不是 `0`。这类错误在没有 RoPE 的测试中完全看不出来，因此 cache 测试必须和位置编码一起覆盖。

### 9.14.6 transpose 后直接 `view`

`[B,H,T,D]` 合并回 `[B,T,d]` 前通常经历 `transpose(1,2)`。没有 `contiguous()` 就 `view` 可能报错，也可能在某些路径上触发隐式复制。shape 正确不代表内存布局正确。

### 9.14.7 loss 没有右移或忽略标签值不一致

`logits[:, :-1]` 与 `labels[:, 1:]` 必须对应。如果数据管线使用了别的 ignore index，却在交叉熵中仍使用 `-100`，padding 会参与训练；反过来，过多的 `-100` 会让有效 token 数变少，loss 的统计也会失真。

### 9.14.8 在低精度中用不合适的 mask 常数

`-1e30` 对 fp16 不一定可表示。使用 `torch.finfo(dtype).min` 或官方 attention API。还要避免一整行全部被 mask；全屏蔽行没有合法概率分布，可能产生 NaN 或没有意义的均匀结果。

### 9.14.9 为了观察 attention 把完整矩阵留在生产路径

attention 权重的内存是 `O(BHT_qT_k)`。在长上下文中，调试输出本身可能成为 OOM 的原因。训练和服务代码应默认不返回权重，只在极短序列、抽样 batch 或专门的 profiling 运行中打开。

把这些失败模式按修复成本排序，会形成一条实用排查路径：先验证配置整数关系和词表范围，再用一条长度 4 的手写样本打印允许矩阵；随后确认 `logits[:, :-1]` 和 `labels[:, 1:]` 的目标索引，接着以固定权重测试 RoPE 和 residual；最后再运行带 cache 的 prefill/decode 对照。比如 `p=5,q=1` 时应有一行 6 个 True；若发现只有第一个 True，问题来自 mask 对齐，不需要先训练模型来观察 loss。若 full-sequence 与 decode 最后 logits 不同，再用各层 K/V 长度、绝对位置和 dtype 三个字段定位，而不是立即怀疑 sampler。这个顺序由错误暴露点决定：越早失败的契约越先检查，避免下游的有限输出掩盖上游语义错误。

回归时至少保留三类 batch：普通无 padding 序列、右侧 padding 且至少一个有效目标、以及会触发配置或 label 错误的输入。对后两类，断言应能明确区分“没有监督目标”和“attention 看不到任何 key”；二者都可能产生 NaN，却需要不同修复。固定同一随机种子与权重可以让输出差异更容易复现，但数值回归仍应采用容差，并把 PyTorch 版本、dtype 和 backend 一并记录。

## 9.15 计算、内存与生产实现的边界

对长度为 `T` 的完整序列，attention 的 score 计算和权重存储都带有 `T^2` 因子；Q/K/V 投影和 MLP 则大致随 `T` 线性增长、随 hidden/intermediate 维度平方或乘积增长。于是短序列上 MLP 和投影可能占主要计算，长序列上 attention 的序列平方项会逐渐成为瓶颈。

KV Cache 把每个 decode step 的历史 K/V 投影从重复计算中移除，但当前 query 仍要读取长度为 `T_cache` 的 key/value，因此单步 attention 读取量仍随历史长度增长。显式 cache 的总内存还随层数、batch、历史长度和 `H_kv·D_h` 增长。降低 KV head 数、压缩 cache、分页分配或使用状态空间结构，都是不同的系统设计，不应混称为“KV Cache 优化”。

生产部署通常会：

1. 使用 `scaled_dot_product_attention` 或 FlashAttention 类 kernel，避免显式 materialize 完整 attention 矩阵。
2. 为不同请求使用分页或 block 化的 cache，减少长短请求混在一起时的碎片。
3. 将 prefill 与 decode 的调度、带宽和算力分开分析。
4. 对 cache dtype、量化、最大序列长度和 batch 上限做容量预算。
5. 用模型官方 tokenizer、chat template 和 special token 规则，而不是把教学版 `input_ids` 直接当成可互换协议。

本章的 `torch.cat` 每次都会创建新的连续张量，适合解释状态如何增长，却不是高并发服务的内存管理方案。用它测出来的速度也不能代表 vLLM、TensorRT-LLM 或其他引擎的吞吐。

用长度变化比较不同资源项，可以避免把所有开销都归到“attention 慢”。完整序列的 score 网格有 `H×T×T` 个位置；从 `T=512` 加到 `T=1024`，若 head 数不变，网格元素约从 `H×262,144` 增至 `H×1,048,576`，扩大四倍。单层 KV cache 公式中的 `T_cache` 则只扩大两倍。融合 SDPA 可以减少显式保存完整权重矩阵的内存流量，但逻辑上仍要为 query 计算与可见 key 的关系；分页 cache 改善的是请求分配和碎片管理，不会让每个仍驻留的 key/value 从容量公式中消失。GQA 把 `H_kv` 降低会减少 cache 体积，但这是模型结构变化，必须同时验证 checkpoint 支持和输出质量。

性能判断至少需要把 prefill 与 decode 分开。prefill 面对长 query，常见瓶颈是大批矩阵计算和 attention 工作量；decode 每一步 query 很短，却要读取历史 KV，可能受内存带宽和并发形态影响。一个教学实现的 `torch.cat`、Python 循环和 `output_attentions` 开关会改变被测路径，所以 benchmark 必须明确输入长度、batch、dtype、cache 长度、后端、预热和是否包含采样。更快的单步 kernel 不自动等于更高的服务吞吐，调度器、不同请求长度和 KV 容量也会参与结果。

## 9.16 练习：从局部正确走向系统正确

1. 将 `RMSNorm` 与 `nn.LayerNorm` 放在同一批输入上，比较均值、均方值和参数数量，并说明两者不能仅凭均方值相同就判定等价。
2. 写出 `B=1,H=2,T_q=2,T_k=5,past_len=3` 的 causal mask，并解释每一行可见 key 的数量。
3. 为 attention 加入显式的 `attn_mask` 参数，支持 padding mask 与 causal mask 的组合；测试两种 mask 都有效时不会出现全屏蔽行。
4. 把 `MultiHeadSelfAttention` 改成使用 PyTorch 官方 `scaled_dot_product_attention`，比较完整序列上的输出形状；再对 `T_q=1,T_k=6` 比较 `is_causal=True`、`causal_lower_right(1,6)` 与显式绝对位置 mask，并说明 lower-right API 的版本状态、bool mask 方向及 eval 时为何必须传 `dropout_p=0.0`。
5. 为 `apply_rope` 增加 `rotary_dim < head_dim` 的配置，只旋转前 `rotary_dim` 个维度，保留其余维度不变。
6. 将本章模型的普通 MHA 改为 GQA，分别记录 query head 数、KV head 数和 cache 内存公式中的变化。
7. 把 prefill 长度从 `4` 改成 `1、2、4`，逐个 token decode，并验证每个位置的增量 logits 与一次性 forward 相近。
8. 将 labels 的一部分设置为 `-100`，手工计算有效 token 的交叉熵，确认 PyTorch loss 的分母只包含未忽略的位置。
9. 故意把 RoPE 的 `start_pos` 固定为 `0`，运行 prefill/decode 一致性测试，观察为什么 shape 检查仍会通过而语义检查失败。
10. 在 `output_attentions=True` 和 `False` 两种模式下比较内存与返回对象，说明为什么可观测性应当有明确的开关。

## 9.17 本章小结

一个 decoder-only Transformer 可以沿着一条清晰的计算链理解：embedding 把整数 token id 变成 `[B,T,d]` 的表示；Q/K/V 投影把它拆成多个 head；scaled dot-product attention 在 causal mask 约束下完成位置间通信；RoPE 把位置关系写入 Q/K；SwiGLU MLP 对每个位置做非线性变换；Pre-Norm 和 residual 把这些变换组合成可堆叠的 block；final norm 与 LM head 产生 `[B,T,V]` logits；右移后的交叉熵把 logits 与下一个 token 对齐。

生成时，prefill 和 decode 共享这套参数，但不共享同一种形状。历史 K/V 被缓存后，query 变成新 token，key/value 变成历史加新 token；因此 causal mask 必须使用绝对位置，RoPE 必须使用 cache 长度作为位置偏移。一个真正有说服力的实现验证，不是“forward 没报错”，而是同时通过 shape、mask、数值稳定性、权重绑定和 prefill/decode 一致性测试。

本章的代码是教学模型，不是任何具体厂商模型的完整复刻。它展示的是一组公开、可解释的 Transformer 基本组件；遇到其他架构时，应先读取目标模型的配置、模型卡和实现，再确认它是否使用 RMSNorm、SwiGLU、RoPE、GQA、MLA、混合注意力或其他状态机制。

## 9.18 资料入口

1. Vaswani et al., [*Attention Is All You Need*（arXiv:1706.03762v7）](https://arxiv.org/abs/1706.03762v7)。用于 scaled dot-product attention、多头 attention、位置编码和 Transformer 主体公式。
2. Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*（arXiv:2104.09864v5）](https://arxiv.org/abs/2104.09864v5)。用于 RoPE 的旋转构造和相对位置信息解释。
3. PyTorch 2.15 [`scaled_dot_product_attention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) 与 [`causal_lower_right`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.attention.bias.causal_lower_right.html)。用于官方 attention API、非方形 mask、dropout 和后端说明；lower-right bias 属 prototype。
4. PyTorch 2.15 [`torch.nn.RMSNorm`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.RMSNorm.html) 与 [`torch.nn.LayerNorm`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html)。用于归一化维度、epsilon 和方差语义。
5. PyTorch 2.15 [`torch.nn.Embedding`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html) 与 [`torch.nn.MultiheadAttention`](https://docs.pytorch.org/docs/2.15/generated/torch.nn.MultiheadAttention.html)。用于 embedding 输入和 attention mask 的 API 语义。

这些入口分别支持基础数学、论文定义和 PyTorch API。实际工程还需要补充目标模型的 revision、tokenizer、chat template、dtype、attention backend 与 serving 引擎文档；不能仅凭本章的教学实现判断一个具体模型的部署行为。
