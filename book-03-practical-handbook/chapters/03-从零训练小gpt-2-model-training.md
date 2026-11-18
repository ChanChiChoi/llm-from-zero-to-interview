# 第 3 章（第二篇）：最小 GPT 与一次参数更新

字符级文本已变成成对的输入与下一个 token 标签。本篇把这对张量送进 GPT，跟踪 embedding、因果注意力、前馈层和语言模型头，最后看交叉熵怎样驱动一次优化器更新。读完后，你应能沿 `[B,T] → [B,T,D] → [B,T,V]` 检查形状，并解释 loss、梯度和 token 预算各自表示什么。

## 3.2 最小 GPT：从 token id 到训练 loss

### 3.2.1 一次前向传播经过哪些形状

字符级 GPT 的数据流可以写成：

```text
input_ids             [B, T]
token embedding       [B, T, D]
position embedding    [T, D]
Transformer blocks    [B, T, D]
final normalization   [B, T, D]
lm head               [B, T, V]
targets               [B, T]
cross entropy         scalar
```

$B$ 是 batch size，$T$ 是上下文长度，$D$ 是 hidden size，$V$ 是词表大小。Transformer block 的输入输出都保持 $[B,T,D]$，只有最后的语言模型头把 hidden 维度映射成词表类别。

沿着这条路径读代码时，可以逐层问：当前张量表示什么，下一层需要什么形状？例如 embedding 把 token id 映射成向量，attention 在序列位置之间交换信息，lm head 再把每个位置的向量映射为 $V$ 个候选分数。代码断言只检查这些接口是否对齐，不会替你判断模型是否学到了有用规律。

token embedding 矩阵 $E\in\mathbb{R}^{V\times D}$ 通过索引取得 $E_{x_{b,t},:}$。如果使用 learned position embedding，位置表 $P\in\mathbb{R}^{T_{\max}\times D}$ 提供位置向量，初始 hidden 为：

```math
H^{(0)}_{b,t,:}=E_{x_{b,t},:}+P_{t,:}
```

当 $T>T_{\max}$ 时，位置表不能直接索引；字符级教学模型通常把 block_size 同时作为最大位置长度。生产模型可能使用 RoPE 或其他位置机制，但“允许更长位置索引”和“长上下文质量不下降”仍是两个不同命题。

### 3.2.2 因果 attention 的数学对象

对一个 head，输入 $X\in\mathbb{R}^{B\times T\times D}$ 先投影为 $Q,K,V$：

```math
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
```

若 head dimension 为 $d_h$，注意力分数是：

```math
S_{b,i,j}=\frac{Q_{b,i,:}K_{b,j,:}^{\mathsf T}}{\sqrt{d_h}}
```

causal mask 令第 $i$ 个 query 只能读取 $j\le i$ 的 key：

```math
M_{i,j}=\begin{cases}
1,&j\le i\\
0,&j>i
\end{cases}
```

softmax 前对 $M_{i,j}=0$ 的位置填入当前 dtype 的最小有限值，得到权重：

```math
A_{b,i,:}=\mathrm{softmax}\left(S_{b,i,:}+\mathrm{mask}_{i,:}\right)
```

然后：

```math
O_{b,i,:}=\sum_{j=0}^{T-1}A_{b,i,j}V_{b,j,:}
```

在教学实现中，mask 采用下三角布尔矩阵。对于每一行，至少保留当前位置，因此不会因为 causal mask 单独造成全 mask 行。加入 padding、局部检索或外部过滤后，仍需要单独检查空候选情况。

### 3.2.3 Pre-LN block 与参数量估算

一个常见的 Pre-LN decoder block 可以写成：

```math
H'=H+\mathrm{Attn}(\mathrm{LN}_1(H))
```

```math
H_{\mathrm{out}}=H'+\mathrm{FFN}(\mathrm{LN}_2(H'))
```

残差让输入保留一条不经过子层变换的路径，LayerNorm 则控制子层接收的数值范围。普通 FFN 的中间维度为 $D_{\mathrm{ff}}$ 时，忽略 bias 的主要参数量约为 $2DD_{\mathrm{ff}}$；四个 $D\times D$ 的 attention 投影约为 $4D^2$。教学代码采用普通 GELU FFN，便于单独观察注意力和训练目标；换成 SwiGLU 会改变参数量，但不会改变 x 和 y 错位一位的约定。

### 3.2.4 一个完整的最小模型实现

下面的代码把 attention、FFN、block、loss 和生成组合在一起。它使用独立的 max_seq_len，而不是把全局变量藏在类内部；读者可以先运行 shape 检查，再接入 3.1 的 get_batch。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or max_seq_len <= 0:
            raise ValueError("dimensions and max_seq_len must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, hidden]")
        if not x.is_floating_point():
            raise TypeError("attention input must use a floating-point dtype")
        batch_size, seq_len, hidden_size = x.shape
        if seq_len <= 0:
            raise ValueError("sequence length must be positive")
        if hidden_size != self.d_model:
            raise ValueError("hidden size does not match the module")
        if seq_len > self.causal_mask.size(0):
            raise ValueError("sequence length exceeds max_seq_len")

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        if d_model <= 0 or expansion <= 0:
            raise ValueError("d_model and expansion must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        hidden_dim = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            num_heads,
            max_seq_len,
            dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        max_seq_len,
        dropout=0.0,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or num_heads <= 0:
            raise ValueError("vocab_size, d_model and num_heads must be positive")
        if num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("num_layers and max_seq_len must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, targets=None, ignore_index=-100):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError(
                "input_ids must be a torch.long tensor with shape [batch, sequence]"
            )
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise TypeError("ignore_index must be an integer")
        batch_size, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("sequence length must be positive")
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("input_ids contain an id outside the vocabulary")

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            if (
                targets.shape != input_ids.shape
                or targets.dtype != torch.long
            ):
                raise ValueError("targets must have the same shape as input_ids")
            invalid_targets = (targets != ignore_index) & (
                (targets < 0) | (targets >= self.vocab_size)
            )
            if invalid_targets.any():
                raise ValueError("targets contain an id outside the vocabulary")
            if not (targets != ignore_index).any():
                raise ValueError("targets contain no valid prediction positions")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=ignore_index,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError(
                "input_ids must be a torch.long tensor with shape [batch, sequence]"
            )
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer")
        if (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError("input_ids contain an id outside the vocabulary")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = input_ids[:, -self.max_seq_len:]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
            return input_ids
        finally:
            if was_training:
                self.train()


torch.manual_seed(42)
model = TinyGPT(
    vocab_size=32,
    d_model=32,
    num_heads=4,
    num_layers=2,
    max_seq_len=16,
)
input_ids = torch.randint(0, 32, (2, 16), dtype=torch.long)
targets = torch.randint(0, 32, (2, 16), dtype=torch.long)
logits, loss = model(input_ids, targets)

assert logits.shape == (2, 16, 32)
assert loss.ndim == 0
print("logits_shape=", tuple(logits.shape))
print("loss_is_finite=", bool(torch.isfinite(loss)))
```
这段代码只实现了一个教学模型：没有 KV Cache、混合精度、分布式并行、权重共享或高效 attention kernel。它的价值是把每个 shape 和每个残差路径暴露出来；在后续换成 fused kernel 后，应保留同样的局部 shape 和数值回归测试。

### 3.2.5 交叉熵与 token 数量

若 logits 为 $Z\in\mathbb{R}^{B\times T\times V}$，标签为 $Y\in\{0,\ldots,V-1\}^{B\times T}$，且 $B>0,T>0$，平均 token loss 为：

```math
L(\theta)=-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}\log p_\theta(Y_{b,t}\mid X_{b,\le t})
```

等价地，用未归一化 logits 表示：

```math
L(\theta)=\frac{1}{BT}\sum_{b,t}\left(\log\sum_{c=0}^{V-1}\exp Z_{b,t,c}-Z_{b,t,Y_{b,t}}\right)
```

PyTorch 2.12 的 `F.cross_entropy` 接收未归一化 logits，类别索引标签使用 `torch.long`，默认 reduction 为对有效损失取均值；不需要先手动 softmax：[官方接口文档](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.cross_entropy.html)。展平只是把每个位置变成一个分类样本：

```text
[B, T, V] -> [B*T, V]
[B, T]   -> [B*T]
```

对于没有 padding 的固定长度 block，分母是 $BT$。如果以后将多个长度不同的序列 padding 到同一长度，padding 位置必须从目标 `targets` 中排除，例如把对应位置设为 `ignore_index=-100` 并传给 `F.cross_entropy`；输入 `input_ids` 仍然必须全部是合法词表 id。本章模型前向会跳过目标中的 `ignore_index` 范围检查，但会拒绝所有目标都被忽略的 batch，因为这时平均 loss 没有定义，不能把它当作有效训练信号。

初始 logits 如果近似均匀，单 token loss 约为：

```math
L_0\approx\log V
```

这是一个数量级检查，不是所有初始化都必然满足的定律。若 $V=32$，$\log 32\approx3.466$。初始 loss 是几十、NaN 或立刻发散时，应先查 labels 范围、logits shape、学习率、mask 和数据 dtype。

### 3.2.6 训练循环：每一步究竟更新了什么

最小训练循环的顺序是：

```text
取 batch -> forward -> 计算 loss -> 清理旧梯度
-> backward -> 可选梯度裁剪 -> optimizer.step
```

下面的示例把数据适配器、评估函数和训练循环放在一起。它会用到 3.1.5 的 `get_batch_from_source`，以及 3.2.4 中定义并初始化的 `TinyGPT`；请在运行过前面相应代码段后接着执行。这里的 `get_batch(split)` 负责选择数据，返回 `(input_ids, targets)`；两个张量必须在同一设备上，或能转到同一设备上。`split` 只能是 `train` 或 `val`。示例用两条合成整数序列演示如何选择数据，不能用于判断泛化能力；真实项目应换成按文档或其他独立对象切分的数据。

> 代码状态：静态审阅，未运行（2026-10-29）

```python
train_tokens = torch.arange(64, dtype=torch.long) % 32
val_tokens = torch.arange(32, dtype=torch.long).flip(0)
split_sources = {"train": train_tokens, "val": val_tokens}


def get_batch(split):
    if split not in split_sources:
        raise ValueError("split must be train or val")
    return get_batch_from_source(
        split_sources[split],
        batch_size=2,
        block_size=16,
        device="cpu",
    )


def estimate_loss(model, get_batch, eval_iters, device):
    if not isinstance(eval_iters, int) or eval_iters <= 0:
        raise ValueError("eval_iters must be a positive integer")
    result = {}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for split in ("train", "val"):
                values = []
                for _ in range(eval_iters):
                    input_ids, targets = get_batch(split)
                    input_ids = input_ids.to(device)
                    targets = targets.to(device)
                    _, loss = model(input_ids, targets)
                    values.append(loss.item())
                result[split] = sum(values) / len(values)
    finally:
        if was_training:
            model.train()
    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
max_steps = 200
eval_interval = 50

for step in range(max_steps):
    if step % eval_interval == 0:
        metrics = estimate_loss(model, get_batch, eval_iters=10, device=device)
        print(
            f"step={step} "
            f"train_loss={metrics['train']:.4f} "
            f"val_loss={metrics['val']:.4f}"
        )

    input_ids, targets = get_batch("train")
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    _, loss = model(input_ids, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```
`zero_grad` 必须发生在下一次反向传播前，否则梯度会默认累加。梯度裁剪应放在 backward 之后、step 之前；若使用 AMP，则先 `scaler.unscale_(optimizer)` 再裁剪。`model.eval()` 只改变 Dropout、BatchNorm 等特定模块的训练/评估行为，不会关闭自动求导；验证循环仍需 `torch.no_grad()`：[PyTorch 2.12 Module](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html) 与 [no_grad](https://docs.pytorch.org/docs/2.12/generated/torch.no_grad.html)。

训练步数、看到的 token 数和更新次数不是同一个量。若每一步使用 $B$ 条长度为 $T$ 的样本，忽略丢弃和重复采样，单步 token 数约为：

```math
N_{\mathrm{token/step}}=B T
```

实际报告中应同时记录 optimizer steps 和 `tokens_seen`。这里它表示累计处理过的 token 位置数；随机窗口可重复抽到同一语料位置，因此不等于唯一 token 数。两个实验都训练 1000 步，但一个使用 $B=8,T=32$，另一个使用 $B=64,T=128$，名义处理量分别是 25.6 万和 819.2 万 token；步数相同并不代表预算相同。

### 3.2.7 生成是另一个计算协议

训练时一次前向会为所有位置计算 logits；生成时只需要最后一个位置的 logits。对当前序列 $x_{0:t}$，生成下一个 token 的步骤是：

```math
p_{t+1}=\mathrm{softmax}\left(\frac{z_t}{\tau}\right),\qquad
x_{t+1}\sim p_{t+1}
```

其中 $\tau>0$ 是 temperature。生成后把新 token 拼回序列，重复直到达到预算或遇到停止条件。教学模型使用 learned position embedding 时，超过最大位置只能截取最近上下文；这会丢失更早信息，但可以避免位置索引越界。

生成函数应保存并恢复模型的训练状态。若调用前模型在训练模式，生成结束后应回到训练模式；如果中途抛出异常，也不能把模型永久留在 eval 模式。前面 TinyGPT.generate 已经展示了这种 try/finally 写法。

另一个容易混淆的事实是：生成文本的随机性来自采样器和随机状态，不是来自训练 loss 本身。要比较两个 checkpoint 的生成质量，应固定 prompt、生成长度、temperature、top-k/top-p、随机种子和 decode 规则；否则看到的差异可能只是随机采样差异。


用极小 shape 核对一次前向：令 $B=2,T=4,V=8,D=4$，输入是 $[2,4]$，embedding 和 attention 的输出都是 $[2,4,4]$，lm head 输出 $[2,4,8]$。这包含 64 个 logits 标量，却只有 8 个分类位置；每个位置的一组 8 个分数对应一个右移标签，标签必须落在 $0$ 到 $7$。例如，两个位置给正确 token 的概率分别是 0.5 和 0.25，它们的平均交叉熵为 $(-\ln0.5-\ln0.25)/2\approx1.040$ nat。打印输入、对应标签和输出形状，通常比只看最后一个 loss 更容易发现错位。

固定一小批样本重复训练，是检查前向、loss、梯度和优化器能否连通的办法。训练 loss 若不下降，先查标签错位、梯度是否为空、优化器是否拿到了模型参数，以及学习率是否合理。即使这 8 个目标位置被完全记住，也只证明模型能拟合这批样本；泛化要在独立窗口上评估。

因果 mask 则决定训练和生成时模型看到的上下文是否一致。长度为 3 时，三个 query 位置允许读取的 key 分别是 $[0]$、$[0,1]$、$[0,1,2]$。局部检查后还可做反事实测试：固定前缀，只修改未来 token，比较较早位置的 logits；若它们变化，说明未来信息仍进入了计算图。这个检查比“loss 看起来正常”更直接地验证了因果约束。



到这里，模型已经能把上下文变成下一个 token 的概率分布。下一篇暂时固定模型参数，专门看 temperature、top-k 和 top-p 如何决定从这个分布中生成什么。
