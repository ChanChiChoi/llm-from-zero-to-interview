# 第六章：Residual、LayerNorm、RMSNorm、Pre-Norm 与训练稳定性

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修以公开论文和技术报告为资料边界：ResNet 对 residual learning 的优化动机，Layer Normalization 对单样本 hidden 维统计的定义，Transformer 原论文中的 residual connection + LayerNorm，RMSNorm 对 re-centering / re-scaling 的拆分，Xiong 等人对 Post-LN / Pre-LN 和 warmup 关系的分析，Admin 对 residual branch 依赖和扰动放大的讨论，DeepNet / DeepNorm 对极深 Transformer 残差缩放与初始化的设计，以及 LLaMA 等公开模型对 RMSNorm / Pre-Norm 风格结构的采用。

本章只讨论公开可验证的 residual、normalization placement、warmup 和训练稳定性机制；不把某个模型的私有初始化、学习率 schedule、loss spike 处理或混合精度 recipe 写成通用标准，也不把 toy demo 的门禁结果当成真实训练 benchmark。

第二轮补强重点有三点：

1. 将 Residual、LayerNorm、RMSNorm、Pre-LN、Post-LN、warmup、residual scaling 和稳定性门禁改成稳定 MathJax 表达。
2. 明确 norm placement 不是孤立技巧，而是和残差主路径、初始化、学习率、optimizer、precision、数据 outlier 共同决定训练稳定性。
3. 补一个 0 依赖 Python demo，审计 residual stream 尺度、update ratio、mean drift 和 warmup 下的 early update pressure。

## 6.1 本章定位

前面几章讲了 attention 和 FFN。现在进入一个更容易被低估、但对大模型训练极其关键的主题：残差连接和归一化。

如果没有 Residual、LayerNorm/RMSNorm、合适的 norm placement、初始化和学习率策略，深层 Transformer 很容易训练不稳定：loss spike、梯度爆炸、早期发散、需要很长 warmup、深层模型无法收敛。

本章要回答的问题是：

1. Residual connection 为什么能让深层网络更容易训练。
2. LayerNorm 到底归一化什么，为什么适合 Transformer。
3. RMSNorm 为什么在现代 LLM 中常见。
4. Post-LN 和 Pre-LN Transformer 有什么区别。
5. 为什么 Post-LN 往往更依赖 warmup，而 Pre-LN 更稳定。
6. DeepNorm、Admin 等方法想解决什么问题。
7. 大模型训练中 norm、residual、初始化、学习率之间有什么关系。

这章的核心观点是：

```text
Transformer 能堆深，不只是因为 attention 强，还因为 residual path 和 normalization 让信息与梯度能稳定流动。
```

## 6.2 资料来源和可信边界

本章主要参考以下公开资料：

1. He et al., 2015, *Deep Residual Learning for Image Recognition*。提出 residual learning，说明残差结构让非常深的网络更容易优化。
2. Ba et al., 2016, *Layer Normalization*。提出 LayerNorm，对单个样本的层内神经元活动做归一化，训练和测试行为一致，适合序列模型。
3. Vaswani et al., 2017, *Attention Is All You Need*。原始 Transformer 使用 residual connection + LayerNorm。
4. Zhang and Sennrich, 2019, *Root Mean Square Layer Normalization*。提出 RMSNorm，去掉 re-centering，仅保留 re-scaling，计算更简单，并能取得接近 LayerNorm 的效果。
5. Xiong et al., 2020, *On Layer Normalization in the Transformer Architecture*。分析 Post-LN 和 Pre-LN，说明 LayerNorm 位置影响初始化时梯度行为，Pre-LN 有助于减少 warmup 依赖。
6. Liu et al., 2020, *Understanding the Difficulty of Training Transformers*。分析 Transformer 训练困难，指出残差分支依赖和扰动放大问题，并提出 Admin 初始化。
7. Wang et al., 2022, *DeepNet: Scaling Transformers to 1,000 Layers*。提出 DeepNorm 和配套初始化，尝试结合 Post-LN 性能和 Pre-LN 稳定性，把 Transformer 扩展到极深层。
8. Touvron et al., 2023, *LLaMA*。LLaMA 等现代 LLM 使用 RMSNorm、Pre-Norm 风格结构，成为开源 LLM 常见 recipe。

需要注意：不同模型的 norm placement、residual scaling、初始化、optimizer、learning rate warmup、precision、parallelism 会互相影响。不能孤立地说某个 norm “一定最好”。

## 6.3 Residual Connection 的直觉

残差连接来自 ResNet 的核心思想。

普通深层网络每层学习：

```math
y=F(x)
```

残差网络让层学习：

```math
y=x+F(x)
```

也就是说，模型不用从零学习完整映射，只需要学习输入上的增量修正。

这有几个好处。

第一，信息直通。

即使 `F(x)` 初始很差，`x` 仍然可以直接传到下一层。

第二，梯度更容易传播。

反向传播时，梯度可以沿 identity path 流动，不必完全穿过复杂非线性模块。

第三，深层网络更容易接近恒等映射。

如果某些层暂时不需要复杂变换，可以让 `F(x)` 接近 0。

Transformer 中每个子层通常都有残差：

```math
h'=h+\mathrm{Attention}(h)
```

```math
h_{\mathrm{out}}=h'+\mathrm{FFN}(h')
```

这让 Transformer 可以逐层在 residual stream 上累积信息。

## 6.4 Residual Stream：现代 Transformer 的主干

可以把 Transformer 中的隐藏状态看作 residual stream。

每个 block 做的事情是：

```text
residual stream
-> attention reads/writes information
-> residual stream
-> FFN processes channel features
-> residual stream
```

Attention 和 FFN 都不是完全替换 hidden state，而是在 residual stream 上写入增量。

这解释了为什么 Transformer 中的残差特别重要：

1. 各层可以不断向同一个信息流写入特征。
2. 早期层的信息有机会保留到后期层。
3. 梯度有较短路径回到浅层。
4. 多个模块像在共同编辑一个表示空间。

但残差也带来问题：如果每层写入的增量尺度不受控，表示会随着层数变深而漂移或爆炸。

这就是 normalization 和 residual scaling 的作用。

## 6.5 LayerNorm 归一化什么

LayerNorm 对每个样本、每个 token 的 hidden dimension 做归一化。

假设一个 token 表示是：

```text
x: [hidden_size]
```

LayerNorm 计算：

```math
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i
```

```math
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
```

```math
\mathrm{LN}(x)_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
```

它和 BatchNorm 的关键区别是：

1. BatchNorm 使用 batch 维度统计。
2. LayerNorm 使用单个样本内部 hidden 维度统计。
3. LayerNorm 训练和推理计算一致。
4. LayerNorm 不依赖 batch size。

这对 NLP/LLM 很重要，因为：

1. 序列长度可变。
2. batch size 可能很小或动态变化。
3. 自回归推理时一次可能只处理一个 token。
4. 分布式训练中 batch 统计同步成本高。

所以 Transformer 使用 LayerNorm 而不是 BatchNorm 是非常自然的选择。

## 6.6 RMSNorm：去掉均值中心化

RMSNorm 是 LayerNorm 的简化变体。

LayerNorm 做两件事：

```text
re-centering: 减去均值
re-scaling: 除以标准差
```

RMSNorm 只做 re-scaling：

```math
\mathrm{rms}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
```

```math
\mathrm{RMSNorm}(x)_i=w_i\frac{x_i}{\mathrm{rms}(x)}
```

它不减均值，也通常没有 bias。

RMSNorm 的直觉：

1. 控制向量尺度。
2. 保留方向信息。
3. 计算更简单。
4. 对大模型训练足够有效。

RMSNorm 论文认为 LayerNorm 中的 re-centering invariance 可能不是必需的，而 re-scaling invariance 更关键。现代 LLM 如 LLaMA 使用 RMSNorm，使它成为开源 LLM 常见组件。

## 6.7 LayerNorm 和 RMSNorm 的代码

简化 LayerNorm：

```python
import torch
import torch.nn as nn


class SimpleLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias
```

简化 RMSNorm：

```python
class SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

对比：LayerNorm 要算均值和方差；RMSNorm 只算均方根。

## 6.8 Post-LN Transformer

原始 Transformer 常被描述为 Post-LN 结构。

一个子层可以写成：

```math
h_{\ell+1}=\mathrm{LN}\bigl(h_\ell+F_\ell(h_\ell)\bigr)
```

也就是说，先做子层和残差相加，再做 LayerNorm。

Post-LN 的优点：

1. 每个 block 输出都被归一化。
2. 早期 Transformer 和很多 encoder-decoder 模型中表现很好。
3. 某些设置下最终性能可能较好。

Post-LN 的问题：

1. 深层训练更容易不稳定。
2. 初始化时靠近输出层的梯度可能较大。
3. 通常更依赖 learning rate warmup。
4. 堆很深时优化更困难。

Xiong 等人的分析指出，Post-LN 中 LayerNorm 的位置会导致初始化时部分梯度行为不佳，因此需要 warmup 避免早期大步更新导致不稳定。

## 6.9 Pre-LN Transformer

Pre-LN 结构把 norm 放到子层内部，也就是子层输入前：

```math
h_{\ell+1}=h_\ell+F_\ell(\mathrm{LN}(h_\ell))
```

如果是 RMSNorm，就是：

```math
h_{\ell+1}=h_\ell+F_\ell(\mathrm{RMSNorm}(h_\ell))
```

Pre-LN 的优点：

1. 残差主路径更接近 identity。
2. 梯度更容易沿 residual path 传播。
3. 深层训练更稳定。
4. 对 warmup 的依赖可能更小。
5. 现代 decoder-only LLM 常用。

Pre-LN 的代价：

1. 每个子层输出后不立即归一化，residual stream 尺度需要其他机制控制。
2. 某些任务或深度下，最终性能和 Post-LN/改进 Post-LN 有 trade-off。
3. 通常还会在最终输出前加 final norm。

现代 LLM 常见结构是：

```text
h = h + Attention(RMSNorm(h))
h = h + MLP(RMSNorm(h))
logits = LMHead(RMSNorm(h))
```

这就是 LLaMA-like 模型常见模式。

## 6.10 为什么 Warmup 重要

Transformer 训练常用 learning rate warmup：

```text
学习率从很小逐步升到目标值，再衰减。
```

Warmup 的作用是避免训练初期参数还没稳定时直接用大步长更新。

一个简单线性 warmup 可以写成：

```math
\eta_t=\eta_{\max}\frac{t}{T_{\mathrm{warmup}}},\qquad 1\le t\le T_{\mathrm{warmup}}
```

其中 `eta_max` 是峰值学习率，`T_warmup` 是 warmup 步数。直觉上，warmup 把早期有效更新幅度压小，给参数尺度、AdamW 动量状态、混合精度和 residual stream 留出稳定时间。

在 Post-LN Transformer 中，初始化时某些层的梯度可能偏大。如果一开始用大学习率，容易导致训练不稳定。

Pre-LN 改善了梯度行为，因此对 warmup 的依赖可能降低。

但这不代表现代 LLM 不需要 warmup。大模型训练还涉及：

1. AdamW 状态建立。
2. 混合精度数值稳定。
3. 数据分布复杂。
4. 大 batch 训练。
5. 分布式通信误差。
6. MoE routing 稳定性。

所以实际工程中，warmup 仍然是常见保险措施。

## 6.11 DeepNorm：极深 Transformer 的残差缩放

DeepNet/DeepNorm 想解决的问题是：如何把 Transformer 堆到极深层，例如 1000 层。

它提出一种新的 normalization/residual scaling 方式，并配合理论推导的初始化，使模型更新保持有界。

DeepNorm 的目标可以理解为：

```text
既想要 Post-LN 的性能，又想要 Pre-LN 的训练稳定性。
```

它说明一个事实：当层数极深时，简单的 residual + norm 可能不够，需要对 residual 分支尺度和初始化做系统设计。

虽然当前主流 decoder-only LLM 不一定直接使用 DeepNorm，但它对面试很有价值，因为它体现了深层 Transformer 训练的本质问题：

```text
每层写入 residual stream 的更新幅度必须被控制，否则深度一大就会不稳定。
```

## 6.12 Admin：残差依赖和扰动放大

*Understanding the Difficulty of Training Transformers* 分析了 Transformer 训练困难，并指出一个重要现象：模型对 residual branch 的依赖会影响训练稳定性。

如果模型过度依赖 residual branch，小参数扰动可能被放大，造成输出显著变化，训练不稳定。

但如果 residual branch 依赖太弱，模型能力又受限。

Admin 初始化试图在训练早期降低不稳定，后期释放模型潜力。

这类工作说明：

```text
训练稳定性不是只靠 norm，也和 residual branch、初始化、学习率、优化器共同决定。
```

## 6.13 Norm Placement 对表示的影响

Post-LN 和 Pre-LN 不只是训练技巧，也影响表示流动。

Post-LN：

```math
h_{\ell+1}=\mathrm{Norm}\bigl(h_\ell+F_\ell(h_\ell)\bigr)
```

每层输出被标准化，尺度更统一，但残差路径被 norm 包住，梯度路径不再是纯 identity。

Pre-LN：

```math
h_{\ell+1}=h_\ell+F_\ell(\mathrm{Norm}(h_\ell))
```

残差主路径更干净，梯度更容易直通，但 residual stream 的尺度可能随层数累积，需要 final norm、初始化、residual scaling 等共同控制。

所以不能只说 Pre-LN “更好”。更准确是：

```text
Pre-LN 更利于深层稳定训练，因此在现代 LLM 中更常见；Post-LN 或 DeepNorm 变体在某些任务和结构中仍有价值。
```

## 6.14 大模型中的常见配置

现代 LLM 常见配置包括：

1. Pre-Norm。
2. RMSNorm。
3. Final RMSNorm。
4. AdamW。
5. Learning rate warmup + cosine/linear decay。
6. Gradient clipping 或 loss spike 监控。
7. bf16/fp8 混合精度。
8. Carefully designed initialization。

一个 LLaMA-like block 可以简化为：

```python
class TransformerBlock(nn.Module):
    def __init__(self, attention, mlp, hidden_size):
        super().__init__()
        self.attn_norm = SimpleRMSNorm(hidden_size)
        self.ffn_norm = SimpleRMSNorm(hidden_size)
        self.attention = attention
        self.mlp = mlp

    def forward(self, x):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
```

这段代码非常简单，但背后是多年训练稳定性经验的结果。

## 6.15 常见训练不稳定现象

训练不稳定可能表现为：

1. loss 突然 spike。
2. loss 变成 NaN。
3. gradient norm 爆炸。
4. activation norm 持续增大。
5. 某些层输出异常。
6. MoE router collapse。
7. attention logits 极端。
8. bf16/fp16 overflow。

排查时不能只看 norm。要同时看：

1. learning rate 和 warmup。
2. optimizer epsilon、beta、weight decay。
3. 初始化。
4. gradient clipping。
5. norm placement。
6. residual scaling。
7. data outlier。
8. mixed precision loss scaling。
9. distributed all-reduce 是否异常。

Norm 是稳定性的一部分，不是全部。

## 6.16 训练稳定性审计指标

训练稳定性不能只看 loss。对 residual、norm placement 和 warmup，至少要审计四类量。

第一，residual stream 的尺度：

```math
a_\ell=\lVert h_\ell\rVert_2
```

如果 `a_l` 随层数持续放大，说明残差写入、初始化、norm placement 或学习率可能有问题。

第二，子层更新相对主路径的比例：

```math
u_\ell=\frac{\lVert F_\ell(\mathrm{Norm}(h_\ell))\rVert_2}{\lVert h_\ell\rVert_2+\epsilon}
```

`u_l` 太大表示每层写入过猛，容易导致 residual stream 漂移；太小则可能表示子层几乎不工作。

第三，均值漂移：

```math
m_\ell=\left|\frac{1}{d}\sum_{i=1}^{d}h_{\ell,i}\right|
```

LayerNorm 会显式 re-center，RMSNorm 不会显式减均值，因此 RMSNorm/Pre-Norm 结构尤其要配合 residual scaling、初始化和 final norm 控制尺度。

第四，早期更新压力。用一个 toy 指标表示 warmup 对早期大步更新的缓冲：

```math
P_{\mathrm{early}}=\eta_1\max_\ell u_\ell
```

其中 `eta_1` 是 warmup 第一步学习率。这个指标不是论文中的标准训练指标，只是面试和教学中用来解释“同样的梯度/更新比例，warmup 会降低早期参数扰动”的直觉。

一个简化门禁可以写成：

```math
G_{\mathrm{norm}}=\mathbb{1}\{a_{\max}\le\tau_a,\ u_{\max}\le\tau_u,\ P_{\mathrm{early}}\le\tau_p,\ \mathrm{NaN}=0\}
```

真实训练中还要补充 loss spike、grad norm、activation outlier、optimizer state、mixed precision overflow、data outlier、分布式 all-reduce 异常和 MoE router collapse。

## 6.16.1 最小可运行 Residual / Norm 稳定性审计 demo

下面的 demo 不依赖 PyTorch，只用标准库构造一个 toy residual stack，比较四种结构：

1. `plain_no_residual`：没有残差主路径，信息容易被每层重写。
2. `residual_no_norm`：有残差但没有归一化，尺度容易随层数漂移。
3. `post_layernorm`：每层输出被 LayerNorm 约束，但 early update pressure 仍依赖 warmup。
4. `pre_rmsnorm_scaled`：Pre-RMSNorm 加 residual scaling，保留残差主路径，同时控制每层写入幅度。

```python
from math import sqrt, tanh


def mean(xs):
    return sum(xs) / len(xs)


def l2_norm(xs):
    return sqrt(sum(x * x for x in xs))


def layer_norm(xs, eps=1e-6):
    mu = mean(xs)
    var = mean([(x - mu) ** 2 for x in xs])
    scale = sqrt(var + eps)
    return [(x - mu) / scale for x in xs]


def rms_norm(xs, eps=1e-6):
    rms = sqrt(mean([x * x for x in xs]) + eps)
    return [x / rms for x in xs]


def sublayer(xs, strength):
    return [strength * tanh(x + 0.2 * (i + 1)) for i, x in enumerate(xs)]


def add(xs, ys):
    return [x + y for x, y in zip(xs, ys)]


def scale(xs, factor):
    return [factor * x for x in xs]


def run_stack(mode, depth=12, strength=0.8, residual_scale=1.0):
    h = [1.0, -0.5, 0.25, 2.0]
    rows = []
    for layer in range(1, depth + 1):
        before_norm = l2_norm(h)
        if mode == "plain_no_residual":
            update = sublayer(h, strength)
            h = update
        elif mode == "residual_no_norm":
            update = sublayer(h, strength)
            h = add(h, scale(update, residual_scale))
        elif mode == "post_layernorm":
            update = sublayer(h, strength)
            h = layer_norm(add(h, scale(update, residual_scale)))
        elif mode == "pre_rmsnorm_scaled":
            update = sublayer(rms_norm(h), strength)
            h = add(h, scale(update, residual_scale))
        else:
            raise ValueError(mode)

        update_norm = l2_norm(update)
        rows.append(
            {
                "layer": layer,
                "before": round(before_norm, 4),
                "update_ratio": round(update_norm / max(before_norm, 1e-9), 4),
                "after": round(l2_norm(h), 4),
                "mean_abs": round(abs(mean(h)), 4),
            }
        )
    return rows


def audit(rows, lr_peak, warmup_steps):
    max_norm = max(row["after"] for row in rows)
    max_update_ratio = max(row["update_ratio"] for row in rows)
    first_lr = lr_peak / warmup_steps
    early_update_pressure = first_lr * max_update_ratio
    return {
        "final_norm": rows[-1]["after"],
        "max_norm": round(max_norm, 4),
        "max_update_ratio": round(max_update_ratio, 4),
        "final_mean_abs": rows[-1]["mean_abs"],
        "early_update_pressure": round(early_update_pressure, 6),
        "gate_pass": (
            max_norm <= 6.0
            and max_update_ratio <= 0.9
            and early_update_pressure <= 0.0005
        ),
    }


configs = {
    "plain_no_residual": ("plain_no_residual", 1.0),
    "residual_no_norm": ("residual_no_norm", 1.0),
    "post_layernorm": ("post_layernorm", 1.0),
    "pre_rmsnorm_scaled": ("pre_rmsnorm_scaled", 1 / sqrt(24)),
}

summaries = {}
for name, (mode, residual_scale) in configs.items():
    rows = run_stack(mode, residual_scale=residual_scale)
    summaries[name] = audit(rows, lr_peak=0.003, warmup_steps=20)

pre_rows = run_stack("pre_rmsnorm_scaled", residual_scale=1 / sqrt(24))
post_rows = run_stack("post_layernorm")

print("summaries=", summaries)
print("pre_rmsnorm_tail=", pre_rows[-3:])
print("post_ln_no_warmup=", audit(post_rows, lr_peak=0.003, warmup_steps=1))
print("post_ln_with_warmup=", audit(post_rows, lr_peak=0.003, warmup_steps=20))
```

预期输出：

```text
summaries= {'plain_no_residual': {'final_norm': 1.2631, 'max_norm': 1.2634, 'max_update_ratio': 1.0431, 'final_mean_abs': 0.6236, 'early_update_pressure': 0.000156, 'gate_pass': False}, 'residual_no_norm': {'final_norm': 19.7276, 'max_norm': 19.7276, 'max_update_ratio': 0.511, 'final_mean_abs': 5.9974, 'early_update_pressure': 7.7e-05, 'gate_pass': False}, 'post_layernorm': {'final_norm': 2.0, 'max_norm': 2.0, 'max_update_ratio': 0.6247, 'final_mean_abs': 0.0, 'early_update_pressure': 9.4e-05, 'gate_pass': True}, 'pre_rmsnorm_scaled': {'final_norm': 5.0581, 'max_norm': 5.0581, 'max_update_ratio': 0.4972, 'final_mean_abs': 2.0237, 'early_update_pressure': 7.5e-05, 'gate_pass': True}}
pre_rmsnorm_tail= [{'layer': 10, 'before': 4.3352, 'update_ratio': 0.2879, 'after': 4.5743, 'mean_abs': 1.7837}, {'layer': 11, 'before': 4.5743, 'update_ratio': 0.2744, 'after': 4.8153, 'mean_abs': 1.9031}, {'layer': 12, 'before': 4.8153, 'update_ratio': 0.2619, 'after': 5.0581, 'mean_abs': 2.0237}]
post_ln_no_warmup= {'final_norm': 2.0, 'max_norm': 2.0, 'max_update_ratio': 0.6247, 'final_mean_abs': 0.0, 'early_update_pressure': 0.001874, 'gate_pass': False}
post_ln_with_warmup= {'final_norm': 2.0, 'max_norm': 2.0, 'max_update_ratio': 0.6247, 'final_mean_abs': 0.0, 'early_update_pressure': 9.4e-05, 'gate_pass': True}
```

这个 demo 要读出四点：

1. `plain_no_residual` 的 `max_update_ratio` 超过 1，说明每层都可能大幅重写表示，缺少稳定主路径。
2. `residual_no_norm` 虽然有残差，但 `max_norm` 增长到 `19.7276`，说明残差写入没有尺度控制会漂移。
3. `post_layernorm` 的输出尺度被 LayerNorm 固定住，但如果不 warmup，`early_update_pressure` 会升到 `0.001874`，门禁失败。
4. `pre_rmsnorm_scaled` 保留残差主路径，并用 RMSNorm + residual scaling 控制更新比例；它没有 re-centering，所以 `final_mean_abs` 不为 0，这也是为什么还需要 final norm、初始化和监控。

## 6.17 常见误区

误区一：LayerNorm 和 BatchNorm 差不多。

更准确：LayerNorm 对单样本 hidden 维做归一化，不依赖 batch size，训练和推理一致，更适合 Transformer。

误区二：RMSNorm 只是 LayerNorm 的近似。

更准确：RMSNorm 去掉 re-centering，保留 re-scaling，计算更简单，在现代 LLM 中被广泛采用。

误区三：Pre-LN 一定全面优于 Post-LN。

更准确：Pre-LN 更稳定、更适合深层 LLM 训练；Post-LN 也有性能优势和改进变体，DeepNorm 等方法试图结合两者优点。

误区四：Warmup 是经验 trick，没有理论意义。

更准确：LayerNorm 位置和初始化梯度行为能解释 warmup 的必要性，尤其是 Post-LN 设置下。

误区五：Residual 只是简单相加。

更准确：Residual 是深层网络信息和梯度流动的主干，残差分支尺度决定训练稳定性。

## 6.18 面试题

### 题 1：Residual connection 为什么重要？

参考回答：

```text
Residual connection 让每层学习输入上的增量，而不是从零学习完整映射。它提供信息直通路径和梯度直通路径，使深层网络更容易优化。在 Transformer 中，attention 和 FFN 都是在 residual stream 上写入增量特征，残差路径是信息累积和梯度传播的主干。
```

### 题 2：LayerNorm 和 BatchNorm 有什么区别？

参考回答：

```text
BatchNorm 使用 batch 维度统计，训练和推理行为不同，并依赖 batch size。LayerNorm 对单个样本的 hidden dimension 做归一化，不依赖 batch size，训练和推理一致，适合变长序列和自回归推理。因此 Transformer 通常使用 LayerNorm/RMSNorm，而不是 BatchNorm。
```

### 题 3：RMSNorm 和 LayerNorm 有什么区别？

参考回答：

```text
LayerNorm 会减去均值并除以标准差，提供 re-centering 和 re-scaling。RMSNorm 不减均值，只用 root mean square 做尺度归一化，主要提供 re-scaling。它计算更简单，实践中在大模型上效果很好，因此 LLaMA 等现代 LLM 常用 RMSNorm。
```

### 题 4：Pre-LN 和 Post-LN 的区别是什么？

参考回答：

```text
Post-LN 是先 residual add 再 LayerNorm，即 x = Norm(x + Sublayer(x))。Pre-LN 是先 Norm 再进入子层，然后 residual add，即 x = x + Sublayer(Norm(x))。Post-LN 每层输出被归一化，但深层训练更容易梯度不稳定，通常依赖 warmup。Pre-LN 保留更干净的 residual 梯度路径，训练更稳定，因此现代 decoder-only LLM 常用 Pre-Norm。
```

### 题 5：为什么 Post-LN 更依赖 warmup？

参考回答：

```text
研究表明 Post-LN Transformer 在初始化时，靠近输出层的参数梯度期望可能较大。如果一开始使用较大学习率，容易导致训练不稳定。Warmup 通过在训练初期使用小学习率，避免大梯度造成参数剧烈更新。Pre-LN 的梯度行为更稳定，因此对 warmup 的依赖较小，但实际大模型训练仍常使用 warmup 作为稳定措施。
```

### 题 6：DeepNorm 解决什么问题？

参考回答：

```text
DeepNorm 想解决极深 Transformer 的训练稳定性问题。它通过修改 residual connection 的归一化和缩放方式，并配套理论初始化，使模型更新在深层网络中保持有界。目标是结合 Post-LN 的性能和 Pre-LN 的稳定性，把 Transformer 扩展到上百甚至上千层。
```

## 6.19 小练习

1. 画出 Post-LN 和 Pre-LN Transformer block 的结构图。
2. 用 PyTorch 实现 LayerNorm 和 RMSNorm，并比较它们的计算差异。
3. 解释为什么 LayerNorm 不依赖 batch size。
4. 说明为什么 residual branch 的输出尺度过大可能导致训练不稳定。
5. 阅读 *On Layer Normalization in the Transformer Architecture* 摘要，解释 warmup 和 norm placement 的关系。
6. 阅读 RMSNorm 摘要，解释 re-centering 和 re-scaling 的区别。
7. 设计一个训练稳定性监控面板，至少包含 loss、grad norm、activation norm、learning rate、NaN 统计。
8. 运行本章 Residual / Norm 稳定性审计 demo，解释 `residual_no_norm` 和 `post_ln_no_warmup` 分别为什么门禁失败。

## 6.20 本章总结

本章讲了 Residual、LayerNorm、RMSNorm、Pre-Norm 和训练稳定性。

核心结论：

1. Residual connection 是深层 Transformer 信息和梯度流动的主干。
2. Attention 和 FFN 都是在 residual stream 上写入增量信息。
3. LayerNorm 对单样本 hidden 维归一化，不依赖 batch size，适合 Transformer。
4. RMSNorm 去掉均值中心化，只做尺度归一化，计算更简单，现代 LLM 常用。
5. Post-LN 是先 residual add 再 norm，Pre-LN 是先 norm 再子层计算后 residual add。
6. Post-LN 往往更依赖 warmup，Pre-LN 深层训练更稳定。
7. DeepNorm、Admin 等方法说明 residual scaling、初始化和 norm placement 共同决定训练稳定性。
8. 大模型训练稳定性不是单一模块决定的，而是 norm、residual、初始化、optimizer、learning rate、precision 和数据共同作用。

下一章会进入 Position Encoding，总览 Sinusoidal、Learned、Relative、RoPE、ALiBi 等位置编码方法，解释 Transformer 为什么必须显式处理位置信息。
