# 第五章：FFN、SwiGLU、Gated MLP 与 MoE FFN

## 5.1 本章定位

很多人学 Transformer 时，会把注意力机制当成唯一主角，把 FFN 当成“后面跟着的两层 MLP”。这是一个常见误区。

在现代大模型中，FFN/MLP 层通常占据大量参数和计算。Attention 负责 token 之间的信息路由，FFN 则负责对每个 token 的表示做非线性变换、特征组合和容量扩展。可以粗略理解为：

```text
Attention: token 之间交换信息
FFN/MLP: 每个 token 内部加工信息
```

本章讲 Transformer block 中另一个核心模块：FFN。从原始 Transformer 的 ReLU FFN，到 GELU，再到现代 LLM 常用的 SwiGLU/Gated MLP，再到 MoE FFN。

学完本章，你应该能回答：

1. Transformer FFN 为什么是 position-wise 的。
2. FFN 在 Transformer 中到底起什么作用。
3. 为什么现代 LLM 常用 SwiGLU，而不是普通 ReLU/GELU FFN。
4. Gated MLP 的门控直觉是什么。
5. 为什么 MoE 通常替换 FFN，而不是替换 attention。
6. Switch、GShard、Mixtral、DeepSeekMoE 这些 MoE FFN 的核心 trade-off 是什么。
7. 面试中如何解释 FFN、SwiGLU 和 MoE 的关系。

## 5.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。原始 Transformer 使用 position-wise feed-forward networks，每个位置独立应用相同的两层全连接网络和 ReLU。
2. Shazeer, 2020, *GLU Variants Improve Transformer*。系统测试 GLU 变体，发现部分 gated FFN 变体相对 ReLU/GELU 能提升 Transformer 质量，SwiGLU 后来成为现代 LLM 常用组件。
3. Chowdhery et al., 2022, *PaLM*。PaLM 是大规模 decoder-only Transformer，公开使用 SwiGLU 等现代结构并展示 scaling 效果。
4. Touvron et al., 2023, *LLaMA*。LLaMA 使用 RMSNorm、SwiGLU、RoPE 等组件，成为开源 LLM 架构基线。
5. Lepikhin et al., 2020, *GShard*。使用 sparsely-gated MoE 和自动 sharding，把 Transformer 扩展到 600B+ 参数级别。
6. Fedus et al., 2021/2022, *Switch Transformers*。简化 MoE routing，用每个 token 选择一个 expert 的方式扩展到 trillion-parameter 模型，并讨论通信、稳定性和低精度训练。
7. Jiang et al., 2024, *Mixtral of Experts*。Mixtral 8x7B 使用 sparse MoE，每层有 8 个 FFN experts，每个 token 选 2 个专家，推理激活参数少于总参数。
8. DeepSeek-AI, 2024, *DeepSeek-V2*。DeepSeekMoE 通过 sparse computation 经济地训练强模型，并和 MLA 一起优化训练和推理效率。

需要注意：不同模型对 FFN hidden size、gate 维度、activation、expert 数、top-k routing、负载均衡的细节可能不同。本章重点讲通用机制和 trade-off，不把某个模型的私有配置泛化成标准。

## 5.3 原始 Transformer FFN 是什么

原始 Transformer 每个 block 中有两个主要子层：

```text
Multi-Head Attention
Position-wise Feed-Forward Network
```

原论文中的 FFN 形式是：

```text
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
```

也就是两层线性层，中间 ReLU。

它叫 position-wise，是因为同一个 FFN 独立作用在每个 token 位置上：

```text
for each token position i:
    y_i = FFN(x_i)
```

不同位置之间不在 FFN 中直接交互。token 之间的信息交换主要由 attention 完成。

这有一个重要设计分工：

```text
Attention: mix across sequence dimension
FFN: transform across hidden/channel dimension
```

从张量形状看：

```text
X: [batch, seq_len, hidden_size]
FFN: hidden_size -> intermediate_size -> hidden_size
Output: [batch, seq_len, hidden_size]
```

每个 token 使用同一组 FFN 参数，所以 FFN 像一个作用在 token 表示上的共享非线性函数。

## 5.4 FFN 为什么重要

如果只有 attention，没有 FFN，模型主要是在不同 token 的 Value 之间做线性加权组合。虽然 attention 权重来自非线性 softmax，但每层表达能力仍然受限。

FFN 提供了几个关键能力。

第一，非线性变换。

FFN 引入 ReLU/GELU/SwiGLU 等非线性，使模型能学习更复杂的特征组合。

第二，通道维度扩展。

典型 FFN 会把 hidden size 扩大到 4 倍左右，再投影回来：

```text
D -> 4D -> D
```

这给每个 token 提供更大的中间特征空间。

第三，知识和模式存储。

很多研究和工程直觉都认为 FFN/MLP 层承担了大量事实、模式和特征转换能力。Attention 负责把相关信息带到当前位置，FFN 负责把这些信息加工成下一层可用表示。

第四，参数容量主体。

在很多 Transformer 配置中，FFN 参数量远大于 attention 投影参数量。因此扩展模型容量时，FFN 是非常关键的部分。

一句话：

```text
Attention 决定 token 从哪里拿信息，FFN 决定拿到信息后怎么加工。
```

## 5.5 ReLU、GELU 到 GLU

原始 Transformer 使用 ReLU。

后来很多语言模型使用 GELU。GELU 相比 ReLU 更平滑，在 BERT/GPT 系列中很常见。

普通 FFN 可以写成：

```text
FFN(x) = W_2 activation(W_1 x)
```

GLU 类结构则引入门控：

```text
GLU(x) = (W_a x) * sigmoid(W_b x)
```

其中 `*` 是逐元素乘法。

直觉上，GLU 把 FFN 分成两条支路：

1. value branch：产生候选特征。
2. gate branch：决定哪些特征通过。

也就是说，不是所有中间特征都同等进入下一层，而是由 gate 控制信息流。

这和 attention 的“加权读取”有点类似，但发生在通道维度，而不是 token 维度。

## 5.6 SwiGLU 是什么

Shazeer 的 *GLU Variants Improve Transformer* 测试了多种 GLU 变体，例如 ReGLU、GEGLU、SwiGLU 等。

SwiGLU 可以理解为：

```text
SwiGLU(x) = Swish(W_gate x) * (W_up x)
```

然后再通过 down projection 回到 hidden size：

```text
output = W_down( Swish(W_gate x) * W_up x )
```

Swish 通常是：

```text
Swish(z) = z * sigmoid(z)
```

现代 LLM 中常见的 MLP 形式是：

```text
gate = SiLU(W_gate x)
up = W_up x
hidden = gate * up
out = W_down hidden
```

这里的 SiLU 和 Swish 基本是同类表达。

SwiGLU 的直觉：

1. `W_up x` 产生候选特征。
2. `SiLU(W_gate x)` 产生平滑门控。
3. 两者逐元素相乘，选择性激活特征。
4. `W_down` 把中间特征投回 hidden space。

这比普通 FFN 多了一条 projection，但通常可以通过调整 intermediate size 控制参数量。

## 5.7 为什么现代 LLM 爱用 SwiGLU

LLaMA、PaLM 等模型把 SwiGLU/Gated FFN 推成了现代 LLM 的常见 recipe。

原因可以从三层理解。

第一，表达能力。

门控结构让模型能对不同 token、不同上下文动态选择通道特征。它比简单的 ReLU/GELU 单支路更灵活。

第二，训练效果。

GLU 变体实验显示部分 gated FFN 能提升 Transformer 质量。实践中 SwiGLU 成为大模型常用选择。

第三，工程可控。

SwiGLU 仍然是矩阵乘法 + elementwise activation + elementwise multiply，非常适合 GPU/TPU。它不像 MoE 那样引入复杂 routing 和通信。

所以 SwiGLU 是一个很好的工程折中：

```text
比普通 FFN 表达强很多，但实现复杂度远低于 MoE。
```

## 5.8 一个 PyTorch 版 SwiGLU MLP

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

输入输出形状：

```text
x: [batch, seq_len, hidden_size]
gate/up: [batch, seq_len, intermediate_size]
out: [batch, seq_len, hidden_size]
```

注意，FFN/MLP 对每个 token 独立作用，不直接混合序列维度。

## 5.9 FFN 参数量和计算量

普通 FFN：

```text
hidden_size -> intermediate_size -> hidden_size
```

参数量约为：

```text
2 * hidden_size * intermediate_size
```

如果 `intermediate_size = 4 * hidden_size`，参数量约是：

```text
8 * hidden_size^2
```

SwiGLU MLP 有三组矩阵：

```text
gate_proj, up_proj, down_proj
```

参数量约为：

```text
3 * hidden_size * intermediate_size
```

为了让参数量接近普通 4D FFN，实际常把 SwiGLU intermediate size 设得比 4D 小，例如接近 `8D/3`。

面试中要能说明：

```text
SwiGLU 多一条 gate projection，所以不能简单在相同 intermediate size 下和普通 FFN 比参数；通常会调小 hidden dim 以控制总参数和 FLOPs。
```

## 5.10 MoE FFN：把一个 FFN 变成多个专家

MoE 的核心思想是条件计算。

普通 dense FFN 是：

```text
每个 token 都经过同一个 FFN。
```

MoE FFN 是：

```text
每个 token 由 router 选择少数几个 experts，只经过被选中的 FFN。
```

每个 expert 通常就是一个 FFN/MLP。

如果有 8 个 experts，每个 token 只选 2 个：

```text
total FFN capacity = 8 experts
active FFN compute per token = 2 experts
```

这带来一个巨大优势：

```text
总参数量可以很大，但每个 token 的激活计算相对较小。
```

这就是 sparse activation。

## 5.11 为什么 MoE 通常替换 FFN，而不是 Attention

MoE 最常见的位置是 FFN 层，而不是 attention 层。

原因包括：

1. FFN 本来就是 token-wise 独立计算，适合按 token 路由到不同专家。
2. FFN 参数量大，替换成专家后能显著扩展模型容量。
3. Attention 负责 token 间交互，如果按专家拆得太复杂，会影响全局信息路由。
4. FFN expert 可以相对独立并行部署，更适合 expert parallelism。
5. MoE 的主要目标是增加参数容量，而 FFN 是容量主体。

所以 MoE FFN 可以理解为：

```text
把 Transformer block 中最占容量的 dense MLP，替换成多个稀疏激活的 MLP experts。
```

## 5.12 Router：MoE 的关键模块

MoE 需要一个 router 决定每个 token 去哪个 expert。

简化形式：

```text
router_logits = x W_router
expert_weights = softmax(router_logits)
top_k_experts = topk(expert_weights, k)
```

然后 token 只送到 top-k experts。

Switch Transformer 使用更简单的 top-1 routing：每个 token 只选一个 expert。Mixtral 使用 top-2 routing：每个 token 选两个 experts，并组合输出。

Top-1 routing 优点：

1. 计算更省。
2. 通信更简单。
3. 路由实现更直接。

Top-2 routing 优点：

1. 表达更强。
2. 对单个 expert 选择错误更鲁棒。
3. 输出可由多个专家组合。

Router 的难点是负载均衡。如果很多 token 都去同一个 expert，就会出现：

1. 热点 expert 过载。
2. 其他 expert 闲置。
3. batch 内通信不均衡。
4. 训练不稳定。

所以 MoE 常配合 load balancing loss、capacity factor、token dropping 或 auxiliary-loss-free balancing 等设计。

## 5.13 GShard、Switch、Mixtral、DeepSeekMoE 的演进

GShard 展示了用 sparsely-gated MoE 和自动 sharding 把 Transformer 扩展到 600B+ 参数级别。它的重点是：

1. 条件计算。
2. 自动分片。
3. 大规模 TPU 训练。
4. 多语言翻译质量提升。

Switch Transformer 进一步简化 MoE，把 routing 简化为 top-1。它强调：

1. 简化通信和计算。
2. 降低训练复杂度。
3. 稳定低精度训练。
4. 用 sparse activation 扩展到 trillion parameters。

Mixtral 8x7B 把 MoE 带入开源 LLM 主流视野。公开摘要指出：每层有 8 个 feedforward blocks，每个 token 选 2 个 experts；每个 token 可访问 47B 参数，但推理只激活约 13B 参数。

DeepSeek-V2 的 DeepSeekMoE 则强调经济训练和高效推理。它和 MLA 组合，说明现代架构会同时优化：

```text
FFN 容量 -> MoE
KV cache -> MLA
```

## 5.14 Dense FFN、SwiGLU、MoE FFN 对比

| 结构 | 核心思想 | 优点 | 代价 |
|---|---|---|---|
| ReLU/GELU FFN | 两层 MLP + 激活 | 简单稳定 | 表达较普通 |
| SwiGLU/Gated MLP | 候选特征乘门控 | 表达更强，现代 LLM 常用 | 多一条 projection |
| MoE FFN | 多个 experts，按 token 选择少数激活 | 总容量大，激活计算低 | routing、通信、负载均衡复杂 |

可以把它们看成逐步增强：

```text
普通 FFN: 一个共享专家
SwiGLU: 一个带门控的共享专家
MoE FFN: 多个带或不带门控的专家，由 router 选择
```

## 5.15 MoE 的工程 Trade-off

MoE 的优势很诱人：更多参数、更少激活计算。

但工程代价也很明显。

第一，通信成本。

不同 token 要发到不同 expert，专家可能分布在不同 GPU/TPU 上，需要 all-to-all 通信。

第二，负载均衡。

Router 可能把大量 token 分到少数 expert，造成热点。

第三，训练稳定性。

MoE 比 dense 模型更容易出现 routing collapse、expert under-utilization、loss spike。

第四，serving 复杂度。

推理时要管理 expert placement、batch dispatch、token combine、cache 和延迟。

第五，评估解释困难。

不同 token 激活不同 expert，模型行为更分散。

所以 MoE 不是免费午餐。

它适合有强系统能力、追求大容量和成本效率的团队。对小团队或小模型，dense SwiGLU 往往更简单可靠。

## 5.16 FFN 和 Attention 的互补关系

一个 Transformer block 可以这样理解：

```text
Attention: 收集上下文信息
FFN: 加工当前 token 表示
Residual: 保留和累积信息
Norm: 稳定数值分布
```

如果 attention 找到了相关证据，但 FFN 不够强，模型可能无法把证据转换成正确推理。

如果 FFN 很强，但 attention 没把相关信息带来，模型只能加工局部或错误信息。

所以现代架构演进不是只优化 attention，也会优化 FFN：

1. SwiGLU 提升 dense FFN 表达。
2. MoE 提升参数容量。
3. 更好的 routing 提升专家利用率。
4. 更好的并行系统降低 MoE 通信代价。

## 5.17 常见误区

误区一：FFN 不重要，Transformer 主要靠 attention。

更准确：attention 负责信息路由，FFN 负责非线性变换和容量扩展，二者都核心。

误区二：SwiGLU 只是换了激活函数。

更准确：SwiGLU 是 gated MLP，包含 gate/up/down 三个投影，门控机制改变了通道特征选择方式。

误区三：MoE 就是让模型更大，所以一定更好。

更准确：MoE 增加总参数但只激活部分专家，质量依赖 routing、负载均衡、数据、训练稳定性和系统实现。

误区四：MoE 推理一定更快。

更准确：激活 FLOPs 可能低，但 all-to-all 通信、expert dispatch 和 batch 不均衡可能带来延迟。

误区五：Mixtral 8x7B 每次都用 47B 参数计算。

更准确：每个 token 可以访问较大总参数空间，但推理只激活少数 experts，对应较少 active parameters。

## 5.18 面试题

### 题 1：Transformer FFN 的作用是什么？

参考回答：

```text
Attention 负责 token 之间的信息交换，FFN 负责对每个 token 的 hidden representation 做非线性通道变换。它通常把 hidden size 扩展到更高维，再投影回来，提供非线性表达、特征组合和模型容量。很多 Transformer 的参数和计算大量集中在 FFN 层，所以它不是附属模块。
```

### 题 2：为什么现代 LLM 常用 SwiGLU？

参考回答：

```text
SwiGLU 是 gated MLP。它用一条 up projection 产生候选特征，用一条 gate projection 经过 SiLU/Swish 产生门控，两者逐元素相乘后再 down projection。相比普通 ReLU/GELU FFN，它能更灵活地选择通道特征，实践中在 Transformer 中效果更好，同时仍然主要是矩阵乘和逐元素操作，工程实现简单。
```

### 题 3：SwiGLU 和普通 FFN 参数量怎么比？

参考回答：

```text
普通 FFN 通常有 up 和 down 两个矩阵，参数约为 2 * hidden_size * intermediate_size。SwiGLU 有 gate、up、down 三个矩阵，参数约为 3 * hidden_size * intermediate_size。因此如果 intermediate size 相同，SwiGLU 参数更多。实际模型通常会把 SwiGLU 的 intermediate size 调小，例如接近 8D/3，让参数量和普通 4D FFN 接近。
```

### 题 4：为什么 MoE 通常替换 FFN？

参考回答：

```text
因为 FFN 是 token-wise 独立计算，参数量大，适合按 token 路由到不同专家。把 FFN 换成多个 experts 后，每个 token 只激活少数专家，就能在保持每 token 计算相对较低的情况下扩大总参数容量。Attention 负责 token 间交互，直接做 MoE 更复杂，也不如 FFN 位置自然。
```

### 题 5：Switch Transformer 和 Mixtral 的 routing 有什么区别？

参考回答：

```text
Switch Transformer 强调简化 MoE routing，每个 token 通常选择一个 expert，即 top-1 routing，降低通信和计算复杂度。Mixtral 8x7B 每层有 8 个 FFN experts，每个 token 选择 2 个 experts，属于 top-2 routing，表达更强但计算和通信也更复杂。两者都是 sparse MoE，但 routing 策略不同。
```

### 题 6：MoE 的主要工程难点是什么？

参考回答：

```text
主要难点包括 all-to-all 通信、负载均衡、routing collapse、expert 利用不均、训练稳定性、低精度训练、serving 时的 expert dispatch 和延迟控制。MoE 降低的是每 token 激活计算，不自动解决通信和系统复杂度，所以它需要模型和系统一起设计。
```

## 5.19 小练习

1. 写出普通 FFN、GEGLU、SwiGLU 的公式，并比较差异。
2. 用 PyTorch 实现一个 SwiGLU MLP，并打印中间张量形状。
3. 计算 hidden size 为 4096 时，普通 4D FFN 的参数量约是多少。
4. 计算同样 hidden size 下，SwiGLU 如果 intermediate size 为 `8D/3`，参数量约是多少。
5. 画一个 MoE FFN 的 routing 流程图：token -> router -> top-k experts -> combine。
6. 阅读 Switch Transformer 摘要，解释为什么它强调 simple and efficient sparsity。
7. 阅读 Mixtral 摘要，解释“47B total / 13B active”这类说法的含义。
8. 讨论为什么 MoE 适合大规模团队，但不一定适合所有小模型项目。

## 5.20 本章总结

本章系统讲了 FFN、SwiGLU、Gated MLP 和 MoE FFN。

核心结论：

1. FFN 是 Transformer block 中和 attention 同等重要的模块。
2. Attention 负责 token 间信息路由，FFN 负责 token 内通道变换和非线性加工。
3. 原始 Transformer 使用 position-wise ReLU FFN，后来很多模型使用 GELU，再后来现代 LLM 大量采用 SwiGLU。
4. SwiGLU 是 gated MLP，通过 gate/up/down 三个投影增强通道特征选择能力。
5. SwiGLU 参数量比较时要注意 intermediate size，不能只看激活函数名。
6. MoE FFN 把一个 dense FFN 变成多个 experts，由 router 为每个 token 选择少数 experts。
7. MoE 的核心价值是扩大总参数容量，同时保持每 token 激活计算相对较低。
8. MoE 的核心代价是 routing、通信、负载均衡、训练稳定性和 serving 复杂度。
9. 现代大模型架构演进中，attention 优化和 FFN/MoE 优化同样关键。

下一章会进入 Residual、LayerNorm、RMSNorm、Pre-Norm 与训练稳定性，重点讲为什么规范化和残差结构决定了深层 Transformer 能否稳定训练。
