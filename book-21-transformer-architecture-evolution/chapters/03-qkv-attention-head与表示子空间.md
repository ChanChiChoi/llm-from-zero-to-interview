# 第三章：Q、K、V、Attention Head 与表示子空间

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修以公开论文、公开技术报告和前序章节为资料边界：Transformer 原论文对 multi-head attention 的定义，BERT 及后续 attention head 分析对 head pattern 的观察，Michel 等人关于 head pruning 和冗余的实验结论，Abnar 和 Zuidema 对跨层 attention flow 的解释边界提醒，Transformer Circuits 对 QK / OV circuit 与 residual stream 的机制化拆解，Shazeer 的 MQA 论文和 Ainslie 等人的 GQA 论文对 K/V head 压缩和推理带宽瓶颈的解释，以及近期公开研究对 attention head 角色跨随机重训稳定性的谨慎讨论。

本章只解释 Q/K/V、attention head、表示子空间、QK / OV 拆解、head 冗余、MHA / MQA / GQA 的形状和成本直觉；不把某个可视化 attention map 写成完整因果解释，不把 toy demo 的数值当成真实模型 benchmark，也不推断闭源模型的内部 head 分工。

第二轮补强重点有三点：

1. 将 Q/K/V、multi-head attention、QK circuit、OV circuit、head dim 和 KV cache 成本改成稳定 MathJax 表达。
2. 明确 head 可解释性的边界：head pattern 是线索，必须结合 ablation、activation patching、QK / OV 分解和任务行为变化。
3. 补一个 0 依赖 Python demo，审计 QK 路由命中、OV 写入目标、head 冗余、解释风险，以及 MHA / GQA / MQA 的 KV cache 粗略成本。

## 3.1 本章定位

上一章从信息路由角度解释了 Self-Attention。本章继续深入：Q、K、V 和 attention head 到底在表示什么？为什么 Transformer 不只做一个 attention，而要做 multi-head attention？每个 head 是否真的学到了不同功能？为什么有些 head 可以剪掉？为什么 MQA、GQA 又要减少 K/V head？

这些问题是 Transformer 架构面试的高频追问。只会说“多头可以关注不同位置”还不够，面试官可能继续问：

1. 多头为什么比单头表达力强？
2. QK 和 OV 可以如何分开理解？
3. head_dim 变大会怎样？head 数变多会怎样？
4. attention head 是否可解释？
5. 如果很多 head 冗余，为什么还要多头？
6. MQA/GQA 为什么减少 K/V head，而不是减少 Q head？

本章会把 attention head 看成“在不同表示子空间中运行的信息路由器”。这比“多个头看不同位置”更准确。

## 3.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。原论文提出 multi-head attention，说明多头允许模型在不同表示子空间、不同位置上联合关注信息。
2. Devlin et al., 2018/2019, *BERT*。BERT 使用多层多头双向 Transformer，证明多头 self-attention 可形成强语言理解表示。
3. Clark et al., 2019, *What Does BERT Look At?*。分析 BERT attention head，发现一些 head 关注 delimiter、固定位置偏移、全句信息，也有 head 与句法关系、共指关系相关。
4. Michel et al., 2019, *Are Sixteen Heads Really Better than One?*。发现训练后的模型中大量 attention head 可在测试时剪掉而不显著影响性能，说明多头存在冗余和层间差异。
5. Abnar and Zuidema, 2020, *Quantifying Attention Flow in Transformers*。指出跨层信息不断混合，raw attention weights 作为解释并不可靠，提出 attention rollout/flow 等后验方法。
6. Shazeer, 2019, *Fast Transformer Decoding: One Write-Head is All You Need*。提出 MQA，指出增量解码时加载大量 K/V tensor 带来内存带宽瓶颈，共享 K/V 可加速推理。
7. Ainslie et al., 2023, *GQA*。提出 grouped-query attention，在 MHA 和 MQA 之间折中，用较少 K/V heads 获得接近 MHA 的质量和接近 MQA 的速度。
8. Transformer Circuits, *A Mathematical Framework for Transformer Circuits*。用 QK circuit 和 OV circuit 拆解 attention head 的“读哪里”和“写什么”两部分，为理解 head 功能提供机制解释视角。
9. 近期关于 attention head role stability 的公开研究提醒：即使两个模型结构和数据相同，随机初始化和训练过程也可能让某些 head 功能位置发生变化，因此不能把“第几层第几个 head”过度解释成稳定命名模块。

需要注意：attention head 的可解释性研究很有启发，但不能把某个 head 的可视化图当成完整因果解释。本章会区分“局部机制解释”“统计相关模式”和“最终模型行为”。

## 3.3 从单头到多头：为什么一个 attention 不够

单头 attention 做的事情是：

```text
在一个表示子空间里，计算每个 token 应该从哪些 token 读取信息，并把它们的 Value 加权混合。
```

但自然语言和代码中的关系不是单一类型。

同一个 token 可能同时需要：

1. 找前一个 token 的局部搭配。
2. 找主语和谓语关系。
3. 找代词指代实体。
4. 找括号、引号、代码块边界。
5. 找函数定义和调用。
6. 找长上下文中的证据句。
7. 找格式模板中的对应字段。

如果只有一个 attention head，它只能生成一张 attention map。

多头 attention 的直觉是：

```text
让模型在多个不同子空间里同时做信息路由，再把结果拼接起来。
```

原论文中的表达是，多头允许模型联合关注不同位置、不同表示子空间的信息。

所以多头不是简单并行重复，而是把 hidden representation 切成多个 head，每个 head 用自己的 `W_Q`、`W_K`、`W_V` 学一个不同的信息路由和内容变换。

## 3.4 多头注意力的数学形式

输入张量可以写成：

```math
X\in\mathbb{R}^{B\times N\times D}
```

其中 `B` 是 batch size，`N` 是序列长度，`D` 是 hidden size。第 `h` 个 head 的投影是：

```math
Q_h=XW_h^Q,\qquad K_h=XW_h^K,\qquad V_h=XW_h^V
```

并且：

```math
Q_h,K_h,V_h\in\mathbb{R}^{B\times N\times d_h}
```

第 `h` 个 head 的注意力权重和输出可以写成：

```math
A_h=\mathrm{softmax}\left(\frac{Q_hK_h^\top}{\sqrt{d_h}}+M\right)
```

```math
O_h=A_hV_h
```

其中 `M` 是可选 mask，例如 causal mask。所有 head 拼接后再通过输出投影混回主 hidden space：

```math
\mathrm{MHA}(X)=\mathrm{Concat}(O_1,\ldots,O_H)W^O
```

如果不考虑一些特殊变体，通常有：

```math
D=H d_h
```

这说明 multi-head attention 至少有三层含义：

1. 每个 head 有自己的 Q/K/V 投影。
2. 每个 head 有自己的 attention map。
3. 所有 head 输出再通过 `W_O` 混合回主 hidden space。

如果只看 attention weights，会漏掉 Value 和 Output projection 的影响。一个 head 不只是“看哪里”，还包括“把看到的信息写成什么”。

## 3.5 QK Circuit 和 OV Circuit

一个 attention head 可以拆成两部分理解。

第一部分是 QK circuit：

```text
QK: 决定从哪里读。
```

它由 `W_Q` 和 `W_K` 决定，影响 attention score：

```math
s_{ij}^{(h)}=(x_iW_h^Q)(x_jW_h^K)^\top
```

第二部分是 OV circuit：

```text
OV: 决定读到的信息如何写入 residual stream。
```

它由 `W_V` 和 `W_O` 决定：

```math
c_{ij}^{(h)}=a_{ij}^{(h)}x_jW_h^VW_h^O
```

其中 `a_ij` 是第 `h` 个 head 中位置 `i` 对位置 `j` 的 attention weight，`W_h^O` 可以理解为输出投影中对应第 `h` 个 head 的那一块。

这个拆法非常重要。

很多初学者只看 attention map，以为 head 的作用就是“关注某个 token”。但真正的 head 功能包括：

1. 通过 QK 选择源 token。
2. 通过 OV 把源 token 的某些特征写到目标位置。

举例：一个 head 可能在 QK 上关注前一个 token，但 OV 写入的是“前一个 token 的词性或括号状态”。另一个 head 也关注前一个 token，但写入的是完全不同的信息。

所以只看 attention weight 不足以完整理解 head。

## 3.6 表示子空间：Head 到底分工在哪里

为什么说 head 工作在不同表示子空间？

因为每个 head 的 Q/K/V 投影矩阵不同。它们把同一个 `X` 投影到不同低维空间：

```text
X -> Q_h/K_h/V_h in head h subspace
```

这些子空间可以学不同的匹配规则：

1. 位置相关规则。
2. 语法依赖规则。
3. 共指规则。
4. 分隔符规则。
5. 局部 n-gram 规则。
6. 长程引用规则。
7. 格式复制规则。

Clark 等人对 BERT 的分析发现，一些 attention heads 会关注 delimiter tokens、特定位置偏移，或者全句信息；也有一些 head 与句法关系、共指关系相关。

这说明 head 确实可能形成某些可观察模式。

但要小心：

```text
head 有模式，不等于 head 是稳定、唯一、不可替代的语义模块。
```

不同模型、不同层、不同随机种子、不同训练阶段，head 的分工都可能变化。

## 3.7 Head 数和 Head Dim 的 Trade-off

假设 hidden size 固定为 `D`，head 数是 `H`，每个 head 维度是：

```math
d_h=\frac{D}{H}
```

增加 head 数会带来：

1. 更多独立 attention maps。
2. 更多不同子空间路由。
3. 每个 head 的维度更小。

减少 head 数会带来：

1. 每个 head 维度更大。
2. attention maps 更少。
3. 可能减少某些并行路由模式。

所以 head 数不是越多越好。

如果 head 太少，模型可能缺少足够的并行关系建模能力。

如果 head 太多，每个 head 维度太小，单个 head 的表达能力可能受限，而且推理时 K/V cache 和内存访问成本也会上升。

现代 LLM 中，head 数、hidden size、KV head 数、tensor parallel 切分方式都会一起设计。它不是纯理论超参数，而是架构和系统共同约束的结果。

## 3.8 为什么有些 Head 可以剪掉

Michel 等人的研究发现，训练好的多头注意力模型中，很多 head 可以在测试时移除而不显著影响性能。有些层甚至可以减少到单个 head。

这似乎和“多头很重要”矛盾。

其实不矛盾。

原因可能包括：

1. 多头在训练时提供优化路径，但训练后部分 head 冗余。
2. 不同 head 之间功能重叠。
3. 某些任务只需要部分关系。
4. 残差连接和 FFN 可以补偿部分 head 删除。
5. 重要 head 分布不均，有些层更依赖多头。

这给我们的启发是：

```text
Multi-head attention 是一种容量和优化机制，不代表每个 head 都同等重要。
```

面试中如果被问“多头是否都必要”，不要简单回答“必要”。更好的回答是：多头整体提高表达和优化能力，但具体训练后可能存在冗余，可用于剪枝、压缩和解释分析。

## 3.9 Head 的可解释性：能看，但不能迷信

Attention head 可视化很直观。你可以画出 attention map，看某个 head 是否关注：

1. 前一个 token。
2. 标点符号。
3. `[CLS]` 或分隔符。
4. 主谓关系。
5. 代词和先行词。
6. 括号匹配。

这很有帮助，但有三个风险。

第一，raw attention 只是单层单 head 的权重。

第二，跨层后信息会混合。Abnar 和 Zuidema 指出，Transformer 中不同 token 的信息跨层不断混合，raw attention weights 作为解释探针并不可靠。

第三，最终行为还受 OV、FFN、残差、LayerNorm、logit lens 等影响。

因此，解释 head 时更严谨的做法是结合：

1. Attention pattern。
2. Head ablation。
3. Activation patching。
4. OV/QK 分解。
5. Input gradient 或 token ablation。
6. 任务行为变化。

简单说：

```text
Attention map 是线索，不是判决书。
```

## 3.10 为什么 MQA/GQA 要动 K/V Head

标准 MHA 中，每个 Q head 都有自己的 K/V head：

```text
num_q_heads = num_kv_heads = H
```

推理时，decoder-only 模型需要为每层每个历史 token 缓存 K/V。

如果 batch size 是 `B`，层数是 `L`，上下文长度是 `T`，KV head 数是 `H_{\mathrm{kv}}`，head dim 是 `d_h`，每个元素占 `b` 字节，那么 KV cache 粗略显存可以写成：

```math
M_{\mathrm{kv}}\approx 2BLTH_{\mathrm{kv}}d_hb
```

前面的 `2` 来自同时缓存 Key 和 Value。

这就是长上下文推理的核心成本之一。

MQA 的做法是：

```text
多个 Q heads 共享一组 K/V。
```

也就是：

```text
num_q_heads = H
num_kv_heads = 1
```

这样 KV cache 和内存带宽大幅下降，增量解码更快。

问题是 MQA 可能带来质量下降，因为所有 query heads 共享同一套 K/V 表示，表达能力受限。

GQA 是折中：

```text
num_q_heads = H
1 < num_kv_heads < H
```

每组 Q heads 共享一个 K/V head。GQA 试图获得接近 MHA 的质量和接近 MQA 的速度。

这解释了为什么现代 LLM 经常从 MHA 走向 GQA：

```text
不是因为多头不重要，而是因为推理阶段 K/V cache 太贵。
```

## 3.11 Q Head 和 KV Head 的分离直觉

为什么减少 K/V head，而不是减少 Q head？

因为 Q head 决定“当前 token 用多少种不同方式发起查询”。保留多个 Q heads，模型仍然可以从不同查询子空间发问。

K/V head 决定“历史 token 暴露多少套索引和内容表示”。减少 K/V heads 可以压缩缓存。

GQA 的直觉是：

```text
让多个 query 子空间共享一部分历史索引和内容表示，在表达能力和缓存成本之间折中。
```

可以类比为：

```text
很多人可以问不同问题，但共享同一个简化档案库。
```

MHA 是每类问题都有独立档案库。MQA 是所有问题共享一个档案库。GQA 是几组问题共享几套档案库。

## 3.12 代码：MHA 与 GQA 的形状差异

标准 MHA：

```text
Q: [B, H, N, d_h]
K: [B, H, N, d_h]
V: [B, H, N, d_h]
```

GQA：

```text
Q: [B, H_q, N, d_h]
K: [B, H_kv, N, d_h]
V: [B, H_kv, N, d_h]
```

其中：

```math
H_q>H_{\mathrm{kv}},\qquad H_q\bmod H_{\mathrm{kv}}=0
```

计算 attention 时，需要把 K/V 按组映射到 Q heads。

简化代码：

```python
import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [batch, num_kv_heads, seq_len, head_dim]
    if n_rep == 1:
        return x
    bsz, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :]
    x = x.expand(bsz, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)
```

如果：

```text
num_q_heads = 32
num_kv_heads = 8
```

那么：

```text
n_rep = 32 / 8 = 4
```

每个 K/V head 会服务 4 个 Q heads。

这个例子说明：MHA/MQA/GQA 的差异不是 attention 公式变了，而是 Q heads 和 K/V heads 的数量关系变了。

## 3.13 Head 和 Residual Stream

Transformer Circuits 的一个重要视角是 residual stream。

可以把每层的模块看成在 residual stream 上读写信息：

```math
r_{\ell+1}=r_\ell+\sum_{h=1}^{H}O_h(r_\ell)
```

这里省略了 normalization、输出投影、FFN 和具体层结构，只强调 attention heads 会从 residual stream 读取信息并把变换后的内容写回。

一个 head 做两件事：

1. 从 residual stream 中读 Q/K/V。
2. 通过 OV 写回 residual stream。

这使得不同 head、不同层可以组合成复杂电路。例如：

1. 早层 head 复制前文 token 信息。
2. 中层 head 组织语法和实体关系。
3. 后层 head 把相关信息送到预测位置。

这种说法是理解机制的有力工具，但不要过度简化成“第几层第几个 head 就固定负责某个任务”。真实大模型有大量 superposition、冗余和分布式表示。

## 3.14 表示子空间和 Superposition

所谓表示子空间，不是说每个 head 都有一个干净、人工可命名的语义空间。

更现实的情况是：

1. 模型隐藏维度有限。
2. 任务特征很多。
3. 不同特征可能叠加在同一组神经元或子空间中。
4. Head 之间可能共享和重叠功能。
5. 表示会随上下文动态变化。

这就是 mechanistic interpretability 中常说的 superposition 问题。

因此，“head 子空间”是一个有用抽象，但不是严格的人工分区。

面试中可以这样表达：

```text
Multi-head attention 给模型提供了多个低维投影和路由通道，每个 head 可以学习不同类型的匹配和信息写入。但这些功能是分布式、可重叠、可能冗余的，不应过度解释成固定语义模块。
```

## 3.15 面试中如何解释 Head 的作用

一个比较完整的回答可以分四层：

第一层，公式层：

```text
每个 head 有独立 W_Q、W_K、W_V，计算自己的 attention map 和 value mixture。
```

第二层，表达层：

```text
多个 head 让模型在不同表示子空间中并行建模不同关系。
```

第三层，机制层：

```text
QK circuit 决定读哪里，OV circuit 决定写什么。
```

第四层，工程层：

```text
head 数影响表达能力、KV cache、内存带宽和推理吞吐。MHA、MQA、GQA 是质量和推理成本之间的 trade-off。
```

这样回答比“多个头关注不同位置”更有深度。

## 3.16 常见误区

误区一：每个 head 都有清晰语义。

更准确：部分 head 可观察到语法、位置、delimiter 等模式，但很多 head 冗余或功能重叠。

误区二：剪掉 head 不影响性能，所以多头没用。

更准确：多头可能对训练、优化和鲁棒性有帮助，训练后存在冗余不等于训练时没用。

误区三：MHA 一定比 GQA 好。

更准确：MHA 表达更完整，但 GQA 在推理 KV cache 和速度上更优，现代 LLM 常需要系统 trade-off。

误区四：Q/K/V 分开只是实现细节。

更准确：Q/K/V 分开是匹配和内容传递解耦的关键设计。

误区五：Attention head 只负责读信息。

更准确：head 既通过 QK 决定读哪里，也通过 OV 决定写什么。

## 3.17 QKV 与多头子空间审计指标

把一个 head 只解释成“看哪里”容易过度简化。第二轮精修后，本章建议用下面这组指标把 QK 路由、OV 写入、head 冗余和 KV cache 成本一起讲清楚。

一个 head 审计样本可以写成：

```math
e_i=(x_i,h_i,t_i,f_i,a_i,o_i,z_i)
```

其中 `x_i` 是输入上下文，`h_i` 是 head 编号，`t_i` 是期望被读取的 token，`f_i` 是期望写入的目标特征，`a_i` 是 attention weights，`o_i` 是该 head 的输出向量，`z_i` 是人工或任务层面的验证结果。

QK 路由命中率：

```math
C_{\mathrm{qk}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\arg\max_j a_{ij}=t_i]
```

OV 写入支持率：

```math
C_{\mathrm{ov}}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\mathrm{top\_feature}(o_i)=f_i]
```

head 冗余率可以粗略定义为：

```math
R_{\mathrm{red}}=1-\frac{|\mathrm{unique}(\mathrm{signature}(h))|}{H}
```

其中 `signature(h)` 可以是一个简化的 head 功能签名，例如“主要读取哪个 token + 主要写入哪个特征”。真实模型里不能只靠这个定义做剪枝，但它适合解释“为什么有些 head 看起来重复”。

GQA 的 KV head 压缩比可以写成：

```math
\rho_{\mathrm{kv}}=\frac{H_q}{H_{\mathrm{kv}}}
```

KV cache 节省率：

```math
S_{\mathrm{kv}}=1-\frac{H_{\mathrm{kv}}}{H_q}
```

一个简化审计门禁可以写成：

```math
G_{\mathrm{head}}=\mathbb{1}[C_{\mathrm{qk}}\ge\tau_q\land C_{\mathrm{ov}}\ge\tau_o\land R_{\mathrm{red}}\le\tau_r\land S_{\mathrm{kv}}\ge\tau_s]
```

这组公式的重点是面试表达：多头不是只有 attention map，还要同时看 QK 是否读对、OV 是否写对、head 是否冗余，以及减少 K/V head 是否真的带来缓存收益。

### 3.17.1 最小可运行 QKV / Head 子空间审计 demo

下面的 demo 不依赖 PyTorch，只用 Python 标准库模拟 5 个 attention heads。它展示三件事：QK 路由命中不等于 OV 一定写对；两个 head 可能形成相似功能签名；从 MHA 到 GQA / MQA 时 KV cache 成本如何按 `num_kv_heads` 下降。

```python
import math

tokens = ["prev_token", "subject", "delimiter", "evidence", "distractor"]
feature_names = ["local_copy", "entity_state", "format_boundary", "evidence_fact"]

key_bank = {
    "prev_token": [1.0, 0.1, 0.0, 0.0],
    "subject": [0.1, 1.0, 0.0, 0.0],
    "delimiter": [0.0, 0.1, 1.0, 0.0],
    "evidence": [0.1, 0.4, 0.0, 1.0],
    "distractor": [0.7, 0.7, 0.0, 0.1],
}

heads = [
    {"name": "previous_token_head", "query": [1.0, 0.0, 0.0, 0.0], "target": "prev_token", "feature": "local_copy"},
    {"name": "subject_coref_head", "query": [0.0, 1.0, 0.0, 0.0], "target": "subject", "feature": "entity_state"},
    {"name": "delimiter_head", "query": [0.0, 0.0, 1.0, 0.0], "target": "delimiter", "feature": "format_boundary"},
    {"name": "delimiter_copy_head", "query": [0.0, 0.0, 0.9, 0.0], "target": "delimiter", "feature": "format_boundary"},
    {"name": "misleading_attention_head", "query": [0.7, 0.7, 0.0, 0.0], "target": "evidence", "feature": "evidence_fact"},
]

value_features = {
    "prev_token": [1.0, 0.1, 0.0, 0.0],
    "subject": [0.0, 1.0, 0.1, 0.0],
    "delimiter": [0.0, 0.0, 1.0, 0.0],
    "evidence": [0.0, 0.2, 0.0, 1.0],
    "distractor": [0.3, 0.3, 0.0, 0.1],
}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [x / total for x in exps]


def weighted_sum(weights, rows):
    return [sum(w * row[i] for w, row in zip(weights, rows)) for i in range(len(rows[0]))]


records = {}
for head in heads:
    scores = [dot(head["query"], key_bank[token]) / math.sqrt(len(head["query"])) for token in tokens]
    weights = softmax(scores)
    top_token = tokens[max(range(len(tokens)), key=lambda i: weights[i])]
    output = weighted_sum(weights, [value_features[token] for token in tokens])
    top_feature = feature_names[max(range(len(feature_names)), key=lambda i: output[i])]
    signature = (top_token, top_feature)
    records[head["name"]] = {
        "top_token": top_token,
        "target": head["target"],
        "qk_hit": top_token == head["target"],
        "top_feature": top_feature,
        "expected_feature": head["feature"],
        "ov_hit": top_feature == head["feature"],
        "signature": signature,
    }

unique_signatures = {tuple(record["signature"]) for record in records.values()}
metrics = {
    "qk_route_hit": round(sum(r["qk_hit"] for r in records.values()) / len(records), 3),
    "ov_write_hit": round(sum(r["ov_hit"] for r in records.values()) / len(records), 3),
    "head_redundancy": round(1 - len(unique_signatures) / len(records), 3),
    "interpretation_risk": round(
        sum(r["qk_hit"] and not r["ov_hit"] or not r["qk_hit"] for r in records.values()) / len(records), 3
    ),
}


def kv_cache_mib(batch, layers, seq_len, kv_heads, head_dim, bytes_per_elem):
    bytes_used = 2 * batch * layers * seq_len * kv_heads * head_dim * bytes_per_elem
    return round(bytes_used / 1024**2, 1)


kv_cache = {
    "mha_32kv": kv_cache_mib(1, 32, 4096, 32, 128, 2),
    "gqa_8kv": kv_cache_mib(1, 32, 4096, 8, 128, 2),
    "mqa_1kv": kv_cache_mib(1, 32, 4096, 1, 128, 2),
}

failed_gates = []
if metrics["qk_route_hit"] < 0.8:
    failed_gates.append("qk_route")
if metrics["ov_write_hit"] < 0.8:
    failed_gates.append("ov_write")
if metrics["head_redundancy"] > 0.25:
    failed_gates.append("head_redundancy")
if metrics["interpretation_risk"] > 0.2:
    failed_gates.append("interpretation_boundary")
if kv_cache["gqa_8kv"] >= kv_cache["mha_32kv"]:
    failed_gates.append("kv_cache_saving")

print(f"records={records}")
print(f"metrics={metrics}")
print(f"kv_cache_mib={kv_cache}")
print(f"failed_gates={failed_gates}")
print(f"qkv_head_gate_pass={not failed_gates}")
```

运行结果应类似：

```text
records={'previous_token_head': {'top_token': 'prev_token', 'target': 'prev_token', 'qk_hit': True, 'top_feature': 'local_copy', 'expected_feature': 'local_copy', 'ov_hit': True, 'signature': ('prev_token', 'local_copy')}, 'subject_coref_head': {'top_token': 'subject', 'target': 'subject', 'qk_hit': True, 'top_feature': 'entity_state', 'expected_feature': 'entity_state', 'ov_hit': True, 'signature': ('subject', 'entity_state')}, 'delimiter_head': {'top_token': 'delimiter', 'target': 'delimiter', 'qk_hit': True, 'top_feature': 'format_boundary', 'expected_feature': 'format_boundary', 'ov_hit': True, 'signature': ('delimiter', 'format_boundary')}, 'delimiter_copy_head': {'top_token': 'delimiter', 'target': 'delimiter', 'qk_hit': True, 'top_feature': 'format_boundary', 'expected_feature': 'format_boundary', 'ov_hit': True, 'signature': ('delimiter', 'format_boundary')}, 'misleading_attention_head': {'top_token': 'distractor', 'target': 'evidence', 'qk_hit': False, 'top_feature': 'entity_state', 'expected_feature': 'evidence_fact', 'ov_hit': False, 'signature': ('distractor', 'entity_state')}}
metrics={'qk_route_hit': 0.8, 'ov_write_hit': 0.8, 'head_redundancy': 0.2, 'interpretation_risk': 0.2}
kv_cache_mib={'mha_32kv': 2048.0, 'gqa_8kv': 512.0, 'mqa_1kv': 64.0}
failed_gates=[]
qkv_head_gate_pass=True
```

这段 demo 的重点不是模拟真实 attention head，而是把面试里的四个关键点连起来：QK 决定读哪里，OV 决定写什么，多头可能存在冗余，GQA / MQA 的缓存收益来自减少 `H_kv`。

## 3.18 面试题

### 题 1：Multi-head attention 为什么有用？

参考回答：

```text
Multi-head attention 让模型在多个表示子空间中并行做信息路由。每个 head 有独立的 Q/K/V 投影，可以学习不同的匹配规则和内容变换，例如局部位置、语法依赖、共指、分隔符、长程引用等。所有 head 的输出再拼接并通过输出投影混合回主 hidden space。它比单头更灵活，但训练后部分 head 可能冗余。
```

### 题 2：QK circuit 和 OV circuit 分别是什么？

参考回答：

```text
QK circuit 决定 attention head 从哪里读信息，也就是 Query 和 Key 的匹配产生 attention pattern。OV circuit 决定读到的信息如何写回 residual stream，也就是 Value projection 和 Output projection 的组合。只看 attention map 只能看到读哪里，不能看到写什么，所以分析 head 时要同时考虑 QK 和 OV。
```

### 题 3：为什么有些 attention head 可以剪掉？

参考回答：

```text
训练后的多头注意力里可能存在冗余。不同 head 的功能可能重叠，某些任务只依赖部分 head，残差和 FFN 也能补偿一部分删除带来的影响。研究发现很多 head 可在测试时移除而不明显影响性能。但这不说明多头没用，因为多头可能在训练优化、表达容量和鲁棒性上仍有作用。
```

### 题 4：MHA、MQA、GQA 的核心区别是什么？

参考回答：

```text
MHA 是每个 query head 都有自己的 key/value head，表达能力强但 KV cache 大。MQA 是多个 query heads 共享一组 key/value，大幅降低增量解码时的 KV cache 和内存带宽，但可能有质量损失。GQA 是折中，让一组 query heads 共享一个 key/value head，减少 KV cache，同时尽量接近 MHA 的质量。
```

### 题 5：为什么减少 K/V head 能加速推理？

参考回答：

```text
Decoder-only 增量推理时，每生成一个 token 都要读取历史 token 的 K/V cache。KV cache 大小和层数、序列长度、KV head 数、head dim 成正比。减少 K/V head 可以减少缓存显存和内存带宽需求，因此 MQA/GQA 能明显提升长上下文和大 batch 推理效率。
```

### 题 6：Attention head 是否可解释？

参考回答：

```text
可以有限解释，但不能迷信。可视化 attention map 能发现一些 head 关注 delimiter、固定位置、语法关系或共指关系。但 raw attention 只是局部权重，跨层后信息会混合，最终行为还取决于 OV、FFN、残差和后续层。更严谨的解释需要结合 ablation、activation patching、attention flow、QK/OV 分解等方法。
```

## 3.19 小练习

1. 手写 MHA 中 Q、K、V 的张量形状，并解释每一维含义。
2. 用自己的话解释“QK 决定读哪里，OV 决定写什么”。
3. 画一个 4-head attention 的示意图，给每个 head 假设一种可能功能。
4. 阅读 *What Does BERT Look At?* 摘要，列出 BERT head 可能学到的几类 pattern。
5. 阅读 *Are Sixteen Heads Really Better than One?* 摘要，解释 head pruning 说明了什么、没有说明什么。
6. 给出 MHA、MQA、GQA 的 KV cache 大小对比公式。
7. 用 PyTorch 实现一个 `repeat_kv` 函数，并解释它在 GQA 中的作用。
8. 运行本章 QKV / Head 子空间审计 demo，解释为什么 `misleading_attention_head` 同时暴露 QK 路由和 OV 写入风险。

## 3.20 本章总结

本章深入讲了 Q、K、V、attention head 和表示子空间。

核心结论：

1. Q/K/V 分开是为了把匹配逻辑和内容传递解耦。
2. Multi-head attention 让模型在多个表示子空间中并行做信息路由。
3. 一个 head 不只是 attention map，还包括 QK circuit 和 OV circuit。
4. QK 决定读哪里，OV 决定写什么。
5. Attention head 可能学到语法、位置、共指、delimiter 等 pattern，但也可能冗余。
6. Head 可解释性需要谨慎，raw attention 不能直接等同于因果解释。
7. Head 数和 head dim 是表达能力、优化、KV cache 和推理吞吐之间的 trade-off。
8. MQA/GQA 减少 K/V heads，是为了解决增量推理中的 KV cache 和内存带宽瓶颈。
9. 现代 LLM 的 attention 设计已经从纯模型结构问题，变成模型架构和 serving 系统共同优化问题。

下一章会系统比较 MHA、MQA、GQA、MLA 的演进与 trade-off，重点讲为什么现代大模型越来越重视 KV cache 压缩和推理效率。
