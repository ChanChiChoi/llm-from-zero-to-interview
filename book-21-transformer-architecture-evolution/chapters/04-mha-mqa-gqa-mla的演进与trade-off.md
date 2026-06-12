# 第四章：MHA、MQA、GQA、MLA 的演进与 Trade-off

## 0. 本讲资料边界与第二轮精修口径

本章第二轮精修以公开论文、技术报告和前序章节为资料边界：Transformer 原论文对 MHA 的定义，Shazeer 的 MQA 论文对增量解码内存带宽瓶颈的说明，Ainslie 等人的 GQA 论文对 MHA / MQA 折中和 uptraining 的讨论，Mistral 7B 技术报告中 GQA 与 Sliding Window Attention 的公开用法，DeepSeek-V2 / V3 技术报告中 MLA 对 KV cache 的低秩联合压缩设计，vLLM / PagedAttention 论文对 serving 侧 KV cache 分页管理的说明，以及 FlashAttention-2 对 attention kernel IO 和并行划分的优化边界。

本章只讨论公开可验证的 attention 架构与 serving 成本关系；不推断闭源模型的未公开实现，不把 DeepSeek-V2 报告中的具体节省比例机械推广到所有模型，也不把 toy demo 的选型分数当成真实 benchmark。

第二轮补强重点有三点：

1. 将 KV cache 显存、KV head 比例、GQA 特例、MLA latent cache 和架构门禁改成稳定 MathJax 表达。
2. 明确 FlashAttention、PagedAttention、MQA / GQA / MLA 分别位于 kernel、serving memory manager 和 model architecture 三个层面。
3. 补一个 0 依赖 Python demo，审计 MHA / GQA / MQA / MLA 的 cache MiB、相对节省率、quality gate、maturity gate 和 serving 选型。

## 4.1 本章定位

前两章已经讲清楚 Self-Attention、Q/K/V、attention head 和表示子空间。本章进入现代大模型注意力结构的工程演进：MHA、MQA、GQA、MLA。

如果只从训练论文看，Multi-Head Attention 已经足够优雅。但到了大模型推理系统里，一个更现实的问题出现了：

```text
生成每个 token 时，模型都要读取历史 token 的 K/V cache。
```

上下文越长、batch 越大、层数越多、KV head 越多，KV cache 越大，内存带宽和显存压力越高。于是注意力结构的优化目标从“表达能力更强”逐渐变成：

```text
表达能力、KV cache、吞吐、延迟、显存、长上下文之间如何折中？
```

学完本章，你应该能回答：

1. MHA、MQA、GQA、MLA 分别是什么。
2. 为什么 MHA 表达强但推理贵。
3. 为什么 MQA 能显著降低 KV cache，但可能损失质量。
4. 为什么 GQA 成为现代 LLM 常用折中方案。
5. MLA 为什么进一步把 K/V cache 压缩成 latent cache。
6. FlashAttention、PagedAttention 和 MQA/GQA/MLA 分别优化什么层面。
7. 面试中如何系统解释 attention 架构演进。

## 4.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Vaswani et al., 2017, *Attention Is All You Need*。提出 Multi-Head Attention，让模型在多个表示子空间联合关注信息。
2. Shazeer, 2019, *Fast Transformer Decoding: One Write-Head is All You Need*。提出 Multi-Query Attention，指出增量解码时加载大量 K/V tensor 带来内存带宽瓶颈，共享 K/V 可显著加速。
3. Ainslie et al., 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*。提出 Grouped-Query Attention，在 MHA 和 MQA 之间折中，质量接近 MHA、速度接近 MQA。
4. Jiang et al., 2023, *Mistral 7B*。公开说明 Mistral 7B 使用 GQA 加速推理，并结合 Sliding Window Attention 降低长序列成本。
5. DeepSeek-AI, 2024, *DeepSeek-V2*。提出 Multi-head Latent Attention，通过把 KV cache 显著压缩到 latent vector 中提升推理效率。
6. DeepSeek-AI, 2024/2025, *DeepSeek-V3 Technical Report*。继续采用 MLA 和 DeepSeekMoE，说明 MLA 已被大规模 MoE 模型验证。
7. Kwon et al., 2023, *PagedAttention/vLLM*。从 serving 系统角度解决 KV cache 动态增长、碎片和重复的问题。
8. Dao, 2023, *FlashAttention-2*。从 GPU kernel 和 IO 角度优化 attention 计算，强调 attention 是长序列扩展主要瓶颈。

需要注意：不同报告对 MLA 的实现细节公开程度不同，本章只基于公开论文和技术报告解释核心思想，不臆测闭源实现。

## 4.3 为什么注意力结构开始围绕 KV Cache 演进

训练时，Transformer 可以一次并行处理整个序列。

推理时，decoder-only LLM 是逐 token 生成：

```text
prefix tokens -> generate token t -> append token t -> generate token t+1
```

为了避免每一步重新计算所有历史 token 的 K/V，系统会缓存每层每个历史 token 的 K/V，这就是 KV cache。

对一个 decoder-only 模型，如果 batch size 是 `B`，层数是 `L`，上下文长度是 `T`，KV head 数是 `H_kv`，每个 head 的维度是 `d_h`，每个元素占 `b` 字节，KV cache 粗略显存可以写成：

```math
M_{\mathrm{kv}}\approx 2BLTH_{\mathrm{kv}}d_hb
```

前面的 `2` 来自 K 和 V 两份缓存。

这说明几个事实：

1. 上下文长度翻倍，KV cache 约翻倍。
2. batch size 翻倍，KV cache 约翻倍。
3. KV head 数越多，cache 越大。
4. 长上下文 serving 时，显存经常被 KV cache 吃掉，而不只是模型权重。
5. 解码每个新 token 时，需要反复读取历史 K/V，内存带宽会成为瓶颈。

因此，现代注意力结构演进的主线之一就是：

```text
如何少存 K/V，少读 K/V，同时尽量不损失模型质量？
```

## 4.4 MHA：表达完整，但 KV Cache 昂贵

标准 Multi-Head Attention 中：

```math
H_q=H_{\mathrm{kv}}=H
```

每个 query head 有自己对应的 key head 和 value head。

张量形状通常是：

```text
Q: [B, H, N, d_h]
K: [B, H, N, d_h]
V: [B, H, N, d_h]
```

MHA 的优势：

1. 每个 head 有独立 Q/K/V 子空间。
2. 表达能力完整。
3. 训练和理论最标准。
4. 原始 Transformer、BERT、早期 GPT 类模型都大量使用。

MHA 的代价：

1. 每个 head 都要缓存 K/V。
2. 长上下文时 cache 显存高。
3. 解码时内存带宽压力大。
4. batch 做大时 KV cache 更容易成为瓶颈。

如果模型有 `32` 个 heads，MHA 就有 `32` 个 K heads 和 `32` 个 V heads。对于长上下文和高并发 serving，这很贵。

所以 MHA 是“表达优先”的设计，而不是“推理成本优先”的设计。

## 4.5 MQA：一个 K/V Head 服务所有 Query Heads

Multi-Query Attention 的核心思想很直接：

```text
保留多个 query heads，但让所有 query heads 共享同一组 K/V。
```

也就是：

```math
H_q=H,\qquad H_{\mathrm{kv}}=1
```

形状变成：

```text
Q: [B, H, N, d_h]
K: [B, 1, N, d_h]
V: [B, 1, N, d_h]
```

这样 KV cache 直接从 `H` 份降到 `1` 份。

MQA 的优势：

1. KV cache 大幅减少。
2. 解码时内存带宽需求降低。
3. 增量推理速度明显提升。
4. 长上下文和大 batch 更友好。

MQA 的代价：

1. 所有 Q heads 共享同一套 K/V 表示。
2. 表达能力可能下降。
3. 某些任务或模型规模下质量可能受影响。
4. 从已有 MHA checkpoint 直接改成 MQA 需要额外处理或再训练。

Shazeer 的论文标题“One Write-Head is All You Need”很形象：推理慢的关键不是 query 数，而是重复读取大量 K/V。把 K/V 合并，可以显著降低带宽压力。

## 4.6 GQA：MHA 和 MQA 之间的折中

Grouped-Query Attention 是更实用的折中。

它让一组 query heads 共享一个 K/V head：

```math
H_q=H,\qquad 1<H_{\mathrm{kv}}<H
```

例如：

```text
num_q_heads = 32
num_kv_heads = 8
```

那么每个 K/V head 服务 `4` 个 query heads。

GQA 的直觉：

```text
不要像 MHA 那样每个 query head 都有独立 K/V，也不要像 MQA 那样所有 query heads 只共享一套 K/V，而是分组共享。
```

GQA 的优势：

1. KV cache 明显小于 MHA。
2. 表达能力通常好于 MQA。
3. 推理速度接近 MQA。
4. 可以从 MHA checkpoint uptraining 得到。
5. 已被 Mistral 7B 等公开模型采用。

GQA 的代价：

1. 仍然比 MQA cache 大。
2. 质量和速度依赖 `num_kv_heads` 选择。
3. 实现中需要处理 Q heads 到 KV heads 的映射。

GQA 是现代 LLM 很常见的工程选择，因为它把质量损失和推理收益平衡得更好。

## 4.7 MHA、MQA、GQA 的统一视角

可以把三者看成同一个轴上的不同点：

| 结构 | Q heads | KV heads | KV cache | 表达能力 | 推理效率 |
|---|---:|---:|---:|---:|---:|
| MHA | H | H | 最大 | 最完整 | 最低 |
| GQA | H | G | 中等 | 接近 MHA | 接近 MQA |
| MQA | H | 1 | 最小 | 可能下降 | 最高 |

其中：

```math
1\le H_{\mathrm{kv}}\le H_q
```

所以：

```math
\mathrm{MHA}: H_{\mathrm{kv}}=H_q,\qquad \mathrm{MQA}: H_{\mathrm{kv}}=1
```

这也是为什么 GQA 可以叫 generalized multi-query attention。

面试中可以这样概括：

```text
MHA、MQA、GQA 的核心差异不是 query heads，而是 KV heads 数量。减少 KV heads 可以降低 KV cache 和内存带宽，但会牺牲一部分 K/V 表示多样性。GQA 是质量和推理成本之间的折中。
```

## 4.8 MLA：从减少 KV Head 到压缩 KV 表示

DeepSeek-V2 提出的 Multi-head Latent Attention 进一步推进了这个方向。

GQA/MQA 的思路是：

```text
减少 KV head 数。
```

MLA 的思路更进一步：

```text
不要直接缓存完整 K/V，而是把 K/V 相关信息压缩成 latent cache。
```

DeepSeek-V2 报告明确说，MLA 通过把 KV cache 显著压缩到 latent vector 中，保证高效推理，并报告 KV cache 减少 93.3%、最大生成吞吐提升 5.76 倍。

从直觉上看，MLA 想解决的问题是：

1. MHA cache 太大。
2. MQA/GQA 靠共享 K/V 减 cache，但可能限制表达。
3. 能否保留多头 attention 的表达，同时用更紧凑的 latent 表示存历史信息？

可以抽象为：

```math
c_t^{\mathrm{kv}}=x_tW_d^{\mathrm{kv}}
```

```math
k_t=c_t^{\mathrm{kv}}W_u^k,\qquad v_t=c_t^{\mathrm{kv}}W_u^v
```

MLA 的价值在于把注意力结构和 serving 成本更深地绑定起来。它不是单纯换一个 head 数，而是在 K/V 表示本身上做压缩设计。

## 4.9 MLA 和 MoE 的关系

DeepSeek-V2/V3 都是 MoE 模型，同时使用 MLA。

这不是偶然。

MoE 的特点是：

1. 总参数量很大。
2. 每个 token 只激活部分专家。
3. 训练计算可以相对经济。
4. serving 时仍然面临显存、通信、batch、KV cache 等压力。

如果一个 MoE 模型支持长上下文，KV cache 仍然会随层数、长度、batch 增长。即使每 token 激活参数较少，KV cache 也不会自动消失。

因此 MLA 对 MoE 很重要：

```text
MoE 降低每 token 计算成本，MLA 降低长上下文推理时的 KV cache 成本。
```

这说明现代大模型架构优化通常不是单点优化，而是组合拳：

```text
MoE + MLA + 高效并行 + FP8/量化 + serving 系统优化
```

## 4.10 FlashAttention、PagedAttention 和 MQA/GQA/MLA 的区别

这些名字容易混在一起，但它们优化层面不同。

FlashAttention 优化的是 attention kernel。

它不改变 attention 数学结果，而是利用 GPU memory hierarchy，减少 HBM 读写，提高计算效率。FlashAttention-2 进一步优化并行和 work partitioning。

PagedAttention 优化的是 KV cache 管理。

它借鉴操作系统分页思想，减少 KV cache 的内存碎片和重复，提升 serving batch 能力。

MQA/GQA/MLA 优化的是模型架构中的 K/V 表示。

它们改变 KV heads 或 KV cache 表示，从源头减少需要存和读的 K/V。

可以总结为：

| 方法 | 优化层面 | 是否改变模型结构 | 主要目标 |
|---|---|---|---|
| FlashAttention | kernel/IO | 否 | 更快更省显存地算 attention |
| PagedAttention | serving memory manager | 否 | 更高效管理 KV cache |
| MQA | attention 架构 | 是 | 用 1 个 KV head 降低 cache |
| GQA | attention 架构 | 是 | 分组 KV head，平衡质量和效率 |
| MLA | attention 架构 | 是 | 用 latent cache 压缩 K/V 表示 |

面试中一定要区分：FlashAttention 不是 MQA，PagedAttention 也不是 GQA。

## 4.11 KV Cache 粗略计算

假设：

```text
L = 32
B = 16
T = 8192
d_h = 128
b = 2
```

如果 MHA 有 `32` 个 KV heads：

```math
M_{\mathrm{mha}}\approx 2\times 16\times 32\times 8192\times 32\times 128\times 2
```

这里第一个 `2` 是 K/V 两份，最后的 `2` 是 bf16 bytes。

如果 GQA 改成 `8` 个 KV heads，cache 约变成 MHA 的：

```math
\frac{8}{32}=\frac{1}{4}
```

如果 MQA 改成 `1` 个 KV head，cache 约变成 MHA 的：

```math
\frac{1}{32}
```

这就是为什么 KV head 数对 serving 成本影响巨大。

当然，实际系统还要考虑 padding、碎片、block size、并行切分、cache quantization、prefix sharing、paged allocation 等因素。

## 4.12 训练和推理视角不同

从训练视角看：

1. MHA 表达最自然。
2. 并行计算完整序列，KV cache 不是同一个问题。
3. 训练关注 FLOPs、通信、激活显存、稳定性。

从推理视角看：

1. 逐 token decode，无法像训练那样完全并行序列维。
2. KV cache 成为长期状态。
3. 内存带宽、显存容量、batch packing、prefix sharing 非常关键。
4. MQA/GQA/MLA 的收益更明显。

这也是为什么很多架构创新越来越“推理优先”。

过去常见思路是：

```text
先训练一个好模型，再想办法部署。
```

现在更常见的是：

```text
训练前就考虑 serving 成本，把推理约束写进架构设计。
```

## 4.13 什么时候选 MHA、MQA、GQA、MLA

如果是小模型、短上下文、研究实验：

1. MHA 简单直接。
2. 对质量最保守。
3. 实现和分析最容易。

如果是追求极致解码速度、能接受一定质量折中：

1. MQA 很有吸引力。
2. KV cache 最省。
3. 适合带宽瓶颈明显的场景。

如果是通用 LLM serving：

1. GQA 往往是更稳的折中。
2. 现代开源模型常用。
3. 兼顾质量、cache、吞吐。

如果是超大模型、长上下文、MoE、高并发 serving：

1. MLA 这类 latent cache 方案更值得关注。
2. 它从表示层面进一步压缩 cache。
3. 实现复杂度和复现难度也更高。

工程选型永远不是只看论文效果，而是看：

```text
质量目标 + 上下文长度 + batch 并发 + 显存预算 + 延迟要求 + 实现复杂度
```

## 4.14 Attention 架构成本审计指标

为了把 MHA / MQA / GQA / MLA 的 trade-off 讲清楚，可以把候选注意力结构抽象成：

```math
a_i=(H_q,H_{\mathrm{kv}},d_h,d_c,d_r,q_i,m_i,c_i)
```

其中 `H_q` 是 query head 数，`H_kv` 是 KV head 数，`d_h` 是 head dim，`d_c` 是 latent cache dim，`d_r` 是额外位置相关缓存维度，`q_i` 是质量代理分，`m_i` 是工程成熟度，`c_i` 是实现复杂度。

普通 MHA / MQA / GQA 的 KV cache 可以写成：

```math
M_{\mathrm{kv}}(a_i)=2BLTH_{\mathrm{kv}}d_hb
```

MLA 这类 latent cache 的粗略缓存可以抽象为：

```math
M_{\mathrm{mla}}(a_i)\approx BLT(d_c+d_r)b
```

其中 `d_c` 是压缩 latent 维度，`d_r` 可以表示需要额外保留的位置相关 Key 部分。这个公式只是教学抽象，具体实现要以公开技术报告和源码为准。

相对 MHA 的节省率：

```math
S_{\mathrm{save}}(a_i)=1-\frac{M(a_i)}{M_{\mathrm{mha}}}
```

Query head 到 KV head 的共享比例：

```math
\rho_{\mathrm{qkv}}=\frac{H_q}{H_{\mathrm{kv}}}
```

一个简化上线门禁可以写成：

```math
G_{\mathrm{attn}}=\mathbb{1}[q_i\ge\tau_q\land S_{\mathrm{save}}(a_i)\ge\tau_s\land m_i\ge\tau_m]
```

这组指标不替代真实 ablation 和 serving 压测，但能让面试表达从“GQA 更快”升级为“它用多少 KV head 换来多少 cache 节省、可能损失多少质量、实现成熟度是否足够”。

### 4.14.1 最小可运行 MHA / GQA / MQA / MLA 成本审计 demo

下面的 demo 不依赖深度学习框架，只用标准库估算四种 attention 结构在同一 serving 场景下的 KV cache MiB、相对 MHA 节省率和简化选型 gate。

```python
configs = [
    {"name": "mha_32kv", "q_heads": 32, "kv_heads": 32, "head_dim": 128, "quality": 1.00, "maturity": 0.95, "complexity": 0.10},
    {"name": "gqa_8kv", "q_heads": 32, "kv_heads": 8, "head_dim": 128, "quality": 0.98, "maturity": 0.90, "complexity": 0.20},
    {"name": "mqa_1kv", "q_heads": 32, "kv_heads": 1, "head_dim": 128, "quality": 0.92, "maturity": 0.82, "complexity": 0.18},
    {"name": "mla_latent", "q_heads": 32, "kv_heads": None, "head_dim": 128, "latent_dim": 512, "rope_dim": 64, "quality": 0.99, "maturity": 0.65, "complexity": 0.45},
]

batch = 16
layers = 32
seq_len = 8192
bytes_per_elem = 2


def cache_mib(cfg):
    if cfg["kv_heads"] is None:
        elems = batch * layers * seq_len * (cfg["latent_dim"] + cfg["rope_dim"])
    else:
        elems = 2 * batch * layers * seq_len * cfg["kv_heads"] * cfg["head_dim"]
    return round(elems * bytes_per_elem / 1024**2, 1)


mha_cache = cache_mib(configs[0])
records = {}
for cfg in configs:
    cache = cache_mib(cfg)
    reduction = round(1 - cache / mha_cache, 3)
    if cfg["kv_heads"] is None:
        qkv_ratio = None
    else:
        qkv_ratio = round(cfg["q_heads"] / cfg["kv_heads"], 3)
    cache_score = min(reduction / 0.95, 1.0)
    score = (
        0.55 * cfg["quality"]
        + 0.25 * cache_score
        + 0.20 * cfg["maturity"]
        - 0.05 * cfg["complexity"]
    )
    failed = []
    if cfg["quality"] < 0.95:
        failed.append("quality")
        score -= 0.08
    if reduction < 0.50:
        failed.append("cache_saving")
    if cfg["maturity"] < 0.70:
        failed.append("maturity")
        score -= 0.04
    records[cfg["name"]] = {
        "cache_mib": cache,
        "reduction_vs_mha": reduction,
        "qkv_ratio": qkv_ratio,
        "serving_score": round(score, 3),
        "failed_gates": failed,
    }

ranked = sorted(records.items(), key=lambda item: item[1]["serving_score"], reverse=True)
ready = [name for name, record in ranked if not record["failed_gates"]]

print(f"records={records}")
print(f"ranked={[name for name, _ in ranked]}")
print(f"ready={ready}")
print(f"best_ready={ready[0] if ready else None}")
```

运行结果应类似：

```text
records={'mha_32kv': {'cache_mib': 65536.0, 'reduction_vs_mha': 0.0, 'qkv_ratio': 1.0, 'serving_score': 0.735, 'failed_gates': ['cache_saving']}, 'gqa_8kv': {'cache_mib': 16384.0, 'reduction_vs_mha': 0.75, 'qkv_ratio': 4.0, 'serving_score': 0.906, 'failed_gates': []}, 'mqa_1kv': {'cache_mib': 2048.0, 'reduction_vs_mha': 0.969, 'qkv_ratio': 32.0, 'serving_score': 0.831, 'failed_gates': ['quality']}, 'mla_latent': {'cache_mib': 4608.0, 'reduction_vs_mha': 0.93, 'qkv_ratio': None, 'serving_score': 0.857, 'failed_gates': ['maturity']}}
ranked=['gqa_8kv', 'mla_latent', 'mqa_1kv', 'mha_32kv']
ready=['gqa_8kv']
best_ready=gqa_8kv
```

这段 demo 的重点是：MHA 质量最稳但 cache 太大，MQA cache 最小但可能有质量门禁，MLA cache 很省但实现成熟度和复现成本要单独评估，GQA 常成为通用 serving 的稳健折中。

## 4.15 常见误区

误区一：MQA/GQA 只是为了减少参数。

更准确：主要是减少 KV cache 和内存带宽，不是参数量优化。

误区二：GQA 一定比 MHA 差很多。

更准确：合适的 GQA 可以质量接近 MHA，同时显著提高推理效率。

误区三：FlashAttention 能解决 KV cache 过大。

更准确：FlashAttention 优化 attention 计算 IO，不从架构上减少历史 K/V cache 总量。

误区四：PagedAttention 是一种 attention 结构。

更准确：PagedAttention 是 serving 系统中的 KV cache 管理算法。

误区五：MLA 只是 GQA 的另一种名字。

更准确：GQA 减少 KV head 数，MLA 压缩 KV 表示到 latent cache，层级不同。

## 4.16 面试题

### 题 1：MHA、MQA、GQA 的区别是什么？

参考回答：

```text
MHA 中每个 query head 都有独立 K/V head，表达能力强但 KV cache 大。MQA 保留多个 query heads，但所有 query heads 共享一个 K/V head，显著降低 KV cache 和内存带宽，但可能有质量损失。GQA 是中间方案，让一组 query heads 共享一个 K/V head，质量通常接近 MHA，推理效率接近 MQA。
```

### 题 2：为什么 MQA/GQA 能提升推理速度？

参考回答：

```text
Decoder-only 增量解码时，每生成一个 token 都要读取历史 token 的 K/V cache。KV cache 大小和 KV head 数成正比。MQA/GQA 减少 KV head 数，从而减少显存占用和内存带宽需求。很多推理场景不是纯算力瓶颈，而是读取 K/V 的带宽瓶颈，所以减少 K/V 能显著提升吞吐。
```

### 题 3：GQA 为什么比 MQA 更常见？

参考回答：

```text
MQA 只有一个 K/V head，cache 最省，但所有 query heads 共享同一套 K/V，可能带来质量损失。GQA 保留多个 K/V groups，让一组 query heads 共享 K/V，在表达能力和推理效率之间折中。实践中 GQA 往往能接近 MHA 质量，同时获得接近 MQA 的推理收益，所以现代 LLM 很常用。
```

### 题 4：MLA 和 GQA 的区别是什么？

参考回答：

```text
GQA 的核心是减少 KV head 数，让多个 query heads 分组共享 K/V head。MLA 的核心是把 K/V 相关信息压缩成 latent cache，再在 attention 时使用或恢复相关表示。也就是说，GQA 主要在 head 数量上省 cache，MLA 更进一步在 K/V 表示本身上做压缩。DeepSeek-V2 报告显示 MLA 能显著降低 KV cache 并提升吞吐。
```

### 题 5：FlashAttention、PagedAttention、GQA 分别优化什么？

参考回答：

```text
FlashAttention 优化 attention kernel 和 GPU IO，不改变数学结果。PagedAttention 优化 serving 系统中的 KV cache 管理，减少碎片和重复。GQA 改变模型 attention 架构，减少 KV head 数，从源头降低 KV cache。三者可以组合使用，分别对应 kernel、memory manager 和 model architecture 三个层面。
```

## 4.17 小练习

1. 推导 MHA、MQA、GQA 的 KV cache 大小比例。
2. 假设 `num_q_heads=40`、`num_kv_heads=8`，说明每个 KV head 服务几个 Q heads。
3. 解释为什么 KV cache 是长上下文 serving 的核心瓶颈之一。
4. 阅读 MQA 论文摘要，找出“memory-bandwidth cost”这句话对应的工程问题。
5. 阅读 GQA 论文摘要，解释为什么它是 MHA 和 MQA 的折中。
6. 阅读 DeepSeek-V2 摘要，解释 MLA、MoE、KV cache 和吞吐之间的关系。
7. 用一张表区分 FlashAttention、PagedAttention、MQA、GQA、MLA。
8. 运行本章成本审计 demo，解释为什么 `gqa_8kv` 在 toy gate 中比 `mqa_1kv` 和 `mla_latent` 更适合作为默认上线方案。

## 4.18 本章总结

本章系统比较了 MHA、MQA、GQA、MLA 的演进与 trade-off。

核心结论：

1. MHA 表达能力完整，但每个 head 都有 K/V，推理 KV cache 昂贵。
2. MQA 让所有 query heads 共享一个 K/V head，显著降低 cache 和内存带宽，但可能损失质量。
3. GQA 让一组 query heads 共享一个 K/V head，是 MHA 和 MQA 之间的实用折中。
4. MLA 进一步把 K/V 信息压缩到 latent cache，是面向长上下文和高效推理的更激进设计。
5. KV cache 是 decoder-only LLM serving 的关键瓶颈之一，影响显存、batch、吞吐和延迟。
6. FlashAttention 优化 kernel，PagedAttention 优化 cache 管理，MQA/GQA/MLA 优化模型结构，三者层面不同但可以组合。
7. 现代注意力设计已经不只是模型表达问题，而是模型架构、推理系统和硬件效率共同优化的问题。

下一章会进入 FFN、SwiGLU、Gated MLP 与 MoE FFN，重点讲 Transformer 中另一个被低估的主力模块：前馈网络。
