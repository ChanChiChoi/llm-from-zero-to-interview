# 第四章：MHA、MQA、GQA、MLA 的演进与 Trade-off

## 4.1 为什么注意力结构要改变 K/V 的保存方式

第二章解释了一个 attention head 如何用 Query、Key 和 Value 读取信息；第三章补充了多头注意力中各 head 如何从完整输入学习自己的投影。本章继续追问：如果每个 query head 都保存一套 K/V，生成长文本时会发生什么？

自回归模型每生成一个 token，都会让新 Query 读取前文。推理系统通常把过去各层的 Key 和 Value 缓存起来，供后续 token 重用，这份运行时状态叫 **KV cache**。历史越长、同时服务的请求越多，缓存就越大；decode 每一步还需要读取这些历史 K/V。因此，注意力架构改动 K/V 时，通常同时影响缓存容量和读取流量。

本章沿一条主线比较四种结构：MHA 为每个 query head 保留独立 K/V；MQA 让所有 query heads 共用一套 K/V；GQA 在两者之间分组共享；MLA 则把每个 token 的 K/V 相关信息联合压缩成较低维的 latent，并额外处理位置信息。第十一章再完整计算不同请求长度、并发和数据类型下的缓存容量。

## 4.2 MHA：每个 Query Head 都有自己的 K/V

标准多头注意力（Multi-Head Attention，MHA）有相同数量的 query heads 和 K/V heads。记 query head 数为 $H_q$，K/V head 数为 $H_{\mathrm{kv}}$；MHA 满足 $H_{\mathrm{kv}}=H_q$。每个 head 使用自己的查询、键和值投影，这是标准 Transformer 的多头结构。

如果每个 Key 和 Value head 的宽度都为 $d_h$，那么一个 token 在一层中需要缓存的元素数为：

```math
E_{\mathrm{KV}}=2H_{\mathrm{kv}}d_h
```

这里的 $2$ 表示一份 Key 和一份 Value。若 Key 与 Value 的宽度不同，应把 $2d_h$ 换成 $d_k+d_v$。只有在每头宽度 $d_h$ 固定时，增加 K/V heads 才会增加缓存元素；如果固定的是总 K/V 投影宽度 $H_{\mathrm{kv}}d_h$，仅把它切成更多 heads 并不会增加总缓存量。MQA/GQA 节省缓存，是因为它们在 query head 数量不变时减少了独立 K/V heads。

## 4.3 MQA：保留多个 Query，共享一套 K/V

多查询注意力（Multi-Query Attention，MQA）保留 $H_q$ 个 query heads，但只保留一个 K/V head，即 $H_{\mathrm{kv}}=1$。它们读取同一组历史 Key 和 Value，不过每个 query head 仍有自己的 Query。

例如，四个 query heads 都可以使用同一组 $K_0,V_0$。由于四个 Query 不同，它们与 $K_0$ 计算出的注意力权重也可以不同，最终从共享的 Value 序列中读出不同内容。共享 K/V 不等于多个 head 被合并成一个 query head。

在 head 宽度相同的条件下，MQA 每个 token 的 K/V 缓存约为对应 MHA 的 $1/H_q$。Shazeer 提出 MQA 时，关注的是增量解码反复读取历史 K/V 所造成的内存带宽开销；论文在当时的模型、任务和 TPU 设置中报告了更快的 decode，也报告了较小的质量变化。这是该论文实验的结论，不是所有模型都能获得相同速度或质量结果的保证（[Shazeer，2019](https://arxiv.org/html/1911.02150)）。

## 4.4 GQA：用 K/V 组数调节共享程度

分组查询注意力（Grouped-Query Attention，GQA）把 query heads 分成若干组，每组共享一对 K/V heads。设有 $H_q$ 个 query heads 和 $H_{\mathrm{kv}}$ 个 K/V heads；当 $H_q$ 能被 $H_{\mathrm{kv}}$ 整除时，每组包含 $g=H_q/H_{\mathrm{kv}}$ 个 query heads。

假设 $H_q=4$ 且 $H_{\mathrm{kv}}=2$。那么 $Q_0,Q_1$ 共用 $K_0,V_0$， $Q_2,Q_3$ 共用 $K_1,V_1$。同一组内的两个 Query 仍分别计算自己的注意力权重，所以共享 K/V 会减少可学习的 K/V 表示数量，但不会让组内的查询行为完全相同。

把 K/V head 数从少到多排列，MQA 是 $H_{\mathrm{kv}}=1$ 的端点，MHA 是 $H_{\mathrm{kv}}=H_q$ 的端点，中间取值就是 GQA。若 head 宽度和其他条件相同，GQA 的每 token 缓存量相对 MHA 约为 $H_{\mathrm{kv}}/H_q$。例如， $H_q=32$ 且 $H_{\mathrm{kv}}=8$ 时，K/V 缓存约为对应 MHA 的四分之一。

GQA 也会改变训练得到的参数。Ainslie 等人展示了一种把已有 MHA checkpoint 转成 MQA/GQA 的做法：先将每组原 K/V 投影做平均池化，再继续预训练使模型适应新结构。论文报告其 T5 设置使用约 5% 的原始预训练计算量进行 uptraining；这个比例绑定论文中的 checkpoint、训练配方和 encoder-decoder 实验，不是通用转换成本（[Ainslie 等，2023](https://aclanthology.org/2023.emnlp-main.298/)）。

## 4.5 MLA：压缩 K/V 表示，并单独保存位置部分

多头潜变量注意力（Multi-head Latent Attention，MLA）与 GQA 的主要差别不在“再选一个更小的 K/V head 数”，而在于缓存什么表示。DeepSeek-V2 的 MLA 把当前 token 的隐藏表示先投影到一个较低维的 K/V latent，再由这个 latent 生成各 head 所需的内容相关 Key 和 Value：

```math
c_t^{KV}=W^{DKV}h_t
```

```math
k_t^C=W^{UK}c_t^{KV},\qquad v_t^C=W^{UV}c_t^{KV}
```

其中， $c_t^{KV}$ 是同一 token 的压缩表示，$k_t^C$ 和 $v_t^C$ 分别表示由它得到的内容 Key 和 Value。推理时可以缓存 $c_t^{KV}$，而不必为每个 head 分别保存完整的 K/V。DeepSeek-V2 还指出，推理实现可将部分上投影吸收到 Query 和输出投影计算中，避免显式展开全部 K/V 张量。

MLA 还需要保留位置信息。普通 RoPE 会让 Key 随 token 位置旋转；若把所有 Key 都压进同一个与位置无关的 latent，就不能直接沿用原来的位置计算。DeepSeek-V2 因此把内容部分与 RoPE 部分拆开：每个 head 使用内容 Key 与 Value，同时各 token 另有一份位置相关 Key $k_t^R$，并对它应用 RoPE：

```math
k_t^R=\mathrm{RoPE}(W^{KR}h_t)
```

Query 也有对应的内容与位置部分。把当前位置的 Query 记为 $q_t$，历史位置 $j$ 的 Key 记为 $k_j$，两者都由内容部分和 RoPE 部分拼接而成，因此注意力点积同时包含内容匹配和位置匹配：

```math
q_t^{\top}k_j=(q_t^C)^{\top}k_j^C+(q_t^R)^{\top}k_j^R
```

推理缓存因此包含 K/V latent 和位置 Key。按 DeepSeek-V2 论文的记号，每 token、每层需保存约 $d_c+d_h^R$ 个元素，其中 latent 宽度记为 $d_c$，位置 Key 宽度记为 $d_h^R$。这个记账方式说明 MLA 与 MQA/GQA 不是同一种“减少 head 数”的操作（[DeepSeek-V2，§2.1](https://arxiv.org/html/2405.04434v5)）。

## 4.6 按每个 Token 的 K/V 状态比较

在 K/V 宽度相同且不计数据类型字节数时，几种结构的每 token、每层缓存元素数可写成：

| 结构 | 缓存的历史表示 | 每 token、每层缓存元素数 |
| --- | --- | ---: |
| MHA | 每个 query head 一组 Key 和 Value | $2H_qd_h$ |
| MQA | 所有 query heads 共用一组 Key 和 Value | $2d_h$ |
| GQA | 每组 query heads 共用一组 Key 和 Value | $2H_{\mathrm{kv}}d_h$ |
| MLA（DeepSeek-V2 形式） | K/V latent 加一份位置 Key | $d_c+d_h^R$ |

前三种公式假设每个 K/V head 的 Key、Value 宽度相等，且各结构使用相同的 head 宽度。只有在这些条件对齐时，MQA/GQA 相对 MHA 的缓存比例才是 $H_{\mathrm{kv}}/H_q$。这个比例描述的是 K/V 缓存元素数，不是全模型参数量或端到端延迟的节省比例。MLA 没有可直接代入的 K/V head 数；应比较 latent 宽度和位置 Key 宽度之和与显式 K/V 宽度，而不是把它当作“一个 KV head”。

若所有层结构一致、每个请求的缓存长度都是 $T$、同时有 $B$ 条请求、元素占 $b$ 字节，则可以将每 token、每层元素数 $E_{\mathrm{KV}}$ 扩展为：

```math
M_{\mathrm{KV}}=B\times L\times T\times b\times E_{\mathrm{KV}}
```

这个简式只说明架构改变 $E_{\mathrm{KV}}$ 的方式。真实服务还要考虑请求长度分布、滑动窗口、层间差异、分页与量化；这些容量和分配问题由[第十一章](./11-kv-cache长上下文推理和显存增长问题.md)完整展开。

## 4.7 论文结果说明了什么，又没有说明什么

不同实验对 MQA、GQA、MLA 的质量与速度取舍并不完全相同。它们使用的模型、训练方式、任务和硬件不同，因此应把结果连同条件一起阅读。

| 来源与实验 | 作者报告的结果 | 适用边界 |
| --- | --- | --- |
| [Shazeer，2019](https://arxiv.org/abs/1911.02150)：翻译与语言建模实验 | MQA 在论文的 TPU 增量解码设置中更快，质量变化较小 | 旧模型、任务、固定形状与 TPU 条件；不是现代 GPU serving 的速度承诺 |
| [Ainslie 等，2023](https://aclanthology.org/2023.emnlp-main.298/)：T5 checkpoint 转换与 uptraining | GQA 在所测任务中接近 MHA 的质量和 MQA 的解码速度 | 论文明确只评估 encoder-decoder 模型；转换成本取决于具体 recipe |
| [DeepSeek-V2，附录 D.1](https://arxiv.org/html/2405.04434v5)：约 7B 的 dense MHA/GQA/MQA 对比 | 四个 hard benchmarks 上 MHA 高于 GQA-8 和 MQA | 为对齐参数量调整了层数；结果限于论文模型和四项任务 |
| [DeepSeek-V2，附录 D.2](https://arxiv.org/html/2405.04434v5)：两种规模的 MoE MLA/MHA 对比 | 作者报告 MLA 的缓存分别约为对应 MHA 变体的 14% 和 4%，并在所列 benchmark 上分数更高 | 激活参数量有差异，仍是特定 MoE、token 预算与 benchmark；不能把整模型吞吐提升归因于 MLA 单一机制 |
| [MHA2MLA，ACL 2025](https://aclanthology.org/2025.acl-long.1597/)：将已有模型适配到 MLA | 通过 partial-RoPE 与联合 SVD 等步骤微调；论文报告特定 Llama2-7B 实验缓存减少 92.19%、LongBench 约下降 1% | 不是零样本替换；数据量、质量和缓存比例绑定论文方法与模型 |

因此，不能从这些论文推出“GQA 一定接近 MHA”或“MLA 总是更好”。可以得到的稳定结论是：减少独立 K/V 状态或压缩其表示，会减少按该公式计算的缓存元素；任务质量与实际解码速度仍需结合训练方法和目标实现判断。

## 4.8 架构、Kernel 与 Cache 管理不是同一种优化

| 方法 | 所在层面 | 主要改变什么 |
| --- | --- | --- |
| MQA、GQA、MLA | 模型的 attention 架构 | 改变 K/V head 的共享方式，或改变历史 K/V 的缓存表示 |
| FlashAttention | attention kernel | 通过分块和在线 softmax 减少计算 attention 时的 HBM 读写；dense 版本仍计算完整 attention，不会改变历史 K/V 的总元素数（[FlashAttention](https://arxiv.org/html/2205.14135v2)） |
| PagedAttention | serving cache 管理 | 通过 block/page 管理变长请求的 KV 内存；不改变模型的 attention 公式（[PagedAttention](https://arxiv.org/abs/2309.06180)） |
| MoE | Transformer 的 FFN 路径 | 通过路由改变每个 token 使用哪些专家；可以与 MQA、GQA 或 MLA 组合，但它不是一种 attention cache 格式，详见[第五章](./05-ffn-swiglu-gated-mlp与moe-ffn.md) |

同一个系统可以同时使用这些方法，因为它们解决的问题不同。[第十章](./10-o-n2-attention的计算与显存瓶颈.md)展开 attention 计算与 FlashAttention；第十一章讲 cache 管理和推理服务成本。线性注意力、SSM、KDA 与门控 MLA 则改变或组合了历史状态表示，分别见[第二十四章](./24-Linear-Attention基本思想与核技巧.md)、[第二十九章](./29-SSM数学直觉.md)、[第四十二章](./42-替代架构横向对比.md)及[第六十三章](./63-gated-mla门控潜变量注意力.md)。

## 4.9 比较架构时需要核对的条件

读论文或模型配置时，可按以下顺序判断 K/V 结构带来的实际变化：

1. **先看表示。** 记录 $H_q$、 $H_{\mathrm{kv}}$ 和每头宽度；如果是 MLA，则记录 latent 宽度和单独的位置 Key 宽度。确认模型的配置与技术报告描述的是同一版本。
2. **再看训练来源。** 模型是原生按该结构训练，还是从其他结构转换后 uptraining/微调？转换后的质量不能仅由目标结构名称推断。
3. **区分缓存节省与服务速度。** 结构公式能估算缓存元素，但实际 decode 速度还取决于 kernel、并行布局、请求长度与 batch。若实现显式复制共享 K/V，架构上的共享也不必然等于目标硬件上的同等流量节省。
4. **在同一任务上复核质量。** 对齐模型规模、训练 token、上下文长度、数据、评测和解码设置；不能把不同论文中的单项分数拼成统一排名。

缓存更小只是结构成本的一部分。实际选择还要同时看质量要求、训练或转换代价、目标硬件实现和服务负载；本章不规定适用于所有模型的 MHA、GQA 或 MLA 默认值。

## 4.10 面试问题与练习

### 面试问题

**问题一：MHA、MQA 和 GQA 的核心区别是什么？**

MHA 每个 query head 有独立 K/V；MQA 让所有 query heads 共用一组 K/V；GQA 则让若干 query heads 共用一组 K/V。Query heads 数量可以保持不变，因此组内不同 Query 仍能得到不同的注意力权重。

**问题二：MLA 与 GQA 的缓存思路有什么不同？**

GQA 仍缓存显式 K/V，但减少 K/V heads 数；MLA 把 K/V 相关信息联合映射到低维 latent，并在 DeepSeek-V2 设计中另缓存位置相关的 RoPE Key。因此不能用一个“KV head 数”完整描述 MLA 的缓存大小。

**问题三：为什么不能说 GQA 总是质量接近 MHA、MLA 总是比 MHA 好？**

质量结果依赖模型规模、训练 recipe、微调/转换方式、任务和评测。GQA 原论文的 uptraining 证据来自 T5 encoder-decoder，而 DeepSeek-V2 的 7B dense 消融在特定 hard benchmarks 上由 MHA 得分更高。每个结果都应按原论文条件解释。

**问题四：FlashAttention 和 PagedAttention 是否能替代 GQA？**

不能。FlashAttention 改进 attention kernel 的 IO；PagedAttention 管理推理时的 KV 内存；GQA 则改变模型中 K/V heads 的共享结构。它们可以组合使用。

### 练习

1. 假设 $H_q=8$ 且 $H_{\mathrm{kv}}=2$。写出每组 query heads 共用的 K/V，并在 K/V head 宽度不变时推导它相对 MHA 的缓存比例。
2. 用一两段话解释为什么两个共用同一 K/V head 的 Query 仍可关注不同历史位置。
3. 对照 MQA、GQA、MLA 的原始论文，分别写出它们改变的是共享 head 数、缓存表示，还是 checkpoint 训练方式，并标注各结果的模型和任务范围。
4. 把 FlashAttention、PagedAttention、GQA、MLA、MoE 放到“kernel / serving cache 管理 / attention 架构 / FFN 路由”四个层面中，并解释哪些方法可以组合。

## 4.11 本章总结

MHA 为每个 query head 保留独立 K/V。MQA 让所有 query heads 共享一组 K/V，GQA 再用组数调节共享程度；这两种结构仍然保留不同的 Query。MLA 改为缓存低维 K/V latent，并额外保存位置相关的 RoPE Key。

在宽度、层数、长度和数据类型等条件对齐时，减少 K/V heads 会按比例降低显式 K/V 缓存元素数；MLA 则按 latent 与位置 Key 的宽度估算。缓存容量公式本身不能推出任务质量或端到端 decode 加速。论文结果、训练方式、kernel 和服务设置都要分别核对。FlashAttention、PagedAttention 与 MoE 作用于不同层面，也可以和这些 attention 架构组合。
