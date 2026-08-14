# 第二十七章：Longformer、BigBird、Reformer 等长序列路线

## 27.1 为什么要回看这些“早期”方案

现在的长上下文模型常用位置扩展、KV 压缩、混合 attention 或更强的训练，但 Longformer、BigBird、Reformer 仍然提供了理解长序列设计的基础语言。它们分别展示了滑窗+全局、局部+随机+全局和 LSH 近似等思路。

学习这些方法不是为了把旧模型直接部署到今天，而是为了理解一个持续存在的问题：当序列长度增长时，哪些远程连接必须保留，哪些可以通过结构先验、随机路径或近似计算替代。

## 27.2 Longformer：局部窗口加任务相关全局

Longformer 对每个位置使用滑动窗口，少量标记位置使用 global attention。局部 attention 的 score 数约为 `O(nw)`；若 global token 数为 `g`，额外全局连接约为 `O(ng)`：

~~~math
C_{\mathrm{Longformer}}\approx O(nw+ng)
~~~

局部窗口适合连续文本，global token 让问题 token、标题或分类标记访问远处内容。global 的位置通常由任务决定，因此长文档问答要考虑 query token 是否正确标注。

它的关键不是“窗口替代全局”，而是把全局预算用在有任务意义的位置。若 global 标记选错，系统可能仍能处理局部语言，却无法完成文档级问答。

## 27.3 BigBird：把图连通性显式放进设计

BigBird 结合 local、random 和 global attention。随机边的作用不是提供语义检索，而是降低图的直径、增加不同区域之间的连接机会。global token 负责全局汇聚，local edge 保留近邻结构。

一个粗略的连接数可以写为：

~~~math
E\approx n(w+r+g)
~~~

其中 `w` 是局部边数，`r` 是随机边数，`g` 是全局边预算。随机 pattern 的可复现性、seed、mask 生成和 batch 对齐很重要；换一个随机图可能改变长程结果。

## 27.4 Reformer：LSH attention 的近似检索

Reformer 使用 locality-sensitive hashing，把相似的 query/key 映射到相同或相邻 bucket，只在 bucket 内计算 attention，目标是减少全量两两比较。LSH 的直觉是：相似向量更可能碰撞，不相似向量更少碰撞。

设哈希函数为 `h(x)`，bucket 为：

~~~math
b=h(x)\in\{1,\ldots,B\}
~~~

模型只在同 bucket 或相邻 bucket 中进行局部 attention。哈希冲突和漏召回是主要风险；多轮 hash 可以提高召回，但会增加计算和随机性。

## 27.5 LSH 的最小教学代码

~~~python
import torch


def random_lsh(x, num_buckets, seed=0):
    # x: [tokens, dim]；输出每个 token 的 bucket id
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(x.size(-1), num_buckets, generator=generator)
    bits = (x @ projection > 0).to(torch.int64)
    powers = 2 ** torch.arange(num_buckets)
    return (bits * powers).sum(-1)


x = torch.randn(12, 8)
print(random_lsh(x, num_buckets=4))
~~~

这不是完整 Reformer 的 bucket 排序、chunk、相邻 bucket 和 reversible layer 实现，只展示输入依赖的分桶。真实实现还要处理同一 bucket 的容量、排序、padding、因果方向和多轮 hash。

## 27.6 三条路线的能力差异

| 方法 | 主要连接 | 远程信息路径 | 主要风险 |
|---|---|---|---|
| Longformer | local + task global | 通过 global token | global 位置选择错误 |
| BigBird | local + random + global | 随机缩短图路径 | 随机漏边和实现开销 |
| Reformer | hash bucket 内局部 | 相似性碰撞 | hash 漏召回、排序成本 |

它们共同减少计算，但没有消除“重要的远程 token 如何被找到”的问题。静态 global 依赖任务标注，随机连接依赖图覆盖，LSH 依赖相似度和 hash 稳定性。

## 27.7 训练和推理的复杂度

这些方法通常需要专门的 mask、排序或块布局。训练时可以并行处理块，但要为不同 bucket/block 对齐；推理时请求长度动态变化，LSH 重分桶和局部 cache 管理可能增加调度复杂度。

当长度不长、GPU dense attention kernel 很成熟时，稀疏理论收益可能不明显；当文档足够长、pattern 规则且任务确实局部时，收益才更可能出现。benchmark 必须在相同硬件、batch、padding 和 kernel 下比较。

## 27.8 长文档任务中的证据路径

构造一个文档问题时，先标注证据的位置和类型：邻近证据、跨章节证据、需要冲突消解的版本证据。然后检查该证据在 mask 图上是否有路径到输出位置，以及路径是否足够短。

例如 Longformer 的 query global token 如果没有被设置为 global，原本应该跨文档读取的证据只剩 local path；BigBird 的随机边可能偶尔连上，却不保证每次稳定；Reformer 的两个相关实体若 hash 到不同 bucket，直接交互被切断。这个分析比只看最终错误更能指导修复。

## 27.9 与现代长上下文方案的关系

位置插值和 RoPE scaling 主要改变坐标表示；Longformer/BigBird/Reformer 主要改变 attention 连接图；FlashAttention 主要改变 kernel 访存；MLA/GQA 主要压缩 KV 表示。它们可以组合，但优化层次不同。

例如，一个模型可以使用 RoPE 扩展位置、局部/global mask 降低计算、FlashAttention kernel 处理局部块、PagedAttention 管理 KV。组合后系统复杂度上升，必须确保每层位置、mask、cache 和 scheduler 契约一致。

## 27.10 失败模式

Longformer 常见问题是 global 标记漏设、global attention 方向错误和局部窗口边界错位；BigBird 关注随机 seed、mask 可复现性、global block 负载和远程证据波动；Reformer 关注 hash bucket 不均、相似 token 漏召回、排序/反排序 bug 和多 hash 的额外成本。

排查可做三步：保存实际 mask 或 bucket assignment；在短序列上与 dense reference 比较 logits；再按证据距离、hash seed、窗口和长度做分桶。若不同 seed 的结果波动大，不能把单次结果写成架构结论。

## 27.11 机制与边界：表达性和工程性必须分别证明

论文可能给出稀疏图的表达能力或近似保证，但条件往往涉及足够的 global token、随机边和层数。真实语言任务还包括位置编码、训练分布、tokenization 和 optimization。工程上要证明的则是 kernel 可用、内存不碎、batch 不被索引拖慢、质量退化在验收条件内。

这些路线没有完全成为通用 LLM 默认架构，部分原因正是 full attention kernel、GQA、FlashAttention、长上下文训练和 KV 管理共同进步后，dense attention 的实际竞争力仍然很强。

## 27.12 面试追问、误区与练习

**问：BigBird 的 random attention 是为了语义相似检索吗？**

标准回答：主要作用是增加稀疏图的连通性、缩短远程路径；它不是一个语义检索器，不保证找到最相关证据。

**问：Reformer 的 LSH attention 为什么有漏召回风险？**

标准回答：相似 token 只是在概率上更容易碰撞，有限 hash、随机投影和 bucket 容量可能把本应交互的 token 分开；增加 hash 或邻近 bucket 能降低风险，但增加成本。

常见误区包括把三者都叫 sliding window、把随机边当全局直连、把理论线性复杂度当真实吞吐，以及忘记任务相关 global token。

练习：在同一长文档上实现 local/global、local/random/global 和 LSH 三种 mask，固定预算比较 evidence recall、hash seed 方差、TTFT 和峰值显存。

## 27.13 三种路线的训练前提

Longformer 的 global token 需要任务知道哪些位置值得全局可见；BigBird 的随机、局部和全局边要保持足够连通；Reformer 的 LSH 需要哈希碰撞和排序在训练、推理中一致。它们不是只改一行 mask。

长文档任务若没有明确 global token，Longformer 可能漏掉证据；LSH 将相似 query 分到不同 bucket 时，Reformer 可能漏检；随机 pattern 虽有理论连通性，却可能不适合精确引用。

## 27.14 从理论稀疏到端到端收益

比较三种方法时，记录实际 score block、排序/哈希时间、global token 数、额外 metadata、prefill、decode 和引用质量。对于不同长度，画出 memory 与任务成功曲线。

如果短文本没有收益、长文本才收益，部署还要考虑请求分布和 fallback；如果平均吞吐增加但 p99 变差，可能是排序、动态 bucket 或 global token 造成批次分裂。

## 27.15 三种方法的失败归因

Longformer 的失败常来自 global token 选错或窗口没有覆盖证据；BigBird 可能受随机边、全局节点容量和 mask 实现影响；Reformer 可能发生 hash 漏召回、bucket 排序和局部邻接不足。它们都可能表现为“长文档回答错”，但修复路径完全不同。

因此评估记录要保留证据位置、global token、随机 seed、hash bucket、候选 block 和最终引用。只保存最终答案，无法判断是图没有连通，还是模型没有使用已连通的证据。

## 27.16 稀疏连接与上下文预算

设局部窗口宽度为 w，全局节点数为 g，随机边数为 r，序列长度为 T，教学式边数可近似写成：

~~~math
E_{\mathrm{sparse}}
\approx T w+Tg+Tr
~~~

实际还要加 global token 的全连接、bucket 排序、padding 和层数。增大 g 可能改善远程信息，但也可能重新引入较高计算；增大 r 可能降低漏连，却增加随机和 metadata 成本。

## 27.17 历史方法对今天系统的影响

Longformer、BigBird 和 Reformer 的价值不只在最终是否成为主流，它们把长上下文问题拆成了窗口、全局节点、随机连接和近似哈希等可复用设计。现代 sliding window、block sparse、retrieval 和 hybrid attention 仍然可以看到这些思想的延续。

读者应把它们当作方法谱系：每条路线都明确牺牲了哪一种任意访问能力，又通过什么结构补偿。这样面对新稀疏模式时，可以比较其信息图和系统契约，而不是只记一个模型名字。

## 27.18 三条路线留下的工程教训

Longformer 提醒我们局部窗口需要任务相关的 global token；BigBird 提醒我们连接图需要可达性和随机/全局补偿；Reformer 提醒我们近似检索需要处理哈希碰撞、排序和稳定性。这些方法即使没有成为今天所有 LLM 的默认结构，仍然解释了现代长上下文设计为什么常见混合连接、分块和外部检索。

## 27.19 历史方法与现代长上下文的公平迁移

不能把早期论文的线性或近线性复杂度直接与现代 dense Transformer benchmark 比较。需要固定参数规模、训练 token、位置长度、数据混合、硬件、kernel、batch 和质量任务；还要说明早期方法是否经过同等长度训练，以及它的 tokenizer 和输出协议是否一致。

如果只复现一个短文档任务，可能得到“稀疏方法已经解决长上下文”的新闻式结论；加入中间证据、多个实体、长输出、引用和失败恢复后，能力差距可能完全不同。历史路线应作为实验假设来源，而不是未经条件转换的排名。

## 27.20 用路径分析决定是否保留 global 连接

对每个任务，记录证据 token 到 query 的图距离、经过的层数、每层是否保留信息以及最终引用是否正确。若图距离很短但引用仍错，问题可能是表示或训练；若图不可达，增加模型规模无法修复结构性缺口。这个分析能指导 global token、窗口和 RAG 的选择。

## 27.21 三条路线的可复现实验

为了避免把历史方法写成新闻摘要，可以用同一组任务重放三类信息图。第一类是局部加 global 的窗口模型，第二类是局部、随机和 global 的混合图，第三类是基于哈希的近似邻居图。固定 tokenizer、参数规模、训练 token、context length 和输出协议，只改变连接机制。

任务至少包括：近邻语法、单个远距数字、两个同名实体消歧、三段证据合并、证据不存在时拒答，以及跨 chunk 恢复。记录每个样本的可达性、最短路径、global/hash 元数据、实际非零 block、kernel fallback、exact recall 和引用支持率。若模型已经能到达证据但仍回答错误，继续增加边不一定有用；若证据根本不可达，应该修改 pattern 或加入 retrieval。

对 Reformer 类 hash 路径，随机 seed 和 bucket 排序必须进入 artifact。对 BigBird 类随机连接，重复实验才能估计漏连概率；对 Longformer 类 global token，必须说明 token 如何选、是否可学习以及容量是否成为瓶颈。这样读者才能理解三条路线各自牺牲和保留了什么。

## 27.22 证据位置的受控实验

要判断稀疏连接是否真的帮助长文档任务，不能只放一个固定位置的 needle。至少应把证据放在开头、中间、结尾和多个相互冲突的位置，并改变 global token、random seed、hash seed、窗口宽度和文档噪声。

每个样本记录证据是否在图上可达、最短路径、实际非零 block、引用是否支持和最终答案是否正确。若 evidence 不可达，属于 mask 结构失败；若可达但答案错，属于训练、表示或解码失败；两者的修复方案不同。

## 27.23 worked example：全局节点不是免费的

设序列长度为 `T=64K`，局部窗口为 `w=256`。把 global token 从 `g=16` 增加到 `g=512`，教学近似的边数从 `T(w+g)` 增长约 `2.1` 倍。远程召回可能改善，但 global token 的 prefill 计算、显存、mask 构造和 batch 形状也会增加。

如果 16 个 global token 已经覆盖 query、标题和证据摘要，再增加到 512 只提高普通摘要分数而没有提高数字 exact recall，就不应继续扩大预算。这个例子提醒读者：全局连接是信息预算和资源预算的共同决策。

## 27.24 从历史方案迁移到现代系统

Longformer、BigBird 和 Reformer 的设计可以迁移为现代系统的实验假设，而不是直接迁移为生产组件。滑窗思想可以与 retrieval 或 hybrid attention 组合；随机连接可以用于研究图可达性；LSH 可以作为近似候选生成器，但最终证据仍应回读原文。

迁移时固定 tokenizer、模型规模、训练长度、硬件、kernel、batch、padding 和输出协议，分别跑 dense baseline、稀疏方案和带外部检索的方案。只有质量、p99、峰值显存、回退率和单位成功成本都进入同一报告，才可以判断旧结构在今天是否仍有工程价值。

## 27.25 三种方法的证据边界

Longformer 见 https://arxiv.org/abs/2004.05150，BigBird 见 https://arxiv.org/abs/2007.14062，Reformer 见 https://arxiv.org/abs/2001.04451。本文中的复杂度为教学近似，实际实现需要查对应仓库和硬件 kernel。

这三条路线把长序列问题分别转化为任务 global、稀疏图连通和近似相似检索。它们的共同教训是：减少连接后，必须明确谁负责把远处重要信息送到当前 query。

## 27.26 三种方法的最小数学对照

Longformer 的局部窗口可以把每个 query 的候选集合限制为邻域 `W(i)`，再加入少量 global 集合 `G`：

```math
\mathcal{V}_{\mathrm{long}}(i)=W(i)\cup G.
```

BigBird 在此基础上加入随机或块级连接 `R`，其集合可以写成：

```math
\mathcal{V}_{\mathrm{big}}(i)=W(i)\cup G\cup R(i).
```

Reformer 的 LSH attention 则先用哈希把相似候选放到同一 bucket，再在 bucket 内做局部交互。它不是一个固定的语义邻域，而是一个由哈希函数、bucket 数和排序决定的近似候选集。三者都减少直接交互，但“谁有机会看到谁”的原因不同：Longformer 依赖任务设计的 global，BigBird 依赖混合图的连通，Reformer 依赖相似项被稳定哈希到一起。

这一区分很重要。若两个版本的实体词面不同但语义相近，Reformer 的 hash collision 可能有帮助；若关键证据与问题相似度低，LSH 未必召回；若问题需要固定的章节标题或特殊元数据，Longformer 的显式 global 可能更可控。

## 27.27 Reformer 的哈希碰撞与稳定性

LSH 路径的风险不是抽象的“近似有误差”，而是具体的候选遗漏和 bucket 边界漂移。改变 hash seed、投影、bucket 数、序列长度或 padding，可能让两个本应相互作用的 token 被分到不同 bucket。多轮推理若每轮重新采样 hash，状态也可能无法稳定复现。

评测可以固定 Q/K 表示，重复多个 hash seed，统计关键证据同 bucket 的概率：

```math
p_{\mathrm{co\mbucket}}
=\frac{1}{S}\sum_{s=1}^{S}
\mathbf{1}\left[h_s(i)=h_s(j)\right].
```

这不是最终注意力正确率，但能把“可能漏边”变成可观测信号。对高风险引用任务，低 `p_co-bucket` 的样本应走 dense 或外部 retrieval 回退；对平滑分类任务，可以接受一定候选误差以换取更低成本。

## 27.28 Global 选择和数据标注

Longformer 的 global token 不是自然存在的特殊词，它需要规则、任务标注或模型设计决定。把 `[CLS]`、问题 token、章节标题、实体 token 全部设为 global，会增加计算和竞争；一个看似“更全”的 global 集合也可能让注意力被高频但无关的元信息吸走。

可用三种策略做对照：固定位置 global、任务字段 global、可学习/路由 global。任务字段策略对抽取和问答可能更直接，但需要可靠的解析器和数据标注；可学习策略减少手工规则，却可能在分布外文档上失效。实验应记录 global 选择准确率、global 数量、引用支持率和对主题摘要的影响，而不能只报告整体 benchmark 分数。

## 27.29 从论文结构到现代长上下文系统

今天的长上下文系统往往不会原样复用某一个历史结构，而是组合几种思想：局部/全局 attention 负责可控信息路径，RAG 负责外部候选和权限过滤，KV/状态缓存负责服务成本，长上下文训练负责让模型学会使用这些路径。把四者放在一起，才是从论文结构迁移到产品的完整问题。

一个现实的选择例子是企业文档问答。文档标题、问题和证据摘要可以成为少量 global 入口；正文先经过权限过滤和 hybrid retrieval，减少无关 token；对最终引用，保留可回读的原文片段，而不是只依赖 global state；如果稀疏 kernel 不支持某种长度，则在 admission 阶段选择 dense 或拒绝，而不是上线后静默 fallback。

因此 Longformer、BigBird 和 Reformer 的价值，主要在于提供可检验的结构假设：哪类边对任务有用、随机连接是否改善可达性、近似检索的漏召回如何衡量。真正的工程结论必须来自 matched baseline、样本级错误归因、kernel 实测和回退演练，而不是来自论文标题或一个线性复杂度符号。
