# 第十九章：Sliding Window Attention、Global Attention 与稀疏注意力

## 19.1 为什么要限制注意力连接

标准 self-attention 在长度 `n` 上构造近似 `n×n` 的关系。长文档、代码、音频和视频 token 会让计算、显存和访存迅速增长。稀疏注意力的基本思路不是让模型少看几句话，而是预先规定或学习一部分允许的连接，让每个 query 只访问局部窗口、全局 token、随机位置或块。

减少连接会改变信息图。它可能降低成本，也可能让远处证据无法在有限层数内到达。稀疏 pattern 因此必须同时考虑复杂度、图的连通性、任务结构、kernel 是否高效以及训练/推理 mask 是否一致。

## 19.2 小白直觉：道路网络而不是“少算一点”

把 token 看作城市，把 attention 连接看作道路。full attention 是每两个城市都有直达航班；滑窗是在相邻城市之间修高速路；global token 是中央枢纽；随机连接是应急道路。道路减少后，城市之间仍可能连通，但远处城市的消息要经过更多中转，且某些路径可能根本不存在。

因此有两个不同指标：单层边数和多层有效路径长度。只报告 `O(nw)` 不能证明全局依赖能被稳定学习。

## 19.3 局部窗口的数学形式

给定窗口半径 `w`，位置 `i` 允许读取：

~~~math
\mathcal{N}(i)=\{j\mid \max(0,i-w+1)\le j\le i\}
~~~

在 causal sliding window 中，score mask 为：

~~~math
M_{ij}=\begin{cases}
0,&j\in\mathcal{N}(i)\\
-\infty,&\text{otherwise}
\end{cases}
~~~

每个 query 约看 `w` 个 key，理论 score 数从 `O(n^2)` 降为 `O(nw)`。若 `w` 固定，随长度近似线性；若窗口随长度增长，复杂度也会变化。

## 19.4 Global attention 和混合连接

Longformer 将局部窗口和任务相关的 global attention 结合：某些 token 可以与全序列交互。global token 可以是 `[CLS]`、问题 token、文档标题或被标记的实体。BigBird 进一步组合 local、random 和 global pattern，以提高图连通性。

一个粗略的边数估算是：

~~~math
E\approx n w + 2ng + nr
~~~

其中 `g` 是 global token 数，`r` 是随机/额外连接数。真正的显存和运行时间还取决于 block layout、padding、稀疏 kernel 和 global token 的实现。

## 19.5 最小 mask 实现

~~~python
import torch


def causal_window_mask(seq_len, window, global_positions=()):
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    allowed = (j <= i) & (j >= i - window + 1)
    for pos in global_positions:
        allowed[pos, :pos + 1] = True
        allowed[:, pos] |= (torch.arange(seq_len) >= pos)
    return allowed


mask = causal_window_mask(8, 3, global_positions=(0,))
print(mask.int())
~~~

这段代码的 global 规则是教学化的。生产 mask 要明确 global query 是否能看未来、普通 query 是否能读 global token、prefix 部分是否双向，以及 batch 中不同样本的 global position 如何编码。一个 off-by-one 就可能导致未来信息泄漏或远处证据断路。

## 19.6 信息传播的层数问题

在纯滑窗中，两个相距 `d` 的位置至少需要大约 `ceil(d/w)` 层的传播路径，实际还受因果方向和内容路由影响。窗口过小会让长距关系变成很深的“接力”；窗口过大又减少节省。global token 可以把路径缩短，但它的表示容量和聚合冲突会成为新瓶颈。

例如一篇文档有 100K token，窗口 1K，只有 24 层。理论上一个早期实体的信息要跨越很多层才能到末尾；即使路径存在，重复的中间转换也可能损失精确数字。这个例子说明“线性 attention 复杂度”不能直接翻译为“长程理解能力保留”。

## 19.7 与块稀疏的关系

GPU 更喜欢规则的 block。将序列分成大小为 `b` 的块，只计算允许的 block pair，可以减少索引开销并改善内存访问。理论上的 token 级稀疏若导致很多小碎片，实际速度可能不如 dense kernel。

块稀疏要考虑：块内未使用 token 的浪费、不同请求长度的 padding、global block 的高 fan-out、梯度和 mask 的存储，以及是否有对应硬件 kernel。最好同时报告理论边数和实际 kernel 时间。

## 19.8 滑窗对不同任务的影响

局部语言建模、实时日志分类和持续对话通常受益于近期信息；文档问答、法律条款比较、代码跨文件引用和多跳推理则更依赖远程连接。把同一个窗口应用到所有任务，会让某些能力出现静默退化。

评估至少应包含：

1. 局部 next-token 和短程依赖。
2. 长距复制和 passkey。
3. 多实体条件绑定。
4. 版本冲突和多段证据合并。
5. 流式吞吐、p99 和显存。

## 19.9 训练和推理的一致性

训练时若使用 full attention、推理时改为窗口，模型可能没有学会依赖路径；反过来，训练时窗口、推理时放大窗口也可能遇到未见过的连接和位置分布。global token 的训练标记、position id、KV cache 以及 chunk 边界必须一致。

窗口滚动时还要决定是否保留起始 sink token、系统提示、文档标题和最近状态。StreamingLLM 的 attention sink 研究说明，简单删除最早 KV 可能造成分布变化；保留若干特殊 token 有时比只保留最近窗口稳定，但这不是任意模型的普适保证。

## 19.10 与 RAG 的边界

稀疏 attention 在模型内部改变信息流，RAG 在模型外部改变输入集合。RAG 能把远处相关文档搬近，但召回和排序有误差；稀疏 attention 不需要外部索引，却可能在输入已知的情况下无法直接连通远处证据。两者可以组合：先检索，再在候选文档内用局部/global pattern 处理。

## 19.11 机制与边界：图性质和 kernel 性能要同时看

可把多层 attention 看成有向图。除了每层边数，还要关心图的直径、可达性、全局 token 的瓶颈、不同 head 的互补性和残差旁路。理论上 BigBird 的稀疏结构可以保持较强表达性质，但实际模型质量还依赖训练、位置编码和任务分布。

硬件上，稀疏并不自动带来 speedup。若有效密度高、pattern 不规则或 batch 太小，索引和 kernel launch 可能吞掉收益。选择稀疏方案要先确认框架是否支持 fused/block-sparse kernel，再用端到端 tokens/s 和 tail latency 验证。

## 19.12 面试追问、误区与练习

**问：滑窗 attention 为什么能降低复杂度？**

标准回答：每个 query 只与固定窗口内约 `w` 个 key 交互，score 数从 `n^2` 降到 `nw`；但远程信息要通过多层传播或 global token 传递，质量和实际速度需要单独验证。

**问：加 global token 就等于恢复 full attention 吗？**

标准回答：不等于。global token 提供汇聚/广播路径，但信息必须经过有限表示和层间传播，不能保证每个 query 对每个原始 token 都有独立的直接访问。

常见误区包括把 sparse mask 当作稀疏矩阵乘就一定快、忽略 causal 方向、把窗口大小当作有效上下文、只测摘要不测精确引用，以及用训练 mask 和推理 mask 不同的实现做错误对比。

练习：对 16K token 设计三种 pattern：纯滑窗、滑窗+global、local+random+global，计算理论边数，并设计能区分它们长程能力的任务。

## 19.13 稀疏图的连通性和信息路径

把 attention mask 看成有向图，节点是 token，边表示可见关系。单层局部窗口的图很稀疏，但远距离节点之间是否存在可达路径，取决于层叠、global token、dilation 和 reset。

可以定义从位置 i 到 j 的最短层数 d(i,j)。d 越大，信息需要经过的表示变换越多，精确复制可能越困难。图连通性是必要条件，不是能力充分条件；还要测路径上的噪声、容量和训练信号。

## 19.14 稀疏实现与稠密 fallback

很多理论稀疏 mask 在实际 kernel 中可能退化为稠密计算，或者为了 batch 对齐产生大量无效 block。压测应确认真正执行的 block 数、显存读写和 kernel 名称。

如果稀疏模式遇到不支持的长度或 head 配置而 fallback，系统要暴露告警。否则线上吞吐下降会被误认为模型本身变慢，而实际是 mask 没有被编译成预期的路径。

## 19.15 图连通不等于证据可用

稀疏 mask 可以让任意两个位置存在一条路径，但路径上的每一层都可能压缩、混合或丢失细节。设 token i 到 j 的最短路径长度为 d(i,j)，路径需要经过的表示变换越多，精确复制和引用越容易受到噪声影响。

因此至少要做三种测试：单个数字远距复制，两个实体冲突选择，多段证据合并。若只有主题摘要保持良好，说明图连通性提供了语义传播，却没有提供可靠的原子证据传输。

## 19.16 窗口大小的质量—成本曲线

滑窗宽度 w 影响每层直接交互数量，也影响远程信息传播的层数。简单估算为：

~~~math
C_{\mathrm{local}}\approx O(Twd),\qquad
d_{\mathrm{propagate}}\approx O\left(\frac{T}{w}\right)
~~~

第二个式子只是链式局部窗口的直觉，global token、dilation 和层 schedule 会改变它。增大窗口可能减少路径长度，却提高 prefill 和 cache 成本；减小窗口省资源，却可能让远程证据经过更多层。

压测要把 w、层数、global token 数和 batch 一起改变，并报告真实 block、TTFT、TPOT、引用支持和峰值显存。

## 19.17 在线服务的边界语义

滑窗系统必须明确窗口在请求开始、chunk 边界、文档分段和多轮对话处是否重置。若 position 连续而 window offset 错一位，模型可能在普通文本上正常，却在边界附近丢掉关键证据。

prefix cache 和 sliding window 也不是天然兼容。共享前缀时要知道哪些层保留全局历史、哪些层只保留窗口，cache 淘汰不能破坏 global token 或 attention sink。runtime 应在 trace 中记录窗口起点和实际执行 mask。

## 19.18 图直径是必要条件，不是充分条件

把 token 看成图节点、允许的 attention 连接看成边，可以用图直径估算信息传播最少需要多少层。滑窗半径为 w 时，远处 token 可能要经过多层才能影响当前 query；加入 global token 可以降低直径，但也可能让 global 节点成为带宽和注意力噪声的瓶颈。

即使图连通，也不代表模型能找到正确证据。路径上的每一层都可能压缩、覆盖或混淆信息；任务需要的方向、实体和版本关系也可能与固定 pattern 不匹配。评测应把图可达性、精确召回、证据引用和最终任务成功分开。

## 19.19 Pattern 选择要由任务驱动

局部窗口适合局部语法、连续信号和相邻代码；扩张窗口适合多尺度上下文；global token 适合章节标题、问题和少量关键实体；随机块可以提高连通性，却会增加 kernel 不规则性。没有一种 sparse pattern 对所有任务都最优。

可以做一个受控实验：固定稠密模型参数和训练预算，只改变 window、dilation、global token 位置和 block size；测 local copy、long-range copy、文档引用、代码跨文件、吞吐、显存和 p99。对 global token 还要做数量消融，观察增加连接带来的收益是否只是更多参数或更多路由提示。

## 19.20 稀疏 Serving 的语义和回退

训练时使用的 mask 必须和推理 kernel 一致。线上混合长度 batch、padding、prefill chunk 和 cache page 可能使稀疏 pattern 发生边界变化；如果没有对应 kernel，runtime 可能悄悄回退到稠密 attention，质量不变但显存和尾延迟突然恶化。

服务应记录实际执行路径、稀疏命中率、fallback 次数、block 数、带宽和 p99。若动态稀疏根据输入选择连接，还要把 pattern 版本、路由结果和回滚语义纳入 trace；不能只在配置中写“sparse=true”。

## 19.21 证据路径也需要预算

滑窗的图连通性只能说明“理论上存在一条路径”，不能说明这条路径足以传递一个原子事实。假设关键数字位于位置 `i`，查询在位置 `j`，局部窗口宽度为 `w`，没有 global token 时，最短传播层数大致随 `|j-i|/w` 增长。每经过一层，表示都可能被其他 token 混合，路径越长，精确复制越容易退化。

因此，长程任务要把路径预算和任务预算放在一起。单点主题分类可以容忍表示压缩，多证据引用则需要同时保留实体、数值、版本和来源。对每个样本记录 `distance(i,j)`、经过的 global 节点数、有效层数、非零 block 数和最终引用支持率，才能知道窗口扩大带来的收益来自更短路径，还是来自更多无关计算。

```math
P_budget = P_reachable * P_preserve * P_read
```

三个概率分别表示证据可达、路径中信息没有被覆盖、查询最终正确读取。它们不是严格独立的统计分解，却能避免把“可达”直接写成“可用”。如果 `P_reachable=1` 而 `P_read` 很低，增加随机边可能没有意义，应该改训练任务、global token 语义或外部检索。

## 19.22 一个窗口选择的手算例子

设序列长度为 16K，比较窗口 `w=512`、`w=2048` 和 `w=4096`。忽略边界时，每层局部连接数大约是 `T*w`，因此窗口从 512 增至 4096，理论 pair 数增加约 8 倍；但从序列开头向末尾传播所需的局部跳数可能减少约 8 倍。两种变化分别影响计算成本和长程路径长度，不能只报告其中一个。

再加入 64 个 global token 后，所有位置都可以把信息写入或读取 global 节点，但 global 节点本身成为共享通道。实验应改变 global token 的语义：章节标题、问题 token、随机 token 和可学习摘要 token，观察它们对数字复制、版本消歧和主题摘要的差异。若随机 global token 也能提升主题摘要，却不能提高引用支持，说明收益可能只是汇聚容量，而不是证据寻址。

上线前还要做稠密 fallback 演练：当输入长度、block size 或 dtype 不满足 kernel 条件时，系统是拒绝、改用稠密路径，还是缩小窗口。每条选择都要记录延迟和质量，且在 admission 阶段告诉调度器真实资源需求。否则一个看似稳定的 sparse 配置可能在异常长度上突然把整批请求送进 dense path。

## 19.23 稀疏模式的选择条件

Longformer 原始论文为 https://arxiv.org/abs/2004.05150，BigBird 为 https://arxiv.org/abs/2007.14062；StreamingLLM 的 attention sink 讨论见 https://arxiv.org/abs/2309.17453。不同框架的 window、global token 和 block-sparse 实现可能不同，不能只依据论文名称推断运行时行为。

稀疏 attention 的核心是用可控信息图换取计算和内存。只有当图的连通性、任务证据路径和硬件 kernel 都过关时，减少连接才会变成真实系统收益。

## 19.24 Causal sliding window 的边界条件

滑动窗口最容易被忽略的是“窗口到底包含哪些位置”。对 causal decoder，位置 `i` 不能读取未来，若窗口宽度为 `w`，一种常见定义是：

```math
\mathcal{V}(i)=\{j\mid \max(0,i-w+1)\le j\le i\}.
```

也有实现把 `w` 定义为半径、把当前 token 之外的历史数记为 `w`，两者会相差一个 token。短序列上这个差异不明显，到了 chunk 边界、KV page 和 prefix cache 处就可能造成 off-by-one。工程测试应显式检查第一个 token、中间 token、窗口刚好满的位置、跨 chunk 的第一个位置和 padding 位置。

如果系统采用 attention sink 或少量永久保留 token，实际可见集合会变成局部窗口与 sink 的并集。cache 淘汰时不能只保留最近 `w` 个 page，否则模型可能丢失维持状态分布所需的开头 token。窗口规则、sink 规则和 position offset 应作为同一个版本化协议，而不是由不同模块各自解释。

## 19.25 Global token 的容量不是无限的

global token 可以把远程信息汇聚后广播，但它不是一条无损总线。若 `g` 个 global token 的维度为 `d`，一次层间传递最多只有大约 `g d` 个数值通道；真实可用容量还受到 attention 权重、层数、训练目标和噪声影响。文档主题摘要可能只需很少通道，多个精确数字、版本和引用则可能迅速超出容量。

可以用一个教学近似描述 global bottleneck：

```math
I(\mathrm{evidence};\mathrm{query})
\le I(\mathrm{evidence};\mathrm{global\ state})
\le C(g,d,L),
```

这里的 `I` 是互信息的直觉记号，`C` 不是公开模型的精确容量定理，而是提醒我们：增加 global token 可能提高信息通道，却不等于所有原始证据都被保留。实验应把主题摘要、单点数字、多个数字、冲突版本和引用位置分开，否则平均任务分数会掩盖瓶颈。

## 19.26 训练长度、位置编码与稀疏 pattern 必须一起验证

在短序列上训练局部窗口，再把窗口和位置范围直接外推到长文档，不能证明模型学会了长程路径。训练阶段如果从未见过跨越多个 global token 的证据组合，推理阶段即使 mask 允许连接，模型也可能不会使用它。

最小的长度消融可以固定模型宽度、数据混合和 optimizer，只改变训练中出现的最大长度，例如 4K、16K、64K；同时固定 window、global token 和 position 方案。评测分为长度内、长度外和边界三组，并记录 exact recall、引用支持、loss、TTFT、峰值显存和 fallback。若长度外质量突然下降，而 dense baseline 没有同样下降，问题可能来自 position 或 sparse path，而不是单纯的“上下文太长”。

还要区分 mask 变化和位置变化的影响。可以做四个对照：dense mask + 原 position、sparse mask + 原 position、dense mask + 长度方案、sparse mask + 长度方案。没有这组 factorial 对照，就不能把收益或退化归因给某一个模块。

## 19.27 从证据图到上线条件

一个长上下文稀疏系统的发布验收条件至少包含四类断言。第一，语义断言：未来 token 不可见，权限过滤后的证据才可见，工具结果不会改变高优先级策略。第二，路径断言：关键证据在预期层数内可达，global/sink 没有被错误淘汰。第三，执行断言：实际 kernel 使用了目标稀疏 pattern，没有静默 dense fallback。第四，服务断言：cache、stream、取消、重试和回滚不会提交未验证的 token。

一次完整压测应把以下基线放在同一表格中：dense attention、固定 sliding window、window + global、sparse + RAG，以及发生 fallback 的路径。对每条路径记录质量切片、证据支持率、TTFT、TPOT、p99、峰值显存、非零 block、fallback 比例和单位成功成本。这样读者可以看清“少算了多少”与“少答错了多少”之间的关系。

稀疏注意力真正值得采用的条件不是复杂度符号更漂亮，而是它在任务需要的证据路径上足够可靠，在硬件上真的执行稀疏 kernel，并且出现异常时有可解释的回退。把这三个条件讲清楚，才算把 sliding window 从一个结构名词讲成了可验证的工程技术。

## 19.28 DeepSeek V3.2 的 DSA：检索阶段与主注意力阶段要分账

DeepSeek V3.2 官方模型卡把 DeepSeek Sparse Attention（DSA）列为面向长上下文的技术突破。面试时可以用一个保守的两阶段抽象理解它：轻量 indexer 先估计哪些历史位置与当前 query 相关，再由主 attention 对有限候选做精确读取。这样讨论的重点不是把 DSA 简化成“只保留 top-k”，而是区分候选召回和最终证据读取。

至少要分开记录 `index_recall`、`final_evidence_recall`、top-k 排序成本、主 KV 与 indexer 表示的字节数、TTFT/TPOT、长程 needle retrieval 和 dense fallback。候选阶段漏掉关键位置时，主 attention 即使计算完全正确也无法恢复；候选很多时，稀疏收益又可能被 indexer、排序和 gather 成本吃掉。

技术报告补充了 DSA 的训练路径：它基于 MLA 的 MQA 模式，让一个 latent KV entry 在 query token 的多个 query heads 间共享。dense warm-up 冻结主模型，只训练 lightning indexer；indexer 把各 head 的 dense attention 分数聚合并做序列维度的 L1 归一化，以 KL loss 对齐主 attention 分布。随后进入 sparse training：indexer 输入与主图 detach，indexer 只由 KL loss 更新，主模型只由 language-modeling loss 更新；每个 query 选择 2,048 个 KV tokens，warm-up 约 2.1B tokens，sparse stage 约 943.7B tokens。

因此复杂度账本要写成“主 attention 从二次复杂度降到与 selected-token 数线性相关，但 indexer 仍有二次项，并依赖更低常数的实现”。官方 Exp README 还指向 DeepGEMM 的 indexer-logit kernel、FlashMLA 的 sparse-attention kernel 和 TileLang 的研究实现，并记录 indexer RoPE 布局 bug 修复。这里的实现入口不能冒充本地复现结果；GLM-5 的 `index_topk`、GLM-5.2 的 IndexShare 和 DeepSeek V4 的 CSA/HCA 也不能迁移成 V3.2 的精确字段。完整层排布、生产 kernel 行为、KV/indexer 字节账本、召回曲线和硬件 profiling 仍待核验。

## 19.29 DeepSeek V3.2 的实现证据账本：最终模型与实验 demo 分开

Artificial Analysis 的精确条目是 `deepseek-v3-2`，当前页面标签为 `Non-reasoning`。2026-09-20 详情快照为 3,625,720 bytes，SHA-256 为 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`；页面的 128K context、约 648B/37B total/active 和 Intelligence Index `16.043537719683` 都是第三方目录字段。DataCurve 当前没有精确的 `mini_swe_agent_deepseek_v3_2_*` 行，所以不能借用 V4 的 DeepSWE 结果。

本轮最容易写错的是把两个 artifact 合成一个“V3.2 配置”。应该维护两行账本，并先固定官方 revision：模型仓库 commit 为 `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`，`config.json` SHA-256 为 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`。

| artifact | 可核验字段 | 允许得出的结论 |
| --- | --- | --- |
| 最终 V3.2 模型资料 | `q_lora_rank=1536`、`kv_lora_rank=512`、61 layers、前三层 dense、256 routed/top-8/1 shared、64 index heads、`index_head_dim=128`、`index_topk=2048`、163840 positions、YaRN factor 40、BF16/FP8 配置 | 固定 revision 的 `config.json` 字段；记录最终模型资料，不自动证明所有生产 kernel |
| V3.2-Exp inference demo | 61 layers、前三层 dense、256 routed/8 active、`q_lora_rank=1536`、`kv_lora_rank=512`、64 index heads、`index_topk=2048`、FP8 index path | 解释实验版推理路径，不覆盖最终 V3.2 字段 |

此前把最终 V3.2 的字段读成与实验不同的数值是证据错误；固定官方配置与 V3.2-Exp demo 都是 `q_lora_rank=1536`。两行账本仍不能合并成一份“生产配置”：最终模型的 revision/config、实验 demo 的代码快照、kernel 和 serving recipe 的证据等级不同。面试中应先问清楚“你说的是最终模型、实验推理 demo、kernel 还是 serving recipe”，再讨论结构和性能。

## 19.30 从 FP8 indexer 到 sparse MLA 的执行顺序

V3.2-Exp 的可读实现可以抽象为：

```text
query/key cache
    -> FP8 index score (`fp8_index`)
    -> non-interleaved indexer RoPE
    -> causal mask
    -> top-k candidate positions
    -> gather latent KV + positional cache
    -> sparse MLA
```

这里有两个经常混淆的 RoPE：indexer 的 non-interleaved layout 与 MLA 主路径的布局不同。把它们当作同一通道排列，会得到看似合理但排序完全不同的候选位置。`top-k` 也只是在候选阶段筛选历史位置，不等于主 attention 已经正确读取了证据；所以要同时测 `index_recall`、`final_evidence_recall` 和 attention 输出误差。

实现还暴露了 prefill/decode 的不同账本：prefill 参考路径使用 MHA，decode 使用 MQA，并把低秩 latent KV 与 positional cache 分开保存；部署路径涉及 FP8 KV cache，而教学 demo 的量化/反量化不能直接当作生产精度和性能结论。输入长度、dtype、cache page 或 kernel 条件不满足时，系统可能回退到 dense path，必须在 trace 中记录实际路径。

## 19.31 kernel、recipe 与模型能力不能互相替代

TileLang 示例把执行拆成 Lightning Indexer、radix/histogram top-k selector 和 sparse MLA，并描述 double buffering 与 FP8 sparse MLA；DeepGEMM 的公开 PR 涉及 FP8 MQA logits，FlashMLA 的公开 PR 涉及 sparse MLA 路径；vLLM recipe 建议 `DP=8, EP=8, TP=1`，并说明 DeepGEMM、TP fallback、FP8/BF16 KV cache 和 `max-num-seqs` 等部署选项。

这些材料证明“公开实现路径可追溯”，不证明最终 V3.2 权重已经在本地用这些 kernel 达到某个吞吐，也不证明 recipe 的 GSM8K 结果是裸模型分数。正式报告至少要分开：

1. 模型能力：固定 checkpoint、prompt、工具和任务集后的结果。
2. kernel 行为：候选召回、排序、gather、cache bytes 和实际执行路径。
3. serving 行为：并行拓扑、batch、TP/EP fallback、TTFT/TPOT 和 p99。
4. recipe 结果：vLLM/DeepGEMM/FlashMLA/lm-eval 版本、硬件、few-shot 和运行参数。

只有四本账都绑定 revision 和 harness，才能讨论“稀疏 attention 是否带来端到端收益”。否则把一个 kernel 的局部加速写成模型能力提升，就是典型的证据越界。

## 19.32 V3.2 的 thinking-with-tools 协议边界

V3.2 的模型卡和技术报告还把 scalable RL、large-scale agentic task synthesis 与 `thinking with tools` 放在同一条后训练主线上。消息协议需要区分 reasoning、tool call、tool result 和 final answer；parser 只能把输出转换成候选事件，不能授予工具权限，也不能证明工具已执行。

V3.2-Speciale 是关联的深度推理变体，官方边界是不支持 tool calling。它不能被当成 V3.2 coding agent 的直接替代品；Speciale 的工具指标应记为 `not_applicable`，而不是当成失败率为零。宿主仍要执行 schema、权限、确认、超时、重试、结果回灌和 artifact verifier。

## 19.33 DSML encoding 是消息协议，不是权限系统

固定 revision 的 `encoding/encoding_dsv32.py`（SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`）把 V3.2 的工具交互落成可审计的文本边界：`system`、`developer`、`user`、`assistant` 和 `tool` role 进入编码器；工具调用使用 DSML 风格的 `<｜DSML｜function_calls>` / `<｜DSML｜invoke>`，思考内容使用 `<think>`/`</think>` 与 `reasoning_content`，工具返回使用 `<function_results>`/`<result>`。参数还要区分普通字符串与 JSON scalar/object/list，不能一律当作字符串拼接。

这给面试题一个明确的分层答案：encoding/parser 负责把消息与候选事件变成模型可读或宿主可读的格式，schema validator 负责结构，权限系统负责“能不能调用”，执行器负责“是否真的执行”，verifier 负责“结果是否满足任务”。即使 DSML 文本解析成功，也不能据此授予权限、确认副作用已经发生，或把模型输出当作工具执行回执。重放测试应保留 reasoning、tool call、tool result、final 的事件边界，并记录异常格式、重复调用和未知执行状态。

本节的公开实现和报告入口见 [DeepSeek V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)、[V3.2-Exp inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)、[TileLang 示例](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)、[DeepGEMM PR](https://github.com/deepseek-ai/DeepGEMM/pull/200)、[FlashMLA PR](https://github.com/deepseek-ai/FlashMLA/pull/98) 和 [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)。

## 19.34 当前 AA 目录漂移与 README 证据边界

2026-09-21 的 Artificial Analysis 详情页仍然是同一个 deepseek-v3-2 条目，但页面结构化字段变为 685B total、37B active、128K context 和 Intelligence Index 16.043537719683；9 月 20 日快照曾显示约 648B total。这个差异应回答为“第三方目录/provider 数据在不同采集时点发生漂移”，不能回答为模型突然扩容、重新训练或换了架构。当前页面没有可用 output-speed/TTFT 字段，也不应从其他 provider 或相邻模型补齐。

V3.2-Exp README 还给出一个很适合面试的实现证据链：实验版本基于 V3.1-Terminus，并用对齐配置做 reasoning without tool use 与 agentic tool use 的发布方对照；2025-11-17 更新明确修复 indexer non-interleaved RoPE 与 MLA interleaved RoPE 的布局差异。这里要把“算法名相同”和“通道/内存布局相同”分开，复现时同时固定 RoPE layout、权重切分、cache 格式和 top-k 边界。

实现入口也要按证据等级回答：TileLang 更适合研究可读性，DeepGEMM 提供 indexer logits/paged logits 的 CUDA 路径，FlashMLA 提供 sparse MLA 路径，SGLang README 给出 dsv32 镜像和 tp=8、dp=8、enable-dp-attention 的服务命令。它们证明公开实现路径存在，不证明本机已经完成 full-weight load、数值正确性、目标硬件 profiling 或线上工具验收。若 vLLM recipe URL 当前返回 404，结论只能是“当前链接/线路不可取得”，不能升级为“vLLM 没有 V3.2 实现”。

这一节的面试追问可以收束成三句：第一，榜单参数字段是目录证据，不是训练变更；第二，README、reference kernel 和 serving recipe 分别回答发布描述、实现路径和部署入口；第三，真正的 production gate 还要经过 revision、依赖、硬件、完整权重、召回/误差、状态恢复、p99 和 tool/verifier acceptance。

## 19.35 DeepSeek V3.2 Search Agent 的上下文管理

V3.2 技术报告 §4.4 把上下文管理作为 Search Agent 的 test-time compute 策略：当工具轨迹 token 使用量超过 context window 的 80% 时，对比三种串行处理方式——`Summary` 摘要溢出轨迹并重新 rollout，`Discard-75%` 丢弃最早 75% 的工具调用历史，`Discard-all` 清空先前工具调用历史并重启上下文。另设 `Parallel-fewest-step` 基线，采样 N 条独立轨迹并选 steps 最少的一条。

评测使用商业 search API。报告称 V3.2 的 128K context 令约 20% 以上测试案例超限；Table 2 的 BrowseComp Pass@1 为无 context management `51.4`、带 context management `67.6*`（星号表示使用该方法）。§4.4 还报告 `Summary` 平均扩展到 364 steps，performance improvement “up to 60.2”；Figure 6 已通过 arXiv v1 SVG 视觉复核，横轴为 Real Steps、纵轴为 Browsecomp，Summary 曲线约在 364 steps 对应 60.2。结合纵轴，该数应视为分数/指标值而非 `+60.2%` 相对增幅。`Discard-all` 得分 67.6，作者称其与 parallel scaling 可比且使用更少 steps。不要将这些 Search Agent harness 结果写成裸模型、DataCurve 或 Artificial Analysis 分数；图中散点无数字标签，未逐点数字化。

面试时可把核心 trade-off 说清楚：摘要以有损压缩换取历史连续性；丢弃前缀或全部工具历史可减少 token 与重复上下文，却可能丢失证据来源、已尝试路径和关键观察。Context reset 本身不会创建持久记忆，生产 harness 应在外部保存来源 URL、可验证观察、未决问题、工具副作用状态和 artifact 引用，并对齐预算比较 Pass@1、steps、工具成本、证据可追溯性与最终任务完成率。一般化的折叠状态与验收方法见第二十册第 18 章 [上下文折叠](../../book-20-agent-harness-runtime/chapters/18-context-folding上下文折叠.md)；报告来源为 [arXiv v1 §4.4](https://arxiv.org/html/2512.02556v1)。

## 19.36 DeepSeek V3.2 Figure 2：索引器不是主注意力

DeepSeek V3.2 技术报告 Figure 2 把稀疏注意力画成两个功能不同的阶段：Lightning Indexer 计算位置分数，Top-k Selector 选择候选 KV entries；候选再交给 Multi-Query Attention（Core Attention），与主 query 一起计算 attention 输出。图注、§2.1 和 Eq. (2) 支持这条数据流。面试回答要避免把 index score 或 top-k 选择误说成 attention 输出：它们负责“找哪些位置”，核心注意力负责“读取候选并形成输出”。

此图解释架构模块关系，不足以证明完整 production kernel、实际召回率或端到端性能。Figure 2 SVG（2026-09-29，经 `10.24.27.134:7890` 获取，269,427 bytes，SHA-256 `1e6bc6c61ea26ae2b8528edb1eab14874832470fd38f115fe9b8feb615facb9a`）已渲染并视觉核验；来源为 [arXiv v1 Figure 2](https://arxiv.org/html/2512.02556v1/v32_arch.svg) 与[论文正文](https://arxiv.org/html/2512.02556v1)。

## 19.37 DeepSeek V3.2 Figure 3：复杂度与服务成本不是一回事

技术报告 Figure 3 分别画出 prefill 与 decode 两个面板：横轴 Token Position，纵轴 Cost Per Million Tokens，比较 DeepSeek-V3.1-Terminus 与 V3.2。视觉上 V3.2 的成本曲线随 token position 增长得更慢，长位置差距明显；极短位置两条线接近并有交叉，因此不能回答成“V3.2 在所有上下文长度都更便宜”，也不应从无数字标签的曲线上抄精确成本。

证据口径同样重要：报告称曲线来自 H800 上实际部署服务的 benchmark，按每 GPU 小时 2 美元的租赁费估算；短序列 prefill 还使用 masked MHA mode 模拟 DSA。它不是 API 价目或独立复现。DSA 将主 attention 复杂度从 `O(L²)` 降至 `O(Lk)`，但 indexer 仍为 `O(L²)`；真实成本还取决于实现、上下文长度与服务路径。面试时要把算法复杂度、kernel 实测和按租赁价折算的服务成本分开。

Figure 3 的 [prefill SVG](https://arxiv.org/html/2512.02556v1/cost_prefilling.svg)（33,398 bytes，SHA-256 `b045c26eb19d92325de7e86aabec905a9ac8bdb6729db94e04c8701693b19705`）与 [decode SVG](https://arxiv.org/html/2512.02556v1/cost_decoding.svg)（31,868 bytes，SHA-256 `5894e01515f7f9e9ef9045ae8e30a092f5cc1e4229936c493c66c4748127c61d`）均已渲染为 1600×1200 并视觉核验；正文与其余图表/公式仍未全部逐项视觉检查。

## 19.38 DeepSeek V3.2 Figure 4：按消息类型保留 reasoning

技术报告 Figure 4 展示 tool-calling 的多轮上下文：同一轮中模型发出 thinking/tool call、接收 tool result 后，前序 thinking 仍在后续输入中；模型继续 reasoning 并完成回答。到下一条 user message 时，图中的新轮输入保留旧工具调用、工具结果和上轮答案，但移除了旧 thinking。它与 §3.2.1 的正文规则相符：只有新 user message 会触发历史 reasoning 的清理，追加 tool-related messages 则保留 reasoning history。

这应理解为消息 role 与上下文回放策略的耦合，不是永久记忆或通用客户端保证。若某个 Agent 把工具输出伪装成 user message，保留条件可能改变；审计时要固定实际消息 role、history builder 和模型版本。Figure 4 原图为 1280×671 JPEG（76,981 bytes，SHA-256 `58623875cc487b3cbbd60955c14801c07d5f526d2b2e875795e3bf5ace511733`），已视觉核验；来源为 [arXiv v1 Figure 4](https://arxiv.org/html/2512.02556v1/figures/template.JPEG) 和[论文 §3.2.1](https://arxiv.org/html/2512.02556v1)。

## 19.39 DeepSeek V3.2 Figure 5：合成 Agent 任务上的 RL 消融

技术报告 Figure 5 比较从 V3.2-SFT checkpoint 出发、仅使用合成 general-agent tasks 并以 non-thinking mode 做 RL 的训练曲线，与 V3.2-SFT 和仅在 search/code environments 做 RL 的 V3.2-Exp 基线。分面覆盖 Tau2-Bench Airline/Retail/Telecom/Overall、MCP-Mark Filesystem/PostgreSQL，以及 MCP-Universe Financial Analysis/Location-Navigation/3D-Designing。曲线总体随训练 steps 上升，但不同环境的幅度和波动并不相同；图中没有标出所有精确数值，不应从像素估读成绩。

这张图支持的结论限于发布方这项训练消融：合成 Agent 数据上的 RL 在若干环境评测中呈现提升。它不是最终 V3.2 的独立 benchmark，也不能证明真实环境泛化的因果、训练数据没有污染，或完整 RL recipe 已公开；结果不属于 Artificial Analysis 或 DataCurve 分数。原图 [Figure 5](https://arxiv.org/html/2512.02556v1/figures/synthesis-rl-plot.png) 为 1432×1024 PNG（222,137 bytes，SHA-256 `2107344cc2002f51bc1922df0bdcabd7afa74bb0f6cb5d3aa996f33e3f0d7cc3`），已视觉核验。

## 19.40 DeepSeek V3.2 Figure 7：MLA 的 MHA 与 MQA 两种执行形态

技术报告 Appendix A 的 Figure 7 并列画出 MLA 的 MHA mode 与 MQA mode。MHA 图中，压缩 latent KV `c_t^KV` 经每头的 `W_UK_i`、`W_UV_i` 投影，形成各 query head 使用的 key/value；MQA 图则让各 query head 共享 latent KV entry `c_t^KV` 与位置 key `k_t^R`，注意力结果再按 head 投影并拼接。正文也明确说明 MLA 的每个 latent vector（KV entry）由该 token 的所有 query heads 共享。面试要点是区分“KV 的存储/投影形式”和“attention 的计算形式”：两幅图不是把 MLA 换成另一套模型架构。

报告图注把 DeepSeek-V3.1-Terminus 的阶段安排写为 training/prefill 用 MHA、decode 用 MQA；V3.2-Exp inference demo 也分别展示 prefill MHA 与 decode MQA 路径。讨论时应把论文对 V3.1-Terminus 的原文与 V3.2-Exp 实现证据分开，不将单一实现推广成所有 V3.2 serving backend 的硬性合同。图示能解释共享 latent KV 与 per-head expansion 的差异，但不单独证明实际缓存字节数、吞吐提升、kernel 覆盖或质量等价。

Figure 7 的 [MHA panel](https://arxiv.org/html/2512.02556v1/MLA-MHA.svg)（244,227 bytes，SHA-256 `afd21a50bbaef58314036862cb6ce44dca81a9d42a414c0969074fbf954b32b4`）与 [MQA panel](https://arxiv.org/html/2512.02556v1/MLA-MQA.svg)（224,311 bytes，SHA-256 `c6aa4f3e2ea222d2a75ef000dca2f9e8c48a820116957c6a741cbd0f253d77c3`）均已渲染并视觉核验。

## 19.41 DeepSeek V3.2 Figure 1：读 benchmark 图先核对评测口径

报告 Figure 1 把 benchmark 分为 Reasoning Capabilities 与 Agentic Capabilities：包含 AIME 2025、HMMT 2025、HLE、Codeforces、SWE Verified、Terminal Bench 2.0、τ²-Bench 和 Tool Decathlon。准确率/Pass@1 使用左侧百分比轴，Codeforces Rating 使用右侧独立量纲；不能把柱子的视觉高度跨指标比较成一个统一排名。图中展示的是 DeepSeek-V3.2-Speciale、V3.2-Thinking 与 GPT-5-High、Claude-4.5-Sonnet、Gemini-3.0-Pro 在报告设定下的结果，不是 AA/DataCurve 排名，也不是独立复现。

图注限定 HMMT 采用 February 2025 赛次、HLE 采用 text-only 子集。评测正文还说明：温度 1.0、上下文 128K；数学任务使用统一的 step-by-step 模板，而 V3.2-Thinking 另用 HLE 官方模板测得 23.9；Figure 1 的 HLE bar 为 25.1。此差异提醒面试回答必须绑定 prompt/template，不能只记一个 benchmark 名称和分数。Tool-use benchmark 使用标准 function-call 格式并启用 thinking；MCP-Universe/MCP-Mark 采用作者内部环境，官方环境可能不同。报告图表因此应作为发布方评测证据，并注明任务版本、模板、环境、模式和指标，而不是抽离为通用模型能力结论。

arXiv v1 的 [Figure 1 SVG](https://arxiv.org/html/2512.02556v1/v32_performance.svg) 经 `10.24.27.134:7890` 获取，HTTP 200，128,338 bytes，SHA-256 `14354740f4d54692b6af6323cc12d3a5f0e0f937bc2b7dd65021307bd7820973`；渲染后已视觉核验图例、分组、双纵轴与标签。

## 19.42 DeepSeek V3.2 DSA Eq. (1)–(4)：分清打分、检索与蒸馏

公开实现核查：V3.2-Exp 官方仓库可访问的根目录列出 README、技术报告 PDF、license、cost image 与 `inference/`；README 指向 inference demo 及外部 kernel 项目，没有提供用于确认该 loss 细节的 trainer/loss 源码路径。这只说明当前公开的 V3.2-Exp 仓库未解答该归一化问题，不代表 DeepSeek 其他仓库或内部训练代码一定不存在。

报告 Eq. (1) 的 Lightning Indexer score 是 query-conditioned 的加权 ReLU 相似度：

$$
I_{t,s}=\sum_{j=1}^{H^I}w^I_{t,j}\,\operatorname{ReLU}(\mathbf q^I_{t,j}\cdot\mathbf k^I_s).
$$

`q^I` 与权重 `w^I` 由 query token `h_t` 得出，`k^I_s` 由历史 token `h_s` 得出。这个分数用于排序；Eq. (1) 本身没有 softmax，因此不是概率，也不是最终 attention weight。Eq. (2) 才把 score 的 Top-k 位置映射成 latent KV 集合，并由主 attention 计算输出：

$$
\mathbf u_t=\operatorname{Attn}\!\left(\mathbf h_t,\{\mathbf c_s\mid I_{t,s}\in\operatorname{TopK}(I_{t,:})\}\right).
$$

Eq. (3) 与 Eq. (4) 描述 indexer 的两阶段训练监督，而非主模型的 attention 公式。Dense warm-up 把所有主 attention heads 的分数求和，并沿 sequence 维做 L1 normalization 得到目标 `p_{t,:}`，再最小化 `D_KL(p_{t,:} || Softmax(I_{t,:}))`；此阶段冻结主模型，仅训练 indexer。Sparse stage 将 indexer input 从主计算图 detach，构造 `S_t=TopK(I_{t,:})`，在选中位置上计算 `D_KL(p_{t,S_t} || Softmax(I_{t,S_t}))`；报告称 indexer 只由该 KL loss 更新，主模型只由 language-modeling loss 更新。它解释了 dense teacher signal 如何迁移到稀疏选择，但不等于对所有未选位置做最终 attention。

复现边界：HTML 源保留的 Eq. (4) 写作 `p_{t,S_t}`，没有在展示式中单独写出子集后的重新归一化步骤；报告正文也未在这段明确展开该细节。不要自行替作者补出 normalization；若要实现一致性，应查公开训练代码或作者补充说明。该段公式来自 arXiv v1 HTML 的 TeX `annotation`，Eq. (1)–(4) 已核对其源文本；虽然本轮经 7890 下载了 PDF（980,616 bytes，SHA-256 `2bec0671778769c159ec389412727d1f3d4889fe1c71564b61edaa24705bd17b`），当前环境没有可用 PDF 渲染/抽取器，因此不声称已在 PDF 页面上视觉核验公式。

## 19.43 DeepSeek V3.2 scalable RL：Eq. (5)–(9) 与稳定化边界

报告 §3.1 先给出 GRPO 目标，再补充四项稳定化方法。按 Eq. (5)，外层对 group 中的 G 条 rollout response 求平均，内层先对每条 response 的 token 目标取长度平均；token 项是 clipped policy objective，减去相对 reference policy 的 KL penalty。面试时要说清楚这是“组内 response 平均 + response 内 token 平均”，不能不加说明地改成对所有 token 一次性平均。

Eq. (6) 定义逐 token importance ratio：

$$
r_{i,t}(\theta)=
\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})}.
$$

这里 old policy 生成 rollout，current policy 提供分子。报告按每条 response 的 outcome reward \(R_i\) 计算 group-centered advantage：

$$
\hat A_{i,t}=R_i-\operatorname{mean}(R_1,\ldots,R_G).
$$

展示的定义明确做了组内中心化；式子没有写组内标准差除法，因此不要把它改述为 z-score advantage。报告文字称其为 group normalization，但公开公式具体展示的是减组均值。

Eq. (7) 把 reference-policy KL 项改写成可在 old-policy rollout 上用 importance ratio 估计的形式：

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})}
\left(
\frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-\log\frac{\pi_{\mathrm{ref}}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
-1\right).
$$

因此这里有两种不同的概率比：外层 \(\pi_\theta/\pi_{\mathrm{old}}\) 是 rollout 分布修正，括号内 \(\pi_{\mathrm{ref}}/\pi_\theta\) 用于构造 KL 项。作者称该修正后的 KL estimator gradient 无偏；他们还指出，当 sampled token 在 current policy 下远低于 reference policy 时，原 K3 估计可能产生无界权重和高噪声，并报告不同领域可采用不同 KL 强度，数学任务有时可弱化或移除 KL。这些是论文给出的分析与经验，不是独立复现，也不构成完整 RL recipe。

Eq. (8) 将 binary mask \(M_{i,t}\) 乘在 clipped policy objective 上，KL penalty 仍在 mask 外；所以不能把它讲成“屏蔽整条样本的全部损失”。Eq. (9) 规定仅当 advantage 为负且整条 response 的平均 log policy divergence 超过阈值 \(\delta\) 时置零：

$$
M_{i,t}=
\begin{cases}
0,& \hat A_{i,t}<0\ \text{且}\ \displaystyle\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\log\frac{\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})}
{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}>\delta,\\
1,&\text{otherwise}.
\end{cases}
$$

论文说明 \(\pi_{\mathrm{old}}\) 使用推理框架直接返回的 sampling probability，以同时反映多轮更新和训练/推理实现差异造成的 off-policy；作者只 mask negative-advantage sequences。尽管 \(M\) 记为 token 下标，触发条件中的 divergence 是 response 级长度平均，不是逐 token 阈值。

同一节还披露 Keep Routing 与 Keep Sampling Mask。前者把 rollout 时推理框架实际走过的 MoE expert routes 带到训练中复用，减小训练/推理框架或策略更新导致的 active parameter subspace 不一致；报告称其自 DeepSeek-V3-0324 起用于 RL pipeline。后者保留 rollout 时 top-p/top-k 截断 mask，并在 current policy 训练时复用，令 old/current policy 使用相同 action support；作者称 top-p 与此策略结合有助于语言一致性。报告没有给出完整 reward 权重、rollout/update 预算、优化器或全部工程细节，因此仍应标为部分公开的稳定化设计。

以上 Eq. (5)–(9) 通过固定 arXiv v1 HTML 的 application/x-tex annotations 核对，HTML 快照 295,170 bytes / SHA-256 5da74d488b218a45a995838b94feb95aacc63c03b806f8462496bfd4bceb07ef；本轮经 7890 请求返回 HTTP 200，和既有快照一致。当前环境无 PDF renderer，故本节是公式源文本核对，不是 PDF 页面视觉核验。完整训练 recipe 和独立复现仍待补证。
