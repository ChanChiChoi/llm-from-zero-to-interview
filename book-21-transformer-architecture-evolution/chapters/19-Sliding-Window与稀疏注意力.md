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
