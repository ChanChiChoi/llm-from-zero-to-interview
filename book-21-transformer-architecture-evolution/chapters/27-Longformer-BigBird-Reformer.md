# 第二十七章：Longformer、BigBird、Reformer 等长序列路线

第十九章已经把稀疏 attention 表示成一张连接图，第二十六章说明这些连接怎样映射到计算块。本章再看三篇代表性工作如何决定图上的边：Longformer 用任务指定的位置补足局部窗口，BigBird 把随机边加入局部和全局连接，Reformer 则用当前表示的哈希结果近似挑选相似位置。它们减少或限制 attention 的候选范围，但做法和遗漏风险不同。

Reformer 还组合了另外两项内存技巧：可逆残差层和按位置分块的前馈计算。这两项处理训练时怎样保存中间激活，并不改变 attention 允许哪些 token 互相读取。把这些机制分开，才能看清三条路线分别省下了什么、又增加了什么成本。

## 27.1 三种长序列路线分别怎样选择连接

长序列 attention 的共同问题是：若每个 query 都与全部 key 计算分数，长度为 $n$ 时就要处理约 $n^2$ 组位置对。稀疏方法减少参与 attention 的候选，但“哪些候选留下来”有多种答案。

Longformer 由任务或配置指定部分位置为 global，其余位置主要读固定邻域；BigBird 在局部和 global 边之外加随机连接；Reformer 则根据当前 query/key 表示的哈希结果，让一部分候选进入同一分块。前两者主要修改预先设定的连接图，Reformer 的连接随输入表示变化，因而可能错过本应交互的位置。

理解这三种设计时要分开三个问题：候选边由谁产生，attention 在哪些位置上归一化，以及训练过程如何保存激活。第十九章负责一般 pattern 与信息路径；第二十六章负责 token mask、block layout 和执行路径；本章聚焦三篇方法各自的设计与论文证据。

## 27.2 Longformer：局部窗口、扩张窗口与任务 global

Longformer 的局部窗口让每个位置读取附近 token。窗口宽度固定时，注意力位置对数量随序列长度近似线性增长。扩张窗口在可见位置之间留出间隔，使同样大小的窗口覆盖更远距离；代价是间隔中的 token 不再是这一层的直接邻居。多个层或 head 可以使用不同窗口和扩张率，组合近邻上下文与较远范围。

长文档分类和问答还需要少数位置汇集全篇信息。Longformer 因此允许任务指定 global 位置，例如分类任务中的 `[CLS]`，或问答中的问题 token。原论文的双向长文档设置中，global 位置可以读取整个序列，其他位置也能读取 global 位置。把第 $i$ 个 query 可见的位置记作集合 $\mathcal{V}_i$，这类规则可写为：

```math
\mathcal{V}_{i}^{\mathrm{Longformer}}=W(i)\cup G.
```

其中 $W(i)$ 是位置 $i$ 的局部或扩张窗口，$G$ 是少量 global 位置。若窗口宽度 $w$ 和 global 数量 $g$ 保持固定，连接数量级为 $O(nw+ng)$。这说明该 pattern 的候选边数如何随长度增长，并不保证特定任务一定保留正确证据。

global 路径在原论文中使用独立的 Q/K/V 投影，与局部窗口的投影分开。作者还提出 Longformer-Encoder-Decoder（LED）用于长文档摘要。[Longformer 原论文](https://arxiv.org/abs/2004.05150)中的机制和任务结果都绑定具体模型与训练设置；双向 encoder 的 global 可见规则不能直接照搬到 causal decoder，后者必须遵守未来位置不可见的约束。

## 27.3 BigBird：local、random 与 global 的稀疏图

BigBird 组合三类边：local 边保留邻近信息，random 边为不同区域增加连接机会，global 位置让指定节点与全序列交互。random 边不是语义检索器，它不读取内容后挑选“最相关”的证据；它提供的是随机图上的连接机会。

记局部窗口为 $W(i)$，位置 $i$ 的随机候选为 $R(i)$，global 位置集合为 $G$。其连接结构可以概括为：

```math
\mathcal{V}_{i}^{\mathrm{BigBird}}=W(i)\cup R(i)\cup G.
```

若每个位置的局部宽度 $w$、随机连接数 $r$ 和 global 数 $g$ 固定，边数数量级为 $O(n(w+r+g))$。论文还区分两种 global 设计：ITC 把输入中已有的位置指定为 global；ETC 则额外加入 global token。两种方法改变的是全局表示来自原输入还是新增位置。

BigBird 的作者将 pattern 按块组织，再用连续块张量、reshape、roll 和有限 gather 形成较规则的计算。这是论文针对块执行的实现路线；具体 mask、block size 与 kernel 能否匹配，仍须按实现确认，通用块布局见第二十六章。

论文在形式化条件下证明了特定 BigBird encoder 类的 universal approximation 性质，并讨论了 encoder-decoder 的 Turing completeness；同一理论分析也指出，某些任务在稀疏图上可能需要比完整 attention 更多的层。例如，对最远向量检索任务，论文在标准复杂度假设下给出稀疏边数约为线性时需要近线性层数的下界。[BigBird 原论文](https://arxiv.org/abs/2007.14062)的这些结果说明稀疏图的表达能力和代价都依赖假设；它们不等于某个训练好的模型必然达到 dense attention 的任务质量，也不证明目标硬件上更快。

## 27.4 Reformer：LSH 候选、排序分块与近似 attention

Reformer 的 locality-sensitive hashing（LSH）attention 根据 query/key 表示的随机投影哈希结果筛候选。相近向量较可能落入同一个 bucket，因此同 bucket 的位置成为近似 attention 的候选。原论文使用共享的 query/key 表示，并对 key 作归一化，使同一位置的 query 与 key 使用一致的哈希结果。

第 $r$ 轮哈希中，query 位置 $i$ 与同 bucket 的 key 候选可写为：

```math
\mathcal{B}_{i}^{(r)}=\{j\mid h_{r}(q_{i})=h_{r}(k_{j})\}.
```

这里的集合表示哈希碰撞关系；实现还会按 bucket 和原序列位置排序，再把排序后的 token 分块。每个 query chunk 会处理本 chunk 及相邻 chunk，以便覆盖 bucket 边界上的候选。因果设置同时携带原始位置索引，排序后仍按原位置屏蔽未来 token。

一次哈希可能把相似位置分开。Reformer 可以用多轮独立哈希扩大候选集合；增加轮数通常也会增加计算。主 attention 在由这些哈希和分块规则确定的集合内执行 softmax，分母只包含入选 key，所以它是对完整 dense attention 的近似，不是对全部 key 的等价重排。论文报告的 $O(L\log L)$ attention 复杂度属于其 bucket、chunk 和哈希轮数设置；哈希计算、排序、填充及具体硬件开销仍须单独考虑。[Reformer 原论文](https://arxiv.org/abs/2001.04451)

## 27.5 Reformer 的可逆 residual 与前馈分块

Reformer 的 LSH attention 减少参与 attention 的位置对，但深层网络训练还要处理每一层的中间激活。可逆 residual block 把隐藏状态拆成两部分：attention 更新一半，前馈层更新另一半。令 $F$ 表示 attention 变换、$G$ 表示前馈变换，耦合更新与逆向重构为：

```math
\begin{aligned}
y_{1}&=x_{1}+F(x_{2}),\\
y_{2}&=x_{2}+G(y_{1}),\\
x_{2}&=y_{2}-G(y_{1}),\\
x_{1}&=y_{1}-F(x_{2}).
\end{aligned}
```

反向传播时，可以从后层状态逐层重构前层激活，减少按深度保存所有中间状态的需求。这个结构改变 block 的残差组织，也会在重构时重新计算变换；它节省的是训练激活存储，并不减少 attention 的候选边。

前馈网络对各 token 位置分别计算，因此可把序列切成若干 chunk 逐段处理，降低宽中间张量的峰值内存。chunking 不会改变前馈层对每个位置执行的映射，也不等于减少 attention 计算。可逆层针对深度方向的激活保存，chunking 针对前馈中间维度的峰值；它们与 LSH attention 优化的是不同资源项。

## 27.6 三条路线的机制差别与证据边界

下面把三条路线放到同一组问题下对照。假设远处某个位置 $j$ 含有回答问题 $i$ 所需的证据：Longformer 在相关位置被设为 global 时可以建立直接边；BigBird 的随机边可能连到该位置，但不保证每次覆盖；Reformer 要看两个表示的哈希和分块是否产生候选。即使图上存在边，模型也可能没有学会保留或使用该证据。

| 路线 | 候选连接由什么决定 | 主要减少或改变的对象 | 需要留意的代价与遗漏 |
| --- | --- | --- | --- |
| Longformer | 固定窗口、扩张规则和任务 global 位置 | attention 连接图 | global 位置依赖任务设定；窗口之外的细节可能缺少直接通路 |
| BigBird | local、随机和 global 连接 | attention 连接图 | 随机边不保证语义召回；表达性结论带有理论条件 |
| Reformer LSH | 当前表示的哈希、排序和分块 | attention 候选集合 | 哈希漏召回、排序/填充与多轮计算成本 |
| Reformer 可逆层与 chunking | 可逆 block 和按位置分块 | 训练激活存储路径 | 需要重构计算；不减少 attention 位置对 |

因此，渐近边数、理论表达性、论文任务分数和实际 kernel 延迟是不同证据。第十九章的 mask pattern 与第二十六章的执行块解释了连接如何落到计算上；跨路线比较质量、系统成本和迁移条件由[第二十八章](28-高效Attention为何未替代标准Attention.md)承担。阅读新的长序列方案时，先确认它改变的是连接、候选、状态还是激活存储，再判断它承诺的收益是否覆盖相应代价。
