# 第二十七章：Longformer、BigBird、Reformer 等长序列路线

## 章节契约

- **定位：** 以 Longformer、BigBird 和 Reformer 三个历史方法为案例，比较固定局部/全局连接、随机稀疏图和内容相关哈希候选如何在长序列中重新安排 attention。章节同时说明 Reformer 的可逆残差与前馈分块解决的是训练激活内存，而非 attention 连接数。
- **前置：** 读者理解标准 self-attention、局部窗口、global/random pattern、causal mask 和 token 连接图；第十九章承担 pattern 与多层信息路径，第十章承担 attention pair 复杂度，第二十六章承担通用 block layout、路由与执行边界。
- **离章结果：** 读者能解释 Longformer 的任务 global 位置、BigBird 的 random 边和 Reformer 的 LSH bucket 分别怎样形成候选；描述 Reformer 的排序/分块、多轮 hash、因果 mask、可逆 residual 及 FFN chunking；区分三者的理论复杂度主张、信息路径和新代价，不把论文设定等同于通用任务质量或目标硬件速度。
- **边界：** 第十九章主讲一般 attention pattern 与信息路径；第二十六章主讲 token mask 到执行块及动态块选择；第二十八章比较高效路线的总体质量、系统成本与迁移条件。这里保留方法内部的专属机制和论文证据，不重讲通用 benchmark、部署、回退或线上评估流程。
- **结构研究依据：** 旧稿诊断与取舍见[第二十七章精修审计](../../../progress/book-21-transformer-architecture-evolution/chapter-27-audit.md)；方法来源、版本和证据边界见[研究记录](../../../research/longformer-bigbird-reformer-methods-2026-11-20.md)。

## 主题分配

| 主题 | 读者需要获得的认识 | 完整主讲位置 |
| --- | --- | --- |
| 三种长序列选边路线 | task-global、random graph、LSH 分别以规则、随机性或当前表示生成可见候选；Reformer 还有独立的激活内存优化 | 27.1 |
| Longformer | 局部/扩张窗口建模邻近和跨层感受野，少量任务指定 global 位置提供全序列读写；encoder 的双向规则不能直接作为 causal decoder 规则 | 27.2；基础 pattern 回链第十九章 |
| BigBird | 局部保邻近、random 提供非语义的跨区边、global 位置作为汇聚/广播节点；理论表达性依赖假设，也存在稀疏图下界 | 27.3；一般执行粒度回链第二十六章 |
| Reformer LSH | 共享 Q/K 表示经随机投影分桶、排序并分块，候选范围上的注意力是 full attention 的近似；多轮 hash 增召回也增加成本 | 27.4 |
| 可逆残差和 FFN 分块 | 从后层状态重构前层激活、按位置分块计算前馈层，分别减少深度存储与宽中间激活；不改变 attention 的候选关系 | 27.5 |
| 方法比较与证据范围 | 对齐谁选边、改变何种计算/存储对象、误差从何处来，以及渐近复杂度不能推出硬件收益 | 27.6；总体取舍回链第二十八章 |

## 正文编号大纲

### 27.1 三种长序列路线分别怎样选择连接

本节界定本章比较对象：Longformer 以局部窗口和任务指定位置构造图，BigBird 在局部和全局之外加入与内容无关的随机连接，Reformer 依据当前 query/key 表示的哈希关系形成近似候选。三者都减少或限制 token 交互，但选择规则、需要保留的结构信息及新增成本不同；Reformer 还用可逆残差和前馈分块处理训练激活内存。本节与第十九章的通用 pattern 定义和第二十六章的通用块映射保持边界。

### 27.2 Longformer：局部窗口、扩张窗口与任务 global

本节说明局部窗口怎样保留近邻 token 交互，扩张间隔怎样以相近连接预算扩大跨层覆盖并跳过部分直接邻居，以及少量任务 global 位置怎样在原论文双向长文档设置中读取和广播全序列信息。涉及分类 `[CLS]`、问答问题 token、global 路径采用独立 Q/K/V 投影、固定窗口和 global 数下的线性连接量，以及面向长文档生成的 LED。论文本身的任务和实现结果绑定其训练、数据和硬件条件；因果 decoder 需另按 causal 规则理解。

### 27.3 BigBird：local、random 与 global 的稀疏图

本节围绕三类边的分工组织 BigBird：local 连接邻近位置，random 边提供跨区机会，global 节点允许特定表示与全序列双向交互。还涉及把已有 token 指定为 global（ITC）或添加 global token（ETC），固定连接预算下的边数规模，以及论文通过 blockify 和块张量运算适配硬件的实现思路。论文的 universal approximation 与 Turing completeness 结论针对明确理论设定，同时给出稀疏图可能需要更多层的下界；这些结论不等于具体模型任务质量或实际加速保证。

### 27.4 Reformer：LSH 候选、排序分块与近似 attention

本节说明 Reformer 共享 query/key 表示并以随机投影哈希把相似向量较可能分到同一 bucket；随后按 bucket 和序列位置排序，切成块并纳入邻近块处理边界容量，因果设置保留原位置以继续屏蔽未来 token。多个独立 hash round 取并集以减少漏候选，同时扩大计算量。Attention 只在得到的候选集合中归一化，因此不是对完整 dense 可见集合的精确等价。论文的 $O(L\log L)$ 结论与 bucket/chunk/hash 设置绑定，还需区分哈希、排序、padding 和具体实现的成本。

### 27.5 Reformer 的可逆 residual 与前馈分块

本节说明 attention 稀疏化之外的两个激活内存机制。可逆 residual block 通过两个相互耦合的状态更新，使反向传播可由后层状态和参数逐层重构前层状态，降低按层保存 activations 的需要；前馈网络对位置独立，可沿序列分 chunk 计算并限制宽中间张量的峰值内存。两者不减少 attention 的 token 对或候选范围；重构带来计算与实现上的取舍。

### 27.6 三条路线的机制差别与证据边界

本节比较 task-global 规则、random graph、LSH 候选及 reversible/chunked activation path 各自改变的对象，说明主要遗漏风险来自 global 位置选择、随机可达性或 hash 漏召回，而计算成本还包括各自的布局、排序和重算。渐近 pair 数、理论表达性、论文任务指标与目标 GPU 的吞吐属于不同证据。本节以三种方法适合回答的架构问题收束，并将跨路线成本、任务质量及迁移判断交给第二十八章。
