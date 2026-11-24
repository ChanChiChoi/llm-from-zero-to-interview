# 第二十八章：高效 Attention 为什么没有完全替代标准 Attention

读完第 24—27 章，我们已经见过核特征、随机特征、稀疏连接和长序列方法。它们都可以被称作“高效 Attention”，但优化的对象并不相同。一个方法可能保留标准 attention，只改变数据如何搬运；另一个方法会减少 token 之间的连接；还有的方法会改写 attention 的数学表达，或把历史压缩成状态。

所以，“为什么没有完全替代”不能只靠一张复杂度表回答。首先要说清楚省下了什么、保留了什么，以及这项取舍是否符合任务所需的信息访问方式。

## 28.1 不同路线，优化不同账目

标准 dense attention 对每个 query 汇总所有允许访问的 key/value。序列越长，需要处理的 query-key 配对数按平方级增长。这个事实指出了计算规模，却没有说明执行时还要搬运多少数据、必须保留多少历史，或具体任务是否允许改变可见连接。

| 路线 | 主要减少的成本 | 保留或改变的对象 |
| --- | --- | --- |
| IO-aware dense attention | attention 中间结果的物化和显存往返 | 保留 dense 交互和标准 attention 的数学运算，配对计算仍随序列长度平方级增长 |
| 稀疏 attention | 被 mask 或选择过程排除的 query-key 配对 | 可见连接图改变；结果取决于保留下来的路径 |
| 线性/核 attention | 通过特征分解或有限特征压低随序列长度增长的计算与状态成本 | 计算对象依赖所选 kernel/特征；对标准 softmax 的输出可能只是近似 |
| KV 压缩 | 自回归解码时每个历史 token 占用的 cache 空间 | 主要压缩被保存的历史表示；attention 仍可以读取全部保留条目 |
| 可逆残差与前馈分块 | 训练时保存的激活或前馈中间结果 | 不改变 attention 候选集合，也不减少 dense 配对计算 |

这几类路线可以组合，也不是穷尽所有定义的唯一分类。近期的 attention 综述也把硬件实现、稀疏交互、KV 压缩和线性 attention 分作不同方向；这种分类帮助我们先确认“效率”指哪一笔成本，并不说明各方向互不相容。[综述分类](https://attention-survey.github.io/files/Attention_Survey.pdf)

例如，FlashAttention 原论文还描述了跳过全零块的 block-sparse 扩展。IO-aware 执行与稀疏连接可以叠加：前者减少数据搬运，后者减少实际计算的配对；组合后的效果仍由稀疏布局、kernel 和硬件共同决定。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

## 28.2 精确 dense attention：保留交互图，改变数据路径

FlashAttention 一类方法回答的是：“如果模型仍要执行 dense attention，能否少写一些中间数据、少在显存与片上存储之间来回搬运？”它把 query、key 和 value 切成块，在块内计算分数并更新 softmax 的运行统计量和部分输出，不必把完整 score 和 probability 矩阵写回显存。反向计算可以重新计算需要的中间分数，以计算换取更少的存储和数据读写。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)

这里的“精确”表示目标仍是同一个 dense attention 运算；改变计算顺序后，有限精度浮点数可能出现微小差别。因为每个允许的 query-key 配对仍需参与计算，序列变长时的平方级配对工作并没有消失。FlashAttention 改善的是执行和中间存储，不是把 dense attention 变成线性 attention。关于配对数和 online softmax 的基础口径，见[第 10 章](10-o-n2-attention的计算与显存瓶颈.md)；kernel、分段 decode 和分页 KV 见[第 20 章](20-FlashAttention与PagedAttention.md)。

FlashAttention-2 继续调整并行划分和线程工作分配，目标仍是提高同一 attention 运算在特定 GPU 上的执行效率。论文报告的速度取决于其 GPU、张量形状、模型和对照实现，不能直接当成任意硬件上的固定加速比。[FlashAttention-2 原论文](https://arxiv.org/abs/2307.08691) 第 10 章解释配对计算与 online softmax，第 20 章展开 kernel、分段 decode 和分页 KV；本节只用它们说明“同一数学运算可以有不同执行成本”。

因此，dense attention 并非只能在朴素实现和新架构之间二选一。改进它的实现，仍然是在使用标准 attention。

## 28.3 稀疏 attention：削减交互，依赖连接先验

稀疏路线直接减少一次 attention 中实际考虑的配对。局部窗口保留邻近 token；global token 为指定位置提供跨段通路；random 连接尝试补足远距离路径；块稀疏方法把 token mask 组织成可跳过的计算块；Reformer 则通过哈希和排序形成候选集合。[Longformer](https://arxiv.org/abs/2004.05150)、[BigBird](https://arxiv.org/abs/2007.14062)和[Reformer](https://arxiv.org/abs/2001.04451)原论文分别给出对应的连接与候选机制。[第 19 章](19-Sliding-Window与稀疏注意力.md)、[第 26 章](26-稀疏与块稀疏Attention.md)和[第 27 章](27-Longformer-BigBird-Reformer.md)进一步解释 pattern、块级布局和具体算法。

Reformer 还把另外两种训练内存技巧与 LSH attention 结合：可逆残差层通过重构前层激活减少保存量，前馈分块降低前馈中间结果的峰值；它们不减少 LSH 的候选 key，也不应算作少了 attention 配对。[第 27 章](27-Longformer-BigBird-Reformer.md)分别说明了这些资源项。

减少配对的代价是：每个 query 能读取哪些 key，不再与 dense attention 相同。对一个局部规律明显的任务，窗口可能保留了关键连接；对需要根据内容从任意远处挑出某段原文的任务，固定窗口未必覆盖那条证据路径。层与层之间可以传递信息，但这不能自动保证每个任务都能在需要的位置取得完整 dense attention 可用的信息。

“精确”也要说明参照对象。给定一张稀疏 mask，算法可以精确计算这张 mask 下的 softmax；但它算出的通常不是无稀疏限制时的 dense softmax。对于哈希候选等近似选择，选边过程本身还会带来候选遗漏的风险。Sparse 的复杂度收益要和任务所需连接、候选选择成本及硬件能否有效跳过对应起来看，不能单凭配对数较少推出质量不变或实际更快。

## 28.4 线性/核路线：分解 attention 并承担算子边界

线性 attention 的核心做法，是利用特征映射把 query-key 相似度写成可结合的形式，再把历史信息汇入有限维状态。因果生成时，状态随 token 更新，当前 query 从状态读取汇总结果。若特征宽度等维度固定，计算量随序列长度线性增长；[第 24 章](24-Linear-Attention基本思想与核技巧.md)给出具体分解与状态递推。

但“线性”并不自动等于“标准 softmax attention 的精确线性实现”。有的方法对自己选定的 kernel 做精确的代数重排；这个 kernel 本身可能不同于 softmax。[Transformers are RNNs](https://proceedings.mlr.press/v119/katharopoulos20a.html)给出有限特征下的线性计算形式。Performer/FAVOR+ 则用有限随机特征估计 softmax kernel，输出带有近似误差，其误差性质依赖特征构造、维度和论文假设；依据见[Performer 原论文](https://arxiv.org/abs/2009.14794v4)。[第 25 章](25-Performer与Linear-Transformer.md)进一步区分 kernel 估计与归一化 attention 输出的误差。

这类路线换来的不是单纯更快的同一计算，而是不同的计算表达和历史访问方式。压缩状态能保留任务有用的汇总信息，却不等于把每个历史 token 及其内容都原样留下，供任意未来 query 精确寻址。具体任务是否能接受这种表示，必须和模型质量一起判断。

## 28.5 替代判断要绑定任务、设备和保留的能力

先判断瓶颈在哪里，才能谈哪条路线合适。如果主要问题是 dense attention 的中间矩阵和数据搬运，而任务仍需要完整的交互集合，IO-aware kernel 可能在保留运算语义的同时降低执行成本。如果任务关系主要落在局部或少数预定路径，稀疏结构值得研究，但要确认这些路径不会排除关键依赖。如果输入是持续流，且有限状态足以支持目标任务，线性或状态路线可能更合适；若 decode 的主要压力是 KV cache 容量，压缩 cache 处理的是存储问题，未必需要替换 attention 的配对计算。

实际表现还依赖序列长度、batch、硬件、kernel 和具体实现。减少算术可能让数据访问、索引、状态更新或其他固定开销变得更显著；某个短序列 benchmark 上的结果不能代表长序列或在线生成的结论。公平比较必须在明确的任务和质量要求下，分别说明算法改变了什么、测量覆盖什么，不把不同论文中的速度数字拼成统一排名。

多种路线也可以并存：保留部分 dense attention，同时在其他位置使用局部连接或递归状态；也可以先改进 dense kernel，再判断是否还需要改变模型交互图。下一章[第 29 章](29-SSM数学直觉.md)转向 SSM，讨论一种明确以递归状态更新为核心的序列模型。它与 attention 有可比较的长序列目标，却不是一种更快实现了的标准 attention。要把 SSM、RNN、CNN 与 attention 放在同一信息流框架下比较，见[第 30 章](30-SSM-RNN-CNN-Attention信息流对比.md)；本章不重复那一层跨架构分析。服务与推理引擎细节回到[第六册](../../book-06-llm-deployment/目录.md)和[第二十四册](../../book-24-llm-inference-engine/目录.md)。跨架构横向比较和综合机制由[第 42 章](42-替代架构横向对比.md)承担。

“高效路线没有完全替代标准 attention”更准确地说，是不同路线解决了不同成本问题，并在语义、访问能力和实现条件上作出了不同取舍。要判断替代是否成立，必须说明目标任务需要保留什么、设备的实际瓶颈是什么，以及证据在哪些条件下测得；复杂度符号本身不足以给出通用答案。
