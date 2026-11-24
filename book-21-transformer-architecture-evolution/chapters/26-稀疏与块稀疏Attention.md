# 第二十六章：稀疏 Attention、局部 Attention 与块稀疏 Attention

第十九章已经说明稀疏 pattern 改变了哪些 query-key 位置可以交互。接下来的问题是：mask 里有很多零，GPU 就一定跳过了这些计算吗？答案取决于 mask 怎样映射到 kernel 执行的块，以及选择这些块本身要花多少工作。

因此，本章把三个对象分开：逻辑连接图规定哪些 token 可以互相读取；block layout 把位置分组为规则计算 tile；kernel 决定实际加载并执行哪些 tile。三者对齐时，稀疏才可能从结构选择变成较少的硬件工作。

## 26.1 从 token mask 到可跳过的计算块

设 $M_{ij}=1$ 表示 query 位置 $i$ 可以读取 key 位置 $j$，$M_{ij}=0$ 表示这条边被 mask。硬件执行时，query 和 key 通常被分成块。记第 $u$ 个 query 块包含的位置为 $Q_u$，第 $v$ 个 key 块包含的位置为 $K_v$。对应的块布局可以写成：

```math
B_{uv}=
\begin{cases}
1,&\exists i\in Q_u,\;j\in K_v\text{ such that }M_{ij}=1,\\
0,&\text{otherwise}.
\end{cases}
```

若 $B_{uv}=0$，整个块对都没有有效连接，可以跳过；若 $B_{uv}=1$，kernel 至少要处理这个块对，并在块内继续遵守原来的 token mask。比如一个 $4\times4$ 的块对里只有两个 token pair 可见，逻辑上只有两条边，但块稀疏 kernel 可能仍以一个 $4\times4$ tile 为单位加载和计算，再屏蔽其余位置。

这就是“稀疏 mask”与“稀疏执行”的差别。PyTorch 的 `BlockMask` 文档也将一个块定义为只有在内部所有位置都被 mask 时才能整体标为稀疏；这种表示便于连续加载和块级计算，但不保证最小的索引存储。[PyTorch FlexAttention BlockMask 文档](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html#blockmask)

## 26.2 Block size：索引、块内多算与硬件粒度

块大小同时影响算法表达和执行效率。块较大时，同样的序列只需维护较少的块索引，矩阵乘也更规整；代价是块内少数有效连接可能让整块一起执行。块较小时，布局能更精确地贴近逻辑 mask，但索引更多，工作也更零散，管理与调度开销可能上升。

需要区分三个数量：原始 mask 中保留的 token pair、布局中选中的块对、kernel 实际执行的矩阵 tile。它们通常不相等。因果对角块、序列尾部的不完整块和 padding 还会改变块内有效位置；某个 kernel 能否在 tile 内进一步省掉无效运算，取决于它支持的 mask 和执行方式，不能从块稀疏这一名称直接推出。

固定一个 block size 也未必适合所有 head。AB-Sparse 的 2026 年预印本报告，不同 attention head 对块粒度的敏感度存在差异，并据此研究按 head 调整块大小及相应 kernel。[AB-Sparse 预印本](https://arxiv.org/abs/2605.12110v1) 这说明块大小既是硬件参数，也会影响哪些 token 被合并进同一块；论文中的改进幅度仍受其模型、任务和实现设置约束。

## 26.3 固定块布局与输入相关 block router

固定布局由规则或配置预先确定，例如每个 query 块读取对角线附近的 key 块。它不需要针对每个输入重新挑选候选，但可能漏掉规则之外的重要证据。动态布局则增加一个 router：router 根据当前 query、key 或较便宜的摘要为候选块打分，形成索引，再由主 attention 读取被选中的原始 K/V。

MInference 展示了一种分开的流程：先离线为不同 attention head 选择 pattern，再针对当前输入在线生成稀疏索引，最后运行相应的稀疏 kernel。对 block-sparse head，论文用聚合后的 query/key 表示估计哪些块更重要；这个估计用于选块，主 attention 仍在选中块的原始 token 上计算。[MInference 论文](https://arxiv.org/abs/2407.02490)

稀疏性也可以在训练中形成。Native Sparse Attention（NSA）把层级压缩、选择和局部路径组合为可训练的稀疏结构，并围绕硬件执行设计计算。[NSA，ACL 2025](https://aclanthology.org/2025.acl-long.1126/) 这和给一个已有 dense 模型临时加上推理时路由，不是同一训练契约。第十九章负责 pattern 的信息路径概念；第八十九章负责 GLM-5/GLM-5.3 DSA 的配置与模型实现，本章只解释 block 路由的通用结构。

## 26.4 Router 的成本与稀疏输出的边界

Top-K 只说明主 attention 最终处理多少候选块，不说明这些候选怎样找到。如果每个 query 都给全部 key blocks 打分，路由仍可能承担很大成本。PISA 的 2026 年预印本把这一点作为问题，提出从粗到细的层级筛选，并在其固定选择预算等条件下声称选择成本为 $O(N\log N)$。[PISA 预印本 v1](https://arxiv.org/abs/2609.31093v1) 这是一个研究方案，不代表所有 router 都具有这一复杂度；每种方法都要把选择、索引构建和后续 attention 分开核算。

对一个 query，设 router 选出的、同时满足当前 mask 的 key 集合为 $S_i$。主 attention 在这个集合内计算：

```math
o_i^{(S)}=
\sum_{j\in S_i}
\frac{\exp(q_i^\top k_j/\sqrt{d_k})}
{\sum_{r\in S_i}\exp(q_i^\top k_r/\sqrt{d_k})}
v_j.
```

只要 kernel 正确执行，这个结果就是对 $S_i$ 内部 attention 的精确计算；但如果 $S_i$ 排除了原本可见的 key，它就不等于 dense attention。遗漏的 key 不仅少贡献一个 value，也不再参与 softmax 分母。router 的候选召回、选中集合上的 attention 误差和最终任务质量因此是三个相关但不能互相替代的量。

## 26.5 训练与执行必须共享同一稀疏契约

一个 block layout 还要保存 query 块到 key/value 块的对应关系。执行时需保持 causal 方向、序列边界、padding、变长样本和块内 mask 一致；不完整尾块只能让有效 token 参加 attention。若不同 head、batch 或 query block 使用不同路由，索引的归属也必须与实际 Q/K/V 一致。错误的映射可能让本应屏蔽的位置进入 attention，或让有效证据被错误裁掉。

训练和推理使用的稀疏策略也可能不同。训练时学习 router 的方法必须说明离散选择如何获得训练信号；仅在推理时插入候选选择，则需要承担分布变化和漏选风险。以模型名、`top_k` 字段或通用 API 名称无法判断完整行为。当前 cuDNN 的 BSA 文档把其操作定义为非因果 attention，并标为实验性接口，正说明具体库对 causal mask、布局和后端支持有自己的合同；这些能力必须按目标版本核对。[cuDNN Block Sparse Attention 文档](https://docs.nvidia.com/cudnn/operations/sparse_attention/bsa)

Prefill 与 decode 的工作形状也不同：prefill 同时有许多 query 位置，decode 往往只有少量新 query 要访问长历史。相同的选块预算和索引方式未必在两阶段具有相同成本。第二十章进一步讲 kernel 的 tile/IO 与 decode 并行；服务压测、请求调度和线上回退由第六册及第二十四册承担。

## 26.6 判断一次稀疏化真正改变了什么

阅读或设计稀疏方案时，可以分四层问清楚：

| 层次 | 要确认的问题 | 不能直接替代它的数字 |
| --- | --- | --- |
| 连接图 | 哪些 token pair 仍可见？ | mask 中的零元素比例 |
| 路由 | 候选块怎样产生，扫描了多少位置？ | 最终保留的 Top-K 数 |
| 执行 | kernel 跳过了哪些 tile，块内还计算什么？ | 理论非零 FLOPs |
| 结果 | 关键证据是否保留，质量和延迟在哪些条件下测得？ | attention 稀疏率或单个 kernel 数字 |

因此，块稀疏的收益需要结构、router 和 kernel 三者对上：选中的块必须符合任务需要；找块的成本不能吞掉省下的计算；目标 kernel 需要真正跳过未选 tile。经典窗口、global/random 与 Longformer/BigBird/Reformer 的信息图分别见[第十九章](19-Sliding-Window与稀疏注意力.md)和[第二十七章](27-Longformer-BigBird-Reformer.md)；dense kernel 和 KV 地址映射见[第二十章](20-FlashAttention与PagedAttention.md)；跨路线的整体取舍见[第二十八章](28-高效Attention为何未替代标准Attention.md)。
