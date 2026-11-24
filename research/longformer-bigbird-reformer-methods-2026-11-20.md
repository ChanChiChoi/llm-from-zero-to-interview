# Longformer、BigBird 与 Reformer 方法机制研究记录

研究日期：2026-11-20。范围为第二十一册第二十七章的职责、方法机制与正文事实核验。读者已理解 attention mask、query-key 连接图和标准 Transformer block。本章需区分三类长序列路线：任务指定的全局连接、固定随机稀疏图，以及由向量哈希生成的近似候选；Reformer 还包括与 attention 稀疏化不同的训练激活内存优化。

## 检索范围与教学材料

- 英文检索词：`Longformer local global attention paper explained`, `BigBird random window global attention theoretical limitations`, `Reformer LSH attention reversible residual chunking`, `Longformer BigBird Reformer lecture notes`。
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)，Beltagy、Peters、Cohan，arXiv 2020；阅读摘要、§3.1—3.2、§4.2.2、长文档任务和 LED 介绍。来源页：论文 HTML。
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)，Zaheer 等，NeurIPS 2020；阅读摘要、§2、§3.2—3.4、Appendix D 实现部分。来源页：arXiv HTML 与 NeurIPS 论文页面。
- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)，Kitaev、Kaiser、Levskaya，ICLR 2020；阅读摘要、§2 LSH attention、§3 reversible Transformer 与 chunking、§5 实验及附录中的多轮 hash 说明。来源页：arXiv HTML。
- 作者/团队解释：[Google Research 的 BigBird 方法说明](https://research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/)与[Reformer 作者说明](https://research.google/blog/reformer-the-efficient-transformer/?m=1)。用于检视 random/window/global 的角色、blockify 思路以及 Reformer 的排序分桶和可逆层直觉，不以博客替代论文边界。
- 教学组织材料：[MIT 6.7960 Deep Learning, Lecture 10: Memory and Sequence Modeling](https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/mit6_7960_f24_lec10.pdf)，2024 年秋季，局部/全局 attention pattern 图示；Kripner 的[Longformer 论文 notes](https://kripner.com/posts/longformer/)，2023-11-23，窗口、扩张、global 位置与实现演进。两者用于比较如何呈现连接图和方法动机，不支持超出其内容的学习效果或性能结论。

## 机制发现

### Longformer

- 局部窗口使每个位置查看固定邻域，固定窗口宽度时 attention pair 数按序列长度近似线性增长；扩张窗口用间隔扩大感受野，但跳过的邻近位置不再有直接边。不同 head/layer 可承担局部与远距范围。
- 任务指定的少量 global 位置在原论文双向长文档设置中既读取全序列，也被其他位置读取；分类可选 `[CLS]`，问答可选问题 token。局部与 global attention 使用两组 Q/K/V 投影。若窗口宽度和 global 位置数固定，连接数量为 $O(nw+ng)$。
- 原论文另提出 LED 用于长文档 encoder-decoder 摘要任务。其具体结论属于论文中的模型、数据集和训练设置；global 选择依赖任务，不代表这些位置自动成为语义检索器。因果生成使用时仍须按 causal mask 解释，而不能把双向 encoder 的全局规则直接照搬。
- 论文指出稀疏 banded 计算需匹配实现；其 chunk/CUDA 实测是指定实现和硬件结果，不能由线性 pair count 推出当前任意 GPU 上的加速。

### BigBird

- BigBird 把局部、随机和 global 连接组合。局部保持邻近上下文，random 边补充跨区连接，global 位置提供汇聚/广播通道；random 连接不是按语义相似度搜索出来的候选。
- 论文区分 ITC（让输入中的已有位置成为 global）和 ETC（增添 global 位置）。若每位置的局部宽度 $w$、随机连接数 $r$ 与 global 数 $g$ 固定，连接数级别为 $O(n(w+r+g))$；global 行/列也计入该数量级。
- 作者把稀疏图 blockify，并用连续块张量、reshape/roll 和有限 gather 组织注意力计算，以便利用 GPU/TPU 的稠密块算子；这属于论文的特定执行组织，通用块映射边界另见第二十六章。
- 论文给出 encoder 的 universal approximation 与特定假设下 encoder-decoder Turing completeness 结论；同一理论部分还用一个最远向量检索任务给出限制：在其标准复杂度假设和稀疏边数条件下，某些任务可能要求远多于 dense attention 的层数。表达性定理不等价于任意 checkpoint 的任务质量保证，实验对照也只支持论文任务与条件。

### Reformer

- Reformer 的 LSH attention 以共享 Q/K 表示，经随机投影 hash 把相似向量较可能放入同一 bucket；之后按 bucket 和原序列位置排序，分 chunk 做 attention，并纳入相邻 chunk 以覆盖边界溢出。因果设置须把原位置随排序一起搬移并据此施加 mask。
- 多轮 hash 将不同轮的候选并集扩大，降低一次碰撞遗漏的概率，也增加注意力工作量。attention 只在 hash/chunk 产生的候选范围内归一化；这近似 full attention 的可见集合，但不是对 dense key 集合的精确等价。论文给出 $O(L\log L)$ attention 复杂度主张，适用其 bucket/chunk 和 hash-round 设置；哈希、排序、填充和硬件效率仍需独立考虑。
- Reformer 的另两项机制不减少 attention pair：reversible residual block 用可逆耦合关系从后层输出重构前层激活，降低随层数保存所有 activations 的需求；position-wise feed-forward 可按序列位置 chunk 处理，降低宽中间张量的峰值内存。两者改变训练内存/重算路径，与 LSH 的连接稀疏是不同优化对象。
- 原论文在复制序列合成任务中观察到多轮数会影响精确重复召回；其文本与图像实验使用论文指定配置。这说明 hash 轮数和任务误差相关，不是所有数据集上稳定的 quality-vs-cost 曲线。

## 方法关系与证据边界

| 路线 | 候选连接如何产生 | 直接改变什么 | 主要新增代价或风险 |
| --- | --- | --- | --- |
| Longformer | 固定邻域与任务指定 global 位置 | 稀疏连接图 | global 位置依赖任务/标注，窗口外位置缺少直接边 |
| BigBird | 固定 local/global 加随机边 | 稀疏连接图 | 随机可达不等于语义召回；理论结论带设定，需适配块执行 |
| Reformer LSH | 基于当前 Q/K 表示的随机 hash、排序和分块 | 近似 attention 候选集合 | hash 漏召回、排序/填充和多轮计算 |
| Reformer reversible/chunking | 可逆层结构与位置分块 | 训练激活存储路径 | 反向重算与实现约束；不减少 attention 的有效候选 |

三者的稀疏复杂度表达不相互替代：固定预算下 Longformer/BigBird 的边数可近似随序列长度线性增长；Reformer 论文的 $O(L\log L)$ 依赖其 LSH 组织和 hash 轮数。任何这些渐近式都不包含或不能直接推出实际 kernel 延迟、任务证据使用质量和系统收益。相关 pattern 定义由第十九章承担，通用 block layout 与动态选块成本由第二十六章承担，跨路线系统取舍由第二十八章承担。
