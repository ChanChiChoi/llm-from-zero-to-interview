# 第十九章：Sliding Window Attention、Global Attention 与稀疏注意力

## 章节契约

- **定位：** 承接第十八章的位置范围与长上下文适配，解释 attention pattern 如何限制或扩展 query 对历史 token 的直接访问，并比较固定稀疏连接与内容选择式稀疏注意力。
- **前置：** 读者已理解因果 mask、自注意力的 Q/K/V 交互和 attention pair 随序列长度增长的基本代价；第八章负责可见性与 causal mask，第十章负责 dense attention 的计算/显存瓶颈。窗口宽度、位置编号和可见方向在本章按需重新定义。
- **离章结果：** 读者能读出 causal sliding window、dilated、global、local/random/global 与 query-conditioned selection 各自允许哪些 token 交互；能解释逐层局部连接何时提供窗口外的信息路径、何时并不保证证据被保留；能区分 pattern 的理论 pair 数、任务级有效上下文和目标硬件上的实际执行收益。
- **边界：** 第十章主讲 attention pair 和 dense 计算/显存，第十一章主讲 KV cache 容量，第十三章主讲长距离证据使用与行为评测，第十五章主讲流式/在线边界，第十八章主讲位置映射和长度适配，第二十章主讲 FlashAttention/PagedAttention/kernel。第二十六章承担块级稀疏布局、动态路由和实现验证；第八十九章提供 GLM-5/DSA 的模型级案例。本章不重复这些专题的实现细节。
- **设计依据：** 对照第二十一册架构审计、本章旧稿及第八、十、十一、十三、十五、十八、二十、二十六、八十九章职责；外部主题、来源和证据边界见[本章研究记录](../../../research/sliding-window-and-sparse-attention-patterns-2026-11-20.md)。

## 研究主题与章节去向

| 主题 | 读者需要获得的认识 | 本章处理 |
| --- | --- | --- |
| Attention pattern 与稀疏连接 | 稀疏 attention 指定哪些 query-key pair 可见；不同 mask 产生不同的信息图，稀疏不自动代表线性复杂度或硬件加速 | 19.1 建立 pattern、可见性与 pair 数的关系，并区分稀疏结构和精确 kernel 重排 |
| Causal sliding window | 窗口只限制单层直接可见的历史；跨层 hidden representation 可以传递较早信息，但路径存在不等于任务读取成功 | 19.2 明确窗口计数约定、边界和层间传播；将行为能力交给第十三章、cache 容量交给第十一章 |
| Dilated / structured patterns | 稀疏连接可以以规则间隔覆盖更远位置；接触范围增大时也可能留下局部空洞；不同结构具有不同渐近复杂度 | 19.3 对照 dilated window 与 Sparse Transformer 的结构化分解；不把所有 sparse 方法概括成线性 |
| Task-global connections | 少量指定位置可形成全序列汇聚/广播路径；Longformer 的 global 设计针对双向文档任务，decoder 必须另外遵守因果方向 | 19.4 解释 Longformer 的局部与 global 连接及固定 global 数时的复杂度边界 |
| Local + random + global | BigBird 通过三种连接组合改善理论图路径；理论表达性不等于特定深度/训练下的任务质量，也不说明硬件效率 | 19.5 说明三类边的职责和论文主张适用范围 |
| 内容选择式稀疏 | query-conditioned indexer 与固定 mask 不同，先估计候选位置，再由主 attention 读取；选择器本身也有计算和状态成本 | 19.6 以 DSA 作短例，完整训练、模型配置和 kernel 留给第二十六、八十九章 |
| 相邻技术边界 | KV retention、位置扩展、外部检索、kernel 优化分别改变不同对象 | 19.6 简短划清 attention 图、cache、位置、RAG 与 kernel 的问题边界，并链接第十一、十三、十五、十八、二十、二十六章 |

## 正文编号大纲

### 19.1 从全连接到稀疏连接图

本节围绕“稀疏 attention 改变什么”建立共同坐标：dense causal attention 让当前位置直接访问全部先前 token，sparse pattern 则排除一部分 query-key pair。模式可以固定、规则，也可以根据 query 内容选择；由此改变的是可见关系和每层信息图。说明固定 pattern 仅在其连接度等条件成立时才得到相应 pair-count 复杂度；它不同于 FlashAttention 对完整 attention 的分块/IO 重排，也不同于通过核近似或状态重写来改变计算形式的线性 attention，后两者分别回链第二十、二十四至二十八章。

### 19.2 Causal Sliding Window：直接读取范围与跨层传播

本节解释因果窗口中当前位置最多读取多少个近期 token，并明确窗口宽度按“包含当前位置在内的可见 token 数”定义。主要知识包括集合边界、窗口刚好满时最早保留位置、固定窗口下 pair 数随序列长度的关系，以及 Mistral 7B v0.1 作为模型案例。还需区分单层直接可见范围与多层表示传播：上一层邻近位置已汇入的内容可能继续传递到更晚 query，但长链能存在不表示原子事实仍被保留、模型会使用它，或任务级有效长度因此扩大。attention sink 仅用于指出训练窗口结构与推理时 KV 保留策略不同；后者由第十一、十五章承担。

### 19.3 Dilated 与结构化稀疏：扩大跨度会改变哪些连接

本节说明 dilation 怎样在同一层的局部连接间隔中加入空隙，以相近的单层连接预算触及更远位置；它与直接增大连续窗口的差别在于远程跨度扩大而部分邻近位置可能不再直接可见。随后用 Sparse Transformer 的结构化 attention factorization 提醒读者：稀疏可以来自多种图形和分解，论文中的 $O(n\sqrt{n})$ 与固定窗口的 $O(nW)$ 是不同构造的结果，不能只凭“sparse”一词推断复杂度。block-level mask 的布局与执行细节留给第二十六章。

### 19.4 Global Attention：为少量位置增加全序列路径

本节以 Longformer 解释 task-motivated global connection 如何和 local window 合用：例如分类任务的 `[CLS]`、问答任务中的 question positions；在原论文的双向设置中，被指定的 global 位置可读全序列，其他 token 也可读这些 global 位置。若全局位置数量固定，论文 pattern 的 pair 数仍随序列长度线性增长。必须将这种 encoder/document 规则与 causal decoder 分开：decoder 的任何全局路径仍须禁止未来信息。global token 缩短信息路径，但自身只提供有限表示通道，不能被描述为无损恢复 dense all-to-all。

### 19.5 BigBird：把局部、随机与全局连接组合起来

本节说明 BigBird 的 local、random、global 三类连接分别提供近邻交互、分散的跨区路径和全序列汇聚/读取。在固定连接数等论文条件下，该组合形成线性数量级的 sparse pair，并支持原论文中的表达性论证。理论上存在表示路径不等于给定模型在有限层数、宽度、训练数据和目标任务上达到 dense 模型质量；random edges 的图连通动机也不证明其在 GPU 上更快。与 Longformer 对照时关注连接类型及其任务角色，不做无条件性能排名。

### 19.6 从固定 mask 到内容选择：动态稀疏与适用边界

本节把固定连接图与 query-conditioned selection 区分开。以 DeepSeek Sparse Attention 为例，轻量 indexer 为历史候选打分，选出的 top-k KV entries 进入主 attention；筛选改变查询实际读取的位置，但 indexer 扫描历史、候选存储与主 attention 读取属于不同成本。只说明两阶段架构关系，不展开 DSA 训练损失、层间复用、配置字段或 kernel 实现，详见第八十九章和第二十六章。

结尾归纳 pattern 选择的判断维度：任务依赖更偏近邻、需要少量 task-global 位置、需要分散远程连接，还是需要内容相关的候选选择；连接可达性回答“有无路径”，任务评测回答“证据是否被用到”，实现 profile 回答“目标硬件是否执行得更快”。RAG 改变的是送入模型的候选文档集合，不是模型内部 attention mask；位置扩展由第十八章、长距离能力评测由第十三章负责，实际稀疏 kernel 与 serving 回退由第二十、二十六章负责。
