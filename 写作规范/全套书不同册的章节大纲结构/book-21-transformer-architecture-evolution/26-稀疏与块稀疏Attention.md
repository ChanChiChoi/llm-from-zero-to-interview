# 第二十六章：稀疏 Attention、局部 Attention 与块稀疏 Attention

## 章节契约

- **定位：** 承接第十九章的 token 级注意力连接图和第二十章的精确 attention kernel，解释逻辑稀疏怎样表示为可跳过的计算块，以及输入相关路由怎样生成这些块。章节中心是表示、选择和执行之间的关系，不重讲经典稀疏 pattern 的分类。
- **前置：** 读者理解 causal mask、Q/K/V attention、softmax 以及矩阵 tile/kernel 的基本含义；第十九章负责窗口、dilation、global/random 与 query-conditioned pattern，第十章负责 attention pair 计算，第十八章负责长上下文位置映射。
- **离章结果：** 读者能区分 token 级逻辑稀疏、block 级布局和 kernel 实际跳过的工作；解释 block size 对索引开销、块内多算和硬件利用率的共同影响；区分静态块布局、输入相关 block router、router 与主 attention 的工作和误差；说明选定集合上的 attention 与 dense attention 的语义差别，以及训练/执行 mask 契约为什么必须对齐。
- **边界：** 第十九章主讲 pattern 允许谁访问谁和跨层信息路径；第二十章主讲 dense FlashAttention、decode 并行和 PagedAttention 地址映射；第二十七章主讲 Longformer、BigBird、Reformer 等方法案例；第二十八章综合比较高效路线与 dense attention；第八十九章承接 GLM-5/GLM-5.3 DSA 配置与模型案例；第六册与第二十四册承接 serving benchmark、kernel profiling 和线上回退流程。
- **结构研究依据：** 旧稿诊断、内容去向和重组决定记录于[第二十六章精修审计](../../../progress/book-21-transformer-architecture-evolution/chapter-26-audit.md)；来源、版本和证据范围见[块稀疏 Attention 布局与动态路由研究记录](../../../research/block-sparse-attention-layout-and-routing-2026-11-20.md)。

## 主题分配

| 主题 | 读者需要获得的认识 | 完整主讲位置 |
| --- | --- | --- |
| 逻辑稀疏与计算 tile | mask 排除了单个 token pair，不代表 kernel 跳过了承载它的 tile；可跳过的 block 必须按布局规则确认 | 26.1；mask 拓扑回链第十九章 |
| 块大小与有效工作量 | 粗粒度 block 提升规则性但增加块内多算，细粒度 block 减少填充却扩大 metadata 和调度负担；不同 head 也可能有不同粒度需求 | 26.2 |
| 静态布局与动态路由 | 固定 block pattern 直接给出索引；动态方法先估计或学习候选，再执行选中 block 上的主 attention | 26.3；经典 pattern 回链第十九、二十七章，模型实现回链第八十九章 |
| 路由成本和输出近似 | router 即使缩小后续 attention，也可能扫描大量候选；在候选集内精确计算不等价于对全部 key 的 dense attention | 26.4 |
| 训练与执行契约 | causality、padding、尾块、head/batch 映射及 router 版本共同决定计算图和可复现性 | 26.5；具体 kernel/runtime 验收回链第二十章及第六册/第二十四册 |
| 适用边界 | 稀疏声明要分别回答删了哪些边、选择器付出什么、kernel 跳过了什么工作，以及质量证据支持什么 | 26.6；跨路线取舍回链第二十八章 |

## 正文编号大纲

### 26.1 从 token mask 到可跳过的计算块

本节说明逻辑可见集合与 kernel 执行粒度不是同一个对象。token 级 mask 对每个 query-key pair 指定能否交互；block-sparse 布局则把 query 和 key 分组成 tile，并为每个 query block 记录需要读取的 key/value blocks。只有整个 tile 都不含有效连接时，kernel 才能整块跳过；被选中的 tile 若包含块内 mask 位置，仍要在块内保持相应的 causal、padding 或其他可见性约束。读者应由此区分“mask 中有很多零”和“硬件没有计算这些区域”。

### 26.2 Block size：索引、填充与硬件粒度

本节围绕 block size 的双向影响组织知识：增大块会减少块索引并让矩阵乘更规整，但逻辑上不需要的 token pair 可能随块一起执行；缩小块提高可见集合的表达精度，却增加索引数量、调度和访存的不规则性。块内有效 pair、实际选中块数、尾部不完整块、causal 对角块和不同 head 的需求共同决定有效计算，理论 token 稀疏率不能单独代表实际执行稀疏率。近期自适应粒度工作只作为异质性例证，其效果仍绑定论文条件。

### 26.3 固定块布局与输入相关 block router

本节区分预先给定的 block layout 与根据输入内容生成的候选块。固定布局由规则或配置决定哪些块保留；动态路由对 query、key 或其低成本摘要估计候选相关性，产出索引后由主 attention 在原始 K/V 上计算。MInference 的 block-sparse 路线说明了“离线决定 head 的模式、在线为当前输入选块、kernel 执行选中块”的分工；NSA 说明稀疏路径也可以作为可训练架构的一部分。两类例子不构成方法优劣排名；第十九章承担 pattern 概念，第八十九章承担 GLM-5 的 DSA 配置和运行实现。

### 26.4 Router 的成本与稀疏输出的边界

本节解释 Top-K 候选数只描述主 attention 的输入规模，不能代替完整成本。若每个 query 都扫描全部 key blocks，路由本身仍可能随上下文长度呈二次增长；分层候选筛选是针对这一瓶颈的研究路线，但其复杂度和收益要绑定具体算法、实现及实验设置。所选 blocks 内可用标准 attention 精确计算并统一归一化，然而被遗漏的 key 不再进入 softmax 分母和 value 汇总，因此输出一般是对 dense attention 的近似。候选召回、attention 输出误差和最终任务表现是相关但不同的证据。

### 26.5 训练与执行必须共享同一稀疏契约

本节统合 block selection 的正确性条件：每个 query block 对应的 key block 列表必须遵守因果方向、padding/packed sequence 边界、最后一块有效长度和 head/batch 映射；块内仍需屏蔽无效 token。可训练路由、推理时动态路由和固定 mask 各自有不同的梯度与版本约定；若训练路径允许的证据在推理路径被裁掉，或反向映射与前向布局不一致，模型语义和结果就会改变。这里说明架构层面的契约，不展开 CUDA 教程、通用单元测试或线上 benchmark 流程。

### 26.6 如何判断一次稀疏化真正改变了什么

本节把稀疏方案分为三本相互独立的账：连接图删除了哪些有效边，block layout 让 kernel 实际跳过了哪些 tile，router 与索引管理增加了多少成本并遗漏了哪些候选。固定 pattern、动态 block selection 和 dense baseline 的差别应按可见关系、选择工作和执行粒度理解。最终总结只给出分析架构声明的判断顺序，并将经典 pattern、dense kernel、模型专属 DSA 以及跨路线的系统比较分别回链对应章节。
