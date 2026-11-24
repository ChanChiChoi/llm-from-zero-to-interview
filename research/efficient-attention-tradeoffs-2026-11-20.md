# 高效 Attention 路线与标准 Attention 的取舍

研究日期：2026-11-20。用途：第二十一册第 28 章的职责修正、正文事实核验与解释组织。此记录只收新研究发现及证据范围；章节采纳和删改决定见 `progress/book-21-transformer-architecture-evolution/chapter-28-audit.md`。

## 问题与范围

读者已理解标准 Transformer、attention 配对计算、FlashAttention、线性/核注意力、稀疏注意力和长序列方法。需要回答：不同“高效 attention”各自减少哪种成本、保留或改变什么，以及为何一个渐近复杂度数字不能推出对所有任务和设备都适用的替代方案。

本次检索围绕 `FlashAttention exact IO-aware`、`FlashAttention-2 work partitioning`、`efficient attention taxonomy hardware sparse compact linear`、`efficient transformers survey`、`efficient attention lecture notes` 和 `attention memory access patterns`。以原论文核验算法含义和复杂度口径，并行查看研究者/工程解释与课程讲义以检查教学组织；不调查当前模型部署比例或生态采用率。

## 核心发现

1. **“高效”不是同一种改动。** FlashAttention 的 dense 路径保留完整 query-key 交互和 attention 的数学运算，靠分块、片上计算和在线归一化减少中间矩阵写回；它没有消除密集配对所需的二次计算。稀疏路线减少允许或实际计算的连接；核/特征路线通过可分解形式或有限特征改写计算，可能替换或近似标准 softmax；KV 压缩主要降低解码阶段的历史状态存储。比较前要先说明省的是数据搬运、交互数、算子成本还是缓存容量。
2. **精确算法也会有执行取舍。** FlashAttention 原论文定理给出标准 attention 输出；在序列长度增长时，其配对运算量仍是平方级，输入输出之外的辅助存储则是线性级。论文的 IO 分析比较 HBM 访问，并指出反向过程可以以重算中间 tile 换取较少的数据读写。实际收益由具体设备、片上存储、shape、dtype 和实现决定。这里的“精确”指算法目标保持同一 attention 运算，不承诺不同浮点执行顺序逐位相同。
3. **稀疏和线性路线改变的对象不同。** 稀疏方法约束 query 可访问的 key 集合，质量依赖 pattern、路由或哈希候选能否覆盖任务需要的信息；线性/核方法依赖可用的特征映射、状态维度和核函数，标准 softmax 的完整 dense 结果不自动随之保留。具体机制与公式分别由第 24—27 章主讲，本章不重复证明或算法细节。
4. **不同路线可以组合。** FlashAttention 原论文也描述跳过零块的 block-sparse 扩展，说明 IO-aware 执行和稀疏连接不是互斥类别；但组合能否有效仍受 pattern、kernel、硬件和任务约束。书稿不把方法类别画成互不相容的阵营。
5. **没有把“未完全替代”写成采用率事实。** 本次公开来源足以支持各路线的机制与边界，不能证明某路线在整个行业、模型家族或线上流量中的采用比例。章节把“没有一种路线在所有任务与资源条件下同时保留相同语义并消除全部成本”作为由机制差异得出的条件化判断，不宣称实际市场排名或生态惯性是已经测得的普遍原因。

## 来源与证据边界

| 来源 | 实际使用内容 | 支持范围与限制 |
| --- | --- | --- |
| Tri Dao 等， [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，NeurIPS 2022；arXiv 正文 §3.1–3.2、附录 D.1 | exact attention 算法、分块与重算、辅助内存与 HBM IO 分析、block-sparse 扩展 | 支持论文所定义算法和复杂度分析；实测仅绑定原论文报告的 GPU、shape、实现与基线，不外推为当前硬件速度。数学精确性不代表跨执行路径逐位浮点一致。 |
| Tri Dao， [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)，2023；正文 §3–4 | 通过非矩阵乘法操作、并行切分和 warp 工作划分改善实现；输出运算不变 | 算法改动与性能数据绑定论文中的 A100/H100、GPT 模型、长度与 baseline。数字不作为一般性速度保证。 |
| Yi Tay 等， [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)，arXiv v2，2022-03 更新 | Reformer、Linformer、Performer、Longformer 等效率路线的分类脉络 | 适合作为历史分类和方法地图；年代较早，不代表当前路线完整，也不支持采用率或普遍效果结论。 |
| Jintao Zhang 等， [Efficient Attention Methods: Hardware-efficient, Sparse, Compact, and Linear Attention](https://attention-survey.github.io/files/Attention_Survey.pdf)，项目托管综述，检索页显示 2025 年材料 | 将方法整理为硬件实现、稀疏交互、KV compact 和线性注意力四类 | taxonomy 是作者提出的整理框架，不等于唯一或正式标准分类；使用它解释优化对象，不把作者的“全面”评价当作已验证事实。未从项目页确认最终 venue/revision。 |
| CUHK-Shenzhen NLP 课程， [Lecture 11: Efficiency in LLMs](https://nlp-course-cuhksz.github.io/materials/2026_fall_slides/Lecture11_Efficiency_in_LLMs.pdf)，2026 秋课程 slides，pp. 26–29、140–145 | 将高效模型分层讲解；用 attention pattern 图比较 dense、sliding window、global 与 random；用 tiled 图和 online softmax 说明 FlashAttention 的数据流 | 仅用于检查教学路径和图示对应，不作为方法定义、复杂度或设备性能的唯一依据。课件把“same output”用于机制说明；正文仍明确浮点运算顺序可能产生微小数值差异。 |
| Aayush Garg， [FlashAttention: Making Attention I/O-Aware](https://huggingface.co/blog/garg-aayush/flash-attention)，Hugging Face Blog | 以 memory hierarchy、tile 和 online softmax 连接机制与 IO 直觉 | 用作解释性来源；原始复杂度和方法事实回到 Dao 等人的原论文。 |

相关的一手证据已在本书前章研究记录覆盖： [线性注意力与状态分解](linear-attention-factorization-state-2026-11-20.md)、[Performer/FAVOR+](performer-favor-plus-random-features-2026-11-20.md)、[块稀疏布局与动态路由](block-sparse-attention-layout-and-routing-2026-11-20.md)、[Longformer/BigBird/Reformer](longformer-bigbird-reformer-methods-2026-11-20.md)；dense kernel 与分页 KV 的职责和版本另见 [第 20 章研究记录](attention-kernel-and-paged-kv-evidence-2026-11-20.md)。这些来源分别支持其原问题，不代表本次复用的旧结论替代了本章的综合判断。

## 证据等级与未验证边界

- FlashAttention、FlashAttention-2 的机制与论文复杂度：`source-backed`；没有在本地执行任何 kernel 或性能比较。
- Survey 的方法类别：`source-backed` 的作者组织框架；不能写成公认且穷尽的唯一分类。
- “不存在对所有任务和资源都成立的单一替代答案”：基于方法优化对象、语义和适用条件差异的条件化综合推断，不是行业采用率或线上部署调查结果。
- 对当前主流模型采用率、生态阻力大小、平均精确检索能力或各类方法跨硬件胜负：本研究未核验，正文不作普遍判断。
