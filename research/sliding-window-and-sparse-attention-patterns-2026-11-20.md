# Sliding Window 与稀疏注意力模式研究记录

研究日期：2026-11-20。范围为第二十一册第十九章的章内结构和正文核验；读者已理解因果 mask、自注意力与 $O(n^2)$ attention pair。问题是如何准确解释 causal sliding window、dilation、task-global、random 和 query-conditioned sparse pattern 的连接规则、信息传播与证据边界，并与相邻章节分工。

## 检索与教学组织

- 搜索日期：2026-11-20；主要使用英文检索词 `sliding window attention`, `Longformer global attention`, `dilated sparse attention`, `BigBird random local global`, `attention sinks sliding window cache`, `DeepSeek DSA indexer top-k`, `sparse attention lecture notes`；搜索 arXiv、NeurIPS proceedings、官方技术博客、大学课程讲义及研究者/工程师教学笔记。
- 课程教学入口：[CUHK-Shenzhen, CSC5051/MDS5110/CSC6052 Lecture 11: Efficiency in LLMs, Fall 2026](https://nlp-course-cuhksz.github.io/materials/2026_fall_slides/Lecture11_Efficiency_in_LLMs.pdf)，阅读稀疏注意力图示部分，第 26—29 页。材料按 full attention → sliding window → dilated window → global+sliding → random+local+global 展示 pattern 的变化；用于发现教学顺序，不证明方法的经验优劣。
- 工程/研究团队讲解入口：[Google Research, “Constructing Transformers For Longer Sequences with Sparse Attention Methods”](https://www.research.google/blog/constructing-transformers-for-longer-sequences-with-sparse-attention-methods/)，阅读 ETC、BigBird pattern 及 blockify 实现说明。它有助于连接全局结构和实现边界；稀疏连接图及硬件实现结论仍分别回到论文和对应章节。
- 研究者/工程师教学入口：[Gunjan Dhanuka, “Learn: Sparse & Sliding-Window Attention”](https://gunjandhanuka.github.io/research-hub/learn/attention/06-sparse-and-sliding-window)，2026-09-10，阅读 learning map、window/dilation/global/BigBird、attention sink 和 FLOPs sections。其结构有助于看到从单层窗口到层间传播、再到全局/随机边及 sink/cache 的讲解路线；“稀疏通常无损”“任意 Longformer 全局规则可直接用于 causal decoder”、简化的深度感受野计算和默认 hardware 效益不能当作普遍结论，分别按原论文和窗口索引约定重写。
- 系统课程讲义：[Stanford MS&E 319, Lecture 5: KV Caching, Sparse Attention, and Compression](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf)，讲者/记录者 Anay Mehrotra，2026-10-06，阅读 §2.2—2.4。该讲义用独立算法呈现 indexer 评分、Top-k 选择与主 attention 的分工，并指出 indexer 仍扫描历史；作为 DSA 讲解路线与成本结构来源，不代表它所用抽象覆盖所有实现。

## 原始研究与核验结论

| 来源与阅读位置 | 支持的正文主张 | 证据边界 |
| --- | --- | --- |
| Child 等，[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)，摘要与方法 | 结构化 sparse factorization 可将论文设置中的 attention 复杂度降为 $O(n\sqrt{n})$；“稀疏注意力”不自动等同于线性复杂度。 | 复杂度属于论文采用的 factorization；不代表所有稀疏模式、实现或端到端模型成本。 |
| Beltagy、Peters、Cohan，[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)，§3.1—3.2、Fig. 2 | Longformer 区分 local window、dilated window 与 task-motivated global attention；固定宽度窗口和少量全局位置使其论文 pattern 的 attention 关系数随序列长度线性增长。其 global 位置能读取全序列，普通位置也能读取它；原论文用于长文档/BERT 风格双向任务，pattern 和实现依任务而定。 | 原论文某些配置的线性复杂度和任务结果不证明任意模型质量或实际加速；encoder 的对称 global 规则不能不加 causal mask 地迁移到 decoder。论文对不同 kernel 实现记录了明显速度差异。 |
| Zaheer 等，[Big Bird: Transformers for Longer Sequences](https://papers.neurips.cc/paper_files/paper/2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)；[论文 PDF](https://papers.neurips.cc/paper_files/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf) | BigBird 组合 local、random、global 三类连接。原论文在其理论设定下讨论 universal approximation 与 Turing completeness，并报告相应实验。 | 理论表达能力结果不是固定深度/宽度下匹配 dense 模型的质量保证，也不证明随机 pattern 在目标硬件上更快；经验结果受论文任务和设置限制。 |
| Jiang 等，[Mistral 7B](https://arxiv.org/abs/2310.06825)，模型 v0.1 摘要与 §2 | Mistral 7B v0.1 报告采用 sliding-window attention，并与 GQA 等结构结合，可作为 causal LLM 中使用窗口模式的具体例子。 | 限定该论文版本；不从模型名推断后续版本、任意上下文利用能力或通用性能。具体配置数字不在本章重复第十七章内容。 |
| Xiao 等，[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)，§1、§3、§4 | 只保留近期 KV 的窗口策略在论文测试中于初始 KV 被淘汰后失稳；保留若干初始 sink KV 与近期窗口的 StreamingLLM 策略可恢复其测试设置中的稳定性。 | 这是推理时 cache-retention 方法，不能与模型训练使用的 sliding-window attention 混为一谈；结果只支持论文中的模型、任务和缓存条件。KV/cache 实现由第十一章、流式边界由第十五章展开。 |
| DeepSeek-AI，[DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)，§2.1、§2.3 | DSA 用 query-conditioned lightning indexer 给历史位置评分、按 Top-k 选 latent KV entries，再由主 attention 读取候选；报告明确将主 attention 与 indexer 的成本分开，后者仍随完整历史扫描增长。 | 架构原理可以作为动态选边的一个例子；报告的性能数字绑定其实现、H800 部署与测量口径。具体 DSA 公式、模型配置、kernel 和 serving 证据分别由第二十一册第八十九章等模型案例及第二十六章承接。 |

## 结构决定与待核边界

- 第十九章以“哪些 query-key 连接被允许，以及这些连接怎样跨层传播”为中心，主讲 causal sliding window、dilation、Longformer global、BigBird local/random/global，并用 DSA 简短说明从固定 pattern 到 query-conditioned selection 的变化。
- 每个复杂度只绑定其定义的 pattern 和固定参数假设。固定窗口宽度 $W$ 时局部 pair 数为 $O(nW)$；加入少量 $g$ 个双向全局位置时，论文式连接数约增加 $O(ng)$；不能由渐近阶直接推断 kernel 时间。
- causal window 的 $W$ 按“包括当前位置在内可见的 token 数”定义。位置 $i$ 的直接可见集合为 $\{j\mid \max(0,i-W+1)\le j\le i\}$；因此单层直接范围和多层潜在传播范围要分开写。具体可达路径不等于读到了证据，更不等于任务成功。
- Longformer 的 task-global 模式原用于特定双向长文档任务；decoder 的全局连接必须另定方向并遵守因果约束。BigBird 的随机连接改善结构图路径的理论动机，不写成任务质量或硬件速度保证。
- StreamingLLM 只作为“训练连接模式与推理 KV 保留策略不同”的短边界提示，不在本章展开 cache；RAG 只用于区分内部注意力连接和输入外部文档选择。
- 不在第十九章展开 block-sparse kernel、mask 编译、benchmark gate、线上 fallback、详细 indexer 训练/实现、DSML、Agent context management、DSA 图表/GRPO 或服务参数。块级模式与动态稀疏实现回链第二十六章，attention kernel 回链第二十章，长程任务能力回链第十三章，位置外推回链第十八章，KV/sink 与流式 cache 回链第十一、十五章，GLM-5 的 DSA 模型案例回链第八十九章。

## 本记录未支持的结论

- 没有一个 pattern 对所有任务最优，也没有“任意稀疏 pattern 与 dense attention 等价”的经验结论。
- 全局 token、随机边或多层传播存在，不足以证明精确证据被保留或查询成功。
- 固定窗口的线性关系数、论文中的复杂度或结构图示不等于目标 GPU 上的 wall-clock speedup。
- attention sink 不保存已被淘汰的中间内容，也不构成任意长远检索能力。
- DSA 的主 attention candidate count 不能单独代表整套 indexer、cache、selection、gather 和 serving 成本。
