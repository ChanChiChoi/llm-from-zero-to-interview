# 替代序列架构的比较坐标与证据边界

研究日期：2026-11-23。研究主题：怎样比较 attention、稀疏/线性 attention、递归状态模型、卷积和混合序列架构的历史表示、任务能力、序列复杂度、硬件映射与生态信号。

## 研究问题与检索

本次研究核对四类主张：架构对比是否处于同一层级；历史以何种形式保存并被当前 token 读取；理论复杂度与训练、prefill、逐 token 生成成本如何区分；公开实验能否支持不带任务条件的“表达力”排名。研究范围是模型内部的序列混合机制与架构证据；完整 serving、产品运行时、当前行业采用比例不在范围内。

2026-11-23 使用网页搜索，并直接阅读 arXiv、PMLR、ICLR、研究者博客和公开课程页面。检索词包括：sequence architecture comparison complexity expressivity hardware、attention recurrent convolution state access lecture notes、linear attention recurrent state computational cost、Mamba retrieval state tracking comparison 2026、architecture ecosystem checkpoint kernel reproducibility，以及“序列架构 信息流 对比”“替代架构 状态 复杂度 硬件”“SSM attention 检索证据”。先查原论文和正式会议页面确认定义、复杂度与实验，再并行阅读研究者解释和课程讲义以检查概念组织。没有运行模型、代码、kernel 或 benchmark；没有做当前模型生态或部署采用率调查。

## 外部发现

### 比较对象有层级，优化内容也不同

Vaswani 等人的 Transformer 原论文按其定义的层比较 self-attention、recurrent 与 convolutional layer，给出序列长度、表示维度、顺序操作数和最长依赖路径的特定分析。该表是结构基线，不是现代实现的普遍速度表。

Dao 等人的 FlashAttention 论文给出一个清楚的执行层对照：算法保留 dense attention 的 $\mathrm{softmax}(QK^{\mathsf T})V$ 运算，FLOPs 仍为 $O(N^2d)$，但通过 tiling 减少 HBM 与片上 SRAM 间读写；其 Theorem 1 还界定了额外内存。由此可区分“更高效执行相同交互”与“改变位置交互集合或历史表示”的结构路线。

Dao 与 Gu 的 State Space Duality 论文建立特定状态空间序列算法与 structured attention matrix 之间的联系。这种联系说明架构家族可以共享代数表示；它不表示任意 attention、SSM 或配置在质量、状态与计算上都等价。

### 固定状态、复杂度与硬件效率是不同描述

Katharopoulos 等人的线性 attention 论文在有限特征映射下将因果历史汇总写成可递推的 numerator/denominator 状态，成本取决于特征宽度和 value 维度。其“线性”依赖固定特征维度和具体 kernel；精确 softmax 的指数 kernel 不是有限维特征的直接等价形式。

Gu 与 Dao 的 Mamba 论文把输入依赖递归与硬件感知 scan 结合：论文说明了 kernel fusion、parallel scan 和 backward recomputation 的作用，并讨论朴素中间状态的形状与内存路径。此证据解释特定算法为何需要专门实现，不证明所有固定状态模型都比 attention 快。

Lahoti 等人的 Mamba-3 ICLR 2026 论文明确指出理论上线性的推理仍可能硬件效率不佳，并给出其方法与 kernel 的评测。论文中的实现和性能判断绑定具体模型、状态维度、精度、硬件与代码，不是所有线性路线的 wall-clock 排名。

### 表达能力必须对应任务定义和训练后的行为

Zoology 研究在 70M—360M 规模、10B Pile token 设置下比较不同 sequence mixer。作者报告其 gated convolution 与 attention 的平均语言模型 perplexity 差距中，关联回忆 token 切片约占 82%；该切片通过特定 n-gram 频率启发式构造。结果支持在这组模型与语料里检查关联回忆，不支持把所有架构的自然语言能力归结为同一检索分数。

Jelassi 等人的 ICML 2024 工作在其固定大小 latent state 的 generalized state space model 形式化和 copying 任务下给出理论与经验结果。它支持相应假设下的复制限制及论文模型的任务差异；不应把该结论改写成任意状态空间模型都不能做任意自然语言任务。

Waleffe 等人的 Mamba-based LM 研究比较约 8B 规模的 Mamba、Mamba-2、Transformer 与 hybrid 模型，分别报告多项下游任务、Phonebook 复制和长上下文结果。纯 SSM 与含 attention 的 hybrid 是不同比较对象；训练 token、格式与任务分布改变时，结果也会变化。

Mamba-3 论文提供较新的任务切片示例：1.5B 模型使用 100B FineWeb-Edu tokens、Llama-3.1 tokenizer 和 2K context。其检索部分以 2K 截断的 cloze 版本评估 TQA、SQuAD、SWDE、FDA 及合成 NIAH；作者报告前两类任务有竞争力、半结构化/非结构化信息抽取偏弱，NIAH 多数设置可匹配或超过若干基线。结果只覆盖这些版本和协议，且该论文指出 Transformer 位置编码与部分 hybrid 结构也是比较条件。

### 教学解释和生态信号

Albert Gu 的 2025 年研究者文章用“递归状态是自回归上下文访问路径”解释 SSM 与 Transformer 的记忆取舍，可用于组织压缩状态与逐 token cache 的概念；其粗略类比和作者判断不是普遍实验结论。

Eva Koroleva 的公开课程第 10 课（2026-10-19 发布、2026-11-10 更新）把状态大小、selection、scan、相同数据上的模型比较和 hybrid 放在同一学习路径中，并以可读 notebook 展示机制。它支持课程如何串起这些概念，不替代架构定义或跨模型性能证据。

本次资料不足以估计各架构的行业采用率或生态成熟度排名。可复核的生态信号应逐候选检查固定版本源码、checkpoint/tokenizer、训练配置、框架与 kernel、权重转换材料、调试资料和维护状态；这些信号不能证明模型质量，也会随时间改变。

## 来源与阅读范围

| 来源 | 阅读位置 | 支持内容与使用边界 |
| --- | --- | --- |
| Ashish Vaswani 等，[Attention Is All You Need](https://arxiv.org/html/1706.03762)，NeurIPS 2017，arXiv HTML v7 | §4 Why Self-Attention，Table 1 | 原始层级比较、每层复杂度、顺序操作数和依赖路径；绑定论文符号与层假设，不外推为现代硬件速度。 |
| Angelos Katharopoulos 等，[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/html/2006.16236)，ICML 2020，arXiv HTML v3 | §§3.2—3.4 | feature width、因果状态递推和线性 attention 成本；特定 feature map 不等于精确 softmax。 |
| Tri Dao 等，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135)，NeurIPS 2022，arXiv HTML v2 | §§3.1—3.2、Theorem 1 | 保持 dense attention 运算并降低数据移动的实例；复杂度和额外内存依赖论文算法定义。 |
| Tri Dao 与 Albert Gu，[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)，ICML 2024 | 摘要与方法定位 | 特定 SSM 与 structured attention 的数学联系；不是全部架构实例等价。 |
| Albert Gu 与 Tri Dao，[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/html/2312.00752v2)，2024-05-31 修订版 | §§3.1、3.3.2、4.5、Appendix E.5 | selective state/scan 的计算和硬件路径；实验数字不作为当前通用性能排名。 |
| Samy Jelassi 等，[Repeat After Me: Transformers are Better than State Space Models at Copying](https://proceedings.mlr.press/v235/jelassi24a.html)，ICML 2024 | 摘要、§§2—4、讨论 | 固定状态 GSSM 形式与复制任务结果；理论与经验范围服从论文模型和任务假设。 |
| Samy Arora 等，[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927) | §2、§3、Appendix C | 特定 mixer 的 Pile perplexity 与 associative recall 诊断；82% 数字绑定该实验和 recall-hit 启发式。 |
| Roger Waleffe 等，[An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)，arXiv v1，2024-06-12 | §§3—4 | 约 8B 纯与混合模型、任务和训练预算比较；不能混记纯 SSM 与 hybrid，也不覆盖所有配置。 |
| Aakash Lahoti 等，[Mamba-3: Improved Sequence Modeling using State Space Principles](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)，ICLR 2026；[arXiv 全文](https://arxiv.org/html/2603.15569) | §§4.1.2、4.2、4.3；表 4 | 1.5B、100B token、2K context 条件下的 retrieval/state-tracking 与实现比较；细节不外推到全部 SSM 或部署负载。 |
| Albert Gu，[On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/)，2025 | “The Tradeoffs of State Space Models and Transformers”“Autoregressive states of sequence models” | 研究者对压缩 state 与 token cache 的解释路径；观点和类比不是普遍能力定理。 |
| Eva Koroleva，[Beyond attention with Mamba, state space models and hybrids](https://blog.korolevalab.com/blog/2026/transformer-architectures-10-mamba/)，2026-10-19 发布、2026-11-10 更新 | 第 10 课，尤其是 state、same data/different machinery、hybrids 部分 | 教学组织、状态追踪和对照实验讲解；课程示范不是独立复现或普遍性能证据。 |

## 综合判断与未验证边界

架构比较的直接依据是信息表示和执行结构；质量与效率结论仍须绑定任务、训练和实现。固定状态描述存储形状不随序列长度增长，不构成任意任务的容量定理；存在 attention 连接也不保证当前 query 学会可靠读取。硬件和生态因素只可对候选版本逐项检查，本研究未建立市场采用率、跨设备 wall-clock 排名或完整 serving 成本。全部实验性结论为 source-backed；没有本地模型复现。
