# 低秩适配与 embedding 可视化的来源证据

检索与阅读日期：2026-11-09

## rsLoRA 的缩放条件

- 来源：[A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)，Damjan Kalajdzievski，arXiv v1（2023-11-28）。
- 实际阅读位置：arXiv HTML 的 §2.2、§3、§4 和附录 A 定理证明。
- 来源支持的事实：论文把 adapter 写为 $\gamma_r BA$，并在 $B$ 零初始化、$A$ 的元素独立同分布且均值为零、方差不随 rank $r$ 改变、$\gamma_r\to0$ 等条件下定义 rank-stabilized。其 Theorem 3.2 得出渐近 rank 稳定的系数满足 $\gamma_r=\alpha/\sqrt r$。论文实验报告了其训练设置下的梯度与性能变化。
- 证据等级：原始预印本的定义、定理、证明和作者实验。
- 适用边界：定理讨论特定初始化和 $r\to\infty$ 下的训练稳定性，不证明任意任务的最优 rank，也不单独保证泛化或任务分数提高。正文应把它作为可比较的缩放方案，不写成所有配置的推荐值。

## DoRA 的幅度与方向参数化

- 来源：[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)，Liu 等，arXiv v6（2024-07-09）。
- 实际阅读位置：arXiv HTML 的 §3.2 权重分解分析、§4.1 方法、§4.2 梯度分析、§5 实验及 §6.1 QDoRA。
- 来源支持的事实：论文沿矩阵列分解权重幅度与单位方向，以一个逐列 magnitude 参数表示长度，并用低秩 $BA$ 更新方向。初始化时幅度取预训练权重的列范数，方向使用预训练权重；低秩增量初始为零。论文称合并后可在推理前写回预训练权重，并在其 LLaMA、LLaVA、VL-BART 任务中报告与 LoRA 的比较结果。
- 证据等级：原始预印本的方法、公式和作者实验。
- 适用边界：实验结果绑定论文的模型、数据和任务；作者把更广领域的适用性留作后续工作。量化合并、特定 PEFT 版本和推理引擎行为仍需按目标实现核验。

## QLoRA 的计算与显存账本

- 来源：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)，Dettmers 等，arXiv v1（2023-05-23）。
- 实际阅读位置：arXiv HTML 的 §2 block-wise quantization 与 LoRA、§3 NF4、Double Quantization、Paged Optimizers 和 QLoRA 正式定义、§4 实验比较及附录 G 显存估计。
- 来源支持的事实：QLoRA 用低比特格式存储冻结的基础权重，反向传播穿过量化基础模型更新低秩 adapter。论文区分存储 dtype 与计算 dtype，描述把权重反量化到计算 dtype 后执行前向/反向，并只为 LoRA adapter 计算参数梯度。其方案结合 NF4、量化缩放常数的二次量化和 paged optimizer。
- 来源中的数值例：论文给定分块大小 64 时，FP32 一级量化常数平均增加 $32/64=0.5$ bit/参数；二次量化示例把该开销降到 $8/64+32/(64\cdot256)=0.127$ bit/参数，节省 $0.373$ bit/参数。这些是论文所述分块和格式下的账本数字。
- 证据等级：原始预印本的定义、算法、作者测量和配置。
- 适用边界：论文的模型吞吐、显存和性能数字不能直接外推到任意框架、设备或量化后端；理想化权重位数不等于训练峰值显存。二次量化元数据、未量化参数、activations、梯度、optimizer state 和 workspace 均会改变总量。

## LoRA-α 缩放系数的 2026 年研究

- 来源：[The Hidden Power of Scaling Factor in LoRA Optimization](https://arxiv.org/abs/2606.12883)，Zhang 等，arXiv v1（2026-06-11）。
- 实际阅读位置：arXiv HTML 摘要、§2 实证扫描、§3 signal-drift 分析、§4 protocol、§5 实验验证、附录 A limitations 和附录 D 实验设置。
- 来源支持的事实：作者把 LoRA 缩放系数 $\alpha$ 和学习率 $\eta$ 作为不同变量研究。主要监督微调扫描使用 Llama 3 1B/8B、Tulu 3 与 OpenThoughts、AdamW 和论文给出的训练配置；作者报告较大的 $\alpha$ 与增大学习率的优化轨迹不同，并将若干实验下的最优 $\alpha$-rank 关系拟合为带大常数的平方根形式。预印本的附录明确说明，$\alpha$ 与 $\eta$ 的不对称作用分析依赖自适应优化器，不能直接外推至 vanilla SGD。
- 证据等级：原始预印本的理论分析与作者设定下的实验。
- 适用边界：目前是预印本；拟合系数会随模型、数据、rank、初始化和优化器配置变化。它支持把缩放系数作为独立实验因素，不支持不经验证地替换所有 LoRA 配方。论文的理论框架与实测结论均应按其声明的 optimizer 和实验范围阅读。

## AQLoRA 的量化层保护与训练速度结果

- 来源：[AQLoRA: A Zero-Search Recipe for Fast Quantized LoRA Fine-Tuning](https://arxiv.org/abs/2608.23816)，Islam，arXiv v2（2026-10-11）。
- 实际阅读位置：arXiv HTML 的 §3 方法、§4 实验设置、§5 主结果、§6 速度分解、§7 消融、§8 限制及附录 C–D rank sweep 与 rsLoRA 组合。
- 来源支持的事实：方法用一次 CPU 扫描评估逐层 NF4 重构误差，并组合高精度保护层与 adapter placement。作者在其六个模型、四类架构和最多 13 次独立计时中报告 speed setting 相对 well-tuned QLoRA 的训练速度结果。重要限制是：作者在五个模型、三个预算下比较后，报告其 NF4-error whole-layer 排名相对参数量匹配的随机选择没有可测优势；速度收益主要随避免反量化的层数变化。论文还报告在共享学习率的若干 rank/rsLoRA 配置中出现明显性能下降，同时说明没有按方法分别调学习率，因此这不能回答降低学习率能否避免该失败。
- 证据等级：单篇最新 arXiv 预印本的系统实验和消融。
- 适用边界：上述速度与质量结果绑定预印本的硬件、软件栈、模型、数据和计时协议；层选择规则的主张受该预印本自身负向消融限制。共享学习率下的 rsLoRA 失败是特定配置观察，不能作为 rsLoRA 渐近定理的反例，也不能推成普遍失败规律。

## 截断 SVD 的历史来源

- 来源：[The Approximation of One Matrix by Another of Lower Rank](https://doi.org/10.1007/BF02288367)，Eckart 和 Young，Psychometrika 1(3)，1936-09，211–218 页。
- 实际阅读位置：Cambridge Core 的题名、作者、期刊卷期、页码与摘要。该页面的全文在本次查阅中不可读。
- 来源支持的事实：书目资料确认低秩矩阵近似问题的原始论文信息；摘要说明其把问题表述为 least-squares，并通过 canonical form 简化求解。
- 证据等级：出版社原论文页面的书目元数据和摘要；章节给出的投影与零空间推导独立展示 Frobenius 范数和谱范数下的下界。
- 适用边界：本次不能把出版社摘要当成逐句支持全部现代 Eckart–Young 定理表述的全文证据。

- 来源：[Symmetric Gauge Functions and Unitarily Invariant Norms](https://academic.oup.com/qjmath/article/11/1/50/1525786)，L. Mirsky，《Quarterly Journal of Mathematics》11(1)，1960，50–59 页，DOI 10.1093/qmath/11.1.50。
- 实际阅读位置：Oxford Academic 的论文题名、卷期、页码和 DOI 书目记录；全文未在本次查阅中获取。
- 证据等级：出版社书目记录。
- 适用边界：正文对 Frobenius 与谱范数的结论由其自身的投影和零空间证明支持；未依赖本次无法读取的 Mirsky 全文细节。

## t-SNE 与 UMAP 的低维布局目标

- 来源：[Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)，van der Maaten 与 Hinton，JMLR 9(86)，2008，2579–2605 页。
- 实际阅读位置：JMLR 正式页面摘要及论文 PDF 中 t-SNE 的算法与目标段落。
- 来源支持的事实：t-SNE 把高维空间的局部邻近关系写成相似度分布，并优化低维映射中的相似度分布；该目标不是保留所有高维点对距离的无损二维表示。
- 证据等级：同行评审原论文页面与正文算法。
- 适用边界：可视化用于观察和提出假设；二维布局不能单独证明语义、因果关系或下游任务质量。

- 来源：[UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)，McInnes、Healy 和 Melville，arXiv v3（2020-09-18）。
- 实际阅读位置：arXiv HTML 的 §2 理论基础、§3 graph construction/layout、§4.3 超参数、§5 实验和 §6 weaknesses。
- 来源支持的事实：论文用局部 fuzzy simplicial sets 构造高维加权邻域图，并以低维图的 cross-entropy 布局目标优化；$n_{neighbors}$ 改变局部度量尺度，$min\_dist$ 改变低维点可聚集程度，原文展示这些参数改变布局。论文也把主要目标限定在局部结构，并讨论全局结构保持的不确定性。
- 证据等级：原始预印本中的定义、算法、参数实验和限制章节。
- 适用边界：原论文参数实验和布局不能证明某个通用设置最佳；降维后的点间距离和簇大小不等同于任务质量或原空间的完整距离结构。
