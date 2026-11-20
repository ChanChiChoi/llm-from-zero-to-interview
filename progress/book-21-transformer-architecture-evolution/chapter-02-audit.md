# 第 2 章《Self-Attention 的信息路由本质》审阅记录

审阅日期：2026-11-19。状态：章级外部扫描、正文精修与本地静态门禁完成；GitHub MathJax 页面渲染和整册连续复读留待全书门禁。

## 读者契约与章节职责

- **书册目标贡献：** 让读者沿一次 self-attention 读取过程看清 Query 如何与 Key 形成分数、softmax 如何产生逐行权重、Value 如何据此混合，并认识权重、可见性和最终任务表现之间的边界。
- **读者前置：** 已理解基础 Transformer 与常见向量/矩阵记号；第一章已说明序列建模和 Transformer 的总体动机。
- **离章结果：** 能手工复走一个小型 Query-Key-Value 例子；说明缩放点积、softmax 的归一化维度、mask 对信息路径的限制，以及 attention weight 不是完整因果解释。
- **读者路径：** 序列位置需要读取什么 → 早期跨序列 attention 与 self-attention 的关系 → Q/K 匹配和 V 混合 → score、scale、softmax → 张量形状与单头 causal 实现 → 双向/因果可见性 → 外部检索、长上下文有效利用与成对计算成本边界。
- **前后交接：** 第 3 章完整展开多头投影、表示子空间与 head；第 7 章主讲位置表示；第 8 章主讲 mask 与注意力模式；第 10、11、13、20 章分别主讲注意力成本、KV cache、长上下文评估和 attention kernel。

## 章级扫描与外部主题去向

- **检索日期：** 2026-11-19。扫描范围由本册目标和“Self-Attention 的信息路由本质”限定，覆盖一般 attention 与 self/cross attention、Q/K/V、scaled dot product、softmax、multi-head、mask、位置信息、解释边界、长上下文任务表现及复杂度。
- **技术来源：** Vaswani 等人的 [Transformer 原论文](https://arxiv.org/html/1706.03762v7)（§§3.2.1–3.2.3、§§3.5、4）；Bahdanau 等人的[序列到序列 attention 论文](https://arxiv.org/html/1409.0473)（§§2.1、3.1）；[BERT 原论文](https://arxiv.org/html/1810.04805)（摘要、引言、§3）；Jain 与 Wallace 的[attention explanation 研究](https://arxiv.org/html/1902.10186v3)和 Wiegreffe 与 Pinter 的[回应与诊断框架](https://arxiv.org/html/1908.04626v2)（摘要、引言）；Michel 等人的[head pruning 研究](https://arxiv.org/html/1905.10650)（摘要、引言）；[Lost in the Middle](https://arxiv.org/html/2307.03172v3)（摘要、引言）。同时查看 Mistral 7B 与 FlashAttention-2 原文摘要，确认更细结构和 kernel 内容有后续专章主讲。
- **教学组织样本：** Stanford CS224N 2024 Lecture 8 先从一般 query-conditioned weighted sum 回到 cross-attention，再引入 self-attention、Q/K/V 和位置顺序（[课程幻灯片，第 28–36 页](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/slides/cs224n-2024-lecture08-transformers.pdf)）；CS224N 2022 Lecture 9 将 vectorized self-attention 展开为 Q/K/V、分数、softmax、Value 加权（[课程幻灯片，第 29–31 页](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/slides/cs224n-2022-lecture09-transformers.pdf)）。这些材料只证明各自课程采用了这些讲解路线，不证明该顺序对所有读者最优。

| 外部主题 | 证据与读者结果 | 决定与去向 |
| --- | --- | --- |
| 从固定上下文向量到按需读取 | Bahdanau 等人的 RNN encoder-decoder attention 为每个目标位置形成独立 context；Transformer 将 attention 用于同一序列各位置 | 在本章开头以短历史桥接解释 self-attention 的问题来源；不扩成完整 NMT 历史 |
| Q/K/V、点积分数、缩放、softmax 与 Value 加权 | Transformer 原论文的算子定义和缩放动机；课程材料呈现可跟随的逐步路径 | 保留为核心主线；新增人工构造的二维小向量锚点，沿分数、softmax 权重和 Value 混合复算 |
| head 与多头子空间 | Transformer 原论文定义多头；Michel 等人在指定机器翻译与 BERT NLI 模型上研究 head 裁剪 | 本章只说明当前权重属于某层/某 head；head 数、冗余和子空间职责完整交给第 3 章 |
| mask 与双向/因果可见性 | Transformer decoder 将非法未来位置在 softmax 前置为负无穷；BERT 使用双向 encoder 表示 | 本章保留足以读懂单头示例的最小区别；Prefix LM、padding/loss mask 和模式细节交第 8 章 |
| attention weight 的解释边界 | Jain 与 Wallace 对 RNN attention 提出忠实性担忧；Wiegreffe 与 Pinter 指出解释定义和测试设计会改变结论 | 将“局部混合系数”与“最终预测因果解释”分开；明确相关实验主要针对 RNN，避免推广为 Transformer/LLM 的通用定论 |
| 长上下文位置敏感性 | Lost in the Middle 在指定模型的多文档问答与合成键值检索中移动相关信息位置 | 只保留“可见不等于能稳定利用”及结论边界；完整评测和机制讨论交第 13 章 |
| 成本和高效 attention 路线 | 全连接关系数随位置数平方增长；FlashAttention-2、Mistral 等来源涉及 kernel 或具体结构设计 | 保留成对关系数的最小直觉；FLOPs/显存、KV cache、kernel 与模型案例交第 10、11、17、20 章及高效 attention 专章 |
| 外部检索与内部 attention | RAG 检索从外部索引选文档，self-attention 在当前输入表示间进行可微分加权混合 | 保留简短机制边界，不扩展成 RAG 系统设计；外部检索和 Agent memory 按书系其他章节主讲 |

## 正文修改与去向

1. 删除原 `2.0 本讲范围与资料` 中的改稿目标、演示计划和资料罗列；用读者结果及相邻章职责说明本章范围。
2. 调整主线为“问题来源—单个读取过程—小向量手算—score/scale/softmax—形状与单头代码—可见性—解释、检索和成本边界”。补清 Bahdanau 跨序列读取如何缓解固定上下文向量的瓶颈，并将开篇与卷积的比较收窄到本章解释的动态权重，避免重复第一章。
3. 加入二维人工向量示例，手算缩放分数、softmax 权重和加权输出；明确数值为教学构造，不代表真实模型向量的可读语义。
4. 修正缩放解释的范围：说明原论文采用的独立、零均值、单位方差假设及其点积方差结论，不再声称任意配置下可保证训练稳定或“早期 attention 过早硬选择”。
5. 重写 attention weight 解释段，区分本层 Value 混合系数和原输入对最终预测的因果贡献；按模型/任务限定 Jain–Wallace 与 Wiegreffe–Pinter 两条相反观点。
6. 收窄 mask、RAG、Lost in the Middle 与复杂度内容，并添加到第 3、8、10、11、13、20 章的本地入口；删除重复的模型清单和未经证据支持的长上下文原因罗列。
7. 删除自拟 `C_route`、`R_future`、`C_row`、entropy gain、`R_interp`、`G_routing` 等路由审计公式及 0 依赖 demo。其目标标签、阈值和“解释支持”来自作者任意设定；entropy 增量不能单独作为训练稳定性指标；人为设置一个中间位置 Key 分数也不能复现 Lost in the Middle 的模型任务结果。
8. 保留现有单头 PyTorch 类代码但不改可执行语句；补充输入/输出 shape、softmax 维度、causal mask 作用和适用边界，并如实标为“静态审阅，未运行”。练习改为复走 mask 和 Value 改变时输出的影响，不再依赖已删除 demo，也不提前要求实现完整多头模块。
9. 新的 attention 定义、Bahdanau/BERT 和 head pruning 来源事实已补入 [attention 来源证据记录](../../research/attention-mechanism-and-interpretation-2026.md)；章节结构、修改理由与交接仅记录在本文件。

## 本地验收与剩余项

- 章节 H2 顺序为 2.1–2.18 连续；原 `2.0` 作者工作流段已移除。
- Markdown 围栏标记共 30 个，配对；逐行静态检查未发现未配对行内 `$`，并按模块 04 修正连续公式列表及全角标点后的定界符间距；6 个新增本地链接均解析到现存章节文件；`git diff --check` 通过。
- 代码块只保留原有单头 PyTorch 实现，状态为“静态审阅，未运行”；本轮没有运行代码、测试、模型、API、benchmark 或 PDF 构建。
- 待全书门禁：GitHub MathJax 页面核对、代码执行验证（如后续阶段决定需要）、整册连续复读及历史外链再检。
- 下一入口：第 3 章《Q、K、V、Attention Head 与表示子空间》，重点避免重讲本章单头加权读取。

## 后续配套维护（2026-11-20）

- 第十三章长上下文精修时再次复核练习 54B，将旧的人工 route score/gate 改为给定 Q/K/V 与 causal mask 的手算，以及只改变目标证据的反事实问题。该练习仍服务本章的 attention 信息路径读者结果；它不模拟真实模型，也不以 attention 权重单独证明预测因果解释。变更及第十三章其余配套同步见[第十三章精修审计](chapter-13-audit.md)。
