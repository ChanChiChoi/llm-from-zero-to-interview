# Memory-Augmented Transformer 架构研究记录

## 研究问题与范围

- 日期：2026-11-23。
- 问题：模型如何把过去输入编码为后续计算可读取的状态；不同 Memory-Augmented Transformer 如何写入、寻址、读取和融合记忆；各论文证据能支持哪些任务与资源结论。
- 范围：第二十一册第四十六章。重点是跨片段或交互步骤保留的模型架构状态，不把用户画像、RAG 文档库、当前请求 KV cache、上下文折叠或 Agent checkpoint 当成同一类对象。
- 读者结果：能比较 K/V 外部缓存、解耦读取网络、潜在情景记忆及原生递归状态的表示和信息路径，理解训练与在线更新边界，并把评测主张限制在论文协议内。
- 检索词：`memory-augmented transformer`、`Memorizing Transformers`、`external key value memory`、`LongMem SideNet`、`Larimar episodic memory`、`native recurrent memory LLM`、`MemT-1.0`、`memory architecture lecture notes`；中文检索“Transformer 外部记忆”“长期记忆架构”“模型内递归记忆”“记忆增强 Transformer 讲义”。
- 来源范围：2022、2023、2024 的代表论文与 2026 年近期预印本；同时查找课程 slides、讲解 notes 和可视化文章。以下事实来自所列原文，教学材料只用于发现讲解组织和误解点。

## 研究发现

1. **记忆的表示和作用范围由架构定义。** Memorizing Transformer 把过去片段生成的 K/V 保存到非可微缓存，并用近似 kNN 为当前 query 取回 top-k 候选；局部注意力与记忆注意力经每 head 的学习门控合并。论文按文档顺序处理 512-token 子序列，每个文档使用独立缓存，必要时丢弃较旧条目。论文在 C4 长文档、PG-19、arXiv Math、GitHub 和 Isabelle 等指定语言建模任务测试，记忆规模最高达 262K tokens；这不证明跨用户或个人事实记忆能力。旧 K/V 与持续更新的 query 参数会有分布偏移，作者将其讨论为 memory staleness。
2. **LongMem 分离生成记忆表示与读取记忆的组件。** 冻结的 backbone LLM 对过去片段编码并输出指定层 K/V；较轻的残差 SideNet 学习从缓存中检索并通过记忆增强层融合到当前表示。冻结 backbone 是为减少单一模型一边变化一边生成旧缓存所造成的表示漂移。其长文建模、ChapterBreak 与大量 in-context demonstrations 实验属于论文模型、数据及协议，不能推导为一般 RAG 或所有长上下文任务的优势。
3. **Larimar 用潜在关联记忆支持事实编辑。** 编码器、记忆读写机制和解码器在通用数据上共同训练；推理时，新情景可通过线性最小二乘/伪逆式更新写入固定形状潜在矩阵，无须对主模型梯度微调。作者在知识编辑、顺序编辑和选择性遗忘设置中报告结果。该结构不等同于存放可直接检索和引用的原始文档；基准中的选择性遗忘也不等同于在向量索引、缓存、备份及衍生副本中完成通用隐私删除。
4. **MemT-1.0 将动态记忆状态纳入 Transformer 计算。** 2026-10-19 发布的 arXiv 预印本在选定 Transformer 层加入并行 memory attention；训练所得 hyper memory 参数在线固定，动态 local memory state 以 Gated DeltaNet-2 稳定 token 级更新，并跨交互步骤继续参与计算。按固定配置，动态状态大小不随已处理 token 数逐项增长。论文在 BEAM、MemT-1.0-Nano、128K history 等设置报告相对 full-context KV cache 的推理与持久存储数字；效率测试使用 Hugging Face inference、两张 GPU、并发 1、batch 1、固定 128-token 输出，且短历史时 MemT 较慢。结果只支持该设置下的观察，不是一般模型/设备速度结论。
5. **不同任务的分数不能直接合并。** Memorizing Transformer 与 LongMem 的长文本建模/示例利用、Larimar 的事实编辑/遗忘、MemT-1.0 的多轮记忆和任务/效率评估，测量的是不同记忆合同。比较时分别问状态是否保存、查询能否读出、模型是否正确使用、更新或遗忘是否损伤其他内容，并记录历史长度、模型、数据、实现和计时设置。

## 直接证据来源

### 原始论文和正式论文页

- **Yuhuai Wu et al., “Memorizing Transformers” (2022)**，https://arxiv.org/abs/2203.08913。阅读摘要、§2–4 与图 2。支持非可微外部 K/V 缓存、局部与 kNN 记忆注意力的门控融合、片段追加/淘汰、文档级清理、分布偏移，以及所用长文本/代码/数学任务与实验设置。论文结果不代表通用个人记忆服务。
- **Weizhi Wang et al., “Augmenting Language Models with Long-Term Memory” (LongMem, NeurIPS 2023)**，https://arxiv.org/abs/2306.07174；正式会议 PDF：https://papers.neurips.cc/paper_files/paper/2023/file/ebd82705f44793b6f9ade5a669d0f0bf-Paper-Conference.pdf。阅读摘要、图 1、§1–2。支持冻结 backbone 编码过去 K/V、SideNet 检索/读取/融合、训练时分离记忆编码与读取，以及论文长文本和 memory-augmented in-context learning 任务。作者报告的分数只适用于所述模型、数据及训练协议。
- **Payel Das et al., “Larimar: Large Language Models with Episodic Memory Control” (ICML 2024)**，正式论文页：https://proceedings.mlr.press/v235/das24a.html；预印本：https://arxiv.org/abs/2403.11901。阅读 §2–5，重点关注架构、inference memory update、sequential editing 和 forgetting。支持潜在固定形状关联记忆、训练阶段的 encoder-memory-decoder 耦合、推理阶段无需 gradient 的一次记忆写入，以及特定事实编辑/遗忘实验。
- **Zhiyu Li et al., “MemT-1.0: Towards Memory-Native Evolution” (arXiv 预印本，2026-10-19)**，https://arxiv.org/abs/2610.26825。阅读摘要、§2.1–2.3、§6.5 及效率测量设置。支持并行原 Attention 与 native memory block、动态状态与可训练 hyper memory、GDN-2 token 级状态更新、固定配置状态容量，以及 BEAM 下 128K history 的条件性推理/存储结果。该文是近期预印本，不据此宣称领域共识或通用性能优势。

### 教学、notes 与可视化材料

- **Stanford CS277, “Advanced Architectures”, Lecture 15 slides**，https://web.stanford.edu/class/cs277/assets/lectures/L15.pdf。阅读 memory-based models 与 Larimar 架构页，用于观察课程如何把显式 memory、读写模块和不同架构放在一起讲。讲义主题还包括医疗报告、SSM 等；其对 memory-augmented networks 的概括不能代表所有不可微缓存或原生递归状态。
- **Hugo Cisneros, “Notes on: Memorizing Transformers” (2022)**，https://hugocisneros.com/notes/wumemorizingtransformers2022/。阅读 query 同时访问 local 与 memory K/V、top-k 选择和 per-head gate 部分；适合作为机制讲解入口，公式和方法事实以 Wu 等人的原论文复核。
- **Michael Brenndoerfer, “Memory Augmentation for Transformers” (2025)**，https://mbrenndoerfer.com/writing/memory-augmentation-transformers-long-context。查看局部注意力与外部 K/V 两路的示意图和 kNN 章节。文章提供可视化组织线索；其中关于标准 attention 容量的宽泛表述和简化代码不作为事实依据，本文不沿用其通用化结论。

## 证据边界与待核事项

- “memory”“long-term memory”不是统一接口或状态格式；每项架构都必须标明保存单位、作用范围、写入时机、读取入口和重置条件。
- 论文中模型状态“跨请求”或“跨轮次”的含义受文档、episode、trajectory 和运行实现影响；不能由方法名称推出跨服务重启或跨用户持久性。
- K/V 缓存、latent associative memory、递归状态在保存粒度、查询可解释性、可覆盖性和历史长度成本上不同；不能用一个通用 key-value 向量检索公式代替具体机制。
- LongMem 对陈旧表示的处理、Larimar 的事实忘却与 MemT-1.0 的状态压缩都是论文中的设计/评测结论，不构成通用保证。
- 未运行本地模型、代码、API 或 benchmark；没有将各论文异构任务分数合并成排名，也未把研究结果外推到生产系统。
