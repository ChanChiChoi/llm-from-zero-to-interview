# 第十九章《Sliding Window Attention、Global Attention 与稀疏注意力》精修审计

审阅日期：2026-11-20。章节职责复核、研究、正文主题重写、结构大纲、配套同步和本轮静态检查已完成。GitHub MathJax 实际渲染、目标读者试读、全章外链终审和第二十一册连续复读仍待后续门禁；按用户要求不更新 PDF 或 `plan.md`。全套书精修 goal 继续 active。

## 读者契约与章节边界

- **前置：** 读者已理解因果 mask、自注意力的 Q/K/V 交互，以及 attention pair 随序列长度增长的基本代价；第八章负责 mask 可见性，第十章负责 dense attention 的计算与显存瓶颈。
- **离章结果：** 能读出 causal sliding window、dilated、task-global、local/random/global 和 query-conditioned selection 各自允许哪些位置交互；能解释逐层局部连接何时提供窗口外信息路径、为何路径不保证任务成功；能区分 pattern pair 数、任务级有效上下文和目标硬件执行收益。
- **本章职责：** 说明稀疏 attention pattern 改变模型内部哪些 query-key 连接及其跨层信息路径，并用固定规则和内容选择式模式作比较。
- **范围外：** attention pair 与 dense 成本回链第十章，KV cache 与流式保留回链第十一、十五章，长距离证据评测回链第十三章，RoPE 和长上下文位置映射回链第十八章，FlashAttention 与 kernel 回链第二十章，块级稀疏布局和动态稀疏执行回链第二十六章；GLM-5/DSA 模型案例由第八十九章承担。

## 旧稿诊断与结构决定

旧稿约 446 行、43 个 H2，将经典稀疏 pattern、图路径/复杂度推导、窗口 mask 代码、kernel 与硬件性能、DeepSeek-V3.2/DSA 模型实现、DSML 协议、Agent 上下文管理和 RL 评测揉在同一章。章节职责由“哪些位置能交互”漂移到 serving、runtime 和具体模型报告；局部公式、任意 gate 与 demo 不能共同形成可靠的读者路径。

按章节精修采用主题重写：保留章名、文件路径与目录入口，将正文收束为六节，从稀疏连接图开始，依次讲 causal sliding window、dilation/结构化稀疏、Longformer global、BigBird local/random/global，再到 query-conditioned selection 与适用边界。完整章节设计见[第十九章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/19-Sliding-Window与稀疏注意力.md)。

## 研究核验与正文决定

研究问题、来源阅读位置、教学组织入口和证据边界见[稀疏注意力模式研究记录](../../research/sliding-window-and-sparse-attention-patterns-2026-11-20.md)。正文中的关键决定如下：

- 将连接 pattern 与精确 attention kernel、核近似/线性 attention 分开；“稀疏”本身不决定复杂度或硬件速度。Sparse Transformer 的 $O(n\sqrt{n})$ 只绑定其结构化分解。
- sliding-window 宽度 $W$ 明确定义为包含当前位置在内的可见 token 数；用可见集合、六行位置表和固定 $W$ 的 pair 数解释直接访问范围，并用两层路径说明较早信息可能传播但不保证细节保留。
- Longformer 的 global 规则及 BigBird 的 local/random/global 组合按原论文任务和理论条件表述。双向 encoder 的 global 规则不能直接移植到 causal decoder；论文的表达性结论不等于给定模型的任务质量或 GPU 加速。
- DSA 只作为内容选择式稀疏的概念例子，区分 indexer 候选评分与主 attention 读取，并指出候选数量不能代表整套 indexer、cache、selection/gather 和 serving 成本。配置、训练和实现细节留给模型案例与第二十六章。
- StreamingLLM 只用来区分训练时 attention pattern 与推理时 KV 保留策略；RAG、位置扩展、任务评测和 kernel 各自属于不同问题。

## 删除内容及后续安置

从本章移除了 DeepSeek-V3.2/DSA 的模型实现细节、图表与 benchmark，DSML 工具协议、Agent context management、GRPO/RL 推导以及相关 serving/kernel 教程。这些内容不属于本章的 pattern 读者结果。相关来源条目仍保留在 `PAPERS.md`，**本轮没有把上述模型专属材料整体迁入其他正文**；后续精修第五十八章 DeepSeek-V2/V3/R1 及相关模型/Agent 章节时，再判断哪些材料有明确读者结果和合适主讲位置。第八十九章只承接 GLM-5 的 DSA 模型案例，不等于已接收全部 V3.2/Agent 内容。

第十九章保留经典 pattern 的概念入口；第二十六章现有正文仍有 pattern taxonomy、图路径和实现边界的交叉。按册级审计决定，等轮到第二十六章时基于完整正文重审职责和重复，不在本轮预先改写。

## 配套同步

- 新建完整章节大纲；书册目录中的第十九章标题未变，无需改目录项。
- `PAPERS.md` 的长上下文来源行补入 Sparse Transformers、Longformer、BigBird、Mistral v0.1 和 StreamingLLM；DSA 一般模式入口改为本章 §19.6，模型案例链接第八十九章。
- `EXERCISES.md` 新增 31X，覆盖 $W=3$ 可见集合、跨层路径、Longformer/BigBird 边界及复杂度与实际能力的区别；`INTERVIEW_BANK.md` 新增 31U，围绕窗口外信息传播作答。
- `GLOSSARY_EN_ZH.md` 增补 Global Attention、Dilated Sliding-Window Attention、Longformer 和 BigBird；`KNOWLEDGE_GRAPH.md` 增加稀疏 pattern 关系链。
- 第八十九章 DSA 章节补入指向本章 §19.6 的概念链接；第二十一册架构审计与全套进度队列更新到第 1—19 章完成、下一章为第二十章。

## 静态检查与遗留门禁

- 正文 H2 `19.1—19.6` 连续；包含一个独立 `math` 块公式和若干行内公式，已核对窗口集合、变量前置、跨层路径解释及行内数学定界符；无程序代码或 Mermaid。
- 正文和大纲 Markdown 围栏均成对，行内 `$` 定界符未发现奇数行；正文与大纲中的本地链接目标存在。`EXERCISES.md` 的 31X、`INTERVIEW_BANK.md` 的 31U 各自唯一；当前第二十一册资料未发现已删去的 §19.29—§19.43 旧章节引用。
- 本轮 `git diff --check` 通过。未运行代码、测试、模型、API、benchmark 或 PDF 构建。
- GitHub 页面上的 MathJax 公式视觉验收尚未完成；还需目标读者试读、逐条核对本章外链，并在全册收尾时连续复读。第二十六章 pattern 重叠及上述未重新安置的模型专属材料按前述后续章节交接处理。

下一入口为[第二十章《FlashAttention、PagedAttention 与 attention kernel 优化》](../../book-21-transformer-architecture-evolution/chapters/20-FlashAttention与PagedAttention.md)。
