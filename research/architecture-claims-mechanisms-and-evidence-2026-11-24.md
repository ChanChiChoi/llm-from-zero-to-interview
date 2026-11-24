# 新架构主张、机制与证据边界研究记录

## 研究范围

- **研究问题：** 面向第二十一册第六十章，研究怎样把“新模型/新架构”主张拆成可核验的对象、信息路径、资源影响和实验结论，并区分模型内部结构、训练、执行实现及模型外部组件。
- **读者前置：** 已理解标准 Transformer、next-token prediction 和常见配置/论文图表；不假设读者有训练 frontier 模型或大规模 serving 的经验。
- **检索日期：** 2026-11-24。
- **检索范围：** 论文原文及会议论文集、官方技术报告、课程讲义和研究者/工程团队的机制讲解。历史方法和最新的机制/任务分析都纳入；外部材料用于确认结构关系与证据边界，不为单一“下一代赢家”背书。
- **检索词：** `architecture comparison hybrid attention SSM controlled experiments model scale task Mamba Transformer paper`; `DeepSeek-V2 Multi-head Latent Attention MLA cache compression official paper`; `Mixture of Experts active parameters total parameters communication overhead Switch Transformer paper`; `Mamba retrieval associative recall limits attention language model paper`; `Hugging Face blog mixture of experts explained tokens experts routing beginner`; `DeepSeek MLA explained intuition KV cache compression blog technical`; `large language model architecture claims parameter count FLOPs memory empirical comparison educational notes`; `RETRO Improving language models by retrieving from trillions of tokens original paper`; `ReAct Synergizing Reasoning and Acting external environment paper`; `Titans Learning to Memorize at Test Time paper`。

## 研究发现与适用边界

### 1. 模型报告中的整体收益不能自动归因到一个组件

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, arXiv v5](https://arxiv.org/abs/2405.04434v5)，DeepSeek-AI，2024-06-19。摘要披露该版本为 236B 总参数、每 token 激活 21B、支持 128K context，并同时采用 MLA 与 DeepSeekMoE；作者报告相对 DeepSeek 67B 的训练成本、KV cache 和最大生成吞吐比较。
- **可支持：** 该版本包含哪些被公开的组件；作者在报告所列对照和口径下报告了哪些结果；MLA 被提出为把 KV 历史压入 latent representation 的结构。
- **不能单独支持：** 将整模型的成本、吞吐或能力结果因果归于 MLA 或 MoE 中某一项；把报告的相对结果外推到其他模型、硬件、负载或实现。组合变化与总体结果必须分开表述。

### 2. MoE 的容量和单 token 激活量是不同观察对象

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)，Fedus、Zoph、Shazeer，JMLR 23，2022。论文将 MoE 描述为按输入选择参数的稀疏激活模型，并讨论路由简化、通信成本和训练稳定性。
- **可支持：** 稀疏路由允许总参数与一次计算中选择的专家子集不同；通信和训练稳定性是论文明确处理的工程约束。
- **不能单独支持：** 任意 MoE 相对同质量 dense 模型都更快、更便宜，或“active parameters”直接等于实际延迟、显存占用和跨设备流量。

### 3. SSM 的机制主张要落到输入依赖状态更新及任务证据

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv v2](https://arxiv.org/abs/2312.00752v2)，Gu 与 Dao 等，2023。论文把若干 SSM 参数改为输入相关，使状态随 token 选择性更新；这种时变机制不再等价于固定卷积，论文以硬件感知 scan 处理并行训练，同时保留逐 token 的递归推理路径。
- **可支持：** Mamba 论文所述的选择性机制、scan 实现和指定实验；论文用 selective copy 等合成任务说明内容依赖选择与固定时不变路径的差别。
- **不能单独支持：** 所有递归状态架构都具有相同记忆能力、所有任务都能从固定大小状态中精确取回信息，或线性复杂度必然对应端到端更快。

- [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)，Waleffe 等，2024。研究在相同数据条件下训练 8B 级 Mamba、Mamba-2、Transformer 与 Mamba-2-Hybrid，训练量最高到 3.5T tokens；报告指出结果随任务而异，长上下文自然语言多文档问答、synthetic RULER 与 Phonebook/copying 任务呈现不同对照关系。
- **可支持：** 同一组受控模型/训练条件下，性能判断须绑定任务构造、提示和指标；不能把某一类 NIAH/复制结果等同于所有长上下文使用。
- **不能单独支持：** 对所有 hybrid、规模、训练配方或真实部署 workload 的普遍排名；作者也明确讨论了任务提示对结果的影响。

- [Sparse Long Memory and Transition Bottlenecks Shape Positional Recall in Mamba Language Models](https://aclanthology.org/2026.findings-emnlp.1944/)，Airlangga、AlQuabeh、Nwadike、Inui，Findings of EMNLP 2026。该研究用结构化 recall 任务分析 Mamba 模型中的位置效应，报告短时状态与稀疏长期通道，以及中段信息未进入长期通道的 transition bottleneck。
- **可支持：** 2026 年有针对特定 Mamba 版本/任务的机制分析，表明“递归处理”不等于所有 token 同样保留；任务位置与记忆迁移可成为检验假设的变量。
- **不能单独支持：** 对所有 SSM、所有 Mamba 版本、其他自然语言任务或任意记忆机制作普遍结论。章节使用时须明确这是特定模型和实验切片。

### 4. 检索、可更新记忆和 Agent 交互环不是同一种模型主干变化

- [Improving Language Models by Retrieving from Trillions of Tokens](https://proceedings.mlr.press/v162/borgeaud22a.html)，Borgeaud 等，ICML/PMLR 162，2022。RETRO 通过根据上下文检索文档块，把检索结果接入自回归语言模型。
- **可支持：** 外部语料及检索通路可以作为与模型内部参数/序列 mixer 不同的信息来源，并可被设计为训练时集成的架构路径。
- **不能单独支持：** 将 RETRO 与任意在线 RAG 系统视为相同实现，或由检索接口推断某闭源模型的内部拓扑。

- [Titans: Learning to Memorize at Test Time](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)，Behrouz 等，NeurIPS 2025。论文提出在 test time 更新的 neural long-term memory，并给出 memory-as-context、memory-as-layer、gated-branch 等接入变体。
- **可支持：** “memory”可指具有自身状态更新规则的模型内神经记忆；接入位置与记忆模块本身是不同设计选择。
- **不能单独支持：** 所有 memory 方法共享同一种状态/容量，也不能据论文原型推断该设计在所有规模或线上任务中占优。

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)，Yao 等，2022。论文让同一个冻结语言模型交替生成推理文本、动作并接收外部环境观察；其主要案例通过 Wikipedia API 或交互环境取信息。
- **可支持：** Agentic 行为可以来自模型提示/输出与宿主环境交互构成的闭环；观察到 action loop 不等于模型 backbone 新增了某个专家层或状态空间算子。
- **不能单独支持：** 将 ReAct 的提示与环境实验推广成所有当前 Agent runtime、生产流程或模型内部架构说明。

## 教学组织与实践解释材料

- [Mixture of Experts Explained](https://huggingface.co/blog/moe)，Hugging Face，文章含 2026 年 2 月后续更新入口。其讲解从 dense FFN 的替代对象进入 router 与 token 到 expert 的分派，再连接训练稳定、通信和 serving 资源问题。适合借鉴“对象—动作—代价”的讲解顺序；性能概括须回到原论文和具体测量条件。
- [Build a DeepSeek Model (From Scratch), Chapter 3: The DeepSeek breakthrough: MLA](https://livebook.manning.com/learn/build-a-deepseek-model-from-scratch/chapter-3)，Manning 公开章节说明页。教学设计先建立 KV-cache/MQA/GQA 的内存问题，再追踪 latent cache、当前 token 的 query、重建/吸收路径和位置通路；有助于让抽象低秩表示落到可追踪对象。页面是大纲而非完整章节，不能据此断言教学效果。
- [DeepSeek's Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)，Lior Sinai，2025-02-22。作者用 Julia 实现和推导展开 MLA，并提醒矩阵方向随列主序约定变化；适合发现公式与代码对齐的讲法，所有模型结构和数字仍回到 DeepSeek 报告/固定实现核对。
- [Model Inference Systems](https://auth.opencourse.inf.ed.ac.uk/sites/default/files/2026-10/model_inference_systems.pdf)，Edinburgh 课程讲义，2026-10。MLA 与 tensor parallel 的讲义示例指出，latent cache 共享与 head-parallel 划分可能带来复制成本。此材料用于理解“局部缓存大小不等于整机服务收益”的教学例子；执行细节仅支持讲义所述配置，部署主题由第六册和第二十四册主讲。

## 与本册现行结构的知识关系

- 第四十二章比较不同序列 mixer 的历史表示、查询路径、复杂度、硬件和任务证据；第五十章讨论架构组合的位置、组合收益与开放问题；第五十九章归纳第五十一至五十八章的固定历史模型案例。
- 第六十章的研究对象应是“如何评价一个架构主张及其证据范围”：既不再给序列家族做一张统一排行榜，也不再提出所有模型都适用的 runtime 审计流程。
- 本章可把 MLA、MoE、SSM/递归状态、外部检索/神经记忆、Agent action-observation loop 作为不同设计层级的对照例，说明它们优化的对象不同；各机制完整推导由对应前序章节、第五十九章及第六十一至九十五章承担。
- 未来判断只宜表述为条件化推论。现有研究说明多种结构都可研究，并暴露任务依赖的差别；所收集材料不能证明单一替代架构将成为所有任务的下一代主干。

## 待核边界

- 论文报告的模型规模、训练 token、缓存数字及吞吐数据均保留各论文版本和对照口径，不合并成跨来源排行榜。
- 2026 Findings EMNLP 的 positional recall 结果只绑定论文中所研究的 Mamba 模型、结构化任务和干预设置。
- 本轮未测量或运行任何模型、benchmark、kernel 或 serving system；研究结论均为 source-backed，不是本地复现。
