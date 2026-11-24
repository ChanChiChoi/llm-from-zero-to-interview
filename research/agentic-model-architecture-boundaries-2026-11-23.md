# Agentic Model Architecture：模型能力与执行闭环

## 研究问题与范围

- 日期：2026-11-23。
- 任务：为第二十一册第 49 章区分 Transformer/语言模型内部结构、工具使用能力的训练或提示路径，以及外部运行闭环；检查 Agent benchmark 中的环境、工具、规则和验收如何影响可归因结论。
- 检索词：`LLM agent architecture model harness environment`、`ReAct prompt-based agent loop`、`Toolformer tool use training API`、`Gorilla API retrieval-aware fine-tuning`、`ToolLLM ICLR tool retriever`、`AgentBench environment evaluation`、`OSWorld execution based evaluation`、`tau-bench user agent policy benchmark`；同步检索大学讲义、课程 notes 与研究者教程。
- 读者任务：读者能够判断一项 Agent 能力发生在模型参数、提示/控制流程、外部工具与运行时中的哪一层，并知道单个 Agent 分数能支持什么结论。
- 范围：单模型工具使用、模型与宿主之间的 action/observation 闭环、外部记忆/检索/验证的接入位置及交互式评测口径。多 Agent 协作、任务规划方法、Agentic RAG、完整 runtime、安全控制与 API 协议只作边界指引，不在本章重复展开。

## 研究发现与证据

### 1. 工具使用可以改变模型训练，也可以发生在提示与控制流程

- **ReAct: Synergizing Reasoning and Acting in Language Models**，Yao et al.；[arXiv 原文](https://arxiv.org/abs/2210.03629)。阅读摘要、§1、方法和附录提示示例。支持“以提示引导冻结模型交替产生语言思考、动作和环境观察”的模式，以及观察如何返回后续决策上下文；该论文的任务和提示设置不证明此模式适合所有 Agent，也不等于新增一种 Transformer block。
- **Toolformer: Language Models Can Teach Themselves to Use Tools**，Schick et al.，NeurIPS 2023；[NeurIPS 正式论文页](https://papers.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)，[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf)。阅读摘要和方法概述。支持通过自监督候选筛选和微调，让语言模型学习何时调用哪些 API、提供参数并把结果用于后续 token 预测；报告的下游提升限于该方法、工具集合和实验。
- **Gorilla: Large Language Model Connected with Massive APIs**，Patil et al.，arXiv 2023；[arXiv 原文](https://arxiv.org/abs/2305.15334)。阅读摘要、§§3—4。支持一种 API 专项微调与检索文档结合的设计，并说明模型能力与测试时 API 文档可共同决定调用表现；API 得分属于作者的 APIBench 数据、测试集和评估程序，不能外推为通用长任务 Agent 能力。
- **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**，Qin et al.，ICLR 2024；[arXiv 原文](https://arxiv.org/abs/2307.16789)，[ICLR 2024 正式论文页](https://proceedings.iclr.cc/paper_files/paper/2024/file/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html)，[正式论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/28e50ee5b72e90b50e7196fde8ea260e-Paper-Conference.pdf)。阅读摘要、工具数据构造、ToolLLaMA/检索器及 ToolEval 相关部分。支持将工具集合、指令数据、模型训练、API 检索与评估器作为共同设计；所用 RapidAPI 工具目录、合成指令和 ToolEval 不代表开放世界 API 覆盖或真实环境任务的普遍能力。

### 2. Agent 证据依赖交互环境和验收方式

- **AgentBench: Evaluating LLMs as Agents**，Liu et al.，ICLR 2024；[OpenReview 正式论文 PDF](https://openreview.net/pdf?id=zAdUB0aCTQ)，[arXiv 页面](https://arxiv.org/abs/2308.03688)。阅读摘要、benchmark 任务与评估描述。包含数据库、网页、操作系统、游戏等多个环境，支持“Agent 评测应明确交互环境与任务协议”；各环境任务及 aggregate score 不能单独证明隐藏的模型模块。
- **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**，Xie et al.，NeurIPS 2024；[NeurIPS 正式论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html)，[论文 PDF](https://papers.nips.cc/paper_files/paper/2024/file/5d413e48f84dc61244b6be550f1cd8f5-Paper-Datasets_and_Benchmarks_Track.pdf)。阅读摘要和任务/环境说明。支持以可初始化真实桌面应用环境、交互任务与执行型评估脚本定义电脑使用 benchmark；数字结论只能对应论文的任务集、初态、模型和执行设置。
- **τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains**，Yao et al.，arXiv 2024；[arXiv 原文](https://arxiv.org/abs/2406.12045)。阅读摘要及 `pass^k` 和数据库终态评估描述。支持同时显式给出模拟用户、多轮交互、领域 API 与策略规则，并以终态核对任务；指标结果受该零售/航空任务、规则和模拟用户设置限定。

以上 benchmark 不是同一任务上的架构对照。它们适合说明模型、提示/控制器、工具、环境、规则、预算与 verifier 共同组成评测配置，不应拼接分数或由行为结果反推私有模型结构。

## 教学与解释来源

- **LLM Powered Autonomous Agents**，Lilian Weng，Lil’Log，2023-06-23；[文章](https://lilianweng.github.io/posts/2023-06-23-agent/)。阅读开篇定义与工具、ReAct、记忆部分。用于观察如何把模型、计划、记忆、工具和环境组织成系统图；文章较早且覆盖面广，不能代替当前模型/框架事实，也不作为本章展开所有 Agent 主题的理由。
- **Lecture 19: LLM agents: tools, loops, harnesses and coding agents**，Carnegie Mellon University，Systems and Toolchains for AI Engineers，2026 秋；[课程 notes](https://kitchingroup.cheme.cmu.edu/f26-06763/lectures/l19/notes.html)。阅读 Agent loop、model request、tool execution、observation 与 harness 说明。用于教学解释模型输出和宿主执行的责任边界；课程中的产品时间线、案例数字和运行经验不作为本章普遍结论。
- **CS224N Lecture 10: RAG and Agents**，Stanford University，2026；[讲义 PDF](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture10-rag-agents.pdf)。阅读 agent architecture 图及 digital/physical environment 分类。用于补充架构层次和环境类型的可视化讲法；图示为概念模型，不是对所有实现的规范。

## 跨册分工和编辑决定

- 第十七册主讲 Agent 闭环、工具调用、ReAct、规划、记忆、Agentic RAG、代码/界面 Agent、Multi-Agent、评估与安全；第二十册主讲 Harness/Runtime 执行、权限、trace、context folding、checkpoint、重放与验收；第二十二册主讲 Function Calling、MCP、A2A、Skill 等协议和生态。
- 第二十一册第 49 章只提供架构层次判断：区分模型参数/提示策略与外部 action-observation 执行环，说明记忆、检索、工具和 verifier 的接入位置，并按 benchmark 配置限制 Agent 能力归因。细节分别链接至以上专题册，不重讲其方法、接口或运行操作。
- 由 ReAct、Toolformer 和 Gorilla 的不同构造可推得：Agent 能力可以部分通过提示、训练、检索和宿主 loop 组合实现，因此不要求一种独立的神经网络 block。此为跨来源架构归纳，不是“所有 Agent 都只由同一种普通 LLM 构成”的普遍定理；专门训练模型、专用视觉/动作模型或更复杂内部结构不在该推论之外。
- 不保留未经来源支持的通用效用、风险、记忆价值或 Agent 总分公式；不保留 toy Python、幂等键/事务/回滚规则和长任务 context folding 操作指南。这些内容不能帮助区分模型架构和外部 Agent 系统层，且与第十七、二十册职责重复。
