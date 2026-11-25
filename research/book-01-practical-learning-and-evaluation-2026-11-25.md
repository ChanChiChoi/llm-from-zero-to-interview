# 第一册实战学习路径与系统评估证据

## 研究范围

- **检索日期：** 2026-11-25。
- **问题：** 第一册如何面向刚进入大模型领域的读者，把模型机制、可复核练习、项目判断和工程面试表达连成一条学习路径；企业知识助手示例中的评估主张应如何限定。
- **范围：** 查阅官方入门课程与教程、工程团队公开实践和官方 RAG 评估文档。课程结构仅作为教学组织样本；评估框架资料用于核对各指标测量对象，不作为跨产品性能结论。
- **检索词：** `PyTorch beginner tutorial training workflow`; `Stanford CS336 course assignments language model`; `RAG retrieval evaluation groundedness completeness official`; `AI agent evaluations task outcome transcript official`; `building effective agents evaluation complexity`。

## 来源与支持范围

| 来源 | 本次查阅内容 | 支持的判断 | 不支持的判断 |
| --- | --- | --- | --- |
| PyTorch, [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)；[Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)，文档访问日期 2026-11-25 | 教程顺序涵盖 tensor、数据、模型、autograd、优化、验证/测试和保存/加载；教程声明需要基础 Python 与深度学习概念 | 以可执行的训练工作流串联基础知识，且把前置要求明确写出，是官方入门教程的组织方式 | 不能据此证明没有 Python/深度学习基础的读者无需补课，或该顺序对所有人效果最佳 |
| Stanford CS336, [Language Modeling from Scratch](https://cs336.stanford.edu/)，2026 课程页面，访问日期 2026-11-25 | 课程列出 tokenizer、模型、优化器、小型 LM、profiling、FlashAttention、分布式、数据与对齐作业；要求线性代数、概率、机器学习和 PyTorch 背景 | 机制、数据、性能、训练系统和对齐可以通过实现任务相互连接；该课程本身是高强度进阶样本 | 不能把该课程的先修门槛或计算要求当作本书小白读者的要求，也不能据其目录保证面试表现 |
| Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)，2026-01-09，访问日期 2026-11-25 | 章节描述 task、trial、grader、transcript、outcome、harness；工具型任务可核对环境最终状态；讨论自动评估、线上监控、用户反馈、A/B 测试、人工审查等互补信息 | 评估记录应保留输入、尝试、工具交互和最终结果；模型最后一句话不总能证明任务成功；单一评估层有覆盖边界 | 属于 Anthropic 的工程实践说明，不是独立对照研究，也不是所有项目必须采用的评估规范 |
| Microsoft Foundry, [RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators)，访问日期 2026-11-25 | 文档将 document retrieval 作为过程评估，groundedness、relevance、response completeness 作为系统评估，并注明输入条件和预览状态 | 检索质量、回答相关、证据支持和答案完整性是不同检查目标，应按链路分开观察 | 具体指标名称、服务能力和 API 会变化；该文档不证明其评估器在其他平台或所有任务上的准确性 |
| Anthropic, [Building effective agents](https://www.anthropic.com/research/building-effective-agents)，访问日期 2026-11-25 | 公开讨论 workflow、routing、parallelization、agent、evaluator-optimizer 等模式的任务适配条件及复杂度取舍 | 工程方案应从任务需要和可测收益出发，逐步增加复杂度；模式名称不能代替目标任务的验证 | 作者实践不能直接外推为某模式在所有团队、模型或业务上的性能优势 |

## 本轮采用的教学判断

1. 第一册以贯穿任务作为知识连接器；企业知识助手、设备手册、请求和评估结果均明确标记为教学构造，不呈现为生产案例。
2. 对实战排查，先区分资料接入、检索、提示、生成、校验、界面和服务等阶段，再选择固定输入下的最小检查；不把“直接换模型/重训”设为默认方案。
3. 对 RAG 质量，分别讨论检索、答案相关/完整、证据支持、风险和服务资源，避免让单一总分承担多个不同判断。
4. 面试练习要求读者说明任务、机制、基线、证据、代价、失败边界和个人职责；本书不把面试框架包装为行业通用评分标准，也不承诺仅凭阅读即可通过面试。
5. 课程和学习计划用于安排学习产物，不宣称固定周数、复习间隔或权重已经证明最优。读者可以依基础调整，并以解释、shape/推导、小实现、失败样本和未知变体作为自查证据。

## 未解决边界

- 不同企业的知识权限、资料更新、答案风险、请求负载和人工复核成本差异很大。正文示例用于练习定位方法，实际项目必须重新定义任务契约和验收条件。
- 模型型 judge 只能评估其输入、rubric 和能力覆盖范围内的内容；高风险样本需要独立校准或人工检查。
- 本研究查阅的课程和厂商文档随版本维护。引用处说明当前用途；若未来改变具体接口、评估器或课程安排，应回到官方页面核对。
