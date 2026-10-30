# 长上下文 Harness 与 Agent Memory：来源证据

检索与阅读日期：2026-10-29。本记录仅存来源支持的技术事实、版本、阅读范围与证据边界；章节审阅决定、去向和修改状态见 `progress/book-02-advanced-100/architecture-audit.md`。

## 长上下文 Harness 评测

- **来源与版本**：[Pham et al., “LongHarness Bench: Stress-Testing Language Model Harnesses for Long-Context Reasoning,” arXiv:2609.38137v1](https://arxiv.org/abs/2609.38137)，提交于 2026-09-29。
- **阅读范围**：arXiv HTML 摘要、任务构造 §3.1、主要结果 §§4.2–4.5。
- **来源支持的事实**：论文提出四项 long-context harness 评测任务，覆盖多条件检索、跨文档记忆识别、程序执行轨迹及等价程序对搜索；其中 Constraint Solving Search 每个样本约含 120K tokens、160 人的材料和 3–5 个条件。论文比较多个模型与四种 harness，报告同一模型在不同 harness/任务下准确率和费用差异很大、没有一个 harness 在四项任务上都领先；GPT-5.6-sol + mini-swe-agent 的四任务 macro accuracy 为 68%。在 Outlier Memo Detection 上，作者报告 RLM 与 mini-swe-agent 得到相近准确率，但 RLM 每实例估算费用高 12.4 倍。
- **证据等级与边界**：source-backed（预印本的自建 benchmark、作者模型/harness 配置和成本估算）。上述分数、成本和任务难度只适用于论文数据、模型版本、API 价格及 harness 实现；不能用来给生产 Agent 或其他 harness 排名。该工作支持把“同一模型搭配不同长上下文处理器的质量—成本变化”作为独立评测对象。

## RAG 路线：自适应、纠错和分层检索

- **Self-RAG**：[Asai et al., ICLR 2024 paper / arXiv:2310.11511](https://arxiv.org/abs/2310.11511)。阅读 arXiv 与 ICLR 官方摘要。论文用 special reflection tokens 训练模型按需决定检索，并对检索段落和生成结果作反思式判定；结果受论文训练数据、模型和任务限制。反思 token 是模型训练和推理方案，不等于外部独立验证器或通用的事实正确保证。
- **Corrective RAG (CRAG)**：[Yan et al., ICLR 2024 paper / arXiv:2401.15884](https://arxiv.org/abs/2401.15884)。阅读 arXiv 摘要。论文以 retrieval evaluator 估计检索文档质量，并按置信度触发不同检索动作；方案还包括对检索文本做分解和重组，并把大规模 web search 用作静态语料的补充。证据范围限于论文设计和四个数据集实验；真实系统还须校准 evaluator、保证搜索结果的权限/来源质量并测量额外调用成本。
- **RAPTOR**：[Sarthi et al., ICLR 2024 / arXiv:2401.18059](https://arxiv.org/abs/2401.18059)。阅读 arXiv 与 ICLR 官方摘要。论文递归地 embedding、聚类并总结 chunk，形成从原始叶节点到高层摘要的树，推理时可按不同抽象层次检索；作者报告指定多步问答任务上的结果。方法把成本部分移到索引和摘要构建，并可能在上层摘要中压缩掉精确条件；效果需按语料更新频率、精确 span 需求和摘要保真度评估。
- **GraphRAG**：[Edge et al., “From Local to Global: A Graph RAG Approach to Query-Focused Summarization,” arXiv:2404.16130](https://arxiv.org/abs/2404.16130)。阅读 arXiv 摘要。方案从文档抽取实体关系图、构建 community summaries；global query 时先逐社区生成 partial answer，再汇总为最终回答。作者在约百万 token 的数据集级 global sensemaking 任务中报告相对 naive RAG 的收益。结论限于论文语料、问题和实现；图谱/社区摘要有索引构建、刷新和来源回溯成本，不能替代局部精确检索。

上述方法分别改变检索决策、低质量召回的恢复、文档层级表示和全局语料摘要路线；机制不可互换，论文实验也不能合并成跨系统的性能排名。

## Agent 长期记忆的写入、使用和安全

- **AdaMem**：[Chen et al., EMNLP 2026 Industry Track](https://aclanthology.org/2026.emnlp-industry.189/)。阅读 ACL Anthology 论文记录和摘要。AdaMem 将记忆写入建模为 preference-conditioned write control：每种交互情境有自然语言 memory policy，并通过周期反馈更新；AdaMem-Bench 使用 6 个并行 persona 和 5 段 10 周故事。摘要报告，在两种抽取模型与两种反馈模式下，其平均 QA accuracy 从 Mem0 的 80.0% 到 84.35%，persistent memory 减少 9.27%；作者同时报告模型仍不能稳定执行选择性写入政策。数值绑定论文基准、baseline 和配置，不代表生产记忆策略的普遍收益。
- **Memora**：[Uddin et al., Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1337/)。阅读 ACL Anthology 论文记录和摘要。基准覆盖数周至数月的对话，评 remembering、reasoning、recommending，并提出 Forgetting-Aware Memory Accuracy 以惩罚对已过期或失效记忆的依赖；作者测试 4 个 LLM 与 6 个 memory agents，报告常见过期记忆复用和新旧信息协调问题。该摘要不提供跨产品的普遍失效率。
- **Mem2ActBench**：[Shen et al., ACL 2026](https://aclanthology.org/2026.acl-long.370/)。阅读 ACL Anthology 论文记录和摘要。论文评估长期间记忆能否用于具体工具动作和参数补全；其数据构造包含 2,029 条对话 session、平均 12 个 user/assistant/tool turns，并从 memory chains 生成 400 个 tool-use tasks，人工评估中 91.3% 被标为强 memory-dependence。结果绑定该合成/整理语料、任务和系统配置。
- **OP-Bench**：[Hu et al., EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1168/)。阅读 ACL Anthology 论文记录和摘要。论文把 over-personalization 分成 Irrelevance、Repetition、Sycophancy，构造 1,700 个 verified instances，并报告所测模型和 memory-augmentation 方法中常出现无关或过度使用用户记忆。该 benchmark 评估对话个性化适用性，不直接测试所有产品中的隐私授权、删除或动作安全。
- **PS-Bench**：[Guo et al., ACL 2026](https://aclanthology.org/2026.acl-long.1260/)，arXiv:2601.17887。阅读 ACL Anthology 论文记录和摘要。论文提出 intent legitimation：无害个人记忆可能使模型对有害请求推断出良性意图；在多个 memory-augmented frameworks 与 base models 上，摘要报告相对 stateless baselines 的 attack success rate 增幅为 15.8%–243.7%。这是论文测试配置的相对变化范围，不是百分点、通用风险率或所有记忆系统的因果效应。
- **MemoryCD**：[Zhang et al., Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1804/)。阅读 ACL Anthology 论文记录和摘要。其 benchmark 从 Amazon Review 相关行为数据构造跨年、多领域 personalization 任务，摘要列出 14 个 base models、6 种 memory baselines、4 类任务和 12 个领域。该摘要支持其数据与任务设计，不足以证明 benchmark 可代表真实用户日常对话或其他隐私/授权制度。

这些来源分别强调写入策略、过期记忆、行动参数使用、过度个性化、个性化安全以及跨领域历史。数据集和结果不可合并为一个“记忆能力”指标；读摘要所得结论限于上述摘要明确披露的信息。
