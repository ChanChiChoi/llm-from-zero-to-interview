# Retrieval-Augmented Architecture 研究记录

## 研究问题与范围

- 日期：2026-11-23。
- 问题：外部语料如何在检索增强生成中形成独立的信息路径；检索可在哪些层次接入生成器；怎样区分检索覆盖、回答忠实度和最终回答质量。
- 范围：第二十一册第四十五章。重点是参数知识、请求上下文与外部语料的关系，检索—生成数据流，提示级和模型内接入方式，按需检索，证据路径失效及评估边界。与第十三章长上下文、第四六章跨请求记忆和第四十九章 Agent 搜索分工。
- 检索词：retrieval-augmented generation/architecture、parametric/non-parametric memory、latent document retrieval、RAG-Sequence/RAG-Token、chunked cross-attention、adaptive/on-demand retrieval、faithfulness、citation quality、RAG evaluation；中文检索检索增强生成、外部证据、模型内检索和引用支持。
- 来源范围：检索原始论文与 ACL Anthology 正式论文页；同时查课程讲义和研究者/工程博客，用于检查读者路径与常见断点。以下结论限于实际打开的论文段落、摘要和页面内容。

## 研究发现

1. **外部语料提供独立于参数的证据路径。** Lewis 等人的 RAG 将预训练 seq2seq 生成器与 Wikipedia 向量索引、神经检索器结合；作者指出外部知识可以直接修订和扩充，但这依赖外部语料及索引本身的维护。该性质不表示检索一定命中、证据一定新或生成一定忠实。
2. **RAG 不只有一种接入位置。** Lewis 等人的模型把检索文档作为潜在变量做 top-K 边缘化；RAG-Sequence 对整段输出使用同一文档，RAG-Token 可为不同输出 token 使用不同文档。REALM 将可学习检索器纳入预训练、微调和推理，检索文档与输入共同进入知识增强编码器。RETRO 按序列块检索邻居，并通过 chunked cross-attention 把检索内容接入自回归 Transformer。三者展示不同的训练耦合和读取路径，不能仅凭名称或单篇论文成绩推导普遍效果。
3. **固定检索与按需检索是不同控制方式。** Self-RAG 训练模型生成 retrieval/reflection tokens，以便决定是否检索、判断检索段落相关性并评价生成内容是否受证据支持。该机制和结果属于论文中的训练与评测设置；它不是所有 RAG 系统已有的通用能力，也不等于 Agent 使用外部搜索工具进行多轮计划和行动。
4. **检索命中不等于回答被证据支持。** Self-RAG 的引言指出，模型可能收到相关段落却未遵循其事实。ALCE 将长文本回答的 fluency、correctness 和 citation quality 分开评估，并以 citation recall/precision 检查主张与引用段落的支持关系。ARES 分别评估 context relevance、answer faithfulness 和 answer relevance，并使用少量人工标注校正自动判断。指标和自动 judge 的有效性仍绑定相应任务、标注及校准条件。
5. **评估应沿处理阶段拆分。** 在有标注证据的任务上，可检查证据是否进入检索结果及最终上下文，再独立检查回答正确性、对给定上下文的忠实度、问题相关性和引用支持。用人工确认的 gold evidence 与实际检索上下文作对照，可以帮助分辨检索损失与生成器的使用失败；ALCE 论文也报告 gold-passage 条件，但 oracle 条件不代表真实检索系统成绩。

## 直接证据来源

### 原始论文与正式论文页

- **Patrick Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks” (2020)**，https://arxiv.org/abs/2005.11401。阅读摘要、引言、§2.1 与 §2.5。支持参数/非参数记忆组合、检索器与生成器的职责，以及 RAG-Sequence/Token 对潜在文档粒度的差异。作者报告的任务结果只适用于论文中的模型、数据和训练设置。
- **Kelvin Guu et al., “REALM: Retrieval-Augmented Language Model Pre-Training” (2020)**，https://arxiv.org/abs/2002.08909。阅读摘要、§1、§3.1–3.3。支持预训练中可学习的知识检索、retrieve-then-predict 表达，以及知识增强编码器将输入和检索文档共同编码。其检索器、训练计算和 Open-QA 结果不代表一般在线 RAG 应用。
- **Sebastian Borgeaud et al., “Improving language models by retrieving from trillions of tokens” (RETRO, 2021 preprint)**，https://arxiv.org/abs/2112.04426。阅读摘要、§1、§2.2–2.4，及 Appendix B.1.3。支持按输入块检索邻居、冻结 BERT 检索器、检索编码器和 chunked cross-attention 的架构描述。论文在特定训练语料、数据库规模和模型配置下报告的结果，不外推至所有模型或在线产品。
- **Akari Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection” (2023)**，https://arxiv.org/abs/2310.11511。阅读摘要、§1 与 §3.1。支持论文中的按需检索、段落相关性判断和生成自我批评机制；不支持将此行为视为普通模型或所有 RAG 管线的默认能力。
- **Tianyu Gao et al., “Enabling Large Language Models to Generate Text with Citations” (ALCE, EMNLP 2023)**，https://arxiv.org/abs/2305.14627。阅读摘要、§1、§3.3、§5.1 和 §5.3。支持将流畅度、正确性和引用质量分开评估，并以 citation recall/precision 衡量声明支持；其自动 NLI 评分和数据集只支持对应实验条件。
- **Jon Saad-Falcon et al., “ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems” (NAACL 2024)**，https://aclanthology.org/2024.naacl-long.20/。阅读正式论文页摘要。支持 context relevance、answer faithfulness、answer relevance 三个评估维度，以及少量人工标注配合 prediction-powered inference 的方法概览；不能把论文报告的评估有效性外推至任意目标语料或 judge。

### 讲解与课程材料

- **CMSC389A, “RAG and Agentic Search” (University of Maryland, Fall 2026)**，https://www.cs.umd.edu/class/fall2026/cmsc389A/RAG-Agentic.html。阅读学习目标、动机、RAG 两阶段流程和 Agentic Search 边界。该课程讲义用于检查从文档规模问题到离线索引、请求时检索和证据送入模型的解释路径；讲义本身不是 RAG 性能或效果证据，且实现细节超出本章范围。
- **Eugene Yan et al., “What We’ve Learned From A Year of Building with LLMs” (2024)**，https://eugeneyan.com/writing/llm-lessons/。阅读 context construction 和 Information Retrieval/RAG 段落。作者的工程观察强调检索片段的相关性、信息密度、细节和上下文组织，可帮助形成失效分析视角；这些实践观察不能单独支持普遍因果或效果主张。

## 证据边界与待核事项

- 不比较 RAG 与长上下文的普遍胜负；不同来源、任务、模型、预算和检索实现会改变结果。
- 不推断检索使答案自动真实、最新、完整或可审计。新鲜度取决于语料源和索引刷新，引用是否成立仍须核对具体声明与来源。
- 不为本章引入具体向量数据库、embedding 模型、分块参数、服务 API、缓存、回滚或生产 SLA；这些既非本章读者结果所必需，也会与检索实现/部署章节职责重叠。
- 未运行本地检索实验、模型、API 或 benchmark。未以这些论文或课程的外部结果声称本书读者学习效果。
