# 推理时计算、实时交互与评估激励：来源证据

记录日期：2026-10-28；补充日期：2026-10-29、2026-10-30、2026-11-13

本记录只保存联网发现的知识点及其证据。来源内容的具体适用范围、与书册的映射、正文修改状态和后续安排记录在 progress/。

## 1. 推理时计算的分配

### 来源

- Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar, “Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters,” ICLR 2025；[arXiv 版本](https://arxiv.org/abs/2408.03314)，[ICLR 2025 论文页与 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf)。
  - 实际阅读范围：arXiv 摘要、引言、测试时计算分配方法和实验结果部分；重点核对搜索/verifier、迭代修订和 compute-matched 比较。
  - 来源支持：在论文使用的数学推理任务与模型设置中，测试时计算的收益会随问题难度、策略和基础模型变化；作者提出按 prompt 难度调整预算，并报告相对 Best-of-N 基线的效率收益。论文也报告，在特定 FLOPs 匹配设置下，较小模型加额外推理计算可与约 14 倍参数规模的预训练模型比较。
  - 证据等级：source-backed（同行评审论文的特定实验结果）。
  - 适用边界：结果依赖该论文的 PaLM 2 系模型、数学问题、verifier/修订方法、预算和评价设置；不得外推为所有模型、任务或生产场景中推理计算都优于扩大预训练。
- DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” arXiv v2 revised 2026-01-04; journal reference: Nature 645, 633–638 (2025); [固定 arXiv v2](https://arxiv.org/abs/2501.12948v2)。
  - 实际阅读范围：摘要、引言、§2.1 及 §2.2 奖励设计、§§3.1–3.2 的训练阶段、主结果表、附录 B.3.2 冷启动数据说明、E.2/E.4 的数学泛化与推理 token 分析、F 的蒸馏比较、G.2 的 PRM/MCTS 失败分析；官方仓库 README 用于核对公开 checkpoint 摘要。
  - 来源支持：作者报告 DeepSeek-R1-Zero 从 DeepSeek-V3-Base 出发、未先做 SFT 而直接进行大规模 RL；报告中的 R1 流水线则加入数千条 cold-start 样本、推理 RL、拒绝采样、混合推理/非推理 SFT 和第二阶段 RL。作者观察到 R1-Zero 的长推理、反思和自我检查变化，同时报告可读性差、重复与语言混杂。论文称以 R1 生成约 800,000 条样本对开源底模做 SFT 蒸馏，蒸馏阶段未再做 RL。表 3 报告 R1 在 AIME 2024 Pass@1 为 79.8%、MATH-500 为 97.3%、LiveCodeBench CoT Pass@1 为 65.9%；评估上限为 32,768 tokens，采样温度 0.6/top-p 0.95，AIME 每题用 64 个回答估计 Pass@1。附录还记录其大规模 RL 配置下 PRM 的细粒度标注、reward hacking 与开销困难，以及 MCTS 训练搜索空间和值模型的扩展问题；作者仍认可 PRM 在 reranking/搜索、MCTS 在配合预训练 value model 时的特定用途。
  - 证据等级：source-backed（作者技术报告中的模型特定描述和实验）。
  - 适用边界：这是特定模型与训练方案的作者报告，不构成对所有 RL 训练、PRM 或 MCTS 的结论；上述 benchmark 分数绑定论文模型、提示/采样配置、数据集和评测 harness，不是独立复现、一般推理能力或生产成功率，也不能证明可见自然语言推理过程忠实。论文说明 n-gram 去污染无法排除测试题改写暴露。来源 v2 是 2026-01-04 修订版本；作者未公开完整数据筛选、奖励权重和训练实现细节。

- Zhai et al., “Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained Policy Optimization,” arXiv:2604.14853v1（2026-04-16），[论文 HTML](https://arxiv.org/html/2604.14853v1)。
  - 实际阅读范围：摘要、§5.1 的实验设置和 §5.2 的主结果表。
  - 来源支持：作者将每题的推理预算选择写成“最大化期望准确率、约束平均计算量”的问题，并提出先用 Lagrangian 方法构造 per-instance 预算标签，再训练轻量分类器近似选择。实验使用 MATH 与 GSM8K 各 200 道题、DeepSeek-V3、GPT-4o-mini 和 Qwen2.5-7B；每题采集 48 个回答，以非重叠窗口比较若干采样预算。作者在这些设置下报告 learned allocator 优于 fixed、random 和 prompt-length heuristic 基线。
  - 证据等级：preprint / source-backed（预印本中的形式化、实验设计和作者报告结果）。
  - 适用边界：实验是数学任务、特定模型与有限题集；表中的平均预算与实际 realized cost 不总是相同，比较必须同时看成本列。该预印本不能证明其分类器可迁移到其他任务、模型或在线延迟目标，也不能把作者的准确率收益当作生产效果。

### 综合证据边界

推理时计算是可以被分配和评估的推理资源。额外采样、搜索、验证或修订可能改善指定任务的成功率，但收益受问题难度、基础模型、预算和选择器能力影响；额外 token 和计算也带来延迟与成本。该研究方向不应被简化成“让模型多想总会更好”。

## 2. 实时多模态会话与工具事件

### 来源

- OpenAI API 文档，[Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)，访问日期 2026-10-28。
  - 实际阅读范围：Realtime session 组件、会话状态和 function calling 流程。
  - 来源支持：文档描述的 Realtime Session 是客户端与模型间的有状态交互，包含 Session、Conversation 和 Responses；接口事件覆盖音频/文本生成、图像输入和 function calling。文档所述 function calling 流程中，客户端检测调用参数、执行自有代码并回传结果。
  - 证据等级：source-backed（单一供应商的当前产品/API 文档）。
  - 适用边界：只说明该 API 的公开接口合同和组件，不证明其他模型的内部架构或所有实时交互系统均采用同一实现；具体字段和行为会随 API 版本变化。

## 3. 准确率式评估与不确定回答

### 来源

- Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang, “Evaluating large language models for accuracy incentivizes hallucinations,” Nature 653, 1047–1051 (2026), [DOI 页面](https://doi.org/10.1038/s41586-026-10549-w)，在线发表于 2026-04-22。
  - 实际阅读范围：摘要、正文中关于预训练误差、封闭式准确率激励、open-rubric 方案、SimpleQA 实验与 Discussion 的限制说明。
  - 来源支持：论文从学习理论出发分析 next-word 预训练如何在其理想化设定下对不可重复事实产生统计性幻觉压力；又指出把 abstention 计为错误的准确率评分会奖励猜测，并提出把评分规则写进问题的 open-rubric 评估。其示例实验使用四个推理模型和 SimpleQA，并按不同错误惩罚测试模型是否调整 abstention。
  - 证据等级：source-backed（同行评审论文的理论分析与有限实验）。
  - 适用边界：作者明确说明该比较不是跨模型的受控基准，采用默认设置且没有调参或成本归一化；closed-rubric 仍可能具有生态效度，correct/incorrect/abstain 三分法也不能覆盖自然语言中的完整校准表达。结果不能简化为“准确率指标必然让所有模型幻觉”，也不意味着开放 rubric 可消除幻觉。

## 4. Test-time scaling 的推理结构分类

- Mohsen Hariri et al., “Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility,” arXiv:2608.04001v2（2026-08-31），[HTML 论文](https://arxiv.org/html/2608.04001)。
  - 实际阅读范围：摘要、§2.1 的问题定义和三类结构分类；用于核对分类本身，未把其大规模实验数字纳入其他结论。
  - 来源支持：作者将固定自回归模型上的预算化推理区分为单轨迹顺序扩展、完整候选生成后再做终局聚合（leaf-level）和在候选完成前搜索部分前缀（prefix-level）。作者指出这三类不是系统层面互斥的，实际方案可以先做 prefix search，再对完整候选聚合。论文还主张将 checkpoint、提示、decoder、controller/reducer、verifier/judge、停止规则、预算和不确定性作为系统级评估对象。
  - 证据等级：source-backed（预印本中的形式化分类和评估框架；截至记录日不是同行评审结论）。
  - 适用边界：该 taxonomy 是描述预算发生位置和控制结构的分析框架，不是效果排名或保证穷尽所有架构。潜在空间循环深度属于不同的模型架构计算轴，不能强行归入可见 token 的三种输出轨迹类别。

## 5. CoT 步骤忠实性评估

- Yuxi Sun et al., “FACT-E: Causality-Inspired Evaluation for Trustworthy Chain-of-Thought Reasoning,” Findings of ACL 2026, pp. 20275–20293，[ACL Anthology 论文页](https://aclanthology.org/2026.findings-acl.1014/)，DOI 10.18653/v1/2026.findings-acl.1014。
  - 实际阅读范围：摘要、§2 的任务定义、§3.1–3.3 的步骤忠实性与答案一致性设计、§4–5 的评估设置/结果、§7 结论及 Limitations。
  - 来源支持：论文以结构因果模型表述轨迹的步骤间依赖，提出用受控扰动作为工具变量估计 intra-chain faithfulness，并与 CoT-to-answer consistency 一起形成轨迹分数；作者在 GSM8K、MATH-500、CommonsenseQA 上报告筛选推理轨迹、选择 in-context exemplars 和检测带噪轨迹的实验结果。
  - 证据等级：source-backed（同行评审论文中的方法和作者报告结果）。
  - 适用边界：标准 FACT-E 需多次调用语言模型，增加推理成本；论文主要针对自我肯定和 shortcut bias，并对已生成轨迹作事后选择。其特定模型/任务上的结果不能证明内部推理轨迹普遍忠实，也不能作为内部状态的直接观测。

## 6. 推理轨迹暴露的接口风险

- Yu-An Lu et al., “Hidden Thoughts Are Not Secret: Reasoning-Trace Exposure in LLMs,” Findings of EMNLP 2026, pp. 49375–49393，[ACL Anthology 论文页](https://aclanthology.org/2026.findings-emnlp.2303/)。
  - 实际阅读范围：摘要、§6 的评估结果、§8 结论、§9 Limitations 与 §10 Ethical Considerations；记录仅保留风险结论，不复述可操作的轨迹诱取方案。
  - 来源支持：研究考察推理接口隐藏原始轨迹后，提示交互是否仍可能产生有用的用户可见推理文本；作者在所测开源模型、任务和蒸馏设置下报告，暴露文本与相应模型内部轨迹相似，并可保留部分下游学习价值。作者指出仅隐藏可见 CoT 可能不足以保护能力信息。
  - 证据等级：source-backed（同行评审论文的有限模型实验与安全讨论）。
  - 适用边界：论文主要评估开源推理模型，使用轨迹相似度和下游效用，而非因果机制分析；用户可见轨迹可能只是近似，不证明恢复了完整或真实内部状态。闭源服务、其他模型家族及更强隔离措施未被普遍覆盖；不据此对任一产品的暴露率下结论。

## 7. 推理轨迹的分层安全评估

- Zhenyu Wu et al., “TRACE: An Evidence-Grounded Benchmark for Safety Evaluation of Large Reasoning Models,” EMNLP 2026, pp. 26911–26935，[ACL Anthology 论文页](https://aclanthology.org/2026.emnlp-main.1174/)。
  - 实际阅读范围：摘要、§3 的基准构造、§4 的 guardrail 评估结果、§5 结论、Limitations 与 Ethical Considerations。
  - 来源支持：TRACE 分别标注 prompt、推理轨迹与最终回复的安全性，并对判断抽取证据；论文报告基准包含中英文 prompt、九类风险，四个推理模型生成轨迹和回复，另评估 18 个 guardrail。作者发现轨迹安全判断较 prompt/最终回复更难，现有模型提取支持证据仍有困难。
  - 证据等级：source-backed（同行评审论文的基准定义及实验结果）。
  - 适用边界：语种仅含中文和英文，基准模型和样本来自指定数据与构造流程；作者也承认人工标注仍可能有噪声。测试结果不能直接视为生产风险率或跨语种安全保证。

## 8. 潜在空间循环深度的测试时计算

- Jonas Geiping et al., “Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach,” arXiv:2502.05171v2（2025-02-17；NeurIPS 2025 Main Conference），[arXiv HTML](https://arxiv.org/html/2502.05171)，[NeurIPS 论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html)。
  - 实际阅读范围：摘要、§4 训练设定、§5 数学与代码基准结果、§6.1 的自适应测试时计算示例、§10 结论。
  - 来源支持：论文通过重复执行同一循环深度计算块，在潜在状态中增加测试时计算，而非仅增加可见推理 token；概念验证模型为 3.5B 参数，训练 800B token。作者在其数学和代码基准上报告随着循环计算增加的性能变化，并将此路线定位为可补充现有 test-time scaling 方法的研究方向。
  - 证据等级：source-backed（NeurIPS 同行评审论文中的单模型研究与作者报告结果）。
  - 适用边界：论文结论基于一个大规模概念验证训练，作者指出数据混合、训练步数、学习率计划和计算规模仍有限；跨模型基准比较也受训练 token 和模型差异影响。该结果不是与 token 生成、搜索或 verifier 方法在统一协议下的普遍胜负结论。

## 9. 推理轨迹干预、并行计算与检索式推理

以下条目在 2026-10-29 通过 ACL Anthology、NeurIPS 和 ICLR 官方论文页核对；PaCoRe 与 ParaThinker 摘要于 2026-11-13 复核。实际阅读范围为论文摘要及书目元数据；具体结论只记录摘要明确陈述的设计和作者结果，不将摘要当作全文实验审计。

- Soumadeep Saha et al., “KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning?”, *Transactions of the Association for Computational Linguistics* 14 (2026), 1308–1328, [ACL Anthology](https://aclanthology.org/2026.tacl-1.59/), DOI 10.1162/tacl.a.729。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：作者构建 1,671 道来自 MATH500、GSM8K、AIME 的数学题及自动抽取的 causal CoT graphs，并在 15 个开源权重模型上做图对齐干预分析。
  - 证据等级：source-backed（同行评审论文摘要中的数据、方法与作者概括）。
  - 适用边界：题型、图抽取方式和模型集合特定；不能由图节点干预敏感推出所有自然语言 CoT 都忠实，也不能将该结果视作内部推理状态的完整观测。
- Jingcheng Hu et al., “PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning,” *Proceedings of ACL 2026*, 27200–27215, [ACL Anthology](https://aclanthology.org/2026.acl-long.1253/), DOI 10.18653/v1/2026.acl-long.1253。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：摘要描述多轮并行轨迹、把发现压缩为受上下文长度约束的消息并传给下一轮，最终聚合回答；系统通过 outcome-based RL 训练。作者报告其 8B 模型在 HMMT 2025 达到 94.5%，并称使用约 200 万 effective test-time tokens 的设置超过其摘要所列 GPT-5 的 93.2%。
  - 证据等级：source-backed（同行评审论文摘要中的方法及作者报告）。
  - 适用边界：只阅读摘要和元数据，无法在本记录中核验该 GPT-5 对照的完整 prompt、预算和 harness 对齐条件；94.5%/93.2% 是作者摘要报告，不能据此做通用模型排名，也不能推出并行推理普遍优于串行策略。
- Hao Wen et al., “ParaThinker: Native Parallel Reasoning for Scaling LLM Test-Time Compute,” *Findings of EMNLP 2026*, 31265–31285, [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1469/)。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：摘要描述单一架构内并行生成多条推理路径并聚合；作者报告在 1.5B 模型与相同 decoding-token 预算下，AIME-24 准确率相对 sequential-reasoning 基线提高 6.5%–20.7%；启用 KV-cache reuse 时，作者报告并行生成的推理延迟低 38.7%–66.8%。
  - 证据等级：source-backed（同行评审论文摘要中的方法和作者报告结果）。
  - 适用边界：准确率和延迟数字绑定 1.5B 模型、指定基准、同 decoding-token 预算与 KV reuse 条件，不是不同硬件或服务负载下的通用延迟结论。
- Vaibhav Singh et al., “ThinkRetrieve: Retrieval-Augmented Reasoning Traces for Test-Time Scaling,” *Findings of EMNLP 2026*, 36409–36426, [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1703/)。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：作者在每个中间推理步骤检索已求解示例轨迹；摘要报告 1.5B—8B 的五个模型及 GSM8K、MATH-500、AIME 2025、SciQ 设置，并报告特定 AIME 结果的相对增益。
  - 证据等级：source-backed（同行评审论文摘要中的方法和作者结果）。
  - 适用边界：轨迹检索依赖语料质量、任务相似性和数字/条件重新绑定；摘要结果不能证明轨迹可跨领域迁移，也不能替代事实检索或 verifier。
- Negar Arabzadeh et al., “RAG over Thinking Traces Can Improve Reasoning Tasks,” *Proceedings of EMNLP 2026*, 44722–44743, [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1964/)。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：论文摘要研究把生成的 thinking traces 作为检索语料，使用离线结构化表示支持 retrieve-then-generate，并报告数学、代码和科学基准上的作者实验结果。
  - 证据等级：source-backed（同行评审论文摘要中的方法与作者报告结果）。
  - 适用边界：结果绑定轨迹来源、结构化方法、模型与 benchmark；检索到的轨迹仍是候选解法，需核对其前提、数字和步骤，不能当作事实证据或正确性证明。
- Isabelle Lee, Sarah Liaw, and Dani Yogatama, “FOL-Traces: Verified First-Order Logic Reasoning Traces at Scale,” *Findings of EACL 2026*, 2181–2203, [ACL Anthology](https://aclanthology.org/2026.findings-eacl.115/), DOI 10.18653/v1/2026.findings-eacl.115。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：作者介绍程序化核验的一阶逻辑推理轨迹数据集和 masked-operation prediction、step-completion 诊断任务；摘要报告五个推理模型在这些任务上的结果。
  - 证据等级：source-backed（同行评审论文摘要中的数据、任务与作者结果）。
  - 适用边界：形式逻辑任务可以程序核验，不代表自然语言 CoT 的过程标签也能同样可靠地产生；诊断分数只对应该数据和任务构造。

## 10. 合成数据与潜变量推理训练

- Anmol Kabra et al., “Learning from Synthetic Data Improves Multi-hop Reasoning,” *ICLR 2026*, [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e95016eceae77e945b84554b0f1bb49-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文摘要与书目元数据。
  - 来源支持：摘要描述用规则生成、包含虚构知识的合成数据进行多跳推理 RL，并报告其模型在指定真实 QA 基准上的迁移；作者将收益解释为知识组合能力。
  - 证据等级：source-backed（同行评审论文摘要中的实验主张）。
  - 适用边界：结论绑定数据生成规则、训练方法、模型和评测集；不能推成任意合成数据都能提高真实世界推理，也不排除模板匹配等其他解释。
- Yuanfu Wang et al., “Native Reasoning Models: Training Language Models to Reason on Unverifiable Data,” *ICLR 2026*, [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文摘要与书目元数据。
  - 来源支持：摘要描述 Native Reasoning Training 使用普通问答对生成潜在推理轨迹，并将轨迹作为潜变量优化；作者在 Llama、Mistral 系列上报告与 verifier-free 基线的比较。
  - 证据等级：source-backed（同行评审论文摘要中的方法与作者实验主张）。
  - 适用边界：结果受论文模型、数据和训练配方约束；“latent reasoning”不是对隐藏内部状态的直接读取，不证明所有不可验证任务都可仅凭自生成轨迹可靠学习。

## 11. 推理与评估基础来源

以下来源于 2026-10-30 复核；除特别标明的章节外，实际阅读限于 arXiv 摘要，结论按论文任务与模型设置限定。

- Jason Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,” arXiv:2201.11903v6，[论文页](https://arxiv.org/abs/2201.11903)。摘要报告在三种大模型的算术、常识和符号推理任务中使用少样本 CoT 后性能提升。证据说明这些实验任务上的结果，不证明可见步骤忠实，也不保证其他模型或任务稳定受益。
- Karl Cobbe et al., “Training Verifiers to Solve Math Word Problems,” arXiv:2110.14168v2，[论文页](https://arxiv.org/abs/2110.14168)。摘要描述 GSM8K 及通过 verifier 为多条候选排序的测试时选择，并报告论文设置下的收益。结果受数据集、模型和 verifier 训练条件约束。
- Qiying Yu et al., “DAPO: An Open-Source LLM Reinforcement Learning System at Scale,” arXiv v2 revised 2025-05-20，[固定版本](https://arxiv.org/abs/2503.14476v2)。实际阅读范围为 arXiv 书目元数据与摘要；来源支持作者开放训练代码和数据，并报告 Qwen2.5-32B base 在 AIME 2024 上达到 50 分。该结果属于完整数据、系统、采样与算法配方，不能归因给单个组件或外推至其他任务。
- Zichen Liu et al., “Understanding R1-Zero-Like Training: A Critical Perspective,” arXiv v2 revised 2025-10-06，[固定版本](https://arxiv.org/abs/2503.20783v2)。实际阅读范围为 arXiv 书目元数据与摘要；作者分析 GRPO 的回答长度偏差，并提出 Dr.GRPO，报告其在特定 7B 数学设置下达到 AIME 2024 43.3%。结果受底模、奖励、预算、任务和 loss 实现限制，“unbiased”应按论文推导假设理解。
- Hunter Lightman et al., “Let’s Verify Step by Step,” arXiv:2305.20050v1，[论文 HTML](https://arxiv.org/html/2305.20050)及[PRM800K 官方仓库](https://github.com/openai/prm800k)。阅读摘要、§§2–6、相关评估设置与仓库数据说明：论文在其 MATH 实验中报告过程监督显著优于结果监督，PRM 在代表性测试子集上解出 78.2% 的题目；论文同时报告主动学习提高该设置下的标注效率，并发布 800,000 条步骤级人工正确性标签。不能外推为其他任务、标注协议或大规模 RL 系统中的过程监督必然优于结果监督；该仓库于 2026-05-29 归档为只读，仓库状态不改变论文结果的适用边界。
- Shunyu Yao et al., “Tree of Thoughts: Deliberate Problem Solving with Large Language Models,” arXiv:2305.10601v2，[论文页](https://arxiv.org/abs/2305.10601)。摘要描述以 thought 为单位进行分支探索、回看和回溯；实验覆盖 Game of 24、Creative Writing、Mini Crosswords 等特定任务。效果不由 ToT 名称本身保证。
- Mark Chen et al., “Evaluating Large Language Models Trained on Code,” arXiv:2107.03374v2，[论文 HTML](https://arxiv.org/html/2107.03374)。阅读 §2.1 与 Appendix A：pass@k 以每题生成 n≥k 个候选、统计通过测试数 c，并用 `1−C(n−c,k)/C(n,k)` 估计从 k 个候选中至少找到一个通过解的成功率；作者说明直接经验频率可能有较高方差。估计依赖候选采样和测试 oracle，pass@k 不等于系统无选择器时的 pass@1。
- Chuan Guo et al., “On Calibration of Modern Neural Networks,” arXiv:1706.04599v2，[论文 HTML](https://arxiv.org/html/1706.04599)。阅读 ECE 定义与摘要：ECE 以分桶内准确率和平均置信度差的加权和近似校准误差；论文还研究分类器温度缩放。论文数据来自图像/文档分类任务，不能据此声称 LLM verifier 在目标任务上已校准。
- Percy Liang et al., “Holistic Evaluation of Language Models,” arXiv:2211.09110v2，[论文页](https://arxiv.org/abs/2211.09110)。摘要列出 accuracy、calibration、robustness、fairness、bias、toxicity、efficiency 七项指标，并说明核心场景覆盖同时受可行性和代表性限制。该来源支持多指标评估设计，不提供适用于任意产品的统一得分或完整覆盖保证。
- Lianmin Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” arXiv:2306.05685v4，[论文 HTML](https://arxiv.org/html/2306.05685)。阅读摘要、位置偏差与冗长偏差部分：论文在其模型、提示和开放式对话数据上分析 position、verbosity、自我增强等偏差，并研究 judge 与人类偏好的一致性。实验不能证明当前 judge 或任意领域具有相同偏差幅度或普遍一致率。

## 12. Provider thinking controls 与跨轮状态协议

- **来源与版本：** Anthropic [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) 与 [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)，访问日期 2026-11-12；OpenAI [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)，访问日期 2026-11-12。
- **实际阅读范围：** Anthropic Thinking 的各模型 `thinking` 模式支持表、显示/省略与 token 计费说明；Preserved thinking 的 model/account/prefix 检查和失效处理；OpenAI Reasoning 的 `configuration_update` 适用模式、压缩兼容性及 reasoning summary 说明。
- **来源支持的事实：** Anthropic 当前接口按模型区分 adaptive、between-tools、enabled/budget 与 disabled 的可用性；例如 Opus 5.5 默认采用 adaptive thinking，Sonnet 5.5 支持特定 effort 范围内的 `between_tools`，而旧式 `budget_tokens` 并非新模型通用接口。该文档说明返回的 thinking block 是 reasoning summary 而非 raw chain of thought；跨轮和工具调用须按协议回传带签名的 block。Preserved-thinking 检查可绑定此前的 system prompt、工具集合和消息前缀，模型或 account 不兼容时可能丢弃 block；具体不匹配处理按当前账户及参数配置而变。OpenAI 当前指南将 `configuration_update` 限定在支持的 GPT-6+ standard single-agent 模式，并说明 raw reasoning tokens 不公开、summary 需显式请求；该更新与自动 compaction/truncation 和 standalone compact 有兼容性限制。
- **证据等级：** source-backed（两家供应商的当前公开 API 文档）；这些材料支持所列接口合同，不揭示底层模型完整推理状态。
- **适用边界：** 文档按模型、API surface、账号和访问日期变化，不能推导其他 provider 或本地模型使用相同协议。由文档差异可推论：`effort`、`thinking` 类型和 level 标签不是跨模型统一的 token/算力单位，也不保证质量提升；需在目标 endpoint 上按任务、模型和成本/延迟指标实测。本文没有执行真实 API 请求。

### 12.1 续接 API 的当前回放与保留合同

- **复核日期与来源：** 2026-11-13 复核 Anthropic [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)、OpenAI [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state)、[Preserve reasoning](https://developers.openai.com/api/docs/guides/preserved-reasoning)与[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)，以及 Google [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)和[Interactions API 概览](https://ai.google.dev/gemini-api/docs/interactions-overview)。
- **实际阅读范围：** Anthropic 对 reasoning block 的原样回传、system/tools/此前 messages 前缀检查、模型与账号兼容规则；OpenAI 服务端会话引用、无状态完整 output 回放、reasoning.context=all_turns、前缀变化后的 reasoning item 丢弃规则和对象保留期限；Gemini Interactions 的 stateful previous_interaction_id 与 stateless thought-block/signature 回放说明。
- **来源支持的事实：** Anthropic 要求客户端回传原响应 content blocks 及其顺序；在运行前缀检查的模型上，thinking block 受此前 system、tools、messages 以及模型和部分账号绑定约束，发生不兼容时可能请求报错或 block 被丢弃。OpenAI 无状态续接时应保留 response output 中所有项目，包括加密 reasoning items 和 assistant phase；支持 persisted reasoning 的模型可将 reasoning.context 设为 all_turns。OpenAI 当前 preserved-reasoning 文档说明，较新的 frontier models 遇到 reasoning item 前缀变化时会丢弃后续 reasoning item。Conversation state 文档说明 Responses 对象默认保存 30 天、可用 store=false 关闭该保存；Conversation 对象及其 items 不受同一 30 天 TTL。Gemini Interactions stateful 模式通过 previous_interaction_id 由服务端管理 thought blocks/signatures；stateless 模式需按原样回送 thought blocks。
- **证据等级：** source-backed（供应商当前官方 API 文档）；未执行线上 API 请求或账户保留策略实验。
- **适用边界：** 条款绑定准确模型、账号、API surface、当前文档和账户设置。Gemini 结论仅指 Interactions API；不能据此推断所有 Gemini endpoint。provider 状态协议不包含宿主工具执行事实，也不构成公开 raw chain-of-thought。Responses 与 Conversation 的文档级保留语义不能替代实际租户的数据治理条款。

## 13. Provider effort controls 与输出上限的职责边界

- **来源与版本：** OpenAI [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)、Anthropic [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)与[Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)、Google [Gemini thinking](https://ai.google.dev/gemini-api/docs/generate-content/thinking)官方 API 文档，访问日期均为 2026-11-13。
- **实际阅读范围：** OpenAI Reasoning 的 effort 值、reasoning mode、usage token 统计和 `max_output_tokens`；Anthropic Effort 的模型差异、thinking/effort 关系、工具使用和 Extended thinking 的 `budget_tokens`/`max_tokens` 规则；Google Gemini thinking 的模型级 `thinkingLevel` 支持表、`thinkingBudget` 弃用提示、`max_output_tokens` 截断与 thought token 计费段落。
- **来源支持的事实：** OpenAI 文档明确 effort 值及默认值取决于模型；支持的接口可把 `reasoning.mode`（standard/pro）与 `reasoning.effort` 分开设置；reasoning token 计入输出 token，`max_output_tokens` 限制 reasoning、可见输出及格式 token 的总生成量。Anthropic 将 `output_config.effort`、thinking mode 和部分旧式手动 `budget_tokens` 分开；手动 token budget 在支持的模型上是目标预算，`max_tokens` 仍限制总输出，effort 与具体模型、thinking mode 和工具行为有关。Google 文档将 `thinkingLevel` 设为 Gemini 3/4 的模型级枚举（支持值和默认值各异），保留 Gemini 2.5 的 `thinkingBudget`，并指出 Gemini 4 已弃用旧字段；`max_output_tokens` 包含 thought tokens，是执行层硬截断，usage 可按字段报告 thought token。
- **证据等级：** source-backed（供应商官方公开 API 文档；未执行线上请求）。
- **适用边界：** 参数名和枚举属于各自 endpoint/model revision 的协议合同。官方文档没有给出可跨供应商比较的 effort→token 或 effort→质量映射，也不能证明 effort 标签代表固定 GPU 计算量、固定搜索算法或某个质量增益。Google 文档按官方域名跳转至 AI Studio 文档页面；本文只记录其 thinking 参数相关页面内容，不记录其它模型能力主张。

## 14. Provider thinking levels、多 Agent effort 与多模态变体

- **来源与访问日期：** Google [Gemini Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Gemini 3.5 Flash-Lite 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)、[Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)、[Interactions API 概览](https://ai.google.dev/gemini-api/docs/interactions-overview)；xAI [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)、[Grok 4.7](https://docs.x.ai/developers/models/grok-4.7)、[Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent)；Qwen [Qwen3-Omni 技术报告](https://arxiv.org/html/2509.17765)与[官方仓库](https://github.com/QwenLM/Qwen3-Omni)。访问日期均为 2026-11-13。
- **实际阅读范围：** Google Thinking 模型支持表、Interactions 的 `max_output_tokens`、thought usage、stateful/stateless signature 和事件步骤；xAI Reasoning 的模型档位表、Grok 4.7 encrypted reasoning 合同和 Grok 4.20 多 Agent 配置表及 beta 说明；Qwen3-Omni 报告摘要、`2.5 流式与并发、`4.3 Captioner、`5 评估和官方仓库模型下载入口。
- **来源支持的事实：**
  - Google 文档列出 `gemini-3.8-flash` 默认 `medium`、支持 `low/medium/high`，不支持 `minimal`；`gemini-3.5-flash-lite` 默认 `minimal`，支持 `minimal/low/medium/high`。Interactions API 的 `max_output_tokens` 包含 thought tokens，是硬截断上限，不改变 `thinking_level`；思考中触顶会以 `incomplete` 停止，已生成 thought tokens 仍计费。thought step 的 signature 为必需加密状态，summary 可缺省或为空；stateful mode 由服务端管理状态，stateless mode 按合同原样回送 thought blocks。Interaction steps 按时间序列呈现 thoughts、工具调用/结果及 model output。
  - xAI 当前文档列出 Grok 4.7 的 `reasoning.effort` 为 `low/medium/high/xhigh`，默认 high，不能关闭 reasoning；Responses API 对该模型总是返回 `reasoning.encrypted_content`，后续请求需把相关 item 原样回传以延续多轮上下文，Chat Completions 不包含该字段。`grok-4.20-multi-agent` 则将 low/medium 映射为 4 agents，将 high/xhigh 映射为 16 agents。该多 Agent research 功能在文档中标为 beta；公开给用户的主要结果由 leader 汇总，子 Agent 中间状态按文档设置加密返回。多 Agent 与工具消耗会计入 usage/计费。
  - Qwen3-Omni 报告区分 Instruct、Thinking 和 Captioner 等发布变体；报告说明 Captioner 是从 Qwen3-Omni-30B-A3B 微调得到的音频描述变体。报告说明 Thinker 处理多模态输入和文本生成、Talker 生成流式语音，音频 codec 速率为 12.5 Hz，即帧间隔约 80 ms。作者称在冷启动、没有先前上下文时的理论首包延迟为 234 ms；其并发表进一步报告并发 1 时 audio/video 为 234/547 ms，并发 4 时 728/1517 ms，并发 6 时 1172/2284 ms。这些数值绑定 vLLM、torch.compile 与 CUDA Graph 设置，不是生产 p99 或 thinking-level 对照结果。
- **证据等级：** source-backed（供应商官方 API/模型文档；Qwen 作者技术报告与官方模型仓库）。未执行线上 provider API 调用，也未本地复现 Qwen 延迟。
- **适用边界：** 字段语义和支持表依 model ID、API surface、服务版本及访问日期变化。不能由 provider 文档推断未公开的内部推理算法，不能跨 provider 比较同名 level，也不能把输出上限当作 reasoning-only budget。Grok 4.20 的 agent 数合同仅适用于 `grok-4.20-multi-agent`；beta 行为可能变化。Qwen 延迟为作者报告的理论配置数据，不能外推为通用线上服务表现。

## 生命周期

推理时计算：active；具体策略效果仍依赖任务与预算。

CoT 忠实性与分层轨迹安全评估：active；FACT-E、TRACE 等结论绑定各自论文设置。推理轨迹接口风险只支持一般风险意识，不代表闭源产品暴露情况。

潜在空间循环深度推理：active；当前规模化结果属于单个概念验证研究，仍需按统一协议比较计算量、质量和延迟。

实时多模态会话接口：active；此处描述的供应商 API 细节需按版本复核。

Provider effort 与输出 token 上限：active；字段语义和模型支持表需按 API revision 复核。

Provider thinking levels 与多 Agent effort：active；Gemini、Grok 接口按 model ID/API surface 复核，Qwen 延迟只按作者报告条件引用。

评估激励分析：active；需结合具体任务效用、错误代价和 abstention 规则设计指标。

## 2026-11-13 选择性预测、转交、级联与自适应计算论文

- **来源与版本：** Geifman 与 El-Yaniv，[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html)，ICML 2019 / PMLR 97；Madras 等，[Predict Responsibly](https://arxiv.org/abs/1711.06664)，arXiv v3，2018-09-07；Chen 等，[FrugalGPT](https://arxiv.org/abs/2305.05176)，arXiv v1，2023-05-09；Graves，[Adaptive Computation Time](https://arxiv.org/abs/1603.08983)，arXiv v6，2017-02-21。2026-11-13 阅读原论文摘要及相关方法/贡献部分；均为论文证据，不是当前 provider 文档。
- **来源支持的发现：** SelectiveNet 把预测与拒答选择集成到深度网络，直接在目标覆盖子集上优化，并报告分类和回归任务的 risk-coverage 实验。Predict Responsibly 建模“自动预测—外部决策者”的两阶段系统，学习何时把决定交给下游决策者，并在论文设定的模拟场景中评估整体准确率与公平性。FrugalGPT 研究提示调整、模型近似与 LLM API 级联，并在查询预算下适应性地选择 API 组合；其成本/准确率结果绑定 2023 年论文所用模型和数据集。Adaptive Computation Time 允许循环神经网络学习每个序列位置采用多少循环步数，实验覆盖合成序列任务和字符级 Wikipedia 语言模型。
- **证据等级与边界：** `source-backed`（ICML 论文与原始 arXiv 论文）。这些工作可分别支持覆盖率—风险权衡、转交决策、预算内模型级联和条件计算的概念依据；不能证明当前闭源 LLM API 提供校准置信度、effort 标签对应固定算力，或论文效果可迁移至现代推理模型路由。当前产品行为须以 provider 官方接口合同和目标任务测量为准。
