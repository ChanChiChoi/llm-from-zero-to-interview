# 推理时计算、实时交互与评估激励：来源证据

记录日期：2026-10-28；补充日期：2026-10-29、2026-10-30

本记录只保存联网发现的知识点及其证据。来源内容的具体适用范围、与书册的映射、正文修改状态和后续安排记录在 progress/。

## 1. 推理时计算的分配

### 来源

- Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar, “Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters,” ICLR 2025；[arXiv 版本](https://arxiv.org/abs/2408.03314)，[ICLR 2025 论文页与 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf)。
  - 实际阅读范围：arXiv 摘要、引言、测试时计算分配方法和实验结果部分；重点核对搜索/verifier、迭代修订和 compute-matched 比较。
  - 来源支持：在论文使用的数学推理任务与模型设置中，测试时计算的收益会随问题难度、策略和基础模型变化；作者提出按 prompt 难度调整预算，并报告相对 Best-of-N 基线的效率收益。论文也报告，在特定 FLOPs 匹配设置下，较小模型加额外推理计算可与约 14 倍参数规模的预训练模型比较。
  - 证据等级：source-backed（同行评审论文的特定实验结果）。
  - 适用边界：结果依赖该论文的 PaLM 2 系模型、数学问题、verifier/修订方法、预算和评价设置；不得外推为所有模型、任务或生产场景中推理计算都优于扩大预训练。
- DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” arXiv v2 revised 2026-01-04; journal reference: Nature 645, 633–638 (2025); [固定 arXiv v2](https://arxiv.org/abs/2501.12948v2)。
  - 实际阅读范围：摘要、引言、R1-Zero 训练方法和作者报告的实验结果。
  - 来源支持：作者报告 DeepSeek-R1-Zero 在没有先做监督微调的设定下直接进行大规模 RL，并观察到回答长度和某些推理行为随训练变化；论文也报告可读性差和语言混杂等问题。
  - 证据等级：source-backed（作者技术报告中的模型特定描述和实验）。
  - 适用边界：这是特定模型与训练方案的作者报告，不构成对所有 RL 训练的结论；论文中的 benchmark 分数、训练过程和推理行为不能证明通用推理能力或自然语言推理过程的忠实性。

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

以下条目在 2026-10-29 通过 ACL Anthology、NeurIPS 和 ICLR 官方论文页核对。实际阅读范围为论文摘要及书目元数据；具体结论只记录摘要明确陈述的设计和作者结果，不将摘要当作全文实验审计。

- Soumadeep Saha et al., “KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning?”, *Transactions of the Association for Computational Linguistics* 14 (2026), 1308–1328, [ACL Anthology](https://aclanthology.org/2026.tacl-1.59/), DOI 10.1162/tacl.a.729。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：作者构建 1,671 道来自 MATH500、GSM8K、AIME 的数学题及自动抽取的 causal CoT graphs，并在 15 个开源权重模型上做图对齐干预分析。
  - 证据等级：source-backed（同行评审论文摘要中的数据、方法与作者概括）。
  - 适用边界：题型、图抽取方式和模型集合特定；不能由图节点干预敏感推出所有自然语言 CoT 都忠实，也不能将该结果视作内部推理状态的完整观测。
- Jingcheng Hu et al., “PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning,” *Proceedings of ACL 2026*, 27200–27215, [ACL Anthology](https://aclanthology.org/2026.acl-long.1253/), DOI 10.18653/v1/2026.acl-long.1253。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：论文摘要描述多轮并行推理轨迹、消息压缩/传递和聚合，并报告通过 outcome-based RL 训练的系统可在固定上下文窗口下扩展有效测试时计算。
  - 证据等级：source-backed（同行评审论文摘要中的方法及作者报告）。
  - 适用边界：摘要中的模型、数学基准、有效 token 预算和上下文管理设置不可与其他系统直接排名；不能推出并行推理普遍优于串行策略。
- Hao Wen et al., “ParaThinker: Native Parallel Reasoning for Scaling LLM Test-Time Compute,” *Findings of EMNLP 2026*, 31265–31285, [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1469/)。
  - 实际阅读范围：摘要与 ACL Anthology 书目元数据。
  - 来源支持：摘要描述单一架构内并行生成多条推理路径并聚合；作者报告 AIME-24 的特定模型/token-budget 对照，以及使用 KV-cache reuse 时的延迟收益。
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
- Hunter Lightman et al., “Let’s Verify Step by Step,” arXiv:2305.20050v1，[论文页](https://arxiv.org/abs/2305.20050)。摘要报告该研究在 MATH 基准上比较 outcome/process supervision，并发布 800,000 条步骤级人工反馈标签 PRM800K。不能外推为所有任务上的过程监督必然优于结果监督。
- Shunyu Yao et al., “Tree of Thoughts: Deliberate Problem Solving with Large Language Models,” arXiv:2305.10601v2，[论文页](https://arxiv.org/abs/2305.10601)。摘要描述以 thought 为单位进行分支探索、回看和回溯；实验覆盖 Game of 24、Creative Writing、Mini Crosswords 等特定任务。效果不由 ToT 名称本身保证。
- Mark Chen et al., “Evaluating Large Language Models Trained on Code,” arXiv:2107.03374v2，[论文 HTML](https://arxiv.org/html/2107.03374)。阅读 §2.1 与 Appendix A：pass@k 以每题生成 n≥k 个候选、统计通过测试数 c，并用 `1−C(n−c,k)/C(n,k)` 估计从 k 个候选中至少找到一个通过解的成功率；作者说明直接经验频率可能有较高方差。估计依赖候选采样和测试 oracle，pass@k 不等于系统无选择器时的 pass@1。
- Chuan Guo et al., “On Calibration of Modern Neural Networks,” arXiv:1706.04599v2，[论文 HTML](https://arxiv.org/html/1706.04599)。阅读 ECE 定义与摘要：ECE 以分桶内准确率和平均置信度差的加权和近似校准误差；论文还研究分类器温度缩放。论文数据来自图像/文档分类任务，不能据此声称 LLM verifier 在目标任务上已校准。
- Percy Liang et al., “Holistic Evaluation of Language Models,” arXiv:2211.09110v2，[论文页](https://arxiv.org/abs/2211.09110)。摘要列出 accuracy、calibration、robustness、fairness、bias、toxicity、efficiency 七项指标，并说明核心场景覆盖同时受可行性和代表性限制。该来源支持多指标评估设计，不提供适用于任意产品的统一得分或完整覆盖保证。
- Lianmin Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” arXiv:2306.05685v4，[论文 HTML](https://arxiv.org/html/2306.05685)。阅读摘要、位置偏差与冗长偏差部分：论文在其模型、提示和开放式对话数据上分析 position、verbosity、自我增强等偏差，并研究 judge 与人类偏好的一致性。实验不能证明当前 judge 或任意领域具有相同偏差幅度或普遍一致率。

## 生命周期

推理时计算：active；具体策略效果仍依赖任务与预算。

CoT 忠实性与分层轨迹安全评估：active；FACT-E、TRACE 等结论绑定各自论文设置。推理轨迹接口风险只支持一般风险意识，不代表闭源产品暴露情况。

潜在空间循环深度推理：active；当前规模化结果属于单个概念验证研究，仍需按统一协议比较计算量、质量和延迟。

实时多模态会话接口：active；此处描述的供应商 API 细节需按版本复核。

评估激励分析：active；需结合具体任务效用、错误代价和 abstention 规则设计指标。
