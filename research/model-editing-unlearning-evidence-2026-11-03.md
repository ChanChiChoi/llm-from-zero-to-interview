# Model Editing 与 Machine Unlearning：来源证据

检索日期：2026-11-03。

本记录只保存联网核验得到的研究事实、版本和证据边界。章节审阅决定、修改记录和未完成项记入 `progress/`。

## 编辑方法与评测

- **ROME** — Meng et al., “Locating and Editing Factual Associations in GPT,” arXiv:2202.05262v5（2023-01-13），[论文 HTML](https://arxiv.org/html/2202.05262)。实际阅读范围：摘要、引言、因果追踪、§3.1 秩一编辑、zsRE/CounterFact 评测和 §3.7 限制。论文在所测 GPT 类模型上报告中间层 MLP、subject 最后 token 的因果作用，并提出 ROME 修改该层权重；CounterFact 与 zsRE 结果不支持把该位置或编辑成功率外推到所有架构、知识类型和生产模型。
- **MEMIT** — Meng et al., “Mass-Editing Memory in a Transformer,” arXiv:2210.07229v2（2023-08-01），[论文 HTML](https://arxiv.org/html/2210.07229)。实际阅读范围：摘要、方法 §4.1—4.3、规模实验和结论。作者在 GPT-J 6B 与 GPT-NeoX 20B 的指定设置中报告可批量写入数千条关联；方法将更新分摊到若干被选中的 MLP 层。该结果是论文模型、数据和评测条件下的能力证据，不是其他模型的容量保证。
- **MEND** — Mitchell et al., “Fast Model Editing at Scale,” arXiv:2110.11309v2（2022-06-13），[论文 HTML](https://arxiv.org/html/2110.11309)。实际阅读范围：摘要、§2 的编辑目标、§3.1 梯度分解/变换和讨论。论文将全连接层梯度写成 rank-one 因子，并训练小型网络变换梯度；适用性仍依赖编辑任务、元训练分布和被测模型，论文中的大模型结论不能外推为任意架构上的快速编辑保证。
- **SERAC** — Mitchell et al., “Memory-Based Model Editing at Scale,” arXiv:2206.06520v1（2022-06-13），[论文 HTML](https://arxiv.org/html/2206.06520)。实际阅读范围：摘要、编辑范围问题、§3 的 cache / scope classifier / counterfactual model、错误分析与资源限制。系统根据范围分类器判断是否命中 edit memory，命中时调用反事实模型，否则走基础模型；分类错判、额外推理成本和持续增长的记忆是论文明确涉及的限制。论文任务结果不能代表通用知识更新或权限控制。
- **EVOKE** — Zhang et al., “Uncovering Overfitting in Large Language Model Editing,” ICLR 2025，[会议论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html)。官方摘要把目标概率过高、复杂任务泛化变差称为 editing overfit，并介绍 EVOKE 与 Learn the Inference。结论来自作者所测编辑方法和任务，不等于每种编辑都会过拟合。
- **LocFT-BF** — Yang et al., “Fine-tuning Done Right in Model Editing,” ICLR 2026，[会议论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/74b8ccb5ad21871fd436867df19c35ac-Abstract-Conference.html)。官方摘要比较逐样本收敛的 depth-first 流程与 mini-batch、epoch-based breadth-first 流程，并报告 LocFT-BF 在论文模型和数据上达到作者所述结果。用于提示加入合理的微调基线；作者摘要中的比较不支持对所有模型或方法做普遍排序。
- **UniEdit** — Chen et al., “UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models,” NeurIPS 2025，[会议论文页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6cc6c2be42c838827d2f1d12baa05539-Abstract-Datasets_and_Benchmarks_Track.html)。官方摘要说明其覆盖 25 个常见知识领域，用知识图谱邻域采样多跳 ripple tests，并由专有 LLM 将子图转换为自然语言。它扩展了领域与关系组合测试，但生成样本的自然语言质量和覆盖仍受其构造流程约束。

## 遗忘目标、数据边界与评估

- **TOFU** — Maini et al., “TOFU: A Task of Fictitious Unlearning for LLMs,” arXiv:2401.06121v1（2024-01-11），[论文页](https://arxiv.org/abs/2401.06121)。摘要报告 200 个合成作者档案、每档 20 组问答，并指出所研究基线未达到“如同从未用 forget data 训练”的目标。虚构资料降低真实个人资料暴露风险，但不能覆盖真实来源重复、隐私攻击和法律流程。
- **WMDP** — Li et al., “The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning,” arXiv:2403.03218v7（2024-05-15），[论文页](https://arxiv.org/abs/2403.03218)。摘要报告生物、网络和化学安全领域共 3,668 道经敏感信息筛除的选择题，并介绍 RMU。它是危险知识代理指标，不能代替真实任务能力、工具可达性或系统风险控制评估。
- **MUSE** — Shi et al., “MUSE: Machine Unlearning Six-Way Evaluation for Language Models,” arXiv:2407.06460v2（2024-07-14），[论文页](https://arxiv.org/abs/2407.06460)。摘要列出逐字记忆、知识记忆、隐私泄露、效用、大规模删除和连续请求六项要求；其摘要报告在 7B 模型、书籍和新闻设置中比较八种算法。结果受其数据、模型、指标和攻击定义限制。
- **Who’s Harry Potter?** — Eldan & Russin, “Who’s Harry Potter? Approximate Unlearning in LLMs,” arXiv:2310.02238v2（2023-10-04），[论文页](https://arxiv.org/abs/2310.02238)。摘要描述以 Llama 2 7B 和哈利·波特文本为对象的替代标签微调方案及作者报告的通用 benchmark 结果。它是特定版权文本和模型的近似遗忘案例，不足以证明广义数据删除或对所有提示不可提取。
- **AltPO** — Mekala et al., “Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models,” arXiv:2409.13474v3（2024-12-17），[论文页](https://arxiv.org/abs/2409.13474)。摘要将负向反馈与 forget 域正向反馈结合，以减少单纯负反馈导致的不连贯回答；该预印本记录 COLING 2025 论文出版信息。结论应限制在该方法、知识任务和其评估中，不等价于 retrain-equivalence。
- **Context-Aware Unlearning** — Peng et al., “Forget to Know, Remember to Use,” ICML 2026, PMLR 306，[会议论文页](https://proceedings.mlr.press/v306/peng26a.html)。官方摘要报告六种 unlearning 方法在目标内容重新由用户提供到上下文时均出现效用下降，并提出补充目标恢复该上下文效用。上下文使用测量与从参数中回忆、直接泄露测试是不同切片；作者结果绑定其模型和任务设置。
- **信息分解审计** — Goel et al., “Auditing Language Model Unlearning via Information Decomposition,” EACL 2026，[ACL Anthology](https://aclanthology.org/2026.eacl-long.35/)。官方摘要介绍 Partial Information Decomposition 的表示层审计，并报告所测遗忘后内部表征中的线性可解码残留与既有重构攻击敏感性相关。可解码信号不等于已发生外部泄露，也不证明任意攻击者可读出它。
- **SUITE** — Peleg et al., “Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric Generalization Problem,” arXiv:2607.09236v1（2026-07-10），[预印本页](https://arxiv.org/abs/2607.09236)。作者指出 forget 测试需覆盖同一目标的多种表达，而 retain 测试需探测更广的非目标邻域，并提出 SUITE 与 JensUn++。该来源仍是预印本；不能把作者的 benchmark 或模型结果写作通用结论。
- **严格术语立场** — Yoon et al., “Position: The Term ‘Machine Unlearning’ Is Overused in LLMs,” ICML 2026, PMLR 306，[会议论文页](https://proceedings.mlr.press/v306/yoon26g.html)。这是立场论文：作者建议将 machine unlearning 限定为针对明确 forget set、并以不含该集合的重新训练模型为参照的删除；他们主张把拒答、知识编辑和目标抑制另行命名。该文表达作者的术语论证，不代表领域已经形成统一定义。
- **Forget-set curation / CleanSlate** — Jha et al., “What to Forget in Unlearning? Forget Set Curation for Language Models,” arXiv:2608.14855v1（2026-08-14），[论文 HTML](https://arxiv.org/html/2608.14855)。实际阅读范围：摘要、引言、问题定义、实验结论与讨论。CleanSlate 针对歌曲和书籍逐字输出抑制，包含 1970—2025 年 4,616 首 Billboard Hot 100 歌曲、50 本书及 7B—32B 模型的逐模型提取资料。论文报告普通词面/子串检索得到的 forget set 抑制不稳定，而 evaluation-aware 选择可压低目标续写、同时造成非目标内容抑制和模型相关能力损失；这一结果说明 forget-set 构造本身也影响结果。证据是预印本、受特定作品范围与模型设置限制，不是对版权删除义务的法律判断。
- **可逆性审计** — Xu et al., “Unlearning Isn’t Deletion: Investigating Reversibility of Machine Unlearning in LLMs,” ICML 2026, PMLR 306，[会议论文页](https://proceedings.mlr.press/v306/xu26cd.html)。官方摘要报告，任务级指标下看似遗忘的行为在部分所测设置中可由少量微调恢复，并提出 PCA、CKA、Fisher information 等表征漂移分析。恢复性是新增评估维度；具体可恢复性和表征变化依论文方法、数据、模型及恢复来源，不能据此断言所有 unlearning 都可逆。
- **邻近的非 LLM 结果** — Ha et al., “Unlearning’s Blind Spots: Over-Unlearning and Prototypical Relearning Attack,” ICML 2026, PMLR 306，[会议论文页](https://proceedings.mlr.press/v306/ha26b.html)。官方摘要聚焦 class-level unlearning，在 CIFAR、TinyImageNet、CASIA-WebFace 等图像数据中研究忘记类邻域损伤和 prototype-based relearning。它提示“保留邻域”和“后续恢复”也值得关注，但攻击对象、数据模态和机制不支持直接外推为 LLM 结果。

## 分片训练与来源

- **SISA** — Bourtoule et al., “Machine Unlearning,” arXiv:1912.03817v3（2020-12-15；IEEE S&P 2021），[论文页](https://arxiv.org/abs/1912.03817)。摘要介绍通过 Sharded, Isolated, Sliced, Aggregated 训练限制单点数据影响，并在 Purchase、SVHN、ImageNet 分类任务中报告特定重训开销结果。它提供训练前设计删除路径的经典思路，但这些实验不是大语言模型分片训练的直接性能证据。
