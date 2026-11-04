# Mechanistic Interpretability 与 SAE 评估证据

检索日期：2026-11-03。资料以原始论文、正式会议论文集和作者公开技术稿为主；下列数字只适用于所列实验对象、模型、指标与配置。

## SAE 表示与评估

### Gated Sparse Autoencoders

- **来源：** Rajamanoharan et al., *Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders*, NeurIPS 2024 论文 [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/01772a8b0420baec00c4d59fe2fbace6-Paper-Conference.pdf)。阅读摘要与方法部分关于 L1 shrinkage 和 gating 的说明。
- **证据支持：** 标准 SAE 的 L1 稀疏惩罚会系统性压低 feature activation。Gated SAE 将“选择哪些方向”与“估计激活幅度”分开，并主要对前者施加稀疏惩罚；作者在其最高 7B 模型和常用超参数区间报告 shrinkage 缓解、可解释性相近及相同重构保真度下激活 feature 数减少。
- **边界：** 这是所测模型和超参数下的 SAE 架构结果，不代表 feature 已被证明语义正确、因果完整或适用于安全监控。

### SAEBench

- **来源：** Karvonen et al., *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*, arXiv v4，2025-06-04，[论文](https://arxiv.org/abs/2503.09532)。阅读摘要、§1、§3.1、§4.1 及附录实验设置。
- **证据支持：** 论文组织八项评估，覆盖 concept detection、interpretability、reconstruction 与 feature disentanglement；基准包含七种 SAE 架构和 200 多个 SAE。论文在 Gemma-2-2B、Pythia-160M 与 Gemma-Scope 补充配置上比较发现，sparsity–fidelity 排名不能稳定推出其他 concept detection、disentanglement 或下游指标排名。
- **边界：** 指标、任务和所测 SAE 集合具有特定模型与数据范围；基准提供多维评估，不构成一个跨任务 SAE 总排名。

### MIB

- **来源：** Mueller et al., *MIB: A Mechanistic Interpretability Benchmark*, ICML 2025, PMLR 267，[论文](https://proceedings.mlr.press/v267/mueller25a.html)。阅读正式论文摘要。
- **证据支持：** MIB 分为 circuit localization 与 causal-variable localization 两条 track，覆盖四项任务和五个模型。所报实验中，attribution 与 mask optimization 在 circuit localization 上表现较好；监督式 DAS 在 causal-variable localization 上表现较好；该 track 中 SAE feature 未优于普通 hidden dimensions。
- **边界：** 这是该基准任务、模型和指标下的比较，不构成所有机制分析问题上的通用方法排序。

### SANITY Checks for Sparse Autoencoders

- **来源：** Korznikov et al., *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?*, arXiv v1，2026-02-15，[论文](https://arxiv.org/abs/2602.14111v1)。阅读摘要、合成 ground-truth 实验、真实激活随机基线、讨论和 §5 limitations。
- **证据支持：** 在该论文的独立、重尾合成 feature 设置中，BatchTopK 的 explained variance 为 71%，但恢复 3,200 个真 feature 中 297 个（约 9%）；JumpReLU 恢复约 7%。在其所选真实模型激活与评估任务中，随机化 SAE 基线与训练 SAE 的 interpretability 分数分别为 0.87/0.90，sparse probing 为 0.69/0.72，causal editing 为 0.73/0.72。
- **边界：** 合成实验假定 feature 激活相互独立；论文聚焦标准 SAE，没有评估 transcoder/crosscoder。作者将结果定位为对当前评估和随机基线检查的挑战，并明确说这不是对 SAE 范式的最终判决。

### Rethinking SAE Evaluation / DAME

- **来源：** Zhang & Wang, *Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy*, arXiv v2，2026-10-23；页面标注接收 NeurIPS 2026，[论文](https://arxiv.org/abs/2610.30474v2)。阅读摘要、§3.1–3.3、§4.1–4.2、附录 B limitations。
- **证据支持：** 论文在 600 多个 SAE、七个 SAE 家族及 28,000 多个 feature 上研究 dominance 分层与解释性。DAME 按平均 dominance 分层抽样，并用 LLM judge 对低、中、高 activation 样本的 activation 强度排序。最高平均 dominance 四分位 Q1 的 perfect-ranking accuracy 为 61.8%，其余三个四分位为 64.1%；65.2% 的 SAE 在 Q1 显示分数缺口。作者另报告 Q1 冗余连接富集度中位数为 1.64×，指标阈值为 decoder cosine similarity 0.7。
- **边界：** 单义性评分仍依赖 LLM judge，作者称其为 proxy。主基准模型主要在 70M–2.7B；较大模型的补充分析范围有限，不能推出规模不变性或所有模型都存在同样的冗余差异。

### Sign-Aware Gated SAE

- **来源：** Wieciech et al., *Sign-Aware Gated Sparse Autoencoders: Modeling Anticorrelated Features with Bi-Jump-ReLU Activations*, arXiv v3，2026-08-04，[论文](https://arxiv.org/abs/2605.28149v3)。阅读摘要、结果与 §5 limitations。
- **证据支持：** SA-GSAE 使用双侧 gate 和有符号幅度，让一个 latent/decoder 方向表示正负两侧。半宽配置在 Pythia-1B、SmolLM3-3B 的三个中层 hook points 与三随机种子的六个设置中，于固定 $L_0=64$ 相较双宽 Gated SAE 的 dead-feature fraction 绝对值低 0.35–0.82；$R^2$ 差距不超过 0.025。全宽配置在一个 SmolLM3-3B residual-stream 设置出现重构崩塌。
- **边界：** 该稿为预印本；没有覆盖 instruction-tuned backbone、超过 3B 参数模型或更深层/长上下文系统。作者说明半宽配置及 threshold reset 等训练条件重要；结果不能单独证明 feature 的人类语义或安全效用。

### MechELK：潜在知识诱发

- **来源：** Park et al., *MechELK: A Mechanistic Interpretability Framework for Eliciting Latent Knowledge in Large Language Models*, arXiv:2605.28825v1，[论文](https://arxiv.org/abs/2605.28825)。阅读摘要、§3 方法、§4.1—4.4 实验与失败分析。当前页面将提交日期写为 2026-04-07，但论文编号属于 2605，页面浏览上下文指向 2026-05；日期元数据内部不一致，引用时不将具体日作为已确认事实。
- **作者报告的机制：** 三阶段流程用 SAE 特征差异提出候选方向、用 activation patching 选层并计算 Causal Knowledge Score (CKS)，再通过推理时 residual-stream 干预尝试诱发目标答案。实验对象列为 Llama-3-8B、Llama-3-70B、Mistral-7B-v0.3；数据包含 TruthfulQA MC1、Quirky LM 和作者整理的 400 个 Deceptive Alignment Benchmark 场景。
- **需保留的报告范围：** 论文报告 Llama-3-8B 上 detection rate 91.4%、false positive rate 12.7%，并在表 1 列出六个模型/数据单元的 elicitation accuracy。正文所说的跨设置平均值与表 1 数值算术平均不一致：表中 MechELK 六值均值约 83.38%，CCS 约 73.98%，直接 probing 约 68.38%，而摘要/正文分别报告 84.7%、78.5%、75.6%。此处仅能记为作者报告，不能据此复述汇总差值。
- **证据边界：** 当前为 arXiv v1 预印本，未见独立复现；DAB 是作者整理的场景集。线性读出、候选答案提示、探针/patching 与干预成功最多支持给定模型和任务上的表示/行为证据，不能推出模型具有人类意义的隐藏信念、欺骗意图或可部署的安全检测能力。方法定义、数据构造、指标分母与表格/文字统计冲突均需后续核验。

### Revising and Falsifying SAE Feature Explanations

- **来源：** Ma, Pfrommer & Sojoudi, *Revising and Falsifying Sparse Autoencoder Feature Explanations*, NeurIPS 2025，[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b6ce2529a79862a2d8ac57ac0e01d90-Abstract-Conference.html)。阅读正式会议论文摘要。
- **证据支持：** 论文以相似度搜索构造近邻负例，用结构化解释与树状迭代改进自动 feature explanation，并报告其解释质量改善及对既有解释方法偏差的揭示。
- **边界：** 自动解释仍是需要反例检验的假设；该研究不证明自然语言标签就是 feature 的唯一语义。

## 因果机制与安全用途

### LLM Forensics

- **来源：** Antoun et al., *LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders*, Findings of EMNLP 2026，ACL Anthology 版本，[论文](https://aclanthology.org/2026.findings-emnlp.2418/)；PDF 阅读摘要、§3.3–3.5、§4.3–4.4 与 limitations。
- **证据支持：** 作者在 Gaperon-1B/8B 的无害语言切换 trigger 中，把 SAE feature 的 trigger-vs-control detection 与干预控制分开测试。部分 attention/MLP features 有高 detector F1，却不能有效抑制或诱发切换；特定 residual-stream features 在某些配置中显示抑制/诱发效果。作者把机制角色概括为 trigger recognizer、residual-stream routing handle 和输出语言追踪。
- **边界：** 实验任务是可自动测量的英译法/德语言切换，不是恶意 payload、工具调用或生产后门。最强经验结论限于 1B 与 8B；24B SAE 训练不可靠，相关干预结果仍属初步。作者支持“角色区分可能迁移”的研究假设，不支持 feature/layer 直接迁移。

### Formal Mechanistic Interpretability

- **来源：** Hadad, Katz & Bassan, *Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees*, ICLR 2026，[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html)。阅读摘要、实验说明与 §7 limitations。
- **证据支持：** 论文研究输入域连续扰动下的 circuit robustness、连续 patching perturbation 下的 robust patching，以及多种 minimality 保证；实验用 α–β-CROWN verifier 在视觉模型上比较 sampling-based circuit discovery。
- **边界：** 论文明确依赖神经网络 verification queries，当前 verifier 对 state-of-the-art 模型仍有扩展性限制；实验不是对前沿语言模型或部署级安全性的保证。
