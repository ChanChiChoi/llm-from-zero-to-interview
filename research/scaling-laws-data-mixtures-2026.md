# Scaling law、数据配比与合成数据：来源证据

记录日期：2026-10-28

本记录保存联网核验到的研究结论、来源版本与适用边界；书内审阅决定和修改过程记录在 `progress/`。

## Scaling law 与计算量估算

- Kaplan et al., “Scaling Laws for Neural Language Models,” arXiv:2001.08361，[论文](https://arxiv.org/abs/2001.08361)，[HTML 正文](https://arxiv.org/html/2001.08361)。
  - 实际阅读范围：摘要、§1 的规律总结、§1.3 符号定义、§2.1 Transformer 参数与 FLOPs 估算、§3 的实验范围。
  - 来源支持：论文用 held-out cross-entropy 研究模型非 embedding 参数量、训练数据 token 数和训练计算量的经验幂律关系，并指出规律受其他变量瓶颈限制。该实验主要覆盖 768 至 1.5B 非 embedding 参数、约 2,200 万至 230 亿 token，常用 1,024 token context。其定义的非 embedding 训练计算约为每 token $6N$ FLOPs。
  - 证据等级：source-backed（指定模型、语料、训练流程与指标上的经验拟合和计算估算）。
  - 适用边界：不是所有架构、数据、训练目标或下游能力的普遍定律。$6N$ 估算忽略次要项，依赖论文中的 Transformer、context 和前后向计算假设；真实 FLOPs 应根据模型与实现测量。

- Hoffmann et al., “Training Compute-Optimal Large Language Models,” arXiv:2203.15556，[论文](https://arxiv.org/abs/2203.15556)。
  - 实际阅读范围：摘要与 Chinchilla compute-optimal 模型的说明。
  - 来源支持：作者训练了 400 多个、约 7,000 万至 160 亿参数的模型，使用约 50 亿至 5,000 亿 token；在论文设定中，compute-optimal 时参数量与训练 token 数大致同比例扩展，参数量翻倍时 token 数也应翻倍。Chinchilla 70B/1.4T 的结果用于验证其预算分析。
  - 证据等级：source-backed（论文模型族、数据和预算范围内的实测与拟合）。
  - 适用边界：该比例不应机械外推到新模型族、语料、架构、数据质量或后训练目标；其核心支持共同规划参数量、token 和计算预算。

## 数据 mixture 与质量

- Xie et al., “DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining,” NeurIPS 2023 / arXiv:2305.10429 v4，[论文](https://arxiv.org/abs/2305.10429)。
  - 实际阅读范围：摘要及实验设置/结果概述。
  - 来源支持：DoReMi 使用 280M proxy model 的 Group DRO 域权重来设定 8B 模型的 mixture；在 The Pile 上相对默认比例，few-shot 平均准确率提高 6.5 个百分点，并以 2.6 倍更少训练步数达到基线准确率。
  - 证据等级：source-backed（论文给出的代理优化方法和其 The Pile/GLaM 实验结果）。
  - 适用边界：不是通用数据比例或生产增益保证；结论受代理模型、域划分、模型、数据集和评测任务限制。

- “Scaling Laws for Optimal Data Mixtures,” NeurIPS 2025，[会议论文摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html)，[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/bc1d640f841f752c689aae20b31198c1-Paper-Conference.pdf)。
  - 实际阅读范围：摘要、引言、§§1–2、4–6 及限制讨论。
  - 来源支持：方法拟合目标域 loss 与模型规模、训练 token 和领域权重向量的关系，再为指定目标和预算优化 mixture；实验覆盖 LLM（最大约 7B 参数、150B token）、神经机器翻译和视觉模型。
  - 证据等级：source-backed（论文实验与拟合范围内的结果）。
  - 适用边界：目标主要是预训练/通用 loss，方法假设不重复数据；拟合权重不直接等于下游任务能力最优比例，也不是跨语料通用配方。

- “Datasets, Documents, and Repetitions: The Practicalities of Unequal Data Quality,” NeurIPS 2025，[会议论文摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2aba6ec20299931d46ddeefd5ddcb442-Abstract-Conference.html)，[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/2aba6ec20299931d46ddeefd5ddcb442-Paper-Conference.pdf)。
  - 实际阅读范围：摘要及 §3–4 的数据构造、重复实验、权重衰减设置和结果。
  - 来源支持：在作者设置中，按 FastText 从 RefinedWeb 选出的 top-10% DCLM-baseline 数据最多重复 10 个 epoch；在 1B、3B、7B、12.6B 模型对照中，可胜过单轮训练的约十倍规模 RefinedWeb 数据。随重复次数调整权重衰减（近似按重复次数平方根变化）是该收益的重要 recipe 条件。
  - 证据等级：source-backed（论文给定语料、模型规模、预算和训练配方中的比较）。
  - 适用边界：论文没有证明 DCLM-baseline 总体优于 RefinedWeb；收益取决于目标 token/参数预算、过滤方法、重复次数和优化 recipe，不可推广为普遍重复策略。

- Kang et al., “Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls,” EMNLP 2025，[ACL Anthology 正式论文](https://aclanthology.org/2025.emnlp-main.544/)，[PDF](https://aclanthology.org/2025.emnlp-main.544.pdf)；arXiv:2510.01631 v1 为早期版本。
  - 实际阅读范围：正式论文摘要、§4.1.3、§5 的结果和限制；核对模型、token 预算、单轮训练及合成数据类型。
  - 来源支持：研究覆盖 1,000 多个模型变体和超过 100,000 GPU 小时；在其较大数据预算的实验中，约 1/3 改写文本与 2/3 自然网页文本的混合，可在特定对照中约快 5–10 倍达到相同验证 loss。单独使用改写数据没有显示同等加速；教材式合成数据单独使用在部分下游域退化。
  - 证据等级：source-backed（EMNLP 正式论文所报告的实验和拟合）。
  - 适用边界：结果依赖具体改写方法、模型规模、预算、任务及单轮训练流程；不能将比例或训练加速当作通用配方，其他合成数据类型也不可等同处理。

- Roberts et al., “Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints,” arXiv:2610.31867 v1，提交日期 2026-10-22，[预印本](https://arxiv.org/abs/2610.31867)。
  - 实际阅读范围：摘要与计算受限、数据受限两种设定的说明。
  - 来源支持：在其 compute-limited 设定中，将改写生成计算计入预算的拟合估计显示，改写比例从 $3\times10^{18}$ FLOPs 时低于 10%，升至 $10^{22}$ FLOPs 时约 32%。
  - 证据等级：preprint / source-backed（预印本中的实验、拟合和外推，不是同行评审后的定论）。
  - 适用边界：比例是该模型与数据约束下的估计；数据受限设定使用不同假设。数十倍 FLOPs 的 scaling forecast 属外推，不能当作已验证的生产训练结论。

- Subramanyam et al., “Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining,” ICLR 2026，[会议论文摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html)，arXiv:2510.03313 v2（2026-02-23），[HTML 正文](https://arxiv.org/html/2510.03313v2)。
  - 实际阅读范围：ICLR 官方摘要；arXiv v2 HTML 的摘要、§3 数据质量度量、§4.2 质量感知 scaling law、§5 的实验设置与结果，重点核对 §5.5、表 2—3 及质量估计边界。
  - 来源支持：作者定义无量纲 $Q\in(0,1]$，用 corruption rate proxy 或 deficiency measure 估计，并提出 $L(N,D,Q)=A/N^\alpha+B/(D^\beta Q^\gamma)+E$。研究在 Paracrawl v8 机器翻译和 C4 因果语言建模上控制合成噪声与数据质量；Huber 拟合报告 NMT 的 $\hat\gamma\approx0.173$、CLM 的 $\hat\gamma\approx0.401$，并报告其未见数据评估符合相似的拟合形式。
  - 证据等级：source-backed（ICLR 2026 论文、明确的任务/数据、合成扰动设置和作者拟合实验）。
  - 适用边界：$Q$ 及 $\gamma$ 绑定论文的质量操作化、噪声机制、模型和数据规模；估计值不能视为普遍常数，也不提供任意语料的通用排序分数。作者关于 moderate corruption 的观察限于其受控任务，不能推出所有清洗都能减少模型规模需求。

## 固定数据下的重复训练与集成

- Kim et al., “Pre-training under infinite compute,” ICLR 2026，[官方会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/79682ef54677a2cb3dfb8383b372037f-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文页摘要；未在本记录中复核全文表格。
  - 来源支持：研究固定可用文本、计算较充足的预训练设定；摘要称单纯增加 epoch 和参数的既有数据受限方法会过拟合，而正则化、参数扩展和独立模型集成可改善拟合的 loss 渐近值。作者报告其联合干预在 200M token 数据设定下以较少数据得到更低渐近 loss，并将集成蒸馏到较小 student。
  - 证据等级：source-backed（ICLR 2026 同行评审论文的官方摘要）。
  - 适用边界：这里只记录摘要能够支持的方向性结论；渐近值、下游增益和蒸馏保留比例均属于作者指定模型、语料、训练方案和外推，不能当作任意数据受限训练的配方。

## 生成式评估的 Scaling Law

- Schaeffer et al., “Pretraining Scaling Laws for Generative Evaluations of Language Models,” ICLR 2026，[官方会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/55d7fbeddfd9bb6dace146cc4fa02ff0-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文页摘要；未在本记录中复核全文推导和实验表格。
  - 来源支持：作者研究用预训练计算量、参数量与 token 数、以及 gold solution likelihood 拟合生成任务 pass@k 的三类 scaling law。摘要明确指出 $k$ 会改变拟合参数和预测性，因此计算预算外还须固定每题尝试次数和 verifier 口径。
  - 证据等级：source-backed（ICLR 2026 同行评审论文的官方摘要）。
  - 适用边界：该项工作针对论文中的生成式任务、pass@k 定义与模型规模范围；不能由摘要推断某个 k 在其他任务上最优，也不能将拟合参数跨 benchmark 迁移。

## 数据质量干预与 Scaling Law

- Ardalani et al., “How Text Quality Interventions Reshape Neural Scaling Laws for LLMs: Empirical Study,” ICLR 2026，[官方会议摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6544c0b53cc1ea51d0c7086b536830a8-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方论文页摘要；未在本记录中复核全文实验协议。
  - 来源支持：摘要列出的干预包括去重、启发式过滤和 LLM 引导改写；作者报告用 23 组 QualityPajama 数据训练超过 2,000 个、100M—8B 参数的模型，观察干预会改变拟合系数和指数，且质量排序与 token/parameter 比例依资源设定而变。
  - 证据等级：source-backed（ICLR 2026 同行评审论文的官方摘要）。
  - 适用边界：摘要中的模型、语料和干预集合有限；不能据此得出所有清洗操作都会改变指数、任意数据集的质量排序，或跨任务通用的 token/parameter 比例。

## 重复数据与 benchmark 污染

- Lee et al., “Deduplicating Training Data Makes Language Models Better,” ACL 2022，[论文](https://aclanthology.org/2022.acl-long.577/)。
  - 实际阅读范围：论文摘要和实验结论概述。
  - 来源支持：论文报告其所研究数据集含近重复样例和长重复片段；特定去重实验中，模型未提示时复现训练文本的频率下降约十倍，且标准数据集验证集有超过 4% 与训练数据重叠。
  - 证据等级：source-backed（指定语料、模型和去重方法的实验结果）。
  - 适用边界：不证明所有重复样本都伤害任务，也不证明任意去重策略对所有模型/领域都有效。

- Brown et al., “Language Models are Few-Shot Learners,” arXiv:2005.14165，[论文](https://arxiv.org/abs/2005.14165)，[HTML 正文](https://arxiv.org/html/2005.14165)。
  - 实际阅读范围：§4 及附录 C 的 GPT-3 benchmark overlap 分析。
  - 来源支持：论文用保守 n-gram overlap 构造清洁子集；多数 benchmark 的分数变化很小，但 PIQA、Winograd 等少数结果被标记，部分数据集因无法可靠构造清洁子集而不报告。作者指出清洁子集与原测试集可能存在分布差异。
  - 证据等级：source-backed（GPT-3、其训练数据、检查方法和 benchmark 范围内的分析）。
  - 适用边界：检测到 overlap 不等于模型记住答案或分数必然抬高；检测方法和清洁子集都可能有误报、遗漏及分布偏差。

- Li and Flanigan, “Task Contamination: Language Models May Not Be Few-Shot Anymore,” AAAI 2024 / arXiv:2312.16337 v1，[论文](https://arxiv.org/abs/2312.16337)，[HTML 正文](https://arxiv.org/html/2312.16337)。
  - 实际阅读范围：摘要、§2–6 的方法与结果，尤其是 chronological analysis、training data inspection 和 task example extraction 的召回率讨论。
  - 来源支持：论文指出训练数据检查和任务例提取在其模型/任务上具有较高精度但低召回，时间分析较高召回但精度较低；格式差异、语料是否公开、任务关键字差异和模型行为会限制检查。研究通过多种方法组合，在特定模型与数据集上找到污染证据。
  - 证据等级：source-backed（12 个模型、分类及语义解析任务的论文分析）。
  - 适用边界：发布日期只能作为污染风险线索；论文样本和模型集合有限，单一检测未发现证据不等于不存在污染，也不能据此把任意性能差异归因于污染。

## 生命周期

本记录的 scaling law、data mixture、合成数据、重复数据与污染检测结论为 active 研究入口。具体训练配方、语料来源和许可、生成成本、评估版本及目标任务均须在项目设置中单独核实。
