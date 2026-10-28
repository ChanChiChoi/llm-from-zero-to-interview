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

- Kang et al., “Demystifying Synthetic Data in LLM Pre-training: A Systematic Study of Scaling Laws, Benefits, and Pitfalls,” arXiv:2510.01631 v1，提交日期 2025-10-02，[论文](https://arxiv.org/abs/2510.01631)，[v1 正文](https://arxiv.org/html/2510.01631v1)。
  - 实际阅读范围：摘要、§4 实验结果、§5 合成数据比例分析及实验设置。
  - 来源支持：研究报告超过 1,000 个模型变体和超过 100,000 GPU 小时；在较大数据预算下，约 1/3 改写文本与 2/3 自然网页文本相较自然网页基线，可快约 5–10 倍达到相同验证 loss。单独使用改写文本并未更快；单独使用教材式生成文本在多个下游域、尤其小预算下表现更差。
  - 证据等级：source-backed（该研究统一训练和评测设置中的模型/数据实验）。
  - 适用边界：结果依赖其改写方法、生成文本类型、模型规模、数据预算和目标域；不能将约 1/3 或 5–10 倍当成通用配方。改写、教材、程序生成题等合成数据不可视为同一类。

- Roberts et al., “Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints,” arXiv:2610.31867 v1，提交日期 2026-10-22，[预印本](https://arxiv.org/abs/2610.31867)。
  - 实际阅读范围：摘要与计算受限、数据受限两种设定的说明。
  - 来源支持：在其 compute-limited 设定中，将改写生成计算计入预算的拟合估计显示，改写比例从 $3\times10^{18}$ FLOPs 时低于 10%，升至 $10^{22}$ FLOPs 时约 32%。
  - 证据等级：preprint / source-backed（预印本中的实验、拟合和外推，不是同行评审后的定论）。
  - 适用边界：比例是该模型与数据约束下的估计；数据受限设定使用不同假设。数十倍 FLOPs 的 scaling forecast 属外推，不能当作已验证的生产训练结论。

- Subramanyam et al., “Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining,” ICLR 2026，[会议论文摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0c80a0c294a16190c8904b9809c5fba-Abstract-Conference.html)。
  - 实际阅读范围：ICLR 官方摘要。
  - 来源支持：论文引入质量参数 $Q$ 并扩展 loss scaling 形式；用机器翻译和自回归建模中的人工控制噪声实验检验该形式，并报告外样本评估。
  - 证据等级：source-backed（会议论文摘要及其陈述的实验）。
  - 适用边界：质量估计依赖其 corruption/deficiency 定义和噪声设置；不等同于任意语料可直接使用的通用质量分，也不证明清洗总能降低模型规模需求。

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
