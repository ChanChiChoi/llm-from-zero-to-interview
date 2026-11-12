# Tokenizer 评估与词表扩展证据（2026）

本记录收录 tokenizer intrinsic 指标的跨模型预测边界、多语 tokenizer 受控比较、模块化词表方案，以及预训练模型扩展词表的 embedding / continued pre-training 证据。结论状态为 `source-backed`：以下事实来自指定论文版本，不表示已在本地复现。

## 跨模型指标与多语受控比较

### No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance

- 来源：ACL Anthology，Findings of EMNLP 2026，正式会议论文。<https://aclanthology.org/2026.findings-emnlp.96/>。
- 阅读范围：论文页面摘要、正文和实验/限制段落；研究约 200 个模型、九种语言和多个 benchmark。
- 来源支持：作者没有发现一个 tokenizer intrinsic 特征能跨模型家族、任务和语言稳定预测下游模型表现；论文提出新的语言学指标能捕获既有指标未表达的方面，但预测力仍会依赖模型家族和语言。
- 证据边界：这是该模型集合和任务覆盖内的分析，不能证明所有 tokenizer 指标在每个任务都无用，也不能据此为新模型做免训练排名。

### TokLens: A Multilingual Lens on Tokenizer Quality for LLMs

- 来源：ACL 2026 Student Research Workshop 正式论文。<https://aclanthology.org/2026.acl-srw.18/>。
- 阅读范围：ACL Anthology 摘要、实验范围和指标描述；24 个 tokenizer、15 种语言、六种 intrinsic 指标。
- 来源支持：在控制模型规模后，英语 benchmark 未见 tokenizer intrinsic 指标的稳定预测关系；在其多语 MMLU-ProX 分析中，部分 tokenizer 指标与逐语言成绩有关联，Qwen2.5 系列实验也报告单 token 保留率与语言扩展斜率的相关性。
- 证据边界：多语相关性可能受预训练数据构成混杂；多语分析只覆盖论文中可比较的有限模型、任务和语言。它不能建立因果关系，也不构成跨模型的通用选型规则。

### Equity with Efficiency: An Empirical Study of Tokenizers for Multilingual Large Language Models

- 来源：ACL Anthology，EMNLP 2026 Main Conference 正式论文。<https://aclanthology.org/2026.emnlp-main.1783/>。
- 阅读范围：论文页面摘要及受控训练设置说明。
- 来源支持：作者在同一数据条件下训练 1.5B 参数模型，并覆盖 11 种东南亚语言。摘要报告 parity-aware BPE 位于 tokenizer 效率与跨语言公平性的 Pareto 前沿；形态驱动的 byte encoding 在部分语义任务表现更好，但计算成本更高。
- 证据边界：模型规模、语种和任务集合有限；该对照不决定其他语言组合、模型规模或硬件上的最优方案。

### To Each Language Its Tokenizer: Modular Tokenizers for Efficient Multilingual LLMs

- 来源：ACL Anthology，Findings of EMNLP 2026。<https://aclanthology.org/2026.findings-emnlp.1019/>。
- 阅读范围：论文页面摘要与框架描述。
- 来源支持：作者提出模块化 BPE/Unigram tokenizer、语言子词表抽取，以及按 batch 使用相关词表子集的预训练方案。
- 证据边界：摘要描述的是论文提出并评估的模块化路线，不足以证明任意生产服务都能在未知语言条件下动态切词或减少端到端成本；部署侧需单独核验语言路由、合并词表和输出头实现。

## Unicode 与输入表示

### From Bytes to Subwords: Challenges of Input Representations in NLP

- 来源：ACL Anthology，Findings of ACL 2026。<https://aclanthology.org/2026.findings-acl.530/>。
- 阅读范围：ACL Anthology 页面及 PDF 第 3 节语言识别实验、Figure 4 和后续讨论。该文为输入表示综述，并讨论 Unicode normalization 与字节/子词表示。
- 来源支持：文中复用 van der Goot (2025) 的 2,307 语言、每种 1,000 样本设置，以 1–5 字符 n-gram Naive Bayes 比较 NFC 与原文；十个 seed split 中原文输入都更高，均值为 91.75% 对 91.70%。论文还讨论 UTF-8 表示的效率/跨脚本公平问题，并给出替代表示的概念验证。
- 证据边界：实验任务是语言识别，模型为指定 Naive Bayes 配置；微小优势不能外推为 LLM 预训练质量或通用规范化政策，其他任务也可能从一致规范化获益。

## 预训练模型的词表扩展

### Beyond Initialization Loss: A Systematic Study of Token Embedding Initialization Strategies for LLM Vocabulary Extension

- 来源：arXiv:2608.03494v1，2026-08-04 版本。<https://arxiv.org/html/2608.03494v1>。
- 阅读范围：版本页、实验设置、embedding composition、短程 CPT 和限制章节。
- 来源支持：研究针对 Nemotron-3-Nano-30B-A3B 的 Hindi 扩词表，加入 25,600 个 Hindi token；在 100,000 篇 Sangraha Hindi 文档上，作者报告 fertility 从 1.95 降至 1.25 tokens/word。论文比较 20 多种初始化配置，并报告该设置中 50-step（约 0.42B tokens）CPT probe 比 step-zero initialization loss 更能预测后续收敛；最佳观察配置对输入与输出 embedding 使用不同的子词组合与校准。
- 证据边界：一个 Hindi 扩词表、一个模型和特定 CPT 数据/预算；50-step probe 的收益需要在其他语言、模型、数据混合和评测任务验证。初始化比较不等价于 tokenizer 构造方法的普遍比较。

### In-Place Tokenizer Expansion for Pre-trained LLMs

- 来源：arXiv:2607.15232v1，2026-07-16 版本。<https://arxiv.org/html/2607.15232>。
- 阅读范围：tokenizer construction、embedding initialization、训练 recipe、语言覆盖、on-device latency、ablation、limitations 及 Appendix D。
- 来源支持：作者在掌握源 tokenizer 设计时延续原 BPE merge ranks，保留旧 token 映射，新 token 以源子词 embedding 均值初始化；LFM2-8B-A1B 案例先进行 embedding-only adaptation，再做全模型继续预训练。作者报告 65K→128K 后 Hindi/Vietnamese/Thai token 数分别约减少 2.4×/2.6×/4.0×；阶段预算包括 600B embedding-only tokens 和 400B continued-pre-training tokens。每字符 2.2×–3.7× decode speedup 是将语言压缩比例与设备每 token decode 吞吐合成的估计；Appendix D 中同模型体在 Q4_0、batch 1、128 个生成 token 的 65K→128K 对照，M4 Max CPU/GPU 和 Snapdragon 8 Elite Gen 5 的 per-token decode 分别下降 8.6%、7.4%、9.4%。
- 证据边界：仅覆盖所述 LFM2 家族、可控的 continued-BPE 构造、训练阶段和指定设备；合成 per-character speedup 不是所有设备上的直接端到端测量，不能外推到第三方 tokenizer 替换。论文还报告了原支持语言回归和不同数据混合的消融，部署/质量结论应连同该范围读取。


## Tokenizer 选择对逐语言语言建模指标的影响

- 来源：Meister、Altıntaş 与 Bosselut，[Language-Specific Effects of Tokenizer Choice in Multilingual Language Models](https://arxiv.org/abs/2610.12144v1)，arXiv v1，2026-10-08。
- 实际阅读范围：摘要与引言；§2.1–2.3 的训练和评估设置；§3–5 的逐语言 BPB 结果与设计消融；§8 limitations。
- 来源支持：作者比较 54 个 tokenizer 与对应语言模型，在同一 decoder-only nanochat d24 架构、约 1.27B 参数、同一模型训练配方和约 9.2B token 预算下评估 31 个训练语言；主要结果以 FLORES+ 的逐语言 bits-per-byte（BPB）报告。论文的主 tokenizer panel 大多每个 tokenizer 只有一次训练；固定 token step 预算时，各 tokenizer 因压缩率不同而见到的底层训练字节数也不同。54 个 tokenizer 间的逐语言 BPB spread 与语言模型训练数据占比负相关（Spearman rho=-0.52，p=0.003）；在排除无空格分词语言、按 tokenizer family 聚合等替代分析下方向仍存在。
- 证据等级：preprint / source-backed（论文 v1 的指定架构、数据、token 预算与评测）。
- 适用边界：这是 BPB 等语言建模结果，不直接证明下游任务能力差异；单一模型架构、训练 recipe 和约 128K vocabulary 限定外推范围。训练数据占比与书写系统在其 31 语言样本中混杂，观察到的相关性不证明低数据占比会造成 tokenizer 敏感性。BPB 跨语言也受 UTF-8 字节长度影响，作者并报告 bits-per-sentence 对照。

## 跨研究比较与未解决问题

“No Single”检验 intrinsic 指标能否跨模型家族、任务与语言预测下游表现；TokLens 报告特定多语 benchmark/模型族中的相关信号。二者的覆盖与分析条件不同，不能直接合并成一个互相否定的结论。它们共同支持把静态指标作为诊断线索，并将目标任务上的受控训练视为能力比较证据。

仍需验证的问题包括：更大模型和更多训练预算下哪些指标具有预测力；在相同架构、原文和 FLOPs 下扩词表的质量/成本边界；模块化词表在未知或混合语言输入上的路由与部署行为；以及不同设备上的真实端到端延迟。现有 source-backed 结果不足以给出跨语言、模型规模和硬件通用的 tokenizer 排名。
