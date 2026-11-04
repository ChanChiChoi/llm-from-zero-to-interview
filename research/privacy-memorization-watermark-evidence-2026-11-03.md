# Privacy、Memorization 与 Watermarking 证据笔记

记录日期：2026-11-03。本文只记录本次联网检索并阅读的研究事实、版本与证据边界，不记录书籍审阅状态或正文修改历史。

## 提示线索控制与匹配对照

### Cue-Resistant Memorization

- 来源：[Do LLMs Really Memorize Personally Identifiable Information? Revisiting PII Leakage with a Cue-Controlled Memorization Framework](https://aclanthology.org/2026.acl-long.1560/)，ACL 2026 Long Papers，PDF：[原文](https://aclanthology.org/2026.acl-long.1560.pdf)，页 33841–33861。
- 阅读范围：论文摘要、§3.2 的 overlap cue 与 CRM 指标、§4 的模型/数据设置、§5.1–5.5 的重建和成员推断实验，以及附录中的攻击实现与结果表。
- 支持的事实：研究以提示与目标之间的归一化最长公共子串衡量 lexical cue，并在低 cue 子集上计算重建命中率。实验包含 MGPT3-1.3B、MGPT3-13B 和 MGPT2-560M；这些模型在公开 mC4 语料上训练。重评覆盖 32 种语言。论文报告，重建命中集中在高提示线索条件，严格低线索阈值下多语言命中率接近零；八种成员推断方法的 AUROC 多在 0.50–0.60 区间。
- 证据状态：正式会议论文的作者实验结果（source-backed）。
- 边界：结果绑定公开 mC4 训练模型、PII 形态、论文提示模板、黑盒查询与低 cue 定义；不能据此断言所有 LLM 没有隐私风险，也未覆盖任意 post-training 新增 PII、白盒攻击或特定生产接口。

### 可提取记忆的匹配非成员校准

- 来源：[Extractable Memorization From First Principles](https://arxiv.org/abs/2607.12649)，arXiv v2，2026-09-25。
- 阅读范围：v2 全文 HTML 的摘要、§3 conformal test、§4 population-rate experiments、§5 census experiments、§6 realistic extractability，以及附录 C.3 的 matching 限制。
- 支持的事实：作者主张提取记忆判断需将训练成员序列与匹配的非训练序列比较，以估计自然可预测性基线；提出面向样本总体的 conformal test 和单文档 census。论文报告，在 Wikipedia 上测试 OLMo 2 32B 时，非训练集 10-token 后缀被生成的频率约为训练后缀的 24%，并将其解释为这种短串设置下的误报基线。
- 证据状态：arXiv 预印本（source-backed，未据此做独立复现）。
- 边界：校准依赖成员/非成员匹配质量、分数和抽样假设；“可被生成”“能从现实查询预算中抽取”及“来自目标训练记录”是不同判断。数字只适用于论文中的语料、模型、后缀长度和解码/查询设置。

## 近似重复、输入侧 PII 与训练更新

### Mosaic Memory

- 来源：[The Mosaic Memory of Large Language Models](https://www.nature.com/articles/s41467-026-68603-0)，Nature Communications，2026-01-29。
- 阅读范围：全文 HTML 的摘要、实验主体、Figure 1 对应方法/结果、SlimPajama 分析和方法附录。
- 支持的事实：受控实验向 GPT-Neo 1.3B、Gemma-2B、Phi-2 和 Llama-3.2-1B 加入 100-token 合成 canary 及其 fuzzy duplicates，以 exact-duplicate-equivalent `ρ` 比较影响。对 token 替换实验，替换约 10% token 的 fuzzy duplicate 在不同模型上保留约 `ρ=0.60–0.65` 的影响；替换 50% 时仍报告 `ρ=0.13–0.19`。作者还在已做文档级去重的 SlimPajama 语料中抽样发现 fuzzy duplicates，并指出其序列级去重不能覆盖全部变体。
- 证据状态：同行评审期刊论文中的受控 canary 实验及有限语料抽样。
- 边界：`ρ` 是作者实验定义的相对记忆影响，不是现实 PII 泄露率。语料分析为抽样探索；不能据此声称所有真实语料中的近重复都产生同等模型效应。实验变换包含 token 替换、插入噪声和 n-gram 重排，不应替换成未经验证的具体变换类别。

### 输入侧 PII 的微调记忆

- 来源：[Unintended Memorization of Sensitive Information in Fine-Tuned Language Models](https://aclanthology.org/2026.eacl-long.304/)，EACL 2026 Long Papers，PDF：[原文](https://aclanthology.org/2026.eacl-long.304.pdf)，页 6461–6480。
- 阅读范围：摘要、§2 的输入侧 PII 设置、§3.3 的 greedy/sampling/cross-memorization 评估定义、§4.1 的频次分析与模型结果表。
- 支持的事实：研究了只出现在输入、未进入训练目标的 PII；包含合成多语言数据和德国临床 discharge-summary 数据。论文 Figure 3 在 German Discharge Summary 设置中的 Llama 3.2 1B、seed 2431 结果使用 enhanced true-prefix attack 的 cross-memorization 提取次数，与真实出现频次比较，作者报告 `R²=0.237`。研究还比较语言、任务和模型规模，并测试 DP、unlearning、正则化和偏好优化等缓解路线。
- 证据状态：正式会议论文的作者实验结果（source-backed）。
- 边界：`R²=0.237` 属于指定数据、模型、seed、增强提示攻击和 cross-memorization 统计；不能解释为频次在所有条件下无关，也不能把多个任务或模型下的结论折叠成统一的泄露率。

### Privacy Ripple Effects

- 来源：[Privacy Ripple Effects from Adding or Removing Personal Information in Language Model Training](https://aclanthology.org/2025.findings-acl.959/)，Findings of ACL 2025，PDF：[原文](https://aclanthology.org/2025.findings-acl.959.pdf)，页 18703–18726。
- 阅读范围：摘要、§3 实验设置、§5 assisted memorization、§6 数据增加/移除实验、结论与 limitation。
- 支持的事实：主要受控实验使用 GPT-2 XL，并补充 Llama 3 8B 与 Gemma 2B；数据包括添加 Enron 邮件的 WikiText 设置及 Pile of Law 设置。作者报告相似 PII 在后续训练阶段可能帮助先前记录变得可提取（assisted memorization）；增加 PII 可增加旧 PII 的提取，移除特定 PII 并重训也可能使其他样本可提取。摘要报告 assisted memorization 在其设置中最高约占三分之一。
- 证据状态：正式会议论文中的受控训练/重训实验（source-backed）。
- 边界：研究以邮件地址等公开数据构造实验，绑定其模型、连续训练步骤、解码和抽取协议；不是每次新增或删除记录都会产生二阶泄露的普遍证明。对近似 unlearning 的影响仍是作者指出的后续问题。

### 合成文本发布后的成员推断

- 来源：[The Canary’s Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](https://proceedings.mlr.press/v267/meeus25a.html)，ICML 2025，PMLR 267，页 43557–43580。
- 阅读范围：PMLR 正式论文页摘要及论文实验/方法部分；模型、任务和攻击细节据论文全文核对。
- 支持的事实：攻击者只访问 LLM 发布的合成文本，仍可对微调数据执行成员推断；作者报告其方法在所测设置下高于随机水平。论文指出为模型黑盒 MIA 制作的分布外 canary 不一定适合合成数据审计，并提出任务内分布的 prefix 加高困惑度 suffix 以增加合成输出中的可检测信号。实验使用 Mistral-7B LoRA，并覆盖 SST-2、AG News 和 SNLI。
- 证据状态：正式会议论文的作者实验结果（source-backed）。
- 边界：结果只说明指定生成模型、LoRA、任务数据、canary 与攻击的可行性；不能推广为任意合成语料可识别其训练成员，也不能与模型查询接口型 MIA 视为同一威胁模型。

## 水印与内容来源

### 改写条件下的文本水印证据

- 来源：[AI Watermark Evidence Fails Forensic Readiness: An Empirical Evaluation](https://arxiv.org/abs/2607.16010)，arXiv 预印本，2026-07。
- 阅读范围：预印本 PDF 中的研究问题、KGW / Unigram / SynthID-Text 的特定配置、改写流程和误报/漏报结果。
- 支持的事实：作者报告 846 次有效意义保持改写运行，覆盖 15 个多样提示；在论文条件下，所有最初检出的 KGW 与 Unigram 样本及某个 SynthID 配置的 98.3% 样本在改写后不再检出。其改写前漏报率分别报告为 70%、83%、80%；某一 SynthID 配置对改写人类对照报告 5.4% 误报。
- 证据状态：预印本的配置级实证（source-backed，未独立复现）。
- 边界：数值取决于水印实现、阈值、初始检测条件、提示、改写与“有效运行”定义；不能外推到所有商用水印或一般争议裁定。

### C2PA Technical Specification 2.5

- 来源：[C2PA Technical Specification 2.5](https://spec.c2pa.org/specifications/specifications/2.5/specs/C2PA_Specification.html)，官方规范，版本历史 F.1 标记 `2.5 - October 2026`。
- 阅读范围：规范首页、版本历史 §F.1、manifest/claim、信任模型及验证流程相关章节。
- 支持的事实：2.5 版本增加更多资产格式与 live media 支持，加入 provenance 能力，并澄清 validation、cryptographic 和 content binding 要求。C2PA 验证针对签名声明、manifest、内容绑定和信任链状态。
- 证据状态：规范正文和版本历史（official specification）。
- 边界：有效签名链支持“某身份对来源/编辑作出可验证声明”的结论；不证明声明内容真实，也不自动证明文本作者身份或模型参与程度。
