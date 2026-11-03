# 领域后训练案例与证据边界

> 本文件只记录联网核对的新知识及来源证据，不记录章节审阅、修改决定或工作进度。

## FINDAP / Llama-Fin

- **来源与版本：** Ke et al., *Demystifying Domain-adaptive Post-training for Financial LLMs*, EMNLP 2025, ACL Anthology 2025.emnlp-main.1579；[论文与 PDF](https://aclanthology.org/2025.emnlp-main.1579/)；2025 年会议论文。
- **阅读位置：** 摘要、§§1、3.1—3.4、4.1、结论和 Limitations。
- **来源支持的事实：** FINDAP 将金融领域能力定义为概念、领域任务、推理和指令跟随；FinRec 联合继续预训练（CPT）与指令微调（IT），随后进行偏好对齐（PA），并以最终答案偏好（FAP）和逐步纠正偏好（SCP）构造结果与过程信号。方案还包括 FinTrain 训练数据和对齐能力框架的 FinEval。实验基于 Llama 3 8B Instruct；作者报告在 17 项新任务中有 13 项优于底模，并称一般知识与指令跟随整体保持有竞争力。
- **证据等级：** 同行评审论文中的作者实验报告；不是跨机构独立复现或跨模型家族的普遍结论。
- **适用边界：** 作者指出未见任务表现仍需提升，完整数据配方依赖成本较高的经验性实验，同一 recipe 未证明能迁移到其他模型家族，且本文四类能力没有覆盖多模态和敏感性等领域要求。逐任务结果并非全部领先，不能简化为所有金融任务均改善。
- **待核验事项：** 在其他底模、不同机构和独立评测集上复现；进一步拆分过程偏好信号对结果的贡献。

## 生成式 ICD 编码的分阶段后训练

- **来源与版本：** Wang et al., *Can Post-Training Turn LLMs into Good Medical Coders? An Empirical Study of Generative ICD Coding*, arXiv:2606.13940v1, 2026-06-11；[v1 论文](https://arxiv.org/abs/2606.13940v1)；预印本，尚非同行评审定论。
- **阅读位置：** 摘要、§§3—5、§8 Limitations、Appendices A—B 与 E—F。
- **来源支持的事实：** 在 MIMIC-III/IV 上以相同输出 parser 和指标比较提示、SFT、GRPO 与渐进式提示注入。SFT 将病历转为可解析代码集合并提供经验标签先验；GRPO 使用解析后代码集合的样本级 F1 奖励；渐进式提示训练时聚焦当前模型漏掉的编码，推理时移除提示。生成实验使用 Qwen2.5-1.5B 和 Qwen3-4B。作者报告 SFT 提供主要能力跃升，GRPO 对全标签空间的代码集合预测有进一步改善；渐进式提示更偏向剩余漏码和宏观指标。
- **证据等级：** 单篇预印本的受控作者实验；标签集合 F1 是 ICD 编码这一特定任务的可计算代理，不代表临床判断质量。
- **适用边界：** 两个较小 Qwen 底模和同一 MIMIC 数据生态不能代表不同医院的文书、编码习惯与标签分布；全标签设置中的稀有代码召回仍困难，PLM-ICD 在全标签任务上仍具竞争力。训练时提示不在推理时提供，因此不能表述为模型依赖推理时外部标签提示。
- **待核验事项：** 跨机构、跨底模、不同标签版本验证；检查样本级 F1 与实际编码质量、稀有码风险和工作流结果之间的对应关系。

## FinGPT 参数配置的文内差异

- **来源与版本：** Yang et al., *FinGPT: Open-Source Financial Large Language Models*, arXiv:2306.06031v2, 2025-11-15；[arXiv HTML](https://arxiv.org/html/2306.06031)。
- **阅读位置：** §3.3.1 和 §5.2.1。
- **来源支持的事实：** §3.3.1 的方法介绍报告 LoRA/QLoRA 将可训练参数从 6.17B 降至 3.67M。后续情感分类实验 §5.2.1 则明确采用 Llama-3.1-8B-Instruct、rank 8、alpha 16，并报告 8.3M 可训练参数。
- **证据等级：** 同一预印本不同段落的配置说明；两组数字不足以当作唯一、通用的 LoRA 参数计数。
- **适用边界：** 不应把一个配置的可训练参数数字推广到所有 FinGPT 任务或 LoRA 设置；引用时须连同段落和配置一起说明。RLSP 的股价反馈是市场结果代理，不证明单条新闻造成价格变化。
- **待核验事项：** 若进行复现，需由固定版本代码和完整 target modules、rank、是否包含 embedding/head 等配置重新计数。

## TIMER 的分布胜率口径和训练范围

- **来源与版本：** Cui et al., *TIMER: temporal instruction modeling and evaluation for longitudinal clinical records*, *npj Digital Medicine* 8, 577 (2025), [PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12475073/), DOI 10.1038/s41746-025-01965-9。
- **阅读位置：** 摘要、Introduction、Methods、§§RQ1—RQ2、Discussion 与 Limitations。
- **来源支持的事实：** 作者用 Gemini-1.5-Pro 生成 Recent-Events、Timeline-Extremes 与 Full-Timeline 三类带时间戳指令，每类 5,000 条；Llama-3.1-8B-Instruct 训练 6 个 epoch，并另用 Qwen-2.5-7B-Instruct 展示跨架构结果。Full-Timeline 相对 Recent-Events 在均匀分布问题上的表列 head-to-head win% advantage 为 +6.50%；MedAlign correctness 从 30.69% 到 34.32%，TIMER-Eval 从 45.02% 到 48.51%。
- **证据等级：** 同行评审期刊论文的作者实验报告；来源是 Stanford STARR EHR，指令/评测问题含模型生成数据。
- **适用边界：** win% advantage 是论文表格中的成对偏好胜率指标，未据此改写成百分点差。作者指出模型生成数据仍需更大规模人工审查，未完成亚群公平性与输出校准评估；原始病历受隐私限制，研究场景和许可条件限制外推。
- **待核验事项：** 对生成样本开展更大规模人工审计，并在更多机构、病种和真实工作流中评估。
