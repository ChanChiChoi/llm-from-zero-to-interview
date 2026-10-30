# 预训练数据、packing 与 checkpoint：来源证据

记录日期：2026-10-28

本记录只保存联网核验得到的知识点、来源和适用边界，不记录逐章审阅过程、书内修改决定或进度状态。

## 自回归预训练与评估污染

- Brown et al., “Language Models are Few-Shot Learners,” arXiv:2005.14165，[论文](https://arxiv.org/abs/2005.14165)。
  - 实际阅读范围：摘要、§1、§2 的训练方法概述，以及 §4 中 benchmark contamination 的分析。
  - 来源支持：GPT-3 是自回归语言模型；论文系统检查了 Common Crawl 等大规模网页训练数据可能带来的测试集重叠。作者认为被检查的大多数数据集受影响有限，但对少数可能抬高结果的数据集标记或不报告结果。
  - 证据等级：source-backed（原始论文模型目标、污染检查流程和 GPT-3 实验结论）。
  - 适用边界：不能由检测到任意文本重叠直接推断 benchmark 分数必然虚高；结论限于该论文所检查的 GPT-3 训练语料、检测方法与 benchmark 集合。

## 重复数据与任务污染检测

- Lee et al., “Deduplicating Training Data Makes Language Models Better,” ACL 2022，[论文](https://aclanthology.org/2022.acl-long.577/)。
  - 实际阅读范围：摘要、§1 的去重方法概述与实验结果总结。
  - 来源支持：作者在所研究的语言模型数据集发现近重复例子和重复子串，提出精确子串匹配及近似全文档匹配方法；其 C4 等数据实验报告了训练/验证文本重叠、重复暴露与模型输出记忆之间的关系，并在特定设置下观察到去重带来的效果。
  - 证据等级：source-backed（数据集分析、方法定义和指定模型实验）。
  - 适用边界：论文中的记忆率、重叠比例、训练效率和困惑度结果属于对应语料、模型规模、去重阈值与任务；不证明任何去重策略都能改善所有能力，也不把记忆风险等同于所有重复文本必然产生伤害。

- Li and Flanigan, “Task Contamination: Language Models May Not Be Few-Shot Anymore,” AAAI 2024 / arXiv:2312.16337，[论文](https://arxiv.org/abs/2312.16337)。
  - 实际阅读范围：摘要、§4 的发布时间比较、§5–6 的训练数据检查和任务样例提取，以及作者对检测 recall 的说明。
  - 来源支持：在论文所测模型和数据集上，较早发布的数据集更常出现高于多数类基线的表现；训练数据检查、任务例提取等方法对部分任务给出污染证据，但检测方法存在低召回局限。
  - 证据等级：source-backed（原始研究的模型/数据集比较与检测分析）。
  - 适用边界：发布日期只是污染可能性的代理变量；研究样本、数据收集时间和模型集合有限，不能单独证明某个特定 benchmark 已污染，也不能仅据此因果归因任意性能差异。

## 文档 packing 与因果注意力边界

- “Packing and Masking Strategies for Pre-Training Sequence Composition,” ACL 2024，[论文 PDF](https://aclanthology.org/2024.acl-long.427.pdf)。
  - 实际阅读范围：摘要、§2 packing/masking 定义、实验结果总结与 Limitations。
  - 来源支持：文档可用 EOS 拼接后组成固定长度 chunk；普通 causal mask 允许后续文档 token 读取 chunk 内较早文档；论文比较该设置与 intra-document causal masking，并在其预训练模型、语料和任务中发现跨文档上下文可能造成干扰，同时报告隔离 mask 的运行开销。
  - 证据等级：source-backed（论文给出的定义、实验和实现开销）。
  - 适用边界：论文不能推出所有跨文档注意力都有害；收益与计算成本取决于模型、文档排序、数据和评测任务。EOS 是序列中的 token，本身不定义 attention mask。

## Checkpoint 与可复现性

- PyTorch, “Saving and Loading Models,” v2.14 教程，[官方文档](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
  - 实际阅读范围：一般 checkpoint 的保存/恢复示例及关于继续训练需保存模型和 optimizer 状态的说明。
  - 来源支持：PyTorch 教程示例将 epoch、模型 `state_dict`、optimizer `state_dict`、loss 等组织到 checkpoint 中；其说明指出恢复训练时 optimizer 状态也重要。
  - 证据等级：source-backed（该版本教程的公开示例和使用建议）。
  - 适用边界：教程给出的普通 PyTorch 状态字典示例不能直接规定 FSDP、ZeRO 或其他分片系统的 checkpoint 格式；完整续训状态依赖实际训练框架。

- PyTorch, “Reproducibility,” v2.14，[官方文档](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。
  - 实际阅读范围：开头的跨平台/版本限制、随机数控制和确定性算子说明。
  - 来源支持：固定种子和选择确定性算法可以减少特定环境内的随机差异；文档明确不保证跨 PyTorch 版本、commit、平台或 CPU/GPU 完全复现，并指出确定性选项可能降低性能。
  - 证据等级：source-backed（PyTorch v2.14 官方限制说明）。
  - 适用边界：仅凭保存随机种子不能保证完整训练逐位重放；数据 worker、外部库、分布式执行和硬件条件仍需按项目具体实现核对。

## Base model 与指令微调

- Ouyang et al., “Training language models to follow instructions with human feedback,” arXiv:2203.02155，[论文](https://arxiv.org/abs/2203.02155)。
  - 实际阅读范围：摘要、§1 的结果说明，以及 §3 方法和 §4 结果的概览。
  - 来源支持：论文比较 GPT-3 与在相同架构基础上经过示范监督和人类反馈微调的 InstructGPT；在论文所用 API prompt 分布和标注者评测中，标注者更偏好微调模型的输出。
  - 证据等级：source-backed（特定模型、prompt 分布、训练流程和人类偏好实验）。
  - 适用边界：这不是所有任务、用户群体或后训练方法的普遍排名，也不能据此声称微调后必然更正确或安全。

## 隐私框架与数据保护

- NIST, [Privacy Framework](https://www.nist.gov/privacy-framework)，页面核验日期 2026-10-30。
  - 实际阅读范围：官方页面的 About、版本入口及 Privacy Framework 1.1 Initial Public Draft 状态说明。
  - 来源支持：NIST 将 Privacy Framework 描述为与利益相关方合作制定的自愿工具，用于组织层面的隐私风险识别与管理；当前页面同时提供 v1.0 资料与 v1.1 初始公开草案入口。
  - 证据等级：source-backed（NIST 官方框架说明）。
  - 适用边界：该框架是自愿风险管理资源，不是特定数据采集行为的法律授权、合规认证或法律意见；v1.1 链接所标状态为 initial public draft，应与正式发布版本区分。

- NIST, [SP 800-122: Guide to Protecting the Confidentiality of Personally Identifiable Information](https://csrc.nist.gov/pubs/sp/800/122/final)，2010-04-06。
  - 实际阅读范围：CSRC 官方出版页的标题、摘要、出版日期和适用对象说明。
  - 来源支持：该指南面向美国联邦机构，给出按上下文识别 PII、评估保密影响并选择防护措施的建议；NIST 鼓励机构按自身要求调整建议。
  - 证据等级：source-backed（NIST 官方指南目录与摘要）。
  - 适用边界：这是面向联邦机构的 2010 年指南，不构成跨法域的 PII 法律定义或针对任何具体训练集的合规判定。

- U.S. Copyright Office, [Copyright and Artificial Intelligence](https://copyright.gov/ai/)，页面核验日期 2026-10-30。
  - 实际阅读范围：官方页面关于 Report Part 3: Generative AI Training 的状态说明。
  - 来源支持：页面称该 Office 于 2025-05-09 发布训练主题 Part 3 的预发布版本，并将最终版本列为 forthcoming。
  - 证据等级：source-backed（美国版权局官方项目状态页）。
  - 适用边界：该报告讨论美国版权法与政策，页面显示的是预发布状态；不能作为其他法域的规则或具体语料的训练授权。

## 生命周期

预训练目标、packing 边界和 checkpoint 教学事实：active。具体数据来源、许可条件、采样配比、attention 边界、分布式状态格式与评估协议均取决于项目配置；论文中的量化结果不可脱离其模型、数据和任务外推。
