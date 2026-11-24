# Qwen2.5、Llama 3.1 与 Gemma 2：开放权重训练工程证据

## 研究范围与检索

- 研究问题：以三个 2024 年公开模型版本为案例，核实训练数据规模与处理、tokenizer、训练阶段/基础设施、权重与训练资料的披露边界及使用条款；不做跨论文性能排名。
- 读者与边界：服务第二十一册第 57 章，解释模型架构史中的训练工程公开证据。第五册主讲训练数据、tokenizer、预训练和后训练机制；第二十一册第 54 章主讲 LLaMA 架构演进，第 59 章主讲通用横向比较。
- 检索日期：2026-11-24。检索平台：网页搜索；优先打开 arXiv、Meta、Qwen、Google DeepMind/Google AI 与 Stanford 的原始页面和官方课程资料。
- 主要检索词：`Qwen2.5 Technical Report 18T tokens`; `The Llama 3 Herd of Models 15.6T tokenizer data curation`; `Gemma 2 report training data distillation 2B 9B 27B`; `Stanford CS336 tokenization data filtering deduplication`; `FineWeb dataset data curation ablations`。
- 本章采用固定版本：Qwen2.5 报告（2024-12-19）；《The Llama 3 Herd of Models》arXiv v3（2024-11-23；论文明确其结果均对应 Llama 3.1）；Gemma 2 报告（2024-07-31）。这些是可审阅的历史案例，不代表 2026 年各系列最新版本。

## 关键发现与证据边界

### Qwen2.5

- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)称预训练数据规模从 Qwen2 的 7T 增至 18T；报告将其归因于数据规模与质量、分阶段数据 mixture、代码和数学数据、筛选/打分等多个选择，而不是单一架构变化。报告还写明其开放权重稠密系列有 0.5B、1.5B、3B、7B、14B、32B、72B 等尺寸，且托管 API 型 Turbo/Plus 属于专有 MoE 型号。证据支持报告声明，不能证明公开了完整语料、过滤产物或服务实现。
- 同一报告记载后训练阶段包含超过百万级示例、SFT/DPO/GRPO 等；不同章节口径有“1 million examples”与“millions of high-quality examples”表述，正文避免将其写成精确的全流程样本计数。报告还区分 4,096 到 32,768 的预训练序列扩展与之后的上下文能力扩展。
- 报告表 1 按尺寸列出许可证：0.5B、1.5B、7B、14B、32B 为 Apache 2.0；3B 为 Qwen Research；72B 为 Qwen。官方 [Qwen2.5 发布页](https://qwenlm.github.io/blog/qwen2.5/)说明除 3B 与 72B 外的此批次模型采用 Apache 2.0。许可结论必须固定模型 ID/权重版本；不据此做法律解释。

### Llama 3.1

- [The Llama 3 Herd of Models, arXiv v3](https://arxiv.org/abs/2407.21783v3)的 Table 1 说明论文结果对应 Llama 3.1；正文将 405B 版本描述为 dense Transformer，报告 15.6T text tokens，并将 8K 序列预训练与后续 128K 长上下文持续训练分开。
- 报告记载 128K tokenizer vocabulary、从 tiktoken 词表扩展的多语言 token，并在一个英语样本上报告字符/token 压缩比由 3.17 提高到 3.94；该数字不能泛化到中文、代码、其他语种或每个 tokenizer 版本。Meta [Llama 3 官方技术博客](https://ai.meta.com/blog/meta-llama-3/)还描述了超过 15T 公共来源 token、非英语数据比例、启发式/NSFW/语义去重/质量分类器、数据 mixture scaling、训练并行和可靠性工程。
- 训练报告与 [Meta Llama 3.1 模型卡/许可证](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct/blob/main/LICENSE)分别支持模型和使用条款事实。该许可证于 2024-07-23 发布，称为 Llama 3.1 Community License；章节只提醒读者按版本阅读，不把它等同于 Apache 2.0 或一般开放数据授权。
- [FineWeb 论文](https://arxiv.org/abs/2406.17557)明确指出 Llama 3 等开放权重模型的预训练数据并未公开，并报告 FineWeb 自身发布了语料、数据处理代码与消融训练模型。该来源支持“公开权重与公开训练数据是不同发布维度”，不能替代 Meta 对特定隐藏语料的说明。

### Gemma 2

- [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)（2024-07-31）与 [Google Gemma 2 模型卡](https://github.com/google-gemma/docs/blob/main/docs/models/gemma-2/index.md)记载：27B、9B、2B 分别使用 13T、8T、2T token，数据主要为英语网页、代码和科学文本；2B 与 9B 采用知识蒸馏训练，27B 则从头训练。报告将蒸馏描述为学生学习教师对下一个 token 的概率分布，正文保留这一差异并限制为该报告的模型选择。
- Google [Gemma 2 技术说明](https://developers.googleblog.com/gemma-explained-new-in-gemma-2/)解释 2B/9B 蒸馏决策，适合作为报告的作者解释；关键机制仍以 arXiv 报告为准。模型卡称两种预训练/指令变体提供开放权重，发布页面另链接 [Gemma Terms of Use](https://ai.google.dev/gemma/terms)。当前条款页标示 2026-04-01 更新；本轮只用它确认当前条款是独立发布物，不把该页面当作未变更的 2024 原始条款。

### 教学组织来源

- [Stanford CS336 Spring 2026 官方课程页](https://cs336.stanford.edu/)将课程按 tokenization/resource accounting、architecture/scaling、data sources、filtering/deduplication/mixing/synthetic data、SFT/RLHF/RLVR 组织，适合观察训练工程各环节怎样衔接。其课程结构是组织经验，不是 Qwen、Llama 或 Gemma 的训练证据。
- [CS336 2025 公开课堂笔记](https://qihongruan.github.io/cs336/)用于检视数据与对齐专题内部的教学层次；这是课堂材料的二次整理，不作为技术事实来源。

## 采用结论与未解决边界

这三例体现的并非同一种“开放”：Qwen2.5 的披露横跨尺寸、数据规模/mixture、后训练与分尺寸许可证；Llama 3.1 的报告较细地呈现数据、tokenizer、训练规模和系统可靠性，但 FineWeb 论文指出其预训练数据不可得；Gemma 2 公开了不同尺寸的数据预算和蒸馏路线，同时以单独 Terms 控制使用。报告披露量、训练结果与许可证都不能彼此替代。

报告没有给读者下载每套完整原始训练语料和全部过滤中间产物，故无法仅凭本文来源复现这些发布模型的完整训练。各报告的模型大小、数据、评估套件和发布时间不同，数据 token 数不能单独推出质量因果或训练效率排名。章节不裁定何者“更开源”，也不将任何一个版本的做法扩展到整个家族。
