# Tokenizer 与训练数据协议核验

## 范围与方法

- 研究问题：复核第5册第3章中 tokenizer 规范化、模板/API、packing 边界和词表扩展主张，重点检查版本依赖与实验外推边界。
- 日期：2026-11-24。
- 资料类型：ACL Anthology 论文与 PDF、arXiv 论文 HTML 版本、SentencePiece 官方文档、Hugging Face Transformers API 文档、NVIDIA NeMo 指定版本指南。
- 本记录只写可复核的研究事实及其适用边界；章节审阅顺序、修改经过与验收状态写入 `progress/`。

## 核验结果

### 规范化与 SentencePiece

- [SentencePiece 官方 normalizer 文档](https://github.com/google/sentencepiece/blob/master/doc/normalization.md)说明默认规则是带空格处理的 Unicode NFKC 变体 `nmt_nfkc`，支持预定义与自定义规则，并可将空格映射为 `▁` 保留序列中的空格信息。具体行为仍受模型文件和参数影响；SentencePiece 的默认规则不代表其他 tokenizer 的默认行为。
- [From Bytes to Subwords，Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.530/)报告了 Unicode normalization 在语言识别任务上的反例。正文所用实验设置为 2,307 种语言、每种 1,000 条样本、1–5 字符 n-gram Naive Bayes；十个划分中不规范化的均值准确率为 91.75%，NFC 为 91.70%。这是语言分类实验，不是 LLM 预训练、生成质量或数据通用清洗效果的比较；微小差值不能单独决定工程规范。

### Tokenizer 指标的预测边界

- [No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance，Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.96/)比较约 200 个语言模型、9 种语言和多类任务，摘要结论是不存在线性或单一 intrinsic tokenizer 特征能跨模型族、任务和语言稳定预测下游表现。
- [TokLens，ACL 2026 Student Research Workshop](https://aclanthology.org/2026.acl-srw.18/)比较 24 个 tokenizer、15 种语言和 6 项 intrinsic 指标；其结果区分英语与多语评估，并指出部分关系受模型规模和预训练数据混杂。指标相关性不能直接解释为 tokenizer 对能力的因果作用。
- [Equity with Efficiency，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1783/)在特定东南亚语言、受控模型训练设置中研究压缩效率与语言公平性；论文的 Pareto/质量结果描述其语料、语言和模型配置，不构成 tokenizer 方案的普遍排名。
- [To Each Language Its Tokenizer，Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1019/)提出按语言抽取模块化子词词表并在批次间选择的研究路线；正文应将其称为方案或候选，不把论文设想直接写成通用生产实践。

### 词表扩展与 embedding 初始化

- [FOCUS，EMNLP 2023](https://arxiv.org/html/2305.14481)以 XLM-R 为主要源模型，保留源/目标词表重叠项，并基于辅助静态 token embedding 空间中的语义相似度组合源 embedding。论文的实验对象是 XLM-R 单语种适配；其方法不能被当作现代生成式 LLM 的默认方案。
- [Beyond Initialization Loss，arXiv v1](https://arxiv.org/html/2608.03494v1)在 Nemotron-3-Nano-30B-A3B 上为 Hindi 增加 25,600 个 token，并以 100,000 篇 Sangraha Hindi 文档测量 token fertility 从 1.95 降到 1.25。该研究比较 20 多种初始化策略，报告初始化 loss/BPB 与短程 CPT 后排名可能变化；50-step 约 0.42B-token 的短程验证在其设置下能帮助选择策略，最佳输入/输出矩阵初始化并不相同。结果限定于论文模型、Hindi 数据和训练配方。
- [In-Place Tokenizer Expansion for Pre-trained LLMs，arXiv v1](https://arxiv.org/html/2607.15232v1)通过延续 BPE merges 保留旧 token 映射、为新 token 做源子词均值初始化，再先训练新 embedding 行、后做全模型多语继续预训练；作者在 LFM2 家族报告 65K 到 128K 的 tokenizer 扩展及 Hindi、Vietnamese、Thai 的 token 数下降。每字符 2.2–3.7 倍速度是压缩数据和参考设备每 token 吞吐合成的估计，较大词表仍带来每 token 成本；结果不能外推为任意第三方 tokenizer 可直接替换。

### 版本化 API 与 packing 实现

- [Hugging Face Transformers 当前 tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)定义 `apply_chat_template` 的 `add_generation_prompt`、`continue_final_message` 与 `sanitize_control_tokens`。文档说明两种生成起点参数不能同时使用；开启 sanitize 时，消息/工具/文档等输入字段中的完整控制 token 字面串按普通文本编码，模板自身写入的标记仍作为控制 token。该接口是滚动文档，落地行为必须按锁定的 Transformers 版本与模板复核。
- [NeMo Framework 25.02 Sequence Packing 指南](https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html)说明其实现将边界累积长度 `cu_seq_lens` 传入变长 attention kernel，从而不计算样本间 attention，理论计算复杂度保持为各序列长度平方和。该结论属于 NeMo 指南描述的指定实现，不能推出所有 packing kernel 具有同一语义或性能。

## 对正文的支持边界

- 本章使用研究结果时，应保留论文/软件版本、模型、语种、数据集、训练阶段、硬件和指标口径；相关性、估计速度和单一实验结果均不能写成通用因果结论。
- Token ids、control token、模板、padding、loss mask、processor 与 packing 行为都由具体 tokenizer/model/framework 版本共同决定；需要固定样本的可回放证据，不能仅凭字段名称或滚动 API 文档断言运行行为。
- 本轮未运行 tokenizer、packing、训练、模型生成或代码示例。
