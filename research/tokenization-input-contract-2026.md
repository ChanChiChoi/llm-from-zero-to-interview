# Tokenization 与聊天输入协议：事实和证据

检索日期：2026-10-28。

## UTF-8 字节数

- RFC 3629（2003-11，Internet Standard）第 3 节规定 UTF-8 字符使用 1–4 个 octet；码点 U+0800 至 U+FFFF 编码为 3 个 octet，补充平面码点可占 4 个 octet。来源：[RFC 3629 §3](https://www.rfc-editor.org/rfc/rfc3629#section-3)。证据等级：标准规范。适用边界：正文示例中的“模”(U+6A21)与“型”(U+578B)各为 3 octet；不能外推成所有汉字都占 3 octet。

## 子词算法与 SentencePiece

- Sennrich 等人的 BPE 论文（arXiv:1508.07909，v5，2016-08）第 3.2 节说明其方法从字符词表开始，统计符号对并反复合并高频相邻对。来源：[论文 HTML §3.2](https://arxiv.org/html/1508.07909v5#S3.SS2)。证据等级：原始论文方法定义。论文实验聚焦特定机器翻译任务，不能由此推出所有现代 tokenizer 的实现或效果。
- Hugging Face Transformers v5.17.0 的官方算法文档分别介绍 BPE、Unigram 和 WordPiece，并说明 WordPiece 的合并分数按组合词频相对单元词频估算，而不是简单选最高频相邻对；文档将其用于讲解实现，不代表所有产品 tokenizer 的内部实现。来源：[Tokenization algorithms](https://huggingface.co/docs/transformers/v5.17.0/tokenizer_summary)，阅读位置：BPE、Unigram、SentencePiece、WordPiece 小节。证据等级：固定版本官方文档。
- Kudo 与 Richardson 的 SentencePiece 论文（arXiv:1808.06226，2018）说明 SentencePiece 面向原始文本，可训练/使用 BPE 与 Unigram 语言模型；SentencePiece 是工具/实现，不是唯一的一种分词算法。来源：[论文 HTML](https://arxiv.org/html/1808.06226)，阅读位置：§2、§3 及结论。论文描述的默认规范化器和具体实现边界不应外推到所有配置。
- Devlin 等人的 BERT 论文（arXiv:1810.04805，2018）在 §3 报告 BERT 使用 30,000 词表的 WordPiece。来源：[论文 HTML §3](https://arxiv.org/html/1810.04805v2#S3)。证据等级：原始论文对该模型配置的报告；它支持 BERT 是实例，不等于完整定义 WordPiece 算法。

## 聊天模板和特殊 token

- Hugging Face Transformers v5.17.0 官方聊天模板文档说明，模板将消息列表格式化为模型所需的 token 序列；模板可包含控制 token，生成提示会影响 assistant 起始边界；把模板先渲染为文本、之后再编码时，可能重复添加特殊 token。来源：[Chat templates](https://huggingface.co/docs/transformers/v5.17.0/chat_templating)，阅读位置：“What are chat templates?”、“Adding generation prompts”及特殊 token 提示。证据等级：固定版本官方文档。
- 上述材料支持“模板、控制 token 和 tokenizer 的特殊 token 配置须作为模型输入接口核对”，但不证明所有模型使用同一角色符号、停止符、拒答符或模板，也不规定任何具体服务 runtime 的工具调用行为。

## 教学预算例的范围

- 4,096-token 窗口内，若每个示例输入分别占 700 和 1,100 token，则单纯相除为约 5.85 和 3.72 个请求长度当量；只装这些输入时，完整输入上限分别为 5 和 3 个。系统消息、模板、已有对话和生成输出都会占用上下文，所以这是算术教学例，不是服务容量承诺，也不据此比较 tokenizer 优劣。

## 未验证边界

- 没有测量任何真实模型的词表、规范化、编码长度、模板输出或服务可用上下文；这些必须针对明确的 tokenizer 文件、模型版本和请求模板实测。
- BPE、WordPiece、Unigram 的概述不构成完整训练算法说明。本材料不评价特定算法在不同语言、语料或任务上的总体优劣。
