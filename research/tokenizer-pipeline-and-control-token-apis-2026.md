# Tokenizer 管线与控制 token API

检索与阅读日期：2026-10-30。本文记录 tokenizer 的规范化、预切分、子词模型、后处理，以及对话模板中的控制 token 清理接口。官方“main/latest”文档是滚动页面；除明确给出的论文版本和源码路径外，不将其视为固定软件发行版。

## Tokenizer 管线分层

- 来源：[Hugging Face Tokenizers：The tokenization pipeline](https://huggingface.co/docs/tokenizers/python/latest/pipeline.html)，官方库文档，阅读日期 2026-10-30。
- 文档描述的调用路径为 normalizer、pre-tokenizer、model、post-processor。pre-tokenizer 将文本切成较小片段并保留原文 offsets；model 在片段上执行 BPE、Unigram、WordLevel 或 WordPiece 规则并映射词表 ID；post-processor 可以加入适合任务格式的特殊 token。
- 文档说明，改变 normalizer 或 pre-tokenizer 通常意味着要重训/重新适配 tokenizer；只改 post-processor 一般不需要重训 tokenizer 模型。这里是 Tokenizers 库文档描述的组件边界，不代表所有实现都必须拆成同样的软件对象。
- 文档中的 `Whitespace()` 与 `Digits(individual_digits=True)` 组合会把 `Call 911!` 预切分为 `Call`、`9`、`1`、`1`、`!`。这是特定配置的预切分结果，尚未表示最终 BPE token 或 token ID。

## BERT WordPiece 的实现边界

- 来源：[Google Research BERT `tokenization.py`](https://github.com/google-research/bert/blob/master/tokenization.py)，官方仓库源码，阅读日期 2026-10-30；链接指向可变的 `master` 分支。
- 该实现对单词按从当前位置开始的最长匹配贪心切分；超过 `max_input_chars_per_word` 默认值 200 的词直接映射为 `[UNK]`；若贪心过程有任一片段无法匹配，也将整个当前词映射为 `[UNK]`，不会回退尝试另一种切分。
- 结论只描述该 BERT 代码路径，不应外推为每个 WordPiece tokenizer 的通用规范。

## Chat template 中的控制 token 清理

- 来源：[Transformers tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.apply_chat_template)，Hugging Face 官方 API 文档，阅读日期 2026-10-30。
- 当前滚动文档的 `apply_chat_template` 参数包含 `sanitize_control_tokens=False`。开启后，消息、工具和文档内容中出现的控制 token 字面文本按普通文本编码，模板本身写入的控制 token 仍作为控制 token。文档将控制 token 定义为 tokenizer 的 special tokens，以及 `tokenizer.chat_control_tokens` 中的新增控制 token；未设置该属性时，会从模板文本及其字符串字面量中推断。
- 该参数只证明当前文档记录的 Transformers 接口行为；该页面未在本次阅读中固定到具体发行版本。不能据此推断其他库、旧版本或自定义序列化链路具有同一行为，也不能将其视为对自然语言 prompt injection 的通用防护。

## 算法论文与延伸材料

- [Formalizing BPE Tokenization](https://arxiv.org/abs/2309.08715)，arXiv:2309.08715，2023-09-15。论文形式化讨论实用 BPE 的语义，并比较 SentencePiece 与 Hugging Face tokenizer 在规则构造下的关系，也讨论增量和流式处理。该论文是形式化研究材料，不替代具体库的当前官方文档。
- [Which Pieces Does Unigram Tokenization Really Need?](https://arxiv.org/abs/2512.12641)，arXiv:2512.12641，2025-12。作者讨论 Unigram tokenization 的训练和推断步骤、参数选择及一个计算更简化的变体。本文只将其作为进阶方法来源；本次未核验其后续同行评审状态，不据此替代 Kudo 与 Richardson 的 SentencePiece 原始论文。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)，arXiv:1808.06226，2018-08-19。论文说明 SentencePiece 可以直接从原始句子训练子词模型，支持不依赖先行词切分的流程；实际算法和配置仍需看对应实现。

## 证据边界

上述官方文档支持其所述库接口和组件职责，不支持“所有 tokenizer 都采用同一管线”的泛化。BERT 源码的具体贪心和未知词行为不代表所有 WordPiece 实现。算法论文支持方法定义与论文讨论范围，不证明某 tokenizer 在特定语言、语料、质量或硬件上的普遍优势。主线文档与 master 源码均可能变化；需要复现实验时应保存包版本、tokenizer 文件、配置及固定源码 revision。
