# 第四册 NLP 与 Tokenization 精修核验（2026-10-22）

## 当前检查点

全文初读 3.1—3.12，局部修订 BPE 编码/解码、基础字符覆盖、重叠 pair、chat template 与 label mask 的职责，以及审计代码。整章尚未验收。

## 已读取来源

- [Transformers tokenizer summary](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)：读取 BPE 基础字符外 UNK 与 byte-level BPE 的 256 字节基础集合说明，支持区分普通 BPE 和完整字节覆盖。
- [Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：读取 tokenize=False 后再次编码需 add_special_tokens=False 的警告，避免重复添加模板已有特殊符号。
- [SentencePiece 仓库](https://github.com/google/sentencepiece)：读取模型文件保存 normalization、词表与切分模型相关说明，作为继续核验线索；本次没有据 README 的可逆性概括扩展正文结论。

以上是动态 main/master 页面和指定段落阅读，不是固定 commit、全部文档或论文全文核验。curl 成功取得页面，不代表全章所有外链均已检查。

## 验证结果

Python 标准库提取并独立编译执行三个 Python 围栏，通过：padding 长度 4/5/9 的 waste 为 1/3；BPE 首个合并 pair 为 (l,o)，频次 3；审计例子 valid_labels 为 1。

新增五组拒绝断言通过：label=-99、bool label、浮点 mask、bool mask、padding 位置被监督。重叠合并 `aaa` 的 (a,a) 从左向右结果为 [aa,a]。git diff --check 通过。

审计器仍仅检查所列结构约束，不验证词表上界、语义、真实 tokenizer 或 causal shift 后训练有效性。未安装 tokenizer 库、下载权重或运行真实模型。

## 整章验收补充

继续复读并修订剩余内容：区分 Unicode 码点/字素簇、字节表示层与文本 API 覆盖；说明特殊 ID 不自动构成安全边界、共享 EOS/PAD 时按位置屏蔽；分开可见/隐藏输出预算，避免重复计数；补因果 shift 后分母、PPL/BPB 反例和窗口评估条件。assistant-only 与全序列监督按训练目标区分，非空目标集合为必要条件。

本次实际读取的新增来源：

- [Unicode UAX #29](https://unicode.org/reports/tr29/) 的 Grapheme Cluster Boundaries 相关说明：多个码点可构成一个用户感知字符，默认字素簇是可定制的近似。
- [Transformers PPL](https://huggingface.co/docs/transformers/main/en/perplexity) 的 tokenizer 依赖、固定长度上下文、strided sliding window 和目标计数说明。
- [BERT tokenization.py](https://github.com/google-research/bert/blob/master/tokenization.py) 的 WordpieceTokenizer 类：贪心最长匹配、词内前缀、无法匹配时整词 UNK。下载的是动态 master，不是固定 commit。临时验证器用 AST 提取该类，仅为字符串输入提供 Unicode 转换/空白切分等价辅助函数，不导入 TensorFlow，验证 abc 在词表 ab/a/##bc 下返回 UNK；这不是完整 BERT tokenizer 测试。
- [Subword Regularization v1 HTML](https://arxiv.org/html/1804.10959v1) 第 3 节相关段落：Unigram 路径乘积、最佳路径、边缘似然和采样的区别；未复现论文实验或声称全文核验。

最终五个 Python 块均独立编译执行通过。Unicode 输出码点/字节/NFC长度为 1/2/1、2/3/1、1/3/1；PPL 为 7.389056/2.718282，BPB 均为 0.721348。补查 padding 非法长度、零浪费、非法 id/special id、label/mask 与监督 padding，及 Unigram 0.49、预算 900 数值；BPE 重叠替换保持通过。16 个 math 块花括号/环境和总围栏结构通过，历史下标及宏已统一；未运行外部公式渲染器。

本章 12 个唯一外链最终均 HTTP 200。首次沙箱脚本无法连接代理，获准后重试成功；可达性不当作原文阅读，未逐篇读取其他论文。章节相对目标、本册目录/简介、下一章和配套练习入口检查通过。EXERCISES.md 增加五项反例/迁移验收，术语表 fertility 分母同步；题库已有相关主题，无需复制摘要。

本章声明范围内验收完成。真实 tokenizer、权重、训练、服务安全与性能均未运行；构造 BPB 只演示单位，不证明无损字节概率。PDF 仍受已有 Pandoc 缺失限制。全书及第二册遗留审计仍未完成。
