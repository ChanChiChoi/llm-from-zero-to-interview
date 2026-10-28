# 第五册第三章：Tokenizer 与词表扩展证据记录

- `date`：2026-10-28
- `chapter`：[`book-05-llm-training/chapters/03-tokenizer与数据格式.md`](../book-05-llm-training/chapters/03-tokenizer与数据格式.md)
- `scope`：补齐第三章的独立研究证据与代码状态；原有 tokenizer、chat template、label mask、packing、扩词表和兼容性主线不变。

## 新增证据

### In-Place Tokenizer Expansion for Pre-trained LLMs

来源：[论文 HTML](https://arxiv.org/html/2607.15232)。阅读了方法、初始化、分阶段继续训练、实验表格和局限说明。

- 方法前提是模型团队保留原 tokenizer 的 BPE 合并规则及 special-token 配置，并可修改原模型 embedding。扩展词表沿用旧 merges、追加新 merges，保持旧 token 到原 embedding 的映射，再将新 token 分解成旧词表片段并组合既有 embedding 初始化。
- 训练分两阶段：先适配 embedding，再做全模型 continued pretraining。论文的 LFM2-8B-A1B 对照从 65K 扩到 128K 词表；作者报告 Hindi、Vietnamese、Thai 的 token 数分别约减少 2.4、2.6、4.0 倍。
- 论文的 per-character decode latency 改善由 tokenizer 压缩率与参考设备 token 吞吐合成估计；设备实验中，某些原来 tokenization 较紧凑的语言每字符解码可退化约 9%。Stage 2 使用 400B tokens 的继续预训练。它们构成扩词表的质量恢复和系统成本，不应只看压缩率。
- 证据仅来自单一模型家族和论文所用语言、数据、硬件。该路线要求控制源 tokenizer；不能直接推广到任意第三方 tokenizer 替换、词表拼接、模型或线上设备。

## 与既有词表初始化研究的区别

第三章已有 [Beyond Initialization Loss](https://arxiv.org/abs/2608.03494) 的入口：它在单一 Hindi/Nemotron 设置下比较新 token embedding 初始化，并报告短程 CPT probe 与 step-zero loss 的差异。该研究关注新 token 如何初始化及早期适配信号；本次补充的 In-Place 路线关注生产方如何延续原 BPE merges、保留旧 token 映射并实施两阶段扩词表。两者问题不同，均需在目标模型和任务中复核，不构成通用 recipe。

## 代码与验证边界

- 第 3.15 节 tokenizer/格式 toy 代码只使用 Python 标准库，采用最长片段匹配及固定 byte fallback；不训练 BPE、不模拟真实 tokenizer offset、attention 或多模态 processor。
- 本地执行第 3.16 节唯一 Python 围栏，环境为 Python 3.12.9 + 标准库；输出与正文的 token/byte 比、byte fallback、assistant label、packing、EOS、扩词表行数一致，8 项主路径断言通过，未知角色、空监督和非法 block size 按约定拒绝。
- 该次执行只验证教学 toy；没有运行真实 BPE、生产 tokenizer、attention、GPU、processor 或模型训练。
- 章节长度超过核心知识单元的 1800 汉字门槛（按正文统计口径排除代码、公式、标题和表格）；第三章共有一个 Python 教学示例。
- 本记录不声称复现论文、训练扩词表模型、测量解码延迟或验证第三方 tokenizer 兼容性。
