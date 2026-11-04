# 频率与长度感知的子文段去重：来源证据

记录日期：2026-11-03

本记录只保存研究事实、来源和证据边界。书籍审阅、正文取舍和修改记录写入 `progress/`。

## Scalable Frequency- and Length-Aware Subdocument Deduplication

- **来源：** Hai Wang et al., [Scalable Frequency- and Length-Aware Subdocument Deduplication for Large Language Model Pretraining](https://arxiv.org/abs/2608.03089)，arXiv:2608.03089v1，2026-08-04。
- **阅读范围：** arXiv HTML 摘要、引言、方法 §4、实验设置 §5.1、FineWeb-Edu 结果 §5.2/表 1、代码网页结果 §5.3/表 2、语料缩减 §5.4/表 3、保留行为 §5.5 和结论。
- **证据等级：** source-backed（arXiv v1 预印本中的方法与受控训练结果）。
- **方法事实：** 作者将子文段重复检测与副本保留策略拆分。检测侧使用自然边界切分、规范化精确 hash 和分布式聚合，以全局计数降低 shard 划分对重复可见性的影响；保留侧结合重复组全局频次和片段字符长度设定副本预算，并按连续片段删除，以控制文档破碎。方法仍受预先切分边界限制，未被分到相同切片的局部重复可能漏检。
- **实验设置：** 训练模型为 Hy-MT2-30B-A3B；FineWeb-Edu 原始语料为 6.28T tokens，另有 561.53B-token 的 Common Crawl 代码网页语料。作者训练 36,000 步、共消费约 294.9B tokens，token 计数使用该模型 tokenizer。FineWeb-Edu 测试报告八项基准的非加权平均；代码网页测试报告五项基准的非加权平均。
- **作者报告的结果：** FineWeb-Edu 完整流水线（先 Doc-MinHash，再子文段去重）平均分为 52.90，Suffix-Array 为 52.79，Keep-One 为 52.14；不做文档级 MinHash 的 Ours (Subdoc-only) 平均分为 52.92，应与完整流水线分开读。代码网页实验中，Ours 五项平均分为 41.40，No-Dedup 为 37.98。不同语料的 benchmark 平均值不可横向合并。
- **适用边界：** 结果来自特定模型、语料、tokenizer、训练预算和作者的任务/评分协议。数据只支持该实验下的比较，不证明子文段去重在任意语料或预算上都提升效果，也不能直接给出可移植的阈值或保留函数参数。
