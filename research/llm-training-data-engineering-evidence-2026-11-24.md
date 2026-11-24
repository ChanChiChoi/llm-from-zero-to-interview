# 训练数据工程章节：公开语料与筛选实验的证据

检索日期：2026-11-24。记录第五册第 2 章核对过的公开研究结果。实验数字只适用于各论文自己的语料、模型、训练预算和评测设置。

## 质量过滤与参考集

- DataComp-LM 的 NeurIPS 2024 论文报告：fastText 过滤器训练集为 400K 个文档，正负样本各 200K；负样本来自较早版本的 RefinedWeb reproduction。其 7B–1x 消融中，OH-2.5 + ELI5 正样本、10% 保留阈值取得该表最高 Core（41.0），但该消融不能与 1B–1x 的过滤方法比较混为同一个排行榜。[DataComp-LM 正式论文](https://papers.neurips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf)；[arXiv HTML v4](https://arxiv.org/html/2406.11794)。
- FineWeb 论文 v2 报告，FineWeb-Edu 的标注来自 Llama-3-70B-Instruct 对 460,000 个网页的评分，其中 410,000 个标注用于训练线性回归器，50,000 个用于验证；评分提示面向小学和中学层级的教育价值。由此扩展出的 1.3T token 子集更适合解释知识与推理任务上的作者结果，不等于跨语种、代码或对话数据的通用质量标签。[FineWeb 论文 v2](https://arxiv.org/html/2406.17557)。
- “Removing Noise, not Finding Gold” v4（2026-06-24 修订，论文注明被 ICML 2026 接收）报告分类器质量过滤能改善下游任务，但不保证改善高质量参考集上的语言建模；论文将此与分类器同时筛选正例参考集联系起来。结论限于其 RedPajama、模型和实验设置。[论文 v4](https://arxiv.org/abs/2510.00866v4)。

## 覆盖与多样性

- “Towards Scalable Data Diversification … via Leverage Score Sampling” v1（2026-09-26）在 Common Crawl web-only、40% 选择设置下报告：DCLM-fastText 的 Base Mean 为 50.21%，Code Mean 的表内值为 266.77（按 $10^{-2}$ BPB 为单位，即 2.6677 BPB）；Lev 的相应结果为 48.91% 与 256.43（即 2.5643 BPB）。两列方向相反；这些作者报告的单一设置结果不构成普遍数据排序，论文也将其观察限定在所用语料和评估任务。[论文 v1，实验表 2](https://arxiv.org/html/2609.32484v1)。

## 公开数据处理案例

- Dolma v1 论文的网页管线按 URL 去重、按文档去重、执行质量和内容过滤，最后做段落级去重。其 PII 规则聚焦高精度正则可检测的邮箱、IP 和电话号码：不超过 5 个匹配片段的文档用专用 token 替换，论文称影响 0.02% 文档；含 6 个及以上匹配片段的文档被移除，影响 0.001%。这是该管线和检测范围的报告，不代表能覆盖所有个人信息类别。[Dolma 论文](https://arxiv.org/abs/2402.00159v2)。
- DataComp-LM 7B–2x 去重消融的正式会议论文表 19 报告，BFF（13-token n-gram、10 shards）取得 MMLU 44.3、Core 45.3、token yield 3T；MinHash + suffix array（16 shards）取得 44.4、45.5、3.2T。作者将其解释为该语料池中的相近评测表现和不同 token yield，不能外推成算法普遍优劣。[DataComp-LM 正式论文表 19](https://papers.neurips.cc/paper_files/paper/2024/file/19e4ea30dded58259665db375885e412-Paper-Datasets_and_Benchmarks_Track.pdf)。
