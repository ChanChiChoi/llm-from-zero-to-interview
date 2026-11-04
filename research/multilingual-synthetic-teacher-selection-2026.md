# 多语言合成数据的 teacher 评估：来源证据

记录日期：2026-11-03

本记录只保存研究事实、来源版本和证据范围，不记录章节审阅进度或修改决定。

## Polyglot Teachers

- Miranda, Vulić, and Korhonen, “Polyglot Teachers: Evaluating Language Models for Multilingual Synthetic Data Generation,” arXiv:2604.11290v3，版本日期 2026-08-28，[固定版本 HTML](https://arxiv.org/html/2604.11290v3)，[PDF](https://arxiv.org/pdf/2604.11290v3)。
- 实际阅读范围：摘要、§§2.1–2.4 的 seed、生成和 student 评估设置；§3.1 的模型规模/benchmark 回归；§4.1–4.2 的预测分析；§§5–6 的讨论与 Tagalog 案例；Limitations；附录 I 中 held-out FilBench 的设置和结果。
- 来源支持：论文比较 10 个 teacher、6 种类型各异的语言，生成超过 140 万条 SFT 样本并训练 240 个 student。teacher 生成策略包括 Generate、Translate 和 Respond。参数规模在其回归中不显著预测 PG-Score（9 个已知参数规模的模型、3 次运行，$N=27$）；多语言 benchmark 平均表现也未显著预测 PG-Score（10 个模型、6 种语言、3 次运行，$N=180$）。作者报告内在质量指标的前四个主成分解释了 93.3% 的指标方差；以这些主成分预测 student 表现的线性模型用 80/20 训练/测试切分，测试集报告 $R^2=0.664$、RMSE $=0.440$。
- 外部验证案例：作者将生成配方应用到未用于 PG-Score 计算的 Tagalog，并在 FilBench 上进行消融。论文报告更换为较高 PG-Score teacher、匹配 teacher/student model family 等措施在该案例中带来可观察变化；此结果是一个中资源语言、特定生成 recipe、student 和 benchmark 的单个案例。
- 证据等级：source-backed（指定 arXiv v3 的作者实验、统计分析和案例；未作本地复现）。
- 适用边界：93.3% 是作者报告的内在质量指标 PCA 方差解释率，不是 student 能力方差解释率；$R^2$ 来自论文所选数据点、特征和随机切分，不代表跨语种或跨任务的普遍预测效力。§2.3 的公式列出四项聚合内在度量，而 Table 4 展示六个主成分；本记录不推断 PCA 输入特征数。统计上“不显著”不证明模型规模毫无影响。作者的语言集只有 6 种；Translate 假设可将英文 prompt 有意义地译到目标语言，可能引入文化定位缺失或 translationese。教师偏差也可能随生成数据传递给 student。
- 未解决问题：更多语系与更低资源语言上的预测稳定性；不同 seed 分布、prompt 翻译质量、生成策略和 student family 对 PG-Score 排名的影响；更大独立数据集上的预注册复核。
