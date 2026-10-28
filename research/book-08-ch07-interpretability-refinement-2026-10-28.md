# 第八册第七章 Interpretability 与 Mechanistic Interpretability 研究与精修记录

- 日期：2026-10-28。
- 范围：按当前 WRITING_SPEC 复读第七章并复核 2026-10-27 章级扫描；围绕 SAE 质量验证、自动 feature 解释的反例、机制干预的因果边界、安全案例和形式化 circuit 研究查看一手来源。旧扫描作为研究线索，不作为正文完成证据。
- 章节职责：从输入归因、表示读出、因果干预推进到局部机制假设；不把 feature 标签当成语义真值、把推理时 steering 混为参数编辑，或将局部机制证据写成安全保证。

## 研究结论与正文落点

| 来源 | 支持的判断 | 适用边界 |
| --- | --- | --- |
| [SAEBench](https://arxiv.org/abs/2503.09532), arXiv v4 / ICML 2025 | 以 8 类指标比较 200 多个 SAE 和 8 种架构；旧 proxy 指标提升不能稳定推出 feature disentanglement 或实际任务效用提升。 | 结论绑定论文的模型、SAE、训练配置与指标；不能据此认定某架构普遍最好或所有 proxy 都无用。 |
| [MIB: A Mechanistic Interpretability Benchmark](https://proceedings.mlr.press/v267/mueller25a.html), ICML 2025 | 两条 track、四类任务、五个模型；attribution 与 mask optimization 在 circuit localization 较强，supervised DAS 在 causal-variable localization 较强，论文所测 SAE features 在后一 track 未胜过普通 hidden dimensions。 | 任务依赖作者建立的高层因果变量和 benchmark；作者讨论 ground-truth completeness、任务范围等局限，不构成全域方法排名。 |
| [Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?](https://arxiv.org/abs/2602.14111), arXiv 预印本 | 合成已知真值设置中，71% explained variance 对应 9% 真 feature recovery；指定真实激活实验的 baseline 与训练 SAE 分数接近（interpretability 0.87/0.90、sparse probing 0.69/0.72、causal editing 0.73/0.72）。 | 单篇预印本，数字只绑定作者的合成分布、模型、SAE 变体、baseline 和评估器；不能外推为 SAE 普遍无效。 |
| [Revising and Falsifying Sparse Autoencoder Feature Explanations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b6ce2529a79862a2d8ac57ac0e01d90-Abstract-Conference.html), NeurIPS 2025 | 通过相似负例挑战过宽 feature 解释，并使用结构化解释和迭代细化。 | 改进特定自动解释生成与评估流程；解释文本依旧是可证伪假设，不是 feature 语义真值。 |
| [LLM Forensics](https://arxiv.org/abs/2609.07746), arXiv v2 / Findings of EMNLP 2026 | 在 1B、8B 模型的无害固定触发器语言切换任务中，SAE 特征可检测 trigger，但 attention/MLP 特征的消融或激活常不能控制切换；部分 residual-stream feature 在该设置中能抑制或诱发行为。 | 任务是受控的英译法/德无害触发器，不是恶意后门、生产系统或一般模型的安全结论。 |
| [Formal Mechanistic Interpretability](https://proceedings.iclr.cc/paper_files/paper/2026/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html), ICLR 2026 | 形式化输入域鲁棒性、连续干预下 robust patching 与 circuit minimality，并在视觉模型上做实验。 | 证明范围依赖指定 circuit、输入域和性质；视觉实验不是前沿语言模型或部署系统安全证明。 |

## 候选去向与架构边界

- 不增章。SAEBench、MIB 和 SANITY 放在 §8.9，用来解释重构、稀疏、可解释性、因果定位和实际效用不可互换。
- 近邻负例证伪放在 §8.8；不把自动解释器输出升级为人工语义标签。
- LLM Forensics 放在 §9.2，说明 trigger detection 与行为控制需分别测量；保留无害任务边界。
- 形式化 circuit 结果放在 §13.4，修正“形式化保证仍完全开放”的过宽说法，同时说明其视觉模型范围。
- 模型参数级持久 unlearning 继续交给第九章；本章只讨论机制定位、干预证据及其交接。

## 正文、公式和数值修订

- 章首 text 链改为 Mermaid 证据闭环：行为观察、归因/读出、候选机制、patch/ablation/负对照、holdout 和行为回归。
- §6.5 补 demo 同输入逐步演算：clean margin 2.0、corrupted margin -1.0、patched margin 1.4；分子 2.4、分母 3.0，恢复分为 0.8。解释它只表示该合成干预恢复 logit-margin 差距，不表示准确率提高 80% 或自然路径充分。
- 展开 §8 SAE 字典训练所需的 checkpoint、层位、token 位置、激活分布、字典宽度与稀疏惩罚；解释稀疏与重构的取舍及其不能直接优化语义正确性或行为控制。
- §8.8 增近邻负例的边界检验和 NeurIPS 2025 证据；区分 feature 语义解释、行为相关性与安全因果性。
- §8.9 加 SAE 指标对照研究和六个 demo 向量的手算：第一行平方误差 0.02、总 SSE 0.09、均值约 (0.70, 0.70, 0.717)、SST 约 13.028、重构保真度约 0.993；阈值 0.5 下平均激活数为 1.0；code feature top-2 purity 为 2/2=1.0。逐项解释这些观察的分母与限制。
- §9 增无害触发器检测/控制分离证据；§13.4 增 ICLR 2026 形式化机制工作及 vision-only 实验边界。
- 展示公式迁移为单美元行内式和 math 围栏，移除 operatorname；章首图使用标准 Mermaid 围栏、加引号节点和管道标签虚线箭头。

## 篇幅与代码验证

- §1—§13 主线解释性正文 10,475 个汉字，高于 1,800 下限。
- §8 SAE 单元 2,772 个解释性汉字；§14 案例 1,511 个解释性汉字，高于 1,000 的辅助单元下限。
- 唯一 Python demo 在 Python 3.12.9 标准库运行，固定输出逐行匹配正文预期块。主要值：patch scores 0.4/0.2/0.8/0.7，max ablation 1.6，path scores 0.75/0.583，reconstruction fidelity 0.993，平均激活数 1.0，三项 top-2 purity 为 1.0。
- 全空输入路径：七个关键指标都显式为 None，决策为 expand_measurement_before_interpreting_metrics；top-k 少于 2 条时所有 purity 为 None 并扩展测量；分母 10^-12 返回 None。向量维度不一致会显式抛出 ValueError，避免 zip 静默截断。
- 未新增仓库测试文件，未运行测试套件；该 demo 不连接真实模型、权重、SAE、hook、harness 或用户流量。

## 配套同步与未关闭门禁

- PAPERS.md 增 SAE benchmark、随机基线、近邻反例、trigger 控制与形式化 circuit 研究入口。
- EXERCISES.md 增 patch、SAE 指标手算、基线比较和缺测传播题。
- INTERVIEW_BANK.md 扩写 SAE 指标、匹配基线及检测/控制边界；术语表已有相关术语，本轮未重复编辑。
- 本地 Markdown 数学围栏、行内美元配对和 Mermaid 源码结构检查通过；目标 GitHub Mermaid/MathJax 渲染尚未验收。未做全章外链 HTTP 扫描或 PDF 构建，因此本章不标记完整门禁通过。
