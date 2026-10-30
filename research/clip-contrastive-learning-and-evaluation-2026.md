# CLIP、SigLIP 与 SigLIP 2 的目标和证据边界

- 检索与阅读日期：2026-10-29。

## CLIP

- 来源：Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，arXiv v1（2021-02-26）。阅读 HTML 全文中的摘要、§§2.2–2.3、§3.1.4–3.1.5、§5–6 和相关段落。
- 证据支持：原论文以图像/文本双编码器和对称 cross-entropy 训练 batch 内图文匹配；其训练数据 WIT 为作者构造的 400 million 图文对。单正样本 batch 中的对角线配对是训练标签，非对角线被当作不匹配组合，但论文也承认实际语义可能存在更复杂关系。
- Prompt 证据：§3.1.4 报告模板工程和在 embedding 空间集成模板能改变其零样本分类结果；具体收益只对应该论文所测数据集、模型与模板，不表示任意任务都能获得同等改善。
- 评估边界：§5 单独分析该论文模型训练数据与 35 个下游数据集的近重复重叠，§6–7 讨论模型局限和影响。该分析不估计后来 checkpoint 的训练重叠率、偏差或安全性。
- 证据等级：论文方法与作者实验为 source-backed；WIT 规模、benchmark 结果和重叠检测只对应该论文版本与实验设置。

## SigLIP

- 来源：Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)，发表于 ICCV 2023；本次读取 arXiv 摘要。另读 Google Research [big_vision SigLIP trainer](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py) 中的训练损失实现。仓库 `main` 是可变版本，访问日期为 2026-10-29。
- 证据支持：论文摘要说明 sigmoid 目标按图文 pair 计算，不需要 CLIP 式全局 softmax 归一化，并报告其特定小 batch、超大 batch 训练实验。官方 trainer 对 batch 的全体图文组合打分，将匹配对标为正、其他配对标为负，把缩放相似度加偏置后送入 log-sigmoid；实现对 pair 项求和后按 batch 行数取平均。
- 数学边界：若对 `B` 个成对样本处理完整 `B×B` 矩阵，来源实现的归一化等价于对 `B²` 项求和后除以 `B`；“对所有 pair 取平均”会多除一个 `B`，因此与其他 loss 混合时数值尺度不同。正文的一般式显式使用归一化量 `Z`，避免把不同 recipe 混为同一约定。
- 计算边界：不需要全局 softmax 分母，不等于全量组合数量线性下降。全体两两组合仍有 `B²` 个；分块、分片、负例抽样和通信实现会影响资源开销。
- 证据等级：论文摘要支持目标类别和作者报告结果；官方仓库实现支持访问时 `main` 分支上的具体标签、缩放、偏置与归一化路径，不证明所有 SigLIP 复现采用相同代码。

## SigLIP 2

- 来源：Tschannen et al., [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)，arXiv v1（2025-02-20）。阅读 HTML 全文摘要、§§2.1–2.4、§§3.1–3.5 和结论。
- 证据支持：作者把 SigLIP sigmoid 目标与 captioning/grounding decoder、自蒸馏、masked prediction、数据筛选及多语言训练组成分阶段 recipe；论文报告该系列在其检索、零样本、定位、稠密预测和 VLM encoder 实验中的比较结果。NaFlex 变体支持可变分辨率/长宽比也是作者方法描述。
- 证据边界：这是组合 recipe 的论文结果，不能把相对 SigLIP 的变化单独归因于 sigmoid 损失，也不能推出其他 checkpoint、任务、评估 harness 或线上系统一定有相同优势。作者报告的表格数值未在本地复现。
- 证据等级：source-backed，限定于论文报告的模型变体、数据、训练阶段和评测协议；无独立复现。

## 教学算例边界

- `s=1`、`b=0` 的单 pair 例中，正标签 logistic loss 为 `-ln(σ(1))≈0.313`；将同一相似度标作负例时为 `-ln(σ(-1))≈1.313`。这是根据公式计算的合成教学输入，不是模型输出或实测结果。
- CLIP 小矩阵 logits、softmax 概率和 prompt 变化同样是手算教学例；不作为任何模型、数据集或生产服务性能证据。

## 未验证项

- 未运行 CLIP、SigLIP 或 SigLIP 2 checkpoint，未复现论文训练、零样本 benchmark、检索或定位结果。
- 未审计任何后来 checkpoint 的训练数据与评估集重叠、授权、偏差或安全表现。
- 官方训练源码指向可变分支；需要讨论其他版本实现时，应先固定 commit 并重新核对。
