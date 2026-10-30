# Tokenization、压缩与多语言评估：来源证据

记录日期：2026-10-29

本记录只保存联网核验到的研究事实、版本、实际阅读范围和证据边界；书内是否采用及修改决定记录在对应 `progress/` 文件。

## 跨语言 tokenizer 成本

- **来源：** Rust et al., [Explaining and Mitigating Crosslingual Tokenizer Inequities](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5b91cefbfa52340af2f16f249572dc76-Abstract-Conference.html)，NeurIPS 2025；[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5b91cefbfa52340af2f16f249572dc76-Paper-Conference.pdf)。
- **阅读范围：** 官方摘要、PDF §2.1–2.3、§3、§4 和 §5.1–5.3。
- **支持事实：** 作者在 97 种语言上训练/比较约 7,000 个单语 tokenizer，并在平行语料上比较同一内容的 token 数。词表规模和 pre-tokenization 会影响跨语言压缩成本；仅增加词表规模不能保证消除差异。其单语实验中的 per-language optimal vocabulary 与 SuperBPE 降低了 token 数或语言间 token-count 方差。
- **证据等级：** source-backed（论文给定语言、平行语料、tokenizer 训练和压缩指标范围内的实证结果）。
- **适用边界：** 跨语言 word segmentation 不稳定，因此同内容 token count 不应与 token/word fertility 混同。实验以单语 tokenizer 为主，没有验证一个通用的共享多语言词表，也没有直接测量这些 tokenizer 下游语言模型的任务质量。

## 非规范切分与模型鲁棒性

- **来源：** [Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2b5c5689fae6fa9a4883e73e511d52c8-Abstract-Conference.html)，NeurIPS 2025；[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/2b5c5689fae6fa9a4883e73e511d52c8-Paper-Conference.pdf)。
- **阅读范围：** 官方摘要及论文主要结果和任务切分实验。
- **支持事实：** 研究在 20 个基准上构造模型训练期间未见的非规范 token 序列。指令微调模型在抽样非规范切分下最高保留 93.4% 的原聚合表现，字符切分下最高保留 90.8%；特定字符串操作/代码理解任务的字符切分结果最高改善 15%，右对齐数字分组在特定大整数运算上改善超过 33%。
- **证据等级：** source-backed（论文所测模型、基准、切分扰动和任务结果）。
- **适用边界：** 不同预训练与指令微调模型表现不同；这些结果不能证明部署时替换 tokenizer、special-token id、template、streaming decoder 或 cache 是安全的。

## 动态词表压缩

- **来源：** [zip2zip](https://proceedings.neurips.cc/paper_files/paper/2025/hash/def682d57535b9e778508b4b1a911034-Abstract-Conference.html)，NeurIPS 2025；[正式论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/def682d57535b9e778508b4b1a911034-Paper-Conference.pdf)。
- **阅读范围：** 官方摘要、引言、§2.1–2.4 和实验结果表。
- **支持事实：** 方法使用 LZW 形成上下文相关的动态 hypertoken，并为动态输入/输出词汇学习配套 embedding/unembedding；作者报告对既有模型做 PEFT uptraining 约需 10 GPU 小时，输入/输出 token 序列减少 15%–40%。在论文指定模型、上下文/生成长度和硬件下，部分实验的端到端延迟改善最高约 40%。
- **证据等级：** source-backed（论文列出的 Phi 系列模型、特定 prompt/generation 工作负载及 H100/Apple M1 测量范围）。
- **适用边界：** token 序列缩短不等于同幅度端到端加速；动态词表维护、输出投影和设备实现都会影响总时延。论文结果不能视为生产 tokenizer 迁移的普遍保证。

- **来源：** [zip2zip++](https://arxiv.org/abs/2610.13605)，arXiv v1，提交于 2026-10-09；[HTML 正文](https://arxiv.org/html/2610.13605v1)，[PDF](https://arxiv.org/pdf/2610.13605)。
- **阅读范围：** 摘要、§1、§3.1–3.4、§4.1、§5 表 1–3、附录 D.4 与 D.8。
- **支持事实：** 论文增强动态 token 的 embedding 几何、size-aware RoPE、LZW 合并容错和多视角 PPL。实验覆盖 4 个 instruction-tuned decoder；LoRA 训练 8,000 步，使用 1B 微调数据，最大合并长度 $M=4$。Phi-3.5-mini 的 GSM8K 从 zip2zip 的 0.36 升至 0.68（未压缩模型 0.75），HumanEval pass@1 从 0.26 升至 0.32（未压缩模型 0.58），IFEval 未改善（0.32 对 0.38）；较大模型的一些代码/指令指标仍可能低于 zip2zip。
- **证据等级：** preprint / source-backed（arXiv v1 预印本中的有限模型、数据和任务实验）。
- **适用边界：** 该预印本显示特定压缩方案在数学与困惑度上恢复更多质量，但代码和指令遵循仍有差距；不构成已完成生产验证或全面能力恢复的证据。

## 非洲语言 token fertility 与任务结果

- **来源：** [The Token Tax: Systematic Bias in Multilingual Tokenization](https://aclanthology.org/2026.africanlp-main.10/)，AfricaNLP 2026 / ACL Anthology；[正式论文 PDF](https://aclanthology.org/2026.africanlp-main.10.pdf)。
- **阅读范围：** 摘要、§3 的指标定义、实验结果与限制讨论。
- **支持事实：** 作者在 AfriMMLU 上评估 10 个 LLM、5 个学科和 16 种非洲语言，报告 tokens-per-word fertility 与准确率的负相关；成本分析用 attention 随序列长度平方变化的假设估算 token 膨胀开销。
- **证据等级：** source-backed（论文评测模型、语言、学科和 AfriMMLU 协议内的相关性结果）。
- **适用边界：** 作者指出 fertility 与预训练数据的可用量/质量等因素共同变化，因此该关联不识别 tokenizer 的单独因果效应。平方成本是模型假设，不是所有长度、硬件或端到端服务中的实测规律。

## 生命周期

这些来源是 token 成本、公平性和压缩—质量取舍的 active 研究入口。跨 tokenizer 的模型质量、服务性能和迁移安全仍需按目标语料、模型、训练配方、硬件和请求分布分别测量。
