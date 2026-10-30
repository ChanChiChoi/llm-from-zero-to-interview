# RoPE 扩展与长上下文评测：事实和证据

检索日期：2026-10-28。

## RoPE 的位置关系

- Su 等人的 RoFormer v4（arXiv 版本日期 2022-08-09）§§3.1–3.3 将 RoPE 表述为：Query 与 Key 的内容投影按位置乘以旋转变换，使点积中的位置依赖以相对位移出现；一般形式给不同二维子空间使用不同频率。论文 §3.3 还在其指定频率设置下推导长距离衰减性质。来源：[RoFormer v4 HTML](https://arxiv.org/html/2104.09864v4)，实际阅读摘要、§§3.1、3.2.1、3.2.2、3.3 和理论推导入口。证据等级：原始论文的公式与理论分析。边界：相对位移进入点积的公式性质不等于整句输出只取决于相对距离；论文关于频率选择的衰减性质不证明任意实现、任意模型或任意长序列都有稳定性能。

## RoPE 位置扩展方案

- Chen 等人的 Position Interpolation (PI) v2 于 2023-06-28 修订。其方法把更长目标窗口的位置索引线性缩小到预训练位置范围内，避免直接把未见过的大位置索引送入原 RoPE 映射；论文在 LLaMA 模型的长文本建模、passkey retrieval 和摘要等实验中报告了扩展结果。来源：[PI v2 HTML](https://arxiv.org/html/2306.15595v2)，实际阅读引言、§2.2、§2.3、§§3.3–3.5。证据等级：论文方法定义、理论分析和指定模型实验。边界：效果受模型、训练数据、微调步骤和评测任务限制；索引压缩改变了位置间隔，需评估原窗口回归，不能把理论界或特定扩展长度说成所有模型保证。
- Peng 等人的 YaRN v3 于 2026-02-06 修订。§§3.1–3.3 讨论高/低频 RoPE 插值及注意力 softmax 温度等组合；实验主要围绕论文声明的模型与微调设置。来源：[YaRN v3 HTML](https://arxiv.org/html/2309.00071v3)，实际阅读摘要、引言、§§3.1–3.3 和 §4.1 训练设置。证据等级：论文方法和作者报告的实验。边界：其效率倍数与可扩展长度仅适用于论文对比条件；不视作所有模型默认配置或普遍吞吐结论。

## 长上下文使用能力与评测

- Liu 等人的 Lost in the Middle v3（2023-11-20）研究多文档问答与 key-value retrieval，报告被检索证据放在上下文中段时，一些模型在其任务设置下表现低于证据位于开头或末尾。来源：[论文 v3 摘要](https://arxiv.org/abs/2307.03172)，阅读摘要和版本历史。证据等级：原始论文的模型与任务实验。边界：不证明所有模型、位置分布、任务或当前版本都呈现相同曲线。
- Bai 等人的 LongBench v2 发表于 ACL 2025，包含 503 道选择题，覆盖单文档/多文档问答、长上下文示例学习、长对话历史、代码仓库和结构化数据等任务；论文摘要列出的上下文范围从 8k 到 2M words。来源：[ACL Anthology 页面](https://aclanthology.org/2025.acl-long.183/)及[论文 PDF](https://aclanthology.org/2025.acl-long.183.pdf)，阅读摘要、引言和数据构造说明。证据等级：同行评审 benchmark 论文及其评测设计。边界：其任务与评分覆盖面仍是一个具体数据集，不等同所有生产长文本场景。
- Yang 等人的 100-LongBench 发表于 Findings of ACL 2025。论文指出不同长上下文 benchmark 可能有合成内容、单样本固定长度和基座知识与长上下文能力混杂等局限，并提出可控长度和基准能力对照设计。来源：[ACL Anthology 页面](https://aclanthology.org/2025.findings-acl.903/)及[论文 PDF](https://aclanthology.org/2025.findings-acl.903.pdf)，阅读摘要、引言、§3.2 LongScore 指标和比较结果。证据等级：同行评审 benchmark 论文及作者提出的评估方案。边界：这是作者对现有 benchmark 的分析和其方案，不代表所有研究都采用同一指标或完整解决评测效度。
- Liu 等人的 LongSocialBench 发表于 EMNLP 2026。论文使用完整线上讨论线程和回复树，重点测试结构化社会讨论证据，不只测试长段文本检索。来源：[ACL Anthology 页面](https://aclanthology.org/2026.emnlp-main.896/)，阅读摘要和评测设计摘要。证据等级：同行评审的任务专项 benchmark。边界：Hacker News、Stack Exchange 与 Reddit 讨论树是特定任务，不代表一般文档理解能力。
- Zhang 等人的《Positional Failures in Long-Context LLMs》于 2026-05-22 提交 arXiv。论文提出在推理评测中同时控制任务位置、填充内容和上下文长度，并报告其九个模型、指定推理任务下的位置与填充交互。来源：[arXiv HTML](https://arxiv.org/html/2605.23170)，阅读摘要、引言、评测设计和结果部分。本轮只以 arXiv 页面为来源，未核验其同行评审版本。证据等级：预印本中的评测方案和模型/任务实验。边界：其模型、prompt/filler、样本数、任务和配置限定结果；不把其数值外推成现有模型的通用中段失败率。
- Du et al., “SagaScale: A Realistic, Scalable, and High-Quality Long-Context Benchmark Built from Full-Length Novels,” arXiv:2601.09723v1（2025-12-27），[arXiv 摘要](https://arxiv.org/abs/2601.09723)。实际阅读摘要和版本信息。作者报告这是英中双语长篇小说 benchmark，平均上下文分别超过 250K 和 320K token，并以 12 个模型比较 Naïve RAG、Agentic RAG 与 full-context 三种方案；在其评测中，直接提供全文有时优于检索方案，Agentic RAG 可缓解 Naïve RAG 的检索瓶颈。证据等级：preprint / source-backed（预印本摘要中的 benchmark 设计和作者实验结论）。边界：具体结果绑定小说语料、问题构造、被测模型及其版本；不能推导为长上下文普遍优于 RAG，或 Agentic RAG 普遍优于 Naïve RAG。

## 未验证边界

- 未运行 RoPE、PI 或 YaRN 教学实现；未 profile attention kernel、显存或长上下文服务。
- 本章位置旋转和 PI 缩放小例均由公式构造；它们不是模型质量、长度上限或服务性能实测。
- 真实服务的有效输入还受 tokenizer、聊天模板、输入截断、输出预留、图像/音频编码和检索预算影响；具体行为需按目标模型与 API 版本核实。
