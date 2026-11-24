# RoPE 长上下文扩展研究记录

研究日期：2026-11-20。主题范围：预训练 RoPE 模型如何改变位置映射、如何用训练适配扩展后的映射，以及论文中的“可支持长度”能支持哪些结论。此记录只保留新知识和来源证据；章节组织与取舍见 progress/book-21-transformer-architecture-evolution/chapter-18-audit.md。

## 主题地图

长上下文扩展至少涉及三种不同变化：位置函数把测试位置映到哪些相位；模型权重是否在扩展映射或长序列上继续训练；所报告的“长度”是否只表示配置/输入上限，还是在指定任务和测试条件下有可用表现。RoPE 的相对位置几何由原论文和本册第七章承担；本章研究重点是如何改造位置映射和训练方案，不重推 RoPE 基础。

从统一压缩位置，到改变各频率缩放，再到逐维/逐位置搜索，方法逐渐从一个全局比例转向非均匀映射。继续预训练、短长窗口混合训练或位置扰动蒸馏改变的是模型适配路径，不能与“只改位置编码”的无训练方法混成一类。配置最大长度、训练序列长度和某个任务上的有效长度仍是不同概念。

## 一手来源及其支持边界

| 来源 | 阅读位置与版本 | 支持的事实 | 不支持的外推 |
| --- | --- | --- | --- |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/html/2104.09864) | arXiv v5，2023-11-08；第 3.2—3.4 节 | RoPE 将绝对位置写为 Q/K 上的旋转，并使注意力交互含有相对位移项；原实验包括长文本分类任务 | 原论文任务不验证任意长度下的 LLM 检索、推理或通用上下文能力 |
| [Extending Context Window of Large Language Models via Position Interpolation](https://arxiv.org/html/2306.15595) | arXiv v2，2023-06-28；第 2、3.1—3.5 节 | PI 将目标位置线性缩到原位置范围；作者在 LLaMA 7B—65B 上报告从 2K 扩到最多 32K，并以最多 1,000 步微调；原长度任务上有轻微退化报告 | 这些长度和效果绑定其 LLaMA、微调数据、训练步数与评测；不能称为普遍上限或所有任务的质量保证 |
| [YaRN: Efficient Context Window Extension of Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf) | ICLR 2024 正式论文；第 2.3、3.1—3.4、4.1—4.5 节及附录 A；同时复核 arXiv v3（2026-02-06） | 论文将 YaRN 定义为 NTK-by-parts 频率插值与 attention logit 温度缩放的组合；“NTK-aware”和 Dynamic NTK 被作者作为先前公开的社区方法回顾；论文报告其特定 LLaMA/Llama 2 设置下的训练效率和长序列结果 | NTK 名称不意味着该社区启发式直接由 NTK 理论推出；论文效率数字不可脱离模型、扩展倍率和对照训练方案外推。论文分段定义、具体代码实现需按所用版本区分 |
| [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/html/2402.13753) | arXiv v1，2024-02-21；第 2—4 节 | 作者识别维度和 token 位置两个非均匀性，以进化搜索得到缩放因子；方案先适配 256K，再逐步到 2,048K，并对 8K 短上下文作恢复；实验使用 LLaMA2 与 Mistral | 2M 是作者报告的特定模型、训练流程和任务结果，不代表无需同等训练/计算即可迁移到任意模型，也不代表普遍可靠推理 |
| [LongRoPE2: Near-Lossless LLM Context Window Scaling](https://arxiv.org/html/2502.20082) | arXiv v1，2025-02-27；第 2—4 节及附录 B | Microsoft 作者提出由 needle-driven perplexity 引导的因子搜索与混合上下文训练：长样本用重缩放 RoPE，短样本用原 RoPE；评估 LLaMA 3 8B 与 Phi-3 mini 3.8B | 128K effective-context 和短上下文保持数字是作者在指定模型、合成/真实任务及训练预算下的主张；该版为预印本，摘要和结论对“保持比例”的口径并不完全相同，正文不使用无条件数字概括 |
| [The Llama 3 Herd of Models](https://arxiv.org/html/2407.21783) | arXiv v3，2024-11-23；第 3.2、3.4.2 节 | 报告记载 Llama 3 使用 RoPE base frequency 500,000；Llama 3 405B 的受支持窗口从 8K 分六阶段扩至 128K，长上下文阶段约用 800B training tokens；作者用短上下文表现恢复与 needle-in-a-haystack 作为其适配判据 | 这是该报告的训练设计和判据，不是“只调 base”的因果实验，也不构成所有模型的通用训练配方或充分验收标准 |
| [MrRoPE: Mixed-radix Rotary Position Embedding](https://proceedings.iclr.cc/paper_files/paper/2026/file/69413f87e5a34897cd010ca698097d0a-Paper-Conference.pdf) | ICLR 2026 会议论文；方法第 3 节、实验第 4 节 | 将 PI、NTK、YaRN 解释为维度缩放策略，并提出均匀/渐进 mixed-radix 变换；无微调实验使用 LLaMA2 7B、LLaMA3 8B、Qwen2.5 3B，并考察 RULER、NIAH、InfiniteBench、LongBench | 论文摘要中的 128K NIAH recall 与相对 YaRN 比较只支持相应模型和任务条件；不证明该方案普遍优于带长上下文训练的方案 |
| [Jet-Long: Efficient Long-Context Extension with Dynamic Bifocal RoPE](https://arxiv.org/html/2607.07740) 与[作者仓库](https://github.com/jet-ai-projects/jet-long) | arXiv v4，2026-09-29；方法第 3 节、实验第 4 节与附录 A；作者仓库记录 NeurIPS 2026 接收 | 在本地 RoPE 保真窗口外，配合随当前长度变化的长程映射；作者在 Qwen3 1.7B/4B/8B、RULER、HELMET-RAG、PG-19 上测试到 128K，并补充 Llama 3 8B 与后训练 Qwen3 检查点 | 论文还需通过融合 attention/缓存校正完成系统实现；作者明确指出缓存角度校正不会重算历史 hidden states/values，因此保留缓存并不等价于整段重算。吞吐数字绑定 H100、软件/内核版本和对照设置；不属于位置映射效果的普遍保证 |
| [Shuffle the Context: RoPE-Perturbed Self-Distillation for Long-Context Adaptation](https://proceedings.mlr.press/v306/li26cs.html) | ICML 2026，PMLR 306:68155–68168；论文摘要与实验结果页 | 通过扰动同一训练序列的 RoPE indices 形成不同视图，并用 self-distillation 约束预测一致性；实验模型为 Llama 3 8B 和 Qwen 3 4B | 论文报告的 RULER 增益绑定其 SFT 设置、模型和任务，不表示位置扰动蒸馏是已验证的通用配方 |

## 当前研究中的方法边界

- **Position Interpolation（PI）：** 将目标位置除以扩展倍率，使新位置映回原训练区间；所有频率共同压缩，因此相邻 token 的相位差也随之缩小。训练数据和微调是论文实验的重要组成。
- **NTK-aware scaling：** 2023 年社区提出的 RoPE base 调整启发式；不同频率变化不相同，尝试减少统一插值对高频位置分辨率的影响。应称为社区方法/启发式，不把名称当成严格 NTK 推论。
- **Dynamic NTK：** 将缩放和当前序列长度关联，属于推理期动态策略；随长度变化的配置及缓存兼容属于具体实现，长序列状态复用回到第十一章。
- **YaRN：** 论文中的关键组合是按频率分段处理（NTK-by-parts）与 attention logit 温度校准。其训练步数/数据效率数字是作者对既有 PI 训练方法的指定比较，需保留实验条件。
- **LongRoPE / LongRoPE2：** 从频率组规则走向维度/位置搜索及逐步扩展；LongRoPE2 又将长短窗口训练分支并用。数百万或 128K 声明应逐个绑定模型、任务、训练步骤和版本。
- **MrRoPE（ICLR 2026）：** 以 mixed-radix 视角统一频率缩放策略，并用渐进变换做无微调实验；属于当前已发表方法，可作为“统一比例不再是唯一参数化”的示例。
- **Jet-Long（NeurIPS 2026）：** 将局部 RoPE 保真窗口与长度自适应的远程位置映射结合；论文的公开结果绑定其 Qwen3/Llama3 检查点和长上下文任务，且缓存校正不等价于重算历史状态。
- **训练适配不只是一种位置映射：** Llama 3 报告展示逐步增加训练序列长度；LongRoPE2 混用原/重缩放 RoPE；ICML 2026 的 Shuffle the Context 则改变位置扰动下的训练一致性目标。它们是不同训练路径，不能单凭摘要中的结果作横向排名。

## 教学与组织来源

| 来源 | 版本与阅读范围 | 可借鉴的组织方式 | 边界 |
| --- | --- | --- | --- |
| [Long context with RoPE scaling, YaRN and FlashAttention — Eva Koroleva](https://blog.korolevalab.com/blog/2026/transformer-architectures-09-long/) | 2026-09-30 发布、2026-11-10 更新；第 3—6 节重点读频率直觉、PI/NTK/Dynamic NTK/YaRN；作者以可运行 notebook 对照库实现 | 用不同转速的“时钟”解释频率组，再按变化对象并排解释 PI、NTK 和 YaRN；特别指出论文公式与模型库代码可能不同 | 本文还混合 FlashAttention、KV/cache 和具体模型 benchmark；高质量讲解不是跨版本行为的权威来源，代码/表现数字须另回论文或固定版本 |
| [AI602: Recent Advances in Deep Learning, Lecture 3 — LLM Inference & Test-Time Compute, Jinwoo Shin, KAIST](https://alinlab.kaist.ac.kr/resource/2026_FALL_AI602/AI602_Lec3.pdf) | 2026 Fall；Long-Context Inference 部分及“Position encoding and context extension”相关幻灯片（含 LongRoPE 页） | 将位置扩展放入长上下文推理单元，并按 PI、Dynamic NTK、YaRN、LongRoPE 的机制序列比较；位置编码之后再分讲 cache 与运行长上下文 | 课程幻灯片有明确教学组织价值；实验曲线和长上下文任务数字仍按原论文核验，本章不复制其 cache/serving 单元 |

## 检索范围与待核边界

检索平台：Web；检索语言：中英文；关键词覆盖 RoPE context extension、Position Interpolation、NTK-aware/Dynamic NTK、YaRN、LongRoPE/LongRoPE2、long-context scaling、tutorial/lecture notes，以及 2026 ICLR/ICML RoPE extension。首轮搜索同时寻找论文、工程讲解和课程 slides，再打开关键原文及正式会议页面核事实。未将 arXiv 综述、搜索摘要或仅摘要审阅的新候选作为正文证据。

2026 ICLR 的 frequency-band 分析与 RoPE-ID、2026 年 RoPE 理论限制预印本属于相邻研究：本轮仅核到官方摘要，因完整假设、模型设置和与本章读者结果的关系尚未审阅，不据其写普遍结论。若未来章节专题需要讨论“新位置机制”或长长度理论边界，应回原文重审；NoPE 与其他位置架构变化分别由本册第 64、67、72 章承担。
