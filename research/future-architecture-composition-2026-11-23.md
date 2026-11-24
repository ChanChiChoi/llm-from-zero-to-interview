# 未来大模型架构：组合粒度与证据边界

研究日期：2026-11-23。研究问题：公开的序列模型架构研究把注意力、递归状态、记忆和稀疏容量放在什么组合位置；当前证据是否足以断言某一种未来架构会占主导。范围限于模型内的信息路径与结构证据，不把服务框架、工具协议或 Agent 运行时当作模型架构。

检索词包括 `hybrid language model architecture block head memory branch`、`Jamba Hymba hybrid architecture ablation`、`Titans memory architecture variants`、`systematic analysis hybrid architectures scale`、`Mamba-3 state tracking`、`score-level SSM attention fusion`，并查找研究者解释文章及高校课程讲义。主要结论均回到一手论文/正式出版页核验；预印本只作为其作者提出设计与报告结果的证据，不据此判断社区采纳或未来胜负。

## 研究发现

1. **组合可以发生在不同粒度。** Jamba 在层间交错 Transformer、Mamba 与 MoE；Hymba 在同一层内并行配置 Attention/SSM heads；Titans 将测试时更新的 neural memory 与短期 Attention、persistent memory 组合，并比较 context、layer、gated-branch 三种接入方式。2026 年 SISA 预印本进一步提出把 SSM 派生的重要性信号加入 Attention score。它们描述不同数据路径，不能统称为一种“hybrid block”。
2. **递归架构本身也仍在演进。** Mamba-3 论文报告在 1.5B 规模、其指定模型与任务中改进 retrieval、state tracking 和语言建模，并提出更表达性的递归、复数状态更新和 MIMO。该结果说明替代路径仍可单独改进；它不意味着该架构在其他规模、任务或硬件上优于 Transformer。
3. **系统性消融有帮助，但外推边界仍在。** Bae 等人的 hybrid 系统比较研究区分 inter-layer 与 intra-layer 融合，实验覆盖 100M 至 3B 等规模与其设定的预训练预算；更大模型及其训练配方不在该实验范围内。其比例、层位置和质量/效率结果应绑定论文实验，不能成为通用最优配方。
4. **组合的收益不是各组件收益的简单相加。** Jamba 的 MoE 与训练配置、Hymba 的 meta tokens 和 KV sharing、Titans 的测试时 memory update 都会共同影响整体结果。要把成绩归于某一结构因素，需要对应消融或匹配基线；仅观察发布模型的总分无法隔离组件贡献。
5. **当前可下的结论是设计空间在扩展，而非已出现统一胜者。** 层间、层内、记忆分支和分数层融合均有研究提案或实验；单一路径也继续改进。异构的模型规模、数据、任务和硬件结果没有组成跨研究统一排名。“未来架构将以可组合形式为主”只能作为待验证的趋势推断，不能写成确定事实。

## 一手来源与使用边界

| 来源 | 版本与阅读范围 | 支持的主张 | 不支持的外推 |
| --- | --- | --- | --- |
| Lieber et al., [Jamba: Hybrid Transformer-Mamba Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a9ed43fa31dc8b4a7d7a673d713dcb5f-Abstract-Conference.html) | ICLR 2025 正式论文页与论文摘要/架构说明 | 层间交错 Transformer、Mamba；部分层加入 MoE；作者研究了若干架构决定 | 不支持混合架构普遍优于所有 Transformer，也不支持某个层比例适用于所有任务 |
| Dong et al., [Hymba: A Hybrid-head Architecture for Small Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/f32def07618040e540e0a6182e290562-Paper-Conference.pdf) | ICLR 2025 正式论文，摘要、§§2.1–2.2 和消融 | 同层并行 Attention/SSM heads；模型还联合使用 meta tokens、局部/全局 Attention 与 KV sharing | 不可把总模型 benchmark 或吞吐单独归因为 hybrid-head；结果受模型尺度、训练和评测设置限制 |
| Behrouz et al., [Titans: Learning to Memorize at Test Time](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html) | NeurIPS 2025 正式论文页与论文 PDF，摘要及 §4 | 神经长期记忆与短期 Attention、persistent memory 组合；记忆有 context/layer/gated-branch 变体 | 不支持把所有架构 memory 都看成测试时可训练神经记忆，亦不支持所有变体在任务上等效 |
| Bae et al., [Hybrid Architectures for Language Models: Systematic Analysis and Design Insights](https://arxiv.org/abs/2510.04800v3) | arXiv v3，2026-04-21；摘要、inter-/intra-layer 定义、实验尺度及局限 | 对不同融合粒度做系统比较；给出 100M–3B 等模型设置及具体比例/位置实验 | 预印本实验不能推出更大基础模型或生产硬件上的通用胜负；作者列出的设计建议仍绑定研究设置 |
| Lahoti et al., [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569) | arXiv v1，2026-03-16；摘要与方法概述 | 更表达性的递归、复数状态更新与 MIMO 是同一 SSM 家族继续演进的路径；报告结果限于其任务和 1.5B 对照条件 | 不支持 SSM 家族整体已取代 Attention，亦不支持跨硬件的速度结论 |
| Shin & Yang, [Forget Attention: Importance-Aware Attention Is All You Need](https://arxiv.org/abs/2606.02332) | arXiv 预印本，2026-06-01；摘要及架构图/引言 | 提出 SISA，把 SSM 派生项加入 Attention score；这是分数层融合的一项新研究提案 | 预印本的小规模实验不证明其已被采用、可扩展或比层间/层内融合更优 |

## 教学与组织来源

- Jiankai Sun 与 Chuanyang Zheng，[Rethinking the Primitives: Next Generation LLM Architecture](https://web.stanford.edu/~jksun/blog/llm-architecture.html)，2026-04。阅读“Architecture Stack”与 hybrid attention 部分，用于观察如何从模型组件重新组织未来架构问题；作者的广泛效果判断不作为独立证据。
- UC Berkeley Scalable AI，[Hybrid Mamba-Attention / Nemotron-3 Pre-Training, Lecture 8](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture8.pdf)，2026-02-12。用于观察课程怎样并列呈现架构、训练决策和硬件条件；幻灯片中的产品/性能数据只代表所展示的模型与配置。
- Tan Ke，[Titans paper-reading notes](https://mrtanke.github.io/posts/2025-11-26-titans-learning-to-memorize-at-test-time/)，2025-11-26。阅读其架构变体与 memory update 解释，并与 NeurIPS 正式论文核对；笔记仅帮助组织说明，不独立支撑实验主张。

## 关联研究记录

- [Attention–SSM 混合架构](hybrid-attention-ssm-architectures-2026-11-23.md)：层间与层内融合、比例/位置消融。
- [MoE、SSM 与 Attention 组合](moe-ssm-attention-composition-2026-11-23.md)：专家化插入位置及对应实验条件。
- [模型记忆架构](memory-augmented-transformer-architectures-2026-11-23.md)：模型内记忆模块及其信息路径。
- [多模态 Mamba 架构](multimodal-mamba-architectures-2026-11-23.md)：多模态与长视频中的结构位置。
- [Agentic Model Architecture](agentic-model-architecture-boundaries-2026-11-23.md)：模型能力与外部执行闭环的边界。
