# 混合序列架构：局部注意、全局读取与递归状态的证据

研究日期：2026-11-24
范围：第二十一册第 65 章，识别局部 attention、全局 attention 与递归/线性序列模块的区别，核对层间和层内组合、状态与缓存含义、代表性模型的公开结构及组合证据。未训练模型、未复现代码或 benchmark。

## 检索记录

从论文、官方技术报告/模型页面和研究团队博客并行检索架构事实与解释材料；另查课程讲义及教学 notes，辅助设计从“可见历史—保存形式—读取方式”展开的讲法。检索词包括 hybrid attention / Transformer-SSM architecture、local sliding-window attention、global attention、recurrent state、inter-layer / intra-layer fusion、Griffin、Jamba、Hymba、Qwen3.8-Flash-Next、Kimi K3。对模型结构、版本和实验数字以论文/技术报告为准；教学材料只用来观察讲解顺序和图示，不承担普遍技术结论。

## 主要来源

| 来源与版本 | 实际阅读位置 | 支持的判断 | 不支持的判断 |
| --- | --- | --- | --- |
| Soham De et al., [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427), arXiv v1, 2024-02-29 | §2.3—2.4、§3、§6.2、附录 E/F | Griffin 在递归模块与 local MQA 之间分配序列混合；论文展示特定窗口和训练条件下的复制/检索结果。它提供“局部窗口直读 + 递归跨距传递”的结构实例 | 不支持递归状态能无损回读任意历史，或 Griffin 的收益可推广到所有混合模型；synthetic 任务结果受模型规模、训练长度和窗口设置限制 |
| Opher Lieber et al., [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887), arXiv v2, 2024-07-03；ICLR 2025 | §2、§3.1—3.2、§6.1—6.2 | 报告的 Jamba 配置以每 8 层 1 个 Transformer attention 层和 7 个 Mamba 层交错，MoE 频率另行设置；1.3B/250B-token ablation 中，1:3 与 1:7 在其所测指标上相近 | 不能据此推出通用最优层比、MoE 与 attention–Mamba 组合的独立因果效应，或跨硬件的吞吐结论 |
| Xin Dong et al., [Hymba: A Hybrid-head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676), arXiv v1, 2024-11-20；ICLR 2025 | §2.1—2.4、§3.3、附录 B—D | Hymba 在同一层并行放置 attention 与 Mamba/SSM heads；模型还结合局部/全局 attention、跨层 KV sharing 和 meta tokens，适合区分“层内并行”与“层间交错” | 整体质量、cache 和 throughput 不能单独归因 hybrid-head；数字只属于对应规模、训练量、硬件和测量协议 |
| Sangmin Bae et al., [Hybrid Architectures for Language Models: Systematic Analysis and Design Insights](https://arxiv.org/abs/2510.04800), arXiv v3, 2026-04-21 | §§3—4.6、附录 N/P、§6 | 系统比较 inter-layer 与 intra-layer 设计，并研究比例、放置和尺寸等变量；作者报告实验覆盖最高 3B 模型、60B 训练 tokens | 不足以将该实验中的设计建议推广到更大模型、更多训练预算、不同算子和生产 workload；作者明确把更大规模验证列为后续工作 |
| Qwen Team, [On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320), arXiv v1, 2026-08-31 | §2.1.1—2.1.2，尤其 GDN 架构消融与 QSA 描述 | Qwen3.8-Flash-Next 报告公开每四层三个 GDN 层加一个全局 attention 层；继续预训练阶段又以 Qwen Sparse Attention（QSA）替换该全局层。报告区分固定递归状态、全局 attention 的逐 token 读取与稀疏索引 | 报告中的消融绑定 28 层、25B-A3B 检查点、训练阶段、上下文和评估 pipeline；不能把单一分数、QSA/GDN 单项或模型长上下文结果概括为普遍规律 |
| Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653v2), arXiv v2, 2026-08-07 | §2.1、§2.1.1—2.1.2、§3.4 | 报告披露每组 3 层 KDA 后接 1 层 Gated MLA，并在主干末尾另设全局 MLA；同时说明 KDA 的位置敏感/近期性路径与 MLA 全局内容交互的分工 | 1M 上下文是模型、长上下文数据、渐进扩展训练和基础设施共同组成的系统结果；不构成 KDA/MLA 比例或任一位置机制的单项因果证明 |

## 教学组织入口

| 材料 | 实际阅读位置 | 使用方式与边界 |
| --- | --- | --- |
| Berkeley Scalable AI, [S2026 Lecture 3: Architectures to Break Bottlenecks](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture3.pdf) | 长上下文架构讲义第 43 页的 hybrid stack | 用于观察如何把“层深度上的混合”与“层内稀疏/并行路径”画成不同问题，并通过 KV 与 SSM state 解释路径账本；不当作某模型性能的原始证据 |
| Berkeley Scalable AI, [S2026 Lecture 8: Hybrid Mamba-Attention](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture8.pdf) | hybrid architecture、state/cache 对照页 | 用于检查可视化层表、窗口/状态/全局层分工的初学者路线；图中成本数字必须按本章所引用模型报告单独核对 |
| AI21, [Introducing Jamba](https://www.ai21.com/blog/announcing-jamba/) | Jamba block 图及其层间/MLP/MoE 解释 | 作者团队博客可帮助定位结构图与术语；架构比例和实验主张以 Jamba 论文为准 |

## 本章采用的结论边界

- “局部/全局”首先描述 attention 可访问的历史范围；“递归状态”描述一种跨 token 保存历史的计算与表示方式。三者不是互斥的模型类别，也不是每个 hybrid 必须同时具备的三个分支。
- 同层并行和不同层交替改变信息融合的时点；局部/全局 attention mask 是另一条设计轴。不能把 sparse attention、窗口 attention 和递归状态都笼统叫作“压缩记忆”。
- 全局可见不保证模型正确定位证据；全局路径采用 latent cache、shared KV 或稀疏索引时，需依据对应结构判断 payload 与可见性。递归状态的序列长度固定也不表示能无损回读逐 token 内容。
- 各模型层比、放置方式与实验收益绑定报告版本和训练/评测条件。Qwen3.8-Flash-Next 的报告尤其显示，full attention 和 QSA 在不同训练阶段承担不同作用；Kimi K3 的 1M 窗口不能归因于一个 attention ratio。
- 本章解释读取模型结构和提出路径级评估问题；不提供通用 serving allocator、scheduler、rollback 或故障排查指南。未进行本地复现。
