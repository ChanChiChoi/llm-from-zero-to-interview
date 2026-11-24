# Attention 与 SSM 混合架构：融合方式与证据边界

研究日期：2026-11-23。主题范围：Decoder-only 语言模型中 Attention 与状态空间模型（SSM/Mamba）的层间交错、层内并行融合及其公开实验；重点核对融合粒度、比例/位置消融、任务条件和性能归因。没有运行本地训练、推理或 benchmark。

检索以原始论文、会议出版页、arXiv 固定版本和方法作者团队说明为主，并检查相关章节、消融表与实验条件。检索主题词（概括主题，不冒充逐字搜索引擎 query）：Attention-SSM hybrid architecture、inter-layer sequential hybrid、intra-layer parallel hybrid、Jamba ablation、Mamba-2-Hybrid、Hymba hybrid-head、hybrid layer ratio and placement、Mamba Transformer memory recall。关键方法和实验数值优先回到论文原文；作者博客仅支持其明确报告的实验与解释。

## 主要发现

1. “Attention + SSM 混合”至少包含不同组织粒度。层间混合将 Attention 层和 SSM 层串接，隐状态依次经过不同模块；层内并行混合则让 Attention heads 与 SSM heads 读取同一输入，再合并输出。二者的数据路径和融合位置不同，不宜合并成一个通用 block 定义。
2. Attention 对显式 key/value 历史提供内容相关查询；SSM 将前缀递归压缩进状态。混合的设计动机是让模型同时拥有两种信息访问路径，不等于已证明每个任务都需要两者，也不意味着固定状态可精确保存所有历史。
3. 混合比例和层位置是实证设计变量。公开消融在模型规模、训练 token、模块实现和任务指标上并不相同；比例不能从一个结果推广成通用最优值。
4. 整模型成绩常含其他共同设计。Jamba 同时采用 MoE；Hymba 同时采用可学习 meta tokens、局部/全局 Attention 和跨层 KV sharing。其整模型结果不能单独归因于混合操作。
5. 吞吐或长上下文效果必须按报告的测量口径限定。部分来源给出预测而非实际测量；长上下文中不同任务也会出现纯 Transformer 与 hybrid 的相对优势变化。

## 来源与证据边界

| 来源 | 版本与阅读范围 | 支持的主张 | 不支持的外推 |
| --- | --- | --- | --- |
| Jamba Team, [Jamba: Hybrid Transformer-Mamba Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf) | ICLR 2025 正式论文；摘要、架构与配置、附录 C.1 及相关消融（PDF §§2–3、App. C） | 层间 Attention/Mamba 交错；8 层 block 中 Attention:Mamba 为 1:7；MoE 可按层加入。1.3B、250B token 消融中，无 MoE hybrid 优于纯 Attention/Mamba，1:3 与 1:7 表现接近，1:7 的计算效率更好。 | 不证明 1:7 普遍最优；发布模型质量/吞吐包含 MoE、训练 recipe、量化及其他配置贡献。 |
| Waleffe et al., [An Empirical Study of Mamba-based Language Models](https://arxiv.org/html/2406.07887v1) | arXiv v1，2024-06-12；§§4.1–4.3、表 6–9 | 8.66B Mamba-2-Hybrid 有 56 层：4 Attention、24 Mamba-2、28 MLP；Attention 与 MLP 分散在网络中。3.5T token 后，其 12 项标准任务平均比对应 Transformer 高 2.65 分。作者预测 batch 32 下长上下文生成最多约 8 倍加速；在其 16K/32K 自然长上下文评测中，Transformer 变体平均约高 1 分，差异尤其出现在多文档问答任务。 | “最高约 8 倍”是文中的预测，不是普遍实测吞吐；整体分数不能外推到其他模型、任务、prompt 或训练配方。 |
| Dong et al., [Hymba: A Hybrid-head Architecture for Small Language Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/f32def07618040e540e0a6182e290562-Paper-Conference.pdf) | ICLR 2025 正式论文；摘要、§2.1、注意力与 cache 消融（表 1、附录 G） | 同一层并行放置 Attention heads 与 SSM heads，使二者处理同一输入；最终模型还使用 meta tokens、以 sliding-window attention 为主、少数全局 Attention 层和跨层 KV sharing。 | 论文报告的总体质量、cache 和吞吐比较是整套设计结果，不能识别 hybrid-head 单独贡献；吞吐依赖其 A100、8K 输入、batch 128 等设置。 |
| Bae et al., [Hybrid Architectures for Language Models: Systematic Analysis and Design Insights](https://arxiv.org/abs/2510.04800v3) | arXiv v3，2026-04-21；HTML §§4.4–4.5 与消融表 5 | 区分 inter-layer 与 intra-layer 融合；主要 inter-layer 比例实验在约 350M/1B、60B token 设置下，1:1 质量较高，约 1:5 是该实验质量/效率的折中；把 Attention block 放在中间的若干配置优于部分首尾/前置方案。 | 这些规模、token 预算和任务下的设计经验不是普遍最优比例或位置。论文对更新模块的外推属于作者判断，仍需相应新实验。 |
| Lee et al., [Understanding and Enhancing Mamba-Transformer Hybrids for Memory Recall and Language Modeling](https://arxiv.org/abs/2510.26912v1) | arXiv v1，2025-10-30；§§3–4、附录 D | 在该预印本采用 Mamba、1:1 比例、sliding-window attention 的设置中，作者比较层间顺序与层内并行的记忆召回和语言建模；报告的相对表现随上下文长度及任务变化。 | 不能据此判定任何尺度和注意力窗口下哪类融合始终更优；预印本结果绑定作者的模型、数据和评测定义。 |
| Gu、Dao 团队，[State Space Duality (Mamba-2) Part IV — The Systems](https://goombalab.github.io/blog/2024/mamba2-part4-systems/) | 作者团队博客，2024；阅读“Results”与“Future Directions” | 作者报告在 2.7B 参数、300B tokens 设置下，6 个 Attention blocks + 58 个 SSD blocks 优于纯 64 个 SSD blocks 和其 Transformer++ 基线，并指出 4–6 个 Attention 层仍需解释。 | 博客报告不是独立复现；不能推广到其他数据、模型或硬件，也不能仅据此判断 Attention 层为何有效。 |

## 编辑使用结论

本章以“历史信息的访问方式互补”解释混合动机，再按层间串接和层内并行区分结构，最后用比例、位置和任务切片说明证据条件。只保留读者判断结构主张所需的架构信息，不写 serving 状态协议、路由回退手册或生产验收流程。第 44 章继续比较 Attention、卷积和 SSM 的模块职责；第 45 章承接模型外部的检索系统。
