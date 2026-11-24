# 第五十二章精修审计

## 章节职责与旧稿诊断

第五十二章承接第九章的 scaling 证据，回答“预训练 FLOPs 预算已定时，怎样在模型参数量与训练 token 之间分配”。本章只主讲 compute-optimal 的目标和成本口径、Chinchilla 的预算前沿与大模型案例，以及 PaLM 所展示的多设备系统扩展。一般 scaling law、数据质量与 mixture 由第九章主讲，attention 成本由第十章主讲，MoE 架构及其训练/服务成本由第二十一、二十二章承接，助手后训练由第五十三章承接。

旧稿有 29 个二级标题，将 scaling、训练数据质量、预算估算、训练生命周期、MoE、服务成本及多个重复的公式/示例并列反复展开。它和第九章的 scaling 与数据讨论重复，也越过第五册训练预算及第六/二十四册系统服务的职责。按 02a 对照本册总纲、目录、旧正文及上下游章节后，将整章组织收束为五节：定义目标与预算、解释 dense $6ND$ 近似、分析 Chinchilla 前沿、用 PaLM 单独呈现系统扩展、再说明数据/协议/目标变化时的迁移边界。旧稿中的必要内容均保留在相应位置，通用 scaling 和服务流程不再重复主讲。

## 研究依据与编辑决定

- Hoffmann 等人的原始论文支撑固定训练 FLOPs 下最小化预训练 loss、400 多个较小训练拟合前沿、Chinchilla/Gopher 相近报告预算案例及该大规模验证点有限等说明。正文将 $D/N\approx20$ 限定为实验估计的记忆点，没有把它写成跨模型与数据集定律。
- 同一论文附录支撑 dense $6ND$ 作为近似账本及详细 FLOPs 对照。正文区分 token exposure、唯一数据和上下文长度，并指出长序列 attention、重计算、通信、设备利用率及 MoE 路由会超出简化口径。
- PaLM 原始论文支撑模型/数据规模、Pathways 与 TPU v4 训练系统，以及 model-FLOPs utilization 和 hardware-FLOPs utilization 的两种口径；后者计入 rematerialization FLOPs。正文明确 PaLM 并非 Chinchilla 的同预算单变量对照。
- NeurIPS 2024 复现研究和 JMLR 2025 数据受限研究分别支撑 scaling 拟合对 FLOPs 计数/训练协议的敏感性，以及唯一数据受限时重复 exposure 的边际价值变化。来源位置、日期和支持边界见[研究记录](../../research/chinchilla-palm-compute-optimal-evidence-2026-11-23.md)。

## 正文和同步

- 按五节完整重写正文，保留“参数—数据预算分配”作为主线；用目标式表达优化对象，用 $6ND$ 连接参数量、token exposure 和训练 FLOPs；用 Chinchilla/Gopher 对照说明预算转移；用 PaLM 说明规模化训练系统；结尾区分预训练 loss、下游表现、后训练目标和推理成本。
- 新建完整五节[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/52-Chinchilla-PaLM-compute-optimal.md)，同步[第二十一册目录](../../book-21-transformer-architecture-evolution/目录.md)的标题、顺序和大纲链接。
- 更新 `EXERCISES.md` 练习 31BJ、`INTERVIEW_BANK.md` 面试题 31BK、`PAPERS.md` 第 49 项和 `KNOWLEDGE_GRAPH.md` 的预算分配/系统扩展两条关系。现有术语条目足够，本轮没有新增词条。

## 验收与遗留事项

- 静态结构复核：正文、大纲、目录的五个小节标题与顺序完全一致；章节、大纲、目录、研究记录、练习和论文索引的本地链接扫描共检查 162 个链接实例，没有发现缺失的本地目标；练习 31BJ、面试题 31BK、论文索引第 49 项各唯一。
- 公式源码与数学含义已复核：两处块级 `math` 围栏闭合并顶格，行内变量和单位采用数学定界符，预算约束、$6ND$、参数/token 比例与正文口径一致。**GitHub MathJax 实际页面渲染尚未验收**，故本章公式门禁仍未全部关闭；不得以源码检查或 `git diff --check` 代替。
- 正文无可执行代码或 Mermaid 图。`git diff --check` 通过。本轮未运行测试、代码、API、模型、benchmark 或 PDF 构建，也未更新 PDF 或 `plan.md`。
- 下一章为第五十三章《InstructGPT、ChatGPT 与 RLHF 对架构使用方式的改变》。目标读者试读、第二十一册连续复读和历史外链终审留待整册门禁；全套书精修 goal 继续 active。
