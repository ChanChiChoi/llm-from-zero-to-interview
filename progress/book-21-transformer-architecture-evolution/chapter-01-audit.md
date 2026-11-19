# 第 1 章《Transformer 为什么能成为大模型基础架构》审阅记录

审阅日期：2026-11-19。状态：正文初轮精修与本地静态门禁完成；整册连续复读和目标平台 MathJax 渲染仍待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 用 Transformer、经典 RNN/LSTM、CNN 和选择性 SSM 的信息路由、训练并行、推理状态/缓存与证据范围，解释 Transformer 为何成为大模型基础路线，以及不同架构的真实取舍。
- **读者前置：** 了解基础 Transformer 组成、next-token prediction 和常见矩阵记号；self-attention 的完整计算由后续章节讲解，本章只唤回理解比较所需的直觉。
- **离章结果：** 读者能区分 Transformer 架构、decoder-only 目标、scaling 证据、上下文学习与生态因素；能说明标准 full attention 与逐 token KV cache 的主要代价；能按相同任务、预算、硬件和评估比较架构，并区分论文结果与推断。
- **读者路径：** 序列建模问题 → 经典 RNN/CNN 的结构取舍 → self-attention 路径与内容路由 → 训练并行性和自回归目标 → scaling/ICL/生态证据 → 长序列代价 → 条件对齐的架构比较。
- **前后交接：** 第 2 章完整讲解 Q/K/V、attention score、softmax 和信息交互；本章只提供架构层面的动机和边界。

## 章级扫描与外部主题去向

- **研究日期：** 2026-11-19。技术问题围绕 Transformer 的历史比较证据、序列维训练并行、next-token 目标与架构的关系、Scaling law 和 Chinchilla 的结论范围、GPT-3 的上下文示例表现、LLaMA 的数据披露、SSM/Mamba 的递推扫描、full-attention 复杂度、KV cache 口径及长上下文位置效应。
- **检索范围：** 原始论文/arXiv HTML，包括 *Attention Is All You Need*、GPT-3、Kaplan scaling laws、Chinchilla、LLaMA、Mamba、Mamba-2/SSD、FlashAttention 和 Lost in the Middle；教材/课程组织参考 Stanford CS224N 2026 Lecture 5 与 CS224N 2023 Note 10。既有证据记录包括 `research/attention-mechanism-and-interpretation-2026.md`、`research/scaled-dot-product-attention-2026.md`、`research/scaling-laws-data-mixtures-2026.md`、`research/transformer-core-mechanisms-2026.md` 和 `research/long-context-evaluation-and-effective-capability-2026-11-03.md`。
- **组织经验观察：** Stanford CS224N Lecture 5 先回顾 RNN 梯度问题和机器翻译，再连接 recurrence 与 attention、自注意力、Transformer，最后触及结果及变体；CS224N Note 10 也采用 recurrent baseline/其限制 → 直接序列交互 → 最小 self-attention → Transformer 组件的路线。本章保留“前置路线—结构变化—效果证据—代价边界”的顺序，并增加 decoder-only、scaling、ICL 和生态的书册目标内容；没有照搬课程标题或公式清单。课程资料证明其具体组织选择，不代表某一编排已被因果研究证明对所有读者最优。

| 外部主题 | 证据/读者结果 | 决定与理由 |
| --- | --- | --- |
| 经典递推与局部卷积的路径、状态和并行特征 | Transformer 原论文 §4；Stanford CS224N 2026 Lecture 5 / Note 10 | 保留为必要前置；把结论限定到经典 RNN/LSTM、标准卷积与标准 attention，说明结构化递推可以并行扫描，避免把属性推广到所有 RNN/SSM。 |
| 内容相关读取与训练并行性 | Transformer 原论文摘要及 §4 | 保留为主线；“一跳”解释为计算图路径长度，不声称无损传递、可靠检索或注意力权重等于解释。明确整段已知时的训练并行不等于自回归生成并行。 |
| Decoder-only 与 next-token prediction | GPT-3 原始论文 | 保留为相互适配的组合；说明目标不属于 Transformer 独有，分类/工具调用也不能一概说成天然生成能力。 |
| scaling、few-shot 与公开模型 | Kaplan、Chinchilla、GPT-3、LLaMA 原始论文 | 保留为路线发展证据；逐项限定为 held-out loss、指定计算预算、论文 few-shot 任务表现和公开数据集披露，不写成普遍能力定律或架构因果证明。 |
| Attention 与 SSM 替代路线 | Mamba、Mamba-2/SSD 原始论文 | 保留一个比较入口；解释选择性更新和并行扫描，不用“递推等于串行”，不声称 SSM 已普遍替代 Transformer。 |
| Attention 计算与 KV cache | 原始复杂度分析、FlashAttention、book21 后续章节 | 保留近似推算；明确 $O(T^2d)$ 是单样本单层 dense attention 的位置交互计算，含 batch 时乘 $B$，不包括投影/FFN；缓存公式纳入 batch、层数、KV heads、head dimension 和元素字节数。 |
| 长上下文位置敏感性 | Lost in the Middle 原始论文 | 保留为限定性提醒；限定到其模型和两类任务，禁止把结果外推为所有当前模型的固定曲线或 attention 的因果缺陷。 |

## 正文修改与去向

1. 更新资料列表为可直接追溯的原始论文，并在 scaling、Chinchilla、GPT-3、LLaMA、Mamba、Mamba-2、FlashAttention 和长上下文结论旁补近距离来源链接。
2. 收窄 RNN/LSTM、CNN 与 attention 的比较，区分经典序列递推、结构化状态空间扫描和标准 full attention；说明注意力路径短并不保证信息无损或任务正确。
3. 区分 Transformer 原论文的机器翻译任务、decoder-only 自回归语言模型、next-token 目标和逐 token 推理；明确工具调用/RAG/Agent memory 需要外部运行时。
4. 将 scaling 结论收窄为论文范围内的 held-out cross-entropy 经验规律；将 Chinchilla 表述为给定预算和模型族中的参数/token 配比实验；修正 LLaMA“公开数据”的含义边界。
5. 将 ICL 改为上下文示例带来的可观察行为，不归因为 attention 单一机制；RAG 和工具结果说明为外部系统如何构造上下文。
6. 将 attention 复杂度说明限定到 pairwise score/value mixing；补充 FlashAttention 的 IO/中间态优化不改变 dense attention 的二次位置交互；改正 KV cache 估算公式及符号。
7. 删除自拟综合分数、门槛 $G_{\mathrm{arch}}$、任意权重和 toy Python 评分 demo，因为它们把主观尺度伪装成一般可复现的架构筛选依据；用条件对齐的定性比较表替代。同步重写简介范围、练习 8、面试答案与章节总结。剩余代码例子只示范跨行上下文关系，标记为“静态审阅，未运行”，未改变可执行语句。

## 研究来源记录

- 新增 [Transformer 基础架构路线的原始证据与边界](../../research/transformer-foundation-evidence-2026-11-19.md)，记录原始 Transformer、GPT-3 与 LLaMA 的具体支持范围及边界。
- 在 [长上下文评测与有效能力来源证据](../../research/long-context-evaluation-and-effective-capability-2026-11-03.md) 补入 Lost in the Middle 的任务设置、结果和版本边界。
- 修改历史和读者去向只保存在本文件；研究文件仅记录来源事实，不写审阅日志。

## 本地验收与剩余项

- 95 章书册基线显示章节连续编号；本章结构保留 1.1–1.18，删除旧的 1.15.1 演示小节。
- 本地静态复核：章节围栏配对、Markdown 数学定界和公式变量口径、章节标题顺序、本地链接、`git diff --check`。**没有运行代码、测试、模型、API 或 PDF 构建。**
- 本章未新增本地正文链接；新增的外部引用直达原始论文/arXiv HTML。
- 待后续全书门禁：GitHub MathJax 实际渲染；整册连续复读；历史外链再检。代码片段未运行，正文状态已如实标注。
- 下一入口：第 2 章《Self-Attention 为什么能成为信息路由机制》，先审章级证据与职责，再精修正文。
