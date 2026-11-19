# 第 8 章《Causal Mask、Prefix LM、Bidirectional Attention 与注意力模式》精修审计

审阅日期：2026-11-19。状态：章节职责审视、章级研究、正文重写、配套同步和静态验收完成。未运行代码、测试、模型、API、benchmark 或 PDF；目标平台公式渲染、读者试读与第二十一册连续复读保留在后续门禁。

## 读者契约与章节职责

- **前置：** 已理解 token 表示、Self-Attention 的 Query/Key/Value 与 softmax，并能阅读基础矩阵记号；第 7 章建立位置表示和 position ID 的入口。
- **离章结果：** 能解释可见性矩阵如何限制信息读取；把输入位置的因果可见范围与向右移位的 next-token 标签对齐；区分双向、Prefix、decoder-only 和 encoder-decoder 的信息流；把 attention、padding、loss、样本边界和 position ID 作为不同约束检查；依据独立样本或连续文本流语义判断 packed data 是否需要跨段隔离，并核对边界标签。
- **章节边界：** 本章讲注意力信息流及其和训练监督的关系，不承诺 API 编码或程序复现。注意力复杂度由第 10 章主讲；KV cache 与 decode 偏移交第 11 章；FlashAttention/kernel 交第 20 章；滑动窗口、稀疏模式和成本比较回链第 19、26 章。第 7 章主讲显式位置表示，第 64 章主讲 NoPE 与因果 mask 隐式顺序线索。

## 外部研究与主题去向

检索日期为 2026-11-19，直接核对原论文、固定版本 PyTorch API/教程和课程材料。查询范围、来源标题、版本、阅读位置、支持范围与未验证边界见[因果训练目标、监督遮罩与生成前缀研究记录](../../research/causal-training-and-loss-masks-2026.md)。

| 主题 | 证据与限制 | 正文去向 |
| --- | --- | --- |
| 因果可见性与标签移位 | 原始 Transformer 使用未来位置遮罩和右移 decoder 输入；API 矩阵对角线需连同 query/目标对齐解释 | 8.1—8.2，明确输入位置 $i$ 可见至 $i$ 且预测 $x_{i+1}$ |
| 双向和去噪 | BERT 的 MLM 说明双向上下文与被监督位置相互独立；BART 用双向 encoder 和自回归 decoder 重建损坏文本 | 8.3、8.5，分开讲信息流与目标 |
| Prefix/Seq2Seq mask | UniLM 用多种 mask 表示单向、双向和 Seq2Seq 可见性，但其 cloze 训练说明 mask 不等于固定 shifted loss | 8.4—8.5 |
| 架构与预训练目标 | GPT-1 提供 decoder LM 实例；UL2 混合 denoising 目标，支持架构、可见性和监督目标分别分析 | 8.5 |
| API 布尔 mask 语义 | PyTorch 2.15 `nn.Transformer` 与 SDPA 对 True 的定义相反；只支持对应固定版本 API 的说明 | 8.6、8.9；不概括为通用 mask 约定 |
| Packed sequences | 变长教程给出该 API 的逻辑边界表示；是否隔离由独立样本/连续流目标决定，跨边界 label 检查是样本独立性下的审计推论 | 8.7 |
| 局部窗口 | Mistral 7B v0.1 报告 `window_size=4096`、`context_len=8192` 和层间传播解释；只限该论文版本配置 | 8.8，区分直接读取与跨层传播 |

## 正文和配套修改

- 将旧稿的工作流/资料清单、重复访谈练习和实现细节改成“信息读取边 → 因果/目标对齐 → 结构模式 → 监督与数据约束 → packed 边界 → 局部模式 → 实现合同”的教学路径。
- 用 $j\le i$ 表示输入 query 位置的因果可见性，并写明该位置预测下一个 token；解释严格下三角和含对角线图示取决于行索引对齐，避免只按三角方向判断泄漏。
- 把可见性、label/loss mask、padding、position ID 与 segment boundary 分开。独立样本 packing 才需要跨样本隔离，连续 stream 可保留跨文档条件；EOS 自身不建立 attention 边界，独立样本下还须检查 shift-label 跨界配对。
- 移除自拟 `G_mask`/mask gate demo、任意聚合通过条件及未绑定版本的 kernel fast-path 判断。项目扩展改为声明数据语义、API 版本和 mask 极性，逐案报告违反边和标签/位置问题。
- 同步[练习 31J—31K 与第 32 题](../../EXERCISES.md)、[面试题 31I—31J](../../INTERVIEW_BANK.md)、[项目说明](../../PROJECTS.md)、[术语表](../../GLOSSARY_EN_ZH.md)、[知识图谱](../../KNOWLEDGE_GRAPH.md)和[论文路线](../../PAPERS.md)。章节标题未变，目录无需改动。

## 静态验收与遗留项

- H2 `8.1—8.10` 连续；6 个 `math` 围栏成对；行内数学定界符 76 个、数量为偶数；未发现数学定界符紧贴全角标点。
- 本章 5 个本地章节链接全部存在；题目 ID 31J/31K 未重复；旧 mask gate/demo 和无版本 kernel fast-path 项已从活跃配套入口移除。
- 正文无代码块；未运行代码、测试、API、kernel、模型、benchmark 或 PDF。静态检查不能替代页面 MathJax 渲染、读者试读、全书连续复读或外链终验。
- 下一章按[目录](../../book-21-transformer-architecture-evolution/目录.md)进入第 9 章《Transformer 的并行性、表达力和 scaling 优势》。
