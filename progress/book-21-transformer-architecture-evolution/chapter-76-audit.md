# 第二十一册第七十六章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行，保留 KDA 与线性注意力这一主题，但将章节职责从机制推导转到 Kimi Linear 的模型级组合与公开证据。旧稿约 149 行，主要讲 KDA 记忆矩阵、递归公式、逐通道门控、简化实现及训练/推理计算形态，和第六十一章的 KDA 状态机制重复；末尾另以零散小节补充 Kimi K3 配置，没有系统说明本章标题所指的 Kimi Linear 如何组合 KDA、全局 MLA、NoPE 与 MoE，也未区分匹配设置的 1.4T 研究对照和 5.7T 发布模型证据。

重写保留既定章节标题和编号，以“结构轴—四层信息路径—层比消融—1.4T 质量对照—5.7T 发布 checkpoint—cache/速度口径—评测归因”组织正文。KDA 状态递推只作必要回顾并回链第六十一章；具体层比、性能数字和评测结论均限定在 Kimi Linear v2 的相应配置、表格或图中。

## 结构与分工

正文与[现行大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/76-KDA与Kimi-Linear混合架构.md)均为 7 个 H2、21 个 H3。第二十一册 reader TOC 与设计 TOC 已同步相同标题、层级和顺序，并链接本章大纲。第六十一章负责 KDA 机制，第六十三章负责 Gated MLA latent cache，第六十四章负责 NoPE 一般性质，第六十五章负责混合路径比较，第六十九章负责 cache 表示，第八十八章负责 Kimi K3 全模型案例。

## 研究依据与证据边界

模型配置、3:1 消融、1.4T matched runs、5.7T 发布 checkpoint、长上下文结果及 cache/prefill/TPOT 读数依据 Kimi Team [Kimi Linear 原论文 v2](https://arxiv.org/html/2510.26692v2)。章节将三种证据分开：1.4T 同规模与训练设置对照支持报告内的组合方案比较，但 KDA、hybrid ratio 和 NoPE 等共同变化，不能作单因素归因；5.7T Kimi Linear 与 Moonlight 的 total parameters 不同，不能当作同总参数消融；Figure 7 的 batch-one prefill 曲线与 Figure 1(b) 依赖较大 batch 的 TPOT 主张不是同一测量口径。论文未给出完整生产硬件、runtime、kernel 与调度条件，正文不外推为线上加速保证。

来源版本、阅读位置、数值和可支持范围见[Kimi Linear 模型级研究记录](../../research/kimi-linear-model-evidence-2026-11-25.md)。研究记录保存新发现及证据；本审计保存正文诊断、编辑决定、同步状态与未完成门禁。

## 正文、配套与验收

- 已重写正文并更新本章完整大纲、第二十一册 reader/design TOC；第六十一、六十五、七十五章及第二册、第四册相关入口已改为区分 KDA 机制和 Kimi Linear 模型案例。
- 已同步 `EXERCISES.md` 前沿架构题、`INTERVIEW_BANK.md` 第 31CJ、`PAPERS.md` 的 Kimi Linear 论文条目和 `KNOWLEDGE_GRAPH.md`。
- 静态检查确认正文、大纲与两份目录的 28 个编号标题名称、层级和顺序一致；代码围栏成对，正文 Markdown 本地链接目标存在，行内数学定界符平衡，比例、模型规模、表格结果和性能口径已对照研究记录；31CJ 编号唯一。`git diff --check` 通过。
- 公式源码与数学含义已静态核对；GitHub MathJax 页面未实际查看，公式视觉验收未完成。目标读者试读、外链终审和第二十一册连续复读仍待全册门禁。
- 未运行测试、构建、模型、API、kernel、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

下一章为第七十七章《CSA/HCA：从压缩 KV 到百万上下文》。第七十七章与第七十四章重复讲解 CSA/HCA 机制的旧稿问题留待按序处理。全套书精修 goal 继续 active。
