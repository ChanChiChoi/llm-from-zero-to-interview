# 第四十章《Mega、Gated State Space、H3 等早期混合路线》精修审计

## 范围与诊断

按全套书精修的主题重写要求完成本章，保留章名、正文文件路径和目录位置。完整复读旧稿后发现 32 个 H2 中有大量重复的 gate saturation/可辨识性诊断、状态容量预算、state schema、snapshot/rollback 与 serving 验收；这类通用工程清单超出本章关于架构信息流的职责。旧稿的 Mega 公式把两条路径直接并列却漏掉求和；H3 被抽象成通用 SSM 加 Conv 的融合块，未呈现原论文的 shift/diagonal SSM 与乘法交互；GSS 还被泛化成输入生成 $A/B/C$ 的状态更新。toy state-conv 代码也不是三篇论文任一方法的实现。

旧稿没有章内有效结构大纲。本轮先按 02a 设计六节大纲并同步第二十一册目录，再依大纲重写正文：语言建模缺口，Mega 的 EMA-gated attention，GSS 的窄 DSS 双流门控，H3 的 key/value 状态交互，混合位置与 Hyena 算子谱系，逐条件实验结果。把 servicing、回滚、量化监控和一般混合模型预算交还给相应工程内容，不再占据本章正文。

## 研究与内容

研究范围是 Mega、GSS、H3 原始论文对算子的定义、组合位置及实验依据；详细来源版本、阅读位置和结论边界见[研究记录](../../research/early-hybrid-sequence-models-2026-11-23.md)。以 Mega v3、GSS v3、H3 camera-ready v3 为结构与结果依据；H3 作者团队博客用于理解其合成任务和教学路径，Hyena 原论文仅用于说明 operator-level 谱系关系。

正文按论文定义修正 Mega：多维 damped EMA 输出影响 query/key、reset/update gate 和候选表征，attention value 来自原输入，update gate 在候选结果与原输入之间插值；Mega-chunk 的注意力分块为 $O(Lc)$，EMA 提供压缩后的跨块上下文。GSS 依次投影两路、仅由窄流经过 DSS、再与另一流逐点相乘，不将其误写为 token-dependent $A/B/C$。H3 依次用 shift SSM 保存前序 key 对齐，用 key/value 乘法建立关联，由 diagonal SSM 延续并从 query 读取；多头外积写入与单头简化公式相互对应。

结果表分别保留 Mega 的 LRA 任务范围与参数/seed 对照、GSS 的 PG19/TPUv4 条件和初始化调参差异、H3 两层合成任务中的 S4D/GSS/attention 对照、Hybrid H3 Pile 训练条件及 1.3B A100 吞吐条件。没有将异构任务、训练预算或 H3 的系统实现合并成通用架构胜负结论。

## 配套同步

新增第四十章结构大纲，目录增加 40.1—40.6 目录项。PAPERS.md 添加 Mega/GSS/H3 原论文及研究记录；KNOWLEDGE_GRAPH.md 增加 EMA、DSS 和两级 SSM 的路径关系；GLOSSARY_EN_ZH.md 新增 Mega、GSS、H3 条目。新增练习 31AS 和面试题 31AP。

## 验收与遗留

- 正文、大纲、目录的六个标题及顺序一致；章节名称、路径和目录位置保留。
- 公式核对覆盖多维 damped EMA、Mega attention 与 reset/update gate/候选项、GSS 双流计算、H3 单头与多头外积状态、Mega-chunk 复杂度及各实验数字的条件。正文包含对比表与结果表，无代码、插图或 Mermaid。
- 本地大纲、正文、研究、审计和题库链接目标已检查；新增题号在对应题库唯一。
- `git diff --check` 通过。GitHub MathJax 页面逐式渲染尚未验收，公式门禁仍开；外链最终逐条复核、目标读者试读和第二十一册连续复读留待全册门禁。
- 未运行测试、代码、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 `plan.md`。

第四十章完成首轮修订。下一章为第四十一章《卷积序列模型在 LLM 中的复兴可能性》。
