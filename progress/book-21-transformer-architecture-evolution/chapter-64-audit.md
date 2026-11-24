# 第 64 章 NoPE 精修审计

审计日期：2026-11-24  
任务类型：主题思想不变重写。保留“没有显式位置变换不等于没有顺序”的中心主题，按有效章节职责重组正文。

## 诊断与范围决定

旧稿有 30 个 H2，反复讲置换等变、因果 mask、顺序交换、padding/packed sequence、runtime position metadata、state reset、长上下文评测和服务检查表。相同观点分散出现多次，且将架构内的位置线索、position ID 协议和缓存/推测解码的运行时生命周期混为一谈；其中部分论述把 causal mask 概括为只提供方向、把隐式状态误差描述成 NoPE 性质，超出本章范围或证据。

修订后章节聚焦五个知识任务：

1. 界定 NoPE 在特定 attention 路径中的含义，区分显式位置注入与整网已有的隐式线索。
2. 说明无位置特征、无位置相关 mask 时的置换等变性，以及固定 causal mask 怎样建立前缀可见不对称。
3. 区分 causal mask 形成的隐式位置表示与上游局部混合/递归状态传递的近期性线索。
4. 分开评估顺序判断、相对距离/近期性和绝对位置索引；将 NoPE 长度泛化限制在具体论文设置。
5. 以 Kimi K3 v2 说明 MLA Q/K 的 NoPE 与 KDA 的位置敏感混合如何共存；不把模型窗口归因于 NoPE 单项。

通用 serving 状态协议、padding 与 packed-sequence 回归、推测解码位置提交、cache 版本管理和可执行 toy 训练方案不再作为本章正文主线。第 63 章负责 MLA latent 与 output gate，第 65 章负责混合层总体安排，第 67 章负责 RoPE 和位置外推，第 88 章负责 Kimi K3 全模型。

## 研究依据与证据边界

事实核验使用 Haviv 等人的 EMNLP 2022 论文、Wang 等人的 ACL 2024 论文、Dawes 等人的 2026 arXiv v1 预印本以及 Kimi K3 v2 技术报告。具体来源版本、阅读位置、教学材料和限制见[NoPE 研究记录](../../research/nope-order-information-evidence-2026-11-24.md)。

正文把 causal mask 的前缀结构和隐式位置 probing 分开叙述；把混合模型的 residual-stream recency 机制标成 2026 预印本中特定架构的解释；将 Kimi K3 的位置职责绑定技术报告 v2。报告中的 1M 窗口同时涉及长上下文数据和渐进训练，不据此单独归因 NoPE。

## 同步与验收

- 正文、大纲、reader TOC 和设计 TOC 的 5 个 H2、10 个 H3 标题名称、层级与顺序一致。
- 同步练习 31BW、面试题 31BX、PAPERS.md 第 61 项、KNOWLEDGE_GRAPH.md 和 NoPE 术语定义。
- 两处公式源码和数学意义已静态核对：置换矩阵同时作用于输入与 mask；causal mask 仅保留 $j\le t$ 的前缀。MathJax 实际页面渲染仍待验收。
- 本章无可执行代码或 Mermaid。未运行测试、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 plan.md。
- 目标读者试读、外链终审及第二十一册连续复读仍属全册门禁。

## 修改文件

正文：`book-21-transformer-architecture-evolution/chapters/64-nope与隐式顺序.md`
结构大纲：`写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/64-NoPE显式位置路径与隐式顺序.md`
研究记录：`research/nope-order-information-evidence-2026-11-24.md`
进度入口：`progress/全套书精修-2026-10.md`

