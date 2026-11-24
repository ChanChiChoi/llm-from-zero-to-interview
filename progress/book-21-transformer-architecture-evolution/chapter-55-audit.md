# 第五十五章精修审计

## 章节职责与旧稿诊断

本章用两个固定论文模型说明不同架构改动的作用位置：Mistral 7B v0.1 组合 GQA 与滑窗注意力；Mixtral 8x7B 保留 Mistral 主干并把每层 FFN 替换为稀疏专家。第十七章负责模型配置字段，第十九章负责局部可见性与跨层传播，第十一章负责 KV cache，第 21/22 章负责 MoE 路由与资源成本，第七十九章负责 Mistral Small 4 的后续模型案例。

旧稿有 28 个 H2，重复 GQA、Mistral 滑窗、MoE 路由/负载、通用服务与回滚流程、benchmark 选择和跨版本兼容检查；其中部分内容由第 4、11、17、19、21、22 章及第六/二十四册承担。按 02a 将本章职责收束为五节：固定版本范围、GQA 与 SWA 的不同资源路径、Mixtral 的 FFN 专家、论文证据比较、将效率归回具体成本路径。未改变书册总纲和既有章节顺序。

## 研究依据与编辑决定

- [Mistral 7B 原论文](../../research/mistral-mixtral-architecture-evidence-2026-11-24.md)支持 GQA/SWA、Table 1 中的 K/V head/window/context 字段及作者报告的 benchmark 结论。正文不把窗口大小、context 字段、理论传播跨度和任务有效上下文混为一谈。
- Mixtral 论文和 Mistral 官方发布页分别支持每层 8 个 FFN experts、每 token top-2、$32\mathrm{k}$ 训练上下文，以及约 $47\mathrm{B}/13\mathrm{B}$ 和更精确的 $46.7\mathrm{B}/12.9\mathrm{B}$ 参数口径。正文明确总权重和激活计算的区别，并保留路由、通信、设备负载成本。
- 修正 Mixtral 结果表述：摘要为总体概括，正文逐项表格存在差异；不写成每个指标、每个比较对象都胜出。比较与单模块因果归因分开。
- Mixtral 权重发布于 2023-12-11，论文发表于 2024-01；时间线统一反映这两个日期。模型使用 Apache 2.0 权重许可，不等同于训练语料和 recipe 全部公开。
- 研究问题、检索词、来源阅读位置与支持边界见[研究记录](../../research/mistral-mixtral-architecture-evidence-2026-11-24.md)。

## 正文和同步

- 正文按有效五节大纲重新组织；缩减重复机制和 servicing 操作说明，保留本章读者需要的路径差异、benchmark 条件和版本/许可信息。
- 更新[第五十五章结构大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/55-Mistral-Mixtral高效与MoE.md)、[第二十一册目录](../../book-21-transformer-architecture-evolution/目录.md)和[架构时间线](../../book-21-transformer-architecture-evolution/主流大模型架构创新时间线.md)。时间线改成速查入口，删去重复的 GQA/MoE 教学清单。
- 新增 EXERCISES.md 练习 31BM、INTERVIEW_BANK.md 面试题 31BN、PAPERS.md 第 52 项和 KNOWLEDGE_GRAPH 路径。沿用已有 GQA、SWA、MoE、KV cache 术语，不重复新增术语表条目。

## 验收与遗留事项

- 正文、大纲、目录的五个 H2 标题及顺序一致；chapter 55 新增的目录、题目、论文记录和正文链接目标均须通过本地静态检查。
- 正文只有行内数学（$4{,}096$、$47\mathrm{B}$、$13\mathrm{B}$、$46.7\mathrm{B}$、$12.9\mathrm{B}$、$32\mathrm{k}$），无块公式、代码或 Mermaid；变量/单位定界符和数学含义已静态核对。**GitHub MathJax 页面未实际查看，公式渲染门禁待验收。**
- `git diff --check` 通过后记录于本审计；未运行测试、代码、模型、API、benchmark 或 PDF 构建，未更新 PDF 或 `plan.md`。
- 下一章为第五十六章《Gemini、Claude、GPT-4V：多模态、长上下文与系统能力》。目标读者试读、外链逐条终审、第二十一册连续复读和行内公式页面渲染仍留待后续门禁；全套书精修 goal 继续 active。
