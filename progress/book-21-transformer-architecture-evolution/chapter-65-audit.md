# 第 65 章 Hybrid Attention 精修审计

审计日期：2026-11-24
任务类型：主题思想不变重写。保留“局部、全局与递归路径如何共同处理历史信息”的中心问题，按有效章节职责重建正文，并补齐详细章节大纲。

## 诊断与范围决定

旧稿有 33 个 H2，反复解释局部窗口、全局 attention、递归状态和长上下文能力，另混入训练阶段假设、泛化模型清单及 allocator、scheduler、抢占、checkpoint/rollback、RAG 等通用 serving 运行流程。部分例子缺少对应来源或条件，且与第 43 章的 Attention–SSM 融合比较、第 44 章的模块职责及第 69—71 章的 cache 类型/生命周期重复。

本章现按六个知识任务组织，共 20 个小节：

1. 区分 attention 的可见位置范围与递归状态保存/更新历史的方式。
2. 把层间交替、层内并行与局部/全局/稀疏读取分成不同结构轴。
3. 用明确标注的教学日志事实追踪窗口内直读、全局寻址、状态保留和跨层传递的差异；不将其描述为线上案例。
4. 比较各路径的状态对象及其随上下文长度变化的计算/缓存形态，不从层比推导端到端性能。
5. 以 Griffin、Jamba、Hymba、Qwen3.8-Flash-Next 和 Kimi K3 的版本化报告说明结构组合，并限制实验外推。
6. 将路径假设转为按距离、信息类型、干扰、路径消融、质量和资源口径分开的评测问题。

移出本章的通用服务管理和故障排查内容不影响本章架构读者结果；cache 表示和服务实现由第 69—71 章承担，模块融合史与职责由第 43—44 章承担，Kimi K3 全模型由第 88 章承担。

## 研究依据与证据边界

本轮核对 Griffin arXiv v1、Jamba arXiv v2、Hymba arXiv v1、Hybrid Architectures for Language Models arXiv v3、Qwen3.8-Flash-Next 技术报告 v1 和 Kimi K3 技术报告 v2；课程讲义及 Jamba 团队博客仅作为讲解路径和结构图入口，不替代模型事实来源。各来源的实际阅读位置、支持/不支持的主张和实验边界见[研究记录](../../research/hybrid-sequence-paths-evidence-2026-11-24.md)。

正文将 local/global 限定为 attention 可见范围，将 recurrent state 限定为随 token 更新、没有逐 token 历史地址的表示路径。Jamba 比例、Qwen 训练阶段替换、Griffin 窗口任务、Hymba 的多组件整模结果和 Kimi K3 1M context 均绑定来源版本及条件；没有把任何一个模型成绩归因于单一模块或给出通用最优比例。

## 同步与验收

- 更新正文、详细章节大纲、第二十一册 reader TOC 和设计 TOC；新增练习 31BX、面试题 31BY、`PAPERS.md` 第 62 项、知识图谱路径链，并扩展既有 Hybrid Architecture 术语。
- 正文、大纲和两份目录的 6 个 H2、20 个 H3 标题名称、层级和顺序一致；相关新增本地链接目标存在，练习/面试题和论文编号各自唯一。
- 行内公式表达窗口/位置符号、序列长度复杂度和模型层比例；已按任务符号、渐近范围与前后解释核对源码和数学含义。无块公式、可执行代码或 Mermaid。
- `git diff --check` 通过。GitHub MathJax 页面尚未实际查看，公式渲染验收仍待全书目标平台门禁；目标读者试读、外链终审和第二十一册连续复读也未关闭。
- 未运行测试、构建、模型、API、kernel 或 benchmark，未更新 PDF 或 `plan.md`。

## 修改文件

- 正文：[第 65 章](../../book-21-transformer-architecture-evolution/chapters/65-hybrid-attention局部全局与递归状态.md)
- 结构：[章节大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/65-Hybrid-Attention局部全局与递归状态.md)
- 证据：[研究记录](../../research/hybrid-sequence-paths-evidence-2026-11-24.md)
- 进度入口：[全套书精修交接](../全套书精修-2026-10.md)
