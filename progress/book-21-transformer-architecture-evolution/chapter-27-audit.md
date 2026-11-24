# 第二十七章精修审计

审阅日期：2026-11-20。当前状态：职责复核、旧稿诊断、章级研究、结构重设计、正文主题重写、配套同步和本地静态检查已完成；GitHub MathJax 实际渲染验收未完成。全套书精修 goal 继续 active；不更新 PDF 或 `plan.md`。

## 旧稿诊断与重写方向

旧稿 245 行、29 个 H2，前四节简述 Longformer、BigBird 与 Reformer，后续则反复展开同一组连接路径、漏证据风险、训练前提、seed/哈希、复杂度、失败归因、benchmark、公平迁移和生产回退检查。第十九章已经主讲窗口、dilation、global/random pattern 与多层信息路径；第二十六章刚完成 token mask、块布局、动态路由和执行边界；第二十八章负责跨路线的总体取舍。旧稿没有充分解释 Reformer 的 reversible residual 和 FFN chunking，却把一个 sign-projection Python demo 当作 Reformer 的最小教学代码；该代码未覆盖原论文的共享 Q/K、哈希排序、chunk/相邻 chunk、causal mask 与多轮 hash。

结构调整后聚焦：三种路线如何产生候选连接；Longformer 的任务指定全局位置及局部/扩张路径；BigBird 的 local/random/global 图与理论边界；Reformer 的 LSH 排序分块和漏召回；Reformer 可逆残差与 FFN 分块如何另行压低训练激活内存；最后对齐比较它们改变的对象与证据边界。删除重复的线上验收流程、benchmark checklist、完整实验 recipe 和不准确的演示代码。

## 研究范围与证据

来源、版本、阅读位置、机制和证据边界见[Longformer、BigBird 与 Reformer 研究记录](../../research/longformer-bigbird-reformer-methods-2026-11-20.md)。本章以三篇原始论文为方法与定理依据，参考 Google Research 作者解释及 MIT 课程图示、Longformer 教学 notes 检查概念组织。论文实验不外推为当前通用模型、GPU 或 serving 结论；BigBird 的理论能力结论保留假设与下界，Reformer 的 $O(L\log L)$ 不替代实际排序和 kernel 成本。

## 正文重写记录

保留章名、路径和目录位置；在 02a 完成结构调整后，将旧稿 29 节的重复主题收束为六节。正文以“谁决定 query 的候选 key”贯穿三种方法：Longformer 通过局部/扩张窗口和任务 global 连接，BigBird 组合 local/random/global 图，Reformer 通过共享 Q/K 表示的 LSH、排序与分块近似选择候选。再把 Reformer 的 reversible residual 与 FFN chunking 单列，说明它们优化训练激活存储而非 attention 连接数。

删去旧稿中不属于机制说明的通用 benchmark、部署/回退和线上 gate 清单；移除未实现论文 LSH 流程的随机投影教学代码。保留四处数学表达：Longformer 和 BigBird 的可见集合、Reformer 的每轮哈希碰撞关系、可逆残差块的更新与重构。BigBird 理论结果同时呈现表达性结论与论文给出的稀疏图下界，避免将形式化能力误写为任意任务保证。

## 同步、校验与遗留门禁

- 新增章节大纲并链接至第二十一册目录；同步 `EXERCISES.md` 练习 31AF、`INTERVIEW_BANK.md` 面试题 31AC、`GLOSSARY_EN_ZH.md` 的 Reformer/LSH/reversible residual 术语、`KNOWLEDGE_GRAPH.md` 长序列方法链及 `PAPERS.md` Reformer 条目。证据见[研究记录](../../research/longformer-bigbird-reformer-methods-2026-11-20.md)。
- 静态检查：H2 27.1—27.6 连续；四处顶格 `math` 块配对，复核候选集合、复杂度边界、哈希集合和可逆更新/重构含义；无代码或 Mermaid；章节内第十九、二十六、二十八章以及大纲、审计、研究记录链接目标存在；31AF、31AC 和新增术语唯一；`git diff --check` 通过。
- GitHub MathJax 页面尚未实际打开，故公式源码与含义已检查，但**公式渲染验收未完成**；四处公式列入第二十一册待验清单。PDF 不属于本轮任务。目标读者试读、外链逐条终审和第二十一册连续复读留待全册门禁。
- 未运行测试、代码、模型、API、kernel、benchmark 或 PDF 构建。全套书精修继续进行，下一章为第二十一册第二十八章《高效 Attention 为什么没有完全替代标准 Attention》。
