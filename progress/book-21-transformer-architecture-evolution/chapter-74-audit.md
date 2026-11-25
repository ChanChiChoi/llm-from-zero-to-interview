# 第二十一册第七十四章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿把长上下文注意力扩写成 RAG 引用忠实度、访问控制、发布协议、回滚和 P99 等通用生产清单，却没有讲清 CSA/HCA 怎样压缩历史、如何为 query 选择或读取压缩条目；原有大纲也只有宽泛目录标题，缺少章内结构与中心思想。

新稿围绕“历史如何成为较少的新 KV 表示，以及当前 query 怎样读取这些表示”重新组织。它明确区分压缩器、CSA indexer 和 core attention：CSA 以 a/b 两条流形成跨块重叠窗口，Lightning Indexer 依独立的压缩 key 候选选 top-k；HCA 以较大的非重叠块压缩，并在所有可用压缩条目上 dense 计算。两条路径均与近期原始 token 窗口共同进入核心注意力。局部窗口、shared-KV MQA、共同归一化和成本/误差分析围绕这一机制展开。

## 结构与分工

正文和详细大纲均为 6 个二级标题、21 个三级标题，覆盖长历史的存储/读取问题、learned pooling、CSA 重叠与索引、HCA 粗压缩与 dense compressed-domain read、近期窗口与 shared KV、复杂度及证据边界。第二十一册 reader TOC 与设计 TOC 已同步标题和层级。

第 73 章负责 SDPA 输出门；第 74 章负责 CSA/HCA 的通用机制；第 77 章保留 DeepSeek-V4 的层配置、百万 token 证据、参考实现和 serving 路径。第 77 章开头已链接回第 74 章；第 77 章其余重复或过度扩写的内容仍留待后续精修。

完整设计见[第七十四章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/74-CSA-HCA压缩历史与查询读取.md)。

## 研究依据与证据边界

核心定义依据 DeepSeek-AI [DeepSeek-V4 技术报告 v1](https://arxiv.org/html/2606.19348v1)，重点核对 §2.3.1—§2.3.4、公式 9—26 及 §3.5.1。Stanford MS&E 319 [Lecture 5 notes](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf) 用于检查逐特征加权 pooling、因果完成与压缩/选择拆分的教学解释；独立研究笔记用于交叉理解，不作为方法定义的一手来源。检索词、版本、阅读位置、来源角色和未验证边界见[CSA/HCA 机制研究记录](../../research/csa-hca-mechanism-evidence-2026-11-24.md)。

正文区分压缩条目与原 token KV，说明 CSA top-k 选择的是压缩条目而非原始 token 或“正确证据”；HCA 的 dense 操作范围是已完成的压缩条目。CSA 同时有表示压缩损失与索引漏选风险，HCA 没有 top-k 漏选但使用更粗的块表示。局部窗口补充近期细节，不恢复任意远端 token 的原始 KV。条目数量仅用于解释候选规模，不等于实际 FLOPs、显存字节数或延迟；V4 具体配置和性能数字留给第 77 章。

## 正文、配套与验收

- 已新增本章详细大纲，改写正文，更新 reader/design TOC；同步 `EXERCISES.md` 第 19 题、`INTERVIEW_BANK.md` 第 31CH 题与第 15 题回答、术语表三个 CSA/HCA 条目、`PAPERS.md` 第 71 项和 `KNOWLEDGE_GRAPH.md`。第 77 章开头增加了机制章节链接与职责说明。
- 正文、大纲、两份目录的 27 个编号标题名称、层级和顺序一致。静态检查确认正文、相关目录和大纲的新增本地链接目标存在，正文围栏成对，无 `\operatorname`；面试题与论文索引编号唯一。`git diff --check` 通过。
- 公式定义、符号与正文机制已静态核对。GitHub MathJax 页面未实际查看，故公式视觉渲染门禁仍待验；目标读者试读、外链终审及第二十一册连续复读也待全册门禁。
- 未运行测试、代码、模型、API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

下一章为第七十五章《Attention Residuals》。全套书精修 goal 继续 active。
