# 第四十五章精修审计：Retrieval-Augmented Architecture

## 诊断与范围

完整复读旧稿 28 个二级标题及正文。旧稿反复讲检索—生成流程、长上下文与检索对比、信息瓶颈、质量门、失败回退和评估门槛；后半混入权限/版本流水线、上下文预算与 query rewrite、缓存失效、索引切换、回滚、p99、请求重放和生产验收等实现/运维清单。还把 query-time 文档检索、模型内部记忆和 Agent 搜索的概念交叉使用，并含有未承担本章读者结果的玩具 Python 代码和多组重复公式。

第四十五章的读者结果应是能解释外部语料如何进入生成模型，比较检索接入位置，沿证据路径定位失效，并分别评估检索上下文、生成忠实度和回答/引用质量。第四十四章承担模型内部的 Attention、卷积和 SSM 信息路径；第十三章承担长上下文中的位置与检索行为；第四十六章承担跨请求记忆的读写与更新；第四十九章承担 Agent 的工具调用、计划和多轮搜索。本章不再重复生产部署手册。

## 结构与正文决定

保留章名、编号与正文路径，经 02a 重组为八节：

1. 区分参数知识、请求上下文与外部语料，明确 RAG 是新增证据路径而非正确性保证。
2. 用离线索引和在线检索两阶段追踪语料、候选、证据组装与生成器的职责。
3. 说明提示边界注入如何把检索片段交给未改动的生成模型，以及该接口不强制证据忠实性。
4. 用 RAG-Sequence/Token、REALM 和 RETRO 比较潜在文档建模、检索增强预训练与 decoder 内 cross-attention。
5. 以 Self-RAG 说明固定检索与模型内按需检索的控制差异，并交接 Agentic Search。
6. 将失效定位到语料/索引、检索候选、上下文组装、生成/引用四段。
7. 分开 context relevance、answer faithfulness、answer relevance、correctness 与 citation support，并用 gold evidence 作受限诊断对照。
8. 与长上下文、长期记忆和 Agent 搜索划界，不建立脱离任务的通用排名。

删除重复的通用 RAG 清单、未被证据支持的固定阈值/比例公式、toy Python 检索排序代码，以及权限 schema、缓存 key、索引回滚、p99、trace replay 和生产门禁流程。保留证据版本、引用支持与索引更新作为影响读者判断的架构边界，不展开为部署方案。

## 研究与证据

新知识及直接来源位置见[检索增强架构研究记录](../../research/retrieval-augmented-architecture-evidence-2026-11-23.md)。依据 Lewis 等人的 RAG 原文区分 RAG-Sequence/Token 的潜在文档粒度；依据 REALM 与 RETRO 原文说明预训练耦合和 chunked cross-attention；依据 Self-RAG 原文说明 retrieval/reflection token；依据 ALCE 与 ARES 说明答案正确性、上下文忠实度、问题相关性和引用支持属于不同评估维度。引用和结果均限于各论文模型、任务、数据和评审协议。

同时查阅马里兰大学 Fall 2026 的 RAG/Agentic Search 课程讲义与 Eugene Yan 等人的 RAG 工程文章，用于核对“语料索引—问题检索—上下文组织—生成”的解释路线及常见上下文断点；它们作为教学/实践入口，不充当普遍效果证据。未运行本地检索实验、代码、模型、API 或 benchmark。

## 同步与验收

- 新建[第四十五章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/45-Retrieval-Augmented-Architecture.md)，八个标题已加入第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)。
- 更新 KNOWLEDGE_GRAPH.md 的语言模型链，新增从参数/上下文/外部语料到检索集成方式、评估轴及跨章边界的路径。
- 新增练习 31BB 和面试题 31BC。PAPERS.md 已有本章涉及的 RAG、REALM、RETRO、Self-RAG、ALCE 与 ARES 条目，因此无需重复添加；现有术语足以表达本章内容，无需增加术语表条目。
- 正文无公式、可执行代码或 Mermaid 图；公式和代码渲染/运行门禁不适用。目标读者试读、外链最终逐项终审及第二十一册连续复读留待全册门禁。
- 静态验收：正文、大纲与目录八个小节标题及顺序一致；六份相关正文/导航/配套文件的本地链接检查无缺失；练习 31BB 与面试题 31BC 各出现一次；`git diff --check` 通过。正文无公式、代码围栏或 Mermaid 图，因此公式渲染、代码运行及图表页面渲染门禁不适用。
- 未运行测试、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 `plan.md`。

本章首轮精修完成，不代表第二十一册或全套书精修完成。下一入口为第四十六章《Memory-Augmented Transformer 与长期记忆架构》。全套书精修 goal 继续 active。
