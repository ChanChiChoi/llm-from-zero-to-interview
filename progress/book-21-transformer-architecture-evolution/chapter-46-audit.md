# 第四十六章精修审计：Memory-Augmented Transformer 与长期记忆架构

## 诊断与范围

完整复读旧稿 29 个二级标题及正文。旧稿将用户/Agent 跨会话记忆、Transformer 的长序列记忆、RAG、普通推理 KV cache、上下文折叠和任务状态混写；后半反复重复写入可信度、来源/版本、权限、删除传播、跨租户隔离、缓存一致性与生产验收。一般 key-value 条目公式及 toy Python 例子并不解释 Memory-Augmented Transformer 的具体计算路径，多个 checklist 和状态机内容也与第四十九章 Agent 记忆/运行状态重叠。

本章读者结果应是能说明过去信息怎样变成后续计算可访问的模型状态，并比较状态表示、寻址/读取、融合位置、更新方式和训练耦合。第四十四章负责 Attention/SSM/卷积信息路径；第四十五章负责外部语料检索；第十一章负责当前生成过程中的 KV cache；第十四章负责递归状态记忆概念；第四十九章负责 Agent 记忆、上下文折叠与任务运行状态。本章不再展开个人记忆数据库、用户权限/删除运维或工具轨迹管理。

## 结构与正文决定

保留章名、章节编号与正文路径，经 02a 重新组织为七节：

1. 定义模型架构语境下的记忆，并区分当前上下文、请求级解码 KV cache、外部语料与 Agent 记忆；说明“长期”由每种方法的状态生命周期定义。
2. 用编码/写入、寻址/读取、融合和更新四个问题形成共同比较坐标，不引入虚构的通用 memory schema、写入阈值或融合公式。
3. 解释 Memorizing Transformer 的局部注意力与近似 kNN K/V 读取、每 head 门控、片段缓存更新、文档级清理和 memory staleness；把实验限制在论文语言建模任务。
4. 解释 LongMem 的冻结 backbone 历史编码和 trainable SideNet 读取/融合分工，以及长文本和多示例 ICL 的证据边界。
5. 解释 Larimar 的潜在情景关联矩阵、训练阶段联合学习和推理阶段一次式事实更新；区分模型事实遗忘与通用隐私删除。
6. 解释 MemT-1.0 的并行原始注意力/原生记忆块、可训练 hyper memory 与动态 local state、GDN-2 token 级更新；将 128K 的速度/存储结果限于预印本所报 BEAM、Nano、双 GPU、batch 1 和 128-token 输出设置。
7. 用状态表示、写入/更新、读取/融合、状态增长和任务证据比较方法，并与标准 KV cache、RAG 和 Agent 运行状态收束边界。

## 研究与证据

新知识和原文阅读位置见[Memory-Augmented Transformer 架构研究记录](../../research/memory-augmented-transformer-architectures-2026-11-23.md)。依据 Memorizing Transformer 原论文核对非可微 K/V 记忆、近似 kNN、局部/记忆注意力门控、文档顺序处理和 staleness；依据 NeurIPS 2023 LongMem 论文核对冻结编码器与 SideNet 的解耦；依据 ICML 2024 Larimar 正式论文核对潜在情景记忆和一次写入；依据 2026-10 MemT-1.0 预印本核对动态状态、hyper memory、GDN-2 更新与测量口径。

同时查看 Stanford CS277 Lecture 15、Hugo Cisneros 的 Memorizing Transformer notes，以及 Michael Brenndoerfer 的可视化解释文章，用于发现机制讲解路径和常见误区。教学材料不替代论文事实核验；可视化文章中对 attention 容量的宽泛解释和简化代码未被当作通用结论。未运行本地模型、代码、API 或 benchmark。

## 同步与验收

- 新建[第四十六章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/46-Memory-Augmented-Transformer.md)，并把七节标题同步到第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)。
- 更新 KNOWLEDGE_GRAPH.md 的模型记忆架构链，补充四类表示/读取路径和第十一、十五、四十九章分工；更新 PAPERS.md 中 LongMem 的归类并加入 Memorizing Transformer、Larimar、MemT-1.0 的论文入口。
- 新增练习 31BD 和面试题 31BE。现有术语足以说明本章，不另增术语表项。
- 正文无公式、可执行代码或 Mermaid 图；公式渲染、代码运行和 Mermaid 页面门禁不适用。目标读者试读、外链最终逐项复核和第二十一册连续复读留待全册门禁。
- 静态验收：正文、大纲与目录七个标题及顺序一致；本章、有效大纲和审计中的本地链接目标存在；新增练习 31BD 与面试题 31BE 各出现一次；`git diff --check` 通过。
- 未运行测试、模型、API、kernel、benchmark 或 PDF 构建；不更新 PDF 或 `plan.md`。

本章首轮精修完成不代表第二十一册或全套书精修完成。下一入口为第四十七章《MoE + SSM + Attention 的未来组合可能》。全套书精修 goal 继续 active。
