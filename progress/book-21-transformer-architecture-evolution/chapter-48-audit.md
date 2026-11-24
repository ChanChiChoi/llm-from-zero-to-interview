# 第四十八章精修审计：多模态模型中的 Mamba 替代与混合

## 诊断与范围

完整复读旧稿 28 个二级标题及正文。旧稿多数内容在讲通用多模态系统、token 预算、训练与 serving、memory/retrieval、安全回退和复现包，既与第十五册的一般多模态架构和 token 化职责重叠，也重复本书第 43—47 章的通用 Attention/SSM 混合、模块路径、检索、记忆和 MoE 组合内容；“预算器”和成本账本等内容没有形成可核验的 Mamba 架构解释。按 02a 将本章职责收窄为状态空间模型进入视觉语言和长视频模型后的架构位置、信息路径与任务证据，并把章名改为“多模态模型中的 Mamba 替代与混合：视觉语言和长视频”。

## 结构与正文决定

经 02a 组织为六节并建立有效大纲：

1. 区分视觉编码器、视觉连接器、语言主干和跨模态融合四种架构位置。
2. 比较 Cobra、VL-Mamba 的主干/连接方式与 mmMamba 的蒸馏转换路线，并用受控对照说明任务表现差异。
3. 对照 VSS 对视觉扫描顺序的处理与 Q-Mamba 的 query 读取、序列压缩。
4. 以 AlignMamba 说明 token 级局部对应和模态级全局对齐可在 SSM 融合前显式处理，限定其情感表示任务范围。
5. 比较 VAMBA 与 TimeViper 中视频状态更新、文本读取、Attention 保留和视觉 token 压缩的位置。
6. 依据摘要、问答、定位、检索、长视频和多模态融合任务判读各论文证据，不把异构模型、数据或输入口径拼成总体排名。

正文保留章节主题组织，删除通用资源模型、token 预算器、服务流程、通用 memory/retrieval、回退和可复现包清单。正文没有公式、可执行代码或 Mermaid 图。

## 研究与证据

新发现、原文阅读位置及边界见[多模态 Mamba 架构研究记录](../../research/multimodal-mamba-architectures-2026-11-23.md)。依据 Cobra、VL-Mamba、Shaking Up VLMs、Q-Mamba、AlignMamba、VAMBA、TimeViper 与 mmMamba 的论文/正式出版页核验结构、实验范围和结果限制；研究者/作者报告只用于辅助理解讲解路径。EMNLP 2024 对照中的 captioning、VQA、reading comprehension、grounding 和合成检索表现限于其匹配的模型与训练设置。VAMBA 的帧数比较和 TimeViper 的吞吐结果分别绑定论文中帧分辨率、token 化、模型、batch、输入输出长度及硬件，不归因于 Mamba 单一因素。

## 同步与验收

- 新建[第四十八章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/48-多模态Mamba替代与混合.md)，更新第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)标题、大纲链接和六节标题。
- 更新 `PAPERS.md`、`KNOWLEDGE_GRAPH.md`，新增练习 31BF 和面试题 31BG；现有术语表已包含 Mamba/SSM 核心术语，无需重复增项。
- 静态验收：正文、大纲、目录六个小节标题及顺序一致；新题号在对应资料中唯一；新增和受影响本地链接目标存在；正文无公式、代码围栏或 Mermaid；`git diff --check` 通过。
- 目标读者试读、外链最终逐项复核及第二十一册连续复读留待全册门禁。本章无公式或 Mermaid，因此不触发对应页面渲染门禁。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建。

本章完成不代表第二十一册或全套书精修完成。
