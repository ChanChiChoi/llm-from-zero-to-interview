# 第二十六章精修审计

审阅日期：2026-11-20。当前状态：职责复核、章节研究、结构重设计、正文主题重写、配套同步与本地静态检查已完成；GitHub MathJax 实际渲染验收未完成。全套书精修 goal 继续 active；不更新 PDF 或 `plan.md`。

## 旧稿诊断与重写方向

旧稿 252 行、27 个 H2，多个主题反复出现：局部/global/random pattern、图路径和远程证据、block size、mask 正确性、动态路由、训练/推理一致性和 benchmark 指标。内容跨入第十九章的 pattern 与信息路径、第十章的复杂度分析、第二十章的 kernel/KV 执行、第八十九章的 GLM-5 DSA 案例及第六册/第二十四册的 serving 验收。旧稿同时包含 Python mask demo、自拟 effective sparsity/real speed 公式、冗长上线清单、重复面试题与练习，但没有把这些材料组织成连贯的“逻辑 mask—块布局—路由—执行—输出变化”解释路径。

本轮按 02a 先形成完整章节大纲，保留章名、文件路径和目录位置，将正文重组为六节：token mask 到可跳过 tile、block size 的资源取舍、固定布局与动态 block router、选择成本和输出近似、训练/执行契约、分析稀疏化实际作用的判断框架。删除经典 pattern 分类复讲、图直径/任务 benchmark 清单和 runtime 实操；相关主题按现行跨章职责回链。

## 研究范围与证据

章级知识与结构研究记录见[块稀疏 Attention 布局与动态路由研究记录](../../research/block-sparse-attention-layout-and-routing-2026-11-20.md)。本章把 token mask 与执行 tile 分开，使用 PyTorch `BlockMask` 文档和 cuDNN BSA 文档核对块布局与选中集合的归一化；用 MInference、NSA 说明推理时动态生成索引与原生训练稀疏架构的差别；用 PISA 预印本核对 router 本身可能仍有显著复杂度；用 AB-Sparse 的预印本观察不同 head 对 block granularity 的异质性。所有论文速度和准确率主张只绑定对应模型、设备、任务和实现，不外推为普遍结论。

教学组织参考 Stanford MS&E 319 Lecture 5 的 indexer/Top-K/main-attention 分步讲法，以及 CUHK CSC6203 Lecture 3 的 pattern 图示。第十九章与第八十九章已经承担 pattern 概念与 GLM-5 DSA 细节，本章不复写其来源摘要或配置。

## 正文重写记录

保留现行章名、文件、标题层级与六节编号，主题重写各节正文。新主线从 token mask 映射到可跳过 tile，继而解释块大小的索引/填充/硬件取舍、固定布局和输入相关路由、router 选择成本与选中集合上的 softmax、训练和执行共用的 mask 契约，最后给出连接图/路由/执行/结果四层判断框架。把第十九章的经典 pattern 与信息路径、第十章的 attention pair 复杂度、第八十九章的 DSA 配置以及第六册和第二十四册的服务验收留在其负责章节，并在正文建立导航。

移除旧稿中重复的 pattern 分类、图路径说明、基准清单、线上 rollout 流程和 Python mask demo；本章没有承诺编码或 API 操作结果，不保留执行代码。保留两处服务于知识结果的公式：块对是否包含可见 token pair 的布局定义，以及 router 选定 key 集合上的归一化 attention。第二条公式明确指出对所选集合精确计算不代表 dense attention 等价。

## 同步、校验与遗留门禁

- 同步第二十一册目录的大纲入口、`EXERCISES.md` 练习 31AE、`INTERVIEW_BANK.md` 面试题 31AB、`GLOSSARY_EN_ZH.md` 的 Block-sparse Attention 术语、`KNOWLEDGE_GRAPH.md` 稀疏执行链，以及 `PAPERS.md` 中 MInference、AB-Sparse、PISA 条目。新的证据和适用边界见[研究记录](../../research/block-sparse-attention-layout-and-routing-2026-11-20.md)。
- 静态检查：正文 H2 26.1—26.6 连续；两处顶格 `math` 围栏配对，复核块级存在性条件与选中集合 softmax 分母；本章没有代码围栏或 Mermaid；相邻章节链接及大纲、审计、研究记录链接目标存在；练习 31AE、面试题 31AB 与术语条目唯一；`git diff --check` 通过。
- GitHub MathJax 页面尚未实际打开，故两处公式的源码与含义已检查，但**公式渲染验收未完成**。PDF 不属于本轮任务。目标读者试读、外链逐条终审和第二十一册连续复读留待全册门禁。
- 未运行测试、示例代码、模型、API、kernel、benchmark 或 PDF 构建。全书精修继续进行，下一章为第二十一册第二十七章《Longformer、BigBird、Reformer 等长序列路线》。
