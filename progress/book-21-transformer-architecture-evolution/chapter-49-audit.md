# 第四十九章精修审计：Agentic Model Architecture

## 诊断与范围

完整复读旧稿 19 个二级标题及正文。原稿把模型决策、Agent 闭环、工具协议、事务/幂等、权限和安全、长任务状态折叠、记忆写入、验证器、评测、故障恢复与代码执行都列为并行主线；多处重复解释同一条模型—工具—观察循环，与第十七册的 Agent 方法/规划/记忆/RAG/评估安全、第二十册的工具执行与 Runtime/Harness、第二十二册的协议边界明显重叠。旧稿的通用动作效用、记忆价值、状态折叠和综合评分公式没有来自统一架构或可比实验的定义，toy Python 也偏向 Runtime 操作，不服务于第二十一册的架构判断。

依据 02a 将本章职责收窄为：模型工具能力如何通过提示、训练或检索形成；模型动作怎样经过宿主进入环境；外部状态、评测结果和内部模型结构如何区分。章名更新为“Agentic Model Architecture：模型能力与执行闭环”。与第四十五、四十六章只说明检索和记忆在本闭环中的信息位置，不重复具体机制。

## 结构与正文决定

按六节重构并建立有效大纲：

1. 区分单次生成、单次工具调用与多轮 Agent 系统。
2. 比较 ReAct 的提示交错、Toolformer 的工具训练、Gorilla 的 API 检索与训练，并说明这些路径可以组合。
3. 沿模型提议、宿主接收、环境动作/回执和下一轮观察，说明职责与事实来源。
4. 区分当前上下文、外部检索、持久记忆和 verifier 的接入位置。
5. 以 AgentBench、OSWorld、τ-bench 对照环境与评测协议，限制跨基准和隐藏结构归因。
6. 依据环境不确定性和反馈价值，讨论固定工作流与模型选择下一步动作的取舍。

正文删除事务重试、权限规则、context folding、详细记忆格式、安全清单和 toy code，保留能说明模型/宿主/环境边界的架构级信息。正文无公式、代码或 Mermaid。

## 研究与证据

新发现、原文位置和证据边界见[Agentic Model Architecture 研究记录](../../research/agentic-model-architecture-boundaries-2026-11-23.md)。ReAct 作为提示型交互模式，Toolformer 作为自监督工具调用训练，Gorilla/ToolLLM 作为 API 文档检索、微调和工具数据/评估组合，展示工具能力可落在不同系统层；这些论文结果限于各自模型、工具集和任务。AgentBench、OSWorld 与 τ-bench 分别提供多环境交互、真实桌面任务和带领域规则/模拟用户的多轮工具交互证据；它们的任务、环境与 verifier 不同，不形成统一排名，也不证明模型内部存在某个未披露模块。CMU 与 Stanford 课程 notes 及 Lilian Weng 的文章用于组织讲解路线，不作为效果证据。

## 同步与验收

- 新建[第四十九章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/49-AgenticModelArchitecture边界.md)，更新第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)的章名、大纲链接和六节标题。
- 更新 PAPERS.md、KNOWLEDGE_GRAPH.md，新增练习 31BG 和面试题 31BH；已有 Agent 与 ReAct 术语条目足以支撑正文，不重复添加术语。
- 静态验收：正文、大纲、目录六个标题及顺序一致；新增题号唯一；本地链接目标存在；正文无公式、代码围栏或 Mermaid；git diff --check 通过。
- 目标读者试读、外链最终逐项复核及第二十一册连续复读留待全册门禁。本章无公式和 Mermaid，不触发相应渲染门禁。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建。

本章完成不代表第二十一册或全套书精修完成。
