# 第四十七章精修审计：MoE、SSM 与 Attention 的组合设计

## 诊断与范围

完整复读旧稿 27 个二级标题及正文。原章把 MoE、SSM 和 Attention 描述成三个分别负责容量、历史和全局读取的预算，随后给出无来源的通用成本相加式、工厂/流水线/档案柜类比、假设性的路由顺序和容量估算代码；这些表达没有对应 Jamba、BlackMamba 或其他具体架构。旧稿后 15 节反复讨论 expert 热点、p99、调度、状态所有权、rollback、安全路由、生产容量规划及恢复协议，属于 serving/runtime 或 Agent 运行状态，且重复多个章节职责。

第四十三章已负责 Attention–SSM 混合结构、融合粒度和比例/层位置证据；第二十一章主讲 MoE 路由，第二十二章主讲 Dense/MoE 系统成本；第四十四章解释卷积、SSM、Attention 的模块信息路径；第五十章承担更广的架构演进判断。本章改为聚焦 MoE 在 Attention–SSM 主干中的插入位置、attention-free SSM-MoE 与 SSM 内部投影专家化。将章名从“未来组合可能”调整为“组合设计：架构位置与证据”，减少无证据预测。

## 结构与正文决定

经 02a 组织为六节，并建立有效大纲：

1. 区分序列混合路径（Attention/SSM）与条件参数路径（MoE），解释为什么三种名称不能定义一个通用 block。
2. 用 Jamba 说明在层间 Attention–Mamba 主干中以 MoE 替换部分 MLP，并分开描述混合比例和 MoE 增量的消融。
3. 用 BlackMamba 说明不保留显式 Attention 路径、由 Mamba 加 routed MLP 组成的另一种结构，限制其模型/数据和效率证据范围。
4. 用 NeurIPS 2025 Routing Mamba 说明共享路由的 SSM 投影专家化，并区分 Samba forward-FLOPs 对照与 FFN-MoE 联合实验。
5. 以比较表并列三种专家化位置和序列混合路径，明确不能从模块名推断 router 输入、expert 功能或通信路径。
6. 说明识别主效应与交互增益所需的控制组和匹配口径，列出仍需实验回答的问题。

正文没有公式、可执行代码或 Mermaid 图。移除了无架构来源的通用成本公式、玩具估算代码及 serving/Agent 运维检查清单；保留有来源的架构层级、消融结果与边界。

## 研究与证据

新知识、原文阅读位置及证据边界见[MoE、SSM 与 Attention 组合架构研究记录](../../research/moe-ssm-attention-composition-2026-11-23.md)。依据 Jamba ICLR 2025 正式论文核对 block 配置、MoE-MLP 插入位置以及 1.3B/250B-token、7B/50B-token 消融；AI21 发布方文章只作为架构图讲解入口。依据 BlackMamba 原始预印本核对 Mamba+routed-MLP 结构及作者报告的参数/训练设置。依据 Routing Mamba NeurIPS 2025 正式论文核对共享路由、Mamba 投影与 Samba 对照；约 23% FLOPs 差值限定在 421M/1.0B 对 511M、SlimPajama 20B tokens、8×A100、4K 验证及单次 4K forward 口径。未将异构论文分数合并为排名。

未运行本地模型、代码、API、kernel 或 benchmark；未重建或更新 PDF。

## 同步与验收

- 新建[第四十七章有效大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/47-MoE-SSM-Attention组合.md)，更新第二十一册[目录](../../book-21-transformer-architecture-evolution/目录.md)的章名、六个小节和大纲链接。
- 更新 `PAPERS.md` 与 `KNOWLEDGE_GRAPH.md`，补入 BlackMamba、Routing Mamba 和三种专家化位置；新增练习 31BE、面试题 31BF。现有术语已足够表达本章，不重复增项。
- 静态验收：正文、大纲、目录六个小节标题与顺序一致；新增/受影响本地链接目标存在；练习 31BE 与面试题 31BF 各唯一；正文不含公式、代码围栏或 Mermaid 图；`git diff --check` 通过。
- 目标读者试读、外链最终逐项复核及第二十一册连续复读留待全册门禁。本章完成不代表第二十一册或全套书精修完成。
