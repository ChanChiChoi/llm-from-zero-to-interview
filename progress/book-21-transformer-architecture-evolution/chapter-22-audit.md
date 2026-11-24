# 第二十二章《Dense 与 MoE：训练、服务和成本的完整取舍》精修审计

审阅日期：2026-11-20。章节职责复核、MoE 系统成本研究、六节结构大纲、正文主题重写和配套同步已完成。本章无公式、代码或 Mermaid；目标读者试读、外链逐条终审和第二十一册连续复读仍待后续门禁。按用户要求不更新 PDF 或 `plan.md`；全套书精修 goal 继续 active。

## 读者契约与章节边界

- **前置：** 第二十一章的 token-to-expert 路由；第五册第四章的 total/active 参数、MoE 训练账本；第五册第七章的 Expert Parallel 与 all-to-all；本册第十五章的 prefill/decode 和第十一章的 KV cache。
- **离章结果：** 能区分总参数、每 token 激活参数、权重驻留、训练更新成本和线上延迟/吞吐；能说明稀疏激活改变了哪些计算、存储与通信工作；能在明确负载、设备和目标质量下比较 Dense 与 MoE，而不从 active 参数或单篇报告推出普遍成本结论。
- **本章职责：** 比较 Dense 与 MoE 在训练、通信和服务阶段的系统资源代价，解释拓扑、实现与负载如何改变结果，并说明比较证据应使用的共同条件。
- **边界：** 第二十一章主讲路由与专家结构；第五册第四章主讲 router loss、capacity、token dropping 与训练账本；第五册第七章主讲并行策略和 all-to-all；第六册及第二十四册主讲完整 serving、调度、SLO 与压测。本章不复述这些教程或提供发布 gate。

## 旧稿诊断与结构决定

旧稿有 21 个 H2，混合训练/服务成本、参数和容量口径、EP/all-to-all 教程、router 与负载诊断、通用成本公式、发布清单及多个收尾总结。第五册第四章已主讲 total/active 参数、router/capacity、训练预算和检查清单；第五册第七章已主讲 all-to-all 与 EP 调度；本册第十一、十五章已分别处理 KV cache 和 prefill/decode。旧稿反复得出“MoE 不一定更便宜/更快”，但没有形成从资源口径、训练有效工作、通信关键路径到分阶段服务的连贯比较路径。

保留章名、文件路径、目录入口和现有章节顺序，将正文重组为六节：资源账本、每次训练更新的实际工作、通信暴露等待与拓扑、prefill/decode 服务差异、比较分母与共同条件、按约束选择路线。完整的现行大纲及各节中心内容见[第二十二章大纲](../../写作规范/全套书不同册的章节大纲结构/book-21-transformer-architecture-evolution/22-Dense与MoE训练服务trade-off.md)。

## 研究核验与正文决定

来源版本、检索范围、实际阅读位置和证据边界见[MoE 系统成本与 Dense/MoE 比较研究记录](../../research/moe-system-costs-and-tradeoffs-2026-11-20.md)。正文分别使用 MegaBlocks 的特定 A100 稀疏训练对照、DeepSeek-V3 作者报告的训练 GPU 小时、Megatron-Core 的 EP64/H100 暴露通信结果，以及 DuoServe 和 METRO 的不同 MoE serving 设置；每个案例都绑定原报告中的模型、实现、硬件、负载或分母。没有将跨论文数字拼成排行榜，也没有把作者报告解释为 Dense 对照或本地复现。

正文把参数量、每 token 激活路径、驻留权重和实测时间分开；解释训练账本还包含路由、负载形状、模型状态、通信及执行效率；把传输字节与关键路径上的暴露等待区分开；分别讨论 prefill 与 decode 对 expert batch、权重驻留/搬运及 KV 状态的影响；最后区分固定预算比质量、固定质量比训练成本和固定服务目标比效率。结论按当前任务、数据、拓扑、runtime、并发和 SLO 条件化。

## 删除与同步

- 删除重复的 total/active 参数与容量教程、MoE loss/overflow 配方、EP/all-to-all 步骤说明、通用单位成本公式、serving 发布检查表、泛化成本结论和重复总结；主讲知识回链至第五册第四/七章及本册第十一/十五/二十一章。
- 新建本章结构大纲和研究记录；本章内链接指向第五册第四章、第二十一章和第十一章，来源链接贴近相应证据主张。
- 在 `EXERCISES.md` 增加 31AA，在 `INTERVIEW_BANK.md` 增加 31X；在 `GLOSSARY_EN_ZH.md` 增加 Resident Weights / Weight Residency 与 Exposed Communication Time；在 `KNOWLEDGE_GRAPH.md` 增加从参数口径、训练状态与通信到分阶段服务及公平比较的关系链；在 `PAPERS.md` 增加 MegaBlocks、Megatron-Core、DuoServe 和 METRO 入口并链接研究记录。

## 静态检查与遗留门禁

- 正文 H2 为连续的 `22.1—22.6`；含两张比较表，无数学定界符、`math`/代码围栏或 Mermaid。公式专项渲染门禁不适用。
- 静态检查确认正文两张表列数一致；正文、大纲、研究记录、审计、架构审计、全局进度和 `PAPERS.md` 本章新增研究记录链接目标均存在；练习 31AA 和面试题 31X 各自唯一；两个新增术语各出现一次；新增审计与进度文本有结尾换行、无尾随空格；`git diff --check` 通过。`PAPERS.md` 全文件扫描仍发现 23 个既有失效链接实例，已登记于[共享索引本地链接审计](../shared-index-local-links-audit-2026-11-19.md)，不由本章新增条目引起。
- 未运行代码、测试、模型、API、kernel、benchmark 或 PDF 构建/更新。
- 目标读者试读、正文外链逐条终审、第二十一册连续复读仍待后续门禁；第十二章两处、第十四章一处、第十八章一处、第二十章两处和第二十一章一处公式的 GitHub MathJax 渲染仍待验。本章不含公式。

下一入口为[第二十三章《小模型、边缘模型与高效 Transformer 改造》](../../book-21-transformer-architecture-evolution/chapters/23-小模型边缘模型与高效Transformer.md)。全套书精修 goal 继续 active。
