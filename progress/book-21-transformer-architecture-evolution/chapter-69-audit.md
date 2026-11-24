# 第二十一册第六十九章精修审计

## 任务与旧稿诊断

本章按“主题思想不变重写”执行。旧稿 324 行、33 个 H2，将混合模型的逐层 cache 表示与通用 serving 管理混写，并多次重复 KV 容量、prefix cache、admission、压缩/淘汰、安全、snapshot/rollback 和端到端测试清单；有些推断将 attention 路径写成必然的精确证据或 fallback 能力。当前第二十一册第十一章已讲普通 KV 生成和容量，第六十五章已比较 local/global/recursive 路径，第六十三章主讲 Gated MLA，后续第七十/七十一章分别承担 latent cache 与递归状态；第二十章及第二十四册承担分页和 serving。

本章重新定位为混合架构中“哪些层保留逐 token 的显式/latent attention 状态，以及层表如何转为逐层 payload”的架构专题。本轮补齐有效 02a 文件，正文保留 hybrid 层间/层内安排、全局/局部/稀疏读取、按层估算和路径证据边界，删除通用 allocator、租户权限、scheduler、prefix sharing、preemption 和生产诊断流程。

## 结构与范围决定

- 普通 K/V、局部窗口 KV、逐 token latent 与递归 state 分别记账；attention 路径不自动等于普通全维 KV。
- 以 Qwen3.8-Next、Kimi K3、Hymba 为模型设计案例，以 Griffin 的 cache 账本和 Berkeley 讲义辅助说明；所有模型比例和质量数字绑定具体来源设置。
- 第十一章提供标准 cache 与基础容量前置；第六十五章负责架构路径比较；第六十三、七十、七十一章分别主讲 Gated MLA、latent cache、recursive state；第二十章和第二十四册负责底层 serving 实现。

## 研究、实施与验收

研究证据入口：[混合架构中的显式 KV 路径研究记录](../../research/explicit-kv-hybrid-architecture-evidence-2026-11-24.md)。QSA 的报告说明 query 级稀疏选择与 attention 计算，但没有在该节规定主 KV 的驻留/淘汰策略，因此正文不由 sparse read 推断主 KV payload 一定减少。

- 正文按四个主题组重写：层与状态表示、混合路径组织、普通 KV 逐层 payload、质量/资源证据。加入按层状态表、批内逐层求和公式及同形状教学算例；将普通 KV 与窗口、latent、递归 state、逻辑 payload 与设备显存分开。
- 同步第 69 章大纲、第二十一册读者目录与设计目录、练习 31CB、面试题 31CC、`PAPERS.md` 第 66 项和 `KNOWLEDGE_GRAPH.md`。术语表已有 GDN/QSA 条目，未重复添加。
- 正文、大纲与两份目录的 17 个标题名称、层级和顺序一致；本章 14 个本地链接目标有效。公式含义和算例已静态复核：同形状 32 层基线为 1 GiB，4 层为 128 MiB，8 层为 256 MiB。`git diff --check` 通过。
- 正文块公式与练习 31CB 的行内比例公式均未做 GitHub MathJax 页面实际渲染；读者试读、外链终审和第二十一册连续复读仍留在最终门禁。未运行代码、模型、benchmark 或 PDF 构建，未更新 PDF / `plan.md`。

第 69 章正文精修完成。第二十一册第 1—69 章已完成首轮正文修订，第 70—95 章共 26 章待精修；下一章为第七十章《Latent Cache：压缩历史表示》。全套书精修 goal 继续 active。
