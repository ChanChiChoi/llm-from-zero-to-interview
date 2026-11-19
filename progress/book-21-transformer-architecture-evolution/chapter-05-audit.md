# 第 5 章《FFN、SwiGLU、Gated MLP 与 MoE FFN》审阅记录

审阅日期：2026-11-19。状态：章级来源扫描、正文精修、配套资料同步和本地静态检查完成；代码未运行，目标平台 MathJax 渲染与第二十一册连续复读留待后续门禁。

## 读者契约与章节职责

- **书册目标贡献：** 让读者能解释 Transformer 中 position-wise FFN 的作用，按一致参数口径比较 dense FFN 与 SwiGLU，并读懂 MoE 的总参数和每 token 使用参数各代表什么。
- **读者前置：** 知道 Transformer block 的 attention 和 token 表示；能阅读向量/矩阵形状和基础乘法。第 2、3 章分别回顾 attention 信息读取与 Q/K/V 表示。
- **离章结果：** 能解释同一 FFN 参数怎样按位置共享并分别计算；写出 GLU/SwiGLU 的两路投影与逐通道调制；计算近似等参数宽度；区分 FFN 门控与 MoE expert 路由；说明 active 参数不是 FLOPs、显存或延迟。
- **读者路径：** token 间信息交换与 token 内通道变换 → position-wise dense FFN → GLU/SwiGLU 门控和参数量 → 实验边界 → 单 token MoE 路由 → total/active 参数 → 工程成本与后续章节。
- **前后交接：** 第 6 章承接 FFN/attention 输出如何进入 residual stream 和 norm；第 21 章主讲 MoE 路线、router、capacity、负载均衡和故障；第 22 章主讲 Dense/MoE 训练、服务、通信和成本；第 54、55、58 章各自承接 LLaMA、Mixtral、DeepSeek 案例。

## 外部研究与主题去向

2026-11-19 通过中英文论文题名、术语和课程检索词完成章级扫描。学科资料检索包含 Transformer/position-wise FFN、GLU/SwiGLU、Shazeer sparse MoE、GShard、Switch、Mixtral、DeepSeekMoE；教学组织检索查阅 Stanford MS&E 319 Autumn 2026 课程大纲与 Lecture 4 MoE 讲义。原文位置、检索词、版本、事实边界和未覆盖范围见[FFN/SwiGLU/MoE 证据记录](../../research/transformer-ffn-swiglu-moe-evidence-2026-11-19.md)。课程资料用于观察组织方式，不据此声称学习效果。

| 外部主题 | 原文支持与读者需要 | 本章去向 |
| --- | --- | --- |
| Position-wise dense FFN | Transformer §3.3 定义逐位置、同参数的两层 ReLU FFN；读者需要看见输入、宽度变化与 token 间边界 | 5.2 解释 $[B,T,D]\rightarrow[B,T,M]\rightarrow[B,T,D]$ 和共享参数 |
| FFN 参数与计算账本 | 两矩阵参数量约 $2DM$；$4D$ 是常见配置点而非普遍标准 | 5.3 给出含义、条件和算术近似 |
| GLU/SwiGLU 结构与实证 | Shazeer 比较 GLU 变体并按等参数/等计算缩小中间维；结果来自 T5/C4 设置，微调噪声较大；PaLM/LLaMA 的采用不等于单因素因果实验 | 5.4—5.7 解释门控、三投影、参数匹配和证据边界 |
| Sparse MoE 路由 | MoE expert 通常为 FFN；token 路由到少数 expert 后加权合并；权重处理有变体 | 5.9 只讲单 token 核心路径，详细 router 进入第 21 章 |
| Total/active 参数与资源 | Mixtral 报告 8 experts/top-2、约 47B total/13B active，并指出 active 参数没有计入服务内存和硬件利用率全貌 | 5.10 区分模型参数账、selected parameters、FLOPs/存储/通信/延迟 |
| Expert specialization 与模型路线 | GShard、Switch、Mixtral、DeepSeekMoE 展示不同路由/专家设计；各自结果受任务、训练、硬件和参数口径约束 | 5.11 用短表说明路线差异，详细机制交第 21、22、55、58 章 |
| 2026 模型参数盘点 | 闭源或快速更新的前沿参数表缺少稳定可比的官方字段，且重复后续模型专题 | 移除旧表；不以产品名或榜单推断未披露结构 |

Stanford MS&E 319 的 Lecture 4 先定义稀疏 MoE 与参数口径，再讲 router/训练、负载均衡、capacity 和设备放置。本章采用先定义再交代资源边界的基础顺序；该课程面向已具数学与算法前置的研究生，不能证明它是本书初学者的唯一最佳路线。本轮没有从公开资料中确认另一个可复核的同主题课程讲义顺序，也没有找到能证明该编排提升学习效果的因果研究。

## 正文取舍与配套同步

- 删除旧“本讲范围与资料”作者指令、虚构的统一 `G_ffn` 准入条件和基于人工 routing 的标准库成本审计 demo。旧 demo 使用任意通过阈值并输出 `best_route`；其 overflow 统计是 token-expert assignments，不能称为唯一 token 数，也不能代表通用架构验收。
- 删除“FFN 知识/模式存储”“SwiGLU 普遍更强”“MoE 必然更快/更好”等超出证据的断言；把 attention 与 FFN 描述为不同计算轴，避免互相替代式归因。
- 删除 2026 frontier 模型参数表；具体且可溯源的模型案例交给第 54、55、58 章。
- 保留短 PyTorch SwiGLU 模块作为 shape 对照，确认代码语句与旧版本相同，补上“静态审阅，未运行”状态；并说明行向量公式与 `nn.Linear` 的权重存储形状如何对应。未运行代码或框架。
- 在 MoE 基础路由公式后加入两个 token 选择不同 top-2 expert 的教学示意表，帮助读者把路由集合与加权输出对上；编号和权重均明确为示意值。
- 按 Mixtral 原文进一步区分 active 参数相关的推理计算量、sparse total 权重对应的服务内存和设备利用率，并说明专家权重可在部署中跨设备分布，不应推断每张 GPU 都存放完整专家集合。
- 将 FFN/SwiGLU 参数 hand-calc 和 MoE total/active 概念练习同步到 `EXERCISES.md`；将面试回答同步到 `INTERVIEW_BANK.md`。
- 在 `PAPERS.md` 增补 GLU Variants、早期 sparse MoE、GShard/Mixtral 和 DeepSeekMoE 来源及条件；删除 `GLOSSARY_EN_ZH.md` 中无证据的 `FFN / MoE Gate` 术语，细化 SwiGLU 定义；从 `KNOWLEDGE_GRAPH.md` 删除对应门禁节点。
- 将 `PROJECTS.md` 的 FFN/MoE 审计输出改为带假设的资源账本，不再输出任意 `failed gates` 或 `best route`，并把通信和延迟留给带硬件/拓扑/workload 数据的测量。

## 验收与遗留风险

- 静态验收：H2 标题 `5.1—5.12` 连续；18 个 Markdown 围栏标记成对；行内数学定界、全角标点间距及 8 个本地章节链接检查通过；共享索引题号唯一；`git diff --check` 通过。
- 代码保持“静态审阅，未运行”；没有运行测试、模型、API、benchmark 或 PDF 构建/更新。
- GitHub MathJax 实际渲染、外链可达性和目标读者试读未确认；缓存或延迟的任何实测结论不在本章声明范围内。
- 后续第 6 章 residual/norm 结构与训练稳定性审视已完成，见[第 6 章审阅记录](chapter-06-audit.md)；当前入口为第 7 章位置编码。全套书精修 goal 继续 active。
