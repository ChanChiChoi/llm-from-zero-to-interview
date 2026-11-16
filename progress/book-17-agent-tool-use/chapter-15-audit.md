# 第十五章《Kimi K3/K3.1 Pro：模型发布、长任务 Harness 与证据边界》精修审计

审阅日期：2026-11-16。状态：完成首轮正文精修；Kimi K3.1 Pro 来源按固定 HF revision 和动态 API/Kimi Code 文档分层记录。比较 demo 为静态审阅、未运行；页面公式渲染、外链全量可达性、真实权重加载和独立 benchmark 均未验证。

## scope 与 reader_path

- **范围：** 完整复读第十五章（15.1—15.13），对照本册简介/目录、第十章评估视角、第十六册 reasoning state、第十八册产品化范围、第二十册 checkpoint/context folding、第二十二册工具协议，以及现有 K3/K3.1 Pro 研究资料。
- **读者前置：** 读者已理解 Agent 的模型、Harness、任务环境和 verifier 分层，并知道长任务可以通过工具、压缩和 checkpoint 继续。
- **本章贡献：** 读者能区分模型发布披露、技术报告/固定配置、provider 接口合同、Harness 实验和本地加载证据；能为长任务重放建立版本化 manifest，并解释“报告有提升”为什么不等于跨 harness 或生产场景的因果结论。
- **阅读路径：** 先从发布文章与配置表区分事实层次，再看 K3 的 KDA/AttnRes 与 XTM 作为历史基线；随后设计长任务 state manifest、复算评测可比性；最后把 K3.1 Pro 的固定结构、API/Kimi Code 接口和 benchmark 条件对齐。跨模型 reasoning state 交给第十六册，持久化 checkpoint/折叠/trace replay 交第二十册与第二十二册。

## chapter_scan 与 evidence

- **日期、问题和范围：** 2026-11-16；检查 K3.1 Pro 是否仍只有发布承诺、模型/API 与 Kimi Code ID 是否相同、K3 到 K3.1 的结构变化是否可由公开配置支持、`reasoning_content` 是否是通用隐藏推理合同，以及长任务分数是否能跨 harness 直接比较。资料类型限官方发布页、公开模型卡/config、产品 API 文档和已引用的机制论文。
- **候选及去向：**

| 候选 | 处理决定 | 证据范围与未覆盖项 |
|---|---|---|
| K3 发布文章、K3 技术报告与固定权重 metadata | 保留为 2026-09 历史基线；区分文章、固定 revision 和 serving recipe 的证据层。 | K3 固定 config/index 支持公开结构与清单字段；metadata 不证明完整下载、GPU kernel 或线上 acceptance。 |
| K3.1 Pro 后训练与 Gated DSA | 纳入版本更新；与 K3 的 24 个 Gated MLA 层分开。 | 固定 HF revision `6d0a0202956796543364d666ab19c6ad1a494ee1`；模型卡/config 支持层数、专家和 top-k 字段，发布方未公开完整训练 recipe 或目标 runtime 实现。 |
| K3.1 Pro API 与 Kimi Code 调用合同 | 纳入 ID、reasoning effort、thinking-only、消息 replay 和缓存失效区别。 | Kimi API 使用 `kimi-k3.1-pro` 且默认 effort `max`；Kimi Code 使用 `k3.1-pro` 等 Code IDs 且文档默认 `high`。动态产品文档的值不合并为通用模型默认值。 |
| Kimi Code Bench 3.0 Long-horizon 与 Terminal-Bench 4.0 | 纳入发布方报告值和 harness 口径；K3 的 2026-09-23 榜单快照改标历史。 | 61.2%/55.95% 是内部 coding-agent benchmark 报告值；Terminal-Bench 4.0 的 K3.1 Pro 使用 Claude Code harness。任务日志、统计不确定性和独立同条件复现不完整。 |
| 新一轮 cyber/xdev 评测细节 | 未扩入本章主线。 | 本章结果是版本与 Harness 证据阅读，不承担漏洞能力/安全基准主讲；相关安全能力和风险仍应按第十一章及其专门研究资料处理。 |

- **来源记录：** K3.1 Pro 发布方信息、固定 HF revision/config、API/Kimi Code 动态文档、评测条件和未验证边界记录在[Kimi K3.1 Pro 来源与证据记录](../../research/model-update-2026-09/kimi-k3.1-pro-source-notes.md)。本审计只记录章节去向和修改决定，不把进度写回 research。

## architecture_decisions 与 coverage_map

- 保留第十五章作为模型/Harness 证据案例，不扩成发布新闻汇总；显示标题扩展为“Kimi K3/K3.1 Pro”，原文件路径不变。
- 将旧 K3 榜单段落标为 `2026-09-23` 历史快照；K3 首发文章中的价格、ID 和缓存命中率也标成对应产品/时间快照。
- 删除正文与 PAPERS 中指向不存在的 `kimi_k3_manifest_audit.py` 的断链和“通过该脚本门禁”的表述；改为直接引用固定 config/index，并限缩为公开 metadata 清单证据，不宣称本地执行结果。
- 新增 K3.1 Pro 版本段落，呈现 post-training 的发布方披露、Gated DSA 结构差异、API/Kimi Code ID 和 effort 差异、消息回放合同及 benchmark/harness 边界。

| 知识点 | 对读者结果的作用 | 深度与去向 |
|---|---|---|
| 发布文章、技术报告、模型卡、固定 config 与实现 | 能判断每种来源可支持什么结论。 | 核心；K3 作历史基线，K3.1 Pro 结构字段固定 revision。完整训练 recipe、runtime kernel 留作未知。 |
| 长任务 Harness 与可恢复状态 | 能写出目标、权限、模型/Harness 版本、工具状态、artifact、checkpoint 和最终提交的 manifest。 | 核心；动态 checkpoint、上下文折叠与 trace replay 交第二十册相关章节。 |
| KDA/AttnRes、Gated MLA 与 Gated DSA | 能按版本区分序列状态、深度残差和稀疏注意力配置。 | 核心支撑；论文讲一般机制，K3/K3.1 config 讲具体公开配置，不外推 serving 性能。 |
| Kimi API / Kimi Code 消息合同 | 能区分 product ID、provider default、reasoning effort、完整 assistant message 和最终 content。 | 核心；跨模型 reasoning state 交第十六册，通用工具 protocol/replay 交第二十二册。 |
| 长任务 benchmark 与可比性 | 能把绝对百分点、相对比例、任务集、effort、harness、fallback 和环境放入同一证据账。 | 核心；官方内部 benchmark 与发布方 leaderboard 保留具体来源和边界。 |

## examples、formula 与 code 状态

- **算例：** §15.5 三条合成 EvalRow 展示同条件与不同 harness/硬件的比较键；输出手工推演为 `0.1 0.5 False`。§15.13 对发布值 `61.2%` 与 `55.95%` 做纯算术差 `5.25` 个百分点、相对差约 `9.4%`，明确这不是统计显著性或因果估计。
- **公式：** `R=F(M,H,E,B,D)` 把 Agent 结果连接模型、Harness、环境、预算和任务分布；新加入的百分数仅使用行内算术。静态检查未发现 `\operatorname`、`\(...\)` 或尖括号下标写法；GitHub MathJax 实际渲染未确认。
- **代码：** §15.5 比较 demo 的可执行语句未改动，现标“静态审阅，未运行（2026-11-16）”。静态核对输入字段与 `comparable` 的四个匹配键，并手算预期输出；未运行代码或测试。§15.11 不再声称不存在的 metadata 审计脚本已执行。
- **围栏和链接静态核对：** 第十五章 5 对 Markdown 围栏标记配对，15.1—15.13 连续；新加的第十六、二十、二十二册本地交叉链接目标有效。K3 fixed artifact 的网页 index、模型 endpoint、外链全量可达性和页面图表/MathJax 未执行终验。

## sync_status 与 open_risks

- 同步第十七册《简介》《目录》显示标题与摘要；更新 `PAPERS.md`、`EXERCISES.md`、`INTERVIEW_BANK.md`、`GLOSSARY_EN_ZH.md`、`BOOK_SERIES.md` 与第四册架构索引；第十七册架构审计推进到第一至十五章首轮精修完成。
- Kimi API/Kimi Code 文档是动态接口合同；将来维护时应重新固定 model ID、effort 默认值和回放字段。模型卡的 `reasoning_content` 不能外推为各家隐藏推理的通用定义。
- Kimi Code Bench 3.0 为发布方内部评测，公开信息不足以重建全部运行与统计不确定性；Terminal-Bench peer 值有跨来源、harness 和 fallback 条件差异。没有 API 请求、权重加载或 benchmark 复跑。
- 第十七册未完事项包括各章记录的动态代码复验、目标页面 MathJax/Mermaid、外链终验和整册沿主线复读；这些不影响本章本轮文本与静态链接修订状态。
- **下一入口：** 第十八册《产品、商业化与落地》，先依据册级审视规范复核简介/目录、架构审计和书系相邻分工，再按顺序开始逐章精修。
