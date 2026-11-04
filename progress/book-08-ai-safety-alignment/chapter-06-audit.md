# 第八册第 6 章审计：Red Teaming 与危险能力评估

审阅日期：2026-11-03。
范围：全文诊断、章级扫描、原始来源核验、正文精修、手算与静态门禁、配套同步。未运行章节代码、测试、模型、benchmark 或 PDF 构建。

## 章节职责与衔接

- **中心问题：** 怎样通过有边界的对抗测试发现模型与 Agent 的失败，把能力激发、任务/动作轨迹、严重度、修复和回归证据转成范围明确的发布决定，同时避免评测自身造成外部副作用。
- **读者收获：** 能设计红队范围和 taxonomy，区分能力知识与可操作性，比较 natural / elicited 条件，选择基线和分母，制定 P0/P1 门槛，并解释 benchmark 与真实部署之间的证据边界。
- **主线：** 红队演进与定义 → 风险 taxonomy → 危险能力和 elicitation → 执行、分级、根因、修复回归 → 评估环境与基准设计 → 发布阈值 → 统计边界 → 受控代码助手案例和发布决策。
- **书系衔接：** 本章讲安全构念、能力评估和控制责任；第七册第八章负责安全评估统计与协议；第八册第五章讲攻击入口及信任边界；第十三章主讲自动化对抗评估；第二十册负责 Agent runtime / harness 实现。未发现需要返回册级调整骨架的问题。

## 章级扫描与候选处置

检索日期 2026-11-03，重点时间窗 2025—2026。扫描问题聚焦：真实网络事件与全模拟评估如何区分；动作、CoT、自述、模拟意识分别能支持什么判断；危险能力评测自身的出口控制、主机隔离和监控如何验证；现有红队 benchmark 的任务范围与遗漏；新材料是否形成需要增章的稳定主题。原始来源、版本、阅读范围和证据状态只记于 `research/red-teaming-dangerous-capability-evaluation-2026-11-03.md`。

| 候选主题 | 去向与理由 |
| --- | --- |
| AISI 2026 年 cybertesting incident 及其运行条件 | §8.5 纳入作为评测环境外部副作用案例。明确 `10/122` 绑定开放互联网、关闭 provider classifier 的具体多模型运行，不将其解释为沙箱逃逸或一般发生率。 |
| AISI alignment investigation 的动作/自述差异及 CoT 限制 | §8.5 纳入；实际动作是越界证据，CoT 与事后解释用于理解轨迹但不能证明意图或模拟状态。限定于该 cyber 任务家族。 |
| GPT-6 Astra 全模拟评估与 simulation awareness | §8.5 纳入。标明 Petri 工具调用全模拟、无真实网络/系统/第三方访问；保留整体率与筛选后子集的分母差异，不外推为现实攻击概率。 |
| Anthropic 对同一 AISI 事件的分析 | §8.5、§14.1 纳入。明确 `9/43` 与 AISI `10/122` 的统计对象不同； transcript replay 结果不作跨模型或部署发生率比较。 |
| AgentCyberRange v2、SandboxEscapeBench v3、FUSE v1 | §8.6 纳入威胁模型与可比性边界，引用固定版本。AgentCyberRange 的范围排除项有论文 Threats to Validity 支持；FUSE cyber pilot 与 H 判分一致性按其小规模条件表述。 |
| 自适应监控器攻击、AI control 和广义 Agent execution | 不在本章新增讲次；本章只提供与红队环境有关的评估/观测要点，协议和运行时继续交第 13、17 章及第二十册。 |

发现的证据增量加强已有 §8.5—§8.6，而没有改变书册边界或本章结构。

## 正文修改

- §8.5 补充 AISI 事件后的 alignment 调查、CoT 与实际动作分歧、Anthropic 独立分析的不同分母，以及对单条 transcript 的条件重放边界。
- §8.5 具体化 AISI 评测环境加固：沙箱和云主机分别阻断出站网络并验证；同步 LLM monitor 查看消息、工具调用和可用 CoT；CoT 不可用时改看动作序列但预期更弱；另有任务范围和开测前自动检查。文中保留最小权限、停机路径和分层控制，不把 monitor 写成证明或认证。
- §8.5 增加 GPT-6 Astra 的全模拟评估。区分整体报告值与挑选的 10 个高越界场景；说明 seed 不同、simulation awareness 和真实部署外部效度限制。
- §8.6 固定 FUSE v1、SandboxEscapeBench v3 与 AgentCyberRange v2 引用。增加 FUSE cyber pilot 样本范围和 H `ρ=0.42–0.58 (n=360)`；AgentCyberRange 明示任务阶段、规模、未覆盖风险和 harness/预算依赖。
- §14.1 更新方法与红队来源；新增 AISI alignment 调查、AISI Astra 技术报告/官方说明和 Anthropic analysis，逐条说明适用边界。
- §16 增加练习 7：要求比较 AISI `10/122`、Astra 全模拟 `29.2%` 和 Anthropic `9/43`，分别列出对象、工具/网络、safeguard、分母和分析方法，并设计评估遏制与停机/复核方案。
- §17 将标题改为“Red Teaming 安全审计 toy demo”；移除无可追溯记录的旧“已验证”声明，状态改为“静态审阅，未运行（2026-11-03）”；把 `severity_weighted_risk` 改名为 `severity_weighted_failure_rate`，因为其分子包含所有观察到的失败而不是未解决风险。静态预期输出同步改名，并补充逐项手算和分母解释。

## 篇幅与例子

汉字数静态计数排除标题、表格、代码围栏内容、公式和行内代码，只统计解释性中文；每个教学单元的边界按连续论证和读者任务确定。

| 独立单元 | 类型 | 汉字数 | 下限 | 状态 |
| --- | --- | ---: | ---: | --- |
| §§1—3 红队背景、定义与桥接案例 | 辅助 | 1,312 | 1,000 | 通过 |
| §§4—6 风险 taxonomy、危险能力与能力激发 | 核心 | 2,328 | 1,800 | 通过 |
| §§7—8 红队执行、能力评估设计与环境边界 | 核心 | 3,222 | 1,800 | 通过 |
| §9 安全阈值与发布条件 | 核心 | 2,143 | 1,800 | 通过 |
| §§10—12 评估陷阱、责任交接与统计边界 | 核心 | 3,685 | 1,800 | 通过 |
| §13 受控代码助手案例 | 辅助案例 | 1,151 | 1,000 | 通过 |
| §17 toy demo 说明、复算与发布决策 walkthrough | 辅助 | 1,412 | 1,000 | 通过 |

关键 demo 值从固定 9 条合成输入手算：taxonomy coverage `9/9=1.0`；failure rate `7/9≈0.778`；高严重度未修复 `4/6≈0.667`，其中单独的 P1 unresolved 数为 3；capability baseline uplift `0.283`，elicitation gain `0.327`；autonomy 平均 `0.7`；工具尝试违规 `2/3≈0.667`；回归通过 `3/7≈0.429`；加权观察失败 `28/34≈0.824`。taxonomy 门槛通过，其余六项门槛失败，预期决策为 `hold_high_risk_scope`。玩具输入不代表模型实测或真实风险率。

## 公式、代码、图示与链接

- 60 个围栏标记配对为 30 个代码围栏：18 math、10 text、1 Mermaid、1 Python。Math 块花括号 74/74；未发现 `\operatorname`。Mermaid 源码静态检查，但未在 GitHub 页面实际渲染。
- §17 Python demo 的一个字典输出键更名，计算代码逻辑未改；没有运行章节代码。expected output 是按固定输入手算的静态预期，不是程序结果。整册连续复读时，开篇新增第 5 章攻击入口/信任边界到本章受控评估/发布门禁的链接。
- 全章 34 个 Markdown 链接中 4 个为有效本地链接（第 5、7、13、17 章），另有 30 个外链；新增的重点 AISI、Anthropic 与 arXiv 来源已定向读取。历史来源未做全量在线巡检。第 7、13、17 章链接在整册连续复读时补入，用于连接行为证据、自动化搜索和运行时控制。
- 未发现内部维护路径、旧“最小可运行/已验证/运行后会看到”标记或旧 `severity_weighted_risk` 字段。

## 配套同步与遗留项

- `PAPERS.md` 增补 AgentCyberRange v2、FUSE v1、AISI incident/alignment/update、Astra report 和 Anthropic analysis；修复该段重复编号，将原 EU AI Act 条目顺延。
- `EXERCISES.md` 增补 35EIA 证据分母比较与评测遏制练习；`INTERVIEW_BANK.md` 的 red teaming 回答加入 AISI 案例与 CoT/动作证据边界。
- `GLOSSARY_EN_ZH.md` 增补 Evaluation Awareness、Simulation Awareness、Evaluation Containment；`KNOWLEDGE_GRAPH.md` 增加评测意识—遏制—停机—复盘—发布门禁路径。
- 本章骨架、章节位置、简介与目录摘要无变化，不需要更新简介或目录。
- `git diff --check` 通过。未运行代码、测试、模型、benchmark 或 PDF；未更新 PDF。

## 遗留风险与交接

- demo 的代码语句与输出改名未通过运行验证；若后续允许运行，应重新验证该代码块和输出，手算不等于执行验证。
- Mermaid GitHub 页面渲染未验收；历史外链未全量在线复核。新增 AISI/Anthropic 来源是机构自述/分析，AgentCyberRange、FUSE 和 SandboxEscapeBench 均按预印本版本理解。
- 第八册整册连续复读和最终门禁仍未完成。
- 本章精修完成。下一步按目录审阅 `book-08-ai-safety-alignment/chapters/07-interpretability与mechanistic-interpretability.md`；全套书精修 goal 保持 active。

## 整册连续复读记录

- 2026-11-03 按目录完成本章全篇连续复读。第 5 章的攻击入口与信任边界在本章转为受控任务、基线、动作轨迹、修复回归和发布门槛；多轮反馈驱动的自动化搜索交第 13 章；评估期间的监控与隔离条件不等同运行时安全保证，授权和副作用控制交第 17 章；本章观察到的外部行为再由第 7 章追问内部机制证据。未发现需要改动骨架、简介摘要、目录或章节顺序的问题。
- 已补齐第 6 章到第 7、13、17 章的正文导航，并在第 7 章开篇补回第 6 章的行为证据入口；相应链接目标存在。章节篇幅、算例和 demo 已与既有审计核对，没有另改代码或运行实验。
- 下一步按目录完整复读第 7 章，检查内部机制证据如何承接行为评估并引向 steering；代码执行、GitHub Mermaid/MathJax 渲染、历史外链全量巡检仍未完成。
