# 第十二章《Agent 综合系统设计与复盘》精修审计

审阅日期：2026-11-16。状态：正文、导航和跨章指标同步完成；章节审计 demo 仅静态审阅与手算，未运行。目标平台 Mermaid/MathJax 呈现、外链全量可达性和代码执行仍待最终验收。

## scope 与 reader_path

- **范围：** 通读 `chapters/12-agent面试题.md` 全部 820 行，并对照本册简介、目录、架构审计、第十章评估指标、第十一章安全指标和相关 research 记录。
- **读者前置：** 已掌握第 1—11 章的 Agent 对象、工具执行、反馈循环、规划、memory、RAG、代码/界面任务、多 Agent、评估与安全基本机制。
- **本章贡献：** 读者把一个业务请求组织成状态、证据、权限、动作和完成条件一致的系统设计；能说明 workflow/Agent 的职责边界，覆盖成功、拒绝、证据不足和未知状态，并用外部证据解释结果。
- **阅读路径：** 先写任务契约并选择执行方式，再组合工具、循环、规划、memory/RAG 和任务型 Agent，继而设计协作、评估与安全；最后以合同审核系统串起状态转换、工单写入、超时恢复和验收。

## 主要诊断与决定

| 候选/断点 | 判断 | 处理 |
|---|---|---|
| 标题和目录定位 | 正文从开篇起就是综合系统教材，文件路径及目录标签仍叫“Agent 面试题”。 | 将目录和简介中的读者可见标题改为“Agent 综合系统设计与复盘”；保留历史文件路径以兼容现有链接。删除正文中的文件路径维护说明。 |
| 前十一章知识回收 | 多数概念在前文已有主讲；本章价值应是把它们放回统一的任务契约和控制边界，而不是重讲完整专章。 | 保留必要回顾和系统层取舍，用合同审核贯穿状态、证据、权限、有限写入、验收与恢复；面向评估、安全和工具调用的细节链接回第十、十一章。 |
| 综合设计缺少可复核闭环 | 原案例描述了五个阶段和失败路径，但缺少一条带版本、阈值、回执及未知状态处理的逐步轨迹。 | 补入合成合同/政策版本、`90 > 60` 的规则判断、`pending_review` 工单写入超时、按幂等键查证和 verifier 验收的完整状态表；明确所有数据均为教学假设。 |
| 系统平面关系 | 六个责任平面以列表呈现，关键循环和独立验收关系不够直观。 | 在 §12.24 增加 Mermaid 责任图，显示候选动作经 schema/业务/权限检查后执行，外部观察回到控制器，证据不足或状态未知回到恢复路径，验收和事件账本独立记录。 |
| 安全评估口径 | `pre_effect_block_rate` 与第十、十一章统一术语不一致；原表也没有区分草稿和未知状态。 | 改用 `unauthorized_block_rate`，按统一分母解释 `unauthorized_proposal_rate`、`unauthorized_execution_rate`；明确 `blocked`、`drafted`、`unknown` 的区别，并复用高风险确认/安全处置口径。 |
| 教学公式 | 规划成本、净收益和证据充分条件需要读者能沿假设复算。 | 给 `C_plan` 加 token 账本算例；给 `NetLift` 标出归一化教学量、系数及硬风险约束；用同一合同证据判断复算 `S(c)`。 |
| 代码状态与输出 | 代码没有可追溯的执行记录；整体项目证据并集覆盖容易掩盖单主题缺项。 | 标为“静态审阅，未运行（2026-11-16）”，将输出说明为手工复核；解释 overall 并集分数 `1.0` 不代表 q4/q5 没有各自缺项。 |

## chapter_scan 与 evidence

- **日期和问题：** 2026-11-16；核对本章如何综合第 1—11 章能力、动态 SDK/行为规范入口是否仍适用、危险动作指标是否沿用第十/十一章合同，以及 Agent benchmark 来源是否仅支持所述环境和评测用途。
- **候选与去向：** workflow/Agent 边界、工具 schema 与执行授权、Plan-Act-Observe、任务拆分、memory/RAG、代码与界面操作、多 Agent 净收益、trace/verifier/安全评估均留在本章作为综合设计决策；协议生态和 runtime 深入内容继续交第二十、二十二册，环境与 benchmark 合同交第十三章。
- **来源复核：** 官方 OpenAI Model Spec 固定到 2026-10-29 版本；Agents SDK Tools、Guardrails、Tracing 和 Agent evals 为动态官方文档，本轮核阅日期为 2026-11-16。AgentBench、WebArena、OSWorld、SWE-bench、GAIA、τ-bench 仍作为具体任务/环境入口，不将其分数拼成通用 Agent 能力值。
- **证据边界：** Model Spec 表示预期模型行为，不是执行授权保证；SDK 文档表示该 SDK 的公开接口，不是通用架构要求；benchmark 结果绑定任务、环境、verifier、模型与 Harness。相关事实和版本边界复用既有[系统设计架构证据](../../research/llm-system-design-architecture-evidence-2026-11-04.md)、[Agent 基础与 SDK 资料](../../research/agent-foundations-2026-11.md)、[安全与 Model Spec 资料](../../research/safety-alignment-overview-2026-11-03.md)及[Agent 评估基准资料](../../research/specialized-frontier-eval-cluster-depth-review-2026-10-27.md)；本轮没有把审阅进度写入 `research/`。

## coverage_map

| 知识点 | 对读者结果的作用 | 正文深度与去向 |
|---|---|---|
| 目标、身份、范围与完成契约 | 为架构决策提供输入及停止条件 | 核心；以合同助手任务契约开篇，并在系统图和状态轨迹中回收。 |
| Workflow、Agent 与工具执行边界 | 决定哪些路径固定、哪些判断交模型、谁允许副作用 | 核心；回顾接口概念，强调 schema、业务检查、策略和 executor 分层。 |
| 反馈、计划、分解、状态与记忆 | 让 observation、权限变化和过期信息真正改变下一步 | 支撑；连接第 3—5 章，不重写各章的完整算法与 lifecycle。 |
| RAG、claim 与证据状态 | 让最终判断能回到获授权、有版本和定位的来源 | 核心；给出合同与政策证据例，完整检索机制回链第 6 章。 |
| Code/UI 与 Multi-Agent 模式 | 按环境、验证手段、并行收益和风险选择执行方式 | 支撑；侧重设计取舍，具体操作细节交第 7—9、14 章。 |
| Result/trace/verifier/safety/cost | 说明完成、阻断、草稿和未知状态，并定位失败 | 核心；与第十、十一章共享指标口径。 |
| 整体架构和工单恢复 | 独立设计一个受限、可追溯的 Agent 系统 | 核心案例；状态表从 `received` 走到 `completed`，覆盖 90/60 规则、超时、幂等查询和外部状态确认。 |

## example_status、formula_status 与 code_status

- **完整案例：** 合成数据 `c-1842:v7`、`p-2026-03`、90 天付款期限和 60 天阈值。证据需授权、版本、生效范围、定位、claim 支持及冲突解决；仅当全部通过才创建 `pending_review` 工单。模拟提交超时后状态为 `unknown`，按幂等键查到工单并由 verifier 核对，才向用户报告已创建、尚未批准。
- **公式/数值：** `C_plan` 按 120 + 2×60 + 3×20 = 300 token 手算，注明不包含工具和等待成本；`NetLift` 使用显式归一化教学假设复算为 0.008，并强调硬风险门槛不可被效用分抵消；`S(c)` 展示全条件为 1、政策冲突时为 0。coverage/vector 等结构式明确是审查表示，不是用户能力评分。
- **代码：** §12.27 示例逻辑只静态阅读，未运行。按源码集合和权重手算得到 q1–q5 分别为 1.000、0.779、1.000、0.973、0.662；overall 为概念 0.955、公式 0.800、demo 0.833、trace 指标 0.857、安全 0.800、评估 0.833、项目证据并集 1.000、取舍 0.833；平均 topic coverage 0.883，弱项 q2/q4/q5，红旗 `missing_confirmation`/`weak_project_evidence`，`all_checks_pass=False`。并集项目证据分不替代逐 topic 缺项。
- **页面/源码状态：** 新 Mermaid 按标签引号和管道箭头规则编写；可做源码级核对，但 GitHub 实际 Mermaid/MathJax 渲染未验。公式源码未运行页面渲染。

## link_status 与 sync_status

- **内部链接：** 第十二章安全评估已回链[第十章](../../book-17-agent-tool-use/chapters/10-agent评估.md)与[第十一章](../../book-17-agent-tool-use/chapters/11-agent安全.md)指标定义；两文件均存在。简介、目录指向原稳定路径，显示名称现统一。
- **配套同步：** 更新 `book-17-agent-tool-use/简介.md`、`目录.md` 的第十二章标题和摘要；更新 `INTERVIEW_BANK.md` 第 121 题，使回答框架覆盖任务契约、责任边界、完整轨迹、未知状态和证据化项目陈述；更新本册架构审计和全套交接。无新增核心术语或论文，故未改 `GLOSSARY_EN_ZH.md`、`PAPERS.md`、`EXERCISES.md`、`PROJECTS.md`。
- **链接限制：** 本轮核对了 Model Spec 当前版本并阅读动态 SDK/评估文档；未逐条访问所有 benchmark 外链，也未做整册全量链接验收。

## 静态检查

- 本轮目标文件的 `git diff --check` 通过；正文、导航、审计和交接文件未发现尾随空白。
- 第十二章共有 18 个 tilde 围栏标记和 8 个反引号围栏标记，均为偶数；静态搜索未发现 `\\operatorname`、`\\(...\\)` 或 `x_{<...}` 写法。
- 第十、十一章本地回链目标存在。新 Mermaid 图只做源码检查，GitHub 页面渲染未确认；未运行代码或测试。

## open_risks 与 handoff

- §12.27 Python 示例和输出示例仍需在声明环境中执行后才能标为已验证；本轮未运行代码、测试、模型或 benchmark。
- GitHub 对新 Mermaid 图和 MathJax 的实际呈现仍待平台验收；全部外部链接可达性和第十七册整册连续复读尚未关闭。
- **下一入口：** 第十三章《AgentWorld 模型与环境闭环》。先确认通用 Agent—环境闭环与同名 benchmark 的边界，再沿状态、action、observation、reset、verifier 和长任务评估主线精修。第十七册第 1—12 章首轮精修完成，整册目标仍待第 13—15 章及连续复读/门禁；全套书 goal 继续 active，不更新 PDF 或 `plan.md`。
