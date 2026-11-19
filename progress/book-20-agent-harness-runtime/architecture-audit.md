# 第二十册《Agent Harness、Coding Agent Runtime 与智能体工程框架》架构审视

审视日期：2026-11-19。阶段：册级初审、发现扫描、简介与目录重构完成；第 1—22 章章级扫描与精修完成，第 23—24 章待逐章处理。

## 审视范围与方法

- 完整阅读本册《简介.md》《目录.md》，盘点 24 个章节的标题、层级和篇幅；阅读第 1—16 章的章目标/范围说明及第 17—24 章小型专题的标题结构，用于建立初始章节地图。此阶段没有通读 24 章正文，地图需在逐章审阅时校正。
- 对照第十七册 Agent 基础、第十九册工程排障和第二十二册协议生态的简介/目录及既有进度交接，确认本册负责运行时工程，不重复主讲 Agent 行为理论、评估统计方法或 MCP/A2A 协议规范。
- 发现扫描日期为 2026-11-17。最新官方接口与项目事实见[Agent Runtime 与 Harness 近期来源扫描](../../research/agent-runtime-and-harness-source-scan-2026-11-17.md)；既有 Agent 评估、代码 Agent benchmark、Tracing、隐私和长任务来源分别见对应 research 记录。本次扫描用于确认候选主题和范围，不是对每章正文的证据核验。

## 书册契约

- **定位与类型：** 面向 Coding Agent 的概念/架构教材和工程手册；从任务输入沿状态、上下文、动作、执行、权限、观察、恢复和验收到最终交接，解释运行时如何把模型能力组织成可控系统。11—13 章是公开产品案例，20—24 章是有版本边界的 provider/API 案例。
- **读者起点：** 已理解 LLM 的 prompt、生成、工具调用基本过程，能读 Python/JSON 和常见命令行任务；建议具备 Git、文件系统、HTTP API 的基本经验。无需有生产 Agent 项目经验；关键前置和补充入口应在简介与每章开头说明。
- **能力目标：** 读者能追踪一次 coding task 从用户约束到工具副作用和最终 artifact 的完整状态；能为工具、文件、终端、权限、上下文、trace/replay 和 evaluation 设计相互一致的接口；能区分模型输出、执行事实、权限决定、持久状态和任务验收；能根据公开材料比较产品，但不把闭源实现猜测写成事实；能说明版本、环境、预算、verifier 和隐私如何限定结论。
- **中心问题：** 如何让 Coding Agent 在真实文件和工具环境中持续完成任务，同时限制副作用、保存可恢复状态、留下足以解释和验收结果的证据？
- **主线：** 建立 Harness/Runtime 概念 → 追踪 coding task 和工具调用 → 管理文件、命令、上下文、权限和 trace → 构造 evaluation → 观察产品案例 → 汇成系统设计与故障排查 → 深入长任务状态、上下文折叠和 harness-aware evaluation → 阅读 provider-specific API 合同。
- **书系分工：** 第十七册讲 Agent 的状态、规划、记忆与工具行为；本册讲执行系统和控制边界。第七册讲评估方法，本册讲运行 harness。第十九册从现场故障诊断切入，本册讲系统构造。第二十二册讲 Function Calling、MCP、A2A、Skill 和生态治理，本册只讲协议如何进入 runtime。第六册讲推理服务与部署；本册不替代模型 serving 教材。第十八册讲产品价值、商业和运营，不在本册扩成产品管理课程。

## 简介与目录审计

### 简介

- 已补充 LLM/工具调用、Python/JSON、Git 和命令行前置；生产经验不再作为隐含门槛。学习结果改为追踪完整动作状态、设计系统边界、区分状态所有者、恢复长任务、评估完整配置和表达证据边界。
- 已把主路线、Coding Agent 设计、长任务、评估排错、产品比较和 API 查询路径写进简介，并明确产品/API 案例与通用机制的区别。

### 目录

- 已按四个阅读组重组目录，保留 24 章路径和编号；补入逐章问题/结果、核心学习路径、长任务进阶入口、Provider/API 查询边界与相邻书链接。
- 第十四章讲 Harness 如何治理 MCP/A2A 接入；协议语义和生态全景链接第二十二册。第十一至十三章比较公开可见产品能力，须与厂商当前版本绑定。

## 初始章节地图

| 章 | 中心问题与读者结果 | 对书册目标的贡献 | 依赖与交接 |
| --- | --- | --- | --- |
| 1 Harness 总览 | 一条修复任务怎样经过模型提议、策略决定、工具副作用、任务状态和验收？能复走闭环，并区分权限、隔离、状态与 trace。 | 建立全书对象、证据边界和运行主线。 | 前置第十七册 Agent 概念；交第 2、3 章。 |
| 2 Agent Runtime 架构 | 一次失败后的续跑中，context、model、parser、policy、executor、state 和 trace 如何协作？能区分 Task 状态与 Runtime phase，并安全处理超时、审批和并发冲突。 | 把组件责任、状态所有权和恢复路径连成一条可追踪的运行过程。 | 依第 1 章；交第 3—10 章。 |
| 3 Coding Agent 工作流 | 代码任务如何从目标澄清走到探索、修改、验证与交接？能跟踪版本变化并用任务证据决定下一步。 | 用一条端到端任务串起工具、文件、上下文、权限、trace 与 evaluation 模块。 | 依第 1—2 章；交第 4—9 章。 |
| 4 工具系统与 Registry | 模型如何发现和调用工具，宿主如何校验参数、权限和结果？ | 定义动作接口与治理入口。 | 交文件、终端、MCP 及权限。 |
| 5 文件系统与代码编辑 | 怎样保护 workspace、用户并发改动和 patch 边界？ | 讲文件副作用和可恢复编辑。 | 依第 4 章；交第 6、8、9 章。 |
| 6 终端执行与命令安全 | 命令如何分级、隔离、确认、取消和限制资源？ | 讲最强副作用执行通道。 | 依第 4—5 章；交权限与 trace。 |
| 7 上下文管理与记忆压缩 | 每轮模型应该看到什么，怎样压缩而保留目标/约束？ | 管理模型输入、预算与信息失真。 | 依第 1—3 章；交第 9、17—18、20—21 章。 |
| 8 权限模型与安全沙箱 | 权限决定、用户确认、环境隔离和审计各自限制什么？ | 建立全书安全边界。 | 依第 4—7 章；细节交第 14—16、22 章。 |
| 9 Trace、日志、回放与可观测性 | 怎样从一次运行恢复输入、动作、结果和副作用？ | 建立复盘和证据层。 | 依第 2、7—8 章；交第 10、17—19 章。 |
| 10 Evaluation Harness | 怎样复现任务、初始化环境、判断 artifact 成功并比较变更？ | 将执行证据变成评估。 | 依第 9 章；方法回链第七册，深化交第 19 章。 |
| 11 Claude Code 架构分析 | 怎样从官方公开资料和可观察证据分析工作循环、权限与执行边界，并标出未知内部？ | 用多表面产品案例练习证据分层和可迁移审阅方法。 | 依第 1—10 章；交第 12 章 OpenCode 和第 13 章产品对比。 |
| 12 OpenCode 架构分析 | 固定版本下，配置、指令、Agent、权限、宿主执行和状态证据如何共同构成一次可审阅任务？ | 用公开合同和定向源码路径分析开放 Runtime，并明确其本身不提供 Agent sandbox。 | 依第 1—10 章；交第 13 章做同维度产品比较，并为第 14—15 章提供扩展与控制边界案例。 |
| 13 Codex 与主流产品比较 | 怎样分清产品入口、Agent/harness、执行环境和评估器，并基于公开证据比较一条修复任务？ | 将产品案例归纳为“产品选择”和“受控 harness 比较”两种方法。 | 依第 11—12 章；不做产品排名；交第 14 章协议集成和第 15 章系统设计。 |
| 14 MCP/A2A 与 Harness 集成 | 外部工具和 agent 如何进入 runtime 的 capability、permission、trace 与 evaluation？ | 讲协议集成责任。 | 依第 4、8—10 章；协议主讲交第二十二册。 |
| 15 Agent Harness 系统设计 | 如何从任务契约设计到状态、执行、恢复和验收边界？ | 以一条修复任务整合前十四章的运行知识，形成可审阅的系统设计与部署取舍。 | 依第 1—14 章；交第 16 章故障诊断，checkpoint 与 workspace 所有权分别深化到第 17、21 章。 |
| 16 Harness 故障诊断与面试题 | 一次错误、危险、卡住或“测试通过但交付不合约”的运行，怎样从证据找到最早偏离并安全恢复？ | 训练契约、事件轨迹、真实状态与 verifier 的分层判断，并把故障固化为回归案例。 | 依第 1—15 章；checkpoint/幂等由第 17 章深化，harness 归因由第 19 章深化。 |
| 17 Long-Running Agent 与 Checkpoint | 中断后哪些事实已成立、哪些动作未派发、哪些结果仍 unknown？如何按证据安全续跑？ | 区分对话、任务 checkpoint、Trace、工作区和远端事实；用 intent/effect ledger、幂等合同、fencing 和独立验收处理恢复。 | 依第 7、9、15—16 章；context folding 交第 18 章，workspace 所有权交第 21 章，provider 状态合同交第 20、23 章。 |
| 18 Context Folding | 长任务轨迹如何折叠成更短的模型输入，同时保留约束、来源、版本和未知副作用？ | 深化第 7 章逐轮输入选择与第 17 章任务恢复；分别检查任务结果、规则保留、证据支持和折叠成本。 | 依第 7、17 章；RAG/memory 回链第 7 章，workspace/Artifact 交第 21 章，provider 状态合同交第 20、23 章；评估方法交第 19 章。 |
| 19 Harness-Aware Evaluation | 结果差异来自模型、harness、环境还是交互？ | 深化第 10 章的归因与轨迹评估。 | 依第 9—10 章；统计方法回链第七册。 |
| 20 Responses API | 怎样沿 Response/item 与工具往返读懂 OpenAI 的状态、续接、保留、流式和用量合同？ | 让读者区分 API response、item、宿主工具执行、conversation state 与任务验收。 | 依第 4、7、9、17 章；workspace 回链第 21 章；结论绑定官方文档访问日期。 |
| 21 Persistent Workspace | 长任务恢复时，哪个状态来源能证明对话、文件、测试结果和外部系统当前各是什么？ | 区分上下文/记忆、checkpoint、workspace、job 与外部系统事实，并让 artifact 和验证结果绑定具体 revision。 | 依第 7、9、17 章；文件编辑回链第 5 章、权限/保留回链第 8 章、评测回链第 10、19 章；与第 18、20、23 章互链。 |
| 22 OpenAI-Compatible API 边界 | 调用方怎样判断一个具体兼容 endpoint 是否保留了应用依赖的字段、工具、状态、流式和用量行为？ | 让读者按 SDK/endpoint/provider/model/日期记录支持、转换、不支持和未知，再决定复用、适配或缩小承诺。 | 依第 20 章；Gemini/Claude 专属合同分别交第 23—24 章；推理引擎服务端 API 实现交第二十四册第 57 章。 |
| 23 Gemini Interactions | 中断后应续接对话历史、事件流还是长解码？哪些外部事实仍由宿主确认？ | 区分 Interaction ID、function call/result ID、SSE event ID 与 continuation token；说明逐轮配置、thought 签名回放、工具副作用及保留合同。 | 依第 17—18、20—22 章；API/runtime 事实绑定官方文档日期，通用工具治理回链第 4、8、17、21 章。 |
| 24 Claude Sonnet 5.5 状态与安全路由 | thinking state、tool transition、safeguard 与 routing 数字如何读？ | 单一 provider/model 的发布与 API 案例。 | 依第 8—10、20—23 章；不把模型能力归因成 runtime 保证。 |

## 发现扫描与候选去向

扫描日期：2026-11-17。问题：书册是否缺少支撑可恢复、可控制、可测量 Agent runtime 的重要主题；近期接口与工程证据是否改变现有章节分工？检索类型包括官方 API/项目文档、协议规范和评估研究。

| 候选主题 | 证据与适配性 | 去向与判断 |
| --- | --- | --- |
| Model×Harness×Environment 与最终 artifact 评估 | 近期 Harness-Bench、TraceTrial 等资料延续配置交互、任务 verifier 和轨迹证据主题；不同研究的任务、重复数和归因定义不能混为同一排名。 | 保留在第 10、19 章；统计方法回第七册。第 19 章需按固定论文版本限制具体数字。 |
| 用户确认不等于执行隔离 | OpenCode 官方 threat model 明言 permission UX 不是 sandbox；工具策略、进程/文件系统隔离和业务授权须分开。 | 第 8、12、15、16 章为候选修订；不新增章节。事实见新增研究来源记录。 |
| Provider conversation state 与任务状态 | Gemini Interactions 的 previous ID 只续接历史；调用级 tools/instructions/config 需重传，stateless thought/tool steps 有签名要求，存储与删除另有合同。 | 第 23 章主讲，并与 17、20、21 章区分对话历史、持久 artifact、checkpoint 和执行回执；不把 API session 说成通用 memory。 |
| Trace 隐私与离线/集成测试边界 | OpenAI Agents SDK 当前文档支持本地确定性 orchestration test，并要求网络/provider/sandbox-owned 行为另测；Tracing 可记录敏感输入输出，配置和保留由部署决定。 | 第 9、10、15、16 章定位同步；通用 telemetry/privacy 不重复第十九册安全章。 |
| 长任务状态、context folding、workspace | 既有研究涵盖任务状态与对话历史、持久记忆、外部 artifact、压缩保真和 checkpoint。 | 保留 17—18、21 章为互补专题；逐章核对同一案例和状态所有权，避免重复主讲。 |
| MCP/A2A 的新语义和生态治理 | 协议版本与客户端能力持续更新；新版本可改变 transport、授权、task 生命周期和可发现能力。 | 第 14 章仅讲 runtime 集成与边界，协议规范和完整生态交第二十二册；每项注明固定 revision/date。 |
| 代码 Agent 产品扩展与模型榜单 | 产品端不断增加 IDE、server、cloud、MCP、subagent、scheduled task 等功能，能力和默认值易过时。 | 第 11—13 章保留公开证据驱动的架构分析，不做产品排名或闭源内部推断；按访问日期更新。 |

本次没有发现必须新增独立章节的新主线。此结论仅覆盖上述范围；逐章扫描若发现核心承诺缺少必要机制，再返回册级调整。

## 架构决定

1. 暂保留 24 章和现有顺序；第 1—16 章提供完整通用路径，第 17—19 章加深可恢复状态、上下文折叠和完整 harness 归因，第 20—24 章作为版本化 Provider/API 查询案例。逐章精修发现实质重复或超载时再调整。
2. 已重写简介与目录，明确真实前置、学习结果、核心路径和进阶/查询路径；目录分组未改变文件编号或路径。
3. 逐章审视示例是否真正解释本章机制，核对公式、可运行代码和验证声明；不把代码长度、自动指标或统一阈值当成深度证明。第 1 章已删除原 toy trace Python audit 和固定 gate，改用可复走的合成状态轨迹与伪代码。第 2 章移除九项未绑定任务口径的组件比例及聚合 gate、长 Python 审计 demo，改用合成失败续跑轨迹和分组件事件记录。第 3 章删除固定工作流总 gate 与长 Python 审计 demo，改用完整任务轨迹和标明分母的独立观察维度。第 4 章删除多项 Registry 覆盖率指标、G_toolreg 聚合门槛和长 Python audit demo，改用完整工具调用轨迹及带匹配分母的分项观察率。第 5 章删除多项文件编辑质量指标、G_fileedit 固定门槛和长 Python 审计 demo，改用路径/版本/实际 diff 的完整合成轨迹及有明确分母的 stale-write 观察；后续章节按各自读者任务决定代码和指标的保留方式。
4. 第 11—13 章仅比较公开接口、可复查行为和固定 revision；第 20—24 章只表达相应 API/provider 的合同，不把历史默认值推广为通用 runtime 定律。
5. 第六章删除命令安全旧审计脚本、`G_cmd` 和固定阈值，改用明确标记的合成执行轨迹及按事件、适用对象和分母复核的观察路径；权限、确认、OS 沙箱、网络/凭据、资源和任务验收各自承担不同边界。
6. 第七章以合成的显示名修复任务讲解上下文来源、优先级、预算、日志/历史压缩、task state、memory 和 freshness；删除旧 Python context auditor、任意指标阈值及 `G_context`，改为按单位和分母报告的手算审阅。Anthropic compaction 保留为带访问日期的 API 合同示例，model-specific thinking 状态回链第二十四章。
7. 第八章以合成显示名修复任务贯穿权限决策、一次性审批、下游业务授权、全执行路径隔离和审计；删除原固定阈值、聚合 G_perm 与长 Python auditor，改用分母明确、互不汇总的手算诊断量。目录摘要仍准确。
8. 第九章用合成显示名修复 trace 连接上下文、模型提议、策略判断、补丁、测试回执与最终状态；厘清 trace/span/event/log/metric/artifact 和 task/session/run 的边界，区分只读回放、固定回执模拟与隔离重跑。删除原聚合 G_trace 和长 Python 审计 demo，隐私部分改为内容最小化、Artifact 分权、导出前 fail-closed，以及绑定当前 SDK 文档的处理器行为；采样与指标采用各自明确分母，不汇总成通用 gate。
9. 第十章把 fixture、attempt、系统配置和 experiment 分开，以独立的教学构造显示名任务说明数据集、环境预检、Runner、verifier、分母、配对比较与报告；分类保留 Agent 结果、环境/运行错误、评分错误和无法判定。删除原 15 项任意聚合的质量率、`G_evalh` 固定准入公式和长 Python audit demo；保留 bucket、环境、验收器、结果/约束、安全、成本、可复现、版本、trace 与报告完整性的观测意图，逐项写清单位和适用分母。第七册承接统计设计，第十九章承接 Model×Harness×Environment 归因。
10. 第十一章把 Claude Code 从“终端型产品”改为多表面、不同执行环境的公开产品案例。按官方文档截至 2026-11-18 的合同组织 gather/action/verify、session/context、permission、OS shell sandbox、checkpoint、hooks/MCP/subagents/skills/plugins 与 OTel；将内部模块图明确标为分析者推断，将未公开 planner、system prompt 和排序/路由机制保留为未知。删除十项覆盖率及 `G_claude` 聚合 gate 和约 140 行 Python audit demo，以合成登录修复流程和人工证据审阅表保留分析目标。第 12 章承接 OpenCode 可检查实现，第 13 章承接跨产品比较。

11. 第十二章固定 OpenCode v2.3.2（release commit c0ad18b）及 2026-11-18 官方文档，按明确标为教学构造的认证修复任务连接配置与指令、Agent/子 Agent、工具提议、action/resource/effect 权限、宿主执行、Session/Event、Snapshot/Diff、compaction 和独立 verifier。用定向源码核对权限求值和 Session 操作入口；把模块图标为分析抽象，并引用项目 SECURITY.md 明确 OpenCode 自身不提供 Agent sandbox。删除旧版工具/字段清单、Plan 只读和权限隔离泛化、覆盖率、G_opencode gate 及约 140 行 Python demo；改用问题—证据—边界人工审阅表。更新练习 35P、面试题 40N 与目录摘要，第 13 章承接跨产品比较。

12. 第十三章完成产品案例横向精修。按 2026-11-18 官方文档与研究证据，校正 Codex 本地/Cloud、Cursor IDE/Cloud、SWE-agent maintenance-only 与 mini-SWE-agent、OpenHands Canvas/ACP/Runtime 的比较单位；以认证响应修复的明确教学构造追踪上下文、提议、派发、回执、diff 和验收。把“产品选择”与“harness 因果比较”分开讲；删除原覆盖率公式、综合 ranking/gate 与 Python toy audit demo，改为条件记录卡和证据边界。更新练习 35Q、面试题 40O；正文取舍见[第十三章精修审计](chapter-13-audit.md)。

13. 第十四章按 MCP `2026-07-28` 与 A2A v1.0.1 官方规范修正协议生命周期边界：区分新版 MCP 无协议会话核心、旧版握手、Draft Tasks extension 与 A2A Task；以订单重试教学构造追踪 MCP 只读发现、A2A AgentCard/委派、身份与本地策略、远端状态、Artifact、本地执行及独立验收。删除原协议覆盖率公式、汇总 gate 和 Python toy audit demo，改为可复走运行轨迹、失败恢复表和无分数评估/回放边界；更新练习 35R 与面试题 40P。详细去向见[第十四章精修审计](chapter-14-audit.md)。
14. 第十五章完成系统综合设计精修。以合成登录错误修复任务串联任务契约、编排选择、架构责任图、Session/Task/Run/Workspace/Artifact/Effect 状态所有权、Context/Model/Capability 边界、权限/隔离/确认/验收、崩溃恢复、评估与部署。删除原通用模块覆盖公式、聚合系统上线 gate 和长 Python 审计 demo；将原有模块知识保留为一条可追踪设计过程，并把失败结果分为已知、拒绝、基础设施错误和 unknown。章节边界与第 16 章故障排查、第 17/21 章 checkpoint/workspace 专题保持衔接；替换练习 35S 并修订面试题 40C、40Q。详细取舍见[第十五章精修审计](chapter-15-audit.md)，来源证据见[系统设计外部资料记录](../../research/agent-harness-system-design-evidence-2026-11-18.md)。
14. 第十五章完成系统综合设计精修。以合成登录错误修复任务串联任务契约、编排选择、架构责任图、Session/Task/Run/Workspace/Artifact/Effect 状态所有权、Context/Model/Capability 边界、权限/隔离/确认/验收、崩溃恢复、评估与部署。删除原通用模块覆盖公式、聚合系统上线 gate 和长 Python 审计 demo；将原有模块知识保留为一条可追踪设计过程，并把失败结果分为已知、拒绝、基础设施错误和 unknown。章节边界与第 16 章故障排查、第 17/21 章 checkpoint/workspace 专题保持衔接；替换练习 35S 并修订面试题 40C、40Q。详细取舍见[第十五章精修审计](chapter-15-audit.md)，来源证据见[系统设计外部资料记录](../../research/agent-harness-system-design-evidence-2026-11-18.md)。

15. 第十六章将“实战坑清单”重构为一条并发编辑故障的证据诊断过程：复述契约、对齐 h0/h1/h2 事件、定位过期写入的最早偏离、区分执行故障与 verifier 漏检，再根据是否已派发/已回执/状态 unknown 选择恢复路径。保留权限、上下文、stale edit、命令副作用、注入、trace、eval、循环成本、MCP 工具过载和环境漂移十类主题，改为症状—证据—判断表；删除通用 coverage 公式、综合 pitfall gate 与约 400 行 Python auditor。将原宽泛定义/设计/产品比较问答回链到第 1、4、6—15 章，新增故障排查问答；用一项多步事故复盘练习替换七项跨章重复练习。详细去向见[第十六章精修审计](chapter-16-audit.md)，证据来源见[故障诊断、重试与运行证据记录](../../research/agent-harness-failure-diagnostics-evidence-2026-11-18.md)。
16. 第十七章从 `task-42` 的 CI 回执迟写和代码审查响应丢失切入，区分模型上下文、Trace、checkpoint、workspace 与服务端状态；保留旧稿的检查点字段、状态机、幂等/提交协议、跨日续跑、版本/租约/预算、故障注入、分母和保留/取消主题，删除任意汇总公式、覆盖阈值及长审计 demo。增加条件写入、outbox/派发、服务方幂等合同、unknown 核查和 fencing 的故障边界；不宣称任意跨系统 exactly-once。补充 Chubby/etcd 原始资料，来源和范围见[checkpoint 与恢复研究记录](../../research/long-running-agent-checkpoint-recovery-evidence-2026-11-18.md)；正文取舍见[第十七章精修审计](chapter-17-audit.md)。同步目录、练习 35EL 和面试题 40AA，下一篇深化 Context Folding。
17. 第十八章将 context folding 定位为对“下一轮模型输入”的有损变换，不等同 transcript、checkpoint、Trace、workspace 或长期 memory。以 `task-42` 演示硬约束、已验证 CI 结果、版本绑定、来源指针、假设和未知审查副作用如何分别折叠和恢复；用一组明确教学预算算例讲 token 预留，不设通用触发百分比。删除旧任意损失函数、coverage/约束总分、权重排序公式与审计 demo；加入 Anthropic/OpenAI 公开合同差异、DeepSeek Search Agent 策略与 2026 context-compaction 评测证据，均限于各自任务、模型和配置。来源证据见[Context Folding 与长任务压缩来源记录](../../research/context-folding-and-compaction-strategy-2026-11-18.md)，正文覆盖和交接见[第十八章精修审计](chapter-18-audit.md)。同步目录摘要、练习 35EM 与面试题 40AB。

18. 第十九章将重点从重复评测流程和发布方案例移到 Model×Harness×Environment 的受控比较、交互、分层结果、Trace 归因、资源成本与 holdout。保留 task/trial/trajectory/outcome 定义和 manifest；第十章继续主讲 fixture、Runner、verifier 与分母，统计设计回第七册第十章。移除未验证的自拟聚合函数、release gate 和不存在的产品 source-note / audit-script 声明；能追溯的 provider 资料链接到既有研究记录或本册第 20、23、24 章。详细取舍见[第十九章精修审计](chapter-19-audit.md)，来源见[Model–Harness 交互与归因研究](../../research/model-harness-interaction-attribution-2026-11-18.md)。同步目录摘要、练习 35EN、面试题 40AC。
19. 第二十章将 Responses API 收窄为 OpenAI 官方接口合同：以一个 read_file 教学构造贯穿 Response 封套、typed output items、自定义 function call/function output 往返、SSE、三种会话续接方式、store/保留、后台状态和 usage。移除旧稿通用化的事件 ACK/任意 sequence 恢复、协议完整性乘积 gate、成本自拟公式和缺失的 GPT-5.6/GPT-6 source notes 与 toy audit script；任务副作用、checkpoint、workspace 和 verifier 分别回链第 17、21 章。来源证据见[Responses API 当前合同记录](../../research/openai-responses-api-contract-evidence-2026-11-18.md)，原稿去向与单篇验收见[第二十章精修审计](chapter-20-audit.md)。同步目录摘要、练习 35EO 和面试题 40AD。
20. 第二十二章将 OpenAI-compatible 收窄为调用方迁移合同审读：以 Anthropic compatibility layer 作完整实例，解释 endpoint 范围、字段移位/忽略/报错、tool schema、reasoning、stream、state、usage 与 provider/model/region 条件。删除五层 taxonomy、`G_compat`/`G_route`/`T_request` 公式、统一 gate 和重复的固定 conformance 清单；将深层工具责任回链第 4、20 章，状态副作用回链第 17 章，服务端 API 鉴权/限流/协议实现回链第二十四册第 57 章。事实与证据范围见[兼容 API 研究记录](../../research/openai-compatible-api-compatibility-evidence-2026-11-19.md)，原稿映射与正文验收见[第二十二章精修审计](chapter-22-audit.md)。同步目录、练习 35EQ、面试题 40AF，并修正一条既有 Responses 兼容面试答法。
21. 第二十三章将 Gemini Interactions 收窄为跨轮状态和恢复边界：用工单评论任务区分 Interaction history、SSE stream、long-decode continuation 与宿主外部副作用；解释 `previous_interaction_id`、function `id`/`call_id`、`event_id` 和 `continuation_token` 各自用途，逐轮配置、thought summary/signature、status、retention 与删除合同。删除 Gemini 3.8 榜单/预算主张、旧 SDK `_gaos` schema 实现和视频 token/质量数字；保留 provider-managed `processing_call/result` 与自定义 function 的边界，以及官方文档对 thought signature 必需性的标记差异。来源见[Gemini Interactions 状态证据记录](../../research/gemini-interactions-state-evidence-2026-11-19.md)，取舍与静态验收见[第二十三章精修审计](chapter-23-audit.md)。同步目录、`BOOK_SERIES.md`、练习与题库、知识图谱。

## 开放风险与交接

- 第 21 章新稿已移除旧稿中指向缺失 Claude audit script 和 Qwen source-note 的链接；Claude Fable 5.1 的 provider thinking-state 合同列为第 24 章候选，须在该章定向核验后决定去向。Qwen3.6 source-note 链接仍在第二十一册第 83 章 §83.15，待该册逐章审阅时处理。
- 第 23 章和 `BOOK_SERIES.md` 对缺失的 `gemini-3.8-flash-source-notes.md` 的引用已删除，改用 2026-11-19 的官方 Interactions 状态证据记录。thought signature 的 requiredness 表述仍存在文档标记差异；本轮未调用真实 endpoint，正文将其保留为待目标接入方确认的边界。
- 第 1 章原有“0 依赖 Python demo”承诺、9 个固定阈值和约 330 行 toy auditor 已移除；内容改由完整合成修复轨迹、最小伪代码和任务特定验收解释。其余章节中仍有 Python demo 或自动 gate 的部分尚待逐章核验；决定删除代码时须保留必要的过程、数值解释和明确去向。
- 第 2 章原有运行指标公式及 `G_runtime` 固定门槛已移除；核心观察意图改为记录 session/tenant 隔离、phase 转移、context 来源与新鲜度、adapter/parser/registry 错误、执行回执、恢复/checkpoint 和取消/超时事件，并要求为任何汇总率声明单位和分母。旧的 GPT-5.6 运行时归属快照由 OpenAI 当前公开 Agents API/SDK/Responses API 对比替代；Qwen3.7 Plus region/scope 能力细节待第二十二册第 47 章复核和承接，不能在第二十册正文里留作无去向内容。
- GitHub 页面公式/图表渲染和全量外链可达性尚未核验；动态产品事实需按每章核验日期固定。
- **当前下一入口：** [第二十四章《Claude Sonnet 5.5 thinking state 与安全路由》](../../book-20-agent-harness-runtime/chapters/24-claude-sonnet-5.5-thinking-state与安全路由.md)。第二十三章的正文取舍见[第二十三章审计](chapter-23-audit.md)，来源范围见[Gemini Interactions 状态证据记录](../../research/gemini-interactions-state-evidence-2026-11-19.md)。第二十册第 24 章和全套书精修 goal 继续 active。
