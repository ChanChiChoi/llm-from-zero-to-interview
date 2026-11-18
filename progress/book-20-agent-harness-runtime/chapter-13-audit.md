# 第二十册第十三章精修审计

日期：2026-11-18  
章节：[第十三章《Codex 与主流 Coding Agent 对比》](../../book-20-agent-harness-runtime/chapters/13-codex与主流coding-agent对比.md)  
状态：章级扫描与正文精修完成；全套书精修 goal 继续 active。

## 本章范围与读者路径

本章承接第十一章 Claude Code 与第十二章 OpenCode 案例，面向已理解 Coding Agent 基本工具调用与运行时边界的读者。中心问题是：比较一项代码修复任务时，怎样区分产品入口、Agent/harness、执行环境与评估器，并让结论能由公开合同或运行证据复核？

阅读路径从一项教学构造的认证修复任务开始：缺失 bearer token 时 `/api/profile` 错误返回 200，目标是改为 401、增加回归测试、限制改动范围且不 push。读者先跟踪输入、提议、宿主派发、实际副作用、测试回执、diff 与独立验收；再读产品 profile 与不同 surface/runtime；最后分别学习完整产品选型和控制变量的 harness 因果研究。

本章对书册的贡献是把前两章的单产品公开证据审阅归纳成跨产品比较方法。读者离开时应能说清比较对象是什么、哪些条件相同或不同、证据支持到哪里，以及哪些仍是 unknown；不以公开能力清单或单次 benchmark 分数替代判断。

## 章级定向扫描

检索日期：2026-11-18。问题：主流 Coding Agent 的公开入口、Agent/平台层次、执行位置、权限与状态合同如何变化；比较产品时应怎样界定单位和证据边界？

- 产品材料使用官方/维护者文档：Codex CLI、Cloud、sandbox 与 ChatGPT/Codex surfaces；Claude Code platforms、permissions、sandbox 与 checkpointing；OpenCode V2 文档及固定 revision（承接第十二章研究）；Cursor Agent 与 Cloud Agents；Aider repo map、chat modes、commands 与 edit formats；SWE-agent/mini-SWE-agent 文档；OpenHands Agent Canvas、ACP agents、architecture 与 runtime。
- 评估材料使用 SWE-agent ACI 论文 v3、coding benchmark position paper v2、固定任务 scaffold 预印本，以及 OpenAI SWE-bench Verified 说明。各项来源支持的事实、来源日期、阅读范围、证据等级和边界集中记录在[本章公开合同与评估证据](../../research/coding-agent-product-comparison-evidence-2026-11-18.md)。
- 组织经验复用[软件架构案例章组织样本](../../research/software-architecture-case-study-organization-2026-11-18.md)。它支持用任务追踪对象交互的案例组织方式，不证明任何一种教学顺序带来普遍学习效果。
- 未运行 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、mini-SWE-agent 或 OpenHands；未复现实验或产品 benchmark。

候选主题与去向：

| 外部主题 | 证据支持 | 正文去向 |
|---|---|---|
| 比较单位与系统层次 | 产品文档显示不同 surface；OpenHands Canvas 可承载内建或 ACP 外部 Agent；Codex/Cursor 同时有本地和云端路径 | 13.3—13.4 定义 surface、harness、执行环境和 verifier；13.11—13.12 解释平台/Agent 与本地/远端路径 |
| Coding Agent 上下文、动作与文件变更 | Aider repo map/modes/edit formats、Cursor IDE/Cloud 文档、Claude Code 与 OpenCode 文档 | 13.9、13.13—13.15；通过认证修复任务说明对象怎样进入上下文和动作链 |
| 权限、用户审批与执行隔离 | Codex sandbox 与 approval 文档、Claude Code shell sandbox 文档、OpenCode SECURITY、Cursor runtime 配置 | 13.5—13.8、13.15；明确策略、审批、OS 隔离和下游业务授权不是同一机制 |
| 产品 lifecycle / 版本状态 | SWE-agent FAQ 表明 maintenance-only 并指向 mini-SWE-agent | 13.10 将 SWE-agent 定位为历史研究/固定版本案例，mini-SWE-agent 定位为维护者描述的后续轻量 scaffold |
| OpenHands 比较层次 | Agent Canvas、ACP agents、Agent Server、runtime/backend 文档 | 13.11 区分平台入口、内建 Agent、外部 Agent 与执行 backend |
| 基准与归因 | SWE-bench Verified、ACI 论文、benchmark position/scaffold 研究 | 13.16、13.19 区分任务验收与组件因果归因，限制可推广结论 |
| 产品选择与 harness 实验 | 多来源综合推论；不是单一来源规定的协议 | 13.17、13.19：选型时记录产品差异；因果研究时固定非研究变量并观察真实轨迹 |

本轮没有发现需增删本册章节的新主题。第十三章保留为产品比较方法章；协议主讲仍由第十四章和第二十二册承担，通用统计评估仍回链第七册与第十章。

## 原有内容去向

| 原稿内容 | 精修决定 |
|---|---|
| Codex 作为“轻量本地 CLI”的描述 | 更新为本地 CLI/IDE/desktop 与远端 Codex Cloud 的不同执行路径，要求按任务核对 workspace、sandbox、approval 和网络设置 |
| Cursor 作为 IDE-only 产品 | 更新为本地 IDE Agent 与 Cloud Agents 并存；云端 VM/runtime 配置不外推到本地 Agent |
| Claude Code 多表面能力 | 保留公开入口，补充 shell sandbox 只适用于文档覆盖的命令路径，不推断其他工具或 surface 共用隔离 |
| OpenCode 开源、权限、Server 和 SDK 特征 | 收束到第十二章固定版本证据；保留“开源不等于部署审计”，重申项目本身不提供 Agent sandbox |
| SWE-agent 当前使用状态 | 纠正为 maintenance-only；把旧论文/固定版本作为研究案例，mini-SWE-agent 作为维护者文档指向的后续轻量 scaffold |
| OpenHands 作为单一平台/Agent 的概括 | 改为区分 Agent Canvas/Server、内建 OpenHands Agent、ACP 外部 Agent 与 runtime/backend |
| 上下文、编辑、权限、评估和场景选型维度 | 保留并用共同合成任务贯通；减少无来源的产品内部推断，直接为关键动态事实加官方来源链接 |
| 13.19 多项 coverage 指标、加权综合分、固定 gate 与 Python toy audit demo | 删除；改成“产品选型 vs harness 因果比较”方法、无分数的条件记录卡、证据/未知/边界报告顺序 |
| 题库和练习中的产品排名/综合 gate | 更新练习 35Q、面试题 40O，要求按具体 surface/runtime 比较，不编造运行结果、不产出通用排名 |

## 正文状态

- **reader_path：** 合成认证修复问题 → 四层比较框架 → 产品入口地图 → Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent/mini-SWE-agent、OpenHands 的公开合同 → 任务路径 → 上下文/编辑/权限/验收 → 选型与受控比较 → 面试、练习和总结。
- **coverage_map：** 核心为比较单位、公开证据等级、本地/云端执行路径、任务上下文与动作链、permission/approval/sandbox/business authorization、benchmark 与 verifier 边界、产品选择与因果研究的差别。支撑内容为 Aider repo map/modes/edit formats、OpenHands ACP/runtime、SWE-agent 生命周期。产品内部未公开机制明确留为 unknown。
- **example_status：** `/api/profile` 认证修复为教学构造。用 `Authorization: absent → HTTP 200`、目标 401、回归测试、范围受限 diff 与不 push 的任务条件，逐步解释定位、提议、派发、执行回执、diff 检查和独立验收；表格不代表任何产品真实运行。例子不能证明产品排名。
- **evidence_status：** 动态官方/项目文档按 2026-11-18 访问日期绑定；OpenCode 版本/source 证据回链第十二章固定 revision 记录；论文与 benchmark 主张绑定对应固定版本及实验范围。新增 Aider edit format 文档已在本轮打开核对并补入 research 记录。
- **formula_status：** 不适用。删除比较覆盖率、加权聚合与通用 gate，不保留总分公式。
- **code_status：** 无可执行代码；保留两个 `text` 架构流程示意，未运行代码或产品。
- **link_status：** 本章、练习 35Q、面试题 40O、第二十册目录与 research 记录的本地 Markdown 链接静态检查缺失 0。
- **sync_status：** 已同步 `EXERCISES.md` 35Q 与 `INTERVIEW_BANK.md` 40O。目录第十三章摘要仍准确，章节数量与结构不变；未改简介。
- **open_risks：** 官方产品文档与 OpenHands `main` 分支会变化，后续引用需按访问日期复查；未做目标读者试读、GitHub 页面渲染和全站外链长期可达性检查。真实权限、网络和账户配置仍须对具体部署验证。

## 静态验收

- H2 标题 13.1—13.22 连续；Markdown fence 共 4 个标记、成 2 对。
- 五份受影响文档的本地 Markdown 链接静态检查缺失 0。
- 第十三章旧 coverage 指标、`G_compare`、coding agent comparison gate 与 Python audit demo 已移除。
- `git diff --check` 通过。
- 未确认 GitHub 页面渲染，未做目标读者试读；未运行产品、代码、测试、模型、API、MCP、插件、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

## 下一入口

[第二十册第十四章《MCP、A2A 与 Harness 集成》](../../book-20-agent-harness-runtime/chapters/14-mcp-a2a与harness集成.md)。先复核书册目录、相关进度交接，并按任务范围读取完整写作规范模块；继续分清协议层与 Runtime 的接入、权限、执行、Trace 和验收责任。
