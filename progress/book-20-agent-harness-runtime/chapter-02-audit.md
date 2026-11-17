# 第二章《Agent Runtime 架构》精修审计

审计日期：2026-11-17。状态：正文首轮精修及静态检查完成；Qwen3.7 Plus 区域/scope 能力细节已明确交给第二十二册第 47 章，待该章精修时完成迁移核对。

## 范围与读者路径

- **scope：** 完整复读并重构本章 0、2.1—2.19 节；复核原稿知识点、第二十册架构地图、章节目录摘要与第 1 章交接。章级候选检查聚焦 Runtime 状态所有权、执行/恢复边界、评估信号和 provider API 的实际责任划分。
- **reader_path：** 读者已了解 Agent 多步行动，能读简单 JSON、文件 diff、命令输出和状态表；本章沿一次失败后续跑说明 session/task/workspace/checkpoint/trace 关系，接着追踪工具 schema、模型动作、策略、执行回执和验收。预期结果是能定位失败所在层、区分 Task 状态与 Runtime phase，并按副作用是否确定选择恢复路径。
- **example_status：** 2.1 使用明确标为教学构造的登录认证修复案例。输入包括目标、限制、先前测试失败、用户预存的文档改动，以及暂停期间用户对 `src/auth.py` 的新修改；轨迹经过恢复、版本核对、上下文构造、`run_tests` 调用、退出码 1、持久化观察和下一步诊断。它说明旧 checkpoint 不能覆盖新的文件事实。2.15 另用 `v0 → v1` 说明并发 patch 的版本冲突。两例均不声称来自真实产品运行。

## 知识覆盖与旧稿去向

| 原稿知识点 | 本轮处理与正文位置 | 读者作用/边界 |
| --- | --- | --- |
| Runtime 与 Harness 的边界 | 保留并重写于 §2.2 | 按状态、控制和副作用责任区分；注明这是本书架构约定，不强加为产品模块名。 |
| UI、Session、Task、Context、Adapter、Parser、Registry、Executor、Permission、State、Trace、Error Handler、Evaluator 组件 | 合并重组于 §§2.5—2.13 | 逐个解释输入/职责/输出，并回到登录任务；避免长组件清单和重复定义。 |
| session/task/workspace/checkpoint/trace 语义 | §2.4 与 §2.10—2.11 | 将 checkpoint 与具体 SDK 的 RunState 分开；明确 session id 不提供身份认证，trace 不等于可恢复状态。 |
| 错误分类、重试和恢复 | §2.9、§2.12、§2.14—2.15 | 保留超时、取消、文件冲突、状态写入失败和未知副作用；恢复前先对账，避免盲目重试。 |
| Runtime 状态机 | §2.14 | 保留合法转移思路，新增 Task 状态与 Runtime phase 的区分；审批通过返回策略复核后再执行。 |
| 并发、取消、超时 | §2.15 | 保留只读并发、副作用并发、取消确认和部分成功；用文件版本 `v0 → v1` 补充可追踪冲突例。 |
| 可扩展性与 Provider 所有权 | §2.16 | 保留适配器、执行后端、trace sink、evaluator 扩展点；以 2026-11-17 官方文档比较 OpenAI Agents API、Agents SDK 与 Responses API，结论只适用于所述产品路线。 |
| Responses API 与 OpenAI-compatible API 的区别 | §2.7 仅保留 Adapter 需检查协议合同的通用原则；完整字段/兼容细节交第二十二册协议专题 | 避免在 Runtime 架构章复制 provider 参考内容；该知识去向须在第二十二册逐章审阅中落实。 |
| 九项运行比例与 `G_runtime` 固定门槛 | 删除公式和聚合 gate；§2.13 改为组件事件记录表并要求报告观察单位、分母，保留拒绝/失败/未知/N/A 的区别 | 原公式没有稳定任务样本、统一判定方式或可证明的普适阈值；单一比例可能掩盖越权、未知副作用和低频高严重度错误。核心观察意图保留，但不假定某个比例就是质量证明。 |
| 长 Python Runtime Step 审计 demo 及输出 | 删除；由合成过程案例、状态表、Mermaid 和 §2.13 事件账本承担讲解 | 原 demo 不属于本章必须交付的代码教程，且旧审计输出不能证明真实产品可靠性。本章未声称代码运行或 benchmark 结果。 |
| GPT-5.6 三层 Runtime 所有权快照 | 删除旧版本时效表述；用当前官方公开 API 路线作有日期案例 | 从模型版本快照改为接口责任范围；来源仍是动态文档，后续复查需按访问日期更新。 |
| Qwen3.7 Plus region/scope 能力矩阵 | 从 Runtime 通则移出，正文链到官方页面并指向第二十二册第 47 章 | 属于 provider/API 合同；目标章目前尚未完成该案例的迁入，列为本轮跨册待办。 |
| 常见误区、最小伪代码 | 将有效边界嵌入组件解释、状态机和练习，不保留独立口号清单或代码块 | Mermaid 和状态转移表足以显示本章流程；章节没有承诺运行代码实作。 |

## 研究与证据

- OpenAI Agents overview 与 Agents API overview 于 2026-11-17 复核。公开文档区分托管 Codex harness 的 Agents API、由应用内 runner 执行 loop 的 Agents SDK，以及由应用控制集成的 Responses API；Agents API 管理 session/orchestration/context compaction/recovery，应用仍提供工具并选择执行环境。该对比仅作为 OpenAI 产品案例，不推广为行业分类。证据与动态文档边界见[Agent Runtime 与 Harness 近期来源扫描](../../research/agent-runtime-and-harness-source-scan-2026-11-17.md)。
- Agents SDK Sessions 文档说明 SQLite session id 用于选择历史，不认证用户或授权访问；RunState reference 描述 SDK 自己的可序列化恢复对象。章节把二者作为不同资源举例，不据此推断外部 workspace 或业务副作用已恢复。
- OpenTelemetry GenAI span 的输入/输出敏感数据边界沿用[Agent 工具调用与安全研究记录](../../research/agent-tool-calling-security-2026.md)中 2026-11-17 的约定复核；正文将 trace 采集、脱敏、访问控制和保留作为部署责任，不声称默认配置。
- Qwen3.7 Plus 官方页面标注 2026-10-11 更新；2026-11-17 复核 Virginia Global 与 US scope 的 `Structured Outputs`、`Web Search` 支持差异。两张表的输入模态集合相同、仅显示顺序不同；已据此修正[模型来源记录](../../research/model-update-2026-09/qwen3.7-plus-source-notes.md)，不再将排序差异写成能力差异。

## 校验与配套

- **formula_status：** 本章没有保留数学公式。这里的核心是责任、状态和副作用流转，不存在需要公式化的通用数值规律；运行指标已改为带事件单位和分母的观测说明。
- **code_status：** 没有可执行代码。单个 Mermaid flowchart 的围栏成对，已静态检查节点和箭头文本语法；未在 GitHub 页面确认实际渲染。
- **link_status：** 新增的第二十二册第 47 章本地目标存在；本章外部链接中，OpenAI Agents/API、Agents SDK Sessions/RunState 和 Qwen 官方页面已打开核对。未做全站外链可达性检查。
- **sync_status：** 目录摘要仍准确描述“沿一个 step 检查上下文、模型适配、动作解析、执行、状态和 trace”，无需改动。未向共享练习/题库新增条目：章节练习已覆盖运行恢复、状态对象、并发和 provider 所有权，新增改动未引入独立学习目标。第二十二册第 47 章待后续承接 Qwen 能力矩阵。
- **静态检查：** 标题 2.1—2.19 连续；唯一 Mermaid 围栏配对；无旧的 `G_runtime`、组件比例、GPT-5.6 快照或正文维护路径；`git diff --check` 通过。未运行代码、测试、模型、API 或 benchmark；未做外部读者试读。

## 遗留项与交接

- 第二十二册第 47 章精修时，把 Qwen3.7 Plus region/scope 能力矩阵及 2026-11-17 证据边界加入 provider 对比，并关闭本条迁移待办；不得把某 scope 的支持情况外推到其他地域或部署。
- 第二十册下一章为[第三章《Coding Agent 工作流》](../../book-20-agent-harness-runtime/chapters/03-coding-agent工作流.md)，承接需求澄清、仓库探索、修改和验收；本章的 Task 状态、workspace 版本和错误恢复可作为例子边界。

## 与第一章示例的分支关系校正

- 第三章端到端复读时发现：第一章的合成任务已走到测试通过和交付，第二章开篇却称其为“重新打开的未完成会话”，若不加说明会造成时间线冲突。
- 已将第二章开篇标为“同一目标的另一条教学构造恢复分支”，明确它不是第一章轨迹的顺序续集。第二章的恢复机制、状态结果和证据边界未改变。
- 第三章会以相同任务合同重新构造完整工作流；各章路径、版本和输出仍是独立的教学构造，不表示共享同一组真实运行事件。
