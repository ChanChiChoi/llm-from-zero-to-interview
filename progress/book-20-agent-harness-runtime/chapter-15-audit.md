# 第二十册第十五章精修审计

日期：2026-11-18  
章节：[第十五章《Agent Harness 系统设计》](../../book-20-agent-harness-runtime/chapters/15-agent-harness系统设计.md)  
状态：章级扫描与正文精修完成；第二十册第 16—24 章及全套书精修 goal 继续 active。

## 章节契约与读者路径

本章承接第 1—14 章已经建立的 Runtime、工作流、工具、文件/终端、上下文、权限/沙箱、Trace/Evaluation、产品案例和 MCP/A2A 集成知识。它回答：怎样从一个有明确限制的 Coding Agent 任务出发，设计出责任清楚、状态可恢复、动作受控、结果能验收并能按证据逐步放量的系统？

用一项明确标为教学构造的登录错误修复任务贯穿设计。读者依次确定任务契约与范围、分配确定性工作流和开放推理、画责任与数据流、区分 Session/Task/Run/Workspace/Artifact/外部副作用、追踪一次请求、设计策略与隔离、核对重试和取消，再说明评估、部署与容量取舍。离开本章时，读者应能提交一份包含读者目标、组件责任、状态所有权、故障恢复和验收证据的系统设计答卷。

## 章级扫描与候选去向

检索日期：2026-11-18。问题：跨框架和管理型 Runtime 中，应用、Harness、执行环境、任务状态、artifact 与外部副作用由谁拥有；生产 Agent 架构指南和系统设计教学样本如何组织这些边界？检查了 OpenAI Agents API 架构/session/environment 文档、Google ADK state/event/artifact 文档、LangGraph persistence/fault-tolerance 文档、AWS Agentic AI Lens、Microsoft Azure 编排与治理指南，以及既有 AOSA 软件架构案例组织样本。新知识与来源限制见[Agent Harness 系统设计外部资料记录](../../research/agent-harness-system-design-evidence-2026-11-18.md)。课程页搜索摘要因正文请求返回 502 未作为关键证据。

| 外部主题 | 读者结果 | 正文去向 | 证据及边界 |
|---|---|---|---|
| 任务需求与系统选择 | 把产品形式、目标、动作范围、验收、数据与负载要求转成系统契约。 | 15.1—15.3 | AWS/Microsoft 架构指南与本章教学构造；厂商建议不能自动成为本项目需求。 |
| 应用、Harness、Runtime 与执行环境的责任 | 能画出任务接收、调度、模型循环、执行器、存储和调用者之间的数据/控制关系。 | 15.4、15.6、15.11 | OpenAI 当前 Agents API 对托管 Harness、Application Server 与可选/自管环境的公开职责；只作为实例。 |
| Session、Task、Run、Workspace、Artifact 与 task state | 不把对话历史、业务状态、运行尝试、文件事实和交付物混为一体。 | 15.5—15.6 | OpenAI session/turn，ADK event/session state/artifact service，LangGraph checkpoint/store；各框架名词不直接互换。 |
| 编排、上下文、模型适配和能力目录 | 让模型处理开放判断，固定阶段、边界、预算和终止条件由 Runtime/程序控制。 | 15.3、15.7 | Azure 的编排模式与风险提示；不主张 multi-agent 默认更优，也不把 API adapter 说成语义完全兼容。 |
| 权限、确认、执行隔离与任务验收 | 区分身份授权、任务策略、人审、执行限制和 verifier 各自证明什么。 | 15.8 | AWS Agentic AI Lens 的显式合同/可观察性/人类监督建议，结合本册第 6、8、14 章具体机制。 |
| Checkpoint、重试、取消和外部副作用 | 在恢复前先确定真实副作用状态；丢失回执时保留 unknown。 | 15.5、15.9 | LangGraph 当前重试和幂等语义、OpenAI managed/self-hosted lifecycle；均不是跨系统 exactly-once 保证。 |
| Trace、Evaluation、部署与容量 | 以版本化任务和分项结果评估候选设计；按实际峰值估运行进程、请求和保留空间。 | 15.10—15.12 | AWS/Microsoft 架构指南及明确标为合成的算术估算；示例数不代表实际流量或推荐配置。 |
| 软件架构案例教学组织 | 从一条系统案例说明高层模块、交互和设计原因，而非罗列普通实现细节。 | 全章叙事结构 | AOSA 编者的编辑目标；能证明其组织选择，不能证明普遍学习效果。 |

外部扫描没有发现需要从本章拆出的独立主题。长任务 checkpoint 和状态恢复的深入机制由第 17 章主讲；持久 workspace 与状态所有权由第 21 章主讲；协议语义由第 22 册主讲。本章只在综合架构中定位接口和责任。第 16 章继续负责按故障现象排查，不重复本章的端到端设计流程。

## 原稿覆盖与本轮取舍

| 原稿内容 | 本轮处理 |
|---|---|
| 功能需求、非功能需求、CLI/IDE/Web/API 入口和生产约束 | 收入任务契约表与部署选择；避免前后重复列举同一类要求。 |
| 高层架构的大型模块清单 | 改为控制、模型/能力、执行、证据四类责任，并用一个 Mermaid 图显示交互；方框代表职责，不暗示必须拆成独立微服务。 |
| Session、Message、ToolCall、PermissionRequest、FileChange、TraceEvent、EvalRun 数据字段 | 改为说明 Session、Task、Run、Workspace、Artifact、Approval、Effect Record 的事实来源与证明边界；Message/ToolCall/TraceEvent 作为事件和尝试数据，不复制成无调用关系的字段目录。字段深度链接第 2、4、8、9、17、21 章。 |
| Context Builder、Orchestrator、Model Adapter、Capability Registry | 保留职责，整合为一次任务中的输入构造、能力投影、provider 适配和有界循环；明确 context 取舍、provider 差异和模型提议不拥有状态。 |
| Permission Engine、Sandbox、File Engine、Diff/Revert 和 MCP/A2A | 保留系统层控制关系；重点讲身份、策略、确认、执行限制、远端连接和 verifier 的证明边界。工具/文件/终端/协议实现细节继续由第 4—8、14、22 章负责。 |
| Trace、Replay 和 Evaluation 指标目录 | 保留每次运行版本记录、数据最小化、固定任务/环境/验收器及错误分类；把统计深度留给第 7 册和本册第 9、10、19 章。 |
| Server、SDK、多客户端、企业治理和分阶段落地 | 整合为本地 CLI、团队自托管和管理型 Harness 的责任表，以及按风险逐步扩展能力的发布路径。 |
| 独立威胁分类清单 | 把注入文本、密钥/生产数据、workspace/远端写入和扩展连接的威胁落在一次任务的边界与决策中；威胁建模细节交第 8、14、16 章。 |
| 12 项覆盖公式、聚合 `G_system` 上线 gate 和长 Python audit demo | 删除。覆盖率和组件出现与否不能证明接口可用、状态正确或任务安全；改为逐任务查看合同、实际派发、回执、恢复和 verifier。 |
| 五个重复系统设计问答和 8 个练习 | 改为四步面试回答与一个端到端口述示例；练习 35S 替换为任务契约、架构数据流、状态所有权、崩溃恢复和固定评估设计。第 16 章保留故障排查与高频问答。 |

## 正文状态

- **reader_path：** 合成登录故障 → 任务契约 → 工作流与 Agent 分工 → 责任图 → 状态/数据所有权 → 一次请求逐步追踪 → context/model/capability → 策略、确认、隔离、验收 → 崩溃/丢失回执/取消恢复 → Trace/Evaluation/发布 → 部署形态和容量估算 → 取舍与面试回答。
- **coverage_map：** 核心为需求/验收、架构责任与调用路径、任务与运行状态所有权、受控副作用和恢复、评估与部署决策。支撑为 context budget、provider adapter/能力目录、容量粗估、多客户端与扩展连接。细粒度工具/权限/协议/长任务 checkpoint 机制由前后章节完整主讲，本章给系统设计所需的最小回顾与入口。
- **example_status：** 一项标为教学构造的登录错误修复贯穿契约、受限读写与测试、用户禁止动作、verifier 和中断情景。容量假设明确标为教学估算：24 tasks/hour、5 worker-minutes/task 得到平均 2 个执行槽位；每 task 6 次模型调用估成 144 calls/hour；50 份/day × 200 KB × 30 days 得约 300 MB 基础 artifact 数据。文中明确这是平均或算术估算，不是运行测量或容量建议；突发、长尾、环境开销、token 分布和 retention 副本仍须按部署数据测量。
- **evidence_status：** OpenAI Agents API 的责任边界、session/turn/lifecycle 行为，Google ADK state/event/artifact service，LangGraph checkpoint/retry/idempotency 语义，AWS 和 Microsoft 架构建议分别绑定官方页面及 2026-11-18 访问日期。教学结构引用既有 AOSA 样本，不把编辑建议说成学习效果证据。未验证 SDK/服务部署兼容性。
- **formula_status：** 不设质量总分或模块覆盖率公式；保留 3 个单位明确的乘除算术示例并逐项解释输入与假设。MathJax 使用 inline delimiters，变量/单位由邻近文字定义。该估算只说明容量提问方法，不推导通用 worker 数或费用。
- **code_status：** 无可执行代码；有 1 个 `text` 任务契约示意块和 1 个 Mermaid 架构图。未运行 Agent、模型、SDK、API 或系统；未把教学事件写成真实运行结果。
- **link_status：** 正文内的 OpenAI、Google ADK、LangGraph、AWS、Microsoft 来源均为本次实际打开核对的官方文档；章节链接第 4、6—8、14、17、21 章及第二十二册目录。正文、研究记录、章节审计、架构审计和目录共检查 51 个相对 Markdown 目标，缺失 0。
- **sync_status：** 已更新目录第 15 章职责、练习 35S、面试题 40C 与 40Q；架构地图下一入口由第 15 章改为第 16 章。简介无逐章摘要，不需要改简介。未改 `plan.md` 或 PDF。
- **open_risks：** 未进行目标读者试读；Mermaid 未在 GitHub 页面确认渲染；容量输入是教学构造，需要目标部署自己的峰值、长尾、token、artifact/trace 保留和 provider 限额数据；没有运行任何代码或外部服务。

## 静态验收

- H2 标题 15.1—15.16 连续；全文 4 个代码围栏标记、两组配对围栏，分别用于 task contract 和 Mermaid；6 个 inline math 定界符成对；旧 coverage 公式、`G_system`、Python audit demo 和失败 gate 字段搜索无命中。
- 正文、研究记录、章节审计、架构审计和目录的 51 个相对 Markdown 目标缺失 0；`git diff --check` 通过。
- 未做 Mermaid 页面渲染、目标读者试读、模型/API/SDK/provider/真实 sandbox/benchmark 测试或 PDF 构建；未更新 PDF 或 `plan.md`。

## 下一入口

[第二十册第十六章《Harness 实战坑与面试题》](../../book-20-agent-harness-runtime/chapters/16-harness实战坑与面试题.md)。继续从章级外部扫描和原稿覆盖图开始，重点核对故障分类、证据与面试答题是否仍带通用覆盖 gate 或 audit demo；第十五章已提供设计方案，下一章负责让读者根据故障现象验证方案。

## 连续复读补充（2026-11-19）

- 沿第十四章的协议边界完整复读本章，并对照第十六章故障诊断、第十七章 checkpoint 和第二十一章 workspace 的分工。简介/目录职责、阅读顺序与既有研究候选去向仍成立。
- 修正三个容易影响运行时实现的表达断点：明确本章 `Task` 是本地业务任务，不是 MCP/A2A 协议任务；将任务接收重试写成按调用者/租户范围持久化的稳定幂等键映射，明确传输层请求 ID 只关联请求与响应，不自动保证幂等；在架构图中补出单任务 Task Verifier，并与离线 Evaluation Runner 分开，展示验收条件和 Artifact 如何决定本地 Task 终态。
- 同步架构审计的连续复读状态，并合并其中重复的第十五章首轮精修条目。
- 复用 2026-11-18 固定的 OpenAI Agents API、Google ADK、LangGraph、AWS/Microsoft 架构文档及 AOSA 组织样本；本轮未扩展研究范围，未运行任何代码、SDK 或外部系统。
- 本轮静态验收结果在全套连续复读记录中汇总：章节 H2 15.1—15.16 连续，4 个 Markdown 围栏标记成 2 对，6 个行内公式定界符成对；改动文档本地 Markdown 链接目标缺失 0；`git diff --check` 通过。未核对 GitHub Mermaid 页面渲染或进行目标读者试读。
- 连续复读进度更新为第 15/24 章，下一篇为第十六章《Harness 实战坑与面试题》。
