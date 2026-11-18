# Agent Harness 系统设计：外部资料与证据边界

检索日期：2026-11-18  
证据状态：source-backed；本记录整理 Agent runtime 架构、持久状态、执行恢复和架构教学组织资料，不表示任一框架或架构在所有任务上更优。

## 研究范围与检索记录

研究问题：设计 Coding Agent Harness 时，哪些职责属于产品应用、Harness/runtime、执行环境、持久状态和外部副作用；任务如何穿过这些边界并留下可恢复、可验收的证据？扫描覆盖当前官方运行时/框架文档、面向 Agent 的架构指南、编排模式说明和软件架构案例教学样本。此轮不比较模型榜单，也不推断闭源产品的未公开实现。

检索平台：Web 搜索、官方文档页面及项目维护者文档。检索词包括：

- `agent harness architecture session environment lifecycle event handler official`
- `LangGraph durable execution persistence checkpoint idempotent side effects official`
- `Google ADK sessions state artifacts runner event loop deployment official`
- `agentic AI architecture design principles production evaluation safety official`
- `AI agent orchestration patterns sequential concurrent handoff architecture official`
- `agentic AI systems course architecture evaluation safety operational readiness syllabus`

范围限制：官方文档是持续更新的产品/框架合同，页面日期按访问日记录；未运行 SDK、部署样例或生产系统。UChicago MPCS 的 2026–27 Agentic AI 系统课程页面在搜索结果中提供了课程目标摘要，但正文打开返回 502，因此不把搜索摘要作为关键证据。教学组织样本沿用此前已核对的 AOSA 编者说明，见[软件架构案例章的组织样本](software-architecture-case-study-organization-2026-11-18.md)。

## 外部主题与来源支持

| 外部主题 | 直接来源支持的内容 | 证据边界 |
|---|---|---|
| 应用、Harness 与执行环境的责任划分 | OpenAI Agents API 架构把 managed harness、可选环境和 application server 分开；环境可以缺省、托管或由应用自管。自管方负责启动、重连、回收和文件持久化；事件处理器失败会阻断后续动作。 | OpenAI 当前 API 的公开合同，不是通用系统架构的唯一分层。环境、session 和 task 的生命周期必须按目标平台核验。 |
| session、turn、event 与成功判定 | OpenAI 文档区分 session 和 turn，要求检查 turn 完成/失败/取消及工具结果；session idle 本身不能证明任务成功。 | 只说明该 API 的状态语义；不能推广到所有 SDK 或产品。 |
| Session state、事件日志与 artifact 存储 | ADK 将事件历史和 session state 分开；状态持久性取决于 SessionService，状态修改通过 event append 提交。二进制 artifact 使用独立 ArtifactService，可按 session 或 user 范围管理。 | ADK 的对象与 API 语义，不规定其他 Harness 必须使用同名组件。 |
| Checkpoint、store 与外部副作用 | LangGraph 把 thread-scoped checkpoint 与跨 thread store 分开。其 fault-tolerance 文档说明 node retry 会从函数开头重跑，外部副作用位于 checkpoint 之外；稳定幂等键须在副作用前持久化，丢失回执应视为 unknown 并查询外部状态。 | 这是 LangGraph 的执行合同和应用侧重试指导；不构成 exactly-once 保证，也不保证外部服务支持幂等或查询。 |
| 编排选择与并发代价 | Azure Architecture Center 列出 sequential、concurrent、group chat、handoff、magentic 等模式，并按依赖和任务开放程度说明适用条件；它同时提示并发共享可变状态会产生一致性问题，过度编排会增加延迟与资源消耗。 | 厂商维护的设计指南；模式名不是跨框架标准，适用性仍取决于工作负载和控制需求。 |
| 生产级架构复核维度 | AWS Agentic AI Lens 把 agent-specific 设计放入运行、可靠性、安全、性能、成本和可持续性维度；其原则强调显式合同、端到端 trace、策略版本管理与按影响程度配置人类监督。Microsoft Cloud Adoption Framework 强调沿用既有平台/工作负载治理，并按用途复杂度和风险选部署路径。 | AWS/Microsoft 的指导和产品生态视角，不是法规、统一标准或对某一架构的效果证明。 |
| 软件架构案例的讲解组织 | AOSA 编者要求案例解释高层模块、模块间互动与设计原因，并说明何种经验可迁移；FAQ 建议按新开发者系统介绍组织，不逐项罗列普通实现细节。 | 证明该书的编辑选择和目标，不证明这一写法适用于所有读者或能因果提升学习效果。 |

## 来源卡

### OpenAI Agents API（官方、动态文档；访问日期 2026-11-18）

- [Architecture](https://developers.openai.com/api/docs/guides/agents-api/architecture)：说明 Harness、Environment、Application server 的边界；环境可选或托管/自管；自管环境由应用负责生命周期；streaming/webhook 与 function tool handler 分属事件和执行责任。
- [Agents API overview](https://developers.openai.com/api/docs/guides/agents-api/overview)：定义 Agent、Environment、Session、Events/items，并说明 session 持久化与托管 Harness 职责。
- [Run and continue sessions](https://developers.openai.com/api/docs/guides/agents-api/sessions)：区分 session/turn，说明 idle 不等于 turn 成功，并要求检查实际 tool result；文档建议对每个逻辑消息提交使用 idempotency key。
- [Sandbox lifecycle](https://developers.openai.com/api/docs/guides/agents-api/environments/lifecycle)：自管环境须协调重连和关闭；session 删除不会自动停止环境；进程崩溃后的 pending input 恢复不受保证；替换 compute 不会自动恢复文件。

### Google Agent Development Kit（ADK，官方文档；访问日期 2026-11-18）

- [State](https://adk.dev/sessions/state/)：区分 event history 与 session state；持久性由 session service 决定；状态变化经 event append 跟踪。
- [Event Loop](https://adk.dev/runtime/event-loop/)：解释 Runner 处理并提交事件后再恢复执行逻辑；event 中可携带 state delta 等意图。
- [Artifacts](https://adk.dev/artifacts/)：artifact 是带版本的二进制数据；其保存和检索交给独立的 artifact service。

### LangGraph（官方框架文档；访问日期 2026-11-18）

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)：区分 thread-scoped checkpointer 与跨 thread store。
- [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)：说明 checkpoint 边界和已完成 node 的 pending writes 恢复。
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)：说明重试的 at-least-once 节点执行、外部副作用的幂等键、丢失确认时将结果保留为 unknown 并查询 provider 的处理方式。

### 架构指南与编排模式（官方文档；访问日期 2026-11-18）

- [AWS Agentic AI Lens](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/)（文档修订信息显示 2026-06-10）：把 agent 运行放入完整 Well-Architected 维度；[Design principles](https://docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens/design-principles.html)强调显式合同、可追踪行为、版本化行为配置及与行动后果相称的人类监督。
- [Microsoft Azure Architecture Center: AI agent orchestration patterns](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns)（页面标记 2026-02-12 更新）：描述多种编排模式及其适用任务与风险；不建议为简单流程引入不必要的协调层。
- [Microsoft Cloud Adoption Framework: Architecture and governance for AI agents](https://learn.microsoft.com/en-us/azure/cloud-adoption-framework/ai-agents/architecture-governance)（页面标记 2026-11-14 更新）：主张沿用现有 landing-zone 治理、按工作负载和风险配置 agent 身份、权限与集中可见性。
- [The Architecture of Open Source Applications, Volume 1 Introduction](https://aosabook.org/en/intro1.html) 与 [FAQ](https://aosabook.org/en/faq.html)：由[既有组织研究记录](software-architecture-case-study-organization-2026-11-18.md)复用；只用于说明案例章的编辑组织方式，不作为技术事实来源。

## 未验证边界

- 本次资料支持的是组件责任、状态/副作用边界和架构审查维度；不提供适用于所有 Coding Agent 的唯一参考架构、固定 SLO、通用上线分数或统一组件清单。
- 厂商指南提出的风险、可靠性和成本控制仍需映射到具体任务、分母、部署环境和数据分类；不能直接当作经统一实验验证的普遍规律。
- SDK、托管服务和 API 行为未在本地复现；所有版本/功能事实应绑定目标实现和访问日期。闭源产品内部实现未作为证据。
- 没有把搜索到的课程摘要当作已读 syllabus，也没有用 AOSA 的组织建议推断学习效果。
