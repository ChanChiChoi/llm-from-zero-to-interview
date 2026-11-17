# 项目交付与事故治理来源（2026-11-17）

记录日期：2026-11-17。本文只记录本次联网核验的来源事实、版本与证据边界；书册审阅、采纳决定和正文修改只记在 `progress/`。

## Google SRE 事件响应

- **来源：** Google SRE [Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/)、[Managing Incidents](https://sre.google/sre-book/managing-incidents/) 和 [Incident Response Workbook](https://sre.google/workbook/incident-response/)。核验日期：2026-11-17；阅读事件响应角色、协调、沟通、缓解、交接和复盘部分。
- **来源支持的事实：** Google 的资料强调明确事故角色、协调多个团队、维护事件状态文档、清楚交接指挥权，并持续向受影响用户和相关方说明影响与缓解情况。事件结束后的复盘用于记录过程、影响、响应中做得好或需改进之处，并将纠正行动纳入后续工作。
- **证据边界：** 这些资料记录 Google SRE 的事件响应实践，不是统一法规、跨组织强制流程或生成式 AI 质量标准。移植到 LLM 产品时，团队仍需定义自己的权限、模型/Prompt/检索版本、工具副作用和用户任务证据。
- **证据等级：** 实践组织发布的官方指南与工程手册；支持其明确描述的流程和角色。

## DORA 软件交付性能指标

- **来源：** DORA [software delivery performance metrics](https://dora.dev/guides/dora-metrics/) 和 [metrics history](https://dora.dev/insights/dora-metrics-history/)。核验日期：2026-11-17；当前指标页面称模型从早期四项指标演进为五项。
- **来源支持的事实：** 当前页面列出五项指标：change lead time、deployment frequency、failed deployment recovery time、change fail rate、deployment rework rate；并将其分为 software delivery throughput 与 instability 两组。演变页面说明 deployment rework rate 是后续增加的第五项，旧页面/旧报告中的四项模型不能自动代表当前完整指标集合。
- **证据边界：** DORA 指标关注软件交付过程。它们不能替代模型/任务质量、访问控制、用户价值或具体 AI 项目的发布合同；不同页面和年度报告须按其当时定义及样本范围引用。
- **证据等级：** DORA 项目官网研究与指南；支持指标名称、分组和其页面公开的演进说明。

## NIST AI RMF 与生成式 AI Profile

- **来源：** NIST [AI Risk Management Framework 页面](https://www.nist.gov/itl/ai-risk-management-framework) 与 [NIST AI 600-1](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)。核验日期：2026-11-17；AI 600-1 于 2024-07-26 发布。
- **来源支持的事实：** NIST 将 AI RMF 描述为自愿使用的风险管理框架；截至核验日，官方页面称 AI RMF 1.0 正在修订。AI 600-1 是生成式 AI profile，提供组织可按自身目标与优先级考虑的风险管理行动。
- **证据边界：** 框架不是法规、认证或某个系统已安全的证明；修订状态以核验日期为准。GDPR 及其他法律适用性事实见[安全数据生命周期与治理来源记录](security-data-lifecycle-and-governance-sources-2026-11-17.md)。
- **证据等级：** NIST 官方项目页与正式出版物；支持框架定位、版本和发布状态。

## RACI 责任分配

- **来源：** Project Management Institute [Stakeholder Management and RACI](https://www.pmi.org/blog/stakeholder-management-raci)。核验日期：2026-11-17；阅读 RACI 角色和人机协作治理说明。
- **来源支持的事实：** PMI 将 RACI 描述为任务/决定的角色矩阵：Responsible 执行工作，Accountable 对结果负责，Consulted 在工作完成前提供意见，Informed 接收进展或决定；该页面建议每项任务由一个 Accountable 负责，Responsible 可以多人。它强调人在 AI 支持流程中仍承担关系、审批和结果责任。
- **证据边界：** RACI 是责任沟通工具，不自动决定技术事实、法律责任、权限和发布结果。不同组织可采用不同责任矩阵；必须绑定具体交付物、决定权限和证据。
- **证据等级：** PMI 官方实践文章；支持其对 RACI 角色和人机责任边界的说明，不构成法规或强制标准。

## 未核验事项

- 未研究任何具体组织的项目治理、事故响应、RACI、发布审批或 DORA 数据。
- 未验证本章合成账本中的任何运行数据；其中任务结果、延迟和成本仅用于教学算术，不代表生产基线或通用阈值。
- 未验证任何具体团队采用 DORA 指标后产生因果改善；指标相关性或研究结论不能直接转成单项目保证。
