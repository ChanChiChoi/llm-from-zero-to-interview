# Agent 产品化与控制资料（2026-11）

## OWASP Top 10 for Agentic Applications 2026

- 来源：[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/download/52117/?tmstv=1765059207)，OWASP GenAI Security Project，版本 2026，发布于 2025-12；配套资源页标注 2025-12-09。
- 阅读位置：PDF 导言第 5—6 页、风险目录及 ASI01—ASI10 条目。
- 来源支持的事实：该版将 Agent 风险拆为目标劫持、工具滥用、身份和权限滥用、供应链、意外代码执行、记忆/上下文污染、Agent 间通信、级联故障、人机信任利用、失控 Agent；导言建议最小权限、避免不必要自主性（least agency）并保持可观察性。ASI09 将自动化偏差、对未经核验的理由过度信任、泛化确认不足列为风险模式。
- 证据等级：source-backed，社区专家审阅的风险分类和缓解建议。
- 边界：风险 taxonomy 不是认证、特定产品评估或通用概率估计；不能据分类名称推断某个具体系统必然存在相应漏洞。

## Securing Agentic Applications V2

- 来源：[Securing Agentic Applications V2: A Builder & Defender's Guide](https://genai.owasp.org/download/58383/?tmstv=1794198262)，OWASP GenAI Security Project，Version 2.0，PDF 首页标注 2026-10；资源页日期为 2026-11-10。
- 阅读位置：第 2 章架构与身份/委派；第 3.1.5 节人审、第 3.1.7 节输入/动作两道校验；第 3.3.8 节行为熔断；第 6.4.4 节工具描述与返回值；第 7 章 Assurance。
- 来源支持的事实：该指南区分不可信上下文进入模型前的 admission validation 与副作用执行前的 action validation；建议按动作影响设计人审、将权限/范围校验放在模型外、记录运行轨迹，并为异常设置限速、暂停、只读降级或停止等控制。工具描述、Schema 和返回值都可能成为攻击面。撤销、事务边界、幂等和补偿应结合具体外部系统设计。
- 证据等级：source-backed，OWASP 社区技术安全指南。
- 边界：指南聚焦技术安全实践，不是法规、产品认证或某供应商 SDK 的能力保证；指南中的控制需结合威胁模型和目标系统验证。

## OpenAI Agents SDK：人工审批与状态恢复

- 来源：[Human in the loop - OpenAI Agents SDK](https://openai.github.io/openai-agents-python/human_in_the_loop/)，在线 Python SDK 文档，检索日期 2026-11-16。
- 阅读位置：Approval flow、long-running approvals、recovering after a crash。
- 来源支持的事实：SDK 工具可要求审批并让 run 暂停，之后通过 RunState 恢复；序列化运行状态可能包含工具参数、待执行调用、应用 context 和审批决定，必须从可信服务端存储读取并重新鉴别/授权审批人；批准动作并不自动保证外部副作用和运行状态原子提交或 exactly-once，文档建议使用目标系统幂等键，无法幂等时先对账，结果仍不确定则保持 unknown、避免自动重放。
- 证据等级：source-backed，特定版本/具体 SDK 的官方使用文档。
- 边界：只证明该 SDK 文档中所述 API 合同，不代表所有 Agent runtime、数据库或外部工具都提供相同行为；幂等键的效果取决于目标服务的原子实现和保留期。

## OpenAI Agents SDK：运行轨迹

- 来源：[Integrations and observability - OpenAI API](https://developers.openai.com/api/docs/guides/agents/integrations-observability)，在线官方文档，检索日期 2026-11-16。
- 阅读位置：Tracing 小节。
- 来源支持的事实：文档描述 SDK 在常规服务端路径可输出结构化 trace，包含模型调用、工具调用、handoff、guardrail 和自定义 span。
- 证据等级：source-backed，特定 SDK/官方平台能力描述。
- 边界：该能力描述不等于所有 Agent 系统默认具备审计；产品仍需验证字段完整性、保留期限、访问控制、脱敏和外部系统日志关联。

## Anthropic：workflow 与 Agent 的复杂度选择

- 来源：[Building effective agents](https://www.anthropic.com/research/building-effective-agents)，Anthropic Engineering，发布日期 2024-12-19；页面当前明确提示原文中不少工具信息已变化。
- 阅读位置：What are agents?、When (and when not) to use agents、Building blocks/workflows/agents。
- 来源支持的事实：该文将 workflow 概括为预定义代码路径下编排 LLM/工具，将 Agent 概括为模型动态决定过程和工具使用；建议从最简单方案开始，workflow 更适合定义清楚的任务，Agent 在需要灵活性和模型驱动选择时才有理由增加复杂度，并指出复杂度可能增加延迟和成本。
- 证据等级：source-backed，厂商工程经验文章。
- 边界：属于 2024 年工程建议，文章现已提示工具生态变化；不是跨方案控制实验，不支持当前框架能力、市场成熟度或产品收益的普遍结论。
