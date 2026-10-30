# Agent 工具调用、安全确认与评测：来源证据

检索与阅读日期：2026-10-30。本文只记录公开资料支持的接口约束、授权建议和评测构造；产品接口与仓库文档为动态页面时，以访问日期和具体 API 模式限定适用范围。

## 工具协议与 schema

- **OpenAI Function Calling**：[官方文档](https://developers.openai.com/api/docs/guides/function-calling)，访问日期 2026-10-30；阅读 tool-call/result 往返、call id、parallel function calls 和 strict mode 部分。
- **来源支持的事实**：API 响应可包含零个、一个或多个函数调用；调用对象带有供结果回传关联的 id。strict mode 要求每个 parameters 对象把 `additionalProperties` 设为 `false`，并把 `properties` 中的所有字段列入 `required`；可选字段通过将 `null` 加入类型表示。文档区分 Responses 与 Chat Completions 的默认行为，因此不能把 strict mode 规则外推为所有 API、供应商或调用模式的统一契约。
- **证据边界**：此页面描述 OpenAI 当前公开 API；不证明任意模型都能选对工具，也不表示 schema 通过就满足业务授权。

- **Anthropic Tool Use**：[How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)，访问日期 2026-10-30；阅读 client-executed 与 server-executed tool 的执行责任说明。
- **来源支持的事实**：用户定义的工具由应用提供 schema、执行返回的 `tool_use`，再将结果作为 `tool_result` 回传；若使用服务器执行的工具，应用的执行/回传责任不同。供应商定义的工具协议不应被视为跨平台通用调用对象。
- **证据边界**：页面属于动态产品文档；实际字段、执行方和循环行为须按 API 与工具类型确认。

## 行动—观察轨迹与 Agent 评估

- **ReAct**：[原论文](https://arxiv.org/abs/2210.03629)，2022 年论文页面，访问日期 2026-10-30；阅读摘要及方法说明。
- **来源支持的事实**：论文研究模型交错生成任务相关推理轨迹与行动，并通过外部知识库或环境取得观察；方法案例采用多步 thought/action/observation 轨迹。
- **证据边界**：该论文支持研究方法的定义，不规定现代服务 API、生产 trace 格式或应公开何种内部推理文本。

- **OpenAI Agent Evals**：[Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)，访问日期 2026-10-30；阅读 trace grading 和 datasets/eval runs 部分。
- **来源支持的事实**：文档将 trace 描述为一次 workflow 中模型调用、工具调用、guardrails 和 handoffs 的端到端记录，并建议在需要可重复对比时使用数据集和 eval runs。
- **证据边界**：该文档是动态产品指南；产品评测工具、保留策略与可用功能可能变化，不能据此声称某个项目的 trace 天然安全。

- **AgentDojo**：[论文](https://arxiv.org/abs/2406.13352)，2024；阅读摘要。
- **来源支持的事实**：该评估环境研究工具返回不可信外部内容时的间接提示注入，包含任务、攻击与防御的可扩展评测设计。
- **证据边界**：论文设置和其攻击成功率不等于其他模型、工具、权限和部署环境的安全结果。

- **ToolSandbox**：[论文](https://arxiv.org/abs/2408.04682)，2024；阅读设计与评估部分。
- **来源支持的事实**：该 benchmark 支持有状态工具执行和对话式多步任务；以轨迹中的 milestones 与不得发生的 minefields 检查任务过程，覆盖工具状态依赖、并行调用顺序和信息不足场景。
- **证据边界**：benchmark 分数只适用于其模型、场景、工具和匹配规则；只能借用测试构造，不应横向当作本地 Agent 的成功率或安全率。

## 高风险动作授权与日志

- **OWASP Transaction Authorization Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)，访问日期 2026-10-30；阅读交易数据展示、服务端强制、授权状态转换、一次性使用、有效期和执行前检查部分。
- **来源支持的事实**：指南要求用户能识别并确认关键交易数据，授权在服务端实施，交易数据在授权期间不得被篡改，授权凭据应限时、按操作唯一且执行前在共享状态中原子检查/消费。
- **证据等级与边界**：source-backed（OWASP 开发安全指南）。将这些要求组合成 Agent 的服务端 `approval_id` 与动作摘要记录，是本项目的工程化设计推论，不是 OWASP 规定的专用 Agent 协议。

- **OWASP Logging Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)，访问日期 2026-10-30；阅读需要移除、遮蔽、散列或加密的日志字段，以及事件关联和访问控制部分。
- **来源支持的事实**：日志字段应由采集目的决定；访问令牌、密码、密钥、敏感个人数据、支付信息和高等级数据通常不应直接写入日志，需要删除、遮蔽、散列或加密，并限制日志访问。
- **证据边界**：指南不能代替具体系统的数据分类、授权、保留期限和审计设计；简单字段名单无法发现嵌在自由文本中的敏感信息。

- **OpenTelemetry GenAI spans**：[语义约定页面](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，`main` 分支动态页面，访问日期 2026-10-30；阅读工具参数/结果属性与 buffered content capture 部分。
- **来源支持的事实**：GenAI 工具参数和结果字段可能包含敏感信息；完整系统指令、用户消息和模型输出通常既敏感又较大，当前约定建议默认不采集内容，按明确配置选择采集或存外部并在 span 记录引用。
- **证据边界**：语义约定处于持续演进状态，且不会自动实施脱敏、访问控制、留存或合规保障。实现时还须固定具体语义约定和 SDK 版本。
