# Agent Runtime 与 Harness 近期来源扫描

核验日期：2026-11-17。本记录保存公开来源支持的机制、接口状态及证据边界；书册候选去向、结构决定和修改进度只记录在 `progress/`。

## Gemini Interactions API 的服务器状态与调用级配置

- **来源与版本：** Google AI for Developers 官方 [Interactions API overview](https://ai.google.dev/gemini-api/docs/interactions-overview) 与 [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)，动态文档，2026-11-17 阅读。
- **阅读位置：** Interactions 的资源模型、stateful/stateless 说明、`previous_interaction_id`、存储/保留段落；Thought signatures 的 stateful 与 stateless 小节。
- **来源支持的事实：** Interactions 文档称该 API 自 2026 年 6 月起 GA，并建议新项目使用。一个 Interaction 是一轮的会话记录，包含按时间排列的输入、thought、工具调用/结果与输出步骤。后续请求带 `previous_interaction_id` 时会恢复历史输入与输出，但 `tools`、`system_instruction` 和 `generation_config` 等参数仍属于当前请求，调用方需在每轮重新提供。默认 `store=true`；当前文档列出付费层 55 天、免费层 1 天保留。`store=false` 与后台执行及后续 `previous_interaction_id` 不兼容。stateless 调用如使用 thinking 或工具，须原样续传模型返回的 thought/function-call 步骤，以保留签名。
- **证据等级与边界：** source-backed（Google 官方 API 文档）。文档为动态页面，保留期、字段和推荐状态须按使用日期重新核对；以上是接口合同，不证明特定项目已配置删除、保存期限或权限控制，也不代表本项目进行了真实 endpoint probe。

## OpenCode 的权限提示与执行隔离

- **来源与版本：** OpenCode 官方 [`SECURITY.md`](https://github.com/anomalyco/opencode/blob/dev/SECURITY.md) 与[工具文档](https://github.com/anomalyco/opencode/blob/dev/packages/web/src/content/docs/tools.mdx)，`dev` 动态分支，2026-11-17 阅读。
- **阅读位置：** Security threat model 的 Overview/No Sandbox/Server Mode；Tools 的 permission 说明。
- **来源支持的事实：** 官方 threat model 明确表示 OpenCode 不提供 sandbox；权限系统用于用户了解和确认命令/文件等动作，不是安全隔离机制；需要真正隔离时应放入 Docker 容器或虚拟机。工具文档说明工具策略可按 allow/deny/ask 配置。
- **证据等级与边界：** source-backed（项目官方文档）。`dev` 分支会改变，以上陈述绑定访问时的公开文档，不能外推到其他产品、部署方式或未来版本；项目是否使用外层容器及其配置仍须另外审查。权限提示和环境隔离应作为两条独立控制检查。

## OpenAI Agents SDK 的 tracing 与离线测试边界

- **来源与版本：** OpenAI Agents SDK 官方动态文档：[Tracing](https://openai.github.io/openai-agents-python/tracing/)、[Configuration](https://openai.github.io/openai-agents-python/config/) 与 [Testing](https://openai.github.io/openai-agents-python/testing/)，2026-11-17 阅读。
- **阅读位置：** tracing 的 trace/span 组成和敏感数据配置；Testing 的范围声明与无网络测试说明。
- **来源支持的事实：** tracing 文档说明 SDK 可记录 workflow、模型生成、工具、guardrail、handoff 等 span，并提供禁用 tracing 与排除敏感输入/输出的配置。Testing 文档说明其 deterministic 工具可在内存中测试 SDK/应用拥有的工具调用、handoff、guardrail、重试、streaming 和 session 行为，不发出模型、sandbox-provider 或 realtime API 请求；provider、网络、sandbox provider 和音频系统拥有的行为需另做 integration testing。
- **证据等级与边界：** source-backed（SDK 官方动态文档）。文档说明公开接口能力，不证明本项目或某个部署版本已采用该设置、日志完全脱敏或 scorer 有效。使用时应固定 SDK 版本、trace processor 和数据保留策略。

## Agent loop、工具执行管线与权限控制边界

- **来源与版本：** OpenAI Agents SDK 官方动态文档：[Running agents](https://openai.github.io/openai-agents-python/running_agents/)、[Tools](https://openai.github.io/openai-agents-python/tools/)、[Human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/)；OpenCode 仓库当前 [`SECURITY.md`](https://github.com/anomalyco/opencode/blob/dev/SECURITY.md)，均于 2026-11-17 阅读。
- **阅读位置：** Agents SDK 的 Runner loop、tool-not-found 与 max-turn 行为；`FunctionTool.__wrapped__` 绕过的执行管线；approval interruption / pause-resume；OpenCode threat model 中的 No Sandbox。
- **来源支持的事实：** Agents SDK 文档将一次运行描述为反复调用模型，遇到最终输出时结束，遇到 handoff 时切换 agent，遇到工具调用时执行并将结果送回后续运行；可设置 turn 上限，未知的 function tool 可按配置报错或将错误作为模型可见输出继续循环。文档还明确，直接调用装饰器暴露的 Python 函数会绕过该 SDK 的 schema validation、context injection、guardrails、timeouts、failure handling 和 tracing。其 HITL 流程可暂停运行、等待工具审批，再用保存的 `RunState` 恢复。OpenCode 当前 threat model 明确称其 permission system 是用户知情和确认界面，不是安全隔离，并建议需要隔离时在 Docker 容器或 VM 中运行。
- **证据等级与边界：** source-backed（供应方/项目官方文档）。Agents SDK 的具体控制顺序和可选行为仅代表该 SDK 文档版本，不能当成所有 Harness 的通用实现。OpenCode `dev` 内容动态变化；它支持“权限提示不等于隔离”的设计边界例证，不代表其他产品实现相同，也不能代替对实际容器、宿主权限、网络和文件系统配置的检查。Approval 只说明获得一项用户/程序决定，不自动证明业务授权、执行隔离或任务成功。

## OpenAI Agents API、Agents SDK 与 Responses API 的运行时归属

- **来源与版本：** OpenAI 官方 [Agents overview](https://developers.openai.com/api/docs/guides/agents)、[Agents API overview](https://developers.openai.com/api/docs/guides/agents-api/overview)、[Agents SDK Sessions](https://openai.github.io/openai-agents-python/sessions/) 与 [RunState reference](https://openai.github.io/openai-agents-python/ref/run_state/)，动态文档，2026-11-17 阅读。
- **阅读位置：** Agents overview 的 Choose your starting point/runtime comparison；Agents API 概览的 managed harness 与 session；SDK Sessions 的历史保存及续接；RunState 的可序列化暂停/恢复边界。
- **来源支持的事实：** 当前 Agents overview 将 Agents API 定义为由 OpenAI 管理 Codex harness 与 session orchestration 的入口；Agents SDK 在应用中运行，应用控制 deployment、storage、approvals 和 runtime integration；Responses API 面向直接调用模型或自行构建 Agent，状态可由应用历史、response chaining 或 Conversations 管理。表格还将执行环境区分为 OpenAI-hosted/self-hosted/no sandbox、应用内 runtime/sandbox integrations，以及 Responses 应用自己的环境。Agents API 文档说明托管服务可保存 session、turns 和 items，并提供 orchestration、context compaction 与 recovery；SDK 的 Session 保存多轮会话历史，`RunState` 则为暂停/审批后的单次运行序列化恢复快照，二者不是同一资源。
- **证据等级与边界：** source-backed（OpenAI 当前公开产品/API 文档）。这是访问日对三种 OpenAI 接入路线的公开比较，不是所有厂商的通用分类，也不代表同名 session、conversation、run state、sandbox、filesystem artifact 或业务任务状态彼此等价。具体持久化内容、删除、数据控制和模型能力应按当前接口文档与实际部署核验。

## 书册发现扫描的主题判断

- 最新资料继续支持把 Agent 结果绑定到 model、harness、environment、budget、任务分布和 verifier，而不是报告裸模型成绩；基准版本、初始环境、重试与最终 artifact 校验都会改变证据范围。详情见[Benchmark 生命周期、评测器偏差与 Agent Harness 评估来源](evaluation-benchmark-lifecycle-and-agent-harness-2026.md)及[Code Agent 评测来源](code-agent-benchmarks-2026-11.md)。
- 服务器托管的对话历史、持久 workspace、任务 checkpoint、工具执行事实和业务真值是不同状态所有者；当前 Interactions 文档只规定 API 资源和续传语义，不把 API history 变成完整业务 checkpoint。
- 工具可见性、用户确认、权限判断、进程/文件系统隔离、业务授权与审计是不同控制层。OpenCode 文档明确提供了一个“权限界面不等于 sandbox”的公开案例；不能据此推断所有 coding agents 的实现。
- 上述材料不支持新增独立章节。它们适合作为第 8、9、10、15、17—24 章的更新/连接候选，具体正文去向与取舍记录在对应 progress 文件。
