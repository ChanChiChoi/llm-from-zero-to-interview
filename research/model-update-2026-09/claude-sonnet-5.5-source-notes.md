# Claude Sonnet 5.5：thinking state、fallback API 合同与发布方证据

资料核对日期：2026-11-19。本文记录 Anthropic 对 Sonnet 5.5 的 Messages API 状态回放、工具选择、拒绝类别、fallback 和发布方安全评估的公开说明；不把官方文档写成真实 endpoint 探测，也不把发布方评测改写为独立复现或生产安全率。System Card 的旧 benchmark 数字和风险说明来源于 2026-11-03 的材料核对，第二十册第 24 章不再复述这些 benchmark 数字。

## 官方来源

- [Claude Sonnet 5.5 System Card](https://www.anthropic.com/claude-sonnet-5-5-system-card)：模型能力、安全评估、风险判断和发布方 safeguards 的主要来源。评测结果绑定卡片所述的 snapshot、任务、工具、harness、effort、权限、safeguards 与 fallback。
- [Introducing Claude Sonnet 5.5](https://www.anthropic.com/claude-sonnet-5-5)：发布方称常规软件开发中的找错和修复可以继续；较高风险 cyber 请求会 visibly fall back 到 Sonnet 5。生物 safeguards 与 Sonnet 5 相同；Sonnet 5.5 还加入 reasoning-extraction safeguards。
- [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)：Anthropic API 拒绝字段、当前类别推荐、fallback 配置、响应轨迹与计费合同。页面检索日期 2026-11-03。
- [Claude Sonnet 5.5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)：当前 API refusal 类别及 `between_tools` 请求转 Sonnet 5 时的接口约束。
- [What's new in Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5)：model ID、thinking 与 effort、强制工具选择、跨模型状态和 beta 功能迁移合同。
- [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking) 与 [Prefix checks on thinking blocks](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/prefix-checks)：typed block 回放、模型/账号/前缀绑定、兼容方向及不匹配时的响应。
- [Prompting Claude Sonnet 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5)：工具间进度块与默认显示行为。
- [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)：模型工具请求、宿主执行与结果回传的调用流程。
- [Claude Cookbook：Extended thinking with tool use](https://platform.claude.com/cookbook/extended-thinking-extended-thinking-with-tool-use)：工具回合 block 回传建议；其账号创建日期默认值说明与当前 Platform 前缀检查指南的描述不一致，见下文。
- [Diátaxis](https://www.diataxis.fr/)：区分教程、操作指南、参考与解释四种文档任务；这是文档组织框架，不是某种章节编排能改善学习效果的研究结论。
- [Anthropic Transparency Hub](https://www.anthropic.com/transparency)：提供 System Card 的公开摘要，不等同于完整 System Card。

## 2026-11-19 核验的 API/runtime 合同

- 官方 Claude API model ID 为 `claude-sonnet-5-5`；adaptive thinking 默认开启，Claude API 默认 effort 为 `high`。Anthropic 说明 Sonnet 5.5 对 effort 行为重新校准，迁移时应重跑 workload-specific effort sweep；effort 是工作深度控制，不是 `max_tokens` 的硬上限。
- Sonnet 5.5 不接受 `thinking.type=disabled`；`thinking.type=between_tools` 可在 `low`、`medium`、`high` effort 下关闭 upfront thinking，`xhigh`/`max` 会返回 400。该模式不接受 `display`、`budget_tokens` 或 `block_binding` 等额外 thinking 字段；不使用工具时会直接输出文本。
- `between_tools` 仍可能返回工具间进度 block。默认 `display="omitted"` 会使普通 thinking block 的可读文本为空；这与 `redacted_thinking` 是不同 block 状态。将 assistant 工具回合回传时，需按原顺序保留 `thinking` 和 `redacted_thinking` 等协议块；只渲染 `text` 的 UI 可能看起来没有进展。
- Sonnet 5.5 不支持 `tool_choice` 为 `any` 或指定 `tool`；`auto` 与 `none` 可用。严格 schema 限制参数形状，不替代宿主的授权、执行和副作用记录。
- client-tool 回合在 `stop_reason="tool_use"` 下返回一个或多个 `tool_use` block；`tool_use.id` 用来与后续 `tool_result.tool_use_id` 配对，宿主执行 client tools。Anthropic server tools 在供应商侧运行，不能把两者都写成应用宿主执行。
- thinking block 的模型兼容具有方向性和版本条件。当前 Preserved thinking 页的 Claude API 兼容表显示：Sonnet 5.5 生成的 blocks 可由 Fable 5.2、Mythos 5.2 和 Opus 5.5 读取；Fable 5.1/Mythos 5.1 生成的 blocks 对 Sonnet 5.5 标为 limited support。不能把兼容性概括为“Sonnet 5.5 不读取 Fable/Mythos 系列”。该表限定于 Claude API；切换到 Managed Agents 或其他云平台，应重新核验对应 surface 的模型方向与账户条件。不可读 block 可被静默丢弃，不能仅依赖请求成功判断状态完整。
- Sonnet 5.5 的 thinking blocks 仅在生成它的账户或关联账户中可用；其他账户的请求可能成功但丢弃 block。文档列出的特定 surface 可通过 `input_transformations` 报告 `organization_binding_mismatch`。
- 前缀检查覆盖生成该 block 前的 `system`、`tools` 和所有旧消息。当前 Platform 指南把 Sonnet 5.5 的未显式配置行为标为“可能 400”，并要求应用准备好 400 与无错误但 mismatch block 到达模型两种结果；`thinking-binding-controls-2026-08-01` 允许请求显式设置 `error` 或 `drop_block`，但该控制字段不应与 `between_tools` 混用。
- **官方文档默认值描述有冲突：**当前 [Platform Prefix checks 指南](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/prefix-checks)说明未设置 `prefix_mismatch_behavior` 时的结果取决于模型，并将 Sonnet 5.5 标为可能 400；官方 [Cookbook](https://platform.claude.com/cookbook/extended-thinking-extended-thinking-with-tool-use)仍称 2026-08-31 起创建的账户默认返回 400、旧账户通过 beta opt in。不能把这两页合并成普遍账户阈值；目标 provider/account 的实际默认行为仍需在对应环境确认。
- Anthropic API 的拒答正常返回 HTTP 200，使用 `stop_reason="refusal"` 及 `stop_details.category`；类别可以缺失。Sonnet 5.5 的 server-side `fallbacks="default"` 当前仅在 Claude API 和 Anthropic 管理的 Google Cloud beta surface 上适用，`cyber` 与 `frontier_llm` 可路由至 Sonnet 5，`bio`、`reasoning_extraction` 与 `general_harms` 不走该默认 fallback。请求模型的 429、overload 或 server error 不触发该 safety fallback。
- 成功 fallback 的返回记录包括实际服务模型、`fallback` block 和 `usage.iterations`。流式输出可以中途切换；仅记录最初的 `message_start` 会漏掉实际 fallback 模型。请求方仍需独立核对工具执行回执与任务完成条件。
- Anthropic 官方文档还说明，若流式 refusal 到达时任意 tool-use block 仍打开，该请求直接返回 refusal，不走 server-side fallback；已完成的 server-tool 结果可随部分回退路径保留。fallback 状态不能替代宿主对 client-tool 执行回执的查询。

## 文档组织参考

本章的主读者任务是理解一次工具回合中 API 状态、宿主执行和任务验收如何分属不同所有者，并查清具体 provider 的兼容条件。因此正文先用一条工具轨迹解释状态，再整理请求兼容表和拒答路由；它不是要求读者照步骤调用 API 的操作教程。Diátaxis 将 explanation、reference、how-to 与 tutorial 视作不同读者需求；本章借其任务区分决定顺序，不声称该框架验证了本章的学习效果。

## 来源支持的事实

- System Card 将 cyber safeguard 描述为三阶段：读取模型内部 activation 的 probe、运行在 Sonnet 5.5 上的轻量 classifier、以及结合 probe verdict 决定是否 block 的独立 LLM classifier。该描述属于发布方披露的控制组成，不公开完整权重、内部实现、阈值或服务端部署细节。
- 发布页说明 Sonnet 5.5 可在常规软件生命周期中帮助查找和修复 bug；更高风险 cyber 请求会 visibly fall back 到 Sonnet 5。发布页另称生物 safeguards 与 Sonnet 5 相同，并介绍了 reasoning extraction safeguards。
- API refusal 是正常的 HTTP 200 响应，包含 `stop_reason: "refusal"` 与可能的 `stop_details.category`。官方文档列出的类别包括 `cyber`、`bio`、`frontier_llm`、`reasoning_extraction` 和 `general_harms`；类别或解释也可能为 `null`。
- 截至核对日期，Anthropic 文档把 `fallbacks: "default"` 标为 Claude API 和 Google Cloud Anthropic-managed surface 上的 beta 能力。Sonnet 5.5 的 server-side default 路由将 `cyber` 与 `frontier_llm` refusal 推荐给 Sonnet 5；`bio`、`reasoning_extraction` 与 `general_harms` refusal 不走该默认重试。确切目标取决于请求模型和类别，文档说明推荐路由可随模型/类别更新。
- API 也允许调用方指定最多三个 permitted fallback models；这属于请求方配置的目标列表，与 server-side default 的推荐不同。fallback target 必须对所请求模型获准，且请求参数须与目标兼容。
- server-side fallback 由安全分类器拒绝触发；请求模型的 rate limit、overload 或 server error 会原样返回，不触发这条 safety fallback。若 fallback 模型也 rate limited 或 overloaded，fallback attempt 不会执行，前一拒绝仍会返回。
- 成功执行 fallback 时，响应的顶层 `model` 标出返回内容的实际模型，`fallback` content block 记录交接及类别，`usage.iterations` 记录各次尝试。只看 requested model 或 HTTP 状态码不足以判断谁生成了最终结果。
- 若请求使用 `thinking: {"type": "between_tools"}`，当前 `2026-07-01` server-side fallback 合同会在 Sonnet 5 的 fallback attempt 中使用 `thinking: {"type": "disabled"}` 和 `display: "omitted"`；这属于 API 参数兼容行为，不说明模型内部推理机制。
- System Card 的发布方评估将 Sonnet 5.5 归为 CB-1 与 Autonomy-1 阈值适用，未达到 CB-2/Autonomy-2；Autonomy-1 风险等级为 low。该等级是 Anthropic 的风险判断，不是无风险或独立认证结论。
- 发布页报告 Terminal-Bench 4.0 `70.6%`，System Card 的结果绑定 66 个任务、Claude Code `--bare`、max effort、5 trials、无外网和资源约束；卡片记录约 1.2% 请求 / 1.5% trials 触发 fallback，故分数包含实际路由结果。
- System Card 报告 OSWorld 2.1 `80.1% partial / 43.5% strict`，绑定 108 个 GUI 任务、最多 500 actions、5 runs 和 100K token 后的 server-side compaction；发布页只列 partial `80.1%`，不能据此丢弃 strict 口径。
- System Card 报告 Gray Swan IPI k=1/10/15 的攻击成功率为 `0.4% / 2.7% / 3.4%`，约 11% rollout fallback 到 Sonnet 5；coding Shade 也有部分请求由 fallback 模型完成。汇总成绩因此描述混合路由系统，不能直接标为裸 Sonnet 5.5 分数。

## 证据边界

- System Card 的数值描述特定发布方 benchmark、模型快照和控制配置；带 safeguard/fallback 的系统结果不能改写成裸 Sonnet 5.5 的能力或生产安全率。
- API 文档说明一组接口合同，不保证所有产品 surface、provider、云托管方式或后续版本的路由完全相同。应在实际响应中记录 requested model、served model、refusal category、fallback event、每次 attempt、工具权限与外部副作用。
- classifier 会有误拒；类别化 fallback 是供应商选择的策略，不是对拒绝内容重新授权。替代模型仍须满足调用方自己的身份、权限、数据范围和 action ledger 约束。
- 公开材料不支持推断 Sonnet 5.5 的参数规模、完整训练配方或未披露 safeguard 内部实现，也不构成独立验证。
