# Claude Sonnet 5.5：System Card 与 fallback API 合同

资料核对日期：2026-11-03。本文只记录 Anthropic 对 Sonnet 5.5 的模型安全披露、拒绝类别和 fallback API 合同；不把发布方评测写成独立复现或生产安全率。

## 官方来源

- [Claude Sonnet 5.5 System Card](https://www.anthropic.com/claude-sonnet-5-5-system-card)：模型能力、安全评估、风险判断和发布方 safeguards 的主要来源。评测结果绑定卡片所述的 snapshot、任务、工具、harness、effort、权限、safeguards 与 fallback。
- [Introducing Claude Sonnet 5.5](https://www.anthropic.com/claude-sonnet-5-5)：发布方称常规软件开发中的找错和修复可以继续；较高风险 cyber 请求会 visibly fall back 到 Sonnet 5。生物 safeguards 与 Sonnet 5 相同；Sonnet 5.5 还加入 reasoning-extraction safeguards。
- [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)：Anthropic API 拒绝字段、当前类别推荐、fallback 配置、响应轨迹与计费合同。页面检索日期 2026-11-03。
- [Claude Sonnet 5.5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide)：当前 API refusal 类别及 `between_tools` 请求转 Sonnet 5 时的接口约束。
- [Anthropic Transparency Hub](https://www.anthropic.com/transparency)：提供 System Card 的公开摘要，不等同于完整 System Card。

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
