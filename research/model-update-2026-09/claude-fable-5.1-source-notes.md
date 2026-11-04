# Claude Fable 5.1：发布方评测、safeguards 与榜单边界

资料核对日期：2026-11-03。本文保存官方发布页、System Card、第三方测量和当前 DeepSWE 页面所支持的版本事实，不把发布方结果改写为裸模型或生产质量结论。

## Anthropic 发布材料

- **发布日期与阅读范围：** Anthropic [Fable 5.1 and Mythos 5.1 发布页](https://www.anthropic.com/claude-fable-and-mythos-5-1)，页面日期为 2026 年 9 月；阅读模型关系、安全说明、benchmark 表及脚注。官方 [System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card) 是完整安全与评测资料入口。
- **模型与 safeguards：** 发布页称 Fable 5.1 与 Mythos 5.1 是同一模型、不同 safeguards 配置；Fable 面向一般可用，Mythos 仅限 trusted-access 项目，侧重网络安全和生命科学。该材料支持发布方对服务访问和安全配置的说明，不提供完整权重或训练方法。
- **发布方 benchmark 表：** 页面报告 Terminal-Bench-Science 0.1 `52.6%`、Terminal-Bench 4.0 `55.8%`（Mythos 5.1 同表为 `60.9%`）、GDPval-AA v2 `1853`、OSWorld 2.0 partial/strict `77.9%/41.7%`、AutomationBench `31.4%`、CursorBench 3.2.0 `73.4%`。这些数字绑定 Anthropic 页面所述的 benchmark revision、effort 与执行设置。
- **安全条件：** Fable 5.1 评测启用了 production safeguards；页面说明在 OSWorld 2.0 的 safeguards intervention 上，将相应任务记为零分，另有 cyber 任务转给 Opus 4.8、biology 任务转给 Opus 5。报告结果时应保留 safeguard 状态、实际执行模型与 fallback 原因。
- **风险等级披露：** Fable 5.1 与 Mythos 5.1 的 System Card 将两者列为 CB-1、未达到 CB-2；Autonomy threat model 1 适用且风险等级为 low，threat model 2 未达到。该结果是发布方按公开威胁模型给出的风险判断，不是无风险声明，也不构成独立认证。

## 第三方配置与 Agent 榜单

- **Artificial Analysis：** [Claude Fable 5.1 页面](https://artificialanalysis.ai/models/claude-fable-5-1)当前标题为 Max、Default Fallback 配置，报告 Intelligence Index `49` 和 1M context。这是第三方 provider/configuration 测量，不等同于 Anthropic 发布方评测。
- **DataCurve DeepSWE：** [DeepSWE v1.1 leaderboard](https://deepswe.datacurve.ai/)本次访问时标注 113 tasks、更新于 2026-09-22；可见 `claude-fable-5[xhigh]`，没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，所有参测模型按页面说明使用 `mini-swe-agent`。Fable 5 的结果不能迁移给 Fable 5.1。
- **缺项边界：** 这是当前页面快照，不证明该版本永远未被测；缺精确行不能解释成零分、失败或 Fable 5.1 与旧版相同。

## 证据边界

发布页与 System Card 支持发布方在其 benchmark、effort、harness、safeguards 和 fallback 条件下报告的结果。Artificial Analysis 和 DataCurve 分别记录自己的 provider/configuration 与 Agent 系统测量。三类证据不可合成一个模型排名，也不能互推参数、架构、完整训练 recipe、独立复现或目标产品效果。引用 OSWorld 时必须保留 partial/strict 标签和版本，不能只留下较大的 partial 分数。
