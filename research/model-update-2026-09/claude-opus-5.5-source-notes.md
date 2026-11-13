# Claude Opus 5.5：发布方评测与配置边界

资料核对日期：2026-11-03。本文记录官方发布材料、System Card、第三方配置和当前 Agent 榜单快照支持的事实及边界；它不是章节审计或独立复现报告。

## 官方发布页与 System Card

- **发布日期与阅读范围：** Anthropic [Claude Opus 5.5 发布页](https://www.anthropic.com/claude-opus-5-5)，页面日期 2026-09-22；阅读性能、成本、安全、benchmark 表格与脚注。官方 [System Card](https://www.anthropic.com/claude-opus-5-5-system-card) PDF 为 17,795,106 bytes，SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`；已按 `PAPERS.md` 所载记录通过 PDF 文本映射只读解析。
- **发布方 benchmark 条件：** 发布页称除特别注明外使用 adaptive thinking/max effort；Terminal-Bench 4.0 的 Opus 5.5 使用 xhigh，页面报告 `66.4%`，System Card 精确值为 `66.36%`，绑定 Claude Code `--bare` 和 5 trials。页面脚注还说明 Opus 5.5 的发布方评测启用了 production safeguards；网络安全任务会由 Opus 4.8 完成，biology 与 frontier LLM development 任务会由 Opus 5 完成。
- **发布页当前 OSWorld 标签：** 发布页目前将 partial `81.8%` 标作 OSWorld 2.1。
- **System Card OSWorld 标签：** System Card 将 partial/strict `81.8%/48.7%` 标作 OSWorld 2.0，并记录 108 tasks、1080p、最多 500 actions、5 runs，以及超过 100K tokens 后的 server-side compaction。两个来源的标签应原样保留；相同 partial 分数不能证明任务修订或执行条件完全相同，不能将 strict `48.7%` 移植为 2.1 结果。
- **其他 System Card 例子：** Terminal-Bench 4.0 `66.36%`、ProgramBench `91.2%`（166 golden tasks）、CoBench 2.1 `55.8%`、AECI `169.36` 与 cyber ACE `301/410=73.4%` 均是各自任务/配置下的发布方数据；其中 ACE 结果使用关闭 safeguards 的设置。不能将这些异质量表合并为单一综合能力分数。
- **风险等级披露：** System Card 将 Opus 5.5 评为 CB-1、未达到 CB-2；Autonomy threat model 1 适用且风险等级为 low，threat model 2 未达到。该评估是发布方在其威胁模型与测量条件下的判断，不等于没有相关能力或不存在未测风险。

## 第三方资料

- **Artificial Analysis：** [Claude Opus 5.5 页面](https://artificialanalysis.ai/models/claude-opus-5-5)当前标题为 Max、Default Fallback 配置，页面报告 Intelligence Index `58`、1M context。该记录属于第三方 provider/configuration 测量，不等于 Anthropic System Card 的发布方 harness。
- **DataCurve DeepSWE：** [DeepSWE v1.1 leaderboard](https://deepswe.datacurve.ai/)当前可见 `claude-opus-5[max]`，没有精确 `mini_swe_agent_claude_opus_5_5_*` 行。当前可见页面快照由 `mini-swe-agent` 运行 113 tasks。不得迁移 Opus 5、Fable 5.1 或其他 Claude 的 Agent 指标；页面缺行也不代表零分。

## 证据边界

官方发布页与 System Card 支持“发布方在指定 effort、harness、任务、safeguards、fallback 和成功定义下报告了结果”。Artificial Analysis 支持其第三方配置测量，DataCurve 页面支持它列出的固定 Agent 配置。它们不能互相替代，也不能证明裸模型跨任务、跨 provider 或生产流量的能力。System Card 与当前发布页的 OSWorld 版本标签存在需保留的差异；未用相同任务文件和协议逐项核对，不能断言标签差异代表何种数据集变更。

## 2026-11-13 API effort 合同复核

- **来源与阅读范围：** Anthropic 当前 [Claude Opus 5.5 模型页](https://platform.claude.com/docs/en/models/opus-5-5/overview)与[Thinking 文档](https://platform.claude.com/docs/en/build-with-claude/thinking)，访问日期 2026-11-13；复核 adaptive thinking、effort 配置、手工 budget 和同步输出限制。
- **来源支持的事实：** Opus 5.5 模型页将 adaptive thinking 列为始终开启，以 `output_config.effort` 控制档位；文档说明该模型不接受 `thinking.type=disabled` 或手工 `budget_tokens`。同步请求的 `max_tokens` 是 thinking、工具调用和可见输出共享的响应上限。
- **证据等级与边界：** `source-backed`（Anthropic 官方当前 API 文档）。这些字段说明公开请求合同，不表示固定的内部 token 预留量、模型推理算法或跨 endpoint 的统一行为；应用仍需用真实 usage 校准任务级成本与时限。
