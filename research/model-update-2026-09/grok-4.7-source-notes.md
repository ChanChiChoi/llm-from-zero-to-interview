# Grok 4.7：发布、API 与评测来源

资料核对日期：2026-11-03。本文记录公开来源支持的名称、接口和指定评测结果；第三方榜单与发布方结果分账，不能据此推断完整架构或一般能力。

## 官方发布与接口

- **发布日期与阅读范围：** [Introducing Grok 4.7](https://x.ai/news/grok-4-7)，页面日期 2026-09-21；阅读发布说明、benchmark 表格与脚注。当前页面品牌显示为 SpaceXSI，链接仍位于 `x.ai` 域名。
- **来源支持的发布方描述：** 页面称 Grok 4.7 使用较大的 base model，进行更长的强化学习训练，训练任务更难且偏向需要数小时完成的问题；页面还称模型改善了自我核验、长上下文管理，并针对 Grok Bot harness 做了适配。这些是发布方描述，不是完整的参数、优化器、奖励、数据或 rollout 配方。
- **发布方 benchmark 表：** 页面报告 CursorBench 4.0 `46.3%`、DeepSWE v1.1 `71.0%`（脚注标为 high effort）、EEBench `64.0%`、AA Briefcase v1.1 `1,657`、Terminal-Bench 4.0 `37.6%`、Harvey Legal Agent Benchmark `19.6%`、HealthBench Professional `56.7%`。它们分别绑定发布方评测设置；该表不能单独支持不同 harness 下的统一比较。
- **接口版本：** [官方模型/API 文档](https://docs.x.ai/developers/models/grok-4.7)当前列出 model ID `grok-4.7`、文本和图像输入、500,000-token context，以及 low/medium/high/xhigh reasoning effort（默认 high）。接口页支持这些 API 合同，不披露内部架构。

## 第三方配置与 Agent 榜单

- **Artificial Analysis：** [Grok 4.7 页面](https://artificialanalysis.ai/models/grok-4-7)当前对应 xhigh provider 配置；页面将其标为 proprietary model，报告 Artificial Analysis Intelligence Index `35` 和 500K context。该分数属于 Artificial Analysis 的测试配置，不等于发布方各 benchmark 结果，也不是独立的通用能力概率。
- **DataCurve DeepSWE：** [DeepSWE leaderboard](https://deepswe.datacurve.ai/)页面在本次访问时标注 v1.1、113 tasks、更新时间 2026-09-22、57/70 configs 和 21/28 models；榜单说明参测模型统一运行于 `mini-swe-agent`。可见 Grok 4.6 `[medium]`，没有精确 Grok 4.7 配置。页面也说明任务来自 91 个仓库、覆盖 5 种语言。
- **缺项边界：** 上述内容只描述本次页面快照。无精确行表示当前可见页面没有可引用的 Grok 4.7 DataCurve 结果；不得迁移 Grok 4.6 数字，也不得把缺行编码为零分、失败或永久未测。

## 证据边界

发布页可支持其公开的训练路线描述和发布方结果；API 文档可支持 model ID 与接口字段；Artificial Analysis 与 DataCurve 分别支持其页面所绑定的 provider/configuration 和 harness/任务快照。公开资料没有由此确认参数量、MoE/稠密结构、完整训练 recipe、生产 kernel 或跨 harness 的能力提升。新页面、模型别名和榜单配置应按访问日期重新核对。
