# 故障排查与事故证据来源

核验日期：2026-11-16。此记录保存公开工程资料支持的方法和证据边界，不记录书籍审阅过程。

## Google SRE 故障排查、监控与事故复盘

- **来源与版本**：Google SRE Book 在线章节：[Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/)、[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)、[Managing Incidents](https://sre.google/sre-book/managing-incidents/) 和 [Postmortem Culture](https://sre.google/sre-book/postmortem-culture/)。访问日期 2026-11-16；网站未在这些章节页面标注固定网页 revision 或更新日期。
- **实际阅读位置**：排障章的 problem report、triage、examine、diagnose、test and treat；监控章的 symptoms versus causes、black-box versus white-box；事故管理章开篇；postmortem 章定义、目标和复核标准。
- **来源支持的事实**：排障章建议报告预期行为、实际行为和可复现方式；严重事件中先止损并保留根因分析所需证据；用实验排除假设，并注意混杂条件、测试副作用和证据不确定性。监控章区分系统症状与可能原因，并说明外部症状信号和内部观测各有用途。事故管理章把限制事件干扰、尽快恢复正常运营作为目标。复盘章将事故影响、缓解/解决动作、根因和预防性后续行动列为记录内容，并强调从系统层面降低复发概率或影响。
- **证据等级与边界**：Google 发布的 SRE 工程手册，能支持其公开描述的可靠性实践；不是大模型任务质量、生成真实性、权限正确性或因果归因的专门标准，也不能证明某一 LLM 系统已满足这些目标。应用到模型系统时仍须自行定义任务契约、样本、权限边界和验收条件。

## 相关动态资料

Agent trace 用于调试、固定数据集用于重复评估的产品指南，记录于[Agent 轨迹诊断与线上评估来源](agent-trace-diagnosis-and-online-evaluation-2026.md)；GenAI telemetry 的敏感内容和默认采集建议，记录于[项目工具与可观测性来源](project-tooling-and-observability-2026.md)。这两类文档均会更新；实施时应固定所用页面、语义约定和 SDK 版本，并核对目标环境的真实行为。
