# Agent 轨迹诊断与线上评估来源

检索与核验日期：2026-10-30。本记录保存公开资料支持的研究方法、发布状态和证据边界；书籍审阅结论与修改历史记录在 `progress/`。

## Agentic CLEAR

- **来源与版本**：[ACL Anthology 2026.acl-demo.74](https://aclanthology.org/2026.acl-demo.74/)，ACL 2026 System Demonstrations，页 750–763，DOI 10.18653/v1/2026.acl-demo.74；另有对应 [arXiv 预印本](https://arxiv.org/abs/2605.22608)。
- **实际阅读位置**：ACL Anthology 论文元数据、摘要、方法概述与实验设置。
- **来源支持的事实**：方法在 system、trace、node 三个粒度产出诊断；先由 judge 评估步骤、完整 trace 和任务 rubric，再聚合跨 trace 的模式。作者在四个 benchmark、七种 agent 设置和数万次 LLM 调用上报告诊断与人工错误标注有较强对齐，并能预测任务成功率。
- **证据等级与边界**：ACL System Demonstrations 正式论文；方法和结果仍是论文作者在所用 benchmark、agent、judge 和协议下报告的证据。不能据此推断另一项目的根因准确率，也不能把 judge 诊断等同于安全验证或因果归因。

## AgentRx

- **来源与版本**：[Microsoft Research 官方博客](https://www.microsoft.com/en-us/research/blog/systematic-debugging-for-ai-agents-introducing-the-agentrx-framework/)，发布于 2026-03-12。
- **实际阅读位置**：博客摘要及 AgentRx 问题、框架、benchmark 和结果概述。
- **来源支持的事实**：发布方介绍用 guarded executable constraints 和逐步证据日志定位首个 critical failure step；AgentRx Benchmark 包含 115 条来自 τ-bench、Flash 和 Magentic-One 的人工标注失败轨迹，附九类 failure taxonomy。博客报告相对 prompting baselines 的失败定位提升 23.6%、根因归因提升 22.9%。
- **证据等级与边界**：Microsoft Research 发布方介绍；样本范围、比较基线和数字按博客所述理解，未在本地复现，也不是对所有 agent 任务的普遍收益承诺。
