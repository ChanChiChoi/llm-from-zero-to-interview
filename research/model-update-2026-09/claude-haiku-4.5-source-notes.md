# Claude Haiku 4.5：Anthropic 官方模型目录核验

核验日期：2026-09-14。证据来自本地保存的 Anthropic Models Overview 页面快照 `/tmp/anthropic-page.html`（快照生成于 2026-09-09，文件大小 462,688 bytes，SHA-256 `4dd1eb9826c5cf9f1a740c5b1c29ecaed4cb9ff3c1e71d8709cb4499766ab9a5`）；网络复访当前仍受 DNS 限制，因此以下内容限定为带快照日期的官方目录字段。

## 已确认字段

- 模型 ID 为 `claude-haiku-4-5-20251001`，官方别名为 `claude-haiku-4-5`；页面名称为 Claude Haiku 4.5，生命周期为 active，`releasedOn` 为 `2025-10-15`，退休不早于 `2026-10-15`。
- 上下文窗口为 200,000 tokens，最大输出为 64,000 tokens；页面将 comparative latency 标为 `fastest`，产品描述为“最快且接近前沿智能”。
- 页面字段显示文本/图像输入、文本输出、视觉和工具使用；thinking 字段为 `extended=yes`、`adaptive=no`，未提供默认 effort 字段。不能据此推断内部推理算法。
- 平台和 ID 字段包括 Claude API（`claude-haiku-4-5-20251001`，别名 `claude-haiku-4-5`）、Bedrock、Vertex AI、Microsoft Foundry 和 AWS。
- 价格快照为输入每百万 token 1 美元、输出每百万 token 5 美元；缓存写入 5 分钟为 1.25 美元、1 小时为 2 美元，读取为 0.1 美元。可靠知识截止为 `2025-02`，训练数据截止为 `2025-07`。
- 页面关联 [announcement](https://www.anthropic.com/news/claude-haiku-4-5)、[system card](https://www.anthropic.com/claude-haiku-4-5-system-card) 和迁移指南入口；本轮未读取这些页面全文。

## 工程解释与边界

200K context 和 64K output 是接口预算，不等于并发、KV cache 容量或长任务成功率。`fastest` 是目录比较字段，不能替代在固定平台、硬件、输入长度和输出预算下的延迟实测。工具支持描述协议能力，实际网络、文件、代码执行、审批、沙箱和审计仍由宿主系统负责。

## 尚待核验

官方模型目录没有披露参数规模、稠密/MoE 架构、训练数据配方、优化器、后训练算法、完整推理机制或独立 benchmark 复现。价格、退休日期和平台可用性需在目标账户与平台上实时复核；system card 和 announcement 仍待逐页核验。

## 书系映射

- 第四册：模型目录字段、上下文/输出边界和产品定位证据等级。
- 第六册与第二十四册：短上下文与低成本模型的 TTFT、TPOT、吞吐、缓存和单位成功成本。
- 第七册：Haiku 4.5 与 Sonnet/Opus/Fable 的固定 harness 对照，区分快速延迟字段和实测延迟。
- 第十七册与第二十册：工具宿主、权限沙箱、超时、回滚和审计责任边界。
