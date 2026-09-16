# DeepSeek-R1-0528：官方 API 发布页核验

核验日期：2026-09-14。证据来自本地保存的 DeepSeek API Docs 页面快照 `/tmp/deepseek-news.html`（页面标题为 “DeepSeek-R1-0528 Release”，页面导航标注发布日期 2025/05/28）；网络复访当前受 DNS 限制，因此以下内容限定为官方发布页快照。

## 页面明确披露

- 发布页标题为 `DeepSeek-R1-0528 Release`，页面导航标注发布日期 2025/05/28。
- 页面自述改进方向包括 benchmark performance、前端能力和 hallucination reduction；这些是发布方摘要，不是本轮独立复现的定量结论。
- 页面明确写出支持 JSON output 与 function calling，并说明 API 使用方式没有变化，链接指向 thinking mode 文档。
- 页面提供开源权重入口：[deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)。权重是否仍可下载、具体许可证和 revision 需在目标仓库实时复核。
- 发布页附有 benchmark 图片和示例 GIF；本轮没有把图片中的数字转录为正式结果，也没有把演示当作独立复现。

## 工程解释与边界

“API 使用方式没有变化”只说明发布页所述兼容性承诺，不等于所有账户、SDK、限流、流式和错误码行为永久一致。JSON/function calling 是输出协议能力，工具执行、权限、网络、沙箱、审批、超时和审计仍由宿主系统负责。开放权重入口也不自动等于许可证、训练数据或完整技术报告已核验。

## 尚待核验

本轮未核验模型卡全文、参数规模、架构、训练/后训练配方、完整 benchmark 表格、硬件、采样设置、发布日期在权重仓库中的 revision 对应关系以及独立复现。Artificial Analysis 或其他榜单中的分数必须与模型 revision、effort、harness 和任务集绑定，不能直接归因于基础模型。

## 书系映射

- 第四册：发布页证据等级、版本日期和开放权重边界。
- 第六册与第二十二册：JSON/function calling、thinking mode、流式和错误兼容性迁移。
- 第七册：复现 benchmark 图片时固定任务、采样、硬件和统计区间。
- 第十七册与第二十册：工具宿主、权限、沙箱和 Agent trace 审计。
