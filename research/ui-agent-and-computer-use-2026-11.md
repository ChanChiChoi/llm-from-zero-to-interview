# UI Agent 与 Computer Use 资料记录

核对日期：2026-11-16。本文只记录公开来源事实、版本和证据边界；章节取舍和正文修改见 `progress/book-17-agent-tool-use/chapter-08-audit.md`。

## 1. OpenAI Computer Use 文档

- 来源：[OpenAI API Computer use](https://developers.openai.com/api/docs/guides/tools-computer-use)，动态官方文档；2026-11-16 核对。
- 类型与证据：供应商官方 API 文档，支持该文档快照中的接口流程和官方安全建议，不证明任意执行环境都实现了这些控制。
- 来源支持：Responses API 的 `computer` tool 返回结构化动作；宿主 action handler 负责把允许的动作转为浏览器或操作系统输入，并通过对应 call ID 回传截图。文档建议在隔离 browser/VM 中执行、限制允许站点与动作、将屏幕内容视为不可信数据、对有实际后果的动作取得确认、设置步数/时间/成本边界并检查真实结果。
- 边界：接口和支持模型会变更。该工具输出动作不等于 API 自动获得或控制宿主浏览器权限；运行时授权、动作执行、状态保留、取消和最终验证仍由应用与环境承担。

## 2. Anthropic Computer Use 文档

- 来源：[Anthropic Claude Platform Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool)，动态官方文档；2026-11-16 核对。
- 类型与证据：供应商官方工具文档，支持工具合同和厂商列出的安全建议。
- 来源支持：当前页面使用 `computer_toolset_20260801` 这一客户端工具集名称，提供截图、鼠标和键盘交互；文档要求调用方在自己控制的环境中执行工具成员。页面将 browser use 列为留在网页内部操作时更合适的另一类工具。官方风险建议包括最小权限的专用 VM/container、限制敏感数据暴露、域名 allowlist、对有现实后果的操作或需要肯定同意的事项保留人工确认，并防范屏幕文本中的 prompt injection。
- 边界：工具集名称、成员、模型兼容性和托管服务可用性依版本变化；不能把该产品合同泛化为所有 Computer Use Agent 的接口规范或安全保证。

## 3. OSWorld 论文与 OSWorld-V2 发布

- 论文来源：[OSWorld 2.0 arXiv v2](https://arxiv.org/abs/2606.29537)，2026-07-13 修订版；核对摘要及版本信息。
- 发布来源：[xlang-ai/OSWorld-V2 `osworld-v2.1` release](https://github.com/xlang-ai/OSWorld-V2/releases/tag/osworld-v2.1)，官方 release；核对活动状态、组件说明和发布日期。
- 类型与证据：论文预印本支持论文版本的任务定义、设置与作者报告；官方 release 支持其固定组件和发布状态；未独立复现 benchmark。
- 来源支持：论文版 OSWorld 2.0 定义 108 个长程 computer-use 工作流；摘要报告在 500 步 binary-completion 设置下，Claude Opus 4.8、maximum thinking、batched calls 的最佳设置达到 20.6%，partial score 为 54.8%。OSWorld-V2 仓库的 `osworld-v2.1` 将代码、任务、资产、模拟网站和 VM 镜像作为版本化配套组件发布，并作为当前推荐版本。
- 边界：论文数据绑定论文任务快照、模型、thinking 配置、批量调用和步数限制，不能改称 v2.1 结果。复现或比较必须固定发布标识和匹配组件，并记录 harness、权限、预算与 verifier；版本发布本身不证明生产任务覆盖或安全能力。

## 未验证事项

- 未运行 Computer Use 工具、OSWorld benchmark、模型或 VM；未验证实际产品账户可用性、成本、延迟或真实站点风险。
- 当前文档和推荐 release 是动态内容；复现时应重新核对固定版本和兼容模型，不以本记录替代接口运行验证。
