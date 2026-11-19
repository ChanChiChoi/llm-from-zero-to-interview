# Claude Code 公开产品合同与架构边界

研究日期：Claude Code 文档扫描为 2026-11-18；OpenTelemetry 术语补核为 2026-11-19

证据状态：source-backed；本记录只整理官方文档支持的产品事实，不表示已在本地运行或验证 Claude Code。

## 研究范围

核对 Claude Code 当前公开资料中与产品表面、Agent 工作循环、上下文和会话状态、权限与 OS 沙箱、扩展机制、checkpoint 及组织可观测性相关的描述。资料均为 Anthropic 官方产品文档，页面内容会随产品更新；下述事实应连同访问日期和页面适用范围阅读。

## 来源与支持范围

| 来源 | 日期/版本 | 直接支持的事实 |
| --- | --- | --- |
| [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works) | 官方在线文档，2026-11-18 访问 | 把产品工作过程概括为 gather context、take action、verify；说明工具结果会影响后续步骤，用户可在循环中介入；公开列出文件、搜索、执行、Web 和代码智能等工具类别；称周边层提供工具并管理模型所见上下文；列出 session transcript、恢复/分叉、Auto memory 和项目指令等行为。 |
| [Platforms and integrations](https://code.claude.com/docs/en/platforms) | 官方在线文档，2026-11-18 访问 | 列出 CLI、Desktop、VS Code、JetBrains、Web、移动端、CI 与其他集成；区分 local、cloud、Remote Control 等执行位置；说明不同表面在交互和功能上存在差异，部分本地配置/记忆/MCP 配置可跨本地表面共享。 |
| [Settings files and precedence](https://code.claude.com/docs/en/settings) | 官方在线文档，2026-11-18 访问 | 列出 user、shared project、project local、managed 等配置来源及范围；说明本机表面与 cloud session 读取设置的差异；指出 Claude Code system prompt 未发布。 |
| [How Claude remembers your project](https://code.claude.com/docs/en/memory) | 官方在线文档，2026-11-18 访问 | 区分用户维护的 CLAUDE.md / AGENTS.md 指令与 Auto memory 记录；两者被加载为上下文而非 enforced configuration；文档建议用 PreToolUse hook 阻止不应执行的工具动作。 |
| [Configure permissions](https://code.claude.com/docs/en/permissions) | 官方在线文档，2026-11-18 访问 | 说明 tool-specific allow/ask/deny 规则、权限模式、管理范围及匹配顺序；规则属于 Claude Code 产品层的工具决策合同。 |
| [Configure the sandboxed Bash tool](https://code.claude.com/docs/en/sandboxing) | 官方在线文档，2026-11-18 访问 | 本机 OS sandbox 作用于 Bash、PowerShell、Monitor 命令及其子进程；当前文档说明默认关闭。内建文件/Web 工具、MCP servers、hooks、plugin monitors、LSP 和辅助进程在此 sandbox 外；存在 unsandboxed retry 路径；native Windows 命令不使用此 sandbox。 |
| [Checkpointing](https://code.claude.com/docs/en/checkpointing) | 官方在线文档，2026-11-18 访问 | checkpoint 在每个开始新 turn 的 prompt 前保存相关文件状态，并追踪 Claude 的 file-editing tools；Bash-only 文件变化、外部手工修改和许多 subagent 编辑不一定能通过当前 session rewind 恢复。 |
| [Hooks reference](https://code.claude.com/docs/en/hooks) | 官方在线文档，2026-11-18 访问 | hook 可关联 session/tool 等生命周期事件；handler 类型包括命令、HTTP、MCP tool、prompt 和 agent 等，不同事件具备不同输入与控制语义。 |
| [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp) | 官方在线文档，2026-11-18 访问 | MCP server 可连接外部工具、数据源和 API；文档区分本地/远端 transport、安装 scope、授权与 server 运行状态。 |
| [Create custom subagents](https://code.claude.com/docs/en/sub-agents) | 官方在线文档，2026-11-18 访问 | subagent 有独立 context window、自定义指令、可配置工具和权限；subagent 属于单会话内部的委派能力。 |
| [Extend Claude with skills](https://code.claude.com/docs/en/skills) | 官方在线文档，2026-11-18 访问 | skills 是可复用任务指引/工作流，可依据 scope 与配置加载。 |
| [Plugins overview](https://code.claude.com/docs/en/plugins) | 官方在线文档，2026-11-18 访问 | plugin 可以把 skills、agents、hooks、MCP servers 等组件作为一组安装和加载。 |
| [Monitoring](https://code.claude.com/docs/en/monitoring-usage) | 官方在线文档，2026-11-18 访问 | 产品支持配置 OpenTelemetry metrics、events/logs 和可选 distributed traces；当前文档把 traces 标为 beta。特定文件路径、命令和工具内容字段受配置开关控制，其中 OTEL_LOG_TOOL_CONTENT 与 OTEL_LOG_TOOL_DETAILS 可让遥测包含敏感项目内容或工具细节。 |
| [What is OpenTelemetry?](https://opentelemetry.io/docs/what-is-opentelemetry/) | OpenTelemetry 官方在线文档，2026-11-19 访问 | 将 OpenTelemetry 定义为生成、收集和导出 traces、metrics、logs 等遥测数据的可观测性框架与工具集；它不是遥测存储或可视化后端。用于准确说明本章采用的 OTel 术语，不扩展 Claude Code 产品合同。 |

## 事实与边界

- 官方工作循环和工具列表支持对产品公开能力的描述；它们不证明内部恰好存在 Session Manager、Context Builder、Action Parser 等同名模块。
- 产品表面、执行位置和配置范围彼此相关但不是同一维度；某个 UI 的名称不能单独证明文件、网络、凭据和数据保留边界。
- 项目指令与 Auto memory 是进入模型上下文的信息源，不等同于强制权限规则。权限判断、用户确认和 OS sandbox 分属不同控制层。
- 本机 shell sandbox 只覆盖文档写明的命令族及子进程。不能从启用 shell sandbox 推断 MCP、hook、file tool 或其他宿主进程也处于该边界内。
- Session transcript、file checkpoint 和 OpenTelemetry 导出是不同观察/恢复能力。它们不自动构成完整执行快照，也不撤销所有外部副作用。
- 公开文档是动态产品合同，不支持对其他版本、未公开模型 prompt、planner、内部路由算法或单次运行行为的无条件外推。
- 本次仅核对公开文档；未执行 Claude Code、未验证本地配置、未复现工具行为或评估产品成功率。
