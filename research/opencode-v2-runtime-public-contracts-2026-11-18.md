# OpenCode v2.3.2 Runtime 合同与架构边界

检索日期：2026-11-18  
证据状态：source-backed。动态官方文档以访问日为准；源码观察绑定 OpenCode v2.3.2 release commit c0ad18b。该记录只保存产品事实、来源和边界，不记录本章审阅决定或修改过程。

## 版本与资料范围

- **固定发布**：[OpenCode v2.3.2 GitHub release](https://github.com/anomalyco/opencode/releases/tag/v2.3.2)，2026-11-18 发布，release 页面列出短 commit c0ad18b；[官方 changelog](https://opencode.ai/changelog/v2.3.2)同日列出该版本。版本演进频繁，以下源码观察不能外推到其他 revision。
- **配置、权限与 Agent 合同**：[Config](https://opencode.ai/docs/config)、[Permissions](https://opencode.ai/docs/permissions)、[Agents](https://opencode.ai/docs/agents)，2026-11-18 访问。
- **工具与扩展**：[Tools](https://opencode.ai/docs/tools)、[MCP servers](https://opencode.ai/docs/mcp-servers)、[Instructions](https://opencode.ai/docs/instructions)、[Server plugins](https://opencode.ai/docs/build/plugins/server)，2026-11-18 访问。
- **状态与 API**：[API reference](https://opencode.ai/docs/api)、[Snapshots](https://opencode.ai/docs/snapshots)、[Compaction](https://opencode.ai/docs/compaction)、[SDK overview](https://opencode.ai/docs/build/sdk)、[Web](https://opencode.ai/docs/cli/web)，2026-11-18 访问。
- **版本迁移**：[Migrate from V1](https://opencode.ai/docs/migrate-v1)，2026-11-18 访问；该文档明确 V1 与 V2 的配置、插件及服务 API 不可按字段名直接视为兼容。
- **固定源码**：[SECURITY.md](https://github.com/anomalyco/opencode/blob/c0ad18b/SECURITY.md)、[core/src/permission.ts](https://github.com/anomalyco/opencode/blob/c0ad18b/packages/core/src/permission.ts)、[core/src/session.ts](https://github.com/anomalyco/opencode/blob/c0ad18b/packages/core/src/session.ts)。本次仅定向审阅这些文件中与威胁模型、权限求值及 Session 操作入口直接相关的部分，不声称完成全仓源码审计。

## 可核验事实

### 配置和指令

- OpenCode 接受 JSON/JSONC 配置。配置来源包括系统目录、全局目录和沿当前工作目录向上发现的项目配置；managed settings 的优先级最高，macOS 还支持 MDM 管理偏好。文档说明项目中的配置文件按目录距离合并，靠近当前工作目录的配置覆盖相同设置，非冲突设置保留；系统设置覆盖普通系统/全局/项目值，managed preferences 再覆盖系统目录值。
- 配置合并不等于组织级能力禁用。文档明确指出单靠 settings 不能阻止用户添加 provider、MCP server、agent 或 plugin；需要使用 managed configuration 中的 policies 执行限制。
- V2 发现全局及环境中的 AGENTS.md；项目规则按目录逐级发现，AGENTS.md 与 CLAUDE.md 有兼容回退规则。指令会与 Agent/provider prompt、工具说明、MCP/skill/reference 指令和 session-specific instruction 一起组成模型输入；官方说明这些来源是组合，而非自动消解彼此冲突。指令文本影响模型上下文，不是文件、进程或网络的 OS 隔离。
- 同一配置文档还暴露模型、agents、permissions、policies、MCP、plugins、skills、worktrees、snapshots、compaction、tool output 等设置。字段和默认值可随版本变化，文档建议以配置 schema 为准。

### Agent、工具和权限

- V2 的内建可见 Agent 包括 Build、Plan、General、Explore。Plan 是 primary agent，可在被要求时写 OpenCode plan 文件；其 shell 命令仍受权限规则控制，因此不能概括成普遍只读沙箱。subagent 在独立 child session 中运行；父 Agent 的 subagent 权限控制它能启动哪些 Agent，child 使用自己的配置权限，不是父权限的自动子集。v2.3.0 起官方 changelog 将 subagents 描述为默认后台运行。
- V2 工具合同与旧稿常见的 bash/todowrite/lsp 列表不同。当前文档的工具目录含文件、搜索、shell、web、skill、question、subagent、execute、session/worktree 和 desktop browser 等入口；可用工具依 Agent、配置和扩展而变。
- 权限规则使用 action、resource、effect 三元组，effect 可为 allow、ask、deny；资源匹配口径因 action 而异。shell action 使用 Shell scanner 生成的命令文本，复合命令可能拆为多个子命令；目录推断是 best effort。默认规则不是通用最小权限策略，当前文档列出外部目录和环境文件的特定默认行为，需按当前生效配置复核。
- 官方 Permissions 文档说明：规则命中顺序、saved approval、非交互客户端处理和 policy 约束会影响最终结果；allow-once 是一次请求，allow-always 会保存项目级允许规则，配置 deny 不被 saved approval 覆盖。
- 固定源码 c0ad18b 的 core/src/permission.ts 中，求值按传入 rule sets 展平并取最后一条匹配规则；无匹配时的回退为 ask。求值会组合 Agent/Session 规则与已保存的项目允许规则，并调用 permission/evaluate plugin hook；显式 deny 在进入 hook 前直接返回。ask 请求进入待处理队列，客户端回复后继续或拒绝对应调用。该实现观察只适用于 c0ad18b 的代码路径，不代表所有插件、命令入口或部署组合的安全性质。
- 官方 [SECURITY.md](https://github.com/anomalyco/opencode/blob/c0ad18b/SECURITY.md) 明确说明 OpenCode 本身不提供 Agent sandbox；permission system 是让用户了解和确认操作的 UX 控制，不是安全隔离。需要隔离时须在 OpenCode 外部配置容器或 VM。该文档还把配置的 MCP server 行为列为 OpenCode 自身信任边界之外。

### MCP、插件与代码模式

- MCP server 可提供 tools、prompts、resources 和 instructions；工具定义会消耗模型上下文。V2 默认可通过 Code Mode 名称空间访问 MCP 工具，也可按 server 配置为直接工具。工具权限仍按规范化 action 名匹配。
- 当前 Tools 文档说明 Code Mode 没有直接文件系统访问、imports 或 timers；程序通过工具目录访问能力。嵌套调用的 built-in/MCP tools 继续检查各自权限；plugin tools 依其声明的权限合同处理。此产品机制不等于运行时进程隔离。
- V2 plugins 可按 Server、TUI、GUI、RPC 分部运行，扩展 tools、hooks、integrations、commands、agents、UI 等。官方 migration 文档明确 V1 plugin implementation 不会因移动目录或重命名配置而自动兼容 V2。
- AGENTS.md、skills、references、MCP instructions、attachments 和 executable tools 是不同输入/能力类型；不能把 instruction file 当授权，也不能把所有 extension 统称为可执行工具。

### Server、SDK 与会话状态

- V2 官方迁移文档说明服务 API 和客户端已改版；网络 API 可由 @opencode/client 使用。当前 API reference 标为 experimental HttpApi，覆盖所选 instance routes。@opencode/sdk 则可把 OpenCode host 放入应用进程，通过内存路由调用，不开 HTTP listener，不能与网络 client 混成一种部署路径。
- V2 Web 文档说明 Web UI 与 TUI 使用同一服务，默认只监听 localhost；浏览器可通过一次性配对链接取得 session cookie。其他 HTTP 客户端使用 Basic auth，用户名固定为 `opencode`。V2 `OPENCODE_PASSWORD` 设置服务密码，迁移文档说明旧变量 `OPENCODE_SERVER_PASSWORD` 仍兼容。若把服务暴露到其他网络接口，网络边界和客户端认证由部署方负责。
- 固定源码 c0ad18b 的 core/src/session.ts 暴露以 session ID 定位的 prompt、compact、revert、interrupt、wait、message 等操作，并将对应调用分派给每个 Session 的处理对象。该代码片段支持“Session facade 协调会话动作”的有限描述，不证明 Session 本身拥有全部业务状态或验收逻辑。
- Snapshot 文档说明：文件 snapshot 需要 Git repository；捕获范围限于 active directory 下的 tracked files 和每个不超过 2 MiB 的非忽略 untracked files；捕获 best effort；不包含 Git metadata、ignored output、外部目录、进程、数据库或网络副作用。undo/revert 会覆盖受影响路径的当前内容，不能当作事务、备份或安全擦除。
- Compaction 文档说明 V2 以摘要加近期对话尾部压缩上下文，过程有损；完整历史仍保存在 session 中，模型请求只带压缩表示与保留尾部。provider native compaction 与本地 summary 有不同可移植性边界。背景摘要可能最终未被采用但仍消耗 token。
- API 事件/接口目录包含 session、permission、tool、step、compaction、filesystem、MCP 和 server 等事件/端点。事件是运行记录入口，不是任务 verifier，也不保证部署方已保留完整日志或敏感信息已被脱敏。

## 证据边界

- 官方文档支持该版本访问日的公开产品合同和配置用法，不证明用户部署已采用某默认值、实际工具调用成功、外部 server 安全或任务完成。
- 固定源码只支持所审阅 revision 中对应代码路径，不证明其他 release、平台、plugin、provider 或环境行为相同；本次未进行完整源代码审计。
- 当前 API reference 把部分 HttpApi 路由标作 experimental。SDK/API 的可调用性不等于稳定企业合同或受保护的远程执行环境。
- SECURITY.md 的“不提供 sandbox”是项目自身威胁模型声明；它不能代替实际部署的 OS/container/VM、网络和凭据审计。
- 本次未运行 OpenCode、MCP server、plugin、model request、shell command 或真实 coding task；没有性能、体验或安全测试数据。
