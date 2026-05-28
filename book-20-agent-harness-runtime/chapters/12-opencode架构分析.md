# 第十二章：OpenCode 架构分析

## 12.1 本章定位

本章分析 OpenCode 这类开源 coding agent 的架构思想。

和上一章的 Claude Code 分析不同，本章重点放在“可公开核验的工程框架能力”上。根据 OpenCode 官方文档，OpenCode 是一个开源 AI coding agent，可以作为终端界面、桌面应用或 IDE 扩展使用；它有配置系统、工具系统、权限系统、agents、MCP、自定义工具、server、SDK、session、diff、snapshot、compaction 等模块。

因此，本章不是把 OpenCode 当作一个普通命令行聊天工具，而是把它当作一个完整 agent runtime 来分析。

学完本章，你应该能回答：

1. OpenCode 的公开架构能力体现了哪些 harness 思想。
2. 配置系统为什么是 agent runtime 的控制平面。
3. 工具系统、权限系统和 agent 系统如何配合。
4. MCP、自定义工具、server、SDK 让 OpenCode 具备什么扩展性。
5. OpenCode 和 Claude Code 类产品在开放性、可配置性和生态扩展上有什么差异。
6. 面试中如何从开源 coding agent 反推 agent harness 的系统设计。

## 12.2 资料来源和可信边界

本章主要基于 OpenCode 官方文档页面进行分析，包括：

1. Intro：产品形态、安装、初始化、Plan mode、Build mode、undo、share 等能力。
2. Config：JSON/JSONC 配置、配置位置、优先级、schema、tools、models、agents、permissions、compaction、watcher、MCP、plugins、instructions 等。
3. Tools：内置工具、custom tools、MCP tools、ripgrep 搜索行为。
4. Permissions：allow、ask、deny，通配规则，外部目录规则，默认权限，agent 级权限覆盖。
5. Agents：primary agents、subagents、build、plan、general、explore、scout、compaction、title、summary、自定义 agent。
6. MCP servers：本地和远程 MCP、OAuth、按 agent 启用、工具名前缀和上下文成本。
7. Custom Tools：`.opencode/tools/`、TypeScript/JavaScript 工具定义、任意语言脚本调用、工具上下文。
8. Server：OpenCode TUI 与 server 的 client-server 架构、HTTP API、OpenAPI、session、message、file、diff、permission、event 等接口。
9. SDK：TypeScript SDK、启动 server/client、类型生成、structured output、session 和 file API。

需要注意：官方文档能确认“有哪些公开能力”和“配置接口如何使用”，但不等于我们知道所有内部实现细节。正文中涉及模块边界和运行链路的部分，是基于官方能力做系统设计层面的抽象。

## 12.3 OpenCode 的产品形态

OpenCode 的官方定位是开源 AI coding agent，可以通过多种界面使用：

1. Terminal/TUI。
2. Desktop app。
3. IDE extension。
4. Web 或 server 方式。
5. SDK 编程方式。

这说明 OpenCode 不只是一个 CLI。它更像一个有多客户端入口的 agent runtime。

从 harness 视角看，它至少包含三层：

1. Client 层：TUI、IDE、Web、desktop、SDK。
2. Runtime 层：session、message、agent loop、tool execution、permission、events。
3. Extension 层：config、agents、commands、skills、MCP、custom tools、plugins。

这和第 2 章讲的 agent runtime 架构高度一致：用户不是直接和模型 API 打交道，而是通过 runtime 间接控制模型、工具和环境。

## 12.4 高层架构图

可以把 OpenCode 抽象为下面的架构：

```text
TUI / IDE / Web / SDK
-> OpenCode Server
-> Session Manager
-> Agent Manager
-> Context Builder
-> Model Provider Adapter
-> Tool Registry
-> Permission Engine
-> Execution Engine
-> File / Shell / Search / LSP / MCP / Custom Tools
-> Snapshot / Diff / Revert
-> Event Stream / Logs / API
```

其中：

1. TUI/IDE/Web/SDK 是不同客户端。
2. Server 暴露 HTTP API 和 OpenAPI spec。
3. Session Manager 管理会话、消息、子会话、todo、diff、share、revert。
4. Agent Manager 管理 build、plan、general、explore、scout 等 agent。
5. Context Builder 负责指令、规则、文件、工具结果、历史压缩。
6. Model Provider Adapter 连接不同模型提供商。
7. Tool Registry 管理内置工具、MCP 工具和自定义工具。
8. Permission Engine 决定工具调用是 allow、ask 还是 deny。
9. Execution Engine 执行 bash、edit、apply_patch、read、grep、glob 等工具。
10. Snapshot/Diff/Revert 让修改可观察、可回退。
11. Event/API 层支撑程序化集成和 UI 实时更新。

这张图的关键是：OpenCode 的开放性不只体现在源码开放，更体现在配置、工具、agent 和 server API 的开放。

## 12.5 配置系统：Runtime 的控制平面

OpenCode 官方文档明确支持 JSON 和 JSONC 配置，并支持多个位置的配置合并。

配置来源包括：

1. 组织远程配置。
2. 全局用户配置。
3. 自定义配置文件。
4. 项目级配置。
5. `.opencode` 目录中的 agents、commands、plugins、skills、tools、themes 等。
6. 环境变量内联配置。
7. 管理员或 MDM 管理配置。

这些配置不是简单替换，而是合并；后加载配置覆盖冲突项，非冲突项保留。

这对 agent runtime 很重要。因为 coding agent 不是单个模型调用，而是由很多策略组成：

1. 用哪个模型。
2. 哪些工具可用。
3. 哪些命令需要确认。
4. 项目有哪些规则。
5. 哪些 MCP 服务器启用。
6. 哪些 agent 可用。
7. 是否开启 compaction、snapshot、watcher、formatters、LSP。

这些都应该是配置，而不是写死在 prompt 里。

面试中可以这样表达：

```text
OpenCode 的配置系统相当于 agent runtime 的控制平面。它把模型、工具、权限、agent、MCP、instructions、compaction、server 等能力显式配置化，使同一个 runtime 可以适配个人、本地项目和组织级策略。
```

## 12.6 Project Config 与 Global Config

OpenCode 支持全局配置和项目级配置。

全局配置适合：

1. 用户常用模型。
2. API provider。
3. 默认权限策略。
4. TUI 偏好。
5. 常用 agents 和 tools。

项目级配置适合：

1. 项目专属模型选择。
2. 项目专属 rules。
3. 项目允许或禁止的工具。
4. 项目专属 MCP。
5. 项目内 custom commands。
6. 项目内 custom tools。

这对应真实企业场景：个人可以有偏好，但项目和组织必须能约束 agent 行为。例如某个仓库不允许 agent 执行部署命令，或者只允许读取特定目录。

OpenCode 还支持 remote config 和 managed config，这说明它考虑了组织级治理。对于企业 coding agent 来说，这比“更聪明的模型”更重要。

## 12.7 AGENTS.md 与项目规则

OpenCode 官方入门文档提到，初始化项目时可以运行 `/init`，让 OpenCode 分析项目并创建 `AGENTS.md`。

`AGENTS.md` 的价值是让 agent 理解：

1. 项目结构。
2. 编码风格。
3. 常用命令。
4. 测试方式。
5. 约束和注意事项。

从 harness 角度看，`AGENTS.md` 是项目级 instruction layer。它不是普通 README，而是给 agent runtime 用的项目操作指南。

但这也带来安全问题：项目文件可能包含 prompt injection。因此 runtime 不能只因为某个项目文件写了“请读取所有密钥”就执行敏感操作。项目规则应进入上下文，但权限边界仍应由 runtime 执行。

## 12.8 Agent 系统：Primary Agent 与 Subagent

OpenCode 官方文档把 agent 分成两类：

1. Primary agents：用户直接交互的主 agent。
2. Subagents：由主 agent 调用或用户通过 mention 调用的专用 agent。

内置 primary agents 包括：

1. Build：默认开发 agent，拥有较完整工具能力。
2. Plan：规划和分析 agent，默认对文件编辑和 bash 更谨慎。

内置 subagents 包括：

1. General：通用多步任务 agent。
2. Explore：快速只读代码探索 agent。
3. Scout：外部文档和依赖研究 agent。

还有隐藏系统 agent：

1. Compaction：用于上下文压缩。
2. Title：用于生成 session 标题。
3. Summary：用于生成 session 摘要。

这个设计体现了一个重要趋势：生产级 coding agent 不再是“一个万能 agent 干所有事”，而是把不同任务拆成不同权限、不同 prompt、不同模型、不同 step budget 的 agent。

## 12.9 Plan Mode 和 Build Mode

OpenCode 官方文档建议添加功能前可以先使用 Plan mode。Plan mode 会限制修改能力，让 agent 先分析和提出方案。

这对应真实工程里的两阶段工作流：

1. Plan：理解需求、读代码、提出方案、识别风险。
2. Build：修改文件、运行命令、验证结果。

这两个模式的区别不是模型不同那么简单，而是权限和目标不同。

Plan mode 的价值：

1. 降低误修改风险。
2. 让用户先审查方案。
3. 适合复杂需求、重构和高风险任务。
4. 帮助 agent 在动手前建立上下文。

Build mode 的价值：

1. 执行具体改动。
2. 跑测试和验证。
3. 形成最终 diff。
4. 总结完成情况。

面试中可以说：Plan/Build 不是 UI 小功能，而是一种权限分层和工作流分阶段设计。

## 12.10 工具系统

OpenCode 官方文档列出的内置工具包括：

1. `bash`：执行 shell 命令。
2. `edit`：对已有文件做精确替换。
3. `write`：创建或覆盖文件。
4. `read`：读取文件内容。
5. `grep`：正则搜索文件内容。
6. `glob`：按 pattern 查找文件。
7. `lsp`：实验性代码智能能力。
8. `apply_patch`：应用 patch。
9. `skill`：加载 skill。
10. `todowrite`：管理任务列表。
11. `webfetch`：获取网页内容。
12. `websearch`：搜索网页内容。
13. `question`：向用户提问。

这些工具覆盖了 coding agent 的核心动作：

1. 理解代码：read、grep、glob、lsp。
2. 修改代码：edit、write、apply_patch。
3. 验证代码：bash。
4. 获取外部知识：webfetch、websearch、MCP。
5. 管理任务：todowrite。
6. 和用户协作：question。
7. 加载专门能力：skill。

这说明 OpenCode 的工具系统已经不是简单 function calling，而是 coding runtime 的动作空间。

## 12.11 工具注册和工具权限

OpenCode 的工具不是只靠“是否启用”控制。根据官方文档，新版本更推荐通过 `permission` 字段控制工具行为。

权限动作有三类：

1. `allow`：直接允许。
2. `ask`：执行前询问用户。
3. `deny`：直接拒绝。

权限可以按工具配置，也可以按输入模式细分。例如：

```json
{
  "permission": {
    "bash": {
      "*": "ask",
      "git status *": "allow",
      "rm *": "deny"
    },
    "edit": "ask"
  }
}
```

这个能力很关键。因为工具风险不只取决于工具名，还取决于参数。

例如同样是 bash：

1. `git status` 风险低。
2. `npm test` 通常可接受。
3. `rm -rf` 风险高。
4. `git push` 可能影响远端。
5. 部署命令可能影响生产环境。

成熟 agent harness 必须能按输入参数做权限判断。

## 12.12 默认权限和安全边界

OpenCode 官方文档提到，默认情况下多数权限偏开放，但 `external_directory` 和 `doom_loop` 默认更谨慎，`.env` 相关读取也有默认拒绝策略。

这体现了几个安全边界：

1. 工作区边界：不应随意访问项目外目录。
2. 敏感文件边界：环境变量文件通常不能读。
3. 循环边界：同一工具反复调用可能表示 agent 卡住。
4. 用户确认边界：高风险动作应进入 ask。

对面试来说，重点不是背 OpenCode 默认值，而是理解：agent runtime 必须把权限做在执行层，而不是只靠 prompt。

Prompt 可以告诉模型“不要做危险事”，但真正的拒绝必须由 permission engine 执行。

## 12.13 文件读写和 Patch Editing

OpenCode 支持多种文件操作：

1. `read` 读取文件。
2. `edit` 做精确字符串替换。
3. `write` 创建或覆盖文件。
4. `apply_patch` 应用 patch。
5. server API 提供文件查找、读取、状态和 session diff。

这说明 OpenCode 对文件编辑的抽象不是单一路径。

不同编辑方式适合不同任务：

1. 小范围修复适合 `edit`。
2. 新增文件适合 `write`。
3. 多文件结构化改动适合 `apply_patch`。
4. 用户审查适合 diff。

文件编辑必须配合 snapshot 和 revert。官方文档提到 snapshot 用于跟踪 agent 操作中的文件变化，使用户能够 undo/revert。对于 coding agent 来说，这是非常重要的安全网。

没有 snapshot 的 agent 一旦改错，只能依赖 git 或用户手动恢复；有 snapshot 的 agent 可以把“这轮会话造成了哪些变化”作为 session 层状态管理。

## 12.14 Shell Execution

OpenCode 的 `bash` 工具让模型可以执行项目环境中的 shell 命令。

这让 agent 能完成闭环：

1. 修改代码。
2. 运行测试。
3. 查看错误。
4. 再修改。
5. 再验证。

但 shell execution 也是最大风险来源之一。

安全设计要考虑：

1. 命令白名单和黑名单。
2. 参数 pattern。
3. 工作目录。
4. 外部目录访问。
5. 网络访问。
6. 删除、部署、提交、推送等高风险命令。
7. 超时和输出截断。
8. 环境变量和 secret。

OpenCode 的 permission object syntax 可以对 bash 命令做模式匹配，这正是终端型 coding agent 需要的能力。

## 12.15 Search、Glob 和 LSP

OpenCode 内置 `grep` 和 `glob`，官方文档说明它们底层使用 ripgrep，并默认尊重 `.gitignore`。

这对大仓库非常重要。

代码理解通常不是先读全仓库，而是：

1. glob 找文件。
2. grep 找符号和调用点。
3. read 读取关键文件片段。
4. lsp 获取 definition、references、hover、symbol 等结构化信息。

如果一个 coding agent 没有高效搜索能力，它很容易把上下文浪费在不相关文件上。

LSP 能力进一步把 agent 从文本搜索推进到语义导航：找定义、找引用、看符号、看调用层级。这对大型 TypeScript、Go、Python、Java 项目尤其有价值。

## 12.16 MCP 集成

OpenCode 官方文档支持 MCP，并支持本地 MCP 和远程 MCP。

MCP 的意义是：把外部系统工具化接入 agent runtime。

常见 MCP 场景：

1. 查询 GitHub issue。
2. 查询 Sentry 错误。
3. 搜索外部文档。
4. 访问数据库。
5. 调用内部平台。
6. 访问云资源。

OpenCode 文档也提醒：MCP server 会增加上下文成本，工具太多可能快速占满上下文窗口。

这说明 MCP 集成不只是“接得越多越好”。真实系统需要：

1. MCP 工具筛选。
2. 按 agent 启用 MCP。
3. 禁用不相关 MCP。
4. 控制 MCP 工具权限。
5. 控制远程 MCP 鉴权。
6. 避免工具描述污染上下文。

OpenCode 支持按 server 名称前缀控制 MCP 工具，例如禁用某个 MCP server 下所有工具。这是 tool namespace 设计的体现。

## 12.17 Custom Tools

OpenCode 支持在 `.opencode/tools/` 或全局配置目录中定义自定义工具。

官方文档说明工具定义用 TypeScript 或 JavaScript 写，但工具内部可以调用任意语言脚本。例如 TypeScript 工具可以调用 Python 脚本。

这对企业落地很关键。

企业里经常有内部系统：

1. 数据库查询。
2. 日志平台。
3. 实验平台。
4. 训练平台。
5. 发布平台。
6. 内部代码搜索。

如果每个系统都靠 prompt 让模型“猜怎么用”，可靠性会很差。更好的方式是封装成 custom tool，给模型一个稳定 schema。

自定义工具还会收到上下文信息，例如 agent、sessionID、messageID、directory、worktree。这意味着工具可以知道当前会话和工作区，而不是孤立函数。

## 12.18 Agents 与 Tool Access 的组合

OpenCode 的 agent 可以配置：

1. description。
2. mode。
3. model。
4. prompt。
5. temperature。
6. steps。
7. permission。
8. task permission。
9. hidden。

这让 agent 不只是 prompt 模板，而是具备自己的权限和运行策略。

例如：

1. Review agent：只能读代码，不能 edit。
2. Docs agent：可以写文档，但不能 bash。
3. Debug agent：可以 read、grep、bash，但 edit 需要 ask。
4. Security auditor：不能改代码，只能提出风险。
5. Explore agent：只读探索代码库。

这种设计的好处是最小权限原则。

不同任务需要不同权限，不应该让所有 agent 都拥有完整 shell 和写文件能力。

## 12.19 Compaction、Summary 和长上下文

OpenCode 文档中有 compaction 配置，也有隐藏 compaction agent、summary agent。

这说明它把长上下文管理作为 runtime 能力，而不是完全交给用户。

长任务中会出现：

1. 文件片段越来越多。
2. 命令输出越来越长。
3. 工具调用历史越来越多。
4. 用户追加要求越来越多。
5. 模型上下文窗口逐渐耗尽。

Compaction 的目标是把旧上下文压缩成仍然有用的任务状态。

一个好的 compaction 不应该只压缩聊天，而要保留：

1. 用户目标。
2. 当前计划。
3. 已修改文件。
4. 当前 diff。
5. 关键错误日志。
6. 重要决策。
7. 未完成任务。

这和第 7 章的上下文管理设计一致。

## 12.20 Server 架构

OpenCode 官方 server 文档给了一个很关键的信息：运行 `opencode` 时会启动 TUI 和 server，TUI 是和 server 通信的 client。`opencode serve` 可以启动 headless HTTP server，并暴露 OpenAPI 3.1 spec。

这意味着 OpenCode 的 UI 和 runtime 是解耦的。

这种架构的好处：

1. TUI、IDE、Web、SDK 可以共享同一个 runtime。
2. 外部程序可以通过 HTTP API 控制 OpenCode。
3. 可以生成类型安全 SDK。
4. 可以订阅事件流。
5. 可以用 server API 做自动化和集成。

Server API 覆盖：

1. health 和 event。
2. project、path、vcs。
3. config、provider、auth。
4. sessions、messages、commands。
5. files、find、symbols。
6. tools、LSP、formatters、MCP。
7. agents、logs、TUI 控制。
8. permission response。

这说明 OpenCode 的 runtime 是可编程的，而不是只能通过一个终端 UI 操作。

## 12.21 Session、Message、Diff 和 Revert

OpenCode server API 明确有 session、message、diff、fork、abort、share、summarize、revert、unrevert、permission response 等接口。

这可以抽象成 session state machine：

```text
session created
-> user message
-> assistant message with tool calls
-> file changes / command outputs
-> diff generated
-> permission requests if needed
-> user approves or rejects
-> session continues, aborts, forks, shares, summarizes, reverts
```

Session 是 coding agent 的核心状态容器。

它不仅保存对话，还保存：

1. 子会话。
2. todo。
3. diff。
4. 权限请求。
5. 消息 parts。
6. share 状态。
7. summary。
8. revert 状态。

这比普通聊天机器人复杂得多。

## 12.22 SDK 与程序化集成

OpenCode 官方 SDK 是 TypeScript/JavaScript client，可以启动 OpenCode server，也可以连接已有 server。

SDK 的价值：

1. 把 OpenCode 嵌入自动化流程。
2. 构建自定义客户端。
3. 连接 CI、review bot、内部平台。
4. 用类型安全方式操作 session 和 file API。
5. 使用 structured output 获取结构化结果。

对于 agent harness 来说，SDK 意味着 runtime 不只是面向人，也可以面向程序。

例如企业可以构建：

1. PR 自动分析机器人。
2. 代码迁移批处理 agent。
3. 文档同步 agent。
4. 安全扫描辅助 agent。
5. 内部 IDE 插件。

这些都需要稳定 API，而不是只靠终端交互。

## 12.23 和 Claude Code 的对比

从公开资料和产品形态看，OpenCode 与 Claude Code 类产品可以从几个角度比较。

开放性：

1. OpenCode 强调开源、配置、server、SDK、custom tools、plugins、MCP。
2. Claude Code 类产品更强调官方模型体验和产品闭环。

工具抽象：

1. OpenCode 官方文档公开列出 built-in tools、custom tools、MCP tools 和 permission 管理。
2. Claude Code 类产品也有工具系统，但内部实现公开程度取决于官方披露。

权限模型：

1. OpenCode 明确用 allow、ask、deny 和 pattern rules 控制权限。
2. Claude Code 类产品也需要权限确认，但具体配置开放性可能不同。

扩展性：

1. OpenCode 的 MCP、自定义工具、plugins、SDK、server 让它更像可扩展 runtime。
2. Claude Code 类产品更像强产品体验优先的 coding agent。

上下文和会话：

1. OpenCode 文档明确有 compaction、summary、session、diff、revert。
2. 这类能力在成熟 coding agent 中都需要存在，但 OpenCode 更适合作为公开学习样本。

面试中不要简单说谁更强，而要说：两者代表不同设计取向，一个偏产品闭环，一个偏开放 runtime 和可配置 harness。

## 12.24 真实工程坑

OpenCode 这类开放 coding agent 在工程落地中常见坑包括：

1. 默认权限过宽，agent 可以自动执行高风险命令。
2. 项目级配置和全局配置合并后产生意外权限。
3. MCP server 太多，工具描述占用大量上下文。
4. 自定义工具没有输入校验，变成新的安全洞。
5. Agent 权限没有区分，review agent 也能写文件。
6. Plan mode 和 Build mode 边界不清，复杂任务直接开写。
7. Snapshot 被关闭后，用户失去会话级 undo 能力。
8. Compaction 丢掉关键错误日志或当前 diff。
9. Server 暴露到网络但没有鉴权。
10. 项目规则文件被 prompt injection 污染，runtime 没有权限兜底。

这些坑都说明：开放能力越强，治理和默认安全越重要。

## 12.25 面试题

### 题 1：如何从 harness 角度分析 OpenCode？

参考回答：

```text
我会把 OpenCode 看成一个可扩展的 coding agent runtime，而不是单纯 CLI。它有多客户端入口，比如 TUI、IDE、Web 和 SDK；中间是 server、session、message、agent manager、context builder、tool registry、permission engine 和 execution engine；底层连接文件系统、shell、search、LSP、MCP 和自定义工具。它的配置系统、agents、permissions、MCP、custom tools、snapshot、diff、revert 和 server API 都体现了 harness 思想。
```

### 题 2：OpenCode 的配置系统为什么重要？

参考回答：

```text
因为 coding agent 的行为不只由模型决定，还由模型选择、工具集合、权限策略、项目规则、MCP、custom tools、agents、compaction 和 server 设置决定。OpenCode 把这些放进 JSON/JSONC 配置，并支持全局、项目、远程、管理配置合并，相当于把 agent runtime 的控制面配置化。这对个人使用和企业治理都很重要。
```

### 题 3：OpenCode 的 permission 设计解决什么问题？

参考回答：

```text
它解决模型工具调用的安全边界问题。OpenCode 把工具动作分成 allow、ask、deny，并支持按工具名和输入 pattern 做细粒度控制。例如 bash 可以允许 git status，询问 npm test，拒绝 rm 命令。这个边界在 runtime 层执行，比只靠 prompt 告诉模型不要做危险事可靠。
```

### 题 4：为什么 MCP 工具不能无限接入？

参考回答：

```text
MCP 能把外部系统接入 agent，但每个 MCP server 和工具描述都会增加上下文成本，也会扩大权限和安全面。工具太多会让模型选择困难、上下文膨胀，并增加误调用风险。因此需要按 agent、任务和权限启用 MCP，而不是默认全开。
```

### 题 5：OpenCode server/SDK 架构有什么价值？

参考回答：

```text
Server/SDK 让 OpenCode 从终端产品变成可编程 runtime。TUI、IDE、Web 和外部程序都可以通过 server API 操作 session、message、file、diff、permission 和 events。这样企业可以基于它做 PR bot、代码迁移 agent、内部 IDE 集成或自动化评估，而不局限于人手动在终端里输入 prompt。
```

## 12.26 小练习

1. 设计一个 OpenCode 项目级 `opencode.json`，要求默认所有 bash 命令 ask，但 `git status` 和 `npm test` allow，`git push` 和删除命令 deny。
2. 设计一个只读 `review` agent，只允许 read、grep、glob、lsp，不允许 edit 和 bash。
3. 设计一个 MCP 接入方案：什么时候全局启用，什么时候只给某个 agent 启用。
4. 画出 TUI 发送 prompt 到 server，再到 session、agent、tool、permission、diff 的链路图。
5. 分析 snapshot、diff、revert 在 coding agent 安全体验中的作用。

## 12.27 本章总结

本章基于 OpenCode 官方文档分析了 OpenCode 的 agent harness 架构。

核心结论：

1. OpenCode 是一个开放、可配置、可扩展的 coding agent runtime。
2. 它的配置系统相当于 runtime 控制平面，管理模型、工具、权限、agents、MCP、instructions、compaction 等策略。
3. 它的 agent 系统把任务拆成 primary agents、subagents 和隐藏系统 agents，体现了权限分层和任务专门化。
4. 它的工具系统覆盖 read、grep、glob、edit、write、apply_patch、bash、lsp、skill、todo、web、question 等核心 coding 动作。
5. Permission engine 是安全边界，必须在 runtime 层执行 allow、ask、deny。
6. MCP、自定义工具、plugins、server 和 SDK 让 OpenCode 从单一产品变成可扩展平台。
7. Snapshot、diff、revert、session、summary、compaction 是长任务和可恢复性的基础。
8. 面试中分析 OpenCode，应强调它如何把模型、工具、权限、上下文和扩展接口组织成一个 agent harness。

下一章会进入 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 等主流 coding agent 的横向对比，重点看产品形态和 harness 设计差异。
