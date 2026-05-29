# 第十三章：Codex 与主流 Coding Agent 对比

## 13.1 本章定位

前面两章分别分析了 Claude Code 和 OpenCode。本章进入横向比较：Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 这些 coding agent 或 coding assistant 到底有什么差异。

本章不做简单产品测评，也不排名谁更强。因为 coding agent 的能力不是单一模型能力决定的，而是由模型、上下文工程、工具系统、编辑方式、权限模型、执行环境、评估体系和产品入口共同决定。

学完本章，你应该能回答：

1. 为什么主流 coding agent 的差异主要体现在 harness 设计上。
2. Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 分别代表什么设计取向。
3. CLI、IDE、Web、Cloud、SDK 这些产品形态会如何影响 agent runtime。
4. repo map、rules、AGENTS.md、CLAUDE.md、terminal、issue、test output 分别如何进入上下文。
5. patch editing、IDE edit、architect/editor、sandbox execution、SWE-bench runner 的差异是什么。
6. 面试中如何系统比较多个 coding agent，而不是停留在“某个产品好用”层面。

## 13.2 资料来源和可信边界

本章主要参考以下公开资料：

1. OpenAI `openai/codex` 官方 GitHub 仓库 README：Codex CLI 是 OpenAI 的本地 coding agent，支持 CLI、IDE、desktop app、Codex Web 等入口。
2. Anthropic Claude Code 官方文档：Claude Code 是能读取代码库、编辑文件、运行命令并集成开发工具的 agentic coding tool，支持 terminal、IDE、desktop app、browser、GitHub Actions、GitLab CI/CD、Agent SDK、sub-agents、skills、hooks、MCP 等能力。
3. OpenCode 官方文档：OpenCode 是开源 AI coding agent，支持 terminal、desktop、IDE、server、SDK、config、tools、permissions、agents、MCP、custom tools、plugins、session、diff、snapshot、undo 等能力。
4. Aider 官方文档：Aider 是终端里的 AI pair programming 工具，强调 git repo、repo map、chat modes、architect/editor、edit formats、lint/test、prompt caching 等。
5. SWE-agent 官方文档和 GitHub 仓库：SWE-agent 面向自动修复 GitHub issue、SWE-bench、研究和可配置实验，强调 agent-computer interface、YAML 配置、tools、environment、trajectory 等。
6. OpenHands 官方 GitHub 仓库：OpenHands 是 AI-driven development 平台，提供 Software Agent SDK、CLI、Local GUI、Cloud、Enterprise、REST API、React UI、RBAC、协作和集成能力。
7. Cursor 官方文档入口：Cursor 是 IDE 内 coding assistant/agent，文档覆盖 Agent、Rules、MCP、Skills、CLI 等能力。部分页面抓取结果较少，本章只采用公开且可确认的高层能力，不推断未公开实现细节。

需要注意：不同产品更新很快，闭源产品的内部实现细节不能从外部完全确认。因此本章重点比较公开能力和可验证的架构取向，不把社区传闻写成确定事实。

## 13.3 为什么不能只比较模型

很多人比较 coding agent 时会说：某个产品更聪明，某个模型更会写代码。

这只说对了一部分。真实 coding agent 至少由七层组成：

```text
User / Issue / IDE Selection / CLI Prompt
-> Context Builder
-> Agent Policy / Planning Loop
-> Tool Registry
-> Permission / Sandbox / Execution Environment
-> File Editing / Diff / Patch / IDE Apply
-> Verification / Test / Trace / Review
```

模型只在中间负责推理和生成动作。其他层决定它能看到什么、能做什么、怎么改文件、能不能运行测试、出了错能不能恢复。

两个系统使用同一个模型，表现也可能差很多：

1. 一个系统有 repo map，另一个只能靠用户手动粘文件。
2. 一个系统能运行测试并读取错误，另一个只能给建议。
3. 一个系统有 patch apply 和 diff review，另一个只输出整段代码。
4. 一个系统有权限沙箱，另一个默认执行所有命令。
5. 一个系统有 trajectory 和 replay，另一个没有可复现记录。

因此，本章比较的核心不是“模型榜单”，而是 harness 设计。

## 13.4 对比总表

可以先用一张表建立整体印象。

| 系统 | 主要形态 | 核心取向 | 典型强项 | 主要风险或边界 |
|---|---|---|---|---|
| Codex | CLI、IDE、desktop app、Codex Web | OpenAI 本地与云端 coding agent | 轻量本地 CLI、OpenAI 模型和 ChatGPT 生态入口 | 公开文档能确认产品形态，但内部 harness 细节需谨慎判断 |
| Claude Code | Terminal、IDE、desktop、web、CI、SDK | 产品化 agentic coding tool | 多入口、工具执行、MCP、skills、hooks、sub-agents、Agent SDK | 闭源能力多，内部实现边界不能过度推断 |
| OpenCode | Terminal、desktop、IDE、server、SDK | 开源可配置 runtime | config、permissions、agents、MCP、custom tools、server API、snapshot/diff/revert | 开放能力越强，权限治理和配置复杂度越高 |
| Cursor | IDE | IDE-native coding assistant/agent | 编辑器上下文、selection、rules、MCP、skills、交互式编辑体验 | 强依赖 IDE 场景，批处理和研究复现实验不是核心形态 |
| Aider | Terminal | Git repo 中的 AI pair programming | repo map、chat modes、architect/editor、edit formats、git 集成、lint/test | 更偏本地结对编程，不是完整企业平台 |
| SWE-agent | CLI、研究框架、benchmark runner | 自动修复 issue 和研究评测 | SWE-bench、trajectory、YAML 配置、agent-computer interface、可复现实验 | 面向研究和任务自动化，日常 IDE 体验不是核心 |
| OpenHands | SDK、CLI、Local GUI、Cloud、Enterprise | AI-driven development 平台 | Software Agent SDK、GUI、REST API、Cloud、Enterprise、RBAC、协作和集成 | 平台复杂度高，部署和治理成本也更高 |

这张表背后的关键是：每个系统优化的不是同一个目标。

Codex、Claude Code、OpenCode 更像通用 coding agent；Cursor 更像 IDE 原生助手；Aider 更像终端结对编程工具；SWE-agent 更像研究和 benchmark agent；OpenHands 更像可部署的软件工程 agent 平台。

## 13.5 Codex：OpenAI 的轻量本地 Coding Agent

根据 `openai/codex` 官方仓库，Codex CLI 是 OpenAI 的 coding agent，可以在本地电脑运行。

公开 README 提到几个关键形态：

1. Codex CLI：本地终端 agent。
2. IDE 插件：面向 VS Code、Cursor、Windsurf 等编辑器。
3. Desktop app：可通过 `codex app` 或 Codex App 页面进入。
4. Codex Web：云端 agent 入口。
5. 支持通过 ChatGPT 账号或 API key 使用。

从 harness 角度看，Codex 的重要意义是：OpenAI 把代码生成能力从“聊天窗口回答代码”推进到“本地运行、读写 repo、执行开发任务”的 agent runtime。

可以把 Codex 抽象成：

```text
CLI / IDE / App / Web
-> Codex Runtime
-> Local workspace or cloud workspace
-> Model reasoning
-> File edits / commands / diffs / task results
```

本地 CLI 的优势：

1. 离代码仓库近，能直接进入项目目录。
2. 适合开发者用自然语言驱动小到中等任务。
3. 能和本地 shell、git、测试命令形成闭环。
4. 安装和启动成本较低。

Web 或 cloud 形态的优势：

1. 可以处理长任务。
2. 可以脱离本地机器执行。
3. 更适合多人协作、远程任务和异步开发。
4. 可以与 ChatGPT 账号体系和云端工作流结合。

面试中不要把 Codex 只理解为“代码补全模型”。在当前语境下，它更接近一个 coding agent 产品线：CLI 解决本地交互，IDE 解决编辑器嵌入，Web/Cloud 解决异步任务和远程执行。

## 13.6 Claude Code：强产品闭环的 Agentic Coding Tool

Claude Code 官方文档把它定义为 agentic coding tool：能读取代码库、编辑文件、运行命令，并和开发工具集成。

公开文档显示它支持多种入口：

1. Terminal CLI。
2. VS Code、Cursor、JetBrains 等 IDE 插件。
3. Desktop app。
4. Web。
5. GitHub Actions 和 GitLab CI/CD。
6. Slack、remote control、routines、browser 等工作流。
7. Agent SDK。

Claude Code 的公开能力中，最值得从 harness 角度关注的是：

1. `CLAUDE.md`：项目级和团队级 instruction/memory。
2. 自动 memory：保存构建命令、调试经验等会话外知识。
3. MCP：连接外部数据源和工具系统。
4. Skills：封装团队可复用工作流。
5. Hooks：在 action 前后运行 shell 命令，例如格式化、lint 或提交前检查。
6. Sub-agents 和 background agents：并行处理不同任务。
7. Agent SDK：构建自定义 agent 和编排逻辑。
8. 多入口 session：终端、IDE、desktop、web 之间迁移任务。

这代表一种产品闭环取向：不只是让 agent 能改代码，而是让它融入开发者日常工作流。

从系统设计看，Claude Code 的重点是：

```text
Developer workflow
-> Claude Code surface
-> Shared coding engine
-> Project memory / skills / hooks / MCP
-> File edit / command execution / git / CI
-> Review / PR / automation
```

它的优势不是某一个工具，而是多入口、多工作流和产品体验整合。比如同一个任务可以从 terminal 开始，在 desktop 里看 diff，在 web 上继续，或者接到 GitHub Actions 做自动 review。

面试回答时可以说：Claude Code 类系统的护城河不只是模型，而是“模型 + 工具 + workflow + memory + permissions + integration”的完整闭环。

## 13.7 OpenCode：开源、可配置、可扩展 Runtime

上一章已经详细分析 OpenCode，这里只做横向定位。

OpenCode 的公开文档显示它是开源 AI coding agent，可用作 terminal interface、desktop app 或 IDE extension，并且有 server、SDK、config、tools、permissions、agents、MCP、custom tools、plugins、snapshot、diff、undo 等能力。

它和 Claude Code 的不同在于：OpenCode 更适合作为可学习、可扩展、可配置的公开 runtime 样本。

OpenCode 的关键设计取向包括：

1. 配置系统是控制平面。
2. Permission engine 用 allow、ask、deny 控制工具风险。
3. Primary agents 和 subagents 体现任务专门化。
4. MCP 和 custom tools 扩展外部工具生态。
5. Server 和 SDK 让 runtime 可编程。
6. Snapshot、diff、revert 支撑可审查和可恢复。

如果要设计一个企业内部 coding agent，OpenCode 提供了很多可借鉴模块：

1. 全局配置和项目配置如何合并。
2. agent 级权限如何覆盖默认权限。
3. custom tool 如何拿到 session、directory、worktree 等上下文。
4. TUI 和 server 如何解耦。
5. session、message、diff、permission response 如何暴露成 API。

它的风险也明显：开放能力越多，治理越难。MCP、plugins、custom tools、server、SDK 都是扩展点，同时也是安全边界。

## 13.8 Cursor：IDE Native Agent

Cursor 的定位和终端 agent 不完全一样。它是 IDE 原生 coding assistant/agent，核心优势来自编辑器上下文。

IDE 内 agent 能天然获得：

1. 当前打开文件。
2. 光标位置和选区。
3. 当前诊断、lint、类型错误。
4. 跳转定义、引用、符号搜索。
5. 编辑器 diff 和 apply。
6. 用户正在看的代码区域。

这和 CLI agent 的上下文入口不同。CLI agent 通常要靠 `grep`、`glob`、`read`、`bash` 去主动探索；IDE agent 可以直接利用用户已经建立的局部上下文。

Cursor 文档入口显示其能力覆盖 Agent、Rules、MCP、Skills、CLI 等。这说明 IDE agent 也在向完整 harness 演进，而不是只做补全。

Cursor 类系统的典型优势：

1. 适合局部代码修改和交互式编辑。
2. 能自然使用 selection、diagnostics、symbols。
3. 用户可以实时审查 diff。
4. 对日常开发体验友好。
5. rules 能把项目规范注入上下文。

边界也很清楚：

1. 对大规模异步任务，IDE UI 不一定是最佳载体。
2. 对 benchmark 和可复现实验，IDE 交互记录通常不如 trajectory runner 明确。
3. 对企业批处理迁移，可能需要 CLI、server 或 SDK 形态补充。

因此，Cursor 代表的是 IDE-native harness：上下文来自编辑器，动作落在编辑器，用户审查也发生在编辑器。

## 13.9 Aider：Git Repo 中的终端结对编程

Aider 官方文档把它定位为终端里的 AI pair programming 工具，工作对象是本地 git repo。

它有几个非常值得学习的设计点。

第一个是 repo map。

Aider 会构建一个简洁的仓库地图，包含关键文件、类、函数、类型和调用签名，并根据 token budget 选择最相关部分发送给模型。这解决了一个核心问题：大仓库不能全部塞进上下文，但 agent 又需要知道整体结构。

repo map 的价值：

1. 让模型看到全仓库的骨架。
2. 帮助模型判断应该打开哪些文件。
3. 降低无关上下文浪费。
4. 让编辑更符合现有抽象。

第二个是 chat modes。

Aider 支持：

1. `code`：直接修改代码。
2. `ask`：讨论代码但不修改。
3. `architect`：先由 architect model 提方案，再由 editor model 转成具体编辑。
4. `help`：回答 aider 使用问题。

这和前面讲的 Plan/Build 非常接近，但 Aider 的 architect/editor 更明确地区分了“方案推理”和“文件编辑”。

第三个是 edit formats。

Aider 很重视让模型以可解析格式表达修改，而不是只生成自然语言代码块。因为文件编辑失败往往不是模型不会写代码，而是 edit format、上下文定位、冲突处理出了问题。

Aider 代表一种小而强的设计取向：

```text
local git repo
-> repo map
-> ask/code/architect modes
-> edit format
-> git diff / lint / test
```

它不一定追求完整平台化，但在“本地仓库 + 终端 + git + 精确编辑”这个场景里很有代表性。

## 13.10 SWE-agent：研究和 Benchmark 取向的 Agent Framework

SWE-agent 的目标不是普通 IDE 助手，而是让语言模型自主使用工具修复真实 GitHub issue，并能跑 SWE-bench 等 benchmark。

官方文档强调：

1. 面向真实 GitHub repository issue。
2. 支持自动修复、网络安全 challenge、coding challenge 等任务。
3. 使用 YAML 配置管理模型、工具、环境和 agent 行为。
4. 面向研究，simple、hackable、configurable。
5. 有 trajectory inspector、batch mode、competitive runs 等能力。
6. 当前推荐 mini-swe-agent，强调更简单但性能接近。

SWE-agent 的关键贡献是 agent-computer interface 思想。也就是说，不只是模型强不强，还要看 agent 和计算机环境之间的接口是否适合软件工程任务。

在 SWE-bench 场景里，agent 需要：

1. 读取 issue 描述。
2. 探索 repo。
3. 编辑文件。
4. 运行测试。
5. 处理失败。
6. 生成 patch。
7. 记录 trajectory。
8. 在 batch 中复现实验。

这和产品型 coding agent 不同。产品型 agent 更关心交互体验，SWE-agent 更关心可复现、可评估、可配置。

面试中可以这样总结：

```text
SWE-agent 代表研究型 coding agent harness。它关心如何把 GitHub issue、repo、tool interface、execution environment、trajectory 和 benchmark runner 组织成可复现实验，用于评估 agent 是否真的能解决软件工程问题。
```

## 13.11 OpenHands：从 Agent 到开发平台

OpenHands 的官方仓库把它定位为 AI-driven development。它不只是一个 CLI，而是一组平台能力。

公开 README 提到几种形态：

1. Software Agent SDK：可组合的 Python library，是底层 agentic tech engine。
2. CLI：类似 Claude Code 或 Codex 的命令行体验，可使用 Claude、GPT 或其他 LLM。
3. Local GUI：本地运行 agent，带 REST API 和 React 单页应用。
4. Cloud：托管版本，支持 GitHub/GitLab 登录。
5. Enterprise：自托管到企业 VPC/Kubernetes，包含企业集成和治理能力。

OpenHands Cloud 还提到：

1. Slack、Jira、Linear 集成。
2. 多用户支持。
3. RBAC 和 permissions。
4. 协作能力，例如 conversation sharing。

这说明 OpenHands 更像“软件工程 agent 平台”，而不是单一编码助手。

从 harness 角度看，它覆盖了更完整的平台层：

```text
SDK / CLI / GUI / Cloud / Enterprise
-> Agent runtime
-> Workspace / container / environment
-> REST API / frontend
-> Integrations / RBAC / collaboration
-> Benchmark / deployment / enterprise governance
```

这种系统适合：

1. 企业内部 agent 平台。
2. 多人协作的软件工程 agent。
3. 云端异步开发任务。
4. 自托管和合规要求较强的场景。
5. 研究、评估、产品形态统一的平台。

代价也很明显：平台复杂度、部署复杂度、权限治理和运维成本都比单机 CLI 高。

## 13.12 产品形态对比：CLI、IDE、Web、Cloud、SDK

不同产品形态会决定 agent 的能力边界。

CLI 的特点：

1. 离本地 shell 和 git 最近。
2. 适合执行命令、跑测试、快速修改。
3. 对开发者透明，容易和现有工具链组合。
4. UI 审查能力通常弱于 IDE 或 desktop。

IDE 的特点：

1. 天然拥有光标、选区、diagnostics、symbols。
2. 适合局部编辑和实时交互。
3. 用户审查 diff 更方便。
4. 不一定适合长时间无人值守任务。

Web/Cloud 的特点：

1. 适合异步长任务。
2. 可以运行在托管环境中。
3. 容易接入 issue、PR、CI、chat 平台。
4. 需要更强 sandbox、secret、权限和资源隔离。

Desktop app 的特点：

1. 可以比终端提供更强可视化 diff 和多 session 管理。
2. 可以结合本地文件和远程 session。
3. 适合需要可视化审查但不想进 IDE 的用户。

SDK 的特点：

1. 面向程序化集成。
2. 可以构建自定义 agent、批处理、内部平台。
3. 更适合企业和研究者。
4. 要求 API 稳定、trace 清晰、权限可控。

因此，产品形态不是壳子，而是 harness 的入口层设计。

## 13.13 上下文工程对比

coding agent 的第一难题是：它到底看到了什么。

主流上下文来源包括：

1. 用户 prompt。
2. 当前文件和选区。
3. repo map 或符号索引。
4. `AGENTS.md`、`CLAUDE.md`、rules、项目配置。
5. git diff、branch、commit history。
6. issue、PR、评论、CI 日志。
7. terminal output、test output、lint output。
8. MCP 工具返回结果。
9. 历史 session summary 和 memory。

不同系统的侧重点不同：

1. Aider 强调 repo map。
2. Cursor 强调 IDE selection、diagnostics 和 rules。
3. Claude Code 强调 `CLAUDE.md`、memory、skills、MCP 和多入口 session。
4. OpenCode 强调 AGENTS.md、config、agents、MCP、session 和 compaction。
5. SWE-agent 强调 issue、repo、environment、trajectory 和 benchmark instance。
6. OpenHands 强调 workspace、SDK、GUI、cloud integrations 和平台上下文。

一个成熟 harness 不能只把所有东西塞给模型，而要做选择：

1. 当前任务需要哪些文件。
2. 哪些规则是真正高优先级。
3. 哪些工具结果需要保留。
4. 哪些日志只需要摘要。
5. 哪些上下文可能包含 prompt injection。
6. 哪些敏感信息不能进入模型。

上下文工程做不好，模型再强也会浪费在错误信息上。

## 13.14 编辑方式对比

coding agent 修改代码的方式大致有几类。

第一类是 whole-file generation。

模型重写整个文件。这种方式简单，但风险很高：

1. 容易覆盖用户未提到的内容。
2. 大文件成本高。
3. diff 不够精确。
4. 容易引入格式和无关改动。

第二类是 search/replace edit。

模型给出原片段和替换片段。优点是局部精确，缺点是原片段必须匹配，否则 apply 失败。

第三类是 unified diff 或 patch。

模型生成 patch，由 runtime apply。优点是适合多文件修改和审查，缺点是 patch 格式和上下文定位可能失败。

第四类是 IDE edit。

在编辑器里应用多处修改，用户可以直接看 diff。适合交互式开发。

第五类是 AST 或语义编辑。

理论上更稳，但实现成本高，而且跨语言支持复杂。多数产品仍主要依赖文本 diff、patch、LSP 辅助和测试验证。

从本书角度看，编辑方式的核心问题是：如何把模型输出转成可审查、可回滚、可验证的文件变化。

## 13.15 工具执行和权限模型对比

coding agent 的工具通常包括：

1. read、grep、glob、LSP。
2. edit、write、patch。
3. bash、test、lint、formatter。
4. git、PR、issue。
5. browser、web、MCP。
6. custom tools 和内部平台工具。

工具越多，能力越强，风险也越大。

权限模型可以分成几类：

1. 只读模式：只能分析，不能修改。
2. 手动确认：每次高风险操作问用户。
3. pattern allow/deny：按工具和参数规则控制。
4. sandbox：允许执行，但限制文件系统、网络和环境变量。
5. cloud workspace：在隔离容器或远程环境执行。
6. enterprise policy：组织级配置和 RBAC。

OpenCode 的 allow/ask/deny 是显式 runtime 权限；Claude Code 和 Codex 类产品通常也需要用户确认、sandbox 或工作区边界；OpenHands 企业形态会更强调 RBAC 和 permissions；SWE-agent 更强调实验环境和 benchmark runner 的可控性。

面试中要强调：权限不能只靠 prompt。Prompt 可以告诉模型不要执行危险命令，但真正的边界必须在 execution layer。

## 13.16 评估方式对比

coding agent 的评估也分多层。

最简单是人工体验：用户觉得是否有用。但它不可复现，也容易被 demo 误导。

更系统的评估包括：

1. 单元测试是否通过。
2. lint、typecheck、format 是否通过。
3. 真实 issue 是否解决。
4. patch 是否最小且可维护。
5. 是否引入安全问题。
6. 是否破坏已有行为。
7. 是否能在 benchmark 上复现。
8. 成本、延迟和失败率。

SWE-bench 是 coding agent 评估中最常被提到的 benchmark 之一，因为它把真实 GitHub issue 和 repo 结合起来，要求 agent 生成能通过测试的 patch。

但 SWE-bench 也不是全部：

1. 它不能覆盖所有产品交互体验。
2. 它和企业私有代码库分布不同。
3. 它更关注 issue fix，不完全覆盖新功能设计、架构重构、协作 review。
4. benchmark 可能存在污染和过拟合风险。

因此，企业落地时通常需要：

1. 公共 benchmark。
2. 私有 issue benchmark。
3. 回归测试集。
4. 人工 review。
5. 安全和权限测试。
6. 成本和延迟监控。

## 13.17 典型场景如何选型

如果目标是日常本地结对编程：

1. Aider、Codex CLI、Claude Code CLI、OpenCode TUI 都适合。
2. 重点看 repo context、编辑成功率、git 集成、测试闭环和成本。

如果目标是 IDE 内高频开发：

1. Cursor、Claude Code IDE、Codex IDE、OpenCode IDE 更适合。
2. 重点看 selection、diagnostics、inline diff、rules 和编辑器集成。

如果目标是异步修 issue 或自动 PR：

1. Codex Web、Claude Code Web/GitHub Actions、OpenHands Cloud、SWE-agent 更相关。
2. 重点看 sandbox、CI 集成、PR workflow、trace、失败恢复。

如果目标是研究 agent 能力：

1. SWE-agent、mini-swe-agent、OpenHands SDK 更适合。
2. 重点看可配置、trajectory、batch mode、benchmark reproducibility。

如果目标是企业内部平台：

1. OpenHands Enterprise、OpenCode server/SDK、自建 Claude Code/Codex workflow 都可能是候选。
2. 重点看 RBAC、审计、权限、私有部署、MCP/custom tools、数据边界和成本治理。

## 13.18 横向比较时最容易犯的错

常见误区包括：

1. 只比较模型，不比较上下文和工具。
2. 只看 demo，不看失败恢复。
3. 只看是否能改代码，不看是否能验证。
4. 只看单次任务，不看长任务和上下文压缩。
5. 只看开源或闭源标签，不看实际扩展点。
6. 只看 benchmark 分数，不看真实业务分布。
7. 忽略权限、secret、网络和外部目录风险。
8. 忽略用户体验和审查成本。
9. 把 IDE assistant、CLI agent、research runner、enterprise platform 混为一谈。
10. 把社区传闻当成官方能力。

面试中如果被问“你怎么看 Claude Code、Codex、Cursor、Aider 这些工具”，不要只说个人体验。更好的回答是：先按产品形态和 harness 层拆开，再比较上下文、工具、权限、编辑、验证和评估。

## 13.19 面试题

### 题 1：为什么 coding agent 的差异不只来自模型？

参考回答：

```text
因为 coding agent 是一个完整 runtime，而不是单次模型调用。模型只是负责推理和生成动作，真正决定效果的还有上下文构建、repo 搜索、文件编辑格式、工具注册、shell 执行、权限沙箱、测试验证、diff review、trace 和 session 管理。两个系统即使用同一个模型，如果一个有 repo map、测试闭环和安全权限，另一个只能生成代码块，最终表现会差很多。
```

### 题 2：Codex、Claude Code、OpenCode 的设计取向有什么不同？

参考回答：

```text
Codex 可以看成 OpenAI 的 coding agent 产品线，覆盖本地 CLI、IDE、desktop app 和 Codex Web，强调 OpenAI 模型与本地/云端开发工作流结合。Claude Code 更强调产品化闭环，支持 terminal、IDE、desktop、web、CI、MCP、skills、hooks、sub-agents 和 Agent SDK。OpenCode 则更像开源可配置 runtime，公开提供 config、permissions、agents、MCP、custom tools、server、SDK、snapshot、diff 和 revert，适合作为学习 agent harness 的样本。
```

### 题 3：Aider 的 repo map 解决什么问题？

参考回答：

```text
repo map 解决大仓库上下文选择问题。模型不能每次读取整个仓库，Aider 会构建包含关键文件、类、函数、签名和依赖关系的简洁仓库地图，并根据 token budget 选择最相关部分发给模型。这样模型能先理解仓库骨架，再决定要打开哪些具体文件，减少无关上下文浪费。
```

### 题 4：SWE-agent 和日常 IDE agent 的差异是什么？

参考回答：

```text
SWE-agent 更像研究和 benchmark 取向的 agent framework，目标是让模型在真实 GitHub repo 中根据 issue 自主探索、编辑、测试并生成 patch，重点是可配置、trajectory、batch mode 和 SWE-bench 复现。IDE agent 更关注开发者交互体验，例如当前选区、diagnostics、inline diff 和实时审查。两者都能写代码，但优化目标不同。
```

### 题 5：企业内部要落地 coding agent，应该重点比较哪些维度？

参考回答：

```text
我会比较七个维度：第一是上下文接入，包括 repo、issue、PR、CI 和内部文档；第二是工具和执行能力，包括 shell、测试、git、MCP 和内部 API；第三是权限和 sandbox，包括 secret、外部目录、网络和高风险命令；第四是编辑和 diff review；第五是 trace、replay 和审计；第六是评估体系，包括私有 issue benchmark、回归测试和人工 review；第七是成本、延迟、部署和 RBAC。企业场景不能只看模型效果。
```

## 13.20 小练习

1. 画一张表，比较 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 的产品形态、上下文来源、工具执行、权限模型和评估方式。
2. 选择一个你常用的代码仓库，设计一个 Aider 风格 repo map，列出哪些文件、类、函数应该进入 map。
3. 设计一个企业 coding agent 权限策略：哪些命令 allow，哪些 ask，哪些 deny。
4. 设计一个私有 SWE-bench 风格评估集：从历史 issue 中抽样，定义输入、期望 patch、测试和评分方式。
5. 比较 CLI agent 和 IDE agent 在“修复一个线上 bug”场景中的优缺点。
6. 设计一个 OpenHands 风格平台架构，包含 SDK、GUI、REST API、workspace、RBAC、trace 和 integrations。

## 13.21 本章总结

本章横向比较了 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、OpenHands 等主流 coding agent 或 coding assistant。

核心结论：

1. coding agent 的差异不只来自模型，更来自 harness 设计。
2. Codex 代表 OpenAI 本地和云端 coding agent 产品线。
3. Claude Code 代表强产品闭环、多入口、多工作流的 agentic coding tool。
4. OpenCode 代表开源、可配置、可扩展的 agent runtime。
5. Cursor 代表 IDE-native agent，优势在编辑器上下文和交互式审查。
6. Aider 代表终端里的 git repo 结对编程，repo map 和 edit format 很有启发。
7. SWE-agent 代表研究和 benchmark 取向，强调 issue、environment、trajectory 和可复现评估。
8. OpenHands 代表平台化方向，把 SDK、CLI、GUI、Cloud、Enterprise、RBAC 和协作集成到同一体系。
9. 企业落地时不能只看 demo，要看上下文、工具、权限、编辑、验证、评估、审计和成本治理。

下一章会进入 MCP、A2A 与 harness 集成，重点讲外部工具协议、跨 agent 通信和 runtime 扩展边界。
