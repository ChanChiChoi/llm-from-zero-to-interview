# 第十三章：Codex 与主流 Coding Agent 对比

## 0. 本讲范围与资料

本章以 **2026-11-18** 能查到的官方产品文档、项目文档和研究论文为准，挑选 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent/mini-SWE-agent 与 OpenHands 作案例。它们不是同一类产品，也不是穷尽市场的清单。逐项事实、来源日期和未验证边界见[本章资料与证据记录](../../research/coding-agent-product-comparison-evidence-2026-11-18.md)。第十一章和第十二章已经分别分析 Claude Code 与固定版本的 OpenCode；本章把重点放在如何对齐比较问题。

阅读时先分清四个层次：用户接触的产品入口（CLI、IDE、Web 等）、运行 Agent 的 harness、实际提供文件/命令/网络能力的执行环境，以及评估任务和 verifier。一个品牌可以同时覆盖多个入口；一个平台也可以承载不同 Agent。把层次分开后，再看上下文、编辑、权限、状态、验收和交接。

本章不报告产品实测结果或实时排名，也不把闭源 prompt、planner、路由、内部状态或隔离强度写成已知事实。文档支持的是公开合同；论文支持的是其指定任务与配置下的观察；两者都不等同于本地部署审计。本章中的认证修复任务是**教学构造**，没有运行这些产品。

## 13.1 本章定位

前面两章分别分析了 Claude Code 和 OpenCode。本章继续看 Codex、Cursor、Aider、SWE-agent/mini-SWE-agent 与 OpenHands，并沿一项小任务比较它们：仓库中的 `/api/profile` 对缺失 bearer token 返回 `200`，我们要修为 `401`、增加回归测试，而且不 push。此时不能只问“哪个 Agent 更强”，还要问用户启动的是哪个入口、代码在哪个执行环境、Agent 看到了什么、哪些动作会真的派发，以及结果由什么证据验收。

产品表现受到模型、工具和预算，也受到上下文、编辑方式、权限、执行环境、交互表面与验证流程影响。只看品牌或模型名称，不能判断这些条件是否相同；只把差异统称为 harness，也会把模型和产品的真实差异抹掉。

学完本章，你应该能回答：

1. 区分产品 surface、Agent/harness、执行环境和评估器。
2. 用同一条任务检查上下文来源、文件与命令动作、权限边界、状态记录和验收证据。
3. 说明 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent/mini-SWE-agent 与 OpenHands 的公开形态，以及资料没有揭示什么。
4. 区分“选用完整产品”与“研究某个 harness 因素”两种比较：前者要披露差异，后者要控制变量。
5. 根据具体任务和约束形成可复核的选择理由，不把局部证据汇成通用排名。

## 13.2 资料来源和可信边界

本章按产品官方文档和项目维护者资料描述公开入口与合同；有关 benchmark、Agent scaffolding 和 ACI 的论点则绑定到具体论文。链接集中放在对应产品小节和[证据记录](../../research/coding-agent-product-comparison-evidence-2026-11-18.md)，并标明来源能支持到哪里。OpenHands 仓库 `main` 与各产品动态文档会变化；这里的访问日期不代表读者当前安装版本。

## 13.3 为什么不能只比较模型

比较结果时，常见的第一句话是“这个 Agent 更聪明”。但同一任务的结果同时取决于模型、产品入口、上下文、工具、权限与运行环境。先把一次任务经过的对象画出来，才知道比较落在哪一层：

```text
User / Issue / IDE Selection / CLI Prompt
-> Public product surface
-> Agent or harness (partly hidden in commercial products)
-> Tool request and host-side policy
-> Local or remote execution environment
-> File changes and task evidence
-> Independent verification and human review
```

图中的层是**比较用的分析框架**，不是任何单一产品已公开的内部模块图。模型会读取上下文并提出动作；宿主是否派发、动作实际运行在哪里、改动怎样留下证据，则取决于公开合同和部署设置。对闭源产品，内部组件边界通常未知；对开源项目，读到某个版本的代码也只支持该版本中实际检查的路径。

即使两个实验使用同一模型，若一边只给 issue，另一边还提供认证 handler、相关测试和失败回执，模型能用的证据就不同；若一边可以修改并运行测试，另一边只能返回建议，执行路径也不同。这是帮助区分变量的概念例子，不是本章产品实测。反过来，完整产品比较中的模型和环境差异本来就是产品条件，不能为了强调 harness 而忽略它们。SWE-agent 论文在指定实验中研究了 Agent-Computer Interface（ACI）；较新的 scaffold 研究也报告实际交互可能偏离配置标签。它们支持把接口与运行轨迹纳入问题，不支持给所有产品排出固定优劣顺序（[SWE-agent 论文 v3](https://arxiv.org/abs/2405.15793v3)，[scaffolding 研究预印本](https://arxiv.org/abs/2608.08654)）。

## 13.4 对比总表

可以先用一张表建立整体印象。

| 比较对象 | 公开入口/层次 | 可观察的产品合同 | 需要单独核实 |
|---|---|---|---|
| Codex | CLI、IDE、desktop 与 Cloud 等不同 surface | 本地仓库路径与 OpenAI 管理计算机上的 Cloud 任务都公开存在 | 当前任务使用的 surface、sandbox/approval、workspace、网络和账号配置 |
| Claude Code | CLI、IDE、desktop、web、CI 等 | 多入口 agentic coding 产品 | 每个 surface 的实际执行位置，以及 shell sandbox 没覆盖的工具边界 |
| OpenCode V2 | TUI/IDE、server、SDK 等 | 可读源码、公开配置和 API 合同的开源 runtime | 安装版本、配置与项目自行提供的隔离；OpenCode 本身不提供 Agent sandbox |
| Cursor | 本地 IDE Agent 与 Cloud Agents | IDE 工作流和远端 VM 工作流并存 | 当前任务使用哪种入口、cloud 环境的网络/secret 设置 |
| Aider | 本地 Git 仓库中的终端 pair programming | repo map、ask/code/architect 等交互模式 | 所用模型、编辑格式、仓库规则与用户给定的运行/验证流程 |
| SWE-agent / mini-SWE-agent | 研究/批处理框架与轻量 CLI scaffold | SWE-agent 处于 maintenance-only；项目文档称 mini-SWE-agent 为后续方向 | 固定的项目版本、配置、模型、环境和复现实验条件 |
| OpenHands | Agent Canvas/Server、内建 Agent 或 ACP 外部 Agent、不同 runtime/backend | 可作为平台与执行入口，也可连接外部 Agent | 选定的是平台、内建 Agent 还是外部 ACP Agent，以及实际 backend/runtime |

这是一张入口地图，不是强弱排名。相同品牌可覆盖多个不同的比较单位；产品默认值和可用能力也会随版本、账户和组织配置变化。后文分别说明公开资料支持的事实与仍需在具体环境中核实的部分。

## 13.5 Codex：同一产品线中的本地与云端执行路径

Codex 不应再被概括为“轻量本地 CLI”。截至本章访问的官方资料，产品入口包括本地 CLI、IDE、desktop，以及在 OpenAI 管理计算机上执行任务的 Codex Cloud；不同入口的可见能力、workspace 与控制边界不能互相代替（[Codex CLI](https://developers.openai.com/codex/cli)、[ChatGPT Work and Codex](https://help.openai.com/en/articles/20001275-chatgpt-work-and-codex)、[Using Codex Cloud](https://help.openai.com/en/articles/20001545-using-codex-cloud)）。

先用一个差异说明为什么 surface 很重要。开发者在本地 CLI 中启动任务时，Codex 的本地工作流面向当前本地仓库；在 Codex Cloud 中，任务会在 OpenAI 管理的计算机和准备好的环境里运行，工作文件位于云端任务 workspace。即使两边都叫 Codex，文件系统、网络、预装依赖和结果交接也可能不同。不能从“Codex 支持云端”推断本地终端命令运行在云上，也不能从本地 workspace-write 推断 Cloud 环境具有相同限制。

OpenAI 的本地 sandbox 文档把 `read-only`、`workspace-write` 和 `danger-full-access` 分为不同文件/命令权限模式，并另行说明 approval policy。审批提示影响是否需要用户作决定；sandbox 限定执行能力；自动审批审阅不改变 sandbox 本身的边界（[Sandboxing](https://learn.chatgpt.com/docs/sandboxing)）。具体任务仍要记录实际选用的模式、审批策略和网络设置，不能仅凭产品名判断。

因此比较 Codex 时，先问三件事：任务在 CLI/IDE/desktop 还是 Cloud 启动？代码实际在哪台机器和哪个 workspace？采用什么 sandbox 与 approval policy？公开文档能定义可选合同，却不能证明某个用户任务当时采用了哪种配置。

## 13.6 Claude Code：多入口产品与有范围的执行控制

Claude Code 官方文档覆盖 CLI、IDE、desktop、web 和自动化等入口；permission、checkpoint、MCP、hooks、skills 与 sandbox 又各有自己的作用范围（[Platforms](https://code.claude.com/docs/en/platforms)、[Permissions](https://code.claude.com/docs/en/permissions)、[Checkpointing](https://code.claude.com/docs/en/checkpointing)）。这使它适合用来观察产品工作流如何穿过不同入口，但公开页面不足以证明各表面共享一套相同执行环境或内部调度逻辑。

比较安全边界时要更具体。Claude Code 文档描述的 shell sandbox 保护的是其覆盖范围内的 shell 命令及相关子进程；不能因此推出所有文件工具、MCP server、hook、插件或其他 surface 都在同一 OS 隔离边界内（[Sandboxing](https://code.claude.com/docs/en/sandboxing)）。权限规则、用户审批和系统级隔离回答不同问题：规则决定某项动作是否可继续，审批让用户授权某一决定，sandbox 限制动作能触及的资源。要分析某个部署，必须继续检查未被 shell sandbox 覆盖的通道。

`CLAUDE.md`、session 和记忆类能力可以为任务提供指令或历史信息，MCP、hooks 和 skills 则扩展连接或工作流。它们分别影响输入和动作路径；文档名称相似不代表它们是同一种状态存储、授权机制或执行隔离。内部 planner、system prompt 与跨 surface 状态迁移实现仍应标作未知。

## 13.7 OpenCode：固定版本下可读的 runtime 案例

第十二章已按 OpenCode v2.3.2 与固定源码 revision 分析配置、权限、Agent、工具、Session、Snapshot、Server 和 SDK。本章只抽取横向比较需要的结论：公开配置和源码让部分路径可以逐项核对，但开源不等于某个部署的全部行为已经被审计。

尤其要保留第十二章的安全结论：OpenCode 自身不提供 Agent sandbox。permission 规则能够控制工具动作是否被允许，却不等同于操作系统对文件、进程和网络的隔离；Server、SDK 和扩展工具也要按实际部署分别检查。版本、源码路径和证据边界见[第十二章](12-opencode架构分析.md)及[OpenCode V2 来源记录](../../research/opencode-v2-runtime-public-contracts-2026-11-18.md)。

## 13.8 Cursor：本地 IDE Agent 与 Cloud Agents 并存

Cursor 的本地 Agent 文档围绕代码库探索、编辑、终端、rules、diff review 与 checkpoints 描述 IDE 工作流；Cloud Agents 文档另行说明远端任务、环境准备、分支交接和可配置 runtime（[Agent overview](https://cursor.com/docs/agent/overview)、[Agent Review](https://cursor.com/docs/agent/agent-review)、[Cloud Agents](https://cursor.com/docs/cloud-agent)、[Choose where Cloud Agents run](https://cursor.com/docs/cloud-agent/self-hosted/choose-runtime)）。所以把 Cursor 归纳为“只能在 IDE 使用”的工具已经不完整。

本地 IDE 入口把用户当前编辑工作流与 Agent 交互放在一起；Cloud Agent 则可以在隔离 VM 或选定的 self-hosted 环境处理任务，并使用配置好的仓库、依赖和网络条件。云端网络、secret、私有网络连通性及 runtime 选项需要按管理员配置核实，不能把 Cloud Agent 的隔离 VM 描述推广到本地 IDE Agent。

比较 Cursor 时先问任务在本地 Agent 还是 Cloud Agent 中运行，再检查上下文如何给入、改动如何 review、工作区如何交接，以及云端环境的网络和凭据策略。编辑器集成是一项产品优势，但不自动说明异步任务、企业控制或评测复现的全部条件。

## 13.9 Aider：Git Repo 中的终端结对编程

Aider 官方文档把它定位为终端里的 AI pair programming 工具，工作对象是本地 Git 仓库（[文档入口](https://aider.chat/docs/index.html)）。

它有几个非常值得学习的设计点。

第一个是 repo map。

Aider 的 repo map 文档说明，它会构建仓库结构摘要，并按 token budget 选择内容提供给模型（[Repository map](https://aider.chat/docs/repomap.html)）。这解决了一个核心问题：大仓库不能全部塞进上下文，但 Agent 又需要知道整体结构。

repo map 的设计目标是：

1. 让模型看到全仓库的骨架。
2. 帮助模型判断应该打开哪些文件。
3. 降低无关上下文浪费。
4. 让编辑更符合现有抽象。

第二个是 chat modes。

Aider 的官方模式文档列出 `code`、`ask` 和 `architect` 等交互模式（[Chat modes](https://aider.chat/docs/usage/modes.html)、[In-chat commands](https://aider.chat/docs/usage/commands)）：

1. `code`：直接修改代码。
2. `ask`：讨论代码但不修改。
3. `architect`：先由 architect model 提方案，再由 editor model 转成具体编辑。
4. `help`：回答 aider 使用问题。

`architect` 模式中，architect model 负责提出方案，editor model 再转成文件编辑。这和前面讲的 Plan/Build 有相似之处，但不应假设两者实现相同。

第三个是 edit formats。Aider 的 `whole` 格式要求模型返回完整文件；`diff` 以搜索/替换块表示局部变更；`udiff` 则采用简化的 unified diff 形式。`architect` 流程还可让一个模型角色先提出方案，再由 editor role 按选定格式生成编辑（[Edit formats](https://aider.chat/docs/more/edit-formats.html)）。例如，本章的认证修复若使用搜索/替换块，旧文件片段必须能在当前版本中定位；即使格式正确，改动仍可能遗漏边界条件或破坏其他行为。格式定义了怎样表达 patch，不替独立测试和 diff review 验证其语义。

Aider 展示了一种围绕本地仓库、上下文地图和明确编辑协议组织的终端工作流：

```text
local git repo
-> repo map
-> ask/code/architect modes
-> edit format
-> git diff / lint / test
```

它的公开文档突出本地 Git 仓库中的结对编程流程；这说明其工具定位，不代表它在所有本地任务上都优于其他 Agent。

## 13.10 SWE-agent 与 mini-SWE-agent：研究 scaffold 的生命周期

SWE-agent 的历史论文和接口设计仍适合说明 Agent-Computer Interface（ACI）是一个可研究的系统因素：Agent 看到的命令和结果如何组织，会影响它与代码环境交互的路径。但项目维护状态已经变化。SWE-agent 当前 FAQ 说明该项目已被 mini-SWE-agent 取代，并处于 maintenance-only；mini-SWE-agent 文档将其定位为更轻量的 CLI 与批处理/研究 scaffold（[SWE-agent FAQ](https://swe-agent.com/latest/faq/)、[mini-SWE-agent overview](https://mini-swe-agent.com/latest/)、[mini-SWE-agent usage](https://mini-swe-agent.com/latest/usage/)）。

因此正文中的“SWE-agent”主要指研究历史和已固定版本论文，不把它描述成仍在积极演进的当前产品。若读者要复现实验，应进一步固定论文版本、仓库 commit、model、环境、工具和 benchmark 配置；维护者文档中的定位或自报结果不是独立性能比较。

ACI 论文支持的结论范围也很具体：在论文报告的模型、任务与设置中，接口设计与任务行为有关。它提出了值得控制或研究的变量，不能单独证明某一种接口在所有模型、任务和产品上都更好（[SWE-agent 论文 v3](https://arxiv.org/abs/2405.15793v3)）。

## 13.11 OpenHands：平台、内建 Agent 与外部 Agent 要分开

OpenHands 的 Agent Canvas 文档描述的是通过 Agent Server backend 路由交互的界面。使用者可以选择 OpenHands 内建 Agent，也可以连接 ACP subprocess Agent；文档列举 Codex、Claude Code、Gemini CLI、Pi 和 OpenCode 等外部 Agent。选择外部 Agent 时，Agent 自己管理其模型、工具和执行路径；Canvas 与 Server 作为平台入口，并不会把这些 Agent 自动变成同一个 runtime（[Agent Canvas setup](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/agent-canvas/first-time-setup.mdx)、[ACP agents](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/agent-canvas/acp-agents.mdx)）。

部署时还要继续区分 Agent Server 与 runtime/backend。OpenHands 架构文档说明运行位置可以有不同方式，例如 local 或 Docker；各模式的文件系统可见性与风险并不相同（[Architecture](https://github.com/OpenHands/OpenHands/blob/main/docs/architecture.md)、[Runtime architecture](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/architecture/runtime.mdx)）。这些页面来自仓库 `main` 分支，结论绑定 2026-11-18 访问时的文档内容。

所以选型或比较时要写全具体对象：OpenHands Agent Canvas + 内建 OpenHands Agent + 某个 runtime，还是 Canvas + ACP 外部 Agent + 由外部 Agent 管理的执行环境？只写“用了 OpenHands”不足以说明模型、工具、权限或副作用发生在哪里。

## 13.12 产品形态对比：CLI、IDE、Web、Cloud、SDK

产品入口会影响用户怎样提供上下文、审阅改动和接收结果，但入口本身不能说明完整执行边界。先把可观察问题拆开：

| 入口 | 读者应检查什么 | 不能单凭入口推断什么 |
|---|---|---|
| CLI | 当前目录/仓库、可用 shell、命令确认和输出回传方式 | 命令实际有多少文件/网络权限；是否存在独立 sandbox |
| IDE | selection、diagnostics、索引、diff 应用与 review 路径 | Agent 是否只使用当前编辑器上下文；后台动作能访问什么 |
| Web/Desktop | 会话入口、任务委派、产物与本地文件如何交接 | 代码是在本地还是远端；不同入口是否共享完整状态 |
| Cloud/远端 workspace | VM/container/runtime、依赖准备、网络/secret、分支或 patch 交付 | 所有云端任务是否使用同一种隔离/网络配置 |
| SDK/API | 调用者提供什么上下文、怎样启动任务、事件/错误如何回传 | SDK 默认就有产品 UI、可恢复状态或部署级治理 |

Codex 和 Cursor 的官方文档都分别描述本地与云端路径；Claude Code 文档也按多个产品 surface 列出能力。因此在比较时应直接记录实际运行的模式，而不是以品牌替代环境说明（[Codex Cloud](https://help.openai.com/en/articles/20001545-using-codex-cloud)、[Cursor Cloud Agents](https://cursor.com/docs/cloud-agent)、[Claude Code platforms](https://code.claude.com/docs/en/platforms)）。

### 用一项修复任务对齐观察点

下面是一项**教学构造**，不是某个产品的运行记录：仓库里 `/api/profile` 对缺失 bearer token 返回了 `200`，目标是修复为 `401` 并加回归测试；改动限于认证处理和相关测试，不创建远端分支、不 push。

为说明如何沿着任务追踪比较，假设初始请求记录是 `Authorization: absent → HTTP 200`，而任务要求它返回 `401`。下表里的动作和回执也都是教学构造，不代表 Codex、Cursor 或任何其他 Agent 的实际输出：

| 步骤 | 教学构造中的状态变化 | 这一步支持什么结论 |
|---|---|---|
| 1. 定位 | 读取认证 handler 和对应测试；找到当前缺失 token 路径 | 能说明任务上下文是否包含相关实现与验收位置；不能说明 Agent 看过整个仓库 |
| 2. 提议 | 提议把无凭据请求拒绝，并新增缺 token 回归测试 | 这是模型/Agent 提议；此时还不能说文件已改变 |
| 3. 派发 | 宿主按授权写入限定文件；push 请求不在允许动作中 | 策略与实际派发分开；被拒绝的 push 不能被当成“没有尝试”或“已有隔离” |
| 4. 执行 | 合成测试回执为新增测试通过、有效凭据测试仍通过 | 只说明指定测试在该假设环境中的回执，不证明全部行为正确 |
| 5. 检查 | 合成 diff 只含认证 handler 与测试；没有远端提交 | diff 支持范围检查；Git/远程副作用仍需结合执行记录核对 |
| 6. 验收 | 独立检查缺 token 得到 401、有效 token 路径未回归、改动范围符合要求 | 这些条件支持本任务的结果；不能外推为产品整体能力排名 |

逐步记录这条路径，比只记“任务成功”更有用：它能区分模型建议、宿主决定、真实副作用、测试回执和任务结论。若实际记录缺少某一环，应把它标为 unknown，而不是用最终回答补齐。

把同一任务交给不同产品前，可以按阶段问同一组问题：

| 阶段 | 需要观察的事实 | 能回答的问题 |
|---|---|---|
| 任务入口 | 本地 CLI、IDE Agent、Cloud task，或由平台承载的 Agent | 用户请求落在哪个 surface？运行代码的主机是谁管理？ |
| 上下文 | issue 文本、仓库指令、认证 handler、相关测试、用户选区或工具输出 | Agent 看到了哪些输入？有哪些遗漏、陈旧或不可信内容？ |
| 动作 | 读取文件、编辑补丁、运行指定测试、调用外部服务 | 哪些动作由 Agent 提议，哪些由宿主实际派发？ |
| 控制边界 | 文件范围、命令审批、sandbox、网络与 secret 设置 | “需要确认”是否和“被隔离”分开？push 是否被单独限制？ |
| 结果状态 | diff、测试输出、任务分支、会话事件、错误记录 | 实际改了什么？执行失败和任务失败是否被区分？ |
| 验收与交接 | 复查缺 token 得到 401、有效 token 行为仍通过、diff 无越界、无 push | 哪些检查证明目标达成？哪些仍需人工审查？ |

例如，Cloud 任务的测试通过只证明那个准备好的 workspace 中对应测试给出了通过回执；它不会自动证明本地工作区也同步了改动、没有遗漏其他接口，或服务端部署已安全。反过来，本地 IDE 的 diff review 也不能证明远端 CI 环境已重现同一结果。比较时把产物、执行回执与独立验收逐一对应起来。

这条任务贯穿后续的上下文、编辑、权限、评估和选型部分。比较表里的事实应来自产品文档或实际可复查的运行记录；本例只提供观察路径，不替任何产品填造结果。

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

公开文档中的例子显示了不同的上下文入口：Aider 的 repo map 按 token 预算挑选仓库结构信息（[Repository map](https://aider.chat/docs/repomap.html)）；Cursor 的 Agent 文档涉及代码库和 IDE 工作流（[Agent overview](https://cursor.com/docs/agent/overview)）；Claude Code 与 OpenCode 各有项目指令、配置及 session 合同（[Claude Code memory](https://code.claude.com/docs/en/memory)、[OpenCode instructions](12-opencode架构分析.md)）；SWE-bench 类型任务以 issue 与仓库作为输入；OpenHands Canvas 则可承载内建 Agent 或 ACP Agent。每项事实只说明文档公开的入口，不证明运行中模型完整接收了所有相关信息。

一个成熟 harness 不能只把所有东西塞给模型，而要做选择：

1. 当前任务需要哪些文件。
2. 哪些规则是真正高优先级。
3. 哪些工具结果需要保留。
4. 哪些日志只需要摘要。
5. 哪些上下文可能包含 prompt injection。
6. 哪些敏感信息不能进入模型。

回到缺 token 修复任务，最先需要的通常是 issue 中的预期行为、认证 handler、相关测试和项目指令；只有测试失败后，新的测试回执才成为下一轮的重要输入。整份安装日志或无关业务模块增加了上下文，却未必帮助决定这次改动。`AGENTS.md` 或 `CLAUDE.md` 一类文件可以提供项目指导，但它们不是授权凭据，也不自动创建文件/网络隔离边界。上下文工程做不好，模型再强也可能基于错误或过期信息采取动作。

## 13.14 编辑方式对比

下面按读者可能观察到的编辑接口分类，不声称盘点了本章所有产品的内部实现。Coding Agent 修改代码的方式大致有几类。

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

这类方法可以保留语法结构关系，但实现成本与语言覆盖是额外约束。对具体产品是否用 AST、patch 或其他内部编辑算法，若官方材料没有说明，就应写作未知；外部能稳定比较的通常是可见变更、冲突处理、diff review 和验收回执。

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

比较权限和执行边界时，至少分开四类机制：

1. **策略决定**：当前主体能否请求某项 tool/action。
2. **用户审批**：是否需要人对这次具体操作作决定。
3. **执行隔离**：进程实际能读写哪些文件、访问哪些网络、继承哪些凭据。
4. **业务授权**：下游服务是否允许当前身份对目标资源执行操作。

产品文档分别展示了不同组合，而不是一个通用的“安全等级”：OpenCode 的 permission 与其自身不提供 Agent sandbox 的边界见第十二章；Codex 文档将 sandbox mode 和 approval policy 分开；Claude Code 的 shell sandbox 只覆盖文档说明的命令路径；Cursor Cloud Agents 的网络与 runtime 也受环境配置影响（[Codex sandboxing](https://learn.chatgpt.com/docs/sandboxing)、[Claude Code sandboxing](https://code.claude.com/docs/en/sandboxing)、[Cursor runtime options](https://cursor.com/docs/cloud-agent/self-hosted/choose-runtime)）。SWE-agent/mini-SWE-agent 是研究 scaffold，不应因“benchmark runner”一词就推断每个环境都能隔离任意副作用。

面试时可以用缺 token 修复任务举例：读取认证文件和写入 patch 是文件权限问题；运行测试是命令派发问题；访问远端 API 是网络权限问题；push 则是新的外部副作用。Prompt 可以表达任务约束，但要限制副作用还需在实际工具派发与执行环境中落实，并留下可检查的回执。

## 13.16 评估方式对比

coding agent 的评估也分多层。

人工体验能发现交互摩擦和审查成本，但单次 demo 很难复现，也容易受任务选择影响。

更系统的评估包括：

1. 单元测试是否通过。
2. lint、typecheck、format 是否通过。
3. 真实 issue 是否解决。
4. patch 是否最小且可维护。
5. 是否引入安全问题。
6. 是否破坏已有行为。
7. 是否能在 benchmark 上复现。
8. 成本、延迟和失败率。

以 SWE-bench Verified 为例，任务从 GitHub issue 和指定仓库出发，应用 patch 后运行 `FAIL_TO_PASS` 与 `PASS_TO_PASS` 测试。OpenAI 的说明同时记录了需求描述不充分、测试拒绝合理解和环境设置失败等问题；官方描述的是这个评测设计及其局限，不是对所有软件工程质量的测量（[Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)）。

测试通过回答的是“指定测试集合在指定环境下是否给出通过回执”。它不能独自回答 patch 是否最小、没有安全回归、符合团队风格或适合维护。要比较产品表现，还要披露模型、Agent/harness、环境、提示、工具权限、预算、重试、verifier、seed 与产物评审口径。否则差异不能归因到某一个层面；已有 benchmark 方法论论文也指出，端到端结果会混合多个系统因素（[Coding Benchmarks Are Misaligned with Agentic Software Engineering](https://arxiv.org/abs/2606.17799)）。

团队可按任务和风险组合不同层次的证据，不要求每个项目采用同一套指标：

1. 公共 benchmark。
2. 私有 issue benchmark。
3. 回归测试集。
4. 人工 review。
5. 安全和权限测试。
6. 成本和延迟监控。

## 13.17 典型场景如何选型

选型不是把每个产品塞进一个固定场景类别，而是先写出这次任务的硬约束，再筛选能满足这些约束的 surface 和配置：

| 任务条件 | 先检查 | 可从哪些公开案例开始核对 |
|---|---|---|
| 代码必须留在本机、开发者逐步 review | 本地执行位置、workspace 边界、diff 与撤销方式、测试命令 | Codex CLI、Claude Code CLI、OpenCode、Aider、Cursor 本地 Agent |
| 编辑器选区和诊断是主要任务上下文 | IDE integration、索引/selection、diff review 和规则来源 | Cursor Agent、Claude Code IDE、Codex IDE、OpenCode IDE |
| 任务需远程异步执行并交回 patch/branch | 云端机器归属、workspace 生命周期、网络/secret、分支和恢复 | Codex Cloud、Cursor Cloud Agents、Claude Code web/automation、OpenHands Canvas/runtime |
| 需要固定配置重复跑研究任务 | 项目版本、模型、Agent scaffold、环境初始化、trajectory 与 verifier | mini-SWE-agent、OpenHands SDK；历史 SWE-agent 研究需固定版本 |
| 需要嵌入既有平台或切换 Agent | API/SDK/ACP 合同、Agent 与执行环境的责任归属、事件和权限映射 | OpenHands Agent Canvas/ACP、OpenCode Server/SDK 及各产品已公开 SDK |

表中只是调研入口，不代表这些对象都满足企业要求，也不构成推荐排名。对任意候选，都要用自己的仓库、权限策略、网络约束和验收标准完成小范围验证；特别注意“产品可提供某能力”不等于“当前租户已启用该能力”。

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
9. 把 IDE Assistant、CLI Agent、research scaffold、平台控制面和执行环境混为一个对象。
10. 因为入口名称相同，就假设本地与云端有相同 workspace、网络和审批策略。
11. 因为公开文档列出能力，就认为某个部署当前已启用该能力。
12. 把社区传闻或产品自述当作独立比较证据。

面试中如果被问“你怎么看 Claude Code、Codex、Cursor、Aider 这些工具”，先说明当前谈的是哪个版本和入口，再沿一个任务比较上下文、工具、权限、执行位置、状态、验收和交付。指出证据来源与未知项，比给出没有条件的强弱判断更可复核。

## 13.19 怎样做一次可复核的横向比较

前面已经得到一个认证修复任务和一组观察维度。现在要避免把这些维度重新压成一个看似客观的总分。产品选择和 harness 因果研究回答的是不同问题，比较前先确定自己在做哪一种。

### 选产品：把差异作为使用条件记录

如果问题是“团队在这项工作里应该使用哪个产品入口”，模型、产品 surface、网络策略、上下文注入方式、权限提示和云端执行都可能是产品的一部分。不要为了让表格整齐而假设这些条件相同。先写清任务范围、硬性约束和验收方式，然后对候选逐项记录实际合同：

| 记录项 | 认证修复任务中的具体问题 | 证据例子 |
|---|---|---|
| 比较单位 | 比较的是 Codex CLI、Codex Cloud、Cursor 本地 Agent、Cursor Cloud Agent，还是 OpenHands Canvas 中的某个 Agent？ | 产品 surface、Agent 名称、固定版本或访问日期 |
| 任务输入 | issue、测试、项目指令、IDE 选区分别如何提供？ | 任务文本、实际可见文件与会话记录 |
| 模型与配置 | 运行时使用什么模型/provider、配置和预算？ | 产品设置、运行日志；未显示的字段标 unknown |
| 工作区 | 哪个仓库副本被修改，结果如何交回？ | 本地路径说明、云端 workspace、branch/patch |
| 动作与边界 | 读取、写入、测试、网络访问和 push 分别受到什么控制？ | permission/approval 配置、sandbox/runtime 文档、执行回执 |
| 状态与恢复 | 中断或失败后保留哪些 session、patch、事件或 checkpoint？ | 产品记录和可复查产物；没有记录就注明 unknown |
| 验收 | 哪个测试或人工 review 判断 401 修复和改动范围？ | verifier 版本、测试结果、diff 与 reviewer 结论 |
| 限制 | 哪些条件无法确认，或不满足团队要求？ | 未公开项、文档适用范围、待验证项 |

这张表是记录卡，不计算产品总分。对“支持隔离执行”这样的表述，要追到具体 surface、运行时和策略；对“测试通过”，要写明测试和仓库状态。文档展示可选能力时，具体用户配置依然可能不同。

### 研究 harness 影响：只改变要研究的因素

如果问题变成“改用某种工具接口是否让同一个模型在该任务上更容易修复”，就进入受控比较。先明确要研究的 harness 因素；若研究的是上下文构建，就把上下文策略作为 treatment，而不是同时要求它保持相同。其余关键条件应尽可能固定，例如任务集、模型 revision、初始用户请求、预算、执行环境和 verifier。保存实际交互轨迹，确认配置的工具真的被调用；若轨迹偏离配置，结果不能简单归因到标签。scaffolding 研究报告过这种偏离，但它的发现受单一任务与具体实现限制（[The Scaffolding Matters More Than the Interface](https://arxiv.org/abs/2608.08654)）。

即使控制了这些因素，结论也只适用于所测任务、模型和环境。要推广到产品选择，还要另外考虑真实用户需要的 IDE、云端任务、审批、集成和交接工作流。benchmark 分数能说明指定 verifier 和实验条件下的结果，不能自动变成“产品整体更好”。

### 把主张、证据和未知并排写

最终报告最好按下面顺序表达：

1. **主张：** 我们想判断哪件具体事？例如“对本地私有仓库的认证 bug 修复，开发者能否 review 并接受 patch？”
2. **条件：** 哪个产品 surface、模型/provider、仓库、配置、权限、网络、预算和验证器参与？
3. **证据：** 哪些来自官方合同，哪些来自固定源码，哪些来自实际运行记录，哪些只是人工判断？
4. **观察：** 任务产物、测试回执、失败和副作用分别是什么？将环境错误、Agent 行为、验收器错误分开。
5. **边界：** 什么没有测到？是否只测一个仓库或一个任务？这条证据不能支持什么更大的判断？

不要把“没有观察到某个风险”写成“该风险不存在”，也不要用任意覆盖率、加权平均或固定 gate 代替条件说明。读者应能从报告还原比较对象与证据，而不是只得到一个名次。

## 13.20 面试题

### 题 1：为什么不能只用模型名称比较 coding agent？

因为完成任务的不只是模型。入口决定上下文怎样进入，harness 提供哪些工具和派发规则，执行环境决定文件、命令和网络边界，verifier 决定哪些结果算通过。若比较完整产品，应把这些条件作为产品的一部分报告；若研究某个组件的因果作用，则要固定其他关键条件。端到端 benchmark 结果不能自动归因到单一组件。

### 题 2：Codex CLI 与 Codex Cloud 有什么比较上的区别？

官方资料把 Codex CLI 描述为本地仓库工作流，把 Codex Cloud 描述为在 OpenAI 管理计算机和任务 workspace 中执行的远端路径。比较时要确认任务实际使用哪个 surface、代码在哪里运行、workspace 如何准备和结果如何交回。还应把 sandbox mode 与 approval policy 分开记录；不能因品牌相同就假设执行边界相同。

### 题 3：Claude Code 的 shell sandbox 是否等于所有工具都被沙箱隔离？

不能这样推断。官方文档说明 shell sandbox 对其覆盖的 shell 命令和相关子进程生效；其他工具、MCP、hook、插件或产品入口要按各自边界核对。permission、用户审批和 OS 隔离也分别解决不同问题，不能互相替代。

### 题 4：怎样比较 OpenCode、Cursor 和 OpenHands？

先把对象写具体。OpenCode 要固定 V2 版本并区分公开 permission 与操作系统隔离；Cursor 要区分本地 IDE Agent 和 Cloud Agent；OpenHands 要说明使用 Canvas/Server、内建 OpenHands Agent 还是 ACP 外部 Agent，以及所选 runtime/backend。之后再对齐任务、模型、仓库、权限、网络、验证器和交付形式。只说品牌不足以说明运行条件。

### 题 5：Aider repo map 与 IDE 上下文有什么差异？

Aider 文档说明 repo map 会根据 token 预算选择仓库结构信息，帮助模型看到相关文件和符号骨架；IDE Agent 的上下文还可能包含当前编辑位置、selection、diagnostics 等编辑器状态。两者说明上下文入口不同，不足以单独证明谁能修得更好。要比较结果，还要固定任务与模型并观察实际被提供的上下文。

### 题 6：SWE-agent 与 mini-SWE-agent 在本章中怎样定位？

SWE-agent 的论文适合作为 ACI 和研究 scaffold 的历史案例；项目 FAQ 现在说明它处于 maintenance-only，并指向 mini-SWE-agent。后者被维护者定位为轻量 CLI 与批处理/研究 scaffold。复现实验应固定具体版本和配置；维护者定位不能替代独立性能对照。

## 13.21 小练习

1. 对本章的合成认证修复任务，写出可检查的成功条件和不允许发生的副作用。说明每条条件由什么产物或观察支持。
2. 为 Codex CLI 与 Codex Cloud 各填一张比较记录卡。哪些合同可从官方文档确认？哪些必须从具体运行配置确认？
3. 为 Cursor 本地 Agent、Cursor Cloud Agent 和 OpenHands Canvas 画出“surface → Agent/harness → execution environment → artifact”的关系图。对无法从公开资料确定的边界标记 unknown。
4. 设计一个只研究工具接口影响的对照实验。列出要固定的模型、任务、预算、环境和 verifier，并说明怎样检查实际工具调用确实遵循配置。
5. 阅读 SWE-bench Verified 说明，解释 FAIL_TO_PASS、PASS_TO_PASS 以及环境设置各能证明什么；列举测试通过仍无法证明的两个工程结论。
6. 任选一个团队任务，写一段不超过一页的选型理由：先列硬约束，再列比较对象、证据、未知和验证计划。不要给产品排名或合成分数。

## 13.22 本章总结

比较 Coding Agent 时，先明确比较的是产品 surface、Agent/harness、执行环境还是评估配置。Codex CLI 与 Codex Cloud、Cursor 本地 Agent 与 Cloud Agent、OpenHands 平台与其内建或 ACP 外部 Agent，都是不同的运行组合。产品名本身不能告诉我们文件、网络、权限和状态边界。

沿一条具体任务追踪输入、工具请求、宿主决定、执行副作用、diff、测试回执和验收结论，再比较上下文、编辑、恢复和治理。公开文档能说明公开合同，固定版本源码能支持具体实现路径，实际运行记录能说明指定配置下发生的事；每类证据都有边界。

选产品时披露产品差异；做 harness 因果研究时控制其余条件并检查真实轨迹。Benchmark、测试、人工 review 和产品体验各自回答不同问题。最终结论应给读者足够信息复核，而不是把复杂系统压成一个总分。
