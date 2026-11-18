# Coding Agent 产品比较：公开合同与评估边界

检索日期：2026-11-18  
证据状态：source-backed；本记录整理各项目公开文档和研究资料支持的事实，不记录章节取舍、正文修改或精修进度。

## 研究问题与范围

核对常见 Coding Agent 产品和研究框架在任务入口、模型/agent 身份、运行位置、workspace、上下文、动作控制、状态恢复和验收方面公开了什么；梳理比较产品时会混淆哪些系统层次，以及基准分数不能支持哪些结论。本轮仅阅读官方产品文档、项目维护者文档、固定版本论文与原始研究；未启动任何产品，也未运行比较任务。

## 官方产品与项目资料

| 系统/资料 | 日期或版本 | 实际阅读范围及来源支持 | 边界 |
| --- | --- | --- | --- |
| OpenAI Codex CLI、Codex Cloud 与 sandbox | 官方动态文档，2026-11-18 访问：[CLI](https://developers.openai.com/codex/cli)、[Using Codex Cloud](https://help.openai.com/en/articles/20001545-using-codex-cloud)、[Sandboxing](https://learn.chatgpt.com/docs/sandboxing)、[ChatGPT Work and Codex](https://help.openai.com/en/articles/20001275-chatgpt-work-and-codex) | CLI 文档将 Codex 描述为在本地仓库中检查、编辑和运行代码的 terminal agent；当前 ChatGPT/Codex 文档列有 CLI、IDE、desktop、web/mobile cloud 等入口。Codex Cloud 在 OpenAI 管理的计算机上运行；任务从预设环境启动并使用单独 workspace/working files，Cloud 是远端执行路径。Codex 本地权限文档分别描述 `read-only`、`workspace-write`、`danger-full-access` 沙箱模式及审批策略；自动审批审阅不改变沙箱边界。 | 这些页面描述当前公开产品合同，不证明某个用户配置实际采用的 profile、sandbox 或网络策略；不同 surface 不能因为同属 Codex 就视为同一执行环境。计划、账户和 rollout 可影响可见入口。 |
| Claude Code | Anthropic 官方动态文档，2026-11-18 访问：[Platforms](https://code.claude.com/docs/en/platforms)、[Permissions](https://code.claude.com/docs/en/permissions)、[Sandboxing](https://code.claude.com/docs/en/sandboxing)、[Checkpointing](https://code.claude.com/docs/en/checkpointing)；逐项来源和边界见[Claude Code 公开产品合同记录](claude-code-public-product-contracts-2026-11-18.md) | 官方资料公开 CLI、IDE、Desktop、Web、CI 等表面及其 local/cloud 执行区别；权限规则、shell sandbox、hooks、MCP、session 和文件 checkpoint 各有作用范围。当前文档限定本机 shell sandbox 覆盖的命令和子进程；不能据此推断所有 file tools、MCP、hooks、插件或平台表面都在相同边界。 | 动态产品合同绑定访问日；不支持推断未公开 planner、完整 system prompt 或产品所有 surface 的行为。 |
| OpenCode V2 | 官方 V2 文档及 release，2026-11-18 访问；固定源码 c0ad18b。详见[OpenCode V2 Runtime 合同与架构边界](opencode-v2-runtime-public-contracts-2026-11-18.md) | V2 公开配置、工具、Agent、权限、Session、Snapshot、Server、Client 和 SDK 合同。固定版本 threat model 表示 OpenCode 本身不提供 Agent sandbox；权限提示与宿主隔离应分开核查。 | 源码事实只适用于 c0ad18b 所审阅的路径；访问日文档不等于用户安装配置或实际部署状态。 |
| Cursor Agent 与 Cloud Agents | Cursor 官方动态文档，2026-11-18 访问：[Agent overview](https://cursor.com/docs/agent/overview)、[Agent Review](https://cursor.com/docs/agent/agent-review)、[Plan Mode](https://cursor.com/docs/agent/plan-mode)、[Cloud Agents](https://cursor.com/docs/cloud-agent)、[Choose where Cloud Agents run](https://cursor.com/docs/cloud-agent/self-hosted/choose-runtime) | Agent 文档公开代码库探索、编辑、终端执行、diff review、checkpoint 和规则等 IDE 能力。Cloud Agents 文档称任务在隔离 VM 中运行，连接代码托管、准备依赖和工作区，并可通过 separate branch/PR handoff；环境网络、secret、私有网络和 self-hosted machine 可配置。 | 文档只说明公开能力和可配置项；不说明每个租户当前启用的网络/凭据策略，也不支持把 Cloud Agent 边界推广到本地 Cursor Agent。 |
| Aider | 项目官方动态文档，2026-11-18 访问：[Documentation](https://aider.chat/docs/index.html)、[Repository map](https://aider.chat/docs/repomap.html)、[Chat modes](https://aider.chat/docs/usage/modes.html)、[In-chat commands](https://aider.chat/docs/usage/commands)、[Edit formats](https://aider.chat/docs/more/edit-formats.html) | 文档把 Aider 定位为在本地 Git repository 中进行的 AI pair programming；repo map 按 token budget 选择仓库结构信息；`ask`、`code`、`architect` 等模式划分交互任务，architect/editor 流程中一个模型角色提出方案，另一个角色转成具体文件编辑；`whole`、搜索/替换 `diff` 与简化的 `udiff` 是不同文件修改协议。 | 项目文档支持产品提供的接口行为和设计目的；它不证明 repo map 在任意任务中减少了误差，也不证明特定 edit format 在任意模型上更准确、成本更低或更安全。 |
| SWE-agent 与 mini-SWE-agent | 项目文档动态页面，2026-11-18 访问：[SWE-agent FAQ](https://swe-agent.com/latest/faq/)、[mini-SWE-agent overview](https://mini-swe-agent.com/latest/)、[mini-SWE-agent usage](https://mini-swe-agent.com/latest/usage/) | SWE-agent 官方 FAQ 现称 SWE-agent 已由 mini-SWE-agent 取代，SWE-agent 处于 maintenance-only。mini-SWE-agent 文档将其定位为轻量 CLI 和批处理/benchmark scaffold，并提供 agent、environment 和 model 组件的配置入口。维护者页面所报 benchmark 数字未作为本记录的通用性能结论。 | SWE-agent 论文/旧配置可用于研究 Agent-Computer Interface 和基准历史；不能在当前产品比较中把维护状态下的旧项目写成仍在积极迭代的同一工具。mini-SWE-agent 的维护者说明和自报数据不是独立对照结果。 |
| OpenHands Agent Canvas、Agent Server 与 runtime | OpenHands 官方仓库/文档 `main` 分支，2026-11-18 访问：[Agent Canvas setup](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/agent-canvas/first-time-setup.mdx)、[ACP agents](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/agent-canvas/acp-agents.mdx)、[Architecture](https://github.com/OpenHands/OpenHands/blob/main/docs/architecture.md)、[Runtime architecture](https://github.com/OpenHands/OpenHands/blob/main/docs/openhands/usage/architecture/runtime.mdx) | Agent Canvas 文档说明界面通过 Agent Server backend 路由会话，用户可选内建 OpenHands agent 或 ACP subprocess；ACP 例子包括 Claude Code、Codex、Gemini CLI、Pi 和 OpenCode，外部 agent 负责自己的模型/工具/执行。运行位置由 backend/runtime 选项承担，官方架构文档区分 local、Docker 等模式，并提示 local host filesystem 风险。 | 仓库 main 与产品云端能力持续变化；Canvas、Agent Server、内建 OpenHands agent、ACP 外部 agent 不是同一比较单位。某种运行模式的边界不能外推到另一 backend。 |

## 研究与基准方法

- Gorinova 等，[Position: Coding Benchmarks Are Misaligned with Agentic Software Engineering](https://arxiv.org/abs/2606.17799)，arXiv v2，2026-07-18。作者主张端到端分数会合并模型、harness、上下文、环境与反馈等因素，并指出单一参考解和缺乏部件级信号的限制。它是一篇 position paper；支持提出比较问题，不是对所有 Coding Agent 结构的因果实验。
- Alier Forment 等，[The Scaffolding Matters More Than the Interface](https://arxiv.org/abs/2608.08654)，arXiv 预印本，2026-08-09。研究在一项固定软件任务上比较 7 种 scaffolding、5 种模型和 MCP/CLI 工具接口；摘要报告实际行为与指定接口不总一致，因此只按配置标签比较可能测到未知混合行为。结果受单一任务、具体实现和预印本评估限制；可支持在比较中检查实际交互轨迹，不能推出其他任务的普遍效果。
- Yang 等，[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793v3)，arXiv v3，2024-11-11。论文把 ACI 设计作为可研究变量，并在 SWE-bench 与 HumanEvalFix 上报告特定接口、模型与配置结果；支持“agent 与计算机的接口设计可能影响其行为和评估结果”，不证明某一种接口在所有系统、模型或任务中占优。
- OpenAI，[Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)，官方说明，2024-08-13 发布，2026-11-18 访问。SWE-bench 任务以 GitHub issue 与仓库为输入，采用 `FAIL_TO_PASS` 和 `PASS_TO_PASS` 测试评估应用后的 patch；该页面也记录了问题描述不充分、测试可能拒绝合理解、环境设置失败等评估风险。它说明功能测试和环境质量共同定义可观察结果，不代表测试通过覆盖全部工程约束。OpenAI 另说明其公开比较使用单 seed 与各 scaffold 最接近公开/默认参数，可能与排行榜设置不同。

## 跨来源可支持的推断及未验证边界

1. 公开系统的“比较对象”不总是同一层：产品套件可横跨 CLI/IDE/Cloud；agent runtime 可嵌入另一平台；研究 scaffold 可用来固定 task/environment 评估；桌面或 Web 界面也可能只是远程执行的控制面。应先声明被比较的是产品 surface、agent/harness、执行环境还是完整 user-facing workflow。
2. 若研究的是某个组件对结果的影响，应尽可能固定模型、任务、工具、预算、环境、网络和 verifier，并记录运行轨迹；若比较的是购买/使用产品，则这些差异本身就是产品条件，需报告而不能假装已控制。以上是依据公开合同和研究范围形成的编辑推断，不是任何单一来源规定的通用实验协议。
3. 即使任务和名称相同，也应检查实际模型 revision、surface/mode、workspace、网络/secret、工具权限、重试/预算、验证器和产物。最终自述、测试回执、patch 和人工 review 各自支持不同结论；必须把未观察项留为 unknown。
4. 本次没有运行 Codex、Claude Code、OpenCode、Cursor、Aider、SWE-agent、mini-SWE-agent 或 OpenHands；没有复现 benchmark、验证账户配置、产品支持区域、发布节奏、性能、成本或实际隔离强度。动态文档在后续使用前须按访问日复核；公开材料不支持产品总体排名或闭源内部推断。
