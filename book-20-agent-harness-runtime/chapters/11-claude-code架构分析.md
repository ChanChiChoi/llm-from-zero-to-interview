# 第十一章：Claude Code 架构分析

## 11.1 本章定位

本章用于分析 Claude Code 这类终端型 coding agent 产品的架构思想。

这里的重点不是复述某个官方内部实现，而是用前十章建立的 harness 视角，理解一个成熟 coding agent 产品通常如何组织用户交互、上下文、工具、文件编辑、终端执行、权限控制、trace 和任务循环。

本章可以参考公开资料和产品行为进行架构推断，但要注意边界：没有官方确认的实现细节，不应写成确定源码结论。面试中也应使用“从系统设计角度看”“这类产品通常会”这样的表达，避免把推测说成事实。

学完本章，你应该能回答：

1. Claude Code 类产品为什么是 agent harness 的典型案例。
2. 终端型 coding agent 和 IDE 型 coding agent 有什么差异。
3. Claude Code 类系统通常包含哪些核心模块。
4. 用户输入如何变成模型上下文和工具动作。
5. 文件编辑、命令执行、权限确认和 trace 如何协同。
6. 面试中如何分析一个真实 coding agent 产品的架构。

## 11.2 为什么 Claude Code 是典型案例

Claude Code 代表的是一种终端型 coding agent 产品形态。

它的典型特点是：

1. 面向真实本地或远程代码库。
2. 通过命令行与用户交互。
3. 能读取文件、搜索代码、编辑文件。
4. 能运行命令、测试和构建。
5. 需要处理权限确认。
6. 需要管理长任务上下文。
7. 需要在多轮工具调用后总结结果。

这类系统的关键不是“模型会不会写代码”，而是能不能把模型安全接入真实开发环境。

从 harness 视角看，它必须解决：

1. 用户任务如何进入系统。
2. 模型每轮看到什么上下文。
3. 模型如何表达动作。
4. 动作如何映射到工具。
5. 工具结果如何返回模型。
6. 危险动作如何确认。
7. 文件修改如何展示 diff。
8. 失败如何恢复。

这正是前十章讲的 agent runtime 主线。

## 11.3 终端型 Coding Agent 的产品形态

终端型 coding agent 和 IDE 型产品不同。

终端型产品优势：

1. 接近开发者真实工作流。
2. 容易访问 shell、git、测试命令。
3. 易于在服务器、容器、远程环境中运行。
4. 对大型仓库和工程任务更自然。
5. 可以用纯文本展示计划、命令、diff 和结果。

终端型产品挑战：

1. UI 交互空间有限。
2. Diff 和多文件导航不如 IDE 直观。
3. 命令执行风险更高。
4. 用户确认流程要尽量清晰。
5. 长日志和上下文展示需要压缩。

IDE 型产品优势：

1. 能直接显示文件和 inline diff。
2. 用户可以局部选择代码。
3. 更容易和编辑器状态集成。
4. 可视化体验更好。

但 IDE 型产品也要处理相同的 harness 问题：上下文、工具、权限、trace、评估。

## 11.4 高层架构图

从系统设计角度，可以把 Claude Code 类系统拆成：

```text
CLI / UI Layer
-> Session Manager
-> Context Builder
-> Model Adapter
-> Action Parser
-> Tool Registry
-> Execution Engine
-> Permission System
-> File System / Terminal / Search Tools
-> Trace Logger
-> Final Response Renderer
```

模块职责：

1. CLI/UI Layer：接收用户输入，展示流式输出、计划、命令、diff 和确认提示。
2. Session Manager：维护当前会话、任务状态和历史摘要。
3. Context Builder：选择文件、工具结果、diff、规则和计划进入模型上下文。
4. Model Adapter：调用 Claude 或其他模型，处理 streaming 和 tool use。
5. Action Parser：解析模型输出中的动作或工具调用。
6. Tool Registry：管理 read、search、edit、bash 等工具。
7. Execution Engine：执行工具，处理超时、错误和结果标准化。
8. Permission System：判断动作是否允许、是否需要用户确认。
9. Trace Logger：记录每轮模型调用、工具调用、文件 diff、命令输出和权限决策。
10. Response Renderer：把最终结果整理给用户。

面试时，这个图比“它调用了大模型”更能体现系统理解。

## 11.5 用户输入如何进入系统

用户输入通常有几类：

1. 自然语言任务。
2. 斜杠命令或系统命令。
3. 文件路径或代码片段。
4. 对当前 agent 行动的确认或拒绝。
5. 中途追加要求。

系统需要先判断输入类型。

例如：

```text
修复 auth 测试失败
```

这是任务输入。

```text
/help
/clear
/resume
```

这更像控制命令，不应该直接交给模型当普通任务。

用户确认：

```text
Approve running npm test -- auth?
```

这属于权限系统的一部分。

输入路由的价值是防止所有文本都混进模型上下文。控制命令、权限确认、自然语言任务应走不同路径。

## 11.6 命令系统

终端型 coding agent 通常会有命令系统。

命令系统解决：

1. 查看帮助。
2. 清理上下文。
3. 切换模型或配置。
4. 管理会话。
5. 查看状态。
6. 控制权限。
7. 退出或中断任务。

这些命令不是模型工具，而是产品控制面。

需要区分：

1. 用户命令：控制 agent 产品行为。
2. 模型工具调用：agent 请求 runtime 执行动作。
3. Shell 命令：通过 Bash 工具在工作区执行。

混淆这三者会导致设计混乱。

面试表达：

```text
我会把终端型 agent 的命令系统分成控制命令和模型工具调用。像 /help、/clear、/resume 是产品控制面，不应该当作普通 prompt；而 read_file、run_command 是模型请求 runtime 执行的工具。Shell 命令又必须经过 Bash 工具和权限系统。
```

## 11.7 上下文构造

Claude Code 类产品的效果很大程度取决于上下文构造。

上下文来源：

1. 用户任务。
2. 系统规则。
3. 项目规则文件。
4. 已读文件。
5. 搜索结果。
6. 最近命令输出。
7. 当前 diff。
8. 会话历史摘要。
9. 当前计划。
10. 工具 schema。

终端型 agent 的上下文难点：

1. 代码库很大。
2. 终端输出很长。
3. 用户可能连续追加要求。
4. 文件不断变化。
5. 模型需要知道哪些动作已执行。

合理策略：

1. 用户目标和最新指令固定保留。
2. 当前任务计划固定保留。
3. 当前编辑文件和测试失败片段高优先级。
4. 长日志压缩成摘要。
5. 旧历史压缩成 task state。
6. 关键文件修改后重新读取或更新摘要。

这对应第七章的 context builder 设计。

## 11.8 工具调用架构

Claude Code 类系统必须有工具调用层。

常见工具：

1. 读文件。
2. 搜索代码。
3. 编辑文件。
4. 执行命令。
5. 查看 diff。
6. 询问用户。

工具调用架构要包含：

1. 工具 schema。
2. 工具描述。
3. 权限等级。
4. 执行入口。
5. 结果标准化。
6. 错误处理。
7. trace 记录。

模型不能直接访问文件系统或 shell。它应该输出结构化动作，由 runtime 校验后执行。

这点非常关键：成熟 agent 产品不是“模型自己操作电脑”，而是模型通过受控工具接口操作环境。

## 11.9 文件编辑机制

文件编辑是 coding agent 的核心能力。

一个稳健的编辑机制应支持：

1. 读取目标文件。
2. 基于上下文生成 patch。
3. 应用 patch。
4. 检查冲突。
5. 生成 diff。
6. 记录 trace。
7. 必要时回滚。

设计重点：

1. 避免 whole-file rewrite。
2. 优先小 patch。
3. 编辑前检查文件是否变化。
4. 不覆盖用户并发修改。
5. 修改后展示 diff。
6. 大范围改动需要确认。

终端型产品尤其需要清晰展示 diff，因为用户不在 IDE 的可视化编辑界面里。

## 11.10 终端执行机制

终端执行能力让 coding agent 能验证自己的修改。

典型用途：

1. 运行测试。
2. 运行 lint。
3. 构建项目。
4. 查看 git 状态。
5. 复现错误。

但终端执行必须受控：

1. 工作目录限制。
2. 超时限制。
3. 命令风险分级。
4. 用户确认。
5. 输出截断和摘要。
6. 环境变量脱敏。
7. trace 记录。

面试中要强调：能跑命令不是重点，能安全跑命令才是重点。

## 11.11 权限与确认

Claude Code 类系统面对真实代码库，必须有权限确认机制。

需要确认的操作通常包括：

1. 写文件。
2. 删除文件。
3. 执行可能有副作用的命令。
4. 安装依赖。
5. 访问网络。
6. Git 高风险操作。
7. 读取敏感文件。

确认提示应该说明：

1. 要执行什么。
2. 为什么执行。
3. 影响哪些文件或资源。
4. 风险等级。
5. 是否可回滚。

权限系统要挡住 prompt injection。即使模型被 README 或网页诱导读取 secret，runtime 也应该拒绝。

## 11.12 会话状态和恢复

Coding agent 任务可能很长。

会话状态应保存：

1. 用户目标。
2. 当前计划。
3. 已读文件。
4. 已修改文件。
5. 工具调用历史。
6. 权限确认历史。
7. 当前 diff。
8. 测试结果。
9. 上下文摘要。

恢复场景：

1. 用户中断后继续。
2. 模型调用失败后重试。
3. 命令超时后恢复。
4. 上下文过长后压缩。
5. 用户手动修改文件后同步。

Session manager 和 state store 决定了 agent 能否处理长任务。

## 11.13 日志和 Trace

Claude Code 类产品也需要 trace。

Trace 用于：

1. 用户审计。
2. Debug 失败。
3. 复现任务。
4. 安全审计。
5. 产品质量改进。

Trace 应记录：

1. 用户任务。
2. 模型调用。
3. 工具调用。
4. 工具结果。
5. 文件 diff。
6. 命令输出摘要。
7. 权限决策。
8. 错误和重试。

这里要注意隐私和安全。Trace 不能无脑保存 secret、完整环境变量、私有 token 或敏感文件内容。

## 11.14 任务循环

Claude Code 类 coding agent 的典型任务循环可以抽象为：

```text
receive user goal
-> build context
-> call model
-> parse action
-> check permission
-> execute tool
-> append observation
-> update state
-> repeat until done or blocked
-> summarize result
```

这个循环看起来简单，但工程难点在边界条件：

1. 模型请求不存在的工具。
2. 工具参数不合法。
3. 文件已被用户改动。
4. 命令超时。
5. 测试失败但原因不明确。
6. 上下文超过窗口。
7. 用户中途改变目标。
8. 权限被拒绝。

成熟 harness 的价值，就是把这些情况变成可控状态，而不是让 agent 崩掉或乱改。

## 11.15 一个示例任务链路

假设用户说：

```text
修复登录接口单测失败，并解释原因。
```

合理链路是：

1. Session manager 创建任务状态。
2. Context builder 加入用户目标、项目规则和工具 schema。
3. 模型先搜索测试文件和登录接口代码。
4. Search tool 返回相关路径和片段。
5. 模型读取测试文件和实现文件。
6. 模型推断失败原因，并请求运行相关测试。
7. Permission system 判断测试命令风险较低，可执行或请求确认。
8. Terminal tool 执行测试并返回失败日志摘要。
9. 模型生成最小 patch。
10. Edit tool 应用 patch 并生成 diff。
11. 模型再次请求运行测试。
12. 测试通过后，agent 总结修改内容、失败原因和验证结果。

这个例子说明：coding agent 的智能不只在模型推理，而在“读、想、改、测、解释”的闭环。

## 11.16 和前面章节的映射

Claude Code 类产品几乎覆盖了本书前十章的所有主题。

对应关系：

1. 第 1 章 harness 总览：Claude Code 是面向代码任务的 harness 产品形态。
2. 第 2 章 agent runtime：它需要 runtime loop、state manager 和 action executor。
3. 第 3 章 coding agent 工作流：它完整覆盖理解、检索、修改、验证、总结。
4. 第 4 章工具系统：它需要文件、搜索、编辑、终端等工具注册和调度。
5. 第 5 章文件系统与代码编辑：它必须安全修改真实仓库。
6. 第 6 章终端执行：它必须运行测试、构建和诊断命令。
7. 第 7 章上下文管理：它要在长任务中压缩历史、保留关键状态。
8. 第 8 章权限模型：它需要确认危险操作并防御注入。
9. 第 9 章 trace：它需要记录执行链路以便审计和复盘。
10. 第 10 章 evaluation harness：它可以被放进任务集里做自动回归评估。

## 11.17 真实工程坑

真实系统中，最容易踩坑的不是模型调用，而是 harness 边界。

常见坑：

1. 把终端输出全塞进上下文，导致重要信息被挤掉。
2. 编辑文件前不重新读取，覆盖用户并发修改。
3. 命令权限太宽，agent 能执行高风险操作。
4. 只记录最终回答，不记录中间工具调用，失败后无法复盘。
5. 工具错误返回不标准，模型无法判断下一步。
6. Diff 展示不清晰，用户不知道 agent 改了什么。
7. 会话恢复只恢复聊天记录，不恢复任务状态。
8. 对 prompt injection 只靠系统提示，不靠 runtime 权限。
9. 把产品控制命令和模型工具调用混在一起。
10. 没有 regression eval，新版本 prompt 改动后质量漂移。

## 11.18 面试题

### 题 1：如何设计 Claude Code 类 coding agent 的高层架构？

参考回答：

```text
我会把它拆成 CLI/UI 层、session manager、context builder、model adapter、action parser、tool registry、execution engine、permission system、trace logger 和 response renderer。用户输入先进入 session，context builder 组织任务、规则、文件片段和工具结果，模型输出结构化动作，runtime 校验权限后执行工具，再把 observation 送回模型，循环直到任务完成或阻塞。核心不是简单调用模型，而是安全、可观测、可恢复地把模型接入代码库、文件系统和终端。
```

### 题 2：为什么 coding agent 需要权限系统？

参考回答：

```text
因为 coding agent 会操作真实代码库和终端。写文件、删除文件、运行命令、访问网络、安装依赖、执行 git 操作都可能有副作用。权限系统要根据动作类型、路径、命令风险和用户策略决定是否允许或确认。只靠 prompt 约束不够，因为模型可能被仓库内容或外部文本注入诱导，真正的边界必须在 runtime 层实现。
```

### 题 3：Claude Code 类系统的上下文构造难点是什么？

参考回答：

```text
难点是代码库、日志和会话历史都可能远超上下文窗口，而且文件会被 agent 或用户持续修改。Context builder 需要保留用户目标、最新指令、当前计划、关键文件片段、测试失败摘要和当前 diff，同时压缩旧历史和长日志。不能把所有内容都塞进去，否则会浪费窗口并降低模型注意力。
```

### 题 4：终端型 coding agent 和 IDE 型 coding agent 有什么差异？

参考回答：

```text
终端型 agent 更接近真实工程环境，适合跑测试、构建、git、容器和远程服务器任务，但 UI 展示有限，权限和日志管理要求更高。IDE 型 agent 更适合局部代码编辑、inline diff 和可视化交互，但也需要同样的工具、上下文、权限和 trace 体系。两者产品形态不同，底层 harness 问题类似。
```

## 11.19 小练习

1. 画出一个 Claude Code 类系统从用户输入到工具执行再到最终回答的链路图。
2. 为 `run_command` 工具设计一个权限分级策略，至少区分只读命令、测试命令、安装命令、删除命令和 git 高风险命令。
3. 设计一个 trace schema，记录一次 coding agent 修改文件并运行测试的全过程。
4. 假设 agent 覆盖了用户刚刚手动修改的文件，分析 harness 应该如何预防和恢复。
5. 设计一个 evaluation task，用于评估 Claude Code 类 agent 是否能修复一个真实单测失败。

## 11.20 本章总结

本章用 harness 视角分析了 Claude Code 类终端型 coding agent。

核心结论：

1. Claude Code 类产品的本质是把模型安全接入真实代码库和终端的 agent harness。
2. 终端型 coding agent 的优势是贴近真实工程环境，挑战是权限、日志、diff 和上下文管理更复杂。
3. 高层架构通常包括 CLI/UI、session manager、context builder、model adapter、tool registry、execution engine、permission system 和 trace logger。
4. 用户命令、模型工具调用和 shell 命令必须分层处理。
5. 文件编辑要避免覆盖并发修改，终端执行要有权限、超时和输出摘要。
6. Trace 和 session state 是长任务恢复、失败复盘和 evaluation harness 的基础。
7. 面试中分析 Claude Code 类系统，应强调 harness、工具、权限、上下文和闭环验证，而不是只说“调用大模型写代码”。

下一章会分析 OpenCode 这类开源 coding agent，从开放配置、工具抽象、权限规则和多模型适配角度继续理解 agent runtime。
