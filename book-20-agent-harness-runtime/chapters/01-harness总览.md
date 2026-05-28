# 第一章：Harness 总览

## 1.1 本章目标

这一章先回答一个核心问题：为什么 coding agent 不能只是“一个会写代码的模型 API”？

如果只把大模型接到聊天窗口里，它最多能生成文本、解释代码、给出修改建议。真正的 coding agent 要能读文件、改文件、跑命令、看报错、继续修复、申请权限、记录过程、支持回滚和评估任务是否完成。把这些能力组织起来的系统层，就是本书所说的 agent harness 或 coding agent runtime。

学完本章，你应该能回答：

1. Harness 是什么，不是什么。
2. Harness 和 model、agent、tool、skill、runtime、evaluation harness 有什么关系。
3. 为什么模型能力越强，harness 反而越重要。
4. 一个 coding agent harness 通常包含哪些模块。
5. Harness 如何把模型输出变成可执行、可控、可观测的动作。
6. 面试中如何从系统设计角度解释 harness。

## 1.2 核心定义

Harness 是把模型能力组织成可执行系统的运行框架。

更具体地说，在 coding agent 场景中，harness 负责把下面这些东西连接起来：

1. 用户任务。
2. 大模型调用。
3. 上下文构造。
4. 工具注册和工具调用。
5. 文件读取和代码编辑。
6. 终端命令执行。
7. 权限控制和用户确认。
8. 沙箱和安全策略。
9. 状态管理和任务进度。
10. trace、日志、回放和评估。

一句话定义：

```text
Agent harness 是包裹在模型外层的工程运行系统，它把模型的文本决策转成受控动作，并把环境反馈重新组织成下一轮模型上下文。
```

这里有两个关键词：受控动作和环境反馈。

模型本身只输出 token。Harness 要做的是：

1. 判断这些 token 是否表示一个工具调用、文件修改、命令执行或最终回答。
2. 校验动作是否合法、安全、可执行。
3. 执行动作并收集结果。
4. 把结果压缩、结构化后交还给模型。
5. 记录整个过程，方便用户审计、系统回放和后续评估。

## 1.3 为什么需要 Harness

单独一个 LLM API 只能生成文本。真实 coding agent 面对的是一个动态环境。

这个环境里有：

1. 一个具体工作目录。
2. 很多源代码文件。
3. 测试、构建、lint、格式化命令。
4. Git 状态和用户未提交改动。
5. 权限边界和危险操作。
6. 长上下文和有限 context window。
7. 命令失败、测试失败和工具异常。
8. 用户随时插入的新指令。
9. 多轮任务状态。

如果没有 harness，模型会遇到几个问题：

1. 它不知道当前真实文件内容，只能凭用户粘贴的片段猜。
2. 它无法安全地修改文件，只能输出建议。
3. 它无法运行测试验证自己的修改。
4. 它无法知道命令失败后的 stderr、退出码和环境状态。
5. 它无法区分可执行动作和危险动作。
6. 它无法稳定记录自己做过什么。
7. 它无法复盘一次任务为什么成功或失败。

所以，harness 的核心价值不是“让模型更聪明”，而是让模型能在真实系统中可靠工作。

面试表达：

```text
我会把 harness 理解成模型外层的运行时和控制层。模型负责生成决策，harness 负责把决策转成受权限约束的动作，比如读文件、改文件、跑测试、调用工具，再把环境反馈组织回上下文。没有 harness，coding agent 只是聊天模型；有了 harness，才可能成为可执行、可观测、可评估的工程系统。
```

## 1.4 Harness 不是什么

理解 harness，也要知道它不是什么。

Harness 不是模型本身。

模型负责生成 token，harness 负责控制 token 变成什么动作。换模型可以提升推理和代码能力，但不会自动解决权限、安全、日志、回放和工具执行问题。

Harness 不只是 prompt。

Prompt 可以告诉模型有哪些工具、应该如何思考、何时调用命令，但真正的工具注册、参数校验、权限确认、执行沙箱和结果记录都在系统层。

Harness 不只是 tool calling。

Tool calling 只是模型和外部函数交互的接口。Harness 还要管理上下文、状态、文件系统、终端、安全、用户交互、trace 和 evaluation。

Harness 不只是 evaluation harness。

传统 evaluation harness 多用于跑 benchmark、收集模型输出和算分。本书中的 agent harness 更广义，是智能体运行时系统。Evaluation harness 是其中一个子模块，用于离线评估、回放和回归测试。

Harness 不等于某个具体产品。

Claude Code、OpenCode、Codex、Cursor、Aider、SWE-agent、OpenHands 都可以看作不同形态的 coding agent 产品或框架。它们背后都会有某种 harness，只是设计取舍不同。

## 1.5 Harness、Agent、Runtime、Tool、Skill 的边界

这些词经常混在一起，需要先区分。

Agent：

```text
Agent 是能根据目标、上下文和环境反馈进行多步决策的系统行为模式。
```

它强调“目标驱动 + 多步行动 + 反馈循环”。

Runtime：

```text
Runtime 是让 agent 行为实际运行起来的执行环境。
```

它强调事件循环、状态管理、工具执行、错误处理和资源控制。

Harness：

```text
Harness 是更偏工程组织和控制的外层框架，负责把模型、runtime、工具、权限、上下文、日志和评估统一起来。
```

在很多语境里，agent runtime 和 agent harness 会重叠。本书会把 harness 作为更大的系统视角，把 runtime 看成其中的执行核心。

Tool：

```text
Tool 是 agent 可以调用的外部能力，例如读文件、写文件、搜索、执行 shell、运行测试、查询 API。
```

Tool 强调单个可调用动作。

Skill：

```text
Skill 是对一类任务能力的封装，可能包含 prompt、工具组合、工作流、知识和约束。
```

例如“代码审查 skill”可能包含读取 diff、检查风险、运行测试、输出 review 结构的完整流程。Skill 比 tool 更高层。

Memory：

```text
Memory 是 agent 在任务内或跨任务保留信息的机制，包括短期上下文、摘要、用户偏好和项目知识。
```

Trace：

```text
Trace 是一次任务执行过程的结构化记录，包括模型输入输出、工具调用、命令结果、文件 diff、权限确认和错误。
```

这些边界可以帮助你在面试中避免把所有东西都叫 agent。

## 1.6 一个 Coding Agent 的基本执行循环

最简单的 agent loop 可以写成：

```text
observe -> think -> act -> observe
```

在 coding agent 中，它会更具体：

```text
接收用户任务
-> 构造上下文
-> 调用模型
-> 解析模型动作
-> 权限校验
-> 执行工具或命令
-> 收集环境反馈
-> 更新状态和 trace
-> 判断是否继续
-> 输出最终结果
```

如果展开为一次修 bug 任务：

1. 用户说“修复登录接口测试失败”。
2. Harness 读取工作目录、相关文件、测试失败日志和 git 状态。
3. Context builder 把关键信息组织成模型输入。
4. 模型决定先搜索登录接口代码。
5. Harness 解析为 search 或 read file 工具调用。
6. Tool executor 执行读取或搜索。
7. Harness 把搜索结果返回模型。
8. 模型决定修改某个文件。
9. Harness 检查文件写权限和 patch 格式。
10. 文件编辑工具应用补丁。
11. 模型决定运行相关测试。
12. Harness 检查命令是否安全，必要时请求用户确认。
13. Terminal executor 运行测试，收集 stdout、stderr、exit code。
14. 如果失败，模型继续修；如果通过，模型总结改动和验证结果。
15. Trace logger 记录完整过程。

模型只是其中的决策引擎。真正让流程可执行的是 harness。

## 1.7 Harness 的核心模块

一个典型 coding agent harness 可以拆成以下模块。

User Interface：

1. 接收用户任务。
2. 展示模型计划、工具调用、文件 diff 和执行结果。
3. 支持用户中断、追问、批准或拒绝操作。

Task Manager：

1. 维护任务状态。
2. 区分当前目标、子任务、已完成动作和待处理问题。
3. 判断任务是否结束。

Context Builder：

1. 选择哪些文件、日志、历史对话和工具结果进入上下文。
2. 做摘要、压缩、裁剪和优先级排序。
3. 避免把无关或敏感内容塞进模型。

Model Adapter：

1. 适配不同模型 API。
2. 管理 system prompt、tool schema、streaming、重试和错误。
3. 处理模型输出格式差异。

Action Parser：

1. 从模型输出中解析工具调用、文件修改、命令执行或最终回答。
2. 校验结构化输出格式。
3. 处理模型输出不合法时的重试或修复。

Tool Registry：

1. 注册可用工具。
2. 定义工具 schema、权限级别和说明。
3. 管理工具版本和执行策略。

Execution Engine：

1. 执行工具调用。
2. 管理并发、超时、重试和错误。
3. 返回结构化结果。

File System Layer：

1. 读文件、列目录、搜索代码。
2. 应用 patch 或编辑文件。
3. 检查用户未提交改动，避免覆盖。

Terminal Executor：

1. 执行 shell 命令。
2. 捕获 stdout、stderr、exit code 和运行时间。
3. 限制危险命令和长时间命令。

Permission System：

1. 区分只读、写入、执行、高风险操作。
2. 对危险动作请求用户确认。
3. 记录授权和拒绝。

Sandbox：

1. 限制文件系统、网络、环境变量和系统命令访问。
2. 降低命令执行和工具调用风险。
3. 隔离不可信代码。

State Store：

1. 保存任务状态、上下文摘要、工具结果和中间计划。
2. 支持恢复、中断和继续。
3. 支持多任务或多 session。

Trace Logger：

1. 记录模型输入输出。
2. 记录工具调用和命令结果。
3. 记录文件 diff、权限确认和错误。
4. 支持 replay 和评估。

Evaluator：

1. 判断任务是否完成。
2. 跑测试、检查 diff、比较输出。
3. 用于离线 benchmark 和回归测试。

这些模块不是每个产品都完整具备，但越接近生产级 coding agent，越需要这些能力。

## 1.8 Harness 的核心价值

Harness 的价值可以概括为六点。

第一，把模型输出变成可执行动作。

模型输出本质是文本。Harness 通过 action parser、tool schema、execution engine，把文本映射到真实动作。

第二，把不可靠模型包裹在可控系统中。

模型可能幻觉、误判、输出非法 JSON、误调用工具。Harness 通过校验、权限、回滚、重试和用户确认降低风险。

第三，提供安全边界。

读文件、写文件、执行命令、访问网络都可能有风险。Harness 必须控制哪些动作允许自动执行，哪些动作需要用户确认，哪些动作永远禁止。

第四，提供可观测性。

当 agent 做错事时，团队需要知道它看到了什么、调用了什么、改了什么、命令返回什么。Trace 是 debug 和复盘的基础。

第五，提供评估和回放能力。

没有 replay，就很难比较两个 agent 版本。Evaluation harness 可以让同一批任务在不同模型、prompt、工具和策略下重复执行，比较成功率、成本和安全。

第六，支持复杂任务的多步执行。

真实任务不是一次问答，而是多步探索、修改、验证和修复。Harness 负责维护任务状态和环境反馈。

## 1.9 为什么模型越强，Harness 越重要

直觉上，模型越强，好像越不需要复杂工程。但真实情况相反：模型越能行动，越需要 harness 约束。

原因有三点。

第一，强模型会提出更复杂动作。

弱模型可能只会回答文本。强模型会主动搜索文件、修改多个模块、运行命令、调用外部工具。如果没有权限和沙箱，风险更大。

第二，强模型的错误更隐蔽。

强模型输出更流畅，可能让错误 diff、错误命令和错误解释看起来很合理。Harness 需要用测试、类型检查、权限、trace 和 review 抵消这种风险。

第三，强模型更适合长任务。

长任务需要状态、记忆、上下文压缩和回放。仅靠模型上下文很难稳定管理几十轮工具调用。

所以，harness 不是弱模型时代的补丁，而是 agent 系统走向生产的必要层。

## 1.10 一个最小 Harness 长什么样

一个教学版 coding agent harness 可以很小。

最小模块：

1. 用户输入。
2. 读取文件工具。
3. 写文件或 apply patch 工具。
4. 执行测试命令工具。
5. 模型调用。
6. 工具调用解析。
7. 简单权限确认。
8. 日志记录。

伪代码：

```text
state = init_task(user_request)

while not state.done:
    context = build_context(state)
    model_output = call_model(context, tool_schemas)
    action = parse_action(model_output)

    if action.type == "final_answer":
        state.done = true
        break

    if not permission_allowed(action):
        user_decision = ask_user(action)
        if user_decision == "deny":
            state.add_observation("User denied action")
            continue

    result = execute_action(action)
    trace.log(model_output, action, result)
    state.add_observation(result)
```

这个循环很简单，但已经包含 harness 的本质：上下文、模型、动作、权限、执行、反馈、日志。

生产级系统会在每一步增加大量工程细节：并发、超时、沙箱、文件冲突、上下文压缩、模型重试、token 成本、trace 存储、评估回放等。

## 1.11 Harness 的安全风险

Harness 把模型接到真实环境，因此安全风险比普通聊天机器人更高。

常见风险：

1. 模型误删文件。
2. 模型覆盖用户未提交改动。
3. 模型执行危险 shell 命令。
4. 模型读取敏感文件。
5. 模型把密钥写入日志。
6. Prompt injection 诱导 agent 执行越权动作。
7. 依赖安装脚本执行恶意代码。
8. 测试命令或构建命令长时间占用资源。
9. 工具返回内容污染模型上下文。
10. Trace 保存了隐私或机密信息。

基本防护：

1. 默认最小权限。
2. 读写分级授权。
3. 高风险命令确认。
4. 文件修改前展示 diff。
5. 不覆盖用户未提交改动。
6. 限制网络、环境变量和系统路径访问。
7. 日志脱敏。
8. 对外部内容标记为不可信。
9. 支持撤销、回滚和人工接管。

面试中要强调：安全不能只靠 prompt 约束，必须在 harness 层做硬控制。

## 1.12 Harness 的评估问题

Coding agent 的评估比普通问答更复杂。

普通问答评估通常比较最终答案。Coding agent 评估还要看：

1. 是否正确理解任务。
2. 是否找到相关文件。
3. 是否修改了正确位置。
4. 是否保持代码风格。
5. 是否通过测试。
6. 是否没有引入回归。
7. 是否没有越权操作。
8. 是否成本和时间可接受。
9. 是否能在失败后恢复。
10. Trace 是否可解释。

常见指标：

1. Task success rate。
2. Test pass rate。
3. Patch correctness。
4. Regression rate。
5. Tool call accuracy。
6. Unauthorized action rate。
7. Human intervention rate。
8. Average turns。
9. Token cost。
10. Wall-clock time。

Evaluation harness 的关键是可重复执行同一批任务。否则很难判断是模型更强、prompt 更好、工具更稳，还是只是运气更好。

## 1.13 Harness 和主流 Coding Agent 产品

虽然不同产品实现不同，但都绕不开 harness 问题。

Claude Code 这类终端 coding agent：

1. 强调命令行交互。
2. 需要读写本地代码库。
3. 需要执行命令和管理权限。
4. 需要长任务上下文和工具调用。

OpenCode 这类开源 coding agent：

1. 强调可扩展工具和多模型适配。
2. 需要清晰的 session、tool、permission 和 trace 设计。
3. 适合作为分析 harness 架构的工程样本。

Codex 或云端 coding agent：

1. 更强调远程沙箱和任务环境。
2. 需要自动运行测试、生成 patch、提交结果。
3. 对安全隔离和 evaluation harness 要求更高。

Cursor、Aider、OpenHands、SWE-agent 等系统：

1. 有的偏 IDE 交互。
2. 有的偏命令行 pair programming。
3. 有的偏自动修复 benchmark 任务。
4. 但都需要工具、文件、上下文、执行和评估。

后续章节会分别分析这些系统背后的共性设计。

## 1.14 常见误区

误区一：Harness 就是把工具列表塞进 prompt。

纠正：工具列表只是模型可见接口。真正的 harness 还包括工具执行、权限、安全、状态和日志。

误区二：模型足够强就不需要 runtime。

纠正：模型越能行动，越需要 runtime 控制风险和维护状态。

误区三：Coding agent 只要能改代码就行。

纠正：生产级 coding agent 还要能验证、回滚、避免覆盖用户改动、记录 trace 和支持评估。

误区四：Evaluation harness 和 agent harness 是一回事。

纠正：Evaluation harness 是评估子系统；agent harness 是完整运行框架。两者相关但不等价。

误区五：权限确认会降低体验，所以越少越好。

纠正：权限确认要分级。低风险只读操作可以自动化，高风险写入、删除、网络和系统命令必须谨慎。

## 1.15 面试题：Harness 是什么

回答要点：

```text
Harness 是模型外层的运行和控制框架。它负责接收用户任务，构造上下文，调用模型，解析模型动作，执行工具、文件编辑和终端命令，同时做权限、安全、状态管理、trace 日志和评估回放。没有 harness，LLM 只是生成文本；有了 harness，模型输出才能变成受控的可执行动作。
```

## 1.16 面试题：为什么 Coding Agent 不能只是模型 API

回答要点：

```text
因为 coding agent 需要和真实代码环境交互。它要读文件、改文件、跑测试、看错误、处理用户未提交改动、申请危险命令权限、记录执行过程，并在失败后恢复。模型 API 只负责生成 token，不负责文件系统、终端、安全、上下文压缩、trace 和评估。这些都需要 harness/runtime 层。
```

## 1.17 面试题：如何设计一个最小 Agent Harness

回答要点：

```text
我会先设计一个最小闭环：用户任务输入、context builder、model adapter、tool registry、action parser、execution engine、permission system、state store 和 trace logger。最小工具包括 read file、search、apply patch 和 run command。每轮循环中，harness 构造上下文，模型输出动作，系统解析并校验权限，执行动作，把结果写入 trace 和 state，再进入下一轮。后续再扩展 sandbox、memory、evaluation harness 和多模型适配。
```

## 1.18 小练习

1. 用自己的话解释 harness、runtime、tool、skill 的区别。
2. 画出一个 coding agent 从用户提问到修改文件再运行测试的执行链路。
3. 列出一个最小 coding agent harness 必须具备的 8 个模块。
4. 设计一个工具权限分级表，至少包含只读、写文件、执行命令和高风险命令。
5. 设计一次 agent 任务 trace，需要记录哪些字段。
6. 思考如果模型输出了非法工具调用，harness 应该如何处理。
7. 思考如果命令执行超时，harness 应该如何反馈给模型和用户。
8. 用 3 分钟回答“为什么 Claude Code/OpenCode/Codex 背后需要 agent runtime”。

## 1.19 本章总结

本章建立了第二十册的主线：Harness 是把模型能力变成可执行系统的工程层。

核心结论：

1. 模型负责生成决策，harness 负责让决策安全、可控、可执行。
2. Harness 不只是 prompt，也不只是 tool calling，而是上下文、工具、文件、终端、权限、状态、trace 和评估的统一框架。
3. Coding agent 的难点不只是“模型会不会写代码”，还包括能否安全读写文件、执行命令、验证结果和复盘过程。
4. 模型越强、任务越长、动作越多，harness 越重要。
5. 面试中理解 harness，能帮助你从“会调用 LLM API”升级到“能设计智能体工程系统”。

下一章会进入 Agent Runtime 架构，系统拆解 user interface、task manager、context builder、model adapter、tool registry、execution engine、permission system、state store 和 trace logger 如何协同工作。
