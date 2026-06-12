# 第一章：Harness 总览

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，重点核对了 OpenAI Agents SDK 关于 agent loop、tool execution、guardrails、sandbox agents、sessions、human-in-the-loop 和 tracing 的公开文档，Claude Code 关于读写代码库、执行命令、MCP、CLAUDE.md、skills、hooks、权限和多环境运行的公开说明，OpenHands 关于 SDK、CLI、GUI、Cloud、RBAC / permissions 和 evaluation infrastructure 的公开 README，以及 SWE-agent / mini-SWE-agent 围绕 GitHub issue 自动修复、工具使用、SWE-bench 和 Agent-Computer Interface 的公开资料边界。

因此，本章只讨论 agent harness / coding agent runtime 的工程抽象：模型输出如何变成受控动作，工具和文件系统如何被系统层校验，终端命令如何受权限和沙箱约束，trace / replay / evaluation harness 如何支撑复盘。它不是某个具体产品的内部架构复刻，不提供绕过权限、执行危险命令、读取敏感文件或自动化高风险操作的技巧。

第二轮新增内容聚焦三点：

1. 把 harness 从“外层框架”落成可度量的运行闭环：action parse、permission gate、tool execution、observation use、state update、trace completeness、budget overrun 和 replay readiness。
2. 用公式区分模型能力、runtime 控制、工具执行、安全边界和评估回放，避免把 coding agent 能力全部归因给模型。
3. 补充一个 0 依赖 Python demo，用 toy agent trace 审计一个最小 harness 是否具备可执行、可控、可观测、可复盘的基本条件。

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

## 1.14.1 关键公式与 Harness 运行指标速查

把第 `i` 次 harness run 的第 `t` 个执行步抽象为：

$$
h_{i,t}=(x_i,c_{i,t},y_{i,t},a_{i,t},p_{i,t},o_{i,t},s_{i,t},\ell_{i,t},b_{i,t})
$$

其中，`x_i` 是用户任务，`c_{i,t}` 是上下文，`y_{i,t}` 是模型输出，`a_{i,t}` 是解析后的动作，`p_{i,t}` 是权限和确认结果，`o_{i,t}` 是工具或命令返回的 observation，`s_{i,t}` 是状态更新，`\ell_{i,t}` 是 trace 记录，`b_{i,t}` 是预算消耗。

定义指示函数：

$$
I(z)=
\begin{cases}
1,& z=\mathrm{true}\\
0,& z=\mathrm{false}
\end{cases}
$$

动作解析合法率：

$$
A_{\mathrm{parse}}=\frac{1}{N}\sum_{i,t} I(a_{i,t}\in A_{\mathrm{valid}})
$$

其中，`A_valid` 是 tool schema、patch schema、命令 schema 或 final answer schema 允许的动作集合。解析合法率低，说明模型输出和 action parser / tool schema 没有对齐。

工具执行成功率：

$$
S_{\mathrm{tool}}=\frac{\sum_{i,t} I(a_{i,t}\in A_{\mathrm{exec}})I(o_{i,t}=\mathrm{success})}{\sum_{i,t} I(a_{i,t}\in A_{\mathrm{exec}})}
$$

其中，`A_exec` 是实际进入执行器的动作集合。它不等于任务成功率，因为工具成功可能只是读对文件或跑完命令，最终 patch 仍可能错。

未授权执行率：

$$
R_{\mathrm{unauth}}=\frac{\sum_{i,t} I(p_{i,t}=\mathrm{blocked})I(a_{i,t}=\mathrm{executed})}{\sum_{i,t} I(p_{i,t}=\mathrm{blocked})}
$$

这个指标必须接近 0。只在 prompt 里告诉模型“不要做危险操作”，不能替代系统层 permission gate。

高风险确认覆盖率：

$$
C_{\mathrm{confirm}}=\frac{\sum_{i,t} I(a_{i,t}\in A_{\mathrm{risk}})I(p_{i,t}=\mathrm{confirmed})}{\sum_{i,t} I(a_{i,t}\in A_{\mathrm{risk}})}
$$

其中，`A_risk` 包含写文件、删除、网络、安装依赖、提交代码、外部 API 调用等高风险动作。不同产品会有不同风险分层，但必须有清晰策略。

Observation 使用率：

$$
U_{\mathrm{obs}}=\frac{\sum_{i,t} I(o_{i,t}\ne \emptyset)I(o_{i,t}\rightarrow s_{i,t+1})}{\sum_{i,t} I(o_{i,t}\ne \emptyset)}
$$

直觉：工具返回失败、测试报错或权限拒绝后，模型下一步必须真的读取并利用 observation。否则 agent 会出现“工具失败但最终声称完成”的 false completion。

状态更新覆盖率：

$$
C_{\mathrm{state}}=\frac{\sum_{i,t} I(s_{i,t}\ \mathrm{updated})}{\sum_{i,t} I(o_{i,t}\ne \emptyset)}
$$

没有 state update，长任务会丢失已经做过什么、失败原因是什么、下一步该验证什么。

Trace 完整率：

$$
C_{\mathrm{trace}}=\frac{1}{N}\sum_{i,t}\frac{|L_{i,t}\cap L_{\mathrm{req}}|}{|L_{\mathrm{req}}|}
$$

其中，`L_req` 至少包含 task、step、model output、action、arguments、permission、execution result、observation、state update、budget 和 final status。Trace 不完整时，debug、replay、安全审计和评估都会失真。

预算超限率：

$$
R_{\mathrm{budget}}=\frac{1}{M}\sum_i I(B_i>B_i^{\max})
$$

其中，`B_i` 可以是 step 数、tool call 数、token、成本、wall-clock time 或 retry 次数。没有 budget gate 的 agent 容易循环、重复搜索和成本失控。

一个保守的 harness 上线门禁可以写成：

$$
G_{\mathrm{harness}}=I(A_{\mathrm{parse}}\ge 0.95)I(S_{\mathrm{tool}}\ge 0.80)I(R_{\mathrm{unauth}}=0)I(C_{\mathrm{confirm}}\ge 0.80)I(U_{\mathrm{obs}}\ge 0.75)I(C_{\mathrm{state}}\ge 0.75)I(C_{\mathrm{trace}}\ge 0.90)I(R_{\mathrm{budget}}=0)
$$

这个门禁故意偏保守。真实产品可以按场景调阈值，但不能只用“最终 diff 看起来对”来证明 coding agent harness 可靠。

## 1.14.2 最小可运行 Harness Loop 审计 demo

下面的 demo 不调用真实模型、不执行真实命令，只用 toy trace 模拟 harness 的核心闭环。它检查：模型动作是否解析合法、权限是否阻断危险动作、工具执行是否成功、observation 是否被后续状态使用、trace 是否完整、预算是否超限，以及最终是否可 replay。

```python
from pprint import pprint

REQUIRED_TRACE_FIELDS = {
    "task",
    "step",
    "model_output",
    "action",
    "arguments",
    "permission",
    "execution_result",
    "observation",
    "state_update",
    "budget",
    "final_status",
}

EXEC_ACTIONS = {"read_file", "apply_patch", "run_tests", "run_shell"}
HIGH_RISK_ACTIONS = {"apply_patch", "run_shell"}


runs = [
    {
        "id": "fix_login_ok",
        "max_steps": 5,
        "max_cost": 6.0,
        "final_status": "completed",
        "final_verified": True,
        "actions": [
            {
                "action": "read_file",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": True,
                "observation_used": True,
                "state_updated": True,
                "cost": 0.8,
                "trace_fields": REQUIRED_TRACE_FIELDS,
            },
            {
                "action": "apply_patch",
                "parsed": True,
                "args_valid": True,
                "permission": "confirmed",
                "confirmed": True,
                "executed": True,
                "success": True,
                "observation_used": True,
                "state_updated": True,
                "cost": 1.5,
                "trace_fields": REQUIRED_TRACE_FIELDS,
            },
            {
                "action": "run_tests",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": True,
                "observation_used": True,
                "state_updated": True,
                "cost": 1.2,
                "trace_fields": REQUIRED_TRACE_FIELDS,
            },
        ],
    },
    {
        "id": "dangerous_shell_blocked",
        "max_steps": 4,
        "max_cost": 4.0,
        "final_status": "blocked",
        "final_verified": False,
        "actions": [
            {
                "action": "run_shell",
                "parsed": True,
                "args_valid": True,
                "permission": "blocked",
                "confirmed": False,
                "executed": False,
                "success": False,
                "observation_used": True,
                "state_updated": True,
                "cost": 0.4,
                "trace_fields": REQUIRED_TRACE_FIELDS - {"execution_result"},
            }
        ],
    },
    {
        "id": "test_failure_ignored",
        "max_steps": 5,
        "max_cost": 5.0,
        "final_status": "completed",
        "final_verified": False,
        "actions": [
            {
                "action": "apply_patch",
                "parsed": True,
                "args_valid": True,
                "permission": "confirmed",
                "confirmed": True,
                "executed": True,
                "success": True,
                "observation_used": True,
                "state_updated": True,
                "cost": 1.6,
                "trace_fields": REQUIRED_TRACE_FIELDS,
            },
            {
                "action": "run_tests",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": False,
                "observation_used": False,
                "state_updated": False,
                "cost": 1.1,
                "trace_fields": REQUIRED_TRACE_FIELDS - {"state_update"},
            },
        ],
    },
    {
        "id": "invalid_action_and_budget",
        "max_steps": 3,
        "max_cost": 3.0,
        "final_status": "failed",
        "final_verified": False,
        "actions": [
            {
                "action": "read_file",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": True,
                "observation_used": True,
                "state_updated": True,
                "cost": 1.0,
                "trace_fields": REQUIRED_TRACE_FIELDS,
            },
            {
                "action": "unknown_tool",
                "parsed": False,
                "args_valid": False,
                "permission": "blocked",
                "confirmed": False,
                "executed": False,
                "success": False,
                "observation_used": True,
                "state_updated": False,
                "cost": 0.8,
                "trace_fields": REQUIRED_TRACE_FIELDS - {"arguments", "execution_result"},
            },
            {
                "action": "read_file",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": True,
                "observation_used": False,
                "state_updated": False,
                "cost": 1.0,
                "trace_fields": REQUIRED_TRACE_FIELDS - {"state_update"},
            },
            {
                "action": "read_file",
                "parsed": True,
                "args_valid": True,
                "permission": "allowed",
                "confirmed": False,
                "executed": True,
                "success": True,
                "observation_used": False,
                "state_updated": False,
                "cost": 1.0,
                "trace_fields": REQUIRED_TRACE_FIELDS - {"budget"},
            },
        ],
    },
]


def ratio(num, den):
    return 0.0 if den == 0 else num / den


all_actions = [action for run in runs for action in run["actions"]]
executed = [action for action in all_actions if action["executed"]]
observed = [action for action in all_actions if action["executed"] or action["permission"] == "blocked"]
blocked = [action for action in all_actions if action["permission"] == "blocked"]
high_risk = [action for action in all_actions if action["action"] in HIGH_RISK_ACTIONS]

trace_scores = [
    len(action["trace_fields"] & REQUIRED_TRACE_FIELDS) / len(REQUIRED_TRACE_FIELDS)
    for action in all_actions
]

budget_overruns = []
for run in runs:
    total_cost = sum(action["cost"] for action in run["actions"])
    if len(run["actions"]) > run["max_steps"] or total_cost > run["max_cost"]:
        budget_overruns.append(run["id"])

blocked_unsafe = [
    run["id"] + ":" + action["action"]
    for run in runs
    for action in run["actions"]
    if action["permission"] == "blocked" and not action["executed"]
]

root_causes = {}
for run in runs:
    causes = []
    if not run["final_verified"] and run["final_status"] == "completed":
        causes.append("false_completion")
    if any(not action["parsed"] or not action["args_valid"] for action in run["actions"]):
        causes.append("invalid_action_schema")
    if any(action["executed"] and action["permission"] == "blocked" for action in run["actions"]):
        causes.append("permission_violation")
    if any(action["permission"] == "blocked" and not action["executed"] for action in run["actions"]):
        causes.append("blocked_by_permission")
    if any(action["executed"] and not action["observation_used"] for action in run["actions"]):
        causes.append("observation_ignored")
    if any(action["executed"] and not action["state_updated"] for action in run["actions"]):
        causes.append("state_not_updated")
    if run["id"] in budget_overruns:
        causes.append("budget_overrun")
    if not causes and run["final_verified"]:
        causes.append("pass")
    root_causes[run["id"]] = causes

metrics = {
    "parse_valid_rate": round(ratio(sum(a["parsed"] and a["args_valid"] for a in all_actions), len(all_actions)), 3),
    "tool_execution_success": round(ratio(sum(a["success"] for a in executed), len(executed)), 3),
    "unauthorized_execution_rate": round(ratio(sum(a["executed"] for a in blocked), len(blocked)), 3),
    "confirmation_coverage": round(ratio(sum(a["confirmed"] for a in high_risk), len(high_risk)), 3),
    "observation_use_rate": round(ratio(sum(a["observation_used"] for a in observed), len(observed)), 3),
    "state_update_coverage": round(ratio(sum(a["state_updated"] for a in observed), len(observed)), 3),
    "trace_completeness": round(sum(trace_scores) / len(trace_scores), 3),
    "budget_overrun_rate": round(len(budget_overruns) / len(runs), 3),
    "task_success_rate": round(ratio(sum(run["final_verified"] for run in runs), len(runs)), 3),
}

gates = {
    "parse_ok": metrics["parse_valid_rate"] >= 0.95,
    "tool_exec_ok": metrics["tool_execution_success"] >= 0.80,
    "permission_ok": metrics["unauthorized_execution_rate"] == 0.0,
    "confirmation_ok": metrics["confirmation_coverage"] >= 0.80,
    "observation_ok": metrics["observation_use_rate"] >= 0.75,
    "state_ok": metrics["state_update_coverage"] >= 0.75,
    "trace_ok": metrics["trace_completeness"] >= 0.90,
    "budget_ok": metrics["budget_overrun_rate"] == 0.0,
    "task_success_ok": metrics["task_success_rate"] >= 0.60,
}

failed_gates = [name for name, ok in gates.items() if not ok]
harness_gate_pass = all(gates.values())

print("metrics:")
pprint(metrics)
print("\nblocked_unsafe:")
pprint(blocked_unsafe)
print("\nbudget_overruns:")
pprint(budget_overruns)
print("\nroot_causes:")
pprint(root_causes)
print("\nfailed_gates:")
pprint(failed_gates)
print("\nharness_gate_pass:", harness_gate_pass)
```

一组可能输出如下：

```text
metrics:
{'budget_overrun_rate': 0.25,
 'confirmation_coverage': 0.667,
 'observation_use_rate': 0.7,
 'parse_valid_rate': 0.9,
 'state_update_coverage': 0.6,
 'task_success_rate': 0.25,
 'tool_execution_success': 0.875,
 'trace_completeness': 0.945,
 'unauthorized_execution_rate': 0.0}

blocked_unsafe:
['dangerous_shell_blocked:run_shell', 'invalid_action_and_budget:unknown_tool']

budget_overruns:
['invalid_action_and_budget']

root_causes:
{'dangerous_shell_blocked': ['blocked_by_permission'],
 'fix_login_ok': ['pass'],
 'invalid_action_and_budget': ['invalid_action_schema',
                               'blocked_by_permission',
                               'observation_ignored',
                               'state_not_updated',
                               'budget_overrun'],
 'test_failure_ignored': ['false_completion',
                          'observation_ignored',
                          'state_not_updated']}

failed_gates:
['parse_ok',
 'confirmation_ok',
 'observation_ok',
 'state_ok',
 'budget_ok',
 'task_success_ok']

harness_gate_pass: False
```

这里 `harness_gate_pass=False` 是故意设计的：它暴露了非法动作、预算超限、测试失败 observation 未使用、状态未更新和 false completion。一个真实 coding agent harness 的价值，正是把这些问题在系统层显性化，而不是只看最终回答是否自信。

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
