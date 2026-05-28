# 第二章：Agent Runtime 架构

## 2.1 本章目标

第一章从整体上解释了 harness 是什么：它是把模型能力变成可执行系统的工程层。本章进一步拆解 runtime 架构，也就是一个 agent 在真实环境里如何被调度、执行、观察、恢复和记录。

如果说 harness 是完整外壳，那么 runtime 就是其中的执行核心。它负责让 agent loop 真正跑起来：什么时候调用模型，什么时候执行工具，怎么保存状态，怎么处理失败，怎么中断和恢复，怎么把每一步写进 trace。

学完本章，你应该能回答：

1. Agent runtime 的核心组件有哪些。
2. 一次 agent step 从输入到输出经历了哪些数据流。
3. Runtime 如何管理任务状态和 session。
4. Runtime 如何处理模型输出非法、工具失败、命令超时和用户中断。
5. 为什么 trace、state store 和 permission system 是 runtime 的核心。
6. 面试中如何设计一个可恢复、可观测、可扩展的 agent runtime。

## 2.2 Runtime 和 Harness 的关系

Harness 和 runtime 经常混用，但可以做一个工程上的区分。

Harness 更像完整系统边界：

1. 用户交互。
2. 模型适配。
3. 工具系统。
4. 文件和终端。
5. 权限和沙箱。
6. 日志和评估。
7. 产品体验和配置。

Runtime 更像运行时执行内核：

1. 维护 session 和 task state。
2. 驱动 agent loop。
3. 调用模型。
4. 解析动作。
5. 调度工具执行。
6. 处理错误和重试。
7. 保存 trace 和中间结果。
8. 判断任务是否继续。

可以这样理解：

```text
Harness = Runtime + Tools + Permissions + Sandbox + UI + Evaluation + Product policies
Runtime = Agent loop + State management + Execution orchestration + Error handling
```

在实际系统中，两者边界不一定严格。但面试中这样拆分，有助于讲清架构层次。

## 2.3 核心组件总览

一个典型 agent runtime 包含以下组件：

1. User Interface。
2. Session Manager。
3. Task Manager。
4. Context Builder。
5. Model Adapter。
6. Action Parser。
7. Tool Registry。
8. Execution Engine。
9. Permission System。
10. State Store。
11. Trace Logger。
12. Error Handler。
13. Evaluator。

可以用一条数据流串起来：

```text
User Request
-> Session Manager
-> Task Manager
-> Context Builder
-> Model Adapter
-> Action Parser
-> Permission System
-> Execution Engine
-> Tool Result
-> State Store / Trace Logger
-> Next Step or Final Answer
```

Runtime 的目标不是让每个组件都复杂，而是让每个责任边界清楚。

如果责任混在一起，后续会出现几个问题：

1. 工具失败时不知道谁负责重试。
2. 上下文过长时不知道谁负责压缩。
3. 权限确认散落在各个工具里，难以审计。
4. trace 不完整，无法 replay。
5. 模型输出格式变化导致整个系统崩溃。
6. session 中断后无法恢复。

## 2.4 User Interface：用户交互层

User Interface 不只是聊天框。对 coding agent 来说，它还要展示和控制 agent 的行动。

UI 需要支持：

1. 接收用户任务。
2. 展示 agent 的计划和当前步骤。
3. 展示工具调用和命令执行。
4. 展示文件 diff。
5. 请求用户确认危险操作。
6. 支持用户中断、继续、补充要求。
7. 展示最终总结和验证结果。

终端型 coding agent 的 UI 重点是：

1. 流式输出。
2. 命令确认。
3. diff 展示。
4. 当前工作目录和文件路径。
5. 错误信息和下一步建议。

IDE 型 coding agent 的 UI 重点是：

1. 文件定位。
2. inline diff。
3. 代码选择区上下文。
4. 测试结果面板。
5. 用户接受或拒绝 patch。

UI 的核心原则：用户必须知道 agent 正在做什么，尤其是写文件、执行命令和访问外部资源时。

## 2.5 Session Manager：会话管理

Session 是一次用户和 agent 的交互上下文。

一个 session 里可能包含：

1. 用户任务。
2. 对话历史。
3. 当前工作目录。
4. 已读文件列表。
5. 已修改文件列表。
6. 工具调用历史。
7. 权限确认历史。
8. 上下文摘要。
9. 当前任务状态。
10. trace id。

Session Manager 要解决三个问题。

第一，隔离不同任务。

不同 session 的上下文、权限和文件修改不能混在一起。

第二，支持中断和恢复。

用户可能关闭终端、网络断开、模型调用失败、命令执行超时。Runtime 应该能从最近的安全状态恢复。

第三，支持用户接管。

用户可能手动修改文件或打断 agent。Session Manager 要检测环境变化，并把变化纳入下一轮上下文。

面试表达：

```text
我会把 session 看成 agent 任务的边界。它保存用户目标、对话历史、工具调用、文件 diff、权限确认和 trace id。好的 session manager 要支持隔离、中断恢复和用户接管，否则长任务很容易因为一次模型失败或命令超时就丢失上下文。
```

## 2.6 Task Manager：任务状态和子任务

Task Manager 管理的是“这件事做到哪里了”。

它和 Session Manager 的区别是：

1. Session 更偏交互和上下文边界。
2. Task 更偏目标、计划、步骤和完成状态。

一个 task state 可以包含：

```text
task_id
goal
current_phase
subtasks
completed_steps
pending_questions
modified_files
validation_status
failure_count
done
```

对于 coding agent，常见 phase 包括：

1. understanding：理解任务。
2. exploration：探索代码库。
3. planning：制定修改计划。
4. editing：编辑代码。
5. validating：运行测试或检查。
6. fixing：根据失败继续修复。
7. summarizing：总结结果。
8. done：任务结束。

Task Manager 的价值：

1. 防止 agent 迷失在长任务里。
2. 支持把大任务拆成子任务。
3. 支持失败后回到上一步。
4. 支持用户查看当前进度。
5. 支持 evaluation harness 判断任务是否完成。

## 2.7 Context Builder：上下文构造器

Context Builder 是 runtime 中最影响效果的模块之一。

它决定模型每一轮能看到什么。

输入来源包括：

1. 用户最新请求。
2. system prompt 和开发者约束。
3. 对话历史。
4. task state。
5. 已读文件片段。
6. 搜索结果。
7. 命令输出。
8. 测试失败日志。
9. 当前 diff。
10. 项目说明文件。
11. 相关工具 schema。
12. 权限和安全策略。

Context Builder 的难点：

1. 上下文窗口有限。
2. 代码库很大。
3. 命令输出可能很长。
4. 历史对话可能包含过时信息。
5. 外部内容可能不可信。
6. 敏感信息不能随便放入模型。

常见策略：

1. 只放和当前步骤相关的文件片段。
2. 对长日志做摘要，并保留关键错误行。
3. 对旧对话做 summary。
4. 对工具结果做结构化压缩。
5. 保留当前目标、约束和未完成事项。
6. 明确标记不可信外部内容。
7. 保留文件路径和行号，方便后续编辑。

Context Builder 的输出不是越多越好，而是要让模型看到“下一步决策最需要的信息”。

## 2.8 Model Adapter：模型适配层

Model Adapter 负责把 runtime 的请求转成具体模型 API 调用。

它要处理：

1. 不同模型的消息格式。
2. system/developer/user/tool message 的组织。
3. tool schema 或 function calling 格式。
4. streaming 输出。
5. token budget。
6. temperature、max tokens 等参数。
7. API 错误、限流和重试。
8. 模型不支持某些工具格式时的兼容。
9. 多模型路由。

为什么需要单独 adapter？

因为不同模型供应商的接口差异很大。如果业务逻辑直接依赖某个模型 API，后续切换模型、做 fallback 或 A/B test 会很困难。

Adapter 的设计原则：

1. 对 runtime 暴露统一接口。
2. 把模型差异封装在 adapter 内部。
3. 记录模型版本、参数和 token 用量。
4. 对 API 错误做标准化。
5. 支持模型能力声明，例如是否支持 tool calling、vision input、JSON mode。

## 2.9 Action Parser：动作解析器

模型输出不一定可靠。Action Parser 的任务是把模型输出解析成 runtime 可以理解的动作。

动作类型可以包括：

1. final_answer。
2. read_file。
3. search_code。
4. apply_patch。
5. run_command。
6. ask_user。
7. update_plan。
8. call_tool。

一个结构化 action 可以长这样：

```text
action_type: run_command
command: npm test -- login
working_dir: /repo
risk_level: medium
reason: verify login test after patch
```

Action Parser 要做校验：

1. action_type 是否存在。
2. 必填参数是否完整。
3. 参数类型是否正确。
4. 文件路径是否在允许目录内。
5. 命令是否包含危险片段。
6. 是否需要用户确认。

当模型输出非法时，runtime 可以：

1. 要求模型重新输出合法格式。
2. 自动修复轻微格式错误。
3. 返回错误 observation 给模型。
4. 终止任务并请求用户介入。

不要假设模型永远输出合法动作。

## 2.10 Tool Registry：工具注册表

Tool Registry 是 runtime 知道“有哪些工具可用”的地方。

每个工具至少要有：

1. 工具名。
2. 描述。
3. 参数 schema。
4. 返回 schema。
5. 权限等级。
6. 风险等级。
7. 超时设置。
8. 是否可自动执行。
9. 是否需要用户确认。
10. 版本信息。

示例：

```text
name: run_command
description: Execute a shell command in the current workspace.
args: command, working_dir, timeout
risk: medium/high depending on command
permission: requires confirmation for write, network, install, delete operations
timeout: 120s
```

Tool Registry 的常见问题：

1. 工具描述太模糊，模型选错工具。
2. 参数 schema 太宽，模型构造危险参数。
3. 工具没有版本，trace 无法复现。
4. 工具权限散落在执行逻辑中，难以审计。
5. 工具返回不结构化，模型难以理解。

好的 Tool Registry 不只是给模型看的工具列表，也是 runtime 的安全和治理入口。

## 2.11 Execution Engine：执行引擎

Execution Engine 负责真正执行 action。

它需要处理：

1. 工具调用。
2. 文件读写。
3. 命令执行。
4. 超时。
5. 取消。
6. 重试。
7. 并发控制。
8. 结果标准化。
9. 错误分类。

执行结果应该结构化：

```text
status: success | failed | timeout | denied | cancelled
stdout: ...
stderr: ...
exit_code: 0
duration_ms: 1234
artifacts: modified files, generated files, logs
error_type: permission_error | command_error | parse_error | timeout
```

为什么要结构化？

因为模型需要读懂结果，trace 需要回放，评估系统需要统计，用户需要审计。

Execution Engine 不能只是调用系统函数。它必须理解 agent 场景里的风险和反馈。

## 2.12 Permission System：权限系统

Permission System 决定 action 能否执行。

常见权限等级：

1. read：读取文件、列目录、搜索代码。
2. write：修改文件、创建文件、删除文件。
3. execute_safe：运行只读或低风险命令。
4. execute_risky：安装依赖、删除、移动、大规模修改、网络访问。
5. external_access：访问外部 API、网络、远程仓库。
6. secret_access：读取环境变量、密钥、凭证。

权限策略可以是：

1. always allow。
2. ask once per session。
3. ask every time。
4. deny by default。
5. allow only in sandbox。

例子：

```text
read file: allow by default inside workspace
apply patch: show diff and allow after user confirmation
run tests: allow if command matches safe allowlist
rm -rf: deny or require explicit high-risk confirmation
network install: ask every time
read .env: deny by default
```

权限系统必须在 runtime 层硬校验，不能只靠 prompt 告诉模型不要乱做。

## 2.13 State Store：状态存储

State Store 保存 runtime 运行过程中的状态。

它可以分为三类。

短期状态：

1. 当前步骤。
2. 当前上下文摘要。
3. 最近工具结果。
4. 未完成计划。

任务状态：

1. task id。
2. subtask 列表。
3. modified files。
4. validation status。
5. failure count。
6. permission decisions。

持久状态：

1. session history。
2. trace id。
3. 用户偏好。
4. 项目级记忆。
5. evaluation records。

State Store 的设计取舍：

1. 存太少，任务无法恢复。
2. 存太多，隐私和成本风险上升。
3. 存非结构化文本，后续难以查询和回放。
4. 存结构化状态，开发成本更高但更可靠。

生产系统通常需要结构化状态和生命周期管理。

## 2.14 Trace Logger：日志与回放

Trace Logger 是 agent runtime 的黑盒记录器。

一次完整 trace 应该记录：

1. 用户请求。
2. system prompt 和关键配置版本。
3. 模型版本和参数。
4. 每轮模型输入摘要。
5. 每轮模型输出。
6. 解析出的 action。
7. 权限判断和用户确认。
8. 工具执行结果。
9. 文件 diff。
10. 命令 stdout、stderr、exit code。
11. 错误和重试。
12. 最终答案。
13. token、时间和成本。

Trace 的价值：

1. Debug：定位 agent 为什么做错。
2. Replay：用新模型或新 prompt 重放任务。
3. Evaluation：统计成功率、工具准确率和成本。
4. Audit：审计文件修改和命令执行。
5. Training data：提取高质量轨迹或 bad cases。

Trace 也有风险：

1. 可能包含敏感代码。
2. 可能包含密钥和环境变量。
3. 可能包含用户隐私。
4. 可能包含企业内部信息。

所以 trace 必须有脱敏、权限和保留周期。

## 2.15 Error Handler：错误处理和恢复

Agent runtime 必须假设任何环节都会失败。

常见错误：

1. 模型 API 超时。
2. 模型输出非法格式。
3. 工具参数缺失。
4. 文件不存在。
5. Patch 应用失败。
6. 命令执行失败。
7. 命令超时。
8. 权限被拒绝。
9. 用户中断。
10. 上下文过长。
11. 工具返回内容过大。

错误处理策略：

1. retry：适合临时 API 错误。
2. repair：适合轻微格式错误。
3. ask model again：把错误作为 observation 返回模型。
4. ask user：需要业务判断或权限确认。
5. rollback：文件修改有问题时回退。
6. degrade：禁用高风险工具或切换简单模式。
7. abort：无法安全继续时终止。

例子：

```text
patch apply failed -> 把失败原因和冲突位置返回模型 -> 模型重新读取文件 -> 生成更小 patch -> 再次应用
```

错误处理能力决定了 agent 是“演示可用”还是“真实可用”。

## 2.16 Evaluator：运行时评估器

Evaluator 可以在 runtime 中做两类事。

第一，任务内验证。

例如：

1. 修改代码后运行测试。
2. 检查 lint。
3. 检查格式。
4. 检查 diff 是否只改了相关文件。
5. 检查是否出现敏感文件读取。

第二，离线评估。

例如：

1. 在 SWE-bench 类任务上跑 agent。
2. 比较不同模型版本成功率。
3. 比较不同工具策略成本。
4. 重放历史失败 trace。
5. 做 regression test。

Runtime 中的 evaluator 不一定是一个复杂模型。很多时候，最可靠的 evaluator 是测试、静态检查、规则和人工审核。

面试中要注意：不要把 LLM-as-a-judge 当成唯一评估方式。Coding agent 最重要的验证信号通常是测试和真实任务成功。

## 2.17 Runtime 的状态机设计

为了让 agent 可控，runtime 可以显式设计状态机。

示例状态：

```text
IDLE
RECEIVED_TASK
BUILDING_CONTEXT
CALLING_MODEL
PARSING_ACTION
WAITING_PERMISSION
EXECUTING_ACTION
OBSERVING_RESULT
UPDATING_STATE
WAITING_USER
COMPLETED
FAILED
CANCELLED
```

状态转移示例：

```text
RECEIVED_TASK -> BUILDING_CONTEXT -> CALLING_MODEL -> PARSING_ACTION
PARSING_ACTION -> WAITING_PERMISSION -> EXECUTING_ACTION
EXECUTING_ACTION -> OBSERVING_RESULT -> UPDATING_STATE -> BUILDING_CONTEXT
PARSING_ACTION -> COMPLETED
EXECUTING_ACTION -> FAILED
WAITING_PERMISSION -> CANCELLED
```

状态机的好处：

1. 每个阶段职责清楚。
2. 错误处理更明确。
3. 用户中断更好处理。
4. trace 更结构化。
5. evaluation harness 更容易重放。

很多看似复杂的 agent runtime，底层都可以理解成状态机加事件循环。

## 2.18 并发、取消和超时

生产 runtime 必须处理并发和取消。

并发场景：

1. 多个用户同时使用 agent。
2. 一个用户开启多个 session。
3. 一个任务中并行搜索多个文件。
4. 多个工具调用排队执行。

需要注意：

1. 同一个工作区的写操作不能乱并发。
2. 命令执行要有超时。
3. 用户取消后要终止后续工具调用。
4. 长任务要定期 checkpoint。
5. 文件状态变化要重新读取。

取消不是简单停止输出。

如果 agent 正在运行命令，runtime 要决定：

1. 是否杀掉子进程。
2. 是否保留部分输出。
3. 是否回滚文件修改。
4. 是否记录 cancelled 状态。
5. 是否允许用户继续恢复。

这些都是 runtime 层的职责。

## 2.19 Runtime 的可扩展性设计

一个好的 runtime 不应该只支持一个模型、一个工具和一种任务。

可扩展点包括：

1. 多模型 adapter。
2. 可插拔 tool registry。
3. 可配置 permission policy。
4. 多种 context builder。
5. 多种 execution backend，例如本地、容器、远程 sandbox。
6. 多种 trace sink，例如本地文件、数据库、可观测性平台。
7. 多种 evaluator。
8. skill 或 workflow 插件。

设计原则：

1. 核心 loop 稳定。
2. 外围能力插件化。
3. 工具 schema 标准化。
4. trace 格式稳定。
5. 权限策略集中管理。
6. 配置和版本可追踪。

扩展性不是一开始就做复杂插件系统，而是先把边界设计清楚，避免所有逻辑粘在一起。

## 2.20 一个最小 Runtime 伪代码

下面是一个更接近 runtime 的伪代码：

```text
def run_agent(session_id, user_request):
    session = session_manager.load_or_create(session_id)
    task = task_manager.create_task(user_request)
    trace = trace_logger.start(session, task)

    while not task.done:
        context = context_builder.build(session, task)
        trace.log_context(context.summary)

        model_response = model_adapter.generate(context)
        trace.log_model_response(model_response)

        action = action_parser.parse(model_response)
        if action.invalid:
            task.add_observation(action.error)
            continue

        if action.type == "final_answer":
            task.done = true
            task.final_answer = action.content
            break

        decision = permission_system.check(action, session)
        trace.log_permission(action, decision)

        if decision.denied:
            task.add_observation("Action denied")
            continue

        result = execution_engine.execute(action)
        trace.log_execution(action, result)

        error_handler.update_task(task, action, result)
        state_store.save(session, task)

        if evaluator.task_complete(task):
            task.done = true

    trace.finish(task.final_answer)
    return task.final_answer
```

这段伪代码不是为了实现细节，而是为了展示 runtime 的责任分层。

## 2.21 常见误区

误区一：Runtime 就是一个 while loop。

纠正：while loop 只是表面。真正重要的是状态、权限、错误处理、trace 和恢复。

误区二：工具执行失败就让模型再试一次。

纠正：有些错误可以重试，有些必须请求用户，有些必须回滚或终止。错误要分类处理。

误区三：上下文越多越好。

纠正：上下文要服务下一步决策。无关文件、长日志和过时历史会干扰模型。

误区四：Trace 只是 debug 日志。

纠正：Trace 还是 replay、evaluation、audit 和训练数据沉淀的基础。

误区五：权限系统放到工具内部即可。

纠正：权限策略应集中管理，工具内部可以二次校验，但不能每个工具各写一套规则。

## 2.22 面试题：Agent Runtime 包含哪些组件

回答要点：

```text
一个 agent runtime 通常包含 session manager、task manager、context builder、model adapter、action parser、tool registry、execution engine、permission system、state store、trace logger、error handler 和 evaluator。它的核心职责是驱动 agent loop：构造上下文、调用模型、解析动作、校验权限、执行工具、记录结果、更新状态，并决定是否进入下一轮。
```

## 2.23 面试题：Runtime 如何处理工具失败

回答要点：

```text
我会先把工具失败结构化，例如 parse_error、permission_error、timeout、command_error、not_found。不同错误采用不同策略：临时错误可以 retry，参数错误可以把错误反馈给模型重新生成，权限错误要请求用户或拒绝，patch 冲突要重新读取文件再生成更小 diff，危险错误要 abort 或 rollback。关键是不要把所有失败都简单交给模型盲目重试。
```

## 2.24 面试题：如何让 Agent 任务可恢复

回答要点：

```text
要让任务可恢复，需要 session manager、state store 和 trace logger 配合。Runtime 每轮保存 task state，包括目标、当前 phase、已执行 action、工具结果、文件 diff、权限决策和上下文摘要。长任务要定期 checkpoint。恢复时重新加载 session 和 task state，检查工作区文件是否变化，再从最近安全状态继续，而不是把整段历史重新丢给模型猜。
```

## 2.25 小练习

1. 画出 agent runtime 的核心组件图。
2. 用状态机表示一次 read file -> edit file -> run test -> fix 的过程。
3. 设计一个 action 数据结构，至少包含 action type、参数、风险等级和 reason。
4. 设计一个 execution result 数据结构，至少包含 status、stdout、stderr、exit code 和 error type。
5. 思考模型输出非法 JSON 时 runtime 应该如何处理。
6. 思考用户在 agent 修改文件后手动改了同一个文件，runtime 应该如何处理。
7. 设计一个 trace schema，用于支持 replay。
8. 用 3 分钟回答“如何设计一个可恢复的 coding agent runtime”。

## 2.26 本章总结

本章把 agent runtime 拆成了更具体的工程组件。

核心结论：

1. Runtime 是 agent loop 的执行内核，负责状态、调度、执行、错误处理和 trace。
2. Session Manager 管交互边界，Task Manager 管目标和进度。
3. Context Builder 决定模型每一轮看到什么，是影响效果的关键模块。
4. Model Adapter 隔离不同模型 API，Action Parser 把模型输出变成结构化动作。
5. Tool Registry、Execution Engine 和 Permission System 共同决定动作是否可执行、如何执行、是否安全。
6. State Store 和 Trace Logger 决定任务是否可恢复、可调试、可回放、可评估。
7. 生产级 runtime 必须处理错误、取消、超时、并发、权限和隐私。

下一章会进入 Coding Agent 工作流，重点讨论一个 coding agent 在真实代码库中如何探索、计划、编辑、验证和总结。
