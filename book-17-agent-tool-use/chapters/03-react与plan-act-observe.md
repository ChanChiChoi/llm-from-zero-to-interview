# 第三章：ReAct 与 Plan-Act-Observe

Agent 的核心不是一次性生成完整答案，而是在任务执行过程中不断思考、行动、观察和调整。ReAct 和 Plan-Act-Observe 都是描述这种循环的经典框架。它们把模型的推理和外部动作连接起来，让模型不只会“想”，还会根据工具返回的现实反馈继续推进任务。

本章系统讲 ReAct 与 Plan-Act-Observe：为什么需要推理和行动交替，action 与 observation 分别是什么，什么时候应该先规划，什么时候应该边做边改，如何控制循环，如何处理失败，以及工程中如何实现稳定 Agent。

## 3.1 ReAct 是什么

ReAct 可以理解为 Reasoning + Acting。模型在每一步先根据目标和当前状态进行推理，然后选择一个动作，执行动作后读取观察结果，再进入下一轮。

典型结构：

```text
Thought: 我需要先查找相关信息
Action: search(query)
Observation: 搜索返回若干结果
Thought: 结果中第一篇更相关，需要打开阅读
Action: open(url)
Observation: 页面内容
Final: 给出答案
```

面试回答：

```text
ReAct 是把 reasoning 和 action 交替结合的 Agent 框架。模型不是一次性回答，而是在每轮根据当前状态思考下一步，调用工具或执行动作，然后根据 observation 更新判断。它适合需要外部信息、工具反馈和多步决策的任务，但需要控制最大步数、工具权限和错误恢复。
```

## 3.2 为什么要把 Thought 和 Action 分开

把 Thought 和 Action 分开有几个好处：

1. 模型先说明意图，再执行动作。
2. 工具调用更容易审计。
3. 错误更容易定位。
4. 系统可以在执行前做权限检查。
5. Observation 可以明确进入下一轮决策。

但实际产品中，不一定要把完整 thought 展示给用户。内部可以保留决策摘要和工具日志，用户侧只展示必要解释和最终结果。

## 3.3 Plan-Act-Observe 是什么

Plan-Act-Observe 是另一个常见 Agent 循环。

```text
Plan: 制定下一步计划
Act: 执行一个动作或工具调用
Observe: 读取执行结果
Update: 更新计划或状态
```

它更强调计划和状态更新。对于复杂任务，模型不应该每一步都临时发挥，而应该先形成一个可修改的计划，再根据观察结果调整。

例如代码修复任务：

```text
Plan: 先运行测试，定位失败，再修改最小代码
Act: run_tests
Observe: test_user_login failed
Plan update: 检查 login 函数边界条件
Act: read_file
Observe: 找到密码为空时未处理
Act: edit_file
Observe: 修改完成
Act: run_tests
Final: 测试通过
```

## 3.4 ReAct 和 Plan-Act-Observe 的区别

二者不是互斥关系。

ReAct 强调：

1. 推理和行动交替。
2. 每一步根据 observation 决策。
3. 适合工具使用和开放探索。

Plan-Act-Observe 强调：

1. 先有计划。
2. 执行后观察反馈。
3. 根据反馈更新计划。
4. 适合多步骤任务管理。

工程上经常组合使用：先让模型生成 plan，再按 ReAct 风格逐步 act 和 observe。

## 3.5 Action 是什么

Action 是 Agent 对外部环境做出的动作。

常见 action：

1. 调用搜索。
2. 查询数据库。
3. 读取文件。
4. 修改文件。
5. 运行测试。
6. 打开网页。
7. 点击按钮。
8. 调用业务 API。
9. 向用户提问。

Action 应该尽量结构化。例如使用 function calling 输出工具名和参数，而不是自由文本。结构化 action 更容易校验、执行、回放和评估。

## 3.6 Observation 是什么

Observation 是环境对 action 的反馈。

它可能是：

1. 工具返回的数据。
2. 错误信息。
3. 测试结果。
4. 页面状态。
5. API 响应。
6. 文件内容。
7. 用户补充信息。

Observation 的质量直接影响下一步决策。如果 observation 太长，模型可能抓不到重点；如果太短，模型可能缺少必要信息。工程上通常会对 observation 做结构化、截断、摘要和错误分类。

## 3.7 什么时候先规划

适合先规划的场景：

1. 任务步骤多。
2. 有多个子目标。
3. 工具调用成本高。
4. 错误代价高。
5. 需要用户确认。
6. 需要跨文件或跨系统操作。

例如“修复一个项目的测试失败”适合先规划；“查询今天北京天气”不需要复杂规划。

规划的价值是减少盲目行动，但计划不应该僵化。Agent 应该根据 observation 更新计划。

## 3.8 什么时候边做边改

适合边做边改的场景：

1. 信息一开始不完整。
2. 工具反馈很重要。
3. 环境状态会变化。
4. 每一步结果决定下一步。
5. 很难提前完整规划。

例如浏览器任务、调试任务、数据探索任务都适合边做边观察。

边做边改的风险是容易迷路。因此需要 step limit、状态摘要和明确停止条件。

## 3.9 循环控制

Agent 循环必须受控。

需要限制：

1. 最大步数。
2. 最大工具调用次数。
3. 最大 token。
4. 最大延迟。
5. 最大重试次数。
6. 高风险动作确认。
7. 终止条件。

终止条件可以是：

1. 任务完成。
2. 信息不足，需要用户补充。
3. 工具不可用。
4. 达到预算上限。
5. 发现目标不可完成。
6. 触发安全策略。

没有循环控制的 Agent 很容易无限调用工具或在错误方向上越走越远。

## 3.10 失败恢复

ReAct 系统中，失败是常态。

常见失败：

1. 工具参数错误。
2. 搜索结果无关。
3. 页面打不开。
4. 测试失败。
5. 权限不足。
6. API 超时。
7. 计划假设错误。

恢复策略：

1. 解析错误原因。
2. 修改参数重试。
3. 更换工具。
4. 请求用户补充。
5. 缩小任务范围。
6. 回滚高风险操作。
7. 停止并报告原因。

好的 Agent 不只是能成功执行 happy path，还要能面对失败做合理选择。

## 3.11 ReAct 的工程实现

一个简化实现：

```python
state = init_state(user_goal)

for step in range(max_steps):
    action = model.decide_next_action(state)
    if action.type == "final":
        return action.answer

    if not policy.allow(action):
        state.add_observation("action blocked by policy")
        continue

    observation = tool_executor.run(action)
    state.add_action(action)
    state.add_observation(observation)

return summarize_incomplete_state(state)
```

真实系统还需要参数校验、权限检查、异常捕获、日志记录、工具超时、重试和状态压缩。

## 3.12 日志和 Trace

Agent 必须记录 trace。

Trace 通常包括：

1. 用户目标。
2. 每一步 action。
3. 工具参数。
4. observation。
5. 错误和重试。
6. 最终结果。
7. token 和延迟。
8. 权限检查结果。

Trace 的价值：

1. 调试失败。
2. 评估工具调用质量。
3. 做安全审计。
4. 复现问题。
5. 训练后续模型。

没有 trace 的 Agent 很难上线维护。

## 3.13 ReAct 的常见失败模式

1. Thought 合理，但 action 选错。
2. Action 正确，但参数错误。
3. Observation 没读懂。
4. 反复调用同一个失败工具。
5. 计划不更新。
6. 过早 final。
7. 达成目标后还继续执行。
8. 工具输出注入影响下一步。
9. 状态上下文过长导致混乱。

这些问题需要通过 schema、controller、memory、policy 和评估共同解决。

## 3.14 面试题：ReAct 解决什么问题

回答要点：

```text
ReAct 解决的是模型只生成答案、不能根据外部反馈行动的问题。它把 reasoning 和 action 结合起来，让模型在每一步先判断当前状态，再调用工具或执行动作，然后根据 observation 更新下一步决策。它适合搜索、代码调试、浏览器任务和复杂工具调用，但需要控制循环、权限和失败恢复。
```

## 3.15 面试题：Plan-Act-Observe 如何落地

回答要点：

```text
我会让 Agent 先根据目标生成一个可修改计划，然后每次只执行一个或少量 action。工具返回 observation 后，系统更新 state，并让模型判断计划是否需要调整。工程上需要 action schema、工具执行器、observation 解析、step limit、预算管理、错误恢复和 trace 日志。高风险 action 要做权限检查或用户确认。
```

## 3.16 本章小结

ReAct 和 Plan-Act-Observe 都强调一个核心思想：Agent 应该在执行中学习环境反馈，而不是一次性猜完整答案。ReAct 把 reasoning 和 action 交替组织起来，Plan-Act-Observe 强调计划、执行、观察和更新。

真正可靠的 Agent 循环需要结构化 action、可理解 observation、清晰 state、受控 controller、失败恢复和 trace 日志。下一章会继续讨论 planning 与 task decomposition，进一步讲复杂任务如何拆解、排序、并行和动态调整。
