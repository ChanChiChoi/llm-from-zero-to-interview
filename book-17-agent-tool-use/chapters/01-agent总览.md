# 第一章：Agent 总览

Agent 是把大模型从“回答问题”推进到“执行任务”的关键形态。普通 chat model 主要根据用户输入生成一段回复；Agent 还会维护目标、拆解任务、选择工具、观察外部反馈、更新状态，并在多轮循环中完成任务。它不是一个单独模型名称，而是一种系统架构。

本章先建立 Agent 的全局框架：Agent 和普通聊天模型、RAG、workflow 的区别是什么；Agent 的核心组成有哪些；目标、状态、工具、观察、执行循环分别负责什么；哪些任务适合 Agent，哪些任务不适合；面试中如何准确表达 Agent 的价值和边界。

## 1.1 Agent 是什么

一个实用定义：Agent 是由 LLM 驱动、能围绕目标进行多步决策，并通过工具或环境反馈执行任务的系统。

它通常包含：

1. Goal：要完成的目标。
2. State：当前任务状态。
3. Tool：可调用的外部能力。
4. Observation：工具或环境返回的反馈。
5. Policy：决定下一步做什么。
6. Memory：保存历史和长期信息。
7. Controller：控制循环、预算、停止和失败恢复。

面试回答：

```text
Agent 不是单纯让模型回答问题，而是让模型在一个受控循环中围绕目标做多步决策。它会维护状态，选择工具，执行动作，读取观察结果，再决定下一步。相比普通 chat model，Agent 更强调任务执行、外部反馈、状态管理和失败恢复。
```

## 1.2 Agent 和 Chat Model 的区别

Chat model 的典型模式是：

```text
user input -> model response
```

Agent 的典型模式是：

```text
goal -> plan -> act -> observe -> update state -> next act -> final result
```

核心区别：

1. Chat model 主要生成文本，Agent 会执行动作。
2. Chat model 通常一轮或少量多轮，Agent 可能多步循环。
3. Chat model 的状态主要在上下文里，Agent 会显式维护任务状态。
4. Chat model 不一定调用工具，Agent 通常依赖工具和环境反馈。
5. Chat model 更像问答系统，Agent 更像任务执行系统。

不要把“能聊天的模型”都叫 Agent。只有当系统具备目标驱动、多步决策、工具调用和反馈闭环时，才更接近 Agent。

## 1.3 Agent 和 RAG 的区别

RAG 的核心是检索增强生成：先从知识库取相关内容，再让模型基于内容回答。

```text
query -> retrieve -> generate
```

Agent 可以包含 RAG，但不等于 RAG。Agentic RAG 会让模型主动决定是否检索、检索什么、是否需要二次检索、是否调用其他工具，以及如何根据观察结果继续行动。

对比：

1. RAG 主要解决知识获取问题。
2. Agent 主要解决多步任务执行问题。
3. RAG 的流程通常较固定。
4. Agent 的流程更动态。
5. RAG 可以是 Agent 的一个工具。

## 1.4 Agent 和 Workflow 的区别

Workflow 是预定义流程。比如：

```text
分类 -> 检索 -> 总结 -> 格式化输出
```

Agent 则允许模型根据状态动态决定下一步。

```text
如果信息不足 -> 检索
如果需要计算 -> 调用计算工具
如果执行失败 -> 重试或换方案
如果目标完成 -> 停止
```

Workflow 的优点是稳定、可控、容易测试；Agent 的优点是灵活、能处理开放任务。工程上常常采用混合模式：外层 workflow 控制关键流程，局部步骤交给 Agent 做动态决策。

## 1.5 Agent 的执行循环

Agent 最经典的循环是：

```text
Think -> Act -> Observe -> Think -> Act -> Observe -> Final
```

更工程化一点：

```text
read goal
inspect state
select action
call tool
receive observation
update state
check stop condition
repeat or finish
```

这个循环的关键不是让模型“想很多”，而是让模型把外部反馈纳入下一步决策。没有 observe，Agent 只是自说自话；没有 state，Agent 容易忘记任务进展；没有 stop condition，Agent 可能无限循环。

## 1.6 Goal：目标

Goal 是 Agent 要完成的任务。

好的 goal 应该明确：

1. 要做什么。
2. 完成标准是什么。
3. 允许使用哪些工具。
4. 有什么约束。
5. 什么时候停止。

不好的 goal 过于模糊，例如“帮我弄好这个项目”。更好的 goal 是“运行测试，定位失败原因，修改最小代码使测试通过，并说明修改内容”。

Agent 系统中，目标不清楚会导致工具误用、循环冗长和结果不可验收。

## 1.7 State：状态

State 是 Agent 当前知道的任务进展。

状态可以包含：

1. 用户目标。
2. 已完成步骤。
3. 当前计划。
4. 工具调用结果。
5. 错误和失败尝试。
6. 中间产物。
7. 剩余预算。
8. 安全约束。

状态管理是 Agent 的核心难点。上下文太少，模型会忘记；上下文太多，模型会被噪声干扰。实际系统通常会用结构化 state、摘要、日志和外部 memory 共同管理任务状态。

## 1.8 Tool：工具

工具是 Agent 能执行外部动作的接口。

常见工具：

1. 搜索工具。
2. 数据库查询。
3. 代码执行。
4. 文件读写。
5. 浏览器操作。
6. API 调用。
7. 计算器。
8. RAG 检索。
9. 日历、邮件、工单系统。

工具让 Agent 能接触外部世界，也带来风险。工具必须有 schema、权限、超时、错误处理和审计日志。越高权限的工具，越需要严格控制。

## 1.9 Observation：观察

Observation 是工具或环境返回的信息。

例如：

```text
测试失败：test_login_invalid_password failed
搜索结果：找到三篇相关文档
API 返回：权限不足
浏览器观察：页面出现验证码
```

Agent 必须根据 observation 调整下一步。如果工具失败，它应该判断是参数错、权限错、环境错，还是目标不可完成。成熟 Agent 的能力不在于从不失败，而在于能读懂失败并恢复。

## 1.10 Controller：控制器

Controller 是 Agent 系统的工程外壳，负责控制循环。

它通常管理：

1. 最大步数。
2. 最大 token。
3. 最大工具调用次数。
4. 超时。
5. 失败重试。
6. 权限检查。
7. 终止条件。
8. 日志和 trace。

没有 controller 的 Agent 很容易失控：无限循环、重复调用工具、乱改文件、成本不可控。工程上，Agent 的可靠性很大程度来自 controller，而不只是模型本身。

## 1.11 Agent 适合什么任务

适合 Agent 的任务通常有这些特点：

1. 需要多步执行。
2. 中途需要外部信息。
3. 需要根据反馈调整计划。
4. 有明确完成标准。
5. 可以通过工具验证结果。
6. 单步 workflow 难以覆盖所有情况。

例如：

1. 代码修复。
2. 数据分析。
3. 复杂检索和报告生成。
4. 自动化运维排查。
5. 浏览器任务。
6. 研究助理。
7. 工单处理。

## 1.12 Agent 不适合什么任务

不适合 Agent 的任务：

1. 简单问答。
2. 低延迟强约束任务。
3. 没有明确完成标准的开放聊天。
4. 高风险不可逆操作。
5. 工具权限无法安全控制的任务。
6. 失败代价很高且无法人工复核的任务。

Agent 不是越复杂越好。如果一个任务用一次模型调用或固定 workflow 就能稳定完成，就没有必要上 Agent。

## 1.13 Agent 的常见失败模式

1. 目标理解错误。
2. 计划过大或过细。
3. 工具选择错误。
4. 工具参数错误。
5. 忽略 observation。
6. 无限循环。
7. 状态污染。
8. 成本失控。
9. 越权操作。
10. 最终答案和执行结果不一致。

这些问题说明，Agent 的难点不是把工具列表塞给模型，而是让系统稳定地规划、执行、观察、修正和停止。

## 1.14 面试题：Agent 和普通 LLM 应用有什么区别

回答要点：

```text
普通 LLM 应用通常是输入到输出的生成系统，例如问答、摘要或分类。Agent 则是目标驱动的多步任务执行系统，它会维护状态、选择工具、执行动作、观察反馈并决定下一步。Agent 更适合需要外部信息、工具执行和动态决策的任务，但也更难控制成本、安全和稳定性。
```

## 1.15 面试题：Agent 系统有哪些核心模块

回答要点：

```text
我会把 Agent 系统拆成 goal、state、planner、tool registry、executor、observation handler、memory、controller 和 logger。Planner 决定下一步，tool registry 描述可用工具，executor 执行工具，observation handler 解析反馈，memory 维护历史，controller 控制预算和停止条件，logger 用于调试、审计和评估。
```

## 1.16 本章小结

Agent 的本质是目标驱动的多步任务执行系统。它和普通 chat model 的区别在于是否具备状态、工具、观察和执行循环；它和 RAG 的区别在于 RAG 主要增强知识获取，而 Agent 关注动态决策和任务完成；它和 workflow 的区别在于 Agent 更灵活，但也更难控制。

真正可靠的 Agent 不是“模型加工具”这么简单，而是需要清晰目标、结构化状态、受控工具、观察反馈、预算管理、失败恢复和安全边界。下一章会进入 tool use 与 function calling，具体讲 Agent 如何安全、稳定、可评估地调用工具。
