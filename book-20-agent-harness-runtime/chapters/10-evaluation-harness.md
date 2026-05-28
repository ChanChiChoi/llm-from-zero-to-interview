# 第十章：Evaluation Harness

## 10.1 本章目标

前一章讲了 trace、日志、回放和可观测性。本章讨论 Evaluation Harness：如何系统评估一个模型、一个 coding agent 或一个 agent runtime 是否真的变好了。

普通模型评估通常是“给 prompt，看答案，算分”。Agent 评估更复杂，因为 agent 会多步行动：读文件、写文件、运行命令、调用工具、处理权限、失败恢复。最终答案只是结果之一，过程是否安全、成本是否可接受、是否修改了无关文件、是否真的通过测试，同样重要。

学完本章，你应该能回答：

1. Evaluation harness 是什么。
2. 普通 LLM evaluation 和 agent evaluation 有什么区别。
3. 一个 evaluation harness 包含哪些核心组件。
4. Agent 评估如何初始化环境、执行任务和判断成功。
5. 如何设计指标、报告和 regression runner。
6. 评估可复现性为什么困难。
7. 面试中如何设计 coding agent evaluation harness。

## 10.2 Evaluation Harness 是什么

Evaluation harness 是自动运行评估任务、收集输出、计算指标、保存结果并支持复现的框架。

它的目标不是只跑一次 benchmark，而是形成稳定评估闭环：

```text
加载数据集
-> 初始化环境
-> 运行模型或 agent
-> 收集 trace 和结果
-> 计算指标
-> 生成报告
-> 对比 baseline
-> 发现回归
```

对普通 LLM，evaluation harness 主要处理：

1. prompt 模板。
2. 模型调用。
3. 输出解析。
4. 指标计算。
5. 结果存储。

对 agent，evaluation harness 还要处理：

1. 任务环境。
2. 工具和权限策略。
3. 多步 trace。
4. 文件 diff。
5. 命令执行。
6. 成功条件。
7. 成本和步数。
8. 过程安全。

一句话：

```text
Evaluation harness 是让 agent 能被稳定、可复现、可比较地测试的工程框架。
```

## 10.3 为什么 Agent Evaluation 更难

Agent evaluation 难在它不是单步输出。

难点包括：

1. 任务过程多步。
2. 工具调用具有副作用。
3. 环境状态会变化。
4. 模型输出有随机性。
5. 测试和依赖可能不稳定。
6. 成功标准不总是最终文本。
7. 过程安全也要评估。
8. 运行成本高。
9. replay 和复现困难。

例子：修复一个 bug。

不能只看 agent 最终说“已修复”。更应该看：

1. 是否修改了正确文件。
2. 是否通过相关测试。
3. 是否没有引入回归。
4. 是否没有修改无关文件。
5. 是否没有运行危险命令。
6. 是否成本和步数合理。
7. 失败时是否可复盘。

所以 agent evaluation 既评估结果，也评估过程。

## 10.4 核心组件

一个 evaluation harness 通常包含：

1. Dataset Loader。
2. Task Environment Manager。
3. Prompt / Instruction Template。
4. Model Adapter。
5. Agent Runner。
6. Tool / Sandbox Manager。
7. Trace Collector。
8. Metric Registry。
9. Result Store。
10. Report Generator。
11. Regression Runner。
12. Baseline Comparator。

数据流：

```text
Dataset
-> Task Environment
-> Agent Runner
-> Trace / Artifacts
-> Metrics
-> Result Store
-> Report
-> Regression Decision
```

每个组件都要版本化，否则结果很难比较。

## 10.5 Dataset Loader

Dataset Loader 负责加载评估任务。

任务可以来自：

1. 人工构造案例。
2. 历史线上 bad cases。
3. benchmark，例如 SWE-bench 类任务。
4. 内部 issue 和 bug fix 记录。
5. replay trace 转换出的 regression case。
6. 安全红队样例。

一个 agent eval task 应包含：

```text
task_id
task_type
user_instruction
repo_or_environment
initial_state
allowed_tools
permission_policy
success_criteria
timeout
expected_risks
metadata
```

任务类型可以包括：

1. Bug fix。
2. Feature implementation。
3. Refactor。
4. Test writing。
5. Code review。
6. Security task。
7. Tool-use task。

数据集要有分桶标签，否则只能看到总分，看不到哪里退化。

## 10.6 Task Environment Manager

Agent 评估必须初始化环境。

环境包括：

1. 代码仓库版本。
2. 依赖版本。
3. 工作目录。
4. 测试数据。
5. 环境变量。
6. 权限策略。
7. 工具版本。
8. sandbox 镜像。

环境初始化流程：

```text
checkout repo at commit
-> install or restore dependencies
-> apply task-specific initial patch if needed
-> configure sandbox
-> run precheck
-> start agent
```

环境不可复现是 agent eval 的常见问题。

例如同一个任务，今天测试通过，明天依赖升级后失败。这时不是 agent 退化，而是环境变了。

所以 evaluation harness 要记录：

1. git commit。
2. dependency lock。
3. container image。
4. tool version。
5. dataset version。
6. environment checksum。

## 10.7 Agent Runner

Agent Runner 负责执行 agent。

它要控制：

1. 模型版本。
2. Prompt 版本。
3. 工具集合。
4. 权限策略。
5. 最大轮数。
6. 最大时间。
7. 最大 token 成本。
8. 是否允许网络。
9. 是否允许写文件。
10. 是否自动运行测试。

Agent Runner 不是简单调用模型。它要运行完整 harness loop，并收集 trace。

常见终止条件：

1. agent 输出 final answer。
2. 任务成功条件满足。
3. 超过最大步数。
4. 超过最大时间。
5. 超过成本预算。
6. 触发安全拦截。
7. runtime error。

终止原因必须进入结果，否则无法分析失败。

## 10.8 Tool 和 Sandbox Manager

Agent eval 需要控制工具和环境副作用。

Tool Manager 负责：

1. 注册工具。
2. 固定工具版本。
3. 模拟或真实执行工具。
4. 记录工具调用。
5. 注入工具错误用于鲁棒性测试。

Sandbox Manager 负责：

1. 隔离文件系统。
2. 限制网络。
3. 控制环境变量。
4. 限制 CPU、内存和时间。
5. 清理环境。

有些评估可以使用 mock tools：

1. 成本低。
2. 可复现。
3. 安全。
4. 但和真实环境有差距。

有些评估必须使用真实工具：

1. 代码编辑任务。
2. 测试执行任务。
3. 文件系统任务。
4. 终端命令任务。

工具真实性和可控性之间需要 trade-off。

## 10.9 成功条件设计

成功条件是 agent eval 的核心。

坏成功条件：

```text
模型最终回答说任务完成。
```

好成功条件：

```text
相关测试通过，新增测试覆盖目标 bug，diff 只修改允许文件，未触发高风险权限，最终回答包含验证结果。
```

常见成功信号：

1. 单元测试通过。
2. 集成测试通过。
3. 目标文件 diff 满足规则。
4. 输出格式正确。
5. 人工或 judge 认为回答正确。
6. 没有安全违规。
7. 没有无关文件修改。

成功条件可以分层：

1. Hard pass：必须满足，例如测试通过。
2. Soft score：加分项，例如步数少、成本低。
3. Guardrail：不能违反，例如读取 secret。

Agent eval 不应该只看一个分数。

## 10.10 Metric Registry

Metric Registry 管理评估指标。

结果指标：

1. Task success rate。
2. Test pass rate。
3. Patch correctness。
4. Answer correctness。
5. Regression rate。

过程指标：

1. Tool call count。
2. Average turns。
3. Invalid action rate。
4. Retry count。
5. Human intervention rate。

成本指标：

1. Token cost。
2. Wall-clock time。
3. Tool execution time。
4. Sandbox cost。

安全指标：

1. Permission denied count。
2. Dangerous command attempt rate。
3. Secret access attempt rate。
4. Network access attempt rate。
5. Unrelated file modification rate。

可复现性指标：

1. Replay success rate。
2. Environment setup success rate。
3. Flaky task rate。

Metric Registry 的价值是统一定义指标，避免每次评估口径不同。

## 10.11 Result Store

Result Store 保存评估结果。

应保存：

1. run_id。
2. dataset version。
3. agent version。
4. model version。
5. prompt version。
6. tool version。
7. environment version。
8. task result。
9. metrics。
10. trace refs。
11. artifacts。
12. failure reason。

为什么要保存这么多版本？

因为评估结论必须可比较。

如果新 run 分数下降，团队需要知道是模型变了、prompt 变了、工具变了、数据集变了，还是环境变了。

## 10.12 Report Generator

评估报告不应该只给一个总分。

报告应包含：

1. 总体指标。
2. 分桶指标。
3. 和 baseline 对比。
4. 失败任务列表。
5. 典型 bad cases。
6. 成本和耗时。
7. 安全事件。
8. 回归任务。
9. flakiness 分析。
10. 是否建议发布。

示例报告摘要：

```text
Overall task success: 62.4% -> 65.1%
Bug fix bucket: +4.2%
Refactor bucket: -2.1%
Dangerous command attempts: unchanged
Token cost: +18%
Recommendation: do not roll out until refactor regression is investigated
```

好的报告要能指导决策，而不是只展示分数上涨。

## 10.13 Regression Runner

Regression Runner 用于防止新版本破坏旧能力。

Regression set 来源：

1. 历史线上失败。
2. 重要客户场景。
3. 安全红队样例。
4. 过去修复过的 bug。
5. 高价值 benchmark 子集。

Regression Runner 应支持：

1. 固定数据集版本。
2. 固定环境。
3. 固定权限策略。
4. 对比 baseline。
5. 阻止发布。
6. 输出失败 trace。

每次改 prompt、工具、模型、权限策略、context builder，都应该跑核心 regression set。

## 10.14 Baseline Comparator

没有 baseline，就没有可信评估。

Baseline 可以是：

1. 上一个稳定 agent 版本。
2. 只读助手。
3. 无工具模型。
4. 人类参考 patch。
5. 其他 agent 产品。
6. 简单规则系统。

比较时要确保：

1. 同一数据集。
2. 同一环境。
3. 同一工具权限。
4. 同一成本预算。
5. 同一成功条件。

否则比较不公平。

例如新 agent 成功率高，但给了更多工具权限和更大 token budget，这不一定说明 agent 本身更强。

## 10.15 Flaky Evaluation

Agent eval 很容易 flaky。

原因：

1. 模型非确定性。
2. 测试本身不稳定。
3. 依赖下载不稳定。
4. 环境差异。
5. 网络波动。
6. 时间相关逻辑。
7. 并发执行干扰。

处理策略：

1. 固定 seed 或降低 temperature。
2. 重复运行关键任务。
3. 标记 flaky tasks。
4. 分离环境失败和 agent 失败。
5. 缓存依赖。
6. 使用 sandbox snapshot。
7. 记录终止原因。

报告中应显示 flakiness，不要把不稳定任务当成确定结论。

## 10.16 Agent Evaluation 的安全维度

Agent 评估必须包含安全。

安全评估任务包括：

1. Prompt injection 文件。
2. 恶意 README。
3. 诱导读取 secret。
4. 诱导执行危险命令。
5. 诱导上传代码。
6. 恶意 install script。
7. 工具返回不可信内容。
8. 权限绕过尝试。

安全指标包括：

1. 越权尝试率。
2. 越权成功率。
3. 危险命令拦截率。
4. Secret 保护率。
5. 用户确认触发率。
6. 误拒率。

只评估 task success，不评估安全，会把危险 agent 误判成强 agent。

## 10.17 Agent Eval 和 Trace 的关系

Trace 是 agent eval 的关键输入。

Trace 可以用于：

1. 判断失败原因。
2. 计算工具调用次数。
3. 统计权限事件。
4. 构建 replay case。
5. 生成 bad case。
6. 分析成本和耗时。

如果没有 trace，eval 只能告诉你“失败了”，不能告诉你为什么失败。

Evaluation harness 应该默认收集 trace，并把 trace ref 写入 result store。

## 10.18 真实坑

常见真实坑：

1. Prompt template 版本不一致。
2. Benchmark 被污染。
3. Agent 评估环境不可复现。
4. 指标只看最终答案，不看过程安全。
5. 成功条件太宽，agent 说完成就算成功。
6. 工具权限不同导致 baseline 不公平。
7. 评估集只覆盖简单任务。
8. Flaky tests 被当成 agent 失败。
9. 没有保存 trace，失败无法归因。
10. 只看平均分，不看分桶和 bad cases。

经验法则：Agent evaluation 评估的是“模型 + runtime + 工具 + 环境 + 权限策略”的整体系统。

## 10.19 面试题：Evaluation Harness 是什么

回答要点：

```text
Evaluation harness 是自动运行评估任务、收集结果、计算指标、保存 trace 并生成报告的框架。对普通 LLM，它主要管理数据集、prompt、模型调用和指标；对 agent，它还要管理任务环境、工具、权限、sandbox、多步 trace、成功条件、成本和安全指标。它的目标是让不同 agent 版本可以稳定、公平、可复现地比较。
```

## 10.20 面试题：如何评估 Coding Agent 是否变好

回答要点：

```text
我不会只看最终成功率。首先要固定数据集、环境、模型、工具权限和成功条件，和 baseline 公平对比。指标上看 task success、test pass、patch correctness、回归率、工具调用次数、token 成本、耗时和安全事件。还要分桶看 bug fix、feature、refactor、review 等任务类型，并查看失败 trace 和 bad cases。只有质量提升且成本、安全、回归可接受，才说明 agent 真的变好。
```

## 10.21 面试题：Agent Eval 如何保证可复现

回答要点：

```text
需要版本化和环境固定。评估结果要记录 dataset version、repo commit、dependency lock、sandbox image、model version、prompt version、tool version、permission policy 和 random seed。每个任务要保存 trace 和 artifacts。对于外部网络、时间相关和 flaky tests，要隔离或标记。否则分数变化可能来自环境变化，而不是 agent 能力变化。
```

## 10.22 小练习

1. 设计一个 coding agent eval task schema。
2. 为 bug fix、feature、refactor 三类任务分别设计成功条件。
3. 设计一个 metric registry，包含质量、成本、安全和可复现性指标。
4. 思考如何从线上失败 trace 构造 regression set。
5. 设计一个 baseline comparator 的公平性检查清单。
6. 思考如何识别 flaky evaluation task。
7. 设计一份 agent eval report 模板。
8. 用 3 分钟回答“为什么 agent eval 不能只看最终答案”。

## 10.23 本章总结

本章讨论了 Evaluation Harness。

核心结论：

1. Evaluation harness 是让 agent 可稳定评估、可比较、可复现的工程框架。
2. Agent eval 比普通 LLM eval 更复杂，因为它包含环境、工具、副作用、权限、trace 和成功条件。
3. 数据集、环境、模型、prompt、工具、权限和指标都要版本化。
4. 成功条件要同时看结果、过程和 guardrails。
5. 评估指标应覆盖质量、成本、过程、安全和可复现性。
6. Baseline 对比必须公平，不能让新 agent 拥有更多权限或预算却直接比较分数。
7. Trace 是 eval 归因和 replay 的基础。
8. 安全评估和 regression runner 是生产级 agent harness 的必要组成。

下一章会进入 Claude Code 架构分析，把前面十章的 harness 概念映射到真实 coding agent 产品的设计视角上。
