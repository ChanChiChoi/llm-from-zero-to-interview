# 第 34 章 Plugin、Action、Tool、Skill、Workflow 的边界

上一章我们讲了 Skill 的定义：Skill 是围绕某类任务目标封装的可复用能力包，通常包含工具、提示、资源、工作流、配置、权限和评估标准。

这一章继续解决一个常见混乱：Plugin、Action、Tool、Skill、Workflow 到底怎么区分？

这些词在不同平台里经常混用。例如有的平台把一次 API 调用叫 Action，有的平台把一组 API 叫 Plugin，有的平台把一个可安装能力叫 Skill，有的平台把固定流程叫 Workflow。名字不统一没关系，关键是工程设计时要知道每个抽象解决什么问题。

本章不追求给所有产品命名“判对错”，而是建立一套清晰的工程语义。

你可以先记住一句话：

> Action 是一次动作，Tool 是可调用能力，Workflow 是步骤编排，Skill 是任务能力包，Plugin 是扩展交付载体。

## 34.1 为什么这些概念容易混淆

混淆主要来自三个原因。

第一，不同平台命名不同。某个平台叫 Tool 的东西，另一个平台可能叫 Action；某个平台叫 Plugin 的东西，另一个平台可能叫 Skill。

第二，这些概念确实有重叠。一个 Plugin 可以包含 Tools，一个 Skill 可以包含 Workflow，一个 Workflow 可以调用 Tools，一个 Action 可以是 Tool 的一次执行。

第三，工程视角和产品视角不同。工程师关心接口、schema、权限和运行时；产品经理关心用户看到什么、如何安装、如何授权、如何评价。

所以，区分这些概念时，不要只看名字，要看它在系统中的职责和粒度。

## 34.2 五个概念的一句话定义

先给出最简定义。

| 概念 | 一句话定义 | 典型问题 |
| --- | --- | --- |
| Action | 一次具体动作或操作实例 | 这次做了什么？ |
| Tool | 可被模型或 Agent 调用的操作能力 | 能调用什么？ |
| Workflow | 多步流程和控制逻辑 | 按什么步骤做？ |
| Skill | 面向任务的可复用能力包 | 能完成哪类任务？ |
| Plugin | 可安装、可加载、可分发的扩展载体 | 如何接入平台？ |

这张表是本章核心。

## 34.3 Action：一次具体动作

Action 通常指一次已经发生或即将发生的具体操作。

例如：

```text
发送一封邮件
点击一个按钮
创建一个工单
查询一次订单
应用一次补丁
运行一次测试
```

Action 强调“动作实例”。

例如 Tool 是：

```text
send_email(to, subject, body)
```

Action 是：

```text
send_email(to="alice@example.com", subject="周报", body="...")
```

也就是说，Tool 是能力定义，Action 是一次执行。

### 34.3.1 Action 的关键字段

一个 Action 记录通常包含：

1. action_id。
2. tool_name。
3. input。
4. output。
5. caller。
6. timestamp。
7. status。
8. risk_level。
9. confirmation。
10. trace_id。

示例：

```json
{
  "action_id": "act_123",
  "tool_name": "send_email",
  "input": {
    "to": "team@example.com",
    "subject": "Weekly Report"
  },
  "status": "completed",
  "risk_level": "medium",
  "confirmed_by": "user_001",
  "trace_id": "trace_abc"
}
```

Action 更偏日志、审计和执行记录。

## 34.4 Tool：可调用能力

Tool 是可被模型、Agent 或 Workflow 调用的操作能力。

Tool 通常有：

1. 名称。
2. 描述。
3. 输入 schema。
4. 输出 schema。
5. 权限。
6. 超时。
7. 错误语义。
8. 幂等要求。

例如：

```json
{
  "name": "create_ticket",
  "description": "Create a support ticket.",
  "input_schema": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "description": { "type": "string" },
      "priority": { "type": "string", "enum": ["low", "medium", "high"] }
    },
    "required": ["title", "description"]
  }
}
```

Tool 强调“可调用接口”。它不一定知道完整业务流程，只负责把一个操作做好。

### 34.4.1 Tool 与 Action 的区别

一句话：

> Tool 是定义，Action 是执行。

就像函数和函数调用的关系：

```text
函数定义：send_email(to, subject, body)
函数调用：send_email("alice@example.com", "Hello", "...")
```

在审计系统里，你通常记录 Action；在能力注册表里，你通常注册 Tool。

## 34.5 Workflow：多步流程

Workflow 是一组步骤和控制逻辑。

它关注：

1. 先做什么。
2. 后做什么。
3. 哪些步骤并行。
4. 失败如何处理。
5. 是否需要人工审批。
6. 条件分支如何选择。
7. 每一步调用哪些 Tool 或 Agent。

例如“生成并发送周报”的 Workflow：

```text
collect_project_updates
  -> collect_git_changes
  -> draft_report
  -> check_sensitive_info
  -> request_user_approval
  -> send_email
```

Workflow 不一定是 AI 特有概念。传统业务系统、自动化平台、CI/CD、数据流水线都有 Workflow。

### 34.5.1 Workflow 的关键字段

一个 Workflow 可以包含：

1. workflow_id。
2. steps。
3. dependencies。
4. conditions。
5. retry_policy。
6. timeout。
7. approval_points。
8. rollback_strategy。
9. input/output mapping。
10. state store。

示例：

```json
{
  "workflow_id": "weekly_report_workflow",
  "steps": [
    { "id": "collect_tasks", "type": "tool", "tool": "query_tasks" },
    { "id": "draft", "type": "prompt", "prompt": "weekly_report_template" },
    { "id": "approval", "type": "human_approval" },
    { "id": "send", "type": "tool", "tool": "send_email" }
  ]
}
```

### 34.5.2 Workflow 与 Tool 的区别

Tool 是单个可调用能力，Workflow 是多个步骤的编排。

一个 Workflow 可以调用多个 Tool，也可以调用 Agent、Prompt、Skill 或人工审批节点。

## 34.6 Skill：面向任务的能力包

Skill 是围绕某类任务目标打包的能力单元。

它可以包含：

1. Tools。
2. Prompts。
3. Resources。
4. Workflow。
5. Config。
6. Permissions。
7. Examples。
8. Eval。

例如“代码评审 Skill”包含：

1. 代码搜索工具。
2. 读取 diff 工具。
3. 安全规范资源。
4. 代码评审 prompt。
5. 风险检查 workflow。
6. 输出格式规范。
7. 质量评估标准。

Skill 强调“用户或 Agent 能复用的一类任务能力”。

### 34.6.1 Skill 与 Workflow 的区别

Workflow 是流程，Skill 是能力包。

一个 Skill 可以包含一个或多个 Workflow，但 Skill 还包含工具、提示、资源、权限、配置和评估。

例如会议总结 Skill 中，Workflow 只是“如何执行会议总结”的步骤；Skill 还包括会议纪要模板、术语表、权限声明和质量标准。

## 34.7 Plugin：扩展交付载体

Plugin 是可安装、可加载、可分发的扩展单元。

它更偏平台接入机制。

Plugin 可能包含：

1. 一个或多个 Tool。
2. 一个或多个 Skill。
3. 一个或多个 Workflow。
4. UI 配置。
5. 认证配置。
6. 运行时代码。
7. Manifest。
8. 依赖声明。

例如一个“GitHub Plugin”可能包含：

1. list_repos Tool。
2. read_issue Tool。
3. create_pr Tool。
4. code_review Skill。
5. issue_triage Workflow。
6. OAuth 配置。

Plugin 强调“如何把能力接入平台”。

### 34.7.1 Plugin 与 Skill 的区别

一句话：

> Plugin 是包装和分发方式，Skill 是能力语义。

一个 Plugin 可以包含多个 Skill。一个 Skill 也可以作为一个 Plugin 发布。具体关系取决于平台设计。

## 34.8 五者的层级关系

可以用一个简单层级理解：

```text
Plugin
  -> Skill
    -> Workflow
      -> Tool
        -> Action
```

但这个层级不是绝对的。

更准确地说：

1. Plugin 是分发和安装边界。
2. Skill 是任务能力边界。
3. Workflow 是流程边界。
4. Tool 是调用边界。
5. Action 是执行实例边界。

这五个边界分别回答不同问题：

| 边界 | 回答的问题 |
| --- | --- |
| Plugin | 这个扩展如何安装、加载、升级？ |
| Skill | 这个能力能完成哪类任务？ |
| Workflow | 这类任务按什么步骤执行？ |
| Tool | 每一步能调用什么操作？ |
| Action | 这一次具体执行了什么？ |

## 34.9 一个完整例子：客服工单插件

假设我们设计一个“客服工单 Plugin”。

### 34.9.1 Plugin 层

```text
customer_support_plugin
```

它负责把客服系统接入 AI 平台，包含认证、工具、技能和配置。

### 34.9.2 Tool 层

Plugin 暴露这些工具：

1. search_tickets。
2. read_ticket。
3. update_ticket。
4. create_ticket。
5. assign_ticket。
6. send_reply。

### 34.9.3 Skill 层

Plugin 包含这些 Skill：

1. 工单分类 Skill。
2. 客服回复草稿 Skill。
3. 客诉升级 Skill。
4. 用户反馈分析 Skill。

### 34.9.4 Workflow 层

“客诉升级 Skill”包含 Workflow：

```text
读取工单 -> 判断风险级别 -> 查询处理规范 -> 生成升级摘要 -> 指派负责人 -> 通知主管
```

### 34.9.5 Action 层

某次执行中发生的 Action：

1. read_ticket(ticket_id=123)。
2. assign_ticket(owner="risk_team")。
3. send_reply(to="user_456")。

这个例子能清楚看到五个概念的不同粒度。

## 34.10 决策树：该用哪个抽象

当你设计一个能力时，可以这样判断：

1. 这是一次已经发生的具体执行吗？如果是，用 Action。
2. 这是一个可调用的单步操作吗？如果是，用 Tool。
3. 这是多个步骤的固定或半固定流程吗？如果是，用 Workflow。
4. 这是围绕某类任务的一组工具、提示、资源和流程吗？如果是，用 Skill。
5. 这是要安装、分发、加载到平台的扩展包吗？如果是，用 Plugin。

例如：

| 能力 | 更适合的抽象 |
| --- | --- |
| read_file(path) | Tool |
| read_file("README.md") 这次调用 | Action |
| 先读文件再生成摘要再检查敏感信息 | Workflow |
| 文档总结能力，包含模板、资源、工具和评估 | Skill |
| 企业文档系统扩展包 | Plugin |

## 34.11 命名不一致时怎么办

真实工作中，你可能遇到平台已经固定命名。

例如某个平台把所有工具都叫 Action，另一个平台把插件里的 API 都叫 Tool，还有平台把 Skill 叫 App。

这时不要纠结名字，而要在设计文档里说明语义：

1. 本文中的 Action 指一次执行实例。
2. 本文中的 Tool 指可调用能力定义。
3. 本文中的 Skill 指面向任务的能力包。
4. 本文中的 Plugin 指可安装扩展包。
5. 本文中的 Workflow 指步骤编排。

只要团队内部语义一致，命名可以适配平台。

## 34.12 权限边界如何对应

不同抽象有不同权限边界。

### 34.12.1 Plugin 权限

安装 Plugin 时，需要审核它可能请求的全部权限。

例如 GitHub Plugin 可能请求 repo.read、repo.write、pull_request.write。

### 34.12.2 Skill 权限

启用 Skill 时，需要审核完成该类任务需要哪些权限。

例如代码评审 Skill 可能只需要读权限，而代码修复 Skill 需要写权限和测试执行权限。

### 34.12.3 Workflow 权限

Workflow 中某些步骤可能需要额外确认。

例如发送邮件、提交 PR、删除文件。

### 34.12.4 Tool 权限

Tool 调用前要检查具体操作权限。

例如 send_email 是否允许发给外部邮箱。

### 34.12.5 Action 权限

Action 是具体执行实例，审计时要记录当时是否授权、谁确认、输入输出是什么。

## 34.13 评估边界如何对应

不同抽象的评估指标也不同。

| 抽象 | 评估重点 |
| --- | --- |
| Action | 是否执行成功、是否符合授权、延迟、错误码 |
| Tool | 调用准确率、参数正确率、错误率、幂等性 |
| Workflow | 步骤成功率、分支正确率、整体耗时、恢复能力 |
| Skill | 任务成功率、输出质量、安全性、用户满意度 |
| Plugin | 安装成功率、兼容性、权限合理性、稳定性 |

这也是为什么不能把所有概念混成一个“工具”。不同层要看不同指标。

## 34.14 常见设计错误

### 34.14.1 把 Plugin 设计成万能大包

一个 Plugin 包含所有能力，权限巨大，版本难管，安全审核困难。

### 34.14.2 把 Skill 做得像单个 Tool

如果 Skill 只封装一个简单 API 调用，那它可能没有必要成为 Skill。

### 34.14.3 Workflow 里隐藏高风险动作

Workflow 中如果包含发送、删除、支付、生产变更等动作，必须显式标记审批点。

### 34.14.4 只记录 Tool，不记录 Action

审计需要知道具体执行了什么。只知道系统有 send_email 工具，不知道哪次给谁发了什么，是不够的。

### 34.14.5 Skill 没有质量标准

没有 Eval 的 Skill 无法判断升级后是否更好。

## 34.15 面试高频题

### 题 1：Action、Tool、Workflow、Skill、Plugin 怎么区分？

参考回答：

Action 是一次具体执行，Tool 是可调用能力定义，Workflow 是多步流程编排，Skill 是围绕某类任务的能力包，Plugin 是可安装、可加载、可分发的扩展载体。它们粒度不同，解决的问题也不同。

### 题 2：Tool 和 Action 的区别是什么？

参考回答：

Tool 是能力定义，例如 send_email(to, subject, body)。Action 是一次具体执行，例如 send_email 给某个用户发送某封邮件。注册表里管理 Tool，审计日志里记录 Action。

### 题 3：Skill 和 Workflow 的区别是什么？

参考回答：

Workflow 是步骤和控制流，Skill 是完整能力包。一个 Skill 可以包含 Workflow，但还包括工具、提示、资源、配置、权限和评估标准。Workflow 关注怎么执行，Skill 关注能完成哪类任务。

### 题 4：Skill 和 Plugin 的区别是什么？

参考回答：

Plugin 是扩展交付载体，关注安装、加载、权限申请、版本和分发。Skill 是能力语义，关注任务目标、工具组合、资源、提示、流程和评估。一个 Plugin 可以包含多个 Skill。

### 题 5：为什么要区分这些概念？

参考回答：

因为它们对应不同的权限边界、评估指标、审计粒度和产品体验。混在一起会导致权限过大、审计不清、能力难复用、质量难评估和平台难治理。

## 34.16 小练习

1. 用 Plugin、Skill、Workflow、Tool、Action 五层拆解一个“会议纪要助手”。
2. 判断以下分别属于哪一层：send_email、send_email 给 Alice、周报生成、周报生成流程、企业邮箱扩展包。
3. 为一个客服 Plugin 设计 3 个 Skill 和 5 个 Tool。
4. 写出 Tool 评估和 Skill 评估的区别。
5. 思考：如果一个 Workflow 被很多 Skill 复用，它应该独立版本管理吗？为什么？

## 34.17 本章小结

本章我们建立了一套工具生态术语边界。

Action 是一次具体动作，Tool 是可调用能力，Workflow 是多步流程，Skill 是面向任务的能力包，Plugin 是可安装扩展载体。它们不是互斥关系，而是不同粒度和不同职责的抽象。

你可以把本章重点记成一句话：

> 名字可以因平台不同而变化，但设计时必须区分执行实例、调用接口、流程编排、任务能力和扩展载体这五个边界。

下一章我们会继续讲 Skill Manifest 与能力描述设计，也就是如何把一个 Skill 的身份、能力、工具、权限、配置和评估标准结构化写出来。
