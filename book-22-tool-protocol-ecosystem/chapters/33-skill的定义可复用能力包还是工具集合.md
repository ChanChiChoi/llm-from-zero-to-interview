# 第 33 章 Skill 的定义：可复用能力包还是工具集合

前面我们讲了 Function Calling、MCP 和 A2A。它们分别解决了模型如何调用工具、Agent 如何连接工具资源、Agent 之间如何协作的问题。

从本章开始，我们进入第五部分：Skill、Plugin 与工具产品化。

这一部分讨论的问题会更偏产品化和平台化：当工具越来越多、Agent 越来越多、流程越来越复杂时，如何把一组能力打包成可安装、可启用、可禁用、可升级、可审核、可评估的能力单元？

这里就会出现一个词：Skill。

Skill 这个词在不同产品、不同平台里含义并不完全一样。有的系统把 Skill 当成一组工具，有的系统把 Skill 当成一个 Agent 能力包，有的系统把 Skill 当成某种 workflow，有的系统把 Skill 当成用户可安装的扩展。

本章先不纠结某一家平台的定义，而是从工程本质上讲清楚：Skill 到底是什么，它和 Tool、Plugin、Action、Workflow 有什么关系，为什么会需要 Skill 这个抽象。

你可以先记住一句话：

> Skill 不是单个工具，而是围绕某个任务目标封装的一组可复用能力、说明、约束、资源和运行方式。

## 33.1 为什么有了 Tool 还需要 Skill

如果只有一个工具，Tool 就够了。

例如：

```text
search_docs(query)
read_file(path)
query_database(sql)
send_email(to, subject, body)
```

但真实业务能力往往不是一个工具就能表达。

例如“生成周报”可能需要：

1. 查询项目进度。
2. 查询任务系统。
3. 汇总 Git 提交。
4. 读取会议纪要。
5. 按固定模板生成报告。
6. 检查敏感信息。
7. 发送给指定人群。

如果把这些都暴露成独立工具，让模型每次自己组合，问题很多：

1. 模型不知道正确顺序。
2. 模型容易漏步骤。
3. 模型不知道业务模板。
4. 模型不知道权限边界。
5. 模型不知道质量标准。
6. 模型每次都要重新规划。
7. 平台难以评估整体任务质量。

Skill 的价值就是把这些围绕同一目标的能力封装起来，让它变成一个可复用的任务能力。

## 33.2 Skill 的直观定义

可以把 Skill 理解成：

> 一个面向任务目标的能力包，包含工具、提示、资源、工作流、权限、配置、示例和评估标准。

例如，一个“周报生成 Skill”可以包含：

1. 能力描述：根据项目数据生成团队周报。
2. 工具集合：任务查询、代码提交查询、文档读取、邮件发送。
3. Prompt 模板：周报写作模板、风险总结模板。
4. Resource：团队周报规范、历史周报样例。
5. Workflow：先收集数据，再生成草稿，再审核，再发送。
6. 权限：读取项目数据、读取代码提交、发送邮件需要确认。
7. 配置：团队名称、收件人、报告语言、时间范围。
8. Eval：内容完整性、事实准确性、格式合规、敏感信息检查。

这已经明显超过单个 Tool 的范围。

## 33.3 Skill 不是简单工具集合

很多人会把 Skill 简化成“多个工具打包”。这只说对了一部分。

一个 Skill 当然可以包含多个工具，但它还应该包含“如何使用这些工具”的知识。

例如，一个“代码修复 Skill”如果只包含：

1. read_file。
2. search_code。
3. apply_patch。
4. run_tests。

这还不够。因为这些只是工具。

真正的代码修复 Skill 还应该知道：

1. 先复现问题，再修改代码。
2. 修改前读取相关上下文。
3. 尽量做最小改动。
4. 修改后运行相关测试。
5. 不要修改无关文件。
6. 生成补丁前展示变更说明。
7. 测试失败要保留日志和解释。

这些是任务策略、工作流和质量标准，不是工具本身。

所以，Skill 更像“能力包”，而不是“工具列表”。

## 33.4 Skill 的组成要素

一个工程上比较完整的 Skill 通常包含以下部分。

### 33.4.1 Manifest

Manifest 是 Skill 的元信息和能力声明。

它通常包含：

1. id。
2. name。
3. version。
4. description。
5. author / owner。
6. supported_tasks。
7. required_permissions。
8. tools。
9. resources。
10. prompts。
11. configuration。
12. safety_policy。
13. eval_spec。

一个简化 Manifest：

```json
{
  "id": "skill.weekly_report",
  "name": "Weekly Report Skill",
  "version": "1.0.0",
  "description": "Generate team weekly reports from project data and meeting notes.",
  "supported_tasks": ["generate_weekly_report"],
  "required_permissions": ["project.read", "docs.read", "email.send"],
  "tools": ["query_tasks", "read_docs", "send_email"],
  "prompts": ["weekly_report_template"],
  "resources": ["kb://team/weekly-report-guideline"]
}
```

### 33.4.2 Tools

Tools 是 Skill 可以使用的操作能力。

Skill 可以引用已有工具，也可以自带工具定义。

关键是：Skill 不只是暴露工具，还要说明这些工具在当前任务中的用途和限制。

### 33.4.3 Prompts

Skill 往往包含固定提示模板。

例如：

1. 报告模板。
2. 审核模板。
3. 提取信息模板。
4. 风格转换模板。
5. 安全检查模板。

Prompt 在 Skill 中不是随意文本，而是可版本化、可评估、可复用的任务模板。

### 33.4.4 Resources

Resources 是 Skill 需要参考的稳定上下文。

例如：

1. 规范文档。
2. 示例输出。
3. 术语表。
4. 业务规则。
5. 模板文件。
6. 安全策略。

一个“合同审查 Skill”如果没有合同条款规范和审查示例，效果会很不稳定。

### 33.4.5 Workflow

Workflow 描述任务步骤。

例如：

```text
collect_inputs -> retrieve_context -> draft_output -> validate -> request_approval -> execute_final_action
```

不是每个 Skill 都需要严格 workflow，但复杂 Skill 通常需要。

### 33.4.6 Configuration

Skill 通常需要配置。

例如：

1. 默认语言。
2. 默认输出格式。
3. 允许的数据源。
4. 收件人列表。
5. 风险阈值。
6. 是否需要人工确认。
7. 最大运行时间。

配置让同一个 Skill 可以适配不同团队或租户。

### 33.4.7 Permissions

Skill 需要声明权限。

例如：

1. 读文档。
2. 查数据库。
3. 写文件。
4. 发消息。
5. 调用外部 API。
6. 执行命令。

权限声明是安装、启用和审核 Skill 的基础。

### 33.4.8 Eval

Skill 应该有质量评估标准。

例如周报 Skill 的 eval 可以包括：

1. 是否覆盖关键项目。
2. 是否引用真实数据。
3. 是否遵守模板。
4. 是否没有泄露敏感信息。
5. 是否生成了可读结论。

没有 eval，Skill 很难持续迭代。

## 33.5 Skill 与 Tool 的区别

可以用表格区分：

| 维度 | Tool | Skill |
| --- | --- | --- |
| 粒度 | 单个操作 | 面向任务的能力包 |
| 输入输出 | 通常 schema 明确 | 可能包含多步输入输出 |
| 是否包含 workflow | 通常不包含 | 可以包含 |
| 是否包含 prompt | 通常不包含 | 经常包含 |
| 是否包含资源 | 通常只是访问资源 | 可以绑定任务相关资源 |
| 权限 | 操作权限 | 能力包权限集合 |
| 评估 | 调用成功率、参数正确率 | 任务成功率、质量、安全和用户满意度 |

一句话：

> Tool 是“能做一个动作”，Skill 是“能完成一类任务”。

## 33.6 Skill 与 Plugin 的区别

Plugin 更偏扩展机制，Skill 更偏能力语义。

Plugin 关注：

1. 如何安装。
2. 如何加载。
3. 如何声明入口。
4. 如何接入平台。
5. 如何管理版本。
6. 如何启用和禁用。

Skill 关注：

1. 能完成什么任务。
2. 需要哪些工具和资源。
3. 有哪些提示和流程。
4. 需要哪些权限。
5. 如何评估质量。

有些平台中，一个 Plugin 可以包含多个 Skill；有些平台中，一个 Skill 本身就是一个可安装 Plugin。概念关系取决于平台实现，但工程上可以这样理解：

> Plugin 是扩展载体，Skill 是能力单元。

## 33.7 Skill 与 Workflow 的区别

Workflow 是流程，Skill 是能力包。

一个 Skill 可以包含 Workflow，但 Skill 不等于 Workflow。

例如“客户投诉处理 Skill”可能包含：

1. 工单读取工具。
2. 客诉分类 prompt。
3. 客服规范资源。
4. 升级流程 workflow。
5. 权限策略。
6. 质量评估。

其中 workflow 只是 Skill 的一部分。

Workflow 更关注步骤顺序和控制流；Skill 更关注完整能力交付。

## 33.8 Skill 与 Agent 的区别

Skill 也不是 Agent。

Agent 是执行主体，Skill 是能力包。一个 Agent 可以安装多个 Skill，一个 Skill 也可以被多个 Agent 使用。

例如：

```text
Coding Agent
  - code_review_skill
  - bug_fix_skill
  - test_generation_skill

Business Agent
  - weekly_report_skill
  - meeting_summary_skill
  - customer_feedback_analysis_skill
```

Agent 负责推理、规划、调用工具和维护状态。Skill 提供任务能力、模板、工具组合和约束。

当然，有些产品会把“技能型 Agent”直接叫 Skill，这只是命名差异。工程上最好区分执行主体和能力包。

## 33.9 为什么 Skill 是产品化抽象

Skill 的价值不仅在工程复用，还在产品化。

一个工具能不能被调用，是工程问题；一个能力能不能被用户理解、安装、授权、评价和治理，是产品问题。

Skill 让平台可以支持：

1. 能力市场。
2. 团队安装。
3. 权限审核。
4. 版本升级。
5. 使用统计。
6. 质量评分。
7. 安全审核。
8. 企业内部复用。
9. 开发者生态。

如果只有零散工具，用户很难知道“我安装这个到底能完成什么任务”。Skill 把工具组合包装成面向任务的能力，用户更容易理解。

## 33.10 一个完整例子：会议总结 Skill

假设我们设计一个“会议总结 Skill”。

### 33.10.1 能力描述

```text
根据会议录音转写、聊天记录和会议文档，生成结构化会议纪要、行动项和风险列表。
```

### 33.10.2 需要的工具

1. 获取会议转写。
2. 读取会议文档。
3. 查询参与人信息。
4. 创建任务。
5. 发送纪要。

### 33.10.3 需要的资源

1. 公司会议纪要模板。
2. 行动项格式规范。
3. 项目术语表。
4. 历史优秀会议纪要样例。

### 33.10.4 工作流

```text
读取会议输入 -> 提取议题 -> 生成摘要 -> 提取行动项 -> 检查敏感信息 -> 生成草稿 -> 请求用户确认 -> 发送
```

### 33.10.5 权限

1. 读取会议转写。
2. 读取相关文档。
3. 读取参与人列表。
4. 创建任务。
5. 发送消息需要确认。

### 33.10.6 评估标准

1. 是否覆盖所有议题。
2. 行动项是否包含负责人和截止时间。
3. 是否没有编造未讨论内容。
4. 是否保留不确定项。
5. 是否遵守纪要模板。
6. 是否没有泄露敏感信息。

这个例子能看出，Skill 不是一个工具，而是完整能力包。

## 33.11 Skill 的边界如何划分

Skill 太小，会碎片化；Skill 太大，会变成“万能助手”。

好的 Skill 边界通常满足：

1. 有明确任务目标。
2. 用户能理解它解决什么问题。
3. 内部工具和资源围绕同一任务。
4. 可以独立安装和禁用。
5. 可以独立评估质量。
6. 权限范围相对清晰。
7. 版本升级不会影响无关能力。

不好的 Skill 示例：

```text
企业智能办公 Skill：可以处理所有办公任务。
```

太大。

也不好的示例：

```text
读取会议标题 Skill。
```

太小，更像 Tool。

更合理的是：

```text
会议总结 Skill
周报生成 Skill
合同审查 Skill
代码评审 Skill
客户反馈分析 Skill
```

## 33.12 Skill 生命周期

一个 Skill 通常会经历：

1. 开发。
2. 本地测试。
3. 安全审核。
4. 发布。
5. 安装。
6. 启用。
7. 使用。
8. 评估。
9. 更新。
10. 回滚。
11. 禁用。
12. 下架。

这说明 Skill 已经不是简单代码片段，而是需要平台治理的产品单元。

## 33.13 常见误区

### 33.13.1 把 Skill 当成工具列表

工具列表只说明“能调用什么”，不能说明“如何完成任务”。Skill 还需要 prompt、资源、workflow、权限和 eval。

### 33.13.2 Skill 边界过大

“万能办公 Skill”“企业助手 Skill”这种边界太大，难以评估、授权和维护。

### 33.13.3 Skill 边界过小

如果一个 Skill 只做一个简单 API 调用，那它更像 Tool。

### 33.13.4 忽略权限声明

Skill 安装时必须让用户或管理员知道它需要哪些权限。否则就是安全黑箱。

### 33.13.5 没有 Eval

没有评估标准，Skill 升级后是否变好无法判断。

## 33.14 面试高频题

### 题 1：Skill 是什么？

参考回答：

Skill 是围绕某类任务目标封装的可复用能力包，通常包含工具、提示模板、资源、工作流、配置、权限、示例和评估标准。它不是单个工具，而是面向任务交付的能力单元。

### 题 2：Skill 和 Tool 有什么区别？

参考回答：

Tool 是单个操作，例如查数据库、读文件、发邮件。Skill 是完成一类任务的能力包，可能包含多个 Tool、Prompt、Resource、Workflow 和权限策略。Tool 关注调用成功，Skill 关注任务成功。

### 题 3：Skill 和 Plugin 有什么区别？

参考回答：

Plugin 更偏扩展载体，关注安装、加载、入口和版本管理；Skill 更偏能力语义，关注能完成什么任务、需要哪些工具和资源、如何执行、如何授权和如何评估。一个 Plugin 可以包含多个 Skill，一个 Skill 也可以作为可安装 Plugin 发布。

### 题 4：如何划分 Skill 边界？

参考回答：

好的 Skill 应该有明确任务目标，用户能理解，内部工具和资源围绕同一任务，可以独立安装、禁用、评估和授权。太大的 Skill 难治理，太小的 Skill 更像 Tool。

### 题 5：为什么 Skill 需要 Eval？

参考回答：

因为 Skill 关注任务质量，不只是工具调用成功。Eval 可以评估事实准确性、格式合规、覆盖完整性、安全性、用户满意度和成本。没有 Eval，Skill 无法可靠升级和治理。

## 33.15 小练习

1. 为“合同审查 Skill”列出 tools、resources、prompts、workflow、permissions 和 eval。
2. 判断以下能力是 Tool 还是 Skill：read_file、生成会议纪要、query_database、代码评审、发送邮件。
3. 设计一个“客户反馈分析 Skill”的边界，不要太大也不要太小。
4. 写一个简化 Skill Manifest，包含 id、name、version、description、tools、permissions。
5. 思考：一个 Skill 是否应该允许自动执行高风险动作？为什么？

## 33.16 本章小结

本章我们讲了 Skill 的定义。

Skill 不是单个工具，也不只是工具集合。它是围绕某类任务目标封装的能力包，通常包含 Tool、Prompt、Resource、Workflow、Configuration、Permission 和 Eval。Tool 解决“一个动作怎么执行”，Skill 解决“一类任务怎么交付”。Plugin 更偏扩展载体，Workflow 更偏流程，Agent 更偏执行主体。

你可以把本章重点记成一句话：

> Skill 是工具生态从工程接口走向产品化能力的关键抽象，它让能力可以被安装、授权、复用、评估和治理。

下一章我们会继续区分 Plugin、Action、Tool、Skill、Workflow 这些容易混淆的概念，建立一套更清晰的工具生态术语体系。
