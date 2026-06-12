# 第 34 章 Plugin、Action、Tool、Skill、Workflow 的边界

上一章我们讲了 Skill 的定义：Skill 是围绕某类任务目标封装的可复用能力包，通常包含工具、提示、资源、工作流、配置、权限和评估标准。

这一章继续解决一个常见混乱：Plugin、Action、Tool、Skill、Workflow 到底怎么区分？

这些词在不同平台里经常混用。例如有的平台把一次 API 调用叫 Action，有的平台把一组 API 叫 Plugin，有的平台把一个可安装能力叫 Skill，有的平台把固定流程叫 Workflow。名字不统一没关系，关键是工程设计时要知道每个抽象解决什么问题。

本章不追求给所有产品命名“判对错”，而是建立一套清晰的工程语义。

你可以先记住一句话：

> Action 是一次动作，Tool 是可调用能力，Workflow 是步骤编排，Skill 是任务能力包，Plugin 是扩展交付载体。

## 34.0 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 MCP Specification 中 tools / resources / prompts 的边界，Agent Skills 开放规范中 Skill 作为包含说明、脚本、参考资料和资产的可复用能力包的口径，OpenAI Apps SDK / Actions 中 tool descriptor、component / resource 和 action 接入的公开资料，以及 Microsoft Copilot Studio 等平台对 actions、plugins、connectors、workflows 和 agents 的常见命名方式。

需要先划清边界：

1. 本章不试图裁判各家平台命名谁对谁错，而是建立工程语义：执行实例、调用接口、流程编排、任务能力和扩展载体是五个不同边界。
2. Tool 在 MCP 和 Apps SDK 等资料中通常强调可调用能力、schema、权限和结果返回；Action 在很多产品里既可能指工具定义，也可能指一次具体执行。本章为了工程审计清晰，把 Action 固定为“执行实例”。
3. Skill 采用“面向任务的能力包”口径，可以包含 tools、prompts、resources、workflow、permissions、configuration 和 eval；它不等于单个 Tool，也不等于纯 Workflow。
4. Plugin / App 更偏安装、认证、分发、加载和版本治理；一个 Plugin 可以包含多个 Tool、Skill 或 Workflow。
5. 本章新增的公式和 Python demo 是教学用边界审计器，不实现真实插件系统、工作流引擎、权限系统、marketplace 或工具运行时。

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

## 34.15 五层边界审计指标与最小 demo

为了把这些概念讲得更可操作，可以把一个候选能力条目写成审计样本：

```math
e_i=(y_i,\hat{y}_i,f_i,g_i,p_i,v_i,a_i,n_i,l_i,z_i)
```

其中，`y_i` 是期望抽象类型，`\hat{y}_i` 是实际建模类型，`f_i` 是字段集合，`g_i` 是粒度，`p_i` 是权限层，`v_i` 是评估层，`a_i` 是审计字段，`n_i` 是命名别名是否说明，`l_i` 是生命周期治理，`z_i` 是 eval label。

对检查项 `j`，定义通过率：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[I_j(e_i)=1]
```

其中，`I_j(e_i)=1` 表示第 `i` 个条目在第 `j` 个边界检查上通过。

五层边界门禁可以写成：

```math
G_{\mathrm{ecosystem\_boundary}}
=\prod_{j\in\mathcal{J}}\mathbb{1}[C_j\ge \tau_j]
```

如果要对一个工具生态方案打分，可以用：

```math
S_{\mathrm{ecosystem\_boundary}}
=\sum_{j\in\mathcal{J}}w_j C_j,\qquad
\sum_{j\in\mathcal{J}}w_j=1
```

这里的关键不是公式复杂，而是把“命名混乱”变成可审计问题：这到底是一次执行、一个可调用接口、一个多步流程、一个任务能力包，还是一个可安装扩展？权限、评估、审计和生命周期应该落在哪一层？

下面是一个 0 依赖 demo，用 toy 条目检查五个抽象边界。

```python
from collections import OrderedDict

REQUIRED = {
    "Action": {"action_id", "tool_name", "input", "output", "caller", "status", "risk", "trace_id"},
    "Tool": {"name", "description", "input_schema", "output_schema", "permission", "timeout", "error_semantics"},
    "Workflow": {"workflow_id", "steps", "dependencies", "conditions", "retry_policy", "approval_points", "state_store"},
    "Skill": {"manifest", "task_goal", "tools", "resources", "prompts", "workflow", "permissions", "eval"},
    "Plugin": {"plugin_id", "manifest", "install", "auth", "version", "entrypoints", "dependencies", "contained_capabilities"},
}
GRANULARITY = {
    "Action": "execution",
    "Tool": "operation",
    "Workflow": "process",
    "Skill": "task_capability",
    "Plugin": "package",
}
PERMISSION_LAYER = {
    "Action": "execution_instance",
    "Tool": "operation_call",
    "Workflow": "step_or_branch",
    "Skill": "task_capability",
    "Plugin": "install_package",
}
EVAL_LAYER = {
    "Action": "execution_success",
    "Tool": "call_quality",
    "Workflow": "process_success",
    "Skill": "task_success",
    "Plugin": "package_health",
}


def make_item(**overrides):
    item = {
        "id": "support_plugin_ok",
        "expected": "Plugin",
        "actual": None,
        "fields": None,
        "granularity": None,
        "permission_layer": None,
        "eval_layer": None,
        "audit_fields": {"trace_id", "actor", "decision", "timestamp", "version"},
        "has_schema": True,
        "has_workflow": True,
        "has_task_goal": True,
        "has_install": True,
        "high_risk": False,
        "approval_point": True,
        "alias_documented": True,
        "lifecycle": {"install", "enable", "upgrade", "rollback", "disable"},
        "eval_label": "pass",
    }
    item.update(overrides)
    expected = item["expected"]
    if item["actual"] is None:
        item["actual"] = expected
    if item["fields"] is None:
        item["fields"] = REQUIRED[expected]
    if item["granularity"] is None:
        item["granularity"] = GRANULARITY[expected]
    if item["permission_layer"] is None:
        item["permission_layer"] = PERMISSION_LAYER[expected]
    if item["eval_layer"] is None:
        item["eval_layer"] = EVAL_LAYER[expected]
    return item


def required_fields(item):
    return REQUIRED[item["expected"]].issubset(item["fields"])


def abstraction_classification(item):
    return item["expected"] == item["actual"]


def granularity_boundary(item):
    return item["granularity"] == GRANULARITY[item["expected"]]


def action_trace(item):
    if item["expected"] != "Action":
        return True
    needed = {"action_id", "input", "output", "caller", "status", "trace_id"}
    return needed.issubset(item["fields"])


def tool_contract(item):
    if item["expected"] != "Tool":
        return True
    needed = {"input_schema", "output_schema", "error_semantics"}
    return item["has_schema"] and needed.issubset(item["fields"])


def workflow_control(item):
    if item["expected"] != "Workflow":
        return True
    needed = {"steps", "dependencies", "retry_policy", "state_store"}
    return item["has_workflow"] and needed.issubset(item["fields"])


def skill_bundle(item):
    if item["expected"] != "Skill":
        return True
    needed = {"tools", "resources", "prompts", "workflow", "permissions", "eval"}
    return item["has_task_goal"] and needed.issubset(item["fields"])


def plugin_packaging(item):
    if item["expected"] != "Plugin":
        return True
    needed = {"manifest", "auth", "version", "entrypoints"}
    return item["has_install"] and needed.issubset(item["fields"])


def permission_layering(item):
    return item["permission_layer"] == PERMISSION_LAYER[item["expected"]]


def eval_layering(item):
    return item["eval_layer"] == EVAL_LAYER[item["expected"]]


def audit_trace_layering(item):
    return {"trace_id", "actor", "decision", "timestamp"}.issubset(item["audit_fields"])


def naming_alias(item):
    return item["alias_documented"]


def governance_lifecycle(item):
    if item["expected"] != "Plugin":
        return True
    needed = {"install", "enable", "upgrade", "rollback", "disable"}
    return needed.issubset(item["lifecycle"])


def high_risk_approval(item):
    return (not item["high_risk"]) or item["approval_point"]


def eval_ready(item):
    return bool(item["eval_label"])


CHECKS = OrderedDict([
    ("required_field_coverage", required_fields),
    ("abstraction_classification", abstraction_classification),
    ("granularity_boundary", granularity_boundary),
    ("action_trace", action_trace),
    ("tool_contract", tool_contract),
    ("workflow_control", workflow_control),
    ("skill_bundle", skill_bundle),
    ("plugin_packaging", plugin_packaging),
    ("permission_layering", permission_layering),
    ("eval_layering", eval_layering),
    ("audit_trace_layering", audit_trace_layering),
    ("naming_alias", naming_alias),
    ("governance_lifecycle", governance_lifecycle),
    ("high_risk_approval", high_risk_approval),
    ("eval_ready", eval_ready),
])

ITEMS = [
    make_item(id="support_plugin_ok", expected="Plugin"),
    make_item(id="send_email_action_ok", expected="Action"),
    make_item(id="create_ticket_tool_ok", expected="Tool"),
    make_item(id="weekly_report_workflow_ok", expected="Workflow"),
    make_item(id="meeting_summary_skill_ok", expected="Skill"),
    make_item(id="tool_called_action_bad", expected="Action", actual="Tool", fields={"tool_name", "input"}, granularity="operation", permission_layer="operation_call", eval_layer="call_quality"),
    make_item(id="tool_without_schema_bad", expected="Tool", fields={"name", "description"}, has_schema=False),
    make_item(id="workflow_hides_delete_bad", expected="Workflow", high_risk=True, approval_point=False),
    make_item(id="skill_as_single_api_bad", expected="Skill", actual="Tool", fields={"tools"}, granularity="operation", has_task_goal=False, permission_layer="operation_call", eval_layer="call_quality"),
    make_item(id="plugin_without_install_bad", expected="Plugin", fields={"manifest", "entrypoints"}, has_install=False, lifecycle={"enable"}),
    make_item(id="permission_layer_mixed_bad", expected="Skill", permission_layer="install_package"),
    make_item(id="eval_layer_mixed_bad", expected="Workflow", eval_layer="task_success"),
    make_item(id="alias_not_documented_bad", expected="Tool", alias_documented=False),
    make_item(id="audit_missing_bad", expected="Action", audit_fields={"trace_id"}),
    make_item(id="eval_missing_bad", expected="Plugin", eval_label=""),
]

metrics = OrderedDict()
failed_by_item = OrderedDict()
for name, fn in CHECKS.items():
    passes = [fn(item) for item in ITEMS]
    metrics[name] = round(sum(passes) / len(passes), 3)
    for item, ok in zip(ITEMS, passes):
        if not ok:
            failed_by_item.setdefault(item["id"], []).append(name)

thresholds = {name: 0.95 for name in CHECKS}
failed_gates = [name for name, value in metrics.items() if value < thresholds[name]]

smoke = OrderedDict([
    ("action_is_execution", abstraction_classification(ITEMS[1]) and granularity_boundary(ITEMS[1])),
    ("tool_has_schema", tool_contract(ITEMS[2])),
    ("caught_skill_as_single_api", not skill_bundle(ITEMS[8])),
    ("caught_plugin_without_install", not plugin_packaging(ITEMS[9])),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_items=", list(failed_by_item.keys()))
print("failed_gates=", failed_gates)
print("ecosystem_boundary_gate_pass=", not failed_gates)
```

运行后可以看到：

```text
smoke= {'action_is_execution': True, 'tool_has_schema': True, 'caught_skill_as_single_api': True, 'caught_plugin_without_install': True}
metrics= {'required_field_coverage': 0.733, 'abstraction_classification': 0.867, 'granularity_boundary': 0.867, 'action_trace': 0.933, 'tool_contract': 0.933, 'workflow_control': 1.0, 'skill_bundle': 0.933, 'plugin_packaging': 0.933, 'permission_layering': 0.8, 'eval_layering': 0.8, 'audit_trace_layering': 0.933, 'naming_alias': 0.933, 'governance_lifecycle': 0.933, 'high_risk_approval': 0.933, 'eval_ready': 0.933}
failed_items= ['tool_called_action_bad', 'tool_without_schema_bad', 'skill_as_single_api_bad', 'plugin_without_install_bad', 'permission_layer_mixed_bad', 'eval_layer_mixed_bad', 'audit_missing_bad', 'alias_not_documented_bad', 'workflow_hides_delete_bad', 'eval_missing_bad']
failed_gates= ['required_field_coverage', 'abstraction_classification', 'granularity_boundary', 'action_trace', 'tool_contract', 'skill_bundle', 'plugin_packaging', 'permission_layering', 'eval_layering', 'audit_trace_layering', 'naming_alias', 'governance_lifecycle', 'high_risk_approval', 'eval_ready']
ecosystem_boundary_gate_pass= False
```

这段 demo 的价值在于：它让“概念边界”不再只是记忆题，而是能落到字段、权限、评估和审计层。面试中如果你能说明 Action 记录在审计日志、Tool 管在 registry、Workflow 管控制流、Skill 管任务能力、Plugin 管安装分发，就能避免把所有东西都叫“工具”。

## 34.16 面试高频题

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

## 34.17 小练习

1. 用 Plugin、Skill、Workflow、Tool、Action 五层拆解一个“会议纪要助手”。
2. 判断以下分别属于哪一层：send_email、send_email 给 Alice、周报生成、周报生成流程、企业邮箱扩展包。
3. 为一个客服 Plugin 设计 3 个 Skill 和 5 个 Tool。
4. 写出 Tool 评估和 Skill 评估的区别。
5. 思考：如果一个 Workflow 被很多 Skill 复用，它应该独立版本管理吗？为什么？
6. 运行本章 demo，把 `tool_called_action_bad` 改成正确的 Action 记录，补齐 action id、input、output、caller、status 和 trace id，再观察哪些指标恢复。
7. 给 demo 增加一个 `workflow_reused_by_two_skills_ok` 样本，说明 Workflow 可以独立版本管理，同时被多个 Skill 引用。

## 34.18 本章小结

本章我们建立了一套工具生态术语边界。

Action 是一次具体动作，Tool 是可调用能力，Workflow 是多步流程，Skill 是面向任务的能力包，Plugin 是可安装扩展载体。它们不是互斥关系，而是不同粒度和不同职责的抽象。

你可以把本章重点记成一句话：

> 名字可以因平台不同而变化，但设计时必须区分执行实例、调用接口、流程编排、任务能力和扩展载体这五个边界。

下一章我们会继续讲 Skill Manifest 与能力描述设计，也就是如何把一个 Skill 的身份、能力、工具、权限、配置和评估标准结构化写出来。
