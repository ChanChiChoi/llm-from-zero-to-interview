# 第 36 章 Skill 的安装、启用、禁用和版本更新

上一章我们讲了 Skill Manifest：一个 Skill 如何声明身份、能力、工具、资源、权限、配置、安全策略和评估标准。

这一章继续讲 Skill 的生命周期管理。

当 Skill 只是一个本地脚本时，生命周期很简单：写好、运行、改掉。但当 Skill 进入企业平台或开放生态后，它就变成了一个需要治理的能力产品。它可能被多个团队安装，被不同用户启用，申请不同权限，经历多次升级，出现安全问题后需要禁用或回滚。

所以，Skill 平台不能只解决“怎么调用”，还要解决：

1. 谁能安装 Skill？
2. 安装时如何审批权限？
3. 安装和启用有什么区别？
4. Skill 可以对哪些用户、团队、租户生效？
5. 禁用后已有任务怎么办？
6. 版本升级如何兼容？
7. 灰度和回滚如何做？
8. 出现安全风险时如何紧急下架？

你可以先记住一句话：

> Skill 生命周期管理的核心，是让能力从“可用”变成“可控、可审计、可升级、可回滚”。

## 36.0 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 Agent Skills / OpenAI Skills 公开资料中 Skill 作为版本化文件包、`SKILL.md` manifest、版本指针、默认版本和 eval 的口径，Semantic Versioning 2.0.0 对 `MAJOR.MINOR.PATCH` 的兼容性语义，Kubernetes Deployment 对 rolling update / rollback 的工程范式，OpenFeature 对 feature flag、evaluation context 和 provider 的抽象，以及 MCP authorization / security best practices 中授权、scope、roots、审计和本地沙箱的治理原则。

需要先划清边界：

1. 本章讲 Skill 从发布、安装、配置、启用、使用、升级、灰度、回滚、禁用、卸载到紧急下架的生命周期治理，不绑定某一家平台的 marketplace、安装器、feature flag 系统、IAM 产品或版本字段。
2. 安装不等于启用，启用不等于每次调用都自动放行；运行时仍要检查权限、配置、版本、上下文策略和风险。
3. Skill 版本治理不只包括代码，也包括 manifest、prompt、workflow、tool 依赖、resource 依赖、配置 schema、安全策略和 eval 门槛。
4. 新增权限、扩大数据范围、改变输出契约、改变高风险 workflow 或改变外部共享策略，都应进入人工审批或至少显式策略门禁。
5. 本章新增的公式和 Python demo 是教学用生命周期审计器，不实现真实安装市场、权限审批系统、灰度发布平台、回滚引擎、任务队列或审计系统。

## 36.1 Skill 生命周期总览

一个完整 Skill 通常会经历这些阶段：

```text
develop -> submit -> review -> publish -> install -> configure -> enable -> use -> evaluate -> update -> rollback / disable / uninstall
```

可以拆成三条线：

1. 开发发布线：开发、提交、审核、发布。
2. 使用管理线：安装、配置、启用、使用、禁用、卸载。
3. 版本治理线：升级、灰度、回滚、废弃、下架。

不同平台可以简化，但生产系统至少要区分发布、安装、启用和使用。

## 36.2 发布、安装、启用、使用的区别

很多人会把这几个词混用。

### 36.2.1 发布

发布是开发者把 Skill 提交到平台或市场，让它成为可被发现的能力。

发布不代表任何用户已经能使用。发布只是进入目录。

### 36.2.2 安装

安装是某个用户、团队或租户把 Skill 加入自己的可用能力集合。

安装通常需要：

1. 查看 Manifest。
2. 审核权限。
3. 接受风险提示。
4. 完成认证配置。
5. 写入安装记录。

### 36.2.3 启用

启用是让 Skill 在某个范围内真正可被 Agent 或用户调用。

一个 Skill 可以已安装但未启用。例如管理员安装了合同审查 Skill，但只对法务团队启用。

### 36.2.4 使用

使用是一次具体调用或任务执行。

使用时需要再次检查权限、配置、版本和上下文策略。

一句话总结：

> 发布是进入平台，安装是加入租户，启用是开放调用，使用是实际执行。

## 36.3 安装流程设计

Skill 安装不是简单点一个按钮。

一个稳妥的安装流程可以是：

1. 用户或管理员选择 Skill。
2. 平台展示 Manifest 摘要。
3. 平台展示权限申请。
4. 平台展示数据访问范围。
5. 平台展示风险等级。
6. 管理员审批。
7. 配置认证和参数。
8. 执行安装前检查。
9. 写入安装记录。
10. 默认未启用或按策略启用。

安装记录可以包含：

```json
{
  "installation_id": "inst_123",
  "skill_id": "skill.contract_review",
  "version": "1.3.0",
  "tenant_id": "tenant_a",
  "installed_by": "admin_001",
  "installed_at": "2026-05-29T12:00:00Z",
  "approved_permissions": ["documents.read", "legal_policy.read"],
  "status": "installed"
}
```

安装记录很重要，因为后续审计需要知道：哪个版本、谁安装、批准了什么权限。

## 36.4 权限审批

安装 Skill 时，平台应该展示权限，而不是隐藏在技术配置里。

例如合同审查 Skill 申请：

1. documents.read：读取用户提供的合同。
2. legal_policy.read：读取内部法务政策。
3. artifacts.write：生成审查报告。

管理员需要知道每个权限的 reason。

权限审批至少要考虑：

1. 权限是否必要。
2. 权限是否过大。
3. 是否涉及敏感数据。
4. 是否允许外部传输。
5. 是否需要用户确认。
6. 是否符合租户策略。

如果 Skill 版本更新时新增权限，也必须重新审批。

## 36.5 配置管理

安装后通常需要配置。

配置分几类：

1. 租户级配置。
2. 团队级配置。
3. 用户级配置。
4. 环境级配置。
5. 安全策略配置。

例如会议总结 Skill：

```json
{
  "default_language": "zh-CN",
  "meeting_note_template": "company_standard_v2",
  "auto_send": false,
  "require_user_approval_before_send": true,
  "retention_days": 30
}
```

配置需要版本化或至少记录变更历史。因为同一个 Skill 行为变化，不一定来自代码升级，也可能来自配置变化。

## 36.6 启用范围

Skill 可以安装在租户里，但只对部分范围启用。

启用范围可以按：

1. 用户。
2. 团队。
3. 角色。
4. 项目。
5. 环境。
6. Agent。
7. 任务类型。

例如：

```json
{
  "skill_id": "skill.contract_review",
  "enabled": true,
  "scope": {
    "teams": ["legal", "procurement"],
    "roles": ["legal_reviewer", "procurement_manager"],
    "agents": ["agent.legal_assistant.v1"]
  }
}
```

启用范围越细，治理越灵活，但配置复杂度也越高。早期系统可以先支持租户级和团队级，后续再扩展到用户级和 Agent 级。

## 36.7 禁用与卸载

禁用和卸载也要区分。

### 36.7.1 禁用

禁用表示 Skill 暂时不能被调用，但安装记录、配置和历史数据保留。

适合场景：

1. 临时安全风险。
2. 质量回归。
3. 依赖工具故障。
4. 权限策略变化。
5. 管理员暂停使用。

### 36.7.2 卸载

卸载表示从租户或用户可用能力集合中移除 Skill。

卸载需要处理：

1. 配置是否删除。
2. Artifact 是否保留。
3. Trace 和 Audit 是否保留。
4. 未完成任务如何处理。
5. 依赖该 Skill 的 Workflow 是否失效。

通常审计数据不能随卸载删除。

## 36.8 禁用时正在运行的任务怎么办

这是一个很实际的问题。

如果 Skill 被禁用时，已有任务正在运行，可以有几种策略：

1. 允许当前任务完成，但不允许新任务启动。
2. 立即取消所有运行中任务。
3. 根据风险级别决定。
4. 进入暂停状态，等待管理员处理。
5. 切换到安全替代版本。

选择取决于禁用原因。

如果是普通维护，可以允许已有任务完成。如果是安全漏洞，应该立即停止相关任务，并标记可能受影响的 Artifact。

## 36.9 版本更新类型

Skill 升级不只是代码升级。可能变化的内容包括：

1. Manifest。
2. Prompt。
3. Workflow。
4. Tool 依赖。
5. Resource 依赖。
6. 权限。
7. 配置 schema。
8. Eval 标准。
9. 安全策略。

因此版本更新要说明变更类型。

常见版本类型：

1. patch：修 bug，不改变接口和权限。
2. minor：新增能力，保持兼容。
3. major：破坏性变更。
4. security：安全修复。
5. policy：权限或安全策略变化。

语义化版本可以作为参考：

```text
MAJOR.MINOR.PATCH
```

但 AI Skill 还要额外关注 prompt、eval 和安全策略变化。

## 36.10 兼容性检查

升级前要检查兼容性。

重点包括：

1. 输入 schema 是否变化。
2. 输出 schema 是否变化。
3. 权限是否新增。
4. 配置项是否新增或删除。
5. Workflow 步骤是否变化。
6. Tool 依赖是否变化。
7. Prompt 行为是否显著变化。
8. Eval 是否通过。

如果输出 schema 变化，上游 Workflow 或 Agent 可能解析失败。

如果权限新增，需要重新审批。

如果 prompt 变化导致输出风格变化，也可能影响下游系统。

## 36.11 灰度发布

Skill 升级最好不要一次性全量发布。

灰度可以按：

1. 用户比例。
2. 团队。
3. 租户。
4. Agent。
5. 任务类型。
6. 风险等级。

例如：

```json
{
  "skill_id": "skill.meeting_summary",
  "from_version": "1.2.0",
  "to_version": "1.3.0",
  "rollout": {
    "stage": "canary",
    "traffic_percent": 10,
    "eligible_teams": ["internal_test_team"]
  }
}
```

灰度期间要监控：

1. 成功率。
2. 错误率。
3. 用户反馈。
4. 安全拦截。
5. 成本。
6. 延迟。
7. Eval 指标。

## 36.12 回滚

回滚是版本治理必备能力。

回滚需要考虑：

1. 旧版本是否仍可用。
2. 配置是否兼容旧版本。
3. Artifact 是否受新版本影响。
4. 正在运行的任务是否切换。
5. Prompt 和 Resource 是否同步回滚。
6. 数据迁移是否可逆。

不是所有升级都能无痛回滚。如果 major 版本改变了输出结构或数据存储格式，回滚可能需要迁移脚本。

因此发布前应该准备 rollback plan。

## 36.13 自动更新还是手动更新

Skill 更新可以分为自动和手动。

适合自动更新的情况：

1. patch 修复。
2. 安全补丁。
3. 不改变权限。
4. 不改变输入输出 schema。
5. Eval 明显通过。

适合手动审批的情况：

1. 新增权限。
2. 改变输出格式。
3. 改变高风险 workflow。
4. 改变外部数据共享策略。
5. major 版本升级。

企业系统通常会允许管理员配置更新策略。

## 36.14 依赖管理

Skill 可能依赖工具、资源、Prompt、模型和其他 Skill。

依赖管理要解决：

1. 依赖是否存在。
2. 版本是否兼容。
3. 权限是否满足。
4. 依赖下线怎么办。
5. 依赖升级是否影响当前 Skill。

例如：

```json
{
  "dependencies": {
    "tools": [
      { "name": "read_document", "version": ">=1.0.0 <2.0.0" }
    ],
    "prompts": [
      { "id": "prompt.contract_review", "version": "2.1.0" }
    ],
    "resources": [
      { "uri": "kb://legal/policy", "version": "2026-05" }
    ]
  }
}
```

依赖不清，升级就会变成灾难。

## 36.15 审计日志

Skill 生命周期中的关键事件都要审计：

1. 发布。
2. 审核。
3. 安装。
4. 权限批准。
5. 配置变更。
6. 启用。
7. 禁用。
8. 卸载。
9. 升级。
10. 回滚。
11. 安全下架。
12. 每次高风险调用。

审计事件示例：

```json
{
  "event_type": "skill_enabled",
  "skill_id": "skill.contract_review",
  "version": "1.3.0",
  "tenant_id": "tenant_a",
  "enabled_by": "admin_001",
  "scope": {
    "teams": ["legal"]
  },
  "timestamp": "2026-05-29T12:00:00Z"
}
```

## 36.16 紧急下架

当 Skill 出现严重安全或质量问题时，平台需要支持紧急下架。

触发条件包括：

1. 数据泄露风险。
2. 权限越权。
3. 高风险幻觉。
4. 依赖被攻击。
5. 输出违反合规要求。
6. 大面积失败。

紧急下架流程：

1. 标记 Skill 为 suspended。
2. 阻止新任务启动。
3. 根据风险取消运行中任务。
4. 标记受影响 Artifact。
5. 通知管理员和受影响用户。
6. 保留审计证据。
7. 发布修复版本或回滚。

紧急下架能力是企业平台必须具备的安全阀。

## 36.17 一个完整例子：合同审查 Skill 升级

假设合同审查 Skill 从 1.3.0 升级到 1.4.0。

变化：

1. 新增“数据处理条款审查”能力。
2. 新增读取隐私政策资源的权限。
3. Prompt 升级到 v3。
4. 输出报告增加 privacy_risk 字段。

平台应该怎么处理？

1. 判断这是 minor 还是 major。因为输出 schema 变化，可能需要兼容检查。
2. 新增权限需要管理员审批。
3. 对依赖 Workflow 做兼容性测试。
4. 用 golden set 跑 eval。
5. 灰度给内部法务团队。
6. 监控风险识别率、引用准确率和误报率。
7. 无异常后逐步扩大。
8. 保留 1.3.0 作为回滚版本。

这个例子说明，Skill 升级不是简单替换文件。

## 36.18 常见误区

### 36.18.1 安装即启用

安装和启用应该分开。管理员可能先安装后配置，再选择范围启用。

### 36.18.2 升级不重新审核权限

只要新增权限或改变数据访问范围，就必须重新审核。

### 36.18.3 禁用后删除审计数据

审计数据不能因为卸载或禁用而删除，否则无法追责。

### 36.18.4 Prompt 变化不算版本变化

Prompt 变化会影响行为，应该纳入版本和 eval。

### 36.18.5 没有回滚计划

任何生产 Skill 升级都应该有回滚方案，尤其是高风险 Skill。

## 36.19 Skill 生命周期审计指标与最小 demo

为了把生命周期治理做成可验证系统，可以把一次 Skill 生命周期记录写成样本：

```math
\ell_i=(r_i,a_i,n_i,e_i,p_i,c_i,k_i,g_i,b_i,d_i,u_i,o_i,q_i,z_i)
```

其中，`r_i` 是发布审核，`a_i` 是安装审批，`n_i` 是启用范围，`e_i` 是运行时使用事件，`p_i` 是权限变化，`c_i` 是配置变化，`k_i` 是兼容性检查，`g_i` 是灰度策略，`b_i` 是回滚计划，`d_i` 是依赖版本，`u_i` 是禁用 / 卸载策略，`o_i` 是审计事件，`q_i` 是 eval / monitoring，`z_i` 是评估标签。

对检查项 `j`，定义通过率：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[I_j(\ell_i)=1]
```

灰度阶段可以同时看质量、安全、成本和延迟门禁：

```math
G_{\mathrm{canary}}
=\mathbb{1}[A_{\mathrm{succ}}\ge \tau_s]\,
\mathbb{1}[E_{\mathrm{err}}\le \tau_e]\,
\mathbb{1}[B_{\mathrm{safety}}=0]\,
\mathbb{1}[R_{\mathrm{cost}}\le \tau_c]
```

这里为了阅读直观，用 `A_{\mathrm{succ}}` 表示成功率，`E_{\mathrm{err}}` 表示错误率，`B_{\mathrm{safety}}` 表示安全拦截或未解决安全回归数量，`R_{\mathrm{cost}}` 表示相对成本。

Skill 生命周期上线门禁可以写成：

```math
G_{\mathrm{skill\_lifecycle}}
=\prod_{j\in\mathcal{J}}\mathbb{1}[C_j\ge \tau_j]
```

综合打分可以写成：

```math
S_{\mathrm{skill\_lifecycle}}
=\sum_{j\in\mathcal{J}}w_j C_j,\qquad
\sum_{j\in\mathcal{J}}w_j=1
```

这里的关键不是公式复杂，而是把“发布、安装、启用、升级、回滚、禁用、卸载”都变成平台能检查的事实：有没有审批、有没有 scope、有没有兼容性结果、有没有灰度监控、有没有回滚路径、有没有保留审计证据。

下面是一个 0 依赖 demo，用 toy lifecycle case 检查 Skill 生命周期治理问题。

```python
from collections import OrderedDict

REQUIRED_AUDIT = {"event_id", "event_type", "skill_id", "version", "actor", "tenant_id", "timestamp", "decision"}
REQUIRED_INSTALL = {"manifest_reviewed", "permission_reviewed", "risk_reviewed", "approver", "install_record"}
SAFE_RUNNING_POLICIES = {"allow_current_finish", "cancel_running", "pause_for_admin", "switch_to_safe_version"}


def make_case(**overrides):
    case = {
        "id": "contract_review_lifecycle_ok",
        "phase": "upgrade",
        "state_transition": ("1.3.0", "1.4.0"),
        "published": True,
        "review": {"manifest": True, "security": True, "eval": True, "owner": "legal-platform-team"},
        "install": {"manifest_reviewed": True, "permission_reviewed": True, "risk_reviewed": True, "approver": "admin_001", "install_record": True},
        "installed": True,
        "enabled": True,
        "enable_scope": {"tenant": "tenant_a", "teams": ["legal"], "agents": ["agent.legal_assistant.v1"]},
        "permissions_added": [],
        "permission_reapproved": True,
        "configuration": {"versioned": True, "change_history": True, "tenant_override": True},
        "running_task_policy": "allow_current_finish",
        "compatibility": {"input": True, "output": True, "config": True, "workflow": True, "prompt": True, "tool": True},
        "rollout": {"strategy": "canary", "traffic_percent": 10, "target": "internal_test_team", "monitors": ["success", "error", "safety", "latency", "cost"]},
        "metrics": {"success_rate": 0.97, "error_rate": 0.01, "safety_blocks": 0, "latency_p95_ms": 1800, "cost_ratio": 1.05},
        "rollback": {"old_version_available": True, "config_compatible": True, "prompt_resource_pinned": True, "migration_reversible": True, "plan": True},
        "dependencies": {"tools": {"read_document": ">=1.0.0 <2.0.0"}, "prompts": {"prompt.contract_review": "2.1.0"}, "resources": {"kb://legal/policy": "2026-05"}},
        "audit": {"event_id": "evt_001", "event_type": "skill_upgrade", "skill_id": "skill.contract_review", "version": "1.4.0", "actor": "admin_001", "tenant_id": "tenant_a", "timestamp": "2026-05-29T12:00:00Z", "decision": "approved"},
        "emergency": {"suspend_supported": True, "block_new_tasks": True, "running_task_control": True, "artifact_marking": True, "notification": True, "evidence_preserved": True},
        "uninstall": {"config_policy": "retain_30_days", "artifact_policy": "retain", "trace_audit_policy": "retain", "workflow_impact_checked": True},
        "eval": {"golden_pass": True, "regression_pass": True, "safety_pass": True, "canary_pass": True},
        "update_policy": "manual_approval",
        "expected_policy": "manual_approval",
    }
    case.update(overrides)
    return case


def review_before_publish(case):
    r = case["review"]
    return case["published"] and r.get("manifest") and r.get("security") and r.get("eval") and bool(r.get("owner"))


def install_approval(case):
    return case["installed"] and REQUIRED_INSTALL.issubset({k for k, v in case["install"].items() if v})


def enable_scope_control(case):
    scope = case["enable_scope"]
    return (not case["enabled"]) or bool(scope.get("tenant")) and bool(scope.get("teams") or scope.get("roles") or scope.get("agents"))


def permission_reapproval(case):
    return (not case["permissions_added"]) or case["permission_reapproved"]


def configuration_versioning(case):
    cfg = case["configuration"]
    return cfg.get("versioned") and cfg.get("change_history")


def running_task_policy(case):
    return case["running_task_policy"] in SAFE_RUNNING_POLICIES


def compatibility_check(case):
    return all(case["compatibility"].values())


def rollout_guard(case):
    rollout = case["rollout"]
    metrics = case["metrics"]
    monitors = set(rollout.get("monitors", []))
    monitor_ok = {"success", "error", "safety", "latency", "cost"}.issubset(monitors)
    canary_ok = rollout.get("strategy") in {"canary", "staged"} and 0 < rollout.get("traffic_percent", 0) <= 25
    metric_ok = metrics["success_rate"] >= 0.95 and metrics["error_rate"] <= 0.02 and metrics["safety_blocks"] == 0 and metrics["cost_ratio"] <= 1.2
    return canary_ok and monitor_ok and metric_ok


def rollback_readiness(case):
    return all(case["rollback"].values())


def dependency_versioning(case):
    deps = case["dependencies"]
    return all(deps.get(kind) for kind in ["tools", "prompts", "resources"])


def audit_log_completeness(case):
    return REQUIRED_AUDIT.issubset(case["audit"].keys())


def emergency_suspend_readiness(case):
    return all(case["emergency"].values())


def uninstall_retention(case):
    u = case["uninstall"]
    return u.get("trace_audit_policy") == "retain" and u.get("artifact_policy") in {"retain", "retain_with_ttl"} and u.get("workflow_impact_checked") is True


def eval_monitoring(case):
    return all(case["eval"].values())


def update_policy_accuracy(case):
    return case["update_policy"] == case["expected_policy"]


CHECKS = OrderedDict([
    ("review_before_publish", review_before_publish),
    ("install_approval", install_approval),
    ("enable_scope_control", enable_scope_control),
    ("permission_reapproval", permission_reapproval),
    ("configuration_versioning", configuration_versioning),
    ("running_task_policy", running_task_policy),
    ("compatibility_check", compatibility_check),
    ("rollout_guard", rollout_guard),
    ("rollback_readiness", rollback_readiness),
    ("dependency_versioning", dependency_versioning),
    ("audit_log_completeness", audit_log_completeness),
    ("emergency_suspend_readiness", emergency_suspend_readiness),
    ("uninstall_retention", uninstall_retention),
    ("eval_monitoring", eval_monitoring),
    ("update_policy_accuracy", update_policy_accuracy),
])

CASES = [
    make_case(id="contract_review_lifecycle_ok"),
    make_case(id="published_without_review_bad", review={"manifest": True, "security": False, "eval": True, "owner": "legal-platform-team"}),
    make_case(id="install_without_permission_review_bad", install={"manifest_reviewed": True, "permission_reviewed": False, "risk_reviewed": True, "approver": "admin_001", "install_record": True}),
    make_case(id="enabled_for_all_agents_bad", enable_scope={"tenant": "tenant_a", "teams": [], "agents": []}),
    make_case(id="new_permission_no_reapproval_bad", permissions_added=["privacy_policy.read"], permission_reapproved=False),
    make_case(id="config_change_unversioned_bad", configuration={"versioned": False, "change_history": False, "tenant_override": True}),
    make_case(id="disable_no_running_policy_bad", running_task_policy="unknown"),
    make_case(id="breaking_output_no_compat_bad", compatibility={"input": True, "output": False, "config": True, "workflow": True, "prompt": True, "tool": True}),
    make_case(id="full_rollout_no_canary_bad", rollout={"strategy": "all_at_once", "traffic_percent": 100, "target": "all", "monitors": ["success"]}),
    make_case(id="canary_safety_regression_bad", metrics={"success_rate": 0.97, "error_rate": 0.01, "safety_blocks": 3, "latency_p95_ms": 1800, "cost_ratio": 1.05}),
    make_case(id="rollback_plan_missing_bad", rollback={"old_version_available": True, "config_compatible": False, "prompt_resource_pinned": True, "migration_reversible": False, "plan": False}),
    make_case(id="dependency_unpinned_bad", dependencies={"tools": {}, "prompts": {"prompt.contract_review": "latest"}, "resources": {}}),
    make_case(id="audit_missing_bad", audit={"event_id": "evt_012", "event_type": "skill_enabled"}),
    make_case(id="emergency_suspend_missing_bad", emergency={"suspend_supported": True, "block_new_tasks": False, "running_task_control": False, "artifact_marking": False, "notification": True, "evidence_preserved": False}),
    make_case(id="uninstall_deletes_audit_bad", uninstall={"config_policy": "delete", "artifact_policy": "delete", "trace_audit_policy": "delete", "workflow_impact_checked": False}),
    make_case(id="auto_update_major_bad", update_policy="auto", expected_policy="manual_approval"),
]

metrics = OrderedDict()
failed_by_case = OrderedDict()
for name, fn in CHECKS.items():
    passes = [fn(case) for case in CASES]
    metrics[name] = round(sum(passes) / len(passes), 3)
    for case, ok in zip(CASES, passes):
        if not ok:
            failed_by_case.setdefault(case["id"], []).append(name)

thresholds = {name: 0.95 for name in CHECKS}
failed_gates = [name for name, value in metrics.items() if value < thresholds[name]]

smoke = OrderedDict([
    ("complete_lifecycle_passes", all(fn(CASES[0]) for fn in CHECKS.values())),
    ("caught_new_permission_no_reapproval", not permission_reapproval(CASES[4])),
    ("caught_full_rollout_no_canary", not rollout_guard(CASES[8])),
    ("caught_delete_audit_on_uninstall", not uninstall_retention(CASES[14])),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_cases=", list(failed_by_case.keys()))
print("failed_gates=", failed_gates)
print("skill_lifecycle_gate_pass=", not failed_gates)
```

运行后可以看到：

```text
smoke= {'complete_lifecycle_passes': True, 'caught_new_permission_no_reapproval': True, 'caught_full_rollout_no_canary': True, 'caught_delete_audit_on_uninstall': True}
metrics= {'review_before_publish': 0.938, 'install_approval': 0.938, 'enable_scope_control': 0.938, 'permission_reapproval': 0.938, 'configuration_versioning': 0.938, 'running_task_policy': 0.938, 'compatibility_check': 0.938, 'rollout_guard': 0.875, 'rollback_readiness': 0.938, 'dependency_versioning': 0.938, 'audit_log_completeness': 0.938, 'emergency_suspend_readiness': 0.938, 'uninstall_retention': 0.938, 'eval_monitoring': 1.0, 'update_policy_accuracy': 0.938}
failed_cases= ['published_without_review_bad', 'install_without_permission_review_bad', 'enabled_for_all_agents_bad', 'new_permission_no_reapproval_bad', 'config_change_unversioned_bad', 'disable_no_running_policy_bad', 'breaking_output_no_compat_bad', 'full_rollout_no_canary_bad', 'canary_safety_regression_bad', 'rollback_plan_missing_bad', 'dependency_unpinned_bad', 'audit_missing_bad', 'emergency_suspend_missing_bad', 'uninstall_deletes_audit_bad', 'auto_update_major_bad']
failed_gates= ['review_before_publish', 'install_approval', 'enable_scope_control', 'permission_reapproval', 'configuration_versioning', 'running_task_policy', 'compatibility_check', 'rollout_guard', 'rollback_readiness', 'dependency_versioning', 'audit_log_completeness', 'emergency_suspend_readiness', 'uninstall_retention', 'update_policy_accuracy']
skill_lifecycle_gate_pass= False
```

这段 demo 的价值在于：它让“生命周期管理”不再是管理员后台功能列表，而是可回归的发布治理系统。面试中可以强调，Skill 进入平台以后，就要像软件产品一样管理 review、install、enable scope、permission delta、canary、rollback、emergency suspend 和 audit retention。

## 36.20 面试高频题

### 题 1：Skill 的发布、安装、启用、使用有什么区别？

参考回答：

发布是开发者把 Skill 提交到平台目录；安装是租户或用户把 Skill 加入自己的能力集合；启用是让 Skill 在某个范围内可被调用；使用是一次具体任务执行。安装不等于启用，启用也不代表每次调用都跳过权限检查。

### 题 2：Skill 安装时应该做哪些检查？

参考回答：

应该检查 Manifest、权限申请、数据访问范围、依赖工具和资源、版本兼容性、安全策略、配置项、开发者可信度和 eval 结果。高风险权限需要管理员审批。

### 题 3：Skill 升级时如何判断是否需要重新审批？

参考回答：

如果新增权限、扩大数据访问范围、改变外部共享策略、改变输出契约、改变高风险 workflow 或升级 major 版本，都应该重新审批。普通 patch 且不改变权限和接口，可以自动更新。

### 题 4：禁用 Skill 时正在运行的任务怎么办？

参考回答：

取决于禁用原因。普通维护可以允许已有任务完成但禁止新任务；安全风险应立即取消或暂停运行中任务，标记受影响 Artifact，并通知管理员。所有操作要写入审计日志。

### 题 5：为什么 Prompt 更新也要纳入版本管理？

参考回答：

Prompt 会直接影响模型行为、输出格式、安全边界和质量。即使代码没变，Prompt 变化也可能导致回归。因此 Prompt 应该版本化，并通过 eval 和灰度验证。

## 36.21 小练习

1. 设计一个 Skill 安装记录，包含 tenant、version、permissions、installed_by 和 status。
2. 为“会议总结 Skill”设计启用范围，要求只对产品团队开放。
3. 写一个 Skill 升级审批规则：哪些变更必须人工审批？
4. 设计一个紧急下架流程，处理 Skill 泄露敏感信息的问题。
5. 思考：Skill 卸载后，历史 Artifact 和 Audit 应该如何保留？
6. 运行本章 demo，把 `new_permission_no_reapproval_bad` 改成重新审批通过，观察 `permission_reapproval` 是否恢复。
7. 给 demo 增加一个 `security_patch_auto_ok` 样本，要求不改变权限、不改变输入输出、eval 全通过，并说明为什么它可以自动更新。

## 36.22 本章小结

本章我们讲了 Skill 的安装、启用、禁用和版本更新。

Skill 生命周期管理的关键是把发布、安装、启用和使用区分开。安装需要权限审批和配置，启用需要范围控制，禁用和卸载需要处理运行中任务、历史 Artifact 和审计数据。版本更新要关注 Manifest、Prompt、Workflow、Tool 依赖、权限、配置 schema 和 Eval 变化，高风险变更需要灰度、审批和回滚方案。

你可以把本章重点记成一句话：

> Skill 一旦产品化，就必须像软件服务一样管理生命周期，而不是像脚本一样随便替换。

下一章我们会继续讲 Skill Marketplace 与企业内部门户，也就是 Skill 如何被发现、选择、安装、审核和运营。
