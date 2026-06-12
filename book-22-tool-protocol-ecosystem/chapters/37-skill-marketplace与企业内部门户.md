# 第 37 章 Skill Marketplace 与企业内部门户

上一章我们讲了 Skill 的安装、启用、禁用和版本更新。那一章关注单个 Skill 的生命周期。

本章把视角再放大一点：当 Skill 多起来之后，用户、团队、企业和开发者如何发现、选择、审核、安装、评价和运营这些 Skill？这就是 Skill Marketplace 或企业内部门户要解决的问题。

Marketplace 这个词容易让人想到公开应用商店，但在企业里，更常见的是内部 Skill 门户：不同团队把自己开发的能力发布到内部平台，其他团队按权限申请使用。它既要像应用商店一样好找、好理解、好安装，又要像企业治理系统一样可控、可审计、可下架。

本章的核心结论是：

> Skill Marketplace 不是简单列表页，而是能力发现、权限审核、质量治理、安全运营和开发者生态的统一入口。

## 37.0 本讲资料边界与第二轮精修口径

第二轮精修时，本讲对齐的是公开资料中稳定的产品和工程抽象：OpenAI 关于 GPT / Skill 能力分发的公开说明、企业应用商店常见的发布审核和管理员管理文档、Microsoft Teams / Copilot Studio 等企业平台对应用发布、组织内分发、管理员治理和安全审查的公开口径。它们共同强调几件事：能力要可发现，发布前要审核，安装和启用要能被管理员控制，权限和数据访问要透明，质量和安全状态要持续运营。

本章不会把某一家平台的字段名、排名算法、审核队列、商业分成、开发者后台或管理员控制台写成通用标准。这里抽象的是 Skill Marketplace / 企业内部门户的稳定工程问题：当 Skill 从几个 demo 增长为一个企业能力生态时，平台如何证明这些能力可搜索、可理解、可审批、可安装、可评分、可监控、可下架和可追责。

## 37.1 为什么需要 Skill Marketplace

当只有几个 Skill 时，可以靠文档和口头传播。

当 Skill 增长到几十、几百甚至上千个时，就会出现问题：

1. 用户不知道有哪些 Skill。
2. 开发者重复造轮子。
3. 不同 Skill 功能重叠。
4. 权限和安全审核分散。
5. 质量好坏无法判断。
6. 版本和维护状态不透明。
7. 下架和紧急通知困难。
8. 企业无法统计哪些能力真正有价值。

Marketplace 的作用是把零散能力变成可运营的生态。

## 37.2 开放 Marketplace 与企业内部门户的区别

开放 Marketplace 面向更广泛开发者和用户，企业内部门户面向公司内部团队。

| 维度 | 开放 Marketplace | 企业内部门户 |
| --- | --- | --- |
| 用户 | 外部用户、开发者 | 内部员工、团队、租户 |
| 开发者 | 第三方为主 | 内部团队为主 |
| 审核重点 | 安全、隐私、滥用、合规 | 权限、数据隔离、内控、成本 |
| 分发方式 | 公开搜索和安装 | 按组织、角色、项目授权 |
| 计费 | 可能有商业分成 | 多为内部成本核算 |
| 风险 | 第三方不可信、供应链风险 | 数据泄露、越权、影子 IT |
| 治理 | 平台规则和开发者协议 | 企业策略、审批流、审计 |

开放 Marketplace 更像生态平台，企业内部门户更像能力治理平台。

## 37.3 Marketplace 的核心模块

一个完整 Skill Marketplace 至少包含这些模块：

1. Skill Catalog：能力目录。
2. Search & Discovery：搜索和发现。
3. Detail Page：详情页。
4. Review & Approval：审核和审批。
5. Installation Manager：安装管理。
6. Permission Center：权限中心。
7. Quality Dashboard：质量看板。
8. Security Center：安全中心。
9. Developer Console：开发者控制台。
10. Operation Console：运营管理后台。

这些模块共同支撑 Skill 从发布到使用的完整链路。

## 37.4 Skill Catalog：能力目录

Skill Catalog 是 Marketplace 的基础。

Catalog 存储的信息来自 Skill Manifest，但会增加运营字段。

常见字段包括：

1. skill_id。
2. name。
3. version。
4. description。
5. category。
6. tags。
7. owner。
8. status。
9. permissions。
10. risk_level。
11. install_count。
12. rating。
13. last_updated。
14. supported_agents。
15. tenant_visibility。

Catalog 不是静态文档，它应该随着版本、状态、权限、评分和使用数据持续更新。

## 37.5 搜索与发现

用户发现 Skill 的方式通常有三类。

### 37.5.1 关键词搜索

用户搜索：

```text
会议总结
合同审查
代码评审
周报生成
```

关键词搜索需要索引：

1. 名称。
2. 描述。
3. 标签。
4. 示例任务。
5. 能力列表。
6. 文档内容。

### 37.5.2 分类浏览

分类可以包括：

1. 办公效率。
2. 研发工程。
3. 数据分析。
4. 法务合规。
5. 客服运营。
6. 安全审计。
7. 财务采购。

分类要稳定，不要过度细分。

### 37.5.3 任务意图推荐

当用户输入任务时，平台可以推荐 Skill。

例如用户说：

```text
帮我把这次会议整理成纪要，并列出行动项。
```

系统可以推荐会议总结 Skill。

推荐可以结合：

1. 任务语义。
2. Skill examples。
3. 用户角色。
4. 团队已安装 Skill。
5. 权限可用性。
6. 历史成功率。

注意，推荐不能只看语义相似度，还要看权限和可用性。

## 37.6 Skill 详情页应该展示什么

一个好的 Skill 详情页应该让用户和管理员回答这些问题：

1. 它能做什么？
2. 适合哪些场景？
3. 不适合哪些场景？
4. 需要哪些权限？
5. 会访问哪些数据？
6. 输出是什么样？
7. 谁维护？
8. 最近是否更新？
9. 质量和安全评分如何？
10. 是否有已知限制？

详情页内容可以包括：

1. 能力说明。
2. 示例任务。
3. 输入输出样例。
4. 权限列表和 reason。
5. 风险级别。
6. 使用指南。
7. 版本历史。
8. 质量指标。
9. 用户评价。
10. 安全审核状态。

如果详情页只写一句“提升办公效率”，用户和管理员都无法做决策。

## 37.7 审核机制

Skill 上架前通常需要审核。

审核分几类。

### 37.7.1 Manifest 审核

检查字段是否完整：

1. 能力描述是否清晰。
2. 输入输出是否稳定。
3. 权限 reason 是否合理。
4. 安全策略是否声明。
5. Eval 是否存在。
6. 版本是否规范。

### 37.7.2 权限审核

检查权限是否过大，是否符合最小权限原则。

例如，一个会议总结 Skill 如果申请 database.write，就很可疑。

### 37.7.3 安全审核

检查：

1. 是否可能泄露敏感信息。
2. 是否会调用外部服务。
3. 是否允许高风险动作。
4. 是否有 prompt injection 防护。
5. 是否有输出脱敏。
6. 是否有人工确认点。

### 37.7.4 质量审核

检查 eval 结果、示例任务表现、失败率、输出格式、引用准确性。

### 37.7.5 维护性审核

检查 owner 是否明确、文档是否完整、是否有联系方式、是否有回滚方案。

审核不是一次性动作。版本升级、权限变化和安全事件都可能触发重新审核。

## 37.8 安装和审批流程

企业内部门户里，安装流程通常不是用户自己随便点。

一个典型流程：

1. 用户申请安装 Skill。
2. 平台展示权限和风险。
3. 团队管理员审批。
4. 安全或 IT 审批高风险权限。
5. 配置认证和参数。
6. 设置启用范围。
7. 记录审计事件。
8. 通知申请人。

审批流可以按风险分级：

| 风险等级 | 审批方式 |
| --- | --- |
| 低风险 | 用户自助安装 |
| 中风险 | 团队管理员审批 |
| 高风险 | 安全/IT/数据 owner 审批 |
| 极高风险 | 默认禁止或人工专项审批 |

## 37.9 评分与评价

Marketplace 需要帮助用户判断 Skill 好不好。

评分来源可以包括：

1. 用户评价。
2. 任务成功率。
3. Eval 分数。
4. 安全审核结果。
5. 响应速度。
6. 维护活跃度。
7. 安装量和留存率。
8. 问题关闭速度。

但评分不能只靠用户打星。因为有些 Skill 用户满意但安全风险高，有些 Skill 低频但对关键业务很重要。

更合理的是多维评分：

1. Quality Score。
2. Safety Score。
3. Reliability Score。
4. Maintenance Score。
5. Adoption Score。

## 37.10 运营指标

Skill Marketplace 是要运营的。

关键指标包括：

1. Skill 总数。
2. 上架通过率。
3. 安装量。
4. 启用量。
5. 活跃 Skill 数。
6. 每个 Skill 的任务成功率。
7. 平均权限风险等级。
8. 安全事件数。
9. 版本更新频率。
10. 用户留存。
11. 重复能力比例。
12. 无人维护 Skill 数。

这些指标能帮助平台判断生态是否健康。

## 37.11 重复能力治理

Marketplace 发展一段时间后，一定会出现重复 Skill。

例如：

1. 三个会议总结 Skill。
2. 五个周报生成 Skill。
3. 两个合同审查 Skill。

重复不一定坏，因为不同团队可能有不同需求。但过度重复会导致用户困惑和维护浪费。

治理方式：

1. 推荐官方或高质量 Skill。
2. 标记适用场景差异。
3. 合并重复 Skill。
4. 下架无人维护 Skill。
5. 鼓励复用底层 Workflow 或 Tool。
6. 建立能力命名规范。

## 37.12 企业内部权限视图

企业门户需要给管理员一个权限视图。

管理员应该能看到：

1. 哪些 Skill 安装在本租户。
2. 每个 Skill 申请了哪些权限。
3. 哪些用户或团队启用了它。
4. 最近调用了哪些高风险动作。
5. 哪些 Skill 有外部数据传输。
6. 哪些 Skill 使用了敏感数据。
7. 哪些 Skill 版本过旧。
8. 哪些 Skill 没有 owner。

没有权限视图，企业很容易出现“影子 Skill”：大家都在用，但没人知道风险。

## 37.13 开发者控制台

对 Skill 开发者来说，Marketplace 也要提供控制台。

开发者需要看到：

1. 提交状态。
2. 审核结果。
3. 安装量。
4. 调用量。
5. 失败率。
6. 用户反馈。
7. Eval 结果。
8. 安全告警。
9. 版本分布。
10. 崩溃和错误日志。

开发者控制台能促进生态质量提升。

## 37.14 安全中心

Marketplace 必须有安全中心。

安全中心负责：

1. 高风险权限巡检。
2. 异常调用检测。
3. 数据泄露告警。
4. 供应链风险监控。
5. 紧急下架。
6. 漏洞通告。
7. 安全审核记录。
8. 受影响用户通知。

Skill 生态越开放，安全中心越重要。

## 37.15 企业内部门户的推荐策略

企业门户推荐 Skill 时，不应该只推荐热门。

推荐信号可以包括：

1. 用户角色。
2. 所属团队。
3. 当前任务。
4. 已安装工具。
5. 权限可用性。
6. 企业推荐。
7. 安全等级。
8. 成功率。
9. 最近维护状态。

例如法务团队更应该看到合同审查、条款比较、合规检查；研发团队更应该看到代码评审、测试生成、日志分析。

推荐系统也要避免把高风险 Skill 推荐给无权限用户。

## 37.16 一个完整例子：企业会议总结 Skill 上架

假设团队开发了一个会议总结 Skill。

上架流程：

1. 开发者提交 Manifest、文档、Eval 结果和安全说明。
2. 平台检查 Manifest 完整性。
3. 安全审核检查权限：读取会议转写、读取参与人列表、创建任务、发送消息。
4. 因为发送消息是中风险动作，要求默认关闭自动发送。
5. 质量审核跑会议纪要 golden set。
6. 审核通过后进入企业门户。
7. 产品团队申请安装。
8. 团队管理员批准读取会议数据。
9. Skill 只对产品团队启用。
10. 使用 2 周后查看成功率、用户反馈和敏感信息拦截记录。
11. 根据反馈升级模板并灰度发布。

这个流程体现了 Marketplace 的价值：能力不是直接丢给用户，而是经过发现、审核、授权、启用和运营。

## 37.17 常见误区

### 37.17.1 Marketplace 只是列表页

列表页只能展示，不能治理。真正的 Marketplace 要支持搜索、审核、安装、权限、评分、运营和安全。

### 37.17.2 只看安装量

安装量高不代表质量高。还要看任务成功率、安全事件、留存率和维护状态。

### 37.17.3 高风险 Skill 自助安装

高风险权限必须有审批。否则很容易造成数据泄露和越权。

### 37.17.4 没有 owner

无人维护的 Skill 是长期风险。Marketplace 应该标记 owner 和维护状态。

### 37.17.5 不治理重复能力

重复能力太多会让用户困惑，也会分散维护资源。

## 37.18 Skill Marketplace 审计指标与最小 demo

把 Marketplace 当成系统设计题时，不要只画“列表页 + 搜索框”。更可面试、更可落地的做法，是把一个 Skill 上架条目看成可审计对象：

```math
s_i=(c_i,q_i,d_i,r_i,p_i,a_i,v_i,m_i,g_i,u_i,h_i,o_i,z_i)
```

其中 $c_i$ 是 catalog metadata，$q_i$ 是搜索和发现信息，$d_i$ 是详情页，$r_i$ 是 review workflow，$p_i$ 是权限透明度，$a_i$ 是安装审批，$v_i$ 是评分和质量信号，$m_i$ 是运营指标，$g_i$ 是重复能力治理，$u_i$ 是管理员权限视图，$h_i$ 是安全中心，$o_i$ 是 owner / maintenance 状态，$z_i$ 是审计事件。

对第 $j$ 个检查项，可以用统一通过率描述：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[g_j(s_i)=1]
```

Marketplace 不应该只用安装量排序。一个更合理的多维列表分可以写成：

```math
S_{\mathrm{listing},i}=w_qQ_i+w_hH_i+w_rR_i+w_mM_i+w_aA_i
```

其中 $Q_i$ 是质量，$H_i$ 是安全健康度，$R_i$ 是可靠性，$M_i$ 是维护状态，$A_i$ 是采用度。权重 $w_q,w_h,w_r,w_m,w_a$ 应由企业治理目标决定。比如法务和安全场景里，$w_h$ 和 $w_r$ 通常比 $w_a$ 更重要，不能因为某个 Skill 热门就默认推荐。

最终可以把 Marketplace 门禁写成：

```math
G_{\mathrm{skill\_marketplace}}=
\mathbb{1}\left[
\min_j C_j\ge \tau
\right]
```

这里的 $\tau$ 是上线阈值。这个公式的意思不是“所有 Skill 都必须完美”，而是平台要能发现薄弱环节：是目录字段缺失，还是权限 reason 不透明，还是高风险 Skill 被错误地自助安装，还是安全中心没有紧急下架能力。

下面是一个 0 依赖 toy demo。它把 16 个 Marketplace listing 送进审计器：一个完整样本和 15 个典型坏样本。这个 demo 适合面试时说明“企业 Skill 门户怎么评估是否可上线”。

```python
from copy import deepcopy

REQUIRED_CATALOG = {
    "skill_id", "name", "version", "description", "category", "tags", "owner",
    "status", "permissions", "risk_level", "last_updated", "visibility",
}
REQUIRED_SEARCH = {"name", "description", "tags", "examples", "capabilities", "owner"}
REQUIRED_SEARCH_SIGNALS = {"quality", "safety", "permission_available", "maintenance"}
REQUIRED_DETAIL = {
    "what_it_does", "not_for", "examples", "inputs_outputs", "permissions",
    "owner", "version_history", "quality_metrics", "security_status", "known_limits",
}
REQUIRED_REVIEW = {"manifest", "permission", "security", "quality", "maintenance"}
REQUIRED_RATING = {"user_rating", "task_success_rate", "eval_score", "safety_score", "maintenance_score"}
REQUIRED_OPS = {
    "skill_total", "publish_pass_rate", "install_count", "enable_count",
    "active_skill_count", "task_success_rate", "avg_permission_risk",
    "security_incidents", "update_frequency", "retention", "duplicate_ratio",
    "orphaned_skill_count",
}
REQUIRED_ADMIN = {
    "installed_skills", "permissions", "enabled_users_or_teams", "high_risk_calls",
    "external_transfers", "sensitive_data", "stale_versions", "owner_missing",
}
REQUIRED_DEV = {
    "submission_status", "review_result", "install_count", "call_count", "failure_rate",
    "feedback", "eval_result", "security_alerts", "version_distribution", "error_logs",
}
REQUIRED_SECURITY = {
    "permission_scan", "anomaly_detection", "data_leakage_alert", "supply_chain_risk",
    "emergency_takedown", "vulnerability_notice", "affected_user_notification",
}
REQUIRED_AUDIT = {"submit", "review", "install", "enable", "block", "uninstall"}
RISK_APPROVAL = {
    "low": "self_service",
    "medium": "team_admin",
    "high": "security_it_data_owner",
    "critical": "blocked_or_special_review",
}
CHECK_ORDER = [
    "catalog_metadata",
    "search_discovery",
    "detail_page_completeness",
    "review_workflow",
    "permission_transparency",
    "install_approval_flow",
    "rating_quality_balance",
    "operations_metrics",
    "duplicate_capability_governance",
    "admin_permission_view",
    "developer_console_readiness",
    "security_center_readiness",
    "recommendation_policy_safety",
    "lifecycle_visibility",
    "owner_maintenance",
    "portal_audit_trace",
]

BASE_LISTING = {
    "id": "meeting_summary_marketplace_ok",
    "catalog_fields": REQUIRED_CATALOG,
    "search_fields": REQUIRED_SEARCH,
    "search_signals": REQUIRED_SEARCH_SIGNALS,
    "detail_fields": REQUIRED_DETAIL,
    "detail_specific": True,
    "review_steps": REQUIRED_REVIEW,
    "permissions": [
        {"name": "calendar.read", "reason": "读取会议标题和时间", "data_class": "internal"},
        {"name": "transcript.read", "reason": "生成会议纪要", "data_class": "confidential"},
    ],
    "risk_level": "medium",
    "install_approval": "team_admin",
    "rating_signals": REQUIRED_RATING,
    "ops_metrics": REQUIRED_OPS,
    "duplicate_policy": {"canonical_skill": True, "scenario_labels": True, "merge_or_deprecate": True},
    "admin_view_fields": REQUIRED_ADMIN,
    "developer_console_fields": REQUIRED_DEV,
    "security_center_fields": REQUIRED_SECURITY,
    "recommendation_policy": {
        "uses_role": True,
        "uses_task": True,
        "uses_permission_filter": True,
        "uses_safety_filter": True,
        "blocks_high_risk_without_auth": True,
    },
    "lifecycle_fields": {"status", "version", "version_history", "last_updated", "deprecation_policy"},
    "owner_active": True,
    "maintenance_sla": True,
    "audit_events": REQUIRED_AUDIT,
}


def listing(name, **updates):
    item = deepcopy(BASE_LISTING)
    item["id"] = name
    for key, value in updates.items():
        item[key] = value
    return item


def without(values, *removed):
    result = set(values)
    for value in removed:
        result.discard(value)
    return result


LISTINGS = [
    BASE_LISTING,
    listing("catalog_missing_owner_bad", catalog_fields=without(REQUIRED_CATALOG, "owner")),
    listing("search_index_missing_examples_bad", search_fields=without(REQUIRED_SEARCH, "examples")),
    listing("detail_page_vague_bad", detail_specific=False),
    listing("review_missing_security_bad", review_steps=without(REQUIRED_REVIEW, "security")),
    listing(
        "permission_reason_hidden_bad",
        permissions=[{"name": "calendar.read", "data_class": "internal"}],
        install_approval="self_service",
    ),
    listing("high_risk_self_install_bad", risk_level="high", install_approval="self_service"),
    listing("rating_only_stars_bad", rating_signals={"user_rating"}),
    listing("ops_metrics_missing_bad", ops_metrics=without(REQUIRED_OPS, "security_incidents")),
    listing("duplicates_ungoverned_bad", duplicate_policy={"canonical_skill": False}),
    listing("admin_view_missing_bad", admin_view_fields=without(REQUIRED_ADMIN, "external_transfers")),
    listing("developer_console_missing_bad", developer_console_fields=without(REQUIRED_DEV, "eval_result")),
    listing("security_center_missing_bad", security_center_fields=without(REQUIRED_SECURITY, "emergency_takedown")),
    listing(
        "unsafe_recommendation_bad",
        recommendation_policy={
            "uses_role": True,
            "uses_task": True,
            "uses_permission_filter": False,
            "uses_safety_filter": False,
            "blocks_high_risk_without_auth": False,
        },
    ),
    listing("stale_owner_bad", owner_active=False, maintenance_sla=False),
    listing("audit_missing_bad", audit_events=without(REQUIRED_AUDIT, "block")),
]


def has_all(actual, required):
    return set(actual) >= set(required)


def permission_transparency_ok(item):
    permissions = item.get("permissions", [])
    reasoned = all(p.get("reason") and p.get("data_class") for p in permissions)
    high_risk_explained = item.get("risk_level") != "high" or item.get("install_approval") == RISK_APPROVAL["high"]
    return bool(permissions) and reasoned and high_risk_explained


def install_approval_ok(item):
    return item.get("install_approval") == RISK_APPROVAL[item.get("risk_level", "low")]


def duplicate_policy_ok(item):
    policy = item.get("duplicate_policy", {})
    return all(policy.get(key) for key in ("canonical_skill", "scenario_labels", "merge_or_deprecate"))


def recommendation_policy_ok(item):
    policy = item.get("recommendation_policy", {})
    return all(policy.get(key) for key in (
        "uses_role", "uses_task", "uses_permission_filter", "uses_safety_filter", "blocks_high_risk_without_auth",
    ))


def audit_listing(item):
    return {
        "catalog_metadata": has_all(item["catalog_fields"], REQUIRED_CATALOG),
        "search_discovery": has_all(item["search_fields"], REQUIRED_SEARCH)
        and has_all(item["search_signals"], REQUIRED_SEARCH_SIGNALS),
        "detail_page_completeness": has_all(item["detail_fields"], REQUIRED_DETAIL) and item["detail_specific"],
        "review_workflow": has_all(item["review_steps"], REQUIRED_REVIEW),
        "permission_transparency": permission_transparency_ok(item),
        "install_approval_flow": install_approval_ok(item),
        "rating_quality_balance": has_all(item["rating_signals"], REQUIRED_RATING),
        "operations_metrics": has_all(item["ops_metrics"], REQUIRED_OPS),
        "duplicate_capability_governance": duplicate_policy_ok(item),
        "admin_permission_view": has_all(item["admin_view_fields"], REQUIRED_ADMIN),
        "developer_console_readiness": has_all(item["developer_console_fields"], REQUIRED_DEV),
        "security_center_readiness": has_all(item["security_center_fields"], REQUIRED_SECURITY),
        "recommendation_policy_safety": recommendation_policy_ok(item),
        "lifecycle_visibility": has_all(item["lifecycle_fields"], {"status", "version", "version_history", "last_updated"}),
        "owner_maintenance": item["owner_active"] and item["maintenance_sla"],
        "portal_audit_trace": has_all(item["audit_events"], REQUIRED_AUDIT),
    }


results = {item["id"]: audit_listing(item) for item in LISTINGS}
metrics = {
    check: round(sum(result[check] for result in results.values()) / len(results), 3)
    for check in CHECK_ORDER
}
failed_listings = [name for name, result in results.items() if not all(result.values())]
failed_gates = [name for name, value in metrics.items() if value < 1.0]
smoke = {
    "complete_marketplace_passes": all(results["meeting_summary_marketplace_ok"].values()),
    "caught_high_risk_self_install": not results["high_risk_self_install_bad"]["install_approval_flow"],
    "caught_unsafe_recommendation": not results["unsafe_recommendation_bad"]["recommendation_policy_safety"],
    "caught_missing_security_center": not results["security_center_missing_bad"]["security_center_readiness"],
}

print("smoke=", smoke)
print("metrics=", metrics)
print("failed_listings=", failed_listings)
print("failed_gates=", failed_gates)
print("skill_marketplace_gate_pass=", not failed_gates and not failed_listings)
```

运行结果应该类似：

```text
smoke= {'complete_marketplace_passes': True, 'caught_high_risk_self_install': True, 'caught_unsafe_recommendation': True, 'caught_missing_security_center': True}
metrics= {'catalog_metadata': 0.938, 'search_discovery': 0.938, 'detail_page_completeness': 0.938, 'review_workflow': 0.938, 'permission_transparency': 0.875, 'install_approval_flow': 0.875, 'rating_quality_balance': 0.938, 'operations_metrics': 0.938, 'duplicate_capability_governance': 0.938, 'admin_permission_view': 0.938, 'developer_console_readiness': 0.938, 'security_center_readiness': 0.938, 'recommendation_policy_safety': 0.938, 'lifecycle_visibility': 1.0, 'owner_maintenance': 0.938, 'portal_audit_trace': 0.938}
failed_listings= ['catalog_missing_owner_bad', 'search_index_missing_examples_bad', 'detail_page_vague_bad', 'review_missing_security_bad', 'permission_reason_hidden_bad', 'high_risk_self_install_bad', 'rating_only_stars_bad', 'ops_metrics_missing_bad', 'duplicates_ungoverned_bad', 'admin_view_missing_bad', 'developer_console_missing_bad', 'security_center_missing_bad', 'unsafe_recommendation_bad', 'stale_owner_bad', 'audit_missing_bad']
failed_gates= ['catalog_metadata', 'search_discovery', 'detail_page_completeness', 'review_workflow', 'permission_transparency', 'install_approval_flow', 'rating_quality_balance', 'operations_metrics', 'duplicate_capability_governance', 'admin_permission_view', 'developer_console_readiness', 'security_center_readiness', 'recommendation_policy_safety', 'owner_maintenance', 'portal_audit_trace']
skill_marketplace_gate_pass= False
```

这个 demo 的重点不是追求复杂算法，而是训练你把 Marketplace 设计题拆成可验证的治理维度。面试时可以这样讲：搜索和详情页解决“用户能不能找到并理解”，审核和审批解决“能不能安全进入租户”，评分和运营解决“是否真的有价值”，管理员视图、安全中心和审计解决“出事后能不能看见、阻断和追责”。

## 37.19 面试高频题

### 题 1：为什么需要 Skill Marketplace？

参考回答：

当 Skill 数量变多后，需要一个统一入口支持能力发现、搜索、安装、权限审批、质量评分、安全审核、版本治理和运营分析。Marketplace 不是简单列表页，而是 Skill 生态的治理平台。

### 题 2：企业内部门户和开放 Marketplace 有什么区别？

参考回答：

开放 Marketplace 面向外部用户和第三方开发者，重点是生态、隐私、滥用和商业分发。企业内部门户面向内部团队，重点是权限、数据隔离、审批流、审计、成本和内部复用。企业门户通常更强调治理和合规。

### 题 3：Skill 详情页应该展示什么？

参考回答：

应该展示能力说明、适用和不适用场景、示例任务、输入输出样例、权限列表和 reason、风险等级、维护团队、版本历史、质量指标、安全审核状态和用户评价。

### 题 4：Skill 上架审核应该包括哪些方面？

参考回答：

包括 Manifest 完整性审核、权限审核、安全审核、质量审核和维护性审核。版本升级、权限变化和安全事件应触发重新审核。

### 题 5：如何治理重复 Skill？

参考回答：

可以通过推荐官方 Skill、标记适用场景差异、合并重复能力、下架无人维护 Skill、鼓励复用底层 Workflow 或 Tool、建立命名规范来治理。重复不一定坏，但需要清晰区分。

## 37.20 小练习

1. 设计一个 Skill 详情页，列出至少 10 个字段。
2. 为企业内部门户设计 Skill 搜索排序规则。
3. 设计一个高风险 Skill 的安装审批流程。
4. 列出 Skill Marketplace 的 8 个运营指标。
5. 思考：如果两个团队都发布了“会议总结 Skill”，平台应该如何处理？
6. 修改 37.18 的 demo，让 `high_risk_self_install_bad` 同时触发权限透明度、安装审批和推荐策略三个失败门禁，并解释为什么高风险 Skill 不能靠用户自助安装。
7. 给 demo 新增一个 `orphaned_popular_skill_bad` 样本：安装量很高但 owner 缺失、版本陈旧、安全中心没有漏洞通知。观察它会拉低哪些指标。

## 37.21 本章小结

本章我们讲了 Skill Marketplace 与企业内部门户。

Marketplace 不是简单列表页，而是 Skill 生态的能力发现、安装审批、权限治理、质量评估、安全运营和开发者反馈平台。开放 Marketplace 更偏生态和外部分发，企业内部门户更偏权限、审计、数据隔离和内部复用。

你可以把本章重点记成一句话：

> Skill 只有进入可发现、可审核、可评分、可运营的门户，才能从单点能力变成可持续发展的工具生态。

下一章我们会继续讲工具和 Skill 的质量评估与安全审核，重点讨论如何判断一个 Skill 是否真的好用、稳定、安全、值得上架。
