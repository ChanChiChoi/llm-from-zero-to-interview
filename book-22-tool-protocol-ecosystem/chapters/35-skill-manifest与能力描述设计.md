# 第 35 章 Skill Manifest 与能力描述设计

前两章我们讲清楚了 Skill 的定义，以及 Plugin、Action、Tool、Skill、Workflow 的边界。

这一章进入 Skill 的工程设计核心：Skill Manifest。

如果说 Skill 是一个能力包，那么 Manifest 就是这个能力包的说明书、身份证、权限申请表、能力声明和治理入口。平台要安装 Skill，需要看 Manifest；Agent 要选择 Skill，需要看 Manifest；安全审核要判断风险，需要看 Manifest；版本升级和评估也离不开 Manifest。

本章重点讲：一个好的 Skill Manifest 应该包含哪些字段，能力描述应该怎么写，权限和配置如何声明，示例和评估如何设计，哪些 Manifest 写法会导致路由错误、安全风险和维护困难。

你可以先记住一句话：

> Skill Manifest 的目标不是“描述得好看”，而是让平台、Agent、管理员和用户都能可靠理解这个 Skill 能做什么、需要什么、会影响什么、如何评估。

## 35.0 本讲资料边界与第二轮精修口径

本章第二轮精修时，重点核对了 Agent Skills 开放规范中 `SKILL.md`、frontmatter、description、scripts / references / assets、progressive disclosure 和 eval 的口径，MCP Specification 中 tools / resources / prompts 的结构化对象边界，OpenAI Apps SDK / Actions 公开资料中 tool descriptor、component / resource 和 action 接入的工程抽象，以及 Microsoft Copilot Studio 等平台对 actions、plugins、connectors、workflows 和 agents 的常见产品命名。

需要先划清边界：

1. 本章讲的是 Skill Manifest 的稳定工程抽象，不把某一家平台的 manifest 字段、YAML frontmatter、tool descriptor、plugin metadata 或 connector 配置写成唯一标准。
2. Manifest 不是宣传文案，而是服务发现、路由、权限审核、安全治理、版本管理、eval 和安装治理共同使用的结构化入口。
3. 能力描述要服务于触发和不触发的边界，不能只写“强大助手”“万能办公”这类无法路由、无法评估的描述。
4. 权限、资源、工具、prompt、workflow、配置、安全策略和 eval 应该能被平台读取和强制执行，而不是只写在人类说明里。
5. 本章新增的公式和 Python demo 是教学用 manifest 审计器，不实现真实 marketplace、安装器、权限系统、tool runtime、prompt registry 或 eval 平台。

## 35.1 为什么 Manifest 很重要

没有 Manifest 的 Skill，就像一个没有说明书的黑盒。

平台不知道：

1. 它叫什么。
2. 它能做什么。
3. 它需要哪些工具。
4. 它需要哪些权限。
5. 它能访问哪些数据。
6. 它输出什么结果。
7. 它适合哪些任务。
8. 它不适合哪些任务。
9. 它如何配置。
10. 它如何评估。

如果这些都靠自然语言口头约定，系统会很难治理。

Manifest 的价值包括：

1. 服务发现：平台能检索和匹配 Skill。
2. 路由选择：Agent 能判断什么时候使用它。
3. 权限审核：管理员能知道它要访问什么。
4. 安全治理：平台能判断风险级别。
5. 版本管理：升级时能判断兼容性。
6. 质量评估：Eval 能绑定到具体能力。
7. 用户理解：用户知道安装后能完成什么。

## 35.2 Manifest 的基本结构

一个 Skill Manifest 可以分成这些部分：

1. identity：身份信息。
2. description：能力描述。
3. capabilities：能力列表。
4. inputs：输入要求。
5. outputs：输出契约。
6. tools：依赖工具。
7. resources：依赖资源。
8. prompts：提示模板。
9. workflow：流程定义。
10. permissions：权限声明。
11. configuration：配置项。
12. safety：安全策略。
13. eval：评估声明。
14. examples：示例任务。
15. versioning：版本和兼容性。

一个简化骨架如下：

```json
{
  "id": "skill.meeting_summary",
  "name": "Meeting Summary Skill",
  "version": "1.0.0",
  "description": "Generate structured meeting notes and action items from transcripts and documents.",
  "capabilities": [],
  "inputs": {},
  "outputs": {},
  "tools": [],
  "resources": [],
  "prompts": [],
  "workflow": {},
  "permissions": [],
  "configuration": {},
  "safety": {},
  "eval": {},
  "examples": []
}
```

不是所有 Skill 都需要每个字段都很复杂，但生产系统至少要覆盖身份、能力、输入输出、工具、权限、安全和版本。

## 35.3 身份信息设计

身份信息回答“这个 Skill 是谁”。

常见字段：

1. id：稳定唯一标识。
2. name：人类可读名称。
3. version：版本号。
4. owner：维护团队。
5. author：作者或发布方。
6. category：分类。
7. tags：标签。
8. homepage：文档入口。
9. status：状态。

示例：

```json
{
  "id": "skill.contract_review",
  "name": "Contract Review Skill",
  "version": "1.3.0",
  "owner": "legal-platform-team",
  "category": "legal",
  "tags": ["contract", "risk_review", "compliance"],
  "status": "stable"
}
```

id 应该稳定，不要随着名称变化而变化。name 可以给用户看，id 用于系统引用。

## 35.4 能力描述怎么写

能力描述是 Manifest 中最容易写坏的部分。

坏描述：

```text
这是一个强大的合同助手，可以帮助你解决各种合同问题。
```

这个描述太泛，无法路由，无法评估，也无法明确边界。

更好的描述：

```text
Review supplier contracts for payment terms, liability clauses, termination conditions, and compliance risks. Output a structured risk report with citations to contract sections.
```

它说明了：

1. 对象：supplier contracts。
2. 任务：review。
3. 关注点：payment terms、liability、termination、compliance。
4. 输出：structured risk report。
5. 证据：citations。

好的能力描述应该具体、可验证、可限制。

## 35.5 Capabilities：能力列表

一个 Skill 可以支持多个相关能力。

例如合同审查 Skill：

```json
{
  "capabilities": [
    {
      "name": "payment_terms_review",
      "description": "Identify unfavorable payment terms and compare them with company policy.",
      "domains": ["procurement", "legal"],
      "risk_level": "medium"
    },
    {
      "name": "liability_clause_review",
      "description": "Review liability and indemnity clauses for high-risk obligations.",
      "domains": ["legal"],
      "risk_level": "high"
    }
  ]
}
```

能力列表不宜太泛，也不宜太碎。

太泛：

```text
legal_help
```

太碎：

```text
find_word_payment
count_clause_lines
extract_party_name
```

更合适的是任务级能力。

## 35.6 Inputs：输入要求

输入要求回答“使用这个 Skill 需要什么”。

例如：

```json
{
  "inputs": {
    "required": [
      {
        "name": "contract_document",
        "type": "resource_ref",
        "accepted_uri_schemes": ["file://", "doc://", "artifact://"]
      }
    ],
    "optional": [
      {
        "name": "review_focus",
        "type": "array<string>",
        "default": ["payment_terms", "liability", "termination"]
      },
      {
        "name": "jurisdiction",
        "type": "string"
      }
    ]
  }
}
```

输入要求能减少无效调用。如果缺少必要输入，Agent 应该先追问或补齐，而不是盲目执行。

## 35.7 Outputs：输出契约

输出契约回答“这个 Skill 会返回什么”。

例如合同审查 Skill 可以声明：

```json
{
  "outputs": {
    "types": ["summary", "risk_report", "citation_list", "artifact"],
    "schema": {
      "type": "object",
      "properties": {
        "overall_risk": { "type": "string", "enum": ["low", "medium", "high"] },
        "findings": { "type": "array" },
        "citations": { "type": "array" },
        "limitations": { "type": "array" }
      }
    },
    "supports_citations": true,
    "supports_confidence": true
  }
}
```

输出契约非常重要，因为上游 Agent、Workflow 或 UI 要根据输出继续处理。如果输出不稳定，整个系统都会不稳定。

## 35.8 Tools：依赖工具声明

Skill 需要声明它依赖哪些 Tool。

例如：

```json
{
  "tools": [
    {
      "name": "read_document",
      "purpose": "Read contract text and section structure.",
      "required": true
    },
    {
      "name": "search_policy",
      "purpose": "Find internal legal policy for comparison.",
      "required": true
    },
    {
      "name": "create_report_artifact",
      "purpose": "Store the final review report.",
      "required": false
    }
  ]
}
```

注意，工具声明最好写 purpose。否则平台只知道 Skill 用了哪些工具，不知道为什么用。

## 35.9 Resources：依赖资源声明

Resources 是 Skill 的稳定知识和上下文来源。

例如：

```json
{
  "resources": [
    {
      "uri": "kb://legal/contract-review-policy",
      "purpose": "Internal policy for contract risk review.",
      "required": true
    },
    {
      "uri": "kb://legal/approved-clause-examples",
      "purpose": "Examples of acceptable clauses.",
      "required": false
    }
  ]
}
```

资源声明有助于审计和更新。如果政策文档升级，可以知道哪些 Skill 受影响。

## 35.10 Prompts：提示模板声明

Skill 里的 Prompt 应该版本化。

例如：

```json
{
  "prompts": [
    {
      "id": "prompt.contract_risk_review.v2",
      "purpose": "Guide the model to identify contract risks and cite evidence.",
      "version": "2.1.0"
    }
  ]
}
```

Prompt 不是随便写在代码里的字符串。它会影响输出质量、安全性和兼容性，应该纳入版本管理和评估。

## 35.11 Workflow：流程声明

复杂 Skill 应该声明 workflow。

例如：

```json
{
  "workflow": {
    "steps": [
      { "id": "read_contract", "type": "tool", "tool": "read_document" },
      { "id": "retrieve_policy", "type": "tool", "tool": "search_policy" },
      { "id": "draft_findings", "type": "prompt", "prompt": "prompt.contract_risk_review.v2" },
      { "id": "validate_citations", "type": "tool", "tool": "check_citations" },
      { "id": "create_report", "type": "tool", "tool": "create_report_artifact" }
    ],
    "approval_points": ["create_report"]
  }
}
```

Workflow 声明让平台知道执行路径，也便于插入审核、监控和失败恢复。

## 35.12 Permissions：权限声明

权限声明是 Manifest 的安全核心。

不要只写：

```json
{
  "permissions": ["all"]
}
```

这等于没有权限设计。

更好的写法：

```json
{
  "permissions": [
    {
      "scope": "documents.read",
      "reason": "Read contract documents provided by the user.",
      "required": true
    },
    {
      "scope": "legal_policy.read",
      "reason": "Compare contract clauses with internal policy.",
      "required": true
    },
    {
      "scope": "artifacts.write",
      "reason": "Create final review report.",
      "required": false
    }
  ]
}
```

权限声明应该包含 reason，因为管理员审核时需要知道为什么需要这些权限。

## 35.13 Configuration：配置项

Skill 配置让同一个 Skill 适配不同租户、团队或用户。

例如：

```json
{
  "configuration": {
    "fields": [
      {
        "name": "default_jurisdiction",
        "type": "string",
        "default": "CN"
      },
      {
        "name": "require_human_approval_for_high_risk",
        "type": "boolean",
        "default": true
      },
      {
        "name": "report_language",
        "type": "string",
        "enum": ["zh-CN", "en-US"],
        "default": "zh-CN"
      }
    ]
  }
}
```

配置项应该有类型、默认值、是否必填、可选范围和说明。

## 35.14 Safety：安全策略声明

Skill 需要声明安全边界。

例如：

```json
{
  "safety": {
    "data_classification_allowed": ["public", "internal", "confidential"],
    "external_sharing": false,
    "requires_human_approval_for": ["high_risk_report", "external_send"],
    "forbidden_actions": ["modify_contract", "send_to_external_party"],
    "redaction": {
      "enabled": true,
      "fields": ["personal_contact", "bank_account"]
    }
  }
}
```

安全策略不是装饰字段，而应该被平台强制执行。

## 35.15 Eval：评估声明

Skill Manifest 应该声明如何评估。

例如合同审查 Skill：

```json
{
  "eval": {
    "metrics": [
      "risk_identification_recall",
      "citation_accuracy",
      "false_positive_rate",
      "format_compliance",
      "sensitive_info_leakage_rate"
    ],
    "golden_sets": ["eval://legal/contract-review-v1"],
    "minimum_quality_gate": {
      "citation_accuracy": 0.95,
      "sensitive_info_leakage_rate": 0.0
    }
  }
}
```

Eval 声明让 Skill 升级有门槛，而不是凭感觉发布。

## 35.16 Examples：示例任务

示例任务对路由和理解很有帮助。

例如：

```json
{
  "examples": [
    {
      "user_request": "请检查这份供应商合同中的付款周期是否对我们不利。",
      "expected_capability": "payment_terms_review",
      "required_inputs": ["contract_document"],
      "expected_output": "risk_report"
    },
    {
      "user_request": "帮我看一下这份合同有没有过高的违约责任。",
      "expected_capability": "liability_clause_review",
      "required_inputs": ["contract_document"],
      "expected_output": "risk_report"
    }
  ]
}
```

好的 examples 可以帮助模型路由，也可以帮助开发者理解边界。

## 35.17 一个完整 Skill Manifest 示例

下面是一个简化但完整的例子：

```json
{
  "id": "skill.contract_review",
  "name": "Contract Review Skill",
  "version": "1.3.0",
  "owner": "legal-platform-team",
  "category": "legal",
  "description": "Review supplier contracts for payment terms, liability clauses, termination conditions, and compliance risks. Output a structured risk report with citations.",
  "capabilities": [
    {
      "name": "payment_terms_review",
      "description": "Identify unfavorable payment terms and compare them with company policy.",
      "risk_level": "medium"
    },
    {
      "name": "liability_clause_review",
      "description": "Review liability and indemnity clauses for high-risk obligations.",
      "risk_level": "high"
    }
  ],
  "inputs": {
    "required": [
      {
        "name": "contract_document",
        "type": "resource_ref",
        "accepted_uri_schemes": ["file://", "doc://", "artifact://"]
      }
    ],
    "optional": [
      {
        "name": "jurisdiction",
        "type": "string",
        "default": "CN"
      }
    ]
  },
  "outputs": {
    "types": ["risk_report", "citation_list", "artifact"],
    "supports_citations": true,
    "supports_confidence": true
  },
  "tools": [
    {
      "name": "read_document",
      "purpose": "Read contract text and section structure.",
      "required": true
    },
    {
      "name": "search_policy",
      "purpose": "Find internal legal policy for comparison.",
      "required": true
    }
  ],
  "resources": [
    {
      "uri": "kb://legal/contract-review-policy",
      "purpose": "Internal policy for contract risk review.",
      "required": true
    }
  ],
  "permissions": [
    {
      "scope": "documents.read",
      "reason": "Read contract documents provided by the user.",
      "required": true
    },
    {
      "scope": "legal_policy.read",
      "reason": "Compare contract clauses with internal policy.",
      "required": true
    }
  ],
  "safety": {
    "external_sharing": false,
    "requires_human_approval_for": ["high_risk_report"],
    "forbidden_actions": ["modify_contract", "send_to_external_party"]
  },
  "eval": {
    "metrics": ["risk_identification_recall", "citation_accuracy", "format_compliance"],
    "minimum_quality_gate": {
      "citation_accuracy": 0.95
    }
  }
}
```

这个 Manifest 不只是给人看的文档，也是平台执行安装、路由、授权、安全审核和评估的依据。

## 35.18 常见 Manifest 设计错误

### 35.18.1 描述过泛

例如“帮助用户处理合同问题”。这会导致路由不准，也无法评估。

### 35.18.2 权限声明过大

例如申请 documents.all、email.send、database.write，但实际只需要 documents.read。权限过大会增加审核成本和安全风险。

### 35.18.3 输入输出不稳定

Manifest 声明输出 risk_report，但实际有时返回纯文本，有时返回表格，有时返回 JSON，上游很难集成。

### 35.18.4 没有安全边界

不声明是否允许外部分享、不声明是否需要人工确认、不声明禁止动作，都会给平台治理带来风险。

### 35.18.5 没有 Eval

Skill 不能只发布不评估。尤其是法律、金融、医疗、安全、代码修改等高风险 Skill，必须有质量门槛。

## 35.19 Skill Manifest 审计指标与最小 demo

为了让 Manifest 设计从“字段清单”变成可审计工程对象，可以把一个候选 manifest 写成样本：

```math
m_i=(a_i,d_i,c_i,u_i,o_i,t_i,r_i,p_i,w_i,s_i,q_i,e_i,x_i,v_i,l_i,z_i)
```

其中，`a_i` 是身份元信息，`d_i` 是能力描述，`c_i` 是 capabilities，`u_i` 是 inputs，`o_i` 是 outputs，`t_i` 是 tools，`r_i` 是 resources / prompts，`p_i` 是 permissions，`w_i` 是 workflow，`s_i` 是 safety，`q_i` 是 configuration，`e_i` 是 eval，`x_i` 是 examples，`v_i` 是 version / lifecycle，`l_i` 是 audit label，`z_i` 是评估标签。

对检查项 `j`，定义覆盖率：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[I_j(m_i)=1]
```

其中，`I_j(m_i)=1` 表示第 `i` 个 manifest 在第 `j` 个检查项上通过。

如果要把 description 用于路由，还可以单独看触发准确率和拒触发准确率：

```math
A_{\mathrm{trigger}}
=\frac{\sum_i \mathbb{1}[\hat{b}_i=b_i]}{N},\qquad
B_{\mathrm{reject}}
=\frac{\sum_i \mathbb{1}[\hat{b}_i=0\wedge b_i=0]}{\sum_i \mathbb{1}[b_i=0]}
```

其中，`b_i=1` 表示这个请求应该触发该 Skill，`\hat{b}_i` 是路由器基于 manifest 的判断。

Skill Manifest 上线门禁可以写成：

```math
G_{\mathrm{skill\_manifest}}
=\prod_{j\in\mathcal{J}}\mathbb{1}[C_j\ge \tau_j]
```

综合打分可以写成：

```math
S_{\mathrm{skill\_manifest}}
=\sum_{j\in\mathcal{J}}w_j C_j,\qquad
\sum_{j\in\mathcal{J}}w_j=1
```

这个公式的核心含义是：Manifest 不是字段越多越好，而是每个字段都要支持真实治理问题，例如“能否稳定触发”“能否知道输入缺什么”“能否证明权限最小化”“能否升级和回滚”“能否用 eval 判断变好还是变坏”。

下面是一个 0 依赖 demo，用 toy manifest 检查常见设计错误。

```python
from collections import OrderedDict
import re

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
VAGUE_WORDS = {"powerful", "smart", "assistant", "all", "everything", "various", "各种", "强大", "万能", "所有"}
GENERIC_CAPABILITIES = {"help", "assist", "do_anything", "legal_help", "office_help"}
ATOMIC_CAPABILITY_PREFIXES = ("find_word", "count_", "extract_field")

REQUIRED_IDENTITY = {"id", "name", "version", "owner", "category", "status"}
REQUIRED_AUDIT = {"owner", "version", "changelog", "review_status"}


def make_manifest(**overrides):
    base = {
        "id": "skill.contract_review",
        "name": "Contract Review Skill",
        "version": "1.3.0",
        "owner": "legal-platform-team",
        "category": "legal",
        "status": "stable",
        "description": "Review supplier contracts for payment terms, liability clauses, termination conditions, and compliance risks. Output a structured risk report with citations.",
        "capabilities": [
            {"name": "payment_terms_review", "description": "Identify unfavorable payment terms and compare with policy.", "risk_level": "medium"},
            {"name": "liability_clause_review", "description": "Review liability and indemnity clauses for high-risk obligations.", "risk_level": "high"},
        ],
        "inputs": {"required": [{"name": "contract_document", "type": "resource_ref", "accepted_uri_schemes": ["file://", "doc://", "artifact://"]}]},
        "outputs": {"types": ["risk_report", "citation_list", "artifact"], "schema": {"overall_risk": "enum", "findings": "array", "citations": "array"}, "supports_citations": True, "supports_confidence": True},
        "tools": [
            {"name": "read_document", "purpose": "Read contract text and section structure.", "required": True},
            {"name": "search_policy", "purpose": "Find internal legal policy for comparison.", "required": True},
        ],
        "resources": [{"uri": "kb://legal/contract-review-policy", "purpose": "Internal policy for contract risk review.", "required": True}],
        "prompts": [{"id": "prompt.contract_risk_review.v2", "purpose": "Guide risk review with citations.", "version": "2.1.0"}],
        "workflow": {"steps": [
            {"id": "read_contract", "type": "tool", "tool": "read_document"},
            {"id": "retrieve_policy", "type": "tool", "tool": "search_policy"},
            {"id": "draft_findings", "type": "prompt", "prompt": "prompt.contract_risk_review.v2"},
            {"id": "validate_citations", "type": "tool", "tool": "check_citations"},
        ], "approval_points": ["create_report"]},
        "permissions": [
            {"scope": "documents.read", "reason": "Read user-provided contract documents.", "required": True},
            {"scope": "legal_policy.read", "reason": "Compare clauses with internal policy.", "required": True},
            {"scope": "artifacts.write", "reason": "Create the final review report.", "required": False},
        ],
        "configuration": {"fields": [
            {"name": "default_jurisdiction", "type": "string", "default": "CN", "description": "Default legal jurisdiction."},
            {"name": "require_human_approval_for_high_risk", "type": "boolean", "default": True, "description": "Require approval before high-risk output is finalized."},
        ]},
        "safety": {"external_sharing": False, "requires_human_approval_for": ["high_risk_report", "external_send"], "forbidden_actions": ["modify_contract", "send_to_external_party"], "redaction": {"enabled": True, "fields": ["personal_contact", "bank_account"]}},
        "eval": {"metrics": ["risk_identification_recall", "citation_accuracy", "format_compliance"], "golden_sets": ["eval://legal/contract-review-v1"], "minimum_quality_gate": {"citation_accuracy": 0.95, "sensitive_info_leakage_rate": 0.0}},
        "examples": [
            {"user_request": "Check whether payment terms in this supplier contract are unfavorable.", "expected_capability": "payment_terms_review", "required_inputs": ["contract_document"], "expected_output": "risk_report"},
            {"user_request": "Review this contract for excessive liability obligations.", "expected_capability": "liability_clause_review", "required_inputs": ["contract_document"], "expected_output": "risk_report"},
            {"user_request": "Send the signed contract to an external party.", "should_trigger": False, "reason": "external sending is outside this Skill boundary"},
        ],
        "lifecycle": {"changelog": ["1.3.0 adds citation validation"], "rollback": True, "deprecation_policy": "keep two minor versions"},
        "audit": {"owner": "legal-platform-team", "version": "1.3.0", "changelog": True, "review_status": "approved"},
        "high_risk": True,
    }
    base.update(overrides)
    return base


def has_keys(item, keys):
    return keys.issubset(item.keys())


def identity_metadata(m):
    return has_keys(m, REQUIRED_IDENTITY) and SEMVER.match(m.get("version", "")) is not None


def description_quality(m):
    desc = m.get("description", "")
    lower = desc.lower()
    has_task = any(word in lower for word in ["review", "identify", "compare", "output", "cite", "citations"])
    too_vague = any(word in lower for word in VAGUE_WORDS)
    return 60 <= len(desc) <= 600 and has_task and not too_vague


def capability_granularity(m):
    caps = m.get("capabilities", [])
    if not 1 <= len(caps) <= 6:
        return False
    for cap in caps:
        name = cap.get("name", "")
        if name in GENERIC_CAPABILITIES or name.startswith(ATOMIC_CAPABILITY_PREFIXES):
            return False
        if not cap.get("description") or cap.get("risk_level") not in {"low", "medium", "high"}:
            return False
    return True


def input_contract(m):
    required = m.get("inputs", {}).get("required", [])
    return bool(required) and all(x.get("name") and x.get("type") for x in required)


def output_contract(m):
    out = m.get("outputs", {})
    return bool(out.get("types")) and bool(out.get("schema")) and out.get("supports_citations") is True


def tool_dependency_purpose(m):
    tools = m.get("tools", [])
    return bool(tools) and all(t.get("name") and t.get("purpose") for t in tools)


def resource_prompt_binding(m):
    resources = m.get("resources", [])
    prompts = m.get("prompts", [])
    resources_ok = bool(resources) and all(r.get("uri") and r.get("purpose") for r in resources)
    prompts_ok = bool(prompts) and all(p.get("id") and p.get("purpose") and p.get("version") for p in prompts)
    return resources_ok and prompts_ok


def workflow_readiness(m):
    workflow = m.get("workflow", {})
    steps = workflow.get("steps", [])
    has_steps = len(steps) >= 3 and all(s.get("id") and s.get("type") for s in steps)
    if m.get("high_risk"):
        return has_steps and bool(workflow.get("approval_points"))
    return has_steps


def permission_least_privilege(m):
    perms = m.get("permissions", [])
    if not perms:
        return False
    bad = {"all", "*", "documents.all", "admin"}
    return all(p.get("scope") not in bad and p.get("reason") for p in perms)


def configuration_schema(m):
    fields = m.get("configuration", {}).get("fields", [])
    return bool(fields) and all(f.get("name") and f.get("type") and "default" in f for f in fields)


def safety_policy(m):
    safety = m.get("safety", {})
    return safety.get("external_sharing") is False and bool(safety.get("forbidden_actions")) and bool(safety.get("requires_human_approval_for"))


def eval_gate(m):
    ev = m.get("eval", {})
    return bool(ev.get("metrics")) and bool(ev.get("golden_sets")) and bool(ev.get("minimum_quality_gate"))


def examples_trigger_coverage(m):
    examples = m.get("examples", [])
    positive = [e for e in examples if e.get("should_trigger", True)]
    negative = [e for e in examples if e.get("should_trigger") is False]
    return len(positive) >= 2 and bool(negative) and all(e.get("user_request") for e in examples)


def version_lifecycle(m):
    life = m.get("lifecycle", {})
    return SEMVER.match(m.get("version", "")) is not None and bool(life.get("changelog")) and life.get("rollback") is True


def audit_readiness(m):
    return has_keys(m.get("audit", {}), REQUIRED_AUDIT)


CHECKS = OrderedDict([
    ("identity_metadata", identity_metadata),
    ("description_quality", description_quality),
    ("capability_granularity", capability_granularity),
    ("input_contract", input_contract),
    ("output_contract", output_contract),
    ("tool_dependency_purpose", tool_dependency_purpose),
    ("resource_prompt_binding", resource_prompt_binding),
    ("workflow_readiness", workflow_readiness),
    ("permission_least_privilege", permission_least_privilege),
    ("configuration_schema", configuration_schema),
    ("safety_policy", safety_policy),
    ("eval_gate", eval_gate),
    ("examples_trigger_coverage", examples_trigger_coverage),
    ("version_lifecycle", version_lifecycle),
    ("audit_readiness", audit_readiness),
])

MANIFESTS = [
    make_manifest(id="contract_review_ok"),
    make_manifest(id="missing_identity_bad", owner=None, version="1.0"),
    make_manifest(id="vague_description_bad", description="A powerful assistant that helps with all things."),
    make_manifest(id="capability_generic_bad", capabilities=[{"name": "legal_help", "description": "Help with legal work.", "risk_level": "medium"}]),
    make_manifest(id="input_missing_required_bad", inputs={"optional": [{"name": "contract_document", "type": "resource_ref"}]}),
    make_manifest(id="output_unstable_bad", outputs={"types": ["text"]}),
    make_manifest(id="tools_no_purpose_bad", tools=[{"name": "read_document", "required": True}]),
    make_manifest(id="resource_prompt_missing_bad", resources=[], prompts=[]),
    make_manifest(id="workflow_no_approval_bad", workflow={"steps": [{"id": "read", "type": "tool"}]}),
    make_manifest(id="permissions_all_bad", permissions=[{"scope": "all", "reason": "convenience", "required": True}]),
    make_manifest(id="config_untyped_bad", configuration={"fields": [{"name": "default_jurisdiction"}]}),
    make_manifest(id="safety_missing_bad", safety={"external_sharing": True}),
    make_manifest(id="eval_missing_bad", eval={}),
    make_manifest(id="examples_missing_bad", examples=[{"user_request": "Review this contract."}]),
    make_manifest(id="audit_missing_bad", audit={"owner": "legal-platform-team"}, lifecycle={"changelog": [], "rollback": False}),
]

metrics = OrderedDict()
failed_by_manifest = OrderedDict()
for name, fn in CHECKS.items():
    passes = [fn(m) for m in MANIFESTS]
    metrics[name] = round(sum(passes) / len(passes), 3)
    for manifest, ok in zip(MANIFESTS, passes):
        if not ok:
            failed_by_manifest.setdefault(manifest["id"], []).append(name)

thresholds = {name: 0.95 for name in CHECKS}
failed_gates = [name for name, value in metrics.items() if value < thresholds[name]]

smoke = OrderedDict([
    ("complete_manifest_passes", all(fn(MANIFESTS[0]) for fn in CHECKS.values())),
    ("caught_vague_description", not description_quality(MANIFESTS[2])),
    ("caught_overbroad_permission", not permission_least_privilege(MANIFESTS[9])),
    ("caught_missing_eval", not eval_gate(MANIFESTS[12])),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_manifests=", list(failed_by_manifest.keys()))
print("failed_gates=", failed_gates)
print("skill_manifest_gate_pass=", not failed_gates)
```

运行后可以看到：

```text
smoke= {'complete_manifest_passes': True, 'caught_vague_description': True, 'caught_overbroad_permission': True, 'caught_missing_eval': True}
metrics= {'identity_metadata': 0.933, 'description_quality': 0.933, 'capability_granularity': 0.933, 'input_contract': 0.933, 'output_contract': 0.933, 'tool_dependency_purpose': 0.933, 'resource_prompt_binding': 0.933, 'workflow_readiness': 0.933, 'permission_least_privilege': 0.933, 'configuration_schema': 0.933, 'safety_policy': 0.933, 'eval_gate': 0.933, 'examples_trigger_coverage': 0.933, 'version_lifecycle': 0.867, 'audit_readiness': 0.933}
failed_manifests= ['missing_identity_bad', 'vague_description_bad', 'capability_generic_bad', 'input_missing_required_bad', 'output_unstable_bad', 'tools_no_purpose_bad', 'resource_prompt_missing_bad', 'workflow_no_approval_bad', 'permissions_all_bad', 'config_untyped_bad', 'safety_missing_bad', 'eval_missing_bad', 'examples_missing_bad', 'audit_missing_bad']
failed_gates= ['identity_metadata', 'description_quality', 'capability_granularity', 'input_contract', 'output_contract', 'tool_dependency_purpose', 'resource_prompt_binding', 'workflow_readiness', 'permission_least_privilege', 'configuration_schema', 'safety_policy', 'eval_gate', 'examples_trigger_coverage', 'version_lifecycle', 'audit_readiness']
skill_manifest_gate_pass= False
```

这段 demo 的价值在于：它把“Manifest 写得好不好”转成了可回归检查。面试中可以强调，Manifest 质量不只是文案质量，而是 discovery、routing、permission、safety、workflow、eval 和 lifecycle 能否被平台稳定消费。

## 35.20 面试高频题

### 题 1：Skill Manifest 是什么？

参考回答：

Skill Manifest 是 Skill 的结构化能力声明和治理入口，描述 Skill 的身份、能力、输入输出、依赖工具、资源、提示、工作流、权限、配置、安全策略、评估标准和示例。平台用它做安装、路由、授权、安全审核、版本管理和评估。

### 题 2：一个好的 Skill Manifest 应该包含哪些字段？

参考回答：

至少包含 id、name、version、description、capabilities、inputs、outputs、tools、resources、prompts、workflow、permissions、configuration、safety、eval 和 examples。生产系统还应包含 owner、category、status、兼容版本、数据分类和审批策略。

### 题 3：能力描述应该怎么写？

参考回答：

能力描述要具体、可验证、可限制。应说明任务对象、处理动作、适用场景、输出形式和边界。避免“万能助手”式描述，也避免过细到单个 API 操作。

### 题 4：为什么权限声明要写 reason？

参考回答：

因为管理员或平台审核时需要知道 Skill 为什么需要某个权限。只有 scope 没有 reason，很难判断权限是否合理。reason 也能帮助后续审计和最小权限优化。

### 题 5：Skill Manifest 和 Agent Card 有什么区别？

参考回答：

Agent Card 描述一个 Agent 的身份、能力、交互模式和服务发现信息；Skill Manifest 描述一个可复用能力包的工具、资源、提示、工作流、权限、配置和评估。Agent 是执行主体，Skill 是能力包。一个 Agent 可以安装多个 Skill。

## 35.21 小练习

1. 为“会议总结 Skill”设计一个 Manifest，至少包含 description、inputs、outputs、tools、permissions、eval。
2. 把“万能办公助手 Skill”的描述改写成 3 个边界清晰的 Skill 描述。
3. 为“代码修复 Skill”列出 5 个安全策略字段。
4. 设计一个 Skill 的 examples 字段，用于帮助路由器识别适用任务。
5. 思考：Skill Manifest 中哪些字段应该参与版本兼容性判断？
6. 运行本章 demo，把 `permissions_all_bad` 改成最小权限声明，观察 `permission_least_privilege` 是否恢复。
7. 给 demo 增加一个 `negative_example_missing_bad` 样本，验证缺少“不该触发”示例会影响路由边界。

## 35.22 本章小结

本章我们讲了 Skill Manifest 与能力描述设计。

Manifest 是 Skill 的结构化说明书，也是平台治理入口。它应该描述身份、能力、输入、输出、工具、资源、提示、工作流、权限、配置、安全、评估和示例。好的能力描述应该具体、可验证、可限制；权限声明应该遵守最小权限并说明 reason；安全策略和 Eval 不应该缺失。

你可以把本章重点记成一句话：

> Skill Manifest 写得越清楚，平台就越容易做路由、授权、审核、升级、评估和治理。

下一章我们会继续讲 Skill 的安装、启用、禁用和版本更新，也就是一个 Skill 从发布到被用户实际使用的生命周期管理。
