# 第 35 章 Skill Manifest 与能力描述设计

前两章我们讲清楚了 Skill 的定义，以及 Plugin、Action、Tool、Skill、Workflow 的边界。

这一章进入 Skill 的工程设计核心：Skill Manifest。

如果说 Skill 是一个能力包，那么 Manifest 就是这个能力包的说明书、身份证、权限申请表、能力声明和治理入口。平台要安装 Skill，需要看 Manifest；Agent 要选择 Skill，需要看 Manifest；安全审核要判断风险，需要看 Manifest；版本升级和评估也离不开 Manifest。

本章重点讲：一个好的 Skill Manifest 应该包含哪些字段，能力描述应该怎么写，权限和配置如何声明，示例和评估如何设计，哪些 Manifest 写法会导致路由错误、安全风险和维护困难。

你可以先记住一句话：

> Skill Manifest 的目标不是“描述得好看”，而是让平台、Agent、管理员和用户都能可靠理解这个 Skill 能做什么、需要什么、会影响什么、如何评估。

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

## 35.19 面试高频题

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

## 35.20 小练习

1. 为“会议总结 Skill”设计一个 Manifest，至少包含 description、inputs、outputs、tools、permissions、eval。
2. 把“万能办公助手 Skill”的描述改写成 3 个边界清晰的 Skill 描述。
3. 为“代码修复 Skill”列出 5 个安全策略字段。
4. 设计一个 Skill 的 examples 字段，用于帮助路由器识别适用任务。
5. 思考：Skill Manifest 中哪些字段应该参与版本兼容性判断？

## 35.21 本章小结

本章我们讲了 Skill Manifest 与能力描述设计。

Manifest 是 Skill 的结构化说明书，也是平台治理入口。它应该描述身份、能力、输入、输出、工具、资源、提示、工作流、权限、配置、安全、评估和示例。好的能力描述应该具体、可验证、可限制；权限声明应该遵守最小权限并说明 reason；安全策略和 Eval 不应该缺失。

你可以把本章重点记成一句话：

> Skill Manifest 写得越清楚，平台就越容易做路由、授权、审核、升级、评估和治理。

下一章我们会继续讲 Skill 的安装、启用、禁用和版本更新，也就是一个 Skill 从发布到被用户实际使用的生命周期管理。
