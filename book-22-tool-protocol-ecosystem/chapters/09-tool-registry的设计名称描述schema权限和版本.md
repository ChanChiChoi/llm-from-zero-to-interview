# 第九章：Tool Registry 的设计：名称、描述、Schema、权限和版本

## 9.1 本章定位

前八章完成了第一部分“工具调用基础”：从 function calling 协议到 schema、tool choice、参数、结果、失败恢复和评估。

从本章开始进入第二部分“工具注册与运行时”。

一个真实企业系统里，工具不会只有两个 demo 函数。它可能有：

1. CRM 查询工具。
2. 订单工具。
3. 库存工具。
4. 报销工具。
5. 知识库搜索工具。
6. 文件工具。
7. 邮件工具。
8. 日历工具。
9. 数据分析工具。
10. 内部审批工具。

如果没有统一注册和治理，这些工具很快会失控：名称混乱、schema 不一致、权限不清、版本不可追踪、owner 不明确、故障没人负责、模型不知道该用哪个、审计无法复现。

Tool Registry 要解决的就是这个问题。

本章的核心观点是：

```text
Tool Registry 不是一个函数列表，而是企业工具能力的契约中心、治理中心和运行时索引。
```

## 9.2 Tool Registry 解决什么问题

没有 Tool Registry 时，工具通常散落在代码里：

```python
tools = [get_weather, search_docs, send_email]
```

这在 demo 中没问题，但规模上来后会遇到：

1. 谁能调用这个工具不清楚。
2. 工具描述没人维护。
3. schema 改了但 eval 没更新。
4. 工具有副作用但没有确认机制。
5. 工具 owner 离职后没人负责。
6. 多个团队注册了相似工具。
7. 模型看到太多工具，选择困难。
8. 线上故障无法按版本复现。
9. 工具下线后历史 trace 解释不了。
10. 不同 provider 需要不同 schema 投影。

Tool Registry 的职责是统一管理：

1. 工具身份。
2. 工具语义。
3. 参数契约。
4. 执行入口。
5. 权限策略。
6. 风险等级。
7. 版本生命周期。
8. owner 和 SLO。
9. eval 和质量指标。
10. provider 投影和运行时元数据。

## 9.3 Registry Item 的基本结构

一个完整的工具注册项可以这样设计：

```json
{
  "name": "get_customer_order_history",
  "display_name": "查询客户订单历史",
  "version": "1.3.0",
  "description": "查询客户订单历史。仅用于已授权客服场景。",
  "input_schema": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string"},
      "limit": {"type": "integer", "minimum": 1, "maximum": 50}
    },
    "required": ["customer_id"],
    "additionalProperties": false
  },
  "runtime": {
    "executor": "crm.get_order_history",
    "timeout_ms": 3000,
    "retry_policy": "read_only_default",
    "idempotent": true
  },
  "security": {
    "required_permissions": ["crm:order:read"],
    "sensitivity": "high",
    "side_effect": false,
    "requires_confirmation": false
  },
  "lifecycle": {
    "status": "active",
    "owner": "crm-platform",
    "created_at": "2026-01-10T00:00:00Z",
    "updated_at": "2026-05-01T00:00:00Z"
  }
}
```

这比模型 API 里的 `tools` 字段多很多信息。

原因是：模型只需要看一部分，但 runtime 和治理系统需要完整契约。

## 9.4 模型可见字段和运行时字段

Tool Registry 中的信息可以分成两类。

第一类是模型可见字段：

1. `name`。
2. `description`。
3. `input_schema`。
4. 参数 description。

这些字段会影响模型是否调用、调用哪个工具、生成什么参数。

第二类是 runtime 字段：

1. executor。
2. timeout。
3. retry policy。
4. permission policy。
5. risk level。
6. owner。
7. version。
8. observability。
9. rollout。
10. deprecation。

这些字段通常不直接传给模型，但 runtime 必须使用。

例如 `requires_confirmation=true` 既可以在 description 中提示模型，也必须由 runtime 强制执行。不能只靠模型自觉。

## 9.5 工具名称设计

工具名称是 Tool Registry 中最基础的标识。

好工具名应该：

1. 稳定。
2. 唯一。
3. 语义清晰。
4. 动词加对象。
5. 能和相似工具区分。
6. 不暴露内部系统细节。

推荐：

1. `get_weather_forecast`。
2. `search_internal_policy`。
3. `create_calendar_event`。
4. `get_customer_order_history`。
5. `draft_email`。

不推荐：

1. `tool1`。
2. `query`。
3. `search`。
4. `do_action`。
5. `crm_api_v2_call`。

名称太宽会导致模型误选。名称太技术化会让模型难以理解业务语义。

在 Registry 中，工具名还承担主键作用。因此不要随便改名。需要改名时，应走版本和迁移流程。

## 9.6 description 设计

description 是模型理解工具的主要来源。

好的 description 要包含：

1. 工具做什么。
2. 适用于什么用户请求。
3. 不适用于什么场景。
4. 数据来源是什么。
5. 是否实时。
6. 是否有副作用。
7. 是否需要确认。
8. 关键权限限制。

差的 description：

```json
{"description":"查询订单"}
```

更好的 description：

```json
{
  "description": "查询当前用户有权限访问的客户订单历史，包括订单状态、下单时间和金额摘要。仅用于客服或售后场景，不用于查询无权限客户、支付凭证或敏感财务明细。"
}
```

description 写得好，可以提升：

1. 工具选择准确率。
2. 参数正确率。
3. 不该调用时的拒绝能力。
4. 安全边界表达。

但 description 不是安全边界。真实权限必须由 runtime 判断。

## 9.7 input_schema 与 output_schema

Registry 至少要保存 input_schema。

input_schema 用于：

1. 给模型生成参数。
2. 给 runtime 校验参数。
3. 给 eval 判断参数正确性。
4. 给文档系统生成说明。

生产系统还建议保存 output_schema。

例如：

```json
{
  "output_schema": {
    "type": "object",
    "properties": {
      "orders": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "order_id": {"type": "string"},
            "status": {"type": "string"},
            "created_at": {"type": "string"}
          }
        }
      }
    }
  }
}
```

output_schema 的价值：

1. 帮助 runtime 校验工具实现是否返回合法结果。
2. 帮助 adapter 把结果投影给模型。
3. 帮助最终回答引用字段。
4. 帮助 eval 判断工具结果使用是否正确。
5. 帮助生成 mock tools。

很多系统只管输入 schema，不管输出 schema，导致工具返回结果随意变动，模型上下文和评估都不稳定。

## 9.8 Runtime 元数据

Registry 需要告诉 runtime 如何执行工具。

常见 runtime 字段：

1. executor 类型。
2. handler 名称。
3. endpoint。
4. timeout。
5. retry policy。
6. concurrency limit。
7. cache policy。
8. idempotency policy。
9. parallel allowed。
10. cancellation behavior。

示例：

```json
{
  "runtime": {
    "executor_type": "http",
    "endpoint": "https://internal-api.example.com/orders/history",
    "method": "POST",
    "timeout_ms": 3000,
    "retry_policy": {
      "max_attempts": 2,
      "backoff": "exponential_jitter"
    },
    "concurrency_limit": 50,
    "cache_ttl_ms": 60000,
    "idempotent": true,
    "parallel_allowed": true
  }
}
```

模型不需要知道 endpoint，但 runtime 需要。

## 9.9 权限元数据

工具权限不能写在代码注释里，应该进入 Registry。

权限字段可以包括：

1. required permissions。
2. allowed roles。
3. tenant constraints。
4. data scope。
5. row-level policy。
6. field-level policy。
7. confirmation requirement。
8. approval workflow。

示例：

```json
{
  "security": {
    "required_permissions": ["crm:order:read"],
    "data_scope": "assigned_customers_only",
    "field_policy": {
      "hide_fields": ["payment_token", "internal_notes"]
    },
    "requires_confirmation": false
  }
}
```

权限分三层：

1. 工具级权限：能不能调用这个工具。
2. 对象级权限：能不能访问这个 customer_id 或 order_id。
3. 字段级权限：能不能看到某些敏感字段。

Registry 保存策略，runtime 执行策略。

## 9.10 风险等级和副作用标记

工具风险等级非常重要。

可以设计：

1. `risk_level=low`：只读、低敏感查询。
2. `risk_level=medium`：读取敏感数据或较高成本。
3. `risk_level=high`：有副作用、修改状态、发送外部消息。
4. `risk_level=critical`：财务、权限、删除、不可逆动作。

还要标记 side effect：

```json
{
  "security": {
    "risk_level": "high",
    "side_effect": true,
    "requires_confirmation": true,
    "requires_idempotency_key": true,
    "auto_execution_allowed": false
  }
}
```

这些字段影响：

1. tool choice policy。
2. 是否允许 auto 模式调用。
3. 是否允许 parallel calls。
4. 是否必须用户确认。
5. 是否需要人工审批。
6. 是否允许重试。
7. trace 和审计级别。

高风险工具不能只靠 description 写“请谨慎使用”。必须由 runtime 强制。

## 9.11 版本管理

工具会演进。

可能变化包括：

1. description 改动。
2. input_schema 改动。
3. output_schema 改动。
4. 权限策略改动。
5. executor 改动。
6. retry policy 改动。
7. 风险等级改动。

Registry 应保存版本。

建议至少记录：

1. semantic version。
2. schema hash。
3. changelog。
4. created_by。
5. approved_by。
6. rollout status。
7. deprecated_at。
8. sunset_at。

示例：

```json
{
  "version": "1.4.0",
  "schema_hash": "sha256:abc123",
  "change_type": "backward_compatible",
  "changelog": "add optional field source_type",
  "approved_by": "tool-governance-team"
}
```

Trace 必须记录工具版本。否则线上问题无法复现。

## 9.12 兼容性规则

不是所有工具改动都兼容。

通常：

1. 新增可选输入字段：大多兼容。
2. 新增必填输入字段：不兼容。
3. 删除输入字段：可能不兼容。
4. 修改字段含义：高度危险。
5. 修改 enum 值：可能不兼容。
6. 输出新增字段：通常兼容。
7. 输出删除字段：可能不兼容。
8. 权限收紧：可能影响用户任务。
9. 权限放宽：需要安全审查。

Registry 可以自动做 schema diff，并给出兼容性判断。

例如：

```text
v1.2.0 → v1.3.0: add optional field top_k, compatible
v1.3.0 → v2.0.0: rename customer_id to account_id, breaking
```

兼容性判断会影响发布方式：兼容变更可以灰度，不兼容变更需要新工具名或 major version。

## 9.13 生命周期状态

工具不应该只有 active / inactive 两种状态。

可以设计生命周期：

1. `draft`：草稿，不能被生产模型使用。
2. `review`：等待审核。
3. `staging`：可在测试环境使用。
4. `active`：生产可用。
5. `deprecated`：不推荐新流量使用，但历史可用。
6. `sunset`：准备下线。
7. `disabled`：禁用，不允许调用。
8. `archived`：归档，只保留历史记录。

不同状态对应不同策略：

| 状态 | 模型可见 | runtime 可执行 | 说明 |
|---|---|---|---|
| draft | 否 | 否 | 开发中 |
| staging | 测试可见 | 测试可执行 | 预发验证 |
| active | 是 | 是 | 生产可用 |
| deprecated | 可选 | 是 | 不推荐新流量 |
| disabled | 否 | 否 | 被禁用 |
| archived | 否 | 否 | 仅审计 |

生命周期要和审批、灰度、回滚、文档和 eval 绑定。

## 9.14 Owner 和责任边界

每个工具必须有 owner。

owner 负责：

1. schema 维护。
2. description 质量。
3. 工具可用性。
4. 权限策略。
5. 故障处理。
6. 文档。
7. eval 样例。
8. 版本发布。
9. 下线通知。

没有 owner 的工具不能进入生产。

Registry 中应记录：

```json
{
  "owner": {
    "team": "crm-platform",
    "oncall": "crm-oncall",
    "contact": "crm-platform@example.com"
  }
}
```

当工具错误率升高、延迟异常、权限投诉或 eval 回归时，系统应能找到责任团队。

## 9.15 Tool Registry 与 Tool Router 的关系

Tool Registry 和 Tool Router 容易混淆。

Registry 回答：

```text
系统有哪些工具？这些工具是什么、怎么调用、谁能用、风险如何？
```

Router 回答：

```text
当前请求应该给模型暴露哪些工具，应该调用哪个工具？
```

Router 依赖 Registry。

流程：

```text
用户请求 + 上下文
  ↓
Router 查询 Registry
  ↓
按权限、场景、风险、版本过滤工具
  ↓
生成候选 tools 传给模型
```

所以 Registry 是工具事实库，Router 是决策层。

第 10 章会专门讲 Tool Router。

## 9.16 Tool Registry 与 MCP Server 的关系

后面会讲 MCP。这里先建立关系。

MCP Server 可以暴露一组 tools、resources 和 prompts。对一个企业平台来说，MCP Server 自己也可以看成工具提供方。

Registry 可以注册：

1. 本地函数工具。
2. HTTP API 工具。
3. 数据库查询工具。
4. MCP Server 暴露的工具。
5. 第三方插件工具。
6. 内部 workflow 工具。

也就是说：

```text
MCP 解决工具和上下文的协议连接，Tool Registry 解决企业内部工具资产管理和治理。
```

二者可以结合：Registry 保存 MCP Server 的 endpoint、capabilities、权限和版本，runtime 动态发现并投影给模型。

## 9.17 Registry 的存储设计

Registry 可以存储在：

1. 配置文件。
2. 数据库。
3. 配置中心。
4. 服务注册中心。
5. GitOps 仓库。
6. 专门的工具平台。

小系统可以用配置文件：

```text
tools/weather.yaml
tools/search_policy.yaml
```

企业系统通常需要数据库和审核流程。

关键能力：

1. 查询工具。
2. 版本历史。
3. 审批流程。
4. 灰度配置。
5. 权限策略。
6. 审计日志。
7. 变更 diff。
8. API 查询。

GitOps 的优点是变更可 review、可回滚。数据库平台的优点是动态配置和权限管理更方便。很多企业会混合使用。

## 9.18 Registry API 设计

Registry 应提供 API。

常见 API：

1. `list_tools(context)`：按上下文列出可用工具。
2. `get_tool(name, version)`：获取工具详情。
3. `resolve_tool(name)`：解析默认版本。
4. `register_tool(spec)`：注册工具。
5. `update_tool(spec)`：更新工具。
6. `deprecate_tool(name, version)`：废弃工具。
7. `validate_tool_spec(spec)`：校验工具定义。
8. `project_for_provider(tool, provider)`：生成 provider 格式。

runtime 常用的是只读 API：

```python
tools = registry.list_tools(
    user=user,
    tenant=tenant,
    scenario="customer_support",
    risk_budget="medium",
)
```

管理后台使用写 API，但写 API 必须有审批和审计。

## 9.19 Provider 投影

同一个 Registry tool，要投影成不同 provider 的格式。

内部定义：

```json
{
  "name": "get_weather_forecast",
  "description": "查询天气预报",
  "input_schema": {...}
}
```

Provider A 需要：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather_forecast",
    "description": "查询天气预报",
    "parameters": {...}
  }
}
```

Provider B 可能需要：

```json
{
  "name": "get_weather_forecast",
  "description": "查询天气预报",
  "input_schema": {...}
}
```

Provider 投影要处理：

1. 字段名差异。
2. JSON Schema 子集差异。
3. strict schema 支持差异。
4. tool choice 支持差异。
5. 描述长度限制。
6. 不支持字段的降级。

不要让业务工具定义直接绑定某一个 provider 格式。

## 9.20 Registry 与文档生成

Registry 也是工具文档来源。

可以自动生成：

1. 工具列表。
2. 参数说明。
3. 示例调用。
4. 权限说明。
5. 风险等级。
6. owner 信息。
7. 版本变更记录。
8. eval 覆盖情况。

文档不是额外负担，而是 Registry 元数据的投影。

如果文档和 schema 分开维护，很容易不一致。更好的方式是以 Registry 为 source of truth。

## 9.21 Registry 与 Eval

每个工具都应该关联 eval。

Registry 可以记录：

1. eval dataset id。
2. golden traces。
3. safety cases。
4. last eval score。
5. last regression result。
6. required passing threshold。

例如：

```json
{
  "quality": {
    "eval_dataset": "crm_order_tool_eval_v3",
    "last_tool_selection_accuracy": 0.94,
    "last_argument_accuracy": 0.91,
    "required_threshold": 0.9
  }
}
```

工具发布前应该跑 eval。高风险工具还要跑安全 eval。

如果 schema 或 description 改动导致 eval 下降，应阻止发布或要求审批。

## 9.22 Registry 与审计

Registry 变更本身也要审计。

需要记录：

1. 谁创建了工具。
2. 谁修改了 schema。
3. 谁放宽了权限。
4. 谁启用了高风险工具。
5. 谁批准上线。
6. 什么时候灰度。
7. 什么时候回滚。
8. 影响了哪些租户和用户。

工具调用 trace 记录“当时使用了哪个版本”，Registry audit 记录“这个版本是谁改的”。二者结合，才能完整追责和复现。

## 9.23 常见错误

### 9.23.1 把 Registry 当函数列表

问题：只存 name 和 handler，没有权限、版本、owner、风险等级。

修复：Registry 应管理完整工具契约和生命周期。

### 9.23.2 description 不维护

问题：工具行为变化了，但模型看到的说明没变。

修复：description 变更纳入 review 和 eval。

### 9.23.3 没有版本

问题：线上 trace 无法复现，schema 改动影响历史行为。

修复：每次工具契约变更都记录版本和 schema hash。

### 9.23.4 权限只写在业务代码里

问题：Router 无法过滤候选工具，审计也看不到权限意图。

修复：Registry 保存权限策略，runtime 强制执行。

### 9.23.5 高风险工具没有风险标记

问题：模型可能在 auto 模式下调用发邮件、删除、转账工具。

修复：标记 side_effect、risk_level、requires_confirmation。

### 9.23.6 没有 owner

问题：工具故障没人处理。

修复：无 owner 不允许生产 active。

### 9.23.7 直接绑定 provider 格式

问题：换模型厂商时大量业务工具定义要改。

修复：内部 Registry 格式和 provider projection 解耦。

### 9.23.8 下线工具不保留历史

问题：历史 trace 无法解释。

修复：工具 archived 后仍保留只读元数据和版本历史。

## 9.24 面试题：如何设计 Tool Registry

面试官可能问：

```text
如果企业有上百个 agent tools，你会怎么设计工具注册中心？
```

可以这样回答。

第一，定义工具元数据：

1. name。
2. description。
3. input_schema。
4. output_schema。
5. version。
6. owner。
7. tags。

第二，定义运行时元数据：

1. executor。
2. timeout。
3. retry policy。
4. concurrency limit。
5. cache policy。
6. idempotency。

第三，定义安全治理：

1. required permissions。
2. data scope。
3. risk level。
4. side effect。
5. confirmation requirement。
6. audit policy。

第四，支持版本和生命周期：

1. draft。
2. review。
3. staging。
4. active。
5. deprecated。
6. disabled。
7. archived。

第五，支持 runtime 查询：

1. 按用户权限过滤。
2. 按租户和场景过滤。
3. 按风险策略过滤。
4. 投影成不同 provider 格式。

第六，支持质量和审计：

1. eval dataset。
2. last eval score。
3. change log。
4. approval log。
5. trace 关联版本。

一句话总结：

```text
Tool Registry 应该是工具契约、权限、安全、版本、owner、eval 和 provider 投影的统一 source of truth，而不是一个简单函数数组。
```

## 9.25 小练习

### 练习 1：判断 Registry 字段缺失

下面工具定义有什么问题？

```json
{
  "name": "send_email",
  "description": "发送邮件",
  "handler": "email.send"
}
```

参考答案：缺少 input_schema、权限、风险等级、副作用标记、确认要求、幂等策略、timeout、owner、version 和 eval 信息。

### 练习 2：工具改动是否兼容

工具从 v1 增加一个可选字段 `top_k`。是否兼容？

参考答案：通常兼容，但仍需跑 eval，确认模型不会因为新字段误填或过度调用。

### 练习 3：高风险工具 Registry 标记

删除文件工具应标记哪些安全字段？

参考答案：`side_effect=true`、`risk_level=critical`、`requires_confirmation=true`、`auto_execution_allowed=false`、`requires_idempotency_key=true`，并设置路径范围权限和审计策略。

### 练习 4：Provider 投影

为什么不应该让业务工具定义直接使用某个模型厂商的 tools 格式？

参考答案：因为不同 provider 字段和能力不同，直接绑定会导致迁移困难。应使用内部统一 Registry 格式，再通过 adapter 投影成 provider 格式。

### 练习 5：工具下线

工具 disabled 后，是否可以删除所有元数据？

参考答案：不能。历史 trace 仍需要解释当时工具 schema、description、权限和版本，因此应 archived 保留只读历史。

## 9.26 本章小结

本章讲了 Tool Registry 的设计。

你需要掌握：

1. Tool Registry 不是函数列表，而是工具契约和治理中心。
2. Registry item 应包含 name、description、input_schema、output_schema、runtime、安全、版本、owner 和质量信息。
3. 模型可见字段和 runtime 字段要区分。
4. 工具名和 description 会直接影响模型选择。
5. input_schema 管参数，output_schema 管结果稳定性。
6. 权限、风险等级、副作用和确认要求必须进入 Registry。
7. 工具版本、schema hash 和 trace 关联是线上复现的基础。
8. 生命周期应包括 draft、review、staging、active、deprecated、disabled、archived。
9. Tool Registry 是 Tool Router、Tool Executor、权限系统、eval 和文档系统的 source of truth。
10. 多 provider 场景下，Registry 内部格式要和 provider tools 格式解耦。

如果只记一句话：

```text
工具越多，越不能靠代码里散落的函数列表管理；必须用 Tool Registry 把工具变成可发现、可治理、可评估、可审计、可演进的企业资产。
```

下一章会讲 Tool Router：什么时候调用哪个工具，重点解释候选工具过滤、意图路由、权限过滤、风险过滤和多工具选择策略。
