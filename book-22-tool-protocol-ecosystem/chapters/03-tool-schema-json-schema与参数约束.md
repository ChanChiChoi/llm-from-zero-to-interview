# 第三章：Tool Schema、JSON Schema 与参数约束

## 3.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，按 `WRITING_PLAN.md` 的要求重新核对了 OpenAI function calling / structured outputs、Anthropic tool use、Google Gemini function calling 和 JSON Schema 官方资料。各家 API 字段名、strict 模式、schema 支持子集和工具执行流程会随平台演进而变化，所以正文只抽象稳定层：工具名称、工具描述、参数 schema、必填字段、类型、枚举、范围、字符串模式、额外字段限制、runtime 校验、业务校验、权限检查和 trace 审计。

本讲不把某一家 provider 的当前字段名写成永久标准，也不把 JSON Schema 当作完整安全系统。更准确的边界是：

1. Tool Schema 约束模型应如何生成结构化参数。
2. JSON Schema 校验参数形状、类型和值域。
3. runtime 负责解析、校验、修复、拒绝、追问和执行。
4. 业务规则、权限、确认和审计负责判断“是否应该执行”。

第二轮重点补强三件事：第一，把 required、type、enum、pattern、range、additionalProperties 和 business validation 拆成可量化指标；第二，补一个 0 依赖 Python demo，直接演示 schema-valid、schema-invalid、可安全修复和业务规则失败的差别；第三，同步百科、题库、练习、术语表、项目和知识图谱，方便后续章节继续复用这些指标。

## 3.1 本章定位

上一章讲了 Function Calling 的输入输出协议：`messages`、`tools`、`tool_calls`、`tool_result`、`finish_reason` 和多轮 tool loop。

本章深入 `tools` 里最关键的部分：Tool Schema。

很多人第一次写工具调用时，会觉得 schema 只是“告诉模型这个函数有哪些参数”。这只说对了一小半。

在生产系统里，Tool Schema 同时承担四件事：

1. 告诉模型什么时候应该调用这个工具。
2. 告诉模型应该生成什么参数。
3. 告诉 runtime 如何校验参数。
4. 告诉治理系统这个工具的权限、风险、版本和边界。

所以 Tool Schema 不是普通注释，也不是 Swagger 文档的简化版，而是模型、runtime、工具实现、安全治理和评估系统之间的共同契约。

本章的核心观点是：

```text
Tool Schema 不是“参数格式说明”，而是工具调用系统的行为约束、校验依据和治理入口。
```

## 3.2 资料来源和可信边界

本章主要参考：

1. JSON Schema 官方规范。JSON Schema 用声明式方式描述 JSON 数据的结构、类型、必填字段和约束。
2. OpenAI tools / structured outputs 相关文档。工具参数常用 JSON Schema 子集描述，并可结合 strict 模式提升结构符合率。
3. Anthropic tool use 文档。工具包含 name、description、input_schema，模型基于工具描述生成结构化输入。
4. Google Gemini function calling 文档。函数声明包含名称、描述和参数 schema。
5. 企业 API 网关、OpenAPI、权限系统和 Agent runtime 的工程实践。

需要注意：不同模型平台对 JSON Schema 的支持不是完整一致的。有的平台只支持 JSON Schema 的一个子集，有的平台对 `oneOf`、`anyOf`、`patternProperties`、复杂嵌套、默认值、nullable 等支持有限。

因此工程上要区分两层：

1. 给模型看的 schema：尽量简单、明确、低歧义。
2. runtime 真实校验的 schema：可以更严格，还要叠加业务规则和权限规则。

不要以为“传了 schema 给模型”就等于“参数一定正确”。模型输出仍然必须由 runtime 校验。

## 3.3 一个最小 Tool Schema

先看一个最小例子：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询指定城市指定日期的天气预报。",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名，例如北京、上海、深圳"
        },
        "date": {
          "type": "string",
          "description": "日期，格式为 YYYY-MM-DD"
        }
      },
      "required": ["city", "date"]
    }
  }
}
```

这个 schema 至少包含三层信息：

1. 工具身份：`name=get_weather`。
2. 工具语义：`description=查询指定城市指定日期的天气预报`。
3. 参数结构：`city` 和 `date`，并且二者必填。

模型看到这个 schema 后，会学到：

1. 当用户问天气时，可以考虑调用 `get_weather`。
2. 调用时需要提供城市和日期。
3. 日期最好生成 `YYYY-MM-DD` 形式。

runtime 看到这个 schema 后，会做：

1. 检查工具名是否存在。
2. 检查 arguments 是否是对象。
3. 检查 `city` 和 `date` 是否存在。
4. 检查字段类型是否是字符串。
5. 进一步检查日期格式是否合法。

注意最后一点：JSON Schema 的 `description` 主要影响模型，不一定会被 validator 当作硬约束。要强制日期格式，需要写 `pattern` 或在业务校验层检查。

## 3.4 Tool Schema 的五个组成部分

生产级 Tool Schema 通常不止 name、description、parameters 三项，而是由五类信息组成。

第一类是模型可见信息：

1. 工具名。
2. 工具描述。
3. 参数名称。
4. 参数描述。
5. 参数类型和枚举值。

第二类是 runtime 校验信息：

1. 必填字段。
2. 类型约束。
3. 数值范围。
4. 字符串格式。
5. 数组长度。
6. 是否允许额外字段。

第三类是执行路由信息：

1. 工具实现位置。
2. endpoint 或 handler 名称。
3. 超时时间。
4. 重试策略。
5. 幂等策略。

第四类是安全治理信息：

1. 是否有副作用。
2. 是否需要用户确认。
3. 需要哪些权限。
4. 是否涉及敏感数据。
5. 是否允许在自动模式下执行。

第五类是观测和评估信息：

1. 工具版本。
2. owner。
3. 标签。
4. 成本估计。
5. SLA。
6. eval case 集合。

不同平台传给模型的 schema 可能只包含前两类，但企业内部 registry 通常需要保存完整元数据。

可以这样理解：

```text
模型 API 里的 tools 是 Tool Schema 的模型可见投影；企业 Tool Registry 里的 schema 才是完整治理对象。
```

## 3.5 JSON Schema 基础：type

JSON Schema 用 `type` 描述数据类型。

常见类型包括：

1. `object`：对象。
2. `array`：数组。
3. `string`：字符串。
4. `number`：数字，包括整数和小数。
5. `integer`：整数。
6. `boolean`：布尔值。
7. `null`：空值。

例如：

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "top_k": {"type": "integer"},
    "include_sources": {"type": "boolean"}
  },
  "required": ["query"]
}
```

这个 schema 表示：

1. arguments 必须是对象。
2. `query` 是字符串。
3. `top_k` 是整数。
4. `include_sources` 是布尔值。
5. `query` 必填。

类型约束能减少明显错误，但不能保证语义正确。

例如：

```json
{"query":"删除所有客户数据","top_k":5,"include_sources":true}
```

从类型上看合法，但业务上可能应该拒绝或转人工。

## 3.6 object：properties、required、additionalProperties

工具参数最常见的顶层结构是 object。

核心字段有三个：

1. `properties`：定义允许有哪些字段。
2. `required`：定义哪些字段必填。
3. `additionalProperties`：定义是否允许额外字段。

示例：

```json
{
  "type": "object",
  "properties": {
    "customer_id": {
      "type": "string",
      "description": "客户 ID"
    },
    "fields": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["name", "email", "phone", "risk_level"]
      }
    }
  },
  "required": ["customer_id"],
  "additionalProperties": false
}
```

`additionalProperties: false` 的作用是拒绝未知字段。

例如模型生成：

```json
{
  "customer_id": "C123",
  "fields": ["name", "email"],
  "include_password": true
}
```

其中 `include_password` 不在 schema 中，应该被拒绝。

这在安全上很重要。否则模型可能生成工具实现未预期的字段，下游代码如果错误地透传这些字段，可能触发越权或危险行为。

建议：生产级工具默认 `additionalProperties: false`，除非确实需要开放字典型参数。

## 3.7 string：description、enum、pattern、format

字符串字段看似简单，实际很容易出问题。

常见约束包括：

1. `description`：自然语言说明，主要帮助模型理解。
2. `enum`：限定候选值。
3. `pattern`：正则约束。
4. `format`：常见格式提示，例如 date、date-time、email、uri。
5. `minLength` / `maxLength`：长度约束。

例如：

```json
{
  "type": "object",
  "properties": {
    "date": {
      "type": "string",
      "description": "日期，格式为 YYYY-MM-DD",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
    },
    "priority": {
      "type": "string",
      "enum": ["low", "medium", "high"]
    },
    "email": {
      "type": "string",
      "format": "email"
    }
  }
}
```

注意 `description` 和 `pattern` 的区别：

1. `description` 告诉模型“最好这样写”。
2. `pattern` 告诉 validator“不符合就拒绝”。

如果只写：

```json
{"description":"日期，格式为 YYYY-MM-DD"}
```

模型仍可能输出：

```json
{"date":"明天"}
```

这不一定是模型坏，而是 schema 没有提供硬约束。

## 3.8 number 和 integer：范围约束

数值字段常用约束包括：

1. `minimum`：最小值。
2. `maximum`：最大值。
3. `exclusiveMinimum`：不含边界的最小值。
4. `exclusiveMaximum`：不含边界的最大值。
5. `multipleOf`：必须是某个数的倍数。

例如搜索工具：

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 20,
      "description": "返回结果数量，范围 1 到 20"
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

如果不限制 `top_k`，模型可能生成 100、1000，导致成本和延迟异常。

数值范围不只是格式问题，也影响系统资源：

1. `top_k` 影响检索成本。
2. `limit` 影响数据库扫描量。
3. `timeout` 影响服务稳定性。
4. `amount` 影响业务风险。
5. `num_candidates` 影响推理成本。

因此数值字段必须有上限，尤其是会影响成本、延迟和风险的参数。

## 3.9 array：items、minItems、maxItems、uniqueItems

数组字段常用于批量查询、字段选择、标签列表、候选项列表。

示例：

```json
{
  "type": "object",
  "properties": {
    "cities": {
      "type": "array",
      "description": "需要查询天气的城市列表，最多 5 个城市",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "maxItems": 5,
      "uniqueItems": true
    }
  },
  "required": ["cities"],
  "additionalProperties": false
}
```

数组必须限制长度。

否则用户问“帮我查全国所有城市天气”，模型可能生成几百个城市，runtime 如果照单执行，就会造成高延迟、高成本甚至触发下游限流。

对于批量工具，还要考虑：

1. 是否允许部分成功。
2. 单个 item 失败如何表示。
3. 总超时时间如何控制。
4. 是否需要拆分批次。
5. 是否允许模型自动扩大数组范围。

schema 只能限制数组形状，批量执行策略仍需 runtime 控制。

## 3.10 enum：降低歧义的利器

`enum` 是工具 schema 中非常有用的约束。

例如订单查询工具：

```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "description": "订单状态过滤条件",
      "enum": ["pending", "paid", "shipped", "delivered", "cancelled"]
    }
  }
}
```

如果不用 enum，模型可能生成：

1. `已支付`。
2. `付款完成`。
3. `PAID`。
4. `payment_success`。
5. `paid_orders`。

这些都可能让下游解析变复杂。

enum 的优点：

1. 限定模型输出空间。
2. 减少同义词漂移。
3. 方便后端直接路由。
4. 方便 eval 统计参数准确率。

enum 的风险是过度收窄。

如果业务状态会频繁变化，schema enum 没有同步更新，模型会无法表达新状态。因此 enum 要和工具版本管理配合。

## 3.11 oneOf、anyOf、allOf：慎用复杂组合

JSON Schema 支持组合约束，例如：

1. `oneOf`：必须满足其中一个 schema。
2. `anyOf`：满足一个或多个 schema。
3. `allOf`：必须同时满足多个 schema。

例如查询用户可以用 email 或 user_id：

```json
{
  "type": "object",
  "oneOf": [
    {
      "properties": {
        "user_id": {"type": "string"}
      },
      "required": ["user_id"]
    },
    {
      "properties": {
        "email": {"type": "string", "format": "email"}
      },
      "required": ["email"]
    }
  ]
}
```

这种写法在规范上合理，但对模型和 provider 支持不一定友好。

更稳的工具 schema 往往会简化为：

```json
{
  "type": "object",
  "properties": {
    "lookup_type": {
      "type": "string",
      "enum": ["user_id", "email"]
    },
    "lookup_value": {
      "type": "string",
      "description": "当 lookup_type=user_id 时填写用户 ID；当 lookup_type=email 时填写邮箱地址"
    }
  },
  "required": ["lookup_type", "lookup_value"],
  "additionalProperties": false
}
```

这不是说不能用 `oneOf`，而是生产中要考虑模型可生成性和 provider 兼容性。

面试中可以说：

```text
Tool schema 应该优先选择模型容易理解、provider 支持稳定、runtime 容易校验的表达。复杂 JSON Schema 组合可以在 runtime 内部使用，但不一定适合直接暴露给模型。
```

## 3.12 nullable 和默认值

很多 API 里会有可选字段和默认值。

例如：

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string"},
    "language": {
      "type": "string",
      "enum": ["zh", "en"],
      "description": "返回语言。若用户没有指定，默认使用 zh"
    }
  },
  "required": ["query"],
  "additionalProperties": false
}
```

这里 `language` 不是必填字段。用户没指定时，runtime 可以补默认值 `zh`。

注意：JSON Schema 里的 `default` 通常只是注解，不一定代表 validator 会自动填充，也不一定代表模型会遵守。

更稳的做法是：

1. description 写清默认行为。
2. runtime 在校验后补默认值。
3. trace 记录哪些字段由 runtime 补全。
4. 对高风险字段不要自动默认，要求用户明确确认。

nullable 也要谨慎。

例如：

```json
{"type": ["string", "null"]}
```

有些 provider 支持，有些不支持。为了兼容，可以改成可选字段，不传表示空。

## 3.13 description：给模型看的行为提示

description 是 Tool Schema 中最像 prompt 的部分。

它不只是解释字段，还会影响模型的工具选择和参数生成。

好的工具 description 应该包含：

1. 工具能做什么。
2. 什么时候应该使用。
3. 什么时候不应该使用。
4. 是否需要实时信息。
5. 是否有副作用。
6. 是否需要用户确认。
7. 输入参数的业务含义。

坏例子：

```json
{
  "name": "search",
  "description": "搜索"
}
```

好一些的例子：

```json
{
  "name": "search_internal_policy",
  "description": "查询公司内部制度、报销政策、假期规则和员工手册。仅用于公司内部政策问题，不用于查询互联网新闻、天气、客户隐私或个人账号信息。"
}
```

参数 description 也要明确。

坏例子：

```json
{"customer_id":{"type":"string","description":"id"}}
```

好一些的例子：

```json
{
  "customer_id": {
    "type": "string",
    "description": "客户唯一 ID，例如 CUST_12345。不要填写客户姓名、手机号或邮箱。"
  }
}
```

description 写得越清楚，模型越不容易把姓名、手机号、邮箱混到 `customer_id` 里。

## 3.14 工具名设计

工具名也会影响模型行为。

推荐工具名：

1. 使用动词加对象。
2. 语义具体。
3. 避免过短和过宽。
4. 避免多个工具名称相似但边界不清。
5. 保持稳定，避免频繁改名。

好名字：

1. `get_weather_forecast`。
2. `search_internal_policy`。
3. `create_calendar_event`。
4. `get_customer_order_history`。
5. `summarize_uploaded_document`。

坏名字：

1. `do_task`。
2. `search`。
3. `query`。
4. `tool1`。
5. `handle_request`。

如果有两个工具：

1. `search_customer`。
2. `search_customer_private_data`。

模型可能难以判断边界。更好的做法是明确拆分：

1. `search_customer_public_profile`。
2. `get_customer_sensitive_profile`。

并在敏感工具 description 里写明权限要求和适用场景。

## 3.15 参数命名设计

参数名也要稳定、具体、低歧义。

推荐：

1. 用业务含义命名，而不是技术缩写。
2. 避免 `data`、`input`、`value` 这类模糊字段。
3. 相似工具使用一致命名。
4. 对 ID、名称、邮箱等容易混淆的字段明确区分。

例如：

```json
{
  "customer_id": {"type": "string"},
  "customer_name": {"type": "string"},
  "customer_email": {"type": "string"}
}
```

比下面更好：

```json
{
  "id": {"type": "string"},
  "name": {"type": "string"},
  "email": {"type": "string"}
}
```

因为模型在长上下文、多工具场景下更容易保持业务语义一致。

参数命名还影响 eval。字段名越稳定，越容易统计参数准确率和错误类型。

## 3.16 参数约束不是安全边界的全部

schema 可以约束参数形状，但不能替代安全系统。

例如删除文件工具：

```json
{
  "name": "delete_file",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"}
    },
    "required": ["path"]
  }
}
```

即使 `path` 是合法字符串，也不代表可以删除。

runtime 还要检查：

1. 路径是否在允许目录内。
2. 用户是否有删除权限。
3. 文件是否受保护。
4. 是否需要用户确认。
5. 是否可以恢复。
6. 是否命中危险模式。

再比如转账工具：

1. schema 可以限制 `amount` 是正数。
2. 但不能判断收款人是否可信。
3. 不能判断这笔交易是否异常。
4. 不能判断用户是否刚刚明确授权。

因此要记住：

```text
Schema validation 只回答“参数格式是否合法”，不回答“这个动作是否应该执行”。
```

## 3.17 业务校验层

除了 JSON Schema，还需要业务校验。

例如会议创建工具：

```json
{
  "title": "周会",
  "start_time": "2026-05-30T10:00:00+08:00",
  "end_time": "2026-05-30T09:30:00+08:00",
  "attendees": ["alice@example.com"]
}
```

从类型上看没问题，但业务上结束时间早于开始时间，应该拒绝。

业务校验包括：

1. 时间区间是否合法。
2. ID 是否存在。
3. 用户是否可访问该对象。
4. 数量是否超过套餐限制。
5. 状态流转是否合法。
6. 是否违反风控规则。
7. 是否需要二次确认。

推荐校验顺序：

```text
JSON parse → JSON Schema validation → normalization → business validation → permission check → confirmation → execution
```

其中 normalization 是指把输入规范化，例如：

1. 去除首尾空格。
2. 城市名标准化。
3. 日期从自然语言解析成标准格式。
4. 枚举值大小写统一。
5. 默认值补全。

但 normalization 不能偷偷改变高风险含义。比如用户说“转 100”，模型生成“转 1000”，不能靠 normalization 修掉，应该拒绝并澄清。

## 3.18 Schema 对模型行为的影响

Tool Schema 会影响模型行为，主要体现在三方面。

第一，影响工具选择。

如果两个工具描述都很宽，模型可能选错。

例如：

```text
search_docs: 搜索文档
search_web: 搜索信息
```

用户问“公司报销政策是什么”，模型可能调用 web search，而不是内部文档搜索。

第二，影响参数生成。

字段描述越明确，模型越可能填对。

例如 `customer_id` 写明“不要填写姓名、手机号或邮箱”，能减少错误类型。

第三，影响是否澄清。

如果 required 字段缺失，模型可能：

1. 向用户追问。
2. 自己猜一个值。
3. 调用工具但参数为空。

schema 和 system 指令要共同告诉模型：

```text
缺少必需参数且无法从上下文可靠推断时，应该向用户澄清，不要编造。
```

## 3.19 什么时候让模型澄清，什么时候让 runtime 修复

参数不完整或不合法时，有两种处理方式：

1. 让模型向用户澄清。
2. runtime 自动修复或补全。

适合自动修复的情况：

1. 大小写归一化，例如 `PAID` → `paid`。
2. 明确日期解析，例如上下文中今天已知，`明天` → 标准日期。
3. 去除空格。
4. 用户明确提供的信息被模型格式化得稍微不标准。
5. 低风险默认值，例如 `language=zh`。

必须澄清的情况：

1. 用户没有提供关键必填信息。
2. 多个候选对象无法区分。
3. 动作有副作用。
4. 参数影响金额、权限、收件人、删除范围。
5. runtime 修复会改变业务含义。

例如用户说：

```text
帮我给张三转 500。
```

如果系统里有多个张三，不能自动选一个。应该澄清收款人。

## 3.20 strict 模式的价值和边界

一些平台支持 strict schema，让模型输出更严格符合 schema。

strict 模式的价值：

1. 减少 JSON parse 错误。
2. 减少字段缺失和类型错误。
3. 提升自动化执行成功率。
4. 降低参数修复成本。
5. 方便做工具调用 eval。

但 strict 模式不保证：

1. 工具一定选对。
2. 参数业务语义一定正确。
3. 动作一定安全。
4. 用户一定授权。
5. 工具结果一定可信。

例如 strict schema 可以让模型必须输出：

```json
{"recipient":"alice@example.com","amount":100}
```

但它不能判断 Alice 是不是正确收款人，也不能判断用户是否真的确认转账。

因此 strict 模式只能降低结构错误，不能替代 runtime 的业务安全。

## 3.21 Schema 版本管理

工具 schema 会演进。

常见变化包括：

1. 新增可选字段。
2. 新增必填字段。
3. 修改 enum 值。
4. 修改字段含义。
5. 删除字段。
6. 工具拆分或合并。

不是所有变化都兼容。

一般来说：

1. 新增可选字段通常向后兼容。
2. 新增必填字段通常不兼容。
3. 删除字段通常不兼容。
4. 修改字段语义高度危险。
5. enum 删除值可能破坏历史调用。

企业工具平台应该记录：

1. tool name。
2. version。
3. schema hash。
4. owner。
5. change log。
6. 上线时间。
7. 灰度范围。
8. 回滚策略。

trace 中也要记录当时使用的 schema version。否则几周后排查问题时，你不知道模型当时看到的是哪个工具描述和参数约束。

## 3.22 Schema 与 Tool Registry

Tool Registry 是工具注册中心。

它不只是存一个函数列表，而是存完整工具契约。

一个 registry item 可以长这样：

```json
{
  "name": "get_customer_order_history",
  "version": "1.3.0",
  "description": "查询客户订单历史。仅用于已授权客服场景。",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string"},
      "limit": {"type": "integer", "minimum": 1, "maximum": 50}
    },
    "required": ["customer_id"],
    "additionalProperties": false
  },
  "runtime": {
    "handler": "crm.get_order_history",
    "timeout_ms": 3000,
    "retry": 1,
    "side_effect": false
  },
  "security": {
    "required_permissions": ["crm:order:read"],
    "sensitive_data": true,
    "requires_user_confirmation": false
  },
  "observability": {
    "owner": "crm-platform",
    "cost_level": "medium",
    "tags": ["crm", "order", "read"]
  }
}
```

传给模型时，可能只投影出：

```json
{
  "name": "get_customer_order_history",
  "description": "查询客户订单历史。仅用于已授权客服场景。",
  "parameters": {"...": "..."}
}
```

执行时，runtime 再读取完整 registry 元数据做权限、超时、审计和路由。

这就是 schema 和 registry 的关系：

```text
Schema 描述工具输入契约，Registry 管理工具完整生命周期。
```

## 3.23 Schema 与 OpenAPI 的关系

很多企业已有 OpenAPI 或 Swagger 文档。能不能直接把 OpenAPI 转成 Tool Schema？

可以，但不能无脑转换。

OpenAPI 面向开发者和 HTTP API，Tool Schema 面向模型和 agent runtime。二者关注点不同。

直接转换会遇到问题：

1. API 太多，模型选择空间过大。
2. endpoint 名称对模型不友好。
3. 参数过细，模型难以正确填写。
4. 认证、租户、分页、内部字段暴露给模型。
5. 有副作用接口缺少确认语义。
6. 错误码和业务状态对模型不友好。

更好的做法是 API façade：

```text
Internal API → Tool wrapper → Clean Tool Schema → Model
```

Tool wrapper 负责：

1. 把复杂 API 聚合成模型友好的工具。
2. 隐藏内部认证和基础设施字段。
3. 提供清晰 description。
4. 做参数标准化和业务校验。
5. 把内部错误转成模型可理解的错误。
6. 加上权限、审计和确认流程。

所以面试中不要说“直接把公司所有 OpenAPI 暴露给模型”。这通常是危险答案。

## 3.24 Schema 设计案例：搜索工具

先看一个差的搜索工具：

```json
{
  "name": "search",
  "description": "搜索",
  "parameters": {
    "type": "object",
    "properties": {
      "q": {"type": "string"},
      "n": {"type": "integer"}
    }
  }
}
```

问题很多：

1. 工具名过宽。
2. description 过短。
3. 参数名不清楚。
4. `q` 没有描述。
5. `n` 没有范围。
6. 没有 required。
7. 允许额外字段。
8. 不知道搜索范围。

更好的版本：

```json
{
  "name": "search_internal_knowledge_base",
  "description": "搜索公司内部知识库，包括产品文档、技术方案、运维手册和常见问题。不要用于搜索互联网新闻、个人隐私或客户敏感数据。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "用户要查询的问题，应该保留关键实体和限定条件",
        "minLength": 2,
        "maxLength": 200
      },
      "top_k": {
        "type": "integer",
        "description": "返回结果数量，默认 5，最大 10",
        "minimum": 1,
        "maximum": 10
      },
      "source_type": {
        "type": "string",
        "description": "限定搜索来源。若用户未指定，使用 all",
        "enum": ["all", "product_doc", "runbook", "faq"]
      }
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

这个版本的优势：

1. 工具边界更清晰。
2. 参数名更明确。
3. `top_k` 有成本上限。
4. `source_type` 用 enum 降低歧义。
5. 禁止额外字段。
6. description 告诉模型不该用于哪些场景。

## 3.25 Schema 设计案例：发邮件工具

发邮件是有副作用工具，schema 设计要更谨慎。

示例：

```json
{
  "name": "draft_email",
  "description": "根据用户要求起草邮件草稿。该工具只创建草稿，不会发送邮件。",
  "parameters": {
    "type": "object",
    "properties": {
      "to": {
        "type": "array",
        "description": "收件人邮箱列表。必须来自用户明确提供或通讯录精确匹配结果。",
        "items": {"type": "string", "format": "email"},
        "minItems": 1,
        "maxItems": 10,
        "uniqueItems": true
      },
      "subject": {
        "type": "string",
        "minLength": 1,
        "maxLength": 120
      },
      "body": {
        "type": "string",
        "minLength": 1,
        "maxLength": 5000
      }
    },
    "required": ["to", "subject", "body"],
    "additionalProperties": false
  }
}
```

为什么这里用 `draft_email` 而不是 `send_email`？

因为草稿工具风险低很多。真正发送可以拆成另一个工具：

```json
{
  "name": "send_email_draft",
  "description": "发送已创建的邮件草稿。该操作有外部副作用，必须在用户确认草稿内容后调用。",
  "parameters": {
    "type": "object",
    "properties": {
      "draft_id": {"type": "string"},
      "confirmation_token": {
        "type": "string",
        "description": "用户确认后由 runtime 生成的确认令牌，模型不能自行编造"
      }
    },
    "required": ["draft_id", "confirmation_token"],
    "additionalProperties": false
  }
}
```

这个设计体现了一个重要原则：

```text
高风险动作可以拆成低风险准备工具 + 用户确认 + 高风险提交工具。
```

Schema 不只是描述参数，也能参与安全流程设计。

## 3.26 Schema 设计案例：数据库查询工具

很多人会想给模型一个 `run_sql` 工具。这很危险。

差的设计：

```json
{
  "name": "run_sql",
  "description": "执行 SQL 查询",
  "parameters": {
    "type": "object",
    "properties": {
      "sql": {"type": "string"}
    },
    "required": ["sql"]
  }
}
```

风险包括：

1. SQL 注入和越权。
2. 访问未授权表。
3. 大查询拖垮数据库。
4. 泄露敏感字段。
5. 模型生成错误 SQL。
6. 难以做细粒度审计。

更好的做法是提供受控查询工具：

```json
{
  "name": "query_sales_metrics",
  "description": "查询销售指标汇总数据。仅支持按日期范围、地区和产品线聚合，不返回客户个人信息。",
  "parameters": {
    "type": "object",
    "properties": {
      "start_date": {
        "type": "string",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
      },
      "end_date": {
        "type": "string",
        "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
      },
      "region": {
        "type": "string",
        "enum": ["all", "north", "east", "south", "west"]
      },
      "metric": {
        "type": "string",
        "enum": ["revenue", "order_count", "average_order_value"]
      }
    },
    "required": ["start_date", "end_date", "metric"],
    "additionalProperties": false
  }
}
```

runtime 内部再把这些受控参数转成安全 SQL。

原则是：

```text
不要把底层通用执行能力直接暴露给模型，要暴露业务语义明确、权限边界清楚、参数可控的工具。
```

## 3.27 Schema Eval：怎么评估 schema 好不好

Tool Schema 不是写完就结束，要评估。

常见指标：

1. 工具选择准确率：该调用时是否调用，不该调用时是否不调用。
2. 工具名准确率：是否选对工具。
3. 参数完整率：required 字段是否都生成。
4. 参数类型正确率：类型是否符合 schema。
5. 参数语义正确率：值是否符合用户意图。
6. 参数修复率：有多少调用需要 runtime 修复。
7. 澄清正确率：缺信息时是否向用户追问。
8. 越权拦截率：危险参数是否被拦住。
9. 执行成功率：校验通过并成功执行的比例。

评估集应该包含：

1. 正常调用样例。
2. 不该调用工具的样例。
3. 缺必填参数样例。
4. 多工具易混淆样例。
5. 参数边界值样例。
6. prompt injection 样例。
7. 高风险动作样例。
8. 多语言和口语化表达样例。

一个 schema 如果在 demo 问题上表现很好，但在“不该调用”和“缺信息澄清”上表现差，仍然不适合上线。

## 3.28 常见 Schema 设计错误

### 3.28.1 工具描述过宽

例如 `search: 搜索任何信息`。

后果：模型过度调用工具，甚至把不需要实时信息的问题也交给搜索。

修复：写清适用和不适用场景。

### 3.28.2 参数名过短

例如 `id`、`q`、`n`。

后果：模型在多工具上下文中容易填错。

修复：用 `customer_id`、`query`、`top_k` 等明确名称。

### 3.28.3 只写 description，不写硬约束

例如 description 写“最多 10 个”，但没有 `maximum` 或 `maxItems`。

后果：模型仍可能生成超大值。

修复：把关键限制写进 JSON Schema 约束，并在 runtime 校验。

### 3.28.4 允许额外字段

没有设置 `additionalProperties: false`。

后果：模型生成未知字段，下游误处理。

修复：默认禁止额外字段。

### 3.28.5 暴露底层危险工具

例如 `run_shell`、`run_sql`、`delete_file` 直接暴露。

后果：安全风险极高。

修复：用受控业务工具包装底层能力，加权限和确认。

### 3.28.6 required 过多

所有字段都 required。

后果：模型在用户没提供信息时容易编造。

修复：只把真正必须的信息设为 required，缺信息时要求澄清。

### 3.28.7 enum 不维护

业务新增状态，但 schema enum 没更新。

后果：模型无法生成合法新状态。

修复：schema 版本管理和发布流程。

### 3.28.8 把 schema 当安全系统

以为 schema 校验通过就能执行。

后果：越权、有副作用、误操作。

修复：schema validation 后继续做业务校验、权限检查、确认和审计。

## 3.29 Schema 约束审计指标与最小 demo

Tool Schema 不能只靠“看起来写得不错”来判断质量。生产系统至少要把 schema 约束拆成可统计指标。

设第 `i` 条工具调用样本为：

```math
s_i=(t_i,a_i,V_i,B_i,r_i)
```

其中 `t_i` 是工具名，`a_i` 是模型生成的 arguments，`V_i` 是该工具的 JSON Schema validator，`B_i` 是业务校验器，`r_i` 是可选的安全修复或 normalization 结果。

最常用的 schema 指标包括：

```math
R_{\mathrm{schema}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[V_i(a_i)=1]
```

其中 `R_schema` 是 schema 合法率，衡量 arguments 是否整体通过 JSON Schema 校验。

```math
R_{\mathrm{req}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[e_i^{\mathrm{req}}=0]
```

其中 `e_i^req` 表示第 `i` 条样本是否存在 required 字段缺失错误。

```math
R_{\mathrm{type}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[e_i^{\mathrm{type}}=0]
```

其中 `R_type` 衡量类型错误是否被压低。

```math
R_{\mathrm{enum}}=\frac{1}{N_{\mathrm{enum}}}\sum_{i=1}^{N_{\mathrm{enum}}}\mathbf{1}[e_i^{\mathrm{enum}}=0]
```

其中 `N_enum` 是包含 enum 约束且模型确实生成了对应字段的样本数。

```math
R_{\mathrm{pat}}=\frac{1}{N_{\mathrm{pat}}}\sum_{i=1}^{N_{\mathrm{pat}}}\mathbf{1}[e_i^{\mathrm{pat}}=0]
```

其中 `R_pat` 衡量日期、邮箱、ID 等字符串模式是否满足。

```math
R_{\mathrm{range}}=\frac{1}{N_{\mathrm{range}}}\sum_{i=1}^{N_{\mathrm{range}}}\mathbf{1}[e_i^{\mathrm{range}}=0]
```

其中 `R_range` 衡量 `top_k`、`limit`、`amount`、`timeout` 等数值范围是否可靠。

```math
B_{\mathrm{extra}}=\frac{1}{N_{\mathrm{extra}}}\sum_{i=1}^{N_{\mathrm{extra}}}\mathbf{1}[\mathrm{extra}_i \ \mathrm{blocked}]
```

其中 `B_extra` 衡量 `additionalProperties: false` 是否真的挡住了未知字段。

```math
R_{\mathrm{biz}}=\frac{1}{N_{\mathrm{valid}}}\sum_{i=1}^{N_{\mathrm{valid}}}\mathbf{1}[B_i(a_i)=1]
```

其中 `R_biz` 只在 schema 已通过的样本上统计业务规则通过率，用来提醒读者：schema valid 之后仍可能业务不合法。

```math
R_{\mathrm{repair}}=\frac{1}{N_{\mathrm{repair}}}\sum_{i=1}^{N_{\mathrm{repair}}}\mathbf{1}[V_i(r_i)=1 \wedge B_i(r_i)=1]
```

其中 `R_repair` 只统计允许安全修复的样本，例如大小写归一化、`P1` 到 `high` 的低风险映射、明确上下文中的“明天”到标准日期。高风险金额、收件人、删除范围不能为了提高 repair rate 而自动改。

最后可以定义一个门禁：

```math
G_{\mathrm{schema}}=\mathbf{1}[
R_{\mathrm{schema}}\ge \tau_s
\wedge R_{\mathrm{req}}\ge \tau_q
\wedge R_{\mathrm{type}}\ge \tau_t
\wedge R_{\mathrm{enum}}\ge \tau_e
\wedge R_{\mathrm{pat}}\ge \tau_p
\wedge R_{\mathrm{range}}\ge \tau_r
\wedge B_{\mathrm{extra}}\ge \tau_x
\wedge R_{\mathrm{biz}}\ge \tau_b]
```

其中各个 `tau` 是上线阈值。真实系统里，高风险工具的阈值通常要比只读搜索工具更严格。

下面是一个 0 依赖 demo。它只实现 JSON Schema 的一个教学子集：`type`、`required`、`properties`、`enum`、`minimum`、`maximum`、`pattern`、`additionalProperties`、数组长度和数组元素类型。真实系统应使用成熟 validator，但这个 demo 足够说明指标怎么算。

```python
import re
from copy import deepcopy


SCHEMAS = {
    "search_docs": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 2, "maxLength": 200},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "source_type": {"type": "string", "enum": ["all", "product_doc", "runbook", "faq"]},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
            "units": {"type": "string", "enum": ["metric", "imperial"]},
        },
        "required": ["city", "date"],
        "additionalProperties": False,
    },
    "create_ticket": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 120},
            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
            "category": {"type": "string", "enum": ["bug", "billing", "security"]},
            "requester_email": {"type": "string", "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
            "security_reviewed": {"type": "boolean"},
        },
        "required": ["title", "priority", "category", "requester_email"],
        "additionalProperties": False,
    },
}


CALLS = [
    {
        "id": "search_ok",
        "tool": "search_docs",
        "arguments": {"query": "报销政策", "top_k": 5, "source_type": "faq"},
    },
    {
        "id": "missing_required",
        "tool": "search_docs",
        "arguments": {"top_k": 3, "source_type": "faq"},
    },
    {
        "id": "wrong_type",
        "tool": "get_weather",
        "arguments": {"city": 100, "date": "2026-06-11"},
    },
    {
        "id": "bad_enum_repairable",
        "tool": "create_ticket",
        "arguments": {
            "title": "登录失败",
            "priority": "P1",
            "category": "bug",
            "requester_email": "alice@example.com",
        },
    },
    {
        "id": "extra_field_blocked",
        "tool": "get_weather",
        "arguments": {"city": "北京", "date": "2026-06-11", "include_private_calendar": True},
    },
    {
        "id": "bad_pattern_repairable",
        "tool": "get_weather",
        "arguments": {"city": "北京", "date": "tomorrow"},
    },
    {
        "id": "range_too_large",
        "tool": "search_docs",
        "arguments": {"query": "SLA", "top_k": 50, "source_type": "all"},
    },
    {
        "id": "business_invalid",
        "tool": "create_ticket",
        "arguments": {
            "title": "导出所有客户数据",
            "priority": "high",
            "category": "security",
            "requester_email": "bob@example.com",
            "security_reviewed": False,
        },
    },
]


def type_matches(value, expected):
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate(value, schema):
    errors = []
    expected_type = schema.get("type")
    if expected_type and not type_matches(value, expected_type):
        return ["type"]

    if expected_type == "object":
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append("required")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append("additional")
        for name, child_schema in properties.items():
            if name in value:
                errors.extend(validate(value[name], child_schema))

    if expected_type == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append("array_length")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append("array_length")
        if schema.get("uniqueItems") and len(set(map(repr, value))) != len(value):
            errors.append("array_unique")
        if "items" in schema:
            for item in value:
                errors.extend(validate(item, schema["items"]))

    if expected_type == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append("length")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append("length")
        if "enum" in schema and value not in schema["enum"]:
            errors.append("enum")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            errors.append("pattern")

    if expected_type in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append("range")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append("range")

    return errors


def has_present_constraint(value, schema, constraint):
    if not isinstance(value, dict) or schema.get("type") != "object":
        return constraint in schema
    for name, child_schema in schema.get("properties", {}).items():
        if name in value and has_present_constraint(value[name], child_schema, constraint):
            return True
    return False


def has_range_constraint(value, schema):
    if not isinstance(value, dict) or schema.get("type") != "object":
        return "minimum" in schema or "maximum" in schema
    for name, child_schema in schema.get("properties", {}).items():
        if name in value and has_range_constraint(value[name], child_schema):
            return True
    return False


def has_extra_field(value, schema):
    if schema.get("type") != "object" or not isinstance(value, dict):
        return False
    properties = schema.get("properties", {})
    return any(name not in properties for name in value)


def business_check(call):
    if call["tool"] == "create_ticket":
        args = call["arguments"]
        if args.get("category") == "security" and args.get("priority") == "high":
            if not args.get("security_reviewed"):
                return False
    return True


def repair(call):
    repaired = deepcopy(call)
    args = repaired["arguments"]
    changed = False
    if repaired["tool"] == "create_ticket" and args.get("priority") == "P1":
        args["priority"] = "high"
        changed = True
    if repaired["tool"] == "get_weather" and args.get("date") == "tomorrow":
        args["date"] = "2026-06-11"
        changed = True
    return repaired if changed else None


def rate(values):
    return round(sum(values) / len(values), 3) if values else 1.0


reports = []
for call in CALLS:
    schema = SCHEMAS[call["tool"]]
    errors = sorted(set(validate(call["arguments"], schema)))
    raw_valid = not errors
    biz_ok = raw_valid and business_check(call)
    fixed = repair(call)
    repair_ok = False
    if fixed:
        fixed_errors = validate(fixed["arguments"], schema)
        repair_ok = not fixed_errors and business_check(fixed)
    reports.append({
        "id": call["id"],
        "errors": errors,
        "schema_valid": raw_valid,
        "business_ok": biz_ok,
        "repair_attempted": fixed is not None,
        "repair_ok": repair_ok,
        "has_enum": has_present_constraint(call["arguments"], schema, "enum"),
        "has_pattern": has_present_constraint(call["arguments"], schema, "pattern"),
        "has_range": has_range_constraint(call["arguments"], schema),
        "has_extra": has_extra_field(call["arguments"], schema),
    })

metrics = {
    "schema_valid_rate": rate([r["schema_valid"] for r in reports]),
    "required_field_pass_rate": rate(["required" not in r["errors"] for r in reports]),
    "type_valid_rate": rate(["type" not in r["errors"] for r in reports]),
    "enum_valid_rate": rate(["enum" not in r["errors"] for r in reports if r["has_enum"]]),
    "pattern_valid_rate": rate(["pattern" not in r["errors"] for r in reports if r["has_pattern"]]),
    "range_valid_rate": rate(["range" not in r["errors"] for r in reports if r["has_range"]]),
    "additional_properties_block_rate": rate(["additional" in r["errors"] for r in reports if r["has_extra"]]),
    "business_rule_pass_rate": rate([r["business_ok"] for r in reports if r["schema_valid"]]),
    "repair_success_rate": rate([r["repair_ok"] for r in reports if r["repair_attempted"]]),
}

thresholds = {
    "schema_valid_rate": 0.95,
    "required_field_pass_rate": 0.98,
    "type_valid_rate": 0.98,
    "enum_valid_rate": 0.98,
    "pattern_valid_rate": 0.98,
    "range_valid_rate": 0.98,
    "additional_properties_block_rate": 1.0,
    "business_rule_pass_rate": 0.99,
    "repair_success_rate": 0.90,
}
failed_gates = [name for name, threshold in thresholds.items() if metrics[name] < threshold]

print("metrics=", metrics)
print("schema_failures=", {r["id"]: r["errors"] for r in reports if r["errors"]})
print("business_failures=", [r["id"] for r in reports if r["schema_valid"] and not r["business_ok"]])
print("repaired_calls=", [r["id"] for r in reports if r["repair_ok"]])
print("failed_gates=", failed_gates)
print("schema_gate_pass=", not failed_gates)
```

这段 demo 的预期输出类似：

```text
metrics= {'schema_valid_rate': 0.25, 'required_field_pass_rate': 0.875, 'type_valid_rate': 0.875, 'enum_valid_rate': 0.8, 'pattern_valid_rate': 0.8, 'range_valid_rate': 0.667, 'additional_properties_block_rate': 1.0, 'business_rule_pass_rate': 0.5, 'repair_success_rate': 1.0}
schema_failures= {'missing_required': ['required'], 'wrong_type': ['type'], 'bad_enum_repairable': ['enum'], 'extra_field_blocked': ['additional'], 'bad_pattern_repairable': ['pattern'], 'range_too_large': ['range']}
business_failures= ['business_invalid']
repaired_calls= ['bad_enum_repairable', 'bad_pattern_repairable']
failed_gates= ['schema_valid_rate', 'required_field_pass_rate', 'type_valid_rate', 'enum_valid_rate', 'pattern_valid_rate', 'range_valid_rate', 'business_rule_pass_rate']
schema_gate_pass= False
```

这个结果说明四件事：

1. `schema_valid_rate` 很低，证明模型参数生成或 schema 约束仍有明显问题。
2. `additional_properties_block_rate=1.0` 是好事，说明额外字段被挡住了。
3. `business_invalid` 通过了 schema，但业务规则不允许执行，证明 schema validation 不能替代业务校验。
4. `bad_enum_repairable` 和 `bad_pattern_repairable` 可以安全修复，但这类修复必须白名单化，不能把高风险参数也自动改掉。

## 3.30 面试题：如何设计一个好的 Tool Schema

面试官可能问：

```text
怎么判断一个 function calling 的 tool schema 设计得好不好？
```

可以从六个维度回答。

第一，语义清晰：

1. 工具名具体。
2. description 写清适用和不适用场景。
3. 参数名表达业务含义。

第二，约束充分：

1. required 合理。
2. 类型明确。
3. enum、范围、长度、pattern 写清。
4. 默认禁止额外字段。

第三，模型友好：

1. schema 不过度复杂。
2. 避免难以生成的组合约束。
3. 缺信息时有澄清策略。
4. description 帮助模型区分相似工具。

第四，runtime 可校验：

1. JSON Schema 校验。
2. 业务规则校验。
3. 参数 normalization。
4. 错误可回填。

第五，安全可治理：

1. 权限边界明确。
2. 有副作用工具标记清楚。
3. 高风险动作需要确认。
4. 敏感字段脱敏和审计。

第六，可演进可评估：

1. 有版本号。
2. trace 记录 schema version。
3. 有 eval case。
4. 能统计选择准确率和参数正确率。

一句话答案：

```text
好的 Tool Schema 应该让模型容易选对工具、生成正确参数，让 runtime 能严格校验和安全执行，让平台能版本管理、审计和评估。
```

## 3.31 小练习

### 练习 1：改进搜索工具

下面的 schema 有哪些问题？

```json
{
  "name": "search",
  "description": "搜索资料",
  "parameters": {
    "type": "object",
    "properties": {
      "q": {"type": "string"},
      "limit": {"type": "integer"}
    }
  }
}
```

参考答案：

1. 工具名过宽。
2. description 没有说明搜索范围。
3. `q` 参数名不够清晰。
4. `limit` 没有范围。
5. 没有 required。
6. 没有 `additionalProperties: false`。
7. 没有说明不适用场景。

### 练习 2：判断是否能自动修复

用户说：“查一下明天北京天气。”模型生成：

```json
{"city":"北京","date":"明天"}
```

schema 要求 `date` 是 `YYYY-MM-DD`。

如果系统知道当前日期是 2026-05-29，能否自动修复？

参考答案：可以。因为用户明确说了“明天”，runtime 可以把它规范化为 `2026-05-30`，并在 trace 中记录 normalization。

### 练习 3：判断是否需要澄清

用户说：“给张三发一下合同。”通讯录里有三个张三。模型生成其中一个邮箱。

能否直接执行？

参考答案：不能。收件人有歧义，而且发合同有外部副作用和隐私风险，必须向用户澄清并确认。

### 练习 4：设计受控数据库工具

为什么不建议直接暴露 `run_sql(sql: string)`？

参考答案：因为它把底层通用执行能力交给模型，存在越权、大查询、敏感字段泄露、错误 SQL 和审计困难等风险。更好的方式是设计业务语义明确的受控查询工具，用 enum、日期范围、权限和聚合维度约束查询空间。

## 3.32 本章小结

本章讲了 Tool Schema、JSON Schema 与参数约束。

你需要掌握：

1. Tool Schema 是模型、runtime、工具实现和治理系统之间的共同契约。
2. JSON Schema 可以描述类型、必填字段、枚举、范围、数组长度、字符串格式等约束。
3. `description` 主要影响模型理解，硬约束必须写进 schema 或业务校验。
4. 生产级工具默认应禁止额外字段。
5. enum 能显著降低参数歧义，但需要版本维护。
6. 复杂组合 schema 要谨慎暴露给模型。
7. schema validation 不能替代业务校验、权限检查和用户确认。
8. 有副作用工具要在 schema 和 registry 中显式标记。
9. Tool Registry 管理的是完整工具生命周期，不只是参数 schema。
10. 不要无脑把 OpenAPI 或底层执行能力直接暴露给模型。
11. Schema 质量需要通过 eval 度量，而不是凭感觉判断。

如果只记一句话：

```text
Tool Schema 设计得越清楚，模型越容易生成可执行的调用；runtime 校验得越严格，系统越能把模型的不确定性控制在安全边界内。
```

下一章会讲 Tool Choice、Parallel Tool Calls 与强制工具调用，重点解释模型什么时候能自由选择工具，什么时候应该强制调用、禁止调用或并行调用。
