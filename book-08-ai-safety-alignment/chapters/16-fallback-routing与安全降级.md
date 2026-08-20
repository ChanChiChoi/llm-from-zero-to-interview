# 第十六章 Fallback Routing：失败时降低能力，不扩大风险

主模型超时、provider 限流、上下文过长、工具协议不兼容、策略服务不可用时，系统通常不能只返回一个错误。它可能需要重试、切换同权限后端、改为只读分析、排队等待、转人工或明确拒绝。问题在于，一次请求在失败之前可能已经完成了工具调用、写入了外部系统或生成了部分 artifact。此时把原始 prompt 原样转发给另一个模型，可能重复提交副作用，也可能把敏感数据送到不具备同等权限和协议能力的后端。

Fallback routing 的目标不是让每一次请求都“看起来成功”，而是让系统在能力下降时仍保持安全不变量：权限不能扩大，已提交的动作不能重复，未知状态不能伪装成失败或成功，敏感数据不能因为换后端而越过范围，用户要知道哪些事情没有完成。

本章把 fallback 视为一个有状态的恢复系统。路由器首先读取请求和外部动作的状态，再判断错误类型、权限、风险、预算、数据范围和候选后端能力，最后选择同路径重试、兼容后端、受限工具、只读信息、人工接管或拒绝。模型只是候选执行组件，不能决定自己的 fallback 权限。

## 16.1 “失败”不是一种状态

### 16.1.1 小白视角：付款超时和模型超时完全不同

普通问答的模型请求超时，通常可以在没有外部副作用的前提下重试；付款工具请求超时却可能意味着付款已经提交，只是客户端没有收到回执。两者都可能显示为 `timeout`，恢复方式却相反：前者可以选择兼容后端，后者必须先查询外部状态。

同样，策略拒绝、上下文过长、工具参数错误、provider 限流和权限撤销也不能共用“再试一次”。安全降级的第一步不是选择备用模型，而是回答：

1. 失败发生在哪一层。
2. 外部状态是否可能已经改变。
3. 当前授权是否仍然有效。
4. 备用路径是否支持同样的数据和协议。
5. 重新执行是否会重复动作。

### 16.1.2 专家视角：路由函数读取状态账本

可以把一个待恢复请求写成：

~~~math
q=(i,s,a,h,r,b,v),
~~~

其中 `i` 是 request/action ID，`s` 是当前状态，`a` 是动作与参数摘要，`h` 是错误和历史事件，`r` 是风险与权限信息，`b` 是剩余预算，`v` 是模型、工具、政策和资源版本。路由器计算：

~~~math
p=\rho(q),
~~~

`p` 不是“哪个模型更强”的单一答案，而是一个包含下一路径、能力范围、用户可见状态、重试预算和审计原因的决定。

### 16.1.3 状态机

一个包含外部动作的请求可以使用以下状态：

~~~text
created -> not_started -> running -> waiting_tool -> awaiting_confirmation -> committed
   |         |              |                 |
   |         |              |                 +--> failed / cancelled
   |         |              +--> unknown
   |         +--> failed / cancelled / unknown
   +--> cancelled

unknown -> query_external_state -> committed / failed / human_review
~~~

`unknown` 不是 `failed`。它表示客户端不知道外部系统是否完成，必须先查询、对账或交给人工。只有当外部契约明确说明没有提交，或者状态查询返回确定失败，系统才可以进入安全重试路径。`committed` 也不等于“模型输出了工具参数”：它表示执行器或外部系统已经确认提交；如果只有模型生成了一个 tool call，而执行器没有接收或提交，该动作仍不能标成 `committed`。

这几个状态对应不同的恢复动作：`not_started` 可以在授权仍有效且预算允许时开始一次执行；`running` 需要判断执行器是否仍持有租约；`unknown` 先查询，查询接口不可用时保持未知；`failed` 只有在失败语义明确且没有外部副作用时才可重试；`committed` 只能读取回执、查询结果或等待后续处理，不能再次提交。状态名称不是装饰性的日志字段，而是决定系统能否产生新副作用的事实依据。图中的 `failed` 只表示外部契约已经给出确定失败；连接中断而无法判断提交结果的情况仍应落到 `unknown`。

## 16.2 安全不变量：降级可以少做，不能越界

### 16.2.1 权限不扩大

如果主路径只有只读权限，fallback 不能切换到拥有写权限的后端；如果当前任务禁止外发，备用模型也不能因为支持外发工具而获得外发权限。可以把权限范围写成：

~~~math
\mathrm{Scope}_{t+1}\subseteq\mathrm{Scope}_{t},
\qquad
\mathrm{Privilege}_{t+1}\leq\mathrm{Privilege}_{t}.
~~~

这描述的是降级路径的安全方向，不要求每次都减少计算能力。一个同权限的副本可以保持相同 scope；一个只读路径则收窄 action。

### 16.2.2 副作用不重复

已经提交的付款、邮件、工单、文件写入或发布不能因为模型超时而自动重复。所有可能改变外部状态的动作都应带有稳定的 `action_id` 和幂等键，备用路径必须读取这个状态，而不是重新构造一个“看起来相同”的动作。

### 16.2.3 未知状态不伪装

如果外部系统的状态未知，用户可见结果应当是“待确认”或“正在查询”，不能说“失败”让用户重新付款，也不能说“成功”让用户以为已经得到回执。自然语言的确定性不能覆盖状态账本的不确定性。

### 16.2.4 证据和版本不丢失

fallback 需要携带原始请求的 policy revision、model revision、tool call ID、artifact hash、权限范围、已读证据和未完成副作用。只传 prompt 而不传状态，会让备用路径从一个错误的中间点重新开始。

### 16.2.5 用户可知但不过度泄露

用户需要知道能力是否下降、动作是否执行、证据是否完整以及下一步是什么；不必看到内部阈值、provider 拓扑或敏感审计字段。解释的粒度要同时满足可行动性和信息最小化。

## 16.3 失败分类与恢复动作

### 16.3.1 传输和容量错误

超时、连接断开、429、worker 重启和暂时无容量，可能适合短暂重试、排队或切换同权限副本，但前提是没有未知副作用。要区分请求是否到达 provider、provider 是否开始生成、工具是否已经调用。

### 16.3.2 协议和 artifact 错误

tokenizer、chat template、structured output、tool schema、流式事件、媒体格式或 context limit 不兼容，不能通过无限重试解决。路由器应选择已声明兼容的 adapter；如果没有兼容后端，就改为只读解释、草稿或人工路径。

### 16.3.3 模型质量错误

引用缺失、结构化输出解析失败、verifier 不通过、事实证据不足和安全分类不确定，通常需要重新生成、检索增强、第二模型或人工检查。它们不等于 provider 不可用，切换模型前要决定是否可以把同一份敏感上下文交给新后端。

### 16.3.4 工具错误

工具参数错误可能在提交前失败，也可能在外部系统已经执行后返回错误。工具超时尤其不能直接归为“未执行”。工具契约应提供查询接口、幂等键、明确的提交语义和状态机。

### 16.3.5 策略和权限错误

安全拒绝、权限撤销、租户范围不匹配和审批过期，不允许通过换模型绕过。正确路径是解释、请求合法的补充信息、转人工或拒绝。策略服务不可用则按动作风险选择受限继续或停止，不能把“查不到权限”当作“有权限”。

### 16.3.6 上下文超限

上下文过长可以改为检索、分块、摘要、异步处理或只读输出，但不能静默截断可能决定安全的证据。改写上下文后要记录丢失的范围、摘要版本和新的证据完整性状态。

### 16.3.7 错误映射表

| 错误 | 可能已有副作用 | 默认恢复 | 不应做的事 |
| --- | --- | --- | --- |
| 主模型超时且无工具调用 | 否 | 有限重试或同权限副本 | 无限重试 |
| provider 限流 | 通常否，但需确认提交状态 | 排队或同权限副本 | 提高权限换后端 |
| tokenizer/协议不兼容 | 否 | 兼容 adapter 或信息路径 | 把错误当容量不足 |
| policy service 不可用 | 未知 | 高风险动作停止 | 直接放行 |
| 工具已提交但回执丢失 | 是 | 查询 action 状态 | 盲目重放 |
| 权限在等待中撤销 | 可能 | 重新授权或停止 | 复用旧 token |
| 用户取消 | 取决于动作 | 停止新副作用并查询状态 | 把取消当回滚完成 |

## 16.4 路由架构：先状态，后模型

一个生产路由器可以拆成以下组件：

~~~text
request + action ledger + policy context
                 |
                 v
          state resolver
                 |
       error classifier / risk check
                 |
         fallback policy engine
          /       |        \
    retry   adapter   restricted / human
          \       |        /
              executor
                 |
        external state + audit trace
~~~

### 16.4.1 state resolver

它先从本地事件、工具查询接口、幂等记录和队列状态中判断动作属于 `not_started`、`running`、`committed`、`failed` 还是 `unknown`。如果不能确认，就保守地保留未知，而不是猜测。

### 16.4.2 error classifier

错误分类器需要读取 HTTP 状态、provider 事件、tool call ID、执行器回执、超时位置和取消原因。单一的异常字符串不够，因为“timeout”可能表示连接前失败、生成中断或外部动作回执丢失。

### 16.4.3 fallback policy engine

路由策略根据错误、风险、预算、权限和后端能力决定下一路径：

~~~math
\mathrm{route}
=\rho(\mathrm{error},\mathrm{risk},\mathrm{state},\mathrm{budget},\mathrm{capability}).
~~~

它要同时返回 `route`、`allowed_scope`、`retry_budget`、`user_message` 和 `reason_code`，方便后续执行和审计。

### 16.4.4 adapter 与 executor

adapter 负责将消息、token、工具 schema、流式事件和媒体输入转换成目标后端支持的形式；executor 负责再次检查权限、参数、资源版本和幂等性。adapter 不能把策略拒绝转换成普通文本让 executor 猜测，executor 也不能因为 adapter 成功解析就认为动作被授权。

## 16.5 Fallback 路径的层级

### 16.5.1 同路径有限重试

适用于无副作用的 transient error。重试次数、总时间和退避策略要有上限：

~~~math
T_{\mathrm{retry\ budget}}
=\sum_{j=1}^{n}(t_j^{\mathrm{backoff}}+t_j^{\mathrm{attempt}})
\leq B_T.
~~~

指数退避可以减少高峰争抢，但不能把用户等待时间无限延长。超过预算后进入排队、备用路径或明确失败。

### 16.5.2 同权限兼容后端

主 provider 限流时，可以切换到已批准的同等数据区域、工具协议和权限范围的后端。备用模型不能默认接收更多数据，也不能自动获得主模型没有的写权限。切换要记录 provider、model、processor 和 policy revision。

### 16.5.3 受限后端

当完整工具链不可用时，路由到只读模型、检索系统、规则引擎或结构化草稿。受限路径必须明确删掉了哪些能力：不能调用的工具、未读取的模态、缩短的上下文、未验证的字段和不支持的外部动作。

### 16.5.4 人工和异步路径

高风险或未知状态可以进入人工队列，长上下文和低时效任务可以进入异步作业。队列消息要携带权限快照、资源版本、action ID、过期时间和用户可见状态，人工处理时仍需重新授权。

### 16.5.5 明确拒绝

没有兼容协议、权限失效、证据缺失或高风险状态无法确认时，拒绝是正确的能力降级。拒绝应说明未完成的动作和可行的下一步，不应伪装成低质量成功。

## 16.6 重试、预算与熔断

### 16.6.1 为什么无限重试会放大风险

每次重试都可能增加费用、延迟、日志复制、上下文暴露和副作用机会。多个组件各自重试时，总尝试次数可能呈乘法增长。假设路由器重试 `n_r` 次、模型客户端重试 `n_m` 次、工具客户端重试 `n_t` 次，最坏尝试数近似为：

~~~math
N_{\mathrm{attempts}}
\leq n_r n_m n_t.
~~~

这只是上界，但它提醒我们必须统一预算和责任，不能每一层都以为自己只重试了一次。

### 16.6.2 预算维度

预算至少包括：

1. 请求总时延和 p99 约束。
2. 模型与工具调用次数。
3. token、GPU 和 provider 费用。
4. 人工队列容量。
5. 外部动作次数和数据范围。
6. 敏感内容被发送到后端的次数。

高风险任务可以拥有更少的自动重试次数，而不是更多的探索预算。

### 16.6.3 熔断与半开

当某个后端连续失败，熔断器可以暂时停止发送请求并转到已批准路径。半开探测只允许低风险、无副作用样本验证恢复；不能用真实支付或生产发布来探测 provider 是否恢复。熔断状态、探测样本和恢复时间都要进入 trace。

## 16.7 工具半成功、幂等与状态恢复

### 16.7.1 状态账本

每个外部动作至少需要：

~~~json
{
  "action_id": "action-synthetic-17",
  "idempotency_key": "idem-synthetic-17",
  "tool": "payment_stub",
  "resource": "account:synthetic-a",
  "status": "unknown",
  "request_digest": "sha256:synthetic-request",
  "policy_revision": "policy-4",
  "principal_revision": "principal-9",
  "created_at": "2026-08-11T10:00:00Z"
}
~~~

`status=unknown` 时，fallback router 的下一动作是查询，而不是生成新的付款请求。状态查询要使用稳定的 action ID 和幂等键，不能依赖模型记忆。

### 16.7.2 付款工具案例

模型请求付款工具后客户端超时，状态为 `unknown`。安全路径是：

1. 查询支付服务状态。
2. 若已完成，读取回执并结束原动作。
3. 若处理中，进入轮询或人工队列，不重复提交。
4. 若明确未提交且授权仍有效，最多重试一次并复用幂等键。
5. 若无法确认，保持未知并阻止自动重试。

备用模型可以帮助解释状态，但不能直接重新调用付款工具。这里重要的是状态可查询、授权仍有效、幂等键稳定，而不是备用模型的语言质量。

### 16.7.3 文件、邮件和发布也一样

文件写入、发送邮件、创建工单和发布配置都可能半成功。每个工具要声明：

- 请求何时算提交。
- 是否支持状态查询。
- 幂等键作用域是什么。
- 重试是否可能重复副作用。
- 失败后是否有补偿或撤销接口。

没有这些契约时，高风险动作应进入人工或明确拒绝，而不是由路由器猜测。

## 16.8 协议与后端适配

### 16.8.1 tokenizer 和 chat template

不同模型的 tokenizer、system message 模板、特殊 token、reasoning item 和上下文计算方式可能不同。adapter 要重新计算 token 预算，验证消息角色和截断策略。主模型的 token 数不能直接复制给备用模型。

### 16.8.2 tool schema

备用后端可能不支持并行工具、结构化参数、函数调用或某个媒体类型。如果它不支持结构化工具，不能把 tool call 退化成普通文本后交给 executor 猜测。正确路径是只读解释、生成待确认草稿、人工处理或拒绝。

### 16.8.3 流式事件

流式响应可能在文本结束前断开，工具事件可能已经发送但最终消息没有到达。路由器要记录每个 event、sequence number 和 action ID，恢复时从 checkpoint 继续或查询状态，而不是把半段文本当成完整结果。

### 16.8.4 多模态和数据区域

备用后端支持的图像分辨率、PDF、音频、数据区域和保留政策可能不同。若主路径收到敏感图像，不能因为备用模型不支持脱敏就原样转发。可以先做本地脱敏、改为 OCR 摘要或转人工；数据处理协议是路由条件，不是部署细节。

### 16.8.5 适配器的验证

adapter 需要验证：

1. 消息角色和 provenance 是否保留。
2. 工具 schema 是否语义等价。
3. 参数类型、枚举和必填字段是否一致。
4. context overflow 是否显式报告。
5. 流式结束和工具调用事件是否可回放。
6. 输出是否绑定正确的 request/action ID。
7. 备用 provider 的权限、区域和留存是否合规。

适配器通过解析不等于策略通过，二者应分别报告。

## 16.9 上下文超限与证据降级

### 16.9.1 静默截断是危险的

如果长文档中包含权限、取消或安全约束，直接截去末尾或中间段可能改变动作含义。路由器应明确选择：

- 检索相关片段。
- 生成带来源的摘要。
- 分批处理并保留片段范围。
- 改为只读回答。
- 进入异步或人工路径。
- 拒绝并说明长度限制。

### 16.9.2 证据完整性状态

可以标记：

~~~math
\mathrm{evidence\_status}
\in\{\mathrm{complete},\mathrm{partial},\mathrm{failed},\mathrm{unknown}\}.
~~~

当状态为 `partial` 或 `unknown` 时，低风险摘要可以继续，但写入、外发、删除和发布不能把它当作完整证据。摘要要带来源范围、生成版本和未覆盖范围。

### 16.9.3 分块不是免费等价

将一个请求分成 (k) 个片段可能需要 (k) 次分类、检索或模型调用，增加延迟和成本；片段边界还可能破坏跨段关系。合并结果时不能简单把局部“安全”拼成全局安全。要为分块策略建立任务级回归集和边界样本。

## 16.10 安全降级等级与用户沟通

### 16.10.1 能力等级

| 等级 | 能力 | 用户可见语义 |
| --- | --- | --- |
| Full | 原路径的完整模型、证据和工具 | 可执行已授权动作 |
| Constrained | 短上下文、只读工具或受限数据 | 能力范围已缩小 |
| Informational | 摘要、引用、解释，不改变外部状态 | 只提供信息，未执行动作 |
| Human | 人工复核或异步队列 | 等待处理或需要确认 |
| Refuse | 无法在当前范围安全完成 | 明确说明未完成和下一步 |

降级必须由服务端策略决定，不能由模型为了满足用户而自选更高权限路径。

### 16.10.2 用户需要知道什么

安全说明至少包括：

1. 当前完成了什么。
2. 哪些动作没有执行。
3. 哪些证据未读取或未验证。
4. 是否需要等待、补充信息或人工确认。
5. 是否存在待查询的外部状态。

例如：“当前完成本地字段提取，未提交报销写入；图片边缘未能完整读取，请重新上传或转人工。”这比“系统繁忙，请重试”更能避免用户重复提交。

### 16.10.3 不泄露内部防护细节

透明不等于暴露内部阈值、provider 地址、规则排列或可被滥用的安全策略。用户可见说明围绕任务状态和下一步，管理员 trace 才保存完整的错误、版本和策略原因。

## 16.11 策略拒绝、权限撤销与服务故障

### 16.11.1 拒绝不能通过换模型绕过

如果主路径因为用户无权访问、策略不允许外发或审批过期而拒绝，备用模型仍然使用同一主体、资源和动作约束。可以改为说明、脱敏摘要、合法的只读替代或人工申诉，但不能把拒绝变成另一种未经授权的调用。

### 16.11.2 策略服务不可用

公开只读、无敏感数据的任务可以在短 TTL 和故障标记下受限继续；写数据库、支付、删除、外发和生产发布应停止或人工接管。策略恢复后，积压请求重新检查主体、资源版本、审批和过期时间，不能全部自动执行。

### 16.11.3 权限在降级期间撤销

路由器切换后，原会话仍可能存在。每次高风险动作前重新读取 `principal_revision` 和 `policy_revision`；发现变化就转 `re-evaluate`。降级不应成为旧授权的避风港。

## 16.12 评估 fallback：看最终状态，不只看文本

### 16.12.1 故障注入集合

发布前应注入：

1. 主模型连接超时、生成中断和 provider 限流。
2. tokenizer、chat template、tool schema 和 cache 不兼容。
3. 上下文超限、媒体处理失败和证据缺失。
4. 策略服务不可用、权限刚刚撤销和审批过期。
5. 工具已提交但响应丢失、工具半成功和外部状态未知。
6. 流式连接断开、worker 重启、队列重复和客户端取消。
7. 备用模型不支持多模态、结构化输出或当前数据区域。

### 16.12.2 安全保持率

设 `S_p` 是主路径拒绝或限制的样本集合，`S_f` 是其 fallback 结果。可以观察：

~~~math
\mathrm{SafetyPreservation}
=\frac{\#\{x\in S_p:\mathrm{fallback}(x)\text{仍遵守原约束}\}}
{|S_p|}.
~~~

它不是“主模型和备用模型输出相似率”。主路径拒绝的请求在 fallback 仍然拒绝或进一步收窄，才算保持安全约束。当 `|S_p|=0` 时，这个指标应报告为 `N/A`，表示本轮没有可评价的主路径限制样本，而不是报告 0 或 1。

### 16.12.3 重复副作用率

~~~math
\mathrm{DuplicateSideEffectRate}
=\frac{\#\text{因 fallback 重复提交的外部动作}}
{\#\text{可能产生外部副作用的 fallback 任务}}.
~~~

分子要通过外部状态或 action ledger 确认，而不是只数模型调用次数。一个模型可能调用两次但 executor 用幂等键只提交一次；也可能只调用一次却因重试中间件提交两次。当没有任何可能产生副作用的 fallback 任务时，重复副作用率也应报告为 `N/A`，不能把“没有样本”误读为“风险为零”。

### 16.12.4 任务帮助性和误解率

安全降级还应报告完成率、误拒率、转人工率、平均和尾延迟、成本、用户是否误以为动作已经完成。只优化阻断率，会把所有请求都导向拒绝；只优化完成率，会把未知状态伪装成成功。

### 16.12.5 状态不变量

每个注入场景都应检查：

- 未授权工具没有执行。
- 已提交动作没有重复。
- 敏感数据没有进入不合规后端。
- 未验证的结果没有被标成完成。
- request/action ID、版本和审计事件连续。
- 恢复后不会越过撤销和过期授权。

这些是状态检查，不是自然语言 judge 可以单独替代的指标。

## 16.13 灰度、熔断与回滚

### 16.13.1 shadow 路由

新的 fallback 路由可以先 shadow 运行：记录它会选择哪个后端、会减少哪些能力、会产生多少成本和人工量，但不改变真实动作。用同一批脱敏请求比较旧路由、新路由、策略决定和最终状态。

### 16.13.2 受限 canary

启用新路径时，先选择低副作用、低流量、低敏感度租户，禁止高风险工具和跨区域数据转发。随着数据收集，逐步放开动作类别，而不是一次性对所有工具生效。

### 16.13.3 回滚配置集合

路由回滚要同时恢复：

~~~text
fallback policy revision
backend allowlist
adapter / tokenizer revision
tool schema mapping
retry and timeout budget
data-region and retention policy
user-facing status schema
audit schema
~~~

如果新路由已经产生了人工工单、草稿或未确认动作，回滚还要规定这些状态由哪个版本继续处理，不能简单换回一个旧 URL。

## 16.14 成本、延迟与安全的联合目标

降级路径不是越快越好。可以把一次请求的简化效用写成：

~~~math
U
=V_{\mathrm{helpful}}
 -C_{\mathrm{latency}}
 -C_{\mathrm{compute}}
 -C_{\mathrm{human}}
 -C_{\mathrm{risk}}.
~~~

小模型可能降低 compute cost，却增加误拒或错误动作的 `C_risk`；人工路径增加延迟和人工成本，却可能降低不可逆副作用。权重应按动作和用户价值切片，而不是用一个全局平均效用掩盖高风险尾部。

## 16.15 一个可运行的 fallback 状态审计 demo

下面的代码使用合成状态，演示路由器先判断动作是否已经发生，再决定有限重试、查询状态、停止或转信息路径。它不会调用真实 provider，也不会执行真实付款。

~~~python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Event:
    name: str
    action_id: str
    state: str
    error: str
    authorized: bool
    side_effect_possible: bool
    retries: int
    backend_supports_tool: bool
    original_scope: str = "read_only"
    status_query_available: bool = True
    status_query_authorized: bool = True


def route(event: Event) -> Dict[str, object]:
    actions: List[str] = []
    scope_rank = {"none": 0, "read_only": 1, "write": 2}
    if event.original_scope not in scope_rank:
        raise ValueError(f"unsupported original scope: {event.original_scope}")

    # First preserve the external fact.  Authorization controls new actions;
    # it must not rewrite an action that may already have happened.
    if event.state == "committed":
        actions.append("report_committed_without_replay")
        decision = "finish"
    elif event.state == "unknown" or event.side_effect_possible:
        # Querying an existing action is separate from replaying the original
        # business action.
        if event.status_query_available and event.status_query_authorized:
            actions.append("query_external_action_status")
            decision = "recover_state"
        else:
            actions.append("preserve_unknown_and_handoff")
            decision = "unknown"
    elif not event.authorized:
        actions.append("stop_and_preserve_denial")
        decision = "deny"
    elif event.state == "failed" and event.error == "policy_denied":
        actions.append("stop_and_preserve_denial")
        decision = "deny"
    elif event.error == "timeout" and event.retries < 1:
        actions.append("retry_once_without_privilege_change")
        decision = "retry"
    elif event.error == "unsupported_tool" or not event.backend_supports_tool:
        actions.append("switch_to_informational_path")
        decision = "informational"
    else:
        actions.append("handoff_to_human")
        decision = "human"

    target_scope = (
        "none"
        if decision in {"deny", "unknown", "finish"}
        else "read_only"
        if decision in {"informational", "recover_state", "human"}
        else event.original_scope
    )
    allowed_scope = min(
        (event.original_scope, target_scope),
        key=lambda scope: scope_rank[scope],
    )
    return {
        "action_id": event.action_id,
        "decision": decision,
        "actions": actions,
        "original_scope": event.original_scope,
        "allowed_scope": allowed_scope,
        "scope_preserved": (
            scope_rank[allowed_scope] <= scope_rank[event.original_scope]
        ),
    }


events = [
    Event("plain_timeout", "a-1", "running", "timeout", True, False, 0, True),
    Event("payment_unknown", "a-2", "unknown", "timeout", True, True, 0, True, "write"),
    Event("policy_denial", "a-3", "failed", "policy_denied", False, False, 0, True, "write"),
    Event("no_tool_backend", "a-4", "running", "unsupported_tool", True, False, 0, False, "write"),
    Event("unknown_without_query", "a-5", "unknown", "timeout", True, True, 0, True, "write", False),
    Event("committed_after_revoke", "a-6", "committed", "timeout", False, True, 0, True, "write"),
    Event("unknown_after_revoke_query_allowed", "a-7", "unknown", "timeout", False, True, 0, True, "write", True, True),
    Event("unknown_after_revoke_query_blocked", "a-8", "unknown", "timeout", False, True, 0, True, "write", True, False),
]


for item in events:
    print(item.name, route(item))
~~~

预期的核心决定是：

~~~text
plain_timeout retry retry_once_without_privilege_change
payment_unknown recover_state query_external_action_status
policy_denial deny stop_and_preserve_denial
no_tool_backend informational switch_to_informational_path
unknown_without_query unknown preserve_unknown_and_handoff
committed_after_revoke finish report_committed_without_replay
unknown_after_revoke_query_allowed recover_state query_external_action_status
unknown_after_revoke_query_blocked unknown preserve_unknown_and_handoff
~~~

实际打印还会包含 `action_id`、`original_scope`、`allowed_scope` 和 `scope_preserved`。这里的 `scope_preserved=True` 只表示允许范围没有超过事件记录的原始范围，并不代表真实工具已经安全；生产系统还要把查询结果、幂等键、版本、参数 digest 和 executor 状态接入同一条 trace。特别要注意：`committed_after_revoke` 仍然报告已经提交的事实，但不允许因为当前授权失效而把历史动作改写成“未执行”；后续查询、退款或补偿动作要重新做权限检查。

这里把两种权限明确分开：`authorized` 表示原业务动作当前是否仍被允许，`status_query_authorized` 表示系统是否有权进行只读状态查询。前者撤销后，系统不能重试付款或执行补偿；后者若仍有效，系统可以查询已有 action 的最终状态，但不能因此恢复原来的写权限。若连状态查询也不被允许，路由器只能保留 `unknown` 并转人工或等待具有权限的内部流程。对于 `committed_after_revoke`，`allowed_scope=none` 表示没有新的外部动作；它不否认历史上已经发生的提交，也不等于用户一定可以看到全部业务细节。

## 16.16 常见失败模式

### 16.16.1 所有错误都重试

无限重试会放大成本和副作用。应先分类错误、查询状态、检查权限，再使用统一预算；未知状态不能直接重放。

### 16.16.2 换模型绕过拒绝

安全拒绝、权限撤销和审批过期属于策略结果，不是模型质量故障。备用模型必须继承主体、资源、动作和范围约束，合法替代只能是信息、澄清、人工或拒绝。

### 16.16.3 把 tool call 转成普通文本

不支持结构化工具的后端输出一段“请调用某工具”的文本，executor 若自行解析，就失去了 schema、来源和权限边界。应转只读、草稿、人工或拒绝。

### 16.16.4 忽略半成功

客户端只看到超时，不代表外部动作没有发生。每个工具要有 action ID、状态查询和幂等键；没有状态查询能力时，高风险动作不应自动重试。

### 16.16.5 静默截断上下文

截断可能丢掉取消、权限和安全约束。应显式报告证据状态，改为检索、摘要、分批或信息路径，并向用户说明未覆盖范围。

### 16.16.6 备用后端使用更大权限

备用模型为了“保证成功”而连接更宽的数据或工具，等于把故障变成权限升级。后端 allowlist、数据区域和工具范围要由服务端固定，路由器只能收窄能力。

### 16.16.7 主备同时提交

并行竞速可以降低延迟，却可能让两个后端都调用外部工具。高副作用动作应先选出一个执行者，或由 executor 使用全局 action ID 和幂等键保证只提交一次。

### 16.16.8 用户不知道降级

用户以为“已完成”而实际上只是生成草稿，会重复提交或产生业务错误。状态说明要区分完成、待确认、未知、未验证和拒绝。

### 16.16.9 只有最终文本，没有状态 trace

自然语言 judge 可能认为“我无法付款”足够安全，却看不到付款已经提交。评估必须检查工具日志、外部状态、文件 diff、权限事件和 action ledger。

### 16.16.10 fallback 自己没有回归集

主模型的安全测试不覆盖路由器、adapter、重试、状态恢复和用户沟通。每个新路径都要加入专门的故障注入、反事实和回放样本。

## 16.17 资料与证据边界

### 16.17.1 可靠性和故障恢复

1. [Google SRE Book：Handling Overload](https://sre.google/sre-book/handling-overload/)：支持过载、排队、负载控制和服务恢复的工程背景。
2. [Google SRE Book：Addressing Cascading Failures](https://sre.google/sre-book/addressing-cascading-failures/)：支持重试放大、熔断和级联故障讨论。
3. [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110)：支持请求方法、状态码、重试语义和幂等性边界。

这些资料说明通用分布式系统的过载、重试和协议语义，不自动说明某个 LLM provider 的内部状态，也不能替代具体工具的提交契约。

### 16.17.2 生成式 AI 风险和应用层边界

1. [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)：支持把 fallback、监控、事故和治理放入生命周期。
2. [NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)：支持生成式 AI 的风险、数据、测量和治理边界。
3. [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)：支持提示注入、敏感数据和工具集成风险。

这些资料支持风险管理和应用防护语言，不证明一个 fallback 路由已经保持所有安全性质。安全保持率、重复副作用和状态一致性必须用本地工具契约和故障注入验证。

### 16.17.3 本章可以支持的结论

本章可以严谨地说：

1. fallback routing 首先是状态恢复和错误分类问题，其次才是模型选择问题。
2. 降级可以减少上下文、工具和动作范围，但不能扩大权限或绕过策略拒绝。
3. 外部动作未知时要查询状态和使用幂等键，不能盲目重试。
4. 协议、tokenizer、媒体、数据区域和工具 schema 是路由条件，不是实现细节。
5. 评估 fallback 必须查看外部状态、审计事件、帮助性和用户误解，而不只是最终文本。
6. 受限继续、人工、异步和拒绝都是合法的降级结果。

本章不能严谨地说：

1. 换一个更强或更小的模型就自动解决失败。
2. HTTP 200 或自然语言“完成”就证明外部动作已正确完成。
3. 主路径拒绝的请求可以安全交给另一个模型重新尝试。
4. 上下文截断、schema 转换和 provider 切换不会改变安全语义。
5. 没有状态查询和幂等契约时，高风险动作可以无限重试。

## 16.18 思考与实践

### 题目一：画一张状态转移图

为“生成代码 patch”和“提交支付”分别画出 `created`、`running`、`unknown`、`committed`、`failed`、`human` 和 `cancelled` 状态。标出哪些状态可以自动重试，哪些必须先查询外部系统。

### 题目二：设计后端能力矩阵

为主模型、同权限备用模型、只读模型和人工路径记录 tokenizer、上下文、工具、多模态、数据区域、延迟和留存约束。说明哪些字段缺失时必须拒绝路由。

### 题目三：设计故障注入

选择一次外发文件、一次代码发布和一次普通问答，分别注入超时、权限撤销、策略服务不可用、工具半成功和上下文超限。为每种情况写出预期的最终状态、用户说明和审计字段。

### 题目四：计算联合指标

给出一批主路径拒绝、fallback 结果、工具提交和用户任务成功记录，分别计算安全保持率、重复副作用率、正常任务完成率和人工接管率。说明每个分母为何不能共用。

## 16.19 结语：可靠的降级是有限服务，不是假装成功

Fallback routing 的本质是：当原路径不可用时，系统要在已知状态、有限权限和明确证据下选择下一步。无副作用的传输故障可以有限重试；provider 限流可以排队或切换同权限副本；工具状态未知必须查询；策略拒绝不能绕过；协议不兼容应改为信息、草稿、人工或拒绝；上下文不足要显式标记证据缺口。

一个成熟的降级系统会保存 request ID、action ID、幂等键、错误分类、模型和工具版本、策略版本、权限范围、状态变化和用户可见语义。它允许低风险任务继续，也能在高风险动作无法确认时停下来。路由器并不负责把所有失败隐藏掉，它负责把失败变成可解释、可恢复、可审计且不会扩大伤害的有限服务。
