# 第 30 章 跨 Agent 权限、身份、审计和信任模型

前面几章我们讲了 A2A 的能力发现、任务委派、消息格式、上下文边界，以及 A2A 与 MCP 的分工。

本章进入 A2A 系统里最关键、也最容易被低估的主题：跨 Agent 权限、身份、审计和信任模型。

多 Agent 系统一旦进入真实企业环境，就不再只是“几个智能体互相帮忙”。它会接触真实用户、真实数据、真实权限、真实业务系统和真实责任边界。此时必须回答一组非常严肃的问题：

1. 这个 Agent 是谁？
2. 它代表谁发起请求？
3. 它有没有权委派给另一个 Agent？
4. 下游 Agent 能不能看到这些上下文？
5. 下游 Agent 能不能继续委派？
6. 哪些动作需要用户确认？
7. 出问题后如何知道责任链？
8. 如何防止不可信 Agent、被污染的上下文或错误结果扩散？

你可以先记住本章核心结论：

> 多 Agent 安全不是“让模型守规矩”，而是用身份、权限、策略、审计和信任边界把每一次委派和上下文流动约束住。

## 30.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求，先核对 A2A 官方 Protocol Specification、A2A enterprise / multi-tenancy / security 主题说明，以及 MCP 官方 Authorization、Security Best Practices 和 Specification，再回到本项目已有章节做通用工程抽象。

资料边界要说清楚：

1. A2A 官方协议把 Agent 间协作建模为 Agent Card、Task、Message、Part、Artifact、TaskStatus、认证授权扩展、企业能力和跨 Agent 通信治理。
2. MCP 官方资料强调 Host / Client / Server 边界、OAuth / authorization、roots、tools / resources / prompts、安全最佳实践和 Host 侧策略执行。
3. 本章只讨论防御性系统设计：跨 Agent 身份、OBO 授权、权限衰减、上下文策略、继续委派控制、人工确认、trace / audit、多租户隔离和信任分级。
4. 本章不提供攻击、绕过权限、伪造身份、规避审计、窃取 token、突破租户隔离或利用第三方 Agent 的做法；示例只用于说明如何发现和阻断风险。
5. 本章不实现真实 OAuth、mTLS、签名校验、KMS、SIEM、DLP、策略引擎或 A2A / MCP server；toy demo 的字段和阈值只是教学用审计模型，不是协议标准。

所以读本章时要把重点放在“责任链能否被系统证明”。一个多 Agent 系统即使最终答案看起来正确，如果身份链、授权链、上下文链、工具链、产物链和审计链断裂，也不能算可上线。

## 30.1 为什么跨 Agent 安全更复杂

单 Agent 系统中，安全边界通常围绕一个 Host 展开：用户请求进入 Host，Agent 在 Host 控制下调用工具，Host 决定是否允许。

多 Agent 系统会多出几层复杂性：

1. 请求可能经过多个 Agent。
2. 每个 Agent 可能有不同维护方。
3. 每个 Agent 可能拥有不同工具权限。
4. 一个 Agent 的输出可能成为另一个 Agent 的输入。
5. 上下文可能被摘要、裁剪、转发和再转发。
6. 任务可能异步执行，跨越较长时间。
7. 结果可能以 Artifact 形式被其他 Agent 复用。
8. 错误、幻觉和越权可能沿协作链传播。

因此，多 Agent 安全不能只看单次工具调用，而要看完整协作链路。

## 30.2 三种身份：用户身份、Agent 身份、服务身份

跨 Agent 系统里，至少要区分三类身份。

### 30.2.1 用户身份

用户身份回答：

> 这个任务最终代表哪个用户或租户发起？

用户身份影响数据访问范围。例如用户 A 可以看华东区域数据，用户 B 可以看全国数据，那么同一个数据分析 Agent 在代表不同用户执行任务时，能访问的数据也应该不同。

### 30.2.2 Agent 身份

Agent 身份回答：

> 当前发起请求或执行任务的是哪个 Agent？

Agent 身份用于判断：

1. 这个 Agent 是否可信。
2. 它是否被允许接收某类任务。
3. 它是否能访问某类上下文。
4. 它是否能继续委派。
5. 它生成的 Artifact 是否可被其他 Agent 使用。

### 30.2.3 服务身份

服务身份回答：

> 这个 Agent 背后的运行服务或部署实例是谁？

例如，同一个 Agent 可能有开发环境、测试环境、生产环境。服务身份可以用于 mTLS、签名、调用来源校验和运行环境隔离。

这三类身份不能混为一谈。一个请求可能是：

```text
用户 user_123 通过 Orchestrator Agent 委派给 DataAnalysisAgent，由 data-agent-prod 服务实例执行。
```

审计时必须能同时看到这三层身份。

## 30.3 On-Behalf-Of：代表用户执行

企业系统中，一个 Agent 很多时候不是以自己的名义访问资源，而是“代表用户”访问资源。

这叫 on-behalf-of，简称 OBO。

例如：

```text
用户请求总控 Agent 分析销售数据。
总控 Agent 委派给数据 Agent。
数据 Agent 查询数据库。
```

数据库权限应该按谁算？

错误答案是：按数据 Agent 的超级权限算。

更合理的答案是：按用户身份和任务授权共同决定。

也就是说，数据 Agent 可以执行分析能力，但它不能突破用户本来拥有的数据权限。

OBO 模型通常需要：

1. 用户身份 token。
2. Agent 身份 token。
3. 任务授权范围。
4. 权限衰减规则。
5. 审计记录。

一个简化授权上下文可以是：

```json
{
  "user": {
    "id": "user_123",
    "tenant": "tenant_a",
    "roles": ["business_analyst"]
  },
  "caller_agent": "agent.orchestrator.v1",
  "assignee_agent": "agent.data_analysis.v1",
  "task_id": "task_123",
  "scopes": ["metrics.read"],
  "constraints": {
    "regions": ["east_china"],
    "data_level": "aggregate_only"
  }
}
```

## 30.4 权限衰减：越往下游权限越少

多 Agent 委派中，一个重要原则是权限衰减。

意思是：下游 Agent 获得的权限不应该大于上游任务所需权限，更不应该大于原用户授权。

例如：

1. 用户允许分析聚合数据。
2. 总控 Agent 委派给数据 Agent。
3. 数据 Agent 只能访问聚合数据，不能访问用户级明细。
4. 报告 Agent 只能接收聚合结论，不能接收原始查询结果。

权限不应该在委派链中扩大。

可以用一条规则记住：

> Delegation should narrow permissions, not expand them.

常见权限衰减维度包括：

1. 数据范围从全量缩小到子集。
2. 数据粒度从明细变成聚合。
3. 操作权限从读写变成只读。
4. 工具权限从多工具变成单工具。
5. 上下文从全文变成摘要。
6. Artifact 从可转发变成不可转发。
7. 有效期从长期变成短期。

## 30.5 跨 Agent 授权检查点

一个 A2A 任务从发起到完成，至少应该有多个授权检查点。

### 30.5.1 委派前检查

委派前要检查：

1. 发起 Agent 是否能委派任务。
2. 下游 Agent 是否在 allowlist 中。
3. 用户是否允许使用该下游 Agent。
4. 任务类型是否匹配下游能力。
5. 当前上下文是否允许传给下游。
6. 是否需要人工确认。

### 30.5.2 接收时检查

下游 Agent 接收任务时也要检查：

1. 调用方身份是否可信。
2. 任务授权是否有效。
3. 上下文数据分类是否可接收。
4. 请求是否超过自身权限。
5. 是否需要拒绝或要求更多授权。

### 30.5.3 工具调用前检查

下游 Agent 内部调用 MCP 工具前，要检查：

1. 当前任务是否允许调用该工具。
2. 当前用户是否有对应资源权限。
3. 工具调用是否符合数据策略。
4. 是否需要用户确认。
5. 工具结果是否需要脱敏。

### 30.5.4 结果返回前检查

结果返回前要检查：

1. 结果是否包含敏感信息。
2. Artifact 是否允许返回给上游。
3. 证据引用是否泄露资源路径。
4. 是否需要脱敏或摘要。
5. 是否允许后续 Agent 使用。

这四个检查点缺任何一个，系统都容易出问题。

## 30.6 信任模型：不是所有 Agent 都等价

多 Agent 系统里，不同 Agent 的信任级别不同。

常见分类包括：

1. 内部核心 Agent。
2. 内部普通 Agent。
3. 沙箱 Agent。
4. 第三方 Agent。
5. 实验性 Agent。
6. 用户自定义 Agent。

不同信任级别决定它能接收什么任务、访问什么上下文、返回什么产物、是否允许继续委派。

例如：

| Agent 类型 | 可接收数据 | 是否可继续委派 | 是否可访问生产工具 |
| --- | --- | --- | --- |
| 内部核心 Agent | confidential | 受控允许 | 受控允许 |
| 内部普通 Agent | internal | 需要审批 | 少量只读 |
| 沙箱 Agent | public / synthetic | 不允许 | 不允许 |
| 第三方 Agent | public / redacted | 不允许 | 不允许 |
| 用户自定义 Agent | 用户授权范围 | 默认不允许 | 默认不允许 |

信任模型要系统强制执行，不能只写在文档里。

## 30.7 Agent Allowlist 与签名

跨 Agent 调用不能随便连任何地址。否则攻击者可以注册一个看似有用的 Agent，诱导总控 Agent 把敏感上下文发过去。

生产系统至少应该有：

1. Agent 注册审核。
2. Agent Card 签名。
3. 调用端 allowlist。
4. 服务端身份校验。
5. 版本固定或版本范围。
6. 运行环境标记。
7. 撤销机制。

Agent Card 签名用于证明能力声明没有被篡改。服务端身份校验用于证明你连接到的确实是那个 Agent，而不是冒名服务。

## 30.8 上下文转发权限

跨 Agent 安全最难的不是“能不能调用”，而是“能不能转发上下文”。

例如，总控 Agent 收到一份机密合同，想委派给合同审查 Agent。

需要判断：

1. 合同审查 Agent 是否可信？
2. 它是否属于内部 Agent？
3. 合同数据分类是否允许传给它？
4. 是否可以传全文，还是只能传片段？
5. 是否允许它保存 Artifact？
6. 是否允许它继续委派给法务 Agent？

context_policy 应该参与授权判断。例如：

```json
{
  "data_classification": "confidential",
  "allow_forwarding": false,
  "allowed_recipients": ["agent.legal_review.internal.v2"],
  "redaction_required": false,
  "retention": "7d"
}
```

如果下游 Agent 不在 allowed_recipients 中，就不能接收上下文。

## 30.9 继续委派权限

一个 Agent 接到任务后，是否可以再委派给别的 Agent？

这必须显式控制。

否则会出现委派链失控：

```text
总控 Agent -> 数据 Agent -> 外部图表 Agent -> 未知摘要 Agent
```

用户和上游 Agent 可能根本不知道数据被传到了哪里。

继续委派策略可以包括：

1. 不允许继续委派。
2. 只允许委派给指定 Agent。
3. 只允许委派非敏感摘要。
4. 继续委派必须回调上游批准。
5. 继续委派必须记录完整 trace。

默认策略应该保守：不允许下游 Agent 自由继续委派。

## 30.10 人工确认与高风险动作

跨 Agent 系统里，有些动作必须人工确认。

高风险动作包括：

1. 把敏感数据传给低信任 Agent。
2. 调用会修改外部系统的 Agent。
3. 生成对外发送的正式文档。
4. 执行生产变更。
5. 访问用户级明细数据。
6. 继续委派给第三方 Agent。
7. 删除或覆盖 Artifact。

确认界面不能只显示“是否允许”。它应该展示：

1. 谁请求。
2. 代表哪个用户。
3. 要委派给谁。
4. 要传递哪些上下文。
5. 数据密级是什么。
6. 下游会做什么。
7. 是否允许继续委派。
8. 风险说明。

用户确认也要进入审计日志。

## 30.11 审计：必须记录完整责任链

多 Agent 系统出问题时，最怕回答不了：

> 到底是谁把什么数据传给了谁？谁调用了哪个工具？谁生成了最终结论？

审计日志至少要记录：

1. 用户身份。
2. 发起 Agent。
3. 接收 Agent。
4. 任务 ID。
5. 消息 ID。
6. 上下文 Resource / Artifact 引用。
7. 数据分类。
8. 授权决策。
9. 工具调用。
10. 状态变更。
11. 人工确认。
12. 最终产物。

一个审计事件可以长这样：

```json
{
  "event_type": "a2a_task_delegated",
  "timestamp": "2026-05-29T12:00:00Z",
  "trace_id": "trace_001",
  "task_id": "task_123",
  "user_id": "user_123",
  "caller_agent": "agent.orchestrator.v1",
  "assignee_agent": "agent.legal_review.v2",
  "context_refs": ["artifact://contract/redacted-v1.pdf"],
  "data_classification": "confidential",
  "decision": "allowed",
  "policy": "internal_confidential_review_policy_v3"
}
```

审计日志要防篡改，并且根据敏感程度做访问控制。

## 30.12 Trace 与 Audit 的区别

Trace 和 Audit 经常混用，但它们关注点不同。

Trace 主要服务于工程排障和可观测性：

1. 请求链路。
2. 延迟。
3. 工具调用。
4. 状态变化。
5. 错误定位。

Audit 主要服务于安全、合规和责任追踪：

1. 谁做了什么。
2. 是否有权限。
3. 是否经过确认。
4. 数据是否被不当访问。
5. 是否符合策略。

二者可以共享 trace_id，但保留内容和访问权限可能不同。Trace 中可能包含调试信息，Audit 中必须保证关键安全事件完整、可查、不可抵赖。

## 30.13 信任传播：结果可信不等于 Agent 可信

一个 Agent 可信，不代表它每次输出都可信。一个低信任 Agent 也可能偶尔给出有用结果。

因此信任要分层：

1. Agent 级信任：这个 Agent 是否来自可信来源。
2. 数据级信任：它使用的数据是否可信。
3. 方法级信任：它是否使用了允许的方法和工具。
4. 结果级信任：输出是否有证据、置信度和验证。
5. 任务级信任：这次任务是否符合策略。

下游 Agent 的结果进入上游上下文时，应该保留这些信息：

```json
{
  "claim": "Contract contains unfavorable payment terms.",
  "produced_by": "agent.legal_review.v2",
  "agent_trust_level": "internal_core",
  "evidence": ["artifact://contract/section-4"],
  "confidence": "high",
  "verified_by": "human_reviewer_001"
}
```

否则系统容易把一个 Agent 的未验证输出当成绝对事实。

## 30.14 多租户隔离

企业 Agent 平台通常是多租户的。不同租户的数据、Agent、工具和 Artifact 必须隔离。

多租户隔离包括：

1. Agent 注册隔离。
2. 用户身份隔离。
3. 资源访问隔离。
4. Artifact 存储隔离。
5. Trace 和 Audit 隔离。
6. 配额和成本隔离。
7. 策略配置隔离。

一个典型错误是：Agent Card 全局可见，但其中某些 Agent 只应该对特定租户可见。服务发现必须考虑 tenant scope。

## 30.15 一个完整例子：合同审查委派

用户说：

```text
请帮我审查这份供应商合同，重点看付款周期和违约责任。
```

系统可以这样处理：

1. 总控 Agent 创建 root task。
2. Host 判断合同为 confidential。
3. 服务发现返回 LegalReviewAgent 和 PaymentTermsAgent。
4. 策略系统过滤掉第三方 Agent。
5. 总控 Agent 只允许委派给内部法务 Agent。
6. 委派消息带 context_policy：禁止继续转发，Artifact 保留 7 天。
7. LegalReviewAgent 接收任务前校验调用方和数据分类。
8. LegalReviewAgent 使用自己的 MCP 文档工具读取合同片段。
9. 结果返回前脱敏无关供应商联系人信息。
10. 系统记录 A2A 委派、MCP 读取、Artifact 生成和用户查看事件。

如果后续报告 Agent 需要生成摘要，它不能自动获得合同全文，只能获得法务 Agent 输出的结构化结论和允许共享的引用。

## 30.16 常见误区

### 30.16.1 把用户权限和 Agent 权限混为一谈

用户有权访问某数据，不代表所有 Agent 都能看到这份数据。Agent 也必须被授权。

### 30.16.2 委派任务时复制全部上下文

这是上下文泄露的常见来源。应该遵守最小上下文和权限衰减。

### 30.16.3 允许下游 Agent 自由继续委派

这会导致数据流向失控。继续委派必须显式授权。

### 30.16.4 只记录最终答案，不记录过程

多 Agent 系统如果不记录任务链、消息链、工具调用和授权决策，出问题后无法审计。

### 30.16.5 信任 Agent 但不验证结果

可信 Agent 也可能犯错。结果仍需要证据链、置信度、限制说明和必要的人类复核。

## 30.17 跨 Agent 安全审计指标与最小 demo

跨 Agent 安全不能只靠 checklist。更稳妥的方式，是把每次委派、上下文转发、MCP 工具调用、结果返回和审计记录都变成可检查样本。

设第 $i$ 个跨 Agent 安全样本为：

```math
s_i=(u_i,a_i,v_i,o_i,p_i,c_i,d_i,r_i,h_i,t_i,z_i)
```

其中 $u_i$ 是用户与租户身份，$a_i$ 是调用 Agent / 接收 Agent / 服务身份，$v_i$ 是 OBO 授权和 scope，$o_i$ 是被请求的操作，$p_i$ 是策略与权限衰减，$c_i$ 是上下文策略，$d_i$ 是继续委派，$r_i$ 是结果与 Artifact 释放，$h_i$ 是人工确认，$t_i$ 是 trace / audit 字段，$z_i$ 是 eval 标签。

对任意检查项 $k$，覆盖率仍然写成统一形式：

```math
C_k=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\{g_k(s_i)=1\}
```

本章建议至少审计这些指标：

1. 身份链覆盖率 $C_{\mathrm{id}}$：是否同时记录用户身份、租户、调用 Agent、接收 Agent、服务实例和 task id。
2. OBO scope 绑定率 $C_{\mathrm{obo}}$：下游请求 scope 是否同时受用户授权和任务授权约束。
3. 委派 allowlist 覆盖率 $C_{\mathrm{allow}}$：接收 Agent 是否在 allowlist 中，Agent Card 是否签名，服务身份是否校验。
4. 权限衰减覆盖率 $C_{\mathrm{atten}}$：数据密级、数据粒度、操作权限和工具权限是否没有向下游扩大。
5. 上下文策略执行率 $C_{\mathrm{ctx}}$：allowed recipients、redaction、retention、forwarding 等策略是否实际生效。
6. 继续委派控制率 $C_{\mathrm{delegate}}$：下游继续委派是否被默认禁止或显式批准。
7. 高风险确认覆盖率 $C_{\mathrm{confirm}}$：写操作、第三方 Agent、敏感数据、生产变更等高风险动作是否经过确认。
8. MCP 工具权限绑定率 $C_{\mathrm{tool}}$：下游 Agent 调 MCP 工具时是否同时检查用户 scope、任务 scope、租户和工具要求。
9. 结果释放控制率 $C_{\mathrm{result}}$：返回结果和 Artifact 是否脱敏、引用化、保留期受控，并避免泄露敏感证据路径。
10. Audit / Trace 完整率 $C_{\mathrm{audit}}$：安全关键事件是否记录 trace id、task id、用户、Agent、策略、决策、数据分类和上下文引用。
11. 信任与证据验证率 $C_{\mathrm{trust}}$：低信任 Agent 输出是否需要证据、人审或 verifier，可信 Agent 输出是否仍保留证据和限制。
12. 租户隔离率 $C_{\mathrm{tenant}}$：资源、Artifact、trace、Agent 可见性和缓存是否绑定同一租户边界。

上线门禁可以写成：

```math
G_{\mathrm{cross\_agent}}=
\mathbf{1}\{
C_{\mathrm{id}}\ge \tau_{\mathrm{id}},
C_{\mathrm{obo}}\ge \tau_{\mathrm{obo}},
C_{\mathrm{allow}}\ge \tau_{\mathrm{allow}},
C_{\mathrm{atten}}\ge \tau_{\mathrm{atten}},
C_{\mathrm{ctx}}\ge \tau_{\mathrm{ctx}},
C_{\mathrm{delegate}}\ge \tau_{\mathrm{delegate}},
C_{\mathrm{confirm}}\ge \tau_{\mathrm{confirm}},
C_{\mathrm{tool}}\ge \tau_{\mathrm{tool}},
C_{\mathrm{result}}\ge \tau_{\mathrm{result}},
C_{\mathrm{audit}}\ge \tau_{\mathrm{audit}},
C_{\mathrm{trust}}\ge \tau_{\mathrm{trust}},
C_{\mathrm{tenant}}\ge \tau_{\mathrm{tenant}}
\}
```

也可以给一个加权分数：

```math
S_{\mathrm{cross\_agent}}=\frac{\sum_k \alpha_k C_k}{\sum_k \alpha_k}
```

但在安全场景里，分数只能辅助排序，不能替代硬门禁。比如租户隔离或高风险确认不过线时，即使平均分很高，也应该阻断上线或限制场景。

下面是一个 0 依赖 demo。它用 12 个 toy trace 模拟合同审查、数据分析和多 Agent 委派中的常见风险。

```python
from collections import OrderedDict
from copy import deepcopy


REQUIRED_ID = {"user_id", "user_tenant", "caller_agent", "assignee_agent", "service_id", "task_id"}
REQUIRED_AUDIT = {"trace_id", "task_id", "user_id", "caller_agent", "assignee_agent", "decision", "policy", "data_classification", "context_refs", "timestamp"}
CLASS_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
DETAIL_RANK = {"synthetic": 0, "aggregate": 1, "redacted": 2, "row_level": 3}

BASE = {
    "identity": REQUIRED_ID,
    "user_scopes": {"contract.read", "legal.review"},
    "task_scopes": {"legal.review"},
    "requested_scopes": {"legal.review"},
    "assignee": "agent.legal.internal.v2",
    "allowed_agents": {"agent.legal.internal.v2"},
    "agent_card_signed": True,
    "service_verified": True,
    "requested_class": "confidential",
    "max_class": "confidential",
    "requested_detail": "redacted",
    "max_detail": "redacted",
    "context_policy": {
        "allowed_recipients": {"agent.legal.internal.v2"},
        "allow_forwarding": False,
        "redaction_required": True,
        "retention_days": 7,
        "allowed_delegatees": set(),
    },
    "context_redacted": True,
    "forwarded_to": None,
    "redelegation_approved": False,
    "high_risk": False,
    "confirmation": None,
    "tool": {"name": "contract_reader", "required_scope": "legal.review", "tenant": "tenant_a", "env": "prod"},
    "user_tenant": "tenant_a",
    "resource_tenant": "tenant_a",
    "artifact_tenant": "tenant_a",
    "trace_tenant": "tenant_a",
    "agent_visible_to_tenant": True,
    "result": {
        "sensitive_leak": False,
        "retention_days": 7,
        "evidence_refs_ok": True,
        "claim_type": "finding",
        "confidence": "high",
        "limitations": True,
        "verified_by": "reviewer_1",
        "agent_trust": "internal_core",
    },
    "audit_fields": REQUIRED_AUDIT,
    "eval_label": True,
}


def make_case(case_id, **overrides):
    case = deepcopy(BASE)
    case["id"] = case_id
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(case.get(key), dict):
            case[key].update(value)
        else:
            case[key] = value
    return case


CASES = [
    make_case("contract_review_happy_path"),
    make_case(
        "aggregate_data_happy_path",
        user_scopes={"metrics.read"},
        task_scopes={"metrics.read"},
        requested_scopes={"metrics.read"},
        assignee="agent.data.internal.v1",
        allowed_agents={"agent.data.internal.v1"},
        requested_class="internal",
        max_class="internal",
        requested_detail="aggregate",
        max_detail="aggregate",
        context_policy={"allowed_recipients": {"agent.data.internal.v1"}, "redaction_required": False, "retention_days": 3},
        tool={"name": "metric_query", "required_scope": "metrics.read", "tenant": "tenant_a", "env": "prod"},
        result={"retention_days": 3, "claim_type": "metric", "confidence": "medium", "verified_by": None, "agent_trust": "internal_standard"},
    ),
    make_case(
        "third_party_agent_not_allowed_bad",
        user_scopes={"contract.read"},
        task_scopes={"contract.read"},
        requested_scopes={"contract.read"},
        assignee="agent.third_party.summary.v1",
        requested_class="confidential",
        max_class="public",
        context_policy={"allowed_recipients": {"agent.legal.internal.v2"}, "retention_days": 1},
        high_risk=True,
        confirmation=None,
        tool=None,
        agent_visible_to_tenant=False,
        result={"retention_days": 1, "claim_type": "blocked", "verified_by": "policy_engine", "agent_trust": "third_party"},
    ),
    make_case(
        "missing_identity_bad",
        identity={"user_id", "caller_agent", "assignee_agent", "task_id"},
        service_verified=False,
        audit_fields={"trace_id", "task_id", "user_id", "caller_agent", "assignee_agent", "decision", "policy"},
    ),
    make_case(
        "obo_scope_expansion_bad",
        user_scopes={"metrics.read"},
        task_scopes={"metrics.read"},
        requested_scopes={"metrics.read", "customer.pii.read"},
        assignee="agent.data.internal.v1",
        allowed_agents={"agent.data.internal.v1"},
        requested_class="restricted",
        max_class="internal",
        requested_detail="row_level",
        max_detail="aggregate",
        context_policy={"allowed_recipients": {"agent.data.internal.v1"}, "retention_days": 1},
        context_redacted=False,
        high_risk=True,
        tool={"name": "customer_export", "required_scope": "customer.pii.read", "tenant": "tenant_a", "env": "prod"},
        result={"sensitive_leak": True, "retention_days": 30, "evidence_refs_ok": False, "claim_type": "metric", "limitations": False, "verified_by": None, "agent_trust": "internal_standard"},
        audit_fields={"trace_id", "task_id", "user_id", "caller_agent", "assignee_agent", "decision", "policy", "timestamp"},
    ),
    make_case(
        "unapproved_agent_bad",
        assignee="agent.unknown.review.v1",
        agent_card_signed=False,
        service_verified=False,
        high_risk=True,
        confirmation="approved",
        tool=None,
        agent_visible_to_tenant=False,
        result={"confidence": "low", "verified_by": None, "agent_trust": "unknown"},
    ),
    make_case(
        "context_forwarding_bad",
        forwarded_to="agent.third_party.summary.v1",
        high_risk=True,
        confirmation=None,
        tool=None,
        result={"claim_type": "summary", "confidence": "medium", "verified_by": None, "agent_trust": "internal_core"},
    ),
    make_case(
        "write_action_no_confirmation_bad",
        user_scopes={"ticket.read", "ticket.write"},
        task_scopes={"ticket.write"},
        requested_scopes={"ticket.write"},
        assignee="agent.support.internal.v1",
        allowed_agents={"agent.support.internal.v1"},
        requested_class="internal",
        max_class="internal",
        context_policy={"allowed_recipients": {"agent.support.internal.v1"}, "retention_days": 3},
        high_risk=True,
        tool={"name": "ticket_update", "required_scope": "ticket.write", "tenant": "tenant_a", "env": "prod"},
        result={"retention_days": 3, "claim_type": "action", "verified_by": None, "agent_trust": "internal_standard"},
    ),
    make_case(
        "tool_permission_mismatch_bad",
        user_scopes={"metrics.read"},
        task_scopes={"metrics.read"},
        requested_scopes={"metrics.read"},
        assignee="agent.data.internal.v1",
        allowed_agents={"agent.data.internal.v1"},
        requested_class="internal",
        max_class="internal",
        requested_detail="aggregate",
        max_detail="aggregate",
        context_policy={"allowed_recipients": {"agent.data.internal.v1"}, "redaction_required": False, "retention_days": 3},
        tool={"name": "billing_refund", "required_scope": "refund.write", "tenant": "tenant_a", "env": "prod"},
        result={"retention_days": 3, "claim_type": "action", "confidence": "medium", "verified_by": None, "agent_trust": "internal_standard"},
    ),
    make_case(
        "audit_missing_bad",
        audit_fields={"trace_id", "task_id", "user_id", "decision"},
        eval_label=False,
        tool=None,
        result={"verified_by": "reviewer_2"},
    ),
    make_case(
        "low_trust_unverified_result_bad",
        user_scopes={"public.read"},
        task_scopes={"public.read"},
        requested_scopes={"public.read"},
        assignee="agent.sandbox.research.v1",
        allowed_agents={"agent.sandbox.research.v1"},
        requested_class="public",
        max_class="public",
        requested_detail="synthetic",
        max_detail="synthetic",
        context_policy={"allowed_recipients": {"agent.sandbox.research.v1"}, "redaction_required": False, "retention_days": 1},
        tool=None,
        result={"retention_days": 1, "evidence_refs_ok": False, "claim_type": "fact", "limitations": False, "verified_by": None, "agent_trust": "sandbox"},
    ),
    make_case(
        "cross_tenant_leak_bad",
        user_scopes={"metrics.read"},
        task_scopes={"metrics.read"},
        requested_scopes={"metrics.read"},
        assignee="agent.data.internal.v1",
        allowed_agents={"agent.data.internal.v1"},
        requested_class="internal",
        max_class="internal",
        requested_detail="aggregate",
        max_detail="aggregate",
        context_policy={"allowed_recipients": {"agent.data.internal.v1"}, "redaction_required": False, "retention_days": 3},
        tool={"name": "metric_query", "required_scope": "metrics.read", "tenant": "tenant_b", "env": "prod"},
        resource_tenant="tenant_b",
        artifact_tenant="tenant_b",
        result={"retention_days": 3, "claim_type": "metric", "confidence": "medium", "verified_by": None, "agent_trust": "internal_standard"},
    ),
]


def check(case):
    policy = case["context_policy"]
    tool = case["tool"]
    result = case["result"]
    identity_ok = REQUIRED_ID <= set(case["identity"]) and case["service_verified"]
    scope_ok = case["requested_scopes"] <= case["user_scopes"] and case["requested_scopes"] <= case["task_scopes"]
    allowlist_ok = case["assignee"] in case["allowed_agents"] and case["agent_card_signed"] and case["service_verified"]
    attenuation_ok = CLASS_RANK[case["requested_class"]] <= CLASS_RANK[case["max_class"]] and DETAIL_RANK[case["requested_detail"]] <= DETAIL_RANK[case["max_detail"]]
    context_ok = case["assignee"] in policy["allowed_recipients"] and (not policy["redaction_required"] or case["context_redacted"]) and (policy["allow_forwarding"] or case["forwarded_to"] is None)
    redelegation_ok = case["forwarded_to"] is None or (case["forwarded_to"] in policy.get("allowed_delegatees", set()) and case["redelegation_approved"])
    confirmation_ok = not case["high_risk"] or case["confirmation"] == "approved"
    tool_ok = tool is None or (tool["required_scope"] in case["requested_scopes"] and tool["tenant"] == case["user_tenant"] and case["resource_tenant"] == case["user_tenant"] and tool["env"] == "prod")
    result_ok = not result["sensitive_leak"] and result["retention_days"] <= policy["retention_days"] and result["evidence_refs_ok"]
    audit_ok = REQUIRED_AUDIT <= case["audit_fields"] and case["eval_label"]
    trust_ok = result["claim_type"] in {"finding", "metric", "action", "blocked", "summary", "fact"} and result["confidence"] in {"low", "medium", "high"} and result["limitations"] and (result["agent_trust"] not in {"sandbox", "third_party", "unknown"} or result["verified_by"] is not None) and result["evidence_refs_ok"]
    tenant_ok = case["resource_tenant"] == case["user_tenant"] and case["artifact_tenant"] == case["user_tenant"] and case["trace_tenant"] == case["user_tenant"] and case["agent_visible_to_tenant"]
    return OrderedDict([
        ("identity_chain_coverage", identity_ok),
        ("obo_scope_binding", scope_ok),
        ("delegation_allowlist", allowlist_ok),
        ("permission_attenuation", attenuation_ok),
        ("context_policy_enforcement", context_ok),
        ("redelegation_control", redelegation_ok),
        ("high_risk_confirmation", confirmation_ok),
        ("mcp_tool_permission_binding", tool_ok),
        ("result_release_control", result_ok),
        ("audit_trace_completeness", audit_ok),
        ("trust_evidence_verification", trust_ok),
        ("tenant_isolation", tenant_ok),
    ])


scores = [check(case) for case in CASES]
metrics = OrderedDict((key, round(sum(score[key] for score in scores) / len(scores), 3)) for key in scores[0])
thresholds = {key: 0.95 for key in metrics}
failed_cases = [case["id"] for case, score in zip(CASES, scores) if not all(score.values())]
failed_gates = [key for key, value in metrics.items() if value < thresholds[key]]
case_score = {case["id"]: score for case, score in zip(CASES, scores)}
smoke = OrderedDict([
    ("contract_review_allowed", all(case_score["contract_review_happy_path"].values())),
    ("aggregate_data_allowed", all(case_score["aggregate_data_happy_path"].values())),
    ("caught_scope_expansion", not all(case_score["obo_scope_expansion_bad"].values())),
    ("caught_cross_tenant_leak", not all(case_score["cross_tenant_leak_bad"].values())),
])

print("smoke=", dict(smoke))
print("metrics=", dict(metrics))
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("cross_agent_security_gate_pass=", not failed_gates)
```

期望输出：

```text
smoke= {'contract_review_allowed': True, 'aggregate_data_allowed': True, 'caught_scope_expansion': True, 'caught_cross_tenant_leak': True}
metrics= {'identity_chain_coverage': 0.833, 'obo_scope_binding': 0.917, 'delegation_allowlist': 0.75, 'permission_attenuation': 0.833, 'context_policy_enforcement': 0.667, 'redelegation_control': 0.917, 'high_risk_confirmation': 0.667, 'mcp_tool_permission_binding': 0.833, 'result_release_control': 0.833, 'audit_trace_completeness': 0.75, 'trust_evidence_verification': 0.75, 'tenant_isolation': 0.75}
failed_cases= ['third_party_agent_not_allowed_bad', 'missing_identity_bad', 'obo_scope_expansion_bad', 'unapproved_agent_bad', 'context_forwarding_bad', 'write_action_no_confirmation_bad', 'tool_permission_mismatch_bad', 'audit_missing_bad', 'low_trust_unverified_result_bad', 'cross_tenant_leak_bad']
failed_gates= ['identity_chain_coverage', 'obo_scope_binding', 'delegation_allowlist', 'permission_attenuation', 'context_policy_enforcement', 'redelegation_control', 'high_risk_confirmation', 'mcp_tool_permission_binding', 'result_release_control', 'audit_trace_completeness', 'trust_evidence_verification', 'tenant_isolation']
cross_agent_security_gate_pass= False
```

这个 demo 的关键不是分数，而是失败切片。你能看到跨 Agent 安全问题往往不是一个单点错误，而是多个边界一起松动：scope 扩大时通常伴随数据粒度扩大、上下文未脱敏、工具越权、结果泄露和审计缺字段；跨租户错误则会同时破坏资源、Artifact、trace 和工具权限绑定。

## 30.18 面试高频题

### 题 1：多 Agent 系统里有哪些身份需要区分？

参考回答：

至少要区分用户身份、Agent 身份和服务身份。用户身份决定代表谁访问资源，Agent 身份决定哪个智能体发起或接收任务，服务身份决定实际运行实例是否可信。审计时需要同时记录这三层身份。

### 题 2：什么是 on-behalf-of 授权？

参考回答：

OBO 表示 Agent 代表用户执行任务。资源访问不应只按 Agent 的全局权限决定，而应结合用户身份、Agent 身份、任务授权和数据策略。这样可以避免 Agent 通过自己的高权限绕过用户原本的数据权限。

### 题 3：为什么跨 Agent 委派需要权限衰减？

参考回答：

因为委派不应该扩大权限。下游 Agent 获得的权限应小于或等于上游任务所需权限，并受用户授权限制。常见衰减包括从明细到聚合、从读写到只读、从全文到摘要、从可转发到不可转发。

### 题 4：A2A 审计应该记录什么？

参考回答：

应记录用户身份、发起 Agent、接收 Agent、任务 ID、消息 ID、上下文引用、数据分类、授权决策、状态变化、MCP 工具调用、人工确认、Artifact 和最终结果。关键安全事件要防篡改、可查询、可追责。

### 题 5：如何防止下游 Agent 继续把敏感数据转发出去？

参考回答：

通过 context_policy、继续委派权限、Agent allowlist、数据分类、访问过期和审计来控制。默认不允许自由继续委派；如需委派，只能给指定 Agent，必要时要求上游或用户确认，并记录完整 trace。

## 30.19 小练习

1. 设计一个 A2A 授权上下文，包含用户身份、调用 Agent、接收 Agent、task_id、scopes 和 constraints。
2. 为“机密合同审查”写一个 context_policy，要求禁止转发、Artifact 保留 7 天、只允许内部法务 Agent 接收。
3. 设计一个审计事件，记录一次从总控 Agent 到数据 Agent 的任务委派。
4. 列出哪些场景必须人工确认。
5. 思考：如果一个下游 Agent 需要调用另一个第三方 Agent 才能完成任务，系统应该如何决策？

## 30.20 本章小结

本章我们讲了跨 Agent 权限、身份、审计和信任模型。

多 Agent 系统必须区分用户身份、Agent 身份和服务身份。Agent 代表用户执行任务时，应使用 on-behalf-of 授权，结合用户权限、Agent 权限、任务权限和数据策略。委派过程中要遵守权限衰减和最小上下文原则，下游 Agent 不能自动获得上游的全部权限，也不能自由继续委派。

审计和 trace 是多 Agent 系统的基础设施。Trace 帮助工程排障，Audit 支撑安全、合规和责任追踪。信任模型也不能只看 Agent 是否可信，还要看数据、方法、结果和任务是否可信。

你可以把本章重点记成一句话：

> 多 Agent 系统真正危险的不是某个 Agent 单独犯错，而是身份、权限和上下文在委派链中失控。

下一章我们会继续讲多 Agent 系统的失败模式：循环、冲突和幻觉传播。
