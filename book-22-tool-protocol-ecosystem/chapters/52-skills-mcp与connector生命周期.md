# 第 52 章 Skills、MCP 与 Connector：能力接入不是一次安装

## 52.1 从“有一个工具”到“有一套能力”

单个函数可以直接注册到模型，但真实应用往往需要一组相关能力：读取仓库、搜索 issue、运行测试、查看部署状态和提交变更。用户看到的是一个 skill，runtime 看到的则是多个工具、资源、提示模板、权限和版本。

MCP、connector 和 skills 解决的问题不同。MCP 提供模型客户端与外部上下文/工具服务的协议化连接；connector 通常描述应用接入某个外部系统的认证、资源和生命周期；skill 是面向任务的能力封装。三者可以组合，但不能把它们当作同一个安全边界。

## 52.2 生命周期从发现开始

一个能力的生命周期可以写成：

```text
discover -> inspect manifest -> authorize
  -> install/connect -> negotiate capability
  -> invoke -> observe -> update/disable -> revoke
```

发现阶段只说明“有一个候选服务”，不能自动安装或授权。manifest 应包含名称、版本、提供方、工具和资源、认证方式、数据范围、网络需求、危险动作和兼容 runtime。

## 52.3 安装和连接的区别

安装可能改变本地文件、依赖和配置；连接可能创建 OAuth token、打开远程会话或访问企业数据。二者都需要审核和可回滚。一个 skill 的 markdown 说明不是权限授予，也不能因为用户选择了它就读取所有工作区。

connector 的凭证应该由 secret manager 管理，模型和普通工具结果只能看到脱敏引用。认证范围、租户、过期和撤销状态都要进入 trace。

## 52.4 capability negotiation

连接后，client 和 server 需要协商协议版本、tools、resources、prompts、sampling、streaming、分页、取消和认证能力。能力表可以写成：

```text
capability -> native / adapted / unsupported
schema version -> supported versions
auth -> scopes and expiry
limits -> rate, size, timeout
side effects -> read / idempotent / irreversible
```

如果 server 能列出一个 tool，不代表当前用户有权调用；如果 client 支持 streaming，不代表连接端到端支持取消。协商结果应该绑定 session 和 trace。

## 52.5 一个代码 connector

skill “修复仓库测试”可能依赖：读取文件 resource、搜索工具、测试工具、patch artifact 和提交审批。连接器先验证仓库、分支和用户权限；MCP server 暴露只读资源与工具；skill 提示模板规定工作流；harness 在提交前执行安全和测试验收条件。

如果用户只授权读取，skill 仍然可以生成 patch，但不能调用写入或发布工具。把“能力可见”与“动作可执行”分开，是生命周期设计的关键。

## 52.6 更新与兼容

工具 schema、资源 URL、认证 scope 和 prompt 模板都会变化。更新时先读取新 manifest，在 shadow 或只读环境中回放 golden calls，再逐步放量。旧版本连接不能因为新版本已安装就自动切换，尤其是外部副作用工具。

兼容性要区分 schema compatible、behavior compatible 和 policy compatible。字段仍然存在，不代表返回语义、错误、分页或安全范围没有改变。

## 52.7 撤销和失效

用户注销、token 过期、组织权限变化、server 漏洞或 connector 不再维护，都可能触发 revoke。撤销要阻止新调用，处理正在执行的请求，清理缓存凭证，保留必要审计，并明确外部动作是否已经提交。

禁用一个 skill 不一定删除已生成的 artifact；删除 connector 也不一定撤销外部服务中的 token。生命周期文档要把这些边界写清楚。

## 52.8 安全与提示注入

MCP resource、网页、issue 和文件都属于外部数据，可能包含提示注入。skill 的 system policy 和 connector 的权限不能被资源内容覆盖。工具执行器要做参数校验、路径限制、网络限制和敏感输出阻断。

connector 如果能访问邮件、云资源或代码仓库，风险不仅来自模型，还来自 credential scope、日志泄露和错误重试。最小权限和人工确认必须在 connector 层落地，而不是寄希望于 prompt。

## 52.9 评估生命周期

评估从发现到撤销都要有测试：manifest 解析、版本协商、认证失败、scope 变化、工具超时、schema 漂移、stream 取消、撤销中的请求和升级回滚。记录安装成功率、连接 p95、工具成功、权限拒绝、恢复、数据外泄和撤销延迟。

## 52.10 常见失败

把 skill 当权限；把 MCP server 当安全边界；安装即自动授权；只测 tool list 不测 execute；connector 版本没有锁定；凭证进入 prompt/trace；更新没有 shadow；撤销只改 UI 不撤 token；工具结果注入系统指令。

## 52.11 面试回答与练习

回答“Skills、MCP 和 connector 如何配合”时，可以说 skill 面向任务工作流，MCP 提供工具/资源/提示的协议化连接，connector 负责外部系统的认证、发现和生命周期；三者都要做 capability negotiation、权限、版本、审计、更新和撤销，不能把安装当授权。

练习一：设计一个代码 connector 的 manifest。

练习二：连接端发现 tool schema 版本不支持时如何降级？

练习三：设计 token revoke 期间已有工具调用的状态机。

### 52.11.1 能力接入的完整生命周期

一个外部能力从被发现到被撤销，至少经过：

```text
discover -> describe -> negotiate -> authorize
         -> execute -> observe -> update/revoke
```

发现到的 tool schema 只是能力描述，不是用户授权；skill 的任务说明也不是安全策略；connector 的 OAuth token 更不是永久权限。每一步都应有版本、租户、主体、有效期和审计记录。

### 52.11.2 三层职责的例子

Skill 可以描述“如何完成月度报表”，包括步骤、证据和验收；MCP server 可以暴露查询报表、读取指标和提交草稿等工具；connector 可以处理目标 SaaS 的认证、分页、限流和撤销。把三者混成一层，往往会让安装 skill 直接获得写权限。

一个更安全的链路是：skill 选择能力，MCP 返回 schema，policy 根据用户和动作决定是否允许，connector 只在获得授权后调用外部系统。每个层都要能单独升级和回滚。

### 52.11.3 worked example：token 撤销竞态

用户撤销 CRM connector token 时，已经排队的读取请求可能尚未执行。执行前要重新查询 token 状态；正在执行的请求要记录开始时的授权 revision；写操作如果已提交，撤销不能假装回滚，必须返回外部状态。

如果 connector 失败后自动换用另一份 token，可能越过用户刚刚做出的撤销决定。撤销是一个状态变更，应传播到 session、队列、cache、worker 和正在等待确认的工具调用。

### 52.11.4 版本与兼容

schema 版本要区分新增可选字段、语义变化和破坏性变化。调用方发现不兼容时，可以选择能力协商、只读降级、旧版本 connector 或人工，而不是发送未知参数。工具结果也要带 provider revision 和时间，避免长期 skill 把过期字段当成当前事实。

### 52.11.5 能力生命周期验收条件

一个能力能够被 Agent 使用，不仅要“发现得到”，还要完成描述、认证、协商、执行和撤销传播。可以把它写成一条生命周期验收条件：

```math
G_{\mathrm{lifecycle}}
=G_{\mathrm{discover}}G_{\mathrm{manifest}}G_{\mathrm{auth}}G_{\mathrm{negotiate}}G_{\mathrm{execute}}G_{\mathrm{revoke}}
```

`G_discover` 表示找到了可信的服务入口，`G_manifest` 表示工具 schema、版本和数据范围可解释，`G_auth` 表示当前主体获得了有效授权，`G_negotiate` 表示调用方和服务端对能力达成共同子集，`G_execute` 表示执行结果可追踪，`G_revoke` 表示撤销、过期和权限变化能够传播到队列与 worker。生命周期中任何一环缺失，都只能把能力当作不可用或只读观察源，不能因为工具名称出现在列表里就允许它产生副作用。

这个表达还区分了安装成功和运行成功。skill 文件可以已经安装，MCP server 也可以已经连接，但如果 token 已撤销或 schema 已不兼容，`G_auth` 或 `G_negotiate` 仍然为零；系统应在执行前拒绝，而不是等外部系统返回一个难以追溯的错误。

生命周期治理的工程取舍是接入速度与可控性之间的平衡：把发现、授权、协商和撤销都做成显式状态，会增加 manifest 维护、握手延迟、审计存储和版本兼容成本，却能让权限变化及时阻断外部副作用；若只追求“一键安装、立即调用”，短期接入更快，长期却容易留下无法撤销、无法归因或静默越权的能力。

## 52.12 connector 的兼容和撤销

能力接入不只看工具名称，还要检查 schema、认证、权限、版本、限流、错误码和数据范围。connector 撤销时，已发出的请求、缓存 token、正在运行的任务和长期记忆都要有清理策略。

升级应先做 capability negotiation 和 golden request，再灰度放量。新 schema 如果改变了参数语义，不能只靠向后兼容字段掩盖。

## 52.13 能力包的版本和来源

Skill、MCP server 和 connector 都可能携带说明、schema、脚本、参考文件和权限声明。安装时应记录来源、版本、依赖、hash、支持的模型和允许的资源范围。一个名称相同的能力包，更新后可能改变工具描述和执行行为。

模型读取 skill 文档时，仍要把外部内容视为不可信数据。权限、系统策略和用户目标不能被普通 reference file 覆盖。

## 52.14 生命周期和兼容性

能力接入经历发现、安装、配置、调用、升级、禁用和删除。升级前要做 schema diff、回归、权限复核和灰度；connector 失效时要有明确错误和替代路径；删除时清理缓存、凭证、索引和 trace 中的敏感字段。

MCP 或其他协议的兼容性不仅是 JSON schema 相同，还包括超时、分页、取消、流式事件、错误码、权限和副作用语义。接口能连通不代表生产行为兼容。

## 52.15 生态位会随模型能力迁移

Skill 可能承载可复用工作流，connector 负责外部系统连接，MCP 提供协议边界，harness 负责状态和权限。随着模型记忆、上下文和内置工具能力增强，部分 skill 可能退化为普通参考资料；稳定的权限、审计和跨供应商接口则更可能保留为基础设施。

评估新生态技术时要问它解决的约束是否仍存在、是否被模型能力吸收、迁移成本多大，以及失败时能否安全回退。热点名称本身不是长期价值证明。

## 52.16 能力连接的状态机

生命周期不是一串 UI 按钮，而是一组需要持久化的状态。一个 connector session 可以经历：

```text
discovered -> inspected -> awaiting_auth -> authorized
           -> negotiating -> ready -> degraded
           -> draining -> revoked | failed
```

`ready` 只表示协议和授权在当前 snapshot 下满足，不表示每个工具都可写；`degraded` 表示某些工具、分页或流式能力不可用，仍可能允许只读调用；`draining` 表示阻止新副作用，但让可安全结束的读取完成。状态转移需要事件和版本，例如 token 过期、scope 改变、server 升级、策略撤销和租户注销。

同一个 MCP server 可以被多个 tenant 使用，因此 session 的 `server_revision`、`tenant_id`、`principal`、`auth_revision` 和 `capability_snapshot` 不能省略。只在全局缓存中保存“server healthy”，会把一个租户的授权错误误认为所有调用都可用。

## 52.17 Manifest、来源和供应链验证

能力包的 manifest 不应只是名称和工具列表。安装或连接前要验证来源、版本、内容 hash、依赖、网络域、文件范围、凭证 scope、危险动作和更新策略。skill 的说明文档、MCP resource、connector 配置和脚本都可能被替换，因此要保存签名或可信来源证明，并在更新时重新检查。

一个简化的能力描述是：

```text
CapabilityManifest {
  name, provider, version, content_hash
  protocol_versions, tool_schemas, resource_scopes
  auth_scopes, network_domains, side_effects
  dependencies, expiration, rollback_version
}
```

`content_hash` 只能证明拿到的内容与登记值一致，不能证明提供方本身可信；仍需结合签名、发布渠道、审计和沙箱。相同名称但不同来源的能力包必须视为不同对象，不能因为 UI 显示名相同就复用授权。

## 52.18 撤销传播与在途请求

撤销是分布式状态变更，存在从用户点击撤销到所有 worker 收到事件的窗口。安全做法是给授权分配单调 `auth_revision`，每次执行前和关键提交前都检查：

```math
G_{\mathrm{auth}}
=\mathbf{1}[r_{\mathrm{request}}\ge r_{\mathrm{required}}]
```

撤销后，队列中的新请求直接拒绝；已经开始的只读请求可以按策略完成；写入请求在提交前必须再次确认授权 revision。若外部动作已经提交，系统不能把撤销描述成回滚，而要记录 effect id、外部结果和补偿路径。

缓存是常见漏洞。清理 token 只清理 secret manager 里的主凭证还不够，连接池、worker 内存、代理缓存、日志、checkpoint 和模型上下文中的引用都要失效。trace 可以保留不可逆审计事实，但不应保留可重放的秘密。

## 52.19 升级、灰度与兼容回退

能力更新应先做 manifest diff，区分新增可选字段、默认值变化、返回语义变化和破坏性删除。之后在 shadow session 或只读租户中回放 golden calls，检查 schema、错误、分页、取消、延迟、权限和数据覆盖。

升级发布可以按以下路径推进：

```text
new manifest
  -> schema/policy review
  -> shadow replay
  -> read-only canary
  -> limited write canary
  -> tenant rollout
  -> rollback or commit
```

回滚不仅是把 server 地址改回旧版本，还要恢复旧 tool schema、connector adapter、auth scope、skill prompt 和 verifier。新版本已经写入的 artifact、缓存和外部副作用不能被版本回滚抹掉，需要单独标记和审计。

## 52.20 Skill、MCP server 和 connector 的层次

Skill 更像可复用的工作流程和知识组织，可能包含说明、参考资料和脚本；MCP server 提供工具、资源或 prompt 的协议接口；connector 则把具体外部系统接入并承载身份、权限和数据范围。三者可以组合，但不应把它们当成同一个安装包或同一种安全边界。

一个 skill 可以调用多个 MCP server，一个 connector 也可能暴露多个工具。安装 skill 不应自动授权 connector，连接器升级也不应改变 skill 的输入输出契约。能力目录需要记录 owner、来源、版本、依赖、权限和撤销方式。

## 52.21 生命周期状态机

一个可审计的状态机可以是：`discovered -> reviewed -> installed -> enabled -> canary -> active -> suspended -> revoked`。每次状态变化绑定 artifact hash、manifest、权限审批、兼容测试和生效范围。`suspended` 表示暂时停止新调用但保留调查信息，`revoked` 则还要处理缓存、在途请求和已生成的凭证。

## 52.22 能力漂移和生态位迁移

模型能力增强后，原本需要复杂 skill 的步骤可能被内置工具、记忆或更强的规划能力吸收；反过来，企业合规、领域流程和外部系统授权仍需要显式 connector。判断一个生态技术是否仍有价值，要看它解决的约束是否仍存在，而不是只看产品页面是否继续使用这个名字。

## 52.23 租户隔离和撤销传播

skill manifest、MCP schema、connector token、缓存结果和 workspace 状态都要带租户与版本。撤销时应阻止新调用、取消未授权的排队任务、使凭证失效、清理可删除缓存并审计在途请求。只从 UI 隐藏一个 skill，不等于 runtime 已经撤销它。

## 52.24 生态组件的兼容性测试

升级 skill、MCP server 或 connector 时，测试 manifest/schema、参数校验、权限、错误、流式事件、缓存、撤销和已有 workspace。新组件能安装不代表旧工作流仍然正确；尤其要检查同名工具、默认参数和数据范围是否发生变化。

## 52.25 被模型能力吸收后的迁移

当模型内置了更强的规划、记忆或工具编排，原有 skill 可能只剩领域约束、审批和参考资料。迁移时应删除重复 prompt、缩小权限和输入范围，并保留旧版本回滚；不能为了保留生态组件而继续注入冗余上下文。

## 52.26 生命周期评估与事故演练

评估指标要覆盖全生命周期：发现到 ready 的时间、认证失败率、能力协商失败率、工具成功率、schema 漂移检测率、撤销传播延迟、在途副作用数、跨租户拒绝率、升级回滚时间和凭证残留率。只测一次 tool call 成功，无法说明连接器在 token 过期或 server 升级后仍然安全。

事故演练可以注入：manifest 被替换、token 在排队期间撤销、server 返回未知字段、分页重复、流式取消丢失、worker 使用旧缓存、写工具在撤销前后竞态，以及租户权限缩小。演练的成功标准是新副作用被阻断、已发生副作用可追踪、敏感凭证不可再用、恢复或人工路径清晰。

能力接入是一个生命周期：发现、协商、授权、执行、更新和撤销。MCP、skills 和 connectors 可以互相组合，但每层的职责和安全边界必须明确。协议可用不代表用户有权调用，工具可见不代表动作已授权；真正的生产能力还要通过来源验证、版本灰度、撤销传播和事故回放。

具体 MCP 版本、connector 字段和 skill 机制以官方协议和产品文档为准；本章的生命周期模型、状态机和验收条件用于迁移和系统设计。
