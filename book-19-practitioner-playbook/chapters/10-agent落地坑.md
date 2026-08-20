# 第十章：Agent 落地坑：从“能调用工具”到“可恢复执行”

Agent 的 demo 很容易让人产生一种错觉：模型能够规划、调用工具、读取结果并继续行动，于是任务就已经自动化了。生产环境会立刻揭开这层错觉。工具参数可能指向错误用户，工具返回 `failed` 可能被模型当成 `success`，网页中的一段文字可能诱导 Agent 外发数据，循环检索可能烧尽预算，执行到一半的任务可能没有任何可恢复状态。

Agent 不是一段更长的 prompt，而是一个会读取外部信息、调用外部能力、改变外部状态的控制系统。模型是其中的决策组件，真正的安全和可靠性还依赖工具 registry、schema、权限服务、状态存储、预算控制、幂等机制、人工确认、trace 和业务后端。

本章从一次 Agent 事故出发，完整讨论计划可执行性、工具选择、参数和业务校验、observation、状态更新、checkpoint、循环停止、预算、prompt injection、高风险确认、最小权限、失败恢复、错误重试、可观测性和分层评估。公式明确动作、任务、风险和工具结果的定义域；代码会区分“没有高风险动作”“没有审计记录”“工具失败”“任务尚未完成”和真正的成功。

## 10.0 先看一场“已经完成”的事故

某公司让 Agent 帮员工提交报销。Agent 成功创建了草稿，但提交工具返回了 `failed`。模型没有读取失败状态，最后向用户说“报销已提交”。业务后台没有提交记录，用户却以为任务已经完成。

同一天，另一个 Agent 根据用户请求修改订单地址。模型生成的 JSON 完全符合 schema，但订单属于另一位用户；工具层没有做 owner 校验，直到人工发现错误才停止。还有一个网页检索 Agent 读到外部页面中的指令，尝试调用发送消息工具；权限层阻断了动作，但 trace 中没有记录注入边界，团队无法判断这是一次偶发错误还是系统性风险。

这三件事看起来都可以被称为“模型判断错误”，但修复位置完全不同：

```text
报销：工具状态与最终任务状态没有绑定
订单：业务参数和权限没有在工具层强校验
网页：不可信 observation 被当成高优先级指令
```

对初学者来说，Agent 像一个会办事的助理：它不能只说“我做了”，还要有系统记录证明动作真的完成。对专家来说，Agent 是带状态转移和外部副作用的执行系统，文本输出只是观察窗口，业务后端状态才是任务真值的一部分。

## 10.1 先区分 Chatbot、Workflow 和 Agent

不是所有调用模型的应用都需要 Agent。

Chatbot 主要生成回复；Workflow 的步骤和分支由程序预先规定；Agent 允许模型在受约束的工具集合和状态空间内选择下一步。三者可以组合，但可靠性和测试方法不同。

```text
固定步骤、固定输入输出、失败分支已知：优先 Workflow
需要自然语言理解但动作固定：模型负责抽取，程序负责执行
需要在多个工具和未知结果中选择路径：才考虑受控 Agent
```

Agent 的自由度越大，状态、预算、权限、观测和恢复的工程成本越高。把一个本可以由三步确定性代码完成的任务交给开放循环，往往同时增加延迟、成本和失败面。

## 10.2 任务合同：模型能做什么，不能做什么

Agent 开始运行前，应把用户目标编译成任务合同：

```text
goal：要完成的业务目标
scope：允许读取和修改的对象、租户和时间范围
success：业务后端如何证明任务完成
required_inputs：缺什么信息必须追问
available_tools：本次任务允许使用的工具
side_effects：哪些动作会改变外部状态
confirmation：哪些动作需要用户或人工确认
budget：步数、模型调用、工具调用、token、成本和时间
stop：成功、失败、阻断、待确认、预算耗尽和未知状态
```

“帮我处理订单”不是可执行目标。系统至少要知道订单 ID、允许的操作、用户身份、是否能修改地址、成功判据和确认要求。缺少这些字段时，Agent 应该进入 `need_input` 或 `need_confirmation`，而不是猜测。

### 10.2.1 成功必须来自外部事实

最终自然语言 `completed` 不是任务成功的充分证据。任务成功应由业务系统、确定性验证器或人工确认给出。例如：

```text
创建草稿成功 != 提交成功
工具返回 accepted != 外部系统最终落库
发送请求成功 != 对方服务处理完成
代码生成成功 != 测试和审查通过
```

Agent 的最终回复应引用真实状态：`success`、`pending`、`failed`、`blocked` 或 `unknown`。如果外部系统只返回异步任务 ID，就应告诉用户仍在处理中，而不是提前宣称完成。

## 10.3 Agent Loop 是一个状态机

一次受控执行通常经过：

```text
目标解析 -> 计划/下一动作 -> 工具选择 -> 参数生成
-> schema 校验 -> 业务校验 -> 权限/确认 -> 工具执行
-> observation 解析 -> 状态更新 -> 停止或继续
```

把第 `i` 个任务写成：

$$
\tau_i=(g_i,P_i,A_i,O_i,S_i,H_i,B_i,Y_i,T_i)
$$

其中 `g_i` 是目标，`P_i` 是计划，`A_i` 是动作序列，`O_i` 是工具 observation，`S_i` 是持久状态，`H_i` 是权限和确认检查，`B_i` 是预算消耗，`Y_i` 是业务后端状态，`T_i` 是 trace。

每次循环都应产生状态差异：动作前读取当前状态，动作后记录结果，再决定下一步。只把所有文本拼进上下文，无法可靠处理长任务、重试、并发、取消和恢复。

### 10.3.1 状态至少要保存什么

```text
原始目标、解析后的实体和约束
当前计划、已完成步骤、待执行步骤
每次工具调用、参数、权限、结果和错误
用户确认、风险等级、幂等键和外部任务 ID
已知事实、未知状态、冲突和需要追问的信息
预算消耗、重试次数、停止原因和版本
```

状态更新不是日志的副作用，而是执行协议的一部分。若下一次恢复只能让模型从聊天记录猜“已经做到了哪一步”，任务就不具备可恢复性。

## 10.4 计划看起来合理，不代表可执行

模型可能生成语言上流畅的计划，但计划引用了不存在的工具、违反依赖顺序、忽略权限或没有失败分支。

例如：

```text
目标：分析本月销售异常并通知负责人
模型计划：查销售数据 -> 分析 -> 找负责人 -> 发邮件
系统现实：只有订单查询工具，没有负责人目录和发邮件权限
```

计划验证至少检查：

```text
每个动作是否映射到本次允许的工具
动作顺序是否满足数据依赖和状态前置条件
所需参数是否已知，缺失时是否会追问
动作是否需要权限或确认
预算和时间是否足够
失败、超时、空结果和人工接管是否有路径
```

如果计划包含不可执行步骤，系统应在工具调用前返回缺失能力或改为草稿，而不是让模型继续编造工具结果。

### 10.4.1 计划可执行性不是集合包含那么简单

如果计划动作序列为 `P_i=(p_1,...,p_n)`，可用动作集合为 `T_i`，至少需要 `p_j in T_i` 对所有 `j` 成立；但这还不够。还要检查依赖图、参数来源、权限和资源预算。重复调用同一工具可能是合法重试，也可能是循环，需要由状态和预算判断。

计划评估应保留第一处不可执行原因：`unknown_tool`、`missing_input`、`dependency_order`、`permission`、`budget` 或 `no_failure_path`。只输出一个 0/1，无法指导修复。

## 10.5 工具 registry 和选择策略

工具 registry 是 Agent 的能力边界。每个工具应包含名称、用途、不适用场景、输入 schema、输出 schema、权限、风险、幂等性、超时和副作用说明。

### 10.5.1 工具描述要避免语义重叠

多个工具都叫“查询”“更新”“搜索”，模型很容易选择错误对象。名称应包含领域和动作，例如 `get_order_status`、`update_order_address`、`create_expense_draft`、`submit_expense`。描述中同时写适用和不适用场景，避免只写营销式简介。

```text
get_order_status：只读，返回订单当前状态；不能修改订单
update_order_address：修改当前用户有权访问的订单地址；需要确认
submit_expense：提交已校验的报销草稿；失败时不代表提交成功
```

### 10.5.2 工具选择是可以独立评估的模块

对第 `m` 个动作，设人工或规则定义的允许工具集合为 `T_m^star`，模型选择为 `a_m`。在 `M>0` 个有标签动作上：

$$
A_{tool}=\frac{\sum_{m=1}^{M}\mathbf{1}[a_m\in T_m^\star]}{M}
$$

如果没有工具选择标注，结果是 `unknown`；如果任务本来不需要工具，指标是 `not_applicable`。不能把“最终任务成功”反推成每一次工具选择都正确，成功可能来自重试或人工补救。

## 10.6 Schema 合法不等于业务合法

工具 schema 解决结构问题，业务校验解决语义问题。一个 JSON 可以完全符合 schema，却把 A 用户的订单 ID 和 B 用户的地址组合起来。

### 10.6.1 多层参数检查

工具调用应至少经过：

```text
解析：是否是合法结构化数据
schema：类型、required、enum、格式、范围和单位
实体：ID 是否存在，日期/金额/时区是否合理
业务：对象是否属于当前用户，状态是否允许该动作
权限：租户、角色、资源和字段级权限
风险：是否需要预览、确认、幂等键或人工审批
```

金额要带币种和精度，日期要带时区和范围，ID 要做归属检查，枚举值不能靠字符串相似度猜测。工具层必须拒绝非法参数；让模型“自己注意”不是安全控制。

### 10.6.2 参数有效率的定义域

设 `u_m` 是第 `m` 次工具调用，`schema(u_m)` 和 `business(u_m)` 分别表示结构与业务检查通过。在 `M>0` 个有记录调用上：

$$
A_{arg}=\frac{\sum_{m=1}^{M}\mathbf{1}[schema(u_m)=1\land business(u_m)=1]}{M}
$$

如果没有工具调用，参数有效率是 `not_applicable`；如果参数没有被记录，状态是 `unknown`。没有调用工具不能被记作 100% 参数正确。

## 10.7 权限、确认和最小授权

Agent 只能请求工具，不能替代权限系统。真正的权限判断必须在执行层，且应绑定用户、租户、资源、环境和权限版本。

### 10.7.1 读写分离和资源归属

```text
只读查询：可以按用户权限返回数据
草稿动作：可以创建可撤销的中间状态
外部写入：需要业务归属校验和幂等键
不可逆动作：需要人工或用户确认，并保留审计
```

不要让所有工具共享一个高权限 service token。测试环境和生产环境要隔离，跨租户访问要硬阻断，读工具和写工具要有不同的凭证和策略。

### 10.7.2 高风险确认不是一句“确定吗”

确认前要展示动作摘要、影响对象、关键参数、范围、不可逆性和失败处理。例如向 128 位客户发邮件时，用户应看到收件人集合、主题、正文版本和是否可撤回，而不是只看到一个模糊按钮。

设高风险动作集合 `H`，其中有明确确认的动作数为 `C_H`。当 `|H|>0` 时：

$$
C_{confirm}=\frac{C_H}{\lvert H\rvert}
$$

当本次没有高风险动作时，该指标是 `not_applicable`，不是自动写成 1；如果高风险识别或确认记录缺失，状态是 `unknown`。没有高风险动作和高风险动作全部确认是两个不同事实。

## 10.8 工具执行状态和 observation

工具调用的生命周期至少要区分：

```text
accepted：请求已被外部系统接收，但可能尚未完成
success：业务后端确认完成
empty：执行成功但没有结果
failed：明确失败
blocked：权限、策略或确认阻断
timeout：在时限内没有结果
unknown：状态无法确认
```

模型不能把 `accepted`、`empty` 或 `pending` 自动改写成 `success`。工具返回应该是结构化对象，例如：

```json
{
  "status": "failed",
  "data": null,
  "error_code": "ORDER_NOT_FOUND",
  "retryable": false,
  "external_id": null
}
```

### 10.8.1 observation 是数据，不是系统指令

工具返回的网页、邮件、工单、文档和用户生成文本都可能包含要求模型改变规则的内容。observation 应带来源、可信级别、字段和安全标签，模型只能把它作为待分析数据。

设有 `M_o>0` 个包含可用 observation 的动作，`u_m=1` 表示 observation 确实影响了下一步状态或决策：

$$
R_{obs}=\frac{\sum_{m=1}^{M_o}u_m}{M_o}
$$

没有 observation 的动作不应进入该分母；如果 observation 存在但后续状态没有记录，不能假设模型使用了它。对空结果，正确使用可能是更新为“未找到”，而不是继续搜索相同 query。

### 10.8.2 工具结果和最终状态必须绑定

最终状态 `completed` 只有在业务后端确认成功时才合法。可以用状态机限制：

```text
tool_failed -> completed：拒绝
tool_pending -> completed：拒绝
permission_blocked -> completed：拒绝
unknown_external_state -> completed：拒绝，转人工或查询
tool_success -> completed：仍需检查任务验收条件
```

这条约束应该在编排器或业务服务中执行，而不是只写在 system prompt 中。

## 10.9 状态、checkpoint 和长任务恢复

长任务要把每个成功动作和外部 ID 持久化。checkpoint 至少包含状态版本、已完成步骤、待执行步骤、工具结果摘要、确认记录、预算和幂等键。

### 10.9.1 用户改变目标时重新规划

用户中途说“不要发邮件了，只生成草稿”，旧计划不能继续执行发送动作。系统应将新目标写入状态，标记旧计划失效，并重新计算待执行步骤和权限确认。

### 10.9.2 恢复不能重复副作用

如果服务在外部写入成功后崩溃，重启时不能仅凭“上一步没有本地结果”再次写入。需要幂等键、外部任务 ID 或查询接口确认状态：

```text
写入前生成幂等键
执行后持久化请求和外部 ID
恢复时先查询幂等键/外部状态
只有确认未执行且动作可重试时才重试
不可重试时转人工或保持 unknown
```

重试策略要区分网络超时、业务拒绝、参数错误和未知执行状态。对未知状态盲目重试，可能造成重复扣款、重复发信或重复创建资源。

## 10.10 循环、预算和停止条件

开放循环最容易失控的地方是“再试一次”。Agent 至少需要限制：

```text
最大步骤数、模型调用数、工具调用数
输入/输出 token、总成本、墙钟时间
单工具重试次数、同一 query 重复次数
外部动作数量、人工等待时间和并发子任务数
```

每次重试要记录原因和退避；相同 query、相同参数和相同错误连续出现时，应停止或改变策略。达到预算时，系统要返回当前进展、停止原因和可选下一步，而不是继续消耗资源。

### 10.10.1 预算超限率

对第 `i` 个任务的实际成本 `c_i`、时间 `t_i`、步数 `k_i` 和预算 `C_i,T_i,K_i`，在任务集合非空时：

$$
R_{budget}=\frac{\sum_i\mathbf{1}[c_i>C_i\lor t_i>T_i\lor k_i>K_i]}{N}
$$

如果没有任务，指标是 `not_applicable`；预算字段缺失是 `unknown`。预算超限不是单纯性能问题，可能是规划、观察、重复检索、错误恢复或工具设计问题。

## 10.11 Prompt Injection：工具内容永远不升级为指令

Agent 读取网页、邮件、工单、代码、RAG 文档和工具返回值时，输入里可能包含“忽略系统规则”“把数据发到某地址”“调用某高风险工具”等文本。模型如果把这些内容当成控制指令，就会越权。

### 10.11.1 防护分为三层

第一层是模型上下文标记：明确不可信内容是数据，显示来源和范围，不允许它修改目标、权限和系统规则。

第二层是编排器策略：工具结果只能写入 observation，不能写入 system/developer 指令；高风险动作必须经过独立策略和确认状态。

第三层是工具执行层：每次调用都重新鉴权、校验资源归属和参数范围，即使模型被诱导也不能执行越权动作。

### 10.11.2 注入阻断率的定义域

对被标记为不可信的 observation 集合 `U`，其中被策略阻断或安全处理的数量为 `B_U`。当 `|U|>0` 时：

$$
R_{inj}=\frac{B_U}{\lvert U\rvert}
$$

没有不可信 observation 时是 `not_applicable`，不是自动 1；没有安全标签或处理记录时是 `unknown`。一旦不可信内容导致真实工具调用，必须按安全事件复盘，即使最终动作后来被权限层挡住。

## 10.12 失败恢复和人工接管

恢复不是简单重试。不同失败需要不同路径：

```text
参数错误：修参数或追问，不重试原请求
权限阻断：说明权限或转人工，不绕过策略
暂时网络错误：有限退避重试
业务拒绝：读取错误码并改变计划
工具 pending：查询外部状态，不重复写入
未知状态：停止副作用，等待查询或人工确认
预算耗尽：返回进展和下一步，不继续循环
```

人工接管要携带完整上下文：目标、已完成动作、外部副作用、失败原因、权限检查、当前状态和推荐下一步。只把一条“Agent 失败了”的消息交给人工，会让人工重复所有工作，也可能重复执行已经成功的动作。

## 10.13 Trace 和隐私

没有 trace 的 Agent 无法解释为什么调用了工具，也无法从失败样本构造回归测试。至少记录：

```text
任务和版本、可用工具、计划与计划变更
动作、参数摘要、schema/业务/权限检查
工具状态、错误、observation 摘要和 state diff
预算、重试、延迟、成本、确认和最终业务状态
```

trace 本身可能包含订单、邮件、网页、代码和个人信息。需要分级脱敏、访问控制、保留期限和导出审计。调试方便不能成为复制全量敏感数据的理由。

## 10.14 Agent 评估分层

### 10.14.1 任务层

任务成功要由业务后端或验证器判断。除了 success rate，还要看 false completion、人工接管、完成时间、用户采纳和任务成本。

### 10.14.2 工具层

单独评估工具选择、schema 参数、业务参数、权限、执行状态、幂等和返回结果解析。工具调用 JSON 合法但写错资源，不能算成功。

### 10.14.3 过程层

评估计划可执行性、步数效率、重复动作、observation 使用、state update、checkpoint、恢复和停止原因。过程指标能解释为什么两个任务最终都失败，但修复方向不同。

### 10.14.4 安全和成本层

评估越权动作、注入成功、敏感数据暴露、高风险确认、预算超限、重试、P95/P99 延迟和单位成功任务成本。一个任务成功率很高但依赖无限重试，不是可靠的 Agent。

## 10.15 典型事故：工具失败但 Agent 声称完成

现象是用户收到“已提交”，后台没有提交记录。排查应先看执行状态和业务后端，而不是先改语言风格。

```text
工具调用是否发出
schema/业务/权限是否通过
工具返回 success、failed、pending 还是 unknown
外部系统是否有最终记录
state 是否写入真实状态
最终回复是否由状态机授权
```

修复通常包括标准化工具状态、把完成回复绑定到 `backend_status=success`、为 pending 提供查询路径、为失败提供原因和下一步，并把这个样本加入回归集。

## 10.16 典型事故：结构合法但更新了错误资源

订单地址事故说明 schema validation 不够。`order_id` 是合法字符串，`address` 也是合法对象，但业务关系可能错误。工具层必须验证当前用户是否拥有订单、订单是否允许修改、地址是否在合法范围、动作是否需要确认，以及重复请求是否幂等。

这类动作要采用 preview/commit 分离：先返回将要修改的对象和差异，用户确认后再写入；写入后返回版本号和审计记录；失败或取消时保留明确状态。

## 10.17 关键公式与状态语义

### 10.17.1 任务成功和误报完成

设 `Y_i` 是后端真实任务状态，`hat Y_i` 是 Agent 对外声称的状态。在 `N>0` 个有业务验收结果的任务上：

$$
R_{success}=\frac{\sum_i\mathbf{1}[Y_i=completed]}{N}
$$

$$
R_{false\_complete}=\frac{\sum_i\mathbf{1}[\hat Y_i=completed\land Y_i\ne completed]}{N}
$$

如果没有后端验收结果，成功率和误报完成率都是 `unknown`，不能用最终文本猜测。`Y_i=pending` 不应被纳入 completed。

### 10.17.2 工具执行成功

对 `M>0` 个工具动作，只有后端确认的 `success` 才计入：

$$
R_{exec}=\frac{\sum_{m=1}^{M}\mathbf{1}[status_m=success]}{M}
$$

`blocked`、`failed`、`pending`、`timeout` 和 `unknown` 都应保持自己的状态，并在错误分析中分开。

### 10.17.3 未授权动作

设 `allow_m=1` 表示执行层允许第 `m` 个动作。在有动作记录且 `M>0` 时：

$$
R_{unauth}=\frac{\sum_{m=1}^{M}\mathbf{1}[allow_m=0]}{M}
$$

没有动作是 `not_applicable`；没有权限记录是 `unknown`。未授权动作被阻断和未授权动作实际执行是不同严重级别，但都要保留事件。

### 10.17.4 Trace 完整性

若任务包含 `K_i>0` 个预期步骤，已记录完整的步骤数为 `L_i`：

$$
C_{trace,i}=\frac{L_i}{K_i}
$$

任务没有预期步骤或 trace schema 不适用时是 `not_applicable`；系统承诺记录但字段缺失时是 `unknown`。空 trace 不能自动算 100% 完整。

## 10.18 最小可运行 Agent 事故审计

下面的 Python 示例只使用标准库，构造五种典型任务：正常销售报告、报销误报完成、错误订单归属、网页结果注入和循环搜索超预算。

### 10.18.1 数据和校验

```python
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Optional


@dataclass(frozen=True)
class Metric:
    value: Optional[float]
    status: str
    reason: str = ""


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def safe_mean(values: Iterable[float], *, empty_status="not_applicable"):
    values = list(values)
    if not values:
        return Metric(None, empty_status, "empty sample set")
    checked = [finite_number(value, "mean.value") for value in values]
    return Metric(sum(checked) / len(checked), "valid")


def safe_ratio(numerator, denominator, *, name="ratio", empty_status="not_applicable"):
    numerator = finite_number(numerator, f"{name}.numerator")
    denominator = finite_number(denominator, f"{name}.denominator")
    if numerator < 0 or denominator < 0:
        raise ValueError(f"{name} cannot be negative")
    if denominator == 0:
        if numerator == 0:
            return Metric(None, empty_status, f"{name} has no denominator")
        raise ValueError(f"{name} has positive numerator and zero denominator")
    return Metric(numerator / denominator, "valid")


def percentile(values, percentage):
    values = [finite_number(value, "percentile.value") for value in values]
    percentage = finite_number(percentage, "percentile.percentage")
    if not values:
        return Metric(None, "not_applicable", "empty sample set")
    if not 0 <= percentage <= 100:
        raise ValueError("percentage must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return Metric(value, "valid")


tasks = [
    {
        "id": "sales_report_ok",
        "expected_tools": ["query_sales", "summarize_findings"],
        "plan_steps": ["query_sales", "summarize_findings"],
        "steps": [
            {"tool": "query_sales", "schema_ok": True, "business_ok": True, "permission": True, "status": "success", "observation_available": True, "observation_used": True, "state_updated": True, "trace": True, "risk": "read", "confirmed": True},
            {"tool": "summarize_findings", "schema_ok": True, "business_ok": True, "permission": True, "status": "success", "observation_available": True, "observation_used": True, "state_updated": True, "trace": True, "risk": "read", "confirmed": True},
        ],
        "untrusted_observation": False, "injection_blocked": True,
        "budget": {"max_steps": 4, "max_cost": 0.08, "max_latency_ms": 3000},
        "actual": {"steps": 2, "cost": 0.032, "latency_ms": 1200},
        "recovery_needed": False, "recovered": True,
        "declared_status": "completed", "backend_status": "completed",
        "task_success": True,
    },
    {
        "id": "expense_false_done",
        "expected_tools": ["create_expense", "submit_expense"],
        "plan_steps": ["create_expense", "submit_expense"],
        "steps": [
            {"tool": "create_expense", "schema_ok": True, "business_ok": True, "permission": True, "status": "success", "observation_available": True, "observation_used": True, "state_updated": True, "trace": True, "risk": "write", "confirmed": True},
            {"tool": "submit_expense", "schema_ok": True, "business_ok": True, "permission": True, "status": "failed", "observation_available": True, "observation_used": False, "state_updated": False, "trace": True, "risk": "write", "confirmed": True},
        ],
        "untrusted_observation": False, "injection_blocked": True,
        "budget": {"max_steps": 4, "max_cost": 0.08, "max_latency_ms": 3000},
        "actual": {"steps": 2, "cost": 0.041, "latency_ms": 1500},
        "recovery_needed": True, "recovered": False,
        "declared_status": "completed", "backend_status": "failed",
        "task_success": False,
    },
    {
        "id": "address_wrong_owner",
        "expected_tools": ["lookup_order", "update_address"],
        "plan_steps": ["lookup_order", "update_address"],
        "steps": [
            {"tool": "lookup_order", "schema_ok": True, "business_ok": True, "permission": True, "status": "success", "observation_available": True, "observation_used": True, "state_updated": True, "trace": True, "risk": "read", "confirmed": True},
            {"tool": "update_address", "schema_ok": True, "business_ok": False, "permission": False, "status": "blocked", "observation_available": True, "observation_used": True, "state_updated": False, "trace": True, "risk": "high", "confirmed": False},
        ],
        "untrusted_observation": False, "injection_blocked": True,
        "budget": {"max_steps": 5, "max_cost": 0.12, "max_latency_ms": 4000},
        "actual": {"steps": 2, "cost": 0.052, "latency_ms": 1800},
        "recovery_needed": True, "recovered": False,
        "declared_status": "blocked", "backend_status": "blocked",
        "task_success": False,
    },
    {
        "id": "web_lookup_injection",
        "expected_tools": ["web_lookup", "summarize_findings"],
        "plan_steps": ["web_lookup", "summarize_findings"],
        "steps": [
            {"tool": "web_lookup", "schema_ok": True, "business_ok": True, "permission": True, "status": "success", "observation_available": True, "observation_used": True, "state_updated": True, "trace": True, "risk": "read", "confirmed": True},
            {"tool": "send_external_message", "schema_ok": True, "business_ok": False, "permission": False, "status": "blocked", "observation_available": True, "observation_used": False, "state_updated": False, "trace": True, "risk": "high", "confirmed": False},
        ],
        "untrusted_observation": True, "injection_blocked": False,
        "budget": {"max_steps": 4, "max_cost": 0.10, "max_latency_ms": 3500},
        "actual": {"steps": 2, "cost": 0.067, "latency_ms": 2200},
        "recovery_needed": True, "recovered": False,
        "declared_status": "blocked", "backend_status": "blocked",
        "task_success": False,
    },
    {
        "id": "looping_search_budget",
        "expected_tools": ["search_kb", "summarize_findings"],
        "plan_steps": ["search_kb", "search_kb", "search_kb", "summarize_findings"],
        "steps": [
            {"tool": "search_kb", "schema_ok": True, "business_ok": True, "permission": True, "status": "empty", "observation_available": True, "observation_used": False, "state_updated": False, "trace": True, "risk": "read", "confirmed": True},
            {"tool": "search_kb", "schema_ok": True, "business_ok": True, "permission": True, "status": "empty", "observation_available": True, "observation_used": False, "state_updated": False, "trace": True, "risk": "read", "confirmed": True},
            {"tool": "search_kb", "schema_ok": True, "business_ok": True, "permission": True, "status": "empty", "observation_available": True, "observation_used": False, "state_updated": False, "trace": False, "risk": "read", "confirmed": True},
            {"tool": "search_kb", "schema_ok": True, "business_ok": True, "permission": True, "status": "empty", "observation_available": True, "observation_used": False, "state_updated": False, "trace": False, "risk": "read", "confirmed": True},
        ],
        "untrusted_observation": False, "injection_blocked": True,
        "budget": {"max_steps": 3, "max_cost": 0.06, "max_latency_ms": 2500},
        "actual": {"steps": 4, "cost": 0.093, "latency_ms": 4300},
        "recovery_needed": True, "recovered": False,
        "declared_status": "stopped", "backend_status": "not_completed",
        "task_success": False,
    },
]


def validate_task(task):
    if not isinstance(task["id"], str) or not task["id"]:
        raise ValueError("task id must be a non-empty string")
    if type(task["task_success"]) is not bool:
        raise ValueError("task_success must be a real bool")
    if type(task["untrusted_observation"]) is not bool or type(task["injection_blocked"]) is not bool:
        raise ValueError("injection flags must be real bools")
    if type(task["recovery_needed"]) is not bool or type(task["recovered"]) is not bool:
        raise ValueError("recovery flags must be real bools")
    if not task["expected_tools"] or not task["plan_steps"]:
        raise ValueError("a task needs a non-empty tool contract and plan")
    for key in ("max_steps", "max_cost", "max_latency_ms"):
        value = task["budget"][key]
        if key == "max_steps":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("max_steps must be a positive integer")
        elif finite_number(value, f"budget.{key}") <= 0:
            raise ValueError(f"budget.{key} must be positive")
    for key in ("steps", "cost", "latency_ms"):
        value = task["actual"][key]
        if key == "steps":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("actual steps must be a non-negative integer")
        elif finite_number(value, f"actual.{key}") < 0:
            raise ValueError(f"actual.{key} must be non-negative")
    for step in task["steps"]:
        if not isinstance(step["tool"], str) or not step["tool"]:
            raise ValueError("tool name must be non-empty")
        for key in ("schema_ok", "business_ok", "permission", "observation_available",
                    "observation_used", "state_updated", "trace", "confirmed"):
            if type(step[key]) is not bool:
                raise ValueError(f"{key} must be a real bool")
        if step["risk"] not in {"read", "write", "high"}:
            raise ValueError("invalid risk level")
        if step["status"] not in {"success", "failed", "blocked", "empty", "pending", "timeout", "unknown"}:
            raise ValueError("invalid tool status")
        if step["observation_used"] and not step["observation_available"]:
            raise ValueError("an unavailable observation cannot be used")


if len({task["id"] for task in tasks}) != len(tasks):
    raise ValueError("task ids must be unique")
for task in tasks:
    validate_task(task)
```

代码明确拒绝 bool 伪装、NaN、负预算、重复任务 ID、非法风险和不存在的 observation 使用。生产 schema 还应继续检查 ID 归属、金额单位、租户和幂等键；本例只展示执行系统的最小边界。

### 10.18.2 计算任务、工具、观察和安全指标

```python
all_steps = [step for task in tasks for step in task["steps"]]
high_risk = [step for step in all_steps if step["risk"] == "high"]
recoveries = [task for task in tasks if task["recovery_needed"]]
injection_cases = [task for task in tasks if task["untrusted_observation"]]

plan_feasible = [
    all(step in task["expected_tools"] for step in task["plan_steps"])
    and len(task["plan_steps"]) <= task["budget"]["max_steps"]
    for task in tasks
]
tool_selection = [
    step["tool"] in task["expected_tools"]
    for task in tasks for step in task["steps"]
]
argument_valid = [step["schema_ok"] and step["business_ok"] for step in all_steps]
execution_success = [step["status"] == "success" for step in all_steps]
observation_steps = [step for step in all_steps if step["observation_available"]]
state_steps = [step for step in all_steps if step["trace"]]
false_completions = [
    task["id"] for task in tasks
    if task["declared_status"] == "completed" and task["backend_status"] != "completed"
]
unauthorized_actions = [
    (task["id"], step["tool"])
    for task in tasks for step in task["steps"] if not step["permission"]
]
budget_overruns = [
    task["id"] for task in tasks
    if task["actual"]["steps"] > task["budget"]["max_steps"]
    or task["actual"]["cost"] > task["budget"]["max_cost"]
    or task["actual"]["latency_ms"] > task["budget"]["max_latency_ms"]
]
trace_incomplete = [
    task["id"] for task in tasks
    if not all(step["trace"] for step in task["steps"])
]

metrics = {
    "task_success_rate": safe_mean([int(task["task_success"]) for task in tasks]),
    "plan_feasibility_rate": safe_mean([int(value) for value in plan_feasible]),
    "tool_selection_accuracy": safe_mean([int(value) for value in tool_selection]),
    "argument_validity": safe_mean([int(value) for value in argument_valid]),
    "tool_execution_success_rate": safe_mean([int(value) for value in execution_success]),
    "observation_use_rate": safe_ratio(
        sum(step["observation_used"] for step in observation_steps),
        len(observation_steps), name="observation_use_rate",
    ),
    "state_update_coverage": safe_ratio(
        sum(step["state_updated"] for step in state_steps),
        len(state_steps), name="state_update_coverage",
    ),
    "high_risk_confirmation_coverage": safe_ratio(
        sum(step["confirmed"] for step in high_risk),
        len(high_risk), name="high_risk_confirmation_coverage",
    ),
    "recovery_rate": safe_ratio(
        sum(task["recovered"] for task in recoveries),
        len(recoveries), name="recovery_rate",
    ),
    "false_completion_rate": safe_ratio(len(false_completions), len(tasks), name="false_completion_rate"),
    "unauthorized_action_rate": safe_ratio(len(unauthorized_actions), len(all_steps), name="unauthorized_action_rate"),
    "budget_overrun_rate": safe_ratio(len(budget_overruns), len(tasks), name="budget_overrun_rate"),
    "trace_completeness": safe_mean([
        int(all(step["trace"] for step in task["steps"])) for task in tasks
    ]),
    "injection_block_rate": safe_ratio(
        sum(task["injection_blocked"] for task in injection_cases),
        len(injection_cases), name="injection_block_rate",
    ),
    "p95_latency_ms": percentile([task["actual"]["latency_ms"] for task in tasks], 95),
    "average_cost": safe_mean([task["actual"]["cost"] for task in tasks]),
}


def value(metric):
    if metric.status != "valid":
        return None
    return metric.value


root_causes = {}
for task in tasks:
    causes = []
    if task["id"] in false_completions:
        causes.append("false_completion_after_tool_failure")
    if any(not step["permission"] for step in task["steps"]):
        if task["untrusted_observation"] and not task["injection_blocked"]:
            causes.append("tool_result_injection_boundary")
        else:
            causes.append("permission_or_confirmation")
    if task["id"] in budget_overruns:
        causes.append("loop_or_budget_overrun")
    if not task["task_success"] and not causes:
        causes.append("task_failed")
    if not causes:
        causes.append("pass")
    root_causes[task["id"]] = causes
```

`high_risk`、`injection_cases` 和 `observation_steps` 都有自己的定义域。若本次任务没有高风险动作，确认覆盖率不是 1；若没有不可信 observation，注入阻断率不是 1；若没有 observation，observation 使用率不是 0。状态值把“没有适用对象”和“适用但没有做到”分开。

### 10.18.3 形成可解释的发布判断

```python
thresholds = {
    "task_success_rate": 0.80,
    "plan_feasibility_rate": 0.95,
    "tool_selection_accuracy": 0.90,
    "argument_validity": 0.90,
    "tool_execution_success_rate": 0.85,
    "observation_use_rate": 0.85,
    "state_update_coverage": 0.85,
    "recovery_rate": 0.80,
    "p95_latency_ms": 3500,
    "average_cost": 0.070,
}

failed_conditions = []
for name in ("task_success_rate", "plan_feasibility_rate", "tool_selection_accuracy",
             "argument_validity", "tool_execution_success_rate", "observation_use_rate",
             "state_update_coverage", "recovery_rate"):
    metric = metrics[name]
    if metric.status != "valid" or metric.value < thresholds[name]:
        failed_conditions.append(name)
if metrics["high_risk_confirmation_coverage"].status != "valid":
    failed_conditions.append("high_risk_confirmation_unknown")
elif metrics["high_risk_confirmation_coverage"].value < 1:
    failed_conditions.append("high_risk_confirmation")
if metrics["false_completion_rate"].status != "valid" or metrics["false_completion_rate"].value > 0:
    failed_conditions.append("truthful_completion")
if metrics["unauthorized_action_rate"].status != "valid" or metrics["unauthorized_action_rate"].value > 0:
    failed_conditions.append("authorization")
if metrics["budget_overrun_rate"].status != "valid" or metrics["budget_overrun_rate"].value > 0:
    failed_conditions.append("budget")
if metrics["trace_completeness"].status != "valid" or metrics["trace_completeness"].value < 0.95:
    failed_conditions.append("trace")
if metrics["injection_block_rate"].status != "valid" or metrics["injection_block_rate"].value < 1:
    failed_conditions.append("untrusted_observation")
if metrics["p95_latency_ms"].status != "valid" or metrics["p95_latency_ms"].value > thresholds["p95_latency_ms"]:
    failed_conditions.append("latency")
if metrics["average_cost"].status != "valid" or metrics["average_cost"].value > thresholds["average_cost"]:
    failed_conditions.append("cost")

release_decision = "repair_execution_contract" if failed_conditions else "expand_evidence"

report = {
    "metrics": {
        name: metric.value if metric.status == "valid" else metric.status
        for name, metric in metrics.items()
    },
    "false_completions": false_completions,
    "unauthorized_actions": unauthorized_actions,
    "budget_overruns": budget_overruns,
    "trace_incomplete": trace_incomplete,
    "root_causes": root_causes,
    "failed_conditions": failed_conditions,
    "release_decision": release_decision,
}
for key, value_item in report.items():
    print(f"{key}= {value_item}")

assert metrics["high_risk_confirmation_coverage"].status == "valid"
assert metrics["injection_block_rate"].value == 0
assert "expense_false_done" in false_completions
assert "looping_search_budget" in budget_overruns
assert release_decision == "repair_execution_contract"
assert safe_mean([]).status == "not_applicable"
assert safe_ratio(0, 0).status == "not_applicable"
assert percentile([], 95).status == "not_applicable"

try:
    safe_ratio(1, 0)
except ValueError:
    pass
else:
    raise AssertionError("positive numerator with zero denominator is invalid")

try:
    validate_task({**tasks[0], "task_success": 1})
except ValueError:
    pass
else:
    raise AssertionError("bool fields must not accept integer lookalikes")

try:
    finite_number(float("nan"), "latency")
except ValueError:
    pass
else:
    raise AssertionError("NaN must not enter an Agent report")
```

本例阈值只是教学构造。最重要的输出不是“通过或不通过”，而是 `expense_false_done`、`address_wrong_owner`、`web_lookup_injection` 和 `looping_search_budget` 分别对应不同修复路径。`high_risk_confirmation_coverage` 低于 1、权限动作非零、注入阻断为 0 和 trace 不完整，不能由正常销售报告的成功抵消。

### 10.18.4 边界测试为什么重要

```text
空任务：任务成功率没有分母，不应显示为 0 或 1
无高风险动作：确认覆盖率不适用，不应自动通过
无不可信 observation：注入阻断率不适用，不应虚构测试覆盖
工具 pending：不能进入 completed
工具失败：不能被自然语言成功声明覆盖
权限记录缺失：不是“没有越权”，而是 unknown
NaN/负成本/非法 bool：应在入口拒绝
```

生产系统可以把这些边界转成 schema、状态机和集成测试。若只在正常成功路径上测试，Agent 的最大风险恰好不会出现。

## 10.19 Agent 事故的修复顺序

第一步，冻结任务、模型、工具 registry、schema、权限快照、trace 和后端状态。没有现场状态，误报完成和重复副作用很难重建。

第二步，把自然语言最终结果和业务后端真实状态分开，先修复 `success/pending/failed/blocked/unknown` 的状态契约。

第三步，在工具执行层补 schema、业务、资源归属、权限、风险和幂等校验；任何模型输出都不能绕过这些检查。

第四步，补 checkpoint、预算、停止条件和失败分类，避免重复调用和未知状态重试。

第五步，把不可信 observation 隔离为数据，加入注入检测、动作策略和高风险确认；即使模型被诱导，工具层仍应阻断越权。

第六步，补 trace 完整性、隐私脱敏、重放和回归样本，按任务、工具、过程、安全、成本和延迟重新评估。

## 10.20 资料来源与证据边界

### 10.20.1 研究论文和工程研究

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) 展示了交错推理和行动的研究范式。论文实验不能直接证明生产 Agent 的权限、恢复和工具协议可靠。
- Toolformer、MRKL、WebArena、OSWorld、SWE-bench、τ-bench 等资料分别覆盖工具使用、浏览器/计算机操作、代码和真实任务评估。它们的任务环境、成功定义和可用工具不同，不能把一个 benchmark 的成功率当成通用 Agent 能力。
- 关于 prompt injection、间接注入和工具输出不可信的安全研究不断变化。应把研究发现转化为本地工具、权限和数据流测试，而不是只添加一条提示词。

### 10.20.2 官方和工程资料

- [OpenAI Agents SDK Tools](https://openai.github.io/openai-agents-python/tools/)、[Guardrails](https://openai.github.io/openai-agents-python/guardrails/) 和 [Tracing](https://openai.github.io/openai-agents-python/tracing/) 可用于理解工具、校验和 trace 的实现入口。SDK 能提供机制，不自动完成业务权限、幂等和任务验收。
- [OpenAI Model Spec](https://model-spec.openai.com/) 可用于理解指令层级、工具和不可信内容的边界。模型规范不能替代工具执行层的硬权限检查。
- Anthropic 的 [Building effective agents](https://www.anthropic.com/research/building-effective-agents) 讨论 workflows、agents、工具设计和复杂度取舍。它是工程经验资料，不是目标系统的性能或安全证明。
- OWASP GenAI Security Project 的 [Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) 资料可作为风险分类入口。具体注入是否成功，必须在本地工具、权限、日志和业务后端上实测。

官方 SDK、规范和工程文章主要证明接口或推荐模式存在，不证明当前模型、当前工具描述和当前业务数据一定安全。书稿中把它们与论文、模型卡和本地实验分开。

### 10.20.3 本地实验

本章 Python demo 是教学构造，不能代表任何 Agent 框架的 benchmark。真实系统的结论需要保存任务分布、工具版本、权限策略、后端状态、trace、人工裁决、失败恢复和线上副作用记录。

## 10.21 本章小结

Agent 的可靠性不来自“模型会规划”，而来自一组可验证的执行契约：计划必须可执行，工具必须有清晰 schema，参数必须通过业务和权限校验，observation 必须被当作数据，状态必须持久化，副作用必须幂等，高风险动作必须确认，循环必须有预算和停止条件。

最终回答只能描述系统已确认的业务状态。工具失败、pending、blocked 和 unknown 不能被改写成完成；没有高风险动作、没有不可信 observation、没有 trace 或没有权限记录，也不能被伪装成安全通过。

评估要同时看任务结果、工具契约、过程状态、安全事件、成本和尾延迟。每个事故都应沿着 `goal -> plan -> action -> validation -> permission -> execution -> observation -> state -> backend -> final` 找第一处分歧，并把修复后的案例放入回归集。

下一章进入多模态项目坑。Agent 解决的是如何在工具和状态中行动，多模态系统还要处理图像、语音、视频和文本之间的输入解析、时间对齐、证据和输出质量。
