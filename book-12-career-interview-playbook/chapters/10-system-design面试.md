# 第十章：大模型系统设计：从服务契约到可恢复系统

大模型系统设计的对象不是一张漂亮的架构图，而是一组在真实约束下能够持续交付结果的服务。模型只是其中一个概率组件：它会产生不确定的文本、工具意图或结构化参数，系统还必须负责身份、权限、上下文、检索、执行、计费、审计、监控、回滚和数据删除。

因此，一个系统设计问题真正要问的是：

1. 用户请求从哪里进入，经过哪些状态，最后以什么证据结束？
2. 哪些工作在线完成，哪些工作应该离线预计算？
3. 输入 token、输出 token、工具调用和视觉内容如何转化为容量？
4. 模型判断错误时，哪一层阻止影响扩大？
5. 质量、延迟、成本、安全和可维护性发生冲突时，谁负责作出取舍？
6. 版本、权限和数据删除发生变化时，旧状态如何失效？
7. 系统变更后，团队凭什么知道它变好了，而不是只在一个 benchmark 上变好？

本章贯穿一个教学系统“衡川企业助手”。它接收企业合同、制度和票据，支持文本问答、文档检索、图像字段抽取和只读工具查询；它可以生成待审核摘要，但不能自行付款、发邮件、修改合同或授予权限。这个边界足够包含 Chat、RAG、多模态、Agent、推理和安全问题，也足够让我们看见各模块之间的责任分工。

文中的容量、延迟和成本数字是教学估算，案例中的请求和事故是合成场景。论文、协议、官方文档、benchmark 和项目实测分别承担不同证据责任，不能互相替代。

## 10.1 系统设计的对象：从模型调用到交付闭环

### 10.1.1 模型输出不是业务结果

一次模型调用通常返回：

- 文本 token；
- 结构化 JSON；
- 工具调用候选；
- 置信或不确定性信号；
- 停止原因和用量信息。

业务系统需要的可能是：

- 一段带来源的合同摘要；
- 一个经过权限过滤的字段；
- 一个已确认成功的只读查询；
- 一个进入人工审核的任务；
- 一个明确标记为未知的执行状态。

二者之间隔着解析、验证、证据绑定、权限检查和业务状态确认。模型说“合同已更新”不代表数据库发生了更新；模型输出了 search_contract 不代表它有权读取那个合同；模型引用了一个 URL 也不代表 URL 中的内容支持它的结论。

### 10.1.2 四个边界

可以先把系统拆成四个边界：

| 边界 | 主要对象 | 典型责任 |
| --- | --- | --- |
| 控制平面 | 模型目录、策略、配置、版本、发布 | 变更、灰度、回滚和权限策略 |
| 数据平面 | 在线请求、检索、推理、工具和响应 | 处理一次具体任务 |
| 证据平面 | 文档、区域、引用、评估样本和 trace | 证明输出依据和复盘事实 |
| 治理平面 | 身份、租户、审计、隐私、事故 | 约束影响、保存责任和删除范围 |

控制平面的配置不应被普通用户请求直接修改；数据平面的模型输出不应绕过治理平面的授权；证据平面中的文档不应因为被检索进上下文就自动获得指令权限。

### 10.1.3 在线链路和离线链路

衡川企业助手至少有两条链路：

~~~text
离线控制链
文档上传 -> 解析/OCR -> 清洗 -> 分块 -> 权限标注
    -> embedding/倒排索引 -> 版本发布 -> 离线评估

在线任务链
请求 -> 身份/租户 -> 任务分类 -> 权限检索
    -> 上下文构建 -> 模型推理 -> 证据绑定
    -> 工具/人工确认 -> 响应 -> trace/反馈
~~~

离线链路适合处理耗时的解析、embedding、去重和评估；在线链路适合处理用户当前意图、权限、新鲜度和需要立即确认的动作。把所有工作都塞进在线链路会增加延迟，把所有工作都提前做完又可能使用过时权限或过时文档。

### 10.1.4 先定义不可接受的失败

质量目标不能只写“回答准确”。系统需要写出不可接受的失败，例如：

- 把一个租户的合同内容返回给另一个租户；
- 在没有证据时给出确定的付款金额；
- 工具超时后宣称动作已经完成；
- 取消请求后仍继续产生外部副作用；
- 文档中的不可信指令改变了系统策略；
- 输出审核为了降低风险误伤所有正常安全咨询；
- 版本回滚后继续复用新版本的缓存和索引。

这些失败的严重度不同，修复层级也不同。把它们都归为“模型回答错误”，会错过真正的系统根因。

### 10.1.5 面试时按一条可复核的链展开

面试官给出宽泛题目时，先用一句话确认业务结果和权限边界，再询问峰值请求、输入/输出长度、响应时限、数据隔离、不可接受失败和成本约束。没有条件时，明确说出暂用的假设，不把它们说成题目事实。随后按“工作量与服务目标 → 一次请求的状态和数据流 → 关键资源与故障路径 → 评估和发布”逐步展开。

例如，“设计合同助手”不能一上来就列向量库、模型和缓存。更清楚的开场是：“我先按只读合同问答设计，回答必须带有用户可访问的文档版本和页码；我会先确认峰值与延迟目标，再走一条请求、检查权限撤销和工具超时，最后估算容量并说明灰度条件。”这段话先固定结果和边界，后续每个组件才有可以检验的理由。若面试官修正并发量或允许的动作，就更新假设和估算，而不是继续沿用旧架构。

## 10.2 服务契约：先把开放问题变成可计算约束

### 10.2.1 功能需求和非功能需求

功能需求描述系统做什么，例如：

1. 用户上传合同和票据；
2. 系统按租户权限检索文档；
3. 用户询问条款、金额和版本差异；
4. 系统返回回答、引用和不确定性；
5. 需要时进入人工复核；
6. 记录可删除的审计信息。

非功能需求描述系统做到什么程度，例如：

- P95 首 token 延迟不超过某个目标；
- 高峰期间关键租户仍有最低吞吐；
- 文档权限变更后旧结果在规定时间内失效；
- 高风险字段必须有可追溯证据；
- 服务出现模型故障时可以降级到检索或人工；
- 每次响应都能关联模型、模板、索引和策略版本。

不要把所有约束写成一个总分。一个系统可以平均延迟很好，却在大文档和高峰租户上超时；也可以平均质量很高，却有极少数严重越权。

把服务等级指标（SLI）定义成可测量的事件和分母，再把服务等级目标（SLO）写成目标值、测量窗口和适用请求集合。Google SRE 对 SLI、SLO 和测量范围作了区分；这是一种工程定义参考，不替代组织自己的服务合同。举例：只统计身份有效、进入交互式问答路径的 1,000 个请求，其中 970 个在 1.5 秒内返回首个有效内容，则该集合的 TTFT-SLI 为 `970 / 1,000 = 97%`；若目标是 95%，本窗口达标。鉴权拒绝请求应另报拒绝率，不能因为它们返回得快就混入问答延迟分母。[Google SRE：Service Level Objectives](https://sre.google/sre-book/service-level-objectives/)

### 10.2.2 用工作量向量描述请求

请求数量不足以描述大模型负载。可以把每个请求表示为一个工作量向量：

~~~math
w
=
(
T_{\mathrm{in}},
T_{\mathrm{out}},
N_{\mathrm{retrieval}},
N_{\mathrm{tool}},
N_{\mathrm{image}},
N_{\mathrm{audio}},
R_{\mathrm{risk}}
)
~~~

其中，T_in 和 T_out 是非负整数形式的输入、输出 token 数，N_retrieval、N_tool、N_image 和 N_audio 是非负整数形式的调用或媒体对象数量，R_risk 表示审核、人工复核或隔离带来的额外工作量。不同分量的单位不一定相同，因此这个向量适合描述负载，不适合直接做加法或比较大小；若要计算容量，还要先给每一维指定资源映射。

两个请求都算一个 request，但一个只生成 30 个 token，另一个包含 100,000 个输入 token、四次检索和三次工具调用，它们对 GPU、数据库、网络和人工队列的影响完全不同。

### 10.2.3 token 需求和内部调用放大

设每秒外部请求数为非负的 `lambda_req`，第 i 类请求占比为非负的 `p_i`，且所有类别占比之和为 1；平均输入和输出 token `T_in,i`、`T_out,i` 也为非负数，则模型 token 速率的教学估算为：

此处每类请求的 `T_in,i` 和 `T_out,i` 按一次外部请求触发的全部模型调用合计；如果手上的数据是单次模型调用的均值，还要乘上该类请求平均触发的模型调用数，不能把 Agent、judge、摘要和重试的 token 漏掉。

~~~math
D_{\mathrm{token}}
=
\lambda_{\mathrm{req}}
\sum_i
p_i
\left(
T_{\mathrm{in},i}
+
T_{\mathrm{out},i}
\right)
~~~

如果一次外部请求平均触发非负的 `a` 次模型调用，工具和重试又带来非负的 `b` 倍内部放大，则内部调用率近似为：

~~~math
\lambda_{\mathrm{internal}}
=
\lambda_{\mathrm{req}}
\times
a
\times
b
~~~

这里的 a 和 b 不是固定常数。Agent 循环、judge、重试、摘要和路由器都会改变它们。只按外部 QPS 规划容量，往往会低估真实负载。

### 10.2.4 延迟预算要拆开

一次在线响应的端到端延迟可以拆成：

~~~math
L_{\mathrm{e2e}}
=
L_{\mathrm{gateway}}
+
L_{\mathrm{queue}}
+
L_{\mathrm{preprocess}}
+
L_{\mathrm{retrieve}}
+
L_{\mathrm{prefill}}
+
L_{\mathrm{decode}}
+
L_{\mathrm{guard}}
+
L_{\mathrm{network}}
~~~

L_gateway 是接入和鉴权，L_queue 是排队，L_preprocess 是 tokenizer、OCR 或模板处理，L_retrieve 是检索和重排，L_prefill 是建立上下文，L_decode 是逐 token 生成，L_guard 是输入/输出审核，L_network 是流式传输和客户端可见的网络时间。将这些项相加的前提是它们使用同一时间单位并且都为非负时长；如果某些阶段并行执行，就不能把它们机械地全部相加，应按关键路径测量。

TTFT 通常关注首个有效输出事件前的时间，TPOT 关注首 token 后的平均 token 间隔。它们不能替代完整 E2E。一个服务 TTFT 很低，但在输出审核或工具确认阶段停顿很久，用户仍然会认为响应慢。

### 10.2.5 Little 定律和排队直觉

在稳定系统中，平均在途请求数 L、非负到达率 lambda 和平均在系统时间 W 可以用 Little 定律描述：

~~~math
L
=
\lambda W
~~~

如果系统每秒处理 20 个请求，平均一个请求在系统中停留 2 秒，则平均在途请求约为 40。这个公式不能代替复杂排队模型，但可以帮助检查容量估算是否数量级合理。

当到达率接近服务能力时，排队时间会比计算时间更敏感。模型优化把 decode 提速 10%，如果请求已经大部分时间在等待 GPU 或等待人工复核，用户未必感受到对应收益。

## 10.3 高层架构：控制平面、数据平面和证据平面

### 10.3.1 控制平面

衡川的控制平面包含：

~~~text
Model Registry
    -> model revision, tokenizer, template, capabilities
Policy Registry
    -> tenant policy, tool policy, risk rules
Data Registry
    -> document/index version, schema, retention
Eval Registry
    -> dataset, judge, thresholds, regression history
Release Controller
    -> canary, rollback, configuration propagation
~~~

每项配置都需要版本、创建者、变更原因和生效时间。线上请求应记录它实际读取到的版本，而不是只记录当前最新版本。否则一次事故发生后，团队无法确定请求当时使用了哪条策略。

### 10.3.2 数据平面

数据平面处理一次任务：

~~~text
Gateway
    -> Auth and Tenant Context
    -> Request Normalizer
    -> Orchestrator
        -> Retriever
        -> Model Router
        -> Model Serving
        -> Tool Executor
    -> Evidence Binder
    -> Output Policy
    -> Stream/Response
~~~

编排器不是“把所有东西都调用一遍”。它应根据任务类型选择路径：简单制度查询可能只需要检索和小模型；扫描票据需要 OCR 和视觉模型；高风险工具候选需要更严格的参数确认；证据不足时应进入澄清或人工状态。

### 10.3.3 证据平面

证据对象不应只是一段拼进 prompt 的字符串。一个文档证据对象可以包含：

~~~text
evidence_id
    -> tenant_id
    -> document_id and document_revision
    -> page/region or text span
    -> parser/OCR revision
    -> permission snapshot
    -> retrieval score and rank
    -> content hash
    -> trust level
~~~

模型输出中的 claim 再关联到一个或多个 evidence_id。这样，回答“合同第二季度付款比例是多少”时，系统不仅保存文本，还能定位到文档版本、页码、区域和权限快照。

### 10.3.4 信任边界

外部文档、用户输入、工具返回和模型输出都可能包含不可信内容。系统应在边界上明确：

~~~text
user request -> defines the requested task
external document -> candidate evidence
tool result -> data plus execution status
model output -> candidate answer or action
server policy -> actual permission and execution authority

external document -/-> change server policy
tool result -/-> grant a new permission
model output -/-> bypass authorization
cached response -/-> bypass current tenant check
~~~

模型可以提出动作，不能授予自己权限；文档可以提供事实，不能修改系统规则；缓存可以减少重复计算，不能跳过当前权限校验。

下图把离线索引发布、在线授权检索、证据绑定和审计放在同一条链上。读图时重点检查三条关系：索引由版本发布控制器激活；在线检索先带身份与租户策略；模型输出须经过证据绑定和输出检查才返回。

~~~mermaid
flowchart LR
  subgraph offline["离线数据准备"]
    source["文档原件"] --> parse["解析与 OCR"]
    parse --> index["带版本的索引"]
  end
  subgraph control["控制与治理平面"]
    release["发布控制器"]
    policy["租户策略与权限版本"]
    registry["模型、模板与工具版本"]
  end
  subgraph online["在线请求链"]
    request["请求"] --> identity["身份与租户"]
    identity --> orchestrator["任务编排"]
    orchestrator --> retrieval["权限过滤检索"]
    retrieval --> model["模型推理与只读工具"]
    model --> binder["证据绑定与输出检查"]
    binder --> response["响应或人工复核"]
  end
  subgraph evidence["证据与审计"]
    store["文档、页码与 revision"]
    trace["trace、审计与删除记录"]
  end
  index --> store
  store --> retrieval
  retrieval --> binder
  release -.->|激活索引版本| retrieval
  policy -.->|当前授权| identity
  registry -.->|固定组件版本| orchestrator
  identity --> trace
  retrieval --> trace
  binder --> trace
  response --> trace
~~~

## 10.4 请求生命周期：状态机比“调用模型”更重要

### 10.4.1 一次任务的状态

衡川的只读问答可以经过：

~~~text
RECEIVED
    -> AUTHORIZED
    -> NORMALIZED
    -> RETRIEVING
    -> CONTEXT_READY
    -> GENERATING
    -> VERIFYING
    -> COMPLETED
~~~

异常路径包括：

~~~text
RECEIVED -> REJECTED
处理中 -> CANCELLED
RETRIEVING -> RETRYABLE_FAILURE
GENERATING -> TIMEOUT
VERIFYING -> NEEDS_REVIEW
写操作已发出 -> UNKNOWN_EXTERNAL_EFFECT
~~~

状态必须有明确含义。TIMEOUT 说明本次请求没有在预算内结束，不等于已经发出的工具动作未生效；但只有已发出且可能有副作用的写操作，才进入 UNKNOWN_EXTERNAL_EFFECT。普通模型超时、检索失败或尚未发送的工具候选不能被记成“外部效果未知”。

下面的状态图把普通问答、只读查询和写操作确认路径分开。UNKNOWN_RESULT 可以在预算内按原 request_id 重试；UNKNOWN_EXTERNAL_EFFECT 之后先查询业务凭证或交人工核对，再进入结果验证。图中的写操作只有经过人工确认和服务端再次授权后才会发送；衡川的默认契约仍是只读查询和人工复核。

~~~mermaid
flowchart TD
  received["RECEIVED"] -->|身份有效| authorized["AUTHORIZED"]
  received -->|拒绝| rejected["REJECTED"]
  authorized --> normalized["NORMALIZED"] --> retrieving["RETRIEVING"]
  retrieving --> context["CONTEXT_READY"] --> generating["GENERATING"]
  generating -->|普通回答| verifying["VERIFYING"]
  generating -->|提出工具候选| proposed["ACTION_PROPOSED"]
  proposed --> policy["服务端授权检查"]
  policy -->|只读工具授权| read["READ_TOOL_DISPATCHED"]
  policy -->|写操作需人工确认| approval["WAITING_USER_APPROVAL"]
  approval -->|确认后重新授权| write["WRITE_DISPATCHED"]
  read -->|有结果| verifying
  read -->|超时无回包| unknown_result["UNKNOWN_RESULT"]
  unknown_result -->|预算内同 request_id 重试| read
  unknown_result -->|预算耗尽| review["NEEDS_REVIEW"]
  write -->|下游凭证| verifying
  write -->|超时且是否生效不明| unknown_effect["UNKNOWN_EXTERNAL_EFFECT"]
  unknown_effect -->|查询状态或人工核对| verifying
  verifying -->|证据充分| completed["COMPLETED"]
  verifying -->|证据不足| review
  retrieving -->|有界临时错误| retry["RETRYABLE_FAILURE"]
  retry -->|预算内且可重试| retrieving
  generating -->|取消并确认停止| cancelled["CANCELLED"]
~~~

### 10.4.2 不变量

状态机应该维护不变量：

1. 未授权请求不能进入检索和工具执行；
2. 一个 request_id 只能有一个终态；
3. 已取消的生成不能继续向客户端发送有效输出；
4. 高风险动作没有确认凭证不能执行；
5. 完成状态必须关联响应版本和证据状态；
6. UNKNOWN_EXTERNAL_EFFECT 状态不能自动重发不可幂等写操作；
7. 删除或撤销权限后，旧缓存不能继续产生可见结果。

不变量是系统设计中比组件名称更重要的内容。它们可以转化为测试、监控和事故检查。

### 10.4.3 幂等和重复提交

客户端重试、网关超时和 worker 重启都可能让同一请求被提交多次。对只读请求，重复通常只是浪费；对写操作，重复可能造成重复付款、重复发送或重复修改。

可以使用：

- 客户端生成的 idempotency_key；
- 服务端 operation_id；
- 下游数据库唯一约束；
- 工具动作的幂等语义；
- 完成结果的可查询凭证。

幂等键应绑定租户、业务动作和请求语义，不能只按自然语言 prompt 哈希。用户在同一合同上提交两个不同版本的更新，不应因为文字相似而错误复用结果。

### 10.4.4 取消和未知

取消请求需要区分：

1. 还在排队，可以直接移除；
2. 正在 prefill，可以尝试终止计算；
3. 正在 decode，可以停止生成并释放 KV；
4. 工具已经发出，不能只取消模型线程；
5. 外部系统状态未知，需要查询或人工核对。

“停止生成”只解决模型计算，不自动撤销已经发出的网络请求。系统应把取消语义传递到每一层，并记录哪一层真正停止。

## 10.5 Chat 服务：会话、上下文与流式协议

### 10.5.1 会话数据模型

一个可靠的会话服务至少要区分：

| 对象 | 作用 | 关键字段 |
| --- | --- | --- |
| Conversation | 会话容器 | tenant、owner、policy、status |
| Message | 用户或助手消息 | role、content、revision、visibility |
| Response | 一次生成尝试 | model、template、usage、finish_reason |
| Branch | 分支重试或编辑 | parent_response、selected |
| Summary | 历史压缩 | source_range、model、confidence |
| Feedback | 用户反馈 | label、reason、created_at |

不能把整个会话存成一个不断覆盖的字符串。编辑历史、重新生成、删除某条消息、审计某次回答和比较模型版本都需要结构化记录。

### 10.5.2 上下文预算

设模型上下文上限为 `L_max > 0`，system/developer 消息、当前用户请求、工具和检索结果、预留输出分别占用非负 token 数 `L_sys`、`L_user`、`L_tool`、`L_out`，历史上下文预算为 `L_hist`，则：

~~~math
L_{\mathrm{hist}}
\le
L_{\mathrm{max}}
-
L_{\mathrm{sys}}
-
L_{\mathrm{user}}
-
L_{\mathrm{tool}}
-
L_{\mathrm{out}}
~~~

当历史超出预算时，系统可以按任务相关性、用户 pin、最近消息、未完成约束和证据重要性选择保留内容。简单保留最后 N 条会丢失早期的权限、格式和任务约束；简单摘要又可能把否定条件、数字和来源压缩错。

### 10.5.3 摘要不是事实数据库

摘要模型可能把“不能发送邮件”压成“可以发送邮件”，也可能丢失合同版本和时间条件。因此摘要应带：

- 来源消息范围；
- 生成模型和版本；
- 时间戳；
- 关键约束；
- 可验证的原始消息引用；
- 是否允许被后续任务直接当作事实。

高风险场景不应只依赖摘要。需要时重新检索原始消息或要求用户确认。

### 10.5.4 SSE 和 WebSocket 的边界

SSE 适合服务端向客户端推送文本事件，浏览器实现简单；WebSocket 适合双向低延迟交互，例如语音帧、用户打断和复杂会话控制。无论使用哪种协议，都要定义：

- event_id；
- sequence；
- response_id；
- token 或 delta；
- heartbeat；
- error；
- done；
- reconnect behavior；
- cancellation。

客户端不能把网络连接断开当作服务端已取消。服务端也不能把已经发送的 partial token 当成最终结果。流式协议需要明确最终状态和重连后的重复去重规则。

### 10.5.5 体感延迟和后台任务

对短回答，首 token 和 token 间隔影响体验；对长文档、视频和多步骤 Agent，用户更需要任务进度和可恢复后台任务。系统可以把请求分为：

- 交互式流式任务；
- 可等待的异步任务；
- 长周期任务；
- 需要人工复核的任务。

所有任务都应有可查询的 task_id、进度、最近事件、取消语义和过期时间。把长任务伪装成一直保持的 HTTP 请求，会让重连、扩缩容和故障恢复变得困难。

## 10.6 推理平台：把 token 预算变成调度和显存

### 10.6.1 Prefill 和 decode 是不同负载

Prefill 读取完整输入并建立 KV Cache，通常更适合批量矩阵计算；decode 每次生成一个或少数 token，需要读取历史 KV 并串行推进。输入长的请求主要压 prefill，输出长的请求主要压 decode，二者混在同一个队列中会产生阶段干扰。

推理服务的演进可以从“同一批 GPU 同时处理 prefill 和 decode”讲起：连续 batching 让请求在生成过程中进入或离开批次，分页式 KV 管理则针对缓存随序列增长、预留空间易产生碎片的问题。PagedAttention 原论文展示了按固定 token 块管理 KV 并在请求间共享块的做法；它的吞吐结果只属于论文中的 vLLM、模型、硬件和负载设置。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180v1)

之后的 prefill/decode 解耦把两个阶段放到不同资源池，以便分别调度；但阶段分离需要传送 KV，也会引入网络、排队和资源配置成本。DistServe 按 TTFT/TPOT 目标配置两阶段资源，Mooncake 把 KVCache 和分层存储纳入调度；TaiChi 的比较指出聚合与分离在不同 TTFT/TPOT 目标下各有适用区间，并提出混合架构。这些工作支持“按负载和 SLO 比较方案”，不支持“分离必然更快”或把单篇论文倍率当成容量系数。[DistServe](https://arxiv.org/abs/2401.09670v3)、[Mooncake](https://arxiv.org/abs/2407.00079v5)、[TaiChi](https://arxiv.org/abs/2508.01989v1)

调度器至少要知道：

- 预计输入 token；
- 预计最大输出 token；
- 当前阶段；
- KV Cache 需求；
- 优先级和截止时间；
- 是否允许抢占；
- 是否有共享前缀。

### 10.6.2 KV Cache 账本

设 transformer 层数为 `L > 0`，并发序列数为 `B >= 0`，每个序列平均缓存长度为 `T >= 0`，KV 头数为 `H_kv > 0`，每个 head 维度为 `d > 0`，每个元素占 `b > 0` 字节，则理想化 KV Cache 大小为：

~~~math
M_{\mathrm{KV}}
\approx
2LBT H_{\mathrm{kv}}d b
~~~

前面的 2 是 K 和 V。这个公式不包含页表、对齐、临时 buffer、共享前缀和碎片。它说明：

- 并发和上下文长度线性增加容量；
- GQA/MQA 减少 H_kv；
- FP8 或其他低精度缓存减少 b，但需要质量回归；
- 长输出会继续增长 T；
- 释放和跨租户隔离是生命周期问题。

做一组数量级估算。假设 `L=32` 层、`B=8` 条并发序列、每条序列 `T=2,048` 个缓存位置、`H_kv=8` 个 KV 头、每头 `d=128` 维、每个元素 `b=2` 字节，则每 token、每条序列约占 `2×32×8×128×2 = 131,072` 字节，也就是 128 KiB。每条 2,048 token 的序列约占 256 MiB，8 条合计约为 2 GiB。这里使用二进制 MiB/GiB，且只是 KV payload 的教学估算；它没有计入权重、页表、临时 buffer、碎片和共享前缀，因此不是显卡采购结论。

### 10.6.3 Paged KV 和连续 batching

如果每条请求按最大上下文连续预留显存，短请求会留下空洞，长请求可能因连续空间不足而无法接入。分页式 KV 管理把缓存拆成固定大小 block，由 block table 记录逻辑序列到物理页的映射。PagedAttention 原论文给出这种逻辑块到物理块的分配和读取机制；当前具体引擎的 kernel、默认 block 大小和调度行为仍须绑定对应版本核对。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180v1)

连续 batching 允许新请求加入正在 decode 的批次，减少 GPU 空闲，但需要调度器协调：

- 新请求的 prefill 是否会阻塞已有 decode；
- 每一步给各请求分配多少 token；
- 一个请求结束后如何回收页；
- 抢占时如何保存和恢复 KV；
- 高优先级租户是否会饿死低优先级请求。

优化吞吐不能只看 tokens/s。还要报告 TTFT、P95/P99、取消率、抢占次数、KV 分配失败和不同请求长度下的公平性。

### 10.6.4 Prefix Cache 和权限

共享前缀可以减少重复 prefill，例如相同的系统指令或公开文档前缀。但前缀缓存键必须包含：

- 模型和 tokenizer 版本；
- chat template 版本；
- 完整前缀内容或安全哈希；
- 租户和权限上下文；
- RAG 文档版本；
- 失效时间和撤销版本。

不能因为两个用户的 system prompt 文本相同，就共享包含私有文档的 KV。缓存命中只表示计算状态可复用，不表示当前用户有权看到缓存产生的内容。

### 10.6.5 量化和投机解码

量化可以减少权重、激活或 KV 的存储，改善成本和吞吐，但要验证：

- 输出质量；
- JSON 和工具参数格式；
- 长上下文稳定性；
- 多语言和多模态任务；
- 安全拒绝和引用；
- 特定硬件 kernel 的实际速度。

投机解码让 draft model 或其他预测器提出候选 token，再由 target model 验证。设每轮提出 k 个 token，平均接受 a 个，验证和调度有固定开销，速度提升不能只由 k/a 推导。实际结果取决于 acceptance length、batch、内存带宽、draft 成本和拒绝后的回退。

### 10.6.6 模型路由

模型路由可根据：

- 任务类型；
- 输入和输出长度；
- 是否需要视觉或工具；
- 质量目标；
- 租户和数据边界；
- 当前负载；
- 成本预算；
- 模型版本灰度比例。

规则路由易解释，学习型路由可能提高质量/成本比，但必须处理分布漂移和置信不足。路由器本身也会增加一次模型或规则调用，不能把它的成本和错误遗漏。

## 10.7 训练平台：从数据 artifact 到可发布 checkpoint

### 10.7.1 训练平台不是一个提交脚本

训练平台至少包含：

~~~text
Dataset Registry
    -> snapshot, provenance, filtering, permissions
Experiment Config
    -> model, tokenizer, optimizer, seed, parallelism
Scheduler
    -> GPU placement, priority, preemption, quota
Training Runtime
    -> data loader, forward/backward, distributed communication
Checkpoint Store
    -> weights, optimizer, scheduler, RNG, sampler
Evaluation
    -> validation, benchmark, safety, business tasks
Model Registry
    -> lineage, approval, deployment metadata
~~~

每个训练结果都要能追溯到数据版本、代码版本、模型初始化、并行配置和评估结果。一个 loss 很低但无法知道训练数据和 tokenizer 的 checkpoint，不是可交付 artifact。

### 10.7.2 数据版本和有效 token

文本样本的字符数不能直接代表训练工作量。应记录 tokenizer 版本、有效 token 数、padding、packing、过滤原因和去重状态。若 batch 中有效 token 数为 n_valid，每个 token 的 loss 为 ell_j，加权平均可以写成：

~~~math
\mathcal{L}_{\mathrm{valid}}
=
\frac{1}{n_{\mathrm{valid}}}
\sum_{j=1}^{n_{\mathrm{valid}}}
\ell_j
~~~

如果把 padding 也算进分母，短样本和长样本混合时，loss 会被无效位置改变。训练平台应保存有效 token 统计，而不是只保存 step 数。

例如样本 A 有 3 个有效 token，其 loss 为 `0.2、0.4、0.6`；样本 B 有 1 个有效 token，loss 为 `0.8`。按有效 token 平均，`(0.2+0.4+0.6+0.8)/4 = 0.5`。若先分别求每条样本的均值，再对两条样本等权平均，则 `(0.4+0.8)/2 = 0.6`，优化目标已经变成“每条样本等权”。两种 reduction 都可能合理，但训练配置、分布式归约和报告必须与所声明的目标一致。

### 10.7.3 分布式训练和故障恢复

DP、FSDP、ZeRO、tensor parallel、pipeline parallel 等方案分别改变参数、梯度、优化器状态和通信的布局。平台需要知道：

- 哪些 rank 持有哪部分状态；
- checkpoint 是否是完整状态还是分片状态；
- 恢复时 world size 是否可以变化；
- 数据 sampler 从哪个位置继续；
- optimizer 和 RNG 是否恢复；
- 节点故障会丢失多少有效 token；
- 恢复后的 loss 是否出现跳变。

“文件存在”不等于 checkpoint 可恢复。恢复测试应实际启动小规模任务，验证 step、数据位置、学习率、梯度累积和评估结果。

版本化文档也能说明为什么恢复合同要写得具体：PyTorch 2.15 的 `torchrun` 文档指出 worker 重启后 rank 可能重新分配，弹性运行时 world size 也可能变化；Distributed Checkpoint 文档描述其支持范围内的分布式保存/加载与重分片能力。它们不表示应用的 optimizer、数据游标和随机状态会自动恢复，也不保证改变拓扑后训练轨迹等价。[PyTorch 2.15 `torchrun`](https://docs.pytorch.org/docs/2.15/elastic/run.html)、[PyTorch 2.15 Distributed Checkpoint](https://docs.pytorch.org/docs/2.15/distributed.checkpoint.html)

checkpoint 周期本身也要根据“保存耗时”和“可接受的丢失工作量”取舍。假设一次 optimizer step 用 2 秒，每 600 步同步保存一次，则两次保存之间的有效训练时间是 `600×2=1,200` 秒，也就是 20 分钟；若一次保存再花 60 秒，每周期总墙钟时间为 1,260 秒，保存开销约占 `60/1,260≈4.8%`，而中途故障最多可能丢失约 20 分钟训练进度。异步保存会改变开销与一致性边界，不能直接套用这个比例；所有数字都是教学假设。

平台还要分清逻辑训练任务、一次 worker attempt 和一个 checkpoint revision。任务可以按 `QUEUED → RUNNING → CHECKPOINTING → EVALUATING → CANDIDATE → APPROVED/REJECTED` 前进；worker 崩溃后新 attempt 接着恢复同一逻辑 run，而不是把第二次启动伪装成一条新实验。评估失败、checkpoint 写入失败和优化目标本身失败也应进入不同状态，方便排查是训练、存储还是验收环节出了问题。

### 10.7.4 监控训练健康

训练监控应同时看：

- loss 和 validation loss；
- gradient norm；
- learning rate；
- 有效 token/s；
- MFU 或设备利用率；
- 通信等待；
- 数据加载等待；
- NaN/Inf；
- checkpoint 成功率；
- GPU 温度和错误；
- 不同数据桶的 loss；
- 评估任务和安全任务。

loss 下降不代表训练健康。模型可能学会了数据泄漏、某一类数据比例发生改变，或只在训练分布上变好。

### 10.7.5 发布 artifact 和回滚

可发布 checkpoint 至少要和以下对象绑定：

- tokenizer；
- chat template；
- 推理配置；
- 量化配置；
- 评估结果；
- 安全策略；
- 训练数据和代码 revision；
- 已知限制；
- 回滚目标。

发布时使用不可变 revision，路由和缓存引用 revision，而不是只使用 latest。回滚不仅回滚权重，还要考虑模板、策略、索引、工具 schema 和监控规则是否一起回滚。

本节把训练平台限定在数据 lineage、任务状态、checkpoint 恢复和发布 artifact 这几个系统接口；它不重讲优化器或并行算法的内部推导。面试官若追问训练目标、梯度或并行细节，应沿训练技术章节继续回答；系统设计层的重点是说明这些设置与哪一次 run、哪份 checkpoint 和哪组评估结果绑定。

## 10.8 RAG 系统：从文档版本到有证据的回答

### 10.8.1 离线索引链路

衡川的文档入库链路可以拆成：

~~~text
Upload
    -> malware/type/size check
    -> parse PDF/Office/web/image
    -> OCR/layout/table extraction
    -> clean/deduplicate
    -> chunk with provenance
    -> attach tenant/permission/revision
    -> embedding and inverted index
    -> offline retrieval evaluation
    -> publish immutable index revision
~~~

文档原件、解析产物、chunk、embedding 和索引都应有版本。重新解析同一个 PDF 可能改变阅读顺序和 chunk 边界，因此不能只覆盖旧向量。

### 10.8.2 Chunk 的边界

chunk 太小会丢失定义和条件，chunk 太大会引入无关段落和上下文成本。合同、表格、代码和制度文档需要不同策略：

- 标题与正文成组；
- 条款编号作为结构键；
- 表格保留行列关系；
- 代码块不在任意标点截断；
- 页面和坐标保留；
- 跨页条款有 parent-child 关系；
- chunk 记录原文范围和文档 revision。

chunk 不是纯文本长度问题，也是证据边界问题。一个数字脱离单位和适用条件后，向量相似度再高也可能造成错误回答。

例如合同第 7 页有一张付款表，第 8 页续写“验收后 30 日内支付”，而补充协议第 2 页又把比例改为 40%。若 parser 只留下孤立数字，或 chunk 将表头、条款条件和补充协议关系切断，检索可能找回“40%”却无法说明它适用哪个阶段。可把每个 chunk 关联到父条款、页码范围和文档 revision，并保留修订间的 supersedes 关系；读者才能区分它是当前条款、历史版本还是另一笔付款。

### 10.8.3 混合检索和重排

向量检索擅长语义相似，倒排检索擅长精确词、编号和罕见实体。可以先取两个候选集合，再用融合和 rerank 产生上下文。对于候选文档集合，Recall@k 可以定义为：

~~~math
\mathrm{Recall@k}
=
\frac{
N_{\mathrm{queries\ with\ relevant\ item\ in\ top\ k}}
}{
N_{\mathrm{queries}}
}
~~~

这里要求 `N_queries > 0`；若评估集没有查询，Recall@k 应报告为未定义，而不是 0。它只衡量相关证据是否出现，不衡量模型是否使用了证据。重排提高上下文精度可能增加延迟，也可能把相似但版本错误的文档排得很高，因此必须把版本和权限作为硬过滤条件，而不是交给相似度。

当向量相似度和倒排检索的原始分数不在同一尺度时，可以先融合名次，再由 reranker 检查候选与问题的具体相关性。倒数排名融合（Reciprocal Rank Fusion, RRF）对文档 d 的分数可写为 `RRF(d)=Σ_j 1/(k+rank_j(d))`，其中 `rank_j` 是 d 在第 j 个检索列表中的名次，k 是平滑常数；它融合名次而不直接相加 BM25 与向量原始分数。[RRF 原论文](https://doi.org/10.1145/1571941.1572114)、[Elasticsearch RRF 文档](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

手算一个构造例子：取 `k=60`，文档 A 在向量检索排第 1、倒排检索排第 30，则 `1/61+1/90≈0.0275`；文档 B 分别排第 3 和第 1，则 `1/63+1/61≈0.0323`，因此融合分数将 B 排在 A 前面。这个名次只决定候选顺序，不证明 B 的文本支持答案；A/B 名次是教学输入，不是检索实测。

例如 20 条标注查询中，有 16 条在 top-k 找到至少一条相关证据，则 `Recall@k = 16/20 = 0.80`。这说明检索覆盖率为 80%，并不能推出最终回答正确；剩余 4 条也应保留为可归因的漏召回样本，而不是从端到端准确率分母中删除。

### 10.8.4 权限必须后于身份、早于生成

检索流程应先得到用户和租户上下文，再过滤可访问文档。不能先把所有文档召回，再让模型决定哪些能看。缓存、向量副本、摘要和上下文也要继承权限版本。

文档访问可以表示为：

~~~math
\mathrm{CanRead}(u,d,t)
=
\mathrm{Identity}(u)
\land
\mathrm{TenantMatch}(u,d)
\land
\mathrm{Policy}(u,d,t)
\land
\mathrm{RevisionVisible}(d,t)
~~~

u 是用户，d 是文档，t 是请求时间或策略版本。这个判断必须由服务端执行，不能依赖模型在回答中自行遵守。

权限撤销还会产生失效竞态：文档库可以已经更新 ACL，而答案缓存仍保留旧引用；因此缓存命中后要用当前策略重新授权，并让权限 revision 参与缓存键。索引重建若有延迟，在线检索的当前权限过滤仍须先挡住被撤销文档；删除请求则继续传播到 OCR、chunk、向量索引、响应缓存、备份和评估 artifact。单独删除原始 PDF 并不能证明副本已不可见。

### 10.8.5 证据选择和引用支持

设回答包含 N_claim 个可验证 claim，其中有 N_supported 个能由实际证据支持，则引用支持率可以写成：

~~~math
\mathrm{CitationSupport}
=
\frac{N_{\mathrm{supported}}}{N_{\mathrm{claim}}}
~~~

只有 `N_claim > 0` 时引用支持率才有定义；没有可验证 claim 的回答应单独报告为“无适用 claim”，不能把它当作 100% 支持。支持不只是“附了一个链接”。需要检查：

- 引用文档版本正确；
- 证据包含 claim 所需事实；
- claim 没有超出证据范围；
- 数字、单位、时间和条件一致；
- 冲突来源已被处理；
- 用户有权访问该来源。

证据不足时，系统应允许“不足以判断”或要求补充条件，而不是用语言流畅度填补空白。

若一个回答有 10 个可验证 claim，其中 8 个被引用证据充分支持，则 `CitationSupport = 8/10 = 80%`，另 2 个 claim 仍需修复或降级为不确定。若回答没有可验证 claim，指标应记为未定义；把空回答或纯寒暄计成 100% 支持会扭曲评估。

把这些步骤串起来看，用户问“验收后应付多少”时，服务先用当前身份和权限确定可见合同，再组合精确条款编号检索与语义检索；索引层过滤已失效 revision，reranker 只在授权且有效的候选中排序。上下文构建器把应付比例、付款阶段、例外条件和页码作为同一个证据对象，生成后再逐条核对 claim 与原文。若主合同和有效补充协议冲突，系统应展示冲突并请求澄清，不能把两个比例平均或挑语言更像答案的一段。

### 10.8.6 RAG 失败归因

RAG 错误可以分层：

1. 文档没入库；
2. 解析、OCR 或表格结构错误；
3. chunk 丢失上下文；
4. 权限过滤漏掉正确文档或放入错误文档；
5. retriever 没召回；
6. reranker 排序错误；
7. context builder 截断或去重错误；
8. 模型忽略证据；
9. 引用绑定错误；
10. 数据版本过期。

每层都要有对应的离线样本和 trace 字段。只看最终答案准确率无法知道应该改 parser、retriever 还是生成模型。

安全测试也应把检索本身纳入攻击面。USENIX Security 2026 的一项研究构造了用于提高恶意片段检索概率的 query trigger 与攻击内容，并在其 embedding-retrieval 设置中评估了间接提示注入；其测试没有覆盖所有 hybrid retrieval/reranker 组合，论文中的攻击率不能当作任意企业 RAG 的发生率。[USENIX Security 2026 论文](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan)

失败归因可以用一组带标注的查询走一遍。假设 50 条查询都带有一段经权限确认的 gold evidence，top-k 只有 40 条召回了相关片段，则权限约束后的 `Recall@k = 40/50 = 80%`。逐条检查剩余 10 条，发现 6 条在 PDF 表格解析时丢失合并单元格，另 4 条的相关 chunk 没进入候选；这时应先修解析器和召回链路，而不是让生成模型改写提示词来掩盖缺证据。再看 40 条已召回查询，若只有 34 条答案的引用确实支持结论，则召回条件下的支持率为 `34/40 = 85%`，端到端证据成功率则是 `34/50 = 68%`。两项比例的分母不同：前者诊断生成与引用，后者把漏召回也留在业务结果中；这些计数是教学输入，不是线上基准。

为让这种区分可复核，trace 至少要保存经过当前 ACL 过滤后的候选 ID、文档与解析 revision、各检索器名次、reranker 输出、最终 context 中的证据 ID，以及生成回答引用到的 ID。日志无需保存完整合同正文；受限 artifact 可以按权限和保留期限另行读取。若 trace 只有最终答案和一个总准确率，团队就无法判断变化来自解析、权限过滤、召回、排序还是生成。

## 10.9 Agent 平台：让模型在边界内完成任务

### 10.9.1 Agent 的真实状态

Agent 不是一个无限循环。它可以表示为：

~~~text
TASK_CREATED
    -> PLAN_PROPOSED
    -> ACTION_AUTHORIZED
    -> TOOL_RUNNING
    -> TOOL_RESULT_RECEIVED
    -> EVIDENCE_CHECKED
    -> NEXT_STEP or TASK_COMPLETED
~~~

异常状态包括 WAITING_USER、WAITING_REVIEW、RETRYABLE_ERROR、UNKNOWN_RESULT、UNKNOWN_EXTERNAL_EFFECT 和 ABORTED。每一步都要保存 task_id、step_id、tool name、参数摘要、策略版本、结果状态和下一步原因。

状态循环可以由应用自己实现，也可以交给具体 Agent runtime 管理。以截至 2026-11-06 的 OpenAI Agents SDK 文档为例，直接调用 Responses API 时由应用持有循环、工具分发和状态处理；SDK runtime 则可管理 turns、工具执行、guardrails、handoffs 和 sessions。这只是一个产品的接口边界，不能据此推导所有 Agent 都应采用同一种状态 owner。[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)

### 10.9.2 工具契约

一个工具定义不应只有名称和自然语言描述，还应包含：

- 输入 schema；
- 输出 schema；
- 资源类型；
- 允许动作；
- 租户和用户范围；
- 风险等级；
- 超时；
- 幂等语义；
- 可重试错误；
- 审计字段；
- 结果确认方法。

模型输出 JSON 通过 schema 校验，只说明参数形状正确。服务端还要检查资源是否属于当前用户、金额是否在范围内、动作是否被授权、当前版本是否仍有效。

schema 的严格程度也和具体 API 有关。例如 OpenAI Function Calling 的 strict mode 按声明的 schema 约束参数形状，但 schema 通过并不表示调用者有权访问该资源；交易授权仍须在服务端校验并绑定到用户确认的关键数据。[OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling)、[OWASP Transaction Authorization](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)

### 10.9.3 工具调用和程序化工作流

当模型需要对多个工具结果做过滤、聚合或循环处理时，可以把部分中间计算放入受限的程序化工作流或沙箱，减少模型往返和上下文污染。这种模式的核心收益是 token 和 round trip 减少，不是把权限交给生成的程序。

程序化步骤仍需要：

- 固定可用 API；
- 输入输出 schema；
- 沙箱和资源预算；
- 网络与文件系统限制；
- 结果验证；
- 审计和可重放；
- 失败后的人机协作。

模型生成了一段程序，不代表程序可信。程序必须像普通不可信代码一样被隔离和验证。

### 10.9.4 工具状态：区分查询结果未知与外部副作用未知

工具调用应至少区分：

- NOT_STARTED：尚未发出；
- RUNNING：服务端已接收；
- SUCCEEDED：有下游凭证；
- FAILED：明确未完成；
- UNKNOWN_RESULT：只读查询未取得回包，查询结果暂时未知，但本身没有写副作用；
- UNKNOWN_EXTERNAL_EFFECT：写操作已发出，无法确定下游是否执行；
- COMPENSATED：已执行补偿或人工核对。

超时不能直接映射为失败。只读查询的 UNKNOWN_RESULT 可以按 request_id 有界重试，同时重新检查权限和数据版本；不可幂等写操作的 UNKNOWN_EXTERNAL_EFFECT 应先查询业务状态或交人工处理，不能直接重发。HTTP 语义也区分幂等与非幂等请求：通信失败后可自动重试幂等请求；非幂等请求则应先能确认原操作未生效或请求语义确实幂等。[RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2)

业务执行器应是操作最终状态的 owner。模型只能提出参数，应用分配稳定的 operation_id 并把结果交给工具执行器；只有执行器收到下游回执或通过业务查询确认后，才把状态写成 SUCCEEDED。Agent 收到确认结果后可以继续计划下一步；它不能根据自己上一次生成的自然语言文本，把“已提交”当作执行凭证。

为降低客户端断线或网关重试导致的重复写入，可以给同一个逻辑动作复用稳定的幂等键（idempotency key），并把键绑定租户、调用者、动作类型和规范化参数摘要；同一键若收到不同参数，应拒绝而不是覆盖原动作。每次尝试仍须重新验证当前身份与授权，幂等键不能充当权限凭证。如果下游不支持幂等写入，平台只能先查询业务状态、等待回执或进入人工核对，不能假定本地去重就阻止了外部系统重复执行。这类意图令牌与安全重试的应用设计可参考 [AWS Builders' Library：Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)；HTTP 幂等语义本身不会替业务系统实现状态核对。

### 10.9.5 Agent 预算

长周期任务要限制：

- 最大步骤数；
- 最大模型调用数；
- 最大输入和输出 token；
- 最大工具调用数；
- 最大并行度；
- 最大外部请求数；
- 最大运行时间；
- 最大人工审核次数；
- 最大费用。

如果模型不断重试，预算本身是安全控制。预算耗尽后要进入明确终态，并把已经完成的证据和未完成的目标告诉用户。

例如一次合同问答可设为最多 5 个 Agent 步骤、3 次模型调用、4 次工具调用和 90 秒总时限；若第二次只读状态查询超时，结果进入 UNKNOWN_RESULT 并消耗剩余时限做有界重试；若写操作已经发出，则进入 UNKNOWN_EXTERNAL_EFFECT 并查询业务凭证。两种状态的恢复方式不同。数字是设计假设，需根据任务风险、依赖延迟和成本实测调整。

### 10.9.6 Prompt injection 的系统边界

网页、邮件、PDF、工具返回和 OCR 内容都应视为不可信数据。系统可以让模型分析它们，但不允许它们修改：

- system/developer policy；
- 用户身份；
- 租户；
- 工具权限；
- 审批状态；
- 审计配置。

高风险动作必须由服务端做权限检查和确认。仅仅在 prompt 中写“不要相信网页指令”不能代替权限和执行器设计。OWASP 2025 Top 10 将 Prompt Injection 单列为 LLM 应用风险；该分类提示设计者把外部内容和权限执行面一起检查，不构成某项防护有效性的实测证据。[OWASP LLM01:2025 Prompt Injection](https://genai.owasp.org/llm-top-10/)

### 10.9.7 在固定工作流和 Agent 循环之间选择

系统设计时不要把“需要调用模型”直接等同于“需要 Agent”。如果步骤和分支在设计时已经确定，例如上传合同后固定执行解析、权限标注、索引、抽样评估和发布，工作流引擎更容易审计、重放和设定终态。Agent 循环适合下一步确实要根据新观察动态选择的任务，例如先检索哪个条款、是否再查补充协议、证据是否已充分。很多系统会采用固定外层工作流，把模型的自由度限制在一个有界的搜索或整理步骤中。

衡川的固定任务“查合同付款日期”可以按既定顺序运行身份校验、权限过滤、主合同与补充协议检索、版本关系检查和引用生成；无需让 Agent 任意选择工具。若用户改问“比较两家供应商的付款风险并找出需要确认的条件”，模型可以在 `search_contract`、`get_policy_status` 等只读工具中逐步选下一步，但每一步仍需服务端重验资源权限，且最多执行预算中指定的轮数。

这一区别能在面试中用分支数解释。假设每一步可从 3 种工具中任选一种、最多规划 5 步，粗略组合上界为 `3^5 = 243` 条工具序列；这不是实际运行次数，因为权限、结果和停止条件会剪掉大量分支，却说明“多给几个工具”会迅速扩大测试空间。为每个分支记录输入、工具状态、证据和停止原因，才能判断 Agent 是找到了答案、耗尽预算，还是陷入重试。

选择时应比较任务开放程度、延迟和成本预算、可用工具的副作用、错误是否可恢复、轨迹是否可复现。若一次确定性函数调用就能完成任务，复杂 Agent 循环只会新增模型调用和失败状态；若真实任务需要根据新证据改变路线，过度固定的工作流又会把大量分支写成脆弱的硬编码。常见的折中是让代码持有身份、权限、预算和终态，让模型只负责有限范围内的计划与下一步候选。

## 10.10 多模态助手：把媒体转成可验证证据

### 10.10.1 图片处理链路

一张图片进入系统后，可能经过：

~~~text
upload
    -> type/size/malware check
    -> EXIF orientation and color normalization
    -> resize/crop or tile
    -> OCR/layout/region detection
    -> vision encoder
    -> VLM or structured extractor
    -> schema/evidence validation
~~~

图像分辨率越高，小字和小物体可能越容易识别，但视觉 token、prefill、显存和网络传输也会增加。动态分辨率和局部 crop 可以节省成本，却要记录裁剪区域，避免模型回答来自被截断的上下文。

### 10.10.2 文档和票据字段抽取

票据金额、税率和日期是高风险字段。输出不应只有：

~~~text
amount: 128.00
~~~

而应包含：

~~~text
field: amount
value: 128.00
source: document_17
page: 1
region: [x1, y1, x2, y2]
ocr_span: span_42
confidence: uncertain
review: required
~~~

如果没有区域或 OCR span，系统应把结果标记为待复核，不能让模型根据常见票据格式补全数字并自动进入报销流程。

例如一张票据的商品小计为 120.00 元、税额为 6.00 元，OCR 把总额读成 128.00 元，而视觉抽取器读成 126.00 元。服务可以计算 `120.00+6.00=126.00`，发现 OCR 字段与算术校验不一致，但不能据此把 126.00 当成已证实的票面金额；它应保存原图版本、总额框的坐标、两个识别结果及各自的文本/视觉证据，并把该字段送人工复核。算术校验用于发现矛盾，不替代原始凭证。

模型给出的 `confidence` 也不能默认当成校准概率。只有在目标票据类型、扫描质量、语言、金额区间和 OCR/VLM 版本上用标注集检查过置信度与错误率的对应关系，才能据此设置自动通过阈值；分布改变时要重做校准。否则用“0.98 置信度”自动通过字段校验，只是把不确定性藏进了一个看似精确的数字。

图像抽取的回归集应分开覆盖旋转、模糊、小字、遮挡、低对比度和多栏表格，并分别报告字段精确率、关键字段错误率、人工复核率和视觉 token/延迟。若只测清晰正向扫描图，平均准确率无法说明手机拍摄或票据折痕下的表现；若只看字段准确率，也可能漏掉系统输出了错误页码或错误区域。

### 10.10.3 音频和视频

实时语音服务需要同时处理音频帧、VAD、partial/final ASR、用户打断、LLM 流式生成和 TTS 播放。视频任务还要处理抽帧、时间窗口、时序一致性、存储和异步进度。

多模态系统不应把所有媒体都直接转成长文本。媒体摘要会丢失区域、时间和不确定性。证据对象应保留 page、region、timestamp、frame_id、OCR/ASR revision 等字段。

例如会议录音中，partial ASR 在 1.2 秒把金额识别成“十五”，final ASR 在 1.8 秒修正为“五十”。如果用户问“最终确认的金额是多少”，系统应绑定 final transcript 的 revision 和音频时间段，不能把早到的 partial 当成稳定结论。视频摘要也要保留抽帧时间：只检查 00:01 和 00:04 两帧，不能证明两者之间无人移动了物体；涉及动作顺序时，应报告采样范围或请求更密集的片段分析。

### 10.10.4 多模态成本

设一张图片经过视觉编码产生 N_img 个 token，文本输入为 N_text，工具和检索结果为 N_extra，则请求的上下文长度近似为：

~~~math
N_{\mathrm{context}}
=
N_{\mathrm{img}}
+
N_{\mathrm{text}}
+
N_{\mathrm{extra}}
~~~

N_img 由分辨率、patch、crop 和压缩策略共同决定。多图片、视频和音频还需要记录媒体数量、时长和采样率。仅按文本 token 计费会低估多模态处理成本。

举一个不绑定特定模型的账本例子：某次请求的实际计数记录为图像与 crop 共 1,024 个视觉 token、文本 1,800 个 token、检索和工具上下文 2,700 个 token，则 `N_context = 1,024 + 1,800 + 2,700 = 5,524` 个 token。这里的视觉 token 数是假设该模型/预处理器已经给出的输入，不表示每张图片固定产生 1,024 个 token；更换分辨率、裁剪规则或模型后必须重新计数。

当预处理器把原图旋转、resize 或 crop 后送入模型时，输出坐标要能映回原图区域；否则审阅者看到的框和原件不一致。视觉 token 总数也不能代替图像分辨率、crop 数、OCR 版本、音频时长和视频采样率。对高分辨率票据，至少要把字段质量、人工复核率和上下文成本一起比较，才能决定是加大分辨率、更多 crop，还是直接转人工。

## 10.11 评估平台：让变化可比较、可回归

### 10.11.1 评估对象的版本向量

一次评估至少绑定：

~~~text
model_revision
tokenizer_and_template
system_policy
prompt_revision
retrieval/index_revision
tool_schema_and_runtime
dataset_revision
decoder_settings
judge_revision
hardware/runtime
~~~

如果只记录模型名称，同名模型的模板、工具、知识库和解码参数变化会混在一起。评估平台应保存原始输出、工具轨迹、引用、人工标注和评审分歧，而不是只保存一个最终分数。

数据集中的一条样本还要有稳定的 sample_id、任务类别、输入和预期行为、允许的工具、权限上下文、文档 revision、grader/rubric revision 与切分标记。更新答案标准或输入文档时创建新 dataset revision，不要原地覆盖旧样本；否则同名 run 的分数变化可能来自数据改了，而不是系统变好。需要保留原始材料时，使用受控 artifact 引用并执行脱敏、权限和保留期限策略。

### 10.11.2 分层评估

衡川可以分为：

1. 单元评估：tokenizer、parser、chunk、schema；
2. 模型评估：文本、视觉、工具参数和安全行为；
3. 组件评估：retriever、reranker、OCR、judge；
4. 端到端评估：任务成功、引用、延迟和成本；
5. 线上评估：真实反馈、申诉、漂移和事故。

组件分数不能直接替代端到端结果。检索 Recall@k 提高，可能让上下文更长、延迟更高；模型答案更准确，可能是因为它看到了不该看的文档；安全分类器召回更高，可能增加正常业务的误拒。

评估 run 也要把“被测系统错误”和“测量工具错误”分开。某个输出格式不合法，通常是被测系统失败；运行器 API 错误是样本执行异常；judge 无法给出可解析结论则是 unscored。Inspect 的 scoring policy 把这些结果分别记录，并说明错误或 unscored 项不进入该 scorer 的准确率分母，但要单独汇报覆盖情况。[Inspect AI Scoring Policy](https://inspect.aisi.org.uk/scoring-policy.html)

例如 100 条样本中，84 条判为正确、7 条判为错误、6 条发生 scorer 执行错误、3 条 judge 无法给结论；若 run 配置允许继续，则已评分准确率为 `84/(84+7)=84/91≈92.3%`，评分覆盖率为 `91/100=91%`。报告只写 92.3% 会隐去 9 条测量缺口；若发布条件要求完整覆盖，这一轮应标成证据不足，而不是直接放行。这里的计数是教学输入，Inspect 文档只支持一种框架处理范例，不规定通用发布标准。

### 10.11.3 配对差异

对同一输入 i，基线和变体的任务指标分别为 M_i^b 和 M_i^v，可以定义：

~~~math
\Delta_i
=
M_i^v-M_i^b
~~~

平均配对差异为：

~~~math
\bar{\Delta}
=
\frac{1}{n}
\sum_{i=1}^{n}\Delta_i
~~~

配对比较能减少样本难度差异，但仍需要检查随机种子、解码、工具状态和评估器是否一致。除了平均差异，还要报告差异的分布、置信区间和按任务桶的变化。

例如对 4 条相同输入，基线结果为 `[1, 0, 1, 0]`，变体结果为 `[1, 1, 1, 0]`，逐条差值为 `[0, 1, 0, 0]`，平均配对差为 `(0+1+0+0)/4 = 0.25`，即该教学样本上提高 25 个百分点。四条样本远不足以证明真实改进；还需要更大样本、分层结果和不确定性区间。

### 10.11.4 人评和 judge

自动 judge 适合大规模筛选，但可能偏好长答案、某种风格和自信语气。人工评审能处理上下文、意图和证据，但成本高、分歧大。高影响任务可以采用：

~~~text
全量自动筛选
    -> 低置信/高严重度/版本差异样本
    -> 双人盲评
    -> 分歧仲裁
    -> 专家复核
~~~

评估器也需要 gold set、对抗样本、长度偏差测试和跨语言校准。不能用与被测模型完全相同的错误上下文证明它自己正确。

截至 2026-11-06，OpenAI 的 Agent 评估指南把单条 trace 用于调试，并建议在需要重复比较时转为 dataset 和 eval runs；这是厂商工具流程的例子，不是所有平台必须照搬的 API。[OpenAI：Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals) 一项 2026 年 rubric-artifact 研究还发现，在其健康与研究问答样本中，不读取候选回答的 probe 仍能预测部分 judge 标签；这提示团队要检查 rubric 或上下文是否泄漏了答案线索，但不证明所有领域的 LLM judge 都不可靠。[Rubric Artifacts 研究](https://arxiv.org/abs/2609.02942v1)

### 10.11.5 回归集和污染

线上 bad case、红队样本和公开 benchmark 都可能进入训练或 prompt 优化。评估平台应维护：

- 公开集；
- 保密集；
- 动态新鲜集；
- 线上抽样集；
- 反事实集；
- 污染监测集。

固定集适合版本比较，动态集更能测试泛化。发布时不应只看公开分数下降或上升，而应看保密、时间切分和真实任务分布。

## 10.12 安全审核：把风险变成系统动作

### 10.12.1 输入、上下文、输出和动作

安全审核不是一个输出过滤器。至少要覆盖：

1. 输入：用户请求、上传媒体和外部内容；
2. 上下文：RAG 文档、历史摘要、工具返回；
3. 输出：文本、结构化参数、引用和媒体；
4. 动作：工具调用、写入、通信和外部副作用；
5. 数据：日志、缓存、trace、训练和评估 artifact。

不同层的控制不同。输出文本可以被拦截或改写，工具动作必须由授权和执行器控制，日志泄露需要数据治理，RAG 越权需要权限过滤。

### 10.12.2 风险分类和动作策略

可以把风险分类映射到动作：

| 风险 | 低风险动作 | 高风险动作 |
| --- | --- | --- |
| 正常问答 | 直接回答 | 证据不足则澄清 |
| 敏感但合法 | 降低细节、保留帮助 | 人工或专业转介 |
| 隐私内容 | 最小化展示 | 阻止跨租户访问 |
| 工具查询 | 只读、审计 | 需要授权和确认 |
| 不可信指令 | 当作数据分析 | 不得改变策略 |
| 不可逆动作 | 预览参数 | 明确确认和补偿 |
| 高风险能力 | 高层防御信息 | 受限访问和人工复核 |

“拒绝更多”不是完整安全目标。系统还要测正常帮助、误拒、证据支持和动作后果。

### 10.12.3 安全指标的分母

设 `N_harmful > 0` 的禁止请求中，给出实质危险帮助的数量为 N_compliant；设 `N_benign > 0` 的合法敏感请求中，不必要拒绝数为 N_overrefuse：

~~~math
\mathrm{HarmfulCompliance}
=
\frac{N_{\mathrm{compliant}}}{N_{\mathrm{harmful}}}
~~~

~~~math
\mathrm{OverRefusal}
=
\frac{N_{\mathrm{overrefuse}}}{N_{\mathrm{benign}}}
~~~

分母必须按语言、轮次、风险等级、是否带工具和是否多模态分层。若某个分层没有样本，应报告未定义；不能用 0 代替缺少观测。一个总体值不能证明所有租户和场景都安全。

例如 200 条禁止请求中有 4 条得到实质危险帮助，则 `HarmfulCompliance = 4/200 = 2%`；600 条合法敏感请求中有 18 条被不必要拒绝，则 `OverRefusal = 18/600 = 3%`。两个比率使用不同请求集合，不能简单相加成“总错误率”；还要按语言、工具与风险层分别报告，避免总体分母掩盖少数严重桶。

### 10.12.4 灰度和发布条件

发布前应形成证据包：

- 目标场景和不适用场景；
- 模型、策略、工具和索引版本；
- 正常、边界、禁止和对抗样本；
- 红队发现和回归结果；
- 隐私、权限和日志审计；
- 延迟、成本和人工处理能力；
- 未解决风险和责任人；
- 灰度范围和回滚方法。

灰度不是只把流量切 1%。还要限制租户、工具权限、数据范围和高风险动作，并监控申诉、未知状态和真实任务失败。

NIST 的生成式 AI 风险管理画像建议在接近部署的条件下展示系统性能、安全和可靠性，并评估所用 TEVV 指标与流程是否有效；它是自愿风险管理参考，不替团队规定具体上线分数或法律门槛。[NIST AI 600-1](https://doi.org/10.6028/NIST.AI.600-1)

因此，高风险系统宜把关键风险设成独立硬门槛，而不是混入总分。例如教学方案可以要求合同金额字段准确率至少 98%、300 个跨租户挑战样本中越权披露为 0，并同时满足 P95 延迟和人工复核容量目标；这些阈值必须由业务风险负责人确定，不是 NIST 或 OWASP 给出的通用数字。即使总体回答分数提高，只要越权、未知副作用或关键金额错误超过门槛，就应阻止灰度或回滚。

即使 300 个挑战样本中观察到 0 次越权，也只能报告“这组样本没有复现”，不能写成真实越权概率为零。报告还应列出租户组合、权限变更类型、缓存冷热状态、文档 revision 和攻击方式覆盖情况；如果 300 条样本只是同一个模板改写，样本数看似很大，实际覆盖仍可能很窄。发布决定应同时使用边界挑战集、常规任务集、线上申诉和事故信号，并保留未覆盖风险及责任人。

## 10.13 容量规划和成本账本

### 10.13.1 GPU 容量的第一版估算

假设平均每秒请求数为非负的 lambda，每个请求平均输入 token 为非负的 T_in、输出 token 为非负的 T_out，目标 worker 的实测输入吞吐和输出吞吐分别为正的 C_in 和 C_out，可以分别估算：

~~~math
G_{\mathrm{in}}
\approx
\frac{\lambda T_{\mathrm{in}}}{C_{\mathrm{in}}}
~~~

~~~math
G_{\mathrm{out}}
\approx
\frac{\lambda T_{\mathrm{out}}}{C_{\mathrm{out}}}
~~~

对同一共享 worker 池，`max(G_in,G_out)` 只能作为第一版下界，而且 `C_in` 与 `C_out` 必须在同一目标混合负载下测得；单独的输入/输出 benchmark 不能直接合成为联合容量。若 prefill 和 decode 分成不同资源池，就分别规划两池的容量，不能把 `max` 当作总 GPU 数。随后还要加入峰值、模型副本、KV Cache、排队和故障余量；不同 GPU 型号的资源也不能只按卡数相加。

### 10.13.2 模型权重和 KV 状态

模型权重内存近似为：

~~~math
M_{\mathrm{weights}}
\approx
P \times b_{\mathrm{weight}}
~~~

P 是参数量，b_weight 是每个参数的字节数。总显存还包括：

~~~math
M_{\mathrm{total}}
=
M_{\mathrm{weights}}
+
M_{\mathrm{KV}}
+
M_{\mathrm{activation}}
+
M_{\mathrm{workspace}}
+
M_{\mathrm{runtime}}
~~~

量化主要减少部分权重或缓存占用，不会自动让所有中间 buffer 和通信内存按同样比例下降。容量规划要以目标 runtime、并发、序列长度和硬件实测为准。

### 10.13.3 任务成本

一次任务成本可以拆成：

~~~math
C_{\mathrm{task}}
=
C_{\mathrm{model}}
+
C_{\mathrm{retrieval}}
+
C_{\mathrm{tool}}
+
C_{\mathrm{storage}}
+
C_{\mathrm{review}}
+
C_{\mathrm{retry}}
~~~

如果成功任务数为 `N_success > 0`，单位成功任务成本为：

~~~math
C_{\mathrm{success}}
=
\frac{C_{\mathrm{task,total}}}{N_{\mathrm{success}}}
~~~

C_retry 不能省略。一个系统平均单次请求便宜，但由于无效重试、工具重复或人工返工，单位成功成本可能更高。

例如一天的模型、检索、工具、存储和复核总成本为 80 元，完成 800 个用户目标，则单位成功任务成本为 `80/800 = 0.10` 元；若同样花费只完成 640 个目标，则为 `80/640 = 0.125` 元。两个系统可能有相同的“每请求 0.08 元”表面成本，成功率不同却使真实单位成功成本不同。此处金额和任务数均为教学输入。

### 10.13.4 一个教学估算

假设每秒 10 个请求，平均输入 2,000 token、平均输出 400 token，则输入负载为 20,000 input token/s，输出负载为 4,000 output token/s。对分离的 prefill/decode 资源池，假设目标长度分布下实测每个 prefill worker 为 10,000 input token/s、每个 decode worker 为 1,500 output token/s，则：

~~~math
D_{\mathrm{in}}
=
10\times2,000
=
20,000\ \mathrm{input\ token/s}
~~~

~~~math
D_{\mathrm{out}}
=
10\times400
=
4,000\ \mathrm{output\ token/s}
~~~

理想情况下，prefill 需要 `ceil(20,000/10,000)=2` 个 worker，decode 需要 `ceil(4,000/1,500)=3` 个 worker。若按 1.5 倍峰值规划，并要求每个资源池可容忍一个 worker 故障，则 prefill 至少为 `ceil(1.5×20,000/10,000)+1=4` 个，decode 至少为 `ceil(1.5×4,000/1,500)+1=5` 个。这里的吞吐值是假设目标部署下的实测值，不能从模型宣传页推出。若 prefill 和 decode 共用 worker，不能把两个分开的吞吐基准机械相除后相加；应在同一请求长度分布和并发下测联合容量，再核对长上下文的 KV Cache、排队、量化、检索和审核开销。

这个数字不能从模型宣传页直接推出。吞吐必须绑定硬件、量化、上下文长度、batch、解码策略、模板、工具和成功率。

### 10.13.5 质量和成本的联合选择

如果版本 A、B 的任务成功率分别为 q_A、q_B，且对应成本 `c_A > 0`、`c_B > 0`，可以比较：

~~~math
\mathrm{QualityPerCost}
=
\frac{q}{c}
~~~

但这个比值不能替代硬约束。高风险任务可能要求 q >= q_min，隐私和越权率还必须低于独立阈值。产品选择要在满足硬约束的候选中比较成本和体验，而不是让廉价但不合格的系统通过平均分获胜。

## 10.14 可靠性、降级和灾备

### 10.14.1 失败分类

在线系统失败可以分为：

- 可重试的网络或瞬时服务错误；
- 输入不可处理；
- 权限拒绝；
- 模型资源不足；
- 检索无证据；
- 工具状态未知；
- 输出审核拦截；
- 人工队列拥塞；
- 数据或版本不一致；
- 依赖完全不可用。

每类失败都应有不同的用户提示、重试、降级和告警。把所有失败返回“请稍后重试”会掩盖权限、数据和模型错误。

### 10.14.2 降级阶梯

衡川可以设计：

~~~text
完整路径
    -> RAG + rerank + strong model + evidence
降级一
    -> RAG + small model + evidence
降级二
    -> exact search + cited passages
降级三
    -> acknowledge unavailable + create review task
隔离路径
    -> block risky tool/action + preserve audit trace
~~~

降级不能把高风险任务静默变成低质量答案。用户需要知道当前回答缺少哪些能力，系统也要记录降级原因和影响范围。

### 10.14.3 重试预算和放大

如果每层服务都独立重试，内部调用会形成乘法放大。假设网关、检索和模型层各自的最大尝试次数 `r_gateway`、`r_retrieve`、`r_model` 都是包含首次尝试在内的正整数，最坏情况下内部尝试数可能接近：

~~~math
A_{\mathrm{retry}}
=
r_{\mathrm{gateway}}
\times
r_{\mathrm{retrieve}}
\times
r_{\mathrm{model}}
~~~

其中 r 是包括首次尝试在内的最大次数。实际系统应由一个任务级 retry budget 统一管理，区分可重试错误和不可重试错误，并避免对未知副作用动作自动重试。

例如网关最多尝试 2 次、检索最多尝试 3 次、模型最多尝试 2 次，且每一层重试都会重新进入下层，那么最深处的模型调用在坏路径上可能达到 `2×3×2 = 12` 次；这里每个次数都包含首次尝试。任务级预算可以把整条链限制在例如 5 次重试以内。Google SRE 描述过 per-request 与 per-client retry budgets 来限制过载时的重试放大；它的具体阈值属于其系统实践，不是通用默认值。[Google SRE：Handling Overload](https://sre.google/sre-book/handling-overload/)

### 10.14.4 背压和公平

当 GPU、检索或人工审核队列饱和时，系统需要背压：

- 限制新任务进入；
- 按 token 和风险估算队列；
- 保护关键租户；
- 暂停低优先级长任务；
- 限制 Agent 并行；
- 返回可重试时间或异步 task_id。

公平调度不能只按请求数。一个长上下文请求可能占用多个短请求的资源；高风险人工审核也需要保留队列容量。

### 10.14.5 灾备和数据删除

灾备副本需要明确 RPO、RTO、加密、访问权限和删除传播。删除一个文档时，要同步考虑：

- 原始对象；
- OCR 和解析产物；
- embedding 和倒排索引；
- 缓存；
- 备份；
- 日志和 trace；
- 评估样本；
- 模型输入或摘要 artifact。

灾备不是“永远保留所有数据”。保留和删除都必须有生命周期、审计和责任主体。

## 10.15 可观测性：把一次回答变成可复盘事件

### 10.15.1 Trace 的层次

一次衡川请求的 trace 可以包含：

~~~text
request
    -> gateway/auth
    -> policy decision
    -> preprocess/OCR
    -> retrieval query
    -> retrieved evidence IDs
    -> rerank
    -> prompt/template revision
    -> model call and usage
    -> tool call/status
    -> output claims and citations
    -> guard decision
    -> user-visible response
~~~

trace 不等于把所有原文都保存下来。敏感内容应采用脱敏、哈希、加密 artifact、访问审批和有限保留。日志字段要支持排障，但不能为了方便把客户合同和完整 prompt 永久存储。OpenTelemetry GenAI spans 的当前语义约定也把系统指令、用户消息和模型输出视为可能敏感且体积较大的内容，建议 instrumentation 默认不采集、由使用方显式选择；该动态约定不会自动替 SDK 实施访问控制或保留策略。[OpenTelemetry GenAI spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)

### 10.15.2 指标四层

可以按四层组织指标：

1. 系统层：QPS、错误率、CPU/GPU、队列、可用性；
2. 模型层：TTFT、TPOT、token、停止原因、格式正确率；
3. 任务层：任务成功、引用支持、工具成功、人工升级；
4. 风险层：越权、注入、隐私告警、误拒、未知状态和申诉。

单一平均值会隐藏长尾和分布漂移。至少按租户、语言、模型版本、任务类型、输入长度和风险等级分桶。

### 10.15.3 版本关联

每个用户可见结果应能够关联：

- request_id；
- model_revision；
- tokenizer/template revision；
- policy revision；
- index/document revision；
- tool schema/runtime revision；
- eval/judge revision；
- hardware/runtime。

版本字段缺失时，线上 bad case 无法复现，灰度和回滚也会变成猜测。

## 10.16 完整案例：合同助手的一次权限与超时事故

### 10.16.1 基线设计

衡川支持合同查询和票据字段抽取。合同索引按租户隔离，模型只能调用 search_contract 和 get_policy_status 两个只读工具。系统使用响应缓存和 prefix cache，文档更新通过索引 revision 发布。

初始设计存在三个隐患：

1. 检索结果缓存键没有包含权限版本；
2. 工具超时统一映射成失败，模型随后根据上下文继续生成；
3. 评估只测回答准确率，没有测证据支持和跨租户访问。

### 10.16.2 事故时间线

一次权限变更后，租户 A 的用户查询合同。系统命中了旧的检索缓存，返回了权限变更前的文档片段。模型生成了看似正确的条款摘要。与此同时，另一个请求的政策查询超时，模型把“未确认”写成“已确认”。

事故没有表现为模型输出乱码，反而表现为流畅、格式正确和低延迟。监控只看错误率和 TTFT，因此没有立即报警。

把合成事件按时间排开，可以看到“低延迟”和“请求成功”掩盖了两种不同错误：

| 相对时间 | 请求/事件 | 当时应保留的证据 | 读数 |
| --- | --- | --- | --- |
| T+0 秒 | 权限服务把租户 A 的权限版本从 `acl-16` 更新为 `acl-17` | 生效时间、主体、资源与新旧版本 | 旧缓存是否仍可见取决于失效和二次授权 |
| T+1 秒 | 请求 `req-731` 命中以租户和文档版本为键、但未带权限版本的缓存 | cache key、命中状态、文档 revision、当前 ACL revision | `acl-17` 请求读到旧授权上下文 |
| T+1.5 秒 | 响应附上合同第 7 版条款摘要 | claim、evidence ID、页码、permission snapshot、TTFT | 约 500 ms 的 TTFT 看起来正常，但内容已无权返回 |
| T+40 秒 | 请求 `req-744` 发出只读状态查询 `op-52`，1.5 秒后客户端超时 | operation ID、工具状态、下游回执和重试预算 | 查询结果进入 UNKNOWN_RESULT；超时只说明回执缺失，不能把未知状态写成已确认 |
| T+42 秒 | 模型将旧上下文中的“待确认”改写为“已确认” | 生成版本、引用、工具状态与终态 | 文本流畅不等于状态已经核实 |

表内 ID、版本和时间都是为教学构造的输入，不代表真实事故日志。

### 10.16.3 根因分层

- 数据层：缓存对象缺少权限版本；
- 编排层：缓存命中绕过了重新授权；
- 协议层：只读查询的 UNKNOWN_RESULT 和写操作的 UNKNOWN_EXTERNAL_EFFECT 没有被区分；
- 模型层：缺少证据不足时的诚实表达；
- 评估层：没有跨租户和超时状态样本；
- 观测层：没有记录证据 revision 和工具最终状态。

如果只做一次安全微调，可能改变模型的拒答语气，却不能修复缓存授权和执行状态。

### 10.16.4 修复顺序

第一步立即隔离受影响缓存和高风险查询，保留审计证据；第二步让缓存键绑定租户、权限版本、文档 revision 和查询规范化；第三步所有缓存命中后重新执行当前权限检查；第四步把只读查询超时记为 UNKNOWN_RESULT，禁止模型声称结果已确认；把有副作用的写操作超时记为 UNKNOWN_EXTERNAL_EFFECT，并要求查询业务凭证或人工核对；第五步新增证据支持、跨租户、权限撤销、缓存失效和工具状态回归；第六步灰度发布，观察真实申诉和任务成功。

### 10.16.5 修复后的结论

修复后可以说：

> 在已覆盖的租户、权限版本、文档 revision、缓存路径和工具超时场景中，事故样本没有再次复现；缓存命中不再绕过授权，未知工具状态不再被写成确认。

不能说：

> 系统已经完全安全，模型不会再产生错误。

前一种说法有对象、条件和证据范围；后一种说法把有限测试写成了普遍保证。

## 10.17 一个可执行的小例子：状态、容量和单位成本

前面的概念如果只停留在架构图上，容易变成漂亮但无法检查的术语。下面用一段很小的 Python 程序，把三个经常被混淆的问题放在一起：

1. 工具超时后，系统应该区分 UNKNOWN_RESULT 和 UNKNOWN_EXTERNAL_EFFECT，而不是擅自改成 FAILED；
2. token 速率要根据输入和输出的实际工作量估算；
3. 成本应该除以成功任务数，而不是简单除以请求数。

### 10.17.1 保留工具超时的未知状态

本代码只示范已发出写操作后进入 `UNKNOWN_EXTERNAL_EFFECT` 的情况；只读查询的 `UNKNOWN_RESULT` 与有界重试规则在 §10.9.4 说明，不由下面这个短函数实现。

已验证（2026-11-06；Python 3.12.9；仅标准库）。

~~~python
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ToolState(Enum):
    NOT_STARTED = "not_started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN_EXTERNAL_EFFECT = "unknown_external_effect"


@dataclass(frozen=True)
class ToolResult:
    state: ToolState
    idempotent: bool


def can_retry(result: ToolResult) -> bool:
    # FAILED 必须表示已确认未完成；UNKNOWN_EXTERNAL_EFFECT 不能据此重发写操作。
    return result.state is ToolState.FAILED and result.idempotent


def cost_per_success(total_cost: float, successful_tasks: int) -> float | None:
    if not isfinite(total_cost) or total_cost < 0:
        raise ValueError("total_cost must be finite and non-negative")
    if (
        not isinstance(successful_tasks, int)
        or isinstance(successful_tasks, bool)
        or successful_tasks < 0
    ):
        raise ValueError("successful_tasks must be a non-negative integer")
    return total_cost / successful_tasks if successful_tasks else None


tool = ToolResult(state=ToolState.UNKNOWN_EXTERNAL_EFFECT, idempotent=False)
request_rate = 10
input_tokens = 2_000
output_tokens = 400
total_cost = 8.0
successful_tasks = 10

if any(
    not isfinite(value) or value < 0
    for value in (request_rate, input_tokens, output_tokens)
):
    raise ValueError("workload values must be finite and non-negative")
input_token_rate = request_rate * input_tokens
output_token_rate = request_rate * output_tokens
token_rate = input_token_rate + output_token_rate
unit_cost = cost_per_success(total_cost, successful_tasks)

print({
    "final_status": tool.state.value,
    "retry_allowed_after_unknown_effect": can_retry(tool),
    "input_token_rate": float(input_token_rate),
    "output_token_rate": float(output_token_rate),
    "total_token_rate": float(token_rate),
    "cost_per_success": round(unit_cost, 3) if unit_cost is not None else None,
})
~~~

输出为：

~~~text
{'final_status': 'unknown_external_effect', 'retry_allowed_after_unknown_effect': False, 'input_token_rate': 20000.0, 'output_token_rate': 4000.0, 'total_token_rate': 24000.0, 'cost_per_success': 0.8}
~~~

这里的重点不是 Python 语法，而是三个字段之间的关系。状态字段描述事实，idempotent 描述动作属性，can_retry 才是一个经过约束的决策。不能因为用户看到了超时，就把状态字段直接改成“失败”；也不能因为某个工具标记为幂等，就跳过当前权限检查。若 `successful_tasks` 为零，`unit_cost` 返回 `None`，表示单位成功任务成本未定义；系统不能把没有完成任务伪装成零成本。

### 10.17.2 从 token 速率到 worker 数量

对同一组请求，24,000 只是输入与输出 token 的合计需求速率，不是所需 GPU 数量。沿用 10.13.4 的教学工作负载和实测假设，分离的 prefill/decode 池分别需要 20,000 input token/s 与 4,000 output token/s。按每个 prefill worker 10,000 input token/s、每个 decode worker 1,500 output token/s，理想化估算分别为：

~~~math
G_{\mathrm{prefill}}
=
\left\lceil
\frac{20\,000}{10\,000}
\right\rceil
=
2
~~~

~~~math
G_{\mathrm{decode}}
=
\left\lceil
\frac{4\,000}{1\,500}
\right\rceil
=
3
~~~

这两个结果只覆盖平均负载。按 1.5 倍峰值并容忍每个池一个 worker 故障，粗略需要 4 个 prefill worker 和 5 个 decode worker。实际部署还要回答至少五个问题：

可以把冗余数复算出来：prefill 峰值为 `1.5×20,000=30,000 input token/s`，3 个 worker 正好能处理这个负载，但其中 1 个故障后只剩 20,000，因此要配 4 个；decode 峰值为 `1.5×4,000=6,000 output token/s`，4 个 worker 才能在故障前达到它，故障后只剩 4,500，因此需要配 5 个。`+1` 是为单个 worker 故障保留的容量，不是额外吞吐系数；假设 worker 同质、吞吐稳定，真实部署仍需测排队和尾延迟。

- 输入和输出是否使用了同一种吞吐定义；
- 输入池的 10,000 token/s 与输出池的 1,500 token/s 分别绑定什么长度分布、batch 和量化配置；
- 高峰流量与平均流量的比例是多少；
- 一个 worker 故障时，剩余 worker 是否仍满足最低服务目标；
- 长上下文请求的 KV Cache 是否会先于算力成为瓶颈。

因此，总量是 9 个分池 worker，而不是一个脱离阶段定义的“等效 token”数字。若使用共享 worker，必须在混合请求上测联合吞吐；分离池的两组吞吐不能直接相加。这个估算仍不是生产容量结论，因为它没有描述排队分布、请求长度长尾、显存碎片、网络带宽和发布期间的容量损失；它只是把“需要一些冗余”变成可以继续测量的假设。

### 10.17.3 为什么要按成功任务计价

设一天收到 100 个请求，其中 80 个完成了用户真正需要的任务，每个请求的平均直接成本为 0.05 元，重试和人工返工另外产生 3 元成本，则：

~~~math
C_{\mathrm{request}}
=
\frac{100\times0.05+3}{100}
=
0.08
~~~

但单位成功任务成本是：

~~~math
C_{\mathrm{success}}
=
\frac{100\times0.05+3}{80}
=
0.10
~~~

如果只汇报 0.08 元，读者会误以为系统比实际更便宜。更严重的是，团队可能通过让系统更早返回一个格式正确但不可用的答案来降低表面成本。成功的定义必须和业务目标绑定，例如“带有正确权限和证据的合同字段被用户接受”，而不是“模型返回了 HTTP 200”。

代码块中的另一组教学输入是 8 元总成本和 10 个成功任务，所以算得 `8/10=0.8` 元/成功任务；这不是 `0.8` 元/请求。若成功数为 0，分母不存在，函数返回 `None` 而不是零。平台比较 A/B 时，必须固定任务成功定义、计费窗口和人工复核成本，否则即使除法正确，分子和分母也不是同一业务口径。

若一个用户目标触发三次模型请求和两次只读工具查询，它在任务成本账本中仍应按一个逻辑任务计数，所有重试和工具成本都进入该任务的分子。对于部分完成的流程，还要预先约定哪些状态算成功、哪些进入待人工队列、哪些失败后仍有可复用产物；不能在看到结果后临时挑一个有利分母。A/B 两组应按稳定 task_id 去重，并使用相同的成功窗口和人工复核规则，否则重复请求可能被误当成多份成功产出。

## 10.18 资料和结论的边界：数字从哪里来，能说明什么

大模型系统设计经常同时引用论文、厂商文档、开源项目、benchmark 和线上监控。它们都可以有价值，但回答的问题不同。把不同来源混成同一类“事实”，会让设计看起来有依据，实际却无法复现。

### 10.18.1 五种常见证据

第一类是原始研究论文。论文适合说明算法的目标、假设、训练设置和在特定实验上的结果。例如某种注意力变体为什么减少计算，应该先读提出它的论文和补充材料；论文中的结果不能自动推导出任何硬件、任何上下文长度下的生产收益。

第二类是官方产品或模型文档。它适合说明 API 字段、上下文上限、支持的输入类型、版本行为和使用限制。官方文档是产品契约的重要来源，但“支持某个上限”不等于在所有任务上都具有相同的有效能力，也不等于成本、延迟和检索质量已经满足目标。

产品入口自身也有生命周期。以 OpenAI 官方 Evals 指南为例，截至 2026-11-06，现有 Evals 平台已于 2026-10-31 转为只读，官方计划于 2026-11-30 关闭，并将新用户引向 Datasets。这个带日期的例子说明系统设计不能把某个产品控制台当成永久接口；它不表示通用评估平台架构已经过时。[OpenAI：Working with evals](https://developers.openai.com/api/docs/guides/evals)

第三类是开源项目的源码、配置和发布说明。它们适合确认默认值、调度逻辑、缓存布局和已知限制。项目首页的一句宣传语不能替代具体版本的源码和可运行配置；同一个项目在不同 commit、后端和硬件上的行为可能不同。

第四类是 benchmark 论文、数据集说明和原始结果。它们可以帮助比较任务上的相对表现，但必须检查数据污染、提示模板、评测器、采样次数和统计不确定性。一个总分不能替代对目标业务样本的端到端评估。

第五类是自己的实测和线上数据。它最接近当前系统，却只对测量的硬件、版本、输入分布和时间窗口负责。一次成功的 demo 说明路径可行，不说明长尾、故障恢复和权限隔离已经成立。

### 10.18.2 给每个数字附上测量条件

一个可复核的性能数字至少应带上：

~~~text
claim
    -> metric definition and denominator
    -> model/tokenizer/template revision
    -> hardware and runtime
    -> quantization and decoding settings
    -> input/output length distribution
    -> batch and concurrency
    -> warm-up and measurement window
    -> percentile or confidence interval
    -> known exclusions
~~~

例如，10.13.4 中的 `10,000 input token/s` 与 `1,500 output token/s` 是构造的教学假设测量值，并且只绑定示例中的硬件、runtime、量化、长度分布和并发。需要知道它们是单请求还是连续 batching，是平均吞吐还是满足 P99 延迟目标时的吞吐，是否包括检索、审核、网络和失败重试。没有这些条件，数字只能作为线索，不能作为容量承诺。

### 10.18.3 把事实、估算和假设分开

书写系统设计结论时，可以把句子分成三种：

- 事实：某个版本的接口明确返回哪些字段，或某次测量在给定条件下得到什么结果；
- 估算：根据工作量和已知吞吐推导出的数量级；
- 假设：尚未验证、但为了继续设计而暂时采用的条件。

例如，“10 个请求每秒需要 24,000 token/s”是给定请求分布下的算术推导；“需要 6 个 worker”还包含峰值、故障和实测吞吐假设；“6 个 worker 可以满足 P99 延迟目标”则必须通过带有真实长度分布的压测验证。

写清楚这三者的区别，不会削弱结论，反而会告诉读者下一步应该测什么。没有证据时，使用“在这些条件下可推导”“需要进一步验证”比伪装成确定事实更专业。

### 10.18.4 一个容易误读的例子：长上下文

“模型或 API 支持很长上下文”至少可以拆成四个不同问题：

1. 接口能否接受这么多 token；
2. tokenizer、runtime 和显存能否在目标配置下处理；
3. 模型能否在长输入中找到并使用相关证据；
4. 任务成功率、延迟和单位成功成本是否仍然可接受。

前两个问题通常可以从文档和运行日志获得证据，第三个需要针对此长度和任务设计检索、跨段推理与干扰测试，第四个需要端到端测量。把四个问题压缩成一个“支持”标签，正是系统设计中最常见的证据越界。

## 10.19 章末练习：从架构名词走到可检查设计

下面的练习都以衡川企业助手为背景。重点不是画出唯一正确的架构，而是让每个决定都能对应到状态、约束、证据和失败处理。

### 10.19.1 练习一：划分边界

为“查询合同付款比例并给出页码引用”画出控制平面、数据平面、证据平面和治理平面。对每一条跨平面数据流写出至少一个不变量。

检查方向：

- 用户身份和租户上下文在检索前已经确定；
- 检索结果带有文档版本、权限快照和页码；
- 模型只能提出回答，不能修改权限；
- 最终回答的 claim 能追溯到 evidence_id；
- 旧版本索引或缓存失效后不能继续提供可见结果。

### 10.19.2 练习二：计算工作量

某高峰时段有三类请求：

表中输入和输出 token 数表示每个外部请求触发的全部模型调用累计值；平均模型调用次数单独用于估算内部调用率，不再乘入 token 量。

| 类型 | 占比 | 输入 token | 输出 token | 平均模型调用次数 |
| --- | ---: | ---: | ---: | ---: |
| 简单制度查询 | 0.5 | 1,000 | 200 | 1 |
| 合同问答 | 0.35 | 8,000 | 500 | 2 |
| 票据抽取 | 0.15 | 4,000 | 300 | 3 |

外部请求率为每秒 20 个。计算平均 token 需求、平均内部模型调用率，并说明为什么不能只用外部 QPS 规划 GPU。

参考计算：

~~~math
\bar{T}
=
0.5(1000+200)
+
0.35(8000+500)
+
0.15(4000+300)
=
4\,220
~~~

~~~math
D_{\mathrm{token}}
=
20\times4\,220
=
84\,400\ \mathrm{token/s}
~~~

~~~math
\bar{a}
=
0.5(1)+0.35(2)+0.15(3)
=
1.65
~~~

因此平均内部模型调用率约为 33 次/秒，且 token 负载还没有计入重试、judge、摘要和异常长请求。

### 10.19.3 练习三：处理工具超时

设计一个查询“报销单状态”的只读工具和一个“提交报销单”的写工具。分别说明网络超时、客户端断开、服务端返回 500 和下游没有返回凭证时，状态机应该保存什么状态，是否允许自动重试。

一个合理的分析应区分：

- 只读查询通常可以根据 request_id 重试，但仍要避免返回旧权限下的结果；
- 写工具收到超时不能直接认为失败，应进入 UNKNOWN_EXTERNAL_EFFECT 并查询业务状态；
- 没有下游凭证时不能声称 SUCCEEDED；
- 客户端断开不等于服务端取消，后台任务要有独立的 task_id 和取消记录。

### 10.19.4 练习四：设计缓存键

为合同检索结果和模型前缀缓存分别设计缓存键。除了查询文本，还要决定是否纳入租户、用户权限版本、文档 revision、模型 revision、模板 revision、策略 revision 和过期时间。解释其中一个字段发生变化时，为什么旧值不能继续复用。

检查方向是：缓存复用的是计算结果，不是授权结果。当前权限检查必须在命中后仍然执行；包含私有证据的 KV 或响应不能仅凭文本前缀相同而跨用户共享。

### 10.19.5 练习五：把“回答正确”拆成指标

为合同金额问答建立一组端到端指标，至少包含：

- 数值正确率；
- 引用覆盖率；
- 引用与 claim 的对应正确率；
- 权限违规率；
- 证据不足时的诚实标记率；
- P95 延迟；
- 单位成功任务成本；
- 人工复核率和误拒率。

为每个指标写出分母、样本来源和版本字段。然后构造一个变体：它的数值正确率提高，但引用覆盖率下降。说明为什么不能只依据总正确率发布它。

### 10.19.6 练习六：复盘权限事故

在 10.16 的事故中，假设日志只有 request_id、模型名和最终文本，没有权限版本、索引 revision、工具状态和 evidence_id。列出无法回答的五个问题，再设计一份最小 trace 字段集合。

至少应能回答：

- 当时用户属于哪个租户、采用哪一版权限；
- 结果来自哪个文档和索引版本；
- 缓存是否命中，命中前后是否重新授权；
- 工具是否真正执行、有没有下游凭证；
- 哪个策略、模板和模型版本生成了最终结果。

### 10.19.7 练习七：写一份有边界的容量结论

根据 10.17 的教学数据，写出一段容量结论，要求明确平均负载、峰值倍率、worker 实测条件、故障冗余和未覆盖风险。不要写“系统可以承载所有请求”，而要说明结论在哪些输入长度、并发和版本范围内成立。

这项练习训练的是工程表达：容量数字只有和测量条件、目标 SLO、失败处理以及剩余不确定性一起出现，才有决策价值。

## 10.20 结语：系统设计是让不确定性停在边界内

大模型系统的复杂性不只来自模型规模，还来自概率输出与确定性业务之间的错位。一个流畅答案可能没有证据，一个成功的 HTTP 响应可能没有完成任务，一个低延迟缓存命中可能绕过了新的权限，一个工具超时也可能隐藏着已经发生的外部副作用。

因此，可靠设计需要几条相互连接的链：

~~~text
需求
    -> 服务契约和不可接受失败
    -> 工作量、状态和权限模型
    -> 受约束的检索、推理和工具执行
    -> 证据、版本和可观测性
    -> 分层评估、灰度、降级和恢复
    -> 有边界的结论与持续修正
~~~

模型能力会变化，硬件和 runtime 会变化，文档和权限也会变化。真正可持续的系统不是假设这些变化不会发生，而是让变化有版本，让失败有状态，让输出有证据，让动作有授权，让事故能够复盘，让结论只覆盖已经验证的范围。
