# 第十二章：Agent 综合系统设计与复盘

前十一章分别讨论了 Agent 的状态循环、工具调用、ReAct、规划、memory、Agentic RAG、Code Agent、Browser/Computer Use、Multi-Agent、评估和安全。本章不再把这些概念拆成互相孤立的问答，而是把它们放回一个完整系统中：当用户要求合同助手“检查一份合同，核对政策，生成风险摘要，并创建待审核工单”时，系统如何从目标走到可验证结果？哪些步骤必须由 workflow 固定，哪些步骤可以交给模型，哪些动作应当停止等待人？

综合设计的难点不在于记住更多名词，而在于保持边界一致。`tool` 在前一章是一个有 schema 的接口，在这里还必须有权限、数据范围、失败状态和审计；`memory` 不只是“记住上下文”，还涉及来源、过期、删除和跨租户隔离；`evaluation` 不只是成功率，还要检查 trace、证据、成本和安全副作用。每个局部设计都要能接回任务契约。

## 0. 本章范围与资料

本章参考 OpenAI Agents SDK 的 tools、guardrails、tracing 公开文档，OpenAI Evals、OpenAI Model Spec，以及 SWE-bench、WebArena、OSWorld、AgentBench、GAIA 和 τ-bench 等公开 Agent 评估资料。它们分别提供 runtime 接口、指令与安全边界、任务定义和环境评估入口；没有任何一个来源可以替代对目标系统的实测。

本章会复用前十一章的机制，但每一节都给出新的系统连接、边界案例或判断依据。安全相关内容只讨论防御性设计，不提供可复用注入提示、绕过权限、规避审计、高风险自动操作或破坏系统的方法。章节文件名保留历史目录兼容性，正文已经改为综合系统教材。

## 12.0 从任务契约开始的综合方法

假设用户说：“请检查供应商合同是否符合当前采购政策；如果有例外，创建一条待审核工单，并告诉我依据。”这句话至少包含四个不同的目标：理解合同、检索政策、判断条件和创建受限业务对象。若直接让一个 Agent 自由调用所有工具，任何一层的误判都可能变成越权读取、错误判断或错误写入。

先把任务写成契约：

~~~text
目标：给出有引用的政策判断，并在确有例外时创建 pending_review 工单
可读范围：当前用户所属租户的合同与政策版本
可写范围：只能创建待审核工单，不能批准、发信或修改权限
证据要求：合同条款定位、政策版本、逐项规则判断
停止条件：权限不足、证据冲突、状态未知或需要批准时暂停
完成条件：工单对象已落库，状态和版本可重新读取
~~~

然后按数据流和控制流走一遍：

1. router 判断这是合同/政策任务，而不是简单聊天。
2. planner 把检索、证据对齐、规则判断和工单创建拆成有依赖的子任务。
3. tool registry 只向当前任务暴露允许的读取和创建工具。
4. controller 维护状态、预算、停止条件和人工升级。
5. evaluator 同时检查证据、业务状态、trace 和风险事件。

这个结构不是固定产品模板。简单任务可以由固定 workflow 完成；多跳研究可以局部使用 Agentic RAG；高风险写入必须由独立执行器复核。系统的价值在于每个选择都有因果理由，而不是组件越多越先进。

### 12.0.1 评估一个综合设计的维度

可以把一个系统设计样本表示为：

~~~math
d_i=(G_i,A_i,T_i,P_i,E_i,S_i,C_i,F_i)
~~~

其中 `G_i` 是任务和完成条件，`A_i` 是 Agent 状态与动作，`T_i` 是工具和数据接口，`P_i` 是权限与数据策略，`E_i` 是验收器，`S_i` 是安全控制，`C_i` 是成本/延迟预算，`F_i` 是失败与恢复路径。这个表示式的作用是提醒设计者不能只描述模型和 prompt；少一个维度，评估就可能失去对象。

对某一任务族，可以定义覆盖向量：

~~~math
\mathbf{v}=(v_{\mathrm{goal}},v_{\mathrm{state}},v_{\mathrm{tool}},v_{\mathrm{evidence}},v_{\mathrm{eval}},v_{\mathrm{safety}},v_{\mathrm{recovery}})
~~~

每个分量不必是一个主观分数，也可以是“是否有可观察契约、是否有测试样本、是否有验收证据”。把设计写成向量比给它一个“准备度”总分更有用，因为安全缺失、证据缺失和成本缺失不能相互抵消。后面的综合案例和 demo 都采用这个思路。

## 12.1 什么是 Agent

Agent 不是一个模型名称，而是一类目标驱动的执行系统。它接收任务，维护与任务相关的状态，选择动作或工具，读取环境反馈，再决定继续、修正、停止或请求人。一个最小闭环可以写成：

~~~math
s_{t+1}=T(s_t,a_t,o_t),\qquad a_t\sim\pi(G,s_t,T,P,B)
~~~

`G` 是目标，`s_t` 是当前状态，`a_t` 是动作，`o_t` 是工具或环境反馈，`T` 是状态更新，`P` 是权限，`B` 是预算。`\pi` 可以由 LLM、规则、workflow 或它们的组合实现。这个公式不意味着模型真的拥有一个显式概率策略，而是用来强调：下一步动作依赖状态、反馈和约束，不是一次性从问题生成答案。

如果系统只把用户问题送进模型并返回文本，它更接近 Chatbot；如果系统能读取外部状态、执行受约束动作并验证结果，才有 Agent 的执行闭环。判断边界时不要只看产品名称，要看是否存在状态转移、工具副作用和反馈驱动的下一步。

### 12.1.1 一个最小轨迹

用户要求查询订单并创建退货申请时，一条最小轨迹可能是：读取用户身份，查询订单，检查退货条件，生成申请草稿，请求确认，创建申请，重新读取申请状态，最后汇报结果。每一步都有不同责任：身份和订单是权限问题，条件判断是证据问题，创建申请是副作用问题，最终汇报是忠实性问题。

如果跳过重新读取状态，系统只能证明“发送过创建请求”；如果把政策文本中的一条指令当成用户授权，系统会把证据和权限混在一起；如果确认后参数被修改，确认也失去了意义。这个轨迹说明 Agent 的定义必须与可审计的状态转换一起理解。

## 12.2 Agent 和普通 Chatbot 有什么区别

Chatbot 的主要交付物是文本；Agent 的交付物是外部任务状态加上对状态的说明。一个客服 chatbot 可以回答“退货政策是什么”，一个客服 Agent 可能还要查询订单、判断是否过期、创建退货申请并验证工单状态。后者多了身份、工具、业务前置条件和副作用，因此必须增加权限、审计和恢复设计。

两者并非互斥。Agent 内部仍然需要模型生成解释，Chatbot 也可能调用一次检索工具。更有用的区分标准是：是否有目标约束、是否有多步状态转移、是否能触发外部变化，以及结果是否由外部验收器确认。只要引入了外部副作用，系统就不能沿用纯文本问答的安全假设。

| 维度 | Chatbot | Agent |
|---|---|---|
| 主要交付物 | 文本回复 | 外部状态变化加说明 |
| 状态 | 常以会话上下文为主 | 任务状态、资源版本、工具结果和记忆 |
| 错误后果 | 误导用户 | 可能产生文件、数据库或外部业务副作用 |
| 主要验收 | 文本质量、事实和风格 | 外部状态、过程、安全、成本和陈述一致性 |

这个表不是产品分类标准，而是工程责任表。一个只读研究助手可能仍然叫 Agent，但它的安全重点和能发送邮件的 workflow Agent 不同；风险应由实际能力和副作用决定。

## 12.3 Agent 和 Workflow 如何取舍

Workflow 把状态、分支和权限预先写明，优点是可预测、可测试、容易估算成本；Agent 把部分决策留给运行时，适合输入不完整、工具反馈决定下一步或任务路径难以穷举的场景。二者不是“旧技术”和“新技术”的替代关系。

可以用三个问题决定边界：流程是否稳定？每个分支是否能写出清楚的验收条件？错误动作是否有较大副作用？如果流程稳定且风险高，优先 workflow；如果探索和判断占主要工作，Agent 可以负责局部；如果任务同时包含固定审批链和开放检索，采用混合架构。合同助手中，政策检索和条款对齐可以由 Agent 处理，但工单状态转移、权限和外部通知仍应由 workflow 与执行器控制。

混合架构的成本是接口复杂度。Agent 输出必须符合 workflow 输入 schema，workflow 也要把拒绝、缺证据和状态未知传回 Agent；如果只在边界处传一段自然语言，错误会在两个控制器之间传播。

一个实用的取舍顺序是：先用确定性规则和 API 覆盖稳定路径，再把开放判断放进 Agent；先让 Agent 产出草稿和证据，再由 workflow 执行高风险写入；最后用配对任务比较“全 workflow”“全 Agent”和“混合方案”的成功率、成本、延迟和风险。架构选择应由测量结果推动，而不是由“自主性”这个标签推动。

## 12.4 如何设计工具调用系统

工具调用系统至少包含 registry、policy engine、executor、observation parser 和 trace logger。registry 描述工具能做什么；policy engine 判断当前主体、资源和风险是否允许；executor 才真正产生副作用；parser 将返回值转换成结构化状态；logger 保存请求、结果、版本和阻断理由。

参数校验要分三层：schema 校验检查类型和必填字段，业务校验检查资源归属、状态和范围，安全校验检查权限、数据出域和风险。JSON 合法不能证明“给正确租户更新正确对象”。重试也要由工具的幂等语义决定，超时写入必须先查询状态，不能把失败字符串直接交给模型自由解释。

合同案例中，Agent 可以选择 `search_policy`、`read_contract` 和 `create_pending_ticket`，但不能看到 `approve_ticket`。即使模型生成了批准调用，executor 仍应在 schema、权限和资源层拒绝，并把结构化拒绝原因返回给状态机。

工具契约还应声明返回结果的语义：`success` 是请求被服务端接受，还是目标对象已持久化？`timeout` 是否可能已经产生副作用？`not_found` 是对象不存在，还是当前身份不可见？如果这些语义没有写清楚，Agent 无法正确恢复，评估器也无法区分工具故障与模型故障。

## 12.5 Function Calling 解决什么问题

Function Calling 解决的是模型与程序之间的接口表达：模型生成工具名和结构化参数，运行时解析并调用函数，再把结果作为 observation 返回。它降低了自然语言解析的不确定性，便于 schema 校验、日志记录和离线评估。

它没有解决三类问题：参数是否符合业务语义，调用者是否有权访问资源，工具结果是否可信。`{"amount": 100}` 可以通过 JSON schema，却可能违反金额上限；一个合法的 `send_email` 调用仍然可能跨越数据边界；工具返回的文本仍然可能包含低信任内容。安全控制必须在执行器和数据流层独立存在。

因此，Function Calling 的正确位置是“模型提议动作与程序执行动作之间的协议层”，不是权限层，也不是事实验证层。结构化接口越清楚，越容易发现它没有覆盖的责任。

## 12.6 ReAct 是什么

ReAct 把推理与行动放进交替循环：模型根据当前任务状态提出动作，工具返回 observation，系统再根据反馈更新状态。它的价值不在于某个固定提示词，而在于让行动依赖真实环境，而不是让模型在没有反馈的情况下连续猜测。

它适合搜索、代码调试和浏览器操作，但会带来循环不收敛、重复动作、成本增长和高风险动作漂移。运行时应设置最大步骤、token/工具预算、无进展检测、权限检查和停止函数。遇到权限不足、证据冲突或外部状态未知时，正确的下一步可能是停止或请求人，而不是继续“思考”。

### 12.6.1 读取一条 ReAct 轨迹

考虑“找出合同中不符合政策的条款”这一任务：

~~~text
state=contract_loaded
action=search_policy(query="payment exception")
observation=policy_v3, clause_4, permission=allowed
state=evidence_candidate
action=read_policy_clause(clause_4)
observation=exception_requires_approval
state=needs_review
action=create_pending_ticket(...)
observation=created(ticket_id=T-17, status=pending_review)
state=completed
~~~

真正推动下一步的是 observation 和状态更新。若 `read_policy_clause` 返回权限不足，正确轨迹应转为 `evidence_incomplete`；若创建工单返回超时，状态应为 `unknown`，而不是继续创建第二张工单。评估器可以检查每次动作是否与前一状态和权限相符，而不是只看最终 ticket 是否存在。

## 12.7 Plan-Act-Observe 如何落地

Plan-Act-Observe 将循环拆成计划、动作和观察三个责任不同的阶段。计划不是承诺书，而是带版本、依赖和验收条件的假设；每个动作执行后，observation 可能使计划失效。系统应保存计划版本、已完成子目标、阻塞原因、下一步候选和预算消耗。

例如合同任务原计划是“读取合同—检索政策—创建工单”，但读取后发现用户没有访问附录的权限，系统不应继续假设证据完整；它可以转为“报告缺失证据并请求授权”，或在允许范围内生成不带最终结论的草稿。计划调整必须说明哪个 observation 导致了变化。

计划还要有版本和依赖。`plan_v1` 如果已经假设附录可读，发现权限不足后不能只把下一步文字改成“继续”；应标记受影响的结论、撤销依赖它的候选动作，并重新计算剩余预算。这样可以防止早期错误假设在长任务中继续传播。

## 12.8 Agent 如何做任务分解

任务分解应从验收条件反推子任务，而不是从名词列表开始。每个子任务需要输入、输出、依赖、资源范围、失败状态和验收器；“研究政策”太宽，“返回指定版本中与例外条件相关的条款定位”才是可验证子目标。

依赖可以表示为 DAG。条款检索和合同解析可能并行，规则判断依赖二者，创建工单又依赖规则判断和权限检查。并行不是免费的：它增加合并、冲突和消息成本；如果两个子任务写同一对象，还必须串行或使用版本控制。失败时优先区分可修复、可降级、需补充信息和不可安全继续四类，而不是统一重试。

一个简单的分解质量检查是让每个子任务回答四个问题：完成后哪个状态字段会改变？由什么外部证据证明？失败后能否重试或回退？它需要哪些权限？如果答不出来，子任务仍然是一个模糊目标，不应直接交给执行器。

## 12.9 一次性规划和动态规划如何取舍

一次性规划把完整路线提前展开，适合状态稳定、步骤短、需要预览的任务；动态规划在每一步根据 observation 重新选择，适合调试、浏览器和开放检索。关键取舍是计划成本与环境不确定性：状态越容易变化，越不应把远期细节当作已确定事实。

一个常见的混合方式是先生成粗粒度目标和停止条件，再只展开当前可验证的一小段。这样既有全局方向，又能在权限、页面或证据变化时回退。评估时要记录计划漂移、重规划次数、无进展步骤和因错误早期计划造成的副作用，而不是只看最后是否完成。

可以把计划成本写成：

~~~math
C_{\mathrm{plan}}=C_{\mathrm{initial}}+n_rC_{\mathrm{replan}}+n_cC_{\mathrm{coord}}
~~~

`n_r` 是重规划次数，`n_c` 是协调和状态合并次数。这个式子不是产品报价，而是提醒我们：动态规划的灵活性有额外成本；如果每次 observation 都触发完整计划重写，系统可能把 token 花在重复描述上，而没有获得新的证据。

## 12.10 Agent Memory 如何设计

Memory 设计要先区分短期状态和长期记忆。短期状态保存当前目标、计划版本、工具结果、错误和待办；长期记忆保存经过验证的用户偏好、项目事实或可复用经验。二者的写入门槛、过期时间、权限和删除语义不同，不能因为都放在向量库里就当成同一类数据。

长期记忆至少带来源、时间、主体、范围、置信度、版本和删除状态。一次网页读取结果可以作为待验证证据，却不能自动升级为长期规则；影响外部动作的记忆还要在使用时重新验证。删除请求要传播到主存储、索引、缓存、摘要和备份，并先用墓碑阻止继续召回。

### 12.10.1 Memory 使用的复核点

每次从长期 memory 读出一条内容时，可以依次检查：主体是否匹配、权限是否匹配、是否过期、来源是否足以支持当前 claim、是否与实时外部状态冲突、是否会改变高风险动作。memory 提供的是候选上下文，不是授权凭证；如果它与当前 API 状态冲突，应优先使用受权的实时状态并记录冲突。

## 12.11 Memory 和 RAG 的区别

RAG 的默认问题是“从外部知识源找到与当前问题相关、可引用的证据”；memory 的默认问题是“哪些与主体和任务有关的状态可以跨轮保留”。它们可以共享 embedding、索引和 reranker，但责任不同：RAG 更关心文档版本、召回和引用，memory 更关心写入授权、主体隔离、过期、修改和删除。

合同助手可以从政策库检索最新条款，却不能把一条政策全文写入用户 memory；用户的“报告偏好”可以长期保存，却不能成为绕过政策权限的规则。设计时应为两个数据流使用不同 namespace、schema、生命周期和验收器。

## 12.12 Agentic RAG 和普通 RAG 的区别

普通 RAG 通常是一次查询、一次召回和一次生成；Agentic RAG 把检索放进受控循环，让系统判断是否缺证据、怎样拆分查询、是否需要第二来源、冲突如何处理和何时停止。它能处理多跳合同或研究问题，但每增加一轮就增加 token、延迟、检索噪声和提示注入暴露面。

因此 Agentic RAG 不应被理解成“让模型随便搜”。它需要查询预算、证据状态、来源权限、claim/证据映射、停止条件和失败回退。若问题简单且知识源稳定，一次 RAG 或固定 workflow 可能更可靠；动态循环只有在它解决了具体的证据缺口时才值得。

## 12.13 如何设计可靠的 Agentic RAG

可靠的 Agentic RAG 需要一个证据状态机：`unknown -> candidate -> verified -> conflicting -> sufficient/insufficient`。查询控制器根据缺口生成下一条查询，检索器返回带来源、版本和权限的证据包，verifier 检查 claim 是否被支持，controller 决定继续、降级为不确定回答或停止。

评估不能只看答案是否像真的。至少要检查召回到的证据是否有权被当前用户使用、引用定位是否准确、关键 claim 是否都有支持、查询是否带来新的有效证据、循环是否重复，以及检索内容中的低信任指令是否影响了控制平面。一个“引用很多但都不支持结论”的答案应判为失败，而不是因为格式漂亮得分。

### 12.13.1 合同审核中的证据状态机

把状态机放进一个具体例子，才能看出它为什么不是给检索流程换一组漂亮的名字。假设合同助手要判断“付款期限超过 60 天时，是否需要例外审批”。用户给出的合同正文里出现了“付款期限为 90 天”，这句话首先只能形成一个候选证据：它有文本定位，但还没有证明合同版本、适用主体和政策版本都正确。

系统可以为每个 claim 建立一条证据记录：

| 字段 | 示例 | 作用 |
|---|---|---|
| `claim_id` | `payment_term_exception` | 标识待验证的结论 |
| `source_id` | `contract-1842` | 指向原始对象，而不是只保存摘录 |
| `version` | `v7` | 防止把旧合同当成当前合同 |
| `locator` | `page=4, clause=3.2` | 让读者和验证器能够回到原文 |
| `authority` | `contract_reader` | 记录当前主体是否有权读取 |
| `state` | `candidate` | 表示支持强度，而非真假二值 |
| `checked_at` | `2026-08-14T...` | 说明状态检查发生的时间 |

随后，控制器还要检索适用的政策版本。如果政策库返回“标准期限不超过 60 天”，并且合同与政策的主体、地区和生效日期一致，两个来源才可以共同把 claim 推进到 `verified`。如果政策检索返回另一份生效日期重叠但阈值不同的版本，状态应变为 `conflicting`；系统可以展示冲突并请求人工判断，却不能用相似度分数较高的一份偷偷覆盖另一份。

对一个结论来说，证据数量本身没有意义，关键是必要条件是否都满足。可以把一个简单的充分性判定写成：

~~~math
S(c)=A(c)\land V(c)\land L(c)\land P(c)\land K(c)
~~~

其中，`A` 表示来源具有读取授权，`V` 表示版本和生效范围有效，`L` 表示引用定位可复核，`P` 表示证据确实支持该 claim，`K` 表示关键冲突已经解决。这个表达式只适用于规则清楚、条件可以明确检查的任务；它不是把复杂事实判断伪装成一个普适的数学真理。

状态变化还必须约束动作。`candidate` 允许继续检索，`verified` 可以生成“符合/不符合”的草稿，`conflicting` 只能生成冲突说明或进入人工处理，`insufficient` 只能说明缺失了什么。若 `create_pending_review_ticket` 的参数要求已验证的合同版本，那么控制器应该在状态不是 `verified` 时拒绝调用，而不是依靠提示词提醒模型“谨慎一点”。这样，证据状态就从文字描述变成了真正影响执行的业务条件。

## 12.14 Code Agent 和普通代码生成有什么区别

普通代码生成交付一段候选文本；Code Agent 交付一个经过约束的工作区变化。它需要观察仓库结构、读取用户已有改动、定位相关代码、生成最小 patch、运行测试、分析失败并验证 diff。代码“看起来合理”不能替代编译、测试、接口兼容和副作用检查。

可靠的 Code Agent 要把仓库快照、允许修改范围、命令策略、依赖变更、测试输出和最终 diff 绑定到同一 trace。用户尚未提交的改动属于输入状态，不能被模型当成可随意覆盖的临时文件；测试命令也可能读写网络、删除文件或泄露环境变量，必须在沙箱和权限范围内执行。

## 12.15 如何设计可靠的 Code Agent

一个可审计的 Code Agent 流程是：冻结或记录初始 workspace；读取任务相关文件和用户改动；形成 patch 计划；在允许范围内写入；运行目标与回归测试；将失败归因到代码、环境、测试或依赖；最后比较 diff、测试结果和任务契约。

“最小 patch”不是少改几行的审美判断，而是减少变更表面积、回归风险和审查成本。若重构确实是修复所必需，应说明范围和验收；若测试失败来自环境缺依赖，不能把“重试成功”说成代码正确。评估要同时看通过率、无关改动、用户改动触碰、危险命令、依赖变化和单位成功成本。

### 12.15.1 什么才算经过验证的 patch

Code Agent 的交付物不是一段 diff，而是 diff 与它所依赖的工作区事实组成的记录。一个实用的抽象是：

~~~math
\mathrm{ValidatedPatch}=(B,\Delta,T,S,E)
~~~

其中，`B` 是修改前的 workspace 基线，`Δ` 是实际变更，`T` 是测试和静态检查结果，`S` 是允许的作用范围，`E` 是副作用记录。只有在基线可识别、变更可重放、测试结果与这份变更对应、范围没有越界，并且副作用已经被记录或被明确禁止时，patch 才能称为“经过验证”。这不是说所有测试都必须通过：测试环境故障可以使结果变成 `inconclusive`，但不能被包装成通过。

例如，用户要求修复一个日期解析错误。Agent 在 `parser.py` 增加两行代码，测试通过，但同时修改了未列入任务范围的配置文件；此时 `T` 可能是成功的，`S` 却不满足，整体交付仍然需要回退或人工复核。反过来，如果只改了目标文件而测试因缺少数据库服务无法启动，`Δ` 和 `S` 可能合格，但 `T` 的状态只能是“未完成验证”，最终说明必须清楚区分“代码变更已生成”和“行为已被测试确认”。

`B` 也不能被忽略。若工作区原本有用户尚未提交的修改，Agent 需要把它们作为基线的一部分保存指纹，或者在发现无法安全区分时停止写入。直接用格式化工具覆盖整棵目录，随后再说“核心测试通过”，会让审查者无法知道哪些变化由 Agent 引入。

因此，Code Agent 的验收通常至少包括四个相互独立的比较：基线与最终工作区的差异、计划范围与实际路径的差异、测试命令声明的范围与真实副作用的差异、最终 claim 与 trace 中事实的差异。这个分解把“模型觉得修好了”变成了可审查的工程对象，也解释了为什么仓库快照、diff、测试输出和命令审计要绑定在同一条 trace 中。

## 12.16 Browser Agent 和 API Tool Use 如何取舍

有稳定、权限清晰且能表达业务状态的 API 时，优先 API tool：参数和返回结构更明确，状态更容易验证，权限更容易绑定对象。Browser Agent 适合没有 API、必须操作现有网页、或需要跨多个旧系统完成流程的场景，但它要处理 DOM、accessibility tree、视觉布局、弹窗、登录状态和页面变化。

浏览器不是天然更通用的 API。一次点击只能说明 UI 接受了动作，不能说明业务对象已经持久化；页面文字也可能来自低信任内容。浏览器路径需要重新观察、动作前置条件、提交后状态验证、身份隔离和高风险确认，并把页面版本或关键截图/状态摘要写进 trace。

### 12.16.1 四种观察通道怎样分工

“使用浏览器”并不等于只能看截图。一个网页 Agent 可以在结构化 API、DOM、accessibility tree 和视觉画面之间选择观察通道；通道不同，能够可靠回答的问题也不同。

| 通道 | 更擅长观察什么 | 常见动作 | 主要盲点 | 适合的验证 |
|---|---|---|---|---|
| API | 业务对象、字段、状态和错误码 | 以结构化参数创建、更新、查询 | 无法覆盖没有 API 的旧页面流程 | 重新读取对象、比较版本和状态 |
| DOM | 元素层级、属性、表单值和可点击节点 | 定位节点、填写表单、提交事件 | 隐藏元素、异步渲染和视觉遮挡可能使语义失真 | 检查节点状态并回读业务结果 |
| accessibility tree | 面向辅助技术暴露的角色、名称、状态和关系 | 按角色和可访问名称操作 | 画布、视觉线索或实现缺陷可能不在树中 | 检查控件状态和页面反馈 |
| 视觉画面 | 布局、图标、验证码外观、弹窗和坐标关系 | 鼠标移动、点击、拖拽、键盘输入 | 文字识别、坐标漂移、低可解释性和遮挡 | 截图对比并结合结构化状态查询 |

选择通道的原则不是“哪个更像人”，而是哪个能以更低的不确定性表达当前任务。例如，修改订单地址时，API 返回的订单版本和地址字段是主要事实；DOM 可以帮助处理没有公开 API 的表单；accessibility tree 对按钮名称和禁用状态往往比像素更稳定；当页面用画布呈现流程图时，视觉观察可能不可避免，但提交后仍应回到 API 或页面业务状态进行确认。

通道之间也可能互相矛盾：DOM 显示按钮已启用，截图却显示它被遮挡；页面提示“保存成功”，API 查询仍返回旧版本。此时不能用语言模型自行裁决，应把矛盾记录为 observation，依据业务对象的版本和持久化状态决定是否继续。视觉通道提供的是观察证据，不是对点击结果的授权；API 返回“允许”也不代表当前用户可以把结果发送给第三方。

## 12.17 如何保证 Computer Use Agent 安全

Computer Use Agent 的安全边界至少分三层。权限层限制账户、文件、网络和业务对象；环境层隔离浏览器 profile、剪贴板、下载目录、凭据和显示设备；操作层对删除、支付、发送、提交、权限修改等动作执行预览、确认、状态验证和审计。屏幕上的文字、弹窗和网页内容仍然是观察数据，不能修改 policy engine。

还要设计失败状态：页面加载失败、登录过期、验证码、点击后状态未知、窗口焦点错误和外部提交超时都不能直接当成功或失败。系统应停止在可解释状态，避免 Agent 在看不清页面时继续点击，或在未知提交后重复操作。

### 12.17.1 观察—动作—验证与 Code Agent 的对照

Computer Use 和 Code Agent 都是“观察环境后产生动作”，但观察对象不同，错误的传播方式也不同。Code Agent 主要观察文件、符号、测试输出和版本控制状态；Computer Use 还要观察焦点、窗口、坐标、加载状态和视觉反馈。前者可以用文本 diff 精确描述大部分变化，后者的一个坐标点击可能只改变了屏幕，而没有改变业务对象。

| 阶段 | Code Agent | Computer Use Agent | 共同的工程要求 |
|---|---|---|---|
| 观察 | 快照、源码、配置、测试和用户改动 | 页面结构、可访问性树、截图、焦点和登录状态 | 记录时间、来源、版本和权限 |
| 动作 | 写文件、运行检查、调用开发工具 | 点击、输入、滚动、上传或提交 | 先校验目标、参数和作用范围 |
| 验证 | diff、编译、测试、接口和工作区状态 | 页面反馈、业务对象状态、版本和服务端回读 | 不以模型自报或单一 UI 提示作为事实 |
| 未知状态 | 命令超时、测试未启动、文件是否写入不明 | 提交超时、焦点错误、点击结果不明 | 查询幂等键或快照，避免盲目重复动作 |

比如 Code Agent 运行迁移测试超时，系统可以检查进程、数据库事务和工作区日志；Computer Use 点击“提交报销”后网络中断，则应先按报销单号查询服务端状态，确认没有成功后才考虑重试。若业务没有查询接口，也不能凭页面是否返回到上一页来判断成功，而应停止并交给人工确认。

这个对照说明了一个重要事实：验证不是动作之后附加的一句“请检查是否成功”，而是动作设计的一部分。每个高影响动作都应预先定义可观察的成功状态、失败状态和未知状态，并为未知状态准备查询、幂等或人工接管路径。这样，视觉模型的识别误差不会自动变成重复付款、重复提交或错误删除。

## 12.18 什么时候需要 Multi-Agent

只有当角色分离直接解决了单 Agent 的瓶颈，才值得使用 Multi-Agent。典型理由包括子任务可以并行、上下文或权限必须隔离、需要独立验证、不同工具/领域有明确责任边界。简单摘要任务拆成 planner、writer、reviewer 往往只增加消息和汇总成本。

判断时要先建立单 Agent 或固定 workflow baseline，再比较成功率、证据质量、冲突率、通信 token、墙钟延迟、单位成功成本和安全事件。多个 Agent 共享同一错误文档或同一模型时，投票不等于独立证据；验证器和外部状态检查通常比增加角色更有价值。

### 12.18.1 用净收益判断是否值得拆分角色

Multi-Agent 的价值不能由“角色数量更多”推导出来。它只有在质量提升足以抵消额外成本、延迟和风险时才有工程意义。可以用相对于单 Agent 或固定 workflow baseline 的净收益做初步分析：

~~~math
\mathrm{NetLift}=\Delta Q-\lambda_c\Delta C-\lambda_l\Delta L-\lambda_r\Delta R
~~~

这里的 `ΔQ` 是任务质量或外部成功率的变化，`ΔC` 是额外 token、工具和基础设施成本，`ΔL` 是墙钟延迟变化，`ΔR` 是权限面、数据流或错误副作用风险的变化；`λ_c`、`λ_l`、`λ_r` 是由业务场景设定的权重。这个式子不是把安全风险换算成金钱后就可以忽略，而是帮助团队显式说出取舍；对支付、删除、医疗或合规任务，`R` 还可能是不可接受的硬约束。

例如，两个并行研究 Agent 让证据支持率提高 0.08，但把延迟提高 1.4 倍、成本提高 2 倍，并且引入了跨租户消息风险。如果任务是低风险的离线研究，质量收益可能值得进一步实验；如果任务会直接修改生产合同，即使平均质量提高，也应该先解决权限隔离和冲突归因，再讨论是否扩大并发。

比较时必须固定任务分布、模型版本、工具、上下文预算、超时和重试策略。否则，Multi-Agent 的提升可能只是因为它获得了更多 token 或更宽的工具权限。还要报告长尾而非只报告平均数：P95 延迟、最坏副作用、冲突率和单位成功任务成本，往往比平均答案分数更能揭示系统是否真正受益。

## 12.19 如何设计 Multi-Agent 系统

先为每个角色写契约：输入 schema、输出 artifact、可见上下文、工具权限、预算、失败状态和 owner。coordinator 根据依赖分配任务、合并结构化结果、处理超时和冲突，但不能代替最终 verifier 或 permission engine。blackboard 保存证据引用和状态索引，完整对话留在各自 trace，减少跨角色污染。

消息要区分事实、证据、假设、建议和动作请求，并带来源、版本、租户、有效期和验证状态。冲突不能靠语言流畅度解决；代码用测试，数据用重算，RAG 用引用支持，业务写入用外部状态，仍无法判断时升级人工。最终结果必须有明确 owner，且能追溯到支持它的 artifact。

### 12.19.1 Blackboard、消息和证据怎样一起工作

blackboard 可以理解为一个共享的“状态索引”，而不是所有 Agent 共用的一段聊天记录。它保存当前任务有哪些 claim、每个 claim 指向哪些 artifact、artifact 处于什么验证状态、谁负责下一步以及它何时过期。原始网页、合同全文、模型内部思路和敏感字段不应因为方便协调就全部复制到 blackboard；它们应留在有访问控制的存储和各自 trace 中，共享端只保留完成协作所需的最小引用。

例如，合同解析 Agent 可以发布一条结构化消息：

```json
{
  "kind": "evidence",
  "claim_id": "payment_term",
  "artifact_id": "contract-1842:v7:clause-3.2",
  "tenant": "tenant-a",
  "state": "candidate",
  "owner": "policy-verifier",
  "expires_at": "2026-08-14T12:00:00Z"
}
```

这条消息只宣称“有一份候选证据”，并没有把候选证据升级为事实，也没有请求执行写入。Policy verifier 读取 artifact 后，可以发布 `verified` 或 `conflicting` 的新版本；coordinator 根据版本号和状态索引更新依赖关系。若同一个 `claim_id` 同时出现两个互相冲突的 artifact，旧消息不能被静默覆盖，系统应保留两者并记录冲突 owner。

消息协议还要处理重放、乱序、过期和重复。每条消息可带 `message_id`、`parent_ids`、单调版本或事件时间；接收者检查租户和权限，再依据幂等键决定是否重复处理。一个晚到的 `verified` 消息不能覆盖已经发现的更新版本；一个过期的授权消息不能让动作重新获得权限。这样，blackboard 是可重建的索引，trace 是各角色的过程证据，外部服务状态则是业务事实，三者不能混为一谈。

角色分离的另一个好处是让责任可定位。若最终结论错误，可以追问是解析 artifact 错、verifier 的规则错、coordinator 合并错，还是外部状态查询错；若所有内容都写进一条自由格式对话，这些错误会被流畅的汇总文本掩盖。Multi-Agent 的协作质量因此取决于契约和证据传播，而不取决于角色名称听起来是否复杂。

## 12.20 如何评估 Agent 系统

评估先定义任务契约和外部验收器，再同时检查结果、过程和陈述。结果层判断目标状态是否达到；过程层检查工具、参数、权限、observation、错误恢复和安全事件；陈述层检查最终 claim 是否由 trace 和外部证据支持。

可验证任务优先使用测试、页面/数据库状态、文件 diff 或引用检查；开放任务再使用明确 rubric、人工抽样和 LLM judge。报告至少包含任务成功、部分成功、工具与参数质量、trace 忠实性、恢复、风险切片、P95 延迟、单位成功任务成本和回归结果。多 Agent 还必须与单 Agent baseline 在相近预算下比较。

### 12.20.1 把危险动作分成三个不同事件

安全评估经常把“模型没有造成事故”简单记成一次安全成功，但至少要区分三种轨迹：系统根本没有提出危险动作；模型提出了危险动作，但独立策略或执行器在副作用发生前阻断；危险动作确实被执行，随后才被发现或补救。三者的风险含义完全不同。

| 轨迹 | 发生了什么 | 可以说明什么 | 不能说明什么 |
|---|---|---|---|
| 未提出 | 模型没有生成或请求高风险动作 | 当前任务和提示下没有观察到该行为 | 不能证明系统遇到诱导时仍然安全 |
| 提出但阻断 | 动作进入 trace，但策略、权限或人工确认拒绝执行 | 防御层在副作用前发挥作用 | 不能把模型本身说成没有风险 |
| 已执行 | executor 接受动作，外部状态发生变化 | 需要检查授权、影响、回滚和审计 | 不能用后续补救抵消原始越权 |

报告指标时应分别统计三类数量。例如，`unauthorized_proposal_rate` 衡量模型提出不应提出的动作，`pre_effect_block_rate` 衡量这些动作在副作用前被阻断的比例，`unauthorized_execution_rate` 则只统计已经改变外部状态的越权动作。它们的分母、任务切片和判定时间必须写清楚；把前两类合并成“未发生事故”，会掩盖策略层依赖或模型行为的退化。

测试还要覆盖边界条件：工具超时后重复请求、权限在规划后被撤销、低信任文档出现在参数附近、人工确认信息过期，以及外部服务返回未知状态。对每条轨迹，评估器都应能回答“危险动作何时被提出、谁阻断、是否产生副作用、最终状态是什么”。只有这样，安全结论才不是对最终文本的猜测，而是对事件链的分析。

## 12.21 为什么 Agent 不能只看最终答案

最终答案只说明 Agent 声称完成了什么。它不能证明测试真的运行过、文件没有被无关修改、数据没有越权读取、工具错误被处理过，或提交动作真的改变了业务状态。

可以构造两条最终文本完全相同的轨迹：一条读取证据、运行验证并重新读取外部状态；另一条没有执行任何工具，只生成“已完成”。如果评估器无法区分它们，说明系统仍在用 Agent 自报代替验收。trace 的意义不是保存更多聊天，而是保存动作、观察、状态、权限、确认和最终 claim 之间的可验证关系。

## 12.22 Agent 安全如何设计

安全设计应从身份、数据、执行和恢复四条边界展开。身份上使用最小权限、短期能力和租户隔离；数据上把网页、文档和工具返回视为低信任 evidence，并在出域前最小化和脱敏；执行上使用 schema/业务/权限三层校验、沙箱、预览和确认；恢复上处理未知状态、幂等、回滚、人工接管和完整审计。

prompt 可以帮助模型理解规则，但不能替代执行器。一个安全系统应能在模型提出危险动作、工具返回低信任内容或 memory 已被污染时，仍由策略和资源服务拒绝越界副作用。

## 12.23 如何防 Prompt Injection

防护的核心不是寻找一条万能过滤规则，而是把低信任内容与控制平面分离。每段内容应携带来源、租户、版本、可信等级和允许用途；证据可以支持 claim，却不能自行改变目标、工具 schema、权限或审批状态。高风险动作还要经过独立的参数、资源、权限和人工确认检查。

评估时同时测漏报和误报：低信任内容不能触发越权动作，但合法引用、经过授权的数据处理和正常网页内容也不应被一律拒绝。发生冲突时，系统应隔离、降级为只读/草稿或升级人工，并保留可解释的阻断记录。

## 12.24 设计一个完整 Agent 系统

可以把合同助手画成几个相互约束的平面：

1. **输入与身份平面**：解析目标、用户、租户、授权和数据分类。
2. **规划与状态平面**：维护 goal、子任务、当前证据、计划版本、预算和停止原因。
3. **工具与执行平面**：registry、schema、业务校验、policy engine、executor、幂等和状态查询。
4. **知识与记忆平面**：RAG 证据、短期状态、长期记忆、来源、版本、过期和删除。
5. **控制与评估平面**：预算、重试、人工确认、verifier、回归、成本和风险切片。
6. **审计与恢复平面**：trace、事件账本、快照、回滚、未知状态和事故响应。

router 可以选择 workflow、单 Agent 或局部 Multi-Agent；planner 不能绕过权限；memory 不能自动成为授权；evaluator 不能只看模型总结；logger 不能把敏感原文无期限复制。生产系统的关键不是组件名称，而是每条跨平面的数据流和控制流都有明确责任。

## 12.25 设计取舍与失败诊断

综合系统最容易犯的错误，是把每个局部组件都“加上”却没有说明它解决了什么约束。自主性提升可能减少人工操作，也可能扩大错误副作用；工具增加可能提高覆盖，也可能增加选择难度和权限面；Multi-Agent 可能并行和互检，也可能增加通信、冲突和责任归因成本。

诊断时先问“失败发生在哪里”：

1. 目标解析错：任务契约或用户澄清不足。
2. 状态错：忽略 observation、版本过期或 memory 污染。
3. 工具错：选择、参数、业务前置条件或权限不匹配。
4. 证据错：检索不足、引用错配、低信任内容进入控制平面。
5. 执行错：沙箱、幂等、未知状态或外部服务处理不当。
6. 验收错：只看最终文本，缺少外部状态或过程 verifier。
7. 成本错：循环、重试、并行或上下文传输超过预算。

每类根因对应不同修复位置。不能用“再加一个 Agent”修复权限问题，也不能用“更强模型”替代外部状态验收；不能因为系统拒绝率高就认为安全，合法任务不可用同样是设计失败。

## 12.26 综合案例：从合同请求到可审计工单

把前面的组件串起来，完整执行一次合同助手任务。用户要求核对供应商合同中的付款期限是否符合政策，并在例外时创建待审核工单。

第一步是身份和范围检查：读取当前用户、租户、合同 id 和版本。第二步是规划：合同解析与政策检索可以并行，但二者输出必须是带来源和定位的 evidence。第三步是规则判断：把合同条款逐项映射到政策条件，缺失字段标记为 unknown，不用模型猜测。第四步是动作策略：若确有例外，只能创建 `pending_review`，参数包括租户、合同版本、证据引用和幂等键；批准、发信和权限修改没有可用工具。第五步是状态确认：重新读取工单，确认对象存在、状态正确、版本匹配，并把最终 claim 绑定到这些事件。

失败路径同样重要。若合同附录无权读取，系统输出“证据不足”并请求授权；若政策版本冲突，保留冲突并升级；若创建工单超时，查询幂等键而不是重复创建；若检索文档包含操作性文字，只把它当 evidence；若用户要求直接批准，策略引擎拒绝并说明可提供的安全替代方案。

这个案例的验收矩阵如下：

| 层 | 验收问题 | 失败例子 |
|---|---|---|
| 目标 | 是否判断了指定合同和政策问题 | 读取了错误合同 |
| 证据 | 每个结论是否有版本和定位 | 引用过期政策 |
| 工具 | 参数、租户、权限是否正确 | 跨租户查询 |
| 状态 | 工单是否真实落库且状态正确 | 只点击按钮未落库 |
| 安全 | 是否越权、出域或未确认写入 | 直接批准或发信 |
| 成本 | 查询、重试和上下文是否可控 | 循环检索无新证据 |
| 说明 | 最终 claim 是否忠实 | 证据不足却说已完成 |

## 12.27 最小可运行 Agent 综合设计审计 demo

下面这个 demo 不调用外部模型，而是构造 5 个综合系统设计记录，检查每个主题是否同时覆盖机制、数量关系、可运行 demo、trace 指标、安全边界、评估方法、项目证据和 trade-off。

它演示的问题是：综合设计不能只列组件名称。一个主题如果没有失败案例、外部验收、权限边界或成本解释，仍然不能支撑可靠系统。代码中的集合是教学标签，不是对读者能力的评分。

这里要区分两种情况：某类证据对当前主题确实不适用，和本来需要该证据但记录中没有它。前者不应进入该主题的加权分母，并显示为 `None`；后者必须进入缺口清单，不能因为集合为空而得到满分。综合覆盖率只能回答记录中声明的维度是否有证据，不能替代对真实系统的运行验收。

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TopicRecord:
    topic_id: str
    title: str
    concepts: set
    formulas: set
    demos: set
    trace_metrics: set
    safety: set
    evaluation: set
    project_evidence: set
    tradeoffs: set
    red_flags: set

    def __post_init__(self):
        for field in ("topic_id", "title"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty text")
        for field in (
            "concepts",
            "formulas",
            "demos",
            "trace_metrics",
            "safety",
            "evaluation",
            "project_evidence",
            "tradeoffs",
            "red_flags",
        ):
            values = getattr(self, field)
            if not isinstance(values, set) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise TypeError(f"{field} must be a set of non-empty strings")


expected = {
    "q1": {
        "concepts": {"goal", "state", "action", "observation", "controller", "trace"},
        "formulas": {"agent_loop_contract"},
        "demos": {"agent_trace_audit"},
        "trace_metrics": {"task_success_rate", "observation_use_rate", "state_update_coverage"},
        "safety": {"permission_policy"},
        "evaluation": {"trace_eval"},
        "project_evidence": set(),
        "tradeoffs": {"agent_vs_workflow"},
    },
    "q2": {
        "concepts": {"tool_registry", "tool_executor", "schema", "function_calling"},
        "formulas": {"tool_selection_accuracy", "argument_validity"},
        "demos": {"tool_call_audit"},
        "trace_metrics": {"schema_valid_rate", "execution_success_rate", "error_recovery_rate"},
        "safety": {"least_privilege", "human_confirmation"},
        "evaluation": {"tool_eval"},
        "project_evidence": set(),
        "tradeoffs": {"schema_vs_business_rule"},
    },
    "q3": {
        "concepts": {"agentic_rag", "retrieval_controller", "evidence_state", "citation"},
        "formulas": {"context_precision", "citation_accuracy"},
        "demos": {"agentic_rag_audit"},
        "trace_metrics": {"new_evidence_gain", "claim_support_rate"},
        "safety": {"untrusted_content", "permission_filter"},
        "evaluation": {"rag_eval"},
        "project_evidence": {"bad_cases", "metrics"},
        "tradeoffs": {"quality_vs_cost"},
    },
    "q4": {
        "concepts": {"code_agent", "repo_understanding", "minimal_patch", "test_feedback"},
        "formulas": {"patch_localization", "validation_coverage"},
        "demos": {"code_agent_audit"},
        "trace_metrics": {"test_pass_rate", "unrelated_change_rate", "command_success_rate"},
        "safety": {"code_sandbox", "user_change_protection"},
        "evaluation": {"programmatic_eval"},
        "project_evidence": {"tests", "owned_work", "bad_cases"},
        "tradeoffs": {"minimal_patch_vs_refactor"},
    },
    "q5": {
        "concepts": {"agent_evaluation", "agent_safety", "multi_agent", "ui_agent"},
        "formulas": {"summary_faithfulness", "unauthorized_action_rate", "safety_profile"},
        "demos": {"agent_eval_audit", "agent_safety_audit"},
        "trace_metrics": {"trace_completeness", "claim_support_rate", "p95_latency", "cost_per_success"},
        "safety": {"least_privilege", "sandbox", "data_flow_guard", "audit_log"},
        "evaluation": {"regression_suite", "human_rubric"},
        "project_evidence": {"baseline", "metrics", "bad_cases"},
        "tradeoffs": {"autonomy_vs_control", "single_vs_multi"},
    },
}


records = [
    TopicRecord(
        "q1",
        "Agent 和普通应用区别",
        {"goal", "state", "action", "observation", "controller", "trace"},
        {"agent_loop_contract"},
        {"agent_trace_audit"},
        {"task_success_rate", "observation_use_rate", "state_update_coverage"},
        {"permission_policy"},
        {"trace_eval"},
        set(),
        {"agent_vs_workflow"},
        set(),
    ),
    TopicRecord(
        "q2",
        "工具调用系统设计",
        {"tool_registry", "tool_executor", "schema", "function_calling"},
        {"tool_selection_accuracy"},
        {"tool_call_audit"},
        {"schema_valid_rate", "execution_success_rate"},
        {"least_privilege"},
        {"tool_eval"},
        set(),
        {"schema_vs_business_rule"},
        {"missing_confirmation"},
    ),
    TopicRecord(
        "q3",
        "Agentic RAG",
        {"agentic_rag", "retrieval_controller", "evidence_state", "citation"},
        {"context_precision", "citation_accuracy"},
        {"agentic_rag_audit"},
        {"new_evidence_gain", "claim_support_rate"},
        {"untrusted_content", "permission_filter"},
        {"rag_eval"},
        {"bad_cases", "metrics"},
        {"quality_vs_cost"},
        set(),
    ),
    TopicRecord(
        "q4",
        "Code Agent 项目深挖",
        {"code_agent", "repo_understanding", "minimal_patch", "test_feedback"},
        {"patch_localization", "validation_coverage"},
        {"code_agent_audit"},
        {"test_pass_rate", "unrelated_change_rate", "command_success_rate"},
        {"code_sandbox", "user_change_protection"},
        {"programmatic_eval"},
        {"tests", "owned_work"},
        {"minimal_patch_vs_refactor"},
        {"weak_project_evidence"},
    ),
    TopicRecord(
        "q5",
        "Agent 评估与安全",
        {"agent_evaluation", "agent_safety", "multi_agent"},
        {"summary_faithfulness", "unauthorized_action_rate"},
        {"agent_eval_audit"},
        {"trace_completeness", "claim_support_rate", "p95_latency"},
        {"least_privilege", "sandbox", "audit_log"},
        {"regression_suite"},
        {"baseline", "metrics"},
        {"autonomy_vs_control"},
        set(),
    ),
]


def coverage(got, want):
    """Return None for an inapplicable category, not a false perfect score."""
    if not isinstance(got, set) or not isinstance(want, set):
        raise TypeError("coverage inputs must be sets")
    if not want:
        return None
    return round(len(got & want) / len(want), 3)


def union(records, field):
    values = set()
    for record in records:
        values |= getattr(record, field)
    return values


def expected_union(field):
    values = set()
    for spec in expected.values():
        values |= spec[field]
    return values


def validate_plan(records, expected, weights):
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list or tuple")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("expected must be a non-empty mapping")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("weights must be a non-empty mapping")
    if any(
        not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        for weight in weights.values()
    ):
        raise ValueError("weights must be finite and non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("weights must have a positive sum")
    topic_ids = set()
    for record in records:
        if not isinstance(record, TopicRecord):
            raise TypeError("records must contain TopicRecord values")
        if record.topic_id in topic_ids:
            raise ValueError(f"duplicate topic_id: {record.topic_id}")
        topic_ids.add(record.topic_id)
        if record.topic_id not in expected:
            raise ValueError(f"missing expected topic: {record.topic_id}")
        if set(weights) != set(expected[record.topic_id]):
            raise ValueError("expected fields and weights must match")
    if topic_ids != set(expected):
        raise ValueError("records and expected topics must have the same IDs")


def score_topic(record, spec, weights):
    parts = {field: coverage(getattr(record, field), spec[field]) for field in weights}
    applicable = {field: value for field, value in parts.items() if value is not None}
    if not applicable:
        return None, parts
    denominator = sum(weights[field] for field in applicable)
    if denominator <= 0:
        return None, parts
    score = round(
        sum(weights[field] * parts[field] for field in applicable) / denominator,
        3,
    )
    return score, parts


weights = {
    "concepts": 0.18,
    "formulas": 0.14,
    "demos": 0.12,
    "trace_metrics": 0.16,
    "safety": 0.16,
    "evaluation": 0.10,
    "project_evidence": 0.08,
    "tradeoffs": 0.06,
}

validate_plan(records, expected, weights)

topic_scores = {}
weak_topics = []
revision_plan = {}

for record in records:
    spec = expected[record.topic_id]
    score, parts = score_topic(record, spec, weights)
    topic_scores[record.topic_id] = score
    missing = {
        field: sorted(spec[field] - getattr(record, field))
        for field in weights
        if spec[field] - getattr(record, field)
    }
    if score is None or score < 0.85 or record.red_flags:
        weak_topics.append(record.topic_id)
        revision_plan[record.topic_id] = {
            "missing": missing,
            "red_flags": sorted(record.red_flags),
            "next_action": "补一个约束、一个可运行 demo、一个失败案例和一个可信资料入口",
        }

overall = {
    "concept_coverage": coverage(union(records, "concepts"), expected_union("concepts")),
    "formula_coverage": coverage(union(records, "formulas"), expected_union("formulas")),
    "demo_coverage": coverage(union(records, "demos"), expected_union("demos")),
    "trace_metric_coverage": coverage(union(records, "trace_metrics"), expected_union("trace_metrics")),
    "safety_coverage": coverage(union(records, "safety"), expected_union("safety")),
    "evaluation_coverage": coverage(union(records, "evaluation"), expected_union("evaluation")),
    "project_evidence_score": coverage(union(records, "project_evidence"), expected_union("project_evidence")),
    "tradeoff_score": coverage(union(records, "tradeoffs"), expected_union("tradeoffs")),
}

red_flags = sorted({flag for record in records for flag in record.red_flags})
observed_scores = [score for score in topic_scores.values() if score is not None]
average_coverage = None if not observed_scores else round(sum(observed_scores) / len(observed_scores), 3)
all_checks_pass = (
    average_coverage is not None
    and average_coverage >= 0.85
    and observed_scores
    and min(observed_scores) >= 0.75
    and overall["safety_coverage"] is not None
    and overall["safety_coverage"] >= 0.90
    and not red_flags
)

empty_plan = []
try:
    validate_plan(empty_plan, expected, weights)
except ValueError:
    pass
else:
    raise AssertionError("empty records must be rejected")

assert coverage(set(), set()) is None
assert score_topic(records[0], {field: set() for field in weights}, weights)[0] is None

try:
    validate_plan(records + [records[0]], expected, weights)
except ValueError:
    pass
else:
    raise AssertionError("duplicate topic IDs must be rejected")

try:
    invalid_weights = dict(weights)
    invalid_weights["concepts"] = float("nan")
    validate_plan(records, expected, invalid_weights)
except ValueError:
    pass
else:
    raise AssertionError("non-finite weights must be rejected")

print(f"topic_scores={topic_scores}")
print(f"overall={overall}")
print(f"red_flags={red_flags}")
print(f"weak_topics={weak_topics}")
print(f"average_coverage={average_coverage}")
print(f"observed_topic_count={len(observed_scores)}")
print(f"all_checks_pass={all_checks_pass}")
print(f"revision_plan={revision_plan}")
```

输出示例：

```text
topic_scores={'q1': 1.0, 'q2': 0.779, 'q3': 1.0, 'q4': 0.973, 'q5': 0.662}
overall={'concept_coverage': 0.955, 'formula_coverage': 0.8, 'demo_coverage': 0.833, 'trace_metric_coverage': 0.857, 'safety_coverage': 0.8, 'evaluation_coverage': 0.833, 'project_evidence_score': 1.0, 'tradeoff_score': 0.833}
red_flags=['missing_confirmation', 'weak_project_evidence']
weak_topics=['q2', 'q4', 'q5']
average_coverage=0.883
observed_topic_count=5
all_checks_pass=False
revision_plan={'q2': {'missing': {'formulas': ['argument_validity'], 'trace_metrics': ['error_recovery_rate'], 'safety': ['human_confirmation']}, 'red_flags': ['missing_confirmation'], 'next_action': '补一个约束、一个可运行 demo、一个失败案例和一个可信资料入口'}, 'q4': {'missing': {'project_evidence': ['bad_cases']}, 'red_flags': ['weak_project_evidence'], 'next_action': '补一个约束、一个可运行 demo、一个失败案例和一个可信资料入口'}, 'q5': {'missing': {'concepts': ['ui_agent'], 'formulas': ['safety_profile'], 'demos': ['agent_safety_audit'], 'trace_metrics': ['cost_per_success'], 'safety': ['data_flow_guard'], 'evaluation': ['human_rubric'], 'project_evidence': ['bad_cases'], 'tradeoffs': ['single_vs_multi']}, 'red_flags': [], 'next_action': '补一个约束、一个可运行 demo、一个失败案例和一个可信资料入口'}}
```

这个 demo 的 `all_checks_pass=False` 不是程序错误，而是说明综合设计记录存在三个缺口：工具调用主题缺少高风险确认口径，Code Agent 主题缺少失败证据，评估与安全主题缺少数据流验收、人工 rubric、成本指标和 single-vs-multi 取舍。它用于发现设计缺口，不用于给人贴“准备好/没准备好”的标签。

## 12.28 第十七册综合练习

1. 为合同助手写出任务契约、状态 schema、工具 registry 和四条信任边界，说明哪些动作只能生成草稿。
2. 给出一个 ReAct 轨迹，其中工具返回状态未知；画出停止、查询、重试和人工接管的状态转移，并说明为什么不能直接重试。
3. 设计 Agentic RAG 的证据账本：为三个 claim 指定来源、版本、定位、权限和支持状态，再构造一个冲突证据样本。
4. 为 Code Agent 设计 workspace 快照与 diff 验收，分别处理用户已有改动、测试失败、依赖升级和危险命令。
5. 为 Browser Agent 列出 API、DOM、accessibility tree 和视觉通道的选择条件，构造一次点击成功但业务状态未改变的失败案例。
6. 设计一个单 Agent 与 Multi-Agent 的配对实验，固定任务、权限和预算，报告成功率、冲突、通信成本、P95 延迟和单位成功任务成本。
7. 为一个敏感文档外部 OCR 请求画数据流图，标出必要字段、禁止字段、脱敏位置、日志位置和删除传播。
8. 解释 demo 中 `all_checks_pass=False` 的每个缺口如何映射到工具 schema、policy engine、executor、verifier 或人工流程的修复。
9. 选取第十七册任意一个章节，写一个“机制—公式—例子—取舍—失败—评估—资料边界”的纵向小节，避免只列定义。

## 12.29 资料入口与证据边界

本章使用的代表性资料入口如下：

- [OpenAI Agents SDK Tools](https://openai.github.io/openai-agents-python/tools/)：工具定义和运行接口；接口存在不等于业务权限配置正确。
- [OpenAI Agents SDK Guardrails](https://openai.github.io/openai-agents-python/guardrails/)：输入/输出和运行时 guardrail 的公开文档；仍需独立执行层保护副作用。
- [OpenAI Model Spec](https://model-spec.openai.com/2025-12-18.html)：指令优先级和行为规范入口；规范不能替代 policy engine。
- [OpenAI Evals](https://github.com/openai/evals)：评估框架入口；具体任务仍需自定义验收器和环境。
- [AgentBench](https://arxiv.org/abs/2308.03688)：多环境 Agent 评估论文。
- [WebArena](https://webarena.dev/)：网页交互环境入口。
- [OSWorld](https://os-world.github.io/)：真实计算机环境中的多模态 Agent 评估入口。
- [SWE-bench](https://www.swebench.com/)：软件工程任务和评估入口。
- [GAIA](https://arxiv.org/abs/2311.12983)：通用助手任务 benchmark 论文。
- [τ-bench](https://arxiv.org/abs/2406.12045)：工具—Agent—用户交互 benchmark 论文。

这些资料分别支持接口语义、风险边界、任务定义或评估环境，不构成一个跨场景的“Agent 能力真值”。模型版本、harness、工具、权限、数据切分、环境镜像和预算变化后，实验结果不能直接横向搬运。对闭源产品和社区榜单，正文只采用公开可核对的接口与条件化结论，不把未披露内部机制写成事实。

## 12.30 本章小结

Agent 的综合设计不是把“大模型、工具、planner、memory 和 multi-agent”排成一张架构图，而是为一个具体任务建立从目标、状态、证据、动作到外部结果的闭环。工具必须有 schema、业务校验和权限；计划必须能被 observation 改写；memory 必须有来源、范围、过期和删除；RAG 必须把 claim 与证据绑定；代码和浏览器操作必须验证真实状态；Multi-Agent 必须有角色契约、消息协议和单 Agent 对照；评估必须同时观察结果、过程、成本和安全。

真正可交付的 Agent 系统，还要诚实处理未知状态、证据不足、权限拒绝和失败恢复。系统的成熟度不由术语数量决定，而由它能否在受限环境中完成任务、解释失败、保护用户和让每个重要结论回到可复核证据来决定。
