# 第三章：ReAct 与 Plan-Act-Observe：让行动受到反馈约束

一个只会生成长答案的模型，无法独立完成需要查询、修改、测试和复核的任务。Agent 要真正做事，必须把“下一步准备做什么”和“环境实际上发生了什么”连接起来。ReAct 和 Plan-Act-Observe 都围绕这个连接展开：模型或控制器提出动作，工具或环境返回观察，系统把观察写回状态，再决定继续、改计划、追问或停止。

这类循环的价值不在于让模型输出更多内部文字，而在于让外部反馈有机会纠正模型。模型可能以为某个文件存在，工具会返回不存在；模型可能认为测试通过，执行器会返回失败；模型可能想发邮件，权限服务会返回需要确认。若系统把这些观察当作下一步决策的输入，错误可以被发现；若系统只让模型连续生成，循环就只是自说自话。

本章先解释 ReAct 和 Plan-Act-Observe 的共同基础，再讨论二者在计划粒度、状态更新和动态重规划上的差别。随后从动作、观察、停止、预算和失败恢复建立形式化表示，并通过一个可运行的 toy trace 审计实验说明如何识别重复动作、忽略观察、过早结束、解析失败和高风险动作被拦截后的恢复问题。

## 0. 资料边界与概念范围

ReAct 论文提供 reasoning 与 acting 交替的研究背景；MRKL Systems 说明语言模型可以与外部模块组合；Plan-and-Solve Prompting 讨论先计划后执行的提示方式；Reflexion 讨论利用反馈进行自我修正的方向。OpenAI Agents SDK 的 tools、guardrails 和 tracing 文档则提供了现代 Agent runtime 中可观察和可控制组件的公开例子。

这些资料的任务、模型、工具和评估条件各不相同。ReAct 不是一个保证所有 Agent 都成功的产品协议，Plan-Act-Observe 也不是只有一种标准消息格式。生产系统可以使用隐藏的内部决策摘要、结构化状态和工具 trace，而不向用户展示完整内部推理文本。本章讨论的是行为和工程边界，不把公开论文中的推理轨迹当成任何闭源模型的内部实现。

本章的代码和任务都是教学构造，只用于审计状态、动作、观察、计划和停止行为。涉及写入、发信和权限阻断时，只说明安全的恢复方向，不提供绕过控制的操作方法。

## 1. 为什么需要一个循环

### 1.1 一次性生成的边界

假设用户说：“查当前退款制度，检查订单是否在期限内，如果符合条件，给出申请步骤，但不要提交退款。”

一次性生成可能出现这些问题：

1. 模型引用的是记忆中的旧制度。
2. 模型没有订单号，却编造了一个查询结果。
3. 模型看到了制度，却没有核对订单日期。
4. 模型把“给出申请步骤”误解成“直接提交申请”。
5. 最终文字听起来完整，却没有外部证据。

如果系统先查制度，再查订单，再根据日期比较结果，每一步都能得到外部观察。查询不到订单时，系统可以追问；制度版本过期时，可以换来源；权限不足时，可以停止。多一步并不自动更可靠，但让错误有了被观察和修正的机会。

### 1.2 小白视角：做题、验算、改错

可以把 Agent 循环想成做一道需要查资料的题：先写下解题计划，做第一步，查看计算结果或资料，再决定第二步。若第一步算错，后续步骤不应假装正确，而要回到错误位置修正。ReAct 强调“想一步、做一步、看结果”；Plan-Act-Observe 强调“有计划地做、观察后更新计划”。

这个类比也说明了停止的重要性。题目做完就应停；资料不足要提问；计算器报错要修参数；重复得到同一错误时不能无限按同一个按钮。

### 1.3 专家视角：闭环的最小状态机

把每一轮看成一个状态机：

~~~text
目标与当前状态
-> 产生候选动作
-> 检查动作
-> 执行动作
-> 接收观察
-> 更新状态和计划
-> 判断停止、追问或继续
~~~

模型负责提出候选，系统负责执行边界。计划不是永久真理，观察也不是自动可信；二者都要进入结构化状态，并带有来源、时间、错误和预算信息。

## 2. ReAct：推理与行动交替

### 2.1 ReAct 的基本形态

ReAct 是 Reasoning and Acting 的缩写。它把面向当前任务的决策摘要与动作交替组织起来：

~~~text
Decision summary: 需要先确认当前退款制度。
Action: search_policy(query=...)
Observation: 找到版本为 2026-07 的制度。
Decision summary: 还缺少订单日期，需要查询订单。
Action: query_order(order_id=...)
Observation: 订单在 6 天前交付。
Final: 根据制度和订单日期给出结论。
~~~

这里的 `Decision summary` 是系统可选的短摘要，不等于必须展示给用户的完整内部推理。真正重要的是 action 和 observation 的因果关系：第二个动作使用了第一个观察提供的制度版本，最终结论同时引用制度和订单状态。

### 2.2 ReAct 的价值

ReAct 适合信息一开始不完整、动作结果会改变下一步、需要探索外部环境的任务。搜索、代码调试、浏览器操作和多步数据调查都具有这种特征。

它带来的主要收益是：

1. 外部观察可以纠正模型的假设。
2. 每一步动作可以单独校验和审计。
3. 失败可以在局部被发现，而不是等到最终答案才暴露。
4. 工具和状态可以插入规则、权限和预算控制。
5. 轨迹可以用于回放、评估和错误归因。

### 2.3 ReAct 的代价

每增加一轮，就增加模型 token、工具延迟、解析错误和上下文负担。模型还可能把“思考更多”误当成“行动更好”，在没有新信息的情况下反复调用同一个工具。ReAct 需要明确的进展条件：本轮观察新增了什么事实，状态改变了什么字段，下一步为什么比上一轮更接近完成。

## 3. Plan-Act-Observe：计划、动作和观察

### 3.1 基本循环

Plan-Act-Observe 可以写成：

~~~text
Plan: 选择当前子目标和完成条件。
Act: 执行一个结构化动作。
Observe: 接收工具或环境反馈。
Update: 更新状态、计划、预算和停止判断。
~~~

代码修复任务可以这样进行：

~~~text
Plan: 先运行测试，定位失败，再做最小修改。
Act: run_tests
Observe: test_login_invalid_password failed
Update: 读取登录函数和相关测试。
Act: read_file
Observe: 空密码分支没有返回错误。
Update: 将修复步骤加入计划，并保留原测试作为回归。
Act: edit_file
Observe: 补丁应用成功。
Act: run_tests
Observe: 全部相关测试通过。
Final: 报告修改文件和验证结果。
~~~

计划在这里提供方向，观察负责改变方向。若第一次测试显示环境依赖缺失，计划就不应继续修改业务代码，而要先处理环境或向用户说明限制。

### 3.2 计划的作用

计划可以保存阶段、依赖和完成条件，帮助系统在长任务中保持全局目标。它还可以在高风险动作前提供预览，让用户知道将要发生什么。

但计划不应写成无法修改的脚本。初始假设可能错误，工具结果可能改变任务结构，外部状态也可能变化。计划必须带版本或更新时间，更新时保留变更原因，避免模型在下一轮继续使用已经失效的步骤。

### 3.3 Update 是不可省略的一步

很多所谓的 Plan-Act-Observe 实际只有 Plan-Act-Observe，没有真正的 Update：工具结果被拼回上下文，模型凭感觉继续生成。这样做会丢掉结构化状态、预算变化和计划变更原因。

Update 至少应写入：新事实、证据来源、已解决事项、未解决事项、动作状态、剩余预算、风险标记和下一步依赖。只有这样，系统才能判断“这一次观察是否带来了进展”。

## 4. ReAct 和 Plan-Act-Observe 的关系

### 4.1 共同点

二者都要求动作受到观察反馈约束，都可以使用工具、状态、计划、控制器和 trace。它们都不等于把完整内部推理展示给用户，也都不保证模型自动拥有执行权限。

### 4.2 关注点不同

ReAct 更强调在每一轮根据当前观察选择下一步，适合开放探索和即时修正；Plan-Act-Observe 更强调计划、状态更新和计划版本，适合阶段较多、依赖关系明显的长任务。

这不是二选一。一个生产 Agent 常常先生成粗粒度计划，再用 ReAct 风格逐步执行：当前计划规定阶段和完成条件，模型每次只提出一个或少量动作，观察回来后更新计划。

### 4.3 三种计划粒度

可以把计划分成三种粒度：

1. 无计划：每轮直接选动作，适合简单查询和低风险任务。
2. 阶段计划：只规定目标、依赖和完成条件，适合代码修复、调查和数据分析。
3. 详细计划：列出具体文件、工具和顺序，适合执行成本高、需要用户预览的任务。

粒度越细，计划越容易在环境变化后失效；粒度越粗，系统越容易迷路。应根据动作成本、风险、可观测性和任务不确定性选择，而不是固定使用一种模式。

## 5. ReAct / PAO 的形式化表示

### 5.1 轨迹

设目标为 `g`，初始计划为 `p_0`，初始状态为 `s_0`。一次轨迹可以写成：

~~~math
\tau=(g,p_0,s_0,d_1,a_1,o_1,s_1,\ldots,d_K,a_K,o_K,s_K,\hat y)
~~~

`d_k` 是第 `k` 步的决策摘要，`a_k` 是动作，`o_k` 是观察，`s_k` 是更新后的状态，`K` 是动作步数，`\hat y` 是最终输出。把决策摘要、动作和观察分开，有助于检查“模型的意图”和“系统的实际行为”是否一致。

### 5.2 动作

动作可以表示为：

~~~math
a_k=(u_k,n_k,\alpha_k,\rho_k)
~~~

`u_k` 是动作类型，`n_k` 是工具名称，`\alpha_k` 是参数，`\rho_k` 是风险和所需授权。`ask_user`、`final` 和 `stop` 也属于控制循环中的动作，它们不一定调用外部工具，但会改变任务状态。

### 5.3 执行前检查

用四个指示量表示执行前检查：

~~~math
I_{\mathrm{allow}}(a_k,s_k)=
I_{\mathrm{schema}}(a_k)\cdot
I_{\mathrm{permission}}(a_k,s_k)\cdot
I_{\mathrm{budget}}(a_k,s_k)\cdot
I_{\mathrm{risk}}(a_k,s_k)
~~~

每个指示量的取值都必须先经过相应检查，可以是 `1`、`0` 或
`unknown`。只有四项都已测量且都为 `1` 时，`I_allow=1`，执行器才
能执行动作并返回观察；只要一项已测量为 `0`，结果就是 `0`；如果没有
足够证据判断，结果应保持 `unknown`，不能把未测量当成允许或拒绝。这里
的乘法只描述四个已知布尔条件同时成立的情形，不意味着模型可以自行把
结果写成 `1`。权限、预算和风险判断必须由系统代码或服务完成。

### 5.4 状态和计划更新

执行结果返回后：

~~~math
s_{k+1}=U(s_k,a_k,o_k)
~~~

如果观察改变了原先的假设，计划也应更新：

~~~math
p_{k+1}=V(p_k,s_{k+1},o_k)
~~~

`U` 负责记录事实、错误、预算和证据；`V` 负责调整阶段、依赖和下一步。不要让模型仅通过一句“计划已更新”伪造状态变化，系统应保存可比较的状态差异。

### 5.5 部分可观察环境

工具结果不是完整环境状态。执行写操作后收到超时，系统可能不知道动作是否已经生效；搜索返回空结果，系统也不知道是资料不存在、查询不对还是权限不足。可以写成：

~~~math
o_k\sim O(\mathord{\cdot}\mid s_{k-1},a_k)
~~~

这里的 `s_{k-1}` 是执行第 `k` 个动作前的状态，`o_k` 是该动作返回
的观察；索引与轨迹中的 `s_0,d_1,a_1,o_1,s_1` 保持一致。这意味着
Agent 需要处理不确定观察，不能把缺少证据当成否定事实，更不能把超时
当成“肯定没有执行”。对状态未知的写操作，应先查询最终状态或转人工。

### 5.6 停止函数

停止判断可以表示为：

~~~math
h(s_k,p_k,b_k)\in
\{\mathrm{continue},\mathrm{final},\mathrm{ask},\mathrm{stop}\}
~~~

`b_k` 是剩余预算。`final` 表示完成条件已满足；`ask` 表示需要用户提供信息或确认；`stop` 表示权限、资源、证据或安全条件不允许继续。把“停止”与“最终成功”分开很重要：系统可以安全停止，但任务仍未完成。

## 6. Action：从意图到实际动作

### 6.1 动作类型

常见动作包括搜索、查询、计算、读取文件、修改文件、运行测试、浏览器操作、调用业务 API、追问用户、生成草稿、请求确认和输出最终结果。每个动作都应有输入、输出、风险、权限和完成条件。

### 6.2 决策摘要不能代替结构化动作

“我想检查订单”是意图，不是可执行动作。真正动作需要工具名、订单 ID、身份、资源范围和只读约束。自由文本动作难以校验，容易把解释性文字当成参数，也难以在日志中统计。

结构化动作还要保存原始模型请求和规范化后的执行请求。若系统自动把日期、用户 ID 或资源名称规范化，应记录变换过程，以便发现执行器是否改变了用户意图。

### 6.3 动作依赖和副作用

动作可以按依赖分为只读、计算、草稿、确认和提交。把这些阶段混成一个 `do_task` 工具，会让模型和系统都难以判断风险。更好的接口是让高风险动作显式经过预览和确认，保持动作边界可见。

### 6.4 动作选择的错误类型

动作错误至少包括：选错工具、参数错误、顺序错误、重复动作、不必要动作、遗漏必要动作和越权动作。评估时应按类型记录，而不是只看最终任务是否成功。一次错误的查询如果碰巧返回正确答案，仍然是需要修复的轨迹问题。

## 7. Observation：反馈、证据和不确定性

### 7.1 Observation 的来源

观察可以来自工具数据、错误返回、测试结果、页面状态、文件内容、用户补充、权限服务和状态查询。来源不同，可信度、时效性和可解释性不同。内部订单服务的结构化状态与网页搜索摘录不能用同一种方式处理。

### 7.2 结构化观察

观察最好有状态、数据、错误、来源、时间、版本和 request id。例如：

~~~json
{
  "status": "permission_denied",
  "data": null,
  "error": {
    "code": "resource_out_of_scope",
    "retryable": false
  },
  "source": "order_service",
  "observed_at": "2026-08-14T10:00:00Z"
}
~~~

如果模型只看到“查询失败”，它可能盲目重试；如果看到 `retryable=false` 和 `resource_out_of_scope`，下一步应是停止或向用户说明边界。

### 7.3 Observation 使用不是把文本放回上下文

调用工具后将结果拼回 prompt，并不能证明 Agent 使用了结果。真正的使用表现为：版本字段改变、下一步工具参数改变、计划阶段改变、最终答案引用结果，或系统明确记录了观察导致的状态差异。

### 7.4 不可信观察

网页、文档、邮件、代码注释和第三方接口文本都可能包含面向模型的指令。它们是数据，不是策略。观察处理器要标注来源，系统策略和权限服务要保持独立，动作执行前还要重新检查权限。即使模型在决策摘要中复述了外部文本，也不能因此改变任务目标或授权范围。

## 8. 计划更新和动态重规划

### 8.1 什么时候先计划

先计划适合步骤较多、工具昂贵、需要跨文件或跨系统、需要用户预览、错误代价高或有明确验收条件的任务。计划可以减少盲目探索，帮助估算资源，并让用户知道系统准备如何工作。

### 8.2 什么时候逐步决策

逐步决策适合信息不完整、环境会变化、每一步观察决定下一步、或任务结构一开始未知的场景。天气查询不需要复杂计划；浏览器排查、代码调试和开放式调查通常需要观察后调整。

### 8.3 计划更新的触发条件

以下情况应触发重规划：

1. 工具返回与计划假设冲突。
2. 关键资源不存在或版本变化。
3. 权限不足或需要用户确认。
4. 预算不足以完成原路径。
5. 新证据改变了任务分解。
6. 连续动作没有产生新的状态变化。
7. 出现新的安全风险或不可逆副作用。

重规划不是把原计划全文重新生成一遍，而是记录哪些假设失效、哪些阶段保留、哪些依赖新增、哪些动作被取消，以及为什么选择新路径。

### 8.4 计划漂移

计划漂移是指执行动作逐渐偏离目标，或者计划中的阶段已变化但系统仍把旧阶段当作当前目标。它可能来自状态更新缺失、上下文过长、工具结果误读或模型为了完成局部动作而忘记全局约束。

检测计划漂移可以比较每一步动作与当前阶段的关系，也可以检查未解决目标是否长期不变。修复方法包括结构化阶段、依赖图、周期性目标复核、无进展检测和人工查看关键节点。

## 9. 循环控制和停止

### 9.1 资源边界

控制器通常限制最大步骤、工具调用数、模型 token、延迟、费用和重试次数。预算要在每轮扣减，工具调用、错误重试和人工等待都应计入相应资源。

### 9.2 进展检测

一个简单的进展定义是：本轮至少新增一个可验证事实、解决一个待办事项、缩小搜索范围或改变了有效计划。相同动作、相同参数、相同错误和相同状态摘要连续出现，通常说明没有进展。

进展检测不能只比较文本是否不同。模型可以用不同措辞重复同一动作；系统应比较规范化动作、参数、错误类别和状态差异。

### 9.3 停止的四种原因

1. `final`：完成条件满足，可以生成结果。
2. `ask`：缺少用户信息或需要确认。
3. `stop`：权限、预算、证据或安全条件不允许继续。
4. `continue`：仍有明确且可验证的下一步。

把四者记录在 trace 中，比只保存“任务失败”更有用。安全停止和能力不足是不同问题，应该由不同团队或组件处理。

### 9.4 过早结束和过晚结束

过早结束会忽略失败观察、未完成子目标或必要验证；过晚结束会造成循环、成本和副作用。停止判断应同时检查完成条件、未解决风险、最后一次观察、剩余预算和动作进展，而不是只由模型输出 `final` 字段决定。

## 10. 失败恢复

### 10.1 参数错误

参数错误通常先修正格式或追问缺失字段。若错误来自资源权限，修正参数不能把越权变成合法；若错误来自业务规则，应把规则返回给模型或用户，而不是让模型继续猜。

### 10.2 搜索无结果

无结果可能意味着查询词不对、索引不完整、权限不足、资料确实不存在或版本不匹配。Agent 可以有限地改写查询或换来源，但应记录尝试次数和结果，不应把空结果变成“没有这项事实”。

### 10.3 测试失败

测试失败观察应该进入状态，并决定下一步是读取错误、检查依赖、修改代码还是报告环境问题。看到测试失败后直接输出“修复完成”属于过早结束，即使最终文字很有信心。

### 10.4 权限阻断

高风险动作被阻断时，正确恢复通常是生成草稿、请求确认、缩小范围或安全停止。系统不应换一个相似工具绕过同一策略，也不应把“被阻断”解释成“执行成功”。

### 10.5 超时和未知状态

只读超时可以有限重试；写操作超时要查询最终状态。若没有可靠状态查询，最安全的结果是记录未知状态并交给人工，不能直接重发。

### 10.6 回滚和检查点

回滚必须由执行器和业务系统支持，不能靠模型生成一句道歉。每个可能产生副作用的动作都应记录资源版本、幂等键和前后状态；只有这样，系统才知道是否可以安全恢复。

## 11. 一个最小工程循环

下面是去掉具体 SDK 的伪代码，展示组件边界：

~~~python
def run_agent_loop(user_goal, model, controller, executor, max_steps):
    state = init_state(user_goal)
    plan = make_initial_plan(state)

    for _ in range(max_steps):
        decision = model.decide_next_action(
            goal=user_goal,
            plan=plan,
            state=controller.model_view(state),
        )

        if decision.type == "final":
            if controller.final_allowed(state, decision.answer):
                return finalize(state, decision.answer)
            decision = controller.request_more_evidence(state)

        allowed, reason = controller.check(decision.action, state)
        if not allowed:
            observation = {"status": "blocked", "reason": reason}
        else:
            observation = executor.run(decision.action)

        state = update_state(state, decision.action, observation)
        plan = update_plan_if_needed(plan, state, observation)

        if controller.should_stop(state, plan):
            return summarize_state(state)

    return summarize_incomplete_state(state)
~~~

这段伪代码没有实现 schema、重试、敏感信息脱敏和并发，但它体现了关键边界：模型只能提出 decision，controller 检查 action，executor 执行，observation 回到 state，plan 依据新状态更新，最终结果还要经过完成条件检查。

## 12. Trace：让循环可以解释和重放

### 12.1 Trace 字段

一条可用轨迹至少保存：

1. 任务 ID、目标和目标版本。
2. 初始状态和计划版本。
3. 每一步 decision summary 和结构化 action。
4. 参数摘要、来源和权限结果。
5. 工具版本、request id、observation 状态和来源。
6. 状态差异、计划变更和预算变化。
7. 错误、重试、阻断和人工确认。
8. 最终状态、停止原因和输出证据。

敏感数据要按照用途最小化保存。Trace 完整不等于保存所有原始内容；真正需要的是足够复盘决策和外部影响的结构化证据。

### 12.2 重放不是重新产生副作用

只读查询可以在固定版本环境中重放；写操作应使用模拟执行、幂等键或状态查询。调试程序若简单地再次调用真实发信或删除接口，可能把一次事故变成两次。Trace 系统要明确区分 replay、dry run 和 execute。

### 12.3 Trace 的评估价值

Trace 可以帮助区分：计划本身错误、动作选择错误、参数错误、执行器错误、观察被忽略、状态未更新、停止错误和权限策略错误。没有 trace 时，所有问题都只能归到最终答案，系统会不断换模型而不修真正的瓶颈。

## 13. ReAct / PAO 的评估指标

### 13.1 任务成功率

对 `N` 条任务：

~~~math
A_{\mathrm{task}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[z_i=1]
~~~

`z_i` 必须由任务完成条件定义。代码任务可能要求测试通过，调查任务可能要求证据覆盖，写操作任务可能要求权限、确认和最终状态都正确。
这里要求 `N>0`；没有任务样本时，任务成功率是 `None`，而不是 `1` 或
`0`。`None` 表示没有测量到该指标，不能拿它宣称系统达标。

### 13.2 动作和计划

每个步骤的动作准确率为：

~~~math
A_{\mathrm{act}}=\frac{1}{N_s}\sum_{i=1}^{N_s}\mathbf{1}[\hat a_i=a_i^*]
~~~

计划遵循率可以检查动作是否服务于当前计划阶段：

~~~math
A_{\mathrm{plan}}=\frac{1}{N_s}\sum_{i=1}^{N_s}I_{\mathrm{align}}(a_i,p_i)
~~~

当动作与执行时的计划阶段一致时，`I_align` 取 1，否则取 0；重规划后应使用新计划版本判断，而不是拿旧计划机械比较。

两者都不能替代任务成功率。一个动作可能符合计划，但计划本身已经过期；一个动作可能偏离初始计划，却是观察后正确的重规划结果。

### 13.3 观察和状态

设 `N_o` 为确实收到可审计 observation 的步骤数。直接回答、没有调用
工具的步骤不进入这个分母；只有 `N_o>0` 时才计算观察使用率：

~~~math
R_{\mathrm{obs}}=\frac{1}{N_o}\sum_{i\in\mathcal{O}}I_{\mathrm{obs}}(o_i)
~~~

其中 `\mathcal{O}` 是收到 observation 的步骤集合。`I_obs` 在观察改变
后续状态或动作时取 1，未改变时取 0；没有观察不能记为 0，也不能把
“模型直接回答”算作使用了 observation。

状态更新覆盖率（对所有需要状态更新的动作步骤）为：

~~~math
R_{\mathrm{state}}=\frac{1}{N_s}\sum_{i=1}^{N_s}I_{\mathrm{state}}(\Delta s_i)
~~~

这里的 `N_s` 必须大于 0；`I_state` 在需要记录的状态差异确实写入
trace 时取 1。状态没有变化和状态更新没有被记录是两件事：前者可能
是合法的空变化，后者则是审计缺口，应在数据中区分。

如果工具返回新版本号却没有状态差异，模型即使在下一轮提到这个版本，也很难审计它是否真正使用了 observation。

### 13.4 重复、过早结束和预算

重复动作率：

~~~math
R_{\mathrm{repeat}}=\frac{1}{N}\sum_{i=1}^{N}I_{\mathrm{repeat}}(\tau_i)
~~~

过早结束率：

~~~math
R_{\mathrm{early}}=\frac{1}{N}\sum_{i=1}^{N}I_{\mathrm{early}}(\tau_i)
~~~

预算超限率：

~~~math
R_{\mathrm{over}}=\frac{1}{N}\sum_{i=1}^{N}I_{\mathrm{over}}(\tau_i)
~~~

`I_repeat`、`I_early` 和 `I_over` 分别表示无进展重复、未满足完成条件就结束、以及超出步数或工具预算；发生对应情况时取 1，否则取 0。
以上按轨迹统计的三个公式都要求 `N>0`；动作级指标中的 `N_s` 要求
`N_s>0`。如果评估集没有动作、观察或可判定的阻断事件，对应指标应
报告为 `None`，而不是用空集合制造一个满分。对恢复指标还要单独说明
分母是“被阻断的步骤数”，只在该分母大于 0 时计算。

这些指标的方向不同：任务成功、动作、计划和观察使用率通常越高越好，重复、过早结束和超限率通常越低越好。报告时不要把它们未经说明地平均成一个总分。

### 13.5 恢复指标

对失败案例，应判断 Agent 是否采取了正确恢复，并最终处于安全、可解释状态。恢复成功不一定等于任务成功：一个被权限阻断后安全停止的任务，恢复状态可能是正确的，业务任务却仍然未完成。把安全恢复和任务完成分开记录，可以避免鼓励危险的强行执行。

令 `\mathcal{B}` 为被阻断的步骤集合，恢复率定义为：

~~~math
R_{\mathrm{recover}}=
\frac{1}{|\mathcal{B}|}
\sum_{k\in\mathcal{B}}I_{\mathrm{recover}}(k),
\qquad |\mathcal{B}|>0
~~~

没有任何阻断步骤时，`R_recover` 是 `None`，因为“没有需要恢复的
案例”不等于“恢复能力达到 100%”。同样，任何未测量的指标都应保留
为 `unknown` 或 `None`，不能为了通过阈值而填入默认值。

## 14. 最小可运行 ReAct / PAO 轨迹审计

下面的 demo 不调用模型或真实 API，只审计六条 toy trace。它包含一个成功查询、一个需要计划更新的检索任务、一个重复循环、一个忽略测试错误的过早结束、一个被阻断后没有恢复的写入动作，以及一个参数解析失败后恢复的计算任务。

~~~python
from collections import Counter


LIMITS = {"max_steps": 4, "max_tool_calls": 3}


TRACES = [
    {
        "id": "weather_success",
        "goal": "answer weather question",
        "expected_tools": ["get_weather"],
        "plan_steps": ["get weather", "answer"],
        "steps": [
            {
                "decision_ok": True,
                "action": "get_weather",
                "args_valid": True,
                "expected_action": "get_weather",
                "observation": "sunny 25C",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "get weather",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            }
        ],
        "plan_updates": 0,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
    {
        "id": "refund_plan_update_success",
        "goal": "answer refund policy with evidence",
        "expected_tools": ["search_docs", "query_order"],
        "plan_steps": ["find policy", "check order", "answer with evidence"],
        "steps": [
            {
                "decision_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "search_docs",
                "observation": "policy says 10 days",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "find policy",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
            {
                "decision_ok": True,
                "action": "query_order",
                "args_valid": True,
                "expected_action": "query_order",
                "observation": "order delivered 6 days ago",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "check order",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
        ],
        "plan_updates": 1,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
    {
        "id": "repeat_loop_failure",
        "goal": "find current inventory",
        "expected_tools": ["query_inventory"],
        "plan_steps": ["query inventory", "answer"],
        "steps": [
            {
                "decision_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "no inventory data",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "decision_ok": True,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "decision_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "decision_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
            {
                "decision_ok": False,
                "action": "search_docs",
                "args_valid": True,
                "expected_action": "query_inventory",
                "observation": "same empty result",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "query inventory",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            },
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "max_steps",
        "expected_stop": "ask",
    },
    {
        "id": "ignored_error_premature_final",
        "goal": "fix failing test",
        "expected_tools": ["run_tests", "read_file"],
        "plan_steps": ["run tests", "inspect error", "patch"],
        "steps": [
            {
                "decision_ok": True,
                "action": "run_tests",
                "args_valid": True,
                "expected_action": "run_tests",
                "observation": "test failed: missing edge case",
                "observation_used": False,
                "state_updated": False,
                "plan_step": "run tests",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": False,
            }
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "final",
        "expected_stop": "continue",
    },
    {
        "id": "blocked_write_not_recovered",
        "goal": "send summary email",
        "expected_tools": ["draft_email", "ask_confirmation"],
        "plan_steps": ["draft", "confirm", "send"],
        "steps": [
            {
                "decision_ok": True,
                "action": "send_email",
                "args_valid": True,
                "expected_action": "draft_email",
                "observation": "blocked: confirmation required",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "send",
                "plan_aligned": False,
                "parse_ok": True,
                "blocked": True,
                "recovered": False,
            }
        ],
        "plan_updates": 0,
        "success": False,
        "stop_reason": "blocked",
        "expected_stop": "ask",
    },
    {
        "id": "parse_failure_recovered",
        "goal": "calculate invoice total",
        "expected_tools": ["calculator"],
        "plan_steps": ["calculate", "answer"],
        "steps": [
            {
                "decision_ok": True,
                "action": "calculator",
                "args_valid": False,
                "expected_action": "calculator",
                "observation": "parse error: expression missing",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "calculate",
                "plan_aligned": True,
                "parse_ok": False,
                "blocked": False,
                "recovered": True,
            },
            {
                "decision_ok": True,
                "action": "calculator",
                "args_valid": True,
                "expected_action": "calculator",
                "observation": "total=128",
                "observation_used": True,
                "state_updated": True,
                "plan_step": "calculate",
                "plan_aligned": True,
                "parse_ok": True,
                "blocked": False,
                "recovered": True,
            },
        ],
        "plan_updates": 1,
        "success": True,
        "stop_reason": "final",
        "expected_stop": "final",
    },
]


def rate(num, den):
    if (
        not isinstance(num, int)
        or isinstance(num, bool)
        or not isinstance(den, int)
        or isinstance(den, bool)
    ):
        raise TypeError("rate expects integer counts")
    if den < 0 or num < 0 or num > den:
        raise ValueError("rate requires 0 <= numerator <= denominator")
    return None if den == 0 else round(num / den, 3)


def at_least(value, threshold):
    return value is not None and value >= threshold


def at_most(value, threshold):
    return value is not None and value <= threshold


def is_zero(value):
    return value is not None and value == 0.0


def has_observation(step):
    value = step.get("observation")
    return value is not None and value != ""


def repeated_action(trace):
    seen = set()
    for step in trace["steps"]:
        signature = (step["action"], step.get("plan_step"))
        if signature in seen:
            return True
        seen.add(signature)
    return False


def over_budget(trace):
    tool_calls = len([s for s in trace["steps"] if s["action"] != "final"])
    return len(trace["steps"]) > LIMITS["max_steps"] or tool_calls > LIMITS["max_tool_calls"]


def premature_final(trace):
    return trace["stop_reason"] == "final" and trace["expected_stop"] != "final"


def audit_react_traces(traces):
    steps = [step for trace in traces for step in trace["steps"]]
    total_steps = len(steps)
    failed_traces = []
    reason_counts = Counter()
    for trace in traces:
        flags = []
        if not trace["success"]:
            flags.append("task_failed")
        if over_budget(trace):
            flags.append("budget_overrun")
        if trace["stop_reason"] != trace["expected_stop"]:
            flags.append("bad_stop")
        if repeated_action(trace):
            flags.append("repeated_action")
        if premature_final(trace):
            flags.append("premature_final")
        if any(not step["parse_ok"] for step in trace["steps"]):
            flags.append("parse_failure")
        if any(not step["plan_aligned"] for step in trace["steps"]):
            flags.append("plan_drift")
        if any(not step["observation_used"] for step in trace["steps"]):
            flags.append("observation_ignored")
        if any(step["blocked"] and not step["recovered"] for step in trace["steps"]):
            flags.append("blocked_not_recovered")
        if flags:
            failed_traces.append(trace["id"])
            reason_counts.update(flags)

    observed_steps = [step for step in steps if has_observation(step)]
    blocked_steps = [step for step in steps if step["blocked"]]
    metrics = {
        "task_success_rate": rate(sum(t["success"] for t in traces), len(traces)),
        "decision_action_alignment": rate(sum(s["decision_ok"] for s in steps), total_steps),
        "action_accuracy": rate(sum(s["action"] == s["expected_action"] for s in steps), total_steps),
        "argument_valid_rate": rate(sum(s["args_valid"] for s in steps), total_steps),
        "plan_adherence_rate": rate(sum(s["plan_aligned"] for s in steps), total_steps),
        "plan_update_coverage": rate(sum(t["plan_updates"] > 0 for t in traces), len(traces)),
        "observation_use_rate": rate(
            sum(s["observation_used"] for s in observed_steps), len(observed_steps)
        ),
        "state_update_coverage": rate(sum(s["state_updated"] for s in steps), total_steps),
        "parse_failure_rate": rate(sum(not s["parse_ok"] for s in steps), total_steps),
        "repeat_action_rate": rate(sum(repeated_action(t) for t in traces), len(traces)),
        "budget_overrun_rate": rate(sum(over_budget(t) for t in traces), len(traces)),
        "premature_final_rate": rate(sum(premature_final(t) for t in traces), len(traces)),
        "stop_correct_rate": rate(
            sum(t["stop_reason"] == t["expected_stop"] for t in traces), len(traces)
        ),
        "blocked_recovery_rate": rate(
            sum(s["recovered"] for s in blocked_steps),
            len(blocked_steps),
        ),
    }
    checks = {
        "success_ok": at_least(metrics["task_success_rate"], 0.75),
        "action_ok": at_least(metrics["action_accuracy"], 0.85),
        "plan_ok": at_least(metrics["plan_adherence_rate"], 0.85),
        "observation_ok": at_least(metrics["observation_use_rate"], 0.90),
        "state_ok": at_least(metrics["state_update_coverage"], 0.90),
        "parse_ok": at_most(metrics["parse_failure_rate"], 0.05),
        "repeat_ok": is_zero(metrics["repeat_action_rate"]),
        "budget_ok": is_zero(metrics["budget_overrun_rate"]),
        "premature_final_ok": is_zero(metrics["premature_final_rate"]),
        "stop_ok": at_least(metrics["stop_correct_rate"], 0.90),
        "blocked_recovery_ok": at_least(metrics["blocked_recovery_rate"], 0.80),
    }
    return {
        "metrics": metrics,
        "failed_traces": failed_traces,
        "top_failure_reasons": reason_counts.most_common(),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


report = audit_react_traces(TRACES)
print("metrics=", report["metrics"])
print("failed_traces=", report["failed_traces"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("checks=", report["checks"])
print("all_checks_pass=", report["all_checks_pass"])
~~~

预期输出如下：

~~~text
metrics= {'task_success_rate': 0.5, 'decision_action_alignment': 0.75, 'action_accuracy': 0.5, 'argument_valid_rate': 0.917, 'plan_adherence_rate': 0.5, 'plan_update_coverage': 0.333, 'observation_use_rate': 0.5, 'state_update_coverage': 0.5, 'parse_failure_rate': 0.083, 'repeat_action_rate': 0.333, 'budget_overrun_rate': 0.167, 'premature_final_rate': 0.167, 'stop_correct_rate': 0.5, 'blocked_recovery_rate': 0.0}
failed_traces= ['repeat_loop_failure', 'ignored_error_premature_final', 'blocked_write_not_recovered', 'parse_failure_recovered']
top_failure_reasons= [('task_failed', 3), ('bad_stop', 3), ('repeated_action', 2), ('plan_drift', 2), ('observation_ignored', 2), ('budget_overrun', 1), ('premature_final', 1), ('blocked_not_recovered', 1), ('parse_failure', 1)]
checks= {'success_ok': False, 'action_ok': False, 'plan_ok': False, 'observation_ok': False, 'state_ok': False, 'parse_ok': False, 'repeat_ok': False, 'budget_ok': False, 'premature_final_ok': False, 'stop_ok': False, 'blocked_recovery_ok': False}
all_checks_pass= False
~~~

### 14.1 先看重复循环

`repeat_loop_failure` 连续五次使用同一个搜索动作、同一个计划阶段和同一种空结果。它不仅超出四步和三次工具调用的限制，还没有把观察写进状态。重复动作率和预算超限率把这个问题明确标出来；真实控制器应在第二次无进展后考虑换工具、追问或停止，而不是等到上限才结束。

### 14.2 再看过早结束

`ignored_error_premature_final` 已经获得测试失败观察，却直接返回 `final`。如果只看它有没有调用 `run_tests`，这个案例可能被误判为完成；观察使用率、状态更新覆盖率和停止正确率揭示了真正问题。

### 14.3 再看阻断后的恢复

`blocked_write_not_recovered` 的发信动作被阻断，系统记录了阻断观察，但没有转为草稿或请求确认。安全系统应该把“动作被拦截”与“任务成功”分开；正确恢复可能是 `ask`，而不是寻找另一个写工具。

### 14.4 解析失败也要保留

`parse_failure_recovered` 最终成功，但第一次计算参数解析失败。成功恢复不应抹掉失败轨迹，否则线上数据会低估解析器和参数生成的脆弱性。恢复指标和失败率应同时报告。

## 15. 实验设计和失败归因

### 15.1 先做 direct baseline

先测一次直接回答或固定 workflow，再加入 ReAct、计划、工具和恢复。这样可以知道循环带来的增益是否超过延迟和复杂度，也能发现 Agent 是否伤害了简单任务。

### 15.2 再做单变量消融

分别改变计划粒度、最大步骤、重复检测、观察摘要、重试次数和状态更新策略。若同时改变模型、工具和 harness，就无法知道质量变化来自哪里。

### 15.3 加入变体和故障

对同一任务改变工具顺序、参数格式、文档版本、缺失字段和环境状态；再注入空结果、超时、权限拒绝、解析错误和冲突观察。真正的闭环能力要在反馈变化时保持正确，而不是只在固定 happy path 上复现模板。

### 15.4 按层归因

- 决策摘要和动作不一致：策略或解析层问题。
- 动作正确但参数错误：参数生成或 schema 层问题。
- 参数正确但执行失败：工具、环境或权限层问题。
- 观察返回但状态不变：观察处理或状态更新问题。
- 状态正确但仍循环：控制器或停止函数问题。
- 安全动作被阻断后强行寻找替代路径：策略边界和恢复问题。

分层归因比简单地增加模型大小更能指导修复。

## 16. 常见失败模式

1. 决策摘要看起来合理，但动作选错。
2. 动作选对，但参数错误。
3. 工具失败后编造成功结果。
4. 观察被拼回上下文，却没有进入状态。
5. 初始计划失效后仍按旧计划执行。
6. 相同动作和错误重复出现。
7. 达到局部目标就过早结束。
8. 达成目标后仍继续调用工具。
9. 外部文档中的文本改变了高风险动作。
10. 上下文压缩丢失了用户约束或资源版本。
11. 写操作被阻断后没有安全恢复。
12. 达到预算后仍继续尝试。

这些问题分别落在决策、动作、参数、观察、状态、计划、控制器和权限层。一个万能提示词很难覆盖所有层，系统需要组合确定性校验、结构化状态、执行器策略和轨迹评估。

## 17. 练习：从循环到证据

### 练习一：画出退款判断轨迹

把“查询退款政策并判断订单是否符合条件”写成一条 PAO 轨迹，至少包含目标、初始计划、制度观察、订单观察、状态差异、计划更新和停止原因。

### 练习二：标注轨迹字段

给出一条三步 ReAct 轨迹，标出 `d_k`、`a_k`、`o_k`、`s_k`、`p_k` 和预算变化。说明哪些内容来自工具，哪些内容只是模型候选。

### 练习三：设计无进展检测

构造三次相同工具和参数得到相同错误的 trace。定义动作指纹、错误指纹和状态差异，说明何时换工具、何时追问、何时停止。

### 练习四：观察被忽略

构造一个测试返回失败但 Agent 直接输出成功的样本。分别写出结果指标、观察使用指标和停止指标为什么会给出不同结论。

### 练习五：设计计划版本

为代码修复任务设计 `plan_v1` 和 `plan_v2`。让 `plan_v2` 因依赖缺失而改变路径，保留变更原因和未完成项，说明如何避免模型继续使用旧阶段。

### 练习六：安全恢复

设计一个发信动作被阻断后的 trace。比较生成草稿、请求确认、缩小收件人范围和安全停止四种恢复，并说明哪些条件下可以进入下一步。

### 练习七：运行审计实验

把 `repeat_loop_failure` 改成第二次空结果后追问用户，把 `blocked_write_not_recovered` 改成请求确认。重新运行程序，观察哪些指标改善，哪些失败仍然存在。

## 18. 本章小结

ReAct 和 Plan-Act-Observe 的共同核心是：动作必须受到环境观察约束，观察必须写回状态，计划必须允许更新，控制器必须决定继续、追问、停止或结束。ReAct 更强调推理摘要与行动交替，PAO 更强调计划、状态和计划版本；生产系统可以把粗粒度计划与逐步 ReAct 组合起来。

可靠循环需要结构化动作、来源清楚的 observation、可比较的状态差异、进展检测、有限预算、错误恢复、权限检查和 trace。过早结束、重复动作、忽略观察、计划漂移和阻断后强行执行，都应在轨迹中单独记录。

评估不能只看最终任务成功率，还要看动作准确率、计划遵循率、观察使用率、状态更新、解析失败、重复率、预算超限、停止正确率和安全恢复。一个最终结果正确的轨迹，可能仍然包含需要修复的动作或证据问题；一个安全停止的轨迹，也不应被误报成任务已经完成。

下一章将进一步讨论 Planning 与 Task Decomposition，研究复杂目标如何拆成有依赖关系的子任务，什么时候串行，什么时候并行，以及失败后如何重排和回退。

## 19. 延伸资料与证据边界

1. ReAct: Synergizing Reasoning and Acting in Language Models，论文：<https://arxiv.org/abs/2210.03629>。
2. MRKL Systems，论文：<https://arxiv.org/abs/2205.00445>。
3. Plan-and-Solve Prompting，论文：<https://arxiv.org/abs/2305.04091>。
4. Reflexion: Language Agents with Verbal Reinforcement Learning，论文：<https://arxiv.org/abs/2303.11366>。
5. OpenAI Agents SDK 文档入口：<https://openai.github.io/openai-agents-python/>。

论文资料支撑研究方法和概念来源，官方文档支撑公开 runtime 组件。论文中的 thought、action 和环境接口与生产系统的内部消息格式不必相同；真实系统还要结合自己的工具版本、权限配置、状态存储、故障样本和评估 harness 独立验证。
