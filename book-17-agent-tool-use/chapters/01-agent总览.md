# 第一章：Agent 总览：让模型在受控环境中完成任务

大语言模型最初主要承担一件事：根据输入生成文本。用户提出问题，模型返回答案；用户继续追问，模型再根据上下文生成下一段答案。Agent 在这个基础上增加了一个重要变化：模型不只是说什么，还要在目标、状态、工具和环境反馈之间持续做决定。

这个变化看起来像是“给模型接几个工具”，实际上改变了系统的工程对象。一次普通问答主要关心答案是否正确、表达是否清楚；一次 Agent 任务还要关心它是否选对工具、参数是否合法、是否读懂反馈、是否更新了状态、是否在预算内停止、是否做了未经授权的事情，以及最终文字是否真的反映了外部世界发生的变化。

本章从一个可读、可复现的任务出发，逐步建立 Agent 的概念和数学表示。读者会看到 Agent 与 Chat Model、RAG、Workflow 的边界，理解 goal、state、action、tool、observation、policy、memory 和 controller 各自解决什么问题，再把这些组件组合成一个循环。最后用一个不依赖外部服务的 Python 审计程序，把“这个 Agent 看起来会做事”转换成可以检查的轨迹指标。

## 0. 先建立阅读边界

Agent 是一个很宽的词。一个只调用一次天气 API 的聊天应用、一个会自主修改代码的编程助手、一个按固定节点运行的审批流程，都可能在产品宣传中被叫作 Agent。若不先明确边界，读者很容易把工具调用、检索、工作流和真正的多步环境交互混为一谈。

本章采用一个实用而不是绝对的定义：

> Agent 是由模型或其他策略驱动、围绕目标在环境中进行多步决策，并通过受控动作获得反馈、更新状态和结束任务的系统。

这个定义有四个关键词。

第一是“目标”。系统必须知道要完成什么，而不是只追求下一段文字看起来流畅。第二是“多步决策”。下一步应当可能受到前一步观察结果影响，而不是把固定脚本换成自然语言。第三是“环境反馈”。动作要有外部结果，模型需要根据结果修正计划。第四是“受控”。权限、预算、停止条件、审计和恢复不能完全交给模型自行决定。

本章只讨论 Agent 总体框架。工具 schema 和 function calling、MCP、ReAct、规划与任务分解、memory、multi-agent、computer use、代码 Agent、长周期 runtime 会在后续章节分别展开；这里为它们建立共同坐标系。

### 0.1 资料如何使用

ReAct、Toolformer 和 MRKL Systems 论文提供了 Agent 推理、工具使用和模块化系统的经典研究背景。它们说明了“语言模型提出动作并利用外部结果”这一类方法为什么有价值，但论文中的任务、工具和实验环境不能直接当作生产系统保证。

OpenAI Agents SDK 等官方文档可以作为 tools、handoffs、guardrails 和 tracing 等公开接口概念的资料来源，但公开接口不等于某个闭源模型的内部实现。NIST 的生成式 AI 风险管理资料和 OWASP LLM 应用安全资料用于说明权限、提示注入、过度代理和审计等工程风险；它们也不替代具体业务的安全评审。

本章中的报销助手、代码修复和 trace 审计都是教学构造。公式用于帮助读者拆解系统，不代表所有产品都用同样的实现。凡是涉及模型能力、长周期成功率或多 Agent 协同规模的结论，都需要绑定模型版本、工具集合、环境、提示词、预算和评测 harness，不能从一个产品页面的宣传数字推导出通用规律。

## 1. 从一个任务看 Agent 为什么存在

设想一个企业报销助手接到请求：

“请检查这张差旅报销单是否符合当前制度，列出不合规项，计算可报销金额，并在需要时生成修改建议。不要自动提交审批。”

这不是一个单纯的问答问题。系统至少需要完成以下工作：

1. 读取当前版本的差旅制度，而不是凭记忆回答。
2. 读取报销单中的日期、城市、金额、发票类型和缺失字段。
3. 将制度条款与报销单逐项对齐。
4. 计算金额，并说明计算所依赖的规则。
5. 区分“资料缺失”“规则不允许”和“金额计算错误”。
6. 生成建议，但不能因为用户说“顺便提交”就绕过本任务中的只读限制。
7. 在资料不足时追问，在证据冲突时停止猜测。

如果模型直接生成一段答案，它可能引用旧制度，也可能把一条经验规则写成公司规定。即使最后金额碰巧正确，证据和权限也可能不正确。Agent 的价值不在于让答案更长，而在于把任务拆成可以观察、执行、校验和恢复的步骤。

### 1.1 小白视角：Agent 像一个受控的办事员

可以把普通聊天模型想成一个善于写作和解释的顾问。你问它“这条制度是什么意思”，它给出说明。Agent 则像一个受控的办事员：它先确认任务目标，查看允许使用的资料，调用查询或计算工具，读取返回结果，再决定下一步。它不能因为自己“觉得应该可以”就直接改变审批状态。

这个类比有一个重要限制：Agent 不是人，也没有天然的责任意识。办事员的权限通常由组织制度和系统账号限制；Agent 的权限必须由软件显式实现。模型生成的“我要发邮件”只是一个建议动作，真正的发信是否执行，应由权限服务、参数校验和业务规则决定。

### 1.2 专家视角：策略与执行必须分层

把 Agent 看成一个策略系统会更准确。模型、规则或规划器负责提出候选动作；控制器负责检查候选动作；执行器负责在环境中执行被允许的动作；观察处理器负责把结果写回状态。这样做的核心原因是，生成动作和产生副作用是两个不同的信任边界。

若把模型输出直接当成执行指令，任何提示注入、参数错误、状态过期或权限误判都可能变成外部副作用。若将动作提议、策略检查和执行记录分开，系统才有机会在执行前拒绝、在执行后审计，并在失败时重放或回滚。

## 2. Agent、Chat Model、RAG 和 Workflow

### 2.1 Chat Model：输入到输出

聊天模型的最小流程可以写成：

~~~text
user input -> model response
~~~

它当然可以有多轮对话，也可以输出 JSON 或工具调用。但如果一次请求只产生一次结构化调用，调用结果不影响后续决策，系统通常还只是“带工具接口的模型调用”，不一定需要称为 Agent。

聊天模型的主要状态通常在上下文窗口中。它的评估重点往往是答案正确性、风格、事实性和拒答行为。只要没有持续的外部动作，权限、幂等、回滚和执行日志的复杂度相对较低。

### 2.2 RAG：把外部知识放进回答上下文

典型 RAG 流程是：

~~~text
query -> retrieve -> rerank -> construct context -> generate answer
~~~

RAG 主要解决知识获取和证据 grounding 问题。它可以固定检索一轮，也可以在检索失败时增加一轮，但核心输出仍然是基于证据生成答案。用户问“公司差旅制度中住宿标准是多少”，通常是 RAG；系统需要找制度、截取条款并给出引用，而不是持续操作外部系统。

### 2.3 Workflow：把流程写死以换取确定性

Workflow 把步骤和分支预先定义好，例如：

~~~text
分类 -> 检索 -> 规则检查 -> 计算 -> 格式化输出
~~~

固定流程的优点是可测试、可观测、延迟稳定、权限边界清楚。对于规则明确、分支有限的审批、数据同步和报表任务，workflow 往往比开放式 Agent 更合适。它的缺点是遇到未预料的分支时需要修改流程，不能灵活探索新的解决路径。

### 2.4 Agent：让下一步依赖于观察结果

Agent 的流程更接近：

~~~text
goal -> inspect state -> select action -> validate -> execute
-> observe -> update state -> decide whether to continue
~~~

如果检索结果显示制度版本过期，下一步可能是查找最新版本；如果报销单缺少发票日期，下一步可能是向用户追问；如果计算工具返回参数错误，下一步可能是修正参数；如果动作需要写入审批系统，下一步可能是展示预览并请求确认。下一步不是固定的，它由当前状态和观察结果共同决定。

### 2.5 四者不是互斥标签

生产系统经常采用混合架构：外层 workflow 负责权限、审计和关键节点，RAG 负责证据获取，Agent 只在开放分支中选择检索、计算或追问动作。一个系统可以同时是 RAG 应用、workflow 和局部 Agent；重要的不是给产品贴上哪个标签，而是明确哪些决策是动态的，哪些动作可产生副作用，以及每个边界由谁负责。

| 系统形态 | 主要问题 | 下一步是否由外部反馈决定 | 常见风险 |
| --- | --- | --- | --- |
| Chat Model | 生成回答 | 通常不决定外部动作 | 幻觉、事实错误 |
| RAG | 获取证据并生成 | 通常有限 | 检索错、引用不支持结论 |
| Workflow | 稳定执行预定义流程 | 由预定义分支决定 | 分支遗漏、流程僵化 |
| Agent | 在环境中完成多步任务 | 通常是 | 工具误用、循环、权限和成本失控 |

## 3. Agent 的形式化表示

形式化不是为了把每个系统都变成复杂的强化学习算法，而是为了让“状态丢了”“观察没用”“预算超了”这些问题有明确的落点。

### 3.1 目标、状态和历史

把一个任务在第 `t` 步的工作状态写成：

~~~math
s_t=(g,x_t,p_t,r_t,b_t,ell_t)
~~~

其中：

- `g` 是不可随意改变的目标和完成标准；
- `x_t` 是当前已确认的事实、字段和中间产物；
- `p_t` 是当前用户身份、工具权限和业务策略；
- `r_t` 是待解决问题、失败记录和风险标记；
- `b_t` 是剩余步数、token、时间和费用预算；
- `ell_t` 是审计日志、版本信息和可恢复检查点。

这里的状态不等于 prompt。Prompt 是送入模型的一次文本或消息集合，状态则是系统对任务的结构化记录。系统可以从状态中选择一部分生成 prompt，也可以把某些字段只交给权限服务而不交给模型。若把所有信息都塞进 prompt，权限、预算和版本字段就容易被自然语言中的其他内容覆盖。

### 3.2 部分可观察环境

Agent 通常不能直接看到完整环境。例如报销制度可能有多个版本，数据库里存在用户没有权限查看的字段，邮件发送服务的真实状态也可能在请求超时后不确定。因此，动作之后得到的是观察而不是完整真相：

~~~math
o_t\sim O(\mathord{\cdot}\mid s_{t-1},a_t)
~~~

`O` 表示环境的观察机制。它可能返回成功、失败、部分结果、超时或权限拒绝。模型不能把“没有看到失败”当成“动作一定成功”，尤其不能把超时后的写操作简单重试为两次发送。

### 3.3 动作和策略

动作可以是读取页面、查询数据库、调用计算器、生成草稿、追问用户、请求人工确认或结束任务。把第 `t` 步的动作写成：

~~~math
a_t\sim\pi_\theta(\mathord{\cdot}\mid g,s_{t-1},h_{t-1},\mathcal{T})
~~~

`h_{t-1}` 是已经记录的历史，`\mathcal{T}` 是可用工具集合，`\pi_\theta` 是由模型、规则和控制器共同实现的策略。这里的策略不一定是一个神经网络；一个确定性的规则分支也属于策略的一部分。

动作还应包含结构化信息：

~~~math
a_t=(u_t,n_t,\alpha_t,\rho_t)
~~~

`u_t` 表示动作类型，`n_t` 表示工具名称，`\alpha_t` 表示参数，`\rho_t` 表示风险级别或所需授权。只有文本中的“我准备发送邮件”没有足够信息，系统还需要知道发给谁、发什么、是否获得授权，以及是否可以重复执行。

### 3.4 状态转移

执行器得到允许动作并接收观察后，状态更新为：

~~~math
s_t=U(s_{t-1},a_t,o_t)
~~~

`U` 应由系统实现，而不是完全让模型在文本中声称“已经更新”。例如，工具返回“制度版本为 2026-07”后，版本字段应写入结构化状态；模型下一轮可以引用它，但不能通过普通文本把已确认版本改成另一个值。

### 3.5 轨迹

一次完整执行可以写成：

~~~math
\tau=(g,s_0,a_1,o_1,s_1,\ldots,a_K,o_K,s_K,\hat y)
~~~

`K` 是动作步数，`\hat y` 是最终对用户的输出。轨迹至少应能回答四个问题：系统当时知道什么，提出了什么动作，环境返回了什么，为什么继续或停止。没有这些记录，失败只能被粗略归因成“模型答错了”。

### 3.6 成功不是一个单独的文本分数

对只读的制度核对任务，可以把成功写成多个条件同时满足：

~~~math
\operatorname{Success}(\tau)=
\operatorname{TaskOK}(\tau)
\land\operatorname{EvidenceOK}(\tau)
\land\operatorname{PermissionOK}(\tau)
\land\operatorname{BudgetOK}(\tau)
~~~

如果任务允许修改文件，还要加入变更范围、测试结果和回滚状态。一个答案文字正确但引用了旧版本，不能算证据正确；一个金额计算正确但绕过了权限，也不能算系统成功。把这些条件拆开记录，才能知道改进应该发生在模型、工具、控制器还是业务规则层。

## 4. 一次 Agent loop 如何运行

### 4.1 解析目标和约束

第一步不是立刻生成工具调用，而是把用户语言转成目标、完成条件、禁止事项、可用工具和需要澄清的字段。对于报销助手，目标可能是“检查并提出建议”，禁止事项是“不要提交审批”，完成条件是“每个不合规项都有制度依据或明确标记为无法确认”。

目标解析不等于让模型自由重写用户请求。系统应保存原始请求，并把解析结果作为候选计划。若解析结果改变了风险等级或增加了写操作，必须回到规则和确认流程，而不是让模型自行扩大任务范围。

### 4.2 检查当前状态

控制器要读取当前版本、用户身份、已有结果、失败记录和预算。任务刚开始时状态可能为空；长任务恢复时，状态可能来自检查点而不是完整历史。系统需要区分“已确认的事实”“模型推测”“用户提供但尚未验证的信息”。把三者都写成普通文本，会让后续模型难以区分可信程度。

### 4.3 选择计划和下一步动作

Planner 可以生成一份多步计划，也可以只生成下一步动作。两者各有取舍。一次性计划便于展示全局结构，却可能在环境变化后过时；逐步决策适应性更强，却可能反复思考、忘记整体目标。实践中常用一个粗粒度计划加逐步重规划：计划只规定阶段和依赖，具体动作在获得观察后重新决定。

### 4.4 在执行前验证动作

执行前至少检查四类问题：工具和动作是否匹配，参数是否符合 schema，权限是否允许，预算和风险是否满足当前策略。对于写入动作，还要检查幂等键、目标资源版本和是否需要用户确认。

这一步的主体应是确定性代码或独立服务。模型可以解释为什么想调用工具，但不能用一段自然语言替代参数校验。尤其要把工具返回的文本视为不可信观察：网页内容、文档、邮件和代码注释都可能包含“请忽略此前规则”的内容，它们不应改变系统策略。

### 4.5 执行并记录结果

执行器只接受通过检查的动作，并记录请求、参数摘要、身份、版本、耗时、结果状态和错误类别。对于敏感数据，日志应遵循最小化原则，不要为了方便调试而永久保存全部原文。

读操作和写操作的执行策略不同。读操作失败通常可以重试或换来源；写操作可能已经产生部分副作用，即使客户端收到超时也不能假定没有发生。执行器应提供明确的状态查询、幂等键和可恢复语义。

### 4.6 处理观察结果

观察处理器要把外部结果转成结构化事件，例如 `document_found`、`missing_field`、`permission_denied`、`write_pending` 或 `test_failed`。模型可以读取事件的说明，但控制器应保留原始结果和可信来源，以便后续审计。

工具成功不代表任务成功。搜索工具成功返回了一堆结果，可能仍然没有找到当前版本；代码执行成功完成了命令，可能只是测试没有覆盖目标行为；数据库写入返回成功，可能仍需要查询确认最终状态。

### 4.7 更新状态并决定是否继续

状态更新应包括新事实、已解决问题、未解决问题、预算变化和下一步依赖。然后由停止函数决定四种结果之一：任务完成、需要继续、需要用户澄清、无法安全继续。

令 `I_complete` 表示完成条件已满足，`I_ask` 表示缺少用户才能提供的信息，`I_safe` 表示权限、预算和安全条件仍允许继续，`I_next` 表示存在可验证的下一步。停止函数可以写成：

~~~math
h(s_t)=
\begin{cases}
\mathrm{finish}, & I_{\mathrm{complete}}=1\\
\mathrm{ask}, & I_{\mathrm{ask}}=1\\
\mathrm{stop}, & I_{\mathrm{safe}}=0\\
\mathrm{continue}, & I_{\mathrm{next}}=1
\end{cases}
~~~

“继续”不应成为默认答案。若没有新的信息来源、没有剩余预算，或者每次重试都返回同一个错误，系统应停止并说明已完成部分，而不是生成更多看似努力的文本。

## 5. Agent 的核心组件

### 5.1 Goal：目标契约

目标应同时描述意图和可验收条件。例如“检查报销单”太宽泛；更可执行的目标是“对当前制度版本逐项检查报销单中的住宿和交通费用，给出每项的依据、计算结果和缺失资料，不提交审批”。

一个目标契约通常包含：输入、输出、成功条件、禁止事项、工具范围、预算、澄清条件和失败报告格式。目标越涉及外部副作用，禁止事项和确认条件越需要结构化，而不是只写在自然语言提示中。

### 5.2 State：任务状态

状态至少应能保存当前事实、计划、工具结果、错误、预算、权限和待确认事项。对一个代码修复任务，状态可能包括当前分支、测试结果、修改文件集合、未解决失败和最后一次补丁；对一个研究任务，状态可能包括问题分解、证据来源、引用映射和仍未核实的结论。

状态有两个常见错误。第一是只保存最近一轮对话，导致模型忘记早期约束。第二是保存了所有内容，却没有标记来源、时间和可信度，导致过期或冲突信息混入当前决策。结构化字段、版本号和状态差异比一份无限增长的文本日志更适合控制循环。

### 5.3 Policy 和 Planner：决定下一步

Policy 负责在当前状态中选择动作；Planner 负责提供阶段目标、依赖关系或候选路径。它们可以由同一个模型实现，也可以由规则、搜索器和模型共同实现。对于有明确规则的步骤，优先使用确定性逻辑；对于开放的检索和解释步骤，再让模型产生候选。

计划越详细不一定越好。计划粒度太粗，无法判断下一步是否可执行；粒度太细，环境一变就全部失效。有效的计划应记录依赖和完成条件，而不是预言每一次具体工具调用。

### 5.4 Tool Registry 和 Executor：描述与执行分离

Tool registry 描述工具名称、用途、输入 schema、输出 schema、风险等级、权限要求和超时策略。Executor 负责实际调用、参数检查、身份传递、错误归类、重试、幂等和日志。

把 registry 和 executor 分开很重要。模型可以看见工具描述，但不能因此获得执行权限。一个用户只有只读权限时，registry 可以不暴露写工具，executor 还应在服务端再次检查身份。即使模型生成了合法 JSON，也不能跳过服务端检查。

### 5.5 Observation Handler：把结果变成可用状态

观察处理器负责解析成功、失败、部分成功、超时、空结果和权限拒绝。它还要记录观察的来源、时间、版本和可信度。对检索结果，来源 URL 和文档版本很重要；对代码执行，环境镜像、依赖版本和测试命令很重要；对业务 API，request id 和最终状态查询很重要。

一个常见反模式是把工具返回的长文本原样拼回 prompt。这样会产生上下文膨胀，也可能把外部文本中的指令混入系统策略。更稳妥的做法是先解析为事件和字段，只将与当前决策有关的内容交给模型，同时保留原始结果供审计。

### 5.6 Memory：跨回合信息，而不是万能事实库

短期历史、当前 state、长期 memory 和 trace log 解决不同问题。当前 state 用于完成本次任务；短期历史帮助模型理解最近上下文；长期 memory 保存跨任务可能复用的偏好或项目知识；trace log 用于复盘，而不是直接当成事实注入下一次 prompt。

长期 memory 需要过期、删除、权限和来源管理。用户曾经说过“我喜欢经济舱”，不代表在所有公司政策和所有差旅场景都可以自动使用；一年前的项目配置也不应覆盖当前仓库中的事实。记忆越持久，越需要时间戳、作用域、访问控制和用户可见的修改机制。

### 5.7 Controller：系统的控制平面

Controller 管理最大步数、token、工具调用次数、延迟、费用、重试、权限、停止条件、人工接管和日志。它不是一个装饰性的 wrapper，而是把模型建议变成可执行动作之前的主要控制层。

没有 controller，常见后果包括重复搜索、无限重试、使用过期状态、修改无关文件、把工具输出当成上级指令、在预算耗尽后继续生成，以及最终答案与真实执行状态不一致。换更强的模型可能改善动作质量，却不能替代权限服务、幂等执行器和审计日志。

### 5.8 Logger 和 Trace：让失败可以重放

一条有用的 trace 不只保存最终答案，还要关联任务 ID、状态版本、模型版本、提示摘要、动作、参数摘要、工具结果、权限结果、耗时、预算变化和最终状态。敏感内容应按业务要求脱敏或分级保存。

可重放不等于无条件重复执行。只读工具通常可以重放；写工具需要模拟执行、幂等键或状态查询。日志系统必须区分“重新计算一遍”和“再次产生副作用”，否则调试本身就可能造成事故。

## 6. 上下文、预算和长周期任务

### 6.1 上下文不是无限工作记忆

一次模型调用可见的上下文长度可以粗略拆为：

~~~math
L_t=L_g+L_s+L_h+L_{\mathrm{tools}}+L_{\mathrm{obs}}+L_{\mathrm{output}}
~~~

其中 `L_g` 是目标和约束，`L_s` 是状态摘要，`L_h` 是历史，`L_{\mathrm{tools}}` 是工具描述，`L_{\mathrm{obs}}` 是观察结果，`L_{\mathrm{output}}` 是本轮输出预算。即使 runtime 接受很长的上下文，也不代表模型可以稳定使用其中每个字段。重复历史、长网页和无关工具描述会挤压真正重要的状态。

### 6.2 预算是一个向量

Agent 的资源不能只用 token 表示。可以记录：

~~~math
B_t=(b_{\mathrm{step}},b_{\mathrm{tok}},b_{\mathrm{tool}},b_{\mathrm{time}},b_{\mathrm{cost}})
~~~

简单任务可能受延迟约束，代码修复可能受执行次数约束，研究任务可能受检索和人工核验成本约束，高风险写操作还受到确认次数和权限范围约束。控制器应在每个循环更新预算，而不是只在最后检查总 token。

### 6.3 上下文压缩和检查点

长任务需要把状态摘要、原始证据和操作日志分开。摘要可以告诉模型“已经完成哪些步骤”，但不能替代关键证据；检查点可以帮助系统恢复，但不能让恢复动作重复产生外部副作用。一个可靠的 checkpoint 至少包括状态版本、已确认事实、待解决事项、工具结果引用和最后一个可查询的外部状态。

上下文折叠也有信息损失风险。若把“用户明确禁止发信”压缩掉，模型下一轮可能错误扩大权限；若把多个合同版本压缩为“已有合同”，引用就失去时间边界。因此压缩策略应对约束、权限、版本和未解决风险设置更高保留优先级。

### 6.4 长周期任务需要三个共同体

长周期 Agent 的能力不能只归因于模型。至少要区分：

1. 模型：理解目标、产生计划、生成动作和利用观察的能力。
2. 环境：文件、浏览器、终端、数据库或模拟世界是否提供真实反馈。
3. Harness：是否有工作区、权限、checkpoint、memory、重试、trace 和恢复机制。

公开资料中的 AgentWorld、长时间代码任务和多 Agent 协作系统经常把这三者一起发布。读者在比较结果时，应先确认评测固定了哪些环境和 harness；如果一个系统更换了工具包装器、上下文策略或恢复机制，分数变化就不能全部归因于基础模型。

在一个简化且近似独立的模型中，长任务成功概率可以写成：

~~~math
P_{\mathrm{success}}\approx
P_{\mathrm{plan}}\cdot
P_{\mathrm{tool}}\cdot
P_{\mathrm{state}}\cdot
P_{\mathrm{recover}}\cdot
P_{\mathrm{permission}}
~~~

这个式子不是生产系统的统计定律。真实环节往往相关，成功概率也会随任务长度变化。它的教学价值在于提醒我们：模型计划能力提高后，状态恢复或权限执行仍可能成为瓶颈。

### 6.5 Multi-Agent 不是免费并行

把任务拆给多个 Agent 可以获得并行探索和角色分工，但也会引入上下文同步、证据合并、重复工作、冲突解决和最终验证成本。多个 Agent 同时得到同一个错误资料时，数量增加不会自动增加独立性；如果没有共享状态版本，协作者还可能基于过期事实行动。

多 Agent 适合拆分相互独立、结果可合并、验证成本可控的子任务。若任务有强顺序依赖或高风险副作用，单一控制器配合受限工具往往更容易审计。并行度应由质量增益、协调开销、延迟和风险共同决定，而不是由“能启动多少个子 Agent”决定。

## 7. 什么时候该用 Agent

### 7.1 适合的任务

Agent 更适合下面几类任务：

1. 任务确实需要多个步骤。
2. 下一步依赖外部反馈。
3. 有明确的完成标准或可验证结果。
4. 工具权限可以被细分和审计。
5. 失败后可以重试、追问或交给人工。
6. 灵活性带来的价值大于额外成本和延迟。

代码修复、复杂资料调查、带证据的运营排查和多阶段数据处理通常符合这些条件。关键判断不是任务听起来多复杂，而是“看到反馈后改变下一步”是否构成任务价值。

### 7.2 不适合的任务

简单分类、固定格式转换、一次检索问答和已有稳定 workflow 覆盖的任务，通常不需要开放式 Agent。若任务要求极低延迟，多个模型回合和工具调用可能反而降低体验。对于不可逆、高风险且无法人工复核的动作，也不应因为 Agent 看起来聪明就开放自主执行。

如果一次模型调用加一个确定性函数已经稳定解决问题，增加 Agent loop 只会增加故障面。工程选择应从最小可行的自动化开始，再根据真实失败案例增加状态、工具或动态规划，而不是默认把所有应用升级成 Agent。

### 7.3 混合架构通常更现实

一个成熟的混合架构可以这样分工：

- workflow 固定身份、审批、日志和不可违反的顺序；
- RAG 提供有版本和引用的证据；
- Agent 在检索、计算、追问等低副作用步骤中动态选择；
- 规则和权限服务检查所有写入动作；
- 人工在高风险或证据冲突时做最终决定。

这种设计牺牲了一部分自由度，换来可解释的责任边界。Agent 越接近外部副作用，越应该缩小自由动作空间，而不是把更多决策都交给模型。

## 8. 失败模式：从“答错”定位到系统层

### 8.1 目标理解错误

模型把“检查并提出建议”理解成“直接修改并提交”，属于目标或约束解析失败。修复方法是保存原始目标、结构化禁止事项、在高风险动作前展示预览，并用目标变体测试模型是否稳定区分“查看”和“执行”。

### 8.2 工具选择错误

模型需要计算却继续搜索，需要查当前版本却调用了旧知识库。此时增加工具数量通常会让问题更糟。应缩小候选工具集合、改善工具描述、记录选择理由，并按任务切片评估选择准确率。

### 8.3 参数错误

工具名称选对并不代表参数正确。缺少订单号、日期格式不合法、目标资源不属于当前用户，都应在执行前被结构化校验。参数错误可以由模型修复，也可以触发追问，但不能把异常字符串原样交给下一次动作选择。

### 8.4 忽略 observation

工具返回权限不足，Agent 却继续假设查询成功；测试返回失败，Agent 却直接报告“已修复”。这通常是 observation schema、状态更新或训练反馈的问题。评估时应检查观察是否改变了状态或后续动作，而不只检查最终答案。

### 8.5 无限循环和过度重试

相同工具、相同参数和相同错误重复出现，说明系统没有识别无进展状态。控制器应记录动作和错误指纹，设置有限重试、退避、替代路径和最终停止原因。重试不是恢复本身，只有当新尝试改变了信息或执行条件时才有价值。

### 8.6 状态污染和旧记忆误用

把上一个用户的项目名、过期制度或另一个仓库的测试结果带入当前任务，会造成跨任务污染。状态必须有任务作用域，memory 必须有来源、时间和权限，恢复时必须验证外部资源版本。

### 8.7 工具结果注入

网页、邮件、文档和代码注释中的文字可能包含面向模型的指令。它们是数据，不是系统策略。观察处理器应将外部内容与系统消息、用户约束和权限规则分层；模型即使读到了“忽略前面的限制”，也不能因此获得新权限。

### 8.8 部分副作用和重复执行

支付、发信、删除和部署等动作可能在客户端超时前已经成功。盲目重试会造成重复副作用。安全执行需要幂等键、状态查询、明确的 pending 状态、预览和回滚方案。对于无法可靠查询状态的外部系统，应默认停下并交给人工，而不是猜测。

### 8.9 最终文字与真实状态不一致

Agent 说“测试全部通过”，但日志显示只运行了公开测试；Agent 说“邮件已发送”，但 API 只返回了排队状态。最终输出应从结构化执行结果生成，或至少经过状态一致性检查。语言流畅不能替代事实来源。

## 9. 如何评估一个 Agent

### 9.1 任务成功率

任务成功率是必要指标，但必须先定义成功条件。对 `N` 个任务，要求 `N` 是正整数：

~~~math
A_{\mathrm{task}}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[z_i=1]
~~~

`z_i` 不是“模型输出非空”，而是该任务是否满足预先定义的完成条件。报销任务可能要求所有关键字段有证据；代码任务可能要求隐藏测试通过且补丁范围合规；只读查询任务可能只要求引用和答案一致。

### 9.2 工具选择和参数

若有 `M` 个被评估的动作决策，要求 `M` 是正整数；工具选择准确率可以写成：

~~~math
A_{\mathrm{tool}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\hat t_j=t_j^*]
~~~

参数合法率则应至少区分 schema 合法和语义正确：

~~~math
R_{\mathrm{arg}}=\frac{1}{M}\sum_{j=1}^{M}\mathbf{1}[\operatorname{schema}(\hat p_j)\land\operatorname{meaning}(\hat p_j)]
~~~

一个参数可以通过 JSON schema，却指向错误用户或错误日期。只报告格式合法率会高估真实能力。

### 9.3 执行、观察和状态更新

工具调用成功率反映执行环境和参数质量，但不是任务完成率。若有 `M>0` 个动作决策，还应测量观察使用率：

~~~math
U_{\mathrm{obs}}=\frac{1}{M}\sum_{j=1}^{M}I_{\mathrm{obs}}(o_j)
~~~

当观察确实改变后续状态或动作时，`I_obs` 取 1，否则取 0。`I_over(B_i)` 在第 `i` 个任务的步数、token 或时间超过上限时取 1。

状态更新覆盖率可以检查每个需要写入的字段是否由系统记录。若工具返回新版本号但状态仍保留旧版本，后续决策即使文字正确也不可信。

### 9.4 停止和资源

停止正确率要同时包含“该停时停”和“不能停时继续”。过早结束会丢失任务，过晚结束会浪费预算。对 `N>0` 个任务，资源指标可以写成：

~~~math
R_{\mathrm{over}}=\frac{1}{N}\sum_{i=1}^{N}I_{\mathrm{over}}(B_i)
~~~

`B_i` 是第 `i` 个任务的实际资源向量，`I_{\mathrm{over}}(B_i)` 在它超过预先定义的步数、token 或时间上限时取 1，否则取 0。产品报告还应给出 P50、P95 和失败任务的资源分布，因为平均延迟可能掩盖少量极慢任务。

### 9.5 权限和轨迹完整性

未授权动作率应按动作计数，并区分“被系统拒绝的尝试”和“已经产生副作用的成功越权”。前者仍然是模型或路由问题，后者是严重的系统安全事件。

轨迹完整性不是保存越多文本越好，而是保存足够的结构化证据：目标版本、动作、参数摘要、执行结果、状态差异、权限判断、预算变化和最终状态。缺少这些字段的任务不能被可靠复盘。

### 9.6 端到端指标不能替代分层指标

一个 Agent 可能任务成功率很高，但工具选择错误率也很高，因为错误工具碰巧返回了相同答案；也可能任务成功率暂时不变，但状态更新、停止和安全指标明显改善，为更长任务提供了基础。因此要同时报告结果、轨迹、资源和安全切片。

评估集还应包含原任务、变体、缺字段、工具超时、权限拒绝、过期数据、恶意观察和需要人工确认的任务。只测“工具都正常、目标都清楚”的 happy path，无法说明 Agent 是否真的具备恢复能力。

### 9.7 长周期评估的环境绑定

研究和产品报告必须写清模型版本、system prompt、工具描述、工具实现、环境镜像、memory 策略、上下文压缩、重试策略、预算和人工介入。SWE 类任务换一个测试 harness，浏览器任务换一个网站状态，企业助手换一个权限配置，结果都可能显著变化。

## 10. 成本和延迟：灵活性需要付费

一次 Agent 请求的教学成本模型可以写成：

~~~math
C=c_{\mathrm{tok}}T+c_{\mathrm{tool}}U+c_{\mathrm{time}}L+c_{\mathrm{human}}H
~~~

`T` 是 token，`U` 是工具调用次数，`L` 是计算或占用时间，`H` 是人工复核工作量，`c` 是对应换算系数。实际系统还应单独记录 GPU 秒、外部 API 费用、存储、日志和失败重试成本，不要把不同资源粗暴地合成一个无法解释的数字。

串行调用的延迟大致是各阶段延迟之和：

~~~math
L_{\mathrm{serial}}\approx\sum_{k=1}^{K}(L_{\mathrm{model},k}+L_{\mathrm{tool},k})
~~~

如果多个相互独立的只读任务可以安全并行，延迟更接近最大分支耗时加协调开销：

~~~math
L_{\mathrm{parallel}}\approx\max_k L_k+L_{\mathrm{coord}}
~~~

并行并不总是更快。共享状态、限流、冲突写入和结果合并会增加协调成本；对高风险动作，串行确认反而更安全。比较 Agent 策略时，应同时报告质量、成本、P95 延迟和副作用风险。

## 11. 安全边界：模型建议不能直接拥有权力

### 11.1 最小权限和动作分级

工具可以按只读、低风险写入、高风险写入和不可自动执行分级。查询制度、读取测试日志和计算金额通常是低副作用动作；发邮件、删除文件、提交付款和修改权限则需要更严格的参数、身份、确认和审计。

权限判断至少要依赖用户身份、资源归属、动作类型、当前目标和资源版本：

~~~math
\operatorname{Allowed}(a,s)=
\operatorname{IdentityOK}(s)\land
\operatorname{ScopeOK}(a,s)\land
\operatorname{PolicyOK}(a,s)\land
\operatorname{BudgetOK}(a,s)
~~~

这个判断应在服务端执行。模型输出的 JSON、解释或自报身份都不能作为唯一授权依据。

### 11.2 预览、确认和幂等

对不可逆动作，系统应先生成预览，展示目标、参数、影响范围和证据，再由有权限的主体确认。执行器应使用幂等键和状态查询，避免网络超时导致重复操作。确认也应绑定具体参数和资源版本，不能让用户确认“发送邮件”后模型偷偷替换收件人。

### 11.3 不可信观察与提示注入

外部文档、网页、邮件和代码都是数据源，不是控制策略。系统提示、开发者约束、用户目标、工具 schema 和外部观察应有清晰的优先级和边界。防御不应只依赖一句“不要被注入”，还要限制工具权限、分离读写、校验参数、记录来源，并在高风险动作前重新确认。

### 11.4 沙箱、人工接管和事故记录

代码执行和浏览器操作应在隔离环境中进行，限制网络、文件、进程、时间和资源。高风险任务需要人工接管，人工看到的应是经过整理的证据和待确认动作，而不是一大段无法核对的模型思考文本。事故处理需要保留版本、轨迹、工具结果和实际外部影响，以便回滚和修复。

## 12. 一个可运行的 Agent trace 审计实验

下面的程序不连接真实工具，也不代表任何产品的上线标准。它构造四条 toy trace，检查任务结果、工具选择、参数、观察使用、状态更新、预算、权限、停止和日志完整性。程序故意包含失败案例，让读者看到“代码可以正常运行”和“系统质量达标”是两件事。

~~~python
from collections import Counter


LIMITS = {"steps": 4, "tokens": 4000, "latency_ms": 10000}

TRACES = [
    {
        "id": "repo_fix",
        "goal": "run tests, make the smallest fix, and report the patch",
        "success": True,
        "tokens": 1800,
        "latency_ms": 8200,
        "stop_correct": True,
        "final": "fixed failing parser test",
        "steps": [
            {"tool": "read_file", "expected": "read_file", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "run_tests", "expected": "run_tests", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "edit_file", "expected": "edit_file", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "run_tests", "expected": "run_tests", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "policy_answer",
        "goal": "answer with citations from the policy documents",
        "success": True,
        "tokens": 1250,
        "latency_ms": 3100,
        "stop_correct": True,
        "final": "grounded answer with one citation",
        "steps": [
            {"tool": "search_docs", "expected": "search_docs", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
            {"tool": "quote_answer", "expected": "quote_answer", "args_valid": True, "allowed": True,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "calendar_overreach",
        "goal": "check availability, but do not send messages",
        "success": False,
        "tokens": 900,
        "latency_ms": 2100,
        "stop_correct": True,
        "final": "blocked unauthorized send action",
        "steps": [
            {"tool": "email_send", "expected": "calendar_lookup", "args_valid": True, "allowed": False,
             "observation_used": True, "state_updated": True},
        ],
    },
    {
        "id": "looping_data_task",
        "goal": "compute a small aggregate and stop",
        "success": False,
        "tokens": 5200,
        "latency_ms": 15100,
        "stop_correct": False,
        "final": "kept searching instead of computing",
        "steps": [
            {"tool": "search_docs", "expected": "calculator", "args_valid": False, "allowed": True,
             "observation_used": False, "state_updated": False},
        ],
    },
]


def rate(good, total):
    if total < 0 or good < 0 or good > total:
        raise ValueError("rate counts must satisfy 0 <= good <= total")
    return None if total == 0 else round(good / total, 3)


def at_least(value, threshold):
    return value is not None and value >= threshold


def trace_complete(trace):
    required_trace_fields = {
        "id", "goal", "steps", "success", "tokens", "latency_ms", "final",
    }
    required_step_fields = {
        "tool", "expected", "args_valid", "allowed",
        "observation_used", "state_updated",
    }
    if not required_trace_fields.issubset(trace):
        return False
    return all(required_step_fields.issubset(step) for step in trace["steps"])


def over_budget(trace):
    return (
        len(trace["steps"]) > LIMITS["steps"]
        or trace["tokens"] > LIMITS["tokens"]
        or trace["latency_ms"] > LIMITS["latency_ms"]
    )


def audit_agent_traces(traces):
    all_steps = [step for trace in traces for step in trace["steps"]]
    total_steps = len(all_steps)

    metrics = {
        "task_success_rate": rate(sum(t["success"] for t in traces), len(traces)),
        "tool_selection_accuracy": rate(
            sum(s["tool"] == s["expected"] for s in all_steps), total_steps
        ),
        "arg_valid_rate": rate(sum(s["args_valid"] for s in all_steps), total_steps),
        "observation_use_rate": rate(
            sum(s["observation_used"] for s in all_steps), total_steps
        ),
        "state_update_coverage": rate(
            sum(s["state_updated"] for s in all_steps), total_steps
        ),
        "budget_overrun_rate": rate(sum(over_budget(t) for t in traces), len(traces)),
        "unauthorized_action_rate": rate(
            sum(not s["allowed"] for s in all_steps), total_steps
        ),
        "stop_correct_rate": rate(sum(t["stop_correct"] for t in traces), len(traces)),
        "trace_completeness": rate(
            sum(trace_complete(t) for t in traces), len(traces)
        ),
    }

    bad_traces = []
    reasons = {}
    for trace in traces:
        flags = []
        if not trace["success"]:
            flags.append("task_failed")
        if over_budget(trace):
            flags.append("budget_overrun")
        if not trace["stop_correct"]:
            flags.append("bad_stop")
        if any(step["tool"] != step["expected"] for step in trace["steps"]):
            flags.append("wrong_tool")
        if any(not step["allowed"] for step in trace["steps"]):
            flags.append("unauthorized_action")
        if any(not step["observation_used"] for step in trace["steps"]):
            flags.append("observation_ignored")
        if flags:
            bad_traces.append(trace["id"])
            reasons[trace["id"]] = flags

    checks = {
        "task_success_ok": at_least(metrics["task_success_rate"], 0.75),
        "tool_selection_ok": at_least(metrics["tool_selection_accuracy"], 0.80),
        "args_ok": at_least(metrics["arg_valid_rate"], 0.95),
        "observation_use_ok": at_least(metrics["observation_use_rate"], 0.90),
        "state_update_ok": at_least(metrics["state_update_coverage"], 0.90),
        "budget_ok": metrics["budget_overrun_rate"] == 0.0,
        "permission_ok": metrics["unauthorized_action_rate"] == 0.0,
        "stop_ok": at_least(metrics["stop_correct_rate"], 0.90),
        "trace_complete_ok": metrics["trace_completeness"] == 1.0,
    }
    return {
        "metrics": metrics,
        "bad_traces": bad_traces,
        "top_failure_reasons": Counter(
            flag for flags in reasons.values() for flag in flags
        ).most_common(),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


report = audit_agent_traces(TRACES)
print("metrics=", report["metrics"])
print("bad_traces=", report["bad_traces"])
print("top_failure_reasons=", report["top_failure_reasons"])
print("checks=", report["checks"])
print("all_checks_pass=", report["all_checks_pass"])
~~~

预期输出如下：

~~~text
metrics= {'task_success_rate': 0.5, 'tool_selection_accuracy': 0.75, 'arg_valid_rate': 0.875, 'observation_use_rate': 0.875, 'state_update_coverage': 0.875, 'budget_overrun_rate': 0.25, 'unauthorized_action_rate': 0.125, 'stop_correct_rate': 0.75, 'trace_completeness': 1.0}
bad_traces= ['calendar_overreach', 'looping_data_task']
top_failure_reasons= [('task_failed', 2), ('wrong_tool', 2), ('unauthorized_action', 1), ('budget_overrun', 1), ('bad_stop', 1), ('observation_ignored', 1)]
checks= {'task_success_ok': False, 'tool_selection_ok': False, 'args_ok': False, 'observation_use_ok': False, 'state_update_ok': False, 'budget_ok': False, 'permission_ok': False, 'stop_ok': False, 'trace_complete_ok': True}
all_checks_pass= False
~~~

### 12.1 读懂第一组指标

四条 trace 中有两条任务成功，因此任务成功率是 `0.5`。所有步骤中有三次工具选择与期望一致，工具选择准确率为 `0.75`。`calendar_overreach` 的动作名称和期望不同，`looping_data_task` 也把本应使用的计算器换成了检索工具，所以错误被轨迹指标暴露出来。

参数合法率为 `0.875`，因为最后一条 trace 的参数无效。注意，演示中的布尔字段已经由人工构造；真实评估必须让 schema 验证器和业务语义检查器产生这些标签，并保存失败样本以检查标签质量。

### 12.2 读懂第二组指标

`calendar_overreach` 的工具调用被标记为不允许。虽然这个 toy trace 没有真的发送邮件，但未授权尝试本身就说明策略或工具选择有问题；如果真实执行器没有再次检查权限，风险会从“被拒绝的错误动作”升级为外部副作用。

`looping_data_task` 超过了 token 和延迟限制，没有使用观察，也没有更新状态。它提醒我们：一个 Agent 可能一直在做看似合理的检索，却没有向完成条件靠近。控制器应当把无进展动作、预算变化和停止原因记录下来。

### 12.3 为什么完整日志仍然不代表质量合格

四条 trace 的必需字段都存在，因此 `trace_complete_ok` 为真；但其他质量条件都没有满足，最终 `all_checks_pass` 为假。日志完整只能说明我们有机会分析问题，不能把错误变成成功。生产系统也应避免把“日志存在”当成“动作安全”。

## 13. 练习：把概念变成设计判断

### 练习一：画出任务状态

为“检查报销单但不提交审批”定义 `s_0`。分别列出目标、制度版本、报销字段、权限、预算、缺失信息和停止条件。说明哪些字段可以由模型提出，哪些字段必须由系统或业务服务确认。

### 练习二：区分四种系统

将下面三个应用分别归入 Chat Model、RAG、Workflow、Agent 或混合架构，并说明依据：

1. 根据一份固定 FAQ 回答用户问题。
2. 检索三份制度，逐条核对报销单并在缺字段时追问。
3. 每天凌晨固定执行数据抽取、校验和入库。

不要只看产品名称，要指出下一步是否依赖外部观察，以及是否产生外部副作用。

### 练习三：分析一次错误循环

设计一条连续三次调用同一搜索工具并得到空结果的轨迹。标出每一步的状态、观察、错误指纹和剩余预算，说明控制器应该何时停止、何时换工具、何时向用户澄清。

### 练习四：评估观察使用

构造两条最终答案相同的 trace：一条真正使用工具返回的版本号，另一条虽然调用了工具却一直使用模型记忆中的旧版本。说明为什么只比较最终答案不足以区分它们。

### 练习五：设计安全工具

为“发送通知邮件”设计工具契约，至少包括收件人、主题、正文、身份、幂等键、预览状态、确认状态、超时和最终状态查询。列出哪些来自网页或邮件正文的内容不能改变发送权限。

### 练习六：修改审计数据

将 demo 中的 `looping_data_task` 改为正确调用计算器、使用观察并在预算内停止。重新运行程序，观察哪些指标改善，哪些指标仍可能受另外一条失败 trace 影响。说明为什么修复一条样本不能证明整个 Agent 可靠。

## 14. 本章小结

Agent 的核心不是“模型加工具”，而是一个围绕目标运行的受控闭环：系统保存状态，策略提出动作，执行器在权限和预算约束下执行，环境返回观察，状态更新器写回事实，停止函数决定继续、追问、结束或安全停止，日志记录整个过程。

Chat Model 主要负责生成回答，RAG 主要负责获取证据，Workflow 主要负责执行预定义流程，Agent 则在外部反馈改变下一步时发挥价值。生产系统往往把它们组合起来，用 workflow 和规则限制高风险边界，把 Agent 的灵活性留给真正需要探索的局部步骤。

可靠性需要分层观察：任务是否完成，工具和参数是否正确，观察是否被使用，状态是否更新，预算是否受控，权限是否正确，停止是否合理，轨迹是否足以复盘。最终文字正确不能抵消越权动作，工具调用成功不能抵消证据错误，日志完整也不能抵消任务失败。

当任务进入长周期、多工具或多 Agent 场景时，模型、环境和 harness 必须一起评估。上下文长度、工具数量和并行度都不是能力本身；只有在新任务、故障反馈、权限限制和资源预算下仍能稳定完成目标，系统才真正具备可用的 Agent 能力。

下一章将从工具 schema 和 function calling 开始，具体讨论如何把动作表示成可验证的调用、如何处理参数和权限、以及工具返回结果如何安全地进入后续状态。

## 15. 延伸资料与证据边界

1. ReAct: Synergizing Reasoning and Acting in Language Models，论文：<https://arxiv.org/abs/2210.03629>。
2. Toolformer: Language Models Can Teach Themselves to Use Tools，论文：<https://arxiv.org/abs/2302.04761>。
3. MRKL Systems: A Modular, Neuro-Symbolic Architecture That Combines Large Language Models, External Knowledge Sources and Discrete Reasoning，论文：<https://arxiv.org/abs/2205.00445>。
4. OpenAI Agents SDK 文档入口：<https://openai.github.io/openai-agents-python/>。
5. NIST AI 600-1 Generative AI Profile：<https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence>。
6. OWASP Top 10 for LLM Applications：<https://genai.owasp.org/llm-top-10/>。

前 3 项主要支撑 Agent、工具使用和模块化推理的研究背景；第 4 项支撑公开 SDK 中的工具、handoff、guardrail 和 tracing 概念；后两项支撑风险管理和应用安全的通用边界。它们不构成任何闭源产品内部实现或目标业务上线结果的证明。真实系统还需要用自身的模型版本、工具实现、权限配置、数据集和故障样本独立验证。
