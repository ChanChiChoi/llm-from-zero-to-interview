# 第七章：Agent 产品落地

## 0. 本章范围与资料

本章讨论一个比“模型会不会调用工具”更具体的问题：怎样把能够观察环境、选择工具、执行动作并根据结果继续推进的 Agent，建设成用户敢用、系统管得住、结果验得出、成本算得清的产品。正文会沿着任务选择、自动化程度、工具权限、人工确认、失败恢复、状态持久化、审计、体验、成本、工作流组合和评估展开。

联网资料主要用于核对通用工程边界，而不是替任何供应商的产品宣传背书。OpenAI Agents SDK 的 tools、guardrails 和 tracing 文档可支持工具、护栏和轨迹需要显式工程化的讨论；OpenAI Evals 文档可支持用可复现评估和 trace 检查任务与工具行为；OpenAI Model Spec 可支持指令层级和不可信工具输出的边界；OWASP LLM Top 10 可支持提示注入、过度代理、敏感信息泄露和资源消耗的风险分类；Anthropic 关于 workflows 与 agents 的工程文章可支持“稳定流程和受控自主性需要组合”的设计判断。

这些资料不能证明某个 Agent 已经安全、可靠或适合你的业务。本章的阈值、场景数据和 Python 审计程序都是教学构造；真实系统仍需结合身份权限、工具契约、业务验收、审计样本、真实成本和风险评估。

本章不替代第十七册 Agent 原理、工具协议、安全评估和多 Agent 章节，也不展开具体框架 API。这里聚焦产品落地：怎么把 Agent 从“会调用工具的 demo”升级为可控、可验证、可审计、可预算、可人审、可进入企业工作流的任务执行产品。

Agent 产品和普通大模型问答产品最大的区别，是 Agent 会执行动作。它可能查询系统、修改文件、提交表单、调用 API、操作浏览器、运行代码或触发业务流程。因此 Agent 产品化的核心不是让模型“更自主”，而是让它在受控范围内可靠执行任务。

本章系统讲 Agent 产品落地：适合 Agent 的任务、工具权限、人工确认、失败恢复、审计日志、成本控制、用户体验、评估指标和常见失败模式。

## 7.1 Agent 产品的本质

Agent 产品不是聊天框加几个工具，而是一个有输入、有状态、有动作、有验收标准的任务执行系统。普通问答的主要输出是文本，错误通常在用户读到文本时暴露；Agent 的输出还包括查询、写入、发送、提交、删除和触发流程，错误可能在用户意识到之前已经改变了外部世界。因此，产品设计的对象不只是模型，还包括执行器、权限系统、状态存储、观察接口、确认界面、回滚机制和审计系统。

一次任务通常包含以下链路：

1. 用户目标：用户想完成什么，成功和失败分别是什么。
2. 任务契约：允许访问哪些资源，禁止哪些动作，预算和截止时间是多少。
3. 计划或工作流：把目标拆成可观察、可停止的步骤。
4. 工具选择：根据当前状态选择查询、计算、写入或外部操作工具。
5. 执行动作：由受控执行器校验参数、权限、风险和幂等性后执行。
6. 观察反馈：读取工具结果、错误、状态变化和外部系统确认。
7. 状态更新：记录完成步骤、未决动作、失败原因和下一步约束。
8. 验收交付：检查结果是否满足业务标准，再向用户报告完成、部分完成或未完成。

因此，“模型能生成正确的工具调用”只是链路中的一小部分。即使单次调用成功，任务也可能因为调用了错误的租户、使用了过期状态、重复扣款、漏掉验收或没有记录证据而失败。

可以用一个简单的产品分界来理解三类系统：

- 聊天产品主要回答问题，动作边界很窄。
- Workflow 产品按预先定义的步骤运行，分支由规则、校验器或少量模型判断决定。
- Agent 产品允许模型在运行时选择步骤、工具和下一步，但必须仍受任务契约、权限、预算和验收器约束。

并不是所有多步任务都需要 Agent。能用明确规则表达的流程，通常应先用 workflow；只有当输入变化大、需要根据观察结果动态选择路径，且这种灵活性带来的收益大于验证成本时，才值得引入 Agent。

## 7.2 哪些任务适合 Agent

“需要多步操作”并不自动意味着适合 Agent。更稳妥的判断至少要同时看四个维度：

1. **路径变化**：不同输入是否需要不同步骤，且变化不能完全由固定规则覆盖。
2. **观察依赖**：下一步是否必须根据前一步返回的状态、证据或错误动态决定。
3. **结果可验收**：任务是否有可执行的完成标准，例如字段一致、测试通过、金额匹配或审批记录存在。
4. **风险可控制**：失败是否能被发现、暂停、撤销或转交人工，权限和资源边界是否清楚。

客服工单补全、内部代码修复、报表草稿和知识检索通常满足这些条件：它们有一定路径变化，结果可以抽查，早期还可以保留人工确认。直接扣款、删除生产数据、修改全局权限等任务即使需要多步判断，也不应因为“模型很聪明”就交给全自动 Agent。它们需要更强的规则、双人审批、事务系统或专门的控制面。

一个实用的选择顺序是：先问能否用确定性 API 或 workflow 完成；如果不能，再问 Agent 的灵活性是否会明显减少人工工作；最后问这种收益能否覆盖评估、监控、权限和恢复的新增成本。若目标模糊、结果无法验证、延迟必须毫秒级、外部状态不可观测或动作不可逆，通常不适合把 Agent 放在自动执行位置。

例如“每周从三个固定系统拉数据并生成同一格式报表”更像 workflow；“根据用户自然语言解释异常、查询不同系统、选择补充数据并形成调查草稿”才可能适合受控 Agent。两者都可以使用模型，但模型承担的自由度和责任不同。

## 7.3 Agent 产品不要一开始全自动

自动化层级是产品和风险决策，不是模型能力的自然结果。一个模型在离线任务上成功率很高，不代表它已经拥有生产系统的授权；真实生产还要考虑输入分布变化、工具故障、权限变化、重复执行、用户误解和无法撤销的副作用。

常见的自动化路径是：

1. **建议模式**：Agent 只给出判断或下一步建议，不执行动作。
2. **草稿模式**：Agent 生成草稿、计划和参数，用户检查后执行。
3. **半自动模式**：低风险查询和可撤销写入自动执行，高风险步骤暂停确认。
4. **审批模式**：Agent 准备完整动作包，由有责任的负责人审批。
5. **自动模式**：仅对边界清楚、权限最小、结果可验收、失败可恢复的任务开放。

升级自动化前，至少要回答五个问题：任务成功率是否在固定版本的评估集上稳定；高风险动作是否有准确的分类；工具和数据权限是否按任务收缩；异常时是否能停止并知道外部状态；用户和审计人员能否回看 Agent 做过什么。任何一项没有可靠证据，都应保留在较低自动化层级，而不是用一个总体平均分替代。

产品界面也要把层级写进任务契约和审计记录，例如 `suggest`、`draft`、`semi_auto`、`approval` 和 `auto`。同一个 Agent 可以在不同租户、工具或风险等级下处于不同层级，不能只在产品首页写一个笼统的“自动化”。

## 7.4 工具权限设计

工具不是模型能力的延伸，而是系统边界的一部分。模型可以提出“读取客户订单”或“发送邮件”的调用请求，但真正决定是否执行的，必须是模型之外的执行器。执行器要重新验证用户身份、租户、资源范围、动作类型、参数、当前状态和风险等级；不能因为模型在上一轮已经说“用户授权了”就跳过检查。

工具至少可以按副作用分成四层：

1. 只读工具：搜索、查询、读取文档或获取状态。
2. 可撤销写入：生成草稿、创建临时文件、添加待处理标签。
3. 业务写入：修改内部记录、提交工单、更新配置或触发审批。
4. 高风险动作：删除、支付、对外发送、修改权限、改变生产系统。

这四层不是简单的 UI 标签，而要对应不同的执行策略。只读工具仍要做数据权限过滤；可撤销写入要有所有者、过期时间和撤销接口；业务写入要有幂等键和结果校验；高风险动作通常还要有明确确认、审批或双人控制。

一次工具授权至少包含三种约束：

1. **用户权限**：当前主体能否访问资源或执行动作。
2. **任务权限**：这个工具和资源是否属于当前任务契约的范围。
3. **风险权限**：动作是否会外发、删除、改权限、改生产数据或产生费用。

例如用户有权访问整个客户系统，并不意味着“整理本周工单”的 Agent 可以修改任意客户记录。用户授权是必要条件，不是任务授权的充分条件。工程上可把任务权限编码为短期 capability：限定工具名、资源集合、动作集合、租户、过期时间和审计 ID，并在每次调用时重新验证。

参数校验也必须在工具层完成。工具不能接受一个看似合法但实际指向任意路径的字符串，不能把模型生成的金额、收件人、SQL、文件路径或权限集合直接传给下游系统。生产执行器应使用 allowlist、类型约束、资源归属检查、大小限制、超时和幂等键。把“不要调用危险工具”写进 prompt，只能减少一部分误用，不能构成安全边界。

## 7.5 人工确认

人工确认不是一个通用的“同意”按钮，而是把即将发生的具体副作用交给有责任的主体做最后判断。确认必须绑定到动作本身：目标资源、参数、金额或内容摘要、执行者、有效期和当前版本。若用户确认后资源状态发生变化，执行器应重新读取状态，必要时让用户再次确认，而不是继续执行旧动作。

通常需要确认的动作包括对外发送、删除或覆盖数据、修改权限、提交审批、产生费用以及影响客户或生产系统的变更。是否需要确认不能只按工具名判断，同一个 `update_record` 在测试租户和生产租户、草稿字段和支付字段上的风险完全不同。

一个合格的确认界面至少展示：

1. Agent 准备做什么，以及不会做什么。
2. 影响对象、租户和资源范围。
3. 将要提交的完整参数或内容摘要。
4. 采取这个动作的依据和前置步骤。
5. 可能的副作用、费用和不可逆部分。
6. 取消、拒绝、修改或转人工的路径。

确认还要有可审计结果：谁在什么时候确认了哪个动作版本，确认是否过期，执行是否与确认内容一致。若确认只显示“Agent 想继续”，用户实际上无法判断自己批准了什么；如果系统允许模型通过改写描述来绕过高风险分类，确认就失去了意义。

## 7.6 失败恢复

Agent 产品一定会遇到失败，但不同失败不能用同一个“再试一次”处理。最有用的第一步是区分失败类型：

1. **暂时失败**：网络超时、限流或服务短暂不可用，可能可以重试。
2. **确定性失败**：参数类型错误、资源不存在或权限不足，重试相同请求没有意义。
3. **信息不足**：缺少日期、对象或用户选择，应请求补充。
4. **目标冲突**：用户要求互相矛盾，应该澄清优先级。
5. **外部状态变化**：记录被别人修改、版本过期或审批已被处理，需要重新读取。
6. **执行状态未知**：请求超时，但下游可能已经成功，不能盲目重试写入。
7. **验收失败**：调用返回成功，但结果不满足业务条件，应停止并进入诊断。

重试前要检查动作是否幂等。读请求通常可以重试；写请求需要幂等键、去重记录或事务语义；支付、发信和删除等动作即使接口返回超时，也必须先查询外部状态，再决定是否补偿。对未知状态直接重试，可能把一次动作变成两次副作用。

恢复策略可以包括换工具、重新检索、请求用户补充、回滚、补偿、输出部分完成结果、转人工或停止。产品要把“失败”“部分完成”和“执行状态未知”分开呈现：前者说明动作未成功，后者说明系统还不能证明成功与否。成熟 Agent 产品不要求永远成功，但必须让失败可控、可解释、可恢复，且不把未知伪装成成功。

## 7.7 状态管理

Agent 的上下文不是可靠的状态数据库。模型上下文可能被截断、压缩或重新组织，工具结果也可能过期。因此，生产系统需要把任务状态持久化在模型之外，并为每个状态变更记录版本。

任务状态至少包括用户目标和约束、当前计划、已完成步骤、工具调用及结果摘要、失败尝试、待确认动作、预算、外部资源版本和最终验收结果。可以把任务建模为状态机，例如“created -> planning -> waiting_confirmation -> executing -> verifying -> completed”，同时允许“failed”“paused”和“unknown_external_state”等分支。状态转换必须由系统校验，模型不能直接把任务字段改成“completed”来绕过验收。

长任务还需要 checkpoint。checkpoint 应包含可以恢复所需的最小状态，而不是把一整段原始上下文无限保存。恢复时先确认权限和外部资源版本仍然有效，再从最后一个安全边界继续。并发执行时要使用版本号、租约或锁，避免两个 Agent 同时修改同一个资源；重复恢复时则依赖幂等键和已完成动作记录避免重复副作用。

## 7.8 审计日志

trace 既是调试数据，也是责任记录。一次可审计的 trace 至少要能关联用户、任务、租户、模型版本、工具版本、每个 action、参数校验、权限判定、工具返回摘要、确认事件、重试、状态转换和最终验收。最好使用不可变事件追加，而不是只保存最后一份任务 JSON；否则很难回答“谁在什么状态下做了什么”。

审计数据应区分三类用途：线上诊断需要足够细的时序，离线评估需要结构化字段，合规审查需要可追责和保留策略。原始 prompt、工具参数和返回值可能包含个人信息、密钥或客户数据，必须按字段脱敏、分级访问和设定保留期限。日志本身也要经过权限控制，不能为了审计而制造新的数据泄露面。

记录执行轨迹不等于保存模型的全部隐藏推理。产品需要的是可验证的计划、动作、观察、状态变化和理由摘要；对外展示时应让用户理解进度和风险，不应把未经验证的内部思路当作事实或承诺。

## 7.9 体验设计

Agent 的体验不能只有“正在执行”。用户需要知道系统的边界、当前状态和下一步选择，尤其是任务很长、动作有副作用或结果不确定时。

一个可用的任务界面应展示目标确认、计划摘要、已完成步骤、当前动作、等待原因、失败原因、待确认动作、证据和最终验收结果。用户应能暂停、取消、修改约束、补充信息和转人工；取消后还要明确哪些动作已经发生，不能把“停止生成”误写成“撤销外部操作”。

进度反馈可以是结构化状态，不必暴露完整内部推理。例如“已读取 3 个系统，发现订单状态冲突，正在等待你的选择”比一串未经核验的思考文本更有用。对长任务，系统还应给出预计剩余步骤、当前预算和最后一次状态更新时间；如果外部调用超时，应明确显示“结果待确认”。

## 7.10 成本控制

Agent 成本不只来自模型 token。对一次任务，可以用下面的教学模型估算：

```math
C_{\mathrm{task}}=
C_{\mathrm{model}}+
C_{\mathrm{tool}}+
C_{\mathrm{retrieval}}+
C_{\mathrm{retry}}+
C_{\mathrm{human}}+
C_{\mathrm{storage}}
```

其中模型成本包括输入、输出和推理预算，工具成本包括外部 API、浏览器、代码执行和数据库资源，重试成本包括失败调用和重新生成，人审成本包括等待和人工处理时间，存储成本包括 trace、附件和评估数据。真正的单位经济账还要按“每个成功完成的任务”而不是“每次调用”核算：

```math
C_{\mathrm{success}}=
\frac{\text{周期总成本}}{\text{周期内验收成功的任务数}}
\quad
\text{当成功任务数}>0
```

Budget manager 应在任务开始时预留预算，在每一步执行前检查剩余预算，并在接近上限时选择停止、降级、转人工或请求用户确认。预算至少应覆盖最大步骤数、工具次数、token、时间和费用；只限制 token 而不限制外部动作次数，仍可能产生不可接受的副作用。

简单问题应降级到普通问答或固定 workflow；重复查询可以缓存，但缓存必须绑定租户、权限和资源版本。成本优化不能通过减少验证、日志或高风险确认实现，否则节约的是账面费用，增加的是事故成本。

## 7.11 Agent 与 Workflow 结合

生产系统常把 Agent 嵌入 workflow，而不是把整个业务流程交给自由循环。Workflow 负责身份、状态转换、审批、超时、重试、回滚和最终提交等稳定边界；Agent 负责难以穷举的分类、信息补全、候选方案生成和有限范围的工具选择。

例如工单处理可以是：

```text
接收工单 -> 规则校验 -> Agent 补充信息 -> 证据检查 -> 人工确认 -> 提交处理 -> 结果验收 -> 记录
```

这里 Agent 即使生成了错误分类，也不能跳过证据检查和人工确认；提交失败时 workflow 可以转入人工队列；结果验收失败时任务状态不会被模型写成完成。这种组合还便于逐步提高自动化：先只让 Agent 生成草稿，再开放低风险查询，最后才考虑有限写入。

选择 workflow 和 Agent 的边界时，应把每个节点的责任写清楚：谁决定下一步、谁校验参数、谁承担副作用、谁可以暂停、谁验收结果。边界越清楚，评估和事故归因越容易。

## 7.12 Agent 产品评估

Agent 的评估不能只看最终回答，也不能把任务完成率、延迟、成本和安全事件压成一个没有解释的分数。一个完整的评估报告至少要同时描述任务结果、工具动作、失败恢复、状态与观察的完整性、高风险动作的确认、权限检查、资源消耗、trace 覆盖以及业务结果。人工接管比例、用户采用率和满意度可以作为产品指标，但它们必须和客观的任务验收、动作记录放在一起解读：用户满意不代表权限正确，工具成功也不代表任务完成。下面的指标分别对应不同对象和不同分母，不能互相替代。

### 7.12.1 把 Agent 评估拆成可解释的指标

可以把一次 Agent 产品任务样本写成：

```math
a_i=(g_i,u_i,\ell_i,T_i,s_i,A_i,O_i,H_i,C_i,L_i,B_i)
```

其中 `g_i` 是用户目标，`u_i` 是用户和角色，`\ell_i` 是自动化层级，`T_i` 是可用工具集合，`s_i` 是结构化状态，`A_i` 是动作序列，`O_i` 是观察结果，`H_i` 是确认或审批记录，`C_i` 是成本，`L_i` 是延迟，`B_i` 是业务结果。每项指标都必须说明它针对的是任务、动作、失败事件、状态变更还是用户结果。

评估时保留三种状态：`passed` 表示数据完整且满足阈值，`failed` 表示数据完整但不满足阈值或发生阻断性事件，`unknown` 表示没有足够记录、没有定义分母或外部状态无法确认。没有高风险动作不等于确认覆盖率为 1，没有失败不等于恢复率为 1，没有工具调用不等于工具成功率为 1。

**1. 任务成功率**

Agent 产品首先要完成任务，而不是只生成看似合理的过程：

```math
R_{\mathrm{task}}=
\frac{\sum_{i\in V_{\mathrm{task}}}y_i}{|V_{\mathrm{task}}|}
\quad
\text{当 }|V_{\mathrm{task}}|>0
```

其中 `y_i=1` 表示第 `i` 个任务满足预先定义的验收标准，例如工单字段正确、代码测试通过、报表生成并被采纳。部分完成可以另行记录，不应在没有规则时由模型自己宣布成功。没有验收标准或没有有效任务时，结果为 `unknown`。

**2. 工具执行成功率**

工具调用要同时看选择、参数、权限、执行结果和后置校验：

```math
R_{\mathrm{tool}}=
\frac{\sum_{j\in V_{\mathrm{tool}}}e_j}{|V_{\mathrm{tool}}|}
\quad
\text{当 }|V_{\mathrm{tool}}|>0
```

其中 `V_{\mathrm{tool}}` 是满足记录和验收条件的工具调用集合，`e_j=1` 表示第 `j` 次调用满足工具契约并得到可验证结果。被权限层拦截的危险调用不能被简单当成“工具失败”；应另记为安全策略正确拦截或违规尝试。没有工具调用时，工具成功率是 `unknown`，而不是 1。

**3. 高风险确认覆盖率**

设 `V_{\mathrm{risk}}` 是被策略标记为需要确认的动作集合，`h_j=1` 表示该动作有清晰、未过期且与实际参数一致的确认或审批：

```math
C_{\mathrm{conf}}=
\frac{\sum_{j\in V_{\mathrm{risk}}}h_j}{|V_{\mathrm{risk}}|}
\quad
\text{当 }|V_{\mathrm{risk}}|>0
```

如果当前评估集没有高风险动作，覆盖率为 `unknown`；系统仍应通过策略单元测试证明遇到高风险动作时会暂停。确认内容必须覆盖对象、参数、理由、风险和动作版本，不能把一个泛化的“允许 Agent 操作”算作每次具体确认。

**4. 未授权动作率**

越权指标的分母应是所有经过权限判定的动作或数据访问事件，而不是只统计成功调用：

```math
R_{\mathrm{unauth}}=
\frac{N_{\mathrm{unauth}}}{N_{\mathrm{auth\_check}}}
\quad
\text{当 }N_{\mathrm{auth\_check}}>0
```

其中 `N_{\mathrm{unauth}}` 是越权工具调用、越权数据访问、越权外发或越权修改事件数。没有权限检查记录时结果是 `unknown`；“没有日志”不是“没有越权”。一条真实越权事件就应触发阻断或事故流程，不能由大量正常调用把平均值冲淡。

**5. 失败恢复率**

设 `V_{\mathrm{recover}}` 是确实需要恢复的失败事件，`r_j=1` 表示通过重试、换工具、澄清、降级、回滚或人工接管得到可验收结果：

```math
R_{\mathrm{rec}}=
\frac{\sum_{j\in V_{\mathrm{recover}}}r_j}{|V_{\mathrm{recover}}|}
\quad
\text{当 }|V_{\mathrm{recover}}|>0
```

没有失败事件时，恢复率为 `unknown`，但可以单独报告“无失败样本”。执行状态未知不能算恢复成功，必须等外部系统确认或转入人工处理。

**6. 状态更新覆盖率和观察使用率**

设 V_state 是按任务契约要求记录的状态变更，s_k=1 表示该变更正确持久化；设 V_obs 是需要影响后续决策的观察，o_k=1 表示观察确实被后续动作或状态转移使用：

```math
C_{\mathrm{state}}=
\frac{\sum_{k\in V_{\mathrm{state}}}s_k}{|V_{\mathrm{state}}|},
\qquad
R_{\mathrm{obs}}=
\frac{\sum_{k\in V_{\mathrm{obs}}}o_k}{|V_{\mathrm{obs}}|}
```

两个分母都必须大于 0。状态覆盖率低，说明 Agent 可能忘记约束、重复动作或误判完成；观察使用率低，说明它拿到了工具结果却没有真正纳入下一步决策。没有要求状态更新或没有需要使用的观察时，分别记为 unknown，不把“不适用”混成通过。

**7. 预算超限率和单位成功成本**

设 V_task 是有预算记录的任务，b_i=1 表示任务超过步数、工具调用、token、时间或费用预算：

```math
R_budget =
\frac{\sum_{i\in V_task}b_i}{|V_task|}
\quad
\text{当 }|V_task|>0
```

单位成功成本要使用真实周期成本和验收成功任务数。当成功任务数为 0 时，单位成功成本是 `unknown`，而不是 0。预算超限率低也不代表体验良好，仍要分解模型、工具、重试、人工和存储成本。

**8. Agent 产品上线状态**

上线判断不应把所有指标相乘成一个布尔值。对每个必要指标保留状态，并按下面规则汇总：

```math
\mathrm{status}=
\begin{cases}
\mathrm{failed},&\exists j:s_j=\mathrm{failed}\\
\mathrm{unknown},&\text{不存在 failed 且 }\exists j:s_j=\mathrm{unknown}\\
\mathrm{passed},&\text{所有必要指标都是 passed}
\end{cases}
```

任务成功、工具契约、权限、确认、恢复、预算、P95 延迟、单位成本、trace 完整、评估覆盖、反馈闭环和业务结果是否属于“必要指标”，要按场景风险决定。高风险动作即使样本很少，也不能由综合分代替专项测试；越权和未知外部状态通常是阻断项。

### 7.12.2 为什么综合分不能替代轨迹审计

综合分可以用于版本排序和排查，但不能替代轨迹审计。一个 Agent 可能任务成功率很高，却依赖过大的工具权限；也可能工具调用成功率很高，却在结果验收和失败恢复上反复出错。评估报告至少要同时展示任务结果、动作结果、风险事件、状态完整性、资源消耗和测试覆盖度，并保留能够回到具体任务和动作的 trace。

如果为了比较版本而计算排序分，只能对有定义域且已经归一化的指标加权：

```math
Q=\frac{\sum_{j\in V}w_jm_j}{\sum_{j\in V}w_j}
\quad\text{当}\quad
\sum_{j\in V}w_j>0
```

这里 `V` 是本次比较中有可靠观测值的指标集合，`m_j` 是第 `j` 个指标归一化后的值，通常要求 `0\le m_j\le1`，`w_j` 是非负且有限的业务权重。未知指标不能按 0 或 1 代入；它们应从排序分的观测集合中排除，同时报告覆盖度：

```math
\mathrm{coverage}=\frac{\sum_{j\in V}w_j}{\sum_{j\in J}w_j}
\quad\text{当}\quad
\sum_{j\in J}w_j>0
```

`J` 是预先约定的全部指标集合。覆盖度低时，即使 `Q` 很高，也只能说明少数指标表现好，不能说明系统已经适合放量。尤其是权限、确认、验收和外部状态这些安全相关指标，只要存在 `unknown`，总体结论就不能写成“全部通过”；它们需要单独补齐记录、专项测试或人工复核。排序分的作用是帮助发现版本差异，最终的产品决定仍应由独立的安全测试、业务验收、回归评估和责任人审批共同作出。

## 7.13 适合先落地的 Agent 场景

早期试点应优先选择“价值清楚、风险可控、结果可验证”的任务。

内部代码助手、工单摘要、知识检索、报表草稿和运维排查辅助通常比直接处理客户资金或生产权限更合适，但“内部”本身不等于低风险，代码仓库和员工数据同样可能包含敏感信息。

一个候选任务至少要有明确的输入范围、允许访问的数据、工具白名单、完成标准、人工介入点、最大预算和失败后的责任人。最好先从只读或可撤销动作开始，让团队获得真实 trace，再逐步开放写入。试点阶段还要保留对照组或人工基线，否则只能知道 Agent 做了什么，不知道它是否真的减少了时间、错误或成本。

当任务涉及删除、付款、对外承诺、权限变更或生产变更时，通常应先把 Agent 放在建议、草稿或审批准备位置。只有当权限、验收、回滚和事故响应都已经验证，才考虑扩大自动执行范围。产品边界应写进任务契约，不应依赖使用者自己猜测。

## 7.14 常见失败模式与诊断顺序

Agent 产品的问题通常不是“模型不会思考”这么简单，而是任务契约和执行系统没有把行动边界落实下来。常见失败可以按第一分歧点诊断：

1. 目标不清或没有验收标准，说明产品定义问题先于模型问题。
2. 计划看似合理但工具权限过大，说明任务权限没有从用户权限中收缩。
3. 参数合法却指向错误资源，说明执行器缺少租户、资源归属或版本校验。
4. 工具超时后重复写入，说明幂等和“外部状态未知”没有建模。
5. 高风险动作没有暂停，说明风险分类或确认绑定失效。
6. 任务中途丢失约束、重复操作，说明状态持久化和 checkpoint 不可靠。
7. 用户不知道系统做到哪一步，说明进度、等待原因和副作用没有被产品化。
8. 线上成本突然升高，说明预算管理、重试上限或复杂任务降级缺失。
9. 结果流畅但不可验收，说明系统把文本质量误当成任务成功。
10. 事故无法复盘，说明 trace、版本、权限和确认记录不完整。

排查时先确定动作是否已经发生，再判断模型、工具、权限、状态和用户界面中的第一处偏离。若先从 prompt 或模型版本开始试错，很容易掩盖真正的控制面缺陷。

## 7.15 企业 Agent 方案的设计顺序

设计企业 Agent 时，第一步不是选模型，而是写任务契约：谁发起任务、目标资源是什么、允许的工具和数据范围是什么、哪些动作只能草拟或必须审批、成功如何验收、预算和截止时间是多少。契约确定后，才能决定哪些节点用 workflow，哪些判断交给 Agent。

第二步是建立执行边界。工具注册表要描述参数、资源类型、权限、风险等级、幂等语义、超时和结果 schema；执行器要在每次调用前后做校验；状态存储要支持版本和 checkpoint；确认服务要绑定到具体动作版本；审计系统要能把用户、任务、动作和结果串起来。

## 7.16 企业 Agent 方案的评估与放量

第三步是做分层评估。

离线评估要覆盖任务成功、工具选择、参数、证据、恢复和高风险策略；沙箱评估要覆盖提示注入、越权、恶意工具结果、资源消耗和重复执行；线上试点则同时记录人工基线、采用率、接管率、成本、延迟和事故。没有完成标准或没有安全覆盖的指标，应保留为 unknown，而不是在综合分中填一个默认值。

最后才是逐步放开自动化。先运行建议或草稿模式，验证 trace 和人工判断；再开放低风险、可撤销动作；对外发送、生产变更、删除、支付和权限变更继续保留审批或人工确认。每次扩大范围都应有版本化的评估结果、回滚方案和明确责任人。

## 7.17 最小可运行 Agent 产品审计 demo

下面这个 demo 用 0 依赖 Python 模拟 Agent 产品评估。它把任务成功、工具执行、高风险确认、失败恢复、状态更新、观察使用、trace 覆盖、越权动作、预算超限、延迟、单位成本、评估集、反馈闭环和业务指标放进同一张表。代码中的阈值、样例数量和成本均为教学构造，用来演示定义域、状态合并和诊断方式，不代表任何行业标准。

```python
import math


UNKNOWN = "unknown"
PASSED = "passed"
FAILED = "failed"


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def ratio(numerator, denominator):
    if (
        not finite_number(numerator)
        or not finite_number(denominator)
        or denominator <= 0
        or numerator < 0
        or numerator > denominator
    ):
        return UNKNOWN
    return numerator / denominator


def min_status(value, threshold):
    if value == UNKNOWN:
        return UNKNOWN
    return PASSED if value >= threshold else FAILED


def max_status(value, threshold):
    if value == UNKNOWN:
        return UNKNOWN
    return PASSED if value <= threshold else FAILED


def overall_status(statuses):
    if FAILED in statuses:
        return FAILED
    if UNKNOWN in statuses:
        return UNKNOWN
    return PASSED


def invalid_fields(agent):
    count_fields = [
        "tasks_total",
        "tasks_completed",
        "tool_calls",
        "tool_successes",
        "high_risk_actions",
        "high_risk_confirmed",
        "recoveries_needed",
        "recoveries_succeeded",
        "state_update_required",
        "state_updates",
        "observations_total",
        "observations_used",
        "trace_complete",
        "auth_checks",
        "unauthorized_actions",
        "budget_overruns",
    ]
    invalid = [
        field
        for field in count_fields
        if (
            not isinstance(agent[field], int)
            or isinstance(agent[field], bool)
            or agent[field] < 0
        )
    ]
    for field in ["eval_ready", "feedback_loop", "business_metric_defined"]:
        if not finite_number(agent[field]) or not 0 <= agent[field] <= 1:
            invalid.append(field)
    for field in ["p95_latency_ms", "latency_slo_ms", "cost_per_task", "cost_slo"]:
        if not finite_number(agent[field]) or agent[field] < 0:
            invalid.append(field)
    bounds = [
        ("tasks_completed", "tasks_total"),
        ("tool_successes", "tool_calls"),
        ("high_risk_confirmed", "high_risk_actions"),
        ("recoveries_succeeded", "recoveries_needed"),
        ("state_updates", "state_update_required"),
        ("observations_used", "observations_total"),
        ("trace_complete", "tasks_total"),
        ("unauthorized_actions", "auth_checks"),
        ("budget_overruns", "tasks_total"),
    ]
    for numerator, denominator in bounds:
        if (
            isinstance(agent[numerator], int)
            and isinstance(agent[denominator], int)
            and agent[numerator] > agent[denominator]
        ):
            invalid.append(numerator + "_exceeds_" + denominator)
    return invalid


def audit_agent_product(agent):
    if not isinstance(agent, dict):
        return {"name": "<invalid>", "status": UNKNOWN, "errors": ["not_a_mapping"]}

    required = [
        "name",
        "mode",
        "tasks_total",
        "tasks_completed",
        "tool_calls",
        "tool_successes",
        "high_risk_actions",
        "high_risk_confirmed",
        "recoveries_needed",
        "recoveries_succeeded",
        "state_update_required",
        "state_updates",
        "observations_total",
        "observations_used",
        "trace_complete",
        "auth_checks",
        "unauthorized_actions",
        "budget_overruns",
        "p95_latency_ms",
        "latency_slo_ms",
        "cost_per_task",
        "cost_slo",
        "eval_ready",
        "feedback_loop",
        "business_metric_defined",
    ]
    missing = [field for field in required if field not in agent]
    if missing:
        return {
            "name": agent.get("name", "<unnamed>"),
            "status": UNKNOWN,
            "errors": ["missing:" + field for field in missing],
        }

    invalid = invalid_fields(agent)
    if invalid:
        return {
            "name": agent["name"],
            "mode": agent["mode"],
            "status": UNKNOWN,
            "errors": ["invalid:" + field for field in invalid],
        }

    metrics = {
        "task_success": ratio(agent["tasks_completed"], agent["tasks_total"]),
        "tool_success": ratio(agent["tool_successes"], agent["tool_calls"]),
        "confirmation_coverage": ratio(
            agent["high_risk_confirmed"], agent["high_risk_actions"]
        ),
        "recovery_rate": ratio(
            agent["recoveries_succeeded"], agent["recoveries_needed"]
        ),
        "state_update_coverage": ratio(
            agent["state_updates"], agent["state_update_required"]
        ),
        "observation_use_rate": ratio(
            agent["observations_used"], agent["observations_total"]
        ),
        "trace_coverage": ratio(agent["trace_complete"], agent["tasks_total"]),
        "unauthorized_rate": ratio(
            agent["unauthorized_actions"], agent["auth_checks"]
        ),
        "budget_overrun_rate": ratio(
            agent["budget_overruns"], agent["tasks_total"]
        ),
        "eval_ready": agent["eval_ready"],
        "feedback_loop": agent["feedback_loop"],
        "business_metric_defined": agent["business_metric_defined"],
    }
    statuses = {
        "task_success": min_status(metrics["task_success"], 0.75),
        "tool_success": min_status(metrics["tool_success"], 0.85),
        "high_risk_confirmation": min_status(
            metrics["confirmation_coverage"], 0.90
        ),
        "recovery": min_status(metrics["recovery_rate"], 0.60),
        "state_update": min_status(metrics["state_update_coverage"], 0.80),
        "observation_use": min_status(metrics["observation_use_rate"], 0.80),
        "trace_coverage": min_status(metrics["trace_coverage"], 0.90),
        "unauthorized_action": (
            FAILED
            if agent["unauthorized_actions"] > 0
            else (
                UNKNOWN
                if metrics["unauthorized_rate"] == UNKNOWN
                else PASSED
            )
        ),
        "budget_overrun": max_status(metrics["budget_overrun_rate"], 0.10),
        "p95_latency": max_status(
            agent["p95_latency_ms"], agent["latency_slo_ms"]
        ),
        "unit_cost": max_status(agent["cost_per_task"], agent["cost_slo"]),
        "eval_ready": min_status(metrics["eval_ready"], 0.80),
        "feedback_loop": min_status(metrics["feedback_loop"], 0.75),
        "business_metric": min_status(
            metrics["business_metric_defined"], 1.0
        ),
    }
    score_values = {
        "task_success": metrics["task_success"],
        "tool_success": metrics["tool_success"],
        "state_update_coverage": metrics["state_update_coverage"],
        "observation_use_rate": metrics["observation_use_rate"],
        "trace_coverage": metrics["trace_coverage"],
        "permission_quality": (
            UNKNOWN
            if metrics["unauthorized_rate"] == UNKNOWN
            else 1.0 - metrics["unauthorized_rate"]
        ),
        "budget_quality": (
            UNKNOWN
            if metrics["budget_overrun_rate"] == UNKNOWN
            else 1.0 - metrics["budget_overrun_rate"]
        ),
        "eval_ready": metrics["eval_ready"],
        "feedback_loop": metrics["feedback_loop"],
    }
    weights = {
        "task_success": 0.22,
        "tool_success": 0.14,
        "state_update_coverage": 0.10,
        "observation_use_rate": 0.08,
        "trace_coverage": 0.10,
        "permission_quality": 0.14,
        "budget_quality": 0.06,
        "eval_ready": 0.08,
        "feedback_loop": 0.08,
    }
    observed_weight = sum(
        weights[name] for name, value in score_values.items() if value != UNKNOWN
    )
    score = (
        UNKNOWN
        if observed_weight == 0
        else sum(
            weights[name] * score_values[name]
            for name in weights
            if score_values[name] != UNKNOWN
        )
        / observed_weight
    )
    return {
        "name": agent["name"],
        "mode": agent["mode"],
        "status": overall_status(list(statuses.values())),
        "agent_score": UNKNOWN if score == UNKNOWN else round(score, 3),
        "score_coverage": round(observed_weight / sum(weights.values()), 3),
        "metrics": {
            name: value if value == UNKNOWN else round(value, 3)
            for name, value in metrics.items()
        },
        "metric_status": statuses,
        "failed_gates": [
            name for name, value in statuses.items() if value == FAILED
        ],
        "unknown_gates": [
            name for name, value in statuses.items() if value == UNKNOWN
        ],
    }


agents = [
    {
        "name": "support_ticket_agent",
        "mode": "semi_auto",
        "tasks_total": 120,
        "tasks_completed": 101,
        "tool_calls": 260,
        "tool_successes": 238,
        "high_risk_actions": 18,
        "high_risk_confirmed": 18,
        "recoveries_needed": 22,
        "recoveries_succeeded": 16,
        "state_update_required": 240,
        "state_updates": 218,
        "observations_total": 260,
        "observations_used": 236,
        "trace_complete": 116,
        "auth_checks": 260,
        "unauthorized_actions": 0,
        "budget_overruns": 8,
        "p95_latency_ms": 4200,
        "latency_slo_ms": 5000,
        "cost_per_task": 0.18,
        "cost_slo": 0.25,
        "eval_ready": 0.86,
        "feedback_loop": 0.80,
        "business_metric_defined": 1.0,
    },
    {
        "name": "code_fix_agent",
        "mode": "copilot",
        "tasks_total": 70,
        "tasks_completed": 48,
        "tool_calls": 190,
        "tool_successes": 168,
        "high_risk_actions": 4,
        "high_risk_confirmed": 4,
        "recoveries_needed": 28,
        "recoveries_succeeded": 15,
        "state_update_required": 150,
        "state_updates": 126,
        "observations_total": 190,
        "observations_used": 151,
        "trace_complete": 66,
        "auth_checks": 190,
        "unauthorized_actions": 0,
        "budget_overruns": 11,
        "p95_latency_ms": 7600,
        "latency_slo_ms": 8000,
        "cost_per_task": 0.42,
        "cost_slo": 0.45,
        "eval_ready": 0.78,
        "feedback_loop": 0.70,
        "business_metric_defined": 1.0,
    },
    {
        "name": "data_ops_agent",
        "mode": "approval",
        "tasks_total": 80,
        "tasks_completed": 55,
        "tool_calls": 210,
        "tool_successes": 171,
        "high_risk_actions": 24,
        "high_risk_confirmed": 18,
        "recoveries_needed": 30,
        "recoveries_succeeded": 17,
        "state_update_required": 190,
        "state_updates": 139,
        "observations_total": 210,
        "observations_used": 149,
        "trace_complete": 65,
        "auth_checks": 210,
        "unauthorized_actions": 2,
        "budget_overruns": 13,
        "p95_latency_ms": 9100,
        "latency_slo_ms": 6500,
        "cost_per_task": 0.31,
        "cost_slo": 0.30,
        "eval_ready": 0.75,
        "feedback_loop": 0.58,
        "business_metric_defined": 1.0,
    },
    {
        "name": "generic_browser_agent",
        "mode": "auto",
        "tasks_total": 60,
        "tasks_completed": 29,
        "tool_calls": 240,
        "tool_successes": 150,
        "high_risk_actions": 30,
        "high_risk_confirmed": 9,
        "recoveries_needed": 35,
        "recoveries_succeeded": 8,
        "state_update_required": 180,
        "state_updates": 80,
        "observations_total": 240,
        "observations_used": 96,
        "trace_complete": 36,
        "auth_checks": 240,
        "unauthorized_actions": 7,
        "budget_overruns": 21,
        "p95_latency_ms": 12500,
        "latency_slo_ms": 7000,
        "cost_per_task": 0.62,
        "cost_slo": 0.28,
        "eval_ready": 0.40,
        "feedback_loop": 0.20,
        "business_metric_defined": 0.0,
    },
]

def run_boundary_checks():
    missing_auth = dict(agents[0])
    del missing_auth["auth_checks"]
    assert audit_agent_product(missing_auth)["status"] == UNKNOWN

    nan_latency = dict(agents[0])
    nan_latency["p95_latency_ms"] = float("nan")
    assert audit_agent_product(nan_latency)["status"] == UNKNOWN

    no_risk_action = dict(agents[0])
    no_risk_action["high_risk_actions"] = 0
    no_risk_action["high_risk_confirmed"] = 0
    assert (
        audit_agent_product(no_risk_action)["metric_status"]["high_risk_confirmation"]
        == UNKNOWN
    )

    no_recovery_event = dict(agents[0])
    no_recovery_event["recoveries_needed"] = 0
    no_recovery_event["recoveries_succeeded"] = 0
    assert audit_agent_product(no_recovery_event)["metrics"]["recovery_rate"] == UNKNOWN

    no_permission_record = dict(agents[0])
    no_permission_record["auth_checks"] = 0
    no_permission_record["unauthorized_actions"] = 0
    assert audit_agent_product(no_permission_record)["metrics"]["unauthorized_rate"] == UNKNOWN


run_boundary_checks()

results = [audit_agent_product(agent) for agent in agents]
ranked = sorted(
    [(r["name"], r["agent_score"], r["status"]) for r in results],
    key=lambda item: (
        item[1] == UNKNOWN,
        0 if item[1] == UNKNOWN else -item[1],
        item[0],
    ),
)
agent_pass = [r["name"] for r in results if r["status"] == PASSED]
needs_rework = {
    r["name"]: {
        "failed": r["failed_gates"],
        "unknown": r["unknown_gates"],
    }
    for r in results
    if r["status"] != PASSED
}

print("ranked=", ranked)
print("agent_pass=", agent_pass)
print("sample_metrics=", results[0]["metrics"])
print("needs_rework=", needs_rework)
```

一组典型输出是：

```text
ranked= [('support_ticket_agent', 0.902, 'passed'), ('code_fix_agent', 0.825, 'failed'), ('data_ops_agent', 0.772, 'failed'), ('generic_browser_agent', 0.553, 'failed')]
agent_pass= ['support_ticket_agent']
sample_metrics= {'task_success': 0.842, 'tool_success': 0.915, 'confirmation_coverage': 1.0, 'recovery_rate': 0.727, 'state_update_coverage': 0.908, 'observation_use_rate': 0.908, 'trace_coverage': 0.967, 'unauthorized_rate': 0.0, 'budget_overrun_rate': 0.067, 'eval_ready': 0.86, 'feedback_loop': 0.8, 'business_metric_defined': 1.0}
needs_rework= {'code_fix_agent': {'failed': ['task_success', 'recovery', 'observation_use', 'budget_overrun', 'eval_ready', 'feedback_loop'], 'unknown': []}, 'data_ops_agent': {'failed': ['task_success', 'tool_success', 'high_risk_confirmation', 'recovery', 'state_update', 'observation_use', 'trace_coverage', 'unauthorized_action', 'budget_overrun', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop'], 'unknown': []}, 'generic_browser_agent': {'failed': ['task_success', 'tool_success', 'high_risk_confirmation', 'recovery', 'state_update', 'observation_use', 'trace_coverage', 'unauthorized_action', 'budget_overrun', 'p95_latency', 'unit_cost', 'eval_ready', 'feedback_loop', 'business_metric'], 'unknown': []}}
```

这个 demo 的重点是：Agent 产品不是越自动越好。`support_ticket_agent` 通过验收，是因为它是半自动、任务边界清楚、确认覆盖和 trace 稳定；`code_fix_agent` 适合继续做 copilot，因为任务成功、恢复、预算和反馈还没过线；`data_ops_agent` 虽然有审批形态，但高风险确认、越权、成本和延迟不过线；`generic_browser_agent` 说明通用全自动浏览器 Agent 如果没有强权限、预算、确认和评估，很难进入生产。

## 7.18 本章小结

Agent 产品化的核心是让模型在受控范围内执行任务。它比普通问答更有价值，也更有风险。落地时要优先选择边界清楚、可验证、低风险的任务，从辅助和半自动开始，逐步扩大自动化能力。

下一章会进入多模态产品落地，讨论图像、语音、视频和多模态理解生成能力如何转化为真实产品体验。
