# 第 13 章 AgentWorld：模型只有进入环境，才真正面对任务

## 13.0 从回答到完成：为什么 Agent 需要环境

让模型回答“如何修复一个文件”，和让它在真实仓库里完成修复，是两种不同的能力。前者只需要生成一段看起来合理的文字；后者需要观察目录结构，读取文件，选择工具，执行命令，理解错误，修改工作区，运行测试，并确认最终文件确实满足任务要求。

例如，用户说：“请修复日期解析错误。”模型可以直接给出一段代码。可是这段代码可能改错文件、覆盖用户尚未提交的修改、依赖一个并不存在的库，或者只让公开测试通过而破坏了隐藏行为。只有仓库状态、测试结果和最终 diff 能够说明任务是否完成，模型的总结不能代替这些事实。

本章把这样的交互环境称为 AgentWorld。它不是某一个必须安装的产品名称，而是一种分析框架：Agent 通过 observation 看到世界，通过 action 改变世界，再根据新的 observation 继续决策。这个世界可以是代码仓库、浏览器、数据库沙箱、文件系统、游戏模拟器，也可以是由多个服务组成的业务工作流。

因此，评估一个 Agent 时不能只记录模型名称。模型能力、工具能力、环境信息量、权限、grader 和 reset 机制共同决定最终结果。一个 Agent 完成任务，不一定说明模型独自具备了全部能力；它可能使用了更强的搜索工具、更宽的观察权限，或者得到了一个过于宽松的成功判据。可靠的环境契约，必须把这些因素分别写出来，才能说明一次成功究竟证明了什么。

## 13.1 AgentWorld 到底在描述什么

AgentWorld 可以看作三个对象的组合：

1. **策略或模型**：根据当前目标、历史轨迹和 observation 产生下一步 action。
2. **环境**：保存真实状态，执行 action，产生状态转移和结果。
3. **harness**：负责初始化任务、过滤观察、限制权限、验证结果、重置环境并记录轨迹。

如果只有模型而没有环境，评估往往是输入到输出的静态比较；如果有环境但没有明确 harness，任务之间会互相污染，或者不同实验获得不同信息。一个可复现的 Agent 系统必须同时记录三者的版本和边界。

环境不等于“把所有工具放给模型”。工具只是 action 的一个载体，网页文本、编译错误和数据库返回也是 observation 的一部分。真正重要的是：哪些状态可以被看见，哪些动作可以执行，动作之后什么状态算成功，发生异常时如何恢复，以及一次任务结束后环境是否回到了可重复的起点。

### 13.1.1 一个贯穿全章的合同助手例子

假设合同助手接到任务：检查供应商合同中的付款期限是否符合当前政策；如果发现例外，只创建一个待审核工单，不得直接批准、发信或修改合同。

这个任务至少包含四种状态：

- 合同和政策文档的版本状态；
- 当前用户、租户和读取权限；
- 规则判断是否有充分证据；
- 工单是否真的落库并处于待审核状态。

如果模型只输出“存在例外，已创建工单”，这句话没有证明四种状态中的任何一种。环境必须让助手读取指定合同版本和政策版本，验证主体与生效日期，调用受限的工单接口，再重新读取工单状态。这样，模型的语言输出只是最后一个 artifact，任务事实则由环境中的对象和事件决定。

## 13.2 把 Agent 写成一个闭环

在离散时间 t，环境真实状态为 s_t，Agent 观察到 o_t，选择动作 a_t，环境根据动作转移到下一个状态，并返回反馈：

~~~math
o_t \sim O(\cdot\mid s_t),\qquad
a_t \sim \pi_\theta(\cdot\mid h_t,o_t),\qquad
s_{t+1}\sim P(\cdot\mid s_t,a_t),\qquad
r_t=R(s_t,a_t,s_{t+1})
~~~

其中，π_θ 是模型或策略，h_t 是到当前为止的历史，O 是观察生成机制，P 是环境转移机制，R 是反馈函数。反馈可以是训练用 reward，也可以只是评测期间的结构化结果；不要因为符号写成 r_t，就把它误认为模型已经获得了可靠的学习标签。

一条轨迹可以写成：

~~~math
\tau=(s_0,o_0,a_0,e_0,s_1,o_1,a_1,e_1,\ldots,s_T,o_T)
~~~

真实的 s_t 通常只由环境和 verifier 持有，模型看到的主要是 o_t、工具返回 e_t 和任务历史。任务是否成功，应由目标谓词 G 作用于最终状态或完整轨迹，而不是由最后一句文字是否包含“完成”决定。先把 T 定义为轨迹结束时环境能够提供的最终状态或验证快照：如果执行被中断，或者外部写入的结果无法查询，T 不是一个可以随意补全的状态。对于一个已经完成且可验证的教学 benchmark，可以使用二值谓词：

~~~math
\mathrm{success}(\tau)=G(s_T,\tau)\in\{0,1\}
~~~

这里的 1 只能表示 verifier 已经确认目标满足，0 表示 verifier 已经确认目标不满足；没有运行测试、reset 失败、外部请求超时且无法回读时，都不能为了得到一个二值分数而擅自选择 0 或 1。代码任务可以把 G 定义为隐藏测试通过、禁止文件未变更且依赖契约未被破坏；浏览器任务可以要求目标业务对象的状态、版本和审计主体都正确；合同任务则可能要求证据充分、工单状态正确且没有执行越权动作。

在更接近生产的系统中，验证结果应保留三态语义。令每一个验收条件的结果 g_i 属于 {1, 0, unknown}，分别表示通过、明确不通过和证据不足，则安全结果可以按下面的规则计算：

~~~math
\mathrm{status}(\tau)=
\begin{cases}
0, & \exists i,\;g_i=0\\
1, & \forall i,\;g_i=1\\
\mathrm{unknown}, & \text{otherwise}
\end{cases}
~~~

因此，unknown 不是一种“暂时当作通过”的方便写法。报告中可以同时给出二值兼容字段 `success = (status == 1)`，但必须保留原始 status 和缺失证据的原因。

### 13.2.1 MDP、POMDP 和普通工作流的差别

如果策略能够直接看到完整状态，并且下一状态只由当前状态和动作决定，可以用 MDP 近似环境。如果模型只能看到状态投影，且历史对判断当前情况有帮助，更接近部分可观测的 POMDP。浏览器登录是否过期、数据库事务是否已经提交、网页按钮后面的业务状态，往往都属于模型无法直接看到的隐藏状态。

可用信念状态表示模型根据历史形成的状态分布：

~~~math
b_t(s)=P(s_t=s\mid o_{0:t},a_{0:t-1})
~~~

模型不一定显式计算 b_t，但环境设计者仍要意识到它在处理不确定性。一次“提交成功”的页面提示可能让模型提高对已提交状态的相信程度；网络超时则可能使状态同时包含“已提交”和“未提交”两种可能。此时直接重复提交不是更努力，而是把不确定性变成重复副作用。

普通 workflow 通常预先规定步骤和状态转移；AgentWorld 允许策略根据 observation 选择下一步，因此必须额外定义停止、重试、回退、未知状态和人工接管。自主性越高，环境契约越不能只写一个最终文本格式。

## 13.3 环境契约的四个基本面

一个可复现的环境至少要写清四类内容：

| 部分 | 要回答的问题 | 合同助手示例 |
|---|---|---|
| 初始状态 | 任务从哪个版本、数据和权限开始 | 合同 v7、政策 v2026-07、租户 A |
| observation | 模型能看到哪些字段，哪些被过滤 | 条款定位和状态，不暴露其他租户 |
| action | 模型可以请求哪些动作，参数和前置条件是什么 | 读取、检索、创建待审核工单 |
| 转移与结束 | 动作如何改变状态，什么时候成功、失败或未知 | 工单重新读取为 pending_review |

还要补充 reset、时间、网络、随机性、资源额度和外部副作用的规则。只写工具名称而不写这些条件，无法判断两个实验是否运行在同一个环境里。

### 13.3.1 任务契约不只是自然语言目标

一份可执行的任务契约可以拆成五层：

1. **目标对象**：合同 ID、版本、政策 ID 和租户。
2. **允许观察**：合同条款、政策条款、当前用户的授权结果。
3. **允许动作**：只读查询和创建待审核工单。
4. **禁止动作**：批准例外、修改合同、向外部收件人发送内容。
5. **验收条件**：关键 claim 有引用，工单状态和版本符合预期，事件账本没有越权操作。

自然语言目标解决“用户想要什么”，结构化契约解决“系统究竟怎样判定完成”。如果用户说“处理一下合同”，环境还需要澄清是生成分析报告、创建审批任务，还是实际批准；不能让模型用一个模糊动词自行扩大权限。

### 13.3.2 初始状态和任务独立性

代码任务的初始状态可以是 commit、依赖锁文件、操作系统镜像和测试数据；浏览器任务的初始状态可以是页面快照、登录身份和数据库记录；业务任务还需要固定时间、地区、汇率或政策版本。

任务独立性不是一句“每次重新开始”。它要求前一条轨迹的文件、cookie、进程、队列、数据库记录、缓存和临时凭据不会影响后一条轨迹。若环境无法做到完全清理，应把残留状态作为变量记录，并采用配对实验或显式失败，而不是假设样本独立。

## 13.4 成功、风险和部分完成必须分开

很多系统只有一个 success 字段，这会掩盖重要差异。合同助手可能已经找到了正确条款，但没有权限读取附录；Code Agent 可能生成了正确 patch，却没有运行测试；浏览器 Agent 可能进入了正确确认页，却没有提交业务对象。这些轨迹不应和“什么都没有做”归为同一类，也不能被记成完整成功。

一个有安全约束的目标谓词可以写成：

~~~math
G_{\mathrm{safe}}(\tau)
=G_{\mathrm{goal}}(\tau)
\land G_{\mathrm{artifact}}(\tau)
\land G_{\mathrm{policy}}(\tau)
\land G_{\mathrm{external}}(\tau)
~~~

四项分别表示目标状态、交付 artifact、策略约束和外部状态都满足。某些任务还需要把 reset 成功作为独立的实验有效性条件，而不是把 reset 混入 Agent 的业务成功。

四项不是四个可以由模型自报的标签，而是四类独立证据。G_goal 检查用户目标是否达成，G_artifact 检查交付物是否存在且对应正确版本，G_policy 检查权限、数据流和禁止动作，G_external 检查真实外部对象是否处于目标状态。任意一项没有可核验结果，整体就应保持 unknown；任意一项明确失败，整体才是 0。

带成本的效用可以用于分析取舍：

~~~math
U(\tau)=\mathbf{1}[G_{\mathrm{safe}}(\tau)=1]-\lambda C(\tau)-\mu R(\tau)
~~~

C 可以包含 token、工具调用、墙钟时间和人工复核；R 可以包含越权尝试、数据出域、不可逆副作用和未知状态；lambda 和 mu 是非负权重。这个式子是分析工具，不代表不同风险都可以随意折算成一个分数。如果 C 或 R 尚未测得，U 也应记为 unknown，而不是把缺失成本写成 0。支付、删除和合规场景通常先要求风险满足硬约束，再讨论成本。

### 13.4.1 部分成功的分层描述

为了保留诊断信息，可以把结果分成以下几类：

| 类别 | 事实 | 例子 |
|---|---|---|
| 未开始 | 没有形成有效 observation 或 action | 登录即失败 |
| 证据取得 | 找到相关对象，但未完成判断 | 找到合同条款，缺政策版本 |
| 判断完成 | 结论有证据，但没有完成外部动作 | 判定例外，未创建工单 |
| 动作提出 | 请求了动作，但被权限或业务规则拒绝 | 工单参数缺版本 |
| 动作完成 | 外部对象已改变，但还未充分验证 | 页面显示保存成功 |
| 安全成功 | 目标、artifact、策略和外部状态都通过验证 | 工单状态正确且可回读 |

这些类别不是给人贴标签，而是把失败放回它发生的阶段。只有这样，修复团队才知道应该改观察接口、工具 schema、模型规划、执行器还是 verifier。

## 13.5 Observation：模型看到的不是环境本身

环境完整状态 s_t 往往包含数据库内部字段、隐藏测试、密钥、其他租户数据和 verifier 结果。模型能够看到的 observation 应是带权限和任务规则的投影：

~~~math
o_t=\Pi_{\mathrm{allowed}}(s_t,\mathrm{role},\mathrm{task},\mathrm{policy})
~~~

如果一个 benchmark 把隐藏答案、内部调试日志或后台数据库权限直接暴露给模型，成功率上升测到的是信息泄漏，而不是规划能力。反过来，如果 observation 完全不包含错误原因，任务就会退化成盲目试错。好的 observation 既不泄露不可见事实，也不故意删掉真实操作者合理能够得到的反馈。

### 13.5.1 Observation 的字段契约

一个结构化 observation 至少应说明：

~~~~json
{
  "source": "contract-service",
  "source_revision": "contract-1842:v7",
  "observed_at": "2026-08-14T09:30:00Z",
  "content_hash": "sha256:...",
  "tenant": "tenant-a",
  "permission_filtered": true,
  "freshness": "live",
  "data": {
    "clause_id": "3.2",
    "text": "付款期限为 90 天",
    "locator": "page=4"
  }
}
~~~~

source 和 source_revision 让系统知道事实来自哪个对象；observed_at 和 freshness 防止旧缓存被当成当前状态；tenant 和 permission_filtered 支持隔离审计；content_hash 用于判断同一版本是否被悄悄替换。不同环境可以增加分页、游标、错误类型、schema 版本和 trace id，但不能让这些字段的含义随调用变化。

Observation 也可能是低信任内容。网页中的文字、文档中的指令和工具返回的自由格式字段可以支持某个 claim，却不应自行修改 Agent 的目标、权限或工具 schema。控制平面和数据平面的分离，是环境闭环的基本安全条件。

### 13.5.2 新鲜度和版本比“看到了”更重要

长任务经常遇到观察过期。模型读取了订单状态为 pending，另一项动作随后把订单改成 cancelled；如果它继续使用旧 observation，就会产生错误判断。环境应通过版本号、ETag、时间戳或单调事件序列帮助执行器发现冲突。

对于允许写入的 action，可以要求调用方携带它所依据的 source_revision。若服务端版本已经变化，执行器返回 stale_observation，而不是静默覆盖新状态。这样，状态变化会进入轨迹，模型可以重新观察，而不是把并发冲突误认为普通工具失败。

## 13.6 Action：可调用不代表语义允许

Action 至少有三层约束：

| 层次 | 检查什么 | 失败示例 |
|---|---|---|
| 语法层 | JSON、类型、必填字段和枚举 | amount 传成字符串 |
| 语义层 | 资源存在、版本匹配、业务前置条件 | 试图关闭已关闭工单 |
| 策略层 | 身份、租户、范围、数据流和风险 | 普通用户批准例外 |

一个动作在当前状态下是否可执行，可以抽象为：

~~~math
\mathrm{Allowed}(a,s)
=\mathrm{SchemaOK}(a)
\land\mathrm{PreconditionOK}(a,s)
\land\mathrm{PolicyOK}(a,s)
\land\mathrm{BudgetOK}(a)
~~~

模型产生 action 只是请求，executor 必须重新进行这些检查。提示词可以告诉模型哪些动作危险，却不能代替服务端授权，也不能把模型输出中的“我已获得许可”当作权限事实。

这里的 s 必须是执行器能够信任的当前状态，而不是模型上一轮文字中的状态描述。如果状态、权限或预算任意一项为 unknown，Allowed 不能返回 true；执行器应先重新观察、缩小权限或把请求交给人工。只有在 schema、前置条件、策略和预算都得到明确证据时，动作才可以进入执行阶段。

### 13.6.1 读动作和写动作的边界

读动作通常也有权限和成本，但失败多数可以重试；写动作会改变外部状态，需要额外的版本、幂等键、预览或确认。例如：

- 读取合同条款：检查租户、版本和字段权限。
- 创建待审核工单：检查证据引用、幂等键和状态前置条件。
- 批准例外：即使模型判断正确，也不应因为它能调用接口就自动拥有批准权限。
- 发送邮件：检查收件人、脱敏结果、附件和人工确认状态。

把读和写都包装成同一种“tool call”会隐藏风险差异。环境契约应明确每个 action 的预计副作用、可逆性、审计字段和未知状态处理。

## 13.7 状态转移、事件账本和未知状态

动作完成后，环境应产生可追踪的事件，而不是只返回一句自然语言。一次写入至少需要记录请求参数的摘要、身份、目标资源、版本、执行结果、外部请求 ID 和验证查询。这样可以回答“谁在什么时候以什么权限请求了什么，外部对象到底变成了什么”。

网络超时尤其容易造成错误。请求可能在服务端已经成功后才丢失响应，也可能根本没有到达服务端。因此执行器应把结果分成 success、rejected、failed 和 unknown。unknown 不是 failed 的同义词，更不能直接触发一次无幂等键的重复写入。

对可重试动作，重试前要检查错误类别、幂等性和当前外部状态。合同助手创建工单时，可以用合同版本加任务 ID 作为幂等键，再查询该键对应的工单；浏览器提交报销后，应先用报销单号回读服务端；数据库事务超时后，应查询事务状态或使用可回滚的隔离环境。

### 13.7.1 事件账本的最小结构

一个有用的事件记录可以包含：

~~~~json
{
  "event_id": "evt-018",
  "trace_id": "trace-42",
  "actor": "agent-session-7",
  "action": "create_pending_review",
  "target": "contract-1842:v7",
  "request_hash": "sha256:...",
  "policy_revision": "policy-v12",
  "result": "unknown",
  "external_request_id": "req-91",
  "verified_state": null,
  "occurred_at": "2026-08-14T09:32:10Z"
}
~~~~

result 是执行器对这次调用的判断，verified_state 是之后对外部对象的回读结果，两者不能混为一谈。result=success 但 verified_state=missing，说明服务端确认了请求接收却没有得到预期对象；result=unknown 但 verified_state=pending_review，则说明动作可能已经成功，系统不应重复创建。

## 13.8 代码环境：从 issue 到可验证 patch

给 Agent 一个 issue、仓库 commit 和测试命令，并不等于给了它一个普通问答题。环境需要定义：

- 初始 commit 和依赖锁定；
- 用户已有未提交改动；
- 可以读取和修改的路径；
- 允许执行的命令及资源额度；
- 公开测试、隐藏测试和禁止触碰的文件；
- 最终 diff、测试结果和副作用的验收方式。

一个典型轨迹是：Agent 先执行只读目录和搜索动作，观察到错误调用链；然后读取相关文件，形成 patch；测试工具返回失败堆栈；Agent 根据堆栈修正 patch，最后运行完整测试并比较工作区状态。每一次 action 都可能改变下一步的可行空间，因此代码 Agent 的能力不能只由第一次生成的代码衡量。

### 13.8.1 “测试通过”为什么仍然不够

代码环境的成功谓词可以写成：

~~~math
G_{\mathrm{code}}(\tau)
=\mathbf{1}[\mathrm{hidden\_tests\_pass}]
\land\mathbf{1}[\mathrm{forbidden\_files\_unchanged}]
\land\mathbf{1}[\mathrm{user\_changes\_preserved}]
\land\mathbf{1}[\mathrm{no\_policy\_violation}]
~~~

若 Agent 修改了测试文件、读取了秘密目录，或者覆盖了用户已有改动，测试通过也不能算安全成功。隐藏测试并不是为了神秘化，而是防止策略只针对公开样例；同时，隐藏测试自身也要经过版本管理和污染审计，不能把答案泄露给 observation。

这里的每一个 indicator 都需要对应的观测记录：hidden_tests_pass 只有在隐藏测试实际运行并得到明确结果时才可取值；测试超时、依赖安装失败或执行器崩溃时应标为 inconclusive。forbidden_files_unchanged 和 user_changes_preserved 也必须比较基线快照与最终快照，不能由模型的说明代填。

一个经过验证的 patch 应绑定基线 B、实际 diff Δ、检查结果 T、允许范围 S 和副作用记录 E：

~~~math
\mathrm{ValidatedPatch}=(B,\Delta,T,S,E)
~~~

测试环境故障可以让 T 变成 inconclusive，但不能被说明成“代码已验证”。这一区分对于 Code Agent 的项目复盘尤其重要：生成了代码、代码能编译、目标行为通过测试、用户可以安全合并，是四个不同断点。

## 13.9 浏览器环境：页面状态不等于业务状态

浏览器环境的页面不断变化：资源加载可能失败，元素位置可能变化，登录状态可能过期，点击之后也可能存在异步提交。Agent 可以通过 API、DOM、accessibility tree 或视觉画面观察，但这些通道回答的问题不同。

| 通道 | 更适合观察 | 盲点 | 更可靠的验证 |
|---|---|---|---|
| API | 业务对象、字段、版本和错误码 | 覆盖不了没有 API 的旧页面 | 重新读取对象和版本 |
| DOM | 元素层级、属性和表单值 | 隐藏元素、异步渲染、遮挡 | 回读业务对象 |
| accessibility tree | 角色、名称、控件状态 | 画布和视觉线索可能缺失 | 检查角色状态和页面反馈 |
| 视觉画面 | 布局、弹窗、图标和坐标关系 | 识别误差、坐标漂移、遮挡 | 截图与结构化状态结合 |

打开正确 URL 只能说明导航成功。若任务是把工单改成“已处理”，真正的成功还要检查目标工单 ID、服务端状态、审计 actor、版本和禁止动作记录。页面上的“保存成功”是 observation，不是最终事实。

### 13.9.1 浏览器写操作的两阶段确认

高影响浏览器动作可以分成预览和提交两个阶段。预览阶段展示目标对象、变更字段、收件人或金额；策略层和用户确认这一组参数；提交阶段使用短期确认 token，并在提交后回读业务状态。页面内容本身不能修改确认 token 的范围。

点击后网络中断时，系统应先查询业务对象，而不是根据页面是否回到上一页决定重试。若没有可查询的业务标识，就不能安全区分成功和失败，应停在未知状态并交给人工。浏览器自动化的“像人一样点击”不能绕过服务端的幂等、权限和审计要求。

## 13.10 数据库和文件环境：一致性比文本完成更重要

数据库 Agent 经常同时面对查询、事务、锁、版本和外部事件。只读查询可以在快照中执行；写入则要定义事务范围、隔离级别、超时、回滚和提交后的验证。对数据修复任务，文件存在不等于 schema 正确，行数变化也不等于业务约束满足。

例如，Agent 要把缺失的客户地区补齐。环境可以要求它先生成变更预览，检查受影响行数和数据来源，执行事务，再运行约束查询和抽样比对。若事务提交后数据库连接超时，不能再次执行相同更新；应根据事务 ID 或版本回读结果。

文件环境也有类似问题。写入一个报告需要检查目标路径、编码、格式、权限和 checksum；如果目标文件原本存在，应明确覆盖、追加还是创建新版本。workspace snapshot、diff 和 artifact verifier 必须共同参与成功判断。

## 13.11 reset：任务结束后把世界带回起点

reset 不是测试框架里一个方便的按钮，而是环境契约的一部分。Gymnasium 的公开环境 API 把 reset、step、observation 和终止信号作为环境交互的基础接口；在 Agent 评估中，还需要为业务数据、登录会话、文件、进程和外部 mock 补充更严格的清理语义。

一次 reset 至少应说明：

1. 哪些状态必须恢复到初始快照；
2. 哪些随机种子、时间和网络响应要固定；
3. 哪些不可逆副作用根本不允许发生；
4. reset 失败怎样被报告；
5. 下一次任务是否可以继续使用该环境。

可以把 reset 有效性写成：

~~~math
\mathrm{ResetOK}
=\mathbf{1}[\mathrm{files\_clean}]
\land\mathbf{1}[\mathrm{db\_clean}]
\land\mathbf{1}[\mathrm{session\_clean}]
\land\mathbf{1}[\mathrm{process\_clean}]
\land\mathbf{1}[\mathrm{external\_mock\_clean}]
~~~

每个指示量都只在对应状态摘要确实可取得时才有定义。若数据库无法连接、外部 mock 的清理结果无法查询，ResetOK 应为 unknown，而不是因为“没有发现残留”就记为 1。实际系统未必能把所有项都恢复为零差异，但必须知道哪些项可逆、哪些项需要新环境、哪些项需要人工核对。reset 失败不能隐藏在下一条任务的失败率里，否则评估样本不再独立。

### 13.11.1 reset 正确性的测试

可以在任务开始和结束时分别保存文件、数据库、cookie、进程、队列、缓存和模拟服务的状态摘要。运行一个故意修改多个资源的探针任务，然后调用 reset，比较结束摘要与初始摘要。只测试“正常任务后 reset”还不够，还要测试异常退出、工具超时、Agent 被杀死和外部服务返回 unknown 的情况。

对于不可逆副作用，教学环境应使用模拟服务、事务回滚、临时账号或带 canary 的隔离环境。生产环境的真实邮件、支付、删除和权限变更不能为了 benchmark 成功率而直接放开；它们需要明确的人工确认、审计和补救路径。

## 13.12 沙箱：隔离既是安全条件，也是公平条件

研究环境通常使用容器、虚拟机或临时 workspace 隔离文件、进程、网络、密钥和资源额度。沙箱的价值不只是防止 Agent 破坏宿主机，也在于让不同 Agent 在相同的工具、依赖和网络条件下比较。

一个环境的资源契约可以列出：

| 资源 | 需要固定的属性 | 典型失败 |
|---|---|---|
| 文件系统 | 根目录、可写路径、磁盘额度 | Agent 读到其他任务文件 |
| 网络 | 域名白名单、代理、带宽和超时 | 一个模型可外搜，另一个不可 |
| 凭据 | 身份、作用域、过期时间 | 测试读取生产密钥 |
| 进程 | CPU、内存、子进程和时限 | 测试无限挂起 |
| 时间 | 时钟、时区、随机种子 | 依赖当前日期的任务漂移 |

沙箱并不自动保证安全。容器配置、挂载、日志、缓存和依赖安装仍可能形成出域路径；环境评估还要测试拒绝动作、资源耗尽和异常退出。对于含不可信文档的任务，数据内容与控制平面也要隔离。

## 13.13 环境反馈：类型比数量重要

编译器的错误通常结构清晰，网页上的“操作失败”却可能同时包含权限、网络、业务规则和页面过期。若把所有错误都映射成同一个负反馈，Agent 很难学会应该修复哪一层。

工具返回至少应区分：

- 参数错误：action schema 或字段语义不合法；
- 权限拒绝：身份不能访问目标资源；
- 业务失败：请求合法，但不满足业务前置条件；
- 可重试故障：服务暂时不可用，重试有边界；
- 过期观察：依据的版本已失效；
- 未知执行状态：无法判断外部副作用是否发生。

这些类别会影响下一步 action。参数错误通常需要修正参数，权限拒绝需要换成可授权的只读路径或请求授权，过期观察需要重新读取，未知状态需要查询幂等键。把它们都称为“工具失败”会迫使模型盲目重试，增加成本和副作用。

### 13.13.1 反馈不是天然的学习标签

环境返回的 reward 可能有 grader 错误、覆盖不足和奖励投机。公开测试通过不等于需求完成，页面文字匹配也不等于业务状态改变。训练或评估中使用 reward 时，应记录它由哪些检查组成、是否有隐藏条件、是否可能被 action 直接操纵。

一个好的反馈事件同时包含结果、原因、可重试性、外部状态和建议的下一观察，而不是只给一个数字。这样既能帮助控制器选择动作，也能让复盘者区分模型问题、工具问题和环境问题。

## 13.14 部分可观测性：模型必须管理不确定性

AgentWorld 中，模型每次只能看到状态的一部分。旧网页、并发更新、权限过滤和外部事件都可能使 observation 与真实状态不同步。策略应把 observation 当成带来源和时间的证据，而不是无条件的事实。

在无法确定外部写入是否完成时，至少要保留三个状态：confirmed_success、confirmed_failure 和 unknown。unknown 的价值在于阻止系统把“没有收到响应”误判成“没有产生副作用”。对于高风险动作，宁可停止并请求人工，也不能为了提高自动完成率把 unknown 强行压成 failure 后重试。

如果环境允许模型读取版本和事件序列，模型可以通过重新观察缩小不确定性；如果环境没有这些接口，任务成功率低可能是环境可观测性不足，而不是模型推理不足。评估报告要把观察信息量和恢复接口写明白。

## 13.15 轨迹：记录事实链，而不是保存更多聊天

一次可审计轨迹至少可以表示为：

~~~~text
observation -> action_request -> policy_result -> tool_result
           -> state_diff -> verifier_result -> next_observation
~~~~

其中 action_request 是模型提出的请求，policy_result 是独立执行层的决定，tool_result 是工具执行反馈，state_diff 是环境状态变化，verifier_result 是对目标的验收。缺少其中任何一项，都可能把模型自报误认为外部事实。

一个事件记录应带 trace_id、时间、主体、环境版本、来源版本、动作参数摘要、结果类别和验证状态。敏感原文不必无限期复制到日志；可以保存受权限保护的引用、hash 和最小字段。trace 的目标是重建因果链，而不是收集一段无法检索的长对话。

## 13.16 外部 verifier：成功必须回到环境

不同环境需要不同 verifier：

| 环境 | 主要 verifier | 不足之处 |
|---|---|---|
| 代码仓库 | 编译、测试、diff、依赖和路径检查 | 测试覆盖不足会漏错 |
| 浏览器业务 | 目标对象、版本、审计 actor 和副作用查询 | 页面反馈可能延迟 |
| 数据库 | 事务状态、约束、checksum 和抽样重算 | 读写隔离可能隐藏并发问题 |
| 合同助手 | claim—证据映射、工单状态和权限审计 | 规则冲突需要人工判断 |

最终文本可以作为一个 artifact 被检查，但不能作为唯一 verifier。一个空操作 Agent 也可以输出“任务已完成”；如果验收器只读这句话，它与真正执行并验证的轨迹无法区分，评估就失去了意义。

## 13.17 轨迹质量的分层验收

任务成功率之外，还应检查：

- observation 是否完整、及时且在权限范围内；
- action 是否满足 schema、业务前置条件和策略；
- 工具是否返回了可解释的结果；
- 环境状态是否按预期变化；
- verifier 是否真的读取了外部状态；
- 成本、延迟和重试是否可接受；
- reset 是否让下一条任务保持独立。

一条硬约束式的表达可以是：

~~~math
G(\tau)
=G_{\mathrm{observation}}
\land G_{\mathrm{action}}
\land G_{\mathrm{state}}
\land G_{\mathrm{verifier}}
\land G_{\mathrm{reset}}
~~~

在单条业务任务中，reset 可能是实验有效性条件而不是业务成功条件；报告时应分别列出，避免把环境故障错误归因给 Agent。

### 13.17.1 失败第一分歧点

复盘轨迹时，先找“正确轨迹和实际轨迹第一次不同的事件”。如果 Agent 读取了过期版本，这是观察新鲜度问题；如果 observation 正确但参数错误，这是 action 语义问题；如果参数正确却被错误权限拒绝，这是执行器问题；如果外部状态改变但 verifier 没有回读，这是验收问题。

第一分歧点比最后的失败文本更有用，因为后续动作往往是早期错误的连锁反应。它还能帮助比较不同模型：一个模型可能偶尔犯第一步错误但能恢复，另一个模型可能第一步正确却在未知状态下重复写入，二者的安全画像不同。

## 13.18 如何公平比较模型、工具和环境

一个 Agent 的最终分数可能由模型、工具和环境共同产生。至少应设置以下对照：

| 对照 | 保持不变 | 想回答的问题 |
|---|---|---|
| 无工具 | 任务和模型尽量相同 | 环境行动是否是主要瓶颈 |
| 受限工具 | 初始状态、预算和权限相同 | 哪类工具带来增益 |
| 完整工具 | 与受限组同一 harness | 工具覆盖与副作用代价 |
| oracle 辅助 | 只替换一个检索/验证组件 | 瓶颈在检索、规划还是执行 |

若完整工具显著提升成功率，不能直接写成“模型推理能力提高”；它说明系统得到了更强的行动接口。若 oracle verifier 只提高候选选择，说明生成可能不是唯一问题。每组都应固定模型版本、上下文预算、超时、重试、网络、随机种子、初始状态和 grader 版本。

### 13.18.1 结果报告至少要拆开的指标

至少报告最终成功率、部分成功分布、动作合法率、观察使用率、恢复率、P50/P95 延迟、token/工具成本、外部副作用和 reset failure。多 Agent 还要和单 Agent baseline 在相同或可解释的预算下比较。

不要只报告平均值。长尾任务可能在 P95 上出现大量重复提交；安全任务可能平均成功率很高，但少量越权动作不可接受。指标应按任务类型、权限、环境难度和错误类别切片，并提供失败样本和轨迹。

## 13.19 环境能力与模型能力的归因

一个任务成功，不等于模型拥有环境提供的全部能力。可用以下问题拆分归因：

- 模型是否能在没有工具时形成正确计划？
- 工具是否把关键搜索、解析或修复自动完成？
- observation 是否泄露了隐藏状态？
- verifier 是否过于宽松？
- 初始数据是否已经包含答案线索？
- 权限是否比真实工作流更宽？

归因实验的核心是只改变一个因素。比如代码任务保留相同 commit 和隐藏测试，只替换工具搜索能力；浏览器任务保留相同账号和页面，只改变 observation 是截图还是结构化状态；合同任务保留同一证据库，只改变是否允许第二来源检索。这样才能把“更强模型”“更强工具”和“更容易环境”分开。

## 13.20 长任务：checkpoint、租约和人工接管

长周期 Agent 的状态不能只放在上下文窗口里。checkpoint 应保存目标版本、已完成子任务、证据引用、工具结果、外部对象版本、待处理动作、预算和恢复原因。恢复时先检查外部版本是否仍然匹配，再继续计划；如果不匹配，应重新观察，而不是盲目从旧文本接着执行。

需要写入外部系统时，可以用租约限制执行时间和主体范围，用幂等键避免重复，用回滚或补偿动作处理可逆变化。对于支付、删除、发信和权限变更等不可逆动作，人工确认应绑定具体资源、参数和版本，不能用一次笼统同意覆盖后续变化。

checkpoint 自身也属于外部状态。它可能过期、被并发任务覆盖或包含过多敏感数据。恢复流程要记录 checkpoint revision、创建者、过期时间和受保护字段，并在任务结束或删除请求时处理其生命周期。

## 13.21 环境版本和复现契约

环境镜像、依赖、浏览器版本、数据库快照、工具 schema、observation filter、权限策略、grader、reset 脚本和模型 revision 都是评测输入。任务记录至少要绑定：

~~~~json
{
  "environment_revision": "env-2026-08-14.3",
  "initial_state_hash": "sha256:...",
  "tool_schema_revision": "tools-v9",
  "observation_schema_revision": "obs-v4",
  "policy_revision": "policy-v12",
  "grader_revision": "grader-v6",
  "model_revision": "model-build-17",
  "reset_result": "clean"
}
~~~~

任何一项变化都可能改变任务难度。环境升级后成功率变化，首先要检查页面、依赖、隐藏测试、时间和 reset 是否变化；只有环境契约稳定，才能把差异归因于模型或控制器。公开 benchmark 的分数只能支持其公开 harness 条件下的结论，不能直接外推到生产系统。

## 13.22 常见环境漏洞和错误归因

**reset 泄漏**：前一个任务的文件、cookie 或数据库记录进入后一个任务，导致样本不独立。

**隐藏状态泄漏**：observation 包含隐藏测试答案、内部字段或不应可见的权限，分数高估能力。

**grader 过宽**：只检查字符串、URL 或公开测试，没有检查目标对象、版本、路径、权限和副作用。

**反馈失真**：权限拒绝、参数错误、网络超时和业务失败被压成同一个类别，模型无法恢复。

**工具权限不对称**：不同模型使用了不同网络、搜索、上下文或重试配置，比较失去意义。

**环境未记录**：没有初始状态 hash、依赖版本、时间、随机种子和 verifier 版本，结果无法重放。

**外部副作用未隔离**：发送邮件、修改订单或写生产数据库被直接用于评测，既不安全也无法重复。

**状态变化未验证**：页面提示成功、命令返回 0 或模型总结完成，被错误地当成业务对象已满足目标。

每一类漏洞都对应不同修复位置。换更强模型不能修复 reset 泄漏；增加重试不能修复错误 grader；扩大权限不能代替缺失的业务状态查询。

## 13.23 数字例子：把成功率拆成可诊断的链条

设 100 个代码任务中，90 个 patch 可以应用，75 个通过公开测试，60 个通过隐藏测试，其中 3 个修改了禁止文件，另外 2 个覆盖了用户已有改动。若最终要求隐藏测试通过、禁止文件未变更、用户改动保留，则安全成功数最多是：

~~~math
N_{\mathrm{safe}}=60-3-2=55
~~~

这里没有假设各项独立；它只是把最终结果按验收条件拆开。若把 60 个隐藏测试通过直接报告为成功，就会把策略违规和用户数据破坏隐藏掉。

在更抽象的分析中，可以写成：

~~~math
P_{\mathrm{safe\ success}}
\approx
P_{\mathrm{apply}}
P_{\mathrm{verify}}
P_{\mathrm{policy\ pass}}
P_{\mathrm{artifact\ valid}}
~~~

这些概率在真实任务中通常相关，不能简单相乘得到精确预测，但分解能帮助定位瓶颈。apply 低通常指向 patch 格式或工具问题，verify 低可能指向实现和测试问题，policy pass 低指向权限/数据流问题，artifact valid 低则可能指向环境或清理契约。

再看一个业务例子：100 次工单任务中，92 次找到合同，80 次找到正确政策版本，70 次规则判断有充分证据，65 次创建请求被接受，63 次重新读取后状态正确。报告应同时给出这些中间比例，不能只说最终成功率为 63%。

## 13.24 最小可运行的环境闭环审计 demo

下面的 demo 不调用外部模型，也不模拟真实沙箱；它用一个小型 workspace 展示四个事实：合法 patch 可以通过测试，越界 patch 会在执行层被拒绝，最终成功依赖 verifier 而不是模型自报，reset 能把文件恢复到基线。

~~~~python
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional


def require_text(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


@dataclass(frozen=True)
class Action:
    kind: str
    path: Optional[str] = None
    content: Optional[str] = None

    def __post_init__(self):
        require_text(self.kind, "kind")
        if self.kind == "patch":
            require_text(self.path, "path")
            if not isinstance(self.content, str):
                raise TypeError("content must be text for patch")
        elif self.kind == "test":
            if self.path is not None or self.content is not None:
                raise ValueError("test does not accept path or content")
        else:
            raise ValueError(f"unsupported action kind: {self.kind}")

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict):
            raise TypeError("action must be a dict")
        allowed = {"kind", "path", "content"}
        unexpected = set(raw) - allowed
        if unexpected:
            raise ValueError(f"unexpected action fields: {sorted(unexpected)}")
        if "kind" not in raw:
            raise ValueError("action requires kind")
        return cls(
            kind=raw["kind"],
            path=raw.get("path"),
            content=raw.get("content"),
        )


@dataclass
class Workspace:
    baseline: dict
    files: dict = field(init=False)
    tests_pass: Optional[bool] = None
    policy_violation: bool = False
    events: list = field(default_factory=list)
    _seen_trace_ids: set = field(init=False, repr=False)
    _active_trace_id: Optional[str] = field(init=False, default=None, repr=False)
    _event_counter: int = field(init=False, default=0, repr=False)

    def __post_init__(self):
        if not isinstance(self.baseline, dict):
            raise TypeError("baseline must be a dict")
        required = {"parser.py", "tests.py", "secret.txt"}
        if set(self.baseline) != required:
            raise ValueError("baseline must contain exactly the teaching workspace files")
        for path, content in self.baseline.items():
            require_text(path, "baseline path")
            require_text(content, f"baseline[{path}]")
        self._seen_trace_ids = set()
        self.reset()

    def reset(self):
        self._active_trace_id = None
        self.files = dict(self.baseline)
        self.tests_pass = None
        self.policy_violation = False
        self.events.clear()

    def start_episode(self, trace_id):
        require_text(trace_id, "trace_id")
        if self._active_trace_id is not None:
            raise RuntimeError("an episode is already active")
        if trace_id in self._seen_trace_ids:
            raise ValueError(f"duplicate trace_id: {trace_id}")
        self._seen_trace_ids.add(trace_id)
        self._active_trace_id = trace_id

    def _record(self, event, **fields):
        if self._active_trace_id is None:
            raise RuntimeError("actions require an active episode")
        self._event_counter += 1
        record = {
            "event_id": f"{self._active_trace_id}:{self._event_counter}",
            "trace_id": self._active_trace_id,
            "event": event,
        }
        record.update(fields)
        self.events.append(record)

    def snapshot_hash(self):
        payload = "\n".join(
            f"{path}={self.files[path]}" for path in sorted(self.files)
        )
        return sha256(payload.encode()).hexdigest()[:12]

    def observe(self):
        visible = {
            path: self.files[path]
            for path in ("parser.py", "tests.py")
        }
        return {
            "visible_files": visible,
            "tests_pass": self.tests_pass,
            "source_revision": self.snapshot_hash(),
        }

    def act(self, action):
        try:
            action = Action.from_mapping(action)
        except (TypeError, ValueError) as error:
            self._record("action_rejected", reason="invalid_schema")
            return {"status": "rejected", "reason": str(error)}

        kind = action.kind
        if kind == "patch":
            path = action.path
            if path != "parser.py":
                self.policy_violation = True
                self._record(
                    "action_rejected",
                    path=path,
                    reason="path_out_of_scope",
                )
                return {"status": "rejected", "reason": "path_out_of_scope"}
            self.files[path] = action.content
            self.tests_pass = None
            self._record("patch_applied", path=path)
            return {"status": "ok"}

        if kind == "test":
            self.tests_pass = (
                self.files["parser.py"] == "fixed"
                and self.files["tests.py"] == "baseline"
            )
            self._record(
                "tests_finished",
                passed=self.tests_pass,
            )
            return {"status": "ok", "passed": self.tests_pass}

        raise AssertionError("Action validation should reject unknown kinds")

    def verify(self):
        forbidden_unchanged = self.files["tests.py"] == self.baseline["tests.py"]
        if self.policy_violation or not forbidden_unchanged or self.tests_pass is False:
            status = "failed"
        elif self.tests_pass is None:
            status = "unknown"
        elif self.files["parser.py"] != "fixed":
            status = "failed"
        else:
            status = "passed"
        return {
            "status": status,
            "success": status == "passed",
            "last_event": self.events[-1]["event"] if self.events else None,
            "state_hash": self.snapshot_hash(),
            "events": list(self.events),
        }


def run_episode(environment, actions, trace_id):
    if not isinstance(actions, (list, tuple)):
        raise TypeError("actions must be a list or tuple")
    environment.start_episode(trace_id)
    for action in actions:
        result = environment.act(action)
        if result["status"] == "rejected":
            break
    return environment.verify()


environment = Workspace({
    "parser.py": "bug",
    "tests.py": "baseline",
    "secret.txt": "hidden",
})

good = run_episode(environment, [
    {"kind": "patch", "path": "parser.py", "content": "fixed"},
    {"kind": "test"},
], "trace-good")
good_hash = good["state_hash"]

environment.reset()
bad = run_episode(environment, [
    {"kind": "patch", "path": "secret.txt", "content": "leaked"},
    {"kind": "test"},
], "trace-bad")

environment.reset()
reset_clean = (
    environment.files == environment.baseline
    and environment.tests_pass is None
    and environment.policy_violation is False
)

print(f"good_success={good['success']}")
print(f"bad_success={bad['success']}")
print(f"bad_last_event={bad['last_event']}")
print(f"good_hash_recorded={len(good_hash) == 12}")
print(f"reset_clean={reset_clean}")
~~~~

运行结果应为：

~~~~text
good_success=True
bad_success=False
bad_last_event=action_rejected
good_hash_recorded=True
reset_clean=True
~~~~

边界测试应证明输入错误不会悄悄改变 workspace，也不会把没有验证过的轨迹算作成功：

~~~~python
environment.reset()
empty = run_episode(environment, [], "trace-empty")
assert empty["status"] == "unknown"
assert empty["success"] is False
assert environment.files == environment.baseline

environment.reset()
invalid_kind = run_episode(environment, [{"kind": "delete"}], "trace-invalid-kind")
assert invalid_kind["status"] == "unknown"
assert invalid_kind["last_event"] == "action_rejected"
assert environment.files == environment.baseline

environment.reset()
empty_path = run_episode(
    environment,
    [{"kind": "patch", "path": "", "content": "fixed"}],
    "trace-empty-path",
)
assert empty_path["last_event"] == "action_rejected"
assert environment.files == environment.baseline

environment.reset()
non_text = run_episode(
    environment,
    [{"kind": "patch", "path": "parser.py", "content": 1}],
    "trace-non-text",
)
assert non_text["last_event"] == "action_rejected"
assert environment.files == environment.baseline

environment.reset()
try:
    run_episode(environment, [], "trace-good")
except ValueError as error:
    assert "duplicate trace_id" in str(error)
else:
    raise AssertionError("duplicate trace IDs must be rejected")

assert len(good["state_hash"]) == 12
assert len(bad["state_hash"]) == 12
event_ids = [
    event["event_id"]
    for result in (good, bad)
    for event in result["events"]
]
assert len(event_ids) == len(set(event_ids))
assert environment.events == []
~~~~

这个 demo 的 verifier 只检查一个教学构造的文件状态，不能证明真实代码正确，也没有实现容器隔离、并发、网络或隐藏测试。它的价值在于把“写入—测试—拒绝—reset—验收”变成可以重放的事件序列。扩展到真实系统时，还要把用户已有改动、命令副作用、依赖版本和外部服务状态纳入记录。

## 13.25 从 demo 回到真实环境

demo 中的 parser.py 相当于一个受控 artifact，tests.py 相当于一个简化的 verifier，secret.txt 相当于不能触碰的资源。真实 Code Agent 至少还需要：

- 保存修改前 workspace 的快照和用户变更；
- 约束每条命令的目录、网络、进程和资源；
- 分离目标测试、回归测试和环境故障；
- 检查依赖、生成文件和无关 diff；
- 在测试超时或外部调用不确定时查询状态；
- 把最终 claim 绑定到实际 diff 和测试结果。

同样的映射可以用于浏览器和数据库。浏览器的业务对象相当于文件，服务端状态查询相当于 verifier，页面截图相当于 observation；数据库的事务和约束相当于环境转移和成功谓词。抽象相同并不意味着实现相同，资料和实验必须说明具体环境。

## 13.26 学习任务：把抽象落实成环境契约

1. 为一个代码修复任务写出初始 commit、可见 observation、可调用 action、隐藏状态、reset 和 success schema。
2. 设计一个浏览器任务，分别写出 URL 成功、页面成功和业务对象成功的判据。
3. 构造一次数据库写入超时的轨迹，说明为什么不能直接重试，以及需要哪个查询接口消除 unknown。
4. 对同一个任务设计无工具、受限工具、完整工具和 oracle verifier 四个对照，列出固定变量和可变变量。
5. 为一个长周期合同任务设计 checkpoint schema，包含证据版本、外部对象版本、幂等键、租约和人工接管原因。
6. 阅读本章 demo，增加一个“用户已有改动被保护”的断言，并说明它属于 artifact 验收还是策略验收。

这些练习的目的，是让读者从“模型会不会做”转向“环境如何证明做成了什么”。答案应包含状态、证据、失败和恢复，而不是只写一段 Agent 提示词。

## 13.27 资料入口与可信边界

- [Gymnasium Environment API](https://gymnasium.farama.org/api/env/)：官方环境接口文档，支持 reset、step、observation 和终止信号的接口语义；它不替读者定义业务权限、隐藏状态或生产安全。
- [Sutton 与 Barto 的 Reinforcement Learning 电子版](http://incompleteideas.net/book/the-book-2nd.html)：MDP、策略、回报和强化学习的经典教材入口；书中的奖励模型不能直接等同于 Agent 业务成功。
- [AgentBench 论文](https://arxiv.org/abs/2308.03688)：多环境 Agent 评估的论文入口，可支持任务环境、动作和结果评估的讨论；其分数依赖具体环境和 harness。
- [WebArena 论文](https://arxiv.org/abs/2307.13854) 与 [项目页](https://webarena.dev/)：真实网站交互环境和任务评测入口；不能把网页任务分数直接外推为所有浏览器操作能力。
- [OSWorld 论文](https://arxiv.org/abs/2404.07972) 与 [项目页](https://os-world.github.io/)：真实计算机环境中的多模态操作评估入口；其结果受桌面镜像、应用版本、观察通道和 grader 影响。
- [SWE-bench 论文](https://arxiv.org/abs/2310.06770) 与 [项目页](https://www.swebench.com/)：软件工程 issue 修复环境入口；通过率依赖仓库版本、测试、补丁应用和评估 harness。
- [BrowserGym 论文](https://arxiv.org/abs/2407.06903)：网页 Agent 环境和任务定义的研究入口；它支持环境设计讨论，不证明任何闭源模型的内部机制。
- [dm_env 接口](https://github.com/google-deepmind/dm_env)：DeepMind 公开的环境接口实现入口，可用于理解 step、observation、reward 和终止的工程表达；它不是本章业务安全策略的唯一规范。
- [OpenAI Evals](https://github.com/openai/evals)：评估框架入口；运行框架本身不能替代外部状态 verifier、权限检查和 reset 审计。

这些资料分别支持经典环境抽象、公开接口、网页/桌面/代码 benchmark 或评估框架。论文中的实验结论、官方 API 的接口语义、本章的教学 demo 和目标生产系统的实测结果属于不同证据层级，不能互相冒充。使用公开 benchmark 时，应同时记录模型版本、提示、工具、观察空间、预算、网络、初始状态、grader 和 reset 版本。

## 13.28 小结

AgentWorld 把“生成一个答案”扩展成“在一个受约束的世界中观察、行动、处理反馈、恢复不确定性，并让环境达到可验证目标”。状态、observation、action、转移、权限、verifier 和 reset 共同决定了任务究竟测到了什么。

读者可以先记住一句话：模型说完成了，不等于世界已经完成。进一步的工程审查还要追问：环境给了模型哪些信息，动作改变了哪些真实状态，失败是否可恢复，成功是否由外部 verifier 证明，实验结束后 reset 是否真的清理，以及结论是否只在当前 harness 条件下成立。

只要这些问题被写进环境契约，Agent 的能力才不再是对一段流畅文本的印象，而是对一条可复核、可重放、可解释轨迹的工程判断。
