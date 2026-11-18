# 第 17 章 Long-Running Agent：任务暂停后怎样安全续跑

一个代码 Agent 接到任务：把 `parser` 依赖从 2.7 升到 3.1，修改兼容代码，跑完集成测试，再申请创建代码审查。测试需要几十分钟；中途 worker 可能重启，用户也可能先离开。真正难的不是让模型“记住昨天聊过什么”，而是重启后知道哪些事实已经成立、哪些动作还没有发生，以及哪些动作可能已经发生但回执丢了。

本章沿着一项明确标为教学构造的 `task-42` 代码任务，逐步设计任务检查点（checkpoint）、外部动作账本和恢复路径。读者进入时应理解工具提议、执行回执、工作区和任务验收的区别；离开时应能为一次中断判断可否重试、何时需要查询外部事实、何时必须保留未知状态或交给人工。基础运行时状态见[第十五章](15-agent-harness系统设计.md)，运行事件与轨迹见[第九章](09-trace日志回放与可观测性.md)。

## 17.1 中断发生在一次真实任务的中间

先看任务运行到一半时留下的事实。初始代码仓库是 `repo@r41`，用户要求不要修改测试文件；任务完成局部改动后，工作区来到 `r42`。Runtime 启动了集成测试 `ci-884`，CI 已经完成并返回成功，但 worker 在把结果写进自己的任务记录前崩溃。第二天用户恢复任务时，CI 任务仍可按 `ci-884` 查询，不必再启动一份测试。

稍后用户批准申请代码审查。执行器向代码托管服务发出创建请求，服务已经创建了 `review-481`，但响应在回到 worker 前丢失。本地记录只有“请求已发出”，没有成功回执。此时，系统不知道操作未发生，还是已经发生；再发一个新的创建请求可能得到第二个审查项。

这两次中断看起来相似，恢复方法却不同：

| 中断点 | 已知事实 | 下一步为什么不同 |
|---|---|---|
| CI 完成，任务记录未更新 | 有稳定的 CI job ID `ci-884`；CI 可查询其完成状态和结果 | 先查原 job，再把结果补进任务记录；不要新建 job |
| 远端审查已创建，响应丢失 | 本地没有成功回执；请求可能已改变远端状态 | 标记为 `unknown`，用原操作身份查询或重试；不能生成新身份直接再建 |

这里的 `task-42`、`r41/r42`、`ci-884` 和 `review-481` 都是教学输入，不代表某个真实产品或服务的保证。例子要说明的是恢复时的证据关系：模型说“已经创建”不是服务端回执，worker 没有收到回执也不是“肯定没有创建”。

## 17.2 四类记录不能互相代替

恢复任务时至少会遇到四种不同记录。

| 记录 | 回答的问题 | 不能单独证明什么 |
|---|---|---|
| 对话历史或模型上下文 | 用户和模型说过什么，模型上次看见什么 | 文件是否仍是那个版本，工具是否真实执行，外部操作是否成功 |
| 事件记录或 Trace | 哪些提议、派发、回执和状态变化按什么顺序发生 | 记录本身是否已持久化，事件里提到的外部状态是否仍然存在 |
| Checkpoint | 恢复程序可以从哪个已保存的任务状态继续 | 远端副作用已回滚，或者工作区仍保持相同版本 |
| 工作区和外部服务 | 文件、CI、工单或审查服务当前实际保存什么 | 这些状态是否满足用户的全部验收要求 |

因此，checkpoint 是可恢复任务状态的一种持久化表示，不是完整聊天备份，也不是所有系统状态的快照。Trace 负责说明运行过程中观察到的事件；工作区和外部服务提供当前事实。它们可能由相同数据库承载，但仍须保留不同语义和更新时机。

第十八章专门讲怎样压缩送给模型的上下文。Context Folding 可以减少 token，但不能代替本章的任务状态持久化。第二十一章负责展开 workspace、artifact 和外部系统的所有权；本章只在需要恢复时核对它们，不重复教授文件版本和合并策略。

## 17.3 Checkpoint 要保存哪些恢复依据

一个有用的 checkpoint 不是“把目前想到的事写进摘要”，而是让恢复程序能回答：这是哪个任务、进行到了哪一步、依据的是哪个环境、有没有未决副作用、恢复时还允许做什么？下面是一份与 `task-42` 对应的示意记录：

| 字段组 | 示例值 | 恢复时的作用 |
|---|---|---|
| 任务身份与契约 | `task-42`；升级 `parser`；不得改测试；申请审查需获批准 | 区分任务，并保留目标、硬约束和验收条件 |
| Checkpoint 顺序 | 序号 18，父版本 17 | 防止较旧 worker 用旧状态覆盖新状态；可用条件写入或 compare-and-swap（比较并交换）检查父版本 |
| 当前阶段与游标 | 等待 CI `ci-884` 的结果 | 知道恢复后应查询原 job，而不是重新开始整项任务 |
| 工作区与 Artifact | `workspace@r42`；patch `patch-42`；CI 结果引用 `ci-884` | 将测试和文件证据绑定到它们实际验证的 revision |
| 外部动作账本 | `effect_id=task-42:review:1`；请求摘要；状态 `unknown` | 标识同一个业务动作，保留待核查的回执空缺 |
| 执行版本 | checkpoint schema、workflow、tool adapter、policy 和环境版本 | 判断旧状态能否继续使用，是否需要迁移或重新验收 |
| 预算与 owner | 累计用量、剩余额度、worker 租约代数 18 | 避免每次恢复都重置预算，或两个 worker 同时推进任务 |
| 完整性与来源 | 父版本、写入事务、artifact digest 等校验材料 | 发现截断、损坏或引用不一致；完整性校验不等于内容正确 |

表中不需要机械照抄成任何框架的固定 schema。哪些字段必须出现，取决于恢复逻辑能否据此安全选择下一步。核心是把模型陈述、已观察事实、未验证假设、Artifact 和未决副作用分开；事实要带上来源和 revision，不能把一段自然语言摘要当成唯一权威记录。上下文折叠中怎样保留这些来源和不确定性见[第十八章](18-context-folding上下文折叠.md)。

## 17.4 在派发副作用前，先留下能恢复的意图

Checkpoint 可以原子保存到自己的数据库，却不能与任意第三方 API 自动成为同一个事务。考虑 `review-481`：本地数据库写入成功后，网络请求才到达远端；远端创建完成后，进程可能在保存回执前掉线。两边之间不存在一个由本地 checkpoint 自动提供的共同提交。

下面的顺序把“本地知道要做什么”和“远端可能已经做了什么”分开记录：

| 步骤 | 本地/远端发生什么 | 中断后恢复程序凭什么行动 |
|---|---|---|
| 1. 建立操作身份 | 生成稳定的 `effect_id`，固定请求内容并记录其摘要 | 同一逻辑动作的后续尝试复用该 ID 和同一请求；改变目标或内容要建立新动作 |
| 2. 持久化意图 | 本地事务保存待办 effect 和任务 checkpoint；若另有消息队列，可把 outbox 记录也放进此事务 | 只有持久化成功后，dispatcher 才能发送请求；事务失败时不得派发 |
| 3. 派发请求 | 执行器向外部服务发送请求，并在服务合同支持时携带幂等键 | worker 可崩溃；重试时用同一操作身份，而不是新的随机 ID |
| 4. 观察结果 | 收到服务端回执、查询结果或明确错误 | 没有响应时只能记 `unknown`，除非服务合同能证明请求未执行 |
| 5. 持久化结果 | 保存服务端对象 ID/回执和新的 checkpoint 阶段 | 后续步骤只依据已保存的结果继续；最终验收仍检查真实产物 |

Transactional outbox（事务发件箱）可以让本地业务记录与“待派发”消息在同一数据库事务中提交，减少本地状态和队列消息只成功一边的双写问题。但 outbox relay 或队列仍可能重复派发，接收方仍需识别重复操作；它不会把本地数据库和第三方服务变成同一笔原子事务。[AWS 对 transactional outbox 的说明](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)也明确列出重复消息及幂等消费者这一边界。

HTTP 的幂等语义描述的是同一请求重复应用后，服务端预期效果与应用一次相同；它不等于每次都返回完全相同的响应，也不代表某个 `POST` API 自动具有幂等性。[RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)定义了方法语义。服务提供的幂等键还要核对参数匹配规则、并发请求行为、作用范围和保留期。例如 DynamoDB 的交易 client token 有特定有效期，且同一 token 搭配不同请求参数会触发错误；这只是该服务的合同，不是所有 key 的通用规则。[DynamoDB transactions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html)

在 `task-42` 中，`effect_id` 代表“申请这一次代码审查”。网络重试是对同一个动作再尝试，不是新建第二个业务动作；如果用户改了目标分支或审查内容，就不能偷偷复用同一个 ID 让服务端把新请求误认成原请求。

## 17.5 先判断动作处于什么状态，再决定是否重试

为任务维护一个 `running`/`failed` 状态仍不够，因为一个子动作可能正在远端执行，而整个任务已经被 worker 标为失败。每个有副作用的 effect 都需要有能表达实际证据的状态。

| Effect 状态 | 含义 | 允许的下一步 |
|---|---|---|
| `not_dispatched` | 根据本地派发合同，能证明请求尚未交给执行器 | 可以派发；若派发记录缺失但执行器可能已调用，不能擅自用此状态 |
| `intent_saved` | 本地已记录意图，尚未取得执行器/服务的明确结果 | 检查 outbox 或执行器记录，再决定是否派发 |
| `unknown` | 请求可能已到达服务端，但结果无法确认 | 查询原操作、重用同一 key（若合同允许），或转人工；不可用新 ID 盲目再建 |
| `committed` | 权威回执或查询确认该业务结果已存在 | 保存回执并推进任务；检查与验收条件是否一致 |
| `confirmed_failed` | 服务端明确确认此请求失败且没有产生目标副作用 | 按错误类型决定修复请求、有限重试或终止 |

这些状态必须由运行时的持久记录、执行器回执和外部查询证据推动，不能由模型回答“成功了”或“我没做”触发。恢复 `task-42` 时，系统先查询 `review-481` 是否已创建；若提供方支持同一幂等键的安全重放，则用相同 ID 和相同请求取回既有结果；若没有可查询也没有去重合同，则保持 `unknown`，交给人工核对。是否存在补偿动作也要看具体业务合同，不能把取消任务误称为回滚远端结果。

恢复流程可以按这个顺序执行：

1. 读取最新 checkpoint，检查序号、父版本和数据完整性；用条件写入取得当前任务的所有权。
2. 核对保存时的 workspace revision、Artifact 与后台 job 的真实状态。
3. 查询所有 `unknown` 或尚未回执的外部 effect；没有可靠查询合同就保留未知。
4. 检查当前 schema、工具、策略、用户授权和剩余预算；旧审批不自动授权新的动作。
5. 将可证实的事实、假设、过期信息和未决 effect 分开，再组装交给模型的恢复上下文。
6. Runtime 按状态继续、重新验证、迁移、补偿、请求用户输入或停止；恢复的目标是完成原任务并保持事实一致，不是让模型尽快继续生成。

## 17.6 让旧 worker 不能越过新 owner

假设 worker A 拿到租约代数 17 后停顿，租约到期后 worker B 接管并写入 checkpoint 18。A 随后恢复网络并继续运行。如果存储只检查“是否持有一把未过期锁”，A 有机会用自己的旧状态覆盖 B 的新状态。

恢复 owner 因此需要单调增加的 fencing token（隔离令牌）：A 使用代数 17 的所有 checkpoint 写入和受控派发，在 B 已取得代数 18 后都必须被拒绝。仅靠本地 sleep 或租约超时不足以阻止暂停中的旧进程；检查必须落在实际接受写入或派发请求的边界上。若第三方服务不认识 fencing token，就由自有的 dispatcher/执行器代理检查 owner，再向外发请求。

任务租约与 workspace 文件锁也不是同一个机制。文件并发、revision 条件写入和冲突合并由[第二十一章](21-persistent-workspace与状态边界.md)展开；这里要保留的结论是：checkpoint 恢复不能跳过对当前 owner、workspace 版本和待执行 effect 的核对。

## 17.7 版本变化时，哪些状态可以继续用

长任务可能隔天、隔周恢复，期间 workflow 代码、checkpoint schema、模型、工具接口或权限策略都可能改变。所有版本都写进一张大表并不能自动使它兼容；恢复程序需要针对“下一步会怎么做”检查兼容性。

| 已保存内容 | 常见处理方式 | 重新验证的原因 |
|---|---|---|
| 已确认的外部对象 ID 和原始回执 | 在对象仍可查、来源可信时复用引用 | 服务数据可能被删除、状态可变或保留期已过 |
| 已生成的 patch、测试报告和其他 Artifact | 保留原版本及 digest；确认它对应的 workspace revision | 新 revision、工具或验收条件可能使报告过期 |
| Workflow/Checkpoint schema | 用显式迁移器生成新版本；保留旧记录供追查 | 字段变化可能改变状态含义，不能只靠反序列化成功 |
| 未完成的工具提议和请求参数 | 对照新 schema、策略和当前授权重新校验 | 旧参数可能已不合法，旧审批也可能过期 |
| 模型内部/Provider opaque state | 只在对应 API 和模型合同允许时复用 | 不能假设隐藏推理状态、KV cache 或签名可跨版本迁移 |
| 建立在旧文件/权限状态上的结论 | 读取当前环境并重跑受影响 verifier | 旧测试结果不自动适用于新文件和新策略 |

Temporal 的 Workflow 需要确定性，才能依据已记录的 Event History replay；外部调用等非确定操作放在 Activity 中。Temporal 同时说明，Activity 可能因失败被执行多次，调用方仍需让业务副作用幂等。[Workflow Definition](https://docs.temporal.io/workflow-definition) 与 [Activity Definition](https://docs.temporal.io/activity-definition)描述的是 Temporal 自身的合同。

LangGraph 则提供图状态 checkpointer、线程级 checkpoint 和不同持久化模式；其文档也说明 retry/interruption 会让受影响 node 从函数开头重新运行。具体恢复边界需绑定使用的语言 SDK、版本、持久化后端和 durability 配置，不能从“启用了 checkpoint”推断节点内副作用会跳过。[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) 和 [fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)是滚动更新的框架文档，本段结论限于访问日期和相应 SDK 页面。

## 17.8 检查点频率要跟着任务边界走

每生成一个 token 就存一次，既昂贵，也没有把重要的副作用边界表达清楚。反过来，如果长时间不保存，进程崩溃后可能需要重复推理、重跑测试，甚至无法分辨一项外部动作是否已经发生。

对 `task-42`，值得保存的时点包括：基准 revision 被接受、patch 形成新 revision、CI job ID 已落盘、CI 结果已确认、用户批准已记录、远端审查请求 intent 已提交、远端回执已保存、任务最终验收完成。决定“继续、等待、重试或停止”的状态发生变化时，checkpoint 应在下一项有副作用的工作开始前达到所要求的持久性级别。

频率越高，崩溃后需重做的已知工作越少，但写入、存储和敏感信息保留成本越高。频率越低，系统更轻，却会扩大“上次已做什么”无法确定的窗口。应按任务阶段、外部副作用风险和持久化后端测量值选择，而不是把某个固定秒数当成所有任务的标准。

预算、租约和取消也需要各自的持久语义。恢复不应为每次新 worker 重置累计 token、工具时间、外部请求次数或绝对截止时间；用户取消可以阻止后续步骤，却不一定能撤回已经派发的远端动作。长时间暂停还要明确释放哪些可重算资源、保留哪些 Artifact、何时删除敏感状态。保留和租户范围由[第二十一章](21-persistent-workspace与状态边界.md)继续讨论。

## 17.9 用故障注入验证恢复，而不是只看模型续写

恢复测试要把进程中止放在状态边界上。每次运行都核对任务状态、workspace revision、Artifact 和外部服务记录；如果某个服务返回不了权威结果，就让该样本保持 `unknown`。

| 故障注入位置 | 恢复时应观察什么 | 错误信号 |
|---|---|---|
| 本地事务提交 intent 前 | 是否能证明派发器尚未发送请求 | 没有 intent 却可能已经出现外部对象 |
| intent/outbox 已持久化、尚未收到派发回执 | 原 outbox 是否只对应同一逻辑 effect | 恢复时创建了新的 effect ID，导致双派发 |
| 服务端接受请求、响应返回前 | 通过原 ID 查询、去重或人工对账 | 把超时记成失败，再用新 ID 重复操作 |
| 收到服务端回执、本地 checkpoint 提交前 | 是否能从 provider 查询或同一 key 重取回执 | 业务对象存在，但任务永远停留在“未执行”或重复创建 |
| worker 租约过期且旧 worker 恢复 | checkpoint 与 dispatcher 是否拒绝旧 fencing token | 新旧 worker 都能提交同一任务的下一步 |
| 暂停期间 workspace 或策略发生变化 | 是否识别 revision/策略差异并重验受影响结果 | 把旧测试结果或旧批准直接沿用 |

报告结果时至少拆开三类观察：计划注入的中断数、最终事实已知的样本、仍无法确认的样本。安全恢复率的分母只应来自预先定义的中断任务集合；unknown 单列，不能静默剔除后让比例变好。重复外部 effect 要以能查到真实业务对象的操作为分母；恢复耗时要说明从哪个恢复事件开始，到自动恢复、人工接管或终态哪个时刻结束。这里不把这些观察合成一个任意总分或通用准入门槛。

2026 年 11 月的 AgentFaultBench 预印本提供了一个恢复评估组织样本：它在隔离 GitLab 中注入工具故障，并检查文件、issue、必要的读取/验证步骤和最终回报；作者明确报告没有人工验证。它能说明“检查最终外部状态和任务过程”是一种可用的评估设计，不能证明该预印本的模型差异能推广到生产环境或其他任务。[AgentFaultBench](https://arxiv.org/abs/2611.20711v1) 的细节和范围见研究记录。

## 17.10 框架文档必须读成一份有版本边界的合同

不同工作流系统的“checkpoint”“retry”“resume”可能指不同持久化单元和恢复边界。读取文档时要问：保存的是整条线程、图状态、单个 step 的结果还是 event history？节点从失败处继续，还是从节点开头重跑？并发 resume 如何仲裁？写入是同步确认还是异步排队？副作用由谁去重？

一篇 2026 年 8 月的单作者预印本将 checkpoint / interrupt / resume 拆成多项可测试性质，并对固定版本的多个 workflow framework 做了确定性 probe。它报告的实现问题尚需独立复现，不能当作所有框架的定论；但该研究展示了一个实用的审阅方向：把“可靠恢复”改写成具体可证伪的合同，固定 API、版本、后端和中断位置，再逐项测试。[Resume Means Resume](https://arxiv.org/abs/2608.03836v1)。本章采用的状态检查和故障注入是工程设计建议，不声称复现其框架 probe。

MIT 6.5840 Spring 2026 的课程和 Lab 3/4 也是组织层面的参考：它从节点失败和持久状态开始，再通过 Raft 与容错 key/value 服务把持久化、请求语义和故障测试连起来。该课程面向具备系统基础的研究生；这里只借鉴从“故障发生了什么”走到“恢复后服务必须维持什么语义”的讲法，不把其编程门槛或实验安排照搬到本章。[课程主页](https://pdos.csail.mit.edu/6.824/index.html)、[Lab 3: Raft](https://pdos.csail.mit.edu/6.5840/labs/lab-raft1.html)、[Lab 4: Fault-tolerant Key/Value Service](https://pdos.csail.mit.edu/6.824/labs/lab-kvraft1.html)。

## 17.11 把 checkpoint、恢复、验收连成一个闭环

回到 `task-42`。worker 重启后，Runtime 读取 checkpoint 18，发现 CI job `ci-884` 已完成，于是补记成功结果；随后检查工作区是否仍是 `r42`。当恢复流程遇到 `effect_id=task-42:review:1` 时，它不根据模型摘要判断是否创建审查，而是查询服务端。若找到了 `review-481`，就持久化回执并继续 verifier；若服务确认未创建且重试合同仍有效，才用同一操作身份重试；若不能确定，就保持 `unknown` 并请求人工核对。

最终是否完成，仍由任务验收条件决定：patch 是否只改了允许的文件，CI 结果是否对应当前 workspace revision，审查对象是否已存在且内容正确，用户约束是否满足。Checkpoint 让任务知道从哪儿恢复，effect ledger 保护外部动作，workspace 和 provider 给出当前事实，verifier 才能决定是否交付。

回答“怎样设计一个能跨中断续跑的 Agent”时，可以先讲一条会失败的操作：先保存 task/revision、执行游标和 effect intent，再派发动作；回执丢失时保留 unknown，用稳定 operation ID 查重或对账；恢复前检查租约、权限、预算和版本；最后从当前 workspace/服务状态做独立验收。仅说“把聊天写进数据库”没有回答恢复后该如何区分计划、执行和已提交事实。

## 17.12 复习与练习

1. 延续 `task-42`：CI `ci-884` 的成功结果对应 `r42`，暂停期间用户把工作区改成 `r43`；代码审查创建请求的回执仍是 `unknown`。写出恢复顺序，并区分哪些事实可以复用、哪些验证必须重跑。
2. 假设代码托管服务没有幂等键，也没有按 client reference 查询的 API。说明恢复程序为什么不能自动把 `unknown` 改成“未执行”，并提出一个可接受的人工对账入口和未来应要求的工具合同。
3. 为一项“用户批准后发布包”的长任务画出三个故障注入点：intent 写入前、发布服务接受后而回执丢失、旧 worker 在新 worker 接管后恢复。分别写出所需持久记录、预期恢复动作和能判定最终结果的外部证据。

## 资料与证据范围

- [Temporal Event History 与恢复](https://docs.temporal.io/evaluate/understanding-temporal)、[Workflow Definition](https://docs.temporal.io/workflow-definition)、[Activity Definition](https://docs.temporal.io/activity-definition) 和 [Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)：框架自身的回放与 Activity 重试合同。
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) 与 [Fault Tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)：对应 SDK 当前的 checkpoint、node retry 和 durability 说明；行为随版本、语言 SDK 与后端配置而变。
- [RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)、[AWS transactional outbox](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html) 与 [DynamoDB transactions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html)：HTTP 幂等语义、跨本地数据库/队列双写的处理方式和特定服务 token 合同。
- [AgentFaultBench](https://arxiv.org/abs/2611.20711v1) 与 [Resume Means Resume](https://arxiv.org/abs/2608.03836v1)：单作者 arXiv 预印本，只作为状态核验评估和 checkpoint conformance 测试的研究样本，不视为经过独立复现的普遍结论。
- 来源阅读位置、访问日期、支持范围和未验证边界详见[研究记录](../../research/long-running-agent-checkpoint-recovery-evidence-2026-11-18.md)。本章没有做真实 Agent/第三方服务运行，也没有对任何框架执行恢复 probe。
