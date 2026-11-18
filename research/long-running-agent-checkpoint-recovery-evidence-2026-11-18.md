# Long-Running Agent 的 checkpoint 与恢复证据

检索日期：2026-11-18。记录对象是长时间任务的持久执行、checkpoint / replay、外部副作用的幂等边界和恢复验证。这里记录领域知识与来源证据，不记录章节修改决定或进度。

## 检索范围与方法

- 目标读者问题：进程、连接或任务执行阶段中断后，系统凭什么判断已完成、未开始、仍在运行或结果未知；什么可以重放；如何证明恢复没有重复副作用。
- 检索词包括 `durable execution workflow replay activity retry idempotency`、`LangGraph persistence checkpoint durability fault tolerance`、`transactional outbox duplicate delivery`、`HTTP idempotent method`、`distributed systems fault tolerance course lab recovery`、`agent checkpoint resume conformance` 和 `state-verified tool fault recovery evaluation`。
- 资料类型：Temporal、LangGraph、AWS、IETF 与 MIT 的官方文档/课程原文；2026 年 8 月和 11 月的 arXiv 预印本。检查了原始页面，不以搜索摘要代替来源。
- 范围边界：本记录不声称穷尽 durable workflow / Agent 框架，不比较生产 SLO，也不把任一厂商的恢复语义推广为通用保证。当前框架文档是可变资料，引用均以 2026-11-18 访问状态为准。

## 主题、证据与边界

| 外部主题 | 原始资料支持的事实 | 证据范围与不支持的结论 |
|---|---|---|
| 工作流日志与恢复游标 | Temporal 将 Workflow 的命令与 Event History 分开；新 Worker 可以依据已保存事件重放确定性的 Workflow 代码，重建本地状态后继续。 | 这是 Temporal 的 durable execution 设计。不能推出所有框架都保存相同粒度的事件、不会重跑工具，或整个外部事务 exactly-once。来源：[Understanding Temporal](https://docs.temporal.io/evaluate/understanding-temporal)、[Workflow Execution](https://docs.temporal.io/workflow-execution)、[Workflow Definition](https://docs.temporal.io/workflow-definition)。访问日期：2026-11-18。 |
| Activity retry 和副作用 | Temporal 文档说明，已完成 Activity 不因 Workflow Replay 再执行；但 Activity 尚未把完成事实写入 Event History 时，Worker 可能已经完成外部动作并崩溃，服务会重试。一次 Activity 的主体可能执行多次；幂等键由被调用服务检查，不能仅由工作流引擎推导。 | 这是 Temporal 的 Activity 边界，是否适用其他执行单元须查对应框架。不能把“工作流观察到一次结果”写成 Activity 主体只运行一次。来源：[Activity Definition](https://docs.temporal.io/activity-definition)，[Retry Policies](https://docs.temporal.io/encyclopedia/retry-policies)。访问日期：2026-11-18。 |
| Agent 图节点重试与 checkpoint 持久性 | LangGraph 文档分别定义 persistence/checkpointer、interrupt/retry 和 `exit` / `async` / `sync` durability mode。文档说明中断或重试可能从受影响节点的函数起点重新执行；checkpoint 能恢复图状态，不会自动包住节点内部的外部副作用。Python 与 JavaScript 页面关于 `sync` 的等待边界有实现/文档差异，不能混为一个跨 SDK 保证。 | 支持“按具体 SDK、版本和持久化后端检查恢复边界”，不支持“所有 LangGraph checkpoint 都等价”或“保存图状态即 exactly-once”。来源：[LangGraph Fault Tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)、[LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[LangGraph Checkpointer Source](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/checkpointers.mdx)。访问日期：2026-11-18；文档为滚动更新。 |
| HTTP 幂等与提供方令牌 | RFC 9110 §9.2.2 把幂等定义为对同一请求重复应用时，服务器的预期效果与单次应用相同。AWS DynamoDB 的 `TransactWriteItems` 支持服务特定的 client token，并说明其 10 分钟有效期、请求参数变化时返回 mismatch。 | HTTP 方法语义不代表某个 API 的读取/查询合同；DynamoDB 的 token 时限不适用于其他服务。使用提供方幂等键必须核实作用域、参数匹配、过期和保存结果规则。来源：[RFC 9110 §9.2.2](https://www.rfc-editor.org/rfc/rfc9110.html#section-9.2.2)、[DynamoDB Transactions](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/transaction-apis.html)。访问日期：2026-11-18。 |
| 本地提交与消息派发的双写 | AWS transactional outbox 将业务数据和待派发事件放在同一数据库事务中，避免一边成功而另一边未记录；后续消息仍可能重复投递，消费者仍应幂等。 | 该模式降低本地数据库与队列之间的双写不一致，不会让数据库与任意第三方 API 成为一个原子事务，也不保证下游没有副作用重复。来源：[Transactional Outbox Pattern](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/transactional-outbox.html)。访问日期：2026-11-18。 |
| 租约接管与 fencing | Google Chubby 论文说明，锁持有者可把含锁世代号的 sequencer 随受保护请求传给资源服务，由接收服务验证有效性并拒绝旧世代；etcd 项目的锁示例演示了租约过期后旧 client 携旧版本写入时，由共享存储端拒绝。 | fencing 只有在实际接受写入/副作用的边界核验代数时才能阻止陈旧请求；worker 自己查看租约时间不够。它不能撤回已经到达远端或仍在途的副作用，也不自动适用于不接受 token 的第三方 API。来源：[The Chubby Lock Service, §2.4](https://research.google.com/archive/chubby-osdi06.pdf)、[etcd contrib/lock example](https://github.com/etcd-io/etcd/blob/main/contrib/lock/README.md)。访问日期：2026-11-18。 |
| Checkpoint / resume 的可检验合同 | 2026-08-04 的单作者 arXiv 预印本提出一组 checkpoint、interrupt、resume 性质，并以 TLA+ 模型和版本固定的 persistence API harness 检查五种 workflow framework。作者报告这些 API 在幂等、并发恢复与状态有效性等维度的行为并不相同。 | 这是未同行评审的单作者预印本；其性质选择、probe、版本、后端和结论范围属于作者的研究设计。具体框架缺陷不能在没有独立复现时写成普遍事实。其价值是提醒审阅者明确要验证的恢复性质，并固定 API/SDK/版本/后端。来源：[Resume Means Resume, arXiv:2608.03836v1](https://arxiv.org/abs/2608.03836v1)。访问日期：2026-11-18。 |
| Agent 工具故障恢复评估 | 2026-11-16 的 AgentFaultBench 预印本在隔离 GitLab 环境中对 LangGraph 与 CrewAI 任务注入工具故障，并检查文件、issue 状态、必需读取/验证步骤和最终回报；作者明确说明未做人工验证。 | 单作者、预印本、指定任务/模型/框架/隔离环境，不能推出生产发生率或某种恢复建议普遍有效。该资料支持“恢复评估同时核实最终外部状态和必需过程”的测试设计示例，不支持对模型/框架排名。来源：[AgentFaultBench, arXiv:2611.20711v1](https://arxiv.org/abs/2611.20711v1)。访问日期：2026-11-18。 |
| 教学组织样本 | MIT 6.5840 Spring 2026 将 fault tolerance、replication、consistency 组织为课程、论文讨论和渐进式编程实验；Lab 3 从单节点失败/重启后的持久状态切入 Raft，Lab 4 再将复制协议用于满足 at-most-once Put 与 linearizable history 的 key/value 服务。 | 这是研究生分布式系统课程的组织案例，前置和编程深度高于本书目标读者；只能借鉴从故障现象到可观察语义、实现边界与测试的连接，不能证明相同顺序对所有读者都有效。来源：[MIT 6.5840 Spring 2026](https://pdos.csail.mit.edu/6.824/index.html)、[Lab 3: Raft](https://pdos.csail.mit.edu/6.5840/labs/lab-raft1.html)、[Lab 4: Fault-tolerant Key/Value Service](https://pdos.csail.mit.edu/6.824/labs/lab-kvraft1.html)。访问日期：2026-11-18。 |

## 证据归纳

1. Checkpoint 是恢复流程使用的持久状态或恢复游标；event history/trace 回答发生顺序，workspace 和外部 provider 提供可查询的现实状态。不同系统可以使用同一底层存储，但这些信息的语义和更新边界不能因此混为一谈。
2. 恢复正确性取决于持久化边界和外部副作用边界是否对齐。进程可能在外部操作已经成功、但本地尚未记录回执时崩溃；因此没有本地回执不是“操作没发生”的证据。
3. 幂等是带范围的契约。稳定操作身份、相同参数、服务器端去重或结果查询、保留期限和并发请求语义共同决定是否可安全重试。一个本地 checkpoint 本身不能把任意第三方操作变成 exactly-once。
4. 在只能提供 at-least-once 派发的系统中，常见设计组合是“本地事务保存 intent/outbox + 重复派发 + 下游幂等/结果对账”。若下游没有可查询、去重或补偿合同，恢复结果可能必须保持 unknown 并转人工处理。
5. 租约提供接管时机，不能单独阻止失联或暂停的旧 worker 继续写入。fencing 需要由数据/副作用接收方校验递增世代；若外部服务不支持，运行时只能在自己的写入和派发边界拒绝陈旧 worker，仍需核查在途副作用。
6. 恢复演练应在状态边界注入故障，并核对 workspace/artifact 和外部对象；只核对模型继续输出或最终文字不足以验证副作用安全。
7. Checkpoint 格式和行为还受 schema、worker/runtime、model/tool/policy 版本、task/workspace revision、持久化后端和并发 owner 影响。迁移要显式识别可复用与需重算的记录。

## 未解决边界

- 不同框架的 checkpoint 写入、pending write、interrupt 消费和并发恢复语义会随 SDK、版本、存储后端和配置变化；没有建立本项目的跨框架实测矩阵。
- 第三方 API 的幂等键与 query-by-client-reference 不能统一抽象为通用保证；每个实际工具 adapter 仍需核对 provider 文档和部署行为。
- 2026 年的 checkpoint conformance 与 AgentFaultBench 预印本尚需独立复现、同行评审与更多任务/框架覆盖，本文不把其经验结果作为稳定工程定律。
- 恢复 SLO 的窗口、是否计入安全人工接管、任务重试与最终业务完成的关系必须由具体产品合同定义；本记录不设统一数值门槛。
