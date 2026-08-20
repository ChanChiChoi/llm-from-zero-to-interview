# 第 16 章 Multi-Agent 与 Ultra Test-Time Compute：从并行候选到可验证协作

## 16.1 为什么增加 Agent 可能有用

一个模型在一次请求中完成检索、规划、代码修改、测试和报告，最容易遇到三个问题：上下文同时装入太多材料，某个早期假设一路传播到最后，以及所有工具都由同一个决策循环直接控制。把任务拆成多个 worker，可以让不同节点拥有不同的输入、工具、权限和验收责任。

这里的关键不是“参与者越多越聪明”。增加 worker 只是增加了更多计算轨迹。只有当这些轨迹带来独立信息、能够产生可验证的产物，并且合并成本没有吞掉收益时，系统才可能比单个 Agent 更好。若十个 worker 读取同一份错误文档、使用同一条错误提示并输出相似的自然语言答案，十票赞成仍然只是一个错误被复制十次。

因此，本章把 multi-agent 理解为一种任务执行架构，而不是一种自动提升智能的算法定律。它至少包含以下角色：

- **worker**：在限定输入、工具和权限下执行一个子任务；
- **coordinator**：维护依赖图、分配预算、接收产物和处理重试；
- **verifier**：用测试、规则、第二来源或独立模型检查产物；
- **merge owner**：处理候选之间的冲突，生成可追踪的合并版本；
- **executor**：在重新检查授权和版本后执行外部副作用。

这几个角色可以由不同模型承担，也可以由同一个模型在不同状态机节点中承担。角色名称本身没有安全或正确性含义，真正重要的是输入输出契约、版本、权限和验收规则。

## 16.2 从 self-consistency 到 multi-agent

self-consistency 的基本做法是对同一问题采样多条回答，再根据最终答案投票。它利用的是候选多样性，通常没有独立工具状态，也没有写入型产物。multi-agent 在此基础上增加了环境和协作协议：worker 可能读取不同资料、运行不同测试、维护独立 workspace，并将结果交给后续节点。

两者的差异可以用一次代码修复任务说明。self-consistency 可能生成十个修复建议，然后让一个选择器挑选其一；multi-agent 则可以让一个 worker 构建调用链，一个 worker 创建最小复现，一个 worker 检查历史变更，最后由测试和安全 worker 验证候选 patch。后者的额外价值不在于“回答更长”，而在于每条轨迹都能携带不同的观察和 artifact。

但 multi-agent 也会带来新的问题：消息要序列化，workspace 要隔离，冲突要合并，失败要区分是否产生了副作用，更多候选还需要更多验证。若任务本身很短、没有外部反馈或不存在可靠的合并规则，单 Agent 或固定 workflow 往往更容易审计。

## 16.3 任务图：哪些节点可以并行

把协作过程表示为有向图：

```math
G=(V,E)
```

`V` 是任务节点集合，例如 `search_logs`、`read_history`、`build_patch` 和 `run_tests`；`E` 是依赖边。若存在边 `u -> v`，表示节点 `v` 至少要等到节点 `u` 的指定产物、版本和状态满足条件后才能开始。边表达的是数据依赖或副作用依赖，不只是“两个角色有关系”。

例如，定位调用链和阅读历史 issue 可能没有数据依赖，可以并行；生成 patch 依赖定位结果，必须等待；部署依赖测试、安全检查和授权，不能仅因为测试 worker 完成就开始。一个更具体的图可以写成：

```text
call_graph ─┐
reproduce   ├─> patch_candidate ─> isolated_test ─┐
history     ─┘                                    ├─> merge_review
security_read ────────────────────────────────────┘
```

图中的“并行”只表示节点可以同时运行，不表示它们可以同时修改同一个外部对象。`reproduce` 可以在隔离目录中生成日志，`patch_candidate` 可以在自己的分支中写文件，但两个节点不能未经协调同时写生产分支。

如果任务图中有大量强顺序边，增加 worker 数不会带来相同比例的加速。图的价值在于先暴露关键路径，再决定哪些节点值得并行。只根据产品名称启动 swarm，无法替代这一步分析。

## 16.4 关键路径与并行效率

设并行阶段有 `n` 个 worker，第 `j` 个 worker 的有效工作时间为 `T_j`，协调、合并和验证时间分别为 `T_coord`、`T_merge` 和 `T_verify`。在 worker 可以同时运行的近似条件下：

```math
T_{\mathrm{parallel}}
\approx \frac{\max_{1\le j\le n}T_j}{\eta}
 +T_{\mathrm{coord}}+T_{\mathrm{merge}}+T_{\mathrm{verify}}
```

这里 `n` 是正整数，所有时间都应是有限的非负数，`0<\eta\le1` 是把队列、资源争用、调度和不完全并行折算进去的效率系数。使用最大 worker 时间而不是平均时间，是因为一个最慢的 worker 仍可能阻塞后继节点。

如果 worker 负载大致均衡，且 `T_work=\sum_j T_j`，可以进一步写成：

```math
T_{\mathrm{parallel}}
 \approx \frac{T_{\mathrm{work}}}{n\eta}
 +T_{\mathrm{coord}}+T_{\mathrm{merge}}+T_{\mathrm{verify}}
```

这只是均衡负载下的近似，不能用于掩盖长尾。共享 GPU、工具队列、网络带宽和合并阶段都会使 `\eta` 下降。并行主要减少 wall-clock 的关键路径，不保证减少 token、GPU 时间或货币成本。

理想加速还受到 Amdahl 定律约束。设总工作量中不可并行部分为 `W_s`，可并行部分为 `W_p`，使用 `n` 个 worker，则：

```math
S(n)\le\frac{W_s+W_p}{W_s+W_p/n}
```

这里 `W_s,W_p` 是同一计量单位下的有限非负工作量，`n` 是正整数，且 `W_s+W_p>0`。当 `W_p=0` 时，理想加速比为 1；当 `W_s` 很大时，继续增加 worker 的收益迅速变小。真实系统还要把消息、调度、重复探索和验证成本加到分母中，因此实际加速通常低于这个上界。

## 16.5 并行是否带来新的信息

并行的质量收益来自信息增益，而不是输出数量。一个子任务是否值得独立运行，可以从三个方面判断：

第一，它是否拥有不同的输入、假设或观察路径。例如日志分析和代码静态分析可能提供互补证据；两个 worker 如果读取完全相同的检索结果，就不应把它们当成两个独立来源。

第二，它是否有清楚的产物。`call_graph.json`、最小复现日志、候选 patch、测试报告和来源表都可以被后续节点处理；“我认为应该加锁”这种没有定位、版本和验证状态的句子不能作为可靠接口。

第三，它是否存在检查方法。数学答案可以由符号计算或独立验证器检查，代码 patch 可以运行测试和静态检查，研究结论可以核对原文和版本。没有 verifier 的多轮投票只能改变置信语气，不能创造事实。

错误相关性是最容易被忽略的因素。多个 worker 可能共享同一个模型偏见、system prompt、检索缓存、污染数据集或错误工具状态。此时新增 worker 的有效信息量远小于新增答案数。评估时应记录模型 revision、prompt revision、检索来源、工具版本和 workspace revision，以便计算共因失败，而不是只计算最终投票是否一致。

在理想的独立同分布假设下，多数投票可能降低随机错误；但真实 Agent 系统往往违反独立性。因而“多数通过”最多是一个候选筛选信号，涉及付款、发布、删除、权限和安全结论时仍需要独立规则或人工确认。

## 16.6 选择协作拓扑

不同任务需要不同的拓扑。拓扑决定消息方向、共享状态和冲突位置，不应由 `swarm` 这个名称预先决定。

| 拓扑 | 主要结构 | 适合场景 | 主要代价 |
| --- | --- | --- | --- |
| 独立探索 | 多个 worker 读不同输入，最后合并 | 多来源研究、候选解生成 | 来源重复和合并冲突 |
| 分工流水线 | 一个节点的产物进入下一个节点 | 检索、分析、验证的固定链路 | 前一步阻塞后续 |
| 主控—worker | coordinator 动态拆分和回收 | 子任务难度不确定的研究或代码任务 | 主控上下文和调度压力 |
| 树形搜索 | 每个节点扩展候选并保留部分状态 | 有明确状态、动作和评分的搜索 | 状态复制、剪枝和预算控制 |
| 批评或投票 | 多个候选相互检查后选择 | 存在多个可行答案且有可靠评审规则 | 共因错误和评审偏差 |

一个简单的判断顺序是：先确认子任务能否独立完成，再确认输出能否结构化表示，最后确认是否有可靠的合并或验证器。只要三个条件都不满足，增加 worker 通常只会增加上下文和协调成本。

固定 workflow 与 multi-agent 也不是互斥的。稳定的读取、解析、测试和审批步骤可以由代码状态机固定；只有候选生成或不确定性高的节点才使用多个 worker。这样既保留了并行搜索的灵活性，也让外部副作用处于可审计的串行路径。

## 16.7 Worker 的输入输出契约

worker 的最小契约应明确以下内容：

- `task_id`、`parent_id` 和角色标识；
- 输入 artifact、`base_revision` 和可见上下文范围；
- 可以使用的工具、数据范围和副作用上限；
- 输出 artifact 的 schema、owner、过期时间和失败语义；
- reasoning、工具、验证和 wall-clock 预算；
- 取消、超时、重试和状态未知时的处理方式。

一个代码定位 worker 的输出可以是：

```json
{
  "artifact_id": "a-callgraph-01",
  "task_id": "bug-17",
  "worker_id": "call-graph",
  "base_revision": "repo@abc123",
  "claim_id": "retry-policy",
  "claim": "retry policy is read while refresh is being updated",
  "evidence": ["src/cache.py:81", "tests/test_refresh.py:44"],
  "evidence_hash": "sha256:7f...",
  "status": "candidate",
  "verification": {"status": "pending", "checks": []},
  "expires_at": "2026-08-14T12:00:00Z"
}
```

`claim` 是结论候选，`evidence` 是来源引用，`evidence_hash` 是对证据集合或受控 artifact 的完整性承诺，`status` 表示生命周期状态。`confidence` 可以帮助安排复核顺序，却不能替代来源；`verified` 也不能仅凭 worker 自报而成立。验证状态必须引用实际测试、规则或第二来源。

artifact 还应有唯一 ID、任务 ID、worker ID、输入基线和版本。没有这些字段，主控无法判断一个报告是否建立在旧 workspace 上，也无法把测试结果绑定到具体 patch。写入型 artifact 应保存 patch 或 immutable revision，而不是只保存“最终文件内容”。

## 16.8 Artifact 的合并与冲突

合并器不应把所有 worker 的长文本直接拼接给另一个模型。更可靠的过程是先按 `claim_id`、`artifact_id` 和 revision 建立索引，再分别处理事实、假设、证据和未决冲突。

可以把合并状态分为：

- `agree_verified`：独立证据一致，且检查已通过；
- `agree_shared_source`：结论一致，但多个 worker 依赖同一来源；
- `conflict`：候选结论或 patch 互相矛盾；
- `insufficient_evidence`：当前证据不足以作出结论；
- `expired`：artifact 超过有效期或基线已变化。

两个 worker 都说“应该增加锁”，并不等于两个独立证据；如果它们都引用同一行代码，支持来源数仍然是一个。相反，一个 worker 的日志复现和另一个 worker 的历史修复虽然结论可能不同，但它们可以形成互补证据。合并器应保留冲突，而不是强行选出语气最确定的一项。

代码 patch 的合并至少检查共同基线、修改路径、用户已有变更、测试对应的 revision、生成文件和外部命令。两个 patch 修改同一行时，系统应生成显式冲突并交给重新计算或人工处理，不能用最后写入覆盖先写入。

## 16.9 生命周期、取消与未知状态

worker 和 artifact 需要明确的生命周期。一个常见的 artifact 状态序列是：

```text
created -> evidence_attached -> tested -> reviewed
                                      ├-> accepted
                                      └-> rejected
```

每次状态迁移都应绑定 `task_id`、`base_revision`、owner、证据 hash、schema version 和过期时间。旧 revision 的 artifact 不能因为重试而覆盖新 revision。若输入代码已经变化，即使旧测试曾经通过，artifact 也应回到待验证状态。

worker 状态还要区分 `queued`、`running`、`succeeded`、`failed`、`blocked`、`cancelled` 和 `unknown`。`unknown` 不是 `failed` 的同义词：客户端在发送创建工单或支付请求后断线，可能不知道动作是否已经执行。直接把 unknown 改成 failed 再重试，可能产生重复副作用。

取消也不是只向模型发送一句“停止”。协调器要向工具执行器传播取消，记录取消时的 artifact 和预算，等待可安全停止的子任务，并把已经提交的副作用与未提交的草稿分开。恢复时先查询外部状态，再决定等待、补偿、重试或人工接管。

## 16.10 Workspace 与权限隔离

共享上下文和共享 workspace 是两个不同的选择。多个 worker 可以共享任务目标和只读证据索引，却不应默认共享所有历史、凭证和写权限。一个实用的分区是：

- 只读证据区：保存来源、版本、hash 和脱敏摘要；
- worker 写区：每个可写 worker 使用独立目录、分支或容器；
- 合并区：只有 merge owner 能生成新的合并 revision；
- 执行区：独立 executor 在重新检查权限后执行外部动作。

隔离 workspace 解决三类问题。第一，worker 不会看到其他 worker 的半成品，实验更可复现。第二，patch、测试结果和输入基线能够绑定，责任可以归因。第三，合并器可以比较差异，而不是猜测某个共享文件是由谁最后写入的。

权限必须与计算预算分离。增加 reasoning token、worker 数量或验证轮次，不应自动增加文件、网络、数据库或生产写权限。高风险动作需要独立的 policy 检查、用户确认或人工审核；即使所有 worker 一致，也不能用投票替代授权。

worker 读取的网页、issue、代码注释和用户上传文档都是外部数据，可能包含提示注入。artifact 中的自然语言不能自动提升为系统指令。后续节点只应把经 schema 解析的字段作为数据使用，所有可执行动作仍须经过工具 allowlist、参数校验和策略服务。

## 16.11 Ultra Test-Time Compute 是预算控制问题

这里的 ultra 不是一个精确的模型类别，而是把更多候选、工具观察、验证、恢复或协作预算投入一次任务。它可以通过更多采样实现，也可以通过搜索、verifier、工具循环或多个 Agent 实现。统一的抽象是：在请求完成前，系统允许花费更多资源来降低任务失败概率。

如果所有预算已经换算为同一个成本单位，可以写成：

```math
\sum_{i=1}^{n}b_i+b_v+b_m\le B
```

`b_i` 是第 `i` 个 worker 的预算，`b_v` 是验证预算，`b_m` 是合并和恢复预算，`B` 是总预算。上述式子要求所有变量是同一账本中的有限非负量，`n` 是正整数。不能把 token、毫秒和副作用次数未经换算直接相加；更准确的工程表示是预算向量：

```math
\mathbf{B}=(B_{\mathrm{model}},B_{\mathrm{tool}},B_{\mathrm{verify}},B_{\mathrm{time}},B_{\mathrm{side}})
```

其中每个分量使用自己的单位和上限，`B_side=0` 明确表示不允许外部副作用。向量预算的比较是逐项约束，而不是把所有单位假装成一个数字。

生成预算和验证预算之间存在实际取舍。若 `b_v` 太小，系统会收集很多候选却无法判断；若 `b_m` 太小，冲突会被文本拼接掩盖；若工具预算耗尽，新增的 reasoning token 也无法获得新的观察。困难任务不一定需要更多 worker，可能更需要一个更强的 verifier 或更好的测试环境。

## 16.12 成本、单位成功成本与边际收益

多 Agent 的成本账本应至少包括模型、工具、通信、合并、验证和人工：

```math
C_{\mathrm{total}}
=\sum_i C_{\mathrm{model},i}
+\sum_i C_{\mathrm{tool},i}
+C_{\mathrm{message}}
+C_{\mathrm{merge}}
+C_{\mathrm{verify}}
+C_{\mathrm{human}}
```

这些成本项必须使用同一个货币或归一化成本单位，且是有限非负数。未测量的成本不能默认为 0；它应记录为缺失，直到账单、容量模型或人工工时补齐。只有“明确没有发生”的资源才能记为 0。

设一批任务的成功数为 `N_success`，单位成功成本定义为：

```math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
```

这里要求 `N_success` 是非负整数。当 `N_success=0` 时，单位成功成本未定义，应返回 `None` 或报告为 `undefined`；不能用 `max(1,N_success)` 把失败批次伪装成一个可比较的成本。比较不同架构时，成功判据、任务集和成本账本也必须一致。

新增 worker 是否值得，可以用边际效用表示：

```math
\Delta U_k
=\Delta P_{\mathrm{success},k}V
-\Delta C_k
-\lambda\Delta R_k
```

`\Delta P_{\mathrm{success},k}` 是在同一任务分布和成功定义下新增第 `k` 个 worker 带来的成功概率增量，取值应在 `[-1,1]`；`V` 是成功的业务价值；`\Delta C_k` 是同一成本单位下的增量成本；`\Delta R_k` 是风险增量；`\lambda\ge0` 是风险折算系数。若风险是不可接受的硬约束，不能让很大的 `V` 抵消它，而应直接拒绝该扩展。

继续扩展还可以写成三个同时成立的条件：

```math
G_{\mathrm{expand}}
=\mathbf{1}[\Delta P_{\mathrm{success}}>\delta]
 \mathbf{1}[C_{\mathrm{extra}}\le C_{\max}]
 \mathbf{1}[R_{\mathrm{extra}}\le R_{\max}]
```

这里 `\delta`、`C_max` 和 `R_max` 应事先定义，所有增量来自相同的评估口径。若概率或风险没有测量，不能用 0 代替以获得一个“允许继续”的结果；应进入信息不足或人工决策状态。

## 16.13 什么时候停止扩张

高预算系统必须在启动前设置最大 worker 数、最大轮次、模型预算、工具时间、验证次数和外部副作用上限。运行中还要观察是否产生新证据。以下信号通常支持停止或降级：

1. 连续若干轮的 artifact 没有新增来源、状态或可区分的候选；
2. 候选已经满足独立 verifier 的验收条件；
3. 新增 worker 只重复同一个共因来源；
4. 剩余时间不足以完成验证和安全检查；
5. 预算已经耗尽，或者风险超过硬性约束；
6. 工具状态未知，继续重试可能产生重复副作用。

“模型还想继续思考”不是充分的继续条件。停止时要保存未完成 worker 的状态、取消原因、已生成 artifact、预算余额和外部动作状态，使任务可以安全恢复或交给人工。

## 16.14 一个跨仓库修复的完整流程

假设用户报告某个服务在高并发刷新配置时偶发 500。协调器先固定仓库 revision、问题描述、允许访问的目录和只读权限，然后建立四个子任务。

**调用链 worker** 读取入口、缓存刷新和异常处理，输出带文件位置的调用链 artifact。它不能修改代码。

**复现 worker** 在隔离 workspace 中构造最小并发测试，记录命令、依赖版本、运行次数和失败日志。它的失败也有价值：如果无法复现，应明确说明条件不足，而不是输出“问题不存在”。

**历史 worker** 只读取版本记录和 issue，寻找相同组件的历史修复。引用必须包含 revision 或原始记录定位，社区转述不能与正式变更混为同一证据等级。

**安全 worker** 检查候选变更可能触碰的敏感路径、网络访问、依赖变化和凭证处理。它不接受 patch worker 的自然语言声称作为结论。

当调用链、复现和历史证据汇合后，patch worker 在隔离分支中生成一个或多个候选 patch。每个候选都记录共同基线、修改文件、未验证假设和测试计划。测试 worker 对每个候选执行公开测试、回归测试和针对原始失败的并发测试；安全 worker 对最终 diff 重新检查。

只有测试、安全检查、版本一致性和授权都满足条件，merge owner 才能生成合并 revision。外部 executor 再次读取当前仓库状态和权限，最后才执行提交或发布。任一环节返回 unknown，都不能直接重试写入。

这个例子展示了并行真正解决的问题：把只读调查和相互独立的实验并行化，同时把 patch 合并和外部副作用留在有版本、有验证、有权限的串行路径。若三个 worker 都直接修改共享工作区，系统并没有获得可靠的并行，只是把竞争条件引入开发流程。

## 16.15 如何证明 multi-agent 值得

实验至少应包含以下对照：direct 或固定 workflow、single-agent、多 Agent 串行、多 Agent 并行，以及并行加独立 verifier。任务集、模型 revision、工具 allowlist、最大总预算、成功定义、数据版本和权限必须对齐。

如果 multi-agent 可以使用十倍 token，却只报告最高准确率，不能据此证明架构有效。至少要记录：

- 任务成功率和按难度、风险、输入长度的切片结果；
- 总模型 token、工具调用、通信量、验证调用和人工时间；
- wall-clock 的中位数、P95/P99、关键路径和合并时间；
- 候选池是否包含正确答案，以及选择器是否选对；
- 共因失败率、冲突率、合并拒绝率和取消恢复率；
- 重复副作用、越权调用、敏感数据暴露和人工接管。

还要做消融实验：去掉并行、去掉 verifier、把隔离 workspace 改为共享 workspace、固定 worker 角色改为动态角色、固定总预算改为无限预算。这样才能区分收益到底来自并行、更多候选、角色分工、工具观察还是验证器。

对同一任务进行配对回放尤其重要。记录 baseline 和 candidate 在每个任务上的成功变化，可以判断新增 worker 修复了哪些样本、又在哪些样本引入了退化。只比较两个总体平均数，会掩盖高风险切片和长尾延迟。

## 16.16 安全边界：并行会放大什么

并行架构会把原本一次发生的错误放大成多次发生。常见风险包括：

- 多个 worker 同时提交相同外部动作，造成重复邮件、重复工单或重复写入；
- 一个被污染的 artifact 被广播给所有后继节点，形成错误共识；
- worker 共享过宽的凭证或上下文，扩大隐私泄露范围；
- 不可信网页或文件中的指令被当作高优先级动作；
- patch worker 修改测试或配置后，其他 worker 在污染环境上得出“通过”；
- 超时状态被当成失败，恢复逻辑重新执行已经成功的动作。

相应的防御不是让主控模型“更小心地想”，而是把风险落到系统约束：最小权限、只读证据、隔离 workspace、不可变 artifact、独立 policy、幂等键、状态查询、参数 schema、审计日志和人工确认。外部副作用要由 executor 统一执行，worker 只能提出经过结构化描述的 proposal。

## 16.17 一个可运行的最小协作审计器

下面的示例不调用模型、网络或文件系统。它用三个 worker artifact 模拟一个候选修复流程，演示四个工程要点：artifact 需要完整身份和证据 hash；相同 claim 的不同方案必须报告冲突；预算必须在同一 `credits` 账本中分配；当成功数为零时，单位成功成本返回 `None`，而不是制造一个分母。

```python
import math


ARTIFACT_STATES = {"candidate", "verified", "rejected", "unknown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be non-empty text")


def require_finite_non_negative(value, name):
    require(
        type(value) in {int, float} and math.isfinite(value) and value >= 0,
        f"{name} must be a finite non-negative number",
    )


def validate_artifacts(artifacts):
    require(isinstance(artifacts, list) and artifacts, "artifacts must be non-empty")
    seen = set()
    required = {
        "artifact_id",
        "task_id",
        "worker_id",
        "base_revision",
        "claim_id",
        "claim",
        "position",
        "evidence_hash",
        "status",
        "verified",
    }
    for artifact in artifacts:
        require(isinstance(artifact, dict), "artifact must be an object")
        require(set(artifact) == required, "artifact schema mismatch")
        require_text(artifact["artifact_id"], "artifact_id")
        require(artifact["artifact_id"] not in seen, "duplicate artifact_id")
        seen.add(artifact["artifact_id"])
        for key in ("task_id", "worker_id", "base_revision", "claim_id", "claim", "position", "evidence_hash"):
            require_text(artifact[key], key)
        require(artifact["status"] in ARTIFACT_STATES, "unknown artifact status")
        require(type(artifact["verified"]) is bool, "verified must be boolean")
        require(not artifact["verified"] or artifact["status"] == "verified", "verified status mismatch")
    return True


def find_conflicts(artifacts):
    validate_artifacts(artifacts)
    positions = {}
    for artifact in artifacts:
        positions.setdefault(artifact["claim_id"], set()).add(artifact["position"])
    return sorted(claim_id for claim_id, values in positions.items() if len(values) > 1)


def allocate_budget(total, allocations):
    require_finite_non_negative(total, "total")
    require(isinstance(allocations, dict) and allocations, "allocations must be non-empty")
    used = 0.0
    for name, amount in allocations.items():
        require_text(name, "allocation name")
        require_finite_non_negative(amount, name)
        used += float(amount)
    require(used <= float(total), "budget exceeded")
    return {"total": float(total), "used": used, "remaining": float(total) - used}


def unit_success_cost(total_cost, success_count):
    require_finite_non_negative(total_cost, "total_cost")
    require(type(success_count) is int and success_count >= 0, "success_count must be a non-negative integer")
    if success_count == 0:
        return None
    return total_cost / success_count


ARTIFACTS = [
    {
        "artifact_id": "a-callgraph",
        "task_id": "bug-17",
        "worker_id": "call-graph",
        "base_revision": "repo@abc123",
        "claim_id": "retry-policy",
        "claim": "refresh reads retry policy while it is updated",
        "position": "add-lock",
        "evidence_hash": "sha256:callgraph",
        "status": "candidate",
        "verified": False,
    },
    {
        "artifact_id": "a-history",
        "task_id": "bug-17",
        "worker_id": "history",
        "base_revision": "repo@abc123",
        "claim_id": "retry-policy",
        "claim": "refresh reads retry policy while it is updated",
        "position": "copy-on-write",
        "evidence_hash": "sha256:history",
        "status": "candidate",
        "verified": False,
    },
    {
        "artifact_id": "a-test",
        "task_id": "bug-17",
        "worker_id": "reproduce",
        "base_revision": "repo@abc123",
        "claim_id": "concurrency-failure",
        "claim": "failure appears only under concurrent refresh",
        "position": "reproduced",
        "evidence_hash": "sha256:test",
        "status": "verified",
        "verified": True,
    },
]


print("artifact_valid", validate_artifacts(ARTIFACTS))
print("budget", allocate_budget(100, {"workers": 55, "tools": 20, "verify": 7}))
print("conflicts", find_conflicts(ARTIFACTS))
print("unit_success_cost", unit_success_cost(82, 0))
```

预期输出为：

```text
artifact_valid True
budget {'total': 100.0, 'used': 82.0, 'remaining': 18.0}
conflicts ['retry-policy']
unit_success_cost None
```

这个 demo 的 `position` 冲突不是错误输入，而是合并阶段必须处理的事实：两个 worker 对同一 claim 提出了不同方案。`a-test` 的 `verified=True` 之所以合法，是因为它的状态同时为 `verified`；真实系统还需要把验证命令、环境、输出 hash 和 verifier 身份写入独立 artifact。`unit_success_cost` 返回 `None` 表示这批任务尚无成功样本，不能据此宣称某种架构便宜。

边界测试应至少包含：重复 `artifact_id`、空 `evidence_hash`、未知状态、`verified` 与状态矛盾、预算超过总额、`NaN` 成本、负成本和零成功数。所有这些情况都应显式拒绝或返回未定义，而不是静默转换成 0。

## 16.18 设计时的检查顺序

面对一个需要“多 Agent”或“更深思考”的任务，可以按以下顺序建立设计：

首先写出单 Agent 或固定 workflow baseline，确认问题确实来自上下文、等待、候选覆盖、验证能力或权限隔离，而不是提示词和数据质量问题。

其次画任务图，标出数据依赖、副作用依赖、可并行节点和关键路径。对于每个候选并行节点，说明它会带来哪种新信息，以及谁能验证该信息。

再次定义 artifact schema、基线 revision、状态和失败语义。没有这些字段，后续成本和质量都无法归因。

然后分开设置模型、工具、验证、时间、通信和副作用预算。预算不足以验证候选时，不应继续盲目增加候选。

最后用固定总资源的对照实验验收，并把共因错误、未知状态、重复副作用和人工接管纳入指标。架构是否值得，取决于单位成功成本、关键路径延迟和风险，而不是 worker 数量。

## 16.19 练习

**练习一：任务图。** 为跨文件代码修复、三来源研究报告和数据分析分别画出 `G=(V,E)`，为每条边写明它传递的是数据依赖还是副作用依赖。

**练习二：artifact 合约。** 设计一个包含 claim、evidence、hypothesis、verified result 和 side effect 的 schema，并为每个字段写出来源、版本、权限和过期语义。

**练习三：预算实验。** 在固定总 token、工具时间和验证次数的条件下，比较单 Agent、并行 worker 和并行加 verifier。列出至少三个可能解释成功率变化的混淆因素，并设计消融。

**练习四：未知状态。** 设计一个创建工单的流程：请求已发送但客户端超时。说明如何通过幂等键和状态查询区分未执行、执行中、成功和未知，以及每种状态是否允许重试。

## 16.20 资料边界与本章结论

Self-Consistency、Tree of Thoughts、Language Agent Tree Search 和多 Agent debate 论文支持“增加候选、搜索或交互轨迹可能改善部分任务”的研究观察；它们不构成任何具体产品的内部实现证明。Amdahl 定律和任务图是通用的并行分析工具，不能替代目标系统的真实 profiling。代码、工具调用、workspace 和权限部分属于工程设计，需要通过目标环境中的故障演练和审计验证。

本章最重要的结论是：multi-agent 是把计算、证据和权限分布到多条轨迹的组织方式；ultra test-time compute 是在可观测预算内购买更多候选、观察和验证。它们只有在任务可以分解、产物可追踪、错误不完全相关、合并可验证且风险可控时才有价值。增加 worker 数量本身不是目标，能够在固定资源下提高可验证的任务成功率，才是值得保留的工程结果。

参考资料：

- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*：<https://arxiv.org/abs/2203.11171>
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*：<https://arxiv.org/abs/2305.10601>
- Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models*：<https://arxiv.org/abs/2310.04406>
- Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate*：<https://arxiv.org/abs/2305.14325>
- Amdahl, *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*：<https://doi.org/10.1145/1465482.1465560>
