# 第 14 章 Agent Swarm：并行协作的通信、隔离与合并

## 14.1 为什么要把一个 Agent 拆成多个

一个 Agent 同时负责检索、规划、执行、验证和写报告时，常常会遇到上下文过长、角色冲突和单一路径错误。把任务拆成多个专门 Agent，可以让检索、代码分析、事实核查和结果整理分别使用适合的工具与上下文。

但“swarm”不是把同一个 prompt 复制十份。多个 Agent 之间需要通信、共享证据、处理冲突和决定谁有权提交。没有这些机制，系统只是并行生成多段文本，最终仍然要靠一个模型猜哪段可信。

## 14.2 Swarm 的基本图景

把 Agent 看成节点，把消息、artifact 和依赖看成边，可以得到一个协作图：

```math
G=(V,E),\qquad
V=\{\mathrm{planner},\mathrm{researcher},\mathrm{builder},\mathrm{tester},\mathrm{reviewer}\}
```

planner 产生任务分解，researcher 产出证据，builder 生成修改，tester 验证，reviewer 决定是否提交。没有依赖的节点可以并行；需要前一节点产物的节点必须等待。

一个任务消息不应只包含自然语言，还应带有 task id、parent id、输入 artifact、schema 版本、权限、deadline 和完成状态。这样主控才能知道“这个结果属于哪一个子任务”，也能在失败时只重试失效节点。

## 14.3 共享记忆和隔离工作区

协作中最容易出现的矛盾是“信息共享”和“状态隔离”。研究 Agent 的证据应该让审查 Agent 看到，但两个代码 worker 不应同时修改同一个工作区。实践中可以共享只读 artifact 和事件索引，为每个可写 worker 分配独立 workspace，最后由 merge worker 根据 diff 和测试结果合并。

共享记忆也要区分事实、假设和未验证建议。可以使用：

```json
{
  "claim": "cache refresh may race with read",
  "evidence": ["trace-42", "src/cache.py:81"],
  "status": "hypothesis",
  "author": "worker-a",
  "revision": "repo-17"
}
```

如果把 hypothesis 写进全局 memory 而没有状态标记，后续 Agent 可能把它当作事实，并沿着错误方向继续扩大工作。

## 14.4 通信成本与关键路径

假设有 `n` 个 worker，单 worker 工作量为 `W_i`，通信和协调成本为 `C_comm`，合并时间为 `T_merge`。总 token 成本大致是：

```math
C_{\mathrm{total}}=\sum_i C_i+C_{\mathrm{comm}}+C_{\mathrm{merge}}
```

wall-clock 受最长依赖路径影响：

```math
T_{\mathrm{wall}}\approx
\max_{p\in\mathrm{paths}}\sum_{i\in p}T_i
+T_{\mathrm{merge}}+T_{\mathrm{queue}}
```

所以并行数量越多不一定越快。若所有 worker 最后都把长报告发给 reviewer，通信和合并会吞掉并行收益；若任务本身存在串行依赖，增加 worker 只能增加空等。

## 14.5 一个研究报告 swarm

用户要求比较三家公司对同一技术的公开声明。planner 将任务拆成三条独立检索路径，每个 researcher 只负责一家公司并记录官方来源、发布时间和原文 claim；另一个 auditor 检查来源等级和是否混入媒体转述；writer 根据经过审核的证据生成比较表。

这里的 writer 不应该直接相信 researcher 的总结，而应读取带 URL、版本和引用片段的 artifact。auditor 发现某项“架构结论”只有社区截图支持时，把状态改为 `unverified`，writer 就必须使用限定语，而不能把它写成确定事实。

## 14.6 一个代码 swarm

planner 先生成任务 DAG。call-graph worker 只读仓库构建调用链；repro worker 在隔离环境中创建最小复现；patch worker 根据两个 artifact 提出候选 diff；test worker 运行公开和隐藏测试；security worker 检查权限和敏感路径。merge worker 只接受带版本和测试结果的 patch。

如果 repro worker 失败，它的失败 artifact 仍然有价值：它说明当前复现条件不足。主控可以只重试该节点或请求人工补充信息，而不是重新启动整个 swarm。

## 14.7 冲突合并

两个 Agent 可能得到矛盾结论。合并器不能只按多数票选择，应比较证据的新鲜度、来源权威性、实验可复现性和任务相关性。

可以给 claim 一个内部排序分数：

```math
Q(c)=w_s S(c)+w_r R(c)+w_v V(c)-w_u U(c)
```

其中 `S` 表示来源质量，`R` 表示与任务的相关性，`V` 表示是否被验证，`U` 表示不确定性。这个分数只能帮助排序，不能替代人工或规则审计；两个低质量来源的多数仍然可能错。

## 14.8 权限与提交者

swarm 中每个角色都应使用最小权限。researcher 只需读公开资料，tester 需要执行沙箱命令，merge worker 可能拥有写 artifact 的权限，但最终生产提交应该属于独立 deployer 或人工批准者。

不要让主控 Agent 因为拥有所有工具就自动变成超级用户。任务委派时应把权限作为消息的一部分，并在每个外部动作前重新校验。worker 的 output 也不应直接成为 executor 命令，必须经过 schema、策略和提交验收条件。

## 14.9 评估 swarm 的真实收益

对比 direct、单 Agent 和 swarm 时固定任务集合、工具、最大总预算和成功判据。记录最终成功率、wall-clock、总 token、通信量、merge 时间、人工接管、错误传播和副作用。

再做结构消融：去掉 reviewer、去掉共享 memory、把并行改成串行、改变 worker 数、限制每个角色的上下文。若去掉 reviewer 后成绩不变，说明 reviewer 可能只是增加成本；若增加 worker 只提高未验证答案数量，说明协作协议没有创造价值。

## 14.10 常见失败模式

所有 Agent 共享一个错误事实，导致错误在系统中传播；共享可写 workspace，导致 worker 互相覆盖；消息没有 schema，主控无法知道产物是否完成；没有 deadline，某个慢 worker 阻塞整个任务；merge 只看文本相似度，不看测试和证据；权限过宽，失败副作用被并行放大。

另一个隐蔽问题是上下文重复。每个 worker 都收到完整历史，通信 token 可能比真正任务还大。更好的做法是传递最小必要上下文和 artifact 引用，需要时再读取原文。

## 14.11 面试回答与练习

回答“Agent swarm 如何设计”时，应从任务 DAG、角色边界、artifact schema、只读共享、隔离 workspace、冲突合并、权限、验证和预算讲起。并行降低关键路径时间，但增加通信和合并成本；是否值得要用固定总预算下的成功率、单位成功成本、延迟和风险验证。

练习一：画一个“研究—审核—写作”swarm 的消息流，标记每条消息的证据字段。

练习二：设计两个代码 worker 的 workspace 隔离和 merge 规则。

练习三：当两个 worker 给出矛盾结论时，列出四种不能只靠多数票的原因。

### 14.11.1 什么时候适合并行

并行适合候选之间相对独立、读取成本高、工具等待长或任务天然按领域拆分的场景。例如分别让 worker 分析日志、代码变更和监控曲线，再由主控合并因果链。它不适合所有步骤共享可变状态、每一步都依赖上一步结论或副作用不可逆的任务。

任务图可以表示依赖：

```text
collect_logs ─┐
inspect_diff ─┼─> synthesize_claims -> verify -> commit
check_metrics ┘
```

`collect_logs`、`inspect_diff` 和 `check_metrics` 可以并行，`commit` 必须在验证后串行执行。让多个 worker 同时写生产系统，不是并行优化，而是未治理的竞争条件。

### 14.11.2 artifact 合并协议

worker 返回的 artifact 应包含 `claim`、`evidence_ids`、`confidence`、`assumptions`、`unresolved_conflicts` 和 `proposed_action`。主控不应把不同 worker 的自然语言直接拼接，而要先按 claim ID 合并，发现同一实体和版本的冲突后进入复核。

如果两个 worker 给出同一个结论但引用的是同一份错误文档，多数票没有增加证据强度。合并器应检查证据独立性、来源版本和是否存在反例；高风险动作还要使用独立 verifier 或人工确认。

### 14.11.3 成本与容量

设 `n` 个 worker 每个消耗 `c_i`，通信和合并消耗 `c_m`，则总成本近似为：

```math
C_{\mathrm{swarm}}=\sum_{i=1}^{n}c_i+c_m
```

Agent Swarm 的 wall-clock 取决于 coordinator 的派发、最慢关键 worker、artifact merge 和 verifier，而不是只看 worker 的最大耗时。共享 GPU 时要观测队列、KV 和工具连接争用；独立加载模型时要把显存和冷启动成本算入总账。部署前应按 worker 数、峰值并发和共享工作区压力做容量实验。

### 14.11.4 失败和取消

Swarm 取消不只是停止一个进程：coordinator 还要标记 artifact 是否已发布、工具副作用是否已提交、子任务是否仍在运行，以及共享 workspace 是否产生未合并 revision。只有这些状态都可观测，系统才能安全地重试、合并或交给人工，而不是让 worker 继续写入已结束的任务。

## 14.12 并行任务的合并协议

并行 Agent 不应直接共享未经版本控制的自然语言记忆。每个 worker 应输出带 owner、revision、来源、测试和冲突状态的 artifact；合并器根据任务 schema 和优先级处理冲突。

如果两个 worker 修改同一个文件或同一个事实，系统应暂停自动合并，要求重算、测试或人工选择。并行度增加后，协调和重试成本可能超过节省的模型时间。

## 14.13 租约、取消和部分完成

并行 worker 需要租约（lease）避免一个已经失联的节点继续写入共享资源。租约应绑定 task id、workspace revision、过期时间和 owner；续租失败后，coordinator 先停止新的副作用，再等待或回收 worker。只杀掉进程而不处理已提交的 artifact，可能让后续重试读到半成品。

取消也要分阶段：尚未开始的任务可以直接标记取消，正在计算的任务要回收临时资源，已产生外部副作用的任务则必须查询提交状态并执行补偿或人工接管。`cancelled` 不等于 `rolled_back`，状态机必须把两者分开。

## 14.14 并行收益的真实验收

设单 Agent 成功率为 `p_1`，swarm 成功率为 `p_s`，单位成功成本分别为 `c_1` 和 `c_s`。只有在 `p_s` 的提升足以覆盖 `c_s`、协调延迟、人工接管和副作用风险时，并行才有产品价值：

```math
U_s=p_sV-c_s-\lambda R_s,
\qquad
U_1=p_1V-c_1-\lambda R_1
```

其中 `V` 是一次成功任务的业务价值，`R` 是风险或人工成本的归一化表示。实验要固定总预算，做 worker 数、串并行结构、reviewer、共享 memory、重试和取消的消融。否则增加 worker 后分数上升，可能只是因为系统花了更多钱。

## 14.15 从任务 DAG 到关键路径

Agent Swarm 首先是一个有依赖关系的任务图，而不是“同时启动几个聊天窗口”。如果任务被拆成节点 `v`，每个节点有计算、工具和通信时间，整体 wall-clock 主要由关键路径决定：

~~~math
T_{\mathrm{wall}}
\approx T_{\mathrm{dispatch}}
 +\max_{p\in\mathcal{P}}\sum_{v\in p}T_v
 +T_{\mathrm{merge}}+T_{\mathrm{verify}}.
~~~

并行只会减少互不依赖节点的重叠时间，不能消除串行的计划、合并、审批和提交。若三个 worker 都需要读取同一份巨大上下文，通信和复制可能让 `T_dispatch` 反而增加。先画 DAG，再判断哪些边可以切断，通常比先决定 worker 数量更重要。

## 14.16 artifact 是协作接口，不是聊天记录

worker 的自然语言回复适合人读，不适合可靠合并。建议把产物定义成稳定 schema：

```json
{
  "claim_id": "retrieval-07",
  "owner": "worker-search-2",
  "revision": "repo@abc123",
  "claim": "旧版政策仍被召回",
  "evidence_ids": ["doc-19", "trace-884"],
  "tests": [{"name": "version_filter", "status": "failed"}],
  "assumptions": ["索引在 10:00 已刷新"],
  "conflicts": [],
  "proposed_action": "按 effective_at 过滤后重放"
}
```

`claim`、证据、假设、测试和建议动作分开后，主控可以只合并已验证的事实，把未解决冲突留给 verifier。artifact 还要有 owner、revision 和过期时间，避免重试 worker 把旧结论写回新任务。

## 14.17 冲突合并需要证据独立性

两个 worker 得到相同答案，不一定意味着证据更强。如果它们都读取了同一份错误网页，所谓多数票只是重复引用。合并器至少应检查：来源是否独立、版本是否相同、观察是否来自不同工具、是否存在反例、测试是否真的执行。

可以把两个 claim 的合并状态分为 `agree_verified`、`agree_shared_source`、`conflict_resolve` 和 `insufficient_evidence`。只有第一类适合自动进入低风险报告；涉及生产改动、支付或安全结论时，即使多个 worker 一致，也要经过独立 verifier 或人工确认。

## 14.18 租约、取消和 exactly-once 副作用

并行 worker 需要租约，租约绑定 task、workspace revision、owner 和过期时间。租约过期后，coordinator 应先阻止新副作用，再决定回收临时计算；不能仅仅杀掉进程，因为工具调用可能已经提交。

外部动作最好由单独的 executor 串行提交，并使用幂等键：

~~~math
\mathrm{commit}(a,k)=
\begin{cases}
\mathrm{return\ existing\ result}, & k\text{ 已提交};\\
\mathrm{execute\ once}, & k\text{ 未提交且授权有效};\\
\mathrm{hold\ for\ review}, & \text{状态或授权未知}.
\end{cases}
~~~

这样多个 worker 可以并行分析和生成候选，但不能并行对同一个生产资源执行副作用。`cancelled`、`committed`、`rolled_back` 和 `unknown` 必须是不同状态，不能用一个布尔值表示。

## 14.19 预算分配和冗余执行

并行度越高，越容易出现 token、GPU、工具连接和人工审核队列的竞争。对一个总预算 `B`，可以先为每个角色分配上限：

~~~math
\sum_{i=1}^{n}B_i+B_{\mathrm{merge}}+B_{\mathrm{verify}}
\le B.
~~~

只有当早期结果显示不确定性仍高时，才增加第二个 worker 或扩大搜索；如果第一个 worker 已经得到充分证据，继续复制任务只是增加成本。对慢 worker 可以设置 deadline 和取消策略，但取消前必须查询工具副作用状态。所谓“ultra”或“swarm”如果没有固定总预算对照，不能证明是在提高效率。

## 14.20 结构消融才能证明并行有价值

评估至少比较 direct、单 Agent、串行多 Agent、并行多 Agent 和带 verifier 的并行多 Agent。固定任务集合、模型版本、工具权限、总 token、最大 wall-clock 和成功判据，记录：

| 维度 | 要回答的问题 |
| --- | --- |
| 成功率 | 并行是否真的减少错误 |
| 单位成功成本 | 多花的 token 和 GPU 是否值得 |
| p50/p95 wall-clock | 关键路径是否缩短 |
| 通信与合并时间 | 协调是否吃掉收益 |
| 证据支持率 | 结论是否有独立依据 |
| 副作用错误率 | 并行是否放大风险 |
| 人工接管率 | 系统是否变得更难治理 |

还要做 worker 数、共享记忆、reviewer、重试、取消和 workspace 隔离的消融。只比较最终准确率而不固定预算，会把“用了更多计算”误判成“协作方法更好”。

## 14.21 什么时候不该用 swarm

强顺序任务不适合勉强并行：每一步都改变下一步输入、共享状态频繁更新、动作不可逆、任务很短或验证成本高于执行成本。此时一个有明确状态机的 Agent 可能比 swarm 更可靠。

即使适合并行，也应限制并行范围。研究阶段可以让多个 worker 读取不同来源，提交阶段则回到单一 owner；代码修改可以在隔离 workspace 中并行，合并和发布必须串行；安全判断可以用不同模型交叉检查，但最终授权应由独立 policy gate 负责。

## 14.22 协作消息要有类型和所有权

worker 之间传递的不是任意聊天文本，而应是有类型的消息：证据、假设、待验证结论、patch、测试结果、阻塞原因和取消通知。每条消息绑定 owner、输入 revision、产物 hash 和有效期，合并器才能判断它是否仍然适用。

例如，一个研究 worker 只能提交“引用了三份资料的候选结论”，不能直接把结论标记为已发布；一个代码 worker 可以提交 patch 和测试日志，但不能替另一个 worker 宣布生产部署。所有权边界越清楚，swarm 越容易审计和回滚。

## 14.23 并行系统的协调账

假设四个 worker 各自耗时 10 秒，独立并行理想耗时约 10 秒；若每个 worker 还需要两次 1 秒消息同步、一次 3 秒冲突合并和一次 5 秒验证，总耗时至少接近 20 秒。此时并行仍可能有价值，但不能只报告“worker 计算时间”。

可以把总成本写成：

```math
C_{\mathrm{swarm}}
=\sum_i C_i+C_{\mathrm{message}}+C_{\mathrm{merge}}
+C_{\mathrm{verify}}+C_{\mathrm{wasted}}.
```

其中 `C_wasted` 包括被取消、重复和冲突后丢弃的工作。它经常在高并发时被忽略，却会直接占用 GPU、上下文和工具配额。

## 14.24 从候选到提交的单一闸门

并行 worker 可以同时探索，但外部提交应回到单一的 policy gate。gate 检查候选是否来自允许版本、证据是否独立、冲突是否解决、测试是否通过、权限是否仍然有效以及动作是否具备幂等键。任何 worker 都不能绕过 gate 直接写生产状态。

这也解释了为什么 swarm 不等于“更多自主权”。并行增加搜索宽度，提交闸门仍然负责把不确定的候选转成受控动作。

## 14.25 swarm 的可观测性

每个 worker 需要有 task id、parent id、输入 artifact、输出 artifact、状态、预算、租约和取消原因。合并 trace 要能回答哪个 worker 提供了最终证据、哪个候选被丢弃、冲突如何解决以及验证耗时多少。只保存总 Agent 的最终文本无法做失败归因。

## 14.26 并行协作的降级路径

worker 超时或冲突时，可以减少并行度、改为只读探索、交给单 Agent 合并或转人工。降级过程不能自动扩大剩余 worker 的权限，也不能把未验证候选当成已完成 artifact。对有外部副作用的任务，提交阶段始终保持单一 owner。

## 14.27 小结与资料边界

Agent Swarm 的核心是协作协议，而不是 worker 数量。结构化 artifact、只读证据、隔离写入、独立 verifier、最小权限、租约、幂等提交和可观测的合并过程，决定了并行是否真的提高任务成功率。

本章的 DAG、协作、sandbox 和 Agent tracing 是通用工程模型。Anthropic 的 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 可作为 agent workflow 的公开参考；具体产品对 swarm、parallel agents 或 worker role 的命名和实现可能不同，若官方没有公开细节，不应从名称推断内部架构。
