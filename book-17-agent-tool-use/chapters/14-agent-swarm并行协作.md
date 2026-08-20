# 第 14 章 Agent Swarm：并行协作的通信、隔离与合并

## 14.0 从多个 Agent 到协作系统

一个 Agent 同时负责检索、规划、执行、验证和写报告时，可能遇到上下文过长、工具过多、角色冲突和单一路径错误。把任务拆成多个专门 Agent，确实可以让不同节点分别处理检索、代码分析、事实核查和结果整理。

但把同一个 prompt 复制十份，不叫协作。多个 Agent 之间还需要通信、共享证据、隔离可变状态、处理冲突、管理预算，并决定谁有权改变外部世界。没有这些机制，系统只是并行生成多段文本，最后仍然要靠一个模型猜哪段更可信。

本章把这种系统称为 Agent Swarm。这里的 swarm 不是一个固定产品名，而是一种协作架构：多个具有明确角色和权限的策略节点，围绕任务图生产 artifact，由协调器或固定 workflow 合并，并由独立 verifier 判断是否可以产生外部副作用。

因此，评估协作系统时要区分“生成了多少份结果”和“增加了多少独立证据”。如果所有 worker 使用相同模型、相同检索结果和相同错误假设，十票赞成也可能只是一次错误被复制了十遍。并行只有在任务依赖、观察来源、权限边界和验收责任都被明确时，才可能带来可解释的收益。

## 14.1 为什么要把一个 Agent 拆成多个

拆分通常有四类正当理由。

第一，子任务可以真正并行。例如分别读取三份互不依赖的合同、分析日志和代码变更，最后再合并结果。并行可以缩短等待外部服务和长文档读取的时间。

第二，上下文或权限需要隔离。研究节点只需访问公开资料，代码节点只需访问临时仓库，提交节点才可以请求特定业务接口。把所有上下文和工具放进一个 Agent，会扩大数据暴露和错误副作用。

第三，需要不同的验证视角。一个节点提出候选解释，另一个节点根据测试、数据库状态或第二来源进行反驳；这种分工有价值的前提是验证者拥有不同的证据或检查机制，而不是只重复同一段生成。

第四，系统需要不同的工具适配。代码节点可能需要编译器，浏览器节点需要页面观察，数据节点需要查询引擎。把工具和角色绑定，可以缩小每个节点的决策空间。

以下理由通常不足以支持拆分：

- 角色名称听起来更复杂；
- 想通过投票掩盖没有 verifier 的事实；
- 单 Agent 只是提示词太长，却没有先整理状态和 artifact；
- 想用多个低成本节点替代一个明确的外部验收器；
- 只比较最终答案，不比较总 token、延迟和副作用。

是否拆分，首先要建立单 Agent 或固定 workflow baseline，再观察并行是否解决了一个可定位的瓶颈。

## 14.2 单 Agent、Workflow 与 Swarm 的边界

三种结构的主要差异不在于是否调用了大模型，而在于谁决定下一步、状态怎样共享、是否允许角色并行。

| 结构 | 决策方式 | 状态共享 | 适用场景 | 主要风险 |
|---|---|---|---|---|
| 单 Agent | 一个策略循环选择动作 | 一个上下文或状态仓 | 工具少、任务短、责任集中 | 上下文膨胀、单点错误 |
| 固定 Workflow | 代码或状态机规定步骤 | 明确的中间 artifact | 步骤稳定、验收清晰 | 对变化适应差 |
| Swarm | 多个角色按任务图协作 | 受契约约束的消息和索引 | 可并行、权限隔离、需要交叉检查 | 通信、冲突和责任归因 |

Swarm 不一定比固定 workflow 更自主，也不一定比单 Agent 更可靠。一个合同任务如果步骤已经稳定，固定 workflow 可能更容易审计；一个需要从多个来源收集证据的研究任务，局部并行可能更有效；一个会直接修改生产对象的任务，则应该把并行限制在只读分析阶段。

## 14.3 用任务图表达依赖

把 Agent 看成节点，把消息、artifact 和依赖看成边，可以得到协作图：

~~~math
\mathcal{G}=(V,E),\qquad
V=\{\mathrm{planner},\mathrm{researcher},\mathrm{builder},\mathrm{tester},\mathrm{reviewer}\}
~~~

如果节点 u 的输出是节点 v 的输入，就存在依赖边 u→v。没有依赖的节点可以并行；需要前一节点产物的节点必须等待。真正的任务图应该表示 artifact 和版本，而不是只表示角色名称。

例如，合同助手可以有以下依赖：

~~~~text
contract_read ─┐
policy_read   ─┼─> rule_check ─> ticket_prepare ─> external_verify
permission    ─┘
~~~~

contract_read、policy_read 和 permission 可以在满足权限条件时并行；rule_check 必须等到三者的结果都达到所需状态；ticket_prepare 只能接收已验证的合同和政策版本；external_verify 必须重新读取工单，而不能只相信 ticket_prepare 的返回文本。

任务图还要区分数据依赖和副作用依赖。两个节点可以共享同一份只读合同，但不能同时对同一个订单执行写操作。把所有节点都连接成串行图会失去并行收益；把有写冲突的节点都标成并行，则会把竞争条件当成优化。

## 14.4 角色契约、所有权和状态

一个 worker 不应只接收一句自然语言任务。任务消息至少要包含：

- task_id 和 parent_id；
- 输入 artifact 及其 revision；
- 输出 artifact 的 schema；
- 可见上下文和权限范围；
- 预算、deadline 和取消规则；
- 允许的外部副作用；
- owner、验收条件和失败状态。

worker 的所有权表示谁负责产出和解释结果，不表示它可以拥有所有工具。researcher 可以拥有公开资料读取权限，tester 可以拥有隔离环境中的执行权限，merge worker 可以写共享索引，但都不应因为收到一条消息就自动获得生产写权限。

一个完整的状态集合至少包括 queued、running、succeeded、failed、blocked、cancel_requested、cancelled 和 unknown。unknown 用于表示工具或外部动作的结果尚未确定；如果把它压成 failed，协调器可能重复执行已经成功的副作用。

这些状态不是按“好坏”排列的分数。`succeeded` 表示该节点的输出已经满足自己的验收条件，`failed` 表示验收器已经确认不满足，`blocked` 表示缺少继续执行所需的输入或权限；`unknown` 则表示证据不足，不能自动转化为成功或失败。只有完成状态转换和相应的外部查询后，协调器才可以决定重试、补充输入或结束任务。

### 14.4.1 责任边界的具体例子

在代码 swarm 中：

- call-graph worker 对调用链 artifact 负责；
- repro worker 对最小复现日志负责；
- patch worker 对候选 diff 负责；
- test worker 对测试命令和环境结果负责；
- security worker 对路径、依赖和敏感数据检查负责；
- merge owner 只负责合并经过验证的 artifact；
- external executor 负责实际提交，并重新检查权限和版本。

这样，最终失败可以追问到某一类 artifact，而不是笼统地说“Agent 没做好”。如果一个 worker 的输出没有 owner、版本和验收条件，它就不适合进入共享状态。

## 14.5 消息不是聊天，artifact 才是协作接口

自然语言回复适合人读，却不适合可靠合并。消息应该区分事实、证据、假设、待验证结论、patch、测试结果、阻塞原因和取消通知，并带有来源和有效期。

一个研究 worker 可以发布：

~~~~json
{
  "kind": "evidence_claim",
  "task_id": "research-17",
  "parent_id": "report-3",
  "claim_id": "cache-race",
  "claim": "cache refresh may race with read",
  "evidence_ids": ["trace-42", "src/cache.py:81"],
  "status": "hypothesis",
  "owner": "worker-a",
  "input_revision": "repo@abc123",
  "expires_at": "2026-08-14T12:00:00Z"
}
~~~~

这条消息只表示候选假设，不表示事实已经验证，更不表示允许执行修复。后续 verifier 可以发布同一 claim_id 的新 revision，并说明使用了什么测试或第二来源。

消息协议还要处理重放、乱序和过期。可以为消息设置 message_id、parent_ids、artifact_hash、monotonic_revision 和 idempotency_key。接收者先检查租户、权限和输入版本，再决定是否接受。一个晚到的旧 revision 不能覆盖已经验证的新 revision；一个过期的授权消息不能重新开启外部写入。

## 14.6 共享记忆和隔离工作区

协作中最难的取舍是“让大家看到足够信息”和“不要让大家同时污染状态”。一个实用设计是：

- blackboard 只保存 claim、artifact 引用、状态索引、owner 和版本；
- 原始文档、完整对话和敏感字段留在受权限控制的存储；
- 研究 worker 共享只读证据；
- 每个可写代码 worker 使用独立 workspace 或分支；
- 只有合并 owner 生成合并 revision；
- 外部副作用由独立 executor 串行处理。

共享 memory 如果直接保存自然语言总结，错误会迅速扩散。假设一个 worker 把“索引可能未刷新”写成确定事实，后续节点就可能跳过版本检查，patch worker 甚至会针对错误原因修改代码。状态字段必须明确区分 fact、hypothesis、unverified 和 conflict。

### 14.6.1 workspace 隔离的原因

两个代码 worker 同时修改同一文件，会出现三类问题：

1. 一个 worker 看到了另一个 worker 的半成品，导致实验不可复现；
2. 后写入的内容覆盖先写入的 patch，责任无法归因；
3. 测试结果无法绑定到具体 diff，合并器不知道哪个版本通过了检查。

独立 workspace 让每个 worker 的输入、diff 和测试结果形成闭环。合并时不应直接复制最终文件，而应比较基线、路径、补丁、冲突和测试。若两个 patch 修改同一行，系统应停止自动合并，重新基于共同基线重算。

## 14.7 Map—Reduce 结构：并行读取，集中合并

很多研究任务适合用 map—reduce 思路理解。map 阶段让多个 worker 分别读取不同来源，输出统一 schema 的候选证据；reduce 阶段按 claim_id 聚合，检查来源版本、冲突和证据独立性；最后 verifier 再决定哪些结论可以发布。

例如用户要求比较三家公司对某技术的公开声明：

1. planner 为每家公司建立独立 task；
2. researcher 只记录官方来源、发布时间、原文 claim 和定位；
3. auditor 检查来源级别、版本和是否混入媒体转述；
4. reducer 按同一 claim_id 合并；
5. verifier 处理冲突和缺失；
6. writer 只读取经过验证或明确标为不确定的 artifact。

如果某项“架构结论”只有社区截图支持，auditor 应把它标为 unverified。writer 可以写“公开资料尚不足以确认”，但不能把截图内容改写成确定事实。并行的价值在于减少独立读取的等待，而不是让更多节点共同制造确定语气。

## 14.8 代码 swarm：把并行限制在候选生成阶段

一个可审计的代码 swarm 可以这样组织：

~~~~text
call_graph ─┐
reproduce   ├─> patch_candidates ─> isolated_tests
security    ─┘                         │
                                      └─> merge_review ─> external_verify
~~~~

call_graph worker 只读仓库构建调用链；reproduce worker 创建最小复现；security worker 检查权限和敏感路径；patch worker 根据 artifact 提出候选 diff；test worker 在各自隔离 workspace 中运行测试；merge_review 比较候选与共同基线；external_verify 重新读取最终状态。

patch worker 不应直接修改共享生产分支。它提交的是 candidate patch、基线 hash、测试日志和未解决假设。若 repro worker 失败，这个失败 artifact 也有价值：它说明当前复现条件不足，主控可以只重试该节点或请求人工补充信息，而不是重新启动整个 swarm。

合并候选时，至少检查：

- 是否基于同一初始 commit；
- 是否触碰用户已有改动；
- 修改范围是否符合 task contract；
- 测试是否对这份 diff 执行；
- 依赖和生成文件是否变化；
- 是否存在未验证的外部命令副作用。

## 14.9 冲突合并：多数票不是证据

两个 Agent 得到矛盾结论时，合并器不能只按文本相似度或多数票选择。应该比较来源新鲜度、来源权威性、实验可复现性、任务相关性、证据独立性和是否存在反例。

可以给候选 claim 一个排序分数：

~~~math
Q(c)=w_sS(c)+w_rR(c)+w_vV(c)-w_uU(c)
~~~

S 表示来源质量，R 表示与任务的相关性，V 表示是否被可复核测试验证，U 表示不确定性；各项应在约定的有限区间内，w_s、w_r、w_v、w_u 是非负权重。若某个候选没有来源版本、验证结果或不确定性记录，Q(c) 就不应被伪造为一个完整数值，可以把它标为 unknown 并优先补证据。这个分数只能帮助安排复核顺序，不能替代 verifier。

独立支持也需要单独定义。若每个证据记录都带有经过审计的 `source_id`，可以计算候选使用了多少个来源组：

~~~math
N_{\mathrm{source}}(c)=
\left|\{\mathrm{source\_id}(e):e\in E(c)\}\right|
~~~

其中 E(c) 是支持 claim c 的证据集合。N_source 只是来源分组数量，不自动证明统计独立、观点独立或没有共同上游缓存；如果 source_id 缺失、版本无法核对或多个来源共享同一原始数据，独立性应记为 unknown，而不是直接增加计数。两个 worker 如果读取了同一份错误文档，它们的答案并不独立；十个 worker 重复同一错误，仍然只有一个错误来源。

合并状态可以明确区分：

| 状态 | 含义 | 后续动作 |
|---|---|---|
| agree_verified | 独立证据一致且外部检查通过 | 可进入低风险结果 |
| agree_shared_source | 结论一致但共享同一来源 | 继续寻找独立检查 |
| conflict_resolve | 证据互相矛盾 | 重算、补充来源或人工判断 |
| insufficient_evidence | 证据不足 | 降级说明，不生成确定结论 |

涉及生产改动、支付、删除或安全结论时，即使多个 worker 一致，也应由独立 verifier 或人工确认。

## 14.10 协调器：调度、重试和停止

coordinator 的职责不是把所有文本拼在一起，而是维护任务图和 artifact 状态。它需要：

1. 根据依赖检查哪些节点可以启动；
2. 为节点分配输入 revision、权限和预算；
3. 记录 heartbeat、deadline 和租约；
4. 接收结构化 artifact；
5. 只重试可重试且没有未知副作用的节点；
6. 处理冲突、超时、取消和部分完成；
7. 把可提交动作交给独立 executor；
8. 在最终报告中保留失败和未验证信息。

协调器如果拥有所有工具，很容易变成一个没有约束的超级 Agent。更安全的设计是让它只能调度和合并索引，不能直接越过执行器写入生产对象。每个外部 action 前，executor 重新检查主体、资源、版本和幂等键。

### 14.10.1 节点状态转移

可以把单个节点写成：

~~~~text
queued -> running -> succeeded
                  ├-> failed -> retryable -> queued
                  ├-> blocked -> needs_input
                  ├-> cancel_requested -> cancelled
                  └-> unknown -> query_state -> resolved
~~~~

retryable 只适用于参数已修正或服务明确返回临时故障的情形。unknown 必须先查询外部状态；如果节点创建了工单但响应丢失，不能直接回到 queued。blocked 说明当前信息或权限不足，需要补充证据或人工输入，不应通过增加 worker 数量掩盖。

## 14.11 租约、取消和部分完成

并行 worker 需要租约，租约绑定 task_id、workspace_revision、owner 和过期时间。租约过期后，coordinator 先停止新的副作用，再决定回收临时计算；不能只杀掉进程，因为工具调用可能已经提交。

取消分为不同阶段：

- 尚未开始的任务可以标记 cancelled；
- 正在计算的任务要回收临时资源并保存已产生的 artifact；
- 已发送外部请求的任务要查询请求状态；
- 已产生不可逆副作用的任务要执行补偿或人工接管；
- 共享索引要标记 artifact 是否发布、过期或需要撤销。

cancelled 不等于 rolled_back。一个 worker 可以被取消，但它已经提交的工单仍然存在；一个 workspace 可以回滚，但外部邮件已经发送。状态机必须保留这些差异，才能选择正确的恢复动作。

### 14.11.1 exactly-once 副作用的工程表达

并行分析可以很宽，生产写入必须很窄。外部 executor 可以用幂等键 k 记录提交结果：

~~~math
\mathrm{commit}(a,k)=
\begin{cases}
\mathrm{return\ existing\ result}, & k\in K_{\mathrm{committed}}\land H_k=H(a);\\
\mathrm{reject\ key\ conflict}, & k\in K_{\mathrm{committed}}\land H_k\ne H(a);\\
\mathrm{hold\ for\ review}, & k\notin K_{\mathrm{committed}}\land \mathrm{state}(a)=\mathrm{unknown};\\
\mathrm{execute\ once}, & k\notin K_{\mathrm{committed}}\land \mathrm{state}(a)\ne\mathrm{unknown}\land A(a)=1.
\end{cases}
~~~

这个表达描述的是业务层的幂等语义，不意味着底层网络天然提供 exactly-once。系统仍要处理请求到达但响应丢失、executor 重启、数据库提交和审计写入不一致等情况。

这里的 K_committed 是已经被持久化记录的幂等键集合，H_k 是记录在该键上的规范化 action hash，H(a) 是当前请求的 hash；A(a)=1 表示 action a 的 schema、权限、版本、前置条件和预算都已通过独立检查；`state(a)` 是外部状态查询的结果。相同幂等键对应不同动作时，不能返回旧结果，否则会把客户端错误隐藏成成功。若 k 不存在但 state(a)=unknown，不能仅因为“尚未看到提交记录”就执行写入。提交前应检查候选 artifact 的基线、证据、测试、冲突和当前权限；提交后要查询外部对象并记录 verified_state。多个 worker 可以并行提出候选，但不应并行对同一个生产资源执行写操作。

## 14.12 通信成本和上下文传输

设 n\ge 0 个 worker 的本地成本为 C_i，消息和上下文传输成本为 C_message，合并与验证成本为 C_merge、C_verify，重复或丢弃工作为 C_wasted，则总成本可以近似写成：

~~~math
C_{\mathrm{swarm}}
=\sum_{i=1}^{n}C_i
+C_{\mathrm{message}}
+C_{\mathrm{merge}}
+C_{\mathrm{verify}}
+C_{\mathrm{wasted}}
~~~

这些成本项都必须使用同一计量单位或经过明确的成本换算，并且每一项都要有实际观测；缺少消息、合并或验证成本时，总成本不能被写成精确的 0。若 n=0，公式中的 worker 求和为 0，但这只表示没有启动 worker，不表示任务已经完成。如果每个 worker 都接收完整历史，C_message 会随上下文复制迅速增长。更好的方式是传递最小必要字段和 artifact 引用，worker 需要原文时再按权限拉取。引用必须带 revision 和 hash，否则同一个 artifact 可能在重试期间被悄悄替换。

四个 worker 各自耗时 10 秒，理想并行计算接近 10 秒；如果还需要协调 3 秒、两轮消息 2 秒、冲突合并 4 秒和验证 5 秒，整体可能接近 24 秒。并行依然可能提高成功率，但不能只报告 worker 计算时间。

## 14.13 关键路径和并行收益

若任务图中每个节点 v 的计算、工具和等待时间为 T_v，关键路径集合为 P，则墙钟时间可近似写成：

~~~math
T_{\mathrm{wall}}
\approx T_{\mathrm{dispatch}}
+\max_{p\in\mathcal{P}}\sum_{v\in p}T_v
+T_{\mathrm{merge}}
+T_{\mathrm{verify}}
+T_{\mathrm{queue}}
~~~

这里的 P 是从可启动节点到终止节点的所有有向路径集合，T_v 必须包含该节点实际的计算、工具等待和重试时间；dispatch、queue、merge 和 verify 也必须使用同一个时间口径。非法或不完整的任务图没有可解释的关键路径，T_wall 应记为 unknown，而不是填 0。并行只会减少互不依赖节点的重叠时间，不能消除计划、合并、审批和提交等串行阶段。若所有 worker 都等待同一个慢 API，增加 worker 只会增加队列；若所有 worker 都把长报告发送给 reviewer，通信和合并会吞掉收益。

所以设计顺序应是：先画依赖图，再估算关键路径和共享资源，最后决定 worker 数量。不要先选一个看起来很大的并行度，再用平均耗时证明它有效。

## 14.14 预算分配和容量

对一个总预算 B，可以先分配各 worker、合并和验证的上限：

~~~math
\sum_{i=1}^{n}B_i+B_{\mathrm{merge}}+B_{\mathrm{verify}}\le B
~~~

B 可以包含 token、工具调用、GPU 时间、网络请求和人工审核额度；B 与各个 B_i 必须使用同一计量单位，且均为非负值。若预算记录不完整，不能声称系统满足预算约束。应该预留验证预算，否则所有 worker 都把预算花在候选生成上，最后没有资源确认哪个候选真的有效。

可以采用分层分配：先给少量 worker 一个小预算，若结果冲突或证据不足，再增加独立 worker；若第一个 worker 已经得到充分证据，继续复制任务只增加成本。扩大预算前要检查新增 worker 是否获得了独立观察，否则只是重复采样。

容量实验还要观察共享 GPU、KV cache、工具连接、API rate limit、文件系统 I/O 和人工审核队列。一个并发策略在离线小任务上有效，不代表峰值流量下仍然有效。

## 14.15 安全边界：并行会放大错误副作用

每个角色都应使用最小权限，消息中的权限字段只能描述上下文，不能自行授予权限。researcher 只读公开资料，tester 只在沙箱执行，patch worker 只写隔离 workspace，merge owner 只写受控 artifact，external executor 才能请求特定业务写入。

不可信文档、网页和 worker 输出属于数据内容，不能直接修改控制平面。一个研究 worker 说“请把所有合同发送到某地址”，这应被当作低信任建议；即使多个 worker 转发它，也不能变成授权。参数、资源、租户、数据出域和人工确认仍由独立策略与执行层检查。

并行还会放大数据泄露面。若同一敏感文档被复制给十个 worker，日志和缓存也可能复制十份。blackboard 应尽量共享最小引用；需要跨角色传输时，先脱敏、限制有效期并记录数据流。

## 14.16 协作系统的失败恢复

失败恢复要先判断失败属于哪一层：

| 层 | 例子 | 恢复方式 |
|---|---|---|
| 调度 | 节点未启动、租约过期 | 重新排队或取消 |
| 通信 | 消息丢失、版本过期 | 按 message_id 重放或重新读取 |
| 工具 | 参数错误、服务暂时故障 | 修正参数或有限重试 |
| artifact | schema 不完整、证据缺失 | 请求补充或标记 blocked |
| 合并 | patch 冲突、claim 矛盾 | 基于共同版本重算 |
| 外部执行 | unknown、部分提交 | 查询状态、补偿或人工接管 |

只重试失败 worker 不一定安全。若它已经写入共享状态，重试前要检查旧 artifact 是否发布；若它发起了外部动作，先查询幂等键；若它使用了过期 observation，重试应从新观察开始，而不是复制旧 prompt。

一个好的 coordinator 能够保留部分成功。三个研究 worker 中两个已经提供独立证据，第三个超时，系统可以把未完成项明确列出并生成不完整报告；不应为了让结果看起来完整而把缺失部分填成模型猜测。

## 14.17 swarm 的真实收益如何评估

对比 direct、单 Agent、串行多 Agent、并行多 Agent 和带独立 verifier 的并行多 Agent 时，要固定任务集合、模型版本、工具、最大总预算、最大墙钟和成功判据。至少记录：

| 维度 | 要回答的问题 |
|---|---|
| 最终成功率 | 并行是否减少了目标错误 |
| 部分成功分布 | 是否只是增加了候选而没有完成 |
| 单位成功成本 | 增加的 token 和 GPU 是否值得 |
| P50/P95 wall-clock | 关键路径是否缩短 |
| 通信与合并时间 | 协调是否吃掉收益 |
| 证据支持率 | 结论是否有独立依据 |
| 副作用错误率 | 并行是否放大风险 |
| 人工接管率 | 系统是否更难治理 |

还要做结构消融：去掉 reviewer、去掉共享 memory、把并行改成串行、改变 worker 数、限制角色上下文、取消 workspace 隔离、去掉独立 verifier。若去掉 reviewer 后结果不变，reviewer 可能只是增加成本；若增加 worker 只提高未验证答案数量，协议没有创造真正价值。

### 14.17.1 用净收益而不是角色数量做结论

设相对于单 Agent baseline 的质量变化为 ΔQ，额外成本为 ΔC，额外延迟为 ΔL，风险变化为 ΔR，则可以用一个分析式表达净收益：

~~~math
\mathrm{NetLift}
=\Delta Q-\lambda_c\Delta C-\lambda_l\Delta L-\lambda_r\Delta R
~~~

λ_c、λ_l 和 λ_r 由业务场景决定。对于离线低风险研究，适度增加成本可能换来更高的证据覆盖；对于支付、删除和权限变更，风险项可能是硬约束，不能用平均质量提升抵消。

这个式子不是通用评分标准，也不能把严重安全事件简单换算成 token。它的作用是迫使实验报告写清：质量提高了多少，花了多少额外资源，延迟和人工负担如何变化，风险是否发生了结构性变化。

## 14.18 什么时候不该使用 Swarm

以下情况通常不适合勉强并行：

- 任务步骤强顺序，每一步都改变下一步输入；
- 多个节点必须频繁写同一个可变状态；
- 动作不可逆，且没有独立的提交 executor；
- 任务很短，通信和合并成本超过执行成本；
- verifier 比候选生成更昂贵，预算不足以验收；
- 角色没有真正不同的工具、观察或责任；
- 结果必须由一个明确 owner 连续负责。

即使适合并行，也应限制并行范围。研究阶段可以让多个 worker 读取不同来源，提交阶段回到单一 owner；代码修改可以在隔离 workspace 中并行，合并和发布必须串行；安全判断可以交叉检查，但最终授权仍由独立执行层负责。

## 14.19 可观测性：能追溯每个 artifact 的来路

每个 worker 都需要 task_id、parent_id、input_revision、output_artifact、状态、预算、租约和取消原因。合并 trace 要能回答：

- 哪个 worker 提供了最终证据；
- 哪些候选被丢弃，为什么；
- 两个结论是否真的使用了独立来源；
- 冲突如何解决；
- 外部提交由谁发起，基于哪个版本；
- 取消时是否已经产生副作用；
- verifier 使用了什么状态和版本。

完整对话可以保留在各自 trace，但共享索引只应保存协作所需的最小字段。这样既能做失败归因，也能减少跨角色数据复制。

## 14.20 最小可运行的 Swarm 协作审计 demo

下面的 demo 使用标准库的线程池模拟三个独立读取 worker。它不调用外部模型，也不声称实现了生产级调度；它只演示三个设计点：worker 返回结构化 artifact，合并器检查证据是否来自不同来源，候选结论即使得到一致支持，也不能直接获得外部提交权限。

~~~~python
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import sleep


def require_text(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")


def require_bool(value, field_name):
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool")


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_group: str
    upstream_group: str
    revision: str
    independence_audited: bool

    def __post_init__(self):
        for field_name in (
            "evidence_id",
            "source_group",
            "upstream_group",
            "revision",
        ):
            require_text(getattr(self, field_name), field_name)
        require_bool(self.independence_audited, "independence_audited")


@dataclass(frozen=True)
class Task:
    task_id: str
    owner: str
    claim_id: str
    claim: str
    evidence: tuple
    revision: str

    def __post_init__(self):
        for field_name in ("task_id", "owner", "claim_id", "claim", "revision"):
            require_text(getattr(self, field_name), field_name)
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise ValueError("task evidence must be a non-empty tuple")
        if any(not isinstance(item, Evidence) for item in self.evidence):
            raise TypeError("task evidence must contain Evidence values")


@dataclass(frozen=True)
class Artifact:
    task_id: str
    owner: str
    claim_id: str
    claim: str
    evidence: tuple
    status: str
    revision: str

    def __post_init__(self):
        for field_name in ("task_id", "owner", "claim_id", "claim", "revision"):
            require_text(getattr(self, field_name), field_name)
        if self.status not in {"candidate", "verified", "failed", "unknown"}:
            raise ValueError(f"unsupported artifact status: {self.status}")
        if not isinstance(self.evidence, tuple):
            raise TypeError("artifact evidence must be a tuple")
        if any(not isinstance(item, Evidence) for item in self.evidence):
            raise TypeError("artifact evidence must contain Evidence values")
        if self.status in {"candidate", "verified"} and not self.evidence:
            raise ValueError("candidate or verified artifacts require evidence")


def worker(task):
    if not isinstance(task, Task):
        raise TypeError("worker requires a Task")
    sleep(0.01)
    return Artifact(
        task_id=task.task_id,
        owner=task.owner,
        claim_id=task.claim_id,
        claim=task.claim,
        evidence=task.evidence,
        status="candidate",
        revision=task.revision,
    )


def validate_artifacts(artifacts):
    if not isinstance(artifacts, (list, tuple)):
        raise TypeError("artifacts must be a list or tuple")
    task_ids = set()
    for artifact in artifacts:
        if not isinstance(artifact, Artifact):
            raise TypeError("artifacts must contain Artifact values")
        if artifact.task_id in task_ids:
            raise ValueError(f"duplicate task_id: {artifact.task_id}")
        task_ids.add(artifact.task_id)


def merge_artifacts(artifacts, external_verifier=False):
    validate_artifacts(artifacts)
    if type(external_verifier) is not bool:
        raise TypeError("external_verifier must be bool")
    if not artifacts:
        return {
            "artifact_count": 0,
            "source_groups": 0,
            "independent_support": 0,
            "merged_status": "insufficient_evidence",
            "commit_allowed": False,
        }

    claims = {(artifact.claim_id, artifact.claim) for artifact in artifacts}
    revisions = {artifact.revision for artifact in artifacts}
    evidence = [item for artifact in artifacts for item in artifact.evidence]
    source_groups = {item.source_group for item in evidence}
    upstream_groups = {item.upstream_group for item in evidence}
    audited = all(item.independence_audited for item in evidence)
    independent_support = (
        len(source_groups)
        if audited and len(source_groups) == len(upstream_groups)
        else 0
    )

    if len(claims) != 1 or len(revisions) != 1:
        merged_status = "conflict_resolve"
    elif any(artifact.status in {"failed", "unknown"} for artifact in artifacts):
        merged_status = "insufficient_evidence"
    elif independent_support >= 2:
        merged_status = "agree_independent"
    else:
        merged_status = "agree_shared_source"

    commit_allowed = (
        merged_status == "verified"
        and external_verifier
        and all(artifact.status == "verified" for artifact in artifacts)
    )
    return {
        "artifact_count": len(artifacts),
        "source_groups": len(source_groups),
        "independent_support": independent_support,
        "merged_status": merged_status,
        "commit_allowed": commit_allowed,
    }


tasks = [
    Task(
        task_id="read-logs",
        owner="worker-logs",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(
            Evidence(
                "trace-42",
                "logs",
                "observability-db",
                "repo@abc123",
                True,
            ),
        ),
        revision="repo@abc123",
    ),
    Task(
        task_id="inspect-diff",
        owner="worker-diff",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(
            Evidence(
                "src/cache.py:81",
                "source",
                "repository",
                "repo@abc123",
                True,
            ),
        ),
        revision="repo@abc123",
    ),
    Task(
        task_id="check-metrics",
        owner="worker-metrics",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(
            Evidence(
                "metric-2026-08-14",
                "metrics",
                "metrics-store",
                "repo@abc123",
                True,
            ),
        ),
        revision="repo@abc123",
    ),
]

with ThreadPoolExecutor(max_workers=3) as pool:
    artifacts = list(pool.map(worker, tasks))

artifacts.sort(key=lambda item: item.task_id)
merged = merge_artifacts(artifacts, external_verifier=False)
for key in (
    "artifact_count",
    "source_groups",
    "independent_support",
    "merged_status",
    "commit_allowed",
):
    print(f"{key}={merged[key]}")
~~~~

运行结果应为：

~~~~text
artifact_count=3
source_groups=3
independent_support=3
merged_status=agree_independent
commit_allowed=False
~~~~

三个 worker 提供了三组经标记且共同上游不同的 source_group，因此在这个教学构造中，合并器可以把结果标为 agree_independent；这仍然不是最终 verified。worker 只发布 candidate artifact，外部 verifier 没有运行，commit_allowed 必须为 False。真实系统还要检查来源是否真正独立、版本是否一致、测试是否反事实有效，以及提交动作的权限和幂等性。

边界测试进一步说明，来源数量不够、共享上游、重复任务和空集合都不能被误报为独立支持：

~~~~python
assert merge_artifacts([])["merged_status"] == "insufficient_evidence"

shared_upstream = [
    Artifact(
        task_id="shared-a",
        owner="worker-a",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(
            Evidence("a", "source-a", "same-cache", "repo@abc123", True),
        ),
        status="candidate",
        revision="repo@abc123",
    ),
    Artifact(
        task_id="shared-b",
        owner="worker-b",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(
            Evidence("b", "source-b", "same-cache", "repo@abc123", True),
        ),
        status="candidate",
        revision="repo@abc123",
    ),
]
assert merge_artifacts(shared_upstream)["independent_support"] == 0
assert merge_artifacts(shared_upstream)["merged_status"] == "agree_shared_source"

try:
    merge_artifacts(shared_upstream + [shared_upstream[0]])
except ValueError as error:
    assert "duplicate task_id" in str(error)
else:
    raise AssertionError("duplicate task IDs must be rejected")

try:
    Artifact(
        task_id="empty",
        owner="worker-empty",
        claim_id="cache-race",
        claim="cache refresh may race with read",
        evidence=(),
        status="candidate",
        revision="repo@abc123",
    )
except ValueError as error:
    assert "require evidence" in str(error)
else:
    raise AssertionError("candidate artifacts require evidence")
~~~~

## 14.21 从 demo 回到生产协作

生产系统不能把 evidence 数量或 source_group 数量直接当作证据质量。三个 worker 可能都从同一缓存读取同一份错误文档，也可能使用相同的模型偏差。合并器应检查来源独立性、抓取时间、版本、访问权限和反例；高风险结论还需要独立执行器、程序化测试或人工判断。

线程池也不等于生产级并发。真实系统需要租约、队列、超时、取消、资源额度、重试、持久化事件、断点恢复和外部状态查询。demo 的价值是展示最小数据结构与决策边界，而不是替代这些基础设施。

## 14.22 学习任务：把并行结构写成可验证系统

1. 为“研究—审核—写作”任务画出 DAG，标明哪些边是数据依赖，哪些边是副作用依赖。
2. 设计一个 artifact schema，分别表达事实、证据、假设、冲突、测试结果和建议动作。
3. 为两个代码 worker 设计共同基线、隔离 workspace、patch 合并和冲突回退。
4. 构造一次外部提交超时的轨迹，说明 coordinator、executor 和 verifier 各自负责什么。
5. 固定总 token 和最大墙钟，比较单 Agent、串行多 Agent、并行多 Agent 和带 verifier 的系统。
6. 在 demo 中加入一个共享错误来源，观察为什么多数票不再代表独立证据。
7. 设计一个取消流程，分别处理尚未开始、正在计算、已发布 artifact 和已产生外部副作用四种状态。

这些练习的答案应该包括状态、版本、所有权、证据、失败和恢复。只写“增加 planner、reviewer 和 worker”还没有完成系统设计。

## 14.23 资料入口与可信边界

- [Anthropic：Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)：官方工程文章，讨论 workflow、路由、并行和 evaluator 的设计取舍；它是公开设计经验，不是所有生产系统的安全保证。
- [OpenAI Agents SDK：Handoffs](https://openai.github.io/openai-agents-python/handoffs/)：官方 SDK 中 handoff 的接口语义入口；接口支持角色转移，不等于自动具备权限隔离、外部状态验收和幂等写入。
- [Microsoft AutoGen：Selector Group Chat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html) 与 [Swarm](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/swarm.html)：官方框架文档，支持理解多角色消息和选择机制；框架 API 不能替代任务契约和独立 verifier。
- [MapReduce 论文入口](https://www.usenix.org/legacy/events/osdi04/tech/full_papers/dean/dean_html/)：经典 map—reduce 并行数据处理模型，可帮助理解“并行读取—集中合并”的结构；它不是 LLM Agent 协议。
- [AI Safety via Debate](https://arxiv.org/abs/1805.00899)：关于通过对抗式辩论辅助判断的研究入口；论文中的实验设定不能直接证明多个同质 worker 就拥有独立证据。
- [AgentBench](https://arxiv.org/abs/2308.03688)：多环境 Agent 评估论文入口，可用于任务环境和工具交互的评估背景；其结果依赖具体环境、权限、工具和 harness。
- [AutoGen 源码仓库](https://github.com/microsoft/autogen)：公开实现入口，适合核对框架演进和示例；仓库内容不等于目标系统的安全审计结论。

本章区分四种证据：经典并行模型的定义、官方框架的接口语义、论文中的受控实验和本地系统的实测结果。官方文档能说明 API 怎么工作，不能证明模型的内部推理；论文能支持特定实验结论，不能自动支持生产场景；教学 demo 只展示机制，不能替代容量、安全和故障注入测试。

## 14.24 小结

Agent Swarm 的核心不是 worker 数量，而是协作协议。任务图决定哪些工作可以并行；角色契约决定谁负责什么；artifact schema 让事实、证据、假设和测试可合并；只读共享和隔离 workspace 减少污染；独立 verifier 和单一 external executor 保护外部状态；租约、取消、幂等和 checkpoint 让失败可恢复；固定预算和结构消融才有资格说明并行是否创造了真实收益。

可以先记住：多人一起写报告，不等于多人一起证明事实。工程审查还要继续追问：这些 worker 是否真的独立，是否看到了相同错误来源，候选是否经过外部验证，谁能够提交副作用，取消时外部状态是什么，以及并行带来的质量提升是否值得通信、延迟、GPU 和治理成本。

当所有重要结论都能回到带版本的 artifact，当所有外部写入都经过受控执行和状态回读，swarm 才从“同时生成很多答案”变成了可审计的并行系统。
