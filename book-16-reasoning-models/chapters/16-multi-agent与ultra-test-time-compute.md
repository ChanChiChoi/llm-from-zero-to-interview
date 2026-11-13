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

## 16.2 从 self-consistency 到并行协调：额外计算改变了什么

额外推理计算有几种不同的组织方式，不能都叫作“多 Agent”，再用参与者数量解释效果。

Self-consistency 对同一问题采样多条轨迹，通常在最终答案层投票；这一经典定义见 [Wang 等人的原始研究](https://arxiv.org/abs/2203.11171)。它增加候选覆盖，但候选可能共享同一模型偏差、提示和检索来源；十条答案一致不等于十个独立证据。

Multi-agent debate 让多个模型实例先分别作答，再在数轮交互中查看并质疑彼此的回答。[Du 等人的多 Agent debate 论文](https://arxiv.org/abs/2305.14325)报告了数学、策略和事实性任务上的改善，但实验结论绑定论文中的模型、提示和任务设置。若实例使用相同模型、共享相同证据，它们可能通过相同错误达成共识，辩论次数也不能代替 verifier。

[Language Agent Tree Search（LATS）](https://arxiv.org/abs/2310.04406)把搜索推进到状态转移层：用树搜索组织候选动作，结合语言模型的价值评估、自我反思和环境反馈。它与“同时生成多份最终答案”不同，关键在于搜索过程能从环境观察得到新状态。论文的结果属于其编程、网页和交互任务设置，不代表所有工具任务都有同样收益。

近期并行推理工作把问题进一步推到协调协议本身。[PaCoRe](https://aclanthology.org/2026.acl-long.1253/)把多条轨迹分轮并行展开，将发现压缩成受上下文限制的消息，再传给下一轮聚合；论文摘要报告系统把 effective test-time compute 扩展到约 200 万 token，并报告 8B 模型在 HMMT 2025 得到 94.5%。该数字是作者摘要中的特定模型和基准结果，摘要没有提供足够的预算与 harness 对齐细节支持跨系统排名。[ParaThinker](https://aclanthology.org/2026.findings-emnlp.1469/)则在单一模型架构内并行产生多条推理路径并聚合；其摘要报告 1.5B 模型在相同 decoding-token 预算下 AIME-24 准确率提升 6.5%–20.7%，并报告启用 KV-cache reuse 时相对 sequential decoding 的推理延迟降低 38.7%–66.8%。这些结果绑定论文的模型、任务、预算和推理实现，不能据此推断所有并行系统都更快。

| 方法 | 额外计算放在哪里 | 新增的协调责任 |
|---|---|---|
| Self-consistency | 多条独立采样轨迹 | 去重、投票或 verifier 选择 |
| Debate | 多轮候选之间交换论据 | 控制共同偏差，验证共识所依赖的来源 |
| LATS | 状态与动作上的搜索节点 | 验证转移、评分函数和环境反馈 |
| PaCoRe | 多轮并行轨迹与消息聚合 | 压缩发现、跨轮传递和最终合成 |
| ParaThinker | 单一架构内的并行路径 | 路径多样性、聚合和推理实现成本 |

因此，选择方案时要问：新增预算买到的是更多样本、互相批评、真实环境观察，还是更宽的并行路径？协调机制必须针对新买到的计算，否则只会重复候选并增加合并成本。provider 产品里的 worker 数映射是另一个层次；例如[第十四章](14-thinking-levels.md) 的 Grok 4.20 文档示例是具体服务合同，不能当作通用 multi-agent 规律。

## 16.3 任务图：哪些节点可以并行

把协作过程表示为有向任务图：节点是需要完成的工作，边表示某个节点必须等待另一个节点提供指定产物、版本或状态。边表达数据依赖或副作用依赖，不只是“两个角色有关系”。

以跨仓库修复为例，节点包括读取调用链、阅读历史 issue、构造复现、生成 patch、运行测试和合并审查。读取调用链、历史 issue 和只读复现可并行；生成 patch 至少要等待足够的定位证据；测试必须针对一个确定的 patch revision；合并还要等待测试、安全检查和授权。可用以下轨迹表示关键依赖：

~~~text
调用链 ─┐
历史记录 ├─> 候选 patch -> 隔离测试 ─┐
并发复现 ┘                            ├─> 合并审查
安全只读检查 ─────────────────────────┘
~~~

图中的“并行”只表示节点可以同时运行，不表示它们可以同时修改同一个外部对象。调用链分析可输出只读报告，patch worker 可写自己的分支；未经协调的两个 worker 不应同时写生产分支。

如果图中有大量强顺序边，增加 worker 数不会带来同比例加速。先画依赖图，再找关键路径，才能判断并行是否值得。

## 16.4 关键路径与并行效率

设一个阶段有 n 个可并行 worker，第 j 个 worker 的有效工作时间为 T_j，协调、合并和验证时间分别为 T_coord、T_merge 和 T_verify。在协调、合并和验证位于并行工作之后这一简化假设下，时间使用秒等同一单位；n 为正整数，所有时间有限且非负；0<η≤1 表示排队、争用和调度折损。阶段延迟可近似为：

~~~math
T_{\mathrm{parallel}}
\approx \frac{\max_{1\le j\le n}T_j}{\eta}
+T_{\mathrm{coord}}+T_{\mathrm{merge}}+T_{\mathrm{verify}}
~~~

例如三个独立 worker 分别需要 4、6、5 秒，η=0.8，协调 1 秒、合并 1 秒、验证 2 秒。关键 worker 的有效延迟是 6/0.8=7.5 秒，再加 1+1+2 秒，得到 T_parallel≈11.5 秒。若同样三项工作串行执行，则工作本身要 4+6+5=15 秒，再加 4 秒协调/合并/验证，总计 19 秒。这个合成算例显示并行减少了关键路径延迟，但并未减少 15 秒的 worker 工作量，也未说明实际系统的 token 或货币成本。

即使所有可并行工作都能理想缩短，总加速仍受串行部分限制。令 S(n) 为 n 个 worker 下的理想加速比，W_s 为串行工作量、W_p 为可并行工作量；两种工作量使用同一单位，n 为正整数且 W_s+W_p>0：

~~~math
S(n)\le\frac{W_s+W_p}{W_s+W_p/n}
~~~

例如串行工作为 20 秒、可并行工作为 80 秒、n=4，则理想上界是 100/(20+80/4)=100/40=2.5 倍；即使 worker 无限增加，上界也只有 100/20=5 倍。真实系统还要付出消息、调度、重复探索和 verifier 成本，因此不能把这个上界当成实际加速承诺。

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

## 16.9 产物生命周期与不确定结果

候选产物的最小生命周期可以写成：

~~~text
created -> evidence_attached -> tested -> reviewed
                                      ├-> accepted
                                      └-> rejected
~~~

状态迁移要绑定 task ID、输入 revision、产物 owner 和证据摘要。若代码或数据基线发生变化，旧测试结果就不能继续支持新 revision，应回到待验证状态。

worker 的运行状态可以包括 queued、running、succeeded、failed、cancelled 和 unknown。unknown 表示调用方暂时拿不到权威结果；它不能被当成失败并直接重发。涉及工具请求的查询、幂等和副作用恢复属于第十九章的工具协议范围；跨进程 checkpoint 属于第二十册 Runtime。本章只把这些状态作为并行候选能否进入聚合和 verifier 的输入。

## 16.10 Workspace 与权限隔离

并行实验要能比较，worker 的输入基线和可写区域必须可区分。一个简化布局是：共享只读证据索引；每个可写 worker 使用独立分支或目录；merge owner 产出新的合并 revision；具备授权的 executor 才能执行外部副作用。这样可以把结果绑定到具体输入和 patch，避免 worker 相互覆盖尚未完成的实验。

计算预算和权限是独立的控制面。增加 token、worker 或 verifier 次数不改变文件、网络、数据库或生产写权限；多个 worker 达成共识也不构成授权。网页、issue、代码注释和上传文件的内容仍是待核验数据，不能自动升级成系统指令。Agent 的完整工具和权限架构见[第十七册第十四章](../../book-17-agent-tool-use/chapters/14-agent-swarm并行协作.md)与[第二十册第八章](../../book-20-agent-harness-runtime/chapters/08-权限模型与安全沙箱.md)。

## 16.11 Ultra Test-Time Compute 是预算控制问题

这里的 ultra 不是一个精确的模型类别，而是把更多候选、工具观察、验证或协调预算投入一次任务。它可以通过更多采样、搜索、verifier、工具循环或多个 Agent 实现。共同点是：任务结束前，系统允许花费更多资源来降低失败概率。跨策略的质量—成本比较见[第七章](07-test-time-compute-scaling.md)，请求预算的计量口径见[第十三章](13-reasoning-effort与推理预算.md)；本章专注多条推理路径的并行组织、协调和验证。

若所有预算已经换算到同一个成本单位，可以写成：

~~~math
\sum_{i=1}^{n}b_i+b_v+b_m\le B
~~~

其中 b_i 是第 i 个 worker 的预算，b_v 是验证预算，b_m 是合并与恢复预算，B 是总预算，n 为正整数。所有值必须是同一账本中的有限非负数。例如，教学账本给三个 worker 分配 25、20、15 credits，验证 20 credits，合并与恢复 5 credits，总用量是 25+20+15+20+5=85 credits；若总额 B=100 credits，则尚余 15 credits。credits 是这里构造的统一单位，不代表某个供应商的价格。

不能把 token、毫秒和副作用次数未经换算直接相加。工程上更准确的表示是按单位逐项限制的预算向量：

~~~math
\mathbf{B}=(B_{\mathrm{model}},B_{\mathrm{tool}},B_{\mathrm{verify}},B_{\mathrm{time}},B_{\mathrm{side}})
~~~

例如，一个合成预算向量可以是 40,000 tokens、8 次工具调用、2 次验证、90 秒 deadline、0 次副作用。每个分量都保留自己的单位；0 次副作用表示禁止外部写入，而不是“便宜的写权限”。

生成预算和验证预算之间存在取舍。若验证预算不足，系统会收集很多候选却无法判断；若合并预算不足，冲突会被文本拼接掩盖；若工具预算耗尽，新增 reasoning token 也无法获得新的观察。困难任务不一定需要更多 worker，可能更需要独立 verifier 或更好的测试环境。

## 16.12 成本、单位成功成本与边际收益

多 Agent 的成本账本至少应包括模型、工具、通信、合并、验证和人工；求和下标 i 遍历批次中发生的相应调用或成本项。以下各项要使用同一个货币或归一化成本单位，不能把未测量的成本默认为零：

~~~math
C_{\mathrm{total}}
=\sum_i C_{\mathrm{model},i}
+\sum_i C_{\mathrm{tool},i}
+C_{\mathrm{message}}
+C_{\mathrm{merge}}
+C_{\mathrm{verify}}
+C_{\mathrm{human}}
~~~

例如一个构造的批次中，模型 64 credits、工具 18、通信 2、合并 8、验证 12、人工 16，总成本为 64+18+2+8+12+16=120 credits。这里每项都是教学输入，真实成本应来自账单、容量模型或人工工时记录。

设成功数 N_success 是非负整数，单位成功成本为：

~~~math
C_{\mathrm{success}}=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
~~~

若上述 120 credits 的批次有 3 个成功任务，则单位成功成本为 120/3=40 credits/成功任务。当 N_success=0 时，分母为零，指标未定义，应该报告 None 或 undefined；不能用 max(1,N_success) 把失败批次伪装成可比较成本。

判断增加第 k 个 worker 是否值得，可估算边际效用：

~~~math
\Delta U_k
=\Delta P_{\mathrm{success},k}V
-\Delta C_k
-\lambda\Delta R_k
~~~

ΔP_success,k 是在同一任务集和成功定义下成功概率的增量，V 是一次成功的价值，ΔC_k 是同单位增量成本，ΔR_k 是风险增量，λ≥0 把风险换算成与 V 同单位的代价。例如 ΔP=0.08、V=100 credits、ΔC=5 credits、ΔR=0.1、λ=10 credits/风险单位，则 ΔU=0.08×100−5−10×0.1=2 credits。这个正值只在教学设定的价值、成本和风险刻度成立时有意义；样本不足时不应把估计值写成确定收益。

若风险属于不可接受的硬约束，就不能让很大的 V 抵消它。可预先设定质量增量、成本和风险门槛：

~~~math
G_{\mathrm{expand}}
=\mathbf{1}[\Delta P_{\mathrm{success}}>\delta]
 \mathbf{1}[C_{\mathrm{extra}}\le C_{\max}]
 \mathbf{1}[R_{\mathrm{extra}}\le R_{\max}]
~~~

其中指示函数条件成立取 1，否则取 0；δ 是最低成功率增量，C_max 和 R_max 是允许的增量成本与风险上限。例如 δ=0.02、C_max=6 credits、R_max=0.15，观测到 ΔP=0.08、C_extra=5 credits、R_extra=0.10，则三个条件分别为 1，G_expand=1；这是合成门槛例；若成本变为 7 credits，第二个条件为 0，系统就不应扩张。未测量的概率或风险不能用 0 填充来得到“允许”结果。

## 16.13 什么时候停止扩张

高预算系统必须在启动前设置最大 worker 数、最大轮次、模型预算、工具时间、验证次数和外部副作用上限。运行中还要观察是否产生新证据。以下信号通常支持停止或降级：

1. 连续若干轮的 artifact 没有新增来源、状态或可区分的候选；
2. 候选已经满足独立 verifier 的验收条件；
3. 新增 worker 只重复同一个共因来源；
4. 剩余时间不足以完成验证和安全检查；
5. 预算已经耗尽，或者风险超过硬性约束；
6. 工具状态未知，继续重试可能产生重复副作用。

“模型还想继续思考”不是充分的继续条件。停止时要保存未完成 worker 的状态、取消原因、已生成 artifact、预算余额和外部动作状态，使任务可以安全恢复或交给人工；持久化 checkpoint 的设计见[第二十册第十七章](../../book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md)。

## 16.14 一个跨仓库修复的并行划分

假设服务在高并发刷新配置时偶发 500。协调器固定仓库 revision、失败日志和只读范围，再建立三条可以独立取证的路径：调用链 worker 找到刷新与重试逻辑；复现 worker 在隔离分支构造最小并发测试；历史 worker 查找相同组件的已合并修复。每条路径都交付带来源、基线 revision 和验证状态的 artifact。

三条证据汇合后，patch worker 根据已验证的根因提交候选 patch。测试 worker 对具体 patch revision 运行原失败复现和回归测试；安全检查针对同一 diff 独立执行。只有测试通过且合并审查引用这些结果时，候选才进入 accepted 状态。若复现失败，或测试与 patch 的 revision 不一致，结论应保持为待验证。

这个任务之所以适合并行，是因为只读调查和复现可以给出互补观察；patch 修改、候选合并和任何部署仍需受依赖边约束。若每个 worker 都直接写共享分支，系统会增加冲突和不确定性，无法把并行加速转化为可验证的成功率。完整代码 Agent 的工作流和 Runtime 交接见[第十七册](../../book-17-agent-tool-use/简介.md)与[第二十册第十七章](../../book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md)。

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

**代码状态：已验证（2026-11-13；CPython 3.12.9；仅 Python 标准库）。**

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
        type(value) in {int, float} and value >= 0,
        f"{name} must be a finite non-negative number",
    )
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    require(finite, f"{name} must be a finite non-negative number")


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
        status = artifact["status"]
        require(isinstance(status, str) and status in ARTIFACT_STATES, "unknown artifact status")
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

PaCoRe 与 ParaThinker 的方法和作者结果依据 2026-11-13 查阅的 ACL Anthology 论文摘要表述；未从摘要推导完整的模型、prompt、硬件或生产服务对照条件。

参考资料：

- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*：<https://arxiv.org/abs/2203.11171>
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*：<https://arxiv.org/abs/2305.10601>
- Zhou et al., *Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models*：<https://arxiv.org/abs/2310.04406>
- Du et al., *Improving Factuality and Reasoning in Language Models through Multiagent Debate*：<https://arxiv.org/abs/2305.14325>
- Hu et al., *PaCoRe: Learning to Scale Test-Time Compute with Parallel Coordinated Reasoning*：<https://aclanthology.org/2026.acl-long.1253/>
- Wen et al., *ParaThinker: Native Parallel Reasoning for Scaling LLM Test-Time Compute*：<https://aclanthology.org/2026.findings-emnlp.1469/>
- Amdahl, *Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities*：<https://doi.org/10.1145/1465482.1465560>
