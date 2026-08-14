# 第 16 章 Multi-Agent 与 Ultra Test-Time Compute：并行探索什么时候值得

## 16.1 多个 Agent 不会自动组成一个更聪明的 Agent

面对复杂研究、跨仓库代码修复或长周期审计，系统常会同时启动多个 Agent：一个负责检索，一个负责提出方案，一个负责运行测试，最后由主控 Agent 合并结果。这类架构常被称为 multi-agent、swarm 或 ultra test-time compute。

它的合理解释不是“人数多所以智商高”，而是把额外推理预算分散到多条候选轨迹，再利用独立证据和验证器减少单一路径的偶然错误。如果任务本身不能拆分、所有 Agent 看到同样的错误上下文、没有共享的产物契约，那么并行只会产生更多重复答案，并把合并工作推给最后一个模型。

## 16.2 从 self-consistency 到带环境的轨迹

self-consistency 通常让模型对同一个问题生成多条文本推理链，最后投票。multi-agent 更进一步：每个 worker 拥有自己的上下文、工具、权限和中间 artifact，输出的是带状态的轨迹，而不是一段孤立文本。

可以把任务图抽象为：

```math
G=(V,E),\qquad
V=\{\mathrm{plan},\mathrm{search},\mathrm{code},\mathrm{verify},\mathrm{merge}\}
```

只有没有强依赖的节点才能真正并行。例如定位调用链和阅读历史 issue 可以并行；但修复 patch 必须等待定位结果，部署操作必须等待验证和授权。把一个串行链路强行复制成多个 worker，不会减少关键路径。

## 16.3 何时并行有信息价值

并行最有价值的情况通常满足三个条件。第一，子任务拥有相对独立的输入或假设，某个 worker 的错误不会立即污染其他 worker。第二，每个 worker 有明确产物，例如证据表、候选 patch 或测试报告，而不是泛泛地“分析一下”。第三，结果可以被规则、测试或独立 verifier 检查。

例如在一个跨仓库 bug 中，可以让 worker A 构建调用链，worker B 创建最小复现，worker C 查找相似历史修复。三者的产物不同，且都可以只读。主控再把候选 patch 交给独立测试 worker。反过来，让四个 Agent 都“自由修复 bug”，最后由一个模型比较四段长文本，通常难以判断谁的证据更可靠。

## 16.4 成本和并行效率

多 Agent 的总成本包括模型、工具、通信和验证：

```math
C_{\mathrm{total}}
=\sum_i C_{\mathrm{model},i}
+\sum_i C_{\mathrm{tool},i}
+C_{\mathrm{coord}}
+C_{\mathrm{verify}}
```

并行的主要收益是降低 wall-clock，而不一定降低 token 或 GPU 成本。若有 `n` 个 worker，平均单 worker 的工作时间为 `T_work/n`，并行效率为 `\eta`，合并时间为 `T_merge`，可以粗略写成：

```math
T_{\mathrm{wall}}\approx\frac{T_{\mathrm{work}}}{n\eta}+T_{\mathrm{merge}}
```

当工具共享、队列拥堵、依赖很多或合并需要读取全部产物时，`\eta` 会很低，`T_merge` 甚至成为主导。增加 worker 数因此可能让总延迟和成本同时上升。

## 16.5 Worker 的最小契约

一个可靠 worker 不应该只返回自然语言。它应声明输入范围、允许动作、产物 schema、证据引用、版本和完成状态。例如代码定位 worker 可以返回：

```json
{
  "role": "call-graph",
  "hypotheses": [
    {"claim": "race on cache refresh", "evidence": ["src/a.py:81"]}
  ],
  "files_read": ["src/a.py", "src/cache.py"],
  "artifacts": ["trace.json"],
  "confidence": "medium",
  "verified": false
}
```

`confidence` 不是证据，`verified` 也不是答案正确的保证。主控应根据来源和测试决定是否采纳。每个 artifact 最好有 hash，防止 worker 在共享 workspace 中写入后又被其他 worker 修改。

## 16.6 一个跨仓库修复流程

主控首先读取 issue，只读地将任务拆成三个子任务。A 搜索相关调用链，B 运行最小复现，C 查找最近版本的类似变更。三者使用隔离 workspace，不能修改测试或生产配置。

如果 A 找到一个可疑的共享变量，B 复现出只有并发运行才出现的失败，C 发现历史修复曾经增加锁，那么主控可以生成候选 patch。候选 patch 交给测试 worker，测试 worker 运行公开和隐藏测试；安全 worker 读取 diff，检查是否修改敏感目录、增加网络访问或扩大权限。只有测试和安全检查都通过，才把 patch 交给人工或发布系统。

这个流程中，worker 数量不是关键。真正关键的是隔离 workspace、明确 artifact、独立验证和提交边界。若 B 直接修改共享工作区，A 的调用链可能建立在 B 的未审阅改动之上；若所有 worker 只跑不完整的公开测试，主控的多数投票也可能一致地选择错误 patch。

## 16.7 Ultra 模式的停止条件

高预算模式必须显式设置最大 Agent 数、最大轮次、总 token、工具时间、重复轨迹阈值和冲突无法解决时的人工路径。继续扩张的收益可以用一个简化验收条件表达：

```math
G_{\mathrm{expand}}
=\mathbf{1}[\Delta P_{\mathrm{success}}>\delta]
\mathbf{1}[C_{\mathrm{extra}}<C_{\max}]
\mathbf{1}[R_{\mathrm{risk}}<R_{\max}]
```

如果新增 worker 没有带来新的证据，或者所有候选都在重复同一个错误假设，系统应停止并说明不确定性。高风险工具即使在预算内，也要经过人工确认或沙箱。

## 16.8 如何证明 multi-agent 值得

实验至少比较三条路径：direct、single-agent 和 multi-agent。任务、工具、最大总预算和成功定义要对齐，不能允许 multi-agent 使用十倍成本后只报告更高的最高分。

除了最终成功率，还要记录 wall-clock、总 token、工具调用、通信量、合并时间、人工接管、重复副作用和安全事件。再做消融：worker 数、角色是否固定、是否共享只读 memory、是否有 verifier、是否并行工具。这样才能知道收益来自并行、角色分工、更多尝试还是验证器。

可以用单位成功成本比较：

```math
C_{\mathrm{success}}
=\frac{\sum_i C_{\mathrm{task},i}}
{\max(1,N_{\mathrm{success}})}
```

如果 multi-agent 成功率只增加 3%，成本增加 5 倍，只有在失败代价极高时才可能合理。

## 16.9 常见失败模式

多数投票可能放大共同错误。worker 如果共享同一个错误 system prompt、污染数据或错误检索结果，投票只会让错误更有信心。

权限过宽会放大泄露和副作用。并行 Agent 应使用最小权限、只读证据和隔离工作区。

没有 artifact 契约时，主控无法区分事实、猜测和未验证结果。所有输出拼成一个长上下文还会引入“合并阶段上下文爆炸”。

另一个常见问题是失败任务不断增加 worker。应对重复轨迹、低信息观察和相同错误设置停止阈值，而不是无限扩张。

## 16.10 面试回答与练习

回答“multi-agent 为什么可能提升效果”时，应说它把 test-time compute 从多条文本采样扩展成多条带工具和状态的任务轨迹；收益取决于任务可拆分性、worker 隔离、artifact 契约、通信和验证，并行降低的是关键路径时间，不保证降低总成本。评估时要对齐总预算，报告成功、成本、延迟、风险和人工接管。

练习一：为研究、代码和数据分析各画一张任务 DAG，标出可并行节点和必须串行的节点。

练习二：设计一个 worker artifact schema，使主控能区分 claim、evidence、hypothesis、verified result 和 side effect。

练习三：给定四个 worker 和一个 merge worker，估算并行收益可能被哪些通信、队列和验证成本抵消。

### 16.10.1 并行计算的收益上限

如果任务有总工作量 `W`，其中不可并行的关键路径为 `W_s`，可并行部分为 `W_p`，使用 `n` 个 worker 后，理想加速比受 Amdahl 定律约束：

```math
S(n)\le\frac{W_s+W_p}{W_s+W_p/n}
```

实际还要加通信、调度、merge、重复探索和验证成本：

```math
C_{\mathrm{total}}=C_{\mathrm{worker}}+C_{\mathrm{message}}
 +C_{\mathrm{merge}}+C_{\mathrm{verify}}
```

如果四个 worker 只是重复读取同一份文档，再由主控重新比较四个长答案，总 token 和合并成本可能超过单 Agent 的一次深思。并行适合任务天然分解、候选彼此独立或工具等待占主导的场景。

### 16.10.2 worker contract 比角色名称重要

每个 worker 应有输入、允许工具、输出 artifact、证据要求、超时和失败语义。一个研究 worker 可以返回 `claims`、`evidence_ids`、`uncertainties` 和 `next_action`；不能只返回一段自然语言，让主控无法判断哪些内容已验证。

写入型 worker 要有独立 workspace 或 patch，而不是共享可变文件。主控合并前检查基线 revision、文件范围、测试结果和权限。两个 worker 对同一文件的修改冲突，应进入 merge 或人工，而不是用最后写入覆盖。

### 16.10.3 成本与容量

设 `n` 个 worker 每个消耗 `c_i`，通信和合并消耗 `c_m`，则总成本近似为：

```math
C_{\mathrm{swarm}}=\sum_{i=1}^{n}c_i+c_m
```

对 test-time search 而言，wall-clock 延迟通常由关键路径上最慢的 worker、通信、合并和验证共同决定，而不是所有 worker 时长的简单相加。共享 GPU 时还要把队列和 KV 争用计入关键路径；每个 worker 都加载完整模型则会放大显存成本。比较 single-agent 与 multi-agent 时，应固定总预算并同时测这些项。

### 16.10.4 失败和取消

test-time search 中，一个 worker 超时可能只是少了一条候选，也可能意味着验证证据尚未返回；主控必须区分未启动、执行中、完成但结果丢失和状态未知，再决定缩小搜索、等待还是回退。取消要传播到工具和子任务，并把候选、验证结果和预算状态一起封存；共享 workspace 仍需 revision 与锁，不能靠 worker 自觉避免冲突。

### 16.10.5 ultra 模式的真正问题

所谓 ultra 或深度模式，本质是允许更多候选、更多工具、更多验证或更多 Agent 协作。它可能提高困难任务成功率，也可能扩大错误搜索、提示注入和外部副作用。评估要比较固定总预算下的单 Agent、高预算单 Agent、多 Agent 和 ultra 路径，而不是只公布 ultra 的最高分。

对高风险任务，worker 之间的多数票不能替代独立证据。多个 worker 共享同一个错误文档、同一个有污染的 prompt 或同一个 verifier 漏洞时，票数越多只会让错误更一致。

## 16.11 先选择拓扑，再决定 worker 数量

multi-agent 不是一个固定架构。常见拓扑至少有四种：并行独立探索、分工流水线、主控—worker 树和相互批评/投票。独立探索适合候选彼此可比较的数学或检索任务；流水线适合“检索—分析—验证”有明确依赖的任务；主控树适合动态拆分；批评/投票适合存在多个可行答案但有可靠合并规则的任务。

选择拓扑前先问三个问题：子任务是否可以独立完成，结果是否能用结构化 artifact 表达，是否有可靠的合并或验证器。如果三个答案都是否，增加 worker 只会增加上下文和协调成本。一个写代码的任务若所有 worker 都修改同一个工作区，表面上是并行，实际上是没有锁的共享内存程序。

## 16.12 Artifact 合约与合并边界

worker 的输出不能只是自然语言段落。最小 artifact 应包含：结论或 patch、证据引用、假设、验证状态、适用范围、失败原因和下一步建议。主控据此决定哪些内容可以合并，哪些只能作为待验证候选。

```json
{
  "worker_id": "w2",
  "claim": "retry storm starts after config revision r17",
  "evidence": [{"source": "logs", "revision": "r17", "hash": "..."}],
  "hypotheses": [{"text": "timeout is too short", "status": "unverified"}],
  "artifacts": [{"kind": "patch", "base_revision": "abc"}],
  "verification": {"tests": [], "status": "pending"},
  "side_effects": []
}
```

写入型 worker 应在隔离分支、临时目录或容器中工作。合并前检查 base revision、文件范围、测试结果和权限；两个 patch 冲突时进入显式 merge 或人工审核，不能用最后写入覆盖。研究 worker 的证据也需要去重和来源排序，否则多个 worker 引用同一篇错误网页会被误判为独立支持。

## 16.13 Ultra 模式的预算分配

把 ultra 理解为“无上限地启动更多 Agent”是错误的。它更接近一个带预算的并行搜索器：系统先分配总计算预算，再决定多少用于候选生成、工具观察、验证、合并和恢复。

设总预算为 `B`，第 `i` 个 worker 的预算为 `b_i`，验证和合并预算分别为 `b_v`、`b_m`，则必须满足：

```math
\sum_{i=1}^{n}b_i+b_v+b_m\le B
```

如果 worker 数量增加而 `b_v` 和 `b_m` 不变，主控可能收集到更多未验证答案，却没有能力判定它们。可以按照候选分歧、预期信息增益和任务风险动态分配预算；当新增 worker 的边际成功率小于边际成本时停止扩张。

并行只减少可并行部分的 wall-clock。若串行关键路径占比为 `s`，理想加速比受：

```math
S(n)\le\frac{1}{s+(1-s)/n}
```

实际还要加调度、GPU 争用、网络、合并和尾部 worker 等成本。报告 ultra 模式时，必须同时给出总 token、总 GPU 时间、关键路径延迟和单位成功成本。

## 16.14 相关错误与对抗性验证

多数投票不是独立正确性的证明。所有 worker 可能共享同一个错误检索结果、system prompt、模型偏见或污染 benchmark。为了检测相关错误，可以改变检索切片、提示模板、模型版本或验证器，并专门加入反例和不可判定样本。

对高风险 Agent，主控应采用最小权限和只读证据；写入动作由独立 policy gate 和用户确认控制。worker 之间不应直接传递未脱敏凭证，也不应把一个 worker 的网页文本当作另一个 worker 的系统指令。评估要报告共因失败率、错误传播率、越权尝试和人工接管，而不只报告投票后的最终准确率。

## 16.15 先区分独立错误和共因错误

多 Agent 的一个隐含假设是不同 worker 的错误足够独立。如果所有 worker 使用同一份错误检索结果、相同的 system prompt、同一个被污染的 benchmark 或相同的错误工具状态，增加 worker 只会复制错误。

可以把投票结果拆成两种收益：独立探索带来的信息增益，以及重复路径带来的冗余。后者不能用多数票消除。工程上应记录 worker 的 evidence source、prompt revision、model revision 和工具状态，计算共因失败率；高风险结论还要引入不同来源的 verifier，而不是只换一个角色名称。

## 16.16 关键路径和 verifier 预算

Ultra 模式经常把预算都花在候选生成，最后没有足够资源验证。设总预算为 `B`，候选、工具观察、验证和合并分别消耗 `B_c`、`B_t`、`B_v`、`B_m`：

~~~math
B_c+B_t+B_v+B_m\le B.
~~~

如果 `B_v` 太小，主控会收集很多无法判断的答案；如果 `B_m` 太小，冲突会被文本拼接掩盖。对代码和数据任务，验证预算可能比第二个候选 worker 更有价值。预算分配应根据当前分歧、证据缺口和任务风险调整，而不是按固定比例复制。

## 16.17 artifact 生命周期和版本条件

每个 worker artifact 应经历 `created`、`evidence_attached`、`tested`、`reviewed`、`accepted` 或 `rejected`。状态转换绑定 task id、base revision、owner、证据 hash 和过期时间。主控重试时，旧 artifact 不能直接覆盖新 revision。

写入型任务尤其需要隔离分支或 workspace。合并前检查 patch 是否基于当前 revision、是否触碰超出声明范围的文件、测试是否在干净环境运行、权限是否仍有效。一个 worker 的自然语言结论可以作为候选，但不能直接变成另一个 worker 的执行命令。

## 16.18 停止条件和边际收益

如果系统只规定“尽量多思考”，任务会在没有新信息时继续消耗预算。可以定义连续若干轮没有新增证据、候选答案已经收敛、验证通过或剩余预算不足时停止。用边际效用表达：

~~~math
\Delta U_k
=\Delta P_{\mathrm{success},k}V
 -\Delta C_k-\lambda\Delta R_k.
~~~

当新增第 `k` 个 worker 的成功率提升不足以覆盖成本和风险，就应停止或转人工。对高风险动作，风险项可以是硬性条件而不是可被价值抵消的软惩罚。

## 16.19 多 Agent 不是所有任务的答案

有些任务天然适合并行：独立资料检索、日志和代码的分开分析、候选方案生成、不同测试策略的探索。有些任务不适合：共享状态频繁写入、每一步都依赖前一步、外部副作用不可逆、任务很短、或者验证本身比生成更贵。

在不适合并行的任务上，单 Agent 加显式状态机可能更可靠。即便适合并行，也可以只并行只读阶段，把合并、审批和执行保留为串行阶段。架构选择应由依赖图和验收指标决定，不由产品名称决定。

## 16.20 评估要固定总资源

比较 direct、single-agent、serial multi-agent、parallel multi-agent 和 parallel-plus-verifier 时，必须固定总 token、最大 GPU 时间、工具权限、数据版本和成功判据。报告最终成功率之外，还要报告：

| 指标 | 解释 |
| --- | --- |
| 单位成功成本 | 总模型、工具和人工成本除以成功任务数 |
| 关键路径延迟 | 最慢依赖链，而非最快 worker |
| 共因失败率 | 多个 worker 是否共享同一错误来源 |
| 合并拒绝率 | 有多少 artifact 不能安全自动合并 |
| 副作用错误率 | 并行是否放大重复或越权动作 |
| 取消恢复率 | 取消后状态是否可恢复 |

如果增加 worker 同时增加总预算，分数提高不能证明协作本身有效；如果只看平均延迟，又可能掩盖尾部 worker 和人工审核队列。

## 16.21 一个可复现的 ultra 实验

可以用跨文件代码修复构造实验：先让单 Agent 在固定仓库上执行，再拆成日志分析、代码定位、测试设计三个 worker，最后由 verifier 检查 patch 和测试。保存每个 worker 的 prompt、revision、artifact、token、工具时间、冲突和取消状态。

做四组消融：去掉并行、去掉 verifier、共享 workspace 改为隔离 workspace、固定总预算改为不受限预算。结果应同时回答“是否更快”“是否更正确”“是否更贵”“是否更容易回滚”。任何外部提交都只在全部检查通过后执行。

## 16.22 并行 worker 的任务分解条件

把一个任务交给多个 Agent 之前，先判断子任务是否具有相对独立的输入、可合并的输出和清楚的验证方式。研究资料可以按来源分片，代码仓库可以按只读目录分片，候选解可以独立生成；但共享数据库的连续写操作、依赖前一步隐藏状态的调试和不可逆的外部动作通常不适合并行。

可以把一个任务图表示为 `G=(V,E)`。若边 `E` 中有大量强顺序依赖，并行度 `p` 增加时，协调成本可能超过计算收益。粗略的完成时间是：

```math
T_{\mathrm{total}}
\approx T_{\mathrm{critical\ path}}
+T_{\mathrm{coordination}}
+T_{\mathrm{verification}}.
```

因此“启动更多 worker”不是优化目标，缩短关键路径并保持验证成本可控才是。

## 16.23 共享上下文和共享 workspace 的区别

多个 worker 可以共享任务说明，却不应默认共享所有历史和写权限。共享上下文提高协作效率，也扩大 prompt injection 和隐私泄露范围；共享 workspace 方便合并，却容易出现互相覆盖、锁竞争和无法归因。更稳妥的做法是只读证据区、隔离写区和单独的合并区。

合并器应接收结构化 artifact，例如结论、证据引用、patch、测试结果和不确定项，而不是把所有 worker 的长文本拼接给一个“总管 Agent”。总管只负责比较和验证，不能因为某个 worker 声称完成就跳过独立检查。

## 16.24 ultra 模式的停止规则

高预算并行系统必须有停止条件：新增 worker 连续若干轮没有新增证据，候选结果已经达到质量验收条件，剩余时间不足以安全验证，或成本超过任务上限。停止时要保存未完成的 worker 状态和取消原因，避免下次恢复时把已取消的动作重新提交。

评测中应做固定总预算和无限预算两种对照。若无限预算的分数提高只是因为用了更多 token、重试和人工时间，不能把它写成架构本身的收益。

## 16.25 小结与资料边界

Multi-agent 是一种组织额外推理计算的方法，不是单独的智能增益定律。可拆分任务、隔离环境、结构化产物、独立验证、边际收益停止和预算约束比“启动多少 Agent”更重要。Ultra 模式只有在新增计算带来可验证的信息，并且成本、容量和风险满足约束时才值得。

本章的任务图、test-time compute 和 Agent 评估框架可由公开研究和工程资料支持；具体产品的 swarm 编排、内部 worker 数和路由策略若未公开，应视为实现细节，不从营销名称推断。
