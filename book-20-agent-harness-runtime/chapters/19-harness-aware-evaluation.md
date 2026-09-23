# 第 19 章 Harness-Aware Evaluation：评估的对象不只是模型

## 19.1 为什么同一个模型会得到不同成绩

模型 A 在一个 coding Agent 中通过了 40% 的任务，在另一个 harness 中通过了 65%。这是否说明模型变强了？不一定。第二个 harness 可能提供了更好的检索器、自动重试、上下文折叠、代码执行器、测试修复器或更宽的工具权限。

Agent 的真实行为由模型、harness、环境、预算和数据共同决定。若把结果只归因给模型，就会把工具和编排的贡献写错，也无法复现线上表现。

## 19.2 用函数描述评测结果

可以把测量结果写成：

```math
R=F(M,H,E,B,D)
```

`M` 是模型和 revision，`H` 是 harness，`E` 是环境，`B` 是 token、工具和时间预算，`D` 是任务数据。任何一个变量改变，结果都可能变化。

例如相同模型：没有工具时只能给修复建议；加入 shell 和测试后可以提交 patch；加入 retry 后遇到编译错误可以恢复；加入隐式网络后可能获得额外资料，但也产生公平和安全问题。所有这些都属于被测系统的一部分。

## 19.3 Harness 提供了什么

一个完整 harness 可能包括 prompt 和角色模板、上下文管理、RAG、工具注册、权限和沙箱、scheduler、retry、verifier、人工确认、trace、成本计量和结果 judge。

这些组件并非装饰。上下文折叠决定模型能否记住约束；工具 schema 决定参数能否被正确解析；verifier 决定成功如何定义；retry 决定暂态故障是否被恢复；权限决定错误动作的后果。

## 19.4 一个公平的对照实验

要比较两个模型，先固定环境和 harness：同一仓库 commit、同一工具版本、同一 timeout、同一最大 token、同一测试和同一权限。再分别运行模型，保存完整轨迹。

如果要比较两个 harness，则固定模型 revision 和总预算，清楚地报告新增组件。例如实验 B 加了检索和一次自动测试修复，那么提升只能表述为“模型加新 harness 的提升”，不能写成“模型本身提升”。

## 19.5 成功判据必须由环境验证

Agent 任务的最终文本经常会声称“已完成”，但 harness 应由环境和 verifier 判断。代码任务检查隐藏测试和 diff；浏览器任务检查最终业务状态；研究任务检查引用和事实；结构化工具任务检查 schema 和副作用。

可以定义任务级分数：

```math
S(\tau)=\mathbf{1}[G(E_T)=1]
```

轨迹 `\tau` 的文本质量可以作为辅助指标，但不能替代环境谓词 `G`。部分成功、可恢复失败和危险失败也应该单独分类。

## 19.6 成本和重试的归属

一次任务的成本至少包括模型、工具、验证、重试和人工：

```math
C_{\mathrm{task}}
=C_{\mathrm{model}}+C_{\mathrm{tool}}+C_{\mathrm{verify}}
+C_{\mathrm{retry}}+C_{\mathrm{human}}
```

如果 harness 自动重试三次，最终成功率可能升高，但单位成功成本和 p99 也可能升高。报告模型成绩时不能只展示最好一次，应该说明总预算和重试策略。

## 19.7 Trace 是评估资产

评测记录应包含模型 revision、prompt、tool schema、每次 action、observation、state diff、verifier、错误类别、token、时间和安全事件。只保存最终答案，无法知道失败来自模型、工具、解析器还是环境。

Trace 还支持回放和回归。模型换版本时可以重放同一任务；harness 换 parser 时可以检查 tool event 是否仍然一致；环境发生 reset bug 时可以定位污染来源。

## 19.8 一个代码 Agent 的评测表

| 指标 | 含义 |
| --- | --- |
| apply rate | patch 能否应用 |
| public pass | 是否通过公开测试 |
| hidden pass | 是否通过隐藏测试 |
| regression | 是否破坏原行为 |
| tool recovery | 工具失败后是否恢复 |
| cost | 模型、工具、验证和重试成本 |
| safety | 越权命令、敏感路径访问和危险副作用 |

如果 harness 提供了自动修复测试的工具，就要单独报告它的调用次数和成功率。否则读者会把工具修复带来的能力误认为模型一次生成的能力。

## 19.9 失败归因

可以按以下顺序诊断：先检查输入和模板是否正确，再检查工具是否返回了预期 observation，随后看模型 action 是否合理，最后看 verifier 是否误判。若同一模型在多个环境都失败，可能是模型能力问题；只在一个 harness 失败，优先查编排、权限或协议。

做组件消融很重要。移除 context folding、retry、RAG、verifier 或特定工具，观察结果变化；也要测试每个组件的副作用，例如 retry 是否重复写操作，RAG 是否引入过期文档。

## 19.10 常见误区

把 benchmark wrapper 当成无关实现细节；用不同工具和不同预算比较模型；只报告最终成功，不报告轨迹和成本；用一个 LLM judge 代替可执行 verifier；把 harness 的检索和修复器贡献归因给模型；只保存 URL 不保存版本和环境。

## 19.11 面试回答与练习

回答“如何公平评估 Agent 模型”时，应写出 `R=F(M,H,E,B,D)`，固定或明确声明 harness、环境、工具和预算；用环境状态和 verifier 定义成功，保存 trace，报告成本、重试、人工和安全；再通过消融区分模型能力与 harness 能力。

练习一：列出一个 coding harness 的所有非模型组件，并说明每个组件可能如何改变成绩。

练习二：设计一个 retry 消融实验，比较成功率、p99、重复副作用和单位成功成本。

练习三：给一次“模型说成功但测试失败”的轨迹做失败归因。

### 19.11.1 如何把模型能力和 harness 能力拆开

一个 Agent 系统的结果可以抽象为：

```math
R=F(M,H,E,B,D)
```

但这个式子不是一句免责声明。要研究各部分贡献，需要做受控消融：固定模型，替换检索器、重试、context folding、verifier 和工具；固定 harness，替换模型；固定总预算，比较单 Agent 和多 Agent。每次只改变一个主要变量，并记录所有外部事件。

例如加入 retry 后成功率上升，可能是暂态网络故障被修复，也可能是系统把失败任务重复提交了多次。必须同时看重复副作用、总 token、工具调用和 p99。只看最终成功率无法判断 harness 是否在掩盖模型问题。

### 19.11.2 因果评测的最小设计

可以建立四个条件：

| 条件 | 模型 | harness | 目的 |
| --- | --- | --- | --- |
| A | M1 | H1 | 基线 |
| B | M2 | H1 | 模型变化 |
| C | M1 | H2 | harness 变化 |
| D | M2 | H2 | 联合部署 |

在同一任务和环境中比较 A/B，估计模型变化；比较 A/C，估计 harness 变化；D 观察交互项。若 H2 只对 M2 有效，说明存在协议或能力匹配，而不是单纯的通用提升。

### 19.11.3 trace 是评测数据的一部分

trace 要保存模型请求、工具 schema、策略决策、环境 snapshot、重试、折叠、验证、artifact 和成本。敏感内容要脱敏，但不能只保存最终文本。回放时应能重建失败前的状态，判断问题来自模型计划、工具执行、环境漂移还是判定器。

### 19.11.4 发布判断

发布条件应同时检查任务成功、危险失败、单位成功成本、尾延迟、取消、恢复和人工接管。一个 harness 让 benchmark 分数上升，却扩大权限、增加不可逆动作或把失败隐藏在摘要中，不应被称为质量提升。

## 19.12 评测 harness 的版本和归因

harness 自己也要版本化：prompt、工具 schema、环境镜像、reset、verifier、重试和预算都要绑定 revision。否则模型升级后分数变化无法归因。

最小对照包括裸模型、固定工具、完整 harness 和故障注入 harness。比较结果时，把模型错误、工具错误、环境错误和评测器错误分开记录。

## 19.13 模型、harness 和环境的归因

一次 Agent 任务成功率可以拆成多个条件事件：模型是否提出正确计划，工具是否被正确调用，权限是否允许，环境是否稳定，验证器是否接受，副作用是否提交。单一 success rate 无法告诉我们瓶颈在哪一层。

可以记录：

~~~math
P_{\mathrm{success}}
=P_{\mathrm{plan}}
\cdot P_{\mathrm{tool}}
\cdot P_{\mathrm{permission}}
\cdot P_{\mathrm{environment}}
\cdot P_{\mathrm{verify}}
~~~

这不是独立性假设，而是排查框架。对每一项做故障注入和重放，才能知道换模型、换工具或改 harness 哪个最有效。

## 19.14 trace replay 和反事实评估

可靠评估要保存模型输入、工具 schema、工具输出、权限决定、文件 diff、环境版本、时间和 grader。重放时可以替换模型或某一步工具，比较后续轨迹、成本和最终状态。

反事实评估要注意副作用：真实写入、发送和删除不能在重放中再次执行。应使用快照、模拟器和幂等接口，确保评估本身不会修改生产资源。

## 19.15 成本和可靠性的联合指标

Agent 可能通过更多轮次提高成功率，也可能通过重试掩盖不稳定。报告应包括：

~~~text
task success
first-attempt success
tool calls
tokens
wall-clock
human interventions
side-effect errors
unit success cost
~~~

按任务难度和风险分桶，比平均值更能指导 harness 设计。高风险任务还应加入错误副作用权重，而不只是完成率。

## 19.16 把“模型能力”拆成可观测的条件事件

最终成功率太粗，无法说明系统为什么失败。一个代码 Agent 的任务至少经过计划、工具参数、权限、环境执行、验证和提交几个边界。可以把一次成功写成条件事件链：

```math
P_{\mathrm{success}}
=P_{\mathrm{plan}}
 P_{\mathrm{action}\mid\mathrm{plan}}
 P_{\mathrm{permission}\mid\mathrm{action}}
 P_{\mathrm{environment}\mid\mathrm{permission}}
 P_{\mathrm{verify}\mid\mathrm{environment}}
 P_{\mathrm{commit}\mid\mathrm{verify}}
```

这个乘积不是说各事件相互独立，而是一个按轨迹分段的排查框架。某一层失败时，trace 需要给出可验证证据：计划是否引用了正确文件，工具参数是否符合 schema，权限策略是否拒绝，环境是否返回超时，验证器是否误判，提交是否因 revision 冲突失败。

例如，模型提出了正确 patch，但 harness 把相对路径解析成了宿主机路径，任务失败就不能记为模型不会修复。相反，工具正常执行而模型反复选择不存在的文件，才更接近模型计划问题。把这些案例都放进一个“失败”桶，会让后续优化方向完全相反。

## 19.17 评测矩阵与受控随机化

比较 Agent 系统时，任务集、环境和预算必须形成可复现的矩阵。任务不应只按领域分组，还应按难度、工具数量、状态长度、是否有副作用和是否需要恢复分桶。每个任务要固定初始环境 snapshot，并在运行前生成独立 workspace，避免前一个模型留下的文件影响后一个模型。

一个最小矩阵如下：

| 维度 | 低风险条件 | 高风险条件 |
| --- | --- | --- |
| 工具 | 只读搜索、计算 | 写文件、发布、外部 API |
| 状态 | 单轮、短上下文 | 长任务、折叠、断线恢复 |
| 验证 | 公开测试 | 隐藏测试、业务状态、人工审批 |
| 故障 | 无故障 | 超时、乱序、权限变化、revision 冲突 |
| 预算 | 固定 token | token、工具时间和重试联合上限 |

每个条件运行多次，并使用固定随机种子或明确记录采样策略。不同模型的顺序也要随机化或平衡，避免环境热缓存、服务负载和时间段成为混淆因素。若实验不能完全随机化，报告中应把限制写清楚，而不是把一次顺序运行的差异当成模型差异。

开放式生成可以使用人工或 judge，但代码、结构化输出和副作用优先用可执行 verifier。人工评分还要保存标注指南、标注者数量、一致性和盲评信息；否则“更好”可能只是评审者知道模型名称后的预期。

## 19.18 从 trace 到数据集：一条失败也有价值

评测 trace 不只是调试日志，它可以成为下一轮回归和训练数据。一个可复用的 trace 记录至少包括：

```text
run_id, task_id, model_revision, harness_revision
environment_snapshot, budget, prompt_hash
event_sequence, tool_schema_hash, policy_decisions
workspace_before, workspace_after, artifacts
verifier_result, error_category, cost, latency
```

用户输入、密钥和企业数据需要脱敏；但脱敏不能破坏任务结构、工具参数关系和 revision 依赖。可以保存哈希、字段类型和受控替身，另由授权系统保留原文映射。没有原始版本、tool call id 和环境 snapshot，后续重放往往只能得到一个与线上不同的“近似失败”。

失败样本要按根因标注，而不是全部丢弃。`wrong_plan`、`invalid_arguments`、`tool_timeout`、`permission_denied`、`verifier_bug`、`environment_drift` 和 `unsafe_side_effect` 对应的修复手段不同。重复出现的 `verifier_bug` 甚至说明评测系统不可信，不能拿它给模型排名。

## 19.19 worked example：retry 提高通过率却降低单位价值

假设同一批 100 个代码任务在两个条件下运行：A 禁止自动重试，B 在读操作和测试上最多重试两次。结果如下：

| 指标 | A：无重试 | B：有重试 |
| --- | ---: | ---: |
| 首次通过任务 | 54 | 54 |
| 最终通过任务 | 54 | 67 |
| 平均工具调用 | 3.1 | 5.8 |
| 平均模型 token | 8.2K | 9.0K |
| p95 延迟 | 42s | 91s |
| 重复写入 | 0 | 2 |

如果只看最终通过率，B 从 54% 变成 67%，似乎明显更好；但它同时扩大了延迟和工具开销，并产生了两次重复写入。若每个成功任务的模型、工具和验证成本分别为 `1.0`、`0.4`、`0.2` 个成本单位，B 的重试平均增加 `0.7`，单位成功成本可能从 `1.6/0.54` 变成 `(1.6+0.7)/0.67`，改善幅度远小于通过率差异；若重复写入有高风险，还应直接触发发布条件。

这个例子也说明 retry 不是单一开关。读操作、纯计算、可查询的 job 和不可逆写操作应有不同策略。每次 retry 都要保存原因、退避时间、请求哈希和副作用状态；否则“系统自动恢复”可能只是把一个未知状态重复发送。

## 19.20 发布条件与反事实检查

最终发布判断应同时考虑成功、质量、成本和危险失败。可以写成硬性条件与软分数的组合：

```math
G_{\mathrm{release}}
=G_{\mathrm{success}}
 G_{\mathrm{safety}}
 G_{\mathrm{replay}}
 G_{\mathrm{cost}}
 G_{\mathrm{latency}}
```

软分数可以排序候选方案，但不能用高平均分抵消一次未授权删除。上线前还要做反事实检查：把同一 trace 在禁止写入的模拟器中重放，确认权限、工具参数和验证顺序不依赖真实副作用；把 verifier 替换成更严格版本，检查结果是否仍成立；把 context folding 或 retry 移除，明确系统收益来自哪个组件。

模型、harness 和环境都要绑定版本。新模型在旧 harness 上变差，可能是 tool schema 不匹配；新 harness 在旧模型上变好，可能是自动修复器接管了更多工作。发布报告应分别给出裸模型、固定工具、完整 harness 和故障注入条件，才能让读者知道被比较的究竟是什么。

## 19.21 评测结果的归因矩阵

一次任务失败可能来自模型计划错误、harness 丢失上下文、工具返回错误、环境配置变化或 verifier 误判。可以固定其中两项，只替换另一项，建立最小归因矩阵：同模型换 harness、同 harness 换模型、同二者换环境、同环境换 verifier。没有这种对照，报告中的“模型提升”可能只是重试或更宽松的成功判据。

## 19.22 重试不是免费能力

重试可能提高成功率，也可能重复外部副作用、污染 trace、延长队列和掩盖根因。评测应分别记录首次成功率、最终成功率、重试次数、重复副作用率和单位成功成本，并把允许重试次数写入实验条件。对不可逆动作，重试前必须先查询幂等状态。

## 19.23 trace replay 的边界

回放可以重现输入、工具事件和状态转移，但不能假设所有外部世界都静态不变。网页、价格、权限和数据库可能已经变化。回放器应区分真实重放、冻结 observation 的离线重放和模拟环境重放，并在结果中标出环境差异。三者可以互相补充，不能把模拟成功写成生产保证。

## 19.24 评测的成功判据要落到 artifact

代码任务的成功是 patch 通过隐藏测试并符合仓库约束，研究任务的成功是结论有可追溯证据，业务任务的成功是外部状态正确更新。模型的最后一句总结只能作为一个 artifact，不能成为唯一判据。评测 harness 要直接读取文件、测试、数据库状态或人工验收结果。

## 19.25 失败轨迹的价值

失败样本应记录首次错误位置、错误类型、环境 observation、工具状态、重试、最终结果和是否可恢复。把所有失败都丢弃会让下一轮只看到成功模板；保留失败并标注责任边界，才能改善模型、harness 或环境的正确层。

## 19.26 发布前的反事实检查

把同一任务在无重试、允许重试、旧 harness、新 harness、冻结 observation 和实时环境下运行。若只有允许重试时成功率上升而单位成本和副作用恶化，不能称为能力提升；若新 harness 减少了工具错误但模型输出不变，应把收益归给 harness。

## 19.27 线上指标和离线分数的分工

离线评测适合比较固定任务、模型和 harness；线上指标还要观察流量分布、用户取消、权限拒绝、工具失败、排队和人工接管。线上成功率上升可能来自任务变简单，离线分数下降也可能来自新流量更难，二者要用 workload bucket 对齐。

## 19.28 评测回归的最小发布包

每次发布保存模型/harness/environment revision、golden trace、失败样本、指标快照、成本、风险和回滚版本。只有最终文本、没有事件和 artifact 的评测包无法证明工具协议和外部状态没有回归。

## 19.29 Harness 评测的阶段性判断与资料边界

Harness-aware evaluation 的核心是承认 Agent 能力属于系统，而不是孤立的模型输出。只有绑定模型、编排、环境、预算、工具和数据，评测结果才可解释、可复现、可用于生产决策。条件事件、任务分桶、trace 回放、组件消融和副作用验收条件，才能把“分数变化”还原为可行动的原因。

公开 benchmark 可以提供任务和指标参考，但任何结果都应保留 harness 版本、工具权限、重试、环境和日期；厂商或框架的默认 harness 不能被默认为中立条件。

## 19.30 Grok 4.20：Multi-agent 评测必须拆开编排收益

Grok 4.20 的 `grok-4.20-multi-agent` 为研究任务提供 4 或 16 个协作 Agent，但 Artificial Analysis 的 `grok-4-20` 页面与 DataCurve 的 `mini-swe-agent` 行不是同一评测对象；当前 DataCurve 没有精确 `mini_swe_agent_grok_4_20_*` 行。评测 manifest 应至少写入：

```text
model_id, snapshot, effort, agent_count
tool set, source allowlist, prompt revision
sub-agent roles, leader policy, max_turns
environment, verifier, retries, cost and latency
```

比较 4 Agent 与 16 Agent 时，不能只保持问题不变，还要记录并行检索是否带来重复来源、错误互相确认、leader 摘要丢证据和额外工具账单。建议把结果拆成四层：单 Agent 事实质量、子 Agent evidence recall、leader synthesis quality、最终 verifier/artifact success。这样才能判断收益来自覆盖面、汇总策略还是更高的 token 预算。

此外，xAI 的普通模型页写 1M maximum prompt，而 Artificial Analysis 当前页面写 2M context；这是 capability manifest 的冲突，不应在评测报告中静默选择较大的数字。请求前应按精确 model ID/endpoint 做上限探测，并把 compaction、prompt cache、工具 schema/result 和输出预算列入成本账本。没有这些字段，所谓“长上下文/多 Agent 更强”无法复现。

## 19.31 Gemini 3.8 Flash：把模型、Interactions 和工具 harness 分层

Gemini 3.8 Flash 的 DataCurve high 行是一个完整系统配置，而不是裸模型测量：`gemini-3-8-flash`、high thinking、`mini-swe-agent`、工具、任务环境和 verifier 共同决定 Pass@1、Pass@4、输出 token、Agent steps 和成本。评测 manifest 应同时记录模型 snapshot、thinking level、`max_output_tokens`、Interactions state 模式、工具 schema、Computer Use 审批策略、缓存命中、任务仓库 revision、重试和 verifier 版本。

Interactions 的 step trace 让归因更细：可以区分模型没有提出正确计划、工具没有执行、工具结果没有被正确回灌、signature/state 丢失、宿主拒绝动作和 verifier 误判。对同一任务做 low/medium/high 消融时，必须固定其余变量，并报告 thinking/visible output、工具失败、重复副作用、TTFT/TPOT、最终 artifact 和单位成功成本。单看 Pass@1 会把长轨迹和高成本隐藏掉。

1M context 也只表示接口预算。长文档的 needle 位置、工具 schema/result、implicit cache hit、缓存前缀、输出预算和多轮状态会改变有效召回与延迟；因此不能用 context 数字代替长上下文评测，也不能把服务侧 implicit caching 写成 GPU KV cache。完整复现实验设计见 [`EXERCISES.md`](../../EXERCISES.md) 与 [`gemini-3.8-flash-source-notes.md`](../../research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。

本轮官方协议复验还要求 manifest 记录 Interaction 的 retention/deletion、`store` 模式、`previous_interaction_id`、每轮重新指定的 tools/system/generation config，以及 signature 字段的实际位置。Thinking 与 Tool combination 文档对标准 function call 是否带 signature 有范围差异，评测器不能用假设字段填充缺失值；应保存原始 response schema、call/result `id`、opaque fields 和 capability-probe 结果。否则所谓 stateless replay 可能在丢失 reasoning/tool context 后仍错误地计为成功。

## 19.32 GLM-5.1：长周期任务要拆分反馈信号与验收门禁

Z.ai 的官方博客 [*GLM-5.1: Towards Long-Horizon Tasks*](https://z.ai/blog/glm-5.1) 给出的重点不是“上下文更长所以自然能工作 8 小时”，而是让 Agent 在工具执行、观察结果、修订策略和验收之间形成外层闭环。博客中的数字属于发布方实验设置，不能改写成 GLM-5.1 的内部推理深度、训练算法或独立复现结果。研究底稿和快照证据见 [`glm-5.1-source-notes.md`](../../research/model-update-2026-09/glm-5.1-source-notes.md)。

这篇博客把长周期任务拆成三种反馈条件，面试时应分别讨论：

| 条件 | 反馈信号与外层循环 | 验收与边界 |
| --- | --- | --- |
| VectorDBBench | Rust ANN 数据库、SIFT-1M 和 Recall ≥ 95% 约束；模型在 50-turn 工具预算内执行，外层循环反复 edit、compile、test、profile、submit。博客报告 600+ iterations、6,000+ tool calls、21.5k QPS，并观察到约第 90、240 轮的结构性策略切换。 | Recall 是质量门槛，QPS 是优化目标；必须绑定数据集、硬件、编译参数、停止条件和 profile 方法。不能把 21.5k QPS 当作裸模型能力。 |
| KernelBench Level 3 | 50 个问题、每题独立 Docker、1 张 H100、最多 1,200 次工具回合；模型通过数值正确性检查后才比较加速。 | `atol=rtol=1e-4` 是 correctness gate；Claude Opus 4.6 与 GPT-5.4 做 benchmark-exploitation 审计并取较低 speedup。博客报告 GLM-5.1 约 3.6×、`torch.compile max-autotune` 为 1.49×，但这是该 harness 的发布方结果。 |
| Linux desktop | 没有单一标量目标；每轮由 self-review harness 检查缺失功能、粗糙样式、坏交互和边界情况，再继续迭代约 8 小时。 | 自评可以产生下一步反馈，却不是独立 verifier。最终仍需读取 artifact、运行测试、检查业务状态或人工验收，不能用“模型说完成了”作为成功判据。 |

这个对照说明长周期 Agent 至少有三层问题：

1. **反馈是否可计算。** 有标量目标时，Recall、QPS 或 kernel correctness 可以驱动搜索；没有标量目标时，只能把 rubric、self-review 和外部检查组合起来，不能假定模型的自评无偏。
2. **外层循环是否真的改变策略。** 600 多轮不等于有效进步。应记录每轮 patch、编译/测试/profile 输出、失败原因、策略切换和回滚；如果模型只是重复同一动作，工具调用数不能证明规划能力。
3. **最终结果是否由独立门禁确认。** 代码、性能和桌面任务都要把模型产出的 artifact 交给 verifier。正确性、性能、反作弊审计和权限安全应分别计数，不能用一个总分掩盖危险副作用或错误验收。

因此，GLM-5.1 的长任务评测 manifest 至少应固定：

```text
model_id, model_revision, provider
task_set, data_revision, initial_snapshot
harness_revision, tool_schema, tool_budget, wall_clock_budget
feedback_signal, correctness_verifier, performance_verifier
anti_exploitation_auditor, retries, human_intervention
artifact_digest, final_state, cost, latency, failure_category
```

比较不同模型时，必须同时固定任务、环境、工具和预算；比较不同 harness 时，则应固定模型 revision，并把新增的 retry、profile、self-review 或 verifier 单独计入。GLM-5.1 博客提供的是一个很好的面试切入点：真正要问的不是“能否连续运行 8 小时”，而是“每一轮拿到什么反馈、何时发生策略更新、什么条件阻止错误继续扩大，以及谁最终确认 artifact 正确”。完整发布方数字、哈希和未公开项见研究笔记；当前没有精确 DataCurve GLM-5.1 行，因此不能把相邻 GLM 版本的 Agent 成绩迁移过来。

## 19.33 Claude Sonnet 5：把 System Card 结果放回 harness

Sonnet 5 的公开结果至少分成三账：Artificial Analysis 的 `max` Intelligence Index、DataCurve 的五档 `mini-swe-agent` 结果、Anthropic System Card/发布方 benchmark。它们的 provider、effort、工具、任务集、环境、verifier、trials、safeguards 和统计方法不同，不能凭模型名称做横向排名。

System Card 的安全数字也需要同样的 manifest。Claude Code 恶意请求拒答率 92.37%、computer-use 恶意任务拒答率 84.68%、Gray Swan IPI 的 28 个场景和 1,130 个去重攻击，测量的是给定策略与 Agent harness 的行为。一个可复现的记录至少包括：

```text
model_id, snapshot, effort, thinking state
tools, permissions, sandbox, network policy
task/scenario revision, safeguards, trials
verifier, retry/timeout, compaction trigger
refusal/action/side-effect/artifact outcomes
```

发布方 benchmark 的 `85.2%` SWE-bench Verified、`80.4%` Terminal-Bench 2.1、`84.7%` BrowseComp 和 `81.2%` OSWorld-Verified，应在报告中保留来源标签；它们不能替换固定任务、固定 harness 下的独立回放。安全评测中的“拒绝”也不能自动计为业务成功，应该与正确执行、权限阻断和人工接管分开统计。

## 19.34 Grok 4.7：长轨迹评测要把状态与发布方 benchmark 分账

Grok 4.7 的 xAI 发布页给出 DeepSWE v1.1 `71.0%`、Terminal-Bench 4.0 `38.0%`、CursorBench 4.0 `46.3%`、Harvey `19.6%`、HealthBench Professional `56.7%` 等结果；它们各自绑定发布方任务、prompt、effort、工具、环境和统计口径。Artificial Analysis 的 `46.4465` Intelligence Index 是另一种 provider 测量，DataCurve 又没有精确 `mini_swe_agent_grok_4_7_*` 行。三者不能排成一张裸模型榜。

对 Grok 4.7 的 harness 回放，建议把每条任务 trace 固定为：

```text
model/revision/effort/provider
encrypted reasoning item / compaction trigger and item
tool schema / allowed_tools / permission / execution receipt
task environment / retries / timeout / verifier
artifact digest / side effects / success / cost / latency
```

自验证可以作为反馈信号，但不能作为唯一验收者。压缩后的 opaque item 还必须与工具回执、未完成副作用和 workspace artifact 一起恢复；否则 benchmark 可能因重复执行或丢失历史而虚高。完整研究记录见 [`grok-4.7-source-notes.md`](../../research/model-update-2026-09/grok-4.7-source-notes.md)。

## 19.35 Claude Opus 5.5：少 token 不等于少责任

Anthropic 对 Opus 5.5 的长任务描述强调更少的工具调用、步骤和输出 token。对 harness 来说，这应转化成效率指标，而不是删除 trace：首次上下文读取、patch、命令、测试、重试、fallback 和 verifier 都要留下可回放事件。否则无法判断成本下降来自真正的 token efficiency，还是来自少做了一项测试。

发布方 benchmark 中的 adaptive/max、xhigh、medium/default 以及 WANDR 的离线搜索、代码执行和 980K task budget 都是不同 harness。评测记录应把 `model + effort + fallback + tools + environment + verifier` 当作一个配置键；AA 的第三方指数和 Anthropic 的发布方结果不能和 DataCurve 的 `mini-swe-agent` 结果拼成裸模型排名。

特别要审计安全路由：Opus 5.5 的 cyber/biology safeguards 可能将任务透明地交给 Opus 4.8 或 Opus 5。trace 必须显示实际模型、触发类别、权限、缓存、重试和最终 artifact；“最终成功”不能在缺少这些字段时归因给 primary model。模型声称完成、工具返回成功和独立 verifier 通过仍是三个不同状态。

Opus 5.5 的官方 API 文档把 harness 的状态账本再推进了一层：thinking block 与生成模型和 conversation 前缀绑定，tool-call 之间的进度默认可能以空的 thinking block 返回，按需 compaction 会产生签名摘要，fast mode 则在同一模型上切换更快推理配置。评测 trace 至少保存 `model_id`、`thinking_binding`、`prefix_hash`、`compaction_digest`、`tool_schema_version`、`speed`、`usage.speed` 和错误/拒答字段；否则无法重放一次“模型能力下降”到底是状态丢失、工具契约不兼容、上下文压缩还是限流。

同样不能把模型级 refusal 当作 harness 级完成。官方契约允许 HTTP 200 搭配 `stop_reason: "refusal"` 和 policy category，fallback 触发后要把实际执行模型和最终 artifact 分开计分。来源：[What's new](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)、[Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)。

System Card 的安全数字也必须进入同一条 trace：Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 为 `0.1%/0.7%/1.0%`，约 18% rollout fallback 到 Opus 4.8。coding Shade 的无 safeguards/probe 开启结果为 `54.61%/11.13%`，computer-use probe 为 `0.04%`，browser auto 为 `0/110`。这些数的分母、probe、fallback 和 safeguards 都要保留，否则无法解释“拒答率下降”到底来自模型行为还是路由。

OSWorld 2.0 的 partial/strict `81.8%/48.7%` 还揭示了 compaction 的评测责任：108 tasks、1080p、最多 500 actions、5 runs、完整 screenshot，以及超过 100K tokens 后的 server-side compaction 都是条件。多 Agent 的约 `2.7x`/`2.8x` speedup 使用 derived latency；harness 应额外记录共享资源、排队和真实 wall-clock，不能直接把论文式 speedup 当作线上 SLO。
