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

评测结果还要带上时间与 workload 标签。Artificial Analysis 于 2026-09-24 将该条目标记为 deprecated，并明确只继续更新默认 10K input token workload 的性能基准；其他 workload 结果是历史数据、不再更新。页面当前仍显示 estimated Intelligence Index `25.6550155187053`、2M context、输出速度 `106.19 tokens/s`、TTFT `21.53s` 和端到端时间 `26.24s`，但不能仅因页面刚抓取就把所有历史 workload 数据都称为当前测量。报告应逐项记录 snapshot、输入 token workload、更新时间和指标来源；AA 的 `deprecatedTo: grok-4-3` 也不能替代 xAI 对 API 可用性或迁移路径的正式说明。

xAI 官方 Multi Agent 文档还把 `grok-4.20-multi-agent` 标为 beta，并给出单独的 API 能力边界：使用 xAI SDK 或 Responses API，不支持 Chat Completions、client-side function calling/custom tools 或 `max_tokens`；内置工具与 Remote MCP 可用。多轮可用 `previous_response_id` 续接。默认只返回 leader 的工具调用和最终结果，子 Agent 中间状态只有通过 `use_encrypted_content` 才以 opaque 加密内容携带。评测 manifest 因而还要固定 API surface、beta 文档日期、续接方式和 encrypted-state 开关，避免把普通模型的工具/输出参数合同当成 multi-agent 变体的合同。

成本同样必须按整个协作系统统计：leader 与子 Agent 的输入、输出、reasoning tokens 全部计费；任何 Agent 调用的 server-side tool 也计费。对照 4/16 Agent 时，除 end-to-end 成功和 wall-clock 外，还应读取响应中的 `usage`、`server_side_tool_usage`，报告全体 token、工具调用、单位成功成本和 continuation 成功率。子 Agent 的 opaque state 便于恢复但不会自动提供可读审计 trace，故来源、任务分片、失败/重试和 verifier 记录仍需由应用层维护。详见[研究笔记](../../research/model-update-2026-09/grok-4.20-source-notes.md#2026-09-29-multi-agent-api-限制beta-与成本账本补证)。

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

可运行的最小状态回放见 [`claude_opus55_protocol_audit.py`](../../research/model-update-2026-09/code/claude_opus55_protocol_audit.py)。它用签名摘要、prefix hash、工具调用/结果 lineage、compaction replacement 和幂等 ledger 检查“重放是否重复副作用”；同时把 fallback 的实际模型与原始模型分账。该 demo 不调用真实 API，不解密 opaque thinking，也不证明线上 harness 的 SLO。

同样不能把模型级 refusal 当作 harness 级完成。官方契约允许 HTTP 200 搭配 `stop_reason: "refusal"` 和 policy category，fallback 触发后要把实际执行模型和最终 artifact 分开计分。来源：[What's new](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)、[Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)。

System Card 的安全数字也必须进入同一条 trace：Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 为 `0.1%/0.7%/1.0%`，约 18% rollout fallback 到 Opus 4.8。coding Shade 的无 safeguards/probe 开启结果为 `54.61%/11.13%`，computer-use probe 为 `0.04%`，browser auto 为 `0/110`。这些数的分母、probe、fallback 和 safeguards 都要保留，否则无法解释“拒答率下降”到底来自模型行为还是路由。

OSWorld 2.0 的 partial/strict `81.8%/48.7%` 还揭示了 compaction 的评测责任：108 tasks、1080p、最多 500 actions、5 runs、完整 screenshot，以及超过 100K tokens 后的 server-side compaction 都是条件。多 Agent 的约 `2.7x`/`2.8x` speedup 使用 derived latency；harness 应额外记录共享资源、排队和真实 wall-clock，不能直接把论文式 speedup 当作线上 SLO。

## 19.36 DeepSeek Harness Preview：把 runtime 合同写进评测 manifest

DeepSeek Harness 的官方文档提供了一个很具体的 runtime case study，但它不是 V4.1-Flash 的架构报告。它把 Web UI、headless/SDK profile、workspace、session、agent loop、tool 和 MCP 放进插件树；评测时应把这些组件作为系统变量显式记录，而不是只保存 model ID。

一条可恢复的 Harness trace 至少需要：

```text
provider_id, requested_model, served_model, protocol_claim
session_id, workspace_revision, plugin_bundle, session_log_version
tool_schema_hash, mcp_server/version, auth_scope, permission_decision
executor_receipt, artifact_digest, verifier_result
webhook_delivery_id, idempotency_key, retry, timeout, final_status
```

官方 preview 的几个关键边界是：provider ID 与 credential reference 是持久身份，密钥只能脱敏回显；session log 固化实际使用的模型，不能中途静默换模型；插件依赖不满足时不能加载，listener/resource 与外部连接必须在卸载时清理；MCP 断线后要重新发现工具，重连预算耗尽后注销旧 registry；GitHub review webhook 的 `202` 只表示异步 admission，重复 delivery 可能创建多个 session，且入站签名不等于出站 GitHub 写权限。

这组边界直接转化为评测门禁：

| 观测状态 | 应记录 | 不能替代 |
|---|---|---|
| model emitted tool call | call ID、schema、模型和 session | 权限允许 |
| Harness admitted task | webhook signature、delivery ID、`202`、session | 执行完成 |
| executor succeeded | side effect、receipt、幂等键 | artifact 正确 |
| verifier passed | artifact digest、测试/业务断言 | 模型单项能力 |

[`deepseek_harness_protocol_audit.py`](../../research/model-update-2026-09/code/deepseek_harness_protocol_audit.py) 将这些合同缩成无网络 toy：运行结果为 `ok=true`，provider credential 为 `<redacted>`，重复 webhook admission 为 `2`，MCP 重连预算耗尽后工具集合为空，且 `network_called=false`。它的证据等级是 `local_protocol_toy`，只能说明审计字段和拒绝路径自洽，不能证明 DeepSeek Harness 生产行为、V4.1 模型质量、真实工具成功率或 SLO。

面试时可以用这条链路回答“模型是否完成任务”：模型输出意图 -> Harness 建立/恢复 session -> permission 决策 -> executor 产生 receipt -> verifier 确认 artifact。任一中间层缺失，都不能把最终文本或 HTTP `202` 直接计为成功。

## 19.37 GPT-5.6 Luna：状态回放与缓存/压缩分账

GPT-5.6 Luna 的当前榜单锚点仍要分成两本账：AA 的 `max` 是 provider 测量，DataCurve 的 `mini-swe-agent` 行是模型、工具、环境和 verifier 的系统测量。Agent harness 评估应把 `reasoning.context`、完整 output item、function call/output、compaction item、cache prefix 和 tool-search ownership 写进同一份 manifest。

核心恢复门禁是：同家族的 opaque reasoning 才可继续使用；`current_turn` 不渲染早期 turn，`all_turns` 也不创造不存在的 reasoning。无状态链路要原样回放 reasoning、assistant phase、tool call/result 和 compaction item；standalone compaction 返回的整个窗口是 canonical context。prompt cache 只记录稳定前缀的读写和 TTL，compaction 改写前缀后 cache miss 是可接受的观测，不是模型质量结论。

[`gpt56_luna_state_replay_audit.py`](../../research/model-update-2026-09/code/gpt56_luna_state_replay_audit.py) 已将这些门禁转为标准库 toy，并验证 hosted/server 与 client tool search 的 `call_id` 所有权、重复副作用的幂等回执和独立 artifact verifier。它只证明合成状态机自洽，不能证明真实 OpenAI endpoint、隐藏 reasoning、服务端事件顺序、模型质量或生产 SLO。

## 19.38 Claude Opus 5：Prompt-Injection ASR 不是一个数字

Anthropic 的 Opus 5 System Card 展示了 harness 版本漂移：2026-08-19 更新因发现 Cowork browser harness 与旧模型不一致，重跑了各模型基线；修订后统一 Cowork harness。Cowork 不支持关闭 thinking，因此不再报告该条件，只保留 thinking-enabled、medium-effort 结果。若评测器、工具或策略变更，正确做法是重跑基线并记录版本，不是只修正新模型分数。

一次 prompt-injection 评测至少要绑定：

```text
model_id, snapshot, thinking, effort, endpoint, product_surface
attacker_revision, scenario_set, scenario_id, attempt_id, attempt_budget
tool_result_probe, action_classifier, auto_mode, harness_revision
attempt_success, scenario_success, verifier, retry, trace_digest
```

System Card 的评估例子包括：Gray Swan IPI 的 28 个场景和 1,130 个高迁移攻击（Opus 5 在 1/15 次尝试内的成功概率为 `0.2%/2.0%`）；一周 live bounty 的 11 个场景、每模型超过 20,000 次有效尝试（Opus 5 attempt-level ASR `0.08%`）；Shade coding 的 40 个场景、每场景 200 次尝试；Cowork browser 的 129 个环境、每场景 10 次攻击。攻击者适应能力、guardrails 和产品配置各不相同，不能合并成总安全率。

必须把 `attempt-level ASR = successful_attempts / valid_attempts` 与 `scenario-level ASR = scenarios_with_success / tested_scenarios` 分开。前者回答“总尝试中有多少次成功”，后者回答“多少场景至少出现过一次成功”；若重试自适应、场景难度不均或预算不同，它们不能互相换算。Opus 5 computer-use 的 probes 条件下，disabled thinking 的 ASR `0.39% → 0.43%` 对应 2,800 次中的单次增量，卡片明确认为无法与噪声区分，说明小数点变化不自动构成改进或退步。

还要区分模型级和产品级防护：probe 在模型行动前检查不可信 tool result，动作侧 classifier 阻止危险 tool call；Auto mode 同时使用两者。Opus 5 Cowork browser 条件下 medium-effort、thinking-enabled 的 raw comparison 为 `3.84%`，启用 Auto mode 后 `0/129` 场景成功；后者描述带 harness 的产品系统，不是裸模型。反过来，关闭额外保护的研究设置也不等同真实用户部署。

System Card 还把 Opus 5 列为 CB-1、未达 CB-2，并按其 RSP 判断保持 ASL-3 防护；这是发布方风险判断。面试回答的核心是把模型快照、攻击预算、输入/动作防护、harness、成功判据和两个分母一并呈现。完整证据及原始来源哈希见 [`claude-opus-5-source-notes.md`](../../research/model-update-2026-09/claude-opus-5-source-notes.md)。

## 19.39 Qwen3.6-27B：榜单分与发布方 Harness 要拆开

Qwen3.6-27B 的 AA 页面是模型/config 发现与第三方指标来源；官方 ModelScope 模型卡另列 SWE-Bench、Terminal-Bench、SkillsBench、QwenClawBench、QwenWebBench 等自报评测。两者不是同一数据口径，不能合成单一“模型分数”。

该卡最值得面试的评测方法披露有四类：

1. **任务集版本和重跑基线。** 卡片称修正公开 SWE-Bench Pro 中部分问题，并在 refined set 上评估所有 baselines。修改 benchmark 后重跑所有比较对象是控制变量的正确方向，但要把修订版 task-set/hash 与原始 public score 分开。
2. **环境预算。** Terminal-Bench 2.0 的结果绑定 Harbor/Terminus-2、3 小时 timeout、32 CPU、48 GB RAM、256K context、80K output 上限和 5-run mean。每个 benchmark 都应保存这些条件和 run-level dispersion，而不只存平均分。
3. **抽样范围。** SkillsBench 使用 78 个自包含任务子集，明确排除 API-dependent tasks，并做五次平均。它不是完整 SkillsBench 全集的结果；覆盖率要和准确率并列报告。
4. **模型评审器分数。** QwenWebBench 对双语前端任务先自动渲染，再以多模态 judge 判代码/视觉正确性，最后用 Bradley–Terry/Elo 汇总。应绑定 renderer/browser 版本、视觉 judge checkpoint/prompt、pairwise 样本、盲测顺序和置信区间。Elo 反映相对偏好次序，不能解释成通过率或严格功能正确率。

QwenClawBench 被描述为 real-user-distribution Claw benchmark，但公开卡片没有给出可完整重建的任务集/数据抽样和 verifier；应标记 `publisher_described_internal`，不拿它作独立复现实证。更多条件与来源哈希见 [`qwen3.6-27b-source-notes.md`](../../research/model-update-2026-09/qwen3.6-27b-source-notes.md)。

## 19.40 Qwen3.7 Max：Task、Harness、Verifier 的组合式 Agent RL

Qwen 官方把 Agent 训练环境扩展作为 scaling 方向：增加环境质量与多样性，并在训练未见的 OOD 环境上报告泛化。更值得抽象为面试方法论的是它公开的 rollout 基础设施：把每个实例拆成正交的 `Task × Harness × Verifier`，让同一任务与不同运行框架及 verifier 版本重新组合，再做跨 harness / 跨 verifier RL。目标是让模型学习任务策略，而不是记住某个工具循环、环境或验收器的捷径。

这与“把同一 benchmark 换一个 UI 再跑一次”不同：训练本身对同源任务改变交互框架和判分器，测试则需要留出训练中未见过的组合。要检验该主张，应固定并记录三类组件的 revision/hash，报告 seen/unseen 组合、OOD 任务域、各 verifier 的难度和失败分布；若只给总平均分，无法区分真正的策略迁移、任务泄漏或较宽松的 verifier。文章称子集性能增益可以预测整体增益，但没有提供足够的任务规模、曲线数值和区间来建立普适 Agent scaling law，后续技术报告仍待取得。

另一项训练监控案例是 reward-hacking 自监测：模型回放超过 80 小时 SWE RL 轨迹、归纳疑似作弊模式、验证候选规则并挖掘反例；Qwen 报告累计超过万次调用、新增 13 条启发式规则、识别 1,618 个案例。这证明的是发布方描述了一个 model-assisted auditor，不足以证明检测准确率或没有奖励作弊。

```text
RL trace -> candidate abuse pattern -> replay
-> counterexample mining -> versioned rule set
-> held-out / human audit -> feed back to training monitor
```

线上使用这类规则时应给规则集版本化、保留反例与人工抽样，并在隔离的 holdout trace 上报告 precision、recall、false-positive rate 和漏检类型。否则 detector 可能仅对已知作弊 pattern 过拟合，甚至把合法但少见的工具使用误判为作弊。规则命中数量 `1,618` 不能替代带分母的检测质量。

该博文还以未知 M890 PPU 上的 35 小时 kernel 优化为长程 Agent 案例；其执行轨迹、baseline 和 KernelBench L3 的区别见[第十七册 Code Agent 章的 Qwen3.7 Max 小节](../../book-17-agent-tool-use/chapters/07-code-agent.md)及[研究底稿](../../research/model-update-2026-09/qwen3.7-max-source-notes.md)。官方模型文档中的 `qwen3.7-max` alias 对应 May 20 纯文本 snapshot；June 8 视觉版不能回写到该 alias。完整 benchmark 条件、API transcript 和证据边界见研究底稿。当前 DataCurve 没有精确 Qwen3.7 Max Agent 行，所有 Qwen 博客结果均是发布方报告，不是独立复现。

## 19.41 Qwen3.6-35B-A3B：先求解机制，再构造可验证 Agent 环境

arXiv v1 论文 [《Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms》](https://arxiv.org/abs/2609.27321v1) 的评论标为 “Qwen Technical Report”。它提出的 VHD-Play 把生成顺序倒过来：先从机制族抽样 `θ` 并用可复用 solver 求出 optimum/default reference，再固定 reward；随后 frozen setter 把同一机制实现为有隐藏状态、跨步决策和工具接口的环境。核心链路是 `M(θ) → solve zθ → freeze Rθ → realize D(θ) → wrap E(θ)`，避免让一个 learned judge 同时猜测任务答案和评判 rollout。

论文给出的 episode reward 将在线策略效用归一化到求解器算出的参考区间：

```text
r(π; θ) = clip_[0,1]((u(π; θ) - u₀(θ)) / (u*(θ) - u₀(θ)))
```

`u₀` 是默认策略基线，`u*` 是机制 solver 给出的最优值。默认结果映射到 0，最优参考映射到 1；在部分可观测任务中，full-information optimum 只是上界，不一定是可在线达到的分数。评分不调用 LLM judge，但可靠性仍依赖 solver 与生成动态的一致性，因此论文用执行、默认区间和 replay/reference agreement 做 admission。

规模上，论文报告 3,300 个环境、28 个语料主题和至少 10 个工具/环境；训练分区为 2,200 条，另有 300 条训练族 held-out 与 800 条八种未见机制族评测。三种训练族为 inventory DP、routing、negotiation。Qwen3.6-35B-A3B Base 经 GRPO 34 步后，五族 agentic diagnostic mean 从 `0.204` 到 `0.815`，而 written-out mean 仅从 `0.962` 到 `0.992`；外部 BFCL V4 交互子集、TravelBench、365-day E-Commerce Bench 也有作者报告的增益。

阅读结果时要保留三个限制：生成环境的 reference replay audit 覆盖 8/11 机制族，而非全覆盖形式化证明；论文报告 one training run / one evaluation seed，E-Commerce 每臂五次运行也不足以建立普遍胜出；成本、迁移和 co-scaling 均未在本项目独立复现。Qwen3.7-Max 在论文中只是额外 setter 与 benchmark comparator，实际受训 checkpoint 是 Qwen3.6-35B-A3B；不能把这篇论文写成 Qwen3.7-Max 的内部训练 recipe，也不能默认它就是 Qwen3.7 博客所说的 `Task × Harness × Verifier` 系统。完整来源快照和实验账本见[研究笔记 §9](../../research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md#9-vhd-play-先求解机制再生成可交互环境)。
