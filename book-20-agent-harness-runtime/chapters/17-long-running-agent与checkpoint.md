# 第 17 章 Long-Running Agent：跨会话任务为什么需要 Checkpoint

## 17.1 一个五分钟做不完的任务

让 Agent “整理一个项目的依赖、修改兼容代码、运行全量测试并准备升级报告”，通常不是一次请求能完成的工作。仓库可能有几十分钟的测试，网络依赖可能中断，用户可能在中途离开，服务也可能需要滚动升级。如果系统只把所有历史放在一次对话里，任务一旦断开就只能从头开始。

Long-running Agent 的核心不是让模型永远在线，而是让任务拥有可暂停、可恢复、可审计的执行状态。Checkpoint 是这个状态的一个稳定快照：它记录任务已经完成什么、当前环境是什么、哪些动作尚未提交，以及恢复时必须重新验证什么。

## 17.2 对话历史不是任务状态

普通对话历史主要回答“用户和模型说过什么”。长任务还需要回答：哪个 workspace revision 已经产生？哪个工具 call 已经执行？测试是否通过？当前权限是否变化？模型使用的版本是什么？

因此状态可以分成四层：

1. conversation：用户消息和可见回答；
2. execution：动作、工具、错误、预算和状态机；
3. artifacts：patch、测试报告、生成文件和引用；
4. environment：仓库、数据库、浏览器和外部系统的版本。

只保存 conversation 会让恢复的 Agent 重新猜测已经发生的外部动作，这是长任务最危险的错误来源。

## 17.3 Checkpoint 的最小内容

可以把一个 checkpoint 抽象为：

```math
K_t=(I,M_t,E_t,A_t,P_t,B_t,V_t)
```

`I` 是任务身份，`M_t` 是可恢复的模型上下文，`E_t` 是环境版本，`A_t` 是已产生的 artifact，`P_t` 是待处理的副作用，`B_t` 是剩余预算，`V_t` 是版本和校验信息。

工程结构可以长这样：

```json
{
  "task_id": "task-42",
  "model_revision": "model@revision",
  "workspace_revision": "git:abc123",
  "status": "waiting_for_tests",
  "facts": [],
  "hypotheses": [],
  "tool_calls": [],
  "artifacts": [],
  "pending_effects": [],
  "budget": {"tokens": 12000, "tool_seconds": 300},
  "schema_version": 3,
  "checksum": "..."
}
```

`pending_effects` 和 `artifacts` 必须分开。已经生成的 patch 不代表已经合并；已经准备的邮件不代表已经发送。恢复系统要对待提交动作重新做权限和确认检查。

## 17.4 Checkpoint 的时机

每个 token 都保存状态不现实，只有在有意义的边界保存才有价值。常见边界包括：工具调用开始和结束、workspace 产生新 revision、测试完成、人工确认、预算档位变化和任务暂停。

对于外部副作用，最好使用两阶段记录：先写入 `intent`，执行器返回 `started`，确认完成后写入 `committed`。如果服务在中间崩溃，恢复逻辑可以查询外部系统或使用幂等键，而不是猜测动作是否发生。

## 17.5 一个跨天代码任务

第一天 Agent 发现依赖冲突，创建隔离分支并生成 patch；测试需要四十分钟，于是任务进入 `waiting`，checkpoint 保存分支 revision、已运行的测试、剩余测试和待确认升级。

第二天用户重新打开任务。runtime 先检查模型和工具版本、分支是否仍然存在、依赖锁文件是否被其他人修改，然后读取测试结果。如果 workspace revision 不一致，Agent 不能直接应用旧 patch，应重新计算 diff 或请求用户解决冲突。

如果测试进程已经在第一天启动但连接断开，系统要根据 job id 查询状态。不能因为 checkpoint 中写着“测试未完成”就再启动一份，造成重复占用资源或两个测试互相覆盖输出。

## 17.6 状态恢复的版本问题

checkpoint 绑定的不只是模型名称，还绑定 tokenizer、chat template、工具 schema、策略版本和 workspace。模型从 `revision-a` 升到 `revision-b` 后，旧的隐藏 reasoning state 或 KV cache 通常不能直接复用；可以复用经过验证的自然语言摘要和 artifact 引用，但要重新执行协议和权限检查。

恢复函数应显式检查：

```text
model revision
tokenizer/template hash
tool schema version
policy version
workspace/environment revision
checkpoint checksum
```

任何关键检查不通过，都应进入迁移、重算或人工确认，而不是静默继续。

## 17.7 预算与租约

长任务不能把一次请求的预算无限延长。每个 checkpoint 可以携带 token、工具时间、GPU 时间和外部调用次数的租约。恢复时只允许使用未过期预算，并根据任务价值重新计算。

任务还需要租约（lease）防止两个 runtime 同时恢复同一个任务。租约过期后，新 worker 可以接管，但接管前必须读取最新 checkpoint、查询工具状态并确认 workspace 锁。否则两个 Agent 会产生竞争写入。

## 17.8 评估长任务的正确性

单轮 success 不足以评估 long-running Agent。测试应主动注入进程重启、网络中断、工具超时、模型 fallback、用户修改 workspace 和权限变化。

可以定义恢复成功率：

```math
R_{\mathrm{resume}}
=\frac{N_{\mathrm{tasks\ resumed\ without\ duplicate\ effects}}}
{\max(1,N_{\mathrm{interrupted\ tasks}})}
```

此外还要记录 checkpoint 写入延迟、恢复时间、重复副作用率、状态丢失率、artifact 一致性和人工接管率。一个能从断点恢复但生成重复付款的系统不能算可靠。

## 17.9 常见失败

最常见的是只保存摘要，不保存来源和环境 revision；恢复后的模型把假设当事实。第二是 checkpoint 写入没有原子性，崩溃后出现“工具已提交但状态未记录”的不一致。第三是没有幂等 key，重试外部动作产生重复副作用。第四是版本升级后静默复用旧状态。第五是长任务没有过期和取消机制，遗留进程持续消耗资源。

## 17.10 面试回答与练习

面试官问“如何设计 long-running Agent”，可以从任务状态机、checkpoint、artifact、workspace revision、工具幂等、租约、版本校验、预算和恢复测试回答。重点不是把聊天历史存到数据库，而是让模型状态、环境状态和外部副作用在中断后仍然可判断。

练习一：为代码 Agent 设计 `waiting_for_test`、`needs_approval` 和 `committed` 三个 checkpoint。

练习二：网络断开时，如何判断一次支付工具调用是否可以重试？

练习三：模型升级后哪些状态可以复用，哪些必须重新计算？

### 17.10.1 checkpoint 的事实模型

checkpoint 不是聊天记录的定期备份，而是任务在某个一致性边界上的事实快照。它至少要有 `task_id`、`checkpoint_id`、父版本、workspace revision、模型/template revision、事实与假设、工具状态、artifact checksum、权限版本、预算和待提交副作用。

```text
checkpoint = (task, parent, state, artifacts,
              environment_revision, policy_revision,
              pending_effects, budget, checksum)
```

写入时要保证快照和索引原子可见；否则可能出现 artifact 已生成但 checkpoint 未记录，或 checkpoint 显示工具已完成而外部系统实际没有提交。

### 17.10.2 commit protocol 和幂等

外部副作用应采用准备、确认、提交或可查询的两阶段语义。工具请求带 `idempotency_key`，服务端保存请求状态；Agent 恢复时先查询，再决定是否继续。对于不支持幂等或查询的工具，默认不自动重试，转人工或要求用户确认。

checkpoint 的 `pending_effects` 要明确区分“计划执行”“已发送但未知”“已确认成功”和“已确认失败”。把这些状态都写成字符串“执行工具”，恢复时就没有安全依据。

### 17.10.3 版本升级与状态迁移

模型或 harness 升级后，可以复用文件 artifact 和已经验证的外部结果，但旧模型的未验证假设、旧 template 下的未完成 tool call 和依赖旧 schema 的 reasoning item 可能必须重新计算。迁移器应给每个字段标记兼容性，而不是整份 checkpoint 静默加载。

可以用 shadow replay 检查迁移：在不提交副作用的环境里恢复旧任务，比较新旧模型的计划、工具参数、测试和最终 artifact。出现不兼容就停止自动恢复，保留人工路径。

### 17.10.4 长任务的取消和保留

用户取消任务后，系统要停止模型、工具和后台 worker，写入最终取消 checkpoint，并标记哪些外部动作已经完成。workspace、日志和敏感 artifact 的保留时间要按租户和合规策略执行；“任务结束”不代表所有状态都可以永久保存。

长任务可靠性的工程取舍也很直接：checkpoint 越频繁，崩溃后需要重算的工作越少，但写入延迟、存储量和版本迁移压力越大；checkpoint 越稀疏，运行开销较低，却会扩大恢复窗口和重复副作用的风险。可以按外部副作用边界、任务阶段和预算动态调整频率，而不是使用一个对所有任务都相同的时间间隔。

## 17.11 checkpoint 的完整性和恢复测试

checkpoint 不只保存聊天摘要，还要保存 workspace revision、已提交 artifact、工具副作用、任务约束、预算、权限、未完成动作和下一步状态。恢复测试要比较恢复前后的事实、diff、工具调用和最终验收。

应故意在写入 checkpoint 前后宕机，验证恢复是否会重复提交、遗漏文件或把计划当成事实。checkpoint 的成功写入也应有 checksum 和原子替换语义。

## 17.12 用状态机定义“恢复后应该继续什么”

长任务最容易出错的地方，是把“下一步建议”当成“当前状态”。例如模型在文本中写下“已经提交 patch”，这可能只是计划，也可能代表执行器已经调用了版本库；如果没有状态机，恢复程序无法判断应该继续、查询还是停止。

可以把任务状态写成：

```math
S=(q,F,H,A,W,P,B,V)
```

其中，`q` 是阶段，`F` 是已验证事实，`H` 是仍未证实的假设，`A` 是 artifact 索引，`W` 是 workspace revision，`P` 是 pending effects，`B` 是剩余预算，`V` 是版本和策略信息。对话摘要只是 `S` 的一个投影，不能替代整个状态。

一个代码任务的阶段可以如下定义：

| 阶段 | 可以做什么 | 不变量 |
| --- | --- | --- |
| `running` | 读取文件、分析和生成临时 patch | 不得把计划写成已提交事实 |
| `waiting_tool` | 等待可查询的工具任务 | 只能用 job id 恢复，不可盲目重启 |
| `needs_approval` | 等待用户批准危险或外部副作用 | 未批准动作不能进入执行器 |
| `committing` | 执行带幂等键的提交 | 同一个 effect 只能有一个业务结果 |
| `unknown` | 网络断开，执行结果未确定 | 先查询外部系统，不能直接重试 |
| `completed` | 验收已通过 | 后续操作只能创建新 revision |
| `failed`/`canceled` | 任务终止或取消 | 记录原因、已发生副作用和可恢复性 |

状态转移必须由事件驱动，而不是由模型输出的某个词触发。一个最小转移集合是：

```text
running -> waiting_tool -> running
running -> needs_approval -> committing
committing -> completed
committing -> unknown -> completed | failed
running -> failed | canceled
```

`unknown` 不是 `failed` 的别名。支付、发送邮件、创建工单等动作在客户端超时后可能已经成功；把它改写为失败并重试，才会制造重复副作用。恢复程序应保留 `unknown`，使用 `effect_id` 查询外部服务，拿到确定结果后再推进状态。

## 17.13 Checkpoint 的原子性、顺序与幂等

Checkpoint 的一致性问题可以用一次提交的四个记录表示：`intent`、`started`、`observed`、`committed`。它们分别回答“准备做什么”“执行器是否接收”“看到了什么结果”“业务结果是否确认”。只存一个 `status=done` 无法表达网络断开发生在哪个边界。

对于没有外部副作用的读取操作，可以先执行再保存观察结果；对于会写入外部系统的动作，更安全的顺序是：

```text
1. 写入 intent(effect_id, request_hash, policy_revision)
2. 原子提交 checkpoint
3. 调用外部系统，并传入 effect_id
4. 保存 started 或 unknown
5. 查询并校验外部结果
6. 保存 observed/committed checkpoint
```

第 2 步和第 3 步之间崩溃时，恢复程序会看到一个未执行或未确认的 intent；第 3 步之后断电时，状态可能是 `unknown`，但由于外部请求带有同一个 `effect_id`，查询和重试不会产生第二个业务对象。幂等接口通常要满足：

```math
\mathrm{apply}(e,r)=\mathrm{apply}(e,r')
\quad\text{for repeated requests with the same effect id}
```

这里的相等是业务结果相等，不是每次返回的时间戳都相同。服务端至少应保存 `effect_id`、请求摘要、最终状态和结果引用；如果请求内容变化却复用了同一个 id，应返回冲突，而不是覆盖原请求。

代码 patch 和支付动作的要求不同。patch 可以写到临时分支，重复生成通常不会产生外部损失；支付则需要外部查询、额度锁定、授权和对账。长任务 harness 不应把所有工具都包装成同一种“可重试函数”，而要在工具 schema 中声明 `read_only`、`idempotent`、`supports_query`、`requires_approval` 等能力。

## 17.14 恢复流程：先核对事实，再让模型继续

恢复不应直接把 checkpoint 拼进新的 prompt。更可靠的流程是先由 runtime 读取事实，再由策略层决定哪些内容可以交给模型：

```text
load latest checkpoint
  -> verify checksum and parent revision
  -> compare model/template/tool/policy versions
  -> inspect workspace and artifact checksums
  -> query every pending or unknown effect
  -> renew lease and recompute remaining budget
  -> build typed recovery context
  -> run model or request human input
```

其中，`typed recovery context` 至少要把事实、假设、待确认动作和过期信息分开。一个已验证的测试结果可以作为 `fact`；模型上一轮猜测的根因只能作为 `hypothesis`；外部系统返回的“未知”应作为 `pending_effect`，不能埋在自然语言摘要里。

恢复时如果 workspace 从 `r17` 变成了 `r18`，不能把基于 `r17` 的旧 patch 无条件应用。可以先计算三方 diff，或在隔离分支重放；若用户修改了同一行、依赖锁文件或权限配置，就进入冲突状态。恢复成功的定义也不只是“模型继续生成”，而是事实没有倒退、没有重复副作用、artifact 可校验，且任务最终验收仍通过。

## 17.15 中断注入与恢复 SLO

长任务的测试要把故障放在状态边界上，而不是只在普通请求上测试超时。下面的案例覆盖了最容易隐藏一致性错误的窗口：

| 注入点 | 预期恢复动作 | 失败信号 |
| --- | --- | --- |
| intent 写入后断电 | 读取 effect id，判断是否真正执行 | 直接重复创建外部对象 |
| 工具已返回但 checkpoint 未写入 | 查询 job 或外部结果并补写观察 | 任务重新启动同一 job |
| workspace 写入一半 | 回滚临时 revision 或恢复原子文件 | 半个 patch 被当成完整 artifact |
| lease 过期 | 新 worker 读取最新版本后接管 | 两个 worker 同时提交 |
| 模型版本升级 | 迁移可兼容字段，重算不可兼容状态 | 静默复用旧 hidden state |
| 用户取消时外部动作运行中 | 停止后续动作，标记已发生副作用 | UI 显示取消但外部已提交未知动作 |

可以把恢复检查条件写成：

```math
G_{\mathrm{resume}}
=G_{\mathrm{version}}
 G_{\mathrm{workspace}}
 G_{\mathrm{effect}}
 G_{\mathrm{budget}}
 G_{\mathrm{permission}}
```

只要某一项为零，系统就应暂停、迁移或请求人工确认。在线指标除了恢复成功率，还应记录重复副作用率：

```math
D_{\mathrm{duplicate}}
=\frac{N_{\mathrm{effects\ with\ duplicate\ business\ result}}
       {\max(1,N_{\mathrm{external\ effects}})}
```

恢复 SLO 可以允许一次安全的人工接管，却不能把重复扣款、越权写入或不可逆删除当作普通失败。这样评测才会把“能继续运行”和“能安全地继续运行”区分开。

## 17.16 checkpoint 不是日志备份：它要能决定下一步

日志回答“过去发生了什么”，checkpoint 还要回答“恢复后允许发生什么”。因此 checkpoint 中应保存状态机位置、已提交 artifact、未完成工具调用、外部状态版本、权限快照、预算和幂等键。只保存完整对话会让恢复器无法区分已提交动作和模型提出但尚未执行的计划。

恢复前可以做三类检查：事实检查，确认外部状态和 artifact 仍存在；兼容检查，确认模型、工具、schema 和 workspace 版本一致；授权检查，确认用户仍允许下一步动作。任何检查失败，都进入重新读取、人工接管或安全终止，而不是盲目把历史文本重新发送给模型。

## 17.17 长任务的进度不是 token 数

一个 Agent 生成了很多 token，不代表任务推进。更有意义的进度信号是已验证的 artifact、已完成的状态转移、已减少的未决问题和仍然有效的证据。可以为任务维护：

```math
P_{\mathrm{task}}
=w_a A+w_s S+w_e E-w_r R,
```

其中 `A` 是 artifact 验证率，`S` 是状态转移完成率，`E` 是证据覆盖，`R` 是未决风险。该分数用于调度和解释，不应替代最终业务 verifier。

## 17.18 故障注入要覆盖“动作已发生但响应丢失”

最危险的中断不是模型还没调用工具，而是工具已经扣款、写库或提交代码，客户端却在返回前断线。恢复测试必须注入这一窗口，检查系统能否用幂等键查询结果、避免重复提交，并向用户明确说明结果是已确认、未知还是未执行。

## 17.19 checkpoint 的压缩和保留策略

checkpoint 不是越频繁越安全。频繁保存会增加写放大、敏感数据留存和恢复选择，保存过少则扩大故障重做范围。可以按状态转移、外部动作、artifact 提交和长时间间隔触发，并为高风险动作设置更严格的检查点。

## 17.20 恢复结果的三态语义

恢复器应把动作结果区分为 `committed`、`not_started` 和 `unknown`。`unknown` 不能被模型改写成成功或失败；系统要查询外部状态、等待回调或交给人工。只有确认未提交，才允许安全重试；只有确认已提交，才可以继续后续步骤。

## 17.21 长任务的资源治理

长期占用 workspace、KV/state、工具租约和队列 slot 会影响其他请求。调度器应为任务设置生命周期、空闲超时、最大 checkpoint 数、并发上限和优先级，并在暂停时释放可重算资源。恢复时重新申请资源和权限，不能把旧租约视为永久有效。

## 17.22 checkpoint 一致性和版本升级

模型升级、工具 schema 变化或 workspace migration 后，旧 checkpoint 可能只能用于只读分析，不能直接继续执行。恢复器应先做版本兼容和状态迁移；迁移失败时重算可重算部分，保留原 checkpoint 作为证据，而不是静默覆盖。

## 17.23 恢复 SLO 的分母

恢复成功率应以发生中断的任务为分母，并分别报告安全恢复、人工接管、重复副作用、状态泄露、恢复延迟和未决未知状态。把未发生故障的正常任务混进分母，会让恢复能力看起来虚高。

## 17.24 长周期任务的阶段性判断与资料边界

Long-running Agent 的可靠性来自状态和环境的一致性，而不是更长的 context。Checkpoint 需要保存事实、假设、工具、artifact、版本、预算和待提交副作用，并在恢复时重新校验权限和外部状态。状态机、幂等键、租约和故障注入把“恢复”从一句产品描述变成可以验收的行为。

本章的 checkpoint、幂等和租约是通用分布式系统抽象；具体 Agent 产品的持久化格式、恢复保证和跨版本能力必须以其官方文档与实现为准。
