# 第 15 章 Preserve Thinking：保留的是可继续执行的状态，不是把隐藏思维全量暴露

## 15.1 多轮工具任务为什么不能只拼聊天记录

一次复杂任务通常不是“一问一答”。模型先分析问题，提出工具调用；工具返回文件、网页或测试结果；模型根据观察修正计划；最后才给出答案或提交一个外部动作。如果每一轮都把所有历史重新序列化成普通文本，系统不仅要重复支付 token，还容易丢掉工具调用 ID、权限决定和提交状态。

因此，很多推理或 Agent API 会提供某种 preserve thinking、reasoning item 或可继续状态的能力。它关注的是下一轮如何继续执行，而不是让客户端看到模型的全部隐藏思维。把它翻译成“把思维链发出来”会同时造成安全、隐私和协议误解。

## 15.2 三种历史必须分开

第一种是可见对话历史。它包括用户消息、模型最终回答和应用允许用户看到的工具摘要。

第二种是执行状态。它包括 request id、tool call id、工具输入、授权结果、执行状态、观察结果、预算、checkpoint 和待提交副作用。这些字段通常是恢复任务真正需要的内容。

第三种是内部推理状态。它可能是 provider 管理的 reasoning token、摘要、latent state 或不可见 item。客户端能否保存、转发和复用，取决于官方协议；应用不能因为看到了某个字段名就假设它是普通文本。

可以给每个状态字段附上三个属性：谁拥有它、谁能读取它、保存多久。比如工具结果可能归属于租户，系统摘要可能只能由 runtime 读取，用户可见解释则需要经过脱敏和引用检查。

## 15.3 工具任务的提交边界

一条工具链可以画成：

```text
thinking -> tool_requested -> authorized -> executing
       -> observed -> thinking -> final
```

`tool_requested` 只是模型提出了意图，`authorized` 说明策略允许，`executing` 说明 executor 已经开始，`committed` 才表示外部副作用已经完成。恢复系统必须保存这些阶段，不能把一个未执行的计划当成已经执行，也不能把执行中的请求当成一定失败。

一个最小的工具事件应该至少带有：

```text
request_id, call_id, tool_name, input_hash,
auth_decision, started_at, finished_at,
result_hash, error, committed, trace_id
```

工具输入可以包含敏感数据，日志通常保存 hash、脱敏摘要和受控引用，而不是无条件复制完整内容。结果也应带版本或来源，方便下一轮推理判断它是否仍然有效。

## 15.4 一个超时例子

模型请求搜索工具，网络在服务端已经收到请求后断开。客户端只看到 timeout。此时“没有搜索结果”和“搜索没有命中”不是一回事：前者是未知状态，后者是一个有效观察。

如果系统只保存一句“搜索失败”，模型重连后可能把未知当成空结果，继续写出错误结论。更好的状态是保存 `call_id`、query hash、是否到达供应商、执行状态、超时类型和重试次数。重连时 runtime 先查询 call 状态；若确定未执行，可以安全重试；若可能已执行，需要使用幂等键或查询副作用，不能盲目再次发送。

这就是 preserve thinking 的工程价值：它让下一轮拥有足够的事实继续工作，而不是让下一轮拥有更多未经验证的文字。

## 15.5 为什么不能直接持久化全部隐藏思维

完整 reasoning 可能包含用户没有授权的敏感材料、系统提示片段、未验证假设和内部安全控制信息。把它原样写入日志会扩大数据泄露面，也会让调试人员把模型草稿误当成事实。

实际系统通常保存三类可审计信息：结构化执行事件、工具结果的来源和 hash、面向用户的结论摘要。若确实需要保存 reasoning item，应由 provider 协议和租户策略明确允许，并设置访问控制、加密、保留期限和删除流程。

摘要也不是天然安全。过度压缩可能丢掉“只读”“不能修改生产”的约束，把猜测写成事实，或删除证据来源。一个有用的摘要要带 source reference、生成时间、模型 revision 和验证状态。例如“猜测配置 X 导致问题”应标记为 hypothesis，而不是记录成 root cause。

## 15.6 热状态、冷状态和恢复校验

可以把状态分成热状态和冷状态。热状态保留在 runtime 内存或低延迟缓存中，支持短时间内继续；冷状态写入 checkpoint、workspace 或事件存储，支持进程重启和长时间恢复。

状态演化可抽象为：

```math
s_{t+1}=F(s_t,o_t,a_t;\theta)
```

恢复时读取 checkpoint：

```math
\hat s_t=\mathrm{Restore}(\mathrm{checkpoint}_t)
```

`Restore` 不应只是反序列化。它要检查模型 revision、tokenizer 和 chat template、工具 schema、权限版本、上下文折叠策略、外部资源版本以及 state checksum。内部 latent state 或 KV cache 通常强绑定模型和 runtime，不能在版本升级后默认复用。

## 15.7 上下文折叠时保留什么

当对话接近 context limit，系统会压缩旧历史。保留一段自然语言摘要比直接删除更好，但摘要必须区分事实、假设、未完成动作和已提交副作用。

例如，下面两条记录看似类似，恢复含义完全不同：

```text
已检查部署配置，推测 timeout 可能导致失败，尚未运行测试。
```

```text
已将 timeout 改为 30s，并完成生产提交。
```

第二条包含外部副作用，必须带提交人、授权和变更 ID。上下文折叠不能把它们压成同一句“timeout 已处理”。对于每个结论，最好保留来源文档或 artifact 引用，以便恢复后的模型重新核对。

## 15.8 安全边界与状态泄露

工具结果本身可能包含不可信指令；内部摘要可能泄露 system prompt；用户可见的错误信息可能被再次注入。状态存储要做租户隔离、加密、访问审计、过期删除和字段级脱敏。

最重要的边界是：模型生成的计划不是授权，状态中出现的动作也不是执行命令。恢复时每一个外部副作用都必须重新经过 auth 和 executor。权限在任务中途发生变化时，旧状态不能自动继承旧授权。

## 15.9 如何评估 preserve thinking

评估不能只测正常多轮对话，要主动中断任务：客户端断开、工具超时、进程重启、模型 fallback、上下文折叠、权限变化和 checkpoint 跨版本恢复。

正确恢复率可以写成：

```math
R_{\mathrm{resume}}
=\frac{N_{\mathrm{tasks\ resumed\ correctly}}}
{\max(1,N_{\mathrm{interrupted\ tasks}})}
```

同时报告重复副作用率、状态泄露率、恢复延迟、摘要事实支持率和跨版本拒绝率。一个系统即使恢复率高，只要它在网络不确定时重复支付或重复发送外部请求，仍然不适合生产。

## 15.10 一个 checkpoint 的教学结构

可以用以下结构描述浏览器或代码 Agent 的 checkpoint：

```json
{
  "task_id": "...",
  "model_revision": "...",
  "workspace_revision": "...",
  "summary": {"facts": [], "hypotheses": [], "open_questions": []},
  "tool_calls": [],
  "artifacts": [],
  "pending_effects": [],
  "budget": {"reasoning": 0, "tools": 0, "deadline": "..."},
  "policy_version": "...",
  "checksum": "..."
}
```

这里的 `pending_effects` 故意和 `artifacts` 分开。artifact 是已经生成的产物，pending effect 是尚未提交的动作。恢复系统在提交前必须再次显示或确认高风险动作，并检查 workspace revision 没有被其他任务覆盖。

## 15.11 常见误区

把 preserve thinking 当成暴露 CoT，是第一个误区。可继续执行的结构化状态和用户可见解释应有不同的权限。

第二个误区是认为保存文本就能恢复。工具 ID、外部状态、权限、版本和幂等性同样重要。

第三个误区是把断线等同于工具没有执行。必须查询执行状态或使用幂等设计。

第四个误区是摘要越短越好。摘要丢掉约束和来源后，恢复任务会变得更危险。

## 15.12 面试回答、练习与边界

回答“多轮工具 Agent 如何保留推理状态”时，应先区分可见历史、结构化执行状态和 provider 管理的内部状态；保存 call ID、观察、权限、预算、checkpoint、artifact 和来源；恢复前校验模型与工具版本，处理外部副作用的幂等性，不能把隐藏思维或未授权计划直接暴露或执行。

练习一：为一个代码 Agent 设计 checkpoint schema，覆盖当前分支、测试结果、待确认修改和工具状态。

练习二：设计一个搜索超时的状态机，分别处理“未发送”“执行中”“成功”“失败”“未知”五种状态。

练习三：列出上下文折叠时必须保留的五类信息，并说明删除哪一类会产生什么风险。

### 15.12.1 三种状态必须分开保存

可见对话是用户和模型交换的文本；执行状态是任务阶段、工具调用、验证结果和待确认动作；环境状态是 workspace、数据库、浏览器和外部服务的真实状态。把三者全部序列化成聊天历史，会让恢复过程无法判断哪些是模型的说法、哪些是已验证事实。

可以用如下结构表达一条状态记录：

```text
state = {
  facts: [{value, source, revision}],
  hypotheses: [{text, status, evidence_ids}],
  actions: [{call_id, args_hash, status, idempotency_key}],
  artifacts: [{uri, checksum, workspace_revision}],
  policy: {principal_revision, policy_revision},
  budget: {used, remaining, deadline}
}
```

恢复时，`facts` 需要按来源复核，`hypotheses` 可以重新评估，`actions` 要查询外部状态，`artifacts` 要校验 checksum，`policy` 要重新读取授权。这样才不会把一段旧摘要当成新的事实。

### 15.12.2 preserve thinking 不等于暴露隐藏思维

系统需要保存的是让任务可继续、可审计、可恢复的结构化状态，而不是把所有内部推理文本原样暴露给用户或下游模型。可保留问题分解、证据 ID、工具结果、失败原因、验证状态和待确认动作；敏感的内部 token、凭证和不必要的私人内容应受访问控制。

如果产品需要给用户解释，应生成面向用户的简洁依据和结果摘要，并把详细 trace 放在受权限保护的审计系统中。可解释性要求来源和决策理由可查，不要求把内部推理过程当作 API 合同。

这种保留方式存在清晰的工程取舍：保存更多结构化证据、版本和工具状态，能够提高恢复率与审计能力，却会增加存储、加密、脱敏和访问控制成本；保存得过少则节省空间，但恢复时更容易把旧假设误当成事实。实际系统应按风险和恢复价值决定保留粒度，而不是追求“全部保存”或“全部删除”。

### 15.12.3 中断恢复的 worked example

代码 Agent 在运行测试时网络断开。checkpoint 显示测试请求已发送但没有响应，状态不能直接记为失败。恢复后先用 `run_id` 查询测试服务；如果已完成，读取结果并继续；如果不存在且动作可幂等，才重新运行；如果无法查询，转为未知并请求人工。

同样的规则适用于支付、发邮件和发布。外部副作用的“未知”是一个真实状态，不是普通异常。把未知状态直接当失败并重试，是重复副作用的主要来源之一。

## 15.13 用状态机和幂等键恢复任务

恢复任务的核心不是把上一轮文字重新发给模型，而是确定每个动作处于什么状态。一个工具调用至少要区分 `planned`、`authorized`、`sent`、`running`、`succeeded`、`failed` 和 `unknown`。网络断开时，`sent` 和 `running` 不能直接改成 `failed`，因为外部服务可能已经执行成功。

可以用幂等键和状态查询处理这种不确定性：

```text
create action(call_id, idempotency_key)
  -> authorize
  -> send once
  -> query status after timeout
  -> commit result or mark unknown
```

恢复规则应按动作类型决定。读取文件可以安全重试；创建订单、发送邮件和发布配置必须先查询 `idempotency_key`；无法查询的高风险动作应转人工，而不是让模型“猜测已经失败”。这也是为什么 preserve thinking 更接近工作流 checkpoint，而不是聊天记录功能。

若将恢复正确性记为 `P_recover`，可以把它拆成：

```math
P_{\mathrm{recover}}
=P_{\mathrm{state\_valid}}
 P_{\mathrm{external\_status\_known}}
 P_{\mathrm{policy\_rechecked}}
 P_{\mathrm{no\_duplicate\_effect}}
```

任一项为零，恢复后的“继续执行”都不应被视为成功。这个乘积是教学上的分解，不意味着各项真正独立，但它能提醒工程师不要只用“模型上下文恢复成功”来定义可靠性。

## 15.14 Checkpoint 的快照、日志与版本

只保存最新快照会丢掉故障前最后一个事件；只保存事件日志则恢复时需要重放很长历史。常见做法是周期性 snapshot 加短的 append-only event log：快照保存可快速加载的状态，事件日志保存快照之后的动作和观察。

```text
snapshot S_k
  + event e_(k+1)
  + event e_(k+2)
  + ...
  -> replayed state S_(k+m)
```

每个事件应带 `trace_id`、`call_id`、schema version、model revision、workspace revision、输入 hash、输出来源和时间。状态 schema 变化时，使用显式 migration；不要让新 runtime 默默解释旧字段。若 `S_k` 的 workspace revision 与当前工作区不同，恢复流程应先重新读取冲突文件，再决定重放或生成 patch。

对安全和隐私，事件日志可以保存脱敏参数、hash 和受控 artifact 引用，而不是无条件保存全部 prompt、凭证或工具结果。审计要求“能证明发生了什么”，不等于“所有原始数据永久保留”。保留期限、租户隔离、加密和删除流程都是状态契约的一部分。

## 15.15 上下文折叠时如何保留证据链

当历史接近 context limit，最危险的做法是让模型把一切压成一句“任务进行中”。折叠后的状态至少要区分四类内容：已验证事实、尚未证实的假设、已执行动作和待执行动作。每条事实应有来源、版本和时间；每个假设应有支持或反驳它的 evidence id。

例如：

```text
facts:       config revision r17 contains timeout=30
hypotheses:  timeout causes retry storm [supported_by: e12]
executed:    read logs with call_id c4
pending:     run read-only reproduction on revision r17
```

折叠不能把“模型认为已完成”改写成“系统已完成”。如果外部状态没有验证，摘要中应保留 `unknown`；如果工具结果来自网页或用户上传文件，还要标出不可信来源，避免下一轮把其中的指令当成系统规则。对需要引用的研究任务，摘要还应保留 URL、文档版本、段落定位或内容 hash。

## 15.16 恢复系统的 SLO 与演练

preserve thinking 不应只在 demo 中演示“断线后还能继续”。生产评估至少包含恢复成功率、恢复延迟、重复副作用率、状态泄露率、版本不兼容拒绝率和人工接管率。要注入模型服务超时、工具执行超时、进程重启、workspace 冲突、权限撤销和上下文折叠，观察系统是否按状态机处理。

一个合格的演练案例是：代码 Agent 已写入 patch，测试任务返回未知，用户随后撤销写权限。恢复时系统应查询测试状态，重新读取授权，禁止新的写操作，并把已有 patch 作为待审核 artifact 展示。若系统只把历史文本交给模型，模型可能在权限已撤销后再次尝试写入。

## 15.17 preserved state 的最小数据结构

保存“继续思考”不能只保存一段可见文本。一个可恢复状态至少要区分：已提交的输出、未提交的候选、模型和 tokenizer 版本、工具事件、外部 observation、权限快照、预算、artifact 引用和校验值。对每个字段还要标注来源是模型生成、用户输入、工具返回还是系统事实。

可以用如下抽象描述状态：

```math
S=(M,T,P,O,A,B,E,H),
```

其中 `M` 是模型版本，`T` 是 tokenizer/template，`P` 是 committed prefix，`O` 是 observations，`A` 是 artifacts，`B` 是 budget，`E` 是 external effects，`H` 是权限和审计信息。恢复前先验证兼容性；任何一项不匹配，都应重算或人工接管，不能把旧状态当作普通 prompt 继续喂给新模型。

## 15.18 可见思考和内部状态的边界

用户可能需要知道系统为什么暂停、用了哪些证据和下一步是什么，但这不等于系统应泄露内部 reasoning token。工程上可以保存结构化的决策摘要、引用、工具结果和失败原因，而把内部推理表示视为受保护状态。摘要也必须标记未知和待验证，不能把模型的假设写成事实。

这一边界对安全尤其重要：如果恢复摘要包含系统秘密、隐藏策略或其他租户的上下文，持久化本身就造成泄露；如果摘要过度压缩，恢复后又可能重复动作。最小必要、分层权限和字段级加密比“把整段上下文全存下来”更可靠。

## 15.19 恢复正确性的反事实测试

把同一任务分别在未中断、工具前中断、工具后中断、权限撤销后中断和模型升级后中断。比较最终 artifact、工具副作用、事件顺序和用户可见状态。理想结果不是每个断点都得到完全相同的 token，而是所有可接受路径都不重复提交、不越权，并能解释差异来自哪里。

如果恢复后的答案更长但事实一致，可以接受；如果输出看似相同却多发了一次邮件，则是严重失败。恢复测试要把业务状态和文本结果同时纳入断言。

## 15.20 状态恢复的 worked example

假设 Agent 已读取三份文档、生成 patch、调用测试工具，测试返回超时，随后用户撤销写权限。恢复时应先查询测试是否实际完成，再重新读取权限和 workspace revision；如果测试未知，只能把 patch 标为待验证，不能再次写文件。模型可以继续分析已有 artifact，但不能继续执行已撤销的写操作。

## 15.21 状态快照的安全保留

保留期限应按字段区分：业务 artifact、审计元数据、工具原文、内部状态和临时候选有不同的删除要求。加密、访问控制、租户隔离和可验证删除必须进入 checkpoint 设计；为了“以后恢复方便”无限期保存全部思考内容会扩大泄露面。

## 15.22 小结与资料边界

本章参考 reasoning API、tool calling、Agent tracing、checkpoint 和状态机工程资料。具体 reasoning item 的可见性、加密状态和跨请求复用能力以对应 provider 的版本化协议为准。
