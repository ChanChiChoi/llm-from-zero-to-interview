# 第 19 章 Interleaved Thinking：工具观察如何改变推理轨迹

## 19.1 从一次性生成到交错循环

设想两个任务。

第一个任务是回答“什么是二分查找”。模型可以在没有外部信息的情况下直接组织答案。第二个任务是修复一个只在高并发时出现的缓存故障。模型必须先读取日志，再查看当前代码和部署版本，提出一个假设，运行复现测试，看到测试结果后修改假设，最后决定是否生成补丁。第二个任务如果被压缩成“一次生成完整计划，最后统一执行”，计划中的很多事实在生成时其实还不存在。

Interleaved thinking 指的是把推理、动作和观察交替放入一个闭环：模型形成当前判断，提出一个动作；执行器完成或拒绝动作；系统把结果作为带结构的 observation 返回；模型根据新信息更新状态，再决定下一步。ReAct 论文把 reasoning trace 与 action 交错起来，强调动作可以取得外部信息、修正计划并处理异常；Toolformer 则研究模型如何决定是否调用工具、传递什么参数以及如何把结果纳入后续生成。两者都说明，工具调用不是答案末尾的一种装饰格式，而是推理过程中的信息获取动作。

这里的“thinking”需要一个边界。它表示系统内部可记录的决策状态、计划片段、工具请求和验证结果，不等于模型的全部内部计算，也不意味着把隐藏推理 token 原样暴露给用户。工程系统真正需要审计的是：模型提出了什么动作，策略是否批准，工具是否启动，观察来自哪里，哪一个版本被验证，以及外部副作用是否已经提交。

一次交错循环可以先用下面的形式表示：

~~~text
observe state
    -> reason about the next step
    -> propose an action
    -> authorize the action
    -> execute the action
    -> receive an observation
    -> update state
    -> repeat or finish
~~~

它和“先规划、后执行”并不是非此即彼的关系。稳定的读取、解析、测试和审批步骤可以由固定工作流承担；只有那些会被外部反馈改变的节点，才需要重新交错推理。这样既保留适应性，也避免把所有简单步骤都交给一个不断循环的语言模型。

## 19.2 状态转移：模型究竟根据什么改变下一步

如果只把上一轮工具结果拼接为一段文本，下一轮模型很难区分事实、假设、过期结果和未完成动作。更可靠的做法是显式维护状态。设第 t 轮状态为：

~~~math
s_t=(x_t,F_t,H_t,P_t,B_t,R_t)
~~~

其中：

- x_t 是任务目标和当前用户约束；
- F_t 是已经确认的事实集合；
- H_t 是仍然存在的候选假设及其证据；
- P_t 是待执行动作和它们的依赖；
- B_t 是剩余的模型、工具、时间和副作用预算；
- R_t 是权限、代码、数据和工具的版本信息。

模型根据状态选择动作 a_t，执行器返回观察 o_{t+1}，系统再计算：

~~~math
s_{t+1}=U(s_t,a_t,o_{t+1})
~~~

U 不是一个只能由模型自由发挥的自然语言函数。它应当包含 schema 校验、权限检查、版本检查和状态机约束。例如，tool_started 事件不能把一个没有授权的工具调用变成“已执行”；一个建立在 repo@r41 上的测试报告也不能自动证明 repo@r42 上的补丁正确。

观察至少要区分四种情况：

| 观察类型 | 含义 | 下一步能否直接使用 |
| --- | --- | --- |
| data | 工具成功返回并通过结构校验的数据 | 可以，但仍需考虑来源和新鲜度 |
| empty | 工具成功执行，但查询没有结果 | 可以作为“未找到”的证据，不能当作工具失败 |
| error | 工具明确报告失败，失败原因已知 | 只能按错误类型决定修复或重试 |
| unknown | 请求可能已经发生，但客户端不知道最终状态 | 先查询或人工处理，不能直接重试写操作 |

empty 和 unknown 的区别很重要。搜索没有命中是一个有意义的观察；支付请求在连接断开后不知道是否扣款，则不是“没有扣款”。如果系统把所有非成功结果都压成 false，模型就会在真正需要查询外部状态的地方错误重试。

## 19.3 什么时候交错推理值得付出成本

交错循环的价值来自外部反馈改变决策，而不是来自循环次数本身。下面三类任务通常值得使用它。

第一类是事实会随环境变化的任务，例如读取线上日志、浏览当前网页、查询库存或检查当前代码版本。模型的先验知识不能替代实时观察。

第二类是错误可以由执行暴露的任务，例如代码编译、SQL 查询、数学程序验证和 API 调试。模型无法仅凭自然语言可靠预测每一个运行时约束，执行器提供的错误信息会改变候选方案。

第三类是动作有依赖和异常分支的任务，例如“读取配置、修改文件、测试、提交补丁”。下一步是否存在，取决于上一步的返回值；固定计划可以给出主路径，但不能预先知道所有分支。

交错推理也有明确的反模式。若每一步都已知、工具延迟远高于模型计算、没有可靠 verifier，或者动作本身不可逆且无法获得状态查询，那么增加循环只会增加等待和风险。此时固定 workflow、批量读取或先生成候选再统一验证，可能更合适。

可以把选择写成一个工程问题：新增一轮工具调用是否带来足以改变决策的信息。若答案是否定的，继续生成更长的 reasoning trace 并不会创造新事实。

## 19.4 工具调用的生命周期

工具调用不是“模型输出了一段 JSON”就完成了。一个可审计的生命周期至少包括以下状态：

| 状态 | 含义 | 是否可能产生外部副作用 | 恢复原则 |
| --- | --- | --- | --- |
| proposed | 模型提出了动作和参数 | 否 | 可以修改、拒绝或取消 |
| authorized | 策略和必要的用户确认已经通过 | 否，执行尚未开始 | 检查版本后再启动 |
| started | 执行器已经接受并开始处理 | 可能 | 断线后视为状态未知，先查询 |
| returned | 工具返回了结果，但结果还未验证 | 可能已经发生 | 校验 schema、hash 和版本 |
| verified | 结果通过了指定的验证条件 | 可能已经发生 | 可以进入业务提交或下一步 |
| committed | 业务上已经确认本次动作的结果 | 是 | 不得盲目再次提交 |
| failed | 已知失败，且失败语义明确 | 未发生或已被工具确认回滚 | 依据错误类型决定是否重试 |
| unknown | 无法确认执行结果 | 不确定 | 查询、补偿或转人工 |
| cancelled | 在执行开始前被取消 | 否 | 不能再由旧调用启动 |

returned 不等于 verified。例如，邮件服务返回 HTTP 200 只能说明服务接受了请求；它不一定说明收件人收到邮件。代码测试工具返回“命令退出 0”也不一定覆盖隐藏测试。系统应明确每一种动作的验证条件，而不是把工具自报的状态当作最终事实。

状态迁移还应有不可违反的约束：

~~~text
proposed -> authorized -> started -> returned -> verified -> committed
       \-> cancelled
started -> failed
started -> unknown
~~~

如果一个调用从 proposed 直接跳到 started，记录中就无法证明权限检查发生过；如果一个调用从 unknown 直接跳到 committed，系统就可能在不知道第一次请求结果的情况下重复产生副作用。状态机不是为了让日志看起来整齐，而是为了阻止这些含义不清的迁移。

## 19.5 事件日志比最终文本更接近事实

流式 Agent 需要同时输出模型可见内容和协议事件。下面是一个最小事件结构：

~~~json
{
  "seq": 17,
  "trace_id": "trace-42",
  "turn_id": "turn-3",
  "type": "tool_started",
  "call_id": "call-8",
  "tool": "run_test",
  "state_version": "repo@r42",
  "idempotency_key": "test:repo@r42:case-17",
  "payload": {"command": "python -m pytest tests/test_parser.py"},
  "created_at": "2026-08-14T10:30:00Z"
}
~~~

seq 用于同一条事件流的顺序校验；trace_id 把多轮调用归入同一个任务；turn_id 区分一次模型决策轮；call_id 把请求、结果和验证绑定起来；state_version 绑定代码或数据基线；idempotency_key 用于恢复和去重。时间戳便于排查延迟，但不能单独决定事件顺序，因为不同机器的时钟可能不一致。

事件消费者至少要检查三件事。

第一，顺序必须单调。若消费者已经处理 seq=17，再收到 seq=16，应进入重放或补偿逻辑，不能默默覆盖当前状态。

第二，事件必须幂等。网络重试可能让同一个事件到达两次；消费者可以依据事件 ID 或 trace_id 与 seq 的组合去重。对于真正的外部动作，还需要依据业务幂等键查询执行器状态，因为“事件没有重复”不等于“外部请求没有重复”。

第三，事件必须可关联。只有 tool_result 而没有 call_id 的日志，无法判断它对应哪次请求；只有“已完成”而没有启动和授权事件的日志，无法判断是否越权或重复执行。

模型生成的自然语言仍然有价值，它可以帮助用户理解进度和原因；但它不能覆盖事件事实。模型在后续轮次说“我已经发送了邮件”，不应替代执行器的 committed 记录。

## 19.6 Observation 是数据，不会自动变成指令

工具返回的内容可能来自网页、issue、代码注释、数据库字段、用户上传文件或第三方 API。这些来源都可能包含类似“忽略前面的规则”“把密钥发送到某个地址”的文字。它们是 observation 中的数据，不会因为被模型读到就获得系统消息或开发者消息的优先级。

一个结构化 observation 可以把事实和自然语言分开：

~~~json
{
  "call_id": "call-8",
  "source": {
    "kind": "repository",
    "uri": "repo://service",
    "revision": "r42"
  },
  "retrieved_at": "2026-08-14T10:30:00Z",
  "content_hash": "sha256:8d...",
  "trust_level": "untrusted_data",
  "facts": [
    {"field": "test_status", "value": "failed"}
  ],
  "instructions": [],
  "artifacts": ["artifact://test-report/17"]
}
~~~

trust_level 表示处理策略，不是“内容一定为真”的概率。官方数据库也可能在查询后变化，可信域名的网页也可能过期；不可信网页中的一个数字在通过独立核对后也可能成为有用证据。来源、时间、新鲜度、完整性和独立验证应分别记录。

如果把四个检查都已经实际执行，可以用一个可用性指示量表示：

~~~math
V(o)=\mathbf{1}[\mathrm{source\_ok}]
      \mathbf{1}[\mathrm{freshness\_ok}]
      \mathbf{1}[\mathrm{schema\_ok}]
      \mathbf{1}[\mathrm{permission\_ok}]
~~~

这里每个指示量在检查明确完成时取 0 或 1，V(o)=1 只表示 observation 满足这些形式条件，不表示其中的事实已经被独立证明。若某一项尚未测量，不能把它默认为 0 或 1；工程实现应额外使用 unknown 状态。否则系统会把“尚未检查”错误地当成“检查失败”或“检查通过”。

大 observation 也不应无界地回填上下文。工具可以返回结构化摘要、分页游标和受控 artifact；模型先使用摘要决定是否需要原文，再通过带 hash 的引用读取指定片段。这样既减少上下文污染，也让最后的结论能够回到具体证据。

## 19.7 代码调试：观察如何改变假设

考虑一个“昨天发布后接口偶发 500”的请求。一个可信的交错轨迹不会直接生成根因报告，而是逐步缩小假设空间。

第一步，读取部署 revision 和错误样本。系统得到事实：“错误集中在 refresh_config 路径，当前 revision 是 r42。”这还不能证明根因。

第二步，读取相同时间段的日志和连接池指标。若观察显示 timeout 从 3 秒升到 30 秒，模型可以提出“连接池耗尽是候选原因”，但仍然只是假设。

第三步，在以 r42 为基线的隔离 workspace 中读取刷新代码和历史 diff。若代码在刷新期间持有锁，并且连接池调用发生在锁内，假设得到更多支持；如果两条证据来自同一个缓存日志，则支持来源数仍然只有一个。

第四步，生成候选 patch，并记录 patch 的基线、修改文件、命令和 diff hash。patch 是 artifact，不是已经提交的生产动作。

第五步，运行公开单元测试和并发复现。公开测试通过只能证明这些测试通过；复现仍失败时，模型必须保留失败观察，不能把“测试通过”写成“问题解决”。

第六步，如果所有验收条件满足，再由独立策略决定是否允许合并或部署。部署权限不应因为模型在前几轮“想得很确定”而自动获得。

可以用事实与假设表记录每轮变化：

| 内容 | 类型 | 证据 | 当前状态 |
| --- | --- | --- | --- |
| timeout=30s | 事实 | 日志 artifact l17 | 已确认 |
| 连接池耗尽导致 500 | 假设 | 指标和调用链 | 待复现 |
| 将调用移出锁范围 | 候选动作 | patch p3 | 待测试 |
| 并发测试 100 次无失败 | 验证结果 | test report t9 | 仅对该测试集成立 |

如果测试工具在启动后断线，结果不是“测试失败”，而是 unknown。恢复器应查询测试执行器；如果无法查询，就报告“未确认”，而不是自动合并 patch。

## 19.8 研究 Agent：证据版本同样重要

研究型任务的外部动作通常是搜索、读取论文、比较实验和生成引用。它没有支付那样明显的副作用，却同样需要状态和来源。

例如，用户询问某方法是否优于基线。Agent 可以先搜索题目和作者，观察结果可能发现预印本、会议版本和项目页同时存在。下一步应读取正式论文的实验设置，确认数据集版本、评价指标和基线实现；如果数字来自项目页而不是论文正文，引用时应明确来源层级。

一条可靠证据至少包含 URL 或文献标识、版本或提交号、获取时间、引用片段和内容 hash。论文摘要可以支持“作者声称研究了什么”，通常不能单独支持“在所有场景下都更好”。博客和二手报道可以用于发现线索，但不能在没有回到原文的情况下承担关键结论。

研究 Agent 也会遇到空结果、过期结果和结果不可比：

- 搜索没有命中，不等于该方法不存在；可能是题目、作者或版本写法不同。
- 论文网页打不开，不等于论文没有实验；这是工具错误或信息未知。
- 两个数字都叫 accuracy，不等于数据集、样本筛选和解码设置相同。
- 一个引用存在，不等于它支持当前句子的全部范围。

因此，交错研究流程应把“找到来源”“读取来源”“解释来源”和“核对结论”分成不同事件。最终答案中的每一个关键断言，都应能回指到至少一个相应 artifact；无法支持的部分应写成限制或未知。

## 19.9 信息增益、预算与继续条件

交错循环不能只用“模型还想继续”作为继续条件。设当前关于任务的候选假设空间为 \mathcal{H}_t，模型对这些假设有一个归一化概率分布 p_t(h)，则不确定性可以写成熵：

~~~math
H_t=-\sum_{h\in\mathcal{H}_t}p_t(h)\log p_t(h)
~~~

要求 \mathcal{H}_t 是有限或可计算的假设集合，p_t(h)\ge0，且总和为 1；当某个概率为 0 时，约定 0\log0=0。一次观察带来的理想信息增益为：

~~~math
I_t=H_t-H_{t+1}
~~~

真实 Agent 通常没有可靠的概率分布，因此不能把模型自报 confidence 直接当作 p_t。工程上可以使用候选数量变化、独立 verifier 分数、证据覆盖或不确定性标注作为 proxy，但应把它们称为近似指标。若连续若干轮没有新增来源、没有改变候选、没有改善 verifier，也没有修复状态问题，继续调用通常只是在重复成本。

预算至少应拆成向量，而不是把不同单位直接相加：

~~~math
\mathbf{B}_t=(B_{\mathrm{model}},B_{\mathrm{tool}},B_{\mathrm{time}},B_{\mathrm{verify}},B_{\mathrm{side}})
~~~

其中各分量分别是剩余模型 token 或计算量、工具调用或工具时间、wall-clock、验证资源和外部副作用次数。B_side=0 明确表示当前阶段不能产生外部副作用。只有在定义了换算价格后，才可以把这些量折算为统一成本；不能把 1000 token、2 秒和一次发信未经说明地相加。

继续一轮动作至少要满足：预算足够、权限仍有效、输入版本没有失效、动作有合理的信息或业务价值，并且风险处于允许范围。如果任何条件未知，状态应是“无法判断”，而不是默认允许。

## 19.10 停止、重试与 unknown

一个健壮的循环需要显式停止条件：

1. 目标已经满足定义好的验证条件；
2. 达到最大工具轮次、模型预算、时间或副作用预算；
3. 连续调用相同工具和相同参数，却没有新增观察；
4. 当前 verifier 分数没有变化，新增计算无法改变候选排序；
5. 权限、代码 revision 或数据版本发生变化，需要重新读取；
6. 用户取消任务；
7. 外部执行状态未知，继续重试可能重复动作。

停止不一定表示成功。系统应分别报告 completed、failed、abstained、cancelled 和 unknown，并保留停止原因。尤其不能把“预算耗尽”写成“任务失败”再据此训练模型，也不能把“工具超时”写成“工具返回空结果”。这些状态的后续动作完全不同。

重试策略取决于动作类型。只读、确定无副作用且参数相同的请求通常可以重试；计算型工具可以依据执行器状态和幂等键重试；写数据库、发邮件、支付和部署动作必须先查询外部状态。若执行器支持幂等键，应让相同意图的重试返回原操作结果，而不是重新创建操作。

“取消”也需要语义。发送取消请求之前尚未启动的调用可以变成 cancelled；已经 started 的调用可能已经产生副作用，取消成功与否必须由执行器确认。客户端断开连接不等于用户取消，服务端应根据产品约定决定继续后台执行、暂停等待恢复，还是主动取消。

## 19.11 两阶段提交与外部副作用

把模型输出分成草稿和动作请求，是保护系统的第一步。高风险操作可以采用应用层的两阶段流程：

~~~text
proposal
    -> policy check
    -> user or automatic approval
    -> execute with idempotency key
    -> query result
    -> verify
    -> commit
~~~

这里的“应用层两阶段”不要和数据库理论中的严格 two-phase commit 混为一谈。它的重点是把“模型建议动作”和“执行外部副作用”隔开，并为恢复、查询和人工确认留下位置；它不能自动让跨系统操作具备原子性。

一个动作请求至少需要目标、参数、影响范围、风险级别、证据、授权版本和幂等键。幂等键 k 应绑定同一个业务意图，例如规范化后的工具名、目标资源、参数和任务版本的摘要：

~~~math
k=\mathrm{Hash}(\mathrm{tool},\mathrm{target},\mathrm{normalized\_args},\mathrm{intent\_revision})
~~~

哈希函数、参数规范化规则和意图版本必须稳定；不要把模型自然语言解释直接当成键，也不要让两个不同业务意图错误共享同一个键。对于同一个 k，执行器应保证重复请求返回同一操作记录，或者明确告诉调用方状态未知。超时后的正确顺序是：查询幂等键、查询外部资源状态、判断是否已经完成，再决定等待、补偿、重试或人工接管。直接重发支付请求是最危险的恢复方式之一。

多步骤工作流可以使用 saga 思路，为已经完成的步骤定义补偿动作。例如创建草稿、锁定库存、发送通知分别可能有取消草稿、释放库存和发送更正通知的补偿。但补偿不是时间倒流：已经发出的邮件、已经被读取的秘密或已经产生的外部通知无法完全恢复成“从未发生过”。用户界面和审计记录必须保留真实状态。

## 19.12 Checkpoint 与恢复

长任务的 checkpoint 不能只保存最近一段模型文本。至少要保存：

- 当前状态和状态 schema 版本；
- 最后确认的事件序号；
- 未完成工具调用、幂等键和执行器查询地址；
- 已提交副作用及其外部记录号；
- 待验证 artifact、代码或数据基线；
- 剩余预算、权限版本和用户取消状态；
- 最近一次 observation 的来源、hash 和有效期。

恢复前可以把条件写成：

~~~math
\mathrm{resume}=1
\iff
\mathrm{log\_valid}
\land\mathrm{schema\_compatible}
\land\mathrm{policy\_valid}
\land\mathrm{side\_effects\_reconciled}
~~~

四个条件都必须已被检查。log_valid 表示事件没有丢失或乱序；schema_compatible 表示恢复器理解旧状态；policy_valid 表示权限和策略仍适用；side_effects_reconciled 表示所有可能已经执行的动作都有明确状态。任何条件未知，都不应自动继续。

恢复成功只表示“可以安全地继续这个状态机”，不表示业务目标已经完成。比如一个支付请求已被确认扣款，但后续开票步骤失败，恢复器应从开票步骤继续，而不是重新支付。

版本变化同样会使 checkpoint 失效。用户在 Agent 运行期间修改了代码，旧 patch 的 base_revision 就不再适用于当前 workspace；检索索引更新后，旧 observation 也可能需要重新验证。恢复器应重新读取变化部分，并把受影响 artifact 标记为待验证。

## 19.13 流式输出与幂等消费

流式响应会让客户端尽早看到进度，但进度文本不能被当成业务提交。可以把可见事件分成三层：

1. draft：模型正在形成计划或参数，用户可以看到，但没有外部副作用；
2. execution：工具已经按权限启动，客户端应显示处理中，而不是显示已完成；
3. commit：执行器和验证器确认了业务结果，客户端才可以显示完成。

客户端可能在任意一个事件之间断线。恢复时要根据最后一个已确认的 seq 请求重放事件；重放必须是幂等的。对只读进度事件，重复显示通常只是界面问题；对写操作，重复消费 action_committed 可能导致重复业务更新，因此消费者应在写入自己的状态前记录去重键，并对外部资源使用业务幂等键。

事件重放还要处理版本兼容。旧客户端可能不认识新的事件类型，协议可以提供向后兼容的摘要；但摘要不能丢掉“是否已启动”“是否已提交”和“是否未知”这些安全语义。把未知事件静默当作成功，会破坏恢复正确性。

协议层也应区分客户端取消、服务端超时和执行器失败。它们可能展示相同的错误页面，却对应不同的补偿路径：客户端取消可能需要停止生成，服务端超时需要查询任务，执行器失败可能可以重试。状态字段不能只写一个自由文本 message。

## 19.14 如何评估交错 Agent

评估不能只放一个“工具正常返回”的 happy path。测试集至少包含：空结果、部分返回、参数 schema 错误、权限拒绝、工具超时、页面变化、数据版本变化、用户取消、模型重试、执行状态未知、提示注入和重复事件重放。

对每个任务记录以下结果：

- 最终任务成功或失败，以及成功判据；
- 有效工具调用和无效工具调用；
- 工具轮次、模型 token、工具等待、验证时间和人工等待；
- unknown 状态的比例及恢复结果；
- 重复副作用次数和未授权动作次数；
- 证据覆盖、引用正确性或代码测试通过情况；
- p50、p95、p99 延迟和单位成功成本。

若任务总数 N_task 是正整数，最终成功率为：

~~~math
\mathrm{success\_rate}=\frac{N_{\mathrm{success}}}{N_{\mathrm{task}}}
~~~

若有效调用数为 N_valid_call，所有模型发出的调用数为 N_all_call，则：

~~~math
\mathrm{valid\_call\_rate}=\frac{N_{\mathrm{valid\_call}}}{N_{\mathrm{all\_call}}}
~~~

两个分母都必须大于 0；没有任务或没有调用时，指标应报告为 undefined，不能用 0 填充后制造可比较的假象。重复副作用率同理：如果没有任何已提交动作，就没有一个有意义的重复率。

成本应分项记录：

~~~math
C_{\mathrm{total}}
=C_{\mathrm{model}}
 +C_{\mathrm{tool}}
 +C_{\mathrm{wait}}
 +C_{\mathrm{verify}}
 +C_{\mathrm{human}}
~~~

所有项必须先换算到同一货币或同一归一化成本单位；没有测量的数据不能默认为 0。单位成功成本为：

~~~math
C_{\mathrm{per\ success}}
=\frac{C_{\mathrm{total}}}{N_{\mathrm{success}}}
~~~

当 N_success=0 时该值未定义，应报告 None 或 undefined。一个系统如果成功率提高但单位成功成本上升十倍，是否值得采用取决于任务价值和失败代价，而不是只看单次回答的长度。

比较交错路径与“一次性生成完整计划再执行”时，必须使用相同任务集、成功定义、工具权限和资源上限。交错路径适合外部反馈能够改变下一步的任务；步骤固定、工具延迟极高的流水线可以采用批量读取或预编译工作流。结论应来自受控实验，而不是来自某个模型发布页对“更会使用工具”的描述。

## 19.15 权限、隔离与协议生态

推理预算和权限不是同一个旋钮。增加模型 token、worker 数量或验证轮次，不应自动增加文件写权限、网络访问、数据库范围或生产部署权限。一个实用的分层如下：

- **只读证据区**：保存来源、版本、hash 和脱敏摘要；
- **实验区**：每个可写 worker 使用隔离目录、容器或分支；
- **合并区**：只有合并服务可以生成新的 revision；
- **执行区**：独立 executor 重新检查策略和参数后执行外部动作。

沙箱可以限制文件、网络、CPU、内存和运行时间，但沙箱本身不是正确性证明。代码可能在资源限制内产生错误结果，工具可能返回带提示注入的内容，测试也可能覆盖不足。安全设计必须把 allowlist、参数校验、秘密隔离、审计、验证和回滚结合起来。

以 MCP 这类工具协议为例，协议可以规定消息结构、工具描述、资源引用和生命周期交互，但协议本身不应被理解为业务授权系统。一个 host 是否允许某个 server 访问文件、某个 tool 是否可以写数据库、一个结果是否需要用户确认，都必须由应用的 policy 和 executor 决定。协议标准化减少了连接成本，却不会自动解决权限、可信度和外部副作用问题。

同样的原则适用于 browser agent、插件、workflow、skill 和 agent runtime。它们可以成为动作接口或调度层，但模型能力增强后，某些复杂 scaffold 可能下沉为实现细节。设计者应关注长期不变的约束：状态可恢复、权限可解释、工具调用可验证、事件可重放、成本可归因，而不是只追逐某个接口名称。

## 19.16 一个可运行的事件协议审计器

下面的示例只使用 Python 标准库。它不是生产级消息系统，而是把本章的几个不变量放在一段可以运行的代码中：事件序号连续；工具必须先授权再启动；返回内容的 hash 必须匹配；同一个幂等键不能提交两次；启动后断线会变成 unknown；取消或未知状态之后不能产生新的提交。

~~~python
from __future__ import annotations

import hashlib
import json
from typing import Any


EVENT_TYPES = {
    "reasoning_started",
    "tool_call_proposed",
    "tool_authorized",
    "tool_started",
    "tool_returned",
    "observation_verified",
    "action_committed",
    "tool_failed",
    "tool_unknown",
    "request_cancelled",
    "response_completed",
}


class ProtocolError(ValueError):
    pass


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def audit(events: list[dict[str, Any]], allowed_tools: set[str]) -> dict[str, Any]:
    if not events:
        raise ProtocolError("event log is empty")

    expected_seq = 1
    trace_id = None
    turn_id = None
    calls: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    committed_keys: set[str] = set()
    cancelled = False

    tool_events = {
        "tool_call_proposed",
        "tool_authorized",
        "tool_started",
        "tool_returned",
        "observation_verified",
        "action_committed",
        "tool_failed",
        "tool_unknown",
    }

    for item in events:
        if not isinstance(item, dict):
            raise ProtocolError("each event must be an object")
        if item.get("seq") != expected_seq:
            raise ProtocolError(
                f"expected seq {expected_seq}, got {item.get('seq')}"
            )
        expected_seq += 1

        event_type = item.get("type")
        if event_type not in EVENT_TYPES:
            raise ProtocolError(f"unknown event type: {event_type}")
        if not isinstance(item.get("trace_id"), str):
            raise ProtocolError("trace_id is required")
        if not isinstance(item.get("turn_id"), str):
            raise ProtocolError("turn_id is required")
        if trace_id is None:
            trace_id = item["trace_id"]
            turn_id = item["turn_id"]
        elif (item["trace_id"], item["turn_id"]) != (trace_id, turn_id):
            raise ProtocolError("trace or turn id changed inside one audit")

        call_id = item.get("call_id")
        if event_type in tool_events and not isinstance(call_id, str):
            raise ProtocolError(f"{event_type} requires call_id")
        if event_type not in tool_events and call_id is not None:
            raise ProtocolError(f"{event_type} must not carry call_id")

        if event_type == "reasoning_started":
            continue

        if event_type == "request_cancelled":
            cancelled = True
            for call in calls.values():
                if call["state"] in {"proposed", "authorized"}:
                    call["state"] = "cancelled"
                elif call["state"] == "started":
                    call["state"] = "unknown"
            continue

        if event_type == "response_completed":
            active = {
                key: call["state"]
                for key, call in calls.items()
                if call["state"]
                in {"proposed", "authorized", "started", "returned", "verified"}
            }
            if active:
                raise ProtocolError(f"response completed with active calls: {active}")
            continue

        payload = item.get("payload")
        if not isinstance(payload, dict):
            raise ProtocolError(f"{event_type} requires object payload")

        call = calls.get(call_id)
        if event_type == "tool_call_proposed":
            if call is not None:
                raise ProtocolError(f"duplicate call_id: {call_id}")
            tool = item.get("tool")
            key = item.get("idempotency_key")
            if tool not in allowed_tools:
                raise ProtocolError(f"tool is not allowlisted: {tool}")
            if not isinstance(key, str) or not key:
                raise ProtocolError("proposal requires idempotency_key")
            if key in seen_keys:
                raise ProtocolError("idempotency key was proposed twice")
            if cancelled:
                raise ProtocolError("new tool call after cancellation")
            seen_keys.add(key)
            calls[call_id] = {
                "state": "proposed",
                "tool": tool,
                "key": key,
                "returned_hash": None,
            }
            continue

        if call is None:
            raise ProtocolError(f"unknown call_id: {call_id}")
        if item.get("tool") != call["tool"]:
            raise ProtocolError("tool name changed for a call")
        if item.get("idempotency_key") != call["key"]:
            raise ProtocolError("idempotency key changed for a call")

        if event_type == "tool_authorized":
            if call["state"] != "proposed":
                raise ProtocolError("authorization requires proposed state")
            if payload.get("authorized") is not True:
                raise ProtocolError("authorization is not granted")
            call["state"] = "authorized"
        elif event_type == "tool_started":
            if call["state"] != "authorized":
                raise ProtocolError("tool_started requires authorization")
            call["state"] = "started"
        elif event_type == "tool_returned":
            if call["state"] != "started":
                raise ProtocolError("tool_returned requires started state")
            if "body" not in payload or not isinstance(
                payload.get("content_hash"), str
            ):
                raise ProtocolError("tool_returned requires body and content_hash")
            if stable_hash(payload["body"]) != payload["content_hash"]:
                raise ProtocolError("observation hash mismatch")
            call["returned_hash"] = payload["content_hash"]
            call["state"] = "returned"
        elif event_type == "observation_verified":
            if call["state"] != "returned":
                raise ProtocolError("verification requires returned state")
            if payload.get("content_hash") != call["returned_hash"]:
                raise ProtocolError("verified hash does not match returned hash")
            call["state"] = "verified"
        elif event_type == "action_committed":
            if call["state"] != "verified":
                raise ProtocolError("commit requires verified state")
            if call["key"] in committed_keys:
                raise ProtocolError("idempotency key was committed twice")
            committed_keys.add(call["key"])
            call["state"] = "committed"
        elif event_type == "tool_failed":
            if call["state"] not in {"authorized", "started"}:
                raise ProtocolError("failure is invalid in the current state")
            call["state"] = "failed"
        elif event_type == "tool_unknown":
            if call["state"] != "started":
                raise ProtocolError("unknown result requires started state")
            call["state"] = "unknown"

    return {
        "ok": True,
        "trace_id": trace_id,
        "calls": {key: call["state"] for key, call in calls.items()},
        "committed_keys": sorted(committed_keys),
    }


def event(
    seq: int, event_type: str, call_id: str | None = None, **fields: Any
) -> dict[str, Any]:
    item = {
        "seq": seq,
        "trace_id": "trace-demo",
        "turn_id": "turn-1",
        "type": event_type,
    }
    if call_id is not None:
        item["call_id"] = call_id
        item["tool"] = fields.pop("tool", "read_file")
        item["idempotency_key"] = fields.pop(
            "idempotency_key", "read:demo:c1"
        )
    item.update(fields)
    return item


body = {"bytes": 42, "revision": "repo@r7"}
normal = [
    event(1, "reasoning_started"),
    event(2, "tool_call_proposed", "c1", payload={"path": "parser.py"}),
    event(3, "tool_authorized", "c1", payload={"authorized": True}),
    event(4, "tool_started", "c1", payload={}),
    event(
        5,
        "tool_returned",
        "c1",
        payload={"body": body, "content_hash": stable_hash(body)},
    ),
    event(
        6,
        "observation_verified",
        "c1",
        payload={"content_hash": stable_hash(body)},
    ),
    event(7, "action_committed", "c1", payload={}),
    event(8, "response_completed"),
]
print(audit(normal, {"read_file", "run_test"}))


unknown = [
    event(1, "reasoning_started"),
    event(
        2,
        "tool_call_proposed",
        "c1",
        tool="run_test",
        idempotency_key="test:demo:c1",
        payload={"case": "parser"},
    ),
    event(
        3,
        "tool_authorized",
        "c1",
        tool="run_test",
        idempotency_key="test:demo:c1",
        payload={"authorized": True},
    ),
    event(
        4,
        "tool_started",
        "c1",
        tool="run_test",
        idempotency_key="test:demo:c1",
        payload={},
    ),
    event(
        5,
        "tool_unknown",
        "c1",
        tool="run_test",
        idempotency_key="test:demo:c1",
        payload={"reason": "connection_lost"},
    ),
    event(6, "response_completed"),
]
print(audit(unknown, {"read_file", "run_test"}))


def expect_error(label: str, events: list[dict[str, Any]]) -> None:
    try:
        audit(events, {"read_file", "run_test"})
    except ProtocolError as exc:
        print(label, str(exc))
    else:
        raise AssertionError(f"{label} should have failed")


unauthorized = [
    event(1, "reasoning_started"),
    event(2, "tool_call_proposed", "c1", payload={"path": "parser.py"}),
    event(3, "tool_started", "c1", payload={}),
]
expect_error("unauthorized", unauthorized)

bad_body = [
    event(1, "reasoning_started"),
    event(2, "tool_call_proposed", "c1", payload={"path": "parser.py"}),
    event(3, "tool_authorized", "c1", payload={"authorized": True}),
    event(4, "tool_started", "c1", payload={}),
    event(
        5,
        "tool_returned",
        "c1",
        payload={"body": body, "content_hash": "sha256:wrong"},
    ),
]
expect_error("hash", bad_body)

cancelled_after_start = [
    event(1, "reasoning_started"),
    event(2, "tool_call_proposed", "c1", payload={"path": "parser.py"}),
    event(3, "tool_authorized", "c1", payload={"authorized": True}),
    event(4, "tool_started", "c1", payload={}),
    event(5, "request_cancelled"),
    event(6, "action_committed", "c1", payload={}),
]
expect_error("cancel", cancelled_after_start)
~~~

正常路径的结果形态是：

~~~text
{'ok': True, 'trace_id': 'trace-demo', 'calls': {'c1': 'committed'}, 'committed_keys': ['read:demo:c1']}
{'ok': True, 'trace_id': 'trace-demo', 'calls': {'c1': 'unknown'}, 'committed_keys': []}
unauthorized tool_started requires authorization
hash observation hash mismatch
cancel commit requires verified state
~~~

这个审计器有意保持简单。它没有实现消息队列持久化、分布式锁、真实执行器查询或数据库事务；它做的是把几个容易被自然语言掩盖的错误变成可测试的失败。生产系统还需要为事件持久化、版本迁移、密钥保护、租户隔离、重放窗口和人工接管定义更完整的契约。

## 19.17 读者练习：把循环设计成可以恢复的系统

**练习一：状态机。** 为“读取文件—生成 patch—运行测试—请求合并”画出状态图，分别标出没有副作用、可能有副作用和已经提交的状态。再加入测试超时和用户取消两条边，说明为什么它们不能都指向 failed。

**练习二：Observation 契约。** 为搜索工具设计返回 schema，至少包含来源、版本、获取时间、内容 hash、空结果、明确失败和未知状态。构造一条包含提示注入的网页结果，说明哪些字段可以作为数据，哪些字段不能直接驱动动作。

**练习三：恢复实验。** 让一个模拟执行器在 started 之后随机断开连接。比较“直接重试”和“先查询幂等键再恢复”两种策略的重复副作用率。没有成功任务时，不要把单位成功成本写成 0。

**练习四：评估设计。** 构造 50 个代码调试任务，其中包含空日志、旧 revision、测试超时和隐藏测试失败。比较一次性计划与交错 Agent 的成功率、单位成功成本、unknown 恢复率和 p95 延迟，并解释哪个指标变化会导致你的结论反转。

## 19.18 本章结论与资料边界

Interleaved thinking 的核心不是把可见思考写得更长，而是让外部观察在正确的协议边界上参与下一步决策。它要求系统保存状态、动作、观察、来源、版本、权限、预算和副作用，而不是把所有内容拼成一段无法恢复的文本。

交错循环适合事实会变化、执行会暴露错误、下一步依赖外部反馈的任务；它不适合无反馈地重复调用工具。工具调用必须经历授权、启动、返回、验证和提交等有明确含义的状态；超时和断线必须保留 unknown，不能为了让流程继续而伪装成失败或成功。高风险动作需要应用层的提议、批准、幂等执行、状态查询和补偿；checkpoint 则要保存“下一步是什么”，而不只是保存上一段回答。

本章的状态机、事件日志、预算账本和审计器是通用工程抽象。ReAct 和 Toolformer 支撑“推理与工具交错”的研究脉络，但它们不等于某个生产协议的完整实现；MCP 等协议文档可以说明消息和工具接口的标准化方向，却不能替代应用自己的权限与副作用策略；OWASP 的 LLM 应用安全资料支持把外部内容视为潜在攻击面，但具体风险仍需在目标系统中通过威胁建模和对抗测试确认；Stripe 和 AWS 关于幂等请求的工程资料说明了重试安全的基本原则，但不同业务的状态查询和补偿语义仍需单独设计。

参考资料：

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*：<https://arxiv.org/abs/2210.03629>
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*：<https://arxiv.org/abs/2302.04761>
- Model Context Protocol Specification：<https://modelcontextprotocol.io/specification/2025-06-18>
- OWASP GenAI Security Project, *LLM Applications Security Risks*：<https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- Stripe API Reference, *Idempotent requests*：<https://docs.stripe.com/api/idempotent_requests>
- AWS Builders' Library, *Making retries safe with idempotent APIs*：<https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/>
