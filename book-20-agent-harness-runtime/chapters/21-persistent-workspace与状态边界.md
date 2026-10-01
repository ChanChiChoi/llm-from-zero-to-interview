# 第 21 章 Persistent Workspace：文件状态、模型记忆与外部系统必须分界

## 21.1 为什么“记住上次对话”不等于拥有工作区

用户对 coding Agent 说：“继续昨天的任务。”他期待的不只是模型记得一句话，而是昨天创建的分支、生成的 patch、失败测试、下载的依赖和未完成的计划仍然可以被检查。

Persistent workspace 是可持续的任务环境；memory 是对历史信息的抽象；model context 是一次调用能够看到的输入。三者如果混在一起，系统会把模型摘要当成真实文件状态，或把一个租户的记忆带到另一个工作区。

## 21.2 三层状态模型

可以把状态分成：

```text
model state: 当前调用可见的 token、item、summary
workspace state: 文件、分支、artifact、依赖、测试结果
external state: 数据库、云资源、工单、邮件和部署系统
```

模型 state 可以丢失或折叠，workspace state 应有 revision 和可回滚机制，external state 可能不可逆。越靠外层，提交门槛越高。

一个摘要“patch 已准备好”不能代替 workspace 中真实的 diff；workspace 中存在 diff 也不能代表外部部署已经完成。

## 21.3 Workspace 的身份和版本

每个 workspace 应有 tenant、project、task、branch 或 snapshot 身份，并记录基础 revision。写入时产生新的 revision，工具结果引用它；并行 Agent 使用独立 workspace 或明确的读写锁。

如果用户在 Agent 暂停期间手动修改文件，恢复时必须检测版本差异。可以用三方合并、重新生成 patch 或请求用户确认，不能把旧上下文强行覆盖新代码。

## 21.4 让 artifact 成为交接对象

Agent 之间最好通过 artifact 交接，而不是把全部工作区塞进 prompt。artifact 可以是 patch、测试报告、日志摘要、数据文件或图片，并带有：

```text
artifact_id, workspace_revision, creator, schema_version,
content_hash, source, status, retention
```

消费者先检查 revision 和 hash，再决定读取内容。生成的文件、未验证的草稿和已经提交的产物要有不同状态，避免把临时文件当作最终结果。

## 21.5 一个跨会话任务

第一天 Agent 在分支 `task-42` 上修改代码，单元测试通过但集成测试未完成。用户关闭浏览器。第二天恢复时，系统读取 workspace revision、测试 job、artifact 和 pending effect；若集成测试已在后台完成，就加载其结果；若分支被用户修改，则标记冲突并重新分析。

模型可以收到一份折叠摘要，但摘要只描述状态，不能伪造文件。真正的文件内容来自 workspace，真正的部署状态来自 deployment API。这样即使模型换版本，任务仍然可以通过外部状态继续。

## 21.6 权限边界

workspace 权限和外部系统权限必须分开。Agent 可以在临时目录写文件，不代表它能写生产数据库；可以读取一个仓库，不代表可以读取其他租户的 workspace。

每次从 workspace 向外部系统提交，都要经过策略、用户确认和审计。把 secret 放进 workspace 也不应让模型自动读取；文件系统沙箱需要阻断敏感路径、符号链接逃逸和不必要网络访问。

## 21.7 清理、保留和隐私

持久化状态会产生成本和隐私责任。要定义 workspace 和 artifact 的保留期限、删除语义、备份范围、租户隔离和审计访问。删除一个对话不一定删除生成文件或外部工单，用户界面要明确说明范围。

对于模型上下文，尽量保存引用而不是复制全部敏感文件。日志中使用 hash 和脱敏摘要，只有具有权限的恢复操作才能读取原文。

## 21.8 一致性和并发

两个 Agent 同时写同一 workspace 会产生覆盖、冲突和难以回放的结果。可以选择串行锁、分支隔离、乐观并发和合并队列。任何策略都要把 workspace revision 放进动作条件：只有基于 `r17` 的 patch 才能应用到 `r17`，否则失败并重新计算。

这个条件类似 compare-and-swap：

```math
\mathrm{apply}(p,r)=r'\quad\mathrm{when\ current}=r;
\qquad\mathrm{apply}(p,r)=\mathrm{conflict}\quad\mathrm{otherwise}
```

## 21.9 评估 workspace 可靠性

测试用户中断、并行 worker、手动修改、恢复、删除、权限变化和外部提交。指标包括 revision 一致率、冲突检测率、错误覆盖率、artifact 可回放率、恢复成功率、重复副作用和清理成功率。

不要只测试 happy path。最有价值的案例是“模型以为文件存在但文件已被删除”“工具返回成功但 workspace 写入失败”“外部提交已发生但 checkpoint 未更新”。

## 21.10 常见误区

把 workspace 当作模型 memory；把模型摘要当作文件事实；所有 Agent 共享可写目录；不保存 revision 和 hash；用永久文件代替有期限 artifact；删除会话却保留敏感日志；从 workspace 自动向外部系统提交。

## 21.11 面试回答与练习

回答“Agent 如何持久化状态”时，应区分模型上下文、workspace 文件和外部系统；为 workspace 设计身份、revision、artifact、锁和回滚；恢复时从真实环境读取事实，重新校验权限和外部副作用；同时设计保留、删除和租户隔离。

练习一：设计一个代码 workspace manifest。

练习二：两个 Agent 基于同一个 revision 生成不同 patch，如何安全合并？

练习三：用户删除对话时，哪些状态必须一起删除，哪些需要单独确认？

### 21.11.1 workspace 是事实层，不是模型记忆

workspace 中的文件、分支、构建产物和测试结果是可以检查的环境状态；模型在上下文中说“我已经修改了文件”只是一个计划或陈述，必须通过文件 revision 和 diff 证实。把聊天摘要当作 workspace 真相，会让恢复后的 Agent 继续使用已经不存在的文件。

一个 workspace manifest 可以记录：

```text
workspace_id / tenant_id / base_revision
head_revision / dirty_files / artifact_checksums
toolchain_revision / lock_owner / retention_policy
```

每次写入都生成可比较的 revision，危险操作先生成 patch 或临时分支，验证后才合并到主工作区。

### 21.11.2 并发和冲突

两个 Agent 从同一个 revision 开始工作时，不能默认后提交者覆盖前者。合并器应检查修改文件、行范围、依赖、测试和权限；冲突时保留双方 patch，进入人工或专门 merge Agent。外部数据库则需要事务、版本号或条件更新。

### 21.11.3 删除和保留边界

删除对话不一定等于删除已提交代码、审计日志或企业知识库中的 artifact；但个人资料、临时凭证和未提交敏感文件可能需要同步删除。系统应把模型历史、workspace、artifact、外部提交和审计保留政策分开声明，支持按租户和数据类别查询。

### 21.11.4 恢复测试

对 workspace 做故障注入：写入中断、锁过期、磁盘只读、模型重启、用户手动修改、工具返回旧 revision 和 checkpoint 损坏。恢复后检查 diff、checksum、锁、权限和外部副作用，不能只看 Agent 是否继续输出文本。

## 21.12 workspace 的事实优先级

当模型记忆、聊天摘要和文件状态冲突时，已提交且可校验的 workspace artifact 应优先。模型说“已经改好”只是一个意图或陈述，必须由 revision、diff 和测试证实。

并发任务要使用锁、版本号或冲突合并；清理和保留策略要绑定用户、任务和租户，不能因为会话结束就无限保留敏感文件。

## 21.13 文件、memory 和数据库的三种真相

Persistent workspace 中的文件、模型 memory 和外部数据库可能同时描述同一个事实，但它们的权限、版本和提交语义不同。文件 diff 是可审阅 artifact，memory 是面向未来任务的摘要，数据库可能是业务权威源，不能因为模型在 prompt 中看到一段文字就把三者等同。

读取时应带来源、revision、时间和 owner；写入时应明确哪个系统成为权威，是否需要人工确认和事务提交。

## 21.14 checkpoint 和恢复的边界

长任务 checkpoint 至少要保存 workspace revision、未提交 diff、任务状态、工具调用、context summary、权限决定和外部资源引用。只保存对话文本无法恢复文件和工具状态，只保存文件又可能丢失模型尚未完成的计划。

恢复后应重新验证文件 checksum、权限、租约、模型 revision 和工具幂等性。对已经提交的副作用不能简单重放，应读取外部系统状态后继续。

## 21.15 安全和租户隔离

持久 workspace 需要目录、文件、artifact、缓存和 memory 的租户隔离。新任务不能因为共享前缀或相同文件名读取旧租户信息；日志、索引和备份也属于数据面。

删除和过期应覆盖主文件、派生摘要、向量索引、缓存和 checkpoint。审计要能回答谁在何时读取、修改、提交或回滚了哪个 artifact。

## 21.16 workspace 的并发修改

用户、Agent、IDE、CI 和后台任务可能同时修改 workspace。编辑前要检查 revision，写入时使用原子 patch 或锁，提交后重新读取 diff。否则一个成功的工具调用可能覆盖另一方的新修改。

checkpoint 要记录基线 revision、当前 diff、冲突和审批状态。恢复时不能把旧 patch 直接覆盖到新文件，应重新计算或请求人工解决。

## 21.17 状态边界的审计问题

一次任务结束后，应能回答哪些信息写入文件、哪些只存 memory、哪些提交到外部系统、哪些被删除。日志、缓存、索引和备份都可能成为派生副本，删除和租户隔离不能只处理主文件。

把 workspace 视为可审阅 artifact，把 memory 视为带来源的候选，把外部系统视为有事务语义的权威源，三者的边界清楚，Agent 才能在长任务中恢复和回滚。

## 21.18 Workspace manifest 与条件写入

持久工作区不能只靠一个目录名识别。至少要有 manifest，记录租户、任务、基线、当前 revision、工具链和保留策略：

```text
WorkspaceManifest {
  workspace_id
  tenant_id
  task_id
  base_revision
  head_revision
  dirty_paths
  artifact_checksums
  toolchain_revision
  lock_owner
  policy_revision
  retention_class
}
```

写入操作要带基线 revision，等价于 compare-and-swap：

```math
\mathrm{write}(p,r_b)
=\begin{cases}
r_{new},&r_{current}=r_b\\
\mathrm{conflict},&r_{current}\ne r_b
\end{cases}
```

这样，用户在 Agent 运行期间修改文件时，旧 Agent 的 patch 会明确失败，而不是悄悄覆盖新内容。`head_revision` 只能在 patch、测试和权限检查完成后推进；临时文件、编译缓存和未验证输出不应伪装成主 revision。

manifest 还要区分文件 artifact 和外部 artifact。文件 checksum 能证明当前内容没有改变，却不能证明镜像、数据集、数据库记录或发布系统仍处于相同状态；后者需要各自的版本或查询接口。恢复时逐类读取权威源，不能只信 workspace 中的旧文本。

## 21.19 三方合并、锁与后台任务

两个 Agent 都从 `r17` 读取文件并生成 patch 时，后提交者不能简单覆盖前者。安全合并至少比较三份内容：共同基线 `r17`、当前 workspace `r18` 和待应用 patch。只有当 patch 修改的区域与 `r17 -> r18` 的变化不冲突，并且依赖、测试和策略检查通过，才可以自动合并。

并发控制可以有三种层次：

| 方式 | 优点 | 代价 |
| --- | --- | --- |
| 串行锁 | 语义最简单 | 长任务阻塞其他用户 |
| 分支/临时目录 | 隔离好、易审阅 | 需要合并和垃圾回收 |
| 乐观版本控制 | 并发度高 | 冲突处理更复杂 |

锁不是万能的。worker 崩溃后锁必须过期，接管者要检查心跳、最近 checkpoint 和 workspace revision；不能看到锁过期就直接写入。CI、IDE、人工编辑器和后台格式化任务也应使用同一套 revision 协议，否则 Agent 之外的写入会绕过保护。

对需要长时间运行的测试或构建，workspace 只保存 job 引用和输出 checksum，真正进程由 job scheduler 管理。恢复时先查询 job 状态，再决定等待、读取结果或取消。把一个运行中的进程状态只写成“测试中”，无法防止重启后重复占用 GPU 或覆盖日志。

## 21.20 保留、删除与租户隔离的闭环

持久 workspace 的数据面不止主目录，还包括快照、patch、缓存、向量索引、日志、备份和 checkpoint。删除流程应先枚举派生副本，再按数据类别执行删除或保留，并返回可审计的删除状态：

```text
request deletion
  -> identify workspace/artifacts/cache/index/backup
  -> apply tenant and legal-retention policy
  -> delete or tombstone each copy
  -> verify inaccessible reads
  -> write deletion audit record
```

删除对话不一定能删除已经提交到企业数据库的记录；反过来，临时凭证、未提交的个人文件和跨任务缓存可能必须立即清除。产品层需要把这些边界说清楚，runtime 层则要用租户和数据类别执行，而不是只删一个 conversation id。

租户隔离要覆盖路径、文件描述符、符号链接、缓存 key、搜索索引和日志查询。`tenant_id` 不能只放在 UI 字段里；每次读取和写入都要在服务端验证 ownership，artifact URL 也要使用短期授权而不是永久可猜测路径。

## 21.21 worked example：用户修改后的安全恢复

第一轮 Agent 基于 `r17` 修改 `src/parser.py`，生成 `patch-41`，但测试尚未完成。用户随后手动修改 `src/parser.py` 的同一函数并提交为 `r18`。原任务恢复时，runtime 发现 checkpoint 的基线为 `r17`，当前为 `r18`，于是执行三方比较：

1. 读取 `r17`、`r18` 和 `patch-41`，确认同一函数存在重叠修改。
2. 不自动应用 patch，保留 `patch-41` 的 checksum 和原始测试结果。
3. 在隔离分支尝试合并，若产生冲突则进入 `needs_merge`。
4. 让用户或专门合并器选择结果，生成新 revision `r19`。
5. 在 `r19` 上重新运行测试，再允许提交或发布。

如果 patch 只修改了与用户改动无关的文件，仍然要重新检查依赖和测试，因为 `r18` 可能改变了配置或接口。恢复的目标是获得一个经过新基线验证的 artifact，不是把旧 Agent 的意图原样执行完。

## 21.22 状态边界的可观测性

每次跨边界操作都应产生事件：读取 workspace、创建 revision、应用 patch、取得锁、产生 artifact、提交外部系统、删除副本和恢复任务。事件包含主体、租户、时间、基线 revision、结果 revision、策略版本和关联 task id。

可以用以下指标观察 workspace 是否真的可靠：

```math
R_{\mathrm{workspace}}
=\frac{N_{\mathrm{tasks\ with\ verified\ artifact\ and\ revision}}
       {N_{\mathrm{completed\ tasks}}},\qquad
       N_{\mathrm{completed\ tasks}}>0
```

`R_workspace` 只在存在已完成任务样本时定义；若 `N_completed tasks=0`，应报告
`not_applicable`，不能用人为下限分母伪造 `0.0` 或 `1.0`。聚合和质量门禁要显式处理
该状态，再决定补充样本或跳过整组评估。

同时记录冲突发现率、错误覆盖率、锁接管成功率、checksum 不一致率、删除验证失败率和跨租户拒绝率。一个系统如果“任务完成率”很高，却没有办法回答某个文件由谁、基于哪个 revision 修改，就不具备可审计的持久工作区能力。

## 21.23 workspace 的版本、锁和租约

持久工作区至少要区分文件内容版本、任务版本和授权版本。模型读取文件后，另一个进程可能已经修改它；提交 patch 时必须检查 base revision，必要时重新合并。锁不是永久占有权，而应带 owner、租约、心跳、超时和接管规则。

一个 workspace 操作可以抽象为：读取 revision、生成候选、检查 diff、运行验证、原子提交。任何一步失败都保留候选 artifact，不能直接覆盖用户当前修改。租约过期后，旧 worker 只能读到任务已失效，不能凭旧 token 继续写入。

## 21.24 删除和隔离也属于 workspace 能力

用户删除项目或撤销租户后，系统不仅要删除目录，还要处理快照、缓存、日志、索引、向量、临时压缩包和恢复点。审计记录可以保留最小元数据，但不能借“为了回放”无限期保留原文和秘密。

隔离测试应模拟两个租户使用相同文件名、相同 prompt、相同 cache key 和相同 worker slot，检查任何 artifact、摘要和状态都不会跨租户出现。路径前缀相同不等于身份相同，真正的 owner 和 ACL 必须进入每个状态对象。

## 21.25 workspace 的可审计交付

任务完成时，交付物应包含最终 revision、diff、测试结果、依赖变化、未解决风险和执行者身份。用户看到的总结可以简洁，但系统内部必须能够回答“谁在何时基于哪个版本做了什么”。没有这个证据链，持久 workspace 只是一个更容易遗留脏状态的共享目录。

## 21.26 workspace 冲突的处理例子

用户在本地修改配置后，Agent 基于旧 revision 生成 patch。提交时发现 base checksum 不匹配，正确动作是保存候选 patch、重新读取当前文件、展示冲突并重新验证；不能强制覆盖，也不能让模型从旧上下文继续猜。冲突本身应成为可查询的 artifact。

## 21.27 workspace 和模型 memory 的边界

workspace 保存可版本化的文件和状态，memory 可能保存偏好、摘要和长期事实。前者有明确 owner、revision 和 diff，后者需要来源、时效和删除策略。把用户文件悄悄写进长期 memory，既会污染后续任务，也可能违反数据保留和租户隔离。

## 21.28 Persistent Workspace 的阶段性判断与资料边界

Persistent workspace 让长任务拥有真实、可版本化的工作环境，但它不是隐式 memory，也不是外部系统授权。可靠设计必须把模型 state、workspace state 和 external state 分开，以 manifest、revision、artifact、权限和提交边界连接它们；并发修改、删除派生副本和后台任务恢复都必须进入同一套审计链路。

本章的版本控制、artifact、租户隔离和分布式一致性是通用工程原则；具体 harness 的 workspace 目录、保留策略和持久化语义以其文档为准。

## 21.29 Claude Fable 5.1：thinking state 有两道独立的绑定门

Persistent workspace 之外，长对话还可能携带 provider 返回的 thinking block。它不是可自由编辑的文本，也不是 workspace 文件；客户端必须把原始 block 当作不透明状态回传。Claude Fable 5.1 的公开协议显示，能否继续使用某个 block 要分别通过两道门：

1. **Model binding：产生 block 的模型是否在目标模型的可读集合内。**
2. **Prefix binding：该 block 之前的 system、tools 和消息前缀是否保持不变。**

两类不匹配不能混成一个“上下文坏了”错误。当前官方文档给出的关键方向如下：

| thinking block 的 producer | 目标模型 | 公开兼容结论 |
| --- | --- | --- |
| Claude Opus 5 或更早的已列 Claude 模型 | Claude Fable 5.1 | 可读 |
| Claude Opus 5.5 | Claude Fable 5.1 | 仅 Claude API 文档明确可读，不外推到其他托管平台 |
| Claude Fable 5.1 | Claude Opus 5 或 Claude Opus 5.5 | 目标模型不可读；block 会被丢弃 |

具体可读集合必须按 provider 和 model pair 查官方矩阵。模型兼容是有方向的：Fable 5.1 能读取 Opus 5.5 的 block，不代表 Opus 5.5 能读取 Fable 5.1 的 block；同样，“可读”也不表示该模型是 refusal fallback 的默认目标。Fable 5.1 的文档列出的默认 fallback target 是 Opus 4.8 与 Opus 5，不能因 Opus 5.5 的单向兼容性而擅自加入。

Model-binding 不匹配时，API 会在目标模型看到请求前丢弃不可读 block；文档说明它不进入 input-token 计费。启用 `thinking-binding-controls-2026-08-01` 后，响应的 `input_transformations` 可报告 `model_binding_mismatch`；不启用时这类丢弃可能不显眼。它是状态兼容/路由事件，不是模型拒答或推理失败。

Prefix-binding 检查的是 block 前面的内容。改写早先消息、重建 system prompt 或 tools、删除中间 turn，都会使其后的 thinking blocks 失效。Fable 5.1 对 2026-08-31 00:00 UTC 及之后创建的 API 账户默认执行该检查；更早的账户要通过 `thinking.block_binding.prefix_mismatch_behavior` 明确启用。严格路径返回 400；显式选择 `drop_block` 时，API 丢弃失效 block 并以 `prefix_binding_mismatch` 记录转换原因。

长会话宜采用 append-only transcript：不要为了改指令而重写旧消息，可使用 mid-conversation system message；工具更新使用 `tool_addition`/`tool_removal`；改 effort 使用 per-message `output_config`；裁剪历史优先采用服务端 context editing 或 compaction。文档还区分了安全删除形状：可以从最旧端移除连续的 thinking-block 前缀；从中间挖掉一个 block 会使后续 blocks 失效。effort-only 的消息级更新可以保留已有 prompt prefix/cache，而修改顶层 thinking 配置或早期内容则不能据此假设 cache 仍命中。

因此，一个可恢复的 Agent trace 至少应分别记录 producer/consumer model ID、API surface、block 顺序与 opaque signature 的引用、prefix digest、历史编辑/压缩事件、`input_transformations` 和实际 fallback model。签名与 thinking 内容仍按 provider 的 opaque state 处理，不应作为可读 CoT 展示给用户，也不能把它当作跨模型通用 memory。

当前复核依据为 Anthropic [Fable 5.1 What's New](https://platform.claude.com/docs/en/models/fable-5-1/whats-new-fable-5-1.md)、[Fable 5.1 migration guide](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide.md) 和通用 [Thinking / preserved thinking](https://platform.claude.com/docs/en/build-with-claude/thinking.md)。对应的零依赖协议样例见 [claude_fable51_state_protocol_audit.py](../../research/model-update-2026-09/code/claude_fable51_state_protocol_audit.py)；它只验证本地合成状态机，不代表真实 Anthropic endpoint probe。

## 21.30 Qwen3.6：历史 thinking trace 不等于持久记忆

“模型能利用旧思考”容易被误解成模型在服务端保存了跨轮记忆。Qwen3.6-35B-A3B 的公开实现更具体：模型卡称它经训练以利用历史 thinking；chat template 决定调用方提交的历史里哪些 assistant reasoning block 会被序列化进本次 prompt。

| 控制项 | 作用范围 | 不能推出什么 |
|---|---|---|
| `enable_thinking` | 当前生成模式；模板控制生成提示词里的 `<think>` 前缀 | 不代表旧 reasoning block 会保留 |
| `preserve_thinking` | 输入 transcript；让历史 assistant `<think>…</think>` block 随 prompt 保留 | 不创建服务端 memory，也不保存调用方未再提交的历史 |
| workspace / 数据库 / memory store | 应用或 Agent runtime 的显式持久化状态 | 不由 chat-template 参数自动创建 |

固定模板在 `preserve_thinking` 未启用时，只输出最后一条 user 消息之后的 assistant reasoning；启用后才将更早的 reasoning block 一并渲染。Qwen 官方称这可能减少重复推理并改善 KV cache 利用，但保留的历史本身也会消耗上下文/KV，实际成本要测，不能把发布方说明写成必然净节省。

请求协议还要区分 endpoint：本地 vLLM/SGLang 示例把开关放在 `chat_template_kwargs`；Model Studio 示例使用相应的顶层字段。相同模型、不同 endpoint 的字段位置不保证相同，宜先做 schema/capability 检查。若 transcript 被删掉、改写或跨会话未重新载入，`preserve_thinking` 无法替应用恢复旧内容。

这与前文 workspace 持久化的层次关系是：模型可处理历史 reasoning ≠ API 自动保存历史 ≠ Agent runtime 将事实写入持久 workspace。一次可靠的长任务系统应明确存储权威事实、压缩/裁剪策略、来源和版本；不要把模型原始 thinking 当作唯一 checkpoint 或用户可见解释。

Qwen 官方发布博客也明确推荐 Agent 使用 `preserve_thinking`，并示例 `qwen3.6-flash` hosted ID 与 OpenClaw/Claude Code/Qwen Code 接入；这些是 API/产品合同示例，不证明真实 endpoint 当前可用，也不改变“调用方必须重传 transcript”的状态边界。博客评测依赖的 user model、judge 和任务执行器另见第二十一册第 83 章 83.15.3。

来源与固定 artifact 记录见 [`Qwen3.6-35B-A3B 资料摘记`](../../research/model-update-2026-09/qwen3.6-35b-a3b-source-notes.md)。本节依据静态官方模型卡、模板和发布博客，不代表真实 API 行为、端到端任务连续性或生产级 memory 评估。
