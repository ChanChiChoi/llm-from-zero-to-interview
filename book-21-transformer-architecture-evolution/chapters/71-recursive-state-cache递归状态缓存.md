# 第 71 章 Recursive State Cache：固定状态不等于没有缓存

## 71.1 state cache 的定义

递归 attention 或 DeltaNet 类层不保存每个历史 token 的完整 KV，而保存随时间更新的状态 `S_t`。它仍然是缓存，只是缓存的单位从 token 变成了 state：

```math
S_t=F(S_{t-1},x_t;\theta_t)
```

如果 state 大小不随 `T` 线性增长，超长上下文的内存曲线可能更平缓；但 state 是压缩表示，不支持像显式 KV 那样任意定位原文。

## 71.2 与 KV cache 的运行时差异

KV 可以按 block 分配、淘汰、共享和重算；recursive state 需要记录 step、层级、门控统计、版本和 reset 边界。它不是一段可以随意拼接的 tensor。

迁移 state 时，模型 revision、dtype、并行布局、位置处理和更新顺序必须一致。服务端如果把 state 绑定在 batch slot，而请求在 continuous batching 中移动，极容易产生错位。

## 71.3 state 的事务模型

一次更新可以分为：

```text
read committed state
  -> create temporary state
  -> apply token updates
  -> verify
  -> commit or rollback
```

speculative decode、工具失败恢复和 preemption 都需要这个边界。draft token 不能直接覆盖正式 state；外部状态未知时也不能把未确认的计划写进 committed state。

## 71.4 speculative decode 的例子

设正式状态为 `S_t`，draft 产生 `x_{t+1:t+K}`。runtime 先在临时副本上更新；target 接受前 `a` 个 token 后，只提交：

```math
S_{t+a}=\mathrm{Update}(S_t,x_{t+1:t+a})
```

拒绝 token 的 state 被丢弃。如果 target 在拒绝位置重新生成一个 token，则该 token 要经过同一套更新和提交逻辑。最终输出、state、grammar 和 usage 必须使用同一个 accepted prefix。

## 71.5 batch 和 padding

batch 中请求长度不同，padding 不能更新 state；请求完成后也不能继续写入。合并/拆分 batch 前后，针对相同 prefix 的 state 应保持数值一致。

一个常见 bug 是为了让 tensor shape 对齐而继续执行 padding update。模型文本结果可能只在长序列或特定 batch 顺序下出错，因此必须使用 checksum 和单请求 baseline 对照。

## 71.6 snapshot、restore 和迁移

state snapshot 至少包含 tensor、step、model revision、dtype、layout、position、request owner 和 checksum。restore 时检查 workspace/task 仍然属于同一租户，模型和 kernel 版本是否兼容，外部工具调用是否可能已经提交。

如果不兼容，可以重新从最近的显式 checkpoint 或文本摘要重算；不能静默把旧 state 喂给新模型。

## 71.7 评测 state 一致性

对同一 prefix 做 full recompute、cache continuation、batch continuation、snapshot restore 和 speculative accept/reject。保存每层 state norm、checksum、最终 logits 和 token。

允许数值误差时给出 tolerance，不能只比较字符串。若差异随长度放大，检查累计误差、state precision、gate 饱和和更新顺序；若只在恢复出现，检查序列化和版本。

## 71.8 失败模式

state reset 漏掉，造成跨请求污染；preemption 只保存 token offset；batch padding 更新 state；拒绝 draft 后不回滚；完成请求仍持有 state；不同 worker 使用不同 layer order；租户隔离缺失；固定 state 被误写成无缓存。

## 71.9 资源和并发

state 固定并不代表没有容量限制。每个请求仍有 state 数量、dtype、snapshot、workspace 和 scheduler metadata。若 state update kernel 吞吐不足，GPU 可能在长 decode 中受限于状态带宽，而不是矩阵乘。

容量规划应测请求数、state GiB、snapshot 频率、恢复延迟、state update time 和 p99。对高风险长任务，可以限制并发或采用专用 state pool。

## 71.10 面试回答与练习

回答“recursive state cache 和 KV cache 有什么不同”时，应说前者保存固定大小的递归历史表示，后者保存按 token 的显式 K/V；前者省长历史存储但有压缩和精确检索代价，serving 重点是 reset、snapshot、restore、rollback、版本和隔离。

练习一：给出一次包含 accepted 和 rejected token 的 trace，标出 state commit 与 rollback。

练习二：设计一个检测 padding 更新 state 的 batch 测试。

练习三：列出 state restore 的五个兼容检查。

### 71.10.1 递归 state 的提交语义

对 token 序列 `x_{1:t}`，state 更新可以写成：

```math
S_t=F(S_{t-1},x_t),\qquad S_0=\mathrm{reset}(request)
```

在普通 decode 中，`S_t` 可以提交；在 speculative decoding 中，draft token 只产生临时 `S'`，target 接受前不能覆盖正式 `S_t`。因此 cache API 需要 `begin_step`、`commit_step` 和 `rollback_step` 这类明确语义。

### 71.10.2 batch 和 padding 的陷阱

如果 padding token 也调用 `F`，state 会被无意义输入改变；如果 batch 重排时把 state 指针按序号而不是 request ID 迁移，两个请求会互相污染。测试应改变 batch size、padding、请求加入/退出顺序、preemption 和恢复位置，比较单请求基线与批处理结果。

### 71.10.3 容量和生命周期

固定大小 state 省去了随 `T` 增长的部分存储，但 state 数量随并发增长。每个 state 仍要有 owner、dtype、层/头、step、position、model revision 和租约。请求取消、超时和 tenant 删除时，state 必须立即回收或进入可审计的清理流程。

### 71.10.4 与显式 KV 的路由

连续流预测、低延迟对话和长日志处理可能适合 recursive state；需要随机引用、审计和精确回放的任务可以保留显式 KV 或 RAG。一个 hybrid router 要能知道当前 state 只能支持什么查询，不能因为 cache 便宜就接受超出它记忆能力的任务。

## 71.11 state 的代数和可组合性

显式 KV 可以按 token block 拼接，递归 state 通常不能随意拼接。若对前缀 A 得到 state S_A，对前缀 B 得到 state S_B，通常不存在一个简单的 concat，使其等于按 A、B 连续处理得到的 S_AB。必须用更新函数继续运行：

~~~math
S_{AB}=F^\star(S_A,B)
~~~

只有在特定线性或可结合结构下，才可能把状态块以组合运算合并。这个差异影响 context parallelism、缓存共享和跨 worker 迁移。

## 71.12 状态版本和可观测性

每个 state snapshot 应记录模型 revision、层 schedule、dtype、position、step、owner、创建原因和校验摘要。监控还要保存 state bytes、update time、snapshot time、restore time、rollback count、跨 worker 迁移次数和异常 reset。

出现答案回归时，按 trace 回放：先比较 token 前缀，再比较每层 state norm 和 checksum，最后比较 logits。只比较最终字符串会把数值漂移、状态错位和采样差异混在一起。

## 71.13 与显式 KV 的能力路由

路由器要知道 state 的能力边界。连续预测、摘要和重复日志可以优先使用 recursive state；精确引用、随机回放、冲突审计和需要用户核验的任务应保留显式 KV 或 retrieval。

如果请求在执行中从低风险摘要升级为高风险发布审计，系统应允许从 state 路径切换到原文路径，并把切换原因写入审计记录。固定使用一种 cache 会让资源优化变成能力误用。

## 71.14 状态回滚不只是复制一个 tensor

递归 state 往往与 step、position、gate 统计、临时 workspace 和 scheduler metadata 一起变化。speculative decode 拒绝候选时，必须回滚所有依赖候选 token 的对象；只恢复主 state 而保留位置或统计量，会产生难以复现的漂移。

可以把一次状态事务表示为：

~~~text
begin(request, base_step)
  -> update(state, token)
  -> update(position, metadata)
  -> verify
  -> commit or rollback
~~~

事务日志应能回答哪些 token 已提交、哪些 token 只在 draft 分支中出现，以及恢复后使用了哪个 model revision。

## 71.15 state migration 的一致性

跨 worker 迁移 state 时，物理 layout、设备 dtype、层顺序和并行 rank 必须一致。若不同 worker 使用不同 kernel 或不同精度，允许的误差要预先定义，并在迁移前后比较 logits 和任务结果。

迁移也涉及安全：state owner、租户、权限和外部工具上下文不能被另一个请求继承。请求取消时，应释放设备内存和元数据租约；只删除 tensor 而保留索引，会造成后续 slot 复用错误。

## 71.16 state cache 的容量模型

递归状态不随 token 线性增长，但会随层数、state 维度、dtype、并发和 snapshot 频率增长。可以写成：

~~~math
M_{\mathrm{state}}
=N_{\mathrm{request}}
\sum_{\ell=1}^{L}d_{\ell}b_{\ell}
+M_{\mathrm{snapshot}}
+M_{\mathrm{metadata}}
~~~

如果每层有多个 state 或同时存在 draft、target 两份分支，实际容量还会增加。固定状态不等于免费状态，也不等于可以取消 admission control。

## 71.17 状态事务的提交语义

生成 token、更新 state、更新 position、调用工具和写入外部副作用不一定同时成功。state cache 可以采用 begin、update、verify、commit/rollback 的事务语义；工具副作用则需要更强的幂等和人工确认。

状态回滚只回滚模型内部对象，不代表已经发出的邮件、写入的数据库或执行的命令可以自动撤销。Agent runtime 必须把 state transaction 和 external side effect 分开记录。

## 71.18 state cache 与租户隔离

状态对象应带 request id、owner、tenant、model revision、position、dtype、checksum 和过期时间。batch slot 复用、worker 迁移、请求取消和异常重启都要释放 owner 和租约。

恢复测试要故意交错两个租户的长流请求，并比较单独运行与批处理运行的 logits。只要出现差异，就应先查 state 生命周期，而不是把错误归因于模型随机性。

## 71.19 从状态定义走向生命周期

Recursive State Cache 是对历史 state 的生命周期管理，不是“把 cache 删除”。只要模型使用递归路径，就必须为 state 建立 owner、step、版本、事务、容量和恢复规则。

资料入口可以参考 Qwen/Kimi 公开模型资料、递归 attention 论文和 SGLang 等 serving 文档；具体 state layout 和事务保证属于实现细节。

## 71.20 state cache 与 KV cache 的边界

KV cache 通常按 token 保存显式 K/V，recursive state cache 保存经过递归更新后的状态。前者有较清晰的 token 地址，后者更像一个随序列推进的 checkpoint。两者都叫 cache，却有不同的生命周期：

| 维度 | KV cache | recursive state cache |
| --- | --- | --- |
| 增长 | 常随 token 或窗口增长 | 通常固定或受控增长 |
| 地址 | token/page 可定位 | 依赖 step 和更新路径 |
| 恢复 | 恢复 K/V 和位置 | 恢复 state、step、版本和更新语义 |
| 证据回读 | 相对直接 | 需要 provenance 或外部索引 |
| 共享 | prefix sharing 较成熟 | 需验证 state owner 和边界 |

把 state 当作普通 KV page 处理，容易遗漏 reset、状态事务和版本不兼容。

## 71.21 state lifecycle

一个可操作的状态机是：

~~~text
EMPTY
  -> RESERVED
  -> PREFILLING
  -> ACTIVE
  -> SNAPSHOTTED
  -> MIGRATING
  -> ACTIVE
  -> COMMITTED
  -> RELEASED
~~~

异常路径还包括 CANCELLED、EXPIRED、CORRUPTED 和 ROLLBACK。每个状态都要定义 owner、租约、可读写权限和可转移的字段。特别是 MIGRATING 期间，旧 worker 和新 worker 不能同时提交不同版本的 state。

## 71.22 owner 和租约

state 的 owner 不应只是一条 request id。实际系统至少需要：

~~~text
tenant
conversation
request
model revision
adapter revision
policy context
logical position
lease expiry
~~~

请求超时、客户端断开或 worker 崩溃后，租约负责回收 state。若没有租约，长任务会把 state 永久占住；若租约太短，正在运行的请求会被误回收。续租和释放操作需要幂等，避免重试造成 double free。

## 71.23 snapshot 的完整性

一个可恢复 snapshot 至少包括：

1. state tensors。
2. logical position 和 segment boundary。
3. model、adapter、position 和 kernel revision。
4. dtype、量化 scale 和 layout。
5. random sampling state。
6. provenance 和 source range。
7. checksum、创建时间和过期时间。

只保存 state tensor 可能让模型“能继续说话”，但不能保证与不间断运行相同。恢复测试应逐 token 比较 logits，允许的误差要与 dtype 和 kernel 的数值特性绑定。

## 71.24 preemption 和迁移

当高优先级请求抢占低优先级请求时，state 要在 GPU、CPU 或远端存储之间迁移。迁移成本近似为：

~~~math
T_{\mathrm{move}}
\approx
\frac{\mathrm{bytes(state)}}{\mathrm{link\ bandwidth}}
+T_{\mathrm{serialize}}
+T_{\mathrm{deserialize}}.
~~~

若 state 小，迁移可能比重新 prefill 便宜；若 state 大、链路慢或压缩/解压耗时，重算反而更快。调度器应基于实际测量做选择，并记录恢复后的质量差异。

## 71.25 batch slot 复用

连续 batching 会复用 batch slot。KV cache 通常有 page table，state cache 还需要把 slot 与 state owner 绑定。请求结束后，必须先阻断后续读写，再释放 slot，最后回收 state。释放顺序错误可能造成新请求短暂读到旧请求 state。

一个最小并发回归是：交错运行两个不同租户的长流请求，随机取消其中一个，再让新请求复用其 slot；比较单独运行与批处理运行的每个有效 token 输出。任何差异都应作为 state isolation bug 处理。

## 71.26 state 的事务语义

工具调用、推测解码和多 Agent 分支都会产生 tentative state。可以为每个 state 维护 committed step 和 speculative step：

~~~math
s_{\mathrm{visible}}
=s_{\mathrm{committed}}
+s_{\mathrm{tentative}}.
~~~

只有经过 target 验证或工具结果确认后，tentative 部分才能提交。工具失败、用户取消或安全策略拒绝时，系统应回滚到最后一个合法 committed step。输出文本回滚而 state 不回滚，会使下一轮模型基于未显示的内容继续推理。

## 71.27 state 与权限边界

递归状态可能包含用户文档、工具结果和隐式偏好。若同一 state 跨权限域继续使用，后续回答即使没有直接引用原文，也可能泄露权限变化前的信息。权限变化、文档撤销和租户切换都应触发：

1. state invalidate 或重建。
2. provenance 重新检查。
3. cache sharing 重新授权。
4. 日志和 snapshot 脱敏。

“同一会话”不能自动等于“永远可以继续使用同一状态”。

## 71.28 容量规划

state cache 容量不仅是 tensor bytes，还包括 metadata、租约和迁移副本：

~~~math
M_{\mathrm{state\ pool}}
=N_{\mathrm{active}}
\left(
M_{\mathrm{tensor}}
+M_{\mathrm{metadata}}
+M_{\mathrm{snapshot}}
+M_{\mathrm{migration}}
\right).
~~~

长运行 Agent 可能同时拥有 active state、checkpoint state 和 retry branch。若只按 active request 数估算，失败重试和迁移高峰会造成 OOM。容量模型要纳入最大 branch 数、保存频率和过期回收延迟。

## 71.29 监控和告警

建议监控：

~~~text
active state count
state bytes
snapshot bytes
lease expiry count
migration rate
restore latency
restore mismatch
cross-tenant access denial
state checksum failure
rollback rate
task success after resume
~~~

不要只看 GPU memory。state 泄露、快照积压和恢复错误可能在显存正常时发生。线上一次恢复后任务失败，应关联模型 revision、kernel、position 和 state checksum。

## 71.30 recursive state 的能力验收条件

固定大小 state 解决的是存储增长问题，不自动解决随机回读、精确引用和跨请求隔离。路由器在接受任务前应判断它需要的是连续预测、局部上下文还是任意位置证据；后两者可能需要显式 KV、RAG 或保留原文。

## 71.31 一次状态错误的定位顺序

出现长对话结果漂移时，先比较 reset 和 owner，再比较 token/position 对齐、padding 和 batch slot，之后检查 dtype、kernel 版本、snapshot 序列化和 speculative rollback。这个顺序从边界错误到数值误差，能避免一开始就重训模型。

## 71.32 state cache 的单位成本

容量不只由 state tensor 决定，还包括每请求 metadata、snapshot、租约、恢复时间和监控。可以用：

```math
C_{\mathrm{task}}
=C_{\mathrm{update}}+C_{\mathrm{snapshot}}
+C_{\mathrm{recovery}}+C_{\mathrm{verification}}.
```

当递归 state 减少显存却增加恢复和验证成本时，必须用成功任务成本判断，而不能只看 bytes/token。

## 71.33 state 和显式证据的组合

递归 state 适合携带连续任务的历史压缩表示，但审计系统仍需保存原始证据、引用位置和版本。不能把 state snapshot 当成可读的业务记录，也不能因为 state 固定大小就删除用户要求保留的文档。

## 71.34 reset 的语义

reset 不是简单把 tensor 置零，还要清除 step、position、gate 统计、owner、租约、grammar 和未提交候选。请求完成、租户删除、权限撤销和模型切换都可能要求不同程度的 reset；每种 reset 都应有测试和审计事件。

## 71.35 state cache 的容量压测

改变并发、平均序列长度、snapshot 频率、迁移比例、拒绝候选比例和 state dtype，记录 state GiB、update time、snapshot/restore 延迟、p99、错误和清理延迟。固定 state 的单请求成本很小，不代表高并发的总资源没有上限。

## 71.36 小结

Recursive State Cache 是一个有 owner、租约、版本、事务、容量和安全边界的状态系统。它的难点不是把 state 放进字典，而是保证批处理、抢占、迁移、恢复、回滚和权限变化后仍然与模型语义一致。只有完成并发隔离和恢复回归，递归架构的低内存优势才有生产意义。
