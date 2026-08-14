# 第 62 章 KV/State Resource Model：把历史表示变成可调度资源

## 62.1 为什么统一叫 cache 会误导

推理服务里可能同时存在显式 KV、latent cache、递归 state、prefix cache、候选 state 和临时 workspace。它们都被叫作 cache，但大小、生命周期、共享条件和回滚语义不同。

容量规划若把所有对象相加成一个模糊的“缓存占用”，调度器就无法回答：哪个请求可以抢占？哪部分可以重算？哪些数据能跨请求共享？取消时要释放什么？

## 62.2 一个资源向量

对请求 `i`，可以用资源向量描述：

```math
r_i=(m_{kv},m_{state},m_{temp},m_{meta},c_{prefill},c_{decode},n_{pages})
```

其中包含显式 KV、递归/latent state、临时 workspace、metadata、prefill/decode 计算和 page 数量。集群容量是所有活跃请求资源向量的和，加上权重、通信和保留余量。

## 62.3 显式 KV 的估算

```math
m_{kv}=2BLTH_{kv}d_hb
```

实际分配通常按 block/page 对齐。若 block size 为 `q`，请求有效 token 为 `T`，分配 token 数是 `q\lceil T/q\rceil`，最后一个 block 的内部浪费约为：

```math
w=q\lceil T/q\rceil-T
```

大量短请求会放大 page metadata 和碎片；长请求会放大 payload。allocator 需要同时监控有效 token 和分配 token。

## 62.4 state 的不同生命周期

递归 state 每个请求可能固定大小，但必须按 step 更新；latent cache 可能按 token增长，也可能按压缩块保存；candidate state 只在 speculative 期间存在；prefix cache 通过引用计数共享。

这些对象不能都采用“低水位释放”。candidate state 可以在 reject 后立即释放，prefix cache 需要考虑共享租约，递归 state 需要在任务恢复前 snapshot，外部 artifact 则不能因为 GPU cache 释放而删除。

## 62.5 admission 和 preemption

调度器在接纳请求时估算最坏 token、输出、工具轮次和 state。内存压力出现时，可以对可重算的 KV 做 recompute，对可迁移的 state 做 swap，对低优先级请求排队；不能随意丢掉工具已提交状态。

一个资源安全条件为：

```math
\sum_i r_i+m_{weights}+m_{comm}+m_{reserve}
\le m_{usable}
```

`m_reserve` 应覆盖碎片、kernel workspace、取消和故障恢复，而不是设为 0。

## 62.6 prefix sharing 的引用计数

共享 prefix 时，多个请求指向同一 cache block。每个 block 要有 refcount、租户、model/template revision、position range 和失效时间。一个请求续写时只创建新 block；任何请求释放都不能删除仍被引用的 prefix。

如果 prefix 内容含有敏感文档，租户权限必须成为 cache key 的一部分。相同 token 序列不代表可以跨用户共享。

## 62.7 speculative 的临时资源

draft token、verify logits、临时 KV/latent、grammar state 和 stream buffer 都是临时资源。accepted prefix 提交后释放其余部分，reject 时回滚。监控中要区分 committed token 和 candidate token，否则会高估使用量并在 OOM 后无法解释。

## 62.8 worked example：为什么会“看似有显存却无法接入"

一张 GPU 显示还有 10 GiB 空闲，但一个新请求需要 6 GiB 连续 workspace、2 GiB KV 和多个 block。由于碎片、通信 buffer 和 CUDA graph 预留，实际可用空间不足，admission 失败。

解决办法可能是整理/分级 allocator、降低 block、迁移可重算请求或把长请求路由专池，而不是简单提高显存利用率阈值。

## 62.9 观测和诊断

每个请求记录 page count、valid tokens、allocated tokens、KV/state/temp/metadata bytes、cache hit、preemption、recompute、swap、rollback 和释放时间。聚合看 p50/p95/p99、碎片率和每个 layer/cache type 的占用。

如果 OOM 只出现在 speculative 或工具 JSON，先查临时 state 和 parser buffer；如果只在取消后出现，查释放路径和引用计数；如果只在恢复后出现，查 snapshot 与旧 cache 是否重复计入。

## 62.10 常见失败

把所有 cache 当同一对象；容量只按 token 不按 page；忽略临时 candidate；prefix cache 无租户隔离；state release 不完整；preemption 丢失外部状态；用平均占用做 admission；没有碎片和 workspace 余量。

## 62.11 面试回答与练习

回答“如何做 KV cache 管理”时，应区分 KV、state、prefix、candidate 和 workspace，设计 page、owner、refcount、生命周期、admission、preemption、recompute 和安全 key；容量和指标要按有效/分配 token、层和请求 bucket 分析。

练习一：给定 block size 和多个请求长度，计算 page waste。

练习二：设计 speculative reject 的资源释放断言。

练习三：解释为什么空闲显存不等于可接纳请求。

### 62.11.1 资源对象的分类

KV cache、latent cache、recursive state、speculative candidate、activation workspace 和通信 buffer 都可能占用 GPU，但生命周期不同。资源模型应为每个对象记录 owner、request、layer、dtype、bytes、reference、状态和回收条件。

```text
resource = (type, owner, revision, shape, bytes,
            state, lease, evict_policy)
```

把所有对象都叫 cache，会让 allocator 无法判断哪些可以重算、哪些必须迁移、哪些包含敏感租户数据。

### 62.11.2 page waste 和有效容量

若 block size 为 `B`，请求实际需要 `T` 个 token，则分配页数为 `ceil(T/B)`，内部浪费近似为：

```math
W_{\mathrm{page}}=\lceil T/B\rceil B-T
```

小 block 减少浪费但增加 metadata 和调度开销，大 block 适合长连续请求却可能降低短请求利用率。长度分布、prefix sharing 和 preemption 一起决定最优值。

### 62.11.3 speculative 和混合 state 的事务

候选 token 生成时，KV/state 应进入临时区；target 接受后把 accepted prefix 提交，拒绝后释放或回滚。断言要检查正式 cache 的 token 数、position、block refcount、state checksum 和 streaming 输出一致。任何 rejected token 进入正式 prefix cache 都可能造成后续请求错误命中。

### 62.11.4 安全与租户隔离

cache key 需要包含 model/template/position/dtype 和 tenant namespace。共享 prefix 可以节省计算，但不能让一个租户的私有文档或工具结果被另一个租户命中。回收和 swap 日志也要避免泄露原始 token，必要时使用加密或只记录 hash。

这里的工程取舍是资源效率与隔离强度之间的平衡：更积极的 prefix sharing 和更大的共享池可以减少重复计算、提高吞吐，却会增加引用计数、租户隔离、失效传播和回收审计的复杂度；按租户严格分池更容易证明安全，但可能牺牲命中率和显存利用率。资源模型必须把这两类成本同时计入 admission 和容量规划。

## 62.12 资源对象的事务和租户隔离

KV、latent 和 recursive state 都应有 owner、版本、step、生命周期、租约和回收状态。prefix sharing 的引用计数、speculative 临时块和 preemption swap 不能互相覆盖。

高并发测试要改变请求加入、完成、取消和重排顺序，并比较单请求基线。跨租户命中 cache 或恢复 state 是硬安全问题，不是普通性能回归。

## 62.13 碎片和状态生命周期

KV page、latent、recursive state 和 workspace artifact 的生命周期不同。请求取消、抢占、prefix sharing、回滚和迁移会造成空洞、临时副本和 metadata 开销。容量模型必须包含碎片和 snapshot，而不是只算有效 token。

allocator 要有 owner、revision、logical position、dtype、page/state schema 和 checksum。释放对象时同时删除物理存储、索引和租约。

## 62.14 状态调度和公平性

长流状态占用时间久，短请求可能在队列中等待；state 迁移可以提高利用率，却需要通信和一致性。scheduler 可按 state size、剩余输出、优先级和 preemption 成本做 admission，但要避免饥饿。

不同架构的状态不能用统一 token 数表示。显式 KV、latent cache 和递归 state 要在调度器中有不同的增长、回滚和恢复语义。

## 62.15 三类状态不能用同一条释放规则

显式 KV 通常随 token 线性增长，local window 可以回收较早位置，latent cache 可能按压缩表示增长，recursive state 则可能保持固定大小但对顺序和 checkpoint 极其敏感。把它们都叫 cache，会导致 allocator 只看字节数而忽略恢复语义。

可以为每个状态对象保存：

~~~text
owner, tenant, model_revision
logical_range, physical_pages
schema, dtype, checksum
created_step, last_used_step
refcount, lease_expiry
rollback_handle, eviction_policy
~~~

状态对象从 active 变成 evictable 前，还要确认没有流式客户端等待、没有工具调用依赖、没有 speculative 子事务引用，也没有被 checkpoint 或 prefix cache 引用。

## 62.16 状态预算与碎片预算

有效容量不等于物理空闲显存。假设 page 大小为 B、请求实际使用 T 个 token，内部浪费为：

~~~math
W_{\mathrm{page}}
=\left\lceil\frac{T}{B}\right\rceil B-T.
~~~

还要加入不同 block size、不同 dtype、跨层布局、空闲 page 不能合并以及 metadata 的碎片。一个资源模型可以写成：

~~~math
M_{\mathrm{physical}}
=M_{\mathrm{logical}}
+M_{\mathrm{page\ waste}}
+M_{\mathrm{metadata}}
+M_{\mathrm{migration}}
+M_{\mathrm{workspace}}.
~~~

因此 admission 要同时检查 logical state budget 和 physical free page。只有前者会造成“逻辑上有容量、分配时 OOM”；只有后者又可能在高碎片时接纳过多请求。

## 62.17 prefix sharing 的引用与租户边界

共享 prefix 时，父 page 只有在所有子请求都释放后才能回收。引用计数更新必须和请求提交、取消、回滚原子发生；否则一个请求取消可能把仍被其他请求使用的 page 标记为空闲。

cache key 还应包含模型 revision、tokenizer、template、position scheme、dtype、租户和权限范围。相同的文本前缀不等于相同的安全上下文：私有检索结果、系统提示和工具权限不能因为字符串相同就跨租户共享。

评估 prefix cache 不能只看 hit rate，还应看命中后的证据权限、失效传播、回收延迟、跨版本误命中和内存占用。安全错误的代价远大于一次 cache miss。

## 62.18 迁移、抢占和递归 state

状态迁移的对象不是一段无意义的字节，而是带有 logical position 和模型 schema 的历史表示。迁移前要冻结写入，记录 checksum 和 version；迁移后重新校验，并用同一后缀继续生成，比较 token 和 state checksum。

抢占时可以选择 recompute、swap 或直接拒绝。显式 KV 适合按 page 迁移，递归 state 可能需要整段 snapshot，latent state 还要绑定压缩器版本。若 state schema 变化，旧 snapshot 不能静默解释成新版本。

## 62.19 状态账本的一个具体计算

假设一张卡有 `M_free` 字节可用于状态，page 大小为 `P` 字节，当前有三个请求。请求 A 逻辑上需要 10.2 个 page，请求 B 需要 4.1 个 page，请求 C 处于 speculative 阶段，正式部分需要 3 个 page、临时候选需要 2 个 page。物理分配必须向上取整，并把临时区和 metadata 算进去：

```math
P_{used}
=\lceil10.2\rceil+\lceil4.1\rceil+3+2+P_{meta}.
```

如果 C 的候选被拒绝，只有临时 2 个 page 和对应 grammar/state 可以释放；A、B 的 page 不能因为同一轮结束而回收。若 A 和 B 共享一个 prefix，父 page 的 refcount 还要保持为 2。这个小例子说明 allocator 必须理解逻辑 token、物理 page、临时事务和引用计数，不能只做一个字节加法。

## 62.20 状态迁移与故障回放

worker 重启时，状态恢复要先检查 model revision、tokenizer/template、position scheme、dtype、state schema 和 logical position。显式 KV 可以按 page 读取，递归 state 可能需要整段 snapshot；如果 schema 不兼容，选择重算通常比静默加载安全。

故障回放要覆盖：请求在 page 分配后取消、prefix refcount 更新中断、speculative commit 前断电、swap 完成但索引未更新、cache 命中后租户权限变化。每个回放检查输出 token、stream sequence、page refcount、checksum、租约和最终释放状态。物理显存没有泄漏还不够，错误 state 被另一个租户命中同样是失败。

## 62.21 资源审计的基础结论

状态资源实验至少覆盖并发加入、取消、prefix sharing、speculative 接受/拒绝、preemption、worker 重启、跨版本发布和跨租户访问。每一步检查 token 数、position、page refcount、schema、checksum、stream event 和最终输出。

## 62.22 三类状态不能使用同一释放规则

显式 KV、递归 state 和 speculative 临时 state 的生命周期不同。显式 KV 通常随 token 追加并在请求结束时释放；递归 state 可能在每个 token 更新固定大小的表示；speculative state 只在 target 验证前存在，接受路径提交、拒绝分支回收。把三者都叫 cache，再使用一条“请求结束释放”的规则，会造成过度占用或错误复用。

| 状态 | 典型生命周期 | 释放/提交条件 |
| --- | --- | --- |
| 正式 KV | request active | 请求结束、取消或抢占重算 |
| 递归 state | chunk/token 更新 | schema 兼容且 checkpoint 提交 |
| 临时 speculative | 一轮候选 | 验证提交或拒绝回收 |
| 共享 prefix | 多请求引用 | refcount 归零且权限有效 |

资源账本应把 logical state 和 physical allocation 分开记录。逻辑上只增加了几个 token，物理上可能新分配 page、metadata、grammar state 和 copy-on-write 缓冲。

## 62.23 reservation、抢占和公平性

调度器要在状态分配前预留足够空间，并为不同租户和请求类型设置公平规则。长请求如果无限占用 state，会让短请求饥饿；只按请求数公平，又可能让一个长上下文消耗掉多数显存。可按 active tokens、state bytes、等待时间和优先级联合计量。

抢占策略有三种典型选择：保留 state 并等待、把 state swap 到 CPU/远端存储、释放后重算。每种选择都要记录恢复成本和 SLO 影响。对高优先级短请求，释放长请求的临时 speculative state 可能比抢占正式 KV 更合理；对将近完成的请求，重算又可能比 swap 更贵。

## 62.24 状态事务的提交顺序

一次 prefix sharing 或 speculative commit 需要同时更新 page、引用计数、logical position、stream event 和审计记录。建议遵循：先分配/增加引用，再写入新状态，完成校验后提交索引，最后释放旧引用。任一步失败，都要有可重放的 rollback。

如果先减少旧 page 的 refcount，再发现新 page 校验失败，其他请求可能已经看不到共享前缀；如果先广播 stream event，再回滚 token，客户端会收到不可撤回的错误序列。状态事务和协议事件必须在设计上对齐。

## 62.25 跨租户和跨版本的安全边界

相同 token prefix 不等于相同上下文。系统提示、用户身份、检索权限、工具 allowlist 和 policy revision 都可能不同。prefix sharing 只能在安全证明覆盖的边界内启用；共享缓存的 key 至少要绑定模型、tokenizer、模板、position、租户和权限域。

跨版本迁移时，state schema、dtype、量化、adapter 和 kernel layout 任何一项改变都可能使旧 state 无法解释。安全默认是排空或重算，而不是静默加载。性能上的 cache miss 是可接受的，跨租户误命中和错误继续生成则是发布阻断项。

## 62.26 状态资源的可观测性

仅监控 GPU memory used 不足以解释 state 问题。至少记录 logical tokens、allocated pages、page waste、refcount、temporary bytes、swap/recompute 次数、preemption reason、cache hit/miss、state schema、租户和请求优先级。指标要能从请求 trace 追到具体 allocator 事件。

一个有用的诊断顺序是：先看请求是否被正确 admission，再看 reservation 是否成功，再看物理分配和碎片，再看 state 生命周期，最后看输出和协议。这样能区分容量不足、碎片、泄漏、错误共享和模型本身的问题。

## 62.27 资源评测的故障矩阵

资源模型要在正常和异常状态都通过：并发加入、取消、prefix sharing、speculative 接受/拒绝、preemption、worker 重启、对象存储故障、跨版本发布和跨租户访问。每个 case 检查 token、position、page refcount、schema、checksum、stream event、最终输出和释放状态。

测试通过不代表实际负载一定安全，但没有这些回放，就无法知道 allocator 在状态竞争中会不会泄漏或复用错误。容量压测还应包含长短混合请求和 burst，而不是只重复相同长度。

## 62.28 资源审计与证据层级

KV/State Resource Model 的核心不是给所有历史表示起一个统一名字，而是把它们放进可调度、可回滚、可审计的资源模型，同时保留各自的生命周期和安全边界。可参考 vLLM PagedAttention、SGLang cache/state 和 GPU serving 文档；具体 allocator、block size 和 state layout 以引擎版本为准。

任何“能省多少显存”的结论都要说明状态类型、序列长度、共享比例、dtype、硬件和是否计入临时 buffer。显式 KV、latent state、递归 state 和 speculative artifact 的公式可以互相对照，但不能把一种状态的测量直接当成另一种状态的实现事实。

## 62.29 state allocator 的对象模型

资源管理器不应只返回一块连续显存。每个 KV、latent、recursive 或 speculative state 都要有 owner、表示类型、shape、dtype、位置范围、版本、租约、可重算标记和回滚指针。这样调度器才能在压力下选择淘汰、迁移、重算或拒绝，而不是随机释放一块 page。

## 62.30 碎片和共享的真实代价

prefix sharing 可以减少重复存储，却增加引用计数、失效传播和租户隔离复杂度；小 block 减少内部浪费，却增加 metadata 和 page table；大 block 便于 kernel，却可能让短请求占用过多空间。block size 应用长短混合、共享比例、迁移和抢占压测决定。

## 62.31 state 资源的回滚测试

测试普通追加、拒绝候选、取消请求、抢占恢复、跨 worker 迁移、cache 失效、租户删除和 OOM。断言不仅是显存回收，还要检查 committed prefix、输出、grammar、usage、owner 和审计是否一致。任何“内存回收了但状态错了”的情况都应视为严重失败。

## 62.32 小结

KV、latent、recursive 和 speculative state 都是有 owner、schema、生命周期和回滚语义的资源。资源模型要同时解释容量、碎片、共享、跨版本和跨租户边界；只有 allocator、协议事件和故障回放一致，显存节省才不会换来错误输出或数据泄露。
