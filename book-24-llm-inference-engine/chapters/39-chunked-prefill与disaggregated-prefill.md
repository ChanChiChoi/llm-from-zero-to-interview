# 第 39 章 Chunked Prefill 与 Disaggregated Prefill

上一章讲了 KV Cache 迁移、共享和路由：PD 分离真正难的地方，是让 Decode worker 正确、高效地使用 Prefill worker 生成的 KV Cache。

本章继续讨论 Prefill 侧的另一个关键问题：长 prompt 怎么处理。

在真实业务中，prompt 可能很长。RAG、Agent、多轮对话、代码生成、长文档问答都会把输入上下文拉得很长。如果一次性把长 prompt 全部 prefill 完，就可能阻塞 decode、拉高 TTFT、造成显存峰值和调度抖动。

Chunked Prefill 和 Disaggregated Prefill 都是在解决这个问题，但它们不是同一个概念。

一句话概括：

> Chunked Prefill 是把长 prompt 的 prefill 切成多个 chunk 调度；Disaggregated Prefill 是把 prefill 作为独立资源池或独立阶段来服务。前者是执行粒度优化，后者是系统架构优化，二者可以结合。

## 39.1 本章目标

读完本章，你应该能讲清：

1. 为什么长 prompt prefill 会成为 serving 瓶颈。
2. Chunked Prefill 解决什么问题。
3. Chunked Prefill 和普通 prefill 的区别。
4. Disaggregated Prefill 和 PD 分离是什么关系。
5. Chunked Prefill 如何影响 TTFT、TPOT 和吞吐。
6. Chunked Prefill 如何和 continuous batching 结合。
7. Chunked Prefill 如何和 KV transfer 结合。
8. 面试中如何区分 Chunked Prefill、Disaggregated Prefill 和 PD 分离。

## 39.2 长 Prompt Prefill 的问题

先看一个普通请求：

```text
prompt length = 128 tokens
max output = 512 tokens
```

Prefill 很短，主要成本在后续 decode。

再看长上下文请求：

```text
prompt length = 32000 tokens
max output = 512 tokens
```

这时 prefill 本身就是一个很大的计算任务。

它会带来：

1. 长时间占用 GPU compute。
2. 一次性写入大量 KV cache。
3. 形成很大的 activation 和 attention kernel 压力。
4. 让短请求排队。
5. 打断正在稳定 decode 的 batch。
6. 拉高整体 TTFT。
7. 放大 tail latency。

在 unified engine 中，长 prompt prefill 的典型问题是：

```text
t0: decode batch 正在稳定输出
t1: 长 prompt 请求到达
t2: scheduler 调度长 prefill
t3: decode batch 被挤占或延迟
t4: 所有 streaming 用户都感觉卡顿
```

这就是前面章节说过的 prefill interruption。

## 39.3 为什么不能总是一次性 Prefill

一次性 prefill 的流程很简单：

```text
take full prompt
  -> run one large prefill forward
  -> produce all prompt KV
  -> enter decode
```

它的优点是：

1. 实现简单。
2. KV 生成一次完成。
3. 状态机简单。
4. 没有 chunk 边界管理。

但对长 prompt，它的问题很明显：

1. 单次计算时间长。
2. 不能灵活插入 decode step。
3. 容易造成调度饥饿。
4. batch shape 可能很极端。
5. 显存峰值更高。
6. TTFT 更容易被长 prompt 主导。

所以需要一种折中：

```text
不要一次吃完整个 prompt，而是分块吃。
```

这就是 Chunked Prefill。

## 39.4 什么是 Chunked Prefill

Chunked Prefill 指的是把一个长 prompt 拆成多个 token chunk，分多次 prefill 执行。

例如：

```text
prompt length = 32000
chunk size = 4096

chunk 0: tokens [0, 4096)
chunk 1: tokens [4096, 8192)
chunk 2: tokens [8192, 12288)
...
chunk 7: tokens [28672, 32000)
```

执行时不是一次性跑 32000 tokens，而是：

```text
run prefill chunk 0
write KV for chunk 0
run prefill chunk 1
write KV for chunk 1
...
run prefill final chunk
write KV for final chunk
then enter decode
```

注意：Chunked Prefill 不是把模型层切开。

它切的是输入 token 序列。

每个 chunk 仍然要经过完整 Transformer 层。

## 39.5 Chunked Prefill 的直觉

可以把长 prompt prefill 想成一个很大的任务。

一次性 prefill 是：

```text
[================ long prefill ================][decode][decode][decode]
```

Chunked Prefill 是：

```text
[prefill chunk][decode][prefill chunk][decode][prefill chunk][decode]...
```

或者在 prefill pool 中：

```text
[prefill chunk A][prefill chunk B][prefill chunk C]...
```

它的核心价值是把一个大任务拆成多个可调度的小任务。

这样 scheduler 可以在 chunk 边界做决策：

1. 是否插入 decode step。
2. 是否调度短 prompt。
3. 是否切换 batch 组合。
4. 是否暂停某个长请求。
5. 是否进行资源让渡。

所以 Chunked Prefill 的本质是调度粒度优化。

## 39.6 Chunked Prefill 解决什么问题

Chunked Prefill 主要解决四类问题。

第一，降低 prefill 对 decode 的长时间阻塞。

长 prefill 被切成多个 chunk 后，decode step 可以在 chunk 间插入。

第二，改善调度公平性。

短请求不必一直等一个超长 prompt prefill 完。

第三，控制显存和 batch 形态。

每次只处理一段 prompt，可以降低单次峰值压力。

第四，给 PD 分离和 KV transfer 提供更细粒度的流水。

Prefill 侧可以逐块产生 KV，Decode 侧或远端 KV cache 可以逐块接收。

## 39.7 Chunked Prefill 不解决什么问题

Chunked Prefill 不是万能的。

它不直接解决：

1. P/D 资源池是否分离。
2. KV cache 如何跨 worker 迁移。
3. Decode worker 如何选。
4. 远端 KV cache 如何管理。
5. 多租户资源隔离。
6. 网络带宽瓶颈。

它只是把 prefill 执行切成 chunk。

如果系统仍然是 unified engine，P/D 仍然会共享同一组 GPU，只是 prefill interruption 被缓解。

如果系统是 PD 分离，Chunked Prefill 可以让 prefill pool 内部调度更细，也可以让 KV transfer 更流水化。

## 39.8 Chunked Prefill 和 Continuous Batching

Continuous batching 的核心是：每个 step 动态组合请求，而不是固定一个 batch 从头跑到尾。

在 decode 阶段，它通常表现为：

```text
step 0: req A, B, C
step 1: req A, B, C, D
step 2: req A, C, D
```

Chunked Prefill 加入后，scheduler 面对的不只是 decode step，还有 prefill chunk。

它要在下面几类工作之间做选择：

1. 新请求的 prefill chunk。
2. 长请求的后续 prefill chunk。
3. 已进入生成阶段的 decode step。
4. KV transfer completion 后等待 decode 的请求。

调度器可以形成这样的时间线：

```text
t0: decode step for running requests
t1: prefill chunk for long request L
t2: decode step for running requests
t3: prefill chunk for short request S
t4: decode step for running requests
t5: prefill chunk for long request L
```

这就是把 prefill chunk 当成可调度单元。

## 39.9 Chunk Size 的影响

Chunk size 是 Chunked Prefill 的核心参数。

如果 chunk size 太大：

1. 单个 chunk 仍然会阻塞 decode。
2. 调度粒度不够细。
3. 长请求仍然容易造成尾延迟。

如果 chunk size 太小：

1. kernel launch 和调度开销增加。
2. attention 计算效率可能下降。
3. KV block 管理更频繁。
4. transfer metadata 更多。
5. 总吞吐可能下降。

所以 chunk size 是折中：

```text
large chunk:
  better compute efficiency
  worse latency fairness

small chunk:
  better scheduling fairness
  more overhead
```

生产系统通常会根据：

1. 模型大小。
2. GPU 类型。
3. prompt 长度分布。
4. decode latency SLO。
5. KV transfer 带宽。
6. batch scheduler 策略。

来选择 chunk size。

## 39.10 Chunked Prefill 对 TTFT 的影响

TTFT 是 time to first token。

对一个长 prompt 请求来说，first token 必须等完整 prompt 的 KV 都准备好，才能开始生成。

所以 Chunked Prefill 不一定降低这个请求自己的理论最短 TTFT。

因为它仍然要处理完整 prompt。

但它会改善系统级 TTFT：

1. 短请求不用被长请求一直挡住。
2. Decode step 可以在 chunk 间执行。
3. Scheduler 可以更公平地服务不同请求。
4. 尾延迟会更稳定。

换句话说：

```text
Chunked Prefill 不会让 32000 token 的计算凭空消失，
但会让这 32000 token 的计算不再一口气堵住整个系统。
```

## 39.11 Chunked Prefill 对 TPOT 的影响

TPOT 是 time per output token。

在 unified engine 中，长 prefill 会打断 decode，导致 TPOT 抖动。

Chunked Prefill 可以让 decode step 更频繁地插入，从而降低 TPOT tail latency。

例如没有 chunking：

```text
decode step
decode step
[long prefill 800ms]
decode step delayed
```

有 chunking：

```text
decode step
[prefill chunk 100ms]
decode step
[prefill chunk 100ms]
decode step
```

TPOT 的峰值会更低。

但 chunk 太小也可能因为调度开销过大而让整体 TPOT 变差。

所以仍然是折中。

## 39.12 Chunked Prefill 对吞吐的影响

从吞吐角度看，Chunked Prefill 有两面性。

正向影响：

1. batch 组合更灵活。
2. GPU 空洞可能减少。
3. 短请求可以更快进入系统。
4. Decode 不容易被长 prefill 长时间饿死。

负向影响：

1. 更多 scheduler 决策。
2. 更多 kernel launch。
3. 更多 KV metadata 更新。
4. 更复杂的 attention mask / position 管理。
5. 可能降低大矩阵计算效率。

所以 Chunked Prefill 的目标不是“无条件提高吞吐”。

更准确地说：

> Chunked Prefill 用一定 overhead 换更好的延迟公平性和更可控的调度。

## 39.13 Chunked Prefill 的状态机

一个支持 Chunked Prefill 的请求，不再只有 `WAITING -> PREFILLING -> DECODING`。

它可能是：

```text
WAITING
  -> PREFILL_CHUNK_RUNNING
  -> PREFILL_CHUNK_DONE
  -> PREFILL_PAUSED
  -> PREFILL_CHUNK_RUNNING
  -> PREFILL_DONE
  -> DECODING
  -> FINISHED
```

需要维护的字段包括：

1. `request_id`。
2. `prompt_len`。
3. `chunk_size`。
4. `next_token_offset`。
5. `prefilled_token_count`。
6. `kv_block_table`。
7. `position_offset`。
8. `attention_state`。
9. `priority`。
10. `deadline`。

其中 `next_token_offset` 很关键。

它表示下一次 prefill chunk 从 prompt 的哪个 token 开始。

## 39.14 Position 和 Attention Mask

Chunked Prefill 不能破坏模型语义。

虽然 prompt 被切成多个 chunk，但对模型来说，它必须等价于一次性处理完整 prompt。

这意味着：

1. position id 必须连续。
2. RoPE 位置必须正确。
3. chunk 后面的 token 必须能 attend 到前面 chunk 的 KV。
4. causal mask 必须保持正确。
5. KV cache 写入位置必须对应全局 token offset。

例如：

```text
chunk 0: tokens [0, 4096), positions [0, 4096)
chunk 1: tokens [4096, 8192), positions [4096, 8192)
```

chunk 1 不能把 position 从 0 重新开始。

否则模型语义就错了。

## 39.15 KV Block 写入

Chunked Prefill 会逐步写入 KV block。

例如 block size 是 16 tokens：

```text
chunk 0 writes blocks 0..255
chunk 1 writes blocks 256..511
chunk 2 writes blocks 512..767
```

Scheduler 需要确保：

1. 已写入的 KV block 不被错误释放。
2. 后续 chunk 能找到前面的 KV。
3. block table 持续更新。
4. chunk 边界不破坏 block 边界。
5. 最后一个 chunk 可能不是完整 block。

在 paged KV cache 中，chunked prefill 的本质是：

```text
incrementally extend the request's KV block table
```

## 39.16 Chunk 边界和 Block 边界

chunk size 和 KV block size 不一定相同。

例如：

```text
chunk size = 4096 tokens
block size = 16 tokens
```

这时一个 chunk 对应 256 个 blocks。

如果：

```text
chunk size = 3000
block size = 16
```

那么 chunk 边界可能落在 block 中间。

这会带来额外复杂度：

1. partial block 写入。
2. block offset 记录。
3. transfer 时的非整块处理。
4. cache reuse 时的边界匹配。

为了简化实现，系统可能倾向于让 chunk size 是 block size 的整数倍。

但这不是必须，只是工程上更方便。

## 39.17 Disaggregated Prefill 是什么

Disaggregated Prefill 字面意思是“分离式 prefill”。

它通常指把 prefill 阶段从统一 serving engine 中拆出来，使用独立的 prefill workers 或 prefill pool 来处理输入阶段。

典型结构：

```text
client
  -> router
  -> prefill pool
       run prefill
       produce KV
  -> decode pool
       receive KV
       generate tokens
```

这基本就是 PD 分离中的 P 侧。

有时文档里会把 Disaggregated Prefill 和 PD Disaggregation 混用。

但为了理解清楚，可以这样区分：

1. Disaggregated Prefill 强调 prefill 从主 decode engine 中拆出来。
2. PD 分离强调 prefill pool 和 decode pool 都作为独立资源池协作。
3. Chunked Prefill 强调 prefill 内部按 token chunk 切分执行。

## 39.18 Chunked Prefill vs Disaggregated Prefill

二者不是同一层面的概念。

| 维度 | Chunked Prefill | Disaggregated Prefill |
| --- | --- | --- |
| 关注点 | 执行粒度 | 系统架构 |
| 切分对象 | prompt tokens | prefill 阶段和资源池 |
| 是否必须跨 worker | 不必须 | 通常是 |
| 主要收益 | 减少长 prefill 阻塞 | P/D 资源隔离和独立扩缩容 |
| 主要代价 | 调度和状态更复杂 | KV transfer 和分布式状态更复杂 |

简单说：

```text
Chunked Prefill:
  how to run a long prefill

Disaggregated Prefill:
  where to run prefill
```

二者可以组合：

```text
run chunked prefill on disaggregated prefill pool
```

## 39.19 三种架构对比

第一种：Unified engine + normal prefill。

```text
same worker:
  full prefill
  decode
```

特点：实现简单，但长 prefill 容易打断 decode。

第二种：Unified engine + chunked prefill。

```text
same worker:
  prefill chunk
  decode step
  prefill chunk
  decode step
```

特点：缓解 prefill interruption，但 P/D 仍然共享资源。

第三种：PD disaggregation + chunked prefill。

```text
prefill pool:
  prefill chunk
  prefill chunk
  prefill chunk

decode pool:
  receive KV
  decode step
  decode step
```

特点：P/D 资源隔离，同时 prefill 内部也有更细粒度调度，但 KV transfer 和状态最复杂。

## 39.20 Chunked Prefill 和 KV Transfer

在 PD 分离中，Chunked Prefill 会影响 KV transfer。

两种策略：

第一，全部 prefill 完再 transfer。

```text
prefill chunk 0
prefill chunk 1
...
prefill chunk N
transfer all KV
decode
```

优点：

1. 状态简单。
2. transfer 只做一次或少数几次。
3. decode 接收完整 prompt KV。

缺点：

1. 不能流水化。
2. TTFT 包含完整 prefill 和完整 transfer。
3. source KV 在 prefill worker 保留时间长。

第二，边 prefill 边 transfer。

```text
prefill chunk 0 -> transfer chunk 0
prefill chunk 1 -> transfer chunk 1
...
prefill chunk N -> transfer chunk N
decode
```

优点：

1. prefill 和 transfer 可以重叠。
2. 降低最后一次集中 transfer 的等待。
3. 更适合长 prompt。

缺点：

1. transfer metadata 更复杂。
2. decode side 要接收 partial KV。
3. 失败清理更复杂。
4. chunk 完整性和顺序要保证。

## 39.21 流水化 Prefill 和 Transfer

边 prefill 边 transfer 可以形成流水线：

```text
time --->

prefill worker: [chunk0][chunk1][chunk2][chunk3]
transfer:          [kv0]  [kv1]  [kv2]  [kv3]
decode worker:       receive receive receive ready
```

这样 final chunk 完成后，前面的 KV 可能已经传完。

TTFT 从：

```text
full_prefill_time + full_transfer_time
```

变成近似：

```text
max(prefill_pipeline_time, transfer_pipeline_time) + final_sync
```

但流水化也要求系统支持：

1. Partial KV metadata。
2. Chunk completion event。
3. Destination block incremental reservation。
4. Transfer ordering。
5. Abort cleanup。
6. Backpressure。

如果实现不成熟，流水化可能带来更多 bug。

## 39.22 Decode 能不能提前开始

一个常见问题是：Prompt 还没全部 prefill 完，decode 能不能开始？

一般自回归生成不能在完整 prompt KV 准备好之前生成第一个新 token。

因为第一个输出 token 需要 attend 到完整 prompt。

所以通常：

```text
all prompt chunks ready
  -> first decode token
```

但有一些特殊场景可以做 overlap 或 speculative-like 优化，例如：

1. Prompt 后半部分不影响某些中间处理。
2. 模型或任务支持分段编码和后续融合。
3. 使用特定近似算法。
4. 多阶段检索或工具调用把上下文逐步注入。

这些不是标准 LLM serving 的默认语义。

所以面试回答要稳：

> 对普通 causal LM，请求进入 decode 前需要完整 prompt 的 KV。Chunked Prefill 可以让 prefill 和调度/transfer 流水化，但不能随意在 prompt 未完整处理时生成首 token。

## 39.23 Prefill Chunk 的调度优先级

Scheduler 要决定下一个执行什么。

可选工作包括：

1. Decode step。
2. 新短请求的 full prefill。
3. 长请求的下一个 prefill chunk。
4. 已等待很久请求的 prefill chunk。
5. 高优先级租户请求。

一种简单策略：

```text
if decode step deadline close:
  run decode
elif high priority short prefill exists:
  run short prefill
elif long prefill chunk has waited too long:
  run next chunk
else:
  maximize batch efficiency
```

真实系统会更复杂。

关键是不要让长 prompt 饿死短请求，也不要让长 prompt 自己永远跑不完。

这需要公平性机制。

## 39.24 Fairness：短请求和长请求

没有 Chunked Prefill 时，长请求一旦开始 prefill，短请求只能等。

有 Chunked Prefill 后，scheduler 可以在 chunk 间插入短请求。

但如果过度偏向短请求，长请求可能一直被切碎和延后。

这叫 starvation。

常见公平性策略包括：

1. Waiting time boost。
2. Per-tenant quota。
3. Max prefill chunks per scheduling window。
4. Decode deadline protection。
5. Long request progress guarantee。

例如：

```text
score = base_priority + waiting_time_bonus + progress_bonus
```

对长请求来说，`progress_bonus` 可以避免它一直只完成前几个 chunk。

## 39.25 Backpressure

Chunked Prefill 和 PD 分离结合时，backpressure 很重要。

如果 prefill pool 很快，decode pool 或 transfer backend 很慢，就会出现：

```text
prefill chunks produced faster than transfer/decode can consume
```

后果：

1. Prefill worker 上 source KV 堆积。
2. Transfer queue 变长。
3. Decode worker reserved blocks 增多。
4. GPU memory 被未消费 KV 占满。
5. TTFT 反而变差。

所以 prefill scheduler 不能只看自己空不空。

它还要看：

1. Pending transfer 数。
2. Decode pool KV capacity。
3. Router 端 request backlog。
4. Source KV retention time。
5. Transfer backend 带宽。

如果下游拥塞，上游 prefill 要限速。

## 39.26 Chunked Prefill 的 Metadata

支持 chunking 后，metadata 会增加。

一个请求需要维护：

```text
request_id
prompt_token_ids
prompt_len
chunk_size
num_chunks
current_chunk_index
next_token_offset
prefilled_token_count
kv_block_table
chunk_to_block_mapping
position_offset
transfer_state_per_chunk
deadline
priority
```

如果结合 PD transfer，还需要：

```text
source_worker_id
destination_worker_id
source_blocks_per_chunk
destination_blocks_per_chunk
chunk_transfer_status
chunk_checksum_or_version
```

这说明 Chunked Prefill 不是只改一个 for loop。

它会影响 scheduler、KV manager、transfer backend 和 request state machine。

## 39.27 Chunked Prefill 的失败处理

失败可能发生在多个阶段。

第一，某个 prefill chunk 失败。

处理方式：

1. 释放该请求已分配但不再使用的 KV。
2. 判断是否能从上一个成功 chunk 重试。
3. 如果状态不可靠，重新从头 prefill。
4. 返回错误。

第二，chunk 已 prefill，但 transfer 失败。

处理方式：

1. 保留 source chunk KV。
2. 重试该 chunk transfer。
3. 或切换 decode worker 后重传所有已完成 chunk。

第三，部分 chunk 已到达 decode worker，但后续 chunk 失败。

处理方式：

1. 释放 decode side partial KV。
2. 清理 block table。
3. 清理 transfer state。
4. 必要时重新 prefill。

这里最重要的是不要留下 partial KV 泄漏。

## 39.28 Chunked Prefill 和 Prefix Cache

Chunked Prefill 可以和 prefix cache 结合。

例如：

```text
prompt = [cached prefix][new suffix]
```

如果 cached prefix 已经有 KV，就只需要 prefill new suffix。

如果 suffix 很长，还可以 chunked prefill：

```text
reuse prefix KV
prefill suffix chunk 0
prefill suffix chunk 1
...
```

这会带来新的边界问题：

1. prefix 长度可能不是 chunk size 整数倍。
2. prefix block 可能是共享 block。
3. suffix block 是私有 block。
4. position offset 要从 prefix length 开始。
5. eviction 不能释放仍被 suffix 请求引用的 prefix。

所以 prefix cache 命中后，不是简单跳过前几个 token。

还要正确接上 KV block table 和 position。

## 39.29 Chunked Prefill 和 Speculative Decoding

Chunked Prefill 和 speculative decoding 是两类不同优化。

Chunked Prefill 优化输入阶段：

```text
long prompt -> chunks
```

Speculative decoding 优化输出阶段：

```text
draft multiple tokens -> verify
```

二者可以同时存在。

例如：

```text
chunked prefill prepares prompt KV
then speculative decoding accelerates generation
```

不要把它们混淆。

一个是 prefill scheduling，一个是 decode acceleration。

## 39.30 Chunked Prefill 和 Pipeline Parallel

Chunked Prefill 也不是 Pipeline Parallel。

Pipeline Parallel 切的是模型层。

```text
stage 0: layers 0-9
stage 1: layers 10-19
stage 2: layers 20-31
```

Chunked Prefill 切的是输入 token。

```text
chunk 0: tokens 0-4095
chunk 1: tokens 4096-8191
```

二者关注点不同。

当然，它们可以组合。

一个长 prompt chunk 仍然可以通过 pipeline parallel 的多个 stage 执行。

## 39.31 实现 Chunked Prefill 的简化伪代码

一个极简调度器可能这样写：

```python
def schedule_step(waiting_prefills, running_decodes):
    if should_run_decode(running_decodes):
        return make_decode_batch(running_decodes)

    prefill_batch = []
    token_budget = MAX_PREFILL_TOKENS_PER_STEP

    for req in waiting_prefills:
        if token_budget <= 0:
            break

        remaining = req.prompt_len - req.prefilled_token_count
        chunk_len = min(req.chunk_size, remaining, token_budget)

        if chunk_len <= 0:
            continue

        prefill_batch.append((req, req.prefilled_token_count, chunk_len))
        token_budget -= chunk_len

    return PrefillChunkBatch(prefill_batch)
```

执行 chunk 后：

```python
def on_prefill_chunk_done(req, chunk_len):
    req.prefilled_token_count += chunk_len

    if req.prefilled_token_count == req.prompt_len:
        req.state = "PREFILL_DONE"
    else:
        req.state = "PREFILL_PAUSED"
```

这只是教学伪代码。

真实系统还要处理 KV block、position、attention metadata、prefix cache、优先级、abort 和 transfer。

## 39.32 常见实现坑

Chunked Prefill 常见坑包括：

1. position id 从每个 chunk 重新开始。
2. chunk 之间 attention mask 不正确。
3. 前面 chunk 的 KV 没有加入后续 chunk 的 context。
4. block table 更新不完整。
5. partial block 处理错误。
6. chunk transfer 完成顺序和 chunk index 不一致。
7. abort 后 partial KV 没有释放。
8. chunk 太小导致吞吐下降。
9. chunk 太大导致 decode 仍然抖动。
10. prefix cache 命中后 position offset 错误。

这些坑本质上都来自一个事实：

> 对模型语义来说，chunked prefill 必须等价于 full prefill。

## 39.33 什么时候应该使用 Chunked Prefill

适合使用 Chunked Prefill 的场景：

1. prompt 长度分布长尾明显。
2. 系统中短请求和长请求混合。
3. decode streaming 平滑度要求高。
4. unified engine 中 prefill interruption 明显。
5. PD prefill pool 中长 prompt 造成队列抖动。
6. 希望 prefill 和 KV transfer 流水化。

不一定需要的场景：

1. prompt 普遍很短。
2. 离线批处理，不关心 streaming latency。
3. 只有少量长请求，且资源充足。
4. 实现复杂度无法接受。
5. chunk overhead 大于收益。

## 39.34 如何调参

调 Chunked Prefill，重点看这些参数：

1. `chunk_size`。
2. `max_prefill_tokens_per_batch`。
3. `decode_priority`。
4. `max_prefill_chunks_per_window`。
5. `prefill_queue_timeout`。
6. `transfer_pipeline_depth`。
7. `max_pending_transfer_chunks`。
8. `long_request_priority_boost`。

观测这些指标：

1. TTFT p50/p95/p99。
2. TPOT p50/p95/p99。
3. prefill chunk latency。
4. decode step latency。
5. prefill queue length。
6. number of paused prefills。
7. chunk transfer latency。
8. GPU utilization。
9. KV cache utilization。
10. request starvation count。

调参目标不是单点最优，而是在吞吐、TTFT、TPOT 和公平性之间找到平衡。

## 39.35 面试官会怎么问

问题一：什么是 Chunked Prefill？

回答要点：把长 prompt 的 prefill 按 token chunk 切成多次执行，每个 chunk 仍经过完整模型层。它把一个大 prefill 任务拆成多个可调度单元，缓解长 prefill 阻塞 decode 和短请求的问题。

问题二：Chunked Prefill 和 PD 分离是什么关系？

回答要点：Chunked Prefill 是 prefill 执行粒度优化，PD 分离是系统架构优化。Chunked Prefill 可以在 unified engine 中使用，也可以在 PD 分离的 prefill pool 中使用。

问题三：Chunked Prefill 会不会降低长 prompt 自己的 TTFT？

回答要点：不一定。普通 causal LM 生成首 token 前仍需要完整 prompt KV，所以长请求自己的理论计算量不变。但它能改善系统级 TTFT 和 TPOT 尾延迟，因为长 prefill 不会一次性堵住整个系统。

问题四：Chunk size 太大或太小有什么问题？

回答要点：太大则调度粒度粗，仍会阻塞 decode；太小则 kernel launch、scheduler、KV metadata 和 transfer overhead 增加，可能降低吞吐。

问题五：Chunked Prefill 如何和 KV transfer 结合？

回答要点：可以全部 prefill 完再 transfer，也可以边 prefill 边 transfer。后者能让 prefill 和 transfer 流水化，但需要 partial KV metadata、chunk completion、目标 block 增量预留、顺序保证和失败清理。

问题六：Disaggregated Prefill 和 Chunked Prefill 有什么区别？

回答要点：Disaggregated Prefill 关注 prefill 在独立资源池执行，回答 where to run prefill；Chunked Prefill 关注长 prompt 如何分块执行，回答 how to run a long prefill。

## 39.36 标准回答模板

如果面试官问“Chunked Prefill 和 Disaggregated Prefill 有什么区别”，可以这样回答：

```text
Chunked Prefill 和 Disaggregated Prefill 是两个层面的概念。

Chunked Prefill 是执行粒度优化。它把一个长 prompt 的 prefill 按 token chunk 拆成多次执行，每个 chunk 仍然经过完整 Transformer 层，并逐步写入 KV cache。这样 scheduler 可以在 chunk 边界插入 decode step 或短请求，缓解长 prefill 对 TPOT 和短请求 TTFT 的影响。

Disaggregated Prefill 是系统架构优化。它把 prefill 从统一 engine 中拆出来，用独立的 prefill worker 或 prefill pool 处理输入阶段，然后把生成的 KV cache 迁移或共享给 decode worker。它强调 P/D 资源隔离、独立扩缩容和降低互相干扰。

二者可以结合：在 disaggregated prefill pool 中使用 chunked prefill。这样长 prompt 会被切成多个 prefill chunk，KV 可以按 chunk 生成，并可能和 transfer 流水化。但代价是 request state、KV block table、position、partial transfer、失败清理都会更复杂。
```

## 39.37 小练习

1. 画出 full prefill 和 chunked prefill 的时间线对比。
2. 解释为什么 chunked prefill 必须保持 position id 连续。
3. 设计一个调度策略，在 decode step 和 prefill chunk 之间做选择。
4. 分析 chunk size 太大和太小的影响。
5. 说明 chunked prefill 如何缓解 prefill interruption。
6. 画出 chunked prefill 与 KV transfer 流水化的时序图。
7. 列出支持 chunked prefill 需要新增的 10 个 metadata 字段。
8. 解释为什么普通 causal LM 通常不能在 prompt 未完整 prefill 前开始 decode。
9. 对比 Chunked Prefill、Disaggregated Prefill、Pipeline Parallel。
10. 设计一组指标评估 chunked prefill 是否有效。

## 39.38 本章总结

Chunked Prefill 是把长 prompt prefill 切成多个 token chunk 执行，让 scheduler 可以在 chunk 边界做调度决策。

它的核心价值不是减少总计算量，而是把一个长时间阻塞的大任务拆成多个可调度的小任务，从而改善系统级 TTFT、TPOT 和公平性。

Disaggregated Prefill 是把 prefill 阶段放到独立资源池中执行，是系统架构层面的优化。Chunked Prefill 是 prefill 内部执行粒度的优化。

二者可以结合：在独立 prefill pool 中对长 prompt 做 chunked prefill，并把 KV 按 chunk 生成、传输或缓存。

实现 Chunked Prefill 时，必须保证它在模型语义上等价于 full prefill：position、attention mask、KV block table、prefix cache 和 partial block 都要正确处理。

下一章会继续讨论多级 KV Cache：GPU、CPU 和远端缓存，看看 KV 如何从单个 worker 的显存对象进一步变成跨层级、跨节点的系统级缓存资源。
